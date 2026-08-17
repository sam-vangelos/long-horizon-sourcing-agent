"""LinkedInBrowser x SessionGovernor integration — P8.1.

Prior to P8.1, governance was opt-in and fail-open: enforcement lived at a
single acquisition-layer call site (linkedin/acquisition.py), and any other
caller that received an already-constructed LinkedInBrowser (the identity
resolver, the deprecated reconciliation service, ad-hoc tools) opened
profiles completely ungoverned — no cap check, no count against
daily_stats.json.

The fix moves governance onto the browser itself: LinkedInBrowser requires a
governor at construction (production bypass is impossible-by-default — the
constructor raises without one), and open_profile()/open_profile_by_url()
check and count against it directly. These tests exercise the REAL browser
methods (not mocks of them) against a real SessionGovernor + a patched
cooldown file, so they prove the fix at the layer that actually matters: any
caller holding a governed browser is now automatically governed, regardless
of which module opens the profile.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import shared.cooldown as cooldown
from linkedin.browser import LinkedInBrowser
from shared.governor import (
    MAX_PROFILE_OPENS_PER_24H,
    GovernorLimitReached,
    SessionGovernor,
    UNGOVERNED_FOR_TESTS,
)


def _configure_governor_paths(monkeypatch, tmp_path):
    governor_dir = tmp_path / "governor"
    monkeypatch.setattr(cooldown, "GOVERNOR_DIR", governor_dir)
    monkeypatch.setattr(cooldown, "DAILY_STATS_FILE", governor_dir / "daily_stats.json")
    monkeypatch.setattr(cooldown, "SESSIONS_LOG", governor_dir / "sessions.jsonl")
    return governor_dir


def _make_open_profile_browser(
    governor,
    profile_urls: tuple[str, ...] = (
        "/talent/profile/ada",
        "/talent/profile/bob",
        "/talent/profile/carol",
    ),
) -> LinkedInBrowser:
    """A LinkedInBrowser whose open_profile_by_url() succeeds against a
    mocked Playwright page, so the real method body (and therefore the
    governor hooks inside it) actually runs end to end."""
    browser = LinkedInBrowser(governor=governor)
    page = MagicMock()
    slidein_panel = MagicMock()
    slidein_panel.wait_for = AsyncMock()
    profile_links = []
    for profile_url in profile_urls:
        link = MagicMock()
        link.get_attribute = AsyncMock(return_value=profile_url)
        profile_links.append(link)
    links = MagicMock()
    links.count = AsyncMock(return_value=len(profile_links))
    links.nth = MagicMock(side_effect=profile_links)
    page.locator = MagicMock(
        side_effect=lambda selector: (
            links
            if selector == 'ol.profile-list a[href*="/talent/profile/"]'
            else slidein_panel
        )
    )
    page.wait_for_timeout = AsyncMock()
    browser._page = page
    browser._ghost_click_locator = AsyncMock(return_value=True)
    browser._governor_test_profile_links = profile_links
    return browser


# ---------------------------------------------------------------------------
# Construction: production bypass is impossible-by-default
# ---------------------------------------------------------------------------


def test_constructor_raises_without_explicit_governor():
    raised = None
    try:
        LinkedInBrowser()
    except ValueError as exc:
        raised = exc
    assert raised is not None
    assert "governor" in str(raised).lower()


def test_constructor_accepts_ungoverned_for_tests_sentinel():
    # Must not raise.
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    assert browser is not None


# ---------------------------------------------------------------------------
# UNGOVERNED_FOR_TESTS is a true no-op — it must not touch real state
# ---------------------------------------------------------------------------


def test_ungoverned_sentinel_does_not_touch_daily_stats(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)
    browser = _make_open_profile_browser(UNGOVERNED_FOR_TESTS)

    asyncio.run(browser.open_profile_by_url("/talent/profile/ada"))

    assert cooldown.get_profile_opens_24h() == 0


# ---------------------------------------------------------------------------
# A real governor counts opens from inside open_profile_by_url itself — this
# is the invariant that closes the identity-resolver / reconciliation gap:
# neither module constructs its own browser or governor, so as soon as the
# browser they're handed is governed, their opens are accounted for with no
# change needed in either module.
# ---------------------------------------------------------------------------


def test_real_governor_records_profile_open_against_daily_stats(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)
    governor = SessionGovernor()
    browser = _make_open_profile_browser(governor)

    assert cooldown.get_profile_opens_24h() == 0

    asyncio.run(browser.open_profile_by_url("/talent/profile/ada"))

    assert cooldown.get_profile_opens_24h() == 1
    assert governor.profile_opens_session == 1


def test_real_governor_records_multiple_opens_across_calls(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)
    governor = SessionGovernor()
    browser = _make_open_profile_browser(governor)

    asyncio.run(browser.open_profile_by_url("/talent/profile/ada"))
    asyncio.run(browser.open_profile_by_url("/talent/profile/bob"))
    asyncio.run(browser.open_profile_by_url("/talent/profile/carol"))

    assert cooldown.get_profile_opens_24h() == 3
    assert governor.profile_opens_session == 3


def test_open_profile_by_url_rejects_substring_identity_collisions():
    browser = _make_open_profile_browser(
        UNGOVERNED_FOR_TESTS,
        profile_urls=(
            "/talent/profile/ada-extra",
            "/talent/profile/ada?tracking=one",
        ),
    )

    asyncio.run(browser.open_profile_by_url("/talent/profile/ada"))

    browser._ghost_click_locator.assert_awaited_once_with(
        browser._governor_test_profile_links[1]
    )


def test_recruiter_identity_parser_rejects_non_profile_urls():
    assert LinkedInBrowser._profile_url_fragment(
        "https://www.linkedin.com/talent/search"
    ) == ""
    assert LinkedInBrowser._profile_url_fragment(
        "https://www.linkedin.com/in/ada"
    ) == ""


def test_exact_slot_lookup_propagates_browser_fatal_error():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser.get_card_slot_count = AsyncMock(return_value=1)
    browser._wait_for_result_slot = AsyncMock(
        side_effect=RuntimeError("browser has been closed")
    )

    raised = None
    try:
        asyncio.run(
            browser.find_result_slot_by_profile_url("/talent/profile/ada")
        )
    except RuntimeError as exc:
        raised = exc

    assert raised is not None
    assert "browser has been closed" in str(raised)


# ---------------------------------------------------------------------------
# The 24h cap enforces from inside the browser method — no DOM interaction
# is attempted once tripped.
# ---------------------------------------------------------------------------


def test_real_governor_blocks_open_when_24h_cap_already_reached(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)
    for _ in range(MAX_PROFILE_OPENS_PER_24H):
        cooldown.record_profile_open()

    governor = SessionGovernor()
    governor.start_session()
    browser = _make_open_profile_browser(governor)

    raised = None
    try:
        asyncio.run(browser.open_profile_by_url("/talent/profile/ada"))
    except GovernorLimitReached as exc:
        raised = exc

    assert raised is not None
    browser._ghost_click_locator.assert_not_called()
    # The cap was already at the ceiling; a blocked attempt must not add
    # another entry.
    assert cooldown.get_profile_opens_24h() == MAX_PROFILE_OPENS_PER_24H
