"""Tests for LinkedIn Recruiter browser crash recovery helpers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from linkedin.browser import LinkedInBrowser
from shared.governor import UNGOVERNED_FOR_TESTS
from linkedin.recruiter_recovery import detect_recruiter_health


def test_check_and_recover_routes_target_crash_into_refresh_recovery():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser.recover_from_target_crash = AsyncMock(return_value=True)

    page = MagicMock()
    type(page).url = PropertyMock(side_effect=RuntimeError("Target crashed"))
    browser._page = page

    recovered = asyncio.run(browser.check_and_recover())

    assert recovered is True
    browser.recover_from_target_crash.assert_awaited_once()


def test_refresh_active_tab_falls_back_to_os_shortcut_after_target_crash():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = MagicMock()
    browser._page.reload = AsyncMock(side_effect=RuntimeError("Target crashed"))
    browser._page.wait_for_timeout = AsyncMock()
    browser._press_key = AsyncMock(side_effect=RuntimeError("Target crashed"))
    browser._send_os_refresh_shortcut = AsyncMock(return_value=True)

    recovered = asyncio.run(browser.refresh_active_tab())

    assert recovered is True
    browser._send_os_refresh_shortcut.assert_awaited_once()


def test_recover_from_target_crash_rebinds_and_restores_search_url():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser.refresh_active_tab = AsyncMock(return_value=True)
    browser._bind_existing_recruiter_page = AsyncMock(side_effect=[False, True])
    browser.navigate_to_search = AsyncMock()
    browser._page = MagicMock()
    type(browser._page).url = PropertyMock(return_value="chrome-error://chromewebdata/")

    with patch("linkedin.browser.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        recovered = asyncio.run(
            browser.recover_from_target_crash(
                "https://www.linkedin.com/talent/hire/3000000006/discover/recruiterSearch"
            )
        )

    assert recovered is True
    browser.navigate_to_search.assert_awaited_once()
    assert sleep_mock.await_count >= 1


# ---------------------------------------------------------------------------
# P6: detect_recruiter_health classification tests
# ---------------------------------------------------------------------------


def test_detect_health_chrome_error_maps_to_aw_snap():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value="chrome-error://chromewebdata/")
    browser._page = page
    assert asyncio.run(detect_recruiter_health(browser)) == "aw_snap"


def test_detect_health_target_crashed_exception():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(side_effect=RuntimeError("Target crashed"))
    browser._page = page
    assert asyncio.run(detect_recruiter_health(browser)) == "target_crashed"


def test_detect_health_cdp_attach_failure():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(side_effect=RuntimeError("Connection refused"))
    browser._page = page
    assert asyncio.run(detect_recruiter_health(browser)) == "cdp_attach_failed"


def test_detect_health_wrong_surface_non_linkedin():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.google.com/")
    browser._page = page
    assert asyncio.run(detect_recruiter_health(browser)) == "wrong_surface"


def test_detect_health_logged_out():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.linkedin.com/login")
    browser._page = page
    assert asyncio.run(detect_recruiter_health(browser)) == "logged_out"


# ---------------------------------------------------------------------------
# Wave 1 Slice 1.1: /checkpoint/ is a block, not a logout
# ---------------------------------------------------------------------------


def _make_browser(url: str) -> LinkedInBrowser:
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value=url)
    page.title = AsyncMock(return_value="")
    body_locator = MagicMock()
    body_locator.inner_text = AsyncMock(return_value="")
    page.locator = MagicMock(return_value=body_locator)
    browser._page = page
    return browser


def test_checkpoint_url_classifies_as_blocked_not_logged_out():
    browser = _make_browser("https://www.linkedin.com/checkpoint/challenge/abc123")
    assert asyncio.run(detect_recruiter_health(browser)) == "blocked_or_rate_limited"


def test_challenge_url_classifies_as_blocked():
    browser = _make_browser("https://www.linkedin.com/challenge/verify")
    assert asyncio.run(detect_recruiter_health(browser)) == "blocked_or_rate_limited"


def test_plain_login_url_still_classifies_as_logged_out():
    browser = _make_browser("https://www.linkedin.com/login")
    assert asyncio.run(detect_recruiter_health(browser)) == "logged_out"


def test_detect_health_lost_project_context():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._project_id = "12345"
    page = MagicMock()
    type(page).url = PropertyMock(
        return_value="https://www.linkedin.com/talent/hire/99999/discover/recruiterSearch"
    )
    browser._page = page
    assert asyncio.run(detect_recruiter_health(browser)) == "lost_project_context"


def test_detect_health_healthy_recruiter():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._project_id = "12345"
    page = MagicMock()
    type(page).url = PropertyMock(
        return_value="https://www.linkedin.com/talent/hire/12345/discover/recruiterSearch"
    )
    browser._page = page
    assert asyncio.run(detect_recruiter_health(browser)) == "healthy"


def test_detect_health_healthy_without_project_id():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._project_id = None
    page = MagicMock()
    type(page).url = PropertyMock(
        return_value="https://www.linkedin.com/talent/search"
    )
    browser._page = page
    assert asyncio.run(detect_recruiter_health(browser)) == "healthy"


# ---------------------------------------------------------------------------
# P8.2: rate-limit / block classification must come from page CONTENT
# (title/body text), checked BEFORE the wrong-surface check — the prior
# implementation matched only the URL, which almost never carries a
# rate-limit signal, so wrong_surface short-circuited first and the block
# was never detected.
# ---------------------------------------------------------------------------


def _browser_with_content(url: str, *, title: str = "", body_text: str = "") -> LinkedInBrowser:
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value=url)
    page.title = AsyncMock(return_value=title)
    body_locator = MagicMock()
    body_locator.inner_text = AsyncMock(return_value=body_text)
    page.locator = MagicMock(return_value=body_locator)
    browser._page = page
    return browser


def test_detect_health_rate_limited_from_content_before_wrong_surface():
    # This URL alone (no /talent, not a known login/crash pattern) would
    # classify as wrong_surface under URL-only matching.
    browser = _browser_with_content(
        "https://www.linkedin.com/some-interstitial",
        title="LinkedIn",
        body_text="You have exceeded the rate limit. Please wait and try again.",
    )
    assert asyncio.run(detect_recruiter_health(browser)) == "blocked_or_rate_limited"


def test_detect_health_rate_limited_from_content_on_talent_url():
    # Same content signal, but this time the URL still looks like a normal
    # Recruiter search — content classification must win regardless.
    browser = _browser_with_content(
        "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        title="LinkedIn Recruiter",
        body_text="429 Too Many Requests",
    )
    assert asyncio.run(detect_recruiter_health(browser)) == "blocked_or_rate_limited"


def test_detect_health_benign_content_does_not_false_positive_rate_limit():
    # Positive control: ordinary page content must not trip the rate-limit
    # classifier and must still fall through to the existing URL logic.
    browser = _browser_with_content(
        "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        title="LinkedIn Recruiter",
        body_text="Search results for Senior ML Engineer",
    )
    assert asyncio.run(detect_recruiter_health(browser)) == "healthy"


def test_get_current_project_id():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._project_id = "42"
    assert browser.get_current_project_id() == "42"


def test_get_current_search_url_returns_none_when_not_recruiter():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.google.com")
    browser._page = page
    assert browser.get_current_search_url() is None


# ---------------------------------------------------------------------------
# Wave 0 Slice 0.2: page-capture hooks on health paths
# ---------------------------------------------------------------------------


def test_something_went_wrong_captures_before_first_retry(tmp_path):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._state_dir = tmp_path

    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.linkedin.com/talent/search")
    page.wait_for_timeout = AsyncMock()

    error_heading = MagicMock()
    error_heading.is_visible = AsyncMock(side_effect=[True, False])
    try_again_btn = MagicMock()
    try_again_btn.is_visible = AsyncMock(return_value=True)

    call_order = []

    async def click_side_effect():
        call_order.append("click")

    try_again_btn.click = AsyncMock(side_effect=click_side_effect)

    def locator_side_effect(selector):
        mock_loc = MagicMock()
        if 'text="Something went wrong"' in selector:
            mock_loc.first = error_heading
        elif 'button:has-text("Try again")' in selector:
            mock_loc.first = try_again_btn
        else:
            mock_loc.first = MagicMock()
        return mock_loc

    page.locator = MagicMock(side_effect=locator_side_effect)
    browser._page = page

    async def capture_side_effect(*args, **kwargs):
        call_order.append("capture")

    capture_mock = AsyncMock(side_effect=capture_side_effect)
    with patch("linkedin.page_capture.capture_page_state", capture_mock):
        recovered = asyncio.run(browser.check_and_recover())

    capture_mock.assert_awaited_once_with(
        browser, tmp_path, reason="health_something_went_wrong"
    )
    try_again_btn.click.assert_awaited_once()
    assert call_order == ["capture", "click"]
    assert recovered is True


def test_check_and_recover_survives_a_raising_capture(tmp_path):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._state_dir = tmp_path

    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.linkedin.com/talent/search")
    page.wait_for_timeout = AsyncMock()

    error_heading = MagicMock()
    error_heading.is_visible = AsyncMock(side_effect=[True, False])
    try_again_btn = MagicMock()
    try_again_btn.is_visible = AsyncMock(return_value=True)
    try_again_btn.click = AsyncMock()

    def locator_side_effect(selector):
        mock_loc = MagicMock()
        if 'text="Something went wrong"' in selector:
            mock_loc.first = error_heading
        elif 'button:has-text("Try again")' in selector:
            mock_loc.first = try_again_btn
        else:
            mock_loc.first = MagicMock()
        return mock_loc

    page.locator = MagicMock(side_effect=locator_side_effect)
    browser._page = page

    capture_mock = AsyncMock(side_effect=RuntimeError("capture boom"))
    with patch("linkedin.page_capture.capture_page_state", capture_mock):
        recovered = asyncio.run(browser.check_and_recover())

    capture_mock.assert_awaited_once_with(
        browser, tmp_path, reason="health_something_went_wrong"
    )
    try_again_btn.click.assert_awaited_once()
    assert recovered is True


def test_login_redirect_captures_before_raise(tmp_path):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._state_dir = tmp_path

    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.linkedin.com/login")
    page.locator = MagicMock(return_value=MagicMock(is_visible=AsyncMock(return_value=False)))
    browser._page = page

    capture_mock = AsyncMock()
    with patch("linkedin.page_capture.capture_page_state", capture_mock):
        with pytest.raises(RuntimeError, match="LinkedIn session expired"):
            asyncio.run(browser.check_and_recover())

    capture_mock.assert_awaited_once_with(
        browser, tmp_path, reason="health_login_redirect"
    )


def test_capture_absent_state_dir_is_a_noop():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._state_dir = None

    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.linkedin.com/login")
    page.locator = MagicMock(return_value=MagicMock(is_visible=AsyncMock(return_value=False)))
    browser._page = page

    capture_mock = AsyncMock()
    with patch("linkedin.page_capture.capture_page_state", capture_mock):
        with pytest.raises(RuntimeError, match="LinkedIn session expired"):
            asyncio.run(browser.check_and_recover())

    capture_mock.assert_not_awaited()


def test_navigate_sidebar_missing_captures_before_raise(tmp_path):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._state_dir = tmp_path

    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.linkedin.com/talent/hire/1/discover/recruiterSearch")
    page.goto = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.locator = MagicMock(
        return_value=MagicMock(is_visible=AsyncMock(return_value=False))
    )
    browser._page = page

    capture_mock = AsyncMock()
    with patch("linkedin.page_capture.capture_page_state", capture_mock):
        with pytest.raises(RuntimeError, match="navigate_to_search failed"):
            asyncio.run(browser.navigate_to_search("https://www.linkedin.com/talent/hire/1/discover/recruiterSearch"))

    capture_mock.assert_awaited_once_with(
        browser, tmp_path, reason="navigate_sidebar_missing"
    )
