"""CDP connect vs Recruiter-tab readiness (deferred validation)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from linkedin.browser import LinkedInBrowser, _is_unusable_cdp_page_url
from shared.governor import UNGOVERNED_FOR_TESTS


def test_is_unusable_cdp_page_url():
    assert _is_unusable_cdp_page_url("chrome://omnibox-popup.top-chrome/") is True
    assert _is_unusable_cdp_page_url("chrome-devtools://devtools/bundled/inspector.html") is True
    assert _is_unusable_cdp_page_url("devtools://devtools/bundled/inspector.html") is True
    assert _is_unusable_cdp_page_url("about:blank") is False
    assert _is_unusable_cdp_page_url("https://www.linkedin.com/talent/search") is False


def test_require_recruiter_tab_raises_when_no_talent_tab():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value="about:blank")
    browser._page = page
    with patch.object(browser, "_bind_existing_recruiter_page", new=AsyncMock(return_value=False)):
        with pytest.raises(RuntimeError, match="LinkedIn Recruiter is required"):
            asyncio.run(browser.require_recruiter_tab())


def test_require_recruiter_tab_updates_project_id_when_bind_succeeds():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(
        return_value="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    browser._page = page
    with patch.object(browser, "_bind_existing_recruiter_page", new=AsyncMock(return_value=True)):
        asyncio.run(browser.require_recruiter_tab())
    assert browser._project_id == "123"


# ---------------------------------------------------------------------------
# R7 — the Chrome-148 created-page must be closed on disconnect (tab-leak fix)
#
# _bind_created_recruiter_page creates a Playwright-owned page (the Chrome-148
# path: adopting the pre-existing tab yields a zombie, so we open our own). It
# records the created OBJECT in self._owned_page. Pre-fix, disconnect() cleared
# refs but never closed self._page, so every connect() that took the created-page
# path leaked a tab in Sam's real profile — he could not re-run a smoke cleanly
# after a selector miss. disconnect() now best-effort-closes ONLY the page WE
# created (the tracked object), never an adopted tab (Sam's own), and survives an
# already-closed page.
#
# These drive connect() with the existing/created/fallback binds stubbed so the
# created-page bind runs (setting _page=AsyncMock + _owned_page=that object)
# without a real CDP/navigation round-trip; disconnect() then exercises the real
# close path.
# ---------------------------------------------------------------------------


def _created_page_mock() -> AsyncMock:
    """An AsyncMock standing in for a Playwright-created page that passes the
    real _bind_created_recruiter_page liveness gate (goto + evaluate + selector)."""
    page = AsyncMock()
    type(page).url = PropertyMock(
        return_value="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    page.goto = AsyncMock()
    page.evaluate = AsyncMock(return_value=True)  # liveness probe must resolve truthy
    page.wait_for_selector = AsyncMock()
    return page


def _browser_for_real_created_bind(page: AsyncMock) -> LinkedInBrowser:
    """A browser wired so the REAL _bind_created_recruiter_page runs end-to-end:
    a CDP-discovered Recruiter URL + a context whose new_page() yields ``page``.
    Nothing here writes _owns_page — only the source line under test (browser.py:383)
    can set it, so deleting that line makes the ownership assertion fail."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    browser._browser = MagicMock()
    browser._browser.contexts = [context]
    browser._input_backend = MagicMock()
    browser._input_backend.initialize = AsyncMock()
    browser._discover_recruiter_url_via_cdp = AsyncMock(
        return_value="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    return browser


def _connected_browser_via_created_page() -> LinkedInBrowser:
    """A browser whose connect() took the created-page path. The created bind is the
    REAL method, driven via stubbed CDP discovery + a context.new_page() that yields a
    live mock page — so _owns_page is set by the real source line, not a test stub."""
    page = _created_page_mock()
    browser = _browser_for_real_created_bind(page)
    browser._input_backend.shutdown = AsyncMock()
    # connect() tries the existing-bind first; force it to miss so the created
    # path runs. _page_is_live is only consulted on the existing-bind branch.
    browser._bind_existing_recruiter_page = AsyncMock(return_value=False)
    browser._page_is_live = AsyncMock(return_value=False)
    return browser


def test_bind_created_recruiter_page_sets_owns_page():
    """TEETH: drive the REAL _bind_created_recruiter_page (not a stub) and assert it
    records the created page OBJECT in browser._owned_page on the success path.
    Deleting that source line makes this fail — the previous version asserted a
    stub's own write and did not protect the real line."""
    page = _created_page_mock()
    browser = _browser_for_real_created_bind(page)
    bound = asyncio.run(browser._bind_created_recruiter_page(preferred_context=None))
    assert bound is True
    assert browser._page is page
    assert browser._owned_page is page  # ownership tracks the exact created OBJECT
    # Confirm we drove the real path: a page was actually created and navigated.
    browser._browser.contexts[0].new_page.assert_awaited_once()
    page.goto.assert_awaited_once()


def test_connect_via_created_page_sets_owns_page():
    """Pin the ownership contract through the REAL connect() -> real created-bind:
    when the existing bind misses and the created path wins, _owns_page is True."""
    browser = _connected_browser_via_created_page()
    asyncio.run(browser.connect())
    assert browser._owned_page is browser._page  # the created object is owned
    assert browser._page is not None


def test_disconnect_closes_owned_created_page():
    """RED pre-fix: created page is owned but disconnect() never closed it
    (tab leak). GREEN post-fix: disconnect() awaits page.close() exactly once."""
    browser = _connected_browser_via_created_page()
    asyncio.run(browser.connect())
    page = browser._page
    asyncio.run(browser.disconnect())
    page.close.assert_awaited_once()
    assert browser._owned_page is None
    assert browser._page is None


def test_disconnect_does_not_close_bound_existing_page():
    """We must NEVER close Sam's own adopted tab. When the existing bind wins,
    _owned_page stays None and disconnect() leaves the page open.

    NOTE: this sets _owned_page=None directly, so it passes even PRE-fix (pre-fix
    disconnect closed no page at all). The discriminating coverage — that the REAL
    connect() leaves the existing-bind page un-owned and un-closed — lives in
    test_connect_existing_bind_then_disconnect_does_not_close_adopted_tab below."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._input_backend = MagicMock()
    browser._input_backend.shutdown = AsyncMock()
    page = AsyncMock()
    browser._page = page
    browser._owned_page = None  # adopted/fallback bind — Sam's tab, not ours
    asyncio.run(browser.disconnect())
    page.close.assert_not_awaited()


def test_connect_existing_bind_then_disconnect_does_not_close_adopted_tab():
    """TEETH on the owned-vs-bound discrimination through the REAL connect() path:
    when the EXISTING bind wins, the bound page is Sam's adopted human tab — connect()
    must leave _owned_page None (only _bind_created_recruiter_page sets it), and
    disconnect() must NOT close it. This is the discriminator the sibling above lacks:
    here _owned_page is whatever the real connect() path produced, never set by the test.
    A regression that set _owned_page in the existing-bind path (or closed a non-owned
    page on disconnect) fails this."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._browser = MagicMock()  # bypass the "browser is None" connect branch
    browser._browser.contexts = [MagicMock()]
    browser._input_backend = MagicMock()
    browser._input_backend.shutdown = AsyncMock()

    page = AsyncMock()
    type(page).url = PropertyMock(
        return_value="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )

    async def _fake_bind_existing(*, preferred_context=None):
        # The real existing-bind binds Sam's own tab and never sets _owns_page.
        browser._page = page
        return True

    browser._bind_existing_recruiter_page = _fake_bind_existing
    browser._page_is_live = AsyncMock(return_value=True)
    # If the created/fallback binds were reached, the test premise is wrong — make
    # them blow up so a routing regression is loud rather than silently masked.
    browser._bind_created_recruiter_page = AsyncMock(
        side_effect=AssertionError("created bind must not run when existing bind wins")
    )
    browser._bind_fallback_page = AsyncMock(
        side_effect=AssertionError("fallback bind must not run when existing bind wins")
    )

    asyncio.run(browser.connect())
    assert browser._owned_page is None  # existing bind never claims ownership
    assert browser._page is page

    asyncio.run(browser.disconnect())
    page.close.assert_not_awaited()  # never close Sam's adopted human tab
    assert browser._page is None


def test_disconnect_survives_already_closed_created_page():
    """disconnect() is best-effort: if the created page is already gone and
    close() raises, disconnect must still complete and clear all refs."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._input_backend = MagicMock()
    browser._input_backend.shutdown = AsyncMock()
    page = AsyncMock()
    page.close = AsyncMock(side_effect=RuntimeError("Target page, context or browser has been closed"))
    browser._page = page
    browser._owned_page = page  # WE created it; the close below raises (already gone)
    # Must not raise.
    asyncio.run(browser.disconnect())
    page.close.assert_awaited_once()
    assert browser._owned_page is None
    assert browser._page is None
    assert browser._browser is None


# ---------------------------------------------------------------------------
# R7 — the fallback bind targets Sam's OWN tab, so it must NOT claim ownership.
# Only _bind_created_recruiter_page sets _owned_page; a fallback-bound page is an
# existing tab in Sam's profile and must survive disconnect() unclosed. This
# drives the REAL _bind_fallback_page with a stubbed context/page and pins that it
# leaves _owned_page None, so a fallback-bound page is never closed on disconnect.
# ---------------------------------------------------------------------------


def test_bind_fallback_page_leaves_owns_page_false():
    """TEETH on the fallback ownership contract: the REAL _bind_fallback_page binds a
    pre-existing tab (Sam's own) and must NEVER set _owned_page. If it did, disconnect()
    would close a tab we did not create. A regression that set _owned_page in the
    fallback path fails this."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._input_backend = MagicMock()
    browser._input_backend.initialize = AsyncMock()

    page = AsyncMock()
    type(page).url = PropertyMock(return_value="https://www.linkedin.com/feed/")
    page.wait_for_load_state = AsyncMock()

    context = MagicMock()
    context.pages = [page]
    browser._browser = MagicMock()
    browser._browser.contexts = [context]

    bound = asyncio.run(browser._bind_fallback_page(preferred_context=None))
    assert bound is True
    assert browser._page is page
    assert browser._owned_page is None  # fallback binds Sam's tab — never owned

    # And disconnect() must therefore leave that adopted tab open.
    browser._input_backend.shutdown = AsyncMock()
    asyncio.run(browser.disconnect())
    page.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# R7 hardening — ownership must track the created PAGE OBJECT, not a flag on
# self._page. self._page is reassigned WITHOUT clearing ownership by every
# rebind path: _bind_existing_recruiter_page (which require_recruiter_tab and
# every apply_* method call at the START of the op), _bind_fallback_page, and
# recover_from_target_crash -> _bind_existing. With the old flag-on-self._page
# scheme, a created-page connect (_owns_page=True, _page=created) followed by an
# existing rebind to a DIFFERENT tab left the flag True while self._page now
# pointed at the rebound (adopted) tab — so disconnect() closed the WRONG tab
# (Sam's adopted human tab) AND leaked the created page. The fix tracks the
# created object in self._owned_page and closes exactly that object on
# disconnect, regardless of what self._page currently points to.
# ---------------------------------------------------------------------------


def test_disconnect_closes_created_page_not_adopted_after_rebind():
    """RED pre-fix: connect via the REAL created-page path (sets ownership on the
    created object), THEN rebind self._page to a DIFFERENT adopted page (as
    require_recruiter_tab/apply do via _bind_existing_recruiter_page), THEN
    disconnect. Pre-fix disconnect closes self._page-by-flag — i.e. the ADOPTED
    tab (Sam's own) — and never closes the created page. GREEN post-fix:
    disconnect closes the CREATED page and leaves the adopted tab open."""
    browser = _connected_browser_via_created_page()
    asyncio.run(browser.connect())
    created_page = browser._page  # the page WE created, now owned

    # Simulate the first sidebar op: require_recruiter_tab() ->
    # _bind_existing_recruiter_page() rebinds self._page to a higher-scored
    # Recruiter tab that is a DIFFERENT object (Sam's adopted human tab).
    adopted_page = AsyncMock()
    type(adopted_page).url = PropertyMock(
        return_value="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )

    async def _rebind_to_adopted(*, preferred_context=None, **_kwargs):
        browser._page = adopted_page
        return True

    browser._bind_existing_recruiter_page = _rebind_to_adopted
    asyncio.run(browser.require_recruiter_tab())
    assert browser._page is adopted_page  # self._page has moved off the created page

    asyncio.run(browser.disconnect())

    # The page WE created is closed (no leak); the adopted tab is never touched.
    created_page.close.assert_awaited_once()
    adopted_page.close.assert_not_awaited()


def test_no_leak_created_page_closed_even_after_rebind():
    """The created page must ALWAYS be closed on disconnect — even though
    self._page was reassigned away from it by a rebind. This is the no-leak
    invariant stated independently of which tab is adopted: ownership follows the
    object we created, not the current self._page."""
    browser = _connected_browser_via_created_page()
    asyncio.run(browser.connect())
    created_page = browser._page

    # Rebind self._page to something else entirely (fallback-style move).
    other_page = AsyncMock()
    browser._page = other_page

    asyncio.run(browser.disconnect())

    created_page.close.assert_awaited_once()  # created object closed: no leak
    assert browser._page is None


def test_disconnect_via_created_no_rebind_still_closes_created():
    """Regression: when NO rebind happens (self._page is still the created page),
    disconnect must still close the created page exactly once. Guards against a
    fix that only closes self._owned_page when it diverges from self._page."""
    browser = _connected_browser_via_created_page()
    asyncio.run(browser.connect())
    created_page = browser._page
    assert browser._owned_page is created_page  # ownership tracks the object

    asyncio.run(browser.disconnect())

    created_page.close.assert_awaited_once()
    assert browser._owned_page is None
    assert browser._page is None


def test_disconnect_does_not_close_purely_adopted_page():
    """When connect() ADOPTS an existing tab (existing bind wins), no page was
    created, so self._owned_page stays None and disconnect() closes nothing —
    even after a later rebind. This is the contract that protects Sam's tabs on a
    seat where the existing bind succeeds (the non-Chrome-148 path)."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._browser = MagicMock()
    browser._browser.contexts = [MagicMock()]
    browser._input_backend = MagicMock()
    browser._input_backend.shutdown = AsyncMock()

    adopted = AsyncMock()
    type(adopted).url = PropertyMock(
        return_value="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )

    async def _bind_existing(*, preferred_context=None):
        browser._page = adopted
        return True

    browser._bind_existing_recruiter_page = _bind_existing
    browser._page_is_live = AsyncMock(return_value=True)
    browser._bind_created_recruiter_page = AsyncMock(
        side_effect=AssertionError("created bind must not run when existing bind wins")
    )

    asyncio.run(browser.connect())
    assert browser._owned_page is None  # nothing was created -> nothing owned

    asyncio.run(browser.disconnect())
    adopted.close.assert_not_awaited()  # purely-adopted tab is never closed
