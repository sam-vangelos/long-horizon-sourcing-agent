"""Adversarial verification of the R7-hardened page-ownership contract.

Under review: disconnect() must close the PAGE OBJECT we created (self._owned_page),
never self._page-by-flag. The hazard the fix closes: self._page is reassigned on
every sidebar op and on crash recovery (require_recruiter_tab / _bind_fallback_page /
recover_from_target_crash -> _bind_existing_recruiter_page) WITHOUT clearing
ownership. Under the old boolean _owns_page scheme, a created-page connect followed
by a rebind to a DIFFERENT adopted tab left the flag True while self._page pointed at
the adopted (human) tab — so disconnect() (a) closed Sam's tab and (b) leaked the
created page.

These tests are deliberately structured to survive on BOTH the pre-fix (flag) source
and the post-fix (object) source so the RED is BEHAVIORAL (an assertion on which
page.close() ran), not a bare AttributeError on a renamed attribute. The
implementer's own suite mostly trips on `_owned_page` not existing pre-fix, which
proves the source changed but NOT that the wrong-close would be caught. The
load-bearing pre-fix proof lives here.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from linkedin.browser import LinkedInBrowser
from shared.governor import UNGOVERNED_FOR_TESTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RECRUITER_URL = "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"


def _live_created_page() -> AsyncMock:
    """A page mock that passes the REAL _bind_created_recruiter_page liveness gate."""
    page = AsyncMock()
    type(page).url = PropertyMock(return_value=RECRUITER_URL)
    page.goto = AsyncMock()
    page.evaluate = AsyncMock(return_value=True)
    page.wait_for_selector = AsyncMock()
    return page


def _adopted_page(url: str = RECRUITER_URL) -> AsyncMock:
    page = AsyncMock()
    type(page).url = PropertyMock(return_value=url)
    return page


def _browser_connected_via_real_created_bind() -> tuple[LinkedInBrowser, AsyncMock]:
    """Drive the REAL connect() down the created-page branch (existing bind forced to
    miss). Returns (browser, created_page). After this, on the fixed source
    browser._owned_page is the created page object; on pre-fix source
    browser._owns_page is True and browser._page is the created page."""
    page = _live_created_page()
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    browser._browser = MagicMock()
    browser._browser.contexts = [context]
    browser._input_backend = MagicMock()
    browser._input_backend.initialize = AsyncMock()
    browser._input_backend.shutdown = AsyncMock()
    browser._discover_recruiter_url_via_cdp = AsyncMock(return_value=RECRUITER_URL)
    browser._bind_existing_recruiter_page = AsyncMock(return_value=False)
    browser._page_is_live = AsyncMock(return_value=False)
    asyncio.run(browser.connect())
    assert browser._page is page  # connect took the created-page path
    return browser, page


# ---------------------------------------------------------------------------
# 1. INDEPENDENT pre-fix RED, expressed BEHAVIORALLY so it runs on flag source.
#
# We emulate, by hand, the exact end-state a created-page connect + a rebind leaves
# under EITHER scheme: the created page object is recorded as owned (object scheme:
# _owned_page=created; flag scheme: _owns_page=True), and self._page has since moved
# to a DIFFERENT adopted tab. We then call the real disconnect() and assert ONLY on
# which page got closed. On the FIXED source this passes (created closed, adopted
# untouched). On the PRE-FIX flag source the same setup makes disconnect() close
# self._page (the adopted tab) and leave the created page open — both assertions
# below fail BEHAVIORALLY (not via AttributeError), which is the proof the
# implementer's AttributeError-tripping tests do not provide.
# ---------------------------------------------------------------------------


def test_rebind_then_disconnect_closes_created_not_adopted_behavioral():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._input_backend = MagicMock()
    browser._input_backend.shutdown = AsyncMock()

    created = _adopted_page()  # the page WE created
    adopted = _adopted_page()  # Sam's own tab a rebind moved self._page to

    # Record ownership under whichever scheme the source-under-test uses. setattr is
    # used for the flag so this test still imports/runs on the object source (the
    # stale _owns_page attribute is simply ignored there).
    browser._owned_page = created  # object scheme (fixed)
    setattr(browser, "_owns_page", True)  # flag scheme (pre-fix)
    browser._page = adopted  # a rebind has moved self._page off the created page

    asyncio.run(browser.disconnect())

    # The created object must be the one closed, regardless of self._page.
    created.close.assert_awaited_once()
    # Sam's adopted tab must NEVER be closed.
    adopted.close.assert_not_awaited()


def test_pure_adopt_disconnect_closes_nothing_behavioral():
    """No page was created (existing/adopt bind won): nothing is owned under either
    scheme, so disconnect() must close NOTHING. Pre-fix flag source also passes this
    (it only closed when _owns_page was True) — included as a contract anchor."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._input_backend = MagicMock()
    browser._input_backend.shutdown = AsyncMock()

    adopted = _adopted_page()
    browser._owned_page = None
    setattr(browser, "_owns_page", False)
    browser._page = adopted

    asyncio.run(browser.disconnect())
    adopted.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# 2. recover_from_target_crash rebind path (item 4) — UNCOVERED by the
# implementer's suite. The crash path calls refresh_active_tab() then
# _bind_existing_recruiter_page(), which rebinds self._page to the recovered
# (adopted) Recruiter tab. After a created-page connect, that rebind must not cause
# disconnect() to close the recovered tab or leak the created page.
# ---------------------------------------------------------------------------


def test_recover_from_target_crash_rebind_preserves_created_ownership():
    browser, created = _browser_connected_via_real_created_bind()
    assert browser._owned_page is created

    recovered = _adopted_page()

    async def _rebind_recovered(*, preferred_context=None, **_kwargs):
        browser._page = recovered
        return True

    # The real recover_from_target_crash: refresh succeeds, then existing-bind rebinds.
    browser.refresh_active_tab = AsyncMock(return_value=True)
    browser._bind_existing_recruiter_page = _rebind_recovered

    ok = asyncio.run(browser.recover_from_target_crash())
    assert ok is True
    assert browser._page is recovered  # crash recovery moved self._page

    # Ownership still follows the created object.
    assert browser._owned_page is created

    asyncio.run(browser.disconnect())
    created.close.assert_awaited_once()  # created page closed: no leak
    recovered.close.assert_not_awaited()  # recovered (adopted) tab untouched


# ---------------------------------------------------------------------------
# 3. Multiple rebinds (item 4) — self._page churns across several adopted tabs;
# ownership must never migrate, and only the created page is closed at the end.
# ---------------------------------------------------------------------------


def test_multiple_rebinds_only_created_closed():
    browser, created = _browser_connected_via_real_created_bind()

    adopted_a = _adopted_page()
    adopted_b = _adopted_page()
    adopted_c = _adopted_page()

    for target in (adopted_a, adopted_b, adopted_c):
        async def _rebind(*, preferred_context=None, _t=target, **_kwargs):
            browser._page = _t
            return True

        browser._bind_existing_recruiter_page = _rebind
        asyncio.run(browser.require_recruiter_tab())
        assert browser._page is target

    asyncio.run(browser.disconnect())

    created.close.assert_awaited_once()
    for adopted in (adopted_a, adopted_b, adopted_c):
        adopted.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. Rebind BACK to the created page (item 4) — if a later bind happens to select
# the very tab we created (self._page == self._owned_page again), disconnect() must
# close it exactly ONCE and must not raise a double-close.
# ---------------------------------------------------------------------------


def test_rebind_back_to_created_no_double_close():
    browser, created = _browser_connected_via_real_created_bind()

    # First rebind away to an adopted tab...
    adopted = _adopted_page()

    async def _rebind_away(*, preferred_context=None, **_kwargs):
        browser._page = adopted
        return True

    browser._bind_existing_recruiter_page = _rebind_away
    asyncio.run(browser.require_recruiter_tab())
    assert browser._page is adopted

    # ...then a later bind selects the created tab again (self._page == _owned_page).
    async def _rebind_back(*, preferred_context=None, **_kwargs):
        browser._page = created
        return True

    browser._bind_existing_recruiter_page = _rebind_back
    asyncio.run(browser.require_recruiter_tab())
    assert browser._page is created
    assert browser._owned_page is created

    asyncio.run(browser.disconnect())
    # Closed exactly once — no double-close even though _page == _owned_page.
    created.close.assert_awaited_once()
    adopted.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. connect-created-then-immediate-disconnect, NO rebind (item 4).
# ---------------------------------------------------------------------------


def test_created_then_immediate_disconnect_closes_created():
    browser, created = _browser_connected_via_real_created_bind()
    assert browser._owned_page is created
    asyncio.run(browser.disconnect())
    created.close.assert_awaited_once()
    assert browser._owned_page is None
    assert browser._page is None


# ---------------------------------------------------------------------------
# 6. IDEMPOTENCE (item 5).
# ---------------------------------------------------------------------------


def test_double_disconnect_closes_created_once_total():
    """Second disconnect() must be a no-op for the created page (already cleared) and
    must not raise. Closed exactly once across BOTH calls."""
    browser, created = _browser_connected_via_real_created_bind()

    asyncio.run(browser.disconnect())
    assert browser._owned_page is None
    # Second disconnect on a torn-down browser — needs a live input backend again
    # (disconnect() shuts it down). Re-stub and ensure no raise / no extra close.
    browser._input_backend = MagicMock()
    browser._input_backend.shutdown = AsyncMock()
    asyncio.run(browser.disconnect())

    created.close.assert_awaited_once()  # exactly once across two disconnects
    assert browser._owned_page is None
    assert browser._page is None


def test_disconnect_completes_when_created_close_raises_after_rebind():
    """Idempotence x no-leak under failure: created page's close() raises (already
    gone), AND self._page has been rebound to an adopted tab. disconnect() must still
    complete, attempt to close the CREATED page (not the adopted one), swallow the
    error, and clear all refs."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._input_backend = MagicMock()
    browser._input_backend.shutdown = AsyncMock()

    created = _adopted_page()
    created.close = AsyncMock(
        side_effect=RuntimeError("Target page, context or browser has been closed")
    )
    adopted = _adopted_page()

    browser._owned_page = created
    browser._page = adopted  # rebound away

    asyncio.run(browser.disconnect())  # must not raise

    created.close.assert_awaited_once()  # attempted to close the CREATED page
    adopted.close.assert_not_awaited()  # never the adopted tab
    assert browser._owned_page is None
    assert browser._page is None
    assert browser._browser is None


# ---------------------------------------------------------------------------
# 7. Failed created-bind must NOT leave an owned page (would be a double-close on a
# subsequent disconnect, since the bind already closed it). Drives the REAL
# _bind_created_recruiter_page down its navigation-failure branch.
# ---------------------------------------------------------------------------


def test_failed_created_bind_leaves_no_owned_page_and_no_double_close():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = AsyncMock()
    type(page).url = PropertyMock(return_value=RECRUITER_URL)
    page.goto = AsyncMock(side_effect=RuntimeError("navigation failed"))
    page.close = AsyncMock()  # the bind closes the page it created on failure
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    browser._browser = MagicMock()
    browser._browser.contexts = [context]
    browser._input_backend = MagicMock()
    browser._input_backend.initialize = AsyncMock()
    browser._input_backend.shutdown = AsyncMock()
    browser._discover_recruiter_url_via_cdp = AsyncMock(return_value=RECRUITER_URL)

    bound = asyncio.run(browser._bind_created_recruiter_page(preferred_context=None))
    assert bound is False
    page.close.assert_awaited_once()  # the bind closed its own failed page
    assert browser._owned_page is None  # ...and did NOT record it as owned

    # disconnect() must not try to close it again (no double-close on the failed page).
    browser._page = None
    asyncio.run(browser.disconnect())
    page.close.assert_awaited_once()  # still exactly one close — disconnect added none
