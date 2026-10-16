"""R8 — the company smoke's results-rail precondition gate.

The ONLY R8 source change in tools/hop4_company_smoke.py is the precondition in
_run: before applying anything it waits for the search-results refinement rail
(aside.left-rail); if that rail is absent it prints a precondition FAIL and
returns 1 WITHOUT calling apply_company_filter. The point is that on a non-results
view (e.g. /advanced, the entry FORM) apply_company_filter fails closed for a
correct reason — the editor/chip live in the rail — and a fail-closed False there
would otherwise be MISREAD as a selector miss (a masked PASS). The gate turns that
into a loud, honest precondition failure.

These tests drive the REAL _run with a fake LinkedInBrowser:
  * rail ABSENT  -> _run returns 1 AND apply_company_filter is NEVER awaited (gate).
  * rail PRESENT -> the gate passes, apply_company_filter IS awaited, _run returns 0.

TEETH: removing the gate's `return 1` (tools/hop4_company_smoke.py:62) makes the
absent-rail test fail — _run would fall through and await apply_company_filter,
violating both the return-code and the never-awaited assertions.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import tools.hop4_company_smoke as smoke


class _FakeLocator:
    """A minimal Playwright-Locator stand-in.

    `.first`, `.locator(...)` and `.filter(...)` all return self so a chained
    selector resolves to one configurable node; `wait_for` and `is_visible` are
    AsyncMocks the test sets per scenario.
    """

    def __init__(self, *, wait_for: AsyncMock | None = None, is_visible: AsyncMock | None = None):
        self.wait_for = wait_for or AsyncMock()
        self.is_visible = is_visible or AsyncMock(return_value=True)
        # The dismiss "X" is hover-gated; the smoke hovers the pill before clicking.
        self.hover = AsyncMock()

    @property
    def first(self):
        return self

    def locator(self, *_args, **_kwargs):
        return self

    def filter(self, *_args, **_kwargs):
        return self


class _FakePage:
    """browser.page: hands back the rail locator the test configured for any selector."""

    def __init__(self, *, rail: _FakeLocator, url: str):
        self._rail = rail
        self.url = url

    def locator(self, *_args, **_kwargs):
        return self._rail


def _fake_browser(*, rail: _FakeLocator, url: str) -> MagicMock:
    """A LinkedInBrowser stand-in whose connect/require_recruiter_tab/disconnect are
    no-ops and whose .page is the fake page above. apply_company_filter is an AsyncMock
    so the test can assert whether the gate let it run."""
    browser = MagicMock()
    browser.connect = AsyncMock()
    browser.require_recruiter_tab = AsyncMock()
    browser.disconnect = AsyncMock()
    browser.page = _FakePage(rail=rail, url=url)
    browser._peek_results_count_text = AsyncMock(return_value="120 results")
    browser.apply_company_filter = AsyncMock(return_value=True)
    browser._ghost_click_locator = AsyncMock()
    return browser


def test_run_gates_when_results_rail_absent_and_never_applies():
    """Rail absent (rail.first.wait_for raises a timeout) -> _run returns 1 and
    apply_company_filter is NEVER awaited. This is the R8 gate: a non-results view is
    a loud precondition FAIL, not a fail-closed False masquerading as a selector miss.

    TEETH: delete tools/hop4_company_smoke.py:62 (`return 1`) and this fails — _run
    falls through and awaits apply_company_filter."""
    rail = _FakeLocator(
        wait_for=AsyncMock(side_effect=TimeoutError("left-rail not attached"))
    )
    browser = _fake_browser(rail=rail, url="https://www.linkedin.com/talent/discover/advanced")

    with patch.object(smoke, "LinkedInBrowser", return_value=browser):
        rc = asyncio.run(smoke._run("Stripe"))

    assert rc == 1, "rail absent must return a precondition FAIL (1)"
    browser.apply_company_filter.assert_not_awaited()
    browser.disconnect.assert_awaited_once()  # finally-block cleanup still runs


def test_run_applies_when_results_rail_present():
    """Rail present (rail.first.wait_for resolves) -> the gate passes and
    apply_company_filter IS awaited. Companion to the gate test: confirms the
    precondition is a gate, not a hard block on the happy path."""
    rail = _FakeLocator(
        wait_for=AsyncMock(return_value=None),
        # apply succeeds; chip visible (chip_present), then after the hover+dismiss
        # cleanup click it is not visible (removed). Two is_visible calls now — the
        # confirm no longer probes the hover-gated dismiss button's visibility.
        is_visible=AsyncMock(side_effect=[True, False]),
    )
    browser = _fake_browser(rail=rail, url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch")

    with patch.object(smoke, "LinkedInBrowser", return_value=browser):
        rc = asyncio.run(smoke._run("Stripe"))

    browser.apply_company_filter.assert_awaited_once()
    assert browser.apply_company_filter.await_args.args[0] == ["Stripe"]
    assert rc == 0, "rail present + apply/chip/cleanup all clean -> PASS (0)"
