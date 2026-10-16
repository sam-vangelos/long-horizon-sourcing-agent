"""The readiness gate run for real against a table-driven DOM.

`assert_search_surface_ready` is the only facet gate no test has ever run
against a DOM: every existing call site mocks the method away
(`tests/test_orchestrator_geography_gate.py:307,333,365,402,451,511,587,683`,
`tests/test_linkedin_search_intelligence.py:395-833`,
`tests/test_linkedin_recruiter_recovery.py:434-565`). That is why nobody
caught what killed `run-20260920-101309`: on LinkedIn's no-results page the
rail is present and hydrated but the result summary and the result cards are
not, so a gate that demanded one of those two refused the page a zero-result
string leaves behind — three strings in a row, against one unchanged page,
into `facet_surface_unreconcilable` (plans/facet-lifecycle-clearing-plan.md
Addendum I2).

These cases drive the production predicate on a real `LinkedInBrowser`
instance whose page reports element counts from a table, so the DOM shape is
the input and the verdict is the output.
"""

from __future__ import annotations

import asyncio

import pytest

from linkedin.browser import (
    SEARCH_KEYWORD_CONTROL_SELECTORS,
    FacetReconciliationError,
    LinkedInBrowser,
)

RAIL = "aside.left-rail"
SUMMARY_TITLE = ".search-query-summary__title"
RESULT_CARD = "ol.profile-list article.profile-list-item"
KEYWORD_TEXTAREA = 'textarea[id*="free-text-single-value-input"]'

PROJECT_ID = "9000000014"
OTHER_PROJECT_ID = "9999999999"
PROJECT_SEARCH_URL = (
    f"https://www.linkedin.com/talent/hire/{PROJECT_ID}/discover/recruiterSearch"
)
FOREIGN_SEARCH_URL = (
    f"https://www.linkedin.com/talent/hire/{OTHER_PROJECT_ID}"
    "/discover/recruiterSearch"
)
PROJECTLESS_SEARCH_URL = "https://www.linkedin.com/talent/search"
PROJECT_PIPELINE_URL = f"https://www.linkedin.com/talent/hire/{PROJECT_ID}"


class _FakeLocator:
    """A locator that knows its selector, its scope, and whether it is shown.

    Scope matters: the keyword controls are only evidence about THIS search
    rail when they are inside it, and `count()` is existence, not visibility
    — a hidden node in a foreign Recruiter document satisfied neither the old
    gate nor what the reconcile needs.
    """

    def __init__(self, page: "_FakePage", selector: str):
        self.page = page
        self.selector = selector

    def locator(self, selector: str) -> "_FakeLocator":
        return _FakeLocator(self.page, f"{self.selector} {selector}")

    @property
    def first(self) -> "_FakeLocator":
        return self

    async def count(self) -> int:
        return int(self.page.counts.get(self.selector, 0))

    async def is_visible(self, timeout: int = 0) -> bool:
        return self.selector in self.page.visible


class _FakePage:
    """A page whose element counts, visibility and URL come from the case."""

    def __init__(
        self,
        counts: dict[str, int],
        visible: set[str] | None = None,
        url: str = PROJECT_SEARCH_URL,
    ):
        self.counts = dict(counts)
        self.visible = set(visible or ())
        self.url = url
        self.waited_ms = 0

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self, selector)

    async def wait_for_timeout(self, ms: int) -> None:
        self.waited_ms += ms


def _browser(
    counts: dict[str, int],
    visible: set[str] | None = None,
    *,
    url: str = PROJECT_SEARCH_URL,
    project_id: str | None = PROJECT_ID,
) -> LinkedInBrowser:
    browser = LinkedInBrowser.__new__(LinkedInBrowser)
    browser._page = _FakePage(counts, visible, url)
    # The project the run is authorized for, pinned on the browser at loop
    # construction (`linkedin/evaluation_loop.py` set_required_project_id).
    browser._required_project_id = project_id
    return browser


def _visible(counts: dict[str, int], hidden) -> set[str]:
    """Everything in the count table is shown unless the case hides it."""
    return set(counts) - set(hidden)


def _assert_ready(counts: dict[str, int], *, hidden=(), **page) -> None:
    asyncio.run(
        _browser(
            counts, _visible(counts, hidden), **page
        ).assert_search_surface_ready(timeout_ms=300)
    )


def _assert_refused(
    counts: dict[str, int], *, gate: str, hidden=(), **page
) -> None:
    browser = _browser(counts, _visible(counts, hidden), **page)
    with pytest.raises(FacetReconciliationError) as exc_info:
        asyncio.run(browser.assert_search_surface_ready(timeout_ms=300))
    assert exc_info.value.gate == gate


def _assert_unready(counts: dict[str, int], *, hidden=(), **page) -> None:
    _assert_refused(counts, gate="search_surface_unready", hidden=hidden, **page)


def _in_rail(selector: str) -> str:
    return f"{RAIL} {selector}"


def test_no_results_page_is_ready_on_the_rail_and_its_keyword_box():
    """The live shape that ended the 09-20 run: rail + Keywords, no results.

    `run-20260918-170049.log:1246-1255` typed the next string's Boolean into
    exactly this page and the string came back with 20 results.
    """
    _assert_ready({RAIL: 1, _in_rail(KEYWORD_TEXTAREA): 1})


@pytest.mark.parametrize("selector", SEARCH_KEYWORD_CONTROL_SELECTORS)
def test_any_visible_sidebar_keyword_control_proves_the_rail_is_usable(selector):
    """Visible is what the gate turns on, so the hidden twin rides along here.

    Asserted as a PAIR: the ready half alone stayed green with `is_visible`
    deleted, which is a lock that cannot fail on the line it exists to hold.
    """
    counts = {RAIL: 1, _in_rail(selector): 1}
    _assert_ready(counts)
    _assert_unready(counts, hidden={_in_rail(selector)})


@pytest.mark.parametrize("selector", SEARCH_KEYWORD_CONTROL_SELECTORS)
def test_a_hidden_keyword_control_is_not_a_usable_rail(selector):
    """`count()` is existence. A collapsed or detached control is not an entry
    point for a Boolean, and `navigate_to_search` has always polled these on
    `is_visible` — the gate now holds the same bar."""
    _assert_unready(
        {RAIL: 1, _in_rail(selector): 1}, hidden={_in_rail(selector)}
    )


@pytest.mark.parametrize("selector", SEARCH_KEYWORD_CONTROL_SELECTORS)
def test_a_keyword_control_outside_the_rail_proves_nothing(selector):
    """The locators were document-global, and `[aria-label*="keywords" i]` is
    a case-insensitive substring — so a non-search Recruiter document with a
    hidden rail node and its own keywords button passed the gate, and the
    opening then typed this project's Boolean into it."""
    _assert_unready({RAIL: 1, selector: 1})


def test_result_summary_still_makes_the_surface_ready():
    _assert_ready({RAIL: 1, SUMMARY_TITLE: 1})


def test_a_result_card_still_makes_the_surface_ready():
    _assert_ready({RAIL: 1, RESULT_CARD: 25})


def test_a_hidden_summary_over_a_hidden_rail_is_not_a_ready_surface():
    """The keyword control was held to `is_visible` and these two were not.

    A summary node and a rail node that exist but are not shown are a foreign
    or half-painted document, which is exactly what the keyword-control fix
    was for — the gate returned ready and the opening typed this project's
    Boolean into it.

    RED against the bare `count()` on the summary: the gate returned.
    """
    _assert_unready(
        {RAIL: 1, SUMMARY_TITLE: 1}, hidden={RAIL, SUMMARY_TITLE}
    )


def test_a_hidden_result_card_over_a_visible_rail_is_not_a_ready_surface():
    _assert_unready({RAIL: 1, RESULT_CARD: 25}, hidden={RESULT_CARD})


def test_a_visible_card_over_a_hidden_rail_is_not_a_ready_surface():
    """The rail itself is the thing the reconcile edits."""
    _assert_unready({RAIL: 1, RESULT_CARD: 25}, hidden={RAIL})


def test_a_bare_rail_with_no_keyword_control_is_unready():
    """Rail markup with no entry point for a Boolean is a half-rendered page."""
    _assert_unready({RAIL: 1})


def test_results_without_a_rail_are_unready():
    """Cards prove a search ran; the reconcile needs the sidebar it edits."""
    _assert_unready(
        {
            SUMMARY_TITLE: 1,
            RESULT_CARD: 25,
            _in_rail(KEYWORD_TEXTAREA): 1,
        }
    )


def test_an_empty_page_is_unready():
    _assert_unready({})


_USABLE_RAIL = {RAIL: 1, _in_rail(KEYWORD_TEXTAREA): 1}


def test_another_projects_search_page_is_refused_by_name():
    """A visible rail and a visible in-rail Keywords box on the WRONG project.

    Every element the gate reads is there — it is a Recruiter search rail, it
    just belongs to someone else's pipeline. The fresh-string path never
    checks the URL (`linkedin/string_driver.py:487-490`), so a ready verdict
    here is followed by this string's geography rewrite and this string's
    Boolean typed into that document; the save boundary only catches it one
    candidate later.

    RED against the DOM-only gate: the gate returned ready.
    """
    _assert_refused(
        _USABLE_RAIL, gate="foreign_search_page", url=FOREIGN_SEARCH_URL
    )


def test_a_recruiter_search_page_with_no_project_id_is_refused_by_name():
    """The global /talent/search view proves nothing about this project, and
    unverified is a mismatch — the same asymmetry `_is_foreign_project_page`
    applies at run-start and at the pre-save boundary."""
    _assert_refused(
        _USABLE_RAIL, gate="foreign_search_page", url=PROJECTLESS_SEARCH_URL
    )


def test_a_non_search_view_of_this_project_is_refused_by_name():
    """Foreign is the wrong project; this is the wrong VIEW of the right one.

    `/talent/hire/<id>` is the pipeline, not the discover search: its rail is
    not the rail the reconcile edits, and the pre-save boundary would pass it
    because the project id matches.
    """
    _assert_refused(
        _USABLE_RAIL, gate="foreign_search_page", url=PROJECT_PIPELINE_URL
    )


def test_this_projects_search_page_with_a_usable_rail_is_ready():
    """The whole point of the URL gate is that it refuses a lookalike and
    nothing else."""
    _assert_ready(_USABLE_RAIL, url=PROJECT_SEARCH_URL)
    _assert_ready(
        _USABLE_RAIL,
        url=f"{PROJECT_SEARCH_URL}?start=25&searchHistoryId=abc",
    )


def test_a_brief_that_pins_no_project_still_needs_a_search_page():
    """Nothing to violate when the brief pins no project, so any Recruiter
    search view is accepted — but a profile page is still not a search."""
    _assert_ready(_USABLE_RAIL, url=PROJECTLESS_SEARCH_URL, project_id=None)
    _assert_refused(
        _USABLE_RAIL,
        gate="foreign_search_page",
        url="https://www.linkedin.com/talent/profile/AEMAAA",
        project_id=None,
    )


def test_a_dead_rail_on_this_projects_search_page_is_still_unready_by_name():
    """The two refusals are different conditions and the loop treats them the
    same way, but the gate name is what the receipt and the capture carry."""
    _assert_unready({RAIL: 1}, url=PROJECT_SEARCH_URL)


def test_the_gate_polls_until_its_timeout_before_refusing():
    browser = _browser({})
    with pytest.raises(FacetReconciliationError):
        asyncio.run(browser.assert_search_surface_ready(timeout_ms=300))
    assert browser._page.waited_ms == 300
