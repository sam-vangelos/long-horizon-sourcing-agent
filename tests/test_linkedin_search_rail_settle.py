"""The fresh-navigation race: LinkedIn's saved-search restore re-renders the
left rail seconds AFTER `navigate_to_search` returns, and whatever starts in
that window loses its keystrokes.

Two locks here:

1. `navigate_to_search` does not hand the page back until the rail signature
   has held still for three consecutive samples (or the 20s cap).
2. `enter_search_string` recovers when the pencil does not open the keyword
   editor: Clear the applied Boolean, then reopen the now-collapsed facet.
   The happy path must never take that fallback (it costs a search).

Live evidence: three campaign run logs (2026-08-18, 2026-09-05, 2026-09-22).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import linkedin.browser as browser_mod
from linkedin.browser import LinkedInBrowser
from linkedin.input_backends import TypingPlan, TypingResult, TypingStep
from shared.governor import UNGOVERNED_FOR_TESTS


TEXTAREA_SELECTOR = 'textarea[id*="free-text-single-value-input"]'

# The rail as it reads sample by sample on a fresh navigation: empty for the
# first two seconds, then the saved search lands and holds.
_RESTORE_SCRIPT = [
    "Add Profile keywords or boolean",
    "Add Profile keywords or boolean",
    "Boolean applied Locations: Brazil 433 results",
    "Boolean applied Locations: Brazil 433 results",
    "Boolean applied Locations: Brazil 433 results",
    "Boolean applied Locations: Brazil 433 results",
]


# ----------------------------------------------------------------------
# navigate_to_search: the rail must settle before the page is handed back
# ----------------------------------------------------------------------


class _RailLocator:
    def __init__(self, page, selector):
        self._page = page
        self._selector = selector

    @property
    def first(self):
        return self

    async def is_visible(self, timeout=None):
        if self._selector == TEXTAREA_SELECTOR:
            return False
        # The collapsed keyword control is what makes the rail "ready" — the
        # same evidence SEARCH_KEYWORD_CONTROL_SELECTORS carries in production.
        return 'aria-label*="Profile keywords"' in self._selector

    async def inner_text(self, timeout=None):
        if self._selector != "aside.left-rail":
            raise AssertionError(f"unexpected inner_text target: {self._selector}")
        return self._page.consume_rail_text()


class _RailPage:
    """A page whose rail text is scripted sample by sample."""

    def __init__(self, rail_texts, clock, urls=None):
        self._rail_texts = list(rail_texts)
        self._urls = list(urls or [])
        self._sample = 0
        self._clock = clock
        self.goto_calls = []

    @property
    def url(self):
        if not self._urls:
            return "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
        return self._urls[min(self._sample, len(self._urls) - 1)]

    def consume_rail_text(self):
        text = self._rail_texts[min(self._sample, len(self._rail_texts) - 1)]
        self._sample += 1
        return text

    async def goto(self, url, **kwargs):
        self.goto_calls.append(url)

    async def wait_for_timeout(self, ms):
        self._clock["t"] += ms / 1000.0

    def locator(self, selector):
        return _RailLocator(self, selector)

    # --- only the created-page bind touches these ---

    async def evaluate(self, expression):
        return True

    async def wait_for_selector(self, selector, **kwargs):
        return object()

    async def close(self):
        self.closed = True


def _rail_browser(page, clock, monkeypatch):
    # raising=False so this file's red run fails on the behavioural assertion,
    # not on the clock seam being absent.
    monkeypatch.setattr(browser_mod, "_monotonic", lambda: clock["t"], raising=False)
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = page
    return browser


def test_navigate_to_search_waits_out_the_saved_search_restore(monkeypatch, capsys):
    clock = {"t": 0.0}
    # Samples 1-2 show the empty rail, sample 3 is the restored saved search,
    # and it holds from there.
    page = _RailPage(
        _RESTORE_SCRIPT,
        clock,
        urls=[
            "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
            "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
            "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch?searchContextId=abc",
        ],
    )
    browser = _rail_browser(page, clock, monkeypatch)

    asyncio.run(browser.navigate_to_search("https://www.linkedin.com/talent/hire/123"))

    # Three identical samples: 1-2 (empty), then the restore at 3, then 4 and 5.
    assert page._sample == 5, "returned before the rail held still for three samples"
    out = capsys.readouterr().out
    assert "[surface] rail settled in 5.0s (saved search restored)" in out, out
    assert out.count("[surface]") == 1, out


def test_navigate_to_search_gives_up_at_the_settle_cap(monkeypatch, capsys):
    clock = {"t": 0.0}

    class _NeverSettles(_RailPage):
        def consume_rail_text(self):
            self._sample += 1
            return f"rail revision {self._sample}"

    page = _NeverSettles([""], clock)
    browser = _rail_browser(page, clock, monkeypatch)

    asyncio.run(browser.navigate_to_search("https://www.linkedin.com/talent/hire/123"))

    out = capsys.readouterr().out
    assert "[surface] rail still changing after 20.0s; proceeding" in out, out
    assert out.count("[surface]") == 1, out
    assert page._sample == 20, "the cap should stop sampling at 20s"


def test_navigate_to_search_reports_a_rail_that_never_moved(monkeypatch, capsys):
    clock = {"t": 0.0}
    page = _RailPage(["Add Profile keywords or boolean"], clock)
    browser = _rail_browser(page, clock, monkeypatch)

    asyncio.run(browser.navigate_to_search("https://www.linkedin.com/talent/hire/123"))

    out = capsys.readouterr().out
    assert "[surface] rail settled in 3.0s (no restore observed)" in out, out


def test_settle_never_raises_when_the_rail_read_fails(monkeypatch, capsys):
    clock = {"t": 0.0}

    class _Broken(_RailPage):
        def consume_rail_text(self):
            self._sample += 1
            raise RuntimeError("Execution context was destroyed")

    page = _Broken([""], clock)
    browser = _rail_browser(page, clock, monkeypatch)

    # A read failure is an unchanged sample, so this settles rather than raising.
    asyncio.run(browser.navigate_to_search("https://www.linkedin.com/talent/hire/123"))

    out = capsys.readouterr().out
    assert "rail settled in 3.0s" in out, out


# ----------------------------------------------------------------------
# _bind_created_recruiter_page: the rebind path must settle too
# ----------------------------------------------------------------------

def _created_page_browser(page, clock, monkeypatch):
    browser = _rail_browser(page, clock, monkeypatch)
    browser._page = None  # nothing is bound yet — that is the point of the path
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    chrome = MagicMock()
    chrome.contexts = [context]
    browser._browser = chrome
    browser._discover_recruiter_url_via_cdp = AsyncMock(
        return_value="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    browser._input_backend.initialize = AsyncMock()
    return browser


def test_created_page_bind_settles_the_rail_before_handing_it_over(
    monkeypatch, capsys
):
    clock = {"t": 0.0}
    page = _RailPage(_RESTORE_SCRIPT, clock)
    browser = _created_page_browser(page, clock, monkeypatch)

    bound = asyncio.run(browser._bind_created_recruiter_page())

    assert bound is True
    assert browser.page is page
    assert page._sample == 5, "bound the page before the rail held still"
    out = capsys.readouterr().out
    assert "[surface] rail settled in 5.0s (saved search restored)" in out, out


def test_created_page_bind_failure_returns_false_without_sampling(
    monkeypatch, capsys
):
    clock = {"t": 0.0}

    class _DeadPage(_RailPage):
        async def goto(self, url, **kwargs):
            raise RuntimeError("net::ERR_ABORTED")

    page = _DeadPage(_RESTORE_SCRIPT, clock)
    browser = _created_page_browser(page, clock, monkeypatch)

    bound = asyncio.run(browser._bind_created_recruiter_page())

    assert bound is False
    assert browser._page is None
    assert page._sample == 0, "a failed navigation must not sample the rail"
    out = capsys.readouterr().out
    assert "[bind] Created-page navigation failed" in out, out
    assert "[surface]" not in out, out


# ----------------------------------------------------------------------
# enter_search_string: Clear-then-Add when the pencil does not open the editor
# ----------------------------------------------------------------------


class _KeywordFacet:
    """The sidebar Keywords facet as a tiny state machine.

    Before Clear, the keyword-control selectors resolve to the pencil (edit);
    after Clear, the facet collapses and the same selectors resolve to the
    "Add Profile keywords or boolean" button — which is how LinkedIn behaves.
    """

    def __init__(self, *, pencil_opens_editor: bool):
        self.pencil_opens_editor = pencil_opens_editor
        self.textarea_visible = False
        self.clear_visible = True
        self.cleared = False
        self.clicks: list[str] = []
        self.selectors_seen: list[str] = []
        self.keys: list[str] = []


class _FacetLocator:
    def __init__(self, facet, name):
        self._facet = facet
        self.name = name

    @property
    def first(self):
        return self

    async def is_visible(self, timeout=None):
        if self.name == "textarea":
            return self._facet.textarea_visible
        if self.name == "clear":
            return self._facet.clear_visible
        return True

    async def wait_for(self, state=None, timeout=None):
        # Playwright raises here; the live runs died on exactly this call.
        if self.name == "textarea" and not self._facet.textarea_visible:
            raise RuntimeError(
                f"Timeout {timeout}ms exceeded waiting for {TEXTAREA_SELECTOR}"
            )

    async def input_value(self):
        return ""

    async def inner_html(self, timeout=None):
        return "<button aria-label='Clear Keywords'></button>"

    async def scroll_into_view_if_needed(self, timeout=None):
        return None


class _FacetPage:
    def __init__(self, facet):
        self._facet = facet
        self.url = "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"

    async def wait_for_timeout(self, ms):
        return None

    async def title(self):
        return "Recruiter Search"

    def locator(self, selector):
        facet = self._facet
        facet.selectors_seen.append(selector)
        lowered = selector.lower()
        if selector == TEXTAREA_SELECTOR:
            return _FacetLocator(facet, "textarea")
        if "clear keywords" in lowered or "pill-facet-clear" in lowered:
            return _FacetLocator(facet, "clear")
        if "keyword" in lowered:
            return _FacetLocator(facet, "add" if facet.cleared else "pencil")
        raise AssertionError(f"Unexpected selector: {selector}")


def _facet_browser(facet):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = _FacetPage(facet)
    browser.go_back_to_results = AsyncMock()
    browser.require_recruiter_tab = AsyncMock()
    browser._wait_for_search_results_ready = AsyncMock(return_value=1350)
    browser._peek_results_count_text = AsyncMock(return_value="100")
    browser._peek_top_card_signature = AsyncMock(return_value=("Ada", "/p/ada"))
    browser._input_backend.type_text = AsyncMock(
        return_value=TypingResult(
            transport="playwright_keyboard",
            duration_ms=2400,
            typo_count=0,
            used_correction=False,
            fallback_char_count=0,
        )
    )

    async def _ghost_click(locator, before_click=None):
        facet.clicks.append(locator.name)
        if locator.name == "pencil" and facet.pencil_opens_editor:
            facet.textarea_visible = True
        if locator.name == "clear":
            facet.clear_visible = False
            facet.cleared = True
        if locator.name == "add":
            facet.textarea_visible = True
        return True

    browser._ghost_click_locator = AsyncMock(side_effect=_ghost_click)
    browser._press_key = AsyncMock(side_effect=lambda key: facet.keys.append(key))
    browser._press_combo = AsyncMock(side_effect=lambda combo: facet.keys.append(combo))
    return browser


def _run_enter(browser):
    plan = TypingPlan(
        steps=[TypingStep(kind="char", value="f", delay_seconds=0.0, source_index=0)]
    )
    with patch("linkedin.browser.asyncio.sleep", new=AsyncMock()), patch(
        "linkedin.browser.human_delay_correlated", side_effect=lambda base, channel: base
    ):
        return asyncio.run(browser.enter_search_string("foo", typing_plan=plan))


def test_enter_search_string_clears_and_reopens_when_the_pencil_is_inert(capsys):
    facet = _KeywordFacet(pencil_opens_editor=False)
    browser = _facet_browser(facet)

    result = _run_enter(browser)

    assert result.typing_result.duration_ms == 2400
    browser._input_backend.type_text.assert_awaited_once()
    assert facet.clicks.count("clear") == 1, facet.clicks
    assert "add" in facet.clicks, facet.clicks
    assert facet.keys.count("Enter") == 1, facet.keys
    out = capsys.readouterr().out
    assert (
        "[keywords] pencil did not open the editor; cleared the applied Boolean "
        "and reopened it" in out
    ), out


def test_enter_search_string_never_clears_when_the_pencil_works(capsys):
    facet = _KeywordFacet(pencil_opens_editor=True)
    browser = _facet_browser(facet)

    _run_enter(browser)

    assert "clear" not in facet.clicks, facet.clicks
    assert not any(
        "Clear Keywords" in sel or "pill-facet-clear" in sel
        for sel in facet.selectors_seen
    ), facet.selectors_seen
    assert "[keywords]" not in capsys.readouterr().out
