"""Browser automation for LinkedIn Recruiter via Playwright.

All selectors verified against docs/linkedin-recruiter-dom-map.md.
Profile is a SLIDE-IN PANEL, not page navigation.

IMPORTANT: This attaches to an existing browser with an active LinkedIn Recruiter session.
It does NOT launch a new browser or handle login.
"""

from __future__ import annotations
import asyncio
import logging
import random
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING
from shared import config
from shared.human_timing import human_delay_correlated
from linkedin.acquisition import _is_browser_disconnect_error
from linkedin.activity_parser import (
    extract_profile_recent_activity_lines,
    extract_profile_status_summary,
    extract_recruiter_activity_from_card_text,
)
from linkedin.input_backends import TypingPlan, TypingResult, build_boolean_typing_plan, create_input_backend
from linkedin.profile_sections import locate_sections
from linkedin.timing_telemetry import TimingRecorder, emit_timing_event

if TYPE_CHECKING:
    from rebrowser_playwright.async_api import Browser, Page, BrowserContext

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
_TARGET_CRASH_PATTERNS = (
    "target crashed",
    "page crashed",
    "session closed",
    "target closed",
)


class _SaveOperationAbort(BaseException):
    """Carry a save-boundary failure through the retry wrapper."""

    def __init__(self, cause: BaseException):
        self.cause = cause
        super().__init__(str(cause))


class _ProfileReadBudgetExhausted(Exception):
    pass


def normalize_facet_value_for_compare(value: object) -> str:
    """Normalize facet chip/request values for already-applied comparisons."""
    return str(value or "").strip().casefold()


@dataclass(frozen=True)
class SearchEntryResult:
    typing_result: TypingResult
    results_wait_ms: int


def _is_target_crash_error(error: BaseException | str) -> bool:
    text = str(error).lower()
    return _is_browser_disconnect_error(error) or any(
        pattern in text for pattern in _TARGET_CRASH_PATTERNS
    )


@dataclass(frozen=True)
class _NameToken:
    """One name word plus whether the SOURCE text marked it as abbreviated."""

    text: str
    truncated: bool


def _fold_name_text(value: object) -> str:
    """Accent- and compatibility-fold a name so 'José García' == 'Jose Garcia'.

    LinkedIn renders the same person's name with diacritics in the profile
    panel and (often) without them in the result-card lockup. Comparing the
    raw strings marks the SAME person as a mismatch, which fails the
    post-click save probe and aborts the run with ``save_not_persisted``.
    NFKD splits each accented character into base + combining mark; dropping
    the combining marks leaves the base letters.
    """
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def _script_has_initials(word: str) -> bool:
    """True for scripts that abbreviate a name to its first letter.

    Latin, Greek and Cyrillic distinguish case, and in those scripts a lone
    letter really is an initial — 'Biao Z.' abbreviates 'Biao Zhang'. Han,
    Hangul, Kana, Hebrew, Arabic, Thai and Devanagari are caseless and do not
    abbreviate this way: a single 王 is a whole surname. Treating one as an
    initial let the expected name '王小明' confirm against a panel showing
    '王 伟' — a different person — through the prefix comparison below.
    """
    return word.upper() != word.lower()


def _name_tokens(value: object, *, limit: int | None = None) -> list[_NameToken]:
    """Tokenize a name, recording which tokens are explicitly abbreviated.

    A token counts as truncated only when the source text carries an explicit
    abbreviation MARKER — never because the token is merely short. The two
    supported markers are a trailing period after a lone letter in a script
    that has initials ('Biao Z.') and an ellipsis after any token in any script
    ('Biao Zh…'). Shortness alone is not a marker: an unmarked lone letter is a
    whole name in Han/Hangul/Kana/Hebrew/Arabic ('王'), and even in Latin, 'A
    Smith' is not evidence that 'Alexander Smith' is the same person. Those two
    markers are the only cases where a prefix comparison is legitimate; every
    other token must agree exactly, because 'Li' is not 'Lin' and 'Smith' is
    not 'Smithson'.
    """
    folded = _fold_name_text(value)
    tokens: list[_NameToken] = []
    for match in re.finditer(r"[^\W_]+", folded):
        word = match.group(0)
        tail = folded[match.end():match.end() + 3]
        ellipsis_marker = tail.startswith("…") or tail.startswith("...")
        initial_marker = (
            len(word) == 1
            and _script_has_initials(word)
            and tail.startswith(".")
        )
        tokens.append(
            _NameToken(text=word, truncated=ellipsis_marker or initial_marker)
        )
        if limit is not None and len(tokens) >= limit:
            break
    return tokens


def _name_token_matches(expected: _NameToken, actual: _NameToken) -> bool:
    if expected.text == actual.text:
        return True
    if expected.truncated and len(expected.text) < len(actual.text):
        return actual.text.startswith(expected.text)
    if actual.truncated and len(actual.text) < len(expected.text):
        return expected.text.startswith(actual.text)
    return False


def _panel_name_line(panel_text: object) -> str:
    """The profile panel's NAME REGION, approximated as its first rendered line.

    KNOWN LIMITATION, deliberate. The right comparison target is the panel's
    dedicated name element, but no selector in this file reads one: every panel
    selector here is a container (`div.profile__main-container`,
    `div.profile-slidein__container`), and the only name-bearing selector,
    `[class*="lockup__title"] a`, is the RESULT CARD's link, not the panel's.
    Pinning a panel name selector needs a live DOM capture nobody has yet, so
    this reads the first line of the container's innerText instead — the panel
    leads with its header lockup, whose first text is the name
    (docs/linkedin-recruiter-dom-map.md, "Profile Header (Lockup)").

    Scanning a free-text WINDOW instead is what let 'Ann Li' confirm against a
    panel whose name is 'Ann Lin' and whose headline reads 'Principal at Ann Li
    Consulting': the expected name matched BODY copy. Leading blank or
    punctuation-only lines are skipped so the first line that could carry a name
    is the one compared.
    """
    for line in str(panel_text or "").splitlines():
        if re.search(r"[^\W_]", line):
            return line
    return ""


def _panel_confirms_expected_name(panel_text: object, expected_name: object) -> bool:
    """True only when the panel's NAME LINE opens with the expected name.

    The expected tokens are matched against the name line's leading tokens, in
    order, token for token. Anchoring at the start of the line (rather than
    searching anywhere inside it) is what survives a panel that renders the
    lockup as one line — 'Ann Lin · 2nd · Ann Li Consulting' — where a span
    search finds 'Ann Li' in the headline and confirms the wrong person.
    Trailing tokens the line carries after the name (a connection-degree badge,
    a suffix the result card omitted) do not block a match; leading ones do.
    """
    expected_tokens = _name_tokens(expected_name)
    if not expected_tokens:
        return False
    name_tokens = _name_tokens(
        _panel_name_line(panel_text), limit=len(expected_tokens)
    )
    if len(name_tokens) < len(expected_tokens):
        return False
    return all(
        _name_token_matches(expected, actual)
        for expected, actual in zip(expected_tokens, name_tokens)
    )


# A Recruiter URL carries its project id in the path: /talent/hire/<id>/…
_PROJECT_ID_IN_URL_RE = re.compile(r"/talent/hire/([^/?#]+)")


def _recruiter_page_project_id(url: object) -> str | None:
    """The Recruiter project id a /talent/hire/<id>/… URL belongs to, if any.

    Only the PATH is read. LinkedIn round-trips redirect targets through query
    strings, and a `/talent/hire/<id>` sitting in one names where the page is
    going, not the project it currently belongs to — believing it would let a
    projectless page claim the brief's project.

    Defined here rather than in the orchestrator because both layers need it and
    the import only runs this direction; `linkedin.orchestrator` imports this
    name so there is exactly one implementation to keep honest.
    """
    path = str(url or "").split("?", 1)[0].split("#", 1)[0]
    match = _PROJECT_ID_IN_URL_RE.search(path)
    if not match:
        return None
    return match.group(1).strip() or None


def recruiter_project_search_url(project_id: object) -> str:
    """The canonical Recruiter search URL for a project, or '' when unpinned."""
    value = str(project_id or "").strip()
    if not value:
        return ""
    return f"https://www.linkedin.com/talent/hire/{value}/discover/recruiterSearch"


def _is_unusable_cdp_page_url(url: str) -> bool:
    """CDP targets that should not be chosen when no Recruiter tab is open yet."""
    lowered = (url or "").strip().lower()
    if not lowered:
        return True
    if "chrome-devtools://" in lowered or lowered.startswith("devtools://"):
        return True
    if "chrome://omnibox" in lowered:
        return True
    return False


async def _retry(coro_fn, retries=MAX_RETRIES, delay=RETRY_DELAY_SECONDS):
    """Retry an async callable up to `retries` times with randomized delay."""
    last_err = None
    for attempt in range(retries):
        try:
            return await coro_fn()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                jittered = delay * random.uniform(0.5, 1.5)
                print(f"    [retry {attempt + 1}/{retries}] {e}")
                await asyncio.sleep(jittered)
    raise last_err


class LinkedInBrowser:
    """Manages a Playwright connection to LinkedIn Recruiter via CDP."""

    def __init__(
        self,
        input_mode: str = "concurrent",
        *,
        governor=None,
        timing_recorder: TimingRecorder | None = None,
        state_dir=None,
    ):
        # P8.1: governance attaches to the browser at construction, not to
        # individual call sites. open_profile()/open_profile_by_url() check
        # and count against this governor themselves. Omitting it raises —
        # production bypass of profile-open governance must be impossible by
        # default. Tests that don't care about governance mechanics pass the
        # explicit shared.governor.UNGOVERNED_FOR_TESTS sentinel.
        if governor is None:
            raise ValueError(
                "LinkedInBrowser requires an explicit governor. Pass the shared "
                "SessionGovernor instance in production code, or "
                "shared.governor.UNGOVERNED_FOR_TESTS in tests."
            )
        self._governor = governor
        self._timing_recorder = timing_recorder
        self._state_dir = state_dir
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._owns_connection = False
        # R7 (hardened): the PAGE OBJECT we created on the Chrome-148 created-page
        # path, or None when we adopted/fell-back to one of Sam's own tabs. We
        # track the object — NOT a flag on self._page — because self._page is
        # reassigned on every sidebar op (require_recruiter_tab ->
        # _bind_existing_recruiter_page) and on crash recovery / fallback binds.
        # A flag-on-self._page would, after such a rebind, point ownership at the
        # rebound (adopted) tab and on disconnect (a) close Sam's tab while
        # (b) leaking the page we actually created. disconnect() closes exactly
        # this object, regardless of what self._page currently points to. Set in
        # _bind_created_recruiter_page's success path; never set for adopted /
        # fallback / recovery binds; cleared on disconnect.
        self._owned_page: Optional[Page] = None
        self.input_mode = input_mode
        self._input_backend = create_input_backend(input_mode)
        self._project_id: Optional[str] = None  # Auto-detected from browser URL
        # F4: the Recruiter project this run is authorized to touch, pinned by the
        # orchestrator from the brief. `_bind_existing_recruiter_page` otherwise
        # scores tabs purely on URL shape, so with several Recruiter tabs open it
        # can bind another project's — on first connect, on every sidebar rebind,
        # and on crash recovery. None means unpinned: scoring is unchanged.
        self._required_project_id: Optional[str] = None
        self._last_search_snapshot: dict[str, Any] = {}
        # Distinguishes save failure modes: None | "save_trigger_not_found"
        # (class rotation) | "save_not_persisted" (clicked but state unchanged).
        self._last_save_failure_reason: Optional[str] = None

    def attach_existing_connection(
        self,
        browser: Browser,
        *,
        context: BrowserContext | None = None,
    ) -> None:
        """Reuse an already-attached CDP browser/context from the session orchestrator."""
        self._browser = browser
        self._context = context
        self._owns_connection = False

    async def connect(self) -> None:
        if self._browser is None:
            from rebrowser_playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.connect_over_cdp(config.CDP_URL)
            self._owns_connection = True

        if not self._browser.contexts:
            raise RuntimeError("No browser contexts found. Is the browser open?")

        if await self._bind_existing_recruiter_page(preferred_context=self._context) and await self._page_is_live():
            print(
                f"  Connected to browser ({self._input_backend.status_label}). "
                f"Active page: {self._page.url}"
            )
            return

        # Chrome 148+ ships the CDP "tab target" model: connect_over_cdp adopts a
        # pre-existing tab's inner page target but never completes the renderer
        # handshake, so the adopted page is a zombie (empty url, evaluate hangs
        # forever, reload hangs). Pages Playwright CREATES are unaffected. When
        # adoption yields nothing live, create our own page and navigate it to the
        # live Recruiter URL (discovered via the browser-level CDP layer, which is
        # unaffected). This keeps the real-Chrome / real-profile stealth posture while
        # sidestepping the broken adopt-a-tab path.
        if await self._bind_created_recruiter_page(preferred_context=self._context):
            print(
                f"  Connected via a Playwright-created page ({self._input_backend.status_label}); "
                f"adopting the existing tab returned no live page (Chrome tab-target model).\n"
                f"  Active page: {self._page.url}"
            )
            return

        if await self._bind_fallback_page(preferred_context=self._context):
            print(
                f"  Connected over CDP ({self._input_backend.status_label}), but no LinkedIn Recruiter tab "
                f"was found yet.\n"
                f"  Active page: {self._page.url}\n"
                f"  Open https://www.linkedin.com/talent (project or Recruiter search) before the first "
                f"search action."
            )
            return

        all_urls = []
        for ctx in self._browser.contexts:
            for page in ctx.pages:
                try:
                    all_urls.append(page.url)
                except Exception:
                    all_urls.append("(unavailable tab url)")
        urls_text = "\n    ".join(all_urls) if all_urls else "(no tabs open)"
        raise RuntimeError(
            "Could not attach Playwright to any browser tab over CDP.\n"
            f"  Found {len(all_urls)} tab(s):\n    {urls_text}\n"
            "  Open a normal page in Chrome (e.g. https://www.linkedin.com/talent ) and retry."
        )

    async def disconnect(self) -> None:
        await self._input_backend.shutdown()
        # R7 (hardened): close ONLY the page WE created (the Chrome-148
        # created-page path), tracked as the OBJECT in self._owned_page — never
        # self._page-by-flag. self._page may by now point at an adopted tab a
        # rebind moved us to (require_recruiter_tab / fallback / crash recovery),
        # so closing self._page would shut Sam's own tab AND leak the created one.
        # Closing the tracked object guarantees the created page is always closed
        # (no leak after any rebind) and an adopted/bound page is never closed (we
        # only ever close the object we created). Best-effort and fully guarded: a
        # disconnect must complete even if the page is already gone.
        if self._owned_page is not None:
            try:
                await self._owned_page.close()
            except Exception as exc:
                log.debug("disconnect: created-page close failed (ignored): %s", exc)
        self._owned_page = None
        if self._owns_connection and self._playwright:
            await self._playwright.stop()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._owns_connection = False

    async def _bind_existing_recruiter_page(
        self,
        *,
        preferred_context: BrowserContext | None = None,
        require_project: bool = False,
    ) -> bool:
        """Bind to the first healthy LinkedIn Recruiter tab in the attached browser.

        ``require_project`` turns the pinned project from a preference into a
        precondition: a tab that is not PROVABLY this brief's project is not a
        bind candidate at all. Callers that operate the bound tab immediately
        (`require_recruiter_tab`, which every sidebar op runs first) pass True,
        because there is no later navigation to correct a wrong bind — the very
        next thing that happens is a search typed into whatever project the tab
        belongs to. `connect()` and crash recovery leave it False: both are
        followed by a project-aware navigation that corrects the page, so
        refusing there would only strand a run whose page was about to be fixed.
        """
        if not self._browser:
            return False

        contexts: list[BrowserContext] = []
        if preferred_context is not None and preferred_context in self._browser.contexts:
            contexts.append(preferred_context)
        for ctx in self._browser.contexts:
            if ctx not in contexts:
                contexts.append(ctx)

        candidates: list[tuple[int, BrowserContext, Page, str]] = []
        for ctx_index, ctx in enumerate(contexts):
            for page in ctx.pages:
                try:
                    url = page.url
                except Exception:
                    continue
                if "linkedin.com/talent" not in url or "/login" in url:
                    continue
                score = 0
                if "/discover/" in url or "/recruiterSearch" in url:
                    score += 3
                if "/talent/hire/" in url:
                    score += 2
                if "/talent/search" in url:
                    score += 1
                # F4: project identity outranks every shape signal. The old
                # scoring maxed out at 5, so another project's search tab tied
                # with the authorized one and enumeration order picked the
                # winner. A matching tab is preferred outright; a provably
                # foreign one is taken only if nothing else is left, so a bind
                # still succeeds and the caller's navigation corrects it.
                #
                # Under `require_project` the same asymmetry F1 established at
                # the pre-save boundary applies here: a page that does not NAME
                # the required project is unverified, and unverified is a
                # mismatch. Both the foreign tab and the projectless one
                # (/talent/search, a bare profile page) are dropped from the
                # candidate list rather than deprioritised, because scoring only
                # decides which tab wins — with the right tab gone, last place
                # still binds.
                if self._required_project_id:
                    page_project = _recruiter_page_project_id(url)
                    if page_project == self._required_project_id:
                        score += 100
                    elif require_project:
                        continue
                    elif page_project:
                        score -= 100
                candidates.append((score - ctx_index, ctx, page, url))

        for _, ctx, page, url in sorted(candidates, key=lambda item: item[0], reverse=True):
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
                await self._input_backend.initialize(page)
            except Exception as e:
                print(f"  [bind] Skipping unstable Recruiter tab ({url[:100]}): {e}")
                continue

            self._context = ctx
            self._page = page
            # One parser. The digits-only regex this replaced disagreed with
            # `_recruiter_page_project_id` (the guard parser, which accepts any
            # path segment), so a non-numeric project bound fine but recorded
            # `_project_id = None` — the bind and the guard describing the same
            # page differently, the exact split F4 unified away between layers.
            self._project_id = _recruiter_page_project_id(url) or self._project_id
            return True
        return False

    async def _bind_fallback_page(
        self,
        *,
        preferred_context: BrowserContext | None = None,
    ) -> bool:
        """Bind to any usable tab when CDP is up but no linkedin.com/talent URL is open."""
        if not self._browser:
            return False

        contexts: list[BrowserContext] = []
        if preferred_context is not None and preferred_context in self._browser.contexts:
            contexts.append(preferred_context)
        for ctx in self._browser.contexts:
            if ctx not in contexts:
                contexts.append(ctx)

        triples: list[tuple[BrowserContext, Page, str]] = []
        for ctx in contexts:
            for page in ctx.pages:
                try:
                    url = page.url or ""
                except Exception:
                    continue
                triples.append((ctx, page, url))

        async def _try_bind(ctx: BrowserContext, page: Page) -> bool:
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
                await self._input_backend.initialize(page)
            except Exception as exc:
                try:
                    preview = page.url[:120]
                except Exception:
                    preview = "(url unavailable)"
                print(f"  [bind] Skipping page ({preview}): {exc}")
                return False
            self._context = ctx
            self._page = page
            self._project_id = None
            return True

        filtered = [(c, p, u) for c, p, u in triples if not _is_unusable_cdp_page_url(u)]

        for ctx, page, url in filtered:
            if url.startswith("http://") or url.startswith("https://"):
                if await _try_bind(ctx, page):
                    return True
        for ctx, page, url in filtered:
            if await _try_bind(ctx, page):
                return True
        for ctx, page, url in triples:
            if await _try_bind(ctx, page):
                return True
        return False

    async def _page_is_live(self) -> bool:
        """True if the bound page has a working renderer (evaluate responds quickly).

        Chrome's tab-target model can leave a CDP-adopted page in a zombie state:
        attached but with no execution context, so evaluate hangs forever. A short
        probe distinguishes a usable page from a zombie without blocking startup.
        """
        page = self._page
        if page is None:
            return False
        try:
            return bool(await asyncio.wait_for(page.evaluate("() => true"), timeout=3))
        except Exception:
            return False

    async def _discover_recruiter_url_via_cdp(self) -> str | None:
        """Read the live Recruiter tab URL from the browser-level CDP target list.

        The browser CDP layer reports target URLs even when an adopted page object's
        own .url is empty (the zombie case), so this is how we recover the URL to
        navigate a freshly-created page to.
        """
        if not self._browser:
            return None
        try:
            session = await self._browser.new_browser_cdp_session()
            result = await session.send("Target.getTargets", {"filter": [{"type": "page"}]})
        except Exception:
            return None
        for info in result.get("targetInfos", []):
            url = info.get("url", "") or ""
            if "linkedin.com/talent" in url and "/login" not in url:
                return url
        return None

    async def _bind_created_recruiter_page(
        self,
        *,
        preferred_context: BrowserContext | None = None,
    ) -> bool:
        """Create a Playwright-owned page and navigate it to the live Recruiter URL.

        The Chrome-148 fix path: adopting a pre-existing tab yields a zombie, but pages
        Playwright creates get a working renderer. We discover the Recruiter URL via the
        browser CDP layer, open our own page in the same (authed) context, navigate to
        it, and bind once the search chrome has hydrated. Returns False (and cleans up
        the created page) if discovery, navigation, or the liveness check fails.
        """
        if not self._browser:
            return False
        url = await self._discover_recruiter_url_via_cdp()
        if not url:
            return False
        context = preferred_context if preferred_context in self._browser.contexts else None
        if context is None:
            context = self._browser.contexts[0] if self._browser.contexts else None
        if context is None:
            return False
        try:
            page = await context.new_page()
        except Exception:
            return False
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            if not await asyncio.wait_for(page.evaluate("() => true"), timeout=8):
                raise RuntimeError("created page failed liveness check")
            # Best-effort: let the SPA build the search chrome before we hand off.
            try:
                await page.wait_for_selector(
                    "aside.left-rail, ol.profile-list", timeout=20000, state="attached"
                )
            except Exception:
                pass
        except Exception as exc:
            print(f"  [bind] Created-page navigation failed ({url[:80]}): {exc}")
            # We close the page we just created here, so it must NOT also be
            # tracked as owned — otherwise disconnect() would double-close it.
            # (Ownership is set only on the success path below, so this is also a
            # defensive guard against a future refactor that moves the assignment
            # earlier.)
            self._owned_page = None
            try:
                await page.close()
            except Exception:
                pass
            return False
        self._context = context
        self._page = page
        # R7 (hardened): WE created this page, so WE own it — track the OBJECT so
        # disconnect() closes exactly this page even after later rebinds move
        # self._page onto an adopted tab. Avoids both the tab-leak (created page
        # never closed) and the wrong-tab close (Sam's adopted tab closed) that a
        # flag-on-self._page produced once require_recruiter_tab rebound the page.
        self._owned_page = page
        await self._input_backend.initialize(page)
        # Same single parser as every other project read (see the bind loop).
        self._project_id = _recruiter_page_project_id(url) or self._project_id
        return True

    async def require_recruiter_tab(self) -> None:
        """Bind THIS BRIEF'S Recruiter tab before any op that drives Recruiter DOM.

        F4. Every sidebar op (`enter_search_string`, the location/company/title
        appliers) calls this first and then immediately operates the tab it
        returns, so the bind IS the project decision — nothing navigates
        afterwards to correct it. Binding on URL shape alone meant that once the
        brief's own tab went away (closed, crashed, or skipped as unstable), the
        next-best Recruiter tab won and the owner's search was typed into
        another project's sidebar; every candidate reviewed and saved from that
        page belonged to the wrong pipeline. Refusing is the correct outcome:
        the run stops resumable with its work still owed, instead of quietly
        completing against a project it was never authorized to touch.
        """
        if await self._bind_existing_recruiter_page(
            preferred_context=self._context,
            require_project=True,
        ):
            # Same single parser as every other project read.
            self._project_id = (
                _recruiter_page_project_id(self._page.url) or self._project_id
            )
            return
        current = ""
        try:
            current = str(self._page.url)
        except Exception:
            current = ""
        scope = (
            f" for project {self._required_project_id}"
            if self._required_project_id
            else ""
        )
        raise RuntimeError(
            f"LinkedIn Recruiter is required on an open browser tab{scope}, "
            "but none was found.\n"
            f"  Current Playwright tab: {current or '(unknown)'}\n"
            "  Open https://www.linkedin.com/talent (your project or Recruiter search), then retry.\n"
            "  CDP is connected — this is not a Chrome launch failure."
        )

    async def _ghost_click(self, selector: str) -> bool:
        """Click using ghost-cursor (Bézier trajectory + Fitts's Law timing).

        Falls back to JS eval click if ghost-cursor unavailable or fails.
        Returns True if ghost-cursor succeeded, False if fell back to JS.
        """
        return await self._input_backend.click_selector(self.page, selector)

    async def _ghost_move(self, selector: str) -> bool:
        """Move cursor to element without clicking. Returns True if succeeded."""
        return await self._input_backend.move_selector(self.page, selector)

    async def _ghost_click_locator(
        self,
        locator,
        *,
        before_click: Callable[[], None] | None = None,
    ) -> bool:
        """Ghost-click a Playwright Locator (not a CSS selector).

        Contract: returns True when a click was performed by ANY means
        (ghost-cursor OR the Playwright fallback below), and raises if no
        click could be performed at all. It must NEVER return False after
        having clicked — callers that perform their own click on a falsy
        return (e.g. save_candidate's JS-evaluate fallback) would otherwise
        double-click a live control. The backend's click_locator returns
        False meaning "did not click"; this wrapper's Playwright fallback
        DID click, so it reports True.

        Reach guard: scroll the target into view BEFORE the ghost click. The
        ghost-cursor backend computes absolute cursor coordinates from the
        element's bounding box; if the element is below the fold those
        coordinates land off-screen and the click silently misses. Playwright's
        own .click() already auto-scrolls, so the guard only matters for the
        ghost-cursor path. A scroll failure must NOT convert a previously
        working click into a raise, so it degrades to attempting the click
        anyway.

        Commit boundary: ``before_click`` is the last check between deciding to
        click and the pointer press, so NOTHING that can move the page may run
        after it. ``locator.click()`` violates that — its actionability wait and
        re-scroll happen inside the call, i.e. after the guard already cleared,
        so a card that shifts (or a slide-in that swaps identity) during that
        work gets clicked anyway. The fallback therefore does the positioning
        itself (``hover`` runs Playwright's full actionability sequence and
        parks the pointer on the element's action point), then runs the guard,
        then presses with ``mouse.down``/``mouse.up`` — the same two events
        ``locator.click()`` would have issued, and the same shape as the
        ghost-cursor backend's own commit. This path is the production one:
        ``python_ghost_cursor`` imports ``playwright.async_api``, which is not
        installed alongside ``rebrowser_playwright``, so
        ``ConcurrentInputBackend._cursor`` is None and ``click_locator``
        always returns False.
        """
        try:
            await locator.scroll_into_view_if_needed(timeout=2000)
        except Exception:
            pass
        if await self._input_backend.click_locator(
            self.page,
            locator,
            about_to_commit=before_click,
        ):
            return True
        # Ghost-cursor unavailable/failed: position first, guard second, press
        # third. Never locator.click() after the guard (see docstring).
        await locator.hover(timeout=5000)
        if before_click is not None:
            before_click()
        await self.page.mouse.down()
        await self.page.mouse.up()
        return True

    async def _human_scroll(self, delta_y: int, *, channel: str) -> int:
        events = await self._input_backend.scroll(
            self.page, delta_y, channel=channel
        )
        if (
            getattr(self, "_profile_read_timing_active", False)
            and isinstance(events, int)
        ):
            self._profile_read_wheel_events += max(0, events)
        return events if isinstance(events, int) else 0

    async def _press_key(self, key: str) -> None:
        handled = await self._input_backend.press_key(self.page, key)
        if not handled:
            await self.page.keyboard.press(key)

    async def _press_combo(self, combo: str) -> None:
        handled = await self._input_backend.press_combo(self.page, combo)
        if not handled:
            await self.page.keyboard.press(combo)

    async def _send_os_refresh_shortcut(self) -> bool:
        """Best-effort macOS Cmd+R fallback when Playwright page methods are unhealthy."""
        if sys.platform != "darwin":
            return False

        script = (
            'tell application "Google Chrome" to activate\n'
            'tell application "System Events"\n'
            '  keystroke "r" using command down\n'
            'end tell'
        )
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )
            await asyncio.sleep(4)
            return True
        except Exception as e:
            print(f"  [recovery] OS-level Cmd+R failed: {e}")
            return False

    async def refresh_active_tab(self) -> bool:
        """Try increasingly forceful refresh mechanisms for the active Recruiter tab."""
        try:
            await self.page.reload(wait_until="domcontentloaded", timeout=30000)
            await self.page.wait_for_timeout(4000)
            return True
        except Exception as reload_error:
            print(f"  [recovery] Page reload failed: {reload_error}")

        try:
            await self._press_key("Meta+R")
            await self.page.wait_for_timeout(4000)
            return True
        except Exception as shortcut_error:
            print(f"  [recovery] Playwright Meta+R failed: {shortcut_error}")

        return await self._send_os_refresh_shortcut()

    async def recover_from_target_crash(self, recovery_url: str | None = None) -> bool:
        """Recover from a Chromium target crash by refreshing and rebinding the Recruiter page."""
        print("  [recovery] Detected browser target crash — attempting tab refresh...")
        if not await self.refresh_active_tab():
            return False

        for _ in range(3):
            try:
                rebound = await self._bind_existing_recruiter_page()
            except Exception:
                rebound = False

            if rebound:
                try:
                    current_url = self.page.url
                except Exception:
                    current_url = ""
                # F4: being on Recruiter was treated as good enough, so a rebind
                # that landed on ANOTHER project's tab was left in place and the
                # run continued against the wrong pipeline until the pre-save
                # guard aborted it. Recruiter is not the same as the right
                # project. The project fallback matters because every production
                # caller (`check_and_recover`) passes no recovery_url at all —
                # without it this correction would never run.
                off_recruiter = (
                    "linkedin.com/talent" not in current_url
                    or "/manage/" in current_url
                )
                wrong_project = bool(
                    self._required_project_id
                    and _recruiter_page_project_id(current_url)
                    != self._required_project_id
                )
                if off_recruiter or wrong_project:
                    target = recovery_url or recruiter_project_search_url(
                        self._required_project_id
                    )
                    if target:
                        await self.navigate_to_search(target)
                return True
            await asyncio.sleep(2)
        return False

    @property
    def page(self) -> Page:
        if not self._page:
            raise RuntimeError("Browser not connected. Call connect() first.")
        return self._page

    def set_required_project_id(self, project_id: object) -> None:
        """Pin the Recruiter project every future tab bind must prefer.

        F4. Called by the orchestrator once the brief is final. Advisory, not a
        guard: it steers `_bind_existing_recruiter_page` toward the right tab so
        the run does not wander into another project on rebind or crash
        recovery. What actually *refuses* a wrong-project save is
        `_assert_brief_project_context` at the irreversible boundary; this only
        keeps the run from repeatedly having to.
        """
        value = str(project_id or "").strip()
        self._required_project_id = value or None

    def get_current_project_id(self) -> str | None:
        """Return the current project ID parsed from the browser URL, if any."""
        return self._project_id

    def get_current_search_url(self) -> str | None:
        """Return the current page URL if it looks like a Recruiter search surface."""
        try:
            url = self.page.url
            if url and "linkedin.com/talent" in url:
                return url
        except Exception:
            pass
        return None

    async def apply_advanced_search_plan(self, plan) -> Any:
        """Delegate to the advanced search controller module."""
        from linkedin.advanced_search import apply_advanced_search_plan, snapshot_controls_from_plan
        result = await apply_advanced_search_plan(self, plan)
        if result.applied_controls or result.success:
            # R5: persist ONLY the controls that cleanly applied, so a partial or
            # dropped control is never recorded — and therefore never replayed —
            # as if it had landed (the keyword Boolean is carried separately on
            # the snapshot). A dimension that is *also* in failed/unsupported is a
            # mixed/failed outcome (e.g. one of two same-dimension controls failed
            # closed); exclude it entirely and fall back to the Boolean rather than
            # record a partial. The remaining gap below the dimension grain is
            # already closed in the apply methods: apply_location_filter /
            # apply_company_filter fail closed unless EVERY requested value landed.
            clean = (
                set(result.applied_controls)
                - set(result.failed_controls)
                - set(result.unsupported_controls)
            )
            self._last_search_snapshot = snapshot_controls_from_plan(
                plan, applied_dimensions=clean
            )
        return result

    async def snapshot_advanced_search_controls(self) -> dict:
        """Return a snapshot of the current sidebar search state."""
        if self._last_search_snapshot:
            return dict(self._last_search_snapshot)
        from linkedin.advanced_search import snapshot_controls_from_plan, AdvancedSearchPlan
        return snapshot_controls_from_plan(AdvancedSearchPlan())

    # ------------------------------------------------------------------
    # Emergency recovery — "break glass" protocol
    # ------------------------------------------------------------------

    async def check_and_recover(self) -> bool:
        """Detect error/stuck states and attempt recovery. Returns True if recovery happened."""
        try:
            _ = self.page.url
        except Exception as e:
            if _is_target_crash_error(e):
                return await self.recover_from_target_crash()
            return False

        try:
            # Check for LinkedIn's "Something went wrong" error page
            error_heading = self.page.locator('text="Something went wrong"').first
            try_again_btn = self.page.locator('button:has-text("Try again")').first

            if await error_heading.is_visible(timeout=1000):
                print("  [recovery] Detected 'Something went wrong' page — attempting recovery...")
                if self._state_dir is not None:
                    try:
                        from linkedin.page_capture import capture_page_state

                        await capture_page_state(
                            self, self._state_dir, reason="health_something_went_wrong"
                        )
                    except Exception:
                        pass
                # Strategy 1: Click "Try again" if present
                try:
                    if await try_again_btn.is_visible(timeout=2000):
                        await try_again_btn.click()
                        await self.page.wait_for_timeout(5000)
                        # Check if recovery succeeded
                        if not await error_heading.is_visible(timeout=2000):
                            print("  [recovery] 'Try again' click succeeded.")
                            return True
                except Exception:
                    pass

                if config.LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED:
                    try:
                        if await error_heading.is_visible(timeout=2000):
                            print(
                                "  [recovery] SWG persists after one retry — leaving page for the live health classifier."
                            )
                            return False
                    except Exception:
                        pass

                # Strategy 2: Reload the page
                print("  [recovery] 'Try again' didn't work — reloading page...")
                await self.page.reload(wait_until="domcontentloaded", timeout=30000)
                await self.page.wait_for_timeout(5000)
                if not await error_heading.is_visible(timeout=2000):
                    print("  [recovery] Page reload succeeded.")
                    return True

                # Strategy 3: Navigate back to LinkedIn Recruiter base
                print("  [recovery] Reload didn't work — navigating to LinkedIn Recruiter home...")
                await self.page.goto("https://www.linkedin.com/talent/search", wait_until="domcontentloaded", timeout=30000)
                await self.page.wait_for_timeout(5000)
                print("  [recovery] Navigated to LI Recruiter home. Will need to re-enter project.")
                return True
        except Exception as e:
            if _is_target_crash_error(e):
                return await self.recover_from_target_crash()
            pass  # No error page detected — normal state

        # Check for blank/empty page (no LinkedIn DOM at all)
        try:
            url = self.page.url
            if "linkedin.com" not in url:
                print(f"  [recovery] Page navigated away from LinkedIn ({url}) — going back...")
                await self.page.go_back(wait_until="domcontentloaded", timeout=30000)
                await self.page.wait_for_timeout(3000)
                return True
        except Exception:
            pass

        # Check for login redirect
        try:
            if "/login" in self.page.url or "/uas/login" in self.page.url:
                print("  [recovery] CRITICAL: Redirected to login page. Session may have expired.")
                print("  [recovery] Please re-authenticate in the browser window and resume the run.")
                if self._state_dir is not None:
                    try:
                        from linkedin.page_capture import capture_page_state

                        await capture_page_state(
                            self, self._state_dir, reason="health_login_redirect"
                        )
                    except Exception:
                        pass
                raise RuntimeError("LinkedIn session expired — re-authenticate and resume.")
        except RuntimeError:
            raise
        except Exception:
            pass

        return False

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    async def navigate_to_search(self, project_url: str) -> None:
        await self.page.goto(project_url, wait_until="domcontentloaded", timeout=30000)
        # Wait for sidebar filters to render. LinkedIn's SPA can take 8-15s
        # to hydrate the sidebar after DOM content loads.
        await self.page.wait_for_timeout(4000)  # minimum wait for initial render
        # Poll for sidebar keyword controls or results summary
        sidebar_selectors = [
            'textarea[id*="free-text-single-value-input"]',
            'button[aria-label*="Profile keywords"]',
            'button[aria-label*="Edit Keywords"]',
            'button[aria-label*="keywords" i]',
            '.search-query-summary__title',
        ]
        for _ in range(12):  # up to ~12 more seconds
            for sel in sidebar_selectors:
                try:
                    if await self.page.locator(sel).first.is_visible(timeout=500):
                        return  # sidebar rendered
                except Exception:
                    continue
            await self.page.wait_for_timeout(500)

        # If we get here, sidebar never appeared
        final_url = self.page.url
        if self._state_dir is not None:
            try:
                from linkedin.page_capture import capture_page_state

                await capture_page_state(
                    self, self._state_dir, reason="navigate_sidebar_missing"
                )
            except Exception:
                pass
        raise RuntimeError(
            f"navigate_to_search failed: sidebar elements not found after 16s. "
            f"Page URL: {final_url[:120]}"
        )

    # ------------------------------------------------------------------
    # Search: enter Boolean into Keywords field
    # ------------------------------------------------------------------

    async def _wait_for_visible_poll(self, locator, *, timeout_ms: int, interval_ms: int) -> bool:
        attempts = max(1, timeout_ms // interval_ms)
        for _ in range(attempts):
            try:
                if await locator.is_visible(timeout=250):
                    return True
            except Exception:
                pass
            await self.page.wait_for_timeout(interval_ms)
        try:
            return await locator.is_visible(timeout=250)
        except Exception:
            return False

    async def _read_textarea_value(self, textarea) -> str:
        try:
            return await textarea.input_value()
        except Exception:
            try:
                return await textarea.evaluate("el => el.value || ''")
            except Exception:
                return ""

    async def _clear_keyword_textarea(self, textarea) -> None:
        await self._ghost_click_locator(textarea)
        await self._press_combo("Meta+A")
        await asyncio.sleep(random.uniform(0.08, 0.18))
        await self._press_key("Backspace")
        value = await self._read_textarea_value(textarea)
        if value:
            await asyncio.sleep(random.uniform(0.08, 0.18))
            await self._press_combo("Meta+A")
            await asyncio.sleep(random.uniform(0.08, 0.18))
            await self._press_key("Backspace")
            value = await self._read_textarea_value(textarea)
        if value:
            raise RuntimeError("keyword textarea did not clear after select-all + backspace")

    async def _peek_results_count_text(self) -> str:
        try:
            el = self.page.locator(".search-query-summary__title").first
            text = (await el.inner_text(timeout=250)).strip()
            if not text:
                return ""
            match = re.search(r"([\d.,]+[KkMm]?\+?)", text)
            return match.group(1) if match else ""
        except Exception:
            return ""

    async def _peek_top_card_signature(self) -> tuple[str, str]:
        try:
            link = self.page.locator(
                'ol.profile-list article.profile-list-item [class*="lockup__title"] a'
            ).first
            name = (await link.inner_text(timeout=250)).strip()
            url = (await link.get_attribute("href")) or ""
            return (name, url)
        except Exception:
            return ("", "")

    async def _wait_for_search_results_ready(
        self,
        *,
        previous_count_text: str,
        previous_top_card_signature: tuple[str, str],
        min_wait_ms: int = 1200,
        interval_ms: int = 150,
        timeout_ms: int = 3000,
    ) -> int:
        waited_ms = min_wait_ms
        await self.page.wait_for_timeout(min_wait_ms)
        while waited_ms < timeout_ms:
            count_text = await self._peek_results_count_text()
            card_signature = await self._peek_top_card_signature()
            if (
                count_text
                and previous_count_text
                and count_text != previous_count_text
            ) or (
                card_signature != ("", "")
                and card_signature != previous_top_card_signature
            ):
                return waited_ms
            await self.page.wait_for_timeout(interval_ms)
            waited_ms += interval_ms
        return timeout_ms

    async def enter_search_string(
        self,
        boolean: str,
        *,
        typing_plan: TypingPlan | None = None,
    ) -> SearchEntryResult:
        """Enter a Boolean into the sidebar Keywords field (NOT the global search bar).

        Uses the project search page sidebar: Clear Keywords button → edit button → textarea.
        NEVER targets input[aria-label*="keyword"] which is the global search bar.
        The setup steps (clear → edit → fill → Enter) are retried on failure.
        The final results wait is NOT retried to avoid double-submitting searches.
        """
        self._last_search_snapshot = {}
        # Pre-flight: dismiss any stale profile slide-in blocking the sidebar
        await self.go_back_to_results()
        await self.require_recruiter_tab()
        typing_plan = typing_plan or build_boolean_typing_plan(boolean)
        previous_count_text = await self._peek_results_count_text()
        previous_top_card_signature = await self._peek_top_card_signature()

        async def _do():
            # Step 1: Reveal the textarea.
            # The sidebar Keywords section can be in several states:
            #   - Collapsed with "Profile keywords" button (field empty)
            #   - Expanded with textarea visible (field empty or being edited)
            #   - Showing applied keywords with edit/clear buttons
            textarea = self.page.locator('textarea[id*="free-text-single-value-input"]').first

            if not await textarea.is_visible(timeout=1000):
                # Textarea not visible — try to expand the Keywords section.
                # Search the sidebar for any button related to keywords.
                expanded = False
                sidebar_keyword_selectors = [
                    'button[aria-label*="Edit Profile keywords"]',
                    'button[aria-label*="Profile keywords"]',
                    'button[aria-label*="Edit Keywords"]',
                    'button[aria-label*="keywords or boolean" i]',
                    'button:has-text("Profile keywords")',
                ]
                for selector in sidebar_keyword_selectors:
                    try:
                        btn = self.page.locator(selector).first
                        if await btn.is_visible(timeout=1500):
                            await self._ghost_click_locator(btn)
                            if await self._wait_for_visible_poll(
                                textarea,
                                timeout_ms=1200,
                                interval_ms=100,
                            ):
                                expanded = True
                                break
                    except Exception:
                        continue

                if not expanded:
                    # Last resort: dump sidebar DOM for debugging
                    try:
                        title = await self.page.title()
                        url = self.page.url
                        print(f"    [DOM-DEBUG] Page: {title} | URL: {url[:100]}")
                        sidebar = self.page.locator('.keywords-facet, [data-test-facet-keywords]').first
                        sidebar_html = await sidebar.inner_html(timeout=3000)
                        import re as _re
                        buttons = _re.findall(r'<button[^>]*aria-label="([^"]*)"[^>]*>', sidebar_html)
                        print(f"    [DOM-DEBUG] Sidebar buttons: {buttons[:15]}")
                    except Exception:
                        try:
                            all_btns = await self.page.locator('button[aria-label]').all()
                            labels = []
                            for b in all_btns[:20]:
                                try:
                                    labels.append(await b.get_attribute('aria-label', timeout=1000))
                                except Exception:
                                    pass
                            print(f"    [DOM-DEBUG] All page button labels: {labels[:20]}")
                        except Exception as dbg_err:
                            print(f"    [DOM-DEBUG] Could not read page: {dbg_err}")

            # Step 3: Fill the sidebar textarea (the ONLY correct target)
            await textarea.wait_for(state="visible", timeout=10000)
            await self._clear_keyword_textarea(textarea)
            typing_result = await self._input_backend.type_text(
                self.page,
                textarea,
                boolean,
                plan=typing_plan,
            )
            await asyncio.sleep(
                human_delay_correlated(
                    random.uniform(
                        config.LINKEDIN_SEARCH_TYPING_PRE_SUBMIT_MIN_SECONDS,
                        config.LINKEDIN_SEARCH_TYPING_PRE_SUBMIT_MAX_SECONDS,
                    ),
                    channel="search_typing_submit",
                )
            )

            # Step 4: Submit
            await self._press_key("Enter")
            return typing_result

        typing_result = await _retry(_do)

        # Step 5: Wait for results (outside retry — never re-submit on timeout)
        results_wait_ms = await self._wait_for_search_results_ready(
            previous_count_text=previous_count_text,
            previous_top_card_signature=previous_top_card_signature,
        )
        return SearchEntryResult(
            typing_result=typing_result,
            results_wait_ms=results_wait_ms,
        )

    async def apply_location_filter(
        self, values: list[str], *, temporal_scope: str = "any"
    ) -> bool:
        """Apply one or more Location filters to the live Recruiter sidebar (hop-4 canary).

        Mirrors enter_search_string's preflight -> baseline -> retry -> verify discipline
        (NOT the _set_location_filter anti-pattern). Selectors are from the 2026-05-29
        Pass-5 DOM capture (docs/linkedin-recruiter-dom-map.md, Location row): the filter
        pane is `aside.left-rail`; the editor input is the typeahead combobox whose
        placeholder contains "location"; options are `li[role=option]` in the
        `ul.artdeco-typeahead__results-list[role=listbox]`; the applied chip is the
        `li[data-test-facet-pills-item]` whose dismiss control is named "Remove {value}"
        (2026-06 capture; the legacy `[data-test-pill-label]` node is gone). The dismiss
        "X" is hover-gated, so confirmation matches the visible pill item, not the button.

        Fail-closed: any miss (no editor, no exact-match option, no chip) returns False so
        the controller falls back to the keyword Boolean
        (advanced_search.apply_advanced_search_plan); a raise is caught by the caller and
        recorded as a failed_control. The exact-match guard is load-bearing — LinkedIn
        surfaces several "New York" rows, so we click only an option whose text matches the
        requested value exactly, never a broad suggestion, and never press Enter.

        Live-verified 2026-05-29 (tools/hop4_location_smoke.py PASS on a real seat:
        chip applied -> confirmed -> removed, results changed). 2026-07-05 live
        runs exposed a fresh-editor typeahead listener race: typing immediately after
        open can leave the option list empty, and miss residue then cascades into
        appended garbage/stray chips unless every attempt clears and one retry fires.

        temporal_scope: accepted for signature parity with the other apply_* methods
        and as a present-tense routing signal upstream, but it does NOT pick the facet's
        scope dropdown here. Per Sam's product decision the Location facet is always
        applied "Current or past" — sourcing on a geography should reach anyone who has
        been there, never current-residents-only. The arg is ignored for the dropdown
        choice on purpose.
        """
        locations = [str(v).strip() for v in (values or []) if str(v).strip()]
        if not locations:
            return False

        # P3a Stage B: per-call exact-match miss options, keyed by requested
        # value for the orchestrator's one-shot facet resolution.
        self.last_location_option_misses: dict[str, list[str]] = {}
        self.last_location_already_applied_count = 0

        self._last_search_snapshot = {}
        await self.go_back_to_results()
        await self.require_recruiter_tab()
        previous_count_text = await self._peek_results_count_text()
        previous_top_card_signature = await self._peek_top_card_signature()

        rail = self.page.locator("aside.left-rail")
        editor_input = rail.locator(
            'input.artdeco-typeahead__input[placeholder*="location" i]'
        ).first

        applied: list[str] = []
        # Chips already on the sidebar (persisted from a previous session on
        # the same browser page) satisfy their values WITHOUT typing —
        # LinkedIn removes an applied facet from the typeahead's suggestion
        # list, so re-typing an applied value can never find its exact-match
        # option and fails closed after burning the whole retry budget
        # (2026-07-06 SPL-MM live aborts: only city/region rows offered for
        # 'Brazil'/'Colombia' while both country chips sat applied).
        try:
            pre_applied = {
                normalize_facet_value_for_compare(chip)
                for chip in await self.read_applied_location_chips()
            }
        except Exception:
            pre_applied = set()
        typed_any = False
        for value in locations:
            if normalize_facet_value_for_compare(value) in pre_applied:
                log.info(
                    "apply_location_filter[%r]: chip already applied — "
                    "skipping typeahead input.",
                    value,
                )
                applied.append(value)
                self.last_location_already_applied_count += 1
                continue
            typed_any = True

            async def _do(value=value):
                # Reveal the Locations editor if it isn't already open.
                if not await editor_input.is_visible(timeout=800):
                    add_btn = rail.locator(
                        'button:has-text("geographic location")'
                    ).first
                    if not await add_btn.is_visible(timeout=2000):
                        log.warning(
                            "apply_location_filter[%r]: gate=editor_open miss — neither the "
                            "location editor input nor the 'geographic location' add-button "
                            "is visible (selector likely rotated).",
                            value,
                        )
                        return False
                    await self._ghost_click_locator(add_btn)
                    if not await self._wait_for_visible_poll(
                        editor_input, timeout_ms=2500, interval_ms=120
                    ):
                        log.warning(
                            "apply_location_filter[%r]: gate=editor_open miss — add-button "
                            "clicked but the typeahead input never became visible.",
                            value,
                        )
                        return False
                # Click the EXACT-match option only — fail closed otherwise.
                # Do NOT relax the exact-match guard: LinkedIn surfaces broad
                # suggestions and clicking a non-exact row applies the wrong
                # geography.
                exact = re.compile(rf"^\s*{re.escape(value)}\s*$")
                option_list = rail.locator(
                    'ul.artdeco-typeahead__results-list[role="listbox"] '
                    'li[role="option"]'
                )
                option = option_list.filter(has_text=exact).first

                async def _clear_editor_best_effort() -> None:
                    try:
                        if await editor_input.is_visible(timeout=250):
                            await self._clear_keyword_textarea(editor_input)
                    except Exception:
                        pass

                async def _type_and_poll_option() -> bool | None:
                    try:
                        await self._clear_keyword_textarea(editor_input)
                    except RuntimeError as exc:
                        log.warning(
                            "apply_location_filter[%r]: gate=editor_clear miss — %s",
                            value,
                            exc,
                        )
                        await _clear_editor_best_effort()
                        return None
                    await self._input_backend.type_text(
                        self.page, editor_input, value, plan=build_boolean_typing_plan(value)
                    )
                    return await self._wait_for_visible_poll(
                        option, timeout_ms=4000, interval_ms=150
                    )

                option_visible = await _type_and_poll_option()
                if option_visible is None:
                    return False
                # LinkedIn's suggestion ranking is nondeterministic: the same
                # query can surface the country row on one attempt and only
                # city/region rows on the next (2026-07-06 SPL-MM live runs:
                # 'Brazil' offered only 'São Paulo, Brazil'-class rows on both
                # polls of one session after applying cleanly twice earlier
                # the same evening). One retry is too little budget against
                # ranking variance — give the exact-match gate a few more
                # draws, with settle pauses, before failing closed.
                attempt = 1
                while not option_visible and attempt < 4:
                    log.warning(
                        "apply_location_filter[%r]: gate=exact_match retry — no option "
                        "matched ^value$ on attempt %d/4; clearing and retrying.",
                        value,
                        attempt,
                    )
                    attempt += 1
                    await asyncio.sleep(0.8 + 0.4 * attempt)
                    option_visible = await _type_and_poll_option()
                    if option_visible is None:
                        return False
                if not option_visible:
                    # Diagnostic gate 2 (option/exact-match): both attempts
                    # produced no ^value$ option. Capture the option texts
                    # actually offered so a seat run reveals whether this is
                    # selector rotation, an empty list, or a value needing
                    # country-aware normalization (e.g. "Colombia, Colombia").
                    try:
                        offered = await option_list.all_inner_texts()
                    except Exception:
                        offered = []
                    self.last_location_option_misses[value] = [
                        str(text).strip() for text in offered[:12] if str(text).strip()
                    ]
                    log.warning(
                        "apply_location_filter[%r]: gate=exact_match miss — no option "
                        "matched ^value$. Offered options: %r",
                        value,
                        offered[:12],
                    )
                    await _clear_editor_best_effort()
                    return False
                await self._ghost_click_locator(option)
                # Confirm via the visible pill item, not the hover-gated dismiss
                # button (2026-06 capture; no legacy [data-test-pill-label]).
                _safe = value.replace("\\", "\\\\").replace('"', '\\"')
                chip = rail.locator(
                    f"li[data-test-facet-pills-item]:has("
                    f'button[data-test-pill-dismiss][aria-label="Remove {_safe}"])'
                ).first
                landed = await self._wait_for_visible_poll(
                    chip, timeout_ms=2500, interval_ms=120
                )
                if not landed:
                    try:
                        landed_pills = await rail.locator(
                            "li[data-test-facet-pills-item]"
                        ).all_inner_texts()
                    except Exception:
                        landed_pills = []
                    log.warning(
                        "apply_location_filter[%r]: gate=chip miss — exact option clicked "
                        "but no 'Remove %s' pill confirmed. Applied pills: %r",
                        value, value, landed_pills[:12],
                    )
                return landed

            try:
                if await _retry(_do):
                    applied.append(value)
            except Exception as exc:
                log.warning("apply_location_filter: %r did not apply: %s", value, exc)

        # R5 fail-closed: a subset apply must not be reported as full success.
        # Returning True for k of N would make the controller record `locations`
        # as applied and persist the full geography to the recovery snapshot.
        # Fail closed unless every requested value produced a confirmed chip;
        # the controller then falls back to the keyword Boolean.
        if len(applied) < len(locations):
            if applied:
                log.warning(
                    "apply_location_filter: only %d of %d locations applied (%r); "
                    "failing closed so a partial is not reported as full success.",
                    len(applied),
                    len(locations),
                    applied,
                )
            return False
        if not typed_any:
            # Every requested value was already on the sidebar — nothing was
            # typed, so the results cannot have changed; the readiness wait
            # below would poll for a change that never comes.
            return True
        # Scope dropdown is always "Current or past" per Sam's product decision —
        # never current-only. The upstream temporal_scope is a routing signal, not a
        # dropdown selector, so it is intentionally not consulted here. LinkedIn's
        # Location facet defaults to current-or-past, so applying the default already
        # honors the invariant; no dropdown manipulation is required.
        # Verify results changed (SECONDARY gate; chip confirmation above is primary).
        await self._wait_for_search_results_ready(
            previous_count_text=previous_count_text,
            previous_top_card_signature=previous_top_card_signature,
        )
        return True

    async def read_applied_location_chips(self) -> list[str]:
        """Read the applied LOCATION pill values off the live sidebar (P3a invariant).

        Returns the facet VALUES parsed from each pill's dismiss control, whose
        accessible name is "Remove {value}" — the same 2026-06 capture the
        apply path's chip-confirm gate keys on (see apply_location_filter).
        Read-only: never mutates the sidebar and never touches
        _last_search_snapshot.

        SCOPED to the Locations facet section (Codex review, Wave 1): a
        rail-wide pill read would let a company/title pill whose value equals
        the requested geography (e.g. a company literally named "Brazil")
        satisfy the invariant while the Location facet is empty. The section
        is identified by containing the location typeahead input or the
        "geographic location" add-button (docs/linkedin-recruiter-dom-map.md,
        Location row). If the section locator misses (DOM rotation), this
        returns [] and the caller's fail-closed re-assert path makes the
        failure LOUD at the first string, never a silent off-geo run —
        verify the section selector on the next live smoke.
        """
        rail = self.page.locator("aside.left-rail")
        location_section = rail.locator(
            'section:has(input.artdeco-typeahead__input[placeholder*="location" i]), '
            'section:has(button:has-text("geographic location"))'
        ).first
        try:
            if not await location_section.count():
                return []
        except Exception:
            return []
        buttons = location_section.locator(
            "li[data-test-facet-pills-item] button[data-test-pill-dismiss]"
        )
        values: list[str] = []
        count = await buttons.count()
        for i in range(count):
            label = await buttons.nth(i).get_attribute("aria-label")
            if label and label.startswith("Remove "):
                value = label[len("Remove "):].strip()
                if value:
                    values.append(value)
        return values

    async def read_applied_company_chips(self) -> list[str]:
        """Read applied Company pill values from the Companies facet section."""
        rail = self.page.locator("aside.left-rail")
        company_section = rail.locator(
            'section:has(input.artdeco-typeahead__input[placeholder*="company" i]), '
            'section:has(button:has-text("Companies or boolean"))'
        ).first
        try:
            if not await company_section.count():
                return []
        except Exception:
            return []
        buttons = company_section.locator(
            "li[data-test-facet-pills-item] button[data-test-pill-dismiss]"
        )
        values: list[str] = []
        count = await buttons.count()
        for i in range(count):
            label = await buttons.nth(i).get_attribute("aria-label")
            if label and label.startswith("Remove "):
                value = label[len("Remove "):].strip()
                if value:
                    values.append(value)
        return values

    async def apply_company_filter(
        self, values: list[str], *, temporal_scope: str = "any"
    ) -> bool:
        """Apply one or more Company filters to the live Recruiter sidebar.

        Sibling of apply_location_filter — same preflight -> baseline -> retry -> verify
        discipline, same fail-closed posture. Selectors are from the 2026-05-29 Pass-5
        capture (docs/linkedin-recruiter-dom-map.md, Companies row + addendum): the filter
        pane is `aside.left-rail`; the merged "Companies" facet reveals via the
        `Companies or boolean` button (LinkedIn prefixes it with "Add" only when the
        facet has no existing pills); the editor input is the typeahead combobox whose
        placeholder contains "company"; options are `li[role=option]` in
        `ul.artdeco-typeahead__results-list[role=listbox]`; the applied chip is the
        `li[data-test-facet-pills-item]` whose dismiss control is named "Remove {value}"
        (2026-06 capture; the legacy `[data-test-pill-label]` node is gone). The dismiss
        "X" is hover-gated, so confirmation matches the visible pill item, not the button.

        Fail-closed: any miss (no editor, no exact-match option, no chip) returns False so
        the controller falls back to the keyword Boolean. The exact-match guard is
        load-bearing — LinkedIn surfaces near-name company-page suggestions, so we click
        only an option whose text matches the requested value exactly, never a broad
        suggestion, and never press Enter.

        LIVE-VERIFIED. tools/hop4_company_smoke.py passed on a real seat (2026-05-31:
        chip applied -> confirmed -> removed, results narrowed), so 'companies' is in
        advanced_search.STABLE_NOW_CONTROLS and this method drives the live sidebar for a
        hybrid lane carrying a company filter. The graduation gate (passing smoke before
        flipping the constant) is the same one apply_location_filter cleared.

        temporal_scope: the merged Companies facet is always applied "Current or Past"
        per Sam's product decision — never current-only. The arg is a present-tense
        routing signal upstream, not a dropdown selector, and is intentionally not
        consulted here.
        """
        companies = [str(v).strip() for v in (values or []) if str(v).strip()]
        if not companies:
            return False

        # P3a Stage B parity with locations: per-call exact-match miss options,
        # keyed by requested value for live diagnostics.
        self.last_company_option_misses: dict[str, list[str]] = {}
        self.last_company_already_applied_count = 0

        self._last_search_snapshot = {}
        await self.go_back_to_results()
        await self.require_recruiter_tab()
        previous_count_text = await self._peek_results_count_text()
        previous_top_card_signature = await self._peek_top_card_signature()

        rail = self.page.locator("aside.left-rail")
        editor_input = rail.locator(
            'input.artdeco-typeahead__input[placeholder*="company" i]'
        ).first

        applied: list[str] = []
        try:
            pre_applied = {
                normalize_facet_value_for_compare(chip)
                for chip in await self.read_applied_company_chips()
            }
        except Exception:
            pre_applied = set()
        typed_any = False
        for value in companies:
            if normalize_facet_value_for_compare(value) in pre_applied:
                log.info(
                    "apply_company_filter[%r]: chip already applied — "
                    "skipping typeahead input.",
                    value,
                )
                applied.append(value)
                self.last_company_already_applied_count += 1
                continue
            typed_any = True

            async def _do(value=value):
                # Reveal the Companies editor if it isn't already open.
                if not await editor_input.is_visible(timeout=800):
                    add_btn = rail.locator(
                        'button:has-text("Companies or boolean")'
                    ).first
                    if not await add_btn.is_visible(timeout=2000):
                        return False
                    await self._ghost_click_locator(add_btn)
                    if not await self._wait_for_visible_poll(
                        editor_input, timeout_ms=2500, interval_ms=120
                    ):
                        return False
                # Click the EXACT-match option only — fail closed otherwise.
                exact = re.compile(rf"^\s*{re.escape(value)}\s*$")
                option_list = rail.locator(
                    'ul.artdeco-typeahead__results-list[role="listbox"] '
                    'li[role="option"]'
                )
                option = option_list.filter(has_text=exact).first

                async def _clear_editor_best_effort() -> None:
                    try:
                        if await editor_input.is_visible(timeout=250):
                            await self._clear_keyword_textarea(editor_input)
                    except Exception:
                        pass

                async def _type_and_poll_option() -> bool | None:
                    try:
                        await self._clear_keyword_textarea(editor_input)
                    except RuntimeError as exc:
                        log.warning(
                            "apply_company_filter[%r]: gate=editor_clear miss — %s",
                            value,
                            exc,
                        )
                        await _clear_editor_best_effort()
                        return None
                    await self._input_backend.type_text(
                        self.page, editor_input, value, plan=build_boolean_typing_plan(value)
                    )
                    return await self._wait_for_visible_poll(
                        option, timeout_ms=4000, interval_ms=150
                    )

                option_visible = await _type_and_poll_option()
                if option_visible is None:
                    return False
                attempt = 1
                while not option_visible and attempt < 4:
                    log.warning(
                        "apply_company_filter[%r]: gate=exact_match retry — no option "
                        "matched ^value$ on attempt %d/4; clearing and retrying.",
                        value,
                        attempt,
                    )
                    attempt += 1
                    await asyncio.sleep(0.8 + 0.4 * attempt)
                    option_visible = await _type_and_poll_option()
                    if option_visible is None:
                        return False
                if not option_visible:
                    try:
                        offered = await option_list.all_inner_texts()
                    except Exception:
                        offered = []
                    self.last_company_option_misses[value] = [
                        str(text).strip() for text in offered[:12] if str(text).strip()
                    ]
                    log.warning(
                        "apply_company_filter[%r]: gate=exact_match miss — no option "
                        "matched ^value$. Offered options: %r",
                        value,
                        offered[:12],
                    )
                    await _clear_editor_best_effort()
                    return False
                await self._ghost_click_locator(option)
                # Confirm the applied chip landed (PRIMARY success gate). Pill renders as
                # li[data-test-facet-pills-item] with a hover-gated dismiss control whose
                # accessible name is "Remove {value}" (2026-06 capture); confirm the pill
                # ITEM is visible rather than the hover-gated button. The legacy
                # [data-test-pill-label] node no longer exists in the current DOM.
                _safe = value.replace("\\", "\\\\").replace('"', '\\"')
                chip = rail.locator(
                    f"li[data-test-facet-pills-item]:has("
                    f'button[data-test-pill-dismiss][aria-label="Remove {_safe}"])'
                ).first
                landed = await self._wait_for_visible_poll(
                    chip, timeout_ms=2500, interval_ms=120
                )
                if not landed:
                    try:
                        landed_pills = await rail.locator(
                            "li[data-test-facet-pills-item]"
                        ).all_inner_texts()
                    except Exception:
                        landed_pills = []
                    log.warning(
                        "apply_company_filter[%r]: gate=chip miss — exact option clicked "
                        "but no 'Remove %s' pill confirmed. Applied pills: %r",
                        value, value, landed_pills[:12],
                    )
                return landed

            try:
                if await _retry(_do):
                    applied.append(value)
            except Exception as exc:
                log.warning("apply_company_filter: %r did not apply: %s", value, exc)

        # R5 fail-closed (identical to apply_location_filter — its byte-identical
        # sibling): a SUBSET apply must NOT be reported as full success, so a
        # partial company filter is never recorded/replayed as if fully applied.
        # Kept honest now even though 'companies' is still MOCK_ONLY (graduation is
        # L1, live-smoke-gated) so the method is correct the moment it graduates.
        if len(applied) < len(companies):
            if applied:
                log.warning(
                    "apply_company_filter: only %d of %d companies applied (%r); "
                    "failing closed so a partial is not reported as full success.",
                    len(applied),
                    len(companies),
                    applied,
                )
            return False
        if not typed_any:
            return True
        # Scope dropdown is always "Current or past" per Sam's product decision —
        # never current-only. The upstream temporal_scope is a present-tense routing
        # signal, not a dropdown selector, so it is intentionally not consulted here.
        # The merged Companies facet defaults to "Current or Past", so the default
        # already honors the invariant.
        # Verify results changed (SECONDARY gate; chip confirmation above is primary).
        await self._wait_for_search_results_ready(
            previous_count_text=previous_count_text,
            previous_top_card_signature=previous_top_card_signature,
        )
        return True

    async def read_applied_title_chips(self) -> list[str]:
        """Read applied Job Title pill values from the Job titles facet section."""
        rail = self.page.locator("aside.left-rail")
        title_section = rail.locator(
            'section:has(input.artdeco-typeahead__input[placeholder*="job title" i]), '
            'section:has(button:has-text("Job titles or boolean"))'
        ).first
        try:
            if not await title_section.count():
                return []
        except Exception:
            return []
        buttons = title_section.locator(
            "li[data-test-facet-pills-item] button[data-test-pill-dismiss]"
        )
        values: list[str] = []
        count = await buttons.count()
        for i in range(count):
            label = await buttons.nth(i).get_attribute("aria-label")
            if label and label.startswith("Remove "):
                value = label[len("Remove "):].strip()
                if value:
                    values.append(value)
        return values

    async def apply_title_filter(
        self, values: list[str], *, temporal_scope: str = "any"
    ) -> bool:
        """Apply one or more Job Title filters to the live Recruiter sidebar.

        Sibling of apply_company_filter / apply_location_filter. Selectors from
        the dev bundle (linkedin-recruiter-selectors.yaml, job_titles entry) and
        the 2026-05-29 Pass-5 capture. LIVE-VERIFIED: tools/hop4_title_smoke.py passed
        on a real seat (2026-05-31: chip applied -> confirmed -> removed), so
        'job_titles' is in advanced_search.STABLE_NOW_CONTROLS and this method drives
        the live sidebar for a hybrid lane carrying a title filter.

        The reveal button is `Job titles or boolean` (stable fragment; LinkedIn
        prefixes with "Add" only when the facet has no existing pills — same
        state-variance as Companies). Editor input placeholder contains
        "job title". Options, chips, and remove follow the shared typeahead
        architecture.

        Scope: the Job titles facet is always applied "Current or Past" per Sam's
        product decision — never current-only. The temporal_scope arg is a present-tense
        routing signal upstream, not a dropdown selector, and is intentionally not
        consulted here. (The scope dropdown selector is recorded in the DOM map under
        Job titles should that decision ever be revisited.)

        Fail-closed: any miss (no editor, no exact-match option, no chip)
        returns False so the controller falls back to keyword.
        """
        titles = [str(v).strip() for v in (values or []) if str(v).strip()]
        if not titles:
            return False

        self.last_title_option_misses: dict[str, list[str]] = {}
        self.last_title_already_applied_count = 0

        self._last_search_snapshot = {}
        await self.go_back_to_results()
        await self.require_recruiter_tab()
        previous_count_text = await self._peek_results_count_text()
        previous_top_card_signature = await self._peek_top_card_signature()

        rail = self.page.locator("aside.left-rail")
        editor_input = rail.locator(
            'input.artdeco-typeahead__input[placeholder*="job title" i]'
        ).first

        applied: list[str] = []
        try:
            pre_applied = {
                normalize_facet_value_for_compare(chip)
                for chip in await self.read_applied_title_chips()
            }
        except Exception:
            pre_applied = set()
        typed_any = False
        for value in titles:
            if normalize_facet_value_for_compare(value) in pre_applied:
                log.info(
                    "apply_title_filter[%r]: chip already applied — "
                    "skipping typeahead input.",
                    value,
                )
                applied.append(value)
                self.last_title_already_applied_count += 1
                continue
            typed_any = True

            async def _do(value=value):
                if not await editor_input.is_visible(timeout=800):
                    add_btn = rail.locator(
                        'button:has-text("Job titles or boolean")'
                    ).first
                    if not await add_btn.is_visible(timeout=2000):
                        return False
                    await self._ghost_click_locator(add_btn)
                    if not await self._wait_for_visible_poll(
                        editor_input, timeout_ms=2500, interval_ms=120
                    ):
                        return False

                exact = re.compile(rf"^\s*{re.escape(value)}\s*$")
                option_list = rail.locator(
                    'ul.artdeco-typeahead__results-list[role="listbox"] '
                    'li[role="option"]'
                )
                option = option_list.filter(has_text=exact).first

                async def _clear_editor_best_effort() -> None:
                    try:
                        if await editor_input.is_visible(timeout=250):
                            await self._clear_keyword_textarea(editor_input)
                    except Exception:
                        pass

                async def _type_and_poll_option() -> bool | None:
                    try:
                        await self._clear_keyword_textarea(editor_input)
                    except RuntimeError as exc:
                        log.warning(
                            "apply_title_filter[%r]: gate=editor_clear miss — %s",
                            value,
                            exc,
                        )
                        await _clear_editor_best_effort()
                        return None
                    await self._input_backend.type_text(
                        self.page, editor_input, value, plan=build_boolean_typing_plan(value)
                    )
                    return await self._wait_for_visible_poll(
                        option, timeout_ms=4000, interval_ms=150
                    )

                option_visible = await _type_and_poll_option()
                if option_visible is None:
                    return False
                attempt = 1
                while not option_visible and attempt < 4:
                    log.warning(
                        "apply_title_filter[%r]: gate=exact_match retry — no option "
                        "matched ^value$ on attempt %d/4; clearing and retrying.",
                        value,
                        attempt,
                    )
                    attempt += 1
                    await asyncio.sleep(0.8 + 0.4 * attempt)
                    option_visible = await _type_and_poll_option()
                    if option_visible is None:
                        return False
                if not option_visible:
                    try:
                        offered = await option_list.all_inner_texts()
                    except Exception:
                        offered = []
                    self.last_title_option_misses[value] = [
                        str(text).strip() for text in offered[:12] if str(text).strip()
                    ]
                    log.warning(
                        "apply_title_filter[%r]: gate=exact_match miss — no option "
                        "matched ^value$. Offered options: %r",
                        value,
                        offered[:12],
                    )
                    await _clear_editor_best_effort()
                    return False
                await self._ghost_click_locator(option)
                # Confirm the applied chip landed (PRIMARY success gate). Pill renders as
                # li[data-test-facet-pills-item] with a hover-gated dismiss control whose
                # accessible name is "Remove {value}" (2026-06 capture); confirm the pill
                # ITEM is visible rather than the hover-gated button. The legacy
                # [data-test-pill-label] node no longer exists in the current DOM.
                _safe = value.replace("\\", "\\\\").replace('"', '\\"')
                chip = rail.locator(
                    f"li[data-test-facet-pills-item]:has("
                    f'button[data-test-pill-dismiss][aria-label="Remove {_safe}"])'
                ).first
                landed = await self._wait_for_visible_poll(
                    chip, timeout_ms=2500, interval_ms=120
                )
                if not landed:
                    try:
                        landed_pills = await rail.locator(
                            "li[data-test-facet-pills-item]"
                        ).all_inner_texts()
                    except Exception:
                        landed_pills = []
                    log.warning(
                        "apply_title_filter[%r]: gate=chip miss — exact option clicked "
                        "but no 'Remove %s' pill confirmed. Applied pills: %r",
                        value, value, landed_pills[:12],
                    )
                return landed

            try:
                if await _retry(_do):
                    applied.append(value)
            except Exception as exc:
                log.warning("apply_title_filter: %r did not apply: %s", value, exc)

        if len(applied) < len(titles):
            if applied:
                log.warning(
                    "apply_title_filter: only %d of %d titles applied (%r); "
                    "failing closed so a partial is not reported as full success.",
                    len(applied),
                    len(titles),
                    applied,
                )
            return False
        if not typed_any:
            return True
        # Scope dropdown is always "Current or past" per Sam's product decision —
        # never current-only. The upstream temporal_scope is a present-tense routing
        # signal, not a dropdown selector, so it is intentionally not consulted here.
        # The Job titles facet defaults to "Current or Past", so the default already
        # honors the invariant.
        await self._wait_for_search_results_ready(
            previous_count_text=previous_count_text,
            previous_top_card_signature=previous_top_card_signature,
        )
        return True

    async def get_result_count(self) -> int:
        """Parse 'N RESULTS' from the page. Alias for get_results_count."""
        return await self.get_results_count()

    # ------------------------------------------------------------------
    # Filter application
    # ------------------------------------------------------------------

    async def apply_permanent_filters(self, filters: dict) -> None:
        """Set Location, Seniority, etc. from the brief's permanent_filters.

        This is complex UI automation — implements Location, stubs the rest.
        """
        print(f"  Applying permanent filters...")

        # Location filter
        location = filters.get("Location") or filters.get("location")
        if location:
            await self._set_location_filter(location)

        # Seniority filter
        seniority = filters.get("seniority")
        if seniority:
            log.warning(
                "LinkedIn automation does not apply seniority from brief (%s)",
                seniority,
            )

        # Years of experience
        years_exp = filters.get("years_experience")
        if years_exp:
            log.warning(
                "LinkedIn automation does not apply years_experience from brief (%s)",
                years_exp,
            )

        # Log any unhandled filters
        handled = {"Location", "location", "seniority", "years_experience",
                    "company_filters", "keywords", "seniority_excluded"}
        for key in filters:
            if key not in handled:
                log.warning(
                    "LinkedIn automation does not apply permanent filter %s=%s",
                    key,
                    filters[key],
                )

    async def _set_location_filter(self, location: str) -> None:
        """Set the Location filter in LinkedIn Recruiter sidebar."""
        try:
            # Find the Location input — typically a typeahead
            loc_input = self.page.locator(
                'input[aria-label*="location" i], input[placeholder*="location" i]'
            ).first
            await loc_input.wait_for(state="visible", timeout=5000)
            # Away-mode contract: every interaction routes through the input
            # backend — a raw locator click()/fill() emits synthetic CDP input
            # even when the session runs in OS-level HID mode.
            await self._ghost_click_locator(loc_input)
            await self._input_backend.type_text(
                self.page, loc_input, location, plan=build_boolean_typing_plan(location)
            )
            await self.page.wait_for_timeout(1500)

            # Select from typeahead dropdown
            option = self.page.locator(
                f'[role="option"]:has-text("{location}"), '
                f'li:has-text("{location}")'
            ).first
            await option.wait_for(state="visible", timeout=5000)
            await self._ghost_click_locator(option)
            await self.page.wait_for_timeout(2000)
            print(f"  Location filter set: {location}")
        except Exception as e:
            print(f"  [warn] Location filter failed: {e} — set manually before running")

    # ------------------------------------------------------------------
    # Results: innerText, count, paginate
    # ------------------------------------------------------------------

    async def get_results_count(self) -> int:
        """Parse "N RESULTS" text from the page. Returns int (-1 if unparseable)."""
        raw = await self.get_results_count_text()
        if not raw:
            return -1
        # Strip suffix like "K+", "M+" and parse
        clean = raw.replace(",", "").strip()
        if clean.upper().endswith("K+"):
            try:
                return int(float(clean[:-2]) * 1000)
            except ValueError:
                return -1
        if clean.upper().endswith("M+"):
            try:
                return int(float(clean[:-2]) * 1_000_000)
            except ValueError:
                return -1
        try:
            return int(clean)
        except ValueError:
            return -1

    async def get_results_count_text(self) -> str:
        """Read the raw result count text from .search-query-summary__title (e.g. '1.2K+ results').

        Waits briefly for LinkedIn to update the result count in the DOM
        after a search is executed.
        """
        for _ in range(5):
            try:
                el = self.page.locator(".search-query-summary__title").first
                text = (await el.inner_text(timeout=3000)).strip()
                if text:
                    # Extract the count portion: "1,234 results" → "1,234", "1.2K+ results" → "1.2K+"
                    match = re.search(r"([\d.,]+[KkMm]?\+?)", text)
                    if match:
                        return match.group(1)
            except Exception:
                pass
            await self.page.wait_for_timeout(1000)
        return ""

    async def scroll_to_load_all_results(self) -> int:
        """Scroll the results list to trigger LinkedIn's lazy loading.

        Scrolls in human-like increments, counting rendered article cards
        after each step. Stops when no new cards appear for 3 consecutive
        scrolls (the page bottom has been reached).
        """
        cards_selector = "ol.profile-list article.profile-list-item"

        if await self.page.locator("ol.profile-list").count() == 0:
            return 0

        MAX_SCROLLS = 50        # safety cap (50×400 = 20,000px, well beyond 26 cards)
        STABLE_ROUNDS = 3       # stop after 3 scrolls with no new cards

        total_scrolled = 0
        prev_count = await self.page.locator(cards_selector).count()
        stable = 0

        for scroll_num in range(1, MAX_SCROLLS + 1):
            if stable >= STABLE_ROUNDS:
                break

            step = random.randint(260, 560)
            await self._human_scroll(step, channel="results_list")
            total_scrolled += step

            # Occasional scan pause while reviewing newly exposed cards.
            if scroll_num % random.randint(4, 6) == 0:
                await asyncio.sleep(human_delay_correlated(0.5, channel="results_list_scan"))

            # Let DOM update after scroll
            await self.page.wait_for_timeout(300)

            current_count = await self.page.locator(cards_selector).count()
            if current_count > prev_count:
                prev_count = current_count
                stable = 0
                if random.random() < 0.35:
                    await asyncio.sleep(
                        human_delay_correlated(
                            random.uniform(0.5, 1.4),
                            channel="results_list_scan",
                        )
                    )
            else:
                stable += 1
                if random.random() < 0.25:
                    await asyncio.sleep(
                        human_delay_correlated(
                            random.uniform(0.15, 0.45),
                            channel="results_list_scan",
                        )
                    )

        # Final wait for remaining renders
        await self.page.wait_for_timeout(800)

        # Scroll back to top
        if total_scrolled > 0:
            await self._human_scroll(-total_scrolled, channel="results_list_return")
        await asyncio.sleep(human_delay_correlated(0.3, channel="results_list_return"))

        return await self.page.locator(cards_selector).count()

    async def get_card_name_url_pairs(self) -> list[dict]:
        """Extract name + profile URL atomically from each rendered article card.

        Returns a list of {"name": str, "url": str} dicts in DOM order.
        Pairs are extracted per-card so name and URL always correspond.
        """
        async def _do():
            cards = self.page.locator("ol.profile-list article.profile-list-item")
            count = await cards.count()
            pairs = []
            for i in range(count):
                card = cards.nth(i)
                name = ""
                url = ""
                try:
                    name_el = card.locator('[class*="lockup__title"] a').first
                    name = (await name_el.inner_text(timeout=2000)).strip()
                    url = (await name_el.get_attribute("href")) or ""
                except Exception:
                    # Try the profile link directly if lockup title fails
                    try:
                        link = card.locator('a[href*="/talent/profile/"]').first
                        url = (await link.get_attribute("href")) or ""
                        name = (await link.inner_text(timeout=2000)).strip()
                    except Exception:
                        pass
                if name or url:
                    pairs.append({"name": name, "url": url})
            return pairs
        return await _retry(_do)

    async def get_results_list_innertext(self) -> str:
        """Get innerText of result list items that have rendered article cards."""
        if await self.page.locator("ol.profile-list").count() == 0:
            return ""
        async def _do():
            # Only include <li> elements that have a rendered <article> child
            rendered_li = self.page.locator(
                "ol.profile-list > li:has(article.profile-list-item)"
            )
            count = await rendered_li.count()
            if count == 0:
                return ""
            texts = []
            for i in range(count):
                texts.append(await rendered_li.nth(i).inner_text(timeout=5000))
            return "\n".join(texts)
        return await _retry(_do)

    async def get_card_innertext(self, card_index: int) -> str:
        """Get innerText of a specific candidate card by 0-based index."""
        async def _do():
            cards = self.page.locator("ol.profile-list article.profile-list-item")
            count = await cards.count()
            if card_index >= count:
                raise IndexError(f"Card index {card_index} out of range ({count} cards)")
            return await cards.nth(card_index).inner_text(timeout=10000)
        return await _retry(_do)

    async def get_card_saved_status(self) -> list[bool]:
        """Check each rendered card for already-saved indicators.

        Saved candidates show 'Change stage' button instead of
        'Save to pipeline'. Returns list of bools in DOM order.
        """
        async def _do():
            cards = self.page.locator("ol.profile-list article.profile-list-item")
            count = await cards.count()
            statuses = []
            for i in range(count):
                card = cards.nth(i)
                try:
                    change_btn = card.locator(
                        ':is(button:has-text("Change stage"), '
                        'button[data-test-change-stage-button])'
                    ).first
                    is_saved = await change_btn.is_visible(timeout=500)
                except Exception:
                    is_saved = False
                statuses.append(is_saved)
            return statuses
        return await _retry(_do)

    async def get_card_slot_count(self) -> int:
        """Count result slots, including offscreen virtualized cards."""
        slots = self.page.locator("ol.profile-list > li")
        return await slots.count()

    @staticmethod
    def _profile_url_fragment(profile_url: str) -> str:
        """Return the exact Recruiter profile identity segment from a URL."""
        raw = str(profile_url or "").strip()
        match = re.search(
            r"/(?:talent/profile|recruiterSearch/profile)/([^/?#]+)",
            raw,
        )
        return match.group(1) if match else ""

    def current_profile_identity_fragment(self) -> str:
        """Return the Recruiter identity displayed in the slide-in, if any."""
        return self._profile_url_fragment(self.page.url)

    async def find_result_slot_by_profile_url(self, profile_url: str) -> int | None:
        """Find a rendered result slot by exact Recruiter profile identity."""
        return await self._find_result_slot_by_fragment(
            self._profile_url_fragment(profile_url)
        )

    async def _find_result_slot_by_fragment(self, expected: str) -> int | None:
        """Find a rendered result slot by an already-extracted identity fragment.

        Split out from `find_result_slot_by_profile_url` so the Wave G card save
        can scope by the identity the slide-in URL already carries, without
        having to reconstruct a profile URL just to have it re-parsed.
        """
        if not expected:
            return None

        for scan_index in range(MAX_RETRIES):
            slot_count = await self.get_card_slot_count()
            for card_index in range(slot_count):
                try:
                    li = await self._wait_for_result_slot(card_index)
                    await li.scroll_into_view_if_needed(timeout=3000)
                    await self.page.wait_for_timeout(250)
                    links = li.locator('a[href*="/talent/profile/"]')
                    for link_index in range(await links.count()):
                        href = (
                            await links.nth(link_index).get_attribute("href")
                        ) or ""
                        if self._profile_url_fragment(href) == expected:
                            return card_index
                except Exception as error:
                    if _is_target_crash_error(error):
                        raise
            if scan_index < MAX_RETRIES - 1:
                await self.page.wait_for_timeout(250)
        return None

    async def _wait_for_result_slot(self, card_index: int, timeout_ms: int = 8000):
        """Wait for a specific results-list slot to exist after DOM re-hydration.

        Recruiter occasionally tears down and rebuilds the results list while the
        slide-in closes or while the viewport is being repositioned. During that
        window, `nth(card_index)` can temporarily stop resolving even though the
        page is still healthy. We wait for the slot count to recover instead of
        failing the whole string on a transient list rebuild.
        """
        list_root = self.page.locator("ol.profile-list").first
        await list_root.wait_for(state="visible", timeout=timeout_ms)

        slots = self.page.locator("ol.profile-list > li")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (timeout_ms / 1000.0)
        last_count = 0
        last_error: Exception | None = None

        while loop.time() < deadline:
            try:
                count = await slots.count()
                last_count = count
                if count > card_index:
                    li = slots.nth(card_index)
                    await li.wait_for(state="attached", timeout=1000)
                    return li
            except Exception as e:
                last_error = e

            await self.page.wait_for_timeout(200)

        details = f"Result slot {card_index + 1} unavailable after waiting (saw {last_count} slots)"
        if last_error:
            details = f"{details}: {last_error}"
        raise TimeoutError(details)

    async def get_card_count(self) -> int:
        cards = self.page.locator("ol.profile-list article.profile-list-item")
        return await cards.count()

    async def focus_card_for_review(self, card_index: int) -> None:
        """Bring a result card into a readable viewport position with visible scrolling."""
        started = time.monotonic()
        succeeded = False
        try:
            li = await self._wait_for_result_slot(card_index)
            try:
                rect = await li.evaluate("""el => {
                    const r = el.getBoundingClientRect();
                    return { top: r.top, bottom: r.bottom, height: r.height };
                }""")
                viewport_h = await self.page.evaluate("() => window.innerHeight")
                target_top = random.randint(
                    130, max(180, min(320, int(viewport_h * 0.35)))
                )

                if rect:
                    delta = rect["top"] - target_top
                    if abs(delta) > 40:
                        await self._human_scroll(
                            int(delta), channel="results_review"
                        )
                        if random.random() < 0.25:
                            correction = random.randint(12, 36)
                            await self._human_scroll(
                                -correction if delta > 0 else correction,
                                channel="results_review",
                            )
                await self.page.wait_for_timeout(450)
                article = li.locator("article.profile-list-item").first
                await article.wait_for(state="visible", timeout=3000)
            except Exception:
                li = await self._wait_for_result_slot(card_index)
                await li.scroll_into_view_if_needed(timeout=3000)
                await self.page.wait_for_timeout(600)
                article = li.locator("article.profile-list-item").first
                await article.wait_for(state="visible", timeout=4000)
            succeeded = True
        finally:
            emit_timing_event(
                self._timing_recorder,
                "card_focus_timing",
                elapsed_ms=round((time.monotonic() - started) * 1000.0, 3),
                card_index=card_index,
                succeeded=succeeded,
            )

    async def get_card_snapshot(self, card_index: int) -> dict:
        """Read one result card's rendered text plus stable DOM metadata."""
        async def _do():
            li = await self._wait_for_result_slot(card_index)
            article = li.locator("article.profile-list-item").first
            await article.wait_for(state="visible", timeout=5000)

            innertext = await article.inner_text(timeout=10000)

            name = ""
            url = ""
            try:
                name_el = article.locator('[class*="lockup__title"] a').first
                name = (await name_el.inner_text(timeout=2000)).strip()
                url = (await name_el.get_attribute("href")) or ""
            except Exception as exc:
                if _is_target_crash_error(exc):
                    raise
                try:
                    link = article.locator('a[href*="/talent/profile/"]').first
                    url = (await link.get_attribute("href")) or ""
                    name = (await link.inner_text(timeout=2000)).strip()
                except Exception as exc:
                    if _is_target_crash_error(exc):
                        raise
                    pass

            already_saved = False
            try:
                change_btn = article.locator(
                    ':is(button:has-text("Change stage"), button[data-test-change-stage-button])'
                ).first
                already_saved = await change_btn.is_visible(timeout=500)
            except Exception as exc:
                if _is_target_crash_error(exc):
                    raise
                already_saved = False

            recruiter_activity = extract_recruiter_activity_from_card_text(innertext)

            return {
                "innertext": innertext,
                "name": name,
                "url": url,
                "already_saved": already_saved,
                "recruiter_activity": recruiter_activity.to_dict(),
            }

        started = time.monotonic()
        snapshot: dict = {}
        succeeded = False
        try:
            snapshot = await _retry(_do)
            succeeded = True
            return snapshot
        finally:
            emit_timing_event(
                self._timing_recorder,
                "card_snapshot_timing",
                elapsed_ms=round((time.monotonic() - started) * 1000.0, 3),
                card_index=card_index,
                text_chars=len(str(snapshot.get("innertext") or "")),
                succeeded=succeeded,
            )

    async def go_to_next_page(self) -> bool:
        """Click the bottom pagination link to advance to the next results page.

        Primary: a.pagination__quick-link--next (bottom of results list)
        Fallback: button[aria-label="Next"] (non-project search pages)
        Verifies the page actually advanced by comparing the first candidate name before/after.
        """
        try:
            # Read first candidate name BEFORE clicking
            first_name_before = ""
            try:
                first_link = self.page.locator('ol.profile-list article.profile-list-item [class*="lockup__title"] a').first
                first_name_before = (await first_link.inner_text(timeout=3000)).strip()
            except Exception:
                pass

            # Try the bottom pagination link first
            next_link = self.page.locator('a.pagination__quick-link--next').first
            try:
                visible = await next_link.is_visible(timeout=3000)
            except Exception:
                visible = False

            if visible:
                await next_link.scroll_into_view_if_needed()
                await self.page.wait_for_timeout(500)
                if not await self._ghost_click('a.pagination__quick-link--next'):
                    await next_link.click()
            else:
                # Fallback: button[aria-label="Next"] for non-project pages
                next_btn = self.page.locator(
                    'button[aria-label="Next"]:not(.skyline-pagination-button)'
                ).first
                if not await next_btn.is_enabled(timeout=3000):
                    return False
                if not await self._ghost_click('button[aria-label="Next"]:not(.skyline-pagination-button)'):
                    await next_btn.click()

            await self.page.wait_for_timeout(3000)

            # Verify the page actually advanced by checking the first candidate name
            if first_name_before:
                try:
                    first_link = self.page.locator('ol.profile-list article.profile-list-item [class*="lockup__title"] a').first
                    first_name_after = (await first_link.inner_text(timeout=3000)).strip()
                    if first_name_after == first_name_before:
                        await self.page.wait_for_timeout(2000)
                        first_name_after = (await first_link.inner_text(timeout=3000)).strip()
                        if first_name_after == first_name_before:
                            print("    [warn] Pagination click did not advance — first candidate unchanged")
                            return False
                except Exception:
                    pass  # Can't verify, assume it worked

            return True
        except Exception:
            return False

    async def go_to_previous_page(self) -> bool:
        async def _do():
            prev_btn = self.page.locator('button[aria-label="Previous"]').first
            if await prev_btn.is_enabled():
                await prev_btn.click()
                await self.page.wait_for_timeout(3000)
                return True
            return False
        try:
            return await _retry(_do)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Profile: slide-in panel (NOT page navigation)
    # ------------------------------------------------------------------

    async def open_profile(self, candidate_name: str) -> None:
        """Click candidate name link to open the slide-in profile panel.

        Tries multiple strategies: Playwright text match, then JS click by name substring.

        P8.1: governed at the source — checks the governor before attempting
        the open and records the open only after it actually succeeds, so
        every caller (acquisition, identity resolver, reconciliation) is
        accounted for without needing its own enforcement.
        """
        self._governor.check_profile_open_or_raise()

        async def _do():
            # Strategy 1: Ghost-cursor with Playwright text match
            selector = f'ol.profile-list article.profile-list-item a:has-text("{candidate_name}")'
            try:
                name_link = self.page.locator(selector).first
                if await name_link.count() > 0:
                    await name_link.wait_for(state="visible", timeout=5000)
                    if not await self._ghost_click(selector):
                        await name_link.evaluate("el => el.click()")
                    await self.page.locator("div.profile-slidein__container").wait_for(
                        state="visible", timeout=10000
                    )
                    await self.page.wait_for_timeout(1500)
                    return
            except Exception:
                pass

            # Strategy 2: JS click — find link by name substring (handles special chars)
            escaped_name = candidate_name.replace("'", "\\'").replace('"', '\\"')
            clicked = await self.page.evaluate(f"""() => {{
                const links = document.querySelectorAll('ol.profile-list article.profile-list-item a');
                // Strip credential suffixes (PhD, MBA, M.Sc., etc.) and trailing punctuation
                let target = '{escaped_name}'.replace(/,?\\s*(PhD|Ph\\.?D|MBA|M\\.?Sc\\.?|M\\.?S\\.?|Dr\\.?|CFA|PMP|PE)\\s*$/gi, '').trim().toLowerCase();
                for (const link of links) {{
                    const text = link.textContent.trim().toLowerCase();
                    if (text.includes(target) || target.includes(text)) {{
                        link.click();
                        return true;
                    }}
                }}
                // Fuzzy: try matching just the first name + last name (ignore middle/suffix)
                const parts = target.split(/\\s+/);
                if (parts.length >= 2) {{
                    const first = parts[0];
                    // Find last part that's not a single letter (initial)
                    let last = parts[parts.length - 1];
                    for (let i = parts.length - 1; i >= 1; i--) {{
                        if (parts[i].length > 1) {{ last = parts[i]; break; }}
                    }}
                    for (const link of links) {{
                        const text = link.textContent.trim().toLowerCase();
                        if (text.includes(first) && text.includes(last)) {{
                            link.click();
                            return true;
                        }}
                    }}
                }}
                return false;
            }}""")
            if not clicked:
                raise Exception(f"Could not find profile link for '{candidate_name}'")
            await self.page.locator("div.profile-slidein__container").wait_for(
                state="visible", timeout=10000
            )
            await self.page.wait_for_timeout(1500)
        await _retry(_do)
        self._governor.record_profile_open()

    async def ensure_card_rendered(self, card_index: int) -> None:
        """Scroll the Nth <li> in the results list into view to force article rendering.

        LinkedIn uses virtual scrolling — <li> containers are always in the DOM
        but <article> elements inside only render when the <li> is visible.
        Call this before open_profile_by_url/open_profile to guarantee the
        target card's links exist in the DOM.
        """
        try:
            li = await self._wait_for_result_slot(card_index)
            await li.scroll_into_view_if_needed(timeout=3000)
            await self.page.wait_for_timeout(600)  # Let article render
        except Exception as e:
            # E3 class. This runs on the save path (`_reopen_profile_for_full_eval_save`
            # calls it before re-opening the panel to save), so a renderer crash
            # swallowed into a warning becomes an unrendered card, which then
            # reads as the benign "candidate not on this page" refusal — the
            # crash disappears and the run keeps going against a dead target.
            if _is_target_crash_error(e):
                raise
            print(f"    [warn] ensure_card_rendered({card_index}) failed: {e}")

    async def open_profile_by_url(self, profile_url: str) -> None:
        """Open a profile by matching the name link href in the results list.

        Uses JS click to bypass overlay issues. Falls back to Playwright locator.
        Caller should call ensure_card_rendered() first to guarantee the article is in the DOM.

        P8.1: governed at the source — see open_profile().
        """
        started = time.monotonic()
        succeeded = False

        async def _do():
            url_fragment = self._profile_url_fragment(profile_url)
            if not url_fragment:
                raise ValueError("Profile URL has no identity fragment")

            links = self.page.locator('ol.profile-list a[href*="/talent/profile/"]')
            for link_index in range(await links.count()):
                link = links.nth(link_index)
                href = (await link.get_attribute("href")) or ""
                if self._profile_url_fragment(href) != url_fragment:
                    continue
                try:
                    await self._ghost_click_locator(link)
                except Exception:
                    await link.evaluate("el => el.click()")
                break
            else:
                raise RuntimeError(
                    f"Could not find exact profile link for URL fragment '{url_fragment}'"
                )

            await self.page.locator("div.profile-slidein__container").wait_for(
                state="visible", timeout=10000
            )
            await self.page.wait_for_timeout(1500)
        try:
            self._governor.check_profile_open_or_raise()
            await _retry(_do)
            self._governor.record_profile_open()
            succeeded = True
        finally:
            emit_timing_event(
                self._timing_recorder,
                "profile_open_timing",
                elapsed_ms=round((time.monotonic() - started) * 1000.0, 3),
                succeeded=succeeded,
            )

    # ── Profile reading patterns ────────────────────────────────────

    _CHUNK_SIZE = 400  # ~4-5 card heights

    _PATTERN_WEIGHTS = {
        "short":  [0.55, 0.05, 0.30, 0.10],
        "medium": [0.40, 0.25, 0.20, 0.15],
        "long":   [0.25, 0.30, 0.15, 0.30],
    }
    _PATTERN_NAMES = ["focused_reader", "skipper", "skimmer", "section_hopper"]

    # Base dwell values. Lowered 2026-07-05 (SPL live run): the prior
    # 1.5-4.0 base + 2-4x careful multiplier through an UNCAPPED log-normal
    # produced 60s+ single-section stares and 3-minute profile reads — not
    # human, and slow. Median pulled down here; the caps below bound the tail.
    _BASE_SHORT_DWELL = 3.0
    _BASE_CHUNK_DWELL_LOW = 1.0
    _BASE_CHUNK_DWELL_HIGH = 2.5
    # A single section dwell never exceeds this (humans don't stare at one
    # profile chunk longer), and the whole-profile lingering is budgeted so a
    # long profile can't compound past this total.
    _MAX_CHUNK_DWELL = 8.0
    _MAX_PROFILE_READ = 35.0
    _SECTION_DIRECTED_NAMES = frozenset({"about", "experience", "education"})

    # Section-directed read dwell budget. Measured from 54 real section-directed
    # reads in the live AIEL campaign (profile_read_timing events, run log
    # 3000000003-aiel-20260729): at the fixed medium interest of 0.5 the prior
    # formula budgeted 22.5s of dwell and the observed wall-clock was min 19.6s /
    # median 29.4s / max 42.0s. The ~7s gap between budget and median is scroll
    # wall-clock, which sits on top of the dwell budget and is not part of it.
    # The target is 12-20s of total visible panel time.
    #
    # Under these coefficients: interest 0.35 budgets 7.2s of dwell (~14s wall),
    # 0.5 budgets 9.0s (~16s wall, mid-band), and 0.9 budgets 13.8s (~21s wall).
    # The high end deliberately runs past the band — a recruiter dwelling longer
    # on an interesting profile is the behavior being modeled, and the variance
    # across profiles is the point of the interest hint.
    @staticmethod
    def _interest_read_budget_seconds(interest: float) -> float:
        """Map interest 0..1 to a visible-read budget in seconds (pure, no I/O)."""
        clamped = min(max(float(interest), 0.0), 1.0)
        return min(3.0 + clamped * 12.0, LinkedInBrowser._MAX_PROFILE_READ)

    @staticmethod
    def _clamp_profile_read_dwell(
        dwell: float,
        elapsed: float,
        *,
        max_chunk: float,
        max_total: float,
    ) -> float:
        """Clamp a profile-read dwell to the per-section cap and the remaining
        per-profile budget. Pure (no I/O) so the timing policy is unit-testable
        without a browser. Preserves human variance BELOW the caps; only the
        pathological log-normal tail and its compounding are removed.
        """
        d = min(max(dwell, 0.0), max_chunk)
        remaining = max_total - elapsed
        if remaining <= 0.0:
            return 0.0
        return min(d, remaining)

    async def _profile_read_dwell(self, dwell: float) -> None:
        """Sleep for a profile-read dwell, clamped per-section and per-profile.
        Every ``channel="profile_read"`` dwell routes through here so the tail
        can't run away (2026-07-05). ``_profile_read_elapsed`` is reset at the
        top of ``simulate_profile_read``.
        """
        elapsed = getattr(self, "_profile_read_elapsed", 0.0)
        d = self._clamp_profile_read_dwell(
            dwell,
            elapsed,
            max_chunk=self._MAX_CHUNK_DWELL,
            max_total=self._MAX_PROFILE_READ,
        )
        self._profile_read_elapsed = elapsed + d
        if (
            config.LINKEDIN_CADENCE_READ_FIX_ENABLED
            and elapsed >= self._MAX_PROFILE_READ
        ):
            raise _ProfileReadBudgetExhausted()
        if d > 0.0:
            await asyncio.sleep(d)

    async def simulate_profile_read(self, interest: float = 0.5) -> None:
        """Scroll through profile panel to simulate a recruiter reading before extraction.

        Selects from 4 reading patterns (focused, skipper, skimmer, hopper)
        weighted by profile length. Produces realistic, varied scroll behavior.
        When ``LINKEDIN_SECTION_DIRECTED_READ_ENABLED``, hops by profile section
        with dwell budget scaled by *interest* (0=low, 1=high).
        """
        started = time.monotonic()
        pattern = "unavailable"
        chunk_count = 0
        self._profile_read_wheel_events = 0
        self._profile_read_timing_active = True
        try:
            # Fresh lingering budget per profile (consumed by
            # _profile_read_dwell).
            self._profile_read_elapsed = 0.0
            container = self.page.locator("div.profile__main-container").first
            try:
                await container.wait_for(state="attached", timeout=5000)
            except Exception:
                return  # Profile not loaded, skip simulation

            try:
                height = await container.evaluate("el => el.scrollHeight")
                viewport_h = await container.evaluate("el => el.clientHeight")
            except Exception:
                pattern = "short_dwell"
                await self._profile_read_dwell(
                    human_delay_correlated(
                        self._BASE_SHORT_DWELL,
                        channel="profile_read",
                    )
                )
                return

            if height <= viewport_h:
                pattern = "short_dwell"
                await self._profile_read_dwell(
                    human_delay_correlated(
                        self._BASE_SHORT_DWELL,
                        channel="profile_read",
                    )
                )
                return

            scrollable = height - viewport_h
            chunk_count = max(1, scrollable // self._CHUNK_SIZE)

            if config.LINKEDIN_SECTION_DIRECTED_READ_ENABLED:
                sections = await locate_sections(container)
                pattern = "section:" + (
                    ">".join(s.name for s in sections) if sections else "none"
                )
                chunk_count = len(sections)
                try:
                    await self._read_section_directed(sections, interest)
                except _ProfileReadBudgetExhausted:
                    pass
            else:
                if chunk_count <= 2:
                    length_cat = "short"
                elif chunk_count <= 5:
                    length_cat = "medium"
                else:
                    length_cat = "long"

                pattern = random.choices(
                    self._PATTERN_NAMES,
                    weights=self._PATTERN_WEIGHTS[length_cat],
                )[0]
                print(
                    f"    [profile-read] {length_cat.title()} profile "
                    f"({chunk_count} chunks) → Pattern: {pattern}"
                )

                try:
                    if pattern == "focused_reader":
                        await self._read_focused(scrollable)
                    elif pattern == "skipper":
                        await self._read_skipper(scrollable)
                    elif pattern == "skimmer":
                        await self._read_skimmer(scrollable)
                    else:
                        await self._read_section_hopper(scrollable)
                except _ProfileReadBudgetExhausted:
                    pass
        finally:
            self._profile_read_timing_active = False
            emit_timing_event(
                self._timing_recorder,
                "profile_read_timing",
                elapsed_ms=round((time.monotonic() - started) * 1000.0, 3),
                pattern=pattern,
                chunk_count=int(chunk_count),
                wheel_events=int(self._profile_read_wheel_events),
            )

    async def _read_section_directed(self, sections: list, interest: float) -> None:
        """Hop down profile sections in document order; dwell budget scales with interest."""
        budget = self._interest_read_budget_seconds(interest)
        try:
            await self.page.locator("div.profile__main-container").first.evaluate(
                "el => { el.scrollTop = 0; }"
            )
        except Exception:
            pass
        target_sections = [
            s for s in sections if s.name in self._SECTION_DIRECTED_NAMES
        ]

        if not target_sections:
            await self._profile_read_dwell(
                human_delay_correlated(
                    self._BASE_SHORT_DWELL,
                    channel="profile_read",
                )
            )
            return

        per_section_share = budget / len(target_sections)
        current_offset = 0.0

        for section in target_sections:
            delta = section.offset - current_offset
            if delta > 0:
                await self._human_scroll(int(delta), channel="profile_read")
                current_offset = section.offset

            if section.name == "experience":
                num_entries = random.randint(2, 3)
                per_entry = per_section_share / num_entries
                for _ in range(num_entries):
                    await self._profile_read_dwell(
                        human_delay_correlated(
                            per_entry,
                            channel="profile_read",
                        )
                    )
            else:
                await self._profile_read_dwell(
                    human_delay_correlated(
                        per_section_share,
                        channel="profile_read",
                    )
                )

    async def _read_focused(self, scrollable: int) -> None:
        """Pattern A — Focused reader: top to bottom with careful/skim per chunk."""
        chunk_size = random.randint(300, 500)
        chunks = max(1, scrollable // chunk_size)
        reread_chunk = random.randint(0, chunks - 1)
        scrolled = 0

        for i in range(chunks):
            px = min(chunk_size, scrollable - scrolled)
            await self._human_scroll(px, channel="profile_read")
            scrolled += px

            if random.random() < 0.4:
                # Careful read: 2-4x base dwell
                base = random.uniform(self._BASE_CHUNK_DWELL_LOW, self._BASE_CHUNK_DWELL_HIGH)
                dwell = base * random.uniform(1.5, 2.5)
            else:
                # Skim: 0.3-0.5x base dwell
                base = random.uniform(self._BASE_CHUNK_DWELL_LOW, self._BASE_CHUNK_DWELL_HIGH)
                dwell = base * random.uniform(0.3, 0.5)

            await self._profile_read_dwell(human_delay_correlated(dwell, channel="profile_read"))

            if i == reread_chunk:
                # Re-read pause
                await self._profile_read_dwell(human_delay_correlated(random.uniform(2.0, 8.0), channel="profile_read"))

        # Scroll back to top
        await self._human_scroll(-scrolled, channel="profile_read_return")
        await asyncio.sleep(human_delay_correlated(0.5, channel="profile_read_return"))

    async def _read_skipper(self, scrollable: int) -> None:
        """Pattern B — Skipper: jump to bottom section, back up, then finish."""
        chunk_size = random.randint(300, 500)

        # 1. Fast scroll to ~60-70% height
        target_1 = int(scrollable * random.uniform(0.6, 0.7))
        chunks_down_1 = max(1, target_1 // chunk_size)
        scrolled = 0
        for _ in range(chunks_down_1):
            px = min(chunk_size, target_1 - scrolled)
            await self._human_scroll(px, channel="profile_read")
            scrolled += px
            base = random.uniform(self._BASE_CHUNK_DWELL_LOW, self._BASE_CHUNK_DWELL_HIGH)
            await self._profile_read_dwell(human_delay_correlated(base * 0.3, channel="profile_read"))

        # 2. Pause
        await self._profile_read_dwell(human_delay_correlated(random.uniform(3.0, 8.0), channel="profile_read"))

        # 3. Scroll back up to ~20-30%
        target_2 = int(scrollable * random.uniform(0.2, 0.3))
        scroll_up = scrolled - target_2
        if scroll_up > 0:
            chunks_up = max(1, scroll_up // chunk_size)
            for _ in range(chunks_up):
                px = min(chunk_size, scroll_up)
                await self._human_scroll(-px, channel="profile_read")
                scrolled -= px
                scroll_up -= px
                base = random.uniform(self._BASE_CHUNK_DWELL_LOW, self._BASE_CHUNK_DWELL_HIGH)
                await self._profile_read_dwell(human_delay_correlated(base * 0.5, channel="profile_read"))

        # 4. Pause
        await self._profile_read_dwell(human_delay_correlated(random.uniform(2.0, 5.0), channel="profile_read"))

        # 5. Scroll to bottom
        remaining = scrollable - scrolled
        if remaining > 0:
            chunks_rest = max(1, remaining // chunk_size)
            for _ in range(chunks_rest):
                px = min(chunk_size, remaining)
                await self._human_scroll(px, channel="profile_read")
                scrolled += px
                remaining -= px
                base = random.uniform(self._BASE_CHUNK_DWELL_LOW, self._BASE_CHUNK_DWELL_HIGH)
                await self._profile_read_dwell(human_delay_correlated(base, channel="profile_read"))

        # Scroll back to top
        await self._human_scroll(-scrolled, channel="profile_read_return")
        await asyncio.sleep(human_delay_correlated(0.5, channel="profile_read_return"))

    async def _read_skimmer(self, scrollable: int) -> None:
        """Pattern C — Skimmer: fast down, slow back up."""
        chunk_size = random.randint(300, 500)
        chunks = max(1, scrollable // chunk_size)
        scrolled = 0

        # 1. Fast scroll down
        for _ in range(chunks):
            px = min(chunk_size, scrollable - scrolled)
            await self._human_scroll(px, channel="profile_read")
            scrolled += px
            base = random.uniform(self._BASE_CHUNK_DWELL_LOW, self._BASE_CHUNK_DWELL_HIGH)
            await self._profile_read_dwell(human_delay_correlated(base * random.uniform(0.3, 0.8), channel="profile_read"))

        # 2. Bottom pause
        await self._profile_read_dwell(human_delay_correlated(random.uniform(1.0, 3.0), channel="profile_read"))

        # 3. Slow scroll back up
        scroll_remaining = scrolled
        while scroll_remaining > 0:
            px = min(chunk_size, scroll_remaining)
            await self._human_scroll(-px, channel="profile_read")
            scroll_remaining -= px
            base = random.uniform(self._BASE_CHUNK_DWELL_LOW, self._BASE_CHUNK_DWELL_HIGH)
            await self._profile_read_dwell(human_delay_correlated(base * random.uniform(1.5, 3.0), channel="profile_read"))

        await asyncio.sleep(human_delay_correlated(0.5, channel="profile_read_return"))

    async def _read_section_hopper(self, scrollable: int) -> None:
        """Pattern D — Section hopper: top to bottom with 1-2 backtracks."""
        chunk_size = random.randint(300, 500)
        chunks = max(1, scrollable // chunk_size)
        backtrack_points = sorted(random.sample(range(1, max(2, chunks)), min(random.randint(1, 2), chunks - 1)))
        scrolled = 0

        for i in range(chunks):
            px = min(chunk_size, scrollable - scrolled)
            await self._human_scroll(px, channel="profile_read")
            scrolled += px
            base = random.uniform(self._BASE_CHUNK_DWELL_LOW, self._BASE_CHUNK_DWELL_HIGH)
            await self._profile_read_dwell(human_delay_correlated(base, channel="profile_read"))

            if i in backtrack_points:
                # Backtrack: scroll UP 1 chunk, pause, then resume
                backtrack_px = min(chunk_size, scrolled)
                await self._human_scroll(-backtrack_px, channel="profile_read")
                scrolled -= backtrack_px
                await self._profile_read_dwell(human_delay_correlated(random.uniform(2.0, 4.0), channel="profile_read"))
                # Resume forward
                await self._human_scroll(backtrack_px, channel="profile_read")
                scrolled += backtrack_px

        # Scroll back to top
        await self._human_scroll(-scrolled, channel="profile_read_return")
        await asyncio.sleep(human_delay_correlated(0.5, channel="profile_read_return"))

    # ── Scroll helpers for post-evaluation dwell ─────────────────

    async def scroll_for_linger(self, chunks_back: int) -> int:
        """Scroll back up N chunks for re-reading, return actual pixels scrolled."""
        requested = chunks_back * random.randint(300, 500)
        try:
            current_top = int(
                await self.page.locator("div.profile__main-container").first.evaluate(
                    "el => el.scrollTop"
                )
            )
        except Exception:
            current_top = 0

        px = min(requested, max(0, current_top))
        if px > 0:
            await self._human_scroll(-px, channel="profile_linger")
        return px

    async def scroll_restore(self, px: int):
        """Scroll back down to restore position."""
        if px > 0:
            await self._human_scroll(px, channel="profile_linger")

    async def get_profile_innertext(self) -> str:
        """Get trimmed innerText from div.profile__main-container.

        Expands all collapsed "Read more" / "see more" sections first,
        then extracts text. Keeps header + Summary + Experience + Education.
        Skips Accomplishments, Volunteer Experience, Personal Information, etc.
        """
        async def _do():
            container = self.page.locator("div.profile__main-container").first
            await container.wait_for(state="attached", timeout=10000)

            # Expand all collapsed sections before extracting text
            await self._expand_all_readmore(container)

            full_text = await container.inner_text(timeout=15000)
            return _trim_profile_text(full_text)
        started = time.monotonic()
        text = ""
        succeeded = False
        try:
            text = await _retry(_do)
            succeeded = True
            return text
        finally:
            emit_timing_event(
                self._timing_recorder,
                "profile_innertext_timing",
                elapsed_ms=round((time.monotonic() - started) * 1000.0, 3),
                text_chars=len(text),
                succeeded=succeeded,
            )

    async def get_profile_recent_recruiting_activity(self) -> list[str]:
        """Read recruiter-facing recent activity lines from the slide-in profile."""

        async def _do():
            container = self.page.locator("div.profile-slidein__container").first
            await container.wait_for(state="visible", timeout=10000)
            text = await container.inner_text(timeout=15000)
            return extract_profile_recent_activity_lines(text)

        return await _retry(_do)

    async def get_profile_status_summary(self) -> dict:
        """Read recruiter-facing status metadata from the slide-in profile."""

        async def _do():
            container = self.page.locator("div.profile-slidein__container").first
            await container.wait_for(state="visible", timeout=10000)
            text = await container.inner_text(timeout=15000)
            return extract_profile_status_summary(text)

        return await _retry(_do)

    async def expand_profile_sections(self) -> None:
        """Expand collapsed profile sections before the read pass (fail-soft)."""
        try:
            container = self.page.locator("div.profile__main-container").first
            await container.wait_for(state="attached", timeout=10000)
            await self._expand_all_readmore(container)
        except Exception:
            pass

    async def _expand_all_readmore(self, container) -> None:
        """Expand collapsed sections in the profile slide-in.

        Only expands content sections that matter for evaluation:
        - About/Summary "see more" links
        - Experience entry description bullets
        - Education details

        Capped at 15 clicks. Uses ghost-cursor for human-like click trajectories.
        Skips skills endorsements, recommendations, and other non-eval sections.
        """
        started = time.monotonic()
        MAX_EXPANSIONS = 15
        expand_texts = ["see more", "read more", "show more", "ver mais"]
        elements_walked = 0
        clicked = 0

        async def _try_expand_in_scope(scope) -> None:
            nonlocal elements_walked, clicked
            if clicked >= MAX_EXPANSIONS:
                return
            clickables = await scope.locator(
                'a, button, [role="button"]'
            ).all()
            for el in clickables:
                if clicked >= MAX_EXPANSIONS:
                    break
                elements_walked += 1
                try:
                    text = (await el.inner_text(timeout=1000)).strip().lower()
                    if len(text) < 20 and any(t in text for t in expand_texts):
                        visible = await el.is_visible()
                        if visible:
                            await self._ghost_click_locator(el)
                            clicked += 1
                            await asyncio.sleep(
                                human_delay_correlated(
                                    config.LINKEDIN_PROFILE_EXPAND_CLICK_DWELL_SECONDS,
                                    channel="profile_expand",
                                )
                            )
                            return
                except Exception:
                    continue

        try:
            if config.LINKEDIN_SECTION_DIRECTED_READ_ENABLED:
                await _try_expand_in_scope(container.locator(".summary-card"))

                experience_count = random.randint(2, 4)
                position_items = await container.locator(
                    ".experience-card .position-item"
                ).all()
                for entry in position_items[:experience_count]:
                    if clicked >= MAX_EXPANSIONS:
                        break
                    await _try_expand_in_scope(entry)
            else:
                clickables = await container.locator(
                    'a, button, [role="button"]'
                ).all()
                for el in clickables:
                    if clicked >= MAX_EXPANSIONS:
                        break
                    elements_walked += 1
                    try:
                        text = (await el.inner_text(timeout=1000)).strip().lower()
                        if len(text) < 20 and any(t in text for t in expand_texts):
                            visible = await el.is_visible()
                            if visible:
                                await self._ghost_click_locator(el)
                                clicked += 1
                                await asyncio.sleep(
                                    human_delay_correlated(
                                        config.LINKEDIN_PROFILE_EXPAND_CLICK_DWELL_SECONDS,
                                        channel="profile_expand",
                                    )
                                )
                    except Exception:
                        continue

            if clicked:
                await self.page.wait_for_timeout(
                    int(config.LINKEDIN_PROFILE_EXPAND_SETTLE_SECONDS * 1000)
                )
                print(f"    [profile] Expanded {clicked} collapsed section(s)")
        finally:
            emit_timing_event(
                self._timing_recorder,
                "profile_expand_timing",
                elapsed_ms=round((time.monotonic() - started) * 1000.0, 3),
                elements_walked=elements_walked,
                clicks_made=clicked,
            )

    async def go_back_to_results(self) -> None:
        """Dismiss the profile slide-in panel if it's open.

        The artdeco-modal-outlet overlay can block all Playwright clicks,
        so we use multiple strategies including JS-based dismissal.
        Never calls go_back() — that navigates away from the search page.
        """
        async def _do():
            slidein = self.page.locator("div.profile-slidein__container")
            if not await slidein.is_visible():
                return  # Slide-in not open — nothing to dismiss

            # Strategy 1: Find and click any close/dismiss button inside the modal via JS
            # (bypasses pointer-events interception)
            closed = await self.page.evaluate("""() => {
                const outlet = document.getElementById('artdeco-modal-outlet');
                if (!outlet) return false;
                // Try multiple close button patterns
                const selectors = [
                    'button[aria-label="Close"]',
                    'button[aria-label="close"]',
                    'button.artdeco-modal__dismiss',
                    'button[data-test-modal-close-btn]',
                    'button.artdeco-button--circle[aria-label]',
                ];
                for (const sel of selectors) {
                    const btn = outlet.querySelector(sel);
                    if (btn) { btn.click(); return true; }
                }
                return false;
            }""")
            if closed:
                await self.page.wait_for_timeout(1000)
                if not await slidein.is_visible():
                    return

            # Strategy 2: Escape key
            await self._press_key("Escape")
            await self.page.wait_for_timeout(1000)
            if not await slidein.is_visible():
                return

            # Strategy 3: JS — click the back/close link in the slide-in header
            await self.page.evaluate("""() => {
                const outlet = document.getElementById('artdeco-modal-outlet');
                if (!outlet) return;
                // Click any <a> or <button> that looks like navigation back
                const links = outlet.querySelectorAll('a, button');
                for (const el of links) {
                    const text = (el.textContent || '').trim().toLowerCase();
                    const label = (el.getAttribute('aria-label') || '').toLowerCase();
                    if (text === '×' || text === 'x' || label.includes('close') ||
                        label.includes('dismiss') || label.includes('back')) {
                        el.click(); return;
                    }
                }
                // Nuclear: hide the modal outlet entirely
                outlet.style.display = 'none';
            }""")
            await self.page.wait_for_timeout(1000)

            # If we hid the outlet, restore it after a beat so future modals work
            await self.page.evaluate("""() => {
                const outlet = document.getElementById('artdeco-modal-outlet');
                if (outlet && outlet.style.display === 'none') {
                    outlet.style.display = '';
                    // Clear its children to remove stale modal content
                    outlet.innerHTML = '';
                }
            }""")
            await self.page.wait_for_timeout(500)
        await _retry(_do)

    async def next_profile_in_panel(self) -> bool:
        """Click "Next candidate" skyline-pagination-button in the slide-in."""
        try:
            buttons = self.page.locator("button.skyline-pagination-button")
            count = await buttons.count()
            for i in range(count):
                text = await buttons.nth(i).inner_text()
                if "next" in text.lower():
                    await buttons.nth(i).click()
                    await self.page.wait_for_timeout(1500)
                    return True
            return False
        except Exception:
            return False

    async def previous_profile_in_panel(self) -> bool:
        try:
            buttons = self.page.locator("button.skyline-pagination-button")
            count = await buttons.count()
            for i in range(count):
                text = await buttons.nth(i).inner_text()
                if "previous" in text.lower():
                    await buttons.nth(i).click()
                    await self.page.wait_for_timeout(1500)
                    return True
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Save candidate to pipeline
    # ------------------------------------------------------------------

    async def is_already_saved(
        self,
        *,
        expected_name: str | None = None,
    ) -> bool:
        """Check if the currently open profile is already saved to the project.

        NOT ON THE SAVE PATH — see `is_already_saved_on_card`. Wave G moved both
        the save click and its confirmation onto the candidate's own result
        card, because this panel read can only answer about whatever the panel
        is currently showing, and the panel carries other people's profile links.
        Do not reintroduce this to `save_candidate`: the name comparison below
        is a heuristic standing in for identity, which is exactly what the card
        scoping replaced. Kept for non-save readers of the panel state.

        When a candidate is already saved, the 'Save to pipeline' button is
        replaced by a 'Change stage' button with different styling.

        With ``expected_name``, the visible 'Change stage' state is only
        believed when the panel actually shows that person: names are
        accent-folded (the card lockup drops diacritics the panel renders) and
        then compared token for token, in order, exactly — a prefix only counts
        where the shorter token is explicitly abbreviated (a lone initial or an
        ellipsis). Loose prefixing confirmed the wrong person ('Ann Li' read as
        'Ann Lin'); raw comparison rejected the right one ('Jose Garcia' vs
        'José García'), which fails the post-click probe and aborts the run.

        Fail-open is deliberate in two places: an unreadable panel and an
        expected name with no word characters both fall back to plain
        visibility, because a DOM read that did not happen is not evidence of a
        mismatch.
        """
        change_stage = self.page.locator(
            'div.profile-slidein__container :is(button:has-text("Change stage"), '
            'button[data-test-change-stage-button])'
        ).first
        visible = await change_stage.is_visible(timeout=2000)
        if not visible or not expected_name:
            return visible
        try:
            panel_text = await self.page.locator(
                "div.profile__main-container"
            ).first.inner_text(timeout=2000)
        except Exception:
            return visible

        if not _name_tokens(expected_name):
            return visible
        return _panel_confirms_expected_name(panel_text, expected_name)

    # Primary (happy-path) save-trigger selector. Kept as a class attribute so
    # the resolver helper and tests reference one source of truth.
    _SAVE_TRIGGER_PRIMARY = (
        "div.profile-slidein__container button.save-to-pipeline__button"
    )
    # Semantic fallback: the save trigger's DOM-verified accessible name is
    # "Save to '{stage}'" (e.g. "Save to 'uncontacted'"). Mirrors the text +
    # data-test defensiveness already used by is_already_saved() for the
    # "Change stage" state, so a class rotation does not silently break saves.
    _SAVE_TRIGGER_FALLBACK = (
        "div.profile-slidein__container "
        ':is(button:has-text("Save to "), button[data-test-save-to-pipeline-button])'
    )

    # ------------------------------------------------------------------
    # Wave G — the save click, scoped to the candidate's own result card
    # ------------------------------------------------------------------
    #
    # Live capture (2026-07-26) walked 25 ancestors up from the profile panel's
    # save control and found NOTHING carrying the open candidate's id, while the
    # panel itself contains other people's profile links (similar profiles). The
    # panel can therefore be guarded but never scoped — which is why five rounds
    # of point-in-time assertions each left a sixth gap. The result card's <li>
    # holds both the candidate's own anchor and a save control, so the click can
    # be rooted in the candidate's identity instead of raced against it.
    #
    # A Recruiter identity fragment is an opaque path segment. Anything outside
    # this alphabet cannot go into a CSS attribute selector.
    _IDENTITY_FRAGMENT_SAFE_RE = re.compile(r"^[A-Za-z0-9_-]+$")

    # The main save button, per docs/linkedin-recruiter-dom-map.md:51-62 which
    # records it on the RESULT CARDS ("12 save buttons found on page (one per
    # visible result card x 2: main + dropdown)"). The dropdown trigger opens a
    # stage menu instead of saving, and the map is explicit that it must not be
    # clicked, so it is excluded three ways: it carries neither
    # `save-to-pipeline__button` nor the data-test attributes, and both its own
    # class and the generic dropdown class are negated here.
    # One synchronous browser turn: verify the project, find the candidate's own
    # card by EXACT title-anchor identity, find exactly one save control inside
    # it, and press it — with nothing awaited in between, so the DOM cannot move
    # between the checks and the click.
    #
    # This replaces a resolve-then-click sequence that was three separate CDP
    # round-trips. Each gap was exploitable and two were demonstrated by driving
    # the real methods:
    #   * the resolved handle was `cards.nth(i)` — a POSITIONAL index into a live
    #     `href*=` substring set. Executed: a handle verified as candidate `AAA`
    #     pointed at `AAA2` once `AAA` virtualised out. Prefix-confusable ids are
    #     ordinary, so this was a reachable wrong-person save.
    #   * the project/identity guard ran in Python one round-trip before the
    #     click, so a page that changed project in between was clicked anyway.
    # Coordinate presses (hover -> mouse.down/up) have the same disease and are
    # not used for saves at all now: `mouse.down()` presses a POINT, and a list
    # reflow after the guard puts a neighbour under it.
    #
    # `dryRun` resolves and verifies WITHOUT pressing, which is what lets
    # resolution be retried while actuation happens exactly once.
    _CARD_SAVE_TRANSACTION_JS = r"""
    (args) => {
      const {fragment, project, dryRun} = args;
      const frag = u => (String(u||'').match(
        /\/(?:talent\/profile|recruiterSearch\/profile)\/([^/?#]+)/)||[])[1] || null;

      // 1. The project is read from the LIVE location in this same turn.
      const pageProject = (location.pathname.match(
        /\/talent\/hire\/([^/?#]+)/)||[])[1] || null;
      if (project && pageProject !== project) {
        return {ok:false, reason:'project_mismatch', saw:pageProject};
      }

      // 2. Exactly one card whose OWN title anchor is exactly this candidate.
      //    Exact fragment equality, never substring: 'AAA' must not match 'AAA2'.
      const owned = [...document.querySelectorAll('ol.profile-list > li')].filter(li => {
        const a = li.querySelector(
          'article.profile-list-item [class*="lockup__title"] a');
        return a && frag(a.getAttribute('href')) === fragment;
      });
      if (owned.length !== 1) {
        return {ok:false, reason: owned.length ? 'card_ambiguous' : 'card_not_found',
                matched: owned.length};
      }
      const li = owned[0];

      // 3. Exactly one save control inside THAT card. The dropdown trigger opens
      //    a stage menu instead of saving and is excluded by class, twice.
      const ctrls = [...li.querySelectorAll('button')].filter(b => {
        const cls = String(b.className || '');
        if (cls.includes('save-to-pipeline__dropdown-trigger')) return false;
        if (cls.includes('artdeco-dropdown__trigger')) return false;
        if (b.matches('[data-test-save-to-first-stage], button.save-to-pipeline__button,'
                    + ' button[data-test-save-to-pipeline-button]')) return true;
        return /^Save to /.test((b.textContent||'').trim());
      });
      if (ctrls.length !== 1) {
        // Saved-state detection. LIVE-CORRECTED 2026-07-27: the attribute this
        // originally keyed on, `data-test-change-stage-button`, does not exist
        // in live Recruiter — a read-only extraction against two genuinely
        // saved cards in project 3000000001 found it absent on BOTH, so the
        // text fallback was silently carrying the whole check on its own. The
        // real control is the move-to-pipeline dropdown trigger, and it carries
        // stable data attributes under a different name. Both observed names are
        // matched, plus the class, plus the original attribute (harmless if it
        // ever returns), and the text stays last as the final fallback.
        const saved = !!li.querySelector(
              'button[data-test-move-to-trigger],'
            + ' button[data-live-test-change-state-trigger],'
            + ' button.move-to-pipeline__dropdown-trigger,'
            + ' button[data-test-change-stage-button]')
          || [...li.querySelectorAll('button')].some(
               b => /Change stage/i.test(b.textContent||''));
        if (saved) return {ok:false, reason:'already_saved'};
        return {ok:false, reason: ctrls.length ? 'trigger_ambiguous' : 'trigger_not_found',
                matched: ctrls.length};
      }

      if (dryRun) return {ok:true, dispatched:false};
      // 4. Press, in the same turn every check above ran in.
      ctrls[0].click();
      return {ok:true, dispatched:true};
    }
    """

    _CARD_TX_REASON_TO_FAILURE = {
        "project_mismatch": "save_project_mismatch",
        "card_not_found": "save_card_not_found",
        "card_ambiguous": "save_card_ambiguous",
        "trigger_not_found": "save_trigger_not_found",
        "trigger_ambiguous": "save_trigger_ambiguous",
        "identity_unusable": "save_card_identity_unusable",
    }

    async def _card_save_transaction(
        self,
        identity_fragment: str,
        *,
        dry_run: bool,
    ) -> dict:
        """Run the atomic card-save transaction. Returns the JS result verbatim."""
        if not identity_fragment or not self._IDENTITY_FRAGMENT_SAFE_RE.match(
            identity_fragment
        ):
            return {"ok": False, "reason": "identity_unusable"}
        return await self.page.evaluate(
            self._CARD_SAVE_TRANSACTION_JS,
            {
                "fragment": identity_fragment,
                "project": self._required_project_id or None,
                "dryRun": bool(dry_run),
            },
        )

    async def is_already_saved_on_card(self, identity_fragment: str) -> bool | None:
        """Whether THIS candidate's card shows the saved state. None = unreadable.

        Answered by the SAME atomic transaction that performs the save, in one
        browser turn, so the card that is read is provably the card that would
        be clicked. The previous implementation resolved a card in Python and
        then read a control off it in a second round-trip, holding a positional
        handle into a live substring-matched set — the identical defect that made
        the click retargetable. Here it would silently mis-answer instead:
        reading a neighbour's saved state terminalizes a SAVE obligation as
        `already_present` and the candidate is never saved at all.

        Tri-state, and the distinction is load-bearing. None (not False) whenever
        the card cannot be read, so callers can tell "provably not saved" from
        "could not look" — spending the latter as the former is what would let a
        save fire against a page nobody verified.
        """
        try:
            outcome = await self._card_save_transaction(
                identity_fragment, dry_run=True
            )
        except Exception as exc:
            # A dead target is not an unreadable card. Swallowing it would turn
            # "the browser crashed" into the same benign None used for "this
            # candidate is not on the rendered page", and the run would carry on
            # against a page that no longer exists (E3, different wrapper).
            if _is_target_crash_error(exc):
                raise
            return None

        if outcome.get("ok"):
            # The save control is present and unambiguous: provably NOT saved.
            return False
        reason = outcome.get("reason")
        if reason == "already_saved":
            return True
        # project_mismatch / card_not_found / card_ambiguous / trigger_ambiguous
        # are all "could not answer for THIS candidate on THIS page".
        return None

    async def _resolve_save_trigger(self, *, timeout: int = 5000):
        """Locate the save-to-pipeline trigger, primary class first then fallback.

        Returns a visible Playwright Locator, or None if neither the primary
        class selector nor the semantic (accessible-name) fallback resolves a
        save trigger. Returning None lets the caller report a class rotation as
        ``save_trigger_not_found`` rather than misattributing it to a
        clicked-but-did-not-persist failure.
        """
        primary = self.page.locator(self._SAVE_TRIGGER_PRIMARY).first
        try:
            await primary.wait_for(state="visible", timeout=timeout)
            return primary
        except Exception:
            pass
        # Class rotation / DOM drift: fall back to the accessible-name pattern.
        fallback = self.page.locator(self._SAVE_TRIGGER_FALLBACK).first
        try:
            await fallback.wait_for(state="visible", timeout=timeout)
            return fallback
        except Exception:
            return None

    async def save_candidate(
        self,
        *,
        before_click: Callable[[], None] | None = None,
    ) -> bool:
        """Save this candidate by clicking the control inside their OWN result card.

        Wave G. The click used to be issued against the profile slide-in, whose
        save control has no identity-bearing ancestor — so the target could only
        ever be guarded by point-in-time assertions racing a live SPA, and five
        review rounds each found a new gap between the last assertion and the
        pointer press. Here the control is *resolved through* the candidate's own
        anchor, so a DOM swap produces a lookup failure instead of a
        wrong-person click. Evaluation is unchanged: the panel still opens and is
        still read for judgment; only the physical click moved.

        Both the already-saved probe and the post-click confirmation read the
        same card, which is what makes the foreign-project already-saved bypass
        structurally impossible rather than separately guarded.

        Refuses rather than falls back: if the card or its control cannot be
        resolved, there is no button this method is entitled to press, and
        clicking the panel's would reintroduce exactly the unscoped click this
        change exists to remove.

        On failure, ``self._last_save_failure_reason`` distinguishes a missing
        card (``"save_card_not_found"``), an ambiguous one
        (``"save_card_ambiguous"``), a missing trigger
        (``"save_trigger_not_found"``, e.g. class rotation) and a click that did
        not persist (``"save_not_persisted"``).
        """
        self._last_save_failure_reason = None
        expected_identity = self.current_profile_identity_fragment()

        def _guard_identity() -> None:
            if (
                not expected_identity
                or self.current_profile_identity_fragment() != expected_identity
            ):
                raise _SaveOperationAbort(
                    RuntimeError("profile identity mismatch during save")
                )

        def _guard_click() -> None:
            if before_click is None:
                return
            try:
                before_click()
            except BaseException as exc:
                raise _SaveOperationAbort(exc) from exc

        async def _probe_saved() -> bool | None:
            """The candidate's OWN card's saved state. None = could not read it.

            None is not False. "I could not look" must never be spent as
            "provably not saved" before the click, nor as a failure that
            overwrites a save that may well have landed after it.
            """
            _guard_identity()
            try:
                state = await self.is_already_saved_on_card(expected_identity)
            except BaseException as exc:
                self._last_save_failure_reason = "save_probe_failed"
                raise _SaveOperationAbort(exc) from exc
            _guard_identity()
            return state

        async def _resolve_only():
            """Retryable phase: prove the card + control are there. Presses nothing."""
            if await _probe_saved() is True:
                return {"ok": True, "already": True}
            outcome = await self._card_save_transaction(
                expected_identity, dry_run=True
            )
            if outcome.get("ok"):
                return outcome
            reason = outcome.get("reason") or "card_not_found"
            if reason == "already_saved":
                return {"ok": True, "already": True}

            if reason == "card_not_found":
                # VIRTUALISATION. The results list keeps an <li> per slot but
                # renders the article only while the slot is near the viewport,
                # so an unrendered card carries no title anchor and the
                # transaction correctly answers "not found" for a candidate who
                # IS on this page. Measured live 2026-07-27: 21 of 25 slots were
                # empty `profile-list__occlusion-area` placeholders until the
                # list was scrolled through.
                #
                # This scroll was lost when resolution moved into the browser
                # transaction — the previous Python resolver called this first.
                # It belongs HERE, in the retryable phase, not inside the
                # transaction: the transaction must stay a pure verify-and-click
                # with no layout-shifting side effects between its checks and the
                # press. Scrolling presses nothing, so retrying it is free.
                await self._find_result_slot_by_fragment(expected_identity)
                outcome = await self._card_save_transaction(
                    expected_identity, dry_run=True
                )
                if outcome.get("ok"):
                    return outcome
                reason = outcome.get("reason") or "card_not_found"
                if reason == "already_saved":
                    return {"ok": True, "already": True}

            self._last_save_failure_reason = self._CARD_TX_REASON_TO_FAILURE.get(
                reason, "save_card_not_found"
            )
            raise Exception(f"Candidate card not resolvable ({reason})")

        # ---- Phase 1: resolution. Retryable, because it presses nothing. ----
        try:
            resolved = await _retry(_resolve_only)
        except _SaveOperationAbort as abort:
            raise abort.cause
        except Exception as e:
            print(f"  [warn] Save refused before any click "
                  f"({self._last_save_failure_reason}): {e}")
            return False
        if resolved.get("already"):
            self._last_save_failure_reason = None
            return True

        # ---- Phase 2: actuation. EXACTLY ONCE, never retried. ----
        # A retry around the press meant one logical save could emit three
        # physical clicks: the dispatch succeeded, confirmation came back False
        # or unreadable, `_do` raised, and `_retry` re-entered and pressed again.
        # Executed against the pre-fix code: 3 presses, in BOTH the "card still
        # says unsaved" and "card unreadable" cases. Anything past this line may
        # have physically saved a real person, so nothing past it may press.
        _guard_click()
        try:
            outcome = await self._card_save_transaction(
                expected_identity, dry_run=False
            )
        except _SaveOperationAbort:
            raise
        except Exception as exc:
            # Either way the press MAY already have gone out before the failure,
            # so the reason is set BEFORE deciding how to surface it. A crash
            # still propagates (the run must stop on a dead target), but it must
            # not propagate as though nothing physical had been attempted — this
            # is the one state where Recruiter can hold a save that local state
            # has no confirmed record of.
            self._last_save_failure_reason = "save_not_confirmed"
            if _is_target_crash_error(exc):
                raise
            print(f"  [warn] Save dispatch failed inconclusively: {exc}")
            return False

        if not outcome.get("dispatched"):
            reason = outcome.get("reason") or "card_not_found"
            if reason == "already_saved":
                self._last_save_failure_reason = None
                return True
            self._last_save_failure_reason = self._CARD_TX_REASON_TO_FAILURE.get(
                reason, "save_card_not_found"
            )
            # The transaction refused inside the browser, before clicking, so no
            # press went out and the obligation is cleanly still owed.
            print(f"  [warn] Save refused inside the commit transaction ({reason})")
            return False

        # ---- Phase 3: confirmation ONLY. This path never presses again. ----
        confirmed = None
        for _ in range(3):
            await self.page.wait_for_timeout(2000)
            try:
                confirmed = await _probe_saved()
            except _SaveOperationAbort as abort:
                # A guard fired while confirming a save that ALREADY dispatched.
                # Record the dispatch honestly rather than reporting a clean abort.
                self._last_save_failure_reason = "save_not_confirmed"
                raise abort.cause
            if confirmed is True:
                self._last_save_failure_reason = None
                return True
        # A click DID go out. False means the card still offers to save, so it
        # demonstrably did not take. None means the card became unreadable and
        # the save may have landed in Recruiter with no local confirmation — a
        # different, worse state that must not be filed under the same reason.
        self._last_save_failure_reason = (
            "save_not_persisted" if confirmed is False else "save_not_confirmed"
        )
        print(f"  [warn] Save dispatched but not confirmed "
              f"({self._last_save_failure_reason})")
        return False

    # ------------------------------------------------------------------
    # Legacy aliases (orchestrator compatibility)
    # ------------------------------------------------------------------

    async def get_results_page_dom(self) -> str:
        return await self.get_results_list_innertext()

    async def get_profile_dom(self) -> str:
        return await self.get_profile_innertext()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _trim_profile_text(full_text: str) -> str:
    """Trim profile innerText to relevant sections only."""
    lines = full_text.split("\n")
    trimmed = []
    skip_sections = {
        "accomplishments", "volunteer experience", "personal information",
        "similar profiles", "projects", "messages", "greenhouse", "feedback",
    }
    current_skip = False

    for line in lines:
        line_lower = line.strip().lower()
        if line_lower in skip_sections:
            current_skip = True
            continue
        if line_lower in {"summary", "experience", "education"}:
            current_skip = False
        if not current_skip:
            trimmed.append(line)

    return "\n".join(trimmed)


def run_sync(coro):
    return asyncio.get_event_loop().run_until_complete(coro)
