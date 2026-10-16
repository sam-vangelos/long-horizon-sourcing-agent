"""Browser automation for LinkedIn Recruiter via Playwright.

All selectors verified against docs/linkedin-recruiter-dom-map.md.
Profile is a SLIDE-IN PANEL, not page navigation.

IMPORTANT: This attaches to an existing browser with an active LinkedIn Recruiter session.
It does NOT launch a new browser or handle login.
"""

from __future__ import annotations
import asyncio
import json
import logging
import random
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING
from urllib.parse import parse_qs, urlparse
from shared import config
from shared.human_timing import human_delay_correlated
from linkedin.acquisition import _is_browser_disconnect_error
from linkedin.activity_parser import (
    extract_profile_recent_activity_lines,
    extract_profile_status_summary,
    extract_recruiter_activity_from_card_text,
)
from linkedin.candidate_note import note_probe_text
from linkedin.input_backends import (
    TypingPlan,
    TypingResult,
    build_boolean_typing_plan,
    build_note_typing_plan,
    create_input_backend,
)
from linkedin.profile_sections import locate_sections
from linkedin.timing_telemetry import TimingRecorder, emit_timing_event

if TYPE_CHECKING:
    from rebrowser_playwright.async_api import Browser, Page, BrowserContext

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
# The prior 21 x 120ms (~2.5s) facet settle budget missed the measured ~3s
# live rail rerender; 31 polls give it ~3.7s without making the wait unbounded.
FACET_SETTLE_ATTEMPTS = 31
FACET_SETTLE_INTERVAL_MS = 120
# How long a facet typeahead gets to repaint its option rows for the name just
# typed. It is the same order as the visibility poll that precedes it, because
# the listbox that poll returned on can be the previous value's: until the rows
# move, a near miss read off them is an answer to somebody else's query.
_FACET_LISTBOX_REPAINT_TIMEOUT_MS = 4000
_FACET_LISTBOX_REPAINT_INTERVAL_MS = 250
# The sidebar's own entry points for a Boolean: the Keywords textarea and the
# controls that open it. A hydrated rail carries one of these whether or not a
# search has returned anything, which is exactly what makes them the right
# readiness evidence — LinkedIn's no-results page keeps the rail and the
# keyword box but has no result summary and no result cards
# (plans/facet-lifecycle-clearing-plan.md Addendum I2). Shared by
# `navigate_to_search` and `assert_search_surface_ready` so the two cannot
# drift apart.
SEARCH_KEYWORD_CONTROL_SELECTORS: tuple[str, ...] = (
    'textarea[id*="free-text-single-value-input"]',
    'button[aria-label*="Profile keywords"]',
    'button[aria-label*="Edit Keywords"]',
    'button[aria-label*="keywords" i]',
)
# How long the left rail must hold still before a caller may touch it.
# A fresh navigation returns on the FIRST visible keyword control, and the
# EMPTY rail carries one: measured on a live navigation, at 4.8s the rail read
# "Add Profile keywords or boolean" with no pills and no summary, while
# LinkedIn's saved-search restore landed ~5-6s after domcontentloaded and
# re-rendered the whole rail (Boolean pill, Locations pill, `searchContextId=`
# in the URL, "433 results"). Typing inside that window is lost to the
# re-render: the location typeahead swallows the keystrokes and the keyword
# editor never opens (run-20260922-040404.log session 1,
# run-20260818-191108.log:29-121, run-20260905-082756.log:1034-1048).
_RAIL_SETTLE_INTERVAL_MS = 1000
_RAIL_SETTLE_STABLE_SAMPLES = 3
_RAIL_SETTLE_MAX_MS = 20000
# Indirection so the settle clock is drivable from tests without sleeping.
_monotonic = time.monotonic
# The filter rail has two states. Every page load and reload opens the SUMMARY
# state (summary pills, an Edit button, qualifications), in which no facet
# wrapper, chip or keyword control exists; Edit opens the edit panel, which IS
# the facet rail every reader here expects. A rail with neither marker is the
# single-state rail, and nothing needs opening.
FILTER_SUMMARY_EDIT_BUTTON = "button[data-test-filter-summary-edit]"
FILTER_EDIT_PANEL = "[data-test-filter-edit-panel]"
_FILTER_EDIT_PANEL_BACK = "button[data-test-filter-edit-panel-back]"
_FILTER_RAIL_WAIT_MS = 8000
_FILTER_RAIL_POLL_MS = 150
_FILTER_EDIT_RECLICK_MS = 3000
# Qualifications (semantic filters) render in the summary state. They persist
# on the project search, are not in the URL, and silently narrow results.
_QUALIFICATION_ITEM = "[data-test-qualification-item]"
_QUALIFICATION_TEXT = "button[data-test-qualification-item-text]"
_QUALIFICATION_REMOVE = "button[data-test-qualification-item-remove]"
_QUALIFICATION_REMOVE_WAIT_MS = 5000
# The result count, newest location first.
RESULT_COUNT_SELECTORS: tuple[str, ...] = (
    "span[data-test-profile-list-num-custom]",
    ".search-query-summary__title",
)
_RESULT_COUNT_RE = re.compile(r"([\d.,]+[KkMm]?\+?)")
# A company or title chip whose scope is not the default carries it as a label
# suffix ("AfterQuery (Current)"); the chip's pill menu names the same scopes
# as data-test-pill-action values.
FACET_SCOPE_DEFAULT = "CURRENT_OR_PAST"
_FACET_SCOPE_SUFFIX_RE = re.compile(
    r"^(.*\S)\s*\((current or past|past not current|current|past)\)$",
    re.IGNORECASE | re.DOTALL,
)
_FACET_SCOPE_BY_SUFFIX = {
    "current": "CURRENT",
    "past": "PAST",
    "current or past": "CURRENT_OR_PAST",
    "past not current": "PAST_NOT_CURRENT",
}
_SCOPED_FACET_DIMENSIONS = frozenset({"companies", "titles"})
_PILL_ACTIONS_TRIGGER = "button[data-test-pill-actions-dropdown-trigger]"
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


class FacetReconciliationError(RuntimeError):
    """The managed Recruiter facet set could not be proved exact."""

    def __init__(self, gate: str, *, receipt: dict[str, Any] | None = None):
        self.gate = gate
        self.receipt = receipt or {}
        super().__init__(f"facet reconciliation failed: {gate}")


def normalize_facet_value_for_compare(value: object) -> str:
    """Normalize facet chip/request values for already-applied comparisons."""
    return str(value or "").strip().casefold()


def split_facet_scope(label: object) -> tuple[str, str | None]:
    """A chip label as (name, scope action), scope None when the label carries none.

    "AfterQuery (Current)" is the company AfterQuery scoped CURRENT; comparing
    the whole label against a requested "AfterQuery" made reconcile dismiss
    the very chip it asked for.
    """
    text = str(label or "").strip()
    match = _FACET_SCOPE_SUFFIX_RE.match(text)
    if not match:
        return text, None
    return match.group(1).strip(), _FACET_SCOPE_BY_SUFFIX[match.group(2).casefold()]


def format_facet_reconciliation_console_line(
    string_id: object,
    receipt: dict[str, Any],
) -> str:
    """Render the single operator line for one facet reconciliation event."""
    failure_gate = str(receipt.get("failure_gate", "") or "")
    if failure_gate:
        return (
            f"  [facet] BLOCKED string #{string_id}: {failure_gate}; "
            "no search entered"
        )

    counts: dict[str, tuple[int, int]] = {}
    for dimension in ("companies", "titles"):
        before = {
            normalize_facet_value_for_compare(value)
            for value in receipt.get("before", {}).get(dimension, []) or []
        }
        desired = {
            normalize_facet_value_for_compare(value)
            for value in receipt.get("desired", {}).get(dimension, []) or []
        }
        counts[dimension] = (len(before - desired), len(desired - before))
    return (
        f"  [facet] string #{string_id}: companies -{counts['companies'][0]} "
        f"+{counts['companies'][1]}, titles -{counts['titles'][0]} "
        f"+{counts['titles'][1]}"
    )


def format_facet_unapplied_console_line(
    string_id: object,
    receipt: dict[str, Any],
) -> str | None:
    """Render the ONE operator line naming values that did not land, or None.

    An exact-match miss is the norm, not an incident: the strategist's
    spelling ("Tess AI") is not always LinkedIn's entity name ("Tess"). The
    string still runs, so the operator needs the narrower fact — which values
    are NOT constraining this search — on one line, next to the receipt.
    """
    unapplied = receipt.get("unapplied") or {}
    parts = [
        f"{dimension}={[str(value) for value in values]}"
        for dimension in ("companies", "titles")
        if (values := list(unapplied.get(dimension) or []))
    ]
    if not parts:
        return None
    line = f"  [facet] string #{string_id} unapplied: {', '.join(parts)}"
    # A name that LinkedIn spells the same as another requested name did not
    # miss the typeahead; it landed on a pill another name already owns. Same
    # line, because the operator's question is the same one: what is not
    # constraining this search.
    collapses = [
        f'"{pair[0]}"+"{pair[1]}"→"{pair[2]}"'
        for dimension in ("companies", "titles")
        for pair in (receipt.get("collisions") or {}).get(dimension) or []
        if len(pair) == 3
    ]
    if collapses:
        line += f" (one pill for two names: {'; '.join(collapses)})"
    scope_misses = (receipt.get("scope_unverified") or {}).get("companies") or []
    if scope_misses:
        line += f" (company scope not set: {', '.join(map(str, scope_misses))})"
    return line


def format_facet_substituted_console_line(
    string_id: object,
    receipt: dict[str, Any],
) -> str | None:
    """Render the ONE operator line naming near-miss takes, or None.

    A taken near miss is not a failure, but it IS a difference between what
    the strategist asked for and what now constrains the search, so the
    operator sees the mapping on one line next to the receipt.
    """
    substituted = receipt.get("substituted") or {}
    parts = []
    for dimension in ("companies", "titles"):
        pairs = substituted.get(dimension) or {}
        if not pairs:
            continue
        rendered = ", ".join(
            f'"{requested}"→"{landed}"' for requested, landed in pairs.items()
        )
        parts.append(f"{dimension} {rendered}")
    if not parts:
        return None
    return f"  [facet] string #{string_id} near-miss: {'; '.join(parts)}"


def facet_misses(receipt: dict[str, Any]) -> frozenset[str]:
    """Every value a reconcile could not land, normalized, across dimensions.

    This is what an advanced-search plan needs to know: the controls it
    carries for those values are already settled, and the ones named here
    settled as failures.
    """
    return frozenset(
        normalize_facet_value_for_compare(value)
        for values in (receipt.get("unapplied") or {}).values()
        for value in values or []
    )


def facet_target_landed(
    requested: set[str],
    receipt: dict[str, Any],
    dimension: str,
) -> set[str]:
    """The subset of ``requested`` a reconcile receipt proves reached the rail.

    Every verification that follows an add-phase reconcile compares against
    THIS, never the request: a string with a recorded miss would otherwise
    fail its own postcondition and be marked errored for succeeding. A value
    taken as a near miss is returned under LinkedIn's spelling, which is the
    only name its pill carries — the whole mapping lives here so no caller has
    to know a substitution happened.
    """
    missed = {
        normalize_facet_value_for_compare(value)
        for value in (receipt.get("unapplied") or {}).get(dimension) or []
    }
    substituted = {
        normalize_facet_value_for_compare(key): landed
        for key, landed in (
            (receipt.get("substituted") or {}).get(dimension) or {}
        ).items()
    }
    landed_set = set()
    for value in requested:
        key = normalize_facet_value_for_compare(value)
        if key in missed:
            continue
        landed_set.add(substituted.get(key, value))
    return landed_set


# ---------------------------------------------------------------------------
# Facet name resolution: the typeahead match rule and its persisted cache.
#
# LinkedIn Recruiter's company/title typeahead answers with ITS entity names,
# not the strategist's spelling: "Mercado Livre" is offered as "Mercado Livre
# Brasil", "Tess AI" as "Tess". Requiring an exact option drops about three
# requested names in ten, so a near miss is TAKEN and recorded, and the
# spelling it resolves to is remembered.
# ---------------------------------------------------------------------------

# Trimmed from each token's edges, never from its middle: "Banco do Brasil
# S.A." keeps its inner dot, while "Architect," and "|" are dressed.
#
# `+` and `&` are NOT in the set: they name entities rather than dress them.
# Stripping them folded "C++ Engineer" onto "C Engineer" and "&Open" onto
# "Open", which is three bugs at once — the typeahead's exact scan took the
# C++ row for a request that said C, a desired set holding both raised
# `titles_desired_normalized_collision` and the string never ran, and two such
# pills on the rail raised the observed twin, which counts toward the
# three-strike stop.
_FACET_NAME_PUNCTUATION = ",.;:!?\"'’“”()[]{}|/\\-–—_*"
# Kept on a token's LEADING edge alone, because there the character is the
# first letter of the name: ".NET Developer" is not "NET Developer" and
# "#1 Cochran" is not "1 Cochran". A trailing "." is still dressing ("Inc.",
# "Architect.").
_FACET_NAME_LEADING_PUNCTUATION_KEPT = ".#"
# A name shorter than this is an initialism, not an entity: "AI" must never
# become the whole company for "AI Lab", and "AI Lab" must never be narrowed
# to "AI". The bar applies to the request and to the option on both sides of
# every non-exact take.
_FACET_NAME_MIN_NEAR_MISS_CHARS = 4
FACET_NAME_RESOLUTIONS_FILENAME = "facet_name_resolutions.json"
_FACET_RESOLUTION_DIMENSIONS = ("companies", "titles")


def facet_name_tokens(value: object) -> tuple[str, ...]:
    """Tokens of a facet name, folded so two spellings of one entity agree.

    Casefold, NFKD-decompose and drop combining marks (so "Itaú" == "Itau"),
    split on whitespace, trim punctuation from each token's edges, and drop
    what is left empty. Punctuation that NAMES the entity survives: `+` and `&`
    anywhere, and a leading `.` or `#`.
    """
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    folded = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    ).casefold()
    tokens = []
    for raw in folded.split():
        token = raw.strip(_FACET_NAME_PUNCTUATION)
        if token and raw[:1] in _FACET_NAME_LEADING_PUNCTUATION_KEPT:
            token = raw[:1] + token
        if token:
            tokens.append(token)
    return tuple(tokens)


def facet_name_key(value: object) -> str:
    """The folded token sequence of a facet name, as one comparable string."""
    return " ".join(facet_name_tokens(value))


def _facet_option_name(text: object) -> str:
    """The entity name of one offered option row.

    ``all_inner_texts`` can return a row's secondary line too (an industry, an
    employee count). The entity is the first line; it is also the text the
    applied pill will carry, which is what confirmation waits for.
    """
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _tokens_contain(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    """Whether ``needle`` appears in ``haystack`` contiguously and in order."""
    if not needle or len(needle) > len(haystack):
        return False
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return True
    return False


def resolve_facet_option(
    requested: object,
    offered: list[object] | tuple[object, ...],
) -> tuple[int, str] | None:
    """Pick the offered typeahead option that satisfies ``requested``.

    Returns ``(index into offered, the option's verbatim name)`` or None.
    Three CLASSES of evidence, tried in order, with LinkedIn's own ranking
    breaking ties only INSIDE a class:

    1. An option whose folded name EQUALS the request — the first such. An
       exact name one row down beats any near miss above it.
    2. Else the first option that CONTAINS the request's tokens contiguously
       ("Mercado Livre" in "Mercado Livre Brasil").
    3. Else the first option the request contains as a leading PREFIX
       ("Tess" for "Tess AI").
    4. Else None — a miss, recorded exactly as an exact-match miss was.

    Classes, not rows, because a container and a prefix are not comparable
    evidence: a container keeps every token the strategist asked for, while a
    prefix throws one away. Ranking them together answered the listbox
    ["Mercado", "Mercado Livre Brasil"] for "Mercado Livre" with "Mercado" —
    a different company — because it reached the prefix row first. A
    container that is wider than the request is the safer error of the two,
    and the console near-miss line is the operator's tripwire for it.

    Only the exact take is unconditional. Both sides of a non-exact take must
    fold to at least ``_FACET_NAME_MIN_NEAR_MISS_CHARS`` characters, so a
    two-letter initialism neither takes an entity that merely carries it
    ("AI" never becomes "AI Lab") nor is taken for one.
    """
    wanted = facet_name_tokens(requested)
    if not wanted:
        return None
    candidates = []
    for index, text in enumerate(offered):
        name = _facet_option_name(text)
        if name:
            candidates.append((index, name, facet_name_tokens(name)))

    for index, name, tokens in candidates:
        if tokens == wanted:
            return index, name
    if len(" ".join(wanted)) < _FACET_NAME_MIN_NEAR_MISS_CHARS:
        return None
    for index, name, tokens in candidates:
        if _tokens_contain(tokens, wanted):
            return index, name
    for index, name, tokens in candidates:
        if (
            len(tokens) < len(wanted)
            and wanted[: len(tokens)] == tokens
            and len(" ".join(tokens)) >= _FACET_NAME_MIN_NEAR_MISS_CHARS
        ):
            return index, name
    return None


def _facet_resolutions_path(state_dir: object) -> Path | None:
    if not state_dir:
        return None
    return Path(state_dir) / FACET_NAME_RESOLUTIONS_FILENAME


def read_facet_name_resolutions(state_dir: object) -> dict[str, dict[str, str]]:
    """Read the resolution sidecar. A missing or corrupt file is an empty cache.

    This is a PROJECTION, not canonical state (AGENTS.md): it only spares the
    typeahead a search it already answered once, so every failure mode here is
    "type the strategist's spelling again", never an error.
    """
    empty: dict[str, dict[str, str]] = {
        dimension: {} for dimension in _FACET_RESOLUTION_DIMENSIONS
    }
    path = _facet_resolutions_path(state_dir)
    if path is None:
        return empty
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return empty
    except Exception as exc:
        log.warning(
            "facet_name_resolutions: %s is unreadable — treating the cache as "
            "empty (%s: %s)",
            path,
            type(exc).__name__,
            exc,
        )
        return empty
    if not isinstance(raw, dict):
        return empty
    for dimension in _FACET_RESOLUTION_DIMENSIONS:
        entries = raw.get(dimension)
        if not isinstance(entries, dict):
            continue
        empty[dimension] = {
            str(key): str(value)
            for key, value in entries.items()
            if str(key) and str(value)
        }
    return empty


def write_facet_name_resolution(
    state_dir: object,
    dimension: str,
    requested: object,
    landed: object | None,
) -> None:
    """Record (or, with ``landed=None``, forget) one resolution. Never raises."""
    path = _facet_resolutions_path(state_dir)
    if path is None:
        return
    key = facet_name_key(requested)
    if not key or dimension not in _FACET_RESOLUTION_DIMENSIONS:
        return
    cache = read_facet_name_resolutions(state_dir)
    if landed:
        cache[dimension][key] = str(landed)
    elif key not in cache[dimension]:
        return
    else:
        cache[dimension].pop(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning(
            "facet_name_resolutions: could not write %s (%s: %s)",
            path,
            type(exc).__name__,
            exc,
        )


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
    letter really is an initial — 'Tao Z.' abbreviates 'Tao Zhang'. Han,
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
    that has initials ('Tao Z.') and an ellipsis after any token in any script
    ('Tao Zh…'). Shortness alone is not a marker: an unmarked lone letter is a
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


# Per-value wall-clock budget for applying a Location facet. Saved-search restoration
# on a created page is asynchronous and can swallow typing. Retry typing and option
# polling within this budget; a populated list with no exact option still fails fast.
LOCATION_APPLY_BUDGET_SECONDS = 90.0

# Profile panel identity accepts either its class or LinkedIn's data-live-test-profile-slidein marker,
# so class renames cannot blind every wait. _PROFILE_PANEL_SCOPE supplies the same identity
# as an :is() prefix for selectors scoped inside that panel.
PROFILE_PANEL_SELECTOR = (
    "div.profile-slidein__container, div[data-live-test-profile-slidein]"
)
_PROFILE_PANEL_SCOPE = (
    ":is(div.profile-slidein__container, div[data-live-test-profile-slidein])"
)


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
        # Governance attaches to the browser at construction, not to
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
        # Track the page OBJECT we created, or None for an adopted/fallback tab.
        # self._page can change during sidebar operations and recovery; a flag on it
        # would transfer ownership to an adopted tab and leak the page we created.
        # Set ownership only on successful created-page binding. disconnect() closes
        # that exact object and clears it, regardless of later binds.
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
        self.last_facet_reconciliation: dict[str, Any] | None = None
        # The last successfully reconciled company/title target. Set only
        # by reconcile_facets on success; read by reload_search_page's flag-on
        # post-reload verification. None = no string owns the surface yet.
        self._owned_facet_target: dict[str, set[str]] | None = None
        # The scope every company chip must carry (a data-test-pill-action
        # value such as "CURRENT"), or None to leave the facet default.
        self._company_facet_scope: str | None = None
        # Qualification texts the last readiness check removed from the search.
        self.last_removed_qualifications: list[str] = []
        # Lazy view of <state_dir>/facet_name_resolutions.json: the strategist's
        # spelling -> the name LinkedIn's typeahead actually offered. A
        # projection, never canonical state; None = not read yet.
        self._facet_name_resolutions: dict[str, dict[str, str]] | None = None
        # Distinguishes save failure modes: None | "save_trigger_not_found"
        # (class rotation) | "save_not_persisted" (clicked but state unchanged).
        self._last_save_failure_reason: Optional[str] = None
        self._last_note_failure_reason: Optional[str] = None
        # Use a monotonic suffix so same-second forensics captures cannot overwrite each other.
        self._forensics_seq = 0

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
        # Close only the created page object in self._owned_page, never the rebound self._page.
        # This prevents both closing an adopted tab and leaking a created one. Cleanup is
        # best-effort: disconnect must complete even if the page is already gone.
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
            # Attached chrome is not a settled rail. The saved-search restore
            # re-renders it a second or two later, and a run that starts on an
            # already-on-search page never calls navigate_to_search, so this is
            # the only place the rebind path can wait that race out.
            await self._settle_search_rail(page)
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
        # Track this created page object so disconnect closes it even after self._page is rebound.
        # Adopting a tab must never transfer ownership or make that tab eligible for cleanup.
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
        ghost-cursor backend's own commit. The ghost path runs when
        ``LINKEDIN_GHOST_CURSOR_ENABLED`` installs the rebrowser alias that
        ``python_ghost_cursor`` needs for ``playwright.async_api``; this hover
        fallback runs when the flag is off or the cursor is unavailable
        (``ConcurrentInputBackend._cursor`` is None or its move failed).
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

    def set_company_facet_scope(self, scope: str | None) -> None:
        """Pin the scope reconcile_facets puts on, and verifies for, every company chip.

        Set once per run from the brief, before anything reconciles, so the
        opening, a mutation and a recovery replay all hold the same scope.
        """
        self._company_facet_scope = str(scope) if scope else None

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

        # Check for a LinkedIn sign-in wall: the login redirect, the enterprise
        # re-auth page, or the one-session-only page. Named (AuthWallError,
        # captured as auth_wall_<variant>) and never clicked. URL only — this
        # runs every loop, so it reads no body. A /checkpoint/ is left to the
        # health classifier's blocked_or_rate_limited detection backoff.
        from linkedin.auth_wall import raise_if_auth_wall

        await raise_if_auth_wall(
            self, read_body=False, skip_variants=("checkpoint",)
        )
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
            *SEARCH_KEYWORD_CONTROL_SELECTORS,
            *RESULT_COUNT_SELECTORS,
            FILTER_SUMMARY_EDIT_BUTTON,
            FILTER_EDIT_PANEL,
        ]
        for _ in range(12):  # up to ~12 more seconds
            for sel in sidebar_selectors:
                try:
                    visible = await self.page.locator(sel).first.is_visible(timeout=500)
                except Exception:
                    continue
                if visible:
                    # Sidebar rendered — but not necessarily finished. Let the
                    # saved-search restore land before handing the page back,
                    # and only then leave the summary state, so the settle
                    # samples one rail rather than the switch between two.
                    await self._settle_search_rail()
                    try:
                        await self.ensure_filter_edit_panel_open()
                    except Exception as exc:
                        if _is_target_crash_error(exc):
                            raise
                        # Advisory here: every reader and the readiness gate
                        # open the panel again and fail closed on their own.
                        log.warning("navigate_to_search: filter edit panel did not open (%s)", exc)
                    return
            await self.page.wait_for_timeout(500)

        # If we get here, sidebar never appeared. On an unattended run the
        # likeliest cause is a LinkedIn sign-in wall, which every later
        # navigation lands on too; name it and stop, touching nothing. Any
        # other page keeps the generic failure.
        final_url = self.page.url
        from linkedin.auth_wall import raise_if_auth_wall

        await raise_if_auth_wall(self, url=final_url)
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

    async def _rail_signature(self, page=None) -> tuple[str, str, bool] | None:
        """What the left rail looks like right now, or None if it can't be read.

        Three things move when LinkedIn restores a saved search: the URL gains
        a `searchContextId`, the rail re-renders its pills and summary, and the
        keyword editor opens or closes. Any read failure returns None, which
        the caller treats as an unchanged sample — settling is advisory, so it
        must never raise into a navigation.

        `page` names the page to read when it is not (yet) the bound one, which
        is how the created-page bind settles a tab before adopting it.
        """
        try:
            target = self.page if page is None else page
            query = urlparse(target.url).query
            values = parse_qs(query).get("searchContextId") or [""]
            context_id = values[0]
            rail_text = await target.locator("aside.left-rail").first.inner_text(
                timeout=1000
            )
            rail_text = " ".join((rail_text or "").split())[:4000]
            editor_open = await target.locator(
                'textarea[id*="free-text-single-value-input"]'
            ).first.is_visible(timeout=250)
            return (context_id, rail_text, bool(editor_open))
        except Exception:
            return None

    async def _settle_search_rail(self, page=None) -> None:
        """Wait until the left rail stops re-rendering, then return.

        `navigate_to_search` returns on the first visible keyword control, and
        an empty rail carries one, so its return proves nothing about the
        saved-search restore that lands a second or two later and re-renders
        the pills, the summary and the keyword facet. Anything that starts in
        that window loses to the re-render — the location typeahead swallows
        keystrokes and the keyword pencil opens nothing.

        So sample the rail once a second and hand the page back only after
        three consecutive identical samples, or at the 20s cap. One console
        line, fail-soft throughout.

        `page` settles a page that is not the bound one — the created-page
        bind uses it, since nothing is bound until that page is ready.
        """
        started = _monotonic()
        previous: tuple[str, str, bool] | None = None
        stable = 0
        changed = False
        try:
            target = self.page if page is None else page
            while (_monotonic() - started) * 1000 < _RAIL_SETTLE_MAX_MS:
                await target.wait_for_timeout(_RAIL_SETTLE_INTERVAL_MS)
                sample = await self._rail_signature(target)
                if sample is None:
                    sample = previous  # an unreadable rail is an unchanged one
                if previous is not None and sample != previous:
                    changed = True
                    stable = 1
                else:
                    stable += 1
                previous = sample
                if stable >= _RAIL_SETTLE_STABLE_SAMPLES:
                    elapsed = _monotonic() - started
                    detail = (
                        "saved search restored" if changed else "no restore observed"
                    )
                    print(f"  [surface] rail settled in {elapsed:.1f}s ({detail})")
                    return
        except Exception:
            return  # settling is advisory; never fail a navigation over it
        print(
            f"  [surface] rail still changing after "
            f"{(_monotonic() - started):.1f}s; proceeding"
        )

    # One page read that names the rail's state: "edit" (the edit panel shows
    # a facet wrapper), "summary" (the summary Edit button shows), "legacy"
    # (neither state's markers exist: the single-state rail) or "pending"
    # (markers exist but neither state is usable yet).
    _FILTER_RAIL_MODE_JS = """
    () => {
      const shown = el => el.getClientRects().length > 0
        && window.getComputedStyle(el).visibility !== 'hidden';
      const panels = [...document.querySelectorAll('[data-test-filter-edit-panel]')];
      if (panels.some(panel => shown(panel)
          && [...panel.querySelectorAll('div.search-facet-wrapper')].some(shown))) {
        return 'edit';
      }
      const edits = [...document.querySelectorAll('button[data-test-filter-summary-edit]')];
      if (edits.some(shown)) return 'summary';
      if (!panels.length && !edits.length
          && !document.querySelector('[data-test-filter-summary-pills]')) return 'legacy';
      return 'pending';
    }
    """

    async def _filter_rail_mode(self) -> str:
        """The rail's state; an unreadable page reads as the single-state rail.

        Reading "legacy" opens nothing, and every reader after it still fails
        closed on a rail it cannot read, so a failed probe never passes a gate.
        """
        try:
            mode = await self.page.evaluate(self._FILTER_RAIL_MODE_JS)
        except Exception as exc:
            if _is_target_crash_error(exc):
                raise
            return "legacy"
        return mode if mode in ("edit", "summary", "pending") else "legacy"

    async def _await_filter_rail_mode(
        self, wanted: frozenset[str], *, timeout_ms: int
    ) -> str:
        mode = await self._filter_rail_mode()
        waited = 0
        while mode not in wanted and waited < timeout_ms:
            await self.page.wait_for_timeout(_FILTER_RAIL_POLL_MS)
            waited += _FILTER_RAIL_POLL_MS
            mode = await self._filter_rail_mode()
        return mode

    async def ensure_filter_edit_panel_open(
        self, *, timeout_ms: int = _FILTER_RAIL_WAIT_MS
    ) -> None:
        """Leave the summary state for the edit panel, where the facet rail lives.

        Idempotent: an open panel, or a single-state rail, clicks nothing.
        From the summary state it clicks Edit (again after
        ``_FILTER_EDIT_RECLICK_MS`` if the first click did not take) and waits
        for the panel and a facet wrapper to show. A panel that never opens
        raises ``FacetReconciliationError("filter_edit_panel_unopened")``.
        """
        mode = await self._filter_rail_mode()
        waited = 0
        last_click: int | None = None
        while mode not in ("edit", "legacy"):
            if mode == "summary" and (
                last_click is None or waited - last_click >= _FILTER_EDIT_RECLICK_MS
            ):
                await self._ghost_click_locator(
                    self.page.locator(f"{FILTER_SUMMARY_EDIT_BUTTON}:visible").first
                )
                last_click = waited
            if waited >= timeout_ms:
                raise FacetReconciliationError("filter_edit_panel_unopened")
            await self.page.wait_for_timeout(_FILTER_RAIL_POLL_MS)
            waited += _FILTER_RAIL_POLL_MS
            mode = await self._filter_rail_mode()

    async def clear_search_qualifications(self) -> list[str]:
        """Remove every qualification from the search; return their texts.

        Cloris never adds a qualification, so any one present was left on the
        project search and would narrow it unseen. They render in the summary
        state only, so an open edit panel is closed (Back) to read them, each
        is removed and confirmed gone, none may remain, and the edit panel is
        reopened. Any step that cannot be proved raises
        ``FacetReconciliationError`` and the string does not search.
        """
        mode = await self._await_filter_rail_mode(
            frozenset({"edit", "summary", "legacy"}), timeout_ms=_FILTER_RAIL_WAIT_MS
        )
        if mode == "legacy":
            return []
        removed: list[str] = []
        try:
            if mode == "edit":
                back = self.page.locator(f"{_FILTER_EDIT_PANEL_BACK}:visible").first
                if not await back.count():
                    raise FacetReconciliationError("qualifications_unreadable")
                await self._ghost_click_locator(back)
            if await self._await_filter_rail_mode(
                frozenset({"summary"}), timeout_ms=_FILTER_RAIL_WAIT_MS
            ) != "summary":
                raise FacetReconciliationError("qualifications_unreadable")
            items = self.page.locator(f"{_QUALIFICATION_ITEM}:visible")
            while (count := await items.count()):
                item = items.first
                label = item.locator(_QUALIFICATION_TEXT).first
                text = await (label if await label.count() else item).inner_text(timeout=2000)
                await item.hover(timeout=5000)
                await self._ghost_click_locator(item.locator(_QUALIFICATION_REMOVE).first)
                waited = 0
                while await items.count() >= count:
                    if waited >= _QUALIFICATION_REMOVE_WAIT_MS:
                        raise FacetReconciliationError("qualification_removal_failed")
                    await self.page.wait_for_timeout(_FILTER_RAIL_POLL_MS)
                    waited += _FILTER_RAIL_POLL_MS
                removed.append(" ".join(text.split()))
            # Hidden rows count too: a qualification nobody can see still
            # narrows the search.
            if await self.page.locator(_QUALIFICATION_ITEM).count():
                raise FacetReconciliationError("qualification_removal_failed")
        except FacetReconciliationError:
            raise
        except Exception as exc:
            if _is_target_crash_error(exc):
                raise
            raise FacetReconciliationError("qualification_removal_failed") from exc
        if removed:
            print(
                f"  [surface] removed {len(removed)} stale qualification(s): "
                + "; ".join(f'"{text}"' for text in removed)
            )
        await self.ensure_filter_edit_panel_open()
        return removed

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

    async def _read_results_count_once(self, *, timeout_ms: int) -> str:
        """The count portion of the first result-count element present, or ""."""
        for selector in RESULT_COUNT_SELECTORS:
            try:
                element = self.page.locator(selector).first
                if not await element.count():
                    continue
                text = (await element.inner_text(timeout=timeout_ms)).strip()
            except Exception:
                continue
            match = _RESULT_COUNT_RE.search(text)
            if match:
                return match.group(1)
        return ""

    async def _peek_results_count_text(self) -> str:
        return await self._read_results_count_once(timeout_ms=250)

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

    async def _clear_then_reopen_keyword_editor(self, textarea) -> bool:
        """Clear the applied Boolean, then reopen the collapsed Keywords facet.

        On a settled page the pencil (button[aria-label="Edit Profile keywords
        or boolean"], class facet-edit-button) opens the textarea in ~0.12s.
        When it does not — the rail is mid-re-render, or the editor is wedged
        behind an applied Boolean — clearing returns the facet to its collapsed
        "Add Profile keywords or boolean" state, whose control does open the
        editor. Fallback only: the happy path does not clear (96dddc1).

        Clicking Clear runs a keyword-less search. The alternative at this
        point is the run dying on `textarea.wait_for`, as in
        run-20260922-040404.log session 1, run-20260818-191108.log:29-121 and
        run-20260905-082756.log:1034-1048.

        Returns True when the textarea is open. Fail-soft: any exception just
        leaves the caller's `expanded` False and the DOM dump follows.
        """
        try:
            clear_btn = self.page.locator(
                '.keywords-facet button[aria-label="Clear Keywords"], '
                ".keywords-facet [data-test-pill-facet-clear]"
            ).first
            if not await clear_btn.is_visible(timeout=1000):
                return False
            await self._ghost_click_locator(clear_btn)
            # The facet collapses when the clear completes.
            for _ in range(30):
                try:
                    if not await clear_btn.is_visible(timeout=250):
                        break
                except Exception:
                    break
                await self.page.wait_for_timeout(100)
            for selector in (
                'button[aria-label*="Profile keywords"]',
                'button[aria-label*="keywords or boolean" i]',
            ):
                try:
                    btn = self.page.locator(selector).first
                    if not await btn.is_visible(timeout=1000):
                        continue
                    await self._ghost_click_locator(btn)
                except Exception:
                    continue
                if await self._wait_for_visible_poll(
                    textarea, timeout_ms=3000, interval_ms=100
                ):
                    print(
                        "    [keywords] pencil did not open the editor; "
                        "cleared the applied Boolean and reopened it"
                    )
                    return True
            return False
        except Exception:
            return False

    async def enter_search_string(
        self,
        boolean: str,
        *,
        typing_plan: TypingPlan | None = None,
    ) -> SearchEntryResult:
        """Enter a Boolean into the sidebar Keywords field (NOT the global search bar).

        Uses the project search page sidebar: edit button → textarea, and Clear
        is the fallback when the edit control does not open the editor.
        NEVER targets input[aria-label*="keyword"] which is the global search bar.
        The setup steps (edit → fill → Enter) are retried on failure.
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
            # The keyword editor lives in the edit panel, not the summary.
            await self.ensure_filter_edit_panel_open()
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
                                timeout_ms=3000,
                                interval_ms=100,
                            ):
                                expanded = True
                                break
                    except Exception:
                        continue

                if not expanded:
                    expanded = await self._clear_then_reopen_keyword_editor(textarea)

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
        """Apply one or more Location filters with preflight, baseline, retry and verification.

        The pane is aside.left-rail. Its typeahead placeholder contains "location";
        options are li[role=option] in ul.artdeco-typeahead__results-list[role=listbox].
        Applied pills use li[data-test-facet-pills-item] and may carry data-test-pill-label.
        Confirm the visible pill, not its hover-gated "Remove {value}" control.
        Selectors are documented in docs/linkedin-recruiter-dom-map.md.

        Any missing editor, exact option or confirmed pill returns False for keyword fallback.
        The caller records raises as failed_control. Match the requested option exactly;
        never select a broad suggestion or press Enter.

        Clear each attempt. A fresh editor may not yet have a typeahead listener, and
        saved-search restoration may swallow typing or apply the requested value mid-flight.
        Budget each value with LOCATION_APPLY_BUDGET_SECONDS, verify the typed text,
        and re-check the confirmed pill on every pass. Applied values are excluded from
        suggestions, so once that pill exists, typing cannot recover the value.
        A populated list without the exact row is a facet-name mismatch: fail fast with
        offered-options evidence for Stage-B resolution rather than spending the whole budget.

        temporal_scope is accepted for signature parity and upstream routing. It never
        selects the dropdown: Location always uses "Current or past", never current-only.
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
        pill_wrapper = rail.locator("div.search-facet-wrapper.facet-locations")
        editor_input = rail.locator(
            'input.artdeco-typeahead__input[placeholder*="location" i]'
        ).first

        applied: list[str] = []
        # Existing sidebar chips satisfy their values without typing. Applied values
        # are excluded from typeahead suggestions, so retyping them cannot find an
        # exact match and would exhaust the retry budget.
        try:
            pre_applied_chips = await self.read_applied_location_chips()
        except Exception:
            pre_applied_chips = None
        if pre_applied_chips is None:
            log.warning("apply_location_filter: applied Locations facet is unreadable")
            return False
        pre_applied = {
            normalize_facet_value_for_compare(chip) for chip in pre_applied_chips
        }
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
                # Re-check the confirmed pill on every pass: saved-search restoration can apply
                # the value after the pre-read. Once applied, it is excluded from suggestions,
                # so the existing pill is the only successful exit.
                async def _clear_editor_best_effort() -> None:
                    try:
                        if await editor_input.is_visible(timeout=250):
                            await self._clear_keyword_textarea(editor_input)
                    except Exception:
                        pass

                deadline = time.monotonic() + LOCATION_APPLY_BUDGET_SECONDS
                attempt = 0
                nonempty_option_misses = 0
                offered: list[str] = []
                while True:
                    attempt += 1
                    if await self._wait_for_facet_pill(
                        pill_wrapper,
                        value,
                        timeout_ms=300,
                        interval_ms=150,
                    ):
                        log.info(
                            "apply_location_filter[%r]: gate=chip_materialized — the "
                            "value's pill is confirmed on the sidebar (saved-search "
                            "restore or a prior half-apply); accepting without typing.",
                            value,
                        )
                        self.last_location_already_applied_count += 1
                        return True
                    try:
                        await self._clear_keyword_textarea(editor_input)
                    except RuntimeError as exc:
                        log.warning(
                            "apply_location_filter[%r]: gate=editor_clear miss — %s",
                            value,
                            exc,
                        )
                        await _clear_editor_best_effort()
                        return False
                    await self._input_backend.type_text(
                        self.page, editor_input, value, plan=build_boolean_typing_plan(value)
                    )
                    typed_back = (await self._read_textarea_value(editor_input)).strip()
                    if typed_back != value:
                        # Before hydration, the editor can swallow keystrokes and trigger no query.
                        # Retry typing before polling an option list whose listener received nothing.
                        if time.monotonic() < deadline:
                            log.warning(
                                "apply_location_filter[%r]: gate=typing_swallowed — "
                                "editor readback %r after typing (attempt %d); "
                                "retrying.",
                                value,
                                typed_back[:40],
                                attempt,
                            )
                            await asyncio.sleep(1.0)
                            continue
                    else:
                        option_visible = await self._wait_for_visible_poll(
                            option, timeout_ms=4000, interval_ms=150
                        )
                        if option_visible:
                            await self._ghost_click_locator(option)
                            # Confirm via the visible pill item, not the
                            # hover-gated dismiss button.
                            landed = await self._wait_for_facet_pill(
                                pill_wrapper,
                                value,
                                timeout_ms=2500,
                                interval_ms=120,
                            )
                            if not landed:
                                try:
                                    landed_pills = await rail.locator(
                                        "li[data-test-facet-pills-item]"
                                    ).all_inner_texts()
                                except Exception:
                                    landed_pills = []
                                log.warning(
                                    "apply_location_filter[%r]: gate=chip miss — exact "
                                    "option clicked but no 'Remove %s' pill confirmed. "
                                    "Applied pills: %r",
                                    value, value, landed_pills[:12],
                                )
                            return landed
                        try:
                            offered = await option_list.all_inner_texts()
                        except Exception:
                            offered = []
                        if offered:
                            # A populated list without the exact value is a facet-name mismatch.
                            # Allow one extra draw for nondeterministic ranking, then fail fast with
                            # offered-options evidence for Stage-B resolution.
                            nonempty_option_misses += 1
                            if nonempty_option_misses >= 2:
                                break
                        # An EMPTY list is ambiguous: the typeahead may not be
                        # hydrated yet, or the value is already applied and
                        # excluded from suggestions with its pill not yet drawn
                        # — the loop-head chip check catches the latter on the
                        # next pass.
                    if time.monotonic() >= deadline:
                        break
                    log.warning(
                        "apply_location_filter[%r]: gate=exact_match retry — no option "
                        "matched ^value$ on attempt %d (%.0fs of budget left); "
                        "clearing and retrying.",
                        value,
                        attempt,
                        max(0.0, deadline - time.monotonic()),
                    )
                    await asyncio.sleep(0.8 + min(2.0, 0.4 * attempt))
                # Diagnostic gate 2 (option/exact-match): capture the option
                # texts actually offered so a seat run reveals whether this is
                # selector rotation, an empty list, or a value needing
                # country-aware normalization (e.g. "Colombia, Colombia").
                self.last_location_option_misses[value] = [
                    str(text).strip() for text in offered[:12] if str(text).strip()
                ]
                log.warning(
                    "apply_location_filter[%r]: gate=exact_match miss — no option "
                    "matched ^value$ within budget. Offered options: %r",
                    value,
                    offered[:12],
                )
                await _clear_editor_best_effort()
                return False

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
        # Location always uses "Current or past"; temporal_scope is only an upstream routing signal.
        # The facet default already supplies that scope, so do not manipulate its dropdown.
        # Chip confirmation is primary; a changed results list is the secondary gate.
        await self._wait_for_search_results_ready(
            previous_count_text=previous_count_text,
            previous_top_card_signature=previous_top_card_signature,
        )
        return True

    async def _read_facet_pill_scope(self, item) -> str | None:
        """One chip's scope action from its visible label; ``None`` means unreadable.

        A label suffix is that scope and a bare label is the default. Its
        aria-labels go stale (a chip set back to default kept "Remove
        AfterQuery (Current)" live), so they are read only for a chip with no
        label node, as ``_read_facet_pill_value`` does.
        """
        label_node = item.locator("[data-test-pill-label]").first
        if await label_node.count():
            label = (await label_node.inner_text()).strip()
            return (split_facet_scope(label)[1] or FACET_SCOPE_DEFAULT) if label else None
        labels = await item.evaluate(
            "item => [item, ...item.querySelectorAll('[aria-label]')]"
            ".map(node => node.getAttribute('aria-label') || '')"
        )
        scopes = [split_facet_scope(text)[1] for text in labels or []]
        return next((scope for scope in scopes if scope), FACET_SCOPE_DEFAULT)

    async def _read_facet_pill_value(self, item) -> str | None:
        label_node = item.locator("[data-test-pill-label]").first
        value = ""
        if await label_node.count():
            value = (await label_node.inner_text()).strip()
        if not value:
            dismiss = item.locator("button[data-test-pill-dismiss]").first
            aria_label = await dismiss.get_attribute("aria-label")
            if aria_label and aria_label.startswith("Remove "):
                value = aria_label[len("Remove "):].strip()
        return value or None

    async def _find_facet_pill_item(self, wrapper, value: str, *, strip_scope: bool = False):
        """Find one visible pill by its rendered label, never its aria-label.

        ``strip_scope`` compares the label without its scope suffix, for the
        company and title chips that carry one.
        """
        target = normalize_facet_value_for_compare(
            split_facet_scope(value)[0] if strip_scope else value
        )
        matches = []
        items = wrapper.locator("li[data-test-facet-pills-item]")
        for index in range(await items.count()):
            item = items.nth(index)
            if not await item.is_visible():
                continue
            label = item.locator("[data-test-pill-label]").first
            observed = (
                (await label.inner_text()).strip()
                if await label.count()
                else await self._read_facet_pill_value(item)
            )
            if strip_scope:
                observed = split_facet_scope(observed)[0]
            if normalize_facet_value_for_compare(observed) == target:
                matches.append(item)
        return matches[0] if len(matches) == 1 else None

    async def _wait_for_facet_pill(
        self,
        wrapper,
        value: str,
        *,
        timeout_ms: int,
        interval_ms: int,
        strip_scope: bool = False,
    ) -> bool:
        attempts = max(1, timeout_ms // interval_ms)
        for _ in range(attempts):
            if await self._find_facet_pill_item(
                wrapper, value, strip_scope=strip_scope
            ) is not None:
                return True
            await self.page.wait_for_timeout(interval_ms)
        return await self._find_facet_pill_item(
            wrapper, value, strip_scope=strip_scope
        ) is not None

    # F9b. Every applied pill on the live rail belongs to some dimension's
    # div.search-facet-wrapper.facet-<name> — 32 of 32 did on the contaminated
    # PRRE-Brazil rail (plans/facet-lifecycle-clearing-plan.md A3). A pill that
    # belongs to none of them means the rail's shape has rotated out from under
    # this reader, and no dimension on it can be honestly called empty. The
    # dimension name is deliberately NOT matched against the three Cloris
    # manages: an operator-applied facet (a manual 8-30+ YOE band) carries its
    # own facet-<name> wrapper and must not fail a run closed.
    #
    # Only the facet rail is scanned. Summary pills and qualification rows are
    # not facet pills, and when the edit panel exists it IS the facet rail, so
    # a pill outside it (a saved-filter preset, the summary) is not one either.
    _UNATTRIBUTED_RAIL_PILLS_JS = """
    rail => {
      const panels = Array.from(
        document.querySelectorAll('[data-test-filter-edit-panel]'));
      return Array.from(
        rail.querySelectorAll('li[data-test-facet-pills-item]')
      ).filter(item => {
        if (item.getClientRects().length === 0) return false;
        if (window.getComputedStyle(item).visibility === 'hidden') return false;
        if (item.closest('[data-test-filter-summary-pills], '
            + '[data-test-qualifications-summary-row], [data-test-qualification-item]')) {
            return false;
        }
        if (panels.length && !panels.some(panel => panel.contains(item))) return false;
        const wrapper = item.closest('div.search-facet-wrapper');
        if (!wrapper) return true;
        return !Array.from(wrapper.classList).some(
            cls => cls.startsWith('facet-') && cls.length > 'facet-'.length
        );
      }).map(item => {
        const dismiss = item.querySelector('button[data-test-pill-dismiss]');
        return (dismiss && dismiss.getAttribute('aria-label')) || '<unlabelled pill>';
      });
    }
    """

    async def _unattributed_rail_pill_labels(self, rail) -> list[str] | None:
        """Dismiss labels of visible rail pills owned by no dimension wrapper.

        ``None`` means the scan itself could not run, which fails closed the
        same way a non-empty result does — an unverifiable rail is not a clean
        one. One page evaluate, so this stays cheap enough to run on every read
        including the settle polls.
        """
        try:
            return await rail.first.evaluate(self._UNATTRIBUTED_RAIL_PILLS_JS)
        except Exception:
            return None

    async def _read_applied_facet_pills(
        self, dimension: str, *, with_scope: bool = False
    ) -> list[tuple[str, str | None]] | None:
        """Read one wrapper-scoped facet as (label, scope); ``None`` means unreadable.

        The label is the chip's verbatim text, scope suffix included. The
        scope is read only ``with_scope`` (None otherwise); one chip whose
        scope is unreadable makes the facet unreadable. The rail is moved
        to its edit panel first: in the summary state there is nothing here to
        read, and a panel that will not open is an unreadable rail.
        """
        try:
            await self.ensure_filter_edit_panel_open()
        except Exception as exc:
            log.warning(
                "read_applied_%s_chips: filter edit panel unavailable (%s)",
                dimension,
                exc,
            )
            return None
        rail = self.page.locator("aside.left-rail")
        wrapper = rail.locator(
            f"div.search-facet-wrapper.facet-{dimension}"
        )
        try:
            wrapper_count = await wrapper.count()
            if wrapper_count == 0:
                add_control = rail.locator(
                    {
                        "locations": 'button:has-text("geographic location")',
                        "companies": 'button:has-text("Companies or boolean")',
                        "titles": 'button:has-text("Job titles or boolean")',
                    }[dimension]
                )
                if not (
                    await rail.count() == 1
                    and await rail.is_visible()
                    and await add_control.count()
                ):
                    return None
                pills: list[tuple[str, str | None]] = []
            elif wrapper_count != 1:
                return None
            else:
                items = wrapper.locator("li[data-test-facet-pills-item]")
                pills = []
                for i in range(await items.count()):
                    item = items.nth(i)
                    if not await item.is_visible():
                        continue
                    value = await self._read_facet_pill_value(item)
                    scope = (
                        await self._read_facet_pill_scope(item) if with_scope else None
                    )
                    if not value or (with_scope and scope is None):
                        return None
                    pills.append((value, scope))
        except Exception:
            return None
        unattributed = await self._unattributed_rail_pill_labels(rail)
        if unattributed is None or unattributed:
            log.warning(
                "read_applied_%s_chips: rail unreadable — %s pill(s) belong to "
                "no facet-<dimension> wrapper: %r",
                {
                    "locations": "location",
                    "companies": "company",
                    "titles": "title",
                }[dimension],
                "unscannable" if unattributed is None else len(unattributed),
                (unattributed or [])[:8],
            )
            return None
        return pills

    async def _read_applied_facet_chips(self, dimension: str) -> list[str] | None:
        """Read one wrapper-scoped facet's names; ``None`` means unreadable.

        Company and title names come without their scope suffix, which
        ``read_applied_company_chip_scopes`` reports separately.
        """
        pills = await self._read_applied_facet_pills(dimension)
        if pills is None:
            return None
        if dimension in _SCOPED_FACET_DIMENSIONS:
            return [split_facet_scope(label)[0] for label, _ in pills]
        return [label for label, _ in pills]

    async def read_applied_company_chip_scopes(self) -> dict[str, str] | None:
        """Company chip name -> its scope action; ``None`` means unreadable."""
        pills = await self._read_applied_facet_pills("companies", with_scope=True)
        if pills is None:
            return None
        return {
            split_facet_scope(label)[0]: scope or FACET_SCOPE_DEFAULT
            for label, scope in pills
        }

    async def _visible_pill_action(self, action: str, *, timeout_ms: int = 3000):
        """The one visible pill-menu item for ``action``, or None."""
        options = self.page.locator(f'[data-test-pill-action="{action}"]:visible')
        waited = 0
        while True:
            if await options.count() == 1:
                return options.first
            if waited >= timeout_ms:
                return None
            await self.page.wait_for_timeout(_FILTER_RAIL_POLL_MS)
            waited += _FILTER_RAIL_POLL_MS

    async def set_company_chip_scope(self, name: str, scope: str) -> bool:
        """Set one company chip's scope through its pill menu and read it back.

        Hover the chip, open its "..." menu
        (``button[data-test-pill-actions-dropdown-trigger]``), click the
        ``[data-test-pill-action=<scope>]`` item, then poll the chip's label
        until it names ``scope``. True only on that read-back; any miss closes
        the menu and returns False, and the caller decides what a chip without
        its scope means.
        """
        wrapper = self.page.locator(
            "aside.left-rail div.search-facet-wrapper.facet-companies"
        )
        try:
            if await wrapper.count() != 1:
                return False
            item = await self._find_facet_pill_item(wrapper, name, strip_scope=True)
            if item is None:
                return False
            await item.hover(timeout=5000)
            trigger = item.locator(_PILL_ACTIONS_TRIGGER).first
            if not await trigger.count():
                aria = json.dumps(f"Dropdown menu for updating {name}", ensure_ascii=False)
                trigger = self.page.locator(f"{_PILL_ACTIONS_TRIGGER}[aria-label={aria}]").first
                if not await trigger.count():
                    return False
            await self._ghost_click_locator(trigger)
            option = await self._visible_pill_action(scope)
            if option is None:
                await self._press_key("Escape")
                return False
            if (await option.get_attribute("aria-checked")) == "true":
                await self._press_key("Escape")
            else:
                await self._ghost_click_locator(option)
        except Exception as exc:
            if _is_target_crash_error(exc):
                raise
            try:
                await self._press_key("Escape")
            except Exception:
                pass
            return False
        key = facet_name_key(name)
        for attempt in range(FACET_SETTLE_ATTEMPTS):
            scopes = await self.read_applied_company_chip_scopes()
            if scopes is not None and any(
                facet_name_key(chip) == key and observed == scope
                for chip, observed in scopes.items()
            ):
                return True
            if attempt < FACET_SETTLE_ATTEMPTS - 1:
                await self.page.wait_for_timeout(FACET_SETTLE_INTERVAL_MS)
        log.warning(
            "set_company_chip_scope[%r]: scope %s did not read back on the chip",
            name,
            scope,
        )
        return False

    async def read_applied_location_chips(self) -> list[str] | None:
        return await self._read_applied_facet_chips("locations")

    async def read_applied_company_chips(self) -> list[str] | None:
        return await self._read_applied_facet_chips("companies")

    async def assert_search_surface_ready(self, *, timeout_ms: int = 15_000) -> None:
        """Wait for the hydrated rail plus somewhere to put this string's search.

        The gate exists for the sidebar work that follows it — reconcile the
        facets, verify geography, type the Boolean — so it asserts what that
        work needs: the rail, and any one of a result summary, a result card,
        or a keyword control. Results are the PREVIOUS string's business, and
        demanding them made LinkedIn's no-results page permanently unready:
        the page a zero-result string leaves behind carries the rail and the
        Keywords box but neither a summary nor a card, so the gate refused it
        for every string that followed and three refusals in a row against one
        unchanged page ended the 09-20 Brazil run
        (plans/facet-lifecycle-clearing-plan.md Addendum I2).

        Every element that decides the verdict must be VISIBLE, and a keyword
        control must also be INSIDE the rail — the same bar
        `navigate_to_search` polls. `count()` is existence, and existence made
        the gate pass on pages that are not this project's search: a
        non-search Recruiter view carries its own keywords button,
        `[aria-label*="keywords" i]` is a case-insensitive substring, and a
        hidden rail node plus a hidden summary or card read as ready, after
        which the opening typed this project's Boolean into a foreign
        document. What is still accepted is the ABSENCE of a summary and of
        cards, which is the no-results page.

        Visible elements are not enough on their own, because a Recruiter
        search rail is a rail wherever it belongs. The URL must also be a
        Recruiter SEARCH view AND provably this project's, by the same two
        predicates run-start applies before it navigates — otherwise another
        project's search page, or this project's pipeline view, reads as ready
        and the fresh-string path, which never checks the URL itself, rewrites
        geography and types this string's Boolean into that document. That
        refusal is named `foreign_search_page` rather than
        `search_surface_unready`: both are live gates the loop recovers from by
        re-anchoring, and the name is what the receipt and the page capture
        carry. The URL is re-read on every poll, so a navigation that lands
        during the wait is honored.

        On the two-state rail the gate also prepares the rail, once: every
        qualification left on the search is removed
        (``clear_search_qualifications``, texts kept in
        ``last_removed_qualifications``) and the edit panel is opened.
        """
        # Imported here because `linkedin.evaluation_support` imports
        # `_recruiter_page_project_id` from this module — the cycle only runs
        # that direction at import time.
        from linkedin.evaluation_support import (
            _is_foreign_project_page,
            _is_recruiter_search_page,
        )

        rail = self.page.locator("aside.left-rail")
        qualifying = [
            *(self.page.locator(selector) for selector in RESULT_COUNT_SELECTORS),
            self.page.locator("ol.profile-list article.profile-list-item"),
            *(
                rail.locator(selector)
                for selector in SEARCH_KEYWORD_CONTROL_SELECTORS
            ),
        ]

        async def _shown(locator) -> bool:
            return bool(await locator.count()) and await locator.first.is_visible()

        def _on_this_projects_search() -> bool:
            try:
                url = self.page.url
            except Exception:
                # Fail CLOSED: "could not read the page" is the one unverified
                # state that must not be the one that passes.
                return False
            return _is_recruiter_search_page(url) and not _is_foreign_project_page(
                url, self._required_project_id
            )

        interval_ms = 150
        waited_ms = 0
        on_this_search = False
        rail_prepared = False
        self.last_removed_qualifications = []
        while True:
            on_this_search = _on_this_projects_search()
            if on_this_search and not rail_prepared:
                # Once, on this project's search: clear any qualification left
                # on the search and open the edit panel. Either one failing
                # raises FacetReconciliationError and the string never searches.
                rail_prepared = True
                self.last_removed_qualifications = (
                    await self.clear_search_qualifications()
                )
            if on_this_search:
                try:
                    if await _shown(rail):
                        for candidate in qualifying:
                            if await _shown(candidate):
                                return
                except Exception:
                    pass
            if waited_ms >= timeout_ms:
                break
            wait_ms = min(interval_ms, timeout_ms - waited_ms)
            await self.page.wait_for_timeout(wait_ms)
            waited_ms += wait_ms
        raise FacetReconciliationError(
            "search_surface_unready" if on_this_search else "foreign_search_page"
        )

    async def remove_facet_value(self, dimension: str, value: str) -> bool:
        """Hover-dismiss one exact observed managed facet pill."""
        if dimension not in {"companies", "locations", "titles"} or not value:
            return False
        wrapper = self.page.locator(
            f"aside.left-rail div.search-facet-wrapper.facet-{dimension}"
        )
        try:
            if await wrapper.count() != 1:
                return False
            item = await self._find_facet_pill_item(
                wrapper, value, strip_scope=dimension in _SCOPED_FACET_DIMENSIONS
            )
            if item is None:
                return False
            dismiss = item.locator("button[data-test-pill-dismiss]").first
            await item.hover(timeout=5000)
            await self._ghost_click_locator(dismiss)
        except Exception:
            return False

        for _ in range(FACET_SETTLE_ATTEMPTS):
            try:
                observed = await self._read_applied_facet_chips(dimension)
                if observed is not None and normalize_facet_value_for_compare(
                    value
                ) not in {
                    normalize_facet_value_for_compare(item) for item in observed
                }:
                    return True
            except Exception:
                return False
            await self.page.wait_for_timeout(FACET_SETTLE_INTERVAL_MS)
        return False

    # ------------------------------------------------------------------
    # Facet name resolution cache (a state-dir sidecar projection)
    # ------------------------------------------------------------------

    def _facet_resolution_cache(self) -> dict[str, dict[str, str]]:
        if self._facet_name_resolutions is None:
            self._facet_name_resolutions = read_facet_name_resolutions(
                self._state_dir
            )
        return self._facet_name_resolutions

    def _facet_resolution_cache_get(
        self, dimension: str, requested: object
    ) -> str | None:
        """LinkedIn's spelling for this request, if a previous string found it."""
        key = facet_name_key(requested)
        if not key:
            return None
        return self._facet_resolution_cache().get(dimension, {}).get(key) or None

    def _facet_resolution_cache_put(
        self, dimension: str, requested: object, landed: object
    ) -> None:
        key = facet_name_key(requested)
        if not key or not landed or dimension not in _FACET_RESOLUTION_DIMENSIONS:
            return
        cache = self._facet_resolution_cache().setdefault(dimension, {})
        if cache.get(key) == str(landed):
            return
        cache[key] = str(landed)
        write_facet_name_resolution(self._state_dir, dimension, requested, landed)

    def _facet_resolution_cache_forget(
        self, dimension: str, requested: object
    ) -> None:
        """Drop a resolution the typeahead has stopped honoring."""
        key = facet_name_key(requested)
        if not key:
            return
        cache = self._facet_resolution_cache().get(dimension) or {}
        if key not in cache:
            return
        cache.pop(key)
        write_facet_name_resolution(self._state_dir, dimension, requested, None)

    async def _take_facet_typeahead_option(
        self,
        *,
        requested: str,
        candidate: str,
        option_list,
        editor_input,
        pill_wrapper,
        rail,
        retries: int,
        log_prefix: str,
    ) -> tuple[str, str | list[str] | None]:
        """Type one name into a facet typeahead and take the option the rule picks.

        Shared verbatim by ``apply_company_filter`` and ``apply_title_filter``
        so the two byte-identical siblings cannot drift on the match rule.
        Returns one of:

        - ``("landed", <the option's visible text>)`` — the pill is confirmed
          under THAT text, which is the only name the rail carries.
        - ``("miss", <offered option texts>)`` — the listbox answered and
          nothing satisfied the request, or it never rendered an option.
        - ``("chip_miss", <offered option texts>)`` — an option was clicked
          but no pill confirmed. The rail may have moved under us, so no
          further spelling is tried for this value.
        - ``("unusable", None)`` — the editor itself could not be cleared.

        The retry budget belongs to an EMPTY listbox. A listbox that renders
        options and satisfies nothing has ANSWERED; retyping the same string
        into it four times only spends the budget to reach the same answer.

        A NEAR MISS may only be taken when the row the rule picks is a name
        the listbox was NOT already offering. The previous value's rows are
        still on the page when the visibility poll fires, and under the
        near-miss rule they can satisfy a name they were never offered for: a
        listbox still showing ["Banco", "Banco Inter"] answers "Banco do
        Brasil" on the prefix rule, and the wrong pill is then confirmed,
        cached and retained by every later string.

        The baseline those names are measured against is read on both sides of
        the clear, because clearing the editor can close the typeahead: read
        only after it, the baseline is empty and the old rows flashing back
        look new. What the baseline holds is entities, not raw row text — a
        reordered list and a ticking employee count are the same rows — and an
        EMPTY read is not evidence about this name in either direction, so it
        neither proves the repaint nor denies it.

        The repaint wait is how LinkedIn is given time to answer; it is not
        what licenses the take. An EXACT row is taken from any list, waiting
        for nothing: it cannot be the right answer to one query and the wrong
        answer to this one.
        """

        async def _clear_editor_best_effort() -> None:
            try:
                if await editor_input.is_visible(timeout=250):
                    await self._clear_keyword_textarea(editor_input)
            except Exception:
                pass

        async def _read_offered() -> list[str]:
            try:
                return await option_list.all_inner_texts()
            except Exception:
                return []

        def _is_exact(resolution: tuple[int, str] | None) -> bool:
            return resolution is not None and facet_name_key(
                resolution[1]
            ) == facet_name_key(requested)

        def _option_keys(texts: list[str]) -> set[str]:
            """The entities one listbox read offers, folded, as a set.

            What decides whether a list is THIS name's answer is which
            entities are on it, not the raw row text: ``all_inner_texts``
            carries a secondary line (an industry, an employee count) that
            ticks over on its own, and a typeahead reorders the rows it
            already had. Either one made a raw inequality read as a repaint,
            and the previous value's rows were then taken for this name.
            Folded with ``facet_name_key``, the same fold the match rule
            applies, so the set and the row it picks are compared on one
            spelling.
            """
            return {
                key
                for key in (
                    facet_name_key(_facet_option_name(text)) for text in texts
                )
                if key
            }

        def _is_new_name(resolution: tuple[int, str] | None) -> bool:
            """Whether the row the rule picked is one the baseline did not hold."""
            return (
                resolution is not None
                and facet_name_key(resolution[1]) not in before_keys
            )

        budget = max(1, retries)
        populated = False
        # Every entity on the page BEFORE this name was typed, read on BOTH
        # sides of the clear and unioned. Clearing the editor can CLOSE the
        # typeahead, so a baseline read only after it is empty — and against an
        # empty baseline the previous value's rows flashing back are new names,
        # which is how "Banco" was taken, confirmed and cached for "Banco do
        # Brasil". Nothing already in here is evidence about this name.
        before_keys: set[str] = set()
        for attempt in range(1, budget + 1):
            before_keys = _option_keys(await _read_offered())
            try:
                await self._clear_keyword_textarea(editor_input)
            except RuntimeError as exc:
                log.warning(
                    "%s: gate=editor_clear miss — %s", log_prefix, exc
                )
                await _clear_editor_best_effort()
                return "unusable", None
            before_keys |= _option_keys(await _read_offered())
            await self._input_backend.type_text(
                self.page,
                editor_input,
                candidate,
                plan=build_boolean_typing_plan(candidate),
            )
            populated = await self._wait_for_visible_poll(
                option_list.first, timeout_ms=4000, interval_ms=150
            )
            if populated:
                break
            if attempt < budget:
                log.warning(
                    "%s: gate=empty_listbox retry — no option rendered for %r "
                    "on attempt %d/%d; clearing and retrying.",
                    log_prefix,
                    candidate,
                    attempt,
                    budget,
                )
                await asyncio.sleep(0.8 + 0.4 * (attempt + 1))

        # An empty listbox still gets its texts read: whatever the DOM holds is
        # the operator's only diagnostic for why the name never landed.
        offered = await _read_offered()
        resolved = resolve_facet_option(requested, offered) if populated else None
        if populated and not _is_exact(resolved) and not _is_new_name(resolved):
            # Either nothing on this list satisfies the request, or the only
            # thing that does is a row the page was already offering before
            # this name was typed. Give LinkedIn the repaint rather than
            # deciding on the previous value's answer.
            waited_ms = 0
            while waited_ms < _FACET_LISTBOX_REPAINT_TIMEOUT_MS:
                await asyncio.sleep(_FACET_LISTBOX_REPAINT_INTERVAL_MS / 1000)
                waited_ms += _FACET_LISTBOX_REPAINT_INTERVAL_MS
                latest = await _read_offered()
                if not latest:
                    # A listbox that unmounts mid-repaint reads empty, and the
                    # rows that come back may be the ones it already had. An
                    # empty read says nothing about this name, so it neither
                    # ends the wait nor qualifies the read after it.
                    continue
                if _option_keys(latest) != before_keys:
                    offered = latest
                    break
            resolved = resolve_facet_option(requested, offered)
        if resolved is not None and not _is_exact(resolved) and not _is_new_name(
            resolved
        ):
            log.warning(
                "%s: gate=stale_listbox miss — %r near-matched %r, a name the "
                "listbox was already offering before it was typed. Offered "
                "options: %r",
                log_prefix,
                requested,
                resolved[1],
                offered[:12],
            )
            resolved = None
        names = [
            str(text).strip() for text in offered[:12] if str(text).strip()
        ]
        if resolved is None:
            log.warning(
                "%s: gate=%s miss — no option satisfied %r (typed %r). "
                "Offered options: %r",
                log_prefix,
                "empty_listbox" if not populated else "name_match",
                requested,
                candidate,
                offered[:12],
            )
            await _clear_editor_best_effort()
            return "miss", names

        index, landed = resolved
        await self._ghost_click_locator(option_list.nth(index))
        # Confirm the applied chip landed (PRIMARY success gate). Pill renders
        # as li[data-test-facet-pills-item] with a hover-gated dismiss control
        # whose accessible name is "Remove {value}" (2026-06 capture); confirm
        # the pill ITEM is visible rather than the hover-gated button. The text
        # to wait for is the LANDED option, not the request: on a near miss
        # the requested spelling never appears on the rail.
        if await self._wait_for_facet_pill(
            pill_wrapper, landed, timeout_ms=2500, interval_ms=120, strip_scope=True
        ):
            return "landed", landed

        try:
            landed_pills = await rail.locator(
                "li[data-test-facet-pills-item]"
            ).all_inner_texts()
        except Exception:
            landed_pills = []
        log.warning(
            "%s: gate=chip miss — option %r clicked for %r but no pill "
            "confirmed. Applied pills: %r",
            log_prefix,
            landed,
            requested,
            landed_pills[:12],
        )
        return "chip_miss", names

    async def reconcile_facets(
        self,
        *,
        companies: set[str],
        titles: set[str],
        verify_only: bool = False,
    ) -> None:
        """Make string-owned Company and Job title facets equal their targets.

        The add phase applies ONE value per applier call. An exact-match miss
        is a property of one value — LinkedIn offers "Tess"/"TESS AI Quality"
        for a strategist's "Tess AI" — and the appliers return False for the
        whole batch if any single value misses, so a batched add made one
        misspelt company fail the entire string. A missed value is recorded
        under ``receipt["unapplied"][dimension]`` and reconciliation
        continues; the postcondition is then ``live == desired - unapplied``
        per dimension, which is also what the owned target records. Removal
        stays fail-closed: a foreign pill that will not leave is
        ``remove_failed``, because running inside someone else's pills is the
        contamination this whole feature exists to prevent.

        A value the appliers take as a NEAR MISS ("Mercado Livre" applied as
        LinkedIn's "Mercado Livre Brasil") is recorded under
        ``receipt["substituted"][dimension]`` as requested -> landed. The
        postcondition, the owned target and the retain/remove diff all compare
        against the LANDED name, because that is the only name the rail
        carries. ``verify_only`` never substitutes: its callers already pass
        landed names, via ``facet_target_landed``.

        With a company scope pinned (``set_company_facet_scope``) every
        company chip must also read back that scope. The add phase sets it
        through each chip's pill menu; a chip that will not take it is
        dismissed and its value recorded ``unapplied`` (and under
        ``receipt["scope_unverified"]``), exactly like a value the typeahead
        refused, so no string searches a company on the default
        current-or-past scope. ``verify_only`` refuses a wrong scope as
        ``company_scope_mismatch``.
        """
        self.last_facet_reconciliation = None
        company_scope = getattr(self, "_company_facet_scope", None)
        # Folded company name -> the scope its chip read back, refreshed by
        # every snapshot while a scope is pinned.
        observed_scopes: dict[str, str] = {}
        scope_unverified: list[str] = []
        dimensions = ("companies", "titles")
        readers = {
            "companies": self.read_applied_company_chips,
            "titles": self.read_applied_title_chips,
        }
        appliers = {
            "companies": self.apply_company_filter,
            "titles": self.apply_title_filter,
        }
        substitution_attrs = {
            "companies": "last_company_substitutions",
            "titles": "last_title_substitutions",
        }

        def _index(values: list[str] | set[str], *, gate: str) -> dict[str, str]:
            """Folded key -> the verbatim name, for one set of facet values.

            The key is ``facet_name_key``, the same fold the typeahead match
            rule uses, so "Itaú" on the rail and a requested "Itau" are one
            entity here too. Under a plain casefold the correct pill read as
            foreign and the remove phase dismissed it. The receipt keeps the
            verbatim text, which is what the rail carries and the operator
            reads.
            """
            indexed: dict[str, str] = {}
            for raw in sorted(values, key=lambda item: str(item).casefold()):
                key = facet_name_key(raw)
                if not key:
                    raise FacetReconciliationError(f"{gate}_blank_value")
                if key in indexed:
                    raise FacetReconciliationError(
                        f"{gate}_normalized_collision"
                    )
                indexed[key] = raw
            return indexed

        desired = {
            "companies": _index(companies, gate="companies_desired"),
            "titles": _index(titles, gate="titles_desired"),
        }
        # Values whose applier refused them. Recorded, never raised.
        unapplied: dict[str, dict[str, str]] = {
            dimension: {} for dimension in dimensions
        }
        # Requested normalized key -> the name LinkedIn actually applies for
        # it. Seeded from the resolution sidecar so a second string RETAINS
        # the pill the first one landed instead of dismissing it as foreign
        # and typing it again; the appliers are authoritative and overwrite
        # or drop each entry as they run. Never seeded for verify_only.
        substituted: dict[str, dict[str, str]] = {
            dimension: {} for dimension in dimensions
        }
        if not verify_only:
            for dimension in dimensions:
                for key, raw in desired[dimension].items():
                    cached = self._facet_resolution_cache_get(dimension, raw)
                    if cached and facet_name_key(cached) != key:
                        substituted[dimension][key] = cached

        def _landed_key(dimension: str, key: str) -> str:
            landed = substituted[dimension].get(key)
            return facet_name_key(landed) if landed else key

        def _target(dimension: str) -> dict[str, str]:
            """Landed normalized key -> the name that should be on the rail."""
            target: dict[str, str] = {}
            for key, raw in desired[dimension].items():
                if key in unapplied[dimension]:
                    continue
                target[_landed_key(dimension, key)] = (
                    substituted[dimension].get(key) or raw
                )
            return target

        def _target_keys(dimension: str) -> set[str]:
            return set(_target(dimension))

        def _owned_keys(dimension: str) -> set[str]:
            """Keys a removal must never touch: the request AND its landed name.

            A pill is foreign only when no requested value claims it. A value
            recorded ``unapplied`` still claims its key: a pill that appeared
            despite a False applier is not evidence that another string owns
            it.
            """
            return set(desired[dimension]) | {
                _landed_key(dimension, key) for key in desired[dimension]
            }

        # [requested that owns the pill, requested that collapsed onto it,
        # the landed name], per dimension.
        collisions: dict[str, list[list[str]]] = {
            dimension: [] for dimension in dimensions
        }

        def _record_collisions(*, seeded_from_cache: bool = False) -> None:
            """Two requested names that land on ONE pill are not two values.

            LinkedIn has one entity for "Tess" and "Tess AI", so both desired
            keys resolve to the same landed key, the target dict collapses,
            and key-set equality read one pill as satisfying two names. The
            second name constrains nothing and the rail has no way to carry
            it, so it is a recorded miss like any other — a missing entity,
            named, rather than a clean receipt over a narrower search.

            ``seeded_from_cache`` is the pre-apply pass, where the only thing
            that can collapse two names is the sidecar. That entry is what
            dropped the name without the typeahead being asked, so it is
            forgotten: otherwise every later string drops the name again on
            the cache's word alone, and LinkedIn is never asked whether it now
            has an entity for it.
            """
            for dimension in dimensions:
                owners: dict[str, str] = {}
                for key in sorted(desired[dimension]):
                    if key in unapplied[dimension]:
                        continue
                    landed_key = _landed_key(dimension, key)
                    owner = owners.get(landed_key)
                    if owner is None:
                        owners[landed_key] = key
                        continue
                    landed_name = str(
                        substituted[dimension].get(key)
                        or desired[dimension][key]
                    )
                    unapplied[dimension][key] = desired[dimension][key]
                    # One name owns the pill; the other is unapplied, and a
                    # value cannot be both substituted and missing.
                    substituted[dimension].pop(key, None)
                    if seeded_from_cache:
                        self._facet_resolution_cache_forget(
                            dimension, desired[dimension][key]
                        )
                    collisions[dimension].append(
                        [
                            desired[dimension][owner],
                            desired[dimension][key],
                            landed_name,
                        ]
                    )

        def _scope_holds(state: dict[str, dict[str, str]]) -> bool:
            return not company_scope or all(
                observed_scopes.get(key) == company_scope
                for key in state["companies"]
            )

        def _keys_hold(state: dict[str, dict[str, str]]) -> bool:
            return all(
                state[dimension].keys() == _target_keys(dimension)
                for dimension in dimensions
            )

        def _satisfied(state: dict[str, dict[str, str]]) -> bool:
            return _keys_hold(state) and _scope_holds(state)

        async def _snapshot() -> dict[str, dict[str, str]]:
            snapshot: dict[str, dict[str, str]] = {}
            for dimension in dimensions:
                observed = await readers[dimension]()
                if observed is None:
                    raise FacetReconciliationError(f"{dimension}_unreadable")
                snapshot[dimension] = _index(
                    observed, gate=f"{dimension}_observed"
                )
            if company_scope:
                scopes = await self.read_applied_company_chip_scopes()
                if scopes is None:
                    raise FacetReconciliationError("companies_unreadable")
                observed_scopes.clear()
                observed_scopes.update(
                    {facet_name_key(name): scope for name, scope in scopes.items()}
                )
            return snapshot

        def _receipt(
            before: dict[str, dict[str, str]],
            after: dict[str, dict[str, str]],
            *,
            failure_gate: str = "",
        ) -> dict[str, Any]:
            receipt: dict[str, Any] = {
                "before": {},
                "desired": {},
                "after": {},
                "unapplied": {},
                "substituted": {},
                "failure_gate": failure_gate,
            }
            for dimension in dimensions:
                before_values = before[dimension]
                desired_values = desired[dimension]
                after_values = after[dimension]
                receipt["before"][dimension] = list(before_values.values())
                receipt["desired"][dimension] = list(desired_values.values())
                receipt["after"][dimension] = list(after_values.values())
                receipt["unapplied"][dimension] = list(
                    unapplied[dimension].values()
                )
                receipt["substituted"][dimension] = {
                    desired[dimension][key]: landed
                    for key, landed in substituted[dimension].items()
                    if key in desired[dimension]
                }
            if any(collisions[dimension] for dimension in dimensions):
                receipt["collisions"] = {
                    dimension: [list(pair) for pair in collisions[dimension]]
                    for dimension in dimensions
                }
            if company_scope:
                receipt["company_scope"] = {
                    "requested": company_scope,
                    "observed": {
                        raw: observed_scopes.get(key)
                        for key, raw in after["companies"].items()
                    },
                }
            if scope_unverified:
                receipt["scope_unverified"] = {"companies": list(scope_unverified)}
            return receipt

        def _record_success(receipt: dict[str, Any]) -> None:
            # Remember the proven target so reload_search_page can
            # re-verify surface ownership after a document reload. It is what
            # LANDED, not what was asked for: verifying against the request
            # would fail every reload on a string with a recorded miss.
            self.last_facet_reconciliation = receipt
            self._owned_facet_target = {
                dimension: set(_target(dimension).values())
                for dimension in dimensions
            }

        if verify_only:
            before = current = None
            last_error = None
            for attempt in range(FACET_SETTLE_ATTEMPTS):
                try:
                    current = await _snapshot()
                    before = before or current
                    last_error = None
                    if _satisfied(current):
                        _record_success(_receipt(before, current))
                        return
                except FacetReconciliationError as exc:
                    last_error = exc
                if attempt < FACET_SETTLE_ATTEMPTS - 1:
                    await self.page.wait_for_timeout(FACET_SETTLE_INTERVAL_MS)

            if last_error:
                gate = last_error.gate
            elif current is not None and _keys_hold(current):
                gate = "company_scope_mismatch"
            else:
                gate = "postcondition_mismatch"
            receipt = (
                _receipt(before, current, failure_gate=gate)
                if before is not None and current is not None
                else {
                    "before": {dimension: None for dimension in dimensions},
                    "desired": {
                        dimension: list(desired[dimension].values())
                        for dimension in dimensions
                    },
                    "after": {dimension: None for dimension in dimensions},
                    "unapplied": {dimension: [] for dimension in dimensions},
                    "substituted": {dimension: {} for dimension in dimensions},
                    "failure_gate": gate,
                }
            )
            self.last_facet_reconciliation = receipt
            raise FacetReconciliationError(gate, receipt=receipt)

        async def _pin_company_scope(before, current) -> None:
            """Put every company chip on the pinned scope, or take it off the rail.

            A chip whose scope will not read back is a failed value: its
            requests are recorded unapplied and the chip is dismissed, so the
            rail never keeps a company on the default scope. A chip that will
            not leave either is ``remove_failed``.
            """
            observed = await self.read_applied_company_chip_scopes()
            if observed is None:
                receipt = _receipt(before, current, failure_gate="companies_unreadable")
                self.last_facet_reconciliation = receipt
                raise FacetReconciliationError("companies_unreadable", receipt=receipt)
            for name, scope in sorted(observed.items(), key=lambda item: item[0].casefold()):
                if scope == company_scope or await self.set_company_chip_scope(
                    name, company_scope
                ):
                    continue
                key = facet_name_key(name)
                scope_unverified.append(name)
                for requested in sorted(desired["companies"]):
                    if (
                        requested not in unapplied["companies"]
                        and _landed_key("companies", requested) == key
                    ):
                        unapplied["companies"][requested] = desired["companies"][requested]
                        substituted["companies"].pop(requested, None)
                if not await self.remove_facet_value("companies", name):
                    receipt = _receipt(before, current, failure_gate="remove_failed")
                    receipt["failure_dimension"] = "companies"
                    receipt["failure_value"] = name
                    self.last_facet_reconciliation = receipt
                    raise FacetReconciliationError("remove_failed", receipt=receipt)

        # A cache that answers two requested names with one spelling collapses
        # them before a single pill is read, and stops being the cache's word.
        _record_collisions(seeded_from_cache=True)

        try:
            before = await _snapshot()
        except FacetReconciliationError as exc:
            exc.receipt = {
                "before": {dimension: None for dimension in dimensions},
                "desired": {
                    dimension: list(desired[dimension].values())
                    for dimension in dimensions
                },
                "after": {dimension: None for dimension in dimensions},
                "unapplied": {dimension: [] for dimension in dimensions},
                "substituted": {dimension: {} for dimension in dimensions},
                "failure_gate": exc.gate,
            }
            self.last_facet_reconciliation = exc.receipt
            raise
        current = before

        for pass_number in range(2):
            if _satisfied(current):
                _record_success(_receipt(before, current))
                return

            for dimension in dimensions:
                while stale_keys := (
                    current[dimension].keys() - _owned_keys(dimension)
                ):
                    key = sorted(stale_keys)[0]
                    raw = current[dimension][key]
                    settled = None
                    if await self.remove_facet_value(dimension, raw):
                        expected_keys = current[dimension].keys() - {key}
                        for attempt in range(FACET_SETTLE_ATTEMPTS):
                            observed = await readers[dimension]()
                            if observed is not None:
                                candidate = _index(
                                    observed,
                                    gate=f"{dimension}_observed",
                                )
                                if candidate.keys() == expected_keys:
                                    settled = candidate
                                    break
                            if attempt < FACET_SETTLE_ATTEMPTS - 1:
                                await self.page.wait_for_timeout(
                                    FACET_SETTLE_INTERVAL_MS
                                )
                    if settled is None:
                        try:
                            after = await _snapshot()
                        except FacetReconciliationError:
                            after = current
                        receipt = _receipt(
                            before, after, failure_gate="remove_failed"
                        )
                        receipt["failure_dimension"] = dimension
                        receipt["failure_value"] = raw
                        self.last_facet_reconciliation = receipt
                        raise FacetReconciliationError(
                            "remove_failed", receipt=receipt
                        )
                    current[dimension] = settled

            for dimension in dimensions:
                missing_keys = [
                    key
                    for key in sorted(desired[dimension])
                    if key not in unapplied[dimension]
                    and _landed_key(dimension, key) not in current[dimension]
                ]
                for key in missing_keys:
                    raw = desired[dimension][key]
                    # The applier is authoritative about which name landed:
                    # clear its record, call it, read it back.
                    setattr(self, substitution_attrs[dimension], {})
                    # ONE value per call: the applier fails the whole batch on
                    # a single match miss, which is the common case.
                    try:
                        landed = bool(await appliers[dimension]([raw]))
                    except Exception as exc:
                        if _is_target_crash_error(exc):
                            # A dead page is not a miss. Keep its identity so
                            # the loop's disconnect recovery still owns it.
                            raise
                        log.warning(
                            "reconcile_facets[%s]: %r raised during apply — "
                            "recording it unapplied (%s: %s)",
                            dimension,
                            raw,
                            type(exc).__name__,
                            exc,
                        )
                        landed = False
                    if not landed:
                        unapplied[dimension][key] = raw
                        # A seeded resolution is the cache's word; a miss is
                        # the rail's. The rail wins, and the sidecar forgets —
                        # otherwise the next string types the same dead
                        # spelling first and spends an attempt proving it dead
                        # again.
                        substituted[dimension].pop(key, None)
                        self._facet_resolution_cache_forget(dimension, raw)
                        continue
                    applied_as = (
                        getattr(self, substitution_attrs[dimension], None) or {}
                    ).get(raw)
                    if applied_as and facet_name_key(applied_as) != key:
                        substituted[dimension][key] = str(applied_as)
                    else:
                        substituted[dimension].pop(key, None)

            # The appliers are what turn two requested names into one landed
            # name, so the collapse is only visible once they have all run.
            _record_collisions()

            if company_scope:
                await _pin_company_scope(before, current)

            try:
                current = await _snapshot()
            except FacetReconciliationError as exc:
                exc.receipt = _receipt(
                    before, current, failure_gate=exc.gate
                )
                self.last_facet_reconciliation = exc.receipt
                raise

            if _satisfied(current):
                _record_success(_receipt(before, current))
                return

            if pass_number == 1:
                receipt = _receipt(
                    before, current, failure_gate="postcondition_mismatch"
                )
                self.last_facet_reconciliation = receipt
                raise FacetReconciliationError(
                    "postcondition_mismatch", receipt=receipt
                )

    async def apply_company_filter(
        self, values: list[str], *, temporal_scope: str = "any"
    ) -> bool:
        """Apply Company filters with the same fail-closed preflight/retry/verify discipline as Location.

        The pane is aside.left-rail. Reveal the merged facet with "Companies or boolean";
        LinkedIn prefixes "Add" only when no pills exist. The editor placeholder contains
        "company"; options are li[role=option] in ul.artdeco-typeahead__results-list[role=listbox].
        Applied pills use li[data-test-facet-pills-item] and may carry data-test-pill-label.
        Confirm the visible pill, not the hover-gated "Remove {value}" control.
        See docs/linkedin-recruiter-dom-map.md for the Companies selectors.

        A missing editor, a listbox that satisfies nothing, or an unconfirmed chip
        returns False for keyword fallback. TAKE THE NEAR MISS: LinkedIn answers a
        strategist's "Mercado Livre" with its own entity name "Mercado Livre
        Brasil", so ``resolve_facet_option`` decides which offered option
        satisfies the request, and every non-exact take is recorded in
        ``last_company_substitutions`` (requested -> landed) so the receipt, the
        console and every later verification name what is actually constraining
        the search. Never press Enter — always click the option. This method
        drives company filters for hybrid lanes through
        advanced_search.STABLE_NOW_CONTROLS.

        temporal_scope is an upstream routing signal, not a dropdown selector.
        The merged Companies facet always uses "Current or Past", never current-only.
        """
        companies = [str(v).strip() for v in (values or []) if str(v).strip()]
        if not companies:
            return False

        # P3a Stage B parity with locations: per-call match-miss options,
        # keyed by requested value for live diagnostics.
        self.last_company_option_misses: dict[str, list[str]] = {}
        # requested -> the name LinkedIn actually applied, for every near miss.
        self.last_company_substitutions: dict[str, str] = {}
        self.last_company_already_applied_count = 0

        self._last_search_snapshot = {}
        await self.go_back_to_results()
        await self.require_recruiter_tab()
        previous_count_text = await self._peek_results_count_text()
        previous_top_card_signature = await self._peek_top_card_signature()

        rail = self.page.locator("aside.left-rail")
        pill_wrapper = rail.locator("div.search-facet-wrapper.facet-companies")
        editor_input = rail.locator(
            'input.artdeco-typeahead__input[placeholder*="company" i]'
        ).first

        applied: list[str] = []
        try:
            pre_applied_chips = await self.read_applied_company_chips()
        except Exception:
            pre_applied_chips = None
        if pre_applied_chips is None:
            log.warning("apply_company_filter: applied Companies facet is unreadable")
            return False
        pre_applied = {
            normalize_facet_value_for_compare(chip) for chip in pre_applied_chips
        }
        option_list = rail.locator(
            'ul.artdeco-typeahead__results-list[role="listbox"] li[role="option"]'
        )
        typed_any = False
        for value in companies:
            # Spellings to type, best first: what a previous string proved
            # LinkedIn calls this company, then what the strategist wrote.
            cached = self._facet_resolution_cache_get("companies", value)
            spellings = [value]
            if cached and normalize_facet_value_for_compare(
                cached
            ) != normalize_facet_value_for_compare(value):
                spellings = [cached, value]
            already = next(
                (
                    spelling
                    for spelling in spellings
                    if normalize_facet_value_for_compare(spelling) in pre_applied
                ),
                None,
            )
            if already is not None:
                log.info(
                    "apply_company_filter[%r]: chip %r already applied — "
                    "skipping typeahead input.",
                    value,
                    already,
                )
                applied.append(value)
                self.last_company_already_applied_count += 1
                if normalize_facet_value_for_compare(
                    already
                ) != normalize_facet_value_for_compare(value):
                    self.last_company_substitutions[value] = already
                continue
            typed_any = True

            async def _do(value=value, spellings=tuple(spellings)):
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
                offered_for_receipt: list[str] = []
                for index, spelling in enumerate(spellings):
                    outcome, payload = await self._take_facet_typeahead_option(
                        requested=value,
                        candidate=spelling,
                        option_list=option_list,
                        editor_input=editor_input,
                        pill_wrapper=pill_wrapper,
                        rail=rail,
                        # A cached spelling is an optimization, not a budget:
                        # one attempt, then fall back to what was requested.
                        retries=1 if index < len(spellings) - 1 else 4,
                        log_prefix=f"apply_company_filter[{value!r}]",
                    )
                    if outcome == "unusable":
                        return False
                    if outcome == "landed":
                        landed = str(payload)
                        if normalize_facet_value_for_compare(
                            landed
                        ) != normalize_facet_value_for_compare(value):
                            self.last_company_substitutions[value] = landed
                            self._facet_resolution_cache_put(
                                "companies", value, landed
                            )
                        else:
                            self._facet_resolution_cache_forget("companies", value)
                        return True
                    offered_for_receipt = list(payload or []) or offered_for_receipt
                    if outcome == "chip_miss":
                        break
                self.last_company_option_misses[value] = offered_for_receipt
                return False

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
        # Companies always uses "Current or Past"; temporal_scope is only an upstream routing signal.
        # The default already supplies that scope. Chip confirmation is primary; changed results are secondary.
        await self._wait_for_search_results_ready(
            previous_count_text=previous_count_text,
            previous_top_card_signature=previous_top_card_signature,
        )
        return True

    async def read_applied_title_chips(self) -> list[str] | None:
        return await self._read_applied_facet_chips("titles")

    async def apply_title_filter(
        self, values: list[str], *, temporal_scope: str = "any"
    ) -> bool:
        """Apply Job Title filters using the shared typeahead architecture.

        Selectors are in linkedin-recruiter-selectors.yaml and docs/linkedin-recruiter-dom-map.md.
        Reveal with "Job titles or boolean"; LinkedIn prefixes "Add" when no pills exist.
        The editor placeholder contains "job title". Options, pills and removal follow
        the Location/Companies contract. This method drives hybrid title filters through
        advanced_search.STABLE_NOW_CONTROLS.

        Always use "Current or Past", never current-only. temporal_scope is accepted for
        upstream routing and does not choose the dropdown. The DOM map records that
        selector if the scope policy changes.

        A missing editor, a listbox that satisfies nothing, or an unconfirmed pill
        returns False for keyword fallback. TAKE THE NEAR MISS: LinkedIn offers its
        own canonical titles, so a strategist's "Tess AI" arrives as "Tess".
        ``resolve_facet_option`` decides which option satisfies the request and
        every non-exact take is recorded in ``last_title_substitutions``
        (requested -> landed), so the receipt and the console name what is
        actually constraining the search.
        """
        titles = [str(v).strip() for v in (values or []) if str(v).strip()]
        if not titles:
            return False

        self.last_title_option_misses: dict[str, list[str]] = {}
        # requested -> the name LinkedIn actually applied, for every near miss.
        self.last_title_substitutions: dict[str, str] = {}
        self.last_title_already_applied_count = 0

        self._last_search_snapshot = {}
        await self.go_back_to_results()
        await self.require_recruiter_tab()
        previous_count_text = await self._peek_results_count_text()
        previous_top_card_signature = await self._peek_top_card_signature()

        rail = self.page.locator("aside.left-rail")
        pill_wrapper = rail.locator("div.search-facet-wrapper.facet-titles")
        editor_input = rail.locator(
            'input.artdeco-typeahead__input[placeholder*="job title" i]'
        ).first

        applied: list[str] = []
        try:
            pre_applied_chips = await self.read_applied_title_chips()
        except Exception:
            pre_applied_chips = None
        if pre_applied_chips is None:
            log.warning("apply_title_filter: applied Titles facet is unreadable")
            return False
        pre_applied = {
            normalize_facet_value_for_compare(chip) for chip in pre_applied_chips
        }
        option_list = rail.locator(
            'ul.artdeco-typeahead__results-list[role="listbox"] li[role="option"]'
        )
        typed_any = False
        for value in titles:
            cached = self._facet_resolution_cache_get("titles", value)
            spellings = [value]
            if cached and normalize_facet_value_for_compare(
                cached
            ) != normalize_facet_value_for_compare(value):
                spellings = [cached, value]
            already = next(
                (
                    spelling
                    for spelling in spellings
                    if normalize_facet_value_for_compare(spelling) in pre_applied
                ),
                None,
            )
            if already is not None:
                log.info(
                    "apply_title_filter[%r]: chip %r already applied — "
                    "skipping typeahead input.",
                    value,
                    already,
                )
                applied.append(value)
                self.last_title_already_applied_count += 1
                if normalize_facet_value_for_compare(
                    already
                ) != normalize_facet_value_for_compare(value):
                    self.last_title_substitutions[value] = already
                continue
            typed_any = True

            async def _do(value=value, spellings=tuple(spellings)):
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
                offered_for_receipt: list[str] = []
                for index, spelling in enumerate(spellings):
                    outcome, payload = await self._take_facet_typeahead_option(
                        requested=value,
                        candidate=spelling,
                        option_list=option_list,
                        editor_input=editor_input,
                        pill_wrapper=pill_wrapper,
                        rail=rail,
                        retries=1 if index < len(spellings) - 1 else 4,
                        log_prefix=f"apply_title_filter[{value!r}]",
                    )
                    if outcome == "unusable":
                        return False
                    if outcome == "landed":
                        landed = str(payload)
                        if normalize_facet_value_for_compare(
                            landed
                        ) != normalize_facet_value_for_compare(value):
                            self.last_title_substitutions[value] = landed
                            self._facet_resolution_cache_put("titles", value, landed)
                        else:
                            self._facet_resolution_cache_forget("titles", value)
                        return True
                    offered_for_receipt = list(payload or []) or offered_for_receipt
                    if outcome == "chip_miss":
                        break
                self.last_title_option_misses[value] = offered_for_receipt
                return False

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
        # Job titles always uses "Current or Past". temporal_scope is an upstream routing signal,
        # not a dropdown selector; the facet default already supplies the required scope.
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
        # "916", "2,345", "1K+", "1.2K+", "3M+"
        clean = raw.replace(",", "").strip().upper().rstrip("+")
        multiplier = {"K": 1000, "M": 1_000_000}.get(clean[-1:], 1)
        if multiplier != 1:
            clean = clean[:-1]
        try:
            return int(float(clean) * multiplier) if multiplier != 1 else int(clean)
        except ValueError:
            return -1

    async def get_results_count_text(self) -> str:
        """The count portion of the result count ("916 results" -> "916", "1.2K+ results" -> "1.2K+").

        Read from ``RESULT_COUNT_SELECTORS`` in order. Waits briefly for
        LinkedIn to render the count after a search; it is briefly absent
        while results reload.
        """
        any_count = self.page.locator(", ".join(RESULT_COUNT_SELECTORS)).first
        for _ in range(5):
            try:
                await any_count.wait_for(state="attached", timeout=3000)
            except Exception:
                pass
            text = await self._read_results_count_once(timeout_ms=3000)
            if text:
                return text
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

            # Addendum D: burst size eases as the list stops growing, so the
            # rhythm is not a metronome of identical chunks. `stable` is the
            # progress signal — 0 while new cards keep arriving, rising near the
            # bottom.
            step = self._results_scroll_chunk(stable)
            await self._human_scroll(step, channel="results_list")
            total_scrolled += step

            # Occasional scan pause while reviewing newly exposed cards.
            if scroll_num % random.randint(4, 6) == 0:
                await asyncio.sleep(human_delay_correlated(0.5, channel="results_list_scan"))

            # Let DOM update after scroll. Floor keeps lazy-render safety; the
            # correlated sample above the floor breaks the fixed 300ms cadence.
            settle_ms = int(max(220, human_delay_correlated(0.32, channel="results_list") * 1000))
            await self.page.wait_for_timeout(settle_ms)

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

        # Scroll back to top in a few uneven steps — one instantaneous return of
        # several thousand px is itself a signature.
        await self._scroll_results_to_top(total_scrolled)

        return await self.page.locator(cards_selector).count()

    @staticmethod
    def _results_scroll_chunk(stable_rounds: int) -> int:
        """Pixels for the next results-list burst, easing as the list settles.

        Pure so the pacing is unit-testable. Big bursts while cards still load,
        short nudges once the count stops climbing — what a human does slowing
        toward the bottom of a list.
        """
        if stable_rounds <= 0:
            return random.randint(400, 700)
        if stable_rounds == 1:
            return random.randint(240, 460)
        return random.randint(150, 300)

    async def _scroll_results_to_top(self, total_scrolled: int) -> None:
        """Return to the top of the results list in 2-3 uneven steps."""
        if total_scrolled <= 0:
            await asyncio.sleep(
                human_delay_correlated(0.3, channel="results_list_return")
            )
            return
        steps = random.randint(2, 3)
        remaining = total_scrolled
        for index in range(steps):
            if index == steps - 1:
                part = remaining
            else:
                part = min(
                    remaining,
                    max(1, int(total_scrolled / steps * random.uniform(0.8, 1.2))),
                )
            if part > 0:
                await self._human_scroll(-part, channel="results_list_return")
                remaining -= part
            if index < steps - 1 and remaining > 0:
                await asyncio.sleep(
                    human_delay_correlated(0.2, channel="results_list_return")
                )

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
            r"/(?:talent/profile|recruiterSearch/profile|"
            r"talent/hire/[^/?#]+(?:/[^/?#]+)*/profile)/([^/?#]+)",
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
        """Find a rendered result slot by its already-extracted identity fragment.
        This avoids reconstructing a profile URL merely to parse that identity again.
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

    # Fallbacks for the next-page control, all inside pagination. A bare
    # button[aria-label="Next"] also matches the "Spotlights" carousel above
    # the results, which is not pagination.
    _PAGINATION_NEXT_FALLBACKS = (
        'a.mini-pagination__quick-link[aria-label*="next" i]',
        'a.mini-pagination__quick-link:has-text("next")',
        ':is(.pagination, .mini-pagination, .artdeco-pagination) '
        'button[aria-label="Next"]:not(.skyline-pagination-button)',
    )
    _PAGINATION_PREVIOUS = (
        "a.pagination__quick-link--previous",
        'a.mini-pagination__quick-link[aria-label*="previous" i]',
        ':is(.pagination, .mini-pagination, .artdeco-pagination) '
        'button[aria-label="Previous"]:not(.skyline-pagination-button)',
    )

    async def _first_visible(self, selectors) -> Any:
        for selector in selectors:
            candidate = self.page.locator(selector).first
            try:
                if await candidate.is_visible():
                    return candidate
            except Exception:
                continue
        return None

    async def results_page_offset(self) -> int | None:
        """The 0-based offset of the results page on screen, from the URL.

        Pagination writes ``start=25``, ``start=50``... into the URL; a fresh
        search carries none, which is page one. A ``start`` that is not a
        non-negative integer is unreadable (None), never page one.
        """
        try:
            url = str(self.page.url)
        except Exception:
            return None
        values = parse_qs(urlparse(url).query).get("start")
        if not values:
            return 0
        try:
            offset = int(values[-1])
        except ValueError:
            return None
        return offset if offset >= 0 else None

    async def go_to_next_page(self) -> bool:
        """Click the bottom pagination link to advance to the next results page.

        Primary: a.pagination__quick-link--next (bottom of results list)
        Fallbacks: ``_PAGINATION_NEXT_FALLBACKS``, all scoped to pagination.
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
                next_control = await self._first_visible(self._PAGINATION_NEXT_FALLBACKS)
                if next_control is None or not await next_control.is_enabled(timeout=3000):
                    return False
                await self._ghost_click_locator(next_control)

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
            prev_control = await self._first_visible(self._PAGINATION_PREVIOUS)
            if prev_control is not None and await prev_control.is_enabled():
                await self._ghost_click_locator(prev_control)
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

        Governed at the source — checks the governor before attempting
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
                    await self.page.locator(PROFILE_PANEL_SELECTOR).wait_for(
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
            await self.page.locator(PROFILE_PANEL_SELECTOR).wait_for(
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

        Governed at the source — see open_profile().
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

            await self.page.locator(PROFILE_PANEL_SELECTOR).wait_for(
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

    # Base dwell budget; the section and profile caps below bound the log-normal tail.
    _BASE_SHORT_DWELL = 3.0
    _BASE_CHUNK_DWELL_LOW = 1.0
    _BASE_CHUNK_DWELL_HIGH = 2.5
    # A single section dwell never exceeds this (humans don't stare at one
    # profile chunk longer), and the whole-profile lingering is budgeted so a
    # long profile can't compound past this total.
    _MAX_CHUNK_DWELL = 8.0
    _MAX_PROFILE_READ = 35.0
    _SECTION_DIRECTED_NAMES = frozenset({"about", "experience", "education"})

    # Vary the profile-read path without increasing the interest dwell budget.
    _TRAVERSAL_STYLES = ("straight", "backtracker", "experience_first", "overshoot")
    _TRAVERSAL_WEIGHTS = (45, 25, 20, 10)
    _ABOUT_SKIP_PROBABILITY = 0.30

    # Section-directed read dwell budget. Measured from 54 real section-directed
    # reads in the live AIEL campaign (profile_read_timing events, run log
    # 9000000012-aiel-20260729): at the fixed medium interest of 0.5 the prior
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
        """Sleep for a profile-read dwell clamped per section and per profile.
        Every channel="profile_read" dwell routes here to bound its tail.
        _profile_read_elapsed resets at the start of simulate_profile_read.
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
                base_pattern = "section:" + (
                    ">".join(s.name for s in sections) if sections else "none"
                )
                chunk_count = len(sections)
                try:
                    await self._read_section_directed(sections, interest)
                except _ProfileReadBudgetExhausted:
                    pass
                # Addendum D: label the read with its traversal style so the
                # path variety is measurable in profile_read_timing telemetry.
                style = getattr(self, "_section_traversal_style", None)
                pattern = f"{base_pattern}|{style}" if style else base_pattern
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

    def _plan_section_traversal(self, target_sections: list, *, style: str | None = None):
        """Plan a section-read path. Pure: no I/O, so the policy is unit-testable.

        Returns ``(style, stops)`` where each stop is
        ``(section, dwell_fraction, overshoot)`` and the fractions sum to 1.0 —
        so whatever the path, the total dwell equals the interest budget and the
        redesign adds no dwell time (only, for backtracks, a little scroll
        travel). ``style`` is forced by callers in tests; ``None`` samples it.
        """
        ordered = sorted(target_sections, key=lambda s: s.offset)
        by_name = {s.name: s for s in ordered}
        names = [s.name for s in ordered]

        # ~30% of reads skip the About dwell — a recruiter who already saw the
        # headline. Only when About is present and something else remains.
        if (
            "about" in by_name
            and len(names) >= 2
            and random.random() < self._ABOUT_SKIP_PROBABILITY
        ):
            names = [n for n in names if n != "about"]

        if style is None:
            style = random.choices(
                self._TRAVERSAL_STYLES, weights=self._TRAVERSAL_WEIGHTS
            )[0]

        have_experience = "experience" in names
        if style == "experience_first" and have_experience and len(names) >= 2:
            order = ["experience"] + [n for n in names if n != "experience"]
        elif style == "backtracker" and have_experience and len(names) >= 2:
            order = list(names) + ["experience"]
        elif style == "overshoot":
            order = list(names)
        else:
            # straight, or a style that cannot apply to this profile's sections
            style = "straight"
            order = list(names)

        overshoot_index = (
            random.randrange(len(order)) if style == "overshoot" and order else None
        )
        share = 1.0 / len(order)
        stops = [
            (by_name[name], share, index == overshoot_index)
            for index, name in enumerate(order)
        ]
        return style, stops

    async def _scroll_to_section(self, delta: int, *, overshoot: bool) -> None:
        """Scroll toward a section by ``delta`` (sign = direction). ``overshoot``
        deliberately passes the heading and corrects back — a human landing move
        — but only when there is room for it to read as one."""
        if overshoot and abs(delta) > 120:
            past = random.randint(40, 120)
            signed_past = past if delta > 0 else -past
            await self._human_scroll(int(delta + signed_past), channel="profile_read")
            await self._human_scroll(int(-signed_past), channel="profile_read")
        else:
            await self._human_scroll(int(delta), channel="profile_read")

    async def _read_section_directed(self, sections: list, interest: float) -> None:
        """Read profile sections along a varied per-read path; dwell budget scales
        with interest and is preserved across whichever path is chosen."""
        self._section_traversal_style = None
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

        style, stops = self._plan_section_traversal(target_sections)
        self._section_traversal_style = style
        current_offset = 0.0

        for section, dwell_fraction, overshoot in stops:
            delta = section.offset - current_offset
            if abs(delta) > 4:
                await self._scroll_to_section(int(delta), overshoot=overshoot)
                current_offset = section.offset

            section_dwell = dwell_fraction * budget
            if section.name == "experience":
                num_entries = random.randint(2, 3)
                per_entry = section_dwell / num_entries
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
                        section_dwell,
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
            container = self.page.locator(PROFILE_PANEL_SELECTOR).first
            await container.wait_for(state="visible", timeout=10000)
            text = await container.inner_text(timeout=15000)
            return extract_profile_recent_activity_lines(text)

        return await _retry(_do)

    async def get_profile_status_summary(self) -> dict:
        """Read recruiter-facing status metadata from the slide-in profile."""

        async def _do():
            container = self.page.locator(PROFILE_PANEL_SELECTOR).first
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

    # Resolve note-composer descendants from the form so controls cannot escape the profile panel.
    _NOTE_MORE_ACTIONS_TRIGGER_PRIMARY = _PROFILE_PANEL_SCOPE + " button[data-test-more-actions-trigger]"
    _NOTE_ADD_MENU_ITEM_PRIMARY = _PROFILE_PANEL_SCOPE + ' [role="button"][data-test-action="add-note"]'
    _NOTE_FORM_PRIMARY = _PROFILE_PANEL_SCOPE + " form.create-edit-note__form"
    _NOTE_FORM_FALLBACK = (
        _PROFILE_PANEL_SCOPE
        + " form:is([data-live-test-create-edit-note-form], "
        "[data-test-create-edit-note-form])"
    )
    _NOTE_TEXTAREA_PRIMARY = "textarea[data-live-test-mentions-edit-textarea]"
    _NOTE_TEXTAREA_FALLBACK = (
        "textarea:is([data-test-mentions-edit-textarea], .mentions-edit__text-box)"
    )
    _NOTE_VISIBILITY_CHECKED = 'input[name="visibility-option"]:checked'
    _NOTE_SUBMIT = "button[data-test-create-edit-note-submit-btn]"
    _NOTE_CANCEL = "button[data-test-create-edit-note-cancel-btn]"
    _NOTE_TITLE = _PROFILE_PANEL_SCOPE + " [data-test-note-list-title]"
    _NOTE_EXPAND_PRIMARY = (
        _PROFILE_PANEL_SCOPE
        + " .notes-expandable-list__expand-btn.note-list__expand-btn"
    )
    _NOTE_EXPAND_FALLBACK = (
        _PROFILE_PANEL_SCOPE
        + " :has(> [data-test-note-list-title]) "
        "[data-test-expandable-list-expand-btn]"
    )

    async def write_candidate_note(
        self,
        note_text: str,
        *,
        expected_name: str,
        before_commit: Callable[[], None] | None = None,
    ) -> bool:
        """Write one project-visible Cloris note, with one non-retryable submit."""
        self._last_note_failure_reason = None
        probe = note_probe_text(note_text)
        expected_identity = self.current_profile_identity_fragment()

        def _guard_identity() -> None:
            if (
                not expected_identity
                or self.current_profile_identity_fragment() != expected_identity
            ):
                self._last_note_failure_reason = "note_panel_identity_mismatch"
                raise _SaveOperationAbort(
                    RuntimeError("profile identity mismatch during note write")
                )

        async def _checked(awaitable):
            _guard_identity()
            result = await awaitable
            _guard_identity()
            return result

        async def _read_note_present() -> bool | None:
            """Return note presence only when the full note list is readable."""
            try:
                title_matches = self.page.locator(self._NOTE_TITLE)
                if await _checked(title_matches.count()) != 1:
                    return None
                title = title_matches.first

                title_text = await _checked(title.inner_text(timeout=2000))
                count_match = re.fullmatch(r"\s*Notes\s*\((\d+)\)\s*", title_text)
                if count_match is None:
                    return None
                if int(count_match.group(1)) == 0:
                    return False

                expand_matches = self.page.locator(self._NOTE_EXPAND_PRIMARY)
                expand_count = await _checked(expand_matches.count())
                if expand_count == 0:
                    expand_matches = self.page.locator(self._NOTE_EXPAND_FALLBACK)
                    expand_count = await _checked(expand_matches.count())
                if expand_count > 1:
                    return None
                if expand_count == 1:
                    # The expand control stays MOUNTED BUT HIDDEN when there is
                    # nothing to expand (measured live with a single note).
                    # Ghost-clicking a hidden element raises, which turned a
                    # perfectly readable one-note list into `note_probe_failed`
                    # — the primitive then refused to recognise its OWN note.
                    # Present-but-hidden means "nothing collapsed", so read on.
                    if await _checked(expand_matches.first.is_visible()):
                        await _checked(
                            self._ghost_click_locator(
                                expand_matches.first,
                                before_click=_guard_identity,
                            )
                        )
                        await _checked(self.page.wait_for_timeout(250))
                        # Re-check VISIBILITY, not presence: the control remains
                        # in the DOM after expanding and only goes hidden.
                        if await _checked(expand_matches.first.is_visible()):
                            return None

                panel_matches = self.page.locator(PROFILE_PANEL_SELECTOR)
                if await _checked(panel_matches.count()) != 1:
                    return None
                panel_text = await _checked(
                    panel_matches.first.inner_text(timeout=2000)
                )
                # The probe is the NOTE'S OWN opening line (note_probe_text),
                # not an agent marker: the note is the evaluation alone. A
                # stored judgment renders deterministically, so a retried save
                # rebuilds the same text and recognises it. Two accepted
                # tradeoffs: a RE-JUDGED candidate renders new text and gets a
                # genuinely new note, and a rail that renders a note SHORTER
                # than the probe window would not detect the duplicate
                # (Recruiter's collapsed render length is not recorded — notes
                # are `defer` in docs/linkedin-recruiter-dom-map.md — so
                # measure before widening it). Both sides are
                # whitespace-normalized so rendered line wrapping cannot break
                # a match. This runs before composition, so no unsaved draft
                # exists.
                return probe in " ".join(panel_text.split())
            except _SaveOperationAbort:
                raise
            except Exception as exc:
                if _is_target_crash_error(exc):
                    raise
                return None

        async def _probe_only() -> bool:
            present = await _read_note_present()
            if present is None:
                self._last_note_failure_reason = "note_probe_failed"
                raise RuntimeError("candidate note list was unreadable")
            return present

        # ---- Phase 1: idempotency probe. Retryable; no note mutation. ----
        # An empty probe would match every panel and silently skip every write,
        # so an empty note is refused here rather than typed into Recruiter.
        if not probe:
            self._last_note_failure_reason = "empty_note"
            return False
        try:
            note_present = await _retry(_probe_only)
        except _SaveOperationAbort as abort:
            raise abort.cause
        except Exception as exc:
            if _is_target_crash_error(exc):
                raise
            self._last_note_failure_reason = "note_probe_failed"
            return False
        if note_present:
            self._last_note_failure_reason = "note_already_present"
            return True

        async def _resolve_more_actions_trigger():
            matches = self.page.locator(self._NOTE_MORE_ACTIONS_TRIGGER_PRIMARY)
            if await _checked(matches.count()) != 1:
                self._last_note_failure_reason = "note_button_not_found"
                raise RuntimeError("candidate more-actions button was not found")
            button = matches.first
            await _checked(button.wait_for(state="visible", timeout=5000))
            label = " ".join(
                str(await _checked(button.inner_text(timeout=2000)) or "").split()
            )
            prefix = "More actions for "
            if not label.startswith(prefix) or not _panel_confirms_expected_name(
                label[len(prefix) :], expected_name
            ):
                self._last_note_failure_reason = "note_button_name_mismatch"
                raise RuntimeError("candidate more-actions button name mismatch")
            return button
        # ---- Phase 2: identity-bearing more-actions resolution. Retryable. ----
        try:
            more_actions = await _retry(_resolve_more_actions_trigger)
        except _SaveOperationAbort as abort:
            raise abort.cause
        except Exception as exc:
            if _is_target_crash_error(exc):
                raise
            self._last_note_failure_reason = self._last_note_failure_reason or "note_button_not_found"
            return False

        async def _resolve_form():
            for selector in (self._NOTE_FORM_PRIMARY, self._NOTE_FORM_FALLBACK):
                form = self.page.locator(selector).first
                try:
                    await _checked(form.wait_for(state="visible", timeout=5000))
                    return form
                except _SaveOperationAbort:
                    raise
                except Exception as exc:
                    if _is_target_crash_error(exc):
                        raise
            return None

        async def _resolve_textarea(form):
            for selector in (
                self._NOTE_TEXTAREA_PRIMARY,
                self._NOTE_TEXTAREA_FALLBACK,
            ):
                matches = form.locator(selector)
                if await _checked(matches.count()) == 1:
                    return matches.first
            return None

        async def _cancel(form, reason: str) -> bool:
            self._last_note_failure_reason = reason
            try:
                matches = form.locator(self._NOTE_CANCEL)
                if await _checked(matches.count()) == 1:
                    await _checked(
                        self._ghost_click_locator(
                            matches.first,
                            before_click=_guard_identity,
                        )
                    )
            except _SaveOperationAbort:
                raise
            except Exception as exc:
                if _is_target_crash_error(exc):
                    raise
            return False

        submit_attempted = False

        def _guard_submit() -> None:
            nonlocal submit_attempted
            _guard_identity()
            if before_commit is not None:
                try:
                    before_commit()
                except BaseException as exc:
                    raise _SaveOperationAbort(exc) from exc
            submit_attempted = True

        async def _cancel_best_effort(form) -> None:
            try:
                matches = form.locator(self._NOTE_CANCEL)
                if await _checked(matches.count()) == 1:
                    await _checked(
                        self._ghost_click_locator(
                            matches.first,
                            before_click=_guard_identity,
                        )
                    )
            except _SaveOperationAbort:
                pass
            except Exception:
                pass
        # ---- Phase 3: compose. Submit is resolved once and pressed once. ----
        try:
            await _checked(self._ghost_click_locator(more_actions, before_click=_guard_identity))
            # NEVER use page.wait_for_function on a LinkedIn page. Its CSP
            # forbids 'unsafe-eval', and wait_for_function evaluates a string as
            # JavaScript in the page: measured live, it raises `EvalError:
            # Evaluating a string as JavaScript violates the following Content
            # Security Policy directive` on EVERY call. That surfaced as a
            # permanent `note_menu_item_not_found` against a healthy menu.
            # Locator waits go through CDP and are not subject to that policy.
            #
            # Waiting on the item's visibility is also the better signal: the
            # trigger's aria-expanded flips synchronously on click, so it proves
            # nothing about the contents, which mount asynchronously (~500ms
            # measured). The timeout is generous because a short one reports a
            # healthy menu as missing.
            menu_matches = self.page.locator(self._NOTE_ADD_MENU_ITEM_PRIMARY)
            await _checked(menu_matches.first.wait_for(state="visible", timeout=10000))
            if await _checked(menu_matches.count()) != 1:
                raise RuntimeError("candidate add-note menu item was not found")
            await _checked(self._ghost_click_locator(menu_matches.first, before_click=_guard_identity))
        except Exception as exc:
            try:
                await self.page.keyboard.press("Escape")
            except Exception:
                pass
            if isinstance(exc, _SaveOperationAbort):
                raise exc.cause
            if _is_target_crash_error(exc):
                raise
            self._last_note_failure_reason = "note_menu_item_not_found"
            return False

        form = None
        try:
            await _checked(
                asyncio.sleep(
                    human_delay_correlated(0.8, channel="note_compose")
                )
            )
            form = await _resolve_form()
            if form is None:
                self._last_note_failure_reason = "note_composer_not_found"
                return False

            textarea = await _resolve_textarea(form)
            if textarea is None:
                return await _cancel(form, "note_composer_not_found")
            await _checked(
                self._ghost_click_locator(
                    textarea,
                    before_click=_guard_identity,
                )
            )
            await _checked(
                self._input_backend.type_text(
                    self.page,
                    textarea,
                    note_text,
                    plan=build_note_typing_plan(note_text),
                )
            )
            typed_text = await _checked(textarea.input_value(timeout=2000))
            if " ".join(str(typed_text).split()) != " ".join(note_text.split()):
                return await _cancel(form, "note_text_mismatch")

            checked_radios = form.locator(self._NOTE_VISIBILITY_CHECKED)
            if await _checked(checked_radios.count()) != 1:
                return await _cancel(form, "note_visibility_not_project")
            visibility_id = await _checked(
                checked_radios.first.get_attribute("id")
            )
            if not str(visibility_id or "").endswith("-PROJECT"):
                return await _cancel(form, "note_visibility_not_project")

            submit_matches = form.locator(self._NOTE_SUBMIT)
            submit_count = await _checked(submit_matches.count())
            if submit_count != 1:
                reason = (
                    "note_submit_ambiguous"
                    if submit_count > 1
                    else "note_submit_not_found"
                )
                return await _cancel(form, reason)
            submit = submit_matches.first
            if not await _checked(submit.is_enabled(timeout=2000)):
                return await _cancel(form, "note_submit_not_found")

            await _checked(
                asyncio.sleep(
                    human_delay_correlated(0.8, channel="note_submit")
                )
            )
            await _checked(
                self._ghost_click_locator(
                    submit,
                    before_click=_guard_submit,
                )
            )
        except _SaveOperationAbort as abort:
            raise abort.cause
        except Exception as exc:
            if _is_target_crash_error(exc):
                raise
            if form is not None:
                await _cancel_best_effort(form)
            self._last_note_failure_reason = (
                "note_not_confirmed" if submit_attempted else "note_not_submitted"
            )
            return False

        # ---- Phase 4: confirmation only. Never presses submit again. ----
        last_probe: bool | None = None
        try:
            for _ in range(3):
                await _checked(self.page.wait_for_timeout(2000))
                # The note's own opening line appearing in the notes list is the
                # ONLY sound proof of persistence. Do NOT also require the
                # composer to have closed: measured live, LinkedIn leaves a
                # fresh EMPTY composer mounted in the rail after a successful
                # submit, so a form-closed condition never becomes true and
                # reports a note that demonstrably landed as
                # `note_not_confirmed`.
                last_probe = await _read_note_present()
                if last_probe is True:
                    self._last_note_failure_reason = None
                    return True
        except _SaveOperationAbort as abort:
            raise abort.cause
        except Exception as exc:
            if _is_target_crash_error(exc):
                raise
            last_probe = None

        await _cancel_best_effort(form)
        self._last_note_failure_reason = (
            "note_not_persisted" if last_probe is False else "note_not_confirmed"
        )
        return False

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
        """Dismiss an open profile slide-in with multiple strategies, including JS dismissal.

        The artdeco-modal-outlet can block Playwright clicks. Never call go_back(), which
        would leave the search page. Record the closing strategy so a later mount failure
        can be attributed; forcibly removing Ember-managed outlet nodes can wedge its modal manager.
        """
        async def _do():
            slidein = self.page.locator(PROFILE_PANEL_SELECTOR)
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
                    self._emit_panel_close_strategy("button", True)
                    return

            # Strategy 2: Escape key
            await self._press_key("Escape")
            await self.page.wait_for_timeout(1000)
            if not await slidein.is_visible():
                self._emit_panel_close_strategy("escape", True)
                return

            # Strategy 3: JS — click the back/close link in the slide-in header,
            # else the nuclear fallback: hide the modal outlet entirely.
            s3_mode = await self.page.evaluate("""() => {
                const outlet = document.getElementById('artdeco-modal-outlet');
                if (!outlet) return 'no_outlet';
                // Click any <a> or <button> that looks like navigation back
                const links = outlet.querySelectorAll('a, button');
                for (const el of links) {
                    const text = (el.textContent || '').trim().toLowerCase();
                    const label = (el.getAttribute('aria-label') || '').toLowerCase();
                    if (text === '×' || text === 'x' || label.includes('close') ||
                        label.includes('dismiss') || label.includes('back')) {
                        el.click(); return 'header_click';
                    }
                }
                // Nuclear: hide the modal outlet entirely
                outlet.style.display = 'none';
                return 'nuclear';
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
            panel_closed = not await slidein.is_visible()
            self._emit_panel_close_strategy(str(s3_mode), panel_closed)
            if s3_mode == "nuclear":
                # Forcibly clearing the outlet can wedge the SPA modal manager.
                # Reload before another panel mount rather than spending retries discovering the wedge.
                print(
                    "    [panel-close] nuclear dismiss — modal outlet "
                    "force-cleared; reloading to clear the wedge"
                )
                await self.reload_search_page()
        await _retry(_do)

    def _emit_panel_close_strategy(self, strategy: str, panel_closed: bool) -> None:
        """Record which dismissal path closed the panel. Fail-soft telemetry."""
        emit_timing_event(
            self._timing_recorder,
            "panel_close_strategy",
            strategy=strategy,
            panel_closed=panel_closed,
        )

    # Failure forensics and panel-wedge recovery.

    _FORENSICS_KEEP = 5  # newest capture groups retained on disk

    async def capture_failure_forensics(self, label: str) -> str | None:
        """Capture a final browser-path failure as screenshot, HTML and context JSON under
        <state_dir>/forensics/. Return the capture stem path or None.

        This runs while the primary path is already failing: never raise, enforce the
        hard 20-second budget, and retain only the newest captures so storage stays bounded.
        """
        try:
            return await asyncio.wait_for(self._capture_forensics(label), timeout=20.0)
        except BaseException:
            return None

    async def _capture_forensics(self, label: str) -> str | None:
        if not self._state_dir or self._page is None:
            return None
        page = self._page
        forensics_dir = Path(self._state_dir) / "forensics"
        forensics_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(label)).strip("-")[:60] or "failure"
        self._forensics_seq += 1
        stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{self._forensics_seq:03d}-{safe_label}"
        base = forensics_dir / stem
        wrote_any = False
        try:
            await page.screenshot(path=str(base.with_suffix(".png")), timeout=8000)
            wrote_any = True
        except Exception:
            pass
        try:
            html = await page.content()
            base.with_suffix(".html").write_text(str(html), encoding="utf-8")
            wrote_any = True
        except Exception:
            pass
        try:
            context = {
                "label": str(label),
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "url": str(getattr(page, "url", "")),
                "viewport": getattr(page, "viewport_size", None),
            }
            base.with_suffix(".json").write_text(
                json.dumps(context, indent=2, default=str), encoding="utf-8"
            )
            wrote_any = True
        except Exception:
            pass
        self._prune_forensics(forensics_dir)
        if not wrote_any:
            return None
        print(f"    [forensics] saved forensics/{stem}")
        return str(base)

    def _prune_forensics(self, forensics_dir: Path) -> None:
        """Keep only the newest capture groups. Stems sort chronologically."""
        try:
            stems = sorted(
                {p.name.rsplit(".", 1)[0] for p in forensics_dir.iterdir() if p.is_file()}
            )
            for stale in stems[: -self._FORENSICS_KEEP] if len(stems) > self._FORENSICS_KEEP else []:
                for p in forensics_dir.glob(f"{stale}.*"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
        except Exception:
            pass

    async def reload_search_page(self) -> bool:
        """Hard-reload the current search page to reset its in-page app state.

        A panel-mount wedge survives SPA navigation, so only a document reload can clear
        it. Reload preserves the URL and query. Facet survival is not a proven contract:
        with LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED, re-verify the last reconciled
        company/title target read-only. A mismatch reports failure for caller escalation;
        never resume candidate work on an unproven sidebar.

        Do not mutate facets here: a mid-string removal reruns the search and corrupts
        page position. Return True only when results are visible and, with the flag on,
        the owned facet target still holds. Never raise.
        """
        try:
            await self.page.reload(wait_until="domcontentloaded", timeout=45000)
            await self.page.locator("ol.profile-list").first.wait_for(
                state="visible", timeout=30000
            )
            await self.page.wait_for_timeout(2500)
        except Exception as e:
            first_line = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"    [panel-recovery] reload failed: {first_line}")
            return False
        try:
            # A reload reopens the rail in its summary state.
            await self.ensure_filter_edit_panel_open()
        except Exception as e:
            print(f"    [panel-recovery] filter edit panel did not reopen: {e}")
            return False
        if (
            config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED
            and self._owned_facet_target is not None
        ):
            try:
                await self.reconcile_facets(
                    companies=set(self._owned_facet_target["companies"]),
                    titles=set(self._owned_facet_target["titles"]),
                    verify_only=True,
                )
            except FacetReconciliationError as e:
                print(
                    "    [panel-recovery] reload facet verify failed: "
                    f"{e.gate} — surface unproven, reporting reload failed"
                )
                return False
        return True

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
        """Read whether the open profile panel shows a saved state, for non-save consumers only.

        Never use this in save_candidate: the panel contains other people's links and
        cannot identify the owner of its save control. Its name check is a heuristic;
        the save path must use is_already_saved_on_card and the candidate's own result card.

        A saved panel replaces "Save to pipeline" with "Change stage". With expected_name,
        believe that state only when the visible name matches: fold accents and compare
        ordered tokens exactly. Prefix matches require explicit abbreviation in the
        shorter token (a lone initial or ellipsis); loose prefixing can identify another person.

        Fail open to plain visibility when the panel is unreadable or expected_name has
        no word characters: an unavailable DOM read is not evidence of a mismatch.
        """
        change_stage = self.page.locator(
            _PROFILE_PANEL_SCOPE + ' :is(button:has-text("Change stage"), '
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
        _PROFILE_PANEL_SCOPE + " button.save-to-pipeline__button"
    )
    # Semantic fallback: the save trigger's DOM-verified accessible name is
    # "Save to '{stage}'" (e.g. "Save to 'uncontacted'"). Mirrors the text +
    # data-test defensiveness already used by is_already_saved() for the
    # "Change stage" state, so a class rotation does not silently break saves.
    _SAVE_TRIGGER_FALLBACK = (
        _PROFILE_PANEL_SCOPE + " "
        ':is(button:has-text("Save to "), button[data-test-save-to-pipeline-button])'
    )

    # Scope save actions through the candidate's exact anchor in their own result card.
    # Panel controls have no identity-bearing ancestor and the panel includes foreign links.
    # Identity fragments are opaque path segments; reject characters outside the safe CSS-selector alphabet.
    _IDENTITY_FRAGMENT_SAFE_RE = re.compile(r"^[A-Za-z0-9_-]+$")

    # Use the main result-card save button, never its dropdown trigger. The dropdown
    # lacks the main save class/data attributes and both dropdown classes are excluded.
    # See docs/linkedin-recruiter-dom-map.md for the recorded controls.
    #
    # In one synchronous browser turn, verify project, locate the candidate's card by
    # exact title-anchor identity, resolve exactly one save control and press it. Await
    # nothing between the checks and the click. Coordinate presses are unsafe because
    # reflow can move a different card under the pointer after identity validation.
    #
    # dryRun resolves and verifies without pressing: resolution may retry, actuation happens once.
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
        // saved cards in project 9000000005 found it absent on BOTH, so the
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
        """Save through the control inside the candidate's own result card.

        Resolve through the candidate's exact anchor so a DOM swap causes a lookup failure,
        not a wrong-person click. The panel still opens and is read for evaluation.
        Already-saved detection and post-click confirmation both read that same card,
        preventing foreign-project panel state from bypassing the save boundary.

        Refuse if the card or control cannot be resolved; never fall back to the unscoped
        panel control. _last_save_failure_reason distinguishes save_card_not_found,
        save_card_ambiguous, save_trigger_not_found and save_not_persisted.
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
                # Virtualized slots have no article/title anchor until they approach the viewport.
                # Render them by scrolling in the retryable phase before the save transaction.
                # The transaction must only verify and click: no layout-changing action may occur
                # between its checks and press. Retrying the scroll does not press save.
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
