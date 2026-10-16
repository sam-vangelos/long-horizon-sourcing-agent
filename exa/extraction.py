"""Extract named people from Exa pages.

``parse_linkedin_result`` deterministically reads the structured
``"Name - Title - Firm | LinkedIn"`` page title. ``extract_people_from_page``
uses the cheap model for unstructured team pages. Model claims pass through
``ground_profile`` both during extraction and candidate assembly: names must
be present, and unsupported title, firm, location, or quote fields are blanked.
Grounding checks the stored ``ExaEvidence.text_excerpt`` and highlights used
to build the prompt, so the validator sees exactly the evidence shown to the
model and injected claims cannot attach one person's quote to another's title.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Callable

from exa.linkedin_render import (
    RENDER_NAME_MISMATCH_NOTE,
    LinkedInRender,
    bind_render,
    parse_exa_linkedin_markdown,
    seat_pair,
    strip_country_suffix,
)
from exa.schemas import ExaEvidence
from shared.failures import ApiBudgetExhaustedError
from shared.identity_resolution import normalize_public_linkedin_handle
from shared.judgment.tool_contracts import fence_untrusted, untrusted_data_instruction
from shared.llm_clients import _parse_json_response, cheap_llm


# How many leading lines of a page's text may yield a location. A city
# line on a LinkedIn profile sits at the top; a "City, State" string
# deep in a bio is somebody's previous job, an office address, or a
# portfolio company's headquarters.
LOCATION_SCAN_LINES = 5

# Longest string that can plausibly be a "<City>, <Region>" label.
LOCATION_MAX_CHARS = 60

# Longest verbatim quote we ask the model for. Not enforced as a reject —
# a longer quote that is genuinely on the page is still honest evidence —
# but it keeps the response small.
EVIDENCE_QUOTE_MAX_CHARS = 200

# Total width of the text window a title or firm must fall inside to
# count as this person's. Applied as half on each side of the place the
# page names them. Team-page entries run 100-300 characters per person,
# so a wider window would let the person above lend their title to the
# person below — which is exactly the failure being prevented.
GROUNDING_WINDOW_CHARS = 300

# How far apart a first and last name may sit and still be one mention.
# Wide enough for "Avery Lin Holt" and "Holt, Avery", narrow enough that
# two different people in one sentence do not fuse.
NAME_TOKEN_SPAN_CHARS = 40


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

def _one_line(text: str) -> str:
    """Collapse anything destined for the run console onto one line.

    Mirrors ``exa.client._one_line`` — same contract, redefined here rather
    than imported so this module carries no transport dependency. The
    leading indent survives (the console idiom is a four-space prefix);
    every other whitespace run, newlines included, becomes one space.
    """

    raw = str(text)
    indent = raw[: len(raw) - len(raw.lstrip(" "))]
    return indent + " ".join(raw.replace("\r", " ").replace("\n", " ").split())


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def fold_ascii(value: str) -> str:
    """Strip diacritics so "Ávery" and "Avery" compare equal.

    NFKD decomposition then drop the combining marks. Deliberately duplicated
    in ``exa/identity.py`` rather than shared: importing this module there
    would drag ``shared.llm_clients`` into the dedupe path, which calls with
    no model in play.
    """

    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _collapse(value: str) -> str:
    return " ".join(str(value or "").split())


def _norm(value: str) -> str:
    """Collapsed, case-folded, diacritic-free — the comparison form."""

    return _collapse(fold_ascii(value)).lower()


def _token_bounded(needle: str, haystack: str, start: int = 0) -> int:
    """Index of ``needle`` in ``haystack`` at token boundaries, or -1.

    Uses alphanumeric lookarounds rather than ``\\b``. ``\\b`` is a transition
    between a word and a non-word character, so
    it misbehaves around punctuation; these lookarounds do not. This is
    what keeps "Partner" from matching inside "Partnerships".
    """

    if not needle or not haystack:
        return -1
    match = re.compile(
        rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    ).search(haystack, start)
    return match.start() if match else -1


def _all_token_bounded(needle: str, haystack: str) -> list[int]:
    positions: list[int] = []
    cursor = 0
    while True:
        found = _token_bounded(needle, haystack, cursor)
        if found < 0:
            return positions
        positions.append(found)
        cursor = found + 1


# ---------------------------------------------------------------------------
# Parsed profile
# ---------------------------------------------------------------------------

@dataclass
class ParsedProfile:
    """One person read off a page, before identity or judgment."""

    name: str
    title: str = ""
    firm: str = ""
    location: str = ""
    linkedin_url: str = ""
    evidence_quote: str = ""
    # The structured LinkedIn page behind this person, when the retrieved
    # text was one. Carried so acquisition can put the real experience
    # and education history on the record instead of a single dateless
    # seat. Always None on the model-extractor path: a firm's team page
    # is not a LinkedIn render.
    render: LinkedInRender | None = None
    # Why a render that WAS parsed is not attached — currently only
    # ``render_name_mismatch``. Empty when the page was not a render at
    # all, and empty when the attachment succeeded.
    render_note: str = ""


# ---------------------------------------------------------------------------
# Deterministic LinkedIn title parsing
# ---------------------------------------------------------------------------

# LinkedIn's public page title ends with a "| LinkedIn" suffix, and the
# separator before it is not always a pipe.
_LINKEDIN_SUFFIX_RE = re.compile(r"\s*[|\-–—]\s*linkedin\s*$", re.IGNORECASE)

# Segment separator: a hyphen, en dash, or em dash WITH surrounding
# space. A bare hyphen splits "Jean-Luc" and "Ex-Google"; the spaces are
# what make this safe.
_SEGMENT_SPLIT_RE = re.compile(r"\s+[-–—]\s+")

# Words that mark a segment as a job title. Singular forms only, matched
# with alphanumeric lookarounds: "Partner" is a title, "Partners" is the
# tail of a firm name ("Bessemer Venture Partners"), and "Founders Fund"
# must not read as a founder. "Lead" is deliberately absent — it is a
# real title word ("Engineering Lead") that also opens firm names ("Lead
# Edge Capital"), and the firm reading is the more common one here.
_TITLE_HINT_WORDS = (
    "adviser",
    "advisor",
    "administrator",
    "analyst",
    "architect",
    "associate",
    "ceo",
    "cfo",
    "chair",
    "chairman",
    "chief",
    "cmo",
    "co-founder",
    "cofounder",
    "consultant",
    "controller",
    "coo",
    "coordinator",
    "counsel",
    "cro",
    "cto",
    "designer",
    "developer",
    "director",
    "editor",
    "engineer",
    "engineering",
    "evp",
    "executive",
    "fellow",
    "founder",
    "head",
    "intern",
    "investing",
    "investor",
    "manager",
    "marketing",
    "officer",
    "operations",
    "operator",
    "owner",
    "partner",
    "president",
    "principal",
    "product",
    "professor",
    "recruiter",
    "researcher",
    "sales",
    "scientist",
    "specialist",
    "staff",
    "strategy",
    "student",
    "supervisor",
    "svp",
    "vp",
    "writer",
)
_TITLE_HINT_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(re.escape(word) for word in _TITLE_HINT_WORDS)
    + r")(?![a-z0-9])"
)

# Trailing tokens that mark a segment as an organization's name.
_FIRM_SUFFIX_TOKENS = frozenset(
    {
        "advisors",
        "capital",
        "fund",
        "group",
        "inc",
        "llc",
        "lp",
        "management",
        "partners",
        "ventures",
    }
)

# A proper-noun run: "San Francisco", "New York", "Sao Paulo".
_PROPER_RUN = r"[A-Z][A-Za-z.'’]*(?:\s[A-Z][A-Za-z.'’]*)*"
_LOCATION_LINE_RE = re.compile(
    rf"^{_PROPER_RUN},\s{_PROPER_RUN}(?:,\s{_PROPER_RUN})?$"
)


def _has_title_word(segment: str) -> bool:
    lowered = _norm(segment)
    if " at " in lowered or "@" in lowered:
        return True
    return bool(_TITLE_HINT_RE.search(lowered))


def _ends_with_firm_suffix(segment: str) -> bool:
    tokens = re.sub(r"[^a-z0-9\s]", " ", _norm(segment)).split()
    return bool(tokens) and tokens[-1] in _FIRM_SUFFIX_TOKENS


def _extract_location(text: str) -> str:
    """A "<City>, <Region>" line near the top of the page text, or "".

    Text only — never highlights. A highlight is a relevance-ranked
    fragment Exa chose, so its position says nothing about whose
    location it is, and a shaped fragment pulled out of the middle of a
    roster would be attributed to whoever happened to be parsed.

    Best effort by design. A location is a filter input downstream, so a
    guessed one silently removes real people; an absent one costs
    nothing but breadth. Only a line that is ENTIRELY a shaped location
    label counts — prose that happens to mention a city does not.
    """

    lines: list[str] = []
    for raw in str(text or "").splitlines():
        stripped = raw.strip()
        if stripped:
            lines.append(stripped)
        if len(lines) >= LOCATION_SCAN_LINES:
            break

    for line in lines:
        if len(line) > LOCATION_MAX_CHARS:
            continue
        if _LOCATION_LINE_RE.match(line):
            return line
    return ""


def parse_linkedin_result(result: Any) -> ParsedProfile | None:
    """Read a people-search hit's page title into a person, or None.

    Returns None when the URL is not a public LinkedIn profile (a
    ``/company/`` page, a ``/pulse/`` article, a Recruiter URL) or when
    no name survives parsing. Both are ordinary outcomes of a web
    search, not errors.

    Segment handling:

    * three or more segments — first is the name, last is the firm, the
      middle is the title;
    * two segments — the second reads as a title when it carries a title
      word and does NOT end in a firm suffix, and as a firm when it ends
      in a firm suffix and carries no title word. When it does both or
      neither ("Jane Doe - Investment Professional") both fields stay
      empty, because the page title still reaches the judge through
      ``ExaEvidence.title`` and a guess written into ``current_firm``
      does not;
    * one segment — a name and nothing else.

    Then a SECOND pass over the page text. A LinkedIn profile returned as
    structured markdown may have a bare name as its title, so the segment
    rules leave title and firm empty. :func:`~exa.linkedin_render.parse_exa_linkedin_markdown`
    reads that page; the current seat fills any field the title left blank,
    and the render rides along on ``render`` so the candidate retains its
    full experience history.

    The second pass never OVERWRITES the title parse. Both readings come
    off the same page, and the page title is LinkedIn's own summary of
    the seat; where they disagree the first read stands and the render is
    still attached for the judge.
    """

    url = getattr(result, "url", "") or ""
    if not normalize_public_linkedin_handle(url):
        return None

    raw_title = str(getattr(result, "title", "") or "")
    stripped = _LINKEDIN_SUFFIX_RE.sub("", raw_title).strip()
    if not stripped:
        return None

    segments = [part.strip() for part in _SEGMENT_SPLIT_RE.split(stripped)]
    segments = [part for part in segments if part]
    if not segments:
        return None

    name = segments[0]
    title = ""
    firm = ""
    if len(segments) >= 3:
        title = " - ".join(segments[1:-1])
        firm = segments[-1]
    elif len(segments) == 2:
        titled = _has_title_word(segments[1])
        firm_shaped = _ends_with_firm_suffix(segments[1])
        if titled and not firm_shaped:
            title = segments[1]
        elif firm_shaped and not titled:
            firm = segments[1]

    text = str(getattr(result, "text", "") or "")
    location = _extract_location(text)

    parsed_render = parse_exa_linkedin_markdown(text)
    render = bind_render(parsed_render, name)
    render_note = ""
    if parsed_render is not None and render is None:
        # A LinkedIn render that is somebody ELSE's page. Dropped, and
        # said so: grafting a stranger's career onto this record is
        # worse than the thin profile, because a thin record understates
        # while a bound-wrong one fabricates.
        render_note = RENDER_NAME_MISMATCH_NOTE

    if render is not None:
        title, firm = seat_pair(render, title, firm)
        # The render's own location line beats the positional scan: it
        # is the page's location FIELD, and it carries the shapes
        # `_extract_location` rejects ("San Francisco Bay Area (US)").
        location = strip_country_suffix(render.location) or location

    return ParsedProfile(
        name=name,
        title=title,
        firm=firm,
        location=location,
        linkedin_url=url,
        render=render,
        render_note=render_note,
    )


# ---------------------------------------------------------------------------
# Grounding — the one place that decides what a page supports
# ---------------------------------------------------------------------------

def _page_segments(evidence: ExaEvidence) -> list[str]:
    """The excerpt and each highlight, normalized, as SEPARATE strings.

    Never joined. Highlights are verbatim extracts of this same page, so
    they count as page text — but they are fragments lifted from
    different places, and concatenating them would put the partner's
    title next to the associate's name in a way the page never did. The
    window rule below measures distance, so distance has to be real:
    every check runs inside one segment.

    Position-sensitive reads use the excerpt alone —
    :func:`_extract_location` and the location branch of
    :func:`ground_profile` — because a highlight's position says nothing
    about whose location it is.
    """

    parts = [
        getattr(evidence, "text_excerpt", "") or "",
        *(getattr(evidence, "highlights", None) or []),
    ]
    return [segment for segment in (_norm(part) for part in parts) if segment]


def quote_is_grounded(quote: str, evidence: ExaEvidence) -> bool:
    """True when a quote really is on the page the evidence retained.

    Empty quotes are NOT grounded. ``"" in anything`` is true in Python,
    which would turn a missing quote into a passing one and disable the
    whole anti-fabrication check.
    """

    normalized = _norm(quote)
    if not normalized:
        return False
    return any(normalized in segment for segment in _page_segments(evidence))


def _name_anchors(name: str, corpus: str) -> list[tuple[int, int]]:
    """Spans where the page names this person.

    A single-token name anchors on that token. A multi-token name
    anchors where its FIRST and LAST tokens both appear within
    ``NAME_TOKEN_SPAN_CHARS`` of each other, in either order — which
    accepts "Avery Holt", "Avery L. Holt", and "Holt, Avery" and
    rejects a page that merely happens to contain both words far apart.
    """

    tokens = re.sub(r"[^a-z0-9\s]", " ", _norm(name)).split()
    if not tokens:
        return []
    if len(tokens) == 1:
        return [
            (start, start + len(tokens[0]))
            for start in _all_token_bounded(tokens[0], corpus)
        ]

    first_positions = _all_token_bounded(tokens[0], corpus)
    last_positions = _all_token_bounded(tokens[-1], corpus)
    anchors: list[tuple[int, int]] = []
    for first in first_positions:
        for last in last_positions:
            low, high = sorted((first, last))
            if high - low <= NAME_TOKEN_SPAN_CHARS:
                anchors.append((low, high + len(tokens[-1])))
    return anchors


def _in_name_window(
    value: str, anchored: list[tuple[str, list[tuple[int, int]]]]
) -> bool:
    """True when ``value`` sits token-bounded beside a name, in one segment."""

    normalized = _norm(value)
    if not normalized:
        return False
    half = GROUNDING_WINDOW_CHARS // 2
    for segment, anchors in anchored:
        for start, end in anchors:
            window = segment[max(0, start - half) : end + half]
            if _token_bounded(normalized, window) >= 0:
                return True
    return False


def ground_profile(
    person: ParsedProfile, evidence: ExaEvidence
) -> ParsedProfile | None:
    """Reduce a claimed person to what the retrieved page supports.

    Returns None when the page does not name the person at all, and
    otherwise a copy with every unsupported field blanked. Never mutates
    the input, and never invents: a blank means "the page did not say",
    which downstream reads as thin evidence rather than as a fact.

    Field by field:

    * ``name`` — must be anchored in the page text, and the surname must
      appear in the person's own quote. A quote that checks out proves a
      sentence exists, not that it is about this person; requiring the
      surname inside it ties the two together.
    * ``title`` and ``firm`` — kept only within
      ``GROUNDING_WINDOW_CHARS`` of where the page names the person, so
      the partner above cannot lend their title to the associate below.
    * ``location`` — from the page text only, token-bounded.
    * ``linkedin_url`` — kept only when its handle slug appears on the
      page. Exa's extracted text carries no hrefs, so a model that
      produces a URL anyway has usually produced a plausible one; and an
      invented URL is worse than none, because ``derive_identity_key``
      would mint a STRONG identity key from it and merge this record
      into a stranger's.
    """

    segments = _page_segments(evidence)
    anchored = [
        (segment, anchors)
        for segment, anchors in (
            (segment, _name_anchors(person.name, segment)) for segment in segments
        )
        if anchors
    ]
    if not anchored:
        return None

    tokens = re.sub(r"[^a-z0-9\s]", " ", _norm(person.name)).split()
    surname = tokens[-1] if tokens else ""
    if _token_bounded(surname, _norm(person.evidence_quote)) < 0:
        return None

    title = person.title if _in_name_window(person.title, anchored) else ""
    firm = person.firm if _in_name_window(person.firm, anchored) else ""

    page_text = _norm(getattr(evidence, "text_excerpt", "") or "")
    location = (
        person.location
        if _token_bounded(_norm(person.location), page_text) >= 0
        else ""
    )

    linkedin_url = ""
    slug = normalize_public_linkedin_handle(person.linkedin_url or "")
    if slug and any(_token_bounded(slug, segment) >= 0 for segment in segments):
        linkedin_url = person.linkedin_url

    return replace(
        person,
        title=title,
        firm=firm,
        location=location,
        linkedin_url=linkedin_url,
    )


# ---------------------------------------------------------------------------
# Cheap-tier page extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You extract people from ONE retrieved web page for a recruiting search.

""" + untrusted_data_instruction() + """

Binding rules:
- Extract ONLY people explicitly named on the page. Never add a person the page does not name. Never complete a partial roster from what you know about the firm.
- Copy each person's title EXACTLY as the page writes it. Never infer, normalize, seniority-adjust, or upgrade a title. If the page gives a person no title, leave "title" empty.
- "firm" is the organization the page says the person works at, written as the page writes it, and it must appear on the page beside that person. If the page does not say, leave it empty.
- "location" only when the page states it for that person. Otherwise empty.
- "linkedin_url" only when the page shows a LinkedIn profile URL for that person. Otherwise empty.
- "evidence_quote" is a VERBATIM substring of the page text below, at most 200 characters, copied character for character. It must contain the person's name, and — whenever you report a title — the exact title text you reported. A person whose quote is not found on the page is discarded.

Return JSON and nothing else:
{"people": [{"name": "", "title": "", "firm": "", "location": "", "linkedin_url": "", "evidence_quote": ""}]}

Return {"people": []} when the page names nobody."""


def _build_user_prompt(evidence: ExaEvidence, firm_hint: str) -> str:
    # The firm hint is the operator's own search input; everything after it
    # was written by whoever published the page, so it goes inside the fence.
    page = [
        f"PAGE URL: {evidence.url or '(unknown)'}",
        f"PAGE TITLE: {evidence.title or '(untitled)'}",
        "",
        "PAGE TEXT:",
        evidence.text_excerpt or "(no text retrieved)",
    ]
    if evidence.highlights:
        page.append("")
        page.append("PAGE HIGHLIGHTS:")
        page.extend(f"- {highlight}" for highlight in evidence.highlights)
    return "\n".join(
        [
            f"FIRM BEING SEARCHED: {firm_hint or '(none given)'}",
            fence_untrusted("\n".join(page)),
        ]
    )


def _coerce_people(payload: Any) -> list[dict]:
    """Pull the people list out of whatever shape came back."""

    if isinstance(payload, str):
        payload = _parse_json_response(payload)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("people")
        if rows is None:
            raise ValueError("response carries no 'people' key")
    else:
        raise ValueError(f"unexpected response type {type(payload).__name__}")
    if not isinstance(rows, list):
        raise ValueError("'people' is not a list")
    return [row for row in rows if isinstance(row, dict)]


def extract_people_from_page(
    evidence: ExaEvidence,
    *,
    firm_hint: str,
    llm: Callable = cheap_llm,
    usage_context: dict | None = None,
) -> list[ParsedProfile]:
    """Read the people a team page or article names, via the cheap model.

    Fail-soft for the faults that are one page's problem: a malformed
    response, an unparseable body, a transport error. Those cost one
    console line and an empty list, because a campaign with eighty more
    queries must not end on a bad page.

    NOT fail-soft for spend exhaustion. ``ApiBudgetExhaustedError`` from
    the cheap tier means the RUN is out of budget, not that this page is
    bad — absorbing it would turn a stop into a silent stream of empty
    pages, and the campaign would run its whole query list producing
    nobody while looking healthy on the console.

    Every surviving person is passed through :func:`ground_profile`, so
    what comes back is what the page supports.
    """

    url = getattr(evidence, "url", "") or "(unknown url)"
    try:
        context = {
            **(usage_context or {}),
            # Canonical keys land LAST: a caller that passes a stale
            # `source` or `stage` must not relabel this call's receipts.
            "stage": "exa_extraction",
            "source": "exa",
            "module": "exa",
        }
        response = llm(
            _EXTRACTION_SYSTEM_PROMPT,
            _build_user_prompt(evidence, firm_hint),
            expect_json=True,
            usage_context=context,
        )
        rows = _coerce_people(response)
    except ApiBudgetExhaustedError:
        raise
    except Exception as exc:  # fail-soft: never raise into acquisition
        print(_one_line(f"    [exa] extraction failed for {url}: {exc}"), flush=True)
        return []

    people: list[ParsedProfile] = []
    for row in rows:
        name = _collapse(row.get("name"))
        if not name:
            continue
        claimed = ParsedProfile(
            name=name,
            title=_collapse(row.get("title")),
            firm=_collapse(row.get("firm")),
            location=_collapse(row.get("location")),
            linkedin_url=_collapse(row.get("linkedin_url")),
            evidence_quote=_collapse(row.get("evidence_quote")),
        )
        if not quote_is_grounded(claimed.evidence_quote, evidence):
            continue
        grounded = ground_profile(claimed, evidence)
        if grounded is not None:
            people.append(grounded)
    return people


__all__ = [
    "EVIDENCE_QUOTE_MAX_CHARS",
    "GROUNDING_WINDOW_CHARS",
    "LOCATION_MAX_CHARS",
    "LOCATION_SCAN_LINES",
    "NAME_TOKEN_SPAN_CHARS",
    "RENDER_NAME_MISMATCH_NOTE",
    "ParsedProfile",
    "extract_people_from_page",
    "fold_ascii",
    "ground_profile",
    "parse_linkedin_result",
    "quote_is_grounded",
]
