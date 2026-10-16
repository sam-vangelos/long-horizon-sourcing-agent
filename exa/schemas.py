"""Exa-specific data types and adapter functions.

The module models Exa API results and maps them to the CandidateSnippet and
CandidateProfileSummary schemas used by judgment. It performs no network calls;
the client lives in ``exa.client``.

The intermediate representation is evidence-first: an ``ExaCandidate`` carries
the retrieved pages it was assembled from and renders them into evaluation.
Because web data can be stale or fabricated, ``verify_before_outreach`` is a
structural record field rather than report prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import re

from shared.brief_v2_schema import EXA_QUERY_LANES
from shared.identity_resolution import normalize_public_linkedin_handle
from shared.schemas import (
    CandidateSnippet,
    CandidateProfileSummary,
    Education,
    Experience,
)

from exa.linkedin_render import (
    RENDER_NAME_MISMATCH_NOTE,
    LinkedInRender,
    bind_render,
    looks_like_location,
    parse_exa_linkedin_markdown,
    seat_pair,
    strip_country_suffix,
)


# Cap on the raw page text an evidence entry retains. Applied at
# ingestion (``__post_init__``), not at serialization, so the in-memory
# object and its dict are never out of step. Highlights are the signal;
# the excerpt is context for a judge that wants more, and an uncapped
# one would balloon the progress file.
#
# Matched to ``exa.acquisition.SEARCH_TEXT_CHARS``, which is what the
# search request actually asks Exa for. At 2000 the store threw away
# half of every page it paid for — on a LinkedIn render that is the
# education section and the earlier positions, cut mid-word.
EVIDENCE_TEXT_EXCERPT_CAP = 4000

# How much of an excerpt reaches the evaluation prompt. Shorter than the
# storage cap: the prompt wants enough to judge on, not the whole page.
EVIDENCE_EXCERPT_RENDER_CHARS = 600

# The same budget for the page that carries the person's identity. A
# LinkedIn profile render is not one page among several — it is the
# career history, and 600 characters of it stops inside the first
# position block. Every other host keeps the 600 above, so one press
# release still cannot spend the prompt.
EVIDENCE_LINKEDIN_EXCERPT_RENDER_CHARS = 1800

# Highlights are capped at ingestion for the same reason the excerpt is.
# Exa's highlight count is a request parameter, not a promise, and
# `to_evidence_text` renders EVERY highlight into the evaluation prompt
# (`:285-286`) with no cap of its own — so an unusually generous
# response would silently inflate both the progress file and the prompt.
EVIDENCE_HIGHLIGHT_MAX_COUNT = 10
EVIDENCE_HIGHLIGHT_MAX_CHARS = 600

_SLUG_SEPARATORS = re.compile(r"[^a-z0-9]+")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    """Lowercase hyphen-slug of an arbitrary string; "" when nothing survives."""
    return _SLUG_SEPARATORS.sub("-", value.strip().lower()).strip("-")


def derive_identity_key(name: str, linkedin_url: str, firm: str) -> str:
    """Return the stable dedupe key for an Exa-discovered person.

    The public LinkedIn handle is the key whenever one is known — it is
    what the cross-module identity service matches on, so an Exa find
    and a LinkedIn find of the same person converge instead of
    double-saving (the CoS slate saw 12 of 53 overlap). The handle
    comes from
    :func:`shared.identity_resolution.normalize_public_linkedin_handle`
    rather than a local regex, because a key derived by different rules
    than the matcher's is a key that silently fails to match: it
    already absorbs scheme, subdomain, relative ``/in/...`` paths,
    trailing slash, query string, and case.

    Without a public handle the key falls back to a name+firm slug.
    That is weaker — two people of the same name at one firm collide —
    but it keeps team-page harvests (which often carry no LinkedIn
    link) inside the same dedupe namespace instead of outside it. The
    ``linkedin.com/in/`` prefix is what tells a consumer which of the
    two it is holding.
    """

    handle = normalize_public_linkedin_handle(linkedin_url or "")
    if handle:
        return f"linkedin.com/in/{handle}"
    return _slugify(f"{name or ''} {firm or ''}")


# ---------------------------------------------------------------------------
# Exa search query (analogous to GitHubSearchQuery / SearchString)
# ---------------------------------------------------------------------------

@dataclass
class ExaSearchQuery:
    """A single query to execute against the Exa API."""
    id: int
    name: str  # human-readable description, e.g. "people_search · Benchmark"
    lane: str  # one of EXA_QUERY_LANES
    firm: str  # target firm name; "" for a firm-agnostic query
    query: str  # the natural-language Exa query
    # The brief's archetype slug, on archetype-lane queries only. It is
    # the id under which this query is REPORTED — see `coverage_key`.
    archetype_id: str = ""
    category: str | None = None  # Exa category: "people" for people_search, else None
    include_domains: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)  # namesake guard
    num_results: int = 10
    status: str = "queued"  # "queued" | "in_progress" | "done" | "skipped" | "error"
    result_count: int = 0
    candidates_discovered: int = 0
    saves: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def coverage_key(self) -> str:
        """The family this query's numbers are reported under.

        A firm lane's three queries are three angles on ONE firm, so
        they belong on one coverage row — the recruiter asks "what did
        Benchmark return", not "what did the team-page query return".
        An archetype query has no such sibling: it is a distinct
        hypothesis about a career shape, and folding all of them under
        the empty firm made six hypotheses report as one anonymous row
        where a productive archetype and a dead one were
        indistinguishable.

        The ``archetype:`` prefix keeps the two namespaces apart, so a
        firm that someone one day names "a01-mbb-to-vc" cannot collide
        with an archetype id.
        """

        if self.lane == "archetype" and self.archetype_id:
            return f"archetype:{self.archetype_id}"
        return self.firm

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExaSearchQuery:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Retrieved evidence
# ---------------------------------------------------------------------------

@dataclass
class ExaEvidence:
    """One retrieved page backing a candidate's record."""
    url: str
    title: str = ""
    published_date: str = ""
    highlights: list[str] = field(default_factory=list)
    text_excerpt: str = ""
    retrieved_at: str = field(default_factory=_now_utc_iso)

    def __post_init__(self) -> None:
        # Cap at ingestion, not at serialization. Capping in `to_dict`
        # made the round trip lossy for a long excerpt — the object
        # held 5000 chars and its dict held 2000, so save-then-load
        # silently changed the record. The object never holds more
        # than the cap.
        if len(self.text_excerpt) > EVIDENCE_TEXT_EXCERPT_CAP:
            self.text_excerpt = self.text_excerpt[:EVIDENCE_TEXT_EXCERPT_CAP]
        # Same rule for highlights, and for the same reason: the object
        # and its dict must never disagree, so the cap lands here rather
        # than in `to_dict`. Count first, then length, so a response of
        # forty highlights is trimmed before any of them is measured.
        highlights = [h for h in self.highlights if isinstance(h, str) and h]
        self.highlights = [
            h[:EVIDENCE_HIGHLIGHT_MAX_CHARS]
            for h in highlights[:EVIDENCE_HIGHLIGHT_MAX_COUNT]
        ]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExaEvidence:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_api(cls, data: dict | None) -> ExaEvidence:
        """Parse from an Exa search/contents result, tolerant of None fields."""
        data = data or {}
        highlights = data.get("highlights") or []
        return cls(
            url=data.get("url") or "",
            title=data.get("title") or "",
            published_date=data.get("publishedDate") or data.get("published_date") or "",
            highlights=[h for h in highlights if isinstance(h, str) and h],
            text_excerpt=data.get("text") or "",  # __post_init__ applies the cap
            retrieved_at=_now_utc_iso(),
        )


# ---------------------------------------------------------------------------
# Render -> profile summary helpers
# ---------------------------------------------------------------------------

_YEAR_RANGE_SPLIT = re.compile(r"\s*[-–—]\s*")


def data_sufficiency(title: str, firm: str) -> str:
    """How much of a person's current seat this record actually knows.

    Lives here rather than in ``exa.acquisition`` (which re-exports it)
    because :meth:`ExaCandidate.from_dict` has to recompute it when it
    recovers a seat off a stored page, and a second copy of this rule
    would be a rule that silently drifts.
    """

    known = bool(str(title or "").strip()) + bool(str(firm or "").strip())
    if known == 2:
        return "sufficient"
    if known == 1:
        return "minimal"
    return "insufficient"


def derive_render_from_evidence(
    evidence: list[ExaEvidence], person_name: str = ""
) -> LinkedInRender | None:
    """Parse the first public LinkedIn page on a record, if it is a render.

    Only a public LinkedIn profile page is parsed. A team page or press release
    is not this shape, and handing either to the parser would invent a career
    from a firm's roster.

    The page must belong to this person. An evidence list can contain a merged,
    misattributed, or redirected URL, so the render's ``# Name`` header must
    bind to ``person_name`` before any fields are attached. Without a name to
    bind to, nothing is attached.
    """

    for page in evidence or []:
        if not normalize_public_linkedin_handle(getattr(page, "url", "") or ""):
            continue
        render = parse_exa_linkedin_markdown(getattr(page, "text_excerpt", "") or "")
        bound = bind_render(render, person_name)
        if bound is not None:
            return bound
    return None


def _year_start(years: str) -> str:
    """"1997 - 1999" -> "1997"; a lone value is the start."""

    parts = [p for p in _YEAR_RANGE_SPLIT.split(str(years or "").strip()) if p]
    return parts[0] if parts else ""


def _year_end(years: str) -> str:
    """"1997 - 1999" -> "1999"; a lone value leaves the end empty.

    Never mirrors the start into the end. "1997-1997" would read as a
    one-year program the page never claimed.
    """

    parts = [p for p in _YEAR_RANGE_SPLIT.split(str(years or "").strip()) if p]
    return parts[1] if len(parts) >= 2 else ""


# The tenure line is Exa's arithmetic over the position dates, not a
# sentence the candidate wrote. Rendering it inline with the About text
# would put a platform-computed number inside the one block of the
# prompt that is entirely the person's own words — and the judge weighs
# first-party claims differently from derived ones. So it gets its own
# labeled line, naming who computed it.
TENURE_LABEL = "Platform-derived tenure (Exa)"


def _about_text(render: LinkedInRender) -> str:
    """The About prose, then the platform-derived tenure on its own line."""

    parts = [render.about.strip()] if render.about.strip() else []
    if render.total_experience_text:
        parts.append(f"{TENURE_LABEL}: {render.total_experience_text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Exa candidate (the intermediate representation)
# ---------------------------------------------------------------------------

@dataclass
class ExaCandidate:
    """A person assembled from one or more retrieved Exa pages."""
    identity_key: str
    name: str
    current_title: str = ""
    current_firm: str = ""
    location: str = ""
    linkedin_url: str = ""
    evidence: list[ExaEvidence] = field(default_factory=list)
    source_query_id: int = 0
    source_lane: str = ""
    # The STABLE identity of the query that found this person —
    # ``coverage_key`` (a firm, or ``archetype:<id>``) paired with
    # ``source_lane``. ``source_query_id`` is an ordinal: the compiler
    # numbers queries 1..N down the active lane list, so dropping a lane
    # or an archetype renumbers everything after it and a person
    # discovered by "query 61" is silently re-credited to whatever query
    # 61 became. This field is what the plan is keyed on; the id stays
    # for the work-unit row and for records written before it existed.
    # Additive with a trailing default, so every stored record loads.
    source_coverage_key: str = ""
    # Structural, not prose: every Exa record is web-derived, so the
    # title and firm above may already be stale. Downstream surfaces
    # read this flag; they do not grep a note for the words "verify".
    verify_before_outreach: bool = True
    data_sufficiency: str = "sufficient"  # "sufficient" | "minimal" | "insufficient"
    # The parsed LinkedIn page behind this record, as a plain dict so the
    # candidate stays JSON round-trippable. Present only when Exa
    # returned a LinkedIn profile render (the people_search lane);
    # ``None`` on a team-page harvest, where there is no such page.
    # Additive with a trailing default, so every older stored record
    # loads unchanged.
    profile_render: dict | None = None
    # Why a LinkedIn render that WAS parsed is not attached — currently
    # only ``render_name_mismatch``, meaning the page's own name header
    # is not this person. Structural so a report can count them; empty
    # is the ordinary case.
    render_note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [e.to_dict() for e in self.evidence]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> ExaCandidate:
        """Rebuild a record and recover a missing profile render from its evidence.

        Stored rows may lack ``profile_render`` while their evidence excerpt
        still carries the LinkedIn page. A pending candidate can therefore be
        rejudged with the full render rather than a thin source record.

        Title and firm move atomically through ``seat_pair``: a stored title
        paired with a firm read from the page describes no real seat. A complete
        stored pair stands; anything less is replaced by the page's pair.
        ``location`` remains fill-only. Recompute ``data_sufficiency`` after the
        recovery, but never derive ``identity_key`` again: it is the campaign's
        dedupe key and changing it would split one person on reload.
        """

        payload = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        payload["evidence"] = [
            ExaEvidence.from_dict(e) for e in (d.get("evidence") or [])
        ]
        candidate = cls(**payload)

        if candidate.profile_render:
            return candidate

        render = derive_render_from_evidence(candidate.evidence, candidate.name)
        if render is None:
            return candidate

        candidate.profile_render = render.to_dict()
        candidate.current_title, candidate.current_firm = seat_pair(
            render, candidate.current_title, candidate.current_firm
        )
        if not candidate.location:
            page_location = strip_country_suffix(render.location)
            if looks_like_location(render.location):
                candidate.location = page_location
        candidate.data_sufficiency = data_sufficiency(
            candidate.current_title, candidate.current_firm
        )
        return candidate

    def evidence_urls(self) -> list[str]:
        """Every retrieved URL backing this record, in discovery order."""
        return [e.url for e in self.evidence if e.url]

    def to_snippet(
        self,
        source_string_id: int = 0,
        source_string_name: str = "",
        page: int = 0,
        result_rank: int = 0,
    ) -> CandidateSnippet:
        """Map to ``CandidateSnippet`` for facial judgment.

        ``experience_entries`` stays empty because Exa has no career history to
        provide at this stage. Evidence reaches judgment through
        ``to_evidence_text`` and nowhere else.
        """

        headline = " at ".join(p for p in (self.current_title, self.current_firm) if p)
        return CandidateSnippet(
            name=self.name,
            headline=headline,
            current_title=self.current_title,
            current_company=self.current_firm,
            location=self.location,
            education_snippet="",  # Exa results don't carry structured education
            profile_url=self.linkedin_url or (self.evidence_urls() or [""])[0],
            source_string_id=source_string_id,
            source_string_name=source_string_name,
            page=page,
            result_rank=result_rank,
            experience_entries=[],
        )

    def render(self) -> LinkedInRender | None:
        """The LinkedIn render behind this record, or None.

        Stored when there is one; otherwise derived from the record's own
        evidence, so a candidate constructed directly — not through
        :meth:`from_dict` — still reaches the judge as a real profile.
        """

        if self.profile_render:
            return LinkedInRender.from_dict(self.profile_render)
        return derive_render_from_evidence(self.evidence, self.name)

    def _seat_experience(self) -> list[Experience]:
        """The record's own current seat as a one-entry history, or []."""

        if not (self.current_title or self.current_firm):
            return []
        return [
            Experience(
                title=self.current_title,
                company=self.current_firm,
                location=self.location,
                summary_bullets=[],
            )
        ]

    def to_profile_summary(self) -> CandidateProfileSummary:
        """Map to ``CandidateProfileSummary`` for full judgment.

        A LinkedIn render supplies dated positions, About text, and education,
        which are rendered together. A team-page harvest keeps the simpler
        shape: one dateless current seat and no additional history.

        ``summary_bullets`` carries a rendered position's own description,
        ``- `` bullets and prose lines alike, which are the person's words
        about the seat. The company blurb
        Exa appends under a position never does: it describes the firm rather
        than the person and must not become first-party evidence. A team-page
        seat has no bullets.
        """

        render = self.render()
        if render is None:
            return CandidateProfileSummary(
                name=self.name,
                profile_url=self.linkedin_url or (self.evidence_urls() or [""])[0],
                headline=" at ".join(
                    p for p in (self.current_title, self.current_firm) if p
                ),
                experiences=self._seat_experience(),
            )

        experiences = [
            Experience(
                title=entry.title,
                company=entry.company,
                location=entry.location,
                start=entry.start,
                end=entry.end,
                summary_bullets=list(entry.bullets),
            )
            for entry in render.experiences
            if entry.title or entry.company
        ]
        if not experiences:
            # A render whose Experience section was empty or unparseable
            # — a page cut before it, a shape this parser does not know.
            # The record's own seat still has to render: an empty
            # Experience list beside a populated ``current_title`` reads
            # to the judge as a person with no career, which is a claim
            # neither the page nor the record makes.
            experiences = self._seat_experience()
        education = [
            Education(
                degree=entry.degree,
                school=entry.school,
                start=_year_start(entry.years),
                end=_year_end(entry.years),
            )
            for entry in render.education
        ]
        return CandidateProfileSummary(
            name=self.name or render.name,
            profile_url=self.linkedin_url or (self.evidence_urls() or [""])[0],
            headline=render.headline
            or " at ".join(p for p in (self.current_title, self.current_firm) if p),
            experiences=experiences,
            education=education,
            about=_about_text(render),
        )

    def to_evidence_text(self) -> str:
        """Format the record for the Exa full-evaluation template."""

        lines = [
            f"Name: {self.name}",
            f"Title: {self.current_title or '(unknown)'}",
            f"Firm: {self.current_firm or '(unknown)'}",
            f"Location: {self.location or '(unknown)'}",
            f"LinkedIn: {self.linkedin_url or '(none found)'}",
        ]

        lines.append("\n═══ EVIDENCE ═══")
        if not self.evidence:
            lines.append("  (no pages retrieved)")
        for ev in self.evidence:
            lines.append(f"  URL: {ev.url}")
            lines.append(f"  Title: {ev.title or '(untitled)'}")
            lines.append(f"  Published: {ev.published_date or 'unknown'}")
            for highlight in ev.highlights:
                lines.append(f"    - {highlight}")
            # A page Exa returned without highlights still carries its
            # text. Without this the judge sees a URL, a title, and a
            # date — a citation with no content behind it — and the
            # high-bar posture drops the person for thin evidence the
            # record actually has.
            if not ev.highlights and ev.text_excerpt.strip():
                excerpt = " ".join(ev.text_excerpt.split())
                lines.append(
                    f"    Excerpt: {excerpt[:EVIDENCE_EXCERPT_RENDER_CHARS]}"
                )

        if self.verify_before_outreach:
            lines.append("\n═══ VERIFY BEFORE OUTREACH ═══")
            lines.append(
                "  Every line above comes from the open web, which lags "
                "LinkedIn. Titles and firms go stale, and firms inflate "
                "titles. Confirm this person's current seat on LinkedIn "
                "before any outreach."
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exa progress checkpoint
# ---------------------------------------------------------------------------

@dataclass
class ExaProgress:
    """Resumable checkpoint for Exa sourcing sessions."""
    brief_name: str
    queries: list[ExaSearchQuery] = field(default_factory=list)
    candidates_discovered: int = 0
    candidates_enriched: int = 0
    candidates_saved: int = 0
    candidates_rejected: int = 0
    candidates_insufficient: int = 0
    current_query_id: int | None = None
    # Global dedup, serialized as a list. Keys are what
    # `derive_identity_key` produces, so a resumed session re-derives
    # the same key and skips the same person.
    discovered_identity_keys: list[str] = field(default_factory=list)
    api_calls_made: int = 0
    exa_spend_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> ExaProgress:
        queries = [ExaSearchQuery.from_dict(q) for q in d.get("queries", [])]
        return cls(
            brief_name=d["brief_name"],
            queries=queries,
            candidates_discovered=d.get("candidates_discovered", 0),
            candidates_enriched=d.get("candidates_enriched", 0),
            candidates_saved=d.get("candidates_saved", 0),
            candidates_rejected=d.get("candidates_rejected", 0),
            candidates_insufficient=d.get("candidates_insufficient", 0),
            current_query_id=d.get("current_query_id"),
            discovered_identity_keys=d.get("discovered_identity_keys", []),
            api_calls_made=d.get("api_calls_made", 0),
            exa_spend_usd=d.get("exa_spend_usd", 0.0),
        )

    @classmethod
    def from_file(cls, path: str) -> ExaProgress:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


__all__ = [
    "EXA_QUERY_LANES",
    "EVIDENCE_EXCERPT_RENDER_CHARS",
    "EVIDENCE_LINKEDIN_EXCERPT_RENDER_CHARS",
    "RENDER_NAME_MISMATCH_NOTE",
    "TENURE_LABEL",
    "EVIDENCE_HIGHLIGHT_MAX_CHARS",
    "EVIDENCE_HIGHLIGHT_MAX_COUNT",
    "EVIDENCE_TEXT_EXCERPT_CAP",
    "ExaCandidate",
    "ExaEvidence",
    "ExaProgress",
    "ExaSearchQuery",
    "data_sufficiency",
    "derive_identity_key",
    "derive_render_from_evidence",
]
