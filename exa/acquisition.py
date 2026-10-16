"""Exa acquisition — one compiled query to candidate records.

The unit of work is ONE query and ONE Exa call. Contents ride on the
search request rather than a follow-up ``/contents`` call, because the
search already returns text and highlights: a second call would double
the per-page content charge to retrieve pages the first call already
paid for.

Everything a query produces is returned in one
:class:`AcquisitionOutcome` — the candidates, the count of results that
came back, why the rest were dropped, and how many Exa calls it cost.
Nothing here writes state, prints a report, or judges anybody. S5 owns
persistence and resume; this module is the pure function it calls.

Drops are counted, not just discarded. "23 results, 4 candidates" reads
as a broken query; "23 results, 4 candidates (namesake 15, not_profile
4)" reads as a firm whose name collides with a bigger one — a brief
problem the recruiter can fix, and the distinction only exists because
the reason is kept.

Assembly re-grounds every person an extractor hands over, through the
same :func:`exa.extraction.ground_profile` the extractor itself used.
The extractor is a parameter, and the record is born here, so the
invariant the record publishes is enforced here.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import aiohttp

from exa.client import ExaAuthError, ExaCreditsError
from exa.cost_meter import ExaBudgetExhaustedError
from exa.extraction import (
    ParsedProfile,
    _one_line,
    extract_people_from_page,
    ground_profile,
    parse_linkedin_result,
)
from exa.schemas import (
    ExaCandidate,
    ExaEvidence,
    ExaSearchQuery,
    # Defined on the schema, re-exported here so every existing caller
    # keeps its import. ``ExaCandidate.from_dict`` recomputes the band
    # when it recovers a seat off a stored page, and two copies of the
    # rule would be a rule that drifts.
    data_sufficiency,
    derive_identity_key,
)
from shared.failures import ApiBudgetExhaustedError
from shared.identity_resolution import (
    normalize_company_name,
    normalize_public_linkedin_handle,
)


# How much of each page the search request asks for. ``maxCharacters``
# on text is deliberately larger than the highlight budget: highlights
# are the signal, the text is what the extractor reads a team roster out
# of.
SEARCH_HIGHLIGHT_CHARS = 400
SEARCH_TEXT_CHARS = 4000

# The drop reasons a query can report. Named as a tuple so the console
# summary and any consumer render the same vocabulary.
DROP_REASONS = (
    "namesake",
    "not_profile",
    "no_name",
    "extraction_empty",
    "fabrication_guard",
)

# The lanes whose results ARE public LinkedIn profiles, parsed
# directly rather than run through the page extractor. `people_search`
# filters a firm × title string to `linkedin.com`; `archetype` (S8b)
# filters a natural-language description of a career shape to the same
# place. Both return profile pages, so both take the same path — parse
# the render, drop anything that is not a profile URL as `not_profile`
# — and neither pays for an extraction call on a page whose structure
# is already known.
PEOPLE_PROFILE_LANES = frozenset({"people_search", "archetype"})

# Hosts that serve other people's profiles rather than a firm's own
# site. A people-search query filters on ``linkedin.com``, so treating a
# domain-filter hit there as firm confirmation would make every LinkedIn
# result inherit the query's firm — the assumption F4 exists to stop.
GENERIC_PROFILE_HOSTS = frozenset({"linkedin.com"})

_SEPARATORS = re.compile(r"[-_\s]+")


# How an absorbed search failure is written into the query's note. The
# resume path reads it back (:func:`is_transient_query_failure`), so the
# producer and the reader share the one prefix.
QUERY_FAILURE_PREFIX = "exa search failed: "


def _exception_family_names(root: type[BaseException]) -> frozenset[str]:
    names: set[str] = set()
    pending = [root]
    while pending:
        cls = pending.pop()
        if cls.__name__ in names:
            continue
        names.add(cls.__name__)
        pending.extend(cls.__subclasses__())
    return frozenset(names)


# Exception types whose absorbed failure says nothing about the query itself:
# the network or Exa's edge failed, and the same request later can succeed.
# Every connection-level aiohttp fault (DNS, TCP, TLS, disconnects, socket and
# connect timeouts), a body that broke mid-read or arrived as the wrong
# content type, and a bare timeout. A 4xx is the query's own fault and is not
# here: it is raised as ``RuntimeError("Exa API error 4xx ...")``.
_TRANSIENT_FAILURE_TYPES = (
    _exception_family_names(aiohttp.ClientConnectionError)
    | _exception_family_names(aiohttp.ClientPayloadError)
    | frozenset({"ContentTypeError", "TimeoutError"})
)

# The client's own retry-exhaustion message: five attempts that all hit a
# 429, a 5xx, or a transport fault.
_RETRIES_EXHAUSTED_RE = re.compile(r"^Exa API request failed after \d+ retries")

_FAILURE_NOTE_RE = re.compile(
    rf"^{re.escape(QUERY_FAILURE_PREFIX)}(?P<kind>[A-Za-z_][A-Za-z0-9_]*): ?(?P<detail>.*)$",
    re.DOTALL,
)


def is_transient_query_failure(note: str) -> bool:
    """Whether a query's failure note records a fault worth running it again for.

    Reads the note :func:`run_query` writes when it absorbs a search
    failure. True for a network or edge fault (see
    ``_TRANSIENT_FAILURE_TYPES``) and for the client's retry exhaustion;
    False for anything else, a 4xx included, so a resume never re-pays for
    a request Exa already refused on its merits.
    """

    match = _FAILURE_NOTE_RE.match(str(note or "").strip())
    if match is None:
        return False
    kind = match.group("kind")
    if kind in _TRANSIENT_FAILURE_TYPES:
        return True
    return kind == "RuntimeError" and bool(
        _RETRIES_EXHAUSTED_RE.match(match.group("detail"))
    )


@dataclass
class AcquisitionOutcome:
    """Everything one executed query produced."""

    query_id: int
    lane: str
    result_count: int = 0
    candidates: list[ExaCandidate] = field(default_factory=list)
    # Sparse: a reason with no drops is absent, so the console summary
    # and any log show what actually happened rather than four zeroes.
    dropped: dict[str, int] = field(default_factory=dict)
    exa_calls: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# Namesake guard
# ---------------------------------------------------------------------------

def _sep_norm(value: str) -> str:
    """Fold hyphens, underscores, and whitespace to single spaces.

    "Accel-KKR", "Accel KKR", and "accel_kkr" are one firm written three
    ways, and an exclude term written one way must catch all three.
    """

    return _SEPARATORS.sub(" ", str(value or "")).strip().lower()


def _term_hits(term_norm: str, haystack_norm: str) -> bool:
    """Check whether ``term_norm`` occurs as a standalone term in ``haystack_norm``.

    Alphanumeric lookarounds avoid the trailing-punctuation failure of ``\\b``
    while keeping short employer names from matching unrelated longer names.
    """
    if not term_norm or not haystack_norm:
        return False
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])", haystack_norm
        )
    )


def _namesake_term(result: Any, exclude_terms: list[str]) -> str | None:
    """The first exclude term this result's TITLE or URL carries.

    Page text is deliberately not scanned. A genuine Accel employee
    whose profile mentions a co-investment with Accel-KKR is exactly the
    person the campaign wants, and dropping them on a prose mention
    trades a real candidate for nothing. The title and the URL are where
    a result's own identity lives; the parsed firm is checked separately
    once it exists.
    """

    if not exclude_terms:
        return None
    haystack = _sep_norm(
        " ".join(
            str(part or "")
            for part in (
                getattr(result, "title", ""),
                getattr(result, "url", ""),
            )
        )
    )
    for term in exclude_terms:
        if _term_hits(_sep_norm(term), haystack):
            return term
    return None


def _firm_is_namesake(firm: str, exclude_terms: list[str]) -> bool:
    """True when a PARSED firm token-matches an exclude term."""

    firm_norm = _sep_norm(normalize_company_name(firm))
    if not firm_norm:
        return False
    for term in exclude_terms:
        term_norm = _sep_norm(normalize_company_name(term))
        if not term_norm:
            continue
        if firm_norm == term_norm or _term_hits(term_norm, firm_norm):
            return True
    return False


# ---------------------------------------------------------------------------
# Candidate assembly
# ---------------------------------------------------------------------------

def _host(url: str) -> str:
    """Lowercase host of a URL, without ``www.``."""

    raw = str(url or "").strip().lower()
    raw = re.sub(r"^[a-z][a-z0-9+.\-]*://", "", raw)
    host = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    host = host.split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _registrable_label(host: str) -> str:
    """The name part of a host: ``cedarlinecapital`` from the .com."""

    parts = [part for part in host.split(".") if part]
    return parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")


def firm_is_grounded_by_page(
    *, firm: str, evidence_url: str, include_domains: list[str]
) -> bool:
    """True when the retrieved page is the firm's OWN site.

    This is the only circumstance in which a query's target firm may be
    written into a candidate record as that person's employer. A query
    encodes search INTENT — "look for chiefs of staff at Cedarline" — and
    a result is not proof the intent was satisfied. A page served from
    ``cedarlinecapital.com`` is proof, because the firm published it.

    A domain-filter match counts, except on the hosts that serve other
    people's profiles: a people-search query filters on ``linkedin.com``
    for every firm in the brief, so honoring that filter would make the
    assumption universal instead of removing it.
    """

    host = _host(evidence_url)
    if not host:
        return False
    firm_norm = re.sub(r"[^a-z0-9]", "", _sep_norm(normalize_company_name(firm)))
    if firm_norm and firm_norm == _registrable_label(host):
        return True
    if host in GENERIC_PROFILE_HOSTS:
        return False
    return any(_host(domain) == host for domain in include_domains or [])


def _evidence_for(evidence: ExaEvidence, quote: str) -> ExaEvidence:
    """Return a per-candidate copy of the page carrying that person's quote.

    The quote is the sentence that earned the candidate's place in the run. Put
    it first because ``to_evidence_text`` may otherwise render generic page
    highlights without the person-specific line.

    Each candidate needs its own copy: one team page can produce several people,
    and sharing an evidence object would attach every person's quote to every
    record.
    """

    data = evidence.to_dict()
    if quote:
        data["highlights"] = [quote, *(data.get("highlights") or [])]
    return ExaEvidence.from_dict(data)


def _candidate_from_profile(
    parsed: ParsedProfile,
    *,
    query: ExaSearchQuery,
    evidence: ExaEvidence,
) -> ExaCandidate:
    firm = parsed.firm
    if not firm and firm_is_grounded_by_page(
        firm=query.firm,
        evidence_url=evidence.url,
        include_domains=query.include_domains or [],
    ):
        firm = query.firm
    linkedin_url = parsed.linkedin_url
    return ExaCandidate(
        identity_key=derive_identity_key(parsed.name, linkedin_url, firm),
        name=parsed.name,
        current_title=parsed.title,
        current_firm=firm,
        location=parsed.location,
        linkedin_url=linkedin_url,
        evidence=[_evidence_for(evidence, parsed.evidence_quote)],
        source_query_id=query.id,
        source_lane=query.lane,
        source_coverage_key=query.coverage_key,
        # Parsed render fields must be present before data sufficiency is computed;
        # otherwise a people-search record with render-derived title/firm data is
        # incorrectly classified as insufficient.
        data_sufficiency=data_sufficiency(parsed.title, firm),
        profile_render=parsed.render.to_dict() if parsed.render else None,
        render_note=parsed.render_note,
    )


def _merge(candidates: list[ExaCandidate]) -> list[ExaCandidate]:
    """Fold candidates sharing an identity key, preserving order.

    One team page and one LinkedIn hit inside the same query are the
    same person twice, and everything either read belongs on one record.
    A field the first read left BLANK is filled from a later one — the
    first read is not more authoritative than the second, it is only
    earlier, and a blank is an absence rather than a finding. A field
    both reads filled keeps the first value.

    Evidence from the same URL merges its highlights instead of being
    discarded, because two reads of one page carry two people's quotes
    and dropping the second would throw away the very sentence
    :func:`_evidence_for` was added to preserve.
    """

    merged: dict[str, ExaCandidate] = {}
    for candidate in candidates:
        existing = merged.get(candidate.identity_key)
        if existing is None:
            merged[candidate.identity_key] = candidate
            continue
        by_url = {ev.url: ev for ev in existing.evidence if ev.url}
        for evidence in candidate.evidence:
            twin = by_url.get(evidence.url) if evidence.url else None
            if twin is None:
                existing.evidence.append(evidence)
                if evidence.url:
                    by_url[evidence.url] = evidence
                continue
            for highlight in evidence.highlights:
                if highlight not in twin.highlights:
                    twin.highlights.append(highlight)
        for attribute in (
            "current_title",
            "current_firm",
            "location",
            "linkedin_url",
            # A team-page read carries no render; a LinkedIn read of the
            # same person inside the same query does. Whichever arrives
            # second must not leave the merged record without it.
            "profile_render",
        ):
            if not getattr(existing, attribute) and getattr(candidate, attribute):
                setattr(existing, attribute, getattr(candidate, attribute))
        existing.data_sufficiency = data_sufficiency(
            existing.current_title, existing.current_firm
        )
    return list(merged.values())


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

def _drop_summary(dropped: dict[str, int]) -> str:
    parts = [
        f"{reason} {dropped[reason]}"
        for reason in DROP_REASONS
        if dropped.get(reason)
    ]
    extra = sorted(set(dropped) - set(DROP_REASONS))
    parts.extend(f"{reason} {dropped[reason]}" for reason in extra if dropped[reason])
    return ", ".join(parts) if parts else "no drops"


def _status_line(client: Any) -> str:
    """The client's spend line, or a placeholder.

    Console rendering is never allowed to break acquisition, and a
    duck-typed client is not obliged to meter anything.
    """

    try:
        return " ".join(str(client.status_line()).split())
    except Exception:
        return "spend unavailable"


# ---------------------------------------------------------------------------
# The query runner
# ---------------------------------------------------------------------------

async def run_query(
    query: ExaSearchQuery,
    *,
    client: Any,
    extractor: Callable = extract_people_from_page,
    usage_context: dict | None = None,
) -> AcquisitionOutcome:
    """Execute one compiled query and return the people it produced.

    Only the explicit transient tuple (``RuntimeError``, ``aiohttp.ClientError``,
    ``asyncio.TimeoutError``) is absorbed into a query note so later queries can
    continue. ``ApiBudgetExhaustedError``, ``ExaAuthError``, ``ExaCreditsError``,
    and ``ValueError`` propagate explicitly, as do every other exception not in
    that transient tuple.

    An absorbed failure still reports ``exa_calls=1`` because the request was
    attempted. The cost meter charges before dispatch and releases every
    attempt Exa provably never billed (never connected, or a non-2xx answer
    with no usage record); this count is the call ledger, not the spend ledger.
    """

    dropped: dict[str, int] = {}

    def _drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    try:
        response = await client.search(
            query.query,
            category=query.category,
            num_results=query.num_results,
            include_domains=query.include_domains or None,
            contents={
                "highlights": {
                    "query": query.query,
                    "maxCharacters": SEARCH_HIGHLIGHT_CHARS,
                },
                "text": {"maxCharacters": SEARCH_TEXT_CHARS},
            },
        )
    except (
        ApiBudgetExhaustedError,
        ExaAuthError,
        ExaCreditsError,
        ValueError,
    ):
        raise
    except (RuntimeError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
        note = _one_line(f"{QUERY_FAILURE_PREFIX}{type(exc).__name__}: {exc}")
        print(
            _one_line(
                f"    [exa] q{query.id} {query.lane} · {query.firm or query.archetype_id or '—'}: "
                f"error — {note} | {_status_line(client)}"
            ),
            flush=True,
        )
        return AcquisitionOutcome(
            query_id=query.id,
            lane=query.lane,
            result_count=0,
            candidates=[],
            dropped=dropped,
            exa_calls=1,
            notes=note,
        )

    results = list(getattr(response, "results", None) or [])
    exclude_terms = [
        term for term in (query.exclude_terms or []) if str(term or "").strip()
    ]
    extraction_context = {
        **(usage_context or {}),
        # Canonical last, so a caller's stale lane or query id cannot
        # mislabel this query's receipts.
        "lane_id": query.lane,
        "string_id": query.id,
    }

    collected: list[ExaCandidate] = []
    for result in results:
        if _namesake_term(result, exclude_terms) is not None:
            _drop("namesake")
            continue

        evidence = ExaEvidence.from_api(_result_dict(result))

        if query.lane in PEOPLE_PROFILE_LANES:
            if not normalize_public_linkedin_handle(
                getattr(result, "url", "") or ""
            ):
                _drop("not_profile")
                continue
            parsed = parse_linkedin_result(result)
            if parsed is None:
                _drop("no_name")
                continue
            if _firm_is_namesake(parsed.firm, exclude_terms):
                _drop("namesake")
                continue
            collected.append(
                _candidate_from_profile(parsed, query=query, evidence=evidence)
            )
            continue

        people = extractor(
            evidence,
            firm_hint=query.firm,
            usage_context=extraction_context,
        )
        if not people:
            _drop("extraction_empty")
            continue
        for person in people:
            grounded = ground_profile(person, evidence)
            if grounded is None:
                _drop("fabrication_guard")
                continue
            if _firm_is_namesake(grounded.firm, exclude_terms):
                _drop("namesake")
                continue
            collected.append(
                _candidate_from_profile(grounded, query=query, evidence=evidence)
            )

    candidates = _merge(collected)

    print(
        _one_line(
            f"    [exa] q{query.id} {query.lane} · {query.firm or query.archetype_id or '—'}: "
            f"{len(results)} results → {len(candidates)} candidates "
            f"({_drop_summary(dropped)}) | {_status_line(client)}"
        ),
        flush=True,
    )

    return AcquisitionOutcome(
        query_id=query.id,
        lane=query.lane,
        result_count=len(results),
        candidates=candidates,
        dropped=dropped,
        exa_calls=1,
        notes="",
    )


def _result_dict(result: Any) -> dict:
    """The API-shaped dict ``ExaEvidence.from_api`` expects.

    ``ExaResult.to_dict`` emits ``published_date``; ``from_api`` reads
    either spelling, so the round trip is lossless.
    """

    if isinstance(result, dict):
        return result
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {
        "url": getattr(result, "url", "") or "",
        "title": getattr(result, "title", "") or "",
        "published_date": getattr(result, "published_date", "") or "",
        "highlights": list(getattr(result, "highlights", None) or []),
        "text": getattr(result, "text", "") or "",
    }


__all__ = [
    "DROP_REASONS",
    "GENERIC_PROFILE_HOSTS",
    "PEOPLE_PROFILE_LANES",
    "SEARCH_HIGHLIGHT_CHARS",
    "SEARCH_TEXT_CHARS",
    "QUERY_FAILURE_PREFIX",
    "AcquisitionOutcome",
    "data_sufficiency",
    "firm_is_grounded_by_page",
    "is_transient_query_failure",
    "run_query",
]
