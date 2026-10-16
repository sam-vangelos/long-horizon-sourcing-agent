"""Compile an Exa brief into executable queries and lint findings.

The compiler is deterministic and offline. Firm lanes cover people search,
team-page harvests, and operating signals; the archetype lane searches
firm-independent career-shape descriptions from ``exa_archetype_queries``.

This source-native compiler does not implement the shared ``LaneCompiler``
protocol because Exa reads its query inputs directly from the brief. It reuses
the shared finding envelope so compiler output has one reporting shape.
"""

from __future__ import annotations

import re
from typing import Any

from exa.schemas import EXA_QUERY_LANES, ExaSearchQuery
from exa.location_gate import LOCATION_FIELD, location_scope_errors
from exa.scope_gate import SCOPE_FIELD, scope_errors
from shared.brief_v2_schema import (
    EXA_ARCHETYPE_DEFAULT_INCLUDE_DOMAINS,
    EXA_ARCHETYPE_DEFAULT_NUM_RESULTS,
    EXA_ARCHETYPE_NUM_RESULTS_MAX,
    EXA_ARCHETYPE_NUM_RESULTS_MIN,
)
from shared.lane_compilers import LaneCompilerFinding

# Short one-token firm names can collide with ordinary words or unrelated firms.
# ``Accel`` and ``Accel-KKR`` illustrate the namesake class this threshold catches.
NAMESAKE_TOKEN_MAX_CHARS = 6

_WHITESPACE = re.compile(r"\s+")


def _collapse(value: str) -> str:
    """Squash runs of whitespace so an absent clause leaves no double space."""
    return _WHITESPACE.sub(" ", value).strip()


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _token_overlap(name: str, pattern_norm: str) -> bool:
    """Check boundary-respecting containment in either direction.

    Alphanumeric lookarounds avoid ``\\b``'s trailing-punctuation edge case,
    while equality and containment keep short names from matching longer firms.
    """
    if not name or not pattern_norm:
        return False
    if name == pattern_norm:
        return True
    return bool(
        re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", pattern_norm)
        or re.search(rf"(?<![a-z0-9]){re.escape(pattern_norm)}(?![a-z0-9])", name)
    )


# ---------------------------------------------------------------------------
# Brief readers — tolerant of both Brief dataclasses
# ---------------------------------------------------------------------------


def _target_firms(brief: Any) -> list[dict]:
    """Well-formed firm entries only; a nameless entry compiles nothing."""
    firms = getattr(brief, "exa_target_firms", None) or []
    if not isinstance(firms, list):
        return []
    return [
        f
        for f in firms
        if isinstance(f, dict)
        and isinstance(f.get("name"), str)
        and f["name"].strip()
    ]


def _active_lanes(brief: Any) -> tuple[str, ...]:
    """The brief's lane subset in canonical order; empty/absent = all three.

    Canonical order rather than the brief's own so query ids are stable
    across two briefs that name the same lanes in a different order.
    """
    lanes = getattr(brief, "exa_query_lanes", None) or []
    if not isinstance(lanes, list):
        return EXA_QUERY_LANES
    requested = {lane for lane in lanes if isinstance(lane, str) and lane}
    if not requested:
        return EXA_QUERY_LANES
    return tuple(lane for lane in EXA_QUERY_LANES if lane in requested)


def active_lanes(brief: Any) -> tuple[str, ...]:
    """Public name for the brief's lane subset.

    The resume plan merge needs it: a lane the brief dropped compiles to
    nothing, so its absence from the compiled query list is
    indistinguishable from a lane that simply has no firms. Only the
    brief can tell those apart, and this is how it says so.
    """

    return _active_lanes(brief)


def _title_terms(brief: Any) -> list[str]:
    terms = getattr(brief, "exa_title_terms", None) or []
    if not isinstance(terms, list):
        return []
    return [t.strip() for t in terms if isinstance(t, str) and t.strip()]


def _firm_title_terms(firm: dict, default: list[str]) -> list[str]:
    """A firm's own ``title_terms`` when it names them, else the brief's.

    A firm whose staff are mostly out of scope (a lab whose engineers would
    flood the slate) is searched with its own seat titles only: req 1238's
    clients count only in human-data, post-training and evaluation seats.
    """

    terms = firm.get("title_terms")
    if isinstance(terms, list):
        own = [t.strip() for t in terms if isinstance(t, str) and t.strip()]
        if own:
            return own
    return default


def _archetype_queries(brief: Any) -> list[dict]:
    """Well-formed archetype entries only; id and text are both required.

    An entry missing either compiles nothing: an id-less query cannot be
    named on the slate, and a text-less one has nothing to search for.
    :func:`shared.brief_v2_schema.validate_exa_fields` has already
    refused both at load, so this is the same belt-and-braces the firm
    reader keeps for blank-string edges.
    """

    entries = getattr(brief, "exa_archetype_queries", None) or []
    if not isinstance(entries, list):
        return []
    return [
        e
        for e in entries
        if isinstance(e, dict)
        and isinstance(e.get("id"), str)
        and e["id"].strip()
        and isinstance(e.get("text"), str)
        and e["text"].strip()
    ]


def _archetype_num_results(entry: dict) -> int:
    """The entry's result count, or the default when it names none.

    Out-of-range and non-integer values fall back rather than raise:
    the validator is the gate, and a compiler that raised here would
    turn an authoring typo into a crashed launch.
    """

    num = entry.get("num_results")
    if isinstance(num, bool) or not isinstance(num, int):
        return EXA_ARCHETYPE_DEFAULT_NUM_RESULTS
    if not (
        EXA_ARCHETYPE_NUM_RESULTS_MIN <= num <= EXA_ARCHETYPE_NUM_RESULTS_MAX
    ):
        return EXA_ARCHETYPE_DEFAULT_NUM_RESULTS
    return num


def _archetype_include_domains(entry: dict) -> list[str]:
    """The entry's domain allowlist, or LinkedIn.

    An explicit empty list reads as "unset", not as "search the open
    web": the archetype lane takes the people_search path, which drops
    every result that is not a public profile URL, so an unfiltered
    archetype query pays for pages it then throws away as
    `not_profile`.
    """

    domains = entry.get("include_domains")
    if not isinstance(domains, list):
        return list(EXA_ARCHETYPE_DEFAULT_INCLUDE_DOMAINS)
    cleaned = [d.strip() for d in domains if isinstance(d, str) and d.strip()]
    return cleaned or list(EXA_ARCHETYPE_DEFAULT_INCLUDE_DOMAINS)


def _geography(brief: Any) -> str:
    """Read geography from either the V2 or compatibility Brief shape.

    V2 carries ``geography`` directly; the compatibility object carries it in
    ``permanent_filters["Location"]``.
    """
    geography = getattr(brief, "geography", "")
    if isinstance(geography, str) and geography.strip():
        return geography.strip()
    filters = getattr(brief, "permanent_filters", None)
    if isinstance(filters, dict):
        location = filters.get("Location", "")
        if isinstance(location, str) and location.strip():
            return location.strip()
    return ""


def _hiring_company(brief: Any) -> str:
    """Return the employer this search must never source from.

    V2 stores it in ``engagement_context["hiring_company"]``; the compatibility
    object may expose the same value at the top level of its raw payload.
    """
    context = getattr(brief, "engagement_context", None)
    if isinstance(context, dict):
        value = context.get("hiring_company")
        if isinstance(value, str) and value.strip():
            return value.strip()
    raw = getattr(brief, "raw", None)
    if isinstance(raw, dict):
        value = raw.get("hiring_company")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _firm_aliases(firm: dict) -> list[str]:
    """The firm's other names, deduped against the primary, order kept."""

    seen = {_norm(firm.get("name"))}
    aliases: list[str] = []
    for alias in firm.get("aliases") or []:
        if not isinstance(alias, str) or not alias.strip():
            continue
        key = _norm(alias)
        if key in seen:
            continue
        seen.add(key)
        aliases.append(alias.strip())
    return aliases


def _firm_expression(firm_name: str, firm: dict) -> str:
    """Name a firm in a query using its primary name and aliases.

    A profile may carry either a partnership's long name or its short name, so
    aliases are rendered in parentheses alongside the primary name. Exa is
    semantic rather than Boolean, making this clearer than an OR expression.
    """

    aliases = _firm_aliases(firm)
    if not aliases:
        return firm_name
    return f"{firm_name} ({', '.join(aliases)})"


def _employer_blacklist(brief: Any) -> list[str]:
    blacklist = getattr(brief, "employer_blacklist", None) or []
    if not isinstance(blacklist, list):
        return []
    return [b.strip() for b in blacklist if isinstance(b, str) and b.strip()]


def _alumni_blacklist(brief: Any) -> list[str]:
    """Employers whose alumni the gates reject; a target firm here is a conflict."""
    blacklist = getattr(brief, "alumni_blacklist", None) or []
    if not isinstance(blacklist, list):
        return []
    return [b.strip() for b in blacklist if isinstance(b, str) and b.strip()]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_exa_queries(brief: Any) -> list[ExaSearchQuery]:
    """Compile a brief into the Exa queries a run executes.

    One query per (lane, firm) pair, in canonical lane order then brief
    firm order, ids sequential from 1. A brief with no target firms
    falls back to one people-search per title term over the brief's
    geography — the campaign still runs, just without the firm
    allowlist that makes the results precise.

    Archetype queries are appended LAST, in brief order, after every
    firm lane. Two reasons: the firm lanes are the precise half of the
    sweep and should spend the budget first, and appending keeps every
    existing query's id exactly where it was, so adding an archetype to
    a brief mid-campaign does not renumber the work units a resume is
    matching against.
    """

    firms = _target_firms(brief)
    lanes = _active_lanes(brief)
    title_terms = _title_terms(brief)
    geography = _geography(brief)

    queries: list[ExaSearchQuery] = []

    def _next_id() -> int:
        return len(queries) + 1

    def _append_archetypes() -> None:
        if "archetype" not in lanes:
            return
        for entry in _archetype_queries(brief):
            entry_id = entry["id"].strip()
            queries.append(
                ExaSearchQuery(
                    id=_next_id(),
                    name=f"archetype · {entry_id}",
                    lane="archetype",
                    firm="",
                    archetype_id=entry_id,
                    query=_collapse(entry["text"]),
                    category="people",
                    include_domains=_archetype_include_domains(entry),
                    num_results=_archetype_num_results(entry),
                )
            )

    if not firms:
        if "people_search" in lanes:
            for term in title_terms:
                queries.append(
                    ExaSearchQuery(
                        id=_next_id(),
                        name=f"people_search · {term}",
                        lane="people_search",
                        firm="",
                        query=_collapse(f"{term} {geography}"),
                        category="people",
                        include_domains=["linkedin.com"],
                    )
                )
        _append_archetypes()
        return queries

    for lane in lanes:
        for firm in firms:
            firm_name = firm["name"].strip()
            exclude_terms = [
                t.strip()
                for t in (firm.get("exclude_terms") or [])
                if isinstance(t, str) and t.strip()
            ]
            domain = firm.get("domain")
            domain = domain.strip() if isinstance(domain, str) else ""
            firm_expression = _firm_expression(firm_name, firm)
            firm_titles = _firm_title_terms(firm, title_terms)

            if lane == "people_search":
                queries.append(
                    ExaSearchQuery(
                        id=_next_id(),
                        name=f"people_search · {firm_name}",
                        lane=lane,
                        firm=firm_name,
                        query=_collapse(
                            f"{' or '.join(firm_titles)} at {firm_expression} "
                            f"{geography}"
                        ),
                        category="people",
                        include_domains=["linkedin.com"],
                        exclude_terms=exclude_terms,
                    )
                )
            elif lane == "team_page":
                queries.append(
                    ExaSearchQuery(
                        id=_next_id(),
                        name=f"team_page · {firm_name}",
                        lane=lane,
                        firm=firm_name,
                        query=_collapse(f"{firm_name} team page investors"),
                        category=None,
                        include_domains=[domain] if domain else [],
                        exclude_terms=exclude_terms,
                    )
                )
            elif lane == "operating_signal":
                queries.append(
                    ExaSearchQuery(
                        id=_next_id(),
                        name=f"operating_signal · {firm_name}",
                        lane=lane,
                        firm=firm_name,
                        query=_collapse(
                            f"{firm_expression} {' or '.join(firm_titles)} "
                            "former founder OR operator OR chief of staff"
                        ),
                        category=None,
                        exclude_terms=exclude_terms,
                    )
                )

    _append_archetypes()

    return queries


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------


def lint_exa_brief(brief: Any) -> list[LaneCompilerFinding]:
    """Findings a recruiter should see before an Exa campaign launches."""

    findings: list[LaneCompilerFinding] = []
    firms = _target_firms(brief)
    lanes = _active_lanes(brief)
    title_terms = _title_terms(brief)

    # Employer conflict is an error, not a warning: a current employee of a
    # blacklisted firm or the hiring company is never a source lead, and
    # nobody who has ever worked at an alumni-blacklisted firm is either.
    hiring_company = _hiring_company(brief)
    protected = {
        _norm(name)
        for name in [hiring_company, *_employer_blacklist(brief), *_alumni_blacklist(brief)]
        if _norm(name)
    }
    for firm in firms:
        firm_name = firm["name"].strip()
        aliases = [
            a.strip()
            for a in (firm.get("aliases") or [])
            if isinstance(a, str) and a.strip()
        ]
        for label in [firm_name, *aliases]:
            label_norm = _norm(label)
            hit = next(
                (name for name in sorted(protected) if _token_overlap(name, label_norm)),
                None,
            )
            if hit is None:
                continue
            findings.append(
                LaneCompilerFinding(
                    code="employer_conflict",
                    severity="error",
                    message=(
                        f"Target firm {label!r} is the hiring company or on the "
                        f"employer blacklist or the alumni blacklist (matched "
                        f"{hit!r}) — this search never sources from it."
                    ),
                    source="exa",
                )
            )
            break

    # No title vocabulary. People-search and operating-signal queries
    # are built out of title terms; without them the people-search
    # query degrades to "at <firm> <geography>" and the operating
    # signal loses its seniority anchor.
    title_dependent = [
        lane for lane in lanes if lane in ("people_search", "operating_signal")
    ]
    if not title_terms and title_dependent:
        findings.append(
            LaneCompilerFinding(
                code="missing_title_terms",
                severity="error",
                message=(
                    "No title terms — the "
                    f"{', '.join(title_dependent)} lane(s) have nothing to "
                    "search for. Name the titles this search is looking for "
                    "(e.g. Senior Associate, Principal, Vice President)."
                ),
                source="exa",
            )
        )

    # No firms is not fatal: title-term queries still compile, but the search is
    # broader and noisier without the named-firm allowlist.
    if not firms:
        findings.append(
            LaneCompilerFinding(
                code="no_target_firms",
                severity="warning",
                message=(
                    "No target firms — the search runs title terms across the "
                    "open web instead of a named firm list, which is broader "
                    "and noisier."
                ),
                source="exa",
            )
        )

    # Nothing to run. The fallback only covers the people_search lane,
    # so no firms plus a lane subset that excludes it compiles zero
    # queries — a launch that burns setup and returns nobody, with only
    # a warning to show for it. Ask the compiler rather than
    # re-deriving the condition, so this stays true when the compiler's
    # rules change.
    if not compile_exa_queries(brief):
        findings.append(
            LaneCompilerFinding(
                code="no_queries_compiled",
                severity="error",
                message=(
                    "This brief compiles no Exa queries — the active lanes have "
                    "nothing to run against. Add target firms, or turn on the "
                    "people search lane, before launching."
                ),
                source="exa",
            )
        )

    # The current-employer scope block (``exa/scope_gate.py``). Malformed, it
    # would run the campaign without the check the brief asked for; an alias
    # in a query would pull a contractor pool in as retrieval. Both refuse.
    errors = scope_errors(
        brief, query_texts=[query.query for query in compile_exa_queries(brief)]
    )
    if errors:
        findings.append(
            LaneCompilerFinding(
                code="employer_scope_invalid",
                severity="error",
                message=f"{SCOPE_FIELD}: " + "; ".join(errors),
                source="exa",
            )
        )

    # The location scope block (``exa/location_gate.py``). Malformed, it
    # would either run without the check the brief asked for or exclude
    # everyone; both refuse.
    location_errors = location_scope_errors(brief)
    if location_errors:
        findings.append(
            LaneCompilerFinding(
                code="location_scope_invalid",
                severity="error",
                message=f"{LOCATION_FIELD}: " + "; ".join(location_errors),
                source="exa",
            )
        )

    # Namesake risk. A short single-token firm name collides with
    # unrelated firms and ordinary words; `exclude_terms` is the guard.
    for firm in firms:
        firm_name = firm["name"].strip()
        has_excludes = any(
            isinstance(t, str) and t.strip()
            for t in (firm.get("exclude_terms") or [])
        )
        if has_excludes:
            continue
        if len(firm_name.split()) != 1 or len(firm_name) > NAMESAKE_TOKEN_MAX_CHARS:
            continue
        findings.append(
            LaneCompilerFinding(
                code="namesake_risk",
                severity="info",
                message=(
                    f"{firm_name!r} is a short one-word firm name with no "
                    "exclusions — expect namesake results (Accel returns "
                    "Accel-KKR). Add exclude terms if that shows up."
                ),
                source="exa",
            )
        )

    # Archetype queries carry their own geography. The firm lanes get
    # the brief's geography appended by the compiler; an archetype
    # query does not, deliberately — its text is a sentence the author
    # wrote and Exa matches semantically, and splicing a location onto
    # the end of someone's prose is how you get "…now Senior
    # Associates on a deal team San Francisco Bay Area". So the author
    # writes the anchor in, and this is the check that they did. A
    # warning rather than an error: a deliberately global archetype is
    # a legitimate thing to write, and the campaign still runs.
    if "archetype" in lanes:
        geography_tokens = _geography_tokens(_geography(brief))
        if geography_tokens:
            for entry in _archetype_queries(brief):
                text_norm = _norm(entry["text"])
                if any(
                    _token_overlap(token, text_norm) for token in geography_tokens
                ):
                    continue
                findings.append(
                    LaneCompilerFinding(
                        code="archetype_missing_geography",
                        severity="warning",
                        message=(
                            f"Archetype query {entry['id'].strip()!r} never "
                            f"mentions {_geography(brief)!r} — the compiler "
                            "does not append it, so this query searches "
                            "everywhere. Write the location into the text if "
                            "that was not the intent."
                        ),
                        source="exa",
                    )
                )

    return findings


def _geography_tokens(geography: str) -> list[str]:
    """The words of a geography, minus the ones that carry no location.

    "SF Bay Area" is three tokens and any one of them appearing in an
    archetype text is evidence the author anchored it. The stop list is
    the connective tissue — "the greater San Francisco region" must not
    pass on "the".
    """

    stop = {"the", "of", "and", "greater", "metro", "metropolitan", "region", "area"}
    return [
        token
        for token in re.findall(r"[a-z0-9]+", _norm(geography))
        if token not in stop
    ]


def blocking_findings(
    findings: list[LaneCompilerFinding],
) -> list[LaneCompilerFinding]:
    """The findings that must stop a launch — errors only."""
    return [f for f in findings if f.severity == "error"]


__all__ = [
    "NAMESAKE_TOKEN_MAX_CHARS",
    "active_lanes",
    "blocking_findings",
    "compile_exa_queries",
    "lint_exa_brief",
]
