"""Designer module — query formation from brief content.

Designer Slice 2. The strategy layer consumes a V2 brief and produces a
list of :class:`designer.schemas.DesignerSearchQuery` objects that
:mod:`designer.acquisition` executes against the per-source clients.

Slice 2 covers Behance only. Slice 3 extends with Google CSE queries
(filtered to portfolio-host domains). Both sources read from the same
V2 brief shape — Cloris doesn't fork the brief contract per source.

Brief content consumed:

- ``capability_areas[*].name`` + ``description`` — the recruiter's
  capability framing. Each capability area becomes one or more queries.
- ``capability_areas[*].behance_specialization_signals`` (optional) —
  list of strings the recruiter pasted that map to Behance creative-
  field tags or specialization vocab. If present, these become
  high-precision query terms.
- ``capability_areas[*].tool_stack_signals`` (optional) — Figma,
  After Effects, Cinema 4D, etc. Combined with capability area name
  to form tool-anchored queries.
- ``design_rubric.calibration_exemplars[*].discipline`` — used to bias
  the query distribution toward the discipline mix the recruiter
  exemplified (e.g., 4 product + 1 brand → product-weighted queries).
- ``geography`` (optional, top-level) — passed through as Behance's
  ``country`` filter when a 2-letter ISO code is detectable.

This module is deliberately small and deterministic. The Slice-2 spec
note on "Opus-driven query generation" lives in :mod:`designer.judgment_templates`
(prompts) plus a future LLM-assisted query expander (Slice 5+ when the
brief polish loop has more signal). For Slice 2, the query set is
recipe-style: walk capability areas, emit per-area queries, dedup.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from designer.schemas import DesignerSearchQuery
from shared.schemas import ExecutionPlan


# Behance's `sort` parameter values, in priority order. The first
# value is what every Slice-2 query uses by default; the others are
# available to callers that want to broaden the discovery surface
# (e.g., a "newer voices" pass would use `published_date`).
BEHANCE_SORT_PRIMARY = "appreciations"
BEHANCE_SORT_VALUES = ("appreciations", "views", "published_date")

# Cap on how many queries a single capability area generates. Without
# this, a brief with 5 specialization signals × 3 tool signals would
# produce 15 queries per capability area; at 4 capability areas that's
# 60 queries, exhausting Behance's per-hour budget on a single brief.
MAX_QUERIES_PER_CAPABILITY_AREA = 6


def select_designer_sources(
    *,
    behance_api_key: str | None = None,
    google_cse_api_key: str | None = None,
    google_cse_id: str | None = None,
) -> tuple[str, ...]:
    """Return the source set the strategy should emit queries for.

    Audit Move #14 — CSE-primary contract. Reads the same env vars
    the health probe checks; returns the source tuple in priority
    order (CSE first when configured, Behance augment when also
    configured). Returns empty tuple when neither is configured —
    callers that hit this should match the health probe's hard
    blocker behavior (refuse to launch).

    Override kwargs match :func:`designer.health.probe_designer_readiness`'s
    surface so tests + the launch-readiness aggregator can resolve
    the source set without two parallel env-reads.
    """

    import os as _os

    effective_cse_key = (
        google_cse_api_key
        if google_cse_api_key is not None
        else _os.environ.get("GOOGLE_CSE_API_KEY", "")
    )
    effective_cse_id = (
        google_cse_id
        if google_cse_id is not None
        else _os.environ.get("GOOGLE_CSE_ID", "")
    )
    effective_behance = (
        behance_api_key
        if behance_api_key is not None
        else _os.environ.get("BEHANCE_API_KEY", "")
    )

    selected: list[str] = []
    if effective_cse_key and effective_cse_id:
        selected.append("google_cse")
    if effective_behance:
        selected.append("behance")
    return tuple(selected)


def form_designer_strategy(
    brief: dict[str, Any],
    *,
    sources: Iterable[str] | None = None,
) -> list[DesignerSearchQuery]:
    """Build the list of discovery queries for a Designer brief.

    Returns a deduped list of :class:`DesignerSearchQuery`. Queries are
    deterministic given identical input — important for tests and for
    runtime-state resume semantics (a re-run of the same brief produces
    the same work-units and so the same canonical work-unit IDs).

    ``sources`` is the set of source names to emit queries for. Audit
    Move #14 default: when ``sources`` is None, resolves via
    :func:`select_designer_sources` (CSE-primary, Behance-augment
    when configured). Pre-Move-14 callers passing ``("behance",)``
    explicitly continue to work unchanged.
    """

    if sources is None:
        sources = select_designer_sources()

    queries: list[DesignerSearchQuery] = []
    seen: set[tuple[str, str]] = set()  # (source, query_text) for dedup

    capability_areas = brief.get("capability_areas") or []
    if not isinstance(capability_areas, list):
        return queries

    discipline = _dominant_discipline(brief)
    geography_code = _country_code(brief)

    for capability_area in capability_areas:
        if not isinstance(capability_area, dict):
            continue
        name = str(capability_area.get("name") or "").strip()
        if not name:
            continue

        # Audit Move #14 — CSE-primary ordering: emit CSE queries
        # BEFORE Behance queries within each capability area so the
        # orchestrator's runtime-state work_units inherit a CSE-first
        # ordering_index. CSE is the supported-for-everyone surface;
        # Behance augments only when its (rare) v2 key is configured.
        if "google_cse" in sources:
            for query in _google_cse_queries_for_capability_area(
                capability_area=capability_area,
                discipline=discipline,
            ):
                key = (query.source, query.query_text.lower())
                if key in seen:
                    continue
                seen.add(key)
                queries.append(query)

        if "behance" in sources:
            for query in _behance_queries_for_capability_area(
                capability_area=capability_area,
                discipline=discipline,
                geography_code=geography_code,
            ):
                key = (query.source, query.query_text.lower())
                if key in seen:
                    continue
                seen.add(key)
                queries.append(query)

    return queries


# ---------------------------------------------------------------------------
# Registry adapter (Multi-Agent Execution Plan Slice 1.6)
# ---------------------------------------------------------------------------


def form_strategy_for_registry(
    brief: Any,
    prior_run_data: dict | None = None,
) -> ExecutionPlan:
    """Uniform-signature adapter wrapping :func:`form_designer_strategy`.

    Correction 3a in ``plans/multi-agent-execution-plan.md``: Designer's
    strategy stage is mechanical query composition, deliberately
    deterministic — Opus is reserved for the vision-evaluation pipeline
    (``designer/vision_evaluation.py:150``), not strategy formation.
    The adapter normalizes the deterministic
    ``list[DesignerSearchQuery]`` output into the uniform
    :class:`ExecutionPlan` shape the launcher registry's
    ``form_strategy_fn`` field expects, without forcing an Opus call.

    ``prior_run_data`` is accepted to match the registry signature but
    not consumed: Designer's strategy is a pure function of brief
    content, and resume-time work-unit identity stability comes from
    deterministic query composition (see :func:`form_designer_strategy`'s
    docstring).

    ``brief`` is polymorphic. The Designer orchestrator hands the V2
    brief in as a raw ``dict``; cross-module callers may hand in a
    compat :class:`shared.brief_loader.Brief` (with the V2 dict on
    ``_new_brief``) or the V2 :class:`shared.brief_schema.Brief`
    (which exposes ``raw_dict()``). The adapter unwraps either form.
    """

    raw = _coerce_brief_to_dict(brief)
    queries = form_designer_strategy(raw)
    return ExecutionPlan(
        strategy_rationale="",
        generated_strings=[asdict(q) for q in queries],
    )


def _coerce_brief_to_dict(brief: Any) -> dict[str, Any]:
    """Coerce a polymorphic brief input into the dict shape
    :func:`form_designer_strategy` consumes."""

    if isinstance(brief, dict):
        return brief
    raw = getattr(brief, "_new_brief", None)
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "raw_dict"):
        candidate = raw.raw_dict()
        if isinstance(candidate, dict):
            return candidate
    if hasattr(brief, "raw_dict"):
        candidate = brief.raw_dict()
        if isinstance(candidate, dict):
            return candidate
    return {}


def _behance_queries_for_capability_area(
    *,
    capability_area: dict[str, Any],
    discipline: str,
    geography_code: str,
) -> list[DesignerSearchQuery]:
    """Form Behance queries for one capability area.

    Recipe:
    1. Always emit the bare capability-area name as a query.
    2. Emit each ``behance_specialization_signals`` value as its own
       query (high precision; recruiter-authored vocab).
    3. Cross-product the top 2 specialization signals with the top 2
       tool signals as combined queries.
    4. Cap at ``MAX_QUERIES_PER_CAPABILITY_AREA``.
    """

    name = str(capability_area.get("name") or "").strip()
    extra_filters: dict[str, Any] = {}
    if geography_code:
        extra_filters["country"] = geography_code

    queries: list[DesignerSearchQuery] = [
        DesignerSearchQuery(
            source="behance",
            query_text=name,
            sort=BEHANCE_SORT_PRIMARY,
            capability_area_name=name,
            discipline=discipline,
            extra_filters=dict(extra_filters),
        )
    ]

    spec_signals = _as_str_list(capability_area.get("behance_specialization_signals"))
    tool_signals = _as_str_list(capability_area.get("tool_stack_signals"))

    for signal in spec_signals:
        queries.append(
            DesignerSearchQuery(
                source="behance",
                query_text=signal,
                sort=BEHANCE_SORT_PRIMARY,
                capability_area_name=name,
                discipline=discipline,
                extra_filters=dict(extra_filters),
            )
        )

    for signal in spec_signals[:2]:
        for tool in tool_signals[:2]:
            queries.append(
                DesignerSearchQuery(
                    source="behance",
                    query_text=f"{signal} {tool}",
                    sort=BEHANCE_SORT_PRIMARY,
                    capability_area_name=name,
                    discipline=discipline,
                    extra_filters=dict(extra_filters),
                )
            )

    return queries[:MAX_QUERIES_PER_CAPABILITY_AREA]


# Maximum CSE queries per capability area. CSE is more expensive
# (paid above 100/day) AND each query also fans out across the
# portfolio-host set in :mod:`designer.acquisition`, so the per-
# capability cap is tighter than Behance's.
MAX_CSE_QUERIES_PER_CAPABILITY_AREA = 3


def _google_cse_queries_for_capability_area(
    *,
    capability_area: dict[str, Any],
    discipline: str,
) -> list[DesignerSearchQuery]:
    """Form CSE queries for one capability area.

    CSE queries DON'T site-restrict at strategy-formation time —
    :mod:`designer.acquisition` fans each query out across the
    portfolio-host set so the per-host quota burn is explicit at the
    acquisition layer rather than baked into the work-unit shape.

    Recipe:
    1. Capability-area name as bare query.
    2. Top 1 specialization signal as query.
    3. Top 1 specialization × top 1 tool as combined query.
    Capped at ``MAX_CSE_QUERIES_PER_CAPABILITY_AREA``.
    """

    name = str(capability_area.get("name") or "").strip()
    queries: list[DesignerSearchQuery] = [
        DesignerSearchQuery(
            source="google_cse",
            query_text=name,
            sort="relevance",
            capability_area_name=name,
            discipline=discipline,
        )
    ]
    spec_signals = _as_str_list(capability_area.get("behance_specialization_signals"))
    tool_signals = _as_str_list(capability_area.get("tool_stack_signals"))

    if spec_signals:
        queries.append(
            DesignerSearchQuery(
                source="google_cse",
                query_text=spec_signals[0],
                sort="relevance",
                capability_area_name=name,
                discipline=discipline,
            )
        )

    if spec_signals and tool_signals:
        queries.append(
            DesignerSearchQuery(
                source="google_cse",
                query_text=f"{spec_signals[0]} {tool_signals[0]} portfolio",
                sort="relevance",
                capability_area_name=name,
                discipline=discipline,
            )
        )

    return queries[:MAX_CSE_QUERIES_PER_CAPABILITY_AREA]


def _dominant_discipline(brief: dict[str, Any]) -> str:
    """Return the most-frequently-tagged discipline across calibration
    exemplars; empty string when the brief carries none."""

    rubric = brief.get("design_rubric")
    if not isinstance(rubric, dict):
        return ""
    exemplars = rubric.get("calibration_exemplars") or []
    if not isinstance(exemplars, list):
        return ""
    counts: dict[str, int] = {}
    for ex in exemplars:
        if not isinstance(ex, dict):
            continue
        discipline = ex.get("discipline")
        if isinstance(discipline, str) and discipline:
            counts[discipline] = counts.get(discipline, 0) + 1
    if not counts:
        return ""
    # Stable: first key with max count when ties.
    return max(counts, key=lambda k: (counts[k], -list(counts.keys()).index(k)))


def _country_code(brief: dict[str, Any]) -> str:
    """Extract a 2-letter ISO country code from the brief's geography
    field if it parses cleanly. Behance's `country` filter expects ISO."""

    geography = brief.get("geography")
    if not isinstance(geography, str):
        return ""
    geography = geography.strip().upper()
    # Conservative: only accept exact 2-letter strings as ISO codes.
    if len(geography) == 2 and geography.isalpha():
        return geography
    # Common natural-language geographies → ISO codes. Slice-2 handles
    # only the ones Cloris's existing customer base sees; broader
    # coverage is a follow-up.
    natural_language_map = {
        "USA": "US",
        "UNITED STATES": "US",
        "UNITED KINGDOM": "GB",
        "UK": "GB",
        "GERMANY": "DE",
        "BRAZIL": "BR",
        "COLOMBIA": "CO",
    }
    return natural_language_map.get(geography, "")


def _as_str_list(value: Any) -> list[str]:
    """Coerce a brief field to a list of non-empty strings."""

    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]
