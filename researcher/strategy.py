"""Researcher strategy formation — Slice 3.

`form_strategy(brief, prior_data)` produces an :class:`ExecutionPlan`
whose ``generated_strings`` carry researcher query dicts (NOT human-
readable Boolean strings — see Researcher Module Spec Opinion 2).

Each query is one work_unit with ``kind="researcher_author_query"``;
the acquisition layer (Slice 4) hydrates it into OpenAlex API calls.

Brief blocks consumed:

- ``capability_areas`` → mapped to OpenAlex Concept IDs (the LLM picks
  the right concepts given the capability prose)
- ``source_config.researcher.research_topics`` → additive prompt context
  (free-text topics the recruiter named; the LLM blends them with the
  concept mapping)
- ``source_config.researcher.conference_allowlist`` → seed
  ``venue_filter`` (the LLM may emit per-venue queries)
- ``source_config.researcher.h_index_floor`` +
  ``papers_in_window_floor`` → passed through to the deterministic
  gates at evaluation time (Slice 5); strategy doesn't filter on them
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from shared.brief_schema import Brief
from shared.brief_v2_schema import source_config_for
from shared.adaptive import (
    AdaptiveAction,
    AdaptationDecision,
    NoiseMarker,
    ScoutMetrics,
    SignalMarker,
)
from shared.schemas import ExecutionPlan


RESEARCHER_QUERY_SCHEMA_KEYS = (
    "topic_concepts",
    "venue_filter",
    "min_year",
    "min_citations",
    "ror_country_filter",
)


@dataclass(frozen=True)
class ResearcherQueryReport:
    query_id: int
    name: str
    topic_concepts: list[str] = field(default_factory=list)
    venue_filter: list[str] = field(default_factory=list)
    min_year: int | None = None
    min_citations: int | None = None
    ror_country_filter: list[str] = field(default_factory=list)
    candidates_discovered: int = 0
    facial_yes_count: int = 0
    facial_no_count: int = 0
    facial_borderline_count: int = 0
    saves_count: int = 0
    rejected_count: int = 0

    @classmethod
    def from_query_stats(cls, query: dict[str, Any], stats: dict[str, Any]) -> "ResearcherQueryReport":
        return cls(
            query_id=int(query.get("id") or 0),
            name=str(query.get("name") or ""),
            topic_concepts=[str(item) for item in query.get("topic_concepts") or []],
            venue_filter=[str(item) for item in query.get("venue_filter") or []],
            min_year=int(query["min_year"]) if query.get("min_year") is not None else None,
            min_citations=(
                int(query["min_citations"])
                if query.get("min_citations") is not None
                else None
            ),
            ror_country_filter=[str(item) for item in query.get("ror_country_filter") or []],
            candidates_discovered=int(stats.get("candidates_discovered") or 0),
            facial_yes_count=int(stats.get("facial_yes_count") or 0),
            facial_no_count=int(stats.get("facial_no_count") or 0),
            facial_borderline_count=int(stats.get("facial_borderline_count") or 0),
            saves_count=int(stats.get("saves_count") or 0),
            rejected_count=int(stats.get("rejected_count") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearcherAdaptiveReport:
    batch_name: str
    query_reports: list[ResearcherQueryReport] = field(default_factory=list)
    source_mix: dict[str, int] = field(default_factory=dict)

    @property
    def queries_run(self) -> int:
        return len(self.query_reports)

    @property
    def total_candidates_discovered(self) -> int:
        return sum(report.candidates_discovered for report in self.query_reports)

    @property
    def total_saves(self) -> int:
        return sum(report.saves_count for report in self.query_reports)

    @property
    def total_rejects(self) -> int:
        return sum(report.rejected_count for report in self.query_reports)

    @property
    def total_facial_yes(self) -> int:
        return sum(report.facial_yes_count for report in self.query_reports)

    @property
    def total_facial_no(self) -> int:
        return sum(report.facial_no_count for report in self.query_reports)

    @property
    def sparse_query_ids(self) -> list[int]:
        return [
            report.query_id
            for report in self.query_reports
            if report.candidates_discovered == 0
        ]

    @property
    def noisy_query_ids(self) -> list[int]:
        return [
            report.query_id
            for report in self.query_reports
            if report.candidates_discovered > 0
            and report.facial_yes_count == 0
            and report.saves_count == 0
        ]

    @property
    def productive_reports(self) -> list[ResearcherQueryReport]:
        return [report for report in self.query_reports if report.saves_count > 0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "query_reports": [report.to_dict() for report in self.query_reports],
            "source_mix": dict(self.source_mix),
            "total_candidates_discovered": self.total_candidates_discovered,
            "total_saves": self.total_saves,
            "total_rejects": self.total_rejects,
            "sparse_query_ids": self.sparse_query_ids,
            "noisy_query_ids": self.noisy_query_ids,
        }

    def to_summary_text(self) -> str:
        lines = [
            f'{self.batch_name}: {self.queries_run} queries, '
            f"{self.total_candidates_discovered} candidates, "
            f"{self.total_saves} saves, {self.total_rejects} rejects."
        ]
        if self.sparse_query_ids:
            lines.append(
                "Sparse queries: "
                + ", ".join(f"#{query_id}" for query_id in self.sparse_query_ids)
            )
        if self.noisy_query_ids:
            lines.append(
                "Noisy queries: "
                + ", ".join(f"#{query_id}" for query_id in self.noisy_query_ids)
            )
        for report in self.query_reports:
            lines.append(
                f"  #{report.query_id} {report.name}: "
                f"topics={report.topic_concepts or '-'} venues={report.venue_filter or '-'} "
                f"candidates={report.candidates_discovered} "
                f"facial_yes={report.facial_yes_count} saves={report.saves_count}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ResearcherAdaptationPlan:
    new_queries: list[dict[str, Any]] = field(default_factory=list)
    skipped_query_ids: list[int] = field(default_factory=list)
    reordered_query_ids: list[int] = field(default_factory=list)
    rationale: str = ""
    decision: AdaptationDecision | None = None


def form_strategy(
    brief: Brief,
    prior_data: dict | None = None,
    *,
    llm_caller: Callable[[str, str], dict] | None = None,
) -> ExecutionPlan:
    """Compose an :class:`ExecutionPlan` for the researcher pipeline.

    ``llm_caller`` is injectable so tests don't require a real Opus call;
    when ``None``, defaults to ``shared.llm_clients.opus_llm`` with
    JSON-expecting kwargs. Production callers leave it ``None``.

    The plan's ``generated_strings`` field carries researcher query
    dicts conforming to :data:`RESEARCHER_QUERY_SCHEMA_KEYS`. The
    orchestrator (Slice 6) wraps each into a SearchString-shaped
    work_unit; the boolean field stays empty per Spec Opinion 2.
    """

    import logging

    logger = logging.getLogger(__name__)

    raw_brief = _brief_raw(brief)
    source_config = source_config_for(raw_brief, "researcher")

    system = _build_system_prompt(brief, source_config)
    user_prompt = _build_user_prompt(brief, source_config, prior_data)

    if llm_caller is None:
        # A.2 cache-gap remediation: switched to opus_llm_cached so the
        # researcher strategy system prompt (>2KB of brief calibration
        # context) is sent with cache_control: {"type": "ephemeral"}
        # for ~90% input-token cost on subsequent calls in the same
        # 5-minute window. Same signature as opus_llm.
        from shared.llm_clients import opus_llm_cached

        def _default_caller(s: str, u: str) -> dict:
            usage_context = {
                "stage": "researcher_strategy",
                "source": "researcher",
                "brief_id": getattr(brief, "id", None),
                "role_title": getattr(brief, "role_title", ""),
            }
            return opus_llm_cached(
                s,
                u,
                expect_json=True,
                max_tokens=16384,
                usage_context=usage_context,
            )

        llm_caller = _default_caller

    # Audit Move #4 R7: schema-validated LLM output with heuristic
    # fallback. Two failure modes converge on the heuristic plan:
    # - LLM call raises (network, rate-limit, parse error from the
    #   shared client's JSON enforcement).
    # - LLM returns a structurally-invalid plan (no generated_strings,
    #   or every query missing topic_concepts AND venue_filter — the
    #   acquisition layer needs at least one filter to make progress).
    # Pre-R7 callers got a cryptic KeyError downstream; the cascade
    # produces a deterministic plan from source_config so the
    # acquisition path always has something to run.
    try:
        result = llm_caller(system, user_prompt)
    except Exception as exc:  # noqa: BLE001 — fall through on any LLM error
        logger.warning(
            "researcher.strategy: LLM call raised %s; falling back to heuristic plan",
            exc.__class__.__name__,
        )
        return _attach_role_strategy(brief, _heuristic_plan(brief, source_config))

    validation_failure = _validate_plan_schema(result)
    if validation_failure is not None:
        logger.warning(
            "researcher.strategy: LLM plan failed schema validation (%s); "
            "falling back to heuristic plan",
            validation_failure,
        )
        return _attach_role_strategy(brief, _heuristic_plan(brief, source_config))

    plan = _parse_plan(result, source_config=source_config)
    return _attach_role_strategy(brief, plan)


def adapt_after_research_batch(
    brief: Brief,
    report: ResearcherAdaptiveReport,
    remaining_queries: list[dict[str, Any]],
    *,
    llm_caller: Callable[[str, str], dict] | None = None,
) -> ResearcherAdaptationPlan:
    """Adapt OpenAlex/Semantic/arXiv search strategy after a scout batch."""

    if not remaining_queries:
        decision = _researcher_decision_from_plan(
            report=report,
            new_queries=[],
            skipped_query_ids=[],
            reordered_query_ids=[],
            rationale="No remaining researcher queries to adapt.",
        )
        return ResearcherAdaptationPlan(rationale=decision.rationale, decision=decision)

    source_config = source_config_for(_brief_raw(brief), "researcher")
    if llm_caller is None:
        from shared.llm_clients import opus_llm_cached

        def _default_caller(s: str, u: str) -> dict:
            usage_context = {
                "stage": "researcher_batch_adaptation",
                "source": "researcher",
                "brief_id": getattr(brief, "id", None),
                "role_title": getattr(brief, "role_title", ""),
                "remaining_query_count": len(remaining_queries),
            }
            return opus_llm_cached(
                s,
                u,
                expect_json=True,
                max_tokens=8192,
                usage_context=usage_context,
            )

        llm_caller = _default_caller

    system = _build_adaptation_system_prompt(brief, source_config)
    user_prompt = _build_adaptation_user_prompt(report, remaining_queries)
    try:
        raw = llm_caller(system, user_prompt)
    except Exception:
        raw = _heuristic_adaptation_response(report, remaining_queries)

    plan = _parse_adaptation_response(raw, report, remaining_queries)
    decision = _researcher_decision_from_plan(
        report=report,
        new_queries=plan.new_queries,
        skipped_query_ids=plan.skipped_query_ids,
        reordered_query_ids=plan.reordered_query_ids,
        rationale=plan.rationale,
    )
    return ResearcherAdaptationPlan(
        new_queries=plan.new_queries,
        skipped_query_ids=plan.skipped_query_ids,
        reordered_query_ids=plan.reordered_query_ids,
        rationale=plan.rationale,
        decision=decision,
    )


# ---------------------------------------------------------------------------
# Registry adapter (Multi-Agent Execution Plan Slice 1.6)
# ---------------------------------------------------------------------------


def _attach_role_strategy(brief: Brief, plan: ExecutionPlan) -> ExecutionPlan:
    from shared.role_strategy import apply_role_strategy_to_plan

    apply_role_strategy_to_plan(brief, plan, merge_lane_templates=False)
    return plan


def form_strategy_for_registry(
    brief: Brief,
    prior_run_data: dict | None = None,
) -> ExecutionPlan:
    """Uniform-signature adapter wrapping :func:`form_strategy`.

    The launcher registry's ``form_strategy_fn`` field expects
    ``form_strategy_for_registry(brief, prior_run_data) -> ExecutionPlan``.
    Researcher's native :func:`form_strategy` already returns an
    :class:`ExecutionPlan`; the only divergence is the keyword-argument
    name (``prior_data`` vs ``prior_run_data``) and the exposed
    ``llm_caller`` injection seam, which stays internal.

    Native callers (``researcher.session_orchestrator``) continue to
    call :func:`form_strategy` directly with their own ``prior_data``
    payload and remain unaffected.
    """

    return form_strategy(brief, prior_data=prior_run_data)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_system_prompt(brief: Brief, source_config: dict[str, Any]) -> str:
    """Assemble the cacheable system prompt — all brief context, no LLM
    output coupling.
    """

    capability_block = _capability_area_block(brief)
    venue_seed = ", ".join(_as_str_list(source_config.get("conference_allowlist")))
    research_topics = ", ".join(_as_str_list(source_config.get("research_topics")))
    discipline = _as_str(source_config.get("discipline"))

    return _SYSTEM_TEMPLATE.format(
        role_title=brief.role_title or "(unspecified)",
        role_summary=getattr(brief, "role_summary", "") or "(none)",
        capability_block=capability_block,
        research_topics=research_topics or "(none specified)",
        venue_seed=venue_seed or "(none specified — pick canonical venues)",
        discipline=discipline or "ml_general",
    )


def _build_user_prompt(
    brief: Brief,
    source_config: dict[str, Any],
    prior_data: dict | None,
) -> str:
    """Assemble the per-call user prompt — instruction + output contract."""

    prior_summary = _summarize_prior_data(prior_data)
    return _USER_TEMPLATE.format(
        prior_summary=prior_summary,
        query_schema_keys=", ".join(RESEARCHER_QUERY_SCHEMA_KEYS),
    )


def _capability_area_block(brief: Brief) -> str:
    areas = getattr(brief, "capability_areas", None) or []
    if not areas:
        return "(no capability areas declared — fall back to brief.role_summary)"
    lines: list[str] = []
    for idx, area in enumerate(areas, start=1):
        name = getattr(area, "name", None) or ""
        description = getattr(area, "description", None) or ""
        lines.append(f"  {idx}. {name}")
        if description:
            lines.append(f"     — {description}")
    return "\n".join(lines)


def _summarize_prior_data(prior_data: dict | None) -> str:
    if not prior_data:
        return "(no prior run data — this is a fresh strategy)"
    lines = ["Prior run hints:"]
    if "queries_explored" in prior_data:
        lines.append(f"  - queries explored: {prior_data['queries_explored']}")
    if "underperforming_concepts" in prior_data:
        lines.append(
            f"  - concepts underperforming: {prior_data['underperforming_concepts']}"
        )
    if "high_yield_venues" in prior_data:
        lines.append(f"  - high-yield venues: {prior_data['high_yield_venues']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plan parsing / normalization
# ---------------------------------------------------------------------------


def _validate_plan_schema(result: Any) -> str | None:
    """Validate the raw LLM plan dict before normalization.

    Audit Move #4 R7. Returns a short diagnostic string when the plan
    is structurally invalid (so :func:`form_strategy` can fall back to
    the heuristic plan), or ``None`` when valid.

    "Valid" is intentionally permissive: missing optional fields
    survive normalization in :func:`_parse_plan`. The validator only
    fires on plans that would yield ZERO usable queries downstream:

    - Result is not a dict (LLM returned a list, string, etc).
    - ``generated_strings`` is absent or empty (no plan to execute).
    - Every query lacks BOTH ``topic_concepts`` AND ``venue_filter``
      AND ``ror_country_filter`` — the acquisition pipeline needs at
      least one filter to make progress against OpenAlex.
    """

    if not isinstance(result, dict):
        return f"result_not_dict type={type(result).__name__}"
    raw_queries = result.get("generated_strings")
    if not isinstance(raw_queries, list) or not raw_queries:
        return "generated_strings_empty_or_missing"

    parseable_count = 0
    for raw in raw_queries:
        if not isinstance(raw, dict):
            continue
        has_filter = (
            _has_non_empty_list(raw.get("topic_concepts"))
            or _has_non_empty_list(raw.get("venue_filter"))
            or _has_non_empty_list(raw.get("ror_country_filter"))
        )
        if has_filter:
            parseable_count += 1
    if parseable_count == 0:
        return "every_query_missing_filters"
    return None


def _has_non_empty_list(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, str) and item.strip():
            return True
    return False


def _heuristic_plan(
    brief: Brief,
    source_config: dict[str, Any],
) -> ExecutionPlan:
    """Build a deterministic researcher plan from the brief's source_config.

    Audit Move #4 R7. Used when the LLM call fails or its plan output
    fails schema validation. Produces a single concept-anchored query
    from research_topics + conference_allowlist; the acquisition path
    always has something to run, even when the LLM is unreachable.

    The heuristic is intentionally minimal — it doesn't try to
    replicate the LLM's branching strategy (RLHF axes / safety vs.
    interpretability, etc.). The recruiter sees a single "default
    plan" run; subsequent runs (when the LLM is back up) get the
    full strategy.
    """

    research_topics = _as_str_list(source_config.get("research_topics"))
    conference_allowlist = _as_str_list(
        source_config.get("conference_allowlist")
    )
    # The LLM normally maps research_topics (free-text) to OpenAlex
    # concept ids; the heuristic doesn't have that lookup. We treat
    # the topics as concept identifiers so OpenAlex's filter passes
    # something through; when topics are absent (or all empty), we
    # fall back to a venue-filter-only plan that still produces
    # results from any of the allowed venues.
    topic_concepts = research_topics
    venue_filter = conference_allowlist
    ror_country_filter: list[str] = []

    name_parts: list[str] = []
    if topic_concepts:
        name_parts.append("+".join(topic_concepts[:2]))
    if venue_filter:
        name_parts.append(f"@{venue_filter[0]}")
    name = " ".join(name_parts) or "researcher heuristic plan"

    query: dict[str, Any] = {
        "id": 1,
        "name": name,
        "boolean": "",
        "string_type": "Recall",
        "topic_concepts": topic_concepts,
        "venue_filter": venue_filter,
        "min_year": 0,
        "min_citations": 0,
        "ror_country_filter": ror_country_filter,
    }

    plan_dict: dict[str, Any] = {
        "strategy_rationale": (
            "Heuristic plan: LLM-driven strategy unavailable; running a "
            "single-query default derived from the brief's research "
            "topics and conference allowlist."
        ),
        "generated_strings": [query],
        "architecture": "concept_first",
    }
    return _parse_plan(plan_dict, source_config=source_config)


def _parse_plan(
    result: dict,
    *,
    source_config: dict[str, Any],
) -> ExecutionPlan:
    """Normalize the LLM output into a clean :class:`ExecutionPlan`.

    Defensive against missing or malformed keys: missing
    ``generated_strings`` ⇒ empty list (caller surfaces an
    "empty_search_results" stop reason); each query dict gets its
    keys defaulted so the acquisition layer can rely on the shape.
    """

    plan_dict = dict(result or {})
    raw_queries = plan_dict.get("generated_strings") or []
    if not isinstance(raw_queries, list):
        raw_queries = []

    normalized: list[dict] = []
    for idx, raw in enumerate(raw_queries):
        if not isinstance(raw, dict):
            continue
        normalized.append(_normalize_query(raw, idx=idx, source_config=source_config))

    plan_dict["generated_strings"] = normalized
    return ExecutionPlan.from_dict(plan_dict)


def _normalize_query(
    raw: dict,
    *,
    idx: int,
    source_config: dict[str, Any],
) -> dict:
    """Coerce one query dict into the canonical schema."""

    topic_concepts = _as_str_list(raw.get("topic_concepts"))
    venue_filter = _as_str_list(raw.get("venue_filter")) or _as_str_list(
        source_config.get("conference_allowlist")
    )
    ror_country_filter = _as_str_list(raw.get("ror_country_filter"))

    min_year_raw = raw.get("min_year")
    min_year = int(min_year_raw) if isinstance(min_year_raw, (int, float)) else 0

    min_citations_raw = raw.get("min_citations")
    min_citations = (
        int(min_citations_raw) if isinstance(min_citations_raw, (int, float)) else 0
    )

    name = _as_str(raw.get("name")) or _default_query_name(
        idx=idx,
        topic_concepts=topic_concepts,
        venue_filter=venue_filter,
    )

    return {
        # SearchString-compatible identity fields so the orchestrator
        # can wrap this into a work_unit uniformly.
        "id": int(raw.get("id") or idx + 1),
        "name": name,
        "boolean": "",  # Spec Opinion 2: no Boolean string layer for researcher.
        "string_type": "Recall",
        # The actual researcher query parameters.
        "topic_concepts": topic_concepts,
        "venue_filter": venue_filter,
        "min_year": min_year,
        "min_citations": min_citations,
        "ror_country_filter": ror_country_filter,
    }


def _build_adaptation_system_prompt(brief: Brief, source_config: dict[str, Any]) -> str:
    venue_seed = ", ".join(_as_str_list(source_config.get("conference_allowlist")))
    research_topics = ", ".join(_as_str_list(source_config.get("research_topics")))
    return f"""You are adapting Cloris's Researcher sourcing plan mid-run.

Role: {getattr(brief, 'role_title', '')}
Summary: {getattr(brief, 'role_summary', '')}

Research topics: {research_topics or '(none specified)'}
Venue seed: {venue_seed or '(none specified)'}

Use native academic-search controls only:
- topic_concepts
- venue_filter
- min_year
- min_citations
- ror_country_filter
- source mix hints for OpenAlex / Semantic Scholar / arXiv

Adapt based on market feedback:
- sparse results: broaden concepts, relax venue/citation/recency filters, or widen source mix
- noisy results: narrow concepts, add venue/affiliation filters, or skip redundant queued work
- strong emerging signal: add adjacent concept/venue lanes and run them soon

Return JSON only:
{{
  "new_researcher_queries": [
    {{
      "name": "...",
      "topic_concepts": ["..."],
      "venue_filter": ["..."],
      "min_year": 2022,
      "min_citations": 5,
      "ror_country_filter": ["US"],
      "source_mix": {{"openalex": true, "semantic_scholar": true, "arxiv": true}}
    }}
  ],
  "skip_query_ids": [1, 2],
  "reorder_query_ids": [5, 4],
  "rationale": "..."
}}"""


def _build_adaptation_user_prompt(
    report: ResearcherAdaptiveReport,
    remaining_queries: list[dict[str, Any]],
) -> str:
    remaining = "\n".join(
        f"  #{query.get('id')}: {query.get('name')} "
        f"topics={query.get('topic_concepts') or '-'} "
        f"venues={query.get('venue_filter') or '-'}"
        for query in remaining_queries
    )
    return f"""{report.to_summary_text()}

Remaining queued researcher queries:
{remaining}

Suggest the next adaptive step."""


def _heuristic_adaptation_response(
    report: ResearcherAdaptiveReport,
    remaining_queries: list[dict[str, Any]],
) -> dict[str, Any]:
    skip_ids = list(report.noisy_query_ids)
    new_queries: list[dict[str, Any]] = []
    rationale_parts: list[str] = []

    if report.sparse_query_ids and remaining_queries:
        seed = dict(remaining_queries[0])
        seed["name"] = f"Broadened: {seed.get('name') or 'researcher query'}"
        seed["venue_filter"] = []
        seed["min_citations"] = 0
        if int(seed.get("min_year") or 0) > 0:
            seed["min_year"] = max(int(seed.get("min_year") or 0) - 2, 0)
        seed["adapted_reason"] = "sparse_results_broadened_filters"
        new_queries.append(seed)
        rationale_parts.append("Sparse scout results; broadened venue/citation/recency filters.")

    productive = report.productive_reports
    if productive:
        top = productive[0]
        adjacent = {
            "name": f"Adjacent venue/concept lane from {top.name}",
            "topic_concepts": top.topic_concepts,
            "venue_filter": top.venue_filter,
            "min_year": top.min_year or 0,
            "min_citations": max((top.min_citations or 0) // 2, 0),
            "ror_country_filter": top.ror_country_filter,
            "adapted_reason": "productive_signal_adjacent_lane",
        }
        new_queries.append(adjacent)
        rationale_parts.append("Saved candidates found; added an adjacent lane around that signal.")

    if skip_ids:
        rationale_parts.append("Noisy zero-save queries marked for skip.")

    return {
        "new_researcher_queries": new_queries[:3],
        "skip_query_ids": skip_ids,
        "reorder_query_ids": [],
        "rationale": " ".join(rationale_parts) or "Heuristic adaptation found no change.",
    }


def _parse_adaptation_response(
    raw: dict[str, Any],
    report: ResearcherAdaptiveReport,
    remaining_queries: list[dict[str, Any]],
) -> ResearcherAdaptationPlan:
    if not isinstance(raw, dict):
        raw = _heuristic_adaptation_response(report, remaining_queries)

    remaining_ids = {
        int(query.get("id") or 0)
        for query in remaining_queries
        if query.get("id") is not None
    }
    used_ids = remaining_ids | {report.query_id for report in report.query_reports}
    next_id = max(used_ids or {0}) + 1
    source_config: dict[str, Any] = {}

    raw_new = raw.get("new_researcher_queries")
    if raw_new is None:
        raw_new = raw.get("generated_strings")
    if not isinstance(raw_new, list):
        raw_new = []

    new_queries: list[dict[str, Any]] = []
    for item in raw_new:
        if not isinstance(item, dict):
            continue
        candidate = dict(item)
        if not candidate.get("id") or int(candidate.get("id") or 0) in used_ids:
            candidate["id"] = next_id
            next_id += 1
        normalized = _normalize_query(candidate, idx=len(new_queries), source_config=source_config)
        normalized["adapted_from_batch"] = report.batch_name
        if isinstance(candidate.get("source_mix"), dict):
            normalized["source_mix"] = dict(candidate["source_mix"])
        if candidate.get("adapted_reason"):
            normalized["adapted_reason"] = str(candidate["adapted_reason"])
        new_queries.append(normalized)

    skip_ids = [
        int(item)
        for item in raw.get("skip_query_ids") or []
        if isinstance(item, int) and item in remaining_ids
    ]
    reorder_ids = [
        int(item)
        for item in raw.get("reorder_query_ids") or []
        if isinstance(item, int) and item in remaining_ids
    ]
    rationale = _as_str(raw.get("rationale")) or "Researcher strategy adapted after scout batch."

    if not new_queries and not skip_ids and not reorder_ids:
        fallback = _heuristic_adaptation_response(report, remaining_queries)
        if fallback != raw:
            return _parse_adaptation_response(fallback, report, remaining_queries)

    return ResearcherAdaptationPlan(
        new_queries=new_queries[:5],
        skipped_query_ids=skip_ids,
        reordered_query_ids=reorder_ids,
        rationale=rationale,
    )


def _researcher_decision_from_plan(
    *,
    report: ResearcherAdaptiveReport,
    new_queries: list[dict[str, Any]],
    skipped_query_ids: list[int],
    reordered_query_ids: list[int],
    rationale: str,
) -> AdaptationDecision:
    action = _classify_researcher_adaptive_action(report, new_queries, skipped_query_ids, reordered_query_ids)
    signal_markers = [
        SignalMarker(
            kind="productive_query",
            label="queries with saves",
            count=len(report.productive_reports),
            examples=[item.name for item in report.productive_reports[:5]],
        )
    ]
    noise_markers = [
        NoiseMarker(
            kind="sparse_query",
            label="queries with no discovered candidates",
            count=len(report.sparse_query_ids),
            examples=[str(item) for item in report.sparse_query_ids[:5]],
        ),
        NoiseMarker(
            kind="noisy_query",
            label="queries with candidates but no facial/full signal",
            count=len(report.noisy_query_ids),
            examples=[str(item) for item in report.noisy_query_ids[:5]],
        ),
    ]
    metrics = ScoutMetrics(
        work_units_run=report.queries_run,
        candidates_discovered=report.total_candidates_discovered,
        facial_yes=report.total_facial_yes,
        facial_no=report.total_facial_no,
        saves=report.total_saves,
        rejects=report.total_rejects,
        signal_markers=signal_markers,
        noise_markers=noise_markers,
    )
    source_payload = {
        "batch_report": report.to_dict(),
        "new_queries": new_queries,
        "skip_query_ids": skipped_query_ids,
        "reorder_query_ids": reordered_query_ids,
    }
    return AdaptationDecision(
        source="researcher",
        action=action,
        lane="academic_search",
        rationale=rationale,
        metrics=metrics,
        work_unit_kind="researcher_author_query",
        work_unit_family="topic_concepts",
        inserted_work_units=[str(query.get("id")) for query in new_queries],
        skipped_work_units=[str(item) for item in skipped_query_ids],
        reordered_work_units=[str(item) for item in reordered_query_ids],
        source_payload=source_payload,
    )


def _classify_researcher_adaptive_action(
    report: ResearcherAdaptiveReport,
    new_queries: list[dict[str, Any]],
    skipped_query_ids: list[int],
    reordered_query_ids: list[int],
) -> AdaptiveAction:
    # Order is intentional: SKIP/REORDER first since those are pure
    # restructurings; then NARROW so noisy-query removal wins tied
    # signals over BROADEN (otherwise a batch with both sparse AND
    # noisy queries would silently broaden noise instead of cutting it).
    if skipped_query_ids and not new_queries:
        return AdaptiveAction.SKIP
    if reordered_query_ids and not new_queries:
        return AdaptiveAction.REORDER
    if report.noisy_query_ids and (skipped_query_ids or new_queries):
        return AdaptiveAction.NARROW
    if report.sparse_query_ids and new_queries:
        return AdaptiveAction.BROADEN
    if report.total_saves > 0 and new_queries:
        return AdaptiveAction.EXPERIMENT
    if new_queries:
        return AdaptiveAction.EXPERIMENT
    if report.total_saves > 0:
        return AdaptiveAction.COMMIT
    return AdaptiveAction.CONTINUE


def _default_query_name(
    *,
    idx: int,
    topic_concepts: list[str],
    venue_filter: list[str],
) -> str:
    parts: list[str] = []
    if topic_concepts:
        parts.append("+".join(topic_concepts[:2]))
    if venue_filter:
        parts.append("/".join(venue_filter[:2]))
    if not parts:
        return f"query-{idx + 1}"
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Brief raw access
# ---------------------------------------------------------------------------


def _brief_raw(brief: Brief) -> dict:
    """Return the V2 raw dict if the Brief was loaded via the V2 path.

    Falls back to a minimal dict synthesized from accessible fields when
    the brief is from the legacy path. Tests can pass any object that
    duck-types as :class:`Brief` plus an optional ``_new_brief`` shim.
    """

    raw = getattr(brief, "_new_brief", None)
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "raw_dict"):
        candidate = raw.raw_dict()
        if isinstance(candidate, dict):
            return candidate
    return {}


def _as_str(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, str) and str(v).strip()]


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


_SYSTEM_TEMPLATE = """\
You are Cloris, generating a search strategy for the Researcher module
against the OpenAlex academic publication graph. The recruiter authored a
brief; your job is to translate it into N concrete OpenAlex queries that
will surface qualified researchers.

ROLE CONTEXT
  Title: {role_title}
  Summary: {role_summary}
  Field discipline: {discipline}

CAPABILITY AREAS (recruiter's "what this person ships"):
{capability_block}

RECRUITER-AUTHORED RESEARCH TOPICS (additive context — blend with the
capability areas; not a replacement):
  {research_topics}

VENUE SEED (recruiter-allowed conferences/journals — start here, expand
only if a capability area is poorly covered):
  {venue_seed}

QUERY DESIGN PRINCIPLES
  1. Each query targets ONE conceptual axis (don't AND four concepts
     together — diluting your concept set will surface nobody).
  2. Spread queries across the capability areas; no single area should
     dominate.
  3. Venue-driven queries (filtering works at NeurIPS/ICML/ICLR) are
     stronger signal than concept-only queries when the brief names
     specific venues — prefer venue+concept combinations.
  4. Year window: default to last 36 months unless the role explicitly
     wants seasoned alumni (e.g., "research scientist with 10+ years
     publishing"); then widen to 60 months.
  5. min_citations is a courtesy filter for budget control, not an
     editorial bar. Use a low number (10–50) so deterministic gates at
     evaluation time can do their job.
"""


_USER_TEMPLATE = """\
{prior_summary}

OUTPUT: A JSON object with the following keys:

  - strategy_rationale (str): one paragraph explaining the overall plan.
  - generated_strings (list of query dicts): each dict has exactly
    these keys: {query_schema_keys}.
      - topic_concepts: list[str] of OpenAlex Concept IDs (e.g.,
        "C2778407487" for Natural Language Processing). Pick 1–3 per
        query; do not over-AND.
      - venue_filter: list[str] of OpenAlex Source IDs or venue names
        (e.g., "S4306420609" for NeurIPS, or just "NeurIPS"). Empty list
        is allowed for concept-only queries.
      - min_year: int, e.g., 2023.
      - min_citations: int, e.g., 20.
      - ror_country_filter: list[str] of ISO country codes (e.g., ["US",
        "GB", "CA"]). Empty list = global.
  - coverage_gaps (list of dicts): areas the strategy didn't reach;
    optional. Empty list is fine.
  - architecture (str): always "concept_first" for researcher v1.
  - architecture_rationale (str): one sentence explaining why this
    architecture fits the brief.

Emit 5–15 queries; spread across the capability areas. Output ONLY the
JSON object — no preamble, no markdown fences.
"""
