"""HITL Market-Intelligence engine phases — The Reflection.

Splits the monolithic :func:`market_intelligence.engine.update_market_intel`
pipeline into four phases that pause/resume around two HITL gates:

    Gate 1 — The Read:    plan          → user reviews + steers
    (in-flight)           research       → no HITL, long-running
    Gate 2 — The Diff:    propose        → user reviews diff
    (terminal)            commit         → brief written

The existing :func:`update_market_intel` function stays intact (used by
the LinkedIn post-run auto-trigger and the `update_market_intel` Tier-A
tool) so no behaviour-change risk to those callers. The phase functions
here re-use the engine's helpers (``_collect_evidence_batches``,
``_build_deterministic_summary``, the backends, ``_build_artifact``,
``_build_agent_state``, ``_merge_external_research_into_sections``)
without touching them.

State persistence model:
- Each phase function is pure with respect to the database — it returns
  a JSON-serializable dict that the API layer persists to
  ``reflection_sessions.state_json``.
- Each phase reads its prior phase's output from the same dict shape.
- This avoids serializing complex internal types (``MarketEvidenceBatch``,
  ``CriticResult``, ``MarketIntelArtifact``) across the wire — instead
  each phase re-derives them from disk when needed.

Trial-day scope (per the implementation plan):
- ``reflection_phase_propose`` ships **structured `brief_recommendations`
  → hunks** only. Full LLM brief rewrite via ``iterate_brief_draft`` is
  a follow-up enhancement.
- The editorial briefing surfaced in Gate 1 is derived **deterministically**
  from ``planner_summary``. A separate LLM-polish call is a follow-up.
- Steering refinement does re-run the planner with the steering note
  woven into ``previous_agent_state``.
"""

from __future__ import annotations

import dataclasses
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from market_intelligence.agent_backends import (
    HeuristicCriticBackend,
    HeuristicPlannerBackend,
    LLMCriticBackend,
    LLMInternalSynthesisBackend,
    LLMPlannerBackend,
    PlannerResult,
)
from cloris.chief_of_staff import (
    ChiefOfStaffAgent,
    ChiefOfStaffSynthesis,
    HeuristicChiefOfStaffSynthesizer,
)
from cloris.chief_of_staff.handoff import (
    build_handoff_payload_from_evidence_batch,
    compose_handoff_context,
)
from market_intelligence.briefing_polish import (
    BriefingPolishBackend,
    EditorialBriefing,
    HeuristicBriefingBackend,
)
from market_intelligence.engine import (
    ExternalResearchResult,
    _build_agent_state,
    _build_artifact,
    _build_deterministic_summary,
    _collect_evidence_batches,
    _emit_stage,
    _explicit_linkedin_batch_is_incomplete,
    _load_previous_agent_state,
    _load_previous_artifact,
    _maybe_build_external_research_backend,
    _merge_external_research_into_sections,
    _merge_external_results,
    _normalize_text,
    derive_market_key,
    resolve_market_intel_agent_state_path,
    resolve_market_intel_artifact_path,
    _resolve_market_intel_run_dir,
    _role_level_from_brief,
    _geography_from_brief,
)
from market_intelligence.schema import (
    MarketEvidenceBatch,
    MarketIdentity,
    render_market_intel_markdown,
    render_market_intel_technical_markdown,
)
from shared.brief_loader import load_brief
from shared.brief_writer import write_brief_atomic
from shared.llm_usage import llm_usage_session
from shared.storage import read_json, write_json


MAX_STEERING_ITERATIONS = 3


class StructuredSectionHunkError(ValueError):
    """P3.7: a hunk tried to replace a structured brief section with prose.

    Raised by ``_apply_hunk_to_brief`` instead of writing — the commit
    endpoint's ValueError handling surfaces it at Gate 2 as a 422.
    """


# ---------------------------------------------------------------------------
# Editorial briefing — superseded by market_intelligence.briefing_polish.
# ---------------------------------------------------------------------------
#
# The original v1 path called _build_editorial_briefing (which truncated
# planner_summary) and _build_intentions (which translated
# external_research_focus). Both are now folded into the
# BriefingPolishBackend / HeuristicBriefingBackend pair in briefing_polish.py.
#
# Module-level singleton: cheap to construct (just the fallback wiring)
# and reused per-call so the polish stage doesn't re-instantiate per
# session.
_BRIEFING_BACKEND = BriefingPolishBackend(fallback=HeuristicBriefingBackend())


# Chief-of-staff cross-source synthesis agent. Same singleton posture as
# _BRIEFING_BACKEND; one extra LLM call per multi-source reflection
# session, gated by ``_chief_of_staff_enabled()`` + a ≥2-candidate-
# producing-sources guard so single-source briefs are byte-equivalent
# to today's behavior. Default is ON when the env var is unset; explicit
# ``false``/``no``/``0`` disables (see ``_chief_of_staff_enabled``).
_CHIEF_OF_STAFF_AGENT = ChiefOfStaffAgent(
    fallback=HeuristicChiefOfStaffSynthesizer()
)


def _chief_of_staff_enabled() -> bool:
    """Chief-of-staff synthesis gate derived from ``CLORIS_CHIEF_OF_STAFF_ENABLED``.

    Default **on**: unset env var or empty string after strip → enabled.

    Explicit disable after case-insensitive, whitespace-trimmed match:
    ``"0"``, ``"false"``, ``"no"``. Any other non-empty token leaves
    synthesis enabled.

    Single-source reflections still skip synthesis; the contributing-
    sources guard and this gate are orthogonal.
    """

    import os

    raw = os.environ.get("CLORIS_CHIEF_OF_STAFF_ENABLED", "").strip().lower()
    return raw not in {"0", "false", "no"}


def _contributing_sources_count(
    evidence_batches: list[MarketEvidenceBatch],
) -> int:
    """Count distinct sources that produced candidates this run.

    A source that ran but surfaced zero candidates has no read for
    the chief-of-staff agent to weigh against the others — counting
    it would inflate the guard and produce synthesis where one
    specialist has nothing substantive to contribute. Saves are NOT
    a precondition: a source with candidates and zero saves still
    has a substantive read (the negative read — *"GitHub returned
    22 maintainers; none cleared the bar."*).
    """

    return len(
        {
            (batch.source or "").strip().lower()
            for batch in evidence_batches
            if int(batch.metrics_summary.get("candidate_volume", 0) or 0) > 0
            and (batch.source or "").strip()
        }
    )


def _per_source_signals_from_batches(
    evidence_batches: list[MarketEvidenceBatch],
) -> dict[str, dict]:
    """Derive the per-source signals payload for the chief-of-staff agent.

    Aggregates across batches per source (a brief may have multiple
    runs of the same source over its lifetime — they roll up into one
    specialist's read). Each entry carries
    ``{"candidate_count", "save_count", "top_lane"}``; only sources
    with ``candidate_count > 0`` are included so the synthesis input
    matches the contributing-sources set the cascade's
    ``specialist_weight_invalid`` route enforces against.

    ``top_lane`` is left as ``None`` for v1 — per-source lane
    derivation requires plumbing lane attribution through the
    ``metrics_summary`` shape, which is out of scope for the
    initial integration. The synthesis agent (LLM and heuristic)
    handle ``top_lane=None`` gracefully; per-source lane plumbing is
    a follow-up that improves grounding density without changing
    the contract.
    """

    out: dict[str, dict] = {}
    for batch in evidence_batches:
        source = (batch.source or "").strip().lower()
        if not source:
            continue
        metrics = batch.metrics_summary or {}
        candidates = int(metrics.get("candidate_volume", 0) or 0)
        if candidates <= 0:
            continue
        saves = int(metrics.get("saved", 0) or 0)
        bucket = out.setdefault(
            source,
            {"candidate_count": 0, "save_count": 0, "top_lane": None},
        )
        bucket["candidate_count"] += candidates
        bucket["save_count"] += saves
    return out


def _brief_id_for_orchestration(
    *, brief: Any, market_identity: MarketIdentity
) -> str:
    """Resolve the orchestration brief_id used by chief_of_staff_runs.

    Audit Move #1. Mirrors
    :func:`cloris.chief_of_staff.agent._brief_id_for_dispatch`'s
    fallback chain so ``merge_handoff_payload`` finds the row that
    ``_persist_dispatch_run`` wrote at dispatch time. The chain is:

    1. ``brief.id`` (canonical when set)
    2. ``brief.raw["brief_id"]`` / ``brief.raw["name"]``
    3. ``brief.role_title``
    4. ``market_identity.market_key`` (last-resort, market-grain)

    Returns ``"unknown"`` only when every fallback yields an empty
    string — same sentinel as the dispatch path.
    """

    direct = getattr(brief, "id", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    raw = getattr(brief, "raw", None)
    if isinstance(raw, dict):
        for key in ("brief_id", "name"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    role_title = getattr(brief, "role_title", None)
    if isinstance(role_title, str) and role_title.strip():
        return role_title.strip()
    market_key = (market_identity.market_key or "").strip()
    return market_key or "unknown"


def _persist_and_read_handoff_payloads(
    *,
    brief_path: str | Path,
    market_identity: MarketIdentity,
    evidence_batches: list[MarketEvidenceBatch],
) -> dict[str, dict] | None:
    """Persist per-source handoff payloads and read them back composed.

    Audit Move #1 — closes the highest-blast-radius "Thing You're Not
    Seeing" finding. For each evidence batch with non-zero candidates,
    builds a structured :class:`HandoffPayload`, merges it into the
    latest ``chief_of_staff_runs`` row's ``handoff_payloads_json``
    keyed by source, and returns the composed prior-handoff context
    for the synthesis prompt.

    Returns ``None`` when:
    - No evidence batches yielded a non-empty payload.
    - The orchestration store has no chief_of_staff_runs row for this
      brief (typical for runs that didn't go through the dispatch
      path; the merge surfaces this case via a ``False`` return that
      we log + ignore so reflection doesn't abort).

    Posture: every failure mode is fail-soft. SQLite errors on the
    orchestration store are logged via _emit_stage and the function
    returns ``None`` so the synthesis call falls back to its
    pre-Move-1 behavior (per_source_signals alone). The fallback is
    indistinguishable from "no prior context found" so the cascade
    discipline at the synthesis layer doesn't have to grow new
    failure modes.
    """

    brief = load_brief(str(brief_path))
    brief_id = _brief_id_for_orchestration(
        brief=brief, market_identity=market_identity
    )

    payloads_to_persist: list[tuple[str, dict]] = []
    for batch in evidence_batches:
        payload = build_handoff_payload_from_evidence_batch(batch)
        if payload is None:
            continue
        payloads_to_persist.append((payload.source, payload.to_dict()))

    if not payloads_to_persist:
        return None

    try:
        from shared.output_paths import resolve_orchestration_db_path
        from shared.runtime_state.orchestration_store import (
            OrchestrationStateStore,
        )

        store = OrchestrationStateStore(resolve_orchestration_db_path())
        merge_count = 0
        for source, payload_dict in payloads_to_persist:
            if store.merge_handoff_payload(
                brief_id=brief_id,
                source=source,
                payload=payload_dict,
            ):
                merge_count += 1
        _emit_stage(
            "reflection.handoff:persisted "
            f"brief_id={brief_id!r} sources={len(payloads_to_persist)} "
            f"merged={merge_count}"
        )
    except Exception as exc:  # noqa: BLE001 — fail-soft observability
        _emit_stage(
            "reflection.handoff:persist_failed "
            f"brief_id={brief_id!r} exc={exc.__class__.__name__}"
        )
        return None

    # Compose the per-source dict from what we just persisted (last-
    # write-wins matches the merge semantics; the read-back path would
    # have re-loaded the same data plus any prior-run payloads if the
    # CoS row already had history).
    composed = compose_handoff_context(
        {source: payload for source, payload in payloads_to_persist}
    )
    return composed


def _truncate(text: str, max_len: int = 240) -> str:
    """Word-boundary truncation with ellipsis, used by _hunk_label.

    Survived the editorial-helper removal because hunk labels still
    need a short-form rendering of long proposal strings.
    """

    text = _normalize_text(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Phase 1 — PLAN
# ---------------------------------------------------------------------------


def reflection_phase_plan(
    *,
    brief_path: str | Path,
    run_dir: str | Path | None = None,
    mode: str = "post_run",
    steering_notes: list[str] | None = None,
) -> dict:
    """Run pre-LLM setup + planner. Idempotent across steering refinements.

    Returns a JSON-serializable dict the API layer writes to
    ``reflection_sessions.state_json``. The dict carries everything
    later phases need to re-derive evidence + execute research +
    build the proposed diff: brief_path, run_dir, mode, market
    identity, planner result (full structured), the editorial
    briefing, intentions list, and steering history.

    A steering re-run is a fresh call with ``steering_notes`` populated.
    The notes get woven into the planner's ``previous_agent_state``
    addendum so the LLM sees them as additional context. The planner
    then produces a new ``planner_result`` whose ``external_research_focus``
    reflects the steering ask.
    """

    brief_path = Path(brief_path)
    if not brief_path.exists():
        raise FileNotFoundError(f"Brief file not found: {brief_path}")

    run_dir_path = _resolve_market_intel_run_dir(
        brief_path=brief_path,
        mode=mode,
        run_dir=run_dir,
        run_id=None,
        legacy_output_dir=None,
        output_dir=None,
        report_path=None,
        allow_live_state_dir=False,
        reconstruct_report_analysis=False,
    )
    if mode in {"post_run", "backfill"} and run_dir_path is None:
        raise ValueError(
            "post_run/backfill reflection requires a finalized run_dir under output/runs/."
        )

    raw = read_json(brief_path)
    brief = load_brief(str(brief_path))
    market_identity = MarketIdentity(
        market_key=derive_market_key(brief, raw),
        role_title=brief.role_title,
        role_level=_role_level_from_brief(brief, raw),
        geography=_geography_from_brief(brief, raw),
        channels_seen=[],
        brief_ids_seen=[],
        brief_versions_seen=[],
    )

    artifact_path = resolve_market_intel_artifact_path(
        brief_path, output_dir=run_dir_path
    )
    agent_state_path = resolve_market_intel_agent_state_path(
        brief_path, output_dir=run_dir_path
    )
    previous_artifact = _load_previous_artifact(artifact_path)
    previous_agent_state = _load_previous_agent_state(agent_state_path)

    evidence_batches = _collect_evidence_batches(
        brief_path=brief_path,
        brief=brief,
        raw=raw,
        mode=mode,
        run_dir=run_dir_path,
        report_path=None,
        previous_artifact=previous_artifact,
        reconstruct_report_analysis=mode == "backfill",
    )
    if not evidence_batches:
        raise RuntimeError("No market-intelligence evidence batches could be resolved")

    deterministic_summary = _build_deterministic_summary(
        market_identity=market_identity,
        evidence_batches=evidence_batches,
        previous_artifact=previous_artifact,
    )

    # Weave steering notes into the planner's view of previous_agent_state.
    # The planner's user prompt dumps previous_agent_state as JSON; an
    # extra "operator_steering_notes" key surfaces naturally there. This
    # is additive — when steering_notes is empty the call is identical
    # to the baseline planner invocation.
    steering_notes = list(steering_notes or [])
    planner_input_state: Any = previous_agent_state
    steering_notes_dropped = False
    if steering_notes:
        if previous_agent_state is None:
            base_state_dict: dict = {}
        else:
            base_state_dict = dict(previous_agent_state.to_dict())
        base_state_dict["operator_steering_notes"] = [
            {"iteration": idx + 1, "note": note}
            for idx, note in enumerate(steering_notes)
            if _normalize_text(note)
        ]
        # Wrap back into the dataclass so the planner backend's signature
        # is unchanged. MarketIntelAgentState.from_dict is forgiving and
        # ignores unknown top-level keys — the steering addendum survives
        # because the planner serializes the agent state with json.dumps,
        # not via from_dict.
        try:
            from market_intelligence.schema import MarketIntelAgentState

            planner_input_state = MarketIntelAgentState.from_dict(base_state_dict)
            # Stash the addendum onto the dataclass for the prompt builder
            # to find. Acceptable because the planner's user prompt dumps
            # ``previous_agent_state.to_dict()`` and to_dict is defined
            # on the dataclass, but to_dict only serializes declared
            # fields. So we fall back to a lightweight wrapper.
            planner_input_state = _AgentStateWithSteering(
                inner=planner_input_state,
                steering_notes=base_state_dict["operator_steering_notes"],
            )
        except (ValueError, TypeError, KeyError) as exc:
            # Schema reconstruction failed — fall through to the unmodified
            # previous_agent_state so the planner still runs, but emit
            # telemetry so the silent-drop is observable in production.
            # Without this emit, the recruiter sees planner output that
            # doesn't reflect their steering note this iteration with no
            # operational signal anything went wrong.
            #
            # Caught exceptions are scoped to the schema-reconstruction
            # surface MarketIntelAgentState.from_dict actually raises:
            # ValueError (missing keys, non-list fields, int/str coercion
            # failures via _require_list and direct casts), TypeError
            # (str/int on incompatible types in nested from_dicts),
            # KeyError (defensive — direct dict access after the missing-
            # keys guard, but possible from None-valued nested fields).
            # AttributeError, ImportError, and other programming-error
            # exceptions propagate so they surface during development
            # rather than getting silently swallowed.
            _emit_stage(
                "reflection.plan:steering_dropped "
                f"reason={exc.__class__.__name__} "
                f"note_len={len(base_state_dict.get('operator_steering_notes') or [])}"
            )
            steering_notes_dropped = True
            planner_input_state = previous_agent_state

    planner_backend = LLMPlannerBackend(fallback=HeuristicPlannerBackend())
    artifact_dir = artifact_path.parent
    token_cost_log_path = artifact_dir / "token-cost-log.jsonl"

    with llm_usage_session(
        token_cost_log_path,
        pipeline="market_intel_reflection",
        market_key=market_identity.market_key,
        mode=mode,
        brief_path=str(brief_path),
        phase="plan",
    ):
        _emit_stage(
            f"reflection.plan:start backend={planner_backend.__class__.__name__} "
            f"steering_iterations={len(steering_notes)}"
        )
        planner_result = planner_backend.plan(
            market_identity=market_identity,
            deterministic_summary=deterministic_summary,
            evidence_batches=evidence_batches,
            previous_artifact=previous_artifact,
            previous_agent_state=planner_input_state,
        )
        _emit_stage(
            "reflection.plan:done "
            f"focus={len(planner_result.external_research_focus)} "
            f"edge_case_focus={len(planner_result.edge_case_research_focus)}"
        )

        # The recruiter-facing briefing is computed from STRUCTURED signals
        # via the polish backend — not from planner_result.planner_summary
        # which is engineer narrative ("Tracking N hypotheses..."). The
        # polish backend handles its own four-route failure cascade and
        # emits its own start/done/fallback _emit_stage logs (see
        # market_intelligence/briefing_polish.py). Lives inside the
        # llm_usage_session block so its tokens land in the same cost log.
        briefing: EditorialBriefing = _BRIEFING_BACKEND.polish(
            market_identity=market_identity,
            deterministic_summary=deterministic_summary,
            planner_result=planner_result,
            steering_notes=steering_notes,
        )

        # Chief-of-staff cross-source synthesis (≥2 candidate-producing
        # sources when the gate is on). Lands inside the same
        # llm_usage_session so the extra LLM call's tokens are captured in
        # the same cost log alongside the planner + polish calls. The
        # gate defaults ON when ``CLORIS_CHIEF_OF_STAFF_ENABLED`` is
        # unset and can be explicitly disabled via ``false``/``no``/``0``
        # (case-insensitive). When the gate reads off or only one source
        # produced candidates, ``chief_of_staff_synthesis_dict`` stays
        # ``None``.
        #
        # Audit Move #1: BEFORE synthesis fires, build per-source
        # handoff payloads from the evidence batches and persist them
        # into ``chief_of_staff_runs.handoff_payloads_json`` keyed by
        # source. Then read the persisted payloads back (so we observe
        # the same shape downstream consumers will see) and pass them
        # to the synthesis call as ``prior_handoff_payloads``. Closes
        # the highest-blast-radius "Thing You're Not Seeing" finding —
        # the schema field is no longer dead code, and the synthesis
        # has cross-source narrative depth.
        chief_of_staff_synthesis_dict: dict | None = None
        if (
            _chief_of_staff_enabled()
            and _contributing_sources_count(evidence_batches) >= 2
        ):
            per_source_signals = _per_source_signals_from_batches(
                evidence_batches
            )
            prior_handoff_payloads = _persist_and_read_handoff_payloads(
                brief_path=brief_path,
                market_identity=market_identity,
                evidence_batches=evidence_batches,
            )
            synthesis: ChiefOfStaffSynthesis = (
                _CHIEF_OF_STAFF_AGENT.synthesize(
                    market_identity=market_identity,
                    per_source_signals=per_source_signals,
                    briefing_paragraph=briefing.paragraph,
                    deterministic_summary=deterministic_summary,
                    prior_handoff_payloads=prior_handoff_payloads,
                )
            )
            chief_of_staff_synthesis_dict = synthesis.to_dict()

    return {
        "phase_outputs": {
            "plan": {
                "planner_result": planner_result.to_dict(),
                "briefing": briefing.to_dict(),
                "chief_of_staff_synthesis": chief_of_staff_synthesis_dict,
                "should_collect_external": bool(
                    planner_result.should_collect_external_research
                ),
                "should_collect_edge_case": bool(
                    planner_result.should_collect_edge_case_research
                ),
            }
        },
        "context": {
            "brief_path": str(brief_path),
            "run_dir": str(run_dir_path) if run_dir_path else None,
            "mode": mode,
            "market_identity": market_identity.to_dict(),
        },
        "steering_history": [
            {"iteration": idx + 1, "note": note, "timestamp": _utc_now()}
            for idx, note in enumerate(steering_notes)
        ],
        "steering_notes_dropped": steering_notes_dropped,
    }


class _AgentStateWithSteering:
    """Tactical bypass around ``MarketIntelAgentState.to_dict()``.

    NOT load-bearing architecture. This wrapper exists only because
    ``MarketIntelAgentState.to_dict()`` (via ``dataclasses.asdict``)
    serializes only declared fields, so an ``operator_steering_notes``
    addendum stashed on the dataclass would be dropped on the way to
    the prompt. Wrapping with ``__getattr__`` delegation + a
    ``to_dict()`` override surfaces the addendum to the planner's user
    prompt (which dumps ``previous_agent_state.to_dict()``) without
    touching the prompt builder or the schema.

    Strategic move: promote ``operator_steering_notes`` to an optional
    first-class field on :class:`MarketIntelAgentState` (with a
    matching slot in ``from_dict`` / ``to_dict`` and a default of
    ``[]``). At that point this wrapper goes away — the planner sees
    steering notes through the same dataclass round-trip every other
    field uses, and the steering-dropped cascade above (which exists
    because ``from_dict`` can fail when reconstructing a possibly-
    malformed prior state) becomes a normal optional-field path.

    Until that promotion lands, treat this wrapper as a tactical
    seam. Don't propagate the pattern to other wrap-and-augment cases
    around dataclasses; promote the field instead.
    """

    def __init__(self, *, inner: Any, steering_notes: list[dict]) -> None:
        self._inner = inner
        self._steering_notes = steering_notes

    def to_dict(self) -> dict:
        base = self._inner.to_dict() if self._inner is not None else {}
        base["operator_steering_notes"] = self._steering_notes
        return base

    def __getattr__(self, name: str) -> Any:
        # Delegation so anything else the planner reads off the agent
        # state continues to work transparently. ``__getattr__`` only
        # fires for names not found on the wrapper itself, which is
        # what we want.
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Phase 2 — RESEARCH
# ---------------------------------------------------------------------------


def reflection_phase_research(*, state: dict) -> dict:
    """Execute external research using the approved planner focus.

    Long-running. The API layer kicks this off in a background thread
    after Gate 1 approval; on completion the API patches the session
    with the new state.

    Returns the input ``state`` dict plus a new ``research`` block
    under ``phase_outputs``. The block carries the
    ``ExternalResearchResult`` as a dict (via ``dataclasses.asdict``),
    plus stage_errors and a summary count for the polling status
    surface.

    If no external research backend is configured (no Perplexity /
    Anthropic key), the research phase is a no-op that records the
    skip reason and lets the propose phase synthesize from internal
    evidence only.
    """

    plan_block = (state.get("phase_outputs") or {}).get("plan") or {}
    context = state.get("context") or {}
    if not plan_block:
        raise ValueError("research phase requires a completed plan phase")
    brief_path = Path(context["brief_path"])
    run_dir = Path(context["run_dir"]) if context.get("run_dir") else None
    mode = context.get("mode", "post_run")

    raw = read_json(brief_path)
    brief = load_brief(str(brief_path))
    market_identity = MarketIdentity.from_dict(context["market_identity"])
    planner_result = _planner_result_from_dict(plan_block["planner_result"])

    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir)
    agent_state_path = resolve_market_intel_agent_state_path(
        brief_path, output_dir=run_dir
    )
    previous_artifact = _load_previous_artifact(artifact_path)
    previous_agent_state = _load_previous_agent_state(agent_state_path)
    evidence_batches = _collect_evidence_batches(
        brief_path=brief_path,
        brief=brief,
        raw=raw,
        mode=mode,
        run_dir=run_dir,
        report_path=None,
        previous_artifact=previous_artifact,
        reconstruct_report_analysis=mode == "backfill",
    )

    external_backend = _maybe_build_external_research_backend()
    artifact_dir = artifact_path.parent
    token_cost_log_path = artifact_dir / "token-cost-log.jsonl"
    external_result: ExternalResearchResult | None = None
    stage_errors: list[str] = []
    skip_reason: str | None = None

    if external_backend is None:
        skip_reason = "no_backend"
    else:
        batch_incomplete = _explicit_linkedin_batch_is_incomplete(
            evidence_batches=evidence_batches, output_dir=None
        )
        if batch_incomplete:
            skip_reason = "incomplete_run"
        elif not (
            planner_result.should_collect_external_research
            or planner_result.should_collect_edge_case_research
        ):
            skip_reason = "planner_disabled"

    if skip_reason is None and external_backend is not None:
        with llm_usage_session(
            token_cost_log_path,
            pipeline="market_intel_reflection",
            market_key=market_identity.market_key,
            mode=mode,
            brief_path=str(brief_path),
            phase="research",
        ):
            if planner_result.should_collect_external_research:
                try:
                    _emit_stage(
                        "reflection.research:start "
                        f"backend={external_backend.__class__.__name__} "
                        f"focus={len(planner_result.external_research_focus)}"
                    )
                    external_result = external_backend.collect(
                        market_identity=market_identity,
                        previous_artifact=previous_artifact,
                        previous_agent_state=previous_agent_state,
                        evidence_batches=evidence_batches,
                        planner_result=planner_result,
                        research_focus=planner_result.external_research_focus,
                        research_mode="general",
                    )
                    _emit_stage(
                        "reflection.research:done "
                        f"sources={len(external_result.sources)} "
                        f"findings={len(external_result.market_findings)} "
                        f"implications={len(external_result.sourcing_implications)}"
                    )
                except Exception as exc:
                    stage_errors.append(f"external_research:{exc}")
                    _emit_stage(f"reflection.research:error {exc}")
            if planner_result.should_collect_edge_case_research:
                try:
                    _emit_stage(
                        "reflection.research:edge_case_start "
                        f"focus={len(planner_result.edge_case_research_focus)}"
                    )
                    edge_case_result = external_backend.collect(
                        market_identity=market_identity,
                        previous_artifact=previous_artifact,
                        previous_agent_state=previous_agent_state,
                        evidence_batches=evidence_batches,
                        planner_result=planner_result,
                        research_focus=planner_result.edge_case_research_focus,
                        research_mode="edge_case",
                        edge_case_reasoning=planner_result.edge_case_research_reasoning,
                    )
                    edge_case_result.edge_case_triggered = True
                    edge_case_result.edge_case_reasoning = (
                        planner_result.edge_case_research_reasoning
                    )
                    edge_case_result.edge_case_focus = list(
                        planner_result.edge_case_research_focus
                    )
                    external_result = _merge_external_results(
                        external_result, edge_case_result
                    )
                except Exception as exc:
                    stage_errors.append(f"edge_case_research:{exc}")
                    _emit_stage(f"reflection.research:edge_case_error {exc}")

    research_payload: dict[str, Any]
    if external_result is None:
        research_payload = {
            "external_result": None,
            "skip_reason": skip_reason or "no_focus",
            "stage_errors": stage_errors,
            "summary": {
                "sources": 0,
                "findings": 0,
                "implications": 0,
            },
        }
    else:
        research_payload = {
            "external_result": dataclasses.asdict(external_result),
            "skip_reason": None,
            "stage_errors": stage_errors,
            "summary": {
                "sources": len(external_result.sources),
                "findings": len(external_result.market_findings),
                "implications": len(external_result.sourcing_implications),
            },
        }

    next_state = dict(state)
    outputs = dict(state.get("phase_outputs") or {})
    outputs["research"] = research_payload
    next_state["phase_outputs"] = outputs
    return next_state


# ---------------------------------------------------------------------------
# Phase 3 — PROPOSE
# ---------------------------------------------------------------------------


def reflection_phase_propose(*, state: dict) -> dict:
    """Run synthesis + critic + build the artifact + compute brief hunks.

    Does NOT write the canonical artifact to disk — that happens at
    commit time. The artifact is held in ``state_json`` as the source
    for the brief diff hunks the user reviews at Gate 2.

    The hunks are derived from the artifact's ``brief_recommendations``
    structured list (the engine's existing taxonomy of brief-mutation
    proposals) projected onto a UI-friendly per-hunk schema. Trial-day
    scope: every recommendation becomes one hunk; the propose phase
    does not call ``iterate_brief_draft`` for a full brief rewrite.
    """

    plan_block = (state.get("phase_outputs") or {}).get("plan") or {}
    research_block = (state.get("phase_outputs") or {}).get("research") or {}
    context = state.get("context") or {}
    if not plan_block:
        raise ValueError("propose phase requires a completed plan phase")

    brief_path = Path(context["brief_path"])
    run_dir = Path(context["run_dir"]) if context.get("run_dir") else None
    mode = context.get("mode", "post_run")

    raw = read_json(brief_path)
    brief = load_brief(str(brief_path))
    market_identity = MarketIdentity.from_dict(context["market_identity"])
    planner_result = _planner_result_from_dict(plan_block["planner_result"])

    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir)
    agent_state_path = resolve_market_intel_agent_state_path(
        brief_path, output_dir=run_dir
    )
    previous_artifact = _load_previous_artifact(artifact_path)
    previous_agent_state = _load_previous_agent_state(agent_state_path)
    evidence_batches = _collect_evidence_batches(
        brief_path=brief_path,
        brief=brief,
        raw=raw,
        mode=mode,
        run_dir=run_dir,
        report_path=None,
        previous_artifact=previous_artifact,
        reconstruct_report_analysis=mode == "backfill",
    )
    deterministic_summary = _build_deterministic_summary(
        market_identity=market_identity,
        evidence_batches=evidence_batches,
        previous_artifact=previous_artifact,
    )

    external_result_dict = research_block.get("external_result")
    external_result: ExternalResearchResult | None = None
    if external_result_dict is not None:
        try:
            external_result = ExternalResearchResult(**external_result_dict)
        except Exception:
            # Defensive: if the persisted dict has unexpected keys
            # (schema drift between phases), fall back to None and let
            # the propose phase synthesize from internal evidence.
            external_result = None

    artifact_dir = artifact_path.parent
    token_cost_log_path = artifact_dir / "token-cost-log.jsonl"
    synthesis_backend = LLMInternalSynthesisBackend(
        fallback_backend=_HeuristicSynthesisBackendShim()
    )
    critic_backend = LLMCriticBackend(fallback=HeuristicCriticBackend())
    stage_errors: list[str] = list(research_block.get("stage_errors") or [])
    preserve_previous_narrative = False

    with llm_usage_session(
        token_cost_log_path,
        pipeline="market_intel_reflection",
        market_key=market_identity.market_key,
        mode=mode,
        brief_path=str(brief_path),
        phase="propose",
    ):
        try:
            generated_sections = synthesis_backend.synthesize(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                evidence_batches=evidence_batches,
                previous_artifact=previous_artifact,
                planner_result=planner_result,
                external_research=external_result,
            )
            generated_sections = _merge_external_research_into_sections(
                generated_sections, external_result
            )
        except Exception as exc:
            preserve_previous_narrative = True
            stage_errors.append(f"synthesis:{exc}")
            _emit_stage(f"reflection.propose:synthesis_error {exc}")
            generated_sections = _merge_external_research_into_sections({}, external_result)

        try:
            critic_result = critic_backend.critique(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                evidence_batches=evidence_batches,
                previous_artifact=previous_artifact,
                planner_result=planner_result,
                draft_sections=generated_sections,
                external_research=external_result,
            )
        except Exception as exc:
            preserve_previous_narrative = True
            stage_errors.append(f"critic:{exc}")
            _emit_stage(f"reflection.propose:critic_error {exc}")
            critic_result = HeuristicCriticBackend().critique(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                evidence_batches=evidence_batches,
                previous_artifact=previous_artifact,
                planner_result=planner_result,
                draft_sections=generated_sections,
                external_research=external_result,
            )

    artifact = _build_artifact(
        brief=brief,
        market_identity=market_identity,
        deterministic_summary=deterministic_summary,
        evidence_batches=evidence_batches,
        previous_artifact=previous_artifact,
        generated_sections=critic_result.keep_sections or generated_sections,
        preserve_previous_narrative=preserve_previous_narrative,
        external_result=external_result,
        section_generation_metadata=critic_result.section_generation_metadata,
        delta_since_last_run=critic_result.delta_since_last_run,
    )
    agent_state = _build_agent_state(
        market_identity=market_identity,
        evidence_batches=evidence_batches,
        previous_agent_state=previous_agent_state,
        planner_result=planner_result,
        critic_result=critic_result,
        external_result=external_result,
    )

    artifact_dict = artifact.to_dict()
    hunks = _build_hunks_from_artifact(artifact_dict, brief_raw=raw)
    # Multi-Agent Execution Plan Slice 3.4: calibration-derived
    # patches. Walks the canonical runtime-state SQLite for every
    # state-dir whose latest run carries this brief_id, aggregates
    # ``judgment_accuracy`` markers via the Slice-3.1 aggregator,
    # gates them through the Slice-3.2 threshold layer, runs the
    # Slice-3.3 translator, and projects the resulting BriefPatch
    # objects onto Gate-2 hunk dicts. Calibration is a proposal layer
    # (recruiter approves at Gate 2 like any other hunk), not
    # autopilot. No-ops cleanly when the brief has no id, when no
    # state-dir matches the brief_id (no run has been completed for
    # this brief yet), when no markers cleared the threshold layer,
    # or when control-plane / aggregator imports fail (defensive
    # against future module reshapes).
    hunks.extend(
        _calibration_propose_hunks(
            brief=brief, brief_raw=raw, brief_path=brief_path
        )
    )
    # P3.6: facial-calibration drift → recalibration hunk. Reads the
    # facial_calibration_observed block this propose phase's own
    # _build_artifact call just computed above (artifact_dict) — no
    # additional I/O. No-ops cleanly per _facial_calibration_drift_propose_hunks'
    # docstring (consecutive-out-of-band counter below threshold, brief has
    # no facial_calibration section, or the observed block has no rate).
    hunks.extend(
        _facial_calibration_drift_propose_hunks(
            artifact_dict=artifact_dict, brief_raw=raw
        )
    )
    # P6.8: this slot previously called the Designer rubric-refinement
    # composer (Multi-Agent Execution Plan Slice 3.5 —
    # ``_designer_rubric_refine_propose_hunks``, still defined above,
    # not deleted). Designer is sunset per the module-consolidation
    # decision (2026-07-02); replaced with the github ecosystem-momentum
    # composer, which becomes meaningful once P6.1's concurrent-agent
    # slice populates maintainership on saves. No-ops cleanly when
    # there's no github-source evidence batch for this run or when its
    # final_judgments carry no maintainership-classified saves yet.
    hunks.extend(
        _github_reflection_propose_hunks(
            brief_raw=raw, evidence_batches=evidence_batches
        )
    )

    next_state = dict(state)
    outputs = dict(state.get("phase_outputs") or {})
    outputs["propose"] = {
        "artifact": artifact_dict,
        "agent_state": agent_state.to_dict(),
        "critic_summary": critic_result.critique_summary,
        "stage_errors": stage_errors,
        "hunks": hunks,
        "brief_at_propose": raw,  # snapshot for diff/commit reference
    }
    next_state["phase_outputs"] = outputs
    return next_state


class _HeuristicSynthesisBackendShim:
    """Minimal heuristic synthesis when no LLM is available.

    The LLM synthesis backend's fallback is the engine's
    ``HeuristicMarketIntelSynthesisBackend``. To avoid a circular
    import (engine → reflection → engine), we re-import lazily here.
    """

    def synthesize(self, **kwargs: Any) -> dict:
        from market_intelligence.engine import HeuristicMarketIntelSynthesisBackend

        return HeuristicMarketIntelSynthesisBackend().synthesize(**kwargs)


# ---------------------------------------------------------------------------
# Phase 4 — COMMIT
# ---------------------------------------------------------------------------


def reflection_commit(
    *,
    state: dict,
    accepted_hunk_ids: list[str],
    edited_hunks: dict[str, dict] | None = None,
) -> dict:
    """Apply accepted (and optionally edited) hunks; persist artifact + brief.

    Steps:
    1. Apply each accepted (and edited-if-present) hunk to the brief
       snapshot taken at propose-time; produce ``next_brief``.
    2. Write the proposed market-intel artifact to disk (canonical +
       history snapshot), matching what ``update_market_intel`` does.
    3. Write the new brief via ``write_brief_atomic`` so a ``versions/``
       snapshot is created.
    4. Return ``{"brief_version_path": <path>, "applied_hunks": [...]}``
       so the API layer can record it on the session row.
    """

    propose_block = (state.get("phase_outputs") or {}).get("propose") or {}
    context = state.get("context") or {}
    if not propose_block:
        raise ValueError("commit requires a completed propose phase")

    brief_path = Path(context["brief_path"])
    run_dir = Path(context["run_dir"]) if context.get("run_dir") else None
    artifact_payload = propose_block.get("artifact") or {}
    hunks = propose_block.get("hunks") or []
    base_brief = dict(propose_block.get("brief_at_propose") or read_json(brief_path))

    accepted_set = set(accepted_hunk_ids or [])
    edited_hunks = dict(edited_hunks or {})

    applied: list[dict] = []
    next_brief = dict(base_brief)
    for hunk in hunks:
        hunk_id = hunk.get("hunk_id")
        if hunk_id not in accepted_set:
            continue
        effective = dict(hunk)
        if hunk_id in edited_hunks:
            effective["after"] = edited_hunks[hunk_id].get(
                "after", effective.get("after")
            )
        try:
            next_brief = _apply_hunk_to_brief(next_brief, effective)
            applied.append(
                {
                    "hunk_id": hunk_id,
                    "section": effective.get("section"),
                    "kind": effective.get("kind"),
                    "edited": hunk_id in edited_hunks,
                }
            )
        except Exception as exc:
            # Skip hunks we can't safely apply — surface in the commit
            # response so the API layer can include them in a soft
            # warning. The brief still commits with the hunks that
            # applied cleanly.
            applied.append(
                {
                    "hunk_id": hunk_id,
                    "section": effective.get("section"),
                    "kind": effective.get("kind"),
                    "skipped_reason": f"apply_failed: {exc}",
                }
            )

    # Persist the market-intel artifact (canonical + history snapshot).
    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir)
    agent_state_path = resolve_market_intel_agent_state_path(
        brief_path, output_dir=run_dir
    )
    artifact_dir = artifact_path.parent
    history_dir = artifact_dir / "history"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_path, artifact_payload)
    if propose_block.get("agent_state"):
        write_json(agent_state_path, propose_block["agent_state"])
    history_stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json(history_dir / f"{history_stem}.json", artifact_payload)

    # Write the new brief atomically. write_brief_atomic creates a
    # versions/<stamp>.json snapshot as a side effect — that path is
    # what we surface to the recruiter as "the new brief version".
    write_brief_atomic(abs_path=brief_path, payload=next_brief)
    versions_dir = brief_path.parent / "versions"
    new_versions = sorted(
        versions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    brief_version_path = (
        str(new_versions[0]) if new_versions else str(brief_path)
    )

    return {
        "brief_version_path": brief_version_path,
        "applied_hunks": applied,
        "next_brief": next_brief,
    }


# ---------------------------------------------------------------------------
# Hunk computation + application
# ---------------------------------------------------------------------------


_HUNK_TARGET_TO_SECTION = {
    "additional_search_terms": "additional_search_terms",
    "employer_signal_rules": "employer_signal_rules",
    "search_priorities": "search_priorities",
    "instructions": "instructions",
    "notes": "notes",
}


def _build_hunks_from_artifact(artifact: dict, *, brief_raw: dict) -> list[dict]:
    """Project artifact ``brief_recommendations`` onto UI hunks.

    Each recommendation becomes one hunk. The hunk schema:

    .. code-block:: json

        {
          "hunk_id": "rec-...",
          "section": "additional_search_terms" | "employer_signal_rules" | ...,
          "kind": "add" | "modify",
          "label": "Short editorial label for the hunk",
          "before": "current value (string or null)",
          "after": "proposed value (string)",
          "rationale": "why Cloris wants this change",
          "confidence": 0.0-1.0,
          "default_approved": true | false
        }

    For trial-day scope only the most common ``target_field`` values
    are surfaced. Unknown targets fall through with ``section="notes"``.
    Hunks where the proposed value is already present in the brief
    are dropped (no-op recommendations).
    """

    recommendations = artifact.get("brief_recommendations") or []
    hunks: list[dict] = []
    for raw_rec in recommendations:
        if not isinstance(raw_rec, dict):
            continue
        rec_id = _normalize_text(raw_rec.get("recommendation_id"))
        target = _normalize_text(raw_rec.get("target_field")).lower()
        proposal = _normalize_text(raw_rec.get("proposal"))
        rationale = _normalize_text(raw_rec.get("reason"))
        confidence = _coerce_confidence(raw_rec.get("confidence"))
        if not proposal:
            continue
        section = _HUNK_TARGET_TO_SECTION.get(target, "notes")
        before, kind = _hunk_before_and_kind(brief_raw, section, proposal)
        if before == proposal:
            # No-op recommendation; skip.
            continue
        hunks.append(
            {
                "hunk_id": rec_id or f"hunk-{len(hunks) + 1}",
                "section": section,
                "kind": kind,
                "label": _hunk_label(section, kind, proposal),
                "before": before,
                "after": proposal,
                "rationale": rationale,
                "confidence": confidence,
                "default_approved": confidence >= 0.65,
                "target_field": target or section,
            }
        )
    return hunks


def _designer_rubric_refine_propose_hunks(
    *,
    brief_raw: dict,
    brief_path: Path,
) -> list[dict]:
    """Project Designer-persisted rubric-refine hunks onto propose-phase shape.

    Multi-Agent Execution Plan Slice 3.5: the Designer
    session-orchestrator's run-end hook
    (``designer/run_end.py:run_end_designer_rubric_refinement``)
    persists proposed ``RubricRefineHunk`` records under
    ``<designer_state_dir>/proposed_rubric_refinement_hunks.json``.
    This helper loads them, maps each onto the propose-phase hunk
    dict shape (``hunk_id`` / ``section`` / ``kind`` / ``label`` /
    ``before`` / ``after`` / ``rationale`` / ``confidence`` /
    ``default_approved`` / ``target_field``), and returns them for
    ``reflection_phase_propose`` to merge into the hunks list.

    Defensive failure modes — every one returns ``[]``:

    - Brief does not target the Designer module
      (``"designer" not in brief.target_modules``).
    - Designer state dir cannot be resolved (lookup raises).
    - Persisted file does not exist (no Designer run has happened
      for this brief yet, OR the run's session-orchestrator failed
      before its run-end hook).
    - Persisted file is malformed (loader returns ``[]``).
    """

    target_modules = brief_raw.get("target_modules")
    if not isinstance(target_modules, list) or "designer" not in target_modules:
        return []

    try:
        from designer.run_end import (
            load_designer_rubric_refinement_hunks,
        )
        from shared.output_paths import resolve_designer_state_dir
    except ImportError:
        return []

    try:
        state_dir = resolve_designer_state_dir(brief_path=brief_path)
    except Exception:
        # Designer state dir resolution can raise on non-Designer
        # briefs that nevertheless name "designer" in target_modules
        # but lack the rest of the brief shape; tolerate and skip.
        return []

    rubric_hunks = load_designer_rubric_refinement_hunks(state_dir)
    if not rubric_hunks:
        return []

    out: list[dict] = []
    for index, hunk in enumerate(rubric_hunks):
        out.append(
            {
                # Stable hunk_id derived from section + index so the
                # frontend's per-hunk approve/skip toggle has a key
                # that survives a re-fetch of the same propose phase.
                "hunk_id": f"rubric-refine-{index + 1}",
                "section": hunk.section,
                "kind": hunk.kind,
                "label": hunk.label,
                "before": hunk.before,
                "after": hunk.after,
                "rationale": hunk.rationale,
                # Rubric refinements ship with deliberately moderate
                # confidence — the recruiter always confirms; Cloris
                # does not auto-apply rubric mutations under any
                # circumstance (see
                # market_intelligence/design_market_intelligence.py's
                # hard preservation contract).
                "confidence": 0.6,
                "default_approved": False,
                "target_field": hunk.section,
            }
        )
    return out


def _github_reflection_propose_hunks(
    *,
    brief_raw: dict,
    evidence_batches: list[MarketEvidenceBatch],
) -> list[dict]:
    """Project github ecosystem-momentum hunks onto propose-phase shape.

    P6.8: wires :func:`market_intelligence.github_reflection.propose_github_hunks`
    into the propose phase, in the slot previously occupied by the
    Designer rubric-refinement composer
    (:func:`_designer_rubric_refine_propose_hunks`, Slice 3.5). Designer
    is sunset per the module-consolidation decision (2026-07-02) — its
    composer function is left in place (not deleted, still callable /
    still tested) but is no longer invoked from
    :func:`reflection_phase_propose`.

    Pools ``final_judgments`` across every github-source evidence
    batch this run gathered (mirrors the calibration layer's
    cross-source merge posture at :func:`_calibration_propose_hunks` —
    a brief can have multiple github run dirs, e.g. current run +
    imported legacy runs) and hands them to
    :func:`~market_intelligence.github_reflection.propose_github_hunks`,
    which already carries its own NEEDS-REVIEW-by-default discipline
    (every emitted hunk's ``confidence`` is below the 0.65
    ``default_approved`` threshold — see
    ``github_reflection._build_hunk``) — this composer proposes at
    Gate 2, it never auto-writes the brief.

    Defensive failure modes — every one returns ``[]``:

    - No github-source evidence batch for this run (classic LinkedIn
      brief, or a github brief with no completed run yet).
    - The pooled ``final_judgments`` are empty, or none carry a
      SAVE-class decision with a maintainership classification (see
      ``propose_github_hunks`` docstring — this is the ordinary case
      until P6.1's concurrent-agent slice populates maintainership on
      saves).
    - Import of :mod:`market_intelligence.github_reflection` fails
      (future module reshape) — lazy import, mirrors the
      ``_calibration_propose_hunks`` / ``_designer_rubric_refine_propose_hunks``
      pattern of keeping this module's import-time graph narrow.
    """

    github_batches = [b for b in evidence_batches if b.source == "github"]
    if not github_batches:
        return []

    final_judgments: list[dict] = []
    for batch in github_batches:
        final_judgments.extend(batch.final_judgments or [])
    if not final_judgments:
        return []

    try:
        from market_intelligence.github_reflection import propose_github_hunks
    except ImportError:
        return []

    return propose_github_hunks(final_judgments=final_judgments, brief_raw=brief_raw)


# ---------------------------------------------------------------------------
# Slice 3.4: calibration → Gate-2 hunks
# ---------------------------------------------------------------------------
#
# Why this lives here (and not in the translator at
# ``market_intelligence/calibration_to_brief.py``): the translator is a
# pure function over (rollup, eligible_areas) — it does not walk the
# runtime-state SQLite or know about brief-id resolution / multi-source
# state-dir enumeration. The reflection-pipeline integration owns those
# concerns because it's the layer that has ``brief_path`` + ``brief.id``
# in scope and that knows about the cross-source merge rule (one
# per-source rollup → one merged rollup → one threshold pass → one
# translator pass → one per-cycle cap).
#
# Cross-source merge rule (Slice card §3.2 corollary): the
# ``MAX_PATCHES_PER_CYCLE = 5`` per-cycle cap is per reflection cycle,
# not per source. If we ran the threshold layer once per source, two
# sources could each surface 5 patches and overwhelm the recruiter at
# Gate 2. So we merge rollups across sources first, then run threshold
# + translator once on the merged surface.

# Calibration patches are deliberately scored just below the
# ``default_approved`` threshold (0.65 at ``_build_hunks_from_artifact``)
# so they default to NEEDS-REVIEW at Gate 2 — calibration is a proposal
# layer, not autopilot, per the slice card. The recruiter must
# affirmatively approve every calibration hunk before it commits.
_CALIBRATION_PATCH_CONFIDENCE: float = 0.55


def _calibration_propose_hunks(
    *,
    brief: Any,
    brief_raw: dict,
    brief_path: Path,
) -> list[dict]:
    """Compute calibration-derived Gate-2 hunks from runtime-state markers.

    Multi-Agent Execution Plan Slice 3.4. Composes the three calibration
    primitives:

    1. ``shared.runtime_state.calibration.aggregate_calibration_markers``
       (Slice 3.1) — pure read of ``judgment_accuracy`` rows from one
       per-source ``runtime_state.sqlite3``.
    2. ``market_intelligence.calibration_thresholds.select_eligible_areas``
       (Slice 3.2) — gates the rollup on per-area + per-cycle thresholds,
       emits ``calibration.proposer:eligible`` telemetry.
    3. ``market_intelligence.calibration_to_brief.translate_eligible_areas``
       (Slice 3.3) — projects eligible areas onto V2 brief patches per
       three pattern rules.

    Cross-source aggregation: a single brief may live in multiple
    state-dirs (e.g., LinkedIn + GitHub). We aggregate per state-dir,
    merge the rollups (additive over counts + per-axis breakdowns +
    weighted_markers_by_area), then run the threshold + translator
    once on the merged surface. The threshold layer's per-cycle cap is
    a per-reflection-cycle cap, not per-source — running it twice
    would let a multi-source brief surface up to 2 × MAX_PATCHES_PER_CYCLE
    patches and overwhelm the recruiter.

    Defensive failure modes — every one returns ``[]``:

    - ``brief.id`` is empty (a brief without an id can't be matched
      against runtime-state ``brief_id``).
    - Imports for control-plane / aggregator / threshold / translator
      fail (future module reshape).
    - ``state_dirs_for_brief_id`` returns nothing (no run has been
      completed for this brief yet).
    - Every per-source aggregation collapses to ``total_markers == 0``
      (no recruiter has marked candidates yet).
    - ``select_eligible_areas`` returns nothing (no area cleared the
      threshold floor).
    - ``translate_eligible_areas`` returns nothing (no per-pattern
      floor met).

    Each defensive branch logs nothing — the threshold layer's
    per-area ``calibration.proposer:eligible`` telemetry is the
    canonical observability surface for "why didn't a patch surface."
    """

    brief_id = _calibration_brief_id(brief, brief_raw)
    if not brief_id:
        return []

    # Lazy imports: keep the reflection module's import-time graph
    # narrow. Mirrors the Slice-3.5 pattern at
    # ``_designer_rubric_refine_propose_hunks`` and
    # ``_HeuristicSynthesisBackendShim.synthesize``.
    try:
        from cloris.control_plane import state_dirs_for_brief_id
        from market_intelligence.calibration_thresholds import (
            select_eligible_areas,
        )
        from market_intelligence.calibration_to_brief import (
            BriefPatch,
            PATCH_KIND_CALIBRATION_EXAMPLES,
            PATCH_KIND_DEPTH_DISTINCTION,
            PATCH_KIND_NON_FIT_PATTERN,
            translate_eligible_areas,
        )
        from shared.runtime_state.calibration import (
            CalibrationRollup,
            aggregate_calibration_markers,
        )
    except ImportError:
        return []

    state_dirs = state_dirs_for_brief_id(brief_id)
    if not state_dirs:
        return []

    rollups: list[CalibrationRollup] = []
    for source, state_dir in state_dirs:
        db_path = state_dir / "runtime_state.sqlite3"
        rollup = aggregate_calibration_markers(
            db_path, brief_id=brief_id, source=source
        )
        if rollup.total_markers > 0:
            rollups.append(rollup)
    if not rollups:
        return []

    merged = _merge_calibration_rollups(rollups, brief_id=brief_id)
    eligible = select_eligible_areas(merged)
    if not eligible:
        return []

    patches = translate_eligible_areas(rollup=merged, eligible_areas=eligible)
    if not patches:
        return []

    # Pre-bind the kind→hunk-kind map locally so the projection loop
    # carries a stable kind string for the frontend's HunkCard
    # renderer to dispatch on. Mirrors how the existing
    # ``_build_hunks_from_artifact`` derives ``kind`` from the section
    # shape, but for calibration patches the kind is already pinned
    # by the translator.
    _PATCH_KIND_TO_HUNK_KIND = {
        PATCH_KIND_NON_FIT_PATTERN: "calibration_non_fit_pattern",
        PATCH_KIND_DEPTH_DISTINCTION: "calibration_depth_distinction",
        PATCH_KIND_CALIBRATION_EXAMPLES: "calibration_examples",
    }

    out: list[dict] = []
    for index, patch in enumerate(patches):
        out.append(_calibration_patch_to_hunk(
            patch=patch,
            index=index,
            hunk_kind=_PATCH_KIND_TO_HUNK_KIND.get(patch.kind, patch.kind),
            brief_raw=brief_raw,
        ))
    return out


def _calibration_brief_id(brief: Any, brief_raw: dict) -> str:
    """Resolve the brief_id used to scope the runtime-state walk.

    Prefers ``brief.id`` (the typed dataclass field at
    ``shared.brief_loader.Brief``); falls back to ``brief_raw["id"]``
    if the dataclass attribute is missing or blank. Returns the empty
    string for any failure to resolve — caller treats that as "no
    matching state-dir possible."
    """

    candidate = getattr(brief, "id", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    raw_id = brief_raw.get("id") if isinstance(brief_raw, dict) else None
    if isinstance(raw_id, str) and raw_id.strip():
        return raw_id.strip()
    return ""


def _merge_calibration_rollups(
    rollups: list[Any],
    *,
    brief_id: str,
) -> Any:
    """Sum per-source ``CalibrationRollup`` objects into one merged rollup.

    Additive over ``counts`` (full-key Counter) and the four per-axis
    breakdowns + ``weighted_markers_by_area``. The merged rollup carries
    ``source=None`` (the merge crosses sources) and ``total_markers``
    re-derived from the summed counts so the field stays consistent
    with the ``counts`` surface even if a per-source rollup's total
    drifted.

    Single-rollup input is returned as-is (no copy needed; the rollup
    is a frozen dataclass with immutable mappings).
    """

    if len(rollups) == 1:
        return rollups[0]

    # Lazy import: dataclass + key types live in shared.runtime_state.
    from collections import Counter

    from shared.runtime_state.calibration import (
        CalibrationRollup,
        CalibrationRollupKey,
    )

    merged_counts: Counter[CalibrationRollupKey] = Counter()
    merged_marker: Counter[str] = Counter()
    merged_area: Counter[str | None] = Counter()
    merged_quartile: Counter[str] = Counter()
    merged_decision: Counter[str | None] = Counter()
    merged_weighted: Counter[str | None] = Counter()

    for rollup in rollups:
        for key, count in rollup.counts.items():
            merged_counts[key] += count
        for marker, count in rollup.by_marker_value.items():
            merged_marker[marker] += count
        for area, count in rollup.by_capability_area.items():
            merged_area[area] += count
        for quartile, count in rollup.by_confidence_quartile.items():
            merged_quartile[quartile] += count
        for decision, count in rollup.by_terminal_decision.items():
            merged_decision[decision] += count
        for area, weighted in rollup.weighted_markers_by_area.items():
            merged_weighted[area] += weighted

    return CalibrationRollup(
        brief_id=brief_id,
        source=None,
        total_markers=sum(merged_counts.values()),
        counts=dict(merged_counts),
        by_marker_value=dict(merged_marker),
        by_capability_area=dict(merged_area),
        by_confidence_quartile=dict(merged_quartile),
        by_terminal_decision=dict(merged_decision),
        weighted_markers_by_area=dict(merged_weighted),
    )


def _calibration_patch_to_hunk(
    *,
    patch: Any,
    index: int,
    hunk_kind: str,
    brief_raw: dict,
) -> dict:
    """Project one ``BriefPatch`` onto the propose-phase hunk dict shape.

    Hunk shape mirrors the existing
    ``_build_hunks_from_artifact`` projection (hunk_id / section /
    kind / label / before / after / rationale / confidence /
    default_approved / target_field) so the frontend's HunkCard
    renderer (``cloris/frontend/src/components/HunkCard.svelte``)
    dispatches uniformly across brief-recommendations and calibration
    hunks. The differences:

    - ``before`` is a structured-payload preview rendered as JSON-ish
      text (the frontend's diff view text-compares it against
      ``after``); empty string when the section currently has no
      content for this area.
    - ``after`` is the JSON-ish rendering of the proposed payload.
    - ``confidence`` is fixed at ``_CALIBRATION_PATCH_CONFIDENCE``
      (below the 0.65 default-approved threshold), so calibration
      patches default to NEEDS-REVIEW.
    """

    after = _render_calibration_after(patch)
    before = _render_calibration_before(patch, brief_raw)
    return {
        # Stable hunk_id derived from kind + index so the frontend's
        # per-hunk approve/skip toggle has a key that survives a
        # re-fetch of the same propose phase.
        "hunk_id": f"calibration-{patch.kind}-{index + 1}",
        "section": patch.target_section,
        "kind": hunk_kind,
        "label": patch.label,
        "before": before,
        "after": after,
        "rationale": patch.rationale,
        "confidence": _CALIBRATION_PATCH_CONFIDENCE,
        "default_approved": False,
        "target_field": patch.target_section,
    }


def _render_calibration_after(patch: Any) -> str:
    """Render the patch payload as recruiter-readable text.

    Three payload shapes (one per pattern; see
    ``market_intelligence/calibration_to_brief.py`` per-pattern
    docstrings):

    - ``non_fit_pattern``: list-of-dicts payload (label / description /
      why_not / examples). Render as the label + description on
      separate lines.
    - ``depth_distinction``: dict with ``section_path`` + ``addendum``.
      Render as ``"<section_path>: <addendum>"``.
    - ``calibration_examples``: TransferabilityExample dict (result /
      source_context / target_context / rationale). Render as a
      one-line summary.
    """

    # Lazy import to avoid importing the translator's constants in
    # the hot path; lookup by kind keeps the dispatch table local.
    from market_intelligence.calibration_to_brief import (
        PATCH_KIND_CALIBRATION_EXAMPLES,
        PATCH_KIND_DEPTH_DISTINCTION,
        PATCH_KIND_NON_FIT_PATTERN,
    )

    payload = patch.payload or {}
    if patch.kind == PATCH_KIND_NON_FIT_PATTERN:
        label = payload.get("label", "")
        description = payload.get("description", "")
        return f"{label}\n{description}".strip()
    if patch.kind == PATCH_KIND_DEPTH_DISTINCTION:
        section_path = payload.get("section_path", patch.target_section)
        addendum = payload.get("addendum", "")
        return f"{section_path}: {addendum}".strip()
    if patch.kind == PATCH_KIND_CALIBRATION_EXAMPLES:
        result = payload.get("result", "")
        source_context = payload.get("source_context", "")
        target_context = payload.get("target_context", "")
        rationale = payload.get("rationale", "")
        return (
            f"{source_context} → {target_context} ({result}): {rationale}"
        ).strip()
    # Unknown kind (e.g., a future pattern not in the dispatch table) —
    # fall back to a stable repr so the frontend still has something to
    # render.
    return str(payload)


def _render_calibration_before(patch: Any, brief_raw: dict) -> str:
    """Render the brief's current state for the patch's target.

    Empty string when the section is unset or has no entry matching
    the patch's capability area — this is the common case for an
    "add" patch (the area isn't in the brief yet). For
    depth_distinction and prose-style modifications, surface the
    existing text so the recruiter sees exactly what would change.
    """

    from market_intelligence.calibration_to_brief import (
        PATCH_KIND_CALIBRATION_EXAMPLES,
        PATCH_KIND_DEPTH_DISTINCTION,
        PATCH_KIND_NON_FIT_PATTERN,
    )

    if patch.kind == PATCH_KIND_NON_FIT_PATTERN:
        existing = brief_raw.get("non_fit_patterns") if isinstance(brief_raw, dict) else None
        if isinstance(existing, list):
            for entry in existing:
                if isinstance(entry, dict):
                    label = entry.get("label", "")
                    if isinstance(label, str) and label == patch.capability_area:
                        return f"{label}\n{entry.get('description', '')}".strip()
        return ""
    if patch.kind == PATCH_KIND_DEPTH_DISTINCTION:
        depth = brief_raw.get("depth_distinction") if isinstance(brief_raw, dict) else None
        if isinstance(depth, dict):
            existing = depth.get("edge_case_guidance", "")
            if isinstance(existing, str):
                return existing
        return ""
    if patch.kind == PATCH_KIND_CALIBRATION_EXAMPLES:
        existing = (
            brief_raw.get("transferability_examples")
            if isinstance(brief_raw, dict)
            else None
        )
        if isinstance(existing, list):
            for entry in existing:
                if isinstance(entry, dict):
                    src = entry.get("source_context", "")
                    if isinstance(src, str) and src == patch.capability_area:
                        return (
                            f"{src} → {entry.get('target_context', '')} "
                            f"({entry.get('result', '')}): "
                            f"{entry.get('rationale', '')}"
                        ).strip()
        return ""
    return ""


# ---------------------------------------------------------------------------
# P3.6: facial-calibration drift → Gate-2 hunk
# ---------------------------------------------------------------------------
#
# The brief authors an expected_yes_rate_low/high band at preflight time
# (shared/preflight_v2.py:87-88); nothing ever fed the observed facial
# yes-rate back into it. market_intelligence.engine._build_artifact now
# computes facial_calibration_observed (actual vs. authored band,
# consecutive_out_of_band_runs) on every ingestion. When that counter
# reaches 2, this proposes ONE recalibration hunk through the same
# Gate-2 path as every other brief mutation — it never writes the brief
# directly.

_FACIAL_CALIBRATION_DRIFT_CONSECUTIVE_THRESHOLD = 2
_FACIAL_CALIBRATION_BAND_FLOOR = 0.05
_FACIAL_CALIBRATION_BAND_CEILING = 0.95


def _facial_calibration_drift_propose_hunks(
    *,
    artifact_dict: dict,
    brief_raw: dict,
) -> list[dict]:
    """Propose a recalibrated facial expected_yes_rate band on sustained drift.

    No-ops (returns ``[]``) when:

    - the artifact carries fewer than 2 consecutive out-of-band runs
      (``facial_calibration_observed.consecutive_out_of_band_runs``);
    - the brief has no ``facial_calibration`` section to patch;
    - the observed block is missing the actual rate (e.g. carried
      forward from a prior no-verdicts/band-not-authored run).

    Proposal formula (deterministic, conservative): recenter the
    authored band on the observed actual_yes_rate, preserving the
    authored band width, clamped to [0.05, 0.95], rounded to 2dp. This
    is a Gate-2 proposal only — the recruiter must approve it like any
    other hunk; the brief is never auto-written.
    """

    observed = artifact_dict.get("facial_calibration_observed") if isinstance(artifact_dict, dict) else None
    if not isinstance(observed, dict):
        return []
    consecutive = int(observed.get("consecutive_out_of_band_runs", 0) or 0)
    if consecutive < _FACIAL_CALIBRATION_DRIFT_CONSECUTIVE_THRESHOLD:
        return []

    fc_section = brief_raw.get("facial_calibration") if isinstance(brief_raw, dict) else None
    if not isinstance(fc_section, dict) or not fc_section:
        return []

    authored_low = fc_section.get("expected_yes_rate_low")
    authored_high = fc_section.get("expected_yes_rate_high")
    actual = observed.get("actual_yes_rate")
    if not isinstance(authored_low, (int, float)) or not isinstance(authored_high, (int, float)):
        return []
    if not isinstance(actual, (int, float)):
        return []

    half_width = (authored_high - authored_low) / 2.0
    new_low = round(
        min(max(actual - half_width, _FACIAL_CALIBRATION_BAND_FLOOR), _FACIAL_CALIBRATION_BAND_CEILING),
        2,
    )
    new_high = round(
        min(max(actual + half_width, _FACIAL_CALIBRATION_BAND_FLOOR), _FACIAL_CALIBRATION_BAND_CEILING),
        2,
    )
    if new_low > new_high:
        new_low, new_high = new_high, new_low

    before = (
        f"expected_yes_rate_low: {authored_low}\nexpected_yes_rate_high: {authored_high}"
    )
    after = f"expected_yes_rate_low: {new_low}\nexpected_yes_rate_high: {new_high}"
    rationale = (
        f"Observed facial yes-rate ({actual:.2f}) has drifted outside the authored "
        f"band [{authored_low}, {authored_high}] for {consecutive} consecutive runs. "
        "Proposed band recenters on the observed rate, preserving the authored width."
    )
    return [
        {
            "hunk_id": "facial-calibration-drift-1",
            "section": "facial_calibration",
            "kind": "facial_yes_rate_band",
            "label": "Recalibrate facial expected yes-rate band",
            "before": before,
            "after": after,
            "rationale": rationale,
            "confidence": _CALIBRATION_PATCH_CONFIDENCE,
            "default_approved": False,
            "target_field": "facial_calibration",
        }
    ]


def _hunk_before_and_kind(
    brief_raw: dict, section: str, proposal: str
) -> tuple[Any, str]:
    """Return (before_value, kind) for a recommendation against a section.

    For list-shaped brief sections (``additional_search_terms``,
    ``employer_signal_rules``, ``search_priorities``), the kind is
    always ``add`` and ``before`` is ``None`` (we're appending).
    For string-shaped sections (``instructions``, ``notes``), if the
    section already has content we treat it as ``modify`` (the
    proposal extends or replaces the prose); otherwise ``add``.
    """

    LIST_SECTIONS = {"additional_search_terms", "employer_signal_rules", "search_priorities"}
    if section in LIST_SECTIONS:
        return None, "add"
    existing = brief_raw.get(section)
    if isinstance(existing, str) and existing.strip():
        return existing, "modify"
    return None, "add"


def _hunk_label(section: str, kind: str, proposal: str) -> str:
    section_labels = {
        "additional_search_terms": "additional search terms",
        "employer_signal_rules": "employer signal rule",
        "search_priorities": "search priority",
        "instructions": "search instructions",
        "notes": "brief notes",
    }
    label = section_labels.get(section, section.replace("_", " "))
    verb = "Add to" if kind == "add" else "Refine"
    short = _truncate(proposal, max_len=80)
    return f"{verb} {label} — {short}"


def _coerce_confidence(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, f))


# P3.6 hardening (FIX 1): the facial-calibration recalibration hunk built by
# _facial_calibration_drift_propose_hunks targets the "facial_calibration"
# section, which is a DICT in the raw brief -- not a list or prose section.
# Its `after` is a fixed two-line string ("expected_yes_rate_low: X\n
# expected_yes_rate_high: Y") kept as a string (not a structured payload)
# because the Gate-2 hunk card renders/edits `after` as text. This regex is
# the strict, narrow parser for that exact shape; anything else is treated
# as malformed and falls through to the P3.7 guard, which refuses rather
# than corrupts.
_FACIAL_YES_RATE_BAND_AFTER_RE = re.compile(
    r"\Aexpected_yes_rate_low:\s*([0-9]+(?:\.[0-9]+)?)\s*\n"
    r"expected_yes_rate_high:\s*([0-9]+(?:\.[0-9]+)?)\s*\Z"
)


def _parse_facial_yes_rate_band_after(after: str) -> tuple[float, float] | None:
    """Parse+validate the facial_yes_rate_band hunk's ``after`` text.

    Returns ``(low, high)`` only for the exact two-line format with
    ``0.0 <= low < high <= 1.0``. Any deviation (extra lines, non-float
    values, an inverted/degenerate/out-of-range band) returns ``None`` so
    the caller falls through to the existing StructuredSectionHunkError
    refusal instead of writing something wrong.
    """

    match = _FACIAL_YES_RATE_BAND_AFTER_RE.match(after.strip())
    if not match:
        return None
    try:
        low = float(match.group(1))
        high = float(match.group(2))
    except ValueError:
        return None
    if not (0.0 <= low < high <= 1.0):
        return None
    return low, high


def _apply_hunk_to_brief(brief: dict, hunk: dict) -> dict:
    """Apply one accepted hunk to the brief, returning the updated dict.

    Pure: does not mutate the input. Skips hunks whose ``after`` is
    empty after normalization.

    MERGE CONTRACT (mirrors
    cloris/frontend/src/components/RefreshBrief.svelte:buildMergedV2):

    - List sections (additional_search_terms, employer_signal_rules,
      search_priorities): dedupe-append. Compare incoming value to
      existing list entries case-insensitively (using ``_normalize_text``
      collapse for whitespace); skip if already present, append if new.
    - Prose sections (instructions, notes): append-with-newline. If
      existing prose is non-empty, append "\\n\\n" + new prose; else
      replace with new prose.
    - facial_calibration (P3.6 recalibration hunk, kind
      "facial_yes_rate_band"): structured merge — parse the fixed
      two-line ``after`` text and update only expected_yes_rate_low/high
      on a copy of the existing dict, preserving sibling keys. This hunk
      kind is Gate-2-only: it never reaches RefreshBrief's
      ``buildMergedBriefV2`` (confirmed — that function has no branch
      for "facial_calibration"), so it is exempt from the lockstep
      requirement below.
    - Other sections (legacy or future): structural replace.

    Drift between this and ``buildMergedV2`` (TS) produces silently-
    different brief writes across Reflection and RefreshBrief. Keep
    them in lockstep — when one moves, the other moves too.
    """

    section = hunk.get("section")
    after = hunk.get("after")
    kind = hunk.get("kind")
    if not section or not isinstance(after, str) or not after.strip():
        return brief
    next_brief = dict(brief)
    LIST_SECTIONS = {"additional_search_terms", "employer_signal_rules", "search_priorities"}
    if section in LIST_SECTIONS:
        existing = list(next_brief.get(section) or [])
        # De-dupe on normalized text so re-running a hunk doesn't
        # double-write.
        normalized_existing = {
            _normalize_text(item).lower() for item in existing if isinstance(item, str)
        }
        normalized_after = _normalize_text(after).lower()
        if normalized_after not in normalized_existing:
            existing.append(after.strip())
        next_brief[section] = existing
        return next_brief
    if section == "facial_calibration" and kind == "facial_yes_rate_band":
        # P3.6 hardening (FIX 1): must run BEFORE the P3.7 guard below —
        # facial_calibration is a dict section, so the guard would
        # otherwise refuse every recalibration hunk unconditionally,
        # making the loop's write side inert even when a recruiter
        # approves it at Gate 2. On any parse/validation failure, fall
        # through to that guard: refuse, never corrupt.
        parsed = _parse_facial_yes_rate_band_after(after)
        existing_fc = next_brief.get(section)
        if parsed is not None and isinstance(existing_fc, dict):
            low, high = parsed
            updated_fc = dict(existing_fc)
            updated_fc["expected_yes_rate_low"] = low
            updated_fc["expected_yes_rate_high"] = high
            next_brief[section] = updated_fc
            return next_brief
        # else: malformed `after`, or no structured dict to merge into —
        # fall through to the P3.7 guard below.
    if kind == "add" or section in {"instructions", "notes"}:
        if section == "instructions" or section == "notes":
            existing = next_brief.get(section)
            if isinstance(existing, str) and existing.strip():
                next_brief[section] = existing.rstrip() + "\n\n" + after.strip()
            else:
                next_brief[section] = after.strip()
            return next_brief
    # P3.7: the fallthrough replace must never corrupt a structured section
    # (e.g. non_fit_patterns, a list of objects) with a prose string. Refuse
    # with a typed error — surfaced at Gate 2 (the commit endpoint's 422
    # path) instead of silently writing a broken brief.
    existing_value = next_brief.get(section)
    if isinstance(existing_value, (list, dict)):
        raise StructuredSectionHunkError(
            f"hunk for section '{section}' would replace a structured "
            f"{type(existing_value).__name__} value with a plain string; "
            "refusing to write"
        )
    next_brief[section] = after.strip()
    return next_brief


# ---------------------------------------------------------------------------
# PlannerResult <-> dict round-trip
# ---------------------------------------------------------------------------


def _planner_result_from_dict(payload: dict) -> PlannerResult:
    """Reconstruct a PlannerResult from its to_dict() output.

    PlannerResult doesn't ship a from_dict, but its fields are all
    JSON-trivial (strings, lists, dicts, bools, optional float). This
    helper handles the round-trip so phases after PLAN can re-derive
    the typed object.
    """

    return PlannerResult(
        planner_summary=str(payload.get("planner_summary") or ""),
        active_hypotheses=list(payload.get("active_hypotheses") or []),
        resolved_hypotheses=list(payload.get("resolved_hypotheses") or []),
        open_unknowns=list(payload.get("open_unknowns") or []),
        research_backlog=list(payload.get("research_backlog") or []),
        update_sections=list(payload.get("update_sections") or []),
        confidence_ceiling_by_section=dict(
            payload.get("confidence_ceiling_by_section") or {}
        ),
        should_collect_external_research=bool(
            payload.get("should_collect_external_research") or False
        ),
        external_research_focus=list(payload.get("external_research_focus") or []),
        should_collect_edge_case_research=bool(
            payload.get("should_collect_edge_case_research") or False
        ),
        edge_case_research_reasoning=str(
            payload.get("edge_case_research_reasoning") or ""
        ),
        edge_case_research_focus=list(payload.get("edge_case_research_focus") or []),
        edge_case_confidence_ceiling=payload.get("edge_case_confidence_ceiling"),
    )
