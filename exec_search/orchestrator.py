"""Runnable Executive Search pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from exec_search.budget import (
    BudgetExhausted,
    DossierSpendTracker,
    predicted_cost_for,
)
from exec_search.evidence_assembly import DossierEvidence, assemble_dossier_evidence
from exec_search.judging import exec_search_full_judge
from exec_search.signals import SignalFailure, SignalRequestContext, SignalResult, known_signal_sources
from exec_search.strategy import form_exec_search_strategy
from market_intelligence.pre_launch import InvestigationFailure, InvestigationPacket, run_pre_launch_investigation
from shared.adaptive import (
    AdaptiveAction,
    AdaptationDecision,
    NoiseMarker,
    ScoutMetrics,
    SignalMarker,
    record_adaptation_decision,
)
from shared.brief_loader import Brief, load_brief
from shared.runtime_state.exec_search import ExecSearchRuntimeStateBridge
from shared.runtime_state.store import RuntimeStateStore
from shared.safety.stop_reasons import RunStopReason
from shared.schemas import CandidateProfileSummary
from shared.storage import log_event


class _BudgetExhaustedStop(Exception):
    """Internal signal — raised inside the candidate loop to break both
    the per-lane and lane loops when the spend cap is hit. Caught at the
    ``ExecSearchPipeline.run`` boundary to drive a clean shutdown that
    marks remaining lanes ``skipped`` and finishes the run with
    ``stop_reason=api_budget_exhausted``.
    """

    def __init__(
        self,
        exhausted: BudgetExhausted,
        exhausted_per_lane: dict[str, int],
    ) -> None:
        super().__init__("budget exhausted")
        self.exhausted = exhausted
        self.exhausted_per_lane = exhausted_per_lane


ExecCandidateDiscoverer = Callable[[dict[str, Any]], list[CandidateProfileSummary]]


@dataclass
class ExecSearchRunStats:
    lanes_total: int = 0
    lanes_completed: int = 0
    candidates_discovered: int = 0
    saves: int = 0
    rejects: int = 0
    insufficient: int = 0
    signal_failures: int = 0
    cost_usd: float = 0.0
    per_lane: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Static map of signal source name → env var that gates availability.
# When the env var is unset the adapter returns
# ``SignalFailure(reason="disabled_no_api_key")`` from ``fetch`` —
# probing here lets the orchestrator skip cost reservation for disabled
# sources so the brief's cap isn't silently burned by reservations that
# never produce evidence.
_SIGNAL_ENV_GATES: dict[str, str] = {
    "news": "NEWSAPI_KEY",
    "crunchbase": "CRUNCHBASE_API_KEY",
    "pitchbook": "PITCHBOOK_API_KEY",
}


def _signal_is_available(source: str) -> bool:
    import os  # noqa: PLC0415 - lazy

    env_var = _SIGNAL_ENV_GATES.get(source)
    if env_var is None:
        # Sources without an explicit env gate (e.g., Perplexity) are
        # assumed available — they handle their own failure modes.
        return True
    return bool((os.environ.get(env_var) or "").strip())


@dataclass
class ExecSearchPipeline:
    brief: Brief
    bridge: ExecSearchRuntimeStateBridge
    investigation_packet: InvestigationPacket | None = None
    candidate_discoverer: ExecCandidateDiscoverer | None = None
    full_llm_caller: Callable[[str, str], str | dict[str, Any]] | None = None
    signal_sources: tuple[str, ...] | None = None
    log_path: Path | None = None
    spend_tracker: DossierSpendTracker | None = None

    def __post_init__(self) -> None:
        if self.log_path is None:
            self.log_path = Path(self.bridge.output_dir) / "run_log.jsonl"
        if self.spend_tracker is None:
            # Honor the brief's recruiter-overridable cap. ``budget.py``
            # documents this as the override path; the previous default
            # silently ignored it.
            cap_usd = float(getattr(self.brief, "dossier_spend_cap_usd", None) or 200.0)
            self.spend_tracker = DossierSpendTracker(cap_usd=cap_usd)

    def run(self, *, run_id: int) -> ExecSearchRunStats:
        log_event(self.log_path, "pipeline_start", mode="full")
        packet_dict = (
            self.investigation_packet.to_dict() if self.investigation_packet else None
        )
        plan = form_exec_search_strategy(
            self.brief,
            investigation_packet=packet_dict,
        )
        lanes = list(plan.generated_strings)
        stats = ExecSearchRunStats(lanes_total=len(lanes))

        finish_status = "completed"
        finish_stop_reason = RunStopReason.NORMAL
        end_status = "ok"

        try:
            index = 0
            while index < len(lanes):
                lane = lanes[index]
                ordering_index = index + 1
                work_unit_id = self.bridge.upsert_lane_work_unit(
                    run_id=run_id,
                    lane=lane,
                    ordering_index=ordering_index,
                    status="in_progress",
                )
                try:
                    per_lane = self._run_one_lane(
                        run_id=run_id,
                        lane=lane,
                        work_unit_id=work_unit_id,
                    )
                except _BudgetExhaustedStop as stop:
                    # The lane was partially processed; capture what landed
                    # before the stop, mark it done, then short-circuit the
                    # remaining lanes as skipped via the post-loop block.
                    stats.lanes_completed += 1
                    stats.cost_usd = self.spend_tracker.accumulated_usd
                    self.bridge.upsert_lane_work_unit(
                        run_id=run_id,
                        lane=lane,
                        ordering_index=ordering_index,
                        status="done",
                        counters=stop.exhausted_per_lane,
                    )
                    stats.per_lane.append(stop.exhausted_per_lane)
                    log_event(
                        self.log_path,
                        "string_complete",
                        string_id=lane.get("id") or ordering_index,
                        **stop.exhausted_per_lane,
                    )
                    # Mark every remaining lane (including any already
                    # planned adaptations) as skipped under the canonical
                    # work-units row so resume / market-intel rollups see
                    # the truth instead of phantom in-progress lanes.
                    for skip_offset, skip_lane in enumerate(
                        lanes[index + 1 :], start=ordering_index + 1
                    ):
                        self.bridge.upsert_lane_work_unit(
                            run_id=run_id,
                            lane=skip_lane,
                            ordering_index=skip_offset,
                            status="skipped",
                        )
                    finish_status = "interrupted"
                    finish_stop_reason = RunStopReason.API_BUDGET_EXHAUSTED
                    end_status = "budget_exhausted"
                    break
                stats.lanes_completed += 1
                stats.candidates_discovered += per_lane["candidates_discovered"]
                stats.saves += per_lane["saves_count"]
                stats.rejects += per_lane["rejected_count"]
                stats.insufficient += per_lane["insufficient_count"]
                stats.signal_failures += per_lane["signal_failures"]
                stats.cost_usd = self.spend_tracker.accumulated_usd
                stats.per_lane.append(per_lane)
                self.bridge.upsert_lane_work_unit(
                    run_id=run_id,
                    lane=lane,
                    ordering_index=ordering_index,
                    status="done",
                    counters=per_lane,
                )
                log_event(
                    self.log_path,
                    "string_complete",
                    string_id=lane.get("id") or ordering_index,
                    **per_lane,
                )
                new_lanes = self._adapt_after_lane(
                    run_id=run_id,
                    lane=lane,
                    per_lane=per_lane,
                    remaining=lanes[index + 1 :],
                    current_ordering_index=ordering_index,
                )
                if new_lanes:
                    lanes[index + 1 : index + 1] = new_lanes
                    stats.lanes_total = len(lanes)
                index += 1
        except Exception as exc:  # noqa: BLE001 - telemetry must capture all failure paths
            finish_status = "error"
            finish_stop_reason = f"error: {type(exc).__name__}"
            end_status = "error"
            log_event(
                self.log_path,
                "pipeline_error",
                error=str(exc),
                error_class=type(exc).__name__,
            )
            raise
        finally:
            log_event(
                self.log_path,
                "pipeline_end",
                status=end_status,
                **{k: v for k, v in stats.as_dict().items() if k != "per_lane"},
            )
            self.bridge.store.finish_run(
                run_id, finish_status, stop_reason=finish_stop_reason
            )
        return stats

    def _run_one_lane(
        self,
        *,
        run_id: int,
        lane: dict[str, Any],
        work_unit_id: int,
    ) -> dict[str, int]:
        candidates = (
            self.candidate_discoverer(lane)
            if self.candidate_discoverer is not None
            else []
        )
        saves = rejects = insufficient = signal_failures = 0
        for candidate in candidates:
            self.bridge.record_discovery(
                run_id=run_id,
                work_unit_id=work_unit_id,
                candidate=candidate,
            )
            self.bridge.record_snippet_extraction(
                run_id=run_id,
                work_unit_id=work_unit_id,
                candidate=candidate,
            )
            evidence = self._assemble_budgeted_evidence(
                run_id=run_id,
                candidate=candidate,
                lane=lane,
            )
            if evidence is None:
                insufficient += 1
                continue
            signal_failures += sum(
                1 for outcome in evidence.signal_outcomes.values() if isinstance(outcome, SignalFailure)
            )
            reserve = self.spend_tracker.reserve(
                source="opus_full_eval",
                cost_usd=predicted_cost_for("opus_full_eval"),
                is_full_eval=True,
            )
            if isinstance(reserve, BudgetExhausted):
                self._record_budget_exhausted(run_id, reserve)
                insufficient += 1
                # The per-candidate Opus reservation is the hard-stop —
                # without it we can't produce dossier saves. Raise to
                # break both loops and trigger the budget-exhausted
                # finish path in ``run``.
                raise _BudgetExhaustedStop(
                    reserve,
                    {
                        "candidates_discovered": len(candidates),
                        "saves_count": saves,
                        "rejected_count": rejects,
                        "insufficient_count": insufficient,
                        "signal_failures": signal_failures,
                    },
                )
            decision = exec_search_full_judge(
                candidate=candidate,
                brief=self.brief,
                dossier_prompt_body=evidence.prompt_body,
                llm_caller=self.full_llm_caller,
            )
            terminal_payload = {
                "dossier_evidence": evidence.prompt_body,
                "signal_outcomes": _signal_outcomes_payload(evidence.signal_outcomes),
                "surface_type": "exec_search_dossier",
                "lane": dict(lane),
            }
            self.bridge.record_full_decision(
                run_id=run_id,
                work_unit_id=work_unit_id,
                candidate=candidate,
                decision=decision,
                terminal_payload=terminal_payload,
            )
            if decision.decision in {"SAVE", "INFERENTIAL_SAVE", "SIGNAL_SAVE"}:
                saves += 1
            else:
                rejects += 1

        return {
            "candidates_discovered": len(candidates),
            "saves_count": saves,
            "rejected_count": rejects,
            "insufficient_count": insufficient,
            "signal_failures": signal_failures,
        }

    def _assemble_budgeted_evidence(
        self,
        *,
        run_id: int,
        candidate: CandidateProfileSummary,
        lane: dict[str, Any],
    ) -> DossierEvidence | None:
        sources = self.signal_sources
        if sources is None:
            sources = known_signal_sources()
        allowed_sources: list[str] = []
        for source in sources:
            # Probe availability before reserving — a disabled provider
            # returns SignalFailure(disabled_no_api_key) regardless of
            # whether we paid for it, so reservation just burns the cap.
            if not _signal_is_available(source):
                continue
            reserve = self.spend_tracker.reserve(
                source=source,
                cost_usd=predicted_cost_for(source),
            )
            if isinstance(reserve, BudgetExhausted):
                self._record_budget_exhausted(run_id, reserve)
                continue
            allowed_sources.append(source)
        context = SignalRequestContext(
            brief_id=self.brief.id or self.brief.role_title,
            trigger_reason="exec_search_dossier",
            identity_hints={"lane": lane},
        )
        return assemble_dossier_evidence(
            candidate=candidate,
            brief=self.brief,
            context=context,
            sources=allowed_sources,
        )

    def _adapt_after_lane(
        self,
        *,
        run_id: int,
        lane: dict[str, Any],
        per_lane: dict[str, int],
        remaining: list[dict[str, Any]],
        current_ordering_index: int,
    ) -> list[dict[str, Any]]:
        action = AdaptiveAction.CONTINUE
        rationale = "Exec-search lane completed; continuing current map."
        signal_markers: list[SignalMarker] = []
        noise_markers: list[NoiseMarker] = []
        new_lanes: list[dict[str, Any]] = []

        if per_lane["saves_count"] > 0:
            action = AdaptiveAction.COMMIT
            rationale = "Lane produced dossier saves; keep company/title thesis active."
            signal_markers.append(
                SignalMarker(kind="dossier_save", label="exec dossier saves", count=per_lane["saves_count"])
            )
        elif per_lane["candidates_discovered"] == 0:
            action = AdaptiveAction.BROADEN
            rationale = "Lane was thin; broaden title/scope hypothesis."
            noise_markers.append(
                NoiseMarker(kind="thin_lane", label="no candidates discovered", count=1)
            )
            if not remaining and not lane.get("adapted_from"):
                adapted = dict(lane)
                adapted["id"] = int(lane.get("id") or 0) + 1000
                adapted["name"] = f"Broadened market map: {lane.get('title') or 'executive'}"
                adapted["lane_type"] = "market_map"
                adapted["company"] = ""
                adapted["adapted_from"] = lane.get("id")
                new_lanes.append(adapted)
        elif per_lane["signal_failures"] > 0:
            action = AdaptiveAction.EXPERIMENT
            rationale = "Public-web signal coverage was partial; keep dossier evaluation but note signal gaps."
            noise_markers.append(
                NoiseMarker(
                    kind="signal_failure",
                    label="public-web signal failures",
                    count=per_lane["signal_failures"],
                )
            )

        # Eagerly upsert adapted lanes as "queued" before the next outer-
        # loop iteration so a crash in this window keeps work_units truthful.
        inserted_work_unit_ids: list[str] = []
        for offset, new_lane in enumerate(new_lanes):
            new_ordering_index = current_ordering_index + 1 + offset
            work_unit_id = self.bridge.upsert_lane_work_unit(
                run_id=run_id,
                lane=new_lane,
                ordering_index=new_ordering_index,
                status="queued",
            )
            inserted_work_unit_ids.append(str(work_unit_id))

        decision = AdaptationDecision(
            source="exec_search",
            action=action,
            lane=str(lane.get("company") or lane.get("lane_type") or "exec_search"),
            rationale=rationale,
            metrics=ScoutMetrics(
                work_units_run=1,
                candidates_discovered=per_lane["candidates_discovered"],
                saves=per_lane["saves_count"],
                rejects=per_lane["rejected_count"],
                insufficient=per_lane["insufficient_count"],
                signal_markers=signal_markers,
                noise_markers=noise_markers,
            ),
            work_unit_kind="exec_search_query",
            work_unit_family=str(lane.get("company") or lane.get("title") or ""),
            inserted_work_units=inserted_work_unit_ids,
            source_payload={"lane": dict(lane), "new_lanes": new_lanes},
        )
        record_adaptation_decision(self.bridge.store, run_id=run_id, decision=decision)
        log_event(self.log_path, "adaptation_decision", **decision.to_dict())
        return new_lanes

    def _record_budget_exhausted(self, run_id: int, exhausted: BudgetExhausted) -> None:
        self.bridge.store.record_event(
            event_type="budget_exhausted",
            run_id=run_id,
            payload=asdict(exhausted),
        )
        log_event(self.log_path, "budget_exhausted", **asdict(exhausted))


def _signal_outcomes_payload(
    outcomes: dict[str, SignalResult | SignalFailure],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, outcome in outcomes.items():
        if isinstance(outcome, SignalResult):
            payload[name] = {
                "status": "ok",
                "section_text": outcome.section_text,
                "citations": list(outcome.citations),
                "raw_payload": dict(outcome.raw_payload or {}),
            }
        else:
            payload[name] = {
                "status": "failed",
                "reason": outcome.reason,
                "detail": outcome.detail,
            }
    return payload


def build_pipeline(
    *,
    brief_path: str | Path,
    state_dir: str | Path,
    resume: bool = False,
    investigate_at_launch: bool = False,
) -> tuple[ExecSearchPipeline, int]:
    brief = load_brief(str(brief_path))
    state_dir_path = Path(state_dir)
    state_dir_path.mkdir(parents=True, exist_ok=True)
    packet: InvestigationPacket | None = None
    if investigate_at_launch:
        # Investigation is a network-bound, brief-mutating side effect.
        # Gate behind an explicit opt-in so the exec-search CLI doesn't
        # silently write market intelligence artifacts onto the brief.
        investigation = run_pre_launch_investigation(
            brief_path=brief_path,
            persist=True,
        )
        if isinstance(investigation, InvestigationPacket):
            packet = investigation
        elif isinstance(investigation, InvestigationFailure):
            log_event(
                state_dir_path / "run_log.jsonl",
                "investigation_failed",
                reason=investigation.reason,
                detail=investigation.detail,
            )
    store = RuntimeStateStore(state_dir_path / "runtime_state.sqlite3")
    bridge = ExecSearchRuntimeStateBridge(
        store=store,
        output_dir=state_dir_path,
        brief_id=brief.id or brief.role_title or Path(brief_path).stem,
        brief_name=brief.role_title or brief.id,
        brief_path=str(brief_path),
    )
    run_id = bridge.start_or_resume_run(resume=resume)
    pipeline = ExecSearchPipeline(brief=brief, bridge=bridge, investigation_packet=packet)
    return pipeline, run_id
