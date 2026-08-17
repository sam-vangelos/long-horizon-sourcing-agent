"""GitHub side-effect service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from github.outreach import generate_outreach
from shared.output_paths import source_exports_root
from shared.execution import SideEffectOutcome
from shared.judger import extract_priority_rank
from shared.storage import append_jsonl, log_event

import github.config as gc

if TYPE_CHECKING:
    from github.orchestrator import GitHubPipeline
    from github.schemas import GitHubCandidate, GitHubProgress, GitHubSearchQuery
    from shared.execution.types import CandidateExecutionEnvelope
    from shared.schemas import OpusDecision


SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}


class GitHubSideEffectsService:
    """Owns GitHub save/outreach/export side effects."""

    def __init__(self, pipeline: "GitHubPipeline"):
        self.pipeline = pipeline

    async def handle_full_decision(
        self,
        *,
        username: str,
        candidate: "GitHubCandidate",
        query: "GitHubSearchQuery",
        progress: "GitHubProgress",
        full_decision: "OpusDecision",
        envelope: "CandidateExecutionEnvelope",
        full_attempt_id: int | None,
    ) -> SideEffectOutcome:
        pipeline = self.pipeline

        if full_decision.decision not in SAVE_DECISIONS:
            pipeline.stats["rejected"] += 1
            progress.candidates_rejected += 1
            pipeline._observer.on_reject(username, candidate, full_decision, query)
            return SideEffectOutcome(
                effect_type="github_candidate_save",
                status="skipped",
                payload={"decision": full_decision.decision},
            )

        pipeline.stats["saved"] += 1
        progress.candidates_saved += 1
        query.saves.append(candidate.user.name or username)

        pipeline._observer.on_save(username, candidate, full_decision, query)
        log_event(
            pipeline.log_path,
            "save",
            username=username,
            name=candidate.user.name,
            confidence=full_decision.confidence,
            decision_path=full_decision.path,
            contact_emails=candidate.contact.emails,
            query_id=query.id,
        )

        outreach_generated = False
        outreach_side_effect = None
        if getattr(pipeline, "_execution_engine", None) and envelope.run_id > 0 and full_attempt_id:
            outreach_side_effect = pipeline._execution_engine.runtime.begin_candidate_side_effect(
                envelope=envelope,
                attempt_id=full_attempt_id,
                effect_type="github_outreach",
                idempotency_key="outreach",
                payload={"decision": full_decision.decision},
            )
        if outreach_side_effect and not outreach_side_effect["should_execute"]:
            pipeline._execution_engine.runtime.record_side_effect_result(
                envelope=envelope,
                attempt_id=full_attempt_id,
                effect_type="github_outreach",
                status="skipped",
                payload={"skip_reason": f"existing_{outreach_side_effect['side_effect']['status']}"},
            )
        else:
            outreach = await generate_outreach(candidate, pipeline.brief_obj, full_decision)
            if outreach and outreach.get("message"):
                candidate.outreach_copy = outreach
                append_jsonl(pipeline.outreach_path, outreach)
                outreach_generated = True
                if outreach_side_effect:
                    pipeline._execution_engine.runtime.complete_candidate_side_effect(
                        side_effect_id=int(outreach_side_effect["side_effect"]["id"]),
                        status="succeeded",
                        payload={"has_message": True},
                    )
                if getattr(pipeline, "_execution_engine", None):
                    pipeline._execution_engine.runtime.record_side_effect_result(
                        envelope=envelope,
                        attempt_id=full_attempt_id,
                        effect_type="github_outreach",
                        status="succeeded",
                        payload={"has_message": True},
                    )
            else:
                pipeline.stats.setdefault("outreach_failures", 0)
                pipeline.stats["outreach_failures"] += 1
                pipeline._observer.on_outreach_failure(username, query)
                if outreach_side_effect:
                    pipeline._execution_engine.runtime.complete_candidate_side_effect(
                        side_effect_id=int(outreach_side_effect["side_effect"]["id"]),
                        status="failed",
                        payload={"has_message": False},
                    )
                if getattr(pipeline, "_execution_engine", None):
                    pipeline._execution_engine.runtime.record_side_effect_result(
                        envelope=envelope,
                        attempt_id=full_attempt_id,
                        effect_type="github_outreach",
                        status="failed",
                        payload={"has_message": False},
                    )

        priority_rank = extract_priority_rank(full_decision.path)
        append_jsonl(
            pipeline.saves_path,
            {
                "username": username,
                "name": candidate.user.name,
                "github_url": candidate.user.profile_url,
                "location": candidate.user.location,
                "bio": candidate.user.bio,
                "company": candidate.user.company,
                "emails": candidate.contact.emails,
                "blog": candidate.user.blog,
                "twitter": candidate.user.twitter_username,
                "decision": full_decision.decision,
                "confidence": full_decision.confidence,
                "decision_path": full_decision.path,
                "priority_rank": priority_rank,
                "rationale": full_decision.rationale,
                "outreach": candidate.outreach_copy if candidate.outreach_copy else None,
                "source_query": query.name,
                "source_channel": query.channel,
                "expansion_seed": query.query if query.channel == "graph_expansion" else None,
            },
        )

        graph_expansion_queued = False
        if full_decision.confidence >= gc.GRAPH_EXPANSION_MIN_CONFIDENCE:
            pipeline._work_unit_service.enqueue_graph_expansion_seed(
                progress=progress,
                username=username,
                reason=full_decision.decision,
                confidence=full_decision.confidence,
                capability_area=full_decision.path,
            )
            graph_expansion_queued = True

        return SideEffectOutcome(
            effect_type="github_candidate_save",
            status="succeeded",
            payload={
                "decision": full_decision.decision,
                "outreach_generated": outreach_generated,
                "graph_expansion_queued": graph_expansion_queued,
            },
        )

    def export_saved_candidates_csv(self) -> SideEffectOutcome | None:
        pipeline = self.pipeline
        if pipeline.stats["saved"] <= 0:
            return None
        try:
            from github.export import export_saved_candidates_csv

            csv_path = export_saved_candidates_csv(
                pipeline.output_dir,
                csv_path=source_exports_root(
                    "github",
                    pipeline.brief_obj.id,
                    output_root=pipeline.output_dir,
                )
                / "saved_candidates.csv",
            )
            pipeline._observer.console.emit_info(f"CSV export: {csv_path}")
            if getattr(pipeline, "_runtime_run_id", None):
                pipeline._runtime_state.record_event(
                    run_id=pipeline._runtime_run_id,
                    event_type="side_effect_result",
                    payload={
                        "effect_type": "github_csv_export",
                        "status": "succeeded",
                        "path": str(csv_path),
                    },
                )
            return SideEffectOutcome(
                effect_type="github_csv_export",
                status="succeeded",
                payload={"path": str(csv_path)},
            )
        except Exception as exc:
            pipeline._observer.console.emit_warn(f"CSV export failed: {exc}")
            if getattr(pipeline, "_runtime_run_id", None):
                pipeline._runtime_state.record_event(
                    run_id=pipeline._runtime_run_id,
                    event_type="side_effect_result",
                    payload={
                        "effect_type": "github_csv_export",
                        "status": "failed",
                        "error": str(exc),
                    },
                )
            return SideEffectOutcome(
                effect_type="github_csv_export",
                status="failed",
                payload={"error": str(exc)},
            )
