"""LinkedIn-specific runtime-state bridge and legacy import helpers."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from linkedin.search_intelligence import (
    LinkedInExperimentState,
    bootstrap_experiment_state,
    reset_experiment_state,
)
from shared.runtime_state.linkedin_artifacts import (
    load_linkedin_history,
    load_linkedin_progress,
    load_linkedin_search_memory,
    rebuild_linkedin_artifacts,
)
from shared.runtime_state.linkedin_experiment_state import load_linkedin_experiment_states
from shared.runtime_state.linkedin_legacy_import import (
    LinkedInLegacyImportContext,
    import_linkedin_legacy_state,
)
from shared.runtime_state.linkedin_progress_sync import sync_linkedin_progress
from shared.runtime_state.linkedin_run_start import start_or_resume_linkedin_run
from shared.runtime_state.store import LINKEDIN_STRING_KIND, RuntimeStateStore
from shared.schemas import CandidateSnippet, OpusDecision, Progress, SearchString

SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}


class LinkedInRuntimeStateBridge:
    """Keeps LinkedIn runtime semantics DB-authoritative while preserving projections."""

    def __init__(
        self,
        *,
        store: RuntimeStateStore,
        output_dir: str | Path,
        brief_id: str,
        brief_name: str,
        brief_path: str | None = None,
    ):
        self.store = store
        self.output_dir = Path(output_dir)
        self.brief_id = brief_id
        self.brief_name = brief_name
        # Phase 3: brief_path is optional so legacy callers (older tests
        # that construct the bridge directly) keep working without an
        # invasive refactor. When set, every start_or_resume_run call
        # computes the brief identity once and forwards it to start_run
        # so runs.brief_path_at_launch / brief_content_hash /
        # brief_snapshot_json get populated.
        self.brief_path = brief_path
        self.progress_path = self.output_dir / "progress.json"
        self.history_path = self.output_dir / f"candidate_history-{brief_id}.jsonl"
        self.search_memory_path = self.output_dir / f"search_memory-{brief_id}.json"
        self.snippets_path = self.output_dir / "snippets.jsonl"
        self.facial_path = self.output_dir / "facial_judgments.jsonl"
        self.profiles_path = self.output_dir / "profile_summaries.jsonl"
        self.final_path = self.output_dir / "final_judgments.jsonl"
        from shared.execution import CandidateExecutionEngine

        self._execution_engine = CandidateExecutionEngine(
            store=self.store,
            output_dir=str(self.output_dir),
            brief_id=self.brief_id,
            source="linkedin",
        )

    def has_runtime_state(self) -> bool:
        latest_run = self.store.get_latest_run(source="linkedin", brief_id=self.brief_id)
        return bool(latest_run or self.store.has_candidates(source="linkedin", brief_id=self.brief_id))

    def has_legacy_state(self) -> bool:
        return bool(self._read_legacy_progress() or self.history_path.exists() or self.search_memory_path.exists())

    def start_or_resume_run(
        self,
        *,
        resume: bool,
        initial_progress: Progress | None = None,
        experiment_states: dict[int, LinkedInExperimentState] | None = None,
    ) -> tuple[int, Progress]:
        self.store.reconcile_open_attempts(source="linkedin", brief_id=self.brief_id)
        self.store.reconcile_pending_side_effects(source="linkedin", brief_id=self.brief_id)
        # Phase 3: compute brief identity at run-start so it pins the
        # exact content the orchestrator was about to execute against.
        # If brief_path wasn't passed at construction, identity is None
        # and the run row falls back to legacy NULL columns.
        from shared.brief_identity import compute_brief_identity

        brief_identity = (
            compute_brief_identity(self.brief_path) if self.brief_path else None
        )
        return start_or_resume_linkedin_run(
            store=self.store,
            output_dir=self.output_dir,
            brief_id=self.brief_id,
            brief_name=self.brief_name,
            resume=resume,
            initial_progress=initial_progress,
            experiment_states=experiment_states,
            legacy_state_exists=self._legacy_state_exists(),
            import_legacy_state=self.import_legacy_state,
            sync_progress=self.sync_progress,
            rebuild_artifacts=self.rebuild_artifacts,
            brief_identity=dict(brief_identity) if brief_identity else None,
        )

    def sync_progress(
        self,
        run_id: int,
        progress: Progress,
        *,
        experiment_states: dict[int, LinkedInExperimentState] | None = None,
        timings: dict[str, float] | None = None,
        validate_status: bool = True,
    ) -> None:
        # Roll up per-lane LLM cost from the run's usage JSONL (written by
        # ``record_llm_usage`` during judge calls) so the work-unit
        # ``metrics_json`` carries a real ``cost_usd`` for ``lane_metrics``
        # to read. Filename mirrors the orchestrator's ``llm_usage_session``
        # sink (``linkedin/orchestrator.py``); fail-soft on absence.
        from shared.runtime_state.linkedin_progress_sync import lane_cost_from_usage_log

        lane_cost_started = time.monotonic()
        try:
            lane_cost_usd = lane_cost_from_usage_log(
                self.output_dir / "token-cost-log.jsonl"
            )
        finally:
            if timings is not None:
                timings["lane_cost_reparse_ms"] = round(
                    (time.monotonic() - lane_cost_started) * 1000.0,
                    3,
                )
        sync_kwargs = dict(
            store=self.store,
            run_id=run_id,
            brief_id=self.brief_id,
            progress=progress,
            experiment_states=experiment_states,
            rebuild_artifacts=self.rebuild_artifacts,
            work_unit_metrics=self._work_unit_metrics,
            lane_cost_usd=lane_cost_usd,
            validate_status=validate_status,
        )
        if timings is not None:
            sync_kwargs["timings"] = timings
        sync_linkedin_progress(**sync_kwargs)

    def load_progress(self, run_id: int) -> Progress:
        return load_linkedin_progress(self.store, run_id)

    def load_experiment_states(
        self,
        run_id: int,
        *,
        progress: Progress | None = None,
    ) -> dict[int, LinkedInExperimentState]:
        return load_linkedin_experiment_states(
            store=self.store,
            run_id=run_id,
            progress=progress,
        )

    def load_search_memory(self) -> dict:
        return load_linkedin_search_memory(self.store, brief_id=self.brief_id)

    def load_history(self) -> tuple[set[str], dict[str, str], set[str]]:
        return load_linkedin_history(
            self.store,
            brief_id=self.brief_id,
            save_decisions=SAVE_DECISIONS,
        )

    def record_missing_identity(
        self,
        *,
        run_id: int,
        search_string: SearchString,
        snippet: CandidateSnippet,
        reason: str = "missing_profile_url",
    ) -> None:
        work_unit_id = self.store.get_work_unit_id(
            run_id,
            kind=LINKEDIN_STRING_KIND,
            source_unit_id=str(search_string.id),
        )
        self.store.record_event(
            run_id=run_id,
            work_unit_id=work_unit_id,
            event_type="linkedin_missing_identity",
            payload={
                "reason": reason,
                "candidate_name": snippet.name,
                "source_string_id": snippet.source_string_id,
                "page": snippet.page,
                "result_rank": snippet.result_rank,
            },
        )

    def record_snippet_extracted(
        self,
        *,
        run_id: int,
        search_string: SearchString,
        snippet: CandidateSnippet,
    ) -> int | None:
        if not snippet.profile_url:
            self.record_missing_identity(run_id=run_id, search_string=search_string, snippet=snippet)
            return None
        envelope = self._execution_engine.envelope(
            source="linkedin",
            brief_id=self.brief_id,
            run_id=run_id,
            work_unit_kind=LINKEDIN_STRING_KIND,
            work_unit_source_id=str(search_string.id),
            identity_key=snippet.profile_url,
            display_name=snippet.name,
            profile_url=snippet.profile_url,
            snippet=snippet,
            source_cursor=self._cursor(search_string, snippet),
        )
        payload = {
            "cursor": envelope.source_cursor,
            "snippet": snippet.to_dict(),
        }
        self._execution_engine.runtime.record_discovery(
            envelope,
            payload=payload["cursor"],
        )
        return self._execution_engine.runtime.record_snippet_extracted(
            envelope,
            payload=payload,
        )

    def start_stage_attempt(
        self,
        *,
        run_id: int,
        search_string: SearchString,
        snippet: CandidateSnippet,
        stage: str,
        payload: dict | None = None,
    ) -> int | None:
        if not snippet.profile_url:
            return None
        attempt_payload = {
            "cursor": self._cursor(search_string, snippet),
            "snippet": snippet.to_dict(),
        }
        if payload:
            attempt_payload.update(payload)
        envelope = self._execution_engine.envelope(
            source="linkedin",
            brief_id=self.brief_id,
            run_id=run_id,
            work_unit_kind=LINKEDIN_STRING_KIND,
            work_unit_source_id=str(search_string.id),
            identity_key=snippet.profile_url,
            display_name=snippet.name,
            profile_url=snippet.profile_url,
            snippet=snippet,
            source_cursor=attempt_payload["cursor"],
        )
        return self._execution_engine.runtime.start_stage(
            envelope,
            stage=stage,
            payload=attempt_payload,
        )

    def finish_stage_success(
        self,
        *,
        run_id: int,
        attempt_id: int | None,
        stage: str,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        profile_summary: CandidateProfileSummary | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        if not attempt_id:
            return
        envelope = self._execution_engine.envelope(
            source="linkedin",
            brief_id=self.brief_id,
            run_id=run_id,
            work_unit_kind=LINKEDIN_STRING_KIND,
            work_unit_source_id=str(snippet.source_string_id),
            identity_key=snippet.profile_url,
            display_name=snippet.name,
            profile_url=snippet.profile_url,
            snippet=snippet,
            source_cursor=self._cursor(search_string=SearchString(id=snippet.source_string_id, name=snippet.source_string_name, boolean=""), snippet=snippet),
        )
        # P4: bridge merges any caller-supplied extra_payload (e.g.
        # ``{"lane": {"lane_id": ..., "lane_name": ...}}`` for bounded
        # non-save review outcomes) on top of the bridge's own keys.
        # Caller keys win on collision so review-time lane attribution
        # can override anything the bridge would default; pre-P4
        # callers pass ``None`` and the merged dict is identical.
        merged_payload: dict[str, Any] = {
            "source_string_id": snippet.source_string_id,
            "timestamp": self._timestamp_from_decision_payload(decision),
        }
        if extra_payload:
            merged_payload.update(extra_payload)
        self._execution_engine.runtime.finish_stage_success(
            attempt_id=attempt_id,
            envelope=envelope,
            stage=stage,
            decision=decision,
            extra_payload=merged_payload,
            profile_summary=profile_summary,
        )

    def finish_stage_failure(
        self,
        *,
        run_id: int,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        error: Exception,
        stage: str = "full",
        payload: dict | None = None,
    ) -> None:
        if not attempt_id:
            return
        envelope = self._execution_engine.envelope(
            source="linkedin",
            brief_id=self.brief_id,
            run_id=run_id,
            work_unit_kind=LINKEDIN_STRING_KIND,
            work_unit_source_id=str(snippet.source_string_id),
            identity_key=snippet.profile_url,
            display_name=snippet.name,
            profile_url=snippet.profile_url,
            snippet=snippet,
            source_cursor=self._cursor(SearchString(id=snippet.source_string_id, name=snippet.source_string_name, boolean=""), snippet),
        )
        self._execution_engine.runtime.finish_stage_failure(
            attempt_id=attempt_id,
            envelope=envelope,
            stage=stage,
            error_or_failure_decision=error,
            extra_payload=payload,
        )

    def finish_failure_decision(
        self,
        *,
        run_id: int,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        payload: dict | None = None,
    ) -> None:
        if not attempt_id:
            return
        envelope = self._execution_engine.envelope(
            source="linkedin",
            brief_id=self.brief_id,
            run_id=run_id,
            work_unit_kind=LINKEDIN_STRING_KIND,
            work_unit_source_id=str(snippet.source_string_id),
            identity_key=snippet.profile_url,
            display_name=snippet.name,
            profile_url=snippet.profile_url,
            snippet=snippet,
            source_cursor=self._cursor(SearchString(id=snippet.source_string_id, name=snippet.source_string_name, boolean=""), snippet),
        )
        self._execution_engine.runtime.finish_stage_failure(
            attempt_id=attempt_id,
            envelope=envelope,
            stage=decision.stage,
            error_or_failure_decision=decision,
            extra_payload=payload,
        )

    def record_side_effect_result(
        self,
        *,
        run_id: int,
        search_string: SearchString,
        snippet: CandidateSnippet,
        attempt_id: int | None,
        effect_type: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        envelope = self._execution_engine.envelope(
            source="linkedin",
            brief_id=self.brief_id,
            run_id=run_id,
            work_unit_kind=LINKEDIN_STRING_KIND,
            work_unit_source_id=str(search_string.id),
            identity_key=snippet.profile_url,
            display_name=snippet.name,
            profile_url=snippet.profile_url,
            snippet=snippet,
            source_cursor=self._cursor(search_string, snippet),
        )
        self._execution_engine.runtime.record_side_effect_result(
            envelope=envelope,
            attempt_id=attempt_id,
            effect_type=effect_type,
            status=status,
            payload=payload,
        )

    def begin_candidate_side_effect(
        self,
        *,
        run_id: int,
        search_string: SearchString,
        snippet: CandidateSnippet,
        attempt_id: int | None,
        effect_type: str,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        envelope = self._execution_engine.envelope(
            source="linkedin",
            brief_id=self.brief_id,
            run_id=run_id,
            work_unit_kind=LINKEDIN_STRING_KIND,
            work_unit_source_id=str(search_string.id),
            identity_key=snippet.profile_url,
            display_name=snippet.name,
            profile_url=snippet.profile_url,
            snippet=snippet,
            source_cursor=self._cursor(search_string, snippet),
        )
        return self._execution_engine.runtime.begin_candidate_side_effect(
            envelope=envelope,
            attempt_id=attempt_id,
            effect_type=effect_type,
            idempotency_key=idempotency_key,
            payload=payload,
        )

    def complete_candidate_side_effect(
        self,
        *,
        side_effect_id: int,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._execution_engine.runtime.complete_candidate_side_effect(
            side_effect_id=side_effect_id,
            status=status,
            payload=payload,
        )

    def save_side_effect_health(self, run_id: int) -> dict[str, Any]:
        """Aggregate the linkedin_save ledger into run-report save health (P1.3).

        Rows are attributed to the run whose ``begin_candidate_side_effect``
        last touched them (the ledger row's run_id), so a retry that lands
        in a later run is that run's success. ``retried_from_prior`` counts
        rows this run touched whose attempt_count > 1 — their earlier
        attempts happened in a prior run or earlier in this one.
        """

        rows = self.store.list_candidate_side_effects(
            source="linkedin", brief_id=self.brief_id
        )
        health: dict[str, Any] = {
            "attempted": 0,
            "succeeded": 0,
            "already_present": 0,
            "failed": 0,
            "failed_permanent": 0,
            "interrupted": 0,
            "retried_from_prior": 0,
            "failed_by_reason": {},
            "failure_rate": 0.0,
        }
        for row in rows:
            if str(row.get("effect_type") or "") != "linkedin_save":
                continue
            if row.get("run_id") != run_id:
                continue
            status = str(row.get("status") or "")
            try:
                payload = json.loads(row.get("payload_json") or "{}")
            except (TypeError, ValueError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            health["attempted"] += 1
            if int(row.get("attempt_count") or 1) > 1:
                health["retried_from_prior"] += 1
            if status == "succeeded":
                health["succeeded"] += 1
                if payload.get("already_present"):
                    health["already_present"] += 1
            elif status == "failed":
                health["failed"] += 1
                reason = str(payload.get("failure_reason") or "unknown")
                health["failed_by_reason"][reason] = (
                    health["failed_by_reason"].get(reason, 0) + 1
                )
            elif status == "failed_permanent":
                health["failed_permanent"] += 1
                reason = str(payload.get("failure_reason") or "attempts_exhausted")
                health["failed_by_reason"][reason] = (
                    health["failed_by_reason"].get(reason, 0) + 1
                )
            elif status == "interrupted":
                health["interrupted"] += 1
        failures = health["failed"] + health["failed_permanent"]
        if health["attempted"]:
            health["failure_rate"] = round(failures / health["attempted"], 4)
        return health

    def rebuild_artifacts(self, run_id: int) -> None:
        rebuild_linkedin_artifacts(
            self.store,
            run_id=run_id,
            output_dir=self.output_dir,
        )

    def record_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any] | None = None,
        run_id: int | None = None,
        work_unit_id: int | None = None,
    ) -> None:
        self.store.record_event(
            run_id=run_id,
            work_unit_id=work_unit_id,
            event_type=event_type,
            payload=payload,
        )

    def import_legacy_state(self, run_id: int) -> None:
        import_linkedin_legacy_state(
            LinkedInLegacyImportContext(
                store=self.store,
                run_id=run_id,
                brief_id=self.brief_id,
                progress_path=self.progress_path,
                history_path=self.history_path,
                search_memory_path=self.search_memory_path,
                snippets_path=self.snippets_path,
                facial_path=self.facial_path,
                profiles_path=self.profiles_path,
                final_path=self.final_path,
                read_legacy_progress=self._read_legacy_progress,
                sync_progress=self.sync_progress,
                search_string_for_id=self._search_string_for_id,
                record_snippet_extracted=self.record_snippet_extracted,
                start_stage_attempt=self.start_stage_attempt,
                finish_failure_decision=self.finish_failure_decision,
                finish_stage_success=self.finish_stage_success,
                save_decisions=SAVE_DECISIONS,
            )
        )

    def restart_string(self, *, run_id: int, progress: Progress, string_id: int) -> None:
        target = next((search_string for search_string in progress.strings if search_string.id == string_id), None)
        if not target:
            return
        # Reset judgment-funnel counters before the direct work-unit rewrite,
        # not only before the trailing sync. If the process dies between those
        # writes, canonical SQLite must already describe the restarted string.
        target.facial_yes_count = 0
        target.facial_no_count = 0
        target.facial_borderline_count = 0
        target.full_reviewed_count = 0
        target.full_outreach_count = 0
        target.full_review_count = 0
        target.full_reject_count = 0
        reset_state: LinkedInExperimentState | None = None
        work_unit = self.store.get_work_unit_by_source_id(
            run_id,
            kind=LINKEDIN_STRING_KIND,
            source_unit_id=str(string_id),
        )
        attempt_ids: list[int] = []
        candidate_keys_to_clear: set[str] = set()
        candidate_ids: set[int] = set()
        if work_unit:
            ordering_index = int(work_unit["ordering_index"])
            payload = _json_loads(work_unit["payload_json"])
            checkpoint = _json_loads(work_unit["checkpoint_json"])
            work_unit_id = int(work_unit["id"])
            with self.store.connect() as conn:
                candidate_rows = conn.execute(
                    """
                    SELECT id, identity_key, terminal_payload_json, last_work_unit_id
                    FROM candidates
                    WHERE source = 'linkedin' AND brief_id = ?
                    """,
                    (self.brief_id,),
                ).fetchall()
                for row in candidate_rows:
                    terminal_payload = _json_loads(row["terminal_payload_json"])
                    source_string_id = terminal_payload.get("source_string_id")
                    if source_string_id == string_id or row["last_work_unit_id"] == work_unit_id:
                        candidate_ids.add(int(row["id"]))
                        candidate_keys_to_clear.add(str(row["identity_key"]))
                rows = conn.execute(
                    """
                    SELECT ca.id, ca.payload_json, ca.source_cursor_json, c.id AS candidate_id, c.identity_key
                    FROM candidate_attempts ca
                    JOIN candidates c ON c.id = ca.candidate_id
                    WHERE c.source = 'linkedin' AND c.brief_id = ?
                    """,
                    (self.brief_id,),
                ).fetchall()
                for row in rows:
                    # P10 actuate #5: this MUST NOT be named `payload` — the
                    # outer work-unit payload (line 568, restarted below at
                    # `payload.update(...)`) was previously clobbered by
                    # whichever candidate_attempts row's payload happened to
                    # be the last one iterated here, silently leaking that
                    # attempt's keys into the restarted work unit.
                    attempt_payload = _json_loads(row["payload_json"])
                    cursor = _json_loads(row["source_cursor_json"])
                    source_string_id = (
                        attempt_payload.get("source_string_id")
                        or attempt_payload.get("cursor", {}).get("source_string_id")
                        or cursor.get("source_string_id")
                    )
                    if source_string_id == string_id or row["candidate_id"] in candidate_ids:
                        attempt_ids.append(int(row["id"]))
                        candidate_keys_to_clear.add(str(row["identity_key"]))
                        candidate_ids.add(int(row["candidate_id"]))
                for attempt_id in attempt_ids:
                    conn.execute("DELETE FROM candidate_attempts WHERE id = ?", (attempt_id,))
                experiment_state = LinkedInExperimentState.from_dict(checkpoint.get("experiment_state"))
                experiment_state = reset_experiment_state(target, experiment_state)
                reset_state = experiment_state
                experiment_state.apply_shadow(target)
                payload.update(
                    {
                        **target.to_dict(),
                        "status": "queued",
                        "pages_reviewed": 1,
                        "phase": "scout",
                        "saves": [],
                        "notes": "",
                        "refinement_stack": [],
                        "search_intent": experiment_state.intent.to_dict(),
                    }
                )
                for identity_key in candidate_keys_to_clear:
                    self.store.clear_candidate_terminal_state(
                        source="linkedin",
                        brief_id=self.brief_id,
                        identity_key=identity_key,
                        conn=conn,
                    )
                    self.store.invalidate_candidate_side_effects(
                        source="linkedin",
                        brief_id=self.brief_id,
                        identity_key=identity_key,
                        conn=conn,
                    )
                self.store.upsert_work_unit(
                    run_id=run_id,
                    source="linkedin",
                    brief_id=self.brief_id,
                    kind=LINKEDIN_STRING_KIND,
                    source_unit_id=str(string_id),
                    display_name=target.name,
                    ordering_index=ordering_index,
                    status="queued",
                    payload=payload,
                    checkpoint={
                        "pages_reviewed": 1,
                        "duplicates_count": 0,
                        "phase": "scout",
                        "refinement_stack": [],
                        "experiment_state": experiment_state.to_dict(),
                    },
                    metrics={
                        **self._work_unit_metrics(target, pages_reviewed=1, duplicates_count=0, block_generated=0, exhausted=0),
                        "experiment_summary": experiment_state.metrics_summary(),
                        "variant_metrics": experiment_state.metrics_summary().get("variants", {}),
                    },
                    family_key=target.family_key,
                    novelty_bucket=target.novelty_bucket,
                    domain_lane=target.domain_lane,
                    counters={
                        "result_count": 0,
                        "candidates_discovered": 0,
                        "facial_yes_count": 0,
                        "facial_no_count": 0,
                        "facial_borderline_count": 0,
                        "saves_count": 0,
                        "rejected_count": 0,
                    },
                    notes="",
                    conn=conn,
                )
                self.store.record_event(
                    run_id=run_id,
                    work_unit_id=work_unit_id,
                    event_type="linkedin_string_restarted",
                    payload={"string_id": string_id, "candidate_ids_cleared": sorted(candidate_ids)},
                    conn=conn,
                )

        target.status = "queued"
        target.pages_reviewed = 1
        target.phase = "scout"
        target.saves = []
        target.notes = ""
        target.refinement_stack = []
        target.result_count = 0
        if reset_state is not None:
            reset_state.apply_shadow(target)
        if string_id in progress.pending_block_string_ids or progress.pending_block_name == target.block:
            progress.pending_block_name = ""
            progress.pending_block_string_ids = []
            progress.pending_block_ready = False
        if progress.current_string_id == string_id:
            progress.current_string_id = None
            progress.current_page = 0
        self.sync_progress(run_id, progress)

    def _legacy_state_exists(self) -> bool:
        return self.has_legacy_state()

    def _read_legacy_progress(self) -> Progress | None:
        if not self.progress_path.exists():
            return None
        progress = Progress.from_file(str(self.progress_path))
        if progress.brief_name != self.brief_name:
            return None
        return progress

    def _search_string_for_id(self, progress: Progress | None, string_id: int) -> SearchString:
        if progress:
            existing = next((item for item in progress.strings if item.id == string_id), None)
            if existing:
                return existing
        return SearchString(id=string_id, name=f"Imported #{string_id}", boolean="")

    @staticmethod
    def _timestamp_from_decision_payload(decision: OpusDecision) -> str | None:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _cursor(search_string: SearchString, snippet: CandidateSnippet) -> dict[str, Any]:
        return {
            "source_string_id": search_string.id,
            "source_string_name": search_string.name,
            "page": snippet.page,
            "result_rank": snippet.result_rank,
        }

    @staticmethod
    def _work_unit_metrics(
        search_string: SearchString,
        *,
        pages_reviewed: int | None = None,
        duplicates_count: int | None = None,
        block_generated: int | None = None,
        exhausted: int | None = None,
    ) -> dict[str, Any]:
        full_outreach = max(
            search_string.full_outreach_count,
            len(search_string.saves),
        )
        full_reviewed = max(
            search_string.full_reviewed_count,
            full_outreach
            + search_string.full_review_count
            + search_string.full_reject_count,
        )
        return {
            "pages_reviewed": search_string.pages_reviewed if pages_reviewed is None else pages_reviewed,
            "profiles_seen": search_string.result_count,
            "profiles_processed": search_string.candidates_count,
            "facial_yes": search_string.facial_yes_count,
            "facial_no": search_string.facial_no_count,
            "facial_borderline": search_string.facial_borderline_count,
            # C2 (slice 15): borderline is its own counter, not silently absorbed into
            # skip. Subtract borderline from the residual to keep skip math honest.
            "facial_skip": max(
                0,
                search_string.candidates_count
                - search_string.facial_yes_count
                - search_string.facial_no_count
                - search_string.facial_borderline_count,
            ),
            "full_save": len(search_string.saves),
            "full_reviewed": full_reviewed,
            "full_outreach": full_outreach,
            "full_review": search_string.full_review_count,
            "full_reject": search_string.full_reject_count,
            "duplicates_count": search_string.duplicates_count if duplicates_count is None else duplicates_count,
            "exhausted": int(exhausted or 0),
            "block_generated": int(block_generated or 0),
        }


def _json_loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw)
