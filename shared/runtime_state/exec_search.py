"""Exec Search runtime-state bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.runtime_state.store import EXEC_SEARCH_QUERY_KIND, RuntimeStateStore
from shared.schemas import CandidateProfileSummary, OpusDecision


def _resolve_recruiter_id() -> int | None:
    """Resolve the acting recruiter for a run-start, fail-soft to None.

    reopen Stage 2 (R5a-3): stamps ``runs.recruiter_id`` so the read-only
    taste aggregator (R5a-4) can attribute adaptation decisions. The
    resolver is the single auth seam (``shared.recruiter_context``); we
    catch broadly because a run launch must never die on recruiter
    resolution — a None recruiter_id is a clean "unknown" (the aggregator
    skips it), whereas a raised exception here would abort the run.
    """

    try:
        from shared.recruiter_context import get_current_recruiter_id

        return get_current_recruiter_id()
    except Exception:  # noqa: BLE001 — resolution must never break a run launch
        return None


class ExecSearchRuntimeStateBridge:
    def __init__(
        self,
        *,
        store: RuntimeStateStore,
        output_dir: str | Path,
        brief_id: str,
        brief_name: str,
        brief_path: str | None = None,
    ) -> None:
        self.store = store
        self.output_dir = Path(output_dir)
        self.brief_id = brief_id
        self.brief_name = brief_name
        self.brief_path = brief_path

    def start_or_resume_run(self, *, resume: bool) -> int:
        self.store.reconcile_open_attempts(source="exec_search", brief_id=self.brief_id)
        self.store.reconcile_pending_side_effects(source="exec_search", brief_id=self.brief_id)
        latest_run = self.store.get_latest_run(source="exec_search", brief_id=self.brief_id)

        from shared.brief_identity import compute_brief_identity

        identity = compute_brief_identity(self.brief_path) if self.brief_path else None
        identity_kwargs: dict[str, Any] = {}
        if identity is not None:
            identity_kwargs = {
                "brief_path_at_launch": identity["brief_path_at_launch"],
                "brief_content_hash": identity["brief_content_hash"],
                "brief_snapshot_json": identity["brief_snapshot_json"],
            }

        recruiter_id = _resolve_recruiter_id()

        if resume and latest_run and self.store.has_work_units(int(latest_run["id"])):
            return self.store.start_run(
                source="exec_search",
                brief_id=self.brief_id,
                output_dir=str(self.output_dir),
                mode="resume",
                resume_state=self.store.get_run_resume_state(int(latest_run["id"])),
                resumed_from_run_id=int(latest_run["id"]),
                clone_work_units_from_run_id=int(latest_run["id"]),
                recruiter_id=recruiter_id,
                **identity_kwargs,
            )
        return self.store.start_run(
            source="exec_search",
            brief_id=self.brief_id,
            output_dir=str(self.output_dir),
            mode="resume" if resume else "fresh",
            resume_state={"brief_name": self.brief_name},
            resumed_from_run_id=int(latest_run["id"]) if resume and latest_run else None,
            recruiter_id=recruiter_id,
            **identity_kwargs,
        )

    def upsert_lane_work_unit(
        self,
        *,
        run_id: int,
        lane: dict[str, Any],
        ordering_index: int,
        status: str,
        counters: dict[str, int] | None = None,
    ) -> int:
        counters = counters or {}
        return self.store.upsert_work_unit(
            run_id=run_id,
            source="exec_search",
            brief_id=self.brief_id,
            kind=EXEC_SEARCH_QUERY_KIND,
            source_unit_id=str(lane.get("id") or ordering_index),
            display_name=str(lane.get("name") or f"lane-{ordering_index}"),
            ordering_index=ordering_index,
            status=status,
            payload=dict(lane),
            checkpoint={"lane_type": lane.get("lane_type"), "company": lane.get("company")},
            metrics={},
            family_key=str(lane.get("company") or lane.get("lane_type") or "exec_search"),
            novelty_bucket="adapted" if lane.get("adapted_from") else "initial",
            domain_lane=str(lane.get("title") or lane.get("scope") or ""),
            counters=counters,
        )

    def record_discovery(
        self,
        *,
        run_id: int,
        work_unit_id: int,
        candidate: CandidateProfileSummary,
    ) -> None:
        self.store.record_candidate_discovery(
            run_id=run_id,
            work_unit_id=work_unit_id,
            source="exec_search",
            brief_id=self.brief_id,
            identity_key=_identity_key(candidate),
            display_name=candidate.name,
            profile_url=candidate.profile_url,
            payload={"profile_summary": candidate.to_dict()},
        )

    def record_snippet_extraction(
        self,
        *,
        run_id: int,
        work_unit_id: int,
        candidate: CandidateProfileSummary,
    ) -> None:
        # Exec Search candidates skip facial triage — discovery seeds come from
        # vetted target-company lists, so the lifecycle goes
        # discovered → snippet_extracted → full_started → full_terminal.
        # record_full_decision handles the rest.
        identity_key = _identity_key(candidate)
        snippet_attempt = self.store.start_attempt(
            run_id=run_id,
            source="exec_search",
            brief_id=self.brief_id,
            identity_key=identity_key,
            stage="snippet",
            work_unit_id=work_unit_id,
            display_name=candidate.name,
            profile_url=candidate.profile_url,
        )
        self.store.finish_attempt_success(
            attempt_id=snippet_attempt,
            new_state="snippet_extracted",
            payload={"profile_summary": candidate.to_dict()},
            run_id=run_id,
        )

    def record_full_decision(
        self,
        *,
        run_id: int,
        work_unit_id: int,
        candidate: CandidateProfileSummary,
        decision: OpusDecision,
        terminal_payload: dict[str, Any],
    ) -> None:
        identity_key = _identity_key(candidate)
        self.store.set_candidate_state(
            run_id=run_id,
            source="exec_search",
            brief_id=self.brief_id,
            identity_key=identity_key,
            new_state="full_started",
            last_work_unit_id=work_unit_id,
        )
        attempt_id = self.store.start_attempt(
            run_id=run_id,
            source="exec_search",
            brief_id=self.brief_id,
            identity_key=identity_key,
            stage="full",
            work_unit_id=work_unit_id,
            display_name=candidate.name,
            profile_url=candidate.profile_url,
        )
        payload = {
            "full_decision": decision.to_dict(),
            "profile_summary": candidate.to_dict(),
            "surface_type": "exec_search_dossier",
            **terminal_payload,
        }
        self.store.finish_attempt_success(
            attempt_id=attempt_id,
            new_state="full_terminal",
            terminal_decision=decision.decision,
            payload=payload,
            terminal_payload=payload,
            run_id=run_id,
        )


def _identity_key(candidate: CandidateProfileSummary) -> str:
    return candidate.profile_url or f"name:{candidate.name.lower().strip()}"
