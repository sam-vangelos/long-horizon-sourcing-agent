"""Runtime attempt lifecycle for LinkedIn sourcing runs.

Owns canonical stage-attempt start/success/failure/abort bookkeeping,
runtime event recording, and lane context projection. ``Pipeline`` delegates
to ``RuntimeAttemptService``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from shared.schemas import (
    CandidateProfileSummary,
    CandidateSnippet,
    OpusDecision,
    SearchString,
)
from shared.runtime_state.store import LINKEDIN_STRING_KIND


@dataclass(frozen=True)
class RuntimeAttemptDeps:
    get_runtime_bridge: Callable[[], Any]
    get_runtime_run_id: Callable[[], int | None]
    get_runtime_state: Callable[[], Any]
    get_in_flight_urls: Callable[[], set[str]]
    get_resume_pending_full_decisions: Callable[[], dict[str, str]]
    get_resume_pending_full_snippets: Callable[[], dict[str, CandidateSnippet]]
    get_resume_pending_full_owner_ids: Callable[[], dict[str, int]]
    funnel_candidate_key: Callable[[CandidateSnippet], str]
    note_page_full_review_settled: Callable[..., None]
    record_outreach_tier_outcome: Callable[..., None]
    variant_id_for_search_string: Callable[[SearchString], str]


class RuntimeAttemptService:
    """Owns runtime stage-attempt lifecycle and event recording."""

    def __init__(self, deps: RuntimeAttemptDeps):
        self.deps = deps

    def _record_runtime_event(
        self,
        *,
        search_string: SearchString | None,
        event_type: str,
        payload: dict,
    ) -> None:
        if not self.deps.get_runtime_run_id():
            return
        work_unit_id = None
        if search_string is not None:
            work_unit_id = self.deps.get_runtime_state().get_work_unit_id(
                self.deps.get_runtime_run_id(),
                kind=LINKEDIN_STRING_KIND,
                source_unit_id=str(search_string.id),
            )
        self.deps.get_runtime_state().record_event(
            run_id=self.deps.get_runtime_run_id(),
            work_unit_id=work_unit_id,
            event_type=event_type,
            payload=payload,
        )

    def _lane_context_for_stage(
        self,
        search_string: SearchString,
        *,
        stage: str,
    ) -> dict[str, str]:
        return {
            "lane_id": search_string.lane_id or "",
            "variant_id": self.deps.variant_id_for_search_string(search_string),
            "stage": stage,
        }

    def _start_runtime_stage_attempt(
        self,
        *,
        search_string: SearchString,
        snippet: CandidateSnippet,
        stage: str,
        payload: dict | None = None,
    ) -> int | None:
        if not self.deps.get_runtime_bridge() or not self.deps.get_runtime_run_id():
            return None
        return self.deps.get_runtime_bridge().start_stage_attempt(
            run_id=self.deps.get_runtime_run_id(),
            search_string=search_string,
            snippet=snippet,
            stage=stage,
            payload=payload,
        )

    def _finish_runtime_stage_success(
        self,
        *,
        attempt_id: int | None,
        stage: str,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        profile_summary: CandidateProfileSummary | None = None,
        extra_payload: dict | None = None,
    ) -> None:
        if self.deps.get_runtime_bridge() and self.deps.get_runtime_run_id():
            self.deps.get_runtime_bridge().finish_stage_success(
                run_id=self.deps.get_runtime_run_id(),
                attempt_id=attempt_id,
                stage=stage,
                snippet=snippet,
                decision=decision,
                profile_summary=profile_summary,
                extra_payload=extra_payload,
            )
        if stage == "full":
            self.deps.record_outreach_tier_outcome(
                snippet=snippet,
                decision=decision,
            )
            self.deps.note_page_full_review_settled(
                snippet=snippet,
                decision=decision,
            )

    def _finish_runtime_stage_failure(
        self,
        *,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        error: Exception,
        stage: str = "full",
        payload: dict | None = None,
    ) -> None:
        if self.deps.get_runtime_bridge() and self.deps.get_runtime_run_id():
            self.deps.get_runtime_bridge().finish_stage_failure(
                run_id=self.deps.get_runtime_run_id(),
                attempt_id=attempt_id,
                snippet=snippet,
                error=error,
                stage=stage,
                payload=payload,
            )
        if stage == "full":
            key = self.deps.funnel_candidate_key(snippet)
            facial_decision = self.deps.get_resume_pending_full_decisions().get(key)
            if facial_decision in {"FACIAL_YES", "FACIAL_BORDERLINE"}:
                self.deps.get_resume_pending_full_decisions()[key] = facial_decision
                self.deps.get_resume_pending_full_owner_ids()[key] = (
                    snippet.source_string_id
                )
                self.deps.get_resume_pending_full_snippets()[key] = snippet

    def _abort_runtime_stage_attempt(
        self,
        *,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        error: BaseException,
        payload: dict | None = None,
    ) -> None:
        """Close an open canonical attempt before propagating a run-level abort.

        Provider budget/auth failures, governor stops, geography violations,
        browser disconnects, and cancellation must never strand an attempt in
        ``running``.  The URL is also released from the in-process in-flight
        set so a later resume can follow canonical retryability semantics.
        """

        failure = (
            error
            if isinstance(error, Exception)
            else RuntimeError(type(error).__name__)
        )
        failure_payload = dict(payload or {})
        # Run/environment aborts are retryable after the operator fixes the
        # provider, browser, geography, or budget condition. They must not
        # permanently dedup the candidate merely because this process stopped.
        failure_payload["force_retryable"] = True
        self._finish_runtime_stage_failure(
            attempt_id=attempt_id,
            snippet=snippet,
            error=failure,
            stage=str(failure_payload.get("stage") or "full"),
            payload=failure_payload,
        )
        self.deps.get_in_flight_urls().discard(snippet.profile_url)

    def _finish_runtime_failure_decision(
        self,
        *,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        payload: dict | None = None,
    ) -> None:
        if self.deps.get_runtime_bridge() and self.deps.get_runtime_run_id():
            self.deps.get_runtime_bridge().finish_failure_decision(
                run_id=self.deps.get_runtime_run_id(),
                attempt_id=attempt_id,
                snippet=snippet,
                decision=decision,
                payload=payload,
            )
