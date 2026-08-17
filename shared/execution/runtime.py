"""Shared runtime helpers for candidate execution semantics."""

from __future__ import annotations

from typing import Any

from shared.failures import RECOVERABLE_ERROR, classify_runtime_failure
from shared.runtime_state import rebuild_compat_projections
from shared.runtime_state.store import RuntimeStateStore

from .types import CandidateExecutionEnvelope

_STAGE_TO_STATE = {
    "facial": "facial_started",
    "full": "full_started",
}


class SharedExecutionRuntime:
    """Canonical runtime operations for candidate-stage lifecycle work."""

    def __init__(
        self,
        *,
        store: RuntimeStateStore,
        output_dir: str,
        brief_id: str,
        source: str,
    ):
        self.store = store
        self.output_dir = output_dir
        self.brief_id = brief_id
        self.source = source
        self._progress_dirty = False
        self._artifacts_dirty = False

    def record_discovery(
        self,
        envelope: CandidateExecutionEnvelope,
        *,
        payload: dict[str, Any] | None = None,
    ) -> int:
        work_unit_id = self.store.get_work_unit_id(
            envelope.run_id,
            kind=envelope.work_unit_kind,
            source_unit_id=envelope.work_unit_source_id,
        )
        candidate_id = self.store.record_candidate_discovery(
            run_id=envelope.run_id,
            work_unit_id=work_unit_id,
            source=envelope.source,
            brief_id=envelope.brief_id,
            identity_key=envelope.identity_key,
            display_name=envelope.display_name,
            profile_url=envelope.profile_url,
            payload=payload or envelope.source_cursor,
        )
        self._artifacts_dirty = True
        return candidate_id

    def record_snippet_extracted(
        self,
        envelope: CandidateExecutionEnvelope,
        *,
        payload: dict[str, Any],
    ) -> int:
        work_unit_id = self.store.get_work_unit_id(
            envelope.run_id,
            kind=envelope.work_unit_kind,
            source_unit_id=envelope.work_unit_source_id,
        )
        attempt_id = self.store.start_attempt(
            run_id=envelope.run_id,
            source=envelope.source,
            brief_id=envelope.brief_id,
            identity_key=envelope.identity_key,
            stage="snippet",
            work_unit_id=work_unit_id,
            payload=payload,
            source_cursor=envelope.source_cursor,
            display_name=envelope.display_name,
            profile_url=envelope.profile_url,
        )
        self.store.finish_attempt_success(
            attempt_id=attempt_id,
            new_state="snippet_extracted",
            payload=payload,
            run_id=envelope.run_id,
        )
        self._artifacts_dirty = True
        return attempt_id

    def start_stage(
        self,
        envelope: CandidateExecutionEnvelope,
        *,
        stage: str,
        payload: dict[str, Any] | None = None,
        batch_key: str | None = None,
    ) -> int:
        work_unit_id = self.store.get_work_unit_id(
            envelope.run_id,
            kind=envelope.work_unit_kind,
            source_unit_id=envelope.work_unit_source_id,
        )
        new_state = _STAGE_TO_STATE.get(stage)
        if new_state:
            self.store.set_candidate_state(
                run_id=envelope.run_id,
                source=envelope.source,
                brief_id=envelope.brief_id,
                identity_key=envelope.identity_key,
                new_state=new_state,
                last_work_unit_id=work_unit_id,
            )
        return self.store.start_attempt(
            run_id=envelope.run_id,
            source=envelope.source,
            brief_id=envelope.brief_id,
            identity_key=envelope.identity_key,
            stage=stage,
            work_unit_id=work_unit_id,
            batch_key=batch_key,
            payload=payload or {},
            source_cursor=envelope.source_cursor,
            display_name=envelope.display_name,
            profile_url=envelope.profile_url,
        )

    def finish_stage_success(
        self,
        *,
        attempt_id: int | None,
        envelope: CandidateExecutionEnvelope,
        stage: str,
        decision: Any,
        extra_payload: dict[str, Any] | None = None,
        profile_summary: Any | None = None,
    ) -> None:
        if not attempt_id:
            return
        attempt_payload, terminal_payload = self._build_stage_payloads(
            envelope=envelope,
            stage=stage,
            decision=decision,
            extra_payload=extra_payload,
            profile_summary=profile_summary,
        )
        terminal_decision = getattr(decision, "decision", None)
        # C1: FACIAL_BORDERLINE is structurally peer to FACIAL_YES — both open
        # the profile and run full evaluation, and full evaluation is where
        # the lifecycle terminates. Clearing terminal_decision keeps the
        # candidate non-terminal at the facial layer for both classes. The
        # canonical row may still carry "FACIAL_BORDERLINE" if a future code
        # path produces it (Step C2+); at the lifecycle level it behaves as
        # "open", not "terminal". DEDUP_BLOCKING_LINKEDIN_DECISIONS continues
        # to omit both decisions for the same reason.
        if stage == "facial" and terminal_decision in ("FACIAL_YES", "FACIAL_BORDERLINE"):
            terminal_decision = None
        new_state = "facial_terminal" if stage == "facial" else "full_terminal"
        self.store.finish_attempt_success(
            attempt_id=attempt_id,
            new_state=new_state,
            terminal_decision=terminal_decision,
            payload=attempt_payload,
            terminal_payload=terminal_payload,
            run_id=envelope.run_id,
        )
        self._artifacts_dirty = True

    def finish_attempt_success(
        self,
        *,
        attempt_id: int | None,
        envelope: CandidateExecutionEnvelope,
        new_state: str,
        payload: dict[str, Any],
        terminal_decision: str | None = None,
    ) -> None:
        if not attempt_id:
            return
        self.store.finish_attempt_success(
            attempt_id=attempt_id,
            new_state=new_state,
            terminal_decision=terminal_decision,
            payload=payload,
            run_id=envelope.run_id,
        )
        self._artifacts_dirty = True

    def finish_stage_failure(
        self,
        *,
        attempt_id: int | None,
        envelope: CandidateExecutionEnvelope,
        stage: str,
        error_or_failure_decision: Exception | Any,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        if not attempt_id:
            return
        payload = {
            "cursor": envelope.source_cursor,
            **(extra_payload or {}),
        }
        if envelope.snippet is not None:
            payload.setdefault("snippet", self._to_payload_dict(envelope.snippet))

        if hasattr(error_or_failure_decision, "decision") and hasattr(error_or_failure_decision, "rationale"):
            decision = error_or_failure_decision
            payload[f"{stage}_decision"] = self._to_payload_dict(decision)
            prompt_capture = self._decision_prompt_capture(decision)
            if prompt_capture:
                payload["prompt_capture"] = prompt_capture
            self.store.finish_attempt_failure(
                attempt_id=attempt_id,
                failure_kind=str(decision.decision).lower(),
                failure_reason=decision.rationale,
                retryable=True,
                payload=payload,
                run_id=envelope.run_id,
            )
        else:
            classification = classify_runtime_failure(error_or_failure_decision, source=envelope.source)
            failure_kind = extra_payload.get("failure_kind_override") if extra_payload else None
            force_terminal = bool((extra_payload or {}).get("force_terminal"))
            retryable = not force_terminal and (
                classification.kind == RECOVERABLE_ERROR
                or bool(
                    (extra_payload or {}).get("profile_extraction_failed")
                    or (extra_payload or {}).get("force_retryable")
                )
            )
            self.store.finish_attempt_failure(
                attempt_id=attempt_id,
                failure_kind=failure_kind or classification.reason,
                failure_reason=classification.detail or str(error_or_failure_decision),
                retryable=retryable,
                payload=payload,
                run_id=envelope.run_id,
            )
        self._artifacts_dirty = True

    def record_terminal_runtime_decision(
        self,
        *,
        attempt_id: int | None,
        envelope: CandidateExecutionEnvelope,
        decision: str,
        payload: dict[str, Any] | None = None,
        new_state: str = "failed_terminal",
    ) -> None:
        if not attempt_id:
            return
        self.store.finish_attempt_success(
            attempt_id=attempt_id,
            new_state=new_state,
            terminal_decision=decision,
            payload=payload or {"cursor": envelope.source_cursor},
            run_id=envelope.run_id,
        )
        self._artifacts_dirty = True

    def record_side_effect_result(
        self,
        *,
        envelope: CandidateExecutionEnvelope,
        attempt_id: int | None,
        effect_type: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not attempt_id or envelope.run_id <= 0:
            return
        candidate = self.store.get_candidate(
            source=envelope.source,
            brief_id=envelope.brief_id,
            identity_key=envelope.identity_key,
        )
        self.store.record_event(
            run_id=envelope.run_id,
            work_unit_id=self.store.get_work_unit_id(
                envelope.run_id,
                kind=envelope.work_unit_kind,
                source_unit_id=envelope.work_unit_source_id,
            ),
            candidate_id=int(candidate["id"]) if candidate else None,
            attempt_id=attempt_id,
            event_type="side_effect_result",
            payload={
                "effect_type": effect_type,
                "status": status,
                **(payload or {}),
            },
        )

    def begin_candidate_side_effect(
        self,
        *,
        envelope: CandidateExecutionEnvelope,
        attempt_id: int | None,
        effect_type: str,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.store.begin_candidate_side_effect(
            run_id=envelope.run_id,
            source=envelope.source,
            brief_id=envelope.brief_id,
            identity_key=envelope.identity_key,
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
        self.store.complete_candidate_side_effect(
            side_effect_id=side_effect_id,
            status=status,
            payload=payload,
        )

    def mark_progress_dirty(self) -> None:
        self._progress_dirty = True

    def mark_artifacts_dirty(self) -> None:
        self._artifacts_dirty = True

    def flush_projections_if_needed(self, *, run_id: int, force_artifacts: bool = False) -> None:
        if not (self._progress_dirty or self._artifacts_dirty or force_artifacts):
            return
        rebuild_compat_projections(
            self.store,
            run_id=run_id,
            output_dir=self.output_dir,
        )
        self._progress_dirty = False
        self._artifacts_dirty = False

    @staticmethod
    def _build_stage_payloads(
        *,
        envelope: CandidateExecutionEnvelope,
        stage: str,
        decision: Any,
        extra_payload: dict[str, Any] | None = None,
        profile_summary: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base_payload: dict[str, Any] = {
            "cursor": envelope.source_cursor,
            f"{stage}_decision": SharedExecutionRuntime._to_payload_dict(decision),
        }
        if envelope.snippet is not None:
            base_payload["snippet"] = SharedExecutionRuntime._to_payload_dict(
                envelope.snippet
            )
        base_payload.update(extra_payload or {})
        if profile_summary is not None:
            base_payload["profile_summary"] = SharedExecutionRuntime._to_payload_dict(
                profile_summary
            )

        attempt_payload = dict(base_payload)
        terminal_payload = dict(base_payload)

        prompt_capture = SharedExecutionRuntime._decision_prompt_capture(decision)
        if prompt_capture:
            attempt_payload["prompt_capture"] = prompt_capture
            observability = SharedExecutionRuntime._observability_payload(
                prompt_capture
            )
            if observability:
                terminal_payload["observability"] = observability
        return attempt_payload, terminal_payload

    @staticmethod
    def _decision_prompt_capture(decision: Any) -> dict[str, Any] | None:
        prompt_capture = getattr(decision, "prompt_capture", None)
        if isinstance(prompt_capture, dict) and prompt_capture:
            return dict(prompt_capture)
        return None

    @staticmethod
    def _observability_payload(
        prompt_capture: dict[str, Any],
    ) -> dict[str, Any]:
        observability: dict[str, Any] = {}
        for key in ("trace_id", "observation_id", "trace_url"):
            value = prompt_capture.get(key)
            if isinstance(value, str) and value:
                observability[key] = value
        schema_version = prompt_capture.get("schema_version")
        if schema_version is not None:
            observability["prompt_capture_schema_version"] = schema_version
        return observability

    @staticmethod
    def _to_payload_dict(payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        if hasattr(payload, "to_dict"):
            return payload.to_dict()
        if isinstance(payload, dict):
            return payload
        raise TypeError(f"unsupported payload type: {type(payload)!r}")
