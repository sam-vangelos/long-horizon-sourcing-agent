"""Shared coordination helpers for production-safety behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.runtime_state.store import RuntimeStateStore

from .stop_reasons import RunStopReason, normalize_stop_reason


class RunSafetyCoordinator:
    """Coordinates stop reasons, reconciliation, and safety events."""

    def __init__(
        self,
        *,
        store: RuntimeStateStore,
        output_dir: str | Path,
        source: str,
        brief_id: str,
    ):
        self.store = store
        self.output_dir = Path(output_dir)
        self.source = source
        self.brief_id = brief_id

    def reconcile_startup(self) -> dict[str, int]:
        attempts = self.store.reconcile_open_attempts(source=self.source, brief_id=self.brief_id)
        side_effects = self.store.reconcile_pending_side_effects(
            source=self.source,
            brief_id=self.brief_id,
        )
        return {
            "attempts_reconciled": attempts,
            "side_effects_reconciled": side_effects,
        }

    def finish_run(
        self,
        *,
        run_id: int,
        status: str,
        stop_reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        normalized = normalize_stop_reason(stop_reason)
        detail = (
            stop_reason
            if stop_reason is not None and stop_reason != normalized
            else None
        )
        self.store.finish_run(
            run_id,
            status,
            stop_reason=stop_reason or normalized,
        )
        event_payload: dict[str, Any] = {
            "status": status,
            "stop_reason": normalized,
            **(payload or {}),
        }
        if detail is not None:
            event_payload["stop_reason_detail"] = detail
        self.store.record_event(
            run_id=run_id,
            event_type="run_stop_reason",
            payload=event_payload,
        )

    def record_governor_limit(self, *, run_id: int, reason: str, payload: dict[str, Any] | None = None) -> None:
        self.store.set_run_stop_reason(run_id, RunStopReason.GOVERNOR_LIMIT)
        self.store.record_event(
            run_id=run_id,
            event_type="governor_limit_reached",
            payload={"reason": reason, **(payload or {})},
        )

    def record_browser_recovery_event(
        self,
        *,
        run_id: int,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.store.record_event(
            run_id=run_id,
            event_type="browser_recovery",
            payload={"status": status, **(payload or {})},
        )
