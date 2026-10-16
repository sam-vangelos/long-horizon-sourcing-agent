"""Canonical run stop reasons for runtime-backed sourcing runs."""

from __future__ import annotations


class RunStopReason:
    NORMAL = "normal"
    GOVERNOR_LIMIT = "governor_limit"
    SESSION_EXPIRED = "session_expired"
    OPERATOR_STOP = "operator_stop"
    OPERATOR_PAUSE = "operator_pause"
    LOCK_CONFLICT = "lock_conflict"
    BROWSER_DISCONNECT_UNRECOVERED = "browser_disconnect_unrecovered"
    API_BUDGET_EXHAUSTED = "api_budget_exhausted"
    FATAL_RUNTIME_ERROR = "fatal_runtime_error"
    # Set by cloris.reconciler when a run is marked status='running' but the
    # worker process has died (no sidecar, or sidecar PID is gone). Drives
    # the "Lost track" UI state — Cloris doesn't know what happened, just
    # that nothing is making progress on this run anymore.
    WORKER_MISSING = "worker_missing"


_KNOWN_STOP_REASONS = {
    RunStopReason.NORMAL,
    RunStopReason.GOVERNOR_LIMIT,
    RunStopReason.SESSION_EXPIRED,
    RunStopReason.OPERATOR_STOP,
    RunStopReason.OPERATOR_PAUSE,
    RunStopReason.LOCK_CONFLICT,
    RunStopReason.BROWSER_DISCONNECT_UNRECOVERED,
    RunStopReason.API_BUDGET_EXHAUSTED,
    RunStopReason.FATAL_RUNTIME_ERROR,
    RunStopReason.WORKER_MISSING,
}


def normalize_stop_reason(stop_reason: str | None) -> str:
    if not stop_reason:
        return RunStopReason.NORMAL
    return stop_reason if stop_reason in _KNOWN_STOP_REASONS else RunStopReason.FATAL_RUNTIME_ERROR
