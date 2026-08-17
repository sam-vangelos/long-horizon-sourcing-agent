"""Shared run-status vocabulary for Cloris control plane and live signal.

Keeps terminal-run and live-worker constants in one place so
``cloris/live_signal`` does not import ``cloris.control_plane``.
"""

from __future__ import annotations

from typing import Literal

TERMINAL_RUN_STATUSES: frozenset[str] = frozenset(
    {
        "abandoned",
        "completed",
        "error",
        "failed",
        "governor_limit_reached",
        "interrupted",
        "succeeded",
    }
)

LIVE_WORKER_STATES: frozenset[str] = frozenset({"alive", "alive_silent"})

AttentionState = Literal[
    "live",
    "silent",
    "stalled",
    "terminal",
    "idle",
    "recovering",
]


def derive_attention_state(
    *,
    worker_state: str,
    latest_run_status: str | None,
    run_stalled: bool,
    lifecycle: str,
) -> AttentionState:
    """Collapse worker + run truth into recruiter-facing attention vocabulary."""

    # Canonical run lifecycle outranks worker-sidecar drift (audit F-3).
    # A stale alive worker + terminal latest_run must not surface as live/stalled.
    if latest_run_status in TERMINAL_RUN_STATUSES:
        return "terminal"
    if run_stalled:
        return "stalled"
    if worker_state == "stale" or lifecycle == "recovering":
        return "recovering"
    if worker_state == "alive_silent":
        return "silent"
    if worker_state in LIVE_WORKER_STATES:
        return "live"
    return "idle"


def live_signal_eligible(
    *,
    worker_state: str,
    latest_run_status: str | None,
) -> bool:
    """True when the homescreen may poll ``/api/run_signal`` for this entry."""

    if worker_state not in LIVE_WORKER_STATES:
        return False
    if latest_run_status is None:
        return True
    return latest_run_status not in TERMINAL_RUN_STATUSES


__all__ = [
    "AttentionState",
    "LIVE_WORKER_STATES",
    "TERMINAL_RUN_STATUSES",
    "derive_attention_state",
    "live_signal_eligible",
]
