"""Tests for shared run-status attention vocabulary."""

from __future__ import annotations

from shared.run_status_constants import (
    derive_attention_state,
    live_signal_eligible,
)


def test_live_signal_eligible_requires_live_worker_and_non_terminal_run() -> None:
    assert live_signal_eligible(worker_state="alive", latest_run_status="running")
    assert not live_signal_eligible(
        worker_state="alive", latest_run_status="completed"
    )
    assert not live_signal_eligible(
        worker_state="missing", latest_run_status="running"
    )


def test_derive_attention_state_stalled_wins_for_non_terminal_run() -> None:
    assert (
        derive_attention_state(
            worker_state="alive",
            latest_run_status="running",
            run_stalled=True,
            lifecycle="searching",
        )
        == "stalled"
    )


def test_terminal_outranks_stalled_with_alive_worker() -> None:
    assert (
        derive_attention_state(
            worker_state="alive",
            latest_run_status="completed",
            run_stalled=True,
            lifecycle="searching",
        )
        == "terminal"
    )
    assert not live_signal_eligible(
        worker_state="alive", latest_run_status="completed"
    )


def test_derive_attention_state_terminal_when_run_over() -> None:
    assert (
        derive_attention_state(
            worker_state="missing",
            latest_run_status="succeeded",
            run_stalled=False,
            lifecycle="finished",
        )
        == "terminal"
    )
