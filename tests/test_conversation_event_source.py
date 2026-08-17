"""Tests for companion JSONL tailers."""

from __future__ import annotations

from pathlib import Path

from cloris.conversation.cost_governor import NarrationSpendGovernor
from cloris.conversation.event_source import (
    RunLogTailCursor,
    is_significant_event_type,
    poll_significant_events,
)


def test_significant_event_types() -> None:
    assert is_significant_event_type("pipeline_start")
    assert not is_significant_event_type("random_tick")


def test_tail_cursor_incremental_reads(tmp_path: Path) -> None:
    log_path = tmp_path / "run_log.jsonl"
    cursor = RunLogTailCursor(path=log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    assert cursor.drain_new_records() == []

    log_path.write_text('{"event": "pipeline_start"}\n', encoding="utf-8")
    first = cursor.drain_new_records()
    assert [r.get("event") for r in first] == ["pipeline_start"]

    log_path.write_text(
        log_path.read_text(encoding="utf-8")
        + '{"event": "circuit_breaker"}\n',
        encoding="utf-8",
    )
    second = cursor.drain_new_records()
    assert [r.get("event") for r in second] == ["circuit_breaker"]


def test_poll_batches_significant_sources(tmp_path: Path) -> None:
    lg = tmp_path / "run_log.jsonl"
    lg.parent.mkdir(parents=True, exist_ok=True)
    lg.write_text(
        '{"event": "pipeline_start"}\n'
        '{"event": "heartbeat"}\n'
        '{"event": "circuit_breaker"}\n',
        encoding="utf-8",
    )
    cursors = {"linkedin:fixture": RunLogTailCursor(path=lg)}
    found = poll_significant_events(cursors)
    types = sorted({ev.event_type for ev in found})
    assert types == ["circuit_breaker", "pipeline_start"]


def test_narration_governor_dual_caps() -> None:
    gov = NarrationSpendGovernor(
        window_s=9999.0,
        max_turns=2,
        max_usd=1.0,
        estimate_per_call_usd=0.01,
    )
    brief = "b1"
    assert gov.allow_call(brief).allowed
    assert gov.allow_call(brief).allowed
    stop = gov.allow_call(brief)
    assert not stop.allowed
    assert stop.suppressed_reason == "turn_cap"

    gov_usd = NarrationSpendGovernor(
        window_s=9999.0,
        max_turns=100,
        max_usd=0.05,
        estimate_per_call_usd=0.03,
    )
    bid = "b2"
    assert gov_usd.allow_call(bid).allowed
    assert gov_usd.allow_call(bid).allowed
    capped = gov_usd.allow_call(bid)
    assert not capped.allowed
    assert capped.suppressed_reason == "dollar_cap"
