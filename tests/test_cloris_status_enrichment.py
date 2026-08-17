"""Phase 4 tests for status enrichment: attempt_health, progress, stalled.

Pins the aggregator-side classification so the UI's promotion logic
("stalled runs go to the attention lane") has a stable contract:

- ``attempt_health`` and ``work_unit_progress`` are populated whenever
  a latest run exists.
- ``run_stalled=True`` only fires when:
  - the worker is alive (not stale, not silent, not missing),
  - the latest success was more than 5 minutes ago,
  - at least 3 recent failures dominate ≥80% of attempts,
  - and the dominant failure_kind is in the retryable set.
- ``stall_failure_kind`` carries the dominant kind so the UI can
  render "stalled · http_429" without consulting the histogram.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cloris.control_plane import aggregate_status
from cloris.worker import build_sidecar, write_sidecar
from shared.runtime_state.store import RuntimeStateStore


def _build_state_dir(state_root: Path, source: str, key: str) -> Path:
    state_dir = state_root / source / key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _write_alive_sidecar(state_dir: Path, *, pid: int) -> None:
    """Sidecar with a fresh heartbeat so the worker reports as 'alive',
    not 'alive_silent'. Phase 4 tests want to exercise the stalled path
    independently of the heartbeat path."""

    fresh_now = datetime.now(timezone.utc).isoformat()
    payload = build_sidecar(
        source="linkedin",
        brief_id="brief-4",
        brief_path=str(state_dir / "brief.json"),
        output_dir=str(state_dir),
        mode="fresh",
        input_mode="concurrent",
        started_at=fresh_now,
        pid=pid,
        run_id=None,
    )
    write_sidecar(state_dir, payload)


def _seed_attempts(
    db_path: Path,
    *,
    run_id: int,
    candidate_id: int,
    failures: int,
    failure_kind: str = "http_429",
    successes: int = 0,
    success_age_s: float = 600.0,
) -> None:
    """Direct SQL insert so we can place ages outside what the store's
    state-machine guards would allow. ``success_age_s`` controls how
    long ago the last success was (relative to now)."""

    conn = sqlite3.connect(str(db_path))
    try:
        now = datetime.now(timezone.utc)
        for _ in range(failures):
            conn.execute(
                "INSERT INTO candidate_attempts(run_id, candidate_id, stage, "
                "attempt_number, status, failure_kind, started_at, ended_at) "
                "VALUES (?, ?, 'snippet', 1, 'failed', ?, ?, ?)",
                (
                    run_id,
                    candidate_id,
                    failure_kind,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        if successes:
            success_when = (now - timedelta(seconds=success_age_s)).isoformat()
            for _ in range(successes):
                conn.execute(
                    "INSERT INTO candidate_attempts(run_id, candidate_id, stage, "
                    "attempt_number, status, started_at, ended_at) "
                    "VALUES (?, ?, 'snippet', 1, 'succeeded', ?, ?)",
                    (run_id, candidate_id, success_when, success_when),
                )
        conn.commit()
    finally:
        conn.close()


def test_aggregator_populates_attempt_health_when_run_exists(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "health-key")
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-4",
        output_dir=str(state_dir),
        mode="fresh",
    )
    candidate_id = store.ensure_candidate(
        source="linkedin", brief_id="brief-4", identity_key="alice"
    )
    _seed_attempts(
        state_dir / "runtime_state.sqlite3",
        run_id=run_id,
        candidate_id=candidate_id,
        failures=2,
        successes=1,
        success_age_s=10.0,
    )

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.attempt_health is not None
    assert entry.attempt_health.failed_in_window == 2
    assert entry.attempt_health.succeeded_in_window == 1
    assert entry.attempt_health.dominant_failure_kind == "http_429"


def test_aggregator_populates_work_unit_progress_for_linkedin(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "progress-key")
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-4",
        output_dir=str(state_dir),
        mode="fresh",
    )
    for i, status in enumerate(["queued", "queued", "in_progress", "done", "done"]):
        store.upsert_work_unit(
            run_id=run_id,
            source="linkedin",
            brief_id="brief-4",
            kind="linkedin_string",
            source_unit_id=str(i),
            display_name=f"unit-{i}",
            ordering_index=i,
            status=status,
        )

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.work_unit_progress is not None
    assert entry.work_unit_progress.kind == "counts"
    assert entry.work_unit_progress.queued == 2
    assert entry.work_unit_progress.in_progress == 1
    assert entry.work_unit_progress.done == 2


def test_aggregator_classifies_stalled_when_alive_with_retryable_failures(
    tmp_path: Path,
) -> None:
    """Alive worker + last success 10 minutes ago + 5 recent http_429s
    out of 5 attempts ⇒ stalled. The UI promotes this to the attention
    lane so the user can intervene."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "stalled-key")
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-4",
        output_dir=str(state_dir),
        mode="fresh",
    )
    candidate_id = store.ensure_candidate(
        source="linkedin", brief_id="brief-4", identity_key="alice"
    )
    _write_alive_sidecar(state_dir, pid=os.getpid())
    _seed_attempts(
        state_dir / "runtime_state.sqlite3",
        run_id=run_id,
        candidate_id=candidate_id,
        failures=5,
        failure_kind="http_429",
        successes=0,
    )

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.run_stalled is True
    assert entry.stall_failure_kind == "http_429"


def test_terminal_run_outranks_stalled_attention_with_alive_worker(
    tmp_path: Path,
) -> None:
    """Completed canonical run + stale alive worker + retryable failures
    must not promote the card to front-of-file attention."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "terminal-stalled-key")
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-4",
        output_dir=str(state_dir),
        mode="fresh",
    )
    candidate_id = store.ensure_candidate(
        source="linkedin", brief_id="brief-4", identity_key="alice"
    )
    store.finish_run(run_id, "completed")
    _write_alive_sidecar(state_dir, pid=os.getpid())
    _seed_attempts(
        state_dir / "runtime_state.sqlite3",
        run_id=run_id,
        candidate_id=candidate_id,
        failures=5,
        failure_kind="http_429",
        successes=0,
    )

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.latest_run is not None
    assert entry.latest_run.status == "completed"
    assert entry.run_stalled is True
    assert entry.attention_state == "terminal"
    assert entry.live_signal_eligible is False


def test_aggregator_does_not_classify_stalled_with_recent_success(
    tmp_path: Path,
) -> None:
    """A recent success means the worker is still making progress;
    failures are normal noise, not a stall."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "healthy-key")
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-4",
        output_dir=str(state_dir),
        mode="fresh",
    )
    candidate_id = store.ensure_candidate(
        source="linkedin", brief_id="brief-4", identity_key="alice"
    )
    _write_alive_sidecar(state_dir, pid=os.getpid())
    _seed_attempts(
        state_dir / "runtime_state.sqlite3",
        run_id=run_id,
        candidate_id=candidate_id,
        failures=5,
        failure_kind="http_429",
        successes=1,
        success_age_s=30.0,
    )

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.run_stalled is False


def test_aggregator_does_not_classify_stalled_for_non_retryable_kind(
    tmp_path: Path,
) -> None:
    """A burst of authentication failures is not a stall — those are
    terminal and the user already needs to handle them via a different
    surface (fix credentials, retry the run). The stall classifier
    only fires on retryable kinds."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "auth-fail-key")
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-4",
        output_dir=str(state_dir),
        mode="fresh",
    )
    candidate_id = store.ensure_candidate(
        source="linkedin", brief_id="brief-4", identity_key="alice"
    )
    _write_alive_sidecar(state_dir, pid=os.getpid())
    _seed_attempts(
        state_dir / "runtime_state.sqlite3",
        run_id=run_id,
        candidate_id=candidate_id,
        failures=5,
        failure_kind="auth_failed",
        successes=0,
    )

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.run_stalled is False


def test_aggregator_does_not_classify_stalled_when_worker_is_stale(
    tmp_path: Path,
) -> None:
    """A stale worker is already classified as such; the stall
    classifier only applies to alive workers (otherwise the UI would
    double-promote)."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "stale-key")
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-4",
        output_dir=str(state_dir),
        mode="fresh",
    )
    candidate_id = store.ensure_candidate(
        source="linkedin", brief_id="brief-4", identity_key="alice"
    )
    # No sidecar ⇒ worker_state="missing" ⇒ stall classifier should
    # short-circuit even though the failure pattern would otherwise
    # match.
    _seed_attempts(
        state_dir / "runtime_state.sqlite3",
        run_id=run_id,
        candidate_id=candidate_id,
        failures=5,
        failure_kind="http_429",
        successes=0,
    )

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.run_stalled is False


def test_aggregator_attempt_health_none_when_no_run_exists(tmp_path: Path) -> None:
    """A state dir whose runtime DB has no runs row should leave
    attempt_health as None — there's nothing to summarize."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "no-run-key")
    RuntimeStateStore(state_dir / "runtime_state.sqlite3")

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.attempt_health is None
    assert entry.work_unit_progress is None
    assert entry.run_stalled is False


def test_aggregator_progress_kind_for_github(tmp_path: Path) -> None:
    """GitHub uses ``github_query`` work units, not ``linkedin_string``."""

    state_dir = _build_state_dir(tmp_path, "github", "gh-key")
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="github",
        brief_id="brief-gh",
        output_dir=str(state_dir),
        mode="fresh",
    )
    store.upsert_work_unit(
        run_id=run_id,
        source="github",
        brief_id="brief-gh",
        kind="github_query",
        source_unit_id="q1",
        display_name="query 1",
        ordering_index=0,
        status="done",
    )

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.work_unit_progress is not None
    assert entry.work_unit_progress.kind == "counts"
    assert entry.work_unit_progress.done == 1


def test_terminal_canonical_with_alive_worker_sets_projection_disagreement(
    tmp_path: Path,
) -> None:
    """Stale projection must not imply live polling when SQLite is terminal."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "terminal-key")
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-terminal",
        output_dir=str(state_dir),
        mode="fresh",
    )
    store.finish_run(run_id, status="completed", stop_reason="governor_limit_reached")
    _write_alive_sidecar(state_dir, pid=os.getpid())

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.latest_run is not None
    assert entry.latest_run.status == "completed"
    assert entry.projection_disagreement is True
    assert entry.live_signal_eligible is False
    assert entry.active is False
    assert entry.attention_state == "terminal"
