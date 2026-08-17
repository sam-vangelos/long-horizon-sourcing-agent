"""Phase 1.6 tests for the worker.json heartbeat updater.

Pins three properties end-to-end:

- ``shared.runtime_state.heartbeat.bump_heartbeat`` atomically updates
  only the ``heartbeat_at`` field, leaves the rest of the sidecar
  unchanged, no-ops on missing/malformed sidecars, and never raises.
- ``RuntimeStateStore`` write checkpoints (start_run, finish_attempt_*,
  upsert_work_unit, finish_run) bump the sidecar heartbeat when a
  sidecar exists at ``state_dir/worker.json``, and skip silently when
  it doesn't (decoy-only runs, tests that don't want the side effect).
- ``cloris.control_plane.aggregate_status`` computes ``heartbeat_age_s``
  and promotes ``worker_state`` to ``alive_silent`` when the age
  exceeds the threshold and the PID is alive.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cloris import control_plane
from cloris.control_plane import (
    ALIVE_SILENT_THRESHOLD_S,
    aggregate_status,
)
from cloris.worker import WORKER_SIDECAR_FILENAME, build_sidecar, write_sidecar
from shared.runtime_state.heartbeat import bump_heartbeat
from shared.runtime_state.store import RuntimeStateStore


# --- bump_heartbeat unit -----------------------------------------------------


def _write_sidecar(state_dir: Path, **overrides) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    fresh_now = datetime.now(timezone.utc).isoformat()
    payload = build_sidecar(
        source="linkedin",
        brief_id="brief-hb",
        brief_path=str(state_dir / "brief.json"),
        output_dir=str(state_dir),
        mode="fresh",
        input_mode="concurrent",
        started_at=fresh_now,
        pid=os.getpid(),
        run_id=None,
    )
    payload.update(overrides)
    return write_sidecar(state_dir, payload)


def test_bump_heartbeat_updates_field_in_place(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    old_heartbeat = "2020-01-01T00:00:00+00:00"
    _write_sidecar(state_dir, heartbeat_at=old_heartbeat)

    assert bump_heartbeat(state_dir) is True

    parsed = json.loads((state_dir / WORKER_SIDECAR_FILENAME).read_text())
    assert parsed["heartbeat_at"] != old_heartbeat
    # heartbeat_at must parse as a tz-aware ISO timestamp so the aggregator's
    # datetime.fromisoformat doesn't choke.
    ts = datetime.fromisoformat(parsed["heartbeat_at"])
    assert ts.tzinfo is not None


def test_bump_heartbeat_preserves_other_fields(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _write_sidecar(state_dir)
    before = json.loads((state_dir / WORKER_SIDECAR_FILENAME).read_text())

    bump_heartbeat(state_dir)
    after = json.loads((state_dir / WORKER_SIDECAR_FILENAME).read_text())

    # Every field other than heartbeat_at must be byte-identical so probes
    # against pid/mode/brief_path stay coherent.
    for key in before:
        if key == "heartbeat_at":
            continue
        assert before[key] == after[key], (
            f"bump_heartbeat must not modify {key!r}"
        )


def test_bump_heartbeat_no_sidecar_is_noop(tmp_path: Path) -> None:
    state_dir = tmp_path / "no-sidecar"
    state_dir.mkdir()
    assert bump_heartbeat(state_dir) is False
    # Must not create a sidecar where there isn't one.
    assert not (state_dir / WORKER_SIDECAR_FILENAME).exists()


def test_bump_heartbeat_malformed_sidecar_is_noop(tmp_path: Path) -> None:
    state_dir = tmp_path / "bad"
    state_dir.mkdir()
    sidecar = state_dir / WORKER_SIDECAR_FILENAME
    sidecar.write_text("not json")
    assert bump_heartbeat(state_dir) is False
    # Malformed file must not be silently rewritten.
    assert sidecar.read_text() == "not json"


def test_bump_heartbeat_atomic_under_concurrent_calls(tmp_path: Path) -> None:
    """Concurrent bumps must never leave a partially-written file.

    The atomicity contract: a reader doing ``json.loads(sidecar.read_text())``
    while writers are racing must always observe either the pre-bump or
    post-bump JSON, never a partial document. We exercise the path with
    a single writer and reader at moderate cadence — high enough to hit
    the os.replace transition window many times, low enough that the
    test stays deterministic under suite-level contention.
    """

    state_dir = tmp_path / "concurrent"
    _write_sidecar(state_dir)
    sidecar = state_dir / WORKER_SIDECAR_FILENAME

    stop = threading.Event()
    json_errors: list[str] = []
    incomplete_observations: list[str] = []
    successful_reads = 0

    def reader() -> None:
        nonlocal successful_reads
        while not stop.is_set():
            try:
                parsed = json.loads(sidecar.read_text())
            except FileNotFoundError:
                # Benign rename race on some filesystems; skip and retry.
                continue
            except json.JSONDecodeError as exc:
                # Partial-write violation. THIS is the atomicity bug
                # the test is hunting; fail loudly if it ever fires.
                json_errors.append(str(exc))
                continue
            if not isinstance(parsed, dict) or "pid" not in parsed:
                incomplete_observations.append(repr(parsed))
            else:
                successful_reads += 1

    def writer() -> None:
        for _ in range(30):
            bump_heartbeat(state_dir)
            time.sleep(0.002)

    rt = threading.Thread(target=reader)
    wt = threading.Thread(target=writer)
    rt.start()
    wt.start()
    wt.join(timeout=5.0)
    stop.set()
    rt.join(timeout=2.0)

    assert json_errors == [], (
        f"partial JSON observed during bump_heartbeat (atomicity violation): "
        f"{json_errors[:3]}"
    )
    assert incomplete_observations == [], (
        f"sidecar observed without expected fields: {incomplete_observations[:3]}"
    )
    assert successful_reads > 0, "reader observed no successful sidecar reads"


# --- RuntimeStateStore integration -------------------------------------------


def test_store_writes_bump_heartbeat_when_sidecar_exists(tmp_path: Path) -> None:
    state_dir = tmp_path
    _write_sidecar(state_dir, heartbeat_at="2020-01-01T00:00:00+00:00")
    db_path = state_dir / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)

    store.start_run(
        source="linkedin",
        brief_id="brief-hb",
        output_dir=str(state_dir),
        mode="fresh",
    )

    parsed = json.loads((state_dir / WORKER_SIDECAR_FILENAME).read_text())
    # Must have advanced past the fixed legacy timestamp.
    assert parsed["heartbeat_at"] != "2020-01-01T00:00:00+00:00"
    ts = datetime.fromisoformat(parsed["heartbeat_at"])
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    assert age < 5.0, f"heartbeat should be fresh; age={age}s"


def test_store_writes_silent_when_no_sidecar(tmp_path: Path) -> None:
    """Tests that don't write a sidecar must not crash on the heartbeat
    path. This is the contract for orchestrators run standalone outside
    of Cloris's launch path."""

    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    # No sidecar at tmp_path; start_run must not raise.
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-hb",
        output_dir=str(tmp_path),
        mode="fresh",
    )
    store.finish_run(run_id, "completed")
    assert not (tmp_path / WORKER_SIDECAR_FILENAME).exists()


# --- aggregator alive_silent classification ---------------------------------


def _build_state_dir(state_root: Path, source: str, key: str) -> Path:
    state_dir = state_root / source / key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def test_aggregator_alive_with_fresh_heartbeat(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "alive-fresh")
    RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    _write_sidecar(state_dir, pid=os.getpid())

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.worker_state == "alive"
    assert entry.heartbeat_age_s is not None
    assert entry.heartbeat_age_s < ALIVE_SILENT_THRESHOLD_S


def test_aggregator_alive_silent_when_heartbeat_stale(tmp_path: Path) -> None:
    """Heartbeat older than the threshold + PID alive ⇒ alive_silent.
    The UI promotes alive_silent to the attention lane."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "alive-silent")
    RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=ALIVE_SILENT_THRESHOLD_S + 60)
    ).isoformat()
    _write_sidecar(state_dir, pid=os.getpid(), heartbeat_at=stale)

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.worker_state == "alive_silent"
    assert entry.heartbeat_age_s is not None
    assert entry.heartbeat_age_s > ALIVE_SILENT_THRESHOLD_S


def test_aggregator_handles_unparseable_heartbeat(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "bad-heartbeat")
    RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    _write_sidecar(state_dir, pid=os.getpid(), heartbeat_at="not-a-timestamp")

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    # Unparseable heartbeat collapses to None; absent staleness signal,
    # the worker is reported as alive (still classified by PID liveness).
    assert entry.heartbeat_age_s is None
    assert entry.worker_state == "alive"


def test_aggregator_floors_negative_heartbeat_age(tmp_path: Path) -> None:
    """A sidecar written by a host whose clock is ahead of ours produces
    a negative wall-clock delta. Floor to 0 rather than emitting a
    negative number that breaks UI rendering downstream."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "future-heartbeat")
    RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    future = (
        datetime.now(timezone.utc) + timedelta(seconds=30)
    ).isoformat()
    _write_sidecar(state_dir, pid=os.getpid(), heartbeat_at=future)

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.heartbeat_age_s == 0.0
    assert entry.worker_state == "alive"
