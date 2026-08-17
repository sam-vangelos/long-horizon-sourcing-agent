"""Tests for the zombie-run reconciler (Phase 1A).

These tests pin the contract of :mod:`cloris.reconciler`:

  - Live workers (recent PID, recent heartbeat) are never reconciled.
  - Workers with no sidecar are reconciled to ``status='abandoned'``.
  - Workers with a stale-PID sidecar are reconciled the same way.
  - The reconciler is idempotent: a re-run after applying mutations
    produces zero further mutations.
  - Mutations carry the canonical stop reason
    :data:`shared.safety.stop_reasons.RunStopReason.WORKER_MISSING`.
  - The reconciler walks every discovered state dir; a single reconcile
    pass handles multiple zombies in one call.
  - The ``apply_mutations`` step re-checks status before writing so a
    benign race (run finalized between read and write) is absorbed
    without an exception.

The fixture pattern follows ``tests/test_cloris_status_aggregation.py``:
build state dirs under a tmp_path-rooted state_root, use
``RuntimeStateStore`` to seed canonical state (``start_run``), and write
``worker.json`` directly via ``cloris.worker.write_sidecar`` to simulate
the various worker-state branches.
"""

from __future__ import annotations

import json
import signal
from datetime import datetime, timezone
from pathlib import Path

from cloris.reconciler import (
    Mutation,
    apply_mutations,
    cleanup_drainable_terminal_lock,
    cleanup_terminal_browser_lock,
    reconcile_orphans,
    reconcile_and_apply,
)
from cloris.worker import WORKER_SIDECAR_FILENAME, write_sidecar
from shared.runtime_state import read_models
from shared.runtime_state.store import RuntimeStateStore
from shared.safety.stop_reasons import RunStopReason


def _build_state_dir(state_root: Path, source: str, key: str) -> Path:
    state_dir = state_root / source / key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _start_running_run(state_dir: Path, *, brief_id: str = "brief-x") -> int:
    db_path = state_dir / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    return store.start_run(
        source="linkedin",
        brief_id=brief_id,
        output_dir=str(state_dir),
        mode="fresh",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _alive_pid() -> int:
    """Return a PID guaranteed to be alive — the test process itself."""
    import os
    return os.getpid()


def _dead_pid() -> int:
    """Return a PID guaranteed NOT to be alive on this host.

    PID 999_999_999 is well above the typical Linux/macOS PID ceiling so
    a real process is essentially impossible. is_pid_alive(999_999_999)
    returns False on every platform we run on.
    """
    return 999_999_999


# ---- reconcile_orphans -----------------------------------------------------


class TestReconcileOrphans:
    def test_no_state_dirs_returns_empty(self, tmp_path: Path) -> None:
        # No source roots created; enumerate yields nothing.
        assert reconcile_orphans(tmp_path) == []

    def test_state_dir_without_db_skipped(self, tmp_path: Path) -> None:
        _build_state_dir(tmp_path, "linkedin", "no-db")
        assert reconcile_orphans(tmp_path) == []

    def test_run_already_finished_not_reconciled(self, tmp_path: Path) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "finished")
        run_id = _start_running_run(state_dir)
        store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
        store.finish_run(run_id, "completed")
        # Even with no sidecar, a non-running run is not reconciled.
        assert reconcile_orphans(tmp_path) == []

    def test_running_run_no_sidecar_reconciled(self, tmp_path: Path) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "no-sidecar")
        run_id = _start_running_run(state_dir)
        # No worker.json — classic zombie scenario.
        mutations = reconcile_orphans(tmp_path)
        assert len(mutations) == 1
        m = mutations[0]
        assert m.source == "linkedin"
        assert m.state_key == "no-sidecar"
        assert m.run_id == run_id
        assert m.new_status == "abandoned"
        assert m.stop_reason == RunStopReason.WORKER_MISSING
        assert m.reason == "missing_sidecar"

    def test_running_run_dead_pid_reconciled(self, tmp_path: Path) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "dead-pid")
        run_id = _start_running_run(state_dir)
        write_sidecar(state_dir, {
            "pid": _dead_pid(),
            "mode": "fresh",
            "input_mode": "concurrent",
            "brief_id": "brief-x",
            "brief_path": str(state_dir / "brief.json"),
            "heartbeat_at": _now_iso(),
            "launcher_version": "test",
        })
        mutations = reconcile_orphans(tmp_path)
        assert len(mutations) == 1
        assert mutations[0].reason == "pid_dead"
        assert mutations[0].run_id == run_id

    def test_running_run_bad_sidecar_pid_reconciled(self, tmp_path: Path) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "bad-pid")
        run_id = _start_running_run(state_dir)
        write_sidecar(state_dir, {
            "pid": "not-an-int",
            "mode": "fresh",
            "input_mode": "concurrent",
            "brief_id": "brief-x",
            "brief_path": str(state_dir / "brief.json"),
            "heartbeat_at": _now_iso(),
            "launcher_version": "test",
        })
        mutations = reconcile_orphans(tmp_path)
        assert len(mutations) == 1
        assert mutations[0].reason == "bad_sidecar"

    def test_running_run_negative_pid_reconciled(self, tmp_path: Path) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "neg-pid")
        _start_running_run(state_dir)
        write_sidecar(state_dir, {
            "pid": -1,
            "mode": "fresh",
            "input_mode": "concurrent",
            "brief_id": "brief-x",
            "brief_path": str(state_dir / "brief.json"),
            "heartbeat_at": _now_iso(),
            "launcher_version": "test",
        })
        mutations = reconcile_orphans(tmp_path)
        assert len(mutations) == 1
        assert mutations[0].reason == "bad_sidecar"

    def test_running_run_alive_worker_not_reconciled(self, tmp_path: Path) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "alive")
        _start_running_run(state_dir)
        # Sidecar with the test process's own PID — alive by definition.
        write_sidecar(state_dir, {
            "pid": _alive_pid(),
            "mode": "fresh",
            "input_mode": "concurrent",
            "brief_id": "brief-x",
            "brief_path": str(state_dir / "brief.json"),
            "heartbeat_at": _now_iso(),
            "launcher_version": "test",
        })
        # Alive worker — even if heartbeat is stale, we don't kill it.
        assert reconcile_orphans(tmp_path) == []

    def test_alive_silent_not_reconciled(self, tmp_path: Path) -> None:
        """alive_silent (PID alive but heartbeat stale) is NOT a zombie.

        Long captcha waits, suspended laptops, and slow page loads can
        all produce alive_silent. Killing those would create a worse
        failure mode than the one we're fixing.
        """
        state_dir = _build_state_dir(tmp_path, "linkedin", "silent")
        _start_running_run(state_dir)
        old_heartbeat = "2020-01-01T00:00:00+00:00"  # stale by years
        write_sidecar(state_dir, {
            "pid": _alive_pid(),
            "mode": "fresh",
            "input_mode": "concurrent",
            "brief_id": "brief-x",
            "brief_path": str(state_dir / "brief.json"),
            "heartbeat_at": old_heartbeat,
            "launcher_version": "test",
        })
        assert reconcile_orphans(tmp_path) == []

    def test_multiple_zombies_in_one_pass(self, tmp_path: Path) -> None:
        for key in ("a", "b", "c"):
            sd = _build_state_dir(tmp_path, "linkedin", key)
            _start_running_run(sd, brief_id=f"brief-{key}")
        # All three lack sidecars → all zombies.
        mutations = reconcile_orphans(tmp_path)
        assert len(mutations) == 3
        assert {m.state_key for m in mutations} == {"a", "b", "c"}

    def test_mutations_sorted_deterministically(self, tmp_path: Path) -> None:
        for key in ("zzz", "aaa", "mmm"):
            sd = _build_state_dir(tmp_path, "linkedin", key)
            _start_running_run(sd, brief_id=f"brief-{key}")
        mutations = reconcile_orphans(tmp_path)
        keys = [m.state_key for m in mutations]
        assert keys == sorted(keys)


# ---- apply_mutations -------------------------------------------------------


class TestApplyMutations:
    def test_empty_mutations_noop(self, tmp_path: Path) -> None:
        assert apply_mutations([]) == 0

    def test_apply_writes_status_abandoned(self, tmp_path: Path) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "z")
        run_id = _start_running_run(state_dir)
        mutations = reconcile_orphans(tmp_path)
        applied = apply_mutations(mutations)
        assert applied == 1
        # Re-read latest run from canonical state — status should be abandoned.
        latest = read_models.latest_run_summary(
            state_dir / "runtime_state.sqlite3"
        )
        assert latest is not None
        assert latest.id == run_id
        assert latest.status == "abandoned"
        assert latest.stop_reason == RunStopReason.WORKER_MISSING

    def test_apply_skips_already_finalized(self, tmp_path: Path) -> None:
        """Race protection: between reconcile and apply, the run finalized.

        The reconciler emits the mutation; before apply runs, the
        orchestrator (in the real world) called finish_run. apply
        should detect this and skip the write.
        """
        state_dir = _build_state_dir(tmp_path, "linkedin", "race")
        run_id = _start_running_run(state_dir)
        mutations = reconcile_orphans(tmp_path)
        # Finalize between reconcile and apply.
        store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
        store.finish_run(run_id, "completed")
        applied = apply_mutations(mutations)
        assert applied == 0
        latest = read_models.latest_run_summary(
            state_dir / "runtime_state.sqlite3"
        )
        assert latest is not None
        # Still 'completed' — the apply did not overwrite the legitimate
        # terminal transition.
        assert latest.status == "completed"


# ---- terminal browser-lock cleanup ----------------------------------------


class TestCleanupTerminalBrowserLocks:
    def test_terminal_browser_disconnect_clears_sidecar_without_db_mutation(
        self, tmp_path: Path
    ) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "browser-ended")
        run_id = _start_running_run(state_dir)
        store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
        store.finish_run(
            run_id,
            "interrupted",
            stop_reason=RunStopReason.BROWSER_DISCONNECT_UNRECOVERED,
        )
        write_sidecar(state_dir, {
            "pid": _alive_pid(),
            "mode": "fresh",
            "input_mode": "concurrent",
            "brief_id": "brief-x",
            "brief_path": str(state_dir / "brief.json"),
            "heartbeat_at": "2020-01-01T00:00:00+00:00",
            "launcher_version": "test",
        })
        signals: list[tuple[int, int]] = []

        cleaned = cleanup_terminal_browser_lock(
            state_dir,
            send_signal=lambda pid, sig: signals.append((pid, sig)),
        )

        assert cleaned is True
        assert signals == [(_alive_pid(), signal.SIGTERM)]
        assert not (state_dir / WORKER_SIDECAR_FILENAME).exists()
        latest = read_models.latest_run_summary(state_dir / "runtime_state.sqlite3")
        assert latest is not None
        assert latest.status == "interrupted"
        assert latest.stop_reason == RunStopReason.BROWSER_DISCONNECT_UNRECOVERED

    def test_non_browser_terminal_run_keeps_sidecar(self, tmp_path: Path) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "operator-stop")
        run_id = _start_running_run(state_dir)
        store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
        store.finish_run(
            run_id,
            "interrupted",
            stop_reason=RunStopReason.OPERATOR_STOP,
        )
        write_sidecar(state_dir, {
            "pid": _alive_pid(),
            "mode": "fresh",
            "input_mode": "concurrent",
            "brief_id": "brief-x",
            "brief_path": str(state_dir / "brief.json"),
            "heartbeat_at": "2020-01-01T00:00:00+00:00",
            "launcher_version": "test",
        })

        assert cleanup_terminal_browser_lock(state_dir) is False
        assert (state_dir / WORKER_SIDECAR_FILENAME).exists()

    def test_drainable_terminal_run_clears_sidecar_without_signaling(
        self, tmp_path: Path
    ) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "api-budget-ended")
        run_id = _start_running_run(state_dir)
        store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
        store.finish_run(
            run_id,
            "interrupted",
            stop_reason=RunStopReason.API_BUDGET_EXHAUSTED,
        )
        write_sidecar(state_dir, {
            "pid": _alive_pid(),
            "mode": "fresh",
            "input_mode": "concurrent",
            "brief_id": "brief-x",
            "brief_path": str(state_dir / "brief.json"),
            "heartbeat_at": "2020-01-01T00:00:00+00:00",
            "launcher_version": "test",
        })
        signals: list[tuple[int, int]] = []

        cleaned = cleanup_drainable_terminal_lock(
            state_dir,
            send_signal=lambda pid, sig: signals.append((pid, sig)),
        )

        assert cleaned is True
        assert signals == []
        assert not (state_dir / WORKER_SIDECAR_FILENAME).exists()

    def test_reconcile_and_apply_also_cleans_terminal_browser_locks(
        self, tmp_path: Path
    ) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "browser-ended")
        run_id = _start_running_run(state_dir)
        store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
        store.finish_run(
            run_id,
            "interrupted",
            stop_reason=RunStopReason.BROWSER_DISCONNECT_UNRECOVERED,
        )
        write_sidecar(state_dir, {
            "pid": _dead_pid(),
            "mode": "fresh",
            "input_mode": "concurrent",
            "brief_id": "brief-x",
            "brief_path": str(state_dir / "brief.json"),
            "heartbeat_at": "2020-01-01T00:00:00+00:00",
            "launcher_version": "test",
        })

        applied, mutations = reconcile_and_apply(tmp_path)

        assert applied == 0
        assert mutations == []
        assert not (state_dir / WORKER_SIDECAR_FILENAME).exists()


# ---- reconcile_and_apply (idempotency end-to-end) --------------------------


class TestReconcileAndApply:
    def test_idempotent_on_rerun(self, tmp_path: Path) -> None:
        state_dir = _build_state_dir(tmp_path, "linkedin", "idem")
        _start_running_run(state_dir)
        applied1, mutations1 = reconcile_and_apply(tmp_path)
        assert applied1 == 1
        assert len(mutations1) == 1
        # Second pass: the run is now status='abandoned' so the reconciler
        # finds nothing.
        applied2, mutations2 = reconcile_and_apply(tmp_path)
        assert applied2 == 0
        assert mutations2 == []

    def test_mixed_state_dirs(self, tmp_path: Path) -> None:
        """One zombie, one alive, one already-finished, one no-DB.

        Only the zombie is reconciled.
        """
        # zombie: running + no sidecar
        sd_zombie = _build_state_dir(tmp_path, "linkedin", "zombie")
        run_zombie = _start_running_run(sd_zombie)

        # alive: running + sidecar with live PID
        sd_alive = _build_state_dir(tmp_path, "linkedin", "alive")
        _start_running_run(sd_alive)
        write_sidecar(sd_alive, {
            "pid": _alive_pid(),
            "mode": "fresh",
            "input_mode": "concurrent",
            "brief_id": "brief-alive",
            "brief_path": str(sd_alive / "brief.json"),
            "heartbeat_at": _now_iso(),
            "launcher_version": "test",
        })

        # finished: not running
        sd_done = _build_state_dir(tmp_path, "linkedin", "done")
        run_done = _start_running_run(sd_done)
        store_done = RuntimeStateStore(sd_done / "runtime_state.sqlite3")
        store_done.finish_run(run_done, "completed")

        # no_db: state dir with no canonical SQLite
        _build_state_dir(tmp_path, "linkedin", "no_db")

        applied, mutations = reconcile_and_apply(tmp_path)
        assert applied == 1
        assert len(mutations) == 1
        assert mutations[0].state_key == "zombie"
        assert mutations[0].run_id == run_zombie

        # Verify each surface is in the expected canonical state.
        zombie_latest = read_models.latest_run_summary(
            sd_zombie / "runtime_state.sqlite3"
        )
        assert zombie_latest and zombie_latest.status == "abandoned"

        alive_latest = read_models.latest_run_summary(
            sd_alive / "runtime_state.sqlite3"
        )
        assert alive_latest and alive_latest.status == "running"

        done_latest = read_models.latest_run_summary(
            sd_done / "runtime_state.sqlite3"
        )
        assert done_latest and done_latest.status == "completed"
