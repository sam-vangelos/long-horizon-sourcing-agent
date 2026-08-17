"""Tests for drain-to-quiescence (Reopen Y.5.9 / F2).

F2 is the last pre-cutover safety FOUNDATION: a primitive that makes one
source quiescent — no live worker, every state dir's LATEST run terminal —
so the deferred Y.6/Y.7 write cutover can re-key candidates with nothing
mutating underneath it. These tests pin the contract the plan's TENTH +
ELEVENTH addenda specify, and in particular the two failures the adversarial
rig caught before any code:

  - the drain-DEADLOCK blocker: ``reconcile_and_apply`` has one non-test
    caller and does NOT auto-run, so the drain must call it each poll or it
    hangs waiting for runs nothing finalizes. ``test_kill_then_reconcile_*``
    and ``test_reconcile_runs_each_poll`` pin that it does.
  - Decision 1's scope contradiction: a historical NON-latest 'running' row
    can never be reconciled (the reconciler heals only the latest run), so
    the drain must NOT wait on it. ``test_stale_nonlatest_running_row_*``
    pins that scoping.

Fixture pattern follows ``tests/test_cloris_zombie_reconciliation.py``: real
state dirs under tmp_path, real canonical rows via ``RuntimeStateStore``,
real sidecars via ``cloris.worker.write_sidecar``. Worker liveness is driven
through a stateful fake clock + a fake ``is_pid_alive`` so the graceful →
hard escalation ladder is exercised deterministically without wall-clock
waits — no fabricated lifecycle values, only the real run statuses prod
produces.
"""

from __future__ import annotations

import signal
from pathlib import Path

import pytest

from cloris import reconciler
from cloris.reconciler import (
    DrainNotPausedError,
    DrainResult,
    DrainTimeoutError,
    drain_source_to_quiescence,
)
from cloris.worker import WORKER_SIDECAR_FILENAME, write_sidecar
from shared.runtime_state import read_models
from shared.runtime_state.store import RuntimeStateStore
from shared.safety.stop_reasons import RunStopReason


_DB = "runtime_state.sqlite3"
_DEAD_PID = 999_999_999  # well above any real PID ceiling — never alive.


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


def _state_dir(root: Path, source: str, key: str) -> Path:
    d = root / source / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _start_running(state_dir: Path, *, source: str, brief_id: str = "brief-x") -> int:
    store = RuntimeStateStore(state_dir / _DB)
    return store.start_run(
        source=source,
        brief_id=brief_id,
        output_dir=str(state_dir),
        mode="fresh",
    )


def _finish(state_dir: Path, run_id: int, status: str = "completed") -> None:
    RuntimeStateStore(state_dir / _DB).finish_run(
        run_id, status, stop_reason=RunStopReason.NORMAL
    )


def _sidecar(state_dir: Path, pid: int) -> None:
    write_sidecar(
        state_dir,
        {
            "pid": pid,
            "source": state_dir.parent.name,
            "mode": "fresh",
            "input_mode": "concurrent",
            "brief_id": "brief-x",
            "brief_path": str(state_dir / "brief.json"),
            "heartbeat_at": "2020-01-01T00:00:00+00:00",
            "launcher_version": "test",
        },
    )


def _status(state_dir: Path) -> str | None:
    latest = read_models.latest_run_summary(state_dir / _DB)
    return None if latest is None else latest.status


class _FakeClock:
    """Monotonic clock that only advances when ``sleep`` is called.

    Injected as both ``monotonic`` and ``sleep`` so the drain's poll cadence
    and escalation bounds are driven by the test, not wall time. ``sleep(dt)``
    advances ``now`` by ``dt``; ``monotonic()`` returns the current value.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        self.now += dt


class _Liveness:
    """Stateful fake for ``is_pid_alive`` driven by the fake clock.

    A pid is alive until ``die_at[pid]`` (a clock reading), then dead. A pid
    with no entry is treated as alive forever (so a never-dying worker drives
    the timeout / SIGKILL paths). ``calls`` records every (pid) query for
    assertions on re-enumeration.
    """

    def __init__(self, clock: _FakeClock, die_at: dict[int, float] | None = None):
        self.clock = clock
        self.die_at = dict(die_at or {})
        self.killed: set[int] = set()

    def is_alive(self, pid) -> bool:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if pid in self.killed:
            return False
        if pid <= 0:
            return False
        threshold = self.die_at.get(pid)
        if threshold is None:
            return True
        return self.clock.now < threshold


@pytest.fixture()
def patch_liveness(monkeypatch: pytest.MonkeyPatch):
    """Install a controllable ``is_pid_alive`` into the reconciler module.

    The drain reads ``is_pid_alive`` through ``cloris.reconciler`` (and so
    does ``reconcile_orphans`` via ``_classify_worker``), so patching the
    name on that module covers both the live-worker enumeration AND the
    reconciler's dead-worker classification in one place.
    """

    def _install(liveness: _Liveness) -> None:
        monkeypatch.setattr(reconciler, "is_pid_alive", liveness.is_alive)

    return _install


# --------------------------------------------------------------------------
# Quiescence already holds
# --------------------------------------------------------------------------


def test_empty_source_drains_immediately(tmp_path: Path) -> None:
    clock = _FakeClock()
    result = drain_source_to_quiescence(
        "designer",
        state_root=tmp_path,
        require_pause=False,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert isinstance(result, DrainResult)
    assert result.source == "designer"
    assert result.signaled_pids == []
    assert result.total_reconciled == 0
    # Fixpoint needs stable_polls consecutive quiescent polls (default 2).
    assert result.polls == 2


def test_terminal_run_no_sidecar_is_quiescent(tmp_path: Path, patch_liveness) -> None:
    patch_liveness(_Liveness(_FakeClock()))
    sd = _state_dir(tmp_path, "designer", "done")
    rid = _start_running(sd, source="designer")
    _finish(sd, rid, "completed")  # terminal, no worker.json
    clock = _FakeClock()

    result = drain_source_to_quiescence(
        "designer",
        state_root=tmp_path,
        require_pause=False,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert result.signaled_pids == []
    assert _status(sd) == "completed"


# --------------------------------------------------------------------------
# Live worker → graceful stop → reconcile finalizes the run
# --------------------------------------------------------------------------


def test_kill_then_reconcile_reaches_fixpoint(tmp_path: Path, patch_liveness) -> None:
    """A live worker with a 'running' run: 1st SIGTERM, the worker dies within
    the graceful window, reconcile (each poll) finalizes the stranded run, and
    the drain reaches quiescence. The run is left 'abandoned' — the reconciler's
    canonical dead-worker status — proving the per-poll reconcile ran."""

    clock = _FakeClock()
    pid = 4242
    liveness = _Liveness(clock, die_at={pid: 4.0})  # dies after ~2 polls of sleep(2)
    patch_liveness(liveness)

    sd = _state_dir(tmp_path, "designer", "live")
    _start_running(sd, source="designer")
    _sidecar(sd, pid)

    signals: list[tuple[int, int]] = []
    result = drain_source_to_quiescence(
        "designer",
        state_root=tmp_path,
        require_pause=False,
        poll_interval_s=2.0,
        graceful_bound_s=120.0,
        send_signal=lambda p, s: signals.append((p, s)),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    # The worker was SIGTERM'd (graceful) — never escalated, since it died
    # before graceful_bound elapsed.
    assert (pid, signal.SIGTERM) in signals
    assert (pid, signal.SIGKILL) not in signals
    assert signals.count((pid, signal.SIGTERM)) == 1
    # Reconcile finalized the stranded run (the deadlock-blocker fix).
    assert result.total_reconciled >= 1
    assert _status(sd) == "abandoned"
    assert pid in result.signaled_pids
    # Sidecar cleared by the drainable-lock cleanup inside reconcile_and_apply.
    assert not (sd / WORKER_SIDECAR_FILENAME).exists()


def test_reconcile_runs_each_poll(tmp_path: Path, patch_liveness) -> None:
    """Without the per-poll reconcile the run would stay 'running' forever and
    the drain would hit the timeout. We prove reconcile is wired by counting
    that the dead worker's run gets finalized even though nothing else calls
    the reconciler."""

    clock = _FakeClock()
    pid = _DEAD_PID  # already dead on poll 1
    patch_liveness(_Liveness(clock, die_at={pid: 0.0}))  # dead from t=0

    sd = _state_dir(tmp_path, "designer", "zombie")
    _start_running(sd, source="designer")
    _sidecar(sd, pid)
    assert _status(sd) == "running"

    result = drain_source_to_quiescence(
        "designer",
        state_root=tmp_path,
        require_pause=False,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert result.total_reconciled >= 1
    assert _status(sd) == "abandoned"


# --------------------------------------------------------------------------
# Graceful → hard escalation ladder (Decision 2)
# --------------------------------------------------------------------------


def test_escalates_to_second_sigterm_then_sigkill(tmp_path: Path, patch_liveness) -> None:
    """A worker that ignores the 1st SIGTERM past the graceful bound gets a 2nd
    SIGTERM; if it still ignores that past the hard bound it gets SIGKILL. The
    SIGKILL is the modeled kill — the fake flips the pid dead on SIGKILL — after
    which reconcile finalizes the run and the drain reaches quiescence."""

    clock = _FakeClock()
    pid = 7777
    liveness = _Liveness(clock)  # never dies on its own

    # Model SIGKILL actually killing: when the drain SIGKILLs, mark dead.
    sent: list[tuple[int, int]] = []

    def _send(p: int, s: int) -> None:
        sent.append((p, s))
        if s == signal.SIGKILL:
            liveness.killed.add(p)

    patch_liveness(liveness)

    sd = _state_dir(tmp_path, "designer", "stubborn")
    _start_running(sd, source="designer")
    _sidecar(sd, pid)

    result = drain_source_to_quiescence(
        "designer",
        state_root=tmp_path,
        require_pause=False,
        poll_interval_s=10.0,
        graceful_bound_s=120.0,
        hard_bound_s=30.0,
        overall_timeout_s=100_000.0,
        send_signal=_send,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    kinds = [s for (p, s) in sent if p == pid]
    assert kinds.count(signal.SIGTERM) == 2  # graceful, then hard
    assert kinds.count(signal.SIGKILL) == 1  # last resort
    # Ordering: first SIGTERM precedes the second precedes the SIGKILL.
    term_idxs = [i for i, (p, s) in enumerate(sent) if p == pid and s == signal.SIGTERM]
    kill_idx = next(i for i, (p, s) in enumerate(sent) if p == pid and s == signal.SIGKILL)
    assert term_idxs[0] < term_idxs[1] < kill_idx
    assert _status(sd) == "abandoned"
    assert result.total_reconciled >= 1


def test_no_second_sigterm_before_graceful_bound(tmp_path: Path, patch_liveness) -> None:
    """If the worker dies just after the 1st SIGTERM but before the graceful
    bound, it must NOT receive a 2nd SIGTERM — the graceful path is honored."""

    clock = _FakeClock()
    pid = 5555
    # Dies at t=5; with poll_interval 2 it's gone by poll 3, well before the
    # 120s graceful bound.
    liveness = _Liveness(clock, die_at={pid: 5.0})
    patch_liveness(liveness)

    sd = _state_dir(tmp_path, "designer", "polite")
    _start_running(sd, source="designer")
    _sidecar(sd, pid)

    sent: list[tuple[int, int]] = []
    drain_source_to_quiescence(
        "designer",
        state_root=tmp_path,
        require_pause=False,
        poll_interval_s=2.0,
        graceful_bound_s=120.0,
        send_signal=lambda p, s: sent.append((p, s)),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    kinds = [s for (p, s) in sent if p == pid]
    assert kinds == [signal.SIGTERM]  # exactly one, graceful only


# --------------------------------------------------------------------------
# Decision 1: scope = latest run per state dir
# --------------------------------------------------------------------------


def test_stale_nonlatest_running_row_does_not_block(tmp_path: Path, patch_liveness) -> None:
    """A state dir whose LATEST run is terminal but which has an older stranded
    'running' row must still drain: Decision 1 scopes quiescence to the latest
    run per state dir (the only row the reconciler can heal). Waiting on the
    historical row would hang forever."""

    patch_liveness(_Liveness(_FakeClock()))
    sd = _state_dir(tmp_path, "designer", "history")
    older = _start_running(sd, source="designer")  # leave it 'running'
    newer = _start_running(sd, source="designer")
    _finish(sd, newer, "completed")  # latest is terminal; older still running

    # Sanity: the older row is genuinely still 'running' in the DB.
    with RuntimeStateStore(sd / _DB).connect() as conn:  # type: ignore[attr-defined]
        rows = {
            r["id"]: r["status"]
            for r in conn.execute("SELECT id, status FROM runs").fetchall()
        }
    assert rows[older] == "running"
    assert rows[newer] == "completed"

    clock = _FakeClock()
    result = drain_source_to_quiescence(
        "designer",
        state_root=tmp_path,
        require_pause=False,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    # Drains (does not time out); the stale historical row is untouched.
    assert isinstance(result, DrainResult)
    with RuntimeStateStore(sd / _DB).connect() as conn:  # type: ignore[attr-defined]
        still = {
            r["id"]: r["status"]
            for r in conn.execute("SELECT id, status FROM runs").fetchall()
        }
    assert still[older] == "running"  # cosmetic residue, by design


# --------------------------------------------------------------------------
# Source isolation
# --------------------------------------------------------------------------


def test_other_source_live_worker_does_not_block(tmp_path: Path, patch_liveness) -> None:
    """A live worker under a DIFFERENT source must not keep this source from
    reaching quiescence. The drain is per-source by construction."""

    clock = _FakeClock()
    gh_pid = 8888
    liveness = _Liveness(clock)  # github worker never dies
    patch_liveness(liveness)

    # designer: nothing live. github: a live worker with a running run.
    gh = _state_dir(tmp_path, "github", "busy")
    _start_running(gh, source="github")
    _sidecar(gh, gh_pid)

    sent: list[tuple[int, int]] = []
    result = drain_source_to_quiescence(
        "designer",
        state_root=tmp_path,
        require_pause=False,
        send_signal=lambda p, s: sent.append((p, s)),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert isinstance(result, DrainResult)
    # The github worker was never signaled by a designer drain.
    assert sent == []
    assert result.signaled_pids == []
    # And github's run is left untouched — drain didn't reconcile it away as a
    # live worker (it's alive); the row stays 'running'.
    assert _status(gh) == "running"


# --------------------------------------------------------------------------
# Re-enumerate, not snapshot
# --------------------------------------------------------------------------


def test_worker_appearing_after_first_poll_is_caught(tmp_path: Path, patch_liveness) -> None:
    """A worker that materializes AFTER the first poll (slipped past the gate
    just before the pause armed) must still be enumerated, signaled, and waited
    on — the drain re-enumerates every poll rather than snapshotting once."""

    clock = _FakeClock()
    pid = 6363
    liveness = _Liveness(clock, die_at={pid: 1_000.0})  # alive a long while
    patch_liveness(liveness)

    sd = _state_dir(tmp_path, "designer", "late")

    sent: list[tuple[int, int]] = []

    # Monkeypatch enumerate so the worker only exists from the 2nd poll on.
    real_enum = reconciler.enumerate_state_dirs
    calls = {"n": 0}

    def _staged_enum(state_root=None):
        calls["n"] += 1
        if calls["n"] <= 1:
            # First enumeration sees nothing — drain would think it's done.
            return iter(())
        return real_enum(state_root)

    # Materialize the worker's row + sidecar up front; gating is via _staged_enum.
    _start_running(sd, source="designer")
    _sidecar(sd, pid)

    # Make the worker die shortly after it's first seen so the drain converges
    # instead of timing out: flip die_at once enumeration starts returning it.
    def _send(p: int, s: int) -> None:
        sent.append((p, s))
        # First SIGTERM models a cooperative stop: schedule death soon after.
        if s == signal.SIGTERM and p == pid:
            liveness.die_at[p] = clock.now + 1.0

    import cloris.reconciler as rec_mod
    import pytest as _pytest  # local alias to avoid shadowing
    mp = _pytest.MonkeyPatch()
    mp.setattr(rec_mod, "enumerate_state_dirs", _staged_enum)
    try:
        result = drain_source_to_quiescence(
            "designer",
            state_root=tmp_path,
            require_pause=False,
            poll_interval_s=2.0,
            send_signal=_send,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    finally:
        mp.undo()

    # The late worker WAS signaled despite being invisible on poll 1.
    assert (pid, signal.SIGTERM) in sent
    assert pid in result.signaled_pids
    assert _status(sd) == "abandoned"
    assert calls["n"] >= 2  # proves more than one enumeration round ran


# --------------------------------------------------------------------------
# require_pause guard (the spawn-race close)
# --------------------------------------------------------------------------


def test_refuses_when_pause_not_armed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With require_pause=True and the pause not armed, the drain refuses up
    front (DrainNotPausedError) rather than chasing a moving target."""

    monkeypatch.setattr(reconciler, "_source_launch_pause_armed", lambda s: False)
    clock = _FakeClock()
    with pytest.raises(DrainNotPausedError):
        drain_source_to_quiescence(
            "designer",
            state_root=tmp_path,
            require_pause=True,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )


def test_proceeds_when_pause_armed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch_liveness) -> None:
    """With the pause armed, require_pause=True proceeds normally."""

    patch_liveness(_Liveness(_FakeClock()))
    monkeypatch.setattr(reconciler, "_source_launch_pause_armed", lambda s: True)
    clock = _FakeClock()
    result = drain_source_to_quiescence(
        "designer",
        state_root=tmp_path,
        require_pause=True,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert isinstance(result, DrainResult)


def test_pause_armed_check_reads_real_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_source_launch_pause_armed reads the SAME orchestration store the spawn
    gate consults — an out-of-process arm is seen here. Pins the integration
    against F1's persisted pause, not a stub."""

    from shared.runtime_state.orchestration_store import OrchestrationStateStore

    orch_db = tmp_path / "orchestration.sqlite3"
    monkeypatch.setattr(
        "shared.output_paths.resolve_orchestration_db_path", lambda: orch_db
    )
    # Not armed yet → False.
    assert reconciler._source_launch_pause_armed("designer") is False
    # Arm it out-of-band (as tools/pause_source_launches.py would) → True.
    OrchestrationStateStore(orch_db).set_source_pause(
        "designer", paused=True, armed_by="test", reason="cutover"
    )
    assert reconciler._source_launch_pause_armed("designer") is True
    # A different source stays unpaused.
    assert reconciler._source_launch_pause_armed("github") is False


# --------------------------------------------------------------------------
# Timeout
# --------------------------------------------------------------------------


def test_times_out_on_undying_worker(tmp_path: Path, patch_liveness) -> None:
    """A worker that never dies (even SIGKILL is dropped by the fake) drives the
    overall timeout; the error carries the still-live pid and non-terminal key."""

    clock = _FakeClock()
    pid = 9191
    liveness = _Liveness(clock)  # never dies, SIGKILL not modeled as fatal here
    patch_liveness(liveness)

    sd = _state_dir(tmp_path, "designer", "immortal")
    _start_running(sd, source="designer")
    _sidecar(sd, pid)

    with pytest.raises(DrainTimeoutError) as exc:
        drain_source_to_quiescence(
            "designer",
            state_root=tmp_path,
            require_pause=False,
            poll_interval_s=5.0,
            graceful_bound_s=10.0,
            hard_bound_s=5.0,
            overall_timeout_s=60.0,
            send_signal=lambda p, s: None,  # signals dropped — worker immortal
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    assert exc.value.source == "designer"
    assert pid in exc.value.live_pids
    assert "immortal" in exc.value.nonterminal_keys
