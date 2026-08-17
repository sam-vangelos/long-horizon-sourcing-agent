"""Zombie-run reconciler.

Background
----------
A run row in canonical state goes ``status='running'`` at start_run and
stays that way until the orchestrator calls finish_run. If the worker
process dies hard — Mac sleep evicting the subprocess, kernel OOM,
``kill -9``, host reboot, anything — finish_run never fires and the row
is stranded. The Cloris UI faithfully renders ``status='running'`` and
shows a "Working" status pill on a brief that has been dead for days.

That contradiction is the deepest UX-as-data-model rot in the product:
the surface tells the truth about the row, but the row hasn't been
reconciled with worker-process reality.

Contract
--------
This module is the reconciler. It is **pure** in the read sense: it
walks state directories, opens canonical SQLite **read-only** via
:mod:`shared.runtime_state.read_models`, and inspects ``worker.json``
sidecars for liveness. It produces a list of :class:`Mutation`
records — what needs to change — without writing anything itself.

The caller (cloris/api.py) is responsible for executing the mutations
through the canonical write path
(:class:`shared.runtime_state.store.RuntimeStateStore.finish_run`).
This separation means:

  - Reconciliation can be unit-tested end-to-end without a live FastAPI.
  - The read-only contract on ``cloris/control_plane.py`` (no
    ``RuntimeStateStore`` import; enforced by tests) stays intact —
    this module is the only place that depends on the writer, and it
    isolates that dependency in the executor function.

Decision logic
--------------
For each state_dir with a canonical SQLite present:

  1. Read latest_run. If ``status != 'running'``, no mutation.
  2. Read worker sidecar (``worker.json``).
  3. Conservative trigger — only mutate when we're certain the worker
     is gone:
       - sidecar missing entirely → ``MISSING_SIDECAR``
       - sidecar present but ``pid`` non-int / non-positive → ``BAD_SIDECAR``
       - sidecar pid dead per :func:`cloris.worker.is_pid_alive` → ``PID_DEAD``
     "alive_silent" (PID alive but heartbeat stale) is **NOT** reconciled
     — a worker can be alive and silent during a long captcha wait or a
     suspended laptop. Killing those would create a worse failure mode.

Idempotency
-----------
A re-run of :func:`reconcile_orphans` after a previous reconciliation
returns an empty mutation list, because the runs that were marked
``status='abandoned'`` no longer match the ``status='running'`` filter.
Tests should pin this property explicitly.
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from cloris.control_plane import (
    enumerate_state_dirs,
    read_worker_sidecar,
    is_runtime_state_corrupt,
)
from cloris.worker import WORKER_SIDECAR_FILENAME, is_pid_alive
from shared.runtime_state import read_models
from shared.safety.stop_reasons import RunStopReason


_RUNTIME_DB_FILENAME = "runtime_state.sqlite3"


# Status to assign when a zombie is reconciled. We use ``abandoned`` rather
# than ``interrupted`` because:
#   - ``interrupted`` already means "operator-initiated stop or
#     cooperative pause" in the existing taxonomy
#     (``StopReason.OPERATOR_STOP`` / ``OPERATOR_PAUSE``).
#   - ``abandoned`` is unambiguous: nobody chose to stop this; the
#     worker process simply went away.
ABANDONED_STATUS = "abandoned"


ReconcileReason = Literal[
    "missing_sidecar",  # worker.json doesn't exist
    "bad_sidecar",      # worker.json exists but pid is missing / non-int / non-positive
    "pid_dead",         # worker.json's pid is a real int but the process is gone
]

TERMINAL_BROWSER_LOCK_STOP_REASONS = frozenset({
    RunStopReason.BROWSER_DISCONNECT_UNRECOVERED,
})
TERMINAL_DRAINABLE_STOP_REASONS = frozenset({
    RunStopReason.BROWSER_DISCONNECT_UNRECOVERED,
    RunStopReason.API_BUDGET_EXHAUSTED,
    RunStopReason.FATAL_RUNTIME_ERROR,
    RunStopReason.WORKER_MISSING,
})
TERMINAL_RUN_STATUSES = frozenset({
    "abandoned",
    "completed",
    "error",
    "failed",
    "governor_limit_reached",
    "interrupted",
    "succeeded",
})


@dataclass(frozen=True)
class Mutation:
    """One canonical-state mutation the caller should execute.

    The reconciler emits these; cloris/api.py applies them through
    :class:`shared.runtime_state.store.RuntimeStateStore`. Splitting
    "decide" from "apply" keeps the reconciler unit-testable without
    a writer dependency.
    """

    source: str
    state_key: str
    state_dir: Path
    run_id: int
    new_status: str
    stop_reason: str
    reason: ReconcileReason
    detected_at: str  # ISO-8601 UTC


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_worker(state_dir: Path) -> ReconcileReason | None:
    """Return a reconcile reason if the worker is gone, else None.

    Mirrors the worker_state derivation in
    :func:`cloris.control_plane.aggregate_status` but conservatively —
    only the genuinely-dead branches map to a reason. ``alive`` and
    ``alive_silent`` both return ``None`` (do not reconcile).
    """
    sidecar = read_worker_sidecar(state_dir)
    if sidecar is None:
        return "missing_sidecar"
    pid_raw = sidecar.get("pid")
    if not isinstance(pid_raw, int) or isinstance(pid_raw, bool) or pid_raw <= 0:
        return "bad_sidecar"
    if not is_pid_alive(pid_raw):
        return "pid_dead"
    return None  # alive — leave alone (covers alive_silent too)


def reconcile_orphans(state_root: Path | None = None) -> list[Mutation]:
    """Walk every state dir; emit mutations for runs whose workers are gone.

    Pure function in the read sense — opens canonical SQLite read-only,
    reads sidecars, returns mutations. The caller executes them.

    Args:
        state_root: optional override for the state root directory. Defaults
            to ``shared.output_paths.STATE_ROOT`` via :func:`enumerate_state_dirs`.

    Returns:
        list of :class:`Mutation` records, possibly empty. Sorted by
        ``(source, state_key, run_id)`` for deterministic ordering.
    """
    mutations: list[Mutation] = []
    now = _utc_now_iso()
    for source, state_dir in enumerate_state_dirs(state_root):
        db_path = state_dir / _RUNTIME_DB_FILENAME
        if not db_path.exists() or is_runtime_state_corrupt(db_path):
            continue
        latest = read_models.latest_run_summary(db_path)
        if latest is None or latest.status != "running":
            continue
        reason = _classify_worker(state_dir)
        if reason is None:
            continue
        mutations.append(
            Mutation(
                source=source,
                state_key=state_dir.name,
                state_dir=state_dir,
                run_id=latest.id,
                new_status=ABANDONED_STATUS,
                stop_reason=RunStopReason.WORKER_MISSING,
                reason=reason,
                detected_at=now,
            )
        )
    mutations.sort(key=lambda m: (m.source, m.state_key, m.run_id))
    return mutations


def apply_mutations(mutations: list[Mutation]) -> int:
    """Execute reconciler mutations against canonical state.

    Opens a fresh :class:`RuntimeStateStore` per state_dir, calls
    ``finish_run`` with the abandoned status. Returns the number of
    mutations actually applied (may be less than the input length if a
    run was already finalized between read and write — a benign race
    we tolerate by re-reading the row and skipping when status has
    moved off 'running').

    Importing ``RuntimeStateStore`` here (and not in
    ``cloris.control_plane``) preserves the read-only contract pinned
    by tests/test_cloris_status_aggregation.py.
    """
    # Lazy import keeps the writer dependency out of read-only call sites
    # that import ``cloris.reconciler`` for type references only.
    from shared.runtime_state.store import RuntimeStateStore

    applied = 0
    for m in mutations:
        db_path = m.state_dir / _RUNTIME_DB_FILENAME
        # Re-check status before write to avoid finalizing a run that
        # legitimately transitioned to terminal between read and apply.
        latest = read_models.latest_run_summary(db_path)
        if latest is None or latest.id != m.run_id or latest.status != "running":
            continue
        store = RuntimeStateStore(db_path)
        store.finish_run(m.run_id, m.new_status, stop_reason=m.stop_reason)
        applied += 1
    return applied


def _sidecar_pid(state_dir: Path) -> int | None:
    sidecar = read_worker_sidecar(state_dir)
    if sidecar is None:
        return None
    pid_raw = sidecar.get("pid")
    if not isinstance(pid_raw, int) or isinstance(pid_raw, bool) or pid_raw <= 0:
        return None
    return pid_raw


def _unlink_sidecar_if_pid(state_dir: Path, pid: int) -> bool:
    sidecar = read_worker_sidecar(state_dir)
    if sidecar is None or sidecar.get("pid") != pid:
        return False
    try:
        (state_dir / WORKER_SIDECAR_FILENAME).unlink()
    except FileNotFoundError:
        return False
    return True


def cleanup_terminal_browser_lock(
    state_dir: Path,
    *,
    send_signal: Callable[[int, int], None] | None = None,
) -> bool:
    """Quietly clear a sidecar lock left after terminal browser recovery.

    This does not mutate canonical runtime state. It only handles the case
    where SQLite already says the run is terminal and the remaining
    ``worker.json`` PID is therefore a stale launch lock. The implementation
    sends best-effort SIGTERM before removing the sidecar so a lingering
    post-run process has a chance to exit, but zombie PIDs cannot be killed;
    the sidecar is still cleared because the run is already ended.
    """

    db_path = state_dir / _RUNTIME_DB_FILENAME
    if not db_path.exists() or is_runtime_state_corrupt(db_path):
        return False
    latest = read_models.latest_run_summary(db_path)
    if latest is None:
        return False
    if latest.status not in TERMINAL_RUN_STATUSES:
        return False
    if latest.stop_reason not in TERMINAL_BROWSER_LOCK_STOP_REASONS:
        return False

    pid = _sidecar_pid(state_dir)
    if pid is None:
        return False

    if is_pid_alive(pid):
        sender = send_signal or os.kill
        try:
            sender(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
        except OSError:
            pass

    return _unlink_sidecar_if_pid(state_dir, pid)


def cleanup_drainable_terminal_lock(
    state_dir: Path,
    *,
    send_signal: Callable[[int, int], None] | None = None,
) -> bool:
    """Clear a launch-blocking sidecar when canonical state is terminal.

    Browser-recovery failures keep their existing SIGTERM behavior. Other
    terminal runs only lose the sidecar lock; the process is not signaled,
    because it may be finishing report/report-copy work that should not block
    the next launch.
    """

    db_path = state_dir / _RUNTIME_DB_FILENAME
    if not db_path.exists() or is_runtime_state_corrupt(db_path):
        return False
    latest = read_models.latest_run_summary(db_path)
    if latest is None or latest.status not in TERMINAL_RUN_STATUSES:
        return False

    pid = _sidecar_pid(state_dir)
    if pid is None:
        return False

    if latest.stop_reason == RunStopReason.BROWSER_DISCONNECT_UNRECOVERED:
        return cleanup_terminal_browser_lock(state_dir, send_signal=send_signal)

    if latest.stop_reason not in TERMINAL_DRAINABLE_STOP_REASONS and latest.status not in {
        "completed",
        "succeeded",
        "error",
        "failed",
        "interrupted",
    }:
        return False

    return _unlink_sidecar_if_pid(state_dir, pid)


def cleanup_terminal_browser_locks(state_root: Path | None = None) -> int:
    """Clear all terminal browser-recovery sidecar locks under state_root."""

    cleaned = 0
    for _, state_dir in enumerate_state_dirs(state_root):
        if cleanup_terminal_browser_lock(state_dir):
            cleaned += 1
    return cleaned


def cleanup_drainable_terminal_locks(state_root: Path | None = None) -> int:
    """Clear all terminal sidecar locks that cannot own active launch work."""

    cleaned = 0
    for _, state_dir in enumerate_state_dirs(state_root):
        if cleanup_drainable_terminal_lock(state_dir):
            cleaned += 1
    return cleaned


def reconcile_and_apply(state_root: Path | None = None) -> tuple[int, list[Mutation]]:
    """Convenience: do the full reconcile-then-apply pass.

    Returns (applied_count, mutations) so callers can log / surface the
    forensic detail of what was reconciled. The mutations list is the
    reconciler's pre-apply view — it may include rows that the
    re-check rejected, in which case applied < len(mutations).
    """
    mutations = reconcile_orphans(state_root)
    applied = apply_mutations(mutations)
    cleanup_drainable_terminal_locks(state_root)
    return applied, mutations


# ---------------------------------------------------------------------------
# Drain-to-quiescence (Reopen Y.5.9 / F2) — the last pre-cutover safety
# foundation. Builds the PRIMITIVE only; the gate (Y.5.10) and the Y.6/Y.7
# write cutover that would CALL this stay out of scope (deferred,
# LinkedIn-production, mid-frontend-churn).
#
# Contract (from the plan's TENTH + ELEVENTH addenda, every anchor
# hand-verified against this tree before coding):
#
#   The drain makes ONE source quiescent — no live writer, no non-terminal
#   latest run — so the write cutover can re-key candidates with nothing
#   mutating underneath it. It:
#
#     (a) enumerates live workers for the source via the worker.json sidecar
#         pid + ``is_pid_alive`` (reusing ``enumerate_state_dirs`` /
#         ``_sidecar_pid``), re-enumerating EVERY poll (never a one-shot
#         snapshot — a worker the pause didn't catch in time could still be
#         coming up);
#     (b) signals each live worker graceful-then-hard (Decision 2):
#         1st SIGTERM (the orchestrator's ``stop_event``,
#         ``linkedin/session_orchestrator.py:335``) → wait a bounded window
#         → 2nd SIGTERM (``:332`` → ``raise KeyboardInterrupt`` so
#         ``finish_run`` still flushes) → SIGKILL last resort. The bound
#         exists because the 1st SIGTERM's ``stop_event`` is only re-checked
#         at coarse ``run_dormant_loop(stop_event, 3600)`` boundaries
#         (``:395``), so a mid-dormant worker can take ~1h to stop
#         gracefully — too slow to block a production cutover window on;
#     (c) runs ``reconcile_and_apply`` ON EACH POLL — load-bearing: it has
#         exactly one non-test caller (``cloris/api/_monolith.py:618``, an
#         HTTP endpoint) and does NOT auto-run, so without this the drain
#         would block forever waiting for runs nothing finalizes (the
#         drain-deadlock blocker the rig caught). Reconcile is what turns a
#         just-killed worker's stranded ``running`` row terminal;
#     (d) BLOCKS until both halves hold as a FIXPOINT — stable across
#         ``stable_polls`` consecutive iterations — every live source PID
#         dead AND the LATEST run per state-dir terminal (Decision 1 scope:
#         the same ``latest_run_summary`` row ``reconcile_orphans`` heals;
#         a stale historical non-latest 'running' row is deliberately NOT
#         waited on — the reconciler can't finalize it, so waiting would
#         hang forever). Bounded by ``overall_timeout_s``.
#
# RACE this closes: a worker could SPAWN mid-drain. The drain therefore
# REFUSES to start unless F1's persisted pause is armed for the source
# (no new spawns past the gate, ``_monolith.py:2456``), and re-enumerates
# every poll so a worker that slipped through just before the pause armed
# is still caught, signaled, and waited on.
# ---------------------------------------------------------------------------


_DRAIN_POLL_INTERVAL_S = 2.0
_DRAIN_GRACEFUL_BOUND_S = 120.0  # Decision 2: wait this long after the 1st
#   SIGTERM before escalating to the 2nd (the orchestrator's hard-stop).
_DRAIN_HARD_BOUND_S = 30.0  # after the 2nd SIGTERM, wait this long before SIGKILL.
_DRAIN_OVERALL_TIMEOUT_S = 1800.0  # 30m overall cap on one source's drain.
_DRAIN_STABLE_POLLS = 2  # fixpoint = both halves hold this many polls running.


class DrainNotPausedError(RuntimeError):
    """Raised when a drain is requested for a source whose launch pause is
    not armed. Draining without the pause armed is unsafe: a new worker
    could spawn past the gate mid-drain and the fixpoint would chase a
    moving target. The caller must arm F1's pause first (Y.5.6)."""


class DrainTimeoutError(RuntimeError):
    """Raised when a source does not reach the quiescence fixpoint within
    ``overall_timeout_s``. Carries the last-seen live PIDs and non-terminal
    state keys so the operator can see what refused to drain."""

    def __init__(self, source: str, live_pids: list[int], nonterminal_keys: list[str]):
        self.source = source
        self.live_pids = live_pids
        self.nonterminal_keys = nonterminal_keys
        super().__init__(
            f"drain of source {source!r} timed out: "
            f"live_pids={live_pids} nonterminal_state_keys={nonterminal_keys}"
        )


@dataclass
class _WorkerSignalState:
    """Per-PID escalation bookkeeping across drain polls.

    Escalation is TIME-based, not poll-count-based: we record the monotonic
    clock at the 1st and 2nd SIGTERM so a slow poll cadence can't skip a
    stage, and a PID that dies between polls simply never advances.
    """

    first_sigterm_at: float | None = None
    second_sigterm_at: float | None = None
    sigkilled: bool = False


@dataclass(frozen=True)
class DrainResult:
    """Forensic outcome of a successful drain (the fixpoint was reached).

    ``polls`` is how many iterations ran; ``total_reconciled`` is the sum of
    runs the per-poll ``reconcile_and_apply`` finalized; ``signaled_pids``
    is every PID the drain ever signaled (for the operator log).
    """

    source: str
    polls: int
    total_reconciled: int
    signaled_pids: list[int] = field(default_factory=list)


def _live_workers_for_source(
    source: str,
    state_root: Path | None,
) -> dict[str, int]:
    """Return ``{state_key: pid}`` for every LIVE worker of ``source``.

    Keyed off the directory-source from :func:`enumerate_state_dirs` (the
    same authority :func:`reconcile_orphans` uses), not the sidecar's own
    ``source`` field. A state dir with no sidecar, a non-int/dead pid, or a
    pid that is not alive contributes nothing. Re-read fresh every poll.
    """

    live: dict[str, int] = {}
    for src, state_dir in enumerate_state_dirs(state_root):
        if src != source:
            continue
        pid = _sidecar_pid(state_dir)
        if pid is None:
            continue
        if is_pid_alive(pid):
            live[state_dir.name] = pid
    return live


def _nonterminal_latest_runs_for_source(
    source: str,
    state_root: Path | None,
) -> list[str]:
    """Return state_keys whose LATEST run is non-terminal, for ``source``.

    Decision 1 scope: only the ``latest_run_summary`` row per state dir —
    the exact row :func:`reconcile_orphans` can finalize. A historical
    non-latest ``running`` row is intentionally ignored; the reconciler
    cannot heal it (``reconciler.py`` heals only the latest run), so
    waiting on it would hang the drain forever. A missing / corrupt DB, or
    a state dir with no run yet, counts as terminal (nothing live to wait
    on).
    """

    nonterminal: list[str] = []
    for src, state_dir in enumerate_state_dirs(state_root):
        if src != source:
            continue
        db_path = state_dir / _RUNTIME_DB_FILENAME
        if not db_path.exists() or is_runtime_state_corrupt(db_path):
            continue
        latest = read_models.latest_run_summary(db_path)
        if latest is None:
            continue
        if latest.status not in TERMINAL_RUN_STATUSES:
            nonterminal.append(state_dir.name)
    return nonterminal


def _signal_live_workers(
    live: dict[str, int],
    states: dict[int, _WorkerSignalState],
    *,
    now: float,
    graceful_bound_s: float,
    hard_bound_s: float,
    send_signal: Callable[[int, int], None],
) -> None:
    """Advance the graceful-then-hard escalation for each live PID.

    Decision 2, per PID, time-gated off the monotonic ``now``:

      - never signaled yet            → 1st SIGTERM (set ``stop_event``).
      - 1st SIGTERM > graceful_bound  → 2nd SIGTERM (KeyboardInterrupt so
                                        ``finish_run`` flushes).
      - 2nd SIGTERM > hard_bound      → SIGKILL (last resort).

    Idempotent within a stage: a PID already past the 1st SIGTERM is not
    re-SIGTERM'd until its bound elapses; an already-SIGKILLed PID is left
    alone. ``ProcessLookupError`` (the PID died between enumerate and
    signal) is swallowed — that is the success case.
    """

    for pid in live.values():
        st = states.setdefault(pid, _WorkerSignalState())
        if st.first_sigterm_at is None:
            _safe_signal(send_signal, pid, signal.SIGTERM)
            st.first_sigterm_at = now
            continue
        if st.second_sigterm_at is None:
            if now - st.first_sigterm_at >= graceful_bound_s:
                _safe_signal(send_signal, pid, signal.SIGTERM)
                st.second_sigterm_at = now
            continue
        if not st.sigkilled and now - st.second_sigterm_at >= hard_bound_s:
            _safe_signal(send_signal, pid, signal.SIGKILL)
            st.sigkilled = True


def _safe_signal(
    send_signal: Callable[[int, int], None], pid: int, sig: int
) -> None:
    """Send ``sig`` to ``pid``, swallowing the benign races.

    A PID that has already exited (``ProcessLookupError``) is the success
    case, not an error. ``PermissionError`` / other ``OSError`` are logged
    by being swallowed too — the fixpoint's ``is_pid_alive`` check is the
    real arbiter of whether the worker is gone, not the signal's return.
    """

    try:
        send_signal(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass
    except OSError:
        pass


def drain_source_to_quiescence(
    source: str,
    *,
    state_root: Path | None = None,
    require_pause: bool = True,
    poll_interval_s: float = _DRAIN_POLL_INTERVAL_S,
    graceful_bound_s: float = _DRAIN_GRACEFUL_BOUND_S,
    hard_bound_s: float = _DRAIN_HARD_BOUND_S,
    overall_timeout_s: float = _DRAIN_OVERALL_TIMEOUT_S,
    stable_polls: int = _DRAIN_STABLE_POLLS,
    send_signal: Callable[[int, int], None] | None = None,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> DrainResult:
    """Drain ONE source to quiescence and return when it is a fixpoint.

    Quiescence = no live worker for ``source`` AND every state dir's LATEST
    run terminal — held STABLE across ``stable_polls`` consecutive polls so
    a worker that dies one poll but whose run isn't yet reconciled doesn't
    read as quiescent prematurely.

    Each poll: (1) re-enumerate live workers + signal them graceful-then-
    hard; (2) ``reconcile_and_apply`` to finalize just-died workers' runs;
    (3) re-check both halves; advance or reset the stability counter.

    Args:
        source: the source to drain (must be in ``known_sources()``).
        state_root: state-tree override (tests pass tmp_path).
        require_pause: refuse unless F1's launch pause is armed for
            ``source`` (the spawn-race guard). Tests that don't exercise the
            spawn gate pass ``False``.
        poll_interval_s: sleep between polls.
        graceful_bound_s: wait after the 1st SIGTERM before the 2nd.
        hard_bound_s: wait after the 2nd SIGTERM before SIGKILL.
        overall_timeout_s: hard cap; raises :class:`DrainTimeoutError`.
        stable_polls: consecutive quiescent polls required for the fixpoint.
        send_signal / monotonic / sleep: injectable seams (default
            ``os.kill`` / ``time.monotonic`` / ``time.sleep``) so the
            escalation ladder is testable without wall-clock waits.

    Raises:
        DrainNotPausedError: ``require_pause`` and the pause is not armed.
        DrainTimeoutError: quiescence not reached within the overall cap.
    """

    sig = send_signal or os.kill
    clock = monotonic or time.monotonic
    nap = sleep or time.sleep

    if require_pause and not _source_launch_pause_armed(source):
        raise DrainNotPausedError(
            f"refusing to drain {source!r}: launch pause not armed (arm F1 "
            f"first so no worker can spawn past the gate mid-drain)"
        )

    states: dict[int, _WorkerSignalState] = {}
    signaled: set[int] = set()
    total_reconciled = 0
    stable = 0
    polls = 0
    start = clock()
    deadline = start + overall_timeout_s

    while True:
        polls += 1
        now = clock()

        live = _live_workers_for_source(source, state_root)
        if live:
            _signal_live_workers(
                live,
                states,
                now=now,
                graceful_bound_s=graceful_bound_s,
                hard_bound_s=hard_bound_s,
                send_signal=sig,
            )
            signaled.update(live.values())

        applied, _ = reconcile_and_apply(state_root)
        total_reconciled += applied

        # Re-check AFTER reconcile so a worker we just killed and whose run
        # we just finalized can count as quiescent in the same poll.
        live_after = _live_workers_for_source(source, state_root)
        nonterminal = _nonterminal_latest_runs_for_source(source, state_root)

        if not live_after and not nonterminal:
            stable += 1
            if stable >= stable_polls:
                return DrainResult(
                    source=source,
                    polls=polls,
                    total_reconciled=total_reconciled,
                    signaled_pids=sorted(signaled),
                )
        else:
            stable = 0

        if clock() >= deadline:
            raise DrainTimeoutError(
                source,
                live_pids=sorted(live_after.values()),
                nonterminal_keys=sorted(nonterminal),
            )

        nap(poll_interval_s)


def _source_launch_pause_armed(source: str) -> bool:
    """Return whether F1's persisted launch pause is armed for ``source``.

    Reads the same orchestration store the spawn gate consults
    (``_monolith.py:2456``). Lazy-imported so ``cloris.reconciler`` stays
    importable by read-only call sites that don't want the orchestration
    store dependency. Fail-CLOSED: if the pause state can't be read, we
    report NOT armed so :func:`drain_source_to_quiescence` refuses rather
    than draining against a possibly-live spawn gate.
    """

    try:
        from shared.output_paths import resolve_orchestration_db_path
        from shared.runtime_state.orchestration_store import (
            OrchestrationStateStore,
        )

        return OrchestrationStateStore(
            resolve_orchestration_db_path()
        ).is_source_paused(source)
    except Exception:  # noqa: BLE001 — unreadable pause => treat as NOT armed
        return False
