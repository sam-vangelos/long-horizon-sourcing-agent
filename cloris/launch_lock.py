"""Phase 1.1: per-state-dir launch lock.

Without this lock, two Cloris UI processes can race-read the worker sidecar,
both observe "no worker," and each spawn their own — corrupting canonical
SQLite state with two writers and confusing every later stop/probe. The lock
serializes the read-sidecar + spawn critical section across Cloris processes
on the same machine.

Cross-machine / NFS flock semantics are out of scope. This guard is correct
for the desktop-app form factor only — documented as a non-goal in the
v0 spec.

Design notes:

- The lock is keyed on the per-brief state directory, not on a global lock,
  so different briefs do not block each other.
- POSIX advisory flock; the lock is process-scoped. The kernel releases it
  on file-descriptor close, including on abrupt process exit, so a crashed
  Cloris process cannot leave the lock permanently held.
- The lock file itself persists between sessions. We deliberately do NOT
  unlink it: deletion races vs other holders make POSIX flock semantics
  murky.
- Acquisition is bounded by ``timeout``. On timeout we raise
  :class:`LaunchLockTimeoutError`, which the route layer maps to HTTP 503
  ("transient unavailability — try again"). This matches the shape of the
  existing 409 (worker_already_running), which is for the steady-state
  "another worker is already running" case, vs 503 for "another launch is
  in flight, retry."

After ``Popen`` returns inside the lock, the spawned worker has not yet
written ``worker.json``. To close the residual race between Popen-return
and the worker's first ``write_sidecar`` call, the lock holder polls for
the sidecar to exist with the expected PID before releasing. See
:func:`wait_for_sidecar`.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from cloris.worker import WORKER_SIDECAR_FILENAME


_LOCK_FILENAME = ".launch.lock"
DEFAULT_LAUNCH_LOCK_TIMEOUT_S = 5.0
DEFAULT_SIDECAR_WAIT_TIMEOUT_S = 2.0
_LOCK_POLL_INTERVAL_S = 0.05
_SIDECAR_POLL_INTERVAL_S = 0.025


class LaunchLockTimeoutError(Exception):
    """Raised when :func:`state_dir_launch_lock` cannot acquire within timeout.

    Maps to HTTP 503 in the launch route. The request was valid; another
    launcher held the lock past the deadline. The user can retry.
    """

    def __init__(self, state_dir: str, timeout: float) -> None:
        super().__init__(
            f"could not acquire launch lock for {state_dir} within {timeout:.2f}s"
        )
        self.state_dir = state_dir
        self.timeout = timeout


@contextmanager
def state_dir_launch_lock(
    state_dir: Path,
    *,
    timeout: float = DEFAULT_LAUNCH_LOCK_TIMEOUT_S,
) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``<state_dir>/.launch.lock``.

    Use as ``with state_dir_launch_lock(state_dir): ...`` around any
    critical section that must not race another launcher (read-sidecar +
    spawn-worker, currently). Releases on context exit even if the body
    raises.

    Times out after ``timeout`` seconds with :class:`LaunchLockTimeoutError`.
    """

    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / _LOCK_FILENAME

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LaunchLockTimeoutError(
                        state_dir=str(state_dir), timeout=timeout
                    )
                time.sleep(_LOCK_POLL_INTERVAL_S)
            except OSError as exc:
                # EWOULDBLOCK / EAGAIN can surface as plain OSError on some
                # platforms instead of BlockingIOError; treat them as
                # "lock held; retry" identically.
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    if time.monotonic() >= deadline:
                        raise LaunchLockTimeoutError(
                            state_dir=str(state_dir), timeout=timeout
                        ) from exc
                    time.sleep(_LOCK_POLL_INTERVAL_S)
                else:
                    raise
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                # Already released (e.g., process forking semantics) — fine.
                pass
    finally:
        os.close(fd)


def wait_for_sidecar(
    state_dir: Path,
    *,
    expected_pid: int,
    timeout: float = DEFAULT_SIDECAR_WAIT_TIMEOUT_S,
) -> bool:
    """Poll for ``worker.json`` to exist with ``pid == expected_pid``.

    Closes the residual race between ``Popen`` returning and the spawned
    worker's first :func:`cloris.worker.write_sidecar` call. The caller
    holds :func:`state_dir_launch_lock` while waiting, so a second launcher
    cannot enter the critical section until the sidecar is durable.

    Returns ``True`` when the sidecar is observed with the expected PID.
    Returns ``False`` after ``timeout`` if the sidecar never materializes —
    the worker is presumed dead before its own ``write_sidecar`` ran. The
    next launcher will treat the missing sidecar as "no worker, proceed,"
    which is the correct behavior in that case.

    Defensively re-reads the sidecar JSON each iteration: a partial write
    is impossible because :func:`cloris.worker.write_sidecar` uses
    ``os.replace`` for atomicity, but malformed JSON or non-dict top-level
    are treated as "not yet ready" rather than crashing the wait.
    """

    sidecar_path = state_dir / WORKER_SIDECAR_FILENAME
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sidecar_path.exists():
            try:
                parsed = json.loads(sidecar_path.read_text())
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
            if isinstance(parsed, dict) and parsed.get("pid") == expected_pid:
                return True
        time.sleep(_SIDECAR_POLL_INTERVAL_S)
    return False
