"""Per-state-dir worker heartbeat updater.

Cloris's ``worker.json`` sidecar carries a ``heartbeat_at`` field that
``cloris.worker.build_sidecar`` initializes to the worker's start time and
never updates. The aggregator therefore cannot distinguish "alive and
working" from "alive but suspended/hung/silent." :func:`bump_heartbeat`
fixes that: the orchestrator calls it from canonical write checkpoints
(start_run, finish_attempt_*, upsert_work_unit, finish_run), so a stale
``heartbeat_at`` literally means "no useful work for N minutes."

Why checkpoint-driven, not timer-driven: a free-running thread updates
heartbeat even when nothing useful is happening (long captcha wait,
sleeping in a backoff). Checkpoint-driven means heartbeat staleness
maps to user-meaningful work boundaries.

Why this lives in ``shared/runtime_state/`` rather than ``cloris/``:
``shared/runtime_state/store.py`` is the caller, and shared cannot
import cloris (cloris depends on shared, not the other way). The
sidecar filename is hard-coded here; the canonical definition still
lives in ``cloris.worker.WORKER_SIDECAR_FILENAME``.

Behavior contract:

- ``bump_heartbeat(state_dir)`` never fabricates a ``worker.json``. When the
  sidecar doesn't exist (every CLI launch from ``tools/launch_*.sh``, which is
  how Cloris is run day to day), it keeps a minimal ``heartbeat`` file (JSON) beside
  the runtime DB instead, so every run, CLI or worker, has a heartbeat the death
  watchdog (``tools/cloris_watchdog.py``) can read::

      {"heartbeat_at": <UTC ISO>, "pid": <os.getpid()>,
       "pids": {"<pid>": {"first_at": <UTC ISO>, "last_at": <UTC ISO>}}}

  ``heartbeat_at``/``pid`` describe the latest write. ``pids`` keeps one entry
  per writing process, because admin tools and backfills open the same store
  and beat too: a visitor's write must not erase the run's own pid. The
  watchdog tells the run's process from a visitor by ``first_at`` (the run's
  process beats at ``start_run``; a visitor first beats later). Entries older
  than 7 days are dropped and at most 16 are kept.
  When the sidecar exists but is malformed (concurrent writer overlap) or
  unreadable, nothing is written, exactly as before. The orchestrator must
  not crash because the heartbeat updater did: nothing here raises.
- The return value still answers only "did worker.json move?": ``True`` when
  the sidecar was refreshed, ``False`` otherwise, including when the
  ``heartbeat`` fallback was written.
- The write is atomic via ``tmp + os.replace``, identical to
  :func:`cloris.worker.write_sidecar`, so a concurrent reader cannot
  observe a partially written file. ``heartbeat`` uses a unique temp
  name per call (threads in one process must not share one) and a short
  best-effort ``flock`` so two writers cannot drop each other's pid entry;
  if the lock is busy for half a second the write goes ahead unlocked.
- Only ``heartbeat_at`` is touched. All other sidecar fields
  (``pid``, ``brief_id``, ``mode``, ``launcher_version``, etc.) are
  preserved verbatim so the API process's reads stay coherent.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

# Hard-coded to match cloris.worker.WORKER_SIDECAR_FILENAME. Kept hard-coded
# rather than imported because shared cannot import cloris (layering rule).
# A drift here would be caught by tests/test_heartbeat.py::test_filename_matches_cloris.
_SIDECAR_FILENAME = "worker.json"

# The heartbeat for runs that have no worker sidecar. No ``.json`` suffix on
# purpose: state-dir ``*.json`` files are run projections, and replay tests
# capture every one of them; this file carries a live pid. Read by
# tools/cloris_watchdog.py, which hard-codes the same name (it must not import
# this package; see the watchdog's module docstring).
HEARTBEAT_FILENAME = "heartbeat"
_HEARTBEAT_LOCK_FILENAME = HEARTBEAT_FILENAME + ".lock"
_PID_RETENTION = timedelta(days=7)
_MAX_PIDS = 16
_LOCK_WAIT_SECONDS = 0.5


def _now() -> str:
    """Return a UTC ISO-8601 timestamp, identical-format to cloris.worker._now.

    Module-level seam so tests can freeze the clock. The format must match
    the worker's initial sidecar write so the aggregator's
    ``datetime.fromisoformat`` parse is consistent across both sources.
    """

    return datetime.now(timezone.utc).isoformat()


def bump_heartbeat(state_dir: Path | str) -> bool:
    """Atomically refresh ``heartbeat_at`` on ``<state_dir>/worker.json``.

    Returns ``True`` if the sidecar was found and updated; ``False`` if
    no update happened (sidecar missing, malformed, or write failed).
    When the sidecar is missing, ``<state_dir>/heartbeat`` is written
    instead (see the module docstring); that does not change the return.
    Best-effort: never raises. Designed to be called from the orchestrator
    at every write checkpoint without becoming a new failure mode.
    """

    try:
        state_dir = Path(state_dir)
        sidecar_path = state_dir / _SIDECAR_FILENAME
        sidecar_exists = sidecar_path.exists()
    except (OSError, TypeError, ValueError):
        return False

    if not sidecar_exists:
        _write_heartbeat_file(state_dir)
        return False

    try:
        raw = sidecar_path.read_text()
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(parsed, dict):
        return False

    parsed["heartbeat_at"] = _now()

    try:
        tmp_path = state_dir / (_SIDECAR_FILENAME + ".heartbeat.tmp")
        tmp_path.write_text(json.dumps(parsed, indent=2, sort_keys=True))
        os.replace(tmp_path, sidecar_path)
    except OSError:
        return False
    return True


def _write_heartbeat_file(state_dir: Path) -> bool:
    """Refresh ``<state_dir>/heartbeat`` atomically; never raises.

    The file is owned by this function alone, so a corrupt or foreign
    payload is simply replaced. ``pid`` is the writing process — the
    watchdog uses the per-pid entries to tell "Cloris died" from
    "Cloris is quiet" without being fooled by a visiting admin tool.
    """

    tmp_path: Path | None = None
    try:
        with _heartbeat_lock(state_dir):
            now = _now()
            pid = os.getpid()
            target = state_dir / HEARTBEAT_FILENAME
            pids = _read_pid_entries(target)
            mine = pids.get(str(pid)) or {}
            first_at = mine.get("first_at") if isinstance(mine.get("first_at"), str) else now
            pids[str(pid)] = {"first_at": first_at, "last_at": now}
            payload = {"heartbeat_at": now, "pid": pid, "pids": _prune_pids(pids, str(pid))}
            fd, raw_tmp = tempfile.mkstemp(
                dir=state_dir, prefix=HEARTBEAT_FILENAME + ".", suffix=".tmp"
            )
            tmp_path = Path(raw_tmp)
            with os.fdopen(fd, "w") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
            os.replace(tmp_path, target)
            tmp_path = None
    except Exception:  # noqa: BLE001 - a heartbeat must never kill a run
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                tmp_path.unlink()
        return False
    return True


def _read_pid_entries(path: Path) -> dict[str, dict]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    pids = raw.get("pids") if isinstance(raw, dict) else None
    if not isinstance(pids, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in pids.items()
        if isinstance(value, dict) and isinstance(value.get("last_at"), str)
    }


def _prune_pids(pids: dict[str, dict], keep: str) -> dict[str, dict]:
    """Drop entries older than the retention window; keep the newest 16."""

    cutoff = datetime.now(timezone.utc) - _PID_RETENTION

    def _last(entry: dict) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(entry["last_at"])
        except (KeyError, TypeError, ValueError):
            return None
        return parsed if parsed.tzinfo is not None else None

    fresh = {
        key: entry
        for key, entry in pids.items()
        if key == keep or ((_last(entry) or cutoff) > cutoff)
    }
    ranked = sorted(
        fresh,
        key=lambda key: (key == keep, _last(fresh[key]) or cutoff),
        reverse=True,
    )
    return {key: fresh[key] for key in ranked[:_MAX_PIDS]}


@contextlib.contextmanager
def _heartbeat_lock(state_dir: Path) -> Iterator[None]:
    """Best-effort exclusive flock for the read-modify-write; never blocks long."""

    handle = None
    try:
        handle = open(state_dir / _HEARTBEAT_LOCK_FILENAME, "a")
        deadline = time.monotonic() + _LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.005)
    except OSError:
        pass
    try:
        yield
    finally:
        if handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
