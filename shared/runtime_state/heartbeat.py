"""Phase 1.6: per-state-dir worker heartbeat updater.

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

- ``bump_heartbeat(state_dir)`` is a best-effort no-op when the sidecar
  doesn't exist (orchestrator running standalone without Cloris launch),
  is malformed (concurrent writer overlap), or fails to read for any
  reason. The orchestrator must not crash because the heartbeat updater
  did.
- The write is atomic via ``tmp + os.replace``, identical to
  :func:`cloris.worker.write_sidecar`, so a concurrent reader cannot
  observe a partially written file.
- Only ``heartbeat_at`` is touched. All other sidecar fields
  (``pid``, ``brief_id``, ``mode``, ``launcher_version``, etc.) are
  preserved verbatim so the API process's reads stay coherent.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Hard-coded to match cloris.worker.WORKER_SIDECAR_FILENAME. Kept hard-coded
# rather than imported because shared cannot import cloris (layering rule).
# A drift here would be caught by tests/test_heartbeat.py::test_filename_matches_cloris.
_SIDECAR_FILENAME = "worker.json"


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
    Best-effort: never raises. Designed to be called from the orchestrator
    at every write checkpoint without becoming a new failure mode.
    """

    state_dir = Path(state_dir)
    sidecar_path = state_dir / _SIDECAR_FILENAME

    if not sidecar_path.exists():
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
