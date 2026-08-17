"""Cloris detached worker (multi-source per Phase F Slice F1).

Exec-replace entrypoint that the API process spawns via
``subprocess.Popen([sys.executable, "-m", "cloris.worker", ...])``. The worker
writes a ``worker.json`` sidecar into the source-specific state directory,
then ``os.execvp``s into the per-source orchestrator argv. After the exec,
the same PID belongs to the orchestrator process — so the sidecar's
``pid`` field stays truthful for any later stop/probe operation.

This module is wrapper code only:

- It does not import the per-source orchestrator modules; it spawns them.
- It does not write canonical SQLite state; the orchestrator does.
- It owns ``worker.json`` as a Cloris-only sidecar (not a runtime-state record).

Phase F Slice F1 surface:

- Accepts ``--source linkedin|github`` and dispatches to the per-source
  orchestrator argv builder via :mod:`cloris.launchers`. Adding a new
  source (e.g., ``researcher``) is a one-line append in the registry;
  this wrapper does not change.
- Sets ``heartbeat_at == started_at`` at spawn; after that the runtime-state
  store bumps the sidecar heartbeat on every canonical write checkpoint
  (``shared/runtime_state/store.py``, Phase 1.6) — the value is live, not
  frozen.
- Exposes ``--mode {fresh, resume}``. ``--mode resume`` threads
  ``--resume`` into the orchestrator argv; the sidecar's ``mode`` field
  reflects the truth (``"fresh"`` or ``"resume"``).
- Exposes ``--fresh`` as explicit fresh-regeneration consent for sources
  whose orchestrator needs a typed fresh flag. It is valid only with
  ``--mode fresh``.
- Sidecar ``input_mode`` is always ``"concurrent"`` for v0 — Cloris does
  not run "away mode" workers.
- ``LAUNCHER_VERSION`` advertises ``"cloris-v0-slice-4"`` so reconciliation
  against older sidecars stays legible. Bumped only when the sidecar
  shape changes; F1 keeps the shape stable so the version stays.

Phase 0 ``worker-binary`` slice (frozen-app context):

- When ``getattr(sys, 'frozen', False)`` is true (PyInstaller/py2app .app),
  the worker cannot ``os.execvp`` into ``[python, -m, MODULE, ...]``
  because there is no python interpreter inside the bundle. Instead,
  the worker dispatches in-process via :func:`_dispatch_in_process` —
  the orchestrator runs in this same PID, the sidecar's ``pid`` field
  stays truthful exactly as it does under ``execvp``.
- The frozen .app ships ``cloris-worker`` as a sibling binary to
  ``Cloris`` under ``Cloris.app/Contents/MacOS/``. The API process
  spawns ``[<bundle>/Contents/MacOS/cloris-worker, ...]`` rather than
  ``[python, -m, cloris.worker, ...]`` — see
  ``cloris/api.py:_build_worker_argv``.

Module-level seams (``_now``, ``_exec``, ``_dispatch_in_process``) exist
so tests can monkeypatch them without ever spawning a real subprocess
or replacing the test process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn, Sequence


WORKER_SIDECAR_FILENAME = "worker.json"
LAUNCHER_VERSION = "cloris-v0-slice-4"


class BriefPathNotFoundError(Exception):
    """Raised by the API launch helper when ``brief_path`` does not exist on disk.

    Imported by :mod:`cloris.api` so the route can map this to HTTP 400. The
    worker CLI does not raise this directly — it expects the API helper to
    have already validated the path.
    """


class WorkerAlreadyRunningError(Exception):
    """Raised when an existing ``worker.json`` references a live PID.

    The route maps this to HTTP 409 Conflict and surfaces ``pid`` plus
    ``state_dir`` in the response body so the client can render an actionable
    error.
    """

    def __init__(self, pid: int, state_dir: str) -> None:
        super().__init__(f"worker already running (pid={pid}, state_dir={state_dir})")
        self.pid = pid
        self.state_dir = state_dir


def build_sidecar(
    *,
    source: str,
    brief_id: str,
    brief_path: str,
    output_dir: str,
    mode: str,
    input_mode: str,
    started_at: str,
    pid: int,
    run_id: int | None = None,
) -> dict:
    """Return the ``worker.json`` payload for a fresh launch.

    Field set is fixed by ``docs/cloris-control-plane-spec.md`` §5; Slice 3
    pins ``heartbeat_at`` to ``started_at`` because no live heartbeat updater
    exists yet.
    """

    return {
        "pid": pid,
        "source": source,
        "brief_id": brief_id,
        "brief_path": brief_path,
        "output_dir": output_dir,
        "run_id": run_id,
        "started_at": started_at,
        "heartbeat_at": started_at,
        "mode": mode,
        "input_mode": input_mode,
        "launcher_version": LAUNCHER_VERSION,
    }


def write_sidecar(state_dir: Path, payload: dict) -> Path:
    """Atomically write ``payload`` to ``<state_dir>/worker.json``.

    Uses ``tmp + os.replace`` so a reader can never observe a partially
    written file. JSON is serialized with ``indent=2, sort_keys=True`` so
    the on-disk artifact is diff-stable.
    """

    state_dir.mkdir(parents=True, exist_ok=True)
    final_path = state_dir / WORKER_SIDECAR_FILENAME
    tmp_path = state_dir / (WORKER_SIDECAR_FILENAME + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp_path, final_path)
    return final_path


def read_sidecar(state_dir: Path) -> dict | None:
    """Return the parsed sidecar dict, or ``None`` for missing/malformed files.

    Never raises. Missing file, unreadable file, malformed JSON, and a JSON
    document whose top level is not an object all collapse to ``None`` —
    callers treat any of those as "no sidecar present" or "stale", which is
    exactly the policy the launch helper applies.
    """

    sidecar_path = state_dir / WORKER_SIDECAR_FILENAME
    if not sidecar_path.exists():
        return None
    try:
        raw = sidecar_path.read_text()
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def is_pid_alive(pid) -> bool:
    """Return whether ``pid`` is currently a live process.

    ``ProcessLookupError`` ⇒ dead. ``PermissionError`` ⇒ alive (we don't own
    the process but it exists). Anything else (malformed input, type errors,
    other ``OSError``) ⇒ treat as stale — the user-approved policy is to err
    on the side of overwriting unparseable sidecars rather than blocking new
    launches.

    Non-positive ints are rejected up front: ``os.kill(0, 0)`` and
    ``os.kill(-1, 0)`` succeed under POSIX with process-group semantics
    (signal every process in the current session / every process the caller
    can signal), which is not what "is this specific PID alive" should
    answer. We treat them as stale to keep the contract single-PID.
    """

    try:
        coerced = int(pid)
    except (TypeError, ValueError):
        return False

    if coerced <= 0:
        return False

    if _pid_is_zombie(coerced):
        return False

    try:
        os.kill(coerced, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _pid_is_zombie(pid: int) -> bool:
    """Best-effort zombie detection for POSIX hosts.

    ``kill(pid, 0)`` reports zombies as existing, but a defunct worker cannot
    do work and must not hold a launch lock. If ``ps`` is unavailable or the
    row cannot be read, fall back to the normal signal probe.
    """

    if os.name != "posix":
        return False
    try:
        proc = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    stat = proc.stdout.strip()
    return bool(stat) and stat[0].upper() == "Z"


def build_session_orchestrator_argv(
    *,
    brief_path: str,
    state_dir: str,
    input_mode: str = "concurrent",
    resume: bool = False,
    fresh: bool = False,
    python_executable: str = sys.executable,
) -> list[str]:
    """Compose the argv for ``os.execvp`` into ``linkedin.session_orchestrator``.

    Pure function: no I/O, no globals beyond ``sys.executable``. Kept
    independently testable so test cases can pin the exact command shape
    without going through ``main``.

    LinkedIn-specific. Phase F Slice F1 introduces the per-source
    registry at :mod:`cloris.launchers`; this remains the LinkedIn
    entry's argv builder so the existing Slice-3/Slice-4 contract is
    preserved byte-for-byte. New sources (e.g., GitHub) build their own
    argv via the registry and never call this function.
    """

    argv: list[str] = [
        python_executable,
        "-m",
        "linkedin.session_orchestrator",
        "--brief",
        brief_path,
        "--state-dir",
        state_dir,
        "--input-mode",
        input_mode,
        "--single-session",
    ]
    if resume and fresh:
        raise ValueError("resume and fresh are mutually exclusive")
    if resume:
        argv.append("--resume")
    if fresh:
        argv.append("--fresh")
    return argv


def _now() -> str:
    """Return a UTC ISO-8601 timestamp.

    Module-level seam so tests can freeze the clock without monkeypatching
    :mod:`datetime`. Production callers should not rely on the format
    beyond "ISO-8601 with timezone".
    """

    return datetime.now(timezone.utc).isoformat()


def _exec(argv: list[str]) -> NoReturn:
    """Replace the current process image with ``argv``.

    Module-level seam so tests can monkeypatch this to a recorder and
    avoid actually replacing the test process. In production this never
    returns; the test stub returns ``None`` and ``main`` falls through to
    its ``return 0``.
    """

    os.execvp(argv[0], argv)
    raise SystemExit(  # pragma: no cover - execvp never returns
        "os.execvp returned unexpectedly"
    )


def _is_frozen() -> bool:
    """Module-level seam mirroring ``shared.user_data_dir.is_frozen_app``.

    Inlined here rather than imported so the worker module stays
    importable under odd test layouts (e.g., a stub
    ``shared.user_data_dir`` that raises). PyInstaller and py2app both
    set ``sys.frozen``; we don't depend on ``sys._MEIPASS`` for the
    same reason.
    """

    return bool(getattr(sys, "frozen", False))


def _dispatch_in_process(source: str, orchestrator_argv: list[str]) -> int:
    """Frozen-app fallback: run the orchestrator's ``main()`` in-process.

    The .app bundle has no python interpreter, so we cannot
    ``os.execvp`` into ``[python, -m, MODULE, ...]``. Instead we
    delegate to the per-source ``in_process_dispatch_fn`` registered
    on :data:`cloris.launchers.LAUNCHERS`; PID stays this process's
    PID so the sidecar's ``pid`` field stays truthful for any later
    stop/probe operation.

    ``orchestrator_argv`` is the same shape the registry's
    ``orchestrator_argv_fn`` produces — ``[python_executable, "-m",
    MODULE_DOTPATH, ...orchestrator-cli-args]``. The per-source
    dispatcher slices off the python invocation prefix and hands the
    orchestrator-cli-args to its ``main()`` via ``sys.argv``.

    Multi-agent-execution Phase 1 Slice 1.2 replaces the legacy
    if/elif source ladder with registry dispatch. Side-effect:
    ``exec_search`` becomes registerable in the frozen .app — closes
    the silent regression at the legacy ladder which only supported
    ``linkedin`` / ``github`` / ``researcher`` / ``designer``. See
    ``plans/multi-agent-execution-plan.md`` §"1.2 in_process_dispatch_fn".

    Returns the orchestrator's exit code (defaults to 0 if main()
    returns ``None``). Exposed at module scope as a test seam so a
    frozen-context test can pass a recorder without invoking a real
    orchestrator.
    """

    from cloris.launchers import LAUNCHERS

    launcher = LAUNCHERS.get(source)
    dispatcher = launcher.in_process_dispatch_fn if launcher is not None else None
    if dispatcher is None:
        sys.stderr.write(
            f"cloris.worker: no in-process dispatch for source={source!r} "
            f"(registered: {sorted(LAUNCHERS.keys())}).\n"
        )
        return 2

    if len(orchestrator_argv) < 3:
        sys.stderr.write(
            f"cloris.worker: malformed orchestrator argv for in-process "
            f"dispatch (need [python, -m, MODULE, ...]): {orchestrator_argv!r}\n"
        )
        return 2

    return dispatcher(orchestrator_argv)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloris.worker",
        description="Cloris detached worker (writes worker.json then execvps into the per-source orchestrator).",
    )
    parser.add_argument(
        "--source",
        default="linkedin",
        help=(
            "Source name registered in cloris.launchers (linkedin/github/...). "
            "Defaults to 'linkedin' so legacy Slice-3 invocations without "
            "--source keep working byte-for-byte."
        ),
    )
    parser.add_argument(
        "--brief",
        required=True,
        help="Path to the brief JSON (forwarded to the per-source orchestrator).",
    )
    parser.add_argument(
        "--brief-id",
        required=True,
        help=(
            "Brief id recorded in worker.json (typically the source's "
            "state-key fn output: derive_brief_id / github_state_key)."
        ),
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help=(
            "Per-source state directory. If absent, resolved via the "
            "registry's state_dir_fn for --source."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=["concurrent"],
        default="concurrent",
        help="Cloris v0 is concurrent-only; 'away' is rejected at the wrapper boundary.",
    )
    parser.add_argument(
        "--mode",
        choices=["fresh", "resume"],
        default="fresh",
        help="fresh = new run; resume = continue an interrupted run.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Explicit fresh-regeneration consent forwarded to source "
            "orchestrators that require a typed fresh flag."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for ``python -m cloris.worker``.

    Steps: parse args → resolve state dir via the registry → write
    sidecar → execvp into the per-source orchestrator argv (also from
    the registry). The ``return 0`` is only reached when ``_exec`` is
    monkeypatched in tests; in production the process image is
    replaced before this line.

    Per Phase F Slice F1, this wrapper is source-generic: the only
    LinkedIn-specific behavior left is the default value of ``--source``
    so legacy callers (Slice 3 launch without ``--source``) keep working.
    """

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.fresh and args.mode != "fresh":
        parser.error("--fresh can only be used with --mode fresh")

    from cloris.launchers import LAUNCHERS

    launcher = LAUNCHERS.get(args.source)
    if launcher is None:
        # Unknown source — exit non-zero so the spawn-side aggregator
        # observes the failure rather than waiting forever for a
        # sidecar that will never appear.
        sys.stderr.write(
            f"cloris.worker: unknown --source '{args.source}'. "
            f"Registered: {sorted(LAUNCHERS.keys())}\n"
        )
        return 2

    if args.state_dir is not None:
        state_dir = Path(args.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
    else:
        state_dir = launcher.state_dir_fn(args.brief)

    payload = build_sidecar(
        source=args.source,
        brief_id=args.brief_id,
        brief_path=args.brief,
        output_dir=str(state_dir),
        mode=args.mode,
        input_mode=args.input_mode,
        started_at=_now(),
        pid=os.getpid(),
        run_id=None,
    )
    write_sidecar(state_dir, payload)

    argv_to_exec = launcher.orchestrator_argv_fn(
        args.brief,
        str(state_dir),
        resume=(args.mode == "resume"),
        fresh=args.fresh,
    )
    if _is_frozen():
        # Frozen .app: dispatch in-process. Same PID, so the sidecar's
        # ``pid`` field stays truthful for any later stop/probe.
        return _dispatch_in_process(args.source, argv_to_exec)
    _exec(argv_to_exec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
