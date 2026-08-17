"""Cloris command-line entrypoint.

Slice 1 ships exactly one subcommand: ``cloris start``. It builds the FastAPI
app via :func:`cloris.app.create_app` and hands off to
:func:`cloris.app.run_app` which owns the full app-process lifecycle (uvicorn
in a background thread, readiness probe, native window launch, clean shutdown).

There is intentionally **no** ``--no-window`` (or any other test-only) flag.
Tests inject a launcher and a server factory through Python kwargs on
``run_app`` instead.
"""

from __future__ import annotations

import argparse
from typing import Sequence


_MISSING_DEPS_HINT = (
    "Cloris requires fastapi and uvicorn (and pywebview for the native "
    "window). Install them with:\n"
    "    pip install fastapi uvicorn pywebview"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the Cloris CLI.

    Exposed at module scope so tests can introspect the registered actions
    (e.g. to assert that no ``--no-window`` flag exists).
    """

    parser = argparse.ArgumentParser(
        prog="cloris",
        description="Cloris desktop shell (v0 / slice 1).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser(
        "start",
        help="Start the Cloris app process (FastAPI + native window).",
        description=(
            "Start the local Cloris app process. Boots a FastAPI server in a "
            "background thread and opens the native window through the "
            "pywebview launcher seam."
        ),
    )
    start.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the local server to (default: 127.0.0.1).",
    )
    start.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to bind (default: 0 = pick a free ephemeral port).",
    )

    # Audit Move #19: schema migration ops surface. Opens a runtime-state
    # store (which runs the additive-idempotent _migrate on connect) and
    # reports the resulting schema_version. See docs/ops-schema-migration.md
    # for the operating contract.
    migrate = subparsers.add_parser(
        "migrate",
        help="Apply pending runtime-state migrations + report schema version.",
        description=(
            "Open a runtime-state SQLite (per-source or orchestration) and "
            "let the additive-idempotent migration runner upgrade the "
            "schema. Prints the resulting schema_version. Safe to run "
            "against any existing database — every migration is idempotent."
        ),
    )
    target = migrate.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--db-path",
        help="Per-source runtime_state.sqlite3 path (e.g., output/state/<source>/<key>/runtime_state.sqlite3).",
    )
    target.add_argument(
        "--orchestration-db-path",
        help="Orchestration runtime_state.sqlite3 path (output/state/orchestration/runtime_state.sqlite3).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the Cloris CLI.

    Returns an exit code so callers (and ``python -m cloris``) can pass it to
    ``sys.exit``.
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "start":
        from cloris.app import require_loopback_host

        try:
            require_loopback_host(args.host)
        except ValueError as exc:
            parser.exit(2, f"cloris: {exc}\n")
        return _run_start(host=args.host, port=args.port)

    if args.command == "migrate":
        return _run_migrate(
            db_path=args.db_path,
            orchestration_db_path=args.orchestration_db_path,
        )

    parser.error(f"unknown command: {args.command!r}")
    return 2


def _run_start(*, host: str, port: int) -> int:
    import os

    try:
        from cloris import app as cloris_app
    except ImportError as exc:  # pragma: no cover - defensive import-error path
        print(_MISSING_DEPS_HINT)
        raise SystemExit(_format_import_error(exc)) from exc

    try:
        fastapi_app = cloris_app.create_app()
    except ImportError as exc:
        print(_MISSING_DEPS_HINT)
        raise SystemExit(_format_import_error(exc)) from exc
    except Exception as exc:
        import sys
        from shared.config import MissingRequiredKeyError
        if isinstance(exc, MissingRequiredKeyError):
            sys.stderr.write(f"cloris: {exc}\n")
            return 1
        raise

    # The browser-observed certification harness starts ``cloris start``
    # in a subprocess and reads ``cloris_certify_ready url=...`` off
    # stdout to discover the bound port. Honor ``CLORIS_CERTIFY_HEADLESS=1``
    # by swapping the pywebview launcher for the headless cert launcher
    # so the same env contract works in both packaged and source modes.
    cert_headless = os.environ.get("CLORIS_CERTIFY_HEADLESS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    launcher = cloris_app.CertificationWindowLauncher() if cert_headless else None
    cloris_app.run_app(fastapi_app, host=host, port=port, launcher=launcher)
    return 0


def _format_import_error(exc: ImportError) -> str:
    return f"cloris: missing dependency ({exc})."


def _run_migrate(
    *,
    db_path: str | None,
    orchestration_db_path: str | None,
) -> int:
    """Apply pending runtime-state migrations + report schema version.

    Audit Move #19. Opens the relevant store (per-source or orchestration);
    the store's __init__ runs the additive-idempotent _migrate which applies
    every pending column / table / index addition. Returns 0 on a clean
    pass, 1 on operational failure (missing path, corrupt DB, etc.).
    """

    import sqlite3
    import sys
    from pathlib import Path

    if db_path:
        path = Path(db_path)
        if not path.parent.exists():
            sys.stderr.write(
                f"cloris migrate: parent directory missing for {path}\n"
            )
            return 1
        try:
            from shared.runtime_state.store import (
                CURRENT_SCHEMA_VERSION,
                RuntimeStateStore,
            )

            RuntimeStateStore(path)
            with sqlite3.connect(str(path)) as conn:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()
            version = row[0] if row else "<unknown>"
            sys.stdout.write(
                f"cloris migrate: per-source store at {path} now at "
                f"schema_version={version} (current={CURRENT_SCHEMA_VERSION})\n"
            )
            return 0
        except Exception as exc:  # noqa: BLE001 — operational fail-soft
            sys.stderr.write(f"cloris migrate: failed to open {path} ({exc!r})\n")
            return 1

    if orchestration_db_path:
        path = Path(orchestration_db_path)
        if not path.parent.exists():
            sys.stderr.write(
                f"cloris migrate: parent directory missing for {path}\n"
            )
            return 1
        try:
            from shared.runtime_state.orchestration_store import (
                CURRENT_ORCHESTRATION_SCHEMA_VERSION,
                OrchestrationStateStore,
            )

            OrchestrationStateStore(path)
            with sqlite3.connect(str(path)) as conn:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key = 'orchestration_schema_version'"
                ).fetchone()
            version = row[0] if row else "<unknown>"
            sys.stdout.write(
                f"cloris migrate: orchestration store at {path} now at "
                f"orchestration_schema_version={version} "
                f"(current={CURRENT_ORCHESTRATION_SCHEMA_VERSION})\n"
            )
            return 0
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"cloris migrate: failed to open {path} ({exc!r})\n")
            return 1

    sys.stderr.write(
        "cloris migrate: --db-path or --orchestration-db-path required\n"
    )
    return 1
