"""Operator arm-path for the persisted per-source launch pause (Reopen Y.5.6 / F1).

The Y.5.2 kill-switch (``CLORIS_PAUSE_LAUNCHES_<SOURCE>``) is process-env only:
it can be armed only where the API server's environment lives, which means it
cannot be armed OUT-OF-PROCESS once the server is running. Y.5.6 adds a durable,
server-observable arm — a ``source_pause`` row in the orchestration DB — that the
in-process spawn gate reads on its NEXT spawn (additive-OR with the env arm).

This tool is that out-of-process arm. It writes the ``source_pause`` row via
``OrchestrationStateStore.set_source_pause``; the running API server sees the
write on the next launch with no restart and no env mutation.

  Arm:     python -m tools.pause_source_launches designer --reason "vendor outage"
  Disarm:  python -m tools.pause_source_launches designer --resume
  Status:  python -m tools.pause_source_launches designer --status
  All:     python -m tools.pause_source_launches --status   (no source = every row)

Pure operator surface — it touches ONLY the orchestration DB's ``source_pause``
table (never a per-state-dir DB, never the recruiter or identity DB). Disarm
leaves the row in place (recording who/when armed it last) but flips ``paused``
to 0 so the gate no longer treats the source as paused.
"""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path


def _store():
    from shared.output_paths import resolve_orchestration_db_path
    from shared.runtime_state.orchestration_store import OrchestrationStateStore

    return OrchestrationStateStore(resolve_orchestration_db_path())


def _status_rows(source: str | None) -> list[dict]:
    """Pure read of the ``source_pause`` table (one source or all)."""

    store = _store()
    with store.connect() as conn:
        if source:
            rows = conn.execute(
                "SELECT source, paused, armed_at, armed_by, reason "
                "FROM source_pause WHERE source = ?",
                ((source or "").strip().lower(),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT source, paused, armed_at, armed_by, reason "
                "FROM source_pause ORDER BY source"
            ).fetchall()
    return [
        {
            "source": r["source"],
            "paused": int(r["paused"] or 0) != 0,
            "armed_at": r["armed_at"],
            "armed_by": r["armed_by"],
            "reason": r["reason"],
        }
        for r in rows
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Arm/disarm the persisted per-source launch pause (Reopen Y.5.6). "
            "Writes the orchestration DB's source_pause row that the in-process "
            "spawn gate reads on its next spawn — the out-of-process equivalent "
            "of the CLORIS_PAUSE_LAUNCHES_<SOURCE> env arm."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Source to pause/resume (e.g. designer, github). Omit with "
        "--status to dump every source_pause row.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Disarm the pause for SOURCE (default action arms it).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the current source_pause state (pure read, no write).",
    )
    parser.add_argument(
        "--reason",
        default="",
        help="Operator reason recorded on the row when arming.",
    )
    parser.add_argument(
        "--armed-by",
        default=None,
        help="Operator identity recorded on the row (default: $USER).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.status:
        print(json.dumps(_status_rows(args.source), indent=2))
        return 0

    if not args.source:
        print(
            json.dumps(
                {"error": "source required unless --status is given"}, indent=2
            )
        )
        return 2

    armed_by = args.armed_by
    if armed_by is None:
        try:
            armed_by = getpass.getuser()
        except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
            armed_by = ""

    store = _store()
    store.set_source_pause(
        args.source,
        paused=not args.resume,
        armed_by=armed_by or "",
        reason=args.reason,
    )
    print(
        json.dumps(
            {
                "source": (args.source or "").strip().lower(),
                "paused": not args.resume,
                "armed_by": armed_by or "",
                "reason": args.reason,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
