#!/usr/bin/env python3
"""Admin entry point for runtime_state repair and projection tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.runtime_state import (
    LinkedInRuntimeStateBridge,
    RuntimeStateStore,
    clear_candidate_terminal_state,
    inspect_candidate_side_effects,
    inspect_orphaned_attempts,
    replay_candidate_side_effect,
    rebuild_compat_projections,
    requeue_work_unit,
)
from shared.schemas import Progress


def _store_for_output(output_dir: str | Path) -> RuntimeStateStore:
    output_dir = Path(output_dir)
    db_path = output_dir / "runtime_state.sqlite3"
    if not db_path.exists():
        raise FileNotFoundError(f"runtime state DB not found: {db_path}")
    return RuntimeStateStore(db_path)


def _resolve_run_id(store: RuntimeStateStore, *, source: str, brief_id: str, run_id: int | None) -> int:
    if run_id is not None:
        return run_id
    latest = store.get_latest_run(source=source, brief_id=brief_id)
    if not latest:
        raise ValueError(f"no run found for {source}:{brief_id}")
    return int(latest["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Official operator surface for runtime_state-backed sourcing runs")
    parser.add_argument("--output-dir", required=True, help="Output directory containing runtime_state.sqlite3")
    parser.add_argument("--source", required=True, choices=["github", "linkedin"], help="Run source namespace")
    parser.add_argument("--brief-id", required=True, help="Brief ID scoped inside runtime_state")
    parser.add_argument("--run-id", type=int, default=None, help="Optional run ID; defaults to latest run for source+brief")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("rebuild-projections", help="Rebuild compatibility projections from runtime_state")
    subparsers.add_parser("inspect-orphans", help="List orphaned in-flight attempts")
    subparsers.add_parser("inspect-stop-reasons", help="List persisted run stop reasons")
    subparsers.add_parser("inspect-side-effects", help="List candidate-scoped side effects")
    subparsers.add_parser("rebuild-linkedin-artifacts", help="Rebuild LinkedIn projections and stage artifacts")
    subparsers.add_parser("inspect-linkedin-orphans", help="List orphaned LinkedIn attempts")
    subparsers.add_parser("import-legacy-linkedin", help="Import legacy LinkedIn progress/history into runtime_state")

    requeue_parser = subparsers.add_parser("requeue-work-unit", help="Requeue a work unit")
    requeue_parser.add_argument("--kind", required=True, help="Work-unit kind, e.g. github_query")
    requeue_parser.add_argument("--source-unit-id", required=True, help="Source-specific work-unit identifier")

    clear_parser = subparsers.add_parser("clear-terminal", help="Clear a candidate terminal state for retry")
    clear_parser.add_argument("--identity-key", required=True, help="Candidate identity key")

    replay_parser = subparsers.add_parser("replay-side-effect", help="Invalidate a candidate side effect so it can be replayed intentionally")
    replay_parser.add_argument("--identity-key", required=True, help="Candidate identity key")
    replay_parser.add_argument("--effect-type", required=True, help="Side-effect type, e.g. github_outreach")

    restart_parser = subparsers.add_parser("restart-linkedin-string", help="Restart one LinkedIn string through runtime_state")
    restart_parser.add_argument("--string-id", required=True, type=int, help="LinkedIn search string ID")

    args = parser.parse_args()

    try:
        store = _store_for_output(args.output_dir)
        run_id = _resolve_run_id(store, source=args.source, brief_id=args.brief_id, run_id=args.run_id)

        if args.command == "rebuild-projections":
            rebuild_compat_projections(store, run_id=run_id, output_dir=args.output_dir)
            print(f"Rebuilt projections for run {run_id} in {Path(args.output_dir)}")
            return

        if args.command == "inspect-orphans":
            rows = inspect_orphaned_attempts(store, source=args.source, brief_id=args.brief_id)
            print(json.dumps(rows, indent=2))
            return

        if args.command == "inspect-stop-reasons":
            print(json.dumps(store.list_runs(source=args.source, brief_id=args.brief_id), indent=2))
            return

        if args.command == "inspect-side-effects":
            rows = inspect_candidate_side_effects(
                store,
                source=args.source,
                brief_id=args.brief_id,
            )
            print(json.dumps(rows, indent=2))
            return

        if args.command == "inspect-linkedin-orphans":
            rows = inspect_orphaned_attempts(store, source="linkedin", brief_id=args.brief_id)
            print(json.dumps(rows, indent=2))
            return

        if args.command == "rebuild-linkedin-artifacts":
            rebuild_compat_projections(store, run_id=run_id, output_dir=args.output_dir)
            print(f"Rebuilt LinkedIn artifacts for run {run_id} in {Path(args.output_dir)}")
            return

        if args.command == "import-legacy-linkedin":
            bridge = LinkedInRuntimeStateBridge(
                store=store,
                output_dir=args.output_dir,
                brief_id=args.brief_id,
                brief_name=args.brief_id,
            )
            bridge.import_legacy_state(run_id)
            bridge.rebuild_artifacts(run_id)
            print(f"Imported legacy LinkedIn state into run {run_id} and rebuilt artifacts")
            return

        if args.command == "requeue-work-unit":
            requeue_work_unit(
                store,
                run_id=run_id,
                kind=args.kind,
                source_unit_id=args.source_unit_id,
                output_dir=args.output_dir,
            )
            print(
                f"Requeued {args.kind}:{args.source_unit_id} on run {run_id} and rebuilt projections"
            )
            return

        if args.command == "clear-terminal":
            clear_candidate_terminal_state(
                store,
                source=args.source,
                brief_id=args.brief_id,
                identity_key=args.identity_key,
            )
            rebuild_compat_projections(store, run_id=run_id, output_dir=args.output_dir)
            print(
                f"Cleared terminal state for {args.identity_key} on {args.source}:{args.brief_id} and rebuilt projections"
            )
            return

        if args.command == "replay-side-effect":
            replayed = replay_candidate_side_effect(
                store,
                source=args.source,
                brief_id=args.brief_id,
                identity_key=args.identity_key,
                effect_type=args.effect_type,
            )
            print(
                f"Invalidated {replayed} {args.effect_type} side-effect row(s) for {args.identity_key}"
            )
            return

        if args.command == "restart-linkedin-string":
            bridge = LinkedInRuntimeStateBridge(
                store=store,
                output_dir=args.output_dir,
                brief_id=args.brief_id,
                brief_name=args.brief_id,
            )
            progress = bridge.load_progress(run_id)
            bridge.restart_string(run_id=run_id, progress=progress, string_id=args.string_id)
            print(f"Restarted LinkedIn string {args.string_id} on run {run_id}")
            return
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
