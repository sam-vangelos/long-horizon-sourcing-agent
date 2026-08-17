"""Executive Search session orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exec_search.session_orchestrator",
        description="Executive Search session orchestrator.",
    )
    parser.add_argument(
        "--brief",
        required=True,
        help="Path to the brief JSON.",
    )
    parser.add_argument(
        "--state-dir",
        required=True,
        help="Per-source state directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run (no-op in Slice 1 stub).",
    )
    parser.add_argument(
        "--investigate",
        action="store_true",
        help=(
            "Run pre-launch market intelligence investigation before sourcing. "
            "Writes artifacts onto the brief filesystem; opt-in only."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Exec Search pipeline."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    brief_path = Path(args.brief)
    if not brief_path.exists():
        sys.stderr.write(
            f"exec_search.session_orchestrator: brief not found at {brief_path}\n"
        )
        return 2

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    try:
        from exec_search.orchestrator import build_pipeline

        pipeline, run_id = build_pipeline(
            brief_path=brief_path,
            state_dir=state_dir,
            resume=args.resume,
            investigate_at_launch=args.investigate,
        )
        stats = pipeline.run(run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - CLI should report cleanly
        sys.stderr.write(
            f"exec_search.session_orchestrator: pipeline failed ({exc!r}).\n"
        )
        return 1

    sys.stdout.write(
        f"exec_search.session_orchestrator: completed run_id={run_id} "
        f"lanes={stats.lanes_completed} candidates={stats.candidates_discovered}.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
