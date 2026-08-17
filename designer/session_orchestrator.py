"""Designer session orchestrator.

Runs the Designer sourcing pipeline and then invokes
:func:`designer.run_end.run_end_designer_rubric_refinement`, which
rolls up the per-principle recruiter feedback marker distribution and
persists proposed ``RUBRIC_REFINE`` hunks under the state-dir for the
reflection pipeline (``market_intelligence/reflection.py``) to
surface. The hook is fail-soft and never blocks run completion.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from shared.observability import observe
from shared.storage import log_event


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="designer.session_orchestrator",
        description="Designer session orchestrator.",
    )
    parser.add_argument(
        "--brief",
        required=True,
        help="Path to the brief JSON.",
    )
    parser.add_argument(
        "--state-dir",
        required=True,
        help="Per-source state directory under output/state/designer/<state_key>/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run (no-op in Slice 1 stub).",
    )
    return parser


@observe(name="designer.run")
def main(argv: Sequence[str] | None = None) -> int:
    """Run Designer sourcing and then the fail-soft rubric-refinement hook."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    brief_path = Path(args.brief)
    if not brief_path.exists():
        sys.stderr.write(
            f"designer.session_orchestrator: brief not found at {brief_path}\n"
        )
        return 2

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    log_path = state_dir / "run_log.jsonl"
    hunks_count = 0
    pipeline_status = "ok"

    try:
        from designer.orchestrator import build_pipeline

        pipeline, run_id = build_pipeline(
            brief_path=brief_path,
            state_dir=state_dir,
            resume=args.resume,
        )
        stats = asyncio.run(pipeline.run(run_id=run_id))
        sys.stdout.write(
            f"designer.session_orchestrator: completed run_id={run_id} "
            f"queries={stats.queries_completed} candidates={stats.candidates_discovered}.\n"
        )
    except Exception as exc:  # noqa: BLE001 — surface runtime failure cleanly
        pipeline_status = "error"
        sys.stderr.write(
            f"designer.session_orchestrator: pipeline failed ({exc!r}).\n"
        )
        log_event(
            log_path,
            "pipeline_error",
            error=str(exc),
            error_class=type(exc).__name__,
            stage="pipeline",
        )
        return_code = 1
    else:
        return_code = 0

    try:
        from designer.run_end import run_end_designer_rubric_refinement

        hunks = run_end_designer_rubric_refinement(
            brief_path=brief_path, state_dir=state_dir
        )
        hunks_count = len(hunks)
        sys.stdout.write(
            f"designer.session_orchestrator: rubric-refinement run-end "
            f"persisted {hunks_count} proposed hunk(s).\n"
        )
    except Exception as exc:  # noqa: BLE001 — synthesis must not fail the run
        pipeline_status = (
            "rubric_refinement_failed" if pipeline_status == "ok" else pipeline_status
        )
        sys.stderr.write(
            f"designer.session_orchestrator: rubric-refinement run-end "
            f"hook failed ({exc!r}); persisted nothing.\n"
        )
        log_event(
            log_path,
            "pipeline_error",
            error=str(exc),
            error_class=type(exc).__name__,
            stage="rubric_refinement",
        )

    log_event(log_path, "designer_run_end", status=pipeline_status, hunks_count=hunks_count)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
