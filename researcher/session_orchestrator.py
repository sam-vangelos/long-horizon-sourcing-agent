"""Researcher session orchestrator — Slice 6.

CLI entry point used by `cloris.launchers._researcher_orchestrator_argv`
and the frozen-app dispatch in `cloris/worker.py:295-304`. Wires the
`ResearcherPipeline` (Slice 6) against the runtime-state store and the
real OpenAlex / Opus clients.

Argv shape (per Slice 1):
  --brief PATH         Path to the brief JSON.
  --state-dir PATH     Per-source state directory.
  --resume             Continue an interrupted run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from researcher.orchestrator import ResearcherPipeline, build_pipeline


logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researcher.session_orchestrator",
        description="Researcher session orchestrator (Slice 6).",
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
        help="Continue an interrupted run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Slice 6 entrypoint.

    Builds the pipeline + runs to completion, returning the process
    exit code. Failures during HTTP / LLM calls are caught at the
    pipeline boundary so a transient error doesn't kill the whole run;
    the runtime-state store retains everything written before the
    failure for clean resume.
    """

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # P7.5(b) — ``--resume`` was theater: parsed here but never forwarded
    # to :func:`build_pipeline`, which hardcodes
    # ``bridge.start_or_resume_run(resume=False)`` (researcher/
    # orchestrator.py). The flag silently re-ran from scratch instead of
    # resuming or failing loudly. Researcher does not support resume yet,
    # so the flag now exits with a clear, honest error instead of lying
    # about what it did.
    if args.resume:
        sys.stderr.write(
            "researcher.session_orchestrator: resume not implemented for "
            "researcher; rerun without --resume to start a fresh run.\n"
        )
        return 2

    brief_path = Path(args.brief)
    if not brief_path.exists():
        sys.stderr.write(
            f"researcher.session_orchestrator: brief not found at {brief_path}\n"
        )
        return 2

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    polite_pool_email = os.environ.get("OPENALEX_POLITE_POOL_EMAIL", "")

    sys.stdout.write(
        f"researcher.session_orchestrator: starting "
        f"brief={brief_path} state_dir={state_dir} resume={args.resume}\n"
    )

    try:
        pipeline, run_id = build_pipeline(
            brief_path=brief_path,
            state_dir=state_dir,
            openalex_polite_pool_email=polite_pool_email,
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"researcher.session_orchestrator: failed to build pipeline: {exc}\n"
        )
        logger.exception("Pipeline construction failed")
        return 2

    sys.stdout.write(
        f"researcher.session_orchestrator: run_id={run_id}; entering main loop.\n"
    )

    try:
        stats = pipeline.run(run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"researcher.session_orchestrator: pipeline.run raised: {exc}\n"
        )
        logger.exception("Pipeline run failed")
        return 1

    sys.stdout.write(
        f"researcher.session_orchestrator: done. "
        f"queries={stats.queries_completed}/{stats.queries_total} "
        f"discovered={stats.candidates_discovered} "
        f"facial_yes={stats.facial_yes} facial_no={stats.facial_no} "
        f"saves={stats.saves} rejects={stats.rejects}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
