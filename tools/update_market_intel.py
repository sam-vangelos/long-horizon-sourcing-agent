#!/usr/bin/env python3
"""Update the per-role market-intelligence artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_intelligence import (
    HeuristicPlannerBackend,
    resolve_market_intel_agent_state_path,
    resolve_market_intel_artifact_path,
    resolve_market_intel_research_log_path,
    update_market_intel,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update the per-role market intelligence artifact"
    )
    parser.add_argument("--brief", required=True, help="Path to the source brief JSON")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Finalized run snapshot directory under output/runs/... to ingest",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="When backfilling from a legacy directory that contains multiple runs, target this specific runtime_state run ID",
    )
    parser.add_argument(
        "--legacy-output-dir",
        default=None,
        help="Legacy mixed output directory to normalize into output/runs/... before ingestion",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Deprecated alias for --run-dir in post_run/scheduled mode or --legacy-output-dir in backfill mode",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional explicit path to run-report.json",
    )
    parser.add_argument(
        "--mode",
        choices=("post_run", "scheduled", "backfill"),
        default="post_run",
        help="Whether this update is tied to a just-finished run, a scheduled refresh, or a historical backfill",
    )
    parser.add_argument(
        "--reconstruct-report-analysis",
        action="store_true",
        help="Reconstruct report-analysis fields from raw run artifacts when a structured report is missing",
    )
    parser.add_argument(
        "--with-external-research",
        action="store_true",
        help="Attempt external research if a backend is configured",
    )
    parser.add_argument(
        "--force-external-research",
        action="store_true",
        help="Force the general external-research leg to run even if the planner would skip it",
    )
    parser.add_argument(
        "--force-edge-case-research",
        action="store_true",
        help="Force the edge-case external-research leg to run even if the planner would skip it",
    )
    parser.add_argument(
        "--heuristic-planner",
        action="store_true",
        help="Use the heuristic planner instead of the default LLM planner to avoid planner latency during replays",
    )
    parser.add_argument(
        "--allow-live-state-dir",
        action="store_true",
        help="Debug-only escape hatch: if pointed at output/state/... or another mutable directory, import it into output/runs/... first",
    )
    args = parser.parse_args()

    external_backend = None
    planner_backend = HeuristicPlannerBackend() if args.heuristic_planner else None
    if args.with_external_research:
        try:
            from market_intelligence.research_agent import (
                build_external_research_backend,
            )

            external_backend = build_external_research_backend()
        except Exception as exc:
            print(f"[warn] Could not initialize research backend: {exc}", file=sys.stderr)

    try:
        artifact = update_market_intel(
            brief_path=args.brief,
            run_dir=args.run_dir,
            run_id=args.run_id,
            legacy_output_dir=args.legacy_output_dir,
            output_dir=args.output_dir,
            report_path=args.report,
            mode=args.mode,
            external_research_backend=external_backend,
            planner_backend=planner_backend,
            with_external_research=args.with_external_research,
            force_external_research=args.force_external_research,
            force_edge_case_research=args.force_edge_case_research,
            allow_live_state_dir=args.allow_live_state_dir,
            reconstruct_report_analysis=args.reconstruct_report_analysis,
        )
    except Exception as exc:
        print(f"[error] Market intel update failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    indexed_runs = [
        record
        for record in artifact.evidence_index.get("runs", [])
        if isinstance(record, dict)
    ]
    resolved_run_dir = None
    selected_record = None
    if args.run_dir:
        selected_record = next(
            (
                record
                for record in indexed_runs
                if (record.get("run_dir") or record.get("output_dir")) == str(Path(args.run_dir).resolve())
            ),
            None,
        )
    if selected_record is None and args.run_id is not None:
        selected_record = next(
            (record for record in indexed_runs if record.get("run_id") == args.run_id),
            None,
        )
    if selected_record is None:
        selected_record = max(
            indexed_runs,
            key=lambda record: str(record.get("generated_at", "")),
            default=None,
        )
    if selected_record:
        resolved_run_dir = selected_record.get("run_dir") or selected_record.get("output_dir")
    artifact_path = resolve_market_intel_artifact_path(
        args.brief,
        output_dir=resolved_run_dir or args.run_dir or args.output_dir or args.legacy_output_dir,
    )
    print("Market intelligence updated.")
    print(f"  Artifact: {artifact_path}")
    print(f"  Markdown: {artifact_path.parent / 'market-intel.md'}")
    print(f"  Technical appendix: {artifact_path.parent / 'market-intel-technical.md'}")
    print(
        "  Agent state: "
        f"{resolve_market_intel_agent_state_path(args.brief, output_dir=resolved_run_dir or args.run_dir or args.output_dir or args.legacy_output_dir)}"
    )
    print(
        "  Research log: "
        f"{resolve_market_intel_research_log_path(args.brief, output_dir=resolved_run_dir or args.run_dir or args.output_dir or args.legacy_output_dir)}"
    )
    print(f"  Version:  {artifact.artifact_version}")
    if resolved_run_dir:
        print(f"  Run dir:   {resolved_run_dir}")
        print(
            "  Research input: "
            f"{Path(resolved_run_dir).resolve() / 'market-intel-research-input.json'}"
        )


if __name__ == "__main__":
    main()
