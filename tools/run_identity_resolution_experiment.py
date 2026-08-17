"""Run retrieval-only GitHub→LinkedIn identity-resolution experiments."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from github.identity_resolution_experiment_report import (
    write_identity_resolution_experiment_jsonl,
    write_identity_resolution_preview_markdown,
    write_identity_resolution_preview_summary,
    write_identity_resolution_experiment_markdown,
    write_identity_resolution_experiment_strategy_csvs,
    write_identity_resolution_experiment_summary,
)
from github.reconciliation_input import (
    export_identity_resolution_experiment_cohort,
)
from linkedin.identity_resolution_experiment import (
    DEFAULT_STRATEGY_ORDER,
    IdentityResolutionExperimentRunner,
    build_strategy_instances,
    load_experiment_inputs,
)
from shared.identity_experiment_schemas import IdentityResolutionGoldLabel
from shared.storage import read_json, read_jsonl


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple retrieval surfaces for GitHub→LinkedIn identity resolution."
    )
    parser.add_argument("--github-output-dir", required=True, help="GitHub run output directory with candidates.jsonl and final_judgments.jsonl")
    parser.add_argument("--output-dir", help="Where to write experiment artifacts; defaults to <github-output-dir>/identity-resolution-experiment")
    parser.add_argument("--gold-labels", help="Completed gold-label JSON file. If omitted, the command only exports the cohort/template.")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGY_ORDER), help="Comma-separated strategy names to run")
    parser.add_argument("--recruiter-search-url", help="Recruiter search URL required when recruiter_name_city is enabled")
    parser.add_argument("--primary-bucket-size", type=int, default=10, help="Leads to sample into each primary cohort bucket")
    parser.add_argument("--sanity-size", type=int, default=10, help="Leads to sample into the sanity cohort")
    parser.add_argument("--seed", type=int, default=17, help="Fixed seed for randomized per-lead strategy ordering")
    parser.add_argument("--write-strategy-csvs", action="store_true", help="Also write per-strategy CSVs")
    parser.add_argument("--preview-only", action="store_true", help="Run the strategies without gold-label scoring and inspect surfaced URLs qualitatively")
    return parser


def _load_gold_labels(path: str | Path) -> dict[str, IdentityResolutionGoldLabel]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        payload = read_jsonl(path)
    else:
        payload = read_json(path)
        if not isinstance(payload, list):
            raise ValueError("Gold labels file must contain a JSON list or JSONL records")
    rows = [IdentityResolutionGoldLabel.from_dict(item) for item in payload if isinstance(item, dict)]
    missing = [row.github_username for row in rows if not row.gold_outcome]
    if missing:
        raise ValueError(
            "Gold labels are incomplete for: "
            + ", ".join(sorted(set(missing))[:10])
        )
    return {row.github_username: row for row in rows}


async def _run(args: argparse.Namespace) -> None:
    github_output_dir = Path(args.github_output_dir)
    output_dir = Path(args.output_dir) if args.output_dir else github_output_dir / "identity-resolution-experiment"
    export_paths = export_identity_resolution_experiment_cohort(
        github_output_dir,
        output_dir,
        primary_bucket_size=max(args.primary_bucket_size, 1),
        sanity_size=max(args.sanity_size, 0),
    )
    print(f"Wrote cohort export: {export_paths['primary']}")
    print(f"Wrote sanity export: {export_paths['sanity']}")
    print(f"Wrote gold-label template: {export_paths['gold_template']}")

    if not args.gold_labels and not args.preview_only:
        print("No --gold-labels provided; stopping after cohort/template export.")
        return

    strategy_names = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if "recruiter_name_city" in strategy_names and not args.recruiter_search_url:
        raise ValueError("--recruiter-search-url is required when recruiter_name_city is enabled")

    primary_leads, sanity_leads = load_experiment_inputs(
        str(github_output_dir),
        primary_bucket_size=max(args.primary_bucket_size, 1),
        sanity_size=max(args.sanity_size, 0),
    )
    gold_labels = _load_gold_labels(args.gold_labels) if args.gold_labels else None
    runner = IdentityResolutionExperimentRunner(
        strategies=build_strategy_instances(strategy_names),
        recruiter_search_url=args.recruiter_search_url or "",
        seed=args.seed,
    )
    rows, tracker_state = await runner.run(primary_leads + sanity_leads, gold_labels=gold_labels)

    jsonl_path = write_identity_resolution_experiment_jsonl(
        output_dir / "identity_resolution_experiment.jsonl",
        rows,
    )
    if args.preview_only:
        summary_json_path = write_identity_resolution_preview_summary(
            output_dir / "identity_resolution_experiment_preview_summary.json",
            rows,
            tracker_state=tracker_state,
        )
        summary_md_path = write_identity_resolution_preview_markdown(
            output_dir / "identity_resolution_experiment_preview_summary.md",
            rows,
            tracker_state=tracker_state,
        )
    else:
        summary_json_path = write_identity_resolution_experiment_summary(
            output_dir / "identity_resolution_experiment_summary.json",
            rows,
            tracker_state=tracker_state,
        )
        summary_md_path = write_identity_resolution_experiment_markdown(
            output_dir / "identity_resolution_experiment_summary.md",
            rows,
            tracker_state=tracker_state,
        )
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_json_path}")
    print(f"Wrote {summary_md_path}")
    if args.write_strategy_csvs:
        csv_paths = write_identity_resolution_experiment_strategy_csvs(output_dir / "strategy_csvs", rows)
        for path in csv_paths:
            print(f"Wrote {path}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
