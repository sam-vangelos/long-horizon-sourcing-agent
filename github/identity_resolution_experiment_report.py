"""Reporting helpers for identity-resolution retrieval experiments."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from linkedin.identity_resolution_experiment import summarize_experiment_rows
from shared.identity_experiment_schemas import StrategyExecutionRecord
from shared.storage import write_json


def write_identity_resolution_experiment_jsonl(
    path: str | Path,
    rows: list[StrategyExecutionRecord],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict()) + "\n")
    return path


def write_identity_resolution_experiment_strategy_csvs(
    output_dir: str | Path,
    rows: list[StrategyExecutionRecord],
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "strategy_name",
        "github_username",
        "candidate_name",
        "cohort_kind",
        "cohort_bucket",
        "query",
        "location_filter",
        "aborted",
        "abort_reason",
        "final_url",
        "page_title",
        "body_excerpt",
        "top1_profile_url",
        "top1_correct",
        "top3_contains_correct",
        "wrong_person_top1",
        "manual_review_required",
        "no_candidate",
        "blocker_state",
        "duration_seconds",
        "interaction_count",
        "notes",
    ]
    paths: list[Path] = []
    by_strategy: dict[str, list[StrategyExecutionRecord]] = {}
    for row in rows:
        by_strategy.setdefault(row.strategy_name, []).append(row)

    for strategy_name, strategy_rows in by_strategy.items():
        path = output_dir / f"{strategy_name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in strategy_rows:
                payload = row.to_dict()
                payload["notes"] = " | ".join(row.notes)
                writer.writerow({column: payload.get(column, "") for column in columns})
        paths.append(path)
    return paths


def write_identity_resolution_experiment_summary(
    path: str | Path,
    rows: list[StrategyExecutionRecord],
    *,
    tracker_state: dict[str, dict] | None = None,
) -> Path:
    path = Path(path)
    write_json(path, summarize_experiment_rows(rows, tracker_state=tracker_state))
    return path


def write_identity_resolution_experiment_markdown(
    path: str | Path,
    rows: list[StrategyExecutionRecord],
    *,
    tracker_state: dict[str, dict] | None = None,
) -> Path:
    summary = summarize_experiment_rows(rows, tracker_state=tracker_state)
    lines = [
        "# Identity Resolution Retrieval Experiment",
        "",
        f"- Total primary leads: {summary['total_primary_leads']}",
        f"- Decision: {summary['decision']['decision']}",
        f"- Winner: {summary['decision']['winner'] or 'None'}",
        f"- Reason: {summary['decision']['reason']}",
        "",
        "## Strategy Metrics",
        "",
        "| Strategy | Top-1 | Top-3 | Wrong top-1 | Manual review | No candidate | Median sec | Eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for strategy_name, strategy_summary in sorted(summary["strategies"].items()):
        metrics = strategy_summary["aggregate_metrics"]
        eligible = "default" if strategy_summary["default_eligible"] else ("viable" if strategy_summary["viable"] else "no")
        lines.append(
            f"| {strategy_name} | "
            f"{metrics['top1_correct_rate']:.2f} | "
            f"{metrics['top3_contains_correct_rate']:.2f} | "
            f"{metrics['wrong_person_top1_rate']:.2f} | "
            f"{metrics['manual_review_rate']:.2f} | "
            f"{metrics['no_candidate_rate']:.2f} | "
            f"{metrics['median_seconds_per_lead']:.1f} | "
            f"{eligible} |"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_identity_resolution_preview_summary(
    rows: list[StrategyExecutionRecord],
    *,
    tracker_state: dict[str, dict] | None = None,
) -> dict:
    strategies: dict[str, list[StrategyExecutionRecord]] = {}
    for row in rows:
        strategies.setdefault(row.strategy_name, []).append(row)

    summary: dict[str, dict] = {}
    for strategy_name, strategy_rows in sorted(strategies.items()):
        executed_rows = [row for row in strategy_rows if not row.aborted]
        surfaced_counts = [len(row.surfaced_candidates) for row in executed_rows]
        summary[strategy_name] = {
            "attempted_leads": len(executed_rows),
            "aborted_leads": sum(1 for row in strategy_rows if row.aborted),
            "coverage_count": sum(1 for row in executed_rows if row.surfaced_candidates),
            "coverage_rate": round(sum(1 for row in executed_rows if row.surfaced_candidates) / max(len(executed_rows), 1), 4),
            "median_seconds_per_lead": round(statistics.median(row.duration_seconds for row in executed_rows), 3)
            if executed_rows
            else 0.0,
            "median_interactions_per_lead": round(statistics.median(row.interaction_count for row in executed_rows), 3)
            if executed_rows
            else 0.0,
            "mean_candidates_surfaced": round(sum(surfaced_counts) / max(len(surfaced_counts), 1), 3),
            "blocker_counts": {
                key: value
                for key, value in {
                    blocker: sum(1 for row in executed_rows if row.blocker_state == blocker)
                    for blocker in sorted({row.blocker_state for row in executed_rows if row.blocker_state})
                }.items()
                if value
            },
            "tracker": (tracker_state or {}).get(strategy_name, {}),
        }
    return {
        "mode": "preview_only",
        "strategies": summary,
    }


def write_identity_resolution_preview_summary(
    path: str | Path,
    rows: list[StrategyExecutionRecord],
    *,
    tracker_state: dict[str, dict] | None = None,
) -> Path:
    path = Path(path)
    write_json(path, build_identity_resolution_preview_summary(rows, tracker_state=tracker_state))
    return path


def write_identity_resolution_preview_markdown(
    path: str | Path,
    rows: list[StrategyExecutionRecord],
    *,
    tracker_state: dict[str, dict] | None = None,
) -> Path:
    summary = build_identity_resolution_preview_summary(rows, tracker_state=tracker_state)
    lines = [
        "# Identity Resolution Retrieval Preview",
        "",
        "This run is qualitative only. Use the JSONL and per-strategy CSVs to inspect surfaced URLs and decide whether a full gold-label benchmark is warranted.",
        "",
        "| Strategy | Leads | Aborted | Coverage | Mean surfaced | Median sec | Median interactions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy_name, strategy_summary in summary["strategies"].items():
        lines.append(
            f"| {strategy_name} | "
            f"{strategy_summary['attempted_leads']} | "
            f"{strategy_summary['aborted_leads']} | "
            f"{strategy_summary['coverage_rate']:.2f} | "
            f"{strategy_summary['mean_candidates_surfaced']:.2f} | "
            f"{strategy_summary['median_seconds_per_lead']:.1f} | "
            f"{strategy_summary['median_interactions_per_lead']:.1f} |"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
