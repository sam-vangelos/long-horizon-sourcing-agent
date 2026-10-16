"""Reporting helpers for GitHub→LinkedIn reconciliation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from github.reconciliation_input import GitHubReconciliationLead
from shared.reconciliation_schemas import ReconciliationDecision
from shared.storage import write_json


def write_reconciliation_jsonl(
    path: str | Path,
    rows: list[dict],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def write_reconciliation_csv(
    path: str | Path,
    rows: list[dict],
) -> Path:
    columns = [
        "github_username",
        "candidate_name",
        "github_url",
        "linkedin_url_hint",
        "matched_profile_url",
        "match_confidence",
        "match_method",
        "matched_name",
        "matched_company",
        "matched_title",
        "matched_location",
        "novelty_pressure",
        "message_count",
        "project_count",
        "view_count",
        "reachout_status",
        "novelty_value",
        "action",
        "rationale",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def build_reconciliation_row(
    lead: GitHubReconciliationLead,
    decision: ReconciliationDecision,
) -> dict:
    match = decision.match_result
    assessment = decision.assessment
    activity = match.recruiter_activity if match else None
    return {
        "github_username": lead.username,
        "candidate_name": lead.candidate_name,
        "github_url": lead.github_url,
        "linkedin_url_hint": lead.linkedin_hints.linkedin_url_hint if lead.linkedin_hints else "",
        "matched_profile_url": match.matched_profile_url if match else "",
        "match_confidence": match.match_confidence if match else 0.0,
        "match_method": match.match_method if match else "",
        "matched_name": match.matched_name if match else "",
        "matched_company": match.matched_company if match else "",
        "matched_title": match.matched_title if match else "",
        "matched_location": match.matched_location if match else "",
        "novelty_pressure": match.novelty_pressure if match else "",
        "message_count": activity.message_count if activity else 0,
        "project_count": activity.project_count if activity else 0,
        "view_count": activity.view_count if activity else 0,
        "reachout_status": assessment.reachout_status if assessment else "",
        "novelty_value": assessment.novelty_value if assessment else "",
        "action": decision.action,
        "rationale": decision.rationale,
        "decision": decision.to_dict(),
        "lead": lead.to_dict(),
    }


def build_reconciliation_summary(rows: list[dict], *, input_stats: dict | None = None) -> dict:
    total = len(rows)
    action_counts: dict[str, int] = {}
    match_buckets = {
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0,
    }
    for row in rows:
        action = str(row.get("action", "") or "")
        action_counts[action] = action_counts.get(action, 0) + 1
        confidence = float(row.get("match_confidence", 0.0) or 0.0)
        if confidence >= 0.85:
            match_buckets["high_confidence"] += 1
        elif confidence >= 0.6:
            match_buckets["medium_confidence"] += 1
        else:
            match_buckets["low_confidence"] += 1
    return {
        "total_leads": total,
        "action_counts": action_counts,
        "match_confidence_distribution": match_buckets,
        "low_confidence_rate": round(match_buckets["low_confidence"] / max(total, 1), 4),
        "manual_review_rate": round(action_counts.get("manual_review", 0) / max(total, 1), 4),
        "input_stats": input_stats or {},
    }


def write_reconciliation_summary(path: str | Path, rows: list[dict], *, input_stats: dict | None = None) -> Path:
    path = Path(path)
    write_json(path, build_reconciliation_summary(rows, input_stats=input_stats))
    return path
