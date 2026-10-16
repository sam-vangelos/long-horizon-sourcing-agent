"""Writers for Recruiter-first reconciliation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from shared.recruiter_identity_schemas import RecruiterIdentityResolution


def build_recruiter_identity_row(result: RecruiterIdentityResolution) -> dict:
    row = result.to_dict()
    row["plausible_profile_reviews_count"] = len(result.plausible_profile_reviews)
    row["ambiguity_multi_review"] = result.ambiguity_multi_review
    top1 = result.top_candidates[0] if result.top_candidates else None
    row["top_candidate_name"] = top1.name if top1 else ""
    row["top_candidate_profile_url"] = top1.profile_url if top1 else ""
    row["top_candidate_company"] = top1.current_company if top1 else ""
    row["top_candidate_location"] = top1.location if top1 else ""
    row["top_candidate_confidence"] = top1.match_confidence if top1 else 0.0
    profile_status = row.get("profile_status") if isinstance(row.get("profile_status"), dict) else {}
    row["profile_saved_by"] = str(profile_status.get("saved_by", "") or "")
    row["profile_message_count"] = int(profile_status.get("message_count", 0) or 0)
    row["profile_project_count"] = int(profile_status.get("project_count", 0) or 0)
    row["profile_view_count"] = int(profile_status.get("view_count", 0) or 0)
    row["profile_last_outbound_contact"] = str(profile_status.get("last_outbound_contact", "") or "")
    return row


def write_recruiter_identity_jsonl(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


# Recruiter-Identity-Collection-Review-Fixes §1: identity_collect rows whose
# Recruiter save click did not persist must NOT be exported as
# "saved/collected", even when identity is confirmed. The canonical row keeps
# collection_action == "COLLECT" (identity truth) but the operational export
# reflects Recruiter persistence truth, so save_failed rows are excluded here.
_IDENTITY_COLLECT_SAVED_EXPORT_STATES = frozenset(
    {
        "saved_now",
        "already_saved",
        "dry_run_skipped",
    }
)


def _row_is_saved_export(row: dict) -> bool:
    """Return True when a resolution row should appear in the "saved/collected" export.

    Mode-aware filter:
      - ``identity_collect``: included when
        ``collection_action == "COLLECT"`` AND
        ``project_save_state in {saved_now, already_saved, dry_run_skipped}``.
        This excludes ``save_failed`` rows (Recruiter-Identity-Collection-Review-
        Fixes §1) so the export reflects actual Recruiter persistence rather
        than identity confirmation alone.
      - ``fit_gated_save`` / legacy: included when
        ``final_action == "SAVE"`` (preserves backward compatibility for the
        existing fit-gated workflow).
    """
    workflow_mode = str(row.get("workflow_mode", "") or "").strip()
    if workflow_mode == "identity_collect":
        if str(row.get("collection_action", "") or "").strip() != "COLLECT":
            return False
        save_state = str(row.get("project_save_state", "") or "").strip()
        return save_state in _IDENTITY_COLLECT_SAVED_EXPORT_STATES
    return str(row.get("final_action", "") or "").strip() == "SAVE"


def write_recruiter_reconciliation_saved_jsonl(path: str | Path, rows: list[dict]) -> Path:
    saved = [row for row in rows if _row_is_saved_export(row)]
    return write_recruiter_identity_jsonl(path, saved)


def write_recruiter_identity_csv(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "github_username",
        "candidate_name",
        "lookup_name",
        "search_location",
        "query",
        "workflow_mode",
        "identity_status",
        "identity_subreason",
        "collection_action",
        "collection_subreason",
        "project_save_state",
        "resolved_query",
        "stop_reason",
        "final_action",
        "final_subreason",
        "identity_classification",
        "linkedin_brief_path",
        "holistic_fit_decision",
        "holistic_fit_confidence",
        "holistic_fit_path",
        "holistic_fit_rationale",
        "rationale",
        "selected_candidate_rank",
        "selected_profile_url",
        "already_saved",
        "opened_profile",
        "recruiter_save_attempted",
        "recruiter_save_succeeded",
        "had_plausible_cards",
        "extraction_failed",
        "ambiguity_multi_review",
        "plausible_profile_reviews_count",
        "novelty_pressure",
        "reachout_status",
        "top_candidate_name",
        "top_candidate_profile_url",
        "top_candidate_company",
        "top_candidate_location",
        "top_candidate_confidence",
        "profile_saved_by",
        "profile_message_count",
        "profile_project_count",
        "profile_view_count",
        "profile_last_outbound_contact",
        "github_company",
        "github_location",
        "github_title",
        "queries_tried",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            queries_tried_value = row.get("queries_tried", []) or []
            writer.writerow(
                {
                    "github_username": row.get("github_username", ""),
                    "candidate_name": row.get("candidate_name", ""),
                    "lookup_name": row.get("lookup_name", ""),
                    "search_location": row.get("search_location", ""),
                    "query": row.get("query", ""),
                    "workflow_mode": row.get("workflow_mode", ""),
                    "identity_status": row.get("identity_status", ""),
                    "identity_subreason": row.get("identity_subreason", ""),
                    "collection_action": row.get("collection_action", ""),
                    "collection_subreason": row.get("collection_subreason", ""),
                    "project_save_state": row.get("project_save_state", ""),
                    "resolved_query": row.get("resolved_query", ""),
                    "stop_reason": row.get("stop_reason", ""),
                    "final_action": row.get("final_action", ""),
                    "final_subreason": row.get("final_subreason", ""),
                    "identity_classification": row.get("identity_classification", ""),
                    "linkedin_brief_path": row.get("linkedin_brief_path", ""),
                    "holistic_fit_decision": row.get("holistic_fit_decision", ""),
                    "holistic_fit_confidence": row.get("holistic_fit_confidence", 0.0),
                    "holistic_fit_path": row.get("holistic_fit_path", ""),
                    "holistic_fit_rationale": row.get("holistic_fit_rationale", ""),
                    "rationale": row.get("rationale", ""),
                    "selected_candidate_rank": row.get("selected_candidate_rank", 0),
                    "selected_profile_url": row.get("selected_profile_url", ""),
                    "already_saved": row.get("already_saved", False),
                    "opened_profile": row.get("opened_profile", False),
                    "recruiter_save_attempted": row.get("recruiter_save_attempted", False),
                    "recruiter_save_succeeded": row.get("recruiter_save_succeeded", False),
                    "had_plausible_cards": row.get("had_plausible_cards", False),
                    "extraction_failed": row.get("extraction_failed", False),
                    "ambiguity_multi_review": row.get("ambiguity_multi_review", False),
                    "plausible_profile_reviews_count": row.get("plausible_profile_reviews_count", 0),
                    "novelty_pressure": row.get("novelty_pressure", ""),
                    "reachout_status": row.get("reachout_status", ""),
                    "top_candidate_name": row.get("top_candidate_name", ""),
                    "top_candidate_profile_url": row.get("top_candidate_profile_url", ""),
                    "top_candidate_company": row.get("top_candidate_company", ""),
                    "top_candidate_location": row.get("top_candidate_location", ""),
                    "top_candidate_confidence": row.get("top_candidate_confidence", 0.0),
                    "profile_saved_by": row.get("profile_saved_by", ""),
                    "profile_message_count": row.get("profile_message_count", 0),
                    "profile_project_count": row.get("profile_project_count", 0),
                    "profile_view_count": row.get("profile_view_count", 0),
                    "profile_last_outbound_contact": row.get("profile_last_outbound_contact", ""),
                    "github_company": row.get("github_company", ""),
                    "github_location": row.get("github_location", ""),
                    "github_title": row.get("github_title", ""),
                    "queries_tried": " | ".join(str(q) for q in queries_tried_value),
                    "notes": " | ".join(row.get("notes", [])),
                }
            )
    return path


def write_recruiter_reconciliation_saved_csv(path: str | Path, rows: list[dict]) -> Path:
    saved = [row for row in rows if _row_is_saved_export(row)]
    return write_recruiter_identity_csv(path, saved)


# Recruiter-Identity-Collection-Summary-Alignment §1: identity_collect rows
# carry compatibility ``final_action == "SAVE"`` for confirmed identities, but
# the summary must NOT count a row as ``SAVE`` when the Recruiter save click
# did not persist (``project_save_state == "save_failed"``). The same set of
# save states that appear in the saved/collected export count as ``SAVE`` here
# so the headline counts and the saved export agree.
_IDENTITY_COLLECT_SAVE_STATES = frozenset(
    {
        "saved_now",
        "already_saved",
        "dry_run_skipped",
    }
)


def _summary_action_for_row(row: dict) -> tuple[str, str]:
    """Return a normalized ``(action, subreason)`` tuple for summary counting.

    Mode-aware (Recruiter-Identity-Collection-Summary-Alignment §1):

    - ``identity_collect`` rows summarize from the identity-collection canonical
      fields, NOT from legacy ``final_action`` (which may still say ``"SAVE"``
      for compatibility even when the save click failed):
        * ``collection_action == "COLLECT"`` AND
          ``project_save_state in {saved_now, already_saved, dry_run_skipped}``
          → ``("SAVE", final_subreason)``
        * ``collection_action == "COLLECT"`` AND
          ``project_save_state == "save_failed"``
          → ``("MANUAL_REVIEW", "tool_failure")``
        * ``collection_action == "MANUAL_REVIEW"`` → ``("MANUAL_REVIEW", collection_subreason or final_subreason)``
        * ``collection_action == "REJECT"`` → ``("REJECT", collection_subreason or final_subreason)``
        * other / missing → fall back to legacy ``final_action`` / ``final_subreason``
    - ``fit_gated_save`` and legacy rows keep the prior behavior:
      ``(final_action, final_subreason)``.

    Subreasons are coupled to the normalized action so action_counts and
    subreason_counts stay internally consistent.
    """
    workflow_mode = str(row.get("workflow_mode", "") or "").strip()
    legacy_action = str(row.get("final_action", "") or "").strip() or "unknown"
    legacy_subreason = str(row.get("final_subreason", "") or "").strip()

    if workflow_mode != "identity_collect":
        return legacy_action, legacy_subreason

    collection_action = str(row.get("collection_action", "") or "").strip()
    collection_subreason = str(row.get("collection_subreason", "") or "").strip()
    project_save_state = str(row.get("project_save_state", "") or "").strip()

    if collection_action == "COLLECT":
        if project_save_state in _IDENTITY_COLLECT_SAVE_STATES:
            return "SAVE", legacy_subreason
        if project_save_state == "save_failed":
            return "MANUAL_REVIEW", "tool_failure"
        # COLLECT without a recognized save state (e.g. partial/legacy row).
        # Fall back to legacy fields rather than guessing a synthetic bucket.
        return legacy_action, legacy_subreason
    if collection_action == "MANUAL_REVIEW":
        return "MANUAL_REVIEW", collection_subreason or legacy_subreason
    if collection_action == "REJECT":
        return "REJECT", collection_subreason or legacy_subreason

    return legacy_action, legacy_subreason


def build_recruiter_identity_summary(
    rows: list[dict],
    *,
    input_stats: dict | None = None,
) -> dict:
    total = len(rows)
    action_counts: dict[str, int] = {}
    subreason_counts: dict[str, int] = {}
    identity_classification_counts: dict[str, int] = {}
    opened_profiles = 0
    top1_saved = 0
    full_judge_calls = 0
    novelty_counts: dict[str, int] = {}
    reachout_counts: dict[str, int] = {}
    workflow_mode_counts: dict[str, int] = {}
    identity_status_counts: dict[str, int] = {}
    collection_action_counts: dict[str, int] = {}
    project_save_state_counts: dict[str, int] = {}
    for row in rows:
        action, sub = _summary_action_for_row(row)
        action_counts[action] = action_counts.get(action, 0) + 1
        if sub:
            subreason_counts[sub] = subreason_counts.get(sub, 0) + 1
        classification = str(row.get("identity_classification", "") or "").strip()
        if classification:
            identity_classification_counts[classification] = (
                identity_classification_counts.get(classification, 0) + 1
            )
        if row.get("opened_profile"):
            opened_profiles += 1
        novelty = str(row.get("novelty_pressure", "") or "").strip()
        if novelty:
            novelty_counts[novelty] = novelty_counts.get(novelty, 0) + 1
        reachout = str(row.get("reachout_status", "") or "").strip()
        if reachout:
            reachout_counts[reachout] = reachout_counts.get(reachout, 0) + 1
        workflow_mode = str(row.get("workflow_mode", "") or "").strip()
        if workflow_mode:
            workflow_mode_counts[workflow_mode] = workflow_mode_counts.get(workflow_mode, 0) + 1
        identity_status = str(row.get("identity_status", "") or "").strip()
        if identity_status:
            identity_status_counts[identity_status] = (
                identity_status_counts.get(identity_status, 0) + 1
            )
        collection_action = str(row.get("collection_action", "") or "").strip()
        if collection_action:
            collection_action_counts[collection_action] = (
                collection_action_counts.get(collection_action, 0) + 1
            )
        project_save_state = str(row.get("project_save_state", "") or "").strip()
        if project_save_state:
            project_save_state_counts[project_save_state] = (
                project_save_state_counts.get(project_save_state, 0) + 1
            )
        top_candidates = row.get("top_candidates", [])
        if isinstance(top_candidates, list) and top_candidates:
            candidate = top_candidates[0]
            if isinstance(candidate, dict) and candidate.get("already_saved"):
                top1_saved += 1
        # Recruiter-Identity-Collection-Review-Fixes §2: count a review only
        # when it actually carries fit-evaluation evidence (holistic_fit_decision
        # or holistic_fit_path populated). identity_collect now opens profiles
        # for identity confirmation without calling full_judge, so review
        # existence alone overstates judge usage. Use the review's own evidence
        # fields, not mode assumptions, so legacy / mixed historical artifacts
        # still summarize correctly.
        reviews = row.get("plausible_profile_reviews", [])
        if isinstance(reviews, list):
            for review in reviews:
                if not isinstance(review, dict):
                    continue
                if review.get("extraction_failed"):
                    continue
                fit_decision = str(review.get("holistic_fit_decision", "") or "").strip()
                fit_path = str(review.get("holistic_fit_path", "") or "").strip()
                if fit_decision or fit_path:
                    full_judge_calls += 1
    return {
        "total_leads": total,
        "action_counts": action_counts,
        "subreason_counts": subreason_counts,
        "identity_classification_counts": identity_classification_counts,
        "workflow_mode_counts": workflow_mode_counts,
        "identity_status_counts": identity_status_counts,
        "collection_action_counts": collection_action_counts,
        "project_save_state_counts": project_save_state_counts,
        "opened_profile_count": opened_profiles,
        "full_judge_call_count": full_judge_calls,
        "top1_already_saved_count": top1_saved,
        "novelty_counts": novelty_counts,
        "reachout_counts": reachout_counts,
        "input_stats": input_stats or {},
    }


def write_recruiter_identity_summary(
    path: str | Path,
    rows: list[dict],
    *,
    input_stats: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_recruiter_identity_summary(rows, input_stats=input_stats)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return path
