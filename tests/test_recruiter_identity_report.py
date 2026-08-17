import json

from github.recruiter_identity_report import (
    build_recruiter_identity_row,
    build_recruiter_identity_summary,
    write_recruiter_reconciliation_saved_jsonl,
)
from shared.recruiter_identity_schemas import PlausibleProfileReview, RecruiterIdentityResolution


def test_build_recruiter_identity_summary_counts_actions_and_saved_top_cards():
    rows = [
        {
            "final_action": "SAVE",
            "final_subreason": "",
            "opened_profile": True,
            "novelty_pressure": "medium",
            "reachout_status": "messaged",
            "top_candidates": [
                {"already_saved": True},
            ],
        },
        {
            "final_action": "MANUAL_REVIEW",
            "final_subreason": "identity_ambiguous",
            "opened_profile": False,
            "novelty_pressure": "low",
            "top_candidates": [
                {"already_saved": False},
            ],
        },
    ]

    summary = build_recruiter_identity_summary(rows, input_stats={"processed_leads": 2})

    assert summary["total_leads"] == 2
    assert summary["action_counts"]["SAVE"] == 1
    assert summary["action_counts"]["MANUAL_REVIEW"] == 1
    assert summary["subreason_counts"]["identity_ambiguous"] == 1
    assert summary["opened_profile_count"] == 1
    assert summary["top1_already_saved_count"] == 1
    assert summary["novelty_counts"]["medium"] == 1
    assert summary["reachout_counts"]["messaged"] == 1
    assert summary["input_stats"]["processed_leads"] == 2


def test_build_recruiter_identity_summary_counts_identity_classification_and_full_judge_calls():
    """Review-Fixes §2: full_judge_call_count is now evidence-based — only
    reviews that actually carry holistic_fit_decision or holistic_fit_path
    count, regardless of how many reviews are present on the row."""
    rows = [
        {
            "final_action": "REJECT",
            "final_subreason": "no_plausible_profile",
            "identity_classification": "no_confident_match",
            "opened_profile": False,
            "plausible_profile_reviews": [],
        },
        {
            "final_action": "REJECT",
            "final_subreason": "no_plausible_profile",
            "identity_classification": "no_confident_match",
            "opened_profile": False,
            "plausible_profile_reviews": [],
        },
        {
            "final_action": "REJECT",
            "final_subreason": "fit_reject",
            "identity_classification": "single_strong_plausible_profile",
            "opened_profile": True,
            "plausible_profile_reviews": [
                {
                    "rank": 1,
                    "extraction_failed": False,
                    "holistic_fit_decision": "REJECT",
                    "holistic_fit_path": "none",
                },
            ],
        },
        {
            "final_action": "REJECT",
            "final_subreason": "fit_reject",
            "identity_classification": "ambiguity_multi_review",
            "opened_profile": True,
            "plausible_profile_reviews": [
                {
                    "rank": 1,
                    "extraction_failed": False,
                    "holistic_fit_decision": "REJECT",
                    "holistic_fit_path": "none",
                },
                {
                    "rank": 2,
                    "extraction_failed": False,
                    "holistic_fit_decision": "INFERENTIAL_SAVE",
                    "holistic_fit_path": "x",
                },
                {
                    "rank": 3,
                    "extraction_failed": False,
                    "holistic_fit_decision": "REJECT",
                    "holistic_fit_path": "none",
                },
            ],
        },
        {
            "final_action": "REJECT",
            "final_subreason": "fit_reject",
            "identity_classification": "single_strong_plausible_profile",
            "opened_profile": True,
            "plausible_profile_reviews": [
                {
                    "rank": 1,
                    "extraction_failed": False,
                    "holistic_fit_decision": "REJECT",
                    "holistic_fit_path": "none",
                },
            ],
        },
    ]

    summary = build_recruiter_identity_summary(rows)

    assert summary["identity_classification_counts"] == {
        "no_confident_match": 2,
        "single_strong_plausible_profile": 2,
        "ambiguity_multi_review": 1,
    }
    assert summary["opened_profile_count"] == 3
    # 0 + 0 + 1 + 3 + 1 = 5 -- matches the on-disk 5-lead recruiter dry-run.
    assert summary["full_judge_call_count"] == 5


def test_build_recruiter_identity_summary_full_judge_excludes_failed_extractions():
    rows = [
        {
            "final_action": "MANUAL_REVIEW",
            "final_subreason": "tool_failure",
            "identity_classification": "single_strong_plausible_profile",
            "opened_profile": True,
            "plausible_profile_reviews": [
                {"rank": 1, "extraction_failed": True},
            ],
        },
        {
            "final_action": "REJECT",
            "final_subreason": "fit_reject",
            "identity_classification": "ambiguity_multi_review",
            "opened_profile": True,
            "plausible_profile_reviews": [
                {
                    "rank": 1,
                    "extraction_failed": False,
                    "holistic_fit_decision": "REJECT",
                    "holistic_fit_path": "none",
                },
                {"rank": 2, "extraction_failed": True},
            ],
        },
    ]

    summary = build_recruiter_identity_summary(rows)

    # Extraction-failed reviews never reach full_judge, so they are not counted.
    assert summary["full_judge_call_count"] == 1


def test_build_recruiter_identity_summary_full_judge_zero_for_identity_collect_reviews():
    """Review-Fixes §2: identity_collect rows now produce opened-profile
    reviews (for identity confirmation) but never call full_judge. Reviews
    with empty holistic_fit fields must NOT inflate full_judge_call_count."""
    rows = [
        {
            "workflow_mode": "identity_collect",
            "final_action": "SAVE",
            "identity_classification": "high_confidence_match",
            "opened_profile": True,
            "plausible_profile_reviews": [
                {
                    "rank": 1,
                    "extraction_failed": False,
                    # No holistic_fit_decision or holistic_fit_path — identity_collect
                    # opens for identity confirmation only.
                    "identity_status": "confirmed",
                },
            ],
        },
        {
            "workflow_mode": "identity_collect",
            "identity_classification": "ambiguity_multi_review",
            "opened_profile": True,
            "plausible_profile_reviews": [
                {"rank": 1, "extraction_failed": False, "identity_status": "confirmed"},
                {"rank": 2, "extraction_failed": False, "identity_status": "no_match"},
            ],
        },
    ]

    summary = build_recruiter_identity_summary(rows)

    assert summary["full_judge_call_count"] == 0


def test_build_recruiter_identity_summary_full_judge_counts_only_evidence_bearing_reviews():
    """Mixed shape: one row has identity_collect opened-profile reviews (no
    holistic fields), another has fit_gated_save reviews with populated
    holistic fields. The summary must count only the evidence-bearing reviews."""
    rows = [
        {
            "workflow_mode": "identity_collect",
            "plausible_profile_reviews": [
                {"rank": 1, "extraction_failed": False, "identity_status": "confirmed"},
            ],
        },
        {
            "workflow_mode": "fit_gated_save",
            "plausible_profile_reviews": [
                {
                    "rank": 1,
                    "extraction_failed": False,
                    "holistic_fit_decision": "SAVE",
                    "holistic_fit_path": "DIRECT:1.A",
                },
                {
                    "rank": 2,
                    "extraction_failed": False,
                    "holistic_fit_decision": "REJECT",
                    "holistic_fit_path": "none",
                },
            ],
        },
    ]

    summary = build_recruiter_identity_summary(rows)

    assert summary["full_judge_call_count"] == 2


def test_build_recruiter_identity_summary_handles_rows_missing_classification_or_reviews():
    rows = [
        # legacy / partial row with no identity_classification and no reviews field
        {"final_action": "SAVE"},
        {"final_action": "REJECT", "final_subreason": "fit_reject"},
    ]

    summary = build_recruiter_identity_summary(rows)

    assert summary["identity_classification_counts"] == {}
    assert summary["full_judge_call_count"] == 0
    assert summary["total_leads"] == 2


def test_build_recruiter_identity_row_includes_plausible_profile_reviews():
    result = RecruiterIdentityResolution(
        github_username="ada",
        candidate_name="Ada Lovelace",
        lookup_name="Ada Lovelace",
    )
    result.ambiguity_multi_review = True
    result.plausible_profile_reviews = [
        PlausibleProfileReview(rank=1, profile_url="/p1", gate_final_action="REJECT"),
        PlausibleProfileReview(rank=2, profile_url="/p2", gate_final_action="SAVE"),
    ]
    row = build_recruiter_identity_row(result)
    assert row["plausible_profile_reviews_count"] == 2
    assert row["ambiguity_multi_review"] is True
    assert isinstance(row.get("plausible_profile_reviews"), list)
    assert len(row["plausible_profile_reviews"]) == 2


def test_write_recruiter_reconciliation_saved_jsonl_filters_rows(tmp_path):
    rows = [
        {"github_username": "a", "final_action": "SAVE"},
        {"github_username": "b", "final_action": "REJECT"},
    ]
    out = write_recruiter_reconciliation_saved_jsonl(tmp_path / "saved.jsonl", rows)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"github_username": "a"' in lines[0]


def test_saved_jsonl_excludes_manual_multi_review_rows(tmp_path):
    rows = [
        {"github_username": "win", "final_action": "SAVE"},
        {
            "github_username": "lose",
            "final_action": "MANUAL_REVIEW",
            "final_subreason": "identity_ambiguous",
            "ambiguity_multi_review": True,
            "plausible_profile_reviews": [{"rank": 1}, {"rank": 2}],
        },
    ]
    out = write_recruiter_reconciliation_saved_jsonl(tmp_path / "s.jsonl", rows)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "win" in lines[0]


def test_identity_collect_saved_export_uses_collection_action_not_legacy_save(tmp_path):
    """In identity_collect mode the saved/collected export must not gate on legacy
    final_action == "SAVE" (plan §2). Rows with collection_action == "COLLECT" AND
    a valid persisted save state are included even when final_action is unset or
    REJECT, and rows with final_action == "SAVE" but no COLLECT are excluded."""
    rows = [
        {
            "github_username": "collected",
            "workflow_mode": "identity_collect",
            "collection_action": "COLLECT",
            "project_save_state": "saved_now",
            "final_action": "",
        },
        {
            "github_username": "ambiguous",
            "workflow_mode": "identity_collect",
            "collection_action": "MANUAL_REVIEW",
            "final_action": "SAVE",
        },
        {
            "github_username": "rejected",
            "workflow_mode": "identity_collect",
            "collection_action": "REJECT",
            "final_action": "",
        },
    ]
    out = write_recruiter_reconciliation_saved_jsonl(tmp_path / "saved.jsonl", rows)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "collected" in lines[0]


def test_identity_collect_save_failed_row_is_excluded_from_saved_export(tmp_path):
    """Review-Fixes §1: an identity_collect row whose Recruiter save click did
    not persist must NOT appear in the saved/collected export, even though the
    canonical row still carries collection_action == "COLLECT"."""
    rows = [
        {
            "github_username": "click_failed",
            "workflow_mode": "identity_collect",
            "collection_action": "COLLECT",
            "project_save_state": "save_failed",
            "final_action": "SAVE",
        },
    ]
    out = write_recruiter_reconciliation_saved_jsonl(tmp_path / "saved.jsonl", rows)
    text = out.read_text(encoding="utf-8").strip()
    assert text == "", f"expected empty saved export, got: {text!r}"


def test_identity_collect_saved_export_includes_already_saved_and_dry_run_states(tmp_path):
    """Review-Fixes §1: saved_now, already_saved, and dry_run_skipped all count
    as "actually persisted enough to export"."""
    rows = [
        {
            "github_username": "fresh_save",
            "workflow_mode": "identity_collect",
            "collection_action": "COLLECT",
            "project_save_state": "saved_now",
        },
        {
            "github_username": "already_saved",
            "workflow_mode": "identity_collect",
            "collection_action": "COLLECT",
            "project_save_state": "already_saved",
        },
        {
            "github_username": "dry_run",
            "workflow_mode": "identity_collect",
            "collection_action": "COLLECT",
            "project_save_state": "dry_run_skipped",
        },
        {
            "github_username": "not_attempted",
            "workflow_mode": "identity_collect",
            "collection_action": "COLLECT",
            "project_save_state": "not_attempted",
        },
    ]
    out = write_recruiter_reconciliation_saved_jsonl(tmp_path / "saved.jsonl", rows)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    usernames = []
    for line in lines:
        if line.strip():
            usernames.append(json.loads(line)["github_username"])
    assert sorted(usernames) == ["already_saved", "dry_run", "fresh_save"]


def test_fit_gated_save_rows_keep_legacy_save_filter(tmp_path):
    """fit_gated_save mode (and unmoded legacy rows) keep the legacy
    final_action == "SAVE" filter for backward compatibility."""
    rows = [
        {"github_username": "legacy_save", "workflow_mode": "fit_gated_save", "final_action": "SAVE"},
        {"github_username": "legacy_reject", "workflow_mode": "fit_gated_save", "final_action": "REJECT"},
    ]
    out = write_recruiter_reconciliation_saved_jsonl(tmp_path / "saved.jsonl", rows)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "legacy_save" in lines[0]


def test_summary_counts_workflow_mode_identity_status_collection_action_save_state():
    rows = [
        {
            "workflow_mode": "identity_collect",
            "identity_status": "confirmed",
            "collection_action": "COLLECT",
            "project_save_state": "saved_now",
            "final_action": "",
        },
        {
            "workflow_mode": "identity_collect",
            "identity_status": "confirmed",
            "collection_action": "COLLECT",
            "project_save_state": "already_saved",
            "final_action": "",
        },
        {
            "workflow_mode": "identity_collect",
            "identity_status": "ambiguous",
            "collection_action": "MANUAL_REVIEW",
            "project_save_state": "not_attempted",
            "final_action": "",
        },
        {
            "workflow_mode": "fit_gated_save",
            "identity_status": "confirmed",
            "collection_action": "",
            "project_save_state": "",
            "final_action": "SAVE",
        },
    ]
    summary = build_recruiter_identity_summary(rows)

    assert summary["workflow_mode_counts"] == {"identity_collect": 3, "fit_gated_save": 1}
    assert summary["identity_status_counts"] == {"confirmed": 3, "ambiguous": 1}
    assert summary["collection_action_counts"] == {"COLLECT": 2, "MANUAL_REVIEW": 1}
    assert summary["project_save_state_counts"] == {
        "saved_now": 1,
        "already_saved": 1,
        "not_attempted": 1,
    }


def test_summary_action_counts_remap_identity_collect_save_failed_to_manual_review(tmp_path):
    """Summary-Alignment §1 + §3 regression case: an identity_collect row with
    collection_action == "COLLECT", project_save_state == "save_failed", AND a
    legacy compatibility final_action == "SAVE" must

      - be excluded from the saved/collected export, and
      - summarize as MANUAL_REVIEW / tool_failure (not as SAVE).
    """
    rows = [
        {
            "github_username": "click_failed",
            "workflow_mode": "identity_collect",
            "collection_action": "COLLECT",
            "collection_subreason": "exact_normalized_profile_name_match",
            "project_save_state": "save_failed",
            # Legacy compatibility — identity_collect still stamps SAVE on the
            # row when identity confirmed. The summary must NOT count this.
            "final_action": "SAVE",
            "final_subreason": "",
        },
    ]

    out = write_recruiter_reconciliation_saved_jsonl(tmp_path / "saved.jsonl", rows)
    assert out.read_text(encoding="utf-8").strip() == ""

    summary = build_recruiter_identity_summary(rows)

    assert summary["action_counts"].get("SAVE", 0) == 0
    assert summary["action_counts"].get("MANUAL_REVIEW", 0) == 1
    assert summary["subreason_counts"].get("tool_failure", 0) == 1
    # Canonical identity-collect counts remain untouched (the row still
    # represents a confirmed identity worth retaining).
    assert summary["collection_action_counts"].get("COLLECT", 0) == 1
    assert summary["project_save_state_counts"].get("save_failed", 0) == 1


def test_summary_action_counts_keep_identity_collect_persisted_states_as_save():
    """Summary-Alignment §1: saved_now, already_saved, and dry_run_skipped all
    count as SAVE in the summary action histogram (matching the saved export)."""
    rows = [
        {
            "workflow_mode": "identity_collect",
            "collection_action": "COLLECT",
            "project_save_state": "saved_now",
            "final_action": "SAVE",
        },
        {
            "workflow_mode": "identity_collect",
            "collection_action": "COLLECT",
            "project_save_state": "already_saved",
            "final_action": "SAVE",
        },
        {
            "workflow_mode": "identity_collect",
            "collection_action": "COLLECT",
            "project_save_state": "dry_run_skipped",
            "final_action": "SAVE",
        },
    ]

    summary = build_recruiter_identity_summary(rows)

    assert summary["action_counts"].get("SAVE", 0) == 3
    assert summary["action_counts"].get("MANUAL_REVIEW", 0) == 0
    assert summary["action_counts"].get("REJECT", 0) == 0


def test_summary_action_counts_route_identity_collect_manual_review_and_reject():
    """Summary-Alignment §1: MANUAL_REVIEW and REJECT collection actions
    summarize as their obvious action equivalents, with subreasons sourced
    from collection_subreason (falls back to legacy final_subreason)."""
    rows = [
        {
            "workflow_mode": "identity_collect",
            "collection_action": "MANUAL_REVIEW",
            "collection_subreason": "multiple_confirmed_identities",
            "project_save_state": "not_attempted",
            "final_action": "MANUAL_REVIEW",
            "final_subreason": "multiple_confirmed_identities",
        },
        {
            "workflow_mode": "identity_collect",
            "collection_action": "REJECT",
            "collection_subreason": "no_recruiter_results",
            "project_save_state": "not_attempted",
            "final_action": "REJECT",
            "final_subreason": "no_plausible_profile",
        },
    ]

    summary = build_recruiter_identity_summary(rows)

    assert summary["action_counts"].get("MANUAL_REVIEW", 0) == 1
    assert summary["action_counts"].get("REJECT", 0) == 1
    assert summary["subreason_counts"].get("multiple_confirmed_identities", 0) == 1
    # collection_subreason wins over legacy final_subreason for the routed bucket.
    assert summary["subreason_counts"].get("no_recruiter_results", 0) == 1


def test_summary_action_counts_keep_legacy_fit_gated_save_behavior_unchanged():
    """Summary-Alignment §1: fit_gated_save and unmoded legacy rows continue to
    derive action_counts / subreason_counts from final_action / final_subreason."""
    rows = [
        {
            "workflow_mode": "fit_gated_save",
            "final_action": "SAVE",
            "final_subreason": "",
        },
        {
            "workflow_mode": "fit_gated_save",
            "final_action": "REJECT",
            "final_subreason": "fit_reject",
        },
        {
            "workflow_mode": "fit_gated_save",
            "final_action": "MANUAL_REVIEW",
            "final_subreason": "identity_ambiguous",
        },
        # Unmoded legacy row — also keeps prior behavior.
        {
            "final_action": "SAVE",
            "final_subreason": "",
        },
    ]

    summary = build_recruiter_identity_summary(rows)

    assert summary["action_counts"]["SAVE"] == 2
    assert summary["action_counts"]["REJECT"] == 1
    assert summary["action_counts"]["MANUAL_REVIEW"] == 1
    assert summary["subreason_counts"]["fit_reject"] == 1
    assert summary["subreason_counts"]["identity_ambiguous"] == 1


def test_summary_action_counts_for_identity_collect_collect_without_save_state_falls_back_to_legacy():
    """Defensive: a malformed/legacy identity_collect row with collection_action
    == "COLLECT" but no recognized project_save_state should fall back to the
    legacy final_action rather than synthesizing a save bucket."""
    rows = [
        {
            "workflow_mode": "identity_collect",
            "collection_action": "COLLECT",
            "project_save_state": "",
            "final_action": "MANUAL_REVIEW",
            "final_subreason": "tool_failure",
        },
    ]

    summary = build_recruiter_identity_summary(rows)

    assert summary["action_counts"].get("SAVE", 0) == 0
    assert summary["action_counts"].get("MANUAL_REVIEW", 0) == 1
    assert summary["subreason_counts"].get("tool_failure", 0) == 1


def test_resolution_round_trips_new_identity_collection_fields():
    """RecruiterIdentityResolution.from_dict must hydrate all new identity-collection
    fields so resume reads do not silently drop them."""
    payload = {
        "github_username": "ada",
        "candidate_name": "Ada Lovelace",
        "lookup_name": "Ada Lovelace",
        "workflow_mode": "identity_collect",
        "identity_status": "confirmed",
        "identity_subreason": "exact_normalized_name_match",
        "collection_action": "COLLECT",
        "collection_subreason": "",
        "project_save_state": "saved_now",
        "queries_tried": ['"Ada Lovelace" AND ("Chase")', '"Ada Lovelace"'],
        "resolved_query": '"Ada Lovelace"',
    }
    obj = RecruiterIdentityResolution.from_dict(payload)
    assert obj.workflow_mode == "identity_collect"
    assert obj.identity_status == "confirmed"
    assert obj.identity_subreason == "exact_normalized_name_match"
    assert obj.collection_action == "COLLECT"
    assert obj.project_save_state == "saved_now"
    assert obj.queries_tried == ['"Ada Lovelace" AND ("Chase")', '"Ada Lovelace"']
    assert obj.resolved_query == '"Ada Lovelace"'

    round_tripped = obj.to_dict()
    for key in (
        "workflow_mode",
        "identity_status",
        "identity_subreason",
        "collection_action",
        "collection_subreason",
        "project_save_state",
        "queries_tried",
        "resolved_query",
    ):
        assert key in round_tripped
