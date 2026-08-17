from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools import run_recruiter_identity_resolver as cli


def test_select_leads_window_respects_offset_and_max():
    leads = list(range(10))

    assert cli._select_leads_window(leads, lead_offset=0, max_leads=3) == [0, 1, 2]
    assert cli._select_leads_window(leads, lead_offset=4, max_leads=3) == [4, 5, 6]
    assert cli._select_leads_window(leads, lead_offset=9, max_leads=3) == [9]
    assert cli._select_leads_window(leads, lead_offset=-5, max_leads=2) == [0, 1]
    assert cli._select_leads_window(leads, lead_offset=3, max_leads=0) == []


def test_seed_or_validate_run_metadata_rejects_mismatched_existing_run(tmp_path: Path):
    path = tmp_path / "recruiter_identity_run_metadata.json"
    metadata = {"github_output_dir": "/tmp/run", "lead_offset": 0, "requested_max_leads": 20}
    cli._seed_or_validate_run_metadata(path, metadata)

    cli._seed_or_validate_run_metadata(path, metadata)

    with pytest.raises(SystemExit, match="different cohort/window"):
        cli._seed_or_validate_run_metadata(
            path,
            {"github_output_dir": "/tmp/run", "lead_offset": 20, "requested_max_leads": 20},
        )


def test_seed_or_validate_run_metadata_reports_per_field_diffs(tmp_path: Path):
    path = tmp_path / "recruiter_identity_run_metadata.json"
    cli._seed_or_validate_run_metadata(
        path,
        {"workflow_mode": "identity_collect", "max_cards": 5},
    )

    with pytest.raises(SystemExit) as excinfo:
        cli._seed_or_validate_run_metadata(
            path,
            {"workflow_mode": "fit_gated_save", "max_cards": 5},
        )
    message = str(excinfo.value)
    assert "workflow_mode" in message
    assert "'identity_collect'" in message
    assert "'fit_gated_save'" in message


def test_seed_or_validate_run_metadata_resume_with_recruiter_url_after_attach(
    tmp_path: Path,
):
    """Followups plan §1: the first metadata write now includes
    recruiter_url_after_attach, so a resume that rebuilds the SAME final-shape
    dict (URL included) must validate cleanly."""
    path = tmp_path / "recruiter_identity_run_metadata.json"
    final_metadata = {
        "workflow_mode": "identity_collect",
        "github_output_dir": "/tmp/run",
        "project_url": "https://www.linkedin.com/talent/hire/123/search",
        "lead_offset": 0,
        "requested_max_leads": 25,
        "target_lead_count": 25,
        "location_filter": "New York",
        "use_current_search": False,
        "max_cards": 5,
        "skip_profile_open": False,
        "dry_run_save": False,
        "recruiter_url_after_attach": "https://www.linkedin.com/talent/hire/123/search",
    }

    cli._seed_or_validate_run_metadata(path, final_metadata)
    cli._seed_or_validate_run_metadata(path, dict(final_metadata))


def test_seed_or_validate_run_metadata_rejects_resume_when_recruiter_url_missing(
    tmp_path: Path,
):
    """Regression guard for the original v1 bug: a resume that supplied metadata
    WITHOUT recruiter_url_after_attach against a file that contains it must
    surface as a clear per-field diff (not be silently accepted)."""
    path = tmp_path / "recruiter_identity_run_metadata.json"
    final_metadata = {
        "workflow_mode": "identity_collect",
        "recruiter_url_after_attach": "https://www.linkedin.com/talent/hire/123/search",
    }
    cli._seed_or_validate_run_metadata(path, final_metadata)

    with pytest.raises(SystemExit) as excinfo:
        cli._seed_or_validate_run_metadata(
            path,
            {"workflow_mode": "identity_collect"},
        )
    assert "recruiter_url_after_attach" in str(excinfo.value)


def test_default_workflow_mode_is_identity_collect():
    parser = cli._build_parser()
    args = parser.parse_args(["--github-output-dir", "/tmp/x", "--use-current-search"])

    assert args.workflow_mode == cli.WORKFLOW_MODE_IDENTITY_COLLECT


def test_workflow_mode_choice_is_validated():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--github-output-dir",
                "/tmp/x",
                "--use-current-search",
                "--workflow-mode",
                "fit-gated-save",
            ]
        )


def test_default_query_expansion_policy_is_auto():
    """Cycle-Audit-Fixes §1: --query-expansion-policy defaults to auto."""
    parser = cli._build_parser()
    args = parser.parse_args(["--github-output-dir", "/tmp/x", "--use-current-search"])
    assert args.query_expansion_policy == cli.DEFAULT_QUERY_EXPANSION_POLICY == "auto"


def test_query_expansion_policy_choice_is_validated():
    """Unknown values are rejected by argparse choices."""
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--github-output-dir",
                "/tmp/x",
                "--use-current-search",
                "--query-expansion-policy",
                "aggressive",
            ]
        )


def test_query_expansion_policy_accepts_explicit_overrides():
    parser = cli._build_parser()
    for value in ("auto", "name_first", "enriched"):
        args = parser.parse_args(
            [
                "--github-output-dir",
                "/tmp/x",
                "--use-current-search",
                "--query-expansion-policy",
                value,
            ]
        )
        assert args.query_expansion_policy == value


def _make_dummy_lead(username: str = "ada", candidate_name: str = "Ada Lovelace"):
    """A lightweight stand-in for GitHubReconciliationLead, sufficient for the
    failure-row builder which only reads attributes via getattr."""
    lead = MagicMock()
    lead.username = username
    lead.candidate_name = candidate_name
    lead.github_url = f"https://github.com/{username}"
    lead.company = ""
    lead.location = ""
    lead.title = ""
    return lead


def test_per_lead_timeout_recovery_first_attempt_times_out_then_retry_succeeds():
    """Plan §7: first attempt times out, browser.check_and_recover() runs once,
    second attempt returns the resolution. No failure row should be raised."""
    expected_resolution = MagicMock(name="resolution")

    call_count = {"n": 0}

    async def _resolve(_lead):
        call_count["n"] += 1
        if call_count["n"] == 1:
            await asyncio.sleep(1.0)
        return expected_resolution

    resolver = MagicMock()
    resolver.resolve_lead = _resolve
    browser = AsyncMock()
    browser.check_and_recover.return_value = True

    result = asyncio.run(
        cli._resolve_lead_with_timeout_and_recovery(
            resolver=resolver,
            browser=browser,
            lead=_make_dummy_lead(),
            timeout_seconds=0.05,
        )
    )

    assert result is expected_resolution
    assert call_count["n"] == 2
    browser.check_and_recover.assert_awaited_once()


def test_per_lead_timeout_recovery_raises_resolver_lead_failure_when_retry_also_times_out():
    """Plan §7: when the retry also times out, ResolverLeadFailure is raised so
    the runner can record a failure row and continue."""

    async def _resolve_always_hangs(_lead):
        await asyncio.sleep(2.0)

    resolver = MagicMock()
    resolver.resolve_lead = _resolve_always_hangs
    browser = AsyncMock()
    browser.check_and_recover.return_value = True

    with pytest.raises(cli.ResolverLeadFailure, match="retry also timed out"):
        asyncio.run(
            cli._resolve_lead_with_timeout_and_recovery(
                resolver=resolver,
                browser=browser,
                lead=_make_dummy_lead(),
                timeout_seconds=0.05,
            )
        )

    browser.check_and_recover.assert_awaited_once()


def test_per_lead_timeout_recovery_raises_resolver_lead_failure_when_retry_throws():
    call_count = {"n": 0}

    async def _resolve(_lead):
        call_count["n"] += 1
        raise RuntimeError(f"browser stuck attempt {call_count['n']}")

    resolver = MagicMock()
    resolver.resolve_lead = _resolve
    browser = AsyncMock()
    browser.check_and_recover.return_value = True

    with pytest.raises(cli.ResolverLeadFailure, match="retry raised"):
        asyncio.run(
            cli._resolve_lead_with_timeout_and_recovery(
                resolver=resolver,
                browser=browser,
                lead=_make_dummy_lead(),
                timeout_seconds=5.0,
            )
        )

    assert call_count["n"] == 2


def test_build_failure_row_carries_tool_failure_provenance():
    """Plan §7: the synthesized failure row must mark identity_status,
    collection_action, and project_save_state so summaries / resume see it."""
    lead = _make_dummy_lead("erosika", "Eri Barrett")
    row = cli._build_failure_row(
        lead=lead,
        workflow_mode=cli.WORKFLOW_MODE_IDENTITY_COLLECT,
        linkedin_brief_path=None,
        note="retry also timed out after 240s",
    )
    assert row["github_username"] == "erosika"
    assert row["candidate_name"] == "Eri Barrett"
    assert row["workflow_mode"] == cli.WORKFLOW_MODE_IDENTITY_COLLECT
    assert row["identity_status"] == "tool_failure"
    assert row["collection_action"] == "MANUAL_REVIEW"
    assert row["project_save_state"] == "not_attempted"
    assert row["final_action"] == "MANUAL_REVIEW"
    assert row["final_subreason"] == "tool_failure"
    assert any("timed out" in note for note in row.get("notes", []))


def test_per_lead_timeout_default_is_set_on_argparser():
    parser = cli._build_parser()
    args = parser.parse_args(["--github-output-dir", "/tmp/x", "--use-current-search"])
    assert args.per_lead_timeout_seconds > 0


def test_load_existing_progress_rows_dedupes_github_username(tmp_path: Path):
    path = tmp_path / "recruiter_identity_resolutions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"github_username": "ada", "final_action": "SAVE"}),
                json.dumps({"github_username": "ada", "final_action": "REJECT"}),
                json.dumps({"github_username": "grace", "final_action": "REJECT"}),
                json.dumps({"github_username": "", "final_action": "REJECT"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = cli._load_existing_progress_rows(path)

    assert [row["github_username"] for row in rows] == ["ada", "grace"]
    assert rows[0]["final_action"] == "SAVE"


def test_append_progress_row_writes_incremental_artifacts(tmp_path: Path):
    rows: list[dict] = []
    input_stats = {
        "total_saved_judgments": 179,
        "lead_offset": 20,
        "requested_max_leads": 10,
        "target_lead_count": 10,
    }
    reject_row = {
        "github_username": "ada",
        "candidate_name": "Ada Lovelace",
        "final_action": "REJECT",
        "final_subreason": "fit_reject",
        "identity_classification": "single_strong_plausible_profile",
        "opened_profile": True,
        "plausible_profile_reviews": [{"rank": 1, "extraction_failed": False}],
        "top_candidates": [{"already_saved": False}],
    }
    save_row = {
        "github_username": "grace",
        "candidate_name": "Grace Hopper",
        "final_action": "SAVE",
        "final_subreason": "",
        "identity_classification": "high_confidence_match",
        "opened_profile": True,
        "plausible_profile_reviews": [{"rank": 1, "extraction_failed": False}],
        "top_candidates": [{"already_saved": False}],
    }

    cli._append_progress_row(tmp_path, rows, reject_row, input_stats=input_stats)
    cli._append_progress_row(tmp_path, rows, save_row, input_stats=input_stats)

    jsonl_path = tmp_path / "recruiter_identity_resolutions.jsonl"
    saved_jsonl_path = tmp_path / "recruiter_reconciliation_saved.jsonl"
    summary_path = tmp_path / "recruiter_identity_resolutions_summary.json"
    csv_path = tmp_path / "recruiter_identity_resolutions.csv"
    saved_csv_path = tmp_path / "recruiter_reconciliation_saved.csv"

    assert jsonl_path.is_file()
    assert csv_path.is_file()
    assert saved_csv_path.is_file()

    jsonl_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["github_username"] for row in jsonl_rows] == ["ada", "grace"]

    saved_rows = [
        json.loads(line)
        for line in saved_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["github_username"] for row in saved_rows] == ["grace"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_leads"] == 2
    assert summary["action_counts"] == {"REJECT": 1, "SAVE": 1}
    assert summary["input_stats"]["processed_leads"] == 2
    assert summary["input_stats"]["lead_offset"] == 20
