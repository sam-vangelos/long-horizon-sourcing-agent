import json

from github.identity_resolution_experiment_report import (
    build_identity_resolution_preview_summary,
    write_identity_resolution_experiment_jsonl,
    write_identity_resolution_preview_markdown,
    write_identity_resolution_preview_summary,
    write_identity_resolution_experiment_markdown,
    write_identity_resolution_experiment_summary,
)
from shared.identity_experiment_schemas import StrategyExecutionRecord


def test_identity_resolution_experiment_report_writers_emit_summary_files(tmp_path):
    rows = [
        StrategyExecutionRecord(
            strategy_name="web_exact_city",
            github_username="ada",
            candidate_name="Ada Lovelace",
            cohort_kind="primary",
            cohort_bucket="easy_exact_name",
            top1_correct=True,
            top3_contains_correct=True,
            duration_seconds=12.0,
            interaction_count=2,
        )
    ]

    jsonl_path = write_identity_resolution_experiment_jsonl(tmp_path / "results.jsonl", rows)
    summary_path = write_identity_resolution_experiment_summary(tmp_path / "summary.json", rows)
    markdown_path = write_identity_resolution_experiment_markdown(tmp_path / "summary.md", rows)

    assert jsonl_path.exists()
    assert summary_path.exists()
    assert markdown_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["decision"]["winner"] == "web_exact_city"
    assert "Identity Resolution Retrieval Experiment" in markdown_path.read_text()


def test_identity_resolution_preview_report_writers_emit_preview_files(tmp_path):
    rows = [
        StrategyExecutionRecord(
            strategy_name="web_exact_city",
            github_username="ada",
            candidate_name="Ada Lovelace",
            cohort_kind="primary",
            cohort_bucket="easy_exact_name",
            duration_seconds=12.0,
            interaction_count=2,
        )
    ]
    rows[0].top1_profile_url = "https://www.linkedin.com/in/ada-lovelace/"

    summary = build_identity_resolution_preview_summary(rows)
    assert summary["mode"] == "preview_only"
    assert summary["strategies"]["web_exact_city"]["attempted_leads"] == 1
    assert summary["strategies"]["web_exact_city"]["aborted_leads"] == 0

    summary_path = write_identity_resolution_preview_summary(tmp_path / "preview.json", rows)
    markdown_path = write_identity_resolution_preview_markdown(tmp_path / "preview.md", rows)

    assert summary_path.exists()
    assert markdown_path.exists()
    assert "qualitative only" in markdown_path.read_text().lower()


def test_identity_resolution_preview_summary_excludes_aborted_rows_from_attempts():
    rows = [
        StrategyExecutionRecord(
            strategy_name="web_exact_city",
            github_username="ada",
            candidate_name="Ada Lovelace",
            cohort_kind="primary",
            cohort_bucket="easy_exact_name",
            duration_seconds=12.0,
            interaction_count=2,
        ),
        StrategyExecutionRecord(
            strategy_name="web_exact_city",
            github_username="grace",
            candidate_name="Grace Hopper",
            cohort_kind="primary",
            cohort_bucket="easy_exact_name",
            aborted=True,
            abort_reason="two consecutive login_wall blockers",
            blocker_state="aborted",
        ),
    ]

    summary = build_identity_resolution_preview_summary(rows)

    assert summary["strategies"]["web_exact_city"]["attempted_leads"] == 1
    assert summary["strategies"]["web_exact_city"]["aborted_leads"] == 1


def test_identity_resolution_strategy_csv_includes_debug_fields(tmp_path):
    from github.identity_resolution_experiment_report import (
        write_identity_resolution_experiment_strategy_csvs,
    )

    row = StrategyExecutionRecord(
        strategy_name="web_exact_city",
        github_username="ada",
        candidate_name="Ada Lovelace",
        cohort_kind="primary",
        cohort_bucket="easy_exact_name",
        final_url="https://www.bing.com/search?q=ada",
        page_title="Ada - Bing",
        body_excerpt="body excerpt",
        notes=["note one", "note two"],
    )

    paths = write_identity_resolution_experiment_strategy_csvs(tmp_path, [row])
    content = paths[0].read_text()

    assert "body_excerpt" in content.splitlines()[0]
    assert "body excerpt" in content
    assert "note one | note two" in content
