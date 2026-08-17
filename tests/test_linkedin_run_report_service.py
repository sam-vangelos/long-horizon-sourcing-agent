"""Unit tests for the extracted LinkedIn RunReportService cluster."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

from shared.schemas import CandidateSnippet, OpusDecision, Progress, SearchString

from linkedin.run_report import RunReportDeps, RunReportService, _PageReport
from linkedin.search_intelligence import bootstrap_experiment_state


def _make_snippet(**kwargs) -> CandidateSnippet:
    defaults = {
        "name": "Test Person",
        "headline": "",
        "current_title": "",
        "current_company": "",
        "location": "Somewhere",
        "education_snippet": "",
        "profile_url": "/talent/profile/test123",
        "source_string_id": 1,
        "source_string_name": "test",
        "page": 1,
        "result_rank": 1,
    }
    defaults.update(kwargs)
    return CandidateSnippet(**defaults)


def _make_service(tmp_path: Path) -> RunReportService:
    output_dir = tmp_path
    brief_path = output_dir / "brief.json"
    brief_path.write_text('{"id": "test", "version": "1"}')

    brief = MagicMock()
    brief.id = "test"
    brief.role_title = "Engineer"
    brief.linkedin_project = "Test Project"
    brief.linkedin_project_id = "test-project"
    brief.has_v2_schema = False
    brief.raw = {"version": "1"}

    stats = {
        "snippets_extracted": 10,
        "facial_yes": 4,
        "facial_borderline": 1,
        "facial_no": 5,
        "saved": 2,
        "rejected": 1,
        "save_attempts": 2,
        "save_failed": 0,
        "high_pressure_candidates_seen": 0,
        "activity_saturated_preview_skips": 0,
        "high_fit_low_novelty_saves": 0,
    }

    deps = RunReportDeps(
        get_brief_obj=lambda: brief,
        brief_path=str(brief_path),
        output_dir=output_dir,
        final_path=output_dir / "final_judgments.jsonl",
        log_path=output_dir / "run_log.jsonl",
        profiles_path=output_dir / "profile_summaries.jsonl",
        get_runtime_db_path=lambda: output_dir / "runtime_state.sqlite3",
        stats=stats,
        get_search_memory=lambda: {},
        get_constraint_manifest=lambda: {},
        get_experiment_states=lambda: {},
        get_runtime_bridge=lambda: MagicMock(),
        get_runtime_run_id=lambda: None,
        get_session_geography_receipt=lambda: {},
        get_bias_monitor=lambda: None,
        get_lint_blocked_strings=lambda: [],
        _adaptation_roi_summary=lambda progress: {"status": "no_adaptation_events"},
        _shadow_cache_hit_rate=lambda shadow_stage=None: None,
        _string_has_seniority_contamination=lambda search_string, profile_index=None: False,
    )
    return RunReportService(deps)


def test_service_reads_dependencies_live_not_snapshotted():
    """Accessors must read live pipeline state, not values snapshotted at construction.

    The staleness mode this locks is REBINDING, not in-place mutation. Pipeline
    reassigns ``self._session_geography_receipt`` to a NEW object mid-run (which
    is why the old design needed a manual ``_sync_run_report_deps``); a snapshot
    field would keep pointing at the object captured at construction and never
    see the replacement.

    Note the earlier draft of this test mutated one dict in place, which proves
    nothing: a snapshotted VALUE field holds that same dict object, so an
    in-place mutation is visible under both designs. The holder below is
    REBOUND, so this test genuinely fails against snapshot semantics.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        service = _make_service(tmp_path)
        # One level of indirection so the accessor can observe a REBIND.
        holder: dict = {"receipt": {}}

        deps = replace(
            service.deps,
            get_session_geography_receipt=lambda: holder["receipt"],
        )
        service = RunReportService(deps)
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1,
                    name="test",
                    boolean="foo",
                    status="done",
                    pages_reviewed=1,
                    candidates_count=10,
                )
            ],
        )

        snapshot_before = service._build_run_report_snapshot(progress)
        assert "session_geography" not in snapshot_before["run_metadata"]

        # REBIND to a brand-new object — the mode a snapshot field cannot see.
        holder["receipt"] = {
            "intended_facets": ["New York"],
            "applied_facets": ["New York"],
        }

        snapshot_after = service._build_run_report_snapshot(progress)
        assert snapshot_after["run_metadata"]["session_geography"] == {
            "intended_facets": ["New York"],
            "applied_facets": ["New York"],
        }


def test_build_run_report_snapshot_top_level_keys():
    with tempfile.TemporaryDirectory() as td:
        service = _make_service(Path(td))
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1,
                    name="test",
                    boolean="foo",
                    status="done",
                    pages_reviewed=1,
                    candidates_count=10,
                )
            ],
        )

        snapshot = service._build_run_report_snapshot(progress)

        assert set(snapshot.keys()) == {
            "schema_version",
            "run_metadata",
            "metrics_summary",
            "string_performance",
            "lane_execution_summary",
            "saved_candidate_summaries",
            "rejected_candidate_summaries",
            "bias_monitor_summary",
            "search_memory_summary",
        }


def test_cost_summary_for_report_no_cost_data_when_log_absent():
    with tempfile.TemporaryDirectory() as td:
        service = _make_service(Path(td))

        summary = service._cost_summary_for_report()

        assert summary == {"status": "no_cost_data"}


def test_page_report_accumulates_saved_skipped_and_save_failed():
    report = _PageReport(string_id=1, string_name="lane-a", page=1, result_count=25)

    snippet = _make_snippet(name="Alice", current_title="Engineer", current_company="Acme")
    decision = OpusDecision(
        stage="full",
        decision="SAVE",
        path="full",
        confidence=0.9,
        rationale="Strong fit",
        candidate_name="Alice",
        profile_url="/talent/profile/alice",
    )
    report.add_saved(snippet, decision)
    report.add_skipped_opened(
        snippet,
        OpusDecision(
            stage="full",
            decision="REJECT",
            path="full",
            confidence=0.1,
            rationale="No",
            candidate_name="Alice",
            profile_url="/talent/profile/alice",
        ),
    )
    report.add_skip_preview("Bob", "preview:low_signal")
    report.add_save_failed(snippet, decision, "timeout")

    assert len(report.saved) == 1
    assert len(report.skipped_opened) == 1
    assert len(report.skipped_preview) == 1
    assert len(report.save_failed) == 1


def test_service_reads_experiment_states_live_not_snapshotted():
    """get_experiment_states must read live pipeline state, not a construction snapshot.

    Wave 6 boundary review P0: Pipeline REBINDS ``self._experiment_states`` to a
    brand-new dict on the run-start path of fresh and resumed runs alike
    (``work_units.py`` calls ``set_experiment_states``), so a value field captured
    at ``__init__`` feeds the report and mid-run block adaptation an empty dict
    forever. The holder below is REBOUND, so this test fails against snapshot
    semantics.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        service = _make_service(tmp_path)
        holder: dict = {"states": {}}
        service = RunReportService(
            replace(service.deps, get_experiment_states=lambda: holder["states"])
        )
        search_string = SearchString(id=1, name="s", boolean="a", status="done")
        assert service._search_intelligence_detail_for_string(search_string) == {}

        # REBIND to a brand-new dict — what the run-start path actually does.
        holder["states"] = {1: bootstrap_experiment_state(search_string)}
        detail = service._search_intelligence_detail_for_string(search_string)
        assert detail != {}
        assert "mode" in detail


def test_service_reads_brief_live_not_snapshotted():
    """get_brief_obj must read live pipeline state, not a construction snapshot.

    Wave 6 boundary review P1: Pipeline rebinds ``self.brief_obj`` at the end of
    preflight (``execution.brief``) and on resume (``_load_v2_brief``); a value
    field would stamp every report with the hollow seed brief's metadata. The
    holder below is REBOUND, so this test fails against snapshot semantics.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        service = _make_service(tmp_path)
        holder: dict = {"brief": service.deps.get_brief_obj()}
        service = RunReportService(
            replace(service.deps, get_brief_obj=lambda: holder["brief"])
        )
        progress = Progress(brief_name="test", strings=[])
        before = service._build_run_report_snapshot(progress)
        assert before["run_metadata"]["role_title"] == "Engineer"

        rebound = MagicMock()
        rebound.id = "test"
        rebound.role_title = "Rebound Director"
        rebound.linkedin_project = "Test Project"
        rebound.linkedin_project_id = "test-project"
        rebound.has_v2_schema = False
        rebound.raw = {"version": "2"}
        holder["brief"] = rebound

        snapshot = service._build_run_report_snapshot(progress)
        assert snapshot["run_metadata"]["role_title"] == "Rebound Director"
        assert snapshot["run_metadata"]["brief_version"] == "2"
