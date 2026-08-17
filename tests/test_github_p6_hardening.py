"""Tests for the P6 OSS Maintainers activation hardening pass.

Covers (plans/hardening-and-consolidation-spec.md §7):
  P6.1 — classify() integration seam
  P6.2 — graph-expansion self-cancellation fix
  P6.3 — adaptation inputs made honest (_build_batch_report) + prior_run_data
  P6.4 — bias monitor lifecycle (load/check_alerts/save/bias_summary)
  P6.5 — geography veto fail-open for unconfigured geographies

Run with: python -m pytest tests/test_github_p6_hardening.py -v
"""

from __future__ import annotations

import asyncio
import csv
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from github.maintainership import MaintainershipClassification
from github.schemas import GitHubProgress, GitHubRepo
from shared.bias_controls import BiasMonitor, DecisionRecord
from shared.storage import append_jsonl

from tests.test_github_pipeline import (
    GitHubPipeline,
    _FakeGitHubClient,
    _attach_runtime_state,
    _make_candidate,
    _make_pipeline,
    _make_query,
    _run_github_pipeline_to_completion,
    github_orchestrator,
)


# ---------------------------------------------------------------------------
# P6.1 — THE integration seam
# ---------------------------------------------------------------------------


class TestClassifyIntegrationSeam:
    def test_target_projects_brief_populates_maintainership_and_evidence(self):
        pipeline = _make_pipeline()
        pipeline._runtime_run_id = None
        pipeline._ensure_services()
        pipeline.brief_obj.target_projects = ["kubernetes/kubernetes"]
        pipeline._client = MagicMock()
        pipeline._execution_engine = MagicMock()
        pipeline._side_effects_service = MagicMock()
        pipeline._side_effects_service.handle_full_decision = AsyncMock(return_value=None)

        candidate = _make_candidate("octocat", "The Octocat")
        query = _make_query(id=5, channel="user_search")
        progress = GitHubProgress(brief_name="test")

        facial_decision = MagicMock(decision="FACIAL_YES", confidence=0.9, rationale="")
        full_decision = MagicMock(decision="SAVE", confidence=0.9, path="direct")
        classification = MaintainershipClassification(
            level="maintainer",
            confidence=0.7,
            evidence_sources=["merge_authority:kubernetes/kubernetes:12PRs"],
            signals={},
        )

        captured: dict = {}

        def _fake_full_judge(evidence_text, brief=None):
            captured["text"] = evidence_text
            return full_decision

        classify_mock = AsyncMock(return_value=classification)
        with patch.object(github_orchestrator, "github_facial_judge_batch", return_value=[facial_decision]), \
             patch.object(github_orchestrator, "github_full_judge", side_effect=_fake_full_judge), \
             patch.object(github_orchestrator, "classify_maintainership", classify_mock):
            asyncio.run(
                pipeline._process_v2_candidates_batch([("octocat", candidate)], query, progress)
            )

        classify_mock.assert_awaited_once_with(
            "octocat", ["kubernetes/kubernetes"], pipeline._client
        )
        assert candidate.maintainership == classification.to_dict()
        assert "MAINTAINERSHIP EVIDENCE" in captured["text"]
        assert "merge_authority:kubernetes/kubernetes:12PRs" in captured["text"]

    def test_no_target_projects_is_byte_identical(self):
        """Behavior-preserving contract: classic github briefs never call
        classify() and candidate.maintainership stays None, so
        to_evidence_text() renders without the MAINTAINERSHIP EVIDENCE block."""
        pipeline = _make_pipeline()
        pipeline._runtime_run_id = None
        pipeline._ensure_services()
        pipeline.brief_obj.target_projects = []
        pipeline._client = MagicMock()
        pipeline._execution_engine = MagicMock()
        pipeline._side_effects_service = MagicMock()
        pipeline._side_effects_service.handle_full_decision = AsyncMock(return_value=None)

        candidate = _make_candidate("torvalds", "Linus Torvalds")
        query = _make_query(id=6, channel="user_search")
        progress = GitHubProgress(brief_name="test")

        facial_decision = MagicMock(decision="FACIAL_YES", confidence=0.9, rationale="")
        full_decision = MagicMock(decision="REJECT", confidence=0.2, path="none")

        captured: dict = {}

        def _fake_full_judge(evidence_text, brief=None):
            captured["text"] = evidence_text
            return full_decision

        classify_mock = AsyncMock()
        with patch.object(github_orchestrator, "github_facial_judge_batch", return_value=[facial_decision]), \
             patch.object(github_orchestrator, "github_full_judge", side_effect=_fake_full_judge), \
             patch.object(github_orchestrator, "classify_maintainership", classify_mock):
            asyncio.run(
                pipeline._process_v2_candidates_batch([("torvalds", candidate)], query, progress)
            )

        classify_mock.assert_not_awaited()
        assert candidate.maintainership is None
        assert "MAINTAINERSHIP EVIDENCE" not in captured["text"]

    def test_classify_failure_is_fail_soft_not_fatal(self):
        """A classify() exception must not take down candidate processing —
        the full judge still runs on evidence without the maintainership
        section."""
        pipeline = _make_pipeline()
        pipeline._runtime_run_id = None
        pipeline._ensure_services()
        pipeline.brief_obj.target_projects = ["kubernetes/kubernetes"]
        pipeline._client = MagicMock()
        pipeline._execution_engine = MagicMock()
        pipeline._side_effects_service = MagicMock()
        pipeline._side_effects_service.handle_full_decision = AsyncMock(return_value=None)
        pipeline._observer.on_error = MagicMock()

        candidate = _make_candidate("someone", "Some One")
        query = _make_query(id=7, channel="user_search")
        progress = GitHubProgress(brief_name="test")

        facial_decision = MagicMock(decision="FACIAL_YES", confidence=0.9, rationale="")
        full_decision = MagicMock(decision="REJECT", confidence=0.2, path="none")
        classify_mock = AsyncMock(side_effect=RuntimeError("rate limited"))

        with patch.object(github_orchestrator, "github_facial_judge_batch", return_value=[facial_decision]), \
             patch.object(github_orchestrator, "github_full_judge", return_value=full_decision), \
             patch.object(github_orchestrator, "classify_maintainership", classify_mock):
            asyncio.run(
                pipeline._process_v2_candidates_batch([("someone", candidate)], query, progress)
            )

        assert candidate.maintainership is None
        pipeline._observer.on_error.assert_called_once()
        assert pipeline._observer.on_error.call_args[0][0] == "maintainership_classify"

    def test_candidates_jsonl_carries_maintainership_for_export(self, tmp_path):
        """P6.1 export gap: acquisition.py appends the candidate record to
        candidates.jsonl BEFORE classify() runs, so the on-disk copy never
        had a maintainership field. The classify() call site must re-append
        so github/export.py's username-keyed join (last record wins) picks
        up the payload."""
        pipeline = _make_pipeline()
        pipeline._runtime_run_id = None
        pipeline._ensure_services()
        pipeline.brief_obj.target_projects = ["kubernetes/kubernetes"]
        pipeline._client = MagicMock()
        pipeline._execution_engine = MagicMock()
        pipeline._side_effects_service = MagicMock()
        pipeline._side_effects_service.handle_full_decision = AsyncMock(return_value=None)
        pipeline.candidates_path = tmp_path / "candidates.jsonl"

        candidate = _make_candidate("octocat", "The Octocat")
        # Simulate acquisition.py's pre-classify append.
        append_jsonl(pipeline.candidates_path, pipeline._candidate_record(candidate))

        query = _make_query(id=8, channel="user_search")
        progress = GitHubProgress(brief_name="test")

        facial_decision = MagicMock(decision="FACIAL_YES", confidence=0.9, rationale="")
        full_decision = MagicMock(decision="SAVE", confidence=0.9, path="direct")
        classification = MaintainershipClassification(
            level="project_lead",
            confidence=0.85,
            evidence_sources=["governance:kubernetes/kubernetes"],
            signals={},
        )

        with patch.object(github_orchestrator, "github_facial_judge_batch", return_value=[facial_decision]), \
             patch.object(github_orchestrator, "github_full_judge", return_value=full_decision), \
             patch.object(github_orchestrator, "classify_maintainership", AsyncMock(return_value=classification)):
            asyncio.run(
                pipeline._process_v2_candidates_batch([("octocat", candidate)], query, progress)
            )

        rows = [json.loads(line) for line in pipeline.candidates_path.read_text().splitlines()]
        assert len(rows) == 2  # pre-classify row + post-classify row
        assert rows[0].get("maintainership") is None
        assert rows[-1]["maintainership"]["level"] == "project_lead"


# ---------------------------------------------------------------------------
# P6.1 exit gate — fixture-driven end-to-end
# ---------------------------------------------------------------------------


def test_p6_exit_gate_facial_yes_classify_judge_save_export(tmp_path):
    """facial-YES -> classify -> judge-with-evidence -> save -> export
    columns populated, with mocked API/judge clients.

    Note: github/export.py joins candidates.jsonl against
    final_judgments.jsonl by candidate_name; no GitHub production code path
    currently writes final_judgments.jsonl (a pre-existing gap outside P6's
    scope — see the executing agent's final report). This fixture authors
    that judgment row directly, the same way tests/test_github_export.py
    does, so the assertion is scoped to what P6.1 actually changed: that
    candidates.jsonl carries a populated maintainership payload the export
    join can read.
    """
    pipeline = _make_pipeline()
    pipeline._runtime_run_id = None
    pipeline._ensure_services()
    pipeline.brief_obj.target_projects = ["kubernetes/kubernetes"]
    pipeline._client = MagicMock()
    pipeline._execution_engine = MagicMock()
    pipeline._side_effects_service = MagicMock()
    pipeline._side_effects_service.handle_full_decision = AsyncMock(return_value=None)
    pipeline.candidates_path = tmp_path / "candidates.jsonl"

    candidate = _make_candidate("octocat", "The Octocat")
    query = _make_query(id=1, channel="user_search")
    progress = GitHubProgress(brief_name="test")

    facial_decision = MagicMock(decision="FACIAL_YES", confidence=1.0, rationale="")
    full_decision = MagicMock(
        decision="SAVE", confidence=0.91, path="DIRECT:Infrastructure"
    )
    classification = MaintainershipClassification(
        level="maintainer",
        confidence=0.82,
        evidence_sources=["merge_authority:kubernetes/kubernetes:14PRs"],
        signals={},
    )

    with patch.object(github_orchestrator, "github_facial_judge_batch", return_value=[facial_decision]), \
         patch.object(github_orchestrator, "github_full_judge", return_value=full_decision), \
         patch.object(github_orchestrator, "classify_maintainership", AsyncMock(return_value=classification)):
        asyncio.run(
            pipeline._process_v2_candidates_batch([("octocat", candidate)], query, progress)
        )

    assert candidate.maintainership == classification.to_dict()

    # Hand-author the SAVE judgment row (see docstring — no production
    # writer exists yet) so export.py's join has a match.
    append_jsonl(
        tmp_path / "final_judgments.jsonl",
        {
            "candidate_name": "The Octocat",
            "stage": "full",
            "decision": "SAVE",
            "confidence": 0.91,
            "path": "DIRECT:Infrastructure",
            "rationale": "Sustained merge authority on kubernetes/kubernetes.",
        },
    )

    from github.export import export_saved_candidates_csv

    csv_path = export_saved_candidates_csv(tmp_path, csv_path=tmp_path / "saved_candidates.csv")

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["Maintainership Level"] == "maintainer"
    assert row["Maintainership Confidence"] == "0.82"
    assert "merge_authority:kubernetes/kubernetes:14PRs" in row["Maintainership Evidence"]
    assert row["Maintainership Target Project"] == "kubernetes/kubernetes"


# ---------------------------------------------------------------------------
# P6.2 — graph-expansion self-cancellation
# ---------------------------------------------------------------------------


class TestGraphExpansionSelfCancellation:
    """Regression coverage for the P6.2 fix.

    The old bug lived behind ``if pipeline._runtime_run_id:`` in
    work_units.py's process_graph_expansion_queue (enqueue-time marking of
    graph_expansion_processed, both in-memory and in runtime_state). A test
    harness with ``_runtime_run_id = None`` never executes that guarded
    block, so it can pass identically whether the bug is present or fixed.
    These tests attach REAL runtime state (mirroring
    TestPriorRunDataSourcing's ``_attach_runtime_state`` pattern) and start
    an actual run so ``_runtime_run_id`` is a truthy DB-backed id — the same
    condition production runs hit.
    """

    def _enqueue_one_seed(self, tmp_path):
        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline._ensure_services()
        run_id, _initial_progress = pipeline._runtime_bridge.start_or_resume_run(resume=False)
        pipeline._runtime_run_id = run_id
        query = _make_query(id=1)
        progress = GitHubProgress(
            brief_name="test",
            queries=[query],
            graph_expansion_queue=[
                {
                    "username": "seed-user",
                    "reason": "SAVE",
                    "confidence": 0.92,
                    "capability_area": "research",
                    "added_at": "2026-01-01T00:00:00+00:00",
                }
            ],
            graph_expansion_processed=[],
        )
        asyncio.run(
            pipeline._work_unit_service.process_graph_expansion_queue(progress, progress.queries)
        )
        return pipeline, progress

    def test_enqueue_does_not_mark_seed_processed(self, tmp_path):
        """Regression: process_graph_expansion_queue used to mark the seed
        processed the moment it created the graph_expansion query, before
        that query ever executed. With a real (truthy) _runtime_run_id this
        reproduces the exact condition the old enqueue-time guard fired
        under."""
        pipeline, progress = self._enqueue_one_seed(tmp_path)

        assert progress.graph_expansion_processed == []
        assert len(progress.queries) == 2
        expansion_query = progress.queries[1]
        assert expansion_query.channel == "graph_expansion"
        assert expansion_query.query == "seed-user"

    def test_expand_graph_actually_fetches_after_enqueue(self, tmp_path):
        """Regression: previously self-cancelled — _expand_graph returned
        (0, []) without ever calling get_followers/get_following because
        the seed was already (wrongly) marked processed at enqueue time."""
        pipeline, progress = self._enqueue_one_seed(tmp_path)
        expansion_query = progress.queries[1]

        client = MagicMock()
        client.get_followers = AsyncMock(return_value=[{"login": "alice"}, {"login": "bob"}])
        client.get_following = AsyncMock(return_value=[{"login": "carol"}])

        pre_dedup_count, usernames = asyncio.run(
            pipeline._expand_graph(client, expansion_query, progress)
        )

        client.get_followers.assert_awaited_once()
        client.get_following.assert_awaited_once()
        assert pre_dedup_count == 3
        assert sorted(usernames) == ["alice", "bob", "carol"]
        assert "seed-user" in progress.graph_expansion_processed

    def test_second_execution_attempt_is_still_guarded(self, tmp_path):
        """The processed-check must still guard against re-processing the
        same seed a second time — the fix moves WHEN it's marked, not
        whether it's marked at all."""
        pipeline, progress = self._enqueue_one_seed(tmp_path)
        expansion_query = progress.queries[1]

        client = MagicMock()
        client.get_followers = AsyncMock(return_value=[{"login": "alice"}])
        client.get_following = AsyncMock(return_value=[])
        asyncio.run(pipeline._expand_graph(client, expansion_query, progress))

        client.get_followers.reset_mock()
        client.get_following.reset_mock()
        pre_dedup_count, usernames = asyncio.run(
            pipeline._expand_graph(client, expansion_query, progress)
        )

        client.get_followers.assert_not_awaited()
        assert (pre_dedup_count, usernames) == (0, [])

    def test_expansion_query_does_not_increment_zero_result_streak(self, tmp_path):
        """The exhaustion-interaction regression named in the spec: a
        real (non-self-cancelled) expansion query that finds followers must
        not feed ExhaustionState's zero-result streak."""
        from github.query_validator import ExhaustionState

        pipeline, progress = self._enqueue_one_seed(tmp_path)
        expansion_query = progress.queries[1]

        client = MagicMock()
        client.get_followers = AsyncMock(return_value=[{"login": "alice"}])
        client.get_following = AsyncMock(return_value=[{"login": "bob"}])

        pre_dedup_count, usernames = asyncio.run(
            pipeline._expand_graph(client, expansion_query, progress)
        )

        exhaustion = ExhaustionState()
        exhaustion.record_query_result(
            channel="graph_expansion",
            saves=0,
            candidates=len(usernames),
            pre_dedup=pre_dedup_count,
            post_dedup=len(usernames),
        )

        assert len(usernames) == 2
        assert exhaustion.channels["graph_expansion"].zero_result_streak == 0


# ---------------------------------------------------------------------------
# P6.3 — adaptation inputs made honest
# ---------------------------------------------------------------------------


class TestBuildBatchReportHonesty:
    def test_previously_zeroed_fields_are_populated(self):
        pipeline = _make_pipeline()
        pipeline._batch_baseline_stats = {"rejected": 1, "insufficient": 0}
        pipeline.stats["rejected"] = 4
        pipeline.stats["insufficient"] = 2

        candidate_a = _make_candidate("alice", "Alice A")
        candidate_a.languages = {"Python": 5000, "Go": 100}
        candidate_a.top_repos = [GitHubRepo(name="repo-a", language="Python")]
        candidate_b = _make_candidate("bob", "Bob B")
        candidate_b.languages = {"Rust": 9000}
        candidate_b.top_repos = [GitHubRepo(name="repo-b", language="Rust")]
        pipeline._batch_save_candidates = [candidate_a, candidate_b]

        batch_stats = [
            {
                "query_id": 1, "name": "q1", "query_string": "x", "channel": "user_search",
                "saves": 2, "candidates": 10, "hit_result_cap": True,
            },
            {
                "query_id": 2, "name": "q2", "query_string": "y", "channel": "code_search",
                "saves": 0, "candidates": 5, "hit_result_cap": False,
            },
        ]

        report = pipeline._build_batch_report(batch_stats)

        assert report.total_rejects == 3
        assert report.total_insufficient == 2
        assert set(report.common_languages_in_saves) == {"Python", "Rust"}
        assert set(report.common_repos_in_saves) == {"repo-a", "repo-b"}
        assert report.queries_hitting_result_cap == [1]

    def test_no_saves_no_rejects_reports_honest_zeros(self):
        """A genuinely quiet batch reports observed zeros, not omission —
        distinguishing this from the affirmative-zero anti-pattern (which
        is claiming a count that was never measured)."""
        pipeline = _make_pipeline()
        pipeline._batch_baseline_stats = dict(pipeline.stats)
        pipeline._batch_save_candidates = []

        batch_stats = [
            {
                "query_id": 1, "name": "q1", "query_string": "x", "channel": "user_search",
                "saves": 0, "candidates": 3, "hit_result_cap": False,
            },
        ]
        report = pipeline._build_batch_report(batch_stats)

        assert report.total_rejects == 0
        assert report.total_insufficient == 0
        assert report.common_languages_in_saves == []
        assert report.common_repos_in_saves == []
        assert report.queries_hitting_result_cap == []


class TestPriorRunDataSourcing:
    def test_form_github_strategy_receives_prior_progress_when_present(self, tmp_path):
        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline.state_dir = pipeline.output_dir
        pipeline.log_path = pipeline.output_dir / "log.jsonl"
        prior_payload = {"brief_name": "test", "candidates_saved": 3, "queries": []}
        pipeline.progress_path.write_text(json.dumps(prior_payload))

        mock_form = MagicMock(return_value=([], "rationale"))
        with patch.object(github_orchestrator, "form_github_strategy", mock_form):
            _run_github_pipeline_to_completion(pipeline)

        mock_form.assert_called_once()
        args, _kwargs = mock_form.call_args
        assert args[0] is pipeline.brief_obj
        assert args[1] == prior_payload

    def test_form_github_strategy_receives_none_when_no_prior_progress(self, tmp_path):
        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline.state_dir = pipeline.output_dir
        pipeline.log_path = pipeline.output_dir / "log.jsonl"
        assert not pipeline.progress_path.exists()

        mock_form = MagicMock(return_value=([], "rationale"))
        with patch.object(github_orchestrator, "form_github_strategy", mock_form):
            _run_github_pipeline_to_completion(pipeline)

        mock_form.assert_called_once_with(pipeline.brief_obj, None)


# ---------------------------------------------------------------------------
# P6.4 — bias monitor lifecycle
# ---------------------------------------------------------------------------


def _v2_brief_for_bias_lifecycle():
    brief = MagicMock()
    brief.id = "bias-lifecycle-brief"
    brief.has_v2_schema = True
    new_brief = MagicMock()
    new_brief.bias_controls = MagicMock(
        max_consecutive_saves=5,
        max_consecutive_rejects=20,
        parse_failure_alarm_rate=0.03,
    )
    new_brief.facial_calibration = MagicMock(
        expected_yes_rate_low=0.25,
        expected_yes_rate_high=0.55,
    )
    brief._new_brief = new_brief
    return brief


class TestBiasMonitorLifecycle:
    def test_loads_checkpoint_at_construction_when_present(self, tmp_path):
        output_dir = tmp_path / "run"
        output_dir.mkdir()
        bias_path = output_dir / "bias_monitor.json"
        bias_path.write_text(json.dumps({
            "decisions": [{
                "candidate_id": "alice", "string_id": "1", "stage": "full",
                "decision": "SAVE", "confidence": 0.9, "capability_area": None,
                "timestamp": 0,
            }],
            "alerts_fired": ["consecutive_saves:1"],
        }))

        brief = _v2_brief_for_bias_lifecycle()
        with patch.object(github_orchestrator, "load_brief", return_value=brief), \
             patch.object(github_orchestrator, "init_judger"):
            pipeline = GitHubPipeline(brief_path="fake.json", output_dir=str(output_dir))

        assert pipeline._bias_monitor is not None
        assert len(pipeline._bias_monitor._decisions) == 1
        assert pipeline._bias_monitor._decisions[0].candidate_id == "alice"
        assert "consecutive_saves:1" in pipeline._bias_monitor._alerts_fired

    def test_fresh_monitor_when_no_checkpoint(self, tmp_path):
        output_dir = tmp_path / "run2"
        output_dir.mkdir()
        brief = _v2_brief_for_bias_lifecycle()
        with patch.object(github_orchestrator, "load_brief", return_value=brief), \
             patch.object(github_orchestrator, "init_judger"):
            pipeline = GitHubPipeline(brief_path="fake.json", output_dir=str(output_dir))

        assert pipeline._bias_monitor is not None
        assert pipeline._bias_monitor._decisions == []

    def test_checkpoint_saved_on_exit(self, tmp_path):
        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline.state_dir = pipeline.output_dir
        pipeline.log_path = pipeline.output_dir / "log.jsonl"
        pipeline.bias_path = pipeline.output_dir / "bias_monitor.json"
        pipeline._bias_monitor = BiasMonitor(
            max_consecutive_saves=5, max_consecutive_rejects=20,
            parse_failure_alarm_rate=0.03,
            expected_facial_yes_low=0.25, expected_facial_yes_high=0.55,
        )
        pipeline._bias_monitor.record_decision(DecisionRecord(
            candidate_id="x", string_id="1", stage="full",
            decision="SAVE", confidence=0.9, capability_area=None,
        ))

        _run_github_pipeline_to_completion(pipeline)

        assert pipeline.bias_path.exists()
        saved = json.loads(pipeline.bias_path.read_text())
        assert len(saved["decisions"]) == 1
        assert saved["decisions"][0]["candidate_id"] == "x"

    def test_bias_summary_passed_at_session_end_not_hardcoded_none(self, tmp_path):
        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline.state_dir = pipeline.output_dir
        pipeline.log_path = pipeline.output_dir / "log.jsonl"
        pipeline._bias_monitor = BiasMonitor(
            max_consecutive_saves=5, max_consecutive_rejects=20,
            parse_failure_alarm_rate=0.03,
            expected_facial_yes_low=0.25, expected_facial_yes_high=0.55,
        )
        pipeline._bias_monitor.record_decision(DecisionRecord(
            candidate_id="x", string_id="1", stage="full",
            decision="SAVE", confidence=0.9, capability_area=None,
        ))

        observer_instance = MagicMock()
        with patch.object(github_orchestrator, "GitHubClient", _FakeGitHubClient), \
             patch.object(github_orchestrator, "SessionObserver", lambda *a, **kw: observer_instance):
            asyncio.run(pipeline.run(resume=False))

        observer_instance.on_session_end.assert_called_once()
        args, _kwargs = observer_instance.on_session_end.call_args
        bias_summary = args[2]
        assert bias_summary is not None
        assert bias_summary["total_decisions"] == 1

    def test_bias_summary_lands_in_written_metrics_artifact(self, tmp_path):
        """P6.4 follow-up: passing bias_summary to on_session_end is not
        enough — SessionObserver.on_session_end (github/observability/
        observer.py) previously ignored the parameter entirely, so the
        P6.4 change was observationally a no-op. This runs the REAL
        SessionObserver (not a mock) so the assertion is on-disk truth."""
        from github.observability.observer import SessionObserver as RealSessionObserver

        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline.state_dir = pipeline.output_dir
        pipeline.log_path = pipeline.output_dir / "log.jsonl"
        pipeline._bias_monitor = BiasMonitor(
            max_consecutive_saves=5, max_consecutive_rejects=20,
            parse_failure_alarm_rate=0.03,
            expected_facial_yes_low=0.25, expected_facial_yes_high=0.55,
        )
        pipeline._bias_monitor.record_decision(DecisionRecord(
            candidate_id="x", string_id="1", stage="full",
            decision="SAVE", confidence=0.9, capability_area=None,
        ))

        with patch.object(github_orchestrator, "GitHubClient", _FakeGitHubClient), \
             patch.object(github_orchestrator, "SessionObserver", RealSessionObserver):
            asyncio.run(pipeline.run(resume=False))

        metrics_files = list(pipeline.output_dir.glob("session_*_metrics.jsonl"))
        assert len(metrics_files) == 1
        events = [json.loads(line) for line in metrics_files[0].read_text().splitlines()]
        final_events = [e for e in events if e.get("event") == "session_final"]
        assert len(final_events) == 1
        assert final_events[0]["bias_summary"]["total_decisions"] == 1

        report_files = list(pipeline.output_dir.glob("session_*_report.md"))
        assert len(report_files) == 1
        assert "Bias Monitor" in report_files[0].read_text()

    def test_no_bias_summary_omits_key_from_metrics_artifact(self, tmp_path):
        """None must not scaffold into a null placeholder in the artifact."""
        from github.observability.observer import SessionObserver as RealSessionObserver

        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline.state_dir = pipeline.output_dir
        pipeline.log_path = pipeline.output_dir / "log.jsonl"
        pipeline._bias_monitor = None

        with patch.object(github_orchestrator, "GitHubClient", _FakeGitHubClient), \
             patch.object(github_orchestrator, "SessionObserver", RealSessionObserver):
            asyncio.run(pipeline.run(resume=False))

        metrics_files = list(pipeline.output_dir.glob("session_*_metrics.jsonl"))
        assert len(metrics_files) == 1
        events = [json.loads(line) for line in metrics_files[0].read_text().splitlines()]
        final_events = [e for e in events if e.get("event") == "session_final"]
        assert len(final_events) == 1
        assert "bias_summary" not in final_events[0]

        report_files = list(pipeline.output_dir.glob("session_*_report.md"))
        assert len(report_files) == 1
        assert "Bias Monitor" not in report_files[0].read_text()


class TestBiasPauseStopsQuery:
    def test_pause_stops_current_query_not_just_prints(self):
        """P6.4 — do NOT copy linkedin's pause-severity theater. A pause
        alert must stop the query it fired in: no more candidates in the
        batch already in flight, and no more candidates fetched for the
        rest of the query."""
        pipeline = _make_pipeline()
        pipeline._runtime_run_id = None
        pipeline._ensure_services()
        pipeline.brief_obj.target_projects = []
        pipeline.brief_obj.has_v2_schema = True
        pipeline._client = None
        pipeline._execution_engine = MagicMock()
        pipeline._side_effects_service = MagicMock()
        pipeline._side_effects_service.handle_full_decision = AsyncMock(return_value=None)
        pipeline._bias_monitor = BiasMonitor(
            max_consecutive_saves=2, max_consecutive_rejects=20,
            parse_failure_alarm_rate=0.03,
            expected_facial_yes_low=0.0, expected_facial_yes_high=1.0,
        )

        usernames = [f"user{i}" for i in range(15)]
        candidates = {u: _make_candidate(u, u) for u in usernames}

        async def _fake_prepare(enricher, username, query, progress, result_rank=0):
            return SimpleNamespace(
                candidate=candidates[username],
                terminal_decision=None,
            )

        pipeline._acquisition_service = MagicMock()
        pipeline._acquisition_service.prepare_candidate_for_evaluation = AsyncMock(
            side_effect=_fake_prepare
        )

        client = MagicMock()
        client.search_users = AsyncMock(
            return_value=(len(usernames), [{"login": u} for u in usernames])
        )
        enricher = MagicMock()
        query = _make_query(id=9, channel="user_search")
        progress = GitHubProgress(brief_name="test")

        def _facial_yes_batch(portfolio_texts, brief):
            return [
                MagicMock(decision="FACIAL_YES", confidence=1.0, rationale="")
                for _ in portfolio_texts
            ]

        with patch.object(github_orchestrator, "github_facial_judge_batch", side_effect=_facial_yes_batch), \
             patch.object(
                 github_orchestrator, "github_full_judge",
                 return_value=MagicMock(decision="SAVE", confidence=0.9, path="direct"),
             ) as full_judge_mock:
            asyncio.run(pipeline._execute_single_query(client, enricher, query, progress))

        # Pause fires after the 2nd consecutive SAVE (max_consecutive_saves=2);
        # the 3rd candidate in the same 10-candidate batch is never judged.
        assert full_judge_mock.call_count == 2
        # The query stopped fetching entirely — usernames[10:] were never
        # even prepared, let alone judged.
        assert pipeline._acquisition_service.prepare_candidate_for_evaluation.call_count == 10
        assert pipeline._bias_pause_active is True
        pipeline._observer.on_query_stopped_early.assert_called_once()
        stop_args = pipeline._observer.on_query_stopped_early.call_args[0]
        assert stop_args[1] == "bias monitor pause"


# ---------------------------------------------------------------------------
# P6.5 — geography veto fail-open for unconfigured geographies
# ---------------------------------------------------------------------------


class TestGeographyVetoFailOpen:
    @staticmethod
    def _pipeline_with_geo(geo: str):
        pipeline = _make_pipeline()
        pipeline.brief_obj.has_v2_schema = False
        pipeline.brief_obj.permanent_filters = {"Location": geo}
        return pipeline

    def test_unconfigured_geo_fails_open_and_counts(self):
        pipeline = self._pipeline_with_geo("Argentina")
        candidate = _make_candidate("juan", "Juan Perez")
        candidate.user.location = ""  # blank location — previously mass-rejected
        query = _make_query(channel="code_search")

        result = pipeline._passes_geography_check(candidate, query)

        assert result is True
        assert pipeline.stats["geo_unconfigured"] == 1

    def test_unconfigured_geo_with_populated_location_also_fails_open(self):
        """Even a candidate with an explicit non-matching-string location
        must fail open for an unconfigured geography — the veto doesn't
        fire at all, not just the blank-location branch."""
        pipeline = self._pipeline_with_geo("India")
        candidate = _make_candidate("dev", "Some Dev")
        candidate.user.location = "Berlin, Germany"
        query = _make_query(channel="code_search")

        result = pipeline._passes_geography_check(candidate, query)

        assert result is True
        assert pipeline.stats["geo_unconfigured"] == 1

    def test_configured_geo_still_vetoes_non_matching_location(self):
        """Fail-open must not weaken the two authored dictionaries."""
        pipeline = self._pipeline_with_geo("Brazil")
        candidate = _make_candidate("jane", "Jane Doe")
        candidate.user.location = "Berlin, Germany"
        query = _make_query(channel="code_search")

        result = pipeline._passes_geography_check(candidate, query)

        assert result is False
        assert "geo_unconfigured" not in pipeline.stats

    def test_configured_geo_still_passes_matching_location(self):
        pipeline = self._pipeline_with_geo("Brazil")
        candidate = _make_candidate("joao", "Joao Silva")
        candidate.user.location = "Sao Paulo, Brazil"
        query = _make_query(channel="code_search")

        result = pipeline._passes_geography_check(candidate, query)

        assert result is True
        assert "geo_unconfigured" not in pipeline.stats

    def test_bare_co_tld_no_longer_passes_colombia(self):
        pipeline = self._pipeline_with_geo("Colombia")
        candidate = _make_candidate("dev", "Some Dev")
        candidate.user.location = ""
        candidate.user.blog = "https://example.co"
        query = _make_query(channel="code_search")

        result = pipeline._passes_geography_check(candidate, query)

        assert result is False

    def test_com_co_tld_still_passes_colombia(self):
        pipeline = self._pipeline_with_geo("Colombia")
        candidate = _make_candidate("dev", "Some Dev")
        candidate.user.location = ""
        candidate.user.blog = "https://empresa.com.co"
        query = _make_query(channel="code_search")

        result = pipeline._passes_geography_check(candidate, query)

        assert result is True

    def test_spanish_cognate_markers_no_longer_trip_portuguese_detection(self):
        """como/para/sobre/sistema/ambiente/utilizar are Spanish cognates
        too — they must no longer count as Portuguese-language evidence."""
        pipeline = self._pipeline_with_geo("Brazil")
        candidate = _make_candidate("dev", "Some Dev")
        candidate.user.location = ""
        candidate.readme_text = (
            "Sistema para utilizar como ambiente de desarrollo sobre la nube"
        )
        query = _make_query(channel="code_search")

        result = pipeline._passes_geography_check(candidate, query)

        assert result is False

    def test_genuine_portuguese_markers_still_pass_brazil(self):
        """The fix must not gut the whole Portuguese-detection signal —
        only the ambiguous Spanish-cognate words."""
        pipeline = self._pipeline_with_geo("Brazil")
        candidate = _make_candidate("dev", "Some Dev")
        candidate.user.location = ""
        candidate.readme_text = (
            "Não é possível também usar o repositório sem configuração "
            "e implementação através do usuário"
        )
        query = _make_query(channel="code_search")

        result = pipeline._passes_geography_check(candidate, query)

        assert result is True


class TestGeoUnconfiguredCounterHonesty:
    """github/acquisition.py calls _passes_geography_check at both the
    light-enrich and full-enrich stages for non-user_search channels
    (mirrors the light/full call sites that feed geo_filtered /
    geo_filtered_light). Unlike geo_filtered, the unconfigured-geography
    branch fails OPEN — it never terminates the candidate — so both calls
    always run for the same candidate, and incrementing on every call
    double-counted a single candidate's fail-open."""

    def test_light_then_full_call_increments_aggregate_once(self):
        pipeline = _make_pipeline()
        pipeline.brief_obj.has_v2_schema = False
        pipeline.brief_obj.permanent_filters = {"Location": "Argentina"}
        candidate = _make_candidate("juan", "Juan Perez")
        candidate.user.location = ""
        query = _make_query(channel="code_search")

        light_result = pipeline._passes_geography_check(candidate, query, stage="light")
        full_result = pipeline._passes_geography_check(candidate, query, stage="full")

        assert light_result is True
        assert full_result is True
        assert pipeline.stats["geo_unconfigured"] == 1
        assert pipeline.stats["geo_unconfigured_light"] == 1
        assert pipeline.stats["geo_unconfigured_full"] == 1

    def test_full_enrich_geo_call_site_passes_full_stage(self):
        """acquisition.py's full-enrich call site (the second of the two
        call sites feeding this branch) must identify itself as 'full' —
        pinning the call-site contract so a future edit can't silently
        drop back to the double-counting default."""
        import inspect

        import github.acquisition as acquisition_mod

        source = inspect.getsource(
            acquisition_mod.GitHubAcquisitionService.prepare_candidate_for_evaluation
        )
        assert 'stage="light"' in source
        assert 'stage="full"' in source

    def test_user_search_channel_never_touches_geo_unconfigured(self):
        """Regression guard: user_search short-circuits inside
        _passes_geography_check before the unconfigured-geo branch, at
        every stage — the stage split must not change that."""
        pipeline = _make_pipeline()
        pipeline.brief_obj.has_v2_schema = False
        pipeline.brief_obj.permanent_filters = {"Location": "Argentina"}
        candidate = _make_candidate("juan", "Juan Perez")
        candidate.user.location = ""
        query = _make_query(channel="user_search")

        result = pipeline._passes_geography_check(candidate, query, stage="full")

        assert result is True
        assert "geo_unconfigured" not in pipeline.stats
        assert "geo_unconfigured_full" not in pipeline.stats
