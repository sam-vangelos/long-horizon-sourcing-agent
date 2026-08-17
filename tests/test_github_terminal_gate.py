"""Regression tests for GitHub terminal-gate honor and run-status truth.

Covers W0-S2: acquisition terminal decisions must block judgment, pre-loop
failures must record error status, and judge TypeError must propagate.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from github.schemas import GitHubProgress
from github.orchestrator import JudgeContractError
from shared.execution import AcquisitionResult

from tests.test_github_pipeline import (
    _attach_runtime_state,
    _make_candidate,
    _make_pipeline,
    _make_query,
    GitHubPipeline,
    github_orchestrator,
)


def _pipeline_for_batch_query():
    pipeline = _make_pipeline()
    pipeline._runtime_run_id = None
    pipeline._ensure_services()
    pipeline.brief_obj.has_v2_schema = True
    return pipeline


async def _run_single_user_query(pipeline, acquisition_result, username: str = "gate_user"):
    pipeline._search_users = AsyncMock(return_value=(1, [username]))
    pipeline._acquisition_service.prepare_candidate_for_evaluation = AsyncMock(
        return_value=acquisition_result
    )
    client = MagicMock()
    enricher = MagicMock()
    query = _make_query(channel="user_search")
    progress = GitHubProgress(brief_name="test")
    facial_batch_mock = MagicMock()
    with patch.object(
        github_orchestrator, "github_facial_judge_batch", facial_batch_mock
    ):
        await pipeline._execute_single_query(client, enricher, query, progress)
    return facial_batch_mock


class TestTerminalDecisionBlocksJudgment:
    @pytest.mark.parametrize(
        "terminal_decision",
        ["GEO_FILTERED", "PRESCREEN_SKIP", "INSUFFICIENT_DATA"],
    )
    def test_terminal_decision_wrapper_returns_none(self, terminal_decision):
        pipeline = _pipeline_for_batch_query()
        candidate = _make_candidate("blocked", "Blocked User")
        pipeline._acquisition_service.prepare_candidate_for_evaluation = AsyncMock(
            return_value=AcquisitionResult(
                candidate=candidate,
                terminal_decision=terminal_decision,
            )
        )

        result = asyncio.run(
            pipeline._prepare_candidate_for_evaluation(
                MagicMock(), "blocked", _make_query(), GitHubProgress(brief_name="test")
            )
        )

        assert result is None

    def test_geo_filtered_never_reaches_facial_judge_batch(self):
        pipeline = _pipeline_for_batch_query()
        candidate = _make_candidate("geo_user", "Geo User")
        acquisition_result = AcquisitionResult(
            candidate=candidate,
            terminal_decision="GEO_FILTERED",
        )
        facial_batch_mock = asyncio.run(
            _run_single_user_query(pipeline, acquisition_result, "geo_user")
        )
        facial_batch_mock.assert_not_called()

    def test_prescreen_skip_never_reaches_facial_judge_batch(self):
        pipeline = _pipeline_for_batch_query()
        candidate = _make_candidate("skip_user", "Skip User")
        acquisition_result = AcquisitionResult(
            candidate=candidate,
            terminal_decision="PRESCREEN_SKIP",
        )
        facial_batch_mock = asyncio.run(
            _run_single_user_query(pipeline, acquisition_result, "skip_user")
        )
        facial_batch_mock.assert_not_called()

    def test_insufficient_data_never_reaches_facial_judge_batch(self):
        pipeline = _pipeline_for_batch_query()
        candidate = _make_candidate("thin_user", "Thin User")
        acquisition_result = AcquisitionResult(
            candidate=candidate,
            terminal_decision="INSUFFICIENT_DATA",
        )
        facial_batch_mock = asyncio.run(
            _run_single_user_query(pipeline, acquisition_result, "thin_user")
        )
        facial_batch_mock.assert_not_called()

    def test_clean_pass_flows_to_judgment(self):
        pipeline = _pipeline_for_batch_query()
        candidate = _make_candidate("clean_user", "Clean User")
        acquisition_result = AcquisitionResult(candidate=candidate)
        pipeline._search_users = AsyncMock(return_value=(1, ["clean_user"]))
        pipeline._acquisition_service.prepare_candidate_for_evaluation = AsyncMock(
            return_value=acquisition_result
        )
        client = MagicMock()
        enricher = MagicMock()
        query = _make_query(channel="user_search")
        progress = GitHubProgress(brief_name="test")
        facial_decision = MagicMock(decision="FACIAL_NO", confidence=1.0, rationale="no")
        with patch.object(
            github_orchestrator, "github_facial_judge_batch", return_value=[facial_decision]
        ) as facial_batch_mock:
            asyncio.run(pipeline._execute_single_query(client, enricher, query, progress))
        facial_batch_mock.assert_called_once()

    def test_empty_enrichment_unchanged(self):
        pipeline = _pipeline_for_batch_query()
        acquisition_result = AcquisitionResult(
            terminal_decision="EMPTY_ENRICHMENT",
            skip_reason="light_enrich returned no candidate",
        )
        pipeline._acquisition_service.prepare_candidate_for_evaluation = AsyncMock(
            return_value=acquisition_result
        )
        wrapper_result = asyncio.run(
            pipeline._prepare_candidate_for_evaluation(
                MagicMock(),
                "empty_user",
                _make_query(),
                GitHubProgress(brief_name="test"),
            )
        )
        assert wrapper_result is None
        facial_batch_mock = asyncio.run(
            _run_single_user_query(pipeline, acquisition_result, "empty_user")
        )
        facial_batch_mock.assert_not_called()


class TestPreLoopFailureRunStatus:
    def test_missing_token_records_error_not_completed(self, tmp_path):
        class _RaisingGitHubClient:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("GITHUB_TOKEN not set")

        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline._ensure_runtime_state()
        pipeline.state_dir = pipeline.output_dir
        pipeline.log_path = pipeline.output_dir / "log.jsonl"
        finish_run_mock = MagicMock()
        pipeline._safety.finish_run = finish_run_mock

        with patch.object(github_orchestrator, "GitHubClient", _RaisingGitHubClient), \
             patch.object(github_orchestrator, "SessionObserver", lambda *a, **kw: MagicMock()):
            with pytest.raises(RuntimeError, match="GITHUB_TOKEN not set"):
                asyncio.run(pipeline.run(resume=False))

        finish_run_mock.assert_called_once()
        assert finish_run_mock.call_args.kwargs["status"] == "error"


class TestJudgeTypeErrorPropagation:
    def test_typeerror_finishes_attempt_and_raises_judge_contract_error(self):
        pipeline = _make_pipeline()
        pipeline._runtime_run_id = None
        pipeline._ensure_services()
        pipeline._execution_engine = MagicMock()
        pipeline._side_effects_service = MagicMock()

        candidate = _make_candidate("alice", "Alice")
        query = _make_query()
        progress = GitHubProgress(brief_name="test")
        facial_decision = MagicMock(decision="FACIAL_YES", confidence=0.9, rationale="")

        finish_mock = MagicMock()
        pipeline._finish_failure_decision_attempt = finish_mock

        with patch.object(
            github_orchestrator, "github_facial_judge_batch", return_value=[facial_decision]
        ), patch.object(
            github_orchestrator,
            "github_full_judge",
            side_effect=TypeError("unexpected keyword argument"),
        ):
            with pytest.raises(JudgeContractError, match="unexpected keyword argument"):
                asyncio.run(
                    pipeline._process_v2_candidates_batch(
                        [("alice", candidate)], query, progress
                    )
                )

        finish_mock.assert_called_once()
        assert pipeline.stats["parse_failures"] == 1

    def test_typeerror_from_query_records_run_error_status(self, tmp_path):
        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline._ensure_runtime_state()
        pipeline._ensure_services()
        pipeline._execution_engine = MagicMock()
        pipeline.state_dir = pipeline.output_dir
        pipeline.log_path = pipeline.output_dir / "log.jsonl"
        finish_run_mock = MagicMock()
        pipeline._safety.finish_run = finish_run_mock

        candidate = _make_candidate("alice", "Alice")
        facial_decision = MagicMock(decision="FACIAL_YES", confidence=0.9, rationale="")

        class _Client:
            limiter = MagicMock(total_calls=0, remaining=MagicMock(return_value=1000))

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def validate_credentials(self):
                return None

        pipeline._acquisition_service.prepare_candidate_for_evaluation = AsyncMock(
            return_value=AcquisitionResult(candidate=candidate)
        )

        with patch.object(github_orchestrator, "GitHubClient", _Client), \
             patch.object(github_orchestrator, "SessionObserver", lambda *a, **kw: MagicMock()), \
             patch.object(
                 github_orchestrator,
                 "form_github_strategy",
                 return_value=([_make_query()], "test"),
             ), patch.object(
                 github_orchestrator, "github_facial_judge_batch", return_value=[facial_decision]
             ), patch.object(
                 github_orchestrator,
                 "github_full_judge",
                 side_effect=TypeError("unexpected keyword argument"),
             ), patch.object(
                 GitHubPipeline,
                 "_search_users",
                 AsyncMock(return_value=(1, ["alice"])),
             ):
            asyncio.run(pipeline.run(resume=False))

        finish_run_mock.assert_called_once()
        assert finish_run_mock.call_args.kwargs["status"] == "error"
        assert pipeline.stats["parse_failures"] == 1
