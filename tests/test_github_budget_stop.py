"""Regression tests for GitHub API budget exhaustion stop behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from github.enricher import GitHubEnricher
from github.outreach import generate_outreach
from github.schemas import ContactInfo, GitHubCandidate, GitHubProgress, GitHubUser
from shared.execution import AcquisitionResult
from shared.failures import ApiBudgetExhaustedError
from shared.safety import RunStopReason
from shared.schemas import OpusDecision

from tests.test_github_pipeline import (
    _attach_runtime_state,
    _make_candidate,
    _make_pipeline,
    _make_query,
    GitHubPipeline,
    github_orchestrator,
)

BUDGET_MESSAGE = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to purchase credits."
)


class TestBudgetExhaustionPropagatesFromJudge:
    def test_full_judge_budget_error_not_recorded_as_judgment_failure(self):
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
            side_effect=ApiBudgetExhaustedError("provider credits exhausted"),
        ):
            with pytest.raises(ApiBudgetExhaustedError, match="credits exhausted"):
                asyncio.run(
                    pipeline._process_v2_candidates_batch(
                        [("alice", candidate)], query, progress
                    )
                )

        finish_mock.assert_not_called()
        assert pipeline.stats.get("parse_failures", 0) == 0

    def test_budget_exhaustion_sets_run_error_status_not_judgment_failure(self, tmp_path):
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
        pipeline.stats["candidates_discovered"] = 3

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
                 side_effect=ApiBudgetExhaustedError("provider credits exhausted"),
             ), patch.object(
                 GitHubPipeline,
                 "_search_users",
                 AsyncMock(return_value=(1, ["alice"])),
             ):
            with pytest.raises(ApiBudgetExhaustedError, match="credits exhausted"):
                asyncio.run(pipeline.run(resume=False))

        finish_run_mock.assert_called_once()
        assert finish_run_mock.call_args.kwargs["status"] == "error"
        assert finish_run_mock.call_args.kwargs["stop_reason"] == RunStopReason.API_BUDGET_EXHAUSTED
        assert pipeline.stats.get("parse_failures", 0) == 0


class TestDayCycleStopsOnBudgetExhaustion:
    def test_day_cycle_does_not_start_second_session_after_budget_exhausted(self, monkeypatch):
        import github.session_orchestrator as so

        session_starts: list[int] = []
        sleep_calls: list[float] = []

        async def fake_run_session(_brief_path, _output_dir, resume=False):
            raise ApiBudgetExhaustedError("provider credits exhausted")

        def fake_record_session_start():
            session_starts.append(len(session_starts) + 1)
            return len(session_starts)

        monkeypatch.setattr(so, "_run_session", fake_run_session)
        # Count real starts so a regression (budget error swallowed by the generic
        # handler) terminates at the daily cap and fails, rather than hanging.
        monkeypatch.setattr(so, "get_sessions_today", lambda: len(session_starts))
        monkeypatch.setattr(so, "record_session_start", fake_record_session_start)
        monkeypatch.setattr(so, "record_session_end", lambda *a, **kw: None)

        governor = MagicMock()
        governor.can_start_session.return_value = (True, "")
        governor.start_session = MagicMock()
        monkeypatch.setattr(so, "GitHubGovernor", lambda: governor)

        async def fake_sleep(duration):
            sleep_calls.append(duration)

        monkeypatch.setattr(so.asyncio, "sleep", fake_sleep)

        exit_code = asyncio.run(
            so.run_day_cycle(
                brief_path="/tmp/brief.json",
                output_dir="/tmp/state",
                single_session=False,
                resume=False,
            )
        )

        assert len(session_starts) == 1
        assert sleep_calls == []
        assert exit_code == 1


class TestNonJudgeBudgetSeams:
    def test_enricher_portfolio_extraction_raises_on_budget_exhaustion(self):
        candidate = GitHubCandidate(
            user=GitHubUser(
                username="devuser",
                name="Dev User",
                profile_url="https://github.com/devuser",
            ),
            contact=ContactInfo(),
        )
        enricher = GitHubEnricher(MagicMock())

        with patch(
            "github.enricher.cheap_llm",
            side_effect=RuntimeError(BUDGET_MESSAGE),
        ):
            with pytest.raises(ApiBudgetExhaustedError, match="credit balance is too low"):
                asyncio.run(enricher._extract_portfolio(candidate))

        assert candidate.portfolio_summary == {}

    def test_outreach_does_not_retry_budget_exhaustion(self):
        candidate = GitHubCandidate(
            user=GitHubUser(
                username="devuser",
                name="Dev User",
                profile_url="https://github.com/devuser",
            ),
            contact=ContactInfo(),
        )
        brief = MagicMock()
        brief.role_title = "ML Engineer"
        brief.role_description = "Build training pipelines."
        eval_result = OpusDecision(
            stage="full",
            decision="SAVE",
            path="DIRECT:ml-infra",
            confidence=0.9,
            rationale="Strong ML infra evidence.",
            candidate_name="Dev User",
            profile_url="https://github.com/devuser",
        )

        with patch(
            "github.outreach.opus_llm",
            side_effect=RuntimeError(BUDGET_MESSAGE),
        ) as mock_opus:
            with pytest.raises(ApiBudgetExhaustedError, match="credit balance is too low"):
                asyncio.run(
                    generate_outreach(candidate, brief, eval_result, max_retries=2)
                )

        assert mock_opus.call_count == 1
