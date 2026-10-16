"""GitHub tier routing: full judge and outreach must pass configured model tiers."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import shared.config as config
import shared.judger as judger
from github.outreach import generate_outreach
from github.schemas import ContactInfo, GitHubCandidate, GitHubUser
from linkedin.judgment_templates import FacialResult, FullEvaluationResult
from shared.schemas import OpusDecision


def _make_brief(*, has_v2: bool = True):
    brief = MagicMock()
    brief.has_v2_schema = has_v2
    brief._new_brief = MagicMock() if has_v2 else None
    return brief


def _make_candidate() -> GitHubCandidate:
    return GitHubCandidate(
        user=GitHubUser(
            username="devuser",
            name="Dev User",
            profile_url="https://github.com/devuser",
        ),
        contact=ContactInfo(),
    )


def _make_eval_result() -> OpusDecision:
    return OpusDecision(
        stage="full",
        decision="SAVE",
        path="DIRECT:ml-infra",
        confidence=0.9,
        rationale="Strong ML infra evidence.",
        candidate_name="Dev User",
        profile_url="https://github.com/devuser",
    )


def test_github_full_judge_passes_full_eval_model_name(monkeypatch):
    sentinel = "sentinel-full-eval-model"
    monkeypatch.setattr(config, "FULL_EVAL_MODEL_NAME", sentinel)

    parsed = FullEvaluationResult(
        decision="SAVE",
        match_type="DIRECT",
        capability_area="ml-infra",
        capability_evidence="",
        depth="BUILDER",
        depth_evidence="",
        transferability="N/A",
        transferability_evidence="",
        case_for="",
        case_against="",
        confidence=0.9,
        post_save_modifier="",
        summary="Strong fit.",
        raw_response="",
    )

    with (
        patch(
            "github.judgment_templates.assemble_github_full_evaluation_system",
            return_value="SYSTEM",
        ),
        patch.object(judger, "opus_llm_cached", return_value="RAW") as mock_cached,
        patch(
            "github.judgment_templates.parse_full_evaluation_response",
            return_value=parsed,
        ),
    ):
        judger.github_full_judge("evidence text", _make_brief(has_v2=True))

    _, kwargs = mock_cached.call_args
    assert kwargs.get("model_name") == sentinel


def test_generate_outreach_passes_outreach_model_name(monkeypatch):
    sentinel = "sentinel-outreach-model"
    monkeypatch.setattr(config, "OUTREACH_MODEL_NAME", sentinel)

    brief = MagicMock()
    brief.role_title = "ML Engineer"
    brief.role_description = "Build training pipelines."

    with patch("github.outreach.opus_llm") as mock_opus:
        mock_opus.return_value = {
            "subject_line": "Your RL work",
            "message": "Hi — your repo maps to our role.",
            "repo_referenced": "ml-toolkit",
            "capability_hook": "training pipelines",
        }
        result = asyncio.run(
            generate_outreach(_make_candidate(), brief, _make_eval_result())
        )

    assert result.get("message")
    _, kwargs = mock_opus.call_args
    assert kwargs.get("model_name") == sentinel


def test_github_facial_judge_does_not_pass_model_name():
    with (
        patch(
            "github.judgment_templates.assemble_github_facial_system",
            return_value="SYSTEM",
        ),
        patch.object(judger, "facial_llm", return_value="YES") as mock_facial,
        patch(
            "github.judgment_templates.parse_facial_response",
            return_value=FacialResult("FACIAL_YES", "looks good", "RAW"),
        ),
    ):
        judger.github_facial_judge("portfolio text", _make_brief(has_v2=True))

    _, kwargs = mock_facial.call_args
    assert "model_name" not in kwargs
