"""CLO-221 S2: the sniper retrieval prior rides the facial USER message.

Unlike the experience prior, this is per search string and therefore cannot
live on cached system surfaces. These tests lock both prefix construction and
the existing ``prompt_prefix`` seam through batch and sequential triage.
"""

from __future__ import annotations

from linkedin.evaluation_loop import EvaluationLoop
from linkedin.evaluation_state import EvaluationRunState

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin.orchestrator import Pipeline
from shared.schemas import OpusDecision, SearchString
from tests.test_linkedin_runtime_state import (
    _make_pipeline,
    _snippet,
    _start_pipeline_runtime,
)


def _prefix_pipeline(*, tightening: str = "", ternary: bool = True) -> EvaluationLoop:
    pipeline = EvaluationLoop(EvaluationRunState())
    pipeline.state.reviewer._tightening_prefix = tightening
    pipeline.state.reviewer._facial_ternary_enabled = lambda: ternary
    return pipeline


def _sniper_string() -> SearchString:
    return SearchString(
        id=1,
        name="precision",
        boolean='"named benchmark"',
        precision_class="sniper",
        evidence_implication="Membership proves named benchmark practice",
    )


@pytest.mark.parametrize(
    ("tightening", "precision_class", "implication"),
    [
        ("", "", ""),
        ("TIGHTEN\n\n", "standard", "Specific practice."),
        ("TIGHTEN\n\n", "recall", "Specific practice."),
        ("TIGHTEN\n\n", "sniper", "  "),
    ],
)
def test_non_sniper_prefix_is_byte_identical_to_tightening(
    monkeypatch, tightening, precision_class, implication
):
    monkeypatch.setattr("shared.config.LINKEDIN_SNIPER_PRIOR_ENABLED", True)
    pipeline = _prefix_pipeline(tightening=tightening)
    search_string = SearchString(
        id=1,
        name="test",
        boolean="test",
        precision_class=precision_class,
        evidence_implication=implication,
    )

    assert pipeline.state.reviewer._facial_prefix_for_string(search_string) == tightening


@pytest.mark.parametrize(
    ("implication", "normalized"),
    [
        ("  Membership proves named benchmark practice  ", "Membership proves named benchmark practice."),
        ("Membership proves named benchmark practice.", "Membership proves named benchmark practice."),
        ("Membership proves named benchmark practice!", "Membership proves named benchmark practice!"),
        ("Membership proves named benchmark practice?", "Membership proves named benchmark practice?"),
    ],
)
def test_sniper_prefix_follows_tightening_and_normalizes_punctuation(
    monkeypatch, implication, normalized
):
    monkeypatch.setattr("shared.config.LINKEDIN_SNIPER_PRIOR_ENABLED", True)
    tightening = "TIGHTENING VERBATIM\n\n"
    pipeline = _prefix_pipeline(tightening=tightening)
    search_string = _sniper_string()
    search_string.evidence_implication = implication

    prefix = pipeline.state.reviewer._facial_prefix_for_string(search_string)

    assert prefix.startswith(tightening)
    assert prefix[len(tightening):] == (
        "RETRIEVAL PRIOR — PRECISION STRING: These candidates were retrieved by a precision search whose required terms only genuine practitioners carry. "
        f"Appearing in these results is itself evidence: {normalized} No other search string re-finds these candidates, so the usual economics of skipping an ambiguous snippet — that another string will surface them — do NOT apply here. "
        "A snippet that is merely sparse or uninformative is NOT a FACIAL_NO: floor it at FACIAL_BORDERLINE so full review can resolve it. FACIAL_NO still applies on affirmative disqualifiers — an entire career in a different line of work, a blacklisted employer, or clearly below the experience floor.\n\n"
    )


def test_binary_posture_and_kill_switch_keep_tightening_only(monkeypatch):
    tightening = "TIGHTENING VERBATIM\n\n"
    search_string = _sniper_string()

    monkeypatch.setattr("shared.config.LINKEDIN_SNIPER_PRIOR_ENABLED", True)
    assert (
        _prefix_pipeline(tightening=tightening, ternary=False)
        .state.reviewer._facial_prefix_for_string(search_string)
        == tightening
    )

    monkeypatch.setattr("shared.config.LINKEDIN_SNIPER_PRIOR_ENABLED", False)
    assert (
        _prefix_pipeline(tightening=tightening)
        .state.reviewer._facial_prefix_for_string(search_string)
        == tightening
    )


def test_batch_facial_threads_the_constructed_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr("shared.config.LINKEDIN_SNIPER_PRIOR_ENABLED", True)
    pipeline = _make_pipeline(str(tmp_path))
    pipeline.state.context.brief_obj.has_v2_schema = True
    pipeline.state.context.brief_obj.employer_blacklist = []
    pipeline.state.context._bias_monitor = None
    pipeline.state.reviewer._tightening_prefix = "TIGHTENING VERBATIM\n\n"
    pipeline.state.reviewer._facial_ternary_enabled = lambda: True
    search_string = _sniper_string()
    progress = _start_pipeline_runtime(pipeline, search_string)
    snippet = _snippet()
    pipeline.state.context.browser.get_card_slot_count = AsyncMock(return_value=1)
    pipeline.state.pages._extract_card_snippet = AsyncMock(return_value=snippet)
    pipeline.state._allocator_service._checkpoint_progress = MagicMock()
    decision = OpusDecision(
        stage="facial",
        decision="FACIAL_NO",
        path="none",
        confidence=1.0,
        rationale="not a match",
        candidate_name=snippet.name,
        profile_url=snippet.profile_url,
    )

    with patch(
        "shared.judger.facial_judge_batch", return_value=[decision]
    ) as facial_judge_batch:
        asyncio.run(
            pipeline.state.pages._review_page_batch(
                search_string,
                1,
                1,
                MagicMock(),
                [],
                pipeline.state.candidates._fresh_string_stats(),
                progress,
            )
        )

    assert facial_judge_batch.call_args.kwargs["prompt_prefix"] == (
        pipeline.state.reviewer._facial_prefix_for_string(search_string)
    )


def test_single_facial_threads_the_constructed_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr("shared.config.LINKEDIN_SNIPER_PRIOR_ENABLED", True)
    pipeline = _make_pipeline(str(tmp_path))
    pipeline.state.context.brief_obj.employer_blacklist = []
    pipeline.state.context._bias_monitor = None
    pipeline.state.reviewer._tightening_prefix = "TIGHTENING VERBATIM\n\n"
    pipeline.state.reviewer._facial_ternary_enabled = lambda: True
    search_string = _sniper_string()
    _start_pipeline_runtime(pipeline, search_string)
    snippet = _snippet()
    pipeline.state.reviewer.attempts._record_runtime_snippet(search_string, snippet)
    pipeline.state.candidates._in_flight_urls.add(snippet.profile_url)
    decision = OpusDecision(
        stage="facial",
        decision="FACIAL_NO",
        path="none",
        confidence=1.0,
        rationale="not a match",
        candidate_name=snippet.name,
        profile_url=snippet.profile_url,
    )

    with patch("linkedin.candidate_review.facial_judge", return_value=decision) as facial_judge:
        asyncio.run(
            pipeline.state.reviewer._evaluate_snippet(
                snippet,
                search_string=search_string,
                string_stats=pipeline.state.candidates._fresh_string_stats(),
            )
        )

    assert facial_judge.call_args.kwargs["prompt_prefix"] == (
        pipeline.state.reviewer._facial_prefix_for_string(search_string)
    )
