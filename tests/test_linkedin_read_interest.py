"""Slice A1 — facial verdict → profile-read interest budget."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.governor import OperatorStopRequested
from shared.schemas import CandidateSnippet, OpusDecision, SearchString


def _make_snippet(
    profile_url: str = "https://www.linkedin.com/talent/profile/test123",
) -> CandidateSnippet:
    return CandidateSnippet(
        name="Test Person",
        headline="Builder",
        current_title="Engineer",
        current_company="Acme",
        location="Somewhere",
        education_snippet="",
        profile_url=profile_url,
        source_string_id=1,
        source_string_name="test",
        page=1,
        result_rank=1,
    )


def _bare_pipeline():
    """Minimal Pipeline shell — only the per-run interest map is needed."""
    from linkedin.orchestrator import Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._profile_read_interest = {}
    return pipeline


def _make_pipeline(output_dir: str):
    """Cheap Pipeline fixture mirroring test_facial_borderline_bias_monitor."""
    with patch("linkedin.orchestrator.load_brief") as mock_brief, \
         patch("linkedin.orchestrator.init_judger"), \
         patch("linkedin.orchestrator.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        mock_brief.return_value = brief

        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline

        return Pipeline(brief_path=str(brief_path), output_dir=output_dir)


def test_facial_yes_stamps_high_interest():
    pipeline = _bare_pipeline()
    snippet = _make_snippet()

    pipeline._stamp_read_interest(snippet, "FACIAL_YES")

    assert pipeline._read_interest_for(snippet) == 0.9


def test_facial_borderline_stamps_low_interest_even_under_binary_posture():
    """The stamp must read the raw verdict, upstream of normalization.

    Binary posture is the only one where the placement is observable: there a
    returned BORDERLINE is normalized into a PARSE_FAILURE, so a stamp taken
    downstream sees a decision the interest map has no entry for and records
    nothing. Under ternary the verdict survives and either placement reads the
    same, which is exactly why the wrong placement would otherwise ship green.
    """
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.brief_obj.has_v2_schema = True
        pipeline.brief_obj.employer_blacklist = []
        pipeline._tightening_prefix = ""
        pipeline._full_evaluate = AsyncMock(return_value=None)
        snippet = _make_snippet()
        borderline = OpusDecision(
            stage="facial",
            decision="FACIAL_BORDERLINE",
            path="none",
            confidence=1.0,
            rationale="ambiguous trajectory",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )

        with patch("linkedin.orchestrator.facial_judge", return_value=borderline), \
             patch("linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED", False):
            asyncio.run(pipeline._evaluate_snippet(snippet))

        assert pipeline._read_interest_for(snippet) == 0.35


def test_interest_map_semantics_for_each_verdict():
    """Mapping semantics only — the batch loop itself is covered by
    ``tests/test_linkedin_page_observability.py``, which drives the real
    ``_review_page_batch``.
    """
    pipeline = _bare_pipeline()

    yes_snippet = _make_snippet("https://www.linkedin.com/talent/profile/yes1")
    borderline_snippet = _make_snippet("https://www.linkedin.com/talent/profile/bord1")
    no_snippet = _make_snippet("https://www.linkedin.com/talent/profile/no1")

    for snippet, decision in (
        (yes_snippet, "FACIAL_YES"),
        (borderline_snippet, "FACIAL_BORDERLINE"),
        (no_snippet, "FACIAL_NO"),
    ):
        pipeline._stamp_read_interest(snippet, decision)

    assert pipeline._read_interest_for(yes_snippet) == 0.9
    assert pipeline._read_interest_for(borderline_snippet) == 0.35
    # FACIAL_NO is never read, so it must not occupy the map at all.
    assert no_snippet.profile_url not in pipeline._profile_read_interest
    assert pipeline._read_interest_for(no_snippet) == 0.5


def test_stamp_records_under_the_profile_url_key():
    """Pin the storage contract, not just that the two methods agree."""
    pipeline = _bare_pipeline()
    snippet = _make_snippet()

    pipeline._stamp_read_interest(snippet, "FACIAL_YES")

    assert pipeline._profile_read_interest == {snippet.profile_url: 0.9}


def test_unstamped_profile_falls_back_to_medium_interest():
    pipeline = _bare_pipeline()
    snippet = _make_snippet()

    assert pipeline._read_interest_for(snippet) == 0.5


def test_full_evaluate_passes_interest_to_acquisition():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _make_snippet()
        pipeline._profile_read_interest[snippet.profile_url] = 0.9
        pipeline._ensure_services = MagicMock()
        pipeline._start_runtime_stage_attempt = MagicMock(return_value=1)
        pipeline._abort_runtime_stage_attempt = MagicMock()
        pipeline._acquisition_service = MagicMock()
        pipeline._acquisition_service.extract_profile_summary = AsyncMock(
            side_effect=OperatorStopRequested()
        )

        with pytest.raises(OperatorStopRequested):
            asyncio.run(pipeline._full_evaluate(snippet))

        pipeline._acquisition_service.extract_profile_summary.assert_awaited_once_with(
            snippet, interest=0.9
        )


def test_open_and_extract_passes_interest_to_acquisition():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _make_snippet()
        pipeline._profile_read_interest[snippet.profile_url] = 0.35
        pipeline._ensure_services = MagicMock()
        pipeline._start_runtime_stage_attempt = MagicMock(return_value=1)
        pipeline._abort_runtime_stage_attempt = MagicMock()
        pipeline._acquisition_service = MagicMock()
        pipeline._acquisition_service.extract_profile_summary = AsyncMock(
            side_effect=OperatorStopRequested()
        )

        with pytest.raises(OperatorStopRequested):
            asyncio.run(
                pipeline._open_and_extract(
                    snippet,
                    search_string=SearchString(id=1, name="test", boolean=""),
                )
            )

        pipeline._acquisition_service.extract_profile_summary.assert_awaited_once_with(
            snippet, interest=0.35
        )
