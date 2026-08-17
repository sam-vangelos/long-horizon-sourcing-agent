"""Telemetry-only coverage for LinkedIn in-page observation accounting."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.contracts import RUN_LOG_EVENTS
from shared.schemas import CandidateSnippet, GlanceResult, OpusDecision, Progress, SearchString
from shared.storage import read_jsonl


NEW_CONTROL_PLANE_EVENTS = {
    "forced_narrow_failed",
    "page_allocator_shadow_checkpoint",
    "page_allocator_shadow_exhaustion",
    "page_allocator_shadow_poison",
    "page_allocator_active_checkpoint",
    "page_allocator_active_exhaustion",
    "page_allocator_active_actuation",
    "page_cap_reached",
    "page_observation_gap",
    "page_render_zero_slots",
    "pagination_exhausted",
    "resume_fastforward_exhausted",
}


def test_control_plane_observability_events_are_registered():
    assert NEW_CONTROL_PLANE_EVENTS <= RUN_LOG_EVENTS


def _make_pipeline(output_dir: str):
    with patch("linkedin.orchestrator.load_brief") as mock_brief, \
         patch("linkedin.orchestrator.init_judger"), \
         patch("linkedin.orchestrator.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = True
        brief.employer_blacklist = []
        brief.permanent_filters = {}
        brief.needs_preflight.return_value = False
        mock_brief.return_value = brief

        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline

        return Pipeline(brief_path=str(brief_path), output_dir=output_dir)


def _snippet(
    name: str,
    *,
    profile_url: str | None = None,
    company: str = "GoodCo",
    rank: int = 1,
) -> CandidateSnippet:
    return CandidateSnippet(
        name=name,
        headline=f"{name} headline",
        current_title="Staff Engineer",
        current_company=company,
        location="New York",
        education_snippet="",
        profile_url=profile_url if profile_url is not None else f"/talent/profile/{name.lower()}",
        source_string_id=7,
        source_string_name="test string",
        page=1,
        result_rank=rank,
    )


def _facial(decision: str, name: str = "Candidate") -> OpusDecision:
    return OpusDecision(
        stage="facial",
        decision=decision,
        path="none",
        confidence=0.9,
        rationale=f"{decision} rationale",
        candidate_name=name,
        profile_url=f"/talent/profile/{name.lower()}",
    )


def _stage_process_string_page(p, *, result_count: int = 80) -> None:
    no_results = MagicMock()
    no_results.is_visible = AsyncMock(return_value=False)
    locator = MagicMock()
    locator.first = no_results

    p.browser.page = MagicMock(
        url="https://www.linkedin.com/talent/hire/test/discover/recruiterSearch"
    )
    p.browser.page.locator.return_value = locator
    p.browser.enter_search_string = AsyncMock()
    p.browser.apply_advanced_search_plan = AsyncMock()
    p.browser.get_results_count_text = AsyncMock(return_value=str(result_count))
    p.browser.get_results_count = AsyncMock(return_value=result_count)
    p.browser.go_to_next_page = AsyncMock(return_value=False)
    p._ensure_browser_healthy = AsyncMock()


def _run_one_batch_page(
    p,
    *,
    snippets: list[CandidateSnippet | None],
    decisions: list[OpusDecision],
    slot_count: int | None = None,
    result_count: int = 80,
) -> SearchString:
    _stage_process_string_page(p, result_count=result_count)
    p.browser.get_card_slot_count = AsyncMock(
        return_value=slot_count if slot_count is not None else len(snippets)
    )
    p.browser.scroll_to_load_all_results = AsyncMock(return_value=0)
    p.browser.get_card_count = AsyncMock(return_value=0)
    p._extract_card_snippet = AsyncMock(side_effect=snippets)
    p._record_runtime_event = MagicMock()

    search_string = SearchString(id=7, name="test string", boolean='"engineer"')
    progress = Progress(
        brief_name="test",
        strings=[search_string],
        current_string_id=7,
        current_page=0,
    )
    progress.save(str(p.progress_path))
    p._progress = progress

    with patch(
        "shared.judger.facial_judge_batch", return_value=decisions
    ), patch("linkedin.orchestrator.config.MAX_PAGES_PER_STRING", 1):
        asyncio.run(p._process_string(search_string, progress))

    return search_string


def test_unidentifiable_cards_are_dropped_before_any_judgment_spend():
    """A card we cannot IDENTIFY is dropped instantly — never judged, never saved.

    Sam's call 2026-07-27, after a live extraction could not readily produce a
    "LinkedIn Member" card to inspect: do not try to reason about an
    unidentifiable candidate, just drop them.

    The load-bearing part is that the test is the identity FRAGMENT, not a
    non-empty string. Every URL below is truthy, so the old
    `if not snippet.profile_url` check let them all through to a facial
    judgment — real model spend — and then the save path, which resolves the
    candidate by exact `/talent/profile/<id>`, could never act on them and
    aborted the run. Dropping costs nothing; discovering it at the save
    boundary costs the run.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        _run_one_batch_page(
            p,
            snippets=[
                _snippet("Anonymous", profile_url="#", rank=1),
                _snippet("PublicOnly", profile_url="https://www.linkedin.com/in/someone", rank=2),
                _snippet("SearchLink", profile_url="/talent/search?keywords=x", rank=3),
                _snippet("Truncated", profile_url="/talent/profile/", rank=4),
                _snippet("Eligible", profile_url="/talent/profile/eligible", rank=5),
            ],
            decisions=[_facial("FACIAL_NO", "Eligible")],
            result_count=80,
        )

        assess_events = [
            event
            for event in read_jsonl(p.log_path)
            if event.get("event") == "linkedin_search_assess"
        ]
        assert assess_events
        observed = assess_events[0]["page_observed"]
        # All four unidentifiable cards dropped; only the identifiable one judged.
        assert observed["skipped_missing_url"] == 4, observed
        assert observed["judged"] == 1, observed
        assert observed["extracted"] == 5, observed


def test_page_observed_flows_to_assess_log_and_runtime_event_payload():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.employer_blacklist = ["BadCo"]
        p._seen_urls.add("/talent/profile/duplicate")

        _run_one_batch_page(
            p,
            snippets=[
                _snippet("Missing", profile_url="", rank=1),
                _snippet("Duplicate", profile_url="/talent/profile/duplicate", rank=2),
                _snippet("Blocked", profile_url="/talent/profile/blocked", company="BadCo", rank=3),
                _snippet("Eligible", profile_url="/talent/profile/eligible", rank=4),
            ],
            decisions=[_facial("FACIAL_NO", "Eligible")],
            result_count=80,
        )

        assess_events = [
            event
            for event in read_jsonl(p.log_path)
            if event.get("event") == "linkedin_search_assess"
        ]
        assert assess_events
        observed = assess_events[0]["page_observed"]
        assert observed == {
            "slots": 4,
            "extracted": 4,
            "judged": 1,
            "errored": 0,
            "skipped_dup": 1,
            "skipped_blacklist": 1,
            "skipped_missing_url": 1,
            "break_reason": "",
            "full_expected": 0,
            "full_settled": 0,
            "priority": 0,
            "standard": 0,
            "outreach": 0,
        }

        runtime_assess = [
            call.kwargs["payload"]
            for call in p._record_runtime_event.call_args_list
            if call.kwargs.get("event_type") == "linkedin_search_assess"
        ]
        assert runtime_assess
        assert runtime_assess[0]["page_observed"] == observed


def test_page_observation_gap_fires_for_dropped_extraction_without_break_reason():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        _run_one_batch_page(
            p,
            snippets=[
                _snippet("One", profile_url="/talent/profile/one", rank=1),
                None,
                _snippet("Three", profile_url="/talent/profile/three", rank=3),
            ],
            decisions=[_facial("FACIAL_NO", "One"), _facial("FACIAL_NO", "Three")],
            slot_count=3,
        )

        gap_events = [
            event
            for event in read_jsonl(p.log_path)
            if event.get("event") == "page_observation_gap"
        ]
        assert len(gap_events) == 1
        assert gap_events[0]["string_id"] == 7
        assert gap_events[0]["page"] == 1
        assert gap_events[0]["slots"] == 3
        assert gap_events[0]["extracted"] == 2


def test_batch_glance_reformulate_does_not_set_break_reason_or_suppress_gap():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._glance_assess = MagicMock(
            return_value=GlanceResult(
                action="reformulate",
                summary="wrong population",
                confidence=0.95,
            )
        )

        with patch("linkedin.orchestrator.config.GLANCE_MIN_SNIPPETS", 1):
            _run_one_batch_page(
                p,
                snippets=[
                    _snippet("One", profile_url="/talent/profile/one", rank=1),
                    None,
                    None,
                ],
                decisions=[_facial("FACIAL_NO", "One")],
                slot_count=3,
            )

        events = read_jsonl(p.log_path)
        gap_events = [
            event for event in events if event.get("event") == "page_observation_gap"
        ]
        assert len(gap_events) == 1
        assess = [
            event
            for event in events
            if event.get("event") == "linkedin_search_assess"
        ][0]
        assert assess["page_observed"]["slots"] == 3
        assert assess["page_observed"]["extracted"] == 1
        assert assess["page_observed"]["break_reason"] == ""


def test_batch_reformulate_glance_reviews_full_page_and_preserves_verdict_telemetry():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        slot_count = 10
        snippets = [
            _snippet(
                f"Eligible {idx}",
                profile_url=f"/talent/profile/eligible-{idx}",
                rank=idx,
            )
            for idx in range(1, slot_count + 1)
        ]
        _stage_process_string_page(p, result_count=80)
        p.browser.get_card_slot_count = AsyncMock(return_value=slot_count)
        p.browser.scroll_to_load_all_results = AsyncMock(return_value=0)
        p.browser.get_card_count = AsyncMock(return_value=0)
        p._extract_card_snippet = AsyncMock(side_effect=snippets)
        p._record_runtime_event = MagicMock()
        p._bias_monitor = None
        p._glance_assess = MagicMock(
            return_value=GlanceResult(
                action="reformulate",
                summary="wrong population dominates",
                confidence=0.94,
                signals={"test": {"noise": True}},
            )
        )
        async def settle_full_review(snippet, *_args, **_kwargs):
            decision = OpusDecision(
                stage="full",
                decision="REJECT",
                path="none",
                confidence=0.9,
                rationale="not enough evidence",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            )
            p._note_page_full_review_settled(
                snippet=snippet,
                decision=decision,
            )
            return decision

        p._full_evaluate = AsyncMock(side_effect=settle_full_review)
        captured_insights = []

        async def _capture_assess(**kwargs):
            captured_insights.append(kwargs["page_insights"])
            return {
                "decision": "stop",
                "rationale": "test stop",
                "page_signal": 0,
                "committed_zero_signal_streak": 0,
            }

        p._assess_string_state = AsyncMock(side_effect=_capture_assess)

        search_string = SearchString(id=7, name="test string", boolean='"engineer"')
        progress = Progress(
            brief_name="test",
            strings=[search_string],
            current_string_id=7,
            current_page=0,
        )
        progress.save(str(p.progress_path))
        p._progress = progress

        facial_decisions = [
            OpusDecision(
                stage="facial",
                decision="FACIAL_YES",
                path="none",
                confidence=0.9,
                rationale="plausible",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            )
            for snippet in snippets
        ]

        with patch("shared.judger.facial_judge_batch", return_value=facial_decisions) as facial_batch, \
             patch("linkedin.orchestrator.config.GLANCE_MIN_SNIPPETS", 8), \
             patch("linkedin.orchestrator.config.FULL_EVAL_PIPELINE_ENABLED", False), \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0):
            asyncio.run(p._process_string(search_string, progress))

        assert p._extract_card_snippet.await_count == slot_count
        observed = p._page_observation()
        assert observed["slots"] == slot_count
        assert observed["extracted"] == slot_count
        assert observed["break_reason"] == ""
        assert [snippet.name for snippet in facial_batch.call_args.args[0]] == [
            snippet.name for snippet in snippets
        ]
        assert [call.args[0].name for call in p._full_evaluate.await_args_list] == [
            snippet.name for snippet in snippets
        ]
        assert [
            event.get("action")
            for event in read_jsonl(p.log_path)
            if event.get("event") == "glance_assess"
        ] == ["reformulate"]
        assert captured_insights
        assert captured_insights[0].glance_action == "reformulate"


def test_batch_glance_sample_excludes_dup_and_blacklist_skips():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.employer_blacklist = ["BlockedCo"]
        duplicate_snippets = [
            _snippet(
                f"Duplicate {idx}",
                profile_url=f"/talent/profile/duplicate-{idx}",
                rank=idx,
            )
            for idx in range(1, 6)
        ]
        for snippet in duplicate_snippets:
            p._seen_urls.add(snippet.profile_url)
        blocked = _snippet(
            "Blocked",
            profile_url="/talent/profile/blocked",
            company="BlockedCo",
            rank=6,
        )
        eligible_snippets = [
            _snippet(
                f"Eligible {idx}",
                profile_url=f"/talent/profile/eligible-{idx}",
                rank=idx + 6,
            )
            for idx in range(1, 9)
        ]
        snippets = duplicate_snippets + [blocked] + eligible_snippets
        p.browser.get_card_slot_count = AsyncMock(return_value=len(snippets))
        p.browser.scroll_to_load_all_results = AsyncMock(return_value=0)
        p.browser.get_card_count = AsyncMock(return_value=0)
        p._extract_card_snippet = AsyncMock(side_effect=snippets)
        p._checkpoint_progress = MagicMock()
        p._preview_skip_pause = AsyncMock()
        p._bias_monitor = None
        p._triage_tightened = False
        p._tightening_prefix = ""
        p._glance_assess = MagicMock(
            return_value=GlanceResult(
                action="proceed",
                summary="eligible pool looks plausible",
                confidence=0.9,
            )
        )

        search_string = SearchString(id=7, name="test string", boolean='"engineer"')
        page_report = MagicMock()
        all_candidates: list[dict] = []
        string_stats = {
            "pages": 1,
            "candidates": 0,
            "duplicates": 0,
            "facial_yes": 0,
            "facial_no": 0,
            "saves": 0,
            "rejects": 0,
        }
        facial_decisions = [
            _facial("FACIAL_NO", snippet.name)
            for snippet in eligible_snippets
        ]

        with patch("shared.judger.facial_judge_batch", return_value=facial_decisions) as facial_batch, \
             patch("linkedin.orchestrator.config.GLANCE_MIN_SNIPPETS", 8), \
             patch("linkedin.orchestrator.config.FULL_EVAL_PIPELINE_ENABLED", False):
            asyncio.run(
                p._review_page_batch(
                    search_string,
                    1,
                    80,
                    page_report,
                    all_candidates,
                    string_stats,
                    None,
                )
            )

        assert p._glance_assess.call_count == 1
        assert [snippet.name for snippet in p._glance_assess.call_args.args[0]] == [
            snippet.name for snippet in eligible_snippets
        ]
        assert [snippet.name for snippet in facial_batch.call_args.args[0]] == [
            snippet.name for snippet in eligible_snippets
        ]


def test_review_page_batch_stamps_read_interest_from_the_raw_verdict(tmp_path):
    """Batch facial path must stamp read-interest from raw verdicts before normalization."""
    p = _make_pipeline(str(tmp_path))
    yes_snippet = _snippet(
        "Yes Candidate",
        profile_url="/talent/profile/read-interest-yes",
        rank=1,
    )
    borderline_snippet = _snippet(
        "Borderline Candidate",
        profile_url="/talent/profile/read-interest-borderline",
        rank=2,
    )
    snippets = [yes_snippet, borderline_snippet]
    p.browser.get_card_slot_count = AsyncMock(return_value=len(snippets))
    p.browser.scroll_to_load_all_results = AsyncMock(return_value=0)
    p.browser.get_card_count = AsyncMock(return_value=0)
    p._extract_card_snippet = AsyncMock(side_effect=snippets)
    p._checkpoint_progress = MagicMock()
    p._preview_skip_pause = AsyncMock()
    p._bias_monitor = None
    p._triage_tightened = False
    p._tightening_prefix = ""
    p._full_evaluate = AsyncMock(
        return_value=OpusDecision(
            stage="full",
            decision="REJECT",
            path="none",
            confidence=0.9,
            rationale="not enough evidence",
            candidate_name="ignored",
            profile_url="ignored",
        )
    )

    search_string = SearchString(id=7, name="test string", boolean='"engineer"')
    page_report = MagicMock()
    all_candidates: list[dict] = []
    string_stats = {
        "pages": 1,
        "candidates": 0,
        "duplicates": 0,
        "facial_yes": 0,
        "facial_no": 0,
        "saves": 0,
        "rejects": 0,
    }
    facial_decisions = [
        OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="none",
            confidence=0.9,
            rationale="plausible",
            candidate_name=yes_snippet.name,
            profile_url=yes_snippet.profile_url,
        ),
        OpusDecision(
            stage="facial",
            decision="FACIAL_BORDERLINE",
            path="none",
            confidence=0.6,
            rationale="ambiguous trajectory",
            candidate_name=borderline_snippet.name,
            profile_url=borderline_snippet.profile_url,
        ),
    ]

    with patch("shared.judger.facial_judge_batch", return_value=facial_decisions), \
         patch("linkedin.orchestrator.config.FULL_EVAL_PIPELINE_ENABLED", False):
        asyncio.run(
            p._review_page_batch(
                search_string,
                1,
                80,
                page_report,
                all_candidates,
                string_stats,
                None,
            )
        )

    # Stamp runs on the raw verdict before normalization, so interest is
    # observable even when BORDERLINE would later be coerced under binary posture.
    assert p._profile_read_interest[yes_snippet.profile_url] == 0.9
    assert p._profile_read_interest[borderline_snippet.profile_url] == 0.35


def test_zero_slot_page_reload_recovery_proceeds_to_page_review():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _stage_process_string_page(p, result_count=1)
        p.browser.page.reload = AsyncMock()
        p.browser.page.wait_for_timeout = AsyncMock()
        p.browser.get_card_slot_count = AsyncMock(side_effect=[0, 1])
        p.browser.scroll_to_load_all_results = AsyncMock(return_value=0)
        p.browser.get_card_count = AsyncMock(return_value=0)
        p._extract_card_snippet = AsyncMock(
            return_value=_snippet("Recovered", profile_url="/talent/profile/recovered")
        )
        p._record_runtime_event = MagicMock()

        search_string = SearchString(id=7, name="test string", boolean='"engineer"')
        state = p._experiment_state_for(search_string)
        state.record_variant_metrics = MagicMock(wraps=state.record_variant_metrics)
        state.note_page_review = MagicMock(wraps=state.note_page_review)
        progress = Progress(
            brief_name="test",
            strings=[search_string],
            current_string_id=7,
            current_page=0,
        )

        with patch(
            "shared.judger.facial_judge_batch",
            return_value=[_facial("FACIAL_NO", "Recovered")],
        ):
            asyncio.run(p._process_string(search_string, progress))

        p.browser.page.reload.assert_awaited_once()
        state.record_variant_metrics.assert_called_once()
        state.note_page_review.assert_called_once()
        zero_slot_events = [
            event
            for event in read_jsonl(p.log_path)
            if event.get("event") == "page_render_zero_slots"
        ]
        assert zero_slot_events == []


def test_page_render_zero_slots_retry_exhaustion_is_loud_not_reviewed():
    from linkedin.orchestrator import PageRenderFailedError

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _stage_process_string_page(p, result_count=80)
        p.browser.page.reload = AsyncMock()
        p.browser.page.wait_for_timeout = AsyncMock()
        p.browser.get_card_slot_count = AsyncMock(side_effect=[0, 0])
        p.browser.scroll_to_load_all_results = AsyncMock(return_value=0)
        p.browser.get_card_count = AsyncMock(return_value=0)
        p._record_runtime_event = MagicMock()

        search_string = SearchString(id=7, name="test string", boolean='"engineer"')
        state = p._experiment_state_for(search_string)
        state.record_variant_metrics = MagicMock(wraps=state.record_variant_metrics)
        state.note_page_review = MagicMock(wraps=state.note_page_review)
        progress = Progress(
            brief_name="test",
            strings=[search_string],
            current_string_id=7,
            current_page=0,
        )

        with pytest.raises(PageRenderFailedError):
            asyncio.run(p._process_string(search_string, progress))

        zero_slot_events = [
            event
            for event in read_jsonl(p.log_path)
            if event.get("event") == "page_render_zero_slots"
        ]
        assert len(zero_slot_events) == 1
        assert zero_slot_events[0]["string_id"] == 7
        assert zero_slot_events[0]["page"] == 1
        assert zero_slot_events[0]["slots"] == 0
        assert zero_slot_events[0]["retry_exhausted"] is True
        p.browser.page.reload.assert_awaited_once()
        state.record_variant_metrics.assert_not_called()
        state.note_page_review.assert_not_called()
