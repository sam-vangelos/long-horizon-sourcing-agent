"""Tests for LinkedIn pipeline dedup semantics.

Run with: python -m pytest tests/test_linkedin_pipeline.py -v
"""

import asyncio
import json
import os
import sqlite3
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared import config
from shared.schemas import (
    AdaptationResponse,
    CandidateSnippet,
    ExecutionPlan,
    GlanceResult,
    OpusDecision,
    Progress,
    SearchString,
)
from shared.reconciliation_schemas import RecruiterActivitySnapshot
from shared.governor import (
    GovernorLimitReached,
    OperatorStopRequested,
    SessionExpired,
)
from shared.failures import ApiBudgetExhaustedError, is_api_budget_exhausted_error
from shared.storage import append_jsonl, read_jsonl
from linkedin.adaptation_signal_state import AdaptationGateConfig
from linkedin.page_allocator import AllocatorPolicyError
from linkedin.search_intelligence import (
    LinkedInPageInsights,
    LinkedInSearchVariant,
    bootstrap_experiment_state,
)
from linkedin.search_mutation import SearchMutationResult
from shared.llm_clients import _parse_json_response

# Where a live run actually sits: the brief's OWN Recruiter project view.
# `https://www.linkedin.com/talent/search` is the global view — it names no
# project, so with a project-pinned brief (every `_make_pipeline` brief here
# pins "test-project") it is the F1 "unverified page" condition, not neutral
# scenery: run-start navigates off it and the pre-save boundary refuses from
# it. Tests that mean to exercise that condition set the URL explicitly.
_PROJECT_SEARCH_URL = (
    "https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
)


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


def _make_pipeline(output_dir: str):
    """Create a Pipeline instance with mocked dependencies for unit testing."""
    with patch("linkedin.orchestrator.load_brief") as mock_brief, \
         patch("linkedin.orchestrator.init_judger"), \
         patch("linkedin.orchestrator.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        # A truthy bare-Mock permanent_filters.get("Location") would read as
        # a phantom geography and trip the P3a fail-closed gate; real briefs
        # carry a dict.
        brief.permanent_filters = {}
        # A bare MagicMock returns a truthy Mock from needs_preflight(),
        # which would spuriously trip the resume regime-guard added for
        # the SPL live-run finding. Real non-preflight briefs return False.
        brief.needs_preflight.return_value = False
        mock_brief.return_value = brief

        # Create a dummy brief file
        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline
        p = Pipeline(brief_path=str(brief_path), output_dir=output_dir)
        return p


def _allow_synthetic_run_completion(p) -> None:
    p._run_health_summary = MagicMock(
        return_value={"green_but_useless": False}
    )
    p._enrich_run_snapshot = MagicMock()


# ---------------------------------------------------------------------------
# judgment runtime validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "timeout_name",
    (
        "FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS",
        "FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS",
    ),
)
def test_nonstreaming_judgment_timeout_requires_wall_clock_floor(
    monkeypatch: pytest.MonkeyPatch,
    timeout_name: str,
) -> None:
    from linkedin.orchestrator import Pipeline

    monkeypatch.setenv("CLORIS_SKIP_STARTUP_VALIDATION", "1")
    for name, value in {
        "LINKEDIN_V2_FACIAL_CONTRACT": "tool",
        "LINKEDIN_V2_FULL_CONTRACT": "tool",
        "LINKEDIN_FACIAL_CONCURRENCY_ENABLED": False,
        "LINKEDIN_FACIAL_MAX_CONCURRENCY": 1,
        "FIREWORKS_JUDGMENT_POLICY_ENABLED": True,
        "FIREWORKS_PROMPT_AFFINITY_ENABLED": False,
        "FIREWORKS_JUDGMENT_STREAM_ENABLED": False,
        "FACIAL_MODEL_NAME": "accounts/fireworks/models/test-facial",
        "FULL_EVAL_MODEL_NAME": "accounts/fireworks/models/test-full",
        "FIREWORKS_FACIAL_REASONING_EFFORT": "high",
        "FIREWORKS_FULL_REASONING_EFFORT": "high",
        "FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS": 300.0,
        "FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS": 300.0,
    }.items():
        monkeypatch.setattr(config, name, value)

    monkeypatch.setattr(config, timeout_name, 299.0)
    with pytest.raises(RuntimeError) as exc_info:
        Pipeline._validate_judgment_runtime_configuration()
    assert timeout_name in str(exc_info.value)
    assert "FIREWORKS_JUDGMENT_STREAM_ENABLED" in str(exc_info.value)

    monkeypatch.setattr(config, "FIREWORKS_JUDGMENT_STREAM_ENABLED", True)
    monkeypatch.setattr(config, timeout_name, 90.0)
    Pipeline._validate_judgment_runtime_configuration()


# ---------------------------------------------------------------------------
# _mark_terminal
# ---------------------------------------------------------------------------

def test_mark_terminal_promotes_url():
    """_mark_terminal moves URL from in-flight to seen."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/abc"
        p._in_flight_urls.add(url)
        assert url not in p._seen_urls

        p._mark_terminal(url)

        assert url in p._seen_urls
        assert url not in p._in_flight_urls


def test_funnel_counts_candidate_once_across_search_strings():
    """One profile URL is one funnel observation, even if retrieval resurfaces it."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        first_string = SearchString(id=1, name="first", boolean="first")
        second_string = SearchString(id=2, name="second", boolean="second")
        first_stats = p._fresh_string_stats()
        second_stats = p._fresh_string_stats()
        first = _make_snippet(
            profile_url="/talent/profile/shared-candidate",
            source_string_id=1,
        )
        resurfaced = _make_snippet(
            profile_url=first.profile_url,
            source_string_id=2,
        )

        p._record_facial_funnel_outcome(
            snippet=first,
            decision="FACIAL_YES",
            search_string=first_string,
            string_stats=first_stats,
        )
        p._record_facial_funnel_outcome(
            snippet=resurfaced,
            decision="FACIAL_YES",
            search_string=second_string,
            string_stats=second_stats,
        )

        outreach = OpusDecision(
            stage="full",
            decision="SAVE",
            path="DIRECT:reliable systems",
            confidence=0.84,
            rationale="Strong enough for outreach.",
            candidate_name=first.name,
            profile_url=first.profile_url,
            save_outcome={
                "status": "failed",
                "persisted": False,
                "already_present": False,
                "failure_reason": "save_not_persisted",
            },
        )
        p._record_full_funnel_outcome(
            snippet=first,
            decision=outreach,
            search_string=first_string,
            string_stats=first_stats,
        )
        p._record_full_funnel_outcome(
            snippet=resurfaced,
            decision=outreach,
            search_string=second_string,
            string_stats=second_stats,
        )

        assert p.stats["facial_yes"] == 1
        assert p.stats["full_reviewed"] == 1
        assert p.stats["full_outreach"] == 1
        assert first_stats["facial_yes"] == 1
        assert first_stats["full_outreach"] == 1
        assert second_stats["facial_yes"] == 0
        assert second_stats["full_reviewed"] == 0
        assert second_stats["full_outreach"] == 0


def test_interrupted_two_page_resume_rehydrates_canonical_funnel_and_pending_full():
    """Resume trusts the run chain's attempts, then adds page-two work cumulatively."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(
            id=7,
            name="two-page",
            boolean="ml",
            status="in_progress",
            pages_reviewed=2,
            # Deliberately corrupt checkpoint values.  Canonical attempt truth
            # must replace these during resume instead of preserving the drift.
            facial_yes_count=91,
            facial_borderline_count=92,
            facial_no_count=93,
            full_reviewed_count=94,
            full_outreach_count=95,
            full_review_count=96,
            full_reject_count=97,
            candidates_count=98,
            saves=["stale checkpoint save"],
        )
        progress = Progress(
            brief_name="test",
            strings=[search_string],
            current_string_id=7,
            current_page=2,
        )
        original_run_id, progress = p._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        search_string = progress.strings[0]

        def record_facial(
            slug: str,
            decision_name: str,
            *,
            rank: int,
        ) -> tuple[CandidateSnippet, int]:
            snippet = _make_snippet(
                name=slug.title(),
                profile_url=f"/talent/profile/{slug}",
                source_string_id=search_string.id,
                source_string_name=search_string.name,
                page=1 if rank < 4 else 2,
                result_rank=rank,
            )
            p._runtime_bridge.record_snippet_extracted(
                run_id=original_run_id,
                search_string=search_string,
                snippet=snippet,
            )
            attempt_id = p._runtime_bridge.start_stage_attempt(
                run_id=original_run_id,
                search_string=search_string,
                snippet=snippet,
                stage="facial",
            )
            facial = OpusDecision(
                stage="facial",
                decision=decision_name,
                path="facial",
                confidence=0.8,
                rationale="canonical facial result",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            )
            p._runtime_bridge.finish_stage_success(
                run_id=original_run_id,
                attempt_id=attempt_id,
                stage="facial",
                snippet=snippet,
                decision=facial,
            )
            return snippet, attempt_id

        def record_full(
            snippet: CandidateSnippet,
            decision_name: str,
            outreach_tier: str = "",
            rationale: str = "canonical full result",
        ) -> int:
            attempt_id = p._runtime_bridge.start_stage_attempt(
                run_id=original_run_id,
                search_string=search_string,
                snippet=snippet,
                stage="full",
            )
            full = OpusDecision(
                stage="full",
                decision=decision_name,
                path="DIRECT:runtime truth",
                confidence=0.82,
                rationale=rationale,
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
                outreach_tier=outreach_tier,
                reject_reason=(
                    "CAPABILITY_INSUFFICIENT"
                    if decision_name == "REJECT"
                    else ""
                ),
            )
            p._runtime_bridge.finish_stage_success(
                run_id=original_run_id,
                attempt_id=attempt_id,
                stage="full",
                snippet=snippet,
                decision=full,
            )
            return attempt_id

        pending, _ = record_facial("pending", "FACIAL_YES", rank=1)
        review, _ = record_facial("review", "FACIAL_BORDERLINE", rank=2)
        record_full(review, "REVIEW_INFERRED")
        saved, _ = record_facial("saved", "FACIAL_YES", rank=3)
        saved_attempt_id = record_full(
            saved,
            "SAVE",
            outreach_tier="PRIORITY",
            rationale="structured tier wins [TIER: STANDARD]",
        )
        save_start = p._runtime_bridge.begin_candidate_side_effect(
            run_id=original_run_id,
            search_string=search_string,
            snippet=saved,
            attempt_id=saved_attempt_id,
            effect_type="linkedin_save",
            idempotency_key="save",
            payload={"search_string_id": search_string.id},
        )
        p._runtime_bridge.complete_candidate_side_effect(
            side_effect_id=int(save_start["side_effect"]["id"]),
            status="succeeded",
            payload={"test_mode": True},
        )
        already_present, _ = record_facial(
            "already-present",
            "FACIAL_YES",
            rank=7,
        )
        already_present_attempt_id = record_full(
            already_present,
            "SAVE",
            outreach_tier="STANDARD",
        )
        already_present_start = p._runtime_bridge.begin_candidate_side_effect(
            run_id=original_run_id,
            search_string=search_string,
            snippet=already_present,
            attempt_id=already_present_attempt_id,
            effect_type="linkedin_save",
            idempotency_key="save",
            payload={"search_string_id": search_string.id},
        )
        p._runtime_bridge.complete_candidate_side_effect(
            side_effect_id=int(already_present_start["side_effect"]["id"]),
            status="succeeded",
            payload={"already_present": True},
        )
        rejected, _ = record_facial("rejected", "FACIAL_YES", rank=4)
        record_full(rejected, "REJECT")
        record_facial("facial-no", "FACIAL_NO", rank=5)
        historical, _ = record_facial("historical", "FACIAL_YES", rank=6)
        historical_attempt_id = record_full(
            historical,
            "SAVE",
            rationale="legacy structured omission [TIER: STANDARD]",
        )
        historical_save = p._runtime_bridge.begin_candidate_side_effect(
            run_id=original_run_id,
            search_string=search_string,
            snippet=historical,
            attempt_id=historical_attempt_id,
            effect_type="linkedin_save",
            idempotency_key="save",
            payload={"search_string_id": search_string.id},
        )
        p._runtime_bridge.complete_candidate_side_effect(
            side_effect_id=int(historical_save["side_effect"]["id"]),
            status="failed",
            payload={"failure_reason": "save_not_persisted"},
        )

        # Persist the intentionally wrong work-unit checkpoint after the real
        # attempts so the regression proves attempts, not the cloned payload,
        # are authoritative for the semantic funnel.
        p._runtime_bridge.sync_progress(original_run_id, progress)
        resumed_run_id, resumed = p._runtime_bridge.start_or_resume_run(resume=True)
        p._runtime_run_id = resumed_run_id

        # Recovery may hydrate the same canonical rows more than once; counters
        # must remain assignments, not accumulate on every reload.
        p._hydrate_resume_funnel_from_runtime(resumed)
        p._hydrate_resume_funnel_from_runtime(resumed)

        resumed_string = resumed.strings[0]
        assert resumed_string.pages_reviewed == 2
        assert resumed_string.candidates_count == 7
        assert resumed_string.facial_yes_count == 5
        assert resumed_string.facial_borderline_count == 1
        assert resumed_string.facial_no_count == 1
        assert resumed_string.full_reviewed_count == 5
        assert resumed_string.full_outreach_count == 3
        assert resumed_string.full_review_count == 1
        assert resumed_string.full_reject_count == 1
        assert resumed_string.saves == [saved.name]
        assert p.stats["facial_yes"] == 5
        assert p.stats["facial_borderline"] == 1
        assert p.stats["facial_no"] == 1
        assert p.stats["full_outreach"] == 3
        assert p.stats["full_review"] == 1
        assert p.stats["full_reject"] == 1
        assert p.stats["saved"] == 1
        assert p.stats["already_present"] == 1
        assert p.stats["save_attempts"] == 3
        assert resumed.candidates_saved == 1
        assert p.stats["outreach_tier_counts"] == {
            "PRIORITY": 1,
            "STANDARD": 2,
        }
        assert {
            pending.profile_url,
            historical.profile_url,
        }.issubset(p._resume_pending_full_decisions)
        assert saved.profile_url not in p._resume_pending_full_decisions
        assert already_present.profile_url not in p._resume_pending_full_decisions

        string_stats = p._string_stats_for_processing(
            resumed_string,
            resuming=True,
        )
        assert string_stats["candidates"] == 7
        assert string_stats["facial_yes"] == 5
        assert string_stats["full_reviewed"] == 5
        assert string_stats["saves"] == 1

        # The page-two facial-positive candidate had no successful full
        # attempt. It remains full-review eligible, but the facial call and
        # candidate/facial counters must not run a second time.
        direct_full = OpusDecision(
            stage="full",
            decision="REVIEW_FLAGGED",
            path="DIRECT:resume",
            confidence=0.7,
            rationale="complete the interrupted candidate",
            candidate_name=pending.name,
            profile_url=pending.profile_url,
        )
        p._full_evaluate = AsyncMock(return_value=direct_full)
        with patch(
            "linkedin.orchestrator.facial_judge",
            side_effect=AssertionError("resume must not repeat facial judgment"),
        ):
            result = asyncio.run(
                p._evaluate_snippet(
                    pending,
                    search_string=resumed_string,
                    string_stats=string_stats,
                )
            )
        assert result is direct_full
        p._full_evaluate.assert_awaited_once()
        assert string_stats["candidates"] == 7
        assert string_stats["facial_yes"] == 5
        assert pending.profile_url not in p._resume_pending_full_decisions
        assert pending.profile_url not in p._resume_pending_full_owner_ids

        # New page-two work accumulates on the canonical baseline rather than
        # replacing it with only the post-resume delta.
        new_candidate = _make_snippet(
            name="New Page Two",
            profile_url="/talent/profile/new-page-two",
            source_string_id=7,
            source_string_name="two-page",
            page=2,
            result_rank=7,
        )
        p._record_facial_funnel_outcome(
            snippet=new_candidate,
            decision="FACIAL_BORDERLINE",
            search_string=resumed_string,
            string_stats=string_stats,
        )
        p._record_full_funnel_outcome(
            snippet=new_candidate,
            decision=OpusDecision(
                stage="full",
                decision="REVIEW_INFERRED",
                path="DIRECT:new",
                confidence=0.75,
                rationale="new page-two review",
                candidate_name=new_candidate.name,
                profile_url=new_candidate.profile_url,
            ),
            search_string=resumed_string,
            string_stats=string_stats,
        )
        string_stats["candidates"] += 1
        p._sync_bounded_page_stats_for_checkpoint(resumed_string, string_stats)
        assert resumed_string.candidates_count == 8
        assert resumed_string.facial_yes_count == 5
        assert resumed_string.facial_borderline_count == 2
        assert resumed_string.full_reviewed_count == 6
        assert resumed_string.full_review_count == 2


def test_resume_tier_fallback_never_overrides_present_structured_currency():
    from linkedin.orchestrator import _persisted_outreach_tier

    assert _persisted_outreach_tier(
        {"rationale": "earlier [TIER: PRIORITY] final [TIER: STANDARD]"}
    ) == "STANDARD"
    assert _persisted_outreach_tier(
        {"outreach_tier": "PRIORITY", "rationale": "[TIER: STANDARD]"}
    ) == "PRIORITY"
    assert _persisted_outreach_tier(
        {"outreach_tier": None, "rationale": "[TIER: PRIORITY]"}
    ) == ""


def test_owner_pending_recovery_waits_for_exact_owning_search_surface():
    """Foreign pending reviews stay inert until their own string is active."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        first = SearchString(
            id=1,
            name="first",
            boolean="one",
            status="in_progress",
            pages_reviewed=1,
        )
        second = SearchString(
            id=2,
            name="second",
            boolean="two",
            status="queued",
            pages_reviewed=1,
        )
        progress = Progress(
            brief_name="test",
            strings=[first, second],
            current_string_id=first.id,
            current_page=2,
        )
        first_pending = _make_snippet(
            name="First Pending",
            profile_url="/talent/profile/first-pending",
            source_string_id=first.id,
            source_string_name=first.name,
            page=1,
        )
        second_pending = _make_snippet(
            name="Second Pending",
            profile_url="/talent/profile/second-pending",
            source_string_id=second.id,
            source_string_name=second.name,
            page=1,
        )
        p._resume_pending_full_decisions = {
            first_pending.profile_url: "FACIAL_YES",
            second_pending.profile_url: "FACIAL_BORDERLINE",
        }
        p._resume_pending_full_snippets = {
            first_pending.profile_url: first_pending,
            second_pending.profile_url: second_pending,
        }
        p._resume_pending_full_owner_ids = {
            first_pending.profile_url: first.id,
            second_pending.profile_url: second.id,
        }
        p.browser.find_result_slot_by_profile_url = AsyncMock(return_value=3)
        p._checkpoint_progress = MagicMock()
        observed: list[int] = []

        async def settle_pending(**kwargs):
            snippet = kwargs["snippets"][0]
            observed.append(snippet.source_string_id)
            p._resume_pending_full_decisions.pop(snippet.profile_url)
            p._resume_pending_full_snippets.pop(snippet.profile_url)
            return False

        p._process_resumed_pending_full_evaluations = AsyncMock(
            side_effect=settle_pending
        )

        asyncio.run(
            p._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=first,
                first_incomplete_page=2,
                string_stats=p._fresh_string_stats(),
            )
        )
        assert observed == [first.id]
        assert second_pending.profile_url in p._resume_pending_full_decisions
        p.browser.find_result_slot_by_profile_url.assert_awaited_once_with(
            first_pending.profile_url
        )

        asyncio.run(
            p._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=second,
                first_incomplete_page=2,
                string_stats=p._fresh_string_stats(),
            )
        )
        assert observed == [first.id, second.id]
        assert not p._resume_pending_full_decisions


def test_owner_pending_recovery_aborts_resumably_after_retryable_failure():
    """It may try later owner-local candidates, but it cannot advance the owner."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=7,
            name="owner",
            boolean="ml",
            status="done",
            pages_reviewed=2,
            candidates_count=1,
            facial_yes_count=1,
        )
        progress = Progress(brief_name="test", strings=[owner])
        pending = _make_snippet(
            name="Retry Me",
            profile_url="/talent/profile/retry-me",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
        )
        later = _make_snippet(
            name="Settle Me",
            profile_url="/talent/profile/settle-me",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
        )
        p._resume_pending_full_decisions = {
            pending.profile_url: "FACIAL_YES",
            later.profile_url: "FACIAL_YES",
        }
        p._resume_pending_full_snippets = {
            pending.profile_url: pending,
            later.profile_url: later,
        }
        p._resume_pending_full_owner_ids = {
            pending.profile_url: owner.id,
            later.profile_url: owner.id,
        }
        p.browser.find_result_slot_by_profile_url = AsyncMock(return_value=0)
        p._candidate_funnel_counted = {pending.profile_url}
        p._facial_funnel_counted = {pending.profile_url}
        p._checkpoint_progress = MagicMock()
        attempted = []

        failure = OpusDecision(
            stage="full",
            decision="JUDGMENT_FAILURE",
            path="none",
            confidence=0.0,
            rationale="retryable provider error",
            candidate_name=pending.name,
            profile_url=pending.profile_url,
        )

        async def return_failure_then_settle(**kwargs):
            snippet = kwargs["snippets"][0]
            attempted.append(snippet)
            if snippet is later:
                p._resume_pending_full_decisions.pop(later.profile_url)
                p._resume_pending_full_snippets.pop(later.profile_url)
                return False
            p._apply_pipelined_full_eval_page_outcome(
                decision=failure,
                snippet=snippet,
                page_num=kwargs["page_num"],
                all_candidates=kwargs["all_candidates"],
                string_stats=kwargs["string_stats"],
                search_string=kwargs["search_string"],
            )
            return False

        p._process_resumed_pending_full_evaluations = AsyncMock(
            side_effect=return_failure_then_settle
        )

        with pytest.raises(
            RuntimeError,
            match="remains unsettled for active owner",
        ):
            asyncio.run(
                p._recover_owner_pending_full_evaluations(
                    progress=progress,
                    search_string=owner,
                    first_incomplete_page=2,
                    string_stats=p._string_stats_for_processing(
                        owner,
                        resuming=True,
                    ),
                )
            )

        assert attempted == [pending, later]
        assert owner.status == "in_progress"
        assert pending.profile_url in p._resume_pending_full_decisions
        assert later.profile_url not in p._resume_pending_full_decisions
        assert owner.candidates_count == 1
        assert owner.facial_yes_count == 1
        assert owner.full_reviewed_count == 0


@pytest.mark.parametrize(
    ("owner_id", "message"),
    [
        (None, "canonical pending full review lacks owning string"),
        (99, "canonical pending full review owner is absent from the queue"),
    ],
)
def test_owner_pending_recovery_fails_closed_when_pending_owner_is_missing(
    owner_id,
    message,
):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        pending = _make_snippet(
            profile_url="/talent/profile/missing-owner",
            source_string_id=99,
        )
        progress = Progress(
            brief_name="test",
            strings=[SearchString(id=7, name="other", boolean="ml", status="done")],
        )
        p._resume_pending_full_decisions = {pending.profile_url: "FACIAL_YES"}
        p._resume_pending_full_snippets = {pending.profile_url: pending}
        p._resume_pending_full_owner_ids = {pending.profile_url: owner_id}
        p._checkpoint_progress = MagicMock()
        p._process_resumed_pending_full_evaluations = AsyncMock()

        with pytest.raises(
            RuntimeError,
            match=message,
        ):
            asyncio.run(
                p._recover_owner_pending_full_evaluations(
                    progress=progress,
                    search_string=progress.strings[0],
                    first_incomplete_page=2,
                    string_stats=p._fresh_string_stats(),
                )
            )

        p._process_resumed_pending_full_evaluations.assert_not_awaited()
        p._checkpoint_progress.assert_called_once_with(
            progress,
            search_string=progress.strings[0],
        )


def test_page_local_pending_review_requires_the_active_string_owner():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        current_page_snippet = _make_snippet(
            profile_url="/talent/profile/shared-profile",
            source_string_id=1,
        )
        p._resume_pending_full_decisions = {
            current_page_snippet.profile_url: "FACIAL_YES",
        }
        p._resume_pending_full_owner_ids = {
            current_page_snippet.profile_url: 2,
        }

        assert p._resume_pending_full_decision(current_page_snippet) is None

        current_page_snippet.source_string_id = 2
        assert (
            p._resume_pending_full_decision(current_page_snippet)
            == "FACIAL_YES"
        )


def test_remaining_queue_count_uses_done_skipped_only_terminal_rule():
    from linkedin.orchestrator import Pipeline

    progress = Progress(
        brief_name="test",
        strings=[
            SearchString(id=1, name="current", boolean="a", status="in_progress"),
            SearchString(id=2, name="queued", boolean="b", status="queued"),
            SearchString(id=3, name="active", boolean="c", status="in_progress"),
            SearchString(id=4, name="legacy", boolean="d", status="error"),
            SearchString(id=5, name="unknown", boolean="e", status="mystery"),
            SearchString(id=6, name="done", boolean="f", status="done"),
            SearchString(id=7, name="skipped", boolean="g", status="skipped"),
        ],
    )

    assert (
        Pipeline._remaining_queued_strings(progress, current_string_id=1)
        == 4
    )


def test_owner_pending_recovery_navigates_to_the_stored_older_page():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=7,
            name="owner",
            boolean="ml",
            status="in_progress",
            pages_reviewed=2,
            result_count=100,
        )
        progress = Progress(
            brief_name="test",
            strings=[owner],
            current_string_id=owner.id,
            current_page=3,
        )
        pending = _make_snippet(
            name="Page Two",
            profile_url="/talent/profile/page-two",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=2,
        )
        p._resume_pending_full_decisions = {
            pending.profile_url: "FACIAL_YES",
        }
        p._resume_pending_full_snippets = {
            pending.profile_url: pending,
        }
        p._resume_pending_full_owner_ids = {
            pending.profile_url: owner.id,
        }
        p._go_to_next_page_with_transient_retry = AsyncMock(
            return_value=(True, False)
        )
        p.browser.find_result_slot_by_profile_url = AsyncMock(return_value=4)
        p._checkpoint_progress = MagicMock()

        async def settle(**_kwargs):
            p._resume_pending_full_decisions.pop(pending.profile_url)
            p._resume_pending_full_snippets.pop(pending.profile_url)
            p._resume_pending_full_owner_ids.pop(pending.profile_url)
            return False

        p._process_resumed_pending_full_evaluations = AsyncMock(
            side_effect=settle
        )

        rendered_page = asyncio.run(
            p._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=owner,
                first_incomplete_page=3,
                string_stats=p._fresh_string_stats(),
            )
        )

        assert rendered_page == 2
        p._go_to_next_page_with_transient_retry.assert_awaited_once_with(
            result_count=100,
            page_num=1,
        )
        p.browser.find_result_slot_by_profile_url.assert_awaited_once_with(
            pending.profile_url
        )


def test_owner_pending_recovery_fails_closed_on_conflicting_owner_metadata():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(id=7, name="owner", boolean="ml")
        pending = _make_snippet(
            profile_url="/talent/profile/conflicting-owner",
            source_string_id=8,
        )
        progress = Progress(brief_name="test", strings=[owner])
        p._resume_pending_full_decisions = {
            pending.profile_url: "FACIAL_YES",
        }
        p._resume_pending_full_snippets = {
            pending.profile_url: pending,
        }
        p._resume_pending_full_owner_ids = {
            pending.profile_url: owner.id,
        }
        p._checkpoint_progress = MagicMock()

        with pytest.raises(
            RuntimeError,
            match="conflicting string ownership",
        ):
            asyncio.run(
                p._recover_owner_pending_full_evaluations(
                    progress=progress,
                    search_string=owner,
                    first_incomplete_page=2,
                    string_stats=p._fresh_string_stats(),
                )
            )

        assert owner.status == "in_progress"
        p._checkpoint_progress.assert_called_once_with(
            progress,
            search_string=owner,
        )


def test_resumed_pending_full_serial_recovers_panel_and_drains_queue():
    """A recovered panel-close failure cannot drop later pending full reviews."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(id=7, name="resume", boolean="ml")
        first = _make_snippet(
            name="First",
            profile_url="/talent/profile/resume-first",
            source_string_id=7,
        )
        second = _make_snippet(
            name="Second",
            profile_url="/talent/profile/resume-second",
            source_string_id=7,
        )
        first_decision = OpusDecision(
            stage="full",
            decision="REJECT",
            path="NONE",
            confidence=0.8,
            rationale="first",
            candidate_name=first.name,
            profile_url=first.profile_url,
        )
        first_decision._panel_stuck = True
        second_decision = OpusDecision(
            stage="full",
            decision="REVIEW_INFERRED",
            path="DIRECT:test",
            confidence=0.8,
            rationale="second",
            candidate_name=second.name,
            profile_url=second.profile_url,
        )
        p._full_evaluate = AsyncMock(side_effect=[first_decision, second_decision])
        p._apply_pipelined_full_eval_page_outcome = MagicMock()
        p._checkpoint_progress = MagicMock()

        async def recover(**kwargs):
            kwargs["decision"]._panel_stuck = False

        p._recover_stuck_profile_panel = AsyncMock(side_effect=recover)
        with patch(
            "linkedin.orchestrator.config.FULL_EVAL_PIPELINE_ENABLED",
            False,
        ):
            panel_stuck = asyncio.run(
                p._process_resumed_pending_full_evaluations(
                    snippets=[first, second],
                    page_report=None,
                    search_string=search_string,
                    all_candidates=[],
                    string_stats=p._fresh_string_stats(),
                    progress=None,
                    page_num=2,
                )
            )

        assert panel_stuck is False
        assert p._full_evaluate.await_count == 2
        p._recover_stuck_profile_panel.assert_awaited_once()


def test_panel_recovery_events_reserve_panel_stuck_for_exhaustion(capsys):
    """Transient lifecycle events must not inflate legacy stuck-panel counts."""

    from linkedin.orchestrator import PanelRecoveryError

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.go_back_to_results = AsyncMock(
            side_effect=[RuntimeError("close missed"), None]
        )
        with patch("linkedin.orchestrator.asyncio.sleep", new=AsyncMock()):
            asyncio.run(
                p._recover_stuck_profile_panel(
                    candidate_name="Recovered Candidate",
                    page_num=2,
                )
            )

        recovered_events = [row.get("event") for row in read_jsonl(p.log_path)]
        assert "panel_recovery_started" in recovered_events
        assert "panel_recovered" in recovered_events
        assert "panel_stuck" not in recovered_events

        p.browser.go_back_to_results = AsyncMock(
            side_effect=RuntimeError("still stuck")
        )
        with patch("linkedin.orchestrator.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(PanelRecoveryError):
                asyncio.run(
                    p._recover_stuck_profile_panel(
                        candidate_name="Exhausted Candidate",
                        page_num=3,
                    )
                )

        failed_events = [
            row
            for row in read_jsonl(p.log_path)
            if row.get("event") == "panel_stuck"
        ]
        assert len(failed_events) == 1
        assert failed_events[0]["recovery_status"] == "failed"
        assert "run handler will checkpoint" in capsys.readouterr().out


def test_checkpoint_progress_persists_mid_page_state():
    """Mid-page checkpoint should persist the current page for resume."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.stats["saved"] = 3
        p.stats["rejected"] = 2

        search_string = SearchString(id=58, name="test", boolean="foo", status="in_progress")
        progress = Progress(brief_name="test", strings=[search_string], current_string_id=58, current_page=1)

        p._checkpoint_progress(progress, search_string=search_string, page_num=2)

        saved = json.loads(Path(td, "progress.json").read_text())
        assert saved["current_string_id"] == 58
        assert saved["current_page"] == 2
        assert saved["candidates_saved"] == 3
        assert saved["candidates_rejected"] == 2
        assert saved["strings"][0]["pages_reviewed"] == 2


def test_checkpoint_progress_does_not_move_page_backwards():
    """Checkpointing an earlier page should not regress resume state."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(
            id=58,
            name="test",
            boolean="foo",
            status="in_progress",
            pages_reviewed=3,
        )
        progress = Progress(brief_name="test", strings=[search_string], current_string_id=58, current_page=3)

        p._checkpoint_progress(progress, search_string=search_string, page_num=2)

        saved = json.loads(Path(td, "progress.json").read_text())
        assert saved["current_page"] == 2
        assert saved["strings"][0]["pages_reviewed"] == 3


# ---------------------------------------------------------------------------
# Dedup checks
# ---------------------------------------------------------------------------

def test_in_flight_blocks_dedup():
    """URL in _in_flight_urls should block at dedup check."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/inflight"
        p._in_flight_urls.add(url)

        # Should be blocked
        assert url in p._in_flight_urls
        assert url not in p._seen_urls
        # Both sets checked together
        assert url in p._seen_urls or url in p._in_flight_urls


def test_seen_blocks_dedup():
    """URL in _seen_urls should block at dedup check."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/seen"
        p._seen_urls.add(url)

        assert url in p._seen_urls or url in p._in_flight_urls


# ---------------------------------------------------------------------------
# Dedup source: history only, not snippets.jsonl
# ---------------------------------------------------------------------------

def test_snippets_jsonl_not_dedup_source():
    """_seen_urls starts empty, NOT populated from snippets.jsonl."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/from-snippets"

        # Write a snippet to snippets.jsonl
        snippet = _make_snippet(profile_url=url)
        append_jsonl(p.snippets_path, snippet.to_dict())

        # Reset dedup state as init paths do
        p._seen_urls = set()
        p._in_flight_urls = set()
        p._prior_outcomes = {}
        p._load_candidate_history()

        # URL should NOT be in _seen_urls (no history entry)
        assert url not in p._seen_urls


def test_crash_after_snippet_retries_on_resume():
    """URL in snippets.jsonl but NOT in history → NOT in _seen_urls on reload."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/crashed"

        # Simulate: snippet extracted but crash before history write
        append_jsonl(p.snippets_path, _make_snippet(profile_url=url).to_dict())
        # No history entry for this URL

        # Simulate resume reload
        p._seen_urls = set()
        p._in_flight_urls = set()
        p._prior_outcomes = {}
        p._load_candidate_history()

        assert url not in p._seen_urls
        assert url not in p._in_flight_urls


def test_terminal_candidate_skipped_on_resume():
    """URL in history with REJECT → IS in _seen_urls on reload."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/rejected"

        # Write history entry
        append_jsonl(p.history_path, {
            "profile_url": url,
            "candidate_name": "Rejected Person",
            "outcome": "REJECT",
            "confidence": 0.9,
            "timestamp": "2026-01-01T00:00:00+00:00",
        })

        # Simulate resume reload
        p._seen_urls = set()
        p._in_flight_urls = set()
        p._prior_outcomes = {}
        p._load_candidate_history()

        assert url in p._seen_urls
        assert p._prior_outcomes[url] == "REJECT"


# ---------------------------------------------------------------------------
# Non-terminal outcomes
# ---------------------------------------------------------------------------

def test_parse_failure_not_terminal():
    """PARSE_FAILURE → URL NOT in _seen_urls, removed from _in_flight_urls."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/parse-fail"
        p._in_flight_urls.add(url)

        # Simulate what the code does on PARSE_FAILURE
        p._in_flight_urls.discard(url)

        assert url not in p._seen_urls
        assert url not in p._in_flight_urls


def test_judgment_failure_not_terminal():
    """JUDGMENT_FAILURE at full stage → not promoted, allows retry."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/judgment-fail"

        # Simulate: facial passed (terminal), then full judgment fails
        p._in_flight_urls.add(url)
        # Facial terminal marking happened
        p._mark_terminal(url)
        p._prior_outcomes[url] = "FACIAL_YES"

        # Now full judgment failure: URL is in _seen_urls from facial
        # but _prior_outcomes is FACIAL_YES → FACIAL_YES recovery path handles it
        assert url in p._seen_urls
        assert p._prior_outcomes[url] == "FACIAL_YES"


def test_activity_saturation_is_context_not_a_full_review_gate():
    """Recruiter activity may inform novelty, but cannot drop a facial YES."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._bias_monitor = None
        snippet = _make_snippet(
            headline="Generic Analyst",
            recruiter_activity=RecruiterActivitySnapshot(message_count=9, project_count=3, view_count=3),
            novelty_pressure="high",
        )
        facial = OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="none",
            confidence=None,
            rationale="Plausible but limited evidence from preview.",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        full = OpusDecision(
            stage="full",
            decision="REJECT",
            path="none",
            confidence=0.8,
            rationale="Reviewed in full.",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        p._full_evaluate = AsyncMock(return_value=full)

        with patch("linkedin.orchestrator.facial_judge", return_value=facial):
            returned = asyncio.run(p._evaluate_snippet(snippet))

        assert returned is full
        p._full_evaluate.assert_awaited_once()
        assert p.stats["activity_saturated_preview_skips"] == 0


def test_build_run_report_snapshot_includes_activity_metrics():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.stats.update(
            {
                "snippets_extracted": 10,
                "facial_yes": 4,
                "facial_no": 6,
                "saved": 2,
                "rejected": 1,
                "high_pressure_candidates_seen": 3,
                "activity_saturated_preview_skips": 2,
                "high_fit_low_novelty_saves": 1,
            }
        )
        progress = Progress(
            brief_name="test",
            strings=[SearchString(id=1, name="test", boolean="foo", status="done", pages_reviewed=1)],
        )

        snapshot = p._build_run_report_snapshot(progress)

        metrics = snapshot["metrics_summary"]
        assert metrics["high_pressure_candidates_seen"] == 3
        assert metrics["activity_saturated_preview_skips"] == 2
        assert metrics["high_fit_low_novelty_saves"] == 1


# ---------------------------------------------------------------------------
# P4.2/P4.3/P4.5 — cost summary, run health, adaptation ROI
# ---------------------------------------------------------------------------


def test_build_run_report_snapshot_defaults_cost_health_and_adaptation_roi_when_no_data():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(
            brief_name="test",
            strings=[SearchString(id=1, name="test", boolean="foo", status="done")],
        )

        snapshot = p._build_run_report_snapshot(progress)

        metrics = snapshot["metrics_summary"]
        assert metrics["cost_summary"] == {"status": "no_cost_data"}
        assert metrics["run_health"] == {"status": "no_runtime_state"}
        assert metrics["adaptation_roi"] == {"status": "no_adaptation_events"}


def test_build_run_report_snapshot_flags_degraded_run_health_and_logs_run_degraded_event():
    from shared.storage import read_jsonl

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.role_title = "Test Role"
        p.brief_obj.linkedin_project = "Test Role"
        p.brief_obj.linkedin_project_id = "test-project"
        p.brief_obj.raw = {"version": "1.0"}

        # A run with an OK-looking event trail but zero saves recorded in
        # the runtime store — green_but_useless.
        run_id = p._runtime_state.start_run(
            source="linkedin",
            brief_id="test",
            output_dir=td,
            mode="fresh",
            resume_state={"brief_name": "test"},
        )
        p._runtime_state.record_event(run_id=run_id, event_type="pipeline_start")
        p._runtime_state.record_event(run_id=run_id, event_type="pipeline_end")
        p._runtime_state.finish_run(run_id, "completed")
        p._runtime_run_id = run_id

        progress = Progress(
            brief_name="test",
            strings=[SearchString(id=1, name="test", boolean="foo", status="done")],
        )

        with patch("shared.llm_clients.opus_llm", return_value=_sample_report_analysis()):
            p._generate_run_report(progress)

        events = read_jsonl(p.log_path)
        degraded_events = [e for e in events if e.get("event") == "run_degraded"]
        assert len(degraded_events) == 1
        assert degraded_events[0]["run_id"] == run_id
        assert "green_but_useless" in degraded_events[0]["reasons"]

        report_json = json.loads(Path(td, "run-report.json").read_text())
        run_health = report_json["metrics_summary"]["run_health"]
        assert run_health["status"] == "ok"
        assert run_health["degraded"] is True


def test_adaptation_roi_summary_computes_synthetic_two_block_comparison():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        append_jsonl(
            p.log_path,
            {
                "event": "block_adaptation",
                "block": "Block A",
                "report": {},
                "inserted_string_ids": [10, 11],
                "displaced_string_ids": [5, 6],
            },
        )
        append_jsonl(
            p.log_path,
            {
                "event": "block_adaptation",
                "block": "Block B",
                "report": {},
                "inserted_string_ids": [20],
                "displaced_string_ids": [],
            },
        )

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=10, name="ins1", boolean="a", status="done", saves=["A", "B"]),
                SearchString(id=11, name="ins2", boolean="b", status="done", saves=["C"]),
                SearchString(id=5, name="disp1", boolean="c", status="skipped"),
                SearchString(id=6, name="disp2", boolean="d", status="skipped"),
                SearchString(id=20, name="ins3", boolean="e", status="done", saves=["D"]),
            ],
        )

        roi = p._adaptation_roi_summary(progress)

        assert roi["status"] == "ok"
        assert len(roi["events"]) == 2

        block_a = next(e for e in roi["events"] if e["block"] == "Block A")
        assert block_a["inserted_total_saves"] == 3
        assert block_a["inserted_mean_saves"] == 1.5
        assert block_a["displaced_total_saves"] == 0
        assert block_a["displaced_mean_saves"] == 0.0
        assert block_a["displaced_count"] == 2

        block_b = next(e for e in roi["events"] if e["block"] == "Block B")
        assert block_b["inserted_total_saves"] == 1
        assert block_b["displaced_count"] == 0
        assert block_b["displaced_mean_saves"] is None

        assert roi["total_inserted_saves"] == 4
        assert roi["total_displaced_saves"] == 0
        assert roi["net_saves_gained"] == 4


# ---------------------------------------------------------------------------
# GLM-5.2 shadow-judge report aggregation
# (linkedin.orchestrator.Pipeline._shadow_facial_summary)
# ---------------------------------------------------------------------------


def test_shadow_facial_summary_is_absent_when_zero_comparisons():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        assert p._shadow_facial_summary() is None


def test_shadow_facial_summary_aggregates_synthetic_mixed_events():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        # Single-comparison event: agrees.
        append_jsonl(
            p.log_path,
            {
                "event": "facial_shadow_comparison",
                "batch": False,
                "candidate_count": 1,
                "primary_decision": "FACIAL_YES",
                "shadow_decision": "FACIAL_YES",
                "agrees": True,
                "shadow_parse_failed": False,
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 100.0,
                "shadow_error": None,
            },
        )
        # Single-comparison event: disagrees.
        append_jsonl(
            p.log_path,
            {
                "event": "facial_shadow_comparison",
                "batch": False,
                "candidate_count": 1,
                "primary_decision": "FACIAL_NO",
                "shadow_decision": "FACIAL_YES",
                "agrees": False,
                "shadow_parse_failed": False,
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 200.0,
                "shadow_error": None,
            },
        )
        # Single-comparison event: shadow parse failure (not comparable).
        append_jsonl(
            p.log_path,
            {
                "event": "facial_shadow_comparison",
                "batch": False,
                "candidate_count": 1,
                "primary_decision": "FACIAL_YES",
                "shadow_decision": "PARSE_FAILURE",
                "agrees": None,
                "shadow_parse_failed": True,
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 300.0,
                "shadow_error": None,
            },
        )
        # Single-comparison event: shadow errored entirely (excluded from
        # parse-failure-rate and yes-rate denominators, still counted in
        # comparisons and primary_yes_rate).
        append_jsonl(
            p.log_path,
            {
                "event": "facial_shadow_comparison",
                "batch": False,
                "candidate_count": 1,
                "primary_decision": "FACIAL_YES",
                "shadow_decision": None,
                "agrees": None,
                "shadow_parse_failed": False,
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 50.0,
                "shadow_error": "timeout",
            },
        )
        # Batch event covering two more candidates: one agrees, one disagrees.
        append_jsonl(
            p.log_path,
            {
                "event": "facial_shadow_comparison",
                "batch": True,
                "candidate_count": 2,
                "primary_decisions": ["FACIAL_NO", "FACIAL_YES"],
                "shadow_decisions": ["FACIAL_NO", "FACIAL_NO"],
                "agrees": [True, False],
                "shadow_parse_failed": [False, False],
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 400.0,
                "shadow_error": None,
            },
        )
        # A non-shadow event on the same log must be ignored.
        append_jsonl(p.log_path, {"event": "facial_error", "error": "unrelated"})

        summary = p._shadow_facial_summary()

    assert summary is not None
    assert summary["model"] == "accounts/fireworks/models/glm-5p2"
    # 1 + 1 + 1 + 1 + 2 = 6 total candidate-level comparisons.
    assert summary["comparisons"] == 6
    # Comparable (agrees is not None): the 2 singles that agreed/disagreed
    # (True, False) + the 2 batch entries (True, False) = 4 comparable,
    # 2 agreements -> 0.5.
    assert summary["agreement_rate"] == 0.5
    # Responded (excludes the 1 shadow-error candidate) = 5; 1 parse failure -> 0.2.
    assert summary["shadow_parse_failure_rate"] == 0.2
    # Primary YES: FACIAL_YES appears for candidates 1, 3, 4, and batch[1] = 4 of 6.
    assert summary["primary_yes_rate"] == round(4 / 6, 4)
    # Shadow valid (non-parse-failed, non-errored) = 4 (2 singles + 2 batch);
    # shadow YES among those: single #1 (YES) + batch[0]? batch shadow_decisions
    # are ["FACIAL_NO", "FACIAL_NO"] -> 0 there; single #2 shadow FACIAL_YES ->
    # 2 shadow YES out of 4 valid -> 0.5.
    assert summary["shadow_yes_rate"] == 0.5
    assert summary["mean_latency_ms"] == round((100 + 200 + 300 + 50 + 400) / 5, 1)


def test_shadow_facial_summary_omitted_from_run_report_snapshot_metrics_when_absent():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(brief_name="test", strings=[])
        # Ported with the A1 slice-1 extraction: _build_run_report_snapshot moved to
        # RunReportService, which calls its OWN methods — Pipeline-level patches no
        # longer intercept them (Wave 6 boundary review P1).
        svc = p._run_report_service
        with patch.object(svc, "_cost_summary_for_report", return_value={"status": "no_cost_data"}), \
             patch.object(svc, "_run_health_summary", return_value={"status": "no_runtime_state"}), \
             patch.object(svc, "_search_intelligence_aggregate", return_value={}), \
             patch.object(svc, "_load_profile_index_for_adaptation", return_value={}), \
             patch.object(svc, "_load_run_report_decisions", return_value=[]), \
             patch.object(svc, "_bias_summary_for_report", return_value=""):
            snapshot = p._build_run_report_snapshot(progress)

    assert "shadow_facial" not in snapshot["metrics_summary"]


# ---------------------------------------------------------------------------
# GLM-5.2 shadow-judge report aggregation — FULL-EVAL sibling
# (linkedin.orchestrator.Pipeline._shadow_full_summary)
# ---------------------------------------------------------------------------


def test_shadow_full_summary_is_absent_when_zero_comparisons():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        assert p._shadow_full_summary() is None


def test_shadow_full_summary_aggregates_synthetic_mixed_events():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        # SAVE vs INFERENTIAL_SAVE — different raw decisions, same save
        # axis -> agrees.
        append_jsonl(
            p.log_path,
            {
                "event": "full_shadow_comparison",
                "primary_decision": "SAVE",
                "shadow_decision": "INFERENTIAL_SAVE",
                "agrees": True,
                "shadow_parse_failed": False,
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 500.0,
                "shadow_error": None,
            },
        )
        # SAVE vs REJECT -> disagrees.
        append_jsonl(
            p.log_path,
            {
                "event": "full_shadow_comparison",
                "primary_decision": "SAVE",
                "shadow_decision": "REJECT",
                "agrees": False,
                "shadow_parse_failed": False,
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 700.0,
                "shadow_error": None,
            },
        )
        # Shadow parse failure -> not comparable.
        append_jsonl(
            p.log_path,
            {
                "event": "full_shadow_comparison",
                "primary_decision": "REJECT",
                "shadow_decision": "PARSE_FAILURE",
                "agrees": None,
                "shadow_parse_failed": True,
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 300.0,
                "shadow_error": None,
            },
        )
        # Shadow errored entirely -> excluded from parse-failure-rate/
        # save-rate denominators, still counted in comparisons/primary rate.
        append_jsonl(
            p.log_path,
            {
                "event": "full_shadow_comparison",
                "primary_decision": "REJECT",
                "shadow_decision": None,
                "agrees": None,
                "shadow_parse_failed": False,
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 100.0,
                "shadow_error": "timeout",
            },
        )
        # A non-shadow / facial-shadow event on the same log must be ignored.
        append_jsonl(p.log_path, {"event": "final_error", "error": "unrelated"})
        append_jsonl(
            p.log_path,
            {
                "event": "facial_shadow_comparison",
                "batch": False,
                "candidate_count": 1,
                "primary_decision": "FACIAL_YES",
                "shadow_decision": "FACIAL_YES",
                "agrees": True,
                "shadow_parse_failed": False,
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 999.0,
                "shadow_error": None,
            },
        )

        summary = p._shadow_full_summary()

    assert summary is not None
    assert summary["model"] == "accounts/fireworks/models/glm-5p2"
    assert summary["comparisons"] == 4  # facial_shadow_comparison row excluded
    # Comparable (agrees is not None): rows 1+2 -> 1 agreement / 2 -> 0.5.
    assert summary["agreement_rate"] == 0.5
    # Responded (excludes the 1 shadow-error row) = 3; 1 parse failure -> 1/3.
    assert summary["shadow_parse_failure_rate"] == round(1 / 3, 4)
    # Primary save-family: SAVE, SAVE = 2 of 4.
    assert summary["primary_save_rate"] == 0.5
    # Shadow valid (non-parse-failed, non-errored) = 2 (INFERENTIAL_SAVE, REJECT);
    # shadow save-family among those = 1 (INFERENTIAL_SAVE) -> 0.5.
    assert summary["shadow_save_rate"] == 0.5
    assert summary["mean_latency_ms"] == round((500 + 700 + 300 + 100) / 4, 1)


def test_shadow_full_summary_omitted_from_run_report_snapshot_metrics_when_absent():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(brief_name="test", strings=[])
        # Ported with the A1 slice-1 extraction: _build_run_report_snapshot moved to
        # RunReportService, which calls its OWN methods — Pipeline-level patches no
        # longer intercept them (Wave 6 boundary review P1).
        svc = p._run_report_service
        with patch.object(svc, "_cost_summary_for_report", return_value={"status": "no_cost_data"}), \
             patch.object(svc, "_run_health_summary", return_value={"status": "no_runtime_state"}), \
             patch.object(svc, "_search_intelligence_aggregate", return_value={}), \
             patch.object(svc, "_load_profile_index_for_adaptation", return_value={}), \
             patch.object(svc, "_load_run_report_decisions", return_value=[]), \
             patch.object(svc, "_bias_summary_for_report", return_value=""):
            snapshot = p._build_run_report_snapshot(progress)

    assert "shadow_full" not in snapshot["metrics_summary"]


# ---------------------------------------------------------------------------
# Per-tier shadow spend honesty: facial-tier and full-eval-tier both land
# as provider="fireworks" rows in token-cost-log.jsonl, distinguished only
# by the shadow_stage field — each summary must sum ONLY its own tier.
# ---------------------------------------------------------------------------


def _write_shadow_cost_rows(log_path, rows):
    for provider, shadow_stage, cost, cache_read, input_tokens in rows:
        record = {
            "provider": provider,
            "estimated_cost_usd": cost,
            "cache_read_input_tokens": cache_read,
            "input_tokens": input_tokens,
        }
        if shadow_stage is not None:
            record["shadow_stage"] = shadow_stage
        append_jsonl(log_path, record)


def test_shadow_facial_summary_cost_excludes_full_shadow_rows():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        append_jsonl(
            p.log_path,
            {
                "event": "facial_shadow_comparison",
                "batch": False,
                "candidate_count": 1,
                "primary_decision": "FACIAL_YES",
                "shadow_decision": "FACIAL_YES",
                "agrees": True,
                "shadow_parse_failed": False,
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 100.0,
                "shadow_error": None,
            },
        )
        _write_shadow_cost_rows(
            p.output_dir / "token-cost-log.jsonl",
            [
                ("fireworks", "facial_shadow", 0.10, 500, 200),
                ("fireworks", "full_shadow", 0.90, 1000, 400),
                ("anthropic", None, 5.0, 0, 0),
            ],
        )

        summary = p._shadow_facial_summary()

    assert summary is not None
    assert summary["shadow_cost_usd"] == 0.10
    # cache-hit rate is over facial_shadow rows only: 500 / (500 + 200).
    assert summary["mean_cache_hit_rate"] == round(500 / 700, 4)


def test_shadow_full_summary_cost_excludes_facial_shadow_rows():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        append_jsonl(
            p.log_path,
            {
                "event": "full_shadow_comparison",
                "primary_decision": "SAVE",
                "shadow_decision": "SAVE",
                "agrees": True,
                "shadow_parse_failed": False,
                "shadow_model": "accounts/fireworks/models/glm-5p2",
                "latency_ms": 500.0,
                "shadow_error": None,
            },
        )
        _write_shadow_cost_rows(
            p.output_dir / "token-cost-log.jsonl",
            [
                ("fireworks", "facial_shadow", 0.10, 500, 200),
                ("fireworks", "full_shadow", 0.90, 1000, 400),
                ("anthropic", None, 5.0, 0, 0),
            ],
        )

        summary = p._shadow_full_summary()

    assert summary is not None
    assert summary["shadow_cost_usd"] == 0.90
    assert summary["mean_cache_hit_rate"] == round(1000 / 1400, 4)


def test_shadow_cache_hit_rate_is_none_when_no_matching_rows():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        # No token-cost-log.jsonl at all.
        assert p._shadow_cache_hit_rate(shadow_stage="facial_shadow") is None

        _write_shadow_cost_rows(
            p.output_dir / "token-cost-log.jsonl",
            [("fireworks", "full_shadow", 0.90, 1000, 400)],
        )
        # File exists but has no facial_shadow rows.
        assert p._shadow_cache_hit_rate(shadow_stage="facial_shadow") is None


def test_run_block_adaptation_records_inserted_and_displaced_string_ids():
    from shared.storage import read_jsonl

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="done", boolean="one", status="done", block="Block A"),
                SearchString(id=2, name="old queued", boolean="two", block="Block A"),
            ],
        )

        response = AdaptationResponse(
            new_strings=[{"boolean": "replacement", "rationale": "replacement"}],
            skip_remaining=[{"string_id": 2, "reason": "superseded"}],
        )

        asyncio.run(
            p._run_block_adaptation(
                "Block A",
                [progress.strings[0]],
                progress,
                lambda *args, **kwargs: response,
            )
        )

        events = read_jsonl(p.log_path)
        block_events = [e for e in events if e.get("event") == "block_adaptation"]
        assert len(block_events) == 1
        assert block_events[0]["inserted_string_ids"] == [3]
        assert block_events[0]["displaced_string_ids"] == [2]


def test_adaptive_new_string_with_lint_error_is_not_inserted():
    """P5 (Wave 2): adaptive strings are the second queueing path — an
    error-severity lint finding blocks insertion there exactly as it blocks
    the initial queue build; healthy adaptive strings still insert."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="done", boolean="one", status="done", block="Block A"),
                SearchString(id=2, name="queued", boolean="two", block="Block A"),
            ],
        )

        response = AdaptationResponse(
            new_strings=[
                {"boolean": '("broken" OR "group"', "rationale": "unbalanced"},
                {"boolean": '"agent platform"', "rationale": "healthy"},
            ],
        )

        asyncio.run(
            p._run_block_adaptation(
                "Block A",
                [progress.strings[0]],
                progress,
                lambda *args, **kwargs: response,
            )
        )

        inserted = [s for s in progress.strings if s.string_type == "Adaptive"]
        assert [s.boolean for s in inserted] == ['"agent platform"']
        assert len(p._lint_blocked_strings) == 1
        assert p._lint_blocked_strings[0]["source"] == "adaptive"
        assert "unbalanced_parenthesis" in p._lint_blocked_strings[0]["codes"]


def test_constraint_manifest_gate_aborts_on_comp_band_brief():
    """P3b (Wave 2): a recruiter-stated comp band has zero owners repo-wide —
    the manifest gate aborts through the orchestrator method run_full calls,
    and the persisted manifest records the zero-owner class."""
    from shared.constraint_manifest import ConstraintManifestError

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.intake_notes = "Stay inside the comp band: $180k - $220k base pay."
        p.brief_obj.instructions = []

        with pytest.raises(ConstraintManifestError, match="compensation"):
            p._enforce_constraint_manifest()

        manifest_path = Path(td) / "constraint_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["classes"]["compensation"]["status"] == "zero_owner"


def test_constraint_manifest_gate_passes_and_persists_on_clean_brief():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.intake_notes = ""
        p.brief_obj.instructions = []
        p.brief_obj.permanent_filters = {"Location": "New York City Metropolitan Area"}

        p._enforce_constraint_manifest()  # must not raise

        manifest = json.loads((Path(td) / "constraint_manifest.json").read_text())
        assert manifest["classes"]["geography"]["status"] == "owned"
        assert p._constraint_manifest["classes"]["geography"]["stated_in_brief"] is True




def test_geography_reassert_counter_counts_every_reapply_once():
    """P3b (Wave 2, lens fix): the reassert counter has ONE owner — the apply
    seam — so recovery re-applies count and the chip-invariant path doesn't
    double-count."""
    from unittest.mock import AsyncMock

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {"Location": "Berlin Metropolitan Area"}
        p.browser = AsyncMock()
        p.browser.apply_location_filter = AsyncMock(return_value=True)

        asyncio.run(p._apply_session_location_filter())
        assert p._session_geography_receipt["reasserts"] == 0

        # Crash-recovery shape: flag reset + plain re-apply.
        p._session_location_applied = False
        asyncio.run(p._apply_session_location_filter())
        assert p._session_geography_receipt["reasserts"] == 1

        # Chip-invariant shape: chips missing -> re-assert through the apply.
        p.browser.read_applied_location_chips = AsyncMock(
            side_effect=[[], ["Berlin Metropolitan Area"]]
        )
        p._session_location_applied = False
        asyncio.run(p._verify_session_geography_chips())
        assert p._session_geography_receipt["reasserts"] == 2
        assert p._session_geography_receipt["verified_applied"] is True


def test_run_report_snapshot_carries_manifest_defer_counts_and_geography_receipt():
    """P3b (Wave 2): the snapshot folds the defer-dimension aggregate into the
    manifest and carries the session-geography receipt in run_metadata."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.intake_notes = ""
        p.brief_obj.instructions = []
        p.brief_obj.permanent_filters = {"Location": "Berlin Metropolitan Area"}
        p.brief_obj.raw = {}
        p._enforce_constraint_manifest()
        p._session_geography_receipt = {
            "intended": ["Berlin Metropolitan Area"],
            "verified_applied": True,
            "reasserts": 2,
        }

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1,
                    name="done",
                    boolean="one",
                    status="done",
                    surface_receipt={"unsupported_controls": ["seniority"]},
                ),
                SearchString(
                    id=2,
                    name="done2",
                    boolean="two",
                    status="done",
                    surface_receipt={"unsupported_controls": ["seniority", "industries"]},
                ),
            ],
        )

        snapshot = p._build_run_report_snapshot(progress)

        manifest = snapshot["metrics_summary"]["constraint_manifest"]
        assert manifest["requested_but_unsupported"] == {"seniority": 2, "industries": 1}
        receipt = snapshot["run_metadata"]["session_geography"]
        assert receipt["intended"] == ["Berlin Metropolitan Area"]
        assert receipt["reasserts"] == 2
        # The persisted manifest was re-written with the aggregate.
        persisted = json.loads((Path(td) / "constraint_manifest.json").read_text())
        assert persisted["requested_but_unsupported"] == {"seniority": 2, "industries": 1}


def test_adaptation_firewall_drops_are_recorded_as_lint_blocked():
    """P5 (Wave 2): ubiquity-gate drops from the adapted-string firewall ride
    the adaptation payload into the orchestrator's lint-blocked record, while
    the surviving adaptation actions (insertions here) still apply."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="done", boolean="one", status="done", block="Block A"),
                SearchString(id=2, name="queued", boolean="two", block="Block A"),
            ],
        )

        response = AdaptationResponse(
            new_strings=[{"boolean": '"agent platform"', "rationale": "healthy survivor"}],
        )
        # What adapt_after_block attaches after the firewall dropped one string.
        setattr(
            response,
            "adapted_string_firewall",
            {
                "passed": True,
                "reports": [],
                "dropped": [
                    {
                        "boolean": '("AI") AND ("Engineer")',
                        "rationale": "too generic",
                        "family_key": "generic",
                        "code": "ubiquitous_and_gate",
                        "message": "AND clause is composed entirely of ubiquitous terms.",
                    }
                ],
            },
        )

        asyncio.run(
            p._run_block_adaptation(
                "Block A",
                [progress.strings[0]],
                progress,
                lambda *args, **kwargs: response,
            )
        )

        inserted = [s for s in progress.strings if s.string_type == "Adaptive"]
        assert [s.boolean for s in inserted] == ['"agent platform"']
        assert len(p._lint_blocked_strings) == 1
        blocked = p._lint_blocked_strings[0]
        assert blocked["source"] == "adaptive"
        assert blocked["codes"] == ["ubiquitous_and_gate"]
        assert blocked["family_key"] == "generic"


def test_block_report_surfaces_lint_warning_counts():
    """P5 (Wave 2): warning-severity lint findings surface in the block report
    so adaptation sees craft health — per-string counts in string_details and
    a rendered line in the summary text."""
    captured: dict = {}

    def _capture_adapt(brief, block_report, *args, **kwargs):
        captured["report"] = block_report
        return AdaptationResponse(no_change=True)

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)

        warned = SearchString(
            id=1,
            name="warned",
            boolean='("$M" OR "fraud analytics")',
            status="done",
            block="Block A",
            boolean_lint={
                "boolean": '("$M" OR "fraud analytics")',
                "findings": [
                    {
                        "severity": "warning",
                        "code": "noop_special_character",
                        "message": "Quoted term contains special characters LinkedIn ignores.",
                        "repair_hint": "Spell the value out in words.",
                    }
                ],
                "has_error": False,
            },
        )
        clean = SearchString(
            id=2, name="clean", boolean='"agent platform"', status="done", block="Block A"
        )
        progress = Progress(
            brief_name="test",
            strings=[warned, clean, SearchString(id=3, name="queued", boolean="x", block="B")],
        )

        asyncio.run(
            p._run_block_adaptation(
                "Block A",
                [warned, clean],
                progress,
                _capture_adapt,
            )
        )

        report = captured["report"]
        details_by_id = {d["string_id"]: d for d in report.string_details}
        assert details_by_id[1]["lint_warnings"] == 1
        assert details_by_id[2]["lint_warnings"] == 0
        summary = report.to_summary_text()
        assert "lint" in summary.lower()


def test_block_adaptation_receives_every_completed_string_and_full_outcome():
    """A completed string cannot disappear from adaptation after string-id repair."""

    captured: dict = {}

    def _capture_adapt(_brief, block_report, _remaining, **_kwargs):
        captured["report"] = block_report
        return AdaptationResponse(no_change=True)

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)
        direct = SearchString(
            id=11,
            name="direct",
            boolean='"reliable systems"',
            status="done",
            block="Block A",
            pages_reviewed=1,
            candidates_count=2,
            saves=["Candidate One"],
            full_reviewed_count=2,
            full_outreach_count=1,
            full_reject_count=1,
        )
        screened = SearchString(
            id=12,
            name="screened",
            boolean='"operations systems"',
            status="done",
            block="Block A",
            pages_reviewed=1,
            candidates_count=2,
            saves=["Candidate Two"],
            full_reviewed_count=2,
            full_outreach_count=1,
            full_review_count=1,
        )
        progress = Progress(
            brief_name="test",
            strings=[
                direct,
                screened,
                SearchString(
                    id=13,
                    name="queued",
                    boolean='"next population"',
                    status="queued",
                    block="Block B",
                ),
            ],
        )

        asyncio.run(
            p._run_block_adaptation(
                "Block A",
                [direct, screened],
                progress,
                _capture_adapt,
            )
        )

    report = captured["report"]
    details = {detail["string_id"]: detail for detail in report.string_details}
    assert set(details) == {11, 12}
    assert report.total_saves == 2
    assert details[11]["full_outreach"] == 1
    assert details[11]["full_reject"] == 1
    assert details[12]["full_outreach"] == 1
    assert details[12]["full_review"] == 1
    assert details[11]["physical_saves"] == 1
    assert details[12]["physical_saves"] == 1


def test_run_block_adaptation_logs_no_change_decline_and_touches_nothing():
    """P11.1: an explicit no_change decision is logged on the block_adaptation
    event and applies nothing — the queue is byte-identical before/after."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="done", boolean="one", status="done", block="Block A"),
                SearchString(id=2, name="queued", boolean="two", block="Block A"),
            ],
        )
        # Compare identity/order/status/boolean/block — not the full
        # to_dict(), since building the block report hydrates cosmetic
        # metadata (family_key, novelty_bucket, ...) on every checkpoint
        # regardless of the adaptation decision. What "touch nothing" means
        # here is that no_change applies no skip/reorder/insert/pivot.
        def _queue_shape(strings):
            return [(s.id, s.status, s.boolean, s.block) for s in strings]

        before = _queue_shape(progress.strings)

        response = AdaptationResponse(no_change=True)

        asyncio.run(
            p._run_block_adaptation(
                "Block A",
                [progress.strings[0]],
                progress,
                lambda *args, **kwargs: response,
            )
        )

        after = _queue_shape(progress.strings)
        assert after == before

        events = read_jsonl(p.log_path)
        block_events = [e for e in events if e.get("event") == "block_adaptation"]
        assert len(block_events) == 1
        assert block_events[0]["no_change"] is True
        assert block_events[0]["inserted_string_ids"] == []
        assert block_events[0]["displaced_string_ids"] == []


def test_run_block_adaptation_applies_requested_checkpoint_cadence_to_queue_segmentation():
    """P11.2: a clamped next_checkpoint_after relabels exactly that many
    upcoming queued strings into one synthetic checkpoint block, overriding
    the static 5-string batch segmentation."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="done", boolean="one", status="done", block="Compound Batch 1"),
                SearchString(id=2, name="q2", boolean="two", block="Compound Batch 2"),
                SearchString(id=3, name="q3", boolean="three", block="Compound Batch 2"),
                SearchString(id=4, name="q4", boolean="four", block="Compound Batch 2"),
                SearchString(id=5, name="q5", boolean="five", block="Compound Batch 3"),
            ],
        )

        response = AdaptationResponse(next_checkpoint_after=2)
        setattr(response, "next_checkpoint_after_requested", 2)

        asyncio.run(
            p._run_block_adaptation(
                "Compound Batch 1",
                [progress.strings[0]],
                progress,
                lambda *args, **kwargs: response,
            )
        )

        queued = [s for s in progress.strings if s.status == "queued"]
        assert [s.id for s in queued] == [2, 3, 4, 5]
        assert queued[0].block == queued[1].block
        assert queued[0].block != "Compound Batch 2"
        assert queued[2].block == "Compound Batch 2"
        assert queued[3].block == "Compound Batch 3"

        events = read_jsonl(p.log_path)
        cadence_events = [e for e in events if e.get("event") == "adaptation_checkpoint_cadence"]
        assert len(cadence_events) == 1
        assert cadence_events[0]["requested"] == 2
        assert cadence_events[0]["applied"] == 2
        assert cadence_events[0]["relabeled_string_ids"] == [2, 3]


def test_run_block_adaptation_leaves_block_labels_untouched_when_cadence_lever_absent():
    """EXIT GATE (spec sec 12 P11): a model response that never sets
    next_checkpoint_after leaves the pre-existing block segmentation
    byte-identical to pre-P11 behavior."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="done", boolean="one", status="done", block="Compound Batch 1"),
                SearchString(id=2, name="q2", boolean="two", block="Compound Batch 2"),
                SearchString(id=3, name="q3", boolean="three", block="Compound Batch 2"),
            ],
        )
        before_blocks = [s.block for s in progress.strings]

        response = AdaptationResponse()

        asyncio.run(
            p._run_block_adaptation(
                "Compound Batch 1",
                [progress.strings[0]],
                progress,
                lambda *args, **kwargs: response,
            )
        )

        assert [s.block for s in progress.strings] == before_blocks
        events = read_jsonl(p.log_path)
        assert not [e for e in events if e.get("event") == "adaptation_checkpoint_cadence"]


# ---------------------------------------------------------------------------
# FACIAL_YES recovery
# ---------------------------------------------------------------------------

def test_facial_yes_prior_triggers_reeval():
    """_prior_outcomes[url] == 'FACIAL_YES' → discarded from _seen_urls for re-eval."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/facial-yes"

        # Simulate: history has FACIAL_YES
        p._seen_urls.add(url)
        p._prior_outcomes[url] = "FACIAL_YES"

        # The dedup check logic discards FACIAL_YES from _seen_urls
        prior = p._prior_outcomes.get(url, "")
        assert prior == "FACIAL_YES"
        p._seen_urls.discard(url)

        # After discard, should pass dedup
        assert url not in p._seen_urls


# ---------------------------------------------------------------------------
# Glance skip
# ---------------------------------------------------------------------------

def test_glance_skip_same_session_only():
    """Glance-skip marks terminal in-session via _mark_terminal, NOT persisted to history."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/glance"

        # Simulate glance skip: add to in-flight then mark terminal
        p._in_flight_urls.add(url)
        p._mark_terminal(url)

        assert url in p._seen_urls
        assert url not in p._in_flight_urls

        # No history entry written — on resume, URL should NOT be in _seen_urls
        p._seen_urls = set()
        p._in_flight_urls = set()
        p._prior_outcomes = {}
        p._load_candidate_history()

        assert url not in p._seen_urls


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("this is not noise", "proceed"),
        ("noise", "noise"),
        ("NOISE.", "noise"),
    ],
)
def test_glance_llm_check_strictly_parses_first_token(reply, expected):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.role_title = "Staff Engineer"
        p.brief_obj.role_description = "Build infrastructure."
        p.brief_obj.minimum_bar = "Relevant engineering experience."

        with patch("shared.llm_clients.cheap_llm", return_value=reply):
            assert (
                p._glance_llm_check(
                    [_make_snippet(name="Glance Candidate", current_title="Engineer")],
                    {"title_cluster": {"noise": True}},
                )
                == expected
            )


# ---------------------------------------------------------------------------
# _restart_string
# ---------------------------------------------------------------------------

def test_restart_string_clears_history():
    """_restart_string removes matching URLs from history, _seen_urls, _prior_outcomes."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url1 = "/talent/profile/string1-candidate"
        url2 = "/talent/profile/other-string-candidate"

        # Write snippets for string 1 and string 2
        append_jsonl(p.snippets_path, _make_snippet(
            profile_url=url1, source_string_id=1,
        ).to_dict())
        append_jsonl(p.snippets_path, _make_snippet(
            profile_url=url2, source_string_id=2, name="Other Person",
        ).to_dict())

        # Write history entries
        append_jsonl(p.history_path, {
            "profile_url": url1, "candidate_name": "Test Person",
            "outcome": "REJECT", "confidence": 0.9,
            "source_string_id": 1,
            "timestamp": "2026-01-01T00:00:00+00:00",
        })
        append_jsonl(p.history_path, {
            "profile_url": url2, "candidate_name": "Other Person",
            "outcome": "SAVE", "confidence": 0.95,
            "source_string_id": 2,
            "timestamp": "2026-01-01T00:00:00+00:00",
        })

        # Load into memory
        p._seen_urls = {url1, url2}
        p._prior_outcomes = {url1: "REJECT", url2: "SAVE"}
        p._in_flight_urls = set()

        # Create a mock progress with string 1
        from shared.schemas import SearchString, Progress
        s1 = SearchString(id=1, name="test string", boolean="test", status="done",
                          pages_reviewed=3, saves=[], notes="")
        s2 = SearchString(id=2, name="other string", boolean="test2", status="done",
                          pages_reviewed=2, saves=[], notes="")
        progress = Progress(brief_name="test", strings=[s1, s2], current_string_id=1)
        progress.save = MagicMock()

        p._restart_string(progress, 1)

        # url1 should be removed from everything
        assert url1 not in p._seen_urls
        assert url1 not in p._prior_outcomes

        # url2 should still be present
        assert url2 in p._seen_urls
        assert p._prior_outcomes[url2] == "SAVE"

        # History file should only have url2's entry
        remaining = read_jsonl(p.history_path)
        remaining_urls = [r["profile_url"] for r in remaining]
        assert url1 not in remaining_urls
        assert url2 in remaining_urls


def test_restart_strings_restarts_multiple_ids():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        append_jsonl(p.snippets_path, _make_snippet(
            profile_url="/talent/profile/string1-candidate",
            source_string_id=1,
        ).to_dict())
        append_jsonl(p.snippets_path, _make_snippet(
            profile_url="/talent/profile/string2-candidate",
            source_string_id=2,
            name="Second Person",
        ).to_dict())

        append_jsonl(p.history_path, {
            "profile_url": "/talent/profile/string1-candidate",
            "candidate_name": "Test Person",
            "outcome": "REJECT",
            "confidence": 0.9,
            "source_string_id": 1,
            "timestamp": "2026-01-01T00:00:00+00:00",
        })
        append_jsonl(p.history_path, {
            "profile_url": "/talent/profile/string2-candidate",
            "candidate_name": "Second Person",
            "outcome": "SAVE",
            "confidence": 0.95,
            "source_string_id": 2,
            "timestamp": "2026-01-01T00:00:00+00:00",
        })

        p._seen_urls = {"/talent/profile/string1-candidate", "/talent/profile/string2-candidate"}
        p._prior_outcomes = {
            "/talent/profile/string1-candidate": "REJECT",
            "/talent/profile/string2-candidate": "SAVE",
        }
        p._in_flight_urls = set()

        s1 = SearchString(id=1, name="one", boolean="a", status="done", pages_reviewed=3, saves=["A"], notes="Error")
        s2 = SearchString(id=2, name="two", boolean="b", status="done", pages_reviewed=2, saves=["B"], notes="Error")
        s3 = SearchString(id=3, name="three", boolean="c", status="queued", pages_reviewed=0, saves=[], notes="")
        progress = Progress(brief_name="test", strings=[s1, s2, s3], current_string_id=2)
        progress.save = MagicMock()

        p._restart_strings(progress, [2, 1, 2])

        assert s1.status == "queued"
        assert s2.status == "queued"
        assert s1.notes == ""
        assert s2.notes == ""
        assert s1.saves == []
        assert s2.saves == []
        assert "/talent/profile/string1-candidate" not in p._seen_urls
        assert "/talent/profile/string2-candidate" not in p._seen_urls
        assert "/talent/profile/string1-candidate" not in p._prior_outcomes
        assert "/talent/profile/string2-candidate" not in p._prior_outcomes


# ---------------------------------------------------------------------------
# In-flight reset on dedup rebuild
# ---------------------------------------------------------------------------

def test_in_flight_reset_on_dedup_rebuild():
    """_in_flight_urls and _prior_outcomes start empty on each dedup rebuild."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        # Simulate some state from a prior run
        p._in_flight_urls = {"/talent/profile/stale"}
        p._prior_outcomes = {"stale": "FACIAL_YES"}

        # Rebuild as init paths do
        p._seen_urls = set()
        p._in_flight_urls = set()
        p._prior_outcomes = {}
        p._load_candidate_history()

        assert len(p._in_flight_urls) == 0
        # _prior_outcomes only populated from history file
        assert "stale" not in p._prior_outcomes


# ---------------------------------------------------------------------------
# _prior_outcomes sync
# ---------------------------------------------------------------------------

def test_prior_outcomes_synced_on_terminal():
    """After history write + _mark_terminal, _prior_outcomes[url] matches the decision."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/synced"

        # Simulate what the code does after a terminal facial decision
        p._in_flight_urls.add(url)
        p._prior_outcomes[url] = "FACIAL_NO"
        p._mark_terminal(url)

        assert url in p._seen_urls
        assert url not in p._in_flight_urls
        assert p._prior_outcomes[url] == "FACIAL_NO"

        # Simulate what the code does after a terminal full decision
        url2 = "/talent/profile/synced2"
        p._in_flight_urls.add(url2)
        p._prior_outcomes[url2] = "SAVE"
        p._mark_terminal(url2)

        assert url2 in p._seen_urls
        assert p._prior_outcomes[url2] == "SAVE"


# ---------------------------------------------------------------------------
# Sequential per-card extraction
# ---------------------------------------------------------------------------

def test_extract_card_snippet_uses_dom_metadata():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.focus_card_for_review = AsyncMock()
        p.browser.get_card_snapshot = AsyncMock(return_value={
            "innertext": "Select Ada Lovelace\nAda Lovelace\nML Engineer",
            "name": "Ada Lovelace",
            "url": "/talent/profile/ada",
            "already_saved": True,
        })

        from shared.schemas import SearchString
        search_string = SearchString(id=9, name="seq", boolean="(test)")

        with patch("linkedin.acquisition.extract_snippet_from_card_innertext", return_value=_make_snippet(
            name="LLM Name",
            profile_url="",
            source_string_id=9,
            source_string_name="seq",
            page=2,
            result_rank=3,
        )), patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            snippet = asyncio.run(p._extract_card_snippet(search_string, page_num=2, card_index=2))

        assert snippet is not None
        assert snippet.name == "Ada Lovelace"
        assert snippet.profile_url == "/talent/profile/ada"
        assert snippet.card_index == 2
        assert snippet.already_saved is True
        p.browser.focus_card_for_review.assert_awaited_once_with(2)


def test_extract_card_snippet_returns_none_when_card_text_missing():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.focus_card_for_review = AsyncMock()
        p.browser.get_card_snapshot = AsyncMock(return_value={
            "innertext": "",
            "name": "",
            "url": "",
            "already_saved": False,
        })

        from shared.schemas import SearchString
        search_string = SearchString(id=9, name="seq", boolean="(test)")

        with patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            snippet = asyncio.run(p._extract_card_snippet(search_string, page_num=1, card_index=0))

    assert snippet is None


def test_extract_card_snippet_uses_dom_metadata_when_card_text_missing():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.focus_card_for_review = AsyncMock()
        p.browser.get_card_snapshot = AsyncMock(return_value={
            "innertext": "",
            "name": "Ada Lovelace",
            "url": "/talent/profile/ada",
            "already_saved": False,
        })

        from shared.schemas import SearchString
        search_string = SearchString(id=9, name="seq", boolean="(test)")

        with patch(
            "linkedin.acquisition.extract_snippet_from_card_innertext",
            return_value=_make_snippet(
                name="Ada Lovelace",
                profile_url="",
                source_string_id=9,
                source_string_name="seq",
                page=1,
                result_rank=1,
            ),
        ) as extract_mock, patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            snippet = asyncio.run(p._extract_card_snippet(search_string, page_num=1, card_index=0))

        assert snippet is not None
        assert snippet.name == "Ada Lovelace"
        assert snippet.profile_url == "/talent/profile/ada"
        extract_mock.assert_called_once()
        assert extract_mock.call_args.kwargs["dom_name"] == "Ada Lovelace"
        assert extract_mock.call_args.kwargs["dom_url"] == "/talent/profile/ada"


def test_extract_card_snippet_returns_none_on_slot_rehydration_error():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.focus_card_for_review = AsyncMock(
            side_effect=TimeoutError("Locator.scroll_into_view_if_needed: Timeout 3000ms exceeded.")
        )
        p.browser.get_card_snapshot = AsyncMock()
        p.browser.go_back_to_results = AsyncMock()

        search_string = SearchString(id=16, name="seq", boolean="(test)")

        with patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            snippet = asyncio.run(p._extract_card_snippet(search_string, page_num=1, card_index=5))

        assert snippet is None
        p.browser.go_back_to_results.assert_awaited_once()
        p.browser.get_card_snapshot.assert_not_awaited()


def test_extract_card_snippet_propagates_browser_fatal_cleanup_failure():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.focus_card_for_review = AsyncMock(
            side_effect=TimeoutError("candidate-local slot timeout")
        )
        p.browser.get_card_snapshot = AsyncMock()
        p.browser.go_back_to_results = AsyncMock(
            side_effect=RuntimeError("browser context closed")
        )
        search_string = SearchString(id=16, name="seq", boolean="(test)")

        with patch(
            "linkedin.acquisition.human_delay_correlated",
            return_value=0.0,
        ), pytest.raises(RuntimeError, match="browser context closed"):
            asyncio.run(
                p._extract_card_snippet(
                    search_string,
                    page_num=1,
                    card_index=5,
                )
            )

        p.browser.go_back_to_results.assert_awaited_once()
        p.browser.get_card_snapshot.assert_not_awaited()


def test_extract_card_snippet_treats_parser_failure_as_candidate_local():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.focus_card_for_review = AsyncMock()
        p.browser.get_card_snapshot = AsyncMock(return_value={
            "innertext": "unusual candidate card",
            "name": "Ada",
            "url": "/talent/profile/ada",
        })
        search_string = SearchString(id=16, name="seq", boolean="(test)")

        with patch(
            "linkedin.acquisition.extract_snippet_from_card_innertext",
            side_effect=ValueError("candidate-local parse defect"),
        ), patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            snippet = asyncio.run(
                p._extract_card_snippet(
                    search_string,
                    page_num=1,
                    card_index=0,
                )
            )

        assert snippet is None


@pytest.mark.parametrize(
    "browser_method",
    ["focus_card_for_review", "get_card_snapshot"],
)
def test_extract_card_snippet_propagates_page_crash_without_cleanup(browser_method):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.focus_card_for_review = AsyncMock()
        p.browser.get_card_snapshot = AsyncMock(return_value={})
        getattr(p.browser, browser_method).side_effect = RuntimeError("Page crashed")
        p.browser.go_back_to_results = AsyncMock()
        search_string = SearchString(id=16, name="seq", boolean="(test)")

        with patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            with pytest.raises(RuntimeError, match="Page crashed"):
                asyncio.run(
                    p._extract_card_snippet(
                        search_string,
                        page_num=1,
                        card_index=0,
                    )
                )

        p.browser.go_back_to_results.assert_not_awaited()


def _run_extraction_only_page(p, search_string, page_num, slot_count):
    p.browser.get_card_slot_count = AsyncMock(return_value=slot_count)
    return asyncio.run(
        p._review_page_sequentially(
            search_string,
            page_num,
            100,
            MagicMock(),
            [],
            {"duplicates": 0},
        )
    )


@pytest.mark.parametrize("batch_mode", [False, True], ids=["sequential", "batch"])
@pytest.mark.parametrize(
    "browser_method",
    ["focus_card_for_review", "get_card_snapshot"],
)
def test_browser_extraction_failures_trip_page_breaker_and_success_resets(
    batch_mode,
    browser_method,
):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = batch_mode
        p._validate_judgment_runtime_configuration = MagicMock()
        failure = TimeoutError("card slot timed out")
        p.browser.focus_card_for_review = AsyncMock()
        p.browser.get_card_snapshot = AsyncMock(
            return_value={"innertext": "", "name": "", "url": ""}
        )
        getattr(p.browser, browser_method).side_effect = [
            failure,
            failure,
            failure,
            failure,
            None if browser_method == "focus_card_for_review" else {
                "innertext": "",
                "name": "",
                "url": "",
            },
            failure,
            failure,
            failure,
            failure,
            failure,
        ]
        search_string = SearchString(id=16, name="seq", boolean="(test)")

        with patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            with pytest.raises(
                RuntimeError,
                match="5 consecutive card extraction failures",
            ):
                _run_extraction_only_page(
                    p,
                    search_string,
                    page_num=1,
                    slot_count=10,
                )


class APIConnectionError(RuntimeError):
    pass


class _ProviderStatusError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"provider returned {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    "failure",
    [_ProviderStatusError(429), _ProviderStatusError(503), APIConnectionError("offline")],
    ids=["429", "503", "api-connection"],
)
def test_extract_card_snippet_propagates_exhausted_systemic_failures_by_identity(
    failure,
):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.focus_card_for_review = AsyncMock()
        p.browser.get_card_snapshot = AsyncMock(
            return_value={
                "innertext": "candidate",
                "name": "Ada",
                "url": "/talent/profile/ada",
            }
        )
        search_string = SearchString(id=16, name="seq", boolean="(test)")

        with patch(
            "linkedin.acquisition.extract_snippet_from_card_innertext",
            side_effect=failure,
        ), patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            with pytest.raises(type(failure)) as exc_info:
                asyncio.run(
                    p._extract_card_snippet(
                        search_string,
                        page_num=1,
                        card_index=0,
                    )
                )

        assert exc_info.value is failure


def test_malformed_json_with_auth_title_stays_local_and_counts_toward_breaker():
    with pytest.raises(RuntimeError) as parse_info:
        _parse_json_response('Authentication Engineer Page crashed 429 {not-json')
    failure = parse_info.value
    assert isinstance(failure.__cause__, json.JSONDecodeError)

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.focus_card_for_review = AsyncMock()
        p.browser.get_card_snapshot = AsyncMock(
            return_value={
                "innertext": "candidate",
                "name": "Ada",
                "url": "/talent/profile/ada",
            }
        )
        search_string = SearchString(id=16, name="seq", boolean="(test)")

        with patch(
            "linkedin.acquisition.extract_snippet_from_card_innertext",
            side_effect=failure,
        ), patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            with pytest.raises(
                RuntimeError,
                match="5 consecutive card extraction failures",
            ):
                _run_extraction_only_page(
                    p,
                    search_string,
                    page_num=1,
                    slot_count=5,
                )


def test_page_raises_after_five_consecutive_card_extraction_failures():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.focus_card_for_review = AsyncMock()
        p.browser.get_card_snapshot = AsyncMock(
            return_value={
                "innertext": "unusual candidate card",
                "name": "Ada",
                "url": "",
            }
        )
        search_string = SearchString(id=16, name="seq", boolean="(test)")

        with patch(
            "linkedin.acquisition.extract_snippet_from_card_innertext",
            side_effect=RuntimeError("Could not parse JSON from LLM response: malformed"),
        ), patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            with pytest.raises(RuntimeError, match="5 consecutive card extraction failures"):
                _run_extraction_only_page(p, search_string, page_num=1, slot_count=5)


def test_card_extraction_breaker_resets_with_success_and_page_or_string_change():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.focus_card_for_review = AsyncMock()
        p.browser.get_card_snapshot = AsyncMock(
            return_value={
                "innertext": "unusual candidate card",
                "name": "Ada",
                "url": "",
            }
        )
        first = SearchString(id=16, name="first", boolean="(one)")
        second = SearchString(id=17, name="second", boolean="(two)")
        failure = RuntimeError("Could not parse JSON from LLM response: malformed")
        side_effects = (
            [failure] * 4
            + [_make_snippet(name="Reset", profile_url="")]
            + [failure] * 4
            + [failure] * 4
            + [failure] * 4
            + [failure] * 5
        )

        with patch(
            "linkedin.acquisition.extract_snippet_from_card_innertext",
            side_effect=side_effects,
        ), patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            _run_extraction_only_page(p, first, page_num=1, slot_count=9)
            _run_extraction_only_page(p, first, page_num=2, slot_count=4)
            _run_extraction_only_page(p, second, page_num=1, slot_count=4)
            with pytest.raises(RuntimeError, match="5 consecutive card extraction failures"):
                _run_extraction_only_page(p, second, page_num=2, slot_count=5)


# ---------------------------------------------------------------------------
# Batch facial parity
# ---------------------------------------------------------------------------


def test_facial_page_containment_switch_defaults_off():
    assert config.LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED is False

def test_batch_full_failures_do_not_increment_facial_yes():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p.browser.get_card_slot_count = AsyncMock(return_value=2)
        p._extract_card_snippet = AsyncMock(side_effect=[
            _make_snippet(name="Alice", profile_url="/talent/profile/alice"),
            _make_snippet(name="Bob", profile_url="/talent/profile/bob"),
        ])
        p._full_evaluate = AsyncMock(side_effect=[
            OpusDecision(
                stage="full", decision="PARSE_FAILURE", path="none",
                confidence=0.0, rationale="[PARSE_FAILURE: bad output]",
                candidate_name="Alice", profile_url="/talent/profile/alice",
            ),
            OpusDecision(
                stage="full", decision="JUDGMENT_FAILURE", path="none",
                confidence=0.0, rationale="[JUDGMENT_FAILURE: timeout]",
                candidate_name="Bob", profile_url="/talent/profile/bob",
            ),
        ])
        p._checkpoint_progress = MagicMock()
        p._bias_monitor = None
        p._triage_tightened = False
        p._tightening_prefix = ""

        search_string = SearchString(id=1, name="batch", boolean="ml")
        page_report = MagicMock()
        all_candidates = []
        string_stats = {
            "pages": 1,
            "candidates": 0,
            "duplicates": 0,
            "facial_yes": 0,
            "facial_no": 0,
            "saves": 0,
            "rejects": 0,
        }

        with patch(
            "shared.judger.facial_judge_batch",
            return_value=[
                OpusDecision(
                    stage="facial", decision="FACIAL_YES", path="none",
                    confidence=1.0, rationale="good signal",
                    candidate_name="Alice", profile_url="/talent/profile/alice",
                ),
                OpusDecision(
                    stage="facial", decision="FACIAL_YES", path="none",
                    confidence=1.0, rationale="good signal",
                    candidate_name="Bob", profile_url="/talent/profile/bob",
                ),
            ],
        ):
            asyncio.run(
                p._review_page_batch(
                    search_string, 1, 0, page_report, all_candidates, string_stats, None,
                )
            )

        # Facial stage truth survives even when the later full calls fail;
        # failures must not fabricate a settled full-profile disposition.
        assert string_stats["facial_yes"] == 2
        assert string_stats.get("full_reviewed", 0) == 0
        assert string_stats["saves"] == 0
        assert string_stats["rejects"] == 0
        assert [c["outcome"] for c in all_candidates] == ["error", "error"]


def test_facial_provider_page_failure_is_contained_and_next_page_starts_clean(
    tmp_path,
    monkeypatch,
):
    p = _make_pipeline(str(tmp_path))
    p.brief_obj.has_v2_schema = True
    p.brief_obj._new_brief = None
    p._bias_monitor = None
    p._triage_tightened = False
    p._tightening_prefix = ""
    p._checkpoint_progress = MagicMock()

    first_page = [
        _make_snippet(
            name=f"Abandoned {index}",
            profile_url=f"/talent/profile/abandoned-{index}",
            result_rank=index,
        )
        for index in range(1, 3)
    ]
    next_page = _make_snippet(
        name="Next Page",
        profile_url="/talent/profile/next-page",
        page=2,
        result_rank=1,
    )
    p.browser.get_card_slot_count = AsyncMock(side_effect=[2, 1])
    p._extract_card_snippet = AsyncMock(side_effect=[*first_page, next_page])

    search_string = SearchString(
        id=1,
        name="contained",
        boolean="ml",
        status="queued",
        notes="before",
    )
    progress = Progress(brief_name="test", strings=[search_string])
    p._runtime_run_id, _ = p._runtime_bridge.start_or_resume_run(
        resume=False,
        initial_progress=progress,
    )
    state = bootstrap_experiment_state(search_string)
    p._experiment_states[search_string.id] = state
    before_state = state.to_dict()
    before_string = search_string.to_dict()
    p._arm_incomplete_page_rollback(search_string, state)
    state.mode = "experiment"
    state.mutations_used = 99
    search_string.status = "done"
    search_string.pages_reviewed = 99
    search_string.notes = "mutated"

    call_count = 0

    def judge_page(batch, *_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError("Fireworks provider read timed out")
        return [
            OpusDecision(
                stage="facial",
                decision="FACIAL_NO",
                path="normal_eligibility",
                confidence=1.0,
                rationale="next page completed",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            )
            for snippet in batch
        ]

    monkeypatch.setattr(config, "LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED", True)
    monkeypatch.setattr(config, "LINKEDIN_FACIAL_CONCURRENCY_ENABLED", False)
    monkeypatch.setattr(config, "LINKEDIN_FACIAL_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(config, "LINKEDIN_FACIAL_TARGET_BATCH_SIZE", 8)
    monkeypatch.setattr(config, "FULL_EVAL_PIPELINE_ENABLED", False)
    monkeypatch.setattr(config, "GLANCE_MIN_SNIPPETS", 99)
    monkeypatch.setattr(config, "EARLY_EXIT_MIN_CANDIDATES", 99)

    page_report = MagicMock()
    all_candidates: list[dict] = []
    string_stats = p._fresh_string_stats()
    with patch("shared.judger.facial_judge_batch", side_effect=judge_page):
        asyncio.run(
            p._review_page_batch(
                search_string,
                1,
                3,
                page_report,
                all_candidates,
                string_stats,
                progress,
            )
        )

        assert state.to_dict() == before_state
        restored_string = SearchString.from_dict(before_string)
        assert (
            search_string.status,
            search_string.pages_reviewed,
            search_string.notes,
        ) == (
            restored_string.status,
            restored_string.pages_reviewed,
            restored_string.notes,
        )
        assert p._incomplete_page_rollbacks == {}

        asyncio.run(
            p._review_page_batch(
                search_string,
                2,
                3,
                page_report,
                all_candidates,
                string_stats,
                progress,
            )
        )

    abandoned_urls = [snippet.profile_url for snippet in first_page]
    records = read_jsonl(p.log_path)
    abandoned_events = [
        record for record in records if record["event"] == "page_abandoned"
    ]
    assert len(abandoned_events) == 1
    assert abandoned_events[0]["string_id"] == search_string.id
    assert abandoned_events[0]["page"] == 1
    assert abandoned_events[0]["candidate_identities"] == abandoned_urls
    assert not [record for record in records if record["event"] == "pipeline_error"]
    assert [candidate["outcome"] for candidate in all_candidates] == [
        "page_abandoned",
        "page_abandoned",
        "facial_no",
    ]
    assert p._prior_outcomes == {
        abandoned_urls[0]: "PAGE_ABANDONED",
        abandoned_urls[1]: "PAGE_ABANDONED",
        next_page.profile_url: "FACIAL_NO",
    }
    assert set(abandoned_urls).issubset(p._seen_urls)
    assert p._in_flight_urls.isdisjoint({*abandoned_urls, next_page.profile_url})
    assert string_stats["page_abandoned_candidates"] == 2
    assert call_count == 2
    assert p._runtime_state.list_orphaned_attempts(
        source="linkedin",
        brief_id="test-project",
    ) == []
    with p._runtime_state.connect() as conn:
        abandoned_attempts = conn.execute(
            "SELECT c.identity_key, c.current_lifecycle_state, "
            "ca.failure_kind, ca.payload_json "
            "FROM candidate_attempts ca "
            "JOIN candidates c ON c.id = ca.candidate_id "
            "WHERE ca.failure_kind = 'page_abandoned' ORDER BY ca.id"
        ).fetchall()
    assert [row["identity_key"] for row in abandoned_attempts] == abandoned_urls
    assert {
        row["current_lifecycle_state"] for row in abandoned_attempts
    } == {"failed_terminal"}
    assert all(
        json.loads(row["payload_json"])["force_terminal"] is True
        for row in abandoned_attempts
    )


def test_facial_provider_breaker_finalizes_page_containment_before_raising(
    tmp_path,
    monkeypatch,
):
    p = _make_pipeline(str(tmp_path))
    p.brief_obj.has_v2_schema = True
    p.brief_obj._new_brief = None
    p._bias_monitor = None
    p._triage_tightened = False
    p._tightening_prefix = ""
    p._checkpoint_progress = MagicMock()
    p.stats["consecutive_facial_provider_failures"] = 2
    snippet = _make_snippet(
        name="Abandoned Before Breaker",
        profile_url="/talent/profile/abandoned-before-breaker",
    )
    p.browser.get_card_slot_count = AsyncMock(return_value=1)
    p._extract_card_snippet = AsyncMock(return_value=snippet)

    search_string = SearchString(
        id=1,
        name="breaker",
        boolean="ml",
        status="queued",
        notes="before",
    )
    progress = Progress(brief_name="test", strings=[search_string])
    p._runtime_run_id, _ = p._runtime_bridge.start_or_resume_run(
        resume=False,
        initial_progress=progress,
    )
    state = bootstrap_experiment_state(search_string)
    p._experiment_states[search_string.id] = state
    before_state = state.to_dict()
    before_string = search_string.to_dict()

    restored_snapshots: list[int] = []
    restore = p._restore_incomplete_page_rollback

    def track_restore(current_string):
        if current_string.id in p._incomplete_page_rollbacks:
            restored_snapshots.append(current_string.id)
        restore(current_string)

    p._restore_incomplete_page_rollback = track_restore

    async def process_page(current_string, current_progress):
        p._arm_incomplete_page_rollback(current_string, state)
        state.mode = "experiment"
        state.mutations_used = 99
        current_string.status = "done"
        current_string.pages_reviewed = 99
        current_string.notes = "mutated"
        return await p._review_page_batch(
            current_string,
            1,
            3,
            MagicMock(),
            [],
            p._fresh_string_stats(),
            current_progress,
        )

    p._process_string_impl = process_page
    monkeypatch.setattr(config, "LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED", True)
    monkeypatch.setattr(config, "LINKEDIN_FACIAL_CONCURRENCY_ENABLED", False)
    monkeypatch.setattr(config, "LINKEDIN_FACIAL_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(config, "LINKEDIN_FACIAL_TARGET_BATCH_SIZE", 8)
    monkeypatch.setattr(config, "FULL_EVAL_PIPELINE_ENABLED", False)
    monkeypatch.setattr(config, "GLANCE_MIN_SNIPPETS", 99)
    monkeypatch.setattr(config, "EARLY_EXIT_MIN_CANDIDATES", 99)

    with patch(
        "shared.judger.facial_judge_batch",
        side_effect=TimeoutError("Fireworks provider read timed out"),
    ):
        with pytest.raises(
            RuntimeError,
            match="facial provider failure threshold reached",
        ):
            asyncio.run(p._process_string(search_string, progress))

    events = [
        record
        for record in read_jsonl(p.log_path)
        if record["event"] == "page_abandoned"
    ]
    assert len(events) == 1
    assert events[0]["candidate_identities"] == [snippet.profile_url]
    assert restored_snapshots == [search_string.id]
    assert state.to_dict() == before_state
    restored_string = SearchString.from_dict(before_string)
    assert (
        search_string.status,
        search_string.pages_reviewed,
        search_string.notes,
    ) == (
        restored_string.status,
        restored_string.pages_reviewed,
        restored_string.notes,
    )


def test_facial_page_containment_off_rethrows_original_batch_error(
    tmp_path,
    monkeypatch,
):
    p = _make_pipeline(str(tmp_path))
    p.brief_obj.has_v2_schema = True
    p.brief_obj._new_brief = None
    p._bias_monitor = None
    p._tightening_prefix = ""
    p._checkpoint_progress = MagicMock()
    snippet = _make_snippet(profile_url="/talent/profile/uncontained")
    p.browser.get_card_slot_count = AsyncMock(return_value=1)
    p._extract_card_snippet = AsyncMock(return_value=snippet)
    failure = TimeoutError("Fireworks provider read timed out")

    monkeypatch.setattr(config, "LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED", False)
    monkeypatch.setattr(config, "LINKEDIN_FACIAL_CONCURRENCY_ENABLED", True)
    monkeypatch.setattr(config, "LINKEDIN_FACIAL_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(config, "LINKEDIN_FACIAL_TARGET_BATCH_SIZE", 8)
    monkeypatch.setattr(config, "GLANCE_MIN_SNIPPETS", 99)

    with patch("shared.judger.facial_judge_batch", side_effect=failure):
        with pytest.raises(TimeoutError) as raised:
            asyncio.run(
                p._review_page_batch(
                    SearchString(id=1, name="off", boolean="ml"),
                    1,
                    1,
                    MagicMock(),
                    [],
                    p._fresh_string_stats(),
                    None,
                )
            )

    assert str(raised.value) == str(failure)
    failure_records = read_jsonl(p.log_path)
    assert not [
        record for record in failure_records
        if record["event"] in {"facial_page_judgment_timing", "page_abandoned"}
    ]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("unclassified page fault"),
        ApiBudgetExhaustedError("credit balance is too low"),
        GovernorLimitReached("profile_limit"),
        OperatorStopRequested(),
        SessionExpired(),
        AllocatorPolicyError("allocator invariant"),
    ],
)
def test_facial_page_containment_rejects_non_provider_controls(error):
    from linkedin.orchestrator import _is_containable_facial_page_error

    assert _is_containable_facial_page_error(error) is False


# ---------------------------------------------------------------------------
# Judgment failure from exception is non-terminal
# ---------------------------------------------------------------------------

def test_judgment_failure_exception_not_terminal():
    """Facial exception -> JUDGMENT_FAILURE -> not in _seen_urls, not in history."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        url = "/talent/profile/exception-candidate"

        # Simulate: candidate enters in-flight
        p._in_flight_urls.add(url)

        # Simulate what orchestrator does on JUDGMENT_FAILURE:
        # is_failure_decision(facial.decision) is True -> discard from in-flight
        p._in_flight_urls.discard(url)

        # Should NOT be terminal
        assert url not in p._seen_urls
        assert url not in p._in_flight_urls
        assert url not in p._prior_outcomes

        # On resume, should NOT be in _seen_urls (no history entry)
        p._seen_urls = set()
        p._in_flight_urls = set()
        p._prior_outcomes = {}
        p._load_candidate_history()

        assert url not in p._seen_urls


# ---------------------------------------------------------------------------
# Resume / adaptation ordering
# ---------------------------------------------------------------------------

def _wire_real_run_full_page_path(p, *, result_counts):
    _allow_synthetic_run_completion(p)
    no_results = MagicMock()
    no_results.is_visible = AsyncMock(return_value=False)
    locator = MagicMock()
    locator.first = no_results
    page = MagicMock(
        url=(
            "https://www.linkedin.com/talent/hire/123/"
            "discover/recruiterSearch"
        )
    )
    page.locator.return_value = locator
    page.wait_for_timeout = AsyncMock()
    p.browser.page = page
    p.browser.connect = AsyncMock()
    p.browser.disconnect = AsyncMock()
    p.browser.navigate_to_search = AsyncMock()
    p.browser.go_back_to_results = AsyncMock()
    p.browser.enter_search_string = AsyncMock()
    p.browser.get_results_count = AsyncMock(side_effect=list(result_counts))
    p.browser.get_results_count_text = AsyncMock(
        side_effect=[str(value) for value in result_counts]
    )
    p.browser.go_to_next_page = AsyncMock(return_value=True)
    p._ensure_browser_healthy = AsyncMock()
    p._apply_session_location_filter = AsyncMock()
    p._enforce_constraint_manifest = MagicMock()
    p._load_candidate_history = MagicMock()
    p._load_search_memory = MagicMock()
    p._evaluate_variant_lifecycle = MagicMock(return_value=None)
    p._plan_variant_experiments = AsyncMock()
    p._print_session_summary = MagicMock()
    p._print_summary = MagicMock()
    p._generate_run_report = MagicMock()
    p._session_expired = MagicMock()
    p._session_expired.is_set.return_value = False


def test_run_full_real_page_path_adaptively_continues_then_stops():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(
            id=1,
            name="owner",
            boolean="one",
            status="in_progress",
        )
        progress = Progress(brief_name="test", strings=[search_string])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[50])

        snippets = [
            _make_snippet(
                name="Page One",
                profile_url="/talent/profile/page-one",
                source_string_id=search_string.id,
                page=1,
            ),
            _make_snippet(
                name="Page Two",
                profile_url="/talent/profile/page-two",
                source_string_id=search_string.id,
                page=2,
            ),
        ]
        p._card_slot_count_or_raise = AsyncMock(side_effect=[1, 1])
        p._extract_card_snippet = AsyncMock(side_effect=snippets)

        async def settle(snippet, *_args, **_kwargs):
            decision = OpusDecision(
                stage="full",
                decision="REJECT",
                path="none",
                confidence=0.9,
                rationale="not a fit",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            )
            p._note_page_full_review_expected(snippet)
            p._note_page_full_review_settled(
                snippet=snippet,
                decision=decision,
            )
            return decision

        p._evaluate_snippet = AsyncMock(side_effect=settle)
        p._assess_string_state = AsyncMock(
            side_effect=[
                {"decision": "continue", "rationale": "more signal", "page": 1},
                {"decision": "stop", "rationale": "adaptive stop", "page": 2},
            ]
        )

        asyncio.run(p.run_full(resume=True))

        saved = json.loads(Path(td, "progress.json").read_text())
        assert saved["strings"][0]["status"] == "done"
        assert p._assess_string_state.await_count == 2
        p.browser.go_to_next_page.assert_awaited_once()


def test_run_full_real_page_path_continues_after_candidate_local_extraction_failure():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(
            id=1,
            name="owner",
            boolean="one",
            status="in_progress",
        )
        progress = Progress(brief_name="test", strings=[search_string])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[25])
        p._card_slot_count_or_raise = AsyncMock(return_value=2)
        p.browser.focus_card_for_review = AsyncMock()
        p.browser.get_card_snapshot = AsyncMock(
            side_effect=[
                ValueError("candidate-local card defect"),
                {
                    "innertext": "Ada Lovelace\nEngineer at Acme",
                    "name": "Ada Lovelace",
                    "url": "/talent/profile/ada",
                    "already_saved": False,
                },
            ]
        )
        snippet = _make_snippet(
            name="Ada Lovelace",
            profile_url="/talent/profile/ada",
            source_string_id=search_string.id,
            page=1,
            result_rank=2,
        )

        async def settle(actual, *_args, **_kwargs):
            decision = OpusDecision(
                stage="full",
                decision="REJECT",
                path="none",
                confidence=0.9,
                rationale="not a fit",
                candidate_name=actual.name,
                profile_url=actual.profile_url,
            )
            p._note_page_full_review_expected(actual)
            p._note_page_full_review_settled(
                snippet=actual,
                decision=decision,
            )
            return decision

        p._evaluate_snippet = AsyncMock(side_effect=settle)
        p._assess_string_state = AsyncMock(
            return_value={
                "decision": "stop",
                "rationale": "adaptive stop",
                "page": 1,
            }
        )

        with patch(
            "linkedin.acquisition.extract_snippet_from_card_innertext",
            return_value=snippet,
        ), patch(
            "linkedin.acquisition.human_delay_correlated",
            return_value=0.0,
        ):
            asyncio.run(p.run_full(resume=True))

        assert p.browser.get_card_snapshot.await_count == 2
        p._evaluate_snippet.assert_awaited_once()
        saved = json.loads(Path(td, "progress.json").read_text())
        assert saved["strings"][0]["status"] == "done"


@pytest.mark.parametrize("legacy_status", ["error", "unexpected_legacy"])
def test_run_full_real_path_resumes_every_nonterminal_legacy_status(
    legacy_status,
):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(
            id=1,
            name="legacy",
            boolean="one",
            status=legacy_status,
        )
        progress = Progress(brief_name="test", strings=[search_string])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[0])

        asyncio.run(p.run_full(resume=True))

        p.browser.enter_search_string.assert_awaited_once_with("one")
        saved = json.loads(Path(td, "progress.json").read_text())
        assert saved["strings"][0]["status"] == "skipped"


def test_run_full_real_path_unexpected_owner_failure_leaves_later_queue_untouched():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        first = SearchString(
            id=1,
            name="first",
            boolean="one",
            status="in_progress",
        )
        second = SearchString(
            id=2,
            name="second",
            boolean="two",
            status="queued",
        )
        progress = Progress(brief_name="test", strings=[first, second])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[25])
        p._ensure_browser_healthy.side_effect = RuntimeError(
            "unexpected owner failure"
        )

        with pytest.raises(RuntimeError, match="unexpected owner failure"):
            asyncio.run(p.run_full(resume=True))

        p.browser.enter_search_string.assert_not_awaited()
        saved = json.loads(Path(td, "progress.json").read_text())
        assert [item["status"] for item in saved["strings"]] == [
            "in_progress",
            "queued",
        ]


def test_page_completes_after_candidate_local_full_extraction_failure():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=1,
            name="owner",
            boolean="one",
            status="in_progress",
        )
        progress = Progress(brief_name="test", strings=[owner])
        p._runtime_run_id, progress = p._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        owner = progress.strings[0]
        _wire_real_run_full_page_path(p, result_counts=[25])
        failed = _make_snippet(
            name="Retry Candidate",
            profile_url="/talent/profile/retry-candidate",
            source_string_id=owner.id,
            page=1,
        )
        p._card_slot_count_or_raise = AsyncMock(side_effect=[1, 0])
        p._extract_card_snippet = AsyncMock(return_value=failed)
        p._acquisition_service.extract_profile_summary = AsyncMock(
            side_effect=RuntimeError("candidate-local extraction failure")
        )
        p._assess_string_state = AsyncMock(
            side_effect=[
                {
                    "decision": "continue",
                    "rationale": "keep running",
                    "page": 1,
                },
                {
                    "decision": "stop",
                    "rationale": "finished",
                    "page": 2,
                },
            ]
        )
        facial = OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="DIRECT:test",
            confidence=0.9,
            rationale="full review required",
            candidate_name=failed.name,
            profile_url=failed.profile_url,
        )

        with patch(
            "linkedin.orchestrator.facial_judge",
            return_value=facial,
        ), patch(
            "linkedin.orchestrator.human_delay_correlated",
            return_value=0.0,
        ):
            asyncio.run(p._process_string(owner, progress))

        key = p._funnel_candidate_key(failed)
        assert p._assess_string_state.await_count == 2
        p.browser.go_to_next_page.assert_awaited_once()
        assert p._experiment_state_for(
            owner
        ).active_allocator_page_cursor() == 3
        assert p._resume_pending_full_decisions[key] == "FACIAL_YES"
        assert p._resume_pending_full_owner_ids[key] == owner.id
        assert p._resume_pending_full_snippets[key] is failed
        with sqlite3.connect(Path(td, "runtime_state.sqlite3")) as conn:
            status = conn.execute(
                "SELECT status FROM work_units "
                "WHERE kind = 'linkedin_string' AND source_unit_id = '1'"
            ).fetchone()[0]
        assert status == "in_progress"


@pytest.mark.parametrize("terminal_status", ["done", "skipped"])
def test_run_full_refuses_terminal_status_while_owner_obligation_stands(
    terminal_status,
):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=1,
            name="owner",
            boolean="one",
            status="in_progress",
        )
        progress = Progress(brief_name="test", strings=[owner])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[])
        pending = _make_snippet(
            name="Still Pending",
            profile_url="/talent/profile/still-pending",
            source_string_id=owner.id,
            page=1,
        )

        async def finish_with_obligation(search_string, _progress):
            search_string.status = terminal_status
            key = p._funnel_candidate_key(pending)
            p._resume_pending_full_decisions[key] = "FACIAL_YES"
            p._resume_pending_full_owner_ids[key] = owner.id
            p._resume_pending_full_snippets[key] = pending
            p._experiment_state_for(
                owner
            ).active_variant.allocator_page_cursor = 2

        p._process_string = AsyncMock(side_effect=finish_with_obligation)
        p._recover_owner_pending_full_evaluations = AsyncMock(return_value=1)

        with pytest.raises(RuntimeError, match="Still Pending"):
            asyncio.run(p.run_full(resume=True))

        assert owner.status == "in_progress"
        p._recover_owner_pending_full_evaluations.assert_awaited_once()


def test_run_full_checkpoints_owner_demotion_before_terminal_recovery_await():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(id=1, name="owner", boolean="one")
        progress = Progress(brief_name="test", strings=[owner])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[])
        pending = _make_snippet(
            name="Still Pending",
            profile_url="/talent/profile/still-pending",
            source_string_id=owner.id,
            page=1,
        )

        async def finish_with_obligation(search_string, actual_progress):
            search_string.status = "done"
            p._resume_pending_full_decisions[pending.profile_url] = "FACIAL_YES"
            p._resume_pending_full_owner_ids[pending.profile_url] = owner.id
            p._resume_pending_full_snippets[pending.profile_url] = pending
            p._experiment_state_for(owner).active_variant.allocator_page_cursor = 2
            p._checkpoint_progress(actual_progress, search_string=search_string)

        async def assert_durable_demotion(**_kwargs):
            with sqlite3.connect(Path(td, "runtime_state.sqlite3")) as conn:
                status = conn.execute(
                    "SELECT status FROM work_units "
                    "WHERE kind = 'linkedin_string' AND source_unit_id = '1'"
                ).fetchone()[0]
            assert status == "in_progress"
            return 1

        p._process_string = AsyncMock(side_effect=finish_with_obligation)
        p._recover_owner_pending_full_evaluations = AsyncMock(
            side_effect=assert_durable_demotion
        )

        with pytest.raises(RuntimeError, match="Still Pending"):
            asyncio.run(p.run_full(resume=True))


def test_operator_stop_after_terminal_checkpoint_keeps_owner_canonical_resumable():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(id=1, name="owner", boolean="one")
        later = SearchString(id=2, name="later", boolean="two")
        progress = Progress(brief_name="test", strings=[owner, later])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[])
        pending = _make_snippet(
            profile_url="/talent/profile/still-pending",
            source_string_id=owner.id,
            page=1,
        )

        async def stop_after_checkpoint(search_string, actual_progress):
            search_string.status = "done"
            p._resume_pending_full_decisions[pending.profile_url] = "FACIAL_YES"
            p._resume_pending_full_owner_ids[pending.profile_url] = owner.id
            p._resume_pending_full_snippets[pending.profile_url] = pending
            p._checkpoint_progress(actual_progress, search_string=search_string)
            raise OperatorStopRequested("operator stop after completed page")

        p._process_string = AsyncMock(side_effect=stop_after_checkpoint)

        with pytest.raises(OperatorStopRequested):
            asyncio.run(p.run_full(resume=True))

        with sqlite3.connect(Path(td, "runtime_state.sqlite3")) as conn:
            statuses = conn.execute(
                "SELECT status FROM work_units "
                "WHERE kind = 'linkedin_string' ORDER BY ordering_index"
            ).fetchall()
        assert statuses == [("in_progress",), ("queued",)]

        resumed_ids = []

        async def stop_on_resumed_owner(search_string, _progress):
            resumed_ids.append(search_string.id)
            raise OperatorStopRequested("stop on resumed owner")

        p._process_string = AsyncMock(side_effect=stop_on_resumed_owner)
        with pytest.raises(OperatorStopRequested):
            asyncio.run(p.run_full(resume=True))

        assert resumed_ids == [owner.id]
        assert later.status == "queued"


def test_run_full_checkpoints_hydrated_owner_demotion_before_queue_selection():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=1,
            name="owner",
            boolean="one",
            status="done",
        )
        later = SearchString(
            id=2,
            name="later",
            boolean="two",
            status="queued",
        )
        progress = Progress(brief_name="test", strings=[owner, later])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[])
        pending = _make_snippet(
            profile_url="/talent/profile/inherited-pending",
            source_string_id=owner.id,
            page=1,
        )

        def hydrate(_progress):
            p._resume_pending_full_decisions[pending.profile_url] = "FACIAL_YES"
            p._resume_pending_full_owner_ids[pending.profile_url] = owner.id
            p._resume_pending_full_snippets[pending.profile_url] = pending

        processed_ids: list[int] = []

        async def stop_on_first_process(search_string, _progress):
            processed_ids.append(search_string.id)
            raise OperatorStopRequested("stop before browser action")

        p._hydrate_resume_funnel_from_runtime = MagicMock(side_effect=hydrate)
        p._process_string = AsyncMock(side_effect=stop_on_first_process)

        with pytest.raises(OperatorStopRequested):
            asyncio.run(p.run_full(resume=True))

        assert processed_ids == [owner.id]
        assert later.status == "queued"
        with sqlite3.connect(Path(td, "runtime_state.sqlite3")) as conn:
            statuses = conn.execute(
                "SELECT status FROM work_units "
                "WHERE kind = 'linkedin_string' ORDER BY ordering_index"
            ).fetchall()
        assert statuses == [("in_progress",), ("queued",)]


def test_terminal_recovery_reenters_owner_surface_at_page_one():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(id=1, name="owner", boolean="one")
        progress = Progress(brief_name="test", strings=[owner])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[])
        pending = _make_snippet(
            profile_url="/talent/profile/page-one",
            source_string_id=owner.id,
            page=1,
        )
        rendered_page = 2

        async def finish_on_page_two(search_string, actual_progress):
            nonlocal rendered_page
            rendered_page = 2
            search_string.status = "done"
            p._resume_pending_full_decisions[pending.profile_url] = "FACIAL_YES"
            p._resume_pending_full_owner_ids[pending.profile_url] = owner.id
            p._resume_pending_full_snippets[pending.profile_url] = pending
            p._experiment_state_for(owner).active_variant.allocator_page_cursor = 3
            p._checkpoint_progress(actual_progress, search_string=search_string)

        async def reset_surface(*_args, **_kwargs):
            nonlocal rendered_page
            rendered_page = 1

        async def settle_only_from_page_one(**_kwargs):
            assert rendered_page == 1
            p._resume_pending_full_decisions.clear()
            p._resume_pending_full_owner_ids.clear()
            p._resume_pending_full_snippets.clear()
            return 1

        p._process_string = AsyncMock(side_effect=finish_on_page_two)
        p._apply_opening_search = AsyncMock(side_effect=reset_surface)
        p._recover_owner_pending_full_evaluations = AsyncMock(
            side_effect=settle_only_from_page_one
        )

        asyncio.run(p.run_full(resume=True))

        p._apply_opening_search.assert_awaited()
        assert owner.status == "done"


def test_run_full_terminalization_recovery_settles_owner_obligation():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=1,
            name="owner",
            boolean="one",
            status="in_progress",
        )
        progress = Progress(brief_name="test", strings=[owner])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[])
        pending = _make_snippet(
            name="Recovered Candidate",
            profile_url="/talent/profile/recovered-candidate",
            source_string_id=owner.id,
            page=1,
        )

        async def finish_with_obligation(search_string, _progress):
            search_string.status = "done"
            key = p._funnel_candidate_key(pending)
            p._resume_pending_full_decisions[key] = "FACIAL_YES"
            p._resume_pending_full_owner_ids[key] = owner.id
            p._resume_pending_full_snippets[key] = pending
            p._experiment_state_for(
                owner
            ).active_variant.allocator_page_cursor = 2

        async def settle_recovered(**kwargs):
            p._clear_resume_pending_full_if_settled(
                snippet=kwargs["snippets"][0],
                decision=OpusDecision(
                    stage="full",
                    decision="REJECT",
                    path="none",
                    confidence=0.9,
                    rationale="settled on retry",
                    candidate_name=pending.name,
                    profile_url=pending.profile_url,
                ),
            )
            return False

        p._process_string = AsyncMock(side_effect=finish_with_obligation)
        p.browser.find_result_slot_by_profile_url = AsyncMock(return_value=4)
        p._process_resumed_pending_full_evaluations = AsyncMock(
            side_effect=settle_recovered
        )

        asyncio.run(p.run_full(resume=True))

        assert owner.status == "done"
        assert not p._resume_pending_full_decisions
        p.browser.find_result_slot_by_profile_url.assert_awaited_once_with(
            pending.profile_url
        )


def test_run_full_terminalization_recovery_failure_names_candidate_and_stops_queue():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=1,
            name="owner",
            boolean="one",
            status="in_progress",
        )
        later = SearchString(
            id=2,
            name="later",
            boolean="two",
            status="queued",
        )
        progress = Progress(brief_name="test", strings=[owner, later])
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[])
        pending = _make_snippet(
            name="Named Outstanding Candidate",
            profile_url="/talent/profile/named-outstanding",
            source_string_id=owner.id,
            page=1,
        )

        async def finish_with_obligation(search_string, _progress):
            search_string.status = "done"
            key = p._funnel_candidate_key(pending)
            p._resume_pending_full_decisions[key] = "FACIAL_BORDERLINE"
            p._resume_pending_full_owner_ids[key] = owner.id
            p._resume_pending_full_snippets[key] = pending
            p._experiment_state_for(
                owner
            ).active_variant.allocator_page_cursor = 2

        p._process_string = AsyncMock(side_effect=finish_with_obligation)
        p.browser.find_result_slot_by_profile_url = AsyncMock(return_value=None)

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(p.run_full(resume=True))

        message = str(exc_info.value)
        assert "Named Outstanding Candidate" in message
        assert "/talent/profile/named-outstanding" in message
        assert owner.status == "in_progress"
        assert later.status == "queued"
        saved = json.loads(Path(td, "progress.json").read_text())
        assert [item["status"] for item in saved["strings"]] == [
            "in_progress",
            "queued",
        ]
        p._process_string.assert_awaited_once()


def test_sequential_page_resumes_pending_full_even_when_card_is_already_saved():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = False
        owner = SearchString(id=1, name="owner", boolean="one")
        pending = _make_snippet(
            name="Pending Save",
            profile_url="/talent/profile/pending-save",
            source_string_id=owner.id,
            source_string_name=owner.name,
            page=1,
        )
        pending.already_saved = True
        p._resume_pending_full_decisions = {
            pending.profile_url: "FACIAL_YES",
        }
        p._resume_pending_full_snippets = {
            pending.profile_url: pending,
        }
        p._resume_pending_full_owner_ids = {
            pending.profile_url: owner.id,
        }
        p._card_slot_count_or_raise = AsyncMock(return_value=1)
        p._extract_card_snippet = AsyncMock(return_value=pending)
        p._evaluate_snippet = AsyncMock(
            return_value=OpusDecision(
                stage="full",
                decision="REVIEW_INFERRED",
                path="DIRECT:resume",
                confidence=0.8,
                rationale="settled",
                candidate_name=pending.name,
                profile_url=pending.profile_url,
            )
        )
        p._checkpoint_progress = MagicMock()
        page_report = MagicMock()

        asyncio.run(
            p._review_page_sequentially(
                owner,
                1,
                25,
                page_report,
                [],
                p._fresh_string_stats(),
            )
        )

        p._evaluate_snippet.assert_awaited_once()
        assert not any(
            call.args == (pending.name, "already_saved")
            for call in page_report.add_skip_preview.call_args_list
        )


def test_run_full_real_path_keeps_foreign_pending_inert_until_owner_surface():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        first = SearchString(
            id=1,
            name="string-one",
            boolean="one",
            status="in_progress",
            pages_reviewed=1,
        )
        second = SearchString(
            id=2,
            name="string-two",
            boolean="two",
            status="queued",
        )
        progress = Progress(
            brief_name="test",
            strings=[first, second],
            current_string_id=first.id,
            current_page=2,
        )
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        _wire_real_run_full_page_path(p, result_counts=[50, 50])

        pending = [
            _make_snippet(
                name=f"String Two Pending {index}",
                profile_url=f"/talent/profile/string-two-{index}",
                source_string_id=second.id,
                source_string_name=second.name,
                page=1,
                result_rank=index,
            )
            for index in range(1, 18)
        ]

        def hydrate(_progress):
            p._resume_pending_full_decisions = {
                snippet.profile_url: "FACIAL_YES"
                for snippet in pending
            }
            p._resume_pending_full_snippets = {
                snippet.profile_url: snippet
                for snippet in pending
            }
            p._resume_pending_full_owner_ids = {
                snippet.profile_url: second.id
                for snippet in pending
            }
            first_state = bootstrap_experiment_state(first)
            first_state.active_variant.allocator_page_cursor = 2
            second_state = bootstrap_experiment_state(second)
            second_state.active_variant.allocator_page_cursor = 1
            p._experiment_states = {
                first.id: first_state,
                second.id: second_state,
            }

        p._hydrate_resume_funnel_from_runtime = MagicMock(side_effect=hydrate)
        p._card_slot_count_or_raise = AsyncMock(side_effect=[0, 0])
        p._evaluate_snippet = AsyncMock()
        p._assess_string_state = AsyncMock(
            return_value={
                "decision": "stop",
                "rationale": "String 1 adaptively complete",
                "page": 2,
            }
        )

        events: list[str] = []

        async def enter_search(boolean):
            events.append(f"surface:{boolean}")

        p.browser.enter_search_string.side_effect = enter_search
        p.browser.go_to_next_page.side_effect = lambda: (
            events.append("page-turn:string-one") or True
        )
        p.browser.find_result_slot_by_profile_url = AsyncMock(return_value=None)
        original_checkpoint = p._checkpoint_progress

        def checkpoint(*args, **kwargs):
            if (
                first.status in {"done", "skipped"}
                and "terminal:string-one" not in events
            ):
                events.append("terminal:string-one")
            return original_checkpoint(*args, **kwargs)

        p._checkpoint_progress = checkpoint

        with pytest.raises(RuntimeError, match="outstanding full review"):
            asyncio.run(p.run_full(resume=True))

        assert events.index("surface:one") < events.index("terminal:string-one")
        assert events.index("terminal:string-one") < events.index("surface:two")
        assert len(p._resume_pending_full_decisions) == 17
        assert p._page_observation()["full_expected"] == 17
        assert p._page_observation()["full_settled"] == 0
        p._evaluate_snippet.assert_not_awaited()
        saved = json.loads(Path(td, "progress.json").read_text())
        assert [item["status"] for item in saved["strings"]] == [
            "done",
            "in_progress",
        ]
        with sqlite3.connect(Path(td, "runtime_state.sqlite3")) as conn:
            allocator_events = conn.execute(
                "SELECT event_type FROM events "
                "WHERE event_type LIKE 'page_allocator_%'"
            ).fetchall()
        assert allocator_events == []


def test_run_full_resume_preserves_persisted_queue_order():
    """Resume should preserve the saved queue order instead of sorting by ID."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        from shared.schemas import Progress, SearchString

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=2, name="second", boolean="two"),
                SearchString(id=1, name="first", boolean="one"),
            ],
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._load_candidate_history = MagicMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        _allow_synthetic_run_completion(p)
        p._session_expired = MagicMock()

        processed_ids = []

        async def fake_process(search_string, progress):
            processed_ids.append(search_string.id)

        p._process_string = fake_process

        asyncio.run(p.run_full(resume=True))

        assert processed_ids == [2, 1]


def test_run_full_resume_never_executes_foreign_pending_before_active_owner():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        current = SearchString(
            id=1,
            name="string-one",
            boolean="a",
            status="in_progress",
        )
        later = SearchString(
            id=2,
            name="string-two",
            boolean="b",
            status="queued",
        )
        progress = Progress(
            brief_name="test",
            strings=[current, later],
            current_string_id=current.id,
            current_page=2,
        )
        progress.save(str(p.progress_path))
        p._load_or_create_progress = MagicMock(return_value=progress)
        p._load_candidate_history = MagicMock()
        p._load_search_memory = MagicMock()
        foreign = _make_snippet(
            name="String Two Pending",
            profile_url="/talent/profile/string-two-pending",
            source_string_id=later.id,
            source_string_name=later.name,
        )

        def hydrate(_progress):
            p._resume_pending_full_decisions = {
                foreign.profile_url: "FACIAL_YES",
            }
            p._resume_pending_full_snippets = {foreign.profile_url: foreign}
            p._resume_pending_full_owner_ids = {foreign.profile_url: later.id}

        p._hydrate_resume_funnel_from_runtime = MagicMock(side_effect=hydrate)
        p._process_resumed_pending_full_evaluations = AsyncMock()
        p._checkpoint_progress = MagicMock()
        p._enforce_constraint_manifest = MagicMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._session_expired = MagicMock()
        p.browser.enter_search_string = AsyncMock()
        p.browser.open_profile_by_url = AsyncMock()
        p.browser.go_to_next_page = AsyncMock()
        p.browser.save_candidate = AsyncMock()

        async def fail_active_owner(search_string, _progress):
            await p.browser.enter_search_string(search_string.boolean)
            await p.browser.open_profile_by_url(
                f"/talent/profile/string-{search_string.id}"
            )
            await p.browser.go_to_next_page()
            await p.browser.save_candidate()
            raise RuntimeError("active owner failed")

        p._process_string = fail_active_owner

        with pytest.raises(RuntimeError, match="active owner failed"):
            asyncio.run(p.run_full(resume=True))

        p._process_resumed_pending_full_evaluations.assert_not_awaited()
        p.browser.enter_search_string.assert_awaited_once_with("a")
        p.browser.open_profile_by_url.assert_awaited_once_with(
            "/talent/profile/string-1"
        )
        p.browser.go_to_next_page.assert_awaited_once()
        p.browser.save_candidate.assert_awaited_once()
        assert later.status == "queued"


def test_run_full_transient_pagination_bypasses_completion_footer():
    from linkedin.orchestrator import TransientPaginationError

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        current = SearchString(id=111, name="A", boolean="a")
        progress = Progress(brief_name="test", strings=[current])
        progress.save(str(p.progress_path))
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._session_expired = MagicMock()
        p._process_string = AsyncMock(
            side_effect=TransientPaginationError("transient pagination retry")
        )

        with pytest.raises(TransientPaginationError, match="transient pagination"):
            asyncio.run(p.run_full(resume=True))

        assert p._progress is not None
        assert p._progress.strings[0].status == "in_progress"
        assert not any(
            row.get("event") == "string_complete"
            and row.get("string_id") == current.id
            for row in read_jsonl(p.log_path)
        )


def test_run_full_active_switch_hands_off_to_selected_root_without_completing_paused_root():
    from linkedin.page_allocator import AllocationAction, AllocationVerdict

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="A", boolean="a", block="Block A"),
                SearchString(id=2, name="B", boolean="b", block="Block A"),
            ],
        )
        progress.save(str(p.progress_path))
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._allocator_active_enabled = MagicMock(return_value=True)
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._session_expired = MagicMock()
        processed_ids = []

        async def fake_process(search_string, live_progress):
            processed_ids.append(search_string.id)
            if search_string.id == 1:
                selected = next(item for item in live_progress.strings if item.id == 2)
                search_string.status = "queued"
                selected.status = "in_progress"
                live_progress.strings[:] = [selected, search_string]
                live_progress.current_string_id = selected.id
                return AllocationVerdict(
                    action=AllocationAction.SWITCH,
                    current_root_id=search_string.id,
                    selected_root_id=selected.id,
                    reason="opening_probe",
                    paused_root_ids=(search_string.id,),
                    ranked_root_ids=(selected.id, search_string.id),
                )
            raise SessionExpired("stop after selected root handoff")

        p._process_string = fake_process

        with pytest.raises(SessionExpired, match="selected root handoff"):
            asyncio.run(p.run_full(resume=True))

        saved = json.loads(Path(td, "progress.json").read_text())
        assert processed_ids == [1, 2]
        assert [(item["id"], item["status"]) for item in saved["strings"]] == [
            (2, "in_progress"),
            (1, "queued"),
        ]
        assert saved["current_string_id"] == 2
        assert not any(
            row.get("event") == "string_complete" and row.get("string_id") == 1
            for row in read_jsonl(p.log_path)
        )
        p._print_session_summary.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        AllocatorPolicyError("allocator pre-image mismatch"),
        RuntimeError("allocator actuation sync failed"),
    ],
    ids=["policy", "raw-sync"],
)
def test_run_full_active_error_keeps_root_retryable_and_stops_run(failure):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1,
                    name="A",
                    boolean="a",
                    status="in_progress",
                    block="Block A",
                    pages_reviewed=2,
                ),
                SearchString(id=2, name="B", boolean="b", block="Block A"),
            ],
            current_string_id=1,
            current_page=2,
        )
        progress.save(str(p.progress_path))
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._allocator_active_enabled = MagicMock(return_value=True)
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._session_expired = MagicMock()
        processed_ids = []

        async def fail_allocator(search_string, _progress):
            processed_ids.append(search_string.id)
            raise failure

        p._process_string = fail_allocator

        with pytest.raises(type(failure), match=str(failure)):
            asyncio.run(p.run_full(resume=True))

        saved = json.loads(Path(td, "progress.json").read_text())
        assert processed_ids == [1]
        assert saved["current_string_id"] == 1
        assert saved["current_page"] == 2
        assert [(item["id"], item["status"]) for item in saved["strings"]] == [
            (1, "in_progress"),
            (2, "queued"),
        ]
        assert not any(
            row.get("event") in {"string_error", "string_complete"}
            and row.get("string_id") == 1
            for row in read_jsonl(p.log_path)
        )
        with sqlite3.connect(Path(td, "runtime_state.sqlite3")) as conn:
            work_units = conn.execute(
                "SELECT source_unit_id, status FROM work_units "
                "WHERE kind = 'linkedin_string' ORDER BY ordering_index"
            ).fetchall()
            run = conn.execute(
                "SELECT status, stop_reason FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert work_units == [("1", "in_progress"), ("2", "queued")]
        assert run == ("error", "fatal_runtime_error")


def test_run_full_string_complete_ignores_shadowing_string_id_in_stats():
    """A stale stats key cannot turn a completed string into a runtime error."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=17,
                    name="completed",
                    boolean="one",
                    status="queued",
                )
            ],
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._load_candidate_history = MagicMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        _allow_synthetic_run_completion(p)
        p._session_expired = MagicMock()
        p._process_string = AsyncMock()
        p.stats["string_id"] = 999
        p.stats["saved"] = 1

        with patch(
            "linkedin.orchestrator.config.LINKEDIN_PAGE_ALLOCATOR_MODE",
            "off",
        ):
            asyncio.run(p.run_full(resume=True))

        run_events = read_jsonl(p.log_path)
        completed = [
            row for row in run_events if row.get("event") == "string_complete"
        ]
        assert len(completed) == 1
        assert completed[0]["string_id"] == 17
        assert p._progress.strings[0].status == "done"
        event_names = {str(row.get("event") or "") for row in run_events}
        assert {
            "pipeline_start",
            "string_started",
            "string_complete",
            "pipeline_end",
        } <= event_names
        assert not any(
            event.startswith("page_allocator_")
            or event
            in {
                "allocator_selection",
                "allocator_actuation",
                "allocator_recovery",
            }
            for event in event_names
        )


def test_run_full_mid_string_exception_aborts_and_keeps_owner_resumable():
    """Unexpected owner failures cannot release the next search string."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=5, name="bad", boolean="one", status="queued"),
                SearchString(id=6, name="next", boolean="two", status="queued"),
            ],
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._load_candidate_history = MagicMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._session_expired = MagicMock()

        processed_ids = []

        async def fake_process(search_string, progress):
            processed_ids.append(search_string.id)
            if search_string.id == 5:
                raise RuntimeError("mid-string boom")

        p._process_string = fake_process

        with pytest.raises(RuntimeError, match="mid-string boom"):
            asyncio.run(p.run_full(resume=True))

        saved = json.loads(Path(td, "progress.json").read_text())
        assert processed_ids == [5]
        assert saved["strings"][0]["status"] == "in_progress"
        assert "mid-string boom" in saved["strings"][0]["notes"]
        assert saved["strings"][1]["status"] == "queued"

        with sqlite3.connect(Path(td, "runtime_state.sqlite3")) as conn:
            rows = conn.execute(
                """
                SELECT source_unit_id, status, notes
                FROM work_units
                WHERE kind = 'linkedin_string'
                ORDER BY ordering_index
                """
            ).fetchall()
        assert [(row[0], row[1]) for row in rows] == [
            ("5", "in_progress"),
            ("6", "queued"),
        ]
        assert "mid-string boom" in rows[0][2]


def test_run_full_pipeline_error_records_traceback(tmp_path):
    p = _make_pipeline(str(tmp_path))
    p.browser.connect = AsyncMock()
    p.browser.disconnect = AsyncMock()
    p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
    p._print_summary = MagicMock()
    p._finalize_run_snapshot = MagicMock(return_value=tmp_path / "frozen")

    async def raise_from_guarded_pipeline():
        raise RuntimeError("pipeline traceback sentinel")

    p._apply_session_location_filter = raise_from_guarded_pipeline

    with pytest.raises(RuntimeError, match="pipeline traceback sentinel"):
        asyncio.run(p.run_full())

    pipeline_error = next(
        event for event in read_jsonl(p.log_path) if event.get("event") == "pipeline_error"
    )
    assert pipeline_error["error"] == "pipeline traceback sentinel"
    assert "Traceback (most recent call last)" in pipeline_error["traceback"]
    assert "raise_from_guarded_pipeline" in pipeline_error["traceback"]


@pytest.mark.parametrize("resumable_status", ["error", "legacy-unknown"])
def test_run_full_resume_executes_any_nonterminal_status_before_later_strings(
    resumable_status,
):
    """Only done/skipped release later strings in the standard path."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=5,
                    name="resume-first",
                    boolean="one",
                    status=resumable_status,
                ),
                SearchString(id=6, name="next", boolean="two", status="queued"),
            ],
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._load_candidate_history = MagicMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        _allow_synthetic_run_completion(p)
        p._session_expired = MagicMock()

        processed_ids = []

        async def fake_process(search_string, progress):
            processed_ids.append(search_string.id)

        p._process_string = fake_process

        asyncio.run(p.run_full(resume=True))

        assert processed_ids == [5, 6]


def test_run_block_adaptation_pivot_keeps_inserted_replacements_queued():
    """Architecture pivots should skip the old queue, not the newly inserted replacements."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)

        from shared.schemas import Progress, SearchString

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="done", boolean="one", status="done", block="Block A"),
                SearchString(id=2, name="old queued", boolean="two", block="Block A"),
                SearchString(id=3, name="later queued", boolean="three", block="Block B"),
            ],
        )
        p._execution_plan = ExecutionPlan(
            strategy_rationale="test",
            architecture="sniper",
            original_architecture="sniper",
        )

        response = AdaptationResponse(
            new_strings=[
                {"boolean": "replacement one", "rationale": "replacement one"},
                {"boolean": "replacement two", "rationale": "replacement two"},
            ],
            pivot_to_architecture="company_first",
            pivot_rationale="switch architectures",
        )

        asyncio.run(
            p._run_block_adaptation(
                "Block A",
                [progress.strings[0]],
                progress,
                lambda *args, **kwargs: response,
            )
        )

        assert [(s.id, s.status) for s in progress.strings] == [
            (1, "done"),
            (4, "queued"),
            (5, "queued"),
            (2, "skipped"),
            (3, "skipped"),
        ]


def test_run_full_rechecks_queue_after_block_adaptation():
    """When adaptation mutates the queue, run_full should process the replacement string next."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)

        from shared.schemas import Progress, SearchString

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="block a", boolean="one", block="Block A"),
                SearchString(id=2, name="block b", boolean="two", block="Block B"),
            ],
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._load_candidate_history = MagicMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        _allow_synthetic_run_completion(p)
        p._session_expired = MagicMock()
        p._execution_plan = ExecutionPlan(
            strategy_rationale="test",
            architecture="sniper",
            original_architecture="sniper",
        )

        processed_ids = []

        async def fake_process(search_string, progress):
            processed_ids.append(search_string.id)

        p._process_string = fake_process

        responses = iter([
            AdaptationResponse(
                new_strings=[{"boolean": "replacement", "rationale": "replacement"}],
                skip_remaining=[{"string_id": 2, "reason": "replace old next string"}],
                pivot_to_architecture="company_first",
                pivot_rationale="switch architectures",
            ),
            AdaptationResponse(),
        ])

        def fake_adapt(*args, **kwargs):
            return next(responses)

        with patch("linkedin.strategy.adapt_after_block", side_effect=fake_adapt):
            asyncio.run(p.run_full(resume=True))

        assert processed_ids == [1, 3]


def test_run_full_resume_preserves_in_progress_string_on_session_expiry():
    """SessionExpired should checkpoint and bubble out without downgrading the interrupted string."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=5,
                    name="interrupted",
                    boolean="one",
                    status="in_progress",
                    block="Block A",
                    pages_reviewed=1,
                ),
                SearchString(id=6, name="next", boolean="two", status="queued", block="Block B"),
            ],
            current_string_id=5,
            current_page=1,
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()

        processed_ids = []

        async def fake_process(search_string, progress):
            processed_ids.append(search_string.id)
            raise SessionExpired("session_duration_cap")

        p._process_string = fake_process

        with pytest.raises(SessionExpired):
            asyncio.run(p.run_full(resume=True))

        saved = json.loads(Path(td, "progress.json").read_text())
        assert processed_ids == [5]
        assert saved["current_string_id"] == 5
        assert saved["current_page"] == 1
        assert saved["strings"][0]["status"] == "in_progress"
        assert saved["strings"][1]["status"] == "queued"


def test_evaluate_snippet_reraises_api_budget_exhaustion_from_facial_judge():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        snippet = _make_snippet()
        p._tightening_prefix = ""

        with patch(
            "linkedin.orchestrator.facial_judge",
            side_effect=RuntimeError(
                "Your credit balance is too low to access the Anthropic API. "
                "Please go to Plans & Billing to upgrade or purchase credits."
            ),
        ):
            with pytest.raises(ApiBudgetExhaustedError):
                asyncio.run(p._evaluate_snippet(snippet))


def test_interrupted_full_evaluation_leaves_complete_pending_ledger_entry():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(id=1, name="owner", boolean="one")
        later = SearchString(id=2, name="later", boolean="two")
        progress = Progress(brief_name="test", strings=[owner, later])
        snippet = _make_snippet(source_string_id=owner.id)
        p._tightening_prefix = ""
        facial = OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="DIRECT:test",
            confidence=0.9,
            rationale="full review required",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        p._full_evaluate = AsyncMock(
            side_effect=OperatorStopRequested("operator stop")
        )

        with patch("linkedin.orchestrator.facial_judge", return_value=facial):
            with pytest.raises(OperatorStopRequested):
                asyncio.run(
                    p._evaluate_snippet(
                        snippet,
                        search_string=owner,
                        string_stats=p._fresh_string_stats(),
                    )
                )

        key = p._funnel_candidate_key(snippet)
        assert p._resume_pending_full_decisions[key] == "FACIAL_YES"
        assert p._resume_pending_full_owner_ids[key] == owner.id
        assert p._resume_pending_full_snippets[key] is snippet
        assert p._validated_owner_pending_full_snippets(
            progress=progress,
            search_string=later,
        ) == []


def test_run_full_resume_preserves_in_progress_string_on_api_budget_exhaustion():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=5,
                    name="interrupted",
                    boolean="one",
                    status="in_progress",
                    block="Block A",
                    pages_reviewed=3,
                ),
                SearchString(id=6, name="next", boolean="two", status="queued", block="Block B"),
            ],
            current_string_id=5,
            current_page=3,
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()

        async def fake_process(search_string, progress):
            raise ApiBudgetExhaustedError(
                "Your credit balance is too low to access the Anthropic API."
            )

        p._process_string = fake_process

        with pytest.raises(ApiBudgetExhaustedError):
            asyncio.run(p.run_full(resume=True))

        saved = json.loads(Path(td, "progress.json").read_text())
        assert saved["current_string_id"] == 5
        assert saved["current_page"] == 3
        assert saved["strings"][0]["status"] == "in_progress"
        assert saved["strings"][1]["status"] == "queued"

        runtime_db = Path(td, "runtime_state.sqlite3")
        with sqlite3.connect(runtime_db) as conn:
            row = conn.execute(
                "SELECT status, stop_reason FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row == ("interrupted", "api_budget_exhausted")


def test_run_full_panel_recovery_error_preserves_string_for_resume():
    """Exhausted panel recovery is run-level and cannot advance the queue."""

    from linkedin.orchestrator import PanelRecoveryError

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=5,
                    name="interrupted",
                    boolean="one",
                    status="in_progress",
                    block="Block A",
                    pages_reviewed=2,
                ),
                SearchString(
                    id=6,
                    name="next",
                    boolean="two",
                    status="queued",
                    block="Block B",
                ),
            ],
            current_string_id=5,
            current_page=2,
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        processed_ids = []

        async def fail_panel_recovery(search_string, progress):
            processed_ids.append(search_string.id)
            raise PanelRecoveryError("profile panel recovery failed")

        p._process_string = fail_panel_recovery

        with pytest.raises(PanelRecoveryError):
            asyncio.run(p.run_full(resume=True))

        saved = json.loads(Path(td, "progress.json").read_text())
        assert processed_ids == [5]
        assert saved["current_string_id"] == 5
        assert saved["current_page"] == 2
        assert saved["strings"][0]["status"] == "in_progress"
        assert saved["strings"][1]["status"] == "queued"
        assert not any(
            row.get("event") == "string_error"
            for row in read_jsonl(p.log_path)
        )


def test_run_full_browser_crash_recovers_then_retries_same_owner():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _allow_synthetic_run_completion(p)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=4,
                    name="completed before recovery",
                    boolean="zero",
                    status="queued",
                    block="Block A",
                ),
                SearchString(
                    id=5,
                    name="interrupted",
                    boolean="one",
                    status="queued",
                    block="Block A",
                ),
            ],
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        snapshot = MagicMock(name="snapshot")
        order = []
        p._capture_recovery_snapshot = AsyncMock(
            side_effect=lambda *_args, **_kwargs: order.append("capture") or snapshot
        )
        p._recovery_service.recover = AsyncMock(
            side_effect=lambda **_kwargs: order.append("recover") or True
        )
        p._reassert_session_location_after_recovery = AsyncMock(
            side_effect=lambda: order.append("reassert")
        )
        start_or_resume_run = MagicMock(
            wraps=p._runtime_bridge.start_or_resume_run
        )
        p._runtime_bridge.start_or_resume_run = start_or_resume_run
        p._run_block_adaptation = AsyncMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()

        calls = {}

        async def fake_process(search_string, progress):
            calls[search_string.id] = calls.get(search_string.id, 0) + 1
            order.append(f"process:{search_string.id}:{calls[search_string.id]}")
            if search_string.id == 5 and calls[search_string.id] == 1:
                raise RuntimeError("Page.evaluate: Target crashed")
            if search_string.id == 5:
                search_string.pages_reviewed = 2

        p._process_string = fake_process

        asyncio.run(p.run_full(resume=True))

        saved = json.loads(Path(td, "progress.json").read_text())
        assert order == [
            "process:4:1",
            "process:5:1",
            "capture",
            "recover",
            "reassert",
            "process:5:2",
        ]
        p._recovery_service.recover.assert_awaited_once_with(
            run_id=p._runtime_run_id,
            snapshot=snapshot,
        )
        start_or_resume_run.assert_called_once_with(resume=True)
        assert [item["status"] for item in saved["strings"]] == ["done", "done"]
        assert saved["strings"][1]["pages_reviewed"] == 2
        adapted_strings = p._run_block_adaptation.await_args.args[1]
        assert len(adapted_strings) == 2
        assert all(
            adapted is canonical
            for adapted, canonical in zip(adapted_strings, p._progress.strings)
        )


@pytest.mark.parametrize(
    ("capture_error", "recovery_error", "secondary_message"),
    [
        (RuntimeError("capture failed"), None, "capture failed"),
        (None, None, "browser recovery failed"),
        (None, RuntimeError("recovery failed"), "recovery failed"),
    ],
    ids=["capture-raised", "returned-false", "recovery-raised"],
)
def test_run_full_browser_recovery_abandonment_skips_report_and_clears_sidecar(
    capture_error,
    recovery_error,
    secondary_message,
):
    """Unrecovered browser loss should ground the run without debrief linger."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        progress = Progress(
            brief_name="test",
            strings=[SearchString(id=5, name="interrupted", boolean="one", status="queued", block="Block A")],
        )
        progress.save(str(p.progress_path))
        Path(td, "worker.json").write_text(json.dumps({"pid": os.getpid()}))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        snapshot = MagicMock(name="snapshot")
        p._capture_recovery_snapshot = AsyncMock(
            return_value=snapshot,
            side_effect=capture_error,
        )
        p._recovery_service.recover = AsyncMock(
            return_value=False,
            side_effect=recovery_error,
        )
        p._reassert_session_location_after_recovery = AsyncMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._finalize_run_snapshot = MagicMock(return_value=Path(td, "frozen"))

        failure = RuntimeError(
            "Page.evaluate: Target page, context or browser has been closed"
        )

        async def fake_process(search_string, progress):
            raise failure

        p._process_string = fake_process

        with pytest.raises(RuntimeError, match="Target page") as raised:
            asyncio.run(p.run_full(resume=True))

        assert raised.value is failure
        p._capture_recovery_snapshot.assert_awaited_once()
        if capture_error is None:
            p._recovery_service.recover.assert_awaited_once_with(
                run_id=p._runtime_run_id,
                snapshot=snapshot,
            )
        else:
            p._recovery_service.recover.assert_not_awaited()
        p._reassert_session_location_after_recovery.assert_not_awaited()
        p._generate_run_report.assert_not_called()
        assert not Path(td, "worker.json").exists()
        assert any(
            secondary_message in note
            for note in getattr(raised.value, "__notes__", [])
        )
        pipeline_error = next(
            row for row in read_jsonl(p.log_path) if row["event"] == "pipeline_error"
        )
        assert secondary_message in pipeline_error["traceback"]
        with sqlite3.connect(Path(td, "runtime_state.sqlite3")) as conn:
            row = conn.execute(
                "SELECT status, stop_reason FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row == ("interrupted", "browser_disconnect_unrecovered")


@pytest.mark.parametrize(
    ("failure_stage", "secondary_error", "expected_row"),
    [
        (
            "reassert",
            RuntimeError("Page.evaluate: Target crashed during reassert"),
            ("interrupted", "browser_disconnect_unrecovered"),
        ),
        (
            "reload",
            RuntimeError("canonical reload failed"),
            ("error", "fatal_runtime_error"),
        ),
    ],
)
def test_run_full_post_recovery_failure_preserves_cause_and_stop_reason(
    failure_stage,
    secondary_error,
    expected_row,
):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=5,
                    name="interrupted",
                    boolean="one",
                    status="queued",
                    block="Block A",
                )
            ],
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        snapshot = MagicMock(name="snapshot")
        p._capture_recovery_snapshot = AsyncMock(return_value=snapshot)
        p._recovery_service.recover = AsyncMock(return_value=True)
        p._reassert_session_location_after_recovery = AsyncMock(
            side_effect=secondary_error if failure_stage == "reassert" else None
        )
        load_progress = MagicMock(wraps=p._runtime_bridge.load_progress)
        if failure_stage == "reload":
            load_progress.side_effect = secondary_error
        p._runtime_bridge.load_progress = load_progress
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._finalize_run_snapshot = MagicMock(return_value=Path(td, "frozen"))

        failure = RuntimeError("Page.evaluate: Target crashed")
        p._process_string = AsyncMock(side_effect=failure)

        with pytest.raises(RuntimeError, match="Target crashed") as raised:
            asyncio.run(p.run_full(resume=True))

        assert raised.value is failure
        assert any(
            str(secondary_error) in note
            for note in getattr(raised.value, "__notes__", [])
        )
        p._recovery_service.recover.assert_awaited_once_with(
            run_id=p._runtime_run_id,
            snapshot=snapshot,
        )
        p._reassert_session_location_after_recovery.assert_awaited_once_with()
        if failure_stage == "reassert":
            load_progress.assert_not_called()
        pipeline_error = next(
            row for row in read_jsonl(p.log_path) if row["event"] == "pipeline_error"
        )
        assert str(secondary_error) in pipeline_error["traceback"]
        with sqlite3.connect(Path(td, "runtime_state.sqlite3")) as conn:
            row = conn.execute(
                "SELECT status, stop_reason FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row == expected_row


def test_run_full_resume_executes_pending_block_adaptation_before_next_string():
    """A completed block that never adapted must adapt before the next queued string runs."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="done", boolean="one", status="done", block="Block A"),
                SearchString(id=2, name="next", boolean="two", status="queued", block="Block B"),
            ],
            pending_block_name="Block A",
            pending_block_string_ids=[1],
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        _allow_synthetic_run_completion(p)
        p._execution_plan = ExecutionPlan(strategy_rationale="test")

        call_order = []

        async def fake_process(search_string, progress):
            call_order.append(f"process:{search_string.id}")

        def fake_adapt(*args, **kwargs):
            call_order.append("adapt")
            return AdaptationResponse()

        p._process_string = fake_process

        with patch("linkedin.strategy.adapt_after_block", side_effect=fake_adapt):
            asyncio.run(p.run_full(resume=True))

        saved = json.loads(Path(td, "progress.json").read_text())
        assert call_order == ["adapt", "process:2"]
        assert saved["pending_block_name"] == ""
        assert saved["pending_block_string_ids"] == []


def test_run_block_adaptation_defers_when_signal_gate_collects_more_signal():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="done", boolean="one", status="done", block="Block A"),
                SearchString(id=2, name="next", boolean="two", status="queued", block="Block B"),
            ],
            pending_block_name="Block A",
            pending_block_string_ids=[1],
            pending_block_ready=True,
        )
        called = False

        def fake_adapt(*args, **kwargs):
            nonlocal called
            called = True
            return AdaptationResponse()

        asyncio.run(
            p._run_block_adaptation(
                "Block A",
                [progress.strings[0]],
                progress,
                fake_adapt,
            )
        )

        assert called is False
        assert progress.strings[1].status == "queued"
        assert progress.pending_block_name == ""
        assert progress.pending_block_string_ids == []
        events = read_jsonl(p.log_path)
        decision = next(event for event in events if event["event"] == "adaptation_decision")
        assert decision["decision"] == "collect_more_signal"
        assert decision["gate_config"] == {
            "min_strings": 1,
            "min_candidates_seen": 1,
            "min_results_seen": 1,
            "cooldown_blocks_remaining": 0,
            "allow_autonomous_reset": False,
            "sprt_lower": None,
            "sprt_upper": None,
        }
        assert any("candidates_seen" in reason for reason in decision["reasons"])


def test_process_string_continues_pagination_instead_of_forced_narrow_below_min_pages():
    """Low-signal pagination keeps paging until the minimum depth, then stops cleanly."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        search_string = SearchString(id=5, name="test", boolean="foo", status="queued")
        progress = Progress(brief_name="test", strings=[search_string], current_string_id=5, current_page=0)

        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results

        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p.browser.page.locator.return_value = locator
        p.browser.enter_search_string = AsyncMock()
        p.browser.get_results_count_text = AsyncMock(return_value="144")
        p.browser.get_results_count = AsyncMock(return_value=144)
        p.browser.go_to_next_page = AsyncMock(return_value=True)

        p._ensure_browser_healthy = AsyncMock()
        p._review_page_sequentially = AsyncMock(return_value=None)
        p._assess_string_state = AsyncMock(
            side_effect=[
                {"decision": "continue", "rationale": "keep paging", "page": 1},
                {"decision": "stop", "rationale": "signal exhausted", "page": 2},
            ]
        )
        # This test owns the assessment sequence; variant lifecycle scoring
        # is covered separately with settled full-profile outcomes.
        p._evaluate_variant_lifecycle = MagicMock(return_value=None)
        p._plan_variant_experiments = AsyncMock()

        with patch("linkedin.orchestrator.human_delay_correlated", return_value=0):
            asyncio.run(p._process_string(search_string, progress))

        assert p._assess_string_state.await_count == 2
        assert p._plan_variant_experiments.await_count == 0
        p.browser.go_to_next_page.assert_awaited_once()
        assert "Stopped after page 2." in (search_string.notes or "")


def test_process_string_checkpoints_started_page_before_review_and_decided_completion():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(id=6, name="test", boolean="foo")
        progress = Progress(brief_name="test", strings=[search_string])
        state = bootstrap_experiment_state(search_string)
        p._experiment_states[search_string.id] = state

        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p.browser.page.locator.return_value = locator
        p.browser.enter_search_string = AsyncMock()
        p.browser.get_results_count_text = AsyncMock(return_value="100")
        p.browser.get_results_count = AsyncMock(return_value=100)
        p._ensure_browser_healthy = AsyncMock()
        p._review_page_sequentially = AsyncMock(return_value=None)
        p._evaluate_variant_lifecycle = MagicMock(return_value=None)
        p._assess_string_state = AsyncMock(
            return_value={"decision": "stop", "rationale": "done", "page": 1}
        )

        checkpoints = []

        def capture_checkpoint(*_args, **_kwargs):
            checkpoints.append(
                (
                    state.active_variant.allocator_page_cursor,
                    search_string.status,
                )
            )

        p._checkpoint_progress = MagicMock(side_effect=capture_checkpoint)

        asyncio.run(p._process_string(search_string, progress))

        assert checkpoints[0] == (1, "queued")
        assert checkpoints[-1] == (2, "done")
        assert state.active_variant.allocator_page_cursor == 2
        p._assess_string_state.assert_awaited_once()


def test_process_string_partial_page_keeps_cursor_on_started_page():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(id=8, name="test", boolean="foo")
        progress = Progress(brief_name="test", strings=[search_string])
        state = bootstrap_experiment_state(search_string)
        p._experiment_states[search_string.id] = state

        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p.browser.page.locator.return_value = locator
        p.browser.enter_search_string = AsyncMock()
        p.browser.get_results_count_text = AsyncMock(return_value="100")
        p.browser.get_results_count = AsyncMock(return_value=100)
        p._ensure_browser_healthy = AsyncMock()
        p._review_page_sequentially = AsyncMock(
            side_effect=RuntimeError("crash during page")
        )
        p._checkpoint_progress = MagicMock()

        with pytest.raises(RuntimeError, match="crash during page"):
            asyncio.run(p._process_string(search_string, progress))

        assert state.active_variant.allocator_page_cursor == 1
        assert p._checkpoint_progress.call_count == 1


def test_process_string_resumes_at_first_incomplete_page_from_canonical_cursor():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(
            id=10,
            name="test",
            boolean="foo",
            pages_reviewed=3,
        )
        progress = Progress(
            brief_name="test",
            strings=[search_string],
            current_string_id=10,
            current_page=3,
        )
        state = bootstrap_experiment_state(search_string)
        state.active_variant.allocator_page_cursor = 4
        p._experiment_states[search_string.id] = state

        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/123/search"
        )
        p.browser.page.locator.return_value = locator
        p.browser.navigate_to_search = AsyncMock()
        p.browser.enter_search_string = AsyncMock()
        p.browser.get_results_count_text = AsyncMock(return_value="100")
        p.browser.get_results_count = AsyncMock(return_value=100)
        p.browser.go_to_next_page = AsyncMock(return_value=True)
        p._ensure_browser_healthy = AsyncMock()

        reviewed_pages = []

        async def review_page(**kwargs):
            reviewed_pages.append(kwargs["page_num"])
            return None

        p._review_page_sequentially = AsyncMock(side_effect=review_page)
        p._evaluate_variant_lifecycle = MagicMock(return_value=None)
        p._assess_string_state = AsyncMock(
            return_value={"decision": "stop", "rationale": "done", "page": 4}
        )
        p._checkpoint_progress = MagicMock()

        with patch("linkedin.orchestrator.human_delay_correlated", return_value=0):
            asyncio.run(p._process_string(search_string, progress))

        assert reviewed_pages == [4]
        assert p.browser.go_to_next_page.await_count == 3
        assert state.active_variant.allocator_page_cursor == 5


def test_process_string_legacy_checkpoint_falls_back_to_pages_reviewed():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(
            id=11,
            name="test",
            boolean="foo",
            pages_reviewed=3,
        )
        progress = Progress(
            brief_name="test",
            strings=[search_string],
            current_string_id=11,
            current_page=3,
        )
        state = bootstrap_experiment_state(search_string)
        assert state.active_variant.allocator_page_cursor == 0
        p._experiment_states[search_string.id] = state

        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/123/search"
        )
        p.browser.page.locator.return_value = locator
        p.browser.navigate_to_search = AsyncMock()
        p.browser.enter_search_string = AsyncMock()
        p.browser.get_results_count_text = AsyncMock(return_value="100")
        p.browser.get_results_count = AsyncMock(return_value=100)
        p.browser.go_to_next_page = AsyncMock(return_value=True)
        p._ensure_browser_healthy = AsyncMock()

        reviewed_pages = []

        async def review_page(**kwargs):
            reviewed_pages.append(kwargs["page_num"])
            return None

        p._review_page_sequentially = AsyncMock(side_effect=review_page)
        p._evaluate_variant_lifecycle = MagicMock(return_value=None)
        p._assess_string_state = AsyncMock(
            return_value={"decision": "stop", "rationale": "done", "page": 3}
        )
        p._checkpoint_progress = MagicMock()

        with patch("linkedin.orchestrator.human_delay_correlated", return_value=0):
            asyncio.run(p._process_string(search_string, progress))

        assert reviewed_pages == [3]
        assert p.browser.go_to_next_page.await_count == 2
        assert state.active_allocator_page_cursor() == 4


def test_process_string_continues_normally_after_bias_alert_fires():
    """Telemetry demotion (2026-07-04 SPL run): a fired count-based bias
    alert on the current string is an OBSERVATION — the page loop must still
    consult the normal continue/stop assessment and finish the string
    through the ordinary path, never break early or write a pause note.
    Replaces the P8.3 pause lock this behavior deliberately inverts."""
    from shared.bias_controls import BiasMonitor, DecisionRecord

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        search_string = SearchString(id=7, name="test", boolean="foo", status="queued")
        progress = Progress(brief_name="test", strings=[search_string], current_string_id=7, current_page=0)

        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results

        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p.browser.page.locator.return_value = locator
        p.browser.enter_search_string = AsyncMock()
        p.browser.get_results_count_text = AsyncMock(return_value="144")
        p.browser.get_results_count = AsyncMock(return_value=144)
        p.browser.go_to_next_page = AsyncMock(return_value=True)

        p._ensure_browser_healthy = AsyncMock()

        p._bias_monitor = BiasMonitor(max_consecutive_saves=1)

        async def _review_page_fires_bias_alert(**kwargs):
            # Simulate what the full-eval loop does inline: record a real
            # save streak and run the real check so the alert FIRES on the
            # current string before the page-level decision point.
            p._bias_monitor.record_decision(DecisionRecord(
                candidate_id="c1",
                string_id="7",
                stage="full",
                decision="SAVE",
                confidence=0.8,
                capability_area=None,
            ))
            p._bias_monitor.check_alerts("7")
            return None

        p._review_page_sequentially = AsyncMock(side_effect=_review_page_fires_bias_alert)
        p._assess_string_state = AsyncMock(
            side_effect=[{"decision": "stop", "rationale": "done", "page": 1}]
        )
        p._plan_variant_experiments = AsyncMock()

        with patch("linkedin.orchestrator.human_delay_correlated", return_value=0):
            asyncio.run(p._process_string(search_string, progress))

        # The alert genuinely fired on this string...
        assert any(
            key.startswith("consecutive_saves:")
            for key in p._bias_monitor.session_summary()["alerts_fired"]
        )
        # ...and the normal decision path still ran: no early break, no
        # pause note, the ordinary stop idiom closed the string.
        assert p._assess_string_state.await_count == 1
        assert "Bias pause" not in (search_string.notes or "")
        assert "Stopped after page 1." in (search_string.notes or "")


def test_bias_alert_never_gates_the_next_candidate_and_logs_flag_event():
    """Lock A for the telemetry demotion, driven through the REAL alert
    seam (_full_evaluate's inline handler), not a flag a refactor could
    rename around: with the tightest possible monitor, the alert firing on
    candidate 1 must not suppress candidate 2's evaluation, the alert must
    land in run_log.jsonl as severity=flag, and the deleted pause plumbing
    must stay deleted."""
    from shared.bias_controls import BiasMonitor

    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        p._bias_monitor = BiasMonitor(max_consecutive_saves=1)

        first = _make_snippet(name="A One", profile_url="/talent/profile/a1")
        second = _make_snippet(name="B Two", profile_url="/talent/profile/b2")

        with patch("linkedin.orchestrator.config") as mock_cfg, \
             patch(
                 "linkedin.orchestrator.full_judge",
                 side_effect=[
                     _baseline_save_decision(name="A One", url="/talent/profile/a1"),
                     _baseline_save_decision(name="B Two", url="/talent/profile/b2"),
                 ],
             ) as mock_full, \
             patch("linkedin.orchestrator.should_request_external_evidence") as mock_gate, \
             patch("linkedin.orchestrator.fetch_external_candidate_evidence") as mock_fetch, \
             patch("linkedin.orchestrator.full_judge_with_external_evidence") as mock_enrich, \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0.0):
            mock_cfg.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED = False
            mock_cfg.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MIN_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MAX_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_BASE_SECONDS = 0
            asyncio.run(p._full_evaluate(first))
            asyncio.run(p._full_evaluate(second))

        # The alert firing on candidate 1 did not gate candidate 2 — the
        # observable ANY reintroduced mid-string abort must trip, whatever
        # its flag is called.
        assert mock_full.call_count == 2
        assert mock_gate.call_count == 0
        assert mock_fetch.call_count == 0
        assert mock_enrich.call_count == 0

        events = read_jsonl(p.log_path)
        bias_events = [e for e in events if e.get("event") == "bias_alert"]
        assert any(
            e.get("severity") == "flag"
            and e.get("alert_type") == "consecutive_saves"
            for e in bias_events
        )
        assert not any(e.get("severity") == "pause" for e in bias_events)
        assert "bias_pauses" not in p.stats
        assert not hasattr(p, "_bias_pause_triggered")


def test_process_string_runs_variant_experiment_before_commit():
    """Large noisy pools can run a sibling experiment and then commit it to pagination."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        search_string = SearchString(id=9, name="scout", boolean="foo", status="queued")
        progress = Progress(brief_name="test", strings=[search_string], current_string_id=9, current_page=0)

        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results

        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p.browser.page.locator.return_value = locator
        p.browser.enter_search_string = AsyncMock()
        p.browser.get_results_count_text = AsyncMock(side_effect=["3.2K+", "120"])
        p.browser.get_results_count = AsyncMock(side_effect=[3200, 120])
        p.browser.go_to_next_page = AsyncMock(return_value=True)

        p._ensure_browser_healthy = AsyncMock()
        p._review_page_sequentially = AsyncMock(
            side_effect=[
                GlanceResult(action="reformulate", summary="all noise", confidence=0.91),
                None,
                None,
            ]
        )
        p._assess_string_state = AsyncMock(
            side_effect=[
                {"decision": "experiment", "rationale": "too broad", "page": 1},
                {"decision": "commit", "rationale": "variant is strong", "page": 1},
                {"decision": "stop", "rationale": "done", "page": 2},
            ]
        )
        # This test supplies its own assessment verdicts. Lifecycle scoring
        # against settled full outcomes is exercised in dedicated tests.
        p._evaluate_variant_lifecycle = MagicMock(return_value=None)
        p._plan_variant_experiments = AsyncMock(
            return_value=[
                LinkedInSearchVariant(
                    variant_id="precision-1",
                    parent_variant_id="root",
                    root_string_id=9,
                    boolean="bar",
                    variant_kind="precision",
                    hypothesis="tighter signal slice",
                    target_result_min=75,
                    target_result_max=400,
                )
            ]
        )
        async def _apply_variant(*, search_string, experiment_state, variant, **_kwargs):
            experiment_state.activate_variant(variant.variant_id)
            return SearchMutationResult(
                applied=True,
                result_count=120,
                result_count_text="120",
            )

        p._search_mutation_executor.apply_variant = AsyncMock(side_effect=_apply_variant)

        with patch("linkedin.orchestrator.human_delay_correlated", return_value=0):
            asyncio.run(p._process_string(search_string, progress))

        assert p._plan_variant_experiments.await_count == 1
        p._search_mutation_executor.apply_variant.assert_awaited_once()
        assert search_string.boolean == "bar"
        assert search_string.refinement_stack == ["foo"]
        p.browser.go_to_next_page.assert_awaited_once()
        assert "Committed precision variant on page 1." in (search_string.notes or "")
        assert "Stopped after page 2." in (search_string.notes or "")


def test_process_string_uses_real_scout_gate_for_large_noisy_pool():
    """A 6k noisy scout page should trigger a bounded sibling experiment before commit."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        search_string = SearchString(id=91, name="scout", boolean="foo", status="queued")
        progress = Progress(brief_name="test", strings=[search_string], current_string_id=91, current_page=0)

        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results

        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p.browser.page.locator.return_value = locator
        p.browser.enter_search_string = AsyncMock()
        p.browser.get_results_count_text = AsyncMock(side_effect=["6K+", "220"])
        p.browser.get_results_count = AsyncMock(side_effect=[6000, 220])
        p._ensure_browser_healthy = AsyncMock()
        p._record_runtime_event = MagicMock()

        async def _review_page(*, search_string, page_num, result_count, page_report, all_candidates, string_stats, progress):
            if search_string.boolean == "foo":
                string_stats["candidates"] += 11
                string_stats["facial_yes"] += 2
                string_stats["facial_no"] += 10
                all_candidates.extend(
                    [
                        {"title": "Applied Scientist", "company": "OpenAI", "facial": "yes", "save_reason": "strong"},
                        {"title": "Research Engineer", "company": "Anthropic", "facial": "yes", "save_reason": "strong"},
                        {"title": "Product Manager", "company": "BankCorp", "facial": "no"},
                        {"title": "Program Manager", "company": "BigCo", "facial": "no"},
                        {"title": "Engineering Manager", "company": "Enterprise Inc", "facial": "no"},
                    ]
                )
                p._latest_page_preview_snippets = [
                    _make_snippet(current_title="Applied Scientist", current_company="OpenAI"),
                    _make_snippet(current_title="Research Engineer", current_company="Anthropic"),
                    _make_snippet(current_title="Product Manager", current_company="BankCorp"),
                    _make_snippet(current_title="Program Manager", current_company="BigCo"),
                ]
                return GlanceResult(action="reformulate", summary="manager-heavy noise dominates", confidence=0.92)

            string_stats["candidates"] += 2
            string_stats["saves"] += 1
            string_stats["full_reviewed"] += 1
            string_stats["full_outreach"] += 1
            all_candidates.extend(
                [
                    {"title": "Staff Applied Scientist", "company": "Anthropic", "outcome": "save"},
                    {"title": "Research Engineer", "company": "OpenAI", "outcome": "facial_yes"},
                ]
            )
            p._latest_page_preview_snippets = [
                _make_snippet(current_title="Staff Applied Scientist", current_company="Anthropic"),
                _make_snippet(current_title="Research Engineer", current_company="OpenAI"),
            ]
            return None

        p._review_page_sequentially = AsyncMock(side_effect=_review_page)
        p._plan_variant_experiments = AsyncMock(
            return_value=[
                LinkedInSearchVariant(
                    variant_id="precision-1",
                    parent_variant_id="root",
                    root_string_id=91,
                    boolean="bar",
                    variant_kind="precision",
                    hypothesis="preserve frontier applied scientists and exclude manager-heavy noise",
                    target_result_min=75,
                    target_result_max=400,
                )
            ]
        )

        async def _apply_variant(*, search_string, experiment_state, variant, **_kwargs):
            experiment_state.activate_variant(variant.variant_id)
            return SearchMutationResult(applied=True, result_count=220, result_count_text="220")

        p._search_mutation_executor.apply_variant = AsyncMock(side_effect=_apply_variant)

        with (
            patch("linkedin.orchestrator.human_delay_correlated", return_value=0),
            patch("linkedin.orchestrator.config.MAX_PAGES_PER_STRING", 1),
        ):
            asyncio.run(p._process_string(search_string, progress))

        assert p._plan_variant_experiments.await_count == 1
        p._search_mutation_executor.apply_variant.assert_awaited_once()
        assert search_string.boolean == "bar"
        assert search_string.refinement_stack == ["foo"]
        assert "Variant precision applied on page 1." in (search_string.notes or "")
        assert "Committed precision variant on page 1." in (search_string.notes or "")
        assess_events = [
            call.kwargs["payload"]
            for call in p._record_runtime_event.call_args_list
            if call.kwargs.get("event_type") == "linkedin_search_assess"
        ]
        assert assess_events
        assert assess_events[0]["real_signal"] is False
        assert assess_events[0]["scout_gate_bucket"] == "precommit_dead_noisy_recovery"
        assert assess_events[0]["noise_dominant"] is True
        assert assess_events[1]["real_signal"] is True
        assert assess_events[1]["full_outreach"] == 1


def test_build_ordered_search_strings_uses_opening_micro_block():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._execution_plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {"boolean": f"foo {idx}", "rationale": f"string {idx}"}
                for idx in range(1, 9)
            ],
            coverage_gaps=[],
        )

        strings = p._build_ordered_search_strings()

        assert [s.block for s in strings[:3]] == ["Compound Batch 1"] * 3
        assert [s.block for s in strings[3:8]] == ["Compound Batch 2"] * 5


def test_run_block_adaptation_uses_opening_checkpoint_mode():
        with tempfile.TemporaryDirectory() as td:
            p = _make_pipeline(td)
            progress = Progress(
                brief_name="test",
                strings=[
                    SearchString(id=1, name="one", boolean="foo", status="done", block="Compound Batch 1"),
                    SearchString(id=2, name="two", boolean="bar", status="queued", block="Compound Batch 2"),
                ],
            )
        block_strings = [
            SearchString(
                id=1,
                name="one",
                boolean="foo",
                status="done",
                block="Compound Batch 1",
                facial_yes_count=1,
                facial_no_count=3,
                candidates_count=4,
                duplicates_count=0,
                saves=[],
                result_count=900,
                pages_reviewed=1,
            )
        ]
        observed = {}

        def fake_adapt(*args, **kwargs):
            observed["checkpoint_mode"] = kwargs["checkpoint_mode"]
            return AdaptationResponse()

        asyncio.run(p._run_block_adaptation("Compound Batch 1", block_strings, progress, fake_adapt))

        assert observed["checkpoint_mode"] == "opening_checkpoint"


def test_run_block_adaptation_treats_adaptive_followup_as_normal_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="adaptive followup", boolean="foo", status="done", block="Compound Batch 1", string_type="Adaptive"),
                SearchString(id=2, name="two", boolean="bar", status="queued", block="Compound Batch 2"),
            ],
        )
        block_strings = [
            SearchString(
                id=1,
                name="adaptive followup",
                boolean="foo",
                status="done",
                block="Compound Batch 1",
                string_type="Adaptive",
                facial_yes_count=1,
                facial_no_count=1,
                candidates_count=2,
                duplicates_count=0,
                saves=["Ada"],
                result_count=220,
                pages_reviewed=1,
                family_key="capital_markets_head_ai",
                novelty_bucket="edge_case",
                domain_lane="capital_markets",
            )
        ]
        observed = {}

        def fake_adapt(*args, **kwargs):
            observed["checkpoint_mode"] = kwargs["checkpoint_mode"]
            return AdaptationResponse()

        asyncio.run(p._run_block_adaptation("Compound Batch 1", block_strings, progress, fake_adapt))

        assert observed["checkpoint_mode"] == "normal_block_checkpoint"


def test_opening_checkpoint_review_signal_is_not_classified_as_all_dead():
    """A recruiter-review outcome is weak signal, not evidence of a dead block."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        block_strings = [
            SearchString(
                id=1,
                name="review signal",
                boolean="foo",
                full_review_count=1,
            ),
            SearchString(id=2, name="no outreach", boolean="bar"),
        ]
        for search_string in block_strings:
            state = p._experiment_state_for(search_string)
            state.last_page_insights = LinkedInPageInsights(
                page=1,
                result_count=200,
                result_window="150-800",
                dominant_non_fit_patterns=["same coherent noise pattern"],
            )

        assert p._opening_checkpoint_all_dead_and_coherent(block_strings) is False


def test_run_block_adaptation_exploitation_bias_promotes_live_lane_and_demotes_dead_family():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        winner = SearchString(
            id=1,
            name="winner",
            boolean="winner",
            status="done",
            block="Compound Batch 1",
            pages_reviewed=2,
            result_count=120,
            saves=["Ada"],
            facial_yes_count=2,
            facial_no_count=3,
            full_reviewed_count=1,
            full_outreach_count=1,
            candidates_count=5,
            family_key="market_infra_head_ai",
            novelty_bucket="edge_case",
            domain_lane="capital_markets",
        )
        loser = SearchString(
            id=2,
            name="loser",
            boolean="loser",
            status="done",
            block="Compound Batch 1",
            pages_reviewed=2,
            result_count=80,
            facial_yes_count=0,
            facial_no_count=8,
            candidates_count=8,
            family_key="dead_hidden_population",
            novelty_bucket="edge_case",
            domain_lane="insurance",
        )

        winner_state = p._experiment_state_for(winner)
        winner_state.commit_variant("root")
        winner_state.family_signal_total = 4
        winner_state.family_saves_total = 1
        winner_state.family_reviewed_total = 1
        winner_state.family_outreach_total = 1
        winner_state.precommit_recovery_attempts_used = 1
        winner_state.last_drift_refinement_summary = {"outcome": "rescued"}

        loser_state = p._experiment_state_for(loser)
        loser_state.commit_variant("root")
        loser_state.family_signal_total = 0
        loser_state.family_saves_total = 0

        progress = Progress(
            brief_name="test",
            strings=[
                winner,
                loser,
                SearchString(
                    id=3,
                    name="same family",
                    boolean="family",
                    block="Compound Batch 2",
                    family_key="market_infra_head_ai",
                    novelty_bucket="edge_case",
                    domain_lane="capital_markets",
                ),
                SearchString(
                    id=4,
                    name="same lane",
                    boolean="lane",
                    block="Compound Batch 2",
                    family_key="adjacent_market_infra",
                    novelty_bucket="edge_case",
                    domain_lane="capital_markets",
                ),
                SearchString(
                    id=5,
                    name="dead family",
                    boolean="dead",
                    block="Compound Batch 2",
                    family_key="dead_hidden_population",
                    novelty_bucket="edge_case",
                    domain_lane="insurance",
                ),
                SearchString(
                    id=6,
                    name="neutral edge",
                    boolean="neutral",
                    block="Compound Batch 2",
                    family_key="neutral_lane",
                    novelty_bucket="edge_case",
                    domain_lane="payments",
                ),
            ],
        )

        captured = {}

        def fake_adapt(*args, **kwargs):
            report = args[1]
            captured["summary"] = report.search_intelligence_summary
            captured["detail"] = report.string_details[0]["search_intelligence"]
            return AdaptationResponse()

        asyncio.run(p._run_block_adaptation("Compound Batch 1", [winner, loser], progress, fake_adapt))

        queued_ids = [s.id for s in progress.strings if s.status == "queued"]
        assert queued_ids == [3, 4, 6, 5]
        assert captured["summary"]["proven_family_keys"] == ["market_infra_head_ai"]
        assert captured["summary"]["dead_family_keys"] == ["dead_hidden_population"]
        assert captured["detail"]["drift_rescue_summary"]["outcome"] == "rescued"


def test_assess_string_state_stops_committed_variant_after_zero_signal_streak():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(id=21, name="builders", boolean="foo")
        state = p._experiment_state_for(search_string)
        state.commit_variant("root")
        state.committed_pages_reviewed = 2
        state.committed_zero_signal_streak = 2

        assessment = asyncio.run(
            p._assess_string_state(
                search_string=search_string,
                experiment_state=state,
                page_num=3,
                result_count=3200,
                string_stats={"facial_no": 4},
                page_stats={"facial_no": 4},
                page_insights=LinkedInPageInsights(
                    page=3,
                    result_count=3200,
                    result_window="150-800",
                    noise_anchors=["Product manager at BankCorp"],
                    dominant_non_fit_patterns=["product-heavy profiles dominate"],
                    glance_action="reformulate",
                ),
                remaining_queued_strings=4,
            )
        )

        assert assessment["decision"] == "stop"
        assert "zero-signal decay limit" in assessment["rationale"]


def test_search_intelligence_aggregate_does_not_label_productive_family_as_dead():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        winner = SearchString(
            id=1,
            name="Winner",
            boolean="winner",
            status="done",
            pages_reviewed=2,
            saves=["Ada Lovelace"],
            full_reviewed_count=1,
            full_outreach_count=1,
            family_key="shared_family",
            domain_lane="market_infra",
        )
        loser = SearchString(
            id=2,
            name="Loser",
            boolean="loser",
            status="done",
            pages_reviewed=1,
            family_key="shared_family",
            domain_lane="market_infra",
        )

        winner_state = p._experiment_state_for(winner)
        winner_state.family_signal_total = 3
        winner_state.family_reviewed_total = 1
        winner_state.family_outreach_total = 1
        loser_state = p._experiment_state_for(loser)
        loser_state.family_signal_total = 0

        summary = p._search_intelligence_aggregate([loser, winner])

        assert summary["proven_family_keys"] == ["shared_family"]
        assert summary["dead_family_keys"] == []


def test_search_intelligence_aggregate_blocks_family_promotion_when_saved_profiles_are_above_band():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        risky = SearchString(
            id=21,
            name="Buy-side broad lane",
            boolean='("BlackRock" OR "Two Sigma") AND ("GenAI" OR "LLM")',
            status="done",
            pages_reviewed=2,
            saves=["Ada Lovelace"],
            family_key="buy_side_generic",
            domain_lane="asset_management",
            seniority_risk="medium",
            title_bucket_risk="low",
            opening_eligible=False,
        )
        risky_state = p._experiment_state_for(risky)
        risky_state.family_signal_total = 3

        summary = p._search_intelligence_aggregate(
            [risky],
            profile_index={
                "ada lovelace": {
                    "name": "Ada Lovelace",
                    "headline": "Senior Managing Director, Global Head of AI",
                    "experiences": [{"title": "Senior Managing Director, Global Head of AI", "company": "BlackRock"}],
                }
            },
        )

        assert summary["proven_family_keys"] == []
        assert summary["contaminated_family_keys"] == ["buy_side_generic"]
        assert summary["contaminated_domain_lanes"] == ["asset_management"]


def test_exploitation_overlay_never_demotes_proven_families_even_if_summary_is_contaminated():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        adaptation = AdaptationResponse()
        remaining = [
            SearchString(
                id=10,
                name="Hot family A",
                boolean="foo",
                family_key="shared_family",
                domain_lane="market_infra",
                novelty_bucket="canonical",
            ),
            SearchString(
                id=11,
                name="Hot family B",
                boolean="bar",
                family_key="shared_family",
                domain_lane="market_infra",
                novelty_bucket="canonical",
            ),
            SearchString(
                id=12,
                name="Hot family C",
                boolean="baz",
                family_key="shared_family",
                domain_lane="market_infra",
                novelty_bucket="canonical",
            ),
            SearchString(
                id=13,
                name="Actually dead family",
                boolean="qux",
                family_key="dead_family",
                domain_lane="payments",
                novelty_bucket="canonical",
            ),
        ]

        overlay = p._apply_exploitation_bias_to_adaptation(
            adaptation=adaptation,
            remaining=remaining,
            block_summary={
                "proven_family_keys": ["shared_family"],
                "proven_domain_lanes": ["market_infra"],
                "dead_family_keys": ["shared_family", "dead_family"],
            },
            checkpoint_mode="normal_block_checkpoint",
        )

        assert overlay["promoted_string_ids"] == [10, 11, 12]
        assert overlay["demoted_string_ids"] == [13]
        assert all(
            action["string_id"] != 10 or action["move_to"] != "last"
            for action in adaptation.reorder
        )
        assert all(
            action["string_id"] != 11 or action["move_to"] != "last"
            for action in adaptation.reorder
        )
        assert all(
            action["string_id"] != 12 or action["move_to"] != "last"
            for action in adaptation.reorder
        )


def test_exploitation_overlay_demotes_contaminated_families_in_strict_seniority_runs():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        adaptation = AdaptationResponse()
        remaining = [
            SearchString(
                id=30,
                name="Risky buy-side lane",
                boolean="foo",
                family_key="buy_side_generic",
                domain_lane="asset_management",
                novelty_bucket="edge_case",
                seniority_risk="medium",
                title_bucket_risk="low",
                opening_eligible=False,
            ),
            SearchString(
                id=31,
                name="Safe capital-markets lane",
                boolean="bar",
                family_key="capital_markets_safe",
                domain_lane="capital_markets",
                novelty_bucket="edge_case",
                seniority_risk="low",
                title_bucket_risk="low",
                opening_eligible=True,
            ),
        ]

        overlay = p._apply_exploitation_bias_to_adaptation(
            adaptation=adaptation,
            remaining=remaining,
            block_summary={
                "proven_family_keys": [],
                "proven_domain_lanes": [],
                "dead_family_keys": [],
                "contaminated_family_keys": ["buy_side_generic"],
                "contaminated_domain_lanes": ["asset_management"],
            },
            checkpoint_mode="opening_checkpoint",
        )

        assert overlay["promoted_string_ids"] == []
        assert overlay["demoted_string_ids"] == [30]
        assert any(
            action["string_id"] == 30 and action["move_to"] == "last"
            for action in adaptation.reorder
        )


def test_assess_string_state_stops_after_single_zero_signal_page_post_failed_drift():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(id=22, name="builders", boolean="foo")
        state = p._experiment_state_for(search_string)
        state.commit_variant("root")
        state.committed_pages_reviewed = 1
        state.committed_zero_signal_streak = 1
        state.last_drift_refinement_summary = {"outcome": "not_rescued"}

        assessment = asyncio.run(
            p._assess_string_state(
                search_string=search_string,
                experiment_state=state,
                page_num=4,
                result_count=3200,
                string_stats={"facial_no": 3},
                page_stats={"facial_no": 3},
                page_insights=LinkedInPageInsights(
                    page=4,
                    result_count=3200,
                    result_window="150-800",
                    noise_anchors=["Program manager at BankCorp"],
                    dominant_non_fit_patterns=["manager-heavy profiles dominate"],
                    glance_action="reformulate",
                ),
                remaining_queued_strings=4,
            )
        )

        assert assessment["decision"] == "stop"
        assert "failed drift rescue" in assessment["rationale"]


def test_assess_pagination_drift_prefers_recall_when_overfit_risk_is_high():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(id=9, name="scout", boolean="foo", status="queued")
        state = p._experiment_state_for(search_string)
        state.commit_variant("root")
        # Snapshot uses one strong anchor, which should make rescue more recall-friendly.
        from linkedin.search_intelligence import LinkedInVariantSnapshot

        state.early_signal_snapshot = LinkedInVariantSnapshot.from_page(
            page_num=1,
            result_count=3200,
            page_insights=LinkedInPageInsights(
                page=1,
                result_count=3200,
                result_window="150-800",
                title_clusters=[{"label": "machine learning engineer", "count": 3}],
                signal_anchors=["ML engineer at OpenAI"],
            ),
            page_stats={"full_reviewed": 1, "full_outreach": 1},
        )
        state.recent_noise_snapshot = LinkedInVariantSnapshot.from_page(
            page_num=3,
            result_count=3200,
            page_insights=LinkedInPageInsights(
                page=3,
                result_count=3200,
                result_window="150-800",
                title_clusters=[{"label": "product manager", "count": 4}],
                noise_anchors=["Product manager at BankCorp"],
                dominant_non_fit_patterns=["product-heavy profiles dominate"],
                glance_action="reformulate",
            ),
            page_stats={"facial_no": 3},
        )
        state.family_outreach_total = 1
        state.active_variant.pages_reviewed = 3
        state.pages_since_last_mutation = 1

        assessment = p._assess_pagination_drift(
            experiment_state=state,
            page_num=3,
            result_count=3200,
            page_stats={"facial_no": 3},
            page_insights=LinkedInPageInsights(
                page=3,
                result_count=3200,
                result_window="150-800",
                noise_anchors=["Product manager at BankCorp"],
                dominant_non_fit_patterns=["product-heavy profiles dominate"],
                glance_action="reformulate",
            ),
            remaining_queued_strings=4,
        )

        assert assessment.decision == "spawn_recall_sibling"
        assert assessment.future_filter_hypothesis.startswith("title filter")


def test_process_string_runs_bounded_drift_rescue():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        search_string = SearchString(id=12, name="scout", boolean="foo", status="queued")
        progress = Progress(
            brief_name="test",
            strings=[search_string, SearchString(id=13, name="other", boolean="bar", status="queued")],
            current_string_id=12,
            current_page=0,
        )

        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results

        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p.browser.page.locator.return_value = locator
        p.browser.enter_search_string = AsyncMock()
        p.browser.get_results_count_text = AsyncMock(side_effect=["3.2K+", "600"])
        p.browser.get_results_count = AsyncMock(side_effect=[3200, 600])
        p.browser.go_to_next_page = AsyncMock(return_value=True)

        p._ensure_browser_healthy = AsyncMock()
        p._review_page_sequentially = AsyncMock(side_effect=[None, None, None, None])
        p._assess_string_state = AsyncMock(
            side_effect=[
                {"decision": "commit", "rationale": "page 1 strong", "page": 1},
                {"decision": "refine_committed", "rationale": "drifting", "page": 2},
                {"decision": "commit", "rationale": "rescued", "page": 1},
                {"decision": "stop", "rationale": "done", "page": 2},
            ]
        )
        p._plan_variant_experiments = AsyncMock()
        p._plan_drift_refinement = AsyncMock(
            return_value=(
                LinkedInSearchVariant(
                    variant_id="drift-12-1",
                    parent_variant_id="root",
                    root_string_id=12,
                    boolean="foo NOT product",
                    variant_kind="precision",
                ),
                {"decision": "refine_committed", "keyword_hypothesis": "exclude product-heavy leakage"},
            )
        )

        async def _apply_variant(
            *,
            search_string,
            experiment_state,
            variant,
            mutation_kind="experiment",
            mutation_summary=None,
            **_kwargs,
        ):
            experiment_state.mark_pending_drift(
                variant_id=variant.variant_id,
                parent_variant_id=experiment_state.committed_variant_id or experiment_state.active_variant_id,
                summary=mutation_summary,
            )
            experiment_state.activate_variant(variant.variant_id)
            return SearchMutationResult(
                applied=True,
                result_count=600,
                result_count_text="600",
            )

        p._search_mutation_executor.apply_variant = AsyncMock(side_effect=_apply_variant)

        with patch("linkedin.orchestrator.human_delay_correlated", return_value=0):
            asyncio.run(p._process_string(search_string, progress))

        assert p._plan_drift_refinement.await_count == 1
        assert p._search_mutation_executor.apply_variant.await_args.kwargs["mutation_kind"] == "drift"
        assert search_string.boolean == "foo NOT product"
        assert search_string.pages_reviewed == 2
        assert "Drift rescue precision applied on page 2." in (search_string.notes or "")


def test_failed_drift_reenters_committed_query_at_its_own_next_page():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        search_string = SearchString(id=14, name="scout", boolean="foo", status="queued")
        progress = Progress(
            brief_name="test",
            strings=[search_string, SearchString(id=15, name="other", boolean="bar")],
            current_string_id=14,
            current_page=0,
        )

        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results

        live_query = {"boolean": ""}

        async def enter_search_string(boolean):
            live_query["boolean"] = boolean

        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p.browser.page.locator.return_value = locator
        p.browser.enter_search_string = AsyncMock(side_effect=enter_search_string)
        p.browser.get_results_count_text = AsyncMock(
            side_effect=["3.2K+", "3.2K+"]
        )
        p.browser.get_results_count = AsyncMock(side_effect=[3200, 3200])
        p.browser.go_to_next_page = AsyncMock(return_value=True)

        p._ensure_browser_healthy = AsyncMock()
        reviewed = []

        async def review_page(**kwargs):
            state = p._experiment_states[search_string.id]
            reviewed.append(
                (
                    state.active_variant_id,
                    kwargs["page_num"],
                    live_query["boolean"],
                )
            )
            return None

        p._review_page_sequentially = AsyncMock(side_effect=review_page)
        p._evaluate_variant_lifecycle = MagicMock(return_value=None)
        p._assess_string_state = AsyncMock(
            side_effect=[
                {"decision": "commit", "rationale": "root is viable", "page": 1},
                {"decision": "refine_committed", "rationale": "root drifted", "page": 2},
                {"decision": "resume_committed", "rationale": "drift failed", "page": 1},
                {"decision": "stop", "rationale": "done", "page": 3},
            ]
        )
        p._plan_drift_refinement = AsyncMock(
            return_value=(
                LinkedInSearchVariant(
                    variant_id="drift-14-1",
                    parent_variant_id="root",
                    root_string_id=14,
                    boolean="foo NOT product",
                    variant_kind="precision",
                ),
                {"decision": "refine_committed"},
            )
        )

        async def apply_drift(*, experiment_state, variant, mutation_summary=None, **_kwargs):
            experiment_state.mark_pending_drift(
                variant_id=variant.variant_id,
                parent_variant_id=experiment_state.committed_variant_id,
                summary=mutation_summary,
            )
            experiment_state.activate_variant(variant.variant_id)
            live_query["boolean"] = variant.boolean
            return SearchMutationResult(
                applied=True,
                result_count=600,
                result_count_text="600",
            )

        p._search_mutation_executor.apply_variant = AsyncMock(side_effect=apply_drift)

        durable_checkpoint = p._checkpoint_progress

        def crash_after_committed_restore(*args, **kwargs):
            durable_checkpoint(*args, **kwargs)
            state = p._experiment_states[search_string.id]
            if (
                "drift-14-1" in state.variants
                and state.active_variant_id == "root"
                and state.active_allocator_page_cursor() == 3
                and live_query["boolean"] == "foo"
                and kwargs.get("page_num") is None
            ):
                raise RuntimeError("crash after committed restore checkpoint")

        p._checkpoint_progress = MagicMock(side_effect=crash_after_committed_restore)

        with (
            patch("linkedin.orchestrator.human_delay_correlated", return_value=0),
            pytest.raises(RuntimeError, match="committed restore checkpoint"),
        ):
            asyncio.run(p._process_string(search_string, progress))

        assert p.browser.go_to_next_page.await_count == 3
        assert p.browser.enter_search_string.await_args_list[-1].args == ("foo",)
        assert p.browser.get_results_count.await_count == 2

        resumed = _make_pipeline(td)
        resumed_progress = resumed._load_or_create_progress()
        resumed_string = next(
            item for item in resumed_progress.strings if item.id == search_string.id
        )
        resumed_state = resumed._experiment_states[search_string.id]

        assert resumed_state.active_variant_id == "root"
        assert resumed_state.current_boolean() == "foo"
        assert resumed_state.active_allocator_page_cursor() == 3

        no_results_b = MagicMock()
        no_results_b.is_visible = AsyncMock(return_value=False)
        locator_b = MagicMock()
        locator_b.first = no_results_b
        resumed.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/123/search"
        )
        resumed.browser.page.locator.return_value = locator_b
        resumed.browser.navigate_to_search = AsyncMock()
        resumed.browser.enter_search_string = AsyncMock(side_effect=enter_search_string)
        resumed.browser.get_results_count_text = AsyncMock(return_value="3.2K+")
        resumed.browser.get_results_count = AsyncMock(return_value=3200)
        resumed.browser.go_to_next_page = AsyncMock(return_value=True)
        resumed._ensure_browser_healthy = AsyncMock()

        async def review_resumed_page(**kwargs):
            state = resumed._experiment_states[search_string.id]
            reviewed.append(
                (
                    state.active_variant_id,
                    kwargs["page_num"],
                    live_query["boolean"],
                )
            )
            return None

        resumed._review_page_sequentially = AsyncMock(
            side_effect=review_resumed_page
        )
        resumed._evaluate_variant_lifecycle = MagicMock(return_value=None)
        resumed._assess_string_state = AsyncMock(
            return_value={"decision": "stop", "rationale": "done", "page": 3}
        )
        resumed._checkpoint_progress = MagicMock()

        with patch("linkedin.orchestrator.human_delay_correlated", return_value=0):
            asyncio.run(resumed._process_string(resumed_string, resumed_progress))

        assert reviewed == [
            ("root", 1, "foo"),
            ("root", 2, "foo"),
            ("drift-14-1", 1, "foo NOT product"),
            ("root", 3, "foo"),
        ]
        state = resumed._experiment_states[search_string.id]
        assert state.variants["root"].allocator_page_cursor == 4
        assert state.variants["drift-14-1"].allocator_page_cursor == 2
        assert state.active_allocator_page_cursor() == 4
        assert resumed.browser.enter_search_string.await_args_list[-1].args == ("foo",)
        assert resumed.browser.go_to_next_page.await_count == 2


def _sample_report_analysis() -> dict:
    return {
        "winning_lanes": [
            {
                "lane": "Research Copilot",
                "string_ids": [2],
                "candidate_examples": ["Mithun Azhagappan"],
                "evidence": "Highest save count from workflow-specific BFSI product language.",
                "why_it_worked": "Product-output vocabulary gated for real builders.",
                "recommended_action": "UNIQUE_REPORT_SENTINEL promote this lane earlier.",
            }
        ],
        "underperforming_lanes": [
            {
                "lane": "Surveillance",
                "string_ids": [8],
                "issue": "Traditional compliance-tech noise.",
                "evidence": "Zero saves across two pages.",
                "recommended_action": "Only run with an explicit GenAI AND-gate.",
            }
        ],
        "coverage_gaps": [
            {
                "gap": "Payments",
                "why_it_matters": "The run never explicitly covered transaction banking.",
                "suggested_search_strategy": "Add payment-orchestration and RTP strings.",
            }
        ],
        "noise_patterns": [
            {
                "pattern": "Product-heavy AI leadership",
                "evidence": "Several product/strategy AI officers were rejected post deep-dive.",
                "mitigation": "Strengthen builder-authoring verbs and architecture language.",
            }
        ],
        "saved_candidate_patterns": {
            "standout_candidates": [{"name": "Mithun Azhagappan", "why": "Goldman AI platform architect."}],
            "common_employers": [{"employer": "JPMorgan", "count": 3, "note": "Strong bank GenAI-convert segment."}],
            "common_titles": [{"title_family": "Executive Director", "count": 1, "note": "Right seniority band."}],
            "archetype_distribution": [{"archetype": "BFSI-native GenAI converts", "count": 4, "note": "Dominant save pattern."}],
            "seniority_notes": ["Many VP-level bank builders were interesting but below the full lab-leadership bar."],
        },
        "adaptation_assessment": {
            "summary": "Adaptation stayed focused on workflow-specific strings.",
            "effective_refinements": ["Narrowing research-copilot language improved precision."],
            "questionable_or_skipped": ["Payments stayed under-covered."],
            "operational_notes": ["Prefer tight workflow language over broad archetype-first queries."],
        },
        "recommendations": {
            "try_next": ["Payments and transaction-banking builders"],
            "avoid_next": ["Ungated surveillance strings"],
            "prioritize_pipeline": ["Mithun Azhagappan"],
        },
        "brief_iteration_hints": {
            "instructions": ["Cover payments in the first search block."],
            "search_priorities": ["Payments / transaction-banking builders"],
            "additional_search_terms": ["payment orchestration", "transaction banking"],
            "intake_notes": "The latest run validated research-copilot lanes and exposed a payments gap.",
            "depth_distinction": {
                "builder_definition": "Still an executive-builder search.",
                "user_definition": "Product and strategy leaders remain out of scope.",
                "edge_case_guidance": "VP bank builders need extra scope scrutiny.",
            },
            "non_fit_patterns": [
                {
                    "label": "Product-heavy AI officer",
                    "description": "Executive product leadership without system-builder authorship.",
                    "why_not": "Wrong depth for this role.",
                    "examples": ["Chief Product & AI Officer"],
                }
            ],
            "minimum_bar_description": "NYC, 15+ years, BFSI depth, and post-2022 GenAI remain hard requirements.",
            "facial_calibration": {
                "expected_yes_rate_low": 0.1,
                "expected_yes_rate_high": 0.22,
                "fast_exit_patterns": ["Pure product history"],
                "trajectory_yes_patterns": ["Big-bank GenAI convert"],
                "trajectory_ambiguous_patterns": ["VP at smaller firm"],
                "trajectory_no_patterns": ["Vendor field CTO without build ownership"],
            },
            "employer_signal_rules": [
                {
                    "tier": "payments_builder",
                    "employer_patterns": ["Visa", "Mastercard"],
                    "evidence_required": "Still requires production builder evidence.",
                    "save_on_employer_alone": False,
                }
            ],
            "calibration_examples": {
                "strong_saves": [{"name": "Mithun Azhagappan", "why": "Strong fit."}],
                "incorrect_saves": [{"name": "Deepinder Gulati", "why": "Product-heavy."}],
                "borderline_verify": [{"name": "Peter Chung", "why": "Check scope carefully."}],
            },
            "notes": "Promote payments in the next draft.",
            "locked_field_cautions": ["Do not relax geography or years-of-experience gates."],
        },
    }


def test_generate_run_report_writes_json_markdown_and_input_artifacts():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.role_title = "Head of Applied AI Lab"
        p.brief_obj.linkedin_project = "Head of Applied AI Lab"
        p.brief_obj.linkedin_project_id = "3000000006"
        p.brief_obj.raw = {"version": "2.1"}
        p.stats.update(
            {
                "snippets_extracted": 12,
                "facial_yes": 4,
                "facial_no": 8,
                "saved": 1,
                "rejected": 2,
            }
        )
        p._search_memory = {
            "project_id": "3000000006",
            "overall": {
                "strings_seen": 1,
                "candidates_seen": 12,
                "duplicates": 2,
                "saves": 1,
                "edge_case_saves": 1,
                "canonical_saves": 0,
            },
            "families": {
                "research_copilot_asset_mgmt": {
                    "family_key": "research_copilot_asset_mgmt",
                    "novelty_bucket": "edge_case",
                    "domain_lane": "asset_management",
                    "status": "active",
                    "status_reason": "",
                    "strings_seen": 1,
                    "candidates_seen": 12,
                    "duplicates": 2,
                    "saves": 1,
                    "dominant_anchors": ["research copilot"],
                }
            },
        }
        p._bias_summary_for_report = MagicMock(return_value="Bias summary sentinel")

        append_jsonl(
            p.final_path,
            {
                "candidate_name": "Mithun Azhagappan",
                "decision": "SAVE",
                "path": "DIRECT",
                "confidence": 0.92,
                "rationale": "Goldman AI platform architect.",
            },
        )
        append_jsonl(
            p.final_path,
            {
                "candidate_name": "Deepinder Gulati",
                "decision": "REJECT",
                "path": "REJECT",
                "confidence": 0.12,
                "rationale": "Product-heavy AI leadership without builder evidence.",
            },
        )

        progress = Progress(
            brief_name="head-ai-lab",
            strings=[
                SearchString(
                    id=2,
                    name="Research copilot lane",
                    boolean="research",
                    status="done",
                    result_count=526,
                    pages_reviewed=4,
                    saves=["Mithun Azhagappan"],
                    notes="Strong lane",
                    facial_yes_count=3,
                    facial_no_count=7,
                    candidates_count=10,
                    family_key="research_copilot_asset_mgmt",
                    novelty_bucket="edge_case",
                    domain_lane="asset_management",
                ),
                SearchString(
                    id=8,
                    name="Surveillance lane",
                    boolean="surveillance",
                    status="skipped",
                    result_count=1100,
                    pages_reviewed=2,
                    notes="Stopped early after noise.",
                    facial_yes_count=0,
                    facial_no_count=6,
                    candidates_count=6,
                    family_key="surveillance_builder",
                    novelty_bucket="edge_case",
                    domain_lane="risk_compliance",
                ),
            ],
        )

        with patch("shared.llm_clients.opus_llm", return_value=_sample_report_analysis()) as opus_spy:
            p._generate_run_report(progress)

        report_input = json.loads(Path(td, "run-report-input.json").read_text())
        report_json = json.loads(Path(td, "run-report.json").read_text())
        report_md = Path(td, "run-report.md").read_text()

        assert report_input["saved_candidate_summaries"][0]["candidate_name"] == "Mithun Azhagappan"
        assert report_input["rejected_candidate_summaries"][0]["candidate_name"] == "Deepinder Gulati"
        assert report_input["search_memory_summary"]["overall"]["families_tracked"] == 1
        assert report_json["winning_lanes"][0]["recommended_action"].startswith("UNIQUE_REPORT_SENTINEL")
        assert report_json["metrics_summary"]["saved"] == 1
        assert "UNIQUE_REPORT_SENTINEL" in report_md
        assert "Payments" in report_md
        opus_spy.assert_called_once()
        assert opus_spy.call_args.kwargs["max_tokens"] == 24576
        assert opus_spy.call_args.kwargs["timeout_seconds"] == 420
        assert (
            opus_spy.call_args.kwargs["model_name"]
            == config.FULL_EVAL_MODEL_NAME
        )


def test_generate_run_report_failure_is_warning_only(capsys):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.role_title = "Head of Applied AI Lab"
        p.brief_obj.linkedin_project = "Head of Applied AI Lab"
        p.brief_obj.linkedin_project_id = "3000000006"
        p.brief_obj.raw = {"version": "2.1"}

        progress = Progress(
            brief_name="head-ai-lab",
            strings=[SearchString(id=2, name="Research lane", boolean="research", status="done", result_count=10)],
        )

        with patch("shared.llm_clients.opus_llm", side_effect=RuntimeError("boom")):
            p._generate_run_report(progress)

        captured = capsys.readouterr()
        assert "Report generation failed: boom" in captured.out
        assert not Path(td, "run-report.json").exists()
        assert not Path(td, "run-report.md").exists()


def test_build_run_report_snapshot_includes_search_intelligence_summary():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.role_title = "Head of Applied AI Lab"
        p.brief_obj.linkedin_project = "Head of Applied AI Lab"
        p.brief_obj.linkedin_project_id = "3000000006"
        p.brief_obj.raw = {"version": "2.1"}
        p.stats["snippets_extracted"] = 12
        p.stats["facial_yes"] = 3
        p.stats["facial_no"] = 9
        p.stats["saved"] = 1
        p.stats["rejected"] = 2

        winner = SearchString(
            id=2,
            name="Market infra lane",
            boolean="market infra",
            status="done",
            result_count=214,
            pages_reviewed=3,
            saves=["Ada"],
            facial_yes_count=3,
            facial_no_count=4,
            full_reviewed_count=1,
            full_outreach_count=1,
            candidates_count=7,
            family_key="market_infra_head_ai",
            novelty_bucket="edge_case",
            domain_lane="capital_markets",
        )
        state = p._experiment_state_for(winner)
        state.commit_variant("root")
        state.family_signal_total = 5
        state.family_saves_total = 1
        state.family_reviewed_total = 1
        state.family_outreach_total = 1
        state.precommit_recovery_attempts_used = 1
        state.drift_attempt_count = 1
        state.last_drift_refinement_summary = {"outcome": "rescued", "decision": "refine_committed"}

        progress = Progress(brief_name="head-ai-lab", strings=[winner])

        snapshot = p._build_run_report_snapshot(progress)

        assert snapshot["metrics_summary"]["strings_with_precommit_experiments"] == 1
        assert snapshot["metrics_summary"]["strings_with_drift_rescue_attempts"] == 1
        assert snapshot["metrics_summary"]["strings_rescued_by_drift"] == 1
        assert snapshot["metrics_summary"]["proven_family_keys"] == ["market_infra_head_ai"]
        assert snapshot["string_performance"][0]["search_intelligence"]["family_signal_total"] == 5
        assert (
            snapshot["string_performance"][0]["search_intelligence"]["drift_rescue_summary"]["outcome"]
            == "rescued"
        )


# ---------------------------------------------------------------------------
# Slice 2: shadow external-evidence augmentation in _full_evaluate
# ---------------------------------------------------------------------------
# These tests pin the analytical-debug shadow block. Baseline canonical state
# must not be affected by anything in the shadow path; the artifact written
# is shadow_final_judgments.jsonl (declared as ANALYTICAL_DEBUG in
# shared/runtime_state/artifacts.py).

from shared.schemas import (
    CandidateProfileSummary,
    Education,
    Experience,
    EvidenceRef,
    ExternalCandidateEvidence,
    ExternalEvidenceFailure,
    ExternalFactBlock,
    ExternalInference,
    TriggerDecision,
)


def _make_full_evaluate_pipeline(td: str):
    """Build a pipeline + the mocks needed to exercise just the shadow block.

    Mocks the browser, profile extraction, baseline full_judge, novelty
    derivation, and side-effects so that running _full_evaluate exercises
    only the canonical lifecycle prelude + the shadow block + the
    save/reject branch.
    """

    p = _make_pipeline(td)
    # has_v2_schema=False → _bias_monitor stays None → no bias loop runs.
    p._bias_monitor = None
    p._triage_tightened = False
    p._tightening_prefix = ""

    # Browser mocks — _full_evaluate calls these.
    p.browser.get_profile_status_summary = AsyncMock(return_value={})
    p.browser.go_back_to_results = AsyncMock()

    # Acquisition service — return a stand-in AcquisitionResult with .profile_summary
    summary = CandidateProfileSummary(
        name="Test Person",
        profile_url="/talent/profile/test123",
        headline="ML Engineer",
        experiences=[
            Experience(
                title="ML Engineer",
                company="Acme",
                start="2020",
                end="present",
                summary_bullets=["X", "Y"],
            )
        ],
        education=[Education(degree="PhD", school="MIT", field="ML")],
        skills_snippet=["python"],
    )
    acquisition = MagicMock()
    acquisition.profile_summary = summary
    p._acquisition_service = MagicMock()
    p._acquisition_service.extract_profile_summary = AsyncMock(return_value=acquisition)

    # Side-effects (save action) — async stub returning a real outcome
    # (P1.2: _full_evaluate consumes the SideEffectOutcome instead of
    # discarding it, so the stub must return the real shape).
    from shared.execution import SideEffectOutcome

    p._side_effects_service = MagicMock()
    p._side_effects_service.handle_save_decision = AsyncMock(
        return_value=SideEffectOutcome(
            effect_type="linkedin_save",
            status="succeeded",
            payload={"test_mode": True},
        )
    )
    p._reopen_profile_for_full_eval_save = AsyncMock()

    # _ensure_services is called inside _full_evaluate; make it a no-op so
    # it doesn't reset the mocks above.
    p._ensure_services = MagicMock()

    # Novelty derivation is orthogonal; return constant pair.
    p._derive_novelty_value = MagicMock(return_value=("medium", "rationale"))

    # Runtime state bridge / run id are not initialized in the fake pipeline,
    # so _start_runtime_stage_attempt / _finish_runtime_stage_success /
    # _record_runtime_event short-circuit on their None checks. Nothing to
    # mock further.
    return p, summary


@pytest.mark.parametrize("method_name", ["_full_evaluate", "_open_and_extract"])
def test_profile_extraction_propagates_browser_fatal_cleanup_failure(
    method_name,
):
    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        p._acquisition_service.extract_profile_summary.side_effect = ValueError(
            "candidate-local profile defect"
        )
        p.browser.go_back_to_results.side_effect = RuntimeError(
            "browser context closed"
        )

        with pytest.raises(RuntimeError, match="browser context closed"):
            asyncio.run(getattr(p, method_name)(_make_snippet()))

        p.browser.go_back_to_results.assert_awaited_once()


@pytest.mark.parametrize("evaluation_path", ["serial", "pipelined"])
@pytest.mark.parametrize(
    ("failure_factory", "expected_type", "abort_reason"),
    [
        (
            lambda: OperatorStopRequested("operator stop"),
            OperatorStopRequested,
            "operator_stop",
        ),
        (
            lambda: SessionExpired("session cap"),
            SessionExpired,
            "session_expired",
        ),
        (
            lambda: GovernorLimitReached("profile cap"),
            GovernorLimitReached,
            "governor_limit",
        ),
        (
            lambda: RuntimeError(
                "Page.evaluate: Target page, context or browser has been closed"
            ),
            RuntimeError,
            "browser_disconnect",
        ),
        (
            lambda: RuntimeError(
                "Your credit balance is too low; purchase credits"
            ),
            ApiBudgetExhaustedError,
            "api_budget_exhausted",
        ),
    ],
    ids=["operator-stop", "session-expired", "governor", "browser", "budget"],
)
def test_profile_activity_enrichment_propagates_run_level_abort(
    evaluation_path,
    failure_factory,
    expected_type,
    abort_reason,
):
    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        snippet = _make_snippet()
        failure = failure_factory()
        p.browser.get_profile_status_summary.side_effect = failure
        p._abort_runtime_stage_attempt = MagicMock()

        with pytest.raises(expected_type):
            if evaluation_path == "serial":
                asyncio.run(p._full_evaluate(snippet))
            else:
                asyncio.run(p._open_and_extract(snippet))

        p._abort_runtime_stage_attempt.assert_called_once()
        payload = p._abort_runtime_stage_attempt.call_args.kwargs["payload"]
        assert payload["run_abort"] == abort_reason
        assert payload["stage"] == "full"


def _baseline_save_decision(*, name: str = "Test Person", url: str = "/talent/profile/test123") -> OpusDecision:
    return OpusDecision(
        stage="full",
        decision="SAVE",
        path="DIRECT:Data Curation",
        confidence=0.62,
        rationale="Strong builder.",
        candidate_name=name,
        profile_url=url,
        outreach_tier="PRIORITY",
    )


def _baseline_reject_decision(**kwargs) -> OpusDecision:
    base = _baseline_save_decision(**kwargs)
    base.decision = "REJECT"
    base.path = "none"
    base.rationale = "Not a fit."
    base.confidence = 0.7
    base.outreach_tier = ""
    base.reject_reason = "CAPABILITY_INSUFFICIENT"
    return base


def _baseline_parse_failure(**kwargs) -> OpusDecision:
    base = _baseline_save_decision(**kwargs)
    base.decision = "PARSE_FAILURE"
    base.path = "none"
    base.rationale = "[PARSE_FAILURE: bad output]"
    base.confidence = 0.0
    return base


def _evidence_with_two_refs() -> ExternalCandidateEvidence:
    return ExternalCandidateEvidence(
        trigger_reason="academic_context",
        identity_confidence=0.7,
        external_fact_blocks=[
            ExternalFactBlock(
                topic="thesis",
                facts=["Thesis on RLHF."],
                evidence_refs=[
                    EvidenceRef(url="https://x.example/a", source_quality="high"),
                    EvidenceRef(url="https://x.example/b", source_quality="medium"),
                ],
                source_quality="high",
            )
        ],
        external_inferences=[],
        unresolved_ambiguities=[],
        do_not_use_for_judgment=[],
        raw_provider_model="sonar-deep-research",
        normalizer_model="",
    )


def _read_shadow(td: str) -> list[dict]:
    return read_jsonl(Path(td, "shadow_final_judgments.jsonl"))


def _assert_baseline_canonical_preserved(
    pipeline,
    *,
    expected_decision: str,
    expected_save_called: bool,
):
    """Common canonical-preservation invariants for shadow-block tests.

    Pins that the save side-effect is governed by the BASELINE decision alone:
    when the baseline is a SAVE-flavored decision, ``handle_save_decision`` is
    awaited exactly once; for non-save baselines, it is never awaited. The
    side-effect call site (orchestrator.py:4162) does not accept a ``decision``
    kwarg, so the structural pin "no enriched OpusDecision can leak into the
    save click" is verified by the save-call signature itself.
    """
    save_mock = pipeline._side_effects_service.handle_save_decision
    if expected_save_called:
        save_mock.assert_awaited_once()
        # Structural pin: the call site only forwards (snippet, runtime_search_string,
        # attempt_id). No decision kwarg is accepted, so an enriched OpusDecision
        # cannot reach this path even if the shadow block produced one.
        assert "decision" not in save_mock.await_args.kwargs
    else:
        assert save_mock.await_count == 0


def test_full_evaluate_save_failure_keeps_pre_actuation_tier_count():
    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        snippet = _make_snippet()
        baseline = _baseline_save_decision()
        p._side_effects_service.handle_save_decision.side_effect = RuntimeError(
            "post-judgment save failure"
        )

        with patch("linkedin.orchestrator.config") as mock_cfg, \
             patch("linkedin.orchestrator.full_judge", return_value=baseline), \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0.0):
            mock_cfg.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED = False
            mock_cfg.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MIN_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MAX_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_BASE_SECONDS = 0
            with pytest.raises(RuntimeError, match="post-judgment save failure"):
                asyncio.run(p._full_evaluate(snippet))

        assert p.stats["outreach_tier_counts"] == {"PRIORITY": 1}
        p._reopen_profile_for_full_eval_save.assert_awaited_once_with(snippet)
        p._side_effects_service.handle_save_decision.assert_awaited_once()


def test_full_evaluate_failed_save_stays_retryable_and_nonterminal():
    from shared.execution import SideEffectOutcome

    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        snippet = _make_snippet()
        p._in_flight_urls.add(snippet.profile_url)
        p._side_effects_service.handle_save_decision.return_value = (
            SideEffectOutcome(
                effect_type="linkedin_save",
                status="failed",
                payload={"failure_reason": "save_not_persisted"},
            )
        )

        with patch("linkedin.orchestrator.config") as mock_cfg, \
             patch(
                 "linkedin.orchestrator.full_judge",
                 return_value=_baseline_save_decision(),
             ), \
             patch(
                 "linkedin.orchestrator.human_delay_correlated",
                 return_value=0.0,
             ):
            mock_cfg.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED = False
            mock_cfg.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS = 0
            with pytest.raises(
                RuntimeError,
                match="save was not durably confirmed",
            ):
                asyncio.run(p._full_evaluate(snippet))

        assert snippet.profile_url not in p._seen_urls
        assert snippet.profile_url not in p._prior_outcomes
        assert snippet.profile_url not in p._in_flight_urls


@pytest.mark.parametrize("save_result", [False, True], ids=["failed", "succeeded"])
def test_resumed_historical_save_with_failed_receipt_retries_real_full_path(
    save_result,
):
    with tempfile.TemporaryDirectory() as td:
        seed = _make_pipeline(td)
        owner = SearchString(
            id=1,
            name="owner",
            boolean="one",
            status="in_progress",
        )
        progress = Progress(brief_name="test", strings=[owner])
        seed._runtime_run_id, progress = (
            seed._runtime_bridge.start_or_resume_run(
                resume=False,
                initial_progress=progress,
            )
        )
        seed._progress = progress
        snippet = _make_snippet(
            name="Ada",
            profile_url="/talent/profile/ada",
            source_string_id=owner.id,
            source_string_name=owner.name,
        )
        facial = OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="none",
            confidence=1.0,
            rationale="strong",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        historical_save = _baseline_save_decision(
            name=snippet.name,
            url=snippet.profile_url,
        )
        seed._record_runtime_snippet(owner, snippet)
        facial_attempt = seed._start_runtime_stage_attempt(
            search_string=owner,
            snippet=snippet,
            stage="facial",
        )
        seed._finish_runtime_stage_success(
            attempt_id=facial_attempt,
            stage="facial",
            snippet=snippet,
            decision=facial,
        )
        full_attempt = seed._start_runtime_stage_attempt(
            search_string=owner,
            snippet=snippet,
            stage="full",
        )
        seed._finish_runtime_stage_success(
            attempt_id=full_attempt,
            stage="full",
            snippet=snippet,
            decision=historical_save,
        )
        receipt = seed._runtime_bridge.begin_candidate_side_effect(
            run_id=seed._runtime_run_id,
            search_string=owner,
            snippet=snippet,
            attempt_id=full_attempt,
            effect_type="linkedin_save",
            idempotency_key="save",
        )
        seed._runtime_bridge.complete_candidate_side_effect(
            side_effect_id=int(receipt["side_effect"]["id"]),
            status="failed",
            payload={"failure_reason": "save_not_persisted"},
        )

        resumed = _make_pipeline(td)
        resumed_progress = resumed._load_or_create_progress()
        resumed._progress = resumed_progress
        resumed._seen_urls = set()
        resumed._prior_outcomes = {}
        resumed._load_candidate_history()
        resumed._hydrate_resume_funnel_from_runtime(resumed_progress)
        assert snippet.profile_url in resumed._saved_urls
        assert (
            resumed._resume_pending_full_decisions[snippet.profile_url]
            == "FACIAL_YES"
        )
        resumed._ensure_services()
        acquisition = MagicMock()
        acquisition.profile_summary = CandidateProfileSummary(
            name=snippet.name,
            profile_url=snippet.profile_url,
            headline="Engineer",
            experiences=[],
            education=[],
            skills_snippet=[],
        )
        resumed._acquisition_service.extract_profile_summary = AsyncMock(
            return_value=acquisition
        )
        resumed.browser.get_profile_status_summary = AsyncMock(return_value={})
        resumed.browser.go_back_to_results = AsyncMock()
        resumed.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        resumed.browser.current_profile_identity_fragment.return_value = "ada"
        resumed.browser.is_already_saved_on_card = AsyncMock(return_value=False)
        resumed.browser.save_candidate = AsyncMock(return_value=save_result)
        resumed.browser.scroll_for_linger = AsyncMock(return_value=0)
        resumed.browser.scroll_restore = AsyncMock()
        resumed._reopen_profile_for_full_eval_save = AsyncMock()
        resumed._derive_novelty_value = MagicMock(
            return_value=("medium", "rationale")
        )

        context = (
            pytest.raises(
                RuntimeError,
                match="save was not durably confirmed",
            )
            if not save_result
            else nullcontext()
        )
        with patch(
            "linkedin.orchestrator.full_judge",
            return_value=_baseline_save_decision(
                name=snippet.name,
                url=snippet.profile_url,
            ),
        ), patch(
            "linkedin.orchestrator.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ), patch(
            "linkedin.orchestrator.config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS",
            0,
        ), patch(
            "linkedin.side_effects.asyncio.sleep",
            new=AsyncMock(),
        ), context:
            asyncio.run(
                resumed._full_evaluate(
                    snippet,
                    search_string=resumed_progress.strings[0],
                )
            )

        resumed.browser.is_already_saved_on_card.assert_awaited()
        resumed.browser.save_candidate.assert_awaited_once()
        receipt_row = resumed._runtime_state.list_candidate_side_effects(
            source="linkedin",
            brief_id="test-project",
        )[0]
        assert receipt_row["status"] == (
            "succeeded" if save_result else "failed"
        )
        candidate = resumed._runtime_state.get_candidate(
            source="linkedin",
            brief_id="test-project",
            identity_key=snippet.profile_url,
        )
        assert (candidate["current_lifecycle_state"] == "full_terminal") is (
            save_result
        )


def test_full_stage_success_counts_each_candidate_tier_once():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        snippet = _make_snippet()
        decision = _baseline_save_decision()

        for _ in range(2):
            p._finish_runtime_stage_success(
                attempt_id=None,
                stage="full",
                snippet=snippet,
                decision=decision,
            )

        assert p.stats["outreach_tier_counts"] == {"PRIORITY": 1}


def test_full_evaluate_refuses_a_projectless_save_and_leaves_the_stage_nonterminal():
    """F1 acceptance (2), end to end through the PRODUCTION caller.

    `_full_evaluate` is the orchestrator method that consumes the save outcome
    and marks the full stage terminal (`_mark_terminal` +
    `_finish_runtime_stage_success`). This drives it with the real
    LinkedInSideEffectsService on a page carrying no Recruiter project id —
    the F1 bypass condition, which the E4 predicate read as "not a mismatch".

    Pre-fix, the save click landed, the canonical receipt read succeeded, and
    the candidate reached `full_terminal` in whatever pipeline that page
    belonged to. The refusal must leave every one of those unwritten.
    """
    import linkedin.orchestrator as orchestrator_mod

    projectless_page_url = (
        "https://www.linkedin.com/talent/recruiterSearch/profile/ada"
    )
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(
            id=1, name="owner", boolean="one", status="in_progress"
        )
        p._runtime_run_id, progress = p._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=Progress(brief_name="test", strings=[owner]),
        )
        p._progress = progress
        snippet = _make_snippet(
            name="Ada",
            profile_url="/talent/profile/ada",
            source_string_id=owner.id,
            source_string_name=owner.name,
        )
        p._record_runtime_snippet(owner, snippet)
        p._bias_monitor = None
        p._triage_tightened = False
        p._tightening_prefix = ""
        p._ensure_services()

        acquisition = MagicMock()
        acquisition.profile_summary = CandidateProfileSummary(
            name=snippet.name,
            profile_url=snippet.profile_url,
            headline="Engineer",
            experiences=[],
            education=[],
            skills_snippet=[],
        )
        p._acquisition_service.extract_profile_summary = AsyncMock(
            return_value=acquisition
        )
        p.browser.get_profile_status_summary = AsyncMock(return_value={})
        p.browser.go_back_to_results = AsyncMock()
        p.browser.page = MagicMock(url=projectless_page_url)
        p.browser.current_profile_identity_fragment.return_value = "ada"
        p.browser.is_already_saved_on_card = AsyncMock(return_value=False)
        p.browser.save_candidate = AsyncMock(return_value=True)
        p.browser.scroll_for_linger = AsyncMock(return_value=0)
        p.browser.scroll_restore = AsyncMock()
        p._reopen_profile_for_full_eval_save = AsyncMock()
        p._derive_novelty_value = MagicMock(return_value=("medium", "rationale"))

        with patch(
            "linkedin.orchestrator.full_judge",
            return_value=_baseline_save_decision(
                name=snippet.name,
                url=snippet.profile_url,
            ),
        ), patch(
            "linkedin.orchestrator.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ), patch(
            "linkedin.orchestrator.config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS",
            0,
        ), patch(
            "linkedin.side_effects.asyncio.sleep",
            new=AsyncMock(),
        ), pytest.raises(orchestrator_mod.ProjectContextMismatchError):
            asyncio.run(
                p._full_evaluate(snippet, search_string=progress.strings[0])
            )

        p.browser.save_candidate.assert_not_awaited()
        receipt_rows = p._runtime_state.list_candidate_side_effects(
            source="linkedin",
            brief_id="test-project",
        )
        assert [row["status"] for row in receipt_rows] == ["failed"]
        candidate = p._runtime_state.get_candidate(
            source="linkedin",
            brief_id="test-project",
            identity_key=snippet.profile_url,
        )
        assert candidate["current_lifecycle_state"] != "full_terminal"
        assert snippet.profile_url not in p._seen_urls
        assert snippet.profile_url not in p._prior_outcomes


# Test A — feature off (default): shadow path is not entered at all.
def test_full_evaluate_feature_off_skips_shadow_path():
    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        snippet = _make_snippet()
        baseline = _baseline_save_decision()
        save_outcome = p._side_effects_service.handle_save_decision.return_value

        async def assert_tier_counted_before_save(**_kwargs):
            assert p.stats["outreach_tier_counts"] == {"PRIORITY": 1}
            return save_outcome

        p._side_effects_service.handle_save_decision.side_effect = (
            assert_tier_counted_before_save
        )

        with patch("linkedin.orchestrator.config") as mock_cfg, \
             patch("linkedin.orchestrator.full_judge", return_value=baseline), \
             patch(
                 "linkedin.orchestrator.should_request_external_evidence"
             ) as mock_gate, \
             patch(
                 "linkedin.orchestrator.fetch_external_candidate_evidence"
             ) as mock_fetch, \
             patch(
                 "linkedin.orchestrator.full_judge_with_external_evidence"
             ) as mock_enrich, \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0.0):
            mock_cfg.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED = False
            mock_cfg.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MIN_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MAX_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_BASE_SECONDS = 0
            decision = asyncio.run(p._full_evaluate(snippet))

        assert decision.decision == "SAVE"
        assert p.stats["outreach_tier_counts"] == {"PRIORITY": 1}
        assert mock_gate.call_count == 0
        assert mock_fetch.call_count == 0
        assert mock_enrich.call_count == 0
        # Shadow file must NOT exist when feature is off.
        assert not Path(td, "shadow_final_judgments.jsonl").exists()
        # Save side-effect was awaited exactly once with the baseline decision.
        _assert_baseline_canonical_preserved(
            p, expected_decision="SAVE", expected_save_called=True
        )
        # Tighten: the SAME snippet object reached the side-effect call (not
        # a copy or a substitute built from the enriched path).
        assert (
            p._side_effects_service.handle_save_decision.await_args.kwargs["snippet"]
            is snippet
        )


# Test B — feature on, gate decides to skip.
def test_full_evaluate_feature_on_gate_skip_writes_skipped_record():
    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        snippet = _make_snippet()

        skip_decision = TriggerDecision(
            should_run=False,
            reason="",
            skip_reason="no_trigger_matched",
            signals={"experience_count": 4, "fired": "none"},
        )

        with patch("linkedin.orchestrator.config") as mock_cfg, \
             patch("linkedin.orchestrator.full_judge", return_value=_baseline_save_decision()), \
             patch(
                 "linkedin.orchestrator.should_request_external_evidence",
                 return_value=skip_decision,
             ) as mock_gate, \
             patch(
                 "linkedin.orchestrator.fetch_external_candidate_evidence"
             ) as mock_fetch, \
             patch(
                 "linkedin.orchestrator.full_judge_with_external_evidence"
             ) as mock_enrich, \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0.0):
            mock_cfg.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED = True
            mock_cfg.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MIN_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MAX_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_BASE_SECONDS = 0
            asyncio.run(p._full_evaluate(snippet))

        assert mock_gate.call_count == 1
        assert mock_fetch.call_count == 0
        assert mock_enrich.call_count == 0

        records = _read_shadow(td)
        assert len(records) == 1
        rec = records[0]
        assert rec["external_evidence_status"] == "skipped_no_trigger"
        assert rec["enriched"] is None
        assert rec["diff"]["computed"] is False
        assert rec["evidence_refs_count"] == 0
        assert rec["identity_confidence"] is None
        assert rec["feature_version"] == "slice2"
        assert rec["candidate_name"] == snippet.name
        assert rec["profile_url"] == snippet.profile_url
        # Tighten: the gate-skip path must NOT block the save click — baseline
        # said SAVE, so the side-effect runs exactly once with the same snippet.
        # (Trigger signals are not persisted on ShadowFullJudgmentRecord, so
        # we do not assert on rec["signals"]; the field does not exist.)
        _assert_baseline_canonical_preserved(
            p, expected_decision="SAVE", expected_save_called=True
        )
        assert (
            p._side_effects_service.handle_save_decision.await_args.kwargs["snippet"]
            is snippet
        )


# Test C — feature on, gate fires, evidence returned, enriched judge differs from baseline.
def test_full_evaluate_enriched_decision_changed_does_not_replace_baseline():
    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        snippet = _make_snippet()

        trigger = TriggerDecision(
            should_run=True,
            reason="academic_context",
            skip_reason="",
            signals={"fired": "academic_context"},
        )
        evidence = _evidence_with_two_refs()
        baseline = _baseline_save_decision()
        enriched = OpusDecision(
            stage="full",
            decision="REJECT",
            path="none",
            confidence=0.55,
            rationale="External evidence undermines fit.",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )

        with patch("linkedin.orchestrator.config") as mock_cfg, \
             patch("linkedin.orchestrator.full_judge", return_value=baseline), \
             patch(
                 "linkedin.orchestrator.should_request_external_evidence",
                 return_value=trigger,
             ), \
             patch(
                 "linkedin.orchestrator.fetch_external_candidate_evidence",
                 return_value=evidence,
             ), \
             patch(
                 "linkedin.orchestrator.full_judge_with_external_evidence",
                 return_value=enriched,
             ), \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0.0):
            mock_cfg.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED = True
            mock_cfg.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MIN_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MAX_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_BASE_SECONDS = 0
            returned = asyncio.run(p._full_evaluate(snippet))

        # Canonical decision is the BASELINE, not the enriched one.
        assert returned.decision == "SAVE"
        # Save side-effect was called exactly once with the baseline-shaped decision.
        # The helper also pins that no `decision` kwarg reaches the call site —
        # i.e., the enriched OpusDecision is structurally incapable of leaking
        # into the side-effect (handle_save_decision's signature only accepts
        # snippet/runtime_search_string/attempt_id; see orchestrator.py:4162).
        _assert_baseline_canonical_preserved(
            p, expected_decision="SAVE", expected_save_called=True
        )
        assert (
            p._side_effects_service.handle_save_decision.await_args.kwargs["snippet"]
            is snippet
        )

        records = _read_shadow(td)
        assert len(records) == 1
        rec = records[0]
        assert rec["external_evidence_status"] == "evidence_present"
        assert rec["evidence_refs_count"] == 2
        assert rec["identity_confidence"] == pytest.approx(0.7)
        assert rec["diff"]["computed"] is True
        assert rec["diff"]["decision_changed"] is True
        assert rec["diff"]["decision_baseline"] == "SAVE"
        assert rec["diff"]["decision_enriched"] == "REJECT"
        # Tighten: pin the raw top-level decision dicts too (not just the diff
        # summary). The shadow record snapshots both OpusDecision.to_dict()s.
        assert rec["baseline"]["decision"] == "SAVE"
        assert rec["enriched"]["decision"] == "REJECT"


# Test D — feature on, provider returns ExternalEvidenceFailure(reason="quota_exhausted").
def test_full_evaluate_provider_failure_does_not_propagate_or_call_enrichment():
    """Baseline canonical state survives a quota_exhausted provider failure.

    NOTE on absorption: the slice 2 absorption layer is the try/except in
    ``_full_evaluate`` itself (no ApiBudgetExhaustedError propagates), already
    pinned below. We deliberately do NOT assert
    ``is_api_budget_exhausted_error(detail) is False`` here, because this
    test's failure detail ("credit balance is too low") is literally one of
    the recognized substring patterns in ``shared/failures.py`` — that string
    helper IS supposed to flag such messages when given them. The invariant
    "the provider's quota detail must not later trip the run-pause helper"
    is enforced by the orchestrator's absorption (no exception escapes this
    code path), not by the helper's string-matching contract. The
    parametrized test below asserts the helper-string property for failure
    shapes whose details do not contain budget patterns.
    """
    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        snippet = _make_snippet()

        trigger = TriggerDecision(
            should_run=True,
            reason="academic_context",
            skip_reason="",
            signals={"fired": "academic_context"},
        )
        failure = ExternalEvidenceFailure(
            reason="quota_exhausted",
            detail="credit balance is too low",
            provider="perplexity",
            http_status=402,
        )

        with patch("linkedin.orchestrator.config") as mock_cfg, \
             patch("linkedin.orchestrator.full_judge", return_value=_baseline_save_decision()), \
             patch(
                 "linkedin.orchestrator.should_request_external_evidence",
                 return_value=trigger,
             ), \
             patch(
                 "linkedin.orchestrator.fetch_external_candidate_evidence",
                 return_value=failure,
             ), \
             patch(
                 "linkedin.orchestrator.full_judge_with_external_evidence"
             ) as mock_enrich, \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0.0):
            mock_cfg.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED = True
            mock_cfg.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MIN_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MAX_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_BASE_SECONDS = 0
            try:
                returned = asyncio.run(p._full_evaluate(snippet))
            except ApiBudgetExhaustedError as exc:  # pragma: no cover
                pytest.fail(
                    f"Shadow path must absorb ExternalEvidenceFailure; got {exc!r}"
                )

        assert returned.decision == "SAVE"
        assert mock_enrich.call_count == 0

        records = _read_shadow(td)
        assert len(records) == 1
        rec = records[0]
        assert rec["external_evidence_status"] == "quota_exhausted"
        assert rec["enriched"] is None
        assert rec["diff"]["computed"] is False
        assert rec["evidence_refs_count"] == 0
        # Tighten: provider failure must not block the save click — baseline
        # said SAVE, so the side-effect runs exactly once with the baseline.
        _assert_baseline_canonical_preserved(
            p, expected_decision="SAVE", expected_save_called=True
        )


# Test E — feature on, evidence returned, enriched judge raises.
def test_full_evaluate_enrichment_exception_is_absorbed():
    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        snippet = _make_snippet()

        trigger = TriggerDecision(
            should_run=True, reason="academic_context", skip_reason="", signals={}
        )
        evidence = _evidence_with_two_refs()

        with patch("linkedin.orchestrator.config") as mock_cfg, \
             patch("linkedin.orchestrator.full_judge", return_value=_baseline_save_decision()), \
             patch(
                 "linkedin.orchestrator.should_request_external_evidence",
                 return_value=trigger,
             ), \
             patch(
                 "linkedin.orchestrator.fetch_external_candidate_evidence",
                 return_value=evidence,
             ), \
             patch(
                 "linkedin.orchestrator.full_judge_with_external_evidence",
                 side_effect=RuntimeError("opus broke"),
             ), \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0.0):
            mock_cfg.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED = True
            mock_cfg.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MIN_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MAX_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_BASE_SECONDS = 0
            returned = asyncio.run(p._full_evaluate(snippet))

        assert returned.decision == "SAVE"
        records = _read_shadow(td)
        assert len(records) == 1
        rec = records[0]
        assert rec["external_evidence_status"] == "evidence_present"
        assert rec["enriched"] is None
        assert rec["diff"]["computed"] is False
        # Tighten: when the gate fired and the provider succeeded but only the
        # enricher broke, compute_judgment_diff records the skip_reason as the
        # external_evidence_status ("evidence_present"). Pinning this confirms
        # the diff dict carries the same status forward into the analytics row.
        assert rec["diff"]["reason"] == "evidence_present"
        # Tighten: enrichment failure must not block the save click — baseline
        # said SAVE, so the side-effect runs exactly once with the baseline.
        _assert_baseline_canonical_preserved(
            p, expected_decision="SAVE", expected_save_called=True
        )


# Test F — feature on, writer itself raises; outer try/except must absorb it.
def test_full_evaluate_writer_exception_is_absorbed():
    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        snippet = _make_snippet()

        trigger = TriggerDecision(
            should_run=False, reason="", skip_reason="no_trigger_matched", signals={}
        )

        with patch("linkedin.orchestrator.config") as mock_cfg, \
             patch("linkedin.orchestrator.full_judge", return_value=_baseline_save_decision()), \
             patch(
                 "linkedin.orchestrator.should_request_external_evidence",
                 return_value=trigger,
             ), \
             patch(
                 "linkedin.orchestrator.record_shadow_full_judgment",
                 side_effect=RuntimeError("disk on fire"),
             ), \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0.0):
            mock_cfg.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED = True
            mock_cfg.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MIN_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MAX_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_BASE_SECONDS = 0
            returned = asyncio.run(p._full_evaluate(snippet))

        # Baseline canonical state is preserved.
        assert returned.decision == "SAVE"
        # Shadow file should not exist (writer was patched to raise before write).
        assert not Path(td, "shadow_final_judgments.jsonl").exists()
        # Save side-effect was awaited exactly once with the baseline-shaped
        # decision (writer-failure path must not block the save click).
        _assert_baseline_canonical_preserved(
            p, expected_decision="SAVE", expected_save_called=True
        )


# Test G — baseline parse/judgment failure: shadow block is skipped entirely.
def test_full_evaluate_baseline_parse_failure_skips_shadow():
    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        snippet = _make_snippet()

        with patch("linkedin.orchestrator.config") as mock_cfg, \
             patch(
                 "linkedin.orchestrator.full_judge",
                 return_value=_baseline_parse_failure(),
             ), \
             patch(
                 "linkedin.orchestrator.should_request_external_evidence"
             ) as mock_gate, \
             patch(
                 "linkedin.orchestrator.fetch_external_candidate_evidence"
             ) as mock_fetch, \
             patch(
                 "linkedin.orchestrator.full_judge_with_external_evidence"
             ) as mock_enrich, \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0.0):
            mock_cfg.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED = True
            mock_cfg.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MIN_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MAX_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_BASE_SECONDS = 0
            returned = asyncio.run(p._full_evaluate(snippet))

        # Parse-failure path returns early in _full_evaluate (well before the
        # bias monitor block), so the shadow block isn't reached. Either way,
        # nothing should be written.
        assert returned.decision == "PARSE_FAILURE"
        assert mock_gate.call_count == 0
        assert mock_fetch.call_count == 0
        assert mock_enrich.call_count == 0
        assert not Path(td, "shadow_final_judgments.jsonl").exists()
        # Tighten: parse-failure path returns early before the save-click
        # branch (orchestrator.py:~3966), so the side-effect must NOT have
        # been awaited at all. The helper pins this with await_count == 0.
        _assert_baseline_canonical_preserved(
            p, expected_decision="PARSE_FAILURE", expected_save_called=False
        )


# Tests A–G above cover one provider-failure shape (quota_exhausted) plus the
# binary feature off/on / gate skip / enriched differs / enrichment exception
# / writer exception / parse-failure paths. The parametrized test below covers
# the seven remaining provider-failure shapes from the slice 2 design with a
# single test body, asserting the same canonical-preservation invariants on
# every shape.
@pytest.mark.parametrize(
    "failure_reason,failure_detail,http_status",
    [
        ("disabled_no_api_key", "PERPLEXITY_API_KEY not set", None),
        ("disabled_by_config", "LINKEDIN_EXTERNAL_EVIDENCE_ENABLED is False", None),
        ("timeout", "request exceeded 90 seconds", None),
        ("parse_failure", "expected JSON object, got list", None),
        (
            "weak_citations",
            "0 citations across 0 fact blocks (minimum: 2)",
            None,
        ),
        (
            "unknown",
            "unexpected: ConnectionResetError(54, 'Connection reset by peer')",
            None,
        ),
        # http_error is structurally similar to the other failure shapes; we
        # cover one with an http_status to exercise that field's persistence.
        ("http_error", "Bad Gateway", 502),
    ],
)
def test_full_evaluate_each_provider_failure_preserves_baseline(
    failure_reason, failure_detail, http_status,
):
    """Every provider-failure reason must absorb cleanly: baseline persists,
    no ApiBudgetExhaustedError propagates, save side effect runs with baseline.

    Note on _finish_runtime_stage_success: ``_make_full_evaluate_pipeline``
    leaves it as the real method (not a Mock); it short-circuits when
    ``_runtime_bridge``/``_runtime_run_id`` are None on the fake pipeline (see
    fixture comment at lines ~2254–2257). We therefore cannot assert a call
    count on it here — the canonical-persistence pin is instead expressed via
    the returned baseline decision and the save-click invariant.
    """
    with tempfile.TemporaryDirectory() as td:
        p, _ = _make_full_evaluate_pipeline(td)
        snippet = _make_snippet()

        trigger = TriggerDecision(
            should_run=True,
            reason="academic_context",
            skip_reason="",
            signals={"fired": "academic_context"},
        )
        failure = ExternalEvidenceFailure(
            reason=failure_reason,
            detail=failure_detail,
            provider="perplexity",
            http_status=http_status,
        )

        with patch("linkedin.orchestrator.config") as mock_cfg, \
             patch("linkedin.orchestrator.full_judge", return_value=_baseline_save_decision()), \
             patch(
                 "linkedin.orchestrator.should_request_external_evidence",
                 return_value=trigger,
             ), \
             patch(
                 "linkedin.orchestrator.fetch_external_candidate_evidence",
                 return_value=failure,
             ) as mock_fetch, \
             patch(
                 "linkedin.orchestrator.full_judge_with_external_evidence"
             ) as mock_enrich, \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0.0):
            mock_cfg.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED = True
            mock_cfg.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MIN_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_MAX_SECONDS = 0
            mock_cfg.LINKEDIN_REJECT_CLOSE_BASE_SECONDS = 0
            try:
                returned = asyncio.run(p._full_evaluate(snippet))
            except ApiBudgetExhaustedError as exc:  # pragma: no cover
                pytest.fail(
                    f"Shadow path must absorb ExternalEvidenceFailure(reason={failure_reason!r}); "
                    f"got {exc!r}"
                )

        # Baseline returned the SAVE decision — canonical state is unaffected.
        assert returned.decision == "SAVE"
        # The provider's ExternalEvidenceFailure was returned by the fetch
        # mock; the enricher must not have been called.
        assert mock_fetch.call_count == 1
        assert mock_enrich.call_count == 0

        # The slice 2 absorption layer must prevent the provider's failure
        # detail from later tripping is_api_budget_exhausted_error. None of
        # these seven detail strings should match the helper's billing
        # patterns; if they ever shift to contain "credit balance is too low"
        # / "quota exceeded" / etc., this assertion will catch it.
        assert is_api_budget_exhausted_error(failure_detail) is False
        assert is_api_budget_exhausted_error(failure_reason) is False

        # Shadow row must record the typed failure verbatim.
        records = _read_shadow(td)
        assert len(records) == 1
        rec = records[0]
        assert rec["external_evidence_status"] == failure_reason
        assert rec["enriched"] is None
        assert rec["diff"]["computed"] is False
        assert rec["evidence_refs_count"] == 0
        assert rec["identity_confidence"] is None

        # Canonical-preservation invariants: the save side-effect must have
        # been awaited exactly once with the baseline-shaped decision.
        _assert_baseline_canonical_preserved(
            p, expected_decision="SAVE", expected_save_called=True
        )
        assert (
            p._side_effects_service.handle_save_decision.await_args.kwargs["snippet"]
            is snippet
        )


# ---------------------------------------------------------------------------
# Ternary facial outcomes remain distinct through persistence and full review
# ---------------------------------------------------------------------------
# Ternary posture: FACIAL_BORDERLINE stays distinct and reaches full review.
# Flag-off: FACIAL_BORDERLINE -> PARSE_FAILURE (fail-loud); routes through
# the standard non-terminal failure path.
# ---------------------------------------------------------------------------


def _make_borderline_decision(name: str = "Test Person", url: str = "/talent/profile/test123") -> OpusDecision:
    return OpusDecision(
        stage="facial",
        decision="FACIAL_BORDERLINE",
        path="none",
        confidence=1.0,
        rationale="snippet matches an ambiguous trajectory",
        candidate_name=name,
        profile_url=url,
        prompt_capture={"logical_call_id": "judge-borderline", "judge_receipt": {"receipt_id": "r1"}},
    )


@pytest.mark.parametrize(
    ("ternary_enabled", "expected_decision"),
    [(True, "FACIAL_BORDERLINE"), (False, "PARSE_FAILURE")],
)
def test_facial_borderline_normalization_preserves_prompt_capture(
    ternary_enabled,
    expected_decision,
):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        original = _make_borderline_decision()
        with patch.object(
            p,
            "_facial_ternary_enabled",
            return_value=ternary_enabled,
        ):
            normalized = p._normalize_facial_decision_for_persistence(original)

    assert normalized.decision == expected_decision
    assert normalized.prompt_capture == original.prompt_capture


def test_facial_borderline_stays_distinct_and_is_full_reviewed_under_ternary():
    """Ternary BORDERLINE is durable review demand, not an aliased YES."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._bias_monitor = None
        p._full_evaluate = AsyncMock(return_value=None)

        snippet = _make_snippet()

        with patch("linkedin.orchestrator.facial_judge", return_value=_make_borderline_decision()), \
             patch("linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED", True):
            asyncio.run(p._evaluate_snippet(snippet))

        assert p._prior_outcomes[snippet.profile_url] == "FACIAL_BORDERLINE"
        assert p.stats["facial_yes"] == 0
        assert p.stats["facial_borderline"] == 1
        assert p.stats["facial_no"] == 0
        assert p.stats.get("parse_failures", 0) == 0
        p._full_evaluate.assert_awaited_once()


def test_facial_borderline_under_flag_off_routes_to_parse_failure():
    """Flag-off: FACIAL_BORDERLINE is a structural surprise; route to PARSE_FAILURE.

    The orchestrator must NOT silently coerce to YES or NO; it must fall
    into the standard non-terminal parse-failure path. Persistence must
    not record FACIAL_BORDERLINE or FACIAL_YES.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._bias_monitor = None
        p._full_evaluate = AsyncMock(return_value=None)

        snippet = _make_snippet()

        with patch("linkedin.orchestrator.facial_judge", return_value=_make_borderline_decision()), \
             patch("linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED", False):
            returned = asyncio.run(p._evaluate_snippet(snippet))

        assert returned is not None
        assert returned.decision == "PARSE_FAILURE"
        assert p.stats.get("parse_failures", 0) == 1
        assert p.stats["facial_yes"] == 0
        assert p.stats["facial_no"] == 0
        assert p._prior_outcomes.get(snippet.profile_url) != "FACIAL_BORDERLINE"
        assert p._prior_outcomes.get(snippet.profile_url) != "FACIAL_YES"
        p._full_evaluate.assert_not_awaited()


def test_facial_yes_unchanged_under_flag_on():
    """Regression: flag-on must not perturb the FACIAL_YES path."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._bias_monitor = None
        p._full_evaluate = AsyncMock(return_value=None)

        snippet = _make_snippet()
        yes_decision = OpusDecision(
            stage="facial", decision="FACIAL_YES", path="none",
            confidence=1.0, rationale="strong signal",
            candidate_name=snippet.name, profile_url=snippet.profile_url,
        )

        with patch("linkedin.orchestrator.facial_judge", return_value=yes_decision), \
             patch("linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED", True):
            asyncio.run(p._evaluate_snippet(snippet))

        assert p._prior_outcomes[snippet.profile_url] == "FACIAL_YES"
        assert p.stats["facial_yes"] == 1
        p._full_evaluate.assert_awaited_once()


def test_facial_no_unchanged_under_flag_on():
    """Regression: flag-on must not perturb the FACIAL_NO path."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._bias_monitor = None
        p._full_evaluate = AsyncMock(return_value=None)

        snippet = _make_snippet()
        no_decision = OpusDecision(
            stage="facial", decision="FACIAL_NO", path="none",
            confidence=1.0, rationale="non-fit pattern",
            candidate_name=snippet.name, profile_url=snippet.profile_url,
        )

        with patch("linkedin.orchestrator.facial_judge", return_value=no_decision), \
             patch("linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED", True):
            returned = asyncio.run(p._evaluate_snippet(snippet))

        assert returned.decision == "FACIAL_NO"
        assert p._prior_outcomes[snippet.profile_url] == "FACIAL_NO"
        assert p.stats["facial_no"] == 1
        assert p.stats["facial_yes"] == 0
        p._full_evaluate.assert_not_awaited()


def test_facial_borderline_is_preserved_in_canonical_success_and_prior_outcomes():
    """The canonical facial attempt and resume map retain the ternary class."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._bias_monitor = None
        p._full_evaluate = AsyncMock(return_value=None)

        captured_decisions: list[str] = []
        original_finish = p._finish_runtime_stage_success

        def _capture(*args, **kwargs):
            captured_decisions.append(kwargs["decision"].decision)
            return original_finish(*args, **kwargs)

        p._finish_runtime_stage_success = _capture

        snippets = [
            _make_snippet(name="Alice", profile_url="/talent/profile/alice"),
            _make_snippet(name="Bob", profile_url="/talent/profile/bob"),
            _make_snippet(name="Carol", profile_url="/talent/profile/carol"),
        ]
        decisions = [
            OpusDecision(stage="facial", decision="FACIAL_YES", path="none",
                         confidence=1.0, rationale="strong",
                         candidate_name="Alice", profile_url="/talent/profile/alice"),
            _make_borderline_decision(name="Bob", url="/talent/profile/bob"),
            OpusDecision(stage="facial", decision="FACIAL_NO", path="none",
                         confidence=1.0, rationale="non-fit",
                         candidate_name="Carol", profile_url="/talent/profile/carol"),
        ]

        with patch("linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED", True):
            for snippet, decision in zip(snippets, decisions):
                with patch("linkedin.orchestrator.facial_judge", return_value=decision):
                    asyncio.run(p._evaluate_snippet(snippet))

        assert captured_decisions == [
            "FACIAL_YES",
            "FACIAL_BORDERLINE",
            "FACIAL_NO",
        ]
        assert p._prior_outcomes["/talent/profile/alice"] == "FACIAL_YES"
        assert p._prior_outcomes["/talent/profile/bob"] == "FACIAL_BORDERLINE"
        assert p._prior_outcomes["/talent/profile/carol"] == "FACIAL_NO"
        assert p.stats["facial_yes"] == 1
        assert p.stats["facial_borderline"] == 1
        assert p.stats["facial_no"] == 1
        assert p._full_evaluate.await_count == 2


def test_batch_early_exit_drains_yes_and_distinct_borderline_before_stopping():
    """Early exit stops pagination only; all acquired positives get full review."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._bias_monitor = None
        p._triage_tightened = False

        full_eval_calls: list[str] = []

        async def _capture_full_evaluate(snippet, *args, **kwargs):
            full_eval_calls.append(snippet.profile_url)
            return None

        p._full_evaluate = AsyncMock(side_effect=_capture_full_evaluate)
        p.browser.get_card_slot_count = AsyncMock(return_value=3)
        p._extract_card_snippet = AsyncMock(side_effect=[
            _make_snippet(name="Alice", profile_url="/talent/profile/alice"),
            _make_snippet(name="Bob", profile_url="/talent/profile/bob"),
            _make_snippet(name="Carol", profile_url="/talent/profile/carol"),
        ])
        p._checkpoint_progress = MagicMock()

        search_string = SearchString(id=1, name="batch", boolean="ml")
        page_report = MagicMock()
        all_candidates: list[dict] = []
        string_stats = {
            "pages": 1, "candidates": 0, "duplicates": 0,
            "facial_yes": 0, "facial_no": 0, "saves": 0, "rejects": 0,
        }

        batch_decisions = [
            OpusDecision(stage="facial", decision="FACIAL_YES", path="none",
                         confidence=1.0, rationale="strong",
                         candidate_name="Alice", profile_url="/talent/profile/alice"),
            _make_borderline_decision(name="Bob", url="/talent/profile/bob"),
            OpusDecision(stage="facial", decision="FACIAL_NO", path="none",
                         confidence=1.0, rationale="non-fit",
                         candidate_name="Carol", profile_url="/talent/profile/carol"),
        ]

        p._get_early_exit_rate = MagicMock(return_value=0.30)

        with patch("shared.judger.facial_judge_batch", return_value=batch_decisions), \
             patch("linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED", True), \
             patch("linkedin.orchestrator.config.FULL_EVAL_PIPELINE_ENABLED", False), \
             patch("linkedin.orchestrator.config.EARLY_EXIT_MIN_CANDIDATES", 3):
            asyncio.run(
                p._review_page_batch(
                    search_string, 1, 0, page_report, all_candidates, string_stats, None,
                )
            )

        assert "/talent/profile/alice" in full_eval_calls
        assert "/talent/profile/bob" in full_eval_calls
        assert "/talent/profile/carol" not in full_eval_calls
        assert p._prior_outcomes["/talent/profile/alice"] == "FACIAL_YES"
        assert p._prior_outcomes["/talent/profile/bob"] == "FACIAL_BORDERLINE"
        assert p._prior_outcomes["/talent/profile/carol"] == "FACIAL_NO"
        assert p.stats["facial_yes"] == 1
        assert p.stats["facial_borderline"] == 1
        assert p.stats["facial_no"] == 1
        assert string_stats["facial_yes"] == 1
        assert string_stats["facial_borderline"] == 1
        assert p._page_observation()["break_reason"] == "early_exit"
        outcomes = [c["outcome"] for c in all_candidates]
        assert outcomes == ["facial_yes", "facial_borderline", "facial_no"]


@pytest.mark.parametrize("full_decision", ["REJECT", "SAVE"])
def test_v2_batch_nonpipelined_success_clears_full_review_obligation(full_decision):
    """The resolved live V2 path must consume the obligation it creates."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._bias_monitor = None
        p._tightening_prefix = ""
        p._triage_tightened = False
        snippet = _make_snippet(name="Ada", profile_url="/talent/profile/ada")
        facial = OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="none",
            confidence=1.0,
            rationale="strong",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        full = OpusDecision(
            stage="full",
            decision=full_decision,
            path="DIRECT:test" if full_decision == "SAVE" else "none",
            confidence=0.9,
            rationale="settled",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        Progress(
            brief_name="test",
            strings=[SearchString(id=1, name="batch", boolean="ml")],
        ).save(str(p.progress_path))
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._session_expired = MagicMock()
        p._session_expired.is_set.return_value = False
        p.browser.enter_search_string = AsyncMock()
        p.browser.get_card_slot_count = AsyncMock(return_value=1)
        p.browser.get_profile_status_summary = AsyncMock(return_value={})
        p.browser.go_back_to_results = AsyncMock()
        p._extract_card_snippet = AsyncMock(return_value=snippet)
        p._get_early_exit_rate = MagicMock(return_value=0.0)
        p._apply_session_location_filter = AsyncMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._finalize_run_snapshot = MagicMock(
            return_value=Path(td, "frozen")
        )
        p._enrich_run_snapshot = MagicMock()
        _allow_synthetic_run_completion(p)
        p.browser.find_result_slot_by_profile_url = AsyncMock()
        p._ensure_services()
        acquisition = MagicMock()
        acquisition.profile_summary = MagicMock()
        acquisition.profile_summary.to_dict.return_value = {}
        p._acquisition_service.extract_profile_summary = AsyncMock(
            return_value=acquisition
        )
        from shared.execution import SideEffectOutcome
        p._side_effects_service.handle_save_decision = AsyncMock(
            return_value=SideEffectOutcome(
                effect_type="linkedin_save",
                status="succeeded",
                payload={},
            )
        )
        p._reopen_profile_for_full_eval_save = AsyncMock()
        p._derive_novelty_value = MagicMock(
            return_value=("medium", "rationale")
        )

        async def process(search_string, progress):
            await p._review_page_batch(
                search_string,
                1,
                0,
                MagicMock(),
                [],
                p._fresh_string_stats(),
                progress,
            )

        p._process_string = process

        with patch("shared.judger.facial_judge_batch", return_value=[facial]), \
             patch("linkedin.orchestrator.full_judge", return_value=full), \
             patch("linkedin.orchestrator.config.FULL_EVAL_PIPELINE_ENABLED", False), \
             patch("linkedin.orchestrator.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
             patch("linkedin.orchestrator.config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS", 0), \
             patch("linkedin.orchestrator.config.LINKEDIN_REJECT_CLOSE_MIN_SECONDS", 0), \
             patch("linkedin.orchestrator.config.LINKEDIN_REJECT_CLOSE_MAX_SECONDS", 0), \
             patch("linkedin.orchestrator.config.LINKEDIN_REJECT_CLOSE_BASE_SECONDS", 0), \
             patch("linkedin.orchestrator.human_delay_correlated", return_value=0):
            asyncio.run(p.run_full(resume=True))

        assert p._progress.strings[0].status == "done"
        with p._runtime_state.connect() as conn:
            full_attempts = conn.execute(
                "SELECT COUNT(*) FROM candidate_attempts WHERE stage = 'full'"
            ).fetchone()[0]
        assert full_attempts == 1
        assert p._resume_pending_full_decisions == {}
        assert p._resume_pending_full_snippets == {}
        assert p._resume_pending_full_owner_ids == {}
        p.browser.find_result_slot_by_profile_url.assert_not_awaited()


def test_resume_connect_abort_checkpoints_hydrated_owner_demotion():
    """A cloned terminal owner is corrected before Recruiter setup can fail."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(id=1, name="owner", boolean="one", status="done")
        later = SearchString(id=2, name="later", boolean="two", status="queued")
        Progress(brief_name="test", strings=[owner, later]).save(
            str(p.progress_path)
        )
        pending = _make_snippet(
            name="Pending",
            profile_url="/talent/profile/pending",
            source_string_id=owner.id,
        )

        def hydrate(_progress):
            p._track_full_review_obligation(pending, "FACIAL_YES")

        p._hydrate_resume_funnel_from_runtime = MagicMock(side_effect=hydrate)
        p.browser.connect = AsyncMock(side_effect=RuntimeError("connect failed"))
        p.browser.disconnect = AsyncMock()
        p._finalize_run_snapshot = MagicMock(return_value=Path(td, "frozen"))

        with pytest.raises(RuntimeError, match="connect failed"):
            asyncio.run(p.run_full(resume=True))

        latest = p._runtime_state.get_latest_run(
            source="linkedin",
            brief_id=p.brief_obj.linkedin_project_id,
        )
        statuses = [
            row["status"]
            for row in p._runtime_state.list_work_units(
                int(latest["id"]),
                kind="linkedin_string",
            )
        ]
        assert statuses == ["in_progress", "queued"]


def test_run_full_second_disconnect_does_not_invoke_recovery_twice():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=5,
                    name="interrupted",
                    boolean='"ML" AND "engineer"',
                    status="queued",
                    block="Block A",
                    lane_id="ml-lane",
                )
            ],
            current_page=2,
        )
        progress.save(str(p.progress_path))

        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        # Project id matches the fixture brief ("test-project"): this test is
        # about the closed-browser abort, so run-start must take the
        # already-on-the-right-project path and not navigate (E4).
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p.browser.get_current_search_url = MagicMock(
            return_value="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p.browser.snapshot_advanced_search_controls = AsyncMock(return_value={"controls": []})
        snapshot = MagicMock(name="snapshot")
        p._capture_recovery_snapshot = AsyncMock(return_value=snapshot)
        p._recovery_service.recover = AsyncMock(return_value=True)
        p._reassert_session_location_after_recovery = AsyncMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()

        failures = [
            RuntimeError(
                "Page.evaluate: Target page, context or browser has been closed"
            ),
            RuntimeError("Page.evaluate: Target crashed again"),
        ]
        calls = {"count": 0}

        async def fake_process(search_string, progress):
            calls["count"] += 1
            raise failures[calls["count"] - 1]

        p._process_string = fake_process

        with pytest.raises(RuntimeError, match="has been closed") as raised:
            asyncio.run(p.run_full(resume=True))

        assert raised.value is failures[0]
        p._capture_recovery_snapshot.assert_awaited_once()
        p._recovery_service.recover.assert_awaited_once_with(
            run_id=p._runtime_run_id,
            snapshot=snapshot,
        )
        p._reassert_session_location_after_recovery.assert_awaited_once_with()
        assert calls["count"] == 2
        saved = json.loads(Path(td, "progress.json").read_text())
        assert saved["strings"][0]["status"] == "in_progress"


def test_process_string_runs_variant_lifecycle_after_probe_metrics():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        search_string = SearchString(id=9, name="exp", boolean="foo", status="queued")
        progress = Progress(brief_name="test", strings=[search_string], current_string_id=9, current_page=0)

        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results

        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p.browser.page.locator.return_value = locator
        p.browser.enter_search_string = AsyncMock()
        p.browser.get_results_count_text = AsyncMock(return_value="120")
        p.browser.get_results_count = AsyncMock(return_value=120)

        p._ensure_browser_healthy = AsyncMock()
        p._review_page_sequentially = AsyncMock(return_value=None)
        p._assess_string_state = AsyncMock(
            return_value={"decision": "experiment", "rationale": "try sibling", "page": 1}
        )
        p._plan_variant_experiments = AsyncMock(return_value=[])

        lifecycle_calls = {"count": 0}

        def _evaluate_lifecycle(**kwargs):
            lifecycle_calls["count"] += 1
            experiment_state = kwargs["experiment_state"]
            experiment_state.active_variant.probe_pages_used = 1
            experiment_state.active_variant.saves = 2
            experiment_state.active_variant.facial_yes = 1
            experiment_state.active_variant.result_count = 120
            experiment_state.active_variant.target_result_min = 50
            experiment_state.active_variant.target_result_max = 300
            return "commit"

        p._evaluate_variant_lifecycle = MagicMock(side_effect=_evaluate_lifecycle)

        with (
            patch("linkedin.orchestrator.human_delay_correlated", return_value=0),
            patch("linkedin.orchestrator.config.MAX_PAGES_PER_STRING", 1),
        ):
            asyncio.run(p._process_string(search_string, progress))

        assert lifecycle_calls["count"] >= 1


# ---------------------------------------------------------------------------
# SPL live-run findings (2026-07-03): resume regime-swap + multi-location.
# ---------------------------------------------------------------------------


def test_session_location_filter_splits_semicolon_values():
    """A ';'-separated Location reaches the browser as SEPARATE facet values.
    Live-caught: one conjunction string exact-match missed as a single facet
    and the run proceeded boolean-only — every save came back off-geo."""
    import asyncio
    from unittest.mock import AsyncMock

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {
            "Location": "New York City Metropolitan Area; San Francisco Bay Area"
        }
        p._session_location_applied = False
        p.browser.apply_location_filter = AsyncMock(return_value=True)
        asyncio.run(p._apply_session_location_filter())
        args, kwargs = p.browser.apply_location_filter.call_args
        assert args[0] == [
            "New York City Metropolitan Area",
            "San Francisco Bay Area",
        ]


def test_resume_refuses_preflight_brief_without_generated_criteria():
    """Resume on a preflight-born seed brief with no generated V2 brief in the
    state dir must REFUSE (typed error), never silently judge on the hollow
    seed via legacy templates — the P9.2 regime-swap class at the resume door."""
    import asyncio

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.needs_preflight = lambda: True
        p.brief_obj.kit_url = ""
        Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1,
                    name="resume owner",
                    boolean="one",
                    status="in_progress",
                )
            ],
        ).save(str(p.progress_path))
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p._get_project_url = lambda: ""
        p._apply_session_location_filter = AsyncMock()
        p._print_summary = lambda: None
        p._generate_run_report = lambda progress: None

        with pytest.raises(RuntimeError, match="resume requires"):
            asyncio.run(p.run_full(resume=True))


# ---------------------------------------------------------------------------
# Wave 3 slice 14 — P1 discharge: the compellingness heuristic consumes
# brief vocabulary instead of carrying an AI/ML prior; off-geo save WARN
# telemetry (defense-in-depth; the fail-closed geography gate enforces).
# ---------------------------------------------------------------------------


def test_compelling_uses_brief_vocabulary_not_builtin_ai_prior():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        # Brief mirrors as REAL lists (a bare MagicMock attr contributes
        # nothing by design — mock-safe reader).
        p.brief_obj.canonical_title_patterns = ["claims platform"]
        p.brief_obj.canonical_framework_patterns = []
        p.brief_obj.canonical_broad_patterns = []
        p.brief_obj.key_terms_by_area = {"Claims": ["claims intake"]}

        # Brief vocabulary + leadership term → compelling.
        assert p._snippet_is_clearly_compelling(
            _make_snippet(headline="Director of Claims Platform Engineering")
        ) is True
        # Leadership without brief vocabulary → not compelling.
        assert p._snippet_is_clearly_compelling(
            _make_snippet(headline="Director of Marketing")
        ) is False


def test_compelling_has_no_builtin_ai_prior_without_brief_vocab():
    """The discharge itself: 'Head of AI' used to be compelling from CODE
    vocabulary; with no brief mirrors it no longer is."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.canonical_title_patterns = []
        p.brief_obj.canonical_framework_patterns = []
        p.brief_obj.canonical_broad_patterns = []
        p.brief_obj.key_terms_by_area = {}

        assert p._snippet_is_clearly_compelling(
            _make_snippet(headline="Head of AI")
        ) is False
        # Structural seniority markers still register, vocabulary-free.
        assert p._snippet_is_clearly_compelling(
            _make_snippet(headline="Distinguished Engineer, Platform")
        ) is True


def test_off_geo_save_warn_counts_and_respects_geography(capsys):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {"Location": "New York City Metropolitan Area"}

        # Off-geo: no shared significant token AND the containment check
        # confirms not-contained (mocked — an unmocked call would hit the
        # provider with its 5-retry backoff).
        with patch(
            "shared.llm_clients.cheap_llm", return_value={"contained": False}
        ) as mock_llm:
            p._warn_if_off_geo_save(_make_snippet(location="San Francisco Bay Area"))
        assert mock_llm.call_count == 1
        assert p.stats.get("off_geo_saves") == 1
        out = capsys.readouterr().out
        assert "[geo-warn]" in out
        assert "unverified" not in out

        # On-geo: shares a token ("york") — no increment, no model call.
        with patch("shared.llm_clients.cheap_llm") as mock_llm:
            p._warn_if_off_geo_save(_make_snippet(location="New York, United States"))
        assert mock_llm.call_count == 0
        assert p.stats.get("off_geo_saves") == 1

        # No geography on the brief → never warns.
        p.brief_obj.permanent_filters = {}
        p._warn_if_off_geo_save(_make_snippet(location="Anywhere Else Entirely"))
        assert p.stats.get("off_geo_saves") == 1


def test_off_geo_warn_suppressed_when_location_is_contained_in_facet(capsys):
    """The Mountain View case (2026-07-04 SPL run): a city inside an applied
    metro facet must not be reported as an off-geo save."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {"Location": "San Francisco Bay Area"}

        with patch(
            "shared.llm_clients.cheap_llm", return_value={"contained": True}
        ) as mock_llm:
            p._warn_if_off_geo_save(
                _make_snippet(location="Mountain View, California, United States")
            )

        assert mock_llm.call_count == 1
        assert p.stats.get("off_geo_saves", 0) == 0
        assert "[geo-warn]" not in capsys.readouterr().out


def test_off_geo_containment_caches_verdicts_and_failures(capsys):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {"Location": "San Francisco Bay Area"}

        # Verdict cached: second save from the same city makes no new call.
        with patch(
            "shared.llm_clients.cheap_llm", return_value={"contained": True}
        ) as mock_llm:
            p._warn_if_off_geo_save(_make_snippet(location="Mountain View, CA"))
            p._warn_if_off_geo_save(_make_snippet(location="Mountain View, CA"))
        assert mock_llm.call_count == 1
        assert p.stats.get("off_geo_saves", 0) == 0

        # Failure cached too: one retry cycle per (location, facets), then
        # the pre-check behavior (warn + count) with an honest marker.
        with patch(
            "shared.llm_clients.cheap_llm", side_effect=RuntimeError("provider down")
        ) as mock_llm:
            p._warn_if_off_geo_save(_make_snippet(location="Lisbon, Portugal"))
            p._warn_if_off_geo_save(_make_snippet(location="Lisbon, Portugal"))
        assert mock_llm.call_count == 1
        assert p.stats.get("off_geo_saves", 0) == 2
        assert "(containment unverified)" in capsys.readouterr().out


def test_off_geo_containment_nonconforming_response_is_unverified(capsys):
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {"Location": "San Francisco Bay Area"}

        with patch(
            "shared.llm_clients.cheap_llm", return_value=["not", "a", "dict"]
        ):
            p._warn_if_off_geo_save(_make_snippet(location="Lisbon, Portugal"))

        assert p.stats.get("off_geo_saves", 0) == 1
        assert "(containment unverified)" in capsys.readouterr().out


def test_snapshot_facial_calibration_carries_band_source():
    """P6 band-provenance reader (Wave 3 slice 14): the snapshot's facial-
    calibration block records where the band came from; pre-provenance
    briefs read 'unknown' (test-honesty lens: the stamp shipped with zero
    coverage — deleting it left the whole suite green)."""
    import types

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj._new_brief = types.SimpleNamespace(
            facial_calibration=types.SimpleNamespace(
                expected_yes_rate_low=0.2,
                expected_yes_rate_high=0.9,
                band_source="loader_default",
            )
        )
        p.stats.update({"facial_yes": 40, "facial_no": 60})
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1, name="t", boolean="x", status="done", pages_reviewed=1
                )
            ],
        )

        snapshot = p._build_run_report_snapshot(progress)
        assert snapshot["metrics_summary"]["facial_calibration"]["band_source"] == (
            "loader_default"
        )

    # A band authored before provenance stamping reads "unknown", never "".
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj._new_brief = types.SimpleNamespace(
            facial_calibration=types.SimpleNamespace(
                expected_yes_rate_low=0.2, expected_yes_rate_high=0.9
            )
        )
        p.stats.update({"facial_yes": 40, "facial_no": 60})
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1, name="t", boolean="x", status="done", pages_reviewed=1
                )
            ],
        )
        snapshot = p._build_run_report_snapshot(progress)
        assert snapshot["metrics_summary"]["facial_calibration"]["band_source"] == (
            "unknown"
        )


def test_coverage_gap_search_string_carries_lane_markers():
    """Codex review, Wave 3 (F2): the coverage-gap queue constructor is a
    third SearchString boundary — markers must survive it like the compound
    and adaptive constructors."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._execution_plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[],
            coverage_gaps=[
                {
                    "gap": "healthcare payers population",
                    "suggested_boolean": '("claims intake")',
                    "family_key": "h",
                    "novelty_bucket": "edge_case",
                    "domain_lane": "healthcare_payers",
                    "domain_lane_raw": "",
                    "undeclared_lane": True,
                }
            ],
        )

        strings = p._build_ordered_search_strings()

        gap_strings = [s for s in strings if s.block == "Coverage Gaps"]
        assert gap_strings, "coverage gap did not queue"
        assert gap_strings[0].undeclared_lane is True


def test_block_report_string_details_carry_lane_markers():
    """Codex review, Wave 3 (F4): the surfaces adaptation reads must see the
    keep-but-flag markers, not just the normalized lane."""
    import asyncio

    from shared.schemas import AdaptationResponse

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._execution_plan = ExecutionPlan(strategy_rationale="t")
        seen: dict = {}

        def spy_adapt_fn(brief, report, remaining, **kwargs):
            seen["details"] = list(report.string_details)
            seen["summary"] = report.to_summary_text()
            return AdaptationResponse(no_change=True)

        done = SearchString(
            id=1,
            name="s1",
            boolean='("claims")',
            status="done",
            result_count=10,
            candidates_count=5,
            pages_reviewed=1,
            domain_lane="payments",
            domain_lane_raw="payments_engineering",
            undeclared_lane=False,
        )
        queued = SearchString(id=2, name="s2", boolean='("y")', status="queued")
        progress = Progress(brief_name="test", strings=[done, queued])
        asyncio.run(
            p._run_block_adaptation("Block 1", [done], progress, spy_adapt_fn)
        )

    detail = seen["details"][0]
    assert detail["domain_lane_raw"] == "payments_engineering"
    assert detail["undeclared_lane"] is False
    assert "remapped from payments_engineering" in seen["summary"]


def test_run_report_string_performance_carries_lane_markers():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1,
                    name="s",
                    boolean='("x")',
                    status="done",
                    pages_reviewed=1,
                    domain_lane="healthcare_payers",
                    undeclared_lane=True,
                )
            ],
        )
        snapshot = p._build_run_report_snapshot(progress)

    entry = snapshot["string_performance"][0]
    assert entry["undeclared_lane"] is True
    assert entry["domain_lane_raw"] == ""


def test_bias_summary_renders_fired_signals_from_monitor_records():
    """The report's bias section renders the monitor's own persisted Alert
    payloads — one definition of the signal, not a parallel heuristic."""
    from linkedin.run_report import bias_summary_for_report
    from shared.bias_controls import BiasMonitor, DecisionRecord

    monitor = BiasMonitor(max_consecutive_saves=1)
    monitor.record_decision(DecisionRecord(
        candidate_id="c1",
        string_id="7",
        stage="full",
        decision="SAVE",
        confidence=0.8,
        capability_area=None,
    ))
    monitor.check_alerts("7")

    text = bias_summary_for_report(monitor)

    assert "Bias signals fired:" in text
    assert "[flag] consecutive_saves (string 7):" in text
    assert "High save density" in text


def test_process_string_rollback_failure_does_not_mask_original():
    """CLO-152: a rollback-restore failure on the unwind path must propagate
    the ORIGINAL string-processing error, not replace it with its own."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        search_string = SearchString(
            id=1,
            name="s",
            boolean="x",
            status="in_progress",
        )
        progress = Progress(brief_name="t", strings=[search_string])

        async def boom(*_args, **_kwargs):
            raise ValueError("original failure")

        p._process_string_impl = boom
        p._restore_incomplete_page_rollback = MagicMock(
            side_effect=RuntimeError("rollback broke")
        )

        with pytest.raises(ValueError, match="original failure"):
            asyncio.run(p._process_string(search_string, progress))

        p._restore_incomplete_page_rollback.assert_called_once_with(
            search_string
        )
