"""LinkedIn search-intelligence and mutation-executor regressions."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from linkedin.browser import SearchEntryResult
from linkedin.page_allocator import PageObservation
from linkedin.search_intelligence import (
    LinkedInExperimentState,
    LinkedInPageInsights,
    LinkedInSearchVariant,
    LinkedInStructuredFilters,
    LinkedInVariantSnapshot,
    bootstrap_experiment_state,
    reset_experiment_state,
    result_window_for_count,
)
from linkedin.search_mutation import (
    LinkedInSearchMutationDeps,
    LinkedInSearchMutationExecutor,
)
from linkedin.input_backends import TypingResult
from shared.schemas import Progress, SearchString
from shared.storage import read_jsonl


def _make_pipeline(output_dir: str):
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
        mock_brief.return_value = brief

        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline

        return Pipeline(brief_path=str(brief_path), output_dir=output_dir)


def _mutation_executor_for(pipeline) -> LinkedInSearchMutationExecutor:
    def set_budget(value: int) -> None:
        pipeline._search_mutation_budget_used = value

    return LinkedInSearchMutationExecutor(
        LinkedInSearchMutationDeps(
            browser=pipeline.browser,
            log_path=pipeline.log_path,
            get_input_mode=lambda: pipeline.input_mode,
            get_runtime_run_id=lambda: pipeline._runtime_run_id,
            get_runtime_state=lambda: pipeline._runtime_state,
            get_search_mutation_budget_used=lambda: pipeline._search_mutation_budget_used,
            set_search_mutation_budget_used=set_budget,
        )
    )


def test_result_window_for_count_matches_policy():
    assert result_window_for_count(7500) == (200, 1200)
    assert result_window_for_count(3200) == (150, 800)
    assert result_window_for_count(900) == (75, 400)
    assert result_window_for_count(200) is None


def test_bootstrap_experiment_state_preserves_legacy_refinement_shadow():
    search_string = SearchString(
        id=7,
        name="RL builders",
        boolean="bar",
        original_boolean="foo",
        refinement_stack=["foo"],
        phase="paginate",
    )

    state = bootstrap_experiment_state(search_string)

    assert state.intent.root_boolean == "foo"
    assert state.active_variant.boolean == "bar"
    assert state.committed_variant_id == state.active_variant_id
    assert search_string.boolean == "bar"
    assert search_string.refinement_stack == ["foo"]


def test_bootstrap_seeds_root_variant_and_intent_from_structured_filters():
    """Slice B part 1: a hybrid SearchString carrying compiled structured filters
    (locations + titles, set by slice A's producer path) seeds BOTH the intent
    (so reset_experiment_state inherits) and the root variant (so the opening
    structured-apply and cross-process resume drive off active.structured_filters).
    """
    search_string = SearchString(
        id=9,
        name="geo leaders",
        boolean='"VP" AND engineering',
        acquisition_mode="linkedin_hybrid",
        structured_filters={
            "titles": ["VP Engineering"],
            "sidebar_filters": {"locations": ["New York City"]},
        },
    )

    state = bootstrap_experiment_state(search_string)

    # Intent carries the filters -> reset_experiment_state inherits them (:1043).
    assert state.intent.structured_filters.titles == ["VP Engineering"]
    assert state.intent.structured_filters.sidebar_filters.get("locations") == [
        "New York City"
    ]
    # Root variant (the opening's active variant) carries them too.
    root = state.active_variant
    assert not root.structured_filters.is_empty()
    assert root.structured_filters.titles == ["VP Engineering"]
    assert root.structured_filters.sidebar_filters.get("locations") == ["New York City"]


def test_bootstrap_all_boolean_string_keeps_root_variant_filters_empty():
    """Slice B part 1 (byte-preserved default): an all-boolean SearchString (no
    structured_filters) leaves the root variant's filters empty and the intent's
    filters empty — the all-boolean lane is untouched."""
    search_string = SearchString(id=10, name="builders", boolean="foo")

    state = bootstrap_experiment_state(search_string)

    assert state.intent.structured_filters.is_empty()
    assert state.active_variant.structured_filters.is_empty()


def test_bootstrap_paginate_phase_without_chain_commits_root():
    """Paginate-implies-committed invariant (2026-07-06 SPL-MM live RCA): a
    chain-less rebuild in paginate mode must commit the root, or the
    committed-path zero-signal accounting never arms and the stop rule is
    unreachable — a 386-result string paginated 16 pages (13 consecutive
    zero-signal) with the streak frozen at 0."""
    search_string = SearchString(
        id=7, name="ops probe", boolean="foo", phase="paginate"
    )

    state = bootstrap_experiment_state(search_string)

    assert state.mode == "paginate"
    assert state.committed_variant_id == "root"
    assert state.active_variant_id == "root"


def test_bootstrap_scout_phase_stays_uncommitted_recon():
    search_string = SearchString(id=8, name="ops probe", boolean="foo", phase="scout")

    state = bootstrap_experiment_state(search_string)

    assert state.mode == "recon"
    assert state.committed_variant_id is None


def test_zero_signal_streak_arms_on_committed_root():
    """The stop rule's operands (committed_pages_reviewed, zero-signal
    streak) must accrue on a directly-committed root — the exact accounting
    that stayed frozen on the live run."""
    search_string = SearchString(id=9, name="ops probe", boolean="foo")
    state = bootstrap_experiment_state(search_string)
    state.mode = "paginate"
    state.commit_variant()

    zero_signal_stats = {
        "saves": 0,
        "facial_yes": 0,
        "rejects": 0,
        "full_reviewed": 0,
        "full_outreach": 0,
        "full_review": 0,
        "full_reject": 0,
        "candidates": 5,
    }
    for page in (1, 2):
        state.record_family_page_metrics(
            page_num=page,
            result_count=386,
            page_stats=zero_signal_stats,
            page_insights=LinkedInPageInsights(
                page=page, result_count=386, result_window="386"
            ),
        )

    assert state.committed_pages_reviewed == 2
    assert state.committed_zero_signal_streak == 2

    # Signal resets the streak — the brake is a decay rule, not a fuse.
    state.record_family_page_metrics(
        page_num=3,
        result_count=386,
        page_stats={
            "saves": 1,
            "facial_yes": 1,
            "rejects": 0,
            "full_reviewed": 1,
            "full_outreach": 1,
            "full_review": 0,
            "full_reject": 0,
            "candidates": 5,
        },
        page_insights=LinkedInPageInsights(page=3, result_count=386, result_window="386"),
    )
    assert state.committed_zero_signal_streak == 0


def test_process_string_direct_paginate_commits_root_variant_for_zero_signal_brake(tmp_path):
    """Orchestrator direct-paginate entry must commit root, not just bootstrap."""
    pipeline = _make_pipeline(str(tmp_path))
    search_string = SearchString(id=11, name="sub-500 pool", boolean="foo", status="queued")
    progress = Progress(
        brief_name="test",
        strings=[search_string],
        current_string_id=11,
        current_page=0,
    )

    no_results = MagicMock()
    no_results.is_visible = AsyncMock(return_value=False)
    locator = MagicMock()
    locator.first = no_results

    pipeline.browser.page = MagicMock(url="https://www.linkedin.com/talent/search")
    pipeline.browser.page.locator.return_value = locator
    pipeline.browser.get_results_count_text = AsyncMock(return_value="386")
    pipeline.browser.get_results_count = AsyncMock(return_value=386)
    pipeline._apply_opening_search = AsyncMock()
    pipeline._ensure_browser_healthy = AsyncMock()
    pipeline._review_page_sequentially = AsyncMock(return_value=None)
    pipeline._checkpoint_progress = MagicMock()

    with patch("linkedin.orchestrator.config.MAX_PAGES_PER_STRING", 1):
        asyncio.run(pipeline._process_string(search_string, progress))

    state = pipeline._experiment_states[search_string.id]
    is_committed_variant_page = (
        state.committed_variant_id is not None
        and state.active_variant_id == state.committed_variant_id
    )
    assert state.mode == "paginate"
    assert state.committed_variant_id == "root"
    assert is_committed_variant_page is True
    assert state.committed_zero_signal_streak == 1


def test_search_mutation_executor_rejects_non_empty_experimental_filters(tmp_path):
    pipeline = _make_pipeline(str(tmp_path))
    search_string = SearchString(id=3, name="builders", boolean="foo")
    state = bootstrap_experiment_state(search_string)
    variant = LinkedInSearchVariant(
        variant_id="precision-1",
        parent_variant_id="root",
        root_string_id=3,
        boolean="foo AND bar",
        variant_kind="precision",
        structured_filters=LinkedInStructuredFilters(titles=["Staff Engineer"]),
    )

    result = asyncio.run(
        _mutation_executor_for(pipeline).apply_variant(
            search_string=search_string,
            experiment_state=state,
            variant=variant,
        )
    )

    assert result.applied is False
    assert result.blocked_reason == "experimental_structured_filters_not_supported"


def test_search_mutation_executor_applies_keyword_variant(tmp_path):
    pipeline = _make_pipeline(str(tmp_path))
    pipeline.browser.go_back_to_results = AsyncMock()
    pipeline.browser.enter_search_string = AsyncMock(
        return_value=SearchEntryResult(
            typing_result=TypingResult(
                transport="playwright_keyboard",
                duration_ms=2100,
                typo_count=1,
                used_correction=True,
                fallback_char_count=0,
            ),
            results_wait_ms=1350,
        )
    )
    pipeline.browser.get_results_count_text = AsyncMock(return_value="220")
    pipeline.browser.get_results_count = AsyncMock(return_value=220)
    pipeline.browser.get_card_snapshot = AsyncMock(return_value={"name": "Ada", "url": "/talent/profile/ada"})

    search_string = SearchString(id=3, name="builders", boolean="foo")
    state = bootstrap_experiment_state(search_string)
    variant = LinkedInSearchVariant(
        variant_id="precision-1",
        parent_variant_id="root",
        root_string_id=3,
        boolean="foo AND bar",
        variant_kind="precision",
        target_result_min=75,
        target_result_max=400,
    )
    state.begin_experiment_round([variant])

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
        result = asyncio.run(
            _mutation_executor_for(pipeline).apply_variant(
                search_string=search_string,
                experiment_state=state,
                variant=variant,
            )
        )

    assert result.applied is True
    assert result.result_count == 220
    assert state.active_variant_id == "precision-1"
    assert pipeline._search_mutation_budget_used == 1
    pipeline.browser.enter_search_string.assert_awaited_once_with("foo AND bar")
    events = read_jsonl(pipeline.log_path)
    applied_event = next(event for event in reversed(events) if event["event"] == "linkedin_search_mutation_applied")
    assert applied_event["input_mode"] == "concurrent"
    assert applied_event["typing_transport"] == "playwright_keyboard"
    assert applied_event["typing_duration_ms"] == 2100
    assert applied_event["typo_count"] == 1
    assert applied_event["used_correction"] is True
    assert applied_event["fallback_char_count"] == 0
    assert applied_event["results_wait_ms"] == 1350


def test_experiment_state_tracks_family_totals_and_drift_snapshots():
    search_string = SearchString(id=11, name="builders", boolean="foo")
    state = bootstrap_experiment_state(search_string)
    state.commit_variant("root")

    page1 = LinkedInPageInsights(
        page=1,
        result_count=1800,
        result_window="150-800",
        title_clusters=[{"label": "machine learning engineer", "count": 4}],
        company_clusters=[{"label": "OpenAI", "count": 2}],
        signal_anchors=["ML engineer at OpenAI", "Research engineer at Anthropic"],
    )
    positive_stats = {
        "candidates": 4,
        "facial_yes": 2,
        "saves": 1,
        "full_reviewed": 2,
        "full_outreach": 1,
        "full_review": 1,
        "full_reject": 0,
    }
    state.record_variant_metrics(page_num=1, result_count=1800, page_stats=positive_stats, page_insights=page1)
    state.record_family_page_metrics(page_num=1, result_count=1800, page_stats=positive_stats, page_insights=page1)

    page3 = LinkedInPageInsights(
        page=3,
        result_count=1800,
        result_window="150-800",
        title_clusters=[{"label": "product manager", "count": 5}],
        company_clusters=[{"label": "BigCo", "count": 3}],
        noise_anchors=["Product manager at BigCo", "Program manager at BankCorp"],
        dominant_non_fit_patterns=["product-heavy profiles dominate recent pages"],
        glance_action="reformulate",
    )
    state.record_variant_metrics(page_num=3, result_count=1800, page_stats={"candidates": 5, "facial_no": 4}, page_insights=page3)
    state.record_family_page_metrics(page_num=3, result_count=1800, page_stats={"candidates": 5, "facial_no": 4}, page_insights=page3)

    summary = state.metrics_summary()

    assert summary["family_pages_reviewed_total"] == 2
    assert summary["family_candidates_total"] == 9
    assert summary["family_signal_total"] == 2
    assert summary["family_saves_total"] == 1
    assert summary["family_outreach_total"] == 1
    assert summary["family_review_total"] == 1
    assert summary["active_variant_pages_reviewed"] == 3
    assert state.early_signal_snapshot is not None
    assert state.recent_noise_snapshot is not None
    assert state.early_signal_snapshot.signal_anchors[0] == "ML engineer at OpenAI"
    assert state.recent_noise_snapshot.noise_anchors[0] == "Product manager at BigCo"


def test_raw_facial_positives_and_full_rejects_are_not_search_signal():
    state = bootstrap_experiment_state(SearchString(id=23, name="builders", boolean="foo"))
    state.commit_variant("root")
    page = LinkedInPageInsights(
        page=1,
        result_count=150,
        result_window="75-400",
        signal_anchors=["Facially plausible candidate", "Another plausible candidate"],
    )
    stats = {
        "candidates": 8,
        "facial_yes": 8,
        "facial_no": 0,
        "saves": 0,
        "rejects": 8,
        "full_reviewed": 8,
        "full_outreach": 0,
        "full_review": 0,
        "full_reject": 8,
    }

    state.record_variant_metrics(page_num=1, result_count=150, page_stats=stats, page_insights=page)
    state.record_family_page_metrics(page_num=1, result_count=150, page_stats=stats, page_insights=page)

    variant = state.active_variant
    assert variant.full_reviewed == 8
    assert variant.full_reject == 8
    assert variant.score() < 0
    assert state.family_signal_total == 0
    assert state.family_reject_total == 8
    assert state.committed_zero_signal_streak == 1
    assert state.early_signal_snapshot is None
    assert state.real_signal_seen() is False


def test_variant_deserialization_maps_legacy_settled_outcome_proxies():
    variant = LinkedInSearchVariant.from_dict(
        {
            "variant_id": "legacy",
            "root_string_id": 2,
            "boolean": "foo",
            "saves": 2,
            "rejects": 3,
            "facial_yes": 9,
        }
    )

    assert variant.full_outreach == 2
    assert variant.full_reject == 3
    assert variant.full_review == 0
    assert variant.full_reviewed == 5
    assert variant.settled_positive_count == 2


def test_experiment_state_round_trip_preserves_full_outcome_totals():
    state = bootstrap_experiment_state(SearchString(id=24, name="builders", boolean="foo"))
    state.record_variant_metrics(
        page_num=1,
        result_count=150,
        page_stats={
            "full_reviewed": 4,
            "full_outreach": 1,
            "full_review": 1,
            "full_reject": 2,
            "facial_yes": 2,
            "facial_borderline": 2,
        },
        page_insights=None,
    )
    state.record_family_page_metrics(
        page_num=1,
        result_count=150,
        page_stats={
            "full_reviewed": 4,
            "full_outreach": 1,
            "full_review": 1,
            "full_reject": 2,
        },
        page_insights=LinkedInPageInsights(page=1, result_count=150, result_window="75-400"),
    )

    restored = LinkedInExperimentState.from_dict(state.to_dict())

    assert restored is not None
    assert restored.family_outreach_total == 1
    assert restored.family_review_total == 1
    assert restored.family_reject_total == 2
    assert restored.family_reviewed_total == 4
    assert restored.active_variant.full_outreach == 1
    assert restored.active_variant.facial_yes == 2
    assert restored.active_variant.facial_borderline == 2


def test_experiment_state_legacy_signal_total_is_rederived_from_saves():
    restored = LinkedInExperimentState.from_dict(
        {
            "root_string_id": 25,
            "intent": {"root_boolean": "foo"},
            "family_signal_total": 15,
            "family_saves_total": 2,
            "variants": {
                "root": {
                    "variant_id": "root",
                    "root_string_id": 25,
                    "boolean": "foo",
                    "saves": 2,
                    "rejects": 3,
                    "facial_yes": 10,
                }
            },
        }
    )

    assert restored is not None
    assert restored.family_outreach_total == 2
    assert restored.family_review_total == 0
    assert restored.family_signal_total == 2
    assert restored.active_variant.full_outreach == 2
    assert restored.active_variant.full_reject == 3


def test_experiment_state_round_trips_variant_local_page_cursors():
    state = bootstrap_experiment_state(
        SearchString(id=25, name="builders", boolean="foo")
    )
    state.commit_variant("root")
    state.set_active_allocator_page_cursor(4)
    drift = LinkedInSearchVariant(
        variant_id="drift-25-1",
        parent_variant_id="root",
        root_string_id=25,
        boolean="foo NOT product",
        variant_kind="precision",
    )
    state.variants[drift.variant_id] = drift
    state.mark_pending_drift(
        variant_id=drift.variant_id,
        parent_variant_id="root",
    )
    state.activate_variant(drift.variant_id)
    state.set_active_allocator_page_cursor(2)

    restored = LinkedInExperimentState.from_dict(state.to_dict())

    assert restored is not None
    assert restored.active_variant_id == drift.variant_id
    assert restored.active_allocator_page_cursor() == 2
    assert restored.variants["root"].allocator_page_cursor == 4
    assert restored.variants[drift.variant_id].allocator_page_cursor == 2

    restored.resume_committed_after_failed_drift()

    assert restored.active_variant_id == "root"
    assert restored.active_allocator_page_cursor() == 4


def _allocator_observation(
    page: int,
    *,
    variant_id: str = "root",
    technical_interruption: bool = False,
    off_policy: bool = False,
) -> PageObservation:
    return PageObservation(
        root_string_id=25,
        variant_id=variant_id,
        page=page,
        slots=5,
        extracted=5,
        full_expected=1,
        full_settled=1,
        priority=1,
        standard=0,
        outreach=1,
        technical_interruption=technical_interruption,
        off_policy=off_policy,
    )


def test_allocator_state_counts_completed_pages_but_only_retains_teaching_pages():
    state = bootstrap_experiment_state(
        SearchString(id=25, name="builders", boolean="foo")
    )
    variant = state.active_variant
    variant.pages_reviewed = 5

    state.record_allocator_observation(_allocator_observation(1))
    state.record_allocator_observation(
        _allocator_observation(2, technical_interruption=True)
    )
    state.record_allocator_observation(_allocator_observation(3, off_policy=True))

    assert variant.allocator_completed_observation_count == 3
    assert variant.allocator_valid_page_count == 1
    assert [item.page for item in variant.allocator_observations] == [1]
    assert state.allocator_last_observation == _allocator_observation(
        3, off_policy=True
    )
    assert state.legacy_unobserved_pages() == 2


def test_allocator_teaching_window_is_bounded_to_last_two_pages():
    state = bootstrap_experiment_state(
        SearchString(id=25, name="builders", boolean="foo")
    )

    for page in (1, 2, 3):
        state.record_allocator_observation(_allocator_observation(page))

    assert state.active_variant.allocator_valid_page_count == 3
    assert state.active_variant.allocator_completed_observation_count == 3
    assert [
        item.page for item in state.active_variant.allocator_observations
    ] == [2, 3]


def test_allocator_rewrite_starts_fresh_window_without_resetting_root_probe():
    state = bootstrap_experiment_state(
        SearchString(id=25, name="builders", boolean="foo")
    )
    state.record_allocator_observation(_allocator_observation(1))
    rewrite = LinkedInSearchVariant(
        variant_id="rewrite-1",
        parent_variant_id="root",
        root_string_id=25,
        boolean="foo AND bar",
        pages_reviewed=2,
    )
    state.variants[rewrite.variant_id] = rewrite
    state.active_variant_id = rewrite.variant_id

    assert state.root_has_valid_probe() is True
    assert state.active_variant.allocator_valid_page_count == 0
    assert state.active_variant.allocator_observations == []
    assert state.legacy_unobserved_pages() == 2

    # A page can settle after local lifecycle moved the active pointer; exact
    # variant identity, not the mutable active pointer, owns the observation.
    state.active_variant_id = "root"
    state.record_allocator_observation(
        _allocator_observation(1, variant_id="rewrite-1")
    )
    assert rewrite.allocator_completed_observation_count == 1
    assert rewrite.allocator_valid_page_count == 1


def test_allocator_shadow_deserialization_defaults_old_and_malformed_fields():
    old = LinkedInExperimentState.from_dict(
        {
            "root_string_id": 25,
            "intent": {"root_boolean": "foo"},
            "variants": {
                "root": {
                    "variant_id": "root",
                    "root_string_id": 25,
                    "boolean": "foo",
                    "pages_reviewed": 4,
                }
            },
        }
    )
    assert old is not None
    assert old.active_variant.allocator_valid_page_count == 0
    assert old.active_variant.allocator_completed_observation_count == 0
    assert old.active_variant.allocator_observations == []
    assert old.legacy_unobserved_pages() == 4
    assert old.root_has_valid_probe() is False

    payload = old.to_dict()
    root_payload = payload["variants"]["root"]
    root_payload["allocator_valid_page_count"] = "not-an-int"
    root_payload["allocator_completed_observation_count"] = {"bad": "shape"}
    root_payload["allocator_observations"] = [None, "bad", {"page": "bad"}]
    payload["allocator_last_observation"] = {"page": "bad"}
    payload["allocator_last_verdict"] = ["bad"]
    payload["allocator_shadow_diverged"] = "yes"
    payload["allocator_causality"] = ["bad"]
    payload["allocator_frontier_expectation"] = "bad"

    restored = LinkedInExperimentState.from_dict(payload)

    assert restored is not None
    assert restored.active_variant.allocator_valid_page_count == 0
    assert restored.active_variant.allocator_completed_observation_count == 0
    assert restored.active_variant.allocator_observations == []
    assert restored.allocator_last_observation is None
    assert restored.allocator_last_verdict == {}
    assert restored.allocator_shadow_diverged is False
    assert restored.allocator_causality == {}
    assert restored.allocator_frontier_expectation == {}


def test_explicit_experiment_reset_discards_allocator_shadow_history():
    search_string = SearchString(id=25, name="builders", boolean="foo")
    state = bootstrap_experiment_state(search_string)
    state.record_allocator_observation(_allocator_observation(1))
    state.allocator_last_verdict = {"action": "switch"}
    state.allocator_shadow_diverged = True
    state.allocator_causality = {"spend_sequence": 1}
    state.allocator_frontier_expectation = {"root_ids": [25, 26]}

    reset = reset_experiment_state(search_string, state)

    assert reset.active_variant.allocator_valid_page_count == 0
    assert reset.active_variant.allocator_completed_observation_count == 0
    assert reset.active_variant.allocator_observations == []
    assert reset.allocator_last_observation is None
    assert reset.allocator_last_verdict == {}
    assert reset.allocator_shadow_diverged is False
    assert reset.allocator_causality == {}
    assert reset.allocator_frontier_expectation == {}


def test_snapshot_weights_only_settled_full_profile_outcomes_as_signal():
    snapshot = LinkedInVariantSnapshot.from_page(
        page_num=1,
        result_count=100,
        page_insights=LinkedInPageInsights(page=1, result_count=100, result_window="75-400"),
        page_stats={
            "facial_yes": 20,
            "facial_no": 2,
            "full_outreach": 1,
            "full_review": 1,
            "full_reject": 4,
        },
    )

    assert snapshot.signal_weight == 4.0
    assert snapshot.noise_weight == 6.0


def test_search_mutation_executor_blocks_second_drift_attempt(tmp_path):
    pipeline = _make_pipeline(str(tmp_path))
    search_string = SearchString(id=3, name="builders", boolean="foo")
    state = bootstrap_experiment_state(search_string)
    state.commit_variant("root")
    state.drift_attempt_count = 1

    variant = LinkedInSearchVariant(
        variant_id="drift-1",
        parent_variant_id="root",
        root_string_id=3,
        boolean="foo NOT bar",
        variant_kind="recall",
    )

    result = asyncio.run(
        _mutation_executor_for(pipeline).apply_variant(
            search_string=search_string,
            experiment_state=state,
            variant=variant,
            mutation_kind="drift",
        )
    )

    assert result.applied is False
    assert result.blocked_reason == "drift_attempt_limit"


def _assess_recon(
    *,
    result_count: int,
    page_stats: dict[str, int],
    page_insights: LinkedInPageInsights,
    precommit_recovery_attempts_used: int = 0,
):
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        search_string = SearchString(id=17, name="builders", boolean="foo")
        state = bootstrap_experiment_state(search_string)
        state.mode = "recon"
        state.precommit_recovery_attempts_used = precommit_recovery_attempts_used

        assessment = asyncio.run(
            pipeline._assess_string_state(
                search_string=search_string,
                experiment_state=state,
                page_num=1,
                result_count=result_count,
                string_stats=dict(page_stats),
                page_stats=page_stats,
                page_insights=page_insights,
                remaining_queued_strings=4,
            )
        )
        events = read_jsonl(pipeline.log_path)
        return assessment, events[-1]


def test_assess_recon_experiments_on_large_noisy_mixed_signal():
    assessment, event = _assess_recon(
        result_count=6000,
        page_stats={
            "facial_yes": 1,
            "facial_no": 10,
            "saves": 0,
            "rejects": 0,
            "full_reviewed": 1,
            "full_outreach": 0,
            "full_review": 1,
            "full_reject": 0,
        },
        page_insights=LinkedInPageInsights(
            page=1,
            result_count=6000,
            result_window="200-1200",
            signal_anchors=["Applied scientist at frontier lab"],
            noise_anchors=["Product manager", "Program manager", "Eng manager"],
            dominant_non_fit_patterns=["manager-heavy page dominates"],
            glance_action="reformulate",
        ),
    )

    assert assessment["decision"] == "experiment"
    assert assessment["scout_gate_bucket"] == "precommit_weak_signal_recovery"
    assert assessment["noise_dominant"] is True
    assert assessment["strong_scout_signal"] is False
    assert event["event"] == "linkedin_search_assess"
    assert event["decision"] == "experiment"
    assert event["page_signal"] == 1
    assert event["facial_no"] == 10
    assert event["noise_dominant"] is True
    assert event["scout_gate_bucket"] == "precommit_weak_signal_recovery"


def test_assess_recon_experiments_on_large_real_signal_noisy_pool():
    assessment, event = _assess_recon(
        result_count=6000,
        page_stats={
            "facial_yes": 2,
            "facial_no": 10,
            "saves": 0,
            "rejects": 0,
            "full_reviewed": 2,
            "full_outreach": 1,
            "full_review": 0,
            "full_reject": 1,
        },
        page_insights=LinkedInPageInsights(
            page=1,
            result_count=6000,
            result_window="200-1200",
            signal_anchors=["Applied scientist at frontier lab", "Research engineer at market infra firm"],
            noise_anchors=["Product manager", "Program manager", "Eng manager"],
            dominant_non_fit_patterns=["manager-heavy page dominates"],
            glance_action="reformulate",
        ),
    )

    assert assessment["decision"] == "experiment"
    assert assessment["real_signal"] is True
    assert assessment["noise_dominant"] is True
    assert assessment["scout_gate_bucket"] == "precommit_real_signal_noisy_recovery"
    assert event["event"] == "linkedin_search_assess"
    assert event["decision"] == "experiment"
    assert event["real_signal"] is True
    assert event["strong_scout_signal"] is True
    assert event["scout_gate_bucket"] == "precommit_real_signal_noisy_recovery"


def test_assess_recon_commits_large_clean_strong_signal():
    assessment, _ = _assess_recon(
        result_count=6000,
        page_stats={"saves": 1, "facial_yes": 1, "facial_no": 2, "rejects": 0},
        page_insights=LinkedInPageInsights(
            page=1,
            result_count=6000,
            result_window="200-1200",
            signal_anchors=["Staff ML engineer at OpenAI", "Principal engineer at Anthropic"],
            noise_anchors=["Software engineer at generic SaaS"],
        ),
    )

    assert assessment["decision"] == "commit"
    assert assessment["scout_gate_bucket"] == "root_real_signal_commit"
    assert assessment["strong_scout_signal"] is True
    assert assessment["noise_dominant"] is False


def test_assess_recon_commits_large_noisy_real_signal_when_budget_is_exhausted():
    assessment, _ = _assess_recon(
        result_count=6000,
        page_stats={
            "facial_yes": 2,
            "facial_no": 10,
            "saves": 0,
            "rejects": 0,
            "full_reviewed": 2,
            "full_outreach": 1,
            "full_review": 0,
            "full_reject": 1,
        },
        page_insights=LinkedInPageInsights(
            page=1,
            result_count=6000,
            result_window="200-1200",
            signal_anchors=["Applied scientist at frontier lab", "Research engineer at market infra firm"],
            noise_anchors=["Product manager", "Program manager", "Eng manager"],
            dominant_non_fit_patterns=["manager-heavy page dominates"],
            glance_action="reformulate",
        ),
        precommit_recovery_attempts_used=2,
    )

    assert assessment["decision"] == "commit"
    assert assessment["real_signal"] is True
    assert assessment["noise_dominant"] is True
    assert assessment["scout_gate_bucket"] == "precommit_real_signal_budget_exhausted_commit"


def test_assess_recon_experiments_large_dead_noisy_pool():
    assessment, _ = _assess_recon(
        result_count=6000,
        page_stats={"facial_no": 10, "saves": 0, "facial_yes": 0, "rejects": 0},
        page_insights=LinkedInPageInsights(
            page=1,
            result_count=6000,
            result_window="200-1200",
            noise_anchors=["Product manager", "Program manager", "Operations lead"],
            dominant_non_fit_patterns=["non-technical leadership dominates"],
            glance_action="reformulate",
        ),
    )

    assert assessment["decision"] == "experiment"
    assert assessment["scout_gate_bucket"] == "precommit_dead_noisy_recovery"
    assert assessment["page_signal"] == 0


def test_assess_recon_commits_mid_sized_weak_signal_pool():
    assessment, _ = _assess_recon(
        result_count=3200,
        page_stats={
            "facial_yes": 1,
            "facial_no": 10,
            "saves": 0,
            "rejects": 0,
            "full_reviewed": 1,
            "full_outreach": 0,
            "full_review": 1,
            "full_reject": 0,
        },
        page_insights=LinkedInPageInsights(
            page=1,
            result_count=3200,
            result_window="150-800",
            signal_anchors=["Applied scientist"],
            noise_anchors=["Product manager", "Program manager", "Director"],
            dominant_non_fit_patterns=["manager-heavy page dominates"],
            glance_action="reformulate",
        ),
    )

    assert assessment["decision"] == "commit"
    assert assessment["real_signal"] is False
    assert assessment["scout_gate_bucket"] == "mid_pool_signal_commit"
    assert assessment["page_signal"] == 1


def test_assess_recon_stops_small_dead_noisy_pool_directly():
    assessment, _ = _assess_recon(
        result_count=400,
        page_stats={"facial_no": 8, "saves": 0, "facial_yes": 0, "rejects": 0},
        page_insights=LinkedInPageInsights(
            page=1,
            result_count=400,
            result_window="direct_paginate",
            noise_anchors=["Product manager", "Program manager"],
            dominant_non_fit_patterns=["manager-heavy page dominates"],
            glance_action="reformulate",
        ),
    )

    assert assessment["decision"] == "stop"
    assert assessment["scout_gate_bucket"] == "small_pool_dead_stop"


def test_assess_recon_stops_large_weak_signal_after_recovery_budget_is_exhausted():
    assessment, _ = _assess_recon(
        result_count=6000,
        page_stats={
            "facial_yes": 1,
            "facial_no": 10,
            "saves": 0,
            "rejects": 0,
            "full_reviewed": 1,
            "full_outreach": 0,
            "full_review": 1,
            "full_reject": 0,
        },
        page_insights=LinkedInPageInsights(
            page=1,
            result_count=6000,
            result_window="200-1200",
            signal_anchors=["Applied scientist at frontier lab"],
            noise_anchors=["Product manager", "Program manager", "Eng manager"],
            dominant_non_fit_patterns=["manager-heavy page dominates"],
            glance_action="reformulate",
        ),
        precommit_recovery_attempts_used=2,
    )

    assert assessment["decision"] == "stop"
    assert assessment["real_signal"] is False
    assert assessment["scout_gate_bucket"] == "precommit_recovery_exhausted_stop"


def test_assess_recon_stops_after_recovery_budget_is_exhausted():
    assessment, _ = _assess_recon(
        result_count=6000,
        page_stats={"facial_no": 10, "saves": 0, "facial_yes": 0, "rejects": 0},
        page_insights=LinkedInPageInsights(
            page=1,
            result_count=6000,
            result_window="200-1200",
            noise_anchors=["Product manager", "Program manager", "Operations lead"],
            dominant_non_fit_patterns=["non-technical leadership dominates"],
            glance_action="reformulate",
        ),
        precommit_recovery_attempts_used=2,
    )

    assert assessment["decision"] == "stop"
    assert assessment["scout_gate_bucket"] == "precommit_recovery_exhausted_stop"


def test_variant_commit_requires_settled_full_profile_signal():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        facial_only = LinkedInSearchVariant(
            variant_id="precision-1",
            parent_variant_id="root",
            root_string_id=1,
            boolean="foo",
            variant_kind="precision",
            target_result_min=75,
            target_result_max=400,
            result_count=220,
            facial_yes=1,
            facial_no=0,
        )
        outreach = LinkedInSearchVariant(
            variant_id="precision-2",
            parent_variant_id="root",
            root_string_id=1,
            boolean="foo",
            variant_kind="precision",
            target_result_min=75,
            target_result_max=400,
            result_count=220,
            full_reviewed=1,
            full_outreach=1,
        )

        assert pipeline._variant_has_earned_commit(facial_only) is False
        assert pipeline._variant_has_earned_commit(outreach) is True


# ---------------------------------------------------------------------------
# SLICE G — surface-aware durable state across a worker-death / cross-process
# resume that reconstructs via bootstrap_experiment_state (NOT the in-memory
# from_dict). The active variant surface + its CURRENT structured_filters must
# survive on the COMPAT SearchString so the bootstrap path agrees with the
# in-memory checkpoint path.
# ---------------------------------------------------------------------------


def test_searchstring_surface_round_trips_and_defaults_empty_on_legacy_payload():
    """Part 1: SearchString gains a durable surface field that round-trips
    through to_dict/from_dict and defaults "" on a legacy payload that predates
    slice G (a checkpoint written before this slice still loads)."""
    s = SearchString(id=1, name="builders", boolean="foo", surface="structured_only")
    payload = s.to_dict()
    assert payload["surface"] == "structured_only"
    assert SearchString.from_dict(payload).surface == "structured_only"

    # Legacy payload (no surface key) -> default "".
    legacy = {"id": 2, "name": "legacy", "boolean": "bar"}
    assert SearchString.from_dict(legacy).surface == ""
    # Default on a fresh dataclass is "" too.
    assert SearchString(id=3, name="x", boolean="y").surface == ""


def test_apply_shadow_persists_active_surface_and_current_filters_on_searchstring():
    """Parts 1+2: apply_shadow writes the active variant's surface AND its CURRENT
    structured_filters onto the SearchString unconditionally, so the compat record
    carries the post-promote/demote state (not the producer-time state)."""
    search_string = SearchString(
        id=4,
        name="geo leaders",
        boolean='"VP" AND engineering',
        acquisition_mode="linkedin_hybrid",
    )
    state = bootstrap_experiment_state(search_string)
    active = state.active_variant
    active.surface = "hybrid"
    active.structured_filters = LinkedInStructuredFilters(
        titles=["VP Engineering"],
        sidebar_filters={"locations": ["New York City"]},
    )

    state.apply_shadow(search_string)

    assert search_string.surface == "hybrid"
    assert search_string.structured_filters["titles"] == ["VP Engineering"]
    assert search_string.structured_filters["sidebar_filters"]["locations"] == [
        "New York City"
    ]


def test_structured_only_variant_survives_bootstrap_resume_without_keyword_reentry():
    """(a) A structured_only variant survives a worker-death -> bootstrap-reconstruct
    resume with surface=="structured_only" intact (NOT ""), and the opening apply does
    NOT re-add the keyword the surface is defined to suppress.

    Mid-run, a structured_only variant carries the filters with NO keyword. apply_shadow
    persists surface + filters onto the compat SearchString; the worker dies; a resume
    that has NO in-memory state reconstructs via bootstrap_experiment_state. With surface
    durable + slice D, the reconstructed variant's surface gates include_keyword off, so
    compile_structured_filters_to_plan emits an empty keyword_boolean — no keyword re-entry.
    """
    producer = SearchString(
        id=5,
        name="structured only lane",
        boolean='"Staff Engineer"',
        acquisition_mode="linkedin_hybrid",
    )
    state = bootstrap_experiment_state(producer)
    active = state.active_variant
    active.surface = "structured_only"
    active.structured_filters = LinkedInStructuredFilters(titles=["Staff Engineer"])
    state.apply_shadow(producer)

    # Cross-process resume: serialize the compat SearchString, drop the in-memory
    # state entirely, reconstruct purely from the SearchString.
    resumed_string = SearchString.from_dict(producer.to_dict())
    resumed = bootstrap_experiment_state(resumed_string)

    # surface durable through the bootstrap path (NOT degraded to "").
    assert resumed.active_variant.surface == "structured_only"
    assert resumed.active_variant.structured_filters.titles == ["Staff Engineer"]

    # The opening apply must NOT re-add the keyword. Drive _apply_opening_search and
    # capture the compiled plan handed to the browser.
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        captured: dict[str, Any] = {}

        async def _capture_plan(plan):
            captured["plan"] = plan

        pipeline.browser.apply_advanced_search_plan = AsyncMock(side_effect=_capture_plan)
        pipeline.browser.enter_search_string = AsyncMock()

        asyncio.run(
            pipeline._apply_opening_search(
                resumed_string, resumed, resumed.current_boolean()
            )
        )

        plan = captured["plan"]
        assert plan.keyword_boolean == ""
        assert not any(c.dimension == "keywords" for c in plan.controls)
        # The structured plan was applied (not the bare keyword path).
        pipeline.browser.enter_search_string.assert_not_called()


def test_mid_run_promote_survives_bootstrap_resume_reapplies_structured():
    """(b) A mid-run PROMOTE (a boolean-lane variant promoted to a title filter)
    survives a bootstrap-reconstruct resume: the compat SearchString carries the
    PROMOTED titles, bootstrap reconstructs them onto the active variant, and the
    resumed opening re-applies the structured search (NOT keyword-only).

    Before slice G the producer-time SearchString.structured_filters were empty (the
    lane started boolean), so the promote was LOST on the bootstrap path. apply_shadow
    now writes the CURRENT filters, so the promoted titles persist.
    """
    producer = SearchString(
        id=6,
        name="boolean lane that promotes",
        boolean='"director" AND payments',
        acquisition_mode="linkedin_hybrid",
        structured_filters={},  # started a bare boolean lane — no producer-time filters
    )
    state = bootstrap_experiment_state(producer)
    # The variant started boolean (empty filters). A mid-run promote lifts a title.
    active = state.active_variant
    active.surface = "hybrid"
    active.variant_kind = "structured_filter"
    active.structured_filters = LinkedInStructuredFilters(titles=["Director of Payments"])
    state.apply_shadow(producer)

    # PROMOTE survived onto the compat SearchString.
    assert producer.structured_filters["titles"] == ["Director of Payments"]

    resumed_string = SearchString.from_dict(producer.to_dict())
    resumed = bootstrap_experiment_state(resumed_string)

    assert resumed.active_variant.structured_filters.titles == ["Director of Payments"]
    assert resumed.active_variant.surface == "hybrid"

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        captured: dict[str, Any] = {}

        async def _capture_plan(plan):
            captured["plan"] = plan

        pipeline.browser.apply_advanced_search_plan = AsyncMock(side_effect=_capture_plan)
        pipeline.browser.enter_search_string = AsyncMock()

        asyncio.run(
            pipeline._apply_opening_search(
                resumed_string, resumed, resumed.current_boolean()
            )
        )

        plan = captured["plan"]
        # hybrid keeps the keyword AND carries the promoted title control.
        assert plan.keyword_boolean == '"director" AND payments'
        assert any(
            c.dimension == "job_titles" and c.values == ["Director of Payments"]
            for c in plan.controls
        )
        pipeline.browser.enter_search_string.assert_not_called()


def test_deliberate_boolean_demote_survives_resume_without_reseeding_filters():
    """(c) A deliberate boolean-demote survives a bootstrap-reconstruct resume: the
    demoted (empty) filters are NOT re-seeded onto the resumed variant, and the
    surface=="boolean" demote marker is preserved.

    apply_shadow now writes the CURRENT (empty) filters, which SUBSUMES the slice-C
    special-case clear: an empty structured_filters on the SearchString -> bootstrap
    seeds nothing. The surface=="boolean" + variant_kind=="structured_filter" marker
    keeps seed_structured_filters_onto_variants from re-seeding even if a stray lane
    filter were present.
    """
    producer = SearchString(
        id=7,
        name="hybrid lane that demotes",
        boolean='"VP" AND fintech',
        acquisition_mode="linkedin_hybrid",
        structured_filters={"titles": ["VP Fintech"]},  # producer-time lane filter
    )
    state = bootstrap_experiment_state(producer)
    active = state.active_variant
    # Deliberate demote: drop the filters, mark surface=boolean + structured_filter kind.
    active.surface = "boolean"
    active.variant_kind = "structured_filter"
    active.structured_filters = LinkedInStructuredFilters()
    state.apply_shadow(producer)

    # The compat SearchString now carries the demote: empty filters + surface=boolean.
    assert LinkedInStructuredFilters.from_dict(producer.structured_filters).is_empty()
    assert producer.surface == "boolean"

    resumed_string = SearchString.from_dict(producer.to_dict())
    resumed = bootstrap_experiment_state(resumed_string)

    # Demote preserved: NO filters re-seeded, surface=boolean intact.
    assert resumed.active_variant.structured_filters.is_empty()
    assert resumed.active_variant.surface == "boolean"


def test_keyword_led_boolean_lane_resumes_byte_identically_to_pre_g():
    """(d) A keyword-led (boolean) lane resumes identically to pre-G: surface=="" and
    empty lane filters; the SearchString default surface is "" and a checkpoint written
    before slice G still loads. The bare keyword opening path is byte-preserved."""
    producer = SearchString(id=8, name="builders", boolean="foo")
    state = bootstrap_experiment_state(producer)
    state.apply_shadow(producer)

    # No surface stamped (default ""), no filters.
    assert producer.surface == ""
    assert LinkedInStructuredFilters.from_dict(producer.structured_filters).is_empty()

    resumed_string = SearchString.from_dict(producer.to_dict())
    resumed = bootstrap_experiment_state(resumed_string)
    assert resumed.active_variant.surface == ""
    assert resumed.active_variant.structured_filters.is_empty()

    # Opening apply takes the bare keyword path (not the structured plan path).
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.browser.apply_advanced_search_plan = AsyncMock()
        pipeline.browser.enter_search_string = AsyncMock()

        asyncio.run(
            pipeline._apply_opening_search(
                resumed_string, resumed, resumed.current_boolean()
            )
        )

        pipeline.browser.enter_search_string.assert_awaited_once_with("foo")
        pipeline.browser.apply_advanced_search_plan.assert_not_called()


def test_reset_experiment_state_preserves_surface_so_restart_does_not_reenter_keyword():
    """Slice G (restart path): the operator hard-reset / requeue path drives
    from_dict(checkpoint) -> reset_experiment_state(target, state) -> apply_shadow(target)
    (shared/runtime_state/linkedin.py:547-550). reset_experiment_state mints a FRESH
    root whose __post_init__ default surface is "", so without reconstruction
    apply_shadow would write that "" onto the compat SearchString — wiping the persisted
    structured_only posture while the intent still carries the filters forward. The NEXT
    bootstrap would then reconstruct surface="" with non-empty filters, and the slice-D
    include_keyword gate (active.surface != "structured_only") would re-admit the keyword
    the lane was defined to suppress. reset_experiment_state now stamps the durable
    surface onto the fresh root before apply_shadow (mirroring bootstrap:1250), so the
    structured_only posture survives a restart and the keyword stays suppressed.
    """
    producer = SearchString(
        id=11,
        name="structured only lane",
        boolean='"Staff Engineer"',
        acquisition_mode="linkedin_hybrid",
    )
    state = bootstrap_experiment_state(producer)
    active = state.active_variant
    active.surface = "structured_only"
    active.variant_kind = "structured_filter"
    active.structured_filters = LinkedInStructuredFilters(titles=["Staff Engineer"])
    state.apply_shadow(producer)
    assert producer.surface == "structured_only"

    # Restart path: checkpoint round-trips, reset mints a fresh root, apply_shadow re-stamps.
    checkpoint_state = LinkedInExperimentState.from_dict(state.to_dict())
    reset_state = reset_experiment_state(producer, checkpoint_state)
    reset_state.apply_shadow(producer)

    # The persisted surface survives the restart (NOT wiped to "").
    assert producer.surface == "structured_only"
    assert producer.structured_filters["titles"] == ["Staff Engineer"]

    # The next bootstrap reconstructs surface="structured_only", so the slice-D gate
    # keeps the keyword suppressed instead of re-admitting it.
    resumed = bootstrap_experiment_state(SearchString.from_dict(producer.to_dict()))
    resumed_active = resumed.active_variant
    assert resumed_active.surface == "structured_only"
    assert resumed_active.structured_filters.titles == ["Staff Engineer"]
    assert (resumed_active.surface != "structured_only") is False  # include_keyword gate OFF


def test_reset_experiment_state_keyword_led_lane_restart_is_byte_preserved():
    """Slice G (restart path, byte-preserved default): a keyword-led (boolean) lane has
    no durable surface (default ""), so reset_experiment_state stamps "" onto the fresh
    root and apply_shadow leaves search_string.surface == "" — identical to pre-slice-G.
    The empty-surface stamp is a no-op; no filters are written or seeded."""
    producer = SearchString(id=12, name="builders", boolean="foo")
    state = bootstrap_experiment_state(producer)
    state.apply_shadow(producer)
    assert producer.surface == ""

    checkpoint_state = LinkedInExperimentState.from_dict(state.to_dict())
    reset_state = reset_experiment_state(producer, checkpoint_state)
    reset_state.apply_shadow(producer)

    assert producer.surface == ""
    assert LinkedInStructuredFilters.from_dict(producer.structured_filters).is_empty()
    assert reset_state.active_variant.surface == ""
