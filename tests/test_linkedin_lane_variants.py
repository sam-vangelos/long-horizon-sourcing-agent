"""Tests for P7a: lane-aware variant lifecycle."""

from __future__ import annotations

from linkedin.search_intelligence import (
    VARIANT_LIFECYCLE_STATUSES,
    LinkedInExperimentState,
    LinkedInPageInsights,
    LinkedInSearchIntent,
    LinkedInSearchVariant,
    LinkedInStructuredFilters,
    compile_lane_variant_to_linkedin,
    spawn_rescue_variant_from_hint,
)
from shared.sourcing_lanes import (
    RESULT_WINDOW_HEALTH_STATES,
    VARIANT_KINDS,
    LaneVariant,
)


# ---------------------------------------------------------------------------
# LaneVariant (shared)
# ---------------------------------------------------------------------------


def test_lane_variant_round_trip():
    lv = LaneVariant(
        variant_id="v1",
        lane_id="ml-eng",
        variant_kind="precision",
        hypothesis="tighter titles",
        status="probing",
        reason="too_broad initial run",
        boolean_intent='"ML" AND "engineer"',
        structured_controls={"titles": ["ML Engineer"]},
        target_result_min=50,
        target_result_max=300,
        probe_budget={"page_limit": 2},
    )
    d = lv.to_dict()
    restored = LaneVariant.from_dict(d)
    assert restored.variant_id == "v1"
    assert restored.lane_id == "ml-eng"
    assert restored.variant_kind == "precision"
    assert restored.status == "probing"
    assert restored.target_result_min == 50
    assert restored.probe_budget == {"page_limit": 2}


def test_variant_kinds_constant():
    assert "original" in VARIANT_KINDS
    assert "keyword_focus" in VARIANT_KINDS
    assert "precision" in VARIANT_KINDS
    assert "recall" in VARIANT_KINDS
    assert "rescue" in VARIANT_KINDS
    assert len(VARIANT_KINDS) == 7


def test_result_window_health_states():
    assert RESULT_WINDOW_HEALTH_STATES == {
        "too_narrow", "too_broad", "noisy", "misleading", "healthy",
    }


# ---------------------------------------------------------------------------
# compile_lane_variant_to_linkedin
# ---------------------------------------------------------------------------


def test_compile_maps_basic_fields():
    lv = LaneVariant(
        variant_id="test-v",
        lane_id="lane-a",
        variant_kind="recall",
        hypothesis="broader titles",
        boolean_intent='"software" OR "developer"',
        target_result_min=100,
        target_result_max=500,
        probe_budget={"page_limit": 3},
    )
    li_var = compile_lane_variant_to_linkedin(lv, root_string_id=42)
    assert li_var.variant_id == "test-v"
    assert li_var.lane_id == "lane-a"
    assert li_var.variant_kind == "recall"
    assert li_var.hypothesis == "broader titles"
    assert li_var.boolean == '"software" OR "developer"'
    assert li_var.root_string_id == 42
    assert li_var.target_result_min == 100
    assert li_var.target_result_max == 500
    assert li_var.probe_page_budget == 3
    assert li_var.status == "planned"


def test_compile_maps_structured_controls():
    lv = LaneVariant(
        variant_id="v-struct",
        lane_id="lane-b",
        structured_controls={"titles": ["Data Scientist"], "companies": ["Meta"]},
    )
    li_var = compile_lane_variant_to_linkedin(lv)
    assert li_var.structured_filters.titles == ["Data Scientist"]
    assert li_var.structured_filters.companies == ["Meta"]


def test_compile_preserves_reason_as_lifecycle_reason():
    lv = LaneVariant(
        variant_id="v-reason",
        lane_id="lane-c",
        reason="too_broad initial window",
    )
    li_var = compile_lane_variant_to_linkedin(lv)
    assert li_var.lifecycle_reason == "too_broad initial window"


# ---------------------------------------------------------------------------
# LinkedInSearchVariant — new fields
# ---------------------------------------------------------------------------


def test_variant_new_fields_round_trip():
    v = LinkedInSearchVariant(
        variant_id="v1",
        parent_variant_id=None,
        root_string_id=1,
        boolean="test",
        lane_id="eng",
        lifecycle_reason="committed_healthy",
        result_window_health="healthy",
        probe_page_budget=3,
        probe_pages_used=2,
    )
    d = v.to_dict()
    assert d["lane_id"] == "eng"
    assert d["lifecycle_reason"] == "committed_healthy"
    assert d["result_window_health"] == "healthy"
    assert d["probe_page_budget"] == 3
    assert d["probe_pages_used"] == 2

    restored = LinkedInSearchVariant.from_dict(d)
    assert restored.lane_id == "eng"
    assert restored.lifecycle_reason == "committed_healthy"
    assert restored.probe_page_budget == 3


def test_variant_empty_new_fields_omitted_from_dict():
    """Existing payloads remain byte-identical for variants without new fields."""
    v = LinkedInSearchVariant(
        variant_id="root",
        parent_variant_id=None,
        root_string_id=1,
        boolean="test",
    )
    d = v.to_dict()
    assert "lane_id" not in d
    assert "lifecycle_reason" not in d
    assert "result_window_health" not in d
    assert "probe_page_budget" not in d


# ---------------------------------------------------------------------------
# classify_result_window
# ---------------------------------------------------------------------------


def test_classify_too_narrow():
    v = LinkedInSearchVariant(
        variant_id="v1", parent_variant_id=None, root_string_id=1,
        boolean="x", result_count=10, target_result_min=50, target_result_max=200,
    )
    assert v.classify_result_window() == "too_narrow"


def test_classify_too_broad():
    v = LinkedInSearchVariant(
        variant_id="v1", parent_variant_id=None, root_string_id=1,
        boolean="x", result_count=5000, target_result_min=50, target_result_max=200,
        pages_reviewed=1, saves=1,
    )
    assert v.classify_result_window() == "too_broad"


def test_classify_noisy():
    v = LinkedInSearchVariant(
        variant_id="v1", parent_variant_id=None, root_string_id=1,
        boolean="x", result_count=5000, target_result_min=50, target_result_max=200,
        pages_reviewed=2, saves=0, facial_yes=0,
    )
    assert v.classify_result_window() == "noisy"


def test_classify_healthy():
    v = LinkedInSearchVariant(
        variant_id="v1", parent_variant_id=None, root_string_id=1,
        boolean="x", result_count=100, target_result_min=50, target_result_max=200,
        pages_reviewed=1, saves=1,
    )
    assert v.classify_result_window() == "healthy"


def test_classify_misleading():
    v = LinkedInSearchVariant(
        variant_id="v1", parent_variant_id=None, root_string_id=1,
        boolean="x", result_count=100, target_result_min=50, target_result_max=200,
        pages_reviewed=1, candidates=10, saves=0, facial_yes=0,
    )
    assert v.classify_result_window() == "misleading"


def test_classify_zero_results():
    v = LinkedInSearchVariant(
        variant_id="v1", parent_variant_id=None, root_string_id=1,
        boolean="x", result_count=0,
    )
    assert v.classify_result_window() == "too_narrow"


# ---------------------------------------------------------------------------
# LinkedInExperimentState — lane_id
# ---------------------------------------------------------------------------


def test_experiment_state_lane_id_round_trip():
    intent = LinkedInSearchIntent(root_boolean="test")
    state = LinkedInExperimentState(root_string_id=1, intent=intent, lane_id="ml-eng")
    d = state.to_dict()
    assert d["lane_id"] == "ml-eng"

    restored = LinkedInExperimentState.from_dict(d)
    assert restored.lane_id == "ml-eng"


def test_experiment_state_no_lane_id_omitted():
    intent = LinkedInSearchIntent(root_boolean="test")
    state = LinkedInExperimentState(root_string_id=1, intent=intent)
    d = state.to_dict()
    assert "lane_id" not in d


# ---------------------------------------------------------------------------
# VARIANT_LIFECYCLE_STATUSES
# ---------------------------------------------------------------------------


def test_lifecycle_statuses_shape():
    assert VARIANT_LIFECYCLE_STATUSES == frozenset({
        "planned", "probing", "active", "explored", "committed", "exhausted", "abandoned",
    })


def test_spawn_recall_rescue_drops_a_structured_filter():
    parent = LinkedInSearchVariant(
        variant_id="v1",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer"',
        structured_filters=LinkedInStructuredFilters(titles=["ML Engineer"]),
        probe_pages_used=1,
    )
    spawned = spawn_rescue_variant_from_hint(
        parent,
        hint={"variant_kind": "recall", "action": "broaden"},
        root_string_id=1,
    )
    assert spawned is not None
    assert spawned.variant_id != parent.variant_id
    assert spawned.structured_filters.titles == []


def test_spawn_precision_rescue_adds_noise_exclusion():
    parent = LinkedInSearchVariant(
        variant_id="v1",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer"',
        last_page_insights=LinkedInPageInsights(
            page=1,
            result_count=5000,
            result_window="too_broad",
            noise_anchors=["recruiter"],
        ),
        probe_pages_used=1,
    )
    spawned = spawn_rescue_variant_from_hint(
        parent,
        hint={"variant_kind": "precision", "action": "narrow"},
        root_string_id=1,
    )
    assert spawned is not None
    assert spawned.boolean != parent.boolean
    assert 'NOT ("recruiter")' in spawned.boolean


def test_spawn_rescue_returns_none_without_material_change():
    parent = LinkedInSearchVariant(
        variant_id="v1",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML"',
        probe_pages_used=1,
    )
    assert (
        spawn_rescue_variant_from_hint(
            parent,
            hint={"variant_kind": "precision", "action": "narrow"},
            root_string_id=1,
        )
        is None
    )


# ---------------------------------------------------------------------------
# Existing behavior preservation
# ---------------------------------------------------------------------------


def test_begin_experiment_round_still_works():
    intent = LinkedInSearchIntent(root_boolean="test")
    state = LinkedInExperimentState(root_string_id=1, intent=intent)
    variants = [
        LinkedInSearchVariant(
            variant_id="exp-1", parent_variant_id="root",
            root_string_id=1, boolean="alt1",
        ),
    ]
    state.begin_experiment_round(variants)
    assert state.mode == "experiment"
    assert "exp-1" in state.planned_variant_ids
    assert state.experiment_round == 1


def test_activate_variant_still_works():
    intent = LinkedInSearchIntent(root_boolean="test")
    state = LinkedInExperimentState(root_string_id=1, intent=intent)
    variants = [
        LinkedInSearchVariant(
            variant_id="exp-1", parent_variant_id="root",
            root_string_id=1, boolean="alt1",
        ),
    ]
    state.begin_experiment_round(variants)
    activated = state.activate_variant("exp-1")
    assert activated.status == "probing"
    assert state.active_variant_id == "exp-1"


def test_commit_variant_still_works():
    intent = LinkedInSearchIntent(root_boolean="test")
    state = LinkedInExperimentState(root_string_id=1, intent=intent)
    state.commit_variant("root")
    assert state.committed_variant_id == "root"
    assert state.mode == "paginate"
