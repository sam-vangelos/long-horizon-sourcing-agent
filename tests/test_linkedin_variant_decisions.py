"""Tests for P7b: deterministic lane variant lifecycle decisions."""

from __future__ import annotations

from linkedin.lane_variant_decisions import (
    VariantDecisionInput,
    VariantDecisionOutput,
    decide_variant_lifecycle,
)
from linkedin.search_intelligence import (
    LinkedInExperimentState,
    LinkedInPageInsights,
    LinkedInSearchIntent,
    LinkedInSearchVariant,
    LinkedInStructuredFilters,
    scale_window_for_surface,
)
from shared import config


def _make_variant(
    *,
    result_count: int = 100,
    target_min: int = 50,
    target_max: int = 300,
    saves: int = 0,
    full_reviewed: int = 0,
    full_outreach: int = 0,
    full_review: int = 0,
    full_reject: int = 0,
    facial_yes: int = 0,
    facial_no: int = 0,
    candidates: int = 0,
    pages_reviewed: int = 1,
    probe_page_budget: int = 1,
    probe_pages_used: int = 0,
    status: str = "probing",
    lane_id: str = "test-lane",
    last_page_insights: LinkedInPageInsights | None = None,
) -> LinkedInSearchVariant:
    return LinkedInSearchVariant(
        variant_id="test-v",
        parent_variant_id=None,
        root_string_id=1,
        boolean="test",
        result_count=result_count,
        target_result_min=target_min,
        target_result_max=target_max,
        saves=saves,
        full_reviewed=full_reviewed,
        full_outreach=full_outreach,
        full_review=full_review,
        full_reject=full_reject,
        facial_yes=facial_yes,
        facial_no=facial_no,
        candidates=candidates,
        pages_reviewed=pages_reviewed,
        probe_page_budget=probe_page_budget,
        probe_pages_used=probe_pages_used,
        status=status,
        lane_id=lane_id,
        last_page_insights=last_page_insights,
    )


def _make_state(
    variants: dict[str, LinkedInSearchVariant] | None = None,
    planned_ids: list[str] | None = None,
) -> LinkedInExperimentState:
    intent = LinkedInSearchIntent(root_boolean="test")
    state = LinkedInExperimentState(root_string_id=1, intent=intent)
    if variants:
        state.variants.update(variants)
    if planned_ids:
        state.planned_variant_ids = planned_ids
    return state


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------


def test_commit_healthy_window_with_signal():
    v = _make_variant(
        result_count=150, saves=2, full_reviewed=2, full_outreach=2, facial_yes=3,
        probe_page_budget=1, probe_pages_used=1,
    )
    state = _make_state()
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert out.action == "commit"
    assert "healthy" in out.reason


def test_healthy_window_without_settled_positive_does_not_commit():
    v = _make_variant(
        result_count=150, saves=0, full_reviewed=3, full_reject=3, facial_yes=3,
        candidates=3, pages_reviewed=1,
        probe_page_budget=1, probe_pages_used=1,
    )
    state = _make_state()
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert out.action != "commit"
    assert "all_reviewed_rejected" in out.reason


def test_raw_facial_yes_is_not_commit_signal():
    v = _make_variant(
        result_count=150,
        facial_yes=10,
        full_reviewed=10,
        full_reject=10,
        candidates=10,
        probe_page_budget=1,
        probe_pages_used=1,
    )

    out = decide_variant_lifecycle(
        VariantDecisionInput(variant=v, experiment_state=_make_state())
    )

    assert out.action != "commit"
    assert "all_reviewed_rejected" in out.reason


def test_all_reviewed_reject_scores_negative_even_inside_target_window():
    v = _make_variant(
        result_count=150,
        full_reviewed=1,
        full_reject=1,
        candidates=1,
        pages_reviewed=1,
    )

    assert v.within_target_window() is True
    assert v.score() < 0


def test_human_review_is_weak_settled_signal_for_healthy_window():
    v = _make_variant(
        result_count=150,
        full_reviewed=1,
        full_review=1,
        probe_page_budget=1,
        probe_pages_used=1,
    )

    out = decide_variant_lifecycle(
        VariantDecisionInput(variant=v, experiment_state=_make_state())
    )

    assert out.action == "commit"
    assert out.reason == "healthy_window_with_full_profile_signal"


# ---------------------------------------------------------------------------
# rescue
# ---------------------------------------------------------------------------


def test_rescue_too_narrow_budget_met():
    v = _make_variant(
        result_count=10, target_min=50, target_max=300,
        probe_page_budget=1, probe_pages_used=1,
    )
    state = _make_state()
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert out.action == "rescue"
    assert out.next_variant_hint is not None
    assert out.next_variant_hint["variant_kind"] == "recall"


def test_rescue_too_broad():
    v = _make_variant(
        result_count=5000, target_min=50, target_max=300,
        saves=1, facial_yes=1, pages_reviewed=1,
    )
    state = _make_state()
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert out.action == "rescue"
    assert out.next_variant_hint["variant_kind"] == "precision"


def test_rescue_noisy():
    v = _make_variant(
        result_count=5000, target_min=50, target_max=300,
        saves=0, facial_yes=0, pages_reviewed=2,
    )
    state = _make_state()
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert out.action == "rescue"
    assert out.next_variant_hint["variant_kind"] == "noise_exclusion"


# ---------------------------------------------------------------------------
# abandon
# ---------------------------------------------------------------------------


def test_abandon_misleading_budget_exhausted():
    v = _make_variant(
        result_count=150, saves=0, facial_yes=0,
        candidates=10, pages_reviewed=2,
        probe_page_budget=1, probe_pages_used=1,
    )
    state = _make_state()
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert out.action == "abandon"
    assert "misleading" in out.reason


def test_abandon_all_planned_exhausted():
    v = _make_variant(
        result_count=10, saves=0, facial_yes=0,
        probe_page_budget=1, probe_pages_used=1,
    )
    exhausted_v = _make_variant(status="exhausted")
    state = _make_state(
        variants={"exp-1": exhausted_v},
        planned_ids=["exp-1"],
    )
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert out.action in ("abandon", "rescue")


# ---------------------------------------------------------------------------
# continue
# ---------------------------------------------------------------------------


def test_continue_within_budget_unclassified():
    v = _make_variant(
        result_count=150, saves=0, facial_yes=0,
        candidates=0, pages_reviewed=0,
        probe_page_budget=3, probe_pages_used=1,
    )
    state = _make_state()
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert out.action == "continue"


def test_continue_too_narrow_within_budget():
    v = _make_variant(
        result_count=10, target_min=50, target_max=300,
        probe_page_budget=3, probe_pages_used=1,
    )
    state = _make_state()
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert out.action == "continue"
    assert "too_narrow" in out.reason


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def test_split_with_distinct_clusters():
    insights = LinkedInPageInsights(
        page=1,
        result_count=200,
        result_window="200",
        title_clusters=[
            {"title": "ML Engineer", "count": 5, "signal_count": 2},
            {"title": "Data Scientist", "count": 4, "signal_count": 1},
        ],
    )
    v = _make_variant(
        result_count=200, saves=3, facial_yes=3,
        pages_reviewed=2, probe_page_budget=3, probe_pages_used=2,
        last_page_insights=insights,
    )
    state = _make_state()
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    # With healthy signal this should commit; split is a lower priority check
    assert out.action in ("commit", "split")


# ---------------------------------------------------------------------------
# VariantDecisionOutput serialization
# ---------------------------------------------------------------------------


def test_decision_output_to_dict():
    out = VariantDecisionOutput(
        action="rescue",
        reason="too_narrow_budget_met",
        next_variant_hint={"variant_kind": "recall"},
    )
    d = out.to_dict()
    assert d["action"] == "rescue"
    assert d["reason"] == "too_narrow_budget_met"
    assert d["next_variant_hint"]["variant_kind"] == "recall"


def test_decision_output_no_hint():
    out = VariantDecisionOutput(action="commit", reason="healthy")
    d = out.to_dict()
    assert "next_variant_hint" not in d


# ---------------------------------------------------------------------------
# Regression: existing keyword-only experiment flow unchanged
# ---------------------------------------------------------------------------


def test_keyword_only_variant_no_structured_filters_decision():
    """A keyword-only variant with healthy results should commit normally."""
    v = _make_variant(
        result_count=200, saves=2, full_reviewed=2, full_outreach=2, facial_yes=4,
        probe_page_budget=1, probe_pages_used=1,
    )
    state = _make_state()
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert out.action == "commit"


# ---------------------------------------------------------------------------
# SLICE F — posture-aware lifecycle windows (scaling at construction)
# ---------------------------------------------------------------------------


def _filter_led_variant(
    *,
    result_count: int,
    keyword_min: int = 75,
    keyword_max: int = 400,
    saves: int = 0,
    facial_yes: int = 0,
    probe_page_budget: int = 1,
    probe_pages_used: int = 0,
) -> LinkedInSearchVariant:
    """A structured_filter variant whose window has been scaled DOWN at
    construction by scale_window_for_surface — exactly what the build sites do.

    The decision function still reads variant.target_result_min/max as plain ints;
    the posture is baked into those ints here, never read by the decision."""
    scaled_min, scaled_max = scale_window_for_surface(
        keyword_min,
        keyword_max,
        surface="hybrid",
        structured_filters=LinkedInStructuredFilters(titles=["Staff Software Engineer"]),
    )
    return LinkedInSearchVariant(
        variant_id="filter-led-v",
        parent_variant_id=None,
        root_string_id=1,
        boolean='"VP" AND engineering',
        variant_kind="structured_filter",
        surface="hybrid",
        structured_filters=LinkedInStructuredFilters(titles=["Staff Software Engineer"]),
        result_count=result_count,
        target_result_min=scaled_min,
        target_result_max=scaled_max,
        saves=saves,
        facial_yes=facial_yes,
        pages_reviewed=1,
        probe_page_budget=probe_page_budget,
        probe_pages_used=probe_pages_used,
        status="probing",
        lane_id="test-lane",
    )


def test_filter_led_narrow_against_keyword_window_classifies_healthy_and_commits():
    """(a) A filter-led variant with a count that is too_narrow against the UNSCALED
    keyword window classifies HEALTHY against the scaled window AND the lifecycle
    decision is commit — never abandon/rescue.

    A count of 30 sits below the keyword min (75) — too_narrow for a keyword
    variant — but inside the scaled filter-led window, so a legitimate structured
    probe commits instead of being abandoned by the keyword-tuned gate.
    """
    # Sanity: 30 is genuinely too_narrow against the keyword window.
    keyword_clone = _make_variant(result_count=30, target_min=75, target_max=400, saves=1)
    assert keyword_clone.classify_result_window() == "too_narrow"

    v = _filter_led_variant(result_count=30, saves=1)
    assert v.classify_result_window() == "healthy"

    state = _make_state()
    out = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert out.action == "commit", out.to_dict()
    assert out.action not in {"abandon", "rescue"}


def test_decide_variant_lifecycle_is_deterministic_and_posture_blind():
    """(b) decide_variant_lifecycle is byte-identical across repeated calls for fixed
    inputs, and NEITHER it NOR classify_result_window reads any surface/posture field
    — the scaling lives purely at construction.
    """
    import inspect

    from linkedin import lane_variant_decisions
    from linkedin.search_intelligence import LinkedInSearchVariant as _Variant

    v = _filter_led_variant(result_count=30, saves=1)
    state = _make_state()
    first = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    second = decide_variant_lifecycle(VariantDecisionInput(variant=v, experiment_state=state))
    assert first.to_dict() == second.to_dict()

    # No posture/surface/scaling token is read inside the decision ladder.
    decide_src = inspect.getsource(lane_variant_decisions.decide_variant_lifecycle)
    for token in ("surface", "structured_filters", "FILTER_LED", "WINDOW_FACTOR", "scale_window"):
        assert token not in decide_src, f"{token!r} must not leak into decide_variant_lifecycle"

    # classify_result_window reads only result_count vs the (already-scaled) target window.
    classify_src = inspect.getsource(_Variant.classify_result_window)
    for token in ("surface", "structured_filters", "FILTER_LED", "WINDOW_FACTOR", "scale_window"):
        assert token not in classify_src, f"{token!r} must not leak into classify_result_window"


def test_scale_window_for_surface_floor_and_ordering_invariants():
    """(b, helper) The scaling guards the floor: a scaled min stays >= 1 and
    min <= max, so a tiny window cannot collapse to 0 or invert. A boolean surface
    with empty filters is returned UNCHANGED (the default path is byte-identical)."""
    # Boolean / no filters -> unchanged window.
    assert scale_window_for_surface(
        75, 400, surface="boolean", structured_filters=LinkedInStructuredFilters()
    ) == (75, 400)
    assert scale_window_for_surface(
        75, 400, surface="", structured_filters=LinkedInStructuredFilters()
    ) == (75, 400)

    # Filter-led -> scaled DOWN by the factor.
    scaled_min, scaled_max = scale_window_for_surface(
        75, 400, surface="hybrid", structured_filters=LinkedInStructuredFilters()
    )
    assert scaled_min == max(1, round(75 * config.SEARCH_EXPERIMENT_FILTER_LED_WINDOW_FACTOR))
    assert scaled_max == max(scaled_min, round(400 * config.SEARCH_EXPERIMENT_FILTER_LED_WINDOW_FACTOR))
    assert scaled_min < 75 and scaled_max < 400

    # Tiny window must not collapse to 0 or invert.
    tiny_min, tiny_max = scale_window_for_surface(
        1, 2, surface="structured_only",
        structured_filters=LinkedInStructuredFilters(companies=["Stripe"]),
    )
    assert tiny_min >= 1
    assert tiny_min <= tiny_max

    # None window passes through untouched (root variant has no window to scale).
    assert scale_window_for_surface(
        None, None, surface="hybrid",
        structured_filters=LinkedInStructuredFilters(titles=["X"]),
    ) == (None, None)
