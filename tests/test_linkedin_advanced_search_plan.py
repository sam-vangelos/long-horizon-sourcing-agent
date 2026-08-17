"""Tests for P3: advanced search controller — plan / classify / lint layer."""

from __future__ import annotations

from types import SimpleNamespace

from linkedin.advanced_search import (
    ALL_KNOWN_CONTROLS,
    DEFER_CONTROLS,
    MOCK_ONLY_CONTROLS,
    STABLE_NOW_CONTROLS,
    AdvancedSearchControl,
    AdvancedSearchPlan,
    ControlApplicationResult,
    classify_control,
    compile_recovery_plan_from_snapshot,
    compile_structured_filters_to_plan,
    lint_boolean_filter_conflicts,
    snapshot_controls_from_plan,
)


# ---------------------------------------------------------------------------
# classify_control
# ---------------------------------------------------------------------------


def test_classify_keywords_stable_now():
    assert classify_control("keywords") == "stable_now"


def test_classify_locations_stable_now():
    # Graduated hop 4 (2026-05-29 live smoke). job_titles + companies graduated in
    # slice H (2026-05-31); only fields_of_study remains mock_only.
    assert classify_control("locations") == "stable_now"


def test_classify_job_titles_stable_now():
    # Graduated slice H (2026-05-31 live smoke: tools/hop4_title_smoke.py PASS).
    assert classify_control("job_titles") == "stable_now"


def test_classify_companies_stable_now():
    # Graduated slice H (2026-05-31 live smoke: tools/hop4_company_smoke.py PASS).
    assert classify_control("companies") == "stable_now"


def test_classify_fields_of_study_mock_only():
    assert classify_control("fields_of_study") == "mock_only"


def test_classify_seniority_defer():
    assert classify_control("seniority") == "defer"


def test_classify_years_of_experience_defer():
    assert classify_control("years_of_experience") == "defer"


def test_classify_industries_defer():
    assert classify_control("industries") == "defer"


def test_classify_unknown():
    assert classify_control("some_future_control") == "unknown"


def test_all_known_controls_covers_all_tiers():
    assert ALL_KNOWN_CONTROLS == STABLE_NOW_CONTROLS | MOCK_ONLY_CONTROLS | DEFER_CONTROLS
    # 4 stable_now (keywords, locations, job_titles, companies) + 1 mock_only
    # (fields_of_study) + 10 defer.
    assert len(ALL_KNOWN_CONTROLS) == 4 + 1 + 10


# ---------------------------------------------------------------------------
# lint_boolean_filter_conflicts
# ---------------------------------------------------------------------------


def test_lint_detects_boolean_duplicating_filter():
    plan = AdvancedSearchPlan(
        keyword_boolean='"machine learning" AND "engineer"',
        controls=[
            AdvancedSearchControl(dimension="job_titles", values=["engineer"]),
        ],
    )
    conflicts = lint_boolean_filter_conflicts(plan)
    assert len(conflicts) == 1
    assert "engineer" in conflicts[0]


def test_lint_no_conflict_when_no_boolean():
    plan = AdvancedSearchPlan(
        keyword_boolean="",
        controls=[AdvancedSearchControl(dimension="job_titles", values=["engineer"])],
    )
    assert lint_boolean_filter_conflicts(plan) == []


def test_lint_no_conflict_when_no_controls():
    plan = AdvancedSearchPlan(keyword_boolean='"test"', controls=[])
    assert lint_boolean_filter_conflicts(plan) == []


def test_lint_ignores_keywords_dimension():
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML engineer"',
        controls=[AdvancedSearchControl(dimension="keywords", values=["ML engineer"])],
    )
    assert lint_boolean_filter_conflicts(plan) == []


def test_lint_detects_multiple_conflicts():
    plan = AdvancedSearchPlan(
        keyword_boolean='"python" AND "google"',
        controls=[
            AdvancedSearchControl(dimension="skills", values=["python"]),
            AdvancedSearchControl(dimension="companies", values=["Google"]),
        ],
    )
    conflicts = lint_boolean_filter_conflicts(plan)
    assert len(conflicts) == 2


# ---------------------------------------------------------------------------
# AdvancedSearchPlan with only keywords (stable_now)
# ---------------------------------------------------------------------------


def test_plan_with_keywords_only():
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML" AND "engineer"',
        controls=[AdvancedSearchControl(dimension="keywords", values=['"ML" AND "engineer"'])],
    )
    assert plan.acquisition_mode == "boolean_only"
    assert len(plan.controls) == 1
    assert classify_control(plan.controls[0].dimension) == "stable_now"


# ---------------------------------------------------------------------------
# Plans with mock_only and defer controls
# ---------------------------------------------------------------------------


def test_plan_mock_only_classified():
    # fields_of_study is the remaining mock_only control after slice H graduated
    # job_titles + companies to stable_now.
    plan = AdvancedSearchPlan(
        controls=[AdvancedSearchControl(dimension="fields_of_study", values=["CS"])],
    )
    assert classify_control(plan.controls[0].dimension) == "mock_only"


def test_plan_defer_classified():
    plan = AdvancedSearchPlan(
        controls=[AdvancedSearchControl(dimension="seniority", values=["Director"])],
    )
    assert classify_control(plan.controls[0].dimension) == "defer"


# ---------------------------------------------------------------------------
# No live application for mock_only / defer
# ---------------------------------------------------------------------------


def test_stable_now_controls():
    # locations graduated hop 4 (2026-05-29); job_titles + companies graduated slice H
    # (2026-05-31), each gated on a passing live smoke.
    assert STABLE_NOW_CONTROLS == frozenset(
        {"keywords", "locations", "job_titles", "companies"}
    )


# ---------------------------------------------------------------------------
# ControlApplicationResult
# ---------------------------------------------------------------------------


def test_result_to_dict():
    result = ControlApplicationResult(
        success=False,
        applied_controls=["keywords"],
        failed_controls=["locations"],
        unsupported_controls=["seniority"],
        fallback_to_boolean=True,
        reason="test",
    )
    d = result.to_dict()
    assert d["success"] is False
    assert "keywords" in d["applied_controls"]
    assert "locations" in d["failed_controls"]
    assert "seniority" in d["unsupported_controls"]
    assert d["fallback_to_boolean"] is True


# ---------------------------------------------------------------------------
# snapshot_controls_from_plan
# ---------------------------------------------------------------------------


def test_snapshot_from_plan():
    plan = AdvancedSearchPlan(
        keyword_boolean='"test"',
        controls=[
            AdvancedSearchControl(dimension="keywords", values=['"test"']),
            AdvancedSearchControl(dimension="fields_of_study", values=["CS"]),
        ],
        acquisition_mode="linkedin_hybrid",
    )
    snap = snapshot_controls_from_plan(plan)
    assert snap["keyword_boolean"] == '"test"'
    assert snap["acquisition_mode"] == "linkedin_hybrid"
    assert len(snap["controls"]) == 2
    assert snap["controls"][0]["tier"] == "stable_now"
    # fields_of_study is the surviving mock_only example post slice H.
    assert snap["controls"][1]["tier"] == "mock_only"


# ---------------------------------------------------------------------------
# Slice D — structured_only expressibility: include_keyword suppresses the
# keyword control AND zeroes plan.keyword_boolean so the downstream re-add
# guards (apply_advanced_search_plan, compile_recovery_plan_from_snapshot)
# stay closed off one lever.
# ---------------------------------------------------------------------------


def _filters(*, titles=None, companies=None, locations=None):
    return SimpleNamespace(
        titles=list(titles or []),
        companies=list(companies or []),
        sidebar_filters={"locations": list(locations or [])} if locations else {},
        advanced_filters={},
    )


def test_compile_include_keyword_false_drops_keyword_control_and_zeroes_boolean():
    """Slice D part 1 / test (a): a structured_only compile passes
    include_keyword=False — no keywords control is emitted AND the returned
    plan.keyword_boolean is the empty string so the apply-time and recovery-time
    if-keyword_boolean guards both stay closed.
    """
    plan = compile_structured_filters_to_plan(
        _filters(locations=["NYC"]),
        keyword_boolean='"ML" AND "engineer"',
        include_keyword=False,
    )
    assert all(c.dimension != "keywords" for c in plan.controls)
    assert plan.keyword_boolean == ""
    # The structured control still compiles — only the keyword is suppressed.
    assert [c.dimension for c in plan.controls] == ["locations"]


def test_compile_include_keyword_default_true_keeps_keyword_control():
    """Slice D regression (part of test (e) at the compile layer): the default
    keeps the keyword-led behavior untouched — keywords control present, boolean
    preserved.
    """
    plan = compile_structured_filters_to_plan(
        _filters(locations=["NYC"]),
        keyword_boolean='"ML" AND "engineer"',
    )
    assert plan.controls[0].dimension == "keywords"
    assert plan.controls[0].values == ['"ML" AND "engineer"']
    assert plan.keyword_boolean == '"ML" AND "engineer"'


def test_recovery_plan_from_structured_only_snapshot_does_not_re_add_keyword():
    """Slice D test (d): a structured_only snapshot persists keyword_boolean=""
    (because the structured_only compile zeroed it before snapshot_controls_from_plan
    recorded it). On replay, compile_recovery_plan_from_snapshot must NOT re-add a
    keyword control — the empty boolean trips the if-keyword_boolean guard at :295.
    """
    structured_only_plan = compile_structured_filters_to_plan(
        _filters(locations=["NYC"]),
        keyword_boolean='"ML" AND "engineer"',
        include_keyword=False,
    )
    snap = snapshot_controls_from_plan(structured_only_plan)
    assert snap["keyword_boolean"] == ""

    snapshot = SimpleNamespace(
        advanced_search_controls=snap,
        keyword_boolean=snap["keyword_boolean"],
    )
    recovered = compile_recovery_plan_from_snapshot(snapshot)
    assert all(c.dimension != "keywords" for c in recovered.controls)
    assert recovered.keyword_boolean == ""
    assert [c.dimension for c in recovered.controls] == ["locations"]


def test_recovery_plan_from_keyword_led_snapshot_still_re_adds_keyword():
    """Slice D regression (test (e), recovery layer): a normal keyword-led
    snapshot still re-adds the keyword control on replay — the :295 guard fires
    only on the empty boolean, not on the structured_only marker leaking.
    """
    keyword_led_plan = compile_structured_filters_to_plan(
        _filters(locations=["NYC"]),
        keyword_boolean='"ML" AND "engineer"',
    )
    snap = snapshot_controls_from_plan(keyword_led_plan)
    snapshot = SimpleNamespace(
        advanced_search_controls=snap,
        keyword_boolean=snap["keyword_boolean"],
    )
    recovered = compile_recovery_plan_from_snapshot(snapshot)
    assert any(c.dimension == "keywords" for c in recovered.controls)
    assert recovered.keyword_boolean == '"ML" AND "engineer"'


def test_recovery_plan_re_adds_from_top_level_boolean_not_controls_copy():
    """Slice D divergence pin: compile_recovery_plan_from_snapshot's re-add guard
    reads the snapshot's TOP-LEVEL keyword_boolean (advanced_search.py:~304), NOT
    the advanced_search_controls['keyword_boolean'] copy. So a structured_only
    compile that zeroes the controls-dict copy is NOT self-protecting — the BURDEN
    is on the producer (orchestrator._capture_recovery_snapshot) to source the
    top-level field from the gated plan.keyword_boolean, not search_string.boolean.

    This is the exact divergence the in-process recovery path used to hit: the
    controls copy was correctly '' while the top-level field still carried the
    boolean, so the keyword leaked back in on replay. Pins the layer contract the
    orchestrator fix depends on; goes red if the guard ever starts reading the
    controls-dict copy instead (which would silently re-mask the producer bug).
    """
    structured_only_plan = compile_structured_filters_to_plan(
        _filters(locations=["NYC"]),
        keyword_boolean='"ML" AND "engineer"',
        include_keyword=False,
    )
    snap = snapshot_controls_from_plan(structured_only_plan)
    assert snap["keyword_boolean"] == ""  # the gated controls copy is zeroed

    # Reproduce the OLD orchestrator divergence: top-level field sourced from
    # search_string.boolean (non-empty) even though the controls copy is ''.
    leaky = SimpleNamespace(advanced_search_controls=snap, keyword_boolean='"ML" AND "engineer"')
    recovered_leaky = compile_recovery_plan_from_snapshot(leaky)
    assert [c.dimension for c in recovered_leaky.controls] == ["keywords", "locations"]
    assert recovered_leaky.keyword_boolean == '"ML" AND "engineer"'

    # The FIXED producer threads plan.keyword_boolean ('') into the top-level field.
    fixed = SimpleNamespace(
        advanced_search_controls=snap,
        keyword_boolean=structured_only_plan.keyword_boolean,
    )
    recovered_fixed = compile_recovery_plan_from_snapshot(fixed)
    assert all(c.dimension != "keywords" for c in recovered_fixed.controls)
    assert recovered_fixed.keyword_boolean == ""
    assert [c.dimension for c in recovered_fixed.controls] == ["locations"]
