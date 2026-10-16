"""Tests for P3: browser-level advanced search integration."""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import pytest

from linkedin.advanced_search import (
    AdvancedSearchControl,
    AdvancedSearchPlan,
    apply_advanced_search_plan,
    _apply_stable_control,
)
from linkedin.browser import (
    FacetReconciliationError,
    LinkedInBrowser,
    format_facet_reconciliation_console_line,
)
from shared.governor import UNGOVERNED_FOR_TESTS
from shared.storage import read_jsonl


def _make_browser_with_enter_search(succeed: bool = True) -> LinkedInBrowser:
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(
        return_value="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    browser._page = page
    if succeed:
        browser.enter_search_string = AsyncMock(
            return_value=MagicMock(typing_result=MagicMock(), results_wait_ms=500)
        )
    else:
        browser.enter_search_string = AsyncMock(side_effect=RuntimeError("Failed"))
    return browser


# ---------------------------------------------------------------------------
# Mocked keyword control application
# ---------------------------------------------------------------------------


def test_apply_keywords_success():
    browser = _make_browser_with_enter_search(succeed=True)
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML" AND "engineer"',
        controls=[AdvancedSearchControl(dimension="keywords", values=['"ML" AND "engineer"'])],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is True
    assert "keywords" in result.applied_controls
    assert result.fallback_to_boolean is False


def test_apply_keywords_failure_triggers_fallback():
    browser = _make_browser_with_enter_search(succeed=False)
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML" AND "engineer"',
        controls=[AdvancedSearchControl(dimension="keywords", values=['"ML" AND "engineer"'])],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is False
    assert result.fallback_to_boolean is True
    assert "keywords" in result.failed_controls


# ---------------------------------------------------------------------------
# Hop 4 (canary: locations) — routing + boundary contract, behind the gate
#
# 'locations' is still in MOCK_ONLY_CONTROLS, so apply_advanced_search_plan shunts it
# to unsupported_controls before _apply_stable_control runs. These tests pin the routing
# branch and the fail-closed empty guard by exercising _apply_stable_control /
# apply_location_filter directly (bypassing the classify gate) — green now, no STABLE_NOW
# flip required. The real apply_location_filter is implemented against the Pass-5 capture
# but live-verified by tools/hop4_location_smoke.py (browser methods are not mock-page
# unit-tested in this codebase); the constant + apply_advanced_search_plan-level pin flips
# are pre-staged in plans/hop4-structured-filter-execution.md until that smoke passes.
# ---------------------------------------------------------------------------


def _make_browser_with_location_filter(succeed: bool = True) -> LinkedInBrowser:
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(
        return_value="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    browser._page = page
    browser.apply_location_filter = AsyncMock(return_value=succeed)
    return browser


def test_apply_stable_control_routes_locations_to_apply_location_filter():
    browser = _make_browser_with_location_filter(succeed=True)
    ctrl = AdvancedSearchControl(dimension="locations", values=["NYC"], temporal_scope="any")
    plan = AdvancedSearchPlan(controls=[ctrl], keyword_boolean='"ML"')
    ok = asyncio.run(_apply_stable_control(browser, ctrl, plan))
    assert ok is True
    browser.apply_location_filter.assert_awaited_once()


def test_apply_stable_control_locations_value_reads_back():
    browser = _make_browser_with_location_filter(succeed=True)
    ctrl = AdvancedSearchControl(
        dimension="locations", values=["New York City"], temporal_scope="current"
    )
    plan = AdvancedSearchPlan(controls=[ctrl], keyword_boolean='"ML"')
    asyncio.run(_apply_stable_control(browser, ctrl, plan))
    # The dispatcher passes values + temporal_scope through to apply_location_filter
    # verbatim. apply_location_filter itself ignores the scope for the facet dropdown
    # and always applies "Current or past"; that neutralization lives in the browser.
    call = browser.apply_location_filter.await_args
    assert call.args[0] == ["New York City"]
    assert call.kwargs.get("temporal_scope") == "current"


def test_apply_location_filter_empty_values_is_noop():
    # Empty/blank values short-circuit to False before any page interaction — the
    # fail-closed contract the controller relies on. The full live flow is verified by
    # tools/hop4_location_smoke.py on a real seat.
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    assert asyncio.run(browser.apply_location_filter([])) is False
    assert asyncio.run(browser.apply_location_filter(["  "])) is False


# ---------------------------------------------------------------------------
# Unsupported controls returned correctly
# ---------------------------------------------------------------------------


def test_mock_only_controls_returned_as_unsupported():
    """R6 pin flip (was: success is True). A plan that requested a structured
    control (fields_of_study -> mock_only -> unsupported) but landed ONLY the
    keyword Boolean is a keyword-only fallback, not the structured plan the caller
    asked for. Reporting success=True here was the masquerade — the recruiter would
    believe a field-of-study filter was applied. The keyword still applied and the
    search runs, so it is surfaced via unsupported_controls + the fallback flag, but
    the result is now success=False / plan_fully_applied=False. (fields_of_study is
    the remaining mock_only control after slice H graduated job_titles + companies.)
    """
    browser = _make_browser_with_enter_search(succeed=True)
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML"',
        controls=[
            AdvancedSearchControl(dimension="keywords", values=['"ML"']),
            AdvancedSearchControl(dimension="fields_of_study", values=["Computer Science"]),
        ],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is False
    assert result.plan_fully_applied is False
    assert result.reason == "structured_controls_dropped_keyword_only"
    assert "fields_of_study" in result.unsupported_controls
    assert "keywords" in result.applied_controls


def test_locations_applied_through_plan_when_stable_now():
    # Graduation positive case: a stable_now 'locations' control routes through
    # apply_advanced_search_plan -> _apply_stable_control -> the browser boundary
    # (stubbed) into applied_controls; fields_of_study stays mock_only -> unsupported.
    browser = _make_browser_with_enter_search(succeed=True)
    browser.apply_location_filter = AsyncMock(return_value=True)
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML"',
        controls=[
            AdvancedSearchControl(dimension="keywords", values=['"ML"']),
            AdvancedSearchControl(dimension="locations", values=["NYC"]),
            AdvancedSearchControl(dimension="fields_of_study", values=["Computer Science"]),
        ],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is True
    assert set(result.applied_controls) == {"keywords", "locations"}
    assert "fields_of_study" in result.unsupported_controls
    browser.apply_location_filter.assert_awaited_once()


def test_keyword_only_when_structured_dropped_is_not_reported_as_full_success():
    """R6 (honesty axis): a hybrid plan that requested a structured dimension
    which fell to unsupported, while ONLY the keyword Boolean applied, must not
    report a fully-applied plan. success=False, the dropped control is in
    unsupported_controls, and to_dict carries plan_fully_applied=False so a
    downstream consumer (recovery, audit, the mutation log) can distinguish
    'everything requested applied' from 'keyword landed, structured dropped'.
    """
    browser = _make_browser_with_enter_search(succeed=True)
    plan = AdvancedSearchPlan(
        keyword_boolean='"staff engineer"',
        controls=[
            AdvancedSearchControl(dimension="keywords", values=['"staff engineer"']),
            AdvancedSearchControl(dimension="fields_of_study", values=["Computer Science"]),
        ],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    # Not a fully-applied structured plan.
    assert result.success is False
    assert result.plan_fully_applied is False
    # The requested structured control is visibly dropped, not silently lost.
    assert result.applied_controls == ["keywords"]
    assert "fields_of_study" in result.unsupported_controls
    assert result.reason == "structured_controls_dropped_keyword_only"
    # The honesty axis survives serialization.
    d = result.to_dict()
    assert d["plan_fully_applied"] is False
    assert d["success"] is False
    assert "fields_of_study" in d["unsupported_controls"]


def test_fully_applied_plan_still_reports_success_and_plan_fully_applied():
    """Positive pin: a genuinely fully-applied plan is unchanged by R6. Every
    requested control applied (keyword + a stable_now locations control routed
    through the stubbed browser boundary), nothing dropped -> success=True AND
    plan_fully_applied=True with the original reason. Guards against the honesty
    branch over-firing and regressing the real success path.
    """
    browser = _make_browser_with_enter_search(succeed=True)
    browser.apply_location_filter = AsyncMock(return_value=True)
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML"',
        controls=[
            AdvancedSearchControl(dimension="keywords", values=['"ML"']),
            AdvancedSearchControl(dimension="locations", values=["NYC"]),
        ],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is True
    assert result.plan_fully_applied is True
    assert result.reason == "all_stable_controls_applied"
    assert set(result.applied_controls) == {"keywords", "locations"}
    assert result.unsupported_controls == []
    assert result.to_dict()["plan_fully_applied"] is True


def test_partial_structured_applied_is_success_but_not_fully_applied():
    """Boundary pin between R6's two regimes: when a structured control DID apply
    (locations) alongside a dropped one (fields_of_study -> unsupported), the search
    is usable so success stays True — but plan_fully_applied is False because not
    everything requested landed. This is the case the keyword-only honesty branch
    must NOT swallow (applied is more than just ['keywords']).
    """
    browser = _make_browser_with_enter_search(succeed=True)
    browser.apply_location_filter = AsyncMock(return_value=True)
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML"',
        controls=[
            AdvancedSearchControl(dimension="keywords", values=['"ML"']),
            AdvancedSearchControl(dimension="locations", values=["NYC"]),
            AdvancedSearchControl(dimension="fields_of_study", values=["Computer Science"]),
        ],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is True
    assert result.plan_fully_applied is False
    assert set(result.applied_controls) == {"keywords", "locations"}
    assert result.unsupported_controls == ["fields_of_study"]


def test_mock_only_without_boolean_fails_closed():
    browser = _make_browser_with_enter_search(succeed=True)
    plan = AdvancedSearchPlan(
        controls=[
            AdvancedSearchControl(dimension="fields_of_study", values=["Computer Science"]),
        ],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is False
    assert result.applied_controls == []
    assert result.unsupported_controls == ["fields_of_study"]
    assert result.reason == "no_supported_controls_applied"
    browser.enter_search_string.assert_not_awaited()


def test_defer_controls_returned_as_unsupported():
    """R6 pin flip (was: success is True). Same masquerade as the mock_only case
    above but with defer controls: seniority + years_of_experience both fall to
    unsupported while only the keyword Boolean applies. Keyword-only is not the
    requested structured plan, so success=False / plan_fully_applied=False; the
    dropped dimensions remain visible in unsupported_controls. This is NOT the
    pure no-supported-controls case (test_mock_only_without_boolean_fails_closed)
    — here a keyword landed, so the guard at the no-applied branch does not fire
    and the dedicated keyword-only honesty branch is what flips success.
    """
    browser = _make_browser_with_enter_search(succeed=True)
    plan = AdvancedSearchPlan(
        keyword_boolean='"test"',
        controls=[
            AdvancedSearchControl(dimension="seniority", values=["Director"]),
            AdvancedSearchControl(dimension="years_of_experience", values=["10+"]),
        ],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is False
    assert result.plan_fully_applied is False
    assert result.reason == "structured_controls_dropped_keyword_only"
    assert "seniority" in result.unsupported_controls
    assert "years_of_experience" in result.unsupported_controls


# ---------------------------------------------------------------------------
# Slice E (part 4) — empty-plan phantom-success short-circuit.
# A plan that lands NOTHING (nothing applied, failed, or unsupported) and has no
# keyword Boolean to fall back on must report success=False /
# reason=empty_plan_nothing_applied — not the phantom success=True /
# all_stable_controls_applied it used to. It contradicts the R6 honesty axis the
# same file enforces above.
# ---------------------------------------------------------------------------


def test_empty_plan_is_not_phantom_success():
    """Slice E test (d): a control-less + keyword-less plan returns success=False,
    reason=empty_plan_nothing_applied (was success=True / all_stable_controls_applied).
    No browser boundary is touched — there is no search to run.
    """
    browser = _make_browser_with_enter_search(succeed=True)
    plan = AdvancedSearchPlan(keyword_boolean="", controls=[])
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is False
    assert result.reason == "empty_plan_nothing_applied"
    assert result.plan_fully_applied is False
    assert result.applied_controls == []
    assert result.failed_controls == []
    assert result.unsupported_controls == []
    browser.enter_search_string.assert_not_awaited()
    assert result.to_dict()["success"] is False


def test_boolean_only_plan_still_reports_success_not_empty_plan():
    """Slice E regression (test (e), gate side): a boolean_led keyword-only plan (a
    keyword Boolean, no structured controls) is a correct, usable search — success=True
    / all_stable_controls_applied. The empty-plan short-circuit fires ONLY when there is
    no keyword to fall back on; a keyword-only lane is never empty.
    """
    browser = _make_browser_with_enter_search(succeed=True)
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML" AND "engineer"',
        controls=[
            AdvancedSearchControl(dimension="keywords", values=['"ML" AND "engineer"'])
        ],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is True
    assert result.reason == "all_stable_controls_applied"
    assert result.plan_fully_applied is True
    assert result.applied_controls == ["keywords"]


# ---------------------------------------------------------------------------
# Section-scoped clear: global "Clear search" is never touched
# ---------------------------------------------------------------------------


def test_no_global_clear_search_call():
    """The controller never calls a global 'Clear search' button."""
    browser = _make_browser_with_enter_search(succeed=True)
    plan = AdvancedSearchPlan(
        keyword_boolean='"test"',
        controls=[AdvancedSearchControl(dimension="keywords", values=['"test"'])],
    )
    asyncio.run(apply_advanced_search_plan(browser, plan))
    page_mock = browser._page
    for call in page_mock.method_calls:
        call_str = str(call).lower()
        assert "clear search" not in call_str or "clear keywords" in call_str


# ---------------------------------------------------------------------------
# Browser delegate methods
# ---------------------------------------------------------------------------


def test_browser_apply_advanced_search_plan_delegates():
    browser = _make_browser_with_enter_search(succeed=True)
    plan = AdvancedSearchPlan(
        keyword_boolean='"test"',
        controls=[AdvancedSearchControl(dimension="keywords", values=['"test"'])],
    )
    result = asyncio.run(browser.apply_advanced_search_plan(plan))
    assert result.success is True


def test_browser_snapshot_advanced_search_controls():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.linkedin.com/talent/search")
    browser._page = page
    snap = asyncio.run(browser.snapshot_advanced_search_controls())
    assert "controls" in snap
    assert "keyword_boolean" in snap


def test_enter_search_string_clears_last_search_snapshot():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.linkedin.com/talent/search")
    browser._page = page

    plan = AdvancedSearchPlan(
        keyword_boolean='"test"',
        controls=[AdvancedSearchControl(dimension="keywords", values=['"test"'])],
    )
    browser.enter_search_string = AsyncMock()
    asyncio.run(browser.apply_advanced_search_plan(plan))
    snap_before = asyncio.run(browser.snapshot_advanced_search_controls())
    assert snap_before.get("controls"), "snapshot should be populated after apply"

    browser._last_search_snapshot = {}
    snap_after = asyncio.run(browser.snapshot_advanced_search_controls())
    assert snap_after["controls"] == [], "snapshot should be empty after clearing"


def test_hybrid_mutation_calls_apply_advanced_search_plan(tmp_path):
    from linkedin.search_intelligence import LinkedInExperimentState, LinkedInSearchIntent, LinkedInSearchVariant, LinkedInStructuredFilters
    from linkedin.search_mutation import LinkedInSearchMutationDeps, LinkedInSearchMutationExecutor
    from linkedin.advanced_search import ControlApplicationResult
    from shared.schemas import SearchString

    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    browser.apply_advanced_search_plan = AsyncMock(
        return_value=ControlApplicationResult(
            success=True,
            applied_controls=["keywords"],
            unsupported_controls=["job_titles"],
            reason="all_stable_controls_applied",
        )
    )
    browser.get_results_count_text = AsyncMock(return_value="120")
    browser.get_results_count = AsyncMock(return_value=120)
    browser.get_card_snapshot = AsyncMock(return_value={})

    log_path = tmp_path / "run_log.jsonl"
    deps = LinkedInSearchMutationDeps(
        browser=browser,
        log_path=log_path,
        get_input_mode=lambda: "concurrent",
        get_runtime_run_id=lambda: None,
        get_runtime_state=lambda: MagicMock(),
        get_search_mutation_budget_used=lambda: 0,
        set_search_mutation_budget_used=lambda _v: None,
    )
    executor = LinkedInSearchMutationExecutor(deps)
    experiment_state = LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean="test"),
        mode="experiment",
    )
    variant = LinkedInSearchVariant(
        variant_id="hybrid-1",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer" NOT ("recruiter" OR "sourcer")',
        structured_filters=LinkedInStructuredFilters(titles=["ML Engineer"]),
    )
    experiment_state.variants[variant.variant_id] = variant
    search_string = SearchString(id=1, name="hybrid", boolean=variant.boolean)

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
        result = asyncio.run(
            executor.apply_variant(
                search_string=search_string,
                experiment_state=experiment_state,
                variant=variant,
                acquisition_mode="linkedin_hybrid",
            )
        )

    browser.apply_advanced_search_plan.assert_awaited_once()
    assert result.applied is True
    assert result.hybrid_partial is True
    assert result.structured_apply is not None
    assert "job_titles" in result.structured_apply["unsupported_controls"]
    executed_events = [e for e in read_jsonl(log_path) if e.get("event") == "string_executed"]
    assert len(executed_events) == 1
    assert executed_events[0]["executed_boolean"] == variant.boolean
    assert " NOT " in executed_events[0]["executed_boolean"]
    assert executed_events[0]["execution_surface"] == "advanced"


# ---------------------------------------------------------------------------
# Slice D — structured_only expressibility at the apply_variant boundary.
# A structured_only variant must (1) compile with include_keyword=False, (2)
# NOT fire the post-apply :232 keyword re-entry, and (3) ABANDON rather than run
# an unbounded keyword-less + control-less search when its only structured dims
# are unsupported.
# ---------------------------------------------------------------------------


def _mutation_deps(browser, log_path: Path | None = None):
    from linkedin.search_mutation import LinkedInSearchMutationDeps

    return LinkedInSearchMutationDeps(
        browser=browser,
        log_path=log_path or Path("test.log"),
        get_input_mode=lambda: "concurrent",
        get_runtime_run_id=lambda: None,
        get_runtime_state=lambda: MagicMock(),
        get_search_mutation_budget_used=lambda: 0,
        set_search_mutation_budget_used=lambda _v: None,
    )


def test_structured_only_variant_compiles_without_keyword_and_skips_re_entry():
    """Slice D test (b): a structured_only variant whose structured dim DOES apply
    (locations -> stable_now) compiles with include_keyword=False, lands no keyword
    control, and the :232 fallback re-entry (which reads variant.boolean DIRECTLY)
    does NOT fire — the keyword is dropped end to end.
    """
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
    )
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from linkedin.advanced_search import ControlApplicationResult
    from shared.schemas import SearchString

    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    # Structured control applied; no keyword in applied_controls (it was suppressed
    # at compile, so apply_advanced_search_plan never re-adds it either).
    browser.apply_advanced_search_plan = AsyncMock(
        return_value=ControlApplicationResult(
            success=True,
            applied_controls=["locations"],
            reason="all_stable_controls_applied",
        )
    )
    browser.enter_search_string = AsyncMock()
    browser.get_results_count_text = AsyncMock(return_value="80")
    browser.get_results_count = AsyncMock(return_value=80)
    browser.get_card_snapshot = AsyncMock(return_value={})

    executor = LinkedInSearchMutationExecutor(_mutation_deps(browser))
    experiment_state = LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean="test"),
        mode="experiment",
    )
    variant = LinkedInSearchVariant(
        variant_id="structured-only-1",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer"',
        surface="structured_only",
        structured_filters=LinkedInStructuredFilters(
            sidebar_filters={"locations": ["NYC"]}
        ),
    )
    experiment_state.variants[variant.variant_id] = variant
    search_string = SearchString(id=1, name="structured-only", boolean=variant.boolean)

    captured: dict = {}
    real_compile = __import__(
        "linkedin.advanced_search", fromlist=["compile_structured_filters_to_plan"]
    ).compile_structured_filters_to_plan

    def _spy_compile(structured_filters, **kwargs):
        captured.update(kwargs)
        return real_compile(structured_filters, **kwargs)

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0), patch(
        "linkedin.advanced_search.compile_structured_filters_to_plan", _spy_compile
    ):
        result = asyncio.run(
            executor.apply_variant(
                search_string=search_string,
                experiment_state=experiment_state,
                variant=variant,
                acquisition_mode="linkedin_hybrid",
            )
        )

    # (1) compile saw include_keyword=False for a structured_only surface.
    assert captured.get("include_keyword") is False
    # (2) the :232 keyword re-entry did NOT fire.
    browser.enter_search_string.assert_not_awaited()
    # The variant still applied (the structured control carried it).
    assert result.applied is True
    assert "keywords" not in (result.structured_apply or {}).get("applied_controls", [])


def test_structured_only_variant_abandons_when_all_structured_dims_unsupported():
    """Slice D test (c) — SAFETY: a structured_only variant whose ONLY structured
    dims are unsupported (fields_of_study -> mock_only) must ABANDON. It must NEVER run
    a keyword-less + control-less search (= the whole candidate population). With
    the keyword suppressed at compile, apply_advanced_search_plan returns
    no_supported_controls_applied; apply_variant rejects and never calls
    enter_search_string.
    """
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
    )
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from shared.schemas import SearchString

    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    # Real controller behaviour: delegate apply_advanced_search_plan to the actual
    # function so the abandon routing (no_supported_controls_applied) is exercised
    # end to end, not stubbed. enter_search_string is the unbounded-search canary.
    browser.enter_search_string = AsyncMock()

    async def _real_apply(plan):
        from linkedin.advanced_search import apply_advanced_search_plan

        return await apply_advanced_search_plan(browser, plan)

    browser.apply_advanced_search_plan = AsyncMock(side_effect=_real_apply)
    browser.get_results_count_text = AsyncMock(return_value="999999")
    browser.get_results_count = AsyncMock(return_value=999999)
    browser.get_card_snapshot = AsyncMock(return_value={})

    executor = LinkedInSearchMutationExecutor(_mutation_deps(browser))
    experiment_state = LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean="test"),
        mode="experiment",
    )
    variant = LinkedInSearchVariant(
        variant_id="structured-only-unsupported",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer"',
        surface="structured_only",
        structured_filters=LinkedInStructuredFilters(
            advanced_filters={"fields_of_study": ["Computer Science"]}
        ),
    )
    experiment_state.variants[variant.variant_id] = variant
    search_string = SearchString(id=1, name="structured-only", boolean=variant.boolean)

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
        result = asyncio.run(
            executor.apply_variant(
                search_string=search_string,
                experiment_state=experiment_state,
                variant=variant,
                acquisition_mode="linkedin_hybrid",
            )
        )

    assert result.applied is False
    assert result.blocked_reason == "structured_controls_unsupported_no_boolean_fallback"
    # SAFETY: no keyword-less, control-less search ever ran.
    browser.enter_search_string.assert_not_awaited()
    assert result.structured_apply["reason"] == "no_supported_controls_applied"


@pytest.mark.parametrize("control", ["companies", "skills"])
@pytest.mark.parametrize("facet_lifecycle", [False, True])
def test_rejected_drift_rolls_back_pending_state(tmp_path, monkeypatch, control, facet_lifecycle):
    from linkedin.advanced_search import ControlApplicationResult
    from linkedin.search_intelligence import (
        LinkedInExperimentState, LinkedInSearchIntent, LinkedInSearchVariant, LinkedInStructuredFilters,
    )
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from shared import config
    from shared.schemas import SearchString

    monkeypatch.setattr(config, "LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", facet_lifecycle)
    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    browser.assert_search_surface_ready = AsyncMock()
    browser.reconcile_facets = AsyncMock()
    browser.last_facet_reconciliation = {}
    browser.apply_advanced_search_plan = AsyncMock(return_value=ControlApplicationResult(
        success=False, applied_controls=[], failed_controls=[control if control == "companies" else "keywords"],
        reason="controls rejected", plan_fully_applied=False,
    ))
    executor = LinkedInSearchMutationExecutor(_mutation_deps(browser, tmp_path / "run.jsonl"))
    state = LinkedInExperimentState(
        root_string_id=1, intent=LinkedInSearchIntent(root_boolean="foo"),
        mode="paginate", committed_variant_id="root", pages_since_last_mutation=1,
    )
    filters = LinkedInStructuredFilters(
        companies=["Acme"] if control == "companies" else [],
        skills=["Python"] if control == "skills" else [],
    )
    variant = LinkedInSearchVariant(variant_id="drift-1", parent_variant_id="root", root_string_id=1,
                                    boolean="foo NOT noise", surface="hybrid", structured_filters=filters)
    state.variants[variant.variant_id] = variant
    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
        result = asyncio.run(executor.apply_variant(
            search_string=SearchString(id=1, name="owner", boolean="foo"),
            experiment_state=state, variant=variant, mutation_kind="drift",
            acquisition_mode="linkedin_hybrid",
        ))
    assert result.applied is False
    expected_reason = (
        "structured_controls_unsupported_no_boolean_fallback"
        if control == "companies" else "structured_controls_not_applied"
    )
    assert result.blocked_reason == expected_reason
    assert state.mode == "paginate"
    assert state.active_variant_id == state.committed_variant_id == "root"
    assert state.drift_attempt_count == 0
    assert state.pending_drift_variant_id is None
    assert state.pending_drift_parent_variant_id is None
    assert state.pending_drift_started_at == ""
    assert state.mutations_used == 0


def test_structured_only_variant_with_empty_filters_never_dispatches():
    """Slice D defense-in-depth: a structured_only variant carrying NO structured
    filters must be rejected at apply_variant entry, never dispatched. With the
    keyword dropped and no controls, the non-hybrid path would otherwise reach
    enter_search_string("") — a keyword-less, control-less whole-population search.
    The orchestrator ingestion guards block this today; this pins apply_variant safe
    in isolation (goes red without the entry guard: enter_search_string is awaited).
    """
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
    )
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from shared.schemas import SearchString

    browser = MagicMock()
    browser.enter_search_string = AsyncMock()
    browser.apply_advanced_search_plan = AsyncMock()

    executor = LinkedInSearchMutationExecutor(_mutation_deps(browser))
    experiment_state = LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean="test"),
        mode="experiment",
    )
    variant = LinkedInSearchVariant(
        variant_id="structured-only-empty",
        parent_variant_id="root",
        root_string_id=1,
        boolean="",
        surface="structured_only",
        structured_filters=LinkedInStructuredFilters(),
    )
    experiment_state.variants[variant.variant_id] = variant
    search_string = SearchString(id=1, name="structured-only-empty", boolean="")

    result = asyncio.run(
        executor.apply_variant(
            search_string=search_string,
            experiment_state=experiment_state,
            variant=variant,
            acquisition_mode="linkedin_hybrid",
        )
    )

    assert result.applied is False
    assert result.blocked_reason == "structured_only_without_filters"
    browser.enter_search_string.assert_not_awaited()
    browser.apply_advanced_search_plan.assert_not_awaited()


def test_boolean_variant_still_re_enters_keyword_after_apply(tmp_path):
    """Slice D test (e) regression: a hybrid/boolean variant (surface defaults to
    keyword-led; include_keyword stays True) still gets the keyword control AND the
    :232 re-entry exactly as before. Here the structured control applied but the
    keyword did not land via the plan, so :232 must re-enter variant.boolean.
    """
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
    )
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from linkedin.advanced_search import ControlApplicationResult
    from linkedin.browser import SearchEntryResult
    from linkedin.input_backends import TypingResult
    from shared.schemas import SearchString

    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    # Structured control landed; keyword NOT in applied_controls -> :232 must fire.
    browser.apply_advanced_search_plan = AsyncMock(
        return_value=ControlApplicationResult(
            success=True,
            applied_controls=["locations"],
            reason="all_stable_controls_applied",
        )
    )
    # Real SearchEntryResult so the downstream mutation log serializes (the live
    # browser returns this; a bare MagicMock typing_result is not JSON-encodable).
    browser.enter_search_string = AsyncMock(
        return_value=SearchEntryResult(
            typing_result=TypingResult(
                transport="keyboard",
                duration_ms=0,
                typo_count=0,
                used_correction=False,
            ),
            results_wait_ms=0,
        )
    )
    browser.get_results_count_text = AsyncMock(return_value="80")
    browser.get_results_count = AsyncMock(return_value=80)
    browser.get_card_snapshot = AsyncMock(return_value={})

    log_path = tmp_path / "run_log.jsonl"
    executor = LinkedInSearchMutationExecutor(_mutation_deps(browser, log_path))
    experiment_state = LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean="test"),
        mode="experiment",
    )
    variant = LinkedInSearchVariant(
        variant_id="hybrid-keyword-led",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer" NOT ("recruiter" OR "sourcer")',
        # surface left default ("") -> keyword-led; include_keyword stays True.
        structured_filters=LinkedInStructuredFilters(
            sidebar_filters={"locations": ["NYC"]}
        ),
    )
    experiment_state.variants[variant.variant_id] = variant
    search_string = SearchString(id=1, name="hybrid", boolean=variant.boolean)

    captured: dict = {}
    real_compile = __import__(
        "linkedin.advanced_search", fromlist=["compile_structured_filters_to_plan"]
    ).compile_structured_filters_to_plan

    def _spy_compile(structured_filters, **kwargs):
        captured.update(kwargs)
        return real_compile(structured_filters, **kwargs)

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0), patch(
        "linkedin.advanced_search.compile_structured_filters_to_plan", _spy_compile
    ):
        result = asyncio.run(
            executor.apply_variant(
                search_string=search_string,
                experiment_state=experiment_state,
                variant=variant,
                acquisition_mode="linkedin_hybrid",
            )
        )

    # include_keyword stays True for a keyword-led surface.
    assert captured.get("include_keyword") is True
    # The :232 re-entry fires exactly as before — keyword re-entered after apply.
    browser.enter_search_string.assert_awaited_once_with(variant.boolean)
    assert result.applied is True
    executed_events = [e for e in read_jsonl(log_path) if e.get("event") == "string_executed"]
    assert [
        (event["execution_surface"], event["executed_boolean"])
        for event in executed_events
    ] == [
        ("advanced", variant.boolean),
        ("keyword", variant.boolean),
    ]
    assert all(" NOT " in event["executed_boolean"] for event in executed_events)


def test_keyword_only_mutation_logs_executed_boolean_with_not_tail(tmp_path):
    """A keyword-only mutation records the exact submitted boolean after entry."""
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
    )
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from linkedin.browser import SearchEntryResult
    from linkedin.input_backends import TypingResult
    from shared.schemas import SearchString

    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    browser.apply_advanced_search_plan = AsyncMock()
    browser.enter_search_string = AsyncMock(
        return_value=SearchEntryResult(
            typing_result=TypingResult(
                transport="keyboard",
                duration_ms=0,
                typo_count=0,
                used_correction=False,
            ),
            results_wait_ms=0,
        )
    )
    browser.get_results_count_text = AsyncMock(return_value="80")
    browser.get_results_count = AsyncMock(return_value=80)
    browser.get_card_snapshot = AsyncMock(return_value={})

    log_path = tmp_path / "run_log.jsonl"
    executor = LinkedInSearchMutationExecutor(_mutation_deps(browser, log_path))
    experiment_state = LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean="test"),
        mode="experiment",
    )
    variant = LinkedInSearchVariant(
        variant_id="keyword-only-not-tail",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer" NOT ("recruiter" OR "sourcer")',
    )
    experiment_state.variants[variant.variant_id] = variant
    search_string = SearchString(id=1, name="keyword-only", boolean=variant.boolean)

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
        result = asyncio.run(
            executor.apply_variant(
                search_string=search_string,
                experiment_state=experiment_state,
                variant=variant,
            )
        )

    assert result.applied is True
    browser.apply_advanced_search_plan.assert_not_awaited()
    browser.enter_search_string.assert_awaited_once_with(variant.boolean)
    executed_events = [e for e in read_jsonl(log_path) if e.get("event") == "string_executed"]
    assert len(executed_events) == 1
    assert executed_events[0]["executed_boolean"] == variant.boolean
    assert " NOT " in executed_events[0]["executed_boolean"]
    assert executed_events[0]["execution_surface"] == "keyword"


# ---------------------------------------------------------------------------
# Slice E — posture-aware fallback + honesty at apply_variant.
#
# When a structured control DROPS at apply, the response depends on the lane
# surface:
#   - boolean / filter_led / hybrid -> DEMOTE-AND-PROCEED: the keyword (or a
#     partial structured set) carries the search; applied=True, hybrid_partial=True,
#     a linkedin_structured_demotion event is emitted consuming the gate reason, and
#     the dropped dim is CLEARED off variant.structured_filters (so the next probe and
#     the recovery snapshot agree on what landed).
#   - structured_only -> ABANDON (no keyword fallback) — covered by the slice-D
#     all-unsupported abandon above; slice E pins that it NEVER silently demotes here.
# ---------------------------------------------------------------------------


def _event_capturing_deps(browser, captured_events: list):
    """Mutation deps whose runtime_state is a spy that captures recorded events.

    Unlike _mutation_deps (run_id None -> _record_event early-returns), this returns a
    truthy run_id and a runtime_state stub exposing get_work_unit_id + record_event, so
    the linkedin_structured_demotion event is observable without a live runtime DB.
    """
    from linkedin.search_mutation import LinkedInSearchMutationDeps

    runtime_state = MagicMock()
    runtime_state.get_work_unit_id = MagicMock(return_value=42)

    def _record_event(*, run_id, work_unit_id, event_type, payload):
        captured_events.append({"event_type": event_type, "payload": payload})

    runtime_state.record_event = MagicMock(side_effect=_record_event)
    return LinkedInSearchMutationDeps(
        browser=browser,
        log_path=Path("test.log"),
        get_input_mode=lambda: "concurrent",
        get_runtime_run_id=lambda: 7,
        get_runtime_state=lambda: runtime_state,
        get_search_mutation_budget_used=lambda: 0,
        set_search_mutation_budget_used=lambda _v: None,
    )


def test_hybrid_unsupported_dim_demotes_and_proceeds_clears_dim_and_emits_event():
    """Slice E test (a): a HYBRID lane whose only structured dim is unsupported
    (fields_of_study -> mock_only) but whose keyword lands -> DEMOTE-AND-PROCEED.

    applied=True, hybrid_partial=True; a linkedin_structured_demotion event is emitted
    consuming the gate reason (structured_controls_dropped_keyword_only); the dropped
    dim is cleared off variant.structured_filters; and the circuit-breaker counter
    ticks. Drives the REAL controller (apply_advanced_search_plan delegates to the live
    function) so the gate reason + unsupported classification are end-to-end real.
    """
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
    )
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from linkedin.browser import SearchEntryResult
    from linkedin.input_backends import TypingResult
    from shared.schemas import SearchString

    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()

    async def _real_apply(plan):
        from linkedin.advanced_search import apply_advanced_search_plan

        return await apply_advanced_search_plan(browser, plan)

    browser.apply_advanced_search_plan = AsyncMock(side_effect=_real_apply)
    # The keyword landed via the plan's keyword re-add inside the controller.
    browser.enter_search_string = AsyncMock(
        return_value=SearchEntryResult(
            typing_result=TypingResult(
                transport="keyboard", duration_ms=0, typo_count=0, used_correction=False
            ),
            results_wait_ms=0,
        )
    )
    browser.get_results_count_text = AsyncMock(return_value="180")
    browser.get_results_count = AsyncMock(return_value=180)
    browser.get_card_snapshot = AsyncMock(return_value={})

    captured_events: list = []
    executor = LinkedInSearchMutationExecutor(
        _event_capturing_deps(browser, captured_events)
    )
    experiment_state = LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean="test"),
        mode="experiment",
    )
    variant = LinkedInSearchVariant(
        variant_id="hybrid-demote-1",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer"',
        surface="hybrid",
        structured_filters=LinkedInStructuredFilters(
            advanced_filters={"fields_of_study": ["Computer Science"]}
        ),
    )
    experiment_state.variants[variant.variant_id] = variant
    search_string = SearchString(id=1, name="hybrid", boolean=variant.boolean)

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
        result = asyncio.run(
            executor.apply_variant(
                search_string=search_string,
                experiment_state=experiment_state,
                variant=variant,
                acquisition_mode="linkedin_hybrid",
            )
        )

    # Demote-and-proceed: the keyword carried the search.
    assert result.applied is True
    assert result.hybrid_partial is True
    # The gate classified the dropped field-of-study dim as unsupported (fields_of_study -> mock_only).
    assert "fields_of_study" in result.structured_apply["unsupported_controls"]
    assert result.structured_apply["reason"] == "structured_controls_dropped_keyword_only"
    # The dropped dim was CLEARED off the variant so the next probe + recovery snapshot agree.
    assert variant.structured_filters.advanced_filters.get("fields_of_study", []) == []
    assert variant.structured_filters.is_empty()
    # A demotion event was emitted, consuming the gate reason + counter.
    demotion_events = [
        e for e in captured_events if e["event_type"] == "linkedin_structured_demotion"
    ]
    assert len(demotion_events) == 1
    payload = demotion_events[0]["payload"]
    assert payload["surface"] == "hybrid"
    assert payload["reason"] == "structured_controls_dropped_keyword_only"
    assert "fields_of_study" in payload["dropped_dimensions"]
    # Circuit-breaker counter ticked.
    assert experiment_state.structured_demotions == 1


def test_hybrid_failed_dim_reports_partial_not_full_success():
    """Slice E audit #2 (part 3): a HYBRID lane whose structured control FAILED at
    verification (not merely unsupported) but whose keyword landed is a PARTIAL —
    applied=True AND hybrid_partial=True, NOT a masqueraded full success.

    Pre-E, hybrid_partial only tracked unsupported_controls; a verification FAILURE
    that left the keyword running was silently reported full (hybrid_partial=False).
    Goes red if a failed structured dim on a hybrid lane reports full success.
    """
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
    )
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from linkedin.advanced_search import ControlApplicationResult
    from linkedin.browser import SearchEntryResult
    from linkedin.input_backends import TypingResult
    from shared.schemas import SearchString

    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    # The gate's stable-control-failed branch: keyword landed, locations FAILED.
    browser.apply_advanced_search_plan = AsyncMock(
        return_value=ControlApplicationResult(
            success=False,
            applied_controls=["keywords"],
            failed_controls=["locations"],
            unsupported_controls=[],
            fallback_to_boolean=True,
            reason="stable_now_control_failed_fallback_to_boolean",
            plan_fully_applied=False,
        )
    )
    browser.enter_search_string = AsyncMock(
        return_value=SearchEntryResult(
            typing_result=TypingResult(
                transport="keyboard", duration_ms=0, typo_count=0, used_correction=False
            ),
            results_wait_ms=0,
        )
    )
    browser.get_results_count_text = AsyncMock(return_value="180")
    browser.get_results_count = AsyncMock(return_value=180)
    browser.get_card_snapshot = AsyncMock(return_value={})

    captured_events: list = []
    executor = LinkedInSearchMutationExecutor(
        _event_capturing_deps(browser, captured_events)
    )
    experiment_state = LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean="test"),
        mode="experiment",
    )
    variant = LinkedInSearchVariant(
        variant_id="hybrid-failed-1",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer"',
        surface="hybrid",
        structured_filters=LinkedInStructuredFilters(
            sidebar_filters={"locations": ["NYC"]}
        ),
    )
    experiment_state.variants[variant.variant_id] = variant
    search_string = SearchString(id=1, name="hybrid", boolean=variant.boolean)

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
        result = asyncio.run(
            executor.apply_variant(
                search_string=search_string,
                experiment_state=experiment_state,
                variant=variant,
                acquisition_mode="linkedin_hybrid",
            )
        )

    assert result.applied is True
    assert result.hybrid_partial is True, (
        "a failed structured dim on a hybrid lane is a partial, not a full success"
    )
    # Demoted + cleared the failed dim; counter ticked; event consumes the FAILED reason.
    assert variant.structured_filters.sidebar_filters.get("locations") is None
    assert experiment_state.structured_demotions == 1
    demotion_events = [
        e for e in captured_events if e["event_type"] == "linkedin_structured_demotion"
    ]
    assert len(demotion_events) == 1
    assert demotion_events[0]["payload"]["reason"] == "stable_now_control_failed_fallback_to_boolean"


def test_structured_only_failed_dim_abandons_without_keyword_re_entry():
    """Slice E test (b): a structured_only lane whose structured control FAILS must
    ABANDON — no keyword re-entry, NO silent demotion to keyword. With the keyword
    suppressed at compile and the only structured dim failing, applied_controls is
    empty, so apply_variant rejects (structured_controls_not_applied) and never calls
    enter_search_string. The circuit-breaker counter does NOT tick (a structured_only
    abandon is not a keyword demotion).
    """
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
    )
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from linkedin.advanced_search import ControlApplicationResult
    from shared.schemas import SearchString

    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    # locations is stable_now, so a structured_only plan with only locations and the
    # keyword suppressed FAILS the location apply -> failed_controls=["locations"],
    # nothing applied. enter_search_string is the keyword-demotion canary.
    browser.enter_search_string = AsyncMock()
    browser.apply_advanced_search_plan = AsyncMock(
        return_value=ControlApplicationResult(
            success=False,
            applied_controls=[],
            failed_controls=["locations"],
            unsupported_controls=[],
            fallback_to_boolean=True,
            reason="stable_now_control_failed_fallback_to_boolean",
            plan_fully_applied=False,
        )
    )
    browser.get_results_count_text = AsyncMock(return_value="999999")
    browser.get_results_count = AsyncMock(return_value=999999)
    browser.get_card_snapshot = AsyncMock(return_value={})

    captured_events: list = []
    executor = LinkedInSearchMutationExecutor(
        _event_capturing_deps(browser, captured_events)
    )
    experiment_state = LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean="test"),
        mode="experiment",
    )
    variant = LinkedInSearchVariant(
        variant_id="structured-only-failed",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer"',
        surface="structured_only",
        structured_filters=LinkedInStructuredFilters(
            sidebar_filters={"locations": ["NYC"]}
        ),
    )
    experiment_state.variants[variant.variant_id] = variant
    search_string = SearchString(id=1, name="structured-only", boolean=variant.boolean)

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
        result = asyncio.run(
            executor.apply_variant(
                search_string=search_string,
                experiment_state=experiment_state,
                variant=variant,
                acquisition_mode="linkedin_hybrid",
            )
        )

    assert result.applied is False
    # Abandons via one of the two no-applied guards (the first, D-era guard fires on
    # nothing-applied regardless of unsupported-vs-failed). Either is an ABANDON — the
    # load-bearing slice-E invariant is that it NEVER falls back to keyword.
    assert result.blocked_reason in {
        "structured_controls_not_applied",
        "structured_controls_unsupported_no_boolean_fallback",
    }
    # SAFETY: a structured_only lane NEVER demotes to keyword.
    browser.enter_search_string.assert_not_awaited()
    # No demotion was recorded — an abandon is not a keyword demotion.
    assert experiment_state.structured_demotions == 0
    assert not [
        e for e in captured_events if e["event_type"] == "linkedin_structured_demotion"
    ]


def test_partial_demote_strips_dropped_dim_from_checkpoint_so_resume_does_not_reseed():
    """(gap, MEDIUM) A PARTIAL demote (one of N structured dims dropped, surface stays
    hybrid, surviving filters NON-empty) must strip the dropped dim from the
    checkpointed search_string.structured_filters in lockstep with the variant — not
    just the variant. apply_shadow's full-demote checkpoint-clear
    (search_intelligence.py:555) only fires when the surviving filter set is empty, and
    is_deliberate_boolean_demotion needs surface=='boolean', so a partial demote reaches
    NEITHER. Left alone, a CROSS-PROCESS resume's bootstrap_experiment_state re-seeds the
    dropped dim onto a fresh legacy variant from the stale checkpoint — re-applying a
    control the sidebar already rejected.

    A title+location hybrid variant whose LOCATION drops (keyword + title land) must,
    after the demote: (1) clear locations off the variant, keep titles [same-process,
    pre-existing]; (2) strip locations from search_string.structured_filters, keep
    titles [the gap fix]; (3) on bootstrap_experiment_state(search_string) re-seed the
    SURVIVING title onto the resumed active variant but NOT the dropped location.

    Goes red if the checkpoint retains the dropped dim and the resume re-seeds it.
    """
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
        bootstrap_experiment_state,
    )
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from linkedin.advanced_search import ControlApplicationResult
    from linkedin.browser import SearchEntryResult
    from linkedin.input_backends import TypingResult
    from shared.schemas import SearchString

    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    # keyword + title landed; the location dim FAILED at verification. Two non-keyword
    # dims were requested (job_titles, locations); only locations dropped -> PARTIAL.
    browser.apply_advanced_search_plan = AsyncMock(
        return_value=ControlApplicationResult(
            success=False,
            applied_controls=["keywords", "job_titles"],
            failed_controls=["locations"],
            unsupported_controls=[],
            fallback_to_boolean=True,
            reason="stable_now_control_failed_fallback_to_boolean",
            plan_fully_applied=False,
        )
    )
    browser.enter_search_string = AsyncMock(
        return_value=SearchEntryResult(
            typing_result=TypingResult(
                transport="keyboard", duration_ms=0, typo_count=0, used_correction=False
            ),
            results_wait_ms=0,
        )
    )
    browser.get_results_count_text = AsyncMock(return_value="180")
    browser.get_results_count = AsyncMock(return_value=180)
    browser.get_card_snapshot = AsyncMock(return_value={})

    captured_events: list = []
    executor = LinkedInSearchMutationExecutor(
        _event_capturing_deps(browser, captured_events)
    )
    experiment_state = LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean="test"),
        mode="experiment",
    )
    variant = LinkedInSearchVariant(
        variant_id="hybrid-partial-demote",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer"',
        surface="hybrid",
        structured_filters=LinkedInStructuredFilters(
            titles=["Staff Engineer"],
            sidebar_filters={"locations": ["NYC"]},
        ),
    )
    experiment_state.variants[variant.variant_id] = variant
    experiment_state.active_variant_id = variant.variant_id
    # Slice A's producer path checkpoints the lane's compiled structured filters onto
    # the SearchString. Seed it with the FULL original set so the strip is observable.
    search_string = SearchString(
        id=1,
        name="hybrid",
        boolean=variant.boolean,
        structured_filters={
            "titles": ["Staff Engineer"],
            "companies": [],
            "skills": [],
            "assessments": [],
            "sidebar_filters": {"locations": ["NYC"]},
            "advanced_filters": {},
        },
    )

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
        result = asyncio.run(
            executor.apply_variant(
                search_string=search_string,
                experiment_state=experiment_state,
                variant=variant,
                acquisition_mode="linkedin_hybrid",
            )
        )

    # PARTIAL: keyword + title carried; location dropped.
    assert result.applied is True
    assert result.hybrid_partial is True
    assert experiment_state.structured_demotions == 1

    # (1) same-process: variant keeps the surviving title, drops the location.
    assert variant.structured_filters.titles == ["Staff Engineer"]
    assert variant.structured_filters.sidebar_filters.get("locations") is None

    # (2) the gap fix: the CHECKPOINT strips the dropped location but keeps the title.
    assert search_string.structured_filters["titles"] == ["Staff Engineer"]
    assert "locations" not in search_string.structured_filters.get("sidebar_filters", {})

    # (3) cross-process: a fresh resume from this checkpoint re-seeds ONLY the surviving
    # title onto the active variant — never the dropped location.
    resumed = bootstrap_experiment_state(search_string)
    resumed_active = resumed.active_variant
    assert resumed_active.structured_filters.titles == ["Staff Engineer"]
    assert resumed_active.structured_filters.sidebar_filters.get("locations") is None


def test_fully_applied_hybrid_reports_success_and_not_partial():
    """Slice E regression (test (e), apply side): a HYBRID lane whose structured
    control AND keyword both land is a full success — applied=True, hybrid_partial
    stays False, the filter is NOT cleared, and NO demotion is recorded. Guards
    against the demote bookkeeping firing on a clean apply.
    """
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
    )
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from linkedin.advanced_search import ControlApplicationResult
    from shared.schemas import SearchString

    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    browser.apply_advanced_search_plan = AsyncMock(
        return_value=ControlApplicationResult(
            success=True,
            applied_controls=["keywords", "locations"],
            failed_controls=[],
            unsupported_controls=[],
            reason="all_stable_controls_applied",
            plan_fully_applied=True,
        )
    )
    browser.enter_search_string = AsyncMock()
    browser.get_results_count_text = AsyncMock(return_value="120")
    browser.get_results_count = AsyncMock(return_value=120)
    browser.get_card_snapshot = AsyncMock(return_value={})

    captured_events: list = []
    executor = LinkedInSearchMutationExecutor(
        _event_capturing_deps(browser, captured_events)
    )
    experiment_state = LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean="test"),
        mode="experiment",
    )
    variant = LinkedInSearchVariant(
        variant_id="hybrid-clean",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer"',
        surface="hybrid",
        structured_filters=LinkedInStructuredFilters(
            sidebar_filters={"locations": ["NYC"]}
        ),
    )
    experiment_state.variants[variant.variant_id] = variant
    search_string = SearchString(id=1, name="hybrid", boolean=variant.boolean)

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
        result = asyncio.run(
            executor.apply_variant(
                search_string=search_string,
                experiment_state=experiment_state,
                variant=variant,
                acquisition_mode="linkedin_hybrid",
            )
        )

    assert result.applied is True
    assert result.hybrid_partial is False
    # The applied filter is preserved — nothing dropped, nothing cleared.
    assert variant.structured_filters.sidebar_filters.get("locations") == ["NYC"]
    assert experiment_state.structured_demotions == 0
    assert not [
        e for e in captured_events if e["event_type"] == "linkedin_structured_demotion"
    ]


# ---------------------------------------------------------------------------
# Save-trigger resolution: semantic fallback survives a class rotation
# (FIX 3 — bare class with no semantic fallback misattributes a selector-miss
#  as a clicked-but-did-not-persist failure). Offline; uses Playwright
#  a synthetic data-URL DOM. Each test owns one event loop and
#  one Playwright connection (setup + assert + teardown inside a single
#  asyncio.run) so chromium children are reaped deterministically.
# ---------------------------------------------------------------------------


# DOM-verified: when a candidate is unsaved, the trigger's accessible name is
# "Save to '{stage}'". In _ROTATED_CLASS the bare ``save-to-pipeline__button``
# class is ABSENT (simulating a class rotation) but the accessible name is present.
# Every fixture includes div.profile__main-container so save_candidate's
# scroll-recovery branch (container.evaluate(...)) resolves instantly instead
# of blocking on Playwright's default action timeout for a missing element.
_SLIDEIN_ROTATED_CLASS = """
<div class="profile__main-container">
  <div class="profile-slidein__container">
    <button type="button" aria-label="Save to 'uncontacted'">Save to 'uncontacted'</button>
  </div>
</div>
"""

_SLIDEIN_HAPPY_PATH = """
<div class="profile__main-container">
  <div class="profile-slidein__container">
    <button type="button" class="save-to-pipeline__button">Save to 'uncontacted'</button>
  </div>
</div>
"""

_SLIDEIN_NO_TRIGGER = """
<div class="profile__main-container">
  <div class="profile-slidein__container">
    <button type="button" class="unrelated-button">Message</button>
  </div>
</div>
<ol class="profile-list">
  <li>
    <article class="profile-list-item">
      <div class="lockup__title">
        <a href="/talent/profile/candidate-1">Candidate</a>
      </div>
    </article>
  </li>
</ol>
"""


def _run_with_offline_page(html: str, body):
    """Run ``body(browser, page)`` against a synthetic offline chromium page.

    Setup, the coroutine body, and teardown all run inside one event loop and
    one Playwright connection. Missing browser transport fails the test.
    """
    from rebrowser_playwright.async_api import async_playwright

    async def _main():
        pw = await async_playwright().start()
        try:
            chromium = await asyncio.wait_for(
                pw.chromium.launch(
                    headless=True,
                    args=["--single-process", "--no-zygote"],
                ),
                timeout=30,
            )
            try:
                page = await chromium.new_page()
                encoded_html = base64.b64encode(html.encode()).decode()
                await page.goto(
                    f"data:text/html;base64,{encoded_html}",
                    wait_until="domcontentloaded",
                )
                browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
                browser._page = page
                return await body(browser, page)
            finally:
                await chromium.close()
        finally:
            await pw.stop()

    return asyncio.run(asyncio.wait_for(_main(), timeout=60))


def test_resolve_save_trigger_uses_semantic_fallback_on_class_rotation():
    """Bare class absent, accessible name present -> fallback locates the trigger.

    This is the seam: on current (pre-fix) code, _resolve_save_trigger does not
    exist / the only selector is the bare class, so resolution fails. After the
    fix, the "Save to '{stage}'" fallback resolves the button.
    """

    async def _body(browser, page):
        # The primary class selector must genuinely miss this DOM.
        assert await page.locator(LinkedInBrowser._SAVE_TRIGGER_PRIMARY).count() == 0
        located = await browser._resolve_save_trigger(timeout=2000)
        assert located is not None, "semantic fallback should locate the save trigger"
        assert "Save to '" in (await located.inner_text())

    _run_with_offline_page(_SLIDEIN_ROTATED_CLASS, _body)


def test_resolve_save_trigger_prefers_primary_class_when_present():
    """Happy path is unchanged: primary class selector resolves the trigger."""

    async def _body(browser, page):
        located = await browser._resolve_save_trigger(timeout=2000)
        assert located is not None
        cls = await located.get_attribute("class")
        assert cls is not None and "save-to-pipeline__button" in cls

    _run_with_offline_page(_SLIDEIN_HAPPY_PATH, _body)


def test_resolve_save_trigger_returns_none_when_no_trigger():
    """Neither primary nor fallback present -> None (distinct failure signal)."""

    async def _body(browser, page):
        assert await browser._resolve_save_trigger(timeout=1000) is None

    _run_with_offline_page(_SLIDEIN_NO_TRIGGER, _body)


def test_save_candidate_distinguishes_trigger_not_found_from_non_persist():
    """A class rotation surfaces 'save_trigger_not_found', NOT a non-persist.

    save_candidate returns False in both failure modes, but
    _last_save_failure_reason must distinguish them so callers stop logging a
    selector-miss as 'save did not persist after click'.
    """

    async def _body(browser, page):
        browser.current_profile_identity_fragment = MagicMock(
            return_value="candidate-1"
        )
        # Keep the REAL DOM resolution, but short timeouts + no retry sleep so
        # the (correctly-failing) save path returns quickly instead of waiting
        # out three 5s locator timeouts.
        real_resolve = browser._resolve_save_trigger

        async def _fast_resolve(*, timeout: int = 5000):
            return await real_resolve(timeout=min(timeout, 300))

        browser._resolve_save_trigger = _fast_resolve
        # _retry sleeps RETRY_DELAY_SECONDS between attempts; run it with
        # delay=0 so the three (correct) failures don't add ~15s of real wall
        # time. Patching asyncio.sleep globally breaks Playwright's own polling,
        # so wrap _retry to force zero delay instead.
        import linkedin.browser as _lb

        real_retry = _lb._retry

        async def _retry_no_delay(coro_fn, retries=3, delay=0):
            return await real_retry(coro_fn, retries=retries, delay=0)

        with patch.object(_lb, "_retry", _retry_no_delay):
            ok = await browser.save_candidate()
        assert ok is False
        assert browser._last_save_failure_reason == "save_trigger_not_found"

    _run_with_offline_page(_SLIDEIN_NO_TRIGGER, _body)


def test_ghost_click_locator_returns_true_after_playwright_fallback():
    """Regression (Cursor review of FIX 3): when ghost-cursor fails, the
    wrapper performs a real pointer click and MUST report True. Returning
    False makes save_candidate's `if not clicked: JS-evaluate click` fallback
    fire a SECOND click on a live save-to-pipeline control — a double-click on
    a mutating action. Backend click_locator returns False ("did not click");
    the wrapper's fallback did click, so the wrapper returns True.

    The fallback positions with hover() and presses with mouse.down/up rather
    than locator.click(), so the commit guard is the last thing to run before
    the press (see _ghost_click_locator's docstring).
    """
    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        page = MagicMock()
        page.mouse.down = AsyncMock()
        page.mouse.up = AsyncMock()
        browser._page = page
        browser._input_backend.click_locator = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.hover = AsyncMock()
        locator.click = AsyncMock()
        result = await browser._ghost_click_locator(locator)
        assert result is True
        locator.hover.assert_awaited_once()
        page.mouse.down.assert_awaited_once()
        page.mouse.up.assert_awaited_once()
        locator.click.assert_not_awaited()

    asyncio.run(_body())


def test_ghost_click_locator_no_fallback_click_when_ghost_succeeds():
    """When ghost-cursor clicks, the Playwright fallback must NOT fire."""
    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser._page = MagicMock()
        browser._input_backend.click_locator = AsyncMock(return_value=True)
        locator = MagicMock()
        locator.click = AsyncMock()
        result = await browser._ghost_click_locator(locator)
        assert result is True
        locator.click.assert_not_awaited()

    asyncio.run(_body())


# ---------------------------------------------------------------------------
# R5 — partial apply must NOT be reported as full success
#
# apply_location_filter / apply_company_filter (byte-identical siblings) used to
# fail closed ONLY on zero applied: with k of N values landing they returned a
# bare True, so the controller recorded the dimension as applied and persisted
# the FULL requested set to the recovery snapshot — a partial geography replayed
# as if fully applied (over-broad results). Fix: fail closed unless EVERY
# requested value produced a confirmed chip.
#
# Offline DOM mirrors the Pass-5 capture: pane `aside.left-rail`; editor input is
# the typeahead combobox whose placeholder contains the facet word; options are
# `li[role=option]` in `ul.artdeco-typeahead__results-list[role=listbox]`; the
# applied chip is `[data-test-pill-label]`. The editor input is pre-rendered
# visible (reveal branch skipped). The "applies" value has an EXACT-match option
# + chip; the "no exact match" value has only a broad suggestion and no chip, so
# its per-value `_do` returns False and it never lands -> partial. Live-page
# helpers (baseline peek / results-ready / humanized typing) are stubbed so the
# test isolates the DOM-resolution + per-value chip gate + fail-closed arithmetic.
# ---------------------------------------------------------------------------

_LOCATIONS_PARTIAL = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-locations">
  <input class="artdeco-typeahead__input" placeholder="Add a geographic location" />
  <ul class="artdeco-typeahead__results-list" role="listbox">
    <li role="option">New York</li>
    <li role="option">Berlin, Germany Area</li>
  </ul>
  <figure class="pills-list-section" data-test-can-have-pills-list-section>
    <ul class="pills-list-section__list">
      <li class="pills-list-section__item" data-test-facet-pills-item>
        <span class="facet-pill">New York<button class="facet-pill__action" aria-label="Remove New York" title="Remove New York" data-test-pill-dismiss data-view-name="search-facet-remove" type="button"></button></span>
      </li>
    </ul>
  </figure>
  </div>
</aside>
"""

_COMPANIES_PARTIAL = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-companies">
  <input class="artdeco-typeahead__input" placeholder="Add a company or boolean" />
  <ul class="artdeep artdeco-typeahead__results-list" role="listbox">
    <li role="option">Acme Inc</li>
    <li role="option">Globex Corporation Worldwide</li>
  </ul>
  <figure class="pills-list-section" data-test-can-have-pills-list-section>
    <ul class="pills-list-section__list">
      <li class="pills-list-section__item" data-test-facet-pills-item>
        <span class="facet-pill">Acme Inc<button class="facet-pill__action" aria-label="Remove Acme Inc" title="Remove Acme Inc" data-test-pill-dismiss data-view-name="search-facet-remove" type="button"></button></span>
      </li>
    </ul>
  </figure>
  </div>
</aside>
"""

_COMPANIES_EXISTING_PILLS_REVEAL = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-companies">
  <section>
    <h2>Companies</h2>
    <button
      type="button"
      onclick="document.querySelector('#company-editor').style.display = 'block'"
    >Companies or boolean</button>
    <ul class="pills-list-section__list">
      <li class="pills-list-section__item" data-test-facet-pills-item><span class="facet-pill">CohnReznick<button class="facet-pill__action" aria-label="Remove CohnReznick" data-test-pill-dismiss type="button"></button></span></li>
      <li class="pills-list-section__item" data-test-facet-pills-item><span class="facet-pill">The D. E. Shaw Group<button class="facet-pill__action" aria-label="Remove The D. E. Shaw Group" data-test-pill-dismiss type="button"></button></span></li>
    </ul>
  </section>
  <input
    id="company-editor"
    class="artdeco-typeahead__input"
    placeholder="Add a company or boolean"
    style="display: none"
  />
  <ul class="artdeco-typeahead__results-list" role="listbox">
    <li role="option" onclick="document.querySelector('#stripe-pill').style.display = 'flex'">Stripe</li>
  </ul>
  <figure class="pills-list-section" data-test-can-have-pills-list-section>
    <ul class="pills-list-section__list">
      <li class="pills-list-section__item" data-test-facet-pills-item id="stripe-pill" style="display: none">
        <span class="facet-pill">Stripe<button class="facet-pill__action" aria-label="Remove Stripe" data-test-pill-dismiss type="button"></button></span>
      </li>
    </ul>
  </figure>
  </div>
</aside>
"""


def _stub_live_page_helpers(browser):
    """Stub the parts of apply_* that need a live Recruiter page, leaving the
    DOM-resolution / chip-gate / fail-closed logic real.

    type_text FILLS the target (not a no-op): the apply loop now verifies the
    typed text actually landed before polling options (2026-08-14 slow-restore
    hardening), so a no-op stub would read as swallowed keystrokes and spin the
    budget loop. The clear stub mirrors that with a real fill("") — under the
    old no-op typing the inputs were always empty, so the keyboard-driven clear
    was already an effective no-op in these tests."""
    browser.go_back_to_results = AsyncMock()
    browser.require_recruiter_tab = AsyncMock()
    browser._peek_results_count_text = AsyncMock(return_value="120")
    browser._peek_top_card_signature = AsyncMock(return_value=("", ""))
    browser._wait_for_search_results_ready = AsyncMock()
    browser.read_applied_location_chips = AsyncMock(return_value=[])
    browser.read_applied_company_chips = AsyncMock(return_value=[])
    browser.read_applied_title_chips = AsyncMock(return_value=[])

    async def _fill_typed_text(_page, locator, value, plan=None):
        try:
            await locator.fill(value)
        except Exception:
            pass

    async def _fill_cleared(textarea):
        try:
            await textarea.fill("")
        except Exception:
            pass

    browser._input_backend.type_text = AsyncMock(side_effect=_fill_typed_text)
    browser._clear_keyword_textarea = AsyncMock(side_effect=_fill_cleared)


def test_apply_location_filter_partial_apply_does_not_report_full_success():
    """2 locations requested; 'New York' has an exact-match option + chip,
    'Berlin' has no exact-match option -> only 1 of 2 lands. The method must NOT
    return bare True (pre-fix it did); a subset apply fails closed so the caller
    falls back to the keyword Boolean instead of believing the geography landed.
    """

    async def _body(browser, page):
        _stub_live_page_helpers(browser)
        # Sanity: the editor input is present (reveal branch skipped) and exactly
        # one of the two requested values has a confirmable chip in this DOM.
        assert await page.locator(
            'aside.left-rail input.artdeco-typeahead__input[placeholder*="location" i]'
        ).count() == 1
        ok = await browser.apply_location_filter(["New York", "Berlin"])
        assert ok is False, "subset apply (1 of 2) must not report full success"
        # P3a Stage B: the gate-2 miss captured the REAL offered options for
        # the missed value (the orchestrator's facet-resolution input) — the
        # landed value must not appear (test-honesty lens, slice 13: this
        # capture had zero coverage through the real DOM path).
        assert browser.last_location_option_misses == {
            "Berlin": ["New York", "Berlin, Germany Area"]
        }

    _run_with_offline_page(_LOCATIONS_PARTIAL, _body)


def test_apply_company_filter_partial_apply_does_not_report_full_success():
    """Mirror of the location partial-apply test for the byte-identical sibling
    apply_company_filter (still MOCK_ONLY-gated, made honest now so it is correct
    the moment it graduates). 'Acme Inc' lands; 'Globex' has no exact-match
    option -> 1 of 2 -> must fail closed, not report bare True.
    """

    async def _body(browser, page):
        _stub_live_page_helpers(browser)
        assert await page.locator(
            'aside.left-rail input.artdeco-typeahead__input[placeholder*="company" i]'
        ).count() == 1
        # The offline DOM's listbox is static, so "Globex" spends the whole
        # repaint wait proving the rows are not its answer. Nothing here is
        # about timing.
        with patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter(["Acme Inc", "Globex"])
        assert ok is False, "subset apply (1 of 2) must not report full success"

    _run_with_offline_page(_COMPANIES_PARTIAL, _body)


def test_apply_location_filter_full_apply_still_reports_success():
    """Positive pin: when EVERY requested location lands a chip, the method still
    returns True. Guards against the fail-closed arithmetic over-firing on a
    genuinely complete apply.
    """

    async def _body(browser, page):
        _stub_live_page_helpers(browser)
        ok = await browser.apply_location_filter(["New York"])
        assert ok is True

    _run_with_offline_page(_LOCATIONS_PARTIAL, _body)


# Captured-DOM regression (2026-06 Recruiter capture, country query "Colombia").
# Models the applied-pill markup returned by that capture: the chip is a
# `li[data-test-facet-pills-item] > span.facet-pill`, the dismiss control is
# `button.facet-pill__action[data-test-pill-dismiss]` named "Remove {value}", and
# that older capture omitted `[data-test-pill-label]` (the 2026-08-21 live DOM
# includes it on every pill). The dismiss "X" is hover-gated
# (style="display:none" here), which is exactly why confirmation must match the
# visible pill ITEM (CSS `:has(button[aria-label=...])`, visibility-independent),
# not the hover-gated button. The country option wraps the whole label in <strong>
# while city rows bold only the fragment, but the exact-match on textContent
# already discriminates country from city. This is the test that would have caught
# the live "Colombia did not apply" failure: the old fixtures encoded
# data-test-pill-label, so they stayed green while production drifted.
_LOCATION_REAL_PILL = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-locations">
  <button class="facet-edit-button" data-test-facet-edit aria-label="Add a Candidate geographic location" type="button">Candidate geographic locations</button>
  <input class="artdeco-typeahead__input ts-common-typeahead__input" role="combobox" placeholder="enter a location\u2026" />
  <ul class="artdeco-typeahead__results-list typeahead-results" role="listbox">
    <li role="option" class="artdeco-typeahead__result" data-live-test-result="0" onclick="document.getElementById('co-pill').style.display='flex'"><strong>Colombia</strong></li>
    <li role="option" class="artdeco-typeahead__result" data-live-test-result="1">Bogot\u00e1, Capital District, <strong>Colombia</strong></li>
  </ul>
  <figure class="pills-list-section" data-test-can-have-pills-list-section>
    <ul class="pills-list-section__list">
      <li class="pills-list-section__item" data-test-facet-pills-item id="co-pill" style="display: none">
        <span class="facet-pill">Colombia<button class="facet-pill__action" aria-label="Remove Colombia" title="Remove Colombia" data-test-pill-dismiss data-view-name="search-facet-remove" type="button" style="display: none"></button></span>
      </li>
    </ul>
  </figure>
  </div>
</aside>
"""

class _FakeLocationState:
    def __init__(
        self,
        *,
        offered_options: list[str],
        visible_options: list[str] | None = None,
        editor_value: str = "",
    ):
        self.offered_options = offered_options
        self.visible_options = set(visible_options if visible_options is not None else offered_options)
        self.editor_value = editor_value
        self.applied_chips: set[str] = set()


class _FakeLocationLocator:
    def __init__(self, state: _FakeLocationState, kind: str, *, pattern=None, value: str = ""):
        self.state = state
        self.kind = kind
        self.pattern = pattern
        self.value = value

    @property
    def first(self):
        if self.kind == "options":
            # The applier polls the listbox for ANY rendered option before it
            # reads the offered names and applies the match rule.
            return _FakeLocationLocator(self.state, "any_option")
        return self

    def filter(self, *, has_text):
        return _FakeLocationLocator(self.state, "option", pattern=has_text)

    def nth(self, index: int):
        """The i-th option row, index-aligned with ``all_inner_texts``."""
        assert self.kind == "options", f"nth() on a {self.kind} locator"
        return _FakeLocationLocator(
            self.state, "option", value=self.state.offered_options[index]
        )

    def _matching_option(self) -> str:
        if self.kind != "option":
            return ""
        if self.value:
            return self.value
        if self.pattern is None:
            return ""
        for option in self.state.offered_options:
            if self.pattern.search(option):
                return option
        return ""

    async def is_visible(self, timeout: int = 0) -> bool:
        if self.kind == "editor":
            return True
        if self.kind == "option":
            return self._matching_option() in self.state.visible_options
        if self.kind == "any_option":
            return any(
                option in self.state.visible_options
                for option in self.state.offered_options
            )
        if self.kind == "chip":
            return self.value in self.state.applied_chips
        return False

    async def input_value(self, timeout: int = 0) -> str:
        # The apply loop reads the editor back after typing to detect
        # swallowed keystrokes; the rig's editor content is state.editor_value.
        if self.kind == "editor":
            return self.state.editor_value
        return ""

    async def all_inner_texts(self):
        # No query, no suggestions: LinkedIn's typeahead renders rows for the
        # text in the editor, so a cleared editor offers nothing. The appliers
        # read the rows before typing to tell this name's listbox from the
        # previous name's, and a rig that answered the same rows either way
        # would call every fresh list stale.
        if self.kind == "options" and not self.state.editor_value:
            return []
        return list(self.state.offered_options)


class _FakeLocationRail:
    def __init__(self, state: _FakeLocationState):
        self.state = state

    def locator(self, selector: str):
        if selector.startswith("div.search-facet-wrapper.facet-"):
            return self
        if selector.startswith("input.artdeco-typeahead__input"):
            return _FakeLocationLocator(self.state, "editor")
        if "ul.artdeco-typeahead__results-list" in selector:
            return _FakeLocationLocator(self.state, "options")
        if "button:has-text" in selector:
            return _FakeLocationLocator(self.state, "add_button")
        marker = 'aria-label="Remove '
        if marker in selector:
            value = selector.split(marker, 1)[1].split('"', 1)[0]
            return _FakeLocationLocator(self.state, "chip", value=value)
        raise AssertionError(f"unexpected selector: {selector}")


class _FakeLocationPage:
    def __init__(self, state: _FakeLocationState):
        self.state = state

    def locator(self, selector: str):
        if selector == "aside.left-rail":
            return _FakeLocationRail(self.state)
        raise AssertionError(f"unexpected page selector: {selector}")


async def _fake_visible_poll(locator, *, timeout_ms: int, interval_ms: int) -> bool:
    return await locator.is_visible(timeout=50)


async def _fake_location_option_click(locator):
    if locator.kind == "option":
        matched = locator._matching_option()
        if matched:
            locator.state.applied_chips.add(matched)
    return True


def _make_fake_location_browser(state: _FakeLocationState) -> LinkedInBrowser:
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = _FakeLocationPage(state)
    _stub_live_page_helpers(browser)
    browser._wait_for_visible_poll = AsyncMock(side_effect=_fake_visible_poll)
    browser._wait_for_facet_pill = AsyncMock(
        side_effect=lambda _wrapper, value, **_kwargs: value in state.applied_chips
    )
    browser._ghost_click_locator = AsyncMock(side_effect=_fake_location_option_click)
    return browser


def test_apply_location_filter_confirms_without_retrying_successful_apply(capsys):
    """Older captured markup: no [data-test-pill-label], country exact-match picks
    <strong>Colombia</strong> over the city rows, and confirmation succeeds even
    though the dismiss control is display:none (hover-gated) — because we confirm
    the visible pill ITEM, not the button. Regression for the June live failure.
    """

    async def _body(browser, page):
        _stub_live_page_helpers(browser)
        # Pin the older captured fallback shape; current live pills carry labels.
        assert await page.locator("[data-test-pill-label]").count() == 0
        ok = await browser.apply_location_filter(["Colombia"])
        assert ok is True
        assert "[retry" not in capsys.readouterr().out

    _run_with_offline_page(_LOCATION_REAL_PILL, _body)


def test_apply_location_filter_retries_once_after_exact_match_poll_miss():
    async def _body():
        value = "San Francisco Bay Area"
        state = _FakeLocationState(
            offered_options=[value],
            visible_options=[],
            editor_value="stale miss text",
        )
        browser = _make_fake_location_browser(state)
        events = []
        type_calls = 0

        async def _clear(locator):
            events.append("clear")
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            nonlocal type_calls
            type_calls += 1
            events.append(("type", text))
            locator.state.editor_value += text
            if type_calls == 2:
                state.visible_options.add(value)

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        ok = await browser.apply_location_filter([value])
        assert ok is True
        assert events == ["clear", ("type", value), "clear", ("type", value)]

    asyncio.run(_body())


def test_apply_location_filter_skips_values_already_applied_as_chips():
    """Chips persisted from a previous session satisfy their values WITHOUT
    typing: LinkedIn removes an applied facet from the typeahead suggestions,
    so re-typing an applied value can never exact-match (2026-07-06 SPL-MM
    live aborts — only city rows offered while the country chips sat
    applied). All-applied short-circuits True with zero editor input."""

    async def _body():
        state = _FakeLocationState(offered_options=[])
        browser = _make_fake_location_browser(state)
        browser.read_applied_location_chips = AsyncMock(
            return_value=["Brazil", "Colombia"]
        )
        browser._clear_keyword_textarea = AsyncMock()
        browser._input_backend.type_text = AsyncMock()

        ok = await browser.apply_location_filter(["Brazil", "Colombia"])

        assert ok is True
        browser._input_backend.type_text.assert_not_awaited()
        browser._clear_keyword_textarea.assert_not_awaited()

    asyncio.run(_body())


@pytest.mark.parametrize(
    ("apply_method", "reader_method"),
    [
        ("apply_location_filter", "read_applied_location_chips"),
        ("apply_company_filter", "read_applied_company_chips"),
        ("apply_title_filter", "read_applied_title_chips"),
    ],
)
def test_apply_facet_filter_blocks_when_applied_set_is_unreadable(
    apply_method, reader_method
):
    async def _body():
        value = "Exact Value"
        state = _FakeLocationState(offered_options=[value])
        browser = _make_fake_location_browser(state)
        setattr(browser, reader_method, AsyncMock(return_value=None))

        ok = await getattr(browser, apply_method)([value])

        assert ok is False
        browser._input_backend.type_text.assert_not_awaited()

    asyncio.run(_body())


def test_apply_location_filter_exhausted_retries_capture_options_and_clear():
    async def _body():
        state = _FakeLocationState(
            offered_options=["New York City Metropolitan Area", "San Francisco Bay Area"],
            editor_value="previous miss",
        )
        browser = _make_fake_location_browser(state)
        events = []

        async def _clear(locator):
            events.append("clear")
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            events.append(("type", text))
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        ok = await browser.apply_location_filter(["Berlin"])
        assert ok is False
        assert browser.last_location_option_misses == {
            "Berlin": ["New York City Metropolitan Area", "San Francisco Bay Area"]
        }
        # Two type-and-poll attempts: a POPULATED option list without the
        # exact row is a facet-name mismatch, not hydration lag, so after one
        # ranking-variance redraw (2026-07-06 SPL-MM) the apply fails fast to
        # hand Stage-B resolution its offered-options capture instead of
        # stalling out the 2026-08-14 wall-clock budget. Final best-effort
        # clear unchanged.
        assert events == ["clear", ("type", "Berlin")] * 2 + ["clear"]

    asyncio.run(_body())


def test_apply_location_filter_clears_between_multiple_values():
    async def _body():
        state = _FakeLocationState(
            offered_options=["New York", "San Francisco Bay Area"],
        )
        browser = _make_fake_location_browser(state)
        typed_values = []

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            locator.state.editor_value += text
            typed_values.append(locator.state.editor_value)

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        ok = await browser.apply_location_filter(["New York", "San Francisco Bay Area"])
        assert ok is True
        assert typed_values == ["New York", "San Francisco Bay Area"]
        assert browser._input_backend.type_text.await_args_list[1].args[2] == "San Francisco Bay Area"

    asyncio.run(_body())


def test_apply_location_filter_clear_failure_fails_closed_and_logs_gate(caplog):
    async def _body():
        state = _FakeLocationState(offered_options=["New York"])
        browser = _make_fake_location_browser(state)
        browser._clear_keyword_textarea = AsyncMock(
            side_effect=RuntimeError("keyword textarea did not clear")
        )

        with caplog.at_level(logging.WARNING, logger="linkedin.browser"):
            ok = await browser.apply_location_filter(["New York"])

        assert ok is False
        assert "gate=editor_clear" in caplog.text
        browser._input_backend.type_text.assert_not_awaited()

    asyncio.run(_body())


def test_apply_location_filter_already_applied_precheck_uses_casefold():
    async def _body():
        state = _FakeLocationState(offered_options=[])
        browser = _make_fake_location_browser(state)
        browser.read_applied_location_chips = AsyncMock(
            return_value=["Sao Paulo, Brazil"]
        )
        browser._clear_keyword_textarea = AsyncMock()
        browser._input_backend.type_text = AsyncMock()

        ok = await browser.apply_location_filter(["sao paulo, brazil"])

        assert ok is True
        assert browser.last_location_already_applied_count == 1
        browser._input_backend.type_text.assert_not_awaited()
        browser._clear_keyword_textarea.assert_not_awaited()

    asyncio.run(_body())


def test_apply_company_filter_skips_values_already_applied_as_chips():
    async def _body():
        state = _FakeLocationState(offered_options=[])
        browser = _make_fake_location_browser(state)
        browser.read_applied_company_chips = AsyncMock(return_value=["Stripe", "OpenAI"])
        browser._clear_keyword_textarea = AsyncMock()
        browser._input_backend.type_text = AsyncMock()

        ok = await browser.apply_company_filter(["Stripe", "OpenAI"])

        assert ok is True
        assert browser.last_company_already_applied_count == 2
        browser._input_backend.type_text.assert_not_awaited()
        browser._clear_keyword_textarea.assert_not_awaited()

    asyncio.run(_body())


def test_apply_company_filter_retry_recovers_on_second_poll():
    async def _body():
        value = "Stripe"
        state = _FakeLocationState(
            offered_options=[value],
            visible_options=[],
            editor_value="stale miss text",
        )
        browser = _make_fake_location_browser(state)
        events = []
        type_calls = 0

        async def _clear(locator):
            events.append("clear")
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            nonlocal type_calls
            type_calls += 1
            events.append(("type", text))
            locator.state.editor_value += text
            if type_calls == 2:
                state.visible_options.add(value)

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        with patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter([value])

        assert ok is True
        assert events == ["clear", ("type", value), "clear", ("type", value)]

    asyncio.run(_body())


def test_apply_company_filter_exhausted_retries_captures_options_and_clears():
    async def _body():
        # The listbox renders nothing for the whole budget (the text nodes
        # exist in the DOM but no option is shown), which is the only shape
        # that still spends four retypes now that a listbox which ANSWERS is
        # a decided miss.
        state = _FakeLocationState(
            offered_options=["Stripe Payments", "Stripe Atlas"],
            visible_options=[],
            editor_value="previous miss",
        )
        browser = _make_fake_location_browser(state)
        events = []

        async def _clear(locator):
            events.append("clear")
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            events.append(("type", text))
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        with patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter(["Stripe"])

        assert ok is False
        assert browser.last_company_option_misses == {
            "Stripe": ["Stripe Payments", "Stripe Atlas"]
        }
        assert events == ["clear", ("type", "Stripe")] * 4 + ["clear"]

    asyncio.run(_body())


def test_apply_company_filter_clears_between_multiple_values():
    async def _body():
        state = _FakeLocationState(offered_options=["Stripe", "OpenAI"])
        browser = _make_fake_location_browser(state)
        typed_values = []

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            locator.state.editor_value += text
            typed_values.append(locator.state.editor_value)

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        ok = await browser.apply_company_filter(["Stripe", "OpenAI"])

        assert ok is True
        assert typed_values == ["Stripe", "OpenAI"]

    asyncio.run(_body())


def test_apply_company_filter_fail_closed_subset_with_option_misses():
    async def _body():
        state = _FakeLocationState(
            offered_options=["Stripe", "Open Systems"],
        )
        browser = _make_fake_location_browser(state)

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        with patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter(["Stripe", "OpenAI"])

        assert ok is False
        assert browser.last_company_option_misses == {
            "OpenAI": ["Stripe", "Open Systems"]
        }

    asyncio.run(_body())


def test_apply_title_filter_skips_values_already_applied_as_chips():
    async def _body():
        state = _FakeLocationState(offered_options=[])
        browser = _make_fake_location_browser(state)
        browser.read_applied_title_chips = AsyncMock(
            return_value=["Software Engineer", "Data Scientist"]
        )
        browser._clear_keyword_textarea = AsyncMock()
        browser._input_backend.type_text = AsyncMock()

        ok = await browser.apply_title_filter(["Software Engineer", "Data Scientist"])

        assert ok is True
        assert browser.last_title_already_applied_count == 2
        browser._input_backend.type_text.assert_not_awaited()
        browser._clear_keyword_textarea.assert_not_awaited()

    asyncio.run(_body())


def test_apply_title_filter_retry_recovers_on_second_poll():
    async def _body():
        value = "Software Engineer"
        state = _FakeLocationState(
            offered_options=[value],
            visible_options=[],
            editor_value="stale miss text",
        )
        browser = _make_fake_location_browser(state)
        events = []
        type_calls = 0

        async def _clear(locator):
            events.append("clear")
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            nonlocal type_calls
            type_calls += 1
            events.append(("type", text))
            locator.state.editor_value += text
            if type_calls == 2:
                state.visible_options.add(value)

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        with patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_title_filter([value])

        assert ok is True
        assert events == ["clear", ("type", value), "clear", ("type", value)]

    asyncio.run(_body())


def test_apply_title_filter_exhausted_retries_captures_options_and_clears():
    async def _body():
        state = _FakeLocationState(
            offered_options=["Senior Software Engineer", "Engineering Manager"],
            visible_options=[],
            editor_value="previous miss",
        )
        browser = _make_fake_location_browser(state)
        events = []

        async def _clear(locator):
            events.append("clear")
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            events.append(("type", text))
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        with patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_title_filter(["Principal Architect"])

        assert ok is False
        assert browser.last_title_option_misses == {
            "Principal Architect": ["Senior Software Engineer", "Engineering Manager"]
        }
        assert events == ["clear", ("type", "Principal Architect")] * 4 + ["clear"]

    asyncio.run(_body())


def test_apply_title_filter_clears_between_multiple_values():
    async def _body():
        state = _FakeLocationState(offered_options=["Software Engineer", "Data Scientist"])
        browser = _make_fake_location_browser(state)
        typed_values = []

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            locator.state.editor_value += text
            typed_values.append(locator.state.editor_value)

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        ok = await browser.apply_title_filter(["Software Engineer", "Data Scientist"])

        assert ok is True
        assert typed_values == ["Software Engineer", "Data Scientist"]

    asyncio.run(_body())


def test_apply_title_filter_fail_closed_subset_with_option_misses():
    async def _body():
        state = _FakeLocationState(
            offered_options=["Software Engineer", "Architect Principal"],
        )
        browser = _make_fake_location_browser(state)

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        with patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_title_filter(["Software Engineer", "Principal Architect"])

        assert ok is False
        assert browser.last_title_option_misses == {
            "Principal Architect": ["Software Engineer", "Architect Principal"]
        }

    asyncio.run(_body())


def test_apply_advanced_search_plan_prints_applied_facet_receipt(capsys):
    browser = _make_browser_with_enter_search(succeed=True)
    browser.apply_company_filter = AsyncMock(return_value=True)
    browser.last_company_already_applied_count = 3
    plan = AdvancedSearchPlan(
        controls=[
            AdvancedSearchControl(
                dimension="companies",
                values=["A", "B", "C", "D", "E"],
            )
        ],
    )

    result = asyncio.run(apply_advanced_search_plan(browser, plan))

    assert result.success is True
    assert capsys.readouterr().out == (
        "  [facet] companies: applied 5/5 (3 already on sidebar)\n"
    )


def test_apply_advanced_search_plan_prints_failed_facet_receipt(capsys):
    browser = _make_browser_with_enter_search(succeed=True)
    browser.apply_company_filter = AsyncMock(return_value=False)
    plan = AdvancedSearchPlan(
        keyword_boolean='"fallback"',
        controls=[
            AdvancedSearchControl(
                dimension="companies",
                values=["Stripe"],
            )
        ],
    )

    result = asyncio.run(apply_advanced_search_plan(browser, plan))

    assert result.success is False
    assert result.fallback_to_boolean is True
    assert capsys.readouterr().out == (
        "  [facet] companies: FAILED — falling back to keyword boolean\n"
    )


def test_apply_company_filter_reveals_editor_when_companies_facet_has_existing_pills():
    """Live regression pin: once Companies already has pills, LinkedIn labels the
    reveal button "Companies or boolean" instead of "Add Companies or boolean".
    The browser must still find that facet-local button, open the editor, and
    complete the exact-match chip gate.
    """

    async def _body(browser, page):
        _stub_live_page_helpers(browser)
        assert await page.locator(
            'aside.left-rail button:has-text("Add Companies or boolean")'
        ).count() == 0
        assert await page.locator(
            'aside.left-rail button:has-text("Companies or boolean")'
        ).count() == 1

        ok = await browser.apply_company_filter(["Stripe"])
        assert ok is True

    _run_with_offline_page(_COMPANIES_EXISTING_PILLS_REVEAL, _body)


# ---------------------------------------------------------------------------
# apply_title_filter tests
# ---------------------------------------------------------------------------

_TITLES_BASIC = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-titles">
  <section>
    <h2>Job titles</h2>
    <button
      type="button"
      onclick="document.querySelector('#title-editor').style.display = 'block'"
    >Job titles or boolean</button>
  </section>
  <input
    id="title-editor"
    class="artdeco-typeahead__input"
    placeholder="enter a job title or boolean"
    style="display: none"
  />
  <ul class="artdeco-typeahead__results-list" role="listbox">
    <li role="option" onclick="document.querySelector('#se-pill').style.display = 'flex'">Software Engineer</li>
    <li role="option">Senior Software Engineer</li>
  </ul>
  <figure class="pills-list-section" data-test-can-have-pills-list-section>
    <ul class="pills-list-section__list">
      <li class="pills-list-section__item" data-test-facet-pills-item id="se-pill" style="display: none">
        <span class="facet-pill">Software Engineer<button class="facet-pill__action" aria-label="Remove Software Engineer" data-test-pill-dismiss type="button"></button></span>
      </li>
    </ul>
  </figure>
  </div>
</aside>
"""

_TITLES_EXISTING_PILLS = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-titles">
  <section>
    <h2>Job titles</h2>
    <button
      type="button"
      onclick="document.querySelector('#title-editor-2').style.display = 'block'"
    >Job titles or boolean</button>
    <figure class="pills-list-section" data-test-can-have-pills-list-section>
      <ul class="pills-list-section__list">
        <li class="pills-list-section__item" data-test-facet-pills-item><span class="facet-pill">Data Scientist<button class="facet-pill__action" aria-label="Remove Data Scientist" data-test-pill-dismiss type="button"></button></span></li>
      </ul>
    </figure>
  </section>
  <input
    id="title-editor-2"
    class="artdeco-typeahead__input"
    placeholder="enter a job title or boolean"
    style="display: none"
  />
  <ul class="artdeco-typeahead__results-list" role="listbox">
    <li role="option" onclick="document.querySelector('#se-pill-2').style.display = 'flex'">Software Engineer</li>
  </ul>
  <figure class="pills-list-section" data-test-can-have-pills-list-section>
    <ul class="pills-list-section__list">
      <li class="pills-list-section__item" data-test-facet-pills-item id="se-pill-2" style="display: none">
        <span class="facet-pill">Software Engineer<button class="facet-pill__action" aria-label="Remove Software Engineer" data-test-pill-dismiss type="button"></button></span>
      </li>
    </ul>
  </figure>
  </div>
</aside>
"""


def test_apply_title_filter_basic_apply():
    """Positive pin: a single title with an exact-match option and chip lands
    successfully via the stable 'Job titles or boolean' reveal selector."""

    async def _body(browser, page):
        _stub_live_page_helpers(browser)
        ok = await browser.apply_title_filter(["Software Engineer"])
        assert ok is True

    _run_with_offline_page(_TITLES_BASIC, _body)


def test_apply_title_filter_reveals_editor_with_existing_pills():
    """Regression pin mirroring the company existing-pills test: when the Job
    titles facet already has pills, LinkedIn drops the 'Add' prefix. The method
    must still find the facet-local button and complete the chip gate."""

    async def _body(browser, page):
        _stub_live_page_helpers(browser)
        assert await page.locator(
            'aside.left-rail button:has-text("Add Job titles or boolean")'
        ).count() == 0
        assert await page.locator(
            'aside.left-rail button:has-text("Job titles or boolean")'
        ).count() == 1

        ok = await browser.apply_title_filter(["Software Engineer"])
        assert ok is True

    _run_with_offline_page(_TITLES_EXISTING_PILLS, _body)


def test_apply_title_filter_no_exact_match_fails_closed():
    """A title with no exact-match option fails closed (returns False), same
    discipline as location/company siblings."""

    async def _body(browser, page):
        _stub_live_page_helpers(browser)
        # Static offline listbox: the repaint wait runs its full budget before
        # the miss is recorded. Nothing here is about timing.
        with patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_title_filter(["Principal Architect"])
        assert ok is False

    _run_with_offline_page(_TITLES_BASIC, _body)


# ---------------------------------------------------------------------------
# PRE-EXISTING method-grain fail-closed: apply_company_filter on a non-results
# view (no left-rail). NOT an R8 addition — this behavior predates R8 and is a
# regression guard for it. (R8's only source change is the smoke's results-rail
# precondition GATE in tools/hop4_company_smoke.py::_run, covered by
# tests/test_hop4_company_smoke.py — not by anything in this file.)
#
# On Chrome 148 connect() can land a created page on /advanced (the entry FORM)
# or another non-results URL where the search-results refinement rail
# (aside.left-rail) — and therefore the 'Add Companies or boolean' button and the
# typeahead editor — are ABSENT. apply_company_filter resolves everything inside
# that rail, so on such a page it fails closed (returns False) WITHOUT raising.
# That fail-closed-False is precisely why the R8 smoke gate exists: on a
# non-results view a correct selector still yields False, which the smoke would
# otherwise misread as a selector miss; the gate detects the absent rail FIRST and
# fails the smoke loudly instead. Production is unaffected (search_mutation only
# applies after a keyword search, i.e. on a populated results rail). Offline DOM
# mirrors the /advanced entry form: the typeahead chrome exists but there is NO
# aside.left-rail, so rail.locator(...) resolves nothing -> editor never visible ->
# Add button never visible -> per-value _do returns False -> 0 of 1 applied -> fail
# closed. Live-page helpers are stubbed so the test isolates the missing-rail
# resolution + fail-closed arithmetic.
# ---------------------------------------------------------------------------

_COMPANIES_NO_RAIL = """
<main class="advanced-search">
  <form class="advanced-search__form">
    <input class="artdeco-typeahead__input" placeholder="Add a company or boolean" />
    <ul class="artdeco-typeahead__results-list" role="listbox">
      <li role="option">Stripe</li>
    </ul>
  </form>
</main>
"""


def test_apply_company_filter_fail_closed_on_missing_rail():
    """PRE-EXISTING (not R8) method-grain fail-closed, kept as an R8 regression
    guard. No aside.left-rail (an /advanced-style entry form) -> apply_company_filter
    returns False and does NOT raise. Even though a 'Stripe' option exists in the DOM,
    it lives OUTSIDE the rail the method scopes to, so the editor/Add button are never
    found and the apply fails closed. This fail-closed-False is the exact reason the R8
    smoke GATE exists (it would otherwise read as a selector miss) — but the gate
    itself is source in tools/hop4_company_smoke.py and is tested in
    tests/test_hop4_company_smoke.py, not here. This test only pins the method's
    behavior: a non-results view is a clean False, never a crash and never a masked
    PASS.
    """

    async def _body(browser, page):
        _stub_live_page_helpers(browser)
        # Precondition mirrors the smoke's: the results rail is genuinely absent
        # (this is what the smoke now detects before applying), while the broad
        # company chrome that fools a naive selector IS present on the page.
        assert await page.locator("aside.left-rail").count() == 0
        assert await page.locator(
            'input.artdeco-typeahead__input[placeholder*="company" i]'
        ).count() == 1
        ok = await browser.apply_company_filter(["Stripe"])
        assert ok is False, "no results rail -> apply_company_filter must fail closed"

    _run_with_offline_page(_COMPANIES_NO_RAIL, _body)


def test_snapshot_records_only_applied_locations_not_requested():
    """R5 recovery-snapshot honesty: when 1 of 2 location controls lands, the
    persisted snapshot must NOT list the un-applied value, so it is never
    replayed as if it had applied. Exercises the browser delegate's snapshot
    filter directly (apply_advanced_search_plan -> snapshot_controls_from_plan
    with the clean-applied dimension set), with apply_location_filter stubbed to
    succeed for 'New York' and fail for 'Berlin'.
    """

    async def _apply(values, *, temporal_scope="any"):
        # Single-value controls (as compile_structured_filters_to_plan produces).
        return values == ["New York"]

    browser = _make_browser_with_enter_search(succeed=True)
    browser.apply_location_filter = AsyncMock(side_effect=_apply)
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML"',
        controls=[
            AdvancedSearchControl(dimension="keywords", values=['"ML"']),
            AdvancedSearchControl(dimension="locations", values=["New York"]),
            AdvancedSearchControl(dimension="locations", values=["Berlin"]),
        ],
    )
    asyncio.run(browser.apply_advanced_search_plan(plan))
    snap = asyncio.run(browser.snapshot_advanced_search_controls())
    recorded_values = [v for c in snap["controls"] for v in c["values"]]
    # The un-applied location must not be persisted...
    assert "Berlin" not in recorded_values
    # ...and because 'Berlin' failed, the mixed 'locations' dimension is excluded
    # wholesale (fail-closed: never record a partial geography). Only the keyword
    # Boolean survives — the snapshot the recruiter would replay.
    recorded_dims = {c["dimension"] for c in snap["controls"]}
    assert recorded_dims == {"keywords"}


# ---------------------------------------------------------------------------
# ADVERSARIAL VERIFIER ADDITIONS (apply-honesty fix)
#
# These pin gaps the implementer's own suite did not cover. All FAIL on the
# pre-fix source (confirmed by stashing linkedin/advanced_search.py +
# linkedin/browser.py and rerunning): they are genuine regressions guards, not
# tautologies that pass either way.
# ---------------------------------------------------------------------------


def test_company_only_plan_through_controller_is_not_full_success():
    """R2 honesty (plan grain), now exercised via fields_of_study: a still-mock_only
    structured control routed through apply_advanced_search_plan must be honest. The
    controller shunts it to unsupported and only the keyword Boolean lands. Pre-fix
    the through-plan path returned success=True / reason 'all_stable_controls_applied'
    (the recruiter believes a structured filter is in place). It must be
    success=False / plan_fully_applied=False with the dropped dimension visible. (This
    used 'companies' before slice H graduated it; fields_of_study is the surviving
    mock_only control that still exercises the through-plan honesty path.)
    """
    browser = _make_browser_with_enter_search(succeed=True)
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML"',
        controls=[
            AdvancedSearchControl(dimension="keywords", values=['"ML"']),
            AdvancedSearchControl(dimension="fields_of_study", values=["Computer Science"]),
        ],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is False
    assert result.plan_fully_applied is False
    assert result.reason == "structured_controls_dropped_keyword_only"
    assert result.applied_controls == ["keywords"]
    assert "fields_of_study" in result.unsupported_controls
    d = result.to_dict()
    assert d["plan_fully_applied"] is False
    assert d["success"] is False


def test_failed_stable_control_serializes_not_fully_applied():
    """Honesty axis on the FAILED branch (the implementer pinned the unsupported
    and keyword-only branches but not this one): when a graduated stable control
    fails verification, success=False AND plan_fully_applied=False must survive
    to_dict, so a consumer reading the serialized mutation event can tell a
    failed structured apply from a clean one. Pre-fix there was no
    plan_fully_applied key at all (AttributeError / missing in to_dict).
    Locations is routed live (stable_now); the stubbed apply_location_filter
    returns False to model a verification miss.
    """
    browser = _make_browser_with_enter_search(succeed=True)
    browser.apply_location_filter = AsyncMock(return_value=False)
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML"',
        controls=[
            AdvancedSearchControl(dimension="keywords", values=['"ML"']),
            AdvancedSearchControl(dimension="locations", values=["NYC"]),
        ],
    )
    result = asyncio.run(apply_advanced_search_plan(browser, plan))
    assert result.success is False
    assert "locations" in result.failed_controls
    assert result.fallback_to_boolean is True
    assert result.reason == "stable_now_control_failed_fallback_to_boolean"
    d = result.to_dict()
    assert "plan_fully_applied" in d
    assert d["plan_fully_applied"] is False


def test_snapshot_excludes_mixed_same_dimension_fail_closed_never_overrecords():
    """R5 (snapshot, conservative direction the implementer documented but did
    not pin): when ONE 'locations' control of several fails while the others
    land, the WHOLE 'locations' dimension is dropped from the recovery snapshot —
    a partial geography is never recorded, so it is never replayed as if fully
    applied. NYC and LA genuinely landed chips but are deliberately under-recorded
    (fail-closed): the only thing this asserts is that the un-applied 'Berlin' and
    the mixed dimension never OVER-record. Pre-fix apply_location_filter could not
    even produce this shape via the plan (it returned bare True on any apply); the
    direct side_effect models the per-value outcome and the dimension-grain
    exclusion in the browser delegate. Guards against a future 'optimization' that
    records the chips that did land (which would resurrect the over-broad replay).
    """

    async def _apply(values, *, temporal_scope="any"):
        return values != ["Berlin"]

    browser = _make_browser_with_enter_search(succeed=True)
    browser.apply_location_filter = AsyncMock(side_effect=_apply)
    plan = AdvancedSearchPlan(
        keyword_boolean='"ML"',
        controls=[
            AdvancedSearchControl(dimension="keywords", values=['"ML"']),
            AdvancedSearchControl(dimension="locations", values=["NYC"]),
            AdvancedSearchControl(dimension="locations", values=["LA"]),
            AdvancedSearchControl(dimension="locations", values=["Berlin"]),
        ],
    )
    asyncio.run(browser.apply_advanced_search_plan(plan))
    snap = asyncio.run(browser.snapshot_advanced_search_controls())
    recorded_values = [v for c in snap["controls"] for v in c["values"]]
    recorded_dims = {c["dimension"] for c in snap["controls"]}
    # The failed value is never persisted.
    assert "Berlin" not in recorded_values
    # The mixed dimension is excluded WHOLESALE — no partial geography recorded.
    assert "locations" not in recorded_dims
    assert recorded_dims == {"keywords"}


# ---------------------------------------------------------------------------
# Wave 2 slice 2.2 — section-directed profile expand (flag-gated)
# ---------------------------------------------------------------------------


def test_expand_stops_after_the_sampled_experience_count():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._ghost_click_locator = AsyncMock()
    page = MagicMock()
    page.wait_for_timeout = AsyncMock()
    browser._page = page

    examined_entries = []

    def make_position_item(idx):
        item = MagicMock()

        def track_locator(selector):
            if 'a, button, [role="button"]' in selector:
                examined_entries.append(idx)
            controls = MagicMock()
            controls.all = AsyncMock(return_value=[])
            return controls

        item.locator = MagicMock(side_effect=track_locator)
        return item

    position_items = [make_position_item(i) for i in range(10)]
    experience_items_mock = MagicMock()
    experience_items_mock.all = AsyncMock(return_value=position_items)

    summary_card = MagicMock()
    summary_clickables = MagicMock()
    summary_clickables.all = AsyncMock(return_value=[])
    summary_card.locator = MagicMock(return_value=summary_clickables)

    def container_locator(selector):
        if selector == ".summary-card":
            return summary_card
        if selector == ".experience-card .position-item":
            return experience_items_mock
        return MagicMock()

    container = MagicMock()
    container.locator = MagicMock(side_effect=container_locator)

    with patch("linkedin.browser.config.LINKEDIN_SECTION_DIRECTED_READ_ENABLED", True), patch(
        "linkedin.browser.random.randint", return_value=3
    ):
        asyncio.run(browser._expand_all_readmore(container))

    assert len(examined_entries) <= 3
    assert examined_entries == [0, 1, 2]


def test_expand_does_not_walk_every_clickable_in_the_container():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._ghost_click_locator = AsyncMock()
    page = MagicMock()
    page.wait_for_timeout = AsyncMock()
    browser._page = page

    container_locator_calls = []

    def make_clickables_mock():
        controls = MagicMock()
        controls.all = AsyncMock(return_value=[])
        return controls

    def make_position_item():
        item = MagicMock()
        item.locator = MagicMock(side_effect=lambda _sel: make_clickables_mock())
        return item

    position_items = [make_position_item() for _ in range(10)]
    experience_items_mock = MagicMock()
    experience_items_mock.all = AsyncMock(return_value=position_items)

    summary_card = MagicMock()
    summary_card.locator = MagicMock(side_effect=lambda _sel: make_clickables_mock())

    def container_locator(selector):
        container_locator_calls.append(selector)
        if selector == 'a, button, [role="button"]':
            raise AssertionError("whole-container clickable walk must not run")
        if selector == ".summary-card":
            return summary_card
        if selector == ".experience-card .position-item":
            return experience_items_mock
        return MagicMock()

    container = MagicMock()
    container.locator = MagicMock(side_effect=container_locator)

    with patch("linkedin.browser.config.LINKEDIN_SECTION_DIRECTED_READ_ENABLED", True), patch(
        "linkedin.browser.random.randint", return_value=3
    ):
        asyncio.run(browser._expand_all_readmore(container))

    assert 'a, button, [role="button"]' not in container_locator_calls
    assert ".summary-card" in container_locator_calls
    assert ".experience-card .position-item" in container_locator_calls


# ---------------------------------------------------------------------------
# 2026-08-14 slow-restore hardening (plans/resume-cursor-and-location-apply-
# hardening.md): the created page's saved search restores client-side at
# ~60–115s (or never), during which the facet editor swallows keystrokes and —
# because an APPLIED value is excluded from typeahead suggestions — a restore
# landing mid-apply closes the typing path permanently. The apply loop is
# wall-clock budgeted, verifies typed text landed, and re-checks the value's
# pill every pass.
# ---------------------------------------------------------------------------


def test_apply_location_filter_recovers_after_swallowed_keystrokes(caplog):
    """Deaf-editor cycles (readback empty) burn cheap retries, not the option
    poll; once typing lands the exact-match flow completes normally."""

    async def _body():
        import linkedin.browser as browser_module

        value = "San Francisco Bay Area"
        state = _FakeLocationState(offered_options=[value])
        browser = _make_fake_location_browser(state)
        type_calls = 0

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            nonlocal type_calls
            type_calls += 1
            if type_calls >= 3:
                locator.state.editor_value = text
                state.visible_options.add(text)

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        with caplog.at_level(logging.WARNING, logger="linkedin.browser"):
            ok = await browser.apply_location_filter([value])

        assert ok is True
        assert type_calls == 3
        assert caplog.text.count("gate=typing_swallowed") == 2

    asyncio.run(_body())


def test_apply_location_filter_accepts_chip_materialized_mid_apply():
    """The saved-search restore applies the value while the loop is typing into
    a deaf editor: the pill is the only exit (an applied value never reappears
    in suggestions), and the loop must take it instead of burning the budget."""

    async def _body():
        value = "San Francisco Bay Area"
        state = _FakeLocationState(offered_options=[])
        browser = _make_fake_location_browser(state)
        type_calls = 0

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            nonlocal type_calls
            type_calls += 1
            if type_calls == 2:
                # The restore lands: pill appears without any option click.
                state.applied_chips.add(text)

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        ok = await browser.apply_location_filter([value])

        assert ok is True
        assert browser.last_location_already_applied_count == 1
        assert value not in browser.last_location_option_misses

    asyncio.run(_body())


def test_apply_location_filter_budget_exhaustion_fails_closed(monkeypatch, caplog):
    """An editor that never accepts keystrokes and never grows a pill fails
    closed at the wall-clock budget with an (empty) offered-options capture, so
    the P3a geography gate aborts exactly as before."""

    import linkedin.browser as browser_module

    monkeypatch.setattr(browser_module, "LOCATION_APPLY_BUDGET_SECONDS", 2.5)

    async def _body():
        value = "San Francisco Bay Area"
        state = _FakeLocationState(offered_options=[])
        browser = _make_fake_location_browser(state)

        async def _clear(locator):
            locator.state.editor_value = ""

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock()  # keystrokes never land

        with caplog.at_level(logging.WARNING, logger="linkedin.browser"):
            ok = await browser.apply_location_filter([value])

        assert ok is False
        assert browser.last_location_option_misses == {value: []}
        assert "gate=typing_swallowed" in caplog.text
        assert "gate=exact_match miss" in caplog.text

    asyncio.run(_body())


# Live 2026-08-21 DOM: each dimension has one dedicated
# div.search-facet-wrapper.facet-<dimension>. Readers must use that wrapper only.
_LOCATION_WRAPPER_ONLY_PILL = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-locations">
    <div class="typeahead-facet">
      <ul>
        <li data-test-facet-pills-item>
          <span class="facet-pill"><span data-test-pill-label>San Francisco Bay Area</span><button aria-label="Remove San Francisco Bay Area" data-test-pill-dismiss type="button"></button></span>
        </li>
      </ul>
    </div>
  </div>
</aside>
"""

_FOREIGN_FACET_PILL_ONLY = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-companies">
    <ul>
      <li data-test-facet-pills-item>
        <span class="facet-pill">Brazil<button aria-label="Remove Brazil" data-test-pill-dismiss type="button"></button></span>
      </li>
    </ul>
  </div>
</aside>
"""


_HEALTHY_RAIL_WITH_EMPTY_COMPANIES = """
<aside class="left-rail">
  <button type="button">Companies or boolean</button>
  <div class="search-facet-wrapper facet-locations">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Brazil</span>
      <button aria-label="Remove Brazil" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
</aside>
"""


_HEALTHY_RAIL_WITH_TRANSITIONAL_EMPTY_COMPANIES_WRAPPER = """
<aside class="left-rail">
  <button type="button">Companies or boolean</button>
  <div class="search-facet-wrapper facet-companies"></div>
  <div class="search-facet-wrapper facet-locations">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Brazil</span>
      <button aria-label="Remove Brazil" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
</aside>
"""

_LEGACY_SECTION_UNION_TRAP = """
<aside class="left-rail">
  <section class="legacy-rail-scope">
    <input class="artdeco-typeahead__input" placeholder="Search geographic location">
    <div class="search-facet-wrapper facet-locations">
      <ul><li data-test-facet-pills-item>
        <span class="facet-pill"><span data-test-pill-label>San Francisco Bay Area</span>
        <button aria-label="Remove San Francisco Bay Area" data-test-pill-dismiss></button></span>
      </li></ul>
    </div>
    <div class="search-facet-wrapper facet-companies">
      <ul><li data-test-facet-pills-item>
        <span class="facet-pill"><span data-test-pill-label>Brazil</span>
        <button aria-label="Remove Brazil" data-test-pill-dismiss></button></span>
      </li></ul>
    </div>
    <div class="search-facet-wrapper facet-titles">
      <ul><li data-test-facet-pills-item>
        <span class="facet-pill"><span data-test-pill-label>Colombia</span>
        <button aria-label="Remove Colombia" data-test-pill-dismiss></button></span>
      </li></ul>
    </div>
  </section>
</aside>
"""


def test_read_applied_location_chips_reads_wrapper_scoped_pills():
    """A pill under the live facet-locations wrapper is read with no <section>
    present anywhere — the rotation that made the reader return [] silently."""

    async def _body(browser, page):
        assert await page.locator("section").count() == 0
        chips = await browser.read_applied_location_chips()
        assert chips == ["San Francisco Bay Area"]

    _run_with_offline_page(_LOCATION_WRAPPER_ONLY_PILL, _body)


def test_read_applied_location_chips_ignores_foreign_facet_pills():
    """A missing Locations wrapper is unreadable, not an empty facet."""

    async def _body(browser, page):
        chips = await browser.read_applied_location_chips()
        assert chips is None

    _run_with_offline_page(_FOREIGN_FACET_PILL_ONLY, _body)


def test_missing_company_wrapper_is_empty_on_a_healthy_rail(caplog):
    async def _body(browser, _page):
        with caplog.at_level(logging.WARNING, logger="linkedin.browser"):
            assert await browser.read_applied_company_chips() == []
        assert caplog.text == ""

    _run_with_offline_page(_HEALTHY_RAIL_WITH_EMPTY_COMPANIES, _body)


def test_other_dimension_pills_do_not_warn_for_empty_companies(caplog):
    async def _body(browser, _page):
        with caplog.at_level(logging.WARNING, logger="linkedin.browser"):
            assert await browser.read_applied_company_chips() == []
        assert caplog.text == ""

    _run_with_offline_page(
        _HEALTHY_RAIL_WITH_TRANSITIONAL_EMPTY_COMPANIES_WRAPPER,
        _body,
    )


def test_location_reader_does_not_union_legacy_section_pills():
    """A place-named company/title in the old section scope is not geography."""

    async def _body(browser, page):
        assert await page.locator(
            'section:has(input[placeholder*="location" i])'
        ).count() == 1
        assert await browser.read_applied_location_chips() == [
            "San Francisco Bay Area"
        ]

    _run_with_offline_page(_LEGACY_SECTION_UNION_TRAP, _body)


_ALL_FACET_WRAPPERS = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-titles">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Applied Scientist</span>
      <button aria-label="Remove Applied Scientist" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
  <div class="search-facet-wrapper facet-locations"></div>
  <div class="search-facet-wrapper facet-companies">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Brazil</span>
      <button aria-label="Remove wrong fallback" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
</aside>
"""


def test_applied_facet_readers_are_wrapper_scoped_and_use_pill_labels():
    async def _body(browser, page):
        assert await browser.read_applied_location_chips() == []
        assert await browser.read_applied_company_chips() == ["Brazil"]
        assert await browser.read_applied_title_chips() == ["Applied Scientist"]

    _run_with_offline_page(_ALL_FACET_WRAPPERS, _body)


_TITLE_DISMISS_FALLBACK = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-titles">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill">Fallback Title
      <button aria-label="Remove Fallback Title" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
</aside>
"""


def test_applied_facet_reader_falls_back_to_dismiss_label():
    async def _body(browser, page):
        assert await browser.read_applied_title_chips() == ["Fallback Title"]

    _run_with_offline_page(_TITLE_DISMISS_FALLBACK, _body)


_UNPARSEABLE_COMPANY_PILL = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-companies">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><button aria-label="Dismiss" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
</aside>
"""


def test_applied_facet_reader_marks_unparseable_visible_pill_unreadable():
    async def _body(browser, page):
        assert await browser.read_applied_company_chips() is None

    _run_with_offline_page(_UNPARSEABLE_COMPANY_PILL, _body)


_FACET_RECONCILIATION_DOM = """
<style>
  li[data-test-facet-pills-item] button[data-test-pill-dismiss] { display: none; }
  li[data-test-facet-pills-item]:hover button[data-test-pill-dismiss] { display: inline; }
</style>
<aside class="left-rail">
  <div class="search-facet-wrapper facet-locations" id="location-wrapper">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Brazil</span>
      <button aria-label="Remove Brazil" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
  <div class="search-facet-wrapper facet-companies">
    <ul>
      <li data-test-facet-pills-item><span class="facet-pill">
        <span data-test-pill-label>A</span>
        <button aria-label="Remove A" data-test-pill-dismiss></button>
      </span></li>
      <li data-test-facet-pills-item><span class="facet-pill">
        <span data-test-pill-label>B</span>
        <button aria-label="Remove B" data-test-pill-dismiss></button>
      </span></li>
    </ul>
  </div>
  <div class="search-facet-wrapper facet-titles">
    <ul><li data-test-facet-pills-item><span class="facet-pill">
      <span data-test-pill-label>T1</span>
      <button aria-label="Remove T1" data-test-pill-dismiss></button>
    </span></li></ul>
  </div>
</aside>
<script>
  window.facetInteractions = [];
  document.querySelector('aside.left-rail').addEventListener('click', event => {
    const button = event.target.closest('button[data-test-pill-dismiss]');
    if (!button) return;
    const wrapper = button.closest('div.search-facet-wrapper');
    const dimension = [...wrapper.classList].find(name => name.startsWith('facet-')).slice(6);
    window.facetInteractions.push(`remove:${dimension}:${button.getAttribute('aria-label').slice(7)}`);
    button.closest('li[data-test-facet-pills-item]').remove();
  });
</script>
"""


_FACET_RECONCILIATION_DOM_EMPTY_MANAGED = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-locations">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Brazil</span>
      <button aria-label="Remove Brazil" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
  <div class="search-facet-wrapper facet-companies"><ul></ul></div>
  <div class="search-facet-wrapper facet-titles"><ul></ul></div>
</aside>
<script>
  window.facetInteractions = [];
  document.querySelector('aside.left-rail').addEventListener('click', event => {
    const button = event.target.closest('button[data-test-pill-dismiss]');
    if (!button) return;
    const wrapper = button.closest('div.search-facet-wrapper');
    const dimension = [...wrapper.classList].find(name => name.startsWith('facet-')).slice(6);
    window.facetInteractions.push(`remove:${dimension}:${button.getAttribute('aria-label').slice(7)}`);
    button.closest('li[data-test-facet-pills-item]').remove();
  });
</script>
"""


_RERENDERING_FACET_RECONCILIATION_DOM = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-locations"></div>
  <div class="search-facet-wrapper facet-companies">
    <ul>
      <li data-test-facet-pills-item><span class="facet-pill">
        <span data-test-pill-label>A</span>
        <button aria-label="Remove A" data-test-pill-dismiss></button>
      </span></li>
      <li data-test-facet-pills-item><span class="facet-pill">
        <span data-test-pill-label>B</span>
        <button aria-label="Remove B" data-test-pill-dismiss></button>
      </span></li>
      <li data-test-facet-pills-item><span class="facet-pill">
        <span data-test-pill-label>C</span>
        <button aria-label="Remove C" data-test-pill-dismiss></button>
      </span></li>
    </ul>
  </div>
  <div class="search-facet-wrapper facet-titles"><ul></ul></div>
</aside>
<script>
  window.facetInteractions = [];
  document.querySelector('aside.left-rail').addEventListener('click', event => {
    const button = event.target.closest('button[data-test-pill-dismiss]');
    if (!button) return;
    const wrapper = button.closest('div.search-facet-wrapper');
    const value = button.closest('li').querySelector('[data-test-pill-label]').textContent.trim();
    const settled = wrapper.cloneNode(true);
    [...settled.querySelectorAll('li[data-test-facet-pills-item]')]
      .find(item => item.querySelector('[data-test-pill-label]').textContent.trim() === value)
      .remove();
    const rendering = wrapper.cloneNode(false);
    rendering.append(document.createElement('ul'));
    wrapper.replaceWith(rendering);
    window.facetInteractions.push(`remove:companies:${value}`);
    setTimeout(() => rendering.replaceWith(settled), 80);
  });
</script>
"""


_ESCAPED_FACET_DISMISS_DOM = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-companies">
    <ul>
      <li data-test-facet-pills-item><span class="facet-pill">
        <span data-test-pill-label>\n          CI&amp;T\n        </span>
        <button aria-label="Remove CI&amp;amp;amp;T" data-test-pill-dismiss></button>
      </span></li>
      <li data-test-facet-pills-item><span class="facet-pill">
        <span data-test-pill-label>Say &quot;Yes&quot;</span>
        <button aria-label="Remove Say &amp;quot;Yes&amp;quot;" data-test-pill-dismiss></button>
      </span></li>
    </ul>
  </div>
</aside>
<script>
  document.querySelector('aside.left-rail').addEventListener('click', event => {
    const button = event.target.closest('button[data-test-pill-dismiss]');
    if (button) button.closest('li[data-test-facet-pills-item]').remove();
  });
</script>
"""


_ESCAPED_COMPANY_APPLY_DOM = """
<aside class="left-rail">
  <div class="search-facet-wrapper facet-companies">
    <input class="artdeco-typeahead__input" placeholder="Add a company or boolean" />
    <ul class="artdeco-typeahead__results-list" role="listbox">
      <li role="option" onclick="document.querySelector('#escaped-company-pill').style.display = 'flex'">CI&amp;T</li>
    </ul>
    <ul>
      <li id="escaped-company-pill" data-test-facet-pills-item style="display: none">
        <span class="facet-pill">
          <span data-test-pill-label>\n            CI&amp;T\n          </span>
          <button aria-label="Remove CI&amp;amp;amp;T" data-test-pill-dismiss></button>
        </span>
      </li>
    </ul>
  </div>
</aside>
"""


async def _append_facet_pill(page, dimension: str, value: str) -> None:
    await page.evaluate(
        """([dimension, value]) => {
          const wrapper = document.querySelector(`div.search-facet-wrapper.facet-${dimension}`);
          const item = document.createElement('li');
          item.setAttribute('data-test-facet-pills-item', '');
          const pill = document.createElement('span');
          pill.className = 'facet-pill';
          const label = document.createElement('span');
          label.setAttribute('data-test-pill-label', '');
          label.textContent = value;
          const button = document.createElement('button');
          button.setAttribute('data-test-pill-dismiss', '');
          button.setAttribute('aria-label', `Remove ${value}`);
          pill.append(label, button);
          item.append(pill);
          wrapper.querySelector('ul').append(item);
          window.facetInteractions.push(`add:${dimension}:${value}`);
        }""",
        [dimension, value],
    )


def test_reconcile_facets_applies_exact_diff_without_touching_retained_value():
    async def _body(browser, page):
        async def _apply_companies(values, **_kwargs):
            for value in values:
                await _append_facet_pill(page, "companies", value)
            return True

        async def _apply_titles(values, **_kwargs):
            for value in values:
                await _append_facet_pill(page, "titles", value)
            return True

        browser.apply_company_filter = AsyncMock(side_effect=_apply_companies)
        browser.apply_title_filter = AsyncMock(side_effect=_apply_titles)

        result = await browser.reconcile_facets(companies={"B", "C"}, titles={"T2"})

        assert result is None
        assert await browser.read_applied_company_chips() == ["B", "C"]
        assert await browser.read_applied_title_chips() == ["T2"]
        assert await page.evaluate("window.facetInteractions") == [
            "remove:companies:A",
            "remove:titles:T1",
            "add:companies:C",
            "add:titles:T2",
        ]
        browser.apply_company_filter.assert_awaited_once_with(["C"])
        browser.apply_title_filter.assert_awaited_once_with(["T2"])

    _run_with_offline_page(_FACET_RECONCILIATION_DOM, _body)


def test_reconcile_facets_rereads_after_each_rerendered_removal():
    async def _body(browser, page):
        await browser.reconcile_facets(companies=set(), titles=set())

        assert await browser.read_applied_company_chips() == []
        assert await page.evaluate("window.facetInteractions") == [
            "remove:companies:A",
            "remove:companies:B",
            "remove:companies:C",
        ]

    _run_with_offline_page(_RERENDERING_FACET_RECONCILIATION_DOM, _body)


def test_reconcile_facets_succeeds_when_corrective_pass_reaches_exact_set():
    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser.read_applied_company_chips = AsyncMock(
            side_effect=[["A"], [], ["Late"], [], []]
        )
        browser.read_applied_title_chips = AsyncMock(side_effect=[[], [], []])
        browser.remove_facet_value = AsyncMock(return_value=True)

        await browser.reconcile_facets(companies=set(), titles=set())

        assert browser.last_facet_reconciliation["after"] == {
            "companies": [],
            "titles": [],
        }
        assert browser.remove_facet_value.await_args_list == [
            call("companies", "A"),
            call("companies", "Late"),
        ]

    asyncio.run(_body())


@pytest.mark.parametrize(
    ("value", "escaped_aria"),
    [
        ("CI&T", "Remove CI&amp;amp;T"),
        ('Say "Yes"', "Remove Say &quot;Yes&quot;"),
    ],
)
def test_remove_facet_value_uses_label_not_escaped_aria(value, escaped_aria):
    async def _body(browser, page):
        item = page.locator(
            "div.facet-companies li[data-test-facet-pills-item]"
        ).filter(has_text=value)
        assert await item.locator("button").get_attribute("aria-label") == escaped_aria

        assert await browser.remove_facet_value("companies", value) is True
        assert await browser.read_applied_company_chips() == [
            other for other in ["CI&T", 'Say "Yes"'] if other != value
        ]

    _run_with_offline_page(_ESCAPED_FACET_DISMISS_DOM, _body)


def test_apply_company_filter_confirms_by_label_not_escaped_aria():
    async def _body(browser, _page):
        _stub_live_page_helpers(browser)

        assert await browser.apply_company_filter(["CI&T"]) is True

    _run_with_offline_page(_ESCAPED_COMPANY_APPLY_DOM, _body)


def test_reconcile_facets_same_set_spends_zero_interactions():
    async def _body(browser, page):
        browser.apply_company_filter = AsyncMock()
        browser.apply_title_filter = AsyncMock()
        browser._ghost_click_locator = AsyncMock()

        await browser.reconcile_facets(companies={"A", "B"}, titles={"T1"})

        assert await page.evaluate("window.facetInteractions") == []
        browser._ghost_click_locator.assert_not_awaited()
        browser.apply_company_filter.assert_not_awaited()
        browser.apply_title_filter.assert_not_awaited()

    _run_with_offline_page(_FACET_RECONCILIATION_DOM, _body)


def test_reconcile_facets_clear_to_empty_preserves_location_byte_for_byte():
    async def _body(browser, page):
        location = page.locator("#location-wrapper")
        before = await location.evaluate("element => element.outerHTML")

        await browser.reconcile_facets(companies=set(), titles=set())

        assert await browser.read_applied_company_chips() == []
        assert await browser.read_applied_title_chips() == []
        assert await location.evaluate("element => element.outerHTML") == before

    _run_with_offline_page(_FACET_RECONCILIATION_DOM, _body)


def test_reconcile_facets_removal_failure_raises_before_any_later_interaction():
    async def _body(browser, page):
        browser._ghost_click_locator = AsyncMock(side_effect=RuntimeError("miss"))
        browser.apply_company_filter = AsyncMock()
        browser.apply_title_filter = AsyncMock()

        with pytest.raises(FacetReconciliationError, match="remove_failed"):
            await browser.reconcile_facets(companies=set(), titles=set())

        assert await browser.read_applied_company_chips() == ["A", "B"]
        assert await browser.read_applied_title_chips() == ["T1"]
        browser.apply_company_filter.assert_not_awaited()
        browser.apply_title_filter.assert_not_awaited()

    _run_with_offline_page(_FACET_RECONCILIATION_DOM, _body)


def test_reconcile_facets_rejects_normalized_collisions():
    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser.read_applied_company_chips = AsyncMock(
            return_value=["Acme", " acme "]
        )
        browser.read_applied_title_chips = AsyncMock(return_value=[])

        with pytest.raises(FacetReconciliationError, match="normalized_collision"):
            await browser.reconcile_facets(companies={"Acme"}, titles=set())

    asyncio.run(_body())


def test_reconcile_facets_records_a_raising_applier_as_unapplied():
    """An applier that RAISES on one value is a miss, not a run-ending fault.

    RED against the batch add: the exception was wrapped as ``apply_failed``
    and raised, which is the 2-minute death loop of runs 23-25 (Addendum G5).
    """

    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser.last_facet_reconciliation = {"stale": True}
        browser.read_applied_company_chips = AsyncMock(return_value=[])
        browser.read_applied_title_chips = AsyncMock(return_value=[])
        browser.apply_company_filter = AsyncMock(
            side_effect=RuntimeError("typeahead failed")
        )

        await browser.reconcile_facets(companies={"C"}, titles=set())

        receipt = browser.last_facet_reconciliation
        assert receipt["unapplied"]["companies"] == ["C"]
        assert receipt["failure_gate"] == ""
        assert receipt["desired"]["companies"] == ["C"]
        assert "stale" not in receipt
        assert browser._owned_facet_target == {"companies": set(), "titles": set()}

    asyncio.run(_body())


def test_reconcile_facets_reraises_a_browser_crash_from_an_applier():
    """A dead target is not an exact-match miss — it must keep its identity."""

    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser.read_applied_company_chips = AsyncMock(return_value=[])
        browser.read_applied_title_chips = AsyncMock(return_value=[])
        browser.apply_company_filter = AsyncMock(
            side_effect=RuntimeError("Target page, context or browser has been closed")
        )

        with pytest.raises(RuntimeError, match="has been closed"):
            await browser.reconcile_facets(companies={"C"}, titles=set())

    asyncio.run(_body())


@pytest.mark.parametrize(
    ("dimension", "companies", "titles"),
    [
        ("companies", {"C"}, set()),
        ("titles", set(), {"T"}),
    ],
)
def test_reconcile_facets_false_applier_result_is_recorded_unapplied(
    dimension, companies, titles
):
    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser.read_applied_company_chips = AsyncMock(return_value=[])
        browser.read_applied_title_chips = AsyncMock(return_value=[])
        browser.apply_company_filter = AsyncMock(return_value=False)
        browser.apply_title_filter = AsyncMock(return_value=False)

        await browser.reconcile_facets(companies=companies, titles=titles)

        receipt = browser.last_facet_reconciliation
        assert receipt["failure_gate"] == ""
        assert receipt["unapplied"][dimension] == sorted(companies or titles)

    asyncio.run(_body())


def test_reconcile_facets_applies_one_value_per_call_and_keeps_the_landed_ones():
    """Two of three titles land; the third misses its exact-match option.

    RED against the batch add: ``apply_title_filter`` was called once with all
    three values and returns False if ANY value misses, so the whole string
    died on one misspelt entity ("Tess AI" is offered as "Tess").
    """

    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser._page = MagicMock()
        browser._page.wait_for_timeout = AsyncMock()
        landed: list[str] = []
        calls: list[list[str]] = []

        async def _apply_titles(values, **_kwargs):
            # Faithful to apply_title_filter: the values that match an exact
            # option land, and the call returns False if ANY value missed.
            calls.append(list(values))
            ok = True
            for value in values:
                if value == "Tess AI":
                    ok = False
                    continue
                landed.append(value)
            return ok

        browser.apply_company_filter = AsyncMock(return_value=True)
        browser.apply_title_filter = AsyncMock(side_effect=_apply_titles)
        browser.read_applied_company_chips = AsyncMock(return_value=[])
        browser.read_applied_title_chips = AsyncMock(side_effect=lambda: list(landed))

        await browser.reconcile_facets(
            companies=set(), titles={"Staff Engineer", "Tess AI", "Research Scientist"}
        )

        assert calls == [["Research Scientist"], ["Staff Engineer"], ["Tess AI"]]
        receipt = browser.last_facet_reconciliation
        assert receipt["failure_gate"] == ""
        assert receipt["unapplied"]["titles"] == ["Tess AI"]
        assert receipt["unapplied"]["companies"] == []
        assert sorted(receipt["after"]["titles"]) == [
            "Research Scientist",
            "Staff Engineer",
        ]
        # The owned target is what LANDED, not what was requested — otherwise
        # every reload re-verification fails on a string with a miss.
        assert browser._owned_facet_target == {
            "companies": set(),
            "titles": {"Staff Engineer", "Research Scientist"},
        }

    asyncio.run(_body())


def test_reload_verify_passes_against_the_landed_set_after_a_miss():
    """The post-reload re-verification reads the owned target, so a string
    that ran with a recorded miss survives a panel-wedge reload."""

    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        page = MagicMock()
        page.reload = AsyncMock()
        page.locator.return_value.first.wait_for = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        browser._page = page
        landed: list[str] = []

        async def _apply_companies(values, **_kwargs):
            ok = True
            for value in values:
                if value == "Tess AI":
                    ok = False
                    continue
                landed.append(value)
            return ok

        browser.apply_company_filter = AsyncMock(side_effect=_apply_companies)
        browser.apply_title_filter = AsyncMock(return_value=True)
        browser.read_applied_company_chips = AsyncMock(
            side_effect=lambda: list(landed)
        )
        browser.read_applied_title_chips = AsyncMock(return_value=[])

        with patch(
            "linkedin.browser.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", True
        ):
            await browser.reconcile_facets(companies={"Acme", "Tess AI"}, titles=set())
            assert await browser.reload_search_page() is True

    asyncio.run(_body())


def _reconciled_plan(misses):
    return AdvancedSearchPlan(
        controls=[
            AdvancedSearchControl(dimension="keywords", values=["python"]),
            AdvancedSearchControl(dimension="job_titles", values=["Staff Engineer"]),
            AdvancedSearchControl(dimension="companies", values=["Acme"]),
            AdvancedSearchControl(dimension="companies", values=["Tess AI"]),
        ],
        keyword_boolean="python",
        reconciled_facet_misses=misses,
    )


def test_reconciled_facet_controls_are_not_typed_again():
    browser = _make_browser_with_enter_search()
    browser.apply_company_filter = AsyncMock(return_value=True)
    browser.apply_title_filter = AsyncMock(return_value=True)

    result = asyncio.run(
        apply_advanced_search_plan(browser, _reconciled_plan(frozenset()))
    )

    browser.apply_company_filter.assert_not_awaited()
    browser.apply_title_filter.assert_not_awaited()
    assert result.applied_controls == ["keywords", "job_titles", "companies", "companies"]
    assert result.failed_controls == []
    assert result.plan_fully_applied is True


def test_a_reconciled_facet_miss_reads_as_a_failed_control():
    browser = _make_browser_with_enter_search()
    browser.apply_company_filter = AsyncMock(return_value=True)
    browser.apply_title_filter = AsyncMock(return_value=True)

    result = asyncio.run(
        apply_advanced_search_plan(browser, _reconciled_plan(frozenset({"tess ai"})))
    )

    browser.apply_company_filter.assert_not_awaited()
    assert result.applied_controls == ["keywords", "job_titles", "companies"]
    assert result.failed_controls == ["companies"]
    assert result.plan_fully_applied is False
    assert result.success is False
    assert result.fallback_to_boolean is True


def test_a_plan_with_no_reconcile_types_its_facet_controls_as_before():
    """The flag-off default (None) leaves the apply path byte-for-byte."""
    browser = _make_browser_with_enter_search()
    browser.apply_company_filter = AsyncMock(return_value=True)
    browser.apply_title_filter = AsyncMock(return_value=True)
    plan = _reconciled_plan(frozenset())
    plan.reconciled_facet_misses = None

    result = asyncio.run(apply_advanced_search_plan(browser, plan))

    assert browser.apply_company_filter.await_count == 2
    browser.apply_title_filter.assert_awaited_once()
    assert result.plan_fully_applied is True


def test_facet_unapplied_console_line_is_one_prebuilt_line():
    from linkedin.browser import format_facet_unapplied_console_line

    line = format_facet_unapplied_console_line(
        40,
        {"unapplied": {"companies": ["Tess AI"], "titles": ["Staff Engineer"]}},
    )

    assert line == (
        "  [facet] string #40 unapplied: companies=['Tess AI'], "
        "titles=['Staff Engineer']"
    )
    assert "\n" not in line


def test_facet_unapplied_console_line_is_silent_when_everything_landed():
    from linkedin.browser import format_facet_unapplied_console_line

    assert (
        format_facet_unapplied_console_line(
            41, {"unapplied": {"companies": [], "titles": []}}
        )
        is None
    )
    assert format_facet_unapplied_console_line(42, {}) is None


def _on_this_projects_search():
    """Satisfy the URL half of the readiness gate.

    The offline page is a `data:` URL, which is not a Recruiter search page.
    These two cases are about the DOM; the URL half is driven for real against
    the table-driven rig in `tests/test_search_surface_readiness.py`.
    """
    return patch(
        "linkedin.evaluation_support._is_recruiter_search_page",
        return_value=True,
    )


def test_search_surface_readiness_requires_rail_and_results_signal():
    # A rail with facet markup in it, because the gate now requires the rail
    # and the qualifying element to be VISIBLE: an empty <aside> has no box,
    # and the live rail is never empty.
    html = """
    <aside class="left-rail"><div class="search-facet-wrapper">Filters</div></aside>
    <div class="search-query-summary__title">120 results</div>
    """

    async def _body(browser, _page):
        with _on_this_projects_search():
            await browser.assert_search_surface_ready(timeout_ms=0)

    _run_with_offline_page(html, _body)


def test_search_surface_readiness_fails_without_count_or_cards():
    async def _body(browser, _page):
        with _on_this_projects_search(), pytest.raises(
            FacetReconciliationError, match="search_surface_unready"
        ):
            await browser.assert_search_surface_ready(timeout_ms=0)

    _run_with_offline_page('<aside class="left-rail"></aside>', _body)


def test_reconcile_facets_verify_only_never_corrects_a_mismatch():
    async def _body(browser, page):
        browser.apply_company_filter = AsyncMock()
        browser.apply_title_filter = AsyncMock()
        browser._ghost_click_locator = AsyncMock()
        browser._page.wait_for_timeout = AsyncMock()

        with pytest.raises(FacetReconciliationError, match="postcondition_mismatch"):
            await browser.reconcile_facets(
                companies=set(),
                titles=set(),
                verify_only=True,
            )

        assert await page.evaluate("window.facetInteractions") == []
        browser._ghost_click_locator.assert_not_awaited()
        browser.apply_company_filter.assert_not_awaited()
        browser.apply_title_filter.assert_not_awaited()
        assert browser._page.wait_for_timeout.await_count == 30

    _run_with_offline_page(_FACET_RECONCILIATION_DOM, _body)


def test_reconcile_facets_verify_only_settles_transient_unreadable_rail():
    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser._page = MagicMock()
        browser._page.wait_for_timeout = AsyncMock()
        browser.read_applied_company_chips = AsyncMock(
            side_effect=[None, ["Acme"]]
        )
        browser.read_applied_title_chips = AsyncMock(return_value=["Engineer"])

        await browser.reconcile_facets(
            companies={"Acme"},
            titles={"Engineer"},
            verify_only=True,
        )

        browser._page.wait_for_timeout.assert_awaited_once_with(120)
        assert browser.last_facet_reconciliation["after"] == {
            "companies": ["Acme"],
            "titles": ["Engineer"],
        }

    asyncio.run(_body())


def test_facet_reconciliation_console_line_is_one_prebuilt_success_line():
    line = format_facet_reconciliation_console_line(
        17,
        {
            "before": {"companies": ["A", "B"], "titles": ["T1"]},
            "desired": {"companies": ["B", "C"], "titles": ["T2"]},
            "failure_gate": "",
        },
    )

    assert line == "  [facet] string #17: companies -1 +1, titles -1 +1"
    assert "\n" not in line


def test_facet_reconciliation_console_line_is_one_prebuilt_blocked_line():
    line = format_facet_reconciliation_console_line(
        18,
        {"failure_gate": "companies_unreadable"},
    )

    assert line == (
        "  [facet] BLOCKED string #18: companies_unreadable; no search entered"
    )
    assert "\n" not in line


# ---------------------------------------------------------------------------
# F9b — the empty-vs-unreadable heuristic must not be rotation-blind.
#
# "Wrapper absent + add-control present" is only evidence of an EMPTY facet if
# every pill on the rail can be attributed to some dimension wrapper. If pills
# sit on the rail that belong to no wrapper at all, the reader is looking at a
# DOM it does not understand: that is UNREADABLE, never empty. Returning [] there
# is the original contamination bug, silently, with no diagnostic.
# ---------------------------------------------------------------------------

_MISSING_COMPANY_WRAPPER_WITH_BARE_RAIL_PILLS = """
<aside class="left-rail">
  <button type="button">Companies or boolean</button>
  <div class="search-facet-wrapper facet-locations">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Brazil</span>
      <button aria-label="Remove Brazil" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
  <ul class="pills-list-section__list">
    <li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Nubank</span>
      <button aria-label="Remove Nubank" data-test-pill-dismiss></button></span>
    </li>
    <li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>CI&amp;T</span>
      <button aria-label="Remove CI&amp;amp;T" data-test-pill-dismiss></button></span>
    </li>
  </ul>
</aside>
"""

_EMPTY_COMPANY_WRAPPER_WITH_BARE_RAIL_PILLS = """
<aside class="left-rail">
  <button type="button">Companies or boolean</button>
  <div class="search-facet-wrapper facet-companies"></div>
  <ul class="pills-list-section__list">
    <li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Nubank</span>
      <button aria-label="Remove Nubank" data-test-pill-dismiss></button></span>
    </li>
  </ul>
</aside>
"""

_CLASSLESS_WRAPPER_HOLDS_THE_PILLS = """
<aside class="left-rail">
  <button type="button">Companies or boolean</button>
  <div class="search-facet-wrapper">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Nubank</span>
      <button aria-label="Remove Nubank" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
</aside>
"""

_UNMANAGED_DIMENSION_WRAPPER_IS_ATTRIBUTABLE = """
<aside class="left-rail">
  <button type="button">Companies or boolean</button>
  <button type="button">Job titles or boolean</button>
  <div class="search-facet-wrapper facet-years-of-experience">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>8 to 30+ years</span>
      <button aria-label="Remove 8 to 30+ years" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
  <div class="search-facet-wrapper facet-locations">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Brazil</span>
      <button aria-label="Remove Brazil" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
</aside>
"""


def test_missing_wrapper_with_unattributable_rail_pills_is_unreadable():
    """A rotated wrapper class reads as cleanly-empty today. It must not.

    The dimension wrapper is gone and the add-control is present — today's
    "empty" evidence — but two company pills are live on the rail under no
    wrapper at all. Claiming [] here is the contamination bug: the next string
    would run inside those two companies while believing nothing is applied.
    """

    async def _body(browser, page):
        assert await page.locator(
            "div.search-facet-wrapper.facet-companies"
        ).count() == 0
        assert await page.locator("li[data-test-facet-pills-item]").count() == 3
        assert await browser.read_applied_company_chips() is None

    _run_with_offline_page(_MISSING_COMPANY_WRAPPER_WITH_BARE_RAIL_PILLS, _body)


def test_empty_wrapper_with_unattributable_rail_pills_is_unreadable():
    """The present-but-empty wrapper path only warns today; it must fail closed."""

    async def _body(browser, _page):
        assert await browser.read_applied_company_chips() is None

    _run_with_offline_page(_EMPTY_COMPANY_WRAPPER_WITH_BARE_RAIL_PILLS, _body)


def test_wrapper_without_a_dimension_class_is_unattributable():
    """A wrapper LinkedIn stopped tagging with facet-<dimension> is a rotation.

    The pills are inside a search-facet-wrapper, so the old diagnostic skipped
    them entirely, and no facet-<dimension> wrapper exists to read them from.
    """

    async def _body(browser, _page):
        assert await browser.read_applied_company_chips() is None

    _run_with_offline_page(_CLASSLESS_WRAPPER_HOLDS_THE_PILLS, _body)


def test_unmanaged_dimension_pills_stay_attributable_and_read_empty(caplog):
    """A facet Cloris does not manage is not a rotation — it must not fail closed.

    Sam applies filters by hand (a manual 8-30+ YOE band on PAE, for one). Those
    pills carry their own facet-<dimension> wrapper, so they are attributable and
    Companies still reads as a legitimately empty facet, silently.
    """

    async def _body(browser, _page):
        with caplog.at_level(logging.WARNING, logger="linkedin.browser"):
            assert await browser.read_applied_company_chips() == []
            assert await browser.read_applied_title_chips() == []
        assert caplog.text == ""

    _run_with_offline_page(_UNMANAGED_DIMENSION_WRAPPER_IS_ATTRIBUTABLE, _body)


_POPULATED_WRAPPER_ALONGSIDE_BARE_RAIL_PILLS = """
<aside class="left-rail">
  <button type="button">Companies or boolean</button>
  <div class="search-facet-wrapper facet-companies">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Nubank</span>
      <button aria-label="Remove Nubank" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
  <ul class="pills-list-section__list">
    <li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>iFood</span>
      <button aria-label="Remove iFood" data-test-pill-dismiss></button></span>
    </li>
  </ul>
</aside>
"""

_HIDDEN_BARE_RAIL_PILL_IS_NOT_APPLIED = """
<aside class="left-rail">
  <button type="button">Companies or boolean</button>
  <div class="search-facet-wrapper facet-companies">
    <ul><li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Nubank</span>
      <button aria-label="Remove Nubank" data-test-pill-dismiss></button></span>
    </li></ul>
  </div>
  <ul class="pills-list-section__list" style="display:none">
    <li data-test-facet-pills-item>
      <span class="facet-pill"><span data-test-pill-label>Template</span>
      <button aria-label="Remove Template" data-test-pill-dismiss></button></span>
    </li>
  </ul>
</aside>
"""


def test_populated_wrapper_is_still_unreadable_beside_unattributable_pills():
    """A partial read is the same silent under-report as a false empty.

    The wrapper answers "one company"; a second company pill is live on the rail
    outside it. Returning the one would leave the other in the applied union and
    invisible to reconciliation, which is the contamination bug with extra steps.
    """

    async def _body(browser, _page):
        assert await browser.read_applied_company_chips() is None

    _run_with_offline_page(_POPULATED_WRAPPER_ALONGSIDE_BARE_RAIL_PILLS, _body)


def test_hidden_rail_pills_never_trip_the_attribution_gate(caplog):
    """Only rendered pills are applied filters; a display:none node is not."""

    async def _body(browser, _page):
        with caplog.at_level(logging.WARNING, logger="linkedin.browser"):
            assert await browser.read_applied_company_chips() == ["Nubank"]
        assert caplog.text == ""

    _run_with_offline_page(_HIDDEN_BARE_RAIL_PILL_IS_NOT_APPLIED, _body)


# ---------------------------------------------------------------------------
# Near-miss takes (Sam's ruling 2026-09-20). The typeahead answers a
# strategist's spelling with LinkedIn's canonical entity name; the run takes
# the near miss and records the SUBSTITUTION, so every downstream verification
# compares against what actually landed.
# ---------------------------------------------------------------------------


def test_apply_company_filter_takes_the_near_miss_and_confirms_the_landed_pill():
    """RED against exact-only: "Mercado Livre" missed while LinkedIn offered
    "Mercado Livre Brasil" first. The take must confirm the LANDED pill text
    (the requested text never appears on the rail) and record the mapping."""

    async def _body():
        state = _FakeLocationState(
            offered_options=[
                "Mercado Livre Brasil",
                "Mercado Libre",
                "RBE | Mercado Livre de Energia",
            ],
        )
        browser = _make_fake_location_browser(state)

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        ok = await browser.apply_company_filter(["Mercado Livre"])

        assert ok is True
        assert state.applied_chips == {"Mercado Livre Brasil"}
        assert browser.last_company_substitutions == {
            "Mercado Livre": "Mercado Livre Brasil"
        }
        assert browser.last_company_option_misses == {}

    asyncio.run(_body())


def test_apply_title_filter_takes_the_prefix_entity_and_records_it():
    """A prefix is taken only when nothing on the list CONTAINS the request.

    ["Tess", "TESS AI Quality"] is the Addendum G5 capture, and the container
    wins there whichever row LinkedIn put first — it keeps the "AI" the
    strategist asked for. Drop the container and the prefix is the only
    evidence left, which is the take this pins.
    """

    async def _body():
        state = _FakeLocationState(offered_options=["Tess", "Tess Energia"])
        browser = _make_fake_location_browser(state)

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        ok = await browser.apply_title_filter(["Tess AI"])

        assert ok is True
        assert state.applied_chips == {"Tess"}
        assert browser.last_title_substitutions == {"Tess AI": "Tess"}

    asyncio.run(_body())


def test_apply_title_filter_prefers_the_container_over_the_prefix():
    """The G5 listbox itself: the QA sub-brand keeps both requested tokens."""

    async def _body():
        state = _FakeLocationState(offered_options=["Tess", "TESS AI Quality"])
        browser = _make_fake_location_browser(state)

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        ok = await browser.apply_title_filter(["Tess AI"])

        assert ok is True
        assert state.applied_chips == {"TESS AI Quality"}
        assert browser.last_title_substitutions == {"Tess AI": "TESS AI Quality"}

    asyncio.run(_body())


def test_apply_company_filter_exact_option_records_no_substitution():
    async def _body():
        state = _FakeLocationState(offered_options=["Stripe", "Stripe Atlas"])
        browser = _make_fake_location_browser(state)

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        assert await browser.apply_company_filter(["Stripe"]) is True
        assert browser.last_company_substitutions == {}

    asyncio.run(_body())


def test_apply_company_filter_empty_listbox_retries_four_times_then_misses():
    """The retry budget belongs to an EMPTY listbox, not to a populated one.

    A listbox that shows options and matches none of them has answered; a
    listbox that shows nothing may still be loading, so the typing is retried
    4x as before.
    """

    async def _body():
        state = _FakeLocationState(offered_options=[], editor_value="previous miss")
        browser = _make_fake_location_browser(state)
        events = []

        async def _clear(locator):
            events.append("clear")
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            events.append(("type", text))
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        with patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter(["Stripe"])

        assert ok is False
        assert events == ["clear", ("type", "Stripe")] * 4 + ["clear"]
        assert browser.last_company_option_misses == {"Stripe": []}
        assert browser.last_company_substitutions == {}

    asyncio.run(_body())


def test_apply_company_filter_populated_listbox_that_matches_nothing_misses_once():
    async def _body():
        state = _FakeLocationState(
            offered_options=["Striped Fabrics", "Strip Mall"],
            editor_value="previous miss",
        )
        browser = _make_fake_location_browser(state)
        events = []

        async def _clear(locator):
            events.append("clear")
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            events.append(("type", text))
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        with patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter(["Stripe"])

        assert ok is False
        assert events == ["clear", ("type", "Stripe"), "clear"]
        assert browser.last_company_option_misses == {
            "Stripe": ["Striped Fabrics", "Strip Mall"]
        }

    asyncio.run(_body())


def test_reconcile_receipt_carries_the_substitution_and_owns_the_landed_name():
    """RED against the add-only receipt: ``substituted`` did not exist, the
    owned target held the REQUESTED name, and ``_satisfied`` compared the
    landed pill against a name the rail will never carry — so a near-miss take
    would have failed its own postcondition."""

    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser._page = MagicMock()
        browser._page.wait_for_timeout = AsyncMock()
        landed: list[str] = []

        async def _apply_companies(values, **_kwargs):
            browser.last_company_substitutions = {}
            for value in values:
                if value == "Mercado Livre":
                    landed.append("Mercado Livre Brasil")
                    browser.last_company_substitutions[value] = "Mercado Livre Brasil"
                else:
                    landed.append(value)
            return True

        browser.apply_company_filter = AsyncMock(side_effect=_apply_companies)
        browser.apply_title_filter = AsyncMock(return_value=True)
        browser.read_applied_company_chips = AsyncMock(side_effect=lambda: list(landed))
        browser.read_applied_title_chips = AsyncMock(return_value=[])

        await browser.reconcile_facets(
            companies={"Mercado Livre", "Nubank"}, titles=set()
        )

        receipt = browser.last_facet_reconciliation
        assert receipt["failure_gate"] == ""
        assert receipt["substituted"]["companies"] == {
            "Mercado Livre": "Mercado Livre Brasil"
        }
        assert receipt["substituted"]["titles"] == {}
        assert receipt["desired"]["companies"] == ["Mercado Livre", "Nubank"]
        assert sorted(receipt["after"]["companies"]) == [
            "Mercado Livre Brasil",
            "Nubank",
        ]
        assert receipt["unapplied"]["companies"] == []
        assert browser._owned_facet_target == {
            "companies": {"Mercado Livre Brasil", "Nubank"},
            "titles": set(),
        }

        # The verify-only reconcile the opening search runs next is driven
        # through facet_target_landed, which maps the request to the landed
        # name. Re-asserting the REQUEST here would fail every near miss.
        from linkedin.browser import facet_target_landed

        await browser.reconcile_facets(
            companies=facet_target_landed(
                {"Mercado Livre", "Nubank"}, receipt, "companies"
            ),
            titles=facet_target_landed(set(), receipt, "titles"),
            verify_only=True,
        )
        assert browser.last_facet_reconciliation["failure_gate"] == ""

    asyncio.run(_body())


def test_two_requested_names_landing_on_one_pill_is_not_two_applied_values():
    """Requested {"Tess", "Tess AI"}; LinkedIn has one entity for both.

    "Tess" lands exactly and "Tess AI" prefix-lands the same pill, so the two
    desired keys collapse in the target dict and the rail carries one pill.
    Key-set equality then read as satisfied with an empty failure gate and a
    receipt still listing both names, while the search was constrained by one
    entity — a MISSING entity dressed as a clean reconcile.

    RED against key-set equality: ``unapplied`` was empty and nothing named
    the collapse.
    """

    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser._page = MagicMock()
        browser._page.wait_for_timeout = AsyncMock()
        landed: list[str] = []

        async def _apply_companies(values, **_kwargs):
            browser.last_company_substitutions = {}
            for value in values:
                if value == "Tess AI":
                    browser.last_company_substitutions[value] = "Tess"
                if "Tess" not in landed:
                    landed.append("Tess")
            return True

        browser.apply_company_filter = AsyncMock(side_effect=_apply_companies)
        browser.apply_title_filter = AsyncMock(return_value=True)
        browser.read_applied_company_chips = AsyncMock(side_effect=lambda: list(landed))
        browser.read_applied_title_chips = AsyncMock(return_value=[])

        await browser.reconcile_facets(companies={"Tess", "Tess AI"}, titles=set())

        receipt = browser.last_facet_reconciliation
        assert receipt["failure_gate"] == ""
        assert receipt["after"]["companies"] == ["Tess"]
        assert receipt["unapplied"]["companies"] == ["Tess AI"]
        assert receipt["substituted"]["companies"] == {}
        assert receipt["collisions"]["companies"] == [["Tess", "Tess AI", "Tess"]]
        assert browser._owned_facet_target == {
            "companies": {"Tess"},
            "titles": set(),
        }

        from linkedin.browser import format_facet_unapplied_console_line

        line = format_facet_unapplied_console_line(44, receipt)
        assert line == (
            "  [facet] string #44 unapplied: companies=['Tess AI'] "
            '(one pill for two names: "Tess"+"Tess AI"→"Tess")'
        )
        assert "\n" not in line

    asyncio.run(_body())


def test_reconcile_drops_a_seeded_substitution_when_the_value_ends_unapplied():
    """The applier is authoritative: a value that misses is ``unapplied``, and
    no substitution survives for it."""

    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser._page = MagicMock()
        browser._page.wait_for_timeout = AsyncMock()
        browser.apply_company_filter = AsyncMock(return_value=False)
        browser.apply_title_filter = AsyncMock(return_value=True)
        browser.read_applied_company_chips = AsyncMock(return_value=[])
        browser.read_applied_title_chips = AsyncMock(return_value=[])
        browser._facet_resolution_cache_put("companies", "Tess AI", "Tess")

        await browser.reconcile_facets(companies={"Tess AI"}, titles=set())

        receipt = browser.last_facet_reconciliation
        assert receipt["unapplied"]["companies"] == ["Tess AI"]
        assert receipt["substituted"]["companies"] == {}
        assert browser._owned_facet_target == {"companies": set(), "titles": set()}

    asyncio.run(_body())


def test_reconcile_retains_a_cached_landed_pill_without_removing_it():
    """A second string asking for "Mercado Livre" must RETAIN the
    "Mercado Livre Brasil" pill the first string landed, not dismiss it as
    foreign and type it again."""

    async def _body(browser, page):
        await _append_facet_pill(page, "companies", "Mercado Livre Brasil")
        browser._facet_resolution_cache_put(
            "companies", "Mercado Livre", "Mercado Livre Brasil"
        )
        browser.apply_company_filter = AsyncMock(return_value=True)
        browser.apply_title_filter = AsyncMock(return_value=True)

        await browser.reconcile_facets(companies={"Mercado Livre"}, titles=set())

        interactions = await page.evaluate("window.facetInteractions")
        assert [event for event in interactions if event.startswith("remove:companies")] == []
        browser.apply_company_filter.assert_not_awaited()
        receipt = browser.last_facet_reconciliation
        assert receipt["substituted"]["companies"] == {
            "Mercado Livre": "Mercado Livre Brasil"
        }
        assert receipt["after"]["companies"] == ["Mercado Livre Brasil"]

    _run_with_offline_page(_FACET_RECONCILIATION_DOM_EMPTY_MANAGED, _body)


def test_reconcile_keeps_a_rail_pill_that_differs_only_by_its_diacritics():
    """"Itaú" on the rail IS the request "Itau", cold cache and all.

    ``facet_name_tokens`` already folds combining marks, which is why the
    typeahead exact-matches the row — but the reconcile compared rail pills
    with a plain casefold, so the correct pill read as foreign, the remove
    phase dismissed it, and the string had to type it back. A dismiss that
    failed would have ended the string ``remove_failed`` while it held the
    company it asked for.
    """

    async def _body(browser, page):
        await _append_facet_pill(page, "companies", "Itaú Unibanco")
        browser.apply_company_filter = AsyncMock(return_value=True)
        browser.apply_title_filter = AsyncMock(return_value=True)

        await browser.reconcile_facets(companies={"Itau Unibanco"}, titles=set())

        interactions = await page.evaluate("window.facetInteractions")
        assert [
            event for event in interactions if event.startswith("remove:")
        ] == []
        browser.apply_company_filter.assert_not_awaited()
        receipt = browser.last_facet_reconciliation
        assert receipt["failure_gate"] == ""
        assert receipt["after"]["companies"] == ["Itaú Unibanco"]

    _run_with_offline_page(_FACET_RECONCILIATION_DOM_EMPTY_MANAGED, _body)


def test_apply_company_filter_writes_and_then_spends_the_resolution_cache(tmp_path):
    """First apply persists LinkedIn's spelling to the state-dir sidecar; the
    second apply types THAT and matches exactly — no near-miss search, and the
    substitution is still recorded so the receipt stays honest."""

    async def _body():
        state = _FakeLocationState(offered_options=["Mercado Livre Brasil"])
        browser = _make_fake_location_browser(state)
        browser._state_dir = tmp_path
        typed: list[str] = []

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            typed.append(text)
            locator.state.editor_value += text

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        assert await browser.apply_company_filter(["Mercado Livre"]) is True
        assert typed == ["Mercado Livre"]

        from linkedin.browser import read_facet_name_resolutions

        assert read_facet_name_resolutions(tmp_path)["companies"] == {
            "mercado livre": "Mercado Livre Brasil"
        }

        # Second string, fresh browser reading the same sidecar, clean rail.
        second_state = _FakeLocationState(offered_options=["Mercado Livre Brasil"])
        second = _make_fake_location_browser(second_state)
        second._state_dir = tmp_path
        second_typed: list[str] = []

        async def _type_second(_page, locator, text, *, plan):
            second_typed.append(text)
            locator.state.editor_value += text

        second._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        second._input_backend.type_text = AsyncMock(side_effect=_type_second)

        assert await second.apply_company_filter(["Mercado Livre"]) is True
        assert second_typed == ["Mercado Livre Brasil"]
        assert second.last_company_substitutions == {
            "Mercado Livre": "Mercado Livre Brasil"
        }

    asyncio.run(_body())


def test_apply_company_filter_falls_back_to_the_request_when_the_cache_is_stale(
    tmp_path,
):
    """A cached spelling LinkedIn no longer offers must not permanently block
    a value the typeahead would answer verbatim today. The cached name gets ONE
    typing attempt — it is an optimization, not a budget — and the request gets
    the usual four."""

    async def _body():
        state = _FakeLocationState(offered_options=[], visible_options=[])
        browser = _make_fake_location_browser(state)
        browser._state_dir = tmp_path
        browser._facet_resolution_cache_put(
            "companies", "Mercado Livre", "Mercado Livre Brasil Holdings"
        )
        typed: list[str] = []

        async def _clear(locator):
            locator.state.editor_value = ""

        async def _type(_page, locator, text, *, plan):
            typed.append(text)
            locator.state.editor_value += text
            # LinkedIn has retired the cached entity; only the request is known.
            if text == "Mercado Livre":
                state.offered_options = ["Mercado Livre"]
                state.visible_options = {"Mercado Livre"}
            else:
                state.offered_options = []
                state.visible_options = set()

        browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
        browser._input_backend.type_text = AsyncMock(side_effect=_type)

        with patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter(["Mercado Livre"])

        assert ok is True
        assert typed == ["Mercado Livre Brasil Holdings", "Mercado Livre"]
        assert browser.last_company_substitutions == {}
        from linkedin.browser import read_facet_name_resolutions

        assert read_facet_name_resolutions(tmp_path)["companies"] == {}

    asyncio.run(_body())


def test_apply_company_filter_skips_typing_when_the_cached_pill_is_already_applied():
    async def _body():
        state = _FakeLocationState(offered_options=[])
        browser = _make_fake_location_browser(state)
        browser._facet_resolution_cache_put(
            "companies", "Mercado Livre", "Mercado Livre Brasil"
        )
        browser.read_applied_company_chips = AsyncMock(
            return_value=["Mercado Livre Brasil"]
        )
        browser._clear_keyword_textarea = AsyncMock()
        browser._input_backend.type_text = AsyncMock()

        ok = await browser.apply_company_filter(["Mercado Livre"])

        assert ok is True
        assert browser.last_company_already_applied_count == 1
        assert browser.last_company_substitutions == {
            "Mercado Livre": "Mercado Livre Brasil"
        }
        browser._input_backend.type_text.assert_not_awaited()

    asyncio.run(_body())


def test_apply_company_filter_rereads_a_listbox_still_showing_the_last_value():
    """The rendered list can still be the PREVIOUS value's when the poll fires.

    Nothing in it satisfies this request, so the take waits for the repaint and
    reads again rather than recording a miss against a stale list. No retyping:
    the query is already correct.

    The stale rows answer the pre-typing snapshot AND the first read after
    typing, which is the live shape — an earlier rig served them only to the
    snapshot, so the post-typing read was already the repainted list and the
    test stayed green with the whole repaint wait deleted.
    """

    async def _body():
        browser, state, patcher = _scripted_listbox_browser(
            [
                ["Nubank", "Nubank Pagamentos"],
                ["Nubank", "Nubank Pagamentos"],
                ["Mercado Livre Brasil"],
            ]
        )

        with patcher, patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter(["Mercado Livre"])

        assert ok is True
        assert state.applied_chips == {"Mercado Livre Brasil"}
        assert browser._input_backend.type_text.await_count == 1
        assert browser.last_company_substitutions == {
            "Mercado Livre": "Mercado Livre Brasil"
        }

    asyncio.run(_body())


def _scripted_listbox_browser(
    script: list[list[str]],
    *,
    before_clear: list[str] | None = None,
    after_clear: list[str] | None = None,
    populates: bool | None = None,
):
    """A rig whose option listbox answers by PHASE, then by read.

    ``script[0]`` is what the listbox holds BEFORE this name is typed, and it
    answers every pre-typing read. ``script[1]`` answers the first read after
    typing and each later read consumes the next entry; the last entry answers
    every read after it. Keying the pre-typing rows to the phase rather than to
    a read index is what lets one case mean the same thing however many reads
    the take makes there — a script keyed by index says something different
    the moment a read is added or removed.

    ``before_clear`` and ``after_clear`` split that first phase for the case it
    exists to describe: clearing the editor CLOSES the typeahead, so the read
    after the clear is empty while the read before it still carries the
    previous value's rows.

    Every served entry becomes the rig's live option list, so a click resolves
    against the rows that were read. ``populates`` forces the visibility poll's
    verdict, which the rig otherwise serves from whatever the last read left
    behind — a listbox the clear closed and the typing re-opened has to say so
    here.
    """
    pre_clear_rows = list(script[0] if before_clear is None else before_clear)
    post_clear_rows = list(script[0] if after_clear is None else after_clear)
    typed_reads = script[1:] or [script[0]]
    state = _FakeLocationState(offered_options=list(pre_clear_rows))
    browser = _make_fake_location_browser(state)
    phase = "before_clear"
    reads = 0
    original = _FakeLocationLocator.all_inner_texts

    async def _all_inner_texts(self):
        nonlocal reads
        if self.kind != "options":
            return await original(self)
        if phase == "before_clear":
            served = pre_clear_rows
        elif phase == "after_clear":
            served = post_clear_rows
        else:
            served = typed_reads[min(reads, len(typed_reads) - 1)]
            reads += 1
        self.state.offered_options = list(served)
        self.state.visible_options = set(served)
        return list(served)

    async def _clear(locator):
        nonlocal phase
        locator.state.editor_value = ""
        phase = "after_clear"

    async def _type(_page, locator, text, *, plan):
        nonlocal phase
        locator.state.editor_value += text
        phase = "after_typing"

    browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
    browser._input_backend.type_text = AsyncMock(side_effect=_type)
    if populates is not None:
        browser._wait_for_visible_poll = AsyncMock(return_value=populates)
    return browser, state, patch.object(
        _FakeLocationLocator, "all_inner_texts", _all_inner_texts
    )


_BANCO_ROWS = ["Banco", "Banco Inter"]


def _take_from_scripted_listbox(script: list[list[str]], tmp_path, **rig):
    """Ask for "Banco do Brasil" against a scripted listbox. Returns the rig."""

    async def _body():
        browser, state, patcher = _scripted_listbox_browser(script, **rig)
        browser._state_dir = tmp_path
        with patcher, patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter(["Banco do Brasil"])
        return ok, browser, state

    return asyncio.run(_body())


def test_a_listbox_that_blanks_and_returns_the_same_names_never_repainted(tmp_path):
    """A typeahead unmounts and remounts the rows it already had.

    One empty read used to force the verdict: the read after it counted as this
    name's answer even when it carried the same rows, so "Banco" was clicked
    for "Banco do Brasil", confirmed, and written to the sidecar — and every
    later string retained the wrong pill without asking the typeahead again.

    RED against `or saw_empty`: ok was True, the pill was "Banco", and the
    cache carried "banco do brasil" -> "Banco".
    """
    ok, browser, state = _take_from_scripted_listbox(
        [_BANCO_ROWS, _BANCO_ROWS, [], _BANCO_ROWS], tmp_path
    )

    assert ok is False
    assert state.applied_chips == set()
    assert browser.last_company_substitutions == {}
    from linkedin.browser import read_facet_name_resolutions

    assert read_facet_name_resolutions(tmp_path)["companies"] == {}


def test_the_same_names_in_a_different_order_are_not_a_repaint(tmp_path):
    """Repaint means the offered NAMES changed, not that the list moved.

    A raw list inequality made LinkedIn's own reordering of the rows it already
    had count as this name's answer.
    """
    ok, browser, state = _take_from_scripted_listbox(
        [_BANCO_ROWS, _BANCO_ROWS, ["Banco Inter", "Banco"]], tmp_path
    )

    assert ok is False
    assert state.applied_chips == set()
    from linkedin.browser import read_facet_name_resolutions

    assert read_facet_name_resolutions(tmp_path)["companies"] == {}


def test_a_changed_secondary_line_is_not_a_repaint(tmp_path):
    """`all_inner_texts` carries a row's employee count, which ticks over.

    The entity is the first line, and the first lines here are unchanged: this
    is still the previous value's list.
    """
    ok, browser, state = _take_from_scripted_listbox(
        [
            ["Banco\n12,000 employees", "Banco Inter"],
            ["Banco\n12,000 employees", "Banco Inter"],
            ["Banco\n12,001 employees", "Banco Inter"],
        ],
        tmp_path,
    )

    assert ok is False
    assert state.applied_chips == set()
    from linkedin.browser import read_facet_name_resolutions

    assert read_facet_name_resolutions(tmp_path)["companies"] == {}


def test_a_new_name_set_is_the_repaint_the_near_miss_waits_for(tmp_path):
    """The one thing that qualifies a near miss: names this list did not have."""
    ok, browser, state = _take_from_scripted_listbox(
        [_BANCO_ROWS, _BANCO_ROWS, ["Banco do Brasil S.A."]], tmp_path
    )

    assert ok is True
    assert state.applied_chips == {"Banco do Brasil S.A."}
    assert browser.last_company_substitutions == {
        "Banco do Brasil": "Banco do Brasil S.A."
    }


def test_the_rows_that_flash_back_after_the_clear_are_not_this_names_answer(
    tmp_path,
):
    """Clearing the editor can CLOSE the listbox, so the baseline is read on
    both sides of the clear.

    A baseline read only after the clear is empty, and the previous value's
    rows flashing back differ from empty — so the wait was skipped, "Banco"
    was clicked for "Banco do Brasil", confirmed, and written to the sidecar
    as the spelling LinkedIn landed. Every later string then retained that
    pill without asking the typeahead again.

    RED against the post-clear-only baseline: ok was True, the pill was
    "Banco", and the cache carried "banco do brasil" -> "Banco".
    """
    ok, browser, state = _take_from_scripted_listbox(
        [_BANCO_ROWS, _BANCO_ROWS],
        tmp_path,
        after_clear=[],
        populates=True,
    )

    assert ok is False
    assert state.applied_chips == set()
    assert browser.last_company_substitutions == {}
    from linkedin.browser import read_facet_name_resolutions

    assert read_facet_name_resolutions(tmp_path)["companies"] == {}


def test_a_superset_of_the_old_rows_does_not_qualify_an_old_row(tmp_path):
    """One row arriving is a changed set, and a changed set was the licence.

    What decides a near miss is whether the row the rule PICKED is new, not
    whether the list around it moved: LinkedIn mounting "Itaú" made "Banco"
    this name's answer.

    RED against the set-difference licence: ok was True and the pill was
    "Banco".
    """
    ok, browser, state = _take_from_scripted_listbox(
        [_BANCO_ROWS, [*_BANCO_ROWS, "Itaú"]], tmp_path
    )

    assert ok is False
    assert state.applied_chips == set()
    assert browser.last_company_substitutions == {}
    from linkedin.browser import read_facet_name_resolutions

    assert read_facet_name_resolutions(tmp_path)["companies"] == {}


def test_a_new_name_beside_the_old_rows_is_taken(tmp_path):
    """The other side of the rule: absence from the baseline is what licenses
    a near miss, and it licenses one the moment it holds — the old rows still
    standing beside the new one do not withhold it.

    This is the lock on not over-correcting into refusing every near miss.
    """
    ok, browser, state = _take_from_scripted_listbox(
        [_BANCO_ROWS, [*_BANCO_ROWS, "Banco do Brasil S.A."]], tmp_path
    )

    assert ok is True
    assert state.applied_chips == {"Banco do Brasil S.A."}
    assert browser.last_company_substitutions == {
        "Banco do Brasil": "Banco do Brasil S.A."
    }


def test_a_blank_read_before_the_new_list_is_not_a_miss(tmp_path):
    """An empty read is not evidence about this name in EITHER direction.

    A blank first read against a non-empty baseline counted as the repaint,
    the wait never ran, and the name the list was about to offer was recorded
    as a miss — so the next string typed it again.

    RED against the empty-read repaint: ok was False and no pill applied.
    """
    ok, browser, state = _take_from_scripted_listbox(
        [_BANCO_ROWS, [], ["Banco do Brasil S.A."]], tmp_path
    )

    assert ok is True
    assert state.applied_chips == {"Banco do Brasil S.A."}
    assert browser.last_company_substitutions == {
        "Banco do Brasil": "Banco do Brasil S.A."
    }
    from linkedin.browser import read_facet_name_resolutions

    assert read_facet_name_resolutions(tmp_path)["companies"] == {
        "banco do brasil": "Banco do Brasil S.A."
    }


def _stale_listbox_browser(stale: list[str], repainted: list[str] | None):
    """A rig whose listbox still holds the PREVIOUS value's rows.

    ``all_inner_texts`` answers ``stale`` for the pre-typing snapshot and the
    first read after typing, then ``repainted`` (or ``stale`` forever, when
    that is None) for every read after. Returns the browser, the state and a
    restore-on-exit patcher.
    """
    state = _FakeLocationState(offered_options=list(stale))
    browser = _make_fake_location_browser(state)
    reads = 0
    original = _FakeLocationLocator.all_inner_texts

    async def _all_inner_texts(self):
        nonlocal reads
        if self.kind != "options":
            return await original(self)
        reads += 1
        if reads <= 2 or repainted is None:
            return list(stale)
        self.state.offered_options = list(repainted)
        self.state.visible_options = set(repainted)
        return list(repainted)

    async def _clear(locator):
        locator.state.editor_value = ""

    async def _type(_page, locator, text, *, plan):
        locator.state.editor_value += text

    browser._clear_keyword_textarea = AsyncMock(side_effect=_clear)
    browser._input_backend.type_text = AsyncMock(side_effect=_type)
    return browser, state, patch.object(
        _FakeLocationLocator, "all_inner_texts", _all_inner_texts
    )


def test_a_near_miss_is_refused_from_a_listbox_that_never_repainted(tmp_path):
    """The rows on the page answer the name that was typed BEFORE this one.

    The applier has just taken "Itaú Unibanco" and the listbox still shows
    ["Banco", "Banco Inter"]. "Banco" folds to a four-character prefix of
    "Banco do Brasil", so the first-read take clicked a company nobody asked
    for, confirmed its pill, and wrote "banco do brasil" -> "Banco" to the
    sidecar — after which every later string retained the wrong pill without
    asking the typeahead again.

    RED against the first-read take: ``ok`` was True, the pill was "Banco",
    and the cache carried the wrong spelling.
    """

    async def _body():
        browser, state, patcher = _stale_listbox_browser(
            ["Banco", "Banco Inter"], None
        )
        browser._state_dir = tmp_path

        with patcher, patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter(["Banco do Brasil"])

        assert ok is False
        assert state.applied_chips == set()
        assert browser.last_company_substitutions == {}
        assert browser.last_company_option_misses == {
            "Banco do Brasil": ["Banco", "Banco Inter"]
        }
        from linkedin.browser import read_facet_name_resolutions

        assert read_facet_name_resolutions(tmp_path)["companies"] == {}

    asyncio.run(_body())


def test_a_near_miss_is_taken_once_the_listbox_repaints_for_this_name(tmp_path):
    """The same stale rows, and the typeahead answers a moment later.

    Waiting for the repaint is what makes the near miss safe, so the value
    still lands — under LinkedIn's spelling, recorded and cached.
    """

    async def _body():
        browser, state, patcher = _stale_listbox_browser(
            ["Banco", "Banco Inter"], ["Banco do Brasil S.A."]
        )
        browser._state_dir = tmp_path

        with patcher, patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter(["Banco do Brasil"])

        assert ok is True
        assert state.applied_chips == {"Banco do Brasil S.A."}
        assert browser.last_company_substitutions == {
            "Banco do Brasil": "Banco do Brasil S.A."
        }
        assert browser._input_backend.type_text.await_count == 1
        from linkedin.browser import read_facet_name_resolutions

        assert read_facet_name_resolutions(tmp_path)["companies"] == {
            "banco do brasil": "Banco do Brasil S.A."
        }

    asyncio.run(_body())


def test_an_exact_row_is_taken_from_a_listbox_that_never_repainted():
    """An exact row cannot be the right answer to a different query.

    The stale-list guard costs nothing here: if the rows the previous value
    left behind happen to carry this name verbatim, it is this name's pill.
    """

    async def _body():
        browser, state, patcher = _stale_listbox_browser(
            ["Nubank", "Nubank Pagamentos"], None
        )

        with patcher, patch("linkedin.browser.asyncio.sleep", AsyncMock()):
            ok = await browser.apply_company_filter(["Nubank"])

        assert ok is True
        assert state.applied_chips == {"Nubank"}
        assert browser.last_company_substitutions == {}

    asyncio.run(_body())


def test_reconcile_forgets_the_cached_spelling_that_collapsed_two_names(tmp_path):
    """A sidecar that answers two requested names with one pill is stale.

    "Tess" and "Tess AI" both map to "Tess", so the collision pass drops the
    second name before the typeahead is ever asked — and left the sidecar
    entry in place, so every later string dropped it again without asking
    LinkedIn whether it now has an entity for it. The entry that caused the
    collapse is forgotten; the name that owns the pill keeps its own.

    RED against the pre-apply collision branch: the entry survived and the
    next string dropped "Tess AI" unasked.
    """

    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser._page = MagicMock()
        browser._page.wait_for_timeout = AsyncMock()
        browser._state_dir = tmp_path
        browser.apply_company_filter = AsyncMock(return_value=True)
        browser.apply_title_filter = AsyncMock(return_value=True)
        browser.read_applied_company_chips = AsyncMock(return_value=["Tess"])
        browser.read_applied_title_chips = AsyncMock(return_value=[])
        browser._facet_resolution_cache_put("companies", "Tess AI", "Tess")

        await browser.reconcile_facets(
            companies={"Tess", "Tess AI"}, titles=set()
        )

        receipt = browser.last_facet_reconciliation
        assert receipt["failure_gate"] == ""
        assert receipt["unapplied"]["companies"] == ["Tess AI"]
        assert receipt["collisions"]["companies"] == [["Tess", "Tess AI", "Tess"]]
        # The pill was never typed for, so the typeahead was never asked.
        browser.apply_company_filter.assert_not_awaited()
        assert browser._facet_resolution_cache_get("companies", "Tess AI") is None
        from linkedin.browser import read_facet_name_resolutions

        assert read_facet_name_resolutions(tmp_path)["companies"] == {}

    asyncio.run(_body())


def test_reconcile_forgets_a_cached_spelling_the_rail_refused(tmp_path):
    """A seeded resolution the typeahead no longer honors must not persist.

    The miss path already drops the in-memory substitution — the rail beats
    the cache — but left the sidecar entry in place, so the next string typed
    the same dead spelling first and spent an attempt on it again.
    """

    async def _body():
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        browser._page = MagicMock()
        browser._page.wait_for_timeout = AsyncMock()
        browser._state_dir = tmp_path
        browser.apply_company_filter = AsyncMock(return_value=False)
        browser.apply_title_filter = AsyncMock(return_value=True)
        browser.read_applied_company_chips = AsyncMock(return_value=[])
        browser.read_applied_title_chips = AsyncMock(return_value=[])
        browser._facet_resolution_cache_put("companies", "Tess AI", "Tess")
        from linkedin.browser import read_facet_name_resolutions

        assert read_facet_name_resolutions(tmp_path)["companies"] == {
            "tess ai": "Tess"
        }

        await browser.reconcile_facets(companies={"Tess AI"}, titles=set())

        assert browser.last_facet_reconciliation["unapplied"]["companies"] == [
            "Tess AI"
        ]
        assert browser._facet_resolution_cache_get("companies", "Tess AI") is None
        assert read_facet_name_resolutions(tmp_path)["companies"] == {}

    asyncio.run(_body())
