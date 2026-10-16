"""P3a geography gate tests (plans/sourcing-rigor-hardening.md, decided 2026-07-03).

Geography is a fail-closed, VERIFIED precondition of searching — never a
monitored outcome. These tests run the seam THROUGH the orchestrator methods
that used to swallow failures (per feedback_failclosed_swallowed_by_wrapper:
test the seam through the production call path, not the helper in isolation):

- ``_apply_session_location_filter``: a False/raising browser apply raises
  GeographyRegimeError; the run never proceeds boolean-only.
- ``_verify_session_geography_chips``: missing chips get ONE re-assert through
  the fail-closed apply, then fail closed.
- ``_apply_opening_search``: the invariant runs BEFORE any keyword entry —
  the single choke point both opening paths (fresh + resume) flow through.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_orchestrator_preflight import _make_pipeline

GEO = "New York City Metropolitan Area; Colombia"
GEO_VALUES = ["New York City Metropolitan Area", "Colombia"]


def _geo_pipeline(tmp_path, geo: str = GEO):
    pipeline = _make_pipeline(str(tmp_path), tmp_path)
    pipeline.state.context.brief_obj.permanent_filters = {"Location": geo} if geo else {}
    pipeline.state.context.browser = AsyncMock()
    return pipeline


# ---------------------------------------------------------------------------
# _apply_session_location_filter — the gate
# ---------------------------------------------------------------------------


def test_failed_apply_raises_instead_of_proceeding_boolean_only(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.context.browser.apply_location_filter.return_value = False

    from linkedin.orchestrator import GeographyRegimeError

    with pytest.raises(GeographyRegimeError) as exc_info:
        asyncio.run(pipeline.state.surface._apply_session_location_filter())

    assert exc_info.value.retryable is True
    assert pipeline.state.surface._session_location_applied is False


def test_apply_exception_raises_geography_regime_error(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.context.browser.apply_location_filter.side_effect = RuntimeError("DOM rotated")

    from linkedin.orchestrator import GeographyRegimeError

    with pytest.raises(GeographyRegimeError) as exc_info:
        asyncio.run(pipeline.state.surface._apply_session_location_filter())

    assert exc_info.value.retryable is False


def test_apply_browser_disconnect_exception_propagates_to_recovery(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    crash = RuntimeError("Locator.is_visible: Target crashed")
    pipeline.state.context.browser.apply_location_filter.side_effect = crash

    from linkedin.orchestrator import (
        GeographyRegimeError,
        _is_browser_disconnect_error,
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(pipeline.state.surface._apply_session_location_filter())

    assert exc_info.value is crash
    assert not isinstance(exc_info.value, GeographyRegimeError)
    assert _is_browser_disconnect_error(exc_info.value)


def test_browser_disconnect_classifier_covers_closed_browser_error_shape():
    from linkedin.orchestrator import _is_browser_disconnect_error

    assert _is_browser_disconnect_error(
        "Mouse.wheel: Target page, context or browser has been closed"
    )


def test_successful_apply_sets_flag_and_splits_semicolon_values(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.context.browser.apply_location_filter.return_value = True

    asyncio.run(pipeline.state.surface._apply_session_location_filter())

    assert pipeline.state.surface._session_location_applied is True
    (values,) = pipeline.state.context.browser.apply_location_filter.call_args.args
    assert values == GEO_VALUES


def test_no_geography_is_a_noop_not_a_gate(tmp_path):
    pipeline = _geo_pipeline(tmp_path, geo="")

    asyncio.run(pipeline.state.surface._apply_session_location_filter())

    pipeline.state.context.browser.apply_location_filter.assert_not_called()


def test_recovery_reassert_propagates_the_gate(tmp_path):
    """A recovery that cannot restore geography must not resume the search."""
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.surface._session_location_applied = True  # recovery resets this
    pipeline.state.context.browser.apply_location_filter.return_value = False

    from linkedin.orchestrator import GeographyRegimeError

    with pytest.raises(GeographyRegimeError):
        asyncio.run(pipeline.state.surface._reassert_session_location_after_recovery())


# ---------------------------------------------------------------------------
# _verify_session_geography_chips — the pre-string invariant
# ---------------------------------------------------------------------------


def test_chips_present_is_a_cheap_noop(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.surface._session_location_applied = True
    pipeline.state.context.browser.read_applied_location_chips.return_value = GEO_VALUES + [
        "Some Company Pill"  # other facets' pills coexist; subset check only
    ]

    asyncio.run(pipeline.state.surface._verify_session_geography_chips())

    pipeline.state.context.browser.apply_location_filter.assert_not_called()


def test_missing_chips_reassert_once_then_pass(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.surface._session_location_applied = True
    pipeline.state.context.browser.read_applied_location_chips.side_effect = [[], GEO_VALUES]
    pipeline.state.context.browser.apply_location_filter.return_value = True

    asyncio.run(pipeline.state.surface._verify_session_geography_chips())

    pipeline.state.context.browser.apply_location_filter.assert_called_once()
    assert pipeline.state.surface._session_location_applied is True


def test_missing_chips_after_reassert_fail_closed(tmp_path):
    """Apply REPORTS success but the chips still are not there → refuse to search."""
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.surface._session_location_applied = True
    pipeline.state.context.browser.read_applied_location_chips.return_value = []
    pipeline.state.context.browser.apply_location_filter.return_value = True

    from linkedin.orchestrator import GeographyRegimeError

    with pytest.raises(GeographyRegimeError) as exc_info:
        asyncio.run(pipeline.state.surface._verify_session_geography_chips())

    assert exc_info.value.retryable is False


def test_unreadable_location_wrapper_reasserts_then_fails_closed(tmp_path):
    """An unreadable wrapper is never interpreted as an empty location set."""
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.surface._session_location_applied = True
    pipeline.state.context.browser.read_applied_location_chips.return_value = None
    pipeline.state.context.browser.apply_location_filter.return_value = False

    from linkedin.orchestrator import GeographyRegimeError

    with pytest.raises(GeographyRegimeError):
        asyncio.run(pipeline.state.surface._verify_session_geography_chips())

    pipeline.state.context.browser.apply_location_filter.assert_called_once()


def test_no_geography_invariant_is_noop(tmp_path):
    pipeline = _geo_pipeline(tmp_path, geo="")

    asyncio.run(pipeline.state.surface._verify_session_geography_chips())

    pipeline.state.context.browser.read_applied_location_chips.assert_not_called()


def test_flagged_exact_geography_is_untouched(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.context.browser.read_applied_location_chips.return_value = list(GEO_VALUES)

    with patch(
        "linkedin.geography_gate.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(pipeline.state.surface._verify_session_geography_chips())

    pipeline.state.context.browser.remove_facet_value.assert_not_awaited()
    pipeline.state.context.browser.apply_location_filter.assert_not_awaited()


def test_flagged_wrong_geography_clears_every_pill_then_reapplies(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    current = [GEO_VALUES[0], "Wrong Region"]
    pipeline.state.context.browser.read_applied_location_chips.side_effect = [
        current,
        ["Wrong Region"],  # F9c: rail settled after the first removal
        [],                # F9c: rail settled after the second
        list(GEO_VALUES),
    ]
    pipeline.state.context.browser.remove_facet_value.return_value = True
    pipeline.state.context.browser.apply_location_filter.return_value = True

    with patch(
        "linkedin.geography_gate.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(pipeline.state.surface._verify_session_geography_chips())

    assert [call.args for call in pipeline.state.context.browser.remove_facet_value.await_args_list] == [
        ("locations", value) for value in current
    ]
    pipeline.state.context.browser.apply_location_filter.assert_awaited_once_with(
        GEO_VALUES,
        temporal_scope="current_or_past",
    )


# ---------------------------------------------------------------------------
# _apply_opening_search — the invariant guards BOTH opening paths
# ---------------------------------------------------------------------------


def test_opening_search_verifies_chips_before_entering_keywords(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    order: list[str] = []

    async def _record_verify():
        order.append("verify")

    async def _record_enter(boolean):
        order.append("enter")

    pipeline.state.surface._verify_session_geography_chips = _record_verify
    pipeline.state.context.browser.enter_search_string = _record_enter

    search_string = MagicMock()
    search_string.acquisition_mode = "linkedin_boolean"
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters.is_empty.return_value = True

    asyncio.run(
        pipeline.state.surface._apply_opening_search(search_string, experiment_state, '("a" OR "b")')
    )

    assert order == ["verify", "enter"]


def test_opening_search_gate_failure_stops_before_any_keyword_entry(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.surface._session_location_applied = True
    pipeline.state.context.browser.read_applied_location_chips.return_value = []
    pipeline.state.context.browser.apply_location_filter.return_value = False  # re-assert fails

    search_string = MagicMock()
    search_string.acquisition_mode = "linkedin_boolean"
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters.is_empty.return_value = True

    from linkedin.orchestrator import GeographyRegimeError

    with pytest.raises(GeographyRegimeError):
        asyncio.run(
            pipeline.state.surface._apply_opening_search(search_string, experiment_state, '("a")')
        )

    pipeline.state.context.browser.enter_search_string.assert_not_called()


def test_flagged_opening_reconciles_before_keyword_and_postverifies(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
    order: list[str] = []

    async def _ready():
        order.append("ready")

    async def _location():
        order.append("location")

    async def _reconcile(*, companies, titles, verify_only=False):
        assert companies == set()
        assert titles == set()
        order.append("postverify" if verify_only else "reconcile")
        pipeline.state.context.browser.last_facet_reconciliation = {
            "before": {"companies": ["Old Co"], "titles": ["Old Title"]},
            "desired": {"companies": [], "titles": []},
            "after": {"companies": [], "titles": []},
            "failure_gate": "",
        }

    async def _enter(_boolean):
        order.append("enter")

    pipeline.state.context.browser.assert_search_surface_ready = _ready
    pipeline.state.surface._verify_session_geography_chips = _location
    pipeline.state.context.browser.reconcile_facets = _reconcile
    pipeline.state.context.browser.enter_search_string = _enter

    search_string = MagicMock(id=17, acquisition_mode="linkedin_boolean")
    filters = MagicMock(companies=[], titles=[])
    filters.is_empty.return_value = True
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters = filters

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(
            pipeline.state.surface._apply_opening_search(search_string, experiment_state, "python")
        )

    assert order == ["ready", "location", "reconcile", "enter", "postverify"]


def test_flag_off_opening_never_reconciles_facets(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
    pipeline.state.surface._verify_session_geography_chips = AsyncMock()
    pipeline.state.context.browser.assert_search_surface_ready = AsyncMock()
    pipeline.state.context.browser.enter_search_string = AsyncMock()
    pipeline.state.context.browser.reconcile_facets = AsyncMock()
    pipeline.state.context.browser.last_facet_reconciliation = {
        "before": {"companies": [], "titles": []},
        "desired": {"companies": [], "titles": []},
        "after": {"companies": [], "titles": []},
        "failure_gate": "",
    }

    search_string = MagicMock(id=20, acquisition_mode="linkedin_boolean")
    filters = MagicMock(companies=[], titles=[])
    filters.is_empty.return_value = True
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters = filters

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        False,
    ):
        asyncio.run(
            pipeline.state.surface._apply_opening_search(search_string, experiment_state, "python")
        )

    pipeline.state.context.browser.reconcile_facets.assert_not_awaited()


def test_flagged_opening_reconcile_failure_enters_no_keyword(tmp_path):
    from linkedin.browser import FacetReconciliationError

    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
    pipeline.state.context.browser.assert_search_surface_ready = AsyncMock()
    pipeline.state.surface._verify_session_geography_chips = AsyncMock()
    pipeline.state.context.browser.reconcile_facets = AsyncMock(
        side_effect=FacetReconciliationError("remove_failed")
    )

    search_string = MagicMock(id=18, acquisition_mode="linkedin_boolean")
    filters = MagicMock(companies=[], titles=[])
    filters.is_empty.return_value = True
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters = filters

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ), pytest.raises(FacetReconciliationError, match="remove_failed"):
        asyncio.run(
            pipeline.state.surface._apply_opening_search(search_string, experiment_state, "python")
        )

    pipeline.state.context.browser.enter_search_string.assert_not_awaited()


def test_flagged_opening_blocks_per_string_location_conflict(tmp_path):
    """F8: a lane-compiled per-string location that differs from the session
    set is a typed block BEFORE any browser interaction — never a silent drop.

    RED against the silent-drop code: the plan applied with the location
    filtered out, so no error was raised and the search ran inside the
    session's pool while attributed to the foreign definition.
    """
    from linkedin.browser import FacetReconciliationError
    from linkedin.search_intelligence import LinkedInStructuredFilters
    from shared.schemas import SearchString

    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
    pipeline.state.context.browser.assert_search_surface_ready = AsyncMock()
    pipeline.state.surface._verify_session_geography_chips = AsyncMock()
    pipeline.state.context.browser.reconcile_facets = AsyncMock()
    pipeline.state.context.browser.apply_advanced_search_plan = AsyncMock()
    pipeline.state.context.browser.enter_search_string = AsyncMock()

    search_string = SearchString(
        id=21,
        name="conflict",
        boolean="python",
        acquisition_mode="linkedin_hybrid",
    )
    filters = LinkedInStructuredFilters(
        companies=["Acme"],
        sidebar_filters={"locations": ["United States"]},
    )
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters = filters

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ), pytest.raises(FacetReconciliationError) as exc_info:
        asyncio.run(
            pipeline.state.surface._apply_opening_search(search_string, experiment_state, "python")
        )

    assert exc_info.value.gate == "per_string_location_conflict"

    # The block fires before any browser interaction.
    pipeline.state.context.browser.assert_search_surface_ready.assert_not_awaited()
    pipeline.state.context.browser.reconcile_facets.assert_not_awaited()
    pipeline.state.context.browser.apply_advanced_search_plan.assert_not_awaited()
    pipeline.state.context.browser.enter_search_string.assert_not_awaited()
    payload = pipeline.state.reviewer.attempts._record_runtime_event.call_args.kwargs["payload"]
    assert payload["failure_gate"] == "per_string_location_conflict"
    assert payload["boundary"] == "opening_search"


def test_flagged_opening_omits_session_identical_location_and_receipts_it(tmp_path):
    """F8: a per-string location set exactly equal to the session geography is
    omitted as redundant, and the omission is visible on the canonical receipt.
    """
    from linkedin.advanced_search import ControlApplicationResult
    from linkedin.search_intelligence import LinkedInStructuredFilters
    from shared.schemas import SearchString

    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
    pipeline.state.context.browser.assert_search_surface_ready = AsyncMock()
    pipeline.state.surface._verify_session_geography_chips = AsyncMock()
    captured = {}

    async def _reconcile(*, companies, titles, verify_only=False):
        pipeline.state.context.browser.last_facet_reconciliation = {
            "before": {"companies": [], "titles": []},
            "desired": {"companies": ["Acme"], "titles": []},
            "after": {"companies": ["Acme"], "titles": []},
            "failure_gate": "",
        }

    async def _apply(plan):
        captured["plan"] = plan
        return ControlApplicationResult(
            success=True,
            applied_controls=["keywords", "companies"],
        )

    pipeline.state.context.browser.reconcile_facets = _reconcile
    pipeline.state.context.browser.apply_advanced_search_plan = _apply

    search_string = SearchString(
        id=22,
        name="redundant",
        boolean="python",
        acquisition_mode="linkedin_hybrid",
    )
    filters = LinkedInStructuredFilters(
        companies=["Acme"],
        sidebar_filters={"locations": list(GEO_VALUES)},
    )
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters = filters
    experiment_state.active_variant.surface = "hybrid"

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(
            pipeline.state.surface._apply_opening_search(search_string, experiment_state, "python")
        )

    dimensions = [control.dimension for control in captured["plan"].controls]
    assert "locations" not in dimensions
    payload = pipeline.state.reviewer.attempts._record_runtime_event.call_args.kwargs["payload"]
    assert payload["redundant_locations_omitted"] == sorted(
        GEO_VALUES, key=str.casefold
    )
    assert payload["failure_gate"] == ""


def test_flagged_hybrid_opening_keeps_managed_controls_for_snapshot(tmp_path):
    from linkedin.advanced_search import ControlApplicationResult
    from linkedin.search_intelligence import LinkedInStructuredFilters
    from shared.schemas import SearchString

    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
    pipeline.state.context.browser.assert_search_surface_ready = AsyncMock()
    pipeline.state.surface._verify_session_geography_chips = AsyncMock()
    captured = {}

    async def _reconcile(*, companies, titles, verify_only=False):
        assert companies == {"Acme"}
        assert titles == {"Staff Engineer"}
        pipeline.state.context.browser.last_facet_reconciliation = {
            "before": {"companies": [], "titles": []},
            "desired": {
                "companies": ["Acme"],
                "titles": ["Staff Engineer"],
            },
            "after": {
                "companies": ["Acme"],
                "titles": ["Staff Engineer"],
            },
            "failure_gate": "",
        }

    async def _apply(plan):
        captured["plan"] = plan
        return ControlApplicationResult(
            success=True,
            applied_controls=["keywords", "companies", "job_titles"],
        )

    pipeline.state.context.browser.reconcile_facets = _reconcile
    pipeline.state.context.browser.apply_advanced_search_plan = _apply

    search_string = SearchString(
        id=19,
        name="hybrid",
        boolean="python",
        acquisition_mode="linkedin_hybrid",
    )
    # Session-identical locations: the F8 gate omits them as redundant (a
    # differing set is a typed block, covered above).
    filters = LinkedInStructuredFilters(
        companies=["Acme"],
        titles=["Staff Engineer"],
        sidebar_filters={"locations": list(GEO_VALUES)},
    )
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters = filters
    experiment_state.active_variant.surface = "hybrid"

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(
            pipeline.state.surface._apply_opening_search(search_string, experiment_state, "python")
        )

    dimensions = [control.dimension for control in captured["plan"].controls]
    assert "companies" in dimensions
    assert "job_titles" in dimensions
    assert "locations" not in dimensions


def test_flagged_opening_postverifies_the_landed_set_and_reports_the_miss(
    tmp_path, capsys
):
    """A company that missed its exact-match option is recorded, not fatal.

    RED against the requested-set post-verify: the verification after keyword
    entry was handed the REQUESTED companies, so a string with one recorded
    miss failed its own postcondition and ended the run.
    """
    from linkedin.advanced_search import ControlApplicationResult
    from linkedin.search_intelligence import LinkedInStructuredFilters
    from shared.schemas import SearchString

    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
    pipeline.state.context.browser.assert_search_surface_ready = AsyncMock()
    pipeline.state.surface._verify_session_geography_chips = AsyncMock()
    seen: dict[str, tuple[set[str], set[str]]] = {}

    async def _reconcile(*, companies, titles, verify_only=False):
        seen["verify" if verify_only else "add"] = (set(companies), set(titles))
        pipeline.state.context.browser.last_facet_reconciliation = {
            "before": {"companies": [], "titles": []},
            "desired": {"companies": ["Acme", "Tess AI"], "titles": []},
            "after": {"companies": ["Acme"], "titles": []},
            "unapplied": {"companies": ["Tess AI"], "titles": []},
            "failure_gate": "",
        }

    pipeline.state.context.browser.reconcile_facets = _reconcile
    pipeline.state.context.browser.apply_advanced_search_plan = AsyncMock(
        return_value=ControlApplicationResult(
            success=True, applied_controls=["keywords", "companies"]
        )
    )

    search_string = SearchString(
        id=23,
        name="partial",
        boolean="python",
        acquisition_mode="linkedin_hybrid",
    )
    filters = LinkedInStructuredFilters(companies=["Acme", "Tess AI"])
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters = filters
    experiment_state.active_variant.surface = "hybrid"

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(
            pipeline.state.surface._apply_opening_search(
                search_string, experiment_state, "python"
            )
        )

    assert seen["add"] == ({"Acme", "Tess AI"}, set())
    # The post-entry verification compares against what LANDED.
    assert seen["verify"] == ({"Acme"}, set())
    payload = pipeline.state.reviewer.attempts._record_runtime_event.call_args.kwargs[
        "payload"
    ]
    assert payload["unapplied"] == {"companies": ["Tess AI"], "titles": []}
    assert payload["failure_gate"] == ""
    assert (
        "  [facet] string #23 unapplied: companies=['Tess AI']"
        in capsys.readouterr().out
    )


def test_flagged_opening_types_each_facet_value_exactly_once(tmp_path, capsys):
    """Under the flag reconcile_facets owns companies and titles end to end,
    so the plan must not type them a second time.

    RED against the double apply: reconcile typed both values, then
    apply_advanced_search_plan typed them AGAIN — four applier calls for two
    values, and the missed one paid its full typeahead retry budget twice.
    The receipt must still name the miss as a failed control.
    """
    from shared.governor import UNGOVERNED_FOR_TESTS
    from linkedin.browser import LinkedInBrowser
    from linkedin.search_intelligence import LinkedInStructuredFilters
    from shared.schemas import SearchString

    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
    pipeline.state.surface._verify_session_geography_chips = AsyncMock()

    landed: list[str] = []
    calls: list[list[str]] = []
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    page.wait_for_timeout = AsyncMock()
    browser._page = page

    async def _apply_companies(values, **_kwargs):
        calls.append(list(values))
        ok = True
        for value in values:
            if value == "Tess AI":
                ok = False
                continue
            if value not in landed:
                landed.append(value)
        return ok

    browser.apply_company_filter = AsyncMock(side_effect=_apply_companies)
    browser.apply_title_filter = AsyncMock(return_value=True)
    browser.read_applied_company_chips = AsyncMock(side_effect=lambda: list(landed))
    browser.read_applied_title_chips = AsyncMock(return_value=[])
    browser.assert_search_surface_ready = AsyncMock()
    browser.enter_search_string = AsyncMock()
    pipeline.state.context.browser = browser

    search_string = SearchString(
        id=24,
        name="once",
        boolean="python",
        acquisition_mode="linkedin_hybrid",
    )
    filters = LinkedInStructuredFilters(companies=["Acme", "Tess AI"])
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters = filters
    experiment_state.active_variant.surface = "hybrid"

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(
            pipeline.state.surface._apply_opening_search(
                search_string, experiment_state, "python"
            )
        )

    assert calls == [["Acme"], ["Tess AI"]]
    fields = search_string.surface_receipt
    assert fields["failed_controls"] == ["companies"]
    assert fields["applied_controls"] == ["keywords", "companies"]
    assert fields["plan_fully_applied"] is False
    assert "[facet] string #24 unapplied: companies=['Tess AI']" in capsys.readouterr().out


def test_postverify_failure_receipt_preserves_initial_before_state(tmp_path):
    from linkedin.browser import FacetReconciliationError

    pipeline = _geo_pipeline(tmp_path)
    initial = {
        "before": {"companies": ["Old Co"], "titles": ["Old Title"]},
        "desired": {"companies": ["Acme"], "titles": ["Staff Engineer"]},
        "after": {"companies": ["Acme"], "titles": ["Staff Engineer"]},
        "failure_gate": "",
    }
    error = FacetReconciliationError(
        "postcondition_mismatch",
        receipt={
            "before": {"companies": ["Acme"], "titles": ["Staff Engineer"]},
            "desired": {"companies": ["Acme"], "titles": ["Staff Engineer"]},
            "after": {"companies": ["Late Co"], "titles": ["Staff Engineer"]},
            "failure_gate": "postcondition_mismatch",
        },
    )

    receipt = pipeline.state.surface._facet_failure_receipt(
        companies={"Acme"},
        titles={"Staff Engineer"},
        gate="postcondition_mismatch",
        error=error,
        before=initial,
    )

    assert receipt["before"] == initial["before"]
    assert receipt["desired"] == initial["desired"]
    assert receipt["after"] == {
        "companies": ["Late Co"],
        "titles": ["Staff Engineer"],
    }


# ---------------------------------------------------------------------------
# _full_evaluate — a regime error mid-profile-extraction is a RUN abort,
# never a soft per-candidate judgment failure (correctness lens, Wave 1)
# ---------------------------------------------------------------------------


def test_full_evaluate_propagates_geography_regime_error(tmp_path):
    """extract_profile_summary → ensure_browser_healthy → recovery re-assert can
    raise GeographyRegimeError mid-profile. The generic profile-extraction
    handler must NOT convert it into judgment_failure_decision (which would
    leave the rest of the string paging off-geography)."""
    from shared.schemas import SearchString
    from linkedin.orchestrator import GeographyRegimeError

    pipeline = _geo_pipeline(tmp_path)
    pipeline._ensure_services = MagicMock()
    pipeline.state.reviewer.attempts._start_runtime_stage_attempt = MagicMock(return_value="attempt-1")
    pipeline.state.reviewer.acquisition = MagicMock()
    pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
        side_effect=GeographyRegimeError("chips absent after mid-profile recovery")
    )

    snippet = MagicMock()
    snippet.name = "Test Candidate"
    snippet.profile_url = "/talent/profile/x"

    with pytest.raises(GeographyRegimeError):
        asyncio.run(
            pipeline.state.reviewer._full_evaluate(
                snippet, None, SearchString(id=1, name="test", boolean="x")
            )
        )


# ---------------------------------------------------------------------------
# F11 — mid-string browser recovery loses surface ownership
#
# _ensure_browser_healthy re-navigates after check_and_recover() and restores
# only the session location. The navigation also loses the owner's exact
# facet state and page cursor, so under the facet-lifecycle flag a MID-STRING
# recovery must stop the string with a named, resumable error; the resume
# path restores surface and cursor together through _apply_opening_search.
# The string-start call site passes owner_restore_follows=True because the
# opening reconcile immediately follows.
# ---------------------------------------------------------------------------


def _health_pipeline(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.surface._maybe_cadence_pause = AsyncMock()
    # The session applied its geography before the fault. The real apply is
    # idempotent on this flag, so the location comes back after the recovery
    # navigation only if that navigation resets it first.
    pipeline.state.surface._session_location_applied = True
    pipeline.state.surface._last_good_url = (
        "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    pipeline.state.context.brief_obj.linkedin_project_id = "123"
    pipeline.state.context.browser.check_and_recover = AsyncMock(return_value=True)
    pipeline.state.context.browser.navigate_to_search = AsyncMock()
    return pipeline


def _assert_location_reapplied_in_the_browser(pipeline) -> None:
    pipeline.state.context.browser.apply_location_filter.assert_awaited_once_with(
        GEO_VALUES, temporal_scope="current_or_past"
    )
    assert pipeline.state.surface._session_location_applied is True


def test_flagged_mid_string_recovery_navigation_raises_named_stop(tmp_path):
    """RED against the uncovered boundary: a mid-string recovery navigation
    returned None and the page loop kept reading candidates from an
    unverified surface and position."""
    pipeline = _health_pipeline(tmp_path)

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ), pytest.raises(Exception) as exc_info:
        asyncio.run(pipeline.state.surface._ensure_browser_healthy())

    assert type(exc_info.value).__name__ == "SurfaceOwnershipLostError"
    # The surface is still healed as far as flag-off behavior goes before the
    # stop, so the resume starts from a healthy page.
    pipeline.state.context.browser.navigate_to_search.assert_awaited_once()
    _assert_location_reapplied_in_the_browser(pipeline)


def test_flagged_pre_opening_recovery_navigation_continues(tmp_path):
    pipeline = _health_pipeline(tmp_path)

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(pipeline.state.surface._ensure_browser_healthy(owner_restore_follows=True))

    pipeline.state.context.browser.navigate_to_search.assert_awaited_once()
    _assert_location_reapplied_in_the_browser(pipeline)


def test_flag_off_mid_string_recovery_navigation_continues(tmp_path):
    pipeline = _health_pipeline(tmp_path)

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        False,
    ):
        asyncio.run(pipeline.state.surface._ensure_browser_healthy())

    pipeline.state.context.browser.navigate_to_search.assert_awaited_once()
    _assert_location_reapplied_in_the_browser(pipeline)


def test_flagged_healthy_check_never_raises(tmp_path):
    pipeline = _health_pipeline(tmp_path)
    pipeline.state.context.browser.check_and_recover = AsyncMock(return_value=False)

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(pipeline.state.surface._ensure_browser_healthy())

    pipeline.state.context.browser.navigate_to_search.assert_not_awaited()


def test_surface_ownership_lost_is_never_a_containable_judgment_fault():
    from linkedin import evaluation_loop as orchestrator_module

    error_cls = getattr(orchestrator_module, "SurfaceOwnershipLostError", None)
    assert error_cls is not None, (
        "SurfaceOwnershipLostError must be a named orchestrator error"
    )
    assert (
        orchestrator_module._is_containable_facial_page_error(error_cls("x"))
        is False
    )


def test_full_evaluate_propagates_surface_ownership_lost(tmp_path):
    """The generic profile-extraction handler converts unknown exceptions to a
    soft per-candidate judgment failure — the wrapper-swallow mode that would
    keep the string paging on a navigated, unverified surface. The named stop
    must re-raise, exactly like GeographyRegimeError."""
    from linkedin import evaluation_loop as orchestrator_module
    from shared.schemas import SearchString

    error_cls = getattr(orchestrator_module, "SurfaceOwnershipLostError", None)
    assert error_cls is not None, (
        "SurfaceOwnershipLostError must be a named orchestrator error"
    )

    pipeline = _geo_pipeline(tmp_path)
    pipeline._ensure_services = MagicMock()
    pipeline.state.reviewer.attempts._start_runtime_stage_attempt = MagicMock(return_value="attempt-1")
    pipeline.state.reviewer.acquisition = MagicMock()
    pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
        side_effect=error_cls("mid-profile recovery re-navigated the surface")
    )

    snippet = MagicMock()
    snippet.name = "Test Candidate"
    snippet.profile_url = "/talent/profile/x"

    with pytest.raises(error_cls):
        asyncio.run(
            pipeline.state.reviewer._full_evaluate(
                snippet, None, SearchString(id=1, name="test", boolean="x")
            )
        )


def test_prepare_full_candidate_propagates_surface_ownership_lost(tmp_path):
    from linkedin import evaluation_loop as orchestrator_module

    error_cls = getattr(orchestrator_module, "SurfaceOwnershipLostError", None)
    assert error_cls is not None, (
        "SurfaceOwnershipLostError must be a named orchestrator error"
    )

    pipeline = _geo_pipeline(tmp_path)
    pipeline._ensure_services = MagicMock()
    pipeline.state.reviewer.attempts._start_runtime_stage_attempt = MagicMock(return_value="attempt-1")
    pipeline.state.reviewer.acquisition = MagicMock()
    pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
        side_effect=error_cls("mid-profile recovery re-navigated the surface")
    )

    snippet = MagicMock()
    snippet.name = "Test Candidate"
    snippet.profile_url = "/talent/profile/x"
    snippet.source_string_id = 1
    snippet.source_string_name = "test"

    with pytest.raises(error_cls):
        asyncio.run(pipeline.state.reviewer._prepare_full_candidate(snippet))


# ---------------------------------------------------------------------------
# P3a Stage B — model-operated geography (Wave 3 slice 13)
#
# The operator's permanent_filters["Location"] is an OVERRIDE that PINS:
# operator values keep Stage A's strict abort-with-options. Only PREFLIGHT-
# emitted facet candidates get the resolution loop — ONE cheap_llm call
# against the REAL typeahead options captured on the miss, a re-apply of the
# resolved names, and a GeographyRegimeError when resolution or the re-apply
# still misses. Every resolution lands in the geography receipt.
# ---------------------------------------------------------------------------

RESOLUTION_OPTIONS = [
    "New York City Metropolitan Area",
    "New York, United States",
]


def _stage_b_pipeline(tmp_path, geo: str = "New York City"):
    pipeline = _geo_pipeline(tmp_path, geo=geo)
    pipeline.state.context.brief_obj.geography_source = "preflight"
    pipeline.state.context.browser.last_location_option_misses = {
        "New York City": list(RESOLUTION_OPTIONS)
    }
    return pipeline


def test_preflight_miss_resolves_against_real_options_then_applies(tmp_path):
    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.state.context.browser.apply_location_filter.side_effect = [False, True]
    pipeline.state.context.browser.read_applied_location_chips.return_value = []

    resolver = MagicMock(
        return_value={
            "resolutions": {"New York City": "New York City Metropolitan Area"}
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        asyncio.run(pipeline.state.surface._apply_session_location_filter())

    assert resolver.call_count == 1
    assert pipeline.state.surface._session_location_applied is True
    # Second apply used the RESOLVED facet name.
    (second_values,) = pipeline.state.context.browser.apply_location_filter.call_args_list[1].args
    assert second_values == ["New York City Metropolitan Area"]
    receipt = pipeline.state.surface._session_geography_receipt
    assert receipt["intended"] == ["New York City Metropolitan Area"]
    assert receipt["verified_applied"] is True
    assert receipt["resolutions"] == [
        {"candidate": "New York City", "resolved": "New York City Metropolitan Area"}
    ]


def test_resolution_never_accepts_an_invented_option(tmp_path):
    """The model must pick from the REAL options; an invented facet name is
    treated as unresolved and the gate raises."""
    from linkedin.orchestrator import GeographyRegimeError

    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.state.context.browser.apply_location_filter.return_value = False

    resolver = MagicMock(
        return_value={"resolutions": {"New York City": "Greater NYC Area (invented)"}}
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        with pytest.raises(GeographyRegimeError):
            asyncio.run(pipeline.state.surface._apply_session_location_filter())

    assert resolver.call_count == 1
    # No second apply attempt with an invented name.
    assert pipeline.state.context.browser.apply_location_filter.call_count == 1


def test_resolution_blocks_when_applied_location_wrapper_is_unreadable(tmp_path):
    """Resolution cannot decide what remains to apply without a readable set."""
    from linkedin.orchestrator import GeographyRegimeError

    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.state.context.browser.apply_location_filter.return_value = False
    pipeline.state.context.browser.read_applied_location_chips.return_value = None

    resolver = MagicMock(
        return_value={
            "resolutions": {"New York City": "New York City Metropolitan Area"}
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        with pytest.raises(GeographyRegimeError):
            asyncio.run(pipeline.state.surface._apply_session_location_filter())

    assert resolver.call_count == 1
    assert pipeline.state.context.browser.apply_location_filter.call_count == 1


def test_still_missing_after_resolution_raises_with_one_model_call(tmp_path):
    from linkedin.orchestrator import GeographyRegimeError

    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.state.context.browser.apply_location_filter.side_effect = [False, False]
    pipeline.state.context.browser.read_applied_location_chips.return_value = []

    resolver = MagicMock(
        return_value={
            "resolutions": {"New York City": "New York City Metropolitan Area"}
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        with pytest.raises(GeographyRegimeError):
            asyncio.run(pipeline.state.surface._apply_session_location_filter())

    assert resolver.call_count == 1  # ONE resolution call, never a loop
    assert pipeline.state.surface._session_location_applied is False


def test_operator_pinned_values_never_resolve(tmp_path):
    """Pin semantics: operator-supplied facet values keep Stage A's strict
    abort — no model call rewrites what the operator typed."""
    from linkedin.orchestrator import GeographyRegimeError

    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.state.context.brief_obj.geography_source = "operator"
    pipeline.state.context.browser.apply_location_filter.return_value = False

    resolver = MagicMock()
    with patch("shared.llm_clients.cheap_llm", resolver):
        with pytest.raises(GeographyRegimeError):
            asyncio.run(pipeline.state.surface._apply_session_location_filter())

    resolver.assert_not_called()


def test_miss_without_captured_options_raises_without_model_call(tmp_path):
    """Gate-1/gate-3 misses capture no typeahead options — nothing to resolve."""
    from linkedin.orchestrator import GeographyRegimeError

    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.state.context.browser.last_location_option_misses = {}
    pipeline.state.context.browser.apply_location_filter.return_value = False

    resolver = MagicMock()
    with patch("shared.llm_clients.cheap_llm", resolver):
        with pytest.raises(GeographyRegimeError):
            asyncio.run(pipeline.state.surface._apply_session_location_filter())

    resolver.assert_not_called()


def test_partial_landing_reapplies_only_missing_values(tmp_path):
    """A subset-apply leaves landed chips on the sidebar; the re-apply must
    target only the still-missing (resolved) values — re-typing an applied
    facet can false-miss at the exact-match gate."""
    pipeline = _geo_pipeline(tmp_path, geo="Colombia; New York City")
    pipeline.state.context.brief_obj.geography_source = "preflight"
    pipeline.state.context.browser.last_location_option_misses = {
        "New York City": list(RESOLUTION_OPTIONS)
    }
    pipeline.state.context.browser.apply_location_filter.side_effect = [False, True]
    pipeline.state.context.browser.read_applied_location_chips.return_value = ["Colombia"]

    resolver = MagicMock(
        return_value={
            "resolutions": {"New York City": "New York City Metropolitan Area"}
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        asyncio.run(pipeline.state.surface._apply_session_location_filter())

    (second_values,) = pipeline.state.context.browser.apply_location_filter.call_args_list[1].args
    assert second_values == ["New York City Metropolitan Area"]
    assert pipeline.state.surface._session_geography_receipt["intended"] == [
        "Colombia",
        "New York City Metropolitan Area",
    ]


def test_resolved_geography_becomes_the_sessions_effective_geography(tmp_path):
    """THE Stage B invariant: after a successful resolution the session's
    geography IS the resolved facet list — the chip invariant verifies the
    resolved chips and never force-reasserts the raw candidate (test-honesty
    lens, slice 13: deleting the override assignment left every test green
    while every successful resolution would self-destruct at the first
    pre-string chip check)."""
    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.state.context.browser.apply_location_filter.side_effect = [False, True]
    pipeline.state.context.browser.read_applied_location_chips.return_value = []

    resolver = MagicMock(
        return_value={
            "resolutions": {"New York City": "New York City Metropolitan Area"}
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        asyncio.run(pipeline.state.surface._apply_session_location_filter())

    assert pipeline.state.surface._session_geography_values() == [
        "New York City Metropolitan Area"
    ]

    # The pre-string chip invariant now verifies the RESOLVED chip: with the
    # resolved name on the sidebar, no re-apply fires and no gate raises.
    pipeline.state.context.browser.read_applied_location_chips.return_value = [
        "New York City Metropolitan Area"
    ]
    apply_calls_before = pipeline.state.context.browser.apply_location_filter.call_count
    asyncio.run(pipeline.state.surface._verify_session_geography_chips())
    assert pipeline.state.context.browser.apply_location_filter.call_count == apply_calls_before


def test_two_candidates_resolving_to_one_facet_dedupe_before_reapply(tmp_path):
    """Two distinct candidates may correctly resolve to the SAME facet name;
    re-typing an already-applied facet gate-2-misses, so the re-apply list
    and the receipt must dedupe (correctness lens, slice 13)."""
    pipeline = _geo_pipeline(tmp_path, geo="NYC; New York City")
    pipeline.state.context.brief_obj.geography_source = "preflight"
    pipeline.state.context.browser.last_location_option_misses = {
        "NYC": list(RESOLUTION_OPTIONS),
        "New York City": list(RESOLUTION_OPTIONS),
    }
    pipeline.state.context.browser.apply_location_filter.side_effect = [False, True]
    pipeline.state.context.browser.read_applied_location_chips.return_value = []

    resolver = MagicMock(
        return_value={
            "resolutions": {
                "NYC": "New York City Metropolitan Area",
                "New York City": "New York City Metropolitan Area",
            }
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        asyncio.run(pipeline.state.surface._apply_session_location_filter())

    (second_values,) = pipeline.state.context.browser.apply_location_filter.call_args_list[1].args
    assert second_values == ["New York City Metropolitan Area"]
    assert pipeline.state.surface._session_geography_receipt["intended"] == [
        "New York City Metropolitan Area"
    ]


def test_stage_b_resolution_failure_classifies_as_geography_apply_transient(tmp_path):
    """A clean not-applied result that escapes the production resolution path is
    a retryable geography apply transient, driven through the real
    _apply_session_location_filter seam."""
    from linkedin.orchestrator import GeographyRegimeError
    from linkedin.session_orchestrator import _session_error_shutdown

    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.state.context.browser.apply_location_filter.side_effect = [False, False]
    pipeline.state.context.browser.read_applied_location_chips.return_value = []

    resolver = MagicMock(
        return_value={
            "resolutions": {"New York City": "New York City Metropolitan Area"}
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        with pytest.raises(GeographyRegimeError) as excinfo:
            asyncio.run(pipeline.state.surface._apply_session_location_filter())

    reason, _kind = _session_error_shutdown(excinfo.value)
    assert reason == "geography_apply_transient"


# ---------------------------------------------------------------------------
# P3a Stage B — provenance (loader) + pin (preflight merge)
# ---------------------------------------------------------------------------


def test_v2_loader_stamps_geography_source_by_shape():
    from shared.brief_loader import _load_v2_brief

    base = {
        "role_title": "Test Role",
        "role_summary": "Test",
        "capability_areas": [
            {
                "name": "Area",
                "description": "d",
                "builder_signals": ["b"],
                "user_signals": [],
            }
        ],
        "depth_distinction": {
            "builder_definition": "b",
            "user_definition": "u",
            "edge_case_guidance": "e",
        },
        "non_fit_patterns": [{"label": "x", "description": "d", "why_not": "w"}],
        "minimum_years_experience": 3,
        "minimum_bar_description": "bar",
        "facial_calibration": {
            "expected_yes_rate_low": 0.2,
            "expected_yes_rate_high": 0.5,
            "fast_exit_patterns": ["p"],
            "trajectory_yes_patterns": ["p"],
            "trajectory_ambiguous_patterns": ["p"],
            "trajectory_no_patterns": ["p"],
        },
    }

    operator = _load_v2_brief({**base, "geography": "New York City Metropolitan Area"})
    assert operator.geography_source == "operator"
    assert operator.permanent_filters["Location"] == "New York City Metropolitan Area"

    preflight = _load_v2_brief(
        {
            **base,
            "geography": {
                "facet_candidates": ["New York City Metropolitan Area"],
                "rationale": "JD names NYC",
            },
        }
    )
    assert preflight.geography_source == "preflight"

    none = _load_v2_brief(base)
    assert none.geography_source == ""
    assert "Location" not in none.permanent_filters


def test_operator_geography_override_pins_over_model_candidates():
    """preflight_to_brief_json: an operator string REPLACES the model's
    structured geography wholesale — the pin, locked."""
    from shared.preflight_v2 import preflight_to_brief_json

    merged = preflight_to_brief_json(
        {
            "geography": {
                "facet_candidates": ["San Francisco Bay Area"],
                "rationale": "model extraction",
            }
        },
        {"geography": "New York City Metropolitan Area"},
    )
    assert merged["geography"] == "New York City Metropolitan Area"


def test_run_report_renders_geography_resolutions():
    from tests.test_lane_key_integrity import _structured_report
    from shared.run_report_schema import render_run_report_markdown

    report = _structured_report({})
    report.run_metadata["session_geography"] = {
        "intended": ["New York City Metropolitan Area"],
        "verified_applied": True,
        "reasserts": 0,
        "resolutions": [
            {
                "candidate": "New York City",
                "resolved": "New York City Metropolitan Area",
            }
        ],
    }
    text = render_run_report_markdown(report)
    assert "New York City→New York City Metropolitan Area" in text
    assert "model-resolved" in text


# ---------------------------------------------------------------------------
# F9c — the geography self-heal must settle the rail between removals.
#
# Every removal re-runs the search and re-renders the rail, so a second removal
# issued against the pre-removal sidebar operates on stale DOM. This is the same
# defect fixed for companies and titles in 13ee404; multi-location sessions are
# the exposure (Brazil is single-location, so it has never fired).
# ---------------------------------------------------------------------------


def _recording_geo_pipeline(tmp_path, reads):
    """Pipeline whose browser records the ORDER of reads, removals and applies."""
    pipeline = _geo_pipeline(tmp_path)
    order: list[str] = []
    remaining = list(reads)

    async def _read():
        order.append("read")
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    async def _remove(_dimension, value):
        order.append(f"remove:{value}")
        return True

    async def _apply(*_args, **_kwargs):
        order.append("apply")
        return True

    pipeline.state.context.browser.read_applied_location_chips.side_effect = _read
    pipeline.state.context.browser.remove_facet_value.side_effect = _remove
    pipeline.state.context.browser.apply_location_filter.side_effect = _apply
    return pipeline, order


def test_flagged_geography_clear_settles_the_rail_between_removals(tmp_path):
    """The second pill is not dismissed until the first one's removal lands."""
    stale = ["Wrong Region A", "Wrong Region B"]
    pipeline, order = _recording_geo_pipeline(
        tmp_path,
        reads=[
            list(stale),          # opening read: both pills wrong
            list(stale),          # poll after removing A: rail not settled yet
            ["Wrong Region B"],   # poll after removing A: settled
            [],                   # poll after removing B: settled
            list(GEO_VALUES),     # post-reapply verification
        ],
    )

    with patch(
        "linkedin.geography_gate.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(pipeline.state.surface._verify_session_geography_chips())

    assert order == [
        "read",
        "remove:Wrong Region A",
        "read",
        "read",
        "remove:Wrong Region B",
        "read",
        "apply",
        "read",
    ]


def test_flagged_geography_clear_fails_closed_when_the_rail_never_settles(tmp_path):
    """A rail that never reflects the first removal stops the run, not the loop.

    Removing the next pill against a sidebar that still shows the previous one is
    how a self-heal silently dismisses the wrong chip.
    """
    from linkedin.orchestrator import GeographyRegimeError

    stale = ["Wrong Region A", "Wrong Region B"]
    pipeline, order = _recording_geo_pipeline(tmp_path, reads=[list(stale)])

    with patch(
        "linkedin.geography_gate.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        with pytest.raises(GeographyRegimeError):
            asyncio.run(pipeline.state.surface._verify_session_geography_chips())

    assert [entry for entry in order if entry.startswith("remove:")] == [
        "remove:Wrong Region A"
    ]
    pipeline.state.context.browser.apply_location_filter.assert_not_awaited()


# ---------------------------------------------------------------------------
# _apply_opening_search — a marked surface is re-anchored before it is judged
# ---------------------------------------------------------------------------


def test_marked_surface_is_reanchored_before_the_readiness_gate(tmp_path):
    """Addendum I2/I7.3: the opening repairs the surface, then inspects it.

    A zero-result string leaves LinkedIn's no-results page behind and marks
    the surface. The next opening must navigate to the project search page and
    re-assert session geography BEFORE the readiness gate reads anything —
    otherwise the gate is still judging the dead page, which is how one
    zero-result string became three strikes and a named stop.
    """
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
    order: list[str] = []
    flag_when_gate_ran: list[str] = []

    async def _navigate(url):
        order.append(f"navigate:{url}")

    def _reset():
        order.append("reset-location-applied")
        pipeline.state.surface.geography._session_location_applied = False

    async def _apply_location():
        order.append("apply-session-location")

    async def _ready():
        order.append("ready")
        flag_when_gate_ran.append(
            pipeline.state.surface.surface_reanchor_pending()
        )

    async def _verify():
        order.append("verify-geography")

    async def _reconcile(*, companies, titles, verify_only=False):
        order.append("postverify" if verify_only else "reconcile")
        pipeline.state.context.browser.last_facet_reconciliation = {
            "before": {"companies": [], "titles": []},
            "desired": {"companies": [], "titles": []},
            "after": {"companies": [], "titles": []},
            "failure_gate": "",
        }

    async def _enter(_boolean):
        order.append("enter")

    pipeline.state.context.browser.navigate_to_search = _navigate
    pipeline.state.surface.geography.reset_location_applied = _reset
    pipeline.state.surface.geography._apply_session_location_filter = _apply_location
    pipeline.state.context.browser.assert_search_surface_ready = _ready
    pipeline.state.surface._verify_session_geography_chips = _verify
    pipeline.state.context.browser.reconcile_facets = _reconcile
    pipeline.state.context.browser.enter_search_string = _enter

    search_string = MagicMock(id=41, acquisition_mode="linkedin_boolean")
    filters = MagicMock(companies=[], titles=[])
    filters.is_empty.return_value = True
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters = filters

    pipeline.state.surface.mark_surface_for_reanchor("zero_results")
    assert pipeline.state.surface.surface_reanchor_pending() == "zero_results"

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(
            pipeline.state.surface._apply_opening_search(
                search_string, experiment_state, "python"
            )
        )

    assert order == [
        "navigate:https://www.linkedin.com/talent/hire/test-project/"
        "discover/recruiterSearch",
        "reset-location-applied",
        "apply-session-location",
        "ready",
        "verify-geography",
        "reconcile",
        "enter",
        "postverify",
    ]
    # Cleared before the gate ran, so one mark buys exactly one re-anchor.
    assert flag_when_gate_ran == [""]
    assert pipeline.state.surface.surface_reanchor_pending() == ""


def test_an_unmarked_surface_is_never_renavigated(tmp_path):
    """The ordinary string-to-string path is untouched: no extra navigation."""
    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
    pipeline.state.context.browser.assert_search_surface_ready = AsyncMock()
    pipeline.state.surface._verify_session_geography_chips = AsyncMock()
    pipeline.state.context.browser.last_facet_reconciliation = {
        "before": {"companies": [], "titles": []},
        "desired": {"companies": [], "titles": []},
        "after": {"companies": [], "titles": []},
        "failure_gate": "",
    }

    search_string = MagicMock(id=42, acquisition_mode="linkedin_boolean")
    filters = MagicMock(companies=[], titles=[])
    filters.is_empty.return_value = True
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters = filters

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ):
        asyncio.run(
            pipeline.state.surface._apply_opening_search(
                search_string, experiment_state, "python"
            )
        )

    pipeline.state.context.browser.navigate_to_search.assert_not_awaited()


def test_a_reanchor_that_cannot_navigate_leaves_the_gates_to_refuse(tmp_path):
    """Fail-soft, and it bypasses nothing: the readiness gate still runs.

    A re-anchor is a repair attempt, not an authority. When the navigation
    raises, the flag is spent, the failure is a run-log fact, and the gate
    that follows decides on the live page's own evidence.
    """
    from linkedin.browser import FacetReconciliationError

    pipeline = _geo_pipeline(tmp_path)
    pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
    pipeline.state.context.browser.navigate_to_search = AsyncMock(
        side_effect=RuntimeError("navigate_to_search failed: sidebar elements not found")
    )
    pipeline.state.context.browser.assert_search_surface_ready = AsyncMock(
        side_effect=FacetReconciliationError("search_surface_unready")
    )

    search_string = MagicMock(id=43, acquisition_mode="linkedin_boolean")
    filters = MagicMock(companies=[], titles=[])
    filters.is_empty.return_value = True
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters = filters

    pipeline.state.surface.mark_surface_for_reanchor("zero_results")

    with patch(
        "linkedin.evaluation_loop.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED",
        True,
    ), pytest.raises(FacetReconciliationError) as exc_info:
        asyncio.run(
            pipeline.state.surface._apply_opening_search(
                search_string, experiment_state, "python"
            )
        )

    assert exc_info.value.gate == "search_surface_unready"
    pipeline.state.context.browser.assert_search_surface_ready.assert_awaited_once()
    assert pipeline.state.surface.surface_reanchor_pending() == ""
    payload = pipeline.state.reviewer.attempts._record_runtime_event.call_args.kwargs[
        "payload"
    ]
    assert payload["failure_gate"] == "search_surface_unready"
