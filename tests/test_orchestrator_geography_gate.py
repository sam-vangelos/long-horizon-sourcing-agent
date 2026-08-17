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
    pipeline.brief_obj.permanent_filters = {"Location": geo} if geo else {}
    pipeline.browser = AsyncMock()
    return pipeline


# ---------------------------------------------------------------------------
# _apply_session_location_filter — the gate
# ---------------------------------------------------------------------------


def test_failed_apply_raises_instead_of_proceeding_boolean_only(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.browser.apply_location_filter.return_value = False

    from linkedin.orchestrator import GeographyRegimeError

    with pytest.raises(GeographyRegimeError) as exc_info:
        asyncio.run(pipeline._apply_session_location_filter())

    assert exc_info.value.retryable is True
    assert pipeline._session_location_applied is False


def test_apply_exception_raises_geography_regime_error(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline.browser.apply_location_filter.side_effect = RuntimeError("DOM rotated")

    from linkedin.orchestrator import GeographyRegimeError

    with pytest.raises(GeographyRegimeError) as exc_info:
        asyncio.run(pipeline._apply_session_location_filter())

    assert exc_info.value.retryable is False


def test_apply_browser_disconnect_exception_propagates_to_recovery(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    crash = RuntimeError("Locator.is_visible: Target crashed")
    pipeline.browser.apply_location_filter.side_effect = crash

    from linkedin.orchestrator import (
        GeographyRegimeError,
        _is_browser_disconnect_error,
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(pipeline._apply_session_location_filter())

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
    pipeline.browser.apply_location_filter.return_value = True

    asyncio.run(pipeline._apply_session_location_filter())

    assert pipeline._session_location_applied is True
    (values,) = pipeline.browser.apply_location_filter.call_args.args
    assert values == GEO_VALUES


def test_no_geography_is_a_noop_not_a_gate(tmp_path):
    pipeline = _geo_pipeline(tmp_path, geo="")

    asyncio.run(pipeline._apply_session_location_filter())

    pipeline.browser.apply_location_filter.assert_not_called()


def test_recovery_reassert_propagates_the_gate(tmp_path):
    """A recovery that cannot restore geography must not resume the search."""
    pipeline = _geo_pipeline(tmp_path)
    pipeline._session_location_applied = True  # recovery resets this
    pipeline.browser.apply_location_filter.return_value = False

    from linkedin.orchestrator import GeographyRegimeError

    with pytest.raises(GeographyRegimeError):
        asyncio.run(pipeline._reassert_session_location_after_recovery())


# ---------------------------------------------------------------------------
# _verify_session_geography_chips — the pre-string invariant
# ---------------------------------------------------------------------------


def test_chips_present_is_a_cheap_noop(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline._session_location_applied = True
    pipeline.browser.read_applied_location_chips.return_value = GEO_VALUES + [
        "Some Company Pill"  # other facets' pills coexist; subset check only
    ]

    asyncio.run(pipeline._verify_session_geography_chips())

    pipeline.browser.apply_location_filter.assert_not_called()


def test_missing_chips_reassert_once_then_pass(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline._session_location_applied = True
    pipeline.browser.read_applied_location_chips.side_effect = [[], GEO_VALUES]
    pipeline.browser.apply_location_filter.return_value = True

    asyncio.run(pipeline._verify_session_geography_chips())

    pipeline.browser.apply_location_filter.assert_called_once()
    assert pipeline._session_location_applied is True


def test_missing_chips_after_reassert_fail_closed(tmp_path):
    """Apply REPORTS success but the chips still are not there → refuse to search."""
    pipeline = _geo_pipeline(tmp_path)
    pipeline._session_location_applied = True
    pipeline.browser.read_applied_location_chips.return_value = []
    pipeline.browser.apply_location_filter.return_value = True

    from linkedin.orchestrator import GeographyRegimeError

    with pytest.raises(GeographyRegimeError) as exc_info:
        asyncio.run(pipeline._verify_session_geography_chips())

    assert exc_info.value.retryable is False


def test_no_geography_invariant_is_noop(tmp_path):
    pipeline = _geo_pipeline(tmp_path, geo="")

    asyncio.run(pipeline._verify_session_geography_chips())

    pipeline.browser.read_applied_location_chips.assert_not_called()


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

    pipeline._verify_session_geography_chips = _record_verify
    pipeline.browser.enter_search_string = _record_enter

    search_string = MagicMock()
    search_string.acquisition_mode = "linkedin_boolean"
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters.is_empty.return_value = True

    asyncio.run(
        pipeline._apply_opening_search(search_string, experiment_state, '("a" OR "b")')
    )

    assert order == ["verify", "enter"]


def test_opening_search_gate_failure_stops_before_any_keyword_entry(tmp_path):
    pipeline = _geo_pipeline(tmp_path)
    pipeline._session_location_applied = True
    pipeline.browser.read_applied_location_chips.return_value = []
    pipeline.browser.apply_location_filter.return_value = False  # re-assert fails

    search_string = MagicMock()
    search_string.acquisition_mode = "linkedin_boolean"
    experiment_state = MagicMock()
    experiment_state.active_variant.structured_filters.is_empty.return_value = True

    from linkedin.orchestrator import GeographyRegimeError

    with pytest.raises(GeographyRegimeError):
        asyncio.run(
            pipeline._apply_opening_search(search_string, experiment_state, '("a")')
        )

    pipeline.browser.enter_search_string.assert_not_called()


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
    pipeline._start_runtime_stage_attempt = MagicMock(return_value="attempt-1")
    pipeline._acquisition_service = MagicMock()
    pipeline._acquisition_service.extract_profile_summary = AsyncMock(
        side_effect=GeographyRegimeError("chips absent after mid-profile recovery")
    )

    snippet = MagicMock()
    snippet.name = "Test Candidate"
    snippet.profile_url = "/talent/profile/x"

    with pytest.raises(GeographyRegimeError):
        asyncio.run(
            pipeline._full_evaluate(
                snippet, None, SearchString(id=1, name="test", boolean="x")
            )
        )


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
    pipeline.brief_obj.geography_source = "preflight"
    pipeline.browser.last_location_option_misses = {
        "New York City": list(RESOLUTION_OPTIONS)
    }
    return pipeline


def test_preflight_miss_resolves_against_real_options_then_applies(tmp_path):
    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.browser.apply_location_filter.side_effect = [False, True]
    pipeline.browser.read_applied_location_chips.return_value = []

    resolver = MagicMock(
        return_value={
            "resolutions": {"New York City": "New York City Metropolitan Area"}
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        asyncio.run(pipeline._apply_session_location_filter())

    assert resolver.call_count == 1
    assert pipeline._session_location_applied is True
    # Second apply used the RESOLVED facet name.
    (second_values,) = pipeline.browser.apply_location_filter.call_args_list[1].args
    assert second_values == ["New York City Metropolitan Area"]
    receipt = pipeline._session_geography_receipt
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
    pipeline.browser.apply_location_filter.return_value = False

    resolver = MagicMock(
        return_value={"resolutions": {"New York City": "Greater NYC Area (invented)"}}
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        with pytest.raises(GeographyRegimeError):
            asyncio.run(pipeline._apply_session_location_filter())

    assert resolver.call_count == 1
    # No second apply attempt with an invented name.
    assert pipeline.browser.apply_location_filter.call_count == 1


def test_still_missing_after_resolution_raises_with_one_model_call(tmp_path):
    from linkedin.orchestrator import GeographyRegimeError

    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.browser.apply_location_filter.side_effect = [False, False]
    pipeline.browser.read_applied_location_chips.return_value = []

    resolver = MagicMock(
        return_value={
            "resolutions": {"New York City": "New York City Metropolitan Area"}
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        with pytest.raises(GeographyRegimeError):
            asyncio.run(pipeline._apply_session_location_filter())

    assert resolver.call_count == 1  # ONE resolution call, never a loop
    assert pipeline._session_location_applied is False


def test_operator_pinned_values_never_resolve(tmp_path):
    """Pin semantics: operator-supplied facet values keep Stage A's strict
    abort — no model call rewrites what the operator typed."""
    from linkedin.orchestrator import GeographyRegimeError

    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.brief_obj.geography_source = "operator"
    pipeline.browser.apply_location_filter.return_value = False

    resolver = MagicMock()
    with patch("shared.llm_clients.cheap_llm", resolver):
        with pytest.raises(GeographyRegimeError):
            asyncio.run(pipeline._apply_session_location_filter())

    resolver.assert_not_called()


def test_miss_without_captured_options_raises_without_model_call(tmp_path):
    """Gate-1/gate-3 misses capture no typeahead options — nothing to resolve."""
    from linkedin.orchestrator import GeographyRegimeError

    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.browser.last_location_option_misses = {}
    pipeline.browser.apply_location_filter.return_value = False

    resolver = MagicMock()
    with patch("shared.llm_clients.cheap_llm", resolver):
        with pytest.raises(GeographyRegimeError):
            asyncio.run(pipeline._apply_session_location_filter())

    resolver.assert_not_called()


def test_partial_landing_reapplies_only_missing_values(tmp_path):
    """A subset-apply leaves landed chips on the sidebar; the re-apply must
    target only the still-missing (resolved) values — re-typing an applied
    facet can false-miss at the exact-match gate."""
    pipeline = _geo_pipeline(tmp_path, geo="Colombia; New York City")
    pipeline.brief_obj.geography_source = "preflight"
    pipeline.browser.last_location_option_misses = {
        "New York City": list(RESOLUTION_OPTIONS)
    }
    pipeline.browser.apply_location_filter.side_effect = [False, True]
    pipeline.browser.read_applied_location_chips.return_value = ["Colombia"]

    resolver = MagicMock(
        return_value={
            "resolutions": {"New York City": "New York City Metropolitan Area"}
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        asyncio.run(pipeline._apply_session_location_filter())

    (second_values,) = pipeline.browser.apply_location_filter.call_args_list[1].args
    assert second_values == ["New York City Metropolitan Area"]
    assert pipeline._session_geography_receipt["intended"] == [
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
    pipeline.browser.apply_location_filter.side_effect = [False, True]
    pipeline.browser.read_applied_location_chips.return_value = []

    resolver = MagicMock(
        return_value={
            "resolutions": {"New York City": "New York City Metropolitan Area"}
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        asyncio.run(pipeline._apply_session_location_filter())

    assert pipeline._session_geography_values() == [
        "New York City Metropolitan Area"
    ]

    # The pre-string chip invariant now verifies the RESOLVED chip: with the
    # resolved name on the sidebar, no re-apply fires and no gate raises.
    pipeline.browser.read_applied_location_chips.return_value = [
        "New York City Metropolitan Area"
    ]
    apply_calls_before = pipeline.browser.apply_location_filter.call_count
    asyncio.run(pipeline._verify_session_geography_chips())
    assert pipeline.browser.apply_location_filter.call_count == apply_calls_before


def test_two_candidates_resolving_to_one_facet_dedupe_before_reapply(tmp_path):
    """Two distinct candidates may correctly resolve to the SAME facet name;
    re-typing an already-applied facet gate-2-misses, so the re-apply list
    and the receipt must dedupe (correctness lens, slice 13)."""
    pipeline = _geo_pipeline(tmp_path, geo="NYC; New York City")
    pipeline.brief_obj.geography_source = "preflight"
    pipeline.browser.last_location_option_misses = {
        "NYC": list(RESOLUTION_OPTIONS),
        "New York City": list(RESOLUTION_OPTIONS),
    }
    pipeline.browser.apply_location_filter.side_effect = [False, True]
    pipeline.browser.read_applied_location_chips.return_value = []

    resolver = MagicMock(
        return_value={
            "resolutions": {
                "NYC": "New York City Metropolitan Area",
                "New York City": "New York City Metropolitan Area",
            }
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        asyncio.run(pipeline._apply_session_location_filter())

    (second_values,) = pipeline.browser.apply_location_filter.call_args_list[1].args
    assert second_values == ["New York City Metropolitan Area"]
    assert pipeline._session_geography_receipt["intended"] == [
        "New York City Metropolitan Area"
    ]


def test_stage_b_resolution_failure_classifies_as_geography_apply_transient(tmp_path):
    """A clean not-applied result that escapes the production resolution path is
    a retryable geography apply transient, driven through the real
    _apply_session_location_filter seam."""
    from linkedin.orchestrator import GeographyRegimeError
    from linkedin.session_orchestrator import _session_error_shutdown

    pipeline = _stage_b_pipeline(tmp_path)
    pipeline.browser.apply_location_filter.side_effect = [False, False]
    pipeline.browser.read_applied_location_chips.return_value = []

    resolver = MagicMock(
        return_value={
            "resolutions": {"New York City": "New York City Metropolitan Area"}
        }
    )
    with patch("shared.llm_clients.cheap_llm", resolver):
        with pytest.raises(GeographyRegimeError) as excinfo:
            asyncio.run(pipeline._apply_session_location_filter())

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
