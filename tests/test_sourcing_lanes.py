"""Tests for shared/sourcing_lanes.py (P1 unification layer)."""

import json

import pytest

from shared.sourcing_lanes import (
    FieldDiagnostic,
    LaneExecution,
    LaneMetrics,
    LaneProbe,
    SearchConstraint,
    SearchHypothesis,
    SearchSlice,
    SourcingLane,
    align_lane_identity,
    apply_lane_fields_to_generated_string,
    apply_lane_fields_to_search_string,
    lane_fields_from_work_unit_item,
    normalize_lane_id,
    validate_lane_identity_alignment,
    validate_lane_probe,
    validate_search_constraint,
    validate_sourcing_lane,
    validate_sourcing_lane_dict,
)
from shared.schemas import SearchString


def _sample_constraint() -> SearchConstraint:
    return SearchConstraint(
        dimension="capability",
        values=["workflow orchestration", "tool calling"],
        rationale="Orchestration work",
        operator="prefer",
        execution_surface="boolean_keyword",
    )


def _sample_hypothesis() -> SearchHypothesis:
    return SearchHypothesis(
        hypothesis_id="fde_delivery_builders",
        label="FDE delivery builders",
        target_archetype="FDE delivery builders",
        why_this_pool_may_exist="Broad entry plus builder proof.",
        capability_signals=["workflow orchestration"],
        hidden_pool_risks=["sales-only noise"],
    )


def _sample_slice() -> SearchSlice:
    return SearchSlice(
        slice_id="fde_delivery_builders_slice",
        hypothesis_id="fde_delivery_builders",
        label="FDE delivery builders",
        objective="Open with broad delivery cohorts.",
        constraints=[_sample_constraint()],
    )


def _sample_execution() -> LaneExecution:
    return LaneExecution(
        lane_id="fde_delivery_builders",
        source="linkedin",
        acquisition_mode="linkedin_boolean",
    )


def _sample_lane() -> SourcingLane:
    return SourcingLane(
        lane_id="fde_delivery_builders",
        lane_name="FDE delivery builders",
        hypothesis=_sample_hypothesis(),
        slice=_sample_slice(),
        execution=_sample_execution(),
    )


@pytest.mark.parametrize(
    "factory,cls",
    [
        (_sample_constraint, SearchConstraint),
        (_sample_hypothesis, SearchHypothesis),
        (_sample_slice, SearchSlice),
        (_sample_execution, LaneExecution),
        (_sample_lane, SourcingLane),
    ],
)
def test_dataclass_round_trip(factory, cls):
    original = factory()
    payload = original.to_dict()
    restored = cls.from_dict(payload)
    assert restored.to_dict() == payload


def test_lane_probe_and_metrics_round_trip():
    probe = LaneProbe(
        probe_id="probe-1",
        lane_id="fde_delivery_builders",
        decision="continue",
        observed_metrics={"result_count": 42},
    )
    metrics = LaneMetrics(lane_id="fde_delivery_builders", save_count=2)
    assert LaneProbe.from_dict(probe.to_dict()).to_dict() == probe.to_dict()
    assert LaneMetrics.from_dict(metrics.to_dict()).to_dict() == metrics.to_dict()


def test_json_round_trip_for_sourcing_lane():
    lane = _sample_lane()
    restored = SourcingLane.from_dict(json.loads(json.dumps(lane.to_dict())))
    assert restored.lane_id == lane.lane_id
    assert restored.hypothesis.label == lane.hypothesis.label


def test_invalid_enum_produces_diagnostic_not_exception():
    constraint = SearchConstraint(
        dimension="capability",
        values=["x"],
        operator="definitely_not_valid",
    )
    issues = validate_search_constraint(constraint)
    assert issues
    assert any(issue.code == "invalid_operator" for issue in issues)
    assert all(isinstance(issue, FieldDiagnostic) for issue in issues)


def test_validate_sourcing_lane_catches_invalid_execution_mode():
    lane = _sample_lane()
    lane.execution.acquisition_mode = "not_a_real_mode"
    issues = validate_sourcing_lane(lane)
    assert any(issue.code == "invalid_acquisition_mode" for issue in issues)


def test_align_lane_identity_from_family_key():
    lane_id, family_key = align_lane_identity(family_key="FDE Delivery Builders")
    assert lane_id == family_key == "fde_delivery_builders"


def test_align_lane_identity_empty_means_legacy():
    assert align_lane_identity() == ("", "")


def test_validate_lane_identity_alignment_detects_drift():
    issues = validate_lane_identity_alignment("payments", "capital_markets")
    assert len(issues) == 1
    assert issues[0].code == "lane_family_drift"


def test_normalize_lane_id_empty_returns_empty():
    assert normalize_lane_id("") == ""
    assert normalize_lane_id(None) == ""


def test_apply_lane_fields_to_generated_string_backfills_lane_id():
    item = {"boolean": "(test)", "family_key": "fde_delivery_builders"}
    updated = apply_lane_fields_to_generated_string(item)
    assert updated["lane_id"] == "fde_delivery_builders"
    assert updated["family_key"] == "fde_delivery_builders"


def test_apply_lane_fields_to_generated_string_does_not_overwrite_conflict():
    item = {
        "boolean": "(test)",
        "lane_id": "payments",
        "family_key": "capital_markets",
    }
    original = dict(item)
    updated = apply_lane_fields_to_generated_string(item)
    assert updated == original


def test_apply_lane_fields_to_search_string_backfills_lane_id():
    ss = SearchString(id=1, name="x", boolean="y", family_key="fde_delivery_builders")
    apply_lane_fields_to_search_string(ss)
    assert ss.lane_id == "fde_delivery_builders"
    assert ss.family_key == "fde_delivery_builders"


def test_validate_lane_probe_requires_ids():
    issues = validate_lane_probe(LaneProbe(probe_id="", lane_id=""))
    assert any(issue.code == "missing_probe_id" for issue in issues)
    assert any(issue.code == "missing_lane_id" for issue in issues)


def test_validate_sourcing_lane_dict_empty_payload_returns_diagnostics() -> None:
    issues = validate_sourcing_lane_dict({})
    assert issues
    assert any(issue.code == "empty_payload" for issue in issues)
    codes = {issue.code for issue in issues}
    assert "missing_hypothesis" in codes
    assert "missing_slice" in codes
    assert "missing_execution" in codes


def test_validate_sourcing_lane_dict_partial_payload_reports_nested_missing() -> None:
    issues = validate_sourcing_lane_dict({"lane_id": "delivery_builders"})
    codes = {issue.code for issue in issues}
    assert "missing_hypothesis" in codes
    assert "missing_slice" in codes
    assert "missing_execution" in codes


def test_validate_sourcing_lane_dict_valid_lane_has_no_errors() -> None:
    issues = validate_sourcing_lane_dict(_sample_lane().to_dict())
    assert not any(issue.severity == "error" for issue in issues)


def test_lane_fields_from_work_unit_item_aligns_family_key() -> None:
    fields = lane_fields_from_work_unit_item(
        {
            "boolean": "(test)",
            "family_key": "fde_delivery_builders",
            "rationale": "Delivery lane",
            "retrieval_recipe": {"family_label": "FDE delivery"},
        }
    )
    assert fields["lane_id"] == "fde_delivery_builders"
    assert fields["lane_name"] == "FDE delivery"
    assert fields["lane_intent"] == "Delivery lane"
    assert fields["acquisition_mode"] == "linkedin_boolean"


def test_lane_fields_from_work_unit_item_lifts_compiler_search_posture() -> None:
    fields = lane_fields_from_work_unit_item(
        {
            "boolean": "(test)",
            "lane_id": "filter_lane",
            "lane_snapshot": {
                "compiler": {
                    "acquisition_mode": "linkedin_hybrid",
                    "search_posture": "structured_only",
                    "query_payload": {
                        "structured_filters": {"companies": ["Stripe"]}
                    },
                }
            },
        }
    )

    assert fields["acquisition_mode"] == "linkedin_hybrid"
    assert fields["search_posture"] == "structured_only"
    assert fields["structured_filters"] == {"companies": ["Stripe"]}


def test_lane_fields_from_work_unit_item_legacy_empty_lane_id() -> None:
    fields = lane_fields_from_work_unit_item({"boolean": "(test)"})
    assert fields["lane_id"] == ""
    assert fields["lane_name"] == ""
