"""Tests for SearchString lane field propagation helpers."""

from shared.schemas import SearchString
from shared.sourcing_lanes import (
    apply_lane_fields_to_search_string,
    lane_fields_from_work_unit_item,
)


def test_search_string_lane_fields_sync_after_hydrate_style_family_key() -> None:
    item = {
        "boolean": "(test)",
        "family_key": "delivery_builders",
        "rationale": "builder proof",
    }
    ss = SearchString(
        id=1,
        name="Compound",
        boolean=item["boolean"],
        family_key="",
        **lane_fields_from_work_unit_item(item),
    )
    ss.family_key = "delivery_builders"
    apply_lane_fields_to_search_string(ss)
    assert ss.lane_id == ss.family_key == "delivery_builders"
    assert ss.lane_intent == "builder proof"


def test_work_unit_own_structured_filters_reach_search_string_as_hybrid() -> None:
    """Slice 1: a generated string carrying its OWN structured_filters reaches the
    SearchString and is promoted to linkedin_hybrid, without a lane-compiler snapshot
    or a family_key -> lane projection match."""
    item = {
        "boolean": '("machine learning")',
        "family_key": "fintech_ml",
        "rationale": "fintech ML builders",
        "structured_filters": {"companies": ["Nubank", "Bancolombia"], "titles": []},
    }
    fields = lane_fields_from_work_unit_item(item)
    assert fields["acquisition_mode"] == "linkedin_hybrid"
    assert fields["structured_filters"]["companies"] == ["Nubank", "Bancolombia"]


def test_work_unit_boolean_normalization_reaches_search_string_fields() -> None:
    report = {
        "changed": True,
        "findings": [
            {
                "code": "token_subset_superstring_pruned",
                "terms": ["reward model development"],
            }
        ],
    }
    fields = lane_fields_from_work_unit_item(
        {
            "boolean": '("reward model")',
            "family_key": "rlhf",
            "boolean_normalization": report,
        }
    )
    ss = SearchString(id=1, name="Adaptive", boolean='("reward model")', **fields)

    assert ss.boolean_normalization == report


def test_work_unit_without_structured_filters_stays_boolean_byte_identical() -> None:
    """Byte-identical default: a plain generated string keeps empty filters and the
    default boolean acquisition mode (the Slice 1 fallback is a no-op)."""
    item = {"boolean": "(test)", "family_key": "plain", "rationale": "x"}
    fields = lane_fields_from_work_unit_item(item)
    assert fields["acquisition_mode"] == "linkedin_boolean"
    assert fields["structured_filters"] == {}


def test_work_unit_empty_structured_filters_lists_stay_boolean() -> None:
    """All-empty structured_filters lists carry no live facet -> not hybrid."""
    item = {
        "boolean": "(test)",
        "family_key": "plain",
        "structured_filters": {"companies": [], "titles": [], "sidebar_filters": {}},
    }
    fields = lane_fields_from_work_unit_item(item)
    assert fields["acquisition_mode"] == "linkedin_boolean"
