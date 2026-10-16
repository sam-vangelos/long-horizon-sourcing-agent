"""Backward compatibility tests for ExecutionPlan and SearchString lane fields."""

from shared.retrieval_design import render_retrieval_design, retrieval_design_from_payload
from shared.schemas import ExecutionPlan, SearchString
from shared.sourcing_lanes import apply_lane_fields_to_generated_string


LEGACY_EXECUTION_PLAN = {
    "strategy_rationale": "legacy plan",
    "architecture": "dragnet",
    "architecture_rationale": "",
    "architecture_success_criteria": [],
    "architecture_pivot_triggers": [],
    "noise_predictions": [],
    "generated_strings": [
        {
            "boolean": '("RLHF" OR "DPO") AND ("frontier lab")',
            "rationale": "mock recall string",
            "family_key": "frontier_alignment",
            "novelty_bucket": "canonical",
            "domain_lane": "general",
        }
    ],
    "retrieval_families": [],
    "coverage_gaps": [],
}


def test_execution_plan_from_dict_loads_legacy_payload_without_lane_fields():
    plan = ExecutionPlan.from_dict(LEGACY_EXECUTION_PLAN)
    assert plan.sourcing_lanes == []
    assert plan.search_hypotheses == []
    assert plan.search_slices == []
    assert len(plan.generated_strings) == 1
    assert "RLHF" in plan.generated_strings[0]["boolean"]


def test_execution_plan_round_trips_new_lane_fields():
    payload = {
        **LEGACY_EXECUTION_PLAN,
        "sourcing_lanes": [{"lane_id": "frontier_alignment", "lane_name": "Frontier"}],
        "search_hypotheses": [{"hypothesis_id": "frontier_alignment", "label": "Frontier"}],
        "search_slices": [{"slice_id": "frontier_alignment_slice", "hypothesis_id": "frontier_alignment"}],
    }
    plan = ExecutionPlan.from_dict(payload)
    restored = ExecutionPlan.from_dict(plan.to_dict())
    assert restored.sourcing_lanes == payload["sourcing_lanes"]
    assert restored.search_hypotheses == payload["search_hypotheses"]
    assert restored.search_slices == payload["search_slices"]


def test_search_string_from_dict_loads_legacy_without_lane_fields():
    legacy = {"id": 1, "name": "Compound", "boolean": '("test")'}
    restored = SearchString.from_dict(legacy)
    assert restored.lane_id == ""
    assert restored.lane_name == ""
    assert restored.acquisition_mode == ""
    assert restored.lane_snapshot == {}


def test_search_string_lane_fields_survive_round_trip():
    payload = {
        "id": 2,
        "name": "Compound",
        "boolean": '("test")',
        "lane_id": "fde_delivery_builders",
        "lane_name": "FDE delivery builders",
        "lane_intent": "builder proof",
        "acquisition_mode": "linkedin_boolean",
        "lane_snapshot": {"variant_index": 1},
        "family_key": "fde_delivery_builders",
    }
    restored = SearchString.from_dict(payload)
    assert restored.to_dict()["lane_id"] == "fde_delivery_builders"
    assert restored.lane_snapshot == {"variant_index": 1}


def _search_string_from_generated_item(gs: dict, *, next_id: int = 1) -> SearchString | None:
    """Mirror linkedin/orchestrator._build_ordered_search_strings field mapping."""
    boolean = gs.get("boolean", "")
    if not boolean:
        return None
    rationale = gs.get("rationale", "")
    return SearchString(
        id=next_id,
        name=f"Compound / {rationale[:60]}" if rationale else "Compound",
        boolean=boolean,
        block="Compound Batch 1",
        subblock="Compound",
        string_type="Precision",
        family_key=gs.get("family_key", ""),
        novelty_bucket=gs.get("novelty_bucket", ""),
        domain_lane=gs.get("domain_lane", ""),
        seniority_risk=gs.get("seniority_risk", ""),
        title_bucket_risk=gs.get("title_bucket_risk", ""),
        opening_eligible=gs.get("opening_eligible"),
        retrieval_recipe=gs.get("retrieval_recipe", {}),
        retrieval_hypothesis_ids=list(gs.get("retrieval_hypothesis_ids", [])),
        lane_id=gs.get("lane_id", ""),
        lane_name=gs.get("lane_name", ""),
        lane_intent=gs.get("lane_intent", ""),
        acquisition_mode=gs.get("acquisition_mode", ""),
        lane_snapshot=gs.get("lane_snapshot", {}),
    )


def test_legacy_generated_strings_remain_executable():
    plan = ExecutionPlan.from_dict(LEGACY_EXECUTION_PLAN)
    for index, gs in enumerate(plan.generated_strings, start=1):
        ss = _search_string_from_generated_item(gs, next_id=index)
        assert ss is not None
        assert ss.boolean
        assert ss.lane_id == ""


def test_rendered_retrieval_strings_remain_executable():
    design = retrieval_design_from_payload(
        {
            "families": [
                {
                    "family_id": "delivery_builders",
                    "label": "Delivery builders",
                    "objective": "Broad entry plus builder proof.",
                    "priority": 90,
                    "enabled": True,
                    "variants_to_emit": 1,
                    "entry_signals": [
                        {"item_id": "entry_1", "label": "Delivery", "terms": ["deployment engineer"]}
                    ],
                    "capability_proxies": [
                        {"item_id": "cap_1", "label": "Capability", "terms": ["workflow orchestration"]}
                    ],
                    "reality_filters": [
                        {"item_id": "real_1", "label": "Reality", "terms": ["production"]}
                    ],
                    "context_constraints": [],
                    "anti_noise": [],
                }
            ],
            "shared_layers": {},
            "edge_case_hypotheses": [],
        }
    )
    _families, strings = render_retrieval_design(design)
    for index, gs in enumerate(strings, start=1):
        ss = _search_string_from_generated_item(gs, next_id=index)
        assert ss is not None
        assert ss.boolean
        assert "deployment engineer" in ss.boolean


def test_apply_lane_fields_aligns_family_key_and_lane_id_on_generated_string():
    gs = {"boolean": "(test)", "family_key": "delivery_builders"}
    updated = apply_lane_fields_to_generated_string(gs)
    ss = _search_string_from_generated_item(updated)
    assert ss is not None
    assert ss.lane_id == ss.family_key == "delivery_builders"
