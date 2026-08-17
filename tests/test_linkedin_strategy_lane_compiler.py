"""Tests for strategy lane compiler wiring — P9."""

from __future__ import annotations

from linkedin.strategy_lane_compiler import apply_linkedin_lane_compiler_to_plan
from shared.schemas import ExecutionPlan
from shared.sourcing_lanes import (
    LaneExecution,
    SearchConstraint,
    SearchHypothesis,
    SearchSlice,
    SourcingLane,
)


def _lane_dict(*, lane_id: str = "senior-pool", mode: str = "linkedin_hybrid") -> dict:
    lane = SourcingLane(
        lane_id=lane_id,
        lane_name="Senior Pool",
        hypothesis=SearchHypothesis(
            hypothesis_id="h1",
            label="Senior",
            target_archetype="leader",
            why_this_pool_may_exist="banks",
        ),
        slice=SearchSlice(
            slice_id="s1",
            hypothesis_id="h1",
            label="Slice",
            objective="find leaders",
            constraints=[
                SearchConstraint(
                    dimension="title",
                    values=["VP Engineering"],
                    execution_surface="linkedin_title_filter",
                    operator="prefer",
                )
            ],
        ),
        execution=LaneExecution(
            lane_id=lane_id,
            source="linkedin",
            acquisition_mode=mode,
            boolean_strategy={"root_boolean": '"VP" AND engineering'},
            structured_filters={"titles": ["VP Engineering"]},
        ),
    )
    return lane.to_dict()


def test_apply_lane_compiler_attaches_metadata_to_generated_string():
    plan = ExecutionPlan(
        strategy_rationale="test",
        generated_strings=[
            {
                "boolean": '"VP" AND engineering',
                "rationale": "senior leaders",
                "lane_id": "senior-pool",
            }
        ],
        sourcing_lanes=[_lane_dict()],
    )
    wired = apply_linkedin_lane_compiler_to_plan(plan)
    assert wired == 1
    item = plan.generated_strings[0]
    assert item["acquisition_mode"] == "linkedin_hybrid"
    assert "compiler" in item["lane_snapshot"]
    assert plan.sourcing_lanes[0]["lane_compiler"]["acquisition_mode"] == "linkedin_hybrid"


def test_apply_lane_compiler_noop_without_sourcing_lanes():
    plan = ExecutionPlan(
        strategy_rationale="test",
        generated_strings=[{"boolean": "test", "rationale": "r"}],
    )
    assert apply_linkedin_lane_compiler_to_plan(plan) == 0
