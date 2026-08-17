"""Tests for fallback acquisition orchestrator wiring — P10."""

from __future__ import annotations

from linkedin.fallback_acquisition import (
    discover_fallback_candidates_for_string,
    fallback_mode_for_search_string,
    record_fallback_discovery,
    sourcing_lane_for_search_string,
)
from linkedin.fallback_search import FakeFallbackProvider, FallbackResult
from shared.schemas import ExecutionPlan, SearchString
from shared.sourcing_lanes import (
    LaneExecution,
    SearchConstraint,
    SearchHypothesis,
    SearchSlice,
    SourcingLane,
)


def _plan_with_fallback_lane() -> ExecutionPlan:
    lane = SourcingLane(
        lane_id="sparse-pool",
        lane_name="Sparse Pool",
        hypothesis=SearchHypothesis(
            hypothesis_id="h1",
            label="Sparse",
            target_archetype="leader",
            why_this_pool_may_exist="hidden",
        ),
        slice=SearchSlice(
            slice_id="s1",
            hypothesis_id="h1",
            label="Slice",
            objective="public discovery",
            constraints=[
                SearchConstraint(
                    dimension="capability",
                    values=["workflow orchestration"],
                    execution_surface="source_native",
                )
            ],
        ),
        execution=LaneExecution(
            lane_id="sparse-pool",
            source="linkedin",
            acquisition_mode="xray",
        ),
    )
    return ExecutionPlan(
        strategy_rationale="test",
        sourcing_lanes=[lane.to_dict()],
        generated_strings=[
            {
                "boolean": "workflow AND orchestration",
                "lane_id": "sparse-pool",
                "acquisition_mode": "xray",
            }
        ],
    )


def test_fallback_mode_detected_from_search_string():
    plan = _plan_with_fallback_lane()
    ss = SearchString(id=1, name="test", boolean="x", lane_id="sparse-pool", acquisition_mode="xray")
    assert fallback_mode_for_search_string(plan, ss) == "xray"


def test_sourcing_lane_resolved_from_plan():
    plan = _plan_with_fallback_lane()
    ss = SearchString(id=1, name="test", boolean="x", lane_id="sparse-pool")
    lane = sourcing_lane_for_search_string(plan, ss)
    assert lane is not None
    assert lane.lane_id == "sparse-pool"


def test_discover_fallback_candidates_uses_provider():
    plan = _plan_with_fallback_lane()
    ss = SearchString(id=1, name="test", boolean="x", lane_id="sparse-pool", acquisition_mode="xray")
    provider = FakeFallbackProvider(
        [FallbackResult(url="https://linkedin.com/in/alice", snippet="engineer")]
    )
    candidates, query = discover_fallback_candidates_for_string(plan, ss, provider=provider)
    assert query is not None
    assert query.query_string.startswith("site:linkedin.com/in")
    assert len(candidates) == 1
    assert candidates[0].save_eligible is False


def test_record_fallback_discovery_emits_events_and_persists():
    ss = SearchString(id=1, name="test", boolean="x", lane_id="sparse-pool", acquisition_mode="xray")
    events: list[tuple[str, dict]] = []
    persisted: list[dict] = []

    record_fallback_discovery(
        candidates=[
            discover_fallback_candidates_for_string(
                _plan_with_fallback_lane(),
                ss,
                provider=FakeFallbackProvider(
                    [FallbackResult(url="https://linkedin.com/in/alice")]
                ),
            )[0][0]
        ],
        query=discover_fallback_candidates_for_string(
            _plan_with_fallback_lane(),
            ss,
            provider=FakeFallbackProvider([]),
        )[1],
        search_string=ss,
        trigger_reason="variant_abandoned",
        record_event=lambda **kwargs: events.append((kwargs["event_type"], kwargs["payload"])),
        append_candidate=persisted.append,
    )

    assert any(event_type == "fallback_discovery_attempt" for event_type, _ in events)
    assert len(persisted) == 1
