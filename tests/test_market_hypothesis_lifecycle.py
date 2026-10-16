"""P3.4 — hypothesis merge lifecycle instead of clobbering."""

import pytest

from market_intelligence.agent_backends import CriticResult, PlannerResult
from market_intelligence.engine import _build_agent_state
from market_intelligence.schema import (
    MarketHypothesis,
    MarketHypothesisStatusError,
    MarketIdentity,
    MarketIntelAgentState,
)


def _identity() -> MarketIdentity:
    return MarketIdentity(
        market_key="mk",
        role_title="ML Engineer",
        role_level="Senior",
        geography="NYC",
        channels_seen=[],
        brief_ids_seen=[],
        brief_versions_seen=[],
    )


def _hypothesis(hid: str, **overrides) -> dict:
    base = {
        "hypothesis_id": hid,
        "statement": f"Hypothesis {hid}",
        "status": "active",
        "confidence": 0.5,
        "rationale": "r",
        "section_targets": ["lane_intelligence"],
        "first_seen_at": "2026-07-01T00:00:00Z",
        "last_seen_at": "2026-07-01T00:00:00Z",
        "supporting_run_refs": ["linkedin:run-1"],
    }
    base.update(overrides)
    return base


def _state(hypotheses: list[MarketHypothesis]) -> MarketIntelAgentState:
    return MarketIntelAgentState(
        schema_version=1,
        market_key="mk",
        updated_at="2026-07-01T00:00:00Z",
        active_hypotheses=hypotheses,
    )


def test_heuristic_updates_no_longer_collapse_the_hypothesis_set():
    """P3.4 red-first: the planner's <=1-hypothesis output used to REPLACE the
    whole active set. Now: merge by hypothesis_id, absent ones persist."""

    previous = _state(
        [
            MarketHypothesis.from_dict(_hypothesis("h1")),
            MarketHypothesis.from_dict(_hypothesis("h2")),
        ]
    )
    planner = PlannerResult(active_hypotheses=[_hypothesis("h3")])

    state = _build_agent_state(
        market_identity=_identity(),
        evidence_batches=[],
        previous_agent_state=previous,
        planner_result=planner,
        critic_result=CriticResult(),
        external_result=None,
    )
    ids = sorted(h.hypothesis_id for h in state.active_hypotheses)
    assert ids == ["h1", "h2", "h3"]
    by_id = {h.hypothesis_id: h for h in state.active_hypotheses}
    assert by_id["h1"].unrefreshed_runs == 1
    assert by_id["h3"].unrefreshed_runs == 0

    # A second heuristic update: still no collapse.
    state2 = _build_agent_state(
        market_identity=_identity(),
        evidence_batches=[],
        previous_agent_state=state,
        planner_result=PlannerResult(active_hypotheses=[_hypothesis("h3")]),
        critic_result=CriticResult(),
        external_result=None,
    )
    assert sorted(h.hypothesis_id for h in state2.active_hypotheses) == ["h1", "h2", "h3"]
    assert {h.hypothesis_id: h.unrefreshed_runs for h in state2.active_hypotheses}["h1"] == 2


def test_unrefreshed_hypothesis_retires_to_resolved_after_five_runs():
    stale = MarketHypothesis.from_dict(_hypothesis("h-old"))
    stale.unrefreshed_runs = 4
    previous = _state([stale])

    state = _build_agent_state(
        market_identity=_identity(),
        evidence_batches=[],
        previous_agent_state=previous,
        planner_result=PlannerResult(),
        critic_result=CriticResult(),
        external_result=None,
    )
    assert state.active_hypotheses == []
    resolved = {h.hypothesis_id: h for h in state.resolved_hypotheses}
    assert resolved["h-old"].status == "resolved"


def test_hypothesis_status_is_typed_on_load():
    with pytest.raises(MarketHypothesisStatusError, match="status must be one of"):
        MarketHypothesis.from_dict(_hypothesis("h-bad", status="vibing"))
