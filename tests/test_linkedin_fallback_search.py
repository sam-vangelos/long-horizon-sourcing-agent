"""Tests for LinkedIn fallback search — P10 (C4-C6)."""

from __future__ import annotations

import pytest

from linkedin.fallback_search import (
    RESOLUTION_STATES,
    FakeFallbackProvider,
    FallbackCandidate,
    FallbackQuery,
    FallbackResult,
    FallbackSearchProvider,
    assert_fallback_save_safety,
    build_xray_query,
    fallback_result_to_candidate,
    resolve_fallback_candidate,
    validate_fallback_candidate,
)
from shared.sourcing_lanes import (
    ACQUISITION_MODES,
    FALLBACK_ACQUISITION_MODES,
    LaneExecution,
    SearchConstraint,
    SearchHypothesis,
    SearchSlice,
    SourcingLane,
)


def _make_lane(
    *,
    lane_id: str = "fb-lane-1",
    constraints: list[SearchConstraint] | None = None,
) -> SourcingLane:
    return SourcingLane(
        lane_id=lane_id,
        lane_name="Fallback Test Lane",
        hypothesis=SearchHypothesis(
            hypothesis_id="h1",
            label="Sparse pool",
            target_archetype="senior_leader",
            why_this_pool_may_exist="not in Recruiter",
        ),
        slice=SearchSlice(
            slice_id="s1",
            hypothesis_id="h1",
            label="Fallback Slice",
            objective="find via public web",
            constraints=constraints or [],
        ),
        execution=LaneExecution(
            lane_id=lane_id,
            source="linkedin",
            acquisition_mode="xray",
        ),
    )


# ---------------------------------------------------------------------------
# C4: Data model tests
# ---------------------------------------------------------------------------


def test_unresolved_not_save_eligible():
    c = FallbackCandidate(source_mode="xray", source_url="https://linkedin.com/in/test")
    assert c.save_eligible is False


def test_resolved_is_save_eligible():
    c = FallbackCandidate(
        source_mode="xray",
        source_url="https://linkedin.com/in/test",
        recruiter_profile_id="R123",
        resolution_state="resolved_recruiter_profile",
    )
    assert c.save_eligible is True


def test_rejected_not_save_eligible():
    c = FallbackCandidate(
        source_mode="xray",
        source_url="https://linkedin.com/in/test",
        resolution_state="rejected_match",
    )
    assert c.save_eligible is False


def test_persistence_round_trip():
    original = FallbackCandidate(
        source_mode="linkedin_public",
        source_url="https://linkedin.com/in/ada",
        name_hint="Ada Lovelace",
        headline_hint="Mathematician",
        evidence_snippets=("published", "analytical engine"),
        lane_id="lane-1",
        variant_id="v-2",
        recruiter_profile_id=None,
        resolution_state="unresolved",
    )
    restored = FallbackCandidate.from_persistence_dict(original.to_persistence_dict())
    assert restored == original


def test_lane_provenance_survives_round_trip():
    c = FallbackCandidate(
        source_mode="xray",
        source_url="https://linkedin.com/in/test",
        lane_id="my-lane",
        variant_id="my-var",
    )
    restored = FallbackCandidate.from_persistence_dict(c.to_persistence_dict())
    assert restored.lane_id == "my-lane"
    assert restored.variant_id == "my-var"


def test_fallback_modes_subset_of_acquisition_modes():
    assert FALLBACK_ACQUISITION_MODES.issubset(ACQUISITION_MODES)


def test_resolved_without_profile_id_fails_validation():
    c = FallbackCandidate(
        source_mode="xray",
        source_url="https://linkedin.com/in/test",
        resolution_state="resolved_recruiter_profile",
        recruiter_profile_id=None,
    )
    errors = validate_fallback_candidate(c)
    assert any("recruiter_profile_id" in e for e in errors)


# ---------------------------------------------------------------------------
# C5: Provider and query builder tests
# ---------------------------------------------------------------------------


def test_xray_query_includes_site_prefix():
    lane = _make_lane(
        constraints=[
            SearchConstraint(dimension="capability", values=["ML"], execution_surface="source_native"),
        ],
    )
    query = build_xray_query(lane)
    assert query.query_string.startswith("site:linkedin.com/in")


def test_xray_query_includes_lane_constraints():
    lane = _make_lane(
        constraints=[
            SearchConstraint(dimension="capability", values=["workflow orchestration"], execution_surface="source_native"),
            SearchConstraint(dimension="location", values=["San Francisco"], execution_surface="source_native"),
        ],
    )
    query = build_xray_query(lane)
    assert '"workflow orchestration"' in query.query_string
    assert '"San Francisco"' in query.query_string
    assert query.lane_id == "fb-lane-1"


def test_fake_provider_returns_results():
    results = [
        FallbackResult(url="https://linkedin.com/in/alice", snippet="engineer"),
        FallbackResult(url="https://linkedin.com/in/bob", snippet="manager"),
    ]
    provider = FakeFallbackProvider(results)
    out = provider.search(FallbackQuery(query_string="test", lane_id="l1"))
    assert len(out) == 2


def test_fake_provider_records_queries():
    provider = FakeFallbackProvider()
    q1 = FallbackQuery(query_string="a", lane_id="l1")
    q2 = FallbackQuery(query_string="b", lane_id="l2")
    provider.search(q1)
    provider.search(q2)
    assert len(provider.queries_received) == 2


def test_result_to_candidate_is_unresolved():
    result = FallbackResult(url="https://linkedin.com/in/test", snippet="engineer")
    candidate = fallback_result_to_candidate(result, lane_id="lane-1")
    assert candidate.resolution_state == "unresolved"
    assert candidate.save_eligible is False


def test_result_to_candidate_carries_lane_id():
    result = FallbackResult(url="https://linkedin.com/in/test")
    candidate = fallback_result_to_candidate(result, lane_id="lane-42", variant_id="v-7")
    assert candidate.lane_id == "lane-42"
    assert candidate.variant_id == "v-7"


def test_no_result_can_skip_resolution():
    results = [
        FallbackResult(url="https://linkedin.com/in/a", title="A - Senior Engineer"),
        FallbackResult(url="https://linkedin.com/in/b"),
        FallbackResult(url="https://linkedin.com/in/c", snippet="c"),
    ]
    for r in results:
        c = fallback_result_to_candidate(r, lane_id="l1")
        assert c.resolution_state == "unresolved"
        assert c.save_eligible is False


def test_fake_provider_satisfies_protocol():
    provider = FakeFallbackProvider()
    assert isinstance(provider, FallbackSearchProvider)


# ---------------------------------------------------------------------------
# C6: Identity resolution and save guard tests
# ---------------------------------------------------------------------------


def test_unresolved_cannot_save():
    c = FallbackCandidate(source_mode="xray", source_url="https://linkedin.com/in/test")
    with pytest.raises(ValueError, match="cannot be saved"):
        assert_fallback_save_safety(c)


def test_resolved_can_save():
    c = FallbackCandidate(
        source_mode="xray",
        source_url="https://linkedin.com/in/test",
        recruiter_profile_id="R123",
        resolution_state="resolved_recruiter_profile",
    )
    assert_fallback_save_safety(c)  # should not raise


def test_candidate_match_cannot_save():
    c = FallbackCandidate(
        source_mode="xray",
        source_url="https://linkedin.com/in/test",
        recruiter_profile_id="R123",
        resolution_state="candidate_match",
    )
    with pytest.raises(ValueError):
        assert_fallback_save_safety(c)


def test_rejected_preserved_as_evidence():
    original = FallbackCandidate(
        source_mode="xray",
        source_url="https://linkedin.com/in/test",
        evidence_snippets=("senior engineer at BigCo",),
        lane_id="lane-1",
    )
    rejected = resolve_fallback_candidate(original, rejection_reason="wrong person")
    assert rejected.resolution_state == "rejected_match"
    assert rejected.source_url == original.source_url
    assert rejected.evidence_snippets == original.evidence_snippets
    assert rejected.lane_id == original.lane_id


def test_state_machine_full_walk():
    # unresolved → candidate_match → resolved
    c = FallbackCandidate(source_mode="xray", source_url="https://linkedin.com/in/test")
    assert c.resolution_state == "unresolved"

    c2 = resolve_fallback_candidate(c, recruiter_profile_id="R1", match_confidence=0.5)
    assert c2.resolution_state == "candidate_match"
    assert c2.save_eligible is False

    c3 = resolve_fallback_candidate(c2, recruiter_profile_id="R1", match_confidence=0.95)
    assert c3.resolution_state == "resolved_recruiter_profile"
    assert c3.save_eligible is True


def test_low_confidence_stays_candidate_match():
    c = FallbackCandidate(source_mode="xray", source_url="https://linkedin.com/in/test")
    resolved = resolve_fallback_candidate(c, recruiter_profile_id="R1", match_confidence=0.5)
    assert resolved.resolution_state == "candidate_match"


def test_resolved_payload_compatible_with_terminal_json():
    c = FallbackCandidate(
        source_mode="xray",
        source_url="https://linkedin.com/in/test",
        recruiter_profile_id="R123",
        resolution_state="resolved_recruiter_profile",
        lane_id="lane-1",
    )
    d = c.to_persistence_dict()
    # Should be nestable under terminal_payload_json["fallback"]
    terminal = {"decision": "SAVE", "fallback": d}
    assert terminal["fallback"]["resolution_state"] == "resolved_recruiter_profile"
    assert terminal["fallback"]["recruiter_profile_id"] == "R123"
    assert terminal["fallback"]["lane_id"] == "lane-1"
