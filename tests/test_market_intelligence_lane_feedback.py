"""Tests for executable planner diffs from market-intel lane feedback (D3).

Pins:
- Diffs validate and round-trip.
- Non-operative prose rejected from planner output.
- Employer/title inventory without internal evidence becomes validation question.
- Conflicting evidence produces validation question, not hard edit.
"""

from __future__ import annotations

import pytest

from market_intelligence.engine import (
    _build_gated_planner_diffs_from_implications,
    _implication_to_planner_diff,
    load_lane_feedback_for_strategy,
)
from shared.brief_iteration import gate_planner_diff
from shared.sourcing_lanes import PlannerDiff, validate_planner_diff


# -- PlannerDiff validation and round-trip --

def test_planner_diff_round_trip():
    diff = PlannerDiff(
        diff_id="d1",
        action="add",
        target_type="hypothesis",
        target_id="hyp-1",
        payload={"label": "ML Infra"},
        internal_evidence=["run:1"],
        external_evidence=["https://example.com"],
        confidence=0.8,
    )
    d = diff.to_dict()
    restored = PlannerDiff.from_dict(d)
    assert restored.diff_id == "d1"
    assert restored.action == "add"
    assert restored.target_type == "hypothesis"
    assert restored.payload == {"label": "ML Infra"}
    assert restored.internal_evidence == ["run:1"]
    assert restored.is_valid()


def test_planner_diff_validation_catches_invalid_action():
    diff = PlannerDiff(diff_id="d1", action="destroy", target_type="hypothesis")
    issues = validate_planner_diff(diff)
    assert any("action" in i.code for i in issues)
    assert not diff.is_valid()


def test_planner_diff_validation_catches_invalid_target():
    diff = PlannerDiff(diff_id="d1", action="add", target_type="alien")
    issues = validate_planner_diff(diff)
    assert any("target_type" in i.code for i in issues)


def test_planner_diff_validation_catches_missing_id():
    diff = PlannerDiff(diff_id="", action="add", target_type="hypothesis")
    issues = validate_planner_diff(diff)
    assert any("diff_id" in i.code for i in issues)


# -- _implication_to_planner_diff --

def test_implication_to_diff_add_hypothesis():
    item = {
        "category": "probe_adjacent_pool",
        "recommendation": "Investigate MLOps engineers",
        "rationale": "Adjacent pool",
        "priority": "high",
        "supporting_run_refs": ["run:1"],
    }
    result = _implication_to_planner_diff(item)
    assert result is not None
    assert result["action"] == "add"
    assert result["target_type"] == "hypothesis"
    assert result["confidence"] == 0.8


def test_implication_to_diff_add_constraint():
    item = {
        "category": "add_title_family",
        "recommendation": "Add ML Platform titles",
        "rationale": "Title fragmentation",
        "suggested_values": ["ML Platform Engineer", "Platform ML"],
        "priority": "medium",
    }
    result = _implication_to_planner_diff(item)
    assert result is not None
    assert result["action"] == "add"
    assert result["target_type"] == "constraint"
    assert result["payload"]["dimension"] == "title_family"
    assert "ML Platform Engineer" in result["payload"]["values"]


def test_implication_to_diff_validation_question():
    item = {
        "category": "validate_hypothesis",
        "recommendation": "Confirm ML infra pool size",
        "rationale": "Uncertain",
        "priority": "low",
    }
    result = _implication_to_planner_diff(item)
    assert result is not None
    assert result["target_type"] == "validation_question"


def test_implication_to_diff_non_compilable_returns_none():
    item = {
        "category": "generic_observation",
        "recommendation": "The market is competitive",
        "rationale": "Everyone knows this",
    }
    result = _implication_to_planner_diff(item)
    assert result is None


def test_implication_to_diff_empty_input():
    assert _implication_to_planner_diff({}) is None
    assert _implication_to_planner_diff({"category": "add_title_family"}) is None


def test_build_gated_planner_diffs_filters_non_compilable():
    items = [
        {
            "category": "probe_adjacent_pool",
            "recommendation": "Investigate MLOps engineers",
            "rationale": "Adjacent pool",
            "supporting_run_refs": ["run:1"],
        },
        {
            "category": "generic_observation",
            "recommendation": "The market is competitive",
            "rationale": "Prose only",
        },
    ]
    diffs = _build_gated_planner_diffs_from_implications(items)
    assert len(diffs) == 1
    assert diffs[0]["target_type"] == "hypothesis"


def test_load_lane_feedback_reads_planner_diffs_from_artifact(tmp_path, monkeypatch):
    import market_intelligence.engine as engine_module

    brief_path = tmp_path / "brief.json"
    brief_path.write_text(
        '{"role_title": "ML Engineer", "geography": "NYC", "role_level": "Senior"}'
    )
    market_dir = tmp_path / "market_intelligence" / "ml_engineer__nyc__senior"
    market_dir.mkdir(parents=True)
    (market_dir / "market-intel.json").write_text(
        '{"planner_diffs": [{"diff_id": "d1", "action": "add", "target_type": "hypothesis", "payload": {}}]}'
    )
    monkeypatch.setattr(
        engine_module,
        "output_root_for_path",
        lambda _path=None: tmp_path,
    )
    feedback = load_lane_feedback_for_strategy(brief_path, output_dir=tmp_path)
    assert len(feedback) == 1
    assert feedback[0]["diff_id"] == "d1"


# -- gate_planner_diff --

def test_gate_passes_valid_diff():
    diff = PlannerDiff(
        diff_id="d1",
        action="add",
        target_type="hypothesis",
        payload={"label": "test"},
        internal_evidence=["run:1"],
    )
    result, warning = gate_planner_diff(diff)
    assert result is diff
    assert warning is None


def test_gate_rejects_invalid_diff():
    diff = PlannerDiff(diff_id="", action="add", target_type="hypothesis")
    result, warning = gate_planner_diff(diff)
    assert result is None
    assert "invalid" in warning


def test_gate_downgrades_employer_inventory_without_internal_evidence():
    diff = PlannerDiff(
        diff_id="d1",
        action="add",
        target_type="constraint",
        payload={
            "dimension": "employer",
            "values": ["Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix"],
        },
        internal_evidence=[],
        external_evidence=["https://example.com"],
    )
    result, warning = gate_planner_diff(diff)
    assert result is not None
    assert result.target_type == "validation_question"
    assert "validation question" in warning


def test_gate_passes_employer_with_internal_evidence():
    diff = PlannerDiff(
        diff_id="d1",
        action="add",
        target_type="constraint",
        payload={
            "dimension": "employer",
            "values": ["Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix"],
        },
        internal_evidence=["run:1"],
        external_evidence=[],
    )
    result, warning = gate_planner_diff(diff)
    assert result is diff
    assert warning is None


def test_gate_downgrades_title_inventory_without_internal_evidence():
    diff = PlannerDiff(
        diff_id="d1",
        action="add",
        target_type="constraint",
        payload={
            "dimension": "title",
            "values": [
                "VP Engineering", "SVP Engineering", "CTO",
                "Director of Engineering", "Head of Engineering",
                "VP Technology",
            ],
        },
        internal_evidence=[],
        external_evidence=["https://example.com"],
    )
    result, warning = gate_planner_diff(diff)
    assert result is not None
    assert result.target_type == "validation_question"


def test_gate_downgrades_retire_with_mixed_evidence():
    diff = PlannerDiff(
        diff_id="d1",
        action="retire",
        target_type="hypothesis",
        target_id="hyp-1",
        internal_evidence=["run:1"],
        external_evidence=["https://example.com"],
    )
    result, warning = gate_planner_diff(diff)
    assert result is not None
    assert result.target_type == "validation_question"
    assert "mixed evidence" in warning.lower()


# -- P3.3: planner-diff lifecycle (retire producer / consumption / expiry) --


def test_retire_diffs_generated_from_saturated_and_noise_lanes():
    from market_intelligence.engine import _build_retire_diffs_from_lanes

    lanes = [
        {
            "lane_key": "canonical_bank",
            "status": "saturated",
            "metrics": {"saves": 0, "duplicate_rate": 0.6},
            "supporting_run_refs": ["linkedin:run-3"],
        },
        {
            "lane_key": "noise_lane",
            "status": "noise",
            "metrics": {"saves": 0, "duplicate_rate": 0.1},
            "supporting_run_refs": [],
        },
        {
            "lane_key": "winning_lane",
            "status": "winning",
            "metrics": {"saves": 5, "duplicate_rate": 0.1},
            "supporting_run_refs": [],
        },
    ]
    diffs = _build_retire_diffs_from_lanes(lanes)
    assert len(diffs) == 2
    by_target = {d["target_id"]: d for d in diffs}
    assert by_target["canonical_bank"]["action"] == "retire"
    assert by_target["canonical_bank"]["target_type"] == "hypothesis"
    assert by_target["noise_lane"]["payload"]["lane_status"] == "noise"
    assert "winning_lane" not in by_target


def test_consumed_diff_is_not_reserved_and_marking_persists(tmp_path, monkeypatch):
    import market_intelligence.engine as engine_module
    from market_intelligence.engine import mark_planner_diffs_consumed

    brief_path = tmp_path / "brief.json"
    brief_path.write_text(
        '{"role_title": "ML Engineer", "geography": "NYC", "role_level": "Senior"}'
    )
    market_dir = tmp_path / "market_intelligence" / "ml_engineer__nyc__senior"
    market_dir.mkdir(parents=True)
    (market_dir / "market-intel.json").write_text(
        '{"planner_diffs": ['
        '{"diff_id": "d1", "action": "add", "target_type": "hypothesis", "payload": {}},'
        '{"diff_id": "d2", "action": "add", "target_type": "constraint", "payload": {}}'
        ']}'
    )
    monkeypatch.setattr(
        engine_module, "output_root_for_path", lambda _path=None: tmp_path
    )
    marked = mark_planner_diffs_consumed(brief_path, ["d1"], output_dir=tmp_path)
    assert marked == 1
    feedback = load_lane_feedback_for_strategy(brief_path, output_dir=tmp_path)
    assert [d["diff_id"] for d in feedback] == ["d2"]
    # Idempotent: marking again marks nothing new.
    assert mark_planner_diffs_consumed(brief_path, ["d1"], output_dir=tmp_path) == 0


def test_planner_diff_lifecycle_expires_unconsumed_and_archives_consumed():
    from market_intelligence._transforms import _apply_planner_diff_lifecycle

    stale = {"diff_id": "old", "action": "add", "target_type": "hypothesis", "payload": {}}
    consumed = {
        "diff_id": "used", "action": "add", "target_type": "hypothesis",
        "payload": {}, "consumed": True, "consumed_at": "2026-07-01T00:00:00Z",
    }

    # Round 1: stale ages to 1, consumed archives immediately.
    active, archived = _apply_planner_diff_lifecycle([stale, consumed], [], [])
    assert [d["diff_id"] for d in archived] == ["used"]
    assert archived[0]["archived_reason"] == "consumed"
    assert len(active) == 1 and active[0]["runs_unconsumed"] == 1

    # Rounds 2-3: stale keeps aging, expires at 3 into the archive.
    active, archived = _apply_planner_diff_lifecycle(active, [], archived)
    assert active[0]["runs_unconsumed"] == 2
    active, archived = _apply_planner_diff_lifecycle(active, [], archived)
    assert active == []
    reasons = {d["diff_id"]: d.get("archived_reason") for d in archived}
    assert reasons["old"] == "expired_unconsumed"

    # A re-emitted diff resets its counter instead of expiring.
    aged = {"diff_id": "fresh", "action": "add", "target_type": "hypothesis",
            "payload": {}, "runs_unconsumed": 2}
    refreshed_emission = {"diff_id": "fresh", "action": "add",
                          "target_type": "hypothesis", "payload": {"v": 2}}
    active, archived2 = _apply_planner_diff_lifecycle([aged], [refreshed_emission], [])
    assert active[0]["runs_unconsumed"] == 0
    assert active[0]["payload"] == {"v": 2}
    assert archived2 == []


# -- P3.5: narrative decay --


def test_narrative_entries_decay_after_five_unsupported_runs():
    from market_intelligence._transforms import (
        _merge_narrative_collection_with_decay,
    )

    entry = {"pool_key": "fintech", "narrative": "deep pool"}
    active = [entry]
    archived_all = []
    for round_number in range(1, 5):
        active, archived = _merge_narrative_collection_with_decay(
            key_field="pool_key",
            current=[],
            previous=active,
            preserve_previous=True,
        )
        archived_all.extend(archived)
        if round_number < 5:
            assert len(active) == 1
            assert active[0]["runs_unsupported"] == round_number
    active, archived = _merge_narrative_collection_with_decay(
        key_field="pool_key", current=[], previous=active, preserve_previous=True,
    )
    archived_all.extend(archived)
    assert active == []
    assert len(archived_all) == 1
    assert archived_all[0]["archived_reason"] == "unsupported_decay"
    assert archived_all[0]["runs_unsupported"] == 5


def test_refreshed_narrative_entry_does_not_decay():
    from market_intelligence._transforms import (
        _merge_narrative_collection_with_decay,
    )

    aged = {"pool_key": "fintech", "narrative": "old", "runs_unsupported": 4}
    active, archived = _merge_narrative_collection_with_decay(
        key_field="pool_key",
        current=[{"pool_key": "fintech", "narrative": "refreshed"}],
        previous=[aged],
        preserve_previous=True,
    )
    assert archived == []
    assert len(active) == 1
    assert active[0]["narrative"] == "refreshed"
    assert active[0]["runs_unsupported"] == 0
    assert active[0]["last_supported_at"]
