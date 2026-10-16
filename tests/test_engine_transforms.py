"""Characterization tests for the market_intelligence.engine pure transforms.

These pin the CURRENT behavior of the dict-in/dict-out finding/implication
transforms and the narrative/lane merge helpers so the Phase 4 P4-1 extraction
into ``market_intelligence/_transforms.py`` is provably behavior-preserving.
They import from ``market_intelligence.engine`` (the names' historical home);
after extraction engine re-exports them, so the import path stays valid and
these must remain green before and after the move.
"""

from __future__ import annotations

from market_intelligence.engine import (
    _default_lane_action,
    _default_lane_confidence,
    _default_lane_why,
    _edge_case_submarket_to_external_context,
    _edge_case_submarket_to_talent_pool,
    _false_negative_hypothesis_to_external_context,
    _finding_to_employer_signal,
    _finding_to_external_context,
    _finding_to_talent_pool,
    _implication_to_brief_recommendation,
    _implication_to_planner_diff,
    _implication_to_retrieval_update,
    _merge_lane_entries,
    _merge_market_thesis,
    _merge_narrative_collection,
    _merge_planner_diffs,
    _self_presentation_pattern_to_talent_pool,
    _title_mapping_to_talent_pool,
)


# ---------------------------------------------------------------------------
# finding -> external_context
# ---------------------------------------------------------------------------

def test_finding_to_external_context_happy_path_filters_blank_refs():
    out = _finding_to_external_context(
        {
            "summary": "Senior PMs cluster in fintech",
            "label": "fintech_pm",
            "evidence_refs": ["ev1", "", "ev2"],
            "confidence": 0.8,
        }
    )
    assert out == {
        "claim": "Senior PMs cluster in fintech",
        "label": "fintech_pm",
        "evidence_refs": ["ev1", "ev2"],
        "confidence": 0.8,
    }


def test_finding_to_external_context_none_without_claim():
    assert _finding_to_external_context({"summary": "   ", "evidence_refs": ["e"]}) is None


def test_finding_to_external_context_none_without_evidence():
    assert _finding_to_external_context({"summary": "x", "evidence_refs": []}) is None


# ---------------------------------------------------------------------------
# finding -> talent_pool
# ---------------------------------------------------------------------------

def test_finding_to_talent_pool_core_pool():
    out = _finding_to_talent_pool(
        {
            "kind": "talent_pool",
            "label": "Growth PM",
            "confidence": 0.7,
            "evidence_refs": ["e1"],
            "supporting_run_refs": ["r1"],
            "summary": "core growth",
        }
    )
    assert out == {
        "pool_key": "growth_pm",
        "label": "Growth PM",
        "status": "core_pool",
        "signal_strength": 0.7,
        "supporting_run_refs": ["r1"],
        "evidence_refs": ["e1"],
        "evidence_summary": "core growth",
        "recommended_search_terms": ["Growth PM"],
    }


def test_finding_to_talent_pool_adjacent_for_title_variant():
    out = _finding_to_talent_pool(
        {"kind": "title_variant", "label": "Platform PM", "confidence": 0.6}
    )
    assert out is not None
    assert out["status"] == "adjacent_pool"
    assert out["pool_key"] == "platform_pm"


def test_finding_to_talent_pool_none_for_unknown_kind():
    assert _finding_to_talent_pool({"kind": "employer_cluster", "label": "X"}) is None


def test_finding_to_talent_pool_none_without_label():
    assert _finding_to_talent_pool({"kind": "talent_pool", "label": ""}) is None


# ---------------------------------------------------------------------------
# edge-case submarket / false-negative -> external_context
# ---------------------------------------------------------------------------

def test_edge_case_submarket_to_external_context_appends_why_missed():
    out = _edge_case_submarket_to_external_context(
        {
            "label": "DevRel",
            "summary": "Developer relations folks",
            "why_it_is_easy_to_miss": "Odd titles",
            "evidence_refs": ["e1"],
            "confidence": 0.6,
        }
    )
    assert out == {
        "claim": "Hidden pool: DevRel. Developer relations folks Why this may be missed: Odd titles",
        "evidence_refs": ["e1"],
        "confidence": 0.6,
    }


def test_edge_case_submarket_to_external_context_none_when_incomplete():
    assert _edge_case_submarket_to_external_context(
        {"label": "DevRel", "summary": "", "evidence_refs": ["e1"]}
    ) is None


def test_false_negative_hypothesis_to_external_context_joins_statement():
    out = _false_negative_hypothesis_to_external_context(
        {
            "statement": "We miss X",
            "why_it_matters": "because Y",
            "evidence_refs": ["e1"],
            "confidence": 0.5,
        }
    )
    assert out == {"claim": "We miss X because Y", "evidence_refs": ["e1"], "confidence": 0.5}


# ---------------------------------------------------------------------------
# edge-case submarket / title-mapping / self-presentation -> talent_pool
# ---------------------------------------------------------------------------

def test_edge_case_submarket_to_talent_pool_uses_slug_when_no_key():
    out = _edge_case_submarket_to_talent_pool(
        {
            "label": "DevRel",
            "submarket_key": "",
            "confidence": 0.55,
            "supporting_run_refs": ["r1"],
            "evidence_refs": ["e1"],
            "summary": "sum",
        }
    )
    assert out == {
        "pool_key": "devrel",
        "label": "DevRel",
        "status": "adjacent_pool",
        "signal_strength": 0.55,
        "supporting_run_refs": ["r1"],
        "evidence_refs": ["e1"],
        "evidence_summary": "sum",
        "recommended_search_terms": ["DevRel"],
    }


def test_title_mapping_to_talent_pool_fixed_signal_strength():
    out = _title_mapping_to_talent_pool(
        {
            "title_family": "Forward Deployed",
            "likely_archetype": "Solutions Eng",
            "caveats": "watch overlap",
            "mapping_key": "fde_map",
        }
    )
    assert out is not None
    assert out["pool_key"] == "fde_map"
    assert out["label"] == "Forward Deployed"
    assert out["status"] == "adjacent_pool"
    assert out["signal_strength"] == 0.58
    assert out["evidence_summary"] == "watch overlap"
    assert out["recommended_search_terms"] == ["Forward Deployed"]


def test_title_mapping_to_talent_pool_none_when_incomplete():
    assert _title_mapping_to_talent_pool({"title_family": "X", "likely_archetype": ""}) is None


def test_self_presentation_pattern_to_talent_pool_appends_reason():
    out = _self_presentation_pattern_to_talent_pool(
        {
            "label": "Builders",
            "pattern": "Describe themselves as builders",
            "why_it_causes_false_negatives": "Keyword filters miss them",
            "pattern_key": "",
        }
    )
    assert out is not None
    assert out["pool_key"] == "builders"
    assert out["status"] == "adjacent_pool"
    assert out["signal_strength"] == 0.52
    assert out["evidence_summary"] == (
        "Describe themselves as builders Keyword filters miss them"
    )


# ---------------------------------------------------------------------------
# finding -> employer_signal
# ---------------------------------------------------------------------------

def test_finding_to_employer_signal_happy_path():
    out = _finding_to_employer_signal(
        {
            "kind": "employer_cluster",
            "label": "Stripe",
            "evidence_refs": ["e1"],
            "supporting_run_refs": ["r1"],
            "summary": "fintech alumni",
            "confidence": 0.6,
        }
    )
    assert out == {
        "cluster_key": "stripe",
        "label": "Stripe",
        "status": "positive",
        "supporting_employers": ["Stripe"],
        "supporting_run_refs": ["r1"],
        "evidence_refs": ["e1"],
        "evidence_summary": "fintech alumni",
        "confidence": 0.6,
    }


def test_finding_to_employer_signal_none_for_wrong_kind():
    assert _finding_to_employer_signal({"kind": "talent_pool", "label": "X"}) is None


# ---------------------------------------------------------------------------
# implication -> retrieval_update / brief_recommendation / planner_diff
# ---------------------------------------------------------------------------

def test_implication_to_retrieval_update_probe_adjacent_pool_adds_hypothesis():
    out = _implication_to_retrieval_update(
        {
            "category": "probe_adjacent_pool",
            "suggested_values": ["growth", ""],
            "rationale": "signal seen",
            "recommendation": "Probe growth PMs",
            "expected_effect": "more saves",
        },
        target_field="search_priorities",
    )
    assert out is not None
    assert out["update_type"] == "layer_update"
    assert out["layer_name"] == "capability_proxies"
    assert out["suggested_values"] == ["growth"]
    assert out["edge_case_hypothesis"]["label"] == "Probe growth PMs"
    assert out["edge_case_hypothesis"]["source"] == "market_intel"


def test_implication_to_retrieval_update_none_without_category():
    assert _implication_to_retrieval_update({"category": ""}, target_field="instructions") is None


def test_implication_to_brief_recommendation_happy_path():
    out = _implication_to_brief_recommendation(
        {
            "category": "add_title_family",
            "recommendation": "Add forward-deployed eng",
            "rationale": "repeated signal",
            "implication_id": "imp-1",
            "suggested_values": ["Forward Deployed Engineer"],
            "priority": "high",
            "supporting_run_refs": ["r1"],
            "evidence_refs": ["e1"],
        }
    )
    assert out is not None
    assert out["recommendation_id"] == "imp-1"
    assert out["target_field"] == "additional_search_terms"
    assert out["proposal"] == "Forward Deployed Engineer"
    assert out["reason"] == "repeated signal"
    assert out["confidence"] == 0.8
    assert out["retrieval_update"]["layer_name"] == "entry_signals"


def test_implication_to_planner_diff_add_title_family():
    out = _implication_to_planner_diff(
        {
            "category": "add_title_family",
            "recommendation": "Add forward-deployed eng",
            "rationale": "repeated signal",
            "implication_id": "imp-1",
            "suggested_values": ["Forward Deployed Engineer"],
            "priority": "medium",
            "supporting_run_refs": ["r1"],
            "evidence_refs": ["e1"],
        }
    )
    assert out is not None
    assert out["diff_id"] == "imp-1"
    assert out["action"] == "add"
    assert out["target_type"] == "constraint"
    assert out["payload"]["dimension"] == "title_family"
    assert out["payload"]["values"] == ["Forward Deployed Engineer"]
    assert out["internal_evidence"] == ["r1"]
    assert out["external_evidence"] == ["e1"]
    assert out["confidence"] == 0.7


def test_implication_to_planner_diff_none_for_unmapped_category():
    assert _implication_to_planner_diff(
        {"category": "totally_unknown", "recommendation": "do thing"}
    ) is None


# ---------------------------------------------------------------------------
# merge helpers
# ---------------------------------------------------------------------------

def test_merge_planner_diffs_current_overrides_previous_by_id():
    previous = [{"diff_id": "a", "v": 1}, {"diff_id": "b", "v": 1}]
    current = [{"diff_id": "b", "v": 2}, {"diff_id": "c", "v": 2}]
    merged = _merge_planner_diffs(previous, current)
    by_id = {d["diff_id"]: d["v"] for d in merged}
    assert by_id == {"a": 1, "b": 2, "c": 2}


def test_merge_narrative_collection_merges_by_key():
    out = _merge_narrative_collection(
        key_field="pool_key",
        current=[{"pool_key": "b", "v": 2}],
        previous=[{"pool_key": "a", "v": 1}, {"pool_key": "b", "v": 1}],
        preserve_previous=True,
    )
    by_key = {item["pool_key"]: item["v"] for item in out}
    assert by_key == {"a": 1, "b": 2}


def test_merge_narrative_collection_clears_when_empty_and_not_preserving():
    out = _merge_narrative_collection(
        key_field="pool_key",
        current=[],
        previous=[{"pool_key": "a"}],
        preserve_previous=False,
    )
    assert out == []


def test_merge_market_thesis_falls_back_to_default_when_empty():
    out = _merge_market_thesis(current={}, previous={}, preserve_previous=False)
    assert out == {
        "summary": "Market thesis not yet synthesized.",
        "supply_assessment": "unknown",
        "competition_assessment": "unknown",
        "external_context": [],
    }


def test_merge_market_thesis_preserves_previous_when_requested():
    previous = {
        "summary": "Strong supply",
        "supply_assessment": "deep",
        "competition_assessment": "high",
        "external_context": [],
    }
    out = _merge_market_thesis(current={}, previous=previous, preserve_previous=True)
    assert out == previous


def test_default_lane_helpers_by_status():
    winning = {"status": "winning", "metrics": {"saves": 5}}
    noise = {"status": "noise", "metrics": {}}
    assert _default_lane_why(winning).startswith("This lane is repeatedly")
    assert _default_lane_action(noise) == "Tighten the lane or retire it."
    assert _default_lane_confidence(winning) == 0.8
    assert _default_lane_confidence({"status": "winning", "metrics": {"saves": 1}}) == 0.7
    assert _default_lane_confidence({"status": "mixed", "metrics": {}}) == 0.6


def test_merge_lane_entries_fills_defaults_and_unions_run_refs():
    deterministic = [
        {
            "lane_key": "lane-1",
            "status": "winning",
            "supporting_run_refs": ["r2"],
        }
    ]
    previous = [{"lane_key": "lane-1", "supporting_run_refs": ["r1"], "confidence": 0.9}]
    merged = _merge_lane_entries(
        deterministic_lanes=deterministic,
        generated_lanes=[],
        previous_lanes=previous,
    )
    assert len(merged) == 1
    lane = merged[0]
    assert lane["supporting_run_refs"] == ["r1", "r2"]
    assert lane["why_it_works"].startswith("This lane is repeatedly")
    assert lane["confidence"] == 0.9
