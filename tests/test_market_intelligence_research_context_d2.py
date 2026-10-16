"""Tests for lane-aware research context and prompt contract (D2).

Pins:
- Research input includes lane/variant records (lane_evidence section).
- Prompt contains clauses about lane underperformance, hidden pools, planner-readable implications.
- Internal vs external evidence distinguished in prompt instructions.
"""

from __future__ import annotations

import json

from market_intelligence.research_context import (
    _lane_evidence_from_snapshot,
    _lane_execution_summary,
)
from market_intelligence.research_prompts import (
    _dump_bundle,
    build_internal_synthesis_system_prompt,
    build_research_system_prompt,
)


def _sp(
    string_id: int,
    domain_lane: str = "",
    family_key: str = "",
    saves: int = 0,
    candidates_count: int = 50,
    search_intelligence: dict | None = None,
) -> dict:
    return {
        "string_id": string_id,
        "domain_lane": domain_lane,
        "family_key": family_key,
        "saves": saves,
        "candidates_count": candidates_count,
        "result_count": 100,
        "pages_reviewed": 4,
        "facial_yes_count": 5,
        "facial_no_count": 40,
        "search_intelligence": search_intelligence or {},
    }


# -- lane_evidence_from_snapshot --

def test_lane_evidence_includes_winning_and_underperforming():
    analysis = {
        "winning_lanes": [{"lane": "ml-infra", "evidence": "high save rate"}],
        "underperforming_lanes": [{"lane": "platform", "issue": "no saves"}],
    }
    result = _lane_evidence_from_snapshot(None, analysis, [])
    assert len(result["winning_lanes"]) == 1
    assert len(result["underperforming_lanes"]) == 1


def test_lane_evidence_extracts_committed_variants():
    sp = [
        _sp(1, domain_lane="alpha", search_intelligence={
            "best_variant": {"variant_id": "v1", "variant_kind": "precision"},
        }),
        _sp(2, domain_lane="alpha", search_intelligence={
            "best_variant": {"variant_id": "v2", "variant_kind": "recall"},
        }),
    ]
    result = _lane_evidence_from_snapshot(None, {}, sp)
    assert len(result["committed_variants"]) == 2
    assert result["committed_variants"][0]["variant_id"] == "v1"
    assert result["committed_variants"][0]["lane_id"] == "alpha"


def test_lane_evidence_deduplicates_variants():
    sp = [
        _sp(1, domain_lane="a", search_intelligence={
            "best_variant": {"variant_id": "v1", "variant_kind": "x"},
        }),
        _sp(2, domain_lane="a", search_intelligence={
            "best_variant": {"variant_id": "v1", "variant_kind": "x"},
        }),
    ]
    result = _lane_evidence_from_snapshot(None, {}, sp)
    assert len(result["committed_variants"]) == 1


def test_lane_evidence_empty_when_no_analysis():
    result = _lane_evidence_from_snapshot(None, {}, [])
    assert result["winning_lanes"] == []
    assert result["underperforming_lanes"] == []
    assert result["committed_variants"] == []


def test_lane_evidence_includes_abandoned_variants():
    sp = [
        _sp(
            1,
            domain_lane="analog-discovery",
            search_intelligence={
                "variants": [
                    {
                        "variant_id": "v-abandon",
                        "variant_kind": "recall",
                        "status": "abandoned",
                        "lifecycle_reason": "probe_budget_exhausted_no_signal",
                    }
                ],
                "last_variant_decision": {
                    "action": "abandon",
                    "variant_id": "v-abandon",
                    "reason": "probe_budget_exhausted_no_signal",
                },
            },
        ),
    ]
    result = _lane_evidence_from_snapshot(None, {}, sp)
    assert len(result["abandoned_variants"]) >= 1
    assert result["abandoned_variants"][0]["variant_id"] == "v-abandon"


# -- Prompt contract tests --

def test_internal_synthesis_prompt_mentions_lane_evidence():
    prompt = build_internal_synthesis_system_prompt()
    assert "lane_evidence" in prompt
    assert "underperforming lanes" in prompt.lower()
    assert "hidden pools" in prompt.lower()
    assert "planner field" in prompt.lower()


def test_internal_synthesis_prompt_requires_operative_findings():
    prompt = build_internal_synthesis_system_prompt()
    assert "non-operative" in prompt.lower()
    assert "open_questions" in prompt


def test_external_research_prompt_mentions_lane_evidence():
    prompt = build_research_system_prompt()
    assert "lane_evidence" in prompt
    assert "underperform" in prompt.lower()
    assert "planner field" in prompt.lower()


def test_external_research_prompt_distinguishes_evidence_types():
    prompt = build_research_system_prompt()
    assert "supporting_run_refs" in prompt
    assert "evidence_refs" in prompt


def test_dump_bundle_caps_oversized_section():
    small = {"lane": "ml-infra", "count": 3}
    small_dump = _dump_bundle(small)
    assert small_dump == json.dumps(small, indent=2, sort_keys=True)

    oversized = {"payload": "y" * 70_000}
    capped = _dump_bundle(oversized)
    assert "bundle truncated" in capped
    assert len(capped) <= 60_000 + len("\n... [bundle truncated: 99999 of 99999 chars omitted]")
