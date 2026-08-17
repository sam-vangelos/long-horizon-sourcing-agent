"""Schema and research-backend tests for market intelligence."""

from __future__ import annotations

import copy

import pytest

from market_intelligence.research_agent import _validate_open_questions
from market_intelligence.schema import (
    MarketIntelAgentState,
    MarketIntelArtifact,
    NARRATIVE_SECTION_POLICIES,
    PROVENANCE_BEARING_SECTIONS,
    render_market_intel_markdown,
    render_market_intel_technical_markdown,
    sanitize_market_intel_payload,
)


def _base_artifact_dict() -> dict:
    return {
        "schema_version": 1,
        "artifact_version": 1,
        "market_identity": {
            "market_key": "head_of_applied_ai_lab__nyc__l8_l9",
            "role_title": "Head of Applied AI Lab",
            "role_level": "L8/L9",
            "geography": "NYC",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["Head of Applied AI Lab"],
            "brief_versions_seen": ["2.1"],
        },
        "freshness": {
            "artifact_updated_at": "2026-04-08T12:00:00+00:00",
            "internal_data_through": "2026-04-08T12:00:00+00:00",
            "external_research_through": "",
            "staleness_days": 0,
        },
        "evidence_index": {
            "runs": [
                {
                    "run_ref": "linkedin:output",
                    "source": "linkedin",
                    "output_dir": "/tmp/output",
                    "brief_version": "2.1",
                    "generated_at": "2026-04-08T12:00:00+00:00",
                }
            ],
            "external_sources": [],
        },
        "aggregate_metrics": {
            "run_count": 1,
            "saved_count": 12,
            "rejected_count": 28,
            "facial_yes_rate": 0.22,
            "save_rate": 0.07,
            "candidate_volume_by_channel": {"linkedin": 180},
        },
        "channel_summaries": {
            "linkedin": {
                "run_count": 1,
                "top_lane_keys": ["research_copilot_asset_mgmt"],
                "saturated_lane_keys": [],
            }
        },
        "lane_intelligence": [
            {
                "lane_key": "research_copilot_asset_mgmt",
                "domain_lane": "asset_management",
                "novelty_bucket": "edge_case",
                "status": "winning",
                "first_seen_at": "2026-04-08T12:00:00+00:00",
                "last_seen_at": "2026-04-08T12:00:00+00:00",
                "supporting_run_refs": ["linkedin:output"],
                "metrics": {
                    "strings_seen": 1,
                    "candidates_seen": 22,
                    "saves": 8,
                    "save_rate": 0.36,
                    "duplicate_rate": 0.04,
                },
                "dominant_anchors": ["research copilot", "asset management"],
                "why_it_works": "Strong workflow specificity.",
                "recommended_action": "Keep this lane active.",
                "confidence": 0.8,
            }
        ],
        "talent_pool_intelligence": [
            {
                "pool_key": "bfsi_native_genai_converts",
                "label": "BFSI-native GenAI converts",
                "status": "core_pool",
                "signal_strength": 0.9,
                "supporting_run_refs": ["linkedin:output"],
                "evidence_summary": "Observed repeatedly in saved candidates.",
                "recommended_search_terms": ["research copilot"],
            }
        ],
        "noise_patterns": [
            {
                "pattern_key": "product_heavy_ai_leadership",
                "label": "Product-heavy AI leadership",
                "severity": "medium",
                "supporting_run_refs": ["linkedin:output"],
                "mitigations": ["Strengthen builder-authorship language."],
                "confidence": 0.7,
            }
        ],
        "employer_signal_intelligence": [
            {
                "cluster_key": "jpmorgan",
                "label": "JPMorgan",
                "status": "positive",
                "supporting_employers": ["JPMorgan"],
                "supporting_run_refs": ["linkedin:output"],
                "evidence_summary": "Repeatedly appeared among saved candidates.",
                "confidence": 0.7,
            }
        ],
        "candidate_signal_summary": {
            "standout_signals": ["builder"],
            "borderline_signals": ["scope"],
            "disqualifying_signals": ["product only"],
        },
        "market_thesis": {
            "summary": "The market looks moderate.",
            "supply_assessment": "moderate",
            "competition_assessment": "high",
            "external_context": [
                {
                    "claim": "Example employer is hiring.",
                    "evidence_refs": ["web:https://example.com/jobs"],
                    "confidence": 0.8,
                }
            ],
        },
        "brief_recommendations": [
            {
                "recommendation_id": "rec-terms",
                "target_field": "additional_search_terms",
                "proposal": "Add payments and research-copilot terms.",
                "reason": "Winning lanes point there.",
                "supporting_run_refs": ["linkedin:output"],
                "confidence": 0.8,
            }
        ],
        "open_questions": [
            {
                "question": "How should we cover payments more directly?",
                "priority": "medium",
                "next_step": "Run one payments-focused lane next cycle.",
                "supporting_run_refs": ["linkedin:output"],
            }
        ],
    }


def test_render_market_intel_markdown_does_not_publish_critic_review_text():
    artifact = _base_artifact_dict()
    artifact["market_thesis"]["summary"] = (
        "The market thesis summary is well-written and directionally correct. "
        "However, it needs the following adjustments: WEAKEN the second claim."
    )

    rendered = render_market_intel_markdown(MarketIntelArtifact.from_dict(artifact))

    assert "well-written and directionally correct" not in rendered
    assert "## Executive Summary" in rendered
    assert "Next-run priority:" in rendered


def test_sanitize_market_intel_payload_strips_review_meta_summary():
    artifact = _base_artifact_dict()
    artifact["market_thesis"]["summary"] = (
        "The market thesis summary is well-written and directionally correct. "
        "However, it needs the following adjustments: KEEP the first claim."
    )

    cleaned = sanitize_market_intel_payload(artifact)

    assert cleaned["market_thesis"]["summary"] == ""


def test_market_intel_artifact_rejects_unsourced_open_questions():
    artifact = _base_artifact_dict()
    artifact["open_questions"] = [
        {
            "question": "What should we investigate next?",
            "priority": "medium",
            "next_step": "Look into it.",
        }
    ]
    with pytest.raises(ValueError, match="open_questions"):
        MarketIntelArtifact.from_dict(artifact)


def test_market_intel_artifact_accepts_open_questions_with_supporting_run_refs():
    artifact = _base_artifact_dict()
    artifact["open_questions"] = [
        {
            "question": "What should we investigate next?",
            "priority": "medium",
            "next_step": "Look into it.",
            "supporting_run_refs": ["linkedin:output"],
        }
    ]
    loaded = MarketIntelArtifact.from_dict(artifact)
    assert loaded.open_questions[0]["supporting_run_refs"] == ["linkedin:output"]


def test_market_intel_artifact_accepts_open_questions_with_evidence_refs():
    artifact = _base_artifact_dict()
    artifact["open_questions"] = [
        {
            "question": "What should we investigate next?",
            "priority": "medium",
            "next_step": "Look into it.",
            "evidence_refs": ["web:https://example.com/jobs"],
        }
    ]
    loaded = MarketIntelArtifact.from_dict(artifact)
    assert loaded.open_questions[0]["evidence_refs"] == ["web:https://example.com/jobs"]


@pytest.mark.parametrize(
    ("section_name", "mutate"),
    [
        (
            "lane_intelligence",
            lambda artifact: artifact["lane_intelligence"].__setitem__(
                0,
                {
                    "lane_key": "research_copilot_asset_mgmt",
                    "domain_lane": "asset_management",
                    "novelty_bucket": "edge_case",
                    "status": "winning",
                    "first_seen_at": "2026-04-08T12:00:00+00:00",
                    "last_seen_at": "2026-04-08T12:00:00+00:00",
                    "supporting_run_refs": [],
                    "metrics": {"strings_seen": 1},
                    "dominant_anchors": ["research copilot"],
                },
            ),
        ),
        (
            "talent_pool_intelligence",
            lambda artifact: artifact["talent_pool_intelligence"].__setitem__(
                0,
                {
                    "pool_key": "bfsi_native_genai_converts",
                    "label": "BFSI-native GenAI converts",
                    "status": "core_pool",
                    "signal_strength": 0.9,
                    "evidence_summary": "Observed repeatedly.",
                },
            ),
        ),
        (
            "noise_patterns",
            lambda artifact: artifact["noise_patterns"].__setitem__(
                0,
                {
                    "pattern_key": "product_heavy_ai_leadership",
                    "label": "Product-heavy AI leadership",
                    "severity": "medium",
                    "mitigations": ["Strengthen builder language."],
                },
            ),
        ),
        (
            "employer_signal_intelligence",
            lambda artifact: artifact["employer_signal_intelligence"].__setitem__(
                0,
                {
                    "cluster_key": "jpmorgan",
                    "label": "JPMorgan",
                    "status": "positive",
                    "supporting_employers": ["JPMorgan"],
                },
            ),
        ),
        (
            "brief_recommendations",
            lambda artifact: artifact["brief_recommendations"].__setitem__(
                0,
                {
                    "recommendation_id": "rec-terms",
                    "target_field": "additional_search_terms",
                    "proposal": "Add payments.",
                    "reason": "Winning lane evidence.",
                    "confidence": 0.8,
                },
            ),
        ),
        (
            "open_questions",
            lambda artifact: artifact["open_questions"].__setitem__(
                0,
                {
                    "question": "How should we cover payments more directly?",
                    "priority": "medium",
                    "next_step": "Run a payments lane.",
                },
            ),
        ),
        (
            "market_thesis.external_context",
            lambda artifact: artifact["market_thesis"].__setitem__(
                "external_context",
                [
                    {
                        "claim": "Example employer is hiring.",
                        "evidence_refs": [],
                        "confidence": 0.8,
                    }
                ],
            ),
        ),
    ],
)
def test_provenance_bearing_sections_reject_unsourced_records(section_name, mutate):
    artifact = _base_artifact_dict()
    mutate(artifact)
    with pytest.raises(ValueError, match=section_name.split(".")[0]):
        MarketIntelArtifact.from_dict(artifact)


def test_provenance_policy_explicitly_includes_open_questions():
    expected = {
        "lane_intelligence",
        "talent_pool_intelligence",
        "noise_patterns",
        "employer_signal_intelligence",
        "brief_recommendations",
        "open_questions",
        "market_thesis.external_context",
    }
    assert set(PROVENANCE_BEARING_SECTIONS) == expected
    assert set(NARRATIVE_SECTION_POLICIES) == expected


def test_validate_open_questions_keeps_only_sourced_entries():
    items = [
        {
            "question": "Valid question",
            "priority": "high",
            "next_step": "Investigate",
            "evidence_refs": ["", "web:https://example.com/jobs", "  "],
        },
        {
            "question": "Unsourced question",
            "priority": "medium",
            "next_step": "Investigate",
        },
        "not-a-dict",
        {
            "question": "",
            "priority": "low",
            "next_step": "Investigate",
            "evidence_refs": ["web:https://example.com/other"],
        },
    ]
    assert _validate_open_questions(items) == [
        {
            "question": "Valid question",
            "priority": "high",
            "next_step": "Investigate",
            "supporting_run_refs": [],
            "evidence_refs": ["web:https://example.com/jobs"],
        }
    ]


def test_market_intel_artifact_accepts_section_generation_metadata_and_renders_it():
    artifact = _base_artifact_dict()
    artifact["section_generation_metadata"] = {
        "lane_intelligence": {
            "generation_mode": "llm_internal",
            "quality_level": "medium",
            "updated_at": "2026-04-08T12:00:00+00:00",
            "notes": ["Derived from internal synthesis."],
            "supporting_run_refs": ["linkedin:output"],
            "evidence_refs": [],
        }
    }
    loaded = MarketIntelArtifact.from_dict(artifact)
    markdown = render_market_intel_markdown(loaded)
    technical_markdown = render_market_intel_technical_markdown(loaded)
    assert loaded.section_generation_metadata["lane_intelligence"]["generation_mode"] == "llm_internal"
    assert "## Executive Summary" in markdown
    assert "## What Changes for the Next Sourcing Run" in markdown
    assert "## Pass Diagnostics" in technical_markdown
    assert "Generation mode: llm_internal" not in markdown
    assert "Generation mode: llm_internal" in technical_markdown


def test_market_intel_agent_state_validates_hypotheses_and_unknowns():
    state = MarketIntelAgentState.from_dict(
        {
            "schema_version": 1,
            "market_key": "head_of_applied_ai_lab__nyc__l8_l9",
            "updated_at": "2026-04-08T12:00:00+00:00",
            "active_hypotheses": [
                {
                    "hypothesis_id": "hyp-001",
                    "statement": "Research-copilot phrasing remains the strongest lane.",
                    "status": "active",
                    "confidence": 0.78,
                    "rationale": "Repeated across run evidence.",
                    "section_targets": ["lane_intelligence"],
                    "first_seen_at": "2026-04-08T12:00:00+00:00",
                    "last_seen_at": "2026-04-08T12:00:00+00:00",
                    "supporting_run_refs": ["linkedin:output"],
                }
            ],
            "resolved_hypotheses": [],
            "open_unknowns": [
                {
                    "question": "How should we cover payments more directly?",
                    "priority": "medium",
                    "next_step": "Run one payments lane next cycle.",
                    "supporting_run_refs": ["linkedin:output"],
                }
            ],
            "research_backlog": [
                {
                    "opportunity_id": "opp-001",
                    "question": "Which employers are actively hiring?",
                    "priority": "high",
                    "status": "queued",
                    "reason": "Needed for the next market thesis pass.",
                    "supporting_run_refs": ["linkedin:output"],
                }
            ],
            "source_registry": [],
            "confidence_by_claim_area": {"market_thesis": 0.62},
            "prior_advisories": [],
            "section_generation_metadata": {
                "market_thesis": {
                    "generation_mode": "heuristic",
                    "quality_level": "low",
                    "updated_at": "2026-04-08T12:00:00+00:00",
                    "notes": ["Single-run only."],
                    "supporting_run_refs": ["linkedin:output"],
                    "evidence_refs": [],
                }
            },
        }
    )
    assert state.active_hypotheses[0].hypothesis_id == "hyp-001"
    assert state.open_unknowns[0]["supporting_run_refs"] == ["linkedin:output"]
