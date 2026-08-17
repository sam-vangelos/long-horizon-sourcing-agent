"""Tests for brief-scoped search family memory."""

from pathlib import Path

import pytest

from shared.brief_loader import load_brief
from shared.brief_schema import DomainLaneHint
from shared.schemas import SearchString
from shared.search_memory import (
    build_search_memory_summary,
    extract_dominant_anchors,
    format_search_memory_summary,
    infer_domain_lane,
    normalize_novelty_bucket,
    update_search_memory,
    wilson_lower_bound,
)


HEAD_AI_V2_BRIEF_PATH = str(
    Path(__file__).parent.parent / "config" / "brief-head-ai-lab-nyc-v2.json"
)

if not Path(HEAD_AI_V2_BRIEF_PATH).is_file():
    pytest.skip(
        "Optional brief-head-ai-lab-nyc-v2.json not found under config/.",
        allow_module_level=True,
    )


def test_search_memory_marks_repeated_high_duplicate_canonical_family_exhausted():
    memory = {}

    first = SearchString(
        id=1,
        name="canonical bank cleanup",
        boolean='("Goldman Sachs" OR "JPMorgan") AND ("GenAI" OR "LLM")',
        pages_reviewed=2,
        candidates_count=18,
        duplicates_count=16,
        saves=["A"],
        family_key="canonical_bank_company_first",
        novelty_bucket="canonical",
        domain_lane="capital_markets",
    )
    second = SearchString(
        id=2,
        name="canonical bank cleanup variant",
        boolean='("Morgan Stanley" OR "Goldman Sachs") AND ("GenAI" OR "RAG")',
        pages_reviewed=2,
        candidates_count=15,
        duplicates_count=18,
        saves=[],
        family_key="canonical_bank_company_first",
        novelty_bucket="canonical",
        domain_lane="capital_markets",
    )

    memory = update_search_memory(memory, "3000000006", [first])
    memory = update_search_memory(memory, "3000000006", [second])

    family = memory["families"]["canonical_bank_company_first"]
    assert family["status"] == "exhausted"
    assert "duplicate overlap" in family["status_reason"].lower()

    summary = build_search_memory_summary(memory)
    assert summary["families"][0]["family_key"] == "canonical_bank_company_first"
    assert summary["families"][0]["status"] == "exhausted"
    assert summary["families"][0]["duplicate_rate"] > 0.4


def test_search_memory_tracks_layer_items_and_edge_case_hypotheses():
    memory = {}
    search_string = SearchString(
        id=3,
        name="delivery builders",
        boolean='("deployment engineer") AND ("workflow orchestration") AND ("production")',
        pages_reviewed=2,
        candidates_count=12,
        duplicates_count=2,
        saves=["Ada", "Grace"],
        family_key="fde_delivery_builders",
        novelty_bucket="edge_case",
        domain_lane="general",
        retrieval_recipe={
            "family_id": "fde_delivery_builders",
            "used_layer_item_ids": {
                "entry_signals": ["entry_delivery"],
                "capability_proxies": ["cap_orchestration"],
                "reality_filters": ["real_production"],
            },
            "applied_hypothesis_ids": ["post_sale_builders"],
        },
        retrieval_hypothesis_ids=["post_sale_builders"],
    )

    memory = update_search_memory(memory, "3000000007", [search_string, search_string])
    summary = build_search_memory_summary(memory)

    assert "entry_delivery" in memory["layer_items"]
    assert memory["hypotheses"]["post_sale_builders"]["status"] == "validated"
    assert summary["layer_items"][0]["layer_item_id"] == "entry_delivery"
    assert summary["hypotheses"][0]["hypothesis_id"] == "post_sale_builders"


# ---------------------------------------------------------------------------
# P6 Wave-2 residual: validated-status vs Wilson-confidence honesty.
# `update_search_memory` keeps the volume-independent 2/2 "validated" status
# gate while `confidence` is now the Wilson lower bound on saves/candidates —
# a hypothesis can be status=validated at confidence~0.10 on tiny volume.
# `format_search_memory_summary` used to render only `status=`, hiding that
# the "validated" label rests on almost no evidence. The formatted text must
# carry both honestly without changing the status gate or the summary shape.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Optional brief argument behavior (Slice 2 Commit 1)
# ---------------------------------------------------------------------------


def test_infer_domain_lane_returns_general_when_brief_is_none():
    """Without a brief, an unmatched lane string defaults to 'general'."""
    lane = infer_domain_lane(
        None,
        '("trade surveillance workflow") AND ("agentic")',
        "Edge-case capital markets workflow population.",
        brief=None,
    )
    assert lane == "general"


def test_infer_domain_lane_returns_brief_lane_when_hints_match():
    """A brief with matching domain_lane_hints drives the lane decision."""
    brief = load_brief(HEAD_AI_V2_BRIEF_PATH)
    brief.domain_lane_hints = [
        DomainLaneHint(
            lane="capital_markets",
            patterns=["trade surveillance", "post-trade", "collateral workflow"],
        ),
        DomainLaneHint(
            lane="clinical_workflows",
            patterns=["telehealth triage", "remote patient monitoring"],
        ),
    ]

    capital_lane = infer_domain_lane(
        None,
        '("trade surveillance workflow") AND ("agentic")',
        "Capital-markets edge-case workflow population.",
        brief=brief,
    )
    clinical_lane = infer_domain_lane(
        None,
        '("telehealth triage" OR "remote patient monitoring")',
        "Clinical edge-case population.",
        brief=brief,
    )
    fallback_lane = infer_domain_lane(
        None,
        "totally unrelated boolean",
        "Nothing in the brief hints matches this.",
        brief=brief,
    )

    assert capital_lane == "capital_markets"
    assert clinical_lane == "clinical_workflows"
    assert fallback_lane == "general"


def test_infer_domain_lane_prefers_explicit_value_over_brief():
    """Explicit metadata is always honored before the brief fallback."""
    brief = load_brief(HEAD_AI_V2_BRIEF_PATH)
    brief.domain_lane_hints = [
        DomainLaneHint(lane="capital_markets", patterns=["trade surveillance"])
    ]

    lane = infer_domain_lane(
        "insurance",
        '("trade surveillance workflow")',
        "explicit lane should win",
        brief=brief,
    )
    assert lane == "insurance"


def test_normalize_novelty_bucket_defaults_to_canonical_without_brief():
    """When no explicit value is supplied, the normalizer defaults to canonical."""
    bucket = normalize_novelty_bucket(
        None,
        '("Goldman Sachs" OR "JPMorgan") AND ("GenAI")',
        "Big-bank cleanup string.",
        brief=None,
    )
    assert bucket == "canonical"


def test_normalize_novelty_bucket_honors_explicit_value():
    """Explicit ``edge_case`` / ``canonical`` values must be preserved."""
    assert (
        normalize_novelty_bucket("edge_case", "boolean", "rationale", brief=None)
        == "edge_case"
    )
    assert (
        normalize_novelty_bucket("canonical", "boolean", "rationale", brief=None)
        == "canonical"
    )


def test_infer_domain_lane_normalizes_brief_hint_lane_label():
    """A `DomainLaneHint` lane written in human form (e.g. 'Capital Markets')
    must be canonicalized to the same shape the explicit-value path produces
    ('capital_markets'). The brief-hint fallback path and the explicit-value
    path are equivalent normalizers."""
    brief = load_brief(HEAD_AI_V2_BRIEF_PATH)
    brief.domain_lane_hints = [
        DomainLaneHint(lane="Capital Markets", patterns=["jpmorgan", "goldman"]),
    ]

    hint_lane = infer_domain_lane(
        None,
        '("JPMorgan" OR "Goldman") AND ("agentic")',
        "Brief-hint match path.",
        brief=brief,
    )
    explicit_lane = infer_domain_lane(
        "Capital Markets",
        boolean="",
        rationale="",
        brief=None,
    )

    assert hint_lane == "capital_markets"
    assert hint_lane == explicit_lane


def test_infer_domain_lane_general_default_still_holds_with_normalization():
    """Existing 'general' default behavior must not regress: a brief with no
    lane hints, called with `brief=None` or with a non-matching boolean,
    still returns 'general'."""
    none_lane = infer_domain_lane(
        None,
        "totally unrelated boolean",
        "Nothing to match.",
        brief=None,
    )
    assert none_lane == "general"

    brief = load_brief(HEAD_AI_V2_BRIEF_PATH)
    brief.domain_lane_hints = [
        DomainLaneHint(lane="Capital Markets", patterns=["jpmorgan"]),
    ]
    no_match_lane = infer_domain_lane(
        None,
        "totally unrelated boolean",
        "No hint pattern matches this text.",
        brief=brief,
    )
    assert no_match_lane == "general"


def test_extract_dominant_anchors_uses_generic_tokenizer_only():
    """The extractor should rely on the lexical tokenizer/stopword path; the old
    `_ANCHOR_PHRASES` short-circuit is gone, and AI/BFSI tokens are no longer
    filtered out as stopwords — they should appear in the anchors instead."""
    text = '("trade surveillance workflow") AND ("GenAI") AND ("BFSI") AND ("LLM")'
    anchors = extract_dominant_anchors(text, limit=8)

    assert "trade" in anchors
    assert "surveillance" in anchors
    assert "genai" in anchors
    assert "bfsi" in anchors
    assert "llm" in anchors
    assert "trade surveillance" not in anchors  # the legacy phrase pre-pass is gone
