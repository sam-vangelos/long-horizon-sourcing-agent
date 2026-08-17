"""P7 Stage A (plans/sourcing-rigor-hardening.md) — lane vocabulary is
brief-derived, never the historical BFSI example list.

Standalone from tests/test_linkedin_strategy.py deliberately: that module
skips wholesale when optional local config briefs are absent, and these
assertions must run everywhere (they lock the lane-collapse fix).
"""

from __future__ import annotations

from shared.brief_loader import Brief
from shared.brief_schema import DomainLaneHint
from linkedin.strategy import _build_strategy_system


def _minimal_brief(**overrides) -> Brief:
    brief = Brief(
        id="lane-vocab-test",
        role_title="Director of Supply Chain Operations",
        role_description="Owns network design and S&OP for a national retailer.",
        kit_url="",
        linkedin_project="",
        linkedin_project_id="",
        minimum_bar="8+ years owning network-level design.",
        archetypes=[{"name": "Network designer"}],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
    )
    for key, value in overrides.items():
        setattr(brief, key, value)
    return brief


def test_strategy_system_domain_lane_guidance_is_brief_derived_not_bfsi():
    brief = _minimal_brief(domain_lane_hints=[])

    system = _build_strategy_system(brief, has_kit=False, use_layered_retrieval=False)

    # The lane-collapse root cause — a BFSI-only example list teaching every
    # non-BFSI brief that the alternative is "general" — is gone.
    assert "capital_markets, risk_compliance" not in system
    assert "bfsi_vendors, general" not in system
    # The replacement instructs deriving brief-specific lanes.
    assert "DERIVE 3-6 lane labels" in system
    assert 'never default every string to "general"' in system


def test_strategy_system_renders_declared_lanes_from_brief_hints():
    brief = _minimal_brief(
        domain_lane_hints=[
            DomainLaneHint(lane="parcel_carriers", patterns=["fedex", "ups"]),
            DomainLaneHint(lane="retail_distribution", patterns=["omnichannel"]),
        ]
    )

    system = _build_strategy_system(brief, has_kit=False, use_layered_retrieval=False)

    assert "## Declared Domain Lanes" in system
    assert "parcel_carriers, retail_distribution" in system
    assert 'use "general" only for a string that genuinely fits none' in system


def test_strategy_system_omits_declared_lanes_section_without_hints():
    brief = _minimal_brief(domain_lane_hints=[])

    system = _build_strategy_system(brief, has_kit=False, use_layered_retrieval=False)

    assert "## Declared Domain Lanes" not in system


def test_opening_checkpoint_guidance_is_vertical_agnostic():
    """Codex review, Wave 1: the opening-checkpoint adaptation block carried
    'BFSI / market-institution' into every brief's mid-run guidance."""
    from linkedin.strategy import _OPENING_CHECKPOINT_GUIDANCE

    assert "BFSI" not in _OPENING_CHECKPOINT_GUIDANCE
    assert "proven productive lanes" in _OPENING_CHECKPOINT_GUIDANCE
    assert "declared lanes" in _OPENING_CHECKPOINT_GUIDANCE
