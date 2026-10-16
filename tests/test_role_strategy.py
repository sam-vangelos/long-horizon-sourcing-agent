"""Tests for shared/role_strategy.py (P8 role-class defaults)."""

import json
from pathlib import Path

import pytest

from shared.brief_loader import Brief, load_brief
from shared.role_strategy import (
    apply_role_strategy_to_plan,
    infer_role_strategy_profile_id,
    resolve_role_strategy_profile,
)
from shared.schemas import ExecutionPlan


def _minimal_v2_payload(**overrides) -> dict:
    payload = {
        "role_title": "Software Engineer",
        "role_level": "IC3",
        "role_summary": "Builds backend systems.",
        "geography": "United States",
        "minimum_years_experience": 4,
        "minimum_bar_description": "Strong backend builder.",
        "linkedin_project": "test-project",
        "capability_areas": [
            {
                "name": "Backend systems",
                "description": "Owns service architecture.",
                "builder_signals": ["distributed systems"],
                "user_signals": ["ticket routing only"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns architecture.",
            "user_definition": "Uses existing services.",
            "edge_case_guidance": "Borderline = full eval.",
        },
    }
    payload.update(overrides)
    return payload


def _write_brief(tmp_path: Path, payload: dict) -> Brief:
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_brief(path)


def _bfs_head_ai_payload() -> dict:
    return _minimal_v2_payload(
        role_title="Head of Applied AI Lab",
        role_level="VP",
        role_summary=(
            "Senior BFS applied AI leader for banking and financial services. "
            "Executive-builder with lab-head scope and executive director analog scope."
        ),
        minimum_years_experience=15,
        capability_areas=[
            {
                "name": "Applied AI leadership",
                "description": "Owns applied AI lab vision and production agent systems.",
                "builder_signals": ["agentic systems", "model evaluation", "governance"],
                "user_signals": ["strategy-only AI advisor"],
            },
            {
                "name": "BFS domain delivery",
                "description": "Ships AI in regulated banking contexts.",
                "builder_signals": ["production deployment", "risk", "compliance"],
                "user_signals": ["generic innovation lab"],
            },
        ],
        employer_signal_rules=[
            {
                "tier": "A",
                "employer_patterns": ["goldman sachs", "jpmorgan", "morgan stanley"],
                "evidence_required": "Bank-native AI leadership scope.",
            }
        ],
    )


def _fde_payload() -> dict:
    return _minimal_v2_payload(
        role_title="Forward Deployed Engineer",
        role_level="IC4",
        role_summary=(
            "Enterprise GenAI forward deployed engineer building customer-facing "
            "production deployments and workflow orchestration."
        ),
        minimum_years_experience=5,
        capability_areas=[
            {
                "name": "Customer deployment",
                "description": "Ships enterprise GenAI into customer environments.",
                "builder_signals": ["production deployment", "customer onboarding"],
                "user_signals": ["sales engineer only"],
            }
        ],
    )


def test_head_of_applied_ai_bfs_brief_infers_senior_bfs_ai_leader(tmp_path: Path) -> None:
    brief = _write_brief(tmp_path, _bfs_head_ai_payload())
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "senior_bfs_ai_leader"
    assert metadata["profile_source"] == "inferred"
    assert "bfs_domain" in metadata["matched_signals"] or "strict_seniority" in metadata["matched_signals"]


def test_fde_brief_infers_fde_enterprise_genai(tmp_path: Path) -> None:
    brief = _write_brief(tmp_path, _fde_payload())
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "fde_enterprise_genai"
    assert metadata["profile_source"] == "inferred"
    assert "fde_title_patterns" in metadata["matched_signals"]


def test_explicit_profile_overrides_inference(tmp_path: Path) -> None:
    payload = _bfs_head_ai_payload()
    payload["role_strategy_profile"] = "ic_frontier_engineer"
    brief = _write_brief(tmp_path, payload)
    profile, metadata = resolve_role_strategy_profile(brief)
    assert profile.profile_id == "ic_frontier_engineer"
    assert metadata["profile_source"] == "explicit"


def test_legacy_brief_falls_back_safely() -> None:
    brief = Brief(
        id="legacy",
        role_title="Platform Engineer",
        role_description="Maintains internal tooling.",
        kit_url="",
        linkedin_project="legacy",
        linkedin_project_id="",
        minimum_bar="",
        archetypes=[],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
    )
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "generic"
    assert metadata["profile_source"] == "generic_fallback"


def test_profile_defaults_create_different_lane_portfolios() -> None:
    bfs_profile, _ = resolve_role_strategy_profile(
        Brief(
            id="bfs",
            role_title="Head of Applied AI Lab",
            role_description="Banking and financial services applied AI leader with lab-head scope.",
            kit_url="",
            linkedin_project="bfs",
            linkedin_project_id="",
            minimum_bar="",
            archetypes=[],
            noise_archetypes=[],
            hard_skips=[],
            clear_skips_from_review=[],
            known_noise_patterns=[],
            permanent_filters={},
            save_instructions={},
            experience_floor={"minimum_years": 15},
            raw={
                "minimum_years_experience": 15,
                "role_level": "VP",
            },
        )
    )
    fde_profile, _ = resolve_role_strategy_profile(
        Brief(
            id="fde",
            role_title="Forward Deployed Engineer",
            role_description="Enterprise GenAI customer deployment engineer.",
            kit_url="",
            linkedin_project="fde",
            linkedin_project_id="",
            minimum_bar="",
            archetypes=[],
            noise_archetypes=[],
            hard_skips=[],
            clear_skips_from_review=[],
            known_noise_patterns=[],
            permanent_filters={},
            save_instructions={},
            experience_floor={},
        )
    )
    bfs_lane_ids = {lane.lane_id for lane in bfs_profile.lane_templates}
    fde_lane_ids = {lane.lane_id for lane in fde_profile.lane_templates}
    assert bfs_lane_ids != fde_lane_ids
    assert "bfs_senior_obvious_pool" in bfs_lane_ids
    assert "fde_capability_led" in fde_lane_ids


def test_apply_role_strategy_to_plan_is_additive_and_preserves_generated_strings() -> None:
    plan = ExecutionPlan(
        strategy_rationale="test",
        generated_strings=[
            {
                "boolean": '("agentic" OR "LLM") AND ("banking")',
                "rationale": "primary executable string",
            }
        ],
    )
    brief = Brief(
        id="fde",
        role_title="Forward Deployed Engineer",
        role_description="Enterprise GenAI deployment engineer.",
        kit_url="",
        linkedin_project="fde",
        linkedin_project_id="",
        minimum_bar="",
        archetypes=[],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
    )
    original_strings = list(plan.generated_strings)
    apply_role_strategy_to_plan(brief, plan)
    assert plan.generated_strings == original_strings
    assert plan.role_strategy_profile == "fde_enterprise_genai"
    assert plan.sourcing_lanes
    assert plan.search_hypotheses
    assert plan.search_slices
    assert plan.role_strategy_metadata["profile_source"] == "inferred"


def test_linkedin_apply_role_strategy_hints(tmp_path: Path) -> None:
    from linkedin.strategy import _apply_role_strategy_hints

    brief = _write_brief(tmp_path, _bfs_head_ai_payload())
    plan = ExecutionPlan(
        strategy_rationale="mock",
        generated_strings=[{"boolean": '("GenAI") AND ("banking")', "rationale": "exec"}],
    )
    _apply_role_strategy_hints(brief, plan)
    assert plan.role_strategy_profile == "senior_bfs_ai_leader"
    assert len(plan.generated_strings) == 1
    assert any(lane["lane_id"] == "bfs_senior_obvious_pool" for lane in plan.sourcing_lanes)


def test_apply_role_strategy_does_not_duplicate_existing_lane_ids() -> None:
    plan = ExecutionPlan(
        strategy_rationale="test",
        sourcing_lanes=[{"lane_id": "fde_capability_led", "lane_name": "Existing"}],
        search_hypotheses=[{"hypothesis_id": "fde_capability_led", "label": "Existing"}],
        search_slices=[{"slice_id": "fde_capability_led_slice", "hypothesis_id": "fde_capability_led"}],
    )
    brief = Brief(
        id="fde",
        role_title="Forward Deployed Engineer",
        role_description="Enterprise GenAI deployment engineer.",
        kit_url="",
        linkedin_project="fde",
        linkedin_project_id="",
        minimum_bar="",
        archetypes=[],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
    )
    apply_role_strategy_to_plan(brief, plan)
    lane_ids = [lane["lane_id"] for lane in plan.sourcing_lanes]
    assert lane_ids.count("fde_capability_led") == 1
    assert "fde_customer_deployment_proof" in lane_ids


def test_apply_role_strategy_without_lane_merge_attaches_metadata_only() -> None:
    plan = ExecutionPlan(
        strategy_rationale="test",
        generated_strings=[{"boolean": '("agentic") AND ("production")', "rationale": "exec"}],
    )
    brief = Brief(
        id="fde",
        role_title="Forward Deployed Engineer",
        role_description="Enterprise GenAI deployment engineer.",
        kit_url="",
        linkedin_project="fde",
        linkedin_project_id="",
        minimum_bar="",
        archetypes=[],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
    )
    apply_role_strategy_to_plan(brief, plan, merge_lane_templates=False)
    assert plan.role_strategy_profile == "fde_enterprise_genai"
    assert plan.sourcing_lanes == []
    assert plan.search_hypotheses == []
    assert plan.search_slices == []
