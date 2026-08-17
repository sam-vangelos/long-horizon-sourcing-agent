"""P1 items 3-5 (Wave 2 slice 9, plans/sourcing-rigor-hardening.md): the
deterministic layer consumes brief-supplied vocabulary; it never carries its
own vertical vocabulary.

- _augment_novelty_metrics: FDE/frontier-company literals -> brief canonical_*
  pattern mirrors (audit R2-F3).
- _strict_seniority_opening_sort_key: frozen BFS lane-rank map -> brief
  domain_lane_hints order; unknown-but-specific lanes rank ABOVE general,
  never below (audit R2-F2).
- boolean_compiler capability discriminators: vertical terms (fraud, banking,
  capital markets...) move behind brief.key_terms_by_area; only structural
  craft terms stay in code (audit R2-F2 sweep).
"""

from __future__ import annotations

from types import SimpleNamespace

from linkedin.boolean_compiler import boolean_lint_context_from_brief, lint_boolean
from linkedin.strategy import (
    _apply_strict_seniority_plan_guardrails,
    _augment_novelty_metrics,
    _strict_seniority_lane_ranks,
)
from shared.schemas import ExecutionPlan


def _codes(report):
    return {finding.code for finding in report.findings}


def _brief_ns(**overrides) -> SimpleNamespace:
    base = dict(
        canonical_title_patterns=[],
        canonical_framework_patterns=[],
        canonical_company_patterns=[],
        canonical_broad_patterns=[],
        domain_lane_hints=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# P1 item 3 — novelty metrics derive from the brief's canonical mirrors
# ---------------------------------------------------------------------------


def test_novelty_metrics_derive_from_brief_canonical_mirrors():
    brief = _brief_ns(
        canonical_title_patterns=["clinical informatics director"],
        canonical_framework_patterns=["epic emr"],
        canonical_company_patterns=["hca healthcare"],
    )
    plan = ExecutionPlan(strategy_rationale="t")

    _augment_novelty_metrics(brief, plan)

    joined = " ".join(
        plan.architecture_success_criteria + plan.architecture_pivot_triggers
    )
    assert "clinical informatics director" in joined
    assert "epic emr" in joined
    assert "hca healthcare" in joined
    assert "FDE" not in joined
    assert "frontier-company" not in joined


def test_novelty_metrics_fall_back_vertical_agnostic_without_mirrors():
    plan = ExecutionPlan(strategy_rationale="t")

    _augment_novelty_metrics(_brief_ns(), plan)

    assert plan.architecture_success_criteria, "metric must still append"
    assert plan.architecture_pivot_triggers
    joined = " ".join(
        plan.architecture_success_criteria + plan.architecture_pivot_triggers
    )
    for vertical_literal in ("FDE", "frontier-company", "agentic", "framework-name"):
        assert vertical_literal not in joined, vertical_literal


def test_novelty_metrics_idempotent_across_calls():
    brief = _brief_ns(canonical_title_patterns=["ops director"])
    plan = ExecutionPlan(strategy_rationale="t")
    _augment_novelty_metrics(brief, plan)
    once = (
        list(plan.architecture_success_criteria),
        list(plan.architecture_pivot_triggers),
    )
    _augment_novelty_metrics(brief, plan)
    assert (
        list(plan.architecture_success_criteria),
        list(plan.architecture_pivot_triggers),
    ) == once


# ---------------------------------------------------------------------------
# P1 item 4 — strict-seniority lane ranks derive from brief hints
# ---------------------------------------------------------------------------


def _hint(lane: str) -> SimpleNamespace:
    return SimpleNamespace(lane=lane, patterns=[])


def test_lane_ranks_follow_brief_hint_order():
    brief = _brief_ns(domain_lane_hints=[_hint("custody_tech"), _hint("payments_infra")])
    ranks = _strict_seniority_lane_ranks(brief)
    assert ranks["custody_tech"] < ranks["payments_infra"]


def test_unknown_specific_lane_ranks_above_general_with_hints():
    brief = _brief_ns(domain_lane_hints=[_hint("custody_tech")])
    ranks = _strict_seniority_lane_ranks(brief)
    unknown_rank = ranks.get("healthcare_payers", ranks["__unknown_specific__"])
    assert unknown_rank < ranks["general"]


def test_bfs_fallback_map_preserved_without_hints_but_unknown_beats_general():
    """No hints -> the BFS-frozen map (self-consistent: this sort only runs on
    briefs is_strict_seniority_brief matched on BFS text). The FIX: an
    unknown-but-specific lane now ranks ABOVE general instead of dead last."""
    ranks = _strict_seniority_lane_ranks(_brief_ns())
    assert ranks["capital_markets"] < ranks["risk_compliance"] < ranks["general"]
    assert ranks["__unknown_specific__"] < ranks["general"]


def _strict_brief(hints: list[SimpleNamespace]) -> SimpleNamespace:
    """A brief that passes is_strict_seniority_brief (BFS text + 12y + trigger)."""
    return SimpleNamespace(
        role_description=(
            "Executive Director analog leader for financial services applied AI."
        ),
        role_summary="",
        minimum_bar="",
        minimum_bar_description="",
        intake_notes="",
        notes="",
        instructions=[],
        minimum_years_experience=15,
        raw={},
        domain_lane_hints=hints,
        canonical_title_patterns=[],
        canonical_framework_patterns=[],
        canonical_company_patterns=[],
        canonical_broad_patterns=[],
    )


def test_plan_guardrails_rank_hinted_specific_lane_above_general():
    """Through the production path (_apply_strict_seniority_plan_guardrails):
    a string honestly labeled with a brief-declared lane opens ahead of one
    lazily labeled general — the code no longer punishes specific labels."""
    brief = _strict_brief([_hint("custody_tech"), _hint("payments_infra")])
    plan = ExecutionPlan(
        strategy_rationale="t",
        generated_strings=[
            {
                "boolean": '"general opener"',
                "rationale": "general",
                "family_key": "g",
                "domain_lane": "general",
                "novelty_bucket": "canonical",
                "opening_eligible": True,
            },
            {
                "boolean": '"custody opener"',
                "rationale": "specific",
                "family_key": "c",
                "domain_lane": "custody_tech",
                "novelty_bucket": "canonical",
                "opening_eligible": True,
            },
        ],
    )

    _apply_strict_seniority_plan_guardrails(brief, plan)

    assert [item["family_key"] for item in plan.generated_strings] == ["c", "g"]


# ---------------------------------------------------------------------------
# P1 item 5 — vertical capability discriminators come from the brief
# ---------------------------------------------------------------------------


def test_vertical_discriminator_no_longer_builtin():
    """"fraud" was a builtin discriminator for every vertical; bare generic-AI
    vocabulary next to it must now warn unless the BRIEF declares the term."""
    report = lint_boolean('"LLM" AND "fraud analytics"')
    assert "bare_generic_ai_term" in _codes(report)


def test_brief_key_terms_supply_vertical_discriminators():
    brief = SimpleNamespace(
        key_terms_by_area={"risk": ["fraud analytics"]},
        abbreviation_collisions=[],
        role_description="",
        role_summary="",
        minimum_bar="",
        minimum_bar_description="",
        intake_notes="",
        notes="",
        instructions=[],
        minimum_years_experience=0,
        raw={},
    )
    context = boolean_lint_context_from_brief(brief)
    report = lint_boolean('"LLM" AND "fraud analytics"', context=context)
    assert "bare_generic_ai_term" not in _codes(report)


def test_structural_discriminators_stay_builtin():
    report = lint_boolean('"LLM" AND "production platform"')
    assert "bare_generic_ai_term" not in _codes(report)


# ---------------------------------------------------------------------------
# Codex review, Wave 3 (F3): the ASSEMBLED strategy and adaptation prompts
# carry no vertical example vocabulary — prompt literals act as few-shot
# anchors, so the kernel-spec worked examples and tapped-market guidance use
# structural placeholders, never a vertical's titles/employers/frameworks.
# ---------------------------------------------------------------------------

_BANNED_PROMPT_VOCAB = (
    "Forward Deployed",
    "FDE",
    "Applied AI",
    "Palantir",
    "Scale AI",
    "frontier",
    "fintech",
    "Machine Learning Engineer",
    "Nubank",
    "copilot",
    "internal AI platform",
)


def _compat_brief(**overrides):
    from shared.brief_loader import Brief

    brief = Brief(
        id="devertical-prompt-test",
        role_title="Supply Chain Network Design Lead",
        role_description=(
            "The obvious pool is tapped — open with edge-case populations."
        ),
        kit_url="",
        linkedin_project="proj-devertical",
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


def test_strategy_system_prompt_carries_no_vertical_example_vocabulary():
    from linkedin.strategy import _build_strategy_system

    brief = _compat_brief()
    system = _build_strategy_system(brief, has_kit=False, use_layered_retrieval=True)

    for banned in _BANNED_PROMPT_VOCAB:
        assert banned not in system, banned
    # The kernel-spec doctrine itself is untouched (keyword↔filter memory:
    # Boolean-first is a non-negotiable invariant — only example VALUES moved).
    assert "the default surface for a token is the Boolean keyword" in system
    assert "NEVER set a location facet here" in system
    assert "canonical employer pool" in system


def test_adaptation_prompt_carries_no_vertical_example_vocabulary():
    from unittest.mock import patch

    from shared.schemas import BlockReport
    from linkedin.strategy import adapt_after_block

    brief = _compat_brief()
    captured: dict = {}

    def _spy(system, user, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return {
            "new_strings": [],
            "skip_remaining": [],
            "reorder": [],
            "noise_updates": [],
        }

    with patch("linkedin.strategy.opus_llm", side_effect=_spy):
        adapt_after_block(
            brief,
            BlockReport(block_name="Block 1", strings_run=1),
            [],
        )

    joined = captured["system"] + captured["user"]
    for banned in _BANNED_PROMPT_VOCAB:
        assert banned not in joined, banned
    # The tapped-market novelty axis survives, structurally phrased.
    assert "canonical-employer pools" in joined
