"""Characterization tests locking the de-verticalization of the judgment
prompt blocks in ``shared/brief_schema.py`` (P8.3,
plans/sourcing-rigor-hardening.md).

A forensic audit found the judgment-prompt blocks
(``seniority_calibration_block``, ``executive_builder_block``,
``decision_matrix_block``) hardcoded ML/AI vocabulary that gets injected
into EVERY brief regardless of vertical — contradicting the
no-hardcoded-vocab pin ``linkedin/judgment_templates.py`` maintains one
layer up (its calibration-vocabulary fallback helpers deliberately never
mention ML/LLM). The same audit found ``is_senior_role`` treating
PRINCIPAL/DISTINGUISHED (IC ranks) as executive-track — handing
Principal-Engineer briefs the executive evidence apparatus — and
``post_evaluation_safety_net`` acting as a code-injected
REJECT->INFERENTIAL_SAVE override: an uncertainty-favors-save hedge that
violates the settled high-bar doctrine ("uncertain candidate -> DROP").

These tests construct ``Brief`` instances directly (no loader — that
wiring is a separate slice) and assert on the assembled LinkedIn
full-evaluation / facial-triage system prompts.
"""

from __future__ import annotations

import dataclasses
import re

from shared.brief_schema import (
    Brief,
    CapabilityArea,
    DepthDistinction,
    EmployerSignalRule,
    FacialCalibration,
    NonFitPattern,
)
from linkedin.judgment_templates import (
    _KNOWN_FIELDS,
    assemble_facial_system,
    assemble_full_evaluation_system,
    assemble_full_evaluation_tool_system,
)


# ---------------------------------------------------------------------------
# Brief construction
# ---------------------------------------------------------------------------


def _build_brief(**overrides) -> Brief:
    """Build a Brief with every required field defaulted to generic,
    vertical-agnostic content. Callers override only the fields they care
    about, so each fixture below stays readable and scoped to its own
    intent.
    """
    defaults: dict = dict(
        role_title="Generic Role",
        role_level="L5",
        role_summary="Generic role used for characterization testing.",
        geography="Remote",
        linkedin_project="Characterization Test Project",
        capability_areas=[
            CapabilityArea(
                name="Generic Capability",
                description="Owns end-to-end delivery of generic capability work.",
                builder_signals=["built the generic capability system"],
                user_signals=["used the generic capability system"],
                key_terms=["generic capability term"],
            )
        ],
        depth_distinction=DepthDistinction(
            builder_definition="Designed and shipped the underlying system.",
            user_definition="Operates a system someone else designed.",
            edge_case_guidance="Default to full evaluation when the signal is ambiguous.",
        ),
        non_fit_patterns=[
            NonFitPattern(
                label="Adjacent execution work",
                description="Executes day-to-day operations without owning system design.",
                why_not="Operational execution, not capability-area ownership.",
                examples=["operations coordinator"],
            )
        ],
        employer_signal_rules=[
            EmployerSignalRule(
                tier="general",
                employer_patterns=["Any employer"],
                evidence_required="Bullets describing owned, hands-on delivery.",
                save_on_employer_alone=False,
            )
        ],
        minimum_years_experience=5,
        minimum_bar_description="Five years of hands-on ownership in the capability areas.",
        facial_calibration=FacialCalibration(
            expected_yes_rate_low=0.25,
            expected_yes_rate_high=0.55,
            fast_exit_patterns=["Entire career in an unrelated function with no adjacent work"],
            trajectory_yes_patterns=["Progression through roles owning the capability areas"],
            trajectory_ambiguous_patterns=["Generalist trajectory with only partial signal"],
            trajectory_no_patterns=["Entire career in a clearly unrelated function"],
        ),
    )
    defaults.update(overrides)
    return Brief(**defaults)


def _non_ml_brief() -> Brief:
    """Director-level, non-ML vertical (supply chain). ``Director`` trips
    ``is_senior_role``, so this exercises the L7+ seniority / executive-
    builder / decision-matrix blocks. Nothing in this fixture's own
    content is ML/AI vocabulary — any such vocabulary found in the
    assembled prompt would have to come from code, not the brief, which is
    exactly the bug this slice fixes.
    """
    return _build_brief(
        role_title="Director of Supply Chain Operations",
        role_level="Director",
        role_summary=(
            "Owns end-to-end supply chain network design and supplier "
            "strategy for a multi-region logistics operation."
        ),
        geography="Chicago",
        linkedin_project="Supply Chain Director Search",
        capability_areas=[
            CapabilityArea(
                name="Network optimization",
                description=(
                    "Designs and re-configures distribution network topology "
                    "to reduce cost and transit time."
                ),
                builder_signals=[
                    "redesigned the distribution network footprint",
                    "built the load-balancing model across regional warehouses",
                ],
                user_signals=["ran reports from an existing network-planning tool"],
                key_terms=["network flow modeling", "linear programming for logistics"],
            ),
            CapabilityArea(
                name="Supplier management",
                description="Builds supplier qualification, negotiation, and risk-management systems.",
                builder_signals=[
                    "built the supplier scorecard and qualification process",
                    "negotiated and structured multi-year supplier contracts",
                ],
                user_signals=["processed purchase orders against existing supplier contracts"],
                key_terms=["supplier risk scoring", "vendor consolidation strategy"],
            ),
        ],
        depth_distinction=DepthDistinction(
            builder_definition="Designed the network topology or supplier program from scratch, owning the tradeoffs.",
            user_definition="Executes against a network or supplier program someone else designed.",
            edge_case_guidance="A regional ops manager who tunes an existing model without owning the design is USER, not BUILDER.",
        ),
        non_fit_patterns=[
            NonFitPattern(
                label="Warehouse operations management",
                description="Runs day-to-day warehouse labor and shift scheduling without network design ownership.",
                why_not="Operational execution inside a network someone else designed, not network or supplier ownership.",
                examples=["warehouse operations manager", "distribution center shift lead"],
            )
        ],
        employer_signal_rules=[
            EmployerSignalRule(
                tier="general_logistics",
                employer_patterns=["Large retailers", "3PL providers", "Global manufacturers"],
                evidence_required="Bullets describing owned network or supplier program design, not just execution.",
                save_on_employer_alone=False,
            )
        ],
        minimum_years_experience=10,
        minimum_bar_description="Ten years of hands-on network design or supplier program ownership.",
        facial_calibration=FacialCalibration(
            expected_yes_rate_low=0.20,
            expected_yes_rate_high=0.45,
            fast_exit_patterns=["Entire career in retail store operations with no network or supplier scope"],
            trajectory_yes_patterns=["Progression from supply chain analyst into network design or supplier strategy ownership"],
            trajectory_ambiguous_patterns=["Generalist operations trajectory with unclear design ownership"],
            trajectory_no_patterns=["Entire career in unrelated finance or sales operations"],
        ),
    )


def _principal_ic_brief() -> Brief:
    """Principal Engineer, minimal content. Exercises the IC-rank
    reclassification: PRINCIPAL used to trip ``is_senior_role`` (handing an
    IC brief the executive evidence apparatus); it no longer does.
    """
    return _build_brief(role_title="Principal Engineer", role_level="Principal")


NON_ML_BRIEF = _non_ml_brief()
PRINCIPAL_IC_BRIEF = _principal_ic_brief()


# ---------------------------------------------------------------------------
# 1. No hardcoded ML/AI vocabulary leaks into a non-ML brief's full-eval prompt
# ---------------------------------------------------------------------------

FORBIDDEN_ML_TERMS = (
    "ML",
    "LLM",
    "GenAI",
    "LangGraph",
    "Pinecone",
    "Franklin Templeton",
    "Head of AI",
    "frontier lab",
    "Agentic AI",
)


def test_non_ml_brief_full_eval_prompt_has_no_hardcoded_ml_vocabulary():
    prompt = assemble_full_evaluation_system(NON_ML_BRIEF)
    for term in FORBIDDEN_ML_TERMS:
        # Word-boundary match: "ML" must not match inside "html", but must
        # still catch a standalone "ML" token.
        pattern = rf"\b{re.escape(term)}\b"
        assert not re.search(pattern, prompt, re.IGNORECASE), (
            f"non-ML brief's full-evaluation prompt leaked hardcoded ML/AI "
            f"vocabulary: {term!r}"
        )


# ---------------------------------------------------------------------------
# 2. No uncertainty-favors-save / default-YES hedge language, either brief
# ---------------------------------------------------------------------------

FORBIDDEN_HEDGE_SUBSTRINGS = (
    "default YES",
    "MUST default",
    "when in doubt, a pattern is ambiguous",
    # The exact disposition text of the removed code-injected L7+ safety net —
    # these make the test discriminate against a revert of that block, not
    # just against hypothetical future hedges (test-honesty lens, Wave 1).
    "Override to INFERENTIAL_SAVE",
    "may produce false negatives on executive profiles",
)


def test_no_default_yes_hedge_language_in_either_prompt():
    for brief in (NON_ML_BRIEF, PRINCIPAL_IC_BRIEF):
        full_prompt = assemble_full_evaluation_system(brief)
        facial_prompt = assemble_facial_system(brief)
        for forbidden in FORBIDDEN_HEDGE_SUBSTRINGS:
            assert forbidden.lower() not in full_prompt.lower(), (
                f"full-evaluation prompt for {brief.role_title!r} contains "
                f"uncertainty-favors-save hedge language: {forbidden!r}"
            )
            assert forbidden.lower() not in facial_prompt.lower(), (
                f"facial prompt for {brief.role_title!r} contains "
                f"uncertainty-favors-save hedge language: {forbidden!r}"
            )


# ---------------------------------------------------------------------------
# 3. post_evaluation_safety_net -> post_evaluation_overrides: brief opt-in only
# ---------------------------------------------------------------------------


def test_post_evaluation_block_absent_by_default_present_on_opt_in():
    prompt = assemble_full_evaluation_system(NON_ML_BRIEF)
    assert "POST-EVALUATION SAFETY NET" not in prompt
    assert "POST-EVALUATION OVERRIDES" not in prompt

    opted_in = dataclasses.replace(
        NON_ML_BRIEF,
        post_evaluation_overrides=(
            "If REJECT for an L8+ candidate with board-level scope, flag "
            "for recruiter review."
        ),
    )
    opted_in_prompt = assemble_full_evaluation_system(opted_in)
    assert "POST-EVALUATION OVERRIDES" in opted_in_prompt
    assert "If REJECT for an L8+ candidate with board-level scope" in opted_in_prompt


# ---------------------------------------------------------------------------
# 4. is_senior_role no longer treats PRINCIPAL as executive-track
# ---------------------------------------------------------------------------


def test_principal_ic_brief_is_not_senior_role():
    assert PRINCIPAL_IC_BRIEF.is_senior_role() is False


def test_principal_ic_brief_gets_trajectory_shape_not_executive_apparatus():
    prompt = assemble_full_evaluation_system(PRINCIPAL_IC_BRIEF)
    assert "TRAJECTORY-SHAPE INFERENCE" in prompt
    assert "SENIORITY CALIBRATION" not in prompt
    assert "EXECUTIVE BUILDER CALIBRATION" not in prompt


# ---------------------------------------------------------------------------
# 5. Control: real executives (L8) still get the executive apparatus
# ---------------------------------------------------------------------------


def test_l8_brief_still_gets_seniority_calibration():
    brief = _build_brief(role_title="Generic Executive Role", role_level="L8")
    assert brief.is_senior_role() is True
    prompt = assemble_full_evaluation_system(brief)
    assert "SENIORITY CALIBRATION" in prompt


# ---------------------------------------------------------------------------
# 6. P3c (Wave 2): employer blacklist is VISIBLE to the full-eval judge
# ---------------------------------------------------------------------------
# Audit R3-F3/R5: the blacklist's only enforcement was the snippet-stage
# current_company substring gate — the full-eval prompt never received it, so
# past-stage enforcement did not exist when the snippet's company was empty.
# Scope stays current-company (DECISION #4, committed): the block gives the
# judge visibility, including that alumni are eligible — not an alumni gate.


def test_employer_blacklist_renders_in_full_eval_employer_block():
    with_blacklist = dataclasses.replace(
        NON_ML_BRIEF, employer_blacklist=["Acme Logistics", "Acme Freight"]
    )
    prompt = assemble_full_evaluation_system(with_blacklist)
    assert "EMPLOYER BLACKLIST" in prompt
    assert "Acme Logistics" in prompt
    assert "Acme Freight" in prompt


def test_employer_blacklist_block_absent_when_blacklist_empty():
    prompt = assemble_full_evaluation_system(NON_ML_BRIEF)
    assert "EMPLOYER BLACKLIST" not in prompt


# ---------------------------------------------------------------------------
# Outreach-judgment remediation: floor-only briefs retain the legacy minimum
# line. A ceiling is leveling context by default and becomes a reject gate only
# when the operator explicitly marks it hard. The full rubric judges whether a
# strong recruiter conversation is warranted, not whether LinkedIn alone proves
# the candidate should be hired.
# ---------------------------------------------------------------------------


def test_floor_only_brief_renders_legacy_minimum_bar_line():
    from linkedin.judgment_templates import assemble_full_evaluation_system

    system = assemble_full_evaluation_system(_build_brief())

    assert (
        "MINIMUM BAR: 5+ years hands-on. Five years of hands-on ownership "
        "in the capability areas." in system
    )
    assert "EXPERIENCE BAND" not in system


def test_band_brief_renders_advisory_ceiling_by_default():
    from linkedin.judgment_templates import assemble_full_evaluation_system

    system = assemble_full_evaluation_system(
        _build_brief(maximum_years_experience=10)
    )

    assert "EXPERIENCE BAND: 5-10 years, soft margin ±2 years" in system
    # No experience_measure on the brief → the template forces the judge to
    # state which tenure its verdict used rather than picking silently.
    assert "STATE which measure" in system
    assert "LEVELING:" in system
    assert "advisory" in system.lower()
    assert "not an automatic reject" in system.lower()
    assert "over-band strength is not transferable downward" not in system
    assert "beyond the soft margin on either side is a REJECT" not in system
    assert "MINIMUM BAR:" not in system


def test_operator_declared_hard_ceiling_retains_reject_gate():
    from linkedin.judgment_templates import assemble_full_evaluation_system

    system = assemble_full_evaluation_system(
        _build_brief(
            maximum_years_experience=10,
            maximum_years_experience_is_hard=True,
            experience_measure="total professional experience",
        )
    )

    assert "EXPERIENCE BAND: 5-10 years, soft margin ±2 years" in system
    assert "hard" in system.lower()
    assert "over-band strength is not transferable downward" in system
    assert "beyond the soft margin on either side is a REJECT" in system
    assert "ADVISORY LEVELING" not in system


def test_band_brief_renders_brief_authored_measure():
    from linkedin.judgment_templates import assemble_full_evaluation_system

    system = assemble_full_evaluation_system(
        _build_brief(
            maximum_years_experience=10,
            experience_measure="total career years since first full-time role",
        )
    )

    assert (
        "Years are measured as: total career years since first full-time role"
        in system
    )
    assert "STATE which measure" not in system


def test_decision_matrix_uses_outreach_standard_and_unknown_depth():
    system = assemble_full_evaluation_system(_build_brief())

    assert "DECISION STANDARD:" in system
    assert "Eligibility is not the decision" in system
    assert "SAVE requires ALL of:" in system
    assert "1. No hard gate failed" in system
    assert "2. A capability case meeting the brief's stated evidence bar" in system
    assert "3. Depth BUILDER, or UNKNOWN with the rest of the case strong" in system
    assert "4. Level ALIGNED (Step 4)" in system
    assert "5. Opportunity coherence COHERENT" in system
    assert "6. Caliber SOLID or STRONG" in system
    assert "BAR_ORDINARY" in system
    assert "STEP_2_DEPTH: BUILDER or USER or UNKNOWN" in system
    assert "missing evidence" in system.lower()
    assert "UNKNOWN, not USER" in system
    assert "Sparse profiles default to REJECT" not in system
    assert "If no inferential save applies AND the profile is sparse, respond REJECT" not in system
    assert "sparsity alone is not an automatic reject" in system.lower()
    assert "continue through Steps 1-4" in system
    assert "THE HEDGE TEST" not in system
    assert "An uncertain save poisons trust in every save" not in system
    assert "must demonstrate hands-on builder depth" not in system
    assert "No depth = reject" not in system


def test_decision_matrix_renders_brief_authored_transfer_bar():
    bar = (
        "demonstrable technical fundamentals (built models or systems, owned "
        "quantitative work) plus operator-grade delivery ownership"
    )
    system = assemble_full_evaluation_system(
        _build_brief(transferable_fundamentals_bar=bar)
    )

    assert bar in system
    assert "flag it TRANSFERABLE_SAVE" in system
    assert "positive evidence connecting to at least one capability area's domain" not in system


def test_full_evaluation_identity_stakes_a_shortlist_claim():
    system = assemble_full_evaluation_system(_build_brief())

    assert "among the strongest plausible matches this market can produce" in system


def test_engagement_context_renders_authored_values():
    brief = _build_brief(market_density="sparse")
    brief.engagement_context = {
        "hiring_company": "Example Organization",
        "engagement_description": "A confidential capability search.",
        "talent_bar_statement": "Only unusually strong ownership cases clear the bar.",
        "selectivity_posture": "selective",
    }

    system = assemble_full_evaluation_system(brief)

    assert "ENGAGEMENT CONTEXT:" in system
    assert "Hiring organization: Example Organization. A confidential capability search." in system
    assert "Talent bar: Only unusually strong ownership cases clear the bar." in system
    assert "Selectivity posture: selective" in system
    assert "Selectivity posture: coverage" not in system
    assert "this market is dense" not in system.lower()


def test_engagement_context_derives_posture_from_market_density():
    cases = (
        ("dense", "selective"),
        ("moderate", "selective"),
        (None, "selective"),
        ("unknown", "selective"),
        ("sparse", "coverage"),
    )

    for market_density, expected_posture in cases:
        system = assemble_full_evaluation_system(
            _build_brief(market_density=market_density)
        )

        assert "ENGAGEMENT CONTEXT:" in system
        assert f"Selectivity posture: {expected_posture}" in system
        if expected_posture == "selective" and market_density != "dense":
            assert "this market is dense" not in system.lower()


def test_full_evaluation_renders_new_stages_bias_guards_and_resolution_rule():
    system = assemble_full_evaluation_system(_build_brief())

    assert "STEP 4 — LEVEL ALIGNMENT" in system
    assert "STEP 5 — OPPORTUNITY COHERENCE" in system
    assert "STEP 6 — CANDIDATE CALIBER" in system
    assert "BIAS GUARDS:" in system
    assert "RESOLUTION RULE:" in system


def test_level_envelope_fallback_renders_brief_role_level():
    system = assemble_full_evaluation_system(
        _build_brief(role_level="L6")
    )

    assert "ROLE LEVEL: L6." in system


def test_full_evaluation_response_format_contains_new_fields():
    system = assemble_full_evaluation_system(_build_brief())

    for field_line in (
        "STEP_1_RECENCY: CURRENT or RECENT or STALE",
        "STEP_4_LEVEL: ALIGNED or ABOVE or BELOW or UNCLEAR",
        "STEP_5_COHERENCE: COHERENT or INCOHERENT or UNCLEAR",
        "STEP_6_CALIBER: STRONG or SOLID or WEAK or UNKNOWN",
        "REJECT_REASON: [exactly one of HARD_GATE",
        "OUTREACH_TIER: [PRIORITY or STANDARD",
    ):
        assert field_line in system


def test_full_evaluation_tool_tail_uses_structured_allocator_fields():
    system = assemble_full_evaluation_tool_system(_build_brief())

    assert "dedicated structured fields" in system
    assert "PRIORITY requires DIRECT + CURRENT + STRONG" in system
    assert "TRANSITIONAL TAGS" not in system


def test_full_evaluation_known_fields_include_phase_one_labels():
    assert {
        "STEP_1_RECENCY",
        "STEP_4_LEVEL",
        "STEP_4_EVIDENCE",
        "STEP_5_COHERENCE",
        "STEP_5_DRIVER",
        "STEP_6_CALIBER",
        "STEP_6_EVIDENCE",
        "REJECT_REASON",
        "OUTREACH_TIER",
    } <= _KNOWN_FIELDS
