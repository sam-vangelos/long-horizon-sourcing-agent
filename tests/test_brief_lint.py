"""Tests for shared/brief_lint.py — the generated-brief injection scanner (P4).

Table-driven: a lint-clean brief passes; each seeded defect yields its named
code at the expected severity. The lint is pure (dict in, findings out), so
these tests double as the executable spec for what "goes live" means.
"""

from __future__ import annotations

from shared.brief_lint import (
    TEMPLATE_DEFAULT_BAND,
    blocking_findings,
    format_findings,
    lint_generated_brief,
)


def _clean_brief() -> dict:
    return {
        "role_title": "Director of Supply Chain Operations",
        "role_summary": "Owns network design and S&OP.",
        "hiring_company": "Acme Logistics",
        "employer_blacklist": ["Acme Logistics"],
        "engagement_context": {
            "hiring_company": "Acme Logistics",
            "engagement_description": "A director search for a network-design leader.",
            "talent_bar_statement": "End-to-end network ownership clears the bar.",
            "selectivity_posture": "selective",
        },
        "capability_areas": [
            {
                "name": "Network optimization",
                "description": "Designs multi-node distribution networks.",
                "builder_signals": ["led a network redesign"],
                "user_signals": ["operated inside an existing plan"],
                "key_terms": ["network design"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Designs the network.",
            "user_definition": "Runs sites within it.",
            "edge_case_guidance": "Ownership of the redesign tips the balance.",
        },
        "non_fit_patterns": [
            {
                "label": "Site operator",
                "description": "Runs a single DC.",
                "why_not": "Never owns network-level design.",
                "examples": ["DC shift manager"],
            }
        ],
        "employer_signal_rules": [
            {
                "tier": "general_tech",
                "employer_patterns": ["FedEx"],
                "evidence_required": "network-design ownership",
                "save_on_employer_alone": False,
            }
        ],
        "facial_calibration": {
            "expected_yes_rate_low": 0.12,
            "expected_yes_rate_high": 0.28,
            "yes_rate_rationale": "Moderate-density director pool.",
            "fast_exit_patterns": ["entire career in retail merchandising"],
            "trajectory_yes_patterns": ["network design lead at a parcel carrier"],
            "trajectory_ambiguous_patterns": ["Operations Director — scope unclear from snippet"],
            "trajectory_no_patterns": ["entire career in store operations"],
        },
        "domain_lane_hints": [
            {"lane": "parcel_carriers", "patterns": ["fedex", "ups"]},
        ],
        # RC3 (2026-07-04): a clean brief carries the opening mirrors, same
        # as it carries lane hints — their absence is a (deliberate) warning.
        "canonical_title_patterns": ["director of supply chain"],
        "canonical_company_patterns": ["fedex"],
        "edge_case_patterns": ["military logistics officer transitioning to industry"],
        # 2026-07-04: a clean brief carries two lint-clean worked compounds
        # (one precision anchor, one recall net) drawn from its OWN vocabulary
        # — network design (a key_term), the canonical title, the canonical
        # employer — so each shares a quoted term with the brief and never
        # trips the off-vocabulary warning.
        "example_compounds": [
            {
                "boolean": '("network design")',
                "purpose": "Precision proof-of-practice on the discriminating artifact term",
                "novelty_bucket": "canonical",
            },
            {
                "boolean": '("network design" OR "director of supply chain") AND ("fedex")',
                "purpose": "Multi-angle recall net across role and employer vocabulary",
                "novelty_bucket": "edge_case",
            },
        ],
    }


def _codes(findings):
    return [f.code for f in findings]


def test_clean_brief_has_no_findings():
    assert lint_generated_brief(_clean_brief()) == []


def test_generated_brief_requires_engagement_context():
    data = _clean_brief()
    data.pop("engagement_context")

    assert "engagement_context_missing" in _codes(
        blocking_findings(lint_generated_brief(data))
    )


def test_generated_brief_requires_valid_selectivity_posture():
    data = _clean_brief()
    data["engagement_context"] = {"selectivity_posture": "balanced"}

    assert "engagement_context_invalid" in _codes(
        blocking_findings(lint_generated_brief(data))
    )


def test_generated_brief_allows_optional_engagement_text_to_be_absent():
    data = _clean_brief()
    data["engagement_context"] = {"selectivity_posture": "coverage"}

    assert lint_generated_brief(data) == []


def test_generated_brief_rejects_non_string_optional_engagement_text():
    data = _clean_brief()
    data["engagement_context"]["talent_bar_statement"] = ["not", "text"]

    assert "engagement_context_invalid" in _codes(
        blocking_findings(lint_generated_brief(data))
    )


def test_candidate_register_terms_warn_on_jd_heading_term():
    data = _clean_brief()
    data["capability_areas"][0]["candidate_register_terms"] = [
        "Required Qualifications"
    ]

    findings = lint_generated_brief(
        data,
        jd_text="""
About the team

Required Qualifications

You will design multi-node distribution networks.
""",
    )

    warning = next(f for f in findings if f.code == "jd_register_in_candidate_terms")
    assert warning.severity == "warning"
    assert "jd_register_in_candidate_terms" not in _codes(blocking_findings(findings))


def test_candidate_register_terms_allow_self_description_terms():
    data = _clean_brief()
    data["capability_areas"][0]["candidate_register_terms"] = [
        "distribution network redesign",
        "S&OP planning",
    ]

    findings = lint_generated_brief(
        data,
        jd_text="""
Responsibilities

You will design multi-node distribution networks.
""",
    )

    assert "jd_register_in_candidate_terms" not in _codes(findings)


def test_hedge_language_in_ambiguous_patterns_is_blocking():
    data = _clean_brief()
    data["facial_calibration"]["trajectory_ambiguous_patterns"] = [
        "Relevant title at a strong company. These MUST default to YES."
    ]
    findings = lint_generated_brief(data)
    assert "hedge_language" in _codes(findings)
    assert "hedge_language" in _codes(blocking_findings(findings))


def test_hedge_language_scans_all_four_pattern_arrays():
    for field in (
        "fast_exit_patterns",
        "trajectory_yes_patterns",
        "trajectory_ambiguous_patterns",
        "trajectory_no_patterns",
    ):
        data = _clean_brief()
        data["facial_calibration"][field] = ["when in doubt, pass them through"]
        assert "hedge_language" in _codes(lint_generated_brief(data)), field


def test_hiring_company_in_positive_tier_is_blocking():
    data = _clean_brief()
    data["employer_signal_rules"].append(
        {
            "tier": "strong_ai",
            "employer_patterns": ["Acme Logistics"],
            "evidence_required": "none",
            "save_on_employer_alone": True,
        }
    )
    findings = lint_generated_brief(data)
    assert "employer_conflict" in _codes(blocking_findings(findings))


def test_seed_blacklist_checked_against_tiers_even_when_model_omitted_it():
    data = _clean_brief()
    data["hiring_company"] = ""
    data["employer_blacklist"] = []
    data["employer_signal_rules"][0]["employer_patterns"] = ["Acme"]
    findings = lint_generated_brief(data, seed_blacklist=["Acme"])
    assert "employer_conflict" in _codes(blocking_findings(findings))


def test_hiring_company_missing_from_blacklist_is_blocking():
    data = _clean_brief()
    data["employer_blacklist"] = []
    findings = lint_generated_brief(data)
    assert "hiring_company_not_blacklisted" in _codes(blocking_findings(findings))


def test_template_default_band_without_rationale_is_blocking():
    data = _clean_brief()
    low, high = TEMPLATE_DEFAULT_BAND
    data["facial_calibration"]["expected_yes_rate_low"] = low
    data["facial_calibration"]["expected_yes_rate_high"] = high
    data["facial_calibration"]["yes_rate_rationale"] = ""
    findings = lint_generated_brief(data)
    assert "band_template_default" in _codes(blocking_findings(findings))


def test_template_default_band_with_rationale_passes():
    data = _clean_brief()
    low, high = TEMPLATE_DEFAULT_BAND
    data["facial_calibration"]["expected_yes_rate_low"] = low
    data["facial_calibration"]["expected_yes_rate_high"] = high
    data["facial_calibration"]["yes_rate_rationale"] = (
        "Dense mid-level pool; the historical default genuinely fits here."
    )
    assert "band_template_default" not in _codes(lint_generated_brief(data))


def test_experience_measure_unstated_qualifier_is_warning_not_blocking():
    data = _clean_brief()
    data["maximum_years_experience"] = 10
    data["experience_measure"] = (
        "Years of relevant post-graduation professional experience owning complex, "
        "multi-stakeholder project delivery and/or client-facing operational roles, "
        "in any industry; the band is soft at both edges per operator calibration."
    )

    findings = lint_generated_brief(data)

    warning = next(
        f for f in findings if f.code == "experience_measure_unstated_qualifier"
    )
    assert warning.severity == "warning"
    assert "experience_measure_unstated_qualifier" not in _codes(
        blocking_findings(findings)
    )


def test_total_professional_experience_measure_does_not_warn():
    data = _clean_brief()
    data["maximum_years_experience"] = 10
    data["experience_measure"] = (
        "Total professional experience, including research and advanced-degree years."
    )

    assert "experience_measure_unstated_qualifier" not in _codes(
        lint_generated_brief(data)
    )


def test_advisory_experience_ceiling_is_valid_without_hard_authorization():
    data = _clean_brief()
    data["maximum_years_experience"] = 10
    data["experience_measure"] = "Total professional experience."
    data["maximum_years_experience_is_hard"] = False

    codes = _codes(lint_generated_brief(data))

    assert "experience_ceiling_hardness_invalid" not in codes
    assert "hard_experience_ceiling_without_ceiling" not in codes
    assert "hard_experience_ceiling_without_operator_authorization" not in codes


def test_hard_experience_ceiling_requires_explicit_operator_authorization():
    data = _clean_brief()
    data["maximum_years_experience"] = 10
    data["experience_measure"] = "Total professional experience."
    data["maximum_years_experience_is_hard"] = True

    unauthorised = blocking_findings(lint_generated_brief(data))
    assert "hard_experience_ceiling_without_operator_authorization" in _codes(
        unauthorised
    )

    authorised = lint_generated_brief(
        data,
        operator_instructions=[
            "Ten years is a hard experience ceiling: reject candidates above it."
        ],
    )
    assert "hard_experience_ceiling_without_operator_authorization" not in _codes(
        authorised
    )


def test_hard_experience_ceiling_requires_a_ceiling_value():
    data = _clean_brief()
    data["maximum_years_experience_is_hard"] = True

    findings = blocking_findings(
        lint_generated_brief(
            data,
            operator_instructions=["Treat the experience ceiling as a hard gate."],
        )
    )

    assert "hard_experience_ceiling_without_ceiling" in _codes(findings)


def test_experience_ceiling_hardness_must_be_boolean():
    data = _clean_brief()
    data["maximum_years_experience"] = 10
    data["experience_measure"] = "Total professional experience."
    data["maximum_years_experience_is_hard"] = "yes"

    findings = blocking_findings(lint_generated_brief(data))

    assert "experience_ceiling_hardness_invalid" in _codes(findings)


def test_operator_instruction_suppresses_matching_experience_measure_qualifier():
    data = _clean_brief()
    data["maximum_years_experience"] = 10
    data["experience_measure"] = "Post-graduation professional experience."

    findings = lint_generated_brief(
        data,
        operator_instructions=[
            "Count post-graduation years only for this search."
        ],
    )

    assert "experience_measure_unstated_qualifier" not in _codes(findings)


def test_negated_operator_instruction_does_not_suppress_experience_measure_qualifier():
    data = _clean_brief()
    data["maximum_years_experience"] = 10
    data["experience_measure"] = "Years of post-graduation full-time experience."

    findings = lint_generated_brief(
        data,
        operator_instructions=[
            (
                "Do not restrict the measure to post-graduation, full-time, "
                "consecutive, or same-industry years."
            )
        ],
    )

    assert "experience_measure_unstated_qualifier" in _codes(findings)


def test_jd_text_suppresses_matching_experience_measure_qualifier():
    data = _clean_brief()
    data["maximum_years_experience"] = 10
    data["experience_measure"] = "Consecutive professional experience."

    findings = lint_generated_brief(
        data,
        jd_text="Candidates should have 4-8 years of consecutive professional experience.",
    )

    assert "experience_measure_unstated_qualifier" not in _codes(findings)


def test_unrelated_lint_input_does_not_suppress_experience_measure_qualifier():
    data = _clean_brief()
    data["maximum_years_experience"] = 10
    data["experience_measure"] = "Post-graduation professional experience."
    data["minimum_bar_description"] = "Post-graduation experience is named here."

    findings = lint_generated_brief(data, seed_blacklist=["post-graduation"])

    assert "experience_measure_unstated_qualifier" in _codes(findings)


def test_null_band_is_blocking():
    data = _clean_brief()
    data["facial_calibration"]["expected_yes_rate_low"] = None
    data["facial_calibration"]["expected_yes_rate_high"] = None
    assert "band_invalid" in _codes(blocking_findings(lint_generated_brief(data)))


def test_inverted_band_is_blocking():
    data = _clean_brief()
    data["facial_calibration"]["expected_yes_rate_low"] = 0.6
    data["facial_calibration"]["expected_yes_rate_high"] = 0.2
    assert "band_invalid" in _codes(blocking_findings(lint_generated_brief(data)))


def test_missing_required_keys_are_blocking():
    findings = lint_generated_brief({"role_title": "X"})
    codes = _codes(blocking_findings(findings))
    assert codes.count("schema_missing_key") >= 4


def test_empty_non_fit_patterns_is_blocking():
    data = _clean_brief()
    data["non_fit_patterns"] = []
    assert "schema_missing_key" in _codes(blocking_findings(lint_generated_brief(data)))


def test_missing_lane_hints_is_warning_not_blocking():
    data = _clean_brief()
    data["domain_lane_hints"] = []
    findings = lint_generated_brief(data)
    assert "missing_domain_lane_hints" in _codes(findings)
    assert blocking_findings(findings) == []


def test_format_findings_renders_one_line_per_finding():
    data = _clean_brief()
    data["domain_lane_hints"] = []
    rendered = format_findings(lint_generated_brief(data))
    assert "[preflight-lint] WARNING: missing_domain_lane_hints" in rendered


def test_unnamed_capability_areas_are_blocking():
    """A non-empty capability_areas list whose entries carry no names is a
    hollow-brief injection the lint advertises catching (empty_capability_areas)."""
    data = _clean_brief()
    data["capability_areas"] = [{"description": "no name here"}]
    findings = lint_generated_brief(data)
    assert "empty_capability_areas" in _codes(blocking_findings(findings))


def test_non_object_response_is_blocking():
    findings = lint_generated_brief([])
    assert _codes(findings) == ["schema_not_object"]


def test_save_hedge_in_edge_case_guidance_is_blocking():
    """depth_distinction.edge_case_guidance renders into the live judge via
    depth_block() — a save-directional disposition there is the audit's exact
    violation class and must block (correctness lens, Wave 1)."""
    data = _clean_brief()
    data["depth_distinction"]["edge_case_guidance"] = (
        "when in doubt, default to YES and let full evaluation decide"
    )
    findings = lint_generated_brief(data)
    assert "hedge_language" in _codes(blocking_findings(findings))


def test_reject_directional_prose_guidance_is_not_blocked():
    """'when in doubt, reject' is doctrine-ALIGNED prose — the prose scan is
    save-directional only and must not false-block it."""
    data = _clean_brief()
    data["depth_distinction"]["edge_case_guidance"] = (
        "When in doubt, reject — a maybe poisons every yes."
    )
    assert lint_generated_brief(data) == []


def test_save_hedge_paraphrase_in_minimum_bar_is_blocking():
    data = _clean_brief()
    data["minimum_bar_description"] = (
        "8+ years; for borderline profiles, err on the side of inclusion."
    )
    findings = lint_generated_brief(data)
    assert "hedge_language" in _codes(blocking_findings(findings))


def test_employer_conflict_requires_token_boundary_not_substring():
    """hiring_company 'Ramp' must not spuriously block the unrelated tier
    employer 'Rampart AI' (correctness lens false-positive)."""
    data = _clean_brief()
    data["hiring_company"] = "Ramp"
    data["employer_blacklist"] = ["Ramp"]
    data["employer_signal_rules"][0]["employer_patterns"] = ["Rampart AI"]
    assert lint_generated_brief(data) == []


def test_employer_conflict_still_fires_on_token_match_inside_longer_pattern():
    data = _clean_brief()
    data["hiring_company"] = "Acme"
    data["employer_blacklist"] = ["Acme"]
    data["employer_signal_rules"][0]["employer_patterns"] = ["Acme Logistics Group"]
    findings = lint_generated_brief(data)
    assert "employer_conflict" in _codes(blocking_findings(findings))


def test_geography_string_and_structured_shapes_pass():
    for geography in (
        "Colombia",
        {"facet_candidates": ["Colombia"], "rationale": "JD names Colombia"},
        {"facet_candidates": [], "rationale": ""},
    ):
        data = _clean_brief()
        data["geography"] = geography
        assert "geography_invalid" not in _codes(lint_generated_brief(data)), geography


def test_malformed_geography_is_blocking():
    for geography in (["Colombia"], {"facet_candidates": "Colombia"}, 42):
        data = _clean_brief()
        data["geography"] = geography
        findings = lint_generated_brief(data)
        assert "geography_invalid" in _codes(blocking_findings(findings)), geography


def test_lane_hint_string_patterns_are_blocking():
    """A bare-string patterns value would explode into single characters
    downstream — malformed hints block; missing hints only warn."""
    data = _clean_brief()
    data["domain_lane_hints"] = [{"lane": "payments_processors", "patterns": "stripe"}]
    findings = lint_generated_brief(data)
    assert "lane_hint_patterns_invalid" in _codes(blocking_findings(findings))


# ---------------------------------------------------------------------------
# Example compounds — the worked search levers the formation prompt renders.
# ---------------------------------------------------------------------------


def test_missing_example_compounds_is_warning_not_blocking():
    data = _clean_brief()
    del data["example_compounds"]
    findings = lint_generated_brief(data)
    assert "missing_example_compounds" in _codes(findings)
    assert blocking_findings(findings) == []


def test_example_compound_list_of_strings_is_blocking():
    """The loader hydrates each entry via ec.get(...); a bare string crashes
    _load_v2_brief with AttributeError, so malformed entries must block."""
    data = _clean_brief()
    data["example_compounds"] = ['("network design")']
    findings = lint_generated_brief(data)
    assert "example_compound_invalid" in _codes(blocking_findings(findings))


def test_example_compound_missing_boolean_is_blocking():
    data = _clean_brief()
    data["example_compounds"] = [
        {"purpose": "no boolean here", "novelty_bucket": "canonical"}
    ]
    findings = lint_generated_brief(data)
    assert "example_compound_invalid" in _codes(blocking_findings(findings))


def test_example_compound_boolean_lint_error_is_blocking():
    data = _clean_brief()
    data["example_compounds"] = [
        {
            "boolean": '("network design" AND',  # unbalanced paren → lint_boolean error
            "purpose": "malformed",
            "novelty_bucket": "canonical",
        }
    ]
    findings = lint_generated_brief(data)
    assert "example_compound_boolean_error" in _codes(blocking_findings(findings))


def test_example_compound_copied_from_illustrative_is_blocking():
    from shared.preflight_v2 import ILLUSTRATIVE_EXAMPLE_COMPOUNDS

    data = _clean_brief()
    data["example_compounds"] = [dict(ILLUSTRATIVE_EXAMPLE_COMPOUNDS[0])]
    findings = lint_generated_brief(data)
    assert "example_compound_copied" in _codes(blocking_findings(findings))


def test_example_compound_copy_check_ignores_whitespace_and_case():
    from shared.preflight_v2 import ILLUSTRATIVE_EXAMPLE_COMPOUNDS

    original = ILLUSTRATIVE_EXAMPLE_COMPOUNDS[0]["boolean"]
    # Re-spaced and re-cased — the normalized copy-check still catches it.
    mutated = "  " + original.upper().replace(" OR ", "   OR   ") + "  "
    data = _clean_brief()
    data["example_compounds"] = [
        {"boolean": mutated, "purpose": "sneaky copy", "novelty_bucket": "canonical"}
    ]
    findings = lint_generated_brief(data)
    assert "example_compound_copied" in _codes(blocking_findings(findings))


def test_example_compound_off_vocabulary_is_warning_not_blocking():
    """A boolean that shares no quoted term with the brief's key_terms /
    canonical patterns is a legitimate paraphrase → warning, never blocking."""
    data = _clean_brief()
    data["example_compounds"] = [
        {
            "boolean": '("unrelated widget" OR "gizmo assembly")',
            "purpose": "shares no term with the brief vocabulary",
            "novelty_bucket": "canonical",
        }
    ]
    findings = lint_generated_brief(data)
    assert "example_compound_off_vocabulary" in _codes(findings)
    assert "example_compound_off_vocabulary" not in _codes(blocking_findings(findings))
