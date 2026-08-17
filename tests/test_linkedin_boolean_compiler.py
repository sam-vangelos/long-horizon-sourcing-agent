"""Tests for linkedin/boolean_compiler.py (P2 warning-only linting)."""

import types

from shared.brief_schema import AbbreviationCollision
from shared.schemas import KitString
from shared.sourcing_lanes import SearchConstraint
from linkedin.boolean_compiler import (
    BooleanLintContext,
    boolean_lint_context_from_brief,
    compile_constraint,
    lint_boolean,
    lint_constraint_compile,
    lint_generated_string,
    summarize_kit_lint,
    ubiquitous_terms_from_brief,
)


HIGH_QUALITY_COMPOUND = (
    '("deployment engineer" OR "implementation engineer") AND '
    '("workflow orchestration" OR "tool calling") AND ("production" OR "deployed")'
)

BROAD_TITLE_STRICT = (
    '("Head of" OR "Director" OR "VP" OR "CTO" OR "Principal") AND '
    '("financial services" OR "banking") AND ("GenAI" OR "LLM")'
)


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def _severities(report) -> dict[str, str]:
    return {finding.code: finding.severity for finding in report.findings}


def test_high_quality_compound_has_no_errors():
    report = lint_boolean(HIGH_QUALITY_COMPOUND)
    assert not report.has_error


def test_unbalanced_parenthesis_is_error():
    report = lint_boolean('("deployment engineer" AND ("production")')
    assert report.has_error
    assert "unbalanced_parenthesis" in _codes(report)


def test_unbalanced_quote_is_error():
    report = lint_boolean('("deployment engineer')
    assert report.has_error
    assert "unbalanced_quote" in _codes(report)


def test_empty_or_group_is_error():
    report = lint_boolean('("a" OR ) AND "production"')
    assert report.has_error
    assert "empty_or_group" in _codes(report)


def test_malformed_operator_is_error():
    report = lint_boolean('("deployment" && "production")')
    assert report.has_error
    assert "malformed_operator" in _codes(report)


def test_quoted_ampersand_with_and_twin_is_not_blocked_or_noop_warned():
    report = lint_boolean('("trust & safety" OR "trust and safety")')

    assert not report.has_error
    assert "malformed_operator" not in _codes(report)
    assert "ampersand_missing_and_twin" not in _codes(report)
    assert "noop_special_character" not in _codes(report)


def test_quoted_ampersand_without_and_twin_warns_but_does_not_error():
    report = lint_boolean('("trust & safety" OR "platform ops")')

    assert not report.has_error
    assert "malformed_operator" not in _codes(report)
    assert "ampersand_missing_and_twin" in _codes(report)
    assert _severities(report)["ampersand_missing_and_twin"] == "warning"


def test_unquoted_bare_ampersand_remains_malformed_operator_error():
    report = lint_boolean("foo & bar")

    assert report.has_error
    assert "malformed_operator" in _codes(report)


def test_doubled_symbol_operators_remain_malformed_operator_errors():
    for boolean in ("foo && bar", "foo || bar"):
        report = lint_boolean(boolean)

        assert report.has_error
        assert "malformed_operator" in _codes(report)


def test_bare_generic_ai_terms_warning():
    report = lint_boolean('("AI" OR "LLM")')
    assert not report.has_error
    assert "bare_generic_ai_term" in _codes(report)
    assert _severities(report)["bare_generic_ai_term"] == "warning"


def test_bare_generic_ai_quoted_singleton_warning():
    report = lint_boolean('"AI"')
    assert "bare_generic_ai_term" in _codes(report)


def test_bare_generic_ai_single_term_group_warning():
    report = lint_boolean('("AI")')
    assert "bare_generic_ai_term" in _codes(report)


def test_bare_generic_ai_with_domain_discriminator_passes():
    # P1 item 5 (Wave 2): "banking" is vertical vocabulary — it discriminates
    # only when the BRIEF declares it (key_terms_by_area), never as a builtin.
    from types import SimpleNamespace

    from linkedin.boolean_compiler import (
        BooleanLintContext,
        boolean_lint_context_from_brief,
    )

    assert "bare_generic_ai_term" in _codes(lint_boolean("AI AND banking"))

    context = boolean_lint_context_from_brief(
        SimpleNamespace(
            key_terms_by_area={"domain": ["banking"]},
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
    )
    report = lint_boolean("AI AND banking", context=context)
    assert "bare_generic_ai_term" not in _codes(report)


def test_generic_ai_with_discriminator_passes_bare_check():
    report = lint_boolean('("GenAI" OR "production")')
    assert "bare_generic_ai_term" not in _codes(report)


def test_morphology_variant_missing_warning():
    report = lint_boolean('("deployment" OR "workflow orchestration")')
    assert "morphology_variant_missing" in _codes(report)


def test_morphology_pair_present_no_warning():
    report = lint_boolean('("deployment" OR "deployments" OR "workflow orchestration")')
    assert "morphology_variant_missing" not in _codes(report)


def test_abbreviation_collision_warning():
    context = BooleanLintContext(
        abbreviation_collisions=(
            AbbreviationCollision(abbreviation="IPO", expansion="Initial Public Offering"),
        )
    )
    report = lint_boolean('("IPO" OR "banking")', context=context)
    assert "abbreviation_collision" in _codes(report)


def test_strict_seniority_broad_title_bucket_warning():
    context = BooleanLintContext(strict_seniority=True)
    report = lint_boolean(BROAD_TITLE_STRICT, context=context)
    assert "strict_seniority_broad_title_bucket" in _codes(report)


def test_overlong_or_group_warning():
    terms = " OR ".join(f'"term{i}"' for i in range(10))
    report = lint_boolean(f"({terms})", context=BooleanLintContext(max_or_group_terms=8))
    assert "overlong_or_group" in _codes(report)


def test_unsupported_wildcard_warning():
    report = lint_boolean('("deploy*" OR "production")')
    assert "unsupported_wildcard" in _codes(report)


def test_boolean_filter_dimension_conflict():
    context = BooleanLintContext(structured_filters={"target_employers": ["Goldman Sachs"]})
    report = lint_boolean('("Goldman Sachs" OR "JPMorgan") AND ("GenAI")', context=context)
    assert "boolean_filter_dimension_conflict" in _codes(report)


def test_compile_constraint_boolean_keyword():
    constraint = SearchConstraint(
        dimension="capability",
        values=["workflow orchestration", "tool calling"],
        execution_surface="boolean_keyword",
        operator="prefer",
    )
    compiled = compile_constraint(constraint)
    assert compiled.boolean_fragment
    assert "workflow orchestration" in compiled.boolean_fragment
    assert "tool calling" in compiled.boolean_fragment


def test_compile_constraint_exclude_wraps_not():
    constraint = SearchConstraint(
        dimension="anti_noise",
        values=["sales engineer"],
        execution_surface="boolean_keyword",
        operator="exclude",
    )
    compiled = compile_constraint(constraint)
    assert compiled.boolean_fragment.startswith("NOT ")


def test_compile_constraint_linkedin_company_filter():
    constraint = SearchConstraint(
        dimension="context",
        values=["Goldman Sachs"],
        execution_surface="linkedin_company_filter",
        operator="prefer",
        temporal_scope="current",
    )
    compiled = compile_constraint(constraint)
    assert compiled.structured_control.get("dimension") == "company"
    assert "Goldman Sachs" in compiled.structured_control.get("values", [])


def test_lint_constraint_compile_surface_ambiguous():
    constraint = SearchConstraint(
        dimension="capability",
        values=["production"],
        operator="require",
        execution_surface="soft_hint",
    )
    compiled = compile_constraint(constraint)
    findings = lint_constraint_compile(constraint, compiled)
    assert any(f.code == "execution_surface_ambiguous" for f in findings)


def test_lint_constraint_compile_temporal_mismatch():
    constraint = SearchConstraint(
        dimension="reality",
        values=["production"],
        operator="prefer",
        execution_surface="boolean_keyword",
        temporal_scope="current",
    )
    compiled = compile_constraint(constraint)
    findings = lint_constraint_compile(constraint, compiled)
    assert any(f.code == "temporal_scope_mismatch" for f in findings)


def test_lint_generated_string_attaches_item_context():
    item = {
        "boolean": '("Goldman Sachs") AND ("GenAI")',
        "retrieval_recipe": {"target_employers": ["Goldman Sachs"]},
    }
    report = lint_generated_string(item)
    assert report.boolean
    assert "boolean_filter_dimension_conflict" in _codes(report)


def test_lint_report_to_dict_round_trip():
    report = lint_boolean('("AI" OR "LLM")')
    payload = report.to_dict()
    assert payload["boolean"]
    assert isinstance(payload["findings"], list)
    assert payload["has_error"] is False


def test_boolean_lint_context_from_brief_strict_seniority():
    class _Brief:
        abbreviation_collisions = ()
        role_description = "Executive director BFSI bank financial services"
        minimum_years_experience = 15

    context = boolean_lint_context_from_brief(_Brief())
    assert context.strict_seniority is True


def test_attach_constraint_lint_to_plan():
    from linkedin.boolean_compiler import attach_constraint_lint_to_plan
    from shared.schemas import ExecutionPlan
    from shared.sourcing_lanes import (
        LaneExecution,
        SearchHypothesis,
        SearchSlice,
        SourcingLane,
    )

    lane = SourcingLane(
        lane_id="lane-1",
        lane_name="Lane",
        hypothesis=SearchHypothesis(
            hypothesis_id="h1",
            label="Label",
            target_archetype="arch",
            why_this_pool_may_exist="because",
        ),
        slice=SearchSlice(
            slice_id="s1",
            hypothesis_id="h1",
            label="Slice",
            objective="objective",
            constraints=[
                SearchConstraint(
                    dimension="title",
                    values=["Director"],
                    execution_surface="boolean_keyword",
                    temporal_scope="current",
                    operator="require",
                )
            ],
        ),
        execution=LaneExecution(lane_id="lane-1", source="linkedin"),
    )
    plan = ExecutionPlan(strategy_rationale="test", sourcing_lanes=[lane.to_dict()])
    attach_constraint_lint_to_plan(plan)
    assert "constraint_lint" in plan.sourcing_lanes[0]
    assert plan.sourcing_lanes[0]["constraint_lint"]


# --- P5.2 noop-token lint checks -------------------------------------------


def test_noop_special_character_fires_on_dollar_ampersand_percent_plus():
    for boolean in ('"AT&T"', '"$M revenue"', '"100% remote"', '"C++ engineer"'):
        report = lint_boolean(boolean)
        assert "noop_special_character" in _codes(report), boolean
        assert _severities(report)["noop_special_character"] == "warning"


def test_at_and_t_abbreviation_keeps_noop_warning_without_ampersand_twin_warning():
    report = lint_boolean('"AT&T"')

    assert not report.has_error
    assert "noop_special_character" in _codes(report)
    assert "ampersand_missing_and_twin" not in _codes(report)


def test_noop_special_character_clean_sibling_does_not_fire():
    report = lint_boolean('"AT and T"')
    assert "noop_special_character" not in _codes(report)


def test_noop_comma_numeral_fires_on_comma_separated_number():
    report = lint_boolean('"1,000 employees"')
    assert "noop_comma_numeral" in _codes(report)
    assert _severities(report)["noop_comma_numeral"] == "warning"


def test_noop_comma_numeral_clean_sibling_does_not_fire():
    report = lint_boolean('"1000 employees"')
    assert "noop_comma_numeral" not in _codes(report)


def test_mid_word_stem_fires_on_hyphen_and_suffix_endings():
    for boolean in ('"re-sequenc"', '"analyz"', '"synthes-"'):
        report = lint_boolean(boolean)
        assert "mid_word_stem" in _codes(report), boolean
        assert _severities(report)["mid_word_stem"] == "warning"


def test_mid_word_stem_exception_words_do_not_fire():
    # "Acme Inc" is the load-bearing case — company terms ending in Inc are
    # routine in booleans and must never read as truncated stems.
    for boolean in (
        '"zinc"',
        '"quiz"',
        '"showbiz"',
        '"biz"',
        '"Acme Inc"',
        '"async"',
        '"data sync"',
        '"func"',
    ):
        report = lint_boolean(boolean)
        assert "mid_word_stem" not in _codes(report), boolean


def test_mid_word_stem_clean_sibling_does_not_fire():
    report = lint_boolean('"analyst"')
    assert "mid_word_stem" not in _codes(report)


def test_not_group_with_and_is_error():
    report = lint_boolean('NOT ("a" AND "b")')
    assert report.has_error
    assert "not_group_contains_and" in _codes(report)
    assert _severities(report)["not_group_contains_and"] == "error"


def test_not_group_with_or_is_clean():
    report = lint_boolean('NOT ("a" OR "b")')
    assert "not_group_contains_and" not in _codes(report)


def test_not_single_term_is_clean():
    report = lint_boolean('NOT "a"')
    assert "not_group_contains_and" not in _codes(report)


def test_not_group_and_at_any_nesting_depth_fires():
    report = lint_boolean('NOT ("a" OR ("b" AND "c"))')
    assert "not_group_contains_and" in _codes(report)


def test_lowercase_not_does_not_trigger_not_group_check():
    # Lowercase `not` is deliberately left to the lowercase_operator warning,
    # not the NOT-group grammar error.
    report = lint_boolean('not ("a" AND "b")')
    assert "not_group_contains_and" not in _codes(report)
    assert "lowercase_operator" in _codes(report)


def test_lowercase_operator_warning_between_groups_has_uppercase_rewrite():
    report = lint_boolean('("a") and ("b")')
    findings = [f for f in report.findings if f.code == "lowercase_operator"]
    assert findings
    assert findings[0].severity == "warning"
    assert "AND" in findings[0].repair_hint


def test_lowercase_operator_uppercase_is_clean():
    report = lint_boolean('("a") AND ("b")')
    assert "lowercase_operator" not in _codes(report)


def test_lowercase_operator_quoted_term_does_not_fire():
    report = lint_boolean('("hand" OR "and")')
    assert "lowercase_operator" not in _codes(report)


def test_boolean_length_cap_fires_over_2000_chars():
    long_boolean = "(" + " OR ".join(f'"term{i}"' for i in range(300)) + ")"
    assert len(long_boolean) > 2000
    report = lint_boolean(long_boolean)
    assert "boolean_length_cap" in _codes(report)
    assert _severities(report)["boolean_length_cap"] == "warning"


def test_boolean_length_cap_clean_under_limit():
    report = lint_boolean('("short" OR "boolean")')
    assert "boolean_length_cap" not in _codes(report)


# --- jd_register_overuse (candidate-register split, WARNING) ----------------


def test_jd_register_overuse_fires_on_live_plan_shaped_key_terms():
    context = boolean_lint_context_from_brief(
        types.SimpleNamespace(
            key_terms_by_area={
                "evaluation": [
                    "eval harness",
                    "rubric design",
                    "model scoring",
                    "quality review",
                ]
            },
            candidate_register_terms_by_area={"evaluation": []},
            abbreviation_collisions=[],
        )
    )
    report = lint_boolean(
        '("eval harness" OR "rubric design" OR "model scoring" OR "candidate evidence")',
        context=context,
    )
    findings = [finding for finding in report.findings if finding.code == "jd_register_overuse"]
    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert "3/4" in findings[0].message
    assert "75%" in findings[0].message
    assert "eval harness" in findings[0].message
    assert "candidate_register_terms" in findings[0].repair_hint


def test_jd_register_overuse_ignores_candidate_register_vocabulary():
    register_terms = [
        "maintained library",
        "authored RFCs",
        "shipped SDKs",
        "triaged issues",
    ]
    context = boolean_lint_context_from_brief(
        types.SimpleNamespace(
            key_terms_by_area={"portfolio": register_terms},
            candidate_register_terms_by_area={"portfolio": register_terms},
            abbreviation_collisions=[],
        )
    )
    report = lint_boolean(
        '("maintained library" OR "authored RFCs" OR "shipped SDKs" OR "triaged issues")',
        context=context,
    )
    assert "jd_register_overuse" not in _codes(report)


def test_jd_register_overuse_respects_four_quoted_term_guard():
    context = boolean_lint_context_from_brief(
        types.SimpleNamespace(
            key_terms_by_area={"evaluation": ["rubric", "calibration"]},
            candidate_register_terms_by_area={"evaluation": []},
            abbreviation_collisions=[],
        )
    )
    report = lint_boolean('("rubric" OR "calibration")', context=context)
    assert "jd_register_overuse" not in _codes(report)


def test_jd_register_overuse_old_brief_uses_key_terms_and_area_names():
    context = boolean_lint_context_from_brief(
        types.SimpleNamespace(
            key_terms_by_area={"evaluation": ["rubric", "calibration", "assessment"]},
            abbreviation_collisions=[],
        )
    )
    report = lint_boolean(
        '("evaluation" OR "rubric" OR "calibration" OR "candidate evidence")',
        context=context,
    )
    assert "jd_register_overuse" in _codes(report)


def test_jd_register_overuse_does_not_bypass_existing_error_gate():
    context = boolean_lint_context_from_brief(
        types.SimpleNamespace(
            key_terms_by_area={
                "evaluation": [
                    "eval harness",
                    "rubric design",
                    "model scoring",
                    "quality review",
                ]
            },
            candidate_register_terms_by_area={"evaluation": []},
            abbreviation_collisions=[],
        )
    )
    report = lint_boolean(
        '("eval harness" OR "rubric design") AND ("model scoring" OR "quality review"',
        context=context,
    )
    assert report.has_error
    assert "unbalanced_parenthesis" in _codes(report)
    assert "jd_register_overuse" not in _codes(report)


# --- ubiquitous_terms_from_brief (R2-F5 live feed) --------------------------


def test_ubiquitous_terms_from_brief_structural_defaults_always_present():
    terms = ubiquitous_terms_from_brief(types.SimpleNamespace())
    assert {"ai", "engineer", "software", "technology"} <= terms


def test_ubiquitous_terms_from_brief_unions_blacklist_categories_normalized():
    category = types.SimpleNamespace(
        label="Vertical noise",
        rationale="too generic in this vertical",
        terms=["Fintech", "  Banking  "],
    )
    brief = types.SimpleNamespace(term_blacklist_categories=[category])
    terms = ubiquitous_terms_from_brief(brief)
    assert {"ai", "engineer", "software", "technology", "fintech", "banking"} <= terms


def test_ubiquitous_terms_from_brief_tolerates_non_list_attribute():
    brief = types.SimpleNamespace(term_blacklist_categories="not-a-list")
    terms = ubiquitous_terms_from_brief(brief)
    assert terms == {"ai", "engineer", "software", "technology"}


def test_ubiquitous_terms_from_brief_tolerates_missing_attribute():
    class _NoBlacklistBrief:
        pass

    terms = ubiquitous_terms_from_brief(_NoBlacklistBrief())
    assert terms == {"ai", "engineer", "software", "technology"}


# --- summarize_kit_lint (advisory-only, R4-F3) ------------------------------


def test_summarize_kit_lint_flags_defect_and_is_advisory():
    kit_strings = [
        KitString(id=1, block="b", subblock="s", string_type="Recall", boolean='"AT&T"'),
    ]
    summary = summarize_kit_lint(kit_strings)
    assert summary
    assert "noop_special_character" in summary
    assert "1/1" in summary


def test_summarize_kit_lint_empty_for_clean_kit_strings():
    kit_strings = [
        KitString(
            id=1,
            block="b",
            subblock="s",
            string_type="Recall",
            boolean='("deployment" OR "deployments")',
        ),
    ]
    assert summarize_kit_lint(kit_strings) == ""


# --- low_signal_and_clause (Cathey Maximum-Inclusion, WARNING) --------------


def test_low_signal_and_clause_fires_on_all_generic_verb_or_group():
    # Cathey Maximum-Inclusion: an AND-required OR group made entirely of
    # generic verbs over-constrains recall.
    report = lint_boolean(
        '("managed" OR "led" OR "owned") AND ("Terraform" OR "Kubernetes")'
    )
    assert not report.has_error
    assert "low_signal_and_clause" in _codes(report)
    assert _severities(report)["low_signal_and_clause"] == "warning"


def test_low_signal_and_clause_fires_on_single_term_group():
    report = lint_boolean('("Terraform" OR "Kubernetes") AND ("hands-on")')
    assert "low_signal_and_clause" in _codes(report)
    assert _severities(report)["low_signal_and_clause"] == "warning"


def test_low_signal_and_clause_clean_on_title_bucket_group():
    # Titles are pool bounds, not filler — none are in the low-signal set.
    report = lint_boolean('("head of" OR "director" OR "VP")')
    assert "low_signal_and_clause" not in _codes(report)


def test_low_signal_and_clause_clean_on_mixed_group():
    # Mixed group fails the all-terms predicate: "Terraform" is real signal.
    report = lint_boolean('("managed" OR "Terraform")')
    assert "low_signal_and_clause" not in _codes(report)


def test_low_signal_and_clause_clean_inside_not_exclusion():
    # An exclusion is not an AND gate; a repair hint there is nonsense.
    report = lint_boolean('NOT ("managed" OR "led")')
    assert "low_signal_and_clause" not in _codes(report)


def test_low_signal_and_clause_clean_when_brief_declares_discriminator():
    # A brief that declares "managed" as key vocabulary un-flags it.
    from types import SimpleNamespace

    context = boolean_lint_context_from_brief(
        SimpleNamespace(
            key_terms_by_area={"craft": ["managed"]},
            abbreviation_collisions=[],
        )
    )
    report = lint_boolean(
        '("managed" OR "led") AND ("Terraform" OR "Kubernetes")',
        context=context,
    )
    assert "low_signal_and_clause" not in _codes(report)


def test_low_signal_terms_disjoint_from_builder_title_and_discriminator_vocab():
    from linkedin.boolean_compiler import (
        _BUILDER_PROOF_PATTERNS,
        _STRUCTURAL_CAPABILITY_DISCRIMINATORS,
        _STRUCTURAL_LOW_SIGNAL_TERMS,
        _TITLE_HEAVY_PATTERNS,
    )

    assert _STRUCTURAL_LOW_SIGNAL_TERMS.isdisjoint(_BUILDER_PROOF_PATTERNS)
    assert _STRUCTURAL_LOW_SIGNAL_TERMS.isdisjoint(_STRUCTURAL_CAPABILITY_DISCRIMINATORS)
    assert _STRUCTURAL_LOW_SIGNAL_TERMS.isdisjoint(_TITLE_HEAVY_PATTERNS)
