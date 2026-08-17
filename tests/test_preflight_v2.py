"""First tests for shared/preflight_v2.py (P9.2c — previously zero coverage).

Covers prompt generation, response parsing (including the markdown-fence
strip and the failure/malformed-JSON path), the brief-json + overrides
merge, and the ``format_for_review`` rendering that P9.3 wires into the
console at generation time.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.preflight_v2 import (
    format_for_review,
    generate_preflight_prompt,
    parse_preflight_response,
    preflight_to_brief_json,
)


# ---------------------------------------------------------------------------
# generate_preflight_prompt
# ---------------------------------------------------------------------------


def test_generate_preflight_prompt_includes_jd_text():
    prompt = generate_preflight_prompt("We need a senior backend engineer.")

    assert "We need a senior backend engineer." in prompt
    assert "GEOGRAPHY CONTEXT" not in prompt


def test_generate_preflight_prompt_includes_geography_when_present():
    prompt = generate_preflight_prompt("JD text", geography="New York, NY")

    assert "GEOGRAPHY CONTEXT" in prompt
    assert "New York, NY" in prompt


def test_generate_preflight_prompt_omits_operator_calibration_by_default():
    prompt = generate_preflight_prompt("JD text")

    assert "OPERATOR CALIBRATION" not in prompt


def test_generate_preflight_prompt_includes_operator_guidance_when_present():
    prompt = generate_preflight_prompt(
        "JD text",
        operator_guidance="- Target 4-10 years of relevant experience.",
    )

    assert "OPERATOR CALIBRATION" in prompt
    assert "Target 4-10 years of relevant experience." in prompt
    # The calibration block instructs structural expression and re-affirms
    # the lint rules — the operator channel must not become a hedge-language
    # injection path.
    assert "never" in prompt
    assert "disposition language" in prompt


def test_preflight_prompt_requires_explicit_operator_authorization_for_hard_ceiling():
    prompt = generate_preflight_prompt("Candidates generally have 4-10 years.")

    assert '"maximum_years_experience_is_hard"' in prompt
    assert "operator" in prompt.lower()
    assert "explicit" in prompt.lower()
    assert "hard" in prompt.lower()
    assert "must be false" in prompt.lower()
    assert "a stated experience range alone" in prompt.lower()
    assert "does not make the ceiling hard" in prompt.lower()


def test_generate_preflight_prompt_omits_intake_notes_by_default():
    prompt = generate_preflight_prompt("JD text")

    assert "RECRUITER INTAKE NOTES" not in prompt
    # None and whitespace-only intake render identically to the default —
    # the channel is fail-soft for every intake-less caller.
    assert prompt == generate_preflight_prompt("JD text", intake_notes=None)
    assert prompt == generate_preflight_prompt("JD text", intake_notes="   ")


def test_generate_preflight_prompt_includes_intake_notes_when_present():
    prompt = generate_preflight_prompt(
        "JD text",
        intake_notes="Priority 1: widget-calibration operations background.",
    )

    assert "RECRUITER INTAKE NOTES" in prompt
    assert "Priority 1: widget-calibration operations background." in prompt
    # The intake renders as its own labeled block AFTER the JD — never
    # concatenated into it: the prompt's register rules treat JD phrasing
    # as requisition-suspect and would discount the intake's curated
    # vocabulary if it arrived labeled as JD text.
    assert prompt.index("JD text") < prompt.index("RECRUITER INTAKE NOTES")
    # The block subordinates itself to operator calibration and re-affirms
    # the candidate-register rule — a new vocabulary channel into brief
    # generation must not bypass the register doctrine.
    assert "outranks both" in prompt
    assert "plausibly write" in prompt


# ---------------------------------------------------------------------------
# parse_preflight_response — parse path
# ---------------------------------------------------------------------------


def test_parse_preflight_response_parses_plain_json():
    raw = json.dumps({"role_title": "Senior Engineer"})

    result = parse_preflight_response(raw)

    assert result == {"role_title": "Senior Engineer"}


def test_parse_preflight_response_strips_markdown_fences():
    raw = "```json\n" + json.dumps({"role_title": "Senior Engineer"}) + "\n```"

    result = parse_preflight_response(raw)

    assert result == {"role_title": "Senior Engineer"}


def test_parse_preflight_response_strips_bare_fences_without_language_tag():
    raw = "```\n" + json.dumps({"role_title": "Senior Engineer"}) + "\n```"

    result = parse_preflight_response(raw)

    assert result == {"role_title": "Senior Engineer"}


# ---------------------------------------------------------------------------
# parse_preflight_response — failure path
# ---------------------------------------------------------------------------


def test_parse_preflight_response_raises_on_malformed_json():
    with pytest.raises(json.JSONDecodeError):
        parse_preflight_response("not json at all")


# ---------------------------------------------------------------------------
# preflight_to_brief_json — apply path
# ---------------------------------------------------------------------------


def test_preflight_to_brief_json_merges_scalar_overrides():
    preflight_data = {"role_title": "Senior Engineer", "market_density": "sparse"}

    result = preflight_to_brief_json(preflight_data, {"market_density": "dense"})

    assert result["role_title"] == "Senior Engineer"
    assert result["market_density"] == "dense"


def test_preflight_to_brief_json_merges_nested_dict_overrides():
    preflight_data = {
        "facial_calibration": {
            "expected_yes_rate_low": 0.25,
            "expected_yes_rate_high": 0.55,
        }
    }

    result = preflight_to_brief_json(
        preflight_data, {"facial_calibration": {"expected_yes_rate_low": 0.1}}
    )

    assert result["facial_calibration"] == {
        "expected_yes_rate_low": 0.1,
        "expected_yes_rate_high": 0.55,
    }


def test_preflight_to_brief_json_noop_without_overrides():
    preflight_data = {"role_title": "Senior Engineer"}

    result = preflight_to_brief_json(preflight_data)

    assert result == preflight_data
    assert result is not preflight_data  # copy, not the same dict


def test_preflight_to_brief_json_canonicalizes_nested_hiring_company_without_mutation():
    preflight_data = {
        "hiring_company": "Acme Fintech",
        "engagement_context": {
            "hiring_company": "Wrong Company",
            "selectivity_posture": "selective",
        },
    }

    result = preflight_to_brief_json(preflight_data)

    assert result["engagement_context"] == {
        "hiring_company": "Acme Fintech",
        "selectivity_posture": "selective",
    }
    assert preflight_data["engagement_context"] == {
        "hiring_company": "Wrong Company",
        "selectivity_posture": "selective"
    }


# ---------------------------------------------------------------------------
# format_for_review
# ---------------------------------------------------------------------------


def _full_preflight_data() -> dict:
    return {
        "role_title": "Senior Backend Engineer",
        "role_level": "Senior",
        "role_summary": "Builds payment infrastructure.",
        "capability_areas": [
            {
                "name": "Payments",
                "description": "Owns payment rails.",
                "builder_signals": ["designs ledger systems"],
                "user_signals": ["integrates a payments API"],
                "key_terms": ["idempotency"],
                "candidate_register_terms": ["payment orchestration"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Builds the payment rails themselves.",
            "user_definition": "Calls a payments API someone else built.",
            "edge_case_guidance": "Look for ownership of failure handling.",
        },
        "non_fit_patterns": [
            {
                "label": "Frontend checkout dev",
                "description": "Builds checkout UI only.",
                "why_not": "Never touches ledger or settlement.",
                "examples": ["Checkout page redesign"],
            }
        ],
        "employer_signal_rules": [
            {
                "tier": "strong_ai",
                "employer_patterns": ["Stripe"],
                "evidence_required": "ledger ownership",
                "save_on_employer_alone": False,
            }
        ],
        "minimum_years_experience": 5,
        "minimum_bar_description": "5+ years owning payment infra end to end.",
        "preflight_confidence_notes": "JD is vague on team size.",
        "facial_calibration": {
            "fast_exit_patterns": ["pure frontend career"],
            "trajectory_yes_patterns": ["ledger systems engineer"],
            "trajectory_ambiguous_patterns": ["backend engineer, domain unclear"],
            "trajectory_no_patterns": ["entire career in marketing analytics"],
        },
    }


def test_format_for_review_renders_all_sections():
    rendered = format_for_review(_full_preflight_data())

    assert "PREFLIGHT REVIEW" in rendered
    assert "Senior Backend Engineer" in rendered
    assert "CAPABILITY AREAS" in rendered
    assert "Payments" in rendered
    assert "Candidate-register terms: payment orchestration" in rendered
    assert "DEPTH DISTINCTION" in rendered
    assert "Builds the payment rails themselves." in rendered
    assert "UNKNOWN" in rendered
    assert "missing evidence" in rendered.lower()
    assert "BUILDER (save)" not in rendered
    assert "USER (reject)" not in rendered
    assert "NON-FIT PATTERNS" in rendered
    assert "Frontend checkout dev" in rendered
    assert "EMPLOYER SIGNAL RULES" in rendered
    assert "Stripe" in rendered
    assert "MINIMUM BAR" in rendered
    assert "5+" in rendered
    assert "PREFLIGHT CONFIDENCE NOTES" in rendered
    assert "JD is vague on team size." in rendered
    assert "FACIAL TRIAGE" in rendered
    assert "ledger systems engineer" in rendered


def test_format_for_review_tolerates_missing_optional_sections():
    minimal = {"role_title": "Senior Backend Engineer"}

    rendered = format_for_review(minimal)

    assert "PREFLIGHT REVIEW" in rendered
    assert "Senior Backend Engineer" in rendered
    # No confidence notes section when the field is absent/empty.
    assert "PREFLIGHT CONFIDENCE NOTES" not in rendered


def test_format_for_review_renders_engagement_context():
    data = _full_preflight_data()
    data["engagement_context"] = {
        "engagement_description": "A focused payments search.",
        "talent_bar_statement": "Direct rail ownership clears the bar.",
        "selectivity_posture": "selective",
    }

    rendered = format_for_review(data)

    assert "Engagement: A focused payments search." in rendered
    assert "Talent bar: Direct rail ownership clears the bar." in rendered
    assert "Selectivity posture: selective" in rendered


def test_real_v2_loader_tolerates_provenance_stamp():
    """P9.3 hardening (Opus review): the provenance stamp must survive the
    REAL loader, not a mocked one — validate_v2_brief ignores unknown keys,
    and this locks that tolerance so a future strict-mode change can't
    silently break provenance-stamped preflight briefs."""
    from shared.brief_loader import _load_v2_brief
    from tests.test_calibration_brief_fields import _minimal_v2_raw

    raw = _minimal_v2_raw()
    raw["provenance"] = {
        "generated_by": "preflight_v2",
        "reviewed": False,
        "generated_at": "2026-07-03T00:00:00+00:00",
    }

    brief = _load_v2_brief(raw)

    assert brief.raw["provenance"]["reviewed"] is False


# ---------------------------------------------------------------------------
# P4 (plans/sourcing-rigor-hardening.md) — the template itself carries no
# disposition language and asks for the new structured fields
# ---------------------------------------------------------------------------


def test_preflight_prompt_template_carries_no_default_yes_hedge():
    prompt = generate_preflight_prompt("Any JD text")

    # The exact hedge sentences the audit confirmed as a doctrine violation
    # (calibration finding #4) must never reappear in the template.
    assert "These MUST default to YES" not in prompt
    assert "when in doubt, a pattern is ambiguous, not NO" not in prompt
    # The ambiguous-pattern field now describes without disposing.
    assert "Describe the pattern only — no disposition language." in prompt


def test_preflight_prompt_template_requests_new_structured_fields():
    prompt = generate_preflight_prompt("Any JD text")

    assert '"hiring_company"' in prompt
    assert '"employer_blacklist"' in prompt
    assert '"engagement_context"' in prompt
    assert '"selectivity_posture"' in prompt
    assert "selective for dense/moderate markets" in prompt
    assert "coverage only for sparse markets" in prompt
    assert "engagement_context.hiring_company" in prompt
    assert '"candidate_register_terms"' in prompt
    assert "qualified candidate would plausibly WRITE" in prompt
    assert "distinct channel from key_terms" in prompt
    assert '"yes_rate_rationale"' in prompt
    assert '"domain_lane_hints"' in prompt
    assert '"geography"' in prompt
    assert '"maximum_years_experience_is_hard"' in prompt
    # Band literals are gone from the schema — null placeholders + derivation
    # rule replace the anchoring 0.25/0.55 example values.
    assert '"expected_yes_rate_low": null' in prompt
    assert "Never reuse a band from an example" in prompt


def test_preflight_prompt_experience_measure_defaults_to_total_professional():
    prompt = generate_preflight_prompt("Any JD text")

    assert (
        "experience_measure may only RESTRICT which years count if the JD or "
        "an operator instruction states the restriction."
    ) in prompt
    assert (
        "Absent a stated restriction, experience_measure counts total professional "
        "experience, including research and advanced-degree years."
    ) in prompt
    assert (
        "'total professional experience, including research and advanced-degree years' "
        "by default"
    ) in prompt


def test_preflight_prompt_candidate_register_channel_is_self_written():
    prompt = generate_preflight_prompt("Any JD text")

    assert (
        "Every quoted term must be plausibly self-written by a candidate on "
        "their profile, never a JD heading phrase."
    ) in prompt
    assert "distribution network design" in prompt
    assert "S&OP planning" in prompt


def test_format_for_review_renders_band_blacklist_and_lane_sections():
    data = _full_preflight_data()
    data["hiring_company"] = "Acme Fintech"
    data["employer_blacklist"] = ["Acme Fintech"]
    data["facial_calibration"]["expected_yes_rate_low"] = 0.15
    data["facial_calibration"]["expected_yes_rate_high"] = 0.35
    data["facial_calibration"]["yes_rate_rationale"] = "Moderate-density pool."
    data["domain_lane_hints"] = [{"lane": "payments_processors", "patterns": ["stripe"]}]
    data["geography"] = {"facet_candidates": ["Colombia"], "rationale": "JD names Colombia."}

    rendered = format_for_review(data)

    assert "Hiring company: Acme Fintech" in rendered
    assert "FACIAL YES-RATE BAND" in rendered
    assert "Moderate-density pool." in rendered
    assert "DOMAIN LANE HINTS" in rendered
    assert "payments_processors" in rendered
    assert "GEOGRAPHY (extracted from JD)" in rendered
    assert "snippet cannot resolve; the evaluation template decides treatment" in rendered
    assert "default YES" not in rendered


def test_format_for_review_distinguishes_advisory_and_hard_experience_ceilings():
    advisory = _full_preflight_data()
    advisory["maximum_years_experience"] = 10
    advisory["experience_measure"] = "total professional experience"
    advisory["maximum_years_experience_is_hard"] = False

    advisory_rendered = format_for_review(advisory)
    assert "10" in advisory_rendered
    assert "advisory" in advisory_rendered.lower()
    assert "not an automatic reject" in advisory_rendered.lower()

    hard = json.loads(json.dumps(advisory))
    hard["maximum_years_experience_is_hard"] = True
    hard_rendered = format_for_review(hard)
    assert "10" in hard_rendered
    assert "hard" in hard_rendered.lower()
    assert "reject gate" in hard_rendered.lower()


def test_format_confidence_notes_returns_banner_or_empty():
    from shared.preflight_v2 import format_confidence_notes

    assert format_confidence_notes({}) == ""
    assert format_confidence_notes({"preflight_confidence_notes": "  "}) == ""

    block = format_confidence_notes(
        {"preflight_confidence_notes": "Role level is ambiguous — confirm the band."}
    )
    assert "⚠  PREFLIGHT CONFIDENCE NOTES — REVIEW THESE CAREFULLY" in block
    assert "Role level is ambiguous — confirm the band." in block


# ---------------------------------------------------------------------------
# RC3 (2026-07-04): opening mirrors — preflight emits the canonical/edge-case
# pattern fields the deterministic opening sort consumes. Standing rule
# (Wave-1 addendum 7): every new preflight field gets a parse→lint→load
# round-trip at introduction.
# ---------------------------------------------------------------------------


from tests.test_orchestrator_preflight import _valid_preflight_dict


def test_engagement_context_round_trips_through_real_loader():
    from shared.brief_loader import _load_v2_brief
    from shared.brief_v2_schema import validate_v2_brief

    data = _valid_preflight_dict()
    brief_json = preflight_to_brief_json(data)

    validate_v2_brief(brief_json)
    brief = _load_v2_brief(brief_json)

    assert brief.engagement_context == data["engagement_context"]
    assert brief._new_brief.engagement_context == data["engagement_context"]


def test_shared_preflight_seam_pins_call_args_and_finalizes_after_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shared.preflight_v2 import (
        PREFLIGHT_MAX_TOKENS,
        PREFLIGHT_STAGE,
        PREFLIGHT_SYSTEM_PROMPT,
        finalize_preflight_v2,
        generate_preflight_v2_once,
    )

    brief = SimpleNamespace(
        id="brief-safe",
        jd_text="We need a senior backend engineer.",
        permanent_filters={"Location": "United States"},
        instructions=["Use the stated senior ownership bar."],
        intake_notes="Prioritize end-to-end ownership.",
        linkedin_project="Strategic Project Lead",
        linkedin_project_id="3000000004",
        kit_url="",
        employer_blacklist=["Acme Fintech"],
    )
    original_context = {
        "brief_id": brief.id,
        "parent_logical_call_id": "preflight-parent",
        "logical_call_id": "must-not-be-reused",
    }
    captured: dict = {}

    def fake_llm(system_prompt, user_prompt, **kwargs):
        captured.update(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                **kwargs,
            }
        )
        assert "logical_call_id" not in kwargs["usage_context"]
        kwargs["usage_context"]["logical_call_id"] = "llm-fresh"
        return json.dumps(_valid_preflight_dict())

    generation = generate_preflight_v2_once(
        brief,
        model_name="accounts/fireworks/models/glm-5p2",
        usage_context=original_context,
        llm_call=fake_llm,
    )

    assert captured["system_prompt"] == PREFLIGHT_SYSTEM_PROMPT
    assert "RECRUITER INTAKE NOTES" in captured["user_prompt"]
    assert "OPERATOR CALIBRATION" in captured["user_prompt"]
    assert captured["expect_json"] is False
    assert captured["max_tokens"] == PREFLIGHT_MAX_TOKENS
    assert captured["model_name"] == "accounts/fireworks/models/glm-5p2"
    assert captured["usage_context"]["stage"] == PREFLIGHT_STAGE
    assert captured["usage_context"]["parent_logical_call_id"] == "preflight-parent"
    assert generation.usage_context["logical_call_id"] == "llm-fresh"
    assert original_context["logical_call_id"] == "must-not-be-reused"

    loaded = SimpleNamespace(has_v2_schema=True)
    loader = MagicMock(return_value=loaded)
    monkeypatch.setattr("shared.brief_loader._load_v2_brief", loader)
    execution = finalize_preflight_v2(brief, generation)

    assert execution.brief is loaded
    assert execution.brief_json["linkedin_project_id"] == "3000000004"
    assert (
        execution.brief_json["source_config"]["linkedin"]["project_id"]
        == "3000000004"
    )
    assert execution.brief_json["provenance"]["reviewed"] is False
    loader.assert_called_once_with(execution.brief_json)


def _mirror_fields() -> dict:
    return {
        "canonical_title_patterns": ["strategic project lead", "delivery lead"],
        "canonical_company_patterns": ["scale ai", "surge ai"],
        "canonical_framework_patterns": [],
        "canonical_broad_patterns": ["program delivery leader"],
        "edge_case_patterns": ["rater program manager at a search engine"],
        "edge_case_company_patterns": ["appen"],
    }


def test_preflight_prompt_asks_for_opening_mirrors():
    prompt = generate_preflight_prompt("JD text")
    for field in (
        "canonical_title_patterns",
        "canonical_company_patterns",
        "canonical_framework_patterns",
        "canonical_broad_patterns",
        "edge_case_patterns",
        "edge_case_company_patterns",
    ):
        assert field in prompt


def test_opening_mirrors_round_trip_parse_lint_load():
    from shared.brief_lint import blocking_findings, lint_generated_brief
    from shared.brief_loader import _load_v2_brief
    from shared.preflight_v2 import preflight_to_brief_json

    data = json.loads(json.dumps({**_valid_preflight_dict(), **_mirror_fields()}))
    findings = lint_generated_brief(data)
    assert blocking_findings(findings) == []
    assert not any(f.code == "missing_opening_mirrors" for f in findings)

    brief_json = preflight_to_brief_json(data)
    brief = _load_v2_brief(brief_json)
    assert brief.canonical_title_patterns == ["strategic project lead", "delivery lead"]
    assert brief.canonical_company_patterns == ["scale ai", "surge ai"]
    assert brief.edge_case_patterns == ["rater program manager at a search engine"]
    assert brief.edge_case_company_patterns == ["appen"]


def test_lint_warns_when_all_opening_mirrors_absent():
    from shared.brief_lint import blocking_findings, lint_generated_brief

    findings = lint_generated_brief(_valid_preflight_dict())
    warning = next(f for f in findings if f.code == "missing_opening_mirrors")
    assert warning.severity == "warning"
    # Warning-only: absence never blocks go-live.
    assert not any(
        f.code == "missing_opening_mirrors" for f in blocking_findings(findings)
    )


def test_lint_errors_on_malformed_mirror_and_hiring_company_in_canonical():
    from shared.brief_lint import blocking_findings, lint_generated_brief

    data = {**_valid_preflight_dict(), **_mirror_fields()}
    data["canonical_title_patterns"] = ["ok", ""]
    data["canonical_company_patterns"] = ["Acme Fintech Data Services"]
    findings = lint_generated_brief(data)
    codes = {f.code for f in blocking_findings(findings)}
    assert "opening_mirror_invalid" in codes
    assert "hiring_company_in_canonical_companies" in codes



def test_maximum_years_round_trips_and_lints():
    from shared.brief_lint import blocking_findings, lint_generated_brief
    from shared.brief_loader import _load_v2_brief
    from shared.preflight_v2 import preflight_to_brief_json

    data = {
        **_valid_preflight_dict(),
        "maximum_years_experience": 8,
        "experience_measure": "total career years since first full-time role",
        "maximum_years_experience_is_hard": False,
    }
    assert blocking_findings(lint_generated_brief(data)) == []

    # _load_v2_brief returns the compat wrapper; the judge consumes the
    # V2 schema object on _new_brief — assert on what production reads.
    brief = _load_v2_brief(preflight_to_brief_json(data))
    assert brief._new_brief.maximum_years_experience == 8
    assert (
        brief._new_brief.experience_measure
        == "total career years since first full-time role"
    )
    assert brief._new_brief.maximum_years_experience_is_hard is False

    hard_data = json.loads(json.dumps(data))
    hard_data["maximum_years_experience_is_hard"] = True
    hard_brief = _load_v2_brief(preflight_to_brief_json(hard_data))
    assert hard_brief._new_brief.maximum_years_experience_is_hard is True

    # A ceiling WITHOUT a measure is unenforceable — the judge otherwise
    # picks whichever tenure supports its gut call — so lint blocks it.
    unmeasured = {**_valid_preflight_dict(), "maximum_years_experience": 8}
    codes = {f.code for f in blocking_findings(lint_generated_brief(unmeasured))}
    assert "band_ceiling_without_measure" in codes

    # No ceiling → None, and both fields are genuinely optional.
    brief = _load_v2_brief(preflight_to_brief_json(_valid_preflight_dict()))
    assert brief._new_brief.maximum_years_experience is None
    assert brief._new_brief.experience_measure == ""
    assert brief._new_brief.maximum_years_experience_is_hard is False


def test_lint_rejects_invalid_experience_band():
    from shared.brief_lint import blocking_findings, lint_generated_brief

    inverted = {**_valid_preflight_dict(), "maximum_years_experience": 3}
    codes = {f.code for f in blocking_findings(lint_generated_brief(inverted))}
    assert "experience_band_invalid" in codes

    wrong_type = {**_valid_preflight_dict(), "maximum_years_experience": "ten"}
    codes = {f.code for f in blocking_findings(lint_generated_brief(wrong_type))}
    assert "experience_band_invalid" in codes


# ---------------------------------------------------------------------------
# Worked example_compounds + sequencing_heuristics (2026-07-04): a JD-only
# brief carries no worked example, so the strategy model composes generically.
# Preflight now authors two for THIS role; the prompt shows the shape without
# donating content; format_for_review surfaces them at the operator QA point.
# ---------------------------------------------------------------------------


def test_preflight_prompt_requests_example_compounds_and_sequencing():
    prompt = generate_preflight_prompt("JD text")

    assert '"example_compounds"' in prompt
    assert '"sequencing_heuristics"' in prompt
    # The model authors EXACTLY 2 and is forbidden from copying the shape demo.
    assert "EXACTLY 2" in prompt
    assert "do not copy" in prompt.lower()
    # The importable illustrative constant reaches the prompt: its first quoted
    # placeholder term survives the JSON escaping of the rendered shape.
    import re

    from shared.preflight_v2 import ILLUSTRATIVE_EXAMPLE_COMPOUNDS

    placeholder = re.findall(
        r'"([^"]*)"', ILLUSTRATIVE_EXAMPLE_COMPOUNDS[0]["boolean"]
    )[0]
    assert placeholder in prompt


def test_illustrative_example_compounds_are_lint_clean_and_shaped():
    """The importable constant is a real, lint-clean pair the copy-check keys
    off: one precision anchor (no AND gate), one multi-angle recall net."""
    from linkedin.boolean_compiler import lint_boolean

    from shared.preflight_v2 import ILLUSTRATIVE_EXAMPLE_COMPOUNDS

    assert len(ILLUSTRATIVE_EXAMPLE_COMPOUNDS) >= 1
    for ec in ILLUSTRATIVE_EXAMPLE_COMPOUNDS:
        assert set(ec) >= {"boolean", "purpose", "novelty_bucket"}
        assert not lint_boolean(ec["boolean"]).has_error, ec["boolean"]
    booleans = [ec["boolean"] for ec in ILLUSTRATIVE_EXAMPLE_COMPOUNDS]
    # One precision proof-of-practice (no AND gate), one recall net (AND + OR).
    assert any(" AND " not in b for b in booleans)
    assert any(" AND " in b and " OR " in b for b in booleans)


def _example_compounds_fixture() -> list[dict]:
    return [
        {
            "boolean": '("idempotency key" OR "ledger reconciliation")',
            "purpose": "Precision proof-of-practice on payments-builder vocabulary",
            "novelty_bucket": "canonical",
        },
        {
            "boolean": '("payment orchestration" OR "payment rails") AND ("settlement" OR "chargeback")',
            "purpose": "Multi-angle recall net, one concept per OR group",
            "novelty_bucket": "edge_case",
        },
    ]


def test_format_for_review_renders_example_compounds_section_when_present():
    data = _full_preflight_data()
    data["example_compounds"] = _example_compounds_fixture()
    data["sequencing_heuristics"] = "Open with the precision anchor, then the recall net."

    rendered = format_for_review(data)

    assert "EXAMPLE COMPOUNDS" in rendered
    assert "idempotency key" in rendered
    assert "Precision proof-of-practice on payments-builder vocabulary" in rendered
    assert "Open with the precision anchor, then the recall net." in rendered


def test_format_for_review_omits_example_compounds_section_when_absent():
    # _full_preflight_data carries no example_compounds — the section must not
    # render (clean omission), mirroring the other only-when-present sections.
    rendered = format_for_review(_full_preflight_data())
    assert "EXAMPLE COMPOUNDS" not in rendered
