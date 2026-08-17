"""Tests for vertical-agnostic calibration in the full-evaluation judgment template
(Slice 2, Commit 3 — judgment-template refactor).

These pins verify that ``linkedin/judgment_templates.py:FULL_EVALUATION_TEMPLATE``
no longer hardcodes ML/AI vocabulary, and that the four new placeholders
(``domain_verbs_block``, ``domain_depth_objects_block``,
``transferability_transfers_block``, ``transferability_does_not_transfer_block``)
render brief-driven content for populated briefs and capability-area-driven
fallback prose for empty briefs without producing orphan section headers,
dangling colons, or malformed transitions.

The pins also confirm the procedural skeleton (Step 1 / Step 2 / Step 3 /
Step 4) and the JSON / response-shape contract markers (DECISION, CONFIDENCE,
STEP_1_MATCH, ...) survive the refactor under both empty and populated inputs.
"""

from __future__ import annotations

from typing import Any

import pytest

from shared.brief_loader import _load_v2_brief
from linkedin.judgment_templates import assemble_full_evaluation_system


# ---------------------------------------------------------------------------
# Brief fixtures
# ---------------------------------------------------------------------------

# Strings that the pre-Slice-2 template hardcoded into the prompt. After
# the Commit 3 refactor, none of these may appear in the assembled
# system prompt — neither for empty briefs nor for non-AI verticals.
FORBIDDEN_HARDCODED_AI_STRINGS = (
    "PyTorch",
    "QLoRA",
    "LLM training data",
    "RL environments",
    "reward models",
    "non-LLM domain",
    "Fine-tuned is a BUILDER",
    "PhD + ML title",
)


def _minimal_v2_raw() -> dict[str, Any]:
    """Minimum viable V2 brief raw dict with no calibration fields populated."""
    return {
        "role_title": "Generic Role",
        "role_level": "L5",
        "role_summary": "Generic vertical-agnostic role for empty-calibration testing.",
        "geography": "New York",
        "linkedin_project": "Test Project",
        "capability_areas": [
            {
                "name": "Capability A",
                "description": "Owns end-to-end implementation of capability A.",
                "builder_signals": ["built capability A"],
                "user_signals": ["consumed capability A"],
                "key_terms": [],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns construction of capability A.",
            "user_definition": "Consumes capability A.",
            "edge_case_guidance": "Defer to evidence.",
        },
        "non_fit_patterns": [],
        "employer_signal_rules": [],
        "minimum_years_experience": 5,
        "minimum_bar_description": "Five years of capability-A ownership.",
        "facial_calibration": {
            "expected_yes_rate_low": 0.25,
            "expected_yes_rate_high": 0.55,
            "fast_exit_patterns": [],
            "trajectory_yes_patterns": [],
            "trajectory_ambiguous_patterns": [],
            "trajectory_no_patterns": [],
        },
    }


def _populated_ai_v2_raw() -> dict[str, Any]:
    """V2 brief raw dict populated with head-AI-style calibration vocabulary."""
    raw = _minimal_v2_raw()
    raw["role_title"] = "Head of Applied AI Lab"
    raw["domain_verbs"] = [
        "fine-tuned",
        "trained",
        "evaluated",
        "designed",
        "built",
    ]
    raw["domain_depth_objects"] = [
        "training pipelines",
        "evaluation harnesses",
        "data curation systems",
    ]
    raw["transferability_examples"] = [
        {
            "result": "transfers",
            "source_context": "evaluation framework design in any ML domain",
            "target_context": "evaluation framework design for LLMs",
            "rationale": "Methodology is the same; the model under evaluation changes.",
        },
        {
            "result": "does_not_transfer",
            "source_context": "classical CFD simulation without ML",
            "target_context": "frontier model training",
            "rationale": "Physics-based simulation does not port to learned methodology.",
        },
    ]
    return raw


def _populated_clinical_v2_raw() -> dict[str, Any]:
    """V2 brief raw dict for a non-AI clinical/healthtech vertical."""
    raw = _minimal_v2_raw()
    raw["role_title"] = "Clinical Decision Support Lead"
    raw["role_summary"] = "Owns clinical decision support tooling and EHR integrations."
    raw["capability_areas"] = [
        {
            "name": "Clinical Decision Support",
            "description": "Owns construction of CDS tooling for clinicians.",
            "builder_signals": ["built CDS workflow"],
            "user_signals": ["used CDS app"],
            "key_terms": [],
        }
    ]
    raw["domain_verbs"] = ["diagnosed", "treated", "implemented"]
    raw["domain_depth_objects"] = [
        "clinical decision support workflow",
        "EHR integration",
        "hospital triage protocol",
    ]
    raw["transferability_examples"] = [
        {
            "result": "transfers",
            "source_context": "hospital triage workflow design",
            "target_context": "remote patient monitoring workflow",
            "rationale": "Workflow orchestration and clinician-handoff judgment transfer.",
        },
        {
            "result": "does_not_transfer",
            "source_context": "consumer wellness app analytics",
            "target_context": "EHR-integrated clinical decision support",
            "rationale": "Clinical-grade data integration and clinician workflow are absent.",
        },
    ]
    return raw


def _system_prompt(raw: dict[str, Any]) -> str:
    return assemble_full_evaluation_system(_load_v2_brief(raw)._new_brief)


# ---------------------------------------------------------------------------
# Empty calibration — no leak of hardcoded ML/AI vocabulary
# ---------------------------------------------------------------------------


def test_empty_calibration_full_eval_system_drops_all_hardcoded_ml_strings():
    """An empty-calibration brief must NOT carry any pre-refactor ML/AI phrases."""
    prompt = _system_prompt(_minimal_v2_raw())
    for forbidden in FORBIDDEN_HARDCODED_AI_STRINGS:
        assert forbidden not in prompt, (
            f"Empty-calibration full-evaluation system prompt still leaks hardcoded "
            f"ML/AI string: {forbidden!r}"
        )


# ---------------------------------------------------------------------------
# Populated AI brief — brief vocabulary appears via helpers
# ---------------------------------------------------------------------------


def test_populated_ai_brief_renders_domain_verbs_through_helper():
    prompt = _system_prompt(_populated_ai_v2_raw())
    for verb in ("fine-tuned", "trained", "evaluated", "designed", "built"):
        assert verb in prompt, f"populated brief did not render domain verb {verb!r}"


def test_populated_ai_brief_renders_depth_objects_through_helper():
    prompt = _system_prompt(_populated_ai_v2_raw())
    for obj in ("training pipelines", "evaluation harnesses", "data curation systems"):
        assert obj in prompt, f"populated brief did not render depth object {obj!r}"


def test_populated_ai_brief_renders_transferability_examples_split_by_result():
    prompt = _system_prompt(_populated_ai_v2_raw())
    transfers_idx = prompt.find("TRANSFERS (methodology is domain-portable):")
    does_not_idx = prompt.find("DOES NOT TRANSFER (domain gap is too wide")
    assert transfers_idx != -1, "TRANSFERS header missing"
    assert does_not_idx != -1, "DOES NOT TRANSFER header missing"
    assert transfers_idx < does_not_idx

    transfers_section = prompt[transfers_idx:does_not_idx]
    does_not_section = prompt[does_not_idx:prompt.find("RESULT:", does_not_idx)]

    assert "evaluation framework design in any ML domain" in transfers_section
    assert "evaluation framework design for LLMs" in transfers_section
    assert "Methodology is the same" in transfers_section

    assert "classical CFD simulation without ML" in does_not_section
    assert "frontier model training" in does_not_section
    assert "Physics-based simulation does not port" in does_not_section

    # Cross-check: a transfers example must NOT bleed into the does-not section.
    assert "evaluation framework design in any ML domain" not in does_not_section
    # ...and a does-not example must NOT bleed into the transfers section.
    assert "classical CFD simulation without ML" not in transfers_section


# ---------------------------------------------------------------------------
# Non-AI vertical — clinical / healthtech vocabulary, no AI leaks
# ---------------------------------------------------------------------------


def test_non_ai_vertical_brief_reflects_clinical_vocabulary():
    prompt = _system_prompt(_populated_clinical_v2_raw())
    for term in (
        "diagnosed",
        "treated",
        "implemented",
        "clinical decision support workflow",
        "EHR integration",
        "hospital triage workflow design",
        "remote patient monitoring workflow",
    ):
        assert term in prompt, f"clinical vertical did not render brief term {term!r}"


def test_non_ai_vertical_brief_does_not_leak_ai_specific_strings():
    prompt = _system_prompt(_populated_clinical_v2_raw())
    for forbidden in FORBIDDEN_HARDCODED_AI_STRINGS:
        assert forbidden not in prompt, (
            f"Clinical-vertical full-evaluation system prompt unexpectedly leaks "
            f"AI-specific string: {forbidden!r}"
        )


# ---------------------------------------------------------------------------
# Empty-collapse — no orphan headers / dangling colons
# ---------------------------------------------------------------------------


def test_empty_calibration_collapse_has_no_orphan_section_markers():
    """An empty-calibration brief must produce a clean prompt:

    - no orphan colon at the end of the domain-builder-verbs bullet line
    - the domain-depth-objects header still has a meaningful bullet beneath it
    - the TRANSFERS / DOES NOT TRANSFER sections are non-empty (fallback bullet)
    - no doubled blank-line transitions where the brief blocks would have appeared
    """
    prompt = _system_prompt(_minimal_v2_raw())

    # The domain-builder-verbs line must have content after the colon.
    verbs_marker = "- Domain builder verbs (signal hands-on construction work in the brief's capability areas):"
    idx = prompt.find(verbs_marker)
    assert idx != -1, "domain-builder-verbs bullet header missing"
    after_colon = prompt[idx + len(verbs_marker):idx + len(verbs_marker) + 200]
    # Must not be just whitespace / newline immediately after colon.
    stripped = after_colon.lstrip(" ")
    assert stripped and not stripped.startswith("\n"), (
        "empty calibration leaves dangling colon on domain-builder-verbs line; "
        f"text after colon: {after_colon!r}"
    )

    # The domain-depth-objects header line must be followed by a non-empty
    # line (not by another section header or blank line).
    depth_header = "- Domain depth objects (artifacts whose creation requires capability-area expertise):"
    depth_idx = prompt.find(depth_header)
    assert depth_idx != -1, "domain-depth-objects bullet header missing"
    after_depth = prompt[depth_idx + len(depth_header):]
    next_lines = after_depth.split("\n", 4)
    # next_lines[0] is the rest of the header line (should be empty/whitespace),
    # next_lines[1] is the immediate next line — must be non-empty.
    assert len(next_lines) >= 2
    assert next_lines[1].strip(), (
        f"empty calibration leaves dangling depth-objects header; next line: {next_lines[1]!r}"
    )
    # The next line must not be the application-layer-objects bullet (i.e.,
    # the helper output / fallback must occupy at least one line).
    assert not next_lines[1].lstrip().startswith("- Application-layer objects"), (
        "empty calibration collapses depth-objects section into the next bullet"
    )

    # Both TRANSFERS and DOES NOT TRANSFER sections must produce at least one
    # bullet under their header, even when no examples exist in the brief.
    transfers_idx = prompt.find("TRANSFERS (methodology is domain-portable):")
    does_not_idx = prompt.find("DOES NOT TRANSFER (domain gap is too wide")
    assert transfers_idx != -1 and does_not_idx != -1
    transfers_section = prompt[transfers_idx:does_not_idx]
    does_not_section = prompt[does_not_idx:prompt.find("RESULT:", does_not_idx)]
    # Each section, after its header, must contain a "-" bullet on a non-empty line.
    assert any(
        line.lstrip().startswith("-")
        for line in transfers_section.split("\n")[1:]
        if line.strip()
    ), "empty TRANSFERS section has no fallback bullet"
    assert any(
        line.lstrip().startswith("-")
        for line in does_not_section.split("\n")[1:]
        if line.strip()
    ), "empty DOES NOT TRANSFER section has no fallback bullet"


# ---------------------------------------------------------------------------
# Procedural skeleton — Step 1-7 markers preserved (P1 evaluator redesign:
# level alignment, opportunity coherence, and caliber became first-class
# steps and DECISION moved to Step 7; plans/glm-evaluator-operating-philosophy.md §D)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_factory",
    [_minimal_v2_raw, _populated_ai_v2_raw, _populated_clinical_v2_raw],
    ids=["empty", "populated_ai", "populated_clinical"],
)
def test_procedural_skeleton_preserved(raw_factory):
    prompt = _system_prompt(raw_factory())
    for marker in (
        "STEP 1 — CAPABILITY MAPPING",
        "STEP 2 — DEPTH TEST",
        "STEP 3 — TRANSFERABILITY",
        "STEP 4 — LEVEL ALIGNMENT",
        "STEP 5 — OPPORTUNITY COHERENCE",
        "STEP 6 — CANDIDATE CALIBER",
        "STEP 7 — DECISION",
        "SPARSE PROFILE CHECK",
        "ENGAGEMENT CONTEXT",
    ):
        assert marker in prompt, f"procedural skeleton lost marker {marker!r}"


# ---------------------------------------------------------------------------
# JSON / response-shape contract markers preserved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_factory",
    [_minimal_v2_raw, _populated_ai_v2_raw, _populated_clinical_v2_raw],
    ids=["empty", "populated_ai", "populated_clinical"],
)
def test_response_contract_markers_preserved(raw_factory):
    """The fields the parser keys off (STEP_*_, DECISION, CONFIDENCE, ...) survive."""
    prompt = _system_prompt(raw_factory())
    for marker in (
        "STEP_1_MATCH:",
        "STEP_1_AREA:",
        "STEP_1_EVIDENCE:",
        "STEP_2_DEPTH:",
        "STEP_2_EVIDENCE:",
        "STEP_3_TRANSFERABILITY:",
        "STEP_3_EVIDENCE:",
        "CASE_FOR:",
        "CASE_AGAINST:",
        "DECISION:",
        "CONFIDENCE:",
        "POST_SAVE_MODIFIER:",
        "SUMMARY:",
    ):
        assert marker in prompt, f"response contract lost marker {marker!r}"


# ---------------------------------------------------------------------------
# P5.2 — trajectory-shape inference ungated from L7+
# ---------------------------------------------------------------------------
# ``Brief.seniority_calibration_block()`` used to return "" for any role
# that isn't ``is_senior_role()``. The Tier-3 reasoning behind it (a
# coherent career arc can be evidence even when no single position is) is
# role-agnostic, so IC/L4-L6 briefs must now get a role-agnostic
# trajectory-shape inference paragraph in the same template slot, while
# L7+ briefs must render byte-identically to before this change.

_EXPECTED_SENIOR_CALIBRATION_BLOCK_L7 = """
SENIORITY CALIBRATION (L7):
At the executive level (L7+), the evidence hierarchy shifts. Senior leaders describe work at a higher abstraction level — "pioneering new paradigms in the field" and "translating R&D into products delivering $10M+ impact" rather than naming the specific tools and pipelines they built. This is expected, not a deficiency.

For L7+ candidates, apply THREE TIERS of evidence:

Tier 1 — Direct Evidence: Profile explicitly names tools, systems, production metrics, action verbs with specificity. Standard evaluation. Strongest signal when present.

Tier 2 — Contextual Inference: Title + company + scope + budget + team size make it overwhelmingly likely the person has the capability. "Head of the target function at a major enterprise, built global R&D teams, $12-15M budget, $10M+ monthly economic impact" → this person almost certainly drove the architecture decisions and built production systems in the brief's capability areas, even though their bullets describe organizational achievements. For L7+ candidates, Tier 2 evidence is CO-PRIMARY with Tier 1 — not a fallback.

Tier 3 — Trajectory Inference: No single position provides evidence, but the career arc does. a rigorous technical education → hands-on senior IC work → executive leadership of the target function → a top-tier organization in the brief's domain. The trajectory tells you the person has deep technical foundations and has led the capability at enterprise scale. For L8-L9 candidates, Tier 3 can independently support a SAVE when the trajectory is unambiguous.

You MUST state which tier supports each of your conclusions. Do not reject a candidate for lacking Tier 1 evidence when Tier 2 or Tier 3 evidence is strong."""


def _senior_v2_raw() -> dict[str, Any]:
    raw = _minimal_v2_raw()
    raw["role_level"] = "L7"
    return raw


def test_ic_brief_seniority_calibration_block_gains_trajectory_shape_inference():
    """Direct unit pin on the brief method (not just the assembled prompt):
    an IC brief must no longer get an empty string from
    ``seniority_calibration_block()``.
    """
    brief = _load_v2_brief(_minimal_v2_raw())._new_brief
    block = brief.seniority_calibration_block()
    assert block != ""
    assert "TRAJECTORY-SHAPE INFERENCE:" in block
    # Executive-only vocabulary must not leak into the IC block.
    for exec_only in ("scope + budget + team size", "Tier 2", "Tier 3", "CO-PRIMARY", "board"):
        assert exec_only not in block, f"IC trajectory block leaked executive term {exec_only!r}"


def test_senior_brief_seniority_calibration_block_byte_identical():
    """P5.2 exit gate: L7+ output is unchanged by the ungating."""
    brief = _load_v2_brief(_senior_v2_raw())._new_brief
    assert brief.seniority_calibration_block() == _EXPECTED_SENIOR_CALIBRATION_BLOCK_L7


def test_ic_brief_full_eval_prompt_now_contains_trajectory_tier():
    """Template-level pin: the IC brief's assembled full-eval prompt contains
    the new trajectory-shape inference block in the
    ``{seniority_calibration_block}`` slot (previously empty for IC briefs).
    """
    prompt = _system_prompt(_minimal_v2_raw())
    assert "TRAJECTORY-SHAPE INFERENCE:" in prompt
    assert "SENIORITY CALIBRATION" not in prompt


def test_senior_brief_full_eval_prompt_unchanged_by_p5_2():
    """Template-level pin: the L7+ assembled full-eval prompt still contains
    the exact pre-P5.2 executive calibration block, and does NOT also pick
    up the new IC-only trajectory-shape inference paragraph.
    """
    prompt = _system_prompt(_senior_v2_raw())
    assert _EXPECTED_SENIOR_CALIBRATION_BLOCK_L7 in prompt
    assert "TRAJECTORY-SHAPE INFERENCE:" not in prompt
    assert prompt.count("SENIORITY CALIBRATION") == 1
