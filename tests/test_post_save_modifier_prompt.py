"""The full-judgment prompt names exactly the post-save modifiers the validator accepts.

The validator (``validate_full_evaluation_semantics``) accepts ``NONE`` plus
each modifier name the brief defines, and nothing else; any other value turns
the whole verdict into PARSE_FAILURE (``invalid_post_save_modifier``). The
prompt's POST_SAVE_MODIFIER output-format line used to say only "name of
modifier that fired", so a brief with no modifiers left the judge free to coin
one (live: ``LLM_OR_EVALUATION_WORK`` on a PRIORITY save, field-engineering
2026-09-25). These tests render the real prompt through the production
assembly path and hold the line's enumeration equal to the validator's set.
"""

from __future__ import annotations

from typing import Any

import pytest

from shared.brief_loader import _load_v2_brief
from shared.judger import _full_tool_validation_inputs
from shared.judgment.templates import (
    assemble_full_evaluation_system,
    assemble_full_evaluation_tool_system,
    parse_full_evaluation_response,
)

_ALPHA = "Client-Facing / Forward-Deployed Delivery Experience"
_BETA = "Open-Source Maintainer Track Record"


def _raw_brief(modifiers: list[dict[str, Any]]) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "role_title": "Generic Role",
        "role_level": "L5",
        "role_summary": "Generic role for post-save-modifier prompt tests.",
        "geography": "New York",
        "linkedin_project": "Test Project",
        "engagement_context": {"selectivity_posture": "selective"},
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
    if modifiers:
        raw["post_save_modifiers"] = modifiers
    return raw


def _two_modifiers() -> list[dict[str, Any]]:
    return [
        {
            "name": _ALPHA,
            "trigger": "Candidate has shipped work directly with customers.",
            "if_present": "Raise confidence by 0.05.",
            "if_absent": "No adjustment.",
            "signals": ["forward-deployed engineer"],
        },
        {
            "name": _BETA,
            "trigger": "Candidate maintains a widely used open-source project.",
            "if_present": "Raise confidence by 0.05.",
            "if_absent": "No adjustment.",
        },
    ]


_CASES = {
    "zero_modifiers": [],
    "two_modifiers": _two_modifiers(),
}


def _loaded(case: str):
    return _load_v2_brief(_raw_brief(_CASES[case]))


def _format_line(prompt: str) -> str:
    """The single POST_SAVE_MODIFIER line in the RESPOND WITH format section."""
    tail = prompt.split("RESPOND WITH EXACTLY THIS FORMAT:", 1)[1]
    lines = [
        line for line in tail.splitlines() if line.startswith("POST_SAVE_MODIFIER:")
    ]
    assert len(lines) == 1, lines
    return lines[0]


def _enumerated_values(line: str) -> list[str]:
    """Read the allowed values the format line names, in order."""
    value = line.split(":", 1)[1].strip()
    assert value.startswith("[") and value.endswith("]"), line
    head = value[1:-1].split(" — ", 1)[0]
    if head.startswith("exactly one of "):
        return [item.strip() for item in head[len("exactly one of "):].split(" | ")]
    assert head == "exactly NONE", line
    return ["NONE"]


def _validator_allowed(loaded) -> set[str]:
    _, modifier_names = _full_tool_validation_inputs(loaded)
    return {*modifier_names, "NONE"}


@pytest.mark.parametrize("assemble", [
    assemble_full_evaluation_system,
    assemble_full_evaluation_tool_system,
], ids=["text", "tool"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_format_line_enumerates_exactly_the_validator_set(case, assemble):
    loaded = _loaded(case)
    line = _format_line(assemble(loaded._new_brief))
    values = _enumerated_values(line)
    assert len(values) == len(set(values)), line
    assert set(values) == _validator_allowed(loaded), line


def test_two_modifier_line_names_both_in_brief_order():
    line = _format_line(assemble_full_evaluation_system(_loaded("two_modifiers")._new_brief))
    assert _enumerated_values(line) == ["NONE", _ALPHA, _BETA]


def test_zero_modifier_line_leaves_no_opening_for_a_free_form_name():
    line = _format_line(assemble_full_evaluation_system(_loaded("zero_modifiers")._new_brief))
    assert _enumerated_values(line) == ["NONE"]
    lowered = line.lower()
    assert "only allowed value" in lowered
    assert "no post-save modifiers" in lowered
    # The pre-fix wording invited a coined name; nothing of it may survive.
    assert "name of modifier" not in lowered
    assert "modifier that fired" not in lowered


@pytest.mark.parametrize("case", sorted(_CASES))
def test_every_enumerated_value_survives_the_validator_on_a_save(case):
    """Round trip: a judge that obeys the line never trips invalid_post_save_modifier."""
    loaded = _loaded(case)
    line = _format_line(assemble_full_evaluation_system(loaded._new_brief))
    capability_areas, modifier_names = _full_tool_validation_inputs(loaded)
    for value in _enumerated_values(line):
        raw = "\n".join([
            "STEP_1_MATCH: DIRECT",
            "STEP_1_AREA: Capability A",
            "STEP_1_EVIDENCE: Built capability A end to end at Acme.",
            "STEP_2_DEPTH: BUILDER",
            "STEP_2_EVIDENCE: Owned the design and build.",
            "STEP_3_TRANSFERABILITY: N/A",
            "STEP_3_EVIDENCE: N/A",
            "STEP_1_RECENCY: CURRENT",
            "STEP_4_LEVEL: ALIGNED",
            "STEP_4_EVIDENCE: Owns the system.",
            "STEP_5_COHERENCE: COHERENT",
            "STEP_5_DRIVER: N/A",
            "STEP_6_CALIBER: STRONG",
            "STEP_6_EVIDENCE: Shipped A; scaled A.",
            "STEP_7_NEAR_MISS: NONE",
            "CASE_FOR: Direct, current, owned work.",
            "CASE_AGAINST: Nothing material.",
            "DECISION: SAVE",
            "CONFIDENCE: 0.8",
            "REJECT_REASON: NONE",
            "OUTREACH_TIER: PRIORITY",
            f"POST_SAVE_MODIFIER: {value}",
            "SUMMARY: Strong direct fit.",
        ])
        result = parse_full_evaluation_response(
            raw,
            require_semantic_v2=True,
            capability_areas=capability_areas,
            post_save_modifiers=modifier_names,
        )
        assert result.decision == "SAVE", (value, result.summary)
        assert result.post_save_modifier == value
