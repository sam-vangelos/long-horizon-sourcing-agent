"""Non-deferrable requirements: a brief-named requirement is never the inferential save's open question.

The SPL competitor run (req 1238) saved 35 AI-data operators with no basis to
infer they read code. Sam ruled 2026-09-25 that a missing code basis is a
REJECT, not a phone-screen question, yet the brief's coverage posture told the
judge "doubt that leaves exactly one conversation-resolvable question goes
through INFERENTIAL_SAVE", and the DECIDE step says the inferential channel
"exists for exactly this shape". The judge followed the template: Alan S.'s
record quotes the rule back.

``engagement_context.non_deferrable_requirements`` names the requirements the
inferential channel may never carry. It renders in the two places the template
grants that deferral, the posture line and the DECIDE step, and renders
nothing at all for a brief without it.
"""

from __future__ import annotations

import pytest

from shared.brief_v2_schema import BriefSchemaError, validate_v2_brief
from shared.judgment.templates import (
    assemble_full_evaluation_system,
    assemble_full_evaluation_tool_system,
)
from tests.test_prompt_characterization import _build_brief


CODE_BASIS = (
    "a basis to infer the person reads code: a CS or engineering degree, a past "
    "software or technical job, or a quantitative PhD"
)

_DECIDE_SEAM = (
    "the outreach conversation IS the follow-up you would otherwise have written "
    "down. When that current direct fit is absent"
)
_COVERAGE_LINE = (
    "Selectivity posture: coverage — this search prioritizes coverage; any "
    "candidate with a genuine, current, level-plausible case is worth surfacing; "
    "doubt that leaves exactly one conversation-resolvable question goes through "
    "INFERENTIAL_SAVE, and the rest resolves to REJECT."
)


def _brief(posture: str = "coverage", requirements=None):
    brief = _build_brief()
    context = {"hiring_company": "Example Org", "selectivity_posture": posture}
    if requirements is not None:
        context["non_deferrable_requirements"] = requirements
    brief.engagement_context = context
    return brief


def test_absent_field_leaves_the_prompt_untouched() -> None:
    system = assemble_full_evaluation_system(_brief())

    assert _DECIDE_SEAM in system
    assert _COVERAGE_LINE in system
    assert "NON-DEFERRABLE" not in system


def test_empty_list_renders_nothing() -> None:
    assert assemble_full_evaluation_system(_brief(requirements=[])) == (
        assemble_full_evaluation_system(_brief())
    )


def test_named_requirement_closes_the_posture_line_deferral() -> None:
    system = assemble_full_evaluation_system(_brief(requirements=[CODE_BASIS]))

    context = system.split("ENGAGEMENT CONTEXT:", 1)[1].split("\n\n", 1)[0]
    assert "NON-DEFERRABLE REQUIREMENTS" in context
    assert CODE_BASIS in context
    assert "never the one open question an INFERENTIAL_SAVE carries" in context


def test_named_requirement_closes_the_decide_step_deferral() -> None:
    system = assemble_full_evaluation_system(_brief(requirements=[CODE_BASIS]))

    decide = system.split("DECIDE — you are the reviewer", 1)[1].split("═══", 1)[0]
    assert "NON-DEFERRABLE" in decide
    assert "REJECT (reason CAPABILITY_INSUFFICIENT), never an INFERENTIAL_SAVE" in decide
    assert "sparse-profile shortcut" in decide


@pytest.mark.parametrize("posture", ["coverage", "selective"])
def test_the_rule_holds_under_either_posture(posture: str) -> None:
    system = assemble_full_evaluation_system(
        _brief(posture=posture, requirements=[CODE_BASIS])
    )
    assert system.count(CODE_BASIS) == 1
    assert "NON-DEFERRABLE" in system.split("DECIDE — you are the reviewer", 1)[1]


def test_tool_contract_prompt_carries_the_rule_too() -> None:
    system = assemble_full_evaluation_tool_system(_brief(requirements=[CODE_BASIS]))
    assert CODE_BASIS in system
    assert "never an INFERENTIAL_SAVE" in system


def _v2(**context) -> dict:
    return {
        "role_title": "Role",
        "capability_areas": [{"name": "A", "description": "a"}],
        "depth_distinction": {
            "builder_definition": "b",
            "user_definition": "u",
            "edge_case_guidance": "e",
        },
        "engagement_context": {"selectivity_posture": "coverage", **context},
    }


@pytest.mark.parametrize("value", ["code basis", [""], [3], {"a": "b"}])
def test_schema_refuses_a_malformed_requirement_list(value) -> None:
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(_v2(non_deferrable_requirements=value))
    assert "engagement_context.non_deferrable_requirements" in str(excinfo.value)


def test_schema_accepts_a_list_of_requirements() -> None:
    validate_v2_brief(_v2(non_deferrable_requirements=[CODE_BASIS]))
