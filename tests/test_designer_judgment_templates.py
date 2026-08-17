"""Designer Slice 2 — text-based judgment templates.

Pins the contract for :mod:`designer.judgment_templates`'s real prompts:

- `assemble_designer_facial_system` returns a system prompt that
  carries the brief's role, capability areas, rubric vocab, non-fit
  patterns, and an explicit OUTPUT FORMAT.
- `assemble_designer_full_system` returns a system prompt that carries
  the same plus depth_distinction and an OUTPUT FORMAT requesting
  DECISION + PATH + CONFIDENCE + SUMMARY.
- Both prompts gracefully omit blocks the brief doesn't carry (a brief
  without `non_fit_patterns` should not surface an empty block).
- The recruiter-authored vocab from `behance_specialization_signals`
  + `tool_stack_signals` lands in the capability block so the LLM can
  reason against it.
- Slice-1 placeholder helpers remain for backward compat (returning
  the placeholder OpusDecision).

Voice / shape sanity:
- DECISION values match the existing facial / full vocab so the
  parser layer (Slice 5+) can rely on them.
- No engineer vocab leaks (no ``snake_case`` identifiers in the
  voice copy).
"""

from __future__ import annotations

import re

import pytest

from designer.judgment_templates import (
    PLACEHOLDER_RATIONALE,
    assemble_designer_facial_system,
    assemble_designer_full_system,
    designer_facial_judge_placeholder,
    designer_full_judge_placeholder,
)


def _full_designer_brief() -> dict:
    return {
        "role_title": "Senior product designer",
        "role_summary": "Owns design surface end-to-end for an enterprise SaaS.",
        "capability_areas": [
            {
                "name": "Design systems",
                "description": "Builds and maintains design systems.",
                "behance_specialization_signals": [
                    "design tokens",
                    "component library",
                ],
                "tool_stack_signals": ["Figma", "Storybook"],
            },
            {
                "name": "Information density",
                "description": "Designs dense data surfaces well.",
            },
        ],
        "depth_distinction": {
            "builder_definition": "Owns the system end-to-end including governance.",
            "user_definition": "Consumes the system without authoring it.",
            "edge_case_guidance": "Borderline = consuming + extending.",
        },
        "non_fit_patterns": [
            {
                "label": "marketing-only portfolio",
                "why_not": "no shipped product evidence",
            }
        ],
        "design_rubric": {
            "principles": [
                {
                    "name": "Visual hierarchy",
                    "description": "How clearly the work guides attention.",
                    "anchors": {
                        "bad": "no",
                        "okay": "weak",
                        "good": "clear",
                        "excellent": "purposeful",
                    },
                },
                {
                    "name": "Compositional balance",
                    "description": "Balance across the canvas.",
                    "anchors": {
                        "bad": "n",
                        "okay": "o",
                        "good": "g",
                        "excellent": "e",
                    },
                },
            ]
        },
    }


def _minimal_designer_brief() -> dict:
    return {
        "role_title": "Designer",
        "capability_areas": [{"name": "Surface design", "description": "Ships product."}],
        "depth_distinction": {
            "builder_definition": "Owns surface end-to-end.",
            "user_definition": "Iterates.",
            "edge_case_guidance": "Borderline.",
        },
    }


# ---------------------------------------------------------------------------
# Slice-2 facial system prompt
# ---------------------------------------------------------------------------


def test_facial_system_prompt_includes_role_and_capability_areas() -> None:
    prompt = assemble_designer_facial_system(_full_designer_brief())
    assert "Senior product designer" in prompt
    assert "Design systems" in prompt
    assert "Information density" in prompt


def test_facial_system_prompt_includes_specialization_and_tool_signals() -> None:
    prompt = assemble_designer_facial_system(_full_designer_brief())
    assert "design tokens" in prompt
    assert "component library" in prompt
    assert "Figma" in prompt
    assert "Storybook" in prompt


def test_facial_system_prompt_includes_non_fit_patterns() -> None:
    prompt = assemble_designer_facial_system(_full_designer_brief())
    assert "marketing-only portfolio" in prompt
    assert "no shipped product evidence" in prompt


def test_facial_system_prompt_includes_rubric_principle_names() -> None:
    prompt = assemble_designer_facial_system(_full_designer_brief())
    assert "Visual hierarchy" in prompt
    assert "Compositional balance" in prompt


def test_facial_system_prompt_declares_output_format() -> None:
    prompt = assemble_designer_facial_system(_full_designer_brief())
    assert "DECISION:" in prompt
    assert "FACIAL_YES" in prompt
    assert "FACIAL_NO" in prompt
    assert "FACIAL_BORDERLINE" in prompt
    assert "REASON:" in prompt


def test_facial_system_prompt_omits_empty_blocks_for_minimal_brief() -> None:
    prompt = assemble_designer_facial_system(_minimal_designer_brief())
    # No non-fit patterns block because the brief carries none.
    assert "Non-fit patterns" not in prompt
    # No rubric vocab block because the brief carries no rubric.
    assert "Design principles the recruiter weights" not in prompt
    # Capability area still surfaces.
    assert "Surface design" in prompt


# ---------------------------------------------------------------------------
# Slice-2 full system prompt
# ---------------------------------------------------------------------------


def test_full_system_prompt_includes_depth_block() -> None:
    prompt = assemble_designer_full_system(_full_designer_brief())
    assert "Owns the system end-to-end including governance" in prompt


def test_full_system_prompt_declares_summary_output_format() -> None:
    prompt = assemble_designer_full_system(_full_designer_brief())
    assert "DECISION:" in prompt
    assert "PATH:" in prompt
    assert "CONFIDENCE:" in prompt
    assert "SUMMARY:" in prompt
    # Decision vocab matches the existing wire contract.
    assert "SAVE" in prompt
    assert "REJECT" in prompt
    assert "INFERENTIAL_SAVE" in prompt


# ---------------------------------------------------------------------------
# Voice sanity — no snake_case engineer vocab
# ---------------------------------------------------------------------------


_SNAKE_CASE_RE = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
# Allowlist for snake_case tokens that ARE part of the contract — the
# OUTPUT FORMAT decision vocab uses uppercase tokens, the brief field
# names the prompt grounds on, and the voice rules reference them
# by name.
_ALLOWLIST = {
    "behance_specialization_signals",
    "tool_stack_signals",
    "design_rubric",
    "calibration_exemplars",
    "design_market",
}


@pytest.mark.parametrize(
    "assembler",
    [assemble_designer_facial_system, assemble_designer_full_system],
)
def test_voice_blocks_no_unexpected_snake_case(assembler) -> None:
    """The voice copy itself (not the contract field names) should not
    leak engineer-vocab snake_case tokens. Allowlist covers field
    names that are intentionally referenced."""

    prompt = assembler(_full_designer_brief())
    leaks = [
        match.group(0)
        for match in _SNAKE_CASE_RE.finditer(prompt)
        if match.group(0) not in _ALLOWLIST
    ]
    # Filter out anything we explicitly chose to mention from the brief
    # (e.g., the recruiter-authored vocab "design tokens" is fine —
    # it's not snake_case).
    assert leaks == [], (
        f"Voice copy leaked snake_case engineer vocab: {leaks!r}. "
        "If a token is intentionally referenced, add it to _ALLOWLIST."
    )


# ---------------------------------------------------------------------------
# Slice-1 placeholder helpers — backward compat
# ---------------------------------------------------------------------------


def test_placeholder_facial_helper_still_returns_placeholder_decision() -> None:
    decision = designer_facial_judge_placeholder(
        candidate_name="Test", profile_url="https://example.com"
    )
    assert decision.stage == "facial"
    assert decision.decision == "FACIAL_NO"
    assert decision.path == "placeholder"
    assert decision.confidence == 0.0
    assert decision.rationale == PLACEHOLDER_RATIONALE


def test_placeholder_full_helper_still_returns_placeholder_decision() -> None:
    decision = designer_full_judge_placeholder(
        candidate_name="Test", profile_url="https://example.com"
    )
    assert decision.stage == "full"
    assert decision.decision == "REJECT"
    assert decision.confidence == 0.0
    assert decision.rationale == PLACEHOLDER_RATIONALE
