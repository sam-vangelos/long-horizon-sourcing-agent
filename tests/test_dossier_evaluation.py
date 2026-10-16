"""Tests for Executive Search Slice 2 — dossier-depth evaluation pipeline.

Pins the contract:

- `Brief.dossier_mode` derives from ``"exec_search" in target_modules``.
- `assemble_full_evaluation_system` swaps the trailing
  ``SUMMARY:`` line for a two-paragraph ``DOSSIER_RATIONALE:`` block
  when ``dossier_mode``.
- Non-dossier briefs produce byte-identical prompt output to the
  legacy format (characterization regression — guards against the
  branch leaking into senior-but-non-exec briefs).
- `parse_full_evaluation_response` reads ``DOSSIER_RATIONALE:`` as
  a multi-paragraph block and writes it into
  ``FullEvaluationResult.summary`` so the wire contract
  (``full_decision.rationale`` is a single string) is preserved.
- `linkedin/strategy.py` strategy prompt appends an exec-search
  ``title_first`` bias when ``target_modules`` carries
  ``"exec_search"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linkedin.judgment_templates import (
    _extract_field,
    _extract_multiline_field,
    assemble_full_evaluation_system,
    parse_full_evaluation_response,
)
from shared.brief_loader import load_brief


def _exec_brief_payload(role_title: str = "VP Engineering") -> dict:
    return {
        "role_title": role_title,
        "role_level": "Executive",
        "role_summary": "Owns engineering leadership for a series-D company.",
        "geography": "United States",
        "minimum_years_experience": 12,
        "minimum_bar_description": "12+ years engineering leadership.",
        "linkedin_project": "exec",
        "capability_areas": [
            {
                "name": "Engineering org leadership",
                "description": "Builds and runs 50+ person engineering orgs.",
                "builder_signals": ["VP-level scope", "headcount growth"],
                "user_signals": ["IC-level work primarily"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns engineering strategy + delivery.",
            "user_definition": "Manages individual teams without org-wide scope.",
            "edge_case_guidance": "Borderline = full eval.",
        },
        "target_modules": ["linkedin", "exec_search"],
        "confidentiality_class": "open",
    }


def _classic_brief_payload(role_title: str = "Senior Forward Deployed Engineer") -> dict:
    """A senior-but-non-exec brief — must NOT trigger dossier_mode."""

    return {
        "role_title": role_title,
        "role_level": "IC6",
        "role_summary": "Builds bespoke customer integrations on top of frontier APIs.",
        "geography": "United States",
        "minimum_years_experience": 7,
        "minimum_bar_description": "7+ years FDE-shaped delivery work.",
        "linkedin_project": "fde",
        "capability_areas": [
            {
                "name": "Forward-deployed delivery",
                "description": "Ships customer integrations end-to-end.",
                "builder_signals": ["bespoke integrations", "customer-facing"],
                "user_signals": ["pure platform work"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns customer outcome end-to-end.",
            "user_definition": "Delivers a feature for the platform team to ship.",
            "edge_case_guidance": "Borderline = full eval.",
        },
        "target_modules": ["linkedin"],
    }


def _write(tmp_path: Path, payload: dict, name: str = "brief.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# Brief.dossier_mode
# ---------------------------------------------------------------------------


def test_dossier_mode_true_when_exec_search_in_target_modules(tmp_path: Path) -> None:
    brief = load_brief(_write(tmp_path, _exec_brief_payload()))
    assert brief._new_brief.dossier_mode is True
    assert brief.target_modules == ["linkedin", "exec_search"]


def test_dossier_mode_false_for_classic_linkedin_brief(tmp_path: Path) -> None:
    brief = load_brief(_write(tmp_path, _classic_brief_payload()))
    assert brief._new_brief.dossier_mode is False
    assert brief.target_modules == ["linkedin"]


def test_dossier_mode_false_when_target_modules_omitted(tmp_path: Path) -> None:
    payload = _classic_brief_payload()
    del payload["target_modules"]
    brief = load_brief(_write(tmp_path, payload))
    assert brief._new_brief.dossier_mode is False
    assert brief.target_modules == []


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_full_eval_prompt_for_classic_brief_uses_summary_tail(
    tmp_path: Path,
) -> None:
    brief = load_brief(_write(tmp_path, _classic_brief_payload()))
    prompt = assemble_full_evaluation_system(brief._new_brief)

    assert "SUMMARY: [one-line evaluation a hiring manager could act on]" in prompt
    assert "DOSSIER_RATIONALE:" not in prompt


def test_full_eval_prompt_for_exec_brief_uses_dossier_rationale_tail(
    tmp_path: Path,
) -> None:
    brief = load_brief(_write(tmp_path, _exec_brief_payload()))
    prompt = assemble_full_evaluation_system(brief._new_brief)

    assert "DOSSIER_RATIONALE:" in prompt
    assert "TWO PARAGRAPHS" in prompt
    assert "Paragraph 1" in prompt
    assert "Paragraph 2" in prompt
    # Legacy SUMMARY tail must not also appear — the swap is a swap,
    # not an append.
    assert (
        "SUMMARY: [one-line evaluation a hiring manager could act on]"
        not in prompt
    )


def test_classic_full_eval_prompt_byte_identical_against_legacy(
    tmp_path: Path,
) -> None:
    """Characterization regression: a senior-but-non-exec brief produces
    the same prompt as before Slice 2 (modulo the trailing tail being
    structured via the new placeholder).

    Specifically, the prompt must end with the legacy
    ``SUMMARY: [...]`` line and contain none of the dossier-mode
    instruction tokens. Guards against `dossier_mode` accidentally
    firing for non-exec senior briefs.
    """

    brief = load_brief(_write(tmp_path, _classic_brief_payload()))
    prompt = assemble_full_evaluation_system(brief._new_brief)

    assert prompt.rstrip().endswith(
        "SUMMARY: [one-line evaluation a hiring manager could act on]"
    )
    for forbidden in ("DOSSIER_RATIONALE:", "TWO PARAGRAPHS", "Paragraph 1"):
        assert forbidden not in prompt


# ---------------------------------------------------------------------------
# Parser — DOSSIER_RATIONALE multi-paragraph
# ---------------------------------------------------------------------------


_DOSSIER_RAW_RESPONSE = """STEP_1_MATCH: DIRECT
STEP_1_AREA: Engineering org leadership
STEP_1_EVIDENCE: Currently SVP at a 800-person org.

STEP_2_DEPTH: BUILDER
STEP_2_EVIDENCE: Owns the engineering org headcount-and-roadmap.

STEP_3_TRANSFERABILITY: N/A
STEP_3_EVIDENCE: N/A

CASE_FOR: Has scaled an org through Series C → D.
CASE_AGAINST: Public-company background only.

DECISION: SAVE
CONFIDENCE: 0.85
POST_SAVE_MODIFIER: NONE
DOSSIER_RATIONALE: Operator-builder profile with 8 years scaling engineering at high-growth series-C/D companies; fits the brief's depth_distinction cleanly because she's owned both architecture and headcount under public-market scrutiny. The trajectory shows two clear inflection points where she rebuilt the org under stress, which is exactly the kind of board-pressure operating posture the search targets.

Scope evidence: led a 400-person engineering org through a $400M ARR run-rate, with two acquisitions under her belt and direct exposure to the audit committee. Adjacency to the client's leadership: shared three years at AcmeCorp with the client's current CTO and was on the diligence team for the BetaInc spin-out. She does not have direct PE/Buyout-stage experience, which is the load-bearing gap the recruiter should pre-frame with the client."""


def test_parser_reads_dossier_rationale_into_summary() -> None:
    result = parse_full_evaluation_response(_DOSSIER_RAW_RESPONSE)
    assert result.decision == "SAVE"
    assert result.confidence == pytest.approx(0.85)
    # Wire contract: summary carries the dossier prose (single string,
    # but multi-paragraph). Downstream surfaces render with
    # `white-space: pre-wrap` so the blank line between paragraphs
    # survives.
    assert "Operator-builder profile" in result.summary
    assert "Scope evidence" in result.summary
    # Paragraph break preserved.
    assert "\n\n" in result.summary


def test_parser_reads_legacy_summary_when_no_dossier_present() -> None:
    """Legacy single-line SUMMARY parses unchanged."""

    raw = """STEP_1_MATCH: DIRECT
STEP_1_AREA: x
STEP_1_EVIDENCE: y

STEP_2_DEPTH: BUILDER
STEP_2_EVIDENCE: y

STEP_3_TRANSFERABILITY: N/A
STEP_3_EVIDENCE: N/A

CASE_FOR: y
CASE_AGAINST: y

DECISION: SAVE
CONFIDENCE: 0.7
POST_SAVE_MODIFIER: NONE
SUMMARY: Strong builder profile a hiring manager can move on."""
    result = parse_full_evaluation_response(raw)
    assert result.summary == "Strong builder profile a hiring manager can move on."
    # No \n\n in a single-line summary.
    assert "\n\n" not in result.summary


def test_parser_prefers_dossier_when_both_blocks_emitted() -> None:
    """Transitional safeguard: if the LLM emits both, dossier wins."""

    raw = (
        _DOSSIER_RAW_RESPONSE
        + "\n\nSUMMARY: This single line should be ignored when DOSSIER_RATIONALE present."
    )
    result = parse_full_evaluation_response(raw)
    assert "Operator-builder profile" in result.summary
    assert "ignored when DOSSIER_RATIONALE" not in result.summary


def test_extract_field_terminates_at_dossier_rationale() -> None:
    """`_KNOWN_FIELDS` must include DOSSIER_RATIONALE so single-line
    extractors don't bleed into the dossier prose."""

    raw = (
        "POST_SAVE_MODIFIER: NONE\n"
        "DOSSIER_RATIONALE: paragraph 1.\n\nparagraph 2."
    )
    psm = _extract_field(raw, "POST_SAVE_MODIFIER:")
    assert psm == "NONE"
    # Multiline extractor handles the dossier block.
    dossier = _extract_multiline_field(raw, "DOSSIER_RATIONALE:")
    assert dossier.startswith("paragraph 1.")
    assert "paragraph 2." in dossier
    assert "\n\n" in dossier


# ---------------------------------------------------------------------------
# Strategy bias (linkedin/strategy.py)
# ---------------------------------------------------------------------------


def test_strategy_prompt_biases_title_first_for_exec_search(
    tmp_path: Path,
) -> None:
    """Architecture is telemetry now (generality hardening item 2): the
    exec_search title_first bias note is gone — exec_search is sunset and
    architecture no longer prescribes composition. title_first survives only
    as one of the Stage-0 classification labels."""

    from linkedin.strategy import _build_strategy_system

    brief = load_brief(_write(tmp_path, _exec_brief_payload()))
    prompt = _build_strategy_system(brief, has_kit=True)

    assert "title_first" in prompt
    assert "executive-search brief" not in prompt
    assert "Strongly prefer **title_first**" not in prompt


def test_strategy_prompt_omits_exec_bias_for_classic_brief(
    tmp_path: Path,
) -> None:
    from linkedin.strategy import _build_strategy_system

    brief = load_brief(_write(tmp_path, _classic_brief_payload()))
    prompt = _build_strategy_system(brief, has_kit=True)

    assert "executive-search brief" not in prompt
    assert "Strongly prefer **title_first**" not in prompt
