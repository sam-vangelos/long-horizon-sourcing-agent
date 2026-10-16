"""Candidate-note rendering contracts for CLO-145."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import re

import pytest

from linkedin.candidate_note import (
    NOTE_PROBE_MAX_CHARS,
    build_candidate_note,
    note_probe_text,
)
from shared.schemas import OpusDecision, SearchString


NOW = datetime(2026, 8, 21, 9, 30)
SEARCH_STRING = SearchString(id=12, name="post-training", boolean="LLM")
RATIONALE = (
    "Senior GenAI data scientist with genuine, recent hands-on LLM fine-tuning "
    "(PEFT/QLoRA on Llama2/Falcon/Mistral) whose current work has drifted to agent "
    "orchestration — worth outreach under the coverage posture, but probe for owned "
    "eval/synthetic-data work and whether she still runs training loops."
)
CAPABILITY_EVIDENCE = (
    "At Dataverso (Mar 2023–Jan 2024) her own bullets state \"Fine-tuning de Large "
    "Language Models (Llama2, Falcon, Mistral) utilizando técnicas de eficiência de "
    "parâmetros (Peft, QLoRA, LoRA)\" alongside building the LLM-based product "
    "Chat2Dash. The current Dell role"
)


@pytest.fixture
def decision() -> OpusDecision:
    """Provide a genuine live SAVE-shaped judgment for every renderer test."""
    return OpusDecision(
        stage="full",
        decision="SAVE",
        path="direct_experience",
        confidence=0.91,
        rationale=RATIONALE,
        candidate_name="Helena D. Vieira",
        profile_url="/talent/profile/cinthia",
        outreach_tier="STANDARD",
        semantic_evidence={
            "capability_area": "Post-training execution",
            "depth": "BUILDER",
            "evidence_recency": "RECENT",
            "capability_evidence": CAPABILITY_EVIDENCE,
        },
    )


def test_full_render_matches_exact_note(decision: OpusDecision) -> None:
    """Evidence-bearing saves render the evaluation alone: rationale, evidence."""
    note = build_candidate_note(decision, SEARCH_STRING, now=NOW)

    assert note == f"{RATIONALE}\n\n{CAPABILITY_EVIDENCE}"


def test_note_carries_no_agent_metadata_or_labels(decision: OpusDecision) -> None:
    """Sam 2026-09-21: the note is the evaluation, with no authorship trail.

    Every metadata token the pre-2026-09-21 note carried is checked here even
    though the fixture sets all of them, because an operator reading the note
    must not be able to tell an agent wrote it.
    """
    note = build_candidate_note(decision, SEARCH_STRING, now=NOW)

    for forbidden in (
        "Cloris",
        "string #",
        "WHY",
        "EVIDENCE",
        "·",
        "Post-training execution",
        "BUILDER",
        "RECENT",
    ):
        assert forbidden not in note
    assert re.search(r"\d{4}-\d{2}-\d{2}", note) is None


def test_mentions_and_control_characters_are_sanitized(decision: OpusDecision) -> None:
    """No literal at-sign or control character may trigger the mentions editor."""
    decision = replace(
        decision,
        rationale="Research Scientist @ AKCIT\r\n\x00\n\nCurrent builder",
        semantic_evidence={
            **decision.semantic_evidence,
            "capability_evidence": "Owned evals @ Dataverso\rProof.",
        },
    )

    note = build_candidate_note(decision, SEARCH_STRING, now=NOW)

    assert "@" not in note
    assert "Research Scientist at AKCIT\n\nCurrent builder" in note
    assert "Owned evals at Dataverso\nProof." in note
    assert "\x00" not in note


@pytest.mark.parametrize(
    "semantic_evidence",
    [
        {},
        None,
        {
            "capability_area": "Post-training execution",
            "depth": "BUILDER",
            "evidence_recency": "RECENT",
            "capability_evidence": "  ",
        },
    ],
)
def test_missing_evidence_degrades_to_the_rationale_alone(
    decision: OpusDecision, semantic_evidence: dict | None
) -> None:
    """Missing semantic evidence leaves the rationale alone, without raising."""
    note = build_candidate_note(
        replace(decision, semantic_evidence=semantic_evidence),
        SEARCH_STRING,
        now=NOW,
    )

    assert note == RATIONALE


def test_empty_judgment_renders_nothing(decision: OpusDecision) -> None:
    """No rationale and no evidence is an empty note, never a bare label."""
    note = build_candidate_note(
        replace(decision, rationale="", semantic_evidence={}),
        SEARCH_STRING,
        now=NOW,
    )

    assert note == ""


def test_evidence_is_the_second_paragraph(decision: OpusDecision) -> None:
    """Evidence follows the rationale as its own paragraph, unlabelled."""
    decision = replace(
        decision,
        rationale="Owned the eval harness.",
        semantic_evidence={
            "capability_area": "Post-training execution",
            "depth": "",
            "evidence_recency": "RECENT",
            "capability_evidence": "Built and operated training loops.",
        },
    )

    note = build_candidate_note(decision, SEARCH_STRING, now=NOW)

    assert note == "Owned the eval harness.\n\nBuilt and operated training loops."


def test_evidence_only_judgment_renders_evidence_alone(
    decision: OpusDecision,
) -> None:
    """A rationale-free judgment must not open with a blank paragraph."""
    note = build_candidate_note(
        replace(
            decision,
            rationale="",
            semantic_evidence={"capability_evidence": "Ran the training loop."},
        ),
        SEARCH_STRING,
        now=NOW,
    )

    assert note == "Ran the training loop."


def test_evidence_truncates_at_a_sentence_boundary(decision: OpusDecision) -> None:
    """Long evidence must prefer a late sentence boundary and append an ellipsis."""
    decision = replace(
        decision,
        semantic_evidence={
            **decision.semantic_evidence,
            "capability_evidence": (
                "First sentence has enough detail. Second sentence also fits. "
                "Third sentence runs beyond the limit by a lot."
            ),
        },
    )

    note = build_candidate_note(
        decision, SEARCH_STRING, now=NOW, evidence_max_chars=70
    )

    assert note.endswith(
        "First sentence has enough detail. Second sentence also fits.…"
    )


def test_overall_cap_shrinks_evidence_before_the_rationale(
    decision: OpusDecision,
) -> None:
    """The hard cap preserves the whole rationale before shrinking evidence."""
    decision = replace(
        decision,
        rationale="Keep this complete rationale.",
        semantic_evidence={
            **decision.semantic_evidence,
            "capability_evidence": "Useful evidence sentence. " * 20,
        },
    )

    note = build_candidate_note(decision, SEARCH_STRING, now=NOW, max_chars=150)

    assert len(note) <= 150
    assert note.startswith("Keep this complete rationale.\n\n")
    assert note.endswith("…")


def test_overall_cap_truncates_the_rationale_at_a_word_boundary(
    decision: OpusDecision,
) -> None:
    """An oversized rationale respects the hard cap at a word boundary."""
    decision = replace(
        decision,
        rationale="Owned model training and evaluation systems across several teams. " * 8,
        semantic_evidence={},
    )

    note = build_candidate_note(decision, SEARCH_STRING, now=NOW, max_chars=100)

    assert len(note) <= 100
    assert note.startswith("Owned model training and evaluation systems")
    assert note.endswith("…")


@pytest.mark.parametrize("verdict", ["SAVE", "TRANSFERABLE_SAVE"])
def test_note_omits_verdict_tier_and_confidence_tokens(
    decision: OpusDecision, verdict: str
) -> None:
    """Saved-candidate presence must carry the verdict without metadata tags."""
    note = build_candidate_note(
        replace(decision, decision=verdict), SEARCH_STRING, now=NOW
    )

    for forbidden in ("SAVE", "TRANSFERABLE_SAVE", "STANDARD", "confidence"):
        assert forbidden not in note


def test_probe_is_the_notes_own_bounded_first_line(decision: OpusDecision) -> None:
    """The duplicate-detection probe is a prefix of the note's first line."""
    note = build_candidate_note(decision, SEARCH_STRING, now=NOW)
    probe = note_probe_text(note)

    assert probe == note.splitlines()[0][:NOTE_PROBE_MAX_CHARS]
    assert len(probe) == NOTE_PROBE_MAX_CHARS
    assert probe in note


def test_probe_normalizes_whitespace_so_dom_wrapping_still_matches() -> None:
    """Rendered notes wrap; the probe is single-spaced so a match survives it."""
    assert note_probe_text("  Owned   the\teval\nharness.  ") == "Owned the eval"


def test_probe_of_an_empty_note_is_empty_and_never_raises() -> None:
    """An empty probe must never be spent as 'this note is already present'."""
    for value in ("", "   ", "\n\n", None):
        assert note_probe_text(value) == ""
