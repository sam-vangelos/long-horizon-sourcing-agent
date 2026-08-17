"""Tests for :mod:`shared.intake_conversation.voice_contract` and the
extended voice assert helpers (Slices 4/5).
"""

from __future__ import annotations

import pytest

from shared.intake_conversation.prompts import (
    OPENER_WITH_PACKET,
    build_orchestrator_system_prompt,
)
from shared.intake_conversation.voice_contract import (
    looks_like_brief_dump,
    phrase_cooldown_violations,
    source_overlap_ratio,
)
from tests.intake_conversation.voice_asserts import (
    assert_no_brief_dump_shape,
    assert_phrase_cooldown,
    assert_source_question_not_redundant,
)


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content, "ts": "2026-05-18T12:00:00+00:00"}


def test_opener_with_packet_avoids_cooldown_phrases() -> None:
    lowered = OPENER_WITH_PACKET.lower()
    for phrase in (
        "one thing",
        "couple things",
        "before i draft",
        "before i start scoping",
    ):
        assert phrase not in lowered


def test_system_prompt_includes_phrase_cooldown_block() -> None:
    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "# PHRASE COOLDOWNS" in prompt
    assert "one thing" in prompt.lower()


def test_phrase_cooldown_violations_detects_early_reuse() -> None:
    transcript = [
        _msg("cloris", "One thing I want to confirm — the close."),
        _msg("recruiter", "Yeah."),
        _msg("cloris", "One thing about Avalara — in-house or vendor?"),
    ]
    violations = phrase_cooldown_violations(transcript)
    assert violations
    with pytest.raises(AssertionError):
        assert_phrase_cooldown(transcript)


def test_phrase_cooldown_passes_when_gap_respected() -> None:
    turns = [_msg("cloris", f"Turn {i}.") for i in range(5)]
    turns[0] = _msg("cloris", "One thing about the close — who owns it?")
    turns[4] = _msg("cloris", "One thing about geography — NYC only?")
    assert_phrase_cooldown(turns)


def test_looks_like_brief_dump_detects_markdown_headings() -> None:
    text = (
        "**Role title:** Senior Tax Associate\n"
        "**Role summary:** Owns sales tax.\n"
        "**Capability areas:** Sales tax compliance\n"
    )
    assert looks_like_brief_dump(text)
    with pytest.raises(AssertionError):
        assert_no_brief_dump_shape(text)


def test_looks_like_brief_dump_passes_on_normal_chat() -> None:
    text = "Got it — sales tax across the production entities. In-house?"
    assert not looks_like_brief_dump(text)
    assert_no_brief_dump_shape(text)


def test_source_overlap_ratio_flags_redundant_question() -> None:
    packet = {
        "job_description_text": (
            "Senior Tax Associate at Northwind. Avalara experience preferred. "
            "Owns multi-state sales tax compliance."
        )
    }
    question = "Does Avalara experience matter for this Avalara sales tax role?"
    ratio = source_overlap_ratio(question, packet)
    assert ratio > 0.5
    with pytest.raises(AssertionError):
        assert_source_question_not_redundant(question, packet, max_overlap=0.5)


def test_source_overlap_ratio_passes_on_specific_follow_up() -> None:
    packet = {
        "job_description_text": "Senior Tax Associate at Northwind. Avalara preferred."
    }
    question = (
        "You mentioned production entities — does the new hire own filings "
        "for all twelve LLCs from day one?"
    )
    assert source_overlap_ratio(question, packet) < 0.75
    assert_source_question_not_redundant(question, packet)
