"""Pre-emit question economy and brief-dump guards (intake Slice 3)."""

from __future__ import annotations

from shared.intake_conversation.conversation_guard import (
    BRIEF_DUMP_REPLACEMENT,
    detect_brief_dump_shape,
    guard_cloris_turn,
)
from shared.intake_conversation.question_economy import (
    apply_pre_emit_guards,
    arbiter_cloris_turn,
)
from shared.intake_conversation.insights import HIRING_MANAGER_PICTURE_KEY


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content, "ts": "2026-05-18T12:00:00+00:00"}


_BFS_PACKET = {
    "job_description_text": (
        "Head of Applied AI for the BFS group at a global bank. Sets AI vision "
        "and architectural guardrails for agentic systems, speaks to bank "
        "executives, and owns applied-AI delivery — not a pure research or "
        "platform hire. The winning person is a quasi-CTO who can still go "
        "deep on agentic design trade-offs."
    ),
    "intake_notes_text": "",
    "files": [],
}

_BFS_MESSAGES = [
    _msg(
        "recruiter",
        "We're hiring a Head of Applied AI for the BFS group — vision, "
        "guardrails, agentic systems, credible with bank executives.",
    ),
]


def test_brief_dump_shape_detects_schema_checklist() -> None:
    dump = """\
## Role
**Role title:** Head of Applied AI

## Capability areas
1. **Production agent systems** — owns agentic delivery
2. **Evaluation infrastructure** — decision-grade evals

## Non-fit patterns
- Pure research arc
- Wrapper-only applied AI

depth_distinction.builder_definition: hands-on substrate owner
source_strategy: linkedin primary
"""
    assert detect_brief_dump_shape(dump)
    result = guard_cloris_turn(dump)
    assert result.blocked
    assert result.replacement_text == BRIEF_DUMP_REPLACEMENT


def test_brief_dump_allows_conversational_summary() -> None:
    turn = (
        "Got it — applied-AI head for the BFS group, agentic guardrails, "
        "credible with bank executives. Is Avalara already in the stack, "
        "or is that still manual?"
    )
    assert not detect_brief_dump_shape(turn)
    assert not guard_cloris_turn(turn).blocked


def test_persona_confirmation_no_longer_rewrites_the_trigger_transcript() -> None:
    """P9.1 regression lock: the guard must never AUTHOR domain content.

    This is the exact trigger transcript that used to fire the
    persona-assumption rewrite (BFS packet + persona-shaped question).
    The rewrite path is deleted entirely — the guard may no longer
    invent a persona description ("a technical architect who... can
    speak to bank executives") and attribute it to the JD/transcript.
    The turn must pass through unmodified (``allow``), regardless of
    how strongly the source packet implies a persona scope.
    """
    bad = (
        "Reading the JD I'm picturing a quasi-CTO of the BFS group. "
        "Is the winning person more the boardroom program shaper, or the "
        "technical architect who can still sit with bank executives?"
    )
    verdict = arbiter_cloris_turn(
        bad,
        v2_draft={"role_title": "Head of Applied AI"},
        source_packet=_BFS_PACKET,
        messages=_BFS_MESSAGES,
        turn_count=2,
    )
    assert verdict.action == "allow"
    assert verdict.pattern is None
    assert verdict.replacement_text is None


def test_persona_confirmation_allowed_without_packet_signal() -> None:
    bad = (
        "Is the winning person more the boardroom program shaper, or the "
        "technical architect?"
    )
    verdict = arbiter_cloris_turn(
        bad,
        v2_draft={},
        source_packet=None,
        messages=[],
        turn_count=0,
    )
    assert verdict.action == "allow"


def test_non_fit_frequency_skipped_before_first_search() -> None:
    bad = (
        "Before we lock the brief — which non-fit pattern is most common in "
        "the pool you'd expect to see?"
    )
    verdict = arbiter_cloris_turn(
        bad,
        v2_draft={"role_title": "Tax Associate"},
        source_packet=None,
        messages=[],
        turn_count=1,
    )
    assert verdict.action == "skip"
    assert verdict.pattern == "non_fit_frequency"
    assert verdict.replacement_text is not None
    assert "most common" not in verdict.replacement_text.lower()


def test_where_to_look_skipped_when_manifest_can_infer() -> None:
    bad = (
        "One thing I still need — where should we look for this Head of "
        "Applied AI search?"
    )
    verdict = arbiter_cloris_turn(
        bad,
        v2_draft={"role_title": "Head of Applied AI", "role_summary": _BFS_PACKET["job_description_text"]},
        source_packet=_BFS_PACKET,
        messages=_BFS_MESSAGES,
        turn_count=2,
    )
    assert verdict.action == "skip"
    assert verdict.pattern == "where_to_look"
    assert verdict.replacement_text is not None
    assert "where should we look" not in verdict.replacement_text.lower()
    assert "LinkedIn" in verdict.replacement_text or "linkedin" in verdict.replacement_text.lower()


def test_apply_pre_emit_guards_brief_dump_wins_over_question_economy() -> None:
    text = """\
## Role
**Role title:** Head of Applied AI

## Capability areas
1. **Production agent systems** — owns agentic delivery
2. **Evaluation infrastructure** — decision-grade evals
3. **Research-engineering interface** — reads papers and ships

## Non-fit patterns
- Pure research arc
- Wrapper-only applied AI
- Manager-only for two years

depth_distinction.builder_definition: hands-on substrate owner
non_fit_patterns[0].label: Pure-research career arc
source_strategy: linkedin primary

Where should we look for this Head of Applied AI?
"""
    emitted, reasons = apply_pre_emit_guards(
        text,
        v2_draft={"role_title": "Head of Applied AI"},
        source_packet=_BFS_PACKET,
        messages=_BFS_MESSAGES,
        turn_count=1,
    )
    assert "brief_dump_shape" in reasons
    assert emitted == BRIEF_DUMP_REPLACEMENT


def test_apply_pre_emit_guards_does_not_author_persona_content_from_insights() -> None:
    """Even with a fully-formed hiring-manager-picture insight in hand,
    the guard must not rewrite a persona-shaped question into invented
    prose (P9.1) — the picture insight has no consumer left in
    question_economy after the persona rewrite path was deleted.
    """
    insights = {
        HIRING_MANAGER_PICTURE_KEY: {
            "summary": (
                "A quasi-CTO of the BFS group who sets AI vision and owns "
                "agentic architecture trade-offs with bank executives."
            ),
            "proof_points": ["Shipped applied AI inside a BFS firm."],
            "screening_translation": "Reject pure advisors without ownership.",
            "confidence": 0.8,
            "source": "combined",
        }
    }
    bad = (
        "Is the winning person more the boardroom program shaper, or the "
        "technical architect who can still sit with bank executives?"
    )
    emitted, reasons = apply_pre_emit_guards(
        bad,
        v2_draft={"role_title": "Head of Applied AI"},
        source_packet=_BFS_PACKET,
        messages=_BFS_MESSAGES,
        intake_insights=insights,
        turn_count=3,
    )
    assert "persona_confirmation" not in reasons
    assert emitted == bad


def test_system_prompt_includes_question_contract_and_brief_ban() -> None:
    from shared.intake_conversation.prompts import build_orchestrator_system_prompt

    prompt = build_orchestrator_system_prompt(
        sufficiency_state=(False, ["role_title"]),
        dropped_turn=False,
        cap_state="normal",
    )
    assert "BRIEF-IN-CHAT BAN" in prompt
    assert "QUESTION CONTRACT" in prompt
    assert "Show me the brief" in prompt
