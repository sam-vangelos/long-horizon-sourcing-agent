"""Tests for the voice-property helpers (Phase C9).

The helpers in :mod:`tests.intake_conversation.voice_asserts` are
imported by the C10 transcript tests and any future intake-voice
regression check. This file pins their behavior:

- Each per-response helper passes on a known-good Cloris turn.
- Each per-response helper raises on a known-bad turn covering its
  specific failure mode.
- Per-conversation helpers walk the full transcript and only inspect
  Cloris turns (recruiter content is ignored).
- The lexical banlist is sourced from the same constant the
  orchestrator system prompt enumerates — change-detection tests
  catch divergence.

Also pins the rewritten C9 system prompt to the structural contract
the C2 scaffold tests asserted (sections present, voice guide cited,
TODO C9 markers gone — voice content has landed).
"""

from __future__ import annotations

import pytest

from shared.intake_conversation.prompts import (
    LEXICAL_TIC_BANLIST,
    LEXICAL_TIC_VERB_BANLIST,
    OPENER_NO_PACKET,
    OPENER_WITH_PACKET,
    SUFFICIENCY_VOLUNTEER_TEMPLATE,
    SOFT_CAP_REMINDER,
    HARD_CAP_FORCE,
    build_orchestrator_system_prompt,
)
from tests.intake_conversation.voice_asserts import (
    assert_back_reference,
    assert_no_bullets,
    assert_no_lexical_tics,
    assert_no_register_slip,
    assert_no_rhetorical_question_transitions,
    assert_no_throat_clearing_openers,
    assert_sentence_count,
    assert_spaced_em_dashes_only,
)


def _msg(role: str, content: str) -> dict:
    return {
        "role": role,
        "content": content,
        "ts": "2026-05-13T12:00:00+00:00",
    }


# -------------------------------------------------------------------------
# assert_no_lexical_tics
# -------------------------------------------------------------------------


def test_lexical_tics_pass_on_clean_turn() -> None:
    assert_no_lexical_tics(
        "You mentioned the Director — does the new hire own the close?"
    )


@pytest.mark.parametrize(
    "tic_text",
    [
        "Crucially, this is a compliance role.",
        "It's worth noting the Director owns the close.",
        "Honestly, that's a clean answer.",
        "load-bearing piece of the role",
        "What does builder depth look like here?",
        "How would you engage with the Director?",
    ],
)
def test_lexical_tics_raise_on_banned_phrase(tic_text: str) -> None:
    with pytest.raises(AssertionError):
        assert_no_lexical_tics(tic_text)


def test_lexical_tics_pass_on_engagement_noun() -> None:
    """The verb 'engage' is forbidden; the noun 'engagement' is fine."""

    assert_no_lexical_tics("How does this fit into the engagement model?")


def test_lexical_tic_constants_are_lowercase() -> None:
    """Banlist entries are matched case-insensitively, so the canonical
    form is lowercase. If a future commit accidentally adds a mixed-case
    entry, the regex would still work but it'd be confusing.
    """

    for tic in LEXICAL_TIC_BANLIST:
        assert tic == tic.lower(), f"Banlist entry {tic!r} should be lowercase"
    for verb in LEXICAL_TIC_VERB_BANLIST:
        assert verb == verb.lower()


# -------------------------------------------------------------------------
# assert_no_bullets
# -------------------------------------------------------------------------


def test_bullets_pass_on_prose() -> None:
    assert_no_bullets(
        "Got it — sales tax compliance, multi-state filings, partner "
        "with the Director on production-entity work."
    )


@pytest.mark.parametrize(
    "bad_text",
    [
        "Three things matter:\n- CPA\n- sales tax reps\n- spreadsheets",
        "Three things matter:\n* CPA\n* sales tax reps",
        "Steps:\n1. CPA\n2. sales tax reps",
    ],
)
def test_bullets_raise_on_bullet_formatting(bad_text: str) -> None:
    with pytest.raises(AssertionError):
        assert_no_bullets(bad_text)


# -------------------------------------------------------------------------
# assert_back_reference
# -------------------------------------------------------------------------


def test_back_reference_passes_on_specific_question() -> None:
    prior = [
        _msg("recruiter", "It's a senior tax associate at Northwind."),
        _msg("recruiter", "She'll work with the Director on sales tax."),
    ]
    cloris = (
        "You mentioned the Director — does the new hire own the close, "
        "or just support it?"
    )
    assert_back_reference(cloris, prior, source_packet=None)


def test_back_reference_raises_on_generic_question() -> None:
    prior = [
        _msg("recruiter", "It's a senior tax associate at Northwind."),
        _msg("recruiter", "She works with the Director."),
    ]
    cloris = "What does the role report into?"
    with pytest.raises(AssertionError):
        assert_back_reference(cloris, prior, source_packet=None)


def test_back_reference_passes_on_non_question_turn() -> None:
    """Statements without questions don't need to back-reference —
    only question-bearing turns do.
    """

    cloris = "Got it. Updating the brief now."
    assert_back_reference(cloris, prior_messages=[], source_packet=None)


def test_back_reference_uses_source_packet_when_present() -> None:
    """A question that references something in the source_packet (not
    in the prior turns) should still pass.
    """

    cloris = "Is the Avalara integration something the new hire would set up?"
    assert_back_reference(
        cloris,
        prior_messages=[],
        source_packet={
            "job_description_text": "Senior Tax Associate at Northwind. Avalara experience preferred."
        },
    )


def test_back_reference_passes_when_no_grounding_context_available() -> None:
    """Opener turns have no prior recruiter messages and no source
    packet — the rule can't apply, so it passes silently rather than
    failing on every cold-start opener.
    """

    cloris = "Hi — what's the role?"
    assert_back_reference(cloris, prior_messages=[], source_packet=None)


# -------------------------------------------------------------------------
# assert_spaced_em_dashes_only
# -------------------------------------------------------------------------


def test_spaced_em_dash_passes() -> None:
    assert_spaced_em_dashes_only(
        "Production incentives sit on the JD — the Director owns them."
    )


def test_unspaced_em_dash_raises() -> None:
    with pytest.raises(AssertionError):
        assert_spaced_em_dashes_only("tax—not planning—compliance")


def test_unspaced_en_dash_raises() -> None:
    with pytest.raises(AssertionError):
        assert_spaced_em_dashes_only("tax–not planning")


def test_no_dashes_passes() -> None:
    assert_spaced_em_dashes_only("Got it. The Director owns the close.")


# -------------------------------------------------------------------------
# assert_sentence_count
# -------------------------------------------------------------------------


def test_sentence_count_passes_under_max() -> None:
    text = "Got it. The Director owns the close. What about the audit support?"
    assert_sentence_count(text, max_sentences=5)


def test_sentence_count_raises_over_max() -> None:
    text = "A. B! C? D. E. F."  # 6 sentences
    with pytest.raises(AssertionError):
        assert_sentence_count(text, max_sentences=5)


# -------------------------------------------------------------------------
# Per-conversation helpers
# -------------------------------------------------------------------------


def test_rhetorical_question_transition_raises_when_present() -> None:
    transcript = [
        _msg("recruiter", "It's a senior tax associate."),
        _msg("cloris", "What does this mean for the search? It means we focus on compliance."),
    ]
    with pytest.raises(AssertionError):
        assert_no_rhetorical_question_transitions(transcript)


def test_rhetorical_question_transition_passes_on_real_question() -> None:
    """Questions that ask the recruiter for input pass — this rule
    catches the rhetorical 'What does this mean? It means…' construction
    only.
    """

    transcript = [
        _msg("recruiter", "Tax associate."),
        _msg("cloris", "What does the day-to-day look like for this role?"),
    ]
    # A real question, not a rhetorical transition. Should pass.
    assert_no_rhetorical_question_transitions(transcript)


def test_throat_clearing_opener_raises_when_present() -> None:
    transcript = [
        _msg("recruiter", "Senior tax associate."),
        _msg("cloris", "Just to clarify, you said senior, right?"),
    ]
    with pytest.raises(AssertionError):
        assert_no_throat_clearing_openers(transcript)


def test_throat_clearing_opener_passes_on_substantive_open() -> None:
    transcript = [
        _msg("recruiter", "Senior tax associate."),
        _msg("cloris", "Got it — senior, not staff. Owns the close end-to-end?"),
    ]
    assert_no_throat_clearing_openers(transcript)


def test_register_slip_raises_on_corporate_cheery() -> None:
    transcript = [
        _msg("recruiter", "Tax associate."),
        _msg("cloris", "Awesome! Tell me more."),
    ]
    with pytest.raises(AssertionError):
        assert_no_register_slip(transcript)


def test_register_slip_passes_on_warm_but_grounded() -> None:
    transcript = [
        _msg("recruiter", "Tax associate."),
        _msg(
            "cloris",
            "Got it — senior tax associate. What does the close look like today?",
        ),
    ]
    assert_no_register_slip(transcript)


# -------------------------------------------------------------------------
# Pin the rewritten C9 system prompt
# -------------------------------------------------------------------------


def test_system_prompt_no_longer_carries_c9_todo_markers() -> None:
    """The C2 scaffold marked sections with ``TODO C9``. After the
    voice rewrite, those markers should be gone — the voice content
    has landed.
    """

    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "TODO C9" not in prompt


def test_system_prompt_cites_voice_guide_by_name_after_rewrite() -> None:
    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "~/Downloads/Sam_Vangelos_Voice_Guide_v2.md" in prompt


def test_system_prompt_includes_all_construction_exemplars() -> None:
    """Each of the §II construction exemplars should appear in the
    prompt's HOW TO CONSTRUCT SENTENCES block.
    """

    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "Colon-as-reveal" in prompt
    assert "Spaced em-dash" in prompt
    assert "Self-correcting aside" in prompt
    assert "Synthesis through convergence" in prompt
    assert "Escalation through catalog" in prompt


def test_system_prompt_includes_specificity_rule_with_pass_example() -> None:
    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "# SPECIFICITY RULE" in prompt
    assert "Director's spreadsheets" in prompt or "Director" in prompt
    assert "Tell me about the role" in prompt  # the failure example


def test_conversation_prompt_translates_backend_labels() -> None:
    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    lower = prompt.lower()
    assert "people who look right on paper but should be screened out" in lower
    assert "what separates someone who has really done" in lower
    assert "where i would look" in lower
    assert "LinkedIn / GitHub / both" not in prompt


def test_system_prompt_enumerates_lexical_banlist() -> None:
    """The §III FORBIDDEN PATTERNS block should enumerate the slime-
    critique kills so future readers know which strings the prompt
    blocks at the model's behavior level (the post-LLM voice tests
    enforce them at the regex level).
    """

    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "load-bearing" in prompt
    assert "builder depth" in prompt
    assert "would run this as" in prompt
    assert "engage" in prompt


def test_system_prompt_includes_few_shot_examples() -> None:
    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "# FEW-SHOT EXAMPLES" in prompt
    # Three labeled scenarios.
    assert "Happy path" in prompt
    assert "Terse recruiter" in prompt
    assert "JD dump" in prompt
    assert "Contradiction" in prompt


def test_system_prompt_carries_hiring_manager_picture_block() -> None:
    """The HIRING-MANAGER PICTURE block is load-bearing: the orchestrator
    must form a vivid picture before drafting. Generic question phrasing
    is explicitly banned; contrastive question phrasing is required.
    """

    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "# HIRING-MANAGER PICTURE" in prompt
    # Banned generic phrasings — the prompt enumerates them as failures.
    assert "What does success look like" in prompt
    assert "What skills are most important" in prompt
    # Contrastive (good) phrasing examples are present.
    assert "boardroom program shaper" in prompt
    assert "architecture trade-offs" in prompt
    # Trope banlist is named explicitly.
    assert "strong communication skills" in prompt.lower()
    assert "team player" in prompt.lower()
    # Recruiter correction semantics surface in the prompt.
    assert "corrected_by_recruiter" in prompt or "recruiter override" in prompt.lower()


def test_brief_understanding_block_now_names_hiring_manager_picture() -> None:
    """The BRIEF UNDERSTANDING checklist should mention the picture so the
    orchestrator treats it as part of "what the brief needs," not an
    optional flourish.
    """

    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    lower = prompt.lower()
    assert "picture of the person the hiring manager actually wants" in lower


def test_few_shot_examples_include_contrastive_picture_question() -> None:
    """The few-shot block should ship a contrastive picture-question
    example so the orchestrator copies the form, not the generic
    "what does success look like?" trope.
    """

    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "Hiring-manager picture" in prompt
    assert "quasi-CTO" in prompt
    # The contrastive form forces a choice.
    assert " or " in prompt


def test_system_prompt_inlines_user_facing_strings_in_sufficiency_block() -> None:
    """The sufficiency-volunteer template should appear in the # SUFFICIENCY
    STATE block when the brief is ready to file — the orchestrator can
    quote it instead of inventing a parallel volunteer phrasing.
    """

    prompt = build_orchestrator_system_prompt(sufficiency_state=(True, []))
    assert SUFFICIENCY_VOLUNTEER_TEMPLATE in prompt


def test_system_prompt_inlines_cap_strings() -> None:
    soft_prompt = build_orchestrator_system_prompt(
        sufficiency_state=(False, []), cap_state="soft"
    )
    assert SOFT_CAP_REMINDER in soft_prompt

    hard_prompt = build_orchestrator_system_prompt(
        sufficiency_state=(False, []), cap_state="hard"
    )
    assert HARD_CAP_FORCE in hard_prompt


# -------------------------------------------------------------------------
# Pin the C9 user-facing strings against their own voice rules
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,value",
    [
        ("OPENER_NO_PACKET", OPENER_NO_PACKET),
        ("OPENER_WITH_PACKET", OPENER_WITH_PACKET),
        ("SUFFICIENCY_VOLUNTEER_TEMPLATE", SUFFICIENCY_VOLUNTEER_TEMPLATE),
        ("SOFT_CAP_REMINDER", SOFT_CAP_REMINDER),
        ("HARD_CAP_FORCE", HARD_CAP_FORCE),
    ],
)
def test_user_facing_strings_pass_lexical_and_dash_checks(
    name: str, value: str
) -> None:
    """Each of the four user-facing strings must pass the lexical-tic
    and spaced-em-dash voice checks. The acid test (C12) catches the
    other voice issues; this is the auditable floor.
    """

    assert_no_lexical_tics(value)
    assert_spaced_em_dashes_only(value)
    assert_no_bullets(value)
