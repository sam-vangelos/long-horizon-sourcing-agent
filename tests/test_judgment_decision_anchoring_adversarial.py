"""Adversarial decision-anchoring pins (verifier additions).

Companion to ``tests/test_judgment_decision_anchoring.py``. The implementer's
suite pinned two flip directions (``REJECT`` + trailing "save" prose, facial
``NO`` + "yes" mention). This file closes the rest of the flip-resistance
matrix the fix must satisfy and asserts the no-collapse invariant explicitly:

  * the REVERSE flip directions — a SAVE-family / FACIAL_YES verdict with
    trailing prose naming REJECT / "no" must keep its own verdict, exactly as
    the forward direction does;
  * every SAVE-family token (``INFERENTIAL_SAVE`` / ``TRANSFERABLE_SAVE`` /
    ``SIGNAL_SAVE``) parses to ITSELF and is never collapsed to bare ``SAVE``
    even though each CONTAINS ``SAVE`` as a substring;
  * ``REVIEW_INFERRED`` / ``REVIEW_FLAGGED`` survive trailing save/reject prose;
  * ``FACIAL_BORDERLINE`` is never collapsed to ``FACIAL_YES`` / ``FACIAL_NO``;
  * the parsers' decision vocabulary is the canonical ``shared.contracts`` set
    BY REFERENCE for the full-eval ladder (cannot drift), and the facial
    normalization map is consistent with ``ACTIVE_FACIAL_DECISIONS``.

These were authored by the adversarial verifier; each REVERSE-flip negative was
hand-confirmed against the pre-fix HEAD parser (it returned the trailing-prose
token) before being committed.
"""

from __future__ import annotations

from linkedin import judgment_templates as jt
from linkedin.judgment_templates import (
    parse_facial_response,
    parse_full_evaluation_response,
)
from shared import contracts
from shared.contracts import ACTIVE_FACIAL_DECISIONS, FULL_DECISIONS, SAVE_DECISIONS


def _full_eval(decision_block: str) -> str:
    # A well-formed BUILDER/DIRECT body so the only variable is the decision
    # block; mirrors the companion file's builder but with a positive shape so
    # the SAVE-family verdicts are situationally plausible end-to-end.
    return (
        "STEP_1_MATCH: DIRECT\n"
        "STEP_1_AREA: ML\n"
        "STEP_1_EVIDENCE: Trains models end to end.\n"
        "STEP_2_DEPTH: BUILDER\n"
        "STEP_2_EVIDENCE: Implements architectures from scratch.\n"
        "STEP_3_TRANSFERABILITY: TRANSFERABLE\n"
        "STEP_3_EVIDENCE: Methodology maps to the brief.\n"
        "CASE_FOR: Direct capability and depth.\n"
        "CASE_AGAINST: Minor domain distance.\n"
        + decision_block
    )


# ---------------------------------------------------------------------------
# REVERSE flip direction — SAVE-family/FACIAL_YES + trailing REJECT/"no" prose.
# The forward direction (REJECT + save / NO + yes) is pinned in the companion
# file; the fix must be symmetric or it is only half-anchored.
# ---------------------------------------------------------------------------


def test_full_eval_save_with_trailing_reject_prose_stays_save():
    raw = _full_eval(
        "DECISION: SAVE\n"
        "Would have been a reject on a weaker profile.\n"
        "CONFIDENCE: 0.80\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Strong direct fit.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "SAVE"


def test_full_eval_save_inline_reject_prose_stays_save():
    raw = _full_eval(
        "DECISION: SAVE -- borderline, nearly a reject\n"
        "CONFIDENCE: 0.62\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Edge save.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "SAVE"


def test_full_eval_inferential_save_with_trailing_reject_prose_stays_itself():
    raw = _full_eval(
        "DECISION: INFERENTIAL_SAVE\n"
        "Not a reject despite the sparse public footprint.\n"
        "CONFIDENCE: 0.42\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Inferred fit.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "INFERENTIAL_SAVE"


def test_full_eval_inferential_save_inline_reject_prose_stays_itself():
    raw = _full_eval(
        "DECISION: INFERENTIAL_SAVE -- close to a reject but signals hold\n"
        "CONFIDENCE: 0.41\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Inferred fit.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "INFERENTIAL_SAVE"


def test_full_eval_transferable_save_with_trailing_reject_prose_stays_itself():
    raw = _full_eval(
        "DECISION: TRANSFERABLE_SAVE\n"
        "On domain alone this would be a reject.\n"
        "CONFIDENCE: 0.55\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Methodology transfers.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "TRANSFERABLE_SAVE"


def test_full_eval_signal_save_with_trailing_reject_prose_stays_itself():
    raw = _full_eval(
        "DECISION: SIGNAL_SAVE -- not a reject on signal strength\n"
        "CONFIDENCE: 0.50\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Signal save.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "SIGNAL_SAVE"


def test_full_eval_review_inferred_with_trailing_save_prose_stays_itself():
    raw = _full_eval(
        "DECISION: REVIEW_INFERRED -- not a clean save, spot check\n"
        "CONFIDENCE: 0.48\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "REVIEW_REASON: inferred_high_priority\n"
        "STRUCTURAL_EVIDENCE: senior title; CS PhD\n"
        "SUMMARY: Spot check.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "REVIEW_INFERRED"


def test_full_eval_review_flagged_with_trailing_reject_prose_stays_itself():
    raw = _full_eval(
        "DECISION: REVIEW_FLAGGED\n"
        "Leaning reject but needs a human look.\n"
        "CONFIDENCE: 0.42\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "REVIEW_REASON: needs_more_evidence\n"
        "SUMMARY: Needs follow-up.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "REVIEW_FLAGGED"


def test_facial_yes_with_trailing_no_prose_stays_facial_yes():
    raw = "DECISION: YES -- this is not a no\nREASON: direct capability signal."
    assert parse_facial_response(raw).decision == "FACIAL_YES"


def test_facial_yes_prefixed_with_facial_no_mention_stays_facial_yes():
    raw = "DECISION: FACIAL_YES -- emphatically not a FACIAL_NO\nREASON: lead signal."
    assert parse_facial_response(raw).decision == "FACIAL_YES"


def test_facial_borderline_with_yes_and_no_prose_stays_borderline():
    """BORDERLINE must not collapse to YES or NO even when the rationale on the
    DECISION line names both — the verdict leads with the BORDERLINE token.
    """

    raw = "DECISION: FACIAL_BORDERLINE -- could read as yes or no\nREASON: ambiguous trajectory."
    assert parse_facial_response(raw).decision == "FACIAL_BORDERLINE"


def test_facial_bare_borderline_resolves_to_borderline():
    raw = "DECISION: BORDERLINE\nREASON: ambiguous trajectory."
    assert parse_facial_response(raw).decision == "FACIAL_BORDERLINE"


# ---------------------------------------------------------------------------
# No-collapse invariant (the trap): each SAVE-family token parses to itself.
# Parametric over the canonical SAVE_DECISIONS set so a future SAVE-family
# member added to shared.contracts is covered automatically.
# ---------------------------------------------------------------------------


def test_every_save_family_token_resolves_to_itself_not_bare_save():
    for token in SAVE_DECISIONS:
        raw = _full_eval(
            f"DECISION: {token}\n"
            "CONFIDENCE: 0.50\n"
            "POST_SAVE_MODIFIER: NONE\n"
            "SUMMARY: x\n"
        )
        decision = parse_full_evaluation_response(raw).decision
        assert decision == token, (
            f"{token} collapsed to {decision!r} (SAVE-family must not collapse)"
        )


def test_non_save_full_decisions_resolve_to_themselves():
    for token in ("REJECT", "REVIEW_INFERRED", "REVIEW_FLAGGED"):
        raw = _full_eval(
            f"DECISION: {token}\n"
            "CONFIDENCE: 0.40\n"
            "POST_SAVE_MODIFIER: NONE\n"
            "REVIEW_REASON: needs_more_evidence\n"
            "STRUCTURAL_EVIDENCE: x\n"
            "SUMMARY: x\n"
        )
        assert parse_full_evaluation_response(raw).decision == token


def test_each_active_facial_class_resolves_to_itself():
    for cls in ACTIVE_FACIAL_DECISIONS:
        raw = f"DECISION: {cls}\nREASON: x"
        assert parse_facial_response(raw).decision == cls


# ---------------------------------------------------------------------------
# Vocabulary canonicity — the full-eval ladder must use shared.contracts BY
# REFERENCE (not a re-listed copy that could drift), and the facial map must
# stay consistent with the active facial contract.
# ---------------------------------------------------------------------------


def test_full_eval_vocab_is_canonical_contracts_object():
    # Imported by reference: a contract change can't be silently shadowed.
    assert jt.FULL_DECISIONS is contracts.FULL_DECISIONS


def test_facial_normalization_map_consistent_with_active_contract():
    prefixed = {k for k in jt._FACIAL_TOKEN_TO_CLASS if k.startswith("FACIAL_")}
    # The prefixed keys the parser normalizes must equal the active facial
    # contract; the map's targets must all be active classes. (Drift guard:
    # this fails loudly if a new active facial class is added to contracts but
    # not taught to the parser map.)
    assert prefixed == set(ACTIVE_FACIAL_DECISIONS)
    assert set(jt._FACIAL_TOKEN_TO_CLASS.values()) <= set(ACTIVE_FACIAL_DECISIONS)


# ---------------------------------------------------------------------------
# Trailing-prose-in-a-later-field — the decision token must come from the
# DECISION line, not from a token that appears inside a later field's prose.
# ---------------------------------------------------------------------------


def test_full_eval_reject_with_save_token_word_in_summary_stays_reject():
    raw = _full_eval(
        "DECISION: REJECT\n"
        "CONFIDENCE: 0.30\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: We could save this for a different brief, but not this one.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "REJECT"


def test_facial_no_with_yes_word_in_reason_stays_facial_no():
    raw = "DECISION: FACIAL_NO\nREASON: a recruiter might say yes elsewhere, not here."
    assert parse_facial_response(raw).decision == "FACIAL_NO"


# ---------------------------------------------------------------------------
# RED-CONFIRMED at pre-fix HEAD: cross-class continuation flips the implementer's
# forward suite did NOT cover. These are the same severity as the original R1
# bug but a DIFFERENT trigger — the old ladder checked the SPECIFIC SAVE-family
# / REVIEW tokens BEFORE bare SAVE/REJECT, so a clean verdict whose trailing
# prose merely NAMED a more-specific token was absorbed by ``_extract_field``
# and flipped to that token. Hand-confirmed: at the pre-fix HEAD parser the
# first returned INFERENTIAL_SAVE (a rejected candidate routed to the recruiter)
# and the second returned REVIEW_INFERRED (a clean save downgraded to review).
# ---------------------------------------------------------------------------


def test_full_eval_reject_with_save_family_token_in_trailing_prose_stays_reject():
    """``DECISION: REJECT`` whose rationale NAMES ``inferential_save`` must stay
    REJECT. Pre-fix HEAD flipped to INFERENTIAL_SAVE (specific-before-bare
    ladder + continuation absorb). This is the original bug's severity class
    via a trigger the implementer's suite missed.
    """

    raw = _full_eval(
        "DECISION: REJECT\n"
        "Not even an inferential_save case given the gap.\n"
        "CONFIDENCE: 0.30\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Outside scope.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "REJECT"


def test_full_eval_reject_inline_save_family_token_stays_reject():
    raw = _full_eval(
        "DECISION: REJECT -- not an inferential_save\n"
        "CONFIDENCE: 0.30\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Outside scope.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "REJECT"


def test_full_eval_reject_with_transferable_save_token_in_prose_stays_reject():
    raw = _full_eval(
        "DECISION: REJECT\n"
        "No transferable_save case here either.\n"
        "CONFIDENCE: 0.30\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Outside scope.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "REJECT"


def test_full_eval_save_with_review_token_in_trailing_prose_stays_save():
    """``DECISION: SAVE`` whose rationale NAMES ``review_inferred`` must stay
    SAVE. Pre-fix HEAD flipped to REVIEW_INFERRED (REVIEW checked before SAVE +
    continuation absorb), downgrading a clean save to non-save review routing.
    """

    raw = _full_eval(
        "DECISION: SAVE\n"
        "No need for review_inferred routing; this is a clear save.\n"
        "CONFIDENCE: 0.80\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Strong direct fit.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "SAVE"


def test_full_eval_save_with_review_flagged_token_in_prose_stays_save():
    raw = _full_eval(
        "DECISION: SAVE\n"
        "Not a review_flagged case.\n"
        "CONFIDENCE: 0.80\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Strong direct fit.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "SAVE"
