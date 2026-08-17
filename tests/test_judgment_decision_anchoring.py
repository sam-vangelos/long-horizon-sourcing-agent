"""Decision-token anchoring regression tests (Phase-1 fix R1 + R2).

These pins fasten the DECISION-extraction contract for the two production
parsers in ``linkedin/judgment_templates.py`` against two live-confirmed
continuation/substring bleeds:

R1 — ``parse_full_evaluation_response`` read the DECISION value via
``_extract_field``, which ABSORBS continuation prose until the next known
field, then ran an UNANCHORED decision ladder that tested ``"SAVE"`` before
``"REJECT"``. A clean ``DECISION: REJECT`` followed by rationale prose
containing the substring "save" (e.g. "not worth a save") flipped to SAVE —
routing a rejected candidate to the recruiter.

R2 — ``parse_facial_response`` matched ``"YES" in value`` on the DECISION line
(unanchored) and scanned ``FACIAL_YES`` before ``FACIAL_NO`` in the whole-raw
fallback. A body that merely mentions the YES token (``DECISION: NO -- not a
yes`` or ``not a FACIAL_YES ... FACIAL_NO``) returned FACIAL_YES — a wrong
facial verdict and a wasted profile open.

The fix anchors both parsers to the DECISION line's OWN value and matches the
decision token EXACTLY against the canonical vocabulary in
``shared.contracts`` (``FULL_DECISIONS`` / ``FACIAL_DECISIONS`` derivatives)
with leading-token / longest-match semantics, so the SAVE-family is never
collapsed and trailing rationale prose can never contribute the verdict.

Each negative test below was confirmed RED at the pre-fix HEAD (the parser
returned the WRONG decision — SAVE / FACIAL_YES). The positive pins lock the
well-formed common case and the SAVE-family / facial-class distinctions so the
fix cannot regress into permissiveness.
"""

from __future__ import annotations

from linkedin.judgment_templates import (
    parse_facial_response,
    parse_full_evaluation_response,
)


# ---------------------------------------------------------------------------
# Full-evaluation payload builder
#
# A complete, well-formed 4-step body whose ONLY variable is the decision block
# (the DECISION line plus whatever trailing prose/fields follow it). This keeps
# every test exercising the real ``parse_full_evaluation_response`` end-to-end,
# not a hand-fed decision string.
# ---------------------------------------------------------------------------


def _full_eval(decision_block: str) -> str:
    return (
        "STEP_1_MATCH: NONE\n"
        "STEP_1_AREA: N/A\n"
        "STEP_1_EVIDENCE: Work falls outside the capability areas.\n"
        "STEP_2_DEPTH: USER\n"
        "STEP_2_EVIDENCE: Uses tools; doesn't build them.\n"
        "STEP_3_TRANSFERABILITY: NOT_TRANSFERABLE\n"
        "STEP_3_EVIDENCE: Methodology gap is too wide.\n"
        "CASE_FOR: Adjacent vocabulary only.\n"
        "CASE_AGAINST: Wrong depth and wrong domain.\n"
        + decision_block
    )


# ---------------------------------------------------------------------------
# Markdown-emphasized field labels
# ---------------------------------------------------------------------------


def test_full_eval_markdown_bold_labels_parse_and_stop_continuation():
    raw = (
        "**STEP_1_MATCH:** DIRECT\n"
        "**STEP_1_AREA:** Applied AI\n"
        "**STEP_1_EVIDENCE:** Built ranking systems with production feedback loops.\n"
        "**STEP_2_DEPTH:** BUILDER\n"
        "**STEP_2_EVIDENCE:** Owned model quality and evaluation loops.\n"
        "**STEP_3_TRANSFERABILITY:** TRANSFERABLE\n"
        "**STEP_3_EVIDENCE:** The methodology maps cleanly to sourcing judgment.\n"
        "**CASE_FOR:** Direct AI systems ownership.\n"
        "**CASE_AGAINST:** Limited recruiter-facing product evidence.\n"
        "**DECISION:** SAVE\n"
        "**CONFIDENCE:** 0.82\n"
        "**POST_SAVE_MODIFIER:** NONE\n"
        "**SUMMARY:** Strong direct fit.\n"
    )

    result = parse_full_evaluation_response(raw)

    assert result.decision == "SAVE"
    assert result.confidence == 0.82
    assert result.match_type == "DIRECT"
    assert result.capability_evidence == (
        "Built ranking systems with production feedback loops."
    )
    assert "STEP_2_DEPTH" not in result.capability_evidence


def test_full_eval_markdown_bold_reject_with_trailing_save_prose():
    raw = _full_eval(
        "**DECISION:** REJECT\n"
        "Not worth a save given the domain gap.\n"
        "**CONFIDENCE:** 0.30\n"
        "**POST_SAVE_MODIFIER:** NONE\n"
        "**SUMMARY:** Adjacent at best.\n"
    )

    assert parse_full_evaluation_response(raw).decision == "REJECT"


def test_facial_markdown_bold_decision_resolves_to_facial_yes():
    raw = (
        "**DECISION:** FACIAL_YES\n"
        "**REASON:** ML research lead is a direct capability signal."
    )

    result = parse_facial_response(raw)

    assert result.decision == "FACIAL_YES"
    assert result.reason == "ML research lead is a direct capability signal."


# ---------------------------------------------------------------------------
# R1 — full-eval continuation-bleed / SAVE-before-REJECT ladder
# ---------------------------------------------------------------------------


def test_full_eval_wellformed_reject_with_trailing_save_prose():
    """The most damning case: a clean ``DECISION: REJECT`` followed by
    rationale prose containing "save" must stay REJECT.

    Pre-fix HEAD: ``_extract_field`` absorbed the continuation line
    ``Not worth a save.`` into the decision value (``"REJECT Not worth a
    save."``), then the ladder matched ``"SAVE"`` before ``"REJECT"`` and
    returned SAVE — sending a rejected candidate to the recruiter.
    """

    raw = _full_eval(
        "DECISION: REJECT\n"
        "Not worth a save given the domain gap.\n"
        "CONFIDENCE: 0.30\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Adjacent at best.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "REJECT"


def test_full_eval_wellformed_reject_with_inline_save_prose():
    """Inline em-dash rationale on the DECISION line itself
    (``DECISION: REJECT -- not worth a save``) must stay REJECT.

    Pre-fix HEAD returned SAVE: the value was ``"REJECT -- not worth a save"``
    and the unanchored ``"SAVE"`` substring test fired before ``"REJECT"``.
    """

    raw = _full_eval(
        "DECISION: REJECT -- not worth a save\n"
        "CONFIDENCE: 0.30\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Adjacent at best.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "REJECT"


def test_full_eval_blank_decision_with_reject_prose_is_not_save():
    """A bare ``DECISION:`` whose value is supplied as following reject prose
    containing "save" must never resolve to SAVE.

    Pre-fix HEAD: ``_extract_field`` absorbed the continuation into the
    decision value and the ``"SAVE"`` substring won -> SAVE. Post-fix the
    DECISION line has no own-value decision token, so it resolves to a
    non-SAVE outcome (REJECT if the prose leads with a decision token,
    otherwise PARSE_FAILURE). Either is acceptable; SAVE is the bug.
    """

    raw = _full_eval(
        "DECISION:\n"
        "This candidate is a clear reject; not worth a save given the gap.\n"
        "CONFIDENCE: 0.30\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Outside scope.\n"
    )
    decision = parse_full_evaluation_response(raw).decision
    assert decision in ("REJECT", "PARSE_FAILURE")
    assert decision != "SAVE"


# ---------------------------------------------------------------------------
# R1 — positive token-correctness pins (SAVE-family must not collapse)
# ---------------------------------------------------------------------------


def test_full_eval_inferential_save_resolves_to_inferential_save_not_save():
    raw = _full_eval(
        "DECISION: INFERENTIAL_SAVE\n"
        "CONFIDENCE: 0.42\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Sparse profile, inferential save.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "INFERENTIAL_SAVE"


def test_full_eval_transferable_save_resolves_to_itself():
    raw = _full_eval(
        "DECISION: TRANSFERABLE_SAVE\n"
        "CONFIDENCE: 0.55\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Methodology transfers.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "TRANSFERABLE_SAVE"


def test_full_eval_signal_save_resolves_to_itself():
    raw = _full_eval(
        "DECISION: SIGNAL_SAVE\n"
        "CONFIDENCE: 0.50\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Signal save.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "SIGNAL_SAVE"


def test_full_eval_plain_save_resolves_to_save():
    raw = _full_eval(
        "DECISION: SAVE\n"
        "CONFIDENCE: 0.82\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Strong direct fit.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "SAVE"


def test_full_eval_plain_reject_resolves_to_reject():
    raw = _full_eval(
        "DECISION: REJECT\n"
        "CONFIDENCE: 0.30\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Not a fit.\n"
    )
    assert parse_full_evaluation_response(raw).decision == "REJECT"


def test_full_eval_review_outcomes_resolve_to_themselves():
    """REVIEW_INFERRED / REVIEW_FLAGGED must still win over bare SAVE/REJECT
    so the non-save review routing stays intact.
    """

    inferred = _full_eval(
        "DECISION: REVIEW_INFERRED\n"
        "CONFIDENCE: 0.48\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "REVIEW_REASON: inferred_high_priority\n"
        "STRUCTURAL_EVIDENCE: senior title; CS PhD\n"
        "SUMMARY: Spot check.\n"
    )
    flagged = _full_eval(
        "DECISION: REVIEW_FLAGGED\n"
        "CONFIDENCE: 0.42\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "REVIEW_REASON: needs_more_evidence\n"
        "RECOMMENDED_NEXT_STEP: Confirm hands-on scope.\n"
        "SUMMARY: Needs follow-up.\n"
    )
    assert parse_full_evaluation_response(inferred).decision == "REVIEW_INFERRED"
    assert parse_full_evaluation_response(flagged).decision == "REVIEW_FLAGGED"


# ---------------------------------------------------------------------------
# R2 — facial DECISION-line substring + whole-raw fallback
# ---------------------------------------------------------------------------


def test_facial_decision_line_no_with_embedded_yes():
    """``DECISION: NO -- not a yes`` must resolve to FACIAL_NO.

    Pre-fix HEAD: the DECISION-line branch tested ``"YES" in value`` before
    ``"NO"``, and "not a yes" contains "YES" -> FACIAL_YES (wrong verdict,
    wasted profile open).
    """

    raw = "DECISION: NO -- not a yes\nREASON: trajectory is clearly outside scope."
    assert parse_facial_response(raw).decision == "FACIAL_NO"


def test_facial_no_conclusion_with_yes_token_mention_is_not_yes():
    """A body with NO DECISION line that mentions the YES token but concludes
    FACIAL_NO must not return FACIAL_YES.

    Pre-fix HEAD: the whole-raw fallback scanned ``FACIAL_YES`` before
    ``FACIAL_NO`` and returned FACIAL_YES off the mention. Post-fix the body
    is ambiguous (two distinct class tokens present) so it fails loud as
    PARSE_FAILURE; a genuine FACIAL_NO is also acceptable. FACIAL_YES is the
    bug.
    """

    raw = "This is not a FACIAL_YES candidate.\nConclusion: FACIAL_NO."
    decision = parse_facial_response(raw).decision
    assert decision in ("FACIAL_NO", "PARSE_FAILURE")
    assert decision != "FACIAL_YES"


# ---------------------------------------------------------------------------
# R2 — positive facial pins
# ---------------------------------------------------------------------------


def test_facial_clean_yes_resolves_to_facial_yes():
    raw = "DECISION: FACIAL_YES\nREASON: ML research lead is a direct capability signal."
    assert parse_facial_response(raw).decision == "FACIAL_YES"


def test_facial_clean_no_resolves_to_facial_no():
    raw = "DECISION: FACIAL_NO\nREASON: entire trajectory is pure frontend."
    assert parse_facial_response(raw).decision == "FACIAL_NO"


def test_facial_bare_yes_decision_line_resolves_to_facial_yes():
    """The parser still tolerates the bare ``YES`` token a model may emit on
    the DECISION line (mapped to the canonical FACIAL_YES class).
    """

    raw = "DECISION: YES\nREASON: direct capability signal."
    assert parse_facial_response(raw).decision == "FACIAL_YES"
