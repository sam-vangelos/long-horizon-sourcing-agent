"""P4 — bounded non-save review outcomes.

Pins the parser, OpusDecision schema, and orchestrator dispatch boundary
that the sourcing-judgment kernel Milestone B P4 introduced:

- ``parse_full_evaluation_response`` recognizes ``REVIEW_INFERRED`` and
  ``REVIEW_FLAGGED`` and pulls the optional ``REVIEW_REASON``,
  ``STRUCTURAL_EVIDENCE``, ``RECOMMENDED_NEXT_STEP`` fields into the
  ``FullEvaluationResult``.
- ``OpusDecision.to_dict()`` filters empty review fields so legacy
  ``SAVE`` / ``REJECT`` rows serialize byte-identically to pre-P4.
- The LinkedIn orchestrator's three-way dispatch routes REVIEW outcomes
  away from ``handle_save_decision`` (no LinkedIn save-click side effect)
  and away from the REJECT counter, and demotes REVIEW outcomes that
  fail the structural-evidence guard.

Wider behavior (lane attribution in ``terminal_payload_json``, full
runtime-state round-trip) is exercised in
``tests/test_linkedin_runtime_state.py``.
"""

from __future__ import annotations

from pathlib import Path

from linkedin.judgment_templates import parse_full_evaluation_response
from shared.contracts import NON_SAVE_REVIEW_DECISIONS, SAVE_DECISIONS
from shared.schemas import OpusDecision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _legacy_save_response() -> str:
    """A canonical SAVE response that pre-P4 parsed cleanly. Used as
    the byte-identical regression anchor."""

    return (
        "STEP_1_MATCH: DIRECT\n"
        "STEP_1_AREA: Agentic Systems\n"
        "STEP_1_EVIDENCE: Built multi-step orchestration platform serving "
        "fortune-100 clients.\n"
        "STEP_2_DEPTH: BUILDER\n"
        "STEP_2_EVIDENCE: Verbs are design / build / ship / own.\n"
        "STEP_3_TRANSFERABILITY: N/A\n"
        "STEP_3_EVIDENCE: N/A\n"
        "CASE_FOR: Operates the core capability area; senior delivery scope.\n"
        "CASE_AGAINST: None material.\n"
        "DECISION: SAVE\n"
        "CONFIDENCE: 0.82\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Strong direct fit on the core capability with senior scope.\n"
    )


def _legacy_reject_response() -> str:
    return (
        "STEP_1_MATCH: NONE\n"
        "STEP_1_AREA: N/A\n"
        "STEP_1_EVIDENCE: Outside the capability area; tooling consultant only.\n"
        "STEP_2_DEPTH: USER\n"
        "STEP_2_EVIDENCE: Uses tools; doesn't build them.\n"
        "STEP_3_TRANSFERABILITY: NOT_TRANSFERABLE\n"
        "STEP_3_EVIDENCE: Gap is too wide.\n"
        "CASE_FOR: Adjacent vocabulary.\n"
        "CASE_AGAINST: Wrong depth and wrong domain.\n"
        "DECISION: REJECT\n"
        "CONFIDENCE: 0.78\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Adjacent at best; not a save.\n"
    )


def _review_inferred_response() -> str:
    """A model response that exercises the new REVIEW_INFERRED path,
    including the optional STRUCTURAL_EVIDENCE list parsed via the
    semicolon delimiter.
    """

    return (
        "STEP_1_MATCH: ADJACENT\n"
        "STEP_1_AREA: Enterprise GenAI Platform\n"
        "STEP_1_EVIDENCE: Senior bank technologist with relevant org scope.\n"
        "STEP_2_DEPTH: BUILDER\n"
        "STEP_2_EVIDENCE: Inferred from PhD + ED title path.\n"
        "STEP_3_TRANSFERABILITY: TRANSFERABLE\n"
        "STEP_3_EVIDENCE: Capability transfers from adjacent BFS work.\n"
        "CASE_FOR: Structural signals justify a human look.\n"
        "CASE_AGAINST: Explicit evidence sparse on profile.\n"
        "DECISION: REVIEW_INFERRED\n"
        "CONFIDENCE: 0.48\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "REVIEW_REASON: inferred_high_priority\n"
        "STRUCTURAL_EVIDENCE: senior bank title; CS PhD; relevant org scope\n"
        "SUMMARY: Preserved for a recruiter spot check.\n"
    )


def _review_flagged_response() -> str:
    return (
        "STEP_1_MATCH: ADJACENT\n"
        "STEP_1_AREA: BFS Domain Applications\n"
        "STEP_1_EVIDENCE: Promising but evidence source is insufficient.\n"
        "STEP_2_DEPTH: BUILDER\n"
        "STEP_2_EVIDENCE: Inferred from prior scope.\n"
        "STEP_3_TRANSFERABILITY: N/A\n"
        "STEP_3_EVIDENCE: N/A\n"
        "CASE_FOR: Concrete next step would resolve ambiguity.\n"
        "CASE_AGAINST: Not enough evidence today.\n"
        "DECISION: REVIEW_FLAGGED\n"
        "CONFIDENCE: 0.42\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "REVIEW_REASON: needs_more_evidence\n"
        "RECOMMENDED_NEXT_STEP: Ask about agentic platform deployments at current employer.\n"
        "SUMMARY: Needs targeted follow-up.\n"
    )


# ---------------------------------------------------------------------------
# Parser pins
# ---------------------------------------------------------------------------


def test_parser_recognizes_review_inferred():
    result = parse_full_evaluation_response(_review_inferred_response())
    assert result.decision == "REVIEW_INFERRED"
    assert result.review_reason_code == "inferred_high_priority"
    assert result.review_structural_evidence == [
        "senior bank title",
        "CS PhD",
        "relevant org scope",
    ]
    assert result.review_recommended_next_step == ""


def test_parser_recognizes_review_flagged():
    result = parse_full_evaluation_response(_review_flagged_response())
    assert result.decision == "REVIEW_FLAGGED"
    assert result.review_reason_code == "needs_more_evidence"
    assert result.review_structural_evidence == []
    assert (
        result.review_recommended_next_step
        == "Ask about agentic platform deployments at current employer."
    )


def test_parser_does_not_inflate_save_decisions_with_review_fields():
    """SAVE / REJECT responses must leave the new fields empty so
    legacy payloads round-trip byte-identically.
    """

    save = parse_full_evaluation_response(_legacy_save_response())
    reject = parse_full_evaluation_response(_legacy_reject_response())

    for r in (save, reject):
        assert r.review_reason_code == ""
        assert r.review_structural_evidence == []
        assert r.review_recommended_next_step == ""

    assert save.decision == "SAVE"
    assert reject.decision == "REJECT"


def test_review_decision_matcher_beats_generic_save_and_reject():
    """REVIEW_INFERRED contains neither "SAVE" nor "REJECT" as a substring
    today, but the parser order (REVIEW first) is part of the contract
    so a future drift cannot silently miscategorize.
    """

    response = _review_inferred_response().replace(
        "DECISION: REVIEW_INFERRED", "DECISION: REVIEW_INFERRED SAVE"
    )
    result = parse_full_evaluation_response(response)
    assert result.decision == "REVIEW_INFERRED"


# ---------------------------------------------------------------------------
# OpusDecision.to_dict regression
# ---------------------------------------------------------------------------


def _opus_save() -> OpusDecision:
    return OpusDecision(
        stage="full",
        decision="SAVE",
        path="DIRECT:Agentic Systems",
        confidence=0.82,
        rationale="Strong direct fit.",
        candidate_name="Ada Lovelace",
        profile_url="/talent/profile/ada",
    )


def _opus_reject() -> OpusDecision:
    return OpusDecision(
        stage="full",
        decision="REJECT",
        path="NONE",
        confidence=0.78,
        rationale="Adjacent at best; not a save.",
        candidate_name="Ada Lovelace",
        profile_url="/talent/profile/ada",
    )


def test_to_dict_filters_empty_review_fields_for_save_decisions():
    payload = _opus_save().to_dict()
    for key in (
        "review_reason_code",
        "review_structural_evidence",
        "review_recommended_next_step",
        "review_ambiguity_reason",
    ):
        assert key not in payload, (
            f"P4 invariant: SAVE OpusDecision.to_dict() must not introduce "
            f"empty review_* keys (found {key!r})"
        )


def test_to_dict_filters_empty_review_fields_for_reject_decisions():
    payload = _opus_reject().to_dict()
    for key in (
        "review_reason_code",
        "review_structural_evidence",
        "review_recommended_next_step",
        "review_ambiguity_reason",
    ):
        assert key not in payload, (
            f"P4 invariant: REJECT OpusDecision.to_dict() must not introduce "
            f"empty review_* keys (found {key!r})"
        )


def test_to_dict_preserves_review_fields_when_populated():
    decision = OpusDecision(
        stage="full",
        decision="REVIEW_INFERRED",
        path="ADJACENT:Enterprise GenAI Platform",
        confidence=0.48,
        rationale="Preserved for a recruiter spot check.",
        candidate_name="Ada Lovelace",
        profile_url="/talent/profile/ada",
        review_reason_code="inferred_high_priority",
        review_structural_evidence=[
            "senior bank title",
            "CS PhD",
            "relevant org scope",
        ],
    )
    payload = decision.to_dict()
    assert payload["decision"] == "REVIEW_INFERRED"
    assert payload["review_reason_code"] == "inferred_high_priority"
    assert payload["review_structural_evidence"] == [
        "senior bank title",
        "CS PhD",
        "relevant org scope",
    ]
    # Other review fields stay empty and remain filtered out.
    assert "review_recommended_next_step" not in payload
    assert "review_ambiguity_reason" not in payload


# ---------------------------------------------------------------------------
# Orchestrator structural-evidence guard (pure helper)
# ---------------------------------------------------------------------------


def _review_decision(
    *,
    decision: str = "REVIEW_INFERRED",
    structural_evidence: list[str] | None = None,
    recommended_next_step: str = "",
    reason_code: str = "inferred_high_priority",
) -> OpusDecision:
    return OpusDecision(
        stage="full",
        decision=decision,
        path="ADJACENT:Enterprise GenAI Platform",
        confidence=0.48,
        rationale="Preserved for a recruiter spot check.",
        candidate_name="Ada Lovelace",
        profile_url="/talent/profile/ada",
        review_reason_code=reason_code,
        review_structural_evidence=structural_evidence
        if structural_evidence is not None
        else ["senior bank title", "CS PhD", "relevant org scope"],
        review_recommended_next_step=recommended_next_step,
    )


def test_non_save_review_decisions_constant_shape():
    assert NON_SAVE_REVIEW_DECISIONS == frozenset(
        {"REVIEW_INFERRED", "REVIEW_FLAGGED"}
    )


def test_review_demotion_helper_passes_strong_inferred():
    from linkedin.orchestrator import review_decision_demotion_reason

    strong = _review_decision(
        structural_evidence=[
            "senior bank title",
            "CS PhD",
            "relevant org scope",
        ],
    )
    assert review_decision_demotion_reason(strong) == ""


def test_review_demotion_helper_demotes_inferred_with_one_signal():
    from linkedin.orchestrator import review_decision_demotion_reason

    weak = _review_decision(structural_evidence=["only one signal"])
    assert (
        review_decision_demotion_reason(weak)
        == "insufficient_structural_evidence"
    )


def test_review_demotion_helper_demotes_inferred_with_no_signals():
    from linkedin.orchestrator import review_decision_demotion_reason

    weak = _review_decision(structural_evidence=[])
    assert (
        review_decision_demotion_reason(weak)
        == "insufficient_structural_evidence"
    )


def test_review_demotion_helper_passes_flagged_with_next_step():
    from linkedin.orchestrator import review_decision_demotion_reason

    flagged = _review_decision(
        decision="REVIEW_FLAGGED",
        structural_evidence=[],
        recommended_next_step="Ask about agentic platform deployments.",
    )
    assert review_decision_demotion_reason(flagged) == ""


def test_review_demotion_helper_demotes_flagged_without_next_step():
    from linkedin.orchestrator import review_decision_demotion_reason

    weak = _review_decision(
        decision="REVIEW_FLAGGED",
        structural_evidence=[],
        recommended_next_step="   ",
    )
    assert (
        review_decision_demotion_reason(weak)
        == "missing_recommended_next_step"
    )


def test_review_demotion_helper_demotes_invalid_reason_code():
    from linkedin.orchestrator import review_decision_demotion_reason

    invalid = _review_decision(
        decision="REVIEW_INFERRED",
        reason_code="nonsense",
        structural_evidence=["senior title", "relevant employer"],
    )
    assert review_decision_demotion_reason(invalid) == "invalid_review_reason_code"


def test_review_demotion_helper_demotes_missing_reason_code():
    from linkedin.orchestrator import review_decision_demotion_reason

    missing = _review_decision(
        decision="REVIEW_FLAGGED",
        reason_code="",
        structural_evidence=[],
        recommended_next_step="Verify current employer on LinkedIn.",
    )
    assert review_decision_demotion_reason(missing) == "invalid_review_reason_code"


def test_review_demotion_helper_ignores_non_review_decisions():
    """SAVE / REJECT decisions must return an empty reason — the guard
    is opt-in by decision class, never blanket.
    """

    from linkedin.orchestrator import review_decision_demotion_reason

    for decision in (_opus_save(), _opus_reject()):
        assert review_decision_demotion_reason(decision) == ""


def test_clear_review_evidence_in_place():
    """The orchestrator clears review evidence on the demotion path so
    the canonical row never carries half-populated review metadata after
    the decision flips to REJECT.
    """

    from linkedin.orchestrator import _clear_review_evidence

    decision = _review_decision(
        structural_evidence=["one"], recommended_next_step="step"
    )
    decision.review_ambiguity_reason = "ambiguous"

    _clear_review_evidence(decision)

    assert decision.review_reason_code == ""
    assert decision.review_structural_evidence == []
    assert decision.review_recommended_next_step == ""
    assert decision.review_ambiguity_reason == ""


# ---------------------------------------------------------------------------
# Save-gate exclusion (orchestrator dispatch invariant)
# ---------------------------------------------------------------------------


def test_orchestrator_save_gate_does_not_include_review_decisions():
    """The orchestrator's dispatch site at the SAVE branch checks
    ``final.decision in SAVE_FAMILY_DECISIONS``.  REVIEW outcomes MUST NOT
    be in that frozen contract so the LinkedIn save click
    can never fire on a preserved-not-saved candidate.

    Pinning this via a source-level grep keeps the contract visible even
    when the orchestrator's dispatch evolves, while pinning the shared set
    separately avoids depending on an inline tuple's formatting.
    """

    source = Path(__file__).parent.parent / "linkedin" / "orchestrator.py"
    text = source.read_text()
    # The named save family is the load-bearing point of the dispatch.
    assert "if final.decision in SAVE_FAMILY_DECISIONS:" in text
    assert SAVE_DECISIONS == frozenset({
        "SAVE",
        "INFERENTIAL_SAVE",
        "TRANSFERABLE_SAVE",
        "SIGNAL_SAVE",
    })
    assert SAVE_DECISIONS.isdisjoint(NON_SAVE_REVIEW_DECISIONS)
    # REVIEW decisions MUST be in the elif branch, not the save branch.
    assert (
        "elif final.decision in NON_SAVE_REVIEW_DECISIONS:" in text
    )
