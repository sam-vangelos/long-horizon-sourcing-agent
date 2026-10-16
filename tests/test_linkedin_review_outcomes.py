"""Review outcomes RETIRED (Sam's ruling, 2026-09-16) — the agent IS the reviewer.

P4 gave the full evaluation a third bucket: REVIEW_INFERRED / REVIEW_FLAGGED,
terminal decisions that parked a candidate for a human spot check. There is no
human spot check. A full evaluation now ends in the recruiter's pipeline (SAVE
and its typed variants) or out of it (REJECT), and a strong case carrying one
conversation-resolvable question goes through INFERENTIAL_SAVE — which reaches
the pipeline — rather than into a review file.

What this file pins, in both directions:

- the parser MAPS a surviving review verdict to REJECT with reject_reason
  ``REVIEW_RETIRED`` and preserves the named verdict on ``legacy_decision``;
  it never routes one through PARSE_FAILURE, because silently losing a
  verdict is how a retirement hides a regression;
- ``OpusDecision.to_dict()`` still round-trips a HISTORICAL review row, so a
  stored payload written before the retirement reads back intact;
- the orchestrator's dispatch has no review branch left to route to.

Wider behavior (lane attribution in ``terminal_payload_json``, full
runtime-state round-trip, historical hydration) is exercised in
``tests/test_linkedin_runtime_state.py``.
"""

from __future__ import annotations

from pathlib import Path

from linkedin.judgment_templates import parse_full_evaluation_response
from shared.contracts import (
    FULL_DECISIONS,
    LEGACY_REVIEW_DECISIONS,
    SAVE_DECISIONS,
)
from shared.schemas import OpusDecision
import pytest


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
    """A pre-retirement REVIEW_INFERRED response, verbatim.

    Kept exactly as a live model once produced it — including the optional
    STRUCTURAL_EVIDENCE list — because the mapping has to survive the real
    shape, not a tidied-up one.
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
# Parser: the retirement mapping
# ---------------------------------------------------------------------------


def test_parser_maps_legacy_review_verdict_to_reject_review_retired():
    for raw, named in (
        (_review_inferred_response(), "REVIEW_INFERRED"),
        (_review_flagged_response(), "REVIEW_FLAGGED"),
    ):
        result = parse_full_evaluation_response(raw)
        assert result.decision == "REJECT"
        assert result.reject_reason == "REVIEW_RETIRED"
        assert result.legacy_decision == named
        # Nothing only a review outcome could own survives the mapping.
        assert result.review_reason_code == ""
        assert result.review_structural_evidence == []
        assert result.review_recommended_next_step == ""
        assert result.outreach_tier is None
        assert result.post_save_modifier == "NONE"


def test_parser_never_routes_a_review_verdict_through_parse_failure():
    """The load-bearing half of the mapping.

    A PARSE_FAILURE is an unreadable response; a retired verdict is a
    perfectly readable one in the wrong vocabulary. Conflating them would
    bury the signal in the parse-failure counter, where nobody reads it.
    """

    for raw in (_review_inferred_response(), _review_flagged_response()):
        assert parse_full_evaluation_response(raw).decision != "PARSE_FAILURE"


def test_parser_does_not_mark_an_ordinary_verdict_as_legacy():
    """SAVE / REJECT responses carry no legacy marker and no review fields."""

    save = parse_full_evaluation_response(_legacy_save_response())
    reject = parse_full_evaluation_response(_legacy_reject_response())

    for r in (save, reject):
        assert r.legacy_decision == ""
        assert r.review_reason_code == ""
        assert r.review_structural_evidence == []
        assert r.review_recommended_next_step == ""

    assert save.decision == "SAVE"
    assert reject.decision == "REJECT"
    assert reject.reject_reason != "REVIEW_RETIRED"


def test_retired_token_still_wins_the_anchoring_contest_before_mapping():
    """Read the wrong token and the mapping records the wrong thing.

    ``REVIEW_INFERRED SAVE`` must resolve as the review verdict (then map
    to REJECT), never as a bare SAVE — the mapping is only as trustworthy
    as the token it reads.
    """

    response = _review_inferred_response().replace(
        "DECISION: REVIEW_INFERRED", "DECISION: REVIEW_INFERRED SAVE"
    )
    result = parse_full_evaluation_response(response)
    assert result.decision == "REJECT"
    assert result.legacy_decision == "REVIEW_INFERRED"


def test_template_decision_line_lists_only_save_family_and_reject():
    """The prompt is the real contract — the schema only guards the tool path.

    A DECISION line that still offers a review outcome re-creates the third
    bucket no matter what the code does with the answer.
    """

    from shared.judgment.templates import FULL_EVALUATION_TEMPLATE

    decision_lines = [
        line
        for line in FULL_EVALUATION_TEMPLATE.split("\n")
        if line.startswith("DECISION:")
    ]
    assert decision_lines == [
        "DECISION: SAVE or REJECT or INFERENTIAL_SAVE or TRANSFERABLE_SAVE "
        "or SIGNAL_SAVE"
    ]
    named = {
        token
        for token in decision_lines[0].split(": ", 1)[1].split(" or ")
    }
    assert named == SAVE_DECISIONS | {"REJECT"}
    assert named.isdisjoint(LEGACY_REVIEW_DECISIONS)

    # The retired output fields are gone from the response format, and the
    # retired outcomes are gone from the rubric prose that fed them.
    for retired_field in (
        "REVIEW_REASON:",
        "STRUCTURAL_EVIDENCE:",
        "RECOMMENDED_NEXT_STEP:",
    ):
        assert retired_field not in FULL_EVALUATION_TEMPLATE
    assert "REVIEW_INFERRED" not in FULL_EVALUATION_TEMPLATE
    assert "REVIEW_FLAGGED" not in FULL_EVALUATION_TEMPLATE

    # And the channel the near-miss case moved TO is still offered, with
    # the identity reason code the retirement added.
    assert "INFERENTIAL_SAVE" in FULL_EVALUATION_TEMPLATE
    assert "IDENTITY_UNRESOLVED" in FULL_EVALUATION_TEMPLATE


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
            f"SAVE OpusDecision.to_dict() must not introduce empty review_* "
            f"keys (found {key!r})"
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
            f"REJECT OpusDecision.to_dict() must not introduce empty review_* "
            f"keys (found {key!r})"
        )


def test_to_dict_still_round_trips_a_historical_review_row():
    """Retiring the OUTCOME must not make its stored rows unreadable.

    Live SQLite stores hold terminal rows written before 2026-09-16; a
    reader that drops their review payload turns a real verdict into a
    blank one on the next report or resume.
    """

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
# The retirement backstop (pure helpers)
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


@pytest.mark.structural
def test_legacy_review_decisions_constant_shape():
    assert LEGACY_REVIEW_DECISIONS == frozenset(
        {"REVIEW_INFERRED", "REVIEW_FLAGGED"}
    )
    assert FULL_DECISIONS.isdisjoint(LEGACY_REVIEW_DECISIONS)


def test_legacy_review_predicate_catches_both_outcomes():
    from linkedin.evaluation_support import is_legacy_review_decision

    for decision in ("REVIEW_INFERRED", "REVIEW_FLAGGED"):
        assert is_legacy_review_decision(_review_decision(decision=decision))


def test_legacy_review_predicate_ignores_live_decisions():
    """The backstop is opt-in by decision class, never blanket."""

    from linkedin.evaluation_support import is_legacy_review_decision

    for decision in (_opus_save(), _opus_reject()):
        assert not is_legacy_review_decision(decision)


def test_review_demotion_helper_is_gone():
    """The P4 structural-evidence guard cannot survive the retirement.

    It answered "has this review outcome earned its parking spot?" — a
    question with no parking spot left to earn. Leaving it importable
    would invite a caller to keep asking it.
    """

    import linkedin.evaluation_support as support

    assert not hasattr(support, "review_decision_demotion_reason")


def test_clear_review_evidence_in_place():
    """The backstop clears review evidence so the canonical row never
    carries half-populated review metadata after the decision flips to
    REJECT.
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
# Dispatch invariants (orchestrator source-level pins)
# ---------------------------------------------------------------------------


@pytest.mark.structural
def test_orchestrator_save_gate_does_not_include_review_decisions():
    """The orchestrator's SAVE branch checks ``final.decision in
    SAVE_FAMILY_DECISIONS``. A review verdict MUST NOT be in that frozen
    contract, so the LinkedIn save click can never fire on one.
    """

    source = Path(__file__).parent.parent / "linkedin" / "candidate_review.py"
    text = source.read_text()
    assert "if final.decision in SAVE_FAMILY_DECISIONS:" in text
    assert SAVE_DECISIONS == frozenset({
        "SAVE",
        "INFERENTIAL_SAVE",
        "TRANSFERABLE_SAVE",
        "SIGNAL_SAVE",
    })
    assert SAVE_DECISIONS.isdisjoint(LEGACY_REVIEW_DECISIONS)


def test_orchestrator_dispatch_has_no_review_branch():
    """Retired 2026-09-16: there is no third branch to route to.

    The dispatch is save-or-reject; a leftover ``elif ... review`` arm
    would mean a review verdict could still take a path of its own.
    """

    source = Path(__file__).parent.parent / "linkedin" / "candidate_review.py"
    text = source.read_text()
    assert "elif final.decision in NON_SAVE_REVIEW_DECISIONS:" not in text
    assert "elif is_legacy_review_decision(final):" not in text
    # The backstop maps rather than branches, and it stamps the countable
    # system reason.
    assert "if is_legacy_review_decision(final):" in text
    assert 'final.reject_reason = "REVIEW_RETIRED"' in text
