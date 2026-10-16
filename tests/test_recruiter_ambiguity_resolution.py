from shared.reconciliation_schemas import LinkedInMatchResult
from shared.recruiter_ambiguity_resolution import (
    SINGLE_STRONG_MIN_GAP_VS_NEXT_RANKED_CARD,
    STRUCTURAL_ANCHOR_SINGLE_OPEN_MIN_MATCH_CONFIDENCE,
    consolidate_multi_profile_reviews,
    is_single_strong_plausible_for_profile_open,
    single_plausible_is_safely_dominant,
)
from shared.recruiter_identity_schemas import RecruiterIdentityCandidate


def test_consolidate_unique_save_winner():
    reviews = [
        {"gate_final_action": "REJECT", "gate_final_subreason": "fit_reject"},
        {"gate_final_action": "SAVE", "gate_final_subreason": ""},
    ]
    out = consolidate_multi_profile_reviews(reviews)
    assert out.winner_index == 1
    assert out.final_action == "SAVE"


def test_consolidate_two_saves_is_ambiguous():
    reviews = [
        {"gate_final_action": "SAVE", "gate_final_subreason": ""},
        {"gate_final_action": "SAVE", "gate_final_subreason": ""},
    ]
    out = consolidate_multi_profile_reviews(reviews)
    assert out.winner_index is None
    assert out.final_action == "MANUAL_REVIEW"
    assert out.final_subreason == "identity_ambiguous"


def test_consolidate_all_fit_reject():
    reviews = [
        {"gate_final_action": "REJECT", "gate_final_subreason": "fit_reject"},
        {"gate_final_action": "REJECT", "gate_final_subreason": "fit_reject"},
    ]
    out = consolidate_multi_profile_reviews(reviews)
    assert out.final_action == "REJECT"
    assert out.final_subreason == "fit_reject"


def test_consolidate_mixed_outcomes_unresolved():
    reviews = [
        {"gate_final_action": "MANUAL_REVIEW", "gate_final_subreason": "borderline_fit"},
        {"gate_final_action": "REJECT", "gate_final_subreason": "fit_reject"},
    ]
    out = consolidate_multi_profile_reviews(reviews)
    assert out.final_action == "MANUAL_REVIEW"
    assert out.final_subreason == "ambiguity_unresolved"


def _cand(
    *,
    url: str,
    confidence: float,
    evidence: list[str],
    ambiguity_reasons: list[str] | None = None,
) -> RecruiterIdentityCandidate:
    return RecruiterIdentityCandidate(
        rank=1,
        profile_url=url,
        name="Ada Lovelace",
        match_confidence=confidence,
        evidence=evidence,
        ambiguity_reasons=ambiguity_reasons or [],
    )


def test_is_single_strong_requires_structural_evidence_and_floor():
    weak = _cand(
        url="/a",
        confidence=0.8,
        evidence=["Exact name match"],
    )
    assert is_single_strong_plausible_for_profile_open(weak) is False

    low = _cand(
        url="/b",
        confidence=0.65,
        evidence=["Exact name match", "Company overlap"],
    )
    assert is_single_strong_plausible_for_profile_open(low) is False

    ok = _cand(
        url="/c",
        confidence=0.75,
        evidence=["Exact name match", "Company overlap"],
    )
    assert is_single_strong_plausible_for_profile_open(ok) is True


def test_anchor_tier_opens_live_style_069_name_company_title():
    """Mirrors dry-run rows (e.g. Wenyue Hua / Keunwoo Choi): ~0.69 + exact name + company + title."""
    c = _cand(
        url="/wenyue",
        confidence=0.69,
        evidence=["Exact name match", "Company overlap", "Title overlap"],
        ambiguity_reasons=["Location mismatch"],
    )
    assert is_single_strong_plausible_for_profile_open(c) is True


def test_anchor_tier_rejects_single_structural_dimension_at_069():
    c = _cand(
        url="/x",
        confidence=0.69,
        evidence=["Exact name match", "Company overlap"],
    )
    assert is_single_strong_plausible_for_profile_open(c) is False


def test_anchor_tier_rejects_below_structural_floor_even_with_three_signals():
    c = _cand(
        url="/y",
        confidence=STRUCTURAL_ANCHOR_SINGLE_OPEN_MIN_MATCH_CONFIDENCE - 0.01,
        evidence=["Exact name match", "Company overlap", "Title overlap"],
    )
    assert is_single_strong_plausible_for_profile_open(c) is False


def test_name_mismatch_blocks_anchor_tier():
    c = _cand(
        url="/z",
        confidence=0.69,
        evidence=["Exact name match", "Company overlap", "Title overlap"],
        ambiguity_reasons=["Name mismatch"],
    )
    assert is_single_strong_plausible_for_profile_open(c) is False


def test_single_plausible_dominance_respects_gap_constant():
    lone = _cand(url="/win", confidence=0.75, evidence=["Company overlap"])
    other = _cand(url="/lose", confidence=0.68, evidence=["Company overlap"])
    scored = [
        (lone, LinkedInMatchResult(matched_profile_url="/win", match_confidence=0.75)),
        (other, LinkedInMatchResult(matched_profile_url="/lose", match_confidence=0.68)),
    ]
    assert lone.match_confidence - 0.68 < SINGLE_STRONG_MIN_GAP_VS_NEXT_RANKED_CARD
    assert single_plausible_is_safely_dominant(lone=lone, all_scored=scored) is False

    scored2 = [
        (lone, LinkedInMatchResult(matched_profile_url="/win", match_confidence=0.75)),
        (other, LinkedInMatchResult(matched_profile_url="/lose", match_confidence=0.55)),
    ]
    assert single_plausible_is_safely_dominant(lone=lone, all_scored=scored2) is True
