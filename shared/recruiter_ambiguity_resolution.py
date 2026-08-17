"""Conservative ambiguity resolution among multiple plausible Recruiter cards."""

from __future__ import annotations

from dataclasses import dataclass

from shared.reconciliation_schemas import LinkedInMatchResult
from shared.recruiter_identity_schemas import RecruiterIdentityCandidate


PLAUSIBLE_MIN_MATCH_CONFIDENCE = 0.45

# Tier 1 (numeric): high manual band + any structural scorer line — original strict path.
SINGLE_STRONG_PLAUSIBLE_MIN_MATCH_CONFIDENCE = 0.72
# Tier 2 (“anchor” path): same-person signals without claiming card-level certainty; floor sits
# above choose_best_match's no_confident_match ceiling behavior and below tier 1, with extra structure.
STRUCTURAL_ANCHOR_SINGLE_OPEN_MIN_MATCH_CONFIDENCE = 0.64
# Mirror the spirit of choose_best_match(..., best - second >= 0.15): reject “tied” top cards.
SINGLE_STRONG_MIN_GAP_VS_NEXT_RANKED_CARD = 0.12


def is_plausible_recruiter_candidate(candidate: RecruiterIdentityCandidate) -> bool:
    """Card-level plausibility: same heuristic as the resolver's plausible list."""
    if candidate.match_confidence < PLAUSIBLE_MIN_MATCH_CONFIDENCE:
        return False
    return not any("name mismatch" in reason.lower() for reason in candidate.ambiguity_reasons)


def dedupe_plausible_by_profile_url(candidates: list[RecruiterIdentityCandidate]) -> list[RecruiterIdentityCandidate]:
    """Preserve rank order; drop duplicate profile URLs."""
    seen: set[str] = set()
    out: list[RecruiterIdentityCandidate] = []
    for candidate in sorted(candidates, key=lambda c: c.rank):
        url = (candidate.profile_url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(candidate)
    return out


def _candidate_has_structural_identity_evidence(candidate: RecruiterIdentityCandidate) -> bool:
    """Non-name-only signals from scorer evidence (name-only plausibles stay card-level MANUAL)."""
    for item in candidate.evidence:
        lowered = str(item).lower()
        if "company" in lowered or "location" in lowered or "title" in lowered:
            return True
    return False


def _structural_evidence_kind_hits(candidate: RecruiterIdentityCandidate) -> dict[str, bool]:
    """Which of company / title / location the scorer surfaced (distinct dimensions)."""
    company = title = location = False
    for item in candidate.evidence:
        lowered = str(item).lower()
        if "company" in lowered:
            company = True
        if "title" in lowered:
            title = True
        if "location" in lowered:
            location = True
    return {"company": company, "title": title, "location": location}


def _has_exact_name_match_evidence(candidate: RecruiterIdentityCandidate) -> bool:
    for item in candidate.evidence:
        if "exact name match" in str(item).lower():
            return True
    return False


def is_single_strong_plausible_for_profile_open(candidate: RecruiterIdentityCandidate) -> bool:
    """True when the lone plausible card justifies opening the profile for holistic review (not SAVE).

    Hybrid policy (resolver still requires ``single_plausible_is_safely_dominant`` separately):

    **Tier 1 — numeric:** confidence >= 0.72 and at least one structural evidence line
    (company / title / location overlap per scorer). Same bar as before for “almost high” cards.

    **Tier 2 — structural anchor:** confidence in [0.64, 0.72) with **exact name match** evidence and
    **at least two** of {company, title, location} hit in evidence. Covers live dry-run rows where the
    card is clearly the same person on multiple career dimensions but location or scoring keeps
    ``choose_best_match`` in manual_review (e.g. ~0.69 + name + company + title).

    Below 0.64, or name-only / single-dimension structure, stays card-level MANUAL without open.
    """
    if not is_plausible_recruiter_candidate(candidate):
        return False
    conf = float(candidate.match_confidence or 0.0)
    if conf + 1e-9 >= SINGLE_STRONG_PLAUSIBLE_MIN_MATCH_CONFIDENCE:
        return _candidate_has_structural_identity_evidence(candidate)
    kinds = _structural_evidence_kind_hits(candidate)
    structural_dims = sum(1 for k in ("company", "title", "location") if kinds[k])
    if conf + 1e-9 < STRUCTURAL_ANCHOR_SINGLE_OPEN_MIN_MATCH_CONFIDENCE:
        return False
    if not _has_exact_name_match_evidence(candidate):
        return False
    if structural_dims < 2:
        return False
    return True


def single_plausible_is_safely_dominant(
    *,
    lone: RecruiterIdentityCandidate,
    all_scored: list[tuple[RecruiterIdentityCandidate, LinkedInMatchResult]],
) -> bool:
    """True when no other surfaced card is within SINGLE_STRONG_MIN_GAP_VS_NEXT_RANKED_CARD in score."""
    best_others = 0.0
    for cand, match in all_scored:
        if (cand.profile_url or "").strip() != (lone.profile_url or "").strip():
            best_others = max(best_others, float(match.match_confidence or 0.0))
    return lone.match_confidence - best_others >= SINGLE_STRONG_MIN_GAP_VS_NEXT_RANKED_CARD


@dataclass(frozen=True)
class MultiProfileOutcome:
    final_action: str
    final_subreason: str
    winner_index: int | None


def consolidate_multi_profile_reviews(reviews: list[dict]) -> MultiProfileOutcome:
    """Pick a unique SAVE winner or fall back to MANUAL_REVIEW / REJECT.

    Rules (conservative):
    - Exactly one reviewed profile gates to SAVE → that profile wins.
    - Two or more SAVE gates → MANUAL_REVIEW / identity_ambiguous (too close to auto-pick).
    - Zero SAVE gates and every review is REJECT with subreason fit_reject → REJECT / fit_reject for the lead.
    - Otherwise → MANUAL_REVIEW / ambiguity_unresolved (mixed signals, engagement blocks, etc.).
    """
    if not reviews:
        return MultiProfileOutcome(
            final_action="MANUAL_REVIEW",
            final_subreason="tool_failure",
            winner_index=None,
        )

    save_indices = [
        index
        for index, review in enumerate(reviews)
        if str(review.get("gate_final_action", "") or "").strip() == "SAVE"
    ]
    if len(save_indices) == 1:
        return MultiProfileOutcome(
            final_action="SAVE",
            final_subreason="",
            winner_index=save_indices[0],
        )
    if len(save_indices) > 1:
        return MultiProfileOutcome(
            final_action="MANUAL_REVIEW",
            final_subreason="identity_ambiguous",
            winner_index=None,
        )

    all_fit_reject = True
    for review in reviews:
        action = str(review.get("gate_final_action", "") or "").strip()
        sub = str(review.get("gate_final_subreason", "") or "").strip()
        if action != "REJECT" or sub != "fit_reject":
            all_fit_reject = False
            break

    if all_fit_reject:
        return MultiProfileOutcome(
            final_action="REJECT",
            final_subreason="fit_reject",
            winner_index=None,
        )

    return MultiProfileOutcome(
        final_action="MANUAL_REVIEW",
        final_subreason="ambiguity_unresolved",
        winner_index=None,
    )
