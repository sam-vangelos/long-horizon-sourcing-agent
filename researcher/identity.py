"""Researcher identity disambiguation — Slice 4.

After acquisition surfaces N candidates per query, the disambiguation
pass filters them against the brief's hard constraints and resolves
common-name collisions. Per Researcher Module Spec Slice 4:

- Filter by ``ror_country`` (geography) — drop candidates whose
  affiliations don't match any allowed country.
- Filter by ``topic_concept`` — drop candidates whose published
  concepts don't overlap the query's concepts (defensive against
  OpenAlex's own filter slop).
- Filter by ``papers_in_window_floor`` — pre-LLM gate to avoid spending
  facial-eval budget on zero-publication candidates.
- ORCID-anchored identity when present (``orcid:{...}``); otherwise
  ``openalex:{author_id}`` per Spec Opinion 3.
- Common-name collisions (≥2 candidates remain after filters with the
  same normalized name) get flagged for ``INFERENTIAL_SAVE`` with a
  manual-review note in their disambiguation result.

The disambiguator does NOT mutate candidates; it returns a list of
:class:`DisambiguationResult` records that wrap each candidate with
the resolved identity key + flags.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from researcher.schemas import (
    ResearcherCandidate,
    identity_key_for_candidate,
)


@dataclass
class DisambiguationResult:
    """One candidate's disambiguation outcome."""

    candidate: ResearcherCandidate
    identity_key: str
    kept: bool
    rejection_reason: str = ""
    needs_manual_review: bool = False
    review_note: str = ""

    def to_dict(self) -> dict:
        return {
            "identity_key": self.identity_key,
            "kept": self.kept,
            "rejection_reason": self.rejection_reason,
            "needs_manual_review": self.needs_manual_review,
            "review_note": self.review_note,
            "candidate": self.candidate.to_dict(),
        }


@dataclass
class DisambiguationSummary:
    """Aggregate counters per disambiguation pass."""

    input_count: int = 0
    kept_count: int = 0
    dropped_country: int = 0
    dropped_concept: int = 0
    dropped_papers_in_window: int = 0
    dropped_missing_identity: int = 0
    flagged_common_name: int = 0
    results: list[DisambiguationResult] = field(default_factory=list)


def disambiguate(
    candidates: Iterable[ResearcherCandidate],
    *,
    allowed_country_codes: list[str] | None = None,
    required_concept_ids: list[str] | None = None,
    papers_in_window_floor: int = 0,
) -> DisambiguationSummary:
    """Run the disambiguation pass and return a summary.

    All filters are AND'd. Empty filter lists mean "no constraint" for
    that axis (e.g., empty ``allowed_country_codes`` ⇒ accept any
    country).
    """

    candidates_list = list(candidates)
    summary = DisambiguationSummary(input_count=len(candidates_list))

    allowed_countries = {c.upper() for c in (allowed_country_codes or [])}
    required_concepts = {c for c in (required_concept_ids or [])}

    # First pass: per-candidate filters.
    surviving: list[DisambiguationResult] = []
    for candidate in candidates_list:
        identity_key = identity_key_for_candidate(candidate)
        if not identity_key:
            summary.dropped_missing_identity += 1
            summary.results.append(
                DisambiguationResult(
                    candidate=candidate,
                    identity_key="",
                    kept=False,
                    rejection_reason="no_identity_key",
                )
            )
            continue

        if allowed_countries and not _country_matches(candidate, allowed_countries):
            summary.dropped_country += 1
            summary.results.append(
                DisambiguationResult(
                    candidate=candidate,
                    identity_key=identity_key,
                    kept=False,
                    rejection_reason="country_filter",
                )
            )
            continue

        if required_concepts and not _concept_overlap(candidate, required_concepts):
            summary.dropped_concept += 1
            summary.results.append(
                DisambiguationResult(
                    candidate=candidate,
                    identity_key=identity_key,
                    kept=False,
                    rejection_reason="concept_filter",
                )
            )
            continue

        if (
            papers_in_window_floor > 0
            and candidate.papers_in_window < papers_in_window_floor
        ):
            summary.dropped_papers_in_window += 1
            summary.results.append(
                DisambiguationResult(
                    candidate=candidate,
                    identity_key=identity_key,
                    kept=False,
                    rejection_reason="papers_in_window_below_floor",
                )
            )
            continue

        surviving.append(
            DisambiguationResult(
                candidate=candidate,
                identity_key=identity_key,
                kept=True,
            )
        )

    # Second pass: common-name collision flagging. Two candidates with
    # the same normalized name AND no ORCID anchor are ambiguous; flag
    # both for manual review at SAVE time.
    by_name: dict[str, list[DisambiguationResult]] = defaultdict(list)
    for result in surviving:
        normalized = _normalize_name(result.candidate.name)
        if normalized:
            by_name[normalized].append(result)

    for normalized, group in by_name.items():
        if len(group) < 2:
            continue
        # If at least one in the group has an ORCID anchor, the
        # ORCID-anchored ones are confidently distinct; only the
        # ORCID-less ones are ambiguous.
        for result in group:
            if not result.identity_key.startswith("orcid:"):
                result.needs_manual_review = True
                result.review_note = (
                    f"common_name_collision name={normalized!r} "
                    f"colliding_count={len(group)} — ORCID missing; "
                    "evaluator should route to INFERENTIAL_SAVE."
                )
                summary.flagged_common_name += 1

    summary.kept_count = len(surviving)
    summary.results.extend(surviving) if False else None  # kept results already in `surviving`
    # Replace summary.results: keep filtered + the SURVIVING (not duplicated).
    # Re-build to put kept results last so the order is deterministic
    # (filtered-out first by reason, then kept).
    rebuilt: list[DisambiguationResult] = []
    rebuilt.extend(r for r in summary.results if not r.kept)
    rebuilt.extend(surviving)
    summary.results = rebuilt

    return summary


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------


def _country_matches(
    candidate: ResearcherCandidate,
    allowed_countries: set[str],
) -> bool:
    """True if any of the candidate's affiliations carries an allowed
    country code. Affiliations come in as ``"MIT (US)"`` style strings
    from acquisition; we extract the parenthesized ISO code.
    """

    for affiliation in candidate.affiliations:
        match = _COUNTRY_RE.search(affiliation)
        if match and match.group(1).upper() in allowed_countries:
            return True
    # Defensive: the OpenAlex raw payload may carry the country code
    # even when our extracted string elided it. Check there too.
    raw = candidate.raw_openalex or {}
    for inst in raw.get("last_known_institutions") or []:
        if not isinstance(inst, dict):
            continue
        country = str(inst.get("country_code") or "").upper()
        if country in allowed_countries:
            return True
    return False


_COUNTRY_RE = re.compile(r"\(([A-Z]{2})\)$")


def _concept_overlap(
    candidate: ResearcherCandidate,
    required_concepts: set[str],
) -> bool:
    """True if any of the candidate's OpenAlex x_concepts overlap the
    required set. Defensive against absent ``x_concepts`` (acquisition
    keeps the raw payload around).
    """

    raw = candidate.raw_openalex or {}
    candidate_concepts = {
        str(c.get("id") or "").rsplit("/", 1)[-1]
        for c in raw.get("x_concepts") or []
        if isinstance(c, dict) and c.get("id")
    }
    return bool(candidate_concepts & required_concepts)


_NAME_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_name(name: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation. Used for
    collision detection only; not a stable identifier."""

    if not name:
        return ""
    lowered = name.strip().lower()
    cleaned = _NAME_NORMALIZE_RE.sub(" ", lowered)
    return " ".join(cleaned.split())
