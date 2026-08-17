"""LinkedIn fallback search — P10 (C4-C6).

Data model, provider protocol, and identity resolution for public LinkedIn
and x-ray fallback acquisition.  Fallback candidates are identity candidates
until resolved back to a Recruiter profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

RESOLUTION_STATES = frozenset({
    "unresolved",
    "candidate_match",
    "resolved_recruiter_profile",
    "rejected_match",
})

RESOLUTION_CONFIDENCE_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# C4: Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FallbackCandidate:
    """A candidate discovered via public LinkedIn or x-ray search.

    ``save_eligible`` is a computed property derived from ``resolution_state``.
    It cannot be set directly.
    """

    source_mode: Literal["linkedin_public", "xray"]
    source_url: str
    name_hint: str | None = None
    headline_hint: str | None = None
    evidence_snippets: tuple[str, ...] = ()
    lane_id: str = ""
    variant_id: str | None = None
    recruiter_profile_id: str | None = None
    resolution_state: str = "unresolved"

    @property
    def save_eligible(self) -> bool:
        return self.resolution_state == "resolved_recruiter_profile"

    def to_persistence_dict(self) -> dict[str, Any]:
        return {
            "source_mode": self.source_mode,
            "source_url": self.source_url,
            "name_hint": self.name_hint,
            "headline_hint": self.headline_hint,
            "evidence_snippets": list(self.evidence_snippets),
            "lane_id": self.lane_id,
            "variant_id": self.variant_id,
            "recruiter_profile_id": self.recruiter_profile_id,
            "resolution_state": self.resolution_state,
        }

    @classmethod
    def from_persistence_dict(cls, payload: dict[str, Any]) -> FallbackCandidate:
        return cls(
            source_mode=payload.get("source_mode", "xray"),
            source_url=payload.get("source_url", ""),
            name_hint=payload.get("name_hint"),
            headline_hint=payload.get("headline_hint"),
            evidence_snippets=tuple(payload.get("evidence_snippets", [])),
            lane_id=payload.get("lane_id", ""),
            variant_id=payload.get("variant_id"),
            recruiter_profile_id=payload.get("recruiter_profile_id"),
            resolution_state=payload.get("resolution_state", "unresolved"),
        )


def validate_fallback_candidate(candidate: FallbackCandidate) -> list[str]:
    """Return a list of validation errors (empty if valid)."""
    errors: list[str] = []
    if candidate.resolution_state not in RESOLUTION_STATES:
        errors.append(f"Unknown resolution_state: {candidate.resolution_state}")
    if candidate.resolution_state == "resolved_recruiter_profile" and not candidate.recruiter_profile_id:
        errors.append("resolved_recruiter_profile requires a recruiter_profile_id")
    return errors


# ---------------------------------------------------------------------------
# C5: Provider protocol and query builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FallbackQuery:
    """A search query for fallback acquisition."""

    query_string: str
    lane_id: str
    variant_id: str | None = None
    source_mode: Literal["linkedin_public", "xray"] = "xray"
    constraints_applied: tuple[str, ...] = ()


@dataclass(frozen=True)
class FallbackResult:
    """A single result from a fallback search provider."""

    url: str
    title: str = ""
    snippet: str = ""
    source_mode: Literal["linkedin_public", "xray"] = "xray"


@runtime_checkable
class FallbackSearchProvider(Protocol):
    def search(self, query: FallbackQuery, *, limit: int = 10) -> list[FallbackResult]: ...


class FakeFallbackProvider:
    """Deterministic provider for tests."""

    def __init__(self, results: list[FallbackResult] | None = None):
        self.results = results or []
        self.queries_received: list[FallbackQuery] = []

    def search(self, query: FallbackQuery, *, limit: int = 10) -> list[FallbackResult]:
        self.queries_received.append(query)
        return self.results[:limit]


def build_xray_query(
    lane: Any,
    variant: Any | None = None,
) -> FallbackQuery:
    """Build a site:linkedin.com/in x-ray query from lane constraints."""
    terms: list[str] = []
    applied: list[str] = []

    constraints = getattr(getattr(lane, "slice", None), "constraints", []) or []
    for c in constraints:
        dim = getattr(c, "dimension", "")
        vals = getattr(c, "values", []) or []
        if dim in ("capability", "title", "entry_signal") and vals:
            terms.extend(f'"{v}"' for v in vals[:3])
            applied.append(dim)
        elif dim == "location" and vals:
            terms.append(f'"{vals[0]}"')
            applied.append(dim)

    if not terms:
        name = getattr(lane, "lane_name", "") or "fallback"
        terms.append(f'"{name}"')

    query_string = "site:linkedin.com/in " + " ".join(terms)
    return FallbackQuery(
        query_string=query_string,
        lane_id=getattr(lane, "lane_id", "") or "",
        variant_id=getattr(variant, "variant_id", None) if variant else None,
        source_mode="xray",
        constraints_applied=tuple(applied),
    )


def fallback_result_to_candidate(
    result: FallbackResult,
    *,
    lane_id: str,
    variant_id: str | None = None,
) -> FallbackCandidate:
    """Convert a provider result into a FallbackCandidate (always unresolved)."""
    name_hint = None
    headline_hint = None
    if result.title:
        parts = result.title.split(" - ", 1)
        name_hint = parts[0].strip() if parts else None
        headline_hint = parts[1].strip() if len(parts) > 1 else None

    return FallbackCandidate(
        source_mode=result.source_mode,
        source_url=result.url,
        name_hint=name_hint,
        headline_hint=headline_hint,
        evidence_snippets=(result.snippet,) if result.snippet else (),
        lane_id=lane_id,
        variant_id=variant_id,
        resolution_state="unresolved",
    )


# ---------------------------------------------------------------------------
# C6: Identity resolution and save guard
# ---------------------------------------------------------------------------


def resolve_fallback_candidate(
    candidate: FallbackCandidate,
    *,
    recruiter_profile_id: str | None = None,
    match_confidence: float = 0.0,
    rejection_reason: str = "",
) -> FallbackCandidate:
    """Transition a fallback candidate's resolution state.

    Returns a new FallbackCandidate (frozen dataclass) with updated state.
    """
    if rejection_reason:
        return FallbackCandidate(
            source_mode=candidate.source_mode,
            source_url=candidate.source_url,
            name_hint=candidate.name_hint,
            headline_hint=candidate.headline_hint,
            evidence_snippets=candidate.evidence_snippets,
            lane_id=candidate.lane_id,
            variant_id=candidate.variant_id,
            recruiter_profile_id=candidate.recruiter_profile_id,
            resolution_state="rejected_match",
        )

    if recruiter_profile_id:
        if match_confidence >= RESOLUTION_CONFIDENCE_THRESHOLD:
            new_state = "resolved_recruiter_profile"
        else:
            new_state = "candidate_match"
        return FallbackCandidate(
            source_mode=candidate.source_mode,
            source_url=candidate.source_url,
            name_hint=candidate.name_hint,
            headline_hint=candidate.headline_hint,
            evidence_snippets=candidate.evidence_snippets,
            lane_id=candidate.lane_id,
            variant_id=candidate.variant_id,
            recruiter_profile_id=recruiter_profile_id,
            resolution_state=new_state,
        )

    return candidate


def assert_fallback_save_safety(candidate: FallbackCandidate) -> None:
    """Raise ValueError if a fallback candidate is not safe to save.

    Defensive guard — call sites: anywhere a save path is entered for a
    fallback-sourced candidate.
    """
    if candidate.resolution_state != "resolved_recruiter_profile":
        raise ValueError(
            f"Fallback candidate {candidate.source_url} cannot be saved: "
            f"resolution_state={candidate.resolution_state}"
        )
    if not candidate.recruiter_profile_id:
        raise ValueError(
            f"Fallback candidate {candidate.source_url} has resolved state "
            f"but no recruiter_profile_id"
        )
