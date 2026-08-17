"""Typed provenance and groundedness checks for market-intelligence claims."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence


_STOPWORDS = {
    "about",
    "after",
    "among",
    "and",
    "are",
    "because",
    "being",
    "but",
    "can",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "not",
    "that",
    "the",
    "their",
    "there",
    "this",
    "through",
    "with",
}


class GroundednessStatus(str, Enum):
    GROUNDED = "grounded"
    PARTIAL = "partial"
    UNGROUNDED = "ungrounded"


@dataclass(frozen=True)
class EvidenceRef:
    """Typed pointer to evidence used by a market-intelligence claim."""

    source_id: str
    source_type: str
    locator: str
    quote: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: "EvidenceRef | Mapping[str, Any] | str") -> "EvidenceRef":
        if isinstance(value, EvidenceRef):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                raise ValueError("evidence ref string must be non-empty")
            prefix, _, locator = raw.partition(":")
            return cls(
                source_id=raw,
                source_type=prefix or "unknown",
                locator=locator or raw,
            )
        if not isinstance(value, Mapping):
            raise ValueError("evidence ref must be a string or object")
        source_id = _optional_ref_string(
            value.get("source_id") or value.get("id"),
            field="source_id",
        )
        source_type = _optional_ref_string(
            value.get("source_type") or value.get("type"),
            field="source_type",
        )
        locator = _optional_ref_string(
            value.get("locator") or value.get("url") or source_id,
            field="locator",
        )
        quote = _optional_ref_string(
            value.get("quote") or value.get("excerpt"),
            field="quote",
        )
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("evidence ref metadata must be an object")
        if not source_id:
            raise ValueError("evidence ref requires source_id")
        if not source_type:
            raise ValueError("evidence ref requires source_type")
        if not locator:
            raise ValueError("evidence ref requires locator")
        return cls(
            source_id=source_id,
            source_type=source_type,
            locator=locator,
            quote=quote,
            metadata=dict(metadata),
        )

    def support_text(self) -> str:
        metadata_text = " ".join(
            value
            for value in self.metadata.values()
            if isinstance(value, str)
        )
        return " ".join(part for part in (self.quote, metadata_text) if part)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "locator": self.locator,
            "quote": self.quote,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GroundednessVerdict:
    claim_id: str
    status: GroundednessStatus
    supported_ref_ids: tuple[str, ...]
    missing_terms: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "status": self.status.value,
            "supported_ref_ids": list(self.supported_ref_ids),
            "missing_terms": list(self.missing_terms),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class MarketClaim:
    claim_id: str
    text: str
    evidence_refs: tuple[EvidenceRef, ...]
    groundedness: GroundednessVerdict | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, default_id: str) -> "MarketClaim":
        if not isinstance(value, Mapping):
            raise ValueError("market claim must be an object")
        raw_refs = value.get("evidence_refs", ())
        if not isinstance(raw_refs, Sequence) or isinstance(raw_refs, (str, bytes)):
            raise ValueError("evidence_refs must be a sequence")
        claim_id = _optional_claim_string(
            value.get("claim_id") or value.get("id") or default_id,
            field="claim_id",
        )
        text = _optional_claim_string(
            value.get("text") or value.get("claim"),
            field="claim text",
        )
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            claim_id=claim_id,
            text=text,
            evidence_refs=tuple(EvidenceRef.from_value(ref) for ref in raw_refs),
            metadata=dict(metadata),
        )

    def with_groundedness(self, verdict: GroundednessVerdict) -> "MarketClaim":
        return MarketClaim(
            claim_id=self.claim_id,
            text=self.text,
            evidence_refs=self.evidence_refs,
            groundedness=verdict,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "groundedness": self.groundedness.to_dict() if self.groundedness else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GroundednessReport:
    claims: tuple[MarketClaim, ...]
    grounded_claims: tuple[MarketClaim, ...]
    quarantined_claims: tuple[MarketClaim, ...]

    @property
    def status(self) -> str:
        return "ok" if not self.quarantined_claims else "quarantine"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "claims": [claim.to_dict() for claim in self.claims],
            "grounded_claims": [claim.claim_id for claim in self.grounded_claims],
            "quarantined_claims": [claim.to_dict() for claim in self.quarantined_claims],
        }


def evaluate_claim_groundedness(
    claim: MarketClaim,
    *,
    minimum_term_coverage: float = 0.6,
) -> GroundednessVerdict:
    """Return a groundedness verdict without silently dropping weak claims."""

    minimum_term_coverage = _require_minimum_term_coverage(minimum_term_coverage)
    claim_terms = _meaningful_terms(claim.text)
    if not claim.evidence_refs:
        return GroundednessVerdict(
            claim_id=claim.claim_id,
            status=GroundednessStatus.UNGROUNDED,
            supported_ref_ids=(),
            missing_terms=claim_terms,
            rationale="No evidence refs were supplied.",
        )

    support_by_ref = {
        ref.source_id: set(_meaningful_terms(ref.support_text()))
        for ref in claim.evidence_refs
    }
    combined_support = set().union(*support_by_ref.values()) if support_by_ref else set()
    supported_terms = tuple(term for term in claim_terms if term in combined_support)
    missing_terms = tuple(term for term in claim_terms if term not in combined_support)
    supported_ref_ids = tuple(
        ref_id for ref_id, terms in support_by_ref.items() if terms.intersection(claim_terms)
    )

    if not claim_terms:
        status = GroundednessStatus.UNGROUNDED
        rationale = "Claim text did not contain groundedness-checkable terms."
    else:
        coverage = len(supported_terms) / len(claim_terms)
        if coverage >= minimum_term_coverage:
            status = GroundednessStatus.GROUNDED
            rationale = "Evidence refs cover the claim's material terms."
        elif supported_terms:
            status = GroundednessStatus.PARTIAL
            rationale = "Evidence refs support only part of the claim."
        else:
            status = GroundednessStatus.UNGROUNDED
            rationale = "Evidence refs do not support the claim's material terms."

    return GroundednessVerdict(
        claim_id=claim.claim_id,
        status=status,
        supported_ref_ids=supported_ref_ids,
        missing_terms=missing_terms,
        rationale=rationale,
    )


def ground_market_claims(
    claims: Sequence[MarketClaim | Mapping[str, Any]],
    *,
    minimum_term_coverage: float = 0.6,
) -> GroundednessReport:
    """Attach groundedness verdicts and quarantine unsupported claims."""

    if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
        raise ValueError("claims must be a sequence")
    minimum_term_coverage = _require_minimum_term_coverage(minimum_term_coverage)
    evaluated: list[MarketClaim] = []
    grounded: list[MarketClaim] = []
    quarantined: list[MarketClaim] = []
    for index, raw_claim in enumerate(claims):
        if isinstance(raw_claim, MarketClaim):
            claim = raw_claim
        elif isinstance(raw_claim, Mapping):
            claim = MarketClaim.from_mapping(
                raw_claim,
                default_id=f"claim-{index + 1}",
            )
        else:
            raise ValueError(f"claims[{index}] must be an object")
        verdict = evaluate_claim_groundedness(
            claim,
            minimum_term_coverage=minimum_term_coverage,
        )
        claim_with_verdict = claim.with_groundedness(verdict)
        evaluated.append(claim_with_verdict)
        if verdict.status == GroundednessStatus.GROUNDED:
            grounded.append(claim_with_verdict)
        else:
            quarantined.append(claim_with_verdict)
    return GroundednessReport(
        claims=tuple(evaluated),
        grounded_claims=tuple(grounded),
        quarantined_claims=tuple(quarantined),
    )


def _meaningful_terms(text: str) -> tuple[str, ...]:
    raw_terms: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
            continue
        if current:
            raw_terms.append("".join(current))
            current = []
    if current:
        raw_terms.append("".join(current))
    deduped: list[str] = []
    for term in raw_terms:
        if len(term) < 3 or term in _STOPWORDS:
            continue
        if term not in deduped:
            deduped.append(term)
    return tuple(deduped)


def _optional_ref_string(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"evidence ref {field} must be a string")
    return value.strip()


def _optional_claim_string(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value.strip()


def _require_minimum_term_coverage(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("minimum_term_coverage must be a number between 0 and 1")
    coverage = float(value)
    if not math.isfinite(coverage) or coverage < 0.0 or coverage > 1.0:
        raise ValueError("minimum_term_coverage must be a number between 0 and 1")
    return coverage
