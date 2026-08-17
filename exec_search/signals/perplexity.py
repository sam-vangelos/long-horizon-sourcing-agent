"""Perplexity signal adapter for executive-search dossier evaluation.

Wraps :func:`shared.external_evidence.provider.fetch_external_candidate_evidence`
— the same Perplexity surface the LinkedIn full-eval already uses for
external-evidence augmentation. Reuse over re-implementation; the
LinkedIn flow has burned in the failure semantics for quota / timeout
/ network errors.

Slice 3 ships this baseline. Slices 4-5 add news / Crunchbase /
PitchBook adapters that share the same :class:`ExecutiveSignalSource`
shape.

Configuration: this adapter inherits the LinkedIn external-evidence
config knobs (``LINKEDIN_EXTERNAL_EVIDENCE_*``) so a single feature
flag controls both surfaces. If the operator wants a separate budget
for exec-search Perplexity calls, that's a future enhancement (track
under :class:`exec_search.budget` once Slice 5 lands).
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.brief_schema import Brief
from shared.schemas import (
    CandidateProfileSummary,
    ExternalCandidateEvidence,
    ExternalEvidenceFailure,
    TriggerDecision,
)
from shared.external_evidence.provider import fetch_external_candidate_evidence

from exec_search.signals import SignalFailure, SignalRequestContext, SignalResult


@dataclass
class PerplexitySignalSource:
    """Off-LinkedIn signal source backed by Perplexity public-web search."""

    name: str = "perplexity"

    def fetch(
        self,
        *,
        candidate: CandidateProfileSummary,
        brief: Brief,
        context: SignalRequestContext,
    ) -> SignalResult | SignalFailure:
        trigger = TriggerDecision(
            should_run=True,
            reason=context.trigger_reason,
        )
        result = fetch_external_candidate_evidence(
            summary=candidate,
            brief=brief,
            trigger=trigger,
            identity_hints=dict(context.identity_hints) if context.identity_hints else None,
        )
        if isinstance(result, ExternalEvidenceFailure):
            return SignalFailure(
                source=self.name,
                reason=result.reason,
                detail=getattr(result, "detail", "") or result.reason,
            )
        if isinstance(result, ExternalCandidateEvidence):
            section_text = _format_perplexity_section(result)
            citations = tuple(_extract_citations(result))
            return SignalResult(
                source=self.name,
                section_text=section_text,
                citations=citations,
                raw_payload=None,
            )
        return SignalFailure(
            source=self.name,
            reason="unknown_provider_result_shape",
            detail=type(result).__name__,
        )


def _format_perplexity_section(evidence: ExternalCandidateEvidence) -> str:
    """Render the Perplexity evidence into a recruiter-readable section.

    The :class:`shared.schemas.ExternalCandidateEvidence` shape carries
    sourced fact blocks (with per-fact citations) and model-synthesized
    inferences. We render each fact block as a topic header + bullets
    and inferences as a separate block flagged as model-derived.
    """

    parts: list[str] = ["Off-LinkedIn signals (Perplexity):"]
    blocks = getattr(evidence, "external_fact_blocks", None) or []
    for block in blocks:
        topic = (getattr(block, "topic", "") or "").strip() or "Untitled"
        facts = [f for f in (getattr(block, "facts", None) or []) if f]
        if not facts:
            continue
        bullets = "\n".join(f"  - {fact}" for fact in facts)
        parts.append(f"### {topic}\n{bullets}")
    inferences = getattr(evidence, "external_inferences", None) or []
    if inferences:
        inference_lines: list[str] = ["### Model-derived inferences"]
        for inf in inferences:
            claim = (getattr(inf, "claim", "") or "").strip()
            if not claim:
                continue
            confidence = float(getattr(inf, "confidence", 0.0) or 0.0)
            inference_lines.append(f"  - {claim} (model confidence: {confidence:.2f})")
        if len(inference_lines) > 1:
            parts.append("\n".join(inference_lines))
    if len(parts) == 1:
        parts.append("[no off-LinkedIn signal returned by Perplexity]")
    citations = _extract_citations(evidence)
    if citations:
        citation_lines = "\n".join(f"  - {url}" for url in citations)
        parts.append(f"### Citations\n{citation_lines}")
    return "\n\n".join(parts)


def _extract_citations(evidence: ExternalCandidateEvidence) -> list[str]:
    """Collect every URL referenced by fact blocks or inferences.

    The :class:`ExternalCandidateEvidence` shape stores citations per
    fact block (``evidence_refs``) and per inference (``basis_refs``).
    We flatten across both, preserving order and de-duplicating.
    """

    seen: set[str] = set()
    ordered: list[str] = []
    for block in getattr(evidence, "external_fact_blocks", None) or []:
        for ref in getattr(block, "evidence_refs", None) or []:
            url = (getattr(ref, "url", "") or "").strip()
            if url and url not in seen:
                seen.add(url)
                ordered.append(url)
    for inf in getattr(evidence, "external_inferences", None) or []:
        for ref in getattr(inf, "basis_refs", None) or []:
            url = (getattr(ref, "url", "") or "").strip()
            if url and url not in seen:
                seen.add(url)
                ordered.append(url)
    return ordered


__all__ = ("PerplexitySignalSource",)
