"""Deterministic normalizer for raw Perplexity candidate-evidence output.

The contract:

- Parses the structured Perplexity JSON payload into ``ExternalCandidateEvidence``.
- Preserves the strict separation between sourced facts, model inferences, and
  unresolved ambiguities — including inferences that lack ``basis_refs``.
- Returns ``ExternalEvidenceFailure`` typed results on parse failure or weak
  citation count. Never raises out to callers.

Slice 1 does NOT invoke a cheap-model second pass: the JSON-Schema-shaped
Perplexity ``response_format`` is structured enough to parse deterministically.
If a future slice needs cheap-model normalization to enforce fact/inference/
ambiguity separation, it must classify any ``cheap_llm`` exception as
``reason="normalizer_failure"`` rather than re-raising.
"""

from __future__ import annotations

import json
from typing import Any

import shared.config as config
from shared.schemas import (
    EvidenceRef,
    ExternalCandidateEvidence,
    ExternalEvidenceFailure,
    ExternalFactBlock,
    ExternalInference,
)


_ALLOWED_QUALITY: frozenset[str] = frozenset({"high", "medium", "low", "unknown"})


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_quality(value: Any) -> str:
    text = _coerce_str(value).lower()
    if text in _ALLOWED_QUALITY:
        return text
    return "unknown"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _coerce_str(item)
        if text:
            out.append(text)
    return out


def _coerce_evidence_ref(item: Any) -> EvidenceRef | None:
    if isinstance(item, str):
        url = item.strip()
        if not url:
            return None
        return EvidenceRef(url=url, title="", source_quality="unknown")
    if not isinstance(item, dict):
        return None
    url = _coerce_str(item.get("url"))
    if not url:
        return None
    return EvidenceRef(
        url=url,
        title=_coerce_str(item.get("title")),
        source_quality=_coerce_quality(item.get("source_quality")),
    )


def _coerce_evidence_refs(value: Any) -> list[EvidenceRef]:
    if not isinstance(value, list):
        return []
    refs: list[EvidenceRef] = []
    for entry in value:
        ref = _coerce_evidence_ref(entry)
        if ref is not None:
            refs.append(ref)
    return refs


def _coerce_fact_block(value: Any) -> ExternalFactBlock | None:
    if not isinstance(value, dict):
        return None
    return ExternalFactBlock(
        topic=_coerce_str(value.get("topic")),
        facts=_coerce_str_list(value.get("facts")),
        evidence_refs=_coerce_evidence_refs(value.get("evidence_refs")),
        source_quality=_coerce_quality(value.get("source_quality")),
    )


def _coerce_inference(value: Any) -> ExternalInference | None:
    if not isinstance(value, dict):
        return None
    claim = _coerce_str(value.get("claim"))
    if not claim:
        return None
    # Inferences without basis_refs are explicitly preserved (uncertainty
    # is part of the contract; we do not silently drop the inference).
    return ExternalInference(
        claim=claim,
        basis_refs=_coerce_evidence_refs(value.get("basis_refs")),
        confidence=_coerce_float(value.get("confidence"), 0.0),
    )


def _total_citation_count(
    fact_blocks: list[ExternalFactBlock],
    inferences: list[ExternalInference],
) -> int:
    total = 0
    for block in fact_blocks:
        total += len(block.evidence_refs)
    for inference in inferences:
        total += len(inference.basis_refs)
    return total


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines[1:] if not line.strip().startswith("```")
        )
    return cleaned.strip()


def _try_parse_json(text: str) -> dict | None:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def normalize_perplexity_response(
    *,
    raw_text: str,
    raw_sources: list[dict],
    trigger_reason: str,
    raw_provider_model: str,
    min_citations: int | None = None,
) -> ExternalCandidateEvidence | ExternalEvidenceFailure:
    """Parse a Perplexity candidate-evidence response into the structured contract.

    ``raw_sources`` is accepted for forward compatibility (e.g. a future slice
    may cross-reference URLs from the structured ``output`` section against the
    declarative ``response_format`` payload). Slice 1 does not yet need it.

    ``min_citations`` defaults to ``config.LINKEDIN_EXTERNAL_EVIDENCE_MIN_CITATIONS``
    when not supplied; the keyword exists so tests can pin the threshold without
    mutating global env state.
    """

    del raw_sources  # forward-compatibility hook — see docstring.

    threshold = (
        int(min_citations)
        if min_citations is not None
        else int(config.LINKEDIN_EXTERNAL_EVIDENCE_MIN_CITATIONS)
    )

    parsed = _try_parse_json(raw_text or "")
    if parsed is None:
        snippet = (raw_text or "")[:240]
        return ExternalEvidenceFailure(
            reason="parse_failure",
            detail=f"could not parse perplexity response as JSON: {snippet!r}",
            provider="perplexity",
        )

    fact_blocks_raw = parsed.get("external_fact_blocks") or []
    inferences_raw = parsed.get("external_inferences") or []

    fact_blocks: list[ExternalFactBlock] = []
    if isinstance(fact_blocks_raw, list):
        for entry in fact_blocks_raw:
            block = _coerce_fact_block(entry)
            if block is not None:
                fact_blocks.append(block)

    inferences: list[ExternalInference] = []
    if isinstance(inferences_raw, list):
        for entry in inferences_raw:
            inference = _coerce_inference(entry)
            if inference is not None:
                inferences.append(inference)

    citation_count = _total_citation_count(fact_blocks, inferences)
    if citation_count < threshold:
        return ExternalEvidenceFailure(
            reason="weak_citations",
            detail=(
                f"citation_count={citation_count} below "
                f"min_citations={threshold}"
            ),
            provider="perplexity",
        )

    return ExternalCandidateEvidence(
        trigger_reason=trigger_reason or _coerce_str(parsed.get("trigger_reason")),
        identity_confidence=_coerce_float(parsed.get("identity_confidence"), 0.0),
        profile_facts_used_for_matching=_coerce_str_list(
            parsed.get("profile_facts_used_for_matching")
        ),
        external_fact_blocks=fact_blocks,
        external_inferences=inferences,
        unresolved_ambiguities=_coerce_str_list(parsed.get("unresolved_ambiguities")),
        do_not_use_for_judgment=_coerce_str_list(parsed.get("do_not_use_for_judgment")),
        raw_provider_model=raw_provider_model,
        normalizer_model="",
    )
