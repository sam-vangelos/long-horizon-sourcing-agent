"""Candidate-evidence Perplexity provider for the LinkedIn full-eval path.

Scope (slice 1):

- Explicit three-attempt Perplexity call for transient failures.
- Returns ``ExternalCandidateEvidence`` on success or ``ExternalEvidenceFailure``
  on any failure. Never raises.
- Quota-shaped errors return ``reason="quota_exhausted"``. They do NOT call into
  ``shared.failures.is_api_budget_exhausted_error`` and do NOT raise
  ``ApiBudgetExhaustedError``: candidate-level external evidence MUST NOT pause
  a LinkedIn run.
- Has zero callers in slice 1 — slice 2 will wire the trigger gate to this
  provider before the final judge.

Patterns for response shape extraction are reimplemented locally here, mirroring
``market_intelligence/research_agent.py``'s ``_extract_perplexity_text`` /
``_extract_perplexity_sources`` — but **not imported** from market_intelligence,
because candidate-level evidence has different failure semantics from market
research.
"""

from __future__ import annotations

import json
from typing import Any

import shared.config as config
from shared.brief_schema import Brief
from shared.failures import classify_runtime_failure
from shared.llm_clients import _retry_with_backoff, get_llm_client
from shared.llm_usage import openai_usage_dict, record_llm_usage
from shared.schemas import (
    CandidateProfileSummary,
    ExternalCandidateEvidence,
    ExternalEvidenceFailure,
    TriggerDecision,
)

from shared.external_evidence.normalizer import normalize_perplexity_response


_QUOTA_MARKERS: tuple[str, ...] = (
    "credit balance is too low",
    "insufficient_quota",
    "insufficient credits",
    "purchase credits",
    "exceeded your current quota",
    "quota exceeded",
    "billing hard limit",
    "out of credit",
    "credit",
    "quota",
    "billing",
    "balance",
)


_TIMEOUT_MARKERS: tuple[str, ...] = (
    "timeout",
    "timed out",
)


def _extract_status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        sc = getattr(response, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def _is_quota_exhausted(exc: BaseException, status: int | None) -> bool:
    if status == 402:
        return True
    text = (str(exc) or exc.__class__.__name__).lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    text = (str(exc) or exc.__class__.__name__).lower()
    return any(marker in text for marker in _TIMEOUT_MARKERS)


def _is_retryable_perplexity_error(exc: Exception) -> bool:
    status = _extract_status_code(exc)
    if _is_quota_exhausted(exc, status):
        return False
    return (
        status == 429
        or status is not None
        and 500 <= status < 600
        or classify_runtime_failure(exc, source="llm").domain == "network"
    )


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _record_external_evidence_llm_error(
    *,
    model: str,
    exc: Exception,
    max_tokens: int,
    instructions_chars: int,
    input_chars: int,
    trigger_reason: str,
) -> None:
    """Best-effort typed LLM error receipt for candidate external evidence."""

    try:
        record_llm_usage(
            provider="perplexity",
            model=model or "perplexity",
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            request={
                "max_tokens": max_tokens,
                "instructions_chars": instructions_chars,
                "input_chars": input_chars,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:240],
            },
            usage_context={
                "stage": "linkedin_external_evidence",
                "trigger_reason": trigger_reason,
            },
            actual_status="error",
        )
    except Exception:  # noqa: BLE001 — telemetry must not break evidence lookup
        pass


def _iter_perplexity_output_items(response: Any) -> list[Any]:
    if isinstance(response, dict):
        output = response.get("output")
        if isinstance(output, list):
            return list(output)
    output = getattr(response, "output", None)
    if isinstance(output, list):
        return list(output)
    if hasattr(response, "model_dump"):
        try:
            payload = response.model_dump()
        except Exception:
            return []
        if isinstance(payload, dict) and isinstance(payload.get("output"), list):
            return list(payload["output"])
    return []


def _item_field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _item_list_field(item: Any, name: str) -> list[Any]:
    value = _item_field(item, name)
    if isinstance(value, list):
        return value
    return []


def _item_type(item: Any) -> str:
    return _normalize_text(_item_field(item, "type"))


def _extract_perplexity_text(response: Any) -> str:
    if isinstance(response, dict):
        output_text = response.get("output_text", "")
        if output_text:
            return str(output_text)
    output_text = getattr(response, "output_text", "")
    if output_text:
        return str(output_text)
    texts: list[str] = []
    for item in _iter_perplexity_output_items(response):
        if _item_type(item) != "message":
            continue
        for content in _item_list_field(item, "content"):
            if _item_type(content) != "output_text":
                continue
            text = _normalize_text(_item_field(content, "text"))
            if text:
                texts.append(text)
    return "\n".join(texts)


def _extract_perplexity_sources(response: Any) -> list[dict]:
    sources: list[dict] = []
    seen: set[str] = set()
    for item in _iter_perplexity_output_items(response):
        item_type = _item_type(item)
        if item_type == "search_results":
            for result in _item_list_field(item, "results"):
                url = _normalize_text(_item_field(result, "url"))
                if not url or url in seen:
                    continue
                seen.add(url)
                sources.append(
                    {
                        "url": url,
                        "title": _normalize_text(_item_field(result, "title")),
                        "kind": "web_search",
                    }
                )
        elif item_type == "fetch_url_results":
            for result in _item_list_field(item, "contents"):
                url = _normalize_text(_item_field(result, "url"))
                if not url or url in seen:
                    continue
                seen.add(url)
                sources.append(
                    {
                        "url": url,
                        "title": _normalize_text(_item_field(result, "title")),
                        "kind": "fetch_url",
                    }
                )
    return sources


def _build_response_format() -> dict:
    """JSON-Schema-shaped response format for Perplexity structured output.

    Maps 1:1 to ``ExternalCandidateEvidence``'s public fields (excluding
    ``raw_provider_model`` / ``normalizer_model`` which are filled locally).
    Kept permissive (no ``required``, no ``additionalProperties: false``) so
    the deterministic normalizer can enforce strictness on parsed output.
    """

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "external_candidate_evidence",
            "schema": {
                "type": "object",
                "properties": {
                    "trigger_reason": {"type": "string"},
                    "identity_confidence": {"type": "number"},
                    "profile_facts_used_for_matching": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "external_fact_blocks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "facts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "evidence_refs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "url": {"type": "string"},
                                            "title": {"type": "string"},
                                            "source_quality": {"type": "string"},
                                        },
                                    },
                                },
                                "source_quality": {"type": "string"},
                            },
                        },
                    },
                    "external_inferences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string"},
                                "basis_refs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "url": {"type": "string"},
                                            "title": {"type": "string"},
                                            "source_quality": {"type": "string"},
                                        },
                                    },
                                },
                                "confidence": {"type": "number"},
                            },
                        },
                    },
                    "unresolved_ambiguities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "do_not_use_for_judgment": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    }


_INSTRUCTIONS = (
    "You are an evidence retrieval assistant for a recruiting evaluation system. "
    "You are NOT making a save/reject judgment about the candidate. Your only job "
    "is to retrieve cited public-web facts about the candidate's employers, lab/"
    "school context, theses, publications, and adjacent public work, then return "
    "them in a strict JSON contract.\n\n"
    "You MUST keep three categories explicitly separate in your output:\n"
    "  1. external_fact_blocks: sourced facts ONLY. Every fact must be backed by "
    "an evidence_ref with an exact URL.\n"
    "  2. external_inferences: model-synthesized claims derived from sourced "
    "facts. Every inference must list its basis_refs.\n"
    "  3. unresolved_ambiguities: anything you cannot confidently attribute to "
    "this specific candidate (common name collisions, ambiguous authorship, "
    "missing identity match) goes here, NOT into facts or inferences.\n\n"
    "If you cannot confidently identify the candidate from the supplied identity "
    "hints, lower identity_confidence and put the uncertainty into "
    "unresolved_ambiguities. Do NOT promote weakly-supported claims into facts. "
    "Do NOT recommend whether to save or reject. Return JSON only."
)


def _build_user_prompt(
    *,
    summary: CandidateProfileSummary,
    brief: Brief,
    trigger: TriggerDecision,
    identity_hints: dict | None,
) -> str:
    hints = identity_hints or {}
    identity_lines: list[str] = []
    for key in ("name", "current_company", "current_title", "school", "degree", "profile_url"):
        value = _normalize_text(hints.get(key))
        if value:
            identity_lines.append(f"- {key}: {value}")

    summary_payload = {
        "name": summary.name,
        "headline": summary.headline,
        "experiences": [
            {
                "title": exp.title,
                "company": exp.company,
                "start": exp.start,
                "end": exp.end,
                "summary_bullets": list(exp.summary_bullets),
            }
            for exp in summary.experiences
        ],
        "education": [
            {
                "degree": edu.degree,
                "school": edu.school,
                "field": edu.field,
                "start": edu.start,
                "end": edu.end,
            }
            for edu in summary.education
        ],
        "skills_snippet": list(summary.skills_snippet),
    }

    role_context = _normalize_text(brief.role_summary) or _normalize_text(brief.role_title)

    parts: list[str] = []
    parts.append(f"Trigger reason: {trigger.reason or 'unspecified'}")
    if role_context:
        parts.append(f"Role context (for relevance only, not for judgment): {role_context}")
    if identity_lines:
        parts.append("Identity hints:\n" + "\n".join(identity_lines))
    parts.append("First-party LinkedIn profile summary (JSON):")
    parts.append(json.dumps(summary_payload, ensure_ascii=False))
    parts.append(
        "Retrieve cited public-web facts about this candidate's employers, "
        "lab/school context, theses, publications, and adjacent public work. "
        "Return JSON matching the schema. Cite every sourced fact."
    )
    return "\n\n".join(parts)


def fetch_external_candidate_evidence(
    *,
    summary: CandidateProfileSummary,
    brief: Brief,
    trigger: TriggerDecision,
    identity_hints: dict | None = None,
) -> ExternalCandidateEvidence | ExternalEvidenceFailure:
    """Fetch external evidence for a candidate via Perplexity.

    Always returns a typed result. Never raises out to callers, including
    quota / timeout / network errors.
    """

    api_key = _normalize_text(getattr(config, "PERPLEXITY_API_KEY", ""))
    if not api_key:
        return ExternalEvidenceFailure(
            reason="disabled_no_api_key",
            detail="PERPLEXITY_API_KEY not set",
            provider="perplexity",
        )

    if not bool(getattr(config, "LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False)):
        return ExternalEvidenceFailure(
            reason="disabled_by_config",
            detail="LINKEDIN_EXTERNAL_EVIDENCE_ENABLED is false",
            provider="perplexity",
        )

    model_name = _normalize_text(getattr(config, "LINKEDIN_EXTERNAL_EVIDENCE_MODEL", ""))
    timeout_seconds = float(
        getattr(config, "LINKEDIN_EXTERNAL_EVIDENCE_TIMEOUT_SECONDS", 90.0) or 90.0
    )
    max_output_tokens = int(
        getattr(config, "LINKEDIN_EXTERNAL_EVIDENCE_MAX_OUTPUT_TOKENS", 4096) or 4096
    )
    preset = _normalize_text(
        getattr(config, "LINKEDIN_EXTERNAL_EVIDENCE_PERPLEXITY_PRESET", "")
    )

    instructions = _INSTRUCTIONS
    user_prompt = _build_user_prompt(
        summary=summary,
        brief=brief,
        trigger=trigger,
        identity_hints=identity_hints,
    )
    response_format = _build_response_format()

    try:
        client = get_llm_client("perplexity", api_key, timeout_seconds)

        extra_body: dict[str, Any] = {"response_format": response_format}
        if preset:
            extra_body["preset"] = preset

        kwargs: dict[str, Any] = {
            "input": user_prompt,
            "instructions": instructions,
            "max_output_tokens": max_output_tokens,
            "extra_body": extra_body,
        }
        if model_name:
            kwargs["model"] = model_name

        response = _retry_with_backoff(
            lambda: client.responses.create(**kwargs),
            label="Perplexity external evidence",
            max_attempts=3,
            is_retryable=_is_retryable_perplexity_error,
        )
    except Exception as exc:
        _record_external_evidence_llm_error(
            model=model_name or "perplexity",
            exc=exc,
            max_tokens=max_output_tokens,
            instructions_chars=len(instructions),
            input_chars=len(user_prompt),
            trigger_reason=trigger.reason,
        )
        status = _extract_status_code(exc)
        if _is_quota_exhausted(exc, status):
            return ExternalEvidenceFailure(
                reason="quota_exhausted",
                detail=str(exc)[:240],
                provider="perplexity",
                http_status=status,
            )
        if _is_timeout(exc):
            return ExternalEvidenceFailure(
                reason="timeout",
                detail=str(exc)[:240],
                provider="perplexity",
                http_status=status,
            )
        if status is not None:
            return ExternalEvidenceFailure(
                reason="http_error",
                detail=str(exc)[:240],
                provider="perplexity",
                http_status=status,
            )
        return ExternalEvidenceFailure(
            reason="unknown",
            detail=str(exc)[:240],
            provider="perplexity",
        )

    raw_text = _extract_perplexity_text(response)
    raw_sources = _extract_perplexity_sources(response)
    response_model = _normalize_text(getattr(response, "model", "")) or model_name or "perplexity"

    record_llm_usage(
        provider="perplexity",
        model=response_model,
        usage=openai_usage_dict(response),
        request={
            "max_tokens": max_output_tokens,
            "instructions_chars": len(instructions),
            "input_chars": len(user_prompt),
        },
        usage_context={
            "stage": "linkedin_external_evidence",
            "trigger_reason": trigger.reason,
        },
    )

    return normalize_perplexity_response(
        raw_text=raw_text,
        raw_sources=raw_sources,
        trigger_reason=trigger.reason,
        raw_provider_model=response_model,
    )
