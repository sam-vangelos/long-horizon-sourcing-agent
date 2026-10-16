"""External research backends for market intelligence."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import shared.config as config
from market_intelligence.engine import ExternalResearchResult
from market_intelligence.research_context import build_research_context_bundle
from market_intelligence.research_prompts import (
    build_perplexity_edge_case_research_instructions,
    build_perplexity_edge_case_research_response_format,
    build_perplexity_edge_case_research_user_prompt,
    build_perplexity_research_instructions,
    build_perplexity_research_response_format,
    build_perplexity_research_user_prompt,
    build_research_system_prompt,
    build_research_user_prompt,
)
from market_intelligence.schema import (
    MarketEvidenceBatch,
    MarketIdentity,
    MarketIntelArtifact,
    sanitize_edge_case_submarkets,
    sanitize_false_negative_hypotheses,
    sanitize_inferred_research_questions,
    sanitize_market_findings,
    sanitize_narrative_items,
    sanitize_self_presentation_patterns,
    sanitize_sourcing_implications,
    sanitize_title_to_archetype_mapping,
)
from shared.llm_clients import _retry_with_backoff, get_llm_client
from shared.llm_usage import anthropic_usage_dict, openai_usage_dict, record_llm_usage


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _record_research_llm_error(
    *,
    provider: str,
    model: str,
    stage: str,
    exc: Exception,
    request: dict[str, Any] | None = None,
    usage_context: dict[str, Any] | None = None,
) -> None:
    """Best-effort typed LLM error receipt for direct research provider calls."""

    error_request = {
        **(request or {}),
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:240],
    }
    context = {"stage": stage}
    context.update(usage_context or {})
    try:
        record_llm_usage(
            provider=provider,
            model=model,
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            request=error_request,
            usage_context=context,
            actual_status="error",
        )
    except Exception:  # noqa: BLE001 — telemetry must not break research
        pass


def _extract_sources(response: Any) -> list[dict]:
    sources: list[dict] = []
    seen_urls: set[str] = set()
    now = _utc_now()
    for block in getattr(response, "content", []):
        if getattr(block, "type", "") != "web_search_tool_result":
            continue
        for result in getattr(block, "content", []):
            url = getattr(result, "url", "")
            title = getattr(result, "title", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(
                    {
                        "source_id": f"web:{url}",
                        "kind": "web_search",
                        "title": title,
                        "url": url,
                        "retrieved_at": now,
                        "used_for": ["market_research"],
                    }
                )
    return sources


def _extract_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", "") == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(parts)


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
        if _perplexity_item_type(item) != "message":
            continue
        for content in _perplexity_item_list_field(item, "content"):
            if _perplexity_item_type(content) != "output_text":
                continue
            text = _normalize_text(_perplexity_item_field(content, "text"))
            if text:
                texts.append(text)
    if texts:
        return "\n".join(texts)
    return ""


def _clean_research_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines[1:] if not line.strip().startswith("```")
        )
    return cleaned.strip()


def _scan_complete_object_items(array_text: str) -> list[dict]:
    items: list[dict] = []
    index = 0
    length = len(array_text)
    while index < length:
        while index < length and array_text[index] in " \r\n\t,":
            index += 1
        if index >= length or array_text[index] == "]":
            break
        if array_text[index] != "{":
            break
        start = index
        depth = 0
        in_string = False
        escape = False
        while index < length:
            char = array_text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = array_text[start : index + 1]
                        try:
                            parsed = json.loads(candidate)
                        except json.JSONDecodeError:
                            return items
                        if isinstance(parsed, dict):
                            items.append(parsed)
                        index += 1
                        break
            index += 1
        else:
            return items
    return items


def _repair_research_json(
    cleaned: str,
    expected_keys: tuple[str, ...],
) -> dict | None:
    repaired: dict[str, list[dict]] = {}
    found_any_key = False
    recovered_any_item = False
    for key in expected_keys:
        token = f'"{key}"'
        position = cleaned.find(token)
        if position == -1:
            repaired[key] = []
            continue
        found_any_key = True
        colon = cleaned.find(":", position + len(token))
        array_start = cleaned.find("[", colon + 1) if colon != -1 else -1
        if array_start == -1:
            repaired[key] = []
            continue
        items = _scan_complete_object_items(cleaned[array_start + 1 :])
        if items:
            recovered_any_item = True
        repaired[key] = items
    if found_any_key and recovered_any_item:
        return repaired
    return None


def _parse_research_json_with_metadata(
    text: str,
    *,
    expected_keys: tuple[str, ...] = (
        "inferred_research_questions",
        "market_findings",
        "sourcing_implications",
        "open_questions",
    ),
) -> tuple[dict, bool]:
    cleaned = _clean_research_json_text(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, False
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed, False
        except json.JSONDecodeError:
            pass

    repaired = _repair_research_json(cleaned, expected_keys)
    if repaired is not None:
        return repaired, True
    raise ValueError(f"Could not parse research output as JSON: {cleaned[:500]}")


def _parse_research_json(
    text: str,
    *,
    expected_keys: tuple[str, ...] = (
        "inferred_research_questions",
        "market_findings",
        "sourcing_implications",
        "open_questions",
    ),
) -> dict:
    parsed, _ = _parse_research_json_with_metadata(text, expected_keys=expected_keys)
    return parsed


def _validate_thesis_context(items: Any) -> list[dict]:
    if not isinstance(items, list):
        return []
    valid: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        evidence_refs = [
            str(ref).strip()
            for ref in (item.get("evidence_refs") or [])
            if str(ref).strip()
        ]
        if not claim or not evidence_refs:
            continue
        valid.append(
            {
                "claim": claim,
                "evidence_refs": evidence_refs,
                "confidence": min(
                    1.0,
                    max(0.0, float(item.get("confidence", 0.5))),
                ),
            }
        )
    return valid


def _validate_open_questions(items: Any) -> list[dict]:
    if not isinstance(items, list):
        return []
    valid: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        supporting_run_refs = [
            str(ref).strip()
            for ref in (item.get("supporting_run_refs") or [])
            if str(ref).strip()
        ]
        evidence_refs = [
            str(ref).strip()
            for ref in (item.get("evidence_refs") or [])
            if str(ref).strip()
        ]
        if not question or not (supporting_run_refs or evidence_refs):
            continue
        valid.append(
            {
                "question": question,
                "priority": str(item.get("priority", "medium")).strip(),
                "next_step": str(
                    item.get("next_step", "Investigate in next sourcing cycle.")
                ).strip(),
                "supporting_run_refs": supporting_run_refs,
                "evidence_refs": evidence_refs,
            }
        )
    return valid


def _canonicalize_evidence_refs(items: list[dict], sources: list[dict]) -> list[dict]:
    url_to_source_id = {
        str(source.get("url", "")).strip(): str(source.get("source_id", "")).strip()
        for source in sources
        if str(source.get("url", "")).strip() and str(source.get("source_id", "")).strip()
    }
    source_ids = {source_id for source_id in url_to_source_id.values() if source_id}
    canonicalized: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        refs = []
        for raw in item.get("evidence_refs", []) or []:
            ref = str(raw).strip()
            if not ref:
                continue
            if ref in source_ids:
                refs.append(ref)
            elif ref in url_to_source_id:
                refs.append(url_to_source_id[ref])
        refs = list(dict.fromkeys(refs))
        supporting_run_refs = [
            str(ref).strip()
            for ref in (item.get("supporting_run_refs") or [])
            if str(ref).strip()
        ]
        if not refs and not supporting_run_refs:
            continue
        normalized = dict(item)
        normalized["supporting_run_refs"] = supporting_run_refs
        normalized["evidence_refs"] = refs
        canonicalized.append(normalized)
    return canonicalized


def _augment_bundle_with_prior_memory(
    research_bundle: dict,
    *,
    previous_artifact: MarketIntelArtifact | None,
    previous_agent_state: Any | None,
) -> dict:
    augmented = dict(research_bundle)
    prior_memory: dict[str, Any] = {}
    if previous_artifact is not None:
        prior_memory["previous_market_thesis_summary"] = _normalize_text(
            previous_artifact.market_thesis.get("summary")
        )
        prior_memory["previous_brief_recommendations"] = [
            {
                "target_field": str(item.get("target_field", "")).strip(),
                "proposal": str(item.get("proposal", "")).strip(),
                "reason": str(item.get("reason", "")).strip(),
            }
            for item in previous_artifact.brief_recommendations[:5]
            if isinstance(item, dict)
        ]
        prior_memory["previous_open_questions"] = [
            {
                "question": str(item.get("question", "")).strip(),
                "priority": str(item.get("priority", "")).strip(),
            }
            for item in previous_artifact.open_questions[:5]
            if isinstance(item, dict)
        ]
        prior_memory["previous_external_context"] = [
            str(item.get("claim", "")).strip()
            for item in previous_artifact.market_thesis.get("external_context", [])[:5]
            if isinstance(item, dict) and _normalize_text(item.get("claim"))
        ]
    if previous_agent_state is not None:
        prior_memory["active_hypotheses"] = [
            {
                "statement": getattr(item, "statement", ""),
                "confidence": getattr(item, "confidence", 0.0),
            }
            for item in getattr(previous_agent_state, "active_hypotheses", [])[:5]
        ]
        prior_memory["open_unknowns"] = [
            {
                "question": str(item.get("question", "")).strip(),
                "priority": str(item.get("priority", "")).strip(),
            }
            for item in getattr(previous_agent_state, "open_unknowns", [])[:5]
            if isinstance(item, dict)
        ]
    if prior_memory:
        augmented["prior_market_memory"] = prior_memory
    return augmented


def _build_market_thesis_context_from_findings(findings: list[dict]) -> list[dict]:
    context: list[dict] = []
    seen: set[str] = set()
    for item in findings:
        label = _normalize_text(item.get("label"))
        summary = _normalize_text(item.get("summary"))
        evidence_refs = [
            ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
        ]
        if not (summary and evidence_refs):
            continue
        claim = summary if label.lower() in summary.lower() else f"{label}: {summary}"
        key = claim.lower()
        if key in seen:
            continue
        seen.add(key)
        context.append(
            {
                "claim": claim,
                "evidence_refs": evidence_refs,
                "confidence": min(
                    1.0,
                    max(0.0, float(item.get("confidence", 0.5) or 0.5)),
                ),
            }
        )
        if len(context) >= 8:
            break
    return context


def _merge_external_open_questions(
    open_questions: list[dict],
    inferred_questions: list[dict],
) -> list[dict]:
    merged = list(open_questions)
    seen = {
        _normalize_text(item.get("question")).lower()
        for item in merged
        if isinstance(item, dict)
    }
    for item in inferred_questions:
        if not isinstance(item, dict) or item.get("status") != "unresolved":
            continue
        question = _normalize_text(item.get("question"))
        if not question or question.lower() in seen:
            continue
        seen.add(question.lower())
        merged.append(
            {
                "question": question,
                "priority": str(item.get("priority", "medium")).strip() or "medium",
                "next_step": _normalize_text(item.get("why_it_matters"))
                or "Validate this in the next sourcing cycle.",
                "supporting_run_refs": item.get("supporting_run_refs", []),
                "evidence_refs": item.get("evidence_refs", []),
            }
        )
    return merged


def _research_expected_keys(research_mode: str) -> tuple[str, ...]:
    if research_mode == "edge_case":
        return (
            "inferred_research_questions",
            "edge_case_submarkets",
            "title_to_archetype_mapping",
            "self_presentation_patterns",
            "false_negative_hypotheses",
            "edge_case_sourcing_implications",
            "open_questions",
        )
    return (
        "inferred_research_questions",
        "market_findings",
        "sourcing_implications",
        "open_questions",
    )


def _retry_caps_message(research_mode: str) -> str:
    if research_mode == "edge_case":
        return (
            "Hard caps: at most 3 inferred_research_questions, 3 edge_case_submarkets, "
            "4 title_to_archetype_mapping items, 3 self_presentation_patterns, "
            "3 false_negative_hypotheses, 3 edge_case_sourcing_implications, and 2 open_questions."
        )
    return (
        "Hard caps: at most 3 inferred_research_questions, 4 market_findings, "
        "3 sourcing_implications, and 2 open_questions."
    )


def _build_edge_case_research_result(
    *,
    parsed: dict,
    sources: list[dict],
    default_supporting_run_refs: list[str],
) -> ExternalResearchResult:
    inferred_questions = _canonicalize_evidence_refs(
        sanitize_inferred_research_questions(
            parsed.get("inferred_research_questions", []),
            default_supporting_run_refs=default_supporting_run_refs,
        ),
        sources,
    )
    edge_case_submarkets = _canonicalize_evidence_refs(
        sanitize_edge_case_submarkets(
            parsed.get("edge_case_submarkets", []),
            default_supporting_run_refs=default_supporting_run_refs,
        ),
        sources,
    )
    title_to_archetype_mapping = _canonicalize_evidence_refs(
        sanitize_title_to_archetype_mapping(
            parsed.get("title_to_archetype_mapping", []),
            default_supporting_run_refs=default_supporting_run_refs,
        ),
        sources,
    )
    self_presentation_patterns = _canonicalize_evidence_refs(
        sanitize_self_presentation_patterns(
            parsed.get("self_presentation_patterns", []),
            default_supporting_run_refs=default_supporting_run_refs,
        ),
        sources,
    )
    false_negative_hypotheses = _canonicalize_evidence_refs(
        sanitize_false_negative_hypotheses(
            parsed.get("false_negative_hypotheses", []),
            default_supporting_run_refs=default_supporting_run_refs,
        ),
        sources,
    )
    edge_case_sourcing_implications = _canonicalize_evidence_refs(
        sanitize_sourcing_implications(
            parsed.get("edge_case_sourcing_implications", []),
            default_supporting_run_refs=default_supporting_run_refs,
        ),
        sources,
    )
    edge_case_open_questions = _canonicalize_evidence_refs(
        _validate_open_questions(parsed.get("open_questions", [])),
        sources,
    )
    edge_case_open_questions = sanitize_narrative_items(
        "open_questions",
        _merge_external_open_questions(edge_case_open_questions, inferred_questions),
    )
    return ExternalResearchResult(
        sources=sources,
        edge_case_inferred_research_questions=inferred_questions,
        edge_case_submarkets=edge_case_submarkets,
        title_to_archetype_mapping=title_to_archetype_mapping,
        self_presentation_patterns=self_presentation_patterns,
        false_negative_hypotheses=false_negative_hypotheses,
        edge_case_sourcing_implications=edge_case_sourcing_implications,
        edge_case_open_questions=edge_case_open_questions,
    )


def _build_external_research_result(
    *,
    parsed: dict,
    sources: list[dict],
    default_supporting_run_refs: list[str],
    research_mode: str = "general",
) -> ExternalResearchResult:
    if research_mode == "edge_case":
        return _build_edge_case_research_result(
            parsed=parsed,
            sources=sources,
            default_supporting_run_refs=default_supporting_run_refs,
        )
    inferred_questions = _canonicalize_evidence_refs(
        sanitize_inferred_research_questions(
            parsed.get("inferred_research_questions", []),
            default_supporting_run_refs=default_supporting_run_refs,
        ),
        sources,
    )
    findings = _canonicalize_evidence_refs(
        sanitize_market_findings(
            parsed.get("market_findings", []),
            default_supporting_run_refs=default_supporting_run_refs,
        ),
        sources,
    )
    implications = _canonicalize_evidence_refs(
        sanitize_sourcing_implications(
            parsed.get("sourcing_implications", []),
            default_supporting_run_refs=default_supporting_run_refs,
        ),
        sources,
    )
    open_questions = _canonicalize_evidence_refs(
        _validate_open_questions(parsed.get("open_questions", [])),
        sources,
    )
    open_questions = sanitize_narrative_items(
        "open_questions",
        _merge_external_open_questions(open_questions, inferred_questions),
    )

    thesis_context = _canonicalize_evidence_refs(
        _validate_thesis_context(parsed.get("market_thesis_context", [])),
        sources,
    )
    if not thesis_context:
        thesis_context = _build_market_thesis_context_from_findings(findings)

    return ExternalResearchResult(
        sources=sources,
        inferred_research_questions=inferred_questions,
        market_findings=findings,
        sourcing_implications=implications,
        market_thesis_context=thesis_context,
        open_questions=open_questions,
    )


def _country_code(country: str) -> str:
    mapping = {
        "united states": "US",
        "usa": "US",
        "canada": "CA",
        "colombia": "CO",
        "mexico": "MX",
        "brazil": "BR",
        "united kingdom": "GB",
        "uk": "GB",
        "england": "GB",
        "france": "FR",
        "germany": "DE",
        "spain": "ES",
        "india": "IN",
    }
    return mapping.get(country.strip().lower(), "")


def _build_perplexity_search_tool(market_identity: MarketIdentity) -> dict:
    tool: dict[str, Any] = {
        "type": "web_search",
        "filters": {"search_recency_filter": "year"},
    }
    geography = _normalize_text(market_identity.geography)
    if geography:
        parts = [part.strip() for part in geography.split(",") if part.strip()]
        country = parts[-1] if parts else ""
        region = parts[-2] if len(parts) >= 2 else ""
        city = parts[-3] if len(parts) >= 3 else (parts[0] if len(parts) >= 1 else "")
        country_code = _country_code(country)
        if country_code:
            user_location = {"country": country_code}
            if region:
                user_location["region"] = region
            if city:
                user_location["city"] = city
            tool["user_location"] = user_location
    return tool


def _iter_perplexity_output_items(response: Any) -> list[Any]:
    if isinstance(response, dict):
        output = response.get("output")
        if isinstance(output, list):
            return list(output)
    output = getattr(response, "output", None)
    if isinstance(output, list):
        return list(output)
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
        if isinstance(payload, dict) and isinstance(payload.get("output"), list):
            return list(payload["output"])
    return []


def _perplexity_item_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("type", "")).strip()
    return str(getattr(item, "type", "")).strip()


def _perplexity_item_field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _perplexity_item_list_field(item: Any, name: str) -> list[Any]:
    value = _perplexity_item_field(item, name)
    if isinstance(value, list):
        return value
    return []


def _append_perplexity_source(
    *,
    sources_by_url: dict[str, dict],
    url: str,
    title: str,
    snippet: str = "",
    kind: str,
    now: str,
) -> None:
    cleaned_url = _normalize_text(url)
    if not cleaned_url:
        return
    cleaned_title = _normalize_text(title)
    existing = sources_by_url.get(cleaned_url)
    if existing:
        used_for = list(existing.get("used_for", []))
        if kind not in used_for:
            used_for.append(kind)
            existing["used_for"] = used_for
        if cleaned_title and not _normalize_text(existing.get("title")):
            existing["title"] = cleaned_title
        cleaned_snippet = _normalize_text(snippet)
        if cleaned_snippet and not _normalize_text(existing.get("snippet")):
            existing["snippet"] = cleaned_snippet
        return
    sources_by_url[cleaned_url] = {
        "source_id": f"web:{cleaned_url}",
        "kind": kind,
        "title": cleaned_title or cleaned_url,
        "url": cleaned_url,
        "snippet": _normalize_text(snippet),
        "retrieved_at": now,
        "used_for": [kind],
    }


def _extract_perplexity_sources(response: Any) -> list[dict]:
    sources_by_url: dict[str, dict] = {}
    now = _utc_now()
    for item in _iter_perplexity_output_items(response):
        item_type = _perplexity_item_type(item)
        if item_type == "search_results":
            for result in _perplexity_item_list_field(item, "results"):
                _append_perplexity_source(
                    sources_by_url=sources_by_url,
                    url=str(_perplexity_item_field(result, "url") or ""),
                    title=str(_perplexity_item_field(result, "title") or ""),
                    snippet=str(_perplexity_item_field(result, "snippet") or ""),
                    kind="web_search",
                    now=now,
                )
        elif item_type == "fetch_url_results":
            for result in _perplexity_item_list_field(item, "contents"):
                _append_perplexity_source(
                    sources_by_url=sources_by_url,
                    url=str(_perplexity_item_field(result, "url") or ""),
                    title=str(_perplexity_item_field(result, "title") or ""),
                    snippet=str(_perplexity_item_field(result, "snippet") or ""),
                    kind="fetch_url",
                    now=now,
                )
    return list(sources_by_url.values())


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    deduped: dict[str, dict] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        url = _normalize_text(source.get("url"))
        source_id = _normalize_text(source.get("source_id"))
        key = url or source_id
        if not key:
            continue
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = dict(source)
            continue
        existing_used_for = {
            _normalize_text(item)
            for item in existing.get("used_for", [])
            if _normalize_text(item)
        }
        new_used_for = {
            _normalize_text(item)
            for item in source.get("used_for", [])
            if _normalize_text(item)
        }
        existing["used_for"] = sorted(existing_used_for | new_used_for)
        if not _normalize_text(existing.get("title")) and _normalize_text(source.get("title")):
            existing["title"] = _normalize_text(source.get("title"))
        if not _normalize_text(existing.get("snippet")) and _normalize_text(source.get("snippet")):
            existing["snippet"] = _normalize_text(source.get("snippet"))
    return list(deduped.values())


def _write_perplexity_debug_artifact(
    *,
    response: Any,
    response_text: str,
    parse_error: str,
    attempt: str,
) -> Path:
    debug_dir = Path(config.PROJECT_ROOT) / "output" / "debug" / "perplexity_failures"
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = debug_dir / f"{timestamp}-{attempt}.json"
    payload: dict[str, Any] = {
        "captured_at": _utc_now(),
        "attempt": attempt,
        "parse_error": parse_error,
        "response_text_length": len(response_text),
        "response_text": response_text,
    }
    if isinstance(response, dict):
        payload["response"] = response
    elif hasattr(response, "model_dump"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload["response"] = response.model_dump()
    else:
        payload["response_repr"] = repr(response)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _build_perplexity_retry_user_prompt(
    user_prompt: str,
    *,
    caps_message: str,
) -> str:
    return (
        user_prompt
        + "\n\nRETRY MODE:\n"
        + "The previous provider response was malformed or truncated.\n"
        + "Return a shorter, stricter JSON payload.\n"
        + caps_message
        + "\n"
        + "Keep every string concise and avoid long rationale paragraphs.\n"
        + "Return JSON only."
    )


def _build_perplexity_tools(
    market_identity: MarketIdentity,
    preset: str,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [_build_perplexity_search_tool(market_identity)]
    if _normalize_text(preset).lower() == "deep-research":
        tools.append({"type": "fetch_url"})
    return tools


class AnthropicResearchBackend:
    """Anthropic Messages API web-search backend for market research."""

    def __init__(
        self,
        model_name: str | None = None,
        max_searches: int = 10,
    ) -> None:
        self.model_name = model_name or config.CHEAP_MODEL_NAME
        self.max_searches = max_searches

    def collect(
        self,
        *,
        market_identity: MarketIdentity,
        previous_artifact: MarketIntelArtifact | None,
        previous_agent_state: Any | None = None,
        evidence_batches: list[MarketEvidenceBatch],
        planner_result: Any | None = None,
        research_focus: list[dict] | None = None,
        research_mode: str = "general",
        edge_case_reasoning: str = "",
    ) -> ExternalResearchResult:
        research_bundle = _augment_bundle_with_prior_memory(
            build_research_context_bundle(
                market_identity,
                evidence_batches,
            ),
            previous_artifact=previous_artifact,
            previous_agent_state=previous_agent_state,
        )
        planner_summary = _normalize_text(
            getattr(planner_result, "planner_summary", "")
            if planner_result is not None
            else ""
        )
        if research_mode == "edge_case":
            system = build_perplexity_edge_case_research_instructions()
            user = build_perplexity_edge_case_research_user_prompt(
                market_identity,
                research_bundle,
                edge_case_focus=research_focus,
                planner_summary=planner_summary,
                edge_case_reasoning=edge_case_reasoning,
            )
        else:
            system = build_research_system_prompt()
            user = build_research_user_prompt(
                market_identity,
                research_bundle,
                selected_questions=research_focus,
                planner_summary=planner_summary,
            )
        try:
            client = get_llm_client(
                "anthropic",
                config.ANTHROPIC_API_KEY,
                300.0,
            )
        except Exception as exc:
            _record_research_llm_error(
                provider="anthropic",
                model=self.model_name,
                stage="anthropic_research",
                exc=exc,
                request={
                    "system_prompt_chars": len(system),
                    "input_chars": len(user),
                    "max_tokens": 8192,
                    "max_searches": self.max_searches,
                },
                usage_context={
                    "module": "market_intelligence",
                    "turn_index": 0,
                },
            )
            raise

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        all_sources: list[dict] = []
        final_text = ""

        for _turn in range(3):
            try:
                response = _retry_with_backoff(
                    lambda: client.messages.create(
                        model=self.model_name,
                        max_tokens=8192,
                        system=system,
                        messages=messages,
                        tools=[
                            {
                                "type": "web_search_20250305",
                                "name": "web_search",
                                "max_uses": self.max_searches,
                            }
                        ],
                    ),
                    label="Anthropic research",
                    max_attempts=3,
                )
            except Exception as exc:
                _record_research_llm_error(
                    provider="anthropic",
                    model=self.model_name,
                    stage="anthropic_research",
                    exc=exc,
                    request={
                        "system_prompt_chars": len(system),
                        "input_chars": len(user),
                        "max_tokens": 8192,
                        "max_searches": self.max_searches,
                    },
                    usage_context={
                        "module": "market_intelligence",
                        "turn_index": _turn,
                    },
                )
                raise

            try:
                record_llm_usage(
                    provider="anthropic",
                    model=self.model_name,
                    usage=anthropic_usage_dict(response),
                    usage_context={
                        "module": "market_intelligence",
                        "stage": "anthropic_research",
                        "turn_index": _turn,
                    },
                )
            except Exception:  # noqa: BLE001 — telemetry must not break research
                pass
            all_sources.extend(_extract_sources(response))
            final_text = _extract_text(response)
            if getattr(response, "stop_reason", "") == "end_turn":
                break
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": "Continue the research and return final JSON only.",
                }
            )

        return _build_external_research_result(
            parsed=_parse_research_json(
                final_text,
                expected_keys=_research_expected_keys(research_mode),
            ),
            sources=all_sources,
            default_supporting_run_refs=[
                batch.run_ref for batch in evidence_batches if _normalize_text(batch.run_ref)
            ],
            research_mode=research_mode,
        )


class PerplexityResearchBackend:
    """Perplexity deep-research backend for market intelligence."""

    def __init__(
        self,
        model_name: str | None = None,
        preset: str | None = None,
        timeout_seconds: float | None = None,
        max_output_tokens: int = 8192,
    ) -> None:
        self.model_name = model_name or config.MARKET_INTEL_EXTERNAL_RESEARCH_MODEL
        self.preset = preset or config.MARKET_INTEL_PERPLEXITY_PRESET
        self.timeout_seconds = (
            timeout_seconds or config.MARKET_INTEL_EXTERNAL_RESEARCH_TIMEOUT_SECONDS
        )
        self.max_output_tokens = max_output_tokens

    def collect(
        self,
        *,
        market_identity: MarketIdentity,
        previous_artifact: MarketIntelArtifact | None,
        previous_agent_state: Any | None = None,
        evidence_batches: list[MarketEvidenceBatch],
        planner_result: Any | None = None,
        research_focus: list[dict] | None = None,
        research_mode: str = "general",
        edge_case_reasoning: str = "",
    ) -> ExternalResearchResult:
        research_bundle = _augment_bundle_with_prior_memory(
            build_research_context_bundle(
                market_identity,
                evidence_batches,
            ),
            previous_artifact=previous_artifact,
            previous_agent_state=previous_agent_state,
        )
        planner_summary = _normalize_text(
            getattr(planner_result, "planner_summary", "")
            if planner_result is not None
            else ""
        )
        if research_mode == "edge_case":
            instructions = build_perplexity_edge_case_research_instructions()
            user = build_perplexity_edge_case_research_user_prompt(
                market_identity,
                research_bundle,
                edge_case_focus=research_focus,
                planner_summary=planner_summary,
                edge_case_reasoning=edge_case_reasoning,
            )
            response_format = build_perplexity_edge_case_research_response_format()
        else:
            instructions = build_perplexity_research_instructions()
            user = build_perplexity_research_user_prompt(
                market_identity,
                research_bundle,
                selected_questions=research_focus,
                planner_summary=planner_summary,
            )
            response_format = build_perplexity_research_response_format()

        extra_body: dict[str, Any] = {}
        if self.preset:
            extra_body["preset"] = self.preset
        extra_body["response_format"] = response_format

        kwargs: dict[str, Any] = {
            "input": user,
            "instructions": instructions,
            "max_output_tokens": self.max_output_tokens,
            "tools": _build_perplexity_tools(market_identity, self.preset),
        }
        if self.model_name:
            kwargs["model"] = self.model_name
        if extra_body:
            kwargs["extra_body"] = extra_body

        model_for_receipt = self.model_name or "perplexity-response-api"
        try:
            client = get_llm_client(
                "perplexity",
                config.PERPLEXITY_API_KEY,
                self.timeout_seconds,
            )
        except Exception as exc:
            _record_research_llm_error(
                provider="perplexity",
                model=model_for_receipt,
                stage="market_intel_external_research",
                exc=exc,
                request={
                    "max_tokens": self.max_output_tokens,
                    "instructions_chars": len(instructions),
                    "input_chars": len(user),
                },
                usage_context={
                    "market_key": market_identity.market_key,
                    "research_mode": research_mode,
                    "attempt": "client_init",
                    "provider_preset": self.preset,
                },
            )
            raise

        default_supporting_run_refs = [
            batch.run_ref for batch in evidence_batches if _normalize_text(batch.run_ref)
        ]
        attempt_payloads: list[tuple[dict, bool, list[dict]]] = []
        debug_paths: list[Path] = []
        parse_failures: list[str] = []

        def _attempt(request_kwargs: dict[str, Any], attempt_name: str) -> None:
            try:
                response = _retry_with_backoff(
                    lambda: client.responses.create(**request_kwargs),
                    label="Perplexity research",
                    max_attempts=3,
                )
            except Exception as exc:
                _record_research_llm_error(
                    provider="perplexity",
                    model=model_for_receipt,
                    stage="market_intel_external_research",
                    exc=exc,
                    request={
                        "max_tokens": request_kwargs.get(
                            "max_output_tokens", self.max_output_tokens
                        ),
                        "instructions_chars": len(instructions),
                        "input_chars": len(str(request_kwargs.get("input", ""))),
                    },
                    usage_context={
                        "market_key": market_identity.market_key,
                        "research_mode": research_mode,
                        "attempt": attempt_name,
                        "provider_preset": self.preset,
                    },
                )
                raise
            record_llm_usage(
                provider="perplexity",
                model=_normalize_text(getattr(response, "model", "")) or self.model_name or "perplexity-response-api",
                usage=openai_usage_dict(response),
                request={
                    "max_tokens": request_kwargs.get("max_output_tokens", self.max_output_tokens),
                    "instructions_chars": len(instructions),
                    "input_chars": len(str(request_kwargs.get("input", ""))),
                },
                usage_context={
                    "stage": "market_intel_external_research",
                    "market_key": market_identity.market_key,
                    "research_mode": research_mode,
                    "attempt": attempt_name,
                    "provider_preset": self.preset,
                },
            )
            response_text = _extract_perplexity_text(response)
            response_sources = _extract_perplexity_sources(response)
            try:
                parsed, repaired = _parse_research_json_with_metadata(
                    response_text,
                    expected_keys=_research_expected_keys(research_mode),
                )
            except Exception as exc:
                debug_paths.append(
                    _write_perplexity_debug_artifact(
                        response=response,
                        response_text=response_text,
                        parse_error=str(exc),
                        attempt=attempt_name,
                    )
                )
                parse_failures.append(f"{attempt_name}:{exc}")
                return
            if repaired:
                debug_paths.append(
                    _write_perplexity_debug_artifact(
                        response=response,
                        response_text=response_text,
                        parse_error="repaired malformed/truncated JSON",
                        attempt=attempt_name,
                    )
                )
            attempt_payloads.append((parsed, repaired, response_sources))

        _attempt(dict(kwargs), "initial")
        if not attempt_payloads or attempt_payloads[0][1]:
            retry_kwargs = dict(kwargs)
            retry_kwargs["input"] = _build_perplexity_retry_user_prompt(
                user,
                caps_message=_retry_caps_message(research_mode),
            )
            retry_kwargs["max_output_tokens"] = max(self.max_output_tokens, 8192)
            _attempt(retry_kwargs, "retry")

        if not attempt_payloads:
            detail = "; ".join(parse_failures) or "unknown parse failure"
            if debug_paths:
                detail += " | debug=" + ", ".join(str(path) for path in debug_paths)
            raise ValueError(f"Perplexity research parsing failed: {detail}")

        all_sources: list[dict] = []
        for _parsed, _repaired, response_sources in attempt_payloads:
            all_sources.extend(response_sources)

        parsed, _repaired, _sources = max(
            attempt_payloads,
            key=lambda item: (
                sum(
                    len(item[0].get(key, []))
                    for key in (
                        "inferred_research_questions",
                        "market_findings",
                        "sourcing_implications",
                        "edge_case_submarkets",
                        "title_to_archetype_mapping",
                        "self_presentation_patterns",
                        "false_negative_hypotheses",
                        "edge_case_sourcing_implications",
                        "open_questions",
                    )
                ),
                0 if item[1] else 1,
            ),
        )
        return _build_external_research_result(
            parsed=parsed,
            sources=_dedupe_sources(all_sources),
            default_supporting_run_refs=default_supporting_run_refs,
            research_mode=research_mode,
        )


def build_external_research_backend() -> Any:
    provider = _normalize_text(config.MARKET_INTEL_EXTERNAL_RESEARCH_PROVIDER).lower()
    if provider in ("", "auto"):
        if _normalize_text(config.PERPLEXITY_API_KEY):
            return PerplexityResearchBackend()
        if _normalize_text(config.ANTHROPIC_API_KEY):
            return AnthropicResearchBackend()
        raise RuntimeError(
            "No external research provider configured. Set PERPLEXITY_API_KEY or ANTHROPIC_API_KEY."
        )
    if provider in {"perplexity", "pplx"}:
        if not _normalize_text(config.PERPLEXITY_API_KEY):
            raise RuntimeError(
                "MARKET_INTEL_EXTERNAL_RESEARCH_PROVIDER=perplexity but PERPLEXITY_API_KEY is not set."
            )
        return PerplexityResearchBackend()
    if provider in {"anthropic", "claude"}:
        if not _normalize_text(config.ANTHROPIC_API_KEY):
            raise RuntimeError(
                "MARKET_INTEL_EXTERNAL_RESEARCH_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set."
            )
        return AnthropicResearchBackend()
    raise RuntimeError(
        f"Unsupported MARKET_INTEL_EXTERNAL_RESEARCH_PROVIDER: {config.MARKET_INTEL_EXTERNAL_RESEARCH_PROVIDER}"
    )


AgentSDKResearchBackend = AnthropicResearchBackend
