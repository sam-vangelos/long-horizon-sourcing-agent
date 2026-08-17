"""CTA-time transcript-to-brief composition for conversational intake."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import shared.config as shared_config
from market_intelligence.briefing_polish import _has_llm_access
from shared.brief_v2_schema import (
    BriefSchemaError,
    normalize_generated_engagement_context,
    validate_v2_brief,
)
from shared.intake_conversation import ConversationMessage
from shared.intake_conversation.insights import (
    HIRING_MANAGER_PICTURE_KEY,
    HIRING_MANAGER_PICTURE_LOCK_PATH,
    is_missing_hiring_manager_success_image,
    normalize_hiring_manager_success_image,
)
from shared.intake_conversation.state import merge_extracted
from shared.intake_conversation.sufficiency import is_ready_to_compose
from shared.llm_clients import opus_llm_cached
from shared.llm_usage import llm_usage_session
from shared.source_capabilities import (
    recommend_source_strategy_from_text,
    source_capability_prompt_block,
    target_modules_from_strategy,
)
from shared.source_packet import (
    ExtractedSourceFile,
    compose_source_packet_text,
    normalize_source_text,
)


COMPOSE_FROM_CONVERSATION_SYSTEM = (
    "You are Cloris composing a V2 sourcing brief from a persisted recruiter "
    "intake transcript. Return JSON only.\n\n"
    "Recover the draft from the transcript, source packet, and current draft. "
    "Respect manual edit locks: do not overwrite locked fields unless the "
    "latest recruiter message explicitly corrected that field.\n\n"
    "Output shape:\n"
    "{\n"
    '  "v2_draft": {\n'
    '    "role_title": "short job title",\n'
    '    "role_summary": "one sentence",\n'
    '    "geography": "string if known",\n'
    '    "capability_areas": [{"name": "short label", "description": "what good looks like"}],\n'
    '    "depth_distinction": {"builder_definition": "...", "user_definition": "...", "edge_case_guidance": "..."},\n'
    '    "non_fit_patterns": [{"label": "...", "why_not": "..."}],\n'
    '    "minimum_bar_description": "...",\n'
    '    "target_modules": ["linkedin"],\n'
    '    "source_strategy": [{"source": "linkedin", "role": "primary", "rationale": "..."}],\n'
    '    "engagement_context": {"selectivity_posture": "selective or coverage", "hiring_company": "optional", "engagement_description": "optional", "talent_bar_statement": "optional"}\n'
    "  },\n"
    '  "intake_insights": {\n'
    '    "hiring_manager_success_image": {\n'
    '      "summary": "ONE vivid sentence picturing the person the hiring manager actually wants. Role-anchored, recruiter-readable. Never write generic tropes.",\n'
    '      "proof_points": ["evidence the hiring manager would recognize as real"],\n'
    '      "screening_translation": "how this picture changes screening behavior",\n'
    '      "confidence": 0.0,\n'
    '      "source": "conversation|source_packet|combined",\n'
    '      "corrected_by_recruiter": false\n'
    "    }\n"
    "  },\n"
    '  "deficits": [{"field": "plain field name", "reason": "why it is still missing"}]\n'
    "}\n\n"
    "Use recruiter-facing language inside string values. Translate internal "
    "ideas this way: non-fit means people who look right on paper but should "
    "be screened out; depth means what separates someone who has really done "
    "this from someone who has only been around it; target modules means "
    "where I would look. Never put raw backend labels in recruiter-facing "
    "strings.\n\n"
    "The ``intake_insights`` block is parallel to ``v2_draft`` — it is NOT a "
    "v2 brief field. The ``hiring_manager_success_image`` is load-bearing: if "
    "you can form a vivid picture from the evidence, emit it; otherwise omit "
    "it. Forbidden phrasing in the picture summary: \"strong communication "
    "skills\", \"team player\", \"self-starter\", \"rockstar\", \"wears many "
    "hats\". Empty is better than trope. If the recruiter explicitly "
    "corrected the picture in their most recent turn, set "
    "``corrected_by_recruiter: true``.\n\n"
    "SOURCE CAPABILITY MANIFEST:\n"
    "Evidence boundaries are not permission to skip a module; they define "
    "what a module cannot prove alone and what companion evidence completes "
    "the read.\n"
    + source_capability_prompt_block()
)


@dataclass(frozen=True)
class ComposeFromConversationResult:
    """CTA-time recovery output.

    The composer returns split payloads:

    - ``v2_draft`` — the full V2 brief draft, validated via
      :func:`shared.brief_v2_schema.validate_v2_brief`.
    - ``insight_updates`` — the ``intake_insights`` payload to merge into
      ``state_json.intake_insights`` via
      :func:`shared.intake_conversation.insights.merge_intake_insights`.
      Independent of ``v2_draft``; survives v2 validation failures.
    - ``insight_deficits`` — recoverable deficits surfaced separately
      from the v2 schema deficit fields (``missing_keys`` / ``invalid_keys``
      / ``deficits``). The brief schema can be valid while the picture is
      still missing; the frontend uses this to drive CTA recovery copy.
    """

    v2_draft: dict[str, Any]
    status: Literal["composed", "deficits"]
    deficits: list[dict[str, str]]
    missing_keys: list[str]
    invalid_keys: list[str]
    source: Literal["llm", "deterministic", "empty"]
    metadata: dict[str, Any]
    insight_updates: dict[str, Any] = field(default_factory=dict)
    insight_deficits: list[dict[str, str]] = field(default_factory=list)


def compose_from_conversation(
    *,
    messages: list[ConversationMessage],
    current_v2_draft: dict[str, Any] | None,
    source_packet: dict[str, Any] | None,
    manually_edited_keys: set[str] | list[str] | tuple[str, ...] = (),
    role_title_hint: str | None = None,
    session_id: int | None = None,
    current_intake_insights: dict[str, Any] | None = None,
) -> ComposeFromConversationResult:
    """Compose a draft from the persisted intake transcript.

    This is the stronger CTA-time recovery path. Incremental extraction may
    stay cheap and lossy; this path reads the whole transcript and source
    packet before "Show me the brief" or "File this brief" proceeds.

    Returns split v2 / insight payloads. ``insight_updates`` is the
    additive update for ``state_json.intake_insights`` — the API
    persistence layer applies it via ``merge_intake_insights``.
    Insight deficits are surfaced via ``insight_deficits`` and never
    inflate ``v2_draft`` schema deficit fields.
    """

    t0 = time.monotonic()
    current = current_v2_draft if isinstance(current_v2_draft, dict) else {}
    current_insights = (
        current_intake_insights
        if isinstance(current_intake_insights, dict)
        else {}
    )
    locked = set(manually_edited_keys or ())
    source_text = _source_text_from_packet(source_packet)
    transcript_text = _transcript_text(messages)
    evidence_text = "\n\n".join(x for x in (source_text, transcript_text) if x.strip())
    composed_at = datetime.now(timezone.utc).isoformat()

    if not evidence_text.strip():
        return _finish(
            v2=current,
            source="empty",
            started_at=t0,
            composed_at=composed_at,
            message_count=len(messages),
            deficits=[
                {
                    "field": "conversation",
                    "reason": "No recruiter transcript or source material is available to compose from.",
                }
            ],
            insight_updates={},
            insight_deficits=_insight_deficits_from_state(current_insights),
        )

    raw_v2: dict[str, Any] | None = None
    raw_insights: dict[str, Any] = {}
    source: Literal["llm", "deterministic"] = "deterministic"
    if evidence_text.strip() and _has_llm_access():
        llm_payload = _llm_compose(
            messages=messages,
            current_v2_draft=current,
            source_packet=source_packet,
            manually_edited_keys=locked,
            role_title_hint=role_title_hint,
            session_id=session_id,
            current_intake_insights=current_insights,
        )
        if llm_payload is not None:
            raw_v2, raw_insights = llm_payload
            source = "llm"

    if raw_v2 is None:
        raw_v2 = _heuristic_compose(
            evidence_text=evidence_text,
            current_v2_draft=current,
            source_packet=source_packet,
            role_title_hint=role_title_hint,
        )

    normalized = _normalize_v2(raw_v2, evidence_text=evidence_text)
    merged = merge_extracted(current, normalized, manually_edited_keys=locked)
    merged = _normalize_v2(merged, evidence_text=evidence_text, locked=locked)

    # Insight payload runs through the shared normalizer. Lock-respecting
    # is the API persistence layer's job (via merge_intake_insights);
    # here we just normalize so trope-shaped or below-floor output drops.
    insight_updates = _normalize_insight_payload(
        raw_insights,
        v2_draft=merged,
        source_packet=source_packet,
        has_packet=bool(_source_text_from_packet(source_packet).strip()),
    )

    insight_deficits = _insight_deficits_after_compose(
        insight_updates=insight_updates,
        current_insights=current_insights,
        locked=locked,
    )

    return _finish(
        v2=merged,
        source=source,
        started_at=t0,
        composed_at=composed_at,
        message_count=len(messages),
        deficits=_semantic_deficits(merged),
        insight_updates=insight_updates,
        insight_deficits=insight_deficits,
    )


def _llm_compose(
    *,
    messages: list[ConversationMessage],
    current_v2_draft: dict[str, Any],
    source_packet: dict[str, Any] | None,
    manually_edited_keys: set[str],
    role_title_hint: str | None,
    session_id: int | None,
    current_intake_insights: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Run the Opus compose pass and return (v2_draft, intake_insights).

    Insights are extracted from the LLM output's ``intake_insights`` block
    so the v2 schema validator never sees them. Both halves are
    independent — a malformed insight does not block v2 composition and
    a malformed v2 draft does not block insight surfacing (the caller
    falls back to the heuristic for v2 in that case while still keeping
    a normalized insight if one came through).
    """

    payload = {
        "messages": [
            {"role": m.get("role"), "content": m.get("content")}
            for m in messages
        ],
        "current_v2_draft": current_v2_draft,
        "current_intake_insights": current_intake_insights or {},
        "source_packet": source_packet or {},
        "manual_edit_locks": sorted(manually_edited_keys),
        "role_title_hint": role_title_hint or "",
    }
    prompt = "Compose the brief from this transcript.\n\nINPUT:\n" + json.dumps(
        payload, indent=2
    )
    try:
        with llm_usage_session(
            _resolve_log_path(),
            session_id=session_id,
            brief_id=None,
            stage="intake_compose_from_conversation",
        ):
            raw = opus_llm_cached(
                COMPOSE_FROM_CONVERSATION_SYSTEM,
                prompt,
                expect_json=True,
                max_tokens=12000,
                usage_context={
                    "stage": "intake_compose_from_conversation",
                    "session_id": session_id,
                },
            )
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    v2 = raw.get("v2_draft")
    if not isinstance(v2, dict):
        # Backwards-compatible fallback: pre-insight responses returned
        # the v2 draft as the root JSON object. Treat the whole dict as
        # v2 minus any insight block.
        v2 = {k: val for k, val in raw.items() if k not in {"intake_insights", "deficits"}}
    insights_block = raw.get("intake_insights")
    if not isinstance(insights_block, dict):
        insights_block = {}
    return v2, insights_block


def _normalize_insight_payload(
    raw: dict[str, Any],
    *,
    v2_draft: dict[str, Any],
    source_packet: dict[str, Any] | None,
    has_packet: bool,
) -> dict[str, Any]:
    """Run each insight value through the shared normalizer.

    Source attribution is derived from whether the source packet
    contributed evidence to this composition. The producer can override
    via the ``source`` field in the LLM output and the normalizer will
    honor declared values.
    """

    if not isinstance(raw, dict) or not raw:
        return {}
    role_context = _role_context_dict(v2_draft, source_packet)
    derived_source = "combined" if has_packet else "conversation"
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key == HIRING_MANAGER_PICTURE_KEY:
            normalized = normalize_hiring_manager_success_image(
                value, role_context, source=derived_source
            )
            if normalized is not None:
                out[key] = normalized
        # Future insight keys would dispatch here; unknown keys are dropped.
    return out


def _role_context_dict(
    v2_draft: dict[str, Any] | None,
    source_packet: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a role-context dict for the insight normalizer.

    Pulls role title / summary / capability area names from v2_draft and
    JD / intake notes / file text from the source packet so the
    ``is_generic_trope`` rule has enough to compare against.
    """

    ctx: dict[str, Any] = {}
    if isinstance(v2_draft, dict):
        for key in ("role_title", "role_summary"):
            value = v2_draft.get(key)
            if isinstance(value, str):
                ctx[key] = value
        cap = v2_draft.get("capability_areas")
        if isinstance(cap, list):
            ctx["capability_areas"] = cap
        for key in ("jd_text", "intake_notes"):
            value = v2_draft.get(key)
            if isinstance(value, str):
                ctx[key] = value
    if isinstance(source_packet, dict):
        text_chunks: list[str] = []
        for key in ("job_description_text", "intake_notes_text"):
            value = source_packet.get(key)
            if isinstance(value, str) and value.strip():
                text_chunks.append(value)
        files = source_packet.get("files")
        if isinstance(files, list):
            for item in files:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        text_chunks.append(text)
        if text_chunks:
            ctx["text"] = "\n\n".join(text_chunks)
    return ctx


def _insight_deficits_after_compose(
    *,
    insight_updates: dict[str, Any],
    current_insights: dict[str, Any],
    locked: set[str],
) -> list[dict[str, str]]:
    """Surface insight deficits separately from v2 schema deficits.

    The "merged preview" used here mirrors what the API will persist:
    apply the new updates over the current insights, but skip any locked
    paths so a recruiter-corrected (and thus locked) picture is not
    counted as missing just because the LLM declined to re-emit it.
    """

    preview: dict[str, Any] = dict(current_insights)
    for key, value in insight_updates.items():
        path = f"intake_insights.{key}"
        if path in locked:
            continue
        preview[key] = value
    return _insight_deficits_from_state(preview)


def _insight_deficits_from_state(
    insights: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Compute the deficit list given the would-be persisted insights."""

    if not isinstance(insights, dict):
        insights = {}
    picture = insights.get(HIRING_MANAGER_PICTURE_KEY)
    if is_missing_hiring_manager_success_image(picture):
        return [
            {
                "field": "hiring_manager_success_image",
                "reason": (
                    "Cloris is still forming the hiring-manager picture — "
                    "keep talking to sharpen it, or file with this gap noted."
                ),
            }
        ]
    return []


def _heuristic_compose(
    *,
    evidence_text: str,
    current_v2_draft: dict[str, Any],
    source_packet: dict[str, Any] | None,
    role_title_hint: str | None,
) -> dict[str, Any]:
    existing = current_v2_draft if isinstance(current_v2_draft, dict) else {}
    packet = source_packet if isinstance(source_packet, dict) else {}
    title = (
        _clean_string(existing.get("role_title"))
        or _clean_string(role_title_hint)
        or _extract_role_title(evidence_text)
    )
    summary = _clean_string(existing.get("role_summary")) or _extract_summary(
        evidence_text, title
    )
    geography = (
        _clean_string(existing.get("geography"))
        or _clean_string(packet.get("geography"))
        or _extract_geography(evidence_text)
    )
    source_strategy = (
        existing.get("source_strategy")
        if isinstance(existing.get("source_strategy"), list)
        else recommend_source_strategy_from_text(evidence_text)
    )
    target_modules = _existing_string_list(existing.get("target_modules"))
    if not target_modules:
        target_modules = target_modules_from_strategy(
            [x for x in source_strategy if isinstance(x, dict)]
        )

    v2: dict[str, Any] = {
        **existing,
        "role_title": title,
        "role_summary": summary,
        "geography": geography,
        "capability_areas": _capability_areas(existing, evidence_text),
        "depth_distinction": _depth_distinction(existing, evidence_text),
        "non_fit_patterns": _non_fit_patterns(existing, evidence_text),
        "minimum_bar_description": _minimum_bar(existing, evidence_text),
        "target_modules": target_modules,
        "source_strategy": source_strategy,
    }
    jd = _clean_string(packet.get("job_description_text"))
    notes = _clean_string(packet.get("intake_notes_text"))
    if jd:
        v2["jd_text"] = jd
    if notes:
        v2["intake_notes"] = notes
    return v2


def _finish(
    *,
    v2: dict[str, Any],
    source: Literal["llm", "deterministic", "empty"],
    started_at: float,
    composed_at: str,
    message_count: int,
    deficits: list[dict[str, str]],
    insight_updates: dict[str, Any] | None = None,
    insight_deficits: list[dict[str, str]] | None = None,
) -> ComposeFromConversationResult:
    missing_keys: list[str] = []
    invalid_keys: list[str] = []
    try:
        validate_v2_brief(v2)
    except BriefSchemaError as exc:
        missing_keys = list(exc.missing_keys)
        invalid_keys = list(exc.invalid_keys)
    ready, missing = is_ready_to_compose(v2)
    if not ready:
        deficits = deficits + [
            {"field": path, "reason": "Still needed before the brief can be filed."}
            for path in missing
        ]
    if missing_keys:
        deficits = deficits + [
            {"field": key, "reason": "Required brief structure is missing."}
            for key in missing_keys
        ]
    if invalid_keys:
        deficits = deficits + [
            {"field": key, "reason": "Brief structure is malformed."}
            for key in invalid_keys
        ]
    deduped = _dedupe_deficits(deficits)
    # ``insight_deficits`` are intentionally NOT promoted into ``missing_keys``
    # / ``invalid_keys`` / the v2 ``deficits`` list — the brief schema can
    # be valid while the picture is still missing. They surface only via
    # the dedicated ``insight_deficits`` field and the metadata mirror.
    deduped_insights = _dedupe_deficits(insight_deficits or [])
    status: Literal["composed", "deficits"] = (
        "composed" if ready and not missing_keys and not invalid_keys else "deficits"
    )
    metadata = {
        "source": source,
        "status": status,
        "composed_at": composed_at,
        "message_count": message_count,
        "elapsed_ms": int((time.monotonic() - started_at) * 1000),
        "missing_keys": missing_keys,
        "invalid_keys": invalid_keys,
        "deficits": deduped,
        "insight_deficits": deduped_insights,
    }
    return ComposeFromConversationResult(
        v2_draft=v2,
        status=status,
        deficits=deduped,
        missing_keys=missing_keys,
        invalid_keys=invalid_keys,
        source=source,
        metadata=metadata,
        insight_updates=dict(insight_updates or {}),
        insight_deficits=deduped_insights,
    )


def _normalize_v2(
    v2: dict[str, Any],
    *,
    evidence_text: str,
    locked: set[str] | None = None,
) -> dict[str, Any]:
    out = dict(v2 or {})
    locked = locked or set()
    if "role_title" not in locked and not _clean_string(out.get("role_title")):
        out["role_title"] = _extract_role_title(evidence_text)
    if "role_summary" not in locked and not _clean_string(out.get("role_summary")):
        out["role_summary"] = _extract_summary(
            evidence_text,
            _clean_string(out.get("role_title")),
        )
    if "geography" not in locked and not _clean_string(out.get("geography")):
        out["geography"] = _extract_geography(evidence_text)

    caps = out.get("capability_areas")
    if "capability_areas" in locked:
        pass
    elif not isinstance(caps, list) or not caps:
        out["capability_areas"] = _capability_areas({}, evidence_text)
    else:
        clean_caps: list[dict[str, str]] = []
        for item in caps:
            if not isinstance(item, dict):
                continue
            clean_caps.append(
                {
                    "name": str(item.get("name") or "Core scope").strip(),
                    "description": str(item.get("description") or "").strip(),
                }
            )
        out["capability_areas"] = clean_caps or _capability_areas({}, evidence_text)

    if "depth_distinction" not in locked:
        depth = out.get("depth_distinction")
        if not isinstance(depth, dict):
            depth = {}
        fallback_depth = _depth_distinction({}, evidence_text)
        out["depth_distinction"] = {
            "builder_definition": str(
                depth.get("builder_definition")
                or fallback_depth["builder_definition"]
            ),
            "user_definition": str(
                depth.get("user_definition") or fallback_depth["user_definition"]
            ),
            "edge_case_guidance": str(
                depth.get("edge_case_guidance")
                or fallback_depth["edge_case_guidance"]
            ),
        }

    if "non_fit_patterns" not in locked:
        out["non_fit_patterns"] = _normalize_patterns(out.get("non_fit_patterns"))
        if not out["non_fit_patterns"]:
            out["non_fit_patterns"] = _non_fit_patterns({}, evidence_text)

    strategy = out.get("source_strategy")
    if "source_strategy" in locked:
        strategy = strategy if isinstance(strategy, list) else []
    else:
        if not isinstance(strategy, list) or not strategy:
            strategy = recommend_source_strategy_from_text(evidence_text)
        else:
            strategy = [x for x in strategy if isinstance(x, dict)]
        out["source_strategy"] = strategy
    modules = _existing_string_list(out.get("target_modules"))
    if "target_modules" not in locked and not modules:
        modules = target_modules_from_strategy(strategy)
    if "target_modules" not in locked:
        out["target_modules"] = modules
    if (
        "minimum_bar_description" not in locked
        and not _clean_string(out.get("minimum_bar_description"))
    ):
        out["minimum_bar_description"] = _minimum_bar({}, evidence_text)
    normalize_generated_engagement_context(out)
    return out


def _source_text_from_packet(source_packet: dict[str, Any] | None) -> str:
    if not isinstance(source_packet, dict):
        return ""
    files: list[ExtractedSourceFile] = []
    raw_files = source_packet.get("files")
    if isinstance(raw_files, list):
        for item in raw_files:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            filename = item.get("filename")
            if not isinstance(text, str) or not isinstance(filename, str):
                continue
            files.append(
                ExtractedSourceFile(
                    filename=filename,
                    content_type=item.get("content_type")
                    if isinstance(item.get("content_type"), str)
                    else None,
                    char_count=int(item.get("char_count") or len(text)),
                    text=text,
                    kind=item.get("kind")
                    if item.get("kind") in {"job_description", "intake_notes", "general"}
                    else "general",
                )
            )
    return compose_source_packet_text(
        job_description_text=str(source_packet.get("job_description_text") or ""),
        intake_notes_text=str(source_packet.get("intake_notes_text") or ""),
        files=files,
    )


def _transcript_text(messages: list[ConversationMessage]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = normalize_source_text(msg.get("content") or "")
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _extract_role_title(text: str) -> str:
    patterns = (
        r"\b(?:role|title|job)\s+(?:is|called|will be)\s+(?:a |an |the )?([A-Z][A-Za-z0-9/&,+\- ]{2,80})",
        r"\b(?:hiring|looking for|need)\s+(?:a |an |the )?([A-Z][A-Za-z0-9/&,+\- ]{2,80})",
        r"\b(Head of [A-Z][A-Za-z0-9/&,+\- ]{2,80})",
        r"\b(Principal [A-Z][A-Za-z0-9/&,+\- ]{2,80})",
        r"\b(Staff [A-Z][A-Za-z0-9/&,+\- ]{2,80})",
        r"\b(Senior [A-Z][A-Za-z0-9/&,+\- ]{2,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _trim_title(match.group(1))
    for line in text.splitlines():
        stripped = line.strip(" #:\t")
        if not stripped or stripped.lower().startswith(("cloris", "recruiter")):
            continue
        if len(stripped.split()) <= 10 and not stripped.endswith((".", "?", "!")):
            return _trim_title(stripped)
    return ""


def _trim_title(value: str) -> str:
    title = re.split(r"[.\n]|\b(?:at|for|who|that|with)\b", value.strip())[0]
    return title.strip(" -,:")[:120]


def _extract_summary(text: str, title: str) -> str:
    paragraph = _first_recruiter_paragraph(text)
    if title and paragraph:
        return f"{title} role scoped from recruiter intake: {paragraph[:220]}"
    return paragraph[:280]


def _first_recruiter_paragraph(text: str) -> str:
    for raw in re.split(r"\n\s*\n|\n", text):
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^(recruiter|cloris):\s*", "", line, flags=re.I)
        if line and not line.lower().startswith("hi "):
            return line
    return ""


def _extract_geography(text: str) -> str:
    lower = text.lower()
    if "remote" in lower:
        return "Remote"
    for pattern in (
        r"\b(New York|NYC|London|San Francisco|Bay Area|Los Angeles|Austin|Chicago)\b",
        r"\b(United States|US|U\.S\.)\b",
    ):
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1)
    return ""


def _capability_areas(existing: dict[str, Any], text: str) -> list[dict[str, str]]:
    current = existing.get("capability_areas")
    if isinstance(current, list) and current:
        return [x for x in current if isinstance(x, dict)]
    sentences = _sentences(text)
    candidates = [
        s
        for s in sentences
        if any(
            marker in s.lower()
            for marker in (
                "own",
                "build",
                "lead",
                "research",
                "design",
                "ship",
                "evaluate",
                "manage",
                "strategy",
                "technical",
            )
        )
    ]
    if not candidates:
        candidates = sentences[:2]
    first = candidates[0] if candidates else "Owns the core scope described in the intake."
    second = candidates[1] if len(candidates) > 1 else ""
    areas = [{"name": _capability_name(first), "description": first[:500]}]
    if second:
        areas.append({"name": _capability_name(second), "description": second[:500]})
    return areas


def _capability_name(sentence: str) -> str:
    lower = sentence.lower()
    if "research" in lower or "publication" in lower:
        return "Research depth"
    if "design" in lower or "portfolio" in lower:
        return "Design judgment"
    if "engineer" in lower or "technical" in lower or "build" in lower:
        return "Technical execution"
    if "lead" in lower or "manage" in lower or "executive" in lower:
        return "Leadership scope"
    if "go-to-market" in lower or "strategy" in lower:
        return "Strategic judgment"
    return "Core scope"


def _depth_distinction(existing: dict[str, Any], text: str) -> dict[str, str]:
    current = existing.get("depth_distinction")
    if isinstance(current, dict) and all(
        isinstance(current.get(k), str)
        for k in ("builder_definition", "user_definition", "edge_case_guidance")
    ):
        return {
            "builder_definition": str(current.get("builder_definition") or ""),
            "user_definition": str(current.get("user_definition") or ""),
            "edge_case_guidance": str(current.get("edge_case_guidance") or ""),
        }
    first = _first_recruiter_paragraph(text)
    return {
        "builder_definition": (
            "Has directly owned the central work, can explain tradeoffs from "
            f"first-hand execution, and can show evidence beyond adjacency. {first[:180]}"
        ).strip(),
        "user_definition": (
            "Has participated in similar work and can operate the known playbook, "
            "but may not have shaped the system or bar personally."
        ),
        "edge_case_guidance": (
            "Treat adjacent profiles as review-only unless they show direct ownership "
            "of the core capability and context similar to this brief."
        ),
    }


def _non_fit_patterns(existing: dict[str, Any], text: str) -> list[dict[str, str]]:
    current = _normalize_patterns(existing.get("non_fit_patterns"))
    if current:
        return current
    lower = text.lower()
    patterns: list[dict[str, str]] = []
    if "manager" in lower or "lead" in lower:
        patterns.append(
            {
                "label": "Title without ownership",
                "why_not": "Looks senior on paper but lacks evidence of owning the core work directly.",
            }
        )
    if "research" in lower or "ai" in lower:
        patterns.append(
            {
                "label": "AI adjacency only",
                "why_not": "Has been near AI work but cannot show real applied depth or shipped judgment.",
            }
        )
    if not patterns:
        patterns.append(
            {
                "label": "Adjacent background only",
                "why_not": "Looks close to the role but lacks direct evidence for the must-have capability.",
            }
        )
    return patterns


def _minimum_bar(existing: dict[str, Any], text: str) -> str:
    existing_bar = _clean_string(existing.get("minimum_bar_description"))
    if existing_bar:
        return existing_bar
    sentences = _sentences(text)
    for sentence in sentences:
        if any(marker in sentence.lower() for marker in ("must", "minimum", "bar", "need")):
            return sentence[:500]
    return (
        "No pass without direct evidence of the core capability, credible context "
        "for the role's scope, and enough signal to act on."
    )


def _normalize_patterns(value: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append({"label": item.strip()[:80], "why_not": item.strip()})
        elif isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or "").strip()
            why = str(item.get("why_not") or item.get("description") or "").strip()
            if label and why:
                out.append({"label": label, "why_not": why})
    return out


def _semantic_deficits(v2: dict[str, Any]) -> list[dict[str, str]]:
    deficits: list[dict[str, str]] = []
    if not _clean_string(v2.get("role_title")):
        deficits.append({"field": "role title", "reason": "No role title was stated."})
    if not _clean_string(v2.get("role_summary")):
        deficits.append({"field": "role summary", "reason": "The role purpose is still missing."})
    if not _clean_string(v2.get("minimum_bar_description")):
        deficits.append({"field": "minimum bar", "reason": "The pass/fail bar is still missing."})
    if not _existing_string_list(v2.get("target_modules")):
        deficits.append({"field": "where I would look", "reason": "No source recommendation is available."})
    return deficits


def _has_useful_existing_draft(v2: dict[str, Any]) -> bool:
    ready, _ = is_ready_to_compose(v2)
    return ready or bool(_clean_string(v2.get("role_title")))


def _existing_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if isinstance(x, str) and x.strip()]


def _clean_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", normalize_source_text(text))
    parts = re.split(r"(?<=[.!?])\s+|\n+", clean)
    return [p.strip(" -") for p in parts if len(p.strip()) > 20]


def _dedupe_deficits(deficits: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for item in deficits:
        field = str(item.get("field") or "")
        reason = str(item.get("reason") or "")
        key = (field, reason)
        if not field or key in seen:
            continue
        seen.add(key)
        out.append({"field": field, "reason": reason})
    return out


def _resolve_log_path() -> Path:
    base = Path(getattr(shared_config, "OUTPUT_DIR", Path("."))) / "intake_logs"
    base.mkdir(parents=True, exist_ok=True)
    return base / "conversation_compose.jsonl"


__all__ = [
    "COMPOSE_FROM_CONVERSATION_SYSTEM",
    "ComposeFromConversationResult",
    "compose_from_conversation",
]
