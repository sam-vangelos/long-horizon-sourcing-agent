"""cheap_llm slot-extraction pass for the conversational intake.

Runs after each orchestrator turn. Reads the conversation transcript +
current ``v2_draft`` + source_packet, and returns a structured
:class:`ExtractionResult` carrying separate ``v2_updates`` (for
``state_json.v2_draft``) and ``insight_updates`` (for
``state_json.intake_insights`` — currently the
``hiring_manager_success_image`` insight). The two streams are kept
distinct so insights never enter ``v2_draft`` even temporarily and the
v2 schema validation gate never sees insight-shaped fields.

Three load-bearing contracts:

1. **Manual-edit conflicts live in the prompt, not in code.** The
   ``manually_edited_keys`` set is passed to cheap_llm with the
   explicit rule: only overwrite a manually-edited slot when the
   recruiter's LATEST turn directly contradicts the current value.
   Never on inference, never on earlier turns. The
   :func:`shared.intake_conversation.state.merge_extracted` and
   :func:`shared.intake_conversation.insights.merge_intake_insights`
   backstops catch mistakes; this prompt rule is the primary defense.

2. **Placeholder strings are dropped before write** via
   :func:`market_intelligence.brief_distillation._looks_like_placeholder`
   — the same gate the read-back redesign uses, so we don't reintroduce
   the slime that earned the read-back rewrite.

3. **Validation gate (v2 only), partial-round safe (P9.4).** The merged
   v2 draft is run through :func:`shared.brief_v2_schema.validate_v2_brief`;
   if the only failure is missing required keys (the brief is incomplete
   — normal during intake), the v2 updates are accepted whole. If the
   failure is structural (``invalid_keys`` non-empty), only the offending
   TOP-LEVEL key(s) this round proposed are dropped — the rest of the
   round's valid updates still apply. Dropped keys are named on
   ``ExtractionResult.dropped_keys`` and logged as
   ``extraction_partial`` so the loss is visible, never silent. Insight
   updates are independently gated by
   :func:`shared.intake_conversation.insights.normalize_hiring_manager_success_image`
   and survive even when v2 validation fails.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from market_intelligence.brief_distillation import (
    PLACEHOLDER_STRINGS,
    _looks_like_placeholder,
)
from shared.brief_v2_schema import BriefSchemaError, validate_v2_brief
from shared.intake_conversation import ConversationMessage
from shared.intake_conversation.insights import (
    HIRING_MANAGER_PICTURE_KEY,
    HIRING_MANAGER_PICTURE_LOCK_PATH,
    normalize_hiring_manager_success_image,
)
from shared.intake_conversation.state import merge_extracted
from shared.llm_clients import cheap_llm
from shared.llm_usage import record_llm_usage
from shared.source_capabilities import source_capability_prompt_block

log = logging.getLogger(__name__)


# Top-level extractor keys that belong on ``state_json.intake_insights``,
# not ``state_json.v2_draft``. The extractor never lets these enter the
# v2_draft merge path even temporarily.
_INSIGHT_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {HIRING_MANAGER_PICTURE_KEY}
)


@dataclass(frozen=True)
class ExtractionResult:
    """Deliberate seam: cheap_llm extraction returns split payloads.

    ``v2_updates`` is merged into ``state_json.v2_draft`` via
    :func:`shared.intake_conversation.state.merge_extracted`.
    ``insight_updates`` is merged into ``state_json.intake_insights`` via
    :func:`shared.intake_conversation.insights.merge_intake_insights`.
    The two streams are intentionally independent so brief-schema
    validity and insight presence never collapse into each other.

    Empty / failure / validation-failure cases all return
    ``ExtractionResult({}, {})`` — never ``None``, never a tuple, never
    a flat dict that callers might iterate as legacy code did.

    ``dropped_keys`` (P9.4) names the top-level ``v2_updates`` keys this
    round proposed but that failed structural validation and were
    dropped. Non-empty ``dropped_keys`` means a PARTIAL round: the
    surviving ``v2_updates`` keys are still valid and were applied. This
    must stay visible to callers rather than silently vanishing —
    see :func:`extract_slots` docstring contract 4.
    """

    v2_updates: dict[str, Any] = field(default_factory=dict)
    insight_updates: dict[str, Any] = field(default_factory=dict)
    dropped_keys: tuple[str, ...] = ()


# Fields we know belong on the v2_draft for intake. Used for the schema
# block in the extractor system prompt.
_KNOWN_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "role_title",
    "role_summary",
    "capability_areas",
    "depth_distinction",
    "non_fit_patterns",
    "minimum_bar_description",
    "target_modules",
    "source_strategy",
    "source_config",
)


def extract_slots(
    *,
    messages: list[ConversationMessage],
    current_v2_draft: dict[str, Any],
    source_packet: dict[str, Any] | None,
    manually_edited_keys: set[str] | list[str] | tuple[str, ...] = (),
    session_id: int | None = None,
    current_intake_insights: dict[str, Any] | None = None,
) -> ExtractionResult:
    """Return an :class:`ExtractionResult` carrying split payloads.

    The cheap_llm pass receives the conversation, the current draft, the
    source packet, the current intake insights (so the prompt can show
    a hiring-manager picture-in-progress without leaking it into v2),
    and the manually-edited dot-paths; it is instructed to return ONLY
    updates the recruiter explicitly stated.

    Top-level keys in ``_INSIGHT_TOP_LEVEL_KEYS`` are split off into
    ``ExtractionResult.insight_updates`` after placeholder scrubbing and
    BEFORE the v2 schema validation gate, so a malformed v2 round never
    drops insight updates and a malformed insight never blocks v2
    extraction. Each insight update is normalized via
    :func:`normalize_hiring_manager_success_image`; trope-shaped or
    structurally incomplete insights are dropped at this layer.

    Returns ``ExtractionResult({}, {})`` on:
    - cheap_llm exception (treated as transient — try again next turn)
    - cheap_llm output not a dict
    - placeholder gate scrubs all values

    On a structural v2 validation failure (P9.4), only the offending
    top-level key(s) are dropped from ``v2_updates`` — the rest of the
    round's valid updates still land, and ``ExtractionResult.dropped_keys``
    names what was dropped. ``insight_updates`` is always independent of
    the v2 gate and survives regardless.
    """

    locked = set(manually_edited_keys or ())

    if not messages:
        # Nothing to extract from. The opener turn (if any) is Cloris-only;
        # extraction has nothing to do until the recruiter has spoken.
        return ExtractionResult({}, {})

    system_prompt = _build_extractor_system_prompt(
        manually_edited_keys=locked
    )
    user_prompt = _build_extractor_user_prompt(
        messages=messages,
        current_v2_draft=current_v2_draft,
        source_packet=source_packet,
        current_intake_insights=current_intake_insights,
    )

    usage_context = {
        "stage": "intake_conversation_extractor",
        "session_id": session_id,
    }

    try:
        raw = cheap_llm(
            system_prompt,
            user_prompt,
            expect_json=True,
            usage_context=usage_context,
        )
    except Exception:  # noqa: BLE001 — extractor failure is non-fatal
        record_llm_usage(
            provider="cheap_llm",
            model="(extractor)",
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
            },
            usage_context={
                "stage": "intake_conversation_extractor",
                "session_id": session_id,
                "result": "exception",
            },
        )
        return ExtractionResult({}, {})

    if not isinstance(raw, dict):
        return ExtractionResult({}, {})

    cleaned = _scrub_placeholders(raw)
    if not cleaned:
        return ExtractionResult({}, {})

    # Split insight-shaped keys off the cheap_llm output BEFORE any v2
    # validation runs. Insights live in their own bag (state_json
    # .intake_insights) and must never enter the v2_draft merge path.
    v2_updates_raw, insight_updates_raw = _split_insight_keys(cleaned)

    # Normalize each insight via the shared product rule. The normalizer
    # rejects trope-shaped output, drops below-floor summaries, and
    # filters placeholder-shaped values. Lock paths are honored by the
    # API persistence layer via merge_intake_insights — at this layer
    # we still pass through normalized updates because the recruiter
    # MAY be issuing a current-turn correction even on a locked path
    # (the cheap_llm prompt is responsible for only emitting under that
    # condition).
    insight_updates = _normalize_insight_payload(
        insight_updates_raw,
        current_v2_draft=current_v2_draft,
        source_packet=source_packet,
    )

    # Schema gate (v2 only): would the merged draft be structurally OK?
    # Tolerate missing-required-keys (incomplete draft is normal during
    # intake). Reject invalid-key shapes, but only the offending
    # top-level key(s) — P9.4: one bad key must not discard the whole
    # round's valid updates. Insight half is independent throughout.
    surviving_v2_updates, dropped_keys = _drop_invalid_v2_keys(
        current_v2_draft, v2_updates_raw, locked
    )
    if dropped_keys:
        log.warning(
            "intake_conversation_extractor extraction_partial "
            "session_id=%s dropped_keys=%s",
            session_id,
            dropped_keys,
        )
        record_llm_usage(
            provider="cheap_llm",
            model="(extractor)",
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
            },
            usage_context={
                "stage": "intake_conversation_extractor",
                "session_id": session_id,
                "result": "partial",
                "dropped_keys": list(dropped_keys),
            },
        )

    return ExtractionResult(surviving_v2_updates, insight_updates, dropped_keys)


def _split_insight_keys(
    cleaned: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition cheap_llm output into v2-shaped and insight-shaped buckets."""

    v2_updates: dict[str, Any] = {}
    insight_updates: dict[str, Any] = {}
    for key, value in cleaned.items():
        if key in _INSIGHT_TOP_LEVEL_KEYS:
            insight_updates[key] = value
        else:
            v2_updates[key] = value
    return v2_updates, insight_updates


def _normalize_insight_payload(
    raw: dict[str, Any],
    *,
    current_v2_draft: dict[str, Any],
    source_packet: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run each insight value through the shared normalizer.

    Source attribution is derived from whether a packet is present:
    ``"combined"`` when both packet and conversation are in play,
    ``"conversation"`` otherwise. The producer can override via the
    ``source`` key in the LLM output and the normalizer will honor the
    declared value if it matches :data:`VALID_SOURCES`.
    """

    if not raw:
        return {}
    role_context = _role_context_from(current_v2_draft, source_packet)
    derived_source = "combined" if isinstance(source_packet, dict) and source_packet else "conversation"
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


def _role_context_from(
    current_v2_draft: dict[str, Any] | None,
    source_packet: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a role-context dict for the insight normalizer.

    Pulls role title / summary / capability area names from v2_draft and
    JD / intake notes / file text from the source packet. The shared
    ``is_generic_trope`` rule walks this dict for token overlap.
    """

    ctx: dict[str, Any] = {}
    if isinstance(current_v2_draft, dict):
        for key in ("role_title", "role_summary"):
            value = current_v2_draft.get(key)
            if isinstance(value, str):
                ctx[key] = value
        cap = current_v2_draft.get("capability_areas")
        if isinstance(cap, list):
            ctx["capability_areas"] = cap
        for key in ("jd_text", "intake_notes"):
            value = current_v2_draft.get(key)
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


# -----------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------


def _build_extractor_system_prompt(*, manually_edited_keys: set[str]) -> str:
    """Build the cheap_llm system prompt.

    The manual-edit block is the load-bearing piece — see the module
    docstring contract #1. The placeholder list mirrors
    :data:`PLACEHOLDER_STRINGS` so the LLM can avoid emitting the same
    strings the post-LLM gate would scrub.
    """

    locked_block = _format_locked_block(manually_edited_keys)
    placeholder_examples = ", ".join(f'"{p}"' for p in PLACEHOLDER_STRINGS)

    # VERTICAL-VOCAB(intake-prompt-examples)
    return f"""You are a structured-data extractor for the Cloris conversational intake.

Your job: read the conversation transcript + current v2_draft + source packet (if present), and return ADDITIVE updates to the v2_draft as a JSON object. Each top-level key is a slot name; each value is what should be ADDED or UPDATED for that slot.

ABSOLUTE RULES:

1. Only return updates for values the recruiter EXPLICITLY stated or directly stated in the source packet. Never infer to fill an empty slot.
2. NEVER write placeholder strings. Examples of placeholders to AVOID: {placeholder_examples}.
3. If unsure about any slot, leave it alone. Empty is better than wrong.
4. Output JSON ONLY. No prose, no commentary.

{locked_block}

SCHEMA (the slots you may update):

- role_title: short string. The job title the role would post under. Short (2-6 words). Never a sentence.
- role_summary: one-sentence string. What the role exists to do.
- capability_areas: ARRAY of objects. Each object has `name` (short label, 1-3 words) and `description` (1-3 sentences). Capability areas are distinct skill domains the hire needs.
  - When updating capability_areas, return the FULL list (including any items the recruiter manually added that you should preserve verbatim). Do not drop or rename manually-added items.
- depth_distinction: object with three string keys: `builder_definition`, `user_definition`, `edge_case_guidance`. Only return sub-keys you have a definitive update for.
- non_fit_patterns: ARRAY of short strings. Profile shapes that look right but aren't.
- minimum_bar_description: string. The line below which no candidate passes.
- target_modules: ARRAY of source keys. Use explicit Cloris recommendations or recruiter corrections about where to look. Valid source keys are listed in the source capability manifest.
- source_strategy: ARRAY of objects with `source`, `role`, and `rationale`. `role` is one of primary, secondary, corroborating, investigation_first. This captures source recommendations separately from launchability.
- source_config: object. Contains per-source dicts like `{{"linkedin": {{"project_id": "..."}}}}`.
- hiring_manager_success_image: OBJECT with this exact shape (this slot does NOT live on v2_draft — it is a separate intake insight):
  {{
    "summary": "ONE sentence picture of the person the hiring manager actually wants. Vivid, role-anchored, recruiter-readable. Never write generic tropes like 'strong communication skills', 'team player', 'self-starter', 'rockstar'.",
    "proof_points": ["evidence the hiring manager would recognize as real, e.g. 'has shipped an applied AI lab inside a BFS firm'"],
    "screening_translation": "how this picture changes screening behavior, e.g. 'reject pure advisors who have only consulted on GenAI programs'",
    "confidence": 0.0-1.0,
    "source": "conversation" | "source_packet" | "combined",
    "corrected_by_recruiter": false
  }}
  - Emit this slot whenever the JD, source packet, or transcript supports a vivid picture. Empty is better than trope.
  - If the recruiter's MOST RECENT message explicitly corrects the picture ("actually, the picture is more X", "no, this person is more the boardroom shaper"), emit a fresh `hiring_manager_success_image` with `corrected_by_recruiter: true`. The API layer uses this flag to lock the field against subsequent extractor / synthesis writes.
  - Never emit `corrected_by_recruiter: true` on inference. The flag means "the recruiter just gave a current-turn correction," nothing else.

SOURCE CAPABILITY MANIFEST:
Evidence boundaries are not permission to skip a source; they define what a source cannot prove alone and what companion evidence completes the read.
{source_capability_prompt_block()}

OUTPUT FORMAT:
JSON object. Top-level keys are slot names. Omit any slot you have no update for.

Example output for a partial conversation:
{{"role_title": "Senior Tax Associate", "capability_areas": [{{"name": "Sales tax compliance", "description": "Owns multi-state sales tax filings end-to-end."}}]}}

Example output when the recruiter just said "let me think for a sec":
{{}}
"""


def _format_locked_block(locked: set[str]) -> str:
    """Render the manual-edit instruction block.

    Empty set produces a no-op block so the prompt stays uniform across
    turns (cheap_llm doesn't need to handle two prompt variants).
    """

    if not locked:
        return (
            "MANUALLY EDITED SLOTS: (none yet — extract freely subject to "
            "the absolute rules above)"
        )

    serialized = ", ".join(sorted(locked))
    return f"""MANUALLY EDITED SLOTS (the recruiter has filled these via the sidebar):
{serialized}

RULE: For any slot in the manually-edited list above, ONLY return an update if the recruiter's MOST RECENT message directly contradicts the current value. Never overwrite based on inference. Never overwrite based on earlier turns. If unsure, leave the slot alone.

If the recruiter says "actually, change the title to X" or otherwise gives an explicit, current-turn instruction to update a manually-edited slot, return the update. Otherwise, do not."""


def _build_extractor_user_prompt(
    *,
    messages: list[ConversationMessage],
    current_v2_draft: dict[str, Any],
    source_packet: dict[str, Any] | None,
    current_intake_insights: dict[str, Any] | None = None,
) -> str:
    """Build the cheap_llm user prompt.

    Same JSON-payload pattern as the orchestrator (and the existing
    ``brief_distillation.py`` distill path). The conversation, current
    draft, source packet, and current intake insights are serialized as
    a single document so cheap_llm can cross-reference latest-turn vs
    prior state. Insights are passed in alongside (not inside)
    ``current_v2_draft`` so the model sees the picture-in-progress
    without conflating it with v2 fields.
    """

    payload: dict[str, Any] = {
        "messages": [
            {"role": m.get("role"), "content": m.get("content")}
            for m in messages
        ],
        "current_v2_draft": current_v2_draft or {},
    }
    if source_packet:
        payload["source_packet"] = source_packet
    if current_intake_insights:
        payload["current_intake_insights"] = current_intake_insights

    return "INPUT:\n" + json.dumps(payload, indent=2)


# -----------------------------------------------------------------------
# Post-LLM gates
# -----------------------------------------------------------------------


def _scrub_placeholders(updates: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``updates`` with placeholder-shaped values dropped.

    Walks recursively. Strings matching :data:`PLACEHOLDER_STRINGS` are
    dropped (key removed from output). String values for ``role_title``
    that look like JD-prose-dumped-into-a-title are dropped via the
    ``kind="role_title"`` branch of :func:`_looks_like_placeholder`.
    """

    if not isinstance(updates, dict):
        return {}
    return _scrub_dict(updates, parent_key=None)


def _scrub_dict(value: dict[str, Any], *, parent_key: str | None) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for k, v in value.items():
        if isinstance(v, str):
            if _looks_like_placeholder(v, kind=k):
                continue
            cleaned[k] = v
        elif isinstance(v, dict):
            sub = _scrub_dict(v, parent_key=k)
            if sub:
                cleaned[k] = sub
        elif isinstance(v, list):
            sub_list = _scrub_list(v, parent_key=k)
            if sub_list:
                cleaned[k] = sub_list
        elif v is None:
            # Drop nulls — extractor should omit, not null-out.
            continue
        else:
            cleaned[k] = v
    return cleaned


def _scrub_list(value: list[Any], *, parent_key: str | None) -> list[Any]:
    cleaned: list[Any] = []
    for item in value:
        if isinstance(item, str):
            if _looks_like_placeholder(item):
                continue
            cleaned.append(item)
        elif isinstance(item, dict):
            sub = _scrub_dict(item, parent_key=parent_key)
            # Don't drop a capability_area just because some sub-fields
            # got scrubbed — only drop if the whole object emptied out.
            if sub:
                cleaned.append(sub)
        elif item is None:
            continue
        else:
            cleaned.append(item)
    return cleaned


def _top_level_key_from_invalid_descriptor(descriptor: str) -> str:
    """Reduce a :class:`BriefSchemaError` invalid-key descriptor to its
    top-level ``v2_draft`` key.

    Descriptors from :func:`shared.brief_v2_schema.validate_v2_brief` are
    either bare top-level keys (``"capability_areas"``) or dotted /
    bracketed paths into one (``"capability_areas[0]"``,
    ``"depth_distinction.builder_definition"``,
    ``"source_config.linkedin.project_id"``). Only the top-level key is
    ever an actual key of ``v2_updates`` — that's the unit this round's
    extraction can drop.
    """

    top = descriptor
    for sep in ("[", "."):
        idx = top.find(sep)
        if idx != -1:
            top = top[:idx]
    return top


def _drop_invalid_v2_keys(
    current_v2_draft: dict[str, Any],
    v2_updates: dict[str, Any],
    manually_edited_keys: set[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Iteratively drop this round's offending top-level v2 keys (P9.4).

    Tolerates missing-required-keys (incomplete draft is normal during
    intake) — that alone never drops anything. A structural shape error
    (``invalid_keys`` non-empty) drops only the offending top-level
    key(s) that THIS round proposed, then re-validates, so a round with
    several unrelated invalid keys still converges to the largest valid
    surviving subset instead of an all-or-nothing round.

    If the offending top-level key is NOT part of this round's
    ``v2_updates`` (the pre-existing draft was already broken,
    independent of this turn), the loop stops and the round's updates
    are accepted as-is — this round cannot fix pre-existing corruption
    by dropping its own unrelated keys, and pretending to would make the
    dropped-keys report a lie.

    Returns ``(surviving_updates, dropped_keys)``.
    """

    surviving = dict(v2_updates)
    dropped: list[str] = []
    while True:
        merged_preview = merge_extracted(
            current_v2_draft, surviving, manually_edited_keys=manually_edited_keys
        )
        try:
            validate_v2_brief(merged_preview)
            return surviving, tuple(dropped)
        except BriefSchemaError as exc:
            if not exc.invalid_keys:
                return surviving, tuple(dropped)
            offending_top = {
                _top_level_key_from_invalid_descriptor(k)
                for k in exc.invalid_keys
            }
            droppable = offending_top & surviving.keys()
            if not droppable:
                return surviving, tuple(dropped)
            for key in sorted(droppable):
                del surviving[key]
                dropped.append(key)


__all__ = ["ExtractionResult", "extract_slots"]
