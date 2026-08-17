"""Question-economy arbiter for conversational intake (pre-emit enforcement).

Every Cloris question must change a sourcing decision. When the source packet,
transcript, or capability manifest already answers the question, rewrite or skip
instead of asking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from shared.source_capabilities import (
    display_name_for_source,
    recommend_source_strategy_from_text,
)
from shared.source_packet import compose_source_packet_text

log = logging.getLogger(__name__)

QuestionEconomyAction = Literal["allow", "rewrite", "skip"]


@dataclass(frozen=True)
class QuestionEconomyVerdict:
    """Outcome of :func:`arbiter_cloris_turn`."""

    action: QuestionEconomyAction
    decision_affected: str | None = None
    replacement_text: str | None = None
    pattern: str | None = None


# Non-fit frequency / ranking before any search has run.
_NON_FIT_FREQUENCY_RE = re.compile(
    r"(?is)"
    r"(?:"
    r"which\s+non[- ]?fit"
    r"|non[- ]?fit.{0,40}(?:most\s+common|commonest|rank|frequency|how\s+often)"
    r"|(?:most\s+common|commonest).{0,40}non[- ]?fit"
    r"|calibrat.{0,40}non[- ]?fit"
    r"|prioriti[sz]e.{0,30}screen[- ]?out"
    r")"
)

# Open-ended source questions when the manifest can recommend a mix.
_WHERE_TO_LOOK_RE = re.compile(
    r"(?is)"
    r"(?:"
    r"where\s+should\s+(?:we|I)\s+look"
    r"|where\s+would\s+you\s+(?:look|search)"
    r"|where\s+to\s+look"
    r"|which\s+sources?\s+should"
    r"|what\s+sources?\s+should\s+(?:we|I)\s+use"
    r")"
    r".{0,80}\?"
)

def _evidence_text(
    *,
    v2_draft: dict[str, Any],
    source_packet: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
) -> str:
    parts: list[str] = []
    if isinstance(v2_draft, dict):
        for key in ("role_title", "role_summary"):
            val = v2_draft.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val)
        for area in v2_draft.get("capability_areas") or []:
            if isinstance(area, dict):
                for sub in ("name", "description"):
                    val = area.get(sub)
                    if isinstance(val, str) and val.strip():
                        parts.append(val)
    if isinstance(source_packet, dict):
        parts.append(
            compose_source_packet_text(
                job_description_text=str(source_packet.get("job_description_text") or ""),
                intake_notes_text=str(source_packet.get("intake_notes_text") or ""),
                files=source_packet.get("files") if isinstance(source_packet.get("files"), list) else [],
                gap_answer_history=[],
            )
        )
    if messages:
        for msg in messages[-6:]:
            if isinstance(msg, dict) and msg.get("role") == "recruiter":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content)
    return "\n".join(parts)


def _before_first_search(*, v2_draft: dict[str, Any], turn_count: int) -> bool:
    strategy = v2_draft.get("source_strategy") if isinstance(v2_draft, dict) else None
    has_strategy = isinstance(strategy, list) and bool(strategy)
    return turn_count <= 3 or not has_strategy


def _format_source_assumption(strategy: list[dict[str, str]]) -> str:
    """Render a recommendation from an already-computed source strategy.

    Callers only invoke this with a non-empty ``strategy`` (guaranteed by
    :func:`recommend_source_strategy_from_text`, which always seeds a
    LinkedIn-primary entry). No authored fallback for an empty list: the
    guard renders a computed recommendation, it does not invent one.
    """
    primary = next(
        (item for item in strategy if item.get("role") == "primary"),
        strategy[0],
    )
    source_key = str(primary.get("source") or "linkedin")
    name = display_name_for_source(source_key)
    extras = [
        display_name_for_source(str(item.get("source") or ""))
        for item in strategy[1:3]
        if isinstance(item, dict) and item.get("source") != source_key
    ]
    if extras:
        tail = ", with " + " and ".join(extras) + " as companion reads"
    else:
        tail = ""
    return (
        f"I'd run {name} as the primary search surface{tail} for this scope — "
        "correct me if that source mix is wrong."
    )


def _strip_matching_questions(text: str, pattern: re.Pattern[str]) -> str:
    """Remove sentences that end in ``?`` and match ``pattern``."""

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        if chunk.endswith("?") and pattern.search(chunk):
            continue
        kept.append(chunk)
    return " ".join(kept).strip()


def arbiter_cloris_turn(
    text: str,
    *,
    v2_draft: dict[str, Any] | None = None,
    source_packet: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    intake_insights: dict[str, Any] | None = None,
    turn_count: int = 0,
    sufficiency_state: tuple[bool, list[str]] | None = None,
) -> QuestionEconomyVerdict:
    """Classify a buffered Cloris turn: allow, rewrite, or skip.

    ``intake_insights`` is accepted for call-site signature stability
    (the API layer threads it through unconditionally) but is not read
    here: the guard's only prior use of it fed the now-deleted
    persona-assumption rewrite (P9.1) and had no other consumer.
    """

    _ = sufficiency_state  # reserved for future sufficiency-aware skips
    _ = intake_insights  # kept for signature stability; see docstring
    stripped = (text or "").strip()
    if not stripped:
        return QuestionEconomyVerdict(action="allow")

    draft = v2_draft if isinstance(v2_draft, dict) else {}

    if _NON_FIT_FREQUENCY_RE.search(stripped) and _before_first_search(
        v2_draft=draft, turn_count=turn_count
    ):
        replacement = _strip_matching_questions(stripped, _NON_FIT_FREQUENCY_RE)
        if not replacement:
            replacement = (
                "I'll capture the screen-out patterns in the brief and rank them "
                "once we've actually seen candidates — tell me what else would "
                "change how you'd evaluate the pool."
            )
        log.warning(
            "question_economy skip non_fit_frequency turn_count=%s",
            turn_count,
        )
        return QuestionEconomyVerdict(
            action="skip",
            decision_affected="non_fit_calibration",
            replacement_text=replacement,
            pattern="non_fit_frequency",
        )

    if _WHERE_TO_LOOK_RE.search(stripped):
        evidence = _evidence_text(
            v2_draft=draft, source_packet=source_packet, messages=messages
        )
        strategy = recommend_source_strategy_from_text(evidence)
        if len(strategy) >= 1 and len(evidence.strip()) >= 80:
            assumption = _format_source_assumption(strategy)
            replacement = _strip_matching_questions(stripped, _WHERE_TO_LOOK_RE)
            if replacement:
                replacement = f"{replacement} {assumption}"
            else:
                replacement = assumption
            log.warning(
                "question_economy skip where_to_look sources=%s",
                [item.get("source") for item in strategy],
            )
            return QuestionEconomyVerdict(
                action="skip",
                decision_affected="source_strategy",
                replacement_text=replacement,
                pattern="where_to_look",
            )

    return QuestionEconomyVerdict(action="allow")


def apply_question_economy(
    text: str,
    *,
    v2_draft: dict[str, Any] | None = None,
    source_packet: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    intake_insights: dict[str, Any] | None = None,
    turn_count: int = 0,
    sufficiency_state: tuple[bool, list[str]] | None = None,
) -> tuple[str, list[str]]:
    """Return ``(emitted_text, patterns)`` after question-economy enforcement."""

    verdict = arbiter_cloris_turn(
        text,
        v2_draft=v2_draft,
        source_packet=source_packet,
        messages=messages,
        intake_insights=intake_insights,
        turn_count=turn_count,
        sufficiency_state=sufficiency_state,
    )
    if verdict.action == "allow":
        return text, []
    if verdict.replacement_text is not None:
        return verdict.replacement_text, [verdict.pattern or verdict.action]
    return text, [verdict.pattern or verdict.action]


def apply_pre_emit_guards(
    text: str,
    *,
    v2_draft: dict[str, Any] | None = None,
    source_packet: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    intake_insights: dict[str, Any] | None = None,
    turn_count: int = 0,
    sufficiency_state: tuple[bool, list[str]] | None = None,
) -> tuple[str, list[str]]:
    """Run brief-dump guard then question economy; return emitted text + reasons."""

    from shared.intake_conversation.conversation_guard import apply_conversation_guard

    guarded, reasons = apply_conversation_guard(text)
    if "brief_dump_shape" in reasons:
        return guarded, reasons

    economical, patterns = apply_question_economy(
        guarded,
        v2_draft=v2_draft,
        source_packet=source_packet,
        messages=messages,
        intake_insights=intake_insights,
        turn_count=turn_count,
        sufficiency_state=sufficiency_state,
    )
    return economical, reasons + patterns


__all__ = [
    "QuestionEconomyAction",
    "QuestionEconomyVerdict",
    "apply_pre_emit_guards",
    "apply_question_economy",
    "arbiter_cloris_turn",
]
