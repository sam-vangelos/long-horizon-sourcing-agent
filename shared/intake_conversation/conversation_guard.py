"""Pre-emit guards for Cloris conversational intake turns.

Detects brief-dump shapes — structured brief prose, schema field names, or
checklist read-backs that belong on the review surface, not in chat.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Recruiter-safe redirect when the model dumps the brief into chat.
BRIEF_DUMP_REPLACEMENT = (
    "I've got the pieces in the draft — use Show me the brief when you want "
    "the structured read-back. I'll keep clarifying here in conversation form."
)

# Internal v2 / review-surface tokens that must not appear in chat turns.
_SCHEMA_FIELD_TOKENS: tuple[str, ...] = (
    "capability_areas",
    "depth_distinction",
    "non_fit_patterns",
    "employer_signal_rules",
    "source_strategy",
    "role_summary",
    "minimum_bar",
    "builder_definition",
    "user_definition",
    "edge_case_guidance",
    "hiring_manager_success_image",
)

_SECTION_HEADER_RE = re.compile(
    r"(?m)^#{1,3}\s+(role|capability|capabilities|non[- ]?fit|source|depth|minimum|employer)\b",
    re.IGNORECASE,
)

_LABELED_SECTION_RE = re.compile(
    r"(?m)^(?:\*\*)?(?:role title|capability areas?|non[- ]?fit patterns?|"
    r"depth distinction|source strategy|minimum bar|employer signal)"
    r"(?:\*\*)?\s*:",
    re.IGNORECASE,
)

_JSON_KEY_RE = re.compile(
    r'"(?:role_title|capability_areas|non_fit_patterns|source_strategy)"\s*:'
)

_NUMBERED_SECTION_RE = re.compile(r"(?m)^\d+\.\s+\S{3,}")

_HARD_GATE_RE = re.compile(r"\(HARD GATE\)", re.IGNORECASE)


@dataclass(frozen=True)
class ConversationGuardResult:
    """Outcome of :func:`guard_cloris_turn`."""

    blocked: bool
    reason: str | None = None
    replacement_text: str | None = None


def detect_brief_dump_shape(text: str) -> bool:
    """Return True when ``text`` reads like a structured brief, not chat."""

    stripped = (text or "").strip()
    if not stripped:
        return False

    lowered = stripped.lower()
    score = 0

    schema_hits = sum(1 for token in _SCHEMA_FIELD_TOKENS if token in lowered)
    if schema_hits >= 2:
        score += 2
    if schema_hits >= 3:
        return True

    if _JSON_KEY_RE.search(stripped):
        score += 2

    if len(_SECTION_HEADER_RE.findall(stripped)) >= 2:
        score += 2

    labeled = _LABELED_SECTION_RE.findall(stripped)
    if len(labeled) >= 3:
        return True
    if len(labeled) >= 2:
        score += 1

    if _HARD_GATE_RE.search(stripped):
        return True

    bullet_lines = [
        line
        for line in stripped.splitlines()
        if line.lstrip().startswith(("- ", "* ", "• "))
    ]
    if len(bullet_lines) >= 6 and len(stripped) >= 600:
        score += 2

    numbered = _NUMBERED_SECTION_RE.findall(stripped)
    if len(numbered) >= 4 and len(stripped) >= 500:
        score += 2

    # Long memo-shaped turns with multiple pseudo-headings (Title Case lines).
    title_lines = sum(
        1
        for line in stripped.splitlines()
        if re.match(r"^[A-Z][A-Za-z0-9 /\-]{2,40}:?\s*$", line.strip())
    )
    if title_lines >= 4 and len(stripped) >= 700:
        score += 1

    return score >= 3


def guard_cloris_turn(text: str) -> ConversationGuardResult:
    """Block brief-dump shapes before SSE emit or transcript persist."""

    if not (text or "").strip():
        return ConversationGuardResult(blocked=False)

    if detect_brief_dump_shape(text):
        log.warning(
            "conversation_guard brief_dump_shape chars=%s preview=%r",
            len(text),
            text[:240],
        )
        return ConversationGuardResult(
            blocked=True,
            reason="brief_dump_shape",
            replacement_text=BRIEF_DUMP_REPLACEMENT,
        )

    return ConversationGuardResult(blocked=False)


def apply_conversation_guard(
    text: str,
    *,
    meta: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Return ``(emitted_text, reasons)`` after the brief-dump guard."""

    _ = meta  # reserved for future turn-plan metadata
    result = guard_cloris_turn(text)
    if result.blocked:
        return result.replacement_text or BRIEF_DUMP_REPLACEMENT, [
            result.reason or "brief_dump_shape"
        ]
    return text, []


__all__ = [
    "BRIEF_DUMP_REPLACEMENT",
    "ConversationGuardResult",
    "apply_conversation_guard",
    "detect_brief_dump_shape",
    "guard_cloris_turn",
]
