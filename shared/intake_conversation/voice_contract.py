"""Phrase cooldowns and voice-contract helpers for conversational intake.

Single source of truth for intake-specific phrase economy. The orchestrator
prompt imports :data:`PHRASE_COOLDOWNS` for model guidance; post-hoc tests
in ``tests/intake_conversation/voice_asserts.py`` enforce the same list.
"""

from __future__ import annotations

import re
from typing import Any

# (phrase, min_turn_gap) — a phrase may not reappear until at least
# ``min_turn_gap`` Cloris turns have elapsed since its last use.
# Large gaps (999) mean effectively once per conversation.
PHRASE_COOLDOWNS: tuple[tuple[str, int], ...] = (
    ("one thing", 4),
    ("before i draft", 999),
    ("before i start scoping", 999),
    ("couple things", 5),
    ("couple of things", 5),
)

# Structural markers that indicate a Cloris turn is rendering a brief
# deliverable inside chat instead of routing to the review surface.
_BRIEF_DUMP_HEADER_RE = re.compile(
    r"(?m)^\s*(?:#{1,3}\s+\S|\*\*[A-Za-z][^*\n]{2,40}:\*\*)"
)
_BRIEF_DUMP_FIELD_LABELS = (
    "role title",
    "role summary",
    "capability areas",
    "depth distinction",
    "non-fit patterns",
    "where i would look",
    "target modules",
    "minimum bar",
)

_SOURCE_TOKEN_RE = re.compile(r"\b[a-z]{4,}\b")
_SOURCE_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "before",
        "could",
        "does",
        "from",
        "have",
        "into",
        "look",
        "more",
        "that",
        "their",
        "there",
        "these",
        "thing",
        "things",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "your",
    }
)


def phrase_cooldown_violations(
    messages: list[dict[str, Any]],
) -> list[str]:
    """Return human-readable violations when cooldown phrases repeat too soon."""

    last_seen: dict[str, int] = {}
    violations: list[str] = []
    cloris_turn = 0

    for msg in messages:
        if msg.get("role") != "cloris":
            continue
        cloris_turn += 1
        content = str(msg.get("content") or "").lower()
        for phrase, gap in PHRASE_COOLDOWNS:
            if phrase not in content:
                continue
            prior = last_seen.get(phrase)
            if prior is not None and (cloris_turn - prior) < gap:
                violations.append(
                    f"phrase {phrase!r} reused after {cloris_turn - prior} "
                    f"Cloris turns (cooldown {gap})"
                )
            last_seen[phrase] = cloris_turn

    return violations


def looks_like_brief_dump(text: str) -> bool:
    """True when ``text`` reads like a structured brief pasted into chat."""

    if not text.strip():
        return False

    header_hits = len(_BRIEF_DUMP_HEADER_RE.findall(text))
    if header_hits >= 2:
        return True
    if header_hits == 1 and len(text) > 400:
        return True

    lowered = text.lower()
    label_hits = sum(1 for label in _BRIEF_DUMP_FIELD_LABELS if label in lowered)
    if label_hits >= 3:
        return True
    if label_hits >= 2 and len(text) > 500:
        return True

    return False


def source_text_from_packet(source_packet: dict[str, Any] | None) -> str:
    """Flatten string fields from a source packet for overlap checks."""

    if not isinstance(source_packet, dict):
        return ""

    parts: list[str] = []
    for key in ("job_description_text", "intake_notes_text", "geography"):
        val = source_packet.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    files = source_packet.get("files")
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
    return "\n".join(parts)


def source_overlap_ratio(
    question: str,
    source_packet: dict[str, Any] | None,
    *,
    min_token_len: int = 4,
) -> float:
    """Return the fraction of substantive question tokens present in source text.

    High overlap on a question-bearing turn suggests Cloris is re-asking
    something the JD already answered instead of inferring and moving on.
    Returns ``0.0`` when there is no question, no source text, or no tokens.
    """

    if "?" not in question:
        return 0.0

    source = source_text_from_packet(source_packet).lower()
    if not source.strip():
        return 0.0

    tokens = [
        match.group(0)
        for match in _SOURCE_TOKEN_RE.finditer(question.lower())
        if match.group(0) not in _SOURCE_STOPWORDS
    ]
    if not tokens:
        return 0.0

    hits = sum(1 for token in tokens if token in source)
    return hits / len(tokens)


__all__ = [
    "PHRASE_COOLDOWNS",
    "looks_like_brief_dump",
    "phrase_cooldown_violations",
    "source_overlap_ratio",
    "source_text_from_packet",
]
