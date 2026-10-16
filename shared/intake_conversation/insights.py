"""Shared product rule for intake-time insights that live alongside ``v2_draft``.

The ``hiring_manager_success_image`` insight is the first inhabitant of
``state_json.intake_insights``. It is **not** a V2 brief field. It is its own
bag, parallel to ``v2_draft``, so brief-schema validity and insight presence
remain independent concerns:

- ``shared.intake_conversation.state.merge_extracted`` and
  ``shared.brief_v2_schema.validate_v2_brief`` stay v2-shaped.
- This module owns normalization, trope rejection, presence checks, and
  the merge primitive for ``intake_insights``.

Three producers depend on this module: the per-turn extractor, the CTA-time
composer, and the source-packet synthesis worker. They each call
``normalize_hiring_manager_success_image`` on raw LLM output before persistence,
and they all merge writes through ``merge_intake_insights`` so recruiter
corrections (``corrected_by_recruiter=true`` plus a ``manually_edited_keys``
entry of ``"intake_insights.hiring_manager_success_image"``) survive.

Concept-level tests in ``tests/test_intake_hiring_manager_image.py`` and
certification both call ``is_missing_hiring_manager_success_image`` and
``is_generic_trope`` — there is intentionally only one place these rules live.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable

from market_intelligence.brief_distillation import (
    PLACEHOLDER_STRINGS,
    _looks_like_placeholder,
)


HIRING_MANAGER_PICTURE_KEY = "hiring_manager_success_image"
INTAKE_INSIGHTS_LOCK_PREFIX = "intake_insights"
HIRING_MANAGER_PICTURE_LOCK_PATH = (
    f"{INTAKE_INSIGHTS_LOCK_PREFIX}.{HIRING_MANAGER_PICTURE_KEY}"
)

# Floor lengths chosen to reject one-word tropes ("Strong leader.") while
# still allowing a tight one-sentence summary. Tuned against the realistic
# JD fixture's heuristic synthesis output.
_SUMMARY_MIN_LEN = 40
_SCREENING_TRANSLATION_MIN_LEN = 30
_PROOF_POINT_MIN_LEN = 12

VALID_SOURCES: frozenset[str] = frozenset(
    {"conversation", "source_packet", "combined", "heuristic"}
)

# Phrases that signal corporate-trope output regardless of role context. The
# list is intentionally small. Each entry is matched as a substring of the
# lowercased summary.
GENERIC_TROPE_PHRASES: tuple[str, ...] = (
    "strong communication skills",
    "strong communication",
    "team player",
    "self-starter",
    "self starter",
    "rockstar",
    "ninja",
    "wears many hats",
    "go-getter",
    "passionate about",
    "results-oriented",
    "results oriented",
    "detail-oriented",
    "detail oriented",
    "strong work ethic",
    "thinks outside the box",
    "fast-paced environment",
)

# Tokens to ignore when computing role-context noun overlap. Common stopwords
# plus the obvious recruiting filler.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
        "for", "from", "has", "have", "he", "her", "him", "his", "how", "i",
        "in", "is", "it", "its", "of", "on", "or", "our", "she", "so", "that",
        "the", "their", "them", "they", "this", "to", "us", "was", "we",
        "were", "what", "when", "where", "which", "who", "why", "will",
        "with", "you", "your", "role", "team", "company", "person", "people",
        "someone", "candidate", "hire", "hiring", "manager", "leader",
        "leadership", "senior", "junior", "engineer", "engineering",
        "experience", "skills", "skill", "ability", "abilities", "strong",
        "good", "great", "must", "should", "would", "could", "needs",
        "looking",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-/]{2,}")


def _coerce_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _coerce_proof_points(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _coerce_str(item)
        if not text:
            continue
        if _looks_like_placeholder(text):
            continue
        if any(text == p for p in PLACEHOLDER_STRINGS):
            continue
        if len(text) < _PROOF_POINT_MIN_LEN:
            continue
        out.append(text[:400])
    # De-duplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if number < 0.0:
            return 0.0
        if number > 1.0:
            return 1.0
        return number
    return 0.0


def _coerce_source(value: Any) -> str:
    text = _coerce_str(value)
    if text in VALID_SOURCES:
        return text
    return ""


def _role_context_tokens(role_context: Any) -> set[str]:
    """Extract the set of role-anchor tokens from arbitrary role context.

    Accepts a dict (typically with ``role_title`` / ``role_summary`` / a
    free-form ``text`` blob), a string, or an iterable of strings. The
    comparison is intentionally lossy: lowercased substrings of length >=3
    minus stopwords. We are looking for evidence that the summary references
    something specific from the role, not for semantic similarity.
    """

    chunks: list[str] = []
    if isinstance(role_context, dict):
        for key in ("role_title", "role_summary", "text", "jd_text", "intake_notes"):
            piece = role_context.get(key)
            if isinstance(piece, str):
                chunks.append(piece)
        # Capability area names are good role anchors (e.g. "agentic design",
        # "applied AI"). Pull the names so the BFS-style fixture matches.
        cap = role_context.get("capability_areas")
        if isinstance(cap, list):
            for item in cap:
                if isinstance(item, dict):
                    name = item.get("name")
                    if isinstance(name, str):
                        chunks.append(name)
    elif isinstance(role_context, str):
        chunks.append(role_context)
    elif isinstance(role_context, Iterable):
        for piece in role_context:
            if isinstance(piece, str):
                chunks.append(piece)

    tokens: set[str] = set()
    for chunk in chunks:
        for match in _TOKEN_RE.findall(chunk.lower()):
            if match in _STOPWORDS:
                continue
            tokens.add(match)
    return tokens


def is_generic_trope(summary: str, role_context: Any) -> bool:
    """Return True iff ``summary`` reads as corporate trope.

    Two-step check:

    1. **Phrase banlist.** If the summary contains any phrase from
       :data:`GENERIC_TROPE_PHRASES` (case-insensitive substring), it is a
       trope regardless of context.
    2. **Role-context anchor.** If the summary lacks any token overlap with
       the role context (no role-specific noun appears at all), it is a trope
       — generic praise unmoored from the role. Empty role context skips
       this check (cannot prove triviality).

    Used by all three producers (extractor, composer, synthesis worker) and
    by certification.
    """

    text = (summary or "").strip()
    if not text:
        return True
    lower = text.lower()
    for phrase in GENERIC_TROPE_PHRASES:
        if phrase in lower:
            return True
    role_tokens = _role_context_tokens(role_context)
    if not role_tokens:
        return False
    summary_tokens: set[str] = set()
    for match in _TOKEN_RE.findall(lower):
        if match in _STOPWORDS:
            continue
        summary_tokens.add(match)
    if not summary_tokens & role_tokens:
        return True
    return False


def normalize_hiring_manager_success_image(
    raw: Any,
    role_context: Any,
    *,
    source: str,
) -> dict[str, Any] | None:
    """Coerce raw LLM output into the canonical insight shape, or return None.

    Returns ``None`` when the payload is unusable: not a dict, missing the
    load-bearing fields (``summary`` + ``screening_translation`` + at least
    one ``proof_point``), the summary is below the floor length, the
    placeholder gate scrubs the summary, or the trope guard rejects it.

    On success returns a dict with the canonical six keys, with
    ``corrected_by_recruiter`` defaulting to False when the producer did
    not signal a current-turn correction.
    """

    if not isinstance(raw, dict):
        return None

    summary = _coerce_str(raw.get("summary"))
    if not summary or len(summary) < _SUMMARY_MIN_LEN:
        return None
    if _looks_like_placeholder(summary):
        return None
    if any(summary == p for p in PLACEHOLDER_STRINGS):
        return None

    screening_translation = _coerce_str(raw.get("screening_translation"))
    if not screening_translation or len(screening_translation) < _SCREENING_TRANSLATION_MIN_LEN:
        return None
    if _looks_like_placeholder(screening_translation):
        return None

    proof_points = _coerce_proof_points(raw.get("proof_points"))
    if not proof_points:
        return None

    if is_generic_trope(summary, role_context):
        return None

    confidence = _coerce_confidence(raw.get("confidence"))

    declared_source = _coerce_source(raw.get("source"))
    chosen_source = declared_source or _coerce_source(source) or "heuristic"

    corrected_raw = raw.get("corrected_by_recruiter")
    corrected = bool(corrected_raw) if isinstance(corrected_raw, bool) else False

    return {
        "summary": summary[:600],
        "proof_points": proof_points[:6],
        "screening_translation": screening_translation[:600],
        "confidence": confidence,
        "source": chosen_source,
        "corrected_by_recruiter": corrected,
    }


def is_missing_hiring_manager_success_image(insight: Any) -> bool:
    """Return True iff the insight is absent or structurally incomplete.

    Used by CTA readiness, certification, and the frontend deficit signal.
    The check matches the contract enforced by
    :func:`normalize_hiring_manager_success_image` so any populated insight
    that came through the normalizer is automatically not-missing.
    """

    if insight is None:
        return True
    if not isinstance(insight, dict):
        return True
    if not insight:
        return True
    summary = _coerce_str(insight.get("summary"))
    if not summary:
        return True
    screening = _coerce_str(insight.get("screening_translation"))
    if not screening:
        return True
    proof = insight.get("proof_points")
    if not isinstance(proof, list):
        return True
    if not any(_coerce_str(item) for item in proof):
        return True
    return False


def merge_intake_insights(
    current: dict[str, Any],
    updates: dict[str, Any],
    *,
    manually_edited_keys: set[str] | list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Mirror of ``shared.intake_conversation.state.merge_extracted`` for the
    ``intake_insights`` bag.

    Lock paths are dot-paths under the ``intake_insights`` namespace, e.g.
    ``"intake_insights.hiring_manager_success_image"``. The set is shared
    with v2 manual-edit locks via ``state_json.conversation_meta
    .manually_edited_keys``; the prefix discriminator keeps the namespaces
    from colliding.

    Insight values are top-level dicts (one per insight key). They are
    replaced wholesale on update (no by-element merge); a recruiter
    correction is by definition a fresh full picture, not a sub-edit.

    Returns a NEW dict; never mutates input. Empty ``updates`` returns a
    deep copy of ``current`` unchanged.
    """

    locked = set(manually_edited_keys or ())
    result: dict[str, Any] = copy.deepcopy(current) if isinstance(current, dict) else {}
    if not isinstance(updates, dict) or not updates:
        return result
    for key, value in updates.items():
        path = f"{INTAKE_INSIGHTS_LOCK_PREFIX}.{key}"
        if path in locked:
            continue
        if value is None:
            continue
        result[key] = copy.deepcopy(value)
    return result


__all__ = [
    "GENERIC_TROPE_PHRASES",
    "HIRING_MANAGER_PICTURE_KEY",
    "HIRING_MANAGER_PICTURE_LOCK_PATH",
    "INTAKE_INSIGHTS_LOCK_PREFIX",
    "VALID_SOURCES",
    "is_generic_trope",
    "is_missing_hiring_manager_success_image",
    "merge_intake_insights",
    "normalize_hiring_manager_success_image",
]
