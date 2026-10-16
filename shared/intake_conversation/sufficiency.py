"""Deterministic sufficiency check for the conversational intake.

The C5 SSE endpoint runs :func:`is_ready_to_compose` after every
extraction round. The orchestrator's prompt is told whether sufficiency
has been reached so it can volunteer 'I think I have what I need…'; the
SSE wire event is sourced from this deterministic check, NOT from
LLM-emitted JSON. Keeps the volunteer behavior judgmental (Cloris
chooses when to surface) while keeping the cross-the-line semantics
predictable (deterministic from v2_draft state).

Threshold (per the source plan):

- ``role_title`` is non-empty and not a placeholder, AND
- at least one ``capability_areas`` entry has a non-placeholder
  description, AND
- ``role_summary`` is non-empty (and not a placeholder) OR at least one
  ``depth_distinction`` sub-field is non-empty (and not a placeholder).

Returns ``(ready, missing)`` where ``missing`` is a dot-path list of
what would need to land to flip ``ready`` to True. The orchestrator's
system prompt uses ``missing`` to focus its next question (or, when
ready, to compose the volunteer message).
"""

from __future__ import annotations

from typing import Any

from market_intelligence.brief_distillation import _looks_like_placeholder


_DEPTH_SUBKEYS: tuple[str, ...] = (
    "builder_definition",
    "user_definition",
    "edge_case_guidance",
)


def is_ready_to_compose(v2_draft: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return ``(ready, missing_dot_paths)`` for the brief draft.

    ``missing_dot_paths`` is the smallest set of slots that would, if
    filled, flip ``ready`` to True. When ``ready`` is True, the list is
    always empty. When False, the list always contains at least one
    entry, and is ordered so the most foundational slot is first
    (recruiter sees "role_title" before "capability_areas[0].description"
    before "role_summary OR depth_distinction").
    """

    if not isinstance(v2_draft, dict):
        return False, ["role_title", "capability_areas[0].description", "role_summary"]

    missing: list[str] = []

    if not _is_real_string(v2_draft.get("role_title"), kind="role_title"):
        missing.append("role_title")

    if not _has_real_capability(v2_draft):
        missing.append("capability_areas[0].description")

    if not (
        _is_real_string(v2_draft.get("role_summary"))
        or _has_real_depth_distinction(v2_draft)
    ):
        missing.append("role_summary OR depth_distinction.{builder_definition|user_definition|edge_case_guidance}")

    return (not missing), missing


def _is_real_string(value: Any, *, kind: str | None = None) -> bool:
    """True iff ``value`` is a non-empty string that isn't a placeholder."""

    if not isinstance(value, str):
        return False
    if not value.strip():
        return False
    if _looks_like_placeholder(value, kind=kind):
        return False
    return True


def _has_real_capability(v2_draft: dict[str, Any]) -> bool:
    """True iff at least one capability_areas entry has a real description."""

    cap_areas = v2_draft.get("capability_areas")
    if not isinstance(cap_areas, list) or not cap_areas:
        return False
    for entry in cap_areas:
        if not isinstance(entry, dict):
            continue
        # name presence is necessary but not sufficient — we want a real
        # description so sourcing has something to ground on.
        if not _is_real_string(entry.get("name")):
            continue
        if _is_real_string(entry.get("description")):
            return True
    return False


def _has_real_depth_distinction(v2_draft: dict[str, Any]) -> bool:
    """True iff any depth_distinction sub-field is a real string."""

    depth = v2_draft.get("depth_distinction")
    if not isinstance(depth, dict):
        return False
    return any(_is_real_string(depth.get(k)) for k in _DEPTH_SUBKEYS)


__all__ = ["is_ready_to_compose"]
