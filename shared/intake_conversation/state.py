"""Pure helpers for mutating ``intake_sessions.state_json`` in conversational
intake flows.

Every function returns a NEW dict — no caller mutates state in place. The
state shape these helpers operate over:

.. code-block:: jsonc

   {
     "messages": [
       {"role": "cloris" | "recruiter", "content": "...", "ts": "...", "meta": {...}}
     ],
     "v2_draft": { ... },          // existing brief schema, unchanged
     "source_packet": { ... },     // existing — JD / notes paste
     "conversation_meta": {
       "turn_count": int,
       "last_extraction_at_turn": int,
       "cost_usd_running_total": float,
       "ready_to_compose": bool,
       "manually_edited_keys": [str, ...]   // dot-paths
     }
     // legacy wizard keys (gap_questions, distillation, etc.) preserved
     // for ?legacy=1 sessions; conversational flow ignores them.
   }

Manual-edit conflict resolution: the load-bearing logic lives in the
:mod:`shared.intake_conversation.extractor` cheap_llm prompt, NOT in code.
The :func:`merge_extracted` helper here is a deliberate defense-in-depth
backstop — when the model returns updates for manually-edited keys, the
merge silently drops them. It is conservative on conflict: when in doubt,
preserve the manual edit. No string-matching, no negation regex, no
inference.
"""

from __future__ import annotations

import copy
from typing import Any

from shared.intake_conversation import ConversationMessage


def append_message(
    state_json: dict[str, Any], msg: ConversationMessage
) -> dict[str, Any]:
    """Return a new state dict with ``msg`` appended to ``messages``.

    Idempotent on the (role, ts, content) tuple of the most recent
    message: a duplicate append is a no-op so retried writes during
    network flakes don't double-stuff the transcript. The check is the
    last message only — older duplicates are not deduped because there
    are legitimate cases of identical short replies later in a session.
    """

    result = dict(state_json)
    messages = list(result.get("messages") or [])
    if messages:
        last = messages[-1]
        if (
            last.get("role") == msg.get("role")
            and last.get("ts") == msg.get("ts")
            and last.get("content") == msg.get("content")
        ):
            result["messages"] = messages
            return result
    messages.append(dict(msg))
    result["messages"] = messages
    return result


def detect_dropped_turn(state_json: dict[str, Any]) -> bool:
    """True iff the last persisted message is a recruiter turn.

    The orchestrator's stream died mid-flight if the recruiter sent a
    message but no Cloris response was persisted before the connection
    closed. The C5 endpoint calls this on session load and threads the
    result into the orchestrator's system prompt so Cloris resumes the
    interrupted thread rather than treating the recruiter's last message
    as a new send.

    False on empty sessions and on sessions whose last message is from
    Cloris (the normal post-turn state).
    """

    messages = state_json.get("messages") or []
    if not messages:
        return False
    return messages[-1].get("role") == "recruiter"


def bump_conversation_meta(
    state_json: dict[str, Any],
    *,
    turn_delta: int = 1,
    cost_delta_usd: float = 0.0,
) -> dict[str, Any]:
    """Return a new state dict with ``conversation_meta`` counters bumped.

    Initializes the ``conversation_meta`` sub-dict on first call so the
    caller never has to handle the missing case. ``last_extraction_at_turn``
    is preserved (or set to 0 on first call) so the extractor can decide
    whether to re-run incrementally.
    """

    result = dict(state_json)
    meta = dict(result.get("conversation_meta") or {})
    meta["turn_count"] = int(meta.get("turn_count", 0)) + int(turn_delta)
    meta["cost_usd_running_total"] = float(
        meta.get("cost_usd_running_total", 0.0)
    ) + float(cost_delta_usd)
    meta.setdefault("last_extraction_at_turn", 0)
    meta.setdefault("manually_edited_keys", [])
    result["conversation_meta"] = meta
    return result


def merge_extracted(
    v2_draft: dict[str, Any],
    updates: dict[str, Any],
    *,
    manually_edited_keys: set[str] | list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Additively merge ``updates`` into a deep copy of ``v2_draft``.

    Defense-in-depth backstop for the extractor prompt's manual-edit
    rule. The extractor is INSTRUCTED not to overwrite manually-edited
    slots without an explicit recruiter contradiction in the latest
    turn; this merge silently drops any update whose dot-path matches an
    entry in ``manually_edited_keys``, in case the extractor misbehaves.

    Path semantics: a top-level key like ``"role_title"`` blocks the
    whole top-level update for that key. A nested path like
    ``"depth_distinction.builder_definition"`` blocks only that
    sub-field — sibling sub-fields under ``depth_distinction`` still
    merge. Lists are replaced wholesale (no by-element merge); the v0
    extractor returns whole capability_areas lists, not partial ones,
    so this matches the wire shape.

    Returns a NEW dict; never mutates input. Empty ``updates`` returns
    a deep copy of ``v2_draft`` unchanged.
    """

    locked = set(manually_edited_keys or ())
    result = copy.deepcopy(v2_draft) if v2_draft else {}
    if not updates:
        return result

    for key, value in updates.items():
        if key in locked:
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            _deep_merge(
                result[key], value, prefix=key, locked=locked
            )
        elif isinstance(value, dict):
            # Filter dropped paths from the new sub-dict before assigning.
            result[key] = _filter_locked(value, prefix=key, locked=locked)
        else:
            result[key] = value
    return result


def _deep_merge(
    dst: dict[str, Any],
    src: dict[str, Any],
    *,
    prefix: str,
    locked: set[str],
) -> None:
    """In-place merge of ``src`` into ``dst``, respecting ``locked`` paths.

    Internal helper for :func:`merge_extracted`. Mutates ``dst`` because
    the public surface already deep-copied; callers outside this module
    should never invoke directly.
    """

    for k, v in src.items():
        path = f"{prefix}.{k}"
        if path in locked:
            continue
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v, prefix=path, locked=locked)
        elif isinstance(v, dict):
            dst[k] = _filter_locked(v, prefix=path, locked=locked)
        else:
            dst[k] = v


def _filter_locked(
    src: dict[str, Any], *, prefix: str, locked: set[str]
) -> dict[str, Any]:
    """Return a deep copy of ``src`` with locked dot-paths removed.

    Used when merging a new sub-dict into a slot that didn't previously
    exist (or wasn't a dict). Keeps the locked-path contract holding even
    on first-write paths.
    """

    result: dict[str, Any] = {}
    for k, v in src.items():
        path = f"{prefix}.{k}"
        if path in locked:
            continue
        if isinstance(v, dict):
            result[k] = _filter_locked(v, prefix=path, locked=locked)
        else:
            result[k] = copy.deepcopy(v)
    return result


__all__ = [
    "append_message",
    "bump_conversation_meta",
    "detect_dropped_turn",
    "merge_extracted",
]
