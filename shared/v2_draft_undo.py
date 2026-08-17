"""Multi-level undo buffer for in-flight intake ``v2_draft`` edits."""

from __future__ import annotations

import copy
from typing import Any

MAX_UNDO = 5


def push_v2_undo(state: dict[str, Any]) -> None:
    """Append a snapshot of the current draft before a mutating operation."""

    v2 = state.get("v2_draft")
    if not isinstance(v2, dict):
        return
    snapshot: dict[str, Any] = {"v2_draft": copy.deepcopy(v2)}
    meta = state.get("v2_draft_polish_meta")
    if isinstance(meta, dict):
        snapshot["polish_meta"] = copy.deepcopy(meta)
    raw_stack = state.get("v2_draft_undo_stack")
    stack = [x for x in raw_stack if isinstance(x, dict)] if isinstance(raw_stack, list) else []
    stack.append(snapshot)
    state["v2_draft_undo_stack"] = stack[-MAX_UNDO:]
    state["v2_draft_prev"] = copy.deepcopy(state["v2_draft_undo_stack"][-1])


def pop_v2_undo_into_state(state: dict[str, Any]) -> bool:
    """Restore ``v2_draft`` from the newest undo snapshot."""

    raw_stack = state.get("v2_draft_undo_stack")
    stack = [x for x in raw_stack if isinstance(x, dict)] if isinstance(raw_stack, list) else []
    snapshot: dict[str, Any] | None = None
    if stack:
        snapshot = stack.pop()
        if stack:
            state["v2_draft_undo_stack"] = stack
            state["v2_draft_prev"] = copy.deepcopy(stack[-1])
        else:
            state.pop("v2_draft_undo_stack", None)
            state.pop("v2_draft_prev", None)
    else:
        prev = state.get("v2_draft_prev")
        if isinstance(prev, dict) and isinstance(prev.get("v2_draft"), dict):
            snapshot = prev
            state.pop("v2_draft_prev", None)
            state.pop("v2_draft_undo_stack", None)
    if snapshot is None or not isinstance(snapshot.get("v2_draft"), dict):
        return False
    state["v2_draft"] = copy.deepcopy(snapshot["v2_draft"])
    meta = snapshot.get("polish_meta")
    if isinstance(meta, dict):
        state["v2_draft_polish_meta"] = copy.deepcopy(meta)
    else:
        state.pop("v2_draft_polish_meta", None)
    return True
