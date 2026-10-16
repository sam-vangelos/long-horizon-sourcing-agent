"""Tests for ``shared.intake_conversation.state`` helpers (Phase C1).

Covers the four pure functions that mutate ``intake_sessions.state_json``
during a conversational intake turn:

- :func:`append_message` — round-trip + idempotency on retried writes.
- :func:`detect_dropped_turn` — recruiter-tail / cloris-tail / empty cases.
- :func:`bump_conversation_meta` — counter init + delta semantics.
- :func:`merge_extracted` — additive merge + manual-edit defense-in-depth.

The merge tests are the load-bearing ones — :func:`merge_extracted` is the
backstop for the C3 extractor's manual-edit prompt rule. If the extractor
misbehaves (returns updates for a manually-edited key without an explicit
recruiter contradiction), this merge has to silently drop them. The
contract: when in doubt, preserve the manual edit.

Round-trip through ``cloris.intake_sessions.patch_intake_session`` is
asserted at the end so the in-memory shape survives JSON serialization
through SQLite without losing keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cloris import intake_sessions as intake_module
from shared.intake_conversation import ConversationMessage
from shared.intake_conversation.state import (
    append_message,
    bump_conversation_meta,
    detect_dropped_turn,
    merge_extracted,
)
from shared.runtime_state.store import RuntimeStateStore


def _msg(role: str, content: str, ts: str = "2026-05-13T12:00:00+00:00") -> ConversationMessage:
    return {"role": role, "content": content, "ts": ts}  # type: ignore[typeddict-item]


# ---------------------------------------------------------------------------
# append_message
# ---------------------------------------------------------------------------


def test_append_message_appends_to_empty_state() -> None:
    state = {}
    msg = _msg("recruiter", "hi")

    result = append_message(state, msg)

    assert result["messages"] == [{"role": "recruiter", "content": "hi", "ts": "2026-05-13T12:00:00+00:00"}]
    # Original input untouched.
    assert state == {}


def test_append_message_appends_to_existing_messages() -> None:
    state = {"messages": [_msg("recruiter", "first", ts="2026-05-13T12:00:00+00:00")]}

    result = append_message(state, _msg("cloris", "second", ts="2026-05-13T12:00:01+00:00"))

    assert len(result["messages"]) == 2
    assert result["messages"][-1]["content"] == "second"


def test_append_message_idempotent_on_duplicate_tail() -> None:
    """A retried write with identical (role, ts, content) is a no-op.

    Network flakes on the SSE persist path can fire the same append
    twice. Production state must not double-stuff the transcript.
    """

    msg = _msg("recruiter", "duplicate", ts="2026-05-13T12:00:00+00:00")
    state = {"messages": [dict(msg)]}

    result = append_message(state, msg)

    assert len(result["messages"]) == 1


def test_append_message_does_not_dedupe_non_tail_duplicates() -> None:
    """Older identical messages don't suppress later legitimate appends.

    "Got it." or other short replies can recur legitimately many turns
    later. Dedup is tail-only on purpose.
    """

    msg = _msg("cloris", "Got it.", ts="2026-05-13T12:00:00+00:00")
    state = {
        "messages": [
            dict(msg),
            _msg("recruiter", "another thing", ts="2026-05-13T12:01:00+00:00"),
        ]
    }
    later = _msg("cloris", "Got it.", ts="2026-05-13T12:02:00+00:00")

    result = append_message(state, later)

    assert len(result["messages"]) == 3
    assert result["messages"][-1]["ts"] == "2026-05-13T12:02:00+00:00"


def test_append_message_preserves_other_state_keys() -> None:
    state = {
        "v2_draft": {"role_title": "Tax Associate"},
        "source_packet": {"raw_text": "JD..."},
        "conversation_meta": {"turn_count": 3},
    }

    result = append_message(state, _msg("recruiter", "hi"))

    assert result["v2_draft"] == {"role_title": "Tax Associate"}
    assert result["source_packet"] == {"raw_text": "JD..."}
    assert result["conversation_meta"] == {"turn_count": 3}


# ---------------------------------------------------------------------------
# detect_dropped_turn
# ---------------------------------------------------------------------------


def test_detect_dropped_turn_false_on_empty_state() -> None:
    assert detect_dropped_turn({}) is False
    assert detect_dropped_turn({"messages": []}) is False


def test_detect_dropped_turn_true_when_recruiter_is_tail() -> None:
    state = {
        "messages": [
            _msg("cloris", "opener"),
            _msg("recruiter", "interrupted send"),
        ]
    }
    assert detect_dropped_turn(state) is True


def test_detect_dropped_turn_false_when_cloris_is_tail() -> None:
    state = {
        "messages": [
            _msg("recruiter", "first"),
            _msg("cloris", "reply landed"),
        ]
    }
    assert detect_dropped_turn(state) is False


# ---------------------------------------------------------------------------
# bump_conversation_meta
# ---------------------------------------------------------------------------


def test_bump_conversation_meta_initializes_on_first_call() -> None:
    state = {}

    result = bump_conversation_meta(state)

    assert result["conversation_meta"]["turn_count"] == 1
    assert result["conversation_meta"]["cost_usd_running_total"] == 0.0
    assert result["conversation_meta"]["last_extraction_at_turn"] == 0
    assert result["conversation_meta"]["manually_edited_keys"] == []


def test_bump_conversation_meta_accumulates() -> None:
    state = bump_conversation_meta({}, turn_delta=1, cost_delta_usd=0.05)
    state = bump_conversation_meta(state, turn_delta=1, cost_delta_usd=0.07)

    assert state["conversation_meta"]["turn_count"] == 2
    assert state["conversation_meta"]["cost_usd_running_total"] == pytest.approx(0.12)


def test_bump_conversation_meta_preserves_manual_edits() -> None:
    state = {
        "conversation_meta": {
            "turn_count": 5,
            "cost_usd_running_total": 0.30,
            "manually_edited_keys": ["role_title"],
            "last_extraction_at_turn": 4,
        }
    }

    result = bump_conversation_meta(state, turn_delta=1, cost_delta_usd=0.10)

    assert result["conversation_meta"]["turn_count"] == 6
    assert result["conversation_meta"]["cost_usd_running_total"] == pytest.approx(0.40)
    assert result["conversation_meta"]["manually_edited_keys"] == ["role_title"]
    assert result["conversation_meta"]["last_extraction_at_turn"] == 4


# ---------------------------------------------------------------------------
# merge_extracted (the load-bearing one — defense-in-depth backstop)
# ---------------------------------------------------------------------------


def test_merge_extracted_additive_on_empty_draft() -> None:
    result = merge_extracted({}, {"role_title": "Tax Associate"})
    assert result == {"role_title": "Tax Associate"}


def test_merge_extracted_does_not_mutate_input() -> None:
    draft = {"role_title": "Old"}
    updates = {"role_summary": "New summary"}

    result = merge_extracted(draft, updates)

    assert draft == {"role_title": "Old"}  # unchanged
    assert result == {"role_title": "Old", "role_summary": "New summary"}


def test_merge_extracted_overwrites_unlocked_top_level_keys() -> None:
    draft = {"role_title": "Old Title"}
    updates = {"role_title": "Tax Associate"}

    result = merge_extracted(draft, updates, manually_edited_keys=set())

    assert result["role_title"] == "Tax Associate"


def test_merge_extracted_drops_locked_top_level_keys() -> None:
    """If the recruiter manually edited role_title, the extractor's update
    for that key is silently dropped — even if the extractor returns one.
    """

    draft = {"role_title": "Manual Title"}
    updates = {"role_title": "Extracted Title", "role_summary": "summary"}

    result = merge_extracted(
        draft, updates, manually_edited_keys={"role_title"}
    )

    assert result["role_title"] == "Manual Title"
    assert result["role_summary"] == "summary"  # sibling key still merges


def test_merge_extracted_drops_locked_nested_paths_only() -> None:
    """Locking ``depth_distinction.builder_definition`` does NOT lock the
    sibling sub-fields under depth_distinction. Granular path semantics.
    """

    draft = {
        "depth_distinction": {
            "builder_definition": "Manual builder def",
            "user_definition": "old user def",
        }
    }
    updates = {
        "depth_distinction": {
            "builder_definition": "extracted builder def (should be dropped)",
            "user_definition": "extracted user def (should land)",
            "edge_case_guidance": "extracted edge guidance (should land)",
        }
    }

    result = merge_extracted(
        draft,
        updates,
        manually_edited_keys={"depth_distinction.builder_definition"},
    )

    assert result["depth_distinction"]["builder_definition"] == "Manual builder def"
    assert result["depth_distinction"]["user_definition"] == "extracted user def (should land)"
    assert result["depth_distinction"]["edge_case_guidance"] == "extracted edge guidance (should land)"


def test_merge_extracted_first_write_respects_locks_too() -> None:
    """Locking a path before the slot exists still blocks the first-write path."""

    draft: dict = {}
    updates = {
        "depth_distinction": {
            "builder_definition": "extracted (locked)",
            "user_definition": "extracted (allowed)",
        }
    }

    result = merge_extracted(
        draft,
        updates,
        manually_edited_keys={"depth_distinction.builder_definition"},
    )

    assert "builder_definition" not in result["depth_distinction"]
    assert result["depth_distinction"]["user_definition"] == "extracted (allowed)"


def test_merge_extracted_replaces_lists_wholesale() -> None:
    """v0 contract: lists (capability_areas) are replaced, not by-element
    merged. The extractor is expected to return whole lists.
    """

    draft = {
        "capability_areas": [
            {"name": "Old", "description": "old desc"}
        ]
    }
    updates = {
        "capability_areas": [
            {"name": "Tax Compliance", "description": "new desc"},
            {"name": "Reporting", "description": "another"},
        ]
    }

    result = merge_extracted(draft, updates)

    assert result["capability_areas"] == [
        {"name": "Tax Compliance", "description": "new desc"},
        {"name": "Reporting", "description": "another"},
    ]


def test_merge_extracted_no_op_on_empty_updates() -> None:
    draft = {"role_title": "Tax Associate"}

    result = merge_extracted(draft, {})

    assert result == {"role_title": "Tax Associate"}
    assert result is not draft  # still a new dict


# ---------------------------------------------------------------------------
# Round-trip through patch_intake_session
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "intake" / "intake_sessions.sqlite3")


def test_state_roundtrips_through_patch_intake_session(store: RuntimeStateStore) -> None:
    """Conversational state survives JSON serialization through the SQLite
    column. If this test fails, the C5 SSE endpoint can't trust that what
    it persists is what subsequent loads return.
    """

    session = intake_module.create_intake_session(store)
    state = {}
    state = append_message(state, _msg("cloris", "Hi — I'm Cloris. Tell me about the role."))
    state = append_message(state, _msg("recruiter", "Senior tax associate at Northwind."))
    state = bump_conversation_meta(state, turn_delta=1, cost_delta_usd=0.012)
    state["v2_draft"] = {"role_title": "Senior Tax Associate"}
    state["conversation_meta"]["manually_edited_keys"] = ["role_summary"]

    intake_module.patch_intake_session(
        store,
        session_id=session["id"],
        current_step="conversation",
        state_json=state,
    )

    refreshed = intake_module.get_intake_session(store, session_id=session["id"])

    assert refreshed is not None
    assert refreshed["current_step"] == "conversation"
    assert refreshed["state_json"]["messages"] == state["messages"]
    assert refreshed["state_json"]["v2_draft"] == {"role_title": "Senior Tax Associate"}
    assert refreshed["state_json"]["conversation_meta"]["turn_count"] == 1
    assert refreshed["state_json"]["conversation_meta"]["cost_usd_running_total"] == pytest.approx(0.012)
    assert refreshed["state_json"]["conversation_meta"]["manually_edited_keys"] == ["role_summary"]


def test_dropped_turn_detection_after_roundtrip(store: RuntimeStateStore) -> None:
    """Crash-mid-stream recovery contract: a session whose tail is a
    recruiter message must read back as ``detect_dropped_turn=True`` after
    a round-trip through SQLite. C5's session-load branch depends on this.
    """

    session = intake_module.create_intake_session(store)
    state = {}
    state = append_message(state, _msg("cloris", "opener"))
    state = append_message(state, _msg("recruiter", "Senior tax associate at Northwind."))
    # Stream "died" before persisting Cloris's reply.

    intake_module.patch_intake_session(
        store, session_id=session["id"], state_json=state
    )

    refreshed = intake_module.get_intake_session(store, session_id=session["id"])

    assert refreshed is not None
    assert detect_dropped_turn(refreshed["state_json"]) is True
