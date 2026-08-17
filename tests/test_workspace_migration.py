"""Slice A.3 (Multi-Agent Production Plan) — workspace tables migration.

Pins the additive schema migration that adds three workspace tables
to ``shared/runtime_state/store.py:_migrate``:

- ``workspace_entries`` — primary workspace row, one per (brief, candidate)
- ``workspace_review_events`` — append-only event log for review actions
- ``workspace_outreach_artifacts`` — optional outreach copy per entry

Per ``docs/cloris-candidate-workspace-spec.md`` §3. The migration is
additive + idempotent; schema version 10 signaled the change. The
current runtime-state schema is now 11 after the accepted-brief corpus
tables landed in the shared store. A.5 ships the writer
(``shared/save_destination/candidate_workspace.py``) that populates
these tables; the recruiter-mutation endpoints (PATCH / POST) land
alongside the frontend workspace card work in Phase C / G.

Tests pin:

- The three tables and their indexes exist after migration.
- ``CURRENT_SCHEMA_VERSION == "11"``.
- Re-running ``_migrate`` is idempotent (no duplicate tables / indexes).
- ``workspace_entries`` schema columns match the spec at
  ``docs/cloris-candidate-workspace-spec.md:50-72``.
- The ``UNIQUE(brief_id, candidate_id)`` constraint fires on duplicate
  insert (the workspace is brief-scoped; a re-save for the same
  candidate updates rather than duplicates).
- ``ON DELETE CASCADE`` on ``workspace_entries.candidate_id`` works
  (deleting a candidate cleans up its workspace entry; deleting a
  workspace entry cleans up its review events + outreach artifacts).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from shared.runtime_state.store import (
    CURRENT_SCHEMA_VERSION,
    RuntimeStateStore,
)


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_current_schema_version_bumped_to_12() -> None:
    """The current runtime-state store schema version pin.

    Bumped 11 -> 12 in the reopen Stage-2 recruiter primitive (501478c); this
    guard had drifted (still pinned '11') and failed since. If you bump again,
    update this assertion and the schema migration notes.
    """

    assert CURRENT_SCHEMA_VERSION == "12", (
        "Runtime-state schema is at '12' (reopen Stage 2). If you bump again, "
        "update this assertion and the schema migration notes."
    )


def test_workspace_tables_present_after_migration(tmp_path: Path) -> None:
    """All three workspace tables exist after store initialization."""

    store = _make_store(tmp_path)
    with store.connect() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'workspace_%'"
            ).fetchall()
        }

    assert tables == {
        "workspace_entries",
        "workspace_review_events",
        "workspace_outreach_artifacts",
    }


def test_workspace_indexes_present_after_migration(tmp_path: Path) -> None:
    """Each workspace table carries the index its query path needs."""

    store = _make_store(tmp_path)
    with store.connect() as conn:
        indexes = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name LIKE 'idx_workspace%'"
            ).fetchall()
        }

    assert indexes == {
        "idx_workspace_brief",
        "idx_workspace_review_events_entry",
        "idx_workspace_outreach_artifacts_entry",
    }


def test_workspace_entries_schema_matches_spec(tmp_path: Path) -> None:
    """The columns listed in the spec at docs:50-72 are all present."""

    store = _make_store(tmp_path)
    with store.connect() as conn:
        cols = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(workspace_entries)"
            ).fetchall()
        }

    expected = {
        "id",
        "brief_id",
        "candidate_id",
        "person_id",
        "surface_type",
        "save_decision",
        "save_confidence",
        "save_rationale",
        "save_path",
        "review_status",
        "review_marked_at",
        "outreach_status",
        "outreach_marked_at",
        "recruiter_notes",
        "contextualization_payload_json",
        "created_at",
        "updated_at",
    }
    assert cols == expected, (
        f"workspace_entries schema diverged from spec; "
        f"missing={expected - cols}, extra={cols - expected}"
    )


def test_migrate_is_idempotent_for_workspace_tables(tmp_path: Path) -> None:
    """Re-running ``_migrate`` is a no-op against an existing schema.

    Mirrors the intake-session idempotency test pattern at
    ``tests/test_intake_sessions.py:test_migrate_is_idempotent_for_intake_sessions``.
    """

    store = _make_store(tmp_path)
    with store.connect() as conn:
        store._migrate(conn)
        store._migrate(conn)

        for table in (
            "workspace_entries",
            "workspace_review_events",
            "workspace_outreach_artifacts",
        ):
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchall()
            assert len(tables) == 1, f"exactly one {table} table"

        for index in (
            "idx_workspace_brief",
            "idx_workspace_review_events_entry",
            "idx_workspace_outreach_artifacts_entry",
        ):
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                (index,),
            ).fetchall()
            assert len(indexes) == 1, f"exactly one {index} index"


def test_workspace_entries_unique_brief_candidate_constraint(tmp_path: Path) -> None:
    """A re-save for the same (brief, candidate) violates the UNIQUE constraint.

    The workspace is brief-scoped; a re-save updates the existing entry
    rather than inserting a duplicate. The writer in A.5 will use
    ``INSERT ... ON CONFLICT(brief_id, candidate_id) DO UPDATE`` to
    handle this; this test pins the constraint that makes that pattern
    safe.
    """

    import sqlite3

    store = _make_store(tmp_path)
    now = _utc_now()
    with store.connect() as conn:
        # Need a candidate row first to satisfy the FK.
        cur = conn.execute(
            """
            INSERT INTO candidates(
                source, brief_id, identity_key, display_name,
                current_lifecycle_state, terminal_decision,
                first_seen_at, last_seen_at
            )
            VALUES ('linkedin', 'brief-x', 'cand-x', 'Test',
                    'evaluated', 'SAVE', ?, ?)
            """,
            (now, now),
        )
        cand_id = int(cur.lastrowid)

        conn.execute(
            """
            INSERT INTO workspace_entries(
                brief_id, candidate_id, save_decision,
                created_at, updated_at
            )
            VALUES ('brief-x', ?, 'SAVE', ?, ?)
            """,
            (cand_id, now, now),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO workspace_entries(
                    brief_id, candidate_id, save_decision,
                    created_at, updated_at
                )
                VALUES ('brief-x', ?, 'SAVE', ?, ?)
                """,
                (cand_id, now, now),
            )


def test_workspace_review_events_cascade_on_entry_delete(tmp_path: Path) -> None:
    """Deleting a workspace_entries row cascades to its review events.

    Pins the ``ON DELETE CASCADE`` on ``workspace_review_events.workspace_entry_id``
    — the append-only event log should not outlive its parent entry.
    """

    store = _make_store(tmp_path)
    now = _utc_now()
    with store.connect() as conn:
        # CASCADE requires foreign_keys pragma ON; tests run with default
        # off, so enable explicitly to verify the cascade behavior.
        conn.execute("PRAGMA foreign_keys = ON")

        cur = conn.execute(
            """
            INSERT INTO candidates(
                source, brief_id, identity_key, display_name,
                current_lifecycle_state, terminal_decision,
                first_seen_at, last_seen_at
            )
            VALUES ('linkedin', 'brief-y', 'cand-y', 'Test',
                    'evaluated', 'SAVE', ?, ?)
            """,
            (now, now),
        )
        cand_id = int(cur.lastrowid)

        cur2 = conn.execute(
            """
            INSERT INTO workspace_entries(
                brief_id, candidate_id, save_decision,
                created_at, updated_at
            )
            VALUES ('brief-y', ?, 'SAVE', ?, ?)
            """,
            (cand_id, now, now),
        )
        entry_id = int(cur2.lastrowid)

        conn.execute(
            """
            INSERT INTO workspace_review_events(
                workspace_entry_id, event_type, created_at
            )
            VALUES (?, 'review_marked', ?)
            """,
            (entry_id, now),
        )

        events_before = conn.execute(
            "SELECT id FROM workspace_review_events WHERE workspace_entry_id=?",
            (entry_id,),
        ).fetchall()
        assert len(events_before) == 1

        conn.execute("DELETE FROM workspace_entries WHERE id=?", (entry_id,))

        events_after = conn.execute(
            "SELECT id FROM workspace_review_events WHERE workspace_entry_id=?",
            (entry_id,),
        ).fetchall()
        assert len(events_after) == 0, (
            "review events should cascade-delete with their parent entry"
        )
