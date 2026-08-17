"""Intake session CRUD for the onboarding flow.

Per ``/Users/jordan.rivera/.claude/plans/ancient-plotting-lemon.md`` A1: the
``intake_sessions`` table holds authoring state, distinct from
run-lifecycle state. This module owns create/read/update/delete plus the
active-session listing logic.

Writes go through :class:`shared.runtime_state.store.RuntimeStateStore`'s
regular ``connect`` contextmanager (read/write SQLite). The read-only
``mode=ro`` path used by :mod:`cloris.control_plane` for status aggregation
is deliberately not used here because intake sessions are mutated by the UI.

The schema lives here in :func:`install_schema`; RuntimeStateStore calls that
installer while initializing the shared SQLite database.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from shared.runtime_state.store import RuntimeStateStore


def install_schema(conn: sqlite3.Connection) -> None:
    """Install intake-session authoring tables into the shared SQLite DB."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS intake_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brief_id_draft TEXT,
            role_title TEXT,
            current_step TEXT NOT NULL,
            state_json TEXT NOT NULL DEFAULT '{}',
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            archived_at TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_intake_sessions_active
        ON intake_sessions(archived_at, completed_at);
        """
    )


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp.

    Matches the format used by :func:`shared.runtime_state.store._utc_now`
    so timestamps across the canonical store and the intake-session table
    stay comparable.
    """

    return datetime.now(timezone.utc).isoformat()


def _row_to_session(row: sqlite3.Row) -> dict:
    """Parse a sqlite3.Row into an :class:`cloris.models.IntakeSession`-shaped dict.

    ``state_json`` is parsed from the TEXT column to a dict; an empty or
    NULL column collapses to ``{}`` so the wire model never has to handle
    the missing case. The dict is structured to match the Pydantic model
    field order so :class:`IntakeSession.model_validate(...)` consumes it
    directly without additional shaping.
    """

    raw_state = row["state_json"]
    if raw_state:
        try:
            parsed_state = json.loads(raw_state)
        except (json.JSONDecodeError, TypeError):
            parsed_state = {}
    else:
        parsed_state = {}
    if not isinstance(parsed_state, dict):
        # The schema stores authoring state as a JSON object. A non-dict
        # value (e.g. legacy row written by a buggy client) collapses to
        # an empty dict so the wire model stays well-typed.
        parsed_state = {}
    return {
        "id": row["id"],
        "brief_id_draft": row["brief_id_draft"],
        "role_title": row["role_title"],
        "current_step": row["current_step"],
        "state_json": parsed_state,
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
        "archived_at": row["archived_at"],
    }


def create_intake_session(
    store: RuntimeStateStore,
    *,
    role_title: str | None = None,
) -> dict:
    """Insert a new intake session and return its hydrated row.

    Initial ``current_step`` is ``"welcome"``; ``state_json`` defaults to
    ``{}``; ``started_at`` and ``updated_at`` are set to the same UTC
    timestamp at creation. ``brief_id_draft`` and the lifecycle timestamps
    (``completed_at`` / ``archived_at``) start as NULL.
    """

    now = _utc_now()
    with store.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO intake_sessions(
                role_title, current_step, state_json,
                started_at, updated_at
            )
            VALUES (?, 'welcome', '{}', ?, ?)
            """,
            (role_title, now, now),
        )
        session_id = int(cursor.lastrowid)
        row = conn.execute(
            "SELECT * FROM intake_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row)


def list_intake_sessions(store: RuntimeStateStore) -> list[dict]:
    """List active (non-archived) sessions, newest first by ``updated_at``.

    Archived sessions (``archived_at`` IS NOT NULL) are excluded so the
    onboarding shell never has to filter at the wire boundary. Ordering by
    ``updated_at`` (rather than ``started_at``) matches the user mental
    model of "most recently touched first."
    """

    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM intake_sessions
            WHERE archived_at IS NULL
            ORDER BY updated_at DESC
            """
        ).fetchall()
        return [_row_to_session(row) for row in rows]


def get_intake_session(
    store: RuntimeStateStore, *, session_id: int
) -> dict | None:
    """Return the session row by id, or ``None`` if not found.

    Returns the row regardless of archived state — the ``GET`` endpoint is
    used for direct deep-links and resume flows where a recruiter may want
    to inspect or unarchive an old session.
    """

    with store.connect() as conn:
        row = conn.execute(
            "SELECT * FROM intake_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row) if row else None


def patch_intake_session(
    store: RuntimeStateStore,
    *,
    session_id: int,
    current_step: str | None = None,
    state_json: dict | None = None,
    role_title: str | None = None,
) -> dict | None:
    """Partial update of an intake session.

    Updates whichever fields are provided (non-None) and always bumps
    ``updated_at``. A patch with no fields set still bumps ``updated_at``
    so a UI heartbeat can keep the session at the top of the list.

    Returns the hydrated session row after update, or ``None`` if the
    session id doesn't exist.
    """

    fields: list[str] = []
    params: list = []
    if current_step is not None:
        fields.append("current_step = ?")
        params.append(current_step)
    if state_json is not None:
        fields.append("state_json = ?")
        params.append(json.dumps(state_json))
    if role_title is not None:
        fields.append("role_title = ?")
        params.append(role_title)
    # Always bump updated_at, even on a no-op patch — see docstring.
    fields.append("updated_at = ?")
    params.append(_utc_now())
    params.append(session_id)

    with store.connect() as conn:
        existing = conn.execute(
            "SELECT id FROM intake_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not existing:
            return None
        conn.execute(
            f"UPDATE intake_sessions SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        row = conn.execute(
            "SELECT * FROM intake_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row)


def complete_intake_session(
    store: RuntimeStateStore,
    *,
    session_id: int,
    brief_id: str,
) -> dict | None:
    """Mark an intake session as completed and stamp its draft brief id.

    Phase D Slice D3 helper. Called by the
    ``POST /api/intake/sessions/{id}/complete`` endpoint after the
    brief file has been written successfully. Sets ``current_step``
    to ``"completed"``, ``completed_at`` to now, ``brief_id_draft`` to
    the freshly-computed brief_id, and bumps ``updated_at``. Returns
    the hydrated row, or ``None`` if the session id doesn't exist.

    Idempotent: if the session is already completed (``completed_at``
    is non-NULL), the timestamps are NOT overwritten — the original
    completion stamp is the audit-meaningful one. Only ``brief_id_draft``
    and ``updated_at`` are updated, so re-completing with a new brief
    file (rare; mostly a defensive path) repoints the session without
    rewriting history.
    """

    now = _utc_now()
    with store.connect() as conn:
        existing = conn.execute(
            "SELECT id, completed_at FROM intake_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not existing:
            return None
        if existing["completed_at"] is None:
            conn.execute(
                """
                UPDATE intake_sessions
                SET current_step = 'completed',
                    completed_at = ?,
                    brief_id_draft = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, brief_id, now, session_id),
            )
        else:
            conn.execute(
                """
                UPDATE intake_sessions
                SET brief_id_draft = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (brief_id, now, session_id),
            )
        row = conn.execute(
            "SELECT * FROM intake_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row)


def archive_intake_session(
    store: RuntimeStateStore, *, session_id: int
) -> dict | None:
    """Archive an intake session (soft-delete): stamp ``archived_at``.

    Idempotent: if the session is already archived, ``archived_at`` is NOT
    overwritten — the original archive stamp is the audit-meaningful one
    (mirrors :func:`complete_intake_session`'s idempotency). ``updated_at``
    is always bumped. Returns the hydrated row, or ``None`` if the session
    id doesn't exist.

    Once archived, :func:`list_intake_sessions`'s ``WHERE archived_at IS
    NULL`` filter excludes the row from the active listing;
    :func:`get_intake_session` (deep-link / resume) still returns it.
    """

    now = _utc_now()
    with store.connect() as conn:
        existing = conn.execute(
            "SELECT id, archived_at FROM intake_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not existing:
            return None
        if existing["archived_at"] is None:
            conn.execute(
                """
                UPDATE intake_sessions
                SET archived_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, session_id),
            )
        else:
            conn.execute(
                "UPDATE intake_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        row = conn.execute(
            "SELECT * FROM intake_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row)


def delete_intake_session(
    store: RuntimeStateStore, *, session_id: int
) -> bool:
    """Hard-delete an intake session.

    Returns ``True`` when a row was deleted, ``False`` otherwise. The route
    layer maps False to HTTP 404. There is no soft-delete here — Slice 1B
    keeps the surface minimal; archive semantics will land in a later slice
    if the product needs them.
    """

    with store.connect() as conn:
        result = conn.execute(
            "DELETE FROM intake_sessions WHERE id = ?", (session_id,)
        )
        return result.rowcount > 0
