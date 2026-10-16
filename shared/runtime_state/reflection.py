"""Reflection session CRUD for the HITL market-intelligence flow.

Mirrors the :mod:`cloris.intake_sessions` module's pattern: row-level
operations on ``reflection_sessions``, with the same posture of treating
the JSON state column as opaque-to-this-layer (the engine phase
functions in :mod:`market_intelligence.reflection` own its shape).

The schema lives here in :func:`install_schema`; RuntimeStateStore calls that
installer while initializing the shared SQLite database.

State machine reminder (driven by the API layer + engine phases):

    planning → plan_approved → researching → awaiting_diff
              → committed (terminal) | discarded (terminal)

A row in ``planning`` carries the planner's first pass (editorial
briefing + intentions). Steering refinements stay in ``planning`` and
bump ``steering_iterations``. ``plan_approved`` is a transient marker
the API uses to gate ``start_research``; once research kicks off the
phase moves to ``researching``. ``awaiting_diff`` is the second HITL
gate — the proposed brief diff is in ``state_json`` and the user
either commits or discards.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from shared.runtime_state.store import RuntimeStateStore


VALID_PHASES = frozenset(
    {
        "planning",
        "plan_approved",
        "researching",
        "awaiting_diff",
        "committed",
        "discarded",
    }
)


def install_schema(conn: sqlite3.Connection) -> None:
    """Install HITL reflection-session tables into the shared SQLite DB."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reflection_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brief_id TEXT NOT NULL,
            source_run_id INTEGER,
            current_phase TEXT NOT NULL,
            state_json TEXT NOT NULL DEFAULT '{}',
            steering_iterations INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            discarded_at TEXT,
            brief_version_committed TEXT,
            research_error TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reflection_sessions_brief_active
        ON reflection_sessions(brief_id, completed_at, discarded_at);
        """
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_session(row: sqlite3.Row) -> dict:
    """Parse a sqlite3.Row into a :class:`ReflectionSession`-shaped dict.

    Same defensive parsing as :func:`cloris.intake_sessions._row_to_session`:
    a corrupted/non-dict ``state_json`` collapses to ``{}`` so the wire
    model never has to deal with the missing case. The engine phases
    sanity-check the shape before reading specific keys.
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
        parsed_state = {}
    return {
        "id": row["id"],
        "brief_id": row["brief_id"],
        "source_run_id": row["source_run_id"],
        "current_phase": row["current_phase"],
        "state_json": parsed_state,
        "steering_iterations": row["steering_iterations"],
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
        "discarded_at": row["discarded_at"],
        "brief_version_committed": row["brief_version_committed"],
        "research_error": row["research_error"],
    }


def create_reflection_session(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    source_run_id: int | None = None,
    initial_state: dict | None = None,
) -> dict:
    """Insert a new reflection session and return its hydrated row.

    The session boots in ``current_phase="planning"``. The initial
    ``state_json`` is empty by default; the caller (API layer) typically
    fires the planner phase synchronously and immediately patches the
    session with the planner result.
    """

    now = _utc_now()
    state_payload = json.dumps(initial_state or {})
    with store.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO reflection_sessions(
                brief_id, source_run_id, current_phase,
                state_json, steering_iterations,
                started_at, updated_at
            )
            VALUES (?, ?, 'planning', ?, 0, ?, ?)
            """,
            (brief_id, source_run_id, state_payload, now, now),
        )
        session_id = int(cursor.lastrowid)
        row = conn.execute(
            "SELECT * FROM reflection_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row)


def get_reflection_session(
    store: RuntimeStateStore, *, session_id: int
) -> dict | None:
    """Return one session by id; ``None`` if missing.

    Returns the row regardless of completed/discarded state — the GET
    endpoint is used both for in-flight resume and for post-mortem
    inspection of recently-finished reflections.
    """

    with store.connect() as conn:
        row = conn.execute(
            "SELECT * FROM reflection_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row) if row else None


def get_active_reflection_for_brief(
    store: RuntimeStateStore, *, brief_id: str
) -> dict | None:
    """Return the active (non-terminal) reflection for a brief, if any.

    Active means ``completed_at IS NULL AND discarded_at IS NULL``.
    Used by the workspace surface to decide whether to render the
    "review what Cloris read" pickup card. Returns the most recently
    updated row when (despite the API guard) more than one happens to
    be active.
    """

    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM reflection_sessions
            WHERE brief_id = ?
              AND completed_at IS NULL
              AND discarded_at IS NULL
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (brief_id,),
        ).fetchone()
        return _row_to_session(row) if row else None


def patch_reflection_state(
    store: RuntimeStateStore,
    *,
    session_id: int,
    state_json: dict | None = None,
    current_phase: str | None = None,
    bump_steering: bool = False,
    research_error: str | None = None,
    clear_research_error: bool = False,
) -> dict | None:
    """Partial update of a reflection session's mutable fields.

    Updates whichever fields are provided; always bumps ``updated_at``.
    ``bump_steering=True`` increments ``steering_iterations`` by 1
    (used by the steering refinement endpoint). ``research_error`` is
    a free-form string surfaced by the research subprocess on failure;
    ``clear_research_error=True`` resets it (e.g., on retry).

    Refuses to patch a session that's already in a terminal phase
    (``committed`` or ``discarded``) — terminal rows are immutable
    audit records. Returns ``None`` if the session id doesn't exist;
    raises ``ValueError`` for terminal-phase mutation attempts.
    """

    if current_phase is not None and current_phase not in VALID_PHASES:
        raise ValueError(f"unknown reflection phase: {current_phase}")

    with store.connect() as conn:
        existing = conn.execute(
            """
            SELECT id, current_phase, completed_at, discarded_at
            FROM reflection_sessions WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if not existing:
            return None
        if existing["completed_at"] is not None or existing["discarded_at"] is not None:
            raise ValueError(
                f"reflection_session {session_id} is terminal "
                f"(phase={existing['current_phase']}); cannot patch"
            )

        fields: list[str] = []
        params: list = []
        if state_json is not None:
            fields.append("state_json = ?")
            params.append(json.dumps(state_json))
        if current_phase is not None:
            fields.append("current_phase = ?")
            params.append(current_phase)
        if bump_steering:
            fields.append("steering_iterations = steering_iterations + 1")
        if research_error is not None:
            fields.append("research_error = ?")
            params.append(research_error)
        elif clear_research_error:
            fields.append("research_error = NULL")
        fields.append("updated_at = ?")
        params.append(_utc_now())
        params.append(session_id)
        conn.execute(
            f"UPDATE reflection_sessions SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        row = conn.execute(
            "SELECT * FROM reflection_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row)


def commit_reflection(
    store: RuntimeStateStore,
    *,
    session_id: int,
    brief_version_path: str,
    final_state: dict | None = None,
) -> dict | None:
    """Mark a reflection session committed; record the brief version path.

    Terminal transition. Sets ``current_phase="committed"``,
    ``completed_at=now``, ``brief_version_committed=<path>``. If
    ``final_state`` is provided, replaces ``state_json`` (typically
    used to record which hunks were accepted vs. skipped vs. edited).

    Idempotent against repeated calls: if already committed, the
    timestamps and version path are NOT overwritten — the original
    commit record is the audit-meaningful one. Returns ``None`` if
    the session id doesn't exist.
    """

    now = _utc_now()
    with store.connect() as conn:
        existing = conn.execute(
            "SELECT id, completed_at FROM reflection_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not existing:
            return None
        if existing["completed_at"] is None:
            fields = [
                "current_phase = 'committed'",
                "completed_at = ?",
                "brief_version_committed = ?",
                "updated_at = ?",
            ]
            params: list = [now, brief_version_path, now]
            if final_state is not None:
                fields.insert(0, "state_json = ?")
                params.insert(0, json.dumps(final_state))
            params.append(session_id)
            conn.execute(
                f"UPDATE reflection_sessions SET {', '.join(fields)} WHERE id = ?",
                params,
            )
        row = conn.execute(
            "SELECT * FROM reflection_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row)


def discard_reflection(
    store: RuntimeStateStore, *, session_id: int
) -> dict | None:
    """Mark a reflection session discarded; brief untouched.

    Terminal transition. Sets ``current_phase="discarded"``,
    ``discarded_at=now``. Idempotent — repeated calls don't overwrite
    the original discard timestamp. Returns ``None`` if the session
    id doesn't exist.

    Note: if a research subprocess is in flight when discard happens,
    it continues to completion (cheaper than canceling the Perplexity
    call mid-flight). The result is dropped on the floor — the
    research subprocess writes back to the session, sees the discarded
    state, and exits without further work.
    """

    now = _utc_now()
    with store.connect() as conn:
        existing = conn.execute(
            "SELECT id, discarded_at FROM reflection_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not existing:
            return None
        if existing["discarded_at"] is None:
            conn.execute(
                """
                UPDATE reflection_sessions
                SET current_phase = 'discarded',
                    discarded_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, session_id),
            )
        row = conn.execute(
            "SELECT * FROM reflection_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row)
