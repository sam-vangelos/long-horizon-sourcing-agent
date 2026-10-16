"""SQLite store for cross-source orchestration runtime state.

Chief-of-staff runs are brief-grain and cross-source. A single run combines
the per-source state directories into one synthesis row containing the
dispatch plan, specialist invocation order, handoff payloads, and synthesis
output. The per-source ``runs`` table is the wrong grain, and writing a
cross-source row into one source database would violate the read-only
boundary of those stores.

The orchestration database is
``output/state/orchestration/runtime_state.sqlite3``. It sits beside the
per-source state directories and is excluded from source enumeration.

Schema:
    chief_of_staff_runs                 -- brief-grain orchestration row
    cross_brief_playbook_observations   -- append-only calibration log
    conversation_threads                -- one row per brief_id
    conversation_turns                  -- persisted chat and narration turns

``cross_brief_playbook_observations`` is brief-agnostic when read, at
per-principal × per-market × per-role-shape grain; ``brief_id`` remains
provenance rather than the grouping key.

Initialization mirrors ``shared.runtime_state.identity_store``: construction
calls ``initialize()``, ``connect()`` commits on exit and yields
``sqlite3.Row`` values, and foreign keys plus WAL are enabled. Read-only
consumers use ``shared.runtime_state.read_models`` helpers so they do not
perform DDL or rewrite metadata.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


CURRENT_ORCHESTRATION_SCHEMA_VERSION = "2"


class OrchestrationStateStore:
    """SQLite-backed orchestration state for chief-of-staff runs.

    Mirrors the ``IdentityStore.connect()`` contract (context manager
    that commits on exit, row factory yields ``sqlite3.Row``, foreign
    keys + WAL pragmas) so callers don't have to relearn a third
    connection style. Instantiation is idempotent: ``initialize()``
    uses ``CREATE TABLE IF NOT EXISTS`` everywhere and rewriting the
    meta schema-version row is a no-op against an already-initialized
    DB.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chief_of_staff_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    brief_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'running',
                    dispatch_plan_json TEXT NOT NULL DEFAULT '{}',
                    invocation_order_json TEXT NOT NULL DEFAULT '[]',
                    handoff_payloads_json TEXT NOT NULL DEFAULT '{}',
                    synthesis_output_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    ended_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chief_of_staff_runs_brief
                    ON chief_of_staff_runs(brief_id, started_at DESC);

                CREATE INDEX IF NOT EXISTS idx_chief_of_staff_runs_principal
                    ON chief_of_staff_runs(principal_id, started_at DESC)
                    WHERE principal_id != '';

                CREATE TABLE IF NOT EXISTS cross_brief_playbook_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    principal_id TEXT NOT NULL,
                    market_key TEXT NOT NULL DEFAULT '',
                    role_shape TEXT NOT NULL DEFAULT '',
                    brief_id TEXT NOT NULL,
                    observation_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_cross_brief_obs_principal
                    ON cross_brief_playbook_observations(
                        principal_id, market_key, role_shape, created_at DESC
                    );

                CREATE INDEX IF NOT EXISTS idx_cross_brief_obs_brief
                    ON cross_brief_playbook_observations(brief_id);

                CREATE TABLE IF NOT EXISTS conversation_threads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    brief_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    ambient_muted INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_threads_brief
                    ON conversation_threads(brief_id);

                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'ok',
                    trace_ref_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (thread_id) REFERENCES conversation_threads(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_turns_thread
                    ON conversation_turns(thread_id, created_at ASC);

                -- Reopen Y.5.6 (F1): persisted per-source launch hard-pause.
                -- The Y.5.2 kill-switch is process-env only
                -- (``CLORIS_PAUSE_LAUNCHES_<SOURCE>``), so an operator can only
                -- arm it where the API server's env lives — not out-of-process.
                -- This table is the durable, server-observable arm: a row with
                -- ``paused = 1`` for a source is read by the in-process spawn
                -- gate on the NEXT spawn (additive-OR with the env arm). An
                -- ABSENT row means "not paused" (no spurious block). It lives in
                -- the orchestration DB (cross-source, brief-agnostic) for the
                -- same structural reason the rest of this store does: a pause
                -- is a global operator decision, not a per-state-dir one.
                CREATE TABLE IF NOT EXISTS source_pause (
                    source TEXT PRIMARY KEY,
                    paused INTEGER NOT NULL DEFAULT 0,
                    armed_at TEXT,
                    armed_by TEXT,
                    reason TEXT
                );
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (
                    "orchestration_schema_version",
                    CURRENT_ORCHESTRATION_SCHEMA_VERSION,
                ),
            )

    def insert_chief_of_staff_run(
        self,
        *,
        brief_id: str,
        principal_id: str = "",
        status: str = "running",
        dispatch_plan: dict | None = None,
        invocation_order: list[str] | None = None,
        handoff_payloads: dict | None = None,
        synthesis_output: dict | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
    ) -> int:
        """Insert one ``chief_of_staff_runs`` row and return its id.

        Each dispatch invocation inserts a new row. Append-only writes let
        read models select the latest attempt while preserving provenance for
        earlier attempts.
        """

        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chief_of_staff_runs(
                    brief_id, principal_id, status,
                    dispatch_plan_json, invocation_order_json,
                    handoff_payloads_json, synthesis_output_json,
                    started_at, ended_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    brief_id,
                    principal_id,
                    status,
                    _json_dumps(dispatch_plan or {}),
                    _json_dumps(invocation_order or []),
                    _json_dumps(handoff_payloads or {}),
                    _json_dumps(synthesis_output or {}),
                    started_at or _utc_now(),
                    ended_at,
                ),
            )
            return int(cursor.lastrowid)

    def merge_handoff_payload(
        self,
        *,
        brief_id: str,
        source: str,
        payload: dict,
    ) -> bool:
        """Merge one source's handoff payload into the latest CoS run.

        Each module's run-end path persists a structured per-source summary
        (top saves, signal summary, and confidence) in the latest
        ``chief_of_staff_runs.handoff_payloads_json`` keyed by source. The
        synthesis caller reads these payloads as cross-source prompt context.

        Behavior:
        - Read the latest run for ``brief_id`` using the same
          ``ORDER BY started_at DESC, id DESC`` ordering as the read model.
        - Parse the existing JSON, defaulting to ``{}`` when missing or malformed.
        - Set ``handoff_payloads[source] = payload`` with last-write-wins
          semantics, then update the row.

        Return ``True`` when a row is updated and ``False`` when no run exists
        for the brief. The return value is non-fatal; callers log and continue.

        SQLite WAL serializes concurrent read-modify-write transactions. The
        serialization order determines which concurrent payload is last.
        """

        source = (source or "").strip().lower()
        if not source:
            return False

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, handoff_payloads_json
                FROM chief_of_staff_runs
                WHERE brief_id = ?
                ORDER BY started_at DESC, id DESC
                LIMIT 1
                """,
                (brief_id,),
            ).fetchone()
            if row is None:
                return False

            existing_json = row["handoff_payloads_json"] or "{}"
            try:
                existing = json.loads(existing_json)
                if not isinstance(existing, dict):
                    existing = {}
            except (TypeError, ValueError, json.JSONDecodeError):
                existing = {}

            existing[source] = dict(payload or {})

            conn.execute(
                """
                UPDATE chief_of_staff_runs
                SET handoff_payloads_json = ?
                WHERE id = ?
                """,
                (_json_dumps(existing), int(row["id"])),
            )
        return True

    # ------------------------------------------------------------------
    # source_pause (Reopen Y.5.6 / F1 — persisted launch hard-pause)
    # ------------------------------------------------------------------

    def set_source_pause(
        self,
        source: str,
        *,
        paused: bool,
        armed_by: str = "",
        reason: str = "",
        now: str | None = None,
    ) -> None:
        """Arm or disarm the persisted launch pause for one source.

        Upsert on the ``source`` PK: the first call INSERTs, every later call
        UPDATEs the same row in place. ``paused=True`` arms the durable pause
        (the spawn gate refuses launches for this source on its next spawn);
        ``paused=False`` disarms it (the row stays, recording who/when, but the
        gate no longer reads it as paused). ``armed_by`` / ``reason`` are
        operator provenance; ``now`` defaults to UTC-now.

        This is the OUT-OF-PROCESS arm: an operator (CLI / admin) writes here,
        and the in-process spawn gate sees the write on its next read — no API
        restart, no env mutation. Source is normalized (stripped, lowercased)
        so the persisted key matches the gate's ``source.lower()`` read.
        """

        key = (source or "").strip().lower()
        if not key:
            raise ValueError("source must be non-empty")
        stamp = now or _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO source_pause(source, paused, armed_at, armed_by, reason)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    paused = excluded.paused,
                    armed_at = excluded.armed_at,
                    armed_by = excluded.armed_by,
                    reason = excluded.reason
                """,
                (key, 1 if paused else 0, stamp, armed_by, reason),
            )

    def is_source_paused(self, source: str) -> bool:
        """Return whether the durable launch pause is armed for one source.

        Pure read. An ABSENT row means "not paused" (the default state — no row
        is ever materialized until an operator arms one), so a fresh store never
        spuriously blocks a launch. A present row contributes ``paused != 0``.
        """

        key = (source or "").strip().lower()
        if not key:
            return False
        with self.connect() as conn:
            row = conn.execute(
                "SELECT paused FROM source_pause WHERE source = ?",
                (key,),
            ).fetchone()
        if row is None:
            return False
        return int(row["paused"] or 0) != 0

    def get_or_create_conversation_thread(self, *, brief_id: str) -> int:
        """Return thread id for brief, creating a row if missing."""

        bid = (brief_id or "").strip()
        if not bid:
            raise ValueError("brief_id required")
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM conversation_threads WHERE brief_id = ?",
                (bid,),
            ).fetchone()
            if row is not None:
                return int(row["id"])
            now = _utc_now()
            conn.execute(
                """
                INSERT INTO conversation_threads (brief_id, created_at, ambient_muted)
                VALUES (?, ?, 0)
                """,
                (bid, now),
            )
            row2 = conn.execute(
                "SELECT id FROM conversation_threads WHERE brief_id = ?",
                (bid,),
            ).fetchone()
            assert row2 is not None
            return int(row2["id"])

    def set_conversation_ambient_muted(
        self, *, brief_id: str, muted: bool
    ) -> None:
        self.get_or_create_conversation_thread(brief_id=brief_id)
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE conversation_threads
                SET ambient_muted = ?
                WHERE brief_id = ?
                """,
                (1 if muted else 0, (brief_id or "").strip()),
            )

    def get_conversation_ambient_muted(self, *, brief_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT ambient_muted FROM conversation_threads
                WHERE brief_id = ?
                """,
                ((brief_id or "").strip(),),
            ).fetchone()
            if row is None:
                return False
            return int(row["ambient_muted"] or 0) != 0

    def insert_conversation_turn(
        self,
        *,
        thread_id: int,
        role: str,
        content: str,
        kind: str = "ok",
        trace_ref: dict | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO conversation_turns(
                    thread_id, role, content, kind, trace_ref_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    role,
                    content,
                    kind,
                    _json_dumps(trace_ref) if trace_ref else None,
                    _utc_now(),
                ),
            )
            return int(cur.lastrowid)

    def list_conversation_turns(
        self,
        *,
        thread_id: int,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Most recent ``limit`` turns, oldest-first for prompt context."""

        lim = max(1, min(int(limit), 200))
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, thread_id, role, content, kind, trace_ref_json, created_at
                FROM conversation_turns
                WHERE thread_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (thread_id, lim),
            ).fetchall()
        out = [
            {
                "id": int(r["id"]),
                "role": r["role"],
                "content": r["content"],
                "kind": r["kind"],
                "created_at": r["created_at"],
            }
            for r in reversed(rows)
        ]
        return out

    def delete_conversation_thread_for_brief(self, *, brief_id: str) -> int:
        """Hard-delete companion data for brief (discard path). Rows cascade."""

        bid = (brief_id or "").strip()
        if not bid:
            return 0
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM conversation_threads WHERE brief_id = ?",
                (bid,),
            )
            return cur.rowcount


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
