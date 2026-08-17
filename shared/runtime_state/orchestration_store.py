"""Orchestration runtime state SQLite store (multi-agent execution Slice 2.3).

Why this is a SEPARATE database from per-state-dir
``runtime_state.sqlite3``:

Chief-of-staff runs are *brief-grain, cross-source*. A single multi-
module brief run merges work across LinkedIn / GitHub / Designer /
Researcher / ExecSearch state-dirs into one synthesis row that names a
dispatch plan, a per-specialist invocation order, the handoff payloads
each specialist produced, and the final synthesis. None of the obvious
homes work:

- A column on the per-source ``runs`` table is wrong-grain — ``runs``
  rows are *per-source per-brief*; chief-of-staff rows are *per-brief
  cross-source*. One CoS run corresponds to N per-source ``runs`` rows.
- A new table inside any per-source SQLite is writer-on-the-wrong-side.
  The §1 read-only invariant the per-source SQLites preserve for
  ``shared.runtime_state.read_models`` callers would be violated by a
  cross-source writer reaching into one source's DB.

So orchestration ships its own SQLite at
``output/state/orchestration/runtime_state.sqlite3``, parallel to the
per-source state-dirs. ``cloris.launchers.known_sources()`` does not
include "orchestration" so ``cloris.control_plane.enumerate_state_dirs``
will not iterate this directory as a per-source state-dir.

Schema:
    chief_of_staff_runs                 -- brief-grain CoS run row
    cross_brief_playbook_observations   -- append-only calibration log
    conversation_threads                -- one row per brief_id (companion chat)
    conversation_turns                  -- persisted chat + narration turns

The ``cross_brief_playbook_observations`` table is brief-AGNOSTIC at
read time (per-principal × per-market × per-role-shape grain) so
calibration like "for principal X, frontier-AI briefs do better when
LinkedIn runs first" can read across briefs without joining N per-source
DBs. ``brief_id`` is recorded for provenance, not for grouping.

Schema-bootstrap shape mirrors ``shared.runtime_state.identity_store``:
``__init__`` runs ``initialize()`` once, ``connect()`` is a context
manager that commits on exit, foreign keys + WAL pragmas. Production
writers / readers land in subsequent slices (Phase 2.5 dispatch
heuristic is the first writer; Phase 2.6 dispatch LLM, Phase 3.6
calibration reconciliation are subsequent readers). Read-only consumers
should go through ``shared.runtime_state.read_models``'s
``chief_of_staff_run_by_brief`` and
``cross_brief_observations_for_principal`` to preserve the writer-vs-
reader split (no DDL or meta rewrite on the read path).
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
        """Insert one ``chief_of_staff_runs`` row and return the new id.

        Slice 2.5's dispatch heuristic is the first production writer for
        this table. The writer is append-only by design: each dispatch
        invocation inserts a new row so read models can return the latest
        attempt while preserving historical provenance.
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

        Audit Move #1. Writes the small-version of the handoff arc:
        each module's run-end path persists a structured per-source
        summary (top saves + signal summary + confidence) into the
        latest ``chief_of_staff_runs`` row's ``handoff_payloads_json``
        keyed by source. The synthesis call site at
        :mod:`market_intelligence.reflection` reads the persisted
        payloads and folds them into the synthesis user prompt's
        cross-source narrative context.

        Behavior:
        - Reads the latest ``chief_of_staff_runs`` row for ``brief_id``
          (matching :func:`shared.runtime_state.read_models.chief_of_staff_run_by_brief`'s
          ``ORDER BY started_at DESC, id DESC`` ordering).
        - Parses the existing ``handoff_payloads_json`` (defaulting to
          ``{}`` on missing / malformed).
        - Sets ``handoff_payloads[source] = payload`` (last-write-wins
          for re-runs; the writer is responsible for delivering the
          intended payload shape).
        - UPDATE the row's ``handoff_payloads_json``.

        Returns ``True`` when a row was updated, ``False`` when no
        ``chief_of_staff_runs`` row exists for the brief (typical for
        single-module briefs that didn't go through dispatch). The
        return value is non-fatal; callers should log + continue.

        Concurrent updates: SQLite WAL gives row-level isolation per
        transaction; two writers merging into the same brief land
        cleanly because each transaction reads-modifies-writes inside
        ``self.connect()``'s commit-on-exit envelope. Last-write-wins
        ordering is determined by SQLite's serialization order.
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
