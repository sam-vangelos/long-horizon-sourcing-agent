"""Global recruiter store — the durable RECRUITER primitive (reopen Stage 1).

The reopen decision: the recruiter, not the brief, is Cloris's durable entity.
Today the data model has NO recruiter entity — the recruiter is implicit,
reconstructed per-brief from columns on brief-scoped rows, and discarded. This
module adds the missing first-class entity.

This is a SEPARATE database from per-state-dir ``runtime_state.sqlite3`` for the
same reason ``identity_store.py`` is: a recruiter spans every brief and every
source, so a recruiter table inside any one source's state_dir cannot reference
the others. Lives at ``output/state/_recruiter/recruiter.sqlite3``, sibling to
``output/state/_identity/identity.sqlite3``.

Schema (mirrors the identity_store persons/brief_persons durable-entity + join
pattern; see ~/Downloads/cloris_primitive_reopen_spec.md §2.1):
    recruiters                 -- the durable entity (no brief_id)
    recruiter_briefs           -- (recruiter_id, brief_id) membership/ownership
    recruiter_taste_signals    -- the compounding axis: learned calibration,
                                  cross-brief, soft-superseded not mutated
    recruiter_candidate_history -- per-(recruiter, person) accretion across briefs

CROSS-DB NOTE (correction to the spec): the spec FKs
``recruiter_candidate_history.person_id REFERENCES persons(id)``, but ``persons``
lives in identity.sqlite3 — a different database. SQLite cannot FK across
databases, so ``person_id`` here is a plain column (a soft cross-DB reference),
exactly as ``candidate_persons`` softly references cross-state-dir candidates.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

CURRENT_RECRUITER_SCHEMA_VERSION = "1"

# Reopen Stage 2: where the init guard points the operator when it
# detects that existing recruiter rows were written under a different
# ``recruiter_id`` than the resolver now returns (the Phase-2 auth swap
# resolved a different principal than Stage-1 data was accreted under).
# The tool itself is a Phase-2 prerequisite — stubbed now, named here so
# the refusal is actionable rather than a bare exception.
RECRUITER_ID_MIGRATION_TOOL = "tools/migrate_recruiter_id_stage1_to_phase2.py"


class RecruiterIdMismatchError(RuntimeError):
    """Raised when the recruiter store holds data under a different id.

    Reopen Stage 2 init guard (adversarial-ledger flaw
    "backfill-corruption"). When :func:`shared.recruiter_context` resolves
    a recruiter id (e.g. Phase-2 auth) that is absent from a non-empty
    recruiters table, opening the store with that ``expected_recruiter_id``
    refuses rather than silently mixing two recruiters' accreted taste
    signals / brief membership. The remediation is the Phase-2 migration
    tool (:data:`RECRUITER_ID_MIGRATION_TOOL`), not a re-run.
    """

    def __init__(self, expected_recruiter_id: int, existing_ids: list[int]):
        self.expected_recruiter_id = expected_recruiter_id
        self.existing_ids = existing_ids
        super().__init__(
            f"recruiter store holds rows under recruiter id(s) {existing_ids!r} "
            f"but the current resolver returned {expected_recruiter_id!r}. "
            f"Stage-1 data must be migrated before a different principal can "
            f"write to it — run {RECRUITER_ID_MIGRATION_TOOL} (Phase-2 "
            f"prerequisite)."
        )

# recruiter_briefs.relationship
RELATIONSHIP_OWNER = "owner"
RELATIONSHIP_COLLABORATOR = "collaborator"

# recruiter_taste_signals.signal_kind
SIGNAL_RUBRIC_REFINEMENT = "rubric_refinement"
SIGNAL_PRINCIPLE_FEEDBACK = "principle_feedback"
SIGNAL_HM_BAR_CORRECTION = "hm_bar_correction"
SIGNAL_ARCHETYPE_PREFERENCE = "archetype_preference"
KNOWN_SIGNAL_KINDS = frozenset(
    {
        SIGNAL_RUBRIC_REFINEMENT,
        SIGNAL_PRINCIPLE_FEEDBACK,
        SIGNAL_HM_BAR_CORRECTION,
        SIGNAL_ARCHETYPE_PREFERENCE,
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RecruiterStore:
    """SQLite-backed global recruiter store.

    Mirrors ``IdentityStore`` / ``RuntimeStateStore.connect()``: context manager
    that commits on exit, ``sqlite3.Row`` factory, foreign-keys + WAL pragmas.
    """

    def __init__(
        self, db_path: str | Path, *, expected_recruiter_id: int | None = None
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize(expected_recruiter_id=expected_recruiter_id)

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

    def initialize(self, *, expected_recruiter_id: int | None = None) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recruiters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_handle TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recruiter_briefs (
                    recruiter_id INTEGER NOT NULL,
                    brief_id TEXT NOT NULL,
                    relationship TEXT NOT NULL DEFAULT 'owner',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (recruiter_id, brief_id),
                    FOREIGN KEY (recruiter_id) REFERENCES recruiters(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_recruiter_briefs_brief
                    ON recruiter_briefs(brief_id);

                CREATE TABLE IF NOT EXISTS recruiter_taste_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recruiter_id INTEGER NOT NULL,
                    signal_kind TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    source_brief_id TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    superseded_by INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (recruiter_id) REFERENCES recruiters(id) ON DELETE CASCADE,
                    FOREIGN KEY (superseded_by) REFERENCES recruiter_taste_signals(id)
                );

                CREATE INDEX IF NOT EXISTS idx_taste_signals_recruiter_domain
                    ON recruiter_taste_signals(recruiter_id, domain)
                    WHERE superseded_by IS NULL;

                CREATE TABLE IF NOT EXISTS recruiter_candidate_history (
                    recruiter_id INTEGER NOT NULL,
                    person_id INTEGER NOT NULL,
                    first_seen_brief TEXT NOT NULL,
                    last_seen_brief TEXT NOT NULL,
                    times_surfaced INTEGER NOT NULL DEFAULT 1,
                    last_lifecycle_state TEXT NOT NULL DEFAULT '',
                    last_recruiter_action TEXT,
                    judgment_accuracy_history_json TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY (recruiter_id, person_id),
                    FOREIGN KEY (recruiter_id) REFERENCES recruiters(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_recruiter_candidate_person
                    ON recruiter_candidate_history(person_id);

                -- Reopen Refactor X: the per-(recruiter, person) CURRENT-STATE
                -- authority. Distinct from recruiter_candidate_history above,
                -- which is an ACCRETION log (times_surfaced, first/last brief).
                -- This table holds the single resolved current lifecycle of a
                -- person FOR a recruiter — the cross-DB roll-up of the
                -- brief-keyed candidates rows that person maps to. Populated by
                -- the resolution-hook fill (recruiter_candidate_fill.py), which
                -- joins person_id -> candidate_persons -> the per-state-dir
                -- candidates row and picks the most-recent. PK (recruiter_id,
                -- person_id) so the fill is an idempotent upsert; person_id is
                -- a soft cross-DB key (persons live in identity.sqlite3), a
                -- plain column with no FK, exactly like
                -- recruiter_candidate_history above. The brief-keyed candidates
                -- table stays the source of truth; this is the recruiter-scoped
                -- materialized current-state, and the divergence metric
                -- (recruiter_candidate_fill.divergence_report) is the gate that
                -- proves the two agree before Refactor Y can lean on it.
                CREATE TABLE IF NOT EXISTS recruiter_candidates (
                    recruiter_id INTEGER NOT NULL,
                    person_id INTEGER NOT NULL,
                    current_lifecycle_state TEXT NOT NULL DEFAULT '',
                    terminal_decision TEXT,
                    terminal_payload_json TEXT NOT NULL DEFAULT '{}',
                    last_source TEXT NOT NULL DEFAULT '',
                    last_identity_key TEXT NOT NULL DEFAULT '',
                    last_seen_at TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (recruiter_id, person_id),
                    FOREIGN KEY (recruiter_id) REFERENCES recruiters(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_recruiter_candidates_person
                    ON recruiter_candidates(person_id);

                -- Reopen Stage 3a: idempotency ledger for the sighting hook.
                -- record_candidate_sighting ACCRETES (bumps times_surfaced on
                -- every call), but the sighting hook fires on every read-path
                -- re-resolution (control_plane.aggregate_workspace /
                -- _monolith.api_identity_pending both call
                -- resolve_persons_for_brief then the hook). Without a guard,
                -- times_surfaced would inflate by one per page load. This
                -- ledger records which (recruiter, person, brief) triples have
                -- already produced a sighting so re-running resolution for the
                -- SAME brief is a no-op, while a NEW brief still accretes — i.e.
                -- times_surfaced stays equal to the count of DISTINCT briefs on
                -- which the recruiter has seen the person (the Stage-1 semantic
                -- asserted by test_candidate_sighting_accretes_across_briefs).
                CREATE TABLE IF NOT EXISTS recruiter_candidate_sightings (
                    recruiter_id INTEGER NOT NULL,
                    person_id INTEGER NOT NULL,
                    brief_id TEXT NOT NULL,
                    first_recorded_at TEXT NOT NULL,
                    PRIMARY KEY (recruiter_id, person_id, brief_id),
                    FOREIGN KEY (recruiter_id) REFERENCES recruiters(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("recruiter_schema_version", CURRENT_RECRUITER_SCHEMA_VERSION),
            )

            # Reopen Stage 2 init guard. Only fires when a caller passes an
            # expected id (Phase-2 auth resolution, or a write site that
            # already resolved the acting recruiter). A non-empty recruiters
            # table whose ids don't include the expected id means the
            # resolver now returns a *different* principal than the data was
            # accreted under — refuse rather than mix two recruiters' taste
            # signals / brief membership. Empty table is fine: the first
            # ``upsert_recruiter`` materializes the row. Default
            # (expected_recruiter_id is None) skips the check entirely so
            # Stage-1 callers and the self-bootstrapping resolver open the
            # store without a chicken-and-egg recursion through
            # ``get_current_recruiter_id``.
            if expected_recruiter_id is not None:
                existing_ids = [
                    int(r["id"])
                    for r in conn.execute("SELECT id FROM recruiters").fetchall()
                ]
                if existing_ids and expected_recruiter_id not in existing_ids:
                    raise RecruiterIdMismatchError(
                        expected_recruiter_id=expected_recruiter_id,
                        existing_ids=existing_ids,
                    )

    # ------------------------------------------------------------------
    # recruiters
    # ------------------------------------------------------------------

    def upsert_recruiter(self, canonical_handle: str, *, display_name: str = "") -> int:
        """Get-or-create a recruiter by canonical_handle; return its id.

        Idempotent: a second call with the same handle returns the existing id
        and refreshes updated_at / display_name (if a non-empty one is given)."""
        handle = canonical_handle.strip().lower()
        if not handle:
            raise ValueError("canonical_handle must be non-empty")
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM recruiters WHERE canonical_handle = ?", (handle,)
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE recruiters SET updated_at = ?, "
                    "display_name = CASE WHEN ? != '' THEN ? ELSE display_name END "
                    "WHERE id = ?",
                    (now, display_name, display_name, row["id"]),
                )
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO recruiters(canonical_handle, display_name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (handle, display_name, now, now),
            )
            return int(cur.lastrowid)

    def get_recruiter(self, recruiter_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM recruiters WHERE id = ?", (recruiter_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_recruiter_by_handle(self, canonical_handle: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM recruiters WHERE canonical_handle = ?",
                (canonical_handle.strip().lower(),),
            ).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # recruiter_briefs (membership)
    # ------------------------------------------------------------------

    def link_brief(
        self, recruiter_id: int, brief_id: str, *, relationship: str = RELATIONSHIP_OWNER
    ) -> None:
        """Record that a recruiter owns/collaborates on a brief. Idempotent."""
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO recruiter_briefs"
                "(recruiter_id, brief_id, relationship, created_at) VALUES (?, ?, ?, ?)",
                (recruiter_id, brief_id, relationship, _utc_now()),
            )

    def briefs_for_recruiter(self, recruiter_id: int) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT brief_id FROM recruiter_briefs WHERE recruiter_id = ? "
                "ORDER BY created_at",
                (recruiter_id,),
            ).fetchall()
            return [r["brief_id"] for r in rows]

    def recruiter_for_brief(self, brief_id: str) -> int | None:
        """Reverse of :meth:`briefs_for_recruiter`: which recruiter owns a
        brief. Returns ``None`` when the brief was never linked.

        Reopen Stage 3a: the sighting hook resolves the acting recruiter from
        the brief itself (the Stage-2 ``recruiter_briefs`` link) rather than
        the ambient resolver, so a sighting lands under the recruiter who
        actually owns the brief even if the resolver later changes. Falls back
        to the ambient resolver + a lazy link at the call site when this
        returns ``None`` (a brief that has runs but was never backfilled).

        The brief→recruiter relationship is 1:N in the schema
        (``recruiter_briefs`` PK is ``(recruiter_id, brief_id)``), but Stage 1
        has exactly one implicit recruiter, so a brief resolves to at most one
        owner today. When several rows exist, the earliest-linked owner wins
        (deterministic, mirrors ``briefs_for_recruiter``'s ``ORDER BY
        created_at``).
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT recruiter_id FROM recruiter_briefs WHERE brief_id = ? "
                "ORDER BY created_at LIMIT 1",
                (brief_id,),
            ).fetchone()
            return int(row["recruiter_id"]) if row is not None else None

    # ------------------------------------------------------------------
    # recruiter_taste_signals (the compounding axis)
    # ------------------------------------------------------------------

    def record_taste_signal(
        self,
        recruiter_id: int,
        *,
        signal_kind: str,
        domain: str,
        payload: dict[str, Any] | None = None,
        source_brief_id: str | None = None,
        confidence: float = 0.5,
    ) -> int:
        """Append a taste signal. Signals are append-only + soft-superseded, never
        mutated in place — so calibration history is auditable."""
        if signal_kind not in KNOWN_SIGNAL_KINDS:
            raise ValueError(
                f"unknown signal_kind {signal_kind!r}; known: {sorted(KNOWN_SIGNAL_KINDS)}"
            )
        import json  # noqa: PLC0415

        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO recruiter_taste_signals"
                "(recruiter_id, signal_kind, domain, source_brief_id, payload_json, "
                "confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    recruiter_id,
                    signal_kind,
                    domain,
                    source_brief_id,
                    json.dumps(payload or {}),
                    confidence,
                    _utc_now(),
                ),
            )
            return int(cur.lastrowid)

    def supersede_signal(self, old_signal_id: int, new_signal_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE recruiter_taste_signals SET superseded_by = ? WHERE id = ?",
                (new_signal_id, old_signal_id),
            )

    def active_taste_signals(
        self, recruiter_id: int, *, domain: str | None = None
    ) -> list[dict[str, Any]]:
        """Live (non-superseded) signals for a recruiter, optionally domain-scoped.
        This is what a future brief hydrates as priors."""
        import json  # noqa: PLC0415

        q = (
            "SELECT * FROM recruiter_taste_signals "
            "WHERE recruiter_id = ? AND superseded_by IS NULL"
        )
        params: list[Any] = [recruiter_id]
        if domain is not None:
            q += " AND domain = ?"
            params.append(domain)
        q += " ORDER BY created_at"
        with self.connect() as conn:
            rows = conn.execute(q, params).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["payload"] = json.loads(d.pop("payload_json") or "{}")
                out.append(d)
            return out

    # ------------------------------------------------------------------
    # recruiter_candidate_history (cross-brief accretion)
    # ------------------------------------------------------------------

    def record_candidate_sighting(
        self,
        recruiter_id: int,
        person_id: int,
        *,
        brief_id: str,
        lifecycle_state: str = "",
        recruiter_action: str | None = None,
    ) -> None:
        """Accrue one sighting of a person by a recruiter. On repeat, bumps
        times_surfaced and updates last_seen_brief / last state / last action;
        first_seen_brief is preserved."""
        now_brief = brief_id
        with self.connect() as conn:
            row = conn.execute(
                "SELECT times_surfaced FROM recruiter_candidate_history "
                "WHERE recruiter_id = ? AND person_id = ?",
                (recruiter_id, person_id),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO recruiter_candidate_history"
                    "(recruiter_id, person_id, first_seen_brief, last_seen_brief, "
                    "times_surfaced, last_lifecycle_state, last_recruiter_action) "
                    "VALUES (?, ?, ?, ?, 1, ?, ?)",
                    (recruiter_id, person_id, now_brief, now_brief, lifecycle_state, recruiter_action),
                )
            else:
                conn.execute(
                    "UPDATE recruiter_candidate_history SET "
                    "last_seen_brief = ?, times_surfaced = times_surfaced + 1, "
                    "last_lifecycle_state = ?, "
                    "last_recruiter_action = COALESCE(?, last_recruiter_action) "
                    "WHERE recruiter_id = ? AND person_id = ?",
                    (now_brief, lifecycle_state, recruiter_action, recruiter_id, person_id),
                )

    def record_candidate_sighting_once(
        self,
        recruiter_id: int,
        person_id: int,
        *,
        brief_id: str,
        lifecycle_state: str = "",
        recruiter_action: str | None = None,
    ) -> bool:
        """Idempotent sighting: accrue exactly once per (recruiter, person,
        brief), no matter how many times it's called.

        Reopen Stage 3a. :meth:`record_candidate_sighting` accretes on every
        call; this wrapper gates it through the ``recruiter_candidate_sightings``
        ledger so the read-path re-resolution (which calls the hook on every
        page load) cannot inflate ``times_surfaced``. Returns ``True`` when this
        call recorded a NEW sighting (first time this person was seen on this
        brief by this recruiter), ``False`` when it was a no-op (already
        recorded).

        Atomic: the ledger insert and the history accrual share one connection
        / transaction, so a crash leaves neither — never a ledger row without
        its sighting (which would silently swallow a real first-sighting) and
        never a sighting without its ledger row (which would let the next read
        double-count). The ``INSERT OR IGNORE`` + ``total_changes`` check is the
        gate: it both detects and claims the triple in a single statement, so
        two concurrent callers can't both think they're first.
        """
        now_brief = brief_id
        with self.connect() as conn:
            before = conn.total_changes
            conn.execute(
                "INSERT OR IGNORE INTO recruiter_candidate_sightings"
                "(recruiter_id, person_id, brief_id, first_recorded_at) "
                "VALUES (?, ?, ?, ?)",
                (recruiter_id, person_id, brief_id, _utc_now()),
            )
            if conn.total_changes == before:
                # Triple already in the ledger — already sighted this brief.
                return False

            row = conn.execute(
                "SELECT times_surfaced FROM recruiter_candidate_history "
                "WHERE recruiter_id = ? AND person_id = ?",
                (recruiter_id, person_id),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO recruiter_candidate_history"
                    "(recruiter_id, person_id, first_seen_brief, last_seen_brief, "
                    "times_surfaced, last_lifecycle_state, last_recruiter_action) "
                    "VALUES (?, ?, ?, ?, 1, ?, ?)",
                    (recruiter_id, person_id, now_brief, now_brief, lifecycle_state, recruiter_action),
                )
            else:
                conn.execute(
                    "UPDATE recruiter_candidate_history SET "
                    "last_seen_brief = ?, times_surfaced = times_surfaced + 1, "
                    "last_lifecycle_state = ?, "
                    "last_recruiter_action = COALESCE(?, last_recruiter_action) "
                    "WHERE recruiter_id = ? AND person_id = ?",
                    (now_brief, lifecycle_state, recruiter_action, recruiter_id, person_id),
                )
            return True

    def candidate_history(self, recruiter_id: int, person_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM recruiter_candidate_history "
                "WHERE recruiter_id = ? AND person_id = ?",
                (recruiter_id, person_id),
            ).fetchone()
            return dict(row) if row else None

    def candidate_history_for_recruiter(self, recruiter_id: int) -> list[dict[str, Any]]:
        """All recruiter_candidate_history rows for a recruiter, ordered by person_id.

        Reopen Stage 3 (R3.1). Bulk sibling of :meth:`candidate_history` (which
        reads one (recruiter, person) row). Pure read over the ACCRETION log —
        count + first/last brief + last lifecycle — feeding the dashboard
        presence panel. No schema change.
        """
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM recruiter_candidate_history "
                "WHERE recruiter_id = ? ORDER BY person_id",
                (recruiter_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # recruiter_candidates (current-state authority — Refactor X)
    # ------------------------------------------------------------------

    def upsert_recruiter_candidate(
        self,
        recruiter_id: int,
        person_id: int,
        *,
        current_lifecycle_state: str,
        terminal_decision: str | None,
        terminal_payload_json: str,
        source: str,
        identity_key: str,
        last_seen_at: str | None,
    ) -> None:
        """Set the CURRENT-STATE row for a (recruiter, person) to the given state.

        Reopen Refactor X. Unlike :meth:`record_candidate_sighting` (which
        ACCRETES), this is a pure upsert of the resolved current lifecycle: the
        first call INSERTs, every later call UPDATEs the same
        ``(recruiter_id, person_id)`` row in place. It carries no count — the
        accretion log already owns "how many times / which briefs"; this owns
        "what state is the person in now, for this recruiter."

        ``terminal_payload_json`` is stored verbatim (the fill already has it as
        a JSON string from the source candidates row — no re-encode), mirroring
        how the brief-keyed candidates table holds the column. ``last_source`` /
        ``last_identity_key`` record which per-state-dir candidate row won the
        most-recent merge, so the divergence metric can point at the exact
        brief-keyed row to diff. Idempotent on the PK: re-running the fill for
        the same person is an UPDATE to one row, never a duplicate."""

        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO recruiter_candidates"
                "(recruiter_id, person_id, current_lifecycle_state, terminal_decision, "
                "terminal_payload_json, last_source, last_identity_key, last_seen_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(recruiter_id, person_id) DO UPDATE SET "
                "current_lifecycle_state = excluded.current_lifecycle_state, "
                "terminal_decision = excluded.terminal_decision, "
                "terminal_payload_json = excluded.terminal_payload_json, "
                "last_source = excluded.last_source, "
                "last_identity_key = excluded.last_identity_key, "
                "last_seen_at = excluded.last_seen_at, "
                "updated_at = excluded.updated_at",
                (
                    recruiter_id,
                    person_id,
                    current_lifecycle_state,
                    terminal_decision,
                    terminal_payload_json,
                    source,
                    identity_key,
                    last_seen_at,
                    now,
                ),
            )

    def recruiter_candidate(
        self, recruiter_id: int, person_id: int
    ) -> dict[str, Any] | None:
        """Read the current-state authority row for a (recruiter, person)."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM recruiter_candidates "
                "WHERE recruiter_id = ? AND person_id = ?",
                (recruiter_id, person_id),
            ).fetchone()
            return dict(row) if row else None

    def delete_recruiter_candidate(self, recruiter_id: int, person_id: int) -> None:
        """Hard-delete the current-state authority row for a (recruiter, person).

        Reopen Y.5.8 (F4). The inverse of :meth:`upsert_recruiter_candidate`:
        where the upsert writes/refreshes a ``(recruiter_id, person_id)`` row,
        this removes it. The merge re-sync (``record_recruiter_merge``) calls it
        for the DROPPED person after an identity merge hard-deletes that person
        from ``identity.persons`` — the authority row that still points at the
        now-nonexistent ``person_id`` is a dangling soft reference (the F3
        ``recruiter_persons_sweep`` flags exactly these), so it must go, not be
        refreshed. Keyed on the same ``(recruiter_id, person_id)`` PK; deleting
        an absent row is a harmless no-op (``DELETE`` of zero rows)."""
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM recruiter_candidates "
                "WHERE recruiter_id = ? AND person_id = ?",
                (recruiter_id, person_id),
            )

    def recruiter_ids_for_persons(self, person_ids: Iterable[int]) -> list[int]:
        """Every recruiter with a ``recruiter_candidates`` row for ANY of ``person_ids``.

        Reopen Y.5.8 (F4). The "all affected recruiters" read for the merge /
        unlink re-sync. The authority keys each row on ``(recruiter_id,
        person_id)`` and there is no reverse "which recruiters reference this
        person" index in the call sites that need it, so this is that lookup:
        ``SELECT DISTINCT recruiter_id ... WHERE person_id IN (...)``. The
        re-sync must touch EVERY such recruiter, not just the one whose brief
        triggered the merge — otherwise a tombstone (an authority row pointing
        at a dropped/changed person) survives in the OTHER recruiters'
        authorities and the F3 sweep keeps flagging them. Ordered for
        determinism; empty / all-absent input is a no-op (empty list)."""
        ids = sorted({int(p) for p in person_ids})
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT recruiter_id FROM recruiter_candidates "
                f"WHERE person_id IN ({placeholders}) ORDER BY recruiter_id",
                tuple(ids),
            ).fetchall()
        return [int(r["recruiter_id"]) for r in rows]
