"""Global cross-module identity SQLite store (Phase F Slice F3).

Why this is a SEPARATE database from per-state-dir
``runtime_state.sqlite3``:

Cross-source identity is, by definition, structurally incapable of
being seen from within a single source's state_dir. A LinkedIn
candidate lives in ``output/state/linkedin/<key>/runtime_state.sqlite3``;
a GitHub candidate of the same human lives in
``output/state/github/<other_key>/runtime_state.sqlite3``. A persons
table inside either DB cannot reference rows in the other. The
F3 architectural-fit critique called this out as the fatal gap in the
original "schema v11 inside runtime_state" plan; the revised contract
ships a separate global store at
``output/state/_identity/identity.sqlite3`` and walks per-state-dir DBs
read-only when resolving.

Schema split (load-bearing):
    persons               -- brief-AGNOSTIC canonical-human row
    brief_persons         -- (brief_id, person_id) membership table
    candidate_persons     -- per-state-dir candidate -> person link
    pending_merge_decisions  -- backend for F6's review-merge affordance

The split between ``persons`` (no brief_id) and ``brief_persons``
(membership) is what unlocks cross-brief calibration in a future phase
("we already worked this person on a different brief"). Brief-keyed
``persons`` would force two rows for the same human, blocking that
loop.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


CURRENT_IDENTITY_SCHEMA_VERSION = "1"


# ``link_kind`` values written to ``candidate_persons.link_kind``.
# Manual is the only kind that auto-resolution refuses to overwrite.
LINK_KIND_AUTO_STRONG = "auto_strong"
LINK_KIND_AUTO_MEDIUM = "auto_medium"
LINK_KIND_MANUAL = "manual"
KNOWN_LINK_KINDS = frozenset(
    {LINK_KIND_AUTO_STRONG, LINK_KIND_AUTO_MEDIUM, LINK_KIND_MANUAL}
)


class IdentityStore:
    """SQLite-backed global identity store.

    Mirrors the ``RuntimeStateStore.connect()`` contract (context
    manager that commits on exit, row factory yields ``sqlite3.Row``,
    foreign keys + WAL pragmas) so callers don't have to relearn a
    second connection style.
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

                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_name TEXT NOT NULL DEFAULT '',
                    canonical_handle TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_persons_canonical_handle
                    ON persons(canonical_handle)
                    WHERE canonical_handle != '';

                CREATE TABLE IF NOT EXISTS brief_persons (
                    brief_id TEXT NOT NULL,
                    person_id INTEGER NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    PRIMARY KEY(brief_id, person_id),
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_brief_persons_brief_id
                    ON brief_persons(brief_id);

                CREATE TABLE IF NOT EXISTS candidate_persons (
                    source TEXT NOT NULL,
                    state_key TEXT NOT NULL,
                    candidate_id INTEGER NOT NULL,
                    person_id INTEGER NOT NULL,
                    brief_id TEXT NOT NULL,
                    link_kind TEXT NOT NULL,
                    match_signal_json TEXT NOT NULL DEFAULT '{}',
                    recruiter_locked INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(source, state_key, candidate_id),
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_candidate_persons_person
                    ON candidate_persons(person_id);
                CREATE INDEX IF NOT EXISTS idx_candidate_persons_brief
                    ON candidate_persons(brief_id);

                CREATE TABLE IF NOT EXISTS pending_merge_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    brief_id TEXT NOT NULL,
                    person_a INTEGER NOT NULL,
                    person_b INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_json TEXT NOT NULL DEFAULT '{}',
                    recruiter_decision TEXT,
                    decided_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(person_a) REFERENCES persons(id) ON DELETE CASCADE,
                    FOREIGN KEY(person_b) REFERENCES persons(id) ON DELETE CASCADE,
                    UNIQUE(brief_id, person_a, person_b)
                );

                CREATE INDEX IF NOT EXISTS idx_pending_merge_brief
                    ON pending_merge_decisions(brief_id)
                    WHERE recruiter_decision IS NULL;
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("identity_schema_version", CURRENT_IDENTITY_SCHEMA_VERSION),
            )
