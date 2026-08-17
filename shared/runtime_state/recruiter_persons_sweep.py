"""Pure-read persons-existence sweep over recruiter_candidates (Reopen Y.5.7 / F3).

The ``recruiter_candidates`` CURRENT-STATE authority keys every row on a
``person_id`` that is a SOFT cross-DB reference: ``persons`` lives in
identity.sqlite3, a different database, so SQLite cannot FK-enforce it (the
recruiter store's own docstring flags this — ``person_id`` is "a plain column
with no FK, exactly like recruiter_candidate_history"). A soft reference can
dangle: an authority row can point at a ``person_id`` that no longer exists in
``identity.persons`` (an identity row hard-deleted, a backfill that wrote under a
stale id, a partial migration). Nothing detects that today.

This sweep is the detector. For EVERY recruiter (``SELECT id FROM recruiters
ORDER BY id`` — the same enumeration the whole-population backfill uses), it
reads that recruiter's ``recruiter_candidates.person_id`` set and checks each
against ``identity.persons(id)`` via ``IdentityStore`` (the canonical reader of
that DB). It returns the dangling ``(recruiter_id, person_id)`` pairs — authority
rows whose person no longer exists.

PURE READ. It opens both DBs through their stores' read paths and SELECTs only;
it writes NOTHING to the recruiter DB and NOTHING to the identity DB. It is a
diagnostic that a future re-sync (NOT this slice) would consume — here it only
reports.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class DanglingCandidate:
    """One ``recruiter_candidates`` row whose ``person_id`` has no identity row."""

    recruiter_id: int
    person_id: int


@dataclass
class PersonsSweepResult:
    """Outcome of the all-recruiters persons-existence sweep (pure read)."""

    recruiters_scanned: int = 0
    candidate_rows_scanned: int = 0
    distinct_persons_checked: int = 0
    dangling: list[DanglingCandidate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "recruiters_scanned": self.recruiters_scanned,
            "candidate_rows_scanned": self.candidate_rows_scanned,
            "distinct_persons_checked": self.distinct_persons_checked,
            "dangling_count": len(self.dangling),
            "dangling": [
                {"recruiter_id": d.recruiter_id, "person_id": d.person_id}
                for d in self.dangling
            ],
        }


@contextmanager
def _readonly(db_path: Path) -> Iterator[sqlite3.Connection | None]:
    """Open ``db_path`` READ-ONLY (``file:...?mode=ro``); yield ``None`` if absent.

    This is the F3 no-write contract enforced at the connection layer, the SAME
    idiom ``recruiter_candidate_fill._read_candidate_current_state`` uses to read
    a per-state-dir DB without ever mutating it: a ``mode=ro`` URI connection
    cannot create the file, cannot run DDL, cannot write — so the sweep's
    pure-read guarantee is structural, not merely "we only issued SELECTs". An
    absent DB yields ``None`` (nothing to read), matching the fill's
    ``if not db_path.exists(): return None`` posture — the sweep never
    materializes a DB as a side effect of reading it (which is exactly what
    going through ``IdentityStore`` / ``RecruiterStore`` constructors would do
    via their ``initialize()`` write).
    """

    if not Path(db_path).exists():
        yield None
        return
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _all_recruiter_ids(recruiter_db_path: Path) -> list[int]:
    """Every recruiter id, oldest-first. Pure read-only of the recruiter DB.

    Mirrors ``backfill_recruiter_store``'s all-recruiters enumeration
    (``SELECT id FROM recruiters ORDER BY id``) — no get-or-create, just the
    recruiters that already exist. Absent DB / missing table => empty.
    """

    with _readonly(recruiter_db_path) as conn:
        if conn is None:
            return []
        try:
            return [
                int(r["id"])
                for r in conn.execute(
                    "SELECT id FROM recruiters ORDER BY id"
                ).fetchall()
            ]
        except sqlite3.OperationalError:
            return []


def _candidate_person_ids(recruiter_db_path: Path, recruiter_id: int) -> list[int]:
    """The ``recruiter_candidates.person_id`` set for one recruiter (read-only).

    The authority table (not the accretion log): these are the persons the
    recruiter has a resolved current-state row for. Ordered for determinism.
    """

    with _readonly(recruiter_db_path) as conn:
        if conn is None:
            return []
        try:
            return [
                int(r["person_id"])
                for r in conn.execute(
                    "SELECT person_id FROM recruiter_candidates "
                    "WHERE recruiter_id = ? ORDER BY person_id",
                    (recruiter_id,),
                ).fetchall()
            ]
        except sqlite3.OperationalError:
            return []


def _existing_person_ids(identity_db_path: Path, person_ids: set[int]) -> set[int]:
    """Which of ``person_ids`` actually exist in ``identity.persons`` (read-only).

    Reads the identity DB READ-ONLY (``mode=ro`` URI) — the canonical no-mutate
    path for cross-DB identity reads. Batched so the sweep is one SELECT per
    recruiter rather than one per person; the ``IN (...)`` set is bounded by a
    recruiter's candidate count. Empty input is a no-op (no query, empty set).
    An absent identity DB yields the empty set, which means EVERY checked
    person_id is treated as dangling — the correct, conservative reading when
    the identity store the authority references does not exist at all.
    """

    if not person_ids:
        return set()

    ordered = sorted(person_ids)
    with _readonly(identity_db_path) as conn:
        if conn is None:
            return set()
        placeholders = ",".join("?" for _ in ordered)
        try:
            rows = conn.execute(
                f"SELECT id FROM persons WHERE id IN ({placeholders})",
                tuple(ordered),
            ).fetchall()
        except sqlite3.OperationalError:
            return set()
    return {int(r["id"]) for r in rows}


def sweep_recruiter_candidate_persons(
    *,
    recruiter_db_path: Path | None = None,
    identity_db_path: Path | None = None,
) -> PersonsSweepResult:
    """Flag every ``recruiter_candidates`` row whose person has no identity row.

    All-recruiters, pure read. For each recruiter (``SELECT id FROM recruiters
    ORDER BY id``), reads its ``recruiter_candidates.person_id`` set and checks
    each against ``identity.persons(id)``; a person_id present in the authority
    but absent from ``persons`` is a dangling soft reference, reported as a
    ``DanglingCandidate(recruiter_id, person_id)``.

    Writes NOTHING — neither the recruiter DB nor the identity DB. Both are
    opened READ-ONLY (``file:...?mode=ro`` URI, via :func:`_readonly`), so the
    no-write guarantee is STRUCTURAL: a read-only connection cannot create the
    file, cannot run DDL, cannot write — not even the schema-version meta
    re-stamp that going through ``IdentityStore`` / ``RecruiterStore``
    constructors would trigger. File mtimes are therefore stable across the
    sweep, and the sweep itself issues only SELECTs.
    """

    if recruiter_db_path is None:
        from shared.output_paths import resolve_recruiter_db_path

        recruiter_db_path = resolve_recruiter_db_path()
    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    result = PersonsSweepResult()
    seen_persons: set[int] = set()

    for recruiter_id in _all_recruiter_ids(recruiter_db_path):
        result.recruiters_scanned += 1
        person_ids = _candidate_person_ids(recruiter_db_path, recruiter_id)
        result.candidate_rows_scanned += len(person_ids)
        if not person_ids:
            continue

        existing = _existing_person_ids(identity_db_path, set(person_ids))
        seen_persons.update(person_ids)
        for person_id in person_ids:
            if person_id not in existing:
                result.dangling.append(
                    DanglingCandidate(
                        recruiter_id=recruiter_id, person_id=person_id
                    )
                )

    result.distinct_persons_checked = len(seen_persons)
    return result
