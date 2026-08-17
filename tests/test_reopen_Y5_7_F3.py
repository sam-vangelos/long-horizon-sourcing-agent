"""Reopen Y.5.7 (F3) — pure-read persons-existence sweep over recruiter_candidates.

``recruiter_candidates.person_id`` is a SOFT cross-DB reference (persons live in
identity.sqlite3, no FK). It can dangle — point at a person_id absent from
``identity.persons``. F3's sweep detects that, all-recruiters, PURE READ. Pins:

- FLAGS A DANGLING ROW: a ``recruiter_candidates`` row whose person_id is NOT in
  ``identity.persons`` is returned in the dangling list (recruiter_id, person_id).
- CLEAN RECRUITER SWEEPS CLEAN: a recruiter whose authority person_ids all exist
  in ``identity.persons`` produces zero dangling rows.
- ALL-RECRUITERS: the sweep enumerates every recruiter (not just one), so a
  dangle under recruiter B is found even when recruiter A is clean.
- PURE READ: the sweep writes NOTHING to either DB — neither the recruiter
  rows nor the identity rows change (row-count + row-content invariant; the
  load-bearing no-write proof).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from shared.runtime_state.identity_store import IdentityStore
from shared.runtime_state.recruiter_persons_sweep import (
    sweep_recruiter_candidate_persons,
)
from shared.runtime_state.recruiter_store import RecruiterStore


@pytest.fixture()
def dbs(tmp_path: Path) -> tuple[Path, Path]:
    """Return (recruiter_db_path, identity_db_path) under tmp_path.

    Both passed EXPLICITLY to the sweep (it accepts both paths), so no resolver
    monkeypatch is needed — the test is hermetic on tmp_path."""

    recruiter_db = tmp_path / "_recruiter" / "recruiter.sqlite3"
    identity_db = tmp_path / "_identity" / "identity.sqlite3"
    recruiter_db.parent.mkdir(parents=True, exist_ok=True)
    identity_db.parent.mkdir(parents=True, exist_ok=True)
    return recruiter_db, identity_db


def _insert_person(identity_db: Path, person_id: int) -> None:
    """Insert a persons row with an explicit id (raw SQL — IdentityStore has no
    insert helper; mirrors how the store's own tests seed rows)."""

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO persons(id, canonical_name, canonical_handle, "
            "created_at, last_seen_at) VALUES (?, '', '', '2026-01-01T00:00:00+00:00', "
            "'2026-01-01T00:00:00+00:00')",
            (person_id,),
        )


def _seed_authority(
    recruiter_db: Path, recruiter_id: int, person_id: int
) -> None:
    """Seed one ``recruiter_candidates`` authority row for (recruiter, person).

    Uses the store's own ``upsert_recruiter_candidate`` (the real write path)
    after materializing the recruiters row the FK needs."""

    store = RecruiterStore(recruiter_db)
    # The recruiters row the recruiter_candidates FK references.
    with store.connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO recruiters(id, canonical_handle, display_name, "
            "created_at, updated_at) VALUES (?, ?, '', "
            "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')",
            (recruiter_id, f"rec-{recruiter_id}"),
        )
    store.upsert_recruiter_candidate(
        recruiter_id,
        person_id,
        current_lifecycle_state="surfaced",
        terminal_decision=None,
        terminal_payload_json="{}",
        source="designer",
        identity_key="k",
        last_seen_at="2026-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# 1. FLAGS A DANGLING ROW.
# ---------------------------------------------------------------------------


def test_sweep_flags_dangling_person(dbs: tuple[Path, Path]) -> None:
    """A recruiter_candidates row whose person_id is absent from
    identity.persons is reported in the dangling list."""

    recruiter_db, identity_db = dbs
    # person 7 exists; person 99 does NOT.
    _insert_person(identity_db, 7)
    _seed_authority(recruiter_db, recruiter_id=1, person_id=7)
    _seed_authority(recruiter_db, recruiter_id=1, person_id=99)

    result = sweep_recruiter_candidate_persons(
        recruiter_db_path=recruiter_db, identity_db_path=identity_db
    )

    dangling = {(d.recruiter_id, d.person_id) for d in result.dangling}
    assert dangling == {(1, 99)}
    assert result.recruiters_scanned == 1
    assert result.candidate_rows_scanned == 2
    assert result.distinct_persons_checked == 2


# ---------------------------------------------------------------------------
# 2. CLEAN RECRUITER SWEEPS CLEAN.
# ---------------------------------------------------------------------------


def test_sweep_clean_recruiter_has_no_dangling(dbs: tuple[Path, Path]) -> None:
    """A recruiter whose authority person_ids all exist in identity.persons
    produces zero dangling rows."""

    recruiter_db, identity_db = dbs
    _insert_person(identity_db, 7)
    _insert_person(identity_db, 8)
    _seed_authority(recruiter_db, recruiter_id=1, person_id=7)
    _seed_authority(recruiter_db, recruiter_id=1, person_id=8)

    result = sweep_recruiter_candidate_persons(
        recruiter_db_path=recruiter_db, identity_db_path=identity_db
    )

    assert result.dangling == []
    assert result.candidate_rows_scanned == 2


# ---------------------------------------------------------------------------
# 3. ALL-RECRUITERS — a dangle under recruiter B is found while A is clean.
# ---------------------------------------------------------------------------


def test_sweep_spans_all_recruiters(dbs: tuple[Path, Path]) -> None:
    """Recruiter 1 clean (person 7 exists), recruiter 2 dangling (person 99
    absent). The sweep enumerates BOTH and flags only recruiter 2's dangle —
    proving it is all-recruiters, not single-recruiter."""

    recruiter_db, identity_db = dbs
    _insert_person(identity_db, 7)
    _seed_authority(recruiter_db, recruiter_id=1, person_id=7)
    _seed_authority(recruiter_db, recruiter_id=2, person_id=99)

    result = sweep_recruiter_candidate_persons(
        recruiter_db_path=recruiter_db, identity_db_path=identity_db
    )

    dangling = {(d.recruiter_id, d.person_id) for d in result.dangling}
    assert dangling == {(2, 99)}
    assert result.recruiters_scanned == 2


# ---------------------------------------------------------------------------
# 4. PURE READ — the sweep writes NOTHING to either DB.
# ---------------------------------------------------------------------------


def test_sweep_writes_nothing_to_either_db(dbs: tuple[Path, Path]) -> None:
    """The sweep is a pure read: neither the identity rows nor the recruiter
    rows change, AND the DB file mtimes are stable. Both DBs are seeded (and
    thus created/initialized) in the seed phase BEFORE the baseline snapshot;
    the sweep then opens them READ-ONLY (``mode=ro`` URI), so it cannot write
    even the schema-version meta row. Two proofs, both asserted hard:
      - row-count + row-content invariance (a write would alter a data row);
      - file-mtime invariance (a read-only connection cannot touch the file)."""

    recruiter_db, identity_db = dbs
    _insert_person(identity_db, 7)
    _seed_authority(recruiter_db, recruiter_id=1, person_id=7)
    _seed_authority(recruiter_db, recruiter_id=1, person_id=99)

    def _ro_query(db_path: Path, sql: str) -> list[tuple]:
        """Snapshot rows via a READ-ONLY connection.

        The snapshot itself MUST NOT write — going through ``IdentityStore`` /
        ``RecruiterStore`` constructors would re-stamp the schema-version meta
        row and move the file mtime, contaminating the mtime measurement this
        test makes. ``mode=ro`` reads without touching the file (verified: a
        sweep-only window leaves both mtimes stable)."""

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            return [tuple(r) for r in conn.execute(sql).fetchall()]
        finally:
            conn.close()

    def _snapshot() -> dict:
        return {
            "persons": _ro_query(
                identity_db,
                "SELECT id, canonical_name, canonical_handle, created_at, "
                "last_seen_at FROM persons ORDER BY id",
            ),
            "recruiter_candidates": _ro_query(
                recruiter_db,
                "SELECT recruiter_id, person_id, current_lifecycle_state, "
                "terminal_decision, terminal_payload_json, last_source, "
                "last_identity_key, last_seen_at FROM recruiter_candidates "
                "ORDER BY recruiter_id, person_id",
            ),
            "recruiters": _ro_query(
                recruiter_db, "SELECT id, canonical_handle FROM recruiters ORDER BY id"
            ),
        }

    before = _snapshot()
    # Capture mtime with NOTHING between this and the sweep but the sweep
    # itself — the read-only snapshot above does not move it (verified), so any
    # mtime delta is attributable solely to the sweep.
    before_mtimes = (identity_db.stat().st_mtime_ns, recruiter_db.stat().st_mtime_ns)

    result = sweep_recruiter_candidate_persons(
        recruiter_db_path=recruiter_db, identity_db_path=identity_db
    )

    after_mtimes = (identity_db.stat().st_mtime_ns, recruiter_db.stat().st_mtime_ns)
    after = _snapshot()

    # Sanity: the sweep did its job (so we know we're asserting no-write on a
    # sweep that actually ran, not a no-op).
    assert {(d.recruiter_id, d.person_id) for d in result.dangling} == {(1, 99)}

    # LOAD-BEARING #1: every data row is byte-for-byte unchanged — the sweep
    # wrote nothing to either DB's persons / recruiter_candidates / recruiters.
    assert after == before
    # LOAD-BEARING #2: file mtimes are unchanged across the sweep window — the
    # read-only (mode=ro) connections cannot touch the file at all (not even the
    # schema-version meta re-stamp a store constructor would do). Structural
    # no-write proof.
    assert after_mtimes == before_mtimes
