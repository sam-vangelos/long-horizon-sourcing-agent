"""Behavioral tests for reopen Refactor X.5 — the whole-population backfill of
the ``recruiter_candidates`` CURRENT-STATE authority for already-sighted persons.

Refactor X (``recruiter_candidate_fill.py`` + the Stage-3a hook) fills the
authority going FORWARD: every time a brief is re-resolved, each sighted person's
merged current-state is upserted. But persons sighted BEFORE Refactor X landed
have a ``recruiter_candidate_history`` accretion row and NO authority row. X.5 is
the one-shot backfill that closes that gap — it reuses the EXISTING per-person
primitive ``fill_recruiter_candidate`` (no new cross-DB join), enumerating the
population from ``recruiter_candidate_history`` (the sighting accretion) per
recruiter.

What these pin (the load-bearing properties):

- BACKFILL: a recruiter with N sighted persons, each with a live brief-keyed
  candidate row but NO ``recruiter_candidates`` row, ends with N authority rows
  AND ``divergence_report(recruiter).missing_in_authority == 0`` (the
  convergence proof — every sighted person now reconciles).
- LEGIT SKIP: a sighted person with NO live candidate row is skipped
  (``candidate_skipped_no_current_state += 1``), is NOT counted as
  missing-corruption (it lands in no divergence bucket), and does not crash.
- IDEMPOTENT: running the backfill TWICE leaves identical authority rows (PK
  upsert, no duplication) and does NOT touch ``recruiter_candidate_history``'s
  ``times_surfaced`` (X.5 writes ONLY ``recruiter_candidates``).
- ALL-RECRUITERS: ``recruiter_id=None`` processes every recruiter in the table
  (>1), each against its own population.
- ADDITIVE: ``ensure_candidate`` + the brief-keyed candidates path are unchanged
  (grep-confirm + a candidate write still behaves identically).

Tests pass explicit ``identity_db_path`` / ``recruiter_db_path`` / ``state_root``
to isolate to ``tmp_path`` — the seeding helpers (``_seed_candidate`` /
``_link_candidate_person``) are copied from ``test_reopen_refactorX.py`` so the
per-state-dir layout (``<root>/state/<source>/<key>/runtime_state.sqlite3``) and
the soft cross-DB ``candidate_persons`` key match exactly what the fill joins on.
The cold/sparse pre-X state is seeded directly via
``record_candidate_sighting_once`` (the same history primitive the hook uses),
WITHOUT ever firing the authority-fill hook.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.runtime_state.identity_store import IdentityStore
from shared.runtime_state.recruiter_candidate_fill import divergence_report
from shared.runtime_state.recruiter_store import RecruiterStore
from shared.runtime_state.store import RuntimeStateStore
from tools.backfill_recruiter_store import backfill_recruiter_candidates


# ---------------------------------------------------------------------------
# Fixtures / helpers (mirrored from test_reopen_refactorX.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def identity_db(tmp_path: Path) -> Path:
    return tmp_path / "_identity" / "identity.sqlite3"


@pytest.fixture()
def recruiter_db(tmp_path: Path) -> Path:
    return tmp_path / "_recruiter" / "recruiter.sqlite3"


@pytest.fixture()
def state_root(tmp_path: Path) -> Path:
    return tmp_path / "state"


@pytest.fixture(autouse=True)
def _reset_resolver() -> None:
    """Every test starts + ends on the Stage-1 default resolver (mirrors X)."""

    from shared.recruiter_context import reset_recruiter_id_resolver

    reset_recruiter_id_resolver()
    yield
    reset_recruiter_id_resolver()


def _seed_candidate(
    state_root: Path,
    *,
    source: str,
    state_key: str,
    brief_id: str,
    identity_key: str,
    lifecycle_state: str,
    terminal_decision: str | None,
    last_seen_at: str,
) -> int:
    """Create a per-state-dir candidate row in the canonical layout and pin its
    current-state + ``last_seen_at`` (the merge key).

    Layout is ``<state_root>/<source>/<state_key>/runtime_state.sqlite3`` — the
    one ``enumerate_state_dirs`` discovers and ``candidate_persons.state_key``
    names. Uses the REAL ``ensure_candidate`` (so the additive-path assertions
    exercise the production writer), then sets terminal state directly because
    the lifecycle guard forbids jumping discovered -> full_terminal in one
    transition. Returns the candidate id. Copied verbatim from
    ``test_reopen_refactorX.py``.
    """

    per_state_dir = state_root / source / state_key
    per_state_dir.mkdir(parents=True, exist_ok=True)
    store = RuntimeStateStore(per_state_dir / "runtime_state.sqlite3")
    cand_id = store.ensure_candidate(
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=f"Cand {identity_key}",
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE candidates SET current_lifecycle_state = ?, "
            "terminal_decision = ?, last_seen_at = ? WHERE id = ?",
            (lifecycle_state, terminal_decision, last_seen_at, cand_id),
        )
    return cand_id


def _link_candidate_person(
    identity_db_path: Path,
    *,
    person_id: int,
    source: str,
    state_key: str,
    candidate_id: int,
    brief_id: str,
) -> None:
    """Write a ``candidate_persons`` link the way the resolver does (the soft
    cross-DB key the fill joins on). Copied from ``test_reopen_refactorX.py``."""

    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO persons"
            "(id, canonical_name, canonical_handle, created_at, last_seen_at) "
            "VALUES (?, ?, '', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (person_id, f"Person {person_id}"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO candidate_persons"
            "(source, state_key, candidate_id, person_id, brief_id, link_kind, "
            "match_signal_json, recruiter_locked, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'auto_strong', '{}', 0, "
            "'2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (source, state_key, candidate_id, person_id, brief_id),
        )


def _seed_sighted_person_with_candidate(
    *,
    identity_db: Path,
    recruiter_db: Path,
    state_root: Path,
    recruiter_id: int,
    person_id: int,
    source: str,
    state_key: str,
    brief_id: str,
    identity_key: str,
    terminal_decision: str | None,
    last_seen_at: str,
) -> int:
    """Seed the cold/sparse pre-X state for one person: a live brief-keyed
    candidate row + a ``candidate_persons`` link + a sighting-history row, but
    NO ``recruiter_candidates`` authority row.

    ``record_candidate_sighting_once`` is the SAME history primitive the hook
    uses (it writes ``recruiter_candidate_history`` + the idempotency ledger),
    but it does NOT fill the authority — so this reproduces exactly the state a
    person sighted before Refactor X landed in. Returns the candidate id.
    """

    cid = _seed_candidate(
        state_root,
        source=source,
        state_key=state_key,
        brief_id=brief_id,
        identity_key=identity_key,
        lifecycle_state="full_terminal",
        terminal_decision=terminal_decision,
        last_seen_at=last_seen_at,
    )
    _link_candidate_person(
        identity_db,
        person_id=person_id,
        source=source,
        state_key=state_key,
        candidate_id=cid,
        brief_id=brief_id,
    )
    store = RecruiterStore(recruiter_db)
    store.record_candidate_sighting_once(recruiter_id, person_id, brief_id=brief_id)
    return cid


# ---------------------------------------------------------------------------
# BACKFILL — every sighted person ends with an authority row; missing == 0
# ---------------------------------------------------------------------------


def test_backfill_fills_authority_for_all_sighted_persons(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """A recruiter with N sighted persons, each with a live brief-keyed
    candidate row but NO authority row (the cold pre-X state). After the
    backfill: all N have authority rows AND
    divergence_report(recruiter).missing_in_authority == 0."""

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    persons = [101, 102, 103]
    for i, pid in enumerate(persons):
        _seed_sighted_person_with_candidate(
            identity_db=identity_db,
            recruiter_db=recruiter_db,
            state_root=state_root,
            recruiter_id=rid,
            person_id=pid,
            source="designer",
            state_key=f"key-{pid}",
            brief_id=f"brief-{pid}",
            identity_key=f"ik-{pid}",
            terminal_decision="SAVE" if i % 2 == 0 else "REJECT",
            last_seen_at=f"2026-0{i + 1}-01T00:00:00+00:00",
        )

    # Pre-state: every person has a sighting-history row, NONE has an authority.
    for pid in persons:
        assert store.candidate_history(rid, pid) is not None
        assert store.recruiter_candidate(rid, pid) is None

    result = backfill_recruiter_candidates(
        rid,
        recruiter_db_path=recruiter_db,
        identity_db_path=identity_db,
        state_root=state_root,
    )

    # Every sighted person now has an authority row.
    assert result.candidate_recruiters_processed == 1
    assert result.candidate_persons_seen == len(persons)
    assert result.candidate_authority_filled == len(persons)
    assert result.candidate_skipped_no_current_state == 0
    assert result.candidate_errors == 0
    for pid in persons:
        auth = store.recruiter_candidate(rid, pid)
        assert auth is not None, f"person {pid} should have an authority row"
        assert auth["current_lifecycle_state"] == "full_terminal"
        assert auth["last_source"] == "designer"

    # CONVERGENCE PROOF: missing_in_authority == 0 (every sighted person now
    # reconciles against the brief-keyed source of truth).
    assert len(result.candidate_divergence) == 1
    div = result.candidate_divergence[0]
    assert div["recruiter_id"] == rid
    assert div["missing_in_authority"] == 0, div
    assert div["matched"] == len(persons)
    assert div["diverged"] == 0
    assert div["orphan_in_authority"] == 0

    # And re-query the live metric directly (not just the cached result dict).
    live = divergence_report(
        rid,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert live.missing_in_authority == 0
    assert live.matched == len(persons)


# ---------------------------------------------------------------------------
# LEGIT SKIP — a sighted person with NO live candidate row is skipped cleanly
# ---------------------------------------------------------------------------


def test_backfill_skips_sighted_person_with_no_live_candidate(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """A person sighted (history row exists) but with NO candidate_persons link
    -> no live current-state. The backfill skips it
    (skipped_no_current_state += 1), it is NOT counted as missing-corruption
    (lands in no divergence bucket), and nothing crashes."""

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    # Person 201 has a live candidate row (will be filled).
    _seed_sighted_person_with_candidate(
        identity_db=identity_db,
        recruiter_db=recruiter_db,
        state_root=state_root,
        recruiter_id=rid,
        person_id=201,
        source="designer",
        state_key="key-201",
        brief_id="brief-201",
        identity_key="ik-201",
        terminal_decision="SAVE",
        last_seen_at="2026-05-01T00:00:00+00:00",
    )
    # Person 202 is SIGHTED but has NO candidate_persons link -> no live state.
    store.record_candidate_sighting_once(rid, 202, brief_id="brief-202")

    result = backfill_recruiter_candidates(
        rid,
        recruiter_db_path=recruiter_db,
        identity_db_path=identity_db,
        state_root=state_root,
    )

    assert result.candidate_persons_seen == 2
    assert result.candidate_authority_filled == 1  # only 201
    assert result.candidate_skipped_no_current_state == 1  # 202 skipped, not error
    assert result.candidate_errors == 0

    # 201 got an authority row; 202 did NOT.
    assert store.recruiter_candidate(rid, 201) is not None
    assert store.recruiter_candidate(rid, 202) is None

    # The skipped person is NOT missing-corruption: it has no live brief-keyed
    # state, so the divergence metric counts it in NO bucket.
    div = result.candidate_divergence[0]
    assert div["missing_in_authority"] == 0, div
    assert div["orphan_in_authority"] == 0, div
    assert div["matched"] == 1
    assert div["diverged"] == 0
    assert 202 not in div["missing_person_ids"]


# ---------------------------------------------------------------------------
# IDEMPOTENT — twice == once; never touches recruiter_candidate_history
# ---------------------------------------------------------------------------


def test_backfill_idempotent_no_duplication_history_untouched(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """Running the backfill TWICE leaves identical authority rows (PK upsert, no
    duplication) and does NOT touch recruiter_candidate_history.times_surfaced
    (X.5 writes ONLY recruiter_candidates)."""

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    persons = [301, 302]
    for pid in persons:
        _seed_sighted_person_with_candidate(
            identity_db=identity_db,
            recruiter_db=recruiter_db,
            state_root=state_root,
            recruiter_id=rid,
            person_id=pid,
            source="designer",
            state_key=f"key-{pid}",
            brief_id=f"brief-{pid}",
            identity_key=f"ik-{pid}",
            terminal_decision="SAVE",
            last_seen_at="2026-04-01T00:00:00+00:00",
        )

    # Snapshot times_surfaced BEFORE either backfill run.
    surfaced_before = {
        pid: store.candidate_history(rid, pid)["times_surfaced"] for pid in persons
    }
    assert all(v == 1 for v in surfaced_before.values())

    first = backfill_recruiter_candidates(
        rid,
        recruiter_db_path=recruiter_db,
        identity_db_path=identity_db,
        state_root=state_root,
    )
    second = backfill_recruiter_candidates(
        rid,
        recruiter_db_path=recruiter_db,
        identity_db_path=identity_db,
        state_root=state_root,
    )

    # Both runs fill the same N (idempotent — the second is all upserts, still
    # counted as filled, never skipped or errored).
    assert first.candidate_authority_filled == len(persons)
    assert second.candidate_authority_filled == len(persons)
    assert second.candidate_errors == 0

    # Exactly ONE authority row per (recruiter, person) — no duplication.
    with store.connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM recruiter_candidates WHERE recruiter_id = ?",
            (rid,),
        ).fetchone()["n"]
    assert total == len(persons)

    # times_surfaced is UNTOUCHED by the candidate backfill (X.5 only writes
    # recruiter_candidates; the accretion log is the hook's, not this arm's).
    for pid in persons:
        assert store.candidate_history(rid, pid)["times_surfaced"] == surfaced_before[pid]

    # And the convergence proof still reads clean after the re-run.
    assert second.candidate_divergence[0]["missing_in_authority"] == 0
    assert second.candidate_divergence[0]["matched"] == len(persons)


# ---------------------------------------------------------------------------
# ALL-RECRUITERS — recruiter_id=None processes every recruiter in the table
# ---------------------------------------------------------------------------


def test_backfill_all_recruiters_mode(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """recruiter_id=None processes >1 recruiter, each against its OWN sighting
    population (recruiter_candidate_history is recruiter-scoped)."""

    store = RecruiterStore(recruiter_db)
    rid_a = store.upsert_recruiter("a@b.com")
    rid_b = store.upsert_recruiter("b@b.com")
    assert rid_a != rid_b

    # Recruiter A sighted person 401; recruiter B sighted persons 501 + 502.
    _seed_sighted_person_with_candidate(
        identity_db=identity_db,
        recruiter_db=recruiter_db,
        state_root=state_root,
        recruiter_id=rid_a,
        person_id=401,
        source="designer",
        state_key="key-401",
        brief_id="brief-401",
        identity_key="ik-401",
        terminal_decision="SAVE",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )
    for pid in (501, 502):
        _seed_sighted_person_with_candidate(
            identity_db=identity_db,
            recruiter_db=recruiter_db,
            state_root=state_root,
            recruiter_id=rid_b,
            person_id=pid,
            source="linkedin",
            state_key=f"key-{pid}",
            brief_id=f"brief-{pid}",
            identity_key=f"ik-{pid}",
            terminal_decision="REJECT",
            last_seen_at="2026-04-01T00:00:00+00:00",
        )

    # None -> all recruiters.
    result = backfill_recruiter_candidates(
        None,
        recruiter_db_path=recruiter_db,
        identity_db_path=identity_db,
        state_root=state_root,
    )

    assert result.candidate_recruiters_processed == 2
    assert result.candidate_persons_seen == 3  # 1 for A, 2 for B
    assert result.candidate_authority_filled == 3
    assert result.candidate_errors == 0

    # Each recruiter's own population is filled and converged.
    assert store.recruiter_candidate(rid_a, 401) is not None
    assert store.recruiter_candidate(rid_b, 501) is not None
    assert store.recruiter_candidate(rid_b, 502) is not None
    # No cross-contamination: A never got B's persons.
    assert store.recruiter_candidate(rid_a, 501) is None

    # One divergence report per recruiter, both clean.
    by_rid = {d["recruiter_id"]: d for d in result.candidate_divergence}
    assert set(by_rid) == {rid_a, rid_b}
    assert by_rid[rid_a]["missing_in_authority"] == 0
    assert by_rid[rid_a]["matched"] == 1
    assert by_rid[rid_b]["missing_in_authority"] == 0
    assert by_rid[rid_b]["matched"] == 2


# ---------------------------------------------------------------------------
# ADDITIVE — ensure_candidate + the brief-keyed candidates path are unchanged
# ---------------------------------------------------------------------------


def test_backfill_additive_brief_keyed_path_unchanged(
    tmp_path: Path, identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """A brief-keyed candidate write behaves identically before and after the
    X.5 backfill runs — the backfill touches ONLY recruiter_candidates, never
    the per-state-dir candidates table."""

    # Seed a sighted person so the backfill has something to do.
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    _seed_sighted_person_with_candidate(
        identity_db=identity_db,
        recruiter_db=recruiter_db,
        state_root=state_root,
        recruiter_id=rid,
        person_id=601,
        source="designer",
        state_key="key-601",
        brief_id="brief-601",
        identity_key="ik-601",
        terminal_decision="SAVE",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )

    # An INDEPENDENT brief-keyed candidate (its own isolated state dir).
    per_state_dir = tmp_path / "isolated" / "designer" / "key-iso"
    per_state_dir.mkdir(parents=True, exist_ok=True)
    rss = RuntimeStateStore(per_state_dir / "runtime_state.sqlite3")
    run_id = rss.start_run(
        source="designer",
        brief_id="brief-iso",
        output_dir=str(per_state_dir),
        mode="discover",
    )
    cand_id = rss.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="designer",
        brief_id="brief-iso",
        identity_key="iso-key",
        display_name="Isolated Person",
        profile_url="https://example.com/iso",
    )
    assert cand_id > 0
    before = rss.get_candidate(
        source="designer", brief_id="brief-iso", identity_key="iso-key"
    )
    assert before["current_lifecycle_state"] == "discovered"

    # Run the candidate backfill.
    backfill_recruiter_candidates(
        rid,
        recruiter_db_path=recruiter_db,
        identity_db_path=identity_db,
        state_root=state_root,
    )

    # The independent brief-keyed row is byte-for-byte unchanged.
    after = rss.get_candidate(
        source="designer", brief_id="brief-iso", identity_key="iso-key"
    )
    assert after == before


def test_backfill_does_not_reference_brief_keyed_writers_in_store() -> None:
    """Grep-confirm the additive contract at the store layer (same invariant the
    X test pins): the brief-keyed candidate writers live entirely outside the
    recruiter authority. ``store.py`` (the brief-keyed candidates path) carries
    no reference to recruiter_candidates / the fill / the upsert, so the X.5
    backfill — which only calls fill_recruiter_candidate + the recruiter store —
    cannot have perturbed it."""

    store_src = (
        Path(__file__).resolve().parents[1] / "shared" / "runtime_state" / "store.py"
    ).read_text()
    assert "recruiter_candidates" not in store_src
    assert "recruiter_candidate_fill" not in store_src
    assert "upsert_recruiter_candidate" not in store_src

    from shared.runtime_state.store import RuntimeStateStore

    assert hasattr(RuntimeStateStore, "ensure_candidate")
    assert hasattr(RuntimeStateStore, "record_candidate_discovery")
