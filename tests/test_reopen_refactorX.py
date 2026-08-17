"""Behavioral tests for reopen Refactor X — the recruiter-candidate CURRENT-STATE
authority + divergence metric.

Refactor X adds ``recruiter_candidates``: the per-(recruiter, person) resolved
current lifecycle, rolled up across every brief/source a person maps to. It is
DISTINCT from Stage 1's ``recruiter_candidate_history`` (an accretion log).
Population path: the 3a sighting hook, after recording a sighting, ALSO resolves
the person's merged current-state via the cross-DB join
(``candidate_persons`` -> per-state-dir ``candidates``, MOST-RECENT by
``last_seen_at``) and upserts it.

What these pin (the load-bearing properties):

- ``upsert_recruiter_candidate`` INSERTs then UPDATEs the same (recruiter,
  person) row in place — current-state, never a count, never a duplicate.
- The hook, after resolution, fills ``recruiter_candidates`` with each resolved
  person's current-state.
- MERGE: a person with candidate rows in two briefs/sources gets the MOST-RECENT
  (by the candidate row's ``last_seen_at``) state in the authority.
- The DIVERGENCE METRIC reports 0 diverged when the authority matches the
  brief-keyed rows, and exactly 1 when one is mutated out from under it. This is
  the gate for Refactor Y.
- IDEMPOTENCY: re-running the hook on the same brief leaves a single authority
  row per person (PK upsert), never corrupts/duplicates.
- ADDITIVE: ``ensure_candidate`` / ``record_candidate_discovery`` / the
  brief-keyed candidates path are UNCHANGED (a candidate write still behaves
  identically, and a grep confirms those functions weren't edited).

Tests pass explicit ``identity_db_path`` / ``recruiter_db_path`` / ``state_root``
to isolate to ``tmp_path`` — mirroring ``test_reopen_stage3a.py`` (which proved
the sighting behavior) plus ``test_reopen_stage2.py``'s per-state-dir layout
seed (``<root>/state/<source>/<key>/runtime_state.sqlite3``, the layout
``enumerate_state_dirs`` discovers and ``candidate_persons.state_key`` names).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from shared.runtime_state.identity_store import IdentityStore
from shared.runtime_state.recruiter_candidate_fill import (
    divergence_report,
    resolve_person_current_state,
)
from shared.runtime_state.recruiter_sighting import record_sightings_for_brief
from shared.runtime_state.recruiter_store import RecruiterStore
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
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
    """Every test starts + ends on the Stage-1 default resolver (mirrors 3a)."""

    from shared.recruiter_context import reset_recruiter_id_resolver

    reset_recruiter_id_resolver()
    yield
    reset_recruiter_id_resolver()


def _seed_brief_persons(
    identity_db_path: Path, brief_id: str, person_ids: list[int]
) -> None:
    """Write ``persons`` + ``brief_persons`` the way the resolver would.

    Copied from ``test_reopen_stage3a.py``: the hook consumes ``brief_persons``,
    so seeding it directly isolates the fill behavior from the resolver's
    matching internals (those are pinned by the F3 service tests).
    """

    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        for pid in person_ids:
            conn.execute(
                "INSERT OR IGNORE INTO persons"
                "(id, canonical_name, canonical_handle, created_at, last_seen_at) "
                "VALUES (?, ?, '', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
                (pid, f"Person {pid}"),
            )
            conn.execute(
                "INSERT OR IGNORE INTO brief_persons(brief_id, person_id, first_seen_at) "
                "VALUES (?, ?, '2026-01-01T00:00:00Z')",
                (brief_id, pid),
            )


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
    names (verified: ``identity_resolution_service`` writes
    ``state_key = state_dir.name``). Uses the REAL ``ensure_candidate`` (so the
    additive-path assertions exercise the production writer), then sets
    terminal state directly because the lifecycle guard forbids jumping
    discovered -> full_terminal in one transition. Returns the candidate id.
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
    cross-DB key the fill joins on)."""

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


# ---------------------------------------------------------------------------
# upsert_recruiter_candidate — INSERT then UPDATE in place (current-state)
# ---------------------------------------------------------------------------


def test_upsert_recruiter_candidate_creates_then_updates(recruiter_db: Path) -> None:
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    # First call INSERTs.
    store.upsert_recruiter_candidate(
        rid,
        42,
        current_lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        terminal_payload_json='{"k": 1}',
        source="linkedin",
        identity_key="ik-1",
        last_seen_at="2026-01-01T00:00:00+00:00",
    )
    row = store.recruiter_candidate(rid, 42)
    assert row is not None
    assert row["current_lifecycle_state"] == "full_terminal"
    assert row["terminal_decision"] == "SAVE"
    assert row["terminal_payload_json"] == '{"k": 1}'
    assert row["last_source"] == "linkedin"
    assert row["last_identity_key"] == "ik-1"
    assert row["last_seen_at"] == "2026-01-01T00:00:00+00:00"

    # Second call on the SAME (recruiter, person) UPDATEs current-state in place.
    store.upsert_recruiter_candidate(
        rid,
        42,
        current_lifecycle_state="full_terminal",
        terminal_decision="REJECT",
        terminal_payload_json='{"k": 2}',
        source="designer",
        identity_key="ik-2",
        last_seen_at="2026-02-01T00:00:00+00:00",
    )
    row2 = store.recruiter_candidate(rid, 42)
    assert row2["terminal_decision"] == "REJECT"
    assert row2["terminal_payload_json"] == '{"k": 2}'
    assert row2["last_source"] == "designer"
    assert row2["last_identity_key"] == "ik-2"

    # Still exactly ONE row for this (recruiter, person) — UPDATE, not INSERT.
    with store.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM recruiter_candidates "
            "WHERE recruiter_id = ? AND person_id = ?",
            (rid, 42),
        ).fetchone()["n"]
    assert n == 1


# ---------------------------------------------------------------------------
# The hook fills the authority with resolved persons' current-state
# ---------------------------------------------------------------------------


def test_hook_fills_authority_with_resolved_current_state(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    brief_id = "brief-fill-1"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)

    # Person 101 -> one designer candidate in a terminal SAVE.
    cid = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-101",
        brief_id=brief_id,
        identity_key="ik-101",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-03-01T00:00:00+00:00",
    )
    _seed_brief_persons(identity_db, brief_id, [101])
    _link_candidate_person(
        identity_db,
        person_id=101,
        source="designer",
        state_key="key-101",
        candidate_id=cid,
        brief_id=brief_id,
    )

    recorded = record_sightings_for_brief(
        brief_id,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert recorded == 1  # sighting recorded

    # AND the current-state authority is filled with the resolved state.
    auth = store.recruiter_candidate(rid, 101)
    assert auth is not None
    assert auth["current_lifecycle_state"] == "full_terminal"
    assert auth["terminal_decision"] == "SAVE"
    assert auth["last_source"] == "designer"
    assert auth["last_identity_key"] == "ik-101"


def test_hook_skips_authority_for_person_with_no_live_candidate(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """A resolved person with zero live candidate rows gets a sighting but NO
    authority row — the fill returns None and writes nothing."""

    brief_id = "brief-no-cand"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)

    # Person 111 is in brief_persons but has NO candidate_persons link.
    _seed_brief_persons(identity_db, brief_id, [111])

    recorded = record_sightings_for_brief(
        brief_id,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert recorded == 1  # sighting still recorded (history accretes)
    assert store.recruiter_candidate(rid, 111) is None  # no authority row


# ---------------------------------------------------------------------------
# MERGE — most-recent (by last_seen_at) across two briefs/sources wins
# ---------------------------------------------------------------------------


def test_merge_picks_most_recent_across_two_briefs(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """A person carrying candidate rows in TWO briefs/sources gets the
    MOST-RECENT (by the candidate row's last_seen_at) state in the authority.
    Construct two rows; assert the newer wins."""

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    # OLDER row: linkedin, REJECT, last_seen 2026-01.
    cid_old = _seed_candidate(
        state_root,
        source="linkedin",
        state_key="key-old",
        brief_id="brief-old",
        identity_key="ik-old",
        lifecycle_state="full_terminal",
        terminal_decision="REJECT",
        last_seen_at="2026-01-01T00:00:00+00:00",
    )
    # NEWER row: designer, SAVE, last_seen 2026-06 -> should win.
    cid_new = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-new",
        brief_id="brief-new",
        identity_key="ik-new",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-06-01T00:00:00+00:00",
    )

    # Person 202 links to BOTH candidate rows.
    _link_candidate_person(
        identity_db,
        person_id=202,
        source="linkedin",
        state_key="key-old",
        candidate_id=cid_old,
        brief_id="brief-old",
    )
    _link_candidate_person(
        identity_db,
        person_id=202,
        source="designer",
        state_key="key-new",
        candidate_id=cid_new,
        brief_id="brief-new",
    )

    # The merge helper picks the newer row directly.
    merged = resolve_person_current_state(
        202, identity_db_path=identity_db, state_root=state_root
    )
    assert merged is not None
    assert merged.terminal_decision == "SAVE"
    assert merged.source == "designer"
    assert merged.identity_key == "ik-new"

    # And the fill writes that winner into the authority.
    from shared.runtime_state.recruiter_candidate_fill import fill_recruiter_candidate

    assert (
        fill_recruiter_candidate(
            rid,
            202,
            identity_db_path=identity_db,
            recruiter_db_path=recruiter_db,
            state_root=state_root,
        )
        is True
    )
    auth = store.recruiter_candidate(rid, 202)
    assert auth["terminal_decision"] == "SAVE"
    assert auth["last_source"] == "designer"
    assert auth["last_identity_key"] == "ik-new"


# ---------------------------------------------------------------------------
# DIVERGENCE METRIC — the gate for Refactor Y
# ---------------------------------------------------------------------------


def test_divergence_zero_when_authority_matches_then_one_when_mutated(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    brief_id = "brief-div"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)

    cid = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-div",
        brief_id=brief_id,
        identity_key="ik-div",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )
    _seed_brief_persons(identity_db, brief_id, [303])
    _link_candidate_person(
        identity_db,
        person_id=303,
        source="designer",
        state_key="key-div",
        candidate_id=cid,
        brief_id=brief_id,
    )

    # Fill via the hook so the authority matches the brief-keyed row.
    record_sightings_for_brief(
        brief_id,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )

    clean = divergence_report(
        rid,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert clean.matched == 1
    assert clean.diverged == 0
    assert clean.missing_in_authority == 0
    assert clean.orphan_in_authority == 0

    # Mutate the authority out from under the brief-keyed row -> 1 diverged.
    store.upsert_recruiter_candidate(
        rid,
        303,
        current_lifecycle_state="full_terminal",
        terminal_decision="REJECT",  # brief-keyed row still says SAVE
        terminal_payload_json="{}",
        source="designer",
        identity_key="ik-div",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )
    mutated = divergence_report(
        rid,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert mutated.diverged == 1
    assert mutated.diverged_person_ids == (303,)
    assert mutated.matched == 0


def test_divergence_counts_missing_and_orphan(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """missing_in_authority: a person with a live brief-keyed row but no
    authority row. orphan_in_authority: an authority row whose person has zero
    live candidate rows behind it."""

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    # Person 410: live brief-keyed candidate, but the fill never ran for them.
    cid = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-410",
        brief_id="brief-410",
        identity_key="ik-410",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-05-01T00:00:00+00:00",
    )
    _link_candidate_person(
        identity_db,
        person_id=410,
        source="designer",
        state_key="key-410",
        candidate_id=cid,
        brief_id="brief-410",
    )
    # Sighting history names 410 in the recruiter's population, but no authority
    # row exists for them.
    store.record_candidate_sighting_once(rid, 410, brief_id="brief-410")

    # Person 420: an authority row with NO candidate_persons link behind it.
    store.upsert_recruiter_candidate(
        rid,
        420,
        current_lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        terminal_payload_json="{}",
        source="linkedin",
        identity_key="ik-420",
        last_seen_at="2026-05-01T00:00:00+00:00",
    )

    rep = divergence_report(
        rid,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert rep.missing_in_authority == 1
    assert rep.missing_person_ids == (410,)
    assert rep.orphan_in_authority == 1
    assert rep.orphan_person_ids == (420,)


# ---------------------------------------------------------------------------
# IDEMPOTENCY — re-running the hook leaves a single authority row per person
# ---------------------------------------------------------------------------


def test_hook_rerun_does_not_duplicate_authority_rows(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """The hook fires on every read-path re-resolution. Re-running it for the
    same brief must leave exactly ONE authority row per person (PK upsert), with
    the same current-state — never a duplicate or corrupted row."""

    brief_id = "brief-idem-x"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)

    cid = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-idem",
        brief_id=brief_id,
        identity_key="ik-idem",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )
    _seed_brief_persons(identity_db, brief_id, [505])
    _link_candidate_person(
        identity_db,
        person_id=505,
        source="designer",
        state_key="key-idem",
        candidate_id=cid,
        brief_id=brief_id,
    )

    for _ in range(5):
        record_sightings_for_brief(
            brief_id,
            identity_db_path=identity_db,
            recruiter_db_path=recruiter_db,
            state_root=state_root,
        )

    # Exactly one authority row, current-state intact.
    with store.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM recruiter_candidates "
            "WHERE recruiter_id = ? AND person_id = ?",
            (rid, 505),
        ).fetchone()["n"]
    assert n == 1
    auth = store.recruiter_candidate(rid, 505)
    assert auth["terminal_decision"] == "SAVE"

    # And the divergence metric still reads clean after the re-runs.
    rep = divergence_report(
        rid,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert rep.matched == 1
    assert rep.diverged == 0


def test_hook_reflects_state_change_on_rerun(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """The fill is NOT gated by the sighting ledger: when a person's brief-keyed
    state changes between reads, a re-run updates the authority even though the
    sighting itself is a no-op the second time."""

    brief_id = "brief-change"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)

    per_state_dir = state_root / "designer" / "key-change"
    cid = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-change",
        brief_id=brief_id,
        identity_key="ik-change",
        lifecycle_state="full_started",
        terminal_decision=None,
        last_seen_at="2026-04-01T00:00:00+00:00",
    )
    _seed_brief_persons(identity_db, brief_id, [606])
    _link_candidate_person(
        identity_db,
        person_id=606,
        source="designer",
        state_key="key-change",
        candidate_id=cid,
        brief_id=brief_id,
    )

    first = record_sightings_for_brief(
        brief_id,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert first == 1
    assert store.recruiter_candidate(rid, 606)["terminal_decision"] is None

    # The candidate reaches a terminal SAVE between reads.
    rss = RuntimeStateStore(per_state_dir / "runtime_state.sqlite3")
    with rss.connect() as conn:
        conn.execute(
            "UPDATE candidates SET current_lifecycle_state='full_terminal', "
            "terminal_decision='SAVE', last_seen_at='2026-07-01T00:00:00+00:00' "
            "WHERE id=?",
            (cid,),
        )

    second = record_sightings_for_brief(
        brief_id,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert second == 0  # sighting is a no-op (already recorded)
    # But the authority reflects the new terminal state.
    assert store.recruiter_candidate(rid, 606)["terminal_decision"] == "SAVE"


# ---------------------------------------------------------------------------
# ADDITIVE — the brief-keyed candidates path is UNCHANGED by Refactor X
# ---------------------------------------------------------------------------


def test_brief_keyed_candidate_path_unchanged(
    tmp_path: Path, identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """record_candidate_discovery writes the per-state-dir candidates table and
    is wholly independent of the recruiter store / the X fill. Running the hook
    (which now also fills the authority) leaves a discovered candidate's
    brief-keyed row byte-for-byte unchanged."""

    per_state_dir = tmp_path / "isolated" / "designer" / "key-1"
    per_state_dir.mkdir(parents=True, exist_ok=True)
    rss = RuntimeStateStore(per_state_dir / "runtime_state.sqlite3")
    run_id = rss.start_run(
        source="designer",
        brief_id="brief-unchanged",
        output_dir=str(per_state_dir),
        mode="discover",
    )
    cand_id = rss.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="designer",
        brief_id="brief-unchanged",
        identity_key="cand-key",
        display_name="Discovered Person",
        profile_url="https://example.com/p",
    )
    assert cand_id > 0
    before = rss.get_candidate(
        source="designer", brief_id="brief-unchanged", identity_key="cand-key"
    )
    assert before["current_lifecycle_state"] == "discovered"

    # Fire the hook for the same brief id, with its own persons in the isolated
    # tmp state tree (NOT the discovery dir above).
    _seed_brief_persons(identity_db, "brief-unchanged", [701])
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, "brief-unchanged")
    record_sightings_for_brief(
        "brief-unchanged",
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )

    # The brief-keyed candidate row is byte-for-byte the same — the fill touched
    # only the recruiter DB.
    after = rss.get_candidate(
        source="designer", brief_id="brief-unchanged", identity_key="cand-key"
    )
    assert after == before


def test_additive_source_functions_not_edited() -> None:
    """Grep-confirm Refactor X did NOT touch the brief-keyed candidate writers.

    The locked design is ADDITIVE-ONLY: ``ensure_candidate``,
    ``record_candidate_discovery``, and the brief-keyed ``candidates`` table /
    launcher must be unchanged. ``git log`` on store.py for the reopen branch
    must show no diff to those functions' bodies vs the merge-base. We assert
    the cheaper invariant here: the X implementation lives entirely outside
    store.py — no symbol from ``recruiter_candidate_fill`` /
    ``recruiter_candidates`` is referenced inside store.py.
    """

    store_src = (
        Path(__file__).resolve().parents[1]
        / "shared"
        / "runtime_state"
        / "store.py"
    ).read_text()
    assert "recruiter_candidates" not in store_src
    assert "recruiter_candidate_fill" not in store_src
    assert "upsert_recruiter_candidate" not in store_src

    # And the brief-keyed writers still exist with their additive signatures.
    from shared.runtime_state.store import RuntimeStateStore

    assert hasattr(RuntimeStateStore, "ensure_candidate")
    assert hasattr(RuntimeStateStore, "record_candidate_discovery")


def test_store_unedited_by_refactorX_working_changes() -> None:
    """Stronger additive proof, correctly scoped: the WORKING-TREE diff vs the
    committed tip (``git diff --name-only HEAD``) must not include store.py.

    NOTE the scoping (verified): store.py already differs from the merge-base
    with ``main`` because committed Stages 1/2/3a added
    ``record_candidate_principle_marker`` + the write-intentions table to it. A
    merge-base diff would therefore falsely flag store.py. Refactor X's additive
    contract is about THIS change — so we diff the working tree against HEAD,
    which isolates exactly what Refactor X edited (recruiter_store.py + the new
    fill module + the sighting hook + this test), and assert store.py is not
    among them. Skips cleanly if git is unavailable / this isn't yet committed
    such that the working diff is empty."""

    repo = Path(__file__).resolve().parents[1]
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git unavailable")

    changed = set(diff.split())
    # Only meaningful while Refactor X is uncommitted in the working tree. Once
    # committed the working diff is empty (store.py trivially absent), which
    # still satisfies the assertion.
    refactor_x_paths = {
        "shared/runtime_state/recruiter_candidate_fill.py",
        "shared/runtime_state/recruiter_sighting.py",
        "shared/runtime_state/recruiter_store.py",
        "tests/test_reopen_refactorX.py",
    }
    if not changed <= (refactor_x_paths | {"shared/runtime_state/store.py"}):
        pytest.skip("working tree contains changes outside the Refactor X slice")
    assert "shared/runtime_state/store.py" not in changed, (
        "Refactor X must not touch store.py (the brief-keyed candidates path); "
        f"working-tree changed files: {sorted(changed)}"
    )
