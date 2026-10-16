"""F4 (Reopen Y.5.8) — post-commit recruiter-authority re-sync on merge / unlink.

Pins the F4 contract end-to-end against REAL identity + recruiter stores:

- MERGE re-sync touches EVERY affected recruiter (C2), not just one: for every
  recruiter holding a ``recruiter_candidates`` row for keep_id OR drop_id, the
  drop_id tombstone is DELETED and the survivor (keep_id) row is refreshed.
- The re-sync runs POST-COMMIT (C1): it reads a committed identity snapshot —
  drop_id is already gone from ``persons`` — so the F3 ``recruiter_persons_sweep``
  goes CLEAN after the merge (no dangling authority row survives).
- FAIL-SOFT: a re-sync that raises does NOT break the merge — the identity
  transaction is already committed (persons collapsed) and the failure is
  swallowed.
- keep_separate fires NO re-sync.
- UNLINK re-sync runs post-commit: both the old and the new person are refreshed
  (neither is tombstoned — unlink deletes no person row).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- seeding helpers (mirror test_identity_resolution_service.py) -----------


def _seed_candidate(
    state_root: Path,
    *,
    source: str,
    state_key: str,
    brief_id: str,
    identity_key: str,
    display_name: str,
    profile_url: str = "",
    terminal_payload: dict | None = None,
) -> int:
    from shared.runtime_state.store import RuntimeStateStore

    state_dir = state_root / source / state_key
    state_dir.mkdir(parents=True, exist_ok=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    candidate_id = store.ensure_candidate(
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=display_name,
        profile_url=profile_url,
        initial_state="full_terminal",
    )
    if terminal_payload is not None:
        with store.connect() as conn:
            conn.execute(
                "UPDATE candidates SET terminal_payload_json = ? WHERE id = ?",
                (json.dumps(terminal_payload), candidate_id),
            )
    return candidate_id


@pytest.fixture()
def paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(state_root, identity_db_path, recruiter_db_path) under tmp_path."""

    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    identity_db = tmp_path / "_identity" / "identity.sqlite3"
    recruiter_db = tmp_path / "_recruiter" / "recruiter.sqlite3"
    return state_root, identity_db, recruiter_db


def _pending_pair(identity_db: Path, brief_id: str) -> tuple[int, int]:
    from shared.runtime_state.identity_store import IdentityStore

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        row = conn.execute(
            "SELECT person_a, person_b FROM pending_merge_decisions WHERE brief_id = ?",
            (brief_id,),
        ).fetchone()
    assert row is not None, "expected a pending merge decision to be created"
    return int(row["person_a"]), int(row["person_b"])


def _authority(recruiter_db: Path, recruiter_id: int, person_id: int) -> dict | None:
    from shared.runtime_state.recruiter_store import RecruiterStore

    return RecruiterStore(recruiter_db).recruiter_candidate(recruiter_id, person_id)


def _person_exists(identity_db: Path, person_id: int) -> bool:
    from shared.runtime_state.identity_store import IdentityStore

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM persons WHERE id = ?", (person_id,)
            ).fetchone()
            is not None
        )


def _seed_two_person_merge(paths) -> tuple[int, int, str]:
    """Seed two same-name LinkedIn candidates; resolve into persons a<b + a
    pending merge. Returns (keep_id, drop_id, brief_id) where keep<drop."""

    state_root, identity_db, _ = paths
    brief_id = "brief_f4_merge"
    _seed_candidate(
        state_root,
        source="linkedin",
        state_key="li_f4",
        brief_id=brief_id,
        identity_key="li-f4-1",
        display_name="Jordan Lee",
        profile_url="https://www.linkedin.com/in/jordan-lee-1/",
    )
    _seed_candidate(
        state_root,
        source="linkedin",
        state_key="li_f4",
        brief_id=brief_id,
        identity_key="li-f4-2",
        display_name="Jordan Lee",
        profile_url="https://www.linkedin.com/in/jordan-lee-2/",
    )

    from shared.identity_resolution_service import resolve_persons_for_brief

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )
    person_a, person_b = _pending_pair(identity_db, brief_id)
    keep_id, drop_id = sorted((person_a, person_b))
    return keep_id, drop_id, brief_id


def _seed_recruiter_with_tombstone_and_survivor(
    paths, *, handle: str, keep_id: int, drop_id: int, state_root_for_fill: Path
) -> int:
    """Create a recruiter and give it BOTH a survivor authority row (keep_id,
    via the real fill) and a tombstone authority row (drop_id, a stale roll-up).
    Returns the recruiter id."""

    _, identity_db, recruiter_db = paths
    from shared.runtime_state.recruiter_candidate_fill import fill_recruiter_candidate
    from shared.runtime_state.recruiter_store import RecruiterStore

    rid = RecruiterStore(recruiter_db).upsert_recruiter(handle)

    # Survivor row — real current-state from keep_id's live candidate link.
    wrote = fill_recruiter_candidate(
        rid,
        keep_id,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root_for_fill,
    )
    assert wrote, "keep_id should have a live candidate row to fill the survivor"

    # Tombstone row — drop_id has an authority row that, post-merge (drop_id
    # hard-deleted from persons), becomes a dangling soft reference the F3 sweep
    # flags. We seed it directly (a stale roll-up) so the merge re-sync has a
    # tombstone to delete.
    RecruiterStore(recruiter_db).upsert_recruiter_candidate(
        rid,
        drop_id,
        current_lifecycle_state="full_terminal",
        terminal_decision="reject",
        terminal_payload_json="{}",
        source="linkedin",
        identity_key="li-f4-2",
        last_seen_at=None,
    )
    return rid


# --- tests ------------------------------------------------------------------


def test_merge_resync_all_recruiters_deletes_tombstone_refreshes_survivor(paths):
    state_root, identity_db, recruiter_db = paths
    keep_id, drop_id, brief_id = _seed_two_person_merge(paths)

    rid_a = _seed_recruiter_with_tombstone_and_survivor(
        paths, handle="rec-a", keep_id=keep_id, drop_id=drop_id,
        state_root_for_fill=state_root,
    )
    rid_b = _seed_recruiter_with_tombstone_and_survivor(
        paths, handle="rec-b", keep_id=keep_id, drop_id=drop_id,
        state_root_for_fill=state_root,
    )

    # Pre-merge: BOTH recruiters hold a tombstone (drop_id) + survivor (keep_id).
    for rid in (rid_a, rid_b):
        assert _authority(recruiter_db, rid, drop_id) is not None
        assert _authority(recruiter_db, rid, keep_id) is not None

    from shared.identity_resolution_service import record_recruiter_merge

    record_recruiter_merge(
        brief_id=brief_id,
        person_a=keep_id,
        person_b=drop_id,
        decision="merge",
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )

    # Identity side: persons collapsed (drop gone, keep survives).
    assert not _person_exists(identity_db, drop_id)
    assert _person_exists(identity_db, keep_id)

    # C2: for EVERY affected recruiter, the tombstone is DELETED and the
    # survivor row is still present (refreshed).
    for rid in (rid_a, rid_b):
        assert _authority(recruiter_db, rid, drop_id) is None, (
            f"recruiter {rid}: drop_id tombstone must be deleted"
        )
        survivor = _authority(recruiter_db, rid, keep_id)
        assert survivor is not None, f"recruiter {rid}: survivor must survive"

    # The oracle: F3 sweep flags ZERO dangling authority rows post-merge.
    from shared.runtime_state.recruiter_persons_sweep import (
        sweep_recruiter_candidate_persons,
    )

    result = sweep_recruiter_candidate_persons(
        recruiter_db_path=recruiter_db, identity_db_path=identity_db
    )
    assert result.dangling == [], (
        f"F3 sweep must be CLEAN post-merge, got {result.to_dict()}"
    )


def test_merge_resync_failure_is_swallowed_merge_still_completes(paths, monkeypatch):
    state_root, identity_db, recruiter_db = paths
    keep_id, drop_id, brief_id = _seed_two_person_merge(paths)
    rid = _seed_recruiter_with_tombstone_and_survivor(
        paths, handle="rec-boom", keep_id=keep_id, drop_id=drop_id,
        state_root_for_fill=state_root,
    )

    # Make the survivor refresh raise — through the prod call path (the helper
    # imports fill_recruiter_candidate from this module at call time).
    import shared.runtime_state.recruiter_candidate_fill as fill_mod

    def _boom(*a, **k):
        raise RuntimeError("synthetic fill failure")

    monkeypatch.setattr(fill_mod, "fill_recruiter_candidate", _boom)

    from shared.identity_resolution_service import record_recruiter_merge

    # Must NOT raise — fail-soft swallows the re-sync failure.
    record_recruiter_merge(
        brief_id=brief_id,
        person_a=keep_id,
        person_b=drop_id,
        decision="merge",
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )

    # The identity transaction committed regardless: persons collapsed.
    assert not _person_exists(identity_db, drop_id)
    assert _person_exists(identity_db, keep_id)
    # The recruiter still exists (the store open / enumeration is unaffected).
    assert _authority(recruiter_db, rid, keep_id) is not None


def test_keep_separate_fires_no_resync(paths, monkeypatch):
    state_root, identity_db, recruiter_db = paths
    keep_id, drop_id, brief_id = _seed_two_person_merge(paths)
    _seed_recruiter_with_tombstone_and_survivor(
        paths, handle="rec-keep", keep_id=keep_id, drop_id=drop_id,
        state_root_for_fill=state_root,
    )

    called = {"resync": False}
    import shared.identity_resolution_service as svc

    real = svc._resync_recruiter_authority_after_identity_change

    def _spy(*a, **k):
        called["resync"] = True
        return real(*a, **k)

    monkeypatch.setattr(
        svc, "_resync_recruiter_authority_after_identity_change", _spy
    )

    svc.record_recruiter_merge(
        brief_id=brief_id,
        person_a=keep_id,
        person_b=drop_id,
        decision="keep_separate",
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )

    assert called["resync"] is False, "keep_separate must not fire the re-sync"
    # Both persons untouched; both authority rows untouched.
    assert _person_exists(identity_db, drop_id)
    assert _person_exists(identity_db, keep_id)


def test_unlink_resync_refreshes_old_and_new_person(paths, monkeypatch):
    state_root, identity_db, recruiter_db = paths
    brief_id = "brief_f4_unlink"

    cand_id = _seed_candidate(
        state_root,
        source="linkedin",
        state_key="li_f4_unlink",
        brief_id=brief_id,
        identity_key="li-f4-u",
        display_name="Unlink Person",
        profile_url="https://www.linkedin.com/in/unlink-person/",
    )

    from shared.identity_resolution_service import (
        record_recruiter_unlink,
        resolve_persons_for_brief,
    )
    from shared.runtime_state.identity_store import IdentityStore
    from shared.runtime_state.recruiter_store import RecruiterStore

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        old_person_id = int(
            conn.execute(
                "SELECT person_id FROM candidate_persons "
                "WHERE source='linkedin' AND state_key='li_f4_unlink' AND candidate_id=?",
                (cand_id,),
            ).fetchone()["person_id"]
        )

    # A recruiter with the OLD person's authority row (so the affected-recruiter
    # lookup finds it via old_person_id).
    rid = RecruiterStore(recruiter_db).upsert_recruiter("rec-unlink")
    from shared.runtime_state.recruiter_candidate_fill import fill_recruiter_candidate

    fill_recruiter_candidate(
        rid, old_person_id, identity_db_path=identity_db,
        recruiter_db_path=recruiter_db, state_root=state_root,
    )

    # Capture which persons the re-sync was asked to REFRESH.
    captured = {}
    import shared.identity_resolution_service as svc

    real = svc._resync_recruiter_authority_after_identity_change

    def _spy(*, refresh_ids, delete_ids, **k):
        captured["refresh"] = tuple(refresh_ids)
        captured["delete"] = tuple(delete_ids)
        return real(refresh_ids=refresh_ids, delete_ids=delete_ids, **k)

    monkeypatch.setattr(
        svc, "_resync_recruiter_authority_after_identity_change", _spy
    )

    record_recruiter_unlink(
        source="linkedin",
        state_key="li_f4_unlink",
        candidate_id=cand_id,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )

    # The candidate moved off old_person_id onto a fresh new_person_id.
    with store.connect() as conn:
        new_person_id = int(
            conn.execute(
                "SELECT person_id FROM candidate_persons "
                "WHERE source='linkedin' AND state_key='li_f4_unlink' AND candidate_id=?",
                (cand_id,),
            ).fetchone()["person_id"]
        )
    assert new_person_id != old_person_id

    # Both old + new person are REFRESHED; nothing is tombstoned (unlink deletes
    # no person row).
    assert set(captured["refresh"]) == {old_person_id, new_person_id}
    assert captured["delete"] == ()

    # The new person now has a live candidate → its authority row was created
    # under the affected recruiter; both persons still exist in identity.
    assert _person_exists(identity_db, old_person_id)
    assert _person_exists(identity_db, new_person_id)
    assert _authority(recruiter_db, rid, new_person_id) is not None

    # F3 sweep CLEAN (no person was deleted, so no dangling reference).
    from shared.runtime_state.recruiter_persons_sweep import (
        sweep_recruiter_candidate_persons,
    )

    result = sweep_recruiter_candidate_persons(
        recruiter_db_path=recruiter_db, identity_db_path=identity_db
    )
    assert result.dangling == [], (
        f"F3 sweep must be CLEAN post-unlink, got {result.to_dict()}"
    )


def test_delete_recruiter_candidate_is_idempotent(paths):
    """Unit: delete removes the row; deleting an absent row is a no-op."""

    _, _, recruiter_db = paths
    from shared.runtime_state.recruiter_store import RecruiterStore

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("rec-del")
    store.upsert_recruiter_candidate(
        rid, 4242,
        current_lifecycle_state="full_terminal",
        terminal_decision=None,
        terminal_payload_json="{}",
        source="linkedin",
        identity_key="k",
        last_seen_at=None,
    )
    assert store.recruiter_candidate(rid, 4242) is not None
    store.delete_recruiter_candidate(rid, 4242)
    assert store.recruiter_candidate(rid, 4242) is None
    # No-op on an already-absent row.
    store.delete_recruiter_candidate(rid, 4242)
    assert store.recruiter_candidate(rid, 4242) is None


def test_recruiter_ids_for_persons_distinct_union(paths):
    """Unit: the all-affected-recruiters read returns every recruiter with a
    row for ANY of the given persons, DISTINCT + ordered."""

    _, _, recruiter_db = paths
    from shared.runtime_state.recruiter_store import RecruiterStore

    store = RecruiterStore(recruiter_db)
    r1 = store.upsert_recruiter("rec-1")
    r2 = store.upsert_recruiter("rec-2")
    r3 = store.upsert_recruiter("rec-3")

    def _seed(rid, pid):
        store.upsert_recruiter_candidate(
            rid, pid,
            current_lifecycle_state="full_terminal",
            terminal_decision=None,
            terminal_payload_json="{}",
            source="linkedin",
            identity_key="k",
            last_seen_at=None,
        )

    _seed(r1, 100)  # r1 has keep
    _seed(r1, 200)  # r1 also has drop -> still distinct
    _seed(r2, 200)  # r2 has drop
    _seed(r3, 999)  # r3 has neither

    assert store.recruiter_ids_for_persons([100, 200]) == [r1, r2]
    assert store.recruiter_ids_for_persons([]) == []
    assert store.recruiter_ids_for_persons([999]) == [r3]
