"""Behavioral tests for reopen Stage 3a — the recruiter candidate-sighting hook.

Stage 3a connects the identity resolver to the durable recruiter primitive:
once ``resolve_persons_for_brief`` has resolved the persons on a brief, the
hook (``shared.runtime_state.recruiter_sighting.record_sightings_for_brief``)
records one sighting per person under the recruiter who owns the brief, into
``recruiter_candidate_history``.

What these pin (the load-bearing properties):

- ``RecruiterStore.recruiter_for_brief`` is the reverse of
  ``briefs_for_recruiter`` — the linked id, or ``None`` when unlinked.
- The hook, given a brief with N resolved persons, writes N sightings under
  the right recruiter_id.
- IDEMPOTENCY (the critical one): running the hook TWICE for the same brief
  does NOT inflate ``times_surfaced``. This proves the read-path re-resolution
  (the hook fires on every page load) can't double-count. A NEW brief still
  accretes — ``times_surfaced`` tracks DISTINCT briefs.
- Fallback: an unlinked brief resolves recruiter_id from
  ``get_current_recruiter_id`` and is LAZILY linked, so the next call resolves
  through the primary path.
- The brief-keyed candidate path (``record_candidate_discovery``) is UNCHANGED
  by Stage 3a.

Tests pass explicit ``identity_db_path`` / ``recruiter_db_path`` to isolate to
``tmp_path`` (mirrors how the F3 service tests pass explicit paths) — except the
fallback test, which deliberately exercises the ambient resolver and so patches
``RECRUITER_ROOT`` the way test_reopen_stage2.py does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.runtime_state.identity_store import IdentityStore
from shared.runtime_state.recruiter_sighting import record_sightings_for_brief
from shared.runtime_state.recruiter_store import RecruiterStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def identity_db(tmp_path: Path) -> Path:
    return tmp_path / "_identity" / "identity.sqlite3"


@pytest.fixture()
def recruiter_db(tmp_path: Path) -> Path:
    return tmp_path / "_recruiter" / "recruiter.sqlite3"


@pytest.fixture(autouse=True)
def _reset_resolver() -> None:
    """Every test starts + ends on the Stage-1 default resolver."""

    from shared.recruiter_context import reset_recruiter_id_resolver

    reset_recruiter_id_resolver()
    yield
    reset_recruiter_id_resolver()


def _seed_brief_persons(
    identity_db_path: Path, brief_id: str, person_ids: list[int]
) -> None:
    """Write ``brief_persons`` rows the way ``resolve_persons_for_brief`` would.

    The hook consumes ``brief_persons`` (per-brief membership, already deduped),
    so seeding it directly isolates the sighting/idempotency behavior from the
    resolver's matching internals — those are pinned by the F3 service tests.
    Each person_id needs a parent ``persons`` row (FK), so mint those first.
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


# ---------------------------------------------------------------------------
# recruiter_for_brief — the reverse lookup
# ---------------------------------------------------------------------------


def test_recruiter_for_brief_returns_linked_id(recruiter_db: Path) -> None:
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, "brief-1")
    assert store.recruiter_for_brief("brief-1") == rid


def test_recruiter_for_brief_none_when_unlinked(recruiter_db: Path) -> None:
    store = RecruiterStore(recruiter_db)
    store.upsert_recruiter("a@b.com")  # exists, but no brief link
    assert store.recruiter_for_brief("never-linked") is None


def test_recruiter_for_brief_is_inverse_of_briefs_for_recruiter(
    recruiter_db: Path,
) -> None:
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, "brief-a")
    store.link_brief(rid, "brief-b")
    for b in store.briefs_for_recruiter(rid):
        assert store.recruiter_for_brief(b) == rid


# ---------------------------------------------------------------------------
# The sighting hook — N persons → N sightings under the owning recruiter
# ---------------------------------------------------------------------------


def test_hook_writes_one_sighting_per_resolved_person(
    identity_db: Path, recruiter_db: Path
) -> None:
    brief_id = "brief-sight-1"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)

    _seed_brief_persons(identity_db, brief_id, [101, 102, 103])

    recorded = record_sightings_for_brief(
        brief_id, identity_db_path=identity_db, recruiter_db_path=recruiter_db
    )
    assert recorded == 3

    # Each person has exactly one sighting under the owning recruiter, on this
    # brief, with first==last==this brief and the default empty lifecycle.
    for pid in (101, 102, 103):
        h = store.candidate_history(rid, pid)
        assert h is not None
        assert h["times_surfaced"] == 1
        assert h["first_seen_brief"] == brief_id
        assert h["last_seen_brief"] == brief_id
        assert h["last_lifecycle_state"] == ""


def test_hook_no_persons_is_noop(identity_db: Path, recruiter_db: Path) -> None:
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, "empty-brief")
    # No brief_persons seeded → nothing to sight.
    assert (
        record_sightings_for_brief(
            "empty-brief", identity_db_path=identity_db, recruiter_db_path=recruiter_db
        )
        == 0
    )


# ---------------------------------------------------------------------------
# IDEMPOTENCY — the critical property
# ---------------------------------------------------------------------------


def test_hook_twice_same_brief_does_not_inflate_times_surfaced(
    identity_db: Path, recruiter_db: Path
) -> None:
    """The load-bearing test: the hook fires on EVERY read-path re-resolution.
    Running it twice (simulating two page loads) must NOT double-count."""

    brief_id = "brief-idem"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)
    _seed_brief_persons(identity_db, brief_id, [201, 202])

    first = record_sightings_for_brief(
        brief_id, identity_db_path=identity_db, recruiter_db_path=recruiter_db
    )
    second = record_sightings_for_brief(
        brief_id, identity_db_path=identity_db, recruiter_db_path=recruiter_db
    )
    assert first == 2  # both new on first pass
    assert second == 0  # nothing new on re-resolution

    # times_surfaced is STILL 1 per person — not inflated by the second call.
    for pid in (201, 202):
        assert store.candidate_history(rid, pid)["times_surfaced"] == 1


def test_hook_many_reruns_keep_times_surfaced_at_one(
    identity_db: Path, recruiter_db: Path
) -> None:
    """Hammer the read path: ten re-resolutions, still one sighting."""

    brief_id = "brief-hammer"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)
    _seed_brief_persons(identity_db, brief_id, [301])

    for _ in range(10):
        record_sightings_for_brief(
            brief_id, identity_db_path=identity_db, recruiter_db_path=recruiter_db
        )
    assert store.candidate_history(rid, 301)["times_surfaced"] == 1


def test_same_person_on_new_brief_accretes(
    identity_db: Path, recruiter_db: Path
) -> None:
    """A DISTINCT brief is a real new sighting — times_surfaced must climb to 2.
    This is the flip side of idempotency: the guard suppresses re-runs of the
    SAME brief, never legitimate cross-brief accretion."""

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, "brief-x")
    store.link_brief(rid, "brief-y")

    _seed_brief_persons(identity_db, "brief-x", [401])
    _seed_brief_persons(identity_db, "brief-y", [401])  # same person, new brief

    # Re-resolve brief-x a few times (no inflation), then brief-y once.
    record_sightings_for_brief(
        "brief-x", identity_db_path=identity_db, recruiter_db_path=recruiter_db
    )
    record_sightings_for_brief(
        "brief-x", identity_db_path=identity_db, recruiter_db_path=recruiter_db
    )
    new_on_y = record_sightings_for_brief(
        "brief-y", identity_db_path=identity_db, recruiter_db_path=recruiter_db
    )
    assert new_on_y == 1

    h = store.candidate_history(rid, 401)
    assert h["times_surfaced"] == 2  # two DISTINCT briefs
    assert h["first_seen_brief"] == "brief-x"  # preserved
    assert h["last_seen_brief"] == "brief-y"  # updated


def test_record_candidate_sighting_once_returns_true_then_false(
    recruiter_db: Path,
) -> None:
    """Unit-level proof of the ledger gate on the store method itself."""

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    assert store.record_candidate_sighting_once(rid, 9, brief_id="b1") is True
    assert store.record_candidate_sighting_once(rid, 9, brief_id="b1") is False
    assert store.candidate_history(rid, 9)["times_surfaced"] == 1
    # A different brief is a fresh triple → True again, accretes.
    assert store.record_candidate_sighting_once(rid, 9, brief_id="b2") is True
    assert store.candidate_history(rid, 9)["times_surfaced"] == 2


# ---------------------------------------------------------------------------
# Fallback — unlinked brief uses the ambient resolver + lazy link
# ---------------------------------------------------------------------------


def test_unlinked_brief_falls_back_to_current_recruiter_and_lazily_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A brief with resolved persons but no recruiter_briefs link must resolve
    the recruiter from get_current_recruiter_id() and lazily link the brief so
    the next call resolves through the primary (reverse-lookup) path."""

    # The ambient resolver resolves against the LIVE RECRUITER_ROOT, so point
    # it at tmp and let the hook resolve recruiter_db via the same helper (pass
    # no explicit recruiter_db_path) so both land in one DB.
    recruiter_root = tmp_path / "state" / "_recruiter"
    recruiter_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("shared.output_paths.RECRUITER_ROOT", recruiter_root)
    recruiter_db = recruiter_root / "recruiter.sqlite3"
    identity_db = tmp_path / "_identity" / "identity.sqlite3"

    brief_id = "unlinked-brief"
    _seed_brief_persons(identity_db, brief_id, [501, 502])

    # No link_brief — the brief is unknown to recruiter_briefs.
    store = RecruiterStore(recruiter_db)
    assert store.recruiter_for_brief(brief_id) is None

    recorded = record_sightings_for_brief(brief_id, identity_db_path=identity_db)
    assert recorded == 2

    # Fallback resolved the Stage-1 implicit recruiter (id 1, bootstrapped by
    # the default resolver) and the sightings landed under it.
    from shared.recruiter_context import STAGE1_RECRUITER_ID

    assert store.candidate_history(STAGE1_RECRUITER_ID, 501)["times_surfaced"] == 1
    assert store.candidate_history(STAGE1_RECRUITER_ID, 502)["times_surfaced"] == 1

    # And the brief is now LAZILY linked, so the reverse lookup resolves it
    # directly on the next call (no fallback path needed again).
    assert store.recruiter_for_brief(brief_id) == STAGE1_RECRUITER_ID

    # Re-running through the now-primary path is still idempotent.
    assert record_sightings_for_brief(brief_id, identity_db_path=identity_db) == 0
    assert store.candidate_history(STAGE1_RECRUITER_ID, 501)["times_surfaced"] == 1


# ---------------------------------------------------------------------------
# The brief-keyed candidate path is UNCHANGED by Stage 3a
# ---------------------------------------------------------------------------


def test_brief_keyed_candidate_path_unchanged(
    tmp_path: Path, identity_db: Path, recruiter_db: Path
) -> None:
    """record_candidate_discovery writes the per-state-dir candidates table and
    is wholly independent of the recruiter store. Stage 3a adds no schema change
    and no call into it — running the sighting hook leaves a discovered
    candidate's brief-keyed row untouched."""

    from shared.runtime_state.store import RuntimeStateStore

    per_state_dir = tmp_path / "state" / "designer" / "key-1"
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

    # Fire the sighting hook for an unrelated brief (with its own persons).
    _seed_brief_persons(identity_db, "brief-unchanged", [601])
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, "brief-unchanged")
    record_sightings_for_brief(
        "brief-unchanged",
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
    )

    # The brief-keyed candidate row is byte-for-byte the same as before — the
    # sighting accretion touched only the recruiter DB.
    after = rss.get_candidate(
        source="designer", brief_id="brief-unchanged", identity_key="cand-key"
    )
    assert after == before
