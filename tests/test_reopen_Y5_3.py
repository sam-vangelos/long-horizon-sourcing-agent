"""Behavioral tests for reopen Y.5.3 — re-sync the recruiter current-state
authority after a brief-keyed candidate mutation.

Refactor X built ``recruiter_candidates`` (the per-(recruiter, person) current-
state authority) and fills it on every read-path re-resolution. Y.5.3 closes the
WRITE-path gap: the three candidate-mutation handlers in ``cloris.api._monolith``
(``append_candidate_note`` / ``set_candidate_user_status`` /
``set_candidate_judgment_accuracy``) now fire
``recruiter_mutation_sync.sync_candidate_mutation`` immediately after the
brief-keyed setter succeeds, so the authority follows a direct recruiter mutation
without waiting for the next read.

What these pin (the load-bearing properties):

- RE-DERIVE: ``sync_candidate_mutation`` resolves
  ``(source, state_key, candidate_id) -> person_id -> recruiter_id`` and re-runs
  the Refactor-X fill, so the authority row reflects the (now-mutated) candidate
  current-state. Returns ``True``.
- NO PERSON LINK: a candidate the resolver hasn't merged into a person returns a
  clean ``False`` — no crash, no authority row written.
- FAIL-SOFT (load-bearing): with ``fill_recruiter_candidate`` monkeypatched to
  RAISE inside the HANDLER path, the mutation endpoint STILL returns its normal
  success — the sync failure is swallowed + debug-logged. A broken authority can
  never break a recruiter's mutation.
- ADDITIVE: the per-state-dir ``store.py`` setters are unchanged (grep guard),
  and the mutation endpoints behave identically when the sync is a no-op (no
  person link).

The direct-call tests reuse the Refactor-X seeders (``_seed_candidate`` /
``_link_candidate_person`` / ``_seed_brief_persons``) which build the canonical
``<state_root>/<source>/<state_key>/runtime_state.sqlite3`` layout
``enumerate_state_dirs`` discovers and ``candidate_persons.state_key`` names. The
endpoint tests mirror ``test_cloris_candidate_actions.py``'s seeding (a real run
under ``brief-1`` so ``_find_state_dir_for_candidate`` matches) and redirect the
three global roots (``STATE_ROOT`` / ``IDENTITY_ROOT`` / ``RECRUITER_ROOT``) to
``tmp_path`` so the cross-DB sync inside the live handler is fully isolated.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from shared.runtime_state.identity_store import IdentityStore
from shared.runtime_state.recruiter_mutation_sync import sync_candidate_mutation
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
    """Every test starts + ends on the Stage-1 default resolver (mirrors X)."""

    from shared.recruiter_context import reset_recruiter_id_resolver

    reset_recruiter_id_resolver()
    yield
    reset_recruiter_id_resolver()


def _seed_brief_persons(
    identity_db_path: Path, brief_id: str, person_ids: list[int]
) -> None:
    """Write ``persons`` + ``brief_persons`` the way the resolver would (copied
    from ``test_reopen_refactorX.py``)."""

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
    current-state + ``last_seen_at`` (copied from ``test_reopen_refactorX.py``).

    Layout is ``<state_root>/<source>/<state_key>/runtime_state.sqlite3`` — the
    one ``enumerate_state_dirs`` discovers and ``candidate_persons.state_key``
    names. Returns the candidate id.
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
    cross-DB key the sync joins on; copied from ``test_reopen_refactorX.py``)."""

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
# RE-DERIVE — the sync re-fills the authority for the mutated candidate's person
# ---------------------------------------------------------------------------


def test_sync_re_derives_authority_for_linked_candidate(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """A candidate whose brief-keyed row the handler set -> calling
    ``sync_candidate_mutation`` directly re-derives the authority row for that
    person to the candidate's current-state. Returns ``True``."""

    brief_id = "brief-sync-1"
    source = "designer"
    state_key = "key-sync-1"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)

    cid = _seed_candidate(
        state_root,
        source=source,
        state_key=state_key,
        brief_id=brief_id,
        identity_key="ik-sync-1",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-03-01T00:00:00+00:00",
    )
    _seed_brief_persons(identity_db, brief_id, [801])
    _link_candidate_person(
        identity_db,
        person_id=801,
        source=source,
        state_key=state_key,
        candidate_id=cid,
        brief_id=brief_id,
    )

    # No authority row yet — the sync is what creates it.
    assert store.recruiter_candidate(rid, 801) is None

    state_dir = state_root / source / state_key
    synced = sync_candidate_mutation(
        brief_id,
        cid,
        state_dir,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert synced is True

    auth = store.recruiter_candidate(rid, 801)
    assert auth is not None
    assert auth["current_lifecycle_state"] == "full_terminal"
    assert auth["terminal_decision"] == "SAVE"
    assert auth["last_source"] == source
    assert auth["last_identity_key"] == "ik-sync-1"


def test_sync_reflects_a_post_fill_state_change(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """After an initial sync, mutating the candidate's current-state and
    re-syncing updates the authority in place (the re-derive is live, not
    one-shot) — the property a mutation handler depends on."""

    brief_id = "brief-sync-2"
    source = "designer"
    state_key = "key-sync-2"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)

    cid = _seed_candidate(
        state_root,
        source=source,
        state_key=state_key,
        brief_id=brief_id,
        identity_key="ik-sync-2",
        lifecycle_state="full_started",
        terminal_decision=None,
        last_seen_at="2026-03-01T00:00:00+00:00",
    )
    _link_candidate_person(
        identity_db,
        person_id=802,
        source=source,
        state_key=state_key,
        candidate_id=cid,
        brief_id=brief_id,
    )
    state_dir = state_root / source / state_key

    sync_candidate_mutation(
        brief_id,
        cid,
        state_dir,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert store.recruiter_candidate(rid, 802)["terminal_decision"] is None

    # The candidate reaches a terminal SAVE (as a real lifecycle transition
    # would, out from under the authority).
    rss = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    with rss.connect() as conn:
        conn.execute(
            "UPDATE candidates SET current_lifecycle_state='full_terminal', "
            "terminal_decision='SAVE', last_seen_at='2026-07-01T00:00:00+00:00' "
            "WHERE id=?",
            (cid,),
        )

    synced = sync_candidate_mutation(
        brief_id,
        cid,
        state_dir,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert synced is True
    auth = store.recruiter_candidate(rid, 802)
    assert auth["current_lifecycle_state"] == "full_terminal"
    assert auth["terminal_decision"] == "SAVE"
    # Still exactly one row — the re-sync is an upsert, never a duplicate.
    with store.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM recruiter_candidates "
            "WHERE recruiter_id = ? AND person_id = ?",
            (rid, 802),
        ).fetchone()["n"]
    assert n == 1


# ---------------------------------------------------------------------------
# NO PERSON LINK — a clean False, no crash, no authority row
# ---------------------------------------------------------------------------


def test_sync_returns_false_when_candidate_has_no_person_link(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """A candidate with NO ``candidate_persons`` link (the resolver never merged
    it) -> ``sync_candidate_mutation`` returns ``False``, writes no authority
    row, and does not raise."""

    brief_id = "brief-nolink"
    source = "designer"
    state_key = "key-nolink"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)

    cid = _seed_candidate(
        state_root,
        source=source,
        state_key=state_key,
        brief_id=brief_id,
        identity_key="ik-nolink",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-03-01T00:00:00+00:00",
    )
    # Deliberately NO _link_candidate_person — the candidate is unlinked.

    state_dir = state_root / source / state_key
    synced = sync_candidate_mutation(
        brief_id,
        cid,
        state_dir,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert synced is False
    # No authority row materialized for any person under this recruiter.
    with store.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM recruiter_candidates WHERE recruiter_id = ?",
            (rid,),
        ).fetchone()["n"]
    assert n == 0


def test_source_state_key_split_from_state_dir_path(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """The load-bearing split: ``(source, state_key)`` come from
    ``state_dir.parent.name`` / ``state_dir.name``. Seed the link under a
    DIFFERENT (source, state_key) than the state_dir encodes and confirm the
    sync does NOT find it (proves it keys off the path split, not candidate_id
    alone — which is unique only within a per-state-dir DB)."""

    brief_id = "brief-split"
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, brief_id)

    cid = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-split",
        brief_id=brief_id,
        identity_key="ik-split",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-03-01T00:00:00+00:00",
    )
    # Link the SAME candidate_id but under a different (source, state_key) —
    # a collision that only resolves if the sync ignored the path split.
    _link_candidate_person(
        identity_db,
        person_id=810,
        source="linkedin",
        state_key="other-key",
        candidate_id=cid,
        brief_id=brief_id,
    )

    state_dir = state_root / "designer" / "key-split"
    synced = sync_candidate_mutation(
        brief_id,
        cid,
        state_dir,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    # The (designer, key-split, cid) triple has no link -> clean False; the
    # mismatched (linkedin, other-key, cid) link is correctly NOT consulted.
    assert synced is False
    assert store.recruiter_candidate(rid, 810) is None


# ---------------------------------------------------------------------------
# Endpoint integration — the handler fires the sync (live, through the API)
# ---------------------------------------------------------------------------


def _seed_endpoint_candidate(tmp_path: Path) -> int:
    """Seed a real run + candidate under ``brief-1`` in the canonical
    ``<tmp_path>/state/linkedin/key/`` layout, mirroring
    ``test_cloris_candidate_actions._seed_candidate`` (so
    ``_find_state_dir_for_candidate`` matches via the latest run's brief_id).
    Returns the candidate id."""

    state_dir = tmp_path / "state" / "linkedin" / "key"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-1",
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": "brief-1"},
    )
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
    )
    conn = sqlite3.connect(f"file:{store.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id FROM candidates WHERE identity_key='li-1'"
        ).fetchone()
        return int(row["id"])
    finally:
        conn.close()


def _redirect_global_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Point STATE_ROOT / IDENTITY_ROOT / RECRUITER_ROOT at tmp_path.

    The candidate-mutation handlers resolve the state_dir via the lazy
    ``STATE_ROOT`` (``state_dirs_for_brief_id`` / ``aggregate_candidate_detail``
    with ``state_root=None``), and the Y.5.3 sync resolves the identity +
    recruiter DBs via ``resolve_identity_db_path`` / ``resolve_recruiter_db_path``
    (which read the ``IDENTITY_ROOT`` / ``RECRUITER_ROOT`` module constants).
    Patching all three keeps the entire cross-DB sync inside the live handler
    isolated to tmp_path — mirrors ``test_cloris_candidate_actions``'s
    ``STATE_ROOT`` redirect, extended to the two recruiter-reopen DBs.
    """

    import shared.output_paths

    monkeypatch.setattr(shared.output_paths, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(
        shared.output_paths, "IDENTITY_ROOT", tmp_path / "state" / "_identity"
    )
    monkeypatch.setattr(
        shared.output_paths, "RECRUITER_ROOT", tmp_path / "state" / "_recruiter"
    )


def test_user_status_endpoint_fires_sync_and_fills_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: PATCH user_status on a candidate WITH a person link ->
    returns 200 AND the recruiter authority row is filled by the handler's
    Y.5.3 sync (the live write-path proof, not the direct-call one)."""

    from fastapi.testclient import TestClient

    from cloris.app import create_app

    candidate_id = _seed_endpoint_candidate(tmp_path)
    _redirect_global_roots(monkeypatch, tmp_path)

    # The candidate is linked to person 901 under the SAME (source, state_key)
    # the handler's state_dir encodes (linkedin/key), so the sync resolves it.
    identity_db = tmp_path / "state" / "_identity" / "identity.sqlite3"
    _link_candidate_person(
        identity_db,
        person_id=901,
        source="linkedin",
        state_key="key",
        candidate_id=candidate_id,
        brief_id="brief-1",
    )

    client = TestClient(create_app())
    response = client.patch(
        f"/api/candidate/brief-1/{candidate_id}",
        json={"user_status": "shortlist"},
    )
    assert response.status_code == 200
    assert response.json()["user_status"] == "shortlist"

    # The handler's sync filled the authority for person 901 under the Stage-1
    # implicit recruiter (id 1 — the default resolver bootstraps + lazy-links).
    from shared.recruiter_context import STAGE1_RECRUITER_ID

    recruiter_db = tmp_path / "state" / "_recruiter" / "recruiter.sqlite3"
    rstore = RecruiterStore(recruiter_db)
    auth = rstore.recruiter_candidate(STAGE1_RECRUITER_ID, 901)
    assert auth is not None
    assert auth["last_source"] == "linkedin"
    assert auth["last_identity_key"] == "li-1"


def test_user_status_endpoint_succeeds_when_sync_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FAIL-SOFT (load-bearing): with ``fill_recruiter_candidate`` monkeypatched
    to RAISE *inside the handler path*, the user_status PATCH STILL returns its
    normal 200 success. The sync failure is swallowed + debug-logged. This is
    the proof a broken authority can never break a recruiter mutation."""

    from fastapi.testclient import TestClient

    from cloris.app import create_app
    import shared.runtime_state.recruiter_candidate_fill as fill_mod

    candidate_id = _seed_endpoint_candidate(tmp_path)
    _redirect_global_roots(monkeypatch, tmp_path)

    # Link the candidate to a person so the sync gets PAST the no-link short
    # circuit and actually reaches fill_recruiter_candidate — otherwise the
    # raise would never fire and the test would prove nothing.
    identity_db = tmp_path / "state" / "_identity" / "identity.sqlite3"
    _link_candidate_person(
        identity_db,
        person_id=911,
        source="linkedin",
        state_key="key",
        candidate_id=candidate_id,
        brief_id="brief-1",
    )

    # sync_candidate_mutation does `from ...recruiter_candidate_fill import
    # fill_recruiter_candidate` at CALL time, so patching the module attribute
    # is what the handler-path call binds against.
    def _boom(*args: object, **kwargs: object) -> bool:
        raise RuntimeError("authority is broken")

    monkeypatch.setattr(fill_mod, "fill_recruiter_candidate", _boom)

    client = TestClient(create_app())
    response = client.patch(
        f"/api/candidate/brief-1/{candidate_id}",
        json={"user_status": "shortlist"},
    )

    # The brief-keyed mutation committed and the response is the normal 200 —
    # the sync raise was swallowed.
    assert response.status_code == 200
    assert response.json()["user_status"] == "shortlist"

    # And no authority row was written (the fill raised before upserting).
    from shared.recruiter_context import STAGE1_RECRUITER_ID

    recruiter_db = tmp_path / "state" / "_recruiter" / "recruiter.sqlite3"
    rstore = RecruiterStore(recruiter_db)
    assert rstore.recruiter_candidate(STAGE1_RECRUITER_ID, 911) is None


def test_note_and_judgment_endpoints_succeed_when_sync_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same fail-soft contract holds for the OTHER two mutation handlers —
    note (POST) and judgment_accuracy (PATCH). Both return their normal 200 even
    when ``fill_recruiter_candidate`` raises in the sync."""

    from fastapi.testclient import TestClient

    from cloris.app import create_app
    import shared.runtime_state.recruiter_candidate_fill as fill_mod

    candidate_id = _seed_endpoint_candidate(tmp_path)
    _redirect_global_roots(monkeypatch, tmp_path)

    identity_db = tmp_path / "state" / "_identity" / "identity.sqlite3"
    _link_candidate_person(
        identity_db,
        person_id=921,
        source="linkedin",
        state_key="key",
        candidate_id=candidate_id,
        brief_id="brief-1",
    )

    def _boom(*args: object, **kwargs: object) -> bool:
        raise RuntimeError("authority is broken")

    monkeypatch.setattr(fill_mod, "fill_recruiter_candidate", _boom)

    client = TestClient(create_app())

    note_resp = client.post(
        f"/api/candidate/brief-1/{candidate_id}/note",
        json={"body": "Reached out"},
    )
    assert note_resp.status_code == 200
    assert len(note_resp.json()["notes"]) == 1

    judg_resp = client.patch(
        f"/api/candidate/brief-1/{candidate_id}/judgment-accuracy",
        json={"judgment_accuracy": "useful"},
    )
    assert judg_resp.status_code == 200
    assert judg_resp.json()["judgment_accuracy"] == "useful"


def test_endpoint_behaves_identically_when_sync_is_a_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADDITIVE: with NO person link (the sync is a clean no-op), the
    user_status PATCH returns exactly the same 200 + body it always has, and no
    recruiter authority row is created. The Y.5.3 hook is invisible when there
    is nothing to sync."""

    from fastapi.testclient import TestClient

    from cloris.app import create_app

    candidate_id = _seed_endpoint_candidate(tmp_path)
    _redirect_global_roots(monkeypatch, tmp_path)
    # Deliberately NO candidate_persons link — the sync short-circuits to False.

    client = TestClient(create_app())
    response = client.patch(
        f"/api/candidate/brief-1/{candidate_id}",
        json={"user_status": "contacted"},
    )
    assert response.status_code == 200
    assert response.json()["user_status"] == "contacted"

    # GET reflects it (the brief-keyed write path is untouched by the no-op sync).
    get_resp = client.get(f"/api/candidate/brief-1/{candidate_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["user_status"] == "contacted"

    # No authority row anywhere — the no-op sync wrote nothing.
    recruiter_db = tmp_path / "state" / "_recruiter" / "recruiter.sqlite3"
    if recruiter_db.exists():
        rstore = RecruiterStore(recruiter_db)
        with rstore.connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM recruiter_candidates"
            ).fetchone()["n"]
        assert n == 0


# ---------------------------------------------------------------------------
# ADDITIVE — store.py setters unchanged by Y.5.3
# ---------------------------------------------------------------------------


def test_store_setters_unchanged_by_y5_3() -> None:
    """The locked design is ADDITIVE: only the new sync module + the 3 handler
    call sites change. The per-state-dir ``store.py`` setters
    (``append_candidate_note`` / ``set_candidate_user_status`` /
    ``set_candidate_judgment_accuracy``) must NOT reference the recruiter
    authority — the cross-DB sync lives entirely in the handler/sibling module,
    not on the store (candidate_id is not globally unique; the store setters
    carry no source/state_key and must not reach cross-DB)."""

    store_src = (
        Path(__file__).resolve().parents[1]
        / "shared"
        / "runtime_state"
        / "store.py"
    ).read_text()
    assert "recruiter_mutation_sync" not in store_src
    assert "sync_candidate_mutation" not in store_src
    assert "recruiter_candidates" not in store_src

    # The brief-keyed setters still exist with their additive signatures.
    assert hasattr(RuntimeStateStore, "append_candidate_note")
    assert hasattr(RuntimeStateStore, "set_candidate_user_status")
    assert hasattr(RuntimeStateStore, "set_candidate_judgment_accuracy")


def test_store_unedited_by_y5_3_working_changes() -> None:
    """Stronger additive proof, scoped to the working tree: the working-tree
    diff vs the committed tip (``git diff --name-only HEAD``) must not include
    store.py. Y.5.3 edits only the new sync module, ``_monolith.py``, and this
    test. Skips cleanly if git is unavailable / nothing is uncommitted."""

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
    y5_3_paths = {
        "cloris/api/_monolith.py",
        "shared/runtime_state/recruiter_mutation_sync.py",
        "tests/test_reopen_Y5_3.py",
    }
    if not changed <= (y5_3_paths | {"shared/runtime_state/store.py"}):
        pytest.skip("working tree contains changes outside the Y.5.3 slice")
    assert "shared/runtime_state/store.py" not in changed, (
        "Y.5.3 must not touch store.py (the brief-keyed candidate setters); "
        f"working-tree changed files: {sorted(changed)}"
    )
