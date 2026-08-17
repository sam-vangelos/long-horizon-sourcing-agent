"""Tests for the intake-session CRUD module + API endpoints (Slice 1B).

Per ``/Users/operator/.claude/plans/ancient-plotting-lemon.md`` Northwind, the
onboarding flow's authoring state lives in a dedicated ``intake_sessions``
SQLite table colocated with the canonical ``runtime_state.sqlite3``
schema. These tests exercise:

- the :mod:`cloris.intake_sessions` CRUD helpers directly against a
  ``RuntimeStateStore`` pointed at ``tmp_path``;
- the FastAPI router endpoints with ``cloris.api.intake`` store seams
  monkeypatched to the same tmp_path-backed store, so route shape + status
  codes get coverage without touching the real ``output/intake/`` tree;
- the schema-migration idempotency guarantee (running ``_migrate`` twice
  on a fresh DB produces exactly one ``intake_sessions`` table and no
  errors);
- the ``ConfigDict(extra="forbid")`` wire contract (POST with an unknown
  field returns HTTP 422).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cloris import intake_sessions as intake_module
from cloris.app import create_app
from shared.runtime_state.store import RuntimeStateStore


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    """Construct a RuntimeStateStore pointed at a tmp intake DB.

    Mirrors the production layout (``intake/intake_sessions.sqlite3``)
    inside tmp_path so any path-shape regressions surface here rather than
    polluting the real output tree.
    """

    return RuntimeStateStore(tmp_path / "intake" / "intake_sessions.sqlite3")


@pytest.fixture
def store(tmp_path: Path) -> RuntimeStateStore:
    """A fresh, migrated RuntimeStateStore for direct CRUD testing."""

    return _make_store(tmp_path)


@pytest.fixture
def api_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """A TestClient whose intake endpoints write to a tmp-path-backed store.

    Monkeypatches both seams:

    - ``cloris.api.intake._intake_store`` — used by POST/PATCH/DELETE
      handlers, returns the tmp-path-backed writer.
    - ``cloris.api.intake._intake_db_path`` — used by the read-only GET
      handlers (which route through ``read_models`` to avoid
      writer instantiation on read paths). Returns the same path the
      tmp_store writes to so both sides see the same DB.
    """

    tmp_db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    tmp_store = RuntimeStateStore(tmp_db_path)

    def fake_intake_store() -> RuntimeStateStore:
        return tmp_store

    def fake_intake_db_path() -> Path:
        return tmp_db_path

    monkeypatch.setattr("cloris.api.intake._intake_store", fake_intake_store)
    monkeypatch.setattr("cloris.api.intake._intake_db_path", fake_intake_db_path)

    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Direct CRUD coverage
# ---------------------------------------------------------------------------


def test_create_intake_session_returns_initial_shape(
    store: RuntimeStateStore,
) -> None:
    session = intake_module.create_intake_session(store, role_title=None)

    assert isinstance(session["id"], int)
    assert session["id"] > 0
    assert session["brief_id_draft"] is None
    assert session["role_title"] is None
    assert session["current_step"] == "welcome"
    assert session["state_json"] == {}
    assert session["started_at"]
    assert session["updated_at"] == session["started_at"]
    assert session["completed_at"] is None
    assert session["archived_at"] is None


def test_create_intake_session_accepts_role_title_hint(
    store: RuntimeStateStore,
) -> None:
    session = intake_module.create_intake_session(
        store, role_title="Head of AI Lab"
    )
    assert session["role_title"] == "Head of AI Lab"
    # Step is still welcome — the optional hint doesn't advance state.
    assert session["current_step"] == "welcome"


def test_list_intake_sessions_returns_newest_first_excluding_archived(
    store: RuntimeStateStore,
) -> None:
    """Active sessions only, ordered by updated_at DESC.

    We bump the second session's updated_at via a patch so the ordering
    contract is exercised even if the create timestamps tie at the
    sub-millisecond level. We then archive the third session via a raw
    SQL UPDATE to verify the WHERE archived_at IS NULL clause filters it
    out.
    """

    a = intake_module.create_intake_session(store, role_title="A")
    b = intake_module.create_intake_session(store, role_title="B")
    c = intake_module.create_intake_session(store, role_title="C")

    # Touch B last so it's newest by updated_at.
    intake_module.patch_intake_session(
        store, session_id=b["id"], current_step="role_basics"
    )

    # Archive C so list_intake_sessions excludes it.
    with store.connect() as conn:
        conn.execute(
            "UPDATE intake_sessions SET archived_at = ? WHERE id = ?",
            ("2026-04-28T00:00:00+00:00", c["id"]),
        )

    sessions = intake_module.list_intake_sessions(store)
    ids_in_order = [s["id"] for s in sessions]

    assert c["id"] not in ids_in_order, "archived session must be excluded"
    assert ids_in_order[0] == b["id"], "newest-first ordering by updated_at"
    assert ids_in_order[1] == a["id"]


def test_get_intake_session_returns_row_or_none(
    store: RuntimeStateStore,
) -> None:
    session = intake_module.create_intake_session(store)

    fetched = intake_module.get_intake_session(store, session_id=session["id"])
    assert fetched is not None
    assert fetched["id"] == session["id"]

    missing = intake_module.get_intake_session(store, session_id=999_999)
    assert missing is None


def test_patch_current_step_only_preserves_state_json(
    store: RuntimeStateStore,
) -> None:
    """Patching only ``current_step`` must not touch ``state_json``.

    Regression guard: the SQL builder must not include ``state_json = ?``
    in the SET list when the caller didn't provide one, or unrelated
    authoring state would silently get clobbered.
    """

    created = intake_module.create_intake_session(store)
    intake_module.patch_intake_session(
        store,
        session_id=created["id"],
        state_json={"role_basics": {"function": "Engineering"}},
    )

    patched = intake_module.patch_intake_session(
        store, session_id=created["id"], current_step="role_basics"
    )
    assert patched is not None
    assert patched["current_step"] == "role_basics"
    assert patched["state_json"] == {"role_basics": {"function": "Engineering"}}
    assert patched["updated_at"] >= created["updated_at"]


def test_patch_state_json_only_preserves_current_step(
    store: RuntimeStateStore,
) -> None:
    """Patching only ``state_json`` must not touch ``current_step``."""

    created = intake_module.create_intake_session(store)
    intake_module.patch_intake_session(
        store, session_id=created["id"], current_step="role_framing"
    )

    patched = intake_module.patch_intake_session(
        store,
        session_id=created["id"],
        state_json={"good_looks_like": "10x platform engineer"},
    )
    assert patched is not None
    assert patched["current_step"] == "role_framing"
    assert patched["state_json"] == {"good_looks_like": "10x platform engineer"}


def test_patch_both_fields_updates_both(
    store: RuntimeStateStore,
) -> None:
    created = intake_module.create_intake_session(store)
    patched = intake_module.patch_intake_session(
        store,
        session_id=created["id"],
        current_step="synthesis",
        state_json={"locked": True, "notes": ["seed1", "seed2"]},
        role_title="Staff ML Engineer",
    )
    assert patched is not None
    assert patched["current_step"] == "synthesis"
    assert patched["state_json"] == {"locked": True, "notes": ["seed1", "seed2"]}
    assert patched["role_title"] == "Staff ML Engineer"


def test_patch_missing_session_returns_none(
    store: RuntimeStateStore,
) -> None:
    result = intake_module.patch_intake_session(
        store, session_id=42_424_242, current_step="welcome"
    )
    assert result is None


def test_delete_intake_session_returns_true_on_hit_and_removes_row(
    store: RuntimeStateStore,
) -> None:
    session = intake_module.create_intake_session(store)
    assert intake_module.delete_intake_session(
        store, session_id=session["id"]
    ) is True
    assert intake_module.get_intake_session(
        store, session_id=session["id"]
    ) is None


def test_delete_missing_session_returns_false(
    store: RuntimeStateStore,
) -> None:
    assert intake_module.delete_intake_session(
        store, session_id=999_999
    ) is False


# ---------------------------------------------------------------------------
# Archive (P10 actuate #3): the archive endpoint (PATCH sets archived_at) —
# the WHERE archived_at IS NULL filter already existed (list_intake_sessions
# above); this fills in the only write path, previously reachable only via a
# raw SQL UPDATE (see test_list_intake_sessions_returns_newest_first_excluding_archived).
# ---------------------------------------------------------------------------


def test_archive_intake_session_sets_archived_at_and_excludes_from_listing(
    store: RuntimeStateStore,
) -> None:
    session = intake_module.create_intake_session(store, role_title="to-archive")
    assert session["archived_at"] is None

    archived = intake_module.archive_intake_session(store, session_id=session["id"])

    assert archived is not None
    assert archived["archived_at"] is not None
    # get_intake_session still returns archived rows (deep-link/resume).
    assert intake_module.get_intake_session(
        store, session_id=session["id"]
    )["archived_at"] == archived["archived_at"]
    # list_intake_sessions excludes it.
    ids = [s["id"] for s in intake_module.list_intake_sessions(store)]
    assert session["id"] not in ids


def test_archive_intake_session_is_idempotent_on_archived_at() -> None:
    """Re-archiving preserves the original archived_at stamp."""

    import time
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        store = _make_store(Path(td))
        session = intake_module.create_intake_session(store, role_title="x")

        first = intake_module.archive_intake_session(store, session_id=session["id"])
        time.sleep(0.01)
        second = intake_module.archive_intake_session(store, session_id=session["id"])

        assert first["archived_at"] == second["archived_at"]
        # updated_at still bumps on the no-op re-archive.
        assert second["updated_at"] >= first["updated_at"]


def test_archive_missing_session_returns_none(
    store: RuntimeStateStore,
) -> None:
    assert intake_module.archive_intake_session(store, session_id=999_999) is None


# ---------------------------------------------------------------------------
# Migration idempotency
# ---------------------------------------------------------------------------


def test_migrate_is_idempotent_for_intake_sessions(tmp_path: Path) -> None:
    """Running ``_migrate`` twice on a fresh DB must be a no-op.

    Constructs a store (which runs ``initialize`` → ``_migrate``), then
    re-invokes ``_migrate`` on a fresh connection. Verifies exactly one
    ``intake_sessions`` table exists and the index is present once.
    """

    store = _make_store(tmp_path)

    with store.connect() as conn:
        # First _migrate happened during __init__. Run it again on the
        # same DB — must not error.
        store._migrate(conn)

        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='intake_sessions'"
        ).fetchall()
        assert len(tables) == 1, "exactly one intake_sessions table"

        indexes = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_intake_sessions_active'"
        ).fetchall()
        assert len(indexes) == 1, "exactly one idx_intake_sessions_active index"

    # Sanity: the table is usable after re-migration.
    session = intake_module.create_intake_session(store)
    assert session["current_step"] == "welcome"


# ---------------------------------------------------------------------------
# API endpoint coverage
# ---------------------------------------------------------------------------


def test_post_create_returns_201_with_session_envelope(
    api_client: TestClient,
) -> None:
    response = api_client.post(
        "/api/intake/sessions",
        json={"role_title": "Director of Talent Strategy"},
    )
    assert response.status_code == 201

    body = response.json()
    assert body["slice"] == "v0-onboarding-slice-1"
    session = body["session"]
    assert session["role_title"] == "Director of Talent Strategy"
    assert session["current_step"] == "welcome"
    assert session["state_json"] == {}
    assert isinstance(session["id"], int)


def test_post_create_with_extra_field_returns_422(
    api_client: TestClient,
) -> None:
    """Slice 1B contract: ``ConfigDict(extra="forbid")`` rejects unknown fields.

    Mirrors the existing wire-discipline pattern from
    :class:`LaunchLinkedInRequest` so onboarding-flow callers can't smuggle
    state through the API by adding extra keys.
    """

    response = api_client.post(
        "/api/intake/sessions",
        json={"role_title": "X", "future_field_we_havent_added": True},
    )
    assert response.status_code == 422


def test_get_list_returns_active_sessions_envelope(
    api_client: TestClient,
) -> None:
    api_client.post("/api/intake/sessions", json={"role_title": "alpha"})
    api_client.post("/api/intake/sessions", json={"role_title": "beta"})

    response = api_client.get("/api/intake/sessions")
    assert response.status_code == 200
    body = response.json()
    assert body["slice"] == "v0-onboarding-slice-1"
    assert len(body["sessions"]) == 2
    titles = {s["role_title"] for s in body["sessions"]}
    assert titles == {"alpha", "beta"}


def test_get_one_returns_session(api_client: TestClient) -> None:
    created = api_client.post(
        "/api/intake/sessions", json={"role_title": "deep"}
    ).json()["session"]

    response = api_client.get(f"/api/intake/sessions/{created['id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["session"]["id"] == created["id"]


def test_get_one_missing_returns_404(api_client: TestClient) -> None:
    response = api_client.get("/api/intake/sessions/999999")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error"] == "intake_session_not_found"
    assert detail["id"] == 999999


def test_patch_updates_fields_and_returns_session(
    api_client: TestClient,
) -> None:
    created = api_client.post(
        "/api/intake/sessions", json={"role_title": "to-be-patched"}
    ).json()["session"]

    response = api_client.patch(
        f"/api/intake/sessions/{created['id']}",
        json={
            "current_step": "exemplars",
            "state_json": {"exemplars": ["alice", "bob"]},
        },
    )
    assert response.status_code == 200
    session = response.json()["session"]
    assert session["current_step"] == "exemplars"
    assert session["state_json"] == {"exemplars": ["alice", "bob"]}
    # Untouched field is preserved.
    assert session["role_title"] == "to-be-patched"


def test_patch_missing_returns_404(api_client: TestClient) -> None:
    response = api_client.patch(
        "/api/intake/sessions/424242",
        json={"current_step": "welcome"},
    )
    assert response.status_code == 404


def test_patch_extra_field_returns_422(api_client: TestClient) -> None:
    created = api_client.post(
        "/api/intake/sessions", json={"role_title": "x"}
    ).json()["session"]

    response = api_client.patch(
        f"/api/intake/sessions/{created['id']}",
        json={"unknown_field": "boom"},
    )
    assert response.status_code == 422


def test_delete_returns_envelope_and_removes_session(
    api_client: TestClient,
) -> None:
    created = api_client.post(
        "/api/intake/sessions", json={"role_title": "doomed"}
    ).json()["session"]

    response = api_client.delete(f"/api/intake/sessions/{created['id']}")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "slice": "v0-onboarding-slice-1",
        "deleted": True,
        "id": created["id"],
    }

    assert api_client.get(
        f"/api/intake/sessions/{created['id']}"
    ).status_code == 404


def test_delete_missing_returns_404(api_client: TestClient) -> None:
    response = api_client.delete("/api/intake/sessions/424242")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error"] == "intake_session_not_found"
    assert detail["id"] == 424242


def test_patch_archive_sets_archived_at_and_removes_from_active_list(
    api_client: TestClient,
) -> None:
    created = api_client.post(
        "/api/intake/sessions", json={"role_title": "archive-me"}
    ).json()["session"]
    assert created["archived_at"] is None

    response = api_client.patch(f"/api/intake/sessions/{created['id']}/archive")

    assert response.status_code == 200
    session = response.json()["session"]
    assert session["id"] == created["id"]
    assert session["archived_at"] is not None

    # GET-by-id still resolves (deep-link/resume), but the active list excludes it.
    assert api_client.get(f"/api/intake/sessions/{created['id']}").status_code == 200
    listed_ids = [s["id"] for s in api_client.get("/api/intake/sessions").json()["sessions"]]
    assert created["id"] not in listed_ids


def test_patch_archive_missing_returns_404(api_client: TestClient) -> None:
    response = api_client.patch("/api/intake/sessions/424242/archive")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error"] == "intake_session_not_found"
    assert detail["id"] == 424242


# ---------------------------------------------------------------------------
# Read-path (mode=ro) coverage for shared.runtime_state.read_models
# ---------------------------------------------------------------------------
#
# The GET endpoints route through ``read_models.list_intake_sessions`` /
# ``get_intake_session`` rather than the writer ``intake_module``, so the
# polled read path doesn't run DDL + ``INSERT OR REPLACE INTO meta`` on
# every call. These tests pin the read-path contract directly so a
# regression that swaps the read primitive back to the writer (or breaks
# the missing-DB / corrupt-DB collapse) surfaces independently of the
# endpoint plumbing covered above.


def test_read_models_list_intake_sessions_matches_writer_shape(
    tmp_path: Path,
) -> None:
    """Read path returns the same dict shape as the writer-side helper.

    Critical for the API endpoint: ``IntakeSession.model_validate(row)``
    consumes the dict from either path, so any drift between read and
    writer shape silently breaks the wire model.
    """

    from shared.runtime_state import read_models as rm

    db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    store = RuntimeStateStore(db_path)
    intake_module.create_intake_session(store, role_title="alpha")
    intake_module.create_intake_session(store, role_title="beta")

    via_writer = intake_module.list_intake_sessions(store)
    via_read = rm.list_intake_sessions(db_path)

    assert [s["role_title"] for s in via_read] == [
        s["role_title"] for s in via_writer
    ]
    assert {k for s in via_read for k in s.keys()} == {
        k for s in via_writer for k in s.keys()
    }


def test_read_models_list_intake_sessions_excludes_archived(
    tmp_path: Path,
) -> None:
    """``WHERE archived_at IS NULL`` enforced on the read path too."""

    from shared.runtime_state import read_models as rm

    db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    store = RuntimeStateStore(db_path)
    keeper = intake_module.create_intake_session(store, role_title="keeper")
    archived = intake_module.create_intake_session(
        store, role_title="archived"
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE intake_sessions SET archived_at = ? WHERE id = ?",
            ("2026-04-28T00:00:00+00:00", archived["id"]),
        )

    rows = rm.list_intake_sessions(db_path)
    ids = {s["id"] for s in rows}
    assert keeper["id"] in ids
    assert archived["id"] not in ids


def test_read_models_get_intake_session_returns_archived_rows(
    tmp_path: Path,
) -> None:
    """``get`` returns archived rows; ``list`` does not — same as writer."""

    from shared.runtime_state import read_models as rm

    db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    store = RuntimeStateStore(db_path)
    archived = intake_module.create_intake_session(
        store, role_title="resumable"
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE intake_sessions SET archived_at = ? WHERE id = ?",
            ("2026-04-28T00:00:00+00:00", archived["id"]),
        )

    fetched = rm.get_intake_session(db_path, session_id=archived["id"])
    assert fetched is not None
    assert fetched["id"] == archived["id"]
    assert fetched["archived_at"] is not None


def test_read_models_intake_collapses_missing_db(tmp_path: Path) -> None:
    """Missing DB yields empty list / None, not an exception."""

    from shared.runtime_state import read_models as rm

    missing = tmp_path / "no" / "such" / "intake.sqlite3"
    assert rm.list_intake_sessions(missing) == []
    assert rm.get_intake_session(missing, session_id=1) is None


def test_read_models_intake_get_unknown_session_returns_none(
    tmp_path: Path,
) -> None:
    from shared.runtime_state import read_models as rm

    db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    store = RuntimeStateStore(db_path)
    intake_module.create_intake_session(store, role_title="present")
    assert rm.get_intake_session(db_path, session_id=42_424_242) is None


def test_read_models_intake_state_json_parses_to_dict(tmp_path: Path) -> None:
    """``state_json`` round-trips via the read path the same way writer does."""

    from shared.runtime_state import read_models as rm

    db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    store = RuntimeStateStore(db_path)
    created = intake_module.create_intake_session(store, role_title="state")
    intake_module.patch_intake_session(
        store,
        session_id=created["id"],
        state_json={"role_basics": {"function": "Engineering"}},
    )

    fetched = rm.get_intake_session(db_path, session_id=created["id"])
    assert fetched is not None
    assert fetched["state_json"] == {
        "role_basics": {"function": "Engineering"}
    }


def test_read_models_intake_path_is_read_only(tmp_path: Path) -> None:
    """The read primitive must not run DDL or INSERT OR REPLACE INTO meta.

    Writer instantiation rewrites the ``schema_version`` row in ``meta``;
    the read path opens via ``mode=ro`` so a polled GET endpoint can't
    silently churn ``meta`` between writer-side updates. Capturing the
    ``meta`` row before/after the read pins this invariant.
    """

    from shared.runtime_state import read_models as rm

    db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    store = RuntimeStateStore(db_path)
    intake_module.create_intake_session(store, role_title="ro")

    with store.connect() as conn:
        before = conn.execute(
            "SELECT key, value FROM meta ORDER BY key"
        ).fetchall()

    rm.list_intake_sessions(db_path)
    rm.get_intake_session(db_path, session_id=1)

    with store.connect() as conn:
        after = conn.execute(
            "SELECT key, value FROM meta ORDER BY key"
        ).fetchall()

    # Tuple-of-tuples comparison so we catch any per-row drift.
    assert [tuple(r) for r in before] == [tuple(r) for r in after]
