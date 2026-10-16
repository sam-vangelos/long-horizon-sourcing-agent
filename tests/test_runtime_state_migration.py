"""Phase 1.5 tests for stop_reason normalization at the write boundary.

The runtime store now:

- Carries a ``stop_reason_detail TEXT`` column on ``runs`` (additive
  ALTER, idempotent).
- Bumps ``meta.schema_version`` from "3" to "4".
- Normalizes any non-canonical ``stop_reason`` value at the write
  boundary (``finish_run`` / ``set_run_stop_reason``); the original
  freeform string lands in ``stop_reason_detail`` so forensics survive.
- One-shot migration of existing legacy rows runs gated on
  ``schema_version < 4`` so a mixed-version process race cannot leave
  un-normalized rows behind (per critique A1).
- ALTER statements are wrapped in ``try/except OperationalError`` for
  ``"duplicate column name"`` so concurrent migrations from two
  ``RuntimeStateStore`` constructions can't crash one another (per
  critique A2).

These tests pin those properties.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from shared.runtime_state.store import (
    CURRENT_SCHEMA_VERSION,
    RuntimeStateStore,
)


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


def _start_run(store: RuntimeStateStore, tmp_path: Path) -> int:
    return store.start_run(
        source="linkedin",
        brief_id="brief-mig",
        output_dir=str(tmp_path),
        mode="fresh",
    )


def test_schema_version_pins_to_current(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with store.connect() as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
    assert row["value"] == CURRENT_SCHEMA_VERSION


def test_stop_reason_detail_column_present(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with store.connect() as conn:
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(runs)").fetchall()
        }
    assert "stop_reason_detail" in cols


def test_authoring_and_workspace_tables_installed_on_fresh_db(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    with store.connect() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "intake_sessions" in tables
    assert "reflection_sessions" in tables
    assert "workspace_entries" in tables
    assert "workspace_review_events" in tables
    assert "workspace_outreach_artifacts" in tables


def test_finish_run_canonical_value_leaves_detail_null(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)
    store.finish_run(run_id, "completed", stop_reason="normal")

    with store.connect() as conn:
        row = conn.execute(
            "SELECT stop_reason, stop_reason_detail FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    assert row["stop_reason"] == "normal"
    assert row["stop_reason_detail"] is None


def test_finish_run_freeform_value_normalizes_and_preserves_detail(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)
    # Today's catchall path at linkedin/session_orchestrator.py:281 stuffs
    # strings like this directly into stop_reason. Phase 1.5 normalizes at
    # the write boundary so the UI sees the canonical enum value, but the
    # original survives in stop_reason_detail.
    store.finish_run(run_id, "failed", stop_reason="error: KeyError")

    with store.connect() as conn:
        row = conn.execute(
            "SELECT stop_reason, stop_reason_detail FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    assert row["stop_reason"] == "fatal_runtime_error"
    assert row["stop_reason_detail"] == "error: KeyError"


def test_set_run_stop_reason_normalizes_in_place(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)
    store.set_run_stop_reason(run_id, "interrupted: KeyboardInterrupt")

    with store.connect() as conn:
        row = conn.execute(
            "SELECT stop_reason, stop_reason_detail FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    assert row["stop_reason"] == "fatal_runtime_error"
    assert row["stop_reason_detail"] == "interrupted: KeyboardInterrupt"


def test_legacy_rows_normalized_on_first_post_migration_open(
    tmp_path: Path,
) -> None:
    """Simulate a DB written by a pre-Phase-1.5 RuntimeStateStore: a row
    whose stop_reason is freeform ("error: KeyError") and whose
    schema_version is "3". On the next instantiation, _migrate must
    normalize the freeform value into the canonical enum and stash the
    original in stop_reason_detail."""

    db_path = tmp_path / "runtime_state.sqlite3"

    # Seed a legacy-shape DB by hand: schema before stop_reason_detail
    # existed, with a freeform stop_reason value.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    conn.execute(
        """
        CREATE TABLE runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            brief_id TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            stop_reason TEXT NOT NULL DEFAULT 'normal',
            started_at TEXT NOT NULL,
            ended_at TEXT,
            resumed_from_run_id INTEGER,
            resume_state_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(resumed_from_run_id) REFERENCES runs(id)
        );
        """
    )
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?)", ("schema_version", "3")
    )
    conn.execute(
        """
        INSERT INTO runs(source, brief_id, output_dir, mode, status,
                         stop_reason, started_at, ended_at)
        VALUES ('linkedin', 'brief-legacy', '/tmp/legacy', 'fresh',
                'failed', 'error: KeyError',
                '2025-01-01T00:00:00+00:00', '2025-01-01T00:01:00+00:00')
        """
    )
    conn.commit()
    conn.close()

    # Open via Phase-1.5 RuntimeStateStore — _migrate must run the
    # normalization pass once.
    RuntimeStateStore(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT stop_reason, stop_reason_detail FROM runs"
    ).fetchone()
    version = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()
    conn.close()

    assert row["stop_reason"] == "fatal_runtime_error"
    assert row["stop_reason_detail"] == "error: KeyError"
    assert version["value"] == CURRENT_SCHEMA_VERSION


def test_migration_idempotent_second_open_is_noop(tmp_path: Path) -> None:
    """Re-opening a Phase-1.5-migrated DB must not double-normalize.
    Specifically: a row whose stop_reason is canonical and whose
    stop_reason_detail already carries the original must keep both
    fields identical across re-opens."""

    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    run_id = _start_run(store, tmp_path)
    store.finish_run(run_id, "failed", stop_reason="error: KeyError")

    # First read.
    with store.connect() as conn:
        before = conn.execute(
            "SELECT stop_reason, stop_reason_detail FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()

    # Second open (simulating a process restart).
    RuntimeStateStore(db_path)
    with store.connect() as conn:
        after = conn.execute(
            "SELECT stop_reason, stop_reason_detail FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()

    assert before["stop_reason"] == after["stop_reason"]
    assert before["stop_reason_detail"] == after["stop_reason_detail"]


def test_concurrent_migration_does_not_crash(tmp_path: Path) -> None:
    """Two threads constructing RuntimeStateStore against the same DB at
    the same time race the ALTER TABLE statements. Per critique A2, the
    duplicate-column-name OperationalError must be tolerated as success.
    Without the fix, one of the two would raise and tear down the
    process."""

    db_path = tmp_path / "runtime_state.sqlite3"
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def construct() -> None:
        try:
            barrier.wait(timeout=2.0)
            RuntimeStateStore(db_path)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=construct) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert errors == [], f"concurrent migration raised: {errors}"


def test_finish_run_event_payload_carries_normalized_and_detail(
    tmp_path: Path,
) -> None:
    """The events stream still gets the original string (in detail) so
    forensic inspectors can see what the orchestrator actually wrote;
    runs.stop_reason gives the canonical value the UI consumes."""

    import json as _json

    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)
    store.finish_run(run_id, "failed", stop_reason="error: KeyError")

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT event_type, payload_json FROM events WHERE run_id = ? "
            "AND event_type = 'run_finished'",
            (run_id,),
        ).fetchall()
    assert len(rows) == 1
    payload = _json.loads(rows[0]["payload_json"])
    assert payload["stop_reason"] == "fatal_runtime_error"
    assert payload["stop_reason_detail"] == "error: KeyError"
