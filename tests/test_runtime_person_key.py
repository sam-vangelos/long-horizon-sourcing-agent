"""W1-S2 runtime-state person keying tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

from github.schemas import GitHubProgress
from github.work_units import GitHubWorkUnitService
from shared.runtime_state import GitHubRuntimeStateBridge
from shared.runtime_state.github import (
    PersonKeySet,
    github_identity_keys_from_seen,
)
from shared.runtime_state.store import RuntimeStateStore


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


def _candidate_columns(conn: sqlite3.Connection) -> set[str]:
    return {row["name"] for row in conn.execute("PRAGMA table_info(candidates)").fetchall()}


def _all_candidate_rows(conn: sqlite3.Connection) -> list[dict]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM candidates ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def _seed_pre_person_key_db(db_path: Path) -> None:
    """Legacy DB shape: candidates table without ``person_key``."""

    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key, value) VALUES ('schema_version', '12');

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
            resume_state_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE candidates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            brief_id TEXT NOT NULL,
            identity_key TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            profile_url TEXT NOT NULL DEFAULT '',
            current_lifecycle_state TEXT NOT NULL,
            terminal_decision TEXT,
            terminal_payload_json TEXT NOT NULL DEFAULT '{}',
            last_work_unit_id INTEGER,
            last_attempt_id INTEGER,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            UNIQUE(brief_id, source, identity_key)
        );

        INSERT INTO runs(
            source, brief_id, output_dir, mode, status, started_at
        ) VALUES (
            'github', 'legacy-brief', '/tmp/legacy', 'fresh', 'running',
            '2025-01-01T00:00:00+00:00'
        );

        INSERT INTO candidates(
            source, brief_id, identity_key, display_name, profile_url,
            current_lifecycle_state, terminal_decision, terminal_payload_json,
            first_seen_at, last_seen_at
        ) VALUES (
            'github', 'legacy-brief', 'Alice', 'Alice',
            'https://github.com/Alice', 'failed_terminal', 'REJECT', '{}',
            '2025-01-01T00:00:00+00:00', '2025-01-01T00:00:00+00:00'
        );
        """
    )
    conn.commit()
    conn.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    store = _make_store(tmp_path)
    run_id = store.start_run(
        source="github",
        brief_id="brief-mig",
        output_dir=str(tmp_path),
        mode="fresh",
    )
    store.ensure_candidate(
        source="github",
        brief_id="brief-mig",
        identity_key="alice",
        person_key="gh:alice",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="github",
        brief_id="brief-mig",
        identity_key="alice",
        new_state="failed_terminal",
        terminal_decision="REJECT",
    )

    with store.connect() as conn:
        before_cols = _candidate_columns(conn)
        before_rows = _all_candidate_rows(conn)

    RuntimeStateStore(db_path)
    RuntimeStateStore(db_path)

    with store.connect() as conn:
        after_cols = _candidate_columns(conn)
        after_rows = _all_candidate_rows(conn)
        index_names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }

    assert before_cols == after_cols
    assert before_rows == after_rows
    assert "person_key" in after_cols
    assert "idx_candidates_brief_person_key" in index_names


def test_pre_migration_state_dir_opens_and_resumes(tmp_path: Path) -> None:
    state_dir = tmp_path / "legacy-state"
    state_dir.mkdir()
    db_path = state_dir / "runtime_state.sqlite3"
    _seed_pre_person_key_db(db_path)

    store = RuntimeStateStore(db_path)

    with store.connect() as conn:
        assert "person_key" in _candidate_columns(conn)
        row = conn.execute(
            "SELECT identity_key, person_key FROM candidates WHERE identity_key = ?",
            ("Alice",),
        ).fetchone()
        assert row is not None
        assert row["person_key"] == "gh:alice"

    person_keys = store.list_terminal_person_keys(source="github", brief_id="legacy-brief")
    assert person_keys == ["gh:alice"]

    RuntimeStateStore(db_path)


def test_backfill_keys_existing_github_rows(tmp_path: Path) -> None:
    state_dir = tmp_path / "backfill-state"
    state_dir.mkdir()
    db_path = state_dir / "runtime_state.sqlite3"
    _seed_pre_person_key_db(db_path)

    store = RuntimeStateStore(db_path)

    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT identity_key, person_key
            FROM candidates
            WHERE source = 'github'
            ORDER BY identity_key
            """
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["identity_key"] == "Alice"
    assert rows[0]["person_key"] == "gh:alice"


def test_same_person_two_hubs_dedupes(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    brief_id = "brief-dedup"
    run_id = store.start_run(
        source="github",
        brief_id=brief_id,
        output_dir=str(tmp_path),
        mode="fresh",
    )

    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="github",
        brief_id=brief_id,
        identity_key="alice",
        display_name="Alice",
        profile_url="https://github.com/alice",
        person_key="gh:alice",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="github",
        brief_id=brief_id,
        identity_key="alice",
        new_state="failed_terminal",
        terminal_decision="REJECT",
    )

    pipeline = MagicMock()
    pipeline._seen_usernames = PersonKeySet()
    pipeline._in_flight_usernames = set()
    pipeline._runtime_bridge = GitHubRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id=brief_id,
        brief_name="test",
    )

    service = GitHubWorkUnitService(pipeline)
    assert service.dedup_usernames(["alice"]) == []

    with store.connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE brief_id = ?",
            (brief_id,),
        ).fetchone()[0]

    assert count == 1
    blocked = store.get_blocked_person_keys(brief_id, ["gh:alice"])
    assert blocked == {"gh:alice"}


def test_mixed_case_login_does_not_create_phantom_row(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    brief_id = "brief-mixed-case"
    run_id = store.start_run(
        source="github",
        brief_id=brief_id,
        output_dir=str(tmp_path),
        mode="fresh",
    )

    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="github",
        brief_id=brief_id,
        identity_key="TorvaldsX",
        display_name="TorvaldsX",
        profile_url="https://github.com/TorvaldsX",
        person_key="gh:torvaldsx",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="github",
        brief_id=brief_id,
        identity_key="TorvaldsX",
        new_state="failed_terminal",
        terminal_decision="REJECT",
    )

    seen = PersonKeySet(["TorvaldsX"])
    progress = GitHubProgress(brief_name="test")

    for _ in range(2):
        progress.discovered_usernames = github_identity_keys_from_seen(seen)
        store.sync_github_progress(run_id, progress)
        seen = PersonKeySet(
            store.list_terminal_person_keys(source="github", brief_id=brief_id)
        )
        progress.discovered_usernames = store.list_terminal_identity_keys(
            source="github",
            brief_id=brief_id,
        )

    with store.connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE brief_id = ?",
            (brief_id,),
        ).fetchone()[0]
        identity_keys = [
            row["identity_key"]
            for row in conn.execute(
                "SELECT identity_key FROM candidates WHERE brief_id = ?",
                (brief_id,),
            ).fetchall()
        ]

    person_keys = store.list_terminal_person_keys(source="github", brief_id=brief_id)

    assert count == 1
    assert identity_keys == ["TorvaldsX"]
    assert len(person_keys) == len(set(person_keys))
    assert person_keys == ["gh:torvaldsx"]
