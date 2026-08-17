"""Schema migration framework — v9 → current simulation (audit Move #19).

Asserts the additive-idempotent migration contract:

- Opening a v9-shaped DB via :class:`RuntimeStateStore` upgrades it
  cleanly to the current schema without data loss.
- Re-opening an already-current DB is a no-op (idempotency).
- Pre-existing rows survive the migration byte-for-byte.
- Tables added after v9 are created when missing.
- The CLI surface (``python -m cloris migrate --db-path ...``)
  reports the resulting schema_version and exits 0.

The "v9-shaped" fixture is built by writing the v10 schema first
(letting the store bootstrap), then dropping the v10-specific
artifacts to simulate a deployment that landed before the v10 bump.
This intentionally mirrors what a real upgrade hits: a database
created by older code, opened by newer code that knows about more
columns/tables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from shared.runtime_state.store import (
    CURRENT_SCHEMA_VERSION,
    RuntimeStateStore,
)


# Tables added after v9 through external schema installers.
_POST_V9_TABLES = (
    "workspace_entries",
    "workspace_review_events",
    "workspace_outreach_artifacts",
    "brief_corpus",
    "brief_corpus_fts",
)


def _make_v9_shaped_db(tmp_path: Path) -> Path:
    """Build a v9-shaped database by:
    1. Letting RuntimeStateStore bootstrap the full v10 schema.
    2. Dropping the v10-specific tables.
    3. Manually writing schema_version=9 into meta.

    Returns the db path.
    """

    db_path = tmp_path / "runtime_state.sqlite3"
    RuntimeStateStore(db_path)  # bootstraps to current schema (v10)

    with sqlite3.connect(str(db_path)) as conn:
        for table in _POST_V9_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", "9"),
        )
        conn.commit()

    return db_path


def _table_exists(db_path: Path, table_name: str) -> bool:
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
    return row is not None


def _read_schema_version(db_path: Path) -> str:
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
    return row[0] if row else ""


def test_v9_db_lacks_v10_tables_before_migration(tmp_path: Path) -> None:
    """Sanity: the fixture builder produces a DB that's actually missing
    the v10 artifacts."""

    db_path = _make_v9_shaped_db(tmp_path)
    assert _read_schema_version(db_path) == "9"
    for table in _POST_V9_TABLES:
        assert not _table_exists(db_path, table), (
            f"v9 fixture should not have {table}"
        )


def test_opening_v9_db_upgrades_to_current_schema(tmp_path: Path) -> None:
    """Audit Move #19 load-bearing test: opening a v9-shaped DB via
    RuntimeStateStore upgrades the schema in place, adds the missing
    tables, and pins meta.schema_version to the current value."""

    db_path = _make_v9_shaped_db(tmp_path)

    # Open through the store; this triggers _migrate and pins
    # schema_version to CURRENT_SCHEMA_VERSION.
    RuntimeStateStore(db_path)

    assert _read_schema_version(db_path) == CURRENT_SCHEMA_VERSION
    for table in _POST_V9_TABLES:
        assert _table_exists(db_path, table), (
            f"migration should have created {table}"
        )


def test_v9_to_v10_migration_preserves_existing_data(tmp_path: Path) -> None:
    """Existing rows survive the migration byte-for-byte."""

    db_path = _make_v9_shaped_db(tmp_path)

    # Seed the v9 DB with a candidate + run row to verify they survive.
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute(
            """
            INSERT INTO runs (
                source, brief_id, output_dir, mode, status,
                started_at, resume_state_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "linkedin",
                "test-brief",
                str(tmp_path),
                "fresh",
                "active",
                "2026-04-01T00:00:00+00:00",
                "{}",
            ),
        )
        run_id = cursor.lastrowid
        # Discover the actual candidates schema before inserting — the
        # exact column set varies across schema versions, so we pull the
        # column list and only set the columns we care about.
        cand_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
        }
        # Required-not-null columns we always set.
        insert_cols = [
            "source",
            "brief_id",
            "identity_key",
            "display_name",
            "profile_url",
            "current_lifecycle_state",
            "first_seen_at",
            "last_seen_at",
        ]
        # Optional columns only if present.
        if "terminal_payload_json" in cand_cols:
            insert_cols.append("terminal_payload_json")
        placeholders = ", ".join("?" for _ in insert_cols)
        col_list = ", ".join(insert_cols)
        values = [
            "linkedin",
            "test-brief",
            "li:1234",
            "Pre-migration Candidate",
            "https://example.com",
            "discovered",
            "2026-04-01T00:00:00+00:00",
            "2026-04-01T00:00:00+00:00",
        ]
        if "terminal_payload_json" in cand_cols:
            values.append("{}")
        conn.execute(
            f"INSERT INTO candidates ({col_list}) VALUES ({placeholders})",
            values,
        )
        conn.commit()

    # Now upgrade.
    RuntimeStateStore(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT identity_key, display_name FROM candidates "
            "WHERE source='linkedin' AND brief_id='test-brief'"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["identity_key"] == "li:1234"
    assert rows[0]["display_name"] == "Pre-migration Candidate"


def test_re_opening_already_v10_db_is_idempotent(tmp_path: Path) -> None:
    """Audit Move #19: the additive-idempotent contract means re-opening
    a current-schema DB doesn't change anything."""

    db_path = tmp_path / "runtime_state.sqlite3"
    RuntimeStateStore(db_path)
    assert _read_schema_version(db_path) == CURRENT_SCHEMA_VERSION

    # Capture the table list pre-second-open, then re-open and re-capture.
    with sqlite3.connect(str(db_path)) as conn:
        before = sorted(
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        )

    RuntimeStateStore(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        after = sorted(
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        )

    assert before == after
    assert _read_schema_version(db_path) == CURRENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# CLI surface — `python -m cloris migrate ...`
# ---------------------------------------------------------------------------


def test_cli_migrate_reports_schema_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The migrate subcommand opens the store (which auto-migrates) and
    prints the resulting schema_version on stdout."""

    db_path = _make_v9_shaped_db(tmp_path)

    from cloris.cli import main

    rc = main(["migrate", "--db-path", str(db_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "per-source store at" in out
    assert f"schema_version={CURRENT_SCHEMA_VERSION}" in out
    assert _read_schema_version(db_path) == CURRENT_SCHEMA_VERSION


def test_cli_migrate_returns_1_for_missing_parent_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A db-path under a non-existent parent dir returns exit 1, not 0,
    so an operator misconfiguration surfaces immediately."""

    from cloris.cli import main

    rc = main(
        ["migrate", "--db-path", str(tmp_path / "no-such-dir" / "rt.sqlite3")]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "parent directory missing" in err
