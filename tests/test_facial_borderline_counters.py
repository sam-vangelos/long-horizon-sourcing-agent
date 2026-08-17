"""C2 (slice 15) — counter widening + projection-rollup hygiene.

Pins the facial_borderline_count counter that was added at the production
schema (shared/schemas.py:SearchString), the SQLite work_units table
(shared/runtime_state/store.py), the metrics_json builder
(shared/runtime_state/linkedin.py:_work_unit_metrics), and the projection
read paths (shared/runtime_state/projections.py).

These tests document:
- The structural counter exists and round-trips end-to-end.
- The facial_skip math in _work_unit_metrics subtracts borderline so a future
  code path that persists raw FACIAL_BORDERLINE rows will not be silently
  absorbed into "skip".
- Backward compat: pre-C2 SQLite databases auto-migrate; pre-C2 projection
  JSON without the field deserializes with default 0.
- The orchestrator now preserves and increments FACIAL_BORDERLINE as its own
  facial outcome rather than folding it into FACIAL_YES.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from shared.runtime_state import RuntimeStateStore
from shared.runtime_state.linkedin import LinkedInRuntimeStateBridge
from shared.runtime_state.store import LINKEDIN_STRING_KIND
from shared.schemas import CandidateSnippet, OpusDecision, SearchString


# ---------------------------------------------------------------------------
# 1. SearchString dataclass shape
# ---------------------------------------------------------------------------


def test_searchstring_dataclass_has_borderline_counter():
    """The production SearchString dataclass exposes facial_borderline_count.

    Defaults to 0; can be set explicitly; survives to_dict / from_dict.
    """
    default = SearchString(id=1, name="x", boolean="y")
    assert default.facial_borderline_count == 0

    explicit = SearchString(id=2, name="x", boolean="y", facial_borderline_count=4)
    assert explicit.facial_borderline_count == 4

    payload = explicit.to_dict()
    assert payload["facial_borderline_count"] == 4

    round_trip = SearchString.from_dict(payload)
    assert round_trip.facial_borderline_count == 4


def test_searchstring_from_dict_backward_compat():
    """SearchString.from_dict tolerates pre-C2 dicts without the new key.

    Old projection JSONLs that predate slice 15 must deserialize cleanly with
    facial_borderline_count defaulting to 0. The dataclass-default mechanism
    in `cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})`
    handles this without an explicit `.get(..., 0)`.
    """
    legacy_payload = {"id": 1, "name": "x", "boolean": "y"}
    restored = SearchString.from_dict(legacy_payload)
    assert restored.facial_borderline_count == 0
    assert restored.facial_yes_count == 0
    assert restored.facial_no_count == 0


# ---------------------------------------------------------------------------
# 2. _work_unit_metrics facial_skip math
# ---------------------------------------------------------------------------


def test_facial_skip_math_subtracts_borderline():
    """C2 fix: facial_skip must NOT silently absorb borderline candidates.

    Pre-C2: skip = candidates - YES - NO  (10 - 3 - 4 = 3)
    Post-C2: skip = candidates - YES - NO - BORDERLINE  (10 - 3 - 4 - 2 = 1)

    The pre-C2 number conflated 2 borderline candidates into "skip"; this
    test pins the corrected math.
    """
    search_string = SearchString(
        id=1,
        name="test",
        boolean="x",
        candidates_count=10,
        facial_yes_count=3,
        facial_no_count=4,
        facial_borderline_count=2,
    )
    metrics = LinkedInRuntimeStateBridge._work_unit_metrics(search_string)
    assert metrics["facial_skip"] == 1
    assert metrics["facial_yes"] == 3
    assert metrics["facial_no"] == 4


def test_facial_skip_math_with_zero_borderline():
    """Regression: with borderline=0 the skip math is unchanged from pre-C2."""
    search_string = SearchString(
        id=1,
        name="test",
        boolean="x",
        candidates_count=10,
        facial_yes_count=3,
        facial_no_count=4,
        facial_borderline_count=0,
    )
    metrics = LinkedInRuntimeStateBridge._work_unit_metrics(search_string)
    assert metrics["facial_skip"] == 3


def test_facial_skip_math_does_not_go_negative():
    """The max(0, ...) floor survives C2: skip is never negative."""
    search_string = SearchString(
        id=1,
        name="test",
        boolean="x",
        candidates_count=5,
        facial_yes_count=3,
        facial_no_count=3,
        facial_borderline_count=0,
    )
    metrics = LinkedInRuntimeStateBridge._work_unit_metrics(search_string)
    assert metrics["facial_skip"] == 0

    overflowing = SearchString(
        id=2,
        name="test",
        boolean="x",
        candidates_count=5,
        facial_yes_count=2,
        facial_no_count=2,
        facial_borderline_count=4,
    )
    overflow_metrics = LinkedInRuntimeStateBridge._work_unit_metrics(overflowing)
    assert overflow_metrics["facial_skip"] == 0


# ---------------------------------------------------------------------------
# 3. SQLite schema + migration
# ---------------------------------------------------------------------------


def test_sqlite_work_units_table_includes_borderline_column(tmp_path: Path):
    """Fresh RuntimeStateStore creates work_units with the new column."""
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    with store.connect() as conn:
        rows = conn.execute("PRAGMA table_info(work_units)").fetchall()
    columns = {row["name"]: row for row in rows}
    assert "facial_borderline_count" in columns
    column = columns["facial_borderline_count"]
    assert column["type"] == "INTEGER"
    assert column["notnull"] == 1
    assert str(column["dflt_value"]) == "0"


def test_sqlite_migration_adds_borderline_column_to_old_db(tmp_path: Path):
    """Existing pre-C2 databases auto-migrate on open.

    Manually create a work_units table that lacks facial_borderline_count and
    has a row with no value for the column. Open a RuntimeStateStore against
    that path. The migration block must add the column with a 0 default, and
    the pre-existing row must read facial_borderline_count = 0.
    """
    db_path = tmp_path / "legacy.sqlite3"
    raw = sqlite3.connect(str(db_path))
    raw.row_factory = sqlite3.Row
    raw.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            brief_id TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            resumed_from_run_id INTEGER,
            resume_state_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE work_units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            brief_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            source_unit_id TEXT NOT NULL,
            display_name TEXT NOT NULL,
            ordering_index INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            checkpoint_json TEXT NOT NULL DEFAULT '{}',
            family_key TEXT NOT NULL DEFAULT '',
            novelty_bucket TEXT NOT NULL DEFAULT '',
            domain_lane TEXT NOT NULL DEFAULT '',
            result_count INTEGER NOT NULL DEFAULT 0,
            candidates_discovered INTEGER NOT NULL DEFAULT 0,
            candidates_enriched INTEGER NOT NULL DEFAULT 0,
            candidates_insufficient INTEGER NOT NULL DEFAULT 0,
            facial_yes_count INTEGER NOT NULL DEFAULT 0,
            facial_no_count INTEGER NOT NULL DEFAULT 0,
            saves_count INTEGER NOT NULL DEFAULT 0,
            rejected_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT NOT NULL DEFAULT '',
            started_at TEXT,
            ended_at TEXT,
            UNIQUE(run_id, kind, source_unit_id)
        );
        """
    )
    raw.execute(
        "INSERT INTO runs(source, brief_id, output_dir, mode, status, started_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("linkedin", "brief-legacy", "/tmp/legacy", "fresh", "running", "2026-04-01T00:00:00+00:00"),
    )
    raw.execute(
        """
        INSERT INTO work_units(
            run_id, source, brief_id, kind, source_unit_id, display_name, status,
            facial_yes_count, facial_no_count
        ) VALUES (1, 'linkedin', 'brief-legacy', 'linkedin_string', '1',
                  'Legacy string', 'done', 5, 7)
        """
    )
    raw.commit()
    raw.close()

    pre_columns = _column_names(db_path, "work_units")
    assert "facial_borderline_count" not in pre_columns

    store = RuntimeStateStore(db_path)
    post_columns = _column_names(db_path, "work_units")
    assert "facial_borderline_count" in post_columns

    with store.connect() as conn:
        row = conn.execute(
            "SELECT facial_yes_count, facial_no_count, facial_borderline_count "
            "FROM work_units WHERE source_unit_id = '1'"
        ).fetchone()
    assert row["facial_yes_count"] == 5
    assert row["facial_no_count"] == 7
    assert row["facial_borderline_count"] == 0


def _column_names(db_path: Path, table: str) -> set[str]:
    raw = sqlite3.connect(str(db_path))
    try:
        rows = raw.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        raw.close()
    return {row[1] for row in rows}


# ---------------------------------------------------------------------------
# 4. SQLite round-trip via upsert_work_unit
# ---------------------------------------------------------------------------


def test_sqlite_round_trip_preserves_borderline_count(tmp_path: Path):
    """A non-zero facial_borderline_count round-trips through the store.

    This test exercises the canonical path that any future code persisting raw
    FACIAL_BORDERLINE rows would use. With slices 13/14 active no production
    path increments this counter, but the canonical store must already
    support the value.
    """
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-c2",
        output_dir=str(tmp_path),
        mode="fresh",
    )

    search_string = SearchString(
        id=42,
        name="C2 round trip",
        boolean="x",
        status="done",
        result_count=20,
        candidates_count=20,
        facial_yes_count=5,
        facial_no_count=8,
        facial_borderline_count=7,
    )
    store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-c2",
        kind=LINKEDIN_STRING_KIND,
        source_unit_id=str(search_string.id),
        display_name=search_string.name,
        ordering_index=0,
        status=search_string.status,
        payload=search_string.to_dict(),
        counters={
            "result_count": search_string.result_count,
            "candidates_discovered": search_string.candidates_count,
            "facial_yes_count": search_string.facial_yes_count,
            "facial_no_count": search_string.facial_no_count,
            "facial_borderline_count": search_string.facial_borderline_count,
            "saves_count": 0,
            "rejected_count": 0,
        },
    )

    with store.connect() as conn:
        row = conn.execute(
            "SELECT facial_yes_count, facial_no_count, facial_borderline_count "
            "FROM work_units WHERE source_unit_id = '42'"
        ).fetchone()
    assert row["facial_yes_count"] == 5
    assert row["facial_no_count"] == 8
    assert row["facial_borderline_count"] == 7


def test_existing_yes_no_counters_unchanged_by_c2(tmp_path: Path):
    """Regression: omitting facial_borderline_count from the counters dict
    leaves yes/no untouched and defaults borderline to 0.

    Pins backward compat for any caller that has not yet been updated to
    populate facial_borderline_count. The store's `int(counters.get(...,
    0))` pattern keeps these calls safe.
    """
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-yes-no",
        output_dir=str(tmp_path),
        mode="fresh",
    )

    store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-yes-no",
        kind=LINKEDIN_STRING_KIND,
        source_unit_id="1",
        display_name="Yes/No only caller",
        ordering_index=0,
        status="done",
        payload={"id": 1},
        counters={
            "result_count": 0,
            "candidates_discovered": 15,
            "facial_yes_count": 5,
            "facial_no_count": 10,
            "saves_count": 0,
            "rejected_count": 0,
        },
    )

    with store.connect() as conn:
        row = conn.execute(
            "SELECT facial_yes_count, facial_no_count, facial_borderline_count "
            "FROM work_units WHERE source_unit_id = '1'"
        ).fetchone()
    assert row["facial_yes_count"] == 5
    assert row["facial_no_count"] == 10
    assert row["facial_borderline_count"] == 0


# ---------------------------------------------------------------------------
# 5. Orchestrator preserves the ternary counter
# ---------------------------------------------------------------------------


def test_orchestrator_increments_distinct_borderline_counter():
    """A parser BORDERLINE increments only the distinct ternary bucket."""
    with tempfile.TemporaryDirectory() as td:
        with patch("linkedin.orchestrator.load_brief") as mock_brief, \
             patch("linkedin.orchestrator.init_judger"), \
             patch("linkedin.orchestrator.LinkedInBrowser"):
            brief = MagicMock()
            brief.id = "test"
            brief.linkedin_project_id = "test-project"
            brief.has_v2_schema = True
            brief.employer_blacklist = []
            mock_brief.return_value = brief

            brief_path = Path(td) / "brief.json"
            brief_path.write_text('{"id": "test"}')

            from linkedin.orchestrator import Pipeline
            p = Pipeline(brief_path=str(brief_path), output_dir=td)

        p._tightening_prefix = ""
        p._bias_monitor = None
        p._full_evaluate = AsyncMock(return_value=None)

        snippet = CandidateSnippet(
            name="Borderline Person",
            headline="",
            current_title="",
            current_company="",
            location="Somewhere",
            education_snippet="",
            profile_url="/talent/profile/borderline",
            source_string_id=1,
            source_string_name="test",
            page=1,
            result_rank=1,
        )

        borderline_decision = OpusDecision(
            stage="facial",
            decision="FACIAL_BORDERLINE",
            path="none",
            confidence=1.0,
            rationale="ambiguous",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )

        with patch(
            "linkedin.orchestrator.facial_judge",
            return_value=borderline_decision,
        ), patch(
            "linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED",
            True,
        ):
            asyncio.run(p._evaluate_snippet(snippet))

        assert p.stats.get("facial_borderline") == 1
        assert p.stats.get("facial_yes") == 0
        assert p.stats.get("facial_no") == 0
