"""The events work-unit index keeps execution-validity probes off a full scan."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from shared.runtime_state.admin import rebuild_compat_projections
from shared.runtime_state.store import RuntimeStateStore
from tests.test_linkedin_contamination_reset import (
    BRIEF_ID,
    _execute_string,
    _invalidate,
    _record_terminal,
    _start_run,
)

INDEX_NAME = "idx_events_work_unit_type"
PROJECTION_FILES = (
    "progress.json",
    f"candidate_history-{BRIEF_ID}.jsonl",
    f"search_memory-{BRIEF_ID}.json",
    "snippets.jsonl",
    "facial_judgments.jsonl",
    "profile_summaries.jsonl",
    "final_judgments.jsonl",
)
WALL_CLOCK_KEYS = {"updated_at", "last_seen_at"}


def _index_columns(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        return [row[2] for row in conn.execute(f"PRAGMA index_info({INDEX_NAME})")]


def _store_with_one_invalidated_execution(tmp_path: Path) -> tuple[RuntimeStateStore, int]:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = _start_run(store)
    kept_unit = _execute_string(store, run_id=run_id, source_unit_id="1")
    dropped_unit = _execute_string(store, run_id=run_id, source_unit_id="2")
    _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=kept_unit,
        identity_key="https://linkedin.com/in/kept",
        decision="REJECT",
    )
    _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=dropped_unit,
        identity_key="https://linkedin.com/in/dropped",
        decision="REJECT",
    )
    _invalidate(store, run_id=run_id, work_unit_id=dropped_unit)
    return store, run_id


def _strip_wall_clock(value):
    if isinstance(value, dict):
        return {k: _strip_wall_clock(v) for k, v in value.items() if k not in WALL_CLOCK_KEYS}
    if isinstance(value, list):
        return [_strip_wall_clock(v) for v in value]
    return value


def _projection_contents(output_dir: Path) -> dict[str, object]:
    contents: dict[str, object] = {}
    for name in PROJECTION_FILES:
        raw = (output_dir / name).read_bytes()
        if name.startswith("search_memory-"):
            contents[name] = _strip_wall_clock(json.loads(raw))
        else:
            contents[name] = raw
    return contents


def test_fresh_and_pre_existing_stores_both_carry_the_index(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    RuntimeStateStore(db_path)
    assert _index_columns(db_path) == ["work_unit_id", "event_type"]

    with sqlite3.connect(db_path) as conn:
        conn.execute(f"DROP INDEX {INDEX_NAME}")
    assert _index_columns(db_path) == []

    RuntimeStateStore(db_path)
    assert _index_columns(db_path) == ["work_unit_id", "event_type"]


def test_projection_rebuild_validity_probes_seek_the_index(tmp_path: Path) -> None:
    store, run_id = _store_with_one_invalidated_execution(tmp_path)
    executed: list[str] = []
    original_connect = RuntimeStateStore.connect

    @contextmanager
    def traced_connect(self):
        with original_connect(self) as conn:
            conn.set_trace_callback(executed.append)
            yield conn

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True), patch.object(
        RuntimeStateStore, "connect", traced_connect
    ):
        rebuild_compat_projections(store, run_id=run_id, output_dir=tmp_path / "out")

    probing = [sql for sql in executed if "execution_invalidation" in sql]
    # Search memory plus the four stage projections each carry a probe.
    assert len(probing) == 5
    with sqlite3.connect(store.db_path) as conn:
        for sql in probing:
            plan = [row[-1] for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]
            probe_steps = [step for step in plan if "execution_invalidation" in step]
            assert probe_steps, plan
            assert all(INDEX_NAME in step for step in probe_steps), plan


def test_projections_are_identical_with_and_without_the_index(tmp_path: Path) -> None:
    store, run_id = _store_with_one_invalidated_execution(tmp_path)
    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        rebuild_compat_projections(store, run_id=run_id, output_dir=tmp_path / "indexed")
        with store.connect() as conn:
            conn.execute(f"DROP INDEX {INDEX_NAME}")
        rebuild_compat_projections(store, run_id=run_id, output_dir=tmp_path / "unindexed")

    indexed = _projection_contents(tmp_path / "indexed")
    assert indexed == _projection_contents(tmp_path / "unindexed")
    final_urls = [
        json.loads(line)["profile_url"]
        for line in (tmp_path / "indexed" / "final_judgments.jsonl").read_text().splitlines()
    ]
    assert final_urls == ["https://linkedin.com/in/kept"]
