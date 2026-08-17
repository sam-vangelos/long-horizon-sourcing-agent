"""Phase 2 tests for ``shared.runtime_state.read_models``.

Pins:

- The hard layering rule: ``read_models.py`` must NOT import
  ``shared.runtime_state.store``. AST-walk the source to enforce.
- Each primitive against missing / corrupt / WAL-not-readable / empty
  / populated DBs.
- Tagged-union semantics on ``work_unit_progress``: ``not_found`` /
  ``empty`` / ``counts`` are distinguishable.
- ``attempt_health`` time-window math (per critique B2: parse-and-compare,
  not string-compare).
- ``has_pending_work`` matches the documented divergence from the
  orchestrator's ``_resume_has_pending_work``.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from shared.runtime_state import read_models
from shared.runtime_state.read_models import (
    AttemptHealth,
    RunSummary,
    WorkUnitProgress,
    attempt_health,
    has_pending_work,
    latest_run_summary,
    work_unit_progress,
)
# Note: ``RunTelemetry`` / ``run_telemetry`` and the helper rows
# (``TelemetryAttempt`` / ``TelemetryEvent``) were retired from
# ``read_models.py`` in the legacy-monorepo import (commit eb9d31f);
# the corresponding tests below were removed in Slice A.0 of the
# Multi-Agent Production Plan to unblock ``make validate``. If a future
# slice restores telemetry primitives, restore the imports + tests at
# the same time.
from shared.runtime_state.store import RuntimeStateStore


# --- layering rule ----------------------------------------------------------


def test_read_models_does_not_import_runtime_state_store() -> None:
    """The whole point of read_models.py is to give read-only consumers
    a path that does NOT trigger RuntimeStateStore.__init__'s DDL +
    INSERT-OR-REPLACE side effects. Importing the store class would
    silently re-introduce that hazard. AST-walk the source so the
    enforcement is mechanical, not stylistic."""

    source = read_models.__file__
    assert source is not None
    tree = ast.parse(Path(source).read_text())

    forbidden = "RuntimeStateStore"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.endswith("runtime_state.store") or module == "shared.runtime_state.store":
                names = [alias.name for alias in node.names]
                assert forbidden not in names, (
                    f"read_models must not import {forbidden} from {module}; "
                    "the writer's __init__ runs DDL on every instantiation."
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "shared.runtime_state.store", (
                    "read_models must not import the writer module."
                )


def test_read_models_does_not_create_db_files(tmp_path: Path) -> None:
    """Reading from a state dir whose runtime_state.sqlite3 does NOT
    exist must not create one. The writer creates the file on
    instantiation; this is exactly the behavior we're trying to avoid
    on the read path."""

    db_path = tmp_path / "runtime_state.sqlite3"
    assert not db_path.exists()

    assert latest_run_summary(db_path) is None
    assert has_pending_work(tmp_path) is None
    assert attempt_health(db_path, run_id=1) == AttemptHealth()
    assert work_unit_progress(db_path, run_id=1, kind="linkedin_string") == (
        WorkUnitProgress(kind="not_found")
    )
    # Bug 2 population audit: the four read helpers added to migrate
    # GET endpoints off RuntimeStateStore must also collapse cleanly
    # on missing DBs and must not create the file as a side effect.
    assert read_models.list_intake_sessions(db_path) == []
    assert read_models.get_intake_session(db_path, session_id=1) is None
    assert read_models.get_active_reflection_for_brief(
        db_path, brief_id="any"
    ) is None
    assert read_models.get_reflection_session(db_path, session_id=1) is None

    assert not db_path.exists(), "read primitives must not create the DB file"


# --- latest_run_summary -----------------------------------------------------


def test_latest_run_summary_missing_db(tmp_path: Path) -> None:
    assert latest_run_summary(tmp_path / "missing.sqlite3") is None


def test_latest_run_summary_corrupt_db(tmp_path: Path) -> None:
    bad = tmp_path / "bad.sqlite3"
    bad.write_bytes(b"not a sqlite db")
    assert latest_run_summary(bad) is None


def test_latest_run_summary_empty_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    RuntimeStateStore(db_path)
    assert latest_run_summary(db_path) is None


def test_latest_run_summary_returns_latest_row(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-rm",
        output_dir=str(tmp_path),
        mode="fresh",
    )
    store.finish_run(run_id, "completed", stop_reason="normal")

    summary = latest_run_summary(db_path)
    assert summary is not None
    assert summary.id == run_id
    assert summary.status == "completed"
    assert summary.stop_reason == "normal"
    assert summary.mode == "fresh"
    assert summary.started_at is not None
    assert summary.ended_at is not None


# --- has_pending_work --------------------------------------------------------


def test_has_pending_work_missing_progress_json_returns_none(tmp_path: Path) -> None:
    """Per the documented divergence: passive read model returns None
    on missing/malformed inputs, NOT True. The orchestrator's bias
    toward "attempt resume" is appropriate for an active worker but
    wrong for a passive observer surface."""

    assert has_pending_work(tmp_path) is None


def test_has_pending_work_truthy_pending_block_strings(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text(
        json.dumps({"pending_block_string_ids": ["s1", "s2"], "strings": []})
    )
    assert has_pending_work(tmp_path) is True


def test_has_pending_work_empty_returns_false(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text(json.dumps({"strings": []}))
    assert has_pending_work(tmp_path) is False


def test_has_pending_work_queued_string_returns_true(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text(
        json.dumps({"strings": [{"id": 1, "status": "queued"}]})
    )
    assert has_pending_work(tmp_path) is True


def test_has_pending_work_strings_not_list_returns_none(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text(
        json.dumps({"strings": "not a list"})
    )
    assert has_pending_work(tmp_path) is None


def test_has_pending_work_malformed_json_returns_none(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text("not json")
    assert has_pending_work(tmp_path) is None


@pytest.mark.parametrize("facial_decision", ["FACIAL_YES", "FACIAL_BORDERLINE"])
def test_has_pending_work_finds_unresolved_full_review_obligation(
    tmp_path: Path,
    facial_decision: str,
) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-rm",
        output_dir=str(tmp_path),
        mode="fresh",
    )
    store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-rm",
        kind="linkedin_string",
        source_unit_id="1",
        display_name="done string",
        ordering_index=0,
        status="done",
    )
    candidate_id = store.ensure_candidate(
        source="linkedin",
        brief_id="brief-rm",
        identity_key="/talent/profile/pending",
        profile_url="/talent/profile/pending",
    )
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO candidate_attempts("
            "run_id, candidate_id, stage, attempt_number, status, payload_json, "
            "source_cursor_json, started_at, ended_at"
            ") VALUES (?, ?, 'facial', 1, 'succeeded', ?, '{}', '', '')",
            (
                run_id,
                candidate_id,
                json.dumps(
                    {"facial_decision": {"decision": facial_decision}}
                ),
                ),
            )

    resumed_run_id = store.start_run(
        source="linkedin",
        brief_id="brief-rm",
        output_dir=str(tmp_path),
        mode="resume",
        resumed_from_run_id=run_id,
    )
    store.upsert_work_unit(
        run_id=resumed_run_id,
        source="linkedin",
        brief_id="brief-rm",
        kind="linkedin_string",
        source_unit_id="1",
        display_name="cloned done string",
        ordering_index=0,
        status="done",
    )

    assert has_pending_work(tmp_path) is True


@pytest.mark.parametrize(
    ("full_decision", "save_succeeded", "expected"),
    [
        ("REJECT", False, False),
        ("SAVE", False, True),
        ("SAVE", True, False),
    ],
)
def test_has_pending_work_uses_full_terminal_and_save_receipt_semantics(
    tmp_path: Path,
    full_decision: str,
    save_succeeded: bool,
    expected: bool,
) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-rm",
        output_dir=str(tmp_path),
        mode="fresh",
    )
    store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-rm",
        kind="linkedin_string",
        source_unit_id="1",
        display_name="done string",
        ordering_index=0,
        status="done",
    )
    candidate_id = store.ensure_candidate(
        source="linkedin",
        brief_id="brief-rm",
        identity_key="/talent/profile/candidate",
        profile_url="/talent/profile/candidate",
    )
    with store.connect() as conn:
        facial_attempt_id = conn.execute(
            "INSERT INTO candidate_attempts("
            "run_id, candidate_id, stage, attempt_number, status, payload_json, "
            "source_cursor_json, started_at, ended_at"
            ") VALUES (?, ?, 'facial', 1, 'succeeded', ?, '{}', '', '')",
            (
                run_id,
                candidate_id,
                json.dumps(
                    {"facial_decision": {"decision": "FACIAL_YES"}}
                ),
            ),
        ).lastrowid
        full_attempt_id = conn.execute(
            "INSERT INTO candidate_attempts("
            "run_id, candidate_id, stage, attempt_number, status, payload_json, "
            "source_cursor_json, started_at, ended_at"
            ") VALUES (?, ?, 'full', 1, 'succeeded', ?, '{}', '', '')",
            (
                run_id,
                candidate_id,
                json.dumps({"full_decision": {"decision": full_decision}}),
            ),
        ).lastrowid
        if save_succeeded:
            conn.execute(
                "INSERT INTO side_effects("
                "run_id, candidate_id, attempt_id, effect_type, idempotency_key, "
                "status, payload_json, created_at, updated_at"
                ") VALUES (?, ?, ?, 'linkedin_save', 'save', 'succeeded', '{}', '', '')",
                (run_id, candidate_id, full_attempt_id or facial_attempt_id),
            )

    assert has_pending_work(tmp_path) is expected


# --- work_unit_progress ------------------------------------------------------


def test_work_unit_progress_not_found_for_unknown_run(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    RuntimeStateStore(db_path)
    result = work_unit_progress(db_path, run_id=9999, kind="linkedin_string")
    assert result.kind == "not_found"


def test_work_unit_progress_empty_when_no_units(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-rm",
        output_dir=str(tmp_path),
        mode="fresh",
    )
    result = work_unit_progress(db_path, run_id=run_id, kind="linkedin_string")
    assert result.kind == "empty"


def test_work_unit_progress_counts_when_units_present(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-rm",
        output_dir=str(tmp_path),
        mode="fresh",
    )
    for i, status in enumerate(["queued", "queued", "in_progress", "done"]):
        store.upsert_work_unit(
            run_id=run_id,
            source="linkedin",
            brief_id="brief-rm",
            kind="linkedin_string",
            source_unit_id=str(i),
            display_name=f"unit-{i}",
            ordering_index=i,
            status=status,
        )

    result = work_unit_progress(db_path, run_id=run_id, kind="linkedin_string")
    assert result.kind == "counts"
    assert result.queued == 2
    assert result.in_progress == 1
    assert result.done == 1


# --- attempt_health ----------------------------------------------------------


def _direct_insert_attempt(
    db_path: Path,
    *,
    run_id: int,
    candidate_id: int,
    status: str,
    failure_kind: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> None:
    """Insert a candidate_attempts row directly. Bypasses the store's
    state-machine guards because attempt_health tests want to seed
    arbitrary status / failure_kind combinations without driving
    candidate lifecycle transitions."""

    started_at = started_at or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO candidate_attempts(run_id, candidate_id, stage, "
            "attempt_number, status, failure_kind, started_at, ended_at) "
            "VALUES (?, ?, 'snippet', 1, ?, ?, ?, ?)",
            (run_id, candidate_id, status, failure_kind, started_at, ended_at),
        )
        conn.commit()
    finally:
        conn.close()


def test_attempt_health_no_attempts_returns_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-rm",
        output_dir=str(tmp_path),
        mode="fresh",
    )
    health = attempt_health(db_path, run_id=run_id)
    assert health.total_attempts_in_window == 0
    assert health.last_success_age_s is None
    assert health.dominant_failure_kind is None


def test_attempt_health_recent_failures_dominate(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-rm",
        output_dir=str(tmp_path),
        mode="fresh",
    )
    candidate_id = store.ensure_candidate(
        source="linkedin", brief_id="brief-rm", identity_key="alice"
    )

    now = datetime.now(timezone.utc)
    for _ in range(8):
        _direct_insert_attempt(
            db_path,
            run_id=run_id,
            candidate_id=candidate_id,
            status="failed",
            failure_kind="http_429",
            started_at=now.isoformat(),
        )
    _direct_insert_attempt(
        db_path,
        run_id=run_id,
        candidate_id=candidate_id,
        status="failed",
        failure_kind="timeout",
        started_at=now.isoformat(),
    )
    _direct_insert_attempt(
        db_path,
        run_id=run_id,
        candidate_id=candidate_id,
        status="succeeded",
        started_at=now.isoformat(),
        ended_at=now.isoformat(),
    )

    health = attempt_health(db_path, run_id=run_id)
    assert health.total_attempts_in_window == 10
    assert health.failed_in_window == 9
    assert health.succeeded_in_window == 1
    assert health.dominant_failure_kind == "http_429"
    # last_success_age_s should be small (we just inserted a success).
    assert health.last_success_age_s is not None
    assert health.last_success_age_s < 5.0


def test_attempt_health_window_excludes_old_attempts(tmp_path: Path) -> None:
    """Per critique B2: parse-and-compare, not string-compare. An
    attempt outside the window (10 minutes ago) must not be counted in
    a 5-minute window."""

    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-rm",
        output_dir=str(tmp_path),
        mode="fresh",
    )
    candidate_id = store.ensure_candidate(
        source="linkedin", brief_id="brief-rm", identity_key="alice"
    )

    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    _direct_insert_attempt(
        db_path,
        run_id=run_id,
        candidate_id=candidate_id,
        status="failed",
        failure_kind="http_429",
        started_at=old,
    )

    health = attempt_health(db_path, run_id=run_id, window_minutes=5)
    assert health.total_attempts_in_window == 0
    assert health.dominant_failure_kind is None


# --- run_telemetry ----------------------------------------------------------
#
# Removed in Slice A.0 of the Multi-Agent Production Plan.
# ``run_telemetry`` / ``RunTelemetry`` / ``TelemetryAttempt`` / ``TelemetryEvent``
# were retired from ``shared.runtime_state.read_models`` in commit
# eb9d31f (legacy-monorepo import); the test functions that exercised
# them stayed behind and broke ``make validate`` collection. The
# block is removed in full here; if a future slice restores telemetry
# primitives, restore the imports + tests at the same time.


# --- intake-session read helpers --------------------------------------------


def _seed_intake_session(
    db_path: Path,
    *,
    role_title: str | None = None,
    archived: bool = False,
    state_json: str = "{}",
) -> int:
    """Insert a row directly into intake_sessions and return its id.

    Bypasses the writer-side helpers so this test module stays free of
    the RuntimeStateStore import path while still exercising the
    read helpers against realistic data. The row is created via direct
    SQL after the writer has set up the schema once at the module level
    (see fixture below)."""

    now = datetime.now(timezone.utc).isoformat()
    archived_at = now if archived else None
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """
            INSERT INTO intake_sessions(
                role_title, current_step, state_json,
                started_at, updated_at, archived_at
            ) VALUES (?, 'welcome', ?, ?, ?, ?)
            """,
            (role_title, state_json, now, now, archived_at),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def test_list_intake_sessions_excludes_archived(tmp_path: Path) -> None:
    """Mirrors the writer-side helper's contract: archived sessions must
    not surface in the active-list view."""

    db_path = tmp_path / "intake.sqlite3"
    RuntimeStateStore(db_path)
    active_id = _seed_intake_session(db_path, role_title="Active Role")
    _seed_intake_session(db_path, role_title="Archived Role", archived=True)

    sessions = read_models.list_intake_sessions(db_path)
    assert len(sessions) == 1
    assert sessions[0]["id"] == active_id
    assert sessions[0]["role_title"] == "Active Role"
    assert sessions[0]["archived_at"] is None


def test_list_intake_sessions_state_json_parsed(tmp_path: Path) -> None:
    db_path = tmp_path / "intake.sqlite3"
    RuntimeStateStore(db_path)
    _seed_intake_session(
        db_path,
        role_title="Parsed Role",
        state_json=json.dumps({"step": 1, "draft": {"k": "v"}}),
    )

    sessions = read_models.list_intake_sessions(db_path)
    assert sessions[0]["state_json"] == {"step": 1, "draft": {"k": "v"}}


def test_list_intake_sessions_corrupt_state_json_collapses_to_dict(
    tmp_path: Path,
) -> None:
    """Defensive parsing: a row with malformed JSON must not break the
    list view (matches the writer-side helper's posture)."""

    db_path = tmp_path / "intake.sqlite3"
    RuntimeStateStore(db_path)
    _seed_intake_session(
        db_path, role_title="Bad JSON", state_json="not valid json"
    )

    sessions = read_models.list_intake_sessions(db_path)
    assert sessions[0]["state_json"] == {}


def test_get_intake_session_returns_archived(tmp_path: Path) -> None:
    """The detail GET returns archived sessions too — recruiters use
    deep-links / unarchive flows that need to inspect the row."""

    db_path = tmp_path / "intake.sqlite3"
    RuntimeStateStore(db_path)
    archived_id = _seed_intake_session(
        db_path, role_title="Archived", archived=True
    )

    result = read_models.get_intake_session(db_path, session_id=archived_id)
    assert result is not None
    assert result["id"] == archived_id
    assert result["archived_at"] is not None


def test_get_intake_session_unknown_id_returns_none(tmp_path: Path) -> None:
    db_path = tmp_path / "intake.sqlite3"
    RuntimeStateStore(db_path)
    assert read_models.get_intake_session(db_path, session_id=999999) is None


# --- reflection read helpers ------------------------------------------------


def _seed_reflection_session(
    db_path: Path,
    *,
    brief_id: str,
    current_phase: str = "planning",
    completed: bool = False,
    discarded: bool = False,
    state_json: str = "{}",
) -> int:
    """Insert a reflection_sessions row directly. Same posture as the
    intake helper above."""

    now = datetime.now(timezone.utc).isoformat()
    completed_at = now if completed else None
    discarded_at = now if discarded else None
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """
            INSERT INTO reflection_sessions(
                brief_id, source_run_id, current_phase, state_json,
                steering_iterations, started_at, updated_at,
                completed_at, discarded_at
            ) VALUES (?, NULL, ?, ?, 0, ?, ?, ?, ?)
            """,
            (
                brief_id,
                current_phase,
                state_json,
                now,
                now,
                completed_at,
                discarded_at,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def test_get_active_reflection_picks_non_terminal(tmp_path: Path) -> None:
    db_path = tmp_path / "intake.sqlite3"
    RuntimeStateStore(db_path)
    active_id = _seed_reflection_session(db_path, brief_id="brief-a")
    _seed_reflection_session(
        db_path, brief_id="brief-a", completed=True
    )

    result = read_models.get_active_reflection_for_brief(
        db_path, brief_id="brief-a"
    )
    assert result is not None
    assert result["id"] == active_id
    assert result["completed_at"] is None
    assert result["discarded_at"] is None


def test_get_active_reflection_returns_none_when_all_terminal(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "intake.sqlite3"
    RuntimeStateStore(db_path)
    _seed_reflection_session(db_path, brief_id="brief-a", completed=True)
    _seed_reflection_session(db_path, brief_id="brief-a", discarded=True)

    assert read_models.get_active_reflection_for_brief(
        db_path, brief_id="brief-a"
    ) is None


def test_get_reflection_session_returns_terminal_rows(tmp_path: Path) -> None:
    """The GET endpoint serves both in-flight resume and post-mortem;
    discarded/completed rows must remain visible by id."""

    db_path = tmp_path / "intake.sqlite3"
    RuntimeStateStore(db_path)
    discarded_id = _seed_reflection_session(
        db_path, brief_id="brief-x", discarded=True
    )

    result = read_models.get_reflection_session(
        db_path, session_id=discarded_id
    )
    assert result is not None
    assert result["id"] == discarded_id
    assert result["discarded_at"] is not None


def test_get_reflection_session_unknown_id_returns_none(tmp_path: Path) -> None:
    db_path = tmp_path / "intake.sqlite3"
    RuntimeStateStore(db_path)
    assert read_models.get_reflection_session(db_path, session_id=999999) is None


# Note: the orphan ``test_run_telemetry_filters_by_run_id`` at the
# bottom of this file was removed in Slice A.0 alongside the rest of
# the telemetry test block (see the deletion notice in the
# ``# --- run_telemetry ---`` section above).
