"""Invalidation-aware LinkedIn learning and projection regressions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from market_intelligence.engine import _extract_metrics_summary, _load_runtime_summary
from shared.adaptive import (
    AdaptiveAction,
    AdaptationDecision,
    record_adaptation_decision,
)
from shared.runtime_state.lane_metrics import lane_metrics_for_run
from shared.runtime_state.linkedin_contamination_reset import (
    ContaminationManifestError,
    ContaminationResetDisabledError,
    apply_linkedin_contamination_reset,
    plan_linkedin_contamination_reset,
)
from shared.runtime_state.projections import (
    project_linkedin_final_judgments,
    project_linkedin_search_memory,
)
from shared.runtime_state.store import LINKEDIN_STRING_KIND, RuntimeStateStore
from shared.schemas import SearchString


BRIEF_ID = "brief-contamination"


def _start_run(
    store: RuntimeStateStore,
    *,
    source: str = "linkedin",
    clone_from: int | None = None,
    recruiter_id: int | None = None,
) -> int:
    return store.start_run(
        source=source,
        brief_id=BRIEF_ID,
        output_dir=str(store.db_path.parent),
        mode="resume" if clone_from else "fresh",
        resume_state={"brief_name": BRIEF_ID},
        clone_work_units_from_run_id=clone_from,
        recruiter_id=recruiter_id,
    )


def _write_string(
    store: RuntimeStateStore,
    *,
    run_id: int,
    status: str,
    source_unit_id: str = "1",
    counters: dict | None = None,
) -> int:
    string = SearchString(
        id=int(source_unit_id),
        name=f"string-{source_unit_id}",
        boolean="python",
        family_key="builders",
        lane_id="lane-builders",
    )
    return store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        kind=LINKEDIN_STRING_KIND,
        source_unit_id=source_unit_id,
        display_name=string.name,
        ordering_index=int(source_unit_id),
        status=status,
        payload=string.to_dict(),
        checkpoint={"pages_reviewed": 1},
        metrics={"pages_reviewed": 1},
        family_key=string.family_key,
        counters=counters or {},
    )


def _execute_string(
    store: RuntimeStateStore,
    *,
    run_id: int,
    source_unit_id: str = "1",
    counters: dict | None = None,
) -> int:
    _write_string(
        store,
        run_id=run_id,
        source_unit_id=source_unit_id,
        status="in_progress",
        counters=counters,
    )
    return _write_string(
        store,
        run_id=run_id,
        source_unit_id=source_unit_id,
        status="done",
        counters=counters,
    )


def _record_terminal(
    store: RuntimeStateStore,
    *,
    run_id: int,
    work_unit_id: int,
    identity_key: str,
    decision: str,
) -> int:
    candidate_id = store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        display_name=identity_key,
        profile_url=identity_key,
    )
    for state in ("snippet_extracted", "facial_started"):
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id=BRIEF_ID,
            identity_key=identity_key,
            new_state=state,
            last_work_unit_id=work_unit_id,
        )
    facial_attempt_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        stage="facial",
        work_unit_id=work_unit_id,
    )
    store.finish_attempt_success(
        attempt_id=facial_attempt_id,
        new_state="facial_terminal",
        terminal_decision=None,
        payload={"facial_decision": {"decision": "FACIAL_YES"}},
        run_id=run_id,
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        new_state="full_started",
        last_work_unit_id=work_unit_id,
    )
    full_decision = {"decision": decision, "profile_url": identity_key}
    full_attempt_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        stage="full",
        work_unit_id=work_unit_id,
        payload={"full_decision": full_decision},
    )
    store.finish_attempt_success(
        attempt_id=full_attempt_id,
        new_state="full_terminal",
        terminal_decision=decision,
        payload={"full_decision": full_decision},
        terminal_payload={"full_decision": full_decision},
        run_id=run_id,
    )
    return candidate_id


def _invalidate(store: RuntimeStateStore, *, run_id: int, work_unit_id: int) -> None:
    store.record_event(
        event_type="linkedin_execution_invalidated",
        run_id=run_id,
        work_unit_id=work_unit_id,
        payload={"reason": "facet_contamination"},
    )


def _record_discovery(
    store: RuntimeStateStore,
    *,
    run_id: int,
    work_unit_id: int,
    identity_key: str,
) -> int:
    return store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        display_name=identity_key,
        profile_url=identity_key,
    )


def _record_open_full_attempt(
    store: RuntimeStateStore,
    *,
    run_id: int,
    work_unit_id: int,
    identity_key: str,
) -> tuple[int, int]:
    candidate_id = _record_discovery(
        store,
        run_id=run_id,
        work_unit_id=work_unit_id,
        identity_key=identity_key,
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        new_state="snippet_extracted",
        last_work_unit_id=work_unit_id,
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        new_state="full_started",
        last_work_unit_id=work_unit_id,
    )
    attempt_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        stage="full",
        work_unit_id=work_unit_id,
        payload={"keep": "attempt-payload"},
        source_cursor={"page": 3, "index": 7},
    )
    return candidate_id, attempt_id


def _table_rows(store: RuntimeStateStore, table: str) -> tuple[tuple, ...]:
    with store.connect() as conn:
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY id ASC").fetchall()
    return tuple(tuple(row) for row in rows)


def _row_by_id(store: RuntimeStateStore, table: str, row_id: int) -> dict:
    with store.connect() as conn:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE id = ?",
            (row_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def test_search_memory_ignores_clones_and_invalidated_executions_only_when_flagged(
    tmp_path: Path,
) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    parent_run = _start_run(store)
    parent_wu = _execute_string(store, run_id=parent_run)
    _start_run(store, clone_from=parent_run)
    _start_run(store, clone_from=parent_run)
    executed_child_run = _start_run(store, clone_from=parent_run)
    _execute_string(store, run_id=executed_child_run)

    with patch(
        "shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED",
        False,
    ):
        legacy = project_linkedin_search_memory(store, brief_id=BRIEF_ID)
    with patch(
        "shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED",
        True,
    ):
        valid_only = project_linkedin_search_memory(store, brief_id=BRIEF_ID)

    assert legacy["overall"]["strings_seen"] == 4
    assert valid_only["overall"]["strings_seen"] == 2

    _invalidate(store, run_id=parent_run, work_unit_id=parent_wu)

    with patch(
        "shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED",
        False,
    ):
        legacy_after = project_linkedin_search_memory(store, brief_id=BRIEF_ID)
    with patch(
        "shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED",
        True,
    ):
        valid_after = project_linkedin_search_memory(store, brief_id=BRIEF_ID)

    assert legacy_after["overall"]["strings_seen"] == 4
    assert valid_after["overall"]["strings_seen"] == 1


def test_invalidated_execution_teaches_nothing_but_keeps_protected_artifacts(
    tmp_path: Path,
) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = _start_run(store)
    work_unit_id = _execute_string(
        store,
        run_id=run_id,
        counters={
            "result_count": 120,
            "candidates_discovered": 3,
            "facial_yes_count": 3,
            "saves_count": 1,
            "rejected_count": 2,
        },
    )
    _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=work_unit_id,
        identity_key="https://linkedin.com/in/ordinary-reject",
        decision="REJECT",
    )
    _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=work_unit_id,
        identity_key="https://linkedin.com/in/protected-save",
        decision="SAVE",
    )
    touched_id = _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=work_unit_id,
        identity_key="https://linkedin.com/in/recruiter-touched",
        decision="REJECT",
    )
    store.set_candidate_user_status(touched_id, "shortlist")
    _invalidate(store, run_id=run_id, work_unit_id=work_unit_id)

    with patch(
        "shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED",
        False,
    ):
        legacy_judgments = project_linkedin_final_judgments(
            store,
            brief_id=BRIEF_ID,
            run_id=run_id,
        )
        legacy_lanes = lane_metrics_for_run(store.db_path, run_id=run_id)
        legacy_summary = _load_runtime_summary(store.db_path, run_id=run_id)

    assert {item["profile_url"] for item in legacy_judgments} == {
        "https://linkedin.com/in/ordinary-reject",
        "https://linkedin.com/in/protected-save",
        "https://linkedin.com/in/recruiter-touched",
    }
    assert legacy_lanes[0].result_count == 120
    assert legacy_lanes[0].save_count == 1
    assert legacy_lanes[0].reject_count == 2
    assert legacy_summary["candidate_volume"] == 3

    with patch(
        "shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED",
        True,
    ):
        protected_judgments = project_linkedin_final_judgments(
            store,
            brief_id=BRIEF_ID,
            run_id=run_id,
        )
        valid_lanes = lane_metrics_for_run(store.db_path, run_id=run_id)
        valid_summary = _load_runtime_summary(store.db_path, run_id=run_id)
        valid_metrics = _extract_metrics_summary(
            None,
            valid_summary,
            protected_judgments,
        )

    assert {item["profile_url"] for item in protected_judgments} == {
        "https://linkedin.com/in/protected-save",
        "https://linkedin.com/in/recruiter-touched",
    }
    assert valid_lanes == tuple()
    assert valid_summary["candidate_volume"] == 0
    assert valid_summary["work_units"] == []
    assert valid_metrics["candidate_volume"] == 0
    assert valid_metrics["saved"] == 0
    assert valid_metrics["rejected"] == 0


def test_recruiter_adaptation_aggregation_excludes_only_invalid_linkedin_work(
    tmp_path: Path,
) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    linkedin_run = _start_run(store, recruiter_id=7)
    linkedin_wu = _execute_string(store, run_id=linkedin_run)
    record_adaptation_decision(
        store,
        run_id=linkedin_run,
        work_unit_id=linkedin_wu,
        decision=AdaptationDecision(
            source="linkedin",
            action=AdaptiveAction.BROADEN,
            lane="linkedin_lane",
            rationale="contaminated",
        ),
    )
    _invalidate(store, run_id=linkedin_run, work_unit_id=linkedin_wu)

    valid_linkedin_run = _start_run(store, recruiter_id=7)
    valid_linkedin_wu = _execute_string(
        store,
        run_id=valid_linkedin_run,
        source_unit_id="2",
    )
    record_adaptation_decision(
        store,
        run_id=valid_linkedin_run,
        work_unit_id=valid_linkedin_wu,
        decision=AdaptationDecision(
            source="linkedin",
            action=AdaptiveAction.EXPERIMENT,
            lane="valid_linkedin_lane",
            rationale="valid",
        ),
    )
    record_adaptation_decision(
        store,
        run_id=valid_linkedin_run,
        decision=AdaptationDecision(
            source="linkedin",
            action=AdaptiveAction.BROADEN,
            lane="unlinked_linkedin_lane",
            rationale="unproved",
        ),
    )

    github_run = _start_run(store, source="github", recruiter_id=7)
    record_adaptation_decision(
        store,
        run_id=github_run,
        decision=AdaptationDecision(
            source="github",
            action=AdaptiveAction.EXPERIMENT,
            lane="repo_mining",
            rationale="valid",
        ),
    )

    with patch(
        "shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED",
        False,
    ):
        legacy = store.aggregate_recruiter_adaptations()
    with patch(
        "shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED",
        True,
    ):
        valid_only = store.aggregate_recruiter_adaptations()

    assert {row["lane"] for row in legacy} == {
        "linkedin_lane",
        "valid_linkedin_lane",
        "unlinked_linkedin_lane",
        "repo_mining",
    }
    # F4. An adaptation decision carries NO work unit — the sole production
    # writer is linkedin/orchestrator.py's gate-deferred branch, which records
    # it with search_string=None. Reading "no work unit" as "invalid execution"
    # therefore does not filter the LinkedIn aggregate, it empties it forever.
    # An unlinked decision is excluded only when its own run was reset.
    assert {row["lane"] for row in valid_only} == {
        "valid_linkedin_lane",
        "unlinked_linkedin_lane",
        "repo_mining",
    }


def test_unlinked_adaptation_from_a_reset_run_is_still_excluded(
    tmp_path: Path,
) -> None:
    """The loosening must not re-admit the false lessons a reset just scrubbed.

    A work-unit-less adaptation decision cannot be attributed to one execution,
    so it is judged at run grain: if anything in its run was invalidated, the
    decision was reasoned from contaminated evidence and must not teach.
    """
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")

    reset_run = _start_run(store, recruiter_id=7)
    reset_wu = _execute_string(store, run_id=reset_run)
    record_adaptation_decision(
        store,
        run_id=reset_run,
        decision=AdaptationDecision(
            source="linkedin",
            action=AdaptiveAction.BROADEN,
            lane="unlinked_from_reset_run",
            rationale="reasoned from contaminated counts",
        ),
    )
    _invalidate(store, run_id=reset_run, work_unit_id=reset_wu)

    clean_run = _start_run(store, recruiter_id=7)
    _execute_string(store, run_id=clean_run, source_unit_id="2")
    record_adaptation_decision(
        store,
        run_id=clean_run,
        decision=AdaptationDecision(
            source="linkedin",
            action=AdaptiveAction.EXPERIMENT,
            lane="unlinked_from_clean_run",
            rationale="valid",
        ),
    )

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        lanes = {row["lane"] for row in store.aggregate_recruiter_adaptations()}

    assert lanes == {"unlinked_from_clean_run"}


def test_work_unit_less_rows_are_dropped_unless_the_query_opts_in(
    tmp_path: Path,
) -> None:
    """The sharp edge behind F4, pinned so the next reader sees it.

    Five call sites LEFT JOIN work_units. A row whose work_unit_id is NULL joins
    to an all-NULL alias, so `wu.started_at IS NOT NULL` is false AND
    `wu.source != 'linkedin'` is NULL — the row vanishes whatever its source,
    including a GitHub one. That is the right answer where every production row
    does carry a work unit, and the wrong answer for adaptation decisions, which
    never do. `run_alias` is how a query says which case it is.
    """
    import sqlite3

    from shared.runtime_state.execution_validity import (
        INVALIDATION_EVENT_TYPE,
        valid_execution_sql,
    )

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE runs(id INTEGER PRIMARY KEY, source TEXT);
        CREATE TABLE work_units(id INTEGER PRIMARY KEY, run_id INTEGER, source TEXT, started_at TEXT);
        CREATE TABLE candidate_attempts(id INTEGER PRIMARY KEY, run_id INTEGER, work_unit_id INTEGER);
        CREATE TABLE events(work_unit_id INTEGER, event_type TEXT);
        INSERT INTO runs VALUES (1, 'linkedin'), (2, 'github');
        INSERT INTO work_units VALUES (1, 1, 'linkedin', '2026-01-01'), (2, 2, 'github', '2026-01-01');
        INSERT INTO candidate_attempts VALUES (10, 1, 1), (11, 2, 2), (12, 1, NULL), (13, 2, NULL);
        """
    )

    def _surviving(**kwargs) -> set[int]:
        rows = conn.execute(
            "SELECT ca.id FROM candidate_attempts ca "
            "JOIN runs ON runs.id = ca.run_id "
            "LEFT JOIN work_units AS wu ON wu.id = ca.work_unit_id "
            f"WHERE ({valid_execution_sql('wu', **kwargs)})"
        ).fetchall()
        return {row[0] for row in rows}

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", False):
        assert _surviving() == {10, 11, 12, 13}

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        # Strict by default — a work-unit-less row is unproved provenance, and
        # the GitHub one (13) goes with it because NULL != 'linkedin' is NULL.
        assert _surviving() == {10, 11}
        # Opted in, both come back: neither run carries an invalidation.
        assert _surviving(run_alias="runs") == {10, 11, 12, 13}
        conn.execute(
            "INSERT INTO events VALUES (1, ?)",
            (INVALIDATION_EVENT_TYPE,),
        )
        # Run 1 was reset, so its unlinked row (12) loses the opt-in again;
        # run 2 is untouched, so 13 keeps it.
        assert _surviving(run_alias="runs") == {11, 13}


def test_lane_metrics_requires_current_attempt_work_unit_provenance(
    tmp_path: Path,
) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = _start_run(store)
    work_unit_id = _execute_string(
        store,
        run_id=run_id,
        counters={"result_count": 20, "saves_count": 1},
    )
    candidate_id = _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=work_unit_id,
        identity_key="https://linkedin.com/in/ambiguous-provenance",
        decision="SAVE",
    )
    with store.connect() as conn:
        candidate = conn.execute(
            "SELECT last_attempt_id FROM candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
        conn.execute(
            "UPDATE candidate_attempts SET work_unit_id = NULL WHERE id = ?",
            (candidate["last_attempt_id"],),
        )

    with patch(
        "shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED",
        False,
    ):
        legacy = lane_metrics_for_run(store.db_path, run_id=run_id)
    with patch(
        "shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED",
        True,
    ):
        valid_only = lane_metrics_for_run(store.db_path, run_id=run_id)

    assert legacy[0].save_count == 1
    assert legacy[0].opened_count == 1
    assert legacy[0].evaluated_count == 1
    assert valid_only[0].result_count == 20
    assert valid_only[0].save_count == 0
    assert valid_only[0].opened_count == 0
    assert valid_only[0].evaluated_count == 0


def test_guarded_reset_releases_only_proved_dirty_unprotected_candidates(
    tmp_path: Path,
) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = _start_run(store)
    dirty_work_unit_id = _execute_string(store, run_id=run_id)
    clean_work_unit_id = _execute_string(
        store,
        run_id=run_id,
        source_unit_id="2",
    )

    dirty_reject_id = _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=dirty_work_unit_id,
        identity_key="https://linkedin.com/in/dirty-reject",
        decision="REJECT",
    )
    dirty_save_key = "https://linkedin.com/in/dirty-save"
    dirty_save_id = _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=dirty_work_unit_id,
        identity_key=dirty_save_key,
        decision="SAVE",
    )
    save_candidate = _row_by_id(store, "candidates", dirty_save_id)
    pending_save = store.begin_candidate_side_effect(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=dirty_save_key,
        attempt_id=int(save_candidate["last_attempt_id"]),
        effect_type="linkedin_save",
        idempotency_key="save",
        payload={"keep": "save-payload"},
    )
    save_effect_id = int(pending_save["side_effect"]["id"])
    store.complete_candidate_side_effect(
        side_effect_id=save_effect_id,
        status="succeeded",
        payload={"keep": "save-result"},
    )

    recruiter_touched_id = _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=dirty_work_unit_id,
        identity_key="https://linkedin.com/in/recruiter-touched-reset",
        decision="REJECT",
    )
    store.set_candidate_user_status(recruiter_touched_id, "shortlist")

    later_clean_key = "https://linkedin.com/in/later-clean"
    _record_discovery(
        store,
        run_id=run_id,
        work_unit_id=dirty_work_unit_id,
        identity_key=later_clean_key,
    )
    later_clean_id = _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=clean_work_unit_id,
        identity_key=later_clean_key,
        decision="REJECT",
    )

    ambiguous_key = "https://linkedin.com/in/ambiguous-reset"
    ambiguous_id = _record_discovery(
        store,
        run_id=run_id,
        work_unit_id=dirty_work_unit_id,
        identity_key=ambiguous_key,
    )
    store.mark_candidate_terminal_runtime(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=ambiguous_key,
        decision="REJECT",
        payload={"keep": "ambiguous"},
        last_work_unit_id=None,
    )

    open_candidate_id, open_attempt_id = _record_open_full_attempt(
        store,
        run_id=run_id,
        work_unit_id=dirty_work_unit_id,
        identity_key="https://linkedin.com/in/dirty-open",
    )
    unrelated_id = _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=clean_work_unit_id,
        identity_key="https://linkedin.com/in/unrelated-clean",
        decision="REJECT",
    )

    protected_candidate_ids = (
        dirty_save_id,
        recruiter_touched_id,
        later_clean_id,
        ambiguous_id,
        unrelated_id,
    )
    protected_candidates_before = {
        candidate_id: _row_by_id(store, "candidates", candidate_id)
        for candidate_id in protected_candidate_ids
    }
    dirty_reject_before = _row_by_id(store, "candidates", dirty_reject_id)
    open_candidate_before = _row_by_id(store, "candidates", open_candidate_id)
    open_attempt_before = _row_by_id(store, "candidate_attempts", open_attempt_id)
    save_effect_before = _row_by_id(store, "side_effects", save_effect_id)
    completed_attempts_before = {
        row[0]: row
        for row in _table_rows(store, "candidate_attempts")
        if row[7] != "started"
    }
    events_before = _table_rows(store, "events")
    receipt_events_before = _table_rows(store, "run_event_log")
    work_units_before = _table_rows(store, "work_units")

    with store.connect() as conn:
        plan = plan_linkedin_contamination_reset(conn, [(run_id, "1")])

    assert {item.candidate_id for item in plan.candidates_to_release} == {
        dirty_reject_id,
        open_candidate_id,
    }
    assert {item.attempt_id for item in plan.attempts_to_abandon} == {
        open_attempt_id
    }
    assert {
        (item.candidate_id, item.reason) for item in plan.untouched_candidates
    } >= {
        (dirty_save_id, "save_family"),
        (recruiter_touched_id, "recruiter_touched"),
        (later_clean_id, "later_clean"),
        (ambiguous_id, "ambiguous"),
    }

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        applied = apply_linkedin_contamination_reset(
            store,
            manifest=[(run_id, "1")],
            reset_id="reset-test",
        )

    assert applied == plan
    dirty_reject_after = _row_by_id(store, "candidates", dirty_reject_id)
    assert dirty_reject_after == {
        **dirty_reject_before,
        "current_lifecycle_state": "failed_retryable",
        "terminal_decision": None,
        "terminal_payload_json": "{}",
    }
    open_candidate_after = _row_by_id(store, "candidates", open_candidate_id)
    assert open_candidate_after == {
        **open_candidate_before,
        "current_lifecycle_state": "failed_retryable",
        "terminal_decision": None,
        "terminal_payload_json": "{}",
    }
    for candidate_id, before in protected_candidates_before.items():
        assert _row_by_id(store, "candidates", candidate_id) == before
    assert _row_by_id(store, "side_effects", save_effect_id) == save_effect_before

    open_attempt_after = _row_by_id(store, "candidate_attempts", open_attempt_id)
    assert open_attempt_after == {
        **open_attempt_before,
        "status": "failed",
        "failure_kind": "contamination_reset",
        "ended_at": open_attempt_after["ended_at"],
    }
    assert open_attempt_after["ended_at"] is not None
    completed_attempts_after = {
        row[0]: row for row in _table_rows(store, "candidate_attempts")
    }
    assert {
        attempt_id: completed_attempts_after[attempt_id]
        for attempt_id in completed_attempts_before
    } == completed_attempts_before
    assert _table_rows(store, "work_units") == work_units_before

    events_after_first_apply = _table_rows(store, "events")
    receipt_events_after_first_apply = _table_rows(store, "run_event_log")
    assert events_after_first_apply[: len(events_before)] == events_before
    assert receipt_events_after_first_apply[: len(receipt_events_before)] == (
        receipt_events_before
    )
    assert {row[5] for row in events_after_first_apply[len(events_before) :]} == {
        "linkedin_contamination_reset",
        "linkedin_execution_invalidated",
        "linkedin_candidate_released_for_contamination",
        "linkedin_attempt_abandoned_for_contamination",
    }
    store.verify_run_event_log(run_id)

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        second_apply = apply_linkedin_contamination_reset(
            store,
            manifest=[(run_id, "1")],
            reset_id="reset-test",
        )
    assert second_apply.is_noop is True
    assert _table_rows(store, "events") == events_after_first_apply
    assert _table_rows(store, "run_event_log") == receipt_events_after_first_apply

    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key="https://linkedin.com/in/dirty-open",
        new_state="full_started",
        last_work_unit_id=clean_work_unit_id,
    )
    replacement_attempt_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key="https://linkedin.com/in/dirty-open",
        stage="full",
        work_unit_id=clean_work_unit_id,
    )
    assert replacement_attempt_id > open_attempt_id


def test_guarded_reset_rolls_back_every_write_when_event_append_fails(
    tmp_path: Path,
) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = _start_run(store)
    work_unit_id = _execute_string(store, run_id=run_id)
    _record_terminal(
        store,
        run_id=run_id,
        work_unit_id=work_unit_id,
        identity_key="https://linkedin.com/in/rollback-reject",
        decision="REJECT",
    )
    tables = (
        "work_units",
        "candidates",
        "candidate_attempts",
        "events",
        "run_event_log",
        "side_effects",
    )
    before = {table: _table_rows(store, table) for table in tables}

    with pytest.raises(ContaminationResetDisabledError):
        apply_linkedin_contamination_reset(
            store,
            manifest=[(run_id, "1")],
            reset_id="disabled",
        )
    assert {table: _table_rows(store, table) for table in tables} == before

    original_record_event = store.record_event
    calls = 0

    def fail_after_second_append(**kwargs) -> None:
        nonlocal calls
        calls += 1
        original_record_event(**kwargs)
        if calls == 2:
            raise RuntimeError("injected reset failure")

    with (
        patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True),
        patch.object(store, "record_event", side_effect=fail_after_second_append),
        pytest.raises(RuntimeError, match="injected reset failure"),
    ):
        apply_linkedin_contamination_reset(
            store,
            manifest=[(run_id, "1")],
            reset_id="rollback",
        )

    assert {table: _table_rows(store, table) for table in tables} == before


def test_reset_manifest_rejects_cloned_unexecuted_work_units(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    parent_run_id = _start_run(store)
    _execute_string(store, run_id=parent_run_id)
    cloned_run_id = _start_run(store, clone_from=parent_run_id)

    with store.connect() as conn:
        with pytest.raises(
            ContaminationManifestError,
            match="not an executed LinkedIn string",
        ):
            plan_linkedin_contamination_reset(conn, [(cloned_run_id, "1")])


# ---------------------------------------------------------------------------
# F5 + durability — run-start protection for reset stores
#
# A store that carries a contamination reset is a repaired campaign: run
# start must refuse an incomplete reset (root event without its completion
# event), refuse any resume/clone whose ancestry contains an invalidated
# execution, and refuse to run at all with the reset flag off (the scrub is
# canonical state, not a read-time preference).
# ---------------------------------------------------------------------------


def _run_start_guard_symbols():
    from shared.runtime_state import execution_validity as validity_module
    from shared.runtime_state import (
        linkedin_contamination_reset as reset_module,
    )

    refused = getattr(validity_module, "ContaminationRunStartRefused", None)
    record_completed = getattr(
        reset_module, "record_contamination_reset_completed", None
    )
    assert refused is not None, (
        "ContaminationRunStartRefused must be a named execution_validity error"
    )
    assert record_completed is not None, (
        "record_contamination_reset_completed must exist on the reset service"
    )
    return refused, record_completed


def _applied_reset_store(tmp_path: Path, *, reset_id: str = "guard-reset"):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = _start_run(store)
    _execute_string(store, run_id=run_id)
    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        apply_linkedin_contamination_reset(
            store, manifest=[(run_id, "1")], reset_id=reset_id
        )
    return store, run_id


def test_apply_writes_root_event_and_completion_is_recordable(tmp_path: Path):
    _refused, record_completed = _run_start_guard_symbols()
    store, _run_id = _applied_reset_store(tmp_path, reset_id="root-evt")

    root_rows = [
        row
        for row in _table_rows(store, "events")
        if row[5] == "linkedin_contamination_reset"
    ]
    assert len(root_rows) == 1, "apply must write exactly one root reset event"

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        record_completed(store, reset_id="root-evt")
    completed_rows = [
        row
        for row in _table_rows(store, "events")
        if row[5] == "linkedin_contamination_reset_completed"
    ]
    assert len(completed_rows) == 1


def test_incomplete_reset_blocks_run_start_until_completion(tmp_path: Path):
    refused, record_completed = _run_start_guard_symbols()
    store, _run_id = _applied_reset_store(tmp_path, reset_id="incomplete")

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        with pytest.raises(refused) as exc_info:
            _start_run(store)
        assert exc_info.value.reason == "reset_incomplete"

        record_completed(store, reset_id="incomplete")
        fresh_run_id = _start_run(store)
    assert fresh_run_id


def test_flag_on_resume_or_clone_from_invalidated_ancestry_is_refused(
    tmp_path: Path,
):
    refused, record_completed = _run_start_guard_symbols()
    store, dirty_run_id = _applied_reset_store(tmp_path, reset_id="ancestry")

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        record_completed(store, reset_id="ancestry")

        with pytest.raises(refused) as clone_info:
            _start_run(store, clone_from=dirty_run_id)
        assert clone_info.value.reason == "invalidated_ancestry"

        with pytest.raises(refused) as resume_info:
            store.start_run(
                source="linkedin",
                brief_id=BRIEF_ID,
                output_dir=str(store.db_path.parent),
                mode="resume",
                resumed_from_run_id=dirty_run_id,
            )
        assert resume_info.value.reason == "invalidated_ancestry"

        # A brand-new run formed from the approved brief has no contaminated
        # ancestry and proceeds with newly initialized work units.
        fresh_run_id = _start_run(store)
        assert fresh_run_id

        # The taint follows the resume CHAIN: a clean run resumed from the
        # invalidated one still refuses as an ancestor.
        chained_run_id = store.start_run(
            source="linkedin",
            brief_id=BRIEF_ID,
            output_dir=str(store.db_path.parent),
            mode="resume",
            resumed_from_run_id=fresh_run_id,
        )
        with store.connect() as conn:
            conn.execute(
                "UPDATE runs SET resumed_from_run_id = ? WHERE id = ?",
                (dirty_run_id, fresh_run_id),
            )
        with pytest.raises(refused) as chain_info:
            store.start_run(
                source="linkedin",
                brief_id=BRIEF_ID,
                output_dir=str(store.db_path.parent),
                mode="resume",
                resumed_from_run_id=chained_run_id,
            )
        assert chain_info.value.reason == "invalidated_ancestry"


def test_reset_store_refuses_to_run_with_flag_off(tmp_path: Path):
    refused, record_completed = _run_start_guard_symbols()
    store, _run_id = _applied_reset_store(tmp_path, reset_id="durable")
    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        record_completed(store, reset_id="durable")

    # Flag off (the default in this suite): the reset is canonical state, so
    # the repaired campaign must refuse rather than silently re-admit
    # invalidated history.
    with pytest.raises(refused) as exc_info:
        _start_run(store)
    assert exc_info.value.reason == "reset_flag_required"


def test_untouched_store_run_start_is_unchanged_flag_off(tmp_path: Path):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    first_run_id = _start_run(store)
    _execute_string(store, run_id=first_run_id)
    resumed_run_id = _start_run(store, clone_from=first_run_id)
    assert resumed_run_id > first_run_id


def test_non_linkedin_sources_never_consult_the_reset_guard(tmp_path: Path):
    _refused, record_completed = _run_start_guard_symbols()
    store, _run_id = _applied_reset_store(tmp_path, reset_id="scoped")
    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        record_completed(store, reset_id="scoped")

    # A GitHub run against the same store is outside the LinkedIn reset's
    # jurisdiction, with the flag on or off.
    github_run_id = _start_run(store, source="github")
    assert github_run_id


# ---------------------------------------------------------------------------
# F6 — the live-advisory reader honors the run-log reset boundary
# ---------------------------------------------------------------------------


def test_live_advisory_ignores_pre_boundary_run_log_rows(tmp_path: Path):
    import json as json_module

    from market_intelligence.live_advisory import _load_recent_events

    run_log_path = tmp_path / "run_log.jsonl"
    rows = [{"event": "string_executed", "marker": index} for index in range(3)]
    rows.append({"event": "contamination_reset_boundary", "reset_id": "r1"})
    rows.append({"event": "string_executed", "marker": 99})
    run_log_path.write_text(
        "\n".join(json_module.dumps(row) for row in rows) + "\n"
    )

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        recent = _load_recent_events(tmp_path)
    assert [row.get("marker") for row in recent] == [99], (
        "flag-on advisories must not re-ingest pre-boundary contaminated rows"
    )

    # Flag off: today's window, byte-identical.
    recent_off = _load_recent_events(tmp_path)
    assert len(recent_off) == 5
