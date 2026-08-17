"""M2 outcome monitor tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.observability_monitors import (
    OutcomeMonitorRecord,
    baseline_rates,
    compute_run_health,
    current_baseline_rates,
    emit_outcome_monitors,
)
from shared.runtime_state import RuntimeStateStore


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


def _start_run(store: RuntimeStateStore, tmp_path: Path, *, brief_id: str) -> int:
    return store.start_run(
        source="linkedin",
        brief_id=brief_id,
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": brief_id},
    )


def _monitor_record(**overrides: object) -> OutcomeMonitorRecord:
    values: dict[str, object] = {
        "db_path": "/tmp/runtime_state.sqlite3",
        "run_id": 1,
        "source": "linkedin",
        "brief_id": "brief",
        "status": "completed",
        "started_at": "2026-06-17T00:00:00Z",
        "ended_at": "2026-06-17T00:01:00Z",
        "all_spans_ok": True,
        "candidates_saved": 0,
        "green_but_useless": False,
        "judge_decisions": 1,
        "judge_parse_failures": 0,
        "judge_parse_failure_rate": 0.0,
    }
    values.update(overrides)
    return OutcomeMonitorRecord(**values)  # type: ignore[arg-type]


def test_green_but_useless_detector_fires_on_seeded_ok_zero_save_run(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, brief_id="brief-green-zero")
    store.record_event(
        run_id=run_id,
        event_type="pipeline_start",
        payload={"seed": "green"},
    )
    store.record_event(
        run_id=run_id,
        event_type="pipeline_end",
        payload={"seed": "green"},
    )
    store.finish_run(run_id, "completed")

    [record] = emit_outcome_monitors([store.db_path])

    assert record.run_id == run_id
    assert record.all_spans_ok is True
    assert record.candidates_saved == 0
    assert record.green_but_useless is True


def test_outcome_monitors_emit_per_run_and_baseline_rates(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    green_run = _start_run(store, tmp_path, brief_id="brief-green")
    store.record_event(run_id=green_run, event_type="pipeline_start")
    store.record_event(run_id=green_run, event_type="pipeline_end")
    store.finish_run(green_run, "completed")

    useful_run = _start_run(store, tmp_path, brief_id="brief-useful")
    work_unit_id = store.upsert_work_unit(
        run_id=useful_run,
        source="linkedin",
        brief_id="brief-useful",
        kind="linkedin_string",
        source_unit_id="1",
        display_name="String 1",
        ordering_index=0,
        status="done",
        counters={"saves_count": 1},
    )
    store.record_candidate_discovery(
        run_id=useful_run,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id="brief-useful",
        identity_key="parse-failure-candidate",
        display_name="Parse Failure",
        profile_url="https://example.test/parse",
    )
    store.set_candidate_state(
        run_id=useful_run,
        source="linkedin",
        brief_id="brief-useful",
        identity_key="parse-failure-candidate",
        new_state="failed_terminal",
        terminal_decision="PARSE_FAILURE",
        terminal_payload={"full_decision": {"decision": "PARSE_FAILURE"}},
        last_work_unit_id=work_unit_id,
    )
    store.record_candidate_discovery(
        run_id=useful_run,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id="brief-useful",
        identity_key="reject-candidate",
        display_name="Reject",
        profile_url="https://example.test/reject",
    )
    store.set_candidate_state(
        run_id=useful_run,
        source="linkedin",
        brief_id="brief-useful",
        identity_key="reject-candidate",
        new_state="failed_terminal",
        terminal_decision="REJECT",
        terminal_payload={"full_decision": {"decision": "REJECT"}},
        last_work_unit_id=work_unit_id,
    )
    store.finish_run(useful_run, "completed")

    records = emit_outcome_monitors([store.db_path])
    assert {record.run_id for record in records} == {green_run, useful_run}

    useful = next(record for record in records if record.run_id == useful_run)
    assert useful.candidates_saved == 1
    assert useful.green_but_useless is False
    assert useful.judge_decisions == 2
    assert useful.judge_parse_failures == 1
    assert useful.judge_parse_failure_rate == 0.5

    rates = baseline_rates(records)
    assert rates.runs_measured == 2
    assert rates.green_but_useless_runs == 1
    assert rates.green_but_useless_rate == 0.5
    assert rates.judge_decisions == 2
    assert rates.judge_parse_failures == 1
    assert rates.judge_parse_failure_rate == 0.5


def test_baseline_rates_rejects_malformed_monitor_records() -> None:
    with pytest.raises(ValueError, match=r"records\[0\] must be an OutcomeMonitorRecord"):
        baseline_rates([{"green_but_useless": True}])  # type: ignore[list-item]

    with pytest.raises(ValueError, match=r"records\[0\].green_but_useless must be a boolean"):
        baseline_rates([_monitor_record(green_but_useless="false")])

    with pytest.raises(ValueError, match=r"records\[0\].judge_decisions must be a non-negative integer"):
        baseline_rates([_monitor_record(judge_decisions="1")])

    with pytest.raises(ValueError, match=r"records\[0\].judge_parse_failures must be a non-negative integer"):
        baseline_rates([_monitor_record(judge_parse_failures=True)])

    with pytest.raises(
        ValueError,
        match=r"records\[0\].judge_parse_failures cannot exceed judge_decisions",
    ):
        baseline_rates(
            [
                _monitor_record(
                    judge_decisions=1,
                    judge_parse_failures=2,
                )
            ]
        )


def test_parse_failure_rate_ignores_malformed_candidate_payload_text(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, brief_id="brief-malformed-candidate")
    work_unit_id = store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-malformed-candidate",
        kind="linkedin_string",
        source_unit_id="1",
        display_name="String 1",
        ordering_index=0,
        status="done",
    )
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id="brief-malformed-candidate",
        identity_key="reject-with-malformed-payload",
        display_name="Malformed",
        profile_url="https://example.test/malformed",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-malformed-candidate",
        identity_key="reject-with-malformed-payload",
        new_state="failed_terminal",
        terminal_decision="REJECT",
        terminal_payload={"full_decision": {"decision": "REJECT"}},
        last_work_unit_id=work_unit_id,
    )
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id="brief-malformed-candidate",
        identity_key="reject-with-typed-parse-payload",
        display_name="Typed Parse",
        profile_url="https://example.test/typed-parse",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-malformed-candidate",
        identity_key="reject-with-typed-parse-payload",
        new_state="failed_terminal",
        terminal_decision="REJECT",
        terminal_payload={"full_decision": {"decision": "PARSE_FAILURE"}},
        last_work_unit_id=work_unit_id,
    )
    with store.connect() as conn:
        conn.execute(
            """
            UPDATE candidates
            SET terminal_payload_json = ?
            WHERE identity_key = ?
            """,
            ("not-json PARSE_FAILURE", "reject-with-malformed-payload"),
        )
    store.finish_run(run_id, "completed")

    [record] = emit_outcome_monitors([store.db_path])

    assert record.judge_decisions == 2
    assert record.judge_parse_failures == 1
    assert record.judge_parse_failure_rate == 0.5


def test_parse_failure_rate_ignores_malformed_attempt_payload_text(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, brief_id="brief-malformed-attempt")
    attempt_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-malformed-attempt",
        identity_key="attempt-with-malformed-payload",
        stage="facial",
        payload={"decision": "REJECT"},
        display_name="Malformed Attempt",
        profile_url="https://example.test/malformed-attempt",
    )
    with store.connect() as conn:
        conn.execute(
            """
            UPDATE candidate_attempts
            SET payload_json = ?
            WHERE id = ?
            """,
            ("not-json PARSE_FAILURE", attempt_id),
        )
    store.finish_run(run_id, "completed")

    [record] = emit_outcome_monitors([store.db_path])

    assert record.judge_decisions == 0
    assert record.judge_parse_failures == 0
    assert record.judge_parse_failure_rate == 0.0


def test_parse_failure_rate_skips_contained_resume_skip_attempt(
    tmp_path: Path,
) -> None:
    """A resume abandon is a synthetic settle, not a judge that failed to parse.

    ``Pipeline._abandon_unrecoverable_pending_full`` records a JUDGMENT_FAILURE
    to settle a pending review whose Recruiter card the resume could not
    re-match. No judge ever ran, so grading it here would report a provider
    parse problem that did not happen — and would trip the parse-failure
    baseline on exactly the runs that were already limping.
    """

    store = _make_store(tmp_path)
    brief_id = "brief-abandon-attempt"
    run_id = _start_run(store, tmp_path, brief_id=brief_id)
    store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key="real-reject",
        stage="full",
        payload={"full_decision": {"decision": "REJECT"}},
        display_name="Real Reject",
        profile_url="https://example.test/real-reject",
    )
    store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key="contained-skip",
        stage="full",
        payload={
            "full_decision": {"decision": "JUDGMENT_FAILURE"},
            "pending_full_recovery_abandoned": True,
            "abandon_reason": "unmatched_profile",
        },
        display_name="Contained Skip",
        profile_url="https://example.test/contained-skip",
    )
    store.finish_run(run_id, "completed")

    [record] = emit_outcome_monitors([store.db_path])

    assert record.judge_decisions == 1
    assert record.judge_parse_failures == 0
    assert record.judge_parse_failure_rate == 0.0


def test_parse_failure_rate_skips_contained_resume_skip_candidate_terminal(
    tmp_path: Path,
) -> None:
    """The marker rides the candidate terminal payload too, so both readers skip.

    ``_judge_parse_counts`` only falls back to the candidate table when no
    attempt carries a decision, but the abandon's marker reaches
    ``terminal_payload_json`` verbatim — the fallback reader would otherwise
    report the same phantom parse failure.
    """

    store = _make_store(tmp_path)
    brief_id = "brief-abandon-candidate"
    run_id = _start_run(store, tmp_path, brief_id=brief_id)
    work_unit_id = store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        kind="linkedin_string",
        source_unit_id="1",
        display_name="String 1",
        ordering_index=0,
        status="done",
    )
    for identity_key, terminal_decision, terminal_payload in (
        (
            "real-reject",
            "REJECT",
            {"full_decision": {"decision": "REJECT"}},
        ),
        (
            "contained-skip",
            "JUDGMENT_FAILURE",
            {
                "full_decision": {"decision": "JUDGMENT_FAILURE"},
                "pending_full_recovery_abandoned": True,
                "abandon_reason": "unmatched_profile",
            },
        ),
    ):
        store.record_candidate_discovery(
            run_id=run_id,
            work_unit_id=work_unit_id,
            source="linkedin",
            brief_id=brief_id,
            identity_key=identity_key,
            display_name=identity_key,
            profile_url=f"https://example.test/{identity_key}",
        )
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id=brief_id,
            identity_key=identity_key,
            new_state="failed_terminal",
            terminal_decision=terminal_decision,
            terminal_payload=terminal_payload,
            last_work_unit_id=work_unit_id,
        )
    store.finish_run(run_id, "completed")

    [record] = emit_outcome_monitors([store.db_path])

    assert record.judge_decisions == 1
    assert record.judge_parse_failures == 0
    assert record.judge_parse_failure_rate == 0.0


def test_current_baseline_rates_handles_missing_output_root(tmp_path: Path) -> None:
    rates = current_baseline_rates(tmp_path / "missing", recent_limit=10)

    assert rates.runs_measured == 0
    assert rates.green_but_useless_rate == 0.0
    assert rates.judge_parse_failure_rate == 0.0


# ---------------------------------------------------------------------------
# P4.3.1 — compute_run_health (run finalization wiring)
# ---------------------------------------------------------------------------


def _seed_judge_decisions(
    store: RuntimeStateStore,
    run_id: int,
    brief_id: str,
    decisions: list[str],
) -> None:
    """Seed one candidate per entry in ``decisions`` as a judge terminal
    decision, mirroring the seeding pattern in
    test_parse_failure_rate_ignores_malformed_candidate_payload_text above."""
    work_unit_id = store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        kind="linkedin_string",
        source_unit_id="1",
        display_name="String 1",
        ordering_index=0,
        status="done",
    )
    for index, decision in enumerate(decisions):
        identity_key = f"candidate-{run_id}-{index}"
        store.record_candidate_discovery(
            run_id=run_id,
            work_unit_id=work_unit_id,
            source="linkedin",
            brief_id=brief_id,
            identity_key=identity_key,
            display_name=identity_key,
            profile_url=f"https://example.test/{identity_key}",
        )
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id=brief_id,
            identity_key=identity_key,
            new_state="failed_terminal",
            terminal_decision=decision,
            terminal_payload={"full_decision": {"decision": decision}},
            last_work_unit_id=work_unit_id,
        )


def test_compute_run_health_flags_green_but_useless(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, brief_id="brief-green")
    store.record_event(run_id=run_id, event_type="pipeline_start")
    store.record_event(run_id=run_id, event_type="pipeline_end")
    store.finish_run(run_id, "completed")

    health = compute_run_health(store.db_path, run_id)

    assert health is not None
    assert health.degraded is True
    assert health.degraded_reasons == ("green_but_useless",)


def test_compute_run_health_flags_parse_failure_rate_above_baseline(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)

    # Baseline: a prior, unrelated run with a clean parse-failure record.
    baseline_run = _start_run(store, tmp_path, brief_id="brief-baseline")
    _seed_judge_decisions(store, baseline_run, "brief-baseline", ["REJECT"] * 10)
    store.finish_run(baseline_run, "completed")

    # Current run: 4 of 5 decisions are PARSE_FAILURE (rate 0.8) — well
    # above both the baseline (0.0) and the floor (0.10).
    current_run = _start_run(store, tmp_path, brief_id="brief-current")
    _seed_judge_decisions(
        store,
        current_run,
        "brief-current",
        ["PARSE_FAILURE"] * 4 + ["REJECT"],
    )
    store.finish_run(current_run, "completed")

    health = compute_run_health(store.db_path, current_run)

    assert health is not None
    assert health.judge_decisions == 5
    assert health.judge_parse_failure_rate == 0.8
    assert health.baseline_judge_parse_failure_rate == 0.0
    assert health.degraded is True
    assert "judge_parse_failure_rate_above_baseline" in health.degraded_reasons


def test_compute_run_health_does_not_flag_high_rate_from_too_few_decisions(
    tmp_path: Path,
) -> None:
    """A high parse-failure RATE computed from a tiny sample (below
    min_decisions_for_baseline_check) must not trip the gate — one bad
    decision out of two is not a signal. A SAVE is included so
    green_but_useless doesn't also fire and confound the assertion."""
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, brief_id="brief-tiny-sample")
    _seed_judge_decisions(store, run_id, "brief-tiny-sample", ["PARSE_FAILURE", "SAVE"])
    store.finish_run(run_id, "completed")

    health = compute_run_health(store.db_path, run_id)

    assert health is not None
    assert health.judge_decisions == 2
    assert health.judge_parse_failure_rate == 0.5
    assert health.green_but_useless is False
    assert health.degraded is False
    assert health.degraded_reasons == ()


def test_compute_run_health_not_degraded_for_clean_useful_run(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, brief_id="brief-useful")
    _seed_judge_decisions(store, run_id, "brief-useful", ["REJECT"] * 8 + ["SAVE"])
    store.finish_run(run_id, "completed")

    health = compute_run_health(store.db_path, run_id)

    assert health is not None
    assert health.green_but_useless is False
    assert health.judge_parse_failure_rate == 0.0
    assert health.degraded is False
    assert health.degraded_reasons == ()


def test_compute_run_health_returns_none_when_run_not_in_db(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, brief_id="brief-real")
    store.finish_run(run_id, "completed")

    assert compute_run_health(store.db_path, run_id + 999) is None
