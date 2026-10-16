"""CLO-243 slice 1: a run that dies records why.

21 of 25 LinkedIn runs between 25 and 30 Sep 2026 ended
``error | fatal_runtime_error`` with ``stop_reason_detail = NULL``: every
FATAL path finished the run with the bare enum, and the exception's text
survived only in stdout. These tests pin the contract that replaced that:

- ``RuntimeStateStore.finish_run`` and ``RunSafetyCoordinator.finish_run``
  take an explicit ``stop_reason_detail`` that is stored single-line and
  length-capped, and carried in both finish events;
- each FATAL path in ``linkedin/evaluation_loop.py`` passes the exception's
  first line (or its type name when the message is empty), prefixed by
  where it happened;
- a post-run cleanup failure no longer overwrites an earlier non-normal
  stop reason; it is appended to the detail instead.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.governor import GovernorLimitReached, OperatorStopRequested, SessionExpired
from shared.runtime_state.store import RuntimeStateStore
from shared.safety import RunSafetyCoordinator, RunStopReason
from shared.schemas import SearchString
from tests.test_seam_strategy_execution import (
    _make_pipeline,
    _seed_run_full_resume_progress,
    _seed_runtime_judge_decisions,
)


_SEARCH_URL = (
    "https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
)


# ---------------------------------------------------------------------------
# Store and coordinator
# ---------------------------------------------------------------------------


def _store_with_run(tmp_path: Path) -> tuple[RuntimeStateStore, int]:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-1",
        output_dir=str(tmp_path),
        mode="fresh",
    )
    return store, run_id


def _events(store: RuntimeStateStore, run_id: int, event_type: str) -> list[dict]:
    with sqlite3.connect(str(store.db_path)) as conn:
        rows = conn.execute(
            "SELECT payload_json FROM events WHERE run_id = ? AND event_type = ? "
            "ORDER BY id",
            (run_id, event_type),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def test_store_finish_run_records_explicit_detail_on_row_and_event(tmp_path):
    store, run_id = _store_with_run(tmp_path)

    store.finish_run(
        run_id,
        "error",
        stop_reason=RunStopReason.FATAL_RUNTIME_ERROR,
        stop_reason_detail="cleanup: disconnect failed",
    )

    run = store.get_run(run_id)
    assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
    assert run["stop_reason_detail"] == "cleanup: disconnect failed"
    [finished] = _events(store, run_id, "run_finished")
    assert finished["stop_reason_detail"] == "cleanup: disconnect failed"


def test_store_finish_run_without_detail_keeps_canonical_null(tmp_path):
    store, run_id = _store_with_run(tmp_path)

    store.finish_run(run_id, "completed", stop_reason=RunStopReason.NORMAL)

    assert store.get_run(run_id)["stop_reason_detail"] is None


def test_store_finish_run_reduces_multiline_detail_to_first_line(tmp_path):
    store, run_id = _store_with_run(tmp_path)

    store.finish_run(
        run_id,
        "error",
        stop_reason=RunStopReason.FATAL_RUNTIME_ERROR,
        stop_reason_detail="\n  runtime: Boom: first line  \nsecond line\nthird",
    )

    assert store.get_run(run_id)["stop_reason_detail"] == "runtime: Boom: first line"


def test_store_finish_run_caps_detail_length(tmp_path):
    store, run_id = _store_with_run(tmp_path)

    store.finish_run(
        run_id,
        "error",
        stop_reason=RunStopReason.FATAL_RUNTIME_ERROR,
        stop_reason_detail="runtime: " + "x" * 5000,
    )

    detail = store.get_run(run_id)["stop_reason_detail"]
    assert len(detail) == 500
    assert detail.startswith("runtime: xxx")
    assert detail.count("…") == 1


def test_store_finish_run_cap_keeps_the_appended_tail(tmp_path):
    """A cleanup or snapshot failure is appended after the runtime detail.
    Capping must not cut that tail off a long runtime line: both why the run
    ended (head) and what failed last (tail) survive."""

    store, run_id = _store_with_run(tmp_path)

    store.finish_run(
        run_id,
        "error",
        stop_reason=RunStopReason.FATAL_RUNTIME_ERROR,
        stop_reason_detail="runtime: Boom: " + "x" * 800 + "; cleanup: disconnect failed",
    )

    detail = store.get_run(run_id)["stop_reason_detail"]
    assert len(detail) == 500
    assert detail.startswith("runtime: Boom: xxx")
    assert detail.endswith("xxx; cleanup: disconnect failed")
    assert detail.count("…") == 1


def test_store_finish_run_blank_detail_records_nothing(tmp_path):
    store, run_id = _store_with_run(tmp_path)

    store.finish_run(
        run_id,
        "error",
        stop_reason=RunStopReason.FATAL_RUNTIME_ERROR,
        stop_reason_detail="   \n  ",
    )

    assert store.get_run(run_id)["stop_reason_detail"] is None


def test_store_finish_run_merges_noncanonical_reason_with_explicit_detail(tmp_path):
    store, run_id = _store_with_run(tmp_path)

    store.finish_run(
        run_id,
        "error",
        stop_reason="fatal_runtime_error: green_but_useless",
        stop_reason_detail="cleanup: disconnect failed",
    )

    run = store.get_run(run_id)
    assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
    assert run["stop_reason_detail"] == (
        "fatal_runtime_error: green_but_useless; cleanup: disconnect failed"
    )


def test_coordinator_finish_run_carries_detail_in_row_and_both_events(tmp_path):
    store, run_id = _store_with_run(tmp_path)
    coordinator = RunSafetyCoordinator(
        store=store,
        output_dir=tmp_path,
        source="linkedin",
        brief_id="brief-1",
    )

    coordinator.finish_run(
        run_id=run_id,
        status="error",
        stop_reason=RunStopReason.FATAL_RUNTIME_ERROR,
        stop_reason_detail="setup_abort: navigate_to_search failed\nTraceback ...",
    )

    run = store.get_run(run_id)
    assert run["status"] == "error"
    assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
    assert run["stop_reason_detail"] == "setup_abort: navigate_to_search failed"
    [stop_event] = _events(store, run_id, "run_stop_reason")
    assert stop_event["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
    assert stop_event["stop_reason_detail"] == "setup_abort: navigate_to_search failed"
    [finished] = _events(store, run_id, "run_finished")
    assert finished["stop_reason_detail"] == "setup_abort: navigate_to_search failed"


def test_coordinator_finish_run_without_detail_is_unchanged(tmp_path):
    store, run_id = _store_with_run(tmp_path)
    coordinator = RunSafetyCoordinator(
        store=store,
        output_dir=tmp_path,
        source="linkedin",
        brief_id="brief-1",
    )

    coordinator.finish_run(
        run_id=run_id,
        status="interrupted",
        stop_reason=RunStopReason.SESSION_EXPIRED,
    )

    assert store.get_run(run_id)["stop_reason_detail"] is None
    [stop_event] = _events(store, run_id, "run_stop_reason")
    assert "stop_reason_detail" not in stop_event


# ---------------------------------------------------------------------------
# run_full FATAL paths
# ---------------------------------------------------------------------------


def _resumable_pipeline(td: str):
    p = _make_pipeline(td)
    _seed_run_full_resume_progress(
        p,
        [SearchString(id=1, name="one", boolean="a", status="queued")],
    )
    p.state.context.browser.connect = AsyncMock()
    p.state.context.browser.disconnect = AsyncMock()
    p.state.context.browser.page = MagicMock(url=_SEARCH_URL)
    p.state._run_report_service._print_session_summary = MagicMock()
    p.state._run_report_service._print_summary = MagicMock()
    p.state._run_report_service._generate_run_report = MagicMock()
    p._finalize_run_snapshot = MagicMock(return_value=Path(td, "frozen-run"))
    p._enrich_run_snapshot = MagicMock()
    return p


def _run(p):
    return p.state.context._runtime_state.get_run(p.state.context._runtime_run_id)


def _finish_events(p) -> list[dict]:
    return _events(
        p.state.context._runtime_state,
        p.state.context._runtime_run_id,
        "run_stop_reason",
    )


def test_runtime_failure_records_type_and_first_line():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p.state.driver._process_string = AsyncMock(
            side_effect=RuntimeError(
                "navigate_to_search failed: sidebar elements not found after 16s\n"
                "Call log:\n  - waiting for locator"
            )
        )

        with pytest.raises(RuntimeError, match="sidebar elements not found"):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        expected = (
            "runtime: RuntimeError: navigate_to_search failed: "
            "sidebar elements not found after 16s"
        )
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == expected
        assert _finish_events(p)[-1]["stop_reason_detail"] == expected


def test_runtime_failure_with_empty_message_records_type_name():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p.state.driver._process_string = AsyncMock(side_effect=TimeoutError())

        with pytest.raises(TimeoutError):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == "runtime: TimeoutError"


def test_uncaught_base_exception_records_runtime_detail():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p.state.driver._process_string = AsyncMock(
            side_effect=SystemExit("synthetic process exit")
        )

        with pytest.raises(SystemExit):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == "runtime: SystemExit: synthetic process exit"


def test_setup_abort_records_first_line():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p.state.context.browser.connect = AsyncMock(
            side_effect=RuntimeError("connect failed\nsecond line of noise")
        )

        with pytest.raises(RuntimeError, match="connect failed"):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == "setup_abort: connect failed"
        assert _finish_events(p)[-1]["stop_reason_detail"] == (
            "setup_abort: connect failed"
        )


def test_setup_abort_with_empty_message_records_type_name():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p.state.context.browser.connect = AsyncMock(side_effect=ConnectionError())

        with pytest.raises(ConnectionError):
            asyncio.run(p.run_full(resume=True))

        assert _run(p)["stop_reason_detail"] == "setup_abort: ConnectionError"


def _save_one(p):
    async def process_with_save(search_string, _progress):
        _seed_runtime_judge_decisions(p, ["SAVE"])
        search_string.saves.append("candidate-save")

    return process_with_save


def test_cleanup_failure_after_clean_run_records_detail():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p.state.context.browser.disconnect = AsyncMock(
            side_effect=RuntimeError("disconnect failed\nTarget closed")
        )
        p.state.driver._process_string = _save_one(p)

        with pytest.raises(RuntimeError, match="disconnect failed"):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == "cleanup: disconnect failed"
        assert _finish_events(p)[-1]["stop_reason_detail"] == (
            "cleanup: disconnect failed"
        )


def test_cleanup_failure_with_empty_message_records_type_name():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p.state.context.browser.disconnect = AsyncMock(side_effect=RuntimeError())
        p.state.driver._process_string = _save_one(p)

        with pytest.raises(RuntimeError):
            asyncio.run(p.run_full(resume=True))

        assert _run(p)["stop_reason_detail"] == "cleanup: RuntimeError"


@pytest.mark.parametrize(
    ("raise_exc", "expected_status", "expected_reason"),
    [
        (
            lambda: SessionExpired("session_duration_cap"),
            "interrupted",
            RunStopReason.SESSION_EXPIRED,
        ),
        (
            lambda: GovernorLimitReached("blocked_or_rate_limited"),
            "governor_limit_reached",
            RunStopReason.GOVERNOR_LIMIT,
        ),
        (
            lambda: OperatorStopRequested(),
            "interrupted",
            RunStopReason.OPERATOR_STOP,
        ),
    ],
    ids=["session_expired", "governor_limit", "operator_stop"],
)
def test_cleanup_failure_keeps_earlier_non_normal_stop_reason(
    raise_exc, expected_status, expected_reason
):
    """The stop reason answers "why did the run end". A browser teardown that
    fails after a session cap already ended the run does not change that
    answer; overwriting it with fatal_runtime_error hid the real cause."""

    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p.state.context.browser.disconnect = AsyncMock(
            side_effect=RuntimeError("disconnect failed")
        )
        exc = raise_exc()

        async def stop(_search_string, _progress):
            raise exc

        p.state.driver._process_string = stop

        with pytest.raises(type(exc)):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        assert run["status"] == expected_status
        assert run["stop_reason"] == expected_reason
        assert run["stop_reason_detail"] == "cleanup: disconnect failed"
        last = _finish_events(p)[-1]
        assert last["status"] == expected_status
        assert last["stop_reason"] == expected_reason
        assert last["stop_reason_detail"] == "cleanup: disconnect failed"


def test_cleanup_failure_after_runtime_failure_appends_to_detail():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p.state.context.browser.disconnect = AsyncMock(
            side_effect=RuntimeError("disconnect failed")
        )
        p.state.driver._process_string = AsyncMock(
            side_effect=RuntimeError("string failed")
        )

        with pytest.raises(RuntimeError, match="string failed"):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == (
            "runtime: RuntimeError: string failed; cleanup: disconnect failed"
        )


def test_snapshot_failure_records_finalization_detail():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p._finalize_run_snapshot = MagicMock(side_effect=RuntimeError("snapshot exploded"))
        p.state.driver._process_string = _save_one(p)

        with pytest.raises(RuntimeError, match="snapshot exploded"):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == (
            "finalization: immutable run snapshot: snapshot exploded"
        )


def test_snapshot_failure_after_runtime_failure_appends_to_detail():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p._finalize_run_snapshot = MagicMock(return_value=None)
        p.state.driver._process_string = AsyncMock(
            side_effect=RuntimeError("string failed")
        )

        with pytest.raises(RuntimeError, match="string failed"):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == (
            "runtime: RuntimeError: string failed; "
            "finalization: immutable LinkedIn run snapshot did not return a path"
        )


def test_snapshot_returning_no_path_on_clean_run_records_fallback_detail():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p._finalize_run_snapshot = MagicMock(return_value=None)
        p.state.driver._process_string = _save_one(p)

        with pytest.raises(RuntimeError, match="did not return a path"):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == (
            "finalization: immutable LinkedIn run snapshot did not return a path"
        )


@pytest.mark.parametrize(
    ("raise_exc", "expected_status", "expected_reason"),
    [
        (
            lambda: SessionExpired("session_duration_cap"),
            "interrupted",
            RunStopReason.SESSION_EXPIRED,
        ),
        (
            lambda: GovernorLimitReached("blocked_or_rate_limited"),
            "governor_limit_reached",
            RunStopReason.GOVERNOR_LIMIT,
        ),
        (
            lambda: OperatorStopRequested(),
            "interrupted",
            RunStopReason.OPERATOR_STOP,
        ),
    ],
    ids=["session_expired", "governor_limit", "operator_stop"],
)
def test_snapshot_failure_keeps_earlier_non_normal_stop_reason(
    raise_exc, expected_status, expected_reason
):
    """Same rule as cleanup: freezing the snapshot happens after the run
    already ended for a non-normal reason, so a snapshot failure is appended
    to the detail instead of overwriting that reason with fatal."""

    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p._finalize_run_snapshot = MagicMock(
            side_effect=RuntimeError("snapshot exploded")
        )
        exc = raise_exc()

        async def stop(_search_string, _progress):
            raise exc

        p.state.driver._process_string = stop

        with pytest.raises(type(exc)):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        expected = "finalization: immutable run snapshot: snapshot exploded"
        assert run["status"] == expected_status
        assert run["stop_reason"] == expected_reason
        assert run["stop_reason_detail"] == expected
        last = _finish_events(p)[-1]
        assert last["status"] == expected_status
        assert last["stop_reason"] == expected_reason
        assert last["stop_reason_detail"] == expected


def test_snapshot_failure_after_honesty_downgrade_keeps_its_reason():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="done")],
        )
        p.state.context.browser.connect = AsyncMock()
        p.state.context.browser.disconnect = AsyncMock()
        p.state.context.browser.page = MagicMock(url=_SEARCH_URL)
        p.state.driver._process_string = AsyncMock()
        p.state._run_report_service._print_session_summary = MagicMock()
        p.state._run_report_service._print_summary = MagicMock()
        p.state._run_report_service._generate_run_report = MagicMock()
        p._finalize_run_snapshot = MagicMock(
            side_effect=RuntimeError("snapshot exploded")
        )
        p._enrich_run_snapshot = MagicMock()

        with pytest.raises(RuntimeError, match="LinkedIn completion rejected"):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == (
            "fatal_runtime_error: green_but_useless; "
            "finalization: immutable run snapshot: snapshot exploded"
        )


def test_finalization_layer_failure_on_clean_run_records_layer_detail():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        save_one = _save_one(p)

        async def save_then_break_usage_ledger(search_string, progress):
            await save_one(search_string, progress)
            usage_cm = MagicMock()
            usage_cm.__exit__.side_effect = RuntimeError("usage ledger locked\nmore")
            p.state._llm_usage_cm = usage_cm

        p.state.driver._process_string = save_then_break_usage_ledger

        with pytest.raises(RuntimeError, match="usage ledger locked"):
            asyncio.run(p.run_full(resume=True))

        run = _run(p)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == (
            "finalization: LLM usage finalization: usage ledger locked"
        )


def test_runtime_detail_names_the_failure_recorded_in_a_note():
    """The post-recovery branch re-raises the ORIGINAL disconnect with the
    resume failure that actually made the run fatal attached as a note; the
    detail must name that note too, not only the first cause."""

    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        original = RuntimeError("original failure\nnoise")
        original.add_note(
            "Browser recovery/resume failed: ValueError: resume exploded\nmore"
        )
        p.state.driver._process_string = AsyncMock(side_effect=original)

        with pytest.raises(RuntimeError, match="original failure"):
            asyncio.run(p.run_full(resume=True))

        assert _run(p)["stop_reason_detail"] == (
            "runtime: RuntimeError: original failure | "
            "Browser recovery/resume failed: ValueError: resume exploded"
        )


def test_cleanup_failure_after_long_runtime_failure_keeps_cleanup_text():
    with tempfile.TemporaryDirectory() as td:
        p = _resumable_pipeline(td)
        p.state.context.browser.disconnect = AsyncMock(
            side_effect=RuntimeError("disconnect failed")
        )
        p.state.driver._process_string = AsyncMock(
            side_effect=RuntimeError("string failed " + "y" * 700)
        )

        with pytest.raises(RuntimeError, match="string failed"):
            asyncio.run(p.run_full(resume=True))

        detail = _run(p)["stop_reason_detail"]
        assert len(detail) == 500
        assert detail.startswith("runtime: RuntimeError: string failed yyy")
        assert detail.endswith("; cleanup: disconnect failed")
