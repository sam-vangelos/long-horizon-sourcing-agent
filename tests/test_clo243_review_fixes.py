"""CLO-243 integration review fixes: where the three slices still lost a stop.

Each test builds the runtime DB through the integrated writers (a real
``run_full`` or ``RuntimeStateStore.finish_run``) and reads it back through
the watchdog, so it checks the whole chain: why the run stopped -> what the
row records -> what the alarm says.

1. A sign-in wall met inside browser recovery was downgraded to
   ``browser_disconnect_unrecovered`` with no detail, so the alarm said
   Chrome was lost instead of asking Sam to sign in.
2. The sign-in alarm always claimed the one-session page, whatever wall
   LinkedIn showed, and cut the detail before its instruction.
3. Sam's own Ctrl+C (or a cancel, a session cap or a governor stop) during
   setup was recorded as a fatal error and alarmed as a crash.
4. A cleanup or snapshot failure after a routine stop (session cap, plain
   governor cap) no longer flips the row to error, so it raised no alarm.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from linkedin.auth_wall import AUTH_WALL_SENTENCES, AuthWallError
from shared.governor import GovernorLimitReached, OperatorStopRequested, SessionExpired
from shared.runtime_state.store import RuntimeStateStore
from tests.test_linkedin_pipeline import _facet_containment_pipeline
from tools import cloris_watchdog as wd

WALL_URL = "https://www.linkedin.com/enterprise-authentication/sessions"
SIGN_IN_TITLE = "Cloris needs you to sign in to LinkedIn Recruiter"
PLAIN_WORDS = "LinkedIn Recruiter wants you to sign in again"
DISCONNECT = "Page.evaluate: Target page, context or browser has been closed"


def _probe() -> wd.Probe:
    return wd.Probe(
        pid_alive=lambda _pid: False,
        lock_holders=lambda _path: None,
        governor=lambda: wd.GovernorState(),
    )


def _scan(state_root: Path, home: Path) -> list[wd.Alarm]:
    return wd.scan(
        state_root,
        {"dirs": {}},
        home=home,
        now=datetime.now(timezone.utc),
        probe=_probe(),
    )


def _run_row(state_dir: Path) -> tuple:
    with sqlite3.connect(state_dir / wd.DB_FILENAME) as conn:
        return conn.execute(
            "SELECT status, stop_reason, stop_reason_detail FROM runs "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()


def _state_dir(tmp_path: Path) -> tuple[Path, Path]:
    state_root = tmp_path / "state"
    state_dir = state_root / "linkedin" / "test"
    state_dir.mkdir(parents=True)
    return state_root, state_dir


# --- 1. a wall met inside browser recovery ------------------------------------


def _disconnect_then_recover(p, recover_error: BaseException) -> None:
    p.state.surface._capture_recovery_snapshot = AsyncMock(return_value=MagicMock())
    p.state._recovery_service.recover = AsyncMock(side_effect=recover_error)

    async def disconnected(_search_string, _progress):
        raise RuntimeError(DISCONNECT)

    p.state.driver._process_string = disconnected


def test_a_wall_met_during_browser_recovery_is_recorded_as_the_wall(tmp_path):
    state_root, state_dir = _state_dir(tmp_path)
    p = _facet_containment_pipeline(str(state_dir), string_count=1)
    wall = AuthWallError("session_conflict", url=WALL_URL)
    _disconnect_then_recover(p, wall)

    with pytest.raises(AuthWallError) as raised:
        asyncio.run(p.run_full(resume=True))
    assert raised.value is wall

    status, stop_reason, detail = _run_row(state_dir)
    assert (status, stop_reason) == ("error", "fatal_runtime_error")
    assert detail.startswith(
        "runtime: AuthWallError: recruiter_reauth_required: session_conflict: "
    )

    alarms = _scan(state_root, tmp_path / "home")
    assert [alarm.kind for alarm in alarms] == ["stopped"]
    assert alarms[0].title == SIGN_IN_TITLE
    assert PLAIN_WORDS in alarms[0].body
    assert "lost the Chrome connection" not in alarms[0].body


def test_a_wall_quoted_by_a_recovery_navigation_failure_is_named(tmp_path):
    """A wall that lost its type (a navigation failure quoting the wall URL)
    is still surfaced as a typed AuthWallError with its variant."""
    _state_root, state_dir = _state_dir(tmp_path)
    p = _facet_containment_pipeline(str(state_dir), string_count=1)
    untyped = RuntimeError(f"navigation failed. Page URL: {WALL_URL}")
    _disconnect_then_recover(p, untyped)

    with pytest.raises(AuthWallError) as raised:
        asyncio.run(p.run_full(resume=True))
    assert raised.value.variant == "session_conflict"
    assert raised.value.__cause__ is untyped

    status, stop_reason, detail = _run_row(state_dir)
    assert (status, stop_reason) == ("error", "fatal_runtime_error")
    assert detail.startswith(
        "runtime: AuthWallError: recruiter_reauth_required: session_conflict: "
    )


def test_a_plain_recovery_failure_still_ends_as_an_unrecovered_disconnect(tmp_path):
    """The wall branch must not swallow the ordinary recovery failure."""
    _state_root, state_dir = _state_dir(tmp_path)
    p = _facet_containment_pipeline(str(state_dir), string_count=1)
    _disconnect_then_recover(p, RuntimeError("rebind failed"))

    with pytest.raises(RuntimeError, match="Target page"):
        asyncio.run(p.run_full(resume=True))

    status, stop_reason, _detail = _run_row(state_dir)
    assert (status, stop_reason) == ("interrupted", "browser_disconnect_unrecovered")


# --- 2. the sign-in alarm names the wall LinkedIn actually showed ------------


def _store_reauth_row(tmp_path: Path, detail: str) -> Path:
    state_root, state_dir = _state_dir(tmp_path)
    store = RuntimeStateStore(state_dir / wd.DB_FILENAME)
    run_id = store.start_run(
        source="linkedin", brief_id="spl", output_dir=str(state_dir), mode="fresh"
    )
    store.finish_run(
        run_id,
        "error",
        stop_reason="fatal_runtime_error",
        stop_reason_detail=detail,
    )
    return state_root


@pytest.mark.parametrize("variant", sorted(AUTH_WALL_SENTENCES))
def test_sign_in_alarm_names_the_wall_and_keeps_its_whole_sentence(tmp_path, variant):
    wall_line = str(AuthWallError(variant)).splitlines()[0]
    state_root = _store_reauth_row(tmp_path, f"runtime: AuthWallError: {wall_line}")

    alarms = _scan(state_root, tmp_path / "home")
    assert [alarm.kind for alarm in alarms] == ["stopped"]
    body = alarms[0].body
    assert alarms[0].title == SIGN_IN_TITLE
    assert PLAIN_WORDS in body
    assert ("one-session-at-a-time" in body) == (variant == "session_conflict")
    # The whole sentence, including the instruction at its end, survives.
    assert AUTH_WALL_SENTENCES[variant] in body
    if variant == "checkpoint":
        assert "backoff" in body.split("Detail:")[0]


# --- 3. a stop during setup keeps its real reason ------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (KeyboardInterrupt(), ("interrupted", "operator_stop")),
        (asyncio.CancelledError(), ("interrupted", "operator_stop")),
        (OperatorStopRequested(), ("interrupted", "operator_stop")),
        (SessionExpired(), ("interrupted", "session_expired")),
        (
            GovernorLimitReached("daily_profile_cap"),
            ("governor_limit_reached", "governor_limit"),
        ),
    ],
    ids=["ctrl_c", "cancel", "operator_stop", "session_cap", "governor_cap"],
)
def test_a_routine_stop_during_setup_is_not_recorded_as_a_crash(
    tmp_path, error, expected
):
    state_root, state_dir = _state_dir(tmp_path)
    p = _facet_containment_pipeline(str(state_dir), string_count=1)
    p.state.context.browser.connect = AsyncMock(side_effect=error)

    with pytest.raises(type(error)):
        asyncio.run(p.run_full(resume=True))

    status, stop_reason, detail = _run_row(state_dir)
    assert (status, stop_reason) == expected
    assert detail is not None and detail.startswith("setup_abort: ")
    assert _scan(state_root, tmp_path / "home") == []


def test_a_real_setup_failure_still_alarms_as_an_error(tmp_path):
    state_root, state_dir = _state_dir(tmp_path)
    p = _facet_containment_pipeline(str(state_dir), string_count=1)
    p.state.context.browser.connect = AsyncMock(side_effect=ConnectionError("no CDP"))

    with pytest.raises(ConnectionError):
        asyncio.run(p.run_full(resume=True))

    assert _run_row(state_dir) == (
        "error",
        "fatal_runtime_error",
        "setup_abort: no CDP",
    )
    assert [alarm.kind for alarm in _scan(state_root, tmp_path / "home")] == ["stopped"]


# --- 4. a teardown failure after a routine stop still alarms ------------------


def test_cleanup_failure_after_a_session_cap_alarms(tmp_path):
    state_root, state_dir = _state_dir(tmp_path)
    p = _facet_containment_pipeline(str(state_dir), string_count=1)

    async def cap_reached(_search_string, _progress):
        raise SessionExpired()

    p.state.driver._process_string = cap_reached
    p.state.context.browser.disconnect = AsyncMock(
        side_effect=RuntimeError("chrome would not let go")
    )

    with pytest.raises(SessionExpired):
        asyncio.run(p.run_full(resume=True))

    status, stop_reason, detail = _run_row(state_dir)
    assert (status, stop_reason) == ("interrupted", "session_expired")
    assert "cleanup: chrome would not let go" in detail

    alarms = _scan(state_root, tmp_path / "home")
    assert [alarm.kind for alarm in alarms] == ["stopped"]
    assert "cleaning up afterwards failed" in alarms[0].body
    assert "cleanup: chrome would not let go" in alarms[0].body


@pytest.mark.parametrize(
    ("status", "stop_reason", "detail"),
    [
        ("interrupted", "session_expired", "cleanup: lock release failed"),
        (
            "governor_limit_reached",
            "governor_limit",
            "daily_profile_cap; finalization: immutable run snapshot: disk full",
        ),
    ],
)
def test_teardown_failure_on_a_quiet_stop_alarms(tmp_path, status, stop_reason, detail):
    state_root, state_dir = _state_dir(tmp_path)
    store = RuntimeStateStore(state_dir / wd.DB_FILENAME)
    run_id = store.start_run(
        source="linkedin", brief_id="spl", output_dir=str(state_dir), mode="fresh"
    )
    store.finish_run(run_id, status, stop_reason=stop_reason, stop_reason_detail=detail)

    alarms = _scan(state_root, tmp_path / "home")
    assert [alarm.kind for alarm in alarms] == ["stopped"]
    assert "cleaning up afterwards failed" in alarms[0].body


@pytest.mark.parametrize(
    ("status", "stop_reason", "detail"),
    [
        ("interrupted", "session_expired", None),
        ("governor_limit_reached", "governor_limit", "daily_profile_cap"),
        ("interrupted", "operator_stop", "cleanup: lock release failed"),
    ],
)
def test_quiet_stops_without_a_teardown_failure_stay_quiet(
    tmp_path, status, stop_reason, detail
):
    state_root, state_dir = _state_dir(tmp_path)
    store = RuntimeStateStore(state_dir / wd.DB_FILENAME)
    run_id = store.start_run(
        source="linkedin", brief_id="spl", output_dir=str(state_dir), mode="fresh"
    )
    store.finish_run(run_id, status, stop_reason=stop_reason, stop_reason_detail=detail)

    assert _scan(state_root, tmp_path / "home") == []
