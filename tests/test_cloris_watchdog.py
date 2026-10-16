"""tools/cloris_watchdog.py — the death alarm (CLO-243 slice 3).

Every test builds its own state dir and runtime DB under ``tmp_path``. The
schema comes from ``RuntimeStateStore`` itself (constructing it creates the
tables; it is never used to write rows here, because ``start_run`` would
also write a heartbeat); rows go in with plain SQL so each timestamp is
exactly what the case needs. Process liveness, lock holders and governor
state are injected through ``Probe``. Nothing touches the real
``output/state``, the real Keychain, osascript, or the network.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from shared.notify import notifier
from shared.runtime_state.store import RuntimeStateStore
from tools import cloris_watchdog as wd

RUN_PID = 51000
VISITOR_PID = 52000


# --- fixtures -----------------------------------------------------------------


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(notifier.DISABLE_ENV, raising=False)
    monkeypatch.setattr(notifier, "_fallback_state_dir", lambda: tmp_path / "notify-fallback")
    path = tmp_path / "watchdog-home"
    path.mkdir()
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "output" / "state"
    path.mkdir(parents=True)
    return path


def _state_dir(root: Path, name: str = "9000000003-spl") -> Path:
    state_dir = root / "linkedin" / name
    state_dir.mkdir(parents=True, exist_ok=True)
    RuntimeStateStore(state_dir / wd.DB_FILENAME)  # schema only
    for leftover in state_dir.glob("heartbeat*"):
        leftover.unlink()
    return state_dir


def _add_run(
    state_dir: Path,
    *,
    status: str,
    started_at: datetime,
    ended_at: datetime | None = None,
    stop_reason: str = "normal",
    detail: str | None = None,
    work_at: datetime | None = None,
) -> int:
    conn = sqlite3.connect(state_dir / wd.DB_FILENAME)
    try:
        cursor = conn.execute(
            """
            INSERT INTO runs(source, brief_id, output_dir, mode, status, stop_reason,
                             stop_reason_detail, started_at, ended_at)
            VALUES ('linkedin', 'brief-spl', ?, 'fresh', ?, ?, ?, ?, ?)
            """,
            (
                str(state_dir),
                status,
                stop_reason,
                detail,
                _iso(started_at),
                _iso(ended_at) if ended_at else None,
            ),
        )
        run_id = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO events(run_id, event_type, created_at) VALUES (?, 'run_started', ?)",
            (run_id, _iso(started_at)),
        )
        if work_at is not None:
            conn.execute(
                "INSERT INTO events(run_id, event_type, created_at) VALUES (?, 'page_done', ?)",
                (run_id, _iso(work_at)),
            )
        conn.commit()
    finally:
        conn.close()
    return run_id


def _heartbeat(state_dir: Path, entries: dict[int, tuple[datetime, datetime]]) -> None:
    pids = {str(pid): {"first_at": _iso(first), "last_at": _iso(last)} for pid, (first, last) in entries.items()}
    latest = max(entries.items(), key=lambda item: item[1][1])
    (state_dir / wd.HEARTBEAT_FILENAME).write_text(
        json.dumps({"heartbeat_at": _iso(latest[1][1]), "pid": latest[0], "pids": pids})
    )


def _probe(*, alive=(), holders=None, governor: wd.GovernorState | None = None) -> wd.Probe:
    alive_set = set(alive)
    return wd.Probe(
        pid_alive=lambda pid: pid in alive_set,
        lock_holders=lambda path: holders,
        governor=lambda: governor or wd.GovernorState(),
    )


def _scan(root: Path, home: Path, now: datetime, probe: wd.Probe, state: dict | None = None):
    state = state if state is not None else {"dirs": {}}
    return wd.scan(root, state, home=home, now=now, probe=probe), state


class _Outbox:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str, str]] = []

    def __call__(self, kind, title, body, dedupe_key, *, state_dir=None) -> bool:
        if any(key == dedupe_key for _, _, _, key in self.sent):
            return False
        self.sent.append((kind, title, body, dedupe_key))
        return True


def _seed_seen(state: dict, state_dir: Path) -> dict:
    """Mark a dir as already known so first-sight rules don't apply."""

    state.setdefault("dirs", {})[str(state_dir.resolve())] = {"seen": [], "stop_alarms": {}}
    return state


# --- running rows: died / stuck / quiet / healthy ------------------------------


def test_healthy_running_run_raises_nothing(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(hours=2)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=3))
    _heartbeat(state_dir, {RUN_PID: (started, now - timedelta(minutes=3))})

    alarms, _ = _scan(root, home, now, _probe(alive={RUN_PID}, holders=[RUN_PID]))
    assert alarms == []


def test_dead_process_with_quiet_run_alarms_died(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(hours=1)
    run_id = _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=8))
    _heartbeat(state_dir, {RUN_PID: (started, now - timedelta(minutes=8))})

    alarms, _ = _scan(root, home, now, _probe(alive=(), holders=[]))
    assert [a.kind for a in alarms] == ["died"]
    assert alarms[0].title == "Cloris died without saying why"
    assert alarms[0].dedupe_key == f"died:{state_dir.resolve()}:{run_id}"
    assert f"run {run_id}" in alarms[0].body


def test_died_needs_quiet_beyond_threshold(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(hours=1)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=2))
    _heartbeat(state_dir, {RUN_PID: (started, now - timedelta(minutes=2))})

    alarms, _ = _scan(root, home, now, _probe(alive=(), holders=[]))
    assert alarms == []


def test_died_on_one_kind_of_evidence_waits_longer(root, home, now) -> None:
    # No lsof answer (holders=None): the dead heartbeat pid alone is weaker
    # evidence, so a normal 12-minute gap between writes must not page Sam.
    state_dir = _state_dir(root)
    started = now - timedelta(hours=1)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=12))
    _heartbeat(state_dir, {RUN_PID: (started, now - timedelta(minutes=12))})
    assert _scan(root, home, now, _probe(alive=(), holders=None))[0] == []

    later = now + timedelta(minutes=5)
    alarms, _ = _scan(root, home, later, _probe(alive=(), holders=None))
    assert [a.kind for a in alarms] == ["died"]


def test_stale_heartbeat_with_live_pid_is_stuck_not_died(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(hours=2)
    run_id = _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=50))
    _heartbeat(state_dir, {RUN_PID: (started, now - timedelta(minutes=50))})

    alarms, _ = _scan(root, home, now, _probe(alive={RUN_PID}, holders=[RUN_PID]))
    assert [a.kind for a in alarms] == ["stuck"]
    assert alarms[0].title == "Cloris looks stuck"
    assert alarms[0].dedupe_key == f"stuck:{state_dir.resolve()}:{run_id}"


def test_stuck_threshold_tolerates_normal_gaps(root, home, now) -> None:
    # 28-29 Sep: healthy runs went up to ~12 minutes between writes.
    state_dir = _state_dir(root)
    started = now - timedelta(hours=2)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=30))
    _heartbeat(state_dir, {RUN_PID: (started, now - timedelta(minutes=30))})

    assert _scan(root, home, now, _probe(alive={RUN_PID}, holders=[RUN_PID]))[0] == []


def test_run_log_activity_counts_as_work(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(hours=2)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=50))
    (state_dir / wd.RUN_LOG_FILENAME).write_text("{}\n")  # mtime = now

    assert _scan(root, home, now, _probe(alive={RUN_PID}, holders=[RUN_PID]))[0] == []


def test_live_console_log_does_not_count_as_work(root, home, now) -> None:
    # The 5-minute status printer writes the console even when the pipeline hangs.
    state_dir = _state_dir(root)
    started = now - timedelta(hours=2)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=50))
    (state_dir / "live-console.log").write_text("[governor] status\n")

    alarms, _ = _scan(root, home, now, _probe(alive={RUN_PID}, holders=[RUN_PID]))
    assert [a.kind for a in alarms] == ["stuck"]


def test_governor_wait_suppresses_stuck(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(hours=2)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=50))
    waiting = wd.GovernorState(waiting=True, why="24h profile-open cap reached (400)")

    alarms, _ = _scan(root, home, now, _probe(alive={RUN_PID}, holders=[RUN_PID], governor=waiting))
    assert alarms == []


def test_governor_wait_does_not_hide_a_death(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(hours=1)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=20))
    _heartbeat(state_dir, {RUN_PID: (started, now - timedelta(minutes=20))})
    waiting = wd.GovernorState(waiting=True, why="forced backoff active")

    alarms, _ = _scan(root, home, now, _probe(alive=(), holders=[], governor=waiting))
    assert [a.kind for a in alarms] == ["died"]


def test_no_heartbeat_and_no_lock_answer_alarms_quiet_only_late(root, home, now) -> None:
    # A run launched by code that predates the heartbeat file: none here, no lock file.
    state_dir = _state_dir(root)
    started = now - timedelta(hours=3)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=60))
    assert _scan(root, home, now, _probe(holders=None))[0] == []

    alarms, _ = _scan(root, home, now + timedelta(minutes=31), _probe(holders=None))
    assert [a.kind for a in alarms] == ["quiet"]
    assert "can't tell whether its process is alive" in alarms[0].body


def test_lock_holder_alone_proves_life_for_old_code_runs(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(hours=3)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=20))

    assert _scan(root, home, now, _probe(holders=[RUN_PID]))[0] == []


def test_visitor_heartbeat_does_not_mask_a_dead_run(root, home, now) -> None:
    # Reviewer case: an admin tool opens the store after Cloris died. Its pid
    # is alive and its beat is fresh, but it first beat long after the row
    # started, so it is not the run's process and its beat is not work.
    state_dir = _state_dir(root)
    started = now - timedelta(hours=1)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=10))
    _heartbeat(
        state_dir,
        {
            RUN_PID: (started, now - timedelta(minutes=10)),
            VISITOR_PID: (now - timedelta(minutes=1), now - timedelta(minutes=1)),
        },
    )

    alarms, _ = _scan(root, home, now, _probe(alive={VISITOR_PID}, holders=[]))
    assert [a.kind for a in alarms] == ["died"]


def test_heartbeat_from_an_earlier_run_is_ignored(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(minutes=40)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=12))
    # Left over from a previous process that stopped before this row began.
    _heartbeat(state_dir, {RUN_PID: (now - timedelta(hours=5), now - timedelta(hours=2))})

    # No lock answer and no owner beat: alive is unknown, so only "quiet" applies.
    assert _scan(root, home, now, _probe(alive=(), holders=None))[0] == []


def test_unparseable_heartbeat_is_unknown_not_an_alarm(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(hours=1)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=20))
    (state_dir / wd.HEARTBEAT_FILENAME).write_text("")

    assert _scan(root, home, now, _probe(holders=None))[0] == []


def test_old_running_corpse_is_ignored(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(hours=40)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(hours=30))

    assert _scan(root, home, now, _probe(alive=(), holders=[]))[0] == []


# --- finished rows: stopped ------------------------------------------------------


def _known(state_dir: Path) -> dict:
    return _seed_seen({"dirs": {}}, state_dir)


def test_error_run_alarms_with_reason_and_detail(root, home, now) -> None:
    state_dir = _state_dir(root)
    run_id = _add_run(
        state_dir,
        status="error",
        started_at=now - timedelta(minutes=20),
        ended_at=now - timedelta(minutes=1),
        stop_reason="fatal_runtime_error",
        detail="navigate_to_search failed: sidebar elements not found after 16s",
    )

    alarms, _ = _scan(root, home, now, _probe(), _known(state_dir))
    assert [a.kind for a in alarms] == ["stopped"]
    assert alarms[0].title == "Cloris stopped (error)"
    assert "it hit an error it could not handle" in alarms[0].body
    assert "Detail: navigate_to_search failed: sidebar elements not found after 16s" in alarms[0].body
    assert alarms[0].dedupe_key == f"stopped:{state_dir.resolve()}:{run_id}"


def test_error_run_without_detail_says_so(root, home, now) -> None:
    state_dir = _state_dir(root)
    _add_run(
        state_dir,
        status="error",
        started_at=now - timedelta(minutes=20),
        ended_at=now - timedelta(minutes=1),
        stop_reason="fatal_runtime_error",
    )

    alarms, _ = _scan(root, home, now, _probe(), _known(state_dir))
    assert "No detail was recorded." in alarms[0].body


@pytest.mark.parametrize(
    "status,stop_reason",
    [
        ("completed", "normal"),
        ("interrupted", "session_expired"),
        ("interrupted", "operator_stop"),
        ("interrupted", "operator_pause"),
        ("governor_limit_reached", "governor_limit"),
    ],
)
def test_routine_endings_stay_quiet(root, home, now, status, stop_reason) -> None:
    state_dir = _state_dir(root)
    _add_run(
        state_dir,
        status=status,
        started_at=now - timedelta(hours=4),
        ended_at=now - timedelta(minutes=1),
        stop_reason=stop_reason,
    )

    assert _scan(root, home, now, _probe(), _known(state_dir))[0] == []


@pytest.mark.parametrize(
    "stop_reason,words",
    [
        ("browser_disconnect_unrecovered", "lost the Chrome connection"),
        ("api_budget_exhausted", "budget ran out"),
    ],
)
def test_alarming_interruptions(root, home, now, stop_reason, words) -> None:
    state_dir = _state_dir(root)
    _add_run(
        state_dir,
        status="interrupted",
        started_at=now - timedelta(hours=1),
        ended_at=now - timedelta(minutes=1),
        stop_reason=stop_reason,
    )

    alarms, _ = _scan(root, home, now, _probe(), _known(state_dir))
    assert [a.kind for a in alarms] == ["stopped"]
    assert words in alarms[0].body


def test_governor_stop_with_block_signal_alarms(root, home, now) -> None:
    state_dir = _state_dir(root)
    _add_run(
        state_dir,
        status="governor_limit_reached",
        started_at=now - timedelta(hours=1),
        ended_at=now - timedelta(minutes=1),
        stop_reason="governor_limit",
    )
    backoff = wd.GovernorState(
        waiting=True,
        why="forced backoff active",
        backoff_recorded_at=now - timedelta(minutes=2),
        backoff_until=now + timedelta(hours=6),
    )

    alarms, _ = _scan(root, home, now, _probe(governor=backoff), _known(state_dir))
    assert [a.kind for a in alarms] == ["stopped"]
    assert "rate-limit or block" in alarms[0].title


@pytest.mark.parametrize(
    "stop_reason,detail",
    [
        ("recruiter_reauth_required", None),
        # The store normalises unknown reasons to fatal_runtime_error and moves
        # the original into stop_reason_detail.
        ("fatal_runtime_error", "recruiter_reauth_required: Only one session is allowed at a time."),
    ],
)
def test_reauth_reason_gets_its_own_words(root, home, now, stop_reason, detail) -> None:
    state_dir = _state_dir(root)
    _add_run(
        state_dir,
        status="error",
        started_at=now - timedelta(minutes=5),
        ended_at=now - timedelta(minutes=1),
        stop_reason=stop_reason,
        detail=detail,
    )

    alarms, _ = _scan(root, home, now, _probe(), _known(state_dir))
    assert alarms[0].title == "Cloris needs you to sign in to LinkedIn Recruiter"
    assert "Sort it out in Chrome yourself" in alarms[0].body


def test_a_twelve_second_run_between_scans_still_alarms(root, home, now) -> None:
    state_dir = _state_dir(root)
    _add_run(
        state_dir,
        status="error",
        started_at=now - timedelta(seconds=70),
        ended_at=now - timedelta(seconds=58),
        stop_reason="fatal_runtime_error",
    )

    alarms, _ = _scan(root, home, now, _probe(), _known(state_dir))
    assert [a.kind for a in alarms] == ["stopped"]


def test_first_sight_ignores_old_endings_but_not_recent_ones(root, home, now) -> None:
    state_dir = _state_dir(root)
    _add_run(
        state_dir,
        status="error",
        started_at=now - timedelta(hours=3),
        ended_at=now - timedelta(hours=2),
        stop_reason="fatal_runtime_error",
    )
    recent = _add_run(
        state_dir,
        status="error",
        started_at=now - timedelta(minutes=15),
        ended_at=now - timedelta(minutes=10),
        stop_reason="lock_conflict",
    )

    alarms, _ = _scan(root, home, now, _probe())
    assert [a.dedupe_key for a in alarms] == [f"stopped:{state_dir.resolve()}:{recent}"]


# --- delivery, dedupe, bursts, exits (through run_once) ------------------------------


def test_already_alerted_run_is_not_sent_again(root, home, now, monkeypatch) -> None:
    sent: list[str] = []
    monkeypatch.setattr(notifier, "_send_macos", lambda t, b: sent.append(t) or True)
    monkeypatch.setattr(notifier, "_send_slack", lambda text: None)
    state_dir = _state_dir(root)
    started = now - timedelta(hours=1)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=8))
    _heartbeat(state_dir, {RUN_PID: (started, now - timedelta(minutes=8))})
    probe = _probe(alive=(), holders=[])

    wd.run_once(root, home, now=now, probe=probe)
    wd.run_once(root, home, now=now + timedelta(minutes=2), probe=probe)
    assert sent == ["Cloris died without saying why"]


def test_burst_of_failures_collapses_to_one_alarm_then_counts(root, home, now) -> None:
    # 28-29 Sep runs 10-15: a session failed every 2-5 minutes.
    state_dir = _state_dir(root)
    outbox = _Outbox()
    probe = _probe()
    _seed = _known(state_dir)
    wd.save_state(home, _seed)

    t = now - timedelta(minutes=30)
    for scan_no in range(3):
        for _ in range(2):
            t += timedelta(minutes=4)
            _add_run(state_dir, status="error", started_at=t - timedelta(seconds=30), ended_at=t,
                     stop_reason="fatal_runtime_error")
        wd.run_once(root, home, now=t + timedelta(seconds=30), probe=probe, send=outbox)
    assert len(outbox.sent) == 1
    assert "2 sessions in this state dir ended this way" in outbox.sent[0][2]

    # After the hour's cooldown, the next failure alarms and reports the held-back count.
    later = now + timedelta(minutes=50)
    _add_run(state_dir, status="error", started_at=later - timedelta(minutes=1), ended_at=later,
             stop_reason="fatal_runtime_error")
    wd.run_once(root, home, now=later + timedelta(seconds=30), probe=probe, send=outbox)
    assert len(outbox.sent) == 2
    assert "4 more like this were held back" in outbox.sent[1][2]


def test_failed_delivery_is_retried_next_scan(root, home, now) -> None:
    state_dir = _state_dir(root)
    wd.save_state(home, _known(state_dir))
    _add_run(state_dir, status="error", started_at=now - timedelta(minutes=3),
             ended_at=now - timedelta(minutes=1), stop_reason="fatal_runtime_error")
    attempts: list[str] = []

    def _down(kind, title, body, key, *, state_dir=None):
        attempts.append(key)
        return False

    wd.run_once(root, home, now=now, probe=_probe(), send=_down)
    outbox = _Outbox()
    wd.run_once(root, home, now=now + timedelta(minutes=2), probe=_probe(), send=outbox)
    assert len(attempts) == 1
    assert [kind for kind, *_ in outbox.sent] == ["stopped"]


def test_campaign_that_gives_up_after_absorbing_errors(root, home, now) -> None:
    state_dir = _state_dir(root)
    wd.save_state(home, _known(state_dir))
    outbox = _Outbox()
    started = now - timedelta(minutes=5)
    _add_run(state_dir, status="error", started_at=started, ended_at=now - timedelta(minutes=1),
             stop_reason="fatal_runtime_error")
    # The persistent campaign process is still alive after the row ended.
    _heartbeat(state_dir, {RUN_PID: (now - timedelta(hours=3), now - timedelta(minutes=1))})

    wd.run_once(root, home, now=now, probe=_probe(alive={RUN_PID}), send=outbox)
    assert [kind for kind, *_ in outbox.sent] == ["stopped"]
    assert "still alive and may retry on its own" in outbox.sent[0][2]

    # Still alive next scan: nothing new.
    wd.run_once(root, home, now=now + timedelta(minutes=2), probe=_probe(alive={RUN_PID}), send=outbox)
    assert len(outbox.sent) == 1

    # The process exits without a newer clean ending: "gave up", once.
    wd.run_once(root, home, now=now + timedelta(minutes=4), probe=_probe(alive=()), send=outbox)
    wd.run_once(root, home, now=now + timedelta(minutes=6), probe=_probe(alive=()), send=outbox)
    assert [kind for kind, *_ in outbox.sent] == ["stopped", "exited"]
    assert outbox.sent[1][1] == "Cloris gave up"


def test_clean_ending_after_an_absorbed_error_cancels_gave_up(root, home, now) -> None:
    state_dir = _state_dir(root)
    wd.save_state(home, _known(state_dir))
    outbox = _Outbox()
    _add_run(state_dir, status="error", started_at=now - timedelta(minutes=5),
             ended_at=now - timedelta(minutes=4), stop_reason="fatal_runtime_error")
    _heartbeat(state_dir, {RUN_PID: (now - timedelta(hours=3), now - timedelta(minutes=4))})
    wd.run_once(root, home, now=now, probe=_probe(alive={RUN_PID}), send=outbox)

    _add_run(state_dir, status="completed", started_at=now - timedelta(minutes=3),
             ended_at=now + timedelta(minutes=1))
    wd.run_once(root, home, now=now + timedelta(minutes=2), probe=_probe(alive=()), send=outbox)
    assert [kind for kind, *_ in outbox.sent] == ["stopped"]


def test_run_already_reported_dead_is_not_reported_again_when_closed(root, home, now, monkeypatch) -> None:
    monkeypatch.setattr(notifier, "_send_macos", lambda t, b: True)
    monkeypatch.setattr(notifier, "_send_slack", lambda text: None)
    state_dir = _state_dir(root)
    started = now - timedelta(hours=1)
    run_id = _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=8))
    _heartbeat(state_dir, {RUN_PID: (started, now - timedelta(minutes=8))})
    wd.run_once(root, home, now=now, probe=_probe(alive=(), holders=[]))
    assert notifier.already_sent(f"died:{state_dir.resolve()}:{run_id}", state_dir=home)

    # A reconciler later closes the row as worker_missing.
    conn = sqlite3.connect(state_dir / wd.DB_FILENAME)
    conn.execute(
        "UPDATE runs SET status='error', stop_reason='worker_missing', ended_at=? WHERE id=?",
        (_iso(now + timedelta(minutes=1)), run_id),
    )
    conn.commit()
    conn.close()
    sent: list[str] = []
    wd.run_once(root, home, now=now + timedelta(minutes=2), probe=_probe(alive=(), holders=[]),
                send=lambda *a, **k: sent.append(a[0]) or True)
    assert sent == []


def test_dry_run_prints_and_saves_nothing(root, home, now, capsys) -> None:
    state_dir = _state_dir(root)
    _add_run(state_dir, status="error", started_at=now - timedelta(minutes=3),
             ended_at=now - timedelta(minutes=1), stop_reason="fatal_runtime_error")

    def _never(*_a, **_k):  # pragma: no cover - must not run
        raise AssertionError("dry run must not send")

    wd.run_once(root, home, dry_run=True, now=now, probe=_probe(), send=_never)
    assert "would send stopped" in capsys.readouterr().out
    assert not (home / wd.STATE_FILENAME).exists()


def test_test_alert_sends_one_notification(monkeypatch, tmp_path) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(wd.notify, "notify_sam", lambda *a, **k: calls.append(a) or True)
    assert wd.main(["--test-alert"]) == 0
    assert len(calls) == 1 and calls[0][0] == "test"


def test_main_scans_the_given_root(root, home, monkeypatch) -> None:
    monkeypatch.setenv(notifier.STATE_DIR_ENV, str(home))
    monkeypatch.setattr(notifier, "_send_macos", lambda t, b: True)
    monkeypatch.setattr(notifier, "_send_slack", lambda text: None)
    assert wd.main(["--state-root", str(root)]) == 0
    assert (home / wd.STATE_FILENAME).exists()


# --- read-only discipline ---------------------------------------------------------


def _listing(state_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(state_dir.iterdir())}


def test_scan_creates_and_changes_no_file_in_the_state_dir(root, home, now) -> None:
    state_dir = _state_dir(root)
    started = now - timedelta(hours=1)
    _add_run(state_dir, status="running", started_at=started, work_at=now - timedelta(minutes=8))
    _add_run(state_dir, status="error", started_at=now - timedelta(minutes=5),
             ended_at=now - timedelta(minutes=1), stop_reason="fatal_runtime_error")
    _heartbeat(state_dir, {RUN_PID: (started, now - timedelta(minutes=8))})
    assert not (state_dir / (wd.DB_FILENAME + "-wal")).exists()
    before = _listing(state_dir)

    wd.run_once(root, home, now=now, probe=_probe(alive=(), holders=[]), send=_Outbox())
    assert _listing(state_dir) == before


def test_scan_reads_rows_still_in_the_wal(root, home, now) -> None:
    # A live writer holds the DB open in WAL mode with an uncheckpointed row.
    state_dir = _state_dir(root)
    writer = sqlite3.connect(state_dir / wd.DB_FILENAME)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO runs(source, brief_id, output_dir, mode, status, stop_reason, started_at, ended_at) "
            "VALUES ('linkedin', 'b', 'x', 'fresh', 'error', 'fatal_runtime_error', ?, ?)",
            (_iso(now - timedelta(minutes=3)), _iso(now - timedelta(minutes=1))),
        )
        writer.commit()
        assert (state_dir / (wd.DB_FILENAME + "-wal")).exists()
        alarms, _ = _scan(root, home, now, _probe(), _known(state_dir))
        assert [a.kind for a in alarms] == ["stopped"]
    finally:
        writer.close()


def test_wal_without_shm_is_skipped_not_created(root, home, now, tmp_path) -> None:
    state_dir = _state_dir(root)
    _add_run(state_dir, status="error", started_at=now - timedelta(minutes=3),
             ended_at=now - timedelta(minutes=1), stop_reason="fatal_runtime_error")
    (state_dir / (wd.DB_FILENAME + "-wal")).write_bytes(b"")
    before = _listing(state_dir)

    alarms, _ = _scan(root, home, now, _probe(), _known(state_dir))
    assert alarms == []
    assert _listing(state_dir) == before


def test_old_schema_without_detail_column(root, home, now) -> None:
    state_dir = root / "linkedin" / "legacy"
    state_dir.mkdir(parents=True)
    conn = sqlite3.connect(state_dir / wd.DB_FILENAME)
    conn.executescript(
        """
        CREATE TABLE runs (id INTEGER PRIMARY KEY, source TEXT, brief_id TEXT, output_dir TEXT,
            mode TEXT, status TEXT, stop_reason TEXT, started_at TEXT, ended_at TEXT);
        CREATE TABLE events (id INTEGER PRIMARY KEY, run_id INTEGER, event_type TEXT, created_at TEXT);
        CREATE TABLE candidate_attempts (id INTEGER PRIMARY KEY, run_id INTEGER, started_at TEXT, ended_at TEXT);
        CREATE TABLE work_units (id INTEGER PRIMARY KEY, run_id INTEGER, started_at TEXT, ended_at TEXT);
        """
    )
    conn.execute(
        "INSERT INTO runs(source, brief_id, output_dir, mode, status, stop_reason, started_at, ended_at) "
        "VALUES ('linkedin', 'b', 'x', 'fresh', 'error', 'fatal_runtime_error', ?, ?)",
        (_iso(now - timedelta(minutes=3)), _iso(now - timedelta(minutes=1))),
    )
    conn.commit()
    conn.close()

    alarms, _ = _scan(root, home, now, _probe(), _known(state_dir))
    assert "No detail was recorded." in alarms[0].body


def test_one_broken_db_does_not_blind_the_others(root, home, now) -> None:
    broken = root / "linkedin" / "broken"
    broken.mkdir(parents=True)
    (broken / wd.DB_FILENAME).write_bytes(b"not a database at all" * 100)
    good = _state_dir(root, "good")
    _add_run(good, status="error", started_at=now - timedelta(minutes=3),
             ended_at=now - timedelta(minutes=1), stop_reason="fatal_runtime_error")
    state = _known(good)
    _seed_seen(state, broken)

    alarms, _ = _scan(root, home, now, _probe(), state)
    assert [a.kind for a in alarms] == ["stopped"]


# --- probes ----------------------------------------------------------------------------


def test_governor_reader(tmp_path: Path, now) -> None:
    path = tmp_path / "daily_stats.json"
    assert wd.read_governor_state(path, now=now).waiting is False  # missing file

    path.write_text("{garbage")
    assert wd.read_governor_state(path, now=now).waiting is False

    path.write_text(json.dumps({"profile_opens": [now.timestamp() - 60] * 400}))
    assert wd.read_governor_state(path, now=now).waiting is True

    path.write_text(json.dumps({"profile_opens": [now.timestamp() - 90000] * 400}))
    assert wd.read_governor_state(path, now=now).waiting is False  # outside the 24h window

    path.write_text(
        json.dumps(
            {
                "profile_opens": [],
                "forced_backoff": {
                    "reason": "blocked_or_rate_limited",
                    "recorded_at": now.timestamp() - 60,
                    "until": now.timestamp() + 3600,
                },
            }
        )
    )
    state = wd.read_governor_state(path, now=now)
    assert state.waiting is True and state.backoff_recorded_at is not None


@pytest.mark.skipif(not Path(wd.LSOF).exists(), reason="lsof not available")
def test_lock_holders_reads_lsof(tmp_path: Path) -> None:
    lock = tmp_path / wd.LOCK_FILENAME
    assert wd.lock_holders(lock) is None  # no lock file: unknown
    lock.write_text("")
    assert wd.lock_holders(lock) == []
    with open(lock, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.getpid() in (wd.lock_holders(lock) or [])


def test_find_state_dirs(root) -> None:
    a = _state_dir(root, "a")
    nested = root / "github" / "x" / "deeper"
    nested.mkdir(parents=True)
    shutil.copy(a / wd.DB_FILENAME, nested / wd.DB_FILENAME)
    assert wd.find_state_dirs(root) == sorted([a, nested])
