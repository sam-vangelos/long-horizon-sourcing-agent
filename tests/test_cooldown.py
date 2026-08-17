import json

import pytest

import shared.cooldown as cooldown


def _configure_governor_paths(monkeypatch, tmp_path):
    governor_dir = tmp_path / "governor"
    monkeypatch.setattr(cooldown, "GOVERNOR_DIR", governor_dir)
    monkeypatch.setattr(cooldown, "DAILY_STATS_FILE", governor_dir / "daily_stats.json")
    monkeypatch.setattr(cooldown, "SESSIONS_LOG", governor_dir / "sessions.jsonl")
    return governor_dir


def test_get_sessions_today_excludes_interrupted_sessions_from_daily_cap(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)

    today = "2026-04-05"
    cooldown._save_raw({
        "profile_opens": [],
        "sessions_today": [
            {
                "date": today,
                "session_num": 1,
                "session_type": "linkedin_sourcing",
                "start_ts": 1,
                "end_ts": 2,
                "profile_opens": 64,
                "reason": "session_duration (3.7h)",
            },
            {
                "date": today,
                "session_num": 2,
                "session_type": "linkedin_sourcing",
                "start_ts": 3,
                "end_ts": 4,
                "profile_opens": 2,
                "reason": "interrupted: KeyboardInterrupt",
            },
        ],
    })

    monkeypatch.setattr(cooldown.time, "strftime", lambda fmt, *args: today if fmt == "%Y-%m-%d" else "12:00 PM")

    assert cooldown.get_sessions_today(session_type="linkedin_sourcing") == 1


def test_get_sessions_today_closes_stale_pidless_sessions(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)

    today = "2026-04-05"
    cooldown._save_raw({
        "profile_opens": [],
        "sessions_today": [
            {
                "date": today,
                "session_num": 3,
                "session_type": "linkedin_sourcing",
                "start_ts": 10,
            },
        ],
    })

    monkeypatch.setattr(cooldown.time, "strftime", lambda fmt, *args: today if fmt == "%Y-%m-%d" else "12:00 PM")
    monkeypatch.setattr(cooldown.time, "time", lambda: 20)

    assert cooldown.get_sessions_today(session_type="linkedin_sourcing") == 0

    data = json.loads(cooldown.DAILY_STATS_FILE.read_text())
    entry = data["sessions_today"][0]
    assert entry["end_ts"] == 20
    assert entry["reason"] == "interrupted: stale_session"
    assert entry["counts_toward_cap"] is False


def test_record_session_end_marks_keyboard_interrupt_as_not_counting(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)

    today = "2026-04-05"
    monkeypatch.setattr(cooldown.time, "strftime", lambda fmt, *args: today if fmt == "%Y-%m-%d" else "12:00 PM")
    monkeypatch.setattr(cooldown.os, "getpid", lambda: 12345)

    session_num = cooldown.record_session_start()
    cooldown.record_session_end(
        session_num=session_num,
        profile_opens=5,
        reason="interrupted: KeyboardInterrupt",
        stats={"saved": 0},
    )

    data = json.loads(cooldown.DAILY_STATS_FILE.read_text())
    entry = data["sessions_today"][0]
    assert entry["counts_toward_cap"] is False

    log_entry = json.loads(cooldown.SESSIONS_LOG.read_text().splitlines()[-1])
    assert log_entry["counts_toward_cap"] is False


# ---------------------------------------------------------------------------
# P8.4(a): typed shutdown_kind must be the source of truth over reason-string
# parsing when the caller supplies it.
# ---------------------------------------------------------------------------


def test_record_session_end_typed_error_kind_does_not_count_despite_generic_reason(monkeypatch, tmp_path):
    """A caller that passes shutdown_kind=ERROR must not count toward the cap
    even when `reason` is a generic message that string-parsing alone would
    have treated as a normal completion (no "interrupted:"/"error:" prefix).
    This is the case that proves the typed path is actually consulted, not
    just coincidentally agreeing with the string heuristic."""
    _configure_governor_paths(monkeypatch, tmp_path)
    today = "2026-04-05"
    monkeypatch.setattr(cooldown.time, "strftime", lambda fmt, *args: today if fmt == "%Y-%m-%d" else "12:00 PM")
    monkeypatch.setattr(cooldown.os, "getpid", lambda: 12345)

    session_num = cooldown.record_session_start()
    cooldown.record_session_end(
        session_num=session_num,
        profile_opens=5,
        reason="something went sideways",  # no interrupted:/error: prefix
        stats={"saved": 0},
        shutdown_kind=cooldown.ShutdownKind.ERROR,
    )

    data = json.loads(cooldown.DAILY_STATS_FILE.read_text())
    assert data["sessions_today"][0]["counts_toward_cap"] is False


def test_record_session_end_typed_completed_kind_counts_toward_cap(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)
    today = "2026-04-05"
    monkeypatch.setattr(cooldown.time, "strftime", lambda fmt, *args: today if fmt == "%Y-%m-%d" else "12:00 PM")
    monkeypatch.setattr(cooldown.os, "getpid", lambda: 12345)

    session_num = cooldown.record_session_start()
    cooldown.record_session_end(
        session_num=session_num,
        profile_opens=5,
        reason="pipeline_complete",
        stats={"saved": 0},
        shutdown_kind=cooldown.ShutdownKind.COMPLETED,
    )

    data = json.loads(cooldown.DAILY_STATS_FILE.read_text())
    assert data["sessions_today"][0]["counts_toward_cap"] is True


def test_record_session_end_without_shutdown_kind_falls_back_to_reason_string(monkeypatch, tmp_path):
    """Legacy fallback: a caller that only has a reason string (no typed
    shutdown_kind) still gets the pre-P8.4 string-parsing behavior."""
    _configure_governor_paths(monkeypatch, tmp_path)
    today = "2026-04-05"
    monkeypatch.setattr(cooldown.time, "strftime", lambda fmt, *args: today if fmt == "%Y-%m-%d" else "12:00 PM")
    monkeypatch.setattr(cooldown.os, "getpid", lambda: 12345)

    session_num = cooldown.record_session_start()
    cooldown.record_session_end(
        session_num=session_num,
        profile_opens=5,
        reason="error: RuntimeError",
        stats={"saved": 0},
    )

    data = json.loads(cooldown.DAILY_STATS_FILE.read_text())
    assert data["sessions_today"][0]["counts_toward_cap"] is False


# ---------------------------------------------------------------------------
# P8.4(c): a stats file missing "profile_opens" must not crash the governor.
# ---------------------------------------------------------------------------


def test_get_profile_opens_24h_survives_missing_profile_opens_key(monkeypatch, tmp_path):
    governor_dir = _configure_governor_paths(monkeypatch, tmp_path)
    governor_dir.mkdir(parents=True, exist_ok=True)
    cooldown.DAILY_STATS_FILE.write_text(json.dumps({"sessions_today": []}))

    assert cooldown.get_profile_opens_24h() == 0


def test_record_profile_open_survives_missing_profile_opens_key(monkeypatch, tmp_path):
    governor_dir = _configure_governor_paths(monkeypatch, tmp_path)
    governor_dir.mkdir(parents=True, exist_ok=True)
    cooldown.DAILY_STATS_FILE.write_text(json.dumps({"sessions_today": []}))

    cooldown.record_profile_open()

    assert cooldown.get_profile_opens_24h() == 1


# ---------------------------------------------------------------------------
# P8.2 follow-up: forced backoff persists across sessions/processes.
# A rate-limit trip must outlive the in-memory governor — the next
# can_start_session() (new process, new governor instance) must refuse
# until the cooldown expires.
# ---------------------------------------------------------------------------


def test_forced_backoff_persists_and_blocks_until_expiry(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)

    cooldown.record_forced_backoff("blocked_or_rate_limited", cooldown_seconds=3600)

    active = cooldown.get_active_backoff()
    assert active is not None
    assert active["reason"] == "blocked_or_rate_limited"
    assert active["until"] > __import__("time").time()


def test_forced_backoff_expires(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)

    cooldown.record_forced_backoff("blocked_or_rate_limited", cooldown_seconds=-1)

    assert cooldown.get_active_backoff() is None


def test_forced_backoff_absent_by_default(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)

    assert cooldown.get_active_backoff() is None


def test_governor_can_start_session_honors_persisted_backoff(monkeypatch, tmp_path):
    """P8.2 follow-up end-to-end: force_backoff on one governor instance
    blocks can_start_session on a FRESH instance (process-restart shape)."""
    _configure_governor_paths(monkeypatch, tmp_path)
    from shared.governor import SessionGovernor

    first = SessionGovernor()
    first.force_backoff("blocked_or_rate_limited")

    fresh = SessionGovernor()
    ok, reason = fresh.can_start_session()
    assert ok is False
    assert "blocked_or_rate_limited" in reason


# ---------------------------------------------------------------------------
# Wave 1 Slice 1.2: fail closed on unreadable governor state.
# ---------------------------------------------------------------------------


def test_missing_stats_file_still_returns_empty_structure(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)

    assert cooldown._load_raw() == {"profile_opens": [], "sessions_today": []}


def test_corrupt_stats_file_raises_rather_than_reporting_zero(monkeypatch, tmp_path):
    governor_dir = _configure_governor_paths(monkeypatch, tmp_path)
    governor_dir.mkdir(parents=True, exist_ok=True)
    cooldown.DAILY_STATS_FILE.write_text("{not json")

    with pytest.raises(cooldown.GovernorStateUnreadable):
        cooldown._load_raw()


def test_can_start_session_refuses_on_unreadable_state(monkeypatch, tmp_path):
    governor_dir = _configure_governor_paths(monkeypatch, tmp_path)
    governor_dir.mkdir(parents=True, exist_ok=True)
    cooldown.DAILY_STATS_FILE.write_text("{not json")

    from shared.governor import SessionGovernor

    ok, reason = SessionGovernor().can_start_session()
    assert ok is False
    assert "unreadable" in reason.lower()


def test_corrupt_state_does_not_silently_clear_an_active_backoff(monkeypatch, tmp_path):
    governor_dir = _configure_governor_paths(monkeypatch, tmp_path)
    governor_dir.mkdir(parents=True, exist_ok=True)
    cooldown.DAILY_STATS_FILE.write_text("{not json")

    with pytest.raises(cooldown.GovernorStateUnreadable):
        cooldown.get_active_backoff()


def test_print_status_survives_unreadable_state(monkeypatch, tmp_path, capsys):
    governor_dir = _configure_governor_paths(monkeypatch, tmp_path)
    governor_dir.mkdir(parents=True, exist_ok=True)
    cooldown.DAILY_STATS_FILE.write_text("{not json")

    cooldown.print_status()

    captured = capsys.readouterr()
    assert "unreadable" in captured.out.lower()


def test_wait_for_governor_window_returns_false_on_unreadable_state(monkeypatch):
    import asyncio

    from linkedin import session_orchestrator as so

    class _Governor:
        def can_start_session(self, session_type="linkedin_sourcing"):
            return False, "daily session limit reached"

    monkeypatch.setattr(so, "_GOVERNOR_WAIT_POLL_SECONDS", 0.01)

    def raise_unreadable():
        raise cooldown.GovernorStateUnreadable("unreadable governor state file")

    monkeypatch.setattr(so.cooldown, "get_active_backoff", raise_unreadable)

    stop = asyncio.Event()
    allowed = asyncio.run(so._wait_for_governor_window(_Governor(), stop))
    assert allowed is False


# ---------------------------------------------------------------------------
# A5: count a daily slot from the first profile open (LINKEDIN_SLOT_ON_FIRST_OPEN)
# ---------------------------------------------------------------------------


def _end_session(monkeypatch, tmp_path, *, opens, kind, reason="whatever happened"):
    """Record one session end and return its stored cap verdict."""
    _configure_governor_paths(monkeypatch, tmp_path)
    today = "2026-08-03"
    monkeypatch.setattr(
        cooldown.time, "strftime", lambda fmt, *a: today if fmt == "%Y-%m-%d" else "12:00 PM"
    )
    monkeypatch.setattr(cooldown.os, "getpid", lambda: 12345)

    session_num = cooldown.record_session_start()
    cooldown.record_session_end(
        session_num=session_num,
        profile_opens=opens,
        reason=reason,
        stats={"saved": 0},
        shutdown_kind=kind,
    )
    data = json.loads(cooldown.DAILY_STATS_FILE.read_text())
    return data["sessions_today"][0]["counts_toward_cap"]


def test_errored_session_that_opened_profiles_burns_a_slot(monkeypatch, tmp_path):
    """The case the flag exists for.

    An errored session touched the account exactly as much as a completed one;
    only the reason it stopped differs, and LinkedIn cannot see that. Under the
    old rule 133 of 162 sessions consumed nothing.
    """
    monkeypatch.setattr(cooldown.config, "LINKEDIN_SLOT_ON_FIRST_OPEN", True)
    assert _end_session(
        monkeypatch, tmp_path, opens=8, kind=cooldown.ShutdownKind.ERROR
    ) is True


def test_operator_stop_never_burns_a_slot(monkeypatch, tmp_path):
    """Development churn, not sourcing — the one exception to the rule above."""
    monkeypatch.setattr(cooldown.config, "LINKEDIN_SLOT_ON_FIRST_OPEN", True)
    assert _end_session(
        monkeypatch, tmp_path, opens=24, kind=cooldown.ShutdownKind.INTERRUPTED
    ) is False


def test_launch_that_opened_nothing_is_free_under_either_policy(monkeypatch, tmp_path):
    for enabled in (True, False):
        monkeypatch.setattr(cooldown.config, "LINKEDIN_SLOT_ON_FIRST_OPEN", enabled)
        assert _end_session(
            monkeypatch, tmp_path / f"n{enabled}", opens=0, kind=cooldown.ShutdownKind.ERROR
        ) is False


def test_completed_session_burns_a_slot_under_either_policy(monkeypatch, tmp_path):
    for enabled in (True, False):
        monkeypatch.setattr(cooldown.config, "LINKEDIN_SLOT_ON_FIRST_OPEN", enabled)
        assert _end_session(
            monkeypatch, tmp_path / f"c{enabled}", opens=3, kind=cooldown.ShutdownKind.COMPLETED
        ) is True


def test_flag_off_preserves_the_old_verdict_exactly(monkeypatch, tmp_path):
    """Flag-off equivalence: only COMPLETED counts, as before."""
    monkeypatch.setattr(cooldown.config, "LINKEDIN_SLOT_ON_FIRST_OPEN", False)
    assert _end_session(
        monkeypatch, tmp_path / "a", opens=8, kind=cooldown.ShutdownKind.ERROR
    ) is False
    assert _end_session(
        monkeypatch, tmp_path / "b", opens=8, kind=cooldown.ShutdownKind.INTERRUPTED
    ) is False


def test_policy_change_does_not_rewrite_stored_verdicts(monkeypatch, tmp_path):
    """Turning the flag on must not retroactively consume today's budget.

    Entries carry an explicit counts_toward_cap, and _entry_counts_toward_cap
    returns the stored value when present, so history keeps the verdict it was
    written with.
    """
    _configure_governor_paths(monkeypatch, tmp_path)
    entry = {
        "date": "2026-08-03",
        "session_num": 1,
        "profile_opens": 24,
        "reason": "error: Request timed out.",
        "end_ts": 10,
        "counts_toward_cap": False,
    }
    monkeypatch.setattr(cooldown.config, "LINKEDIN_SLOT_ON_FIRST_OPEN", True)
    assert cooldown._entry_counts_toward_cap(entry) is False


def test_can_start_session_ignores_session_count(monkeypatch, tmp_path):
    """CLO-153 (Sam's 2026-08-11 ruling): session COUNTS never gate a launch.

    Ten counted sessions today — far past the retired 3/day cap — including a
    crashed slot-burning one, must leave can_start_session True while opens
    are under budget."""
    from shared.governor import SessionGovernor

    _configure_governor_paths(monkeypatch, tmp_path)
    today_iso = "2026-08-11"
    monkeypatch.setattr(
        cooldown.time,
        "strftime",
        lambda fmt, *args: today_iso if fmt == "%Y-%m-%d" else "12:00 PM",
    )
    entries = [
        {
            "date": today_iso,
            "session_num": n,
            "session_type": "linkedin_sourcing",
            "start_ts": n,
            "end_ts": n + 1,
            "profile_opens": 30,
            "reason": (
                "error: The read operation timed out"
                if n == 1
                else "session_duration (4.2h)"
            ),
            "counts_toward_cap": True,
        }
        for n in range(1, 11)
    ]
    cooldown._save_raw({"profile_opens": [], "sessions_today": entries})

    ok, reason = SessionGovernor().can_start_session(
        session_type="linkedin_sourcing"
    )
    assert ok is True
    assert reason == "ok"
    assert cooldown.get_sessions_today(session_type="linkedin_sourcing") == 10


def test_opens_cap_still_refuses_after_session_cap_removal(monkeypatch, tmp_path):
    """The 24h profile-open budget remains the volume gate (CLO-153)."""
    import time as _time

    from shared.governor import SessionGovernor

    _configure_governor_paths(monkeypatch, tmp_path)
    now = _time.time()
    cooldown._save_raw(
        {"profile_opens": [now - 60.0] * 400, "sessions_today": []}
    )

    ok, reason = SessionGovernor().can_start_session(
        session_type="linkedin_sourcing"
    )
    assert ok is False
    assert "24h profile open cap" in reason


def test_recorded_today_counts_every_entry_counted_or_not(monkeypatch, tmp_path):
    """CLO-153 review finding: the banners say "recorded today" and must
    count exactly that — every entry, whether or not it occupied a slot."""
    import os

    _configure_governor_paths(monkeypatch, tmp_path)
    today_iso = "2026-08-11"
    monkeypatch.setattr(
        cooldown.time,
        "strftime",
        lambda fmt, *args: today_iso if fmt == "%Y-%m-%d" else "12:00 PM",
    )
    cooldown._save_raw({
        "profile_opens": [],
        "sessions_today": [
            {
                "date": today_iso,
                "session_num": 1,
                "session_type": "linkedin_sourcing",
                "start_ts": 1,
                "end_ts": 2,
                "profile_opens": 40,
                "reason": "session_duration (4.2h)",
                "counts_toward_cap": True,
            },
            {
                "date": today_iso,
                "session_num": 2,
                "session_type": "linkedin_sourcing",
                "start_ts": 3,
                "end_ts": 4,
                "profile_opens": 0,
                "reason": "error: navigate_to_search failed",
                "counts_toward_cap": False,
            },
            {
                "date": today_iso,
                "session_num": 3,
                "session_type": "linkedin_sourcing",
                "start_ts": 5,
                "pid": os.getpid(),
            },
        ],
    })

    recorded = cooldown.get_sessions_recorded_today(
        session_type="linkedin_sourcing"
    )
    counted = cooldown.get_sessions_today(session_type="linkedin_sourcing")
    assert recorded == 3
    assert counted == 2
