"""Persistent rolling-window tracker for profile opens.

Stores timestamped entries in ~/.sourcing-governor/daily_stats.json.
On every read, prunes entries older than the window (default 24h).
Thread-safe via file-level atomic writes.
"""

import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from shared import config

GOVERNOR_DIR = Path.home() / ".sourcing-governor"
DAILY_STATS_FILE = GOVERNOR_DIR / "daily_stats.json"
SESSIONS_LOG = GOVERNOR_DIR / "sessions.jsonl"

WINDOW_SECONDS = 24 * 3600  # 24 hours


class GovernorStateUnreadable(RuntimeError):
    """Raised when the governor stats file exists but cannot be parsed."""


class ShutdownKind(str, Enum):
    """P8.4: typed session-end classification for cap-counting.

    Callers that know WHY a session ended (session_orchestrator.py) should
    pass this explicitly to record_session_end() instead of relying on
    string-parsing a free-text `reason` another module formatted. String
    parsing (_reason_counts_toward_cap) remains as the fallback ONLY for
    checkpoint entries written before this existed and for callers that
    still only have a reason string.
    """

    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    ERROR = "error"


def _ensure_dir():
    GOVERNOR_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw() -> dict:
    """Load the raw stats file. Returns empty structure if missing."""
    _ensure_dir()
    if not DAILY_STATS_FILE.exists():
        return {"profile_opens": [], "sessions_today": []}
    try:
        return json.loads(DAILY_STATS_FILE.read_text())
    except (json.JSONDecodeError, KeyError) as exc:
        raise GovernorStateUnreadable(
            f"unreadable governor state file: {DAILY_STATS_FILE}"
        ) from exc


def _save_raw(data: dict):
    """Atomic write to stats file."""
    _ensure_dir()
    tmp = DAILY_STATS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(DAILY_STATS_FILE)


def _prune(data: dict, now: Optional[float] = None) -> dict:
    """Remove entries older than the rolling window."""
    now = now or time.time()
    cutoff = now - WINDOW_SECONDS
    # P8.4: guard the missing-key case — a corrupt or hand-edited stats file
    # without "profile_opens" must not crash every governor check (this ran
    # on literally every open_profile*() call and every can_start_session()).
    data["profile_opens"] = [
        ts for ts in data.get("profile_opens", []) if ts > cutoff
    ]
    # Sessions: prune to calendar day
    today = time.strftime("%Y-%m-%d")
    data["sessions_today"] = [
        s for s in data.get("sessions_today", [])
        if s.get("date") == today
    ]
    return data


def _pid_is_alive(pid: int | None) -> bool:
    """Best-effort liveness check for a local process id."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reason_counts_toward_cap(reason: str) -> bool:
    """Return True when a finished session should consume a daily slot.

    P8.4 legacy fallback: this infers cap-counting by string-parsing a
    free-text `reason` another module (session_orchestrator.py) formatted
    with an f-string. Used only when the caller doesn't pass a typed
    `shutdown_kind` (record_session_end) and for entries in stats files
    written before ShutdownKind existed (_entry_counts_toward_cap).
    """
    normalized = (reason or "").strip().lower()
    if not normalized or normalized == "unknown":
        return False
    if normalized.startswith("interrupted:"):
        return False
    if normalized.startswith("error:"):
        return False
    return True


def _shutdown_kind_counts_toward_cap(shutdown_kind: Union["ShutdownKind", str]) -> bool:
    """Typed replacement for _reason_counts_toward_cap's string parsing.

    Only ShutdownKind.COMPLETED consumes a daily slot; INTERRUPTED and ERROR
    do not. Accepts a raw string too (e.g. "completed") so callers aren't
    forced to import the enum for a one-off comparison.
    """
    value = shutdown_kind.value if isinstance(shutdown_kind, ShutdownKind) else str(shutdown_kind or "")
    return value.strip().lower() == ShutdownKind.COMPLETED.value


def _shutdown_kind_counts_from_first_open(
    shutdown_kind: Union["ShutdownKind", str],
) -> bool:
    """A5: everything except an operator stop consumes a slot.

    What the cap is trying to bound is how often the account gets touched, and
    an errored session touched it exactly as much as a completed one — the
    only difference is on our side. An operator stop is the exception because
    it is development churn rather than sourcing, and it is distinguishable:
    session_orchestrator maps OperatorStopRequested, KeyboardInterrupt and
    CancelledError to INTERRUPTED and everything else to ERROR. A kill that
    never reaches that handler leaves no typed signal and is therefore counted,
    which is the safe direction — the account was touched and nothing proves
    the operator meant it.

    The caller still requires ``profile_opens > 0``, so a launch that attaches
    and fails without opening anything stays free under either policy.
    """
    value = shutdown_kind.value if isinstance(shutdown_kind, ShutdownKind) else str(shutdown_kind or "")
    return value.strip().lower() != ShutdownKind.INTERRUPTED.value


def _entry_counts_toward_cap(entry: dict) -> bool:
    """Backfill cap-counting behavior for old and new session entries."""
    if "counts_toward_cap" in entry:
        return bool(entry.get("counts_toward_cap"))
    return entry.get("profile_opens", 0) > 0 and _reason_counts_toward_cap(entry.get("reason", ""))


def _reconcile_stale_sessions(data: dict, now: Optional[float] = None) -> bool:
    """Close orphaned in-progress sessions so they stop consuming slots forever."""
    now = now or time.time()
    changed = False
    for entry in data.get("sessions_today", []):
        if "end_ts" in entry:
            inferred = _entry_counts_toward_cap(entry)
            if entry.get("counts_toward_cap") != inferred:
                entry["counts_toward_cap"] = inferred
                changed = True
            continue

        pid = entry.get("pid")
        if pid is not None and _pid_is_alive(pid):
            continue

        entry["end_ts"] = now
        entry.setdefault("profile_opens", 0)
        entry["reason"] = entry.get("reason") or "interrupted: stale_session"
        entry["counts_toward_cap"] = False
        changed = True

    return changed


def _load_current_data(now: Optional[float] = None) -> dict:
    """Load, prune, reconcile, and persist the live governor snapshot."""
    data = _prune(_load_raw(), now=now)
    changed = _reconcile_stale_sessions(data, now=now)
    if changed:
        _save_raw(data)
    return data


# P8.2 follow-up: a rate-limit/block signal must outlive the in-memory
# governor. force_backoff() trips the current session immediately (in-memory);
# this record makes the NEXT session honor it too — can_start_session()
# refuses while the cooldown is live, so a blocked account is not re-probed
# after the ordinary 75-180 min dormant gap. Default duration is a safety
# parameter, not a product decision: long enough to clearly exceed the
# dormant cadence, short enough that a false positive costs under a day.
# Flagged for Sam's tuning in the P8 commit body.
RATE_LIMIT_BACKOFF_SECONDS = 6 * 3600


def record_forced_backoff(reason: str, cooldown_seconds: float = RATE_LIMIT_BACKOFF_SECONDS) -> None:
    """Persist a forced-backoff window (rate-limit/block signal)."""
    data = _prune(_load_raw())
    data["forced_backoff"] = {
        "reason": reason,
        "recorded_at": time.time(),
        "until": time.time() + cooldown_seconds,
    }
    _save_raw(data)


def get_active_backoff() -> Optional[dict]:
    """Return the live forced-backoff entry, or None if absent/expired."""
    data = _load_raw()
    entry = data.get("forced_backoff")
    if not isinstance(entry, dict):
        return None
    if time.time() >= float(entry.get("until", 0)):
        # Expired — clear it so the file doesn't carry stale state forever.
        data.pop("forced_backoff", None)
        _save_raw(data)
        return None
    return entry


def get_profile_opens_24h() -> int:
    """Count profile opens in the rolling 24h window."""
    data = _prune(_load_raw())
    _save_raw(data)
    return len(data["profile_opens"])


def record_profile_open():
    """Record a single profile open with current timestamp."""
    data = _prune(_load_raw())
    data["profile_opens"].append(time.time())
    _save_raw(data)


def get_sessions_recorded_today(session_type: Optional[str] = None) -> int:
    """Count every session entry recorded today, with no cap semantics.

    ``get_sessions_today`` answers the historical slot question (entries that
    occupy or occupied a counted slot); this answers the display question —
    "which session of the day is this?" — and the CLO-153 banners that say
    "recorded" count exactly this.
    """
    data = _load_current_data()
    return sum(
        1
        for entry in data["sessions_today"]
        if not session_type or entry.get("session_type") == session_type
    )


def get_sessions_today(session_type: Optional[str] = None) -> int:
    """Count sessions that currently occupy a daily sourcing slot."""
    data = _load_current_data()
    count = 0
    for entry in data["sessions_today"]:
        if session_type and entry.get("session_type") != session_type:
            continue
        if "end_ts" not in entry:
            count += 1
            continue
        if _entry_counts_toward_cap(entry):
            count += 1
    return count


def record_session_start(session_type: str = "linkedin_sourcing") -> int:
    """Record a new session start. Returns session number for today."""
    data = _load_current_data()
    today = time.strftime("%Y-%m-%d")
    session_num = max((entry.get("session_num", 0) for entry in data["sessions_today"]), default=0) + 1
    data["sessions_today"].append({
        "date": today,
        "session_num": session_num,
        "session_type": session_type,
        "start_ts": time.time(),
        "pid": os.getpid(),
    })
    _save_raw(data)
    return session_num


def record_session_end(
    session_num: int,
    profile_opens: int,
    reason: str,
    stats: dict,
    counts_toward_cap: Optional[bool] = None,
    shutdown_kind: Optional[Union["ShutdownKind", str]] = None,
):
    """Update the session entry in daily_stats and append to sessions log.

    ``reason`` stays a free-text field for logging/debugging. ``shutdown_kind``
    (P8.4) is the typed signal cap-counting should use when the caller knows
    it explicitly (session_orchestrator.py does); string-parsing ``reason``
    remains the fallback for callers that only have a reason string and for
    stats-file entries written before ShutdownKind existed.
    """
    # Update the daily_stats entry with completion info
    data = _load_current_data()
    today = time.strftime("%Y-%m-%d")
    session_entry = None
    if counts_toward_cap is None:
        if shutdown_kind is not None:
            counts = (
                _shutdown_kind_counts_from_first_open
                if config.LINKEDIN_SLOT_ON_FIRST_OPEN
                else _shutdown_kind_counts_toward_cap
            )
            counts_toward_cap = profile_opens > 0 and counts(shutdown_kind)
        else:
            counts_toward_cap = profile_opens > 0 and _reason_counts_toward_cap(reason)
    for entry in data["sessions_today"]:
        if entry["session_num"] == session_num and entry.get("date") == today:
            entry["end_ts"] = time.time()
            entry["profile_opens"] = profile_opens
            entry["reason"] = reason
            entry["counts_toward_cap"] = counts_toward_cap
            session_entry = entry
            break
    _save_raw(data)

    # Also append to sessions log for historical record
    _ensure_dir()
    log_entry = {
        "session_type": session_entry.get("session_type", "unknown") if session_entry else "unknown",
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(session_entry["start_ts"])) if session_entry and "start_ts" in session_entry else None,
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shutdown_reason": reason,
        "counts_toward_cap": counts_toward_cap,
        "profile_opens_count": profile_opens,
        "saves_count": stats.get("saved", 0) if stats else 0,
    }
    with open(SESSIONS_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def print_status():
    """Print current 24h stats for --status flag."""
    try:
        opens_24h = get_profile_opens_24h()
        sessions = get_sessions_recorded_today()
    except GovernorStateUnreadable:
        print("governor state unreadable — cannot report stats")
        return

    print(f"Profile opens (rolling 24h): {opens_24h}/400")
    print(f"Sessions recorded today: {sessions} (no count gate — CLO-153)")
    print("Time-of-day window: DISABLED (24h operation)")
    print(f"Current time: {time.strftime('%I:%M %p')}")

    if opens_24h >= 400:
        print("\n⚠ 24h profile open cap reached. No sourcing sessions available.")
    else:
        remaining = 400 - opens_24h
        print(f"\n✓ Ready to source. {remaining} profile opens remaining in 24h budget.")
