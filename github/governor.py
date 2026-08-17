"""GitHub Session Governor — enforces hard safety limits on GitHub sourcing sessions.

Adapted from governor.py for API-based sourcing. Tracks API calls instead of
browser profile opens. No time-of-day window needed (no anti-detection concern).

All limits are constants. No flags, no env vars, no escape hatches.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import github.config as gc


# ---------------------------------------------------------------------------
# HARD LIMITS
# ---------------------------------------------------------------------------

MAX_SESSION_DURATION_SECONDS = gc.MAX_SESSION_DURATION_SECONDS
MAX_ENRICHMENTS_PER_SESSION = gc.MAX_ENRICHMENTS_PER_SESSION
MAX_SESSIONS_PER_DAY = gc.MAX_SESSIONS_PER_DAY

# State file
_STATS_FILE = gc.GITHUB_STATE_DIR / "daily_stats.json"
_SESSIONS_LOG = gc.GITHUB_STATE_DIR / "sessions.jsonl"


class GitHubGovernorLimitReached(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class GitHubSessionExpired(Exception):
    def __init__(self, reason: str = "session_duration_cap"):
        self.reason = reason
        super().__init__(reason)


# ---------------------------------------------------------------------------
# Persistent state helpers
# ---------------------------------------------------------------------------

def _load_stats() -> dict:
    gc.GITHUB_STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not _STATS_FILE.exists():
        return {"api_calls": [], "enrichments": [], "sessions_today": []}
    try:
        return json.loads(_STATS_FILE.read_text())
    except (json.JSONDecodeError, KeyError):
        return {"api_calls": [], "enrichments": [], "sessions_today": []}


def _save_stats(data: dict):
    gc.GITHUB_STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _STATS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(_STATS_FILE)


def _prune(data: dict) -> dict:
    """Remove entries older than 24 hours."""
    cutoff = time.time() - 24 * 3600
    data["api_calls"] = [ts for ts in data.get("api_calls", []) if ts > cutoff]
    data["enrichments"] = [ts for ts in data.get("enrichments", []) if ts > cutoff]
    today = time.strftime("%Y-%m-%d")
    data["sessions_today"] = [
        s for s in data.get("sessions_today", []) if s.get("date") == today
    ]
    return data


# ---------------------------------------------------------------------------
# Public stat helpers
# ---------------------------------------------------------------------------

def get_sessions_today() -> int:
    data = _prune(_load_stats())
    _save_stats(data)
    count = 0
    for entry in data["sessions_today"]:
        has_ended = "end_ts" in entry
        did_work = entry.get("enrichments", 0) > 0
        if did_work or not has_ended:
            count += 1
    return count


def record_session_start() -> int:
    data = _prune(_load_stats())
    today = time.strftime("%Y-%m-%d")
    session_num = len(data["sessions_today"]) + 1
    data["sessions_today"].append({
        "date": today,
        "session_num": session_num,
        "start_ts": time.time(),
    })
    _save_stats(data)
    return session_num


def record_session_end(session_num: int, enrichments: int, reason: str, stats: dict):
    data = _load_stats()
    today = time.strftime("%Y-%m-%d")
    for entry in data["sessions_today"]:
        if entry["session_num"] == session_num and entry.get("date") == today:
            entry["end_ts"] = time.time()
            entry["enrichments"] = enrichments
            entry["reason"] = reason
            break
    _save_stats(data)

    gc.GITHUB_STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_num": session_num,
        "enrichments": enrichments,
        "shutdown_reason": reason,
        "stats": stats,
    }
    with open(_SESSIONS_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def print_status():
    """Print current GitHub sourcing stats."""
    sessions = get_sessions_today()
    print(f"GitHub sessions today: {sessions}/{MAX_SESSIONS_PER_DAY}")
    print(f"Current time: {time.strftime('%I:%M %p')}")
    if sessions >= MAX_SESSIONS_PER_DAY:
        print("\n⚠ Daily session cap reached.")
    else:
        print(f"\n✓ Ready to source. {MAX_SESSIONS_PER_DAY - sessions} sessions remaining.")


# ---------------------------------------------------------------------------
# Governor class
# ---------------------------------------------------------------------------

class GitHubGovernor:
    """Enforces hard limits for a single GitHub sourcing session."""

    def __init__(self):
        self._session_start: float = 0.0
        self._enrichments_session: int = 0
        self._active: bool = False
        self._shutdown_reason: Optional[str] = None

    def can_start_session(self) -> tuple[bool, str]:
        sessions_today = get_sessions_today()
        if sessions_today >= MAX_SESSIONS_PER_DAY:
            return False, f"Daily session cap reached ({sessions_today}/{MAX_SESSIONS_PER_DAY})"
        return True, "ok"

    def start_session(self):
        self._session_start = time.time()
        self._enrichments_session = 0
        self._active = True
        self._shutdown_reason = None

    def end_session(self) -> dict:
        self._active = False
        return {
            "enrichments_session": self._enrichments_session,
            "duration_seconds": int(time.time() - self._session_start),
            "shutdown_reason": self._shutdown_reason or "normal",
        }

    def record_enrichment(self):
        self._enrichments_session += 1

    def check_limits(self) -> Optional[str]:
        """Non-raising limit check. Returns reason string or None."""
        if not self._active:
            return None

        elapsed = time.time() - self._session_start
        if elapsed >= MAX_SESSION_DURATION_SECONDS:
            self._shutdown_reason = f"session_duration ({elapsed/3600:.1f}h)"
            return self._shutdown_reason

        if self._enrichments_session >= MAX_ENRICHMENTS_PER_SESSION:
            self._shutdown_reason = f"enrichment_cap ({self._enrichments_session}/{MAX_ENRICHMENTS_PER_SESSION})"
            return self._shutdown_reason

        return None

    def check_limits_or_raise(self):
        reason = self.check_limits()
        if reason:
            raise GitHubGovernorLimitReached(reason)

    @property
    def enrichments_session(self) -> int:
        return self._enrichments_session

    @property
    def elapsed_seconds(self) -> float:
        if self._session_start == 0:
            return 0
        return time.time() - self._session_start

    @property
    def shutdown_reason(self) -> Optional[str]:
        return self._shutdown_reason

    def should_enter_enrichment_only(self, limiter_remaining: int) -> bool:
        """Check if we should stop new searches and only enrich existing candidates."""
        return limiter_remaining < 500

    def status_line(self) -> str:
        elapsed = self.elapsed_seconds
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        max_h = MAX_SESSION_DURATION_SECONDS // 3600
        return (
            f"Enrichments: {self._enrichments_session}/{MAX_ENRICHMENTS_PER_SESSION} | "
            f"Time: {h}:{m:02d}:{s:02d}/{max_h}:00:00"
        )
