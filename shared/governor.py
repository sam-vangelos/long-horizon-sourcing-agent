"""Session Governor — enforces hard safety limits on sourcing sessions.

Profile-open limits are constants. Session duration is owned by the session
caller when provided, with a governor backstop for hard-stop safety. No
flags, no env vars, no "just five more" escape hatches.

The daily session-COUNT cap was removed by Sam's ruling on 2026-08-11
(CLO-153, plans/session-cap-removal.md): session counts never gate, refuse,
or delay a launch or resume — only profile-open budgets and detection
backoffs govern. Sessions are still RECORDED (cooldown.record_session_start /
record_session_end, counts_toward_cap as a historical annotation); do not
reintroduce a count gate here as an oversight fix.
"""

import random
import time
from typing import Optional

import shared.cooldown as cooldown
from shared.cooldown import GovernorStateUnreadable

# ──────────────────────────────────────────────────────────────────────
# HARD LIMITS (do not make these configurable)
# ──────────────────────────────────────────────────────────────────────

GOVERNOR_LEGACY_SESSION_DURATION_MIN_SECONDS = 3.5 * 3600
GOVERNOR_LEGACY_SESSION_DURATION_MAX_SECONDS = 4.5 * 3600
GOVERNOR_SESSION_BACKSTOP_GRACE_SECONDS = 600
MAX_PROFILE_OPENS_PER_SESSION = 200
MAX_PROFILE_OPENS_PER_24H = 400


def _format_hms(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


class GovernorLimitReached(Exception):
    """Raised when a governor limit is hit."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class SessionExpired(Exception):
    """Raised when the session duration cap is reached cooperatively."""
    def __init__(self, reason: str = "session_duration_cap"):
        self.reason = reason
        super().__init__(reason)


class OperatorStopRequested(Exception):
    """Raised when an operator stop is honored at a safe pipeline boundary."""
    def __init__(self, reason: str = "operator_stop"):
        self.reason = reason
        super().__init__(reason)


class SessionGovernor:
    """Enforces hard limits for a single sourcing session."""

    def __init__(self):
        self._session_start: float = 0.0
        self._session_duration_limit_seconds: float = 0.0
        self._profile_opens_session: int = 0
        self._active: bool = False
        self._shutdown_reason: Optional[str] = None
        # P8.2: set by force_backoff() when a hard external signal (LinkedIn
        # rate-limiting / blocking) is detected. Once set, every subsequent
        # limit check raises immediately regardless of the normal caps —
        # there is no refresh-and-retry path for a tripped governor.
        self._forced_backoff_reason: Optional[str] = None

    # ── Pre-session checks ──────────────────────────────────────────

    def can_start_session(self, session_type: str = "linkedin_sourcing") -> tuple[bool, str]:
        """Check all preconditions before starting a session.
        Returns (ok, reason).
        """
        try:
            # P8.2 follow-up: a persisted rate-limit/block backoff outranks every
            # other precondition — a blocked account must not be re-probed after
            # the ordinary dormant gap just because the process restarted.
            backoff = cooldown.get_active_backoff()
            if backoff is not None:
                remaining_min = max(0, (backoff["until"] - time.time()) / 60)
                return False, (
                    f"Forced backoff active ({backoff['reason']}), "
                    f"{remaining_min:.0f} min remaining"
                )

            # 24h profile open cap
            opens_24h = cooldown.get_profile_opens_24h()
            if opens_24h >= MAX_PROFILE_OPENS_PER_24H:
                return False, f"24h profile open cap reached ({opens_24h}/{MAX_PROFILE_OPENS_PER_24H})"

            return True, "ok"
        except GovernorStateUnreadable:
            return False, "governor state unreadable — refusing to start"

    # ── Session lifecycle ───────────────────────────────────────────

    def start_session(self, session_duration_seconds: float | None = None):
        """Mark session start. Call can_start_session() first."""
        self._session_start = time.time()
        if session_duration_seconds is None:
            self._session_duration_limit_seconds = random.uniform(
                GOVERNOR_LEGACY_SESSION_DURATION_MIN_SECONDS,
                GOVERNOR_LEGACY_SESSION_DURATION_MAX_SECONDS,
            )
        else:
            self._session_duration_limit_seconds = (
                float(session_duration_seconds) + GOVERNOR_SESSION_BACKSTOP_GRACE_SECONDS
            )
        self._profile_opens_session = 0
        self._active = True
        self._shutdown_reason = None
        self._forced_backoff_reason = None

    def end_session(self) -> dict:
        """Mark session end. Returns summary dict."""
        self._active = False
        return {
            "profile_opens_session": self._profile_opens_session,
            "profile_opens_24h": cooldown.get_profile_opens_24h(),
            "duration_seconds": int(time.time() - self._session_start),
            "shutdown_reason": self._shutdown_reason or "normal",
        }

    # ── Profile open hook ───────────────────────────────────────────
    #
    # P8.1: governance now attaches at LinkedInBrowser construction —
    # open_profile()/open_profile_by_url() call check_profile_open_or_raise()
    # and record_profile_open() on themselves via the governor they were
    # constructed with. The old wrap_browser()/unwrap_browser() monkeypatch
    # (dead — no callers) is removed; this is the single source of truth for
    # profile-open counting now.

    def _record_open(self):
        self._profile_opens_session += 1
        cooldown.record_profile_open()

    def record_profile_open(self):
        """Record one successful profile open explicitly."""
        self._record_open()

    def check_profile_open_or_raise(self):
        """Check limits before attempting a profile open."""
        self._check_limits_or_raise()

    # ── Forced backoff (P8.2) ────────────────────────────────────────

    def force_backoff(self, reason: str) -> None:
        """Immediately and permanently trip the governor for this session.

        Used when a hard external signal (LinkedIn rate-limiting / blocking)
        is detected — the session must stop now, not retry. Once tripped,
        every subsequent check_profile_open_or_raise()/check_limits() call
        raises GovernorLimitReached with this reason, regardless of the
        normal caps, until the next start_session() clears it.
        """
        self._shutdown_reason = reason
        self._forced_backoff_reason = reason
        # P8.2 follow-up: persist the backoff so the NEXT session (new
        # process, new governor) refuses to start until the cooldown
        # expires — the in-memory trip alone was erased by start_session().
        cooldown.record_forced_backoff(reason)

    # ── Limit checks ───────────────────────────────────────────────

    def _check_limits_or_raise(self):
        """Check all limits. Raises GovernorLimitReached if any exceeded."""
        if self._forced_backoff_reason:
            raise GovernorLimitReached(self._forced_backoff_reason)

        if not self._active:
            return

        # Session duration
        elapsed = time.time() - self._session_start
        if elapsed >= self._session_duration_limit_seconds:
            self._shutdown_reason = f"session_duration ({elapsed/3600:.1f}h)"
            raise GovernorLimitReached(self._shutdown_reason)

        # Session profile opens
        if self._profile_opens_session >= MAX_PROFILE_OPENS_PER_SESSION:
            self._shutdown_reason = f"session_profile_cap ({self._profile_opens_session}/{MAX_PROFILE_OPENS_PER_SESSION})"
            raise GovernorLimitReached(self._shutdown_reason)

        # 24h profile opens
        opens_24h = cooldown.get_profile_opens_24h()
        if opens_24h >= MAX_PROFILE_OPENS_PER_24H:
            self._shutdown_reason = f"24h_profile_cap ({opens_24h}/{MAX_PROFILE_OPENS_PER_24H})"
            raise GovernorLimitReached(self._shutdown_reason)

    def check_limits(self) -> Optional[str]:
        """Non-raising limit check. Returns reason string or None."""
        try:
            self._check_limits_or_raise()
            return None
        except GovernorLimitReached as e:
            return e.reason

    # ── Status ──────────────────────────────────────────────────────

    @property
    def profile_opens_session(self) -> int:
        return self._profile_opens_session

    @property
    def elapsed_seconds(self) -> float:
        if self._session_start == 0:
            return 0
        return time.time() - self._session_start

    @property
    def session_duration_limit_seconds(self) -> float:
        return self._session_duration_limit_seconds

    @property
    def shutdown_reason(self) -> Optional[str]:
        return self._shutdown_reason

    def status_line(self) -> str:
        """One-line status for console output."""
        elapsed = self.elapsed_seconds
        opens_24h = cooldown.get_profile_opens_24h()
        return (
            f"Profile opens: {self._profile_opens_session}/{MAX_PROFILE_OPENS_PER_SESSION} (session) | "
            f"{opens_24h}/{MAX_PROFILE_OPENS_PER_24H} (24h) | "
            f"Time: {_format_hms(elapsed)}/{_format_hms(self._session_duration_limit_seconds)}"
        )


class _UngovernedForTests:
    """Explicit, test-only stand-in for a real SessionGovernor.

    P8.1: LinkedInBrowser requires a governor at construction so production
    code cannot accidentally open profiles ungoverned. Tests that don't care
    about governance mechanics pass this sentinel instead of standing up a
    real SessionGovernor (which would mutate ~/.sourcing-governor/daily_stats.json).
    Every check is a no-op; every record is a no-op. Never pass this from a
    production code path — construction sites outside tests/ must pass the
    shared SessionGovernor instance.
    """

    def check_profile_open_or_raise(self) -> None:
        return None

    def record_profile_open(self) -> None:
        return None

    def force_backoff(self, reason: str) -> None:
        return None


UNGOVERNED_FOR_TESTS = _UngovernedForTests()
