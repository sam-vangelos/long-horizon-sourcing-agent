"""LINKEDIN_CAMPAIGN_PERSIST: the day cycle survives closed governor windows.

Operator ruling (2026-07-27): an unattended campaign must not formally die at
the daily session cap or on a transient session error and wait for a human
relaunch — it sleeps through the closed window and resumes when the governor
reopens it. Persistence changes WHEN the process exits, never how much it
sources: every volume cap (sessions/day, opens/session, opens/24h) is enforced
exactly as before, and a COMPLETED campaign still exits.

Everything here is flag-gated; the first test class proves flag-off behavior
is byte-identical to the legacy cycle.
"""

from __future__ import annotations

import asyncio
import math
import random

from pathlib import Path

import pytest

from linkedin import session_orchestrator as so
from shared import config


# --- flag-off: legacy behavior preserved ------------------------------------


def test_defaults_reproduce_legacy_behavior() -> None:
    assert config.LINKEDIN_CAMPAIGN_PERSIST is False
    assert config.LINKEDIN_SESSION_ERROR_RETRIES == 0
    assert config.LINKEDIN_DORMANT_MEDIAN_MINUTES == 110
    assert config.LINKEDIN_DORMANT_MIN_MINUTES == 75
    assert config.LINKEDIN_DORMANT_MAX_MINUTES == 180


def test_flag_off_never_retries_an_error(monkeypatch) -> None:
    monkeypatch.setattr(config, "LINKEDIN_CAMPAIGN_PERSIST", False)
    monkeypatch.setattr(config, "LINKEDIN_SESSION_ERROR_RETRIES", 5)

    assert so._should_retry_session_error(
        "error: RuntimeError", consecutive_error_resumes=0
    ) is False


def test_dormant_defaults_match_the_historical_shape(monkeypatch) -> None:
    monkeypatch.setattr(config, "LINKEDIN_DORMANT_MEDIAN_MINUTES", 110.0)
    monkeypatch.setattr(config, "LINKEDIN_DORMANT_MIN_MINUTES", 75.0)
    monkeypatch.setattr(config, "LINKEDIN_DORMANT_MAX_MINUTES", 180.0)
    random.seed(7)

    samples = [so._sample_dormant_duration() for _ in range(500)]

    assert all(75 * 60 <= s <= 180 * 60 for s in samples)
    median = sorted(samples)[len(samples) // 2]
    assert abs(median - 110 * 60) < 15 * 60  # log-normal centered on the median


# --- dormant shape overrides -------------------------------------------------


def test_dormant_overrides_are_respected(monkeypatch) -> None:
    # The operator's stated short-break shape: ~20-45 min.
    monkeypatch.setattr(config, "LINKEDIN_DORMANT_MEDIAN_MINUTES", 30.0)
    monkeypatch.setattr(config, "LINKEDIN_DORMANT_MIN_MINUTES", 20.0)
    monkeypatch.setattr(config, "LINKEDIN_DORMANT_MAX_MINUTES", 45.0)
    random.seed(7)

    samples = [so._sample_dormant_duration() for _ in range(500)]

    assert all(20 * 60 <= s <= 45 * 60 for s in samples)
    median = sorted(samples)[len(samples) // 2]
    assert abs(median - 30 * 60) < 6 * 60


def test_degenerate_dormant_config_cannot_go_below_a_minute(monkeypatch) -> None:
    monkeypatch.setattr(config, "LINKEDIN_DORMANT_MEDIAN_MINUTES", 0.0)
    monkeypatch.setattr(config, "LINKEDIN_DORMANT_MIN_MINUTES", -5.0)
    monkeypatch.setattr(config, "LINKEDIN_DORMANT_MAX_MINUTES", 0.0)

    assert so._sample_dormant_duration() >= 60.0


# --- waiting out a closed governor window ------------------------------------


class _Governor:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0

    def can_start_session(self, session_type="linkedin_sourcing"):
        self.calls += 1
        ok = self.verdicts.pop(0) if self.verdicts else True
        return ok, "ok" if ok else "daily session limit reached"


def test_wait_for_governor_window_returns_when_it_reopens(monkeypatch) -> None:
    monkeypatch.setattr(so, "_GOVERNOR_WAIT_POLL_SECONDS", 0.01)
    governor = _Governor([False, False, True])
    stop = asyncio.Event()

    allowed = asyncio.run(so._wait_for_governor_window(governor, stop))

    assert allowed is True
    assert governor.calls == 3


def test_wait_for_governor_window_yields_to_operator_shutdown(monkeypatch) -> None:
    monkeypatch.setattr(so, "_GOVERNOR_WAIT_POLL_SECONDS", 30.0)
    governor = _Governor([False])  # then never asked again
    stop = asyncio.Event()

    async def scenario():
        task = asyncio.create_task(so._wait_for_governor_window(governor, stop))
        await asyncio.sleep(0.05)
        stop.set()  # Ctrl-C during the sleep must not wait out the poll
        return await asyncio.wait_for(task, timeout=1.0)

    assert asyncio.run(scenario()) is False


# --- bounded error absorption ------------------------------------------------


def test_error_retry_is_bounded_and_resets_semantics(monkeypatch) -> None:
    monkeypatch.setattr(config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(config, "LINKEDIN_SESSION_ERROR_RETRIES", 2)

    assert so._should_retry_session_error("error: RuntimeError", consecutive_error_resumes=0)
    assert so._should_retry_session_error("error: TimeoutError", consecutive_error_resumes=1)
    # Third consecutive error exceeds the budget — the campaign raises.
    assert not so._should_retry_session_error("error: RuntimeError", consecutive_error_resumes=2)


@pytest.mark.parametrize(
    "reason",
    [
        "geography_regime_error",
        "constraint_manifest_error",
        "preflight_regime_error",
        "interrupted: KeyboardInterrupt",
        "pipeline_complete",
        None,
    ],
)
def test_classified_reasons_are_never_absorbed(monkeypatch, reason) -> None:
    # Deterministic config errors loop forever if retried; operator interrupts
    # are a stop, not a fault. Only the unclassified "error: <Type>" family
    # qualifies for absorption.
    monkeypatch.setattr(config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(config, "LINKEDIN_SESSION_ERROR_RETRIES", 5)

    assert so._should_retry_session_error(reason, consecutive_error_resumes=0) is False


# --- the continuable set -----------------------------------------------------


def test_cap_shaped_reasons_cover_every_governor_limit_string() -> None:
    # The governor's own reason strings (shared/governor.py) must all be
    # recognized as "window closed", or a mid-session cap hit exits a
    # persistent campaign the way it exited the legacy one.
    for reason in (
        "session_duration (4.2h)",
        "session_profile_cap (200/200)",
        "24h_profile_cap (400/400)",
    ):
        assert reason.startswith(so._CAP_SHAPED_SHUTDOWN_PREFIXES), reason
    # Campaign completion is NOT cap-shaped — persistence never resurrects a
    # finished campaign.
    assert not "pipeline_complete".startswith(so._CAP_SHAPED_SHUTDOWN_PREFIXES)
    # And a DETECTION event ends unattended operation, full stop: the real
    # forced-backoff reason string (linkedin/recruiter_recovery.py:340) must
    # never read as continuable — auto-resuming into a possibly-flagged seat
    # is the one move persistence must never make.
    assert not "blocked_or_rate_limited".startswith(so._CAP_SHAPED_SHUTDOWN_PREFIXES)


# --- error backoff vs dormant gap (2026-07-30) -------------------------------
# The absorb path originally slept _sample_dormant_duration() — the
# anti-detection gap for sessions that actually touched LinkedIn. An absorbed
# session error made few or no opens (the 2026-07-30 live case: a Fireworks
# APITimeoutError, 0 opens), so the 41-minute dormant sleep burned a sixth of
# the session budget on a network blip. The two sleeps are now distinct.


def test_error_backoff_is_short_bounded_and_escalating() -> None:
    random.seed(11)
    first = [so._sample_error_backoff(1) for _ in range(200)]
    second = [so._sample_error_backoff(2) for _ in range(200)]
    huge = [so._sample_error_backoff(9) for _ in range(200)]

    assert all(60 <= s <= 120 for s in first)      # ~90s ±25%
    assert all(120 <= s <= 240 for s in second)    # ~180s ±25%
    assert all(s <= 300 * 1.25 for s in huge)      # hard cap
    # And it is categorically NOT the dormant gap: even the cap sits far
    # below the dormant minimum (20 min in the launcher, 75 min default).
    assert max(huge) < 20 * 60


def test_zero_activity_absorb_path_uses_backoff_not_dormant() -> None:
    # Zero-activity errors use the short backoff. A resumed renderer crash
    # after LinkedIn activity takes the dormant branch instead.
    # The loop lives in _run_day_cycle_with_browser_lock; read the module.
    src = Path(so.__file__).read_text()
    assert "error_absorbed_this_cycle" in src
    assert "error_absorbed_this_cycle and session_profile_opens <= 0" in src
    assert "_sample_error_backoff(consecutive_error_resumes)" in src
    assert '"dormant gap"' in src


def test_browser_crash_resume_switch_defaults_off() -> None:
    assert config.LINKEDIN_BROWSER_CRASH_RESUME_ENABLED is False


def test_facial_tightening_is_flag_gated_default_off() -> None:
    # The bias monitor's single verdict-affecting path (prompt injection at
    # both facial sites) rides LINKEDIN_FACIAL_TIGHTENING_ENABLED, default
    # false — completing the 2026-07-04 telemetry demotion. Telemetry paths
    # (alerts, block-report band, string_context) are deliberately ungated.
    assert config.LINKEDIN_FACIAL_TIGHTENING_ENABLED is False

    orch_src = Path(so.__file__).parent.joinpath("orchestrator.py").read_text()
    assert orch_src.count("config.LINKEDIN_FACIAL_TIGHTENING_ENABLED") == 2
    # Both injection sites sit behind the flag.
    for chunk in orch_src.split("LINKEDIN_FACIAL_TIGHTENING_ENABLED")[1:]:
        assert "get_tightening_status" in chunk[:600]


# --- 2026-07-30 adversarial audit fixes --------------------------------------
# An independent gpt-5.6-sol xhigh review of ce5ffa8..HEAD returned DO NOT SHIP
# with 18 findings. These lock the behavioural ones in the persistence layer.


def test_detection_backoff_ends_persistence_instead_of_being_waited_out(monkeypatch) -> None:
    # THE severest finding. _wait_for_governor_window looped until
    # can_start_session() went true WITHOUT reading the reason — and
    # cooldown.record_forced_backoff() persists a 6h block after a
    # rate-limit/blocked signal. So a persistent campaign would have slept off
    # a DETECTION event and re-probed a possibly-flagged seat unattended, the
    # one thing this feature's own docstring says it must never do.
    monkeypatch.setattr(so, "_GOVERNOR_WAIT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(
        so.cooldown, "get_active_backoff",
        lambda: {"reason": "blocked_or_rate_limited", "until": 9e18},
    )
    governor = _Governor([False, False, True])  # would eventually allow
    stop = asyncio.Event()

    allowed = asyncio.run(so._wait_for_governor_window(governor, stop))

    assert allowed is False, "persistence waited out a detection backoff"
    assert governor.calls == 1, "must refuse on the FIRST refusal, not poll on"


def test_volume_window_is_still_waited_out(monkeypatch) -> None:
    # The fix must not turn every refusal into a stop — daily/24h caps are
    # exactly what persistence exists to sleep through.
    monkeypatch.setattr(so, "_GOVERNOR_WAIT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(so.cooldown, "get_active_backoff", lambda: None)
    governor = _Governor([False, False, True])
    stop = asyncio.Event()

    assert asyncio.run(so._wait_for_governor_window(governor, stop)) is True
    assert governor.calls == 3


def test_a_session_that_opened_profiles_is_never_short_backoff_retried(monkeypatch) -> None:
    # The short backoff is justified ONLY by "no LinkedIn activity happened".
    # A lifecycle invariant blowing up after 199 opens has both a real activity
    # footprint and a deterministic fault; retrying it on a 90s timer is wrong
    # on both counts.
    monkeypatch.setattr(config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(config, "LINKEDIN_SESSION_ERROR_RETRIES", 2)

    assert so._should_retry_session_error(
        "error: APITimeoutError", consecutive_error_resumes=0, profile_opens=0
    ) is True
    assert so._should_retry_session_error(
        "error: ValueError", consecutive_error_resumes=0, profile_opens=199
    ) is False
    assert so._should_retry_session_error(
        "error: APITimeoutError", consecutive_error_resumes=0, profile_opens=1
    ) is False


def test_error_counter_resets_after_any_clean_cap_ended_session() -> None:
    # The reset was keyed to the literal "session_duration_cap", but the
    # governor emits "session_duration (4.2h)", "session_profile_cap (...)"
    # and "24h_profile_cap (...)" — so a healthy session between two errors
    # was counted as consecutive.
    src = Path(so.__file__).read_text()
    assert "_CAP_SHAPED_SHUTDOWN_PREFIXES\n            ) or shutdown_reason ==" in src
