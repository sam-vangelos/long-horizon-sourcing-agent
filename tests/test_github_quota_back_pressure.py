"""Tests for the GitHub rate limiter's quota-pooling back-pressure
(audit Move #21).

The existing :class:`RateLimiter` already blocks until reset when
``remaining == 0``. Move #21 adds graceful back-pressure when
``remaining`` is non-zero but below a configured fraction of
``limit``, so a burst doesn't push the limiter into the
budget-exhausted slow-path. The pacing math:

    back_pressure = (reset_at - now) / remaining

evenly spreads the residual budget across the rest of the window
(capped at 60s per call to keep individual call latency bounded).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from shared.rate_limiter import (
    BACK_PRESSURE_THRESHOLD_FRACTION,
    EndpointBudget,
    RateLimiter,
)


# ---------------------------------------------------------------------------
# EndpointBudget.back_pressure_seconds — pure logic
# ---------------------------------------------------------------------------


def test_back_pressure_is_zero_when_remaining_uninitialized() -> None:
    """No headers seen yet ⇒ no back-pressure (budget defaults are
    local-counting only)."""

    budget = EndpointBudget(name="rest", limit=5000, window=3600)
    assert budget.back_pressure_seconds() == 0.0


def test_back_pressure_is_zero_when_remaining_above_threshold() -> None:
    """At 50% remaining, no back-pressure — the budget is healthy."""

    budget = EndpointBudget(name="rest", limit=5000, window=3600)
    budget.update_from_headers(remaining=2500, reset_at=time.time() + 1800)
    assert budget.back_pressure_seconds() == 0.0


def test_back_pressure_is_zero_when_window_already_reset() -> None:
    """If the reset is in the past, the limiter should treat the
    window as fresh — no back-pressure."""

    budget = EndpointBudget(name="rest", limit=5000, window=3600)
    budget.update_from_headers(remaining=10, reset_at=time.time() - 100)
    assert budget.back_pressure_seconds() == 0.0


def test_back_pressure_is_zero_when_remaining_is_zero() -> None:
    """remaining == 0 is the exhausted path; seconds_until_available()
    handles that via the regular budget.available() check, not
    back-pressure."""

    budget = EndpointBudget(name="rest", limit=5000, window=3600)
    budget.update_from_headers(remaining=0, reset_at=time.time() + 600)
    assert budget.back_pressure_seconds() == 0.0


def test_back_pressure_paces_residual_budget_evenly() -> None:
    """200 remaining, 600s until reset → pace 600/200 = 3.0s per call."""

    budget = EndpointBudget(name="rest", limit=5000, window=3600)
    reset = time.time() + 600
    budget.update_from_headers(remaining=200, reset_at=reset)

    # 200/5000 = 4% which is below the 10% threshold ⇒ pacing fires.
    delay = budget.back_pressure_seconds()
    # Allow ±1s wobble because back_pressure_seconds reads time.time()
    # internally — the test compares against a fresh sample.
    assert 2.0 < delay < 4.5


def test_back_pressure_caps_at_60_seconds() -> None:
    """Avoid pathological per-call latency. Even with 1 remaining and
    a long window, no call should be paced more than 60s."""

    budget = EndpointBudget(name="rest", limit=5000, window=3600)
    budget.update_from_headers(remaining=1, reset_at=time.time() + 3600)
    delay = budget.back_pressure_seconds()
    assert delay <= 60.0


def test_back_pressure_threshold_is_ten_percent_of_limit() -> None:
    """Threshold pinning so a future re-tune is intentional."""

    assert BACK_PRESSURE_THRESHOLD_FRACTION == 0.10


# ---------------------------------------------------------------------------
# RateLimiter.acquire — integration: back-pressure fires after
# budget.available() passes
# ---------------------------------------------------------------------------


def test_acquire_applies_back_pressure_when_quota_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The acquire() loop should sleep the back-pressure interval
    after the regular budget.available() check passes."""

    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    # human_delay is the source of the per-call min-spacing variance;
    # pin it to a known floor so the back-pressure delay is the
    # dominant signal.
    from shared import rate_limiter as rl_mod

    monkeypatch.setattr(rl_mod, "human_delay", lambda lo, hi: 0.0)

    limiter = RateLimiter()
    # Simulate a fresh budget update from a 1% remaining response.
    limiter.update_from_headers(
        "rest",
        {
            "X-RateLimit-Limit": "5000",
            "X-RateLimit-Remaining": "50",  # 1% of 5000
            "X-RateLimit-Reset": str(int(time.time() + 1000)),
        },
    )

    asyncio.run(limiter.acquire("rest"))

    # The regular budget.available() check passes (remaining > 0), so
    # the only sleep should be the back-pressure delay (~20s = 1000/50).
    assert sleeps, "expected back-pressure sleep, got none"
    # 1000s / 50 remaining = 20s pacing target. The min-spacing delay
    # is pinned to 0 above, so any sleep here IS the back-pressure.
    assert any(15.0 < s < 25.0 for s in sleeps), (
        f"expected ~20s back-pressure delay; saw sleeps={sleeps}"
    )


def test_acquire_does_not_apply_back_pressure_when_quota_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At 90% remaining, no back-pressure — the only sleep should be
    the inter-request min-spacing (here pinned to 0)."""

    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    from shared import rate_limiter as rl_mod

    monkeypatch.setattr(rl_mod, "human_delay", lambda lo, hi: 0.0)

    limiter = RateLimiter()
    limiter.update_from_headers(
        "rest",
        {
            "X-RateLimit-Limit": "5000",
            "X-RateLimit-Remaining": "4500",  # 90% — healthy
            "X-RateLimit-Reset": str(int(time.time() + 600)),
        },
    )

    asyncio.run(limiter.acquire("rest"))
    # No back-pressure ⇒ no large sleeps. The min-spacing path is
    # pinned to 0 so we expect either no sleeps or only zero-duration
    # ones from the spacing path.
    assert all(s <= 1.0 for s in sleeps), (
        f"expected no large sleeps for healthy quota; saw {sleeps}"
    )


def test_back_pressure_does_not_fire_with_local_only_counting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before any X-RateLimit-* headers have landed, back-pressure
    must NOT fire — local-only counting is conservative by design.
    (Pre-Move-21 behavior is byte-identical when remaining < 0.)"""

    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    from shared import rate_limiter as rl_mod

    monkeypatch.setattr(rl_mod, "human_delay", lambda lo, hi: 0.0)

    limiter = RateLimiter()
    # No update_from_headers — budget.remaining stays at -1 (uninit).
    asyncio.run(limiter.acquire("rest"))
    assert all(s <= 1.0 for s in sleeps), (
        f"expected no back-pressure with uninit budget; saw {sleeps}"
    )
