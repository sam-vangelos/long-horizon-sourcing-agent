"""Token bucket rate limiter with per-endpoint budgets for GitHub API.

Tracks remaining requests using X-RateLimit-* response headers when available,
falls back to local token bucket counting otherwise. Uses human_timing for
inter-request spacing to avoid metronomic patterns.

Usage:
    limiter = RateLimiter()
    await limiter.acquire("search")       # blocks until budget available
    # ... make API call ...
    limiter.update_from_headers("search", response_headers)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from shared.human_timing import human_delay


# ---------------------------------------------------------------------------
# Endpoint budget definitions
# ---------------------------------------------------------------------------

# Audit Move #21: when server-reported `remaining` falls below this
# fraction of `limit`, RateLimiter.acquire() applies back-pressure —
# sleeping `(reset_at - now) / remaining` so the residual budget paces
# evenly across the rest of the window. Without this, a burst that
# leaves 200/5000 will exhaust to 0 quickly and the operator sits
# waiting on the budget-exhausted slow-path.
BACK_PRESSURE_THRESHOLD_FRACTION: float = 0.10


@dataclass
class EndpointBudget:
    """Rate limit state for a single endpoint class."""
    name: str
    limit: int          # max requests per window
    window: int         # window duration in seconds
    remaining: int = -1  # -1 = not yet initialized from headers
    reset_at: float = 0.0  # unix timestamp when window resets
    local_count: int = 0   # local count (used when headers not yet available)
    local_window_start: float = 0.0

    def _init_local(self):
        if self.local_window_start == 0.0:
            self.local_window_start = time.time()

    def consume_local(self):
        """Track a request locally (before headers are available)."""
        self._init_local()
        now = time.time()
        if now - self.local_window_start >= self.window:
            self.local_count = 0
            self.local_window_start = now
        self.local_count += 1

    def available(self) -> bool:
        """Check if a request can be made right now."""
        now = time.time()
        # If we have server-reported limits, use those
        if self.remaining >= 0:
            if now >= self.reset_at:
                return True  # Window has reset
            return self.remaining > 0
        # Fall back to local counting
        self._init_local()
        if now - self.local_window_start >= self.window:
            return True
        return self.local_count < self.limit

    def seconds_until_available(self) -> float:
        """How long until the next request is allowed."""
        if self.available():
            return 0.0
        now = time.time()
        if self.remaining >= 0 and self.reset_at > now:
            return self.reset_at - now
        # Local counting fallback
        return max(0.0, self.window - (now - self.local_window_start))

    def back_pressure_seconds(
        self,
        *,
        threshold_fraction: float = BACK_PRESSURE_THRESHOLD_FRACTION,
    ) -> float:
        """Return a back-pressure delay when ``remaining`` is below the
        threshold but not yet exhausted.

        Audit Move #21: paces the residual budget evenly across the
        rest of the window so a burst doesn't push us into the
        budget-exhausted slow-path. Returns 0.0 when:
        - Server-reported state is uninitialized (``remaining < 0``).
        - The window has already reset (``now >= reset_at``).
        - ``remaining`` is at or above the threshold (``threshold *
          limit``).
        - ``remaining`` is zero — that's the budget-exhausted case
          handled by ``available()`` + ``seconds_until_available()``.

        Otherwise returns ``(reset_at - now) / remaining`` so the next
        ``remaining`` calls land evenly between now and the reset.
        """

        if self.remaining < 0 or self.limit <= 0:
            return 0.0
        now = time.time()
        if now >= self.reset_at:
            return 0.0
        if self.remaining <= 0:
            return 0.0
        threshold = max(1, int(self.limit * threshold_fraction))
        if self.remaining >= threshold:
            return 0.0
        seconds_until_reset = self.reset_at - now
        # Pace the remaining `self.remaining` calls evenly until the
        # window resets. Cap at a reasonable per-call ceiling to keep
        # individual call latency bounded.
        per_call_pacing = seconds_until_reset / self.remaining
        return min(per_call_pacing, 60.0)

    def update_from_headers(self, remaining: int, reset_at: float, limit: Optional[int] = None):
        """Update state from GitHub X-RateLimit-* response headers."""
        self.remaining = remaining
        self.reset_at = reset_at
        if limit is not None:
            self.limit = limit


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

# Minimum spacing between requests to same endpoint (avoids burst patterns)
_MIN_SPACING_SECONDS = 0.3
_MAX_SPACING_SECONDS = 1.5


class RateLimiter:
    """Per-endpoint token bucket rate limiter for GitHub API.

    Endpoint classes:
        "rest"        — general REST API (5,000/hr)
        "search"      — user/repo search (30/min, shares REST budget)
        "code_search" — code search (10/min, most restrictive)
    """

    def __init__(self):
        import github.config as gc

        self._budgets: dict[str, EndpointBudget] = {
            "rest": EndpointBudget(
                name="rest",
                limit=gc.REST_RATE_LIMIT,
                window=gc.REST_RATE_WINDOW,
            ),
            "search": EndpointBudget(
                name="search",
                limit=gc.SEARCH_RATE_LIMIT,
                window=gc.SEARCH_RATE_WINDOW,
            ),
            "code_search": EndpointBudget(
                name="code_search",
                limit=gc.CODE_SEARCH_RATE_LIMIT,
                window=gc.CODE_SEARCH_RATE_WINDOW,
            ),
        }
        self._last_request_time: dict[str, float] = {}
        self._total_calls: int = 0

    async def acquire(self, endpoint: str = "rest") -> float:
        """Wait until a request is allowed for the given endpoint class.

        Also enforces minimum spacing between requests AND audit Move
        #21 back-pressure when remaining quota is low (but not yet
        exhausted) — paces the residual budget evenly across the rest
        of the window.

        Returns the actual wait time in seconds.
        """
        total_wait = 0.0
        budget = self._budgets.get(endpoint, self._budgets["rest"])

        # Wait for rate limit budget
        while not budget.available():
            wait = budget.seconds_until_available()
            if wait > 0:
                # Add a small buffer to avoid racing the reset
                wait = min(wait + 1.0, wait * 1.05 + 0.5)
                print(f"    [rate-limit] {endpoint}: waiting {wait:.1f}s for budget reset")
                await asyncio.sleep(wait)
                total_wait += wait

        # Audit Move #21 back-pressure: when remaining is below the
        # 10% threshold (server-reported), pace this call so the
        # residual budget spreads evenly across the rest of the
        # window. No-op when remaining is healthy or uninitialized.
        back_pressure = budget.back_pressure_seconds()
        if back_pressure > 0:
            print(
                f"    [rate-limit] {endpoint}: back-pressure {back_pressure:.1f}s "
                f"(remaining={budget.remaining}/{budget.limit})"
            )
            await asyncio.sleep(back_pressure)
            total_wait += back_pressure

        # Enforce minimum spacing (avoid burst patterns)
        last = self._last_request_time.get(endpoint, 0.0)
        elapsed = time.time() - last
        min_spacing = human_delay(_MIN_SPACING_SECONDS, _MAX_SPACING_SECONDS)
        if elapsed < min_spacing:
            gap = min_spacing - elapsed
            await asyncio.sleep(gap)
            total_wait += gap

        # Record the request
        budget.consume_local()
        self._last_request_time[endpoint] = time.time()
        self._total_calls += 1

        return total_wait

    def update_from_headers(self, endpoint: str, headers: dict) -> None:
        """Update rate limit state from GitHub response headers.

        GitHub includes these headers on every response:
            X-RateLimit-Limit: 5000
            X-RateLimit-Remaining: 4999
            X-RateLimit-Reset: 1372700873  (unix timestamp)
        """
        budget = self._budgets.get(endpoint, self._budgets["rest"])

        remaining = headers.get("X-RateLimit-Remaining")
        reset_at = headers.get("X-RateLimit-Reset")
        limit = headers.get("X-RateLimit-Limit")

        if remaining is not None and reset_at is not None:
            budget.update_from_headers(
                remaining=int(remaining),
                reset_at=float(reset_at),
                limit=int(limit) if limit else None,
            )

    def remaining(self, endpoint: str = "rest") -> int:
        """How many requests remain in the current window."""
        budget = self._budgets.get(endpoint, self._budgets["rest"])
        if budget.remaining >= 0:
            return budget.remaining
        # Local estimate
        return max(0, budget.limit - budget.local_count)

    @property
    def total_calls(self) -> int:
        return self._total_calls

    def status_line(self) -> str:
        """One-line status for console output."""
        parts = []
        for name, budget in self._budgets.items():
            rem = self.remaining(name)
            parts.append(f"{name}: {rem}/{budget.limit}")
        return f"API calls: {self._total_calls} total | " + " | ".join(parts)
