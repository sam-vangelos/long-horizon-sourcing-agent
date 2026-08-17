"""Per-source minimum-spacing rate limiter.

Slice 2: each Researcher source client (OpenAlex, Semantic Scholar,
arXiv) enforces its own rate constraint via this small helper. The
existing :mod:`shared.rate_limiter` is GitHub-specific (token-bucket,
X-RateLimit-* header semantics, asyncio); the academic-publication
sources need something simpler — synchronous, source-agnostic, with
just a minimum-spacing guarantee.

Single-process semantics. Tests inject a fake clock to verify spacing
without real waits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MinSpacingLimiter:
    """Block until ``min_spacing_seconds`` have elapsed since the last call.

    The clock is injectable so tests can assert spacing without sleeping.
    Default uses :func:`time.monotonic` for the timer and :func:`time.sleep`
    for the wait.
    """

    min_spacing_seconds: float
    _last_call_at: float = -1.0
    _now: Callable[[], float] = field(default=time.monotonic)
    _sleep: Callable[[float], None] = field(default=time.sleep)

    def wait(self) -> float:
        """Block until the next call is allowed; return the wait duration."""

        now = self._now()
        if self._last_call_at < 0:
            self._last_call_at = now
            return 0.0
        elapsed = now - self._last_call_at
        gap = self.min_spacing_seconds - elapsed
        waited = 0.0
        if gap > 0:
            self._sleep(gap)
            waited = gap
            now = self._now()
        self._last_call_at = now
        return waited
