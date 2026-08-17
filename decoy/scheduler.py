"""Burst scheduling — timing for decoy agent activity bursts.

Provides two timing profiles:
  - Dormant: between sourcing sessions (median ~25 min between bursts)
  - Interleave: during active sourcing (median ~30 min between bursts)
"""

from shared.human_timing import human_delay


class BurstScheduler:
    """Generates autocorrelated inter-burst intervals."""

    def __init__(self, mode: str = "dormant"):
        """
        Args:
            mode: "dormant" or "interleave"
        """
        if mode == "dormant":
            # mu=3.2, sigma=0.6 in log-seconds → median ~25 min
            self._mu = 3.2
            self._sigma = 0.6
            self._min_s = 600    # 10 min
            self._max_s = 5400   # 90 min
        elif mode == "interleave":
            # mu=3.4, sigma=0.5 → median ~30 min (less frequent during sourcing)
            self._mu = 3.4
            self._sigma = 0.5
            self._min_s = 900    # 15 min
            self._max_s = 3600   # 60 min
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self._weight = 0.3  # Autocorrelation weight
        self._last_interval: float = 0.0

    def next_interval(self) -> float:
        """Get the next inter-burst interval in seconds.

        Uses human_delay for log-normal sampling, then applies autocorrelation.
        The mu/sigma values are in log-seconds, producing a raw sample in seconds
        that we treat as minutes (matching the spec convention).
        """
        import math
        import random

        # Sample raw from log-normal (result in ~minutes, convert to seconds)
        raw_minutes = math.exp(self._mu + self._sigma * random.gauss(0, 1))
        raw_seconds = raw_minutes * 60

        # Clamp
        fresh = max(self._min_s, min(self._max_s, raw_seconds))

        # Autocorrelation
        if self._last_interval > 0:
            blended = self._weight * self._last_interval + (1 - self._weight) * fresh
            result = max(self._min_s, min(self._max_s, blended))
        else:
            result = fresh

        self._last_interval = result
        return result
