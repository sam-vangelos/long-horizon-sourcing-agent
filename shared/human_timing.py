"""Human-like timing distributions for browser automation.

Human inter-event times follow log-normal distributions (heavy-tailed:
mostly quick, occasionally slow) with temporal autocorrelation (consecutive
delays are correlated). Uniform random delays are detectable by entropy-based
classifiers with near-100% accuracy.

Usage:
    from human_timing import human_delay, human_delay_correlated

    await asyncio.sleep(human_delay(1.5, 4.0))                       # single delay
    await asyncio.sleep(human_delay_correlated(2.0))                 # default stream
    await asyncio.sleep(human_delay_correlated(0.04, channel="scroll"))
"""

import math
import random

# Log-normal parameters (per deep research: μ≈1.5, σ≈0.8 for inter-action delays)
# These produce a median of ~4.5s with a long tail up to ~30s.
# For shorter actions we scale down.
_DEFAULT_MU = 1.5
_DEFAULT_SIGMA = 0.8

# State for correlated delays. Keep independent streams so micro-scroll cadence
# does not leak into page-turn, profile-read, or save/close pacing.
_last_delay_by_channel: dict[str, float] = {}
_CORRELATION_WEIGHT = 0.3  # How much the previous delay influences the next


def human_delay(low: float, high: float, mu: float = 0.0, sigma: float = 0.6) -> float:
    """Sample a log-normal delay, clamped to [low, high].

    If mu is 0, it's auto-calculated to center the distribution's median
    at the midpoint of [low, high].
    """
    if mu == 0:
        # Set mu so median = midpoint of range
        midpoint = (low + high) / 2.0
        mu = math.log(max(midpoint, 0.1))

    sample = random.lognormvariate(mu, sigma)
    return max(low, min(high, sample))


def reset_human_timing(channel: str | None = None) -> None:
    """Reset correlated timing state."""
    if channel is None:
        _last_delay_by_channel.clear()
        return
    _last_delay_by_channel.pop(channel, None)


def human_delay_correlated(base: float, spread: float = 0.6, *, channel: str = "default") -> float:
    """Sample a delay with temporal autocorrelation.

    Consecutive calls produce correlated values — a long pause makes the
    next pause slightly longer, and vice versa. This matches human behavior
    where attention shifts produce clustered timing.

    base: center of the delay range
    spread: log-normal sigma (higher = more variance)
    channel: independent pacing stream name
    """

    mu = math.log(max(base, 0.1))
    raw = random.lognormvariate(mu, spread)
    last_delay = _last_delay_by_channel.get(channel, 0.0)

    # Blend with previous delay for autocorrelation
    if last_delay > 0:
        blended = (1 - _CORRELATION_WEIGHT) * raw + _CORRELATION_WEIGHT * last_delay
    else:
        blended = raw

    # Clamp to reasonable bounds (0.3x to 4x the base)
    result = max(base * 0.3, min(base * 4.0, blended))
    _last_delay_by_channel[channel] = result
    return result
