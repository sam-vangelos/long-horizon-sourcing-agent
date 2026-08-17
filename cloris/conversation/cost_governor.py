"""Cost and rate limits for ambient narration SSE."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass


NARRATION_WINDOW_S = 30 * 60
MAX_NARRATION_TURNS_PER_WINDOW = 30
MAX_NARRATION_USD_PER_WINDOW = 1.50
DEFAULT_ESTIMATE_USD_PER_NARRATOR_CALL = 0.035
"""Conservative standby when tokenizer billing isn't plumbed."""

BATCH_DEBOUNCE_S = 30.0


@dataclass(frozen=True)
class GovernorDecision:
    allowed: bool
    reason: str | None
    suppressed_reason: str | None = None


class NarrationSpendGovernor:
    """Tracks per-brief sliding-window spend (turn count + USD estimate)."""

    def __init__(
        self,
        *,
        window_s: float = NARRATION_WINDOW_S,
        max_turns: int = MAX_NARRATION_TURNS_PER_WINDOW,
        max_usd: float = MAX_NARRATION_USD_PER_WINDOW,
        estimate_per_call_usd: float = DEFAULT_ESTIMATE_USD_PER_NARRATOR_CALL,
    ) -> None:
        self._window_s = window_s
        self._max_turns = max_turns
        self._max_usd = max_usd
        self._estimate = estimate_per_call_usd
        self._events: dict[str, deque[tuple[float, float]]] = defaultdict(deque)

    def _trim(self, brief_id: str, now: float) -> None:
        dq = self._events[brief_id]
        while dq and now - dq[0][0] > self._window_s:
            dq.popleft()

    def allow_call(self, brief_id: str) -> GovernorDecision:
        now = time.monotonic()
        self._trim(brief_id, now)
        dq = self._events[brief_id]
        turn_count = len(dq)
        window_usd = sum(usd for _, usd in dq)
        if turn_count >= self._max_turns:
            return GovernorDecision(
                False, "turn_cap", suppressed_reason="turn_cap",
            )
        if window_usd >= self._max_usd:
            return GovernorDecision(
                False, "dollar_cap", suppressed_reason="dollar_cap",
            )
        dq.append((now, self._estimate))
        return GovernorDecision(True, None)

    def record_actual_cost(self, brief_id: str, *, estimate_usd: float) -> None:
        """Optional: replace the last appended estimate with a better number."""

        dq = self._events.get(brief_id)
        if not dq:
            return
        ts, _ = dq[-1]
        dq[-1] = (ts, max(estimate_usd, 0.0))


_GLOBAL_GOVERNOR = NarrationSpendGovernor()


def default_narration_governor() -> NarrationSpendGovernor:
    return _GLOBAL_GOVERNOR
