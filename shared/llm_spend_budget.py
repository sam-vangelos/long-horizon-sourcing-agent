"""Process-local, concurrency-safe reservation budget for Fireworks calls.

The live GLM canary uses this as a hard pre-dispatch guard.  Each logical call
reserves a conservative upper bound for every allowed attempt before touching
the provider.  Complete measured usage settles to actual estimated cost;
partial or unavailable usage keeps the full reservation charged so missing
telemetry can never create imaginary budget headroom.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from shared.failures import ApiBudgetExhaustedError
from shared.llm_usage import estimate_usage_cost_usd


FIREWORKS_SPEND_COHORT_CONTEXT_KEY = "_fireworks_spend_reservation_cohort"


@dataclass(frozen=True, slots=True)
class FireworksSpendReservation:
    reservation_id: int
    cap_usd: float
    reserved_usd: float


_LOCK = threading.Lock()
_CAP_USD: float | None = None
_COMMITTED_USD = 0.0
_NEXT_ID = 1
_ACTIVE: dict[int, float] = {}


def _positive_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return numeric


def reserve_fireworks_spend(
    *,
    cap_usd: float,
    model: str,
    input_token_upper_bound: int,
    max_output_tokens: int,
    max_attempts: int,
) -> FireworksSpendReservation | None:
    """Reserve a worst-case logical-call charge or fail before dispatch.

    A zero cap disables this optional guard for non-live/offline callers that
    carry their own spend governor.  Positive caps are immutable while any
    reservation or committed charge exists in the process.
    """

    cap = float(cap_usd)
    if cap == 0:
        return None
    cap = _positive_finite(cap, "Fireworks spend cap")
    if isinstance(input_token_upper_bound, bool) or int(input_token_upper_bound) < 0:
        raise ValueError("input_token_upper_bound must be non-negative")
    if isinstance(max_output_tokens, bool) or int(max_output_tokens) < 1:
        raise ValueError("max_output_tokens must be positive")
    if isinstance(max_attempts, bool) or int(max_attempts) < 1:
        raise ValueError("max_attempts must be positive")
    worst_case, rate_source = estimate_usage_cost_usd(
        model=model,
        input_tokens=int(input_token_upper_bound) * int(max_attempts),
        output_tokens=int(max_output_tokens) * int(max_attempts),
    )
    if worst_case is None or rate_source == "unknown":
        raise ApiBudgetExhaustedError(
            f"local Fireworks spend guard has no verified price for {model!r}"
        )
    reservation_usd = _positive_finite(worst_case, "Fireworks reservation")

    global _CAP_USD, _NEXT_ID
    with _LOCK:
        if _CAP_USD is None:
            _CAP_USD = cap
        elif not math.isclose(_CAP_USD, cap, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("Fireworks spend cap changed inside one process")
        projected = _COMMITTED_USD + sum(_ACTIVE.values()) + reservation_usd
        if projected > cap + 1e-12:
            raise ApiBudgetExhaustedError(
                "local Fireworks spend cap cannot accommodate the next bounded call "
                f"(cap=${cap:.6f}, committed_or_reserved=${projected:.6f})"
            )
        reservation_id = _NEXT_ID
        _NEXT_ID += 1
        _ACTIVE[reservation_id] = reservation_usd
    return FireworksSpendReservation(reservation_id, cap, reservation_usd)


def settle_fireworks_spend(
    reservation: FireworksSpendReservation | None,
    *,
    measured_cost_usd: float | None,
    usage_complete: bool,
) -> None:
    """Settle one reservation, conservatively charging unknown usage in full."""

    if reservation is None:
        return
    global _COMMITTED_USD
    with _LOCK:
        reserved = _ACTIVE.pop(reservation.reservation_id, None)
        if reserved is None:
            raise RuntimeError("Fireworks spend reservation was already settled")
        if usage_complete:
            if measured_cost_usd is None:
                raise ValueError("complete Fireworks usage requires measured cost")
            charge = float(measured_cost_usd)
            if not math.isfinite(charge) or charge < 0:
                raise ValueError("measured Fireworks cost must be finite and non-negative")
        else:
            charge = reserved
        _COMMITTED_USD += charge
        exceeded = _COMMITTED_USD + sum(_ACTIVE.values()) > reservation.cap_usd + 1e-12
    if exceeded:
        raise ApiBudgetExhaustedError(
            "measured Fireworks spend exceeded its conservative reservation/cap; "
            "provider work has stopped"
        )


def synchronize_fireworks_spend_cohort(
    reservation: FireworksSpendReservation | None,
    cohort: threading.Barrier | None,
    *,
    timeout_seconds: float,
) -> None:
    """Release a reserved call only when its whole dispatch cohort is funded.

    Live facial workers use one shared barrier.  Every worker reserves before
    waiting here; a sibling that cannot reserve aborts the barrier.  Successful
    siblings then release their unused reservations at zero cost and fail before
    provider dispatch, preventing a partially paid page.
    """

    if cohort is None:
        return
    if reservation is None:
        try:
            cohort.abort()
        finally:
            raise RuntimeError(
                "Fireworks spend cohort requires the live spend guard"
            )
    timeout = _positive_finite(timeout_seconds, "Fireworks cohort timeout")
    try:
        cohort.wait(timeout=timeout)
    except threading.BrokenBarrierError as exc:
        settle_fireworks_spend(
            reservation,
            measured_cost_usd=0.0,
            usage_complete=True,
        )
        raise ApiBudgetExhaustedError(
            "local Fireworks spend cohort could not fund every concurrent call; "
            "no call in the cohort was dispatched"
        ) from exc
    except BaseException:
        # The provider call starts only after this barrier returns.  An operator
        # interrupt or cancellation while waiting must therefore abort siblings
        # and release this untouched reservation rather than strand process-local
        # budget in ``_ACTIVE``.
        try:
            cohort.abort()
        except Exception:
            pass
        settle_fireworks_spend(
            reservation,
            measured_cost_usd=0.0,
            usage_complete=True,
        )
        raise


def fireworks_spend_budget_snapshot() -> dict[str, float | int | bool | None]:
    with _LOCK:
        return {
            "enabled": _CAP_USD is not None,
            "cap_usd": _CAP_USD,
            "committed_usd": round(_COMMITTED_USD, 6),
            "active_reservations": len(_ACTIVE),
            "active_reserved_usd": round(sum(_ACTIVE.values()), 6),
        }


def reset_fireworks_spend_budget_for_testing() -> None:
    global _CAP_USD, _COMMITTED_USD, _NEXT_ID
    with _LOCK:
        _CAP_USD = None
        _COMMITTED_USD = 0.0
        _NEXT_ID = 1
        _ACTIVE.clear()


__all__ = [
    "FIREWORKS_SPEND_COHORT_CONTEXT_KEY",
    "FireworksSpendReservation",
    "fireworks_spend_budget_snapshot",
    "reserve_fireworks_spend",
    "reset_fireworks_spend_budget_for_testing",
    "settle_fireworks_spend",
    "synchronize_fireworks_spend_cohort",
]
