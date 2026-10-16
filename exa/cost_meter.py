"""Per-run Exa spend meter — fail-closed reservation, measured settlement.

Every Exa call reserves a worst-case charge from :data:`PRICE_TABLE`
BEFORE the request leaves the process; a reservation that would cross
the run's cap raises :class:`ExaBudgetExhaustedError` instead of
dispatching. When the response comes back, ``settle`` replaces that
reservation with the API's own ``costDollars.total``.

When a response carries no usable ``costDollars``, the ESTIMATE stays
charged. Missing telemetry must never create imaginary budget headroom
— the same rule ``shared/llm_spend_budget.py`` applies to Fireworks.
"""

from __future__ import annotations

import math
import threading
from datetime import datetime, timezone
from pathlib import Path

from shared.failures import ApiBudgetExhaustedError
from shared.storage import append_jsonl, read_jsonl

import exa.config as ec


RECEIPT_SCHEMA_VERSION = "exa-usage-v1"

_EPSILON = 1e-12


class ExaBudgetExhaustedError(ApiBudgetExhaustedError):
    """Raised when the next Exa call would cross this run's spend cap."""


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------

def _content_type_count(contents: dict | None) -> int:
    """How many billable content types a ``contents`` block requests.

    Exa bills $1/1k pages PER content type, so text+highlights+summary on
    one page is three charges. A key counts as requested when it is
    present and not explicitly falsy (``{}`` and ``True`` both request
    the type; ``False``/``None`` do not).
    """

    if not isinstance(contents, dict):
        return 0
    count = 0
    for key in ec.PRICE_TABLE["content_types"]:
        if key not in contents:
            continue
        value = contents[key]
        if value is False or value is None:
            continue
        count += 1
    return count


def _finite_non_negative(value: object) -> float | None:
    """Coerce a JSON number to a usable dollar amount, or None."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return numeric


# ---------------------------------------------------------------------------
# Meter
# ---------------------------------------------------------------------------

class ExaCostMeter:
    """Process-local spend meter for one Exa run.

    ``cap_usd == 0`` disables the guard (offline callers, or callers with
    their own governor); any positive cap is enforced pre-dispatch.
    """

    def __init__(self, *, cap_usd: float, sink_path: Path | None = None):
        cap = float(cap_usd)
        if not math.isfinite(cap) or cap < 0:
            raise ValueError("Exa spend cap must be finite and >= 0")
        self._cap_usd = cap
        self._sink_path = Path(sink_path) if sink_path is not None else None
        self._committed_usd = 0.0
        self._call_count = 0
        self._lock = threading.Lock()

    # ── Estimates ────────────────────────────────────────────────────

    def estimate_search_usd(
        self,
        *,
        search_type: str,
        num_results: int,
        contents: dict | None,
    ) -> float:
        """Worst-case dollars for one ``/search`` call.

        Base request price by search type, plus $1/1k for every result
        past the tenth, plus $1/1k per page per requested content type.
        """

        table = ec.PRICE_TABLE
        base = table["search_request_usd"].get(
            search_type, table["search_request_usd"]["auto"]
        )
        results = max(int(num_results), 0)
        extra_results = max(results - int(table["search_results_included"]), 0)
        content_charges = (
            _content_type_count(contents) * results * table["content_page_usd"]
        )
        return base + extra_results * table["search_extra_result_usd"] + content_charges

    def estimate_contents_usd(
        self,
        *,
        num_urls: int,
        contents: dict | None,
    ) -> float:
        """Worst-case dollars for one ``/contents`` call.

        Floored at one content type. ``/contents`` has no "return
        nothing" mode: when the caller names no type (``None``, ``{}``,
        or every type explicitly off) Exa still serves — and bills —
        default text. Estimating that as $0 would let an unbounded
        number of billed calls slip past the cap.
        """

        urls = max(int(num_urls), 0)
        types = max(_content_type_count(contents), 1)
        return types * urls * ec.PRICE_TABLE["content_page_usd"]

    # ── Reservation / settlement ─────────────────────────────────────

    def reserve(self, estimate_usd: float) -> None:
        """Charge a worst-case estimate, or refuse the call outright.

        Fail-closed: the charge lands BEFORE dispatch, so a request whose
        outcome is unknown (sent, then lost) leaves its estimate committed
        rather than silently freeing budget. An attempt Exa provably never
        billed is handed back through :meth:`release`.
        """

        estimate = float(estimate_usd)
        if not math.isfinite(estimate) or estimate < 0:
            raise ValueError("Exa cost estimate must be finite and >= 0")
        with self._lock:
            if self._cap_usd == 0:
                self._committed_usd += estimate
                return
            projected = self._committed_usd + estimate
            if projected > self._cap_usd + _EPSILON:
                raise ExaBudgetExhaustedError(
                    "Exa run budget cannot accommodate the next call "
                    f"(cap=${self._cap_usd:.6f}, "
                    f"committed=${self._committed_usd:.6f}, "
                    f"estimate=${estimate:.6f})"
                )
            self._committed_usd = projected

    def release(self, estimate_usd: float) -> None:
        """Return one attempt's estimate that provably bought nothing.

        Two cases qualify, both decided by ``exa.client.ExaClient._request``:
        the connection was never established (DNS, TCP, TLS — the request
        never left the machine), or the server answered with a non-2xx
        status and no usage record. A fault after the request was sent
        leaves the outcome unknown — Exa may have run the search and lost
        the response — so that attempt stays charged. Cap-independent (a
        released charge never needs permission) and never drops below zero.
        """

        amount = float(estimate_usd)
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Exa cost release must be finite and >= 0")
        with self._lock:
            self._committed_usd = max(self._committed_usd - amount, 0.0)

    def settle(
        self,
        *,
        endpoint: str,
        estimate_usd: float,
        cost_dollars: dict | None,
        request_id: str | None,
        num_results: int,
    ) -> None:
        """Replace one reservation with the measured charge and log a receipt.

        The receipt append is fail-soft: usage telemetry must never take
        down a live run.
        """

        actual_usd = None
        if isinstance(cost_dollars, dict):
            actual_usd = _finite_non_negative(cost_dollars.get("total"))

        with self._lock:
            if actual_usd is not None:
                self._committed_usd += actual_usd - float(estimate_usd)
                if self._committed_usd < 0:
                    self._committed_usd = 0.0
            self._call_count += 1
            committed_total = self._committed_usd

        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "request_id": request_id,
            "estimate_usd": round(float(estimate_usd), 6),
            "actual_usd": None if actual_usd is None else round(actual_usd, 6),
            "usage_status": "measured" if actual_usd is not None else "unavailable",
            "num_results": int(num_results),
            "committed_total_usd": round(committed_total, 6),
        }
        self._append_receipt(receipt)

    def _append_receipt(self, receipt: dict) -> None:
        """Diagnostic persistence must never influence a live run."""

        if self._sink_path is None:
            return
        try:
            append_jsonl(self._sink_path, receipt)
        except Exception:
            pass

    # ── Status ───────────────────────────────────────────────────────

    @property
    def cap_usd(self) -> float:
        return self._cap_usd

    @property
    def committed_usd(self) -> float:
        with self._lock:
            return self._committed_usd

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "cap_enabled": self._cap_usd != 0,
                "cap_usd": self._cap_usd,
                "committed_usd": round(self._committed_usd, 6),
                "call_count": self._call_count,
            }

    def status_line(self) -> str:
        """One prebuilt line for the run console — never JSON, never multi-line."""

        with self._lock:
            committed = self._committed_usd
            cap = self._cap_usd
            calls = self._call_count
        if cap == 0:
            return f"Exa spend: ${committed:.4f}, no cap | {calls} calls"
        return f"Exa spend: ${committed:.4f} of ${cap:.2f} cap | {calls} calls"


# ---------------------------------------------------------------------------
# The receipt ledger
# ---------------------------------------------------------------------------

def ledger_spend_usd(path: Path) -> tuple[float | None, int]:
    """``(dollars, receipts)`` summed from an ``exa-usage.jsonl`` ledger.

    The ledger is the spend: one receipt per response that came back, with
    ``actual_usd`` when Exa measured it and ``estimate_usd`` when it did not
    (missing telemetry is never free). It is NOT the meter, which also
    holds reservations for in-flight and lost attempts and dies with its
    session. ``(None, 0)`` for a missing, unreadable, or empty ledger.
    """

    try:
        rows = [row for row in read_jsonl(path) if isinstance(row, dict)]
    except (OSError, ValueError):
        return None, 0
    total = 0.0
    counted = 0
    for row in rows:
        value = row.get("actual_usd")
        if _finite_non_negative(value) is None:
            value = row.get("estimate_usd")
        amount = _finite_non_negative(value)
        if amount is not None:
            total += amount
            counted += 1
    if not counted:
        return None, 0
    return round(total, 6), counted


__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "ExaBudgetExhaustedError",
    "ExaCostMeter",
    "ledger_spend_usd",
]
