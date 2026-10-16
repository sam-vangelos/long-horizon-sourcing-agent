"""Offline tests for the Exa spend meter (S1).

Locks the fail-closed reservation rule, the "missing telemetry keeps the
estimate charged" rule, the receipt contract, and the price table.
"""

import pytest

from exa.cost_meter import (
    RECEIPT_SCHEMA_VERSION,
    ExaBudgetExhaustedError,
    ExaCostMeter,
)
from shared.failures import ApiBudgetExhaustedError
from shared.storage import read_jsonl


_FULL_CONTENTS = {"text": True, "highlights": True, "summary": {"query": "role"}}


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------

def test_reserve_raises_when_cap_would_be_exceeded():
    meter = ExaCostMeter(cap_usd=0.010)

    meter.reserve(0.007)
    with pytest.raises(ExaBudgetExhaustedError):
        meter.reserve(0.007)

    # The refused call is not charged, and the guard is a shared
    # budget-exhaustion error so run-level handlers already catch it.
    assert meter.committed_usd == pytest.approx(0.007)
    assert issubclass(ExaBudgetExhaustedError, ApiBudgetExhaustedError)


def test_zero_cap_disables_guard():
    meter = ExaCostMeter(cap_usd=0)

    for _ in range(100):
        meter.reserve(1.0)

    assert meter.committed_usd == pytest.approx(100.0)
    assert meter.snapshot()["cap_enabled"] is False


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

def test_settle_replaces_estimate_with_measured_total():
    meter = ExaCostMeter(cap_usd=1.0)

    meter.reserve(0.037)
    meter.settle(
        endpoint="/search",
        estimate_usd=0.037,
        cost_dollars={"total": 0.016, "search": {"neural": 0.007, "keyword": 0}},
        request_id="req-1",
        num_results=3,
    )

    assert meter.committed_usd == pytest.approx(0.016)
    assert meter.call_count == 1


@pytest.mark.parametrize(
    "cost_dollars",
    [None, {}, {"total": None}, {"total": "free"}, {"total": -1.0}],
)
def test_settle_without_cost_dollars_keeps_estimate_charged(cost_dollars):
    meter = ExaCostMeter(cap_usd=1.0)

    meter.reserve(0.037)
    meter.settle(
        endpoint="/search",
        estimate_usd=0.037,
        cost_dollars=cost_dollars,
        request_id=None,
        num_results=0,
    )

    assert meter.committed_usd == pytest.approx(0.037)
    assert meter.call_count == 1


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------

def test_receipt_written_per_call_with_schema_version(tmp_path):
    sink = tmp_path / "usage" / "exa-usage.jsonl"
    meter = ExaCostMeter(cap_usd=1.0, sink_path=sink)

    meter.reserve(0.007)
    meter.settle(
        endpoint="/search",
        estimate_usd=0.007,
        cost_dollars={"total": 0.006},
        request_id="r1",
        num_results=2,
    )
    meter.reserve(0.002)
    meter.settle(
        endpoint="/contents",
        estimate_usd=0.002,
        cost_dollars=None,
        request_id=None,
        num_results=1,
    )

    records = read_jsonl(sink)
    assert len(records) == 2
    assert {record["schema_version"] for record in records} == {RECEIPT_SCHEMA_VERSION}
    assert RECEIPT_SCHEMA_VERSION == "exa-usage-v1"

    measured, unavailable = records
    assert measured["endpoint"] == "/search"
    assert measured["request_id"] == "r1"
    assert measured["usage_status"] == "measured"
    assert measured["actual_usd"] == pytest.approx(0.006)
    assert measured["estimate_usd"] == pytest.approx(0.007)
    assert measured["num_results"] == 2
    assert measured["committed_total_usd"] == pytest.approx(0.006)
    assert measured["timestamp"].endswith("+00:00")

    assert unavailable["endpoint"] == "/contents"
    assert unavailable["request_id"] is None
    assert unavailable["usage_status"] == "unavailable"
    assert unavailable["actual_usd"] is None
    assert unavailable["estimate_usd"] == pytest.approx(0.002)
    assert unavailable["committed_total_usd"] == pytest.approx(0.008)


def test_receipt_sink_failure_is_fail_soft(tmp_path):
    # Parent path is a FILE, so append_jsonl's mkdir cannot succeed.
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("occupied", encoding="utf-8")
    meter = ExaCostMeter(cap_usd=0, sink_path=blocker / "exa-usage.jsonl")

    meter.reserve(0.007)
    meter.settle(
        endpoint="/search",
        estimate_usd=0.007,
        cost_dollars={"total": 0.006},
        request_id="r1",
        num_results=1,
    )

    assert meter.call_count == 1
    assert meter.committed_usd == pytest.approx(0.006)


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------

def test_estimate_search_usd_matches_price_table():
    meter = ExaCostMeter(cap_usd=0)

    # $7/1k requests covers the first 10 results.
    assert meter.estimate_search_usd(
        search_type="auto", num_results=10, contents=None
    ) == pytest.approx(0.007)

    # +$1/1k per result past the tenth.
    assert meter.estimate_search_usd(
        search_type="auto", num_results=25, contents=None
    ) == pytest.approx(0.007 + 15 * 0.001)

    # Deep tiers carry their own request price.
    assert meter.estimate_search_usd(
        search_type="deep", num_results=10, contents=None
    ) == pytest.approx(0.012)
    assert meter.estimate_search_usd(
        search_type="deep-reasoning", num_results=10, contents=None
    ) == pytest.approx(0.015)

    # $1/1k per page PER content type — text + highlights + summary is 3x.
    assert meter.estimate_search_usd(
        search_type="auto", num_results=10, contents=_FULL_CONTENTS
    ) == pytest.approx(0.007 + 3 * 10 * 0.001)

    # An explicitly-disabled content type is not billed.
    assert meter.estimate_search_usd(
        search_type="auto",
        num_results=10,
        contents={"text": True, "highlights": False},
    ) == pytest.approx(0.007 + 10 * 0.001)


def test_estimate_contents_usd_charges_per_page_per_type():
    meter = ExaCostMeter(cap_usd=0)

    assert meter.estimate_contents_usd(
        num_urls=5, contents=_FULL_CONTENTS
    ) == pytest.approx(3 * 5 * 0.001)
    # contents=None takes Exa's default (text) — estimated as one type,
    # never as free.
    assert meter.estimate_contents_usd(num_urls=5, contents=None) == pytest.approx(
        5 * 0.001
    )
    assert meter.estimate_contents_usd(num_urls=0, contents=None) == 0.0


@pytest.mark.parametrize(
    "contents", [None, {}, {"text": False}, {"text": False, "summary": False}]
)
def test_estimate_contents_usd_treats_empty_map_as_default_text(contents):
    """``/contents`` has no free mode — naming no type still serves text.

    Estimating an empty contents map at $0 would let unlimited billed
    ``/contents`` calls walk straight through the cap.
    """

    meter = ExaCostMeter(cap_usd=0)

    assert meter.estimate_contents_usd(
        num_urls=10, contents=contents
    ) == pytest.approx(10 * 0.001)


def test_release_returns_an_unspent_estimate_without_going_negative():
    meter = ExaCostMeter(cap_usd=0.010)

    meter.reserve(0.007)
    meter.release(0.007)
    assert meter.committed_usd == pytest.approx(0.0)

    # Released budget is genuinely available again.
    meter.reserve(0.007)
    assert meter.committed_usd == pytest.approx(0.007)

    # Over-release floors at zero rather than manufacturing headroom by
    # going negative.
    meter.release(0.050)
    assert meter.committed_usd == pytest.approx(0.0)

    # Release is cap-independent: it never needs permission to refund.
    capped = ExaCostMeter(cap_usd=0.001)
    capped.release(0.5)
    assert capped.committed_usd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

def test_status_line_is_single_line():
    meter = ExaCostMeter(cap_usd=5.0)
    meter.reserve(0.0421)
    meter.settle(
        endpoint="/search",
        estimate_usd=0.0421,
        cost_dollars={"total": 0.0421},
        request_id="r",
        num_results=6,
    )

    line = meter.status_line()
    assert line == "Exa spend: $0.0421 of $5.00 cap | 1 calls"
    assert "\n" not in line
    assert "{" not in line

    uncapped = ExaCostMeter(cap_usd=0)
    assert "\n" not in uncapped.status_line()
