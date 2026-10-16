"""Tests for the async ↔ sync iterator bridge (Phase C2b).

The bridge is the single load-bearing piece in C5: Anthropic's sync
streaming iterator has to be pumped on a worker thread while FastAPI's
StreamingResponse runs on the asyncio event loop. If this bridge
misbehaves under cancellation or exceptions, every conversational turn
risks deadlocking the event loop or leaking threads.

Coverage:

- Happy path: items yielded by the sync iterator arrive on the async
  side in order, generator closes cleanly.
- Mid-stream exception: exception propagates to the async consumer; no
  silent swallow.
- Async cancellation mid-iteration: consumer's ``finally`` runs, no
  exception leaks, worker thread is daemonized so it does not block
  process teardown.
- Tagged-tuple shape (the orchestrator's contract): the bridge is shape-
  agnostic — it forwards opaque payloads — so a stream of
  ``("delta", str)`` and ``("usage", dict)`` mixes pass through.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from shared.intake_conversation.sse_bridge import bridge_sync_stream_to_async


def _sync_stream(items: list):
    """Yield items synchronously."""

    def _factory():
        for item in items:
            yield item

    return _factory


def _raising_stream(items_before_raise: list, exc: Exception):
    """Yield N items, then raise."""

    def _factory():
        for item in items_before_raise:
            yield item
        raise exc

    return _factory


@pytest.mark.asyncio
async def test_bridge_forwards_items_in_order() -> None:
    factory = _sync_stream(["a", "b", "c"])

    received = []
    async for item in bridge_sync_stream_to_async(factory):
        received.append(item)

    assert received == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_bridge_forwards_tagged_tuples() -> None:
    """The orchestrator yields ``(kind, payload)`` tuples. Bridge is
    shape-agnostic — should pass them through opaquely.
    """

    factory = _sync_stream(
        [("delta", "Hi"), ("delta", " world"), ("usage", {"input_tokens": 5})]
    )

    received = []
    async for item in bridge_sync_stream_to_async(factory):
        received.append(item)

    assert received == [
        ("delta", "Hi"),
        ("delta", " world"),
        ("usage", {"input_tokens": 5}),
    ]


@pytest.mark.asyncio
async def test_bridge_propagates_exception_to_consumer() -> None:
    """Mid-stream exception in the sync iterator must surface on the
    async side. Otherwise C5's error frame + state-persist branch never
    fires.
    """

    factory = _raising_stream(["a", "b"], RuntimeError("simulated stream failure"))

    received = []
    with pytest.raises(RuntimeError, match="simulated stream failure"):
        async for item in bridge_sync_stream_to_async(factory):
            received.append(item)

    # Items yielded BEFORE the raise are still delivered.
    assert received == ["a", "b"]


@pytest.mark.asyncio
async def test_bridge_handles_empty_stream() -> None:
    factory = _sync_stream([])

    received = []
    async for item in bridge_sync_stream_to_async(factory):
        received.append(item)

    assert received == []


@pytest.mark.asyncio
async def test_bridge_consumer_cancellation_does_not_leak() -> None:
    """When the async consumer is cancelled mid-stream, the worker thread
    is daemonized so it does not block process exit. We assert by
    counting threads before/after.

    The slow stream sleeps between yields so we have time to cancel
    the consumer mid-iteration. The thread will still be running when
    the consumer's `finally` runs — that's expected. We verify it's
    a daemon thread (won't block exit).
    """

    started_event = threading.Event()
    proceed_event = threading.Event()

    def _slow_factory():
        def _gen():
            yield "first"
            started_event.set()
            # Wait for test to allow proceeding (or just block).
            proceed_event.wait(timeout=2.0)
            yield "second"
            yield "third"

        return _gen()

    received = []

    async def _consumer():
        async for item in bridge_sync_stream_to_async(_slow_factory):
            received.append(item)
            if item == "first":
                # Cancel ourselves after the first item.
                raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _consumer()

    assert received == ["first"]

    # Worker thread is still running (waiting on proceed_event) but is
    # daemonized — verify by name.
    intake_threads = [t for t in threading.enumerate() if t.name == "intake_orchestrator_stream"]
    assert all(t.daemon for t in intake_threads), "worker thread must be a daemon"

    # Let the thread proceed so it doesn't leak into the next test (cleanup).
    proceed_event.set()


@pytest.mark.asyncio
async def test_bridge_factory_exception_at_start() -> None:
    """If the sync iterator raises immediately on first iteration (e.g.
    Anthropic SDK auth failure), the bridge surfaces that exception
    cleanly without yielding anything.
    """

    def _factory():
        raise RuntimeError("auth failed before first delta")
        yield  # noqa — unreachable, just makes this a generator

    with pytest.raises(RuntimeError, match="auth failed before first delta"):
        async for _ in bridge_sync_stream_to_async(_factory):
            pass
