"""Async ↔ sync iterator bridge for the conversational intake SSE endpoint.

Anthropic's ``client.messages.stream(...)`` is a sync context manager that
yields text deltas via a sync iterator. FastAPI's ``StreamingResponse``
wants an async generator. The orchestrator at C2 is a sync generator that
wraps the Anthropic stream and yields tagged tuples.

The single load-bearing piece in C5 is bridging the sync generator into
the async SSE response without blocking the event loop. This module is
the production-ready helper that does so.

Pattern: spawn a worker thread that drains the sync iterator, push each
yielded item onto an :class:`asyncio.Queue` from inside the thread via
:func:`asyncio.run_coroutine_threadsafe`, and yield from the queue on the
async side. Errors and end-of-stream are signaled via tagged tuples so
the consumer never has to inspect thread state directly.

Cancellation contract: when the async consumer is cancelled (SSE client
disconnects mid-stream, the FastAPI request is closed), the worker
thread is daemonized so it does not block process exit. The Anthropic
stream context manager will eventually exit naturally as the iterator
exhausts; we waste at most one Opus completion's worth of tokens. We do
not attempt a hard cancel of the in-flight stream because the SDK does
not expose a clean abort API.

C2b spike outcome (2026-05-13): pattern verified via :mod:`tests.test_intake_conversation_sse_bridge`.
- Happy path: deltas arrive incrementally on the async side. ✓
- Mid-stream exception: re-raised on the async side, generator closes. ✓
- Async cancellation mid-stream: consumer's ``finally`` runs, daemon
  thread continues briefly and exits without blocking. ✓
- The pattern is what C5 imports.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from typing import AsyncIterator, Callable, Iterator, TypeVar

T = TypeVar("T")


async def bridge_sync_stream_to_async(
    sync_iter_factory: Callable[[], Iterator[T]],
) -> AsyncIterator[T]:
    """Yield items from ``sync_iter_factory()`` on the async event loop.

    Spawns a daemon worker thread that drains the sync iterator and
    forwards each item over an :class:`asyncio.Queue`. The current event
    loop is captured at first await so cross-thread queue puts dispatch
    correctly under :func:`asyncio.run_coroutine_threadsafe`.

    Re-raises exceptions raised by the sync iterator on the consumer
    side so the SSE generator's exception handler (yield error frame +
    persist partial state) can run without inspecting thread internals.
    """

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _safe_put(item: tuple) -> None:
        """Put on the consumer's queue, swallowing loop-closure errors.

        The consumer can cancel (SSE client disconnect) or the test
        loop can tear down before the producer thread finishes draining
        the iterator. Either way the put will fail with CancelledError
        or RuntimeError; from the producer's perspective there is no
        consumer to notify, so we drop silently rather than letting
        the worker thread surface a noisy unhandled-exception warning.
        """

        try:
            asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
        except BaseException:  # noqa: BLE001 — consumer is gone
            pass

    def _producer() -> None:
        try:
            for item in sync_iter_factory():
                _safe_put(("item", item))
        except BaseException as exc:  # noqa: BLE001 — funnel everything to the consumer
            _safe_put(("error", exc))
        finally:
            _safe_put(("done", None))

    # Copy the calling coroutine's ContextVars (in particular,
    # ``shared.llm_usage._USAGE_LOG_PATH`` set by ``llm_usage_session``)
    # so ``record_llm_usage`` calls inside ``opus_llm_cached_stream``
    # write to the right session log when run from the worker thread.
    # Without this, raw ``threading.Thread`` would lose the context.
    ctx = contextvars.copy_context()

    thread = threading.Thread(
        target=ctx.run,
        args=(_producer,),
        name="intake_orchestrator_stream",
        daemon=True,
    )
    thread.start()

    try:
        while True:
            kind, payload = await queue.get()
            if kind == "done":
                return
            if kind == "error":
                raise payload  # type: ignore[misc]
            yield payload
    finally:
        # Daemon thread + Anthropic stream context manager handle teardown.
        # We don't join — the producer may still be draining a partial
        # Opus completion and we don't want to block consumer's cleanup.
        pass


__all__ = ["bridge_sync_stream_to_async"]
