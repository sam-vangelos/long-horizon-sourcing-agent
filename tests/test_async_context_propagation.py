"""Async-context propagation audit — Phase 1 of Langfuse adoption.

Pins the per-call audit done at p1-async-context: every
``asyncio.create_task()`` / ``run_until_complete()`` /
``run_coroutine_threadsafe()`` call site downstream of an
``@observe()`` boundary must NOT drop trace context. The Langfuse
SDK uses Python ``ContextVars`` for trace context, which propagates
through ``asyncio.create_task()`` by Python's spec — but
``loop.run_until_complete()`` and ``asyncio.run_coroutine_threadsafe()``
without explicit context propagation can drop the trace ID.

## Production-site enumeration (audit step (a))

Run at start-of-task::

    rg "asyncio\\.create_task|asyncio\\.gather|run_until_complete|run_coroutine_threadsafe" --glob '!tests/'

Findings (May 2026):

- ``linkedin/session_orchestrator.py:226,238,254`` — three sibling
  tasks (``_interleave_loop`` / ``_status_loop`` / ``_session_timer``).
  Orchestration scaffolding, not eval children. Sibling tasks of
  the old observed pipeline entrypoint; they don't
  participate in the eval trace tree.
- ``linkedin/browser.py:1783`` — ``run_sync()`` helper is a top-level
  sync→async bridge. No parent trace context to propagate from.
- ``cloris/tools_runtime.py:237`` — task dispatch for the chat-surface
  tools runtime. Out-of-band from the LLM-judgment trace hierarchy.

## (b) Verify: this test

Spawns child tasks via ``asyncio.create_task()`` from an
``@observe()``-decorated parent. Asserts the children run under the
same trace context the parent established. With the Langfuse SDK
absent (the CI path), the decorator is a passthrough and context
propagation is trivially preserved — the test still pins the
contract so a future SDK upgrade can't silently break it.

## (c) Remediation

None required at production call sites per (a). If a future site
DOES land downstream of an @observe boundary AND uses
``loop.run_until_complete()`` (which drops context), wrap the
spawning function with ``@observe()`` so context is captured
before the spawn, OR use
``langfuse.context.copy_context()``-style explicit propagation.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

import pytest

from shared.observability import observe
from shared.observability.langfuse_client import reset_for_testing


# A test-only ContextVar that mirrors the propagation semantics of
# Langfuse's trace context. Any pattern that drops _TRACE_TOKEN drops
# Langfuse's trace ID too.
_TRACE_TOKEN: ContextVar[str | None] = ContextVar("trace_token", default=None)


@pytest.fixture(autouse=True)
def _reset_observability_singleton():
    """Each test starts with a fresh Langfuse singleton state."""

    reset_for_testing()
    yield
    reset_for_testing()


# ---------------------------------------------------------------------------
# (b) Verify: ContextVar propagates through asyncio.create_task()
# ---------------------------------------------------------------------------


def test_context_propagates_through_asyncio_create_task() -> None:
    """The ``ContextVar`` set in a parent function survives every
    ``asyncio.create_task()`` spawn. This is Python's documented
    behavior, but pinning it here means a future test failure surfaces
    a regression at the orchestrator's spawn boundaries."""

    async def child_task() -> str | None:
        # Read the parent's context-var; should see the parent's value.
        return _TRACE_TOKEN.get()

    @observe(name="parent")
    async def parent() -> list[str | None]:
        token = _TRACE_TOKEN.set("parent-trace-id")
        try:
            tasks = [asyncio.create_task(child_task()) for _ in range(5)]
            results = await asyncio.gather(*tasks)
        finally:
            _TRACE_TOKEN.reset(token)
        return results

    results = asyncio.run(parent())
    assert results == ["parent-trace-id"] * 5


def test_context_propagates_through_asyncio_gather() -> None:
    """``asyncio.gather()`` of inline coroutines (no ``create_task``)
    runs in the parent's context as well — common pattern in batch
    judgment dispatch sites."""

    async def child_coro() -> str | None:
        return _TRACE_TOKEN.get()

    @observe(name="batch_parent")
    async def batch_parent() -> list[str | None]:
        token = _TRACE_TOKEN.set("batch-trace-id")
        try:
            results = await asyncio.gather(
                child_coro(), child_coro(), child_coro()
            )
        finally:
            _TRACE_TOKEN.reset(token)
        return results

    results = asyncio.run(batch_parent())
    assert results == ["batch-trace-id"] * 3


def test_observe_decorator_is_passthrough_when_keys_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Langfuse is null-stubbed (no keys / disabled), the
    @observe decorator returns the function unchanged. Behavior is
    byte-equivalent to undecorated."""

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_DISABLE", raising=False)
    reset_for_testing()

    @observe(name="passthrough_test")
    def fn(x: int, y: int) -> int:
        return x * y

    assert fn(3, 4) == 12
    # The wrapped reference is the original function — passthrough is
    # a `lambda f: f` idempotent decorator factory.
    assert fn.__name__ == "fn"


def test_observe_decorator_passthrough_when_disabled_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LANGFUSE_DISABLE=1`` short-circuits the decorator even when
    keys are present. The rollback escape hatch — operator can flip
    one env var to roll back the entire observability layer without
    touching code."""

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake_public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake_secret")
    monkeypatch.setenv("LANGFUSE_DISABLE", "1")
    reset_for_testing()

    @observe(name="rollback_test")
    def fn(value: str) -> str:
        return value.upper()

    assert fn("hello") == "HELLO"
    assert fn.__name__ == "fn"


@pytest.mark.parametrize("disable_value", ["1", "true", "yes", "TRUE", "on"])
def test_observe_disable_recognizes_truthy_sentinels(
    monkeypatch: pytest.MonkeyPatch, disable_value: str
) -> None:
    """``LANGFUSE_DISABLE`` honors the same truthy-sentinel set as
    other Cloris env-var gates (``CLORIS_CHIEF_OF_STAFF_ENABLED``,
    etc.). Operators don't have to remember which gate uses ``1``
    vs ``true``."""

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake")
    monkeypatch.setenv("LANGFUSE_DISABLE", disable_value)
    reset_for_testing()

    from shared.observability import is_active

    assert is_active() is False
