"""Cloris observability layer for optional Langfuse tracing.

Re-export a graceful-degradation ``@observe()`` decorator and helpers so
LLM clients, judgment functions, and orchestrators import this boundary
instead of the SDK directly. This keeps callers independent of SDK version,
lets tests run without Langfuse credentials, and centralizes the
``LANGFUSE_DISABLE=1`` escape hatch.

## Posture

The decorator returned by :func:`observe` has two shapes:

1. **Passthrough** — when ``LANGFUSE_DISABLE=1``, when
   ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are absent, when the SDK
   is not installed, or when the client degraded after initialization. The
   decorator is ``lambda f: f`` and call semantics match the undecorated
   function.
2. **Real** — when a healthy Langfuse client is wired. The decorator delegates
   to the SDK's ``@observe()`` and emits a span around each call.

## Attribute namespace conventions

Span attributes use subsystem prefixes: ``cascade.*`` for chief-of-staff and
market-intelligence cascades, ``vision.*`` for Designer evaluation,
``judge.*`` for ``shared/judger.py``, ``dispatch.*`` for chief-of-staff
dispatch, and ``cost.*`` / ``latency.*`` for runtime metrics. All attributes
use snake_case; do not reuse these prefixes for unrelated namespaces.

## Public surface

- :func:`observe` — the decorator boundary.
- :func:`update_current_observation` — attach cost, token, and cache stats.
- :func:`update_current_trace` — attach trace-level metadata.
- :func:`flush` — force-flush the outbound buffer.
- :func:`is_active` — true only when a real client is wired.

The null and client-like implementation types stay private; callers use the
helpers above.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, TypeVar

from shared.observability.langfuse_client import (
    flush,
    get_client,
    get_current_observation_id,
    get_current_trace_id,
    get_trace_url,
    is_active,
    update_current_observation,
    update_current_trace,
)

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])


def _passthrough_decorator(*_args: Any, **_kwargs: Any) -> Callable[[F], F]:
    """Decorator factory that returns the function untouched.

    Mirrors the ``@observe(...)`` calling convention so call sites
    don't need to branch on whether Langfuse is wired:

        @observe(as_type="generation")
        def opus_llm(...): ...

    is byte-equivalent to ``def opus_llm(...): ...`` when the
    passthrough fires.
    """

    def _decorator(fn: F) -> F:
        return fn

    return _decorator


def observe(
    *args: Any,
    **kwargs: Any,
) -> Callable[[F], F]:
    """Graceful-degradation wrapper around the Langfuse @observe decorator.

    Two routes:

    1. **Active.** Real Langfuse client wired (keys set, not disabled,
       SDK installed, not network-degraded). The wrapper delegates to
       Langfuse's ``@observe(...)`` decorator which emits a span on
       every call.
    2. **Passthrough.** Any of the null-paths in
       :func:`shared.observability.langfuse_client.get_client` apply.
       Returns the function untouched. Call semantics are byte-
       equivalent to undecorated.

    Usage matches the Langfuse SDK directly:

    .. code-block:: python

        from shared.observability import observe

        @observe(as_type="generation", name="opus_llm")
        def opus_llm(system, user): ...

        @observe(name="judge.facial")
        def facial_judge(snippet, brief): ...

    The wrapper resolves the routing at decoration time (NOT call
    time) so production overhead is one branch evaluated at module
    import. If the env-var state changes mid-process (e.g., a test
    flips ``LANGFUSE_DISABLE``), call :func:`langfuse_client.reset_for_testing`
    to re-evaluate.

    Per-call exceptions inside the SDK's ``@observe`` body
    (instrumentation hiccups, schema-validation errors) are NOT
    swallowed here — Langfuse's own ``@observe`` is responsible for
    its internal failure modes. If those bleed into caller paths in
    practice, wrap the SDK call in a per-call try/except at this
    layer; the sticky-degrade in
    :class:`langfuse_client._LangfuseClientLike` already covers
    ``update_current_observation`` / ``update_current_trace`` /
    ``flush``.
    """

    if not is_active():
        return _passthrough_decorator(*args, **kwargs)

    # Lazy import — only reached when a real client is wired, so the
    # repo-without-langfuse path stays in the passthrough branch and
    # this import never runs.
    try:
        # v3+ SDK: ``from langfuse import observe``.
        from langfuse import observe as _real_observe  # type: ignore[import-not-found]
    except ImportError:
        try:
            # v2 SDK: ``from langfuse.decorators import observe``.
            from langfuse.decorators import observe as _real_observe  # type: ignore[import-not-found]
        except ImportError:
            logger.debug(
                "langfuse @observe decorator unavailable; falling back to passthrough"
            )
            return _passthrough_decorator(*args, **kwargs)

    # Defensive shim: even when the real decorator is wired, swallow
    # any exception it raises at decoration time so a malformed
    # SDK-version mismatch doesn't crash module import. The
    # passthrough preserves the caller's path; the lost spans surface
    # as gaps in the trace UI rather than runtime errors.
    try:
        kwargs.update(capture_input=False, capture_output=False)
        return _real_observe(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "langfuse @observe decoration failed (%s); falling back to passthrough",
            exc.__class__.__name__,
        )
        return _passthrough_decorator(*args, **kwargs)


def observe_as_passthrough(fn: F) -> F:
    """Test-only helper that always returns the function untouched.

    Useful for tests that need to assert behavior with the
    passthrough decorator regardless of env-var state. Production
    callers must use :func:`observe`.
    """

    @wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return _wrapper  # type: ignore[return-value]


__all__ = [
    "flush",
    "get_client",
    "get_current_observation_id",
    "get_current_trace_id",
    "get_trace_url",
    "is_active",
    "observe",
    "observe_as_passthrough",
    "update_current_observation",
    "update_current_trace",
]
