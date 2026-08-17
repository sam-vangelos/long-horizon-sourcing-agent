"""Cloris observability layer — Phase 1 of Langfuse adoption.

Re-exports a graceful-degradation ``@observe()`` decorator + helpers
so callers (LLM client wrappers, judgment functions, orchestrators)
import from this module rather than ``langfuse`` directly. This
decouples the codebase from the Langfuse SDK version, lets tests +
CI run cleanly without Langfuse credentials, and centralizes the
rollback escape hatch (``LANGFUSE_DISABLE=1``) at one wrapper layer.

## Posture (mirrors langfuse_client.py)

The decorator returned by :func:`observe` is one of two shapes:

1. **Passthrough** — when ``LANGFUSE_DISABLE=1`` OR
   ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are absent OR
   the Langfuse SDK is not installed OR the client has degraded
   post-init (sticky network failure). The decorator becomes
   ``lambda f: f`` and call semantics are byte-equivalent to the
   undecorated function.
2. **Real** — when a healthy Langfuse client is wired. The decorator
   delegates to the SDK's ``@observe()`` which emits a span around
   each call.

## Attribute namespace conventions

Span attribute names follow per-subsystem prefixes so aggregate
filters compose cleanly across briefs:

- ``cascade.*`` — chief-of-staff synthesis + dispatch fallback routes,
  market-intelligence polish/planner cascades.
- ``vision.*`` — Designer's 4-layer vision-eval guard.
- ``judge.*`` — span names + attributes for ``shared/judger.py``
  entry points.
- ``dispatch.*`` — chief-of-staff dispatch-specific attributes.
- ``cost.*`` and ``latency.*`` — reserved for runtime metrics.

All attribute names use snake_case. Do not reuse these prefixes for
unrelated namespaces.

## Public surface

- :func:`observe` — the decorator. Replaces ``from langfuse import observe``.
- :func:`update_current_observation` — pass-through to the active
  span; used by ``shared/llm_usage.py:record_llm_usage`` to attach
  cost / token / cache stats.
- :func:`update_current_trace` — pass-through for trace-level metadata.
- :func:`flush` — force-flush the outbound buffer (end-of-run path).
- :func:`is_active` — boolean: True iff a real client is wired.

The :class:`langfuse_client._NullClient` and
:class:`langfuse_client._LangfuseClientLike` types stay private; use
the helpers above.
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
