"""Langfuse client singleton — Phase 1 of Langfuse adoption.

Module-singleton wrapper around the Langfuse SDK so callers (LLM
client wrappers, judgment functions, orchestrators) never import
``langfuse`` directly. This keeps the codebase decoupled from the
Langfuse SDK version + lets tests and CI run without Langfuse
credentials.

## Defensive-null posture

Mirrors :func:`shared.runtime_state.calibration._open_readonly`'s
``mode=ro`` defensive pattern: missing config / unreadable resource
collapses to a non-raising no-op. Three null paths converge on
``_NullClient``:

1. **Keys absent.** ``LANGFUSE_PUBLIC_KEY`` or ``LANGFUSE_SECRET_KEY``
   unset (or empty) → no client, no-op stub. Lets tests + CI run
   cleanly without credentials.
2. **Disabled by env.** ``LANGFUSE_DISABLE=1`` (or any truthy
   sentinel) short-circuits even when keys are set. Rollback escape
   hatch for perf regressions or network hiccups bleeding into
   orchestrator latency. Same posture as keys-absent.
3. **Sticky network degradation.** Any network call that raises
   (timeout, 5xx, rate-limit) flips the module-singleton to a
   permanent no-op for the rest of the process. One bad call stops
   further attempts so we don't spam retries on every span emission.
   Re-instantiation requires process restart — that's intentional;
   in-process recovery would mask sustained outages.

## Secret handling

``LANGFUSE_SECRET_KEY`` is read once at instantiation and never
echoed. The singleton's ``__repr__`` returns a sentinel
("present" / "absent") rather than the raw key. Callers MUST NOT
include the secret key in span attributes, JSONL ``run_log`` records,
test fixtures, or error messages.

## Public surface

- :func:`get_client` — module singleton accessor; caller treats as
  opaque (only ``update_current_observation`` is consumed externally).
- :func:`update_current_observation` — pass-through to the active
  trace's current observation, used by
  ``shared/llm_usage.py:record_llm_usage`` to attach cost / token /
  cache stats to the in-flight LLM span.
- :func:`is_active` — boolean: True iff a real client is wired
  (keys present, not disabled, not network-degraded).
- :func:`reset_for_testing` — test helper to clear the singleton
  state between fixtures. NOT for production callers.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-singleton state (process-wide; thread-safe via _LOCK)
# ---------------------------------------------------------------------------


_LOCK = threading.Lock()
_INSTANTIATED = False
_CLIENT: "_LangfuseClientLike | None" = None


# Sentinel values that disable the client even when keys are present.
# Mirrors the truthy-coercion convention in
# ``market_intelligence.reflection._chief_of_staff_enabled`` so the
# operator's mental model stays consistent across env-var gates.
_DISABLE_SENTINELS: frozenset[str] = frozenset({"1", "true", "yes", "on"})


def _is_disabled_by_env() -> bool:
    raw = (os.environ.get("LANGFUSE_DISABLE", "") or "").strip().lower()
    return raw in _DISABLE_SENTINELS


# ---------------------------------------------------------------------------
# No-op stub — returned when Langfuse is absent / disabled / degraded
# ---------------------------------------------------------------------------


class _NullClient:
    """Drop-in stand-in for the real Langfuse client.

    Every method is a no-op. The shape mirrors the subset of the
    real client our codebase touches; adding a new method here
    (when a new caller lands) is the deliberate seam — it keeps the
    no-op surface explicit and prevents accidental behavior leaks.
    """

    is_null: bool = True

    def __repr__(self) -> str:
        # Never echoes any credential material; the sentinel keeps
        # the repr safe to log.
        return "<LangfuseClient: null/inactive>"

    def update_current_observation(self, **_kwargs: Any) -> None:
        return None

    def update_current_trace(self, **_kwargs: Any) -> None:
        return None

    def flush(self) -> None:
        return None

    def get_current_trace_id(self) -> str | None:
        return None

    def get_current_observation_id(self) -> str | None:
        return None

    def get_trace_url(self, *, trace_id: str | None = None) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Real-client wrapper — adapts the Langfuse SDK surface our callers consume
# ---------------------------------------------------------------------------


class _LangfuseClientLike:
    """Thin adapter over the Langfuse SDK.

    The SDK's import path varies across major versions (v2 exposes
    ``from langfuse.decorators import langfuse_context``; v3 exposes
    ``from langfuse import get_client``). The adapter resolves the
    available surface lazily so callers don't pin a major version.

    Sticky-degrade contract: any network call that raises an
    exception flips ``self._degraded`` to True and re-routes future
    calls to no-ops. The per-call ``try/except`` is intentionally
    broad — Langfuse SDK exceptions span network, schema-validation,
    and async-flush errors; we treat all of them as
    "Langfuse unavailable; preserve the caller's path."
    """

    is_null: bool = False

    def __init__(self, *, public_key: str, secret_key: str, host: str) -> None:
        self._public_key_present = bool(public_key)
        self._secret_key_present = bool(secret_key)
        self._host = host
        self._degraded = False
        self._inner: Any = None
        self._inner_module: Any = None

        # Lazy import so a repo without ``langfuse`` installed (CI without
        # the dep) still imports this module cleanly. The exception path
        # collapses to a degraded singleton so the calling code keeps
        # running — same behavior as the keys-absent path.
        try:
            import langfuse  # type: ignore[import-not-found]

            self._inner_module = langfuse
        except ImportError as exc:
            logger.debug(
                "langfuse SDK not installed; client degrading to no-op: %s",
                exc,
            )
            self._degraded = True
            return

        # Two SDK shapes: v3+ exposes get_client() as the primary; v2
        # exposes Langfuse() class directly. Try v3 first, fall through
        # to v2 if get_client is absent.
        try:
            if hasattr(langfuse, "get_client"):
                self._inner = langfuse.get_client(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
            else:
                self._inner = langfuse.Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
        except Exception as exc:  # noqa: BLE001 — sticky degrade
            logger.debug(
                "langfuse client instantiation failed; degrading to no-op: %s",
                exc.__class__.__name__,
            )
            self._degraded = True

    def __repr__(self) -> str:
        # Never echoes the raw secret key. Sentinel-only summary so
        # operator log dumps stay safe to share.
        public_state = "present" if self._public_key_present else "absent"
        secret_state = "present" if self._secret_key_present else "absent"
        degrade_state = "degraded" if self._degraded else "active"
        return (
            f"<LangfuseClient: public_key={public_state} "
            f"secret_key={secret_state} host={self._host} state={degrade_state}>"
        )

    def update_current_observation(self, **kwargs: Any) -> None:
        if self._degraded or self._inner is None:
            return None
        try:
            self._inner.update_current_observation(**kwargs)
        except Exception as exc:  # noqa: BLE001 — sticky degrade
            logger.debug(
                "langfuse update_current_observation failed; "
                "client now degraded for remainder of process: %s",
                exc.__class__.__name__,
            )
            self._degraded = True

    def update_current_trace(self, **kwargs: Any) -> None:
        if self._degraded or self._inner is None:
            return None
        try:
            self._inner.update_current_trace(**kwargs)
        except Exception as exc:  # noqa: BLE001 — sticky degrade
            logger.debug(
                "langfuse update_current_trace failed; "
                "client now degraded for remainder of process: %s",
                exc.__class__.__name__,
            )
            self._degraded = True

    def flush(self) -> None:
        if self._degraded or self._inner is None:
            return None
        try:
            self._inner.flush()
        except Exception as exc:  # noqa: BLE001 — sticky degrade
            logger.debug(
                "langfuse flush failed; "
                "client now degraded for remainder of process: %s",
                exc.__class__.__name__,
            )
            self._degraded = True

    def get_current_trace_id(self) -> str | None:
        if self._degraded or self._inner is None:
            return None
        try:
            return self._inner.get_current_trace_id()
        except Exception as exc:  # noqa: BLE001 — sticky degrade
            logger.debug(
                "langfuse get_current_trace_id failed; "
                "client now degraded for remainder of process: %s",
                exc.__class__.__name__,
            )
            self._degraded = True
            return None

    def get_current_observation_id(self) -> str | None:
        if self._degraded or self._inner is None:
            return None
        try:
            return self._inner.get_current_observation_id()
        except Exception as exc:  # noqa: BLE001 — sticky degrade
            logger.debug(
                "langfuse get_current_observation_id failed; "
                "client now degraded for remainder of process: %s",
                exc.__class__.__name__,
            )
            self._degraded = True
            return None

    def get_trace_url(self, *, trace_id: str | None = None) -> str | None:
        if self._degraded or self._inner is None:
            return None
        try:
            return self._inner.get_trace_url(trace_id=trace_id)
        except Exception as exc:  # noqa: BLE001 — sticky degrade
            logger.debug(
                "langfuse get_trace_url failed; "
                "client now degraded for remainder of process: %s",
                exc.__class__.__name__,
            )
            self._degraded = True
            return None


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_client() -> "_LangfuseClientLike | _NullClient":
    """Return the module-singleton Langfuse client.

    Idempotent: first call instantiates from env vars; subsequent
    calls return the same instance. Thread-safe via ``_LOCK``.

    Returns a :class:`_NullClient` when:
    - ``LANGFUSE_DISABLE=1`` (or any truthy sentinel).
    - ``LANGFUSE_PUBLIC_KEY`` or ``LANGFUSE_SECRET_KEY`` is unset / empty.
    - The Langfuse SDK is not installed.
    - SDK instantiation raises (network, schema, etc.).
    - A previous network call degraded the client (sticky).
    """

    global _INSTANTIATED, _CLIENT
    if _INSTANTIATED:
        # Return a fresh _NullClient view if the active instance
        # degraded post-init — keeps the "is_active" check honest.
        if _CLIENT is None:
            return _NullClient()
        if hasattr(_CLIENT, "_degraded") and getattr(_CLIENT, "_degraded"):
            return _NullClient()
        return _CLIENT

    with _LOCK:
        if _INSTANTIATED:
            assert _CLIENT is not None
            return _CLIENT

        if _is_disabled_by_env():
            logger.debug(
                "LANGFUSE_DISABLE truthy; using null client (rollback path)"
            )
            _CLIENT = _NullClient()
            _INSTANTIATED = True
            return _CLIENT

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
        host = (
            os.environ.get("LANGFUSE_HOST", "").strip()
            or "https://cloud.langfuse.com"
        )

        if not public_key or not secret_key:
            _CLIENT = _NullClient()
            _INSTANTIATED = True
            return _CLIENT

        # Real client. Construction can fail (SDK not installed, bad
        # host, etc.); the wrapper itself self-degrades on failure.
        client = _LangfuseClientLike(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        if client._degraded:
            _CLIENT = _NullClient()
        else:
            _CLIENT = client
        _INSTANTIATED = True
        return _CLIENT


def is_active() -> bool:
    """True iff a real (non-null, non-degraded) client is wired."""

    client = get_client()
    return not getattr(client, "is_null", True)


def update_current_observation(**kwargs: Any) -> None:
    """Pass-through helper used by ``shared.llm_usage.record_llm_usage``.

    Centralizes the "fetch singleton, call method" sequence so the
    LLM usage sink doesn't have to know about the singleton/null-stub
    distinction. No-ops when the singleton is null or degraded.
    """

    client = get_client()
    client.update_current_observation(**kwargs)


def update_current_trace(**kwargs: Any) -> None:
    """Pass-through helper to set trace-level metadata.

    Used by orchestrator entry points to tag traces with
    ``brief_id``, ``source``, etc. — attributes that should apply to
    the entire trace, not a single span.
    """

    client = get_client()
    client.update_current_trace(**kwargs)


def flush() -> None:
    """Force-flush the Langfuse client's outbound buffer.

    Useful at end-of-run paths so per-brief spans land in the
    cloud project before the process exits. No-op when null/degraded.
    """

    client = get_client()
    client.flush()


def get_current_trace_id() -> str | None:
    """Return the active Langfuse trace id, or ``None`` when inactive."""

    client = get_client()
    return client.get_current_trace_id()


def get_current_observation_id() -> str | None:
    """Return the active Langfuse observation id, or ``None`` when inactive."""

    client = get_client()
    return client.get_current_observation_id()


def get_trace_url(trace_id: str | None = None) -> str | None:
    """Return a trace URL for the active trace or the supplied ``trace_id``."""

    client = get_client()
    return client.get_trace_url(trace_id=trace_id)


def reset_for_testing() -> None:
    """Clear the module-singleton state. **Test-only.**

    Lets fixtures install different env-var configurations across
    tests without polluting state. Production callers must not call
    this — the singleton is designed to be process-wide.
    """

    global _INSTANTIATED, _CLIENT
    with _LOCK:
        _INSTANTIATED = False
        _CLIENT = None


__all__ = [
    "flush",
    "get_client",
    "get_current_observation_id",
    "get_current_trace_id",
    "get_trace_url",
    "is_active",
    "reset_for_testing",
    "update_current_observation",
    "update_current_trace",
]
