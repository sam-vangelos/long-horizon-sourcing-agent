"""Cached Anthropic readiness probe for launch pre-flight."""

from __future__ import annotations

import time
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from shared import config
from shared.failures import is_api_budget_exhausted_error
from shared.llm_usage import anthropic_usage_dict, record_llm_usage


AnthropicHealthState = Literal[
    "healthy",
    "missing",
    "unhealthy",
    "api_budget_exhausted",
]


@dataclass(frozen=True)
class AnthropicHealth:
    state: AnthropicHealthState
    message: str
    checked_at: str | None
    cache_age_s: float | None = None


@dataclass(frozen=True)
class AnthropicReadinessBlocker:
    kind: str
    message: str
    remediation: str
    code: str


_CACHE: tuple[float, AnthropicHealth] | None = None


def clear_cache() -> None:
    global _CACHE
    _CACHE = None


def _anthropic_api_key() -> str:
    """Read the live key so Settings/onboarding updates work without restart."""

    return str(os.getenv("ANTHROPIC_API_KEY") or config.ANTHROPIC_API_KEY or "")


def probe_anthropic_health(*, force: bool = False) -> AnthropicHealth:
    """Return cached Anthropic health, making one tiny live call if needed."""

    global _CACHE
    now = time.monotonic()
    if not force and _CACHE is not None:
        checked_mono, cached = _CACHE
        age = max(0.0, now - checked_mono)
        if age <= config.ANTHROPIC_HEALTH_CACHE_SECONDS:
            return AnthropicHealth(
                state=cached.state,
                message=cached.message,
                checked_at=cached.checked_at,
                cache_age_s=age,
            )

    checked_at = datetime.now(timezone.utc).isoformat()
    api_key = _anthropic_api_key().strip()
    if not api_key:
        health = AnthropicHealth(
            state="missing",
            message="Anthropic key is missing.",
            checked_at=checked_at,
            cache_age_s=0.0,
        )
        _CACHE = (now, health)
        return health

    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key,
            timeout=20.0,
        )
        message = client.messages.create(
            model=config.OPUS_MODEL_NAME,
            max_tokens=1,
            system="You are a health check. Reply with one token.",
            messages=[{"role": "user", "content": "ok"}],
        )
        _record_probe_usage(
            usage=anthropic_usage_dict(message),
            request={
                "max_tokens": 1,
                "system_prompt_chars": len(
                    "You are a health check. Reply with one token."
                ),
                "user_prompt_chars": 2,
            },
            actual_status="ok",
        )
    except Exception as exc:  # pragma: no cover - exact provider classes vary
        _record_probe_usage(
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            request={
                "max_tokens": 1,
                "system_prompt_chars": len(
                    "You are a health check. Reply with one token."
                ),
                "user_prompt_chars": 2,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:240],
            },
            actual_status="error",
        )
        if is_api_budget_exhausted_error(exc):
            health = AnthropicHealth(
                state="api_budget_exhausted",
                message="Anthropic API credits need attention.",
                checked_at=checked_at,
                cache_age_s=0.0,
            )
        else:
            health = AnthropicHealth(
                state="unhealthy",
                message="Anthropic is not ready for a run.",
                checked_at=checked_at,
                cache_age_s=0.0,
            )
        _CACHE = (now, health)
        return health

    health = AnthropicHealth(
        state="healthy",
        message="Anthropic is ready.",
        checked_at=checked_at,
        cache_age_s=0.0,
    )
    _CACHE = (now, health)
    return health


def _record_probe_usage(
    *,
    usage: dict,
    request: dict,
    actual_status: str,
) -> None:
    try:
        record_llm_usage(
            provider="anthropic",
            model=config.OPUS_MODEL_NAME,
            usage=usage,
            request={
                **request,
                "health_probe": True,
            },
            usage_context={
                "module": "cloris",
                "stage": "anthropic_health_probe",
            },
            actual_status=actual_status,
        )
    except Exception:
        pass


def launch_readiness_blocker() -> AnthropicReadinessBlocker | None:
    """Return a recruiter-safe blocker when Anthropic cannot launch work."""

    health = probe_anthropic_health()
    if health.state == "healthy":
        return None
    if health.state == "missing":
        return AnthropicReadinessBlocker(
            kind="auth",
            message="API setup needs attention.",
            remediation="Add your Anthropic key in Settings, then retry.",
            code="anthropic_missing",
        )
    if health.state == "api_budget_exhausted":
        return AnthropicReadinessBlocker(
            kind="auth",
            message="API credits need attention.",
            remediation="Add credits in Anthropic, then retry this search.",
            code="api_budget_exhausted",
        )
    return AnthropicReadinessBlocker(
        kind="net",
        message="API readiness needs attention.",
        remediation="Check Anthropic setup in Settings, then retry.",
        code="anthropic_unhealthy",
    )
