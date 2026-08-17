"""Anthropic health pre-flight probes."""

from __future__ import annotations

import types
import sys
from types import SimpleNamespace

import pytest

from cloris import anthropic_health


class _Messages:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc

    def create(self, **_kwargs):
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=3,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            )
        )


class _Client:
    def __init__(
        self,
        *,
        api_key: str,
        timeout: float,
        exc: Exception | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.messages = _Messages(exc)


def _install_anthropic(monkeypatch: pytest.MonkeyPatch, exc: Exception | None = None) -> None:
    def _anthropic_ctor(*, api_key: str, timeout: float):
        return _Client(api_key=api_key, timeout=timeout, exc=exc)

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        types.SimpleNamespace(Anthropic=_anthropic_ctor),
    )


def test_anthropic_health_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic_health.clear_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(anthropic_health.config, "ANTHROPIC_API_KEY", "")

    health = anthropic_health.probe_anthropic_health(force=True)

    assert health.state == "missing"
    blocker = anthropic_health.launch_readiness_blocker()
    assert blocker is not None
    assert blocker.code == "anthropic_missing"


def test_anthropic_health_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic_health.clear_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    _install_anthropic(monkeypatch)
    usage_calls: list[dict] = []
    monkeypatch.setattr(
        anthropic_health,
        "record_llm_usage",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    health = anthropic_health.probe_anthropic_health(force=True)

    assert health.state == "healthy"
    assert anthropic_health.launch_readiness_blocker() is None
    assert usage_calls
    assert usage_calls[0]["actual_status"] == "ok"
    assert usage_calls[0]["usage"]["input_tokens"] == 3
    assert usage_calls[0]["usage_context"]["stage"] == "anthropic_health_probe"


def test_anthropic_health_budget_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic_health.clear_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    _install_anthropic(monkeypatch, RuntimeError("credit balance is too low"))
    usage_calls: list[dict] = []
    monkeypatch.setattr(
        anthropic_health,
        "record_llm_usage",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    health = anthropic_health.probe_anthropic_health(force=True)

    assert health.state == "api_budget_exhausted"
    blocker = anthropic_health.launch_readiness_blocker()
    assert blocker is not None
    assert blocker.code == "api_budget_exhausted"
    assert usage_calls
    assert usage_calls[0]["actual_status"] == "error"
    assert usage_calls[0]["request"]["error_type"] == "RuntimeError"
