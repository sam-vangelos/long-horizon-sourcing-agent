"""External research backend constructor failures must be visible in logs."""

from __future__ import annotations

import logging

import pytest

from market_intelligence import engine as engine_mod


def test_maybe_build_external_research_backend_logs_ctor_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    def _boom() -> None:
        raise RuntimeError("configured backend missing")

    monkeypatch.setattr(
        "market_intelligence.research_agent.build_external_research_backend",
        _boom,
        raising=True,
    )

    with caplog.at_level(logging.WARNING, logger="market_intelligence.engine"):
        assert engine_mod._maybe_build_external_research_backend() is None

    assert "External research backend unavailable" in caplog.text
    assert "RuntimeError" in caplog.text
