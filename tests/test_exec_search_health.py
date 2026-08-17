"""Tests for the Executive Search launch-readiness probe (Phase 2.2).

Pins the contract :mod:`exec_search.health` provides for the launch-
readiness aggregator at ``cloris/api.py:_readiness_blockers``:

- ``probe_exec_search_readiness()`` returns a structured
  :class:`ReadinessReport` with a config blocker for missing
  ``ANTHROPIC_API_KEY``.
- ``ready=True`` when the key is configured.
- The optional signal keys (``CRUNCHBASE_API_KEY`` / ``NEWSAPI_KEY``
  / ``PITCHBOOK_API_KEY``) are deliberately NOT surfaced as
  blockers; their adapters degrade gracefully.

The probe doesn't hit any real network.
"""

from __future__ import annotations

import pytest

import exec_search.health as exec_search_health


def test_exec_search_readiness_blocks_when_anthropic_key_missing() -> None:
    """No ANTHROPIC_API_KEY → config blocker mentioning the dossier evaluator."""

    report = exec_search_health.probe_exec_search_readiness(
        anthropic_api_key="",
    )

    assert report.ready is False
    assert len(report.blockers) == 1
    blocker = report.blockers[0]
    assert blocker.kind == "config"
    assert "Anthropic" in blocker.message
    assert "ANTHROPIC_API_KEY" in blocker.remediation


def test_exec_search_readiness_passes_when_anthropic_key_present() -> None:
    """ANTHROPIC_API_KEY set → ready=True, no blockers."""

    report = exec_search_health.probe_exec_search_readiness(
        anthropic_api_key="real_anthropic",
    )

    assert report.ready is True
    assert report.blockers == ()


def test_exec_search_readiness_does_not_block_on_missing_signal_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crunchbase / NewsAPI / PitchBook keys are optional — adapters degrade gracefully.

    Surfacing them as launch blockers would contradict the "honest
    signal-unavailable placeholder" contract in
    ``exec_search/signals/crunchbase.py`` etc.
    """

    monkeypatch.delenv("CRUNCHBASE_API_KEY", raising=False)
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    monkeypatch.delenv("PITCHBOOK_API_KEY", raising=False)

    report = exec_search_health.probe_exec_search_readiness(
        anthropic_api_key="real_anthropic",
    )

    assert report.ready is True
    assert report.blockers == ()


def test_exec_search_readiness_falls_back_to_shared_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``anthropic_api_key`` defaults to None, the probe reads shared.config."""

    monkeypatch.setattr(exec_search_health.config, "ANTHROPIC_API_KEY", "")

    report = exec_search_health.probe_exec_search_readiness()

    assert report.ready is False
    assert len(report.blockers) == 1
    assert "Anthropic" in report.blockers[0].message


def test_exec_search_readiness_returns_readiness_report_shape() -> None:
    """Public contract: probe returns the dataclass shape the aggregator unions over."""

    report = exec_search_health.probe_exec_search_readiness(
        anthropic_api_key="real_anthropic",
    )

    assert isinstance(report, exec_search_health.ReadinessReport)
    assert isinstance(report.blockers, tuple)
    for blocker in report.blockers:
        assert isinstance(blocker, exec_search_health.ReadinessBlocker)
