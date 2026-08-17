"""Tests for the Researcher launch-readiness probe (Phase 2.2).

Pins the contract :mod:`researcher.health` provides for the launch-
readiness aggregator at ``cloris/api.py:_readiness_blockers``:

- ``probe_researcher_readiness()`` returns a structured
  :class:`ReadinessReport` with a config blocker for missing
  ``ANTHROPIC_API_KEY`` (which would crash the first judge call).
- ``ready=True`` when the key is configured.
- ``OPENALEX_POLITE_POOL_EMAIL`` is deliberately NOT a blocker;
  OpenAlex is queryable without it (the orchestrator passes ``""``
  when unset). Surfacing it would gate the recruiter on a non-fatal
  etiquette signal.

The probe doesn't hit any real network — every check is a pure env-var
presence read.
"""

from __future__ import annotations

import pytest

import researcher.health as researcher_health


def test_researcher_readiness_blocks_when_anthropic_key_missing() -> None:
    """No ANTHROPIC_API_KEY → config blocker pointing at .env setup."""

    report = researcher_health.probe_researcher_readiness(
        anthropic_api_key="",
    )

    assert report.ready is False
    assert len(report.blockers) == 1
    blocker = report.blockers[0]
    assert blocker.kind == "config"
    assert "Anthropic" in blocker.message
    assert "ANTHROPIC_API_KEY" in blocker.remediation


def test_researcher_readiness_passes_when_anthropic_key_present() -> None:
    """ANTHROPIC_API_KEY set → ready=True, no blockers."""

    report = researcher_health.probe_researcher_readiness(
        anthropic_api_key="real_anthropic",
    )

    assert report.ready is True
    assert report.blockers == ()


def test_researcher_readiness_does_not_block_on_missing_polite_pool_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENALEX_POLITE_POOL_EMAIL absence does NOT block.

    The orchestrator at ``researcher/session_orchestrator.py:75``
    reads the env var and passes ``""`` when unset; OpenAlex accepts
    that. Surfacing this as a launch blocker would gate the recruiter
    on a non-fatal etiquette signal — contrary to the
    linkedin/github precedent of "only block on what would actually
    fail the run."
    """

    monkeypatch.delenv("OPENALEX_POLITE_POOL_EMAIL", raising=False)

    report = researcher_health.probe_researcher_readiness(
        anthropic_api_key="real_anthropic",
    )

    assert report.ready is True
    assert report.blockers == ()


def test_researcher_readiness_falls_back_to_shared_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``anthropic_api_key`` defaults to None, the probe reads shared.config."""

    monkeypatch.setattr(researcher_health.config, "ANTHROPIC_API_KEY", "")

    report = researcher_health.probe_researcher_readiness()

    assert report.ready is False
    assert len(report.blockers) == 1
    assert "Anthropic" in report.blockers[0].message


def test_researcher_readiness_returns_readiness_report_shape() -> None:
    """Public contract: probe returns the dataclass shape the aggregator unions over."""

    report = researcher_health.probe_researcher_readiness(
        anthropic_api_key="real_anthropic",
    )

    assert isinstance(report, researcher_health.ReadinessReport)
    assert isinstance(report.blockers, tuple)
    for blocker in report.blockers:
        assert isinstance(blocker, researcher_health.ReadinessBlocker)
