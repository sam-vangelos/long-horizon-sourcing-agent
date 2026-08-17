"""Tests for the Designer launch-readiness probe (audit Move #14).

Pins the CSE-primary contract :mod:`designer.health` provides for the
launch-readiness aggregator at ``cloris/api.py:_readiness_blockers``:

- ``probe_designer_readiness()`` returns a structured
  :class:`ReadinessReport` with hard blockers for missing
  ``ANTHROPIC_API_KEY`` and for the case where BOTH acquisition
  sources (CSE + Behance) are absent.
- Behance is no longer a hard blocker on its own — Adobe stopped
  issuing v2 keys in 2020. CSE-only is a supported posture per
  audit Move #14.
- ``ready=True`` when ANTHROPIC_API_KEY is set AND at least one
  source (CSE or Behance) is configured.
- Soft warnings about source-mix tradeoffs land in
  ``informational_notes`` — non-blocking; the launch UI surfaces
  them inline.

The probe doesn't hit any real network — every check is a pure env-var
presence read.
"""

from __future__ import annotations

import pytest

import designer.health as designer_health


# ---------------------------------------------------------------------------
# Hard blockers
# ---------------------------------------------------------------------------


def test_designer_readiness_blocks_when_no_acquisition_source_configured() -> None:
    """Audit Move #14: with neither CSE nor Behance configured, the
    probe blocks (the recruiter can't discover candidates without at
    least one source)."""

    report = designer_health.probe_designer_readiness(
        behance_api_key="",
        anthropic_api_key="real_anthropic",
        google_cse_api_key="",
        google_cse_id="",
    )
    assert report.ready is False
    blockers = report.blockers
    assert len(blockers) == 1
    blocker = blockers[0]
    assert blocker.kind == "config"
    assert "acquisition source" in blocker.message
    assert "GOOGLE_CSE_API_KEY" in blocker.remediation
    assert "BEHANCE_API_KEY" in blocker.remediation


def test_designer_readiness_passes_with_cse_only_per_move_14() -> None:
    """Audit Move #14 load-bearing: CSE-only is a supported posture.
    No more "blocked indefinitely without pre-2020 Behance key."
    """

    report = designer_health.probe_designer_readiness(
        behance_api_key="",
        anthropic_api_key="real_anthropic",
        google_cse_api_key="real_cse_key",
        google_cse_id="real_cse_id",
    )
    assert report.ready is True
    assert report.blockers == ()
    # Soft warning that Behance is missing surfaces in
    # informational_notes (non-blocking).
    assert any(
        "Behance" in note or "BEHANCE_API_KEY" in note
        for note in report.informational_notes
    )


def test_designer_readiness_passes_with_behance_only_legacy_posture() -> None:
    """Behance-only continues to work (the legacy posture). CSE
    absence surfaces as a soft note recommending the CSE-primary
    architecture but doesn't block."""

    report = designer_health.probe_designer_readiness(
        behance_api_key="real_behance",
        anthropic_api_key="real_anthropic",
        google_cse_api_key="",
        google_cse_id="",
    )
    assert report.ready is True
    assert report.blockers == ()
    assert any(
        "GOOGLE_CSE_API_KEY" in note for note in report.informational_notes
    )


def test_designer_readiness_passes_with_both_sources_no_notes() -> None:
    """Both sources configured = the richest signal mix. No
    informational notes (no source-mix tradeoffs to surface)."""

    report = designer_health.probe_designer_readiness(
        behance_api_key="real_behance",
        anthropic_api_key="real_anthropic",
        google_cse_api_key="real_cse_key",
        google_cse_id="real_cse_id",
    )
    assert report.ready is True
    assert report.blockers == ()
    assert report.informational_notes == ()


def test_designer_readiness_blocks_when_anthropic_key_missing() -> None:
    """No ANTHROPIC_API_KEY → config blocker pointing at vision
    evaluator. Same posture across all source mixes."""

    report = designer_health.probe_designer_readiness(
        behance_api_key="real_behance",
        anthropic_api_key="",
        google_cse_api_key="real_cse",
        google_cse_id="real_cse_id",
    )
    assert report.ready is False
    assert len(report.blockers) == 1
    blocker = report.blockers[0]
    assert blocker.kind == "config"
    assert "Anthropic" in blocker.message
    assert "ANTHROPIC_API_KEY" in blocker.remediation


def test_designer_readiness_reports_both_blockers_when_neither_source_nor_anthropic() -> None:
    """When the recruiter has nothing configured, all blockers
    surface together so the remediation is one-shot."""

    report = designer_health.probe_designer_readiness(
        behance_api_key="",
        anthropic_api_key="",
        google_cse_api_key="",
        google_cse_id="",
    )
    assert report.ready is False
    assert len(report.blockers) == 2
    messages = {b.message for b in report.blockers}
    assert any("acquisition source" in m for m in messages)
    assert any("Anthropic" in m for m in messages)


def test_designer_readiness_treats_partial_cse_config_as_unconfigured() -> None:
    """A CSE API key without a CSE ID (or vice versa) doesn't count
    as "CSE configured" — both env vars are required for the client
    to instantiate."""

    report_no_id = designer_health.probe_designer_readiness(
        behance_api_key="",
        anthropic_api_key="real_anthropic",
        google_cse_api_key="real_cse_key",
        google_cse_id="",
    )
    assert report_no_id.ready is False
    assert any(
        "acquisition source" in b.message for b in report_no_id.blockers
    )

    report_no_key = designer_health.probe_designer_readiness(
        behance_api_key="",
        anthropic_api_key="real_anthropic",
        google_cse_api_key="",
        google_cse_id="real_cse_id",
    )
    assert report_no_key.ready is False


def test_designer_readiness_falls_back_to_config_and_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When kwargs default to None, the probe reads shared.config + os.environ."""

    monkeypatch.setattr(designer_health.config, "ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("BEHANCE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)

    report = designer_health.probe_designer_readiness()

    assert report.ready is False
    messages = {b.message for b in report.blockers}
    # Both blockers fire: no acquisition source AND no Anthropic.
    assert any("acquisition source" in m for m in messages)
    assert any("Anthropic" in m for m in messages)


def test_designer_readiness_returns_readiness_report_shape() -> None:
    """Public contract: probe returns the dataclass shape the aggregator unions over."""

    report = designer_health.probe_designer_readiness(
        behance_api_key="real_behance",
        anthropic_api_key="real_anthropic",
        google_cse_api_key="real_cse",
        google_cse_id="real_cse_id",
    )

    assert isinstance(report, designer_health.ReadinessReport)
    assert isinstance(report.blockers, tuple)
    assert isinstance(report.informational_notes, tuple)
    for blocker in report.blockers:
        assert isinstance(blocker, designer_health.ReadinessBlocker)
