"""Tests for Executive Search Slice 3 — off-LinkedIn signal interface.

Pins the contract:

- :data:`SIGNAL_REGISTRY` carries the registered signal sources.
- :class:`PerplexitySignalSource` wraps
  :func:`shared.external_evidence.provider.fetch_external_candidate_evidence`
  and translates the result into a :class:`SignalResult` /
  :class:`SignalFailure`.
- :func:`assemble_dossier_evidence` composes the LinkedIn profile +
  per-source signal sections into one prompt body string.
- Adapter exceptions degrade to :class:`SignalFailure`, never raise
  out to callers.
- Failed signals render a "section unavailable" placeholder rather
  than disappearing silently.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from exec_search.evidence_assembly import (
    DossierEvidence,
    assemble_dossier_evidence,
    fetch_signals,
)
from exec_search.signals import (
    SIGNAL_REGISTRY,
    SignalFailure,
    SignalRequestContext,
    SignalResult,
    get_signal_source,
    known_signal_sources,
)
from exec_search.signals.perplexity import PerplexitySignalSource
from shared.brief_schema import (
    Brief,
    CapabilityArea,
    DepthDistinction,
    FacialCalibration,
    MarketDensity,
)
from shared.schemas import (
    CandidateProfileSummary,
    Education,
    Experience,
    EvidenceRef,
    ExternalCandidateEvidence,
    ExternalEvidenceFailure,
    ExternalFactBlock,
    ExternalInference,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _exec_brief() -> Brief:
    return Brief(
        role_title="VP Engineering",
        role_level="Executive",
        role_summary="Owns engineering leadership.",
        geography="United States",
        linkedin_project="exec",
        capability_areas=[
            CapabilityArea(
                name="Engineering org leadership",
                description="Builds and runs 50+ person orgs.",
                builder_signals=["VP-level scope"],
                user_signals=["IC-level work"],
            )
        ],
        depth_distinction=DepthDistinction(
            builder_definition="Owns strategy + delivery.",
            user_definition="Manages individual teams.",
            edge_case_guidance="Borderline = full eval.",
        ),
        non_fit_patterns=[],
        employer_signal_rules=[],
        minimum_years_experience=12,
        minimum_bar_description="12+ years.",
        facial_calibration=FacialCalibration(
            expected_yes_rate_low=0.3,
            expected_yes_rate_high=0.5,
            fast_exit_patterns=[],
            trajectory_yes_patterns=[],
            trajectory_ambiguous_patterns=[],
            trajectory_no_patterns=[],
        ),
        market_density=MarketDensity.MODERATE,
        target_modules=["linkedin", "exec_search"],
    )


def _candidate() -> CandidateProfileSummary:
    return CandidateProfileSummary(
        name="Jane Doe",
        headline="VP Engineering at AcmeCorp",
        profile_url="https://linkedin.com/in/jane",
        experiences=[
            Experience(
                title="VP Engineering",
                company="AcmeCorp",
                start="2020",
                end="present",
                summary_bullets=["Owned 400-person engineering org."],
            ),
        ],
        education=[
            Education(
                degree="BS",
                field="Computer Science",
                school="MIT",
                start="2005",
                end="2009",
            ),
        ],
        skills_snippet=["leadership", "scaling"],
    )


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


def test_signal_registry_includes_perplexity_baseline() -> None:
    assert "perplexity" in SIGNAL_REGISTRY
    assert "perplexity" in known_signal_sources()


def test_known_signal_sources_returns_stable_ascending_order() -> None:
    sources = known_signal_sources()
    assert sources == tuple(sorted(sources))


def test_get_signal_source_raises_for_unknown() -> None:
    with pytest.raises(KeyError):
        get_signal_source("not_a_signal_source")


def test_perplexity_signal_source_implements_protocol() -> None:
    source = get_signal_source("perplexity")
    assert source.name == "perplexity"
    assert hasattr(source, "fetch")


# ---------------------------------------------------------------------------
# Perplexity adapter
# ---------------------------------------------------------------------------


def test_perplexity_adapter_translates_failure_into_signal_failure() -> None:
    """Provider returns ExternalEvidenceFailure → adapter returns SignalFailure."""

    failure = ExternalEvidenceFailure(
        reason="disabled_by_config",
        detail="LINKEDIN_EXTERNAL_EVIDENCE_ENABLED is false",
        provider="perplexity",
    )
    with patch(
        "exec_search.signals.perplexity.fetch_external_candidate_evidence",
        return_value=failure,
    ):
        source = PerplexitySignalSource()
        result = source.fetch(
            candidate=_candidate(),
            brief=_exec_brief(),
            context=SignalRequestContext(brief_id="b1"),
        )
    assert isinstance(result, SignalFailure)
    assert result.source == "perplexity"
    assert result.reason == "disabled_by_config"


def test_perplexity_adapter_renders_evidence_into_recruiter_section() -> None:
    """Provider returns ExternalCandidateEvidence → adapter returns SignalResult
    with section_text containing topic blocks + citations."""

    evidence = ExternalCandidateEvidence(
        trigger_reason="dossier_full_eval",
        identity_confidence=0.9,
        external_fact_blocks=[
            ExternalFactBlock(
                topic="Career trajectory",
                facts=[
                    "Joined AcmeCorp as SVP in 2020.",
                    "Promoted to VP Engineering in 2022.",
                ],
                evidence_refs=[
                    EvidenceRef(url="https://acme.com/about", title="About"),
                ],
            ),
            ExternalFactBlock(
                topic="Board memberships",
                facts=["Board seat at BetaInc since 2021."],
                evidence_refs=[
                    EvidenceRef(url="https://beta.com/board"),
                ],
            ),
        ],
        external_inferences=[
            ExternalInference(
                claim="Likely on the diligence team for the BetaInc spin-out.",
                confidence=0.7,
            ),
        ],
    )
    with patch(
        "exec_search.signals.perplexity.fetch_external_candidate_evidence",
        return_value=evidence,
    ):
        source = PerplexitySignalSource()
        result = source.fetch(
            candidate=_candidate(),
            brief=_exec_brief(),
            context=SignalRequestContext(brief_id="b1"),
        )
    assert isinstance(result, SignalResult)
    assert result.source == "perplexity"
    assert "Career trajectory" in result.section_text
    assert "Board memberships" in result.section_text
    assert "Model-derived inferences" in result.section_text
    assert "https://acme.com/about" in result.section_text
    assert "https://beta.com/board" in result.section_text
    assert "https://acme.com/about" in result.citations


def test_perplexity_adapter_renders_empty_evidence_with_placeholder() -> None:
    evidence = ExternalCandidateEvidence(
        trigger_reason="dossier_full_eval",
        identity_confidence=0.0,
        external_fact_blocks=[],
        external_inferences=[],
    )
    with patch(
        "exec_search.signals.perplexity.fetch_external_candidate_evidence",
        return_value=evidence,
    ):
        source = PerplexitySignalSource()
        result = source.fetch(
            candidate=_candidate(),
            brief=_exec_brief(),
            context=SignalRequestContext(brief_id="b1"),
        )
    assert isinstance(result, SignalResult)
    assert "[no off-LinkedIn signal returned by Perplexity]" in result.section_text


# ---------------------------------------------------------------------------
# Evidence assembly
# ---------------------------------------------------------------------------


def test_assemble_dossier_evidence_includes_profile_and_signals() -> None:
    """Assembled body has profile section + all requested signal sections."""

    evidence = ExternalCandidateEvidence(
        trigger_reason="dossier_full_eval",
        identity_confidence=0.9,
        external_fact_blocks=[
            ExternalFactBlock(
                topic="Career trajectory",
                facts=["Joined AcmeCorp as SVP in 2020."],
                evidence_refs=[],
            ),
        ],
    )
    with patch(
        "exec_search.signals.perplexity.fetch_external_candidate_evidence",
        return_value=evidence,
    ):
        out = assemble_dossier_evidence(
            candidate=_candidate(),
            brief=_exec_brief(),
            context=SignalRequestContext(brief_id="b1"),
        )

    assert isinstance(out, DossierEvidence)
    # Profile section.
    assert "## Candidate profile" in out.prompt_body
    assert "Jane Doe" in out.prompt_body
    assert "VP Engineering at AcmeCorp" in out.prompt_body
    # Perplexity section.
    assert "## Off-LinkedIn signal: perplexity" in out.prompt_body
    assert "Career trajectory" in out.prompt_body
    # Telemetry — outcome dict carries the SignalResult.
    assert isinstance(out.signal_outcomes["perplexity"], SignalResult)


def test_assemble_dossier_evidence_renders_failure_as_unavailable_section() -> None:
    failure = ExternalEvidenceFailure(
        reason="quota_exhausted",
        detail="Perplexity quota exceeded",
        provider="perplexity",
    )
    with patch(
        "exec_search.signals.perplexity.fetch_external_candidate_evidence",
        return_value=failure,
    ):
        out = assemble_dossier_evidence(
            candidate=_candidate(),
            brief=_exec_brief(),
            context=SignalRequestContext(brief_id="b1"),
        )
    assert "## Off-LinkedIn signal: perplexity" in out.prompt_body
    assert "Signal unavailable" in out.prompt_body
    assert "quota_exhausted" in out.prompt_body
    assert isinstance(out.signal_outcomes["perplexity"], SignalFailure)


def test_assemble_dossier_evidence_handles_adapter_exception() -> None:
    """If an adapter raises despite the contract, the assembler catches
    it and degrades to a single-source SignalFailure rather than aborting
    the whole dossier."""

    def explode(**kwargs: Any) -> Any:
        raise RuntimeError("boom")

    with patch(
        "exec_search.signals.perplexity.fetch_external_candidate_evidence",
        side_effect=explode,
    ):
        out = assemble_dossier_evidence(
            candidate=_candidate(),
            brief=_exec_brief(),
            context=SignalRequestContext(brief_id="b1"),
        )
    failure = out.signal_outcomes["perplexity"]
    assert isinstance(failure, SignalFailure)
    assert failure.reason == "adapter_exception"
    assert "RuntimeError" in failure.detail


def test_fetch_signals_unknown_source_returns_failure() -> None:
    outcomes = fetch_signals(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
        sources=["does_not_exist"],
    )
    assert isinstance(outcomes["does_not_exist"], SignalFailure)
    assert outcomes["does_not_exist"].reason == "unknown_source"


def test_fetch_signals_respects_caller_provided_source_subset() -> None:
    """When ``sources=[]``, no adapters fire; outcomes is empty."""

    outcomes = fetch_signals(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
        sources=[],
    )
    assert outcomes == {}
