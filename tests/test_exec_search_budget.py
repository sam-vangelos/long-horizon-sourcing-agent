"""Tests for Executive Search Slice 5 — per-search dossier-spend tracker.

Pins the contract:

- :class:`DossierSpendTracker.reserve` returns
  :class:`BudgetReservation` while accumulated cost stays under the
  cap, and :class:`BudgetExhausted` once a reservation would push
  the total over the cap.
- The tracker tracks BOTH cost and per-source breakdown so the
  ``BudgetExhausted`` event can render a recruiter-facing summary.
- The soft eval-count alarm fires exactly once when
  ``accumulated_evals`` first reaches the threshold.
- :func:`predicted_cost_for` returns deterministic numbers for the
  five known sources and ``0.0`` for unknowns.
- The Crunchbase / PitchBook adapters are registered after Slice 5.
- :class:`CrunchbaseSignalSource` / :class:`PitchBookSignalSource`
  return :class:`SignalFailure(reason="upstream_5xx")` when the
  client raises a 5xx error, exercising the per-source circuit-
  breaker without aborting the dossier eval.
- The brief schema accepts ``dossier_spend_cap_usd`` and
  ``company_stage_signals`` and the loader hydrates them onto the
  structured ``Brief``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from exec_search.budget import (
    DEFAULT_DOSSIER_SPEND_CAP_USD,
    SOFT_EVAL_COUNT_ALARM_THRESHOLD,
    BudgetExhausted,
    BudgetReservation,
    DossierSpendTracker,
    predicted_cost_for,
)
from exec_search.signals import (
    SIGNAL_REGISTRY,
    SignalFailure,
    SignalRequestContext,
    SignalResult,
    known_signal_sources,
)
from exec_search.signals.crunchbase import (
    CompanyStageSignal,
    CrunchbaseApiError,
    CrunchbaseClient,
    CrunchbaseSignalSource,
)
from exec_search.signals.pitchbook import (
    PitchBookApiError,
    PitchBookClient,
    PitchBookCompanySignal,
    PitchBookSignalSource,
)
from shared.brief_loader import load_brief
from shared.brief_schema import (
    Brief,
    CapabilityArea,
    DepthDistinction,
    FacialCalibration,
    MarketDensity,
)
from shared.schemas import CandidateProfileSummary, Experience


# ---------------------------------------------------------------------------
# DossierSpendTracker
# ---------------------------------------------------------------------------


def test_default_cap_constant() -> None:
    assert DEFAULT_DOSSIER_SPEND_CAP_USD == pytest.approx(200.0)


def test_reserve_under_cap_returns_reservation() -> None:
    tracker = DossierSpendTracker(cap_usd=10.0)
    out = tracker.reserve(source="perplexity", cost_usd=0.02)
    assert isinstance(out, BudgetReservation)
    assert out.accumulated_usd == pytest.approx(0.02)
    assert tracker.remaining_usd == pytest.approx(9.98)


def test_reserve_over_cap_returns_exhausted_without_charging() -> None:
    tracker = DossierSpendTracker(cap_usd=1.0)
    tracker.reserve(source="opus_full_eval", cost_usd=0.9)
    out = tracker.reserve(source="opus_full_eval", cost_usd=0.5)
    assert isinstance(out, BudgetExhausted)
    # Accumulated unchanged after rejection.
    assert tracker.accumulated_usd == pytest.approx(0.9)
    assert out.cap_usd == pytest.approx(1.0)
    assert out.accumulated_usd == pytest.approx(0.9)
    assert out.requested_cost_usd == pytest.approx(0.5)


def test_by_source_breakdown_tracks_each_source_separately() -> None:
    tracker = DossierSpendTracker(cap_usd=100.0)
    tracker.reserve(source="perplexity", cost_usd=0.05)
    tracker.reserve(source="news", cost_usd=0.01)
    tracker.reserve(source="perplexity", cost_usd=0.05)
    breakdown = tracker.by_source
    assert breakdown["perplexity"] == pytest.approx(0.10)
    assert breakdown["news"] == pytest.approx(0.01)


def test_full_eval_increments_eval_counter() -> None:
    tracker = DossierSpendTracker(cap_usd=1000.0)
    tracker.reserve(source="opus_full_eval", cost_usd=1.0, is_full_eval=True)
    tracker.reserve(source="opus_full_eval", cost_usd=1.0, is_full_eval=True)
    tracker.reserve(source="perplexity", cost_usd=0.02)
    assert tracker.accumulated_evals == 2


def test_soft_alarm_fires_exactly_once_at_threshold() -> None:
    tracker = DossierSpendTracker(
        cap_usd=10000.0, soft_eval_count_threshold=3
    )
    a = tracker.reserve(source="opus_full_eval", cost_usd=1.0, is_full_eval=True)
    b = tracker.reserve(source="opus_full_eval", cost_usd=1.0, is_full_eval=True)
    c = tracker.reserve(source="opus_full_eval", cost_usd=1.0, is_full_eval=True)
    d = tracker.reserve(source="opus_full_eval", cost_usd=1.0, is_full_eval=True)
    assert isinstance(a, BudgetReservation) and not a.soft_alarm_fired
    assert isinstance(b, BudgetReservation) and not b.soft_alarm_fired
    assert isinstance(c, BudgetReservation) and c.soft_alarm_fired
    # Subsequent eval does NOT re-fire the alarm.
    assert isinstance(d, BudgetReservation) and not d.soft_alarm_fired


def test_adjust_corrects_cost_after_the_fact() -> None:
    tracker = DossierSpendTracker(cap_usd=10.0)
    tracker.reserve(source="opus_full_eval", cost_usd=1.5)
    tracker.adjust(source="opus_full_eval", delta_usd=-0.3)
    assert tracker.accumulated_usd == pytest.approx(1.2)
    assert tracker.by_source["opus_full_eval"] == pytest.approx(1.2)


def test_predicted_cost_for_known_sources() -> None:
    assert predicted_cost_for("perplexity") > 0
    assert predicted_cost_for("news") > 0
    assert predicted_cost_for("crunchbase") > 0
    assert predicted_cost_for("pitchbook") > 0
    assert predicted_cost_for("opus_full_eval") > 0


def test_predicted_cost_for_unknown_source_is_zero() -> None:
    assert predicted_cost_for("unknown") == 0.0


# ---------------------------------------------------------------------------
# Registry — Crunchbase + PitchBook landed
# ---------------------------------------------------------------------------


def test_registry_includes_crunchbase_and_pitchbook_after_slice_5() -> None:
    sources = known_signal_sources()
    assert "crunchbase" in sources
    assert "pitchbook" in sources


# ---------------------------------------------------------------------------
# Crunchbase adapter
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
                name="x",
                description="y",
                builder_signals=["z"],
                user_signals=[],
            )
        ],
        depth_distinction=DepthDistinction(
            builder_definition="a", user_definition="b", edge_case_guidance="c",
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
                summary_bullets=[],
            ),
        ],
    )


def test_crunchbase_adapter_disabled_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRUNCHBASE_API_KEY", raising=False)
    source = CrunchbaseSignalSource()
    out = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(out, SignalFailure)
    assert out.reason == "disabled_no_api_key"


def test_crunchbase_adapter_renders_signal_into_section() -> None:
    client = CrunchbaseClient(api_key="stub")
    client.fetch_company = MagicMock(  # type: ignore[method-assign]
        return_value=CompanyStageSignal(
            company_name="AcmeCorp",
            stage="active",
            last_funding_round="series_d",
            last_funding_amount_usd=80_000_000.0,
            last_funding_at="2024-09-01",
            headcount_estimate="500-1000",
            operating_status="active",
        )
    )
    source = CrunchbaseSignalSource(client=client)
    out = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(out, SignalResult)
    assert out.source == "crunchbase"
    assert "AcmeCorp" in out.section_text
    assert "series_d" in out.section_text
    assert "80,000,000" in out.section_text


def test_crunchbase_adapter_translates_5xx_into_circuit_breaker() -> None:
    client = CrunchbaseClient(api_key="stub")
    def _explode(company_name: str) -> CompanyStageSignal | None:
        raise CrunchbaseApiError("5xx", status_code=503)
    client.fetch_company = _explode  # type: ignore[method-assign]
    source = CrunchbaseSignalSource(client=client)
    out = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(out, SignalFailure)
    assert out.reason == "upstream_5xx"


def test_crunchbase_adapter_no_match_renders_placeholder() -> None:
    client = CrunchbaseClient(api_key="stub")
    client.fetch_company = MagicMock(return_value=None)  # type: ignore[method-assign]
    source = CrunchbaseSignalSource(client=client)
    out = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(out, SignalResult)
    assert "no Crunchbase match" in out.section_text


# ---------------------------------------------------------------------------
# PitchBook adapter
# ---------------------------------------------------------------------------


def test_pitchbook_adapter_disabled_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PITCHBOOK_API_KEY", raising=False)
    source = PitchBookSignalSource()
    out = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(out, SignalFailure)
    assert out.reason == "disabled_no_api_key"


def test_pitchbook_adapter_renders_pe_backed_section() -> None:
    client = PitchBookClient(api_key="stub")
    client.fetch_company = MagicMock(  # type: ignore[method-assign]
        return_value=PitchBookCompanySignal(
            company_name="AcmeCorp",
            deal_type="growth",
            deal_stage="series_d",
            last_deal_at="2024-09-01",
            last_deal_amount_usd=80_000_000.0,
            pe_backed=True,
            investors=("Sequoia", "Andreessen Horowitz"),
        )
    )
    source = PitchBookSignalSource(client=client)
    out = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(out, SignalResult)
    assert "PE-backed: yes" in out.section_text
    assert "Sequoia" in out.section_text


def test_pitchbook_adapter_translates_5xx_into_circuit_breaker() -> None:
    client = PitchBookClient(api_key="stub")
    def _explode(company_name: str) -> PitchBookCompanySignal | None:
        raise PitchBookApiError("5xx", status_code=502)
    client.fetch_company = _explode  # type: ignore[method-assign]
    source = PitchBookSignalSource(client=client)
    out = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(out, SignalFailure)
    assert out.reason == "upstream_5xx"


# ---------------------------------------------------------------------------
# Schema additions hydrated through loader
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(payload))
    return path


def _exec_brief_payload() -> dict:
    return {
        "role_title": "VP Engineering",
        "role_summary": "y",
        "geography": "United States",
        "linkedin_project": "exec",
        "capability_areas": [
            {
                "name": "x",
                "description": "y",
                "builder_signals": ["z"],
                "user_signals": [],
            }
        ],
        "depth_distinction": {
            "builder_definition": "a",
            "user_definition": "b",
            "edge_case_guidance": "c",
        },
        "target_modules": ["linkedin", "exec_search"],
    }


def test_loader_hydrates_default_dossier_spend_cap(tmp_path: Path) -> None:
    brief = load_brief(_write(tmp_path, _exec_brief_payload()))
    assert brief._new_brief.dossier_spend_cap_usd == pytest.approx(200.0)
    assert brief._new_brief.company_stage_signals == {}


def test_loader_hydrates_recruiter_overridden_cap(tmp_path: Path) -> None:
    payload = _exec_brief_payload()
    payload["dossier_spend_cap_usd"] = 350.5
    payload["company_stage_signals"] = {"target_stage": "series_d"}
    brief = load_brief(_write(tmp_path, payload))
    assert brief._new_brief.dossier_spend_cap_usd == pytest.approx(350.5)
    assert brief._new_brief.company_stage_signals == {"target_stage": "series_d"}


def test_loader_falls_back_to_default_on_malformed_cap(tmp_path: Path) -> None:
    payload = _exec_brief_payload()
    payload["dossier_spend_cap_usd"] = "not a number"
    payload["company_stage_signals"] = ["not a dict"]
    brief = load_brief(_write(tmp_path, payload))
    assert brief._new_brief.dossier_spend_cap_usd == pytest.approx(200.0)
    assert brief._new_brief.company_stage_signals == {}
