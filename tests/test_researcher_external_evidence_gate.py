"""Tests for the researcher-side external-evidence trigger gate
(audit Move #23).

Asserts:

- ``orcid_missing`` fires when the candidate has no ORCID anchor.
- ``thin_publication_record`` fires when works_count is small and
  ORCID is present.
- ``recent_burst`` fires on a high papers_in_window relative to
  works_count, ORCID present, larger publication record.
- ``no_trigger_matched`` is the steady-state — established
  researchers with ORCID + sufficient publication history don't
  invoke the cross-check.
- The gate signals dict carries the raw inputs so downstream
  telemetry can replay decisions deterministically.
- Researcher pipeline emits ``external_evidence_*`` log events at the
  full-judge call site (per Move #23 wiring) so cross-module trace
  consumers see the cross-check decision even before a real arXiv +
  news provider lands.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from researcher.orchestrator import ResearcherPipeline
from researcher.schemas import ResearcherCandidate, ResearcherPaper
from shared.external_evidence import (
    should_request_external_evidence_for_researcher,
)
from shared.runtime_state.researcher import ResearcherRuntimeStateBridge
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Pure-gate behavior
# ---------------------------------------------------------------------------


def _candidate(
    *,
    orcid: str = "",
    works_count: int = 0,
    papers_in_window: int = 0,
    h_index: int = 0,
) -> ResearcherCandidate:
    return ResearcherCandidate(
        author_id="A1",
        orcid=orcid,
        name="Test",
        works_count=works_count,
        papers_in_window=papers_in_window,
        h_index=h_index,
    )


def test_gate_fires_orcid_missing_when_no_anchor() -> None:
    decision = should_request_external_evidence_for_researcher(
        candidate=_candidate(orcid="", works_count=20, h_index=12)
    )
    assert decision.should_run is True
    assert decision.reason == "orcid_missing"
    assert decision.signals["fired"] == "orcid_missing"
    assert decision.signals["has_orcid"] is False


def test_gate_fires_thin_publication_record() -> None:
    """ORCID present, works_count < 5 → thin record cross-check."""

    decision = should_request_external_evidence_for_researcher(
        candidate=_candidate(
            orcid="0000-0001-1111-1111", works_count=3, papers_in_window=2
        )
    )
    assert decision.should_run is True
    assert decision.reason == "thin_publication_record"
    assert decision.signals["fired"] == "thin_publication_record"


def test_gate_fires_recent_burst_when_papers_in_window_dense() -> None:
    """ORCID present, larger record (works_count >= 5), but
    papers_in_window is dense relative to total works → recent
    activity spike worth cross-checking."""

    decision = should_request_external_evidence_for_researcher(
        candidate=_candidate(
            orcid="0000-0001-1111-1111",
            works_count=10,
            papers_in_window=8,  # 8 of 10 in the recent window
            h_index=5,
        )
    )
    assert decision.should_run is True
    assert decision.reason == "recent_burst"


def test_gate_does_not_fire_for_established_researcher() -> None:
    """ORCID present, plenty of works, papers_in_window proportional —
    no trigger. The most common steady-state."""

    decision = should_request_external_evidence_for_researcher(
        candidate=_candidate(
            orcid="0000-0001-1111-1111",
            works_count=50,
            papers_in_window=6,
            h_index=20,
        )
    )
    assert decision.should_run is False
    assert decision.skip_reason == "no_trigger_matched"
    assert decision.signals["fired"] == "none"


def test_gate_signals_carry_raw_inputs() -> None:
    """Telemetry-replay contract: the signals dict is enough to
    reproduce the decision."""

    decision = should_request_external_evidence_for_researcher(
        candidate=_candidate(
            orcid="0000-0001-1111-1111",
            works_count=10,
            papers_in_window=8,
            h_index=5,
        )
    )
    assert decision.signals["has_orcid"] is True
    assert decision.signals["works_count"] == 10
    assert decision.signals["papers_in_window"] == 8
    assert decision.signals["h_index"] == 5


# ---------------------------------------------------------------------------
# Pipeline-side wiring: gate fires telemetry at the full-judge call site
# ---------------------------------------------------------------------------


class _StubOpenAlexClient:
    def __init__(self, responses_by_concept: dict[str, list[dict]]) -> None:
        self.responses = responses_by_concept

    def search_authors(self, **kwargs: Any) -> dict:
        concepts = kwargs.get("concept_ids") or []
        key = ",".join(concepts) if concepts else ""
        return {"meta": {"next_cursor": ""}, "results": self.responses.get(key, [])}


def _author_payload(*, author_id: str, name: str, orcid: str = "") -> dict:
    """Author payload that survives both the disambiguation floor
    (papers_in_window_floor=3) AND the facial-judge h_index fast-exit
    (h_index_floor=8 for ml_general). h_index is bumped above the
    floor; works_count stays low so the orcid_missing /
    thin_publication_record gates can be the load-bearing trigger
    when ORCID presence varies."""

    return {
        "id": f"https://openalex.org/{author_id}",
        "display_name": name,
        "orcid": orcid,
        "summary_stats": {"h_index": 10},
        "cited_by_count": 200,
        "works_count": 3,
        "counts_by_year": [
            {"year": 2025, "works_count": 2},
            {"year": 2024, "works_count": 2},
        ],
        "last_known_institutions": [{"display_name": "MIT", "country_code": "US"}],
        "x_concepts": [{"id": "https://openalex.org/C1"}],
    }


def _stub_brief() -> SimpleNamespace:
    return SimpleNamespace(
        id="researcher-ext-evidence-test",
        role_title="Researcher",
        capability_areas=[
            SimpleNamespace(name="Post-training", description="RLHF/DPO/SFT.")
        ],
        depth_distinction=SimpleNamespace(
            builder_definition="First-author publications.",
            user_definition="Cites without publishing.",
            edge_case_guidance="Borderline = full eval.",
        ),
        _new_brief={"source_config": {"researcher": {"discipline": "ml_general"}}},
    )


def _strategy_response() -> dict:
    return {
        "generated_strings": [
            {
                "id": 1,
                "name": "q1",
                "topic_concepts": ["C1"],
                "ror_country_filter": ["US"],
            },
        ],
    }


def _facial_yes(_s: str, _u: str) -> dict:
    return {
        "decision": "FACIAL_YES",
        "rationale": "Cross-thesis hit",
        "confidence": 0.9,
    }


def _full_save(_s: str, _u: str) -> dict:
    return {
        "decision": "SAVE",
        "path": "first_author",
        "confidence": 0.92,
        "rationale": "Strong first-author work.",
    }


def _read_events(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out: list[dict] = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def test_pipeline_emits_external_evidence_unavailable_when_orcid_missing(
    tmp_path: Path,
) -> None:
    """A candidate without ORCID: orcid_missing trigger fires; pipeline
    emits ``external_evidence_unavailable`` (Reopen P7.5(c) — no
    provider is wired yet, so the gate-fires telemetry name must not
    claim a fetch that never happened; ``external_evidence_fetched`` is
    reserved for LinkedIn's real-fetch path)."""

    state_dir = tmp_path / "researcher" / "key"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    brief = _stub_brief()
    bridge = ResearcherRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief.id,
        brief_name=brief.role_title,
    )
    run_id = bridge.start_or_resume_run(resume=False)

    # ORCID absent ⇒ orcid_missing fires.
    client = _StubOpenAlexClient(
        {"C1": [_author_payload(author_id="A1", name="No-ORCID Author", orcid="")]}
    )
    pipeline = ResearcherPipeline(
        brief=brief,
        bridge=bridge,
        openalex_client=client,
        facial_llm_caller=_facial_yes,
        full_llm_caller=_full_save,
        strategy_llm_caller=lambda _s, _u: _strategy_response(),
    )
    pipeline.run(run_id=run_id)

    events = _read_events(state_dir / "run_log.jsonl")
    unavailable = [
        e for e in events if e["event"] == "external_evidence_unavailable"
    ]
    assert unavailable, (
        "expected external_evidence_unavailable event when orcid is missing"
    )
    assert unavailable[0]["reason"] == "orcid_missing"
    assert unavailable[0]["fired"] == "orcid_missing"
    assert unavailable[0]["has_orcid"] is False
    fetched = [e for e in events if e["event"] == "external_evidence_fetched"]
    assert not fetched, (
        "researcher must never claim external_evidence_fetched — no "
        "provider is wired yet (that event is LinkedIn's real-fetch path)"
    )


def test_pipeline_emits_external_evidence_skipped_for_established_researcher(
    tmp_path: Path,
) -> None:
    """A high-h-index, ORCID-anchored researcher with a long publication
    record should NOT trigger the cross-check; the pipeline emits
    ``external_evidence_skipped``."""

    state_dir = tmp_path / "researcher" / "key"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    brief = _stub_brief()
    bridge = ResearcherRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief.id,
        brief_name=brief.role_title,
    )
    run_id = bridge.start_or_resume_run(resume=False)

    payload = _author_payload(
        author_id="A1", name="Established", orcid="0000-0001-1111-1111"
    )
    payload["works_count"] = 50  # plenty of works → no thin/burst trigger
    payload["counts_by_year"] = [
        {"year": 2024, "works_count": 3},
        {"year": 2023, "works_count": 3},
        {"year": 2022, "works_count": 2},
    ]
    payload["summary_stats"] = {"h_index": 25}

    client = _StubOpenAlexClient({"C1": [payload]})
    pipeline = ResearcherPipeline(
        brief=brief,
        bridge=bridge,
        openalex_client=client,
        facial_llm_caller=_facial_yes,
        full_llm_caller=_full_save,
        strategy_llm_caller=lambda _s, _u: _strategy_response(),
    )
    pipeline.run(run_id=run_id)

    events = _read_events(state_dir / "run_log.jsonl")
    skipped = [e for e in events if e["event"] == "external_evidence_skipped"]
    assert skipped, "expected external_evidence_skipped for established researcher"
    assert skipped[0]["skip_reason"] == "no_trigger_matched"
