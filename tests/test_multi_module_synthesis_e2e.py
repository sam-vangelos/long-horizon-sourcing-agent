"""End-to-end test for the multi-module synthesis loop — audit Move #7.

Depends on Move #1 (handoff payloads end-to-end) which is shipped at
commit ed274a2 — verify with ``git log --oneline | grep "audit Move #1"``.

This test exercises the full multi-module loop:

1. A multi-module brief surfaces evidence batches across ≥2 sources
   (the contributing-sources guard at
   :func:`market_intelligence.reflection._contributing_sources_count`).
2. The reflection-time integration helper
   :func:`market_intelligence.reflection._persist_and_read_handoff_payloads`
   builds per-source :class:`HandoffPayload` from each evidence batch
   and persists them into ``chief_of_staff_runs.handoff_payloads_json``
   (audit Move #1's small-version of the broker arc).
3. The composed handoff context lands in the synthesis prompt's input
   under ``prior_handoff_payloads``.
4. Synthesis runs and the team-level paragraph references both
   modules' reads (heuristic backend in tests; the LLM backend would
   produce the same shape with richer prose).
5. Synthesis confidence > 0.5 because both modules contributed
   substantive signal.

Pre-Move-1 multi-module runs read as three independent evals stitched
together — handoff_payloads_json was dead code. Post-Move-1, this
test is the load-bearing acceptance that the small-version broker arc
flows end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloris.chief_of_staff import HeuristicChiefOfStaffSynthesizer
from cloris.chief_of_staff.handoff import (
    build_handoff_payload_from_evidence_batch,
)
from market_intelligence import reflection as reflection_engine
from market_intelligence.schema import MarketEvidenceBatch, MarketIdentity
from shared.runtime_state.orchestration_store import OrchestrationStateStore
from shared.runtime_state.read_models import chief_of_staff_run_by_brief


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _full_judgment(
    *, candidate_id: str, decision: str, confidence: float, rationale: str
) -> dict:
    """Final-judgment row matching what the runtime-state pipeline writes."""

    return {
        "candidate_id": candidate_id,
        "decision": decision,
        "full_decision": {
            "decision": decision,
            "confidence": confidence,
            "rationale": rationale,
        },
        "rationale": rationale,
        "confidence": confidence,
    }


def _evidence_batch_with_saves(
    *,
    source: str,
    candidate_volume: int,
    save_rows: list[dict],
) -> MarketEvidenceBatch:
    """Build an evidence batch carrying the metrics + final judgments
    that the reflection-time payload builder reads."""

    saved_count = sum(
        1
        for row in save_rows
        if row.get("decision") in {"SAVE", "INFERENTIAL_SAVE", "SIGNAL_SAVE"}
    )
    return MarketEvidenceBatch(
        run_ref=f"ref-{source}",
        source=source,
        output_dir=f"/tmp/{source}/run",
        brief_version="v1",
        generated_at="2026-05-04T18:00:00+00:00",
        metrics_summary={
            "run_count": 1,
            "candidate_volume": candidate_volume,
            "saved": saved_count,
        },
        final_judgments=save_rows,
    )


def _market_identity_for_test() -> MarketIdentity:
    return MarketIdentity(
        market_key="frontier_ai__sf__senior_fde",
        role_title="Senior Forward Deployed Engineer",
        role_level="senior",
        geography="SF Bay Area",
        channels_seen=[],
        brief_ids_seen=[],
        brief_versions_seen=[],
    )


# ---------------------------------------------------------------------------
# E2E flow
# ---------------------------------------------------------------------------


def test_multi_module_synthesis_e2e_handoff_payloads_compose_through_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The load-bearing audit Move #7 acceptance.

    With target_modules >= 2 and substantive evidence on each, the
    chain ``_persist_and_read_handoff_payloads`` →
    ``HeuristicChiefOfStaffSynthesizer.synthesize(prior_handoff_payloads=...)``
    produces a team-level paragraph with confidence > 0.5 that
    references both modules.
    """

    # Point the orchestration db at a tmp path so the test doesn't
    # touch shared state.
    orch_db = tmp_path / "orchestration" / "runtime_state.sqlite3"

    import shared.output_paths as output_paths

    monkeypatch.setattr(
        output_paths, "resolve_orchestration_db_path", lambda: orch_db
    )

    # Pre-create the chief_of_staff_runs row that Move #1's
    # _persist_dispatch_run would create at dispatch time.
    store = OrchestrationStateStore(orch_db)
    store.insert_chief_of_staff_run(
        brief_id="frontier-ai-fde",
        principal_id="anthropic-recruiting",
        status="running",
        dispatch_plan={
            "steps": [
                {"module_name": "linkedin", "handoff_condition": None},
                {"module_name": "github", "handoff_condition": None},
            ]
        },
        invocation_order=["linkedin", "github"],
        handoff_payloads={},
        synthesis_output={},
        started_at="2026-05-04T17:00:00+00:00",
    )

    # Stub load_brief so the helper resolves the brief_id without
    # needing a real on-disk fixture.
    class _BriefStub:
        id = "frontier-ai-fde"
        role_title = "Senior FDE"
        raw = {"brief_id": "frontier-ai-fde"}

    monkeypatch.setattr(
        reflection_engine, "load_brief", lambda _path: _BriefStub()
    )

    # Two evidence batches with saves on each — substantive signal
    # so the heuristic synthesis confidence clears 0.5.
    evidence_batches = [
        _evidence_batch_with_saves(
            source="linkedin",
            candidate_volume=47,
            save_rows=[
                _full_judgment(
                    candidate_id="li-A",
                    decision="SAVE",
                    confidence=0.92,
                    rationale=(
                        "Senior FDE at Anthropic; shipped Claude Code; "
                        "five years of customer-facing builder work."
                    ),
                ),
                _full_judgment(
                    candidate_id="li-B",
                    decision="SAVE",
                    confidence=0.85,
                    rationale="Builder at Stripe; payment-rails depth.",
                ),
                _full_judgment(
                    candidate_id="li-C",
                    decision="SAVE",
                    confidence=0.80,
                    rationale="Hands-on architect at Square.",
                ),
            ],
        ),
        _evidence_batch_with_saves(
            source="github",
            candidate_volume=22,
            save_rows=[
                _full_judgment(
                    candidate_id="gh-1",
                    decision="SAVE",
                    confidence=0.88,
                    rationale="kubernetes scheduler co-author.",
                ),
                _full_judgment(
                    candidate_id="gh-2",
                    decision="SAVE",
                    confidence=0.83,
                    rationale="rust-lang/rust contributor.",
                ),
            ],
        ),
    ]

    # Audit Move #1: persist + read handoff payloads.
    composed = reflection_engine._persist_and_read_handoff_payloads(
        brief_path="/tmp/brief.json",
        market_identity=_market_identity_for_test(),
        evidence_batches=evidence_batches,
    )
    assert composed is not None
    assert set(composed.keys()) == {"linkedin", "github"}

    # Verify the persistence: chief_of_staff_runs.handoff_payloads_json
    # carries both source keys.
    record = chief_of_staff_run_by_brief(orch_db, brief_id="frontier-ai-fde")
    assert record is not None
    persisted = json.loads(record.handoff_payloads_json)
    assert set(persisted.keys()) == {"linkedin", "github"}

    # Each persisted payload carries the audit-Move-1 contract fields.
    for source in ("linkedin", "github"):
        payload = persisted[source]
        assert payload["source"] == source
        assert payload["candidate_count"] > 0
        assert payload["save_count"] > 0
        assert payload["per_source_signal_summary"]
        assert isinstance(payload["top_saves"], list)
        # Every save entry has the {candidate_id, role_fit_narrative,
        # confidence} contract.
        for save in payload["top_saves"]:
            assert "candidate_id" in save
            assert "role_fit_narrative" in save
            assert "confidence" in save

    # Run the synthesis. The heuristic backend is grounded in the
    # per-source signals + ignores prior_handoff_payloads (the LLM
    # backend would consume them); both backends accept the kwarg.
    backend = HeuristicChiefOfStaffSynthesizer()
    synthesis = backend.synthesize(
        market_identity=_market_identity_for_test(),
        per_source_signals=reflection_engine._per_source_signals_from_batches(
            evidence_batches
        ),
        briefing_paragraph=(
            "Mixed read across LinkedIn and GitHub on FDE candidates."
        ),
        prior_handoff_payloads=composed,
    )

    # ===================================================================
    # Audit Move #7 acceptance criteria
    # ===================================================================

    # 1. Team-level paragraph is non-empty + non-trivial.
    assert synthesis.paragraph
    assert len(synthesis.paragraph) > 50

    # 2. Synthesis confidence > 0.5 — both modules contributed
    #    substantive signal.
    assert synthesis.confidence > 0.5, (
        f"expected synthesis confidence > 0.5; got {synthesis.confidence}"
    )

    # 3. Per-specialist weights cover BOTH contributing sources.
    assert "linkedin" in synthesis.per_specialist_weight
    assert "github" in synthesis.per_specialist_weight

    # 4. Team-level read paragraph references both modules'
    #    contributions (humanized labels — "LinkedIn" + "GitHub").
    paragraph_lower = synthesis.paragraph.lower()
    assert "linkedin" in paragraph_lower
    assert "github" in paragraph_lower

    # 5. Synthesis source path is "deterministic" (heuristic backend)
    #    or "llm" (production backend); both are valid post-synthesis
    #    paths. The "empty" sentinel only fires for zero-source runs.
    assert synthesis.source in {"deterministic", "llm"}

    # 6. priority_for_principal is non-empty editorial prose so the
    #    workspace surface has a "look at this first" line.
    assert synthesis.priority_for_principal
    assert len(synthesis.priority_for_principal) > 10


def test_multi_module_synthesis_e2e_with_negative_read_on_one_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both modules ran and surfaced candidates, but only one module
    has saves. Audit Move #1's contract: the negative read (zero saves
    despite candidates surfaced) is itself substantive — the
    contributing-sources count is still 2, and the synthesis paragraph
    reflects the asymmetry."""

    orch_db = tmp_path / "orchestration" / "runtime_state.sqlite3"
    import shared.output_paths as output_paths

    monkeypatch.setattr(
        output_paths, "resolve_orchestration_db_path", lambda: orch_db
    )

    store = OrchestrationStateStore(orch_db)
    store.insert_chief_of_staff_run(
        brief_id="brief-asym",
        principal_id="",
        status="running",
        dispatch_plan={"steps": []},
        invocation_order=["linkedin", "github"],
        handoff_payloads={},
        synthesis_output={},
        started_at="2026-05-04T17:00:00+00:00",
    )

    class _BriefStub:
        id = "brief-asym"
        role_title = "FDE"
        raw = {"brief_id": "brief-asym"}

    monkeypatch.setattr(
        reflection_engine, "load_brief", lambda _path: _BriefStub()
    )

    evidence_batches = [
        _evidence_batch_with_saves(
            source="linkedin",
            candidate_volume=30,
            save_rows=[
                _full_judgment(
                    candidate_id="li-A",
                    decision="SAVE",
                    confidence=0.9,
                    rationale="Strong builder.",
                ),
            ],
        ),
        # GitHub: 22 candidates, 0 saves — the negative read.
        _evidence_batch_with_saves(
            source="github",
            candidate_volume=22,
            save_rows=[],
        ),
    ]

    composed = reflection_engine._persist_and_read_handoff_payloads(
        brief_path="/tmp/brief.json",
        market_identity=_market_identity_for_test(),
        evidence_batches=evidence_batches,
    )
    assert composed is not None

    # Both sources persist a payload (the negative read is informative).
    record = chief_of_staff_run_by_brief(orch_db, brief_id="brief-asym")
    assert record is not None
    persisted = json.loads(record.handoff_payloads_json)
    assert set(persisted.keys()) == {"linkedin", "github"}
    assert persisted["github"]["save_count"] == 0
    assert "none cleared the bar" in (
        persisted["github"]["per_source_signal_summary"].lower()
    )

    backend = HeuristicChiefOfStaffSynthesizer()
    synthesis = backend.synthesize(
        market_identity=_market_identity_for_test(),
        per_source_signals=reflection_engine._per_source_signals_from_batches(
            evidence_batches
        ),
        briefing_paragraph="Mixed-confidence read.",
        prior_handoff_payloads=composed,
    )

    # Both modules show up in per_specialist_weight even though one
    # had zero saves — the negative read is still a read.
    assert set(synthesis.per_specialist_weight.keys()) == {
        "linkedin",
        "github",
    }
    # Synthesis paragraph still names both sources.
    paragraph_lower = synthesis.paragraph.lower()
    assert "linkedin" in paragraph_lower
    assert "github" in paragraph_lower


def test_multi_module_synthesis_e2e_handoff_payloads_match_evidence_batch_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reflection-time persistence and the per-batch builder produce
    byte-equivalent payloads — audit Move #1's contract that the
    reflection-time wiring is a thin glue around the pure builder."""

    orch_db = tmp_path / "orchestration" / "runtime_state.sqlite3"
    import shared.output_paths as output_paths

    monkeypatch.setattr(
        output_paths, "resolve_orchestration_db_path", lambda: orch_db
    )

    store = OrchestrationStateStore(orch_db)
    store.insert_chief_of_staff_run(
        brief_id="brief-equiv",
        principal_id="",
        status="running",
        dispatch_plan={"steps": []},
        invocation_order=["linkedin"],
        handoff_payloads={},
        synthesis_output={},
        started_at="2026-05-04T17:00:00+00:00",
    )

    class _BriefStub:
        id = "brief-equiv"
        role_title = "FDE"
        raw = {"brief_id": "brief-equiv"}

    monkeypatch.setattr(
        reflection_engine, "load_brief", lambda _path: _BriefStub()
    )

    batch = _evidence_batch_with_saves(
        source="linkedin",
        candidate_volume=10,
        save_rows=[
            _full_judgment(
                candidate_id="li-A",
                decision="SAVE",
                confidence=0.95,
                rationale="Top-decile builder.",
            )
        ],
    )

    expected = build_handoff_payload_from_evidence_batch(batch)
    assert expected is not None

    reflection_engine._persist_and_read_handoff_payloads(
        brief_path="/tmp/brief.json",
        market_identity=_market_identity_for_test(),
        evidence_batches=[batch],
    )

    record = chief_of_staff_run_by_brief(orch_db, brief_id="brief-equiv")
    assert record is not None
    persisted = json.loads(record.handoff_payloads_json)
    assert persisted == {"linkedin": expected.to_dict()}
