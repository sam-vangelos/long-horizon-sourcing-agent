"""End-to-end test for the chief-of-staff handoff payloads — audit Move #1.

Closes the highest-blast-radius "Thing You're Not Seeing" finding from
the production-readiness audit: handoff_payloads_json was dead code,
multi-module runs read as three independent evals stitched together.

This test pins:

- Per-source :class:`HandoffPayload` is built from a
  :class:`MarketEvidenceBatch` correctly (top_saves projection,
  signal-density confidence, deterministic signal summary).
- The payload round-trips through
  :meth:`OrchestrationStateStore.merge_handoff_payload` and reads
  back via :func:`chief_of_staff_run_by_brief` byte-equivalent.
- The reflection-time integration helper
  :func:`market_intelligence.reflection._persist_and_read_handoff_payloads`
  walks evidence batches, persists each source's payload into the
  latest CoS run row, and returns a composed prior-handoff context
  for the synthesis prompt.
- The synthesis call (heuristic backend) accepts and ignores
  ``prior_handoff_payloads`` cleanly — the LLM backend would consume
  it, the heuristic just ignores it without raising.
- ``compose_handoff_context`` produces the JSON shape the synthesis
  prompt builder consumes, with all required per-source fields.

The full multi-module synthesis e2e (Move #7) builds on this — that
test asserts the synthesis confidence > 0.5 and the team-level
paragraph cites both modules' top_saves narratives end-to-end. This
test pins the substrate Move #7 will exercise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloris.chief_of_staff import (
    HeuristicChiefOfStaffSynthesizer,
)
from cloris.chief_of_staff.handoff import (
    MAX_TOP_SAVES_PER_SOURCE,
    HandoffPayload,
    build_handoff_payload_from_evidence_batch,
    compose_handoff_context,
)
from market_intelligence.schema import MarketEvidenceBatch, MarketIdentity
from shared.runtime_state.orchestration_store import OrchestrationStateStore
from shared.runtime_state.read_models import chief_of_staff_run_by_brief


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _full_judgment(
    *, candidate_id: str, decision: str, confidence: float, rationale: str
) -> dict:
    """One row in MarketEvidenceBatch.final_judgments shape."""

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


def _batch_with_saves(
    *,
    source: str,
    candidate_volume: int,
    save_rows: list[dict],
) -> MarketEvidenceBatch:
    """Build a multi-judgment batch — saves + non-saves so the helper
    has to filter SAVE-class rows."""

    return MarketEvidenceBatch(
        run_ref=f"ref-{source}",
        source=source,
        output_dir="/tmp/fake",
        brief_version="v1",
        generated_at="2026-05-04T00:00:00+00:00",
        metrics_summary={
            "candidate_volume": candidate_volume,
            "saved": sum(
                1
                for row in save_rows
                if row.get("decision") in {"SAVE", "INFERENTIAL_SAVE"}
            ),
        },
        final_judgments=save_rows,
    )


def _market_identity() -> MarketIdentity:
    return MarketIdentity(
        market_key="frontier_ai__sf__senior",
        role_title="Senior FDE",
        role_level="senior",
        geography="SF",
        channels_seen=[],
        brief_ids_seen=[],
        brief_versions_seen=[],
    )


# ---------------------------------------------------------------------------
# build_handoff_payload_from_evidence_batch
# ---------------------------------------------------------------------------


class TestBuildHandoffPayload:
    def test_payload_extracts_top_saves_in_confidence_order(self) -> None:
        batch = _batch_with_saves(
            source="linkedin",
            candidate_volume=10,
            save_rows=[
                _full_judgment(
                    candidate_id="li-A",
                    decision="SAVE",
                    confidence=0.9,
                    rationale="Senior FDE shipped at Anthropic.",
                ),
                _full_judgment(
                    candidate_id="li-B",
                    decision="SAVE",
                    confidence=0.7,
                    rationale="Strong builder at Stripe.",
                ),
                _full_judgment(
                    candidate_id="li-C",
                    decision="REJECT",
                    confidence=0.1,
                    rationale="Off-thesis.",
                ),
                _full_judgment(
                    candidate_id="li-D",
                    decision="SAVE",
                    confidence=0.95,
                    rationale="Top-decile shipping engineer.",
                ),
            ],
        )
        payload = build_handoff_payload_from_evidence_batch(batch)
        assert payload is not None
        assert payload.source == "linkedin"
        assert payload.candidate_count == 10
        assert payload.save_count == 3
        assert len(payload.top_saves) == 3
        confidences = [save["confidence"] for save in payload.top_saves]
        assert confidences == sorted(confidences, reverse=True)
        assert payload.top_saves[0]["candidate_id"] == "li-D"
        assert payload.top_saves[0]["confidence"] == 0.95

    def test_payload_caps_top_saves_at_max(self) -> None:
        batch = _batch_with_saves(
            source="github",
            candidate_volume=20,
            save_rows=[
                _full_judgment(
                    candidate_id=f"gh-{i}",
                    decision="SAVE",
                    confidence=0.9 - i * 0.05,
                    rationale=f"Maintainer evidence {i}.",
                )
                for i in range(MAX_TOP_SAVES_PER_SOURCE + 3)
            ],
        )
        payload = build_handoff_payload_from_evidence_batch(batch)
        assert payload is not None
        assert len(payload.top_saves) == MAX_TOP_SAVES_PER_SOURCE

    def test_payload_returns_none_for_zero_candidates(self) -> None:
        batch = _batch_with_saves(
            source="github", candidate_volume=0, save_rows=[]
        )
        assert build_handoff_payload_from_evidence_batch(batch) is None

    def test_payload_returns_none_for_empty_source(self) -> None:
        batch = _batch_with_saves(
            source="", candidate_volume=5, save_rows=[]
        )
        assert build_handoff_payload_from_evidence_batch(batch) is None

    def test_payload_signal_summary_reads_as_recruiter_prose(self) -> None:
        """Deterministic prose; no engineer vocabulary, no snake_case."""

        batch = _batch_with_saves(
            source="linkedin",
            candidate_volume=47,
            save_rows=[
                _full_judgment(
                    candidate_id="li-A",
                    decision="SAVE",
                    confidence=0.9,
                    rationale="Strong builder.",
                ),
                _full_judgment(
                    candidate_id="li-B",
                    decision="SAVE",
                    confidence=0.85,
                    rationale="Proven shipper.",
                ),
                _full_judgment(
                    candidate_id="li-C",
                    decision="SAVE",
                    confidence=0.8,
                    rationale="Hands-on architect.",
                ),
            ],
        )
        payload = build_handoff_payload_from_evidence_batch(batch)
        assert payload is not None
        assert "LinkedIn" in payload.per_source_signal_summary
        assert "47" in payload.per_source_signal_summary
        assert "_" not in payload.per_source_signal_summary

    def test_payload_signal_summary_handles_zero_saves_negative_read(self) -> None:
        batch = _batch_with_saves(
            source="github", candidate_volume=22, save_rows=[]
        )
        payload = build_handoff_payload_from_evidence_batch(batch)
        assert payload is not None
        assert payload.save_count == 0
        assert "none cleared the bar" in payload.per_source_signal_summary

    def test_payload_confidence_is_signal_density_not_llm_self_rated(
        self,
    ) -> None:
        """A run with candidates + saves + top_saves narratives = 1.0.
        No saves = ~0.33. No candidates = 0.0 (filtered earlier)."""

        rich_batch = _batch_with_saves(
            source="linkedin",
            candidate_volume=10,
            save_rows=[
                _full_judgment(
                    candidate_id="li-A",
                    decision="SAVE",
                    confidence=0.9,
                    rationale="Strong builder.",
                )
            ],
        )
        rich_payload = build_handoff_payload_from_evidence_batch(rich_batch)
        assert rich_payload is not None
        assert rich_payload.confidence >= 0.99  # 1.0 with rounding tolerance

        thin_batch = _batch_with_saves(
            source="github", candidate_volume=22, save_rows=[]
        )
        thin_payload = build_handoff_payload_from_evidence_batch(thin_batch)
        assert thin_payload is not None
        assert 0.3 <= thin_payload.confidence <= 0.34  # ~1/3


# ---------------------------------------------------------------------------
# Round-trip through orchestration store
# ---------------------------------------------------------------------------


class TestHandoffRoundTrip:
    def test_payload_round_trips_through_merge_and_read(
        self, tmp_path: Path
    ) -> None:
        """End-to-end persistence: build payload from a batch, merge
        into chief_of_staff_runs, read back via the read helper.
        Asserts byte-equivalence of all fields."""

        db_path = tmp_path / "orchestration" / "runtime_state.sqlite3"
        store = OrchestrationStateStore(db_path)
        store.insert_chief_of_staff_run(
            brief_id="brief-handoff-rt",
            principal_id="principal-1",
            status="running",
            dispatch_plan={"steps": []},
            invocation_order=["linkedin"],
            handoff_payloads={},
            synthesis_output={},
            started_at="2026-05-04T17:00:00+00:00",
        )

        batch = _batch_with_saves(
            source="linkedin",
            candidate_volume=10,
            save_rows=[
                _full_judgment(
                    candidate_id="li-A",
                    decision="SAVE",
                    confidence=0.92,
                    rationale="FDE at Anthropic; shipped Claude Code.",
                )
            ],
        )
        payload = build_handoff_payload_from_evidence_batch(batch)
        assert payload is not None

        merged = store.merge_handoff_payload(
            brief_id="brief-handoff-rt",
            source=payload.source,
            payload=payload.to_dict(),
        )
        assert merged is True

        record = chief_of_staff_run_by_brief(db_path, brief_id="brief-handoff-rt")
        assert record is not None
        persisted = json.loads(record.handoff_payloads_json)
        assert persisted == {"linkedin": payload.to_dict()}


# ---------------------------------------------------------------------------
# compose_handoff_context — the synthesis-prompt-input shape
# ---------------------------------------------------------------------------


class TestComposeHandoffContext:
    def test_returns_none_for_empty_input(self) -> None:
        assert compose_handoff_context(None) is None
        assert compose_handoff_context({}) is None

    def test_normalizes_persisted_payloads_to_synthesis_input_shape(
        self,
    ) -> None:
        composed = compose_handoff_context(
            {
                "linkedin": {
                    "source": "linkedin",
                    "candidate_count": 47,
                    "save_count": 5,
                    "confidence": 0.78,
                    "per_source_signal_summary": "LinkedIn read...",
                    "top_saves": [
                        {
                            "candidate_id": "li-A",
                            "role_fit_narrative": "Senior FDE.",
                            "confidence": 0.9,
                        }
                    ],
                },
                "github": {
                    "source": "github",
                    "candidate_count": 22,
                    "save_count": 0,
                    "confidence": 0.33,
                    "per_source_signal_summary": "GitHub read...",
                    "top_saves": [],
                },
            }
        )
        assert composed is not None
        assert set(composed.keys()) == {"linkedin", "github"}
        for key, expected_count in (("linkedin", 47), ("github", 22)):
            entry = composed[key]
            assert entry["source"] == key
            assert entry["candidate_count"] == expected_count
            assert "per_source_signal_summary" in entry
            assert isinstance(entry["top_saves"], list)
            assert isinstance(entry["confidence"], float)


# ---------------------------------------------------------------------------
# Reflection-time wiring: _persist_and_read_handoff_payloads
# ---------------------------------------------------------------------------


class TestPersistAndReadHandoffPayloads:
    def test_persists_per_source_payloads_and_returns_composed_context(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the brief has a CoS run row, the helper persists each
        per-source payload and returns the composed context."""

        from market_intelligence import reflection as reflection_engine
        import shared.output_paths as output_paths

        # Point orchestration db_path at tmp.
        orch_db = tmp_path / "orchestration" / "runtime_state.sqlite3"

        def _stub_orchestration_db_path() -> Path:
            return orch_db

        monkeypatch.setattr(
            output_paths,
            "resolve_orchestration_db_path",
            _stub_orchestration_db_path,
        )

        # Pre-create the CoS row (dispatch path would do this).
        store = OrchestrationStateStore(orch_db)
        store.insert_chief_of_staff_run(
            brief_id="frontier-ai-fde",
            principal_id="",
            status="running",
            dispatch_plan={"steps": []},
            invocation_order=["linkedin", "github"],
            handoff_payloads={},
            synthesis_output={},
            started_at="2026-05-04T17:00:00+00:00",
        )

        # A minimal Brief stub: only the identity-resolution attrs the
        # helper reads (id / raw / role_title).
        class _BriefStub:
            id = "frontier-ai-fde"
            role_title = "Senior FDE"
            raw = {"brief_id": "frontier-ai-fde"}

        monkeypatch.setattr(
            reflection_engine, "load_brief", lambda _path: _BriefStub()
        )

        evidence_batches = [
            _batch_with_saves(
                source="linkedin",
                candidate_volume=10,
                save_rows=[
                    _full_judgment(
                        candidate_id="li-A",
                        decision="SAVE",
                        confidence=0.9,
                        rationale="Senior FDE.",
                    )
                ],
            ),
            _batch_with_saves(
                source="github",
                candidate_volume=22,
                save_rows=[],
            ),
        ]

        composed = reflection_engine._persist_and_read_handoff_payloads(
            brief_path="/tmp/brief.json",
            market_identity=_market_identity(),
            evidence_batches=evidence_batches,
        )
        assert composed is not None
        assert set(composed.keys()) == {"linkedin", "github"}
        assert composed["linkedin"]["candidate_count"] == 10
        assert composed["github"]["save_count"] == 0

        # Verify the persisted state matches.
        record = chief_of_staff_run_by_brief(
            orch_db, brief_id="frontier-ai-fde"
        )
        assert record is not None
        persisted = json.loads(record.handoff_payloads_json)
        assert set(persisted.keys()) == {"linkedin", "github"}
        assert persisted["linkedin"]["candidate_count"] == 10

    def test_returns_none_when_no_evidence_batch_yields_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from market_intelligence import reflection as reflection_engine

        class _BriefStub:
            id = "no-saves-brief"
            role_title = "X"
            raw = {}

        monkeypatch.setattr(
            reflection_engine, "load_brief", lambda _path: _BriefStub()
        )

        evidence_batches = [
            _batch_with_saves(source="linkedin", candidate_volume=0, save_rows=[]),
            _batch_with_saves(source="", candidate_volume=10, save_rows=[]),
        ]

        composed = reflection_engine._persist_and_read_handoff_payloads(
            brief_path="/tmp/brief.json",
            market_identity=_market_identity(),
            evidence_batches=evidence_batches,
        )
        assert composed is None


# ---------------------------------------------------------------------------
# Synthesis backend accepts the new kwarg
# ---------------------------------------------------------------------------


class TestSynthesisAcceptsHandoffKwarg:
    def test_heuristic_backend_accepts_and_ignores_prior_handoff_payloads(
        self,
    ) -> None:
        """The heuristic backend's deterministic narrative is built from
        per_source_signals alone; the prior_handoff_payloads kwarg is
        accepted for signature uniformity with the LLM backend."""

        backend = HeuristicChiefOfStaffSynthesizer()
        synthesis = backend.synthesize(
            market_identity=_market_identity(),
            per_source_signals={
                "linkedin": {
                    "candidate_count": 47,
                    "save_count": 3,
                    "top_lane": None,
                },
                "github": {
                    "candidate_count": 22,
                    "save_count": 1,
                    "top_lane": None,
                },
            },
            briefing_paragraph="Single-source briefing prose.",
            prior_handoff_payloads={
                "linkedin": {"source": "linkedin", "candidate_count": 47},
                "github": {"source": "github", "candidate_count": 22},
            },
        )
        assert synthesis.paragraph
        assert synthesis.source == "deterministic"
        assert synthesis.confidence > 0.0
