"""Tests for the chief-of-staff integration into ``reflection_phase_plan``.

The agent itself is comprehensively tested in
``tests/test_chief_of_staff_agent.py`` (cascade, telemetry, heuristic,
schema validator, drift checker). This file covers the slice 2
integration glue:

- ``_chief_of_staff_enabled`` — default-on; explicit disable via
  ``0`` / ``false`` / ``no`` after strip + lowercasing.
- ``_contributing_sources_count`` — counts distinct candidate-
  producing sources; excludes zero-candidate sources; treats
  whitespace / case the same way.
- ``_per_source_signals_from_batches`` — aggregates across multiple
  batches per source; excludes zero-candidate sources; normalizes
  source keys to lowercased.

``reflection_phase_plan`` is exercised with patched evidence batches and
run-dir resolution under pytest so the chief-of-staff arm runs on
multiple contributing sources without a full ``output/runs/`` fixture tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from market_intelligence import reflection as reflection_engine
from market_intelligence.schema import MarketEvidenceBatch


def _batch(
    *,
    source: str,
    candidate_volume: int,
    saved: int = 0,
    run_ref: str = "ref-x",
) -> MarketEvidenceBatch:
    """Build a minimal :class:`MarketEvidenceBatch` with the metrics
    fields the integration helpers actually read.

    Other dataclass fields default to empty / None — the helpers do
    not touch them, so a thin shape is enough.
    """

    return MarketEvidenceBatch(
        run_ref=run_ref,
        source=source,
        output_dir="/tmp/fake",
        brief_version="v1",
        generated_at="2026-05-04T00:00:00+00:00",
        metrics_summary={
            "run_count": 1,
            "candidate_volume": int(candidate_volume),
            "saved": int(saved),
            "facial_yes": 0,
            "facial_no": 0,
            "rejected": 0,
        },
    )


def _minimal_v2_brief_for_reflection() -> dict:
    """Valid V2 brief dict for :func:`load_brief` / ``reflection_phase_plan``.

    Mirrors the minimal shape in ``tests/test_brief_loader.py`` —
    capability_areas + depth_distinction so ``_is_v2_brief`` holds.
    """

    return {
        "role_title": "VP Engineering",
        "role_summary": "Owns engineering org for a series-C company.",
        "geography": "United States",
        "linkedin_project": "exec-search-vp-eng",
        "minimum_years_experience": 12,
        "minimum_bar_description": "10+ years engineering leadership.",
        "capability_areas": [
            {
                "name": "Org leadership",
                "description": "Builds and runs 50+ person engineering orgs.",
                "builder_signals": ["VP-level scope", "headcount growth"],
                "user_signals": ["IC-level work primarily"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns engineering strategy + delivery.",
            "user_definition": "Manages individual teams without org-wide scope.",
            "edge_case_guidance": "Borderline = full eval.",
        },
    }


# ---------------------------------------------------------------------------
# Env-var gate
# ---------------------------------------------------------------------------


class TestChiefOfStaffEnabled:
    def test_unset_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLORIS_CHIEF_OF_STAFF_ENABLED", raising=False)
        assert reflection_engine._chief_of_staff_enabled() is True

    def test_empty_string_enables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLORIS_CHIEF_OF_STAFF_ENABLED", "")
        assert reflection_engine._chief_of_staff_enabled() is True

    @pytest.mark.parametrize(
        "raw",
        [
            "0",
            "false",
            "no",
            "FALSE",
            "No ",
            " 0 ",
        ],
    )
    def test_explicit_disable_with_falsy_sentinels(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("CLORIS_CHIEF_OF_STAFF_ENABLED", raw)
        assert reflection_engine._chief_of_staff_enabled() is False

    @pytest.mark.parametrize(
        "raw",
        [
            "1",
            "true",
            "yes",
            "anything-else",
            "literally-any-non-disable-string",
            "TRUE",
            "YES ",
        ],
    )
    def test_truthy_values_still_enable(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("CLORIS_CHIEF_OF_STAFF_ENABLED", raw)
        assert reflection_engine._chief_of_staff_enabled() is True

    def test_whitespace_tolerant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLORIS_CHIEF_OF_STAFF_ENABLED", "  true  ")
        assert reflection_engine._chief_of_staff_enabled() is True


# ---------------------------------------------------------------------------
# Candidate-producing sources guard
# ---------------------------------------------------------------------------


class TestContributingSourcesCount:
    def test_zero_when_empty(self) -> None:
        assert reflection_engine._contributing_sources_count([]) == 0

    def test_zero_when_no_candidates_anywhere(self) -> None:
        batches = [
            _batch(source="linkedin", candidate_volume=0),
            _batch(source="github", candidate_volume=0),
        ]
        assert reflection_engine._contributing_sources_count(batches) == 0

    def test_one_when_one_source_produced_candidates(self) -> None:
        batches = [
            _batch(source="linkedin", candidate_volume=47, saved=3),
            _batch(source="github", candidate_volume=0),
        ]
        assert reflection_engine._contributing_sources_count(batches) == 1

    def test_two_when_two_distinct_candidate_producing_sources(self) -> None:
        batches = [
            _batch(source="linkedin", candidate_volume=47),
            _batch(source="github", candidate_volume=22),
        ]
        assert reflection_engine._contributing_sources_count(batches) == 2

    def test_dedupes_multiple_runs_of_same_source(self) -> None:
        batches = [
            _batch(source="linkedin", candidate_volume=47, run_ref="r1"),
            _batch(source="linkedin", candidate_volume=12, run_ref="r2"),
        ]
        assert reflection_engine._contributing_sources_count(batches) == 1

    def test_no_save_does_not_block_count(self) -> None:
        # A source with candidates but zero saves still counts — the
        # negative read is informative.
        batches = [
            _batch(source="linkedin", candidate_volume=47, saved=3),
            _batch(source="github", candidate_volume=22, saved=0),
        ]
        assert reflection_engine._contributing_sources_count(batches) == 2

    def test_case_normalized(self) -> None:
        # Defensive — source keys land lowercased throughout the
        # codebase; if a batch arrives with mixed case, the helper
        # collapses it to one bucket so the integration's >=2 guard
        # doesn't double-count.
        batches = [
            _batch(source="LinkedIn", candidate_volume=47),
            _batch(source="linkedin", candidate_volume=12),
        ]
        assert reflection_engine._contributing_sources_count(batches) == 1

    def test_blank_source_excluded(self) -> None:
        batches = [
            _batch(source="", candidate_volume=10),
            _batch(source="linkedin", candidate_volume=20),
        ]
        assert reflection_engine._contributing_sources_count(batches) == 1


# ---------------------------------------------------------------------------
# Per-source signal derivation
# ---------------------------------------------------------------------------


class TestPerSourceSignalsFromBatches:
    def test_empty_batches_returns_empty(self) -> None:
        assert reflection_engine._per_source_signals_from_batches([]) == {}

    def test_excludes_zero_candidate_sources(self) -> None:
        batches = [
            _batch(source="linkedin", candidate_volume=47, saved=3),
            _batch(source="github", candidate_volume=0),
        ]
        signals = reflection_engine._per_source_signals_from_batches(batches)
        assert set(signals.keys()) == {"linkedin"}
        assert signals["linkedin"]["candidate_count"] == 47
        assert signals["linkedin"]["save_count"] == 3
        assert signals["linkedin"]["top_lane"] is None

    def test_aggregates_across_multiple_batches_per_source(self) -> None:
        batches = [
            _batch(source="linkedin", candidate_volume=47, saved=3),
            _batch(source="linkedin", candidate_volume=12, saved=1),
            _batch(source="github", candidate_volume=22, saved=0),
        ]
        signals = reflection_engine._per_source_signals_from_batches(batches)
        assert signals["linkedin"]["candidate_count"] == 59
        assert signals["linkedin"]["save_count"] == 4
        assert signals["github"]["candidate_count"] == 22
        assert signals["github"]["save_count"] == 0

    def test_normalizes_source_keys_to_lowercase(self) -> None:
        batches = [
            _batch(source="LinkedIn", candidate_volume=47, saved=3),
            _batch(source="linkedin", candidate_volume=12, saved=1),
        ]
        signals = reflection_engine._per_source_signals_from_batches(batches)
        assert set(signals.keys()) == {"linkedin"}
        assert signals["linkedin"]["candidate_count"] == 59

    def test_keys_match_contributing_sources_count(self) -> None:
        # The two helpers must agree: any source in the signals dict
        # must also be counted by _contributing_sources_count, and
        # vice versa. This invariant is what makes the cascade's
        # specialist_weight_invalid route safe — the agent only sees
        # sources the integration has already classified as
        # contributing.
        batches = [
            _batch(source="linkedin", candidate_volume=47, saved=3),
            _batch(source="github", candidate_volume=22, saved=0),
            _batch(source="researcher", candidate_volume=0),
        ]
        count = reflection_engine._contributing_sources_count(batches)
        signals = reflection_engine._per_source_signals_from_batches(batches)
        assert count == len(signals) == 2
        assert "researcher" not in signals

    def test_blank_source_excluded(self) -> None:
        batches = [
            _batch(source="", candidate_volume=10, saved=1),
            _batch(source="linkedin", candidate_volume=20, saved=2),
        ]
        signals = reflection_engine._per_source_signals_from_batches(batches)
        assert set(signals.keys()) == {"linkedin"}


# ---------------------------------------------------------------------------
# reflection_phase_plan — chief-of-staff synthesis (gate default-on)
# ---------------------------------------------------------------------------


class TestReflectionPhasePlanChiefOfStaffSynthesis:
    def test_synthesis_populated_multi_source_env_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Env unset → gate on; ≥2 contributing sources → synthesize runs."""

        monkeypatch.delenv("CLORIS_CHIEF_OF_STAFF_ENABLED", raising=False)

        brief_path = tmp_path / "brief.json"
        brief_path.write_text(json.dumps(_minimal_v2_brief_for_reflection()))
        run_dir = tmp_path / "resolved-run-dir"
        run_dir.mkdir(parents=True)

        batches = [
            _batch(source="linkedin", candidate_volume=5, run_ref="run-li"),
            _batch(source="github", candidate_volume=3, run_ref="run-gh"),
        ]

        with (
            patch.object(
                reflection_engine,
                "_resolve_market_intel_run_dir",
                return_value=run_dir,
            ),
            patch.object(
                reflection_engine,
                "_collect_evidence_batches",
                return_value=batches,
            ),
        ):
            plan = reflection_engine.reflection_phase_plan(
                brief_path=brief_path,
                run_dir=run_dir,
                mode="post_run",
            )

        plan_block = (plan.get("phase_outputs") or {}).get("plan") or {}
        synthesis = plan_block.get("chief_of_staff_synthesis")
        assert synthesis is not None
        assert isinstance(synthesis, dict)
        assert str(synthesis.get("paragraph") or "").strip()
