"""Tests for P3.6: facial calibration closes the loop.

Covers the four seams:
  1. linkedin/orchestrator.py:_build_run_report_snapshot — the per-run
     facial_calibration comparison block + cross-run drift warning.
  2. market_intelligence/engine.py:_compute_facial_calibration_observed —
     the artifact-level facial_calibration_observed block + the
     consecutive-out-of-band counter lifecycle.
  3. market_intelligence/reflection.py:_facial_calibration_drift_propose_hunks
     — the Gate-2 recalibration hunk.

Run with: python -m pytest tests/test_facial_calibration_loop.py -v
"""

import json
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.schemas import Progress, SearchString
from tests.test_linkedin_pipeline import _make_pipeline

from market_intelligence import MarketIdentity
from market_intelligence.engine import (
    _build_artifact,
    _compute_facial_calibration_observed,
)
from market_intelligence.reflection import (
    StructuredSectionHunkError,
    _apply_hunk_to_brief,
    _facial_calibration_drift_propose_hunks,
)
from market_intelligence.schema import MarketEvidenceBatch


def _progress() -> Progress:
    return Progress(
        brief_name="test",
        strings=[SearchString(id=1, name="test", boolean="foo", status="done", pages_reviewed=1)],
    )


def _authored_band(low: float, high: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(expected_yes_rate_low=low, expected_yes_rate_high=high)


# ---------------------------------------------------------------------------
# (a)/(b) Snapshot: linkedin/orchestrator.py:_build_run_report_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_facial_calibration_out_of_band():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj._new_brief = types.SimpleNamespace(
            facial_calibration=_authored_band(0.2, 0.3)
        )
        p.stats.update({"facial_yes": 80, "facial_no": 20})

        snapshot = p._build_run_report_snapshot(_progress())
        fc = snapshot["metrics_summary"]["facial_calibration"]

        assert fc["status"] == "ok"
        assert fc["actual_yes_rate"] == 0.8
        assert fc["authored_low"] == 0.2
        assert fc["authored_high"] == 0.3
        assert fc["deviation_from_band"] == 0.5
        assert fc["out_of_band"] is True
        # No prior artifact on disk -> no drift warning yet, even though
        # this run itself is out of band (drift requires 2 consecutive).
        assert fc["calibration_drift_warning"] is False


def test_snapshot_facial_calibration_in_band():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj._new_brief = types.SimpleNamespace(
            facial_calibration=_authored_band(0.2, 0.9)
        )
        p.stats.update({"facial_yes": 40, "facial_no": 60})

        snapshot = p._build_run_report_snapshot(_progress())
        fc = snapshot["metrics_summary"]["facial_calibration"]

        assert fc["status"] == "ok"
        assert fc["actual_yes_rate"] == 0.4
        assert fc["deviation_from_band"] == 0.0
        assert fc["out_of_band"] is False
        assert fc["calibration_drift_warning"] is False


def test_snapshot_facial_calibration_no_verdicts_is_not_an_affirmative_zero():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj._new_brief = types.SimpleNamespace(
            facial_calibration=_authored_band(0.2, 0.3)
        )
        p.stats.update({"facial_yes": 0, "facial_no": 0})

        snapshot = p._build_run_report_snapshot(_progress())
        fc = snapshot["metrics_summary"]["facial_calibration"]

        # Doctrine invariant: no affirmative 0.0 rate when "missing" is
        # representable — the block must say so, not report 0.0.
        assert fc == {"status": "no_facial_verdicts"}


def test_snapshot_facial_calibration_band_not_authored():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)  # has_v2_schema=False by default
        p.stats.update({"facial_yes": 5, "facial_no": 5})

        snapshot = p._build_run_report_snapshot(_progress())
        fc = snapshot["metrics_summary"]["facial_calibration"]

        assert fc["status"] == "band_not_authored"
        assert fc["actual_yes_rate"] == 0.5
        assert "authored_low" not in fc
        assert "out_of_band" not in fc


# ---------------------------------------------------------------------------
# (c) Snapshot: cross-run drift warning
# ---------------------------------------------------------------------------


def _write_prior_artifact(path: Path, consecutive: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"facial_calibration_observed": {"consecutive_out_of_band_runs": consecutive}}
        )
    )


def test_snapshot_drift_warning_true_when_out_of_band_and_prior_out_of_band():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj._new_brief = types.SimpleNamespace(
            facial_calibration=_authored_band(0.2, 0.3)
        )
        p.stats.update({"facial_yes": 80, "facial_no": 20})

        prior_path = Path(td) / "market-intel.json"
        _write_prior_artifact(prior_path, consecutive=1)

        with patch(
            "market_intelligence.engine.resolve_market_intel_artifact_path",
            return_value=prior_path,
        ):
            snapshot = p._build_run_report_snapshot(_progress())
        fc = snapshot["metrics_summary"]["facial_calibration"]

        assert fc["out_of_band"] is True
        assert fc["calibration_drift_warning"] is True


def test_snapshot_drift_warning_false_when_prior_was_in_band():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj._new_brief = types.SimpleNamespace(
            facial_calibration=_authored_band(0.2, 0.3)
        )
        p.stats.update({"facial_yes": 80, "facial_no": 20})

        prior_path = Path(td) / "market-intel.json"
        _write_prior_artifact(prior_path, consecutive=0)

        with patch(
            "market_intelligence.engine.resolve_market_intel_artifact_path",
            return_value=prior_path,
        ):
            snapshot = p._build_run_report_snapshot(_progress())
        fc = snapshot["metrics_summary"]["facial_calibration"]

        assert fc["out_of_band"] is True
        assert fc["calibration_drift_warning"] is False


def test_snapshot_drift_warning_false_when_no_prior_artifact_on_disk():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj._new_brief = types.SimpleNamespace(
            facial_calibration=_authored_band(0.2, 0.3)
        )
        p.stats.update({"facial_yes": 80, "facial_no": 20})

        missing_path = Path(td) / "does-not-exist" / "market-intel.json"

        with patch(
            "market_intelligence.engine.resolve_market_intel_artifact_path",
            return_value=missing_path,
        ):
            snapshot = p._build_run_report_snapshot(_progress())
        fc = snapshot["metrics_summary"]["facial_calibration"]

        assert fc["out_of_band"] is True
        assert fc["calibration_drift_warning"] is False


def test_snapshot_drift_lookup_failure_is_warning_only(capsys):
    """A corrupt/unreadable prior artifact must never crash snapshot building."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj._new_brief = types.SimpleNamespace(
            facial_calibration=_authored_band(0.2, 0.3)
        )
        p.stats.update({"facial_yes": 80, "facial_no": 20})

        malformed_path = Path(td) / "malformed-market-intel.json"
        malformed_path.write_text("{not valid json")

        with patch(
            "market_intelligence.engine.resolve_market_intel_artifact_path",
            return_value=malformed_path,
        ):
            snapshot = p._build_run_report_snapshot(_progress())
        fc = snapshot["metrics_summary"]["facial_calibration"]

        assert fc["out_of_band"] is True
        assert fc["calibration_drift_warning"] is False


# ---------------------------------------------------------------------------
# (d) Artifact: market_intelligence/engine.py:_compute_facial_calibration_observed
# ---------------------------------------------------------------------------


def _batch(generated_at: str, metrics_summary: dict, run_ref: str = "run") -> MarketEvidenceBatch:
    return MarketEvidenceBatch(
        run_ref=run_ref,
        source="linkedin",
        output_dir=f"/tmp/{run_ref}",
        brief_version="1.0",
        generated_at=generated_at,
        report={"metrics_summary": metrics_summary},
    )


def _new_style_fc_metrics(actual: float, low: float, high: float, out_of_band: bool) -> dict:
    deviation = 0.0 if low <= actual <= high else round(min(abs(actual - low), abs(actual - high)), 4)
    return {
        "facial_calibration": {
            "status": "ok",
            "actual_yes_rate": actual,
            "authored_low": low,
            "authored_high": high,
            "deviation_from_band": deviation,
            "out_of_band": out_of_band,
        }
    }


def test_artifact_consecutive_out_of_band_counter_increments_and_resets():
    batch1 = _batch(
        "2026-07-01T00:00:00+00:00",
        _new_style_fc_metrics(0.8, 0.2, 0.3, True),
        run_ref="run-1",
    )
    observed_1 = _compute_facial_calibration_observed(
        evidence_batches=[batch1], previous={}, brief=None
    )
    assert observed_1["out_of_band"] is True
    assert observed_1["consecutive_out_of_band_runs"] == 1

    batch2 = _batch(
        "2026-07-02T00:00:00+00:00",
        _new_style_fc_metrics(0.82, 0.2, 0.3, True),
        run_ref="run-2",
    )
    observed_2 = _compute_facial_calibration_observed(
        evidence_batches=[batch1, batch2],
        previous={"facial_calibration_observed": observed_1},
        brief=None,
    )
    assert observed_2["consecutive_out_of_band_runs"] == 2
    assert observed_2["run_ref"] == "run-2"  # latest by generated_at

    batch3 = _batch(
        "2026-07-03T00:00:00+00:00",
        _new_style_fc_metrics(0.25, 0.2, 0.3, False),
        run_ref="run-3",
    )
    observed_3 = _compute_facial_calibration_observed(
        evidence_batches=[batch1, batch2, batch3],
        previous={"facial_calibration_observed": observed_2},
        brief=None,
    )
    assert observed_3["out_of_band"] is False
    assert observed_3["consecutive_out_of_band_runs"] == 0


def test_artifact_no_facial_verdicts_carries_previous_forward_without_incrementing():
    previous_block = {"consecutive_out_of_band_runs": 2, "status": "ok", "out_of_band": True}
    batch = _batch(
        "2026-07-04T00:00:00+00:00",
        {"facial_calibration": {"status": "no_facial_verdicts"}},
    )
    observed = _compute_facial_calibration_observed(
        evidence_batches=[batch],
        previous={"facial_calibration_observed": previous_block},
        brief=None,
    )
    assert observed == previous_block


def test_artifact_no_evidence_batches_carries_previous_forward():
    previous_block = {"consecutive_out_of_band_runs": 1}
    observed = _compute_facial_calibration_observed(
        evidence_batches=[],
        previous={"facial_calibration_observed": previous_block},
        brief=None,
    )
    assert observed == previous_block


def test_artifact_old_style_report_recomputes_from_raw_counts():
    """A pre-P3.6 report has no metrics_summary.facial_calibration key."""
    batch = _batch(
        "2026-07-05T00:00:00+00:00",
        {"facial_yes": 80, "facial_no": 20},
        run_ref="legacy-run",
    )
    brief = types.SimpleNamespace(
        has_v2_schema=True,
        _new_brief=types.SimpleNamespace(facial_calibration=_authored_band(0.2, 0.3)),
    )
    observed = _compute_facial_calibration_observed(
        evidence_batches=[batch], previous={}, brief=brief
    )
    assert observed["actual_yes_rate"] == 0.8
    assert observed["out_of_band"] is True
    assert observed["consecutive_out_of_band_runs"] == 1


def test_artifact_old_style_report_band_not_authored_when_brief_lacks_calibration():
    batch = _batch(
        "2026-07-06T00:00:00+00:00",
        {"facial_yes": 5, "facial_no": 5},
        run_ref="legacy-run-2",
    )
    brief = types.SimpleNamespace(has_v2_schema=False)
    observed = _compute_facial_calibration_observed(
        evidence_batches=[batch], previous={}, brief=brief
    )
    assert observed == {"status": "band_not_authored"}


def test_artifact_computation_never_raises_on_malformed_batch():
    """Fail-soft: a batch whose report is a garbage shape must degrade to
    the previous block, never raise out of ingestion."""
    previous_block = {"consecutive_out_of_band_runs": 3}
    bad_batch = _batch("2026-07-07T00:00:00+00:00", {"facial_calibration": "not-a-dict"})
    # Corrupt the report further to force an exception path.
    bad_batch.report = {"metrics_summary": None}
    observed = _compute_facial_calibration_observed(
        evidence_batches=[bad_batch],
        previous={"facial_calibration_observed": previous_block},
        brief=None,
    )
    assert observed == previous_block


# ---------------------------------------------------------------------------
# (e) Reflection: market_intelligence/reflection.py:_facial_calibration_drift_propose_hunks
# ---------------------------------------------------------------------------


def test_reflection_no_hunk_when_counter_below_threshold():
    artifact_dict = {
        "facial_calibration_observed": {
            "consecutive_out_of_band_runs": 1,
            "actual_yes_rate": 0.8,
        }
    }
    brief_raw = {"facial_calibration": {"expected_yes_rate_low": 0.2, "expected_yes_rate_high": 0.3}}

    assert (
        _facial_calibration_drift_propose_hunks(artifact_dict=artifact_dict, brief_raw=brief_raw)
        == []
    )


def test_reflection_no_hunk_when_brief_has_no_facial_calibration_section():
    artifact_dict = {
        "facial_calibration_observed": {
            "consecutive_out_of_band_runs": 3,
            "actual_yes_rate": 0.8,
        }
    }
    brief_raw: dict = {}

    assert (
        _facial_calibration_drift_propose_hunks(artifact_dict=artifact_dict, brief_raw=brief_raw)
        == []
    )


def test_reflection_proposes_exactly_one_recentered_band_hunk_at_threshold():
    artifact_dict = {
        "facial_calibration_observed": {
            "consecutive_out_of_band_runs": 2,
            "actual_yes_rate": 0.8,
        }
    }
    brief_raw = {"facial_calibration": {"expected_yes_rate_low": 0.2, "expected_yes_rate_high": 0.3}}

    hunks = _facial_calibration_drift_propose_hunks(artifact_dict=artifact_dict, brief_raw=brief_raw)

    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk["section"] == "facial_calibration"
    assert hunk["kind"] == "facial_yes_rate_band"
    assert hunk["target_field"] == "facial_calibration"
    # NEEDS-REVIEW default — calibration is a proposal, never autopilot.
    assert hunk["default_approved"] is False
    assert hunk["confidence"] < 0.65
    # Recentered on 0.8 preserving the authored width of 0.1: [0.75, 0.85].
    assert "0.75" in hunk["after"]
    assert "0.85" in hunk["after"]
    assert set(hunk.keys()) == {
        "hunk_id",
        "section",
        "kind",
        "label",
        "before",
        "after",
        "rationale",
        "confidence",
        "default_approved",
        "target_field",
    }


def test_reflection_recentered_band_clamps_to_ceiling():
    artifact_dict = {
        "facial_calibration_observed": {
            "consecutive_out_of_band_runs": 2,
            "actual_yes_rate": 0.98,
        }
    }
    brief_raw = {"facial_calibration": {"expected_yes_rate_low": 0.1, "expected_yes_rate_high": 0.7}}

    hunks = _facial_calibration_drift_propose_hunks(artifact_dict=artifact_dict, brief_raw=brief_raw)

    assert len(hunks) == 1
    # Width 0.6, half-width 0.3: naive high = 1.28, clamped to 0.95.
    assert "0.95" in hunks[0]["after"]


def test_reflection_never_writes_the_brief():
    """The hunk is a Gate-2 proposal only; the function must not mutate brief_raw."""
    brief_raw = {"facial_calibration": {"expected_yes_rate_low": 0.2, "expected_yes_rate_high": 0.3}}
    original = json.loads(json.dumps(brief_raw))
    artifact_dict = {
        "facial_calibration_observed": {
            "consecutive_out_of_band_runs": 2,
            "actual_yes_rate": 0.8,
        }
    }

    _facial_calibration_drift_propose_hunks(artifact_dict=artifact_dict, brief_raw=brief_raw)

    assert brief_raw == original


# ---------------------------------------------------------------------------
# (f) Apply: market_intelligence/reflection.py:_apply_hunk_to_brief
#
# FIX 1 (adversarial review): the facial_yes_rate_band hunk's ``after`` is a
# plain string ("expected_yes_rate_low: X\nexpected_yes_rate_high: Y")
# targeting the "facial_calibration" section, which is a DICT in the raw
# brief. Before the fix, this hits the P3.7 StructuredSectionHunkError guard
# unconditionally -- a recruiter who approves the recalibration at Gate 2
# gets no band change. These tests must fail against the pre-fix code.
# ---------------------------------------------------------------------------


def _facial_band_hunk(after: str) -> dict:
    return {
        "hunk_id": "facial-calibration-drift-1",
        "section": "facial_calibration",
        "kind": "facial_yes_rate_band",
        "label": "Recalibrate facial expected yes-rate band",
        "before": "expected_yes_rate_low: 0.2\nexpected_yes_rate_high: 0.3",
        "after": after,
        "rationale": "drift",
        "confidence": 0.4,
        "default_approved": False,
        "target_field": "facial_calibration",
    }


def test_apply_facial_band_hunk_updates_only_the_two_rate_keys():
    brief = {
        "facial_calibration": {
            "expected_yes_rate_low": 0.2,
            "expected_yes_rate_high": 0.3,
            "notes": "authored at preflight",
        },
        "instructions": "keep this untouched",
    }
    hunk = _facial_band_hunk("expected_yes_rate_low: 0.75\nexpected_yes_rate_high: 0.85")

    updated = _apply_hunk_to_brief(brief, hunk)

    assert updated["facial_calibration"]["expected_yes_rate_low"] == 0.75
    assert updated["facial_calibration"]["expected_yes_rate_high"] == 0.85
    # Sibling key preserved.
    assert updated["facial_calibration"]["notes"] == "authored at preflight"
    # Untouched sections preserved; input not mutated.
    assert updated["instructions"] == "keep this untouched"
    assert brief["facial_calibration"]["expected_yes_rate_low"] == 0.2


@pytest.mark.parametrize(
    "after",
    [
        # Extra prose line.
        "expected_yes_rate_low: 0.2\nexpected_yes_rate_high: 0.3\nplease approve",
        # Non-float value.
        "expected_yes_rate_low: not-a-number\nexpected_yes_rate_high: 0.3",
        # low >= high.
        "expected_yes_rate_low: 0.5\nexpected_yes_rate_high: 0.3",
        "expected_yes_rate_low: 0.3\nexpected_yes_rate_high: 0.3",
        # Out of [0, 1] range.
        "expected_yes_rate_low: -0.1\nexpected_yes_rate_high: 0.3",
        "expected_yes_rate_low: 0.2\nexpected_yes_rate_high: 1.5",
    ],
)
def test_apply_facial_band_hunk_malformed_after_refuses_not_corrupts(after):
    brief = {
        "facial_calibration": {
            "expected_yes_rate_low": 0.2,
            "expected_yes_rate_high": 0.3,
        }
    }
    hunk = _facial_band_hunk(after)

    with pytest.raises(StructuredSectionHunkError):
        _apply_hunk_to_brief(brief, hunk)

    # Refusal must not have mutated the input brief.
    assert brief["facial_calibration"]["expected_yes_rate_low"] == 0.2
    assert brief["facial_calibration"]["expected_yes_rate_high"] == 0.3


# ---------------------------------------------------------------------------
# (g) Artifact: market_intelligence/engine.py:_compute_facial_calibration_observed
# FIX 2 (idempotent re-ingestion) + FIX 3 (band-change resets counter)
# ---------------------------------------------------------------------------


def test_artifact_reingestion_of_same_run_is_idempotent():
    """FIX 2: re-observing the SAME run_ref must not inflate the counter."""
    batch = _batch(
        "2026-07-01T00:00:00+00:00",
        _new_style_fc_metrics(0.8, 0.2, 0.3, True),
        run_ref="run-1",
    )
    observed_1 = _compute_facial_calibration_observed(
        evidence_batches=[batch], previous={}, brief=None
    )
    assert observed_1["consecutive_out_of_band_runs"] == 1

    # Re-ingest the SAME run (finalize, then a reflection propose that
    # rebuilds from the on-disk artifact, then a manual
    # tools/update_market_intel.py run) -- must be a no-op.
    observed_2 = _compute_facial_calibration_observed(
        evidence_batches=[batch],
        previous={"facial_calibration_observed": observed_1},
        brief=None,
    )
    assert observed_2 == observed_1
    assert observed_2["consecutive_out_of_band_runs"] == 1

    # A genuinely new run still increments.
    batch2 = _batch(
        "2026-07-02T00:00:00+00:00",
        _new_style_fc_metrics(0.82, 0.2, 0.3, True),
        run_ref="run-2",
    )
    observed_3 = _compute_facial_calibration_observed(
        evidence_batches=[batch, batch2],
        previous={"facial_calibration_observed": observed_2},
        brief=None,
    )
    assert observed_3["consecutive_out_of_band_runs"] == 2


def test_artifact_band_change_resets_counter():
    """FIX 3: drift measured against a revised band must not accumulate
    toward a warning about the OLD band."""
    batch1 = _batch(
        "2026-07-01T00:00:00+00:00",
        _new_style_fc_metrics(0.8, 0.2, 0.3, True),
        run_ref="run-1",
    )
    observed_1 = _compute_facial_calibration_observed(
        evidence_batches=[batch1], previous={}, brief=None
    )
    assert observed_1["consecutive_out_of_band_runs"] == 1

    # Recruiter revises the band between runs (0.2-0.3 -> 0.1-0.2). The new
    # run is out-of-band against the NEW band -- counter restarts at 1.
    batch2 = _batch(
        "2026-07-02T00:00:00+00:00",
        _new_style_fc_metrics(0.6, 0.1, 0.2, True),
        run_ref="run-2",
    )
    observed_2 = _compute_facial_calibration_observed(
        evidence_batches=[batch1, batch2],
        previous={"facial_calibration_observed": observed_1},
        brief=None,
    )
    assert observed_2["authored_low"] == 0.1
    assert observed_2["authored_high"] == 0.2
    assert observed_2["consecutive_out_of_band_runs"] == 1


# ---------------------------------------------------------------------------
# (h) Wiring seam: market_intelligence/engine.py:_build_artifact
# FIX 4(a): the artifact's facial_calibration_observed key must actually
# come from _compute_facial_calibration_observed, not be dropped or
# recomputed inline on the way into the artifact.
# ---------------------------------------------------------------------------


def test_build_artifact_facial_calibration_observed_comes_from_compute_seam():
    identity = MarketIdentity.from_dict(
        {
            "market_key": "forward_deployed_engineer__new_york__ic5_ic6",
            "role_title": "Forward Deployed Engineer",
            "role_level": "IC5-IC6",
            "geography": "New York, New York, United States",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["3000000007"],
            "brief_versions_seen": ["1.3"],
        }
    )
    batch = MarketEvidenceBatch(
        run_ref="linkedin:output/runs/linkedin/3000000007/run-9",
        source="linkedin",
        output_dir="/tmp/run",
        brief_version="1.3",
        generated_at="2026-07-09T00:00:00+00:00",
    )
    deterministic_summary = {
        "freshness": {"generated_at": batch.generated_at},
        "evidence_index": {
            "runs": [
                {
                    "run_ref": batch.run_ref,
                    "source": batch.source,
                    "output_dir": batch.output_dir,
                    "brief_version": batch.brief_version,
                    "generated_at": batch.generated_at,
                }
            ]
        },
        "aggregate_metrics": {},
        "channel_summaries": {},
        "lane_intelligence": [],
        "candidate_signal_summary": {},
    }
    sentinel = {
        "status": "ok",
        "consecutive_out_of_band_runs": 7,
        "run_ref": batch.run_ref,
    }

    with patch(
        "market_intelligence.engine._compute_facial_calibration_observed",
        return_value=sentinel,
    ) as mock_compute:
        artifact = _build_artifact(
            brief=types.SimpleNamespace(
                role_title="Forward Deployed Engineer",
                retrieval_design={},
                raw={},
                search_priorities=[],
                additional_search_terms=[],
            ),
            market_identity=identity,
            deterministic_summary=deterministic_summary,
            evidence_batches=[batch],
            previous_artifact=None,
            generated_sections={},
            preserve_previous_narrative=False,
            external_result=None,
            section_generation_metadata={},
            delta_since_last_run={},
        )

    mock_compute.assert_called_once()
    _, call_kwargs = mock_compute.call_args
    assert call_kwargs["evidence_batches"] == [batch]
    assert artifact.to_dict()["facial_calibration_observed"] == sentinel
