"""Tests for ``tools/evaluate_shadow_thresholds.py`` (slice 7).

Hard rules pinned by these tests:

- No live calls. No imports from ``linkedin/``, ``github/``, or
  ``market_intelligence/``. Stdlib + pytest + the slice-4 aggregator (only
  for fixture construction in the smoke section) + the slice-7 evaluator.
- ``evaluate`` is a pure function: same input -> same output, no I/O.
- Verdict precedence: ``INSUFFICIENT_SAMPLE`` dominates ``FAIL`` which
  dominates ``PASS``.
- Comparators are inclusive (``<=`` / ``>=``).
- Zero-denominator handling: no ``ZeroDivisionError``; rates are ``0.0``.
- Default invocation writes nothing to disk; only ``--report-out`` writes.
- The shipped default thresholds file loads cleanly.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import evaluate_shadow_thresholds as ev  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _default_thresholds_dict() -> dict:
    return {
        "$schema_version": "slice7-v1",
        "description": "test fixture",
        "minimum_sample": {
            "total_rows_read_min": 50,
            "total_compared_min": 25,
        },
        "decision_quality": {
            "max_save_to_reject_rate": 0.05,
            "min_reject_to_save_lift": 0.10,
        },
        "evidence_quality": {
            "max_unavailable_evidence_rate": 0.30,
            "max_weak_citation_rate": 0.20,
        },
    }


def _summary(
    *,
    total_rows_read: int = 100,
    total_compared: int = 50,
    reject_to_save: int = 10,
    save_to_reject: int = 1,
    unavailable_external_evidence: int = 15,
    weak_citations: int = 5,
) -> dict:
    return {
        "total_rows_read": total_rows_read,
        "total_compared": total_compared,
        "reject_to_save": reject_to_save,
        "save_to_reject": save_to_reject,
        "unavailable_external_evidence": unavailable_external_evidence,
        "weak_citations": weak_citations,
    }


def _make_args(**overrides) -> argparse.Namespace:
    base = {
        "summary": "-",
        "thresholds": None,
        "report_out": None,
        "quiet": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_thresholds(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _write_summary_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1-2. load_thresholds
# ---------------------------------------------------------------------------


def test_load_thresholds_happy_path(tmp_path):
    p = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    loaded = ev.load_thresholds(p)
    assert loaded["minimum_sample"]["total_rows_read_min"] == 50
    assert loaded["decision_quality"]["max_save_to_reject_rate"] == 0.05


def test_load_thresholds_missing_minimum_sample_section(tmp_path):
    bad = _default_thresholds_dict()
    del bad["minimum_sample"]
    p = _write_thresholds(tmp_path / "bad.json", bad)
    with pytest.raises(ValueError) as exc_info:
        ev.load_thresholds(p)
    assert "minimum_sample" in str(exc_info.value)


def test_load_thresholds_missing_nested_key(tmp_path):
    bad = _default_thresholds_dict()
    del bad["decision_quality"]["min_reject_to_save_lift"]
    p = _write_thresholds(tmp_path / "bad.json", bad)
    with pytest.raises(ValueError) as exc_info:
        ev.load_thresholds(p)
    assert "min_reject_to_save_lift" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3-4. load_summary
# ---------------------------------------------------------------------------


def test_load_summary_from_file(tmp_path):
    p = _write_summary_json(tmp_path / "s.json", _summary())
    loaded = ev.load_summary(str(p))
    assert loaded["total_rows_read"] == 100


def test_load_summary_from_stdin(monkeypatch):
    payload = json.dumps(_summary(total_rows_read=42))
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    loaded = ev.load_summary("-")
    assert loaded["total_rows_read"] == 42


def test_load_summary_rejects_non_object_top_level(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        ev.load_summary(str(p))


# ---------------------------------------------------------------------------
# 5. evaluate PASS
# ---------------------------------------------------------------------------


def test_evaluate_pass_case():
    summary = _summary(
        total_rows_read=100,
        total_compared=50,
        reject_to_save=10,
        save_to_reject=1,
        unavailable_external_evidence=15,
        weak_citations=5,
    )
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    assert report.verdict == ev.VERDICT_PASS
    assert all(c.passed for c in report.sample_checks)
    assert all(c.passed for c in report.quality_checks)
    assert report.inputs["save_to_reject_rate"] == pytest.approx(0.02)
    assert report.inputs["reject_to_save_rate"] == pytest.approx(0.20)
    assert report.inputs["unavailable_evidence_rate"] == pytest.approx(0.15)
    assert report.inputs["weak_citation_rate"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 6. INSUFFICIENT_SAMPLE: quality checks still computed
# ---------------------------------------------------------------------------


def test_evaluate_insufficient_sample_still_computes_quality_diagnostics():
    summary = _summary(
        total_rows_read=10,
        total_compared=5,
        reject_to_save=2,
        save_to_reject=0,
        unavailable_external_evidence=1,
        weak_citations=0,
    )
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    assert report.verdict == ev.VERDICT_INSUFFICIENT_SAMPLE
    assert any(not c.passed for c in report.sample_checks)
    assert len(report.quality_checks) == 4
    by_name = {c.name: c for c in report.quality_checks}
    assert "max_save_to_reject_rate" in by_name
    assert "min_reject_to_save_lift" in by_name
    assert "max_unavailable_evidence_rate" in by_name
    assert "max_weak_citation_rate" in by_name


# ---------------------------------------------------------------------------
# 7. FAIL decision-quality: save_to_reject too high
# ---------------------------------------------------------------------------


def test_evaluate_fail_save_to_reject_rate_too_high():
    summary = _summary(
        total_rows_read=100,
        total_compared=50,
        save_to_reject=10,
        reject_to_save=10,
        unavailable_external_evidence=10,
        weak_citations=5,
    )
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    assert report.verdict == ev.VERDICT_FAIL
    by_name = {c.name: c for c in report.quality_checks}
    assert by_name["max_save_to_reject_rate"].passed is False
    assert report.inputs["save_to_reject_rate"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# 8. FAIL decision-quality: low lift
# ---------------------------------------------------------------------------


def test_evaluate_fail_low_reject_to_save_lift():
    summary = _summary(
        total_rows_read=100,
        total_compared=50,
        save_to_reject=1,
        reject_to_save=2,
        unavailable_external_evidence=10,
        weak_citations=5,
    )
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    assert report.verdict == ev.VERDICT_FAIL
    by_name = {c.name: c for c in report.quality_checks}
    assert by_name["min_reject_to_save_lift"].passed is False
    assert report.inputs["reject_to_save_rate"] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# 9. FAIL evidence-quality: too many unavailable
# ---------------------------------------------------------------------------


def test_evaluate_fail_unavailable_evidence_rate_too_high():
    summary = _summary(
        total_rows_read=100,
        total_compared=50,
        save_to_reject=1,
        reject_to_save=10,
        unavailable_external_evidence=40,
        weak_citations=5,
    )
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    assert report.verdict == ev.VERDICT_FAIL
    by_name = {c.name: c for c in report.quality_checks}
    assert by_name["max_unavailable_evidence_rate"].passed is False
    assert report.inputs["unavailable_evidence_rate"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# 10. FAIL evidence-quality: too many weak citations
# ---------------------------------------------------------------------------


def test_evaluate_fail_weak_citation_rate_too_high():
    summary = _summary(
        total_rows_read=100,
        total_compared=50,
        save_to_reject=1,
        reject_to_save=10,
        unavailable_external_evidence=10,
        weak_citations=25,
    )
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    assert report.verdict == ev.VERDICT_FAIL
    by_name = {c.name: c for c in report.quality_checks}
    assert by_name["max_weak_citation_rate"].passed is False
    assert report.inputs["weak_citation_rate"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 11. Sample-failure dominance
# ---------------------------------------------------------------------------


def test_evaluate_sample_failure_dominates_quality_failure():
    """Even a catastrophic save_to_reject rate cannot flip the verdict to FAIL
    when the sample is below the floor -- the verdict must be
    INSUFFICIENT_SAMPLE."""
    summary = _summary(
        total_rows_read=10,
        total_compared=4,
        save_to_reject=2,
        reject_to_save=0,
        unavailable_external_evidence=8,
        weak_citations=8,
    )
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    assert report.verdict == ev.VERDICT_INSUFFICIENT_SAMPLE
    assert report.inputs["save_to_reject_rate"] == pytest.approx(0.50)
    by_name = {c.name: c for c in report.quality_checks}
    assert by_name["max_save_to_reject_rate"].passed is False


# ---------------------------------------------------------------------------
# 12. Boundary inclusivity: at-the-limit values pass
# ---------------------------------------------------------------------------


def test_evaluate_at_threshold_boundary_passes():
    summary = _summary(
        total_rows_read=50,
        total_compared=25,
        save_to_reject=1,  # 1/25 = 0.04 (<= 0.05 inclusive)
        reject_to_save=3,  # 3/25 = 0.12 (>= 0.10 inclusive)
        unavailable_external_evidence=15,  # 15/50 = 0.30 (<= 0.30 inclusive)
        weak_citations=10,  # 10/50 = 0.20 (<= 0.20 inclusive)
    )
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    assert report.verdict == ev.VERDICT_PASS
    by_name_sample = {c.name: c for c in report.sample_checks}
    assert by_name_sample["min_total_rows_read"].passed is True
    assert by_name_sample["min_total_compared"].passed is True
    by_name_quality = {c.name: c for c in report.quality_checks}
    for name in (
        "max_save_to_reject_rate",
        "min_reject_to_save_lift",
        "max_unavailable_evidence_rate",
        "max_weak_citation_rate",
    ):
        assert by_name_quality[name].passed is True, name


# ---------------------------------------------------------------------------
# 13. Zero-denominator handling
# ---------------------------------------------------------------------------


def test_evaluate_zero_denominators_no_division_error():
    summary = _summary(
        total_rows_read=0,
        total_compared=0,
        save_to_reject=0,
        reject_to_save=0,
        unavailable_external_evidence=0,
        weak_citations=0,
    )
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    assert report.verdict == ev.VERDICT_INSUFFICIENT_SAMPLE
    assert report.inputs["save_to_reject_rate"] == 0.0
    assert report.inputs["reject_to_save_rate"] == 0.0
    assert report.inputs["unavailable_evidence_rate"] == 0.0
    assert report.inputs["weak_citation_rate"] == 0.0


# ---------------------------------------------------------------------------
# 14-15. format_report
# ---------------------------------------------------------------------------


def test_format_report_quiet_returns_only_verdict():
    summary = _summary()
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    out = ev.format_report(report, quiet=True)
    assert out == report.verdict
    assert "\n" not in out


def test_format_report_full_contains_rates_and_all_checks():
    summary = _summary()
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    out = ev.format_report(report, quiet=False)
    assert "Shadow Bakeoff Acceptance Evaluation" in out
    assert "Verdict:" in out
    assert "save_to_reject_rate" in out
    assert "reject_to_save_rate" in out
    assert "unavailable_evidence_rate" in out
    assert "weak_citation_rate" in out
    assert "min_total_rows_read" in out
    assert "min_total_compared" in out
    assert "max_save_to_reject_rate" in out
    assert "min_reject_to_save_lift" in out
    assert "max_unavailable_evidence_rate" in out
    assert "max_weak_citation_rate" in out


def test_format_report_insufficient_sample_marks_quality_diagnostic_only():
    summary = _summary(total_rows_read=10, total_compared=5)
    report = ev.evaluate(summary=summary, thresholds=_default_thresholds_dict())
    assert report.verdict == ev.VERDICT_INSUFFICIENT_SAMPLE
    out = ev.format_report(report, quiet=False)
    assert "[N/A - insufficient sample]" in out


# ---------------------------------------------------------------------------
# 16. run end-to-end PASS
# ---------------------------------------------------------------------------


def test_run_end_to_end_pass(capsys, tmp_path):
    s = _write_summary_json(tmp_path / "s.json", _summary())
    t = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    args = _make_args(summary=str(s), thresholds=str(t))
    rc = ev.run(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Verdict: PASS" in out


# ---------------------------------------------------------------------------
# 17. run end-to-end FAIL
# ---------------------------------------------------------------------------


def test_run_end_to_end_fail(capsys, tmp_path):
    s = _write_summary_json(
        tmp_path / "s.json",
        _summary(save_to_reject=10, reject_to_save=10),
    )
    t = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    args = _make_args(summary=str(s), thresholds=str(t))
    rc = ev.run(args)
    out = capsys.readouterr().out
    assert rc == 1
    assert "Verdict: FAIL" in out


# ---------------------------------------------------------------------------
# 18. run end-to-end INSUFFICIENT_SAMPLE
# ---------------------------------------------------------------------------


def test_run_end_to_end_insufficient_sample(capsys, tmp_path):
    s = _write_summary_json(
        tmp_path / "s.json",
        _summary(total_rows_read=10, total_compared=5),
    )
    t = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    args = _make_args(summary=str(s), thresholds=str(t))
    rc = ev.run(args)
    out = capsys.readouterr().out
    assert rc == 2
    assert "Verdict: INSUFFICIENT_SAMPLE" in out


# ---------------------------------------------------------------------------
# 19. --report-out writes structured JSON
# ---------------------------------------------------------------------------


def test_run_writes_structured_report_out(capsys, tmp_path):
    s = _write_summary_json(tmp_path / "s.json", _summary())
    t = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    out_path = tmp_path / "report.json"
    args = _make_args(
        summary=str(s),
        thresholds=str(t),
        report_out=str(out_path),
    )
    rc = ev.run(args)
    capsys.readouterr()
    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert "inputs" in payload
    assert "sample_checks" in payload
    assert "quality_checks" in payload
    assert payload["thresholds_path"] == str(t)
    assert payload["summary_path"] == str(s)
    assert len(payload["sample_checks"]) == 2
    assert len(payload["quality_checks"]) == 4
    sample_first = payload["sample_checks"][0]
    for key in ("name", "passed", "observed", "threshold", "comparator", "category"):
        assert key in sample_first


# ---------------------------------------------------------------------------
# 20. --thresholds override flips FAIL -> PASS
# ---------------------------------------------------------------------------


def test_run_thresholds_override_flips_fail_to_pass(capsys, tmp_path):
    summary = _summary(save_to_reject=10, reject_to_save=10)  # FAIL on defaults
    s = _write_summary_json(tmp_path / "s.json", summary)
    default_t = _write_thresholds(tmp_path / "default.json", _default_thresholds_dict())

    args_default = _make_args(summary=str(s), thresholds=str(default_t))
    rc_default = ev.run(args_default)
    capsys.readouterr()
    assert rc_default == 1

    looser = _default_thresholds_dict()
    looser["decision_quality"]["max_save_to_reject_rate"] = 0.50
    override_t = _write_thresholds(tmp_path / "override.json", looser)

    args_override = _make_args(summary=str(s), thresholds=str(override_t))
    rc_override = ev.run(args_override)
    out = capsys.readouterr().out
    assert rc_override == 0
    assert "Verdict: PASS" in out


# ---------------------------------------------------------------------------
# 21. The shipped default thresholds file loads cleanly
# ---------------------------------------------------------------------------


def test_default_thresholds_file_loads_cleanly():
    """Guards the shipped config/perplexity-bakeoff-thresholds.json against
    drifting out of step with the loader's required-key schema."""
    loaded = ev.load_thresholds(ev.DEFAULT_THRESHOLDS_PATH)
    assert "minimum_sample" in loaded
    assert "decision_quality" in loaded
    assert "evidence_quality" in loaded
    assert isinstance(
        loaded["minimum_sample"]["total_rows_read_min"], int
    )
    assert isinstance(
        loaded["minimum_sample"]["total_compared_min"], int
    )
    assert isinstance(
        loaded["decision_quality"]["max_save_to_reject_rate"], (int, float)
    )
    assert isinstance(
        loaded["decision_quality"]["min_reject_to_save_lift"], (int, float)
    )
    assert isinstance(
        loaded["evidence_quality"]["max_unavailable_evidence_rate"], (int, float)
    )
    assert isinstance(
        loaded["evidence_quality"]["max_weak_citation_rate"], (int, float)
    )
