"""Tests for ``tools/experiments/recommend_facial_gate.py`` (slice 10).

Hard rules pinned by these tests:

- No live calls. No imports from ``linkedin/``, ``github/``,
  ``market_intelligence/``, or any production module. Stdlib + pytest +
  the slice-10 recommendation tool only.
- ``evaluate`` is a pure function: same input -> same output, no I/O.
- Verdict precedence: ``NOT_ENOUGH_DATA`` dominates
  ``INVESTIGATE_REGRESSION`` dominates the policy labels.
- Policy precedence: ``TRY_LOOSER_BINARY`` dominates
  ``EXPERIMENT_TERNARY_ONLY`` (smallest safe slice).
- Comparators are inclusive (``<=`` / ``>=``).
- Zero-denominator handling: no ``ZeroDivisionError``; rates / ratios are
  ``0.0``.
- Default invocation writes nothing to disk; only ``--report-out`` writes.
- The shipped default thresholds file loads cleanly.
- Baseline parse-failure breach is NOT a veto; baseline is the reference.
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

from tools.experiments import recommend_facial_gate as rec  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _default_thresholds_dict() -> dict:
    return {
        "$schema_version": "slice10-v1",
        "description": "test fixture",
        "minimum_sample": {"total_snippets_min": 50},
        "parse_quality": {"max_parse_failure_rate": 0.05},
        "recovery_quality": {
            "min_variant_only_recovered_saves": 1,
            "min_reach_full_eval_lift_ratio": 0.10,
            "min_ternary_borderline_signal_ratio": 0.10,
        },
        "cost_quality": {"max_token_proxy_ratio_vs_baseline": 3.0},
    }


def _variant_dict(
    *,
    variant: str,
    total_snippets: int = 120,
    facial_yes: int = 0,
    facial_no: int = 0,
    facial_borderline: int = 0,
    parse_failures: int = 0,
    reach_full_eval: int = 0,
    input_token_proxy_total: int = 400_000,
    ternary_policy: str = "open_borderline",
) -> dict:
    """Mirror ``VariantResult.to_dict()`` exactly.

    Read tools/experiments/facial_gate_experiment.py:VariantResult.to_dict
    to keep this in lock-step with the harness shape.
    """

    return {
        "variant": variant,
        "ternary_policy": ternary_policy,
        "total_snippets": total_snippets,
        "facial_yes": facial_yes,
        "facial_no": facial_no,
        "facial_borderline": facial_borderline,
        "parse_failures": parse_failures,
        "reach_full_eval": reach_full_eval,
        "latency_total_seconds": 1.0,
        "latency_p50_seconds": 0.01,
        "latency_p95_seconds": 0.02,
        "input_token_proxy_total": input_token_proxy_total,
        "output_token_proxy_total": 1000,
        "cost_per_reached_full_eval_proxy": 0.0,
    }


def _comparison_dict(
    *,
    compared: str,
    variant_only_recovered_saves: int = 0,
    likely_false_negatives_under_variant: int = 0,
    ternary_policy: str = "open_borderline",
) -> dict:
    """Mirror analyze_recovery output shape exactly."""

    return {
        "baseline_variant": "baseline",
        "compared_variant": compared,
        "ternary_policy": ternary_policy,
        "shared_total": 120,
        "agreement": 100,
        "disagreement": 20,
        "baseline_yes_variant_no": 0,
        "baseline_no_variant_yes": variant_only_recovered_saves,
        "baseline_no_variant_borderline": 0,
        "baseline_yes_variant_borderline": 0,
        "other_disagreement": 0,
        "recovery_evidence_available": True,
        "baseline_saves_recovered": 10,
        "variant_saves_recovered": 10 + variant_only_recovered_saves,
        "variant_only_recovered_saves": variant_only_recovered_saves,
        "likely_false_negatives_under_variant": likely_false_negatives_under_variant,
        "likely_false_negative_urls": [],
    }


def _summary(
    *,
    baseline: dict | None = None,
    looser: dict | None = None,
    ternary: dict | None = None,
    looser_cmp: dict | None = None,
    ternary_cmp: dict | None = None,
) -> dict:
    variants: dict = {}
    if baseline is not None:
        variants["baseline"] = baseline
    if looser is not None:
        variants["looser"] = looser
    if ternary is not None:
        variants["ternary"] = ternary
    comparisons: dict = {}
    if looser_cmp is not None:
        comparisons["looser"] = looser_cmp
    if ternary_cmp is not None:
        comparisons["ternary"] = ternary_cmp
    return {
        "variants": variants,
        "comparisons": comparisons,
        "config": {},
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


# Healthy reusable fixtures: each variant clears the floor with
# realistic distributions. Token-proxy totals are sized so token ratio
# stays well below 3.0.
def _healthy_baseline() -> dict:
    return _variant_dict(
        variant="baseline",
        total_snippets=120,
        facial_yes=42,
        facial_no=76,
        parse_failures=2,
        reach_full_eval=42,
        input_token_proxy_total=412_800,
    )


def _healthy_looser() -> dict:
    return _variant_dict(
        variant="looser",
        total_snippets=120,
        facial_yes=58,
        facial_no=59,
        parse_failures=3,
        reach_full_eval=58,
        input_token_proxy_total=425_600,
    )


def _healthy_ternary() -> dict:
    return _variant_dict(
        variant="ternary",
        total_snippets=120,
        facial_yes=53,
        facial_no=49,
        facial_borderline=18,
        parse_failures=0,
        reach_full_eval=71,
        input_token_proxy_total=587_200,
    )


# ---------------------------------------------------------------------------
# 1-2. load_thresholds
# ---------------------------------------------------------------------------


def test_load_thresholds_happy_path(tmp_path):
    p = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    loaded = rec.load_thresholds(p)
    assert loaded["minimum_sample"]["total_snippets_min"] == 50
    assert loaded["parse_quality"]["max_parse_failure_rate"] == 0.05
    assert (
        loaded["recovery_quality"]["min_variant_only_recovered_saves"] == 1
    )
    assert loaded["cost_quality"]["max_token_proxy_ratio_vs_baseline"] == 3.0


def test_load_thresholds_missing_top_level_section(tmp_path):
    bad = _default_thresholds_dict()
    del bad["minimum_sample"]
    p = _write_thresholds(tmp_path / "bad.json", bad)
    with pytest.raises(ValueError) as exc_info:
        rec.load_thresholds(p)
    assert "minimum_sample" in str(exc_info.value)


def test_load_thresholds_missing_nested_key(tmp_path):
    bad = _default_thresholds_dict()
    del bad["recovery_quality"]["min_ternary_borderline_signal_ratio"]
    p = _write_thresholds(tmp_path / "bad.json", bad)
    with pytest.raises(ValueError) as exc_info:
        rec.load_thresholds(p)
    assert "min_ternary_borderline_signal_ratio" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3-5. load_summary
# ---------------------------------------------------------------------------


def test_load_summary_from_file(tmp_path):
    p = _write_summary_json(
        tmp_path / "s.json",
        _summary(
            baseline=_healthy_baseline(),
            looser=_healthy_looser(),
            looser_cmp=_comparison_dict(
                compared="looser", variant_only_recovered_saves=4
            ),
        ),
    )
    loaded = rec.load_summary(str(p))
    assert "variants" in loaded
    assert "baseline" in loaded["variants"]
    assert "looser" in loaded["variants"]


def test_load_summary_from_stdin(monkeypatch):
    summary = _summary(
        baseline=_healthy_baseline(),
        ternary=_healthy_ternary(),
        ternary_cmp=_comparison_dict(
            compared="ternary", variant_only_recovered_saves=5
        ),
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(summary)))
    loaded = rec.load_summary("-")
    assert "ternary" in loaded["variants"]


def test_load_summary_rejects_non_object_top_level(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        rec.load_summary(str(p))


def test_load_summary_rejects_missing_variants(tmp_path):
    p = _write_summary_json(tmp_path / "s.json", {"comparisons": {}})
    with pytest.raises(ValueError) as exc_info:
        rec.load_summary(str(p))
    assert "variants" in str(exc_info.value)


def test_load_summary_rejects_missing_baseline(tmp_path):
    p = _write_summary_json(
        tmp_path / "s.json",
        {"variants": {"looser": _healthy_looser()}, "comparisons": {}},
    )
    with pytest.raises(ValueError) as exc_info:
        rec.load_summary(str(p))
    assert "baseline" in str(exc_info.value)


def test_load_summary_rejects_baseline_only(tmp_path):
    """No non-baseline variant means nothing to recommend on."""
    p = _write_summary_json(
        tmp_path / "s.json",
        {"variants": {"baseline": _healthy_baseline()}, "comparisons": {}},
    )
    with pytest.raises(ValueError) as exc_info:
        rec.load_summary(str(p))
    assert "looser" in str(exc_info.value) or "ternary" in str(
        exc_info.value
    )


# ---------------------------------------------------------------------------
# 6. evaluate KEEP_BINARY: nobody clears the recovery floor
# ---------------------------------------------------------------------------


def test_evaluate_keep_binary_when_no_variant_clears_recovery_floor():
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=44,
            facial_no=74,
            parse_failures=2,
            reach_full_eval=44,
            input_token_proxy_total=425_600,
        ),
        ternary=_variant_dict(
            variant="ternary",
            total_snippets=120,
            facial_yes=42,
            facial_no=72,
            facial_borderline=4,
            parse_failures=2,
            reach_full_eval=46,
            input_token_proxy_total=587_200,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=0
        ),
        ternary_cmp=_comparison_dict(
            compared="ternary", variant_only_recovered_saves=0
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_KEEP_BINARY


# ---------------------------------------------------------------------------
# 7. evaluate TRY_LOOSER_BINARY
# ---------------------------------------------------------------------------


def test_evaluate_try_looser_binary_when_looser_clears_all_gates():
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=58,
            facial_no=59,
            parse_failures=2,  # 2/120 = 0.0167 (<= 0.05)
            reach_full_eval=58,  # +0.381 lift vs 42
            input_token_proxy_total=433_440,  # ratio 1.05
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=3
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_TRY_LOOSER_BINARY


# ---------------------------------------------------------------------------
# 8. evaluate EXPERIMENT_TERNARY_ONLY (looser absent)
# ---------------------------------------------------------------------------


def test_evaluate_experiment_ternary_only_when_only_ternary_qualifies():
    summary = _summary(
        baseline=_healthy_baseline(),
        ternary=_healthy_ternary(),
        ternary_cmp=_comparison_dict(
            compared="ternary", variant_only_recovered_saves=5
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_EXPERIMENT_TERNARY_ONLY


# ---------------------------------------------------------------------------
# 9. evaluate NOT_ENOUGH_DATA
# ---------------------------------------------------------------------------


def test_evaluate_not_enough_data_when_any_variant_below_floor():
    """Looser has total_snippets=10 (below floor=50) -> NOT_ENOUGH_DATA."""
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=10,
            facial_yes=5,
            parse_failures=0,
            reach_full_eval=5,
            input_token_proxy_total=35_000,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=0
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_NOT_ENOUGH_DATA


# ---------------------------------------------------------------------------
# 10. Sample dominance over regression
# ---------------------------------------------------------------------------


def test_sample_failure_dominates_parse_failure_breach():
    """Even with parse-failure breach + total_snippets below floor,
    verdict is NOT_ENOUGH_DATA (sample failure dominates)."""
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=10,
            facial_yes=5,
            parse_failures=5,  # 5/10 = 0.50 (would breach 0.05)
            reach_full_eval=5,
            input_token_proxy_total=35_000,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=2
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_NOT_ENOUGH_DATA


# ---------------------------------------------------------------------------
# 11. evaluate INVESTIGATE_REGRESSION (parse-failure breach)
# ---------------------------------------------------------------------------


def test_evaluate_investigate_regression_on_parse_failure_breach():
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=58,
            facial_no=50,
            parse_failures=12,  # 12/120 = 0.10 (> 0.05)
            reach_full_eval=58,
            input_token_proxy_total=425_600,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=4
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_INVESTIGATE_REGRESSION


# ---------------------------------------------------------------------------
# 12. Regression dominance over policy
# ---------------------------------------------------------------------------


def test_regression_dominates_strong_policy_recovery():
    """Parse-failure breach AND looser has strong recovery numbers ->
    INVESTIGATE_REGRESSION (regression dominates policy)."""
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=80,
            facial_no=28,
            parse_failures=12,  # breach
            reach_full_eval=80,  # huge lift
            input_token_proxy_total=425_600,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=10
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_INVESTIGATE_REGRESSION


# ---------------------------------------------------------------------------
# 13. evaluate INVESTIGATE_REGRESSION (false-negative regression)
# ---------------------------------------------------------------------------


def test_evaluate_investigate_regression_on_false_negative_regression():
    """Variant has likely_false_negatives_under_variant > 0 (baseline=0)
    -> INVESTIGATE_REGRESSION."""
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=58,
            facial_no=59,
            parse_failures=2,
            reach_full_eval=58,
            input_token_proxy_total=425_600,
        ),
        looser_cmp=_comparison_dict(
            compared="looser",
            variant_only_recovered_saves=3,
            likely_false_negatives_under_variant=2,
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_INVESTIGATE_REGRESSION


# ---------------------------------------------------------------------------
# 14. Baseline parse-failure breach is NOT a veto
# ---------------------------------------------------------------------------


def test_baseline_parse_failure_breach_is_not_a_veto():
    """Baseline parse_failure_rate=0.20 (catastrophic), but looser has
    healthy parse rate + strong recovery. Verdict: TRY_LOOSER_BINARY.
    Baseline is the reference, not a candidate for recommendation."""
    summary = _summary(
        baseline=_variant_dict(
            variant="baseline",
            total_snippets=120,
            facial_yes=42,
            facial_no=54,
            parse_failures=24,  # 24/120 = 0.20 — would breach
            reach_full_eval=42,
            input_token_proxy_total=412_800,
        ),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=58,
            facial_no=59,
            parse_failures=3,  # 0.025 — well under
            reach_full_eval=58,
            input_token_proxy_total=425_600,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=4
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_TRY_LOOSER_BINARY


# ---------------------------------------------------------------------------
# 15. Ternary borderline degenerate -> not EXPERIMENT_TERNARY_ONLY
# ---------------------------------------------------------------------------


def test_ternary_with_zero_borderline_cannot_be_experiment_ternary_only():
    """Ternary recovers saves and clears cost+lift, but facial_borderline=0
    (ratio 0/120 = 0.0 < 0.10). Verdict cannot be EXPERIMENT_TERNARY_ONLY.
    With no looser, falls to KEEP_BINARY."""
    summary = _summary(
        baseline=_healthy_baseline(),
        ternary=_variant_dict(
            variant="ternary",
            total_snippets=120,
            facial_yes=58,
            facial_no=62,
            facial_borderline=0,
            parse_failures=0,
            reach_full_eval=58,
            input_token_proxy_total=587_200,
        ),
        ternary_cmp=_comparison_dict(
            compared="ternary", variant_only_recovered_saves=4
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_KEEP_BINARY


# ---------------------------------------------------------------------------
# 16. Token-proxy gate rejects ternary
# ---------------------------------------------------------------------------


def test_ternary_token_proxy_breach_blocks_experiment_ternary_only():
    """Ternary token_proxy_ratio = 4.5 (exceeds 3.0). With no looser ->
    KEEP_BINARY."""
    summary = _summary(
        baseline=_healthy_baseline(),  # baseline tokens = 412800
        ternary=_variant_dict(
            variant="ternary",
            total_snippets=120,
            facial_yes=53,
            facial_no=49,
            facial_borderline=18,
            parse_failures=0,
            reach_full_eval=71,
            input_token_proxy_total=1_857_600,  # ratio 4.5
        ),
        ternary_cmp=_comparison_dict(
            compared="ternary", variant_only_recovered_saves=5
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_KEEP_BINARY


# ---------------------------------------------------------------------------
# 17. Both qualify -> TRY_LOOSER_BINARY wins
# ---------------------------------------------------------------------------


def test_both_qualify_prefers_try_looser_binary():
    """When both looser and ternary qualify, prefer TRY_LOOSER_BINARY.
    Rationale: looser is a string-edit to the production prompt with no
    parser/contract change; ternary requires changes to
    parse_facial_response, OpusDecision, and downstream consumers.
    Smallest safe slice wins."""
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_healthy_looser(),
        ternary=_healthy_ternary(),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=4
        ),
        ternary_cmp=_comparison_dict(
            compared="ternary", variant_only_recovered_saves=5
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_TRY_LOOSER_BINARY


# ---------------------------------------------------------------------------
# 18. Inclusive comparator boundary
# ---------------------------------------------------------------------------


def test_inclusive_comparator_boundary_passes():
    """Looser variant_only_recovered_saves=1 (== floor=1) AND
    reach_lift_ratio=0.10 (== floor=0.10) -> PASS, TRY_LOOSER_BINARY."""
    summary = _summary(
        baseline=_variant_dict(
            variant="baseline",
            total_snippets=100,
            facial_yes=40,
            facial_no=60,
            parse_failures=0,
            reach_full_eval=40,
            input_token_proxy_total=400_000,
        ),
        looser=_variant_dict(
            variant="looser",
            total_snippets=100,
            facial_yes=44,
            facial_no=56,
            parse_failures=0,
            reach_full_eval=44,  # +0.10 exactly
            input_token_proxy_total=400_000,  # ratio 1.0
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=1
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_TRY_LOOSER_BINARY
    inputs = report.inputs
    assert inputs["pairwise_vs_baseline"]["looser"][
        "reach_lift_ratio"
    ] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 19. Zero-denominator handling
# ---------------------------------------------------------------------------


def test_zero_baseline_reach_full_eval_no_division_error():
    """baseline.reach_full_eval=0 -> reach_lift_ratio=0.0, no
    ZeroDivisionError. Sample floor still passes."""
    summary = _summary(
        baseline=_variant_dict(
            variant="baseline",
            total_snippets=120,
            facial_yes=0,
            facial_no=120,
            parse_failures=0,
            reach_full_eval=0,
            input_token_proxy_total=412_800,
        ),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=10,
            facial_no=110,
            parse_failures=0,
            reach_full_eval=10,
            input_token_proxy_total=425_600,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=2
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert (
        report.inputs["pairwise_vs_baseline"]["looser"]["reach_lift_ratio"]
        == 0.0
    )
    # baseline reach=0, looser lift=0.0 < 0.10 floor -> can't qualify
    # (no false-negative regression, no parse breach) -> KEEP_BINARY.
    assert report.verdict == rec.VERDICT_KEEP_BINARY


def test_zero_baseline_token_proxy_no_division_error():
    summary = _summary(
        baseline=_variant_dict(
            variant="baseline",
            total_snippets=120,
            facial_yes=42,
            facial_no=78,
            parse_failures=0,
            reach_full_eval=42,
            input_token_proxy_total=0,
        ),
        looser=_healthy_looser(),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=4
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert (
        report.inputs["pairwise_vs_baseline"]["looser"]["token_proxy_ratio"]
        == 0.0
    )


# ---------------------------------------------------------------------------
# 20. format_report quiet mode
# ---------------------------------------------------------------------------


def test_format_report_quiet_returns_only_verdict():
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_healthy_looser(),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=4
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    out = rec.format_report(report, quiet=True)
    assert out == report.verdict
    assert "\n" not in out


# ---------------------------------------------------------------------------
# 21. format_report full mode covers every label rendering path
# ---------------------------------------------------------------------------


def test_format_report_full_keep_binary_renders_all_sections():
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=42,
            facial_no=76,
            parse_failures=2,
            reach_full_eval=42,
            input_token_proxy_total=425_600,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=0
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_KEEP_BINARY
    out = rec.format_report(report, quiet=False)
    assert "Facial-Gate Variant Recommendation" in out
    assert "Verdict: KEEP_BINARY" in out
    assert "Per-variant counters" in out
    assert "Pairwise vs baseline" in out
    assert "Sample-size checks" in out
    assert "Regression checks" in out
    assert "Policy checks" in out


def test_format_report_full_try_looser_renders():
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_healthy_looser(),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=4
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_TRY_LOOSER_BINARY
    out = rec.format_report(report, quiet=False)
    assert "Verdict: TRY_LOOSER_BINARY" in out


def test_format_report_full_experiment_ternary_only_renders():
    summary = _summary(
        baseline=_healthy_baseline(),
        ternary=_healthy_ternary(),
        ternary_cmp=_comparison_dict(
            compared="ternary", variant_only_recovered_saves=5
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_EXPERIMENT_TERNARY_ONLY
    out = rec.format_report(report, quiet=False)
    assert "Verdict: EXPERIMENT_TERNARY_ONLY" in out


def test_format_report_full_not_enough_data_marks_diagnostic_prefix():
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=10,
            facial_yes=5,
            parse_failures=0,
            reach_full_eval=5,
            input_token_proxy_total=35_000,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=0
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_NOT_ENOUGH_DATA
    out = rec.format_report(report, quiet=False)
    assert "[N/A — insufficient sample]" in out


def test_format_report_full_investigate_regression_marks_policy_diagnostic():
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=58,
            facial_no=50,
            parse_failures=12,  # breach
            reach_full_eval=58,
            input_token_proxy_total=425_600,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=4
        ),
    )
    report = rec.evaluate(
        summary=summary, thresholds=_default_thresholds_dict()
    )
    assert report.verdict == rec.VERDICT_INVESTIGATE_REGRESSION
    out = rec.format_report(report, quiet=False)
    assert "[N/A — regression detected]" in out


# ---------------------------------------------------------------------------
# 22. run end-to-end for each verdict
# ---------------------------------------------------------------------------


def test_run_end_to_end_keep_binary(capsys, tmp_path):
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=42,
            facial_no=76,
            parse_failures=2,
            reach_full_eval=42,
            input_token_proxy_total=425_600,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=0
        ),
    )
    s = _write_summary_json(tmp_path / "s.json", summary)
    t = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    rc = rec.run(_make_args(summary=str(s), thresholds=str(t)))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Verdict: KEEP_BINARY" in out


def test_run_end_to_end_try_looser_binary(capsys, tmp_path):
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_healthy_looser(),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=4
        ),
    )
    s = _write_summary_json(tmp_path / "s.json", summary)
    t = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    rc = rec.run(_make_args(summary=str(s), thresholds=str(t)))
    out = capsys.readouterr().out
    assert rc == 1
    assert "Verdict: TRY_LOOSER_BINARY" in out


def test_run_end_to_end_experiment_ternary_only(capsys, tmp_path):
    summary = _summary(
        baseline=_healthy_baseline(),
        ternary=_healthy_ternary(),
        ternary_cmp=_comparison_dict(
            compared="ternary", variant_only_recovered_saves=5
        ),
    )
    s = _write_summary_json(tmp_path / "s.json", summary)
    t = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    rc = rec.run(_make_args(summary=str(s), thresholds=str(t)))
    out = capsys.readouterr().out
    assert rc == 2
    assert "Verdict: EXPERIMENT_TERNARY_ONLY" in out


def test_run_end_to_end_not_enough_data(capsys, tmp_path):
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=10,
            facial_yes=5,
            parse_failures=0,
            reach_full_eval=5,
            input_token_proxy_total=35_000,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=0
        ),
    )
    s = _write_summary_json(tmp_path / "s.json", summary)
    t = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    rc = rec.run(_make_args(summary=str(s), thresholds=str(t)))
    out = capsys.readouterr().out
    assert rc == 3
    assert "Verdict: NOT_ENOUGH_DATA" in out


def test_run_end_to_end_investigate_regression(capsys, tmp_path):
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=58,
            facial_no=50,
            parse_failures=12,  # breach
            reach_full_eval=58,
            input_token_proxy_total=425_600,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=4
        ),
    )
    s = _write_summary_json(tmp_path / "s.json", summary)
    t = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    rc = rec.run(_make_args(summary=str(s), thresholds=str(t)))
    out = capsys.readouterr().out
    assert rc == 4
    assert "Verdict: INVESTIGATE_REGRESSION" in out


# ---------------------------------------------------------------------------
# 23. --report-out writes structured JSON
# ---------------------------------------------------------------------------


def test_run_writes_structured_report_out(capsys, tmp_path):
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_healthy_looser(),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=4
        ),
    )
    s = _write_summary_json(tmp_path / "s.json", summary)
    t = _write_thresholds(tmp_path / "t.json", _default_thresholds_dict())
    out_path = tmp_path / "report.json"
    args = _make_args(
        summary=str(s),
        thresholds=str(t),
        report_out=str(out_path),
    )
    rc = rec.run(args)
    capsys.readouterr()
    assert rc == 1
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "TRY_LOOSER_BINARY"
    assert "inputs" in payload
    assert "sample_checks" in payload
    assert "regression_checks" in payload
    assert "policy_checks" in payload
    assert payload["thresholds_path"] == str(t)
    assert payload["summary_path"] == str(s)
    sample_first = payload["sample_checks"][0]
    for key in (
        "name",
        "passed",
        "observed",
        "threshold",
        "comparator",
        "category",
    ):
        assert key in sample_first


# ---------------------------------------------------------------------------
# 24. --thresholds override flips INVESTIGATE_REGRESSION -> TRY_LOOSER_BINARY
# ---------------------------------------------------------------------------


def test_run_thresholds_override_flips_regression_to_try_looser(
    capsys, tmp_path
):
    summary = _summary(
        baseline=_healthy_baseline(),
        looser=_variant_dict(
            variant="looser",
            total_snippets=120,
            facial_yes=58,
            facial_no=50,
            parse_failures=12,  # breach under default
            reach_full_eval=58,
            input_token_proxy_total=425_600,
        ),
        looser_cmp=_comparison_dict(
            compared="looser", variant_only_recovered_saves=4
        ),
    )
    s = _write_summary_json(tmp_path / "s.json", summary)
    default_t = _write_thresholds(
        tmp_path / "default.json", _default_thresholds_dict()
    )

    rc_default = rec.run(
        _make_args(summary=str(s), thresholds=str(default_t))
    )
    capsys.readouterr()
    assert rc_default == 4

    looser_thresholds = _default_thresholds_dict()
    looser_thresholds["parse_quality"]["max_parse_failure_rate"] = 0.50
    override_t = _write_thresholds(
        tmp_path / "override.json", looser_thresholds
    )
    rc_override = rec.run(
        _make_args(summary=str(s), thresholds=str(override_t))
    )
    out = capsys.readouterr().out
    assert rc_override == 1
    assert "Verdict: TRY_LOOSER_BINARY" in out


# ---------------------------------------------------------------------------
# 25. The shipped default thresholds file loads cleanly
# ---------------------------------------------------------------------------


def test_default_thresholds_file_loads_cleanly():
    """Guards the shipped config/facial-gate-recommendation-thresholds.json
    against drifting out of step with the loader's required-key schema."""
    loaded = rec.load_thresholds(rec.DEFAULT_THRESHOLDS_PATH)
    assert "minimum_sample" in loaded
    assert "parse_quality" in loaded
    assert "recovery_quality" in loaded
    assert "cost_quality" in loaded
    assert isinstance(loaded["minimum_sample"]["total_snippets_min"], int)
    assert isinstance(
        loaded["parse_quality"]["max_parse_failure_rate"], (int, float)
    )
    assert isinstance(
        loaded["recovery_quality"]["min_variant_only_recovered_saves"],
        int,
    )
    assert isinstance(
        loaded["recovery_quality"]["min_reach_full_eval_lift_ratio"],
        (int, float),
    )
    assert isinstance(
        loaded["recovery_quality"]["min_ternary_borderline_signal_ratio"],
        (int, float),
    )
    assert isinstance(
        loaded["cost_quality"]["max_token_proxy_ratio_vs_baseline"],
        (int, float),
    )
