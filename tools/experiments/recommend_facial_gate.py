#!/usr/bin/env python3
"""Recommendation interpreter for the facial-gate variant comparison harness.

Slice 10 of perplexity-evidence-augmentation. This is the **interpreter** of
slice 5's harness output (``tools/experiments/facial_gate_experiment.py
--json-out``). It is opt-in (consumes a ``--summary`` JSON path),
threshold-driven (committed defaults at
``config/facial-gate-recommendation-thresholds.json``,
``--thresholds <path>`` overrides), and it does **NOT move the production
gate**. The verdict is a checkable artifact for operator review, not an
automation switch.

The deliverable answers "should the gate move?" with a labeled,
threshold-driven recommendation. It does not move the gate; it produces a
verdict an operator can act on.

Hard guarantees:

- No live LLM or network calls.
- No imports from ``linkedin/``, ``github/``, ``market_intelligence/``, or
  any production module. Stdlib only.
- No edits to ``shared/judgment/templates.py:FACIAL_TRIAGE_TEMPLATE``,
  ``parse_facial_response``, ``parse_facial_batch_response``, or
  ``shared/judger.py``.
- No edits to the brief schema. Recommendation thresholds are
  operator-tunable knobs over offline data, not evaluation criteria.
- Default invocation writes nothing to disk. ``--report-out <path>`` opts in
  to writing a structured report to a path the operator names.
- ``--summary -`` reads a single JSON object from stdin. JSON-lines is not
  supported on this surface (the upstream is the harness's ``--json-out``,
  which writes one object).

Verdict labels and exit codes (distinct from slice 7's 0/1/2 so the two
evaluators can be wired into the same shell pipeline without conflicting):

- ``KEEP_BINARY``              -> exit 0
- ``TRY_LOOSER_BINARY``        -> exit 1
- ``EXPERIMENT_TERNARY_ONLY``  -> exit 2
- ``NOT_ENOUGH_DATA``          -> exit 3
- ``INVESTIGATE_REGRESSION``   -> exit 4

Verdict precedence (sample failure dominates regression dominates policy):

1. If ``total_snippets < total_snippets_min`` for ANY variant in
   ``summary["variants"]`` -> ``NOT_ENOUGH_DATA``. Other check categories
   still run for diagnostics but do not drive the verdict.
2. Else if any non-baseline variant fails ``max_parse_failure_rate`` ->
   ``INVESTIGATE_REGRESSION``. Baseline's parse rate is reported but is
   not gated against -- baseline is the reference, not a candidate.
3. Else if any non-baseline variant has
   ``likely_false_negatives_under_variant > 0`` (baseline is treated as 0
   because the harness's ``analyze_recovery`` only emits this field for
   non-baseline comparisons) -> ``INVESTIGATE_REGRESSION``. This is a
   regression direction veto even if the absolute number is uncalibrated.
4. Else if ``looser`` is present AND looser cleared the recovery floor AND
   looser's reach-lift-ratio AND looser's token-proxy-ratio gates pass ->
   ``TRY_LOOSER_BINARY``.
5. Else if ``ternary`` is present AND ternary cleared the recovery floor
   AND ternary's reach-lift-ratio AND ternary's token-proxy-ratio gates
   pass AND ternary's borderline-signal-ratio gate passes ->
   ``EXPERIMENT_TERNARY_ONLY``.
6. Else -> ``KEEP_BINARY``.

If both looser and ternary qualify, ``TRY_LOOSER_BINARY`` wins. Looser is a
string-edit to the production prompt with no parser/contract change;
ternary requires changes to ``parse_facial_response``, ``OpusDecision``,
and downstream consumers. Smallest safe slice wins.

Numerical contract (pinned by tests, mirrors slice 7):

- ``parse_failure_rate(variant)        = variant.parse_failures / variant.total_snippets``
- ``reach_lift_ratio(variant, baseline) = (variant.reach_full_eval - baseline.reach_full_eval) / baseline.reach_full_eval``
- ``token_proxy_ratio(variant, baseline) = variant.input_token_proxy_total / baseline.input_token_proxy_total``
- ``borderline_signal_ratio(variant)   = variant.facial_borderline / variant.total_snippets``
- Comparators are inclusive (``<=`` / ``>=``).
- Zero-denominator handling: rates and ratios are ``0.0``; sample-floor
  gates already fail in that case so the verdict will be
  ``NOT_ENOUGH_DATA`` regardless.

CLI shape::

    python tools/experiments/recommend_facial_gate.py
        --summary <path|->
        [--thresholds <path>]
        [--report-out PATH]
        [--quiet]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_THRESHOLDS_PATH = (
    PROJECT_ROOT / "config" / "facial-gate-recommendation-thresholds.json"
)


VERDICT_KEEP_BINARY = "KEEP_BINARY"
VERDICT_TRY_LOOSER_BINARY = "TRY_LOOSER_BINARY"
VERDICT_EXPERIMENT_TERNARY_ONLY = "EXPERIMENT_TERNARY_ONLY"
VERDICT_NOT_ENOUGH_DATA = "NOT_ENOUGH_DATA"
VERDICT_INVESTIGATE_REGRESSION = "INVESTIGATE_REGRESSION"


_EXIT_CODES = {
    VERDICT_KEEP_BINARY: 0,
    VERDICT_TRY_LOOSER_BINARY: 1,
    VERDICT_EXPERIMENT_TERNARY_ONLY: 2,
    VERDICT_NOT_ENOUGH_DATA: 3,
    VERDICT_INVESTIGATE_REGRESSION: 4,
}


# Required nested keys in the threshold spec. Keep this list in lock-step
# with config/facial-gate-recommendation-thresholds.json.
_REQUIRED_THRESHOLD_KEYS: dict[str, tuple[str, ...]] = {
    "minimum_sample": ("total_snippets_min",),
    "parse_quality": ("max_parse_failure_rate",),
    "recovery_quality": (
        "min_variant_only_recovered_saves",
        "min_reach_full_eval_lift_ratio",
        "min_ternary_borderline_signal_ratio",
    ),
    "cost_quality": ("max_token_proxy_ratio_vs_baseline",),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """Result of a single threshold check.

    ``observed`` and ``threshold`` are intentionally typed as the raw
    numeric (int for counts, float for rates/ratios) -- the caller decides
    how to format them.
    """

    name: str
    passed: bool
    observed: float
    threshold: float
    comparator: str
    category: str  # "sample" | "regression" | "policy"


@dataclass
class RecommendationReport:
    verdict: str
    sample_checks: list[CheckResult] = field(default_factory=list)
    regression_checks: list[CheckResult] = field(default_factory=list)
    policy_checks: list[CheckResult] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_thresholds(path: Path) -> dict:
    """Load and validate the threshold spec.

    Validates that the four top-level required sections
    (``minimum_sample``, ``parse_quality``, ``recovery_quality``,
    ``cost_quality``) and each section's required nested keys are present.
    Raises ``ValueError`` with a clear message if any are missing.

    Does NOT type-check values beyond JSON parsing. Operators are expected
    to keep the values numeric -- the evaluator will surface a ``TypeError``
    at compare time if they don't, which is the right failure mode for a
    misconfigured operator-controlled file.
    """

    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"thresholds file {path} must be a JSON object at the top level"
        )

    missing_top = [k for k in _REQUIRED_THRESHOLD_KEYS if k not in data]
    if missing_top:
        raise ValueError(
            f"thresholds file {path} is missing required top-level "
            f"section(s): {missing_top}"
        )

    for section, required in _REQUIRED_THRESHOLD_KEYS.items():
        section_value = data.get(section)
        if not isinstance(section_value, dict):
            raise ValueError(
                f"thresholds file {path}: section {section!r} must be a "
                f"JSON object"
            )
        missing_nested = [k for k in required if k not in section_value]
        if missing_nested:
            raise ValueError(
                f"thresholds file {path}: section {section!r} is missing "
                f"required key(s): {missing_nested}"
            )

    return data


def load_summary(path_or_stdin: str) -> dict:
    """Load the harness summary JSON from a file path or ``-`` (stdin).

    Accepts a single JSON object only. JSON-lines input is not supported
    here -- the upstream is ``facial_gate_experiment.py --json-out``, which
    writes one object.

    Validates that the loaded dict has ``variants`` with ``baseline`` plus
    at least one of ``looser`` / ``ternary``. The harness emits per-variant
    counters under the top-level key ``"variants"`` (see
    ``tools/experiments/facial_gate_experiment.py:run`` near the
    ``write_summary_json`` call).

    Raises ``FileNotFoundError`` if the named path doesn't exist and
    ``json.JSONDecodeError`` on bad JSON. Raises ``ValueError`` if the
    payload isn't a top-level object or the shape is unrecognized.
    """

    if path_or_stdin == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path_or_stdin).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(
            "summary input must be a JSON object at the top level "
            "(JSON-lines is not supported)"
        )

    variants = data.get("variants")
    if not isinstance(variants, dict):
        raise ValueError(
            "summary input is missing required top-level key 'variants' "
            "or it is not a JSON object; expected the shape produced by "
            "tools/experiments/facial_gate_experiment.py --json-out"
        )
    if "baseline" not in variants:
        raise ValueError(
            "summary['variants'] must include a 'baseline' entry "
            "(baseline is the reference for all pairwise gates); "
            "re-run the harness with --variants baseline ..."
        )
    if not any(k in variants for k in ("looser", "ternary")):
        raise ValueError(
            "summary['variants'] must include at least one non-baseline "
            "variant ('looser' or 'ternary'); without one there is "
            "nothing to recommend on"
        )
    return data


# ---------------------------------------------------------------------------
# Pure evaluation
# ---------------------------------------------------------------------------


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _parse_failure_rate(variant: dict) -> float:
    return _safe_rate(
        variant.get("parse_failures", 0), variant.get("total_snippets", 0)
    )


def _borderline_signal_ratio(variant: dict) -> float:
    return _safe_rate(
        variant.get("facial_borderline", 0), variant.get("total_snippets", 0)
    )


def _reach_lift_ratio(variant: dict, baseline: dict) -> float:
    base_reach = baseline.get("reach_full_eval", 0)
    if base_reach <= 0:
        return 0.0
    return (
        float(variant.get("reach_full_eval", 0)) - float(base_reach)
    ) / float(base_reach)


def _token_proxy_ratio(variant: dict, baseline: dict) -> float:
    base_tokens = baseline.get("input_token_proxy_total", 0)
    if base_tokens <= 0:
        return 0.0
    return float(variant.get("input_token_proxy_total", 0)) / float(
        base_tokens
    )


def _likely_false_negatives_for(
    summary: dict, variant_name: str
) -> int:
    """Pull the per-comparison ``likely_false_negatives_under_variant`` count.

    The harness's ``analyze_recovery`` emits this field only on pairwise
    comparisons in ``summary["comparisons"]``, never on baseline itself.
    Baseline is treated as 0 (it is the reference for all comparisons).
    Missing / malformed entries also return 0; the report will surface this
    as a ``[DIAGNOSTIC]`` line.
    """

    comparisons = summary.get("comparisons") or {}
    if not isinstance(comparisons, dict):
        return 0
    cmp = comparisons.get(variant_name)
    if not isinstance(cmp, dict):
        return 0
    val = cmp.get("likely_false_negatives_under_variant", 0)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def evaluate(
    *, summary: dict, thresholds: dict
) -> RecommendationReport:
    """Evaluate a harness summary against the recommendation thresholds.

    Pure: no I/O, no clocks, no randomness. Same input -> same output.

    Computes all three check categories regardless of which one drives the
    verdict so operators see why the verdict was what it was.
    """

    variants = summary.get("variants") or {}
    baseline = variants.get("baseline") or {}
    looser = variants.get("looser")
    ternary = variants.get("ternary")

    sample_thresholds = thresholds["minimum_sample"]
    parse_thresholds = thresholds["parse_quality"]
    recovery_thresholds = thresholds["recovery_quality"]
    cost_thresholds = thresholds["cost_quality"]

    total_snippets_min = int(sample_thresholds["total_snippets_min"])
    max_parse_failure_rate = float(parse_thresholds["max_parse_failure_rate"])
    min_recovered = int(
        recovery_thresholds["min_variant_only_recovered_saves"]
    )
    min_reach_lift = float(
        recovery_thresholds["min_reach_full_eval_lift_ratio"]
    )
    min_borderline_signal = float(
        recovery_thresholds["min_ternary_borderline_signal_ratio"]
    )
    max_token_ratio = float(
        cost_thresholds["max_token_proxy_ratio_vs_baseline"]
    )

    # --- Sample-size checks -----------------------------------------------
    sample_checks: list[CheckResult] = []
    for name in ("baseline", "looser", "ternary"):
        v = variants.get(name)
        if v is None:
            continue
        observed = int(v.get("total_snippets", 0))
        sample_checks.append(
            CheckResult(
                name=f"{name}.total_snippets >= {total_snippets_min}",
                passed=observed >= total_snippets_min,
                observed=observed,
                threshold=total_snippets_min,
                comparator=">=",
                category="sample",
            )
        )

    # --- Regression checks (parse-failure veto + false-negative regression).
    regression_checks: list[CheckResult] = []
    for name in ("looser", "ternary"):
        v = variants.get(name)
        if v is None:
            continue
        rate = _parse_failure_rate(v)
        regression_checks.append(
            CheckResult(
                name=f"{name}.parse_failure_rate <= {max_parse_failure_rate:.3f}",
                passed=rate <= max_parse_failure_rate,
                observed=rate,
                threshold=max_parse_failure_rate,
                comparator="<=",
                category="regression",
            )
        )
        # Baseline is treated as 0 because the harness only emits
        # likely_false_negatives_under_variant on non-baseline comparisons.
        observed_fn = _likely_false_negatives_for(summary, name)
        regression_checks.append(
            CheckResult(
                name=f"{name}.likely_false_negatives_under_variant <= baseline (0)",
                passed=observed_fn <= 0,
                observed=observed_fn,
                threshold=0,
                comparator="<=",
                category="regression",
            )
        )

    # --- Policy checks (recovery + cost) ---------------------------------
    policy_checks: list[CheckResult] = []
    for name in ("looser", "ternary"):
        v = variants.get(name)
        if v is None:
            continue
        comparisons = summary.get("comparisons") or {}
        cmp = comparisons.get(name) or {}
        recovered = int(cmp.get("variant_only_recovered_saves", 0))
        policy_checks.append(
            CheckResult(
                name=f"{name}.variant_only_recovered_saves >= {min_recovered}",
                passed=recovered >= min_recovered,
                observed=recovered,
                threshold=min_recovered,
                comparator=">=",
                category="policy",
            )
        )
        lift = _reach_lift_ratio(v, baseline)
        policy_checks.append(
            CheckResult(
                name=f"{name}.reach_lift_ratio >= {min_reach_lift:.3f}",
                passed=lift >= min_reach_lift,
                observed=lift,
                threshold=min_reach_lift,
                comparator=">=",
                category="policy",
            )
        )
        token_ratio = _token_proxy_ratio(v, baseline)
        policy_checks.append(
            CheckResult(
                name=f"{name}.token_proxy_ratio <= {max_token_ratio:.3f}",
                passed=token_ratio <= max_token_ratio,
                observed=token_ratio,
                threshold=max_token_ratio,
                comparator="<=",
                category="policy",
            )
        )
        if name == "ternary":
            ratio = _borderline_signal_ratio(v)
            policy_checks.append(
                CheckResult(
                    name=(
                        f"ternary.borderline_signal_ratio "
                        f">= {min_borderline_signal:.3f}"
                    ),
                    passed=ratio >= min_borderline_signal,
                    observed=ratio,
                    threshold=min_borderline_signal,
                    comparator=">=",
                    category="policy",
                )
            )

    flat_thresholds = {
        "total_snippets_min": total_snippets_min,
        "max_parse_failure_rate": max_parse_failure_rate,
        "min_variant_only_recovered_saves": min_recovered,
        "min_reach_full_eval_lift_ratio": min_reach_lift,
        "min_ternary_borderline_signal_ratio": min_borderline_signal,
        "max_token_proxy_ratio_vs_baseline": max_token_ratio,
    }

    verdict = _decide_verdict(
        variants=variants,
        baseline=baseline,
        looser=looser,
        ternary=ternary,
        summary=summary,
        thresholds=flat_thresholds,
        sample_checks=sample_checks,
        regression_checks=regression_checks,
    )

    inputs = _build_inputs(
        variants=variants,
        baseline=baseline,
        looser=looser,
        ternary=ternary,
        summary=summary,
        thresholds=flat_thresholds,
    )

    return RecommendationReport(
        verdict=verdict,
        sample_checks=sample_checks,
        regression_checks=regression_checks,
        policy_checks=policy_checks,
        inputs=inputs,
    )


def _decide_verdict(
    *,
    variants: dict,
    baseline: dict,
    looser: dict | None,
    ternary: dict | None,
    summary: dict,
    thresholds: dict,
    sample_checks: list[CheckResult],
    regression_checks: list[CheckResult],
) -> str:
    if any(not c.passed for c in sample_checks):
        return VERDICT_NOT_ENOUGH_DATA
    if any(not c.passed for c in regression_checks):
        return VERDICT_INVESTIGATE_REGRESSION

    min_recovered = int(
        thresholds["min_variant_only_recovered_saves"]
    )
    min_reach_lift = float(thresholds["min_reach_full_eval_lift_ratio"])
    max_token_ratio = float(thresholds["max_token_proxy_ratio_vs_baseline"])
    min_borderline_signal = float(
        thresholds["min_ternary_borderline_signal_ratio"]
    )
    comparisons = summary.get("comparisons") or {}

    looser_qualifies = False
    if looser is not None:
        cmp = comparisons.get("looser") or {}
        recovered = int(cmp.get("variant_only_recovered_saves", 0))
        lift = _reach_lift_ratio(looser, baseline)
        token_ratio = _token_proxy_ratio(looser, baseline)
        looser_qualifies = (
            recovered >= min_recovered
            and lift >= min_reach_lift
            and token_ratio <= max_token_ratio
        )

    ternary_qualifies = False
    if ternary is not None:
        cmp = comparisons.get("ternary") or {}
        recovered = int(cmp.get("variant_only_recovered_saves", 0))
        lift = _reach_lift_ratio(ternary, baseline)
        token_ratio = _token_proxy_ratio(ternary, baseline)
        borderline_ratio = _borderline_signal_ratio(ternary)
        ternary_qualifies = (
            recovered >= min_recovered
            and lift >= min_reach_lift
            and token_ratio <= max_token_ratio
            and borderline_ratio >= min_borderline_signal
        )

    # Looser dominates ternary if both qualify: looser is a smaller
    # production change (string-edit to FACIAL_TRIAGE_TEMPLATE) than
    # ternary (parser + OpusDecision contract change).
    if looser_qualifies:
        return VERDICT_TRY_LOOSER_BINARY
    if ternary_qualifies:
        return VERDICT_EXPERIMENT_TERNARY_ONLY
    return VERDICT_KEEP_BINARY


def _build_inputs(
    *,
    variants: dict,
    baseline: dict,
    looser: dict | None,
    ternary: dict | None,
    summary: dict,
    thresholds: dict,
) -> dict:
    """Build the diagnostic ``inputs`` dict shipped with the report."""

    comparisons = summary.get("comparisons") or {}

    def _per_variant(name: str, v: dict) -> dict:
        return {
            "total_snippets": int(v.get("total_snippets", 0)),
            "facial_yes": int(v.get("facial_yes", 0)),
            "facial_no": int(v.get("facial_no", 0)),
            "facial_borderline": int(v.get("facial_borderline", 0)),
            "parse_failures": int(v.get("parse_failures", 0)),
            "reach_full_eval": int(v.get("reach_full_eval", 0)),
            "input_token_proxy_total": int(
                v.get("input_token_proxy_total", 0)
            ),
            "parse_failure_rate": _parse_failure_rate(v),
            "borderline_signal_ratio": _borderline_signal_ratio(v),
        }

    per_variant: dict[str, dict] = {}
    for name in ("baseline", "looser", "ternary"):
        v = variants.get(name)
        if v is None:
            continue
        per_variant[name] = _per_variant(name, v)

    pairwise: dict[str, dict] = {}
    for name in ("looser", "ternary"):
        v = variants.get(name)
        if v is None:
            continue
        cmp = comparisons.get(name) or {}
        pairwise[name] = {
            "variant_only_recovered_saves": int(
                cmp.get("variant_only_recovered_saves", 0)
            ),
            "likely_false_negatives_under_variant": _likely_false_negatives_for(
                summary, name
            ),
            "reach_lift_ratio": _reach_lift_ratio(v, baseline),
            "token_proxy_ratio": _token_proxy_ratio(v, baseline),
        }

    return {
        "per_variant": per_variant,
        "pairwise_vs_baseline": pairwise,
        "thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------


def _fmt_rate(value: float) -> str:
    return f"{value:.3f}"


def _fmt_signed_ratio(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.3f}"


def _fmt_check_line(
    check: CheckResult, *, suppression: str | None
) -> str:
    status = "PASS" if check.passed else "FAIL"
    if suppression:
        prefix = f"[N/A — {suppression}]"
    elif check.category == "policy" and check.name.startswith(
        "ternary.borderline_signal_ratio"
    ):
        # Borderline signal-ratio is a precondition for the ternary policy
        # path, not a global gate; render it as diagnostic when it can't
        # change the verdict.
        prefix = f"[{status}]"
    else:
        prefix = f"[{status}]"
    if isinstance(check.threshold, float) and not float(
        check.threshold
    ).is_integer():
        threshold_disp = f"{check.threshold:.3f}"
    elif check.category == "sample":
        threshold_disp = f"{int(check.threshold)}"
    else:
        threshold_disp = f"{check.threshold:.3f}"
    if check.category == "sample":
        observed_disp = f"{int(check.observed)}"
    elif (
        check.category == "regression"
        and check.name.endswith("baseline (0)")
    ):
        observed_disp = f"{int(check.observed)}"
    elif check.category == "policy" and check.name.endswith(
        f">= {int(check.threshold)}"
    ) and float(check.threshold).is_integer():
        observed_disp = f"{int(check.observed)}"
    else:
        observed_disp = _fmt_rate(check.observed)
    return f"  {prefix} {check.name:<60} : {observed_disp}"


def format_report(
    report: RecommendationReport, *, quiet: bool
) -> str:
    """Render the report for stdout.

    With ``quiet=True``, returns just the verdict on a single line. With
    ``quiet=False``, returns the full multi-section human report.
    """

    if quiet:
        return report.verdict

    inputs = report.inputs
    per_variant = inputs.get("per_variant") or {}
    pairwise = inputs.get("pairwise_vs_baseline") or {}

    lines: list[str] = []
    lines.append("=== Facial-Gate Variant Recommendation ===")
    lines.append(f"Verdict: {report.verdict}")
    lines.append("")

    lines.append("Per-variant counters:")
    for name in ("baseline", "looser", "ternary"):
        pv = per_variant.get(name)
        if pv is None:
            continue
        lines.append(
            f"  {name:<10}: total_snippets={pv['total_snippets']}  "
            f"reach_full_eval={pv['reach_full_eval']}  "
            f"facial_borderline={pv['facial_borderline']}  "
            f"parse_failures={pv['parse_failures']}  "
            f"input_token_proxy_total={pv['input_token_proxy_total']}"
        )
    lines.append("")

    lines.append("Pairwise vs baseline:")
    if not pairwise:
        lines.append("  (no non-baseline variants in summary)")
    else:
        for name in ("looser", "ternary"):
            p = pairwise.get(name)
            if p is None:
                continue
            lines.append(
                f"  {name:<10}: "
                f"variant_only_recovered_saves={p['variant_only_recovered_saves']}  "
                f"likely_false_negatives_under_variant={p['likely_false_negatives_under_variant']}  "
                f"reach_lift_ratio={_fmt_signed_ratio(p['reach_lift_ratio'])}  "
                f"token_proxy_ratio={p['token_proxy_ratio']:.3f}"
            )
    lines.append("")

    lines.append("Sample-size checks:")
    for check in report.sample_checks:
        lines.append(_fmt_check_line(check, suppression=None))
    lines.append("")

    if report.verdict == VERDICT_NOT_ENOUGH_DATA:
        regression_suppression = "insufficient sample"
        policy_suppression = "insufficient sample"
    elif report.verdict == VERDICT_INVESTIGATE_REGRESSION:
        regression_suppression = None
        policy_suppression = "regression detected"
    else:
        regression_suppression = None
        policy_suppression = None

    lines.append("Regression checks:")
    for check in report.regression_checks:
        lines.append(
            _fmt_check_line(check, suppression=regression_suppression)
        )
    lines.append("")

    lines.append("Policy checks:")
    for check in report.policy_checks:
        lines.append(_fmt_check_line(check, suppression=policy_suppression))

    # Diagnostic note: false-negative regression check uses the harness's
    # per-comparison field; baseline is treated as 0 since the harness
    # never emits this field on baseline (analyze_recovery is pairwise).
    lines.append("")
    lines.append(
        "Note: likely_false_negatives_under_variant is sourced from the "
        "harness's pairwise 'comparisons' block; baseline is treated as 0 "
        "(analyze_recovery emits this field only on non-baseline pairs). "
        "Calibration depends on should_request_external_evidence; "
        "advisory in v1."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON report writer (atomic; mirrors slice 7)
# ---------------------------------------------------------------------------


def _check_to_dict(check: CheckResult) -> dict:
    d = asdict(check)
    d["passed"] = bool(d["passed"])
    return d


def write_report_json(
    report: RecommendationReport,
    path: Path,
    *,
    thresholds_path: str,
    summary_path: str,
) -> None:
    """Atomically write the structured report to ``path``.

    Mirrors ``tools/evaluate_shadow_thresholds.py:write_report_json``.
    Writes to a sibling temp file then ``os.replace`` to the final path so
    a crash mid-write cannot leave a half-written report on disk.
    """

    try:
        path = Path(path)
        if path.exists() and path.is_dir():
            print(
                f"ERROR: --report-out path is a directory: {path}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        parent = path.parent if path.parent != Path("") else Path(".")
        parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "verdict": report.verdict,
            "inputs": report.inputs,
            "sample_checks": [
                _check_to_dict(c) for c in report.sample_checks
            ],
            "regression_checks": [
                _check_to_dict(c) for c in report.regression_checks
            ],
            "policy_checks": [
                _check_to_dict(c) for c in report.policy_checks
            ],
            "thresholds_path": thresholds_path,
            "summary_path": summary_path,
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"ERROR: failed to write --report-out {path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Read inputs, evaluate, print, optionally write JSON, return exit code."""

    thresholds_path = (
        Path(args.thresholds) if args.thresholds else DEFAULT_THRESHOLDS_PATH
    )
    try:
        thresholds = load_thresholds(thresholds_path)
    except FileNotFoundError as exc:
        print(
            f"ERROR: thresholds file not found: {thresholds_path}: {exc}",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(f"ERROR: invalid thresholds file: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: thresholds file is not valid JSON: "
            f"{thresholds_path}: {exc}",
            file=sys.stderr,
        )
        return 1

    summary_arg = args.summary
    try:
        summary = load_summary(summary_arg)
    except FileNotFoundError as exc:
        print(
            f"ERROR: summary file not found: {summary_arg}: {exc}",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(f"ERROR: invalid summary input: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: summary input is not valid JSON: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        report = evaluate(summary=summary, thresholds=thresholds)
    except KeyError as exc:
        print(
            f"ERROR: summary is missing required key: {exc}",
            file=sys.stderr,
        )
        return 1

    print(format_report(report, quiet=bool(args.quiet)))

    if args.report_out:
        write_report_json(
            report,
            Path(args.report_out),
            thresholds_path=str(thresholds_path),
            summary_path=str(summary_arg),
        )

    return _EXIT_CODES[report.verdict]


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recommendation interpreter for the facial-gate variant "
            "comparison harness. Produces KEEP_BINARY / TRY_LOOSER_BINARY "
            "/ EXPERIMENT_TERNARY_ONLY / NOT_ENOUGH_DATA / "
            "INVESTIGATE_REGRESSION. Analytical/debug only; the verdict "
            "does NOT move the production facial gate."
        )
    )
    parser.add_argument(
        "--summary",
        required=True,
        help=(
            "Path to the harness's --json-out file, or '-' to read a "
            "single JSON object from stdin. JSON-lines is not supported."
        ),
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help=(
            "Optional path to a threshold-spec JSON. Defaults to "
            "config/facial-gate-recommendation-thresholds.json relative "
            "to the repo root."
        ),
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help=(
            "Optional path to write the structured recommendation report "
            "as JSON. Default: nothing is written to disk."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Print only the verdict line on stdout. Default: full "
            "human-readable check-by-check report."
        ),
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
