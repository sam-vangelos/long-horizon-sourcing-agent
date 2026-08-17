#!/usr/bin/env python3
"""Acceptance-threshold evaluator for shadow-bakeoff aggregator summaries.

Slice 7 of perplexity-evidence-augmentation. This is an *analytical / debug*
tool. It consumes the JSON summary produced by
``tools/aggregate_shadow_judgments.py --json-out PATH`` and a small
threshold-spec JSON (default: ``config/perplexity-bakeoff-thresholds.json``)
and emits a deterministic ``PASS`` / ``FAIL`` / ``INSUFFICIENT_SAMPLE``
verdict for the bakeoff.

This tool **mechanically evaluates** whether a shadow bakeoff result clears
explicit, committed acceptance thresholds. It does NOT decide cutover policy
-- cutover decisions remain a human-in-the-loop call. The verdict is a
checkable artifact, not a release switch.

Hard guarantees:

- No live LLM or network calls.
- No imports from ``linkedin/``, ``github/``, ``market_intelligence/``, or
  ``shared/``. Stdlib only.
- Reads a single JSON file (or stdin) and the threshold spec; never writes
  under ``output/``, ``runtime_state.sqlite3``, ``final_judgments.jsonl``,
  ``shadow_final_judgments.jsonl``, canonical projections, or any per-run
  state directory.
- Default invocation writes nothing to disk. ``--report-out <path>`` opts in
  to writing a structured pass/fail report to a path the user names.
- ``--summary -`` reads a single JSON object from stdin. JSON-lines input
  is NOT supported on this surface (the upstream is the aggregator's
  ``--json-out``, which writes one object).

Numerical contract (pinned by tests):

- ``save_to_reject_rate     = save_to_reject / total_compared``
- ``reject_to_save_rate     = reject_to_save / total_compared``
- ``unavailable_evidence_rate = unavailable_external_evidence / total_rows_read``
- ``weak_citation_rate      = weak_citations / total_rows_read``
- Comparators are inclusive (``<=`` / ``>=``): a rate exactly at the
  threshold passes.
- Zero-denominator handling: if ``total_compared == 0`` or
  ``total_rows_read == 0`` the corresponding rates are ``0.0`` (no
  ``ZeroDivisionError``). Sample-size gates already failed in that case, so
  the verdict will be ``INSUFFICIENT_SAMPLE`` regardless.

Verdict precedence:

1. If any sample-size check fails -> ``INSUFFICIENT_SAMPLE`` (sample
   failure dominates: a run with too few rows but excellent rates is
   ``INSUFFICIENT_SAMPLE``, not ``PASS``).
2. Else if any quality check fails -> ``FAIL``.
3. Else -> ``PASS``.

Exit codes:

- ``0`` -> ``PASS``
- ``1`` -> ``FAIL``
- ``2`` -> ``INSUFFICIENT_SAMPLE``

CLI shape::

    python tools/evaluate_shadow_thresholds.py --summary <path|->
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "config" / "perplexity-bakeoff-thresholds.json"


VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"


_EXIT_CODES = {
    VERDICT_PASS: 0,
    VERDICT_FAIL: 1,
    VERDICT_INSUFFICIENT_SAMPLE: 2,
}


# Required nested keys in the threshold spec. Keep this list in lock-step
# with config/perplexity-bakeoff-thresholds.json.
_REQUIRED_THRESHOLD_KEYS: dict[str, tuple[str, ...]] = {
    "minimum_sample": ("total_rows_read_min", "total_compared_min"),
    "decision_quality": (
        "max_save_to_reject_rate",
        "min_reject_to_save_lift",
    ),
    "evidence_quality": (
        "max_unavailable_evidence_rate",
        "max_weak_citation_rate",
    ),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """Result of a single threshold check.

    ``observed`` and ``threshold`` are intentionally typed as the raw
    numeric (int for counts, float for rates) -- the caller decides how to
    format them.
    """

    name: str
    passed: bool
    observed: float
    threshold: float
    comparator: str
    category: str  # "sample" | "decision_quality" | "evidence_quality"


@dataclass
class EvaluationReport:
    verdict: str
    sample_checks: list[CheckResult] = field(default_factory=list)
    quality_checks: list[CheckResult] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_thresholds(path: Path) -> dict:
    """Load and validate the threshold spec.

    Validates that the top-level required keys (``minimum_sample``,
    ``decision_quality``, ``evidence_quality``) and each section's required
    nested keys are present. Raises ``ValueError`` with a clear message if
    any are missing.

    Does NOT type-check the values beyond JSON parsing. Operators are
    expected to keep the values numeric -- the evaluator will surface a
    ``TypeError`` at compare time if they don't, which is the right failure
    mode for a misconfigured operator-controlled file.
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
            f"key(s): {missing_top}"
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
    """Load the aggregator summary JSON from a file path or ``-`` (stdin).

    Accepts a single JSON object only. JSON-lines input is not supported
    here -- the upstream is ``aggregate_shadow_judgments.py --json-out``,
    which produces one object.

    Raises ``FileNotFoundError`` if the named path doesn't exist and
    ``json.JSONDecodeError`` on bad JSON. Raises ``ValueError`` if the
    payload isn't a top-level object.
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
    return data


# ---------------------------------------------------------------------------
# Pure evaluation
# ---------------------------------------------------------------------------


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def evaluate(*, summary: dict, thresholds: dict) -> EvaluationReport:
    """Evaluate a summary against a threshold spec. Pure: no I/O.

    Both ``sample_checks`` and ``quality_checks`` are always populated --
    the latter is computed even when sample checks fail so operators can
    see them as diagnostics. Only ``sample_checks`` drive the verdict when
    they fail.
    """

    total_rows_read = int(summary["total_rows_read"])
    total_compared = int(summary["total_compared"])
    save_to_reject = int(summary.get("save_to_reject", 0))
    reject_to_save = int(summary.get("reject_to_save", 0))
    unavailable = int(summary.get("unavailable_external_evidence", 0))
    weak_citations = int(summary.get("weak_citations", 0))

    sample_thresholds = thresholds["minimum_sample"]
    decision_thresholds = thresholds["decision_quality"]
    evidence_thresholds = thresholds["evidence_quality"]

    total_rows_read_min = int(sample_thresholds["total_rows_read_min"])
    total_compared_min = int(sample_thresholds["total_compared_min"])
    max_save_to_reject_rate = float(decision_thresholds["max_save_to_reject_rate"])
    min_reject_to_save_lift = float(decision_thresholds["min_reject_to_save_lift"])
    max_unavailable_rate = float(evidence_thresholds["max_unavailable_evidence_rate"])
    max_weak_citation_rate = float(evidence_thresholds["max_weak_citation_rate"])

    save_to_reject_rate = _safe_rate(save_to_reject, total_compared)
    reject_to_save_rate = _safe_rate(reject_to_save, total_compared)
    unavailable_rate = _safe_rate(unavailable, total_rows_read)
    weak_citation_rate = _safe_rate(weak_citations, total_rows_read)

    sample_checks: list[CheckResult] = [
        CheckResult(
            name="min_total_rows_read",
            passed=total_rows_read >= total_rows_read_min,
            observed=total_rows_read,
            threshold=total_rows_read_min,
            comparator=">=",
            category="sample",
        ),
        CheckResult(
            name="min_total_compared",
            passed=total_compared >= total_compared_min,
            observed=total_compared,
            threshold=total_compared_min,
            comparator=">=",
            category="sample",
        ),
    ]

    quality_checks: list[CheckResult] = [
        CheckResult(
            name="max_save_to_reject_rate",
            passed=save_to_reject_rate <= max_save_to_reject_rate,
            observed=save_to_reject_rate,
            threshold=max_save_to_reject_rate,
            comparator="<=",
            category="decision_quality",
        ),
        CheckResult(
            name="min_reject_to_save_lift",
            passed=reject_to_save_rate >= min_reject_to_save_lift,
            observed=reject_to_save_rate,
            threshold=min_reject_to_save_lift,
            comparator=">=",
            category="decision_quality",
        ),
        CheckResult(
            name="max_unavailable_evidence_rate",
            passed=unavailable_rate <= max_unavailable_rate,
            observed=unavailable_rate,
            threshold=max_unavailable_rate,
            comparator="<=",
            category="evidence_quality",
        ),
        CheckResult(
            name="max_weak_citation_rate",
            passed=weak_citation_rate <= max_weak_citation_rate,
            observed=weak_citation_rate,
            threshold=max_weak_citation_rate,
            comparator="<=",
            category="evidence_quality",
        ),
    ]

    if any(not c.passed for c in sample_checks):
        verdict = VERDICT_INSUFFICIENT_SAMPLE
    elif any(not c.passed for c in quality_checks):
        verdict = VERDICT_FAIL
    else:
        verdict = VERDICT_PASS

    inputs = {
        "total_rows_read": total_rows_read,
        "total_compared": total_compared,
        "save_to_reject": save_to_reject,
        "reject_to_save": reject_to_save,
        "unavailable_external_evidence": unavailable,
        "weak_citations": weak_citations,
        "save_to_reject_rate": save_to_reject_rate,
        "reject_to_save_rate": reject_to_save_rate,
        "unavailable_evidence_rate": unavailable_rate,
        "weak_citation_rate": weak_citation_rate,
        "total_rows_read_min": total_rows_read_min,
        "total_compared_min": total_compared_min,
        "max_save_to_reject_rate": max_save_to_reject_rate,
        "min_reject_to_save_lift": min_reject_to_save_lift,
        "max_unavailable_evidence_rate": max_unavailable_rate,
        "max_weak_citation_rate": max_weak_citation_rate,
    }

    return EvaluationReport(
        verdict=verdict,
        sample_checks=sample_checks,
        quality_checks=quality_checks,
        inputs=inputs,
    )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------


def _fmt_rate(value: float) -> str:
    return f"{value:.3f}"


def _fmt_check_line(check: CheckResult, *, suppressed: bool) -> str:
    status = "PASS" if check.passed else "FAIL"
    if suppressed:
        prefix = "[N/A - insufficient sample]"
    else:
        prefix = f"[{status}]"
    if isinstance(check.threshold, float) and not float(check.threshold).is_integer():
        threshold_disp = f"{check.threshold:.3f}"
    elif check.category == "sample":
        threshold_disp = f"{int(check.threshold)}"
    else:
        threshold_disp = f"{check.threshold:.3f}"
    if check.category == "sample":
        observed_disp = f"{int(check.observed)}"
    else:
        observed_disp = _fmt_rate(check.observed)
    body = (
        f"{check.name} {check.comparator} {threshold_disp}"
    )
    return f"  {prefix} {body:<48} : {observed_disp}"


def format_report(report: EvaluationReport, *, quiet: bool) -> str:
    """Render the report for stdout.

    With ``quiet=True``, returns just the verdict on a single line. With
    ``quiet=False``, returns the full multi-section human report.
    """

    if quiet:
        return report.verdict

    inputs = report.inputs
    lines: list[str] = []
    lines.append("=== Shadow Bakeoff Acceptance Evaluation ===")
    lines.append(f"Verdict: {report.verdict}")
    lines.append("")
    lines.append("Inputs:")
    lines.append(
        f"  total_rows_read              : {int(inputs['total_rows_read'])}"
    )
    lines.append(
        f"  total_compared               : {int(inputs['total_compared'])}"
    )
    lines.append(
        f"  save_to_reject_rate          : "
        f"{_fmt_rate(inputs['save_to_reject_rate'])}  "
        f"({int(inputs['save_to_reject'])} / {int(inputs['total_compared'])})"
    )
    lines.append(
        f"  reject_to_save_rate          : "
        f"{_fmt_rate(inputs['reject_to_save_rate'])}  "
        f"({int(inputs['reject_to_save'])} / {int(inputs['total_compared'])})"
    )
    lines.append(
        f"  unavailable_evidence_rate    : "
        f"{_fmt_rate(inputs['unavailable_evidence_rate'])}  "
        f"({int(inputs['unavailable_external_evidence'])} / {int(inputs['total_rows_read'])})"
    )
    lines.append(
        f"  weak_citation_rate           : "
        f"{_fmt_rate(inputs['weak_citation_rate'])}  "
        f"({int(inputs['weak_citations'])} / {int(inputs['total_rows_read'])})"
    )
    lines.append("")

    suppressed = report.verdict == VERDICT_INSUFFICIENT_SAMPLE

    lines.append("Sample-size checks:")
    for check in report.sample_checks:
        lines.append(_fmt_check_line(check, suppressed=False))
    lines.append("")

    lines.append("Decision-quality checks:")
    for check in report.quality_checks:
        if check.category != "decision_quality":
            continue
        lines.append(_fmt_check_line(check, suppressed=suppressed))
    lines.append("")

    lines.append("Evidence-quality checks:")
    for check in report.quality_checks:
        if check.category != "evidence_quality":
            continue
        lines.append(_fmt_check_line(check, suppressed=suppressed))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON report writer (atomic)
# ---------------------------------------------------------------------------


def _check_to_dict(check: CheckResult) -> dict:
    d = asdict(check)
    d["passed"] = bool(d["passed"])
    return d


def write_report_json(
    report: EvaluationReport,
    path: Path,
    *,
    thresholds_path: str,
    summary_path: str,
) -> None:
    """Atomically write the structured report to ``path``.

    Mirrors ``tools/aggregate_shadow_judgments.py:write_summary_json``.
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
            "sample_checks": [_check_to_dict(c) for c in report.sample_checks],
            "quality_checks": [_check_to_dict(c) for c in report.quality_checks],
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

    thresholds_path = Path(args.thresholds) if args.thresholds else DEFAULT_THRESHOLDS_PATH
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
            f"ERROR: thresholds file is not valid JSON: {thresholds_path}: {exc}",
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
            "Evaluate a shadow-bakeoff aggregator summary against committed "
            "acceptance thresholds. Produces PASS / FAIL / "
            "INSUFFICIENT_SAMPLE. Analytical/debug only; no canonical state "
            "writes."
        )
    )
    parser.add_argument(
        "--summary",
        required=True,
        help=(
            "Path to the aggregator's --json-out file, or '-' to read a "
            "single JSON object from stdin. JSON-lines is not supported."
        ),
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help=(
            "Optional path to a threshold-spec JSON. Defaults to "
            "config/perplexity-bakeoff-thresholds.json relative to the "
            "repo root."
        ),
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help=(
            "Optional path to write the structured pass/fail report as "
            "JSON. Default: nothing is written to disk."
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
