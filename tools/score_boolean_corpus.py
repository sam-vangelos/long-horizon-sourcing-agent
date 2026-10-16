#!/usr/bin/env python3
"""Score a corpus of Boolean strings and print the calibration table.

Two sources. ``--fixture`` reads a JSON file shaped like
tests/fixtures/boolean_corpus/ml_runs_76.json — an object with a ``strings``
list, each entry carrying ``id`` and ``boolean``. ``--plans`` takes a glob of
execution plans and reads the ``boolean`` of every entry in each plan's
``generated_strings``.

The plan path is READ-ONLY and the tool enforces it: it refuses any ``--out``
that resolves inside the repo's ``output/`` tree, which is immutable run
output and is rebuilt through code paths, never by hand.

    python tools/score_boolean_corpus.py \
        --fixture tests/fixtures/boolean_corpus/ml_runs_76.json \
        --lexicon linkedin/boolean_score/lexicons/research-engineer-post-training.json \
        [--out report.json]

    python tools/score_boolean_corpus.py \
        --plans 'output/state/linkedin/*/execution_plan.json' --lexicon <path>
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linkedin.boolean_score.lexicon import LexiconError, load_lexicon  # noqa: E402
from linkedin.boolean_score.score import (  # noqa: E402
    PARSIMONY,
    WEIGHTS,
    ScoreReport,
    score,
)

# Parsimony multiplies rather than contributing a weighted share, so it is not
# in WEIGHTS; the table still reports it beside the four that are.
REPORTED_COMPONENT_KEYS = (*WEIGHTS, "structure", PARSIMONY)

OUTPUT_DIR = ROOT / "output"
_TOP_N = 5


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_fixture(path: str | Path) -> list[dict[str, Any]]:
    """Entries from a corpus fixture: ``{"strings": [{"id", "boolean"}, ...]}``."""
    payload = json.loads(Path(path).read_text())
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("strings", [])):
        boolean = item.get("boolean")
        if not isinstance(boolean, str) or not boolean.strip():
            continue
        entries.append({"id": str(item.get("id", index)), "boolean": boolean})
    return entries


def read_plans(pattern: str) -> list[dict[str, Any]]:
    """Every generated string in every execution plan the glob matches.

    Read-only by construction: the plans are opened and never written.
    """
    entries: list[dict[str, Any]] = []
    for plan_path in sorted(glob.glob(pattern)):
        path = Path(plan_path)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        label = path.parent.name
        for index, item in enumerate(payload.get("generated_strings", [])):
            boolean = item.get("boolean") if isinstance(item, dict) else None
            if not isinstance(boolean, str) or not boolean.strip():
                continue
            entries.append({"id": f"{label}-{item.get('id', index)}", "boolean": boolean})
    return entries


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------


def summarize(reports: Sequence[ScoreReport]) -> dict[str, Any]:
    composites = [report.composite for report in reports]
    summary: dict[str, Any] = {
        "count": len(reports),
        "legal": sum(1 for report in reports if report.legal),
        "composite": _distribution(composites),
        "components": {},
        "findings": {},
    }
    for name in REPORTED_COMPONENT_KEYS:
        values = [
            report.components[name]
            for report in reports
            if report.components[name] is not None
        ]
        summary["components"][name] = {
            "weight": WEIGHTS.get(name),
            "mean": statistics.fmean(values) if values else None,
            "applicable": len(values),
            "applicability_rate": len(values) / len(reports) if reports else 0.0,
        }
    tally: dict[str, int] = {}
    for report in reports:
        for finding in report.findings:
            tally[finding.code] = tally.get(finding.code, 0) + 1
    summary["findings"] = dict(sorted(tally.items(), key=lambda item: (-item[1], item[0])))
    summary["anchored"] = sum(1 for report in reports if report.anchor_present)
    return summary


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "q1": None, "median": None, "q3": None, "max": None}
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4) if len(ordered) > 1 else [ordered[0]] * 3
    return {
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "q1": quartiles[0],
        "median": quartiles[1],
        "q3": quartiles[2],
        "max": ordered[-1],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _component_line(report: ScoreReport) -> str:
    parts = []
    for name in REPORTED_COMPONENT_KEYS:
        value = report.components[name]
        parts.append(f"{name}={'n/a' if value is None else format(value, '.3f')}")
    return " ".join(parts)


def render(summary: dict[str, Any], scored: Sequence[tuple[str, ScoreReport]]) -> list[str]:
    lines: list[str] = []
    composite = summary["composite"]
    lines.append(f"strings scored: {summary['count']}  (legal: {summary['legal']})")
    lines.append(
        "composite  mean {mean:.4f}   min {min:.4f}  q1 {q1:.4f}  "
        "median {median:.4f}  q3 {q3:.4f}  max {max:.4f}".format(**composite)
    )
    lines.append(f"class anchor present: {summary['anchored']}/{summary['count']}")
    lines.append("")
    lines.append("component            weight   mean   applicable")
    for name, stats in summary["components"].items():
        mean = "n/a" if stats["mean"] is None else format(stats["mean"], ".4f")
        weight = "  x  " if stats["weight"] is None else format(stats["weight"], ".2f")
        lines.append(
            f"  {name:<19} {weight:>5}  {mean:>6}   "
            f"{stats['applicable']:>3}/{summary['count']} "
            f"({stats['applicability_rate']:.0%})"
        )
    lines.append("  (x = multiplies the weighted mean rather than joining it)")
    lines.append("")
    ranked = sorted(scored, key=lambda item: item[1].composite, reverse=True)
    lines.append(f"five highest")
    for label, report in ranked[:_TOP_N]:
        lines.append(f"  {report.composite:.4f}  {label}")
        lines.append(f"          composite {report.composite:.4f}  {_component_line(report)}")
    lines.append("")
    lines.append(f"five lowest")
    for label, report in ranked[-_TOP_N:][::-1]:
        lines.append(f"  {report.composite:.4f}  {label}")
        lines.append(f"          composite {report.composite:.4f}  {_component_line(report)}")
    lines.append("")
    lines.append("findings")
    if summary["findings"]:
        for code, count in summary["findings"].items():
            lines.append(f"  {count:>4}  {code}")
    else:
        lines.append("  none")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _check_out_path(raw: str) -> Path:
    destination = Path(raw)
    resolved = (destination if destination.is_absolute() else ROOT / destination).resolve()
    if resolved == OUTPUT_DIR or OUTPUT_DIR in resolved.parents:
        raise SystemExit(
            "refusing to write under output/: finalized run output is rebuilt "
            "through code paths, never by this tool. Pick a path outside it."
        )
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", help="corpus fixture JSON")
    source.add_argument("--plans", help="glob of execution_plan.json files (read-only)")
    parser.add_argument("--lexicon", required=True, help="role lexicon JSON")
    parser.add_argument("--out", help="write the full per-string reports here")
    args = parser.parse_args(argv)

    destination = _check_out_path(args.out) if args.out else None

    try:
        lexicon = load_lexicon(args.lexicon)
    except LexiconError as error:
        print(f"lexicon: {error}", file=sys.stderr)
        return 2

    if args.fixture:
        fixture_path = Path(args.fixture)
        if not fixture_path.exists():
            print(f"no corpus at {fixture_path}", file=sys.stderr)
            return 2
        entries = read_fixture(fixture_path)
    else:
        entries = read_plans(args.plans)

    if not entries:
        print("no Boolean strings found", file=sys.stderr)
        return 2

    scored = [(entry["id"], score(entry["boolean"], lexicon)) for entry in entries]
    summary = summarize([report for _label, report in scored])
    print("\n".join(render(summary, scored)))

    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "role": lexicon.role,
                    "count": summary["count"],
                    "summary": summary,
                    "reports": [
                        {"id": label, **report.to_dict()} for label, report in scored
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
