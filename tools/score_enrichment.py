#!/usr/bin/env python3
"""Score every Boolean before and after enrichment, and name what is left.

Three corpora, because they answer different questions. The in-role 52 are
the rater-ranked strings, so they are the ones whose scores can be checked
against a human judgment. The 76 are the whole calibration corpus, two
thirds of it for roles this lexicon was not written for, which is where an
enricher that only works on its own vocabulary shows up. The real plans are
production output and the only place the wiring will ever run.

The point of the run is not the mean. It is the RESIDUE: the strings
enrichment cannot help, grouped by the defect that is still there. Adding
forms cannot remove an umbrella term from a group, cannot put back an anchor
the model never wrote, cannot un-exclude a title the string excludes itself
with. Those groups are the post-processor's boundary, and they are Stage 3's
input.

    python tools/score_enrichment.py
    python tools/score_enrichment.py --plans 'output/state/linkedin/*/execution_plan.json'
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

from linkedin.boolean_score.enrich import enrich  # noqa: E402
from linkedin.boolean_score.lexicon import (  # noqa: E402
    LEXICON_DIR,
    LexiconError,
    Lexicon,
    load_lexicon,
)
from linkedin.boolean_score.parser import parse  # noqa: E402
from linkedin.boolean_score.score import (  # noqa: E402
    PARSIMONY,
    WEIGHTS,
    ScoreReport,
    score,
)

OUTPUT_DIR = ROOT / "output"
FIXTURES = ROOT / "tests" / "fixtures" / "boolean_corpus"
LEXICON_PATH = LEXICON_DIR / "research-engineer-post-training.json"
DEFAULT_PLANS = str(ROOT / "output" / "state" / "linkedin" / "*" / "execution_plan.json")
DEFAULT_OUT = ROOT / "output" / "boolean-rl" / "enrichment"
REPORTED_COMPONENT_KEYS = (*WEIGHTS, "structure", PARSIMONY)


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------


def read_fixture(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text())
    entries = []
    for index, item in enumerate(payload.get("strings", [])):
        boolean = item.get("boolean")
        if not isinstance(boolean, str) or not boolean.strip():
            continue
        entries.append({"id": str(item.get("id", index)), "boolean": boolean})
    return entries


def read_plans(pattern: str) -> list[dict[str, str]]:
    """Every authored Boolean in every matching plan, read-only.

    Both surfaces the wiring will touch: ``generated_strings[].boolean`` and
    ``coverage_gaps[].suggested_boolean``. The sibling corpus tool reads only
    the first, which is why its count and this one's differ.
    """
    entries: list[dict[str, str]] = []
    for plan_path in sorted(glob.glob(pattern)):
        path = Path(plan_path)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        label = path.parent.name
        for index, item in enumerate(payload.get("generated_strings", []) or []):
            boolean = item.get("boolean") if isinstance(item, dict) else None
            if isinstance(boolean, str) and boolean.strip():
                entries.append({"id": f"{label}-s{index}", "boolean": boolean})
        for index, item in enumerate(payload.get("coverage_gaps", []) or []):
            boolean = item.get("suggested_boolean") if isinstance(item, dict) else None
            if isinstance(boolean, str) and boolean.strip():
                entries.append({"id": f"{label}-g{index}", "boolean": boolean})
    return entries


def in_role_ids() -> set[str]:
    """The 52 rater-ranked ids — the same set the weight fitter calls in-role."""
    payload = json.loads((FIXTURES / "rater_targets.json").read_text())
    return {row["id"] for row in payload["strings"]}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure(
    entries: Sequence[dict[str, str]], lexicon: Lexicon
) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        original = entry["boolean"]
        if not parse(original).legal:
            continue
        report = enrich(original, lexicon)
        before = score(original, lexicon)
        after = score(report.enriched, lexicon)
        rows.append(
            {
                "id": entry["id"],
                "boolean": original,
                "enriched": report.enriched,
                "changed": not report.unchanged,
                "truncated": report.truncated,
                "before": before,
                "after": after,
                "delta": after.log_composite - before.log_composite,
            }
        )
    return rows


def _component_means(
    rows: Sequence[dict[str, Any]], side: str
) -> dict[str, float | None]:
    means: dict[str, float | None] = {}
    for key in REPORTED_COMPONENT_KEYS:
        values = [
            row[side].components[key]
            for row in rows
            if row[side].components.get(key) is not None
        ]
        means[key] = statistics.fmean(values) if values else None
    return means


def _composites(rows: Sequence[dict[str, Any]], side: str) -> list[float]:
    return sorted(row[side].composite for row in rows)


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
    return values[index]


# ---------------------------------------------------------------------------
# Residue
# ---------------------------------------------------------------------------

RESIDUE_ORDER = (
    "umbrella_in_group",
    "self_exclusion",
    "grab_bag",
    "missing_anchor",
    "unmapped_terms",
    "off_role_relevance",
    "already_clean",
)


def residue_reasons(report: ScoreReport, lexicon: Lexicon) -> list[str]:
    """Why enrichment could not lift this string. Vocabulary cannot fix shape."""
    reasons = []
    bloat = report.bloat
    if bloat.get("umbrella_in_group", 0):
        reasons.append("umbrella_in_group")
    if bloat.get("self_exclusion", 0):
        reasons.append("self_exclusion")
    if bloat.get("grab_bag", 0):
        reasons.append("grab_bag")
    if not report.anchor_present and not report.anchored_by_filter:
        reasons.append("missing_anchor")
    unmapped = sum(
        1
        for group in report.groups
        for concept_id in group.concept_ids
        if concept_id is None
    )
    total = sum(len(group.terms) for group in report.groups) or 1
    if unmapped / total > 0.5:
        reasons.append("unmapped_terms")
    relevance = report.components.get("relevance")
    if relevance is not None and relevance < 0.7:
        reasons.append("off_role_relevance")
    return reasons or ["already_clean"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_block(name: str, rows: Sequence[dict[str, Any]], lexicon: Lexicon) -> list[str]:
    if not rows:
        return [f"{name}: no legal strings", ""]
    improved = [row for row in rows if row["delta"] > 1e-9]
    held = [row for row in rows if abs(row["delta"]) <= 1e-9]
    worse = [row for row in rows if row["delta"] < -1e-9]
    changed = [row for row in rows if row["changed"]]

    before = _component_means(rows, "before")
    after = _component_means(rows, "after")
    bc, ac = _composites(rows, "before"), _composites(rows, "after")

    lines = [
        f"{name}: {len(rows)} legal strings, {len(changed)} rewritten",
        f"  improved {len(improved)}  held {len(held)}  regressed {len(worse)}"
        f"   ({(len(improved) + len(held)) / len(rows):.1%} do not regress)",
        "",
        "  component            before    after    delta",
    ]
    for key in REPORTED_COMPONENT_KEYS:
        b, a = before[key], after[key]
        if b is None or a is None:
            lines.append(f"    {key:<19}     —        —        —")
            continue
        lines.append(f"    {key:<19} {b:>7.4f}  {a:>7.4f}  {a - b:>+7.4f}")
    lines.append("")
    lines.append("  composite            before    after    delta")
    for label, fraction in (
        ("min", 0.0),
        ("q1", 0.25),
        ("median", 0.5),
        ("q3", 0.75),
        ("max", 1.0),
    ):
        b, a = _quantile(bc, fraction), _quantile(ac, fraction)
        lines.append(f"    {label:<19} {b:>7.4f}  {a:>7.4f}  {a - b:>+7.4f}")
    bm, am = statistics.fmean(bc), statistics.fmean(ac)
    lines.append(f"    {'mean':<19} {bm:>7.4f}  {am:>7.4f}  {am - bm:>+7.4f}")
    lines.append("")

    stuck = [row for row in rows if row["delta"] <= 1e-9]
    tally: dict[str, int] = {}
    for row in stuck:
        for reason in residue_reasons(row["after"], lexicon):
            tally[reason] = tally.get(reason, 0) + 1
    lines.append(f"  residue: {len(stuck)} strings enrichment does not lift")
    for reason in RESIDUE_ORDER:
        if reason in tally:
            lines.append(f"    {tally[reason]:>4}  {reason}")
    lines.append("")
    return lines


def _check_out_path(raw: str) -> Path:
    destination = Path(raw)
    if not destination.is_absolute():
        destination = ROOT / destination
    resolved = destination.resolve()
    if resolved == OUTPUT_DIR or OUTPUT_DIR in resolved.parents:
        raise SystemExit(
            "refusing to write under output/: finalized run output is rebuilt "
            "through code paths, never by this tool. Pick a path outside it."
        )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lexicon", default=str(LEXICON_PATH))
    parser.add_argument("--plans", default=DEFAULT_PLANS)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args(argv)

    destination = _check_out_path(args.out)
    try:
        lexicon = load_lexicon(args.lexicon)
    except LexiconError as error:
        print(f"lexicon: {error}", file=sys.stderr)
        return 2

    corpus = read_fixture(FIXTURES / "ml_runs_76.json")
    in_role = in_role_ids()
    blocks = {
        "in-role 52": [row for row in corpus if row["id"] in in_role],
        "the 76": corpus,
        "real plans": read_plans(args.plans),
    }

    lines: list[str] = [f"role: {lexicon.role}", ""]
    measured: dict[str, list[dict[str, Any]]] = {}
    for name, entries in blocks.items():
        rows = measure(entries, lexicon)
        measured[name] = rows
        lines.extend(render_block(name, rows, lexicon))

    print("\n".join(lines))

    destination.mkdir(parents=True, exist_ok=True)
    residue_payload = {}
    for name, rows in measured.items():
        residue_payload[name] = [
            {
                "id": row["id"],
                "reasons": residue_reasons(row["after"], lexicon),
                "composite_before": round(row["before"].composite, 4),
                "composite_after": round(row["after"].composite, 4),
                "delta_log": round(row["delta"], 4),
                "shape": row["after"].shape,
                "bloat": {k: round(v, 4) for k, v in row["after"].bloat.items() if v},
                "boolean": row["boolean"],
            }
            for row in rows
            if row["delta"] <= 1e-9
        ]
    (destination / "residue.json").write_text(
        json.dumps(residue_payload, indent=2, ensure_ascii=False) + "\n"
    )
    (destination / "summary.txt").write_text("\n".join(lines) + "\n")
    regressions = {
        name: [
            {
                "id": row["id"],
                "before": round(row["before"].composite, 4),
                "after": round(row["after"].composite, 4),
                "bloat_before": {
                    k: round(v, 4) for k, v in row["before"].bloat.items() if v
                },
                "bloat_after": {
                    k: round(v, 4) for k, v in row["after"].bloat.items() if v
                },
                "boolean": row["boolean"],
                "enriched": row["enriched"],
            }
            for row in rows
            if row["delta"] < -1e-9
        ]
        for name, rows in measured.items()
    }
    (destination / "regressions.json").write_text(
        json.dumps(regressions, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\nwrote {destination}/residue.json, regressions.json, summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
