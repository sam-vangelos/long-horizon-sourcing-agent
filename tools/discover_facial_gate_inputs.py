#!/usr/bin/env python3
"""Real-run input discoverer for the facial-gate recommendation workflow.

Slice 11 of perplexity-evidence-augmentation. This is an *analytical / debug*
tool. It walks ``output/runs/linkedin/<brief-id>/<run-dir>/`` and reports
which finalized run-dirs carry the three inputs the slice-5 harness
(``tools/experiments/facial_gate_experiment.py``) needs:

- ``snippets.jsonl`` -- **required**. Without it the harness cannot run.
- ``profile_summaries.jsonl`` -- optional but improves the
  ``likely_false_negatives_under_variant`` heuristic.
- ``final_judgments.jsonl`` -- optional but enables save-recovery analysis.

Output is a usability classification per run plus a one-line headline.
Operators can pipe one usable run's paths into the slice-5 harness
invocation. When at least one usable run is found, the tool emits a
copy-pasteable harness command for the highest-quality usable run.

Hard guarantees:

- No live LLM or network calls. No imports from ``linkedin/``, ``github/``,
  ``market_intelligence/``, or any production module. Stdlib +
  ``shared.output_paths`` only (path-convention module with no LLM,
  orchestration, or runtime-state side effects).
- Never writes under ``output/state/``, ``output/runs/``,
  ``output/market_intelligence/``, ``output/exports/``, ``output/archive/``,
  ``output/cache/``, or ``output/debug/`` -- ``--json-out`` rejects any
  path that resolves under those subtrees.
- Default invocation writes nothing to disk. ``--json-out <path>`` opts in
  to writing the structured report to a path the operator names outside
  ``output/``.
- Empty discovery (zero usable runs) exits ``0``: this is a query, not a
  hard input requirement. ``SystemExit(1)`` is reserved for hard input
  errors (``--json-out`` resolves under a protected path).

LinkedIn-only for v1 (mirrors slice 6's ``--discover linkedin`` choice):
the harness only handles LinkedIn snippets today; reporting GitHub
run-dirs would be misleading.

CLI shape::

    python tools/discover_facial_gate_inputs.py
        [--source linkedin]
        [--brief-id BRIEF_ID]
        [--require-finals]
        [--require-summaries]
        [--limit N]
        [--json-out PATH]
        [--include-legacy]
        [--include-unknown-brief]

Companions in the perplexity-evidence-augmentation feature:

- ``tools/aggregate_shadow_judgments.py --discover linkedin`` -- discovers
  ``shadow_final_judgments.jsonl`` under ``output/state/linkedin/`` for the
  shadow bakeoff (slice 4/6). DIFFERENT artifact, DIFFERENT location.
- ``tools/experiments/facial_gate_experiment.py`` -- the slice-5 harness
  this tool produces inputs for.
- ``tools/experiments/recommend_facial_gate.py`` -- slice 10 interpreter
  of the harness's ``--json-out``.
- ``docs/facial-gate-experiment-runbook.md`` -- operator workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from shared.output_paths import (  # noqa: E402
    RUNS_ROOT,
    classify_output_location,
    is_run_dir,
)


_DISCOVERY_SOURCES = frozenset({"linkedin"})

_SNIPPETS_FILENAME = "snippets.jsonl"
_PROFILE_SUMMARIES_FILENAME = "profile_summaries.jsonl"
_FINAL_JUDGMENTS_FILENAME = "final_judgments.jsonl"


# Output subtrees the ``--json-out`` writer refuses to write into. Mirrors
# the directory taxonomy in ``shared/output_paths.py``. Anything that
# ``classify_output_location`` returns from this set blocks the write; the
# only allowed targets are ``"external"`` (path is outside ``output/``)
# paths.
_PROTECTED_LOCATIONS = frozenset(
    {
        "output_root",
        "legacy_output",
        "state_root",
        "state_dir",
        "runs_root",
        "run_dir",
        "market_root",
        "market_dir",
        "exports_dir",
        "archive_dir",
        "cache_dir",
        "debug_dir",
    }
)


USABILITY_FULL = "usable_full"
USABILITY_RECOVERY_ONLY = "usable_recovery_only"
USABILITY_MINIMAL = "usable_minimal"
USABILITY_INCOMPLETE = "incomplete_no_snippets"


_USABILITY_ORDER: tuple[str, ...] = (
    USABILITY_FULL,
    USABILITY_RECOVERY_ONLY,
    USABILITY_MINIMAL,
    USABILITY_INCOMPLETE,
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunInputClassification:
    """Per-run-dir classification of the three facial-gate harness inputs.

    ``snippets_size_bytes`` / ``summaries_size_bytes`` / ``finals_size_bytes``
    are ``None`` when the corresponding ``has_*`` is ``False``. The size is
    reported only as a human-readable display hint; this tool never reads
    file contents.

    ``incomplete_reason`` is the empty string for any ``usable_*`` row; it
    only carries a description on ``incomplete_no_snippets``.
    """

    source: str
    brief_id: str
    run_dir: Path
    run_label: str
    has_snippets: bool
    has_profile_summaries: bool
    has_final_judgments: bool
    snippets_size_bytes: Optional[int]
    summaries_size_bytes: Optional[int]
    finals_size_bytes: Optional[int]
    usability: str
    incomplete_reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_dir"] = str(self.run_dir)
        return d


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _is_legacy_name(name: str) -> bool:
    """Match the canonical legacy-import shape from ``resolve_run_dir``.

    ``shared/output_paths.py:resolve_run_dir`` produces
    ``imported-{run_stamp}__legacy-{N}`` for legacy archive imports. We
    require BOTH the ``imported-`` prefix AND the ``__legacy-`` infix so a
    hypothetical future ``imported-something-else__run-1`` would not be
    mistakenly skipped.
    """

    return name.startswith("imported-") and "__legacy-" in name


def discover_run_dirs(
    *,
    source: str,
    runs_root: Optional[Path] = None,
    include_legacy: bool,
    include_unknown_brief: bool,
    brief_id_filter: Optional[str],
) -> list[Path]:
    """Walk the runs root and return every directory that is a run-dir.

    A "run-dir" is any path classified as ``"run_dir"`` by
    ``shared.output_paths.is_run_dir`` -- i.e. ``output/runs/<source>/<brief>/<run>``
    with at least four parts after ``output``. We do NOT require the run
    name to begin with a timestamp; the canonical ``resolve_run_dir`` may
    produce ``imported-...__legacy-N`` names too. The legacy filter is the
    only name-pattern gate.

    Filter rules:

    - ``include_legacy=False`` (default) skips run-dir names that match the
      ``imported-...__legacy-...`` shape. With ``include_legacy=True``,
      they are listed.
    - ``include_unknown_brief=False`` (default) skips the entire
      ``<runs_root>/<source>/unknown/`` subtree. With
      ``include_unknown_brief=True``, those run-dirs are listed.
    - ``brief_id_filter`` is a substring match against the brief-id
      directory name (case-sensitive, no glob). ``None`` disables the
      filter.

    Returns a deterministically sorted, deduplicated list. Sort key is
    ``"<brief-id>/<run-label>"`` so output is stable across runs.

    Pure modulo the filesystem: no network, no LLM, no writes.
    """

    if source not in _DISCOVERY_SOURCES:
        raise ValueError(
            f"unsupported discovery source: {source!r} "
            f"(allowed: {sorted(_DISCOVERY_SOURCES)})"
        )

    root = Path(runs_root) if runs_root is not None else Path(RUNS_ROOT)
    source_root = root / source
    if not source_root.exists() or not source_root.is_dir():
        return []

    seen: set[Path] = set()
    out: list[Path] = []

    for brief_dir in sorted(source_root.iterdir(), key=lambda p: p.name):
        if not brief_dir.is_dir():
            continue
        brief_id = brief_dir.name
        if brief_id == "unknown" and not include_unknown_brief:
            continue
        if brief_id_filter is not None and brief_id_filter not in brief_id:
            continue

        for run_dir in sorted(brief_dir.iterdir(), key=lambda p: p.name):
            if not run_dir.is_dir():
                continue
            if not is_run_dir(run_dir):
                continue
            if not include_legacy and _is_legacy_name(run_dir.name):
                continue
            try:
                key = run_dir.resolve()
            except OSError:
                key = run_dir
            if key in seen:
                continue
            seen.add(key)
            out.append(run_dir)

    out.sort(key=lambda p: f"{p.parent.name}/{p.name}")
    return out


def classify_run_inputs(run_dir: Path) -> RunInputClassification:
    """Inspect a single run-dir for the three facial-gate harness inputs.

    Pure modulo the filesystem: ``Path.is_file()`` and ``Path.stat().st_size``
    only. Does not open or read file contents -- presence and size are the
    only signals this tool needs.

    Classification rules (the only product judgment in this tool; pinned
    in tests):

    - ``usable_full``: snippets + profile_summaries + final_judgments all
      present. Best -- ``analyze_recovery`` produces full numbers and the
      ``likely_false_negatives_under_variant`` heuristic fires.
    - ``usable_recovery_only``: snippets + final_judgments, no
      profile_summaries. Recovery counters are computable; the
      false-negative heuristic is degraded (per slice 5: requires
      summaries to fire).
    - ``usable_minimal``: snippets only. Variant counters are computable;
      recovery is not. Still useful for operator inspection.
    - ``incomplete_no_snippets``: snippets missing (with or without the
      others). The harness cannot run at all on this directory.

    ``incomplete_reason`` is populated only on ``incomplete_no_snippets``.
    For all ``usable_*`` it is the empty string.
    """

    run_dir = Path(run_dir)
    snippets_path = run_dir / _SNIPPETS_FILENAME
    summaries_path = run_dir / _PROFILE_SUMMARIES_FILENAME
    finals_path = run_dir / _FINAL_JUDGMENTS_FILENAME

    has_snippets = snippets_path.is_file()
    has_summaries = summaries_path.is_file()
    has_finals = finals_path.is_file()

    snippets_size = snippets_path.stat().st_size if has_snippets else None
    summaries_size = summaries_path.stat().st_size if has_summaries else None
    finals_size = finals_path.stat().st_size if has_finals else None

    if not has_snippets:
        usability = USABILITY_INCOMPLETE
        incomplete_reason = (
            f"missing snippets.jsonl (required input for the slice-5 "
            f"harness); summaries={'present' if has_summaries else 'missing'}, "
            f"finals={'present' if has_finals else 'missing'}"
        )
    elif has_summaries and has_finals:
        usability = USABILITY_FULL
        incomplete_reason = ""
    elif has_finals and not has_summaries:
        usability = USABILITY_RECOVERY_ONLY
        incomplete_reason = ""
    else:
        usability = USABILITY_MINIMAL
        incomplete_reason = ""

    brief_id = run_dir.parent.name if run_dir.parent is not None else ""
    source = (
        run_dir.parent.parent.name
        if run_dir.parent is not None and run_dir.parent.parent is not None
        else ""
    )

    return RunInputClassification(
        source=source,
        brief_id=brief_id,
        run_dir=run_dir,
        run_label=run_dir.name,
        has_snippets=has_snippets,
        has_profile_summaries=has_summaries,
        has_final_judgments=has_finals,
        snippets_size_bytes=snippets_size,
        summaries_size_bytes=summaries_size,
        finals_size_bytes=finals_size,
        usability=usability,
        incomplete_reason=incomplete_reason,
    )


def _empty_buckets() -> dict[str, int]:
    return {key: 0 for key in _USABILITY_ORDER}


def build_report(classifications: list[RunInputClassification]) -> dict:
    """Aggregate per-run classifications into a structured report.

    Pure: no I/O. Same input -> same output.

    Shape (pinned by tests)::

        {
          "total_runs_discovered": int,
          "usable_full": int,
          "usable_recovery_only": int,
          "usable_minimal": int,
          "incomplete_no_snippets": int,
          "by_brief_id": {<brief-id>: {<usability>: int, ...}, ...},
          "runs": [<RunInputClassification.to_dict()>, ...],
        }

    The ``runs`` list is in the same deterministic order returned by
    ``discover_run_dirs`` (caller-driven). The ``by_brief_id`` inner dicts
    always carry all four usability keys so consumers can read counts
    without ``.get``.
    """

    report: dict = {
        "total_runs_discovered": len(classifications),
        "by_brief_id": {},
        "runs": [c.to_dict() for c in classifications],
    }
    report.update(_empty_buckets())

    for c in classifications:
        report[c.usability] = report.get(c.usability, 0) + 1
        brief_bucket = report["by_brief_id"].setdefault(
            c.brief_id, _empty_buckets()
        )
        brief_bucket[c.usability] = brief_bucket.get(c.usability, 0) + 1

    report["by_brief_id"] = dict(
        sorted(report["by_brief_id"].items(), key=lambda kv: kv[0])
    )
    return report


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


def _fmt_size(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "absent"
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f}MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.1f}GB"


def _classification_line(c_dict: dict) -> str:
    """Render one classification dict as a single stdout line.

    Uses ``- `` prefix for ``usable_*`` rows and ``! `` for
    ``incomplete_no_snippets`` so operators can scan failure rows quickly.
    """

    usability = c_dict.get("usability", "")
    if usability == USABILITY_INCOMPLETE:
        prefix = "!"
    else:
        prefix = "-"

    snippets_disp = _fmt_size(c_dict.get("snippets_size_bytes"))
    summaries_disp = _fmt_size(c_dict.get("summaries_size_bytes"))
    finals_disp = _fmt_size(c_dict.get("finals_size_bytes"))

    body = (
        f"{c_dict.get('brief_id', '')}/{c_dict.get('run_label', '')} | "
        f"{usability} | "
        f"snippets={snippets_disp} "
        f"summaries={summaries_disp} "
        f"finals={finals_disp}"
    )
    if usability == USABILITY_INCOMPLETE:
        reason = c_dict.get("incomplete_reason", "")
        if reason:
            body = f"{body} | reason={reason}"
    return f"{prefix} {body}"


def format_report(report: dict, *, limit: int) -> str:
    """Render the human-readable stdout block.

    Headline shows the four usability counts and a per-brief breakdown.
    Per-run listing is capped at ``limit`` (``0`` = no limit). The cap
    counts both usable and incomplete rows; incomplete rows are kept in
    the listing so operators see what's broken, not just what works.
    """

    lines: list[str] = []
    lines.append("=== Facial-Gate Input Discovery ===")
    total = int(report.get("total_runs_discovered", 0))
    lines.append(f"total_runs_discovered     : {total}")
    for key in _USABILITY_ORDER:
        lines.append(f"{key:<26}: {int(report.get(key, 0))}")
    lines.append("")

    lines.append("By brief-id:")
    by_brief = report.get("by_brief_id") or {}
    if not by_brief:
        lines.append("  (no briefs)")
    else:
        for brief_id in sorted(by_brief.keys()):
            counts = by_brief[brief_id]
            count_disp = "  ".join(
                f"{k}={int(counts.get(k, 0))}" for k in _USABILITY_ORDER
            )
            lines.append(f"  {brief_id:<24}: {count_disp}")
    lines.append("")

    runs = list(report.get("runs") or [])
    if not runs:
        lines.append("Runs:")
        lines.append("  (no run-dirs discovered)")
        return "\n".join(lines)

    if limit and limit > 0:
        shown = runs[:limit]
        remainder = max(0, len(runs) - limit)
    else:
        shown = runs
        remainder = 0

    lines.append("Runs:")
    for c_dict in shown:
        lines.append(_classification_line(c_dict))
    if remainder > 0:
        lines.append(f"... and {remainder} more")

    return "\n".join(lines)


def _pick_recommended(
    classifications: list[RunInputClassification],
) -> Optional[RunInputClassification]:
    """Pick the highest-quality usable run for the recommended invocation.

    Quality preference: ``usable_full`` > ``usable_recovery_only`` >
    ``usable_minimal``. Within a quality bucket, "most recent" is the
    lexicographically latest run-dir name (timestamps in this repo are
    ISO8601 with zero-padded fields, so lexicographic order matches
    chronological).
    """

    for tier in (
        USABILITY_FULL,
        USABILITY_RECOVERY_ONLY,
        USABILITY_MINIMAL,
    ):
        bucket = [c for c in classifications if c.usability == tier]
        if not bucket:
            continue
        bucket.sort(key=lambda c: f"{c.brief_id}/{c.run_label}")
        return bucket[-1]
    return None


def format_recommended_invocation(
    classification: Optional[RunInputClassification],
) -> str:
    """Render the "Recommended next step" stdout section.

    When ``classification`` is ``None`` (no usable runs anywhere), emit
    the spec's exact "No usable runs found" guidance pointing the
    operator at ``linkedin/session_orchestrator.py`` to capture a real
    run.

    When a usable run is provided, emit a copy-pasteable harness
    invocation. ``--profile-summaries`` is omitted for
    ``usable_recovery_only``; both ``--profile-summaries`` and
    ``--final-judgments`` are omitted for ``usable_minimal`` (and a
    quality-degraded note is added).
    """

    if classification is None:
        return "\n".join(
            [
                "Recommended next step:",
                "",
                "No usable runs found. The harness cannot be run against this repo today.",
                "Missing: snippets.jsonl in any output/runs/linkedin/<brief-id>/<run-dir>/.",
                "Capture a real LinkedIn run via linkedin/session_orchestrator.py first.",
            ]
        )

    run_dir = str(classification.run_dir)
    lines: list[str] = []
    lines.append(
        "Recommended next step (run the harness against the best usable run):"
    )
    lines.append("")
    lines.append("    python3 tools/experiments/facial_gate_experiment.py \\")
    lines.append("      --experiment \\")
    lines.append("      --variants baseline looser ternary \\")
    lines.append(f"      --snippets {run_dir}/snippets.jsonl \\")
    lines.append("      --brief config/<your-brief>.json \\")

    if classification.usability == USABILITY_FULL:
        lines.append(
            f"      --profile-summaries {run_dir}/profile_summaries.jsonl \\"
        )
        lines.append(
            f"      --final-judgments {run_dir}/final_judgments.jsonl \\"
        )
    elif classification.usability == USABILITY_RECOVERY_ONLY:
        lines.append(
            f"      --final-judgments {run_dir}/final_judgments.jsonl \\"
        )

    lines.append("      --max-candidates 100 \\")
    lines.append("      --json-out facial_experiment_summary.json")
    lines.append("")
    lines.append("Then evaluate against thresholds:")
    lines.append("")
    lines.append(
        "    python3 tools/experiments/recommend_facial_gate.py "
        "--summary facial_experiment_summary.json"
    )

    if classification.usability == USABILITY_MINIMAL:
        lines.append("")
        lines.append(
            "Note: this run is usable_minimal -- profile_summaries.jsonl and "
            "final_judgments.jsonl are absent, so the harness's recovery "
            "analysis and likely_false_negatives_under_variant heuristic "
            "will be quality-degraded. The recommendation tool will likely "
            "report KEEP_BINARY for lack of recovery signal."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON-out writer (atomic, never silently swallows errors)
# ---------------------------------------------------------------------------


def write_report_json(report: dict, path: Path) -> None:
    """Atomically write ``report`` to ``path`` as pretty JSON.

    Refuses to write to any path that resolves under a protected
    ``output/`` subtree (state, runs, market_intelligence, exports,
    archive, cache, debug, or the bare ``output/`` root). On rejection,
    prints a clear stderr message and raises ``SystemExit(1)``.

    Refuses to write to a path that exists and is a directory (mirrors
    slice 7's check).

    Writes to a sibling temp file then ``os.replace`` so a crash mid-write
    cannot leave a half-written report on disk.
    """

    try:
        path = Path(path)
        try:
            classification = classify_output_location(path)
        except Exception:
            classification = "external"
        if classification in _PROTECTED_LOCATIONS:
            print(
                f"ERROR: --json-out path resolves under a protected output "
                f"subtree ({classification}): {path}. Pick a path outside "
                f"output/state/, output/runs/, output/market_intelligence/, "
                f"output/exports/, output/archive/, output/cache/, "
                f"output/debug/.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if path.exists() and path.is_dir():
            print(
                f"ERROR: --json-out path is a directory: {path}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        parent = path.parent if path.parent != Path("") else Path(".")
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, sort_keys=True)
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
            f"ERROR: failed to write --json-out {path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace, *, runs_root: Optional[Path] = None) -> int:
    """Discover, classify, print, optionally write JSON, return exit code.

    ``runs_root`` is an injection seam for tests; the production CLI never
    sets it (defaults to ``shared.output_paths.RUNS_ROOT``).

    Returns:

    - ``0`` always for the discovery query (mirrors slice 6's empty-
      discovery exit code).
    - ``1`` only on hard input errors -- specifically: ``--json-out``
      resolves under a protected ``output/`` subtree or is an existing
      directory.
    """

    source = args.source
    discovered_dirs = discover_run_dirs(
        source=source,
        runs_root=runs_root,
        include_legacy=bool(args.include_legacy),
        include_unknown_brief=bool(args.include_unknown_brief),
        brief_id_filter=args.brief_id,
    )

    classifications = [classify_run_inputs(d) for d in discovered_dirs]

    if args.require_finals:
        classifications = [
            c
            for c in classifications
            if c.usability == USABILITY_INCOMPLETE
            or c.has_final_judgments
        ]
    if args.require_summaries:
        classifications = [
            c
            for c in classifications
            if c.usability == USABILITY_INCOMPLETE
            or c.has_profile_summaries
        ]

    report = build_report(classifications)
    print(format_report(report, limit=int(args.limit)))
    print()

    usable = [
        c for c in classifications if c.usability != USABILITY_INCOMPLETE
    ]
    recommended = _pick_recommended(usable) if usable else None
    print(format_recommended_invocation(recommended))

    if args.json_out:
        write_report_json(report, Path(args.json_out))

    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover real-run inputs for the facial-gate recommendation "
            "workflow. Walks output/runs/linkedin/<brief-id>/<run-dir>/ "
            "and reports which finalized runs carry the snippets, "
            "profile_summaries, and final_judgments artifacts the slice-5 "
            "harness needs. Analytical/debug only; never writes under "
            "output/."
        )
    )
    parser.add_argument(
        "--source",
        default="linkedin",
        choices=sorted(_DISCOVERY_SOURCES),
        help=(
            "Source adapter to discover for. LinkedIn-only in v1; the "
            "harness only handles LinkedIn snippets today."
        ),
    )
    parser.add_argument(
        "--brief-id",
        default=None,
        help=(
            "Optional substring filter against the brief-id directory "
            "name. Case-sensitive. No glob."
        ),
    )
    parser.add_argument(
        "--require-finals",
        action="store_true",
        help=(
            "Drop usable runs that lack final_judgments.jsonl from the "
            "listing. Incomplete runs (no snippets) are still listed so "
            "operators see what's broken."
        ),
    )
    parser.add_argument(
        "--require-summaries",
        action="store_true",
        help=(
            "Drop usable runs that lack profile_summaries.jsonl from the "
            "listing. Incomplete runs (no snippets) are still listed so "
            "operators see what's broken."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help=(
            "Cap the per-run stdout listing. 0 means no limit. Default 20."
        ),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help=(
            "Optional path to write the structured report as JSON. "
            "Default: nothing is written to disk. The path must NOT "
            "resolve under output/state/, output/runs/, "
            "output/market_intelligence/, output/exports/, "
            "output/archive/, output/cache/, or output/debug/ -- the "
            "tool exits 1 if it does."
        ),
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help=(
            "By default, run-dirs with the imported-...__legacy-N shape "
            "(produced by shared.output_paths.resolve_run_dir for legacy "
            "archive imports) are skipped. With this flag, they are "
            "listed."
        ),
    )
    parser.add_argument(
        "--include-unknown-brief",
        action="store_true",
        help=(
            "By default, output/runs/linkedin/unknown/ (legacy "
            "unsupervised runs without a real brief id) is skipped "
            "entirely. With this flag, those run-dirs are listed."
        ),
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
