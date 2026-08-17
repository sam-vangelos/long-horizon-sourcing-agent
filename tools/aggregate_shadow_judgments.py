#!/usr/bin/env python3
"""Offline batch aggregator for ``shadow_final_judgments.jsonl``.

Slice 4 of perplexity-evidence-augmentation. This is an *analytical / debug*
tool. It reads one or more ``shadow_final_judgments.jsonl`` files (declared
``ANALYTICAL_DEBUG`` in ``shared/runtime_state/artifacts.py``) and produces a
human summary on stdout, plus an optional structured JSON summary on disk.

Slice 6 adds ``--discover linkedin`` which globs
``output/state/linkedin/*/shadow_final_judgments.jsonl`` so operators can run a
real-run bakeoff without composing path conventions by hand. See
``docs/perplexity-evidence-bakeoff-runbook.md`` for the operator workflow.

Hard guarantees:

- No live LLM or network calls.
- No imports from ``linkedin/``, ``github/``, or ``market_intelligence/``.
  (``shared.output_paths`` is allowed; it is a path-convention module with no
  LLM, orchestration, or runtime-state side effects.)
- Never writes to ``output/``, ``runtime_state.sqlite3``,
  ``final_judgments.jsonl``, ``shadow_final_judgments.jsonl`` (it READS this),
  canonical projections, or any per-run state directory.
- Default invocation writes nothing to disk. ``--json-out <path>`` opts in to
  writing the structured summary to a path the user names.
- ``--discover linkedin`` reads ``output/state/linkedin/`` and prints to
  stdout; it never writes under ``output/``.
- Robust to partial / malformed / future-version rows: a JSONL line that
  fails to parse is counted as a parse failure and the tool continues.

CLI shape::

    python tools/aggregate_shadow_judgments.py [PATHS...]
        [--glob <pattern>]
        [--discover {linkedin}]
        [--changed-only | --save-flips-only]
        [--limit N]
        [--json-out PATH]

Exit codes: ``0`` on success regardless of decision-comparison outcomes;
``1`` only on hard input errors (no inputs at all -- no paths, no glob, no
discover -- conflicting flags, ``--json-out`` path is a directory, etc.).
``--discover linkedin`` that resolves zero files exits ``0``: discovery is a
query, not a hard input requirement.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Optional


# Make ``python tools/aggregate_shadow_judgments.py ...`` runnable without
# ``PYTHONPATH=.`` -- mirrors ``tools/check_repo_hygiene.py``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from shared.output_paths import source_state_root  # noqa: E402


_SHADOW_FILENAME = "shadow_final_judgments.jsonl"


_DISCOVERY_SOURCES = frozenset({"linkedin"})


# Treat all three save flavors as a save for slice-4 flip semantics. This
# mirrors the LinkedIn orchestrator's own treatment of the save set.
_SAVE_DECISIONS = frozenset({"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE"})


# Known ``external_evidence_status`` values that get their own counter in the
# summary. Anything else is bucketed under ``other_statuses``.
_KNOWN_STATUSES = frozenset(
    {
        "evidence_present",
        "skipped_no_trigger",
        "weak_citations",
        "quota_exhausted",
        "timeout",
        "parse_failure",
        "disabled_no_api_key",
        "disabled_by_config",
    }
)


# ---------------------------------------------------------------------------
# I/O: input resolution and record iteration
# ---------------------------------------------------------------------------


def resolve_input_paths(
    paths: list[str],
    glob_pattern: Optional[str],
    *,
    discovered: Optional[list[Path]] = None,
) -> list[Path]:
    """Resolve explicit paths, an optional glob, and optional discovered paths.

    - Explicit paths are taken as-is and filtered for existence.
    - ``glob_pattern`` is resolved relative to the current working directory.
    - ``discovered`` is an already-resolved list of ``Path`` objects (e.g.
      from ``discover_shadow_files``); they participate in the same
      dedup/missing-filter pipeline so a file produced by both an explicit
      argument and discovery is processed once.
    - Order preserves first-seen across the merged list: explicit first, then
      glob hits in glob order, then discovered in the order supplied.
    - Returns ``[]`` only when nothing was provided AND the glob matched
      nothing AND discovered was empty.
    """

    seen: set[Path] = set()
    resolved: list[Path] = []

    def _add(candidate: Path) -> None:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in seen:
            return
        if not candidate.exists() or not candidate.is_file():
            return
        seen.add(key)
        resolved.append(candidate)

    for raw in paths or []:
        _add(Path(raw))

    if glob_pattern:
        for hit in _glob.glob(glob_pattern, recursive=True):
            _add(Path(hit))

    for hit in discovered or []:
        _add(Path(hit))

    return resolved


def discover_shadow_files(source: str = "linkedin") -> list[Path]:
    """Return all ``shadow_final_judgments.jsonl`` files under a source state root.

    For ``source == "linkedin"``, walks ``output/state/linkedin/`` and returns
    every direct ``<brief>/shadow_final_judgments.jsonl`` it finds. Pure modulo
    the filesystem: no network, no LLM, no writes. Returns deduped paths
    sorted by their resolved string form for determinism.

    If the source state root does not exist (e.g. a fresh checkout, or in a
    test where the function has been monkeypatched to a missing dir), returns
    ``[]`` -- this is a discovery query, not a hard input requirement.
    """

    if source not in _DISCOVERY_SOURCES:
        raise ValueError(
            f"unsupported discovery source: {source!r} "
            f"(allowed: {sorted(_DISCOVERY_SOURCES)})"
        )

    try:
        root = source_state_root(source)
    except Exception:
        return []

    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return []

    seen: set[Path] = set()
    out: list[Path] = []
    for child in sorted(root_path.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        candidate = child / _SHADOW_FILENAME
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)

    out.sort(key=lambda p: str(p))
    return out


def iter_records(
    paths: Iterable[Path],
) -> Iterator[tuple[Optional[dict], Optional[str]]]:
    """Yield ``(parsed_dict, None)`` for good rows, ``(None, raw_line)`` for parse failures.

    Uses ``json.loads`` per line, NOT ``shared.storage.read_jsonl`` -- the
    latter would raise on a malformed line and break aggregation across files.
    Empty / whitespace-only lines are silently skipped (they are not parse
    failures).
    """

    for path in paths:
        try:
            fh = open(path, "r", encoding="utf-8")
        except OSError as exc:
            print(
                f"WARNING: failed to open {path}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        try:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    yield None, line
                    continue
                if not isinstance(parsed, dict):
                    yield None, line
                    continue
                yield parsed, None
        finally:
            fh.close()


# ---------------------------------------------------------------------------
# Aggregation (pure)
# ---------------------------------------------------------------------------


def _empty_summary() -> dict:
    return {
        "total_rows_read": 0,
        "total_parse_failures": 0,
        "total_compared": 0,
        "same_decision": 0,
        "decision_changed": 0,
        "reject_to_save": 0,
        "save_to_reject": 0,
        "path_only_changes": 0,
        "rationale_only_changes": 0,
        "confidence_delta_avg": 0.0,
        "confidence_delta_abs_avg": 0.0,
        "unavailable_external_evidence": 0,
        "weak_citations": 0,
        "quota_exhausted": 0,
        "timeout": 0,
        "parse_failure_provider": 0,
        "disabled_no_api_key": 0,
        "disabled_by_config": 0,
        "skipped_no_trigger": 0,
        "evidence_present_total": 0,
        "gate_trigger_breakdown": {},
        "evidence_refs_count_avg": 0.0,
        "identity_confidence_avg": 0.0,
        "other_statuses": {},
    }


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def aggregate(
    records: Iterable[tuple[Optional[dict], Optional[str]]],
) -> dict:
    """Consume ``(record, raw)`` tuples and return the summary dict.

    Pure: no I/O, no clocks, no randomness. Same input -> same output.
    """

    summary = _empty_summary()
    confidence_deltas: list[float] = []
    evidence_ref_counts: list[int] = []
    identity_confidences: list[float] = []
    gate_trigger_breakdown: dict[str, int] = {}
    other_statuses: dict[str, int] = {}

    for parsed, raw in records:
        if parsed is None:
            summary["total_parse_failures"] += 1
            continue

        summary["total_rows_read"] += 1

        status_raw = parsed.get("external_evidence_status", "")
        status = str(status_raw) if status_raw is not None else ""

        trigger_reason_raw = parsed.get("trigger_reason", "")
        trigger_reason = (
            str(trigger_reason_raw) if trigger_reason_raw is not None else ""
        )
        gate_trigger_breakdown[trigger_reason] = (
            gate_trigger_breakdown.get(trigger_reason, 0) + 1
        )

        # Status bucketing.
        if status == "evidence_present":
            summary["evidence_present_total"] += 1
            refs_count = _safe_int(parsed.get("evidence_refs_count"))
            evidence_ref_counts.append(refs_count)
        elif status == "skipped_no_trigger":
            summary["skipped_no_trigger"] += 1
        elif status == "weak_citations":
            summary["weak_citations"] += 1
        elif status == "quota_exhausted":
            summary["quota_exhausted"] += 1
        elif status == "timeout":
            summary["timeout"] += 1
        elif status == "parse_failure":
            summary["parse_failure_provider"] += 1
        elif status == "disabled_no_api_key":
            summary["disabled_no_api_key"] += 1
        elif status == "disabled_by_config":
            summary["disabled_by_config"] += 1
        else:
            other_statuses[status] = other_statuses.get(status, 0) + 1

        # Identity confidence: include when not None and numerically coercible.
        identity = parsed.get("identity_confidence")
        if identity is not None:
            try:
                identity_confidences.append(float(identity))
            except (TypeError, ValueError):
                pass

        # Decision-comparison stats only over rows where the diff was computed.
        diff = parsed.get("diff")
        if not isinstance(diff, dict) or not diff.get("computed"):
            summary["unavailable_external_evidence"] += 1
            continue

        summary["total_compared"] += 1
        decision_changed = bool(diff.get("decision_changed"))
        path_changed = bool(diff.get("path_changed"))
        rationale_changed = bool(diff.get("rationale_changed"))
        baseline_dec = str(diff.get("decision_baseline", ""))
        enriched_dec = str(diff.get("decision_enriched", ""))

        if decision_changed:
            summary["decision_changed"] += 1
            if (
                baseline_dec == "REJECT"
                and enriched_dec in _SAVE_DECISIONS
            ):
                summary["reject_to_save"] += 1
            elif (
                baseline_dec in _SAVE_DECISIONS
                and enriched_dec == "REJECT"
            ):
                summary["save_to_reject"] += 1
        else:
            summary["same_decision"] += 1
            if path_changed:
                summary["path_only_changes"] += 1
            elif rationale_changed:
                summary["rationale_only_changes"] += 1

        confidence_deltas.append(_safe_float(diff.get("confidence_delta", 0.0)))

    # Averages.
    if confidence_deltas:
        summary["confidence_delta_avg"] = sum(confidence_deltas) / len(
            confidence_deltas
        )
        summary["confidence_delta_abs_avg"] = sum(
            abs(d) for d in confidence_deltas
        ) / len(confidence_deltas)
    else:
        summary["confidence_delta_avg"] = 0.0
        summary["confidence_delta_abs_avg"] = 0.0

    summary["evidence_refs_count_avg"] = (
        (sum(evidence_ref_counts) / len(evidence_ref_counts))
        if evidence_ref_counts
        else 0.0
    )
    summary["identity_confidence_avg"] = (
        (sum(identity_confidences) / len(identity_confidences))
        if identity_confidences
        else 0.0
    )

    summary["gate_trigger_breakdown"] = dict(
        sorted(gate_trigger_breakdown.items(), key=lambda kv: kv[0])
    )
    summary["other_statuses"] = dict(
        sorted(other_statuses.items(), key=lambda kv: kv[0])
    )
    return summary


# ---------------------------------------------------------------------------
# Row selection for "Materially changed cases" listing
# ---------------------------------------------------------------------------


def _is_save_flip(diff: dict) -> bool:
    if not diff.get("computed"):
        return False
    if not diff.get("decision_changed"):
        return False
    baseline_dec = str(diff.get("decision_baseline", ""))
    enriched_dec = str(diff.get("decision_enriched", ""))
    if baseline_dec == "REJECT" and enriched_dec in _SAVE_DECISIONS:
        return True
    if baseline_dec in _SAVE_DECISIONS and enriched_dec == "REJECT":
        return True
    return False


def _is_changed_row(diff: dict) -> bool:
    if not diff.get("computed"):
        return False
    return bool(
        diff.get("decision_changed")
        or diff.get("path_changed")
        or diff.get("rationale_changed")
    )


def select_changed_rows(
    records: list[dict],
    *,
    save_flips_only: bool,
) -> list[dict]:
    """Filter and sort rows for the "Materially changed cases" listing.

    Sorted by ``abs(diff.confidence_delta)`` descending, with a stable
    secondary sort on ``candidate_name`` then ``profile_url`` so the same
    inputs always produce the same order.

    Rows where ``diff.computed is False`` are always excluded.
    """

    matched: list[dict] = []
    for row in records:
        diff = row.get("diff")
        if not isinstance(diff, dict):
            continue
        if save_flips_only:
            if not _is_save_flip(diff):
                continue
        else:
            if not _is_changed_row(diff):
                continue
        matched.append(row)

    def _sort_key(row: dict) -> tuple:
        diff = row.get("diff") or {}
        delta = _safe_float(diff.get("confidence_delta", 0.0))
        name = str(row.get("candidate_name", ""))
        url = str(row.get("profile_url", ""))
        # Negate abs(delta) so descending; tie-break ascending by name, url.
        return (-abs(delta), name, url)

    matched.sort(key=_sort_key)
    return matched


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


def _fmt_float(value: float, *, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _fmt_signed(value: float, *, decimals: int = 3) -> str:
    return f"{value:+.{decimals}f}"


_HEADLINE_FIELDS: tuple[str, ...] = (
    "total_compared",
    "reject_to_save",
    "save_to_reject",
    "path_only_changes",
    "rationale_only_changes",
    "unavailable_external_evidence",
    "weak_citations",
    "quota_exhausted",
)


def format_headline(summary: dict) -> str:
    """Render the eight-field headline block for stdout.

    Surfaces the operator-facing top-line numbers in this exact order:

    - total_compared
    - reject_to_save
    - save_to_reject
    - path_only_changes
    - rationale_only_changes
    - unavailable_external_evidence
    - weak_citations
    - quota_exhausted

    Stdout-only formatting: this does NOT mutate the summary dict and does
    NOT change ``--json-out`` shape.
    """

    label_width = max(len(label) for label in _HEADLINE_FIELDS)
    lines = ["=== Headline ==="]
    for field in _HEADLINE_FIELDS:
        value = summary.get(field, 0)
        lines.append(f"{field.ljust(label_width)} : {value}")
    return "\n".join(lines)


def format_summary(summary: dict) -> str:
    """Render a human-readable summary block for stdout."""

    rows = [
        ("total_rows_read", summary["total_rows_read"]),
        ("total_parse_failures", summary["total_parse_failures"]),
        ("total_compared", summary["total_compared"]),
        ("same_decision", summary["same_decision"]),
        ("decision_changed", summary["decision_changed"]),
        ("reject_to_save", summary["reject_to_save"]),
        ("save_to_reject", summary["save_to_reject"]),
        ("path_only_changes", summary["path_only_changes"]),
        ("rationale_only_changes", summary["rationale_only_changes"]),
        (
            "confidence_delta_avg",
            _fmt_signed(summary["confidence_delta_avg"]),
        ),
        (
            "confidence_delta_abs_avg",
            _fmt_float(summary["confidence_delta_abs_avg"]),
        ),
        (
            "unavailable_external_evidence",
            summary["unavailable_external_evidence"],
        ),
        ("weak_citations", summary["weak_citations"]),
        ("quota_exhausted", summary["quota_exhausted"]),
        ("timeout", summary["timeout"]),
        ("parse_failure_provider", summary["parse_failure_provider"]),
        ("disabled_no_api_key", summary["disabled_no_api_key"]),
        ("disabled_by_config", summary["disabled_by_config"]),
        ("skipped_no_trigger", summary["skipped_no_trigger"]),
        ("evidence_present_total", summary["evidence_present_total"]),
        (
            "evidence_refs_count_avg",
            _fmt_float(summary["evidence_refs_count_avg"], decimals=2),
        ),
        (
            "identity_confidence_avg",
            _fmt_float(summary["identity_confidence_avg"], decimals=3),
        ),
    ]

    label_width = max(len(label) for label, _ in rows)
    lines = ["=== Shadow Judgment Summary ==="]
    for label, value in rows:
        lines.append(f"{label.ljust(label_width)} : {value}")

    breakdown = summary.get("gate_trigger_breakdown") or {}
    lines.append("")
    lines.append("gate_trigger_breakdown:")
    if breakdown:
        max_key = max(len(k) if k else len("(empty)") for k in breakdown.keys())
        for key in sorted(breakdown.keys()):
            display_key = key if key else "(empty)"
            lines.append(
                f"  {display_key.ljust(max_key)} : {breakdown[key]}"
            )
    else:
        lines.append("  (none)")

    other = summary.get("other_statuses") or {}
    lines.append("")
    lines.append("other_statuses:")
    if other:
        max_key = max(len(k) if k else len("(empty)") for k in other.keys())
        for key in sorted(other.keys()):
            display_key = key if key else "(empty)"
            lines.append(f"  {display_key.ljust(max_key)} : {other[key]}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)


def format_changed_rows(rows: list[dict], *, limit: int) -> str:
    """Render the per-row "Materially changed cases" listing.

    ``limit == 0`` is treated as no-limit. ``limit > 0`` truncates and adds
    "and N more" when exceeded.
    """

    header = "=== Materially changed cases ==="
    if not rows:
        return header + "\n(no rows match the filter)"

    if limit and limit > 0:
        shown = rows[:limit]
        remainder = max(0, len(rows) - limit)
    else:
        shown = rows
        remainder = 0

    lines: list[str] = [header]
    for row in shown:
        diff = row.get("diff") or {}
        baseline_dec = str(diff.get("decision_baseline", "") or "?")
        enriched_dec = str(diff.get("decision_enriched", "") or "?")
        baseline_path = str(diff.get("path_baseline", "") or "")
        enriched_path = str(diff.get("path_enriched", "") or "")
        delta = _safe_float(diff.get("confidence_delta", 0.0))
        refs = _safe_int(row.get("evidence_refs_count", 0))
        trigger_reason = str(row.get("trigger_reason", "") or "")
        candidate_name = str(row.get("candidate_name", "") or "")
        profile_url = str(row.get("profile_url", "") or "")

        lines.append(
            "- "
            f"{candidate_name} | "
            f"{baseline_dec} -> {enriched_dec} | "
            f"path: {baseline_path} -> {enriched_path} | "
            f"conf: {_fmt_signed(delta)} | "
            f"refs={refs} | "
            f"trigger={trigger_reason} | "
            f"url={profile_url}"
        )

    if remainder > 0:
        lines.append(f"... and {remainder} more")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON-out writer (atomic, never silently swallows errors)
# ---------------------------------------------------------------------------


def write_summary_json(summary: dict, path: Path) -> None:
    """Atomically write ``summary`` to ``path`` as pretty JSON.

    Writes to a sibling temp file then ``os.replace`` to the final path so a
    crash mid-write cannot leave a half-written summary on disk. On any
    failure, raises ``SystemExit(1)`` after printing a clear error to stderr.
    """

    try:
        path = Path(path)
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
                json.dump(summary, fh, indent=2, sort_keys=True)
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


def run(args: argparse.Namespace) -> int:
    """Resolve inputs, aggregate, print, optionally write JSON, return exit code."""

    if args.changed_only and args.save_flips_only:
        print(
            "ERROR: --changed-only and --save-flips-only are mutually exclusive",
            file=sys.stderr,
        )
        return 1

    explicit_paths = list(args.paths or [])
    glob_pattern = args.glob
    discover_source = getattr(args, "discover", None)

    discovered_paths: list[Path] = []
    if discover_source:
        discovered_paths = discover_shadow_files(discover_source)

    # Empty-invocation rule: exit 1 only when the user gave NO signal at all
    # -- no paths, no glob, no --discover. ``--discover linkedin`` that
    # resolves zero files is a valid "nothing here yet" answer and exits 0.
    no_signal = (
        not explicit_paths
        and not glob_pattern
        and not discover_source
    )
    if no_signal:
        print(
            "ERROR: no input files resolved (pass paths or --glob or "
            "--discover; missing files are silently dropped)",
            file=sys.stderr,
        )
        return 1

    paths = resolve_input_paths(
        explicit_paths,
        glob_pattern,
        discovered=discovered_paths,
    )

    if discover_source and not paths and not explicit_paths and not glob_pattern:
        # Friendly empty-discovery message; not an error. Use the canonical
        # display form (``output/state/<source>/``) regardless of where
        # ``source_state_root`` has been redirected (tests can patch it).
        print(f"No shadow files found under output/state/{discover_source}/")
        return 0

    if not paths:
        # User passed paths/glob/discover but nothing resolved to a real file.
        print(
            "ERROR: no input files resolved (pass paths or --glob or "
            "--discover; missing files are silently dropped)",
            file=sys.stderr,
        )
        return 1

    parsed_records: list[dict] = []
    parse_failures: list[str] = []

    for parsed, raw in iter_records(paths):
        if parsed is None:
            parse_failures.append(raw or "")
        else:
            parsed_records.append(parsed)

    def _replay() -> Iterator[tuple[Optional[dict], Optional[str]]]:
        for row in parsed_records:
            yield row, None
        for raw in parse_failures:
            yield None, raw

    summary = aggregate(_replay())

    if discover_source:
        brief_dirs = {p.parent for p in discovered_paths}
        print(
            f"Discovered {len(discovered_paths)} shadow files across "
            f"{len(brief_dirs)} briefs"
        )
        print()

    print(format_headline(summary))
    print()
    print(format_summary(summary))

    if args.changed_only or args.save_flips_only:
        rows = select_changed_rows(
            parsed_records,
            save_flips_only=bool(args.save_flips_only),
        )
        print()
        print(format_changed_rows(rows, limit=args.limit))

    if args.json_out:
        write_summary_json(summary, Path(args.json_out))

    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate one or more shadow_final_judgments.jsonl files and "
            "print a human summary. Analytical/debug only; never writes "
            "canonical state."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="One or more paths to shadow_final_judgments.jsonl files.",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help=(
            "Optional glob pattern (resolved relative to cwd, recursive ** "
            "supported)."
        ),
    )
    parser.add_argument(
        "--discover",
        default=None,
        choices=sorted(_DISCOVERY_SOURCES),
        help=(
            "Discover shadow files automatically by walking the source "
            "state root. ``--discover linkedin`` reads "
            "``output/state/linkedin/*/shadow_final_judgments.jsonl``. "
            "Discovered paths merge with explicit PATHS and --glob via the "
            "same dedup pipeline. See "
            "docs/perplexity-evidence-bakeoff-runbook.md."
        ),
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help=(
            "After the summary, list rows where decision/path/rationale "
            "changed. Mutually exclusive with --save-flips-only."
        ),
    )
    parser.add_argument(
        "--save-flips-only",
        action="store_true",
        help=(
            "After the summary, list only REJECT<->save flips. Mutually "
            "exclusive with --changed-only."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help=(
            "Cap the per-row listing length. 0 means no limit. Default 50. "
            "Only meaningful with --changed-only or --save-flips-only."
        ),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help=(
            "Optional path to write the structured summary as JSON. "
            "Default: nothing is written to disk."
        ),
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
