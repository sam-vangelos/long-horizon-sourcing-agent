#!/usr/bin/env python3
"""Step B hold report tool for the perplexity-evidence-augmentation feature.

The Step B hold checklist (produced operationally, not yet committed as a
doc) requires the operator to enable
``LINKEDIN_FACIAL_BORDERLINE_ENABLED=true`` for one or more real LinkedIn
run cycles, then evaluate each completed run for hold-eligibility. This
slice replaces the manual spreadsheet flow with one command:

    python3 tools/step_b_hold_report.py [--brief-id ID] [--limit N]
                                         [--report-out PATH]
                                         [--include-legacy]
                                         [--include-unknown-brief]

The tool walks ``output/runs/linkedin/<brief-id>/<run-dir>/`` and emits a
per-run verdict of ``PASS``, ``BLOCKED``, or ``NOT_APPLICABLE``, plus an
aggregate summary suitable for tracking the §4a "≥3 PASS runs across ≥2
briefs" gate from the Step B hold checklist.

Why the borderline counter is reconstructed from disk
----------------------------------------------------

The slice-16 ``BiasMonitor._facial_borderline_counts`` counter is
session-local and **not** persisted by ``BiasMonitor.save_checkpoint``
(see ``shared/bias_controls.py:481-498``: ``save_checkpoint`` writes
``decisions`` and ``alerts_fired`` only). Therefore
``bias_monitor-<brief-id>.json`` does NOT carry a usable
``facial_borderline_count`` for completed runs.

The on-disk source of truth is the slice-13 rationale canary in
``facial_judgments.jsonl``: every borderline-aliased decision lands with
rationale prefixed ``"[BORDERLINE\u2192YES alias]"`` (note the unicode
right-arrow ``\u2192``, not ASCII ``->``). See
``linkedin/orchestrator.py:586-589`` for the persistence boundary.

Flag-off canary
---------------

When ``LINKEDIN_FACIAL_BORDERLINE_ENABLED=False`` and the model emits
``FACIAL_BORDERLINE`` anyway, the orchestrator converts the decision to
``parse_failure_decision(reason="facial_borderline_under_flag_off", ...)``
(see ``linkedin/orchestrator.py:597-609``). The persisted row carries the
literal ``"facial_borderline_under_flag_off"`` substring in either the
``rationale`` or the parse-failure detail. Any positive count means the
flag was off mid-run and the data is untrustworthy for the hold sample.

Hard guarantees
---------------

- No live LLM or network calls. No imports from ``linkedin/``,
  ``github/``, ``market_intelligence/``, or any production module.
  Stdlib + ``shared.output_paths`` only.
- Never writes under ``output/state/``, ``output/runs/``,
  ``output/market_intelligence/``, ``output/exports/``,
  ``output/archive/``, ``output/cache/``, or ``output/debug/`` --
  ``--report-out`` rejects any path resolving under those subtrees.
- Default invocation writes nothing to disk. ``--report-out <path>``
  opts in to writing the structured JSON report to a path the operator
  names outside ``output/``.
- Empty discovery (zero LinkedIn runs) exits ``0`` with the
  ``"No completed LinkedIn runs found"`` message. ``SystemExit(1)`` is
  reserved for hard input errors (``--report-out`` resolves under a
  protected path or is an existing directory).
- LinkedIn-only for v1 (mirrors slice 11's ``--source linkedin`` choice
  in ``tools/discover_facial_gate_inputs.py``).

Companions in the perplexity-evidence-augmentation feature
----------------------------------------------------------

- ``tools/discover_facial_gate_inputs.py`` -- slice 11 input discoverer
  the harness reads. SAME shape, DIFFERENT question (does this run carry
  the harness inputs vs. is this run hold-eligible).
- ``tools/aggregate_shadow_judgments.py`` -- slice 4 shadow-judgment
  discovery for the Perplexity bakeoff.
- ``shared/bias_controls.py:_register_facial_decision`` -- the
  in-session borderline counter this tool reconstructs from disk.
- ``linkedin/orchestrator.py:_normalize_facial_decision_for_persistence``
  -- the persistence boundary that creates the
  ``[BORDERLINE\u2192YES alias]`` and
  ``facial_borderline_under_flag_off`` canaries this tool reads.
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

_FACIAL_JUDGMENTS_FILENAME = "facial_judgments.jsonl"
_SNIPPETS_FILENAME = "snippets.jsonl"
_PROFILE_SUMMARIES_FILENAME = "profile_summaries.jsonl"
_FINAL_JUDGMENTS_FILENAME = "final_judgments.jsonl"


# Borderline alias rationale prefix written by
# linkedin/orchestrator.py:_normalize_facial_decision_for_persistence
# under LINKEDIN_FACIAL_BORDERLINE_ENABLED=True. Note the unicode
# right-arrow (U+2192), NOT an ASCII "->": this is the canonical canary.
_BORDERLINE_ALIAS_PREFIX = "[BORDERLINE\u2192YES alias]"

# Substring written into the parse-failure detail / rationale when the
# model emits FACIAL_BORDERLINE under LINKEDIN_FACIAL_BORDERLINE_ENABLED
# =False. Either appearance proves the flag was off mid-run.
_FLAG_OFF_CANARY_SUBSTRING = "facial_borderline_under_flag_off"


# Output subtrees the ``--report-out`` writer refuses to write into.
# Mirrors slice 11 verbatim. ``classify_output_location`` returning any
# of these blocks the write; only ``"external"`` (path is outside
# ``output/``) is allowed.
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


# Verdicts. Pinned by tests; downstream consumers (the aggregate
# summary, the per-run formatter) read directly off these constants.
VERDICT_PASS = "PASS"
VERDICT_BLOCKED = "BLOCKED"
VERDICT_NOT_APPLICABLE = "NOT_APPLICABLE"

_VERDICT_ORDER: tuple[str, ...] = (
    VERDICT_PASS,
    VERDICT_BLOCKED,
    VERDICT_NOT_APPLICABLE,
)


# Blocking-gate names. These are the values populated into
# ``RunHoldReport.blocking_gate`` and surfaced in the per-run listing so
# the operator can see which check failed without re-deriving it.
GATE_NONE = ""
GATE_FLAG_NEVER_ON = "flag_never_on"
GATE_FLAG_OFF_CANARY_PRESENT = "flag_off_canary_present"
GATE_MISSING_FACIAL_ARTIFACT = "missing_facial_artifact"
GATE_MISSING_SNIPPETS = "missing_snippets"
GATE_MISSING_PROFILE_SUMMARIES = "missing_profile_summaries"
GATE_FINAL_JUDGMENTS_ZERO_BYTES = "final_judgments_zero_bytes"
GATE_INSUFFICIENT_FACIAL_DECISIONS = "insufficient_facial_decisions"
GATE_NO_BORDERLINE_OBSERVED = "no_borderline_observed"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunHoldReport:
    """Per-run hold-eligibility report.

    All fields named in the slice spec's "Required per-run report shape"
    table are populated. Pure data: no I/O happens on construction; the
    classifier populates this from filesystem inspection.

    ``blocking_gate`` is the empty string for ``PASS`` and
    ``NOT_APPLICABLE``; for ``BLOCKED`` it carries one of the ``GATE_*``
    constants identifying the first gate that failed.
    """

    source: str
    brief_id: str
    run_dir: Path
    run_label: str
    flag_was_on: bool
    flag_off_canary_count: int
    total_facial_decisions: int
    facial_borderline_count: int
    facial_borderline_rate: float
    facial_yes_count: int
    facial_non_skip_total: int
    facial_open_rate: float
    snippets_present: bool
    profile_summaries_present: bool
    final_judgments_present_and_nonempty: bool
    facial_judgments_present: bool
    bias_monitor_checkpoint_present: bool
    eligible_for_step_b_hold: bool
    verdict: str
    blocking_gate: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_dir"] = str(self.run_dir)
        return d


# ---------------------------------------------------------------------------
# Pure helpers (filesystem-bound but no network / no LLM / no writes)
# ---------------------------------------------------------------------------


def _iter_jsonl_rows(path: Path):
    """Yield parsed JSON objects from a JSONL file, skipping malformed lines.

    Tolerates malformed lines per the spec's "tolerate per-line; return
    counts from valid lines" rule. Does not raise on JSON errors; bad
    lines silently disappear from the count. Does not raise on missing
    files either -- caller is responsible for the presence check.
    """

    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(obj, dict):
                    yield obj
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
        return


def _count_borderline_alias_rationales(facial_path: Path) -> int:
    """Count rows whose ``rationale`` starts with the unicode borderline canary.

    The canonical canary is ``"[BORDERLINE\u2192YES alias]"`` (unicode
    right-arrow U+2192). An ASCII ``->`` form is NOT counted -- the
    orchestrator only writes the unicode form, so an ASCII match would
    be a false signal.
    """

    count = 0
    for row in _iter_jsonl_rows(facial_path):
        rationale = row.get("rationale")
        if isinstance(rationale, str) and rationale.startswith(
            _BORDERLINE_ALIAS_PREFIX
        ):
            count += 1
    return count


def _count_flag_off_canary(facial_path: Path) -> int:
    """Count rows that carry the flag-off canary substring.

    The canary appears in ``rationale`` (parse-failure detail prose) on
    rows produced by ``parse_failure_decision(reason=
    "facial_borderline_under_flag_off", ...)``. We also accept the
    substring appearing in a top-level ``decision`` field defensively in
    case a future change persists the reason there.
    """

    count = 0
    for row in _iter_jsonl_rows(facial_path):
        rationale = row.get("rationale", "")
        decision = row.get("decision", "")
        rationale_str = rationale if isinstance(rationale, str) else ""
        decision_str = decision if isinstance(decision, str) else ""
        if (
            _FLAG_OFF_CANARY_SUBSTRING in rationale_str
            or _FLAG_OFF_CANARY_SUBSTRING in decision_str
        ):
            count += 1
    return count


def _count_facial_decisions(facial_path: Path) -> tuple[int, int, int]:
    """Return ``(total, facial_yes, non_skip_total)`` over valid rows.

    - ``total``: every JSON-parseable dict row counts (matches the wc -l
      semantics on a clean file but is robust to malformed lines).
    - ``facial_yes``: rows with ``decision == "FACIAL_YES"``. This is
      post-alias because the alias rewrites the persisted decision to
      ``FACIAL_YES`` (see
      ``linkedin/orchestrator.py:_normalize_facial_decision_for_persistence``).
    - ``non_skip_total``: rows with ``decision != "FACIAL_SKIP"``. The
      open-rate denominator excludes skips, mirroring slice 16's
      ``facial_open_rate`` semantics.

    Rows with a missing or non-string ``decision`` field count toward
    ``total`` but are excluded from ``facial_yes`` and counted as
    non-skip (since they cannot be confirmed as a SKIP).
    """

    total = 0
    facial_yes = 0
    non_skip_total = 0
    for row in _iter_jsonl_rows(facial_path):
        total += 1
        decision = row.get("decision")
        if isinstance(decision, str):
            if decision == "FACIAL_YES":
                facial_yes += 1
            if decision != "FACIAL_SKIP":
                non_skip_total += 1
        else:
            non_skip_total += 1
    return total, facial_yes, non_skip_total


def _is_legacy_name(name: str) -> bool:
    """Match the canonical legacy-import shape from ``resolve_run_dir``.

    Mirrors slice 11's helper exactly:
    ``shared/output_paths.py:resolve_run_dir`` produces
    ``imported-{run_stamp}__legacy-{N}`` for legacy archive imports. We
    require BOTH the ``imported-`` prefix AND the ``__legacy-`` infix so
    a hypothetical future ``imported-something-else__run-1`` would not
    be mistakenly skipped.
    """

    return name.startswith("imported-") and "__legacy-" in name


def discover_runs(
    *,
    source: str = "linkedin",
    runs_root: Optional[Path] = None,
    include_legacy: bool,
    include_unknown_brief: bool,
    brief_id_filter: Optional[str],
) -> list[Path]:
    """Walk the runs root and return every run-shaped directory.

    Mirrors slice 11's ``discover_run_dirs`` exactly so the two tools
    classify the same set of run-dirs given the same flags. The
    ``runs_root`` parameter is the test-isolation seam; production CLI
    leaves it ``None`` and the tool defaults to
    ``shared.output_paths.RUNS_ROOT``.

    Filter rules:

    - ``include_legacy=False`` (default) skips run-dir names matching
      the ``imported-...__legacy-...`` shape.
    - ``include_unknown_brief=False`` (default) skips the entire
      ``<runs_root>/<source>/unknown/`` subtree.
    - ``brief_id_filter`` is a substring match against the brief-id
      directory name (case-sensitive, no glob). ``None`` disables the
      filter.

    Returns a deterministically sorted, deduplicated list. Sort key is
    ``"<brief-id>/<run-label>"`` so output is stable across runs.
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


def _classify_run(run_dir: Path) -> RunHoldReport:
    """Inspect a single run-dir and produce a ``RunHoldReport``.

    Pure modulo the filesystem: ``Path.is_file()``, ``Path.stat()``, and
    line-by-line JSON parsing of ``facial_judgments.jsonl`` only. Never
    writes.

    Eligibility logic (pinned by tests):

    A run is eligible iff ALL of:

    - ``flag_was_on`` is True (otherwise borderline observability is
      structurally absent -> ``NOT_APPLICABLE``).
    - ``flag_off_canary_count == 0`` (any positive count means the flag
      was off mid-run -> ``BLOCKED``).
    - ``snippets_present``, ``profile_summaries_present``,
      ``final_judgments_present_and_nonempty``,
      ``facial_judgments_present`` all True (otherwise the run is not
      analyzable end-to-end -> ``BLOCKED``).
    - ``total_facial_decisions >= 1`` (sample-size minimum from the
      Step B hold checklist's per-run §2a stability gate).

    Verdict assignment:

    - ``NOT_APPLICABLE``: ``flag_was_on`` is False. The run is
      informational; it predates flag-on or was a binary run.
    - ``BLOCKED``: ``flag_was_on`` is True but eligibility fails on the
      canary or an artifact gate. ``blocking_gate`` carries the failed
      gate name.
    - ``PASS``: eligible AND ``facial_borderline_count >= 1``. This
      single run is suitable to count toward the Step B hold sample.
      The hold checklist requires multiple PASS runs across briefs
      (§4a: >=3 runs across >=2 briefs).
    """

    run_dir = Path(run_dir)
    facial_path = run_dir / _FACIAL_JUDGMENTS_FILENAME
    snippets_path = run_dir / _SNIPPETS_FILENAME
    summaries_path = run_dir / _PROFILE_SUMMARIES_FILENAME
    finals_path = run_dir / _FINAL_JUDGMENTS_FILENAME

    facial_present = facial_path.is_file()
    snippets_present = snippets_path.is_file()
    profile_summaries_present = summaries_path.is_file()
    final_judgments_present_and_nonempty = (
        finals_path.is_file() and finals_path.stat().st_size > 0
    )

    brief_id = run_dir.parent.name if run_dir.parent is not None else ""
    source = (
        run_dir.parent.parent.name
        if run_dir.parent is not None and run_dir.parent.parent is not None
        else ""
    )

    bias_monitor_path = run_dir / f"bias_monitor-{brief_id}.json"
    bias_monitor_checkpoint_present = bias_monitor_path.is_file()

    if facial_present:
        borderline_count = _count_borderline_alias_rationales(facial_path)
        flag_off_canary_count = _count_flag_off_canary(facial_path)
        total, facial_yes, non_skip_total = _count_facial_decisions(facial_path)
    else:
        borderline_count = 0
        flag_off_canary_count = 0
        total = 0
        facial_yes = 0
        non_skip_total = 0

    # The flag was demonstrably on at some point during this run iff we
    # observe either the borderline alias canary (only fires under
    # flag-on) OR the flag-off parse-failure canary (only fires when the
    # flag was off but the model emitted BORDERLINE). The flag-off canary
    # on its own is not "flag was on" -- it is "flag was off and the
    # model produced BORDERLINE" -- but for the purposes of "is this run
    # part of the Step B hold sample at all?" any borderline-related
    # signal proves the run engaged the borderline machinery and so the
    # operator intended it as a hold-sample run.
    flag_was_on = bool(borderline_count > 0 or flag_off_canary_count > 0)

    if total > 0:
        facial_borderline_rate = float(borderline_count) / float(total)
    else:
        facial_borderline_rate = 0.0
    if non_skip_total > 0:
        facial_open_rate = float(facial_yes) / float(non_skip_total)
    else:
        facial_open_rate = 0.0

    blocking_gate = GATE_NONE
    eligible = False
    verdict = VERDICT_NOT_APPLICABLE

    if not flag_was_on:
        verdict = VERDICT_NOT_APPLICABLE
        blocking_gate = GATE_FLAG_NEVER_ON
    else:
        if flag_off_canary_count > 0:
            verdict = VERDICT_BLOCKED
            blocking_gate = GATE_FLAG_OFF_CANARY_PRESENT
        elif not facial_present:
            verdict = VERDICT_BLOCKED
            blocking_gate = GATE_MISSING_FACIAL_ARTIFACT
        elif not snippets_present:
            verdict = VERDICT_BLOCKED
            blocking_gate = GATE_MISSING_SNIPPETS
        elif not profile_summaries_present:
            verdict = VERDICT_BLOCKED
            blocking_gate = GATE_MISSING_PROFILE_SUMMARIES
        elif not final_judgments_present_and_nonempty:
            verdict = VERDICT_BLOCKED
            blocking_gate = GATE_FINAL_JUDGMENTS_ZERO_BYTES
        elif total < 1:
            verdict = VERDICT_BLOCKED
            blocking_gate = GATE_INSUFFICIENT_FACIAL_DECISIONS
        else:
            eligible = True
            if borderline_count >= 1:
                verdict = VERDICT_PASS
                blocking_gate = GATE_NONE
            else:
                # Eligible-shape (artifacts present, flag was on, no
                # flag-off canary, sample size met) but no borderline
                # signal observed. Spec PASS rule requires
                # borderline_count >= 1, so this is BLOCKED on the
                # signal gate.
                verdict = VERDICT_BLOCKED
                blocking_gate = GATE_NO_BORDERLINE_OBSERVED
                eligible = False

    return RunHoldReport(
        source=source,
        brief_id=brief_id,
        run_dir=run_dir,
        run_label=run_dir.name,
        flag_was_on=flag_was_on,
        flag_off_canary_count=flag_off_canary_count,
        total_facial_decisions=total,
        facial_borderline_count=borderline_count,
        facial_borderline_rate=facial_borderline_rate,
        facial_yes_count=facial_yes,
        facial_non_skip_total=non_skip_total,
        facial_open_rate=facial_open_rate,
        snippets_present=snippets_present,
        profile_summaries_present=profile_summaries_present,
        final_judgments_present_and_nonempty=final_judgments_present_and_nonempty,
        facial_judgments_present=facial_present,
        bias_monitor_checkpoint_present=bias_monitor_checkpoint_present,
        eligible_for_step_b_hold=eligible,
        verdict=verdict,
        blocking_gate=blocking_gate,
    )


def _empty_verdict_buckets() -> dict[str, int]:
    return {key: 0 for key in _VERDICT_ORDER}


def build_aggregate_summary(reports: list[RunHoldReport]) -> dict:
    """Aggregate per-run verdicts into a structured summary.

    Pure: no I/O. Same input -> same output.

    Shape (pinned by tests)::

        {
          "total_runs_discovered": int,
          "PASS": int,
          "BLOCKED": int,
          "NOT_APPLICABLE": int,
          "by_brief_id": {<brief-id>: {<verdict>: int, ...}, ...},
          "pass_briefs": [<brief-id>, ...],   # sorted
          "pass_runs_total": int,
          "hold_threshold_met": bool,         # >=3 PASS runs across >=2 briefs
        }

    The ``by_brief_id`` inner dicts always carry all three verdict keys
    so consumers can read counts without ``.get``.
    """

    summary: dict = {
        "total_runs_discovered": len(reports),
        "by_brief_id": {},
    }
    summary.update(_empty_verdict_buckets())

    pass_briefs: set[str] = set()
    for r in reports:
        summary[r.verdict] = summary.get(r.verdict, 0) + 1
        bucket = summary["by_brief_id"].setdefault(
            r.brief_id, _empty_verdict_buckets()
        )
        bucket[r.verdict] = bucket.get(r.verdict, 0) + 1
        if r.verdict == VERDICT_PASS:
            pass_briefs.add(r.brief_id)

    summary["by_brief_id"] = dict(
        sorted(summary["by_brief_id"].items(), key=lambda kv: kv[0])
    )
    summary["pass_briefs"] = sorted(pass_briefs)
    summary["pass_runs_total"] = int(summary.get(VERDICT_PASS, 0))
    summary["hold_threshold_met"] = bool(
        summary["pass_runs_total"] >= 3 and len(pass_briefs) >= 2
    )
    return summary


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


def _per_run_line(report: RunHoldReport) -> str:
    """Render one ``RunHoldReport`` as a single stdout line.

    ``-`` prefix for ``PASS`` and ``NOT_APPLICABLE`` rows; ``!`` prefix
    for ``BLOCKED`` rows so operators can scan failure rows quickly.
    Always includes ``flag_on`` and the borderline rate; for ``BLOCKED``
    rows the ``blocking gate`` is appended so the operator does not
    have to re-derive which check failed.
    """

    if report.verdict == VERDICT_BLOCKED:
        prefix = "!"
    else:
        prefix = "-"

    rate = report.facial_borderline_rate
    body = (
        f"{report.brief_id}/{report.run_label} | "
        f"{report.verdict} | "
        f"flag_on={report.flag_was_on} "
        f"borderline_count={report.facial_borderline_count}/"
        f"{report.total_facial_decisions} "
        f"(rate={rate:.4f}) "
        f"flag_off_canary={report.flag_off_canary_count}"
    )
    if report.verdict == VERDICT_BLOCKED and report.blocking_gate:
        body = f"{body} | blocking gate: {report.blocking_gate}"
    elif report.verdict == VERDICT_NOT_APPLICABLE and report.blocking_gate:
        body = f"{body} | reason: {report.blocking_gate}"
    return f"{prefix} {body}"


def format_per_run_listing(
    reports: list[RunHoldReport], *, limit: int
) -> str:
    """Render the per-run listing block.

    Listing is capped at ``limit`` (``0`` = no limit). The cap counts
    every verdict; we keep ``BLOCKED`` rows in the listing so the
    operator sees what is broken, not just what works.
    """

    lines: list[str] = []
    lines.append("Per-run:")
    if not reports:
        lines.append("  (no run-dirs discovered)")
        return "\n".join(lines)

    if limit and limit > 0:
        shown = reports[:limit]
        remainder = max(0, len(reports) - limit)
    else:
        shown = reports
        remainder = 0

    for r in shown:
        lines.append(_per_run_line(r))
    if remainder > 0:
        lines.append(f"... and {remainder} more")
    return "\n".join(lines)


def format_aggregate_summary(summary: dict) -> str:
    """Render the human-readable headline + by-brief block.

    Headline shows the three verdict counts and a per-brief breakdown.
    Always closes with a fixed footnote pointing the operator at the
    Step B hold checklist's §4a multi-run gate so a single ``PASS`` is
    not mistaken for hold completion.
    """

    lines: list[str] = []
    lines.append("=== Step B Hold Report ===")
    total = int(summary.get("total_runs_discovered", 0))
    lines.append(f"total_runs_discovered      : {total}")
    for key in _VERDICT_ORDER:
        lines.append(f"{key:<27}: {int(summary.get(key, 0))}")
    lines.append(
        f"hold_threshold_met (>=3 runs / >=2 briefs): "
        f"{bool(summary.get('hold_threshold_met', False))}"
    )

    lines.append("")
    lines.append("By brief-id:")
    by_brief = summary.get("by_brief_id") or {}
    if not by_brief:
        lines.append("  (no briefs)")
    else:
        for brief_id in sorted(by_brief.keys()):
            counts = by_brief[brief_id]
            count_disp = "  ".join(
                f"{k}={int(counts.get(k, 0))}" for k in _VERDICT_ORDER
            )
            lines.append(f"  {brief_id:<24}: {count_disp}")
    return "\n".join(lines)


_HOLD_FOOTNOTE = (
    "Note: 'PASS' means this single run is suitable to count toward the "
    "Step B hold sample. The Step B hold checklist requires >=3 PASS runs "
    "across >=2 briefs (sec 4a). See plans/perplexity-evidence-augmentation.md "
    "and the Step B hold checklist artifact in conversation."
)


# ---------------------------------------------------------------------------
# JSON-out writer (atomic, never silently swallows errors)
# ---------------------------------------------------------------------------


def write_report_json(
    reports: list[RunHoldReport], summary: dict, path: Path
) -> None:
    """Atomically write the structured report to ``path`` as pretty JSON.

    Mirrors slice 11's ``write_report_json``: refuses paths under any
    protected ``output/`` subtree (state, runs, market_intelligence,
    exports, archive, cache, debug, or the bare ``output/`` root) and
    refuses to overwrite an existing directory. On rejection, prints a
    clear stderr message and raises ``SystemExit(1)``.

    Writes to a sibling temp file then ``os.replace`` so a crash mid-
    write cannot leave a half-written report on disk.

    Output JSON shape::

        {
          "summary": <build_aggregate_summary output>,
          "runs": [<RunHoldReport.to_dict()>, ...],
        }
    """

    try:
        path = Path(path)
        try:
            classification = classify_output_location(path)
        except Exception:
            classification = "external"
        if classification in _PROTECTED_LOCATIONS:
            print(
                f"ERROR: --report-out path resolves under a protected output "
                f"subtree ({classification}): {path}. Pick a path outside "
                f"output/state/, output/runs/, output/market_intelligence/, "
                f"output/exports/, output/archive/, output/cache/, "
                f"output/debug/.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if path.exists() and path.is_dir():
            print(
                f"ERROR: --report-out path is a directory: {path}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        parent = path.parent if path.parent != Path("") else Path(".")
        parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "summary": summary,
            "runs": [r.to_dict() for r in reports],
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


def run(
    args: argparse.Namespace, *, runs_root: Optional[Path] = None
) -> int:
    """Discover, classify, print, optionally write JSON, return exit code.

    ``runs_root`` is the test-injection seam; production CLI never sets
    it (defaults to ``shared.output_paths.RUNS_ROOT``).

    Returns:

    - ``0`` always for the report query (mirrors slice 11's empty-
      discovery exit code).
    - ``1`` only on hard input errors -- specifically: ``--report-out``
      resolves under a protected ``output/`` subtree or is an existing
      directory.
    """

    source = args.source
    discovered_dirs = discover_runs(
        source=source,
        runs_root=runs_root,
        include_legacy=bool(args.include_legacy),
        include_unknown_brief=bool(args.include_unknown_brief),
        brief_id_filter=args.brief_id,
    )

    if not discovered_dirs:
        print("No completed LinkedIn runs found")
        if args.report_out:
            empty_summary = build_aggregate_summary([])
            write_report_json([], empty_summary, Path(args.report_out))
        return 0

    reports = [_classify_run(d) for d in discovered_dirs]
    summary = build_aggregate_summary(reports)

    print(format_aggregate_summary(summary))
    print()
    print(format_per_run_listing(reports, limit=int(args.limit)))
    print()
    print(_HOLD_FOOTNOTE)

    if args.report_out:
        write_report_json(reports, summary, Path(args.report_out))

    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step B hold report: walks output/runs/linkedin/<brief-id>/"
            "<run-dir>/ and emits a per-run PASS/BLOCKED/NOT_APPLICABLE "
            "verdict for the LINKEDIN_FACIAL_BORDERLINE_ENABLED hold "
            "checklist. Reconstructs the borderline count from the "
            "[BORDERLINE\u2192YES alias] rationale canary in "
            "facial_judgments.jsonl because BiasMonitor.save_checkpoint "
            "does not persist _facial_borderline_counts. Analytical/"
            "debug only; never writes under output/."
        )
    )
    parser.add_argument(
        "--source",
        default="linkedin",
        choices=sorted(_DISCOVERY_SOURCES),
        help=(
            "Source adapter to report on. LinkedIn-only in v1; the hold "
            "checklist only covers the LinkedIn facial gate today."
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
        "--limit",
        type=int,
        default=20,
        help=(
            "Cap the per-run stdout listing. 0 means no limit. Default 20."
        ),
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help=(
            "Optional path to write the structured hold report as JSON. "
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
