#!/usr/bin/env python3
"""Developer-facing comparison tool for the Perplexity evidence augmentation feature.

This is a *debug* tool. It exists so engineers can iterate on the prompt + the
evidence-shape contract on a single candidate without touching LinkedIn, the
browser, runtime state, or projections.

Hard guarantees:

- No live LinkedIn browsing. Reads a ``CandidateProfileSummary`` JSON document
  (file path or stdin via ``--profile-summary -``).
- No production writes. Nothing is written to ``output/``,
  ``runtime_state.sqlite3``, ``final_judgments.jsonl``,
  ``shadow_final_judgments.jsonl``, canonical projections, or any per-run
  state directory. Stdout only.
- Runs even when ``LINKEDIN_EXTERNAL_EVIDENCE_ENABLED`` is ``False``. The
  enable flag governs production behavior, not this tool. The tool consults
  ``shared.config.PERPLEXITY_API_KEY`` directly. If the key is empty AND
  ``--skip-external`` is not passed AND ``--evidence-fixture`` is not passed,
  the tool prints ``status: disabled_no_api_key`` and still runs the baseline
  judge.

CLI shape (see ``--help`` for full usage)::

    python tools/compare_external_evidence.py \\
        --profile-summary path/to/summary.json \\
        --brief config/brief-head-ai-lab-nyc-v2.json \\
        --evidence-fixture path/to/evidence.json

Flag interactions:

- ``--skip-external`` short-circuits the gate + provider entirely. It wins
  over both ``--evidence-fixture`` and ``--force-trigger``: when ``--skip-external``
  is passed the tool prints ``status: skipped_by_flag`` and does not load the
  fixture or call the gate.
- ``--evidence-fixture <path>`` loads an ``ExternalCandidateEvidence`` from
  disk INSTEAD of calling Perplexity. When present (and ``--skip-external``
  is absent) the gate is also bypassed: status is ``fixture_loaded``.
- ``--force-trigger {sparse_profile,academic_context}`` bypasses the gate
  heuristic and goes straight to the provider with a synthetic
  ``TriggerDecision``. Only meaningful when neither ``--skip-external`` nor
  ``--evidence-fixture`` is set.

Exit codes:

- ``0`` on success regardless of whether baseline and enriched decisions
  agreed.
- ``1`` on hard input errors only (file not found, JSON parse error on
  inputs, brief load failure).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import shared.config as config
from shared.brief_loader import load_brief
from shared.external_evidence import (
    fetch_external_candidate_evidence,
    should_request_external_evidence,
)
from shared.external_evidence.shadow_writer import compute_judgment_diff
from shared.judger import (
    full_judge,
    full_judge_with_external_evidence,
)
from shared.schemas import (
    CandidateProfileSummary,
    ExternalCandidateEvidence,
    ExternalEvidenceFailure,
    OpusDecision,
    TriggerDecision,
)


_FORCE_TRIGGER_CHOICES = ("sparse_profile", "academic_context")
_EVIDENCE_REF_CAP = 10


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_profile_summary(path_or_stdin: str) -> CandidateProfileSummary:
    """Load a ``CandidateProfileSummary`` from a path or from stdin.

    ``path_or_stdin == "-"`` reads JSON from ``sys.stdin``. Any path that does
    not exist raises ``FileNotFoundError``. Malformed JSON raises
    ``json.JSONDecodeError``.
    """

    if path_or_stdin == "-":
        text = sys.stdin.read()
        data = json.loads(text)
    else:
        path = Path(path_or_stdin)
        if not path.exists():
            raise FileNotFoundError(f"profile summary file not found: {path}")
        with path.open() as fh:
            data = json.load(fh)
    return CandidateProfileSummary.from_dict(data)


def load_evidence_fixture(path: str) -> ExternalCandidateEvidence:
    """Load an ``ExternalCandidateEvidence`` JSON from disk."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"evidence fixture file not found: {p}")
    with p.open() as fh:
        data = json.load(fh)
    return ExternalCandidateEvidence.from_dict(data)


# ---------------------------------------------------------------------------
# Identity hints (small, source-agnostic — do NOT reuse LinkedIn orchestrator's)
# ---------------------------------------------------------------------------


def _build_identity_hints(summary: CandidateProfileSummary) -> dict:
    hints: dict = {
        "name": summary.name,
        "headline": summary.headline,
        "profile_url": summary.profile_url,
    }
    if summary.experiences:
        first_exp = summary.experiences[0]
        if first_exp.company:
            hints["current_company"] = first_exp.company
        if first_exp.title:
            hints["current_title"] = first_exp.title
    if summary.education:
        first_edu = summary.education[0]
        if first_edu.school:
            hints["school"] = first_edu.school
        if first_edu.degree:
            hints["degree"] = first_edu.degree
    return hints


# ---------------------------------------------------------------------------
# Evidence orchestration
# ---------------------------------------------------------------------------


def gather_evidence(
    *,
    summary: CandidateProfileSummary,
    brief,
    force_trigger: Optional[str],
    skip_external: bool,
    evidence_fixture: Optional[str],
    perplexity_api_key: str,
) -> tuple[
    Optional[ExternalCandidateEvidence],
    str,
    Optional[TriggerDecision],
]:
    """Decide and (maybe) fetch external evidence.

    Returns ``(evidence_or_none, status, trigger_or_none)``. Never raises;
    every failure mode degrades to ``(None, "<status>", trigger_or_none)``.

    Flag precedence (highest first): ``skip_external`` > ``evidence_fixture``
    > ``force_trigger`` > gate heuristic.
    """

    if skip_external:
        return None, "skipped_by_flag", None

    if evidence_fixture is not None:
        try:
            evidence = load_evidence_fixture(evidence_fixture)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            return None, "fixture_load_failure", None
        return evidence, "fixture_loaded", None

    if force_trigger is not None:
        trigger = TriggerDecision(
            should_run=True,
            reason=force_trigger,
            skip_reason="",
            signals={"forced": True},
        )
    else:
        trigger = should_request_external_evidence(summary=summary, brief=brief)
        if not trigger.should_run:
            return None, "skipped_no_trigger", trigger

    if not (perplexity_api_key or "").strip():
        return None, "disabled_no_api_key", trigger

    identity_hints = _build_identity_hints(summary)
    result = fetch_external_candidate_evidence(
        summary=summary,
        brief=brief,
        trigger=trigger,
        identity_hints=identity_hints,
    )
    if isinstance(result, ExternalCandidateEvidence):
        return result, "evidence_present", trigger
    if isinstance(result, ExternalEvidenceFailure):
        return None, result.reason or "unknown", trigger
    return None, "unknown", trigger


# ---------------------------------------------------------------------------
# Pretty-printers
# ---------------------------------------------------------------------------


def _evidence_refs_urls(evidence: ExternalCandidateEvidence) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for block in evidence.external_fact_blocks:
        for ref in block.evidence_refs:
            url = (ref.url or "").strip()
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    for inference in evidence.external_inferences:
        for ref in inference.basis_refs:
            url = (ref.url or "").strip()
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def format_evidence_block(
    evidence_or_failure_or_none,
    *,
    status: str,
    trigger_reason: str,
) -> str:
    """Render the ``=== External Evidence ===`` section."""

    lines: list[str] = ["=== External Evidence ===", f"status: {status}"]
    lines.append(f"trigger_reason: {trigger_reason or ''}")

    if isinstance(evidence_or_failure_or_none, ExternalCandidateEvidence):
        evidence = evidence_or_failure_or_none
        try:
            ic_text = f"{float(evidence.identity_confidence):.2f}"
        except (TypeError, ValueError):
            ic_text = "n/a"
        lines.append(f"identity_confidence: {ic_text}")
        lines.append(f"fact_blocks: {len(evidence.external_fact_blocks)}")
        lines.append(f"inferences: {len(evidence.external_inferences)}")
        lines.append(
            f"unresolved_ambiguities: {len(evidence.unresolved_ambiguities)}"
        )
        urls = _evidence_refs_urls(evidence)
        lines.append(f"evidence_refs (count={len(urls)}):")
        if urls:
            for url in urls[:_EVIDENCE_REF_CAP]:
                lines.append(f"  - {url}")
            if len(urls) > _EVIDENCE_REF_CAP:
                remaining = len(urls) - _EVIDENCE_REF_CAP
                lines.append(f"  ... and {remaining} more")
    else:
        lines.append("identity_confidence: n/a")
        lines.append("fact_blocks: 0")
        lines.append("inferences: 0")
        lines.append("unresolved_ambiguities: 0")
        lines.append("evidence_refs (count=0):")

    return "\n".join(lines)


def format_judgment_block(
    label: str,
    decision: Optional[OpusDecision],
    *,
    fallback_reason: str = "",
) -> str:
    """Render a baseline / enriched judgment section."""

    header = f"=== {label} ==="
    if decision is None:
        reason = fallback_reason or "unknown"
        return f"{header}\nenriched: null (reason={reason})"
    payload = decision.to_dict()
    return header + "\n" + json.dumps(payload, indent=2, sort_keys=True)


def format_diff_block(
    diff: dict,
    *,
    baseline: Optional[OpusDecision] = None,
    enriched: Optional[OpusDecision] = None,
) -> str:
    """Render the ``=== Diff ===`` section.

    Reads the dict shape produced by
    ``shared.external_evidence.shadow_writer.compute_judgment_diff``.
    Optional ``baseline`` / ``enriched`` are used only to surface the raw
    confidence values alongside the delta — they are not used to alter diff
    semantics.
    """

    lines = ["=== Diff ==="]
    if not diff.get("computed"):
        reason = diff.get("reason") or "unknown"
        lines.append(f"diff: not computed (reason={reason})")
        return "\n".join(lines)

    decision_changed = bool(diff.get("decision_changed"))
    path_changed = bool(diff.get("path_changed"))
    rationale_changed = bool(diff.get("rationale_changed"))
    decision_baseline = diff.get("decision_baseline", "")
    decision_enriched = diff.get("decision_enriched", "")
    path_baseline = diff.get("path_baseline", "")
    path_enriched = diff.get("path_enriched", "")
    try:
        delta = float(diff.get("confidence_delta", 0.0))
    except (TypeError, ValueError):
        delta = 0.0

    lines.append(
        f"decision_changed: {decision_changed}  "
        f"(baseline={decision_baseline} | enriched={decision_enriched})"
    )
    lines.append(
        f"path_changed:     {path_changed}  "
        f"(baseline={path_baseline} | enriched={path_enriched})"
    )
    lines.append(f"rationale_changed: {rationale_changed}")

    if baseline is not None and enriched is not None:
        try:
            b_conf = float(baseline.confidence)
            e_conf = float(enriched.confidence)
            lines.append(
                f"confidence_delta: {delta:+.3f}  "
                f"(baseline={b_conf:.3f} | enriched={e_conf:.3f})"
            )
        except (TypeError, ValueError):
            lines.append(f"confidence_delta: {delta:+.3f}")
    else:
        lines.append(f"confidence_delta: {delta:+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def run_comparison(
    args: argparse.Namespace,
    *,
    opus_baseline: Callable[[CandidateProfileSummary, object], OpusDecision] = full_judge,
    opus_enriched: Callable[
        [CandidateProfileSummary, ExternalCandidateEvidence, object],
        OpusDecision,
    ] = full_judge_with_external_evidence,
) -> int:
    """Top-level: load inputs, gather evidence, judge, print, return exit code.

    The two judge callables are injectable parameters so tests can swap them
    for mocks. Defaults are the real ones.
    """

    try:
        summary = load_profile_summary(args.profile_summary)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"ERROR: failed to parse profile summary JSON: {exc}", file=sys.stderr)
        return 1

    try:
        brief = load_brief(args.brief)
    except FileNotFoundError as exc:
        print(f"ERROR: brief file not found: {exc}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"ERROR: failed to load brief: {exc}", file=sys.stderr)
        return 1

    api_key = getattr(config, "PERPLEXITY_API_KEY", "") or ""

    evidence, status, trigger = gather_evidence(
        summary=summary,
        brief=brief,
        force_trigger=args.force_trigger,
        skip_external=args.skip_external,
        evidence_fixture=args.evidence_fixture,
        perplexity_api_key=api_key,
    )

    trigger_reason = ""
    if isinstance(evidence, ExternalCandidateEvidence) and evidence.trigger_reason:
        trigger_reason = evidence.trigger_reason
    elif trigger is not None:
        trigger_reason = trigger.reason or ""

    baseline_decision = opus_baseline(summary, brief)
    if evidence is not None:
        enriched_decision = opus_enriched(summary, evidence, brief)
    else:
        enriched_decision = None

    diff = compute_judgment_diff(
        baseline_decision,
        enriched_decision,
        skip_reason=status if enriched_decision is None else "",
    )

    brief_id = getattr(brief, "id", "") or getattr(brief, "role_title", "")

    print("=== Candidate ===")
    print(f"name: {summary.name}")
    print(f"profile_url: {summary.profile_url}")
    print(f"brief: {brief_id}")
    print()
    print(format_evidence_block(evidence, status=status, trigger_reason=trigger_reason))
    print()
    print(
        format_judgment_block(
            "Baseline judgment (canonical)",
            baseline_decision,
        )
    )
    print()
    print(
        format_judgment_block(
            "Enriched judgment (shadow / debug)",
            enriched_decision,
            fallback_reason=status,
        )
    )
    print()
    print(
        format_diff_block(
            diff,
            baseline=baseline_decision,
            enriched=enriched_decision,
        )
    )

    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline vs Perplexity-enriched judgment on a single "
            "candidate. Developer-only; never writes to disk."
        )
    )
    parser.add_argument(
        "--profile-summary",
        required=True,
        help=(
            "Path to a CandidateProfileSummary JSON document, or '-' to read "
            "from stdin."
        ),
    )
    parser.add_argument(
        "--brief",
        required=True,
        help="Path to a brief JSON consumable by shared.brief_loader.load_brief.",
    )
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help=(
            "Skip the gate + provider entirely. Wins over --evidence-fixture "
            "and --force-trigger."
        ),
    )
    parser.add_argument(
        "--force-trigger",
        choices=_FORCE_TRIGGER_CHOICES,
        default=None,
        help=(
            "Bypass the gate heuristic and force a trigger. Ignored when "
            "--skip-external or --evidence-fixture is set."
        ),
    )
    parser.add_argument(
        "--evidence-fixture",
        default=None,
        help=(
            "Path to an ExternalCandidateEvidence JSON. When set, the gate "
            "and provider are not called. Ignored when --skip-external is "
            "set."
        ),
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return run_comparison(args)


if __name__ == "__main__":
    raise SystemExit(main())
