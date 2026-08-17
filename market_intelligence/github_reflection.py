"""Github-specific reflection / market-intelligence packet (OSS Maintainers Slice 9).

Per OSS Maintainers Module Spec §9 + §11, the LinkedIn-only
:func:`market_intelligence.research_context.maybe_build_and_persist_research_packet`
needs a github-source variant so post-run reflection narrates
ecosystem momentum for recruiter-named ``target_projects``.

The narrative carries:

- **Maintainer-mass per target project** — count of saved candidates
  classified at each maintainership level (contributor / maintainer
  / project_lead) per project, derived from the run's
  ``final_judgments.jsonl`` payloads.
- **Per-target-project save rate** — saves anchored to each named
  project (via a candidate's classifier ``evidence_sources`` listing).
- **Classifier confidence distribution** — histogram per level so
  the recruiter can spot "mostly low-confidence project_leads"
  patterns that suggest a calibration drift (per spec §13.1).
- **Per-signal contribution rollup** — which signals fired most
  often across saved candidates, so post-trial calibration can
  detect "right for the wrong reason" patterns.

Spec §14 default: write github-specific in this slice; generalize
to a shared abstraction in Phase 3 cleanup. The shared abstraction
would unify this module with
:mod:`market_intelligence.research_context`'s LinkedIn variant once
both have hardened against real customer signal.

Failure-mode posture: every read is fail-soft. A missing
``final_judgments.jsonl`` returns an empty narrative; the wider
batch flow doesn't depend on the narrative being non-empty (it's
post-run reflection material, not control state).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from market_intelligence.schema import MarketEvidenceBatch
from shared.storage import read_jsonl, write_json

logger = logging.getLogger(__name__)


SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}

MAX_TOP_PROJECTS = 10
MAX_TOP_SIGNALS = 10


# Audit Move #25: mirror briefing_polish._emit_stage so per-module
# reflection composers' stage logs interleave cleanly with the wider
# market-intel run trace.
def _emit_stage(message: str) -> None:
    """Mirror briefing_polish._emit_stage."""

    import sys

    print(f"[market-intel] {message}", file=sys.stderr, flush=True)


def maybe_build_and_persist_github_research_packet(
    batch: MarketEvidenceBatch,
) -> MarketEvidenceBatch:
    """Build + persist a github-source research packet, in place on ``batch``.

    Mirrors the LinkedIn variant in
    :func:`market_intelligence.research_context.maybe_build_and_persist_research_packet`
    in shape but assembles a github-specific ecosystem narrative
    rather than the LinkedIn lane / search-memory analysis. Returns
    the updated batch unchanged when ``batch.source != "github"``
    (no-op for non-github sources).
    """

    if batch.source != "github":
        return batch

    output_dir = Path(batch.output_dir)
    research_input_path = output_dir / "github-research-input.json"
    final_judgments = batch.final_judgments or _load_optional_jsonl(
        output_dir / "final_judgments.jsonl"
    )

    narrative = build_github_ecosystem_narrative(
        final_judgments=final_judgments,
    )

    packet = {
        "context_metadata": {
            "source": "github",
            "run_ref": batch.run_ref,
            "brief_version": batch.brief_version,
            "generated_at": batch.generated_at,
            "context_quality": _context_quality_label(narrative),
            "analysis_provenance": "github_reflection.maybe_build_and_persist_github_research_packet",
        },
        "ecosystem_momentum": narrative,
        "metrics_summary": batch.metrics_summary,
    }

    try:
        write_json(research_input_path, packet)
    except OSError as exc:
        logger.warning(
            "github_reflection: failed to persist packet at %s (%s); "
            "narrative attached to batch without disk write",
            research_input_path,
            exc,
        )
    else:
        batch.research_input_path = str(research_input_path)
    batch.research_context = packet
    batch.context_quality = packet["context_metadata"]["context_quality"]
    batch.analysis_provenance = packet["context_metadata"]["analysis_provenance"]
    return batch


def build_github_ecosystem_narrative(
    *,
    final_judgments: list[dict] | None,
) -> dict[str, Any]:
    """Pure transform: final_judgments → ecosystem narrative dict.

    Separated from the persist step so tests can exercise the
    aggregation without touching disk. Returns a dict with five
    top-level keys (per-project save count, per-project mass per
    level, confidence histogram per level, per-signal contribution
    rollup, totals). Empty ``final_judgments`` ⇒ structurally-valid
    narrative with all zeros (the recruiter still gets a packet,
    just one that says "no saves to narrate").
    """

    save_records = list(_iter_save_records(final_judgments or []))

    by_project_count: dict[str, int] = defaultdict(int)
    by_project_level: dict[str, Counter] = defaultdict(Counter)
    by_level_confidence: dict[str, list[float]] = defaultdict(list)
    signal_counter: Counter = Counter()
    saves_with_classification = 0

    for record in save_records:
        maintainership = record.get("maintainership") or {}
        if not isinstance(maintainership, dict):
            continue
        level = maintainership.get("level")
        confidence = maintainership.get("confidence")
        evidence_sources = maintainership.get("evidence_sources") or []
        signals = maintainership.get("signals") or {}

        if level not in {"contributor", "maintainer", "project_lead"}:
            continue
        saves_with_classification += 1

        # Per-project rollup keyed by the projects cited in
        # evidence_sources. Each evidence string starts with
        # `<signal_kind>:<owner/repo>` (per Slice 4's
        # `evidence_sources` convention).
        cited_projects: set[str] = set()
        for src in evidence_sources:
            if not isinstance(src, str) or ":" not in src:
                continue
            _kind, _, rest = src.partition(":")
            project = rest.split(":", 1)[0].strip().lower()
            if project:
                cited_projects.add(project)
                signal_counter[_kind.strip()] += 1
        for project in cited_projects:
            by_project_count[project] += 1
            by_project_level[project][level] += 1

        if isinstance(confidence, (int, float)):
            by_level_confidence[level].append(round(float(confidence), 3))

        # Also count per-signal contribution from the structured
        # signals dict (independent of evidence_sources serialization).
        for signal_name, signal_value in signals.items():
            if isinstance(signal_value, (int, float)) and signal_value > 0:
                signal_counter[signal_name] += 1

    top_projects = _top_projects_payload(by_project_count, by_project_level)
    confidence_histograms = {
        level: _confidence_histogram(values)
        for level, values in by_level_confidence.items()
    }
    top_signals = [
        {"signal": name, "count": count}
        for name, count in signal_counter.most_common(MAX_TOP_SIGNALS)
    ]

    return {
        "totals": {
            "saves_total": len(save_records),
            "saves_with_classification": saves_with_classification,
        },
        "top_projects": top_projects,
        "confidence_histograms": confidence_histograms,
        "top_signals": top_signals,
    }


# ---------------------------------------------------------------------------
# Hunk composer (audit Move #9)
# ---------------------------------------------------------------------------


# Maintainership level ordering (low to high), mirrors
# ``shared.brief_v2_schema.MAINTAINERSHIP_LEVEL_ORDER``. The hunk
# composer uses this to propose a "lower the floor by one rung"
# modification when the run's saves cluster below the recruiter's
# declared level.
_MAINTAINERSHIP_ORDER: tuple[str, ...] = (
    "contributor",
    "maintainer",
    "project_lead",
)


def propose_github_hunks(
    *,
    final_judgments: list[dict] | None,
    brief_raw: dict | None = None,
) -> list[dict]:
    """Compose Gate-2 hunks from a github run's final-judgments.

    Brief snapshot is optional; when supplied, the composer can emit
    "before" values for hunks that propose modifying existing brief
    fields (``target_projects``, ``maintainership_level``).

    Hunks emitted (per audit plan exemplars):

    - ``broaden_target_projects`` — fires when a notable cluster of
      saves cite projects outside the brief's declared
      ``target_projects``.
    - ``lower_maintainership_threshold`` — fires when the brief's
      floor is above ``contributor`` and the run's saves cluster
      below the floor in the maintainership distribution.

    Empty inputs ⇒ empty hunk list.
    """

    judgments = final_judgments or []
    if not judgments:
        _emit_stage("reflection.github:start judgments=0 result=empty_input")
        return []

    _emit_stage(f"reflection.github:start judgments={len(judgments)}")

    narrative = build_github_ecosystem_narrative(final_judgments=judgments)
    saves_total = narrative["totals"]["saves_total"]
    if saves_total == 0:
        _emit_stage(
            "reflection.github:end hunks_proposed=0 reason=zero_saves"
        )
        return []

    _emit_stage(
        f"reflection.github:narrative_built saves={saves_total} "
        f"top_projects={len(narrative.get('top_projects') or [])}"
    )

    brief_raw = brief_raw or {}
    declared_projects_raw = brief_raw.get("target_projects") or []
    declared_projects = {
        p.strip().lower()
        for p in declared_projects_raw
        if isinstance(p, str) and p.strip()
    }
    declared_level = brief_raw.get("maintainership_level")

    hunks: list[dict] = []

    # Hunk 1: broaden_target_projects when ≥1 saved candidate cites
    # projects outside the declared list at notable volume.
    top_projects = narrative.get("top_projects") or []
    out_of_list = [
        item
        for item in top_projects
        if item["project"].strip().lower() not in declared_projects
    ]
    if (
        declared_projects
        and out_of_list
        and out_of_list[0]["save_count"] >= max(2, saves_total // 5)
    ):
        proposed_additions = [item["project"] for item in out_of_list[:5]]
        proposed = list(declared_projects_raw) + proposed_additions
        hunks.append(
            _build_hunk(
                hunk_id="github-broaden-target-projects",
                section="target_projects",
                kind="modify",
                label="Broaden target projects",
                before=", ".join(declared_projects_raw)
                if declared_projects_raw
                else None,
                after=", ".join(proposed),
                rationale=(
                    f"Saved candidates cited {len(out_of_list)} project(s) "
                    "outside the declared target_projects list — the top "
                    f"hits account for {sum(item['save_count'] for item in out_of_list[:3])} "
                    f"of {saves_total} total saves. Consider weighting "
                    "those projects in."
                ),
                confidence=0.6,
                target_field="target_projects",
            )
        )

    # Hunk 2: lower_maintainership_threshold when the declared floor
    # is above contributor and the saves' maintainership distribution
    # has notable mass at lower levels (suggesting valuable candidates
    # are passing the floor only because the classifier is generous).
    if (
        isinstance(declared_level, str)
        and declared_level in _MAINTAINERSHIP_ORDER
        and declared_level != _MAINTAINERSHIP_ORDER[0]
    ):
        level_counts: Counter = Counter()
        for item in top_projects:
            for level, count in (item.get("by_level") or {}).items():
                level_counts[level] += count
        below_floor_count = sum(
            count
            for level, count in level_counts.items()
            if level in _MAINTAINERSHIP_ORDER
            and _MAINTAINERSHIP_ORDER.index(level)
            < _MAINTAINERSHIP_ORDER.index(declared_level)
        )
        if below_floor_count >= max(2, saves_total // 4):
            current_idx = _MAINTAINERSHIP_ORDER.index(declared_level)
            proposed_level = _MAINTAINERSHIP_ORDER[current_idx - 1]
            hunks.append(
                _build_hunk(
                    hunk_id="github-lower-maintainership-threshold",
                    section="maintainership_level",
                    kind="modify",
                    label="Lower maintainership threshold",
                    before=declared_level,
                    after=proposed_level,
                    rationale=(
                        f"{below_floor_count} of {saves_total} saves "
                        "carried maintainership classifications below "
                        f"the declared floor ({declared_level}). "
                        "Lowering by one rung exposes those candidates "
                        "to the full evaluator — they may still get "
                        "rejected at full eval, but the recruiter "
                        "should see them."
                    ),
                    confidence=0.55,
                    target_field="maintainership_level",
                )
            )

    _emit_stage(
        f"reflection.github:end hunks_proposed={len(hunks)} "
        f"sections={sorted({h['section'] for h in hunks})}"
    )
    return hunks


def _build_hunk(
    *,
    hunk_id: str,
    section: str,
    kind: str,
    label: str,
    before: str | None,
    after: str,
    rationale: str,
    confidence: float,
    target_field: str,
) -> dict:
    return {
        "hunk_id": hunk_id,
        "section": section,
        "kind": kind,
        "label": label,
        "before": before,
        "after": after,
        "rationale": rationale,
        "confidence": confidence,
        "default_approved": confidence >= 0.65,
        "target_field": target_field,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_save_records(final_judgments: Iterable[dict]) -> Iterable[dict]:
    """Yield records corresponding to SAVE-class terminal decisions.

    The shape of ``final_judgments`` rows varies a little by source;
    we accept either ``decision`` or ``terminal_decision`` as the
    label key, and we accept either a top-level ``maintainership``
    field or a nested ``terminal_payload.candidate_record.maintainership``
    path. Defensive coercion so this works against current run
    artifacts and survives a future refactor.
    """

    for row in final_judgments:
        if not isinstance(row, dict):
            continue
        decision = (
            row.get("decision")
            or row.get("terminal_decision")
            or row.get("full_decision", {}).get("decision")
        )
        if decision not in SAVE_DECISIONS:
            continue
        # Pull the maintainership classification from any of the
        # known shapes.
        maintainership = row.get("maintainership")
        if not isinstance(maintainership, dict):
            payload = row.get("terminal_payload") or {}
            if isinstance(payload, dict):
                cr = payload.get("candidate_record") or {}
                if isinstance(cr, dict):
                    maintainership = cr.get("maintainership")
        if isinstance(maintainership, dict):
            row = {**row, "maintainership": maintainership}
        yield row


def _top_projects_payload(
    by_project_count: dict[str, int],
    by_project_level: dict[str, Counter],
) -> list[dict]:
    """Return the top-N projects by save count with per-level breakdown."""

    sorted_projects = sorted(
        by_project_count.items(), key=lambda x: (-x[1], x[0])
    )[:MAX_TOP_PROJECTS]
    return [
        {
            "project": project,
            "save_count": count,
            "by_level": dict(by_project_level[project]),
        }
        for project, count in sorted_projects
    ]


def _confidence_histogram(values: list[float]) -> dict[str, int]:
    """Return a 4-bucket confidence histogram for telemetry inspection.

    Buckets: ``[0.0, 0.25)``, ``[0.25, 0.5)``, ``[0.5, 0.75)``,
    ``[0.75, 1.0]``. Spec §13.1 mentions per-signal contribution
    inspection; this is the analogous distribution surface for
    confidence values.
    """

    buckets = {"0.0-0.25": 0, "0.25-0.5": 0, "0.5-0.75": 0, "0.75-1.0": 0}
    for v in values:
        if v < 0.25:
            buckets["0.0-0.25"] += 1
        elif v < 0.5:
            buckets["0.25-0.5"] += 1
        elif v < 0.75:
            buckets["0.5-0.75"] += 1
        else:
            buckets["0.75-1.0"] += 1
    return buckets


def _context_quality_label(narrative: dict[str, Any]) -> str:
    """Return a short editorial label for the narrative quality.

    Mirrors the LinkedIn variant's convention: the wider
    market-intelligence consumers want a banded label rather than a
    raw number to render in surfaces. ``empty`` when no saves;
    ``thin`` when fewer than 5 classified saves; ``substantive``
    otherwise.
    """

    classified = (
        narrative.get("totals", {}).get("saves_with_classification", 0)
    )
    if classified <= 0:
        return "empty"
    if classified < 5:
        return "thin"
    return "substantive"


def _load_optional_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return list(read_jsonl(path))
    except Exception as exc:  # noqa: BLE001 — fail-soft per spec §12
        logger.warning(
            "github_reflection: failed to read %s (%s); treating as empty",
            path,
            exc,
        )
        return []
