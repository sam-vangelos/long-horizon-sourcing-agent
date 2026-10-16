"""Designer design-market intelligence reflection.

Assemble a market-intelligence artifact from the Designer candidate pool,
recruiter feedback markers, and cross-check results. Also propose
``RUBRIC_REFINE`` hunks for recruiter review against
``design_rubric.discipline_weight_overrides`` and
``design_rubric.calibration_exemplars``.

The artifact is deterministic markdown persisted at
``output/market_intelligence/<brief_state_key>/design_market.md``. Rubric preservation is hard: polished
``design_rubric`` output must equal the seeded rubric byte-for-byte unless the
recruiter explicitly approves a ``RUBRIC_REFINE`` hunk, and this reflection
path never mutates the rubric without that approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from market_intelligence.schema import MarketEvidenceBatch
from shared.storage import read_jsonl, write_json


# A principle needs at least this many useful_guidance markers, with few or no
# off_rubric markers, before a weight-up hunk is proposed; the inverse yields a
# weight-down hunk. The threshold is conservative for small feedback samples.
DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT = 3

# Cap RUBRIC_REFINE proposals per run so the recruiter sees signal rather than
# noise; choose the top entries by feedback count delta.
DEFAULT_MAX_RUBRIC_REFINE_HUNKS = 5


def maybe_build_and_persist_design_research_packet(
    batch: MarketEvidenceBatch,
) -> MarketEvidenceBatch:
    """Attach Designer run evidence for market-intelligence synthesis."""

    if batch.source != "designer":
        return batch

    output_dir = Path(batch.output_dir)
    run_log = read_jsonl(output_dir / "run_log.jsonl")
    adaptation_events = [
        event for event in run_log if event.get("event") == "adaptation_decision"
    ]
    pipeline_end = next(
        (event for event in reversed(run_log) if event.get("event") == "pipeline_end"),
        {},
    )
    packet = {
        "context_metadata": {
            "source": "designer",
            "run_ref": batch.run_ref,
            "brief_version": batch.brief_version,
            "generated_at": batch.generated_at,
            "context_quality": "runtime_evidence",
            "analysis_provenance": (
                "design_market_intelligence."
                "maybe_build_and_persist_design_research_packet"
            ),
        },
        "portfolio_signal_summary": {
            "runtime_summary": batch.runtime_summary,
            "metrics_summary": batch.metrics_summary,
            "pipeline_end": pipeline_end,
        },
        "adaptation_timeline": adaptation_events,
    }
    path = output_dir / "designer-research-input.json"
    try:
        write_json(path, packet)
    except OSError:
        pass
    else:
        batch.research_input_path = str(path)
    batch.research_context = packet
    batch.context_quality = "runtime_evidence"
    batch.analysis_provenance = packet["context_metadata"]["analysis_provenance"]
    return batch


@dataclass(frozen=True)
class RubricRefineHunk:
    """One proposed change to a design-rubric weight or exemplar.

    ``HunkCard.svelte`` dispatches on ``kind`` and renders this as
    ``rubric_refine``. ``before`` and ``after`` carry the JSON-serialized rubric
    fragment; ``rationale`` is recruiter-readable Cloris prose.
    """

    label: str  # short editorial label, e.g. "Weight Visual hierarchy higher for product"
    section: str  # "design_rubric.discipline_weight_overrides" | "design_rubric.calibration_exemplars"
    kind: str  # "rubric_refine" — single value for now; future may split
    before: str
    after: str
    rationale: str


@dataclass
class DesignMarketArtifact:
    """The editorial markdown the recruiter reads as design-market intelligence.

    It contains pool composition, discipline distribution, top fields/tools,
    recruiter feedback by principle, cross-check disagreement, and proposed
    rubric-refinement rationales. Markdown keeps it renderable by the workspace
    reading surface.
    """

    brief_state_key: str
    markdown: str
    proposed_hunks: tuple[RubricRefineHunk, ...] = field(default_factory=tuple)


def assemble_design_market_artifact(
    *,
    brief_state_key: str,
    pool_composition: dict[str, int],
    discipline_distribution: dict[str, int],
    top_fields: list[tuple[str, int]],
    top_tools: list[tuple[str, int]],
    feedback_marker_distribution: dict[str, dict[str, int]],
    cross_check_disagreement_count: int = 0,
    cross_check_total_count: int = 0,
    proposed_hunks: Iterable[RubricRefineHunk] = (),
) -> DesignMarketArtifact:
    """Build the editorial markdown for the design-market artifact.

    Inputs are precomputed from the candidate pool, recruiter annotations, and
    cross-check pass. This function only renders; it performs no I/O or LLM
    calls.
    """

    sections: list[str] = []

    sections.append("# Design-market intelligence")
    sections.append(
        "Cloris's read on the candidate pool that surfaced for this brief."
    )
    sections.append("")

    sections.append("## Pool composition")
    if pool_composition:
        for source, count in sorted(pool_composition.items(), key=lambda kv: -kv[1]):
            sections.append(f"- **{source}** — {count}")
    else:
        sections.append("_No candidates surfaced._")
    sections.append("")

    sections.append("## Discipline distribution")
    if discipline_distribution:
        for discipline, count in sorted(
            discipline_distribution.items(), key=lambda kv: -kv[1]
        ):
            sections.append(f"- **{discipline}** — {count}")
    else:
        sections.append("_No discipline tags collected._")
    sections.append("")

    sections.append("## Fields surfacing")
    if top_fields:
        for field_name, count in top_fields[:10]:
            sections.append(f"- {field_name} ({count})")
    else:
        sections.append("_No fields collected._")
    sections.append("")

    sections.append("## Tool stack surfacing")
    if top_tools:
        for tool_name, count in top_tools[:10]:
            sections.append(f"- {tool_name} ({count})")
    else:
        sections.append("_No tool signals collected._")
    sections.append("")

    sections.append("## Recruiter feedback per principle")
    if feedback_marker_distribution:
        for principle, markers in sorted(feedback_marker_distribution.items()):
            tally = ", ".join(
                f"{marker}: {count}" for marker, count in sorted(markers.items())
            )
            sections.append(f"- **{principle}** — {tally}")
    else:
        sections.append("_No recruiter feedback yet._")
    sections.append("")

    if cross_check_total_count > 0:
        sections.append("## Cross-check (Sonnet 4.6) disagreement")
        sections.append(
            f"- Cross-check ran on {cross_check_total_count} top-decile candidates; "
            f"models disagreed on {cross_check_disagreement_count} of them "
            f"({_safe_pct(cross_check_disagreement_count, cross_check_total_count)})."
        )
        sections.append("")

    proposed_hunks_tuple = tuple(proposed_hunks)
    if proposed_hunks_tuple:
        sections.append("## Proposed rubric refinements")
        sections.append(
            "Cloris would tighten the rubric based on recruiter feedback. "
            "Each proposal needs explicit recruiter approval before it lands "
            "in the brief — the rubric is otherwise preserved byte-for-byte "
            "across reflection polish."
        )
        for hunk in proposed_hunks_tuple:
            sections.append(f"- **{hunk.label}** — {hunk.rationale}")
        sections.append("")

    markdown = "\n".join(sections).rstrip() + "\n"
    return DesignMarketArtifact(
        brief_state_key=brief_state_key,
        markdown=markdown,
        proposed_hunks=proposed_hunks_tuple,
    )


def _safe_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0%"
    return f"{round(100 * numerator / denominator)}%"


# ---------------------------------------------------------------------------
# Rubric refinement proposals from recruiter feedback marker rollup
# ---------------------------------------------------------------------------


def propose_rubric_refinements(
    *,
    feedback_marker_distribution: dict[str, dict[str, int]],
    discipline: str,
    current_rubric: dict[str, Any],
    threshold: int = DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT,
    max_hunks: int = DEFAULT_MAX_RUBRIC_REFINE_HUNKS,
) -> list[RubricRefineHunk]:
    """Propose ``RUBRIC_REFINE`` hunks from per-principle feedback.

    Compute ``useful_count - off_rubric_count`` for each principle. Deltas at or
    above ``threshold`` propose a weight increase; deltas at or below its
    negative propose a decrease; near-ties produce no hunk. Return the top
    ``max_hunks`` by absolute delta.

    Mutate only ``discipline_weight_overrides`` in the proposed hunk. Principle
    text remains outside this hunk type and is edited through the rubric
    authoring surface.
    """

    if not isinstance(current_rubric, dict):
        return []
    overrides = (
        current_rubric.get("discipline_weight_overrides")
        if isinstance(current_rubric.get("discipline_weight_overrides"), dict)
        else {}
    )
    discipline_overrides: dict[str, float] = (
        dict(overrides.get(discipline, {})) if isinstance(overrides, dict) else {}
    )

    proposals: list[tuple[int, RubricRefineHunk]] = []  # (priority, hunk)

    for principle_name, markers in feedback_marker_distribution.items():
        if not isinstance(markers, dict):
            continue
        useful = int(markers.get("useful_guidance", 0))
        off_rubric = int(markers.get("off_rubric", 0))
        delta = useful - off_rubric

        current_weight = float(discipline_overrides.get(principle_name, 1.0))

        if delta >= threshold:
            new_weight = round(min(current_weight + 0.3, 5.0), 2)
            if new_weight == current_weight:
                continue
            hunk = RubricRefineHunk(
                label=f"Weight {principle_name} higher for {discipline}",
                section="design_rubric.discipline_weight_overrides",
                kind="rubric_refine",
                before=f"{discipline}: {{{principle_name}: {current_weight}}}",
                after=f"{discipline}: {{{principle_name}: {new_weight}}}",
                rationale=(
                    f"Recruiters marked {principle_name} as useful guidance "
                    f"on {useful} candidates and off-rubric on {off_rubric}. "
                    f"Cloris would weight it higher for {discipline} briefs going forward."
                ),
            )
            proposals.append((delta, hunk))
        elif delta <= -threshold:
            new_weight = round(max(current_weight - 0.3, 0.0), 2)
            if new_weight == current_weight:
                continue
            hunk = RubricRefineHunk(
                label=f"Weight {principle_name} lower for {discipline}",
                section="design_rubric.discipline_weight_overrides",
                kind="rubric_refine",
                before=f"{discipline}: {{{principle_name}: {current_weight}}}",
                after=f"{discipline}: {{{principle_name}: {new_weight}}}",
                rationale=(
                    f"Recruiters marked {principle_name} as off-rubric on "
                    f"{off_rubric} candidates and useful on only {useful}. "
                    f"Cloris would deprioritize it for {discipline} briefs going forward."
                ),
            )
            proposals.append((abs(delta), hunk))

    proposals.sort(key=lambda item: -item[0])
    return [hunk for _, hunk in proposals[:max_hunks]]


# ---------------------------------------------------------------------------
# Rubric byte-equality preservation in reflection polish
# ---------------------------------------------------------------------------


def reflection_design_rubric_drift(
    *,
    seeded: dict,
    polished: dict,
    approved_rubric_hunks: tuple[RubricRefineHunk, ...] = (),
) -> str | None:
    """Return drift when polished ``design_rubric`` differs from its seed.

    The polished rubric must equal the seeded rubric byte-for-byte. Approved
    ``RUBRIC_REFINE`` hunks are applied to the seed before polish, so equality
    is checked against that already-approved input; unapproved mutations fall
    through the cascade. Return ``None`` for no drift or a short descriptor for
    fallback.
    """

    seed = seeded.get("design_rubric")
    if not isinstance(seed, dict) or not seed:
        return None
    polish = polished.get("design_rubric")
    if not isinstance(polish, dict) or not polish:
        return "dropped"
    if polish != seed:
        return f"mutated approved_hunks={len(approved_rubric_hunks)}"
    return None
