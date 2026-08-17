"""Designer module — design-market intelligence reflection polish.

Designer Slice 9. Sibling of :mod:`market_intelligence.briefing_polish`
(general reflection polish on PlannerResult) for the Designer-specific
post-run synthesis: which design schools (RISD, ArtCenter, RCA),
agencies (Pentagram, IDEO, MetaLab), and named portfolios are
appearing in the candidate pool — surfaced as a market-intel artifact
at ``output/market_intelligence/<brief_state_key>/design_market.md``.

Two outputs:

1. ``DesignMarketArtifact`` — the editorial markdown the recruiter
   reads as standalone analysis. Persisted at
   ``output/market_intelligence/<brief_state_key>/design_market.md``.
2. ``RUBRIC_REFINE`` hunks — proposed brief-iteration changes to
   ``design_rubric.discipline_weight_overrides`` /
   ``design_rubric.calibration_exemplars`` based on per-principle
   recruiter feedback marker distribution. Surfaced via the existing
   ``HunkCard.svelte`` flow alongside other reflection hunks (Slice 9
   appends ``rubric_refine`` to that component's ``KIND_LABELS`` map).

Hard preservation contract (mirrors brief polish
``_design_rubric_drift``): the rubric in the polished v2_draft MUST
equal the seeded rubric byte-for-byte unless the recruiter explicitly
approves a ``RUBRIC_REFINE`` hunk. The reflection polish backend
NEVER mutates the rubric without recruiter sign-off. The cascade
entry that enforces this is named in this module so other modules'
sibling drift helpers can append at the next available position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from market_intelligence.schema import MarketEvidenceBatch
from shared.storage import read_jsonl, write_json


# Slice 9: the threshold at which a per-principle feedback marker
# rollup justifies a proposed rubric refinement. A principle that has
# accumulated ≥ this many "useful_guidance" markers (and zero or
# few "off_rubric" markers) gets a "weight up" hunk; the inverse gets
# a "weight down" hunk. Conservative — recruiter feedback is small-N
# at first and we don't want to propose changes from one outlier.
DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT = 3

# Slice 9: maximum number of RUBRIC_REFINE hunks to propose per run.
# Surface restraint — too many hunks means the recruiter sees
# noise instead of signal. The orchestrator picks the top-N by
# strength of the feedback signal (count delta).
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
    """One proposed change to ``design_rubric.discipline_weight_overrides``
    or ``design_rubric.calibration_exemplars``.

    Hunk shape mirrors :class:`cloris.frontend.lib.reflection.types.ReflectionHunk`
    (text-shaped) — the ``HunkCard.svelte`` renderer dispatches on
    ``kind`` and renders this hunk under the new ``rubric_refine``
    label (Slice 9 appends to the component's ``KIND_LABELS`` map).

    The ``before`` / ``after`` fields carry the JSON-serialized rubric
    fragment so the recruiter can see exactly what would change. The
    ``rationale`` is recruiter-readable Cloris-voice prose.
    """

    label: str  # short editorial label, e.g. "Weight Visual hierarchy higher for product"
    section: str  # "design_rubric.discipline_weight_overrides" | "design_rubric.calibration_exemplars"
    kind: str  # "rubric_refine" — single value for now; future may split
    before: str
    after: str
    rationale: str


@dataclass
class DesignMarketArtifact:
    """The editorial markdown the recruiter reads as design-market intel.

    Contents:
    - Pool composition: source mix (Behance vs Google CSE vs LinkedIn-resolved).
    - Discipline distribution across saved candidates.
    - Top fields / tools surfacing across the pool (Behance creative-fields).
    - Recruiter feedback marker rollup per principle.
    - Cross-check disagreement summary (Slice 8 cross-check pass).
    - Proposed rubric refinements (RUBRIC_REFINE hunk rationales).

    Formatted as markdown so the workspace surface can render it in
    a Reading panel (mirrors the existing market_intelligence prose
    artifacts).
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

    All inputs are pre-computed by the orchestrator from the candidate
    pool + recruiter annotations + cross-check pass. This function
    is pure rendering — no I/O, no LLM calls. Slice 9 ships the
    deterministic shape; an LLM-polished prose pass (Cloris-voice
    reshape) is a follow-up.
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
    """Propose ``RUBRIC_REFINE`` hunks from per-principle feedback distribution.

    Slice 9 logic:
    - For each principle, compute (useful_count - off_rubric_count).
    - Strongly positive (>= threshold): propose weight ↑ for the discipline.
    - Strongly negative (<= -threshold): propose weight ↓.
    - Tied or near-tied: no proposal.
    - Pick top-N by absolute delta, capped at ``max_hunks``.

    The proposal mutates ``discipline_weight_overrides`` only — never
    the principle definitions themselves. Principle text edits would
    require a separate hunk class (deferred to a follow-up; the
    rubric editor authoring chapter at intake is the primary editing
    surface for principle text).
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
    """Reflection-polish counterpart of brief_polish's ``_design_rubric_drift``.

    Same hard preservation contract: the polished output's
    ``design_rubric`` must equal the seeded version byte-for-byte
    UNLESS the recruiter explicitly approved RUBRIC_REFINE hunks
    (in which case those mutations are expected and the function
    validates only that no UNAPPROVED mutations slipped in).

    Returns ``None`` for no drift; a short descriptor string for
    drift the cascade should fall back on.

    Slice 9 ships the simplest version: when ``approved_rubric_hunks``
    is empty, the rubric must byte-equal the seed. When non-empty,
    Slice 9 still requires byte-equality (the orchestrator applies
    the approved hunks to the seed BEFORE polish, so the polished
    output should still byte-equal the post-approval rubric). A more
    sophisticated "did the polish apply the approvals correctly" check
    is a follow-up.
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
