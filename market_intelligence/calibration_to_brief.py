"""Translate calibration rollups into recruiter-reviewable V2 brief patches.

The module is a pure translator from ``CalibrationRollup`` plus eligible areas to
``BriefPatch`` instances consumed by the reflection layer.

Three patch rules are supported:

1. A high ``wrong`` rate on an area proposes a ``non_fit_pattern`` payload that the
   recruiter can edit before approval.
2. A high ``off_rubric`` rate on ``SAVE`` rows in the high-confidence quartile proposes
   a ``depth_distinction`` addendum. The rollup exposes quartiles rather than raw
   confidence, so ``q4 = [0.75, 1.0]`` is the operational filter and is stricter than
   the source ``> 0.7`` threshold.
3. A high ``useful`` rate proposes a ``calibration_examples`` patch whose target is
   the canonical V2 ``transferability_examples`` field.

The threshold layer requires ``MIN_MARKERS_PER_AREA = 5`` weighted markers per
area; this translator adds ``MIN_MARKERS_PER_PATTERN = 3`` raw markers per
pattern, so combined signal cannot emit a patch unless its individual marker
class also clears the minimum. Every qualifying pattern emits a patch.
Pattern order is ``non_fit_pattern`` → ``depth_distinction`` →
``calibration_examples``; area order follows the eligible-area input.

Designer rubric refinements are wrapped in the same ``BriefPatch`` surface and
retain the empty-input failure posture of the underlying proposer.

Reflection ingestion remains the caller's responsibility. The existing hunk applier
handles list-of-strings and prose sections; structured payloads still need an
apply-time path, so these proposals remain recruiter-reviewable at Gate 2 until then.
The per-pattern threshold is explicit so it can be tuned from collected data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from market_intelligence.calibration_thresholds import EligibleArea
from shared.runtime_state.calibration import (
    CalibrationRollup,
    CalibrationRollupKey,
)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


# Per-pattern marker floor. The threshold layer's
# ``MIN_MARKERS_PER_AREA = 5`` is the per-area eligibility floor on
# weighted markers; this constant is the per-pattern floor on raw
# markers of the pattern's marker class. Layered: an area must clear
# both the upstream eligibility AND the per-pattern floor for a patch
# to surface. Tested for off-by-one at
# ``tests/test_calibration_to_brief.py::test_wrong_at_floor_emits_non_fit_pattern``
# (>= 3 fires; 2 doesn't).
MIN_MARKERS_PER_PATTERN: int = 3


# Quartile band used by the depth_distinction filter. The rollup exposes static
# quartiles, so q4 (``[0.75, 1.0]``) is the operational high-confidence band. Keep
# this as a named constant so any future widening is an explicit change.
HIGH_CONFIDENCE_QUARTILE: str = "q4"


# Terminal-decision string the depth_distinction rule scopes to.
# Mirrors the runtime-state writer's terminal_decision contract at
# ``shared/runtime_state/store.py``. Pinning the constant means the
# depth_distinction filter cannot drift away from the canonical
# decision-enum value silently.
SAVE_DECISION: str = "SAVE"


# Patch kind strings consumed by reflection integration. The
# ``calibration_examples`` kind targets the canonical V2
# ``transferability_examples`` field.
PATCH_KIND_NON_FIT_PATTERN: str = "non_fit_pattern"
PATCH_KIND_DEPTH_DISTINCTION: str = "depth_distinction"
PATCH_KIND_CALIBRATION_EXAMPLES: str = "calibration_examples"
PATCH_KIND_RUBRIC_REFINE: str = "rubric_refine"


# ---------------------------------------------------------------------------
# Dataclass — the reflection wire surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BriefPatch:
    """One proposed brief patch derived from calibration markers.

    Frozen so callers cannot mutate a proposal before reflection wraps it as a
    Gate-2 hunk.

    Fields:

    - ``kind`` is one of the ``PATCH_KIND_*`` constants and selects downstream
      hunk rendering.
    - ``target_section`` is the V2 brief field being changed. Generic patterns
      target ``non_fit_patterns``, ``depth_distinction``, or
      ``transferability_examples``; Designer routing passes through the rubric
      section, including dotted paths.
    - ``capability_area`` identifies the area; Designer patches use the empty
      string because they are per-principle.
    - ``label`` is the recruiter-facing short label (80 characters or fewer recommended).
    - ``rationale`` explains the proposal in recruiter-readable prose.
    - ``payload`` carries the structured value for the target field.
    - ``n_markers_for_kind`` records the markers that triggered this pattern for
      telemetry and per-hunk confidence weighting; Designer routing uses the
      per-principle ``useful_guidance + off_rubric`` total.
    """

    kind: str
    target_section: str
    capability_area: str
    label: str
    rationale: str
    payload: dict[str, Any]
    n_markers_for_kind: int


# ---------------------------------------------------------------------------
# Per-area count walk
# ---------------------------------------------------------------------------


def _count_marker_in_area(
    counts: dict[CalibrationRollupKey, int],
    *,
    capability_area: str,
    marker_value: str,
    quartile_filter: frozenset[str] | None = None,
    decision_filter: frozenset[str | None] | None = None,
) -> int:
    """Sum counts in ``counts`` for a single ``(area, marker)`` pattern.

    Walks the full-key counts surface; filters can pin the quartile or
    terminal decision (used by the depth_distinction pattern's q4 +
    SAVE scope). When a filter is ``None``, all values pass.

    Pure read — does not mutate ``counts``. The walk is per-call rather
    than precomputed because the per-pattern translation only needs
    three or four (area, marker) lookups per eligible area, and the
    rollup's full ``counts`` surface is small (one row per marker
    permutation actually observed).
    """

    total = 0
    for key, count in counts.items():
        if key.capability_area != capability_area:
            continue
        if key.marker_value != marker_value:
            continue
        if quartile_filter is not None and key.confidence_quartile not in quartile_filter:
            continue
        if decision_filter is not None and key.terminal_decision not in decision_filter:
            continue
        total += count
    return total


# ---------------------------------------------------------------------------
# Per-pattern translators
# ---------------------------------------------------------------------------


def _maybe_non_fit_pattern_patch(
    area: EligibleArea,
    *,
    counts: dict[CalibrationRollupKey, int],
) -> BriefPatch | None:
    """Build a non-fit patch from ``wrong`` markers in any quartile or decision.

    ``>= MIN_MARKERS_PER_PATTERN`` is required. The payload follows
    ``shared.brief_schema.NonFitPattern`` and leaves
    ``examples`` empty for recruiter completion at Gate 2.
    """

    n_wrong = _count_marker_in_area(
        counts,
        capability_area=area.capability_area,
        marker_value="wrong",
    )
    if n_wrong < MIN_MARKERS_PER_PATTERN:
        return None
    return BriefPatch(
        kind=PATCH_KIND_NON_FIT_PATTERN,
        target_section="non_fit_patterns",
        capability_area=area.capability_area,
        label=f"Non-fit pattern: {area.capability_area}",
        rationale=(
            f"The recruiter marked {n_wrong} candidate(s) attributed to "
            f"{area.capability_area} as 'wrong' — the area itself is "
            "starting to look like a false-positive shape. Adding it to "
            "non_fit_patterns lets future runs deprioritize matching "
            "profiles before they reach Cloris's full evaluation."
        ),
        payload={
            "label": area.capability_area,
            "description": (
                f"Candidates surfacing as {area.capability_area} that the "
                f"recruiter rejected as not-fit ({n_wrong} marker(s) in "
                "the calibration history)."
            ),
            "why_not": (
                f"Recruiter feedback shows {area.capability_area} surfaces "
                "profiles that look adjacent to the brief but consistently "
                "fail the recruiter's criteria. Edit this why_not before "
                "approving so it captures the specific mismatch."
            ),
            "examples": [],
        },
        n_markers_for_kind=n_wrong,
    )


def _maybe_depth_distinction_patch(
    area: EligibleArea,
    *,
    counts: dict[CalibrationRollupKey, int],
) -> BriefPatch | None:
    """Build a depth-distinction patch from high-confidence ``SAVE`` mismatches.

    The filter is ``off_rubric`` in q4 ``SAVE`` rows at or above
    ``MIN_MARKERS_PER_PATTERN``. The payload names the nested
    ``depth_distinction.edge_case_guidance`` path so application can deep-write
    prose rather than treat it as a list of dictionaries.
    """

    n_off_rubric_high_conf_save = _count_marker_in_area(
        counts,
        capability_area=area.capability_area,
        marker_value="off_rubric",
        quartile_filter=frozenset({HIGH_CONFIDENCE_QUARTILE}),
        decision_filter=frozenset({SAVE_DECISION}),
    )
    if n_off_rubric_high_conf_save < MIN_MARKERS_PER_PATTERN:
        return None
    addendum = (
        f"{area.capability_area}: the recruiter has flagged "
        f"{n_off_rubric_high_conf_save} high-confidence save(s) as "
        "off-rubric. When evaluating in this area, refine the criteria "
        "so high-confidence saves match the recruiter's actual definition "
        "of fit — Cloris's reads in this area are landing in the right "
        "ballpark on confidence but on the wrong axis."
    )
    return BriefPatch(
        kind=PATCH_KIND_DEPTH_DISTINCTION,
        target_section="depth_distinction",
        capability_area=area.capability_area,
        label=f"Depth distinction: clarify edge case for {area.capability_area}",
        rationale=(
            "When Cloris was confident it was right and the recruiter said "
            "off-rubric, that's strong signal the depth-distinction "
            "criteria need refinement for this area."
        ),
        payload={
            "section_path": "depth_distinction.edge_case_guidance",
            "addendum": addendum,
        },
        n_markers_for_kind=n_off_rubric_high_conf_save,
    )


def _maybe_calibration_examples_patch(
    area: EligibleArea,
    *,
    counts: dict[CalibrationRollupKey, int],
) -> BriefPatch | None:
    """Build a transferability-example patch from ``useful`` markers in any quartile
    or decision; ``>= MIN_MARKERS_PER_PATTERN`` is required.

    The payload follows ``shared.brief_schema.TransferabilityExample`` and targets
    canonical V2 ``transferability_examples`` rather than deprecated
    ``calibration_examples``.
    """

    n_useful = _count_marker_in_area(
        counts,
        capability_area=area.capability_area,
        marker_value="useful",
    )
    if n_useful < MIN_MARKERS_PER_PATTERN:
        return None
    return BriefPatch(
        kind=PATCH_KIND_CALIBRATION_EXAMPLES,
        target_section="transferability_examples",
        capability_area=area.capability_area,
        label=f"Calibration example: {area.capability_area} transfers",
        rationale=(
            f"The recruiter marked {n_useful} candidate(s) in "
            f"{area.capability_area} as useful — Cloris's reads in this "
            "area are landing on signal the recruiter values. Capturing "
            "this as a transferability example confirms the pattern for "
            "future runs."
        ),
        payload={
            "result": "transfers",
            "source_context": area.capability_area,
            "target_context": area.capability_area,
            "rationale": (
                f"Recruiter confirmed {n_useful} useful read(s) on "
                f"candidates attributed to {area.capability_area}. The "
                "pattern is durable enough to bake into the brief."
            ),
        },
        n_markers_for_kind=n_useful,
    )


# Inner emission order. Tested at
# ``tests/test_calibration_to_brief.py::test_area_with_all_three_patterns_emits_three_in_pinned_order``.
# Stable so the recruiter sees patches in the same order across re-fetches
# of the same propose phase, and so the per-area multi-pattern review at
# Gate 2 has predictable visual ordering.
_PATTERN_TRANSLATORS = (
    _maybe_non_fit_pattern_patch,
    _maybe_depth_distinction_patch,
    _maybe_calibration_examples_patch,
)


# ---------------------------------------------------------------------------
# Public translator: capability-area patterns
# ---------------------------------------------------------------------------


def translate_eligible_areas(
    *,
    rollup: CalibrationRollup,
    eligible_areas: list[EligibleArea],
) -> list[BriefPatch]:
    """Translate threshold-eligible areas into V2 brief patches.

    Pure function. Walks ``eligible_areas`` in input order (preserving
    the threshold layer's ranking — see
    ``calibration_thresholds.select_eligible_areas``); for each area,
    runs the three per-pattern translators in pinned order and
    collects the patches that fire.

    An eligible area whose per-pattern floor is met by NONE of the
    three patterns produces no patches — the per-pattern floor
    (:data:`MIN_MARKERS_PER_PATTERN`) layers on top of the per-area
    eligibility floor and either can fail independently.

    The ``rollup`` argument's ``counts`` field is the only surface
    consumed; the per-axis breakdowns are not read here (translator
    correctness needs full-key resolution to filter on quartile +
    decision for the depth_distinction pattern).
    """

    counts = dict(rollup.counts)
    patches: list[BriefPatch] = []
    for area in eligible_areas:
        for translator in _PATTERN_TRANSLATORS:
            patch = translator(area, counts=counts)
            if patch is not None:
                patches.append(patch)
    return patches


# ---------------------------------------------------------------------------
# Public translator: Designer rubric refinements
# ---------------------------------------------------------------------------


def translate_designer_rubric_refinements(
    *,
    feedback_marker_distribution: dict[str, dict[str, int]],
    discipline: str,
    current_rubric: Any,
) -> list[BriefPatch]:
    """Wrap ``propose_rubric_refinements`` output as ``BriefPatch``.

    This translator keeps Designer rubric refinements on the same surface as
    capability-area patches. Empty distributions, disciplines, or non-dict rubrics
    return ``[]`` like the underlying proposer. The import stays lazy so callers
    using calibration-only paths do not load Designer dependencies.
    """

    if not feedback_marker_distribution:
        return []
    if not discipline:
        return []
    if not isinstance(current_rubric, dict):
        return []

    # Lazy import: see module docstring rationale.
    from market_intelligence.design_market_intelligence import (
        propose_rubric_refinements,
    )

    hunks = propose_rubric_refinements(
        feedback_marker_distribution=feedback_marker_distribution,
        discipline=discipline,
        current_rubric=current_rubric,
    )
    patches: list[BriefPatch] = []
    for hunk in hunks:
        n_markers = _per_principle_marker_count(
            hunk.label, feedback_marker_distribution
        )
        patches.append(
            BriefPatch(
                kind=PATCH_KIND_RUBRIC_REFINE,
                target_section=hunk.section,
                # Designer rubric refinements are per-principle, not
                # per-capability-area; the empty string is the explicit
                # "no capability area" sentinel matching the test pin
                # at ``test_designer_routing_emits_rubric_refine_patches``.
                capability_area="",
                label=hunk.label,
                rationale=hunk.rationale,
                payload={
                    "kind": hunk.kind,
                    "before": hunk.before,
                    "after": hunk.after,
                },
                n_markers_for_kind=n_markers,
            )
        )
    return patches


def _per_principle_marker_count(
    label: str,
    feedback_marker_distribution: dict[str, dict[str, int]],
) -> int:
    """Best-effort principle-name to marker-count lookup.

    The proposer embeds the principle in the hunk label, but the hunk has no
    structured principle field. Match known principle names against that label
    and sum ``useful_guidance`` plus ``off_rubric`` markers. Return 0 when the
    label does not identify a principle so every patch still carries a numeric
    marker count.
    """

    for principle, markers in feedback_marker_distribution.items():
        if principle in label:
            useful = int(markers.get("useful_guidance", 0))
            off_rubric = int(markers.get("off_rubric", 0))
            return useful + off_rubric
    return 0


__all__ = [
    "BriefPatch",
    "HIGH_CONFIDENCE_QUARTILE",
    "MIN_MARKERS_PER_PATTERN",
    "PATCH_KIND_CALIBRATION_EXAMPLES",
    "PATCH_KIND_DEPTH_DISTINCTION",
    "PATCH_KIND_NON_FIT_PATTERN",
    "PATCH_KIND_RUBRIC_REFINE",
    "SAVE_DECISION",
    "translate_designer_rubric_refinements",
    "translate_eligible_areas",
]
