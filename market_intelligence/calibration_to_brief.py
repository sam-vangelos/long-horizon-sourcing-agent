"""Calibration → V2 brief patch translator (Slice 3.3).

Phase 3.3 of the multi-agent execution plan
(``plans/multi-agent-execution-plan.md`` §3.3, lines 927-956). Sits
between the threshold layer
(``market_intelligence/calibration_thresholds.py``: Slice 3.2) and the
reflection-pipeline integration
(``market_intelligence/reflection.py:reflection_phase_propose``: Slice
3.4). Pure function over a :class:`CalibrationRollup` plus a list of
:class:`EligibleArea`; emits :class:`BriefPatch` instances the
reflection pipeline projects onto Gate-2 hunks.

## Three patterns the slice card prescribes

Per the slice card lines 932-941:

1. **High ``wrong`` rate on a capability area → ``non_fit_pattern``
   patch.** The recruiter has marked Cloris's reads in this area
   "wrong" enough times that the area itself looks like a false-
   positive shape. Emits a V2 ``non_fit_patterns`` payload (label /
   description / why_not / examples) the recruiter can edit before
   approving at Gate 2.

2. **High ``off_rubric`` rate on ``SAVE`` rows in the high-confidence
   band → ``depth_distinction`` patch.** The slice card says
   "confidence > 0.7"; the aggregator already buckets confidence into
   static quartiles (``q4 = [0.75, 1.0]``), and the threshold layer's
   weighted bonus uses the same ``> 0.7`` cut. We pin to the q4
   quartile here for two reasons: the rollup carries quartiles, not
   raw confidence, so the cut is operational; and ``q4`` is a touch
   stricter (``>= 0.75`` vs ``> 0.7``), which biases toward fewer but
   higher-signal patches. ``test_off_rubric_high_confidence_pinned_to_q4_constant``
   pins this so any future widening to ``q3 ∪ q4`` lands as a
   deliberate diff.

3. **High ``useful`` rate on a capability area → ``calibration_examples``
   patch.** The recruiter has confirmed Cloris's reads. Emits a V2
   ``transferability_examples`` payload (the canonical field per the
   deprecation manifest at
   :data:`shared.brief_v2_schema.DEPRECATED_KEYS_BY_VERSION` —
   ``calibration_examples`` is deprecated in favor of
   ``transferability_examples``). The slice card's
   ``calibration_examples`` vocabulary stays in :data:`PATCH_KIND_CALIBRATION_EXAMPLES`
   for traceability against the plan, but the actual brief field the
   patch targets is the canonical V2 ``transferability_examples``
   list.

## Per-pattern marker floor

The threshold layer enforces a per-area eligibility floor
(``MIN_MARKERS_PER_AREA = 5`` weighted markers); the translator adds
a per-pattern floor (:data:`MIN_MARKERS_PER_PATTERN` = 3 raw markers
of the pattern's marker class) on top. Without the per-pattern floor,
an area that cleared upstream eligibility on combined signal could
still produce a noisy patch when no single marker class crosses a
meaningful count. The two floors compose: an area with 6 weighted
markers split 2 wrong / 2 off_rubric / 2 useful clears upstream but
emits no patches because no individual pattern reaches 3.

## Multi-pattern emission per area

When more than one pattern's floor is met for a single eligible area,
all qualifying patches surface — the recruiter approves or rejects each
independently at Gate 2. Inner ordering is fixed:
``non_fit_pattern`` → ``depth_distinction`` → ``calibration_examples``.
Multi-area ordering preserves the eligible-area input order
(threshold-layer ranking propagates through; see
``calibration_thresholds.select_eligible_areas`` for the ranking key).

## Designer routing

:func:`translate_designer_rubric_refinements` wraps the existing
:func:`market_intelligence.design_market_intelligence.propose_rubric_refinements`
output as :class:`BriefPatch` instances. This keeps Slice 3.5
(Designer per-principle wiring) and Slice 3.4 (reflection ingestion)
on a uniform ``BriefPatch`` surface — the reflection pipeline doesn't
need a second projection helper for rubric refinements vs. capability-
area patches. Defensive on empty inputs (mirrors
``designer/run_end.compute_designer_rubric_refinement_hunks``'s failure
posture).

## Out of scope for this slice

- Reflection-pipeline ingestion (Slice 3.4 at
  ``market_intelligence/reflection.py:reflection_phase_propose``) — the
  caller wires aggregator → threshold → translator → hunks.
- Brief application semantics (commit-time mutation of structured
  fields like ``non_fit_patterns`` / ``transferability_examples``) —
  the existing ``_apply_hunk_to_brief`` handles list-of-strings and
  prose sections; structured-payload application is its own follow-up.
  Until then, calibration patches surface at Gate 2 as recruiter-
  reviewable proposals; the recruiter can approve and Cloris records
  the approval, but the actual brief mutation for structured fields
  may be a manual step until the apply-time extension lands.
- Per-pattern threshold tuning. The :data:`MIN_MARKERS_PER_PATTERN`
  constant is a starting point; once the
  ``calibration.proposer:eligible`` log accumulates real data
  (post-trial deployment), a dedicated calibration-tuning pass adjusts
  it.
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


# Quartile band the depth_distinction filter pins to. The slice card
# says "confidence > 0.7"; the aggregator's static bands at
# ``shared/runtime_state/calibration.py:201-217`` make this
# operational as ``q4 = [0.75, 1.0]``. Pinned to a constant rather
# than open-coded so a future widening (say ``q3 ∪ q4``) lands as a
# deliberate diff with
# ``tests/test_calibration_to_brief.py::test_off_rubric_high_confidence_pinned_to_q4_constant``
# in the PR.
HIGH_CONFIDENCE_QUARTILE: str = "q4"


# Terminal-decision string the depth_distinction rule scopes to.
# Mirrors the runtime-state writer's terminal_decision contract at
# ``shared/runtime_state/store.py``. Pinning the constant means the
# depth_distinction filter cannot drift away from the canonical
# decision-enum value silently.
SAVE_DECISION: str = "SAVE"


# Patch ``kind`` strings. These are the surface the
# reflection-pipeline integration (Slice 3.4) inspects when projecting
# patches onto Gate-2 hunks. Deliberately mirrors the slice card's
# vocabulary (``non_fit_pattern`` / ``depth_distinction`` /
# ``calibration_examples``) for traceability against the plan, even
# where the actual brief field the patch targets has a different
# canonical name (e.g., ``calibration_examples`` → V2
# ``transferability_examples``).
PATCH_KIND_NON_FIT_PATTERN: str = "non_fit_pattern"
PATCH_KIND_DEPTH_DISTINCTION: str = "depth_distinction"
PATCH_KIND_CALIBRATION_EXAMPLES: str = "calibration_examples"
PATCH_KIND_RUBRIC_REFINE: str = "rubric_refine"


# ---------------------------------------------------------------------------
# Dataclass — the wire surface to Slice 3.4
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BriefPatch:
    """One proposed brief patch derived from calibration markers.

    Frozen so callers can't mutate proposed patches in flight before
    reflection ingestion (Slice 3.4) wraps them as Gate-2 hunks.

    Fields:

    - ``kind`` is one of the ``PATCH_KIND_*`` constants. Stable
      identifier for downstream consumers (Slice 3.4's hunk
      projection picks rendering by kind).
    - ``target_section`` is the V2 brief schema field the patch
      mutates. For the three generic patterns this is
      ``non_fit_patterns`` / ``depth_distinction`` /
      ``transferability_examples``; for Designer routing it is the
      ``RubricRefineHunk.section`` (e.g.,
      ``design_rubric.discipline_weight_overrides``). Dotted
      sections are passed through verbatim — the apply-time helper
      decides how to deep-write them.
    - ``capability_area`` is the area the patch is attributed to
      (for Designer routing this is the empty string — the patch
      is per-principle, not per-capability-area).
    - ``label`` is a short editorial label (≤ 80 chars
      recommended). Slice 3.4's hunk projection surfaces it.
    - ``rationale`` is recruiter-readable Cloris-voice prose
      explaining why Cloris is proposing this patch.
    - ``payload`` is the structured payload for the brief field —
      see the per-pattern functions for shape contracts.
    - ``n_markers_for_kind`` is the count of markers that triggered
      this specific pattern (for Designer routing: the per-principle
      ``useful_guidance + off_rubric`` total). Surfaced for telemetry
      / per-hunk confidence weighting; not required for apply-time.
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
    """Slice card lines 932-933. ``wrong`` markers in ``area``,
    any quartile, any decision. ``>= MIN_MARKERS_PER_PATTERN`` →
    one patch.

    Payload shape mirrors :class:`shared.brief_schema.NonFitPattern`
    (label / description / why_not / examples). ``examples`` is
    deliberately empty — the recruiter fills in concrete examples at
    Gate 2 if they want to keep the pattern.
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
    """Slice card lines 934-936. ``off_rubric`` markers in ``area``
    on ``SAVE`` rows in the q4 quartile. ``>= MIN_MARKERS_PER_PATTERN``
    → one patch.

    Payload carries ``section_path`` (``depth_distinction.edge_case_guidance``
    — the prose field where the addendum lands) and ``addendum`` (the
    proposed prose). The reflection-pipeline integration (Slice 3.4)
    inspects ``section_path`` when projecting onto a Gate-2 hunk so the
    apply-time helper knows the target is a nested prose field, not a
    list-of-dicts.
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
    """Slice card lines 937-938. ``useful`` markers in ``area``,
    any quartile, any decision. ``>= MIN_MARKERS_PER_PATTERN`` → one
    patch.

    Payload shape mirrors :class:`shared.brief_schema.TransferabilityExample`
    (result / source_context / target_context / rationale). The
    target_section is the canonical V2 ``transferability_examples``
    field, not the deprecated ``calibration_examples`` field; see the
    deprecation manifest at
    :data:`shared.brief_v2_schema.DEPRECATED_KEYS_BY_VERSION`.
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
    """Wrap :func:`propose_rubric_refinements` output as ``BriefPatch``.

    Slice 3.5 (Designer per-principle wiring) persists
    ``RubricRefineHunk`` objects via the Designer session
    orchestrator's run-end hook (``designer/run_end.py``). This helper
    is the *translator-side* surface: it lets a caller (the
    reflection-pipeline integration if/when it wants to compute
    Designer patches inline rather than load them from the
    Slice-3.5-persisted artifact) get a uniform ``BriefPatch`` list
    that mixes capability-area patterns and rubric refinements
    together.

    Mirrors the failure posture at
    ``designer/run_end.compute_designer_rubric_refinement_hunks`` —
    every degenerate input (empty distribution, empty discipline,
    non-dict rubric) returns ``[]`` rather than raising. That keeps
    the call site at Slice 3.4 unconditional: it can call this
    helper without first checking whether the brief is Designer-
    targeted.

    Lazy import of :func:`propose_rubric_refinements` so this module
    stays free of any transitive Designer dependency at import time —
    the threshold layer + capability-area translation paths don't
    need Designer code loaded.
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
    """Best-effort principle-name → marker count lookup.

    The proposer's hunk label embeds the principle name in either
    ``"Weight {principle} higher for {discipline}"`` or
    ``"Weight {principle} lower for {discipline}"`` form
    (``market_intelligence/design_market_intelligence.py:262-289``).
    We don't get the principle name back as a structured field on the
    hunk (the underlying contract was string-shaped for the recruiter-
    facing card), so we recover it by matching the principle names in
    ``feedback_marker_distribution`` against the label.

    Returns 0 when no principle matches — that keeps the
    ``n_markers_for_kind`` field present on every BriefPatch instead
    of optional, while staying defensive against a future label
    reshaping that breaks the parse.
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
