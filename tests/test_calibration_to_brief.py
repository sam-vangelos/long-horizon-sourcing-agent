"""Tests for the calibration-to-brief translator (Slice 3.3).

Multi-Agent Execution Plan §3.3 (lines 884-911). Pins:

- Per-pattern rules (slice card lines 901-908):
  - High ``wrong`` rate → ``non_fit_pattern`` patch with the V2
    ``non_fit_patterns`` payload shape.
  - High ``off_rubric`` rate on ``SAVE`` rows in the q4 confidence
    band → ``depth_distinction`` clarification patch (the q4 cut is
    a touch stricter than the slice card's "> 0.7" wording; the
    translator's module docstring documents the trade-off).
  - High ``useful`` rate → ``calibration_examples`` patch with the V2
    ``transferability_examples`` payload shape (the slice card's
    "calibration_examples" vocabulary maps to the canonical V2
    field per the deprecation manifest at
    :data:`shared.brief_v2_schema.DEPRECATED_KEYS_BY_VERSION`).
- Per-pattern marker floor (``MIN_MARKERS_PER_PATTERN`` = 3): an
  eligible area whose per-pattern count falls below the floor produces
  no patch for that pattern.
- Multi-pattern emission per area when more than one pattern's floor
  is met (deterministic order: non_fit_pattern → depth_distinction →
  calibration_examples).
- Multi-area emission preserves the eligible-area input order
  (the threshold layer's ranking propagates through).
- Designer-specific routing: ``translate_designer_rubric_refinements``
  wraps :func:`market_intelligence.design_market_intelligence.propose_rubric_refinements`
  output as ``BriefPatch`` instances. Defensive on empty inputs.
- Patch payload shapes match the V2 brief schema fields they target
  (sanity check against ``shared/brief_schema.py`` dataclasses).

The translator is a pure function over a ``CalibrationRollup`` + a
list of ``EligibleArea``. Tests build those directly (no DB walk);
end-to-end aggregator coverage already lives in
``tests/test_calibration_aggregator.py`` and threshold-layer coverage
in ``tests/test_calibration_thresholds.py``.
"""

from __future__ import annotations

from collections import Counter

import pytest

from market_intelligence.calibration_thresholds import EligibleArea
from market_intelligence.calibration_to_brief import (
    BriefPatch,
    HIGH_CONFIDENCE_QUARTILE,
    MIN_MARKERS_PER_PATTERN,
    PATCH_KIND_CALIBRATION_EXAMPLES,
    PATCH_KIND_DEPTH_DISTINCTION,
    PATCH_KIND_NON_FIT_PATTERN,
    PATCH_KIND_RUBRIC_REFINE,
    SAVE_DECISION,
    translate_designer_rubric_refinements,
    translate_eligible_areas,
)
from shared.runtime_state.calibration import (
    CalibrationRollup,
    CalibrationRollupKey,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _eligible(
    area: str,
    *,
    n_markers: int = 6,
    weighted: int = 6,
    saturated: bool = False,
) -> EligibleArea:
    """Build an ``EligibleArea`` for translator-only tests.

    Real eligibility math is exercised in
    ``tests/test_calibration_thresholds.py``; here we just need a
    well-shaped area object the translator can iterate.
    """

    return EligibleArea(
        capability_area=area,
        n_markers=n_markers,
        confidence_weighted_count=weighted,
        saturated=saturated,
    )


def _rollup_from_keys(
    *,
    counts: dict[CalibrationRollupKey, int],
) -> CalibrationRollup:
    """Build a ``CalibrationRollup`` from a per-key counts dict.

    Per-axis breakdowns + weighted-by-area are derived deterministically
    so the fixture only needs to declare the full-key counts. Mirrors
    the aggregator's row-walk math in spirit; the translator only reads
    ``counts`` so the per-axis fields are populated for shape
    completeness, not for translator correctness.
    """

    by_marker: Counter[str] = Counter()
    by_area: Counter[str | None] = Counter()
    by_quartile: Counter[str] = Counter()
    by_decision: Counter[str | None] = Counter()
    weighted_by_area: Counter[str | None] = Counter()
    for key, count in counts.items():
        by_marker[key.marker_value] += count
        by_area[key.capability_area] += count
        by_quartile[key.confidence_quartile] += count
        by_decision[key.terminal_decision] += count
        weighted_by_area[key.capability_area] += count
    return CalibrationRollup(
        brief_id="brief-test",
        source=None,
        total_markers=sum(counts.values()),
        counts=dict(counts),
        by_marker_value=dict(by_marker),
        by_capability_area=dict(by_area),
        by_confidence_quartile=dict(by_quartile),
        by_terminal_decision=dict(by_decision),
        weighted_markers_by_area=dict(weighted_by_area),
    )


def _key(
    area: str,
    marker: str,
    *,
    quartile: str = "q3",
    decision: str | None = "SAVE",
) -> CalibrationRollupKey:
    """Shorthand for :class:`CalibrationRollupKey` instantiation."""

    return CalibrationRollupKey(
        capability_area=area,
        marker_value=marker,
        confidence_quartile=quartile,
        terminal_decision=decision,
    )


# ---------------------------------------------------------------------------
# Per-pattern: high `wrong` rate → non_fit_pattern
# ---------------------------------------------------------------------------


def test_wrong_dominant_area_emits_non_fit_pattern_patch() -> None:
    """Slice card lines 901-902. 5 wrong markers in one area → one
    ``non_fit_pattern`` patch with the V2 ``non_fit_patterns`` payload
    shape.
    """

    area = "Foundation Models Research"
    rollup = _rollup_from_keys(
        counts={_key(area, "wrong"): 5},
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert len(patches) == 1
    patch = patches[0]
    assert patch.kind == PATCH_KIND_NON_FIT_PATTERN
    assert patch.target_section == "non_fit_patterns"
    assert patch.capability_area == area
    assert patch.n_markers_for_kind == 5

    payload = patch.payload
    assert set(payload.keys()) == {"label", "description", "why_not", "examples"}
    assert isinstance(payload["label"], str) and area in payload["label"]
    assert isinstance(payload["description"], str)
    assert "5" in payload["description"]
    assert isinstance(payload["why_not"], str)
    assert payload["examples"] == []


def test_wrong_below_floor_does_not_emit_non_fit_pattern() -> None:
    """A wrong count of 2 (below ``MIN_MARKERS_PER_PATTERN`` = 3) does
    not produce a non_fit_pattern patch even when the area cleared
    upstream eligibility on combined signal.
    """

    area = "Area Below Floor"
    rollup = _rollup_from_keys(
        counts={
            _key(area, "wrong"): 2,
            _key(area, "useful"): 2,
        },
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert patches == []


def test_wrong_at_floor_emits_non_fit_pattern() -> None:
    """The floor is ``>=``, not ``>``: 3 wrong markers → patch fires.
    Pins the off-by-one.
    """

    area = "Just Enough"
    rollup = _rollup_from_keys(
        counts={_key(area, "wrong"): MIN_MARKERS_PER_PATTERN},
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert len(patches) == 1
    assert patches[0].kind == PATCH_KIND_NON_FIT_PATTERN
    assert patches[0].n_markers_for_kind == MIN_MARKERS_PER_PATTERN


# ---------------------------------------------------------------------------
# Per-pattern: high `off_rubric` rate on saves with confidence > 0.7
# ---------------------------------------------------------------------------


def test_off_rubric_high_confidence_saves_emit_depth_distinction_patch() -> None:
    """Slice card lines 903-906. ``off_rubric`` markers on ``SAVE``
    rows in the q4 confidence band → ``depth_distinction`` patch.
    """

    area = "AI Infrastructure"
    rollup = _rollup_from_keys(
        counts={
            _key(area, "off_rubric", quartile="q4", decision="SAVE"): 4,
        },
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert len(patches) == 1
    patch = patches[0]
    assert patch.kind == PATCH_KIND_DEPTH_DISTINCTION
    assert patch.target_section == "depth_distinction"
    assert patch.capability_area == area
    assert patch.n_markers_for_kind == 4

    payload = patch.payload
    assert set(payload.keys()) == {"section_path", "addendum"}
    assert payload["section_path"] == "depth_distinction.edge_case_guidance"
    assert area in payload["addendum"]
    assert "4" in payload["addendum"]


def test_off_rubric_low_confidence_does_not_emit_depth_distinction() -> None:
    """``off_rubric`` markers in q3 (below the high-confidence band)
    do not surface a depth_distinction patch — the slice card
    explicitly scopes to high-confidence saves.
    """

    area = "Low Conf Area"
    rollup = _rollup_from_keys(
        counts={
            _key(area, "off_rubric", quartile="q3", decision="SAVE"): 5,
        },
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert patches == []


def test_off_rubric_on_reject_does_not_emit_depth_distinction() -> None:
    """``off_rubric`` markers on ``REJECT`` rows (false negatives) are
    a different signal class — the depth_distinction rule is for
    saved candidates the recruiter says we judged on the wrong axis.
    """

    area = "Reject Area"
    rollup = _rollup_from_keys(
        counts={
            _key(area, "off_rubric", quartile="q4", decision="REJECT"): 5,
        },
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert patches == []


def test_off_rubric_high_confidence_pinned_to_q4_constant() -> None:
    """Pin the high-confidence quartile constant so a future widening
    (e.g., q3+q4) lands as a deliberate diff with this assertion in
    the PR.
    """

    assert HIGH_CONFIDENCE_QUARTILE == "q4"


def test_off_rubric_below_floor_does_not_emit_depth_distinction() -> None:
    """2 off_rubric+SAVE+q4 markers (below floor) → no patch."""

    area = "Few Markers"
    rollup = _rollup_from_keys(
        counts={
            _key(area, "off_rubric", quartile="q4", decision="SAVE"): 2,
        },
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert patches == []


# ---------------------------------------------------------------------------
# Per-pattern: high `useful` rate → calibration_examples
# ---------------------------------------------------------------------------


def test_useful_dominant_area_emits_calibration_examples_patch() -> None:
    """Slice card lines 907-908. 4 useful markers → one
    ``calibration_examples`` patch (target section is the canonical
    V2 ``transferability_examples`` field per the deprecation
    manifest).
    """

    area = "Frontend Performance"
    rollup = _rollup_from_keys(
        counts={_key(area, "useful"): 4},
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert len(patches) == 1
    patch = patches[0]
    assert patch.kind == PATCH_KIND_CALIBRATION_EXAMPLES
    assert patch.target_section == "transferability_examples"
    assert patch.capability_area == area
    assert patch.n_markers_for_kind == 4

    payload = patch.payload
    assert set(payload.keys()) == {
        "result",
        "source_context",
        "target_context",
        "rationale",
    }
    assert payload["result"] == "transfers"
    assert payload["source_context"] == area
    assert payload["target_context"] == area
    assert isinstance(payload["rationale"], str) and area in payload["rationale"]


def test_useful_below_floor_does_not_emit_calibration_examples() -> None:
    """2 useful markers (below floor) → no patch."""

    area = "Quiet Area"
    rollup = _rollup_from_keys(counts={_key(area, "useful"): 2})

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert patches == []


# ---------------------------------------------------------------------------
# Multi-pattern per area + multi-area emission
# ---------------------------------------------------------------------------


def test_area_with_two_patterns_above_floor_emits_two_patches() -> None:
    """Multiple patches per area are allowed when more than one
    pattern's floor is met. Inner order: ``non_fit_pattern`` →
    ``depth_distinction`` → ``calibration_examples``.
    """

    area = "Mixed Signal Area"
    rollup = _rollup_from_keys(
        counts={
            _key(area, "wrong"): 4,
            _key(area, "useful"): 5,
        },
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert [p.kind for p in patches] == [
        PATCH_KIND_NON_FIT_PATTERN,
        PATCH_KIND_CALIBRATION_EXAMPLES,
    ]
    assert all(p.capability_area == area for p in patches)


def test_area_with_all_three_patterns_emits_three_in_pinned_order() -> None:
    """All three generic patterns above floor for one area → three
    patches in the pinned order.
    """

    area = "Triple Threat"
    rollup = _rollup_from_keys(
        counts={
            _key(area, "wrong"): 3,
            _key(area, "off_rubric", quartile="q4", decision="SAVE"): 3,
            _key(area, "useful"): 3,
        },
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert [p.kind for p in patches] == [
        PATCH_KIND_NON_FIT_PATTERN,
        PATCH_KIND_DEPTH_DISTINCTION,
        PATCH_KIND_CALIBRATION_EXAMPLES,
    ]


def test_multi_area_preserves_eligible_area_input_order() -> None:
    """Two eligible areas → patches emitted in the order the
    threshold layer ranked them. Provenance from the upstream
    ranking is preserved end-to-end.
    """

    area_a = "Area Alpha"
    area_b = "Area Bravo"
    rollup = _rollup_from_keys(
        counts={
            _key(area_a, "useful"): 4,
            _key(area_b, "wrong"): 4,
        },
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area_b), _eligible(area_a)],
    )

    assert [(p.capability_area, p.kind) for p in patches] == [
        (area_b, PATCH_KIND_NON_FIT_PATTERN),
        (area_a, PATCH_KIND_CALIBRATION_EXAMPLES),
    ]


def test_eligible_area_with_no_pattern_above_floor_emits_nothing() -> None:
    """An area can clear upstream eligibility (combined weighted count
    >= 5) but still produce no patches if no single pattern's floor
    is met. The translator's per-pattern floor is independent.
    """

    area = "Spread Thin"
    rollup = _rollup_from_keys(
        counts={
            _key(area, "wrong"): 2,
            _key(area, "useful"): 2,
            _key(area, "off_rubric", quartile="q4", decision="SAVE"): 2,
        },
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible(area)],
    )

    assert patches == []


def test_empty_eligible_areas_emits_nothing() -> None:
    """No eligible areas → no patches, no exception."""

    rollup = _rollup_from_keys(counts={})

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[],
    )

    assert patches == []


# ---------------------------------------------------------------------------
# Designer routing
# ---------------------------------------------------------------------------


def _designer_rubric_with_principle(principle: str) -> dict:
    """Minimal rubric carrying one principle the proposer can score."""

    return {
        "principles": [
            {
                "name": principle,
                "description": "test principle",
                "anchors": {
                    "bad": "x",
                    "okay": "x",
                    "good": "x",
                    "excellent": "x",
                },
                "weight": 1.0,
            }
        ],
        "discipline_weight_overrides": {},
        "calibration_exemplars": [],
    }


def test_designer_routing_emits_rubric_refine_patches() -> None:
    """Strong positive feedback (useful_guidance >> off_rubric) on a
    principle → a ``rubric_refine`` ``BriefPatch`` wrapping the
    underlying ``RubricRefineHunk`` shape.
    """

    principle = "Visual hierarchy"
    rubric = _designer_rubric_with_principle(principle)

    patches = translate_designer_rubric_refinements(
        feedback_marker_distribution={
            principle: {"useful_guidance": 5, "off_rubric": 0},
        },
        discipline="product",
        current_rubric=rubric,
    )

    assert len(patches) == 1
    patch = patches[0]
    assert patch.kind == PATCH_KIND_RUBRIC_REFINE
    assert patch.target_section == "design_rubric.discipline_weight_overrides"
    assert patch.capability_area == ""
    assert principle in patch.label
    assert "product" in patch.label
    assert set(patch.payload.keys()) == {"kind", "before", "after"}
    assert patch.payload["kind"] == "rubric_refine"
    assert principle in patch.payload["before"]
    assert principle in patch.payload["after"]


def test_designer_routing_empty_feedback_emits_nothing() -> None:
    """Empty per-principle distribution → empty patch list."""

    patches = translate_designer_rubric_refinements(
        feedback_marker_distribution={},
        discipline="product",
        current_rubric=_designer_rubric_with_principle("X"),
    )

    assert patches == []


def test_designer_routing_empty_discipline_emits_nothing() -> None:
    """Empty discipline → empty patch list (mirrors the
    ``compute_designer_rubric_refinement_hunks`` failure posture at
    ``designer/run_end.py:104-110``).
    """

    patches = translate_designer_rubric_refinements(
        feedback_marker_distribution={"X": {"useful_guidance": 5}},
        discipline="",
        current_rubric=_designer_rubric_with_principle("X"),
    )

    assert patches == []


def test_designer_routing_invalid_rubric_emits_nothing() -> None:
    """Non-dict rubric → empty patch list."""

    patches = translate_designer_rubric_refinements(
        feedback_marker_distribution={"X": {"useful_guidance": 5}},
        discipline="product",
        current_rubric=None,  # type: ignore[arg-type]
    )

    assert patches == []


def test_designer_routing_below_proposer_threshold_emits_nothing() -> None:
    """When the underlying ``propose_rubric_refinements`` doesn't
    cross its own threshold (delta below
    :data:`market_intelligence.design_market_intelligence.DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT`),
    no patches surface.
    """

    principle = "Borderline"
    rubric = _designer_rubric_with_principle(principle)

    patches = translate_designer_rubric_refinements(
        feedback_marker_distribution={
            principle: {"useful_guidance": 1, "off_rubric": 0},
        },
        discipline="product",
        current_rubric=rubric,
    )

    assert patches == []


# ---------------------------------------------------------------------------
# Constants & shape contract
# ---------------------------------------------------------------------------


def test_per_pattern_floor_constant_value() -> None:
    """Pin the per-pattern floor so a tuning change shows up in the
    PR diff with this assertion.
    """

    assert MIN_MARKERS_PER_PATTERN == 3


def test_save_decision_constant() -> None:
    """Pin the save-decision string so the depth_distinction filter
    stays aligned with the runtime-state writer's terminal_decision
    contract (``shared/runtime_state/store.py``).
    """

    assert SAVE_DECISION == "SAVE"


def test_brief_patch_is_frozen() -> None:
    """``BriefPatch`` is a frozen dataclass so callers can't mutate
    proposed patches in flight before reflection ingestion (Slice 3.4)
    consumes them.
    """

    patch = BriefPatch(
        kind=PATCH_KIND_NON_FIT_PATTERN,
        target_section="non_fit_patterns",
        capability_area="Area",
        label="x",
        rationale="y",
        payload={},
        n_markers_for_kind=3,
    )
    with pytest.raises(Exception):  # FrozenInstanceError under dataclasses
        patch.kind = "mutated"  # type: ignore[misc]


def test_unattributed_area_in_rollup_does_not_break_translator() -> None:
    """The aggregator's ``None`` capability_area bucket would never
    reach the translator (the threshold layer drops it at
    ``calibration_thresholds.py:251-253``). Defensive: if a future
    caller hands an eligible-area list that nevertheless includes a
    matching None-area marker in the rollup, the translator's
    per-area count walk simply does not match it (because the
    EligibleArea.capability_area is typed ``str``, never ``None``).
    """

    rollup = _rollup_from_keys(
        counts={
            _key("Real", "wrong"): 3,
            CalibrationRollupKey(
                capability_area=None,
                marker_value="wrong",
                confidence_quartile="q3",
                terminal_decision="SAVE",
            ): 50,
        },
    )

    patches = translate_eligible_areas(
        rollup=rollup,
        eligible_areas=[_eligible("Real")],
    )

    assert len(patches) == 1
    assert patches[0].capability_area == "Real"
    assert patches[0].n_markers_for_kind == 3
