"""Designer Slice 9 — design-market intelligence reflection polish.

Pins:

- ``assemble_design_market_artifact`` produces stable markdown that
  carries the documented sections (pool composition, discipline
  distribution, fields, tools, recruiter feedback, cross-check
  disagreement, proposed rubric refinements).
- Empty inputs collapse to honest "no data" text rather than absent
  sections (the recruiter sees what's measured, not what's hidden).
- ``propose_rubric_refinements`` proposes weight-up / weight-down
  hunks based on ``useful_guidance`` − ``off_rubric`` deltas crossing
  the threshold; tied or near-tied principles produce no proposal.
- The proposal cap honors ``max_hunks``.
- ``reflection_design_rubric_drift`` returns None for byte-equal
  rubrics; descriptor string for mutations.
"""

from __future__ import annotations

import pytest

from market_intelligence.design_market_intelligence import (
    DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT,
    DesignMarketArtifact,
    RubricRefineHunk,
    assemble_design_market_artifact,
    propose_rubric_refinements,
    reflection_design_rubric_drift,
)


# ---------------------------------------------------------------------------
# assemble_design_market_artifact
# ---------------------------------------------------------------------------


def test_artifact_contains_all_documented_sections() -> None:
    artifact = assemble_design_market_artifact(
        brief_state_key="senior_product_designer",
        pool_composition={"behance": 12, "google_cse": 5},
        discipline_distribution={"product": 8, "brand": 4},
        top_fields=[("UI/UX", 10), ("Branding", 4)],
        top_tools=[("Figma", 9), ("After Effects", 3)],
        feedback_marker_distribution={
            "Visual hierarchy": {"useful_guidance": 5, "off_rubric": 1}
        },
        cross_check_disagreement_count=1,
        cross_check_total_count=3,
    )
    md = artifact.markdown
    assert "# Design-market intelligence" in md
    assert "## Pool composition" in md
    assert "## Discipline distribution" in md
    assert "## Fields surfacing" in md
    assert "## Tool stack surfacing" in md
    assert "## Recruiter feedback per principle" in md
    assert "## Cross-check (Sonnet 4.6) disagreement" in md
    # Counts surface.
    assert "behance" in md
    assert "12" in md
    assert "Visual hierarchy" in md
    assert "useful_guidance: 5" in md


def test_artifact_handles_empty_inputs_with_no_data_copy() -> None:
    artifact = assemble_design_market_artifact(
        brief_state_key="empty_test",
        pool_composition={},
        discipline_distribution={},
        top_fields=[],
        top_tools=[],
        feedback_marker_distribution={},
    )
    assert "_No candidates surfaced._" in artifact.markdown
    assert "_No discipline tags collected._" in artifact.markdown
    assert "_No recruiter feedback yet._" in artifact.markdown


def test_artifact_omits_cross_check_section_when_no_cross_check_run() -> None:
    artifact = assemble_design_market_artifact(
        brief_state_key="x",
        pool_composition={"behance": 1},
        discipline_distribution={},
        top_fields=[],
        top_tools=[],
        feedback_marker_distribution={},
        cross_check_disagreement_count=0,
        cross_check_total_count=0,
    )
    assert "Cross-check" not in artifact.markdown


def test_artifact_renders_proposed_rubric_refinements_when_present() -> None:
    hunk = RubricRefineHunk(
        label="Weight Visual hierarchy higher for product",
        section="design_rubric.discipline_weight_overrides",
        kind="rubric_refine",
        before="product: {Visual hierarchy: 1.0}",
        after="product: {Visual hierarchy: 1.3}",
        rationale="Recruiters consistently marked Visual hierarchy as useful guidance.",
    )
    artifact = assemble_design_market_artifact(
        brief_state_key="x",
        pool_composition={"behance": 1},
        discipline_distribution={},
        top_fields=[],
        top_tools=[],
        feedback_marker_distribution={},
        proposed_hunks=(hunk,),
    )
    assert "## Proposed rubric refinements" in artifact.markdown
    assert "Weight Visual hierarchy higher for product" in artifact.markdown


# ---------------------------------------------------------------------------
# propose_rubric_refinements
# ---------------------------------------------------------------------------


def test_propose_returns_empty_for_no_feedback() -> None:
    proposals = propose_rubric_refinements(
        feedback_marker_distribution={},
        discipline="product",
        current_rubric={},
    )
    assert proposals == []


def test_propose_weight_up_when_useful_minus_off_rubric_meets_threshold() -> None:
    proposals = propose_rubric_refinements(
        feedback_marker_distribution={
            "Visual hierarchy": {
                "useful_guidance": DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT,
                "off_rubric": 0,
            }
        },
        discipline="product",
        current_rubric={"discipline_weight_overrides": {"product": {}}},
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.section == "design_rubric.discipline_weight_overrides"
    assert "higher" in p.label
    assert "1.0" in p.before
    assert "1.3" in p.after  # default +0.3 step


def test_propose_weight_down_when_off_rubric_exceeds_useful_at_threshold() -> None:
    proposals = propose_rubric_refinements(
        feedback_marker_distribution={
            "Visual hierarchy": {
                "useful_guidance": 0,
                "off_rubric": DEFAULT_FEEDBACK_THRESHOLD_FOR_REFINEMENT,
            }
        },
        discipline="product",
        current_rubric={"discipline_weight_overrides": {"product": {"Visual hierarchy": 1.0}}},
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert "lower" in p.label
    assert "0.7" in p.after  # default -0.3 step


def test_propose_returns_empty_when_delta_is_below_threshold() -> None:
    proposals = propose_rubric_refinements(
        feedback_marker_distribution={
            "Visual hierarchy": {"useful_guidance": 1, "off_rubric": 0}
        },
        discipline="product",
        current_rubric={},
    )
    assert proposals == []


def test_propose_caps_at_max_hunks() -> None:
    distribution = {
        f"Principle {i}": {"useful_guidance": 5, "off_rubric": 0}
        for i in range(20)
    }
    proposals = propose_rubric_refinements(
        feedback_marker_distribution=distribution,
        discipline="product",
        current_rubric={},
        max_hunks=3,
    )
    assert len(proposals) == 3


def test_propose_ranks_strongest_signal_first() -> None:
    distribution = {
        "Strong signal": {"useful_guidance": 10, "off_rubric": 0},  # delta=10
        "Weak signal": {"useful_guidance": 4, "off_rubric": 0},  # delta=4
    }
    proposals = propose_rubric_refinements(
        feedback_marker_distribution=distribution,
        discipline="product",
        current_rubric={},
    )
    assert proposals[0].label == "Weight Strong signal higher for product"


# ---------------------------------------------------------------------------
# reflection_design_rubric_drift
# ---------------------------------------------------------------------------


def test_reflection_drift_returns_none_when_seed_has_no_rubric() -> None:
    assert (
        reflection_design_rubric_drift(seeded={}, polished={}) is None
    )


def test_reflection_drift_returns_none_for_byte_equal_rubric() -> None:
    rubric = {"principles": [{"name": "x"}]}
    assert (
        reflection_design_rubric_drift(
            seeded={"design_rubric": rubric},
            polished={"design_rubric": dict(rubric)},
        )
        is None
    )


def test_reflection_drift_returns_dropped_when_polished_drops_rubric() -> None:
    rubric = {"principles": [{"name": "x"}]}
    descriptor = reflection_design_rubric_drift(
        seeded={"design_rubric": rubric},
        polished={},
    )
    assert descriptor == "dropped"


def test_reflection_drift_returns_descriptor_for_mutation() -> None:
    rubric_a = {"principles": [{"name": "x"}]}
    rubric_b = {"principles": [{"name": "x"}, {"name": "y"}]}
    descriptor = reflection_design_rubric_drift(
        seeded={"design_rubric": rubric_a},
        polished={"design_rubric": rubric_b},
    )
    assert descriptor is not None
    assert "mutated" in descriptor
