"""Designer module Slice 1 — schema additions.

Pins the V2 schema contract for `design_rubric`:

- `design_rubric` is recognized at the V2 surface (legacy → V2 merge
  routes it into `v2_data`, not `unknown_keys`).
- A well-formed rubric (six principles × four anchors + discipline
  weight overrides + calibration exemplars + hard reject patterns)
  validates cleanly.
- Each malformation surfaces a structured `invalid_keys` descriptor
  pointing at the bad field — so the brief-edit endpoint can return
  a 422 the recruiter can act on without grepping a stack trace.
- The `"designer"` source is recognized in `SOURCE_CONFIG_*_BY_SOURCE`.

The default rubric at `config/design-rubrics/default.json` is the
canonical fixture for "what good looks like" — these tests load it as
the happy-path payload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.brief_v2_schema import (
    RECOGNIZED_DESIGN_DISCIPLINES,
    RECOGNIZED_RUBRIC_ANCHORS,
    RECOGNIZED_V2_KEYS,
    SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE,
    SOURCE_CONFIG_REQUIRED_KEYS_BY_SOURCE,
    BriefDesignRubric,
    BriefSchemaError,
    CalibrationExemplar,
    RubricPrinciple,
    merge_legacy_brief,
    validate_v2_brief,
)


# Path to the default rubric — kept relative to the repo root so the
# test is location-agnostic (works from `pytest tests/` and
# `pytest tests/test_designer_schema.py`).
_DEFAULT_RUBRIC_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "design-rubrics" / "default.json"
)


def _minimal_v2_brief() -> dict:
    return {
        "role_title": "Senior product designer",
        "capability_areas": [
            {"name": "Product surface design", "description": "Ships shipped product."}
        ],
        "depth_distinction": {
            "builder_definition": "Owns surface end-to-end.",
            "user_definition": "Iterates on shipped surfaces.",
            "edge_case_guidance": "Borderline = surface ownership.",
        },
    }


def _well_formed_rubric() -> dict:
    """A minimal but well-formed `design_rubric` (one principle, all anchors).

    Trimmed for test legibility; the production default rubric at
    `config/design-rubrics/default.json` carries six principles and
    is exercised separately via `test_default_rubric_validates`.
    """

    return {
        "principles": [
            {
                "name": "Visual hierarchy",
                "description": "How clearly the work guides the viewer's attention.",
                "anchors": {
                    "bad": "No discernible hierarchy.",
                    "okay": "Hierarchy is present but weak.",
                    "good": "Clear hierarchy.",
                    "excellent": "Hierarchy is purposeful and confident.",
                },
                "weight": 1.0,
            }
        ],
        "discipline_weight_overrides": {
            "product": {"Visual hierarchy": 1.3},
        },
        "calibration_exemplars": [
            {
                "portfolio_url": "https://example.com/portfolio",
                "discipline": "product",
                "verdict": "yes",
                "per_principle_reasoning": {"Visual hierarchy": "Strong primary focal points throughout."},
                "overall_reasoning": "Clearly senior-grade product design output.",
            }
        ],
        "hard_reject_patterns": ["layout-only portfolios with no shipped product evidence"],
    }


# ---------------------------------------------------------------------------
# RECOGNIZED_V2_KEYS / SOURCE_CONFIG_* registry membership
# ---------------------------------------------------------------------------


def test_design_rubric_is_recognized_v2_key() -> None:
    """Recruiter-authored `design_rubric` lands in v2_data, not unknown_keys."""

    assert "design_rubric" in RECOGNIZED_V2_KEYS


def test_designer_source_is_recognized_source_config_key() -> None:
    """Designer briefs can carry an empty `source_config.designer = {}` block."""

    assert "designer" in SOURCE_CONFIG_REQUIRED_KEYS_BY_SOURCE
    assert "designer" in SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE
    # No per-source keys for v1; the rubric lives at top-level.
    assert SOURCE_CONFIG_REQUIRED_KEYS_BY_SOURCE["designer"] == frozenset()
    assert SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE["designer"] == frozenset()


def test_recognized_disciplines_cover_known_set() -> None:
    """The wizard discipline picker uses this set."""

    assert RECOGNIZED_DESIGN_DISCIPLINES == frozenset(
        {"product", "brand", "motion", "illustration", "ux", "other"}
    )


def test_recognized_rubric_anchors_are_ordered_low_to_high() -> None:
    """Anchor order matters editorially — low to high."""

    assert RECOGNIZED_RUBRIC_ANCHORS == ("bad", "okay", "good", "excellent")


# ---------------------------------------------------------------------------
# validate_v2_brief — happy paths
# ---------------------------------------------------------------------------


def test_validate_v2_brief_accepts_brief_without_design_rubric() -> None:
    """`design_rubric` is optional. A non-Designer brief should validate cleanly."""

    validate_v2_brief(_minimal_v2_brief())  # No raise.


def test_validate_v2_brief_accepts_brief_with_well_formed_rubric() -> None:
    payload = _minimal_v2_brief()
    payload["design_rubric"] = _well_formed_rubric()
    validate_v2_brief(payload)  # No raise.


def test_validate_v2_brief_accepts_empty_design_rubric() -> None:
    """Recruiter cleared the rubric → empty dict, not invalid."""

    payload = _minimal_v2_brief()
    payload["design_rubric"] = {}
    validate_v2_brief(payload)  # No raise.


def test_default_rubric_validates() -> None:
    """The shipped `config/design-rubrics/default.json` is a valid rubric."""

    rubric = json.loads(_DEFAULT_RUBRIC_PATH.read_text())
    payload = _minimal_v2_brief()
    payload["design_rubric"] = rubric
    validate_v2_brief(payload)  # No raise.

    # Every principle has all four anchor levels and a weight in [0, 5].
    for idx, principle in enumerate(rubric["principles"]):
        for level in RECOGNIZED_RUBRIC_ANCHORS:
            assert level in principle["anchors"], f"principle[{idx}] missing anchor {level}"
        weight = principle.get("weight", 1.0)
        assert 0.0 <= weight <= 5.0


# ---------------------------------------------------------------------------
# validate_v2_brief — malformation surfaces
# ---------------------------------------------------------------------------


def test_validate_v2_brief_rejects_non_dict_design_rubric() -> None:
    payload = _minimal_v2_brief()
    payload["design_rubric"] = "not a dict"
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)
    assert "design_rubric" in excinfo.value.invalid_keys


def test_validate_v2_brief_rejects_principle_missing_anchor() -> None:
    payload = _minimal_v2_brief()
    rubric = _well_formed_rubric()
    rubric["principles"][0]["anchors"].pop("excellent")
    payload["design_rubric"] = rubric
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)
    assert "design_rubric.principles[0].anchors.excellent" in excinfo.value.invalid_keys


def test_validate_v2_brief_rejects_principle_missing_name() -> None:
    payload = _minimal_v2_brief()
    rubric = _well_formed_rubric()
    rubric["principles"][0]["name"] = ""
    payload["design_rubric"] = rubric
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)
    assert "design_rubric.principles[0].name" in excinfo.value.invalid_keys


def test_validate_v2_brief_rejects_weight_out_of_range() -> None:
    payload = _minimal_v2_brief()
    rubric = _well_formed_rubric()
    rubric["principles"][0]["weight"] = 7.5  # > 5.0 ceiling
    payload["design_rubric"] = rubric
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)
    assert "design_rubric.principles[0].weight" in excinfo.value.invalid_keys


def test_validate_v2_brief_rejects_invalid_calibration_exemplar_verdict() -> None:
    payload = _minimal_v2_brief()
    rubric = _well_formed_rubric()
    rubric["calibration_exemplars"][0]["verdict"] = "maybe"  # not in {yes,no,borderline}
    payload["design_rubric"] = rubric
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)
    assert "design_rubric.calibration_exemplars[0].verdict" in excinfo.value.invalid_keys


def test_validate_v2_brief_rejects_non_string_hard_reject_pattern() -> None:
    payload = _minimal_v2_brief()
    rubric = _well_formed_rubric()
    rubric["hard_reject_patterns"] = ["legitimate pattern", 42]  # int is not a string
    payload["design_rubric"] = rubric
    with pytest.raises(BriefSchemaError) as excinfo:
        validate_v2_brief(payload)
    assert "design_rubric.hard_reject_patterns" in excinfo.value.invalid_keys


# ---------------------------------------------------------------------------
# Dataclass shape — hydration target for the brief loader
# ---------------------------------------------------------------------------


def test_brief_design_rubric_constructs_with_defaults() -> None:
    """Minimum-viable BriefDesignRubric instantiation."""

    rubric = BriefDesignRubric()
    assert rubric.principles == ()
    assert rubric.discipline_weight_overrides == {}
    assert rubric.calibration_exemplars == ()
    assert rubric.hard_reject_patterns == ()


def test_rubric_principle_carries_anchor_dict() -> None:
    principle = RubricPrinciple(
        name="Visual hierarchy",
        description="How clearly the work guides attention.",
        anchors={"bad": "no", "okay": "weak", "good": "clear", "excellent": "purposeful"},
        weight=1.3,
    )
    assert principle.anchors["good"] == "clear"
    assert principle.weight == 1.3


def test_calibration_exemplar_carries_per_principle_reasoning() -> None:
    exemplar = CalibrationExemplar(
        portfolio_url="https://example.com/portfolio",
        discipline="product",
        verdict="yes",
        per_principle_reasoning={"Visual hierarchy": "Strong focal points."},
        overall_reasoning="Clearly senior-grade work.",
    )
    assert exemplar.per_principle_reasoning["Visual hierarchy"] == "Strong focal points."
    assert exemplar.verdict == "yes"


# ---------------------------------------------------------------------------
# Legacy merge — design_rubric is V2-bucket, not unknown
# ---------------------------------------------------------------------------


def test_merge_legacy_brief_routes_design_rubric_to_v2_data() -> None:
    """A recruiter-authored design_rubric must not land in `unknown_keys`."""

    payload = _minimal_v2_brief()
    payload["design_rubric"] = _well_formed_rubric()
    merged = merge_legacy_brief(payload)
    assert "design_rubric" in merged.v2_data
    assert "design_rubric" not in merged.unknown_keys
    assert "design_rubric" not in merged.deprecated_keys
