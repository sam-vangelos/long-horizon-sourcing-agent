"""Designer Slice 4 — `_design_rubric_drift` cascade entry.

Pins the preservation contract for `design_rubric` across brief polish:

- A seeded draft without a rubric → no drift signal (nothing to preserve).
- A seeded rubric byte-equal to the polished output → no drift.
- The polished output dropping the rubric → drift descriptor "dropped".
- The polished output mutating any sub-field (principles, anchors,
  weights, exemplars, hard_reject_patterns, discipline_weight_overrides)
  → drift descriptor identifying the mutation class.
- The full LLM-cascade behavior: when an LLM polish mutates the rubric,
  `_cascade_done(seeded, t0)` returns the heuristic-seeded draft instead
  (via the named cascade entry). Test the cascade integration via
  injecting a fake LLM that returns a mutated rubric.
- Heuristic backend hydrates `design_rubric` from chapter_captures
  pass-through.
- System prompt text carries the new preservation rule clause.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from market_intelligence.brief_polish import (
    BriefPolishBackend,
    HeuristicBriefPolishBackend,
    _describe_rubric_drift,
    _design_rubric_drift,
    build_brief_polish_system_prompt,
    build_brief_polish_user_prompt,
)


def _well_formed_rubric() -> dict:
    return {
        "principles": [
            {
                "name": "Visual hierarchy",
                "description": "How clearly the work guides attention.",
                "anchors": {
                    "bad": "no",
                    "okay": "weak",
                    "good": "clear",
                    "excellent": "purposeful",
                },
                "weight": 1.0,
            }
        ],
        "discipline_weight_overrides": {
            "product": {"Visual hierarchy": 1.3},
        },
        "calibration_exemplars": [
            {
                "portfolio_url": "https://example.com",
                "discipline": "product",
                "verdict": "yes",
                "per_principle_reasoning": {"Visual hierarchy": "strong"},
                "overall_reasoning": "good fit",
            }
        ],
        "hard_reject_patterns": ["layout-only portfolios"],
    }


# ---------------------------------------------------------------------------
# _design_rubric_drift — pure helper
# ---------------------------------------------------------------------------


def test_drift_returns_none_when_seed_has_no_rubric() -> None:
    seeded = {"role_title": "x"}
    polished = {"role_title": "x"}
    assert _design_rubric_drift(seeded=seeded, polished=polished) is None


def test_drift_returns_none_when_seed_rubric_is_empty_dict() -> None:
    """Recruiter cleared the rubric → no preservation contract to enforce."""

    seeded = {"design_rubric": {}}
    polished = {"design_rubric": {}}
    assert _design_rubric_drift(seeded=seeded, polished=polished) is None


def test_drift_returns_none_for_byte_equal_rubric() -> None:
    rubric = _well_formed_rubric()
    seeded = {"design_rubric": rubric}
    polished = {"design_rubric": dict(rubric)}  # same content
    assert _design_rubric_drift(seeded=seeded, polished=polished) is None


def test_drift_returns_dropped_when_polished_omits_rubric() -> None:
    seeded = {"design_rubric": _well_formed_rubric()}
    polished = {"role_title": "x"}  # no design_rubric
    assert _design_rubric_drift(seeded=seeded, polished=polished) == "dropped"


def test_drift_detects_principle_count_change() -> None:
    seeded = {"design_rubric": _well_formed_rubric()}
    polished_rubric = _well_formed_rubric()
    polished_rubric["principles"].append(
        {
            "name": "Conceptual strength",
            "description": "Idea-level depth.",
            "anchors": {"bad": "n", "okay": "o", "good": "g", "excellent": "e"},
            "weight": 1.0,
        }
    )
    descriptor = _design_rubric_drift(
        seeded=seeded, polished={"design_rubric": polished_rubric}
    )
    assert descriptor is not None
    assert "principle_count_changed" in descriptor


def test_drift_detects_principle_mutation() -> None:
    seeded = {"design_rubric": _well_formed_rubric()}
    polished_rubric = _well_formed_rubric()
    polished_rubric["principles"][0]["name"] = "VISUAL HIERARCHY (improved)"
    descriptor = _design_rubric_drift(
        seeded=seeded, polished={"design_rubric": polished_rubric}
    )
    assert descriptor is not None
    assert "principle[0]_mutated" in descriptor


def test_drift_detects_discipline_weight_overrides_mutation() -> None:
    seeded = {"design_rubric": _well_formed_rubric()}
    polished_rubric = _well_formed_rubric()
    polished_rubric["discipline_weight_overrides"]["product"]["Visual hierarchy"] = 99.0
    descriptor = _design_rubric_drift(
        seeded=seeded, polished={"design_rubric": polished_rubric}
    )
    assert descriptor is not None
    assert "discipline_weight_overrides_mutated" in descriptor


def test_drift_detects_calibration_exemplars_mutation() -> None:
    seeded = {"design_rubric": _well_formed_rubric()}
    polished_rubric = _well_formed_rubric()
    polished_rubric["calibration_exemplars"][0]["verdict"] = "no"
    descriptor = _design_rubric_drift(
        seeded=seeded, polished={"design_rubric": polished_rubric}
    )
    assert descriptor is not None
    assert "calibration_exemplars_mutated" in descriptor


def test_drift_detects_hard_reject_patterns_mutation() -> None:
    seeded = {"design_rubric": _well_formed_rubric()}
    polished_rubric = _well_formed_rubric()
    polished_rubric["hard_reject_patterns"].append("invented pattern")
    descriptor = _design_rubric_drift(
        seeded=seeded, polished={"design_rubric": polished_rubric}
    )
    assert descriptor is not None
    assert "hard_reject_patterns_mutated" in descriptor


def test_describe_rubric_drift_returns_short_string_for_log_line() -> None:
    descriptor = _describe_rubric_drift(
        seed=_well_formed_rubric(), polish={"principles": []}
    )
    # Bounded length — not a full diff dump, just a class-of-mutation tag.
    assert len(descriptor) < 200


# ---------------------------------------------------------------------------
# Heuristic backend hydration
# ---------------------------------------------------------------------------


def test_heuristic_hydrates_design_rubric_from_chapter_capture() -> None:
    backend = HeuristicBriefPolishBackend()
    rubric = _well_formed_rubric()
    chapter_captures = {
        "role": {"title": "Senior product designer"},
        "good_looks": {"prose": "Ships product surface end-to-end."},
        "where_to_look": {"target_modules": ["designer"]},
        "design_rubric": rubric,
    }
    result = backend.polish(chapter_captures=chapter_captures)
    assert result.v2_draft.get("design_rubric") == rubric


def test_heuristic_omits_design_rubric_when_chapter_capture_empty() -> None:
    backend = HeuristicBriefPolishBackend()
    chapter_captures = {
        "role": {"title": "Designer"},
        "good_looks": {"prose": "Ships product."},
    }
    result = backend.polish(chapter_captures=chapter_captures)
    # Absent key, not empty dict — `_design_rubric_drift` treats the
    # two equivalently but the cleaner shape on disk is "no key."
    assert "design_rubric" not in result.v2_draft


# ---------------------------------------------------------------------------
# System prompt carries the preservation clause
# ---------------------------------------------------------------------------


def test_system_prompt_carries_design_rubric_preservation_rule() -> None:
    prompt = build_brief_polish_system_prompt()
    assert "design_rubric" in prompt
    assert "byte-for-byte" in prompt
    assert "principles, anchors, weights" in prompt


def test_system_prompt_schema_response_shape_includes_design_rubric() -> None:
    prompt = build_brief_polish_system_prompt()
    # The response schema in the prompt teaches the LLM what shape to
    # return; design_rubric must be in the documented shape so the LLM
    # passes it through rather than silently dropping it.
    assert '"design_rubric"' in prompt


def test_user_prompt_threads_design_rubric_capture_into_input() -> None:
    rubric = _well_formed_rubric()
    chapter_captures = {
        "role": {"title": "Designer"},
        "good_looks": {"prose": "x"},
        "design_rubric": rubric,
    }
    seeded_v2_draft = {"role_title": "Designer", "design_rubric": rubric}
    user_prompt = build_brief_polish_user_prompt(
        chapter_captures=chapter_captures, seeded_v2_draft=seeded_v2_draft
    )
    # Both the chapter-capture payload and the seeded_v2_draft carry
    # the rubric so the LLM has the byte-equality target in front of it.
    assert "design_rubric" in user_prompt
    assert "Visual hierarchy" in user_prompt


# ---------------------------------------------------------------------------
# Full cascade integration — LLM mutates rubric → fallback to heuristic
# ---------------------------------------------------------------------------


@pytest.fixture
def chapter_captures_with_rubric() -> dict[str, Any]:
    return {
        "role": {"title": "Senior product designer"},
        "good_looks": {
            "prose": (
                "Ships shipped product end-to-end. Owns design system at the "
                "team level. Strong at typography and information density."
            )
        },
        "lookalikes": {"exemplars_prose": "Designers like X and Y.", "non_fit_prose": ""},
        "where_to_look": {"target_modules": ["designer"]},
        "design_rubric": _well_formed_rubric(),
    }


def test_cascade_falls_back_when_llm_mutates_rubric(
    chapter_captures_with_rubric: dict[str, Any],
) -> None:
    """End-to-end: an LLM polish that returns a mutated rubric should
    cascade through `_design_rubric_drift` to the heuristic seed."""

    fake_llm_output = {
        "role_title": "Senior product designer",
        "capability_areas": [
            {"name": "Capability area 1", "description": "Ships shipped product end-to-end. Owns design system at the team level."}
        ],
        "depth_distinction": {
            "builder_definition": "",
            "user_definition": "",
            "edge_case_guidance": "",
        },
        "non_fit_patterns": [],
        "target_modules": ["designer"],
        # Mutated rubric — added a principle the recruiter didn't write.
        "design_rubric": {
            **_well_formed_rubric(),
            "principles": _well_formed_rubric()["principles"]
            + [
                {
                    "name": "LLM-invented principle",
                    "description": "added by the model",
                    "anchors": {"bad": "n", "okay": "o", "good": "g", "excellent": "e"},
                    "weight": 1.0,
                }
            ],
        },
    }

    with patch("market_intelligence.brief_polish.opus_llm", return_value=fake_llm_output), \
         patch("market_intelligence.brief_polish._has_llm_access", return_value=True):
        backend = BriefPolishBackend()
        result = backend.polish(
            chapter_captures=chapter_captures_with_rubric, role_title="Senior product designer"
        )

    # Cascade fired → result.source is the heuristic ("deterministic"),
    # not "llm". The heuristic-seeded draft preserves the recruiter's
    # original rubric byte-for-byte.
    assert result.source == "deterministic"
    assert result.v2_draft.get("design_rubric") == _well_formed_rubric()


def test_cascade_succeeds_when_llm_preserves_rubric(
    chapter_captures_with_rubric: dict[str, Any],
) -> None:
    """Conversely: an LLM polish that preserves the rubric byte-equally
    sails through the cascade to a successful llm-source result."""

    rubric = _well_formed_rubric()
    fake_llm_output = {
        "role_title": "Senior product designer",
        "capability_areas": [
            {
                "name": "Product surface",
                "description": (
                    "Ships shipped product end-to-end. Owns design system at "
                    "the team level."
                ),
            }
        ],
        "depth_distinction": {
            "builder_definition": "",
            "user_definition": "",
            "edge_case_guidance": "",
        },
        "non_fit_patterns": [],
        "target_modules": ["designer"],
        "design_rubric": rubric,  # byte-equal preservation
    }

    with patch("market_intelligence.brief_polish.opus_llm", return_value=fake_llm_output), \
         patch("market_intelligence.brief_polish._has_llm_access", return_value=True):
        backend = BriefPolishBackend()
        result = backend.polish(
            chapter_captures=chapter_captures_with_rubric, role_title="Senior product designer"
        )

    assert result.source == "llm"
    assert result.v2_draft.get("design_rubric") == rubric
