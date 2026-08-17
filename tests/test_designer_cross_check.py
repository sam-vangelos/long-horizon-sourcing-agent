"""Designer Slice 8 — Sonnet 4.6 cross-check on top-decile.

Pins:

- ``select_top_decile_for_cross_check`` ranks by
  confidence × verdict_score (yes > borderline > no), takes the top
  fraction (default 10%) bounded by [min_count, max_count].
- Candidates with non-empty ``fallback_reason`` are excluded
  from cross-check selection (no point cross-checking a fallback
  judgment).
- ``detect_principle_disagreements`` returns disagreements where
  abs(primary.score - cross.score) >= the configured anchor delta
  (default: 2).
- ``run_sonnet_cross_check_pass`` produces one
  :class:`VisionEvaluationResult` per selected candidate, using the
  Claude pricing model for cost estimation.
- ``attach_cross_check_to_judgment`` returns a new judgment with the
  cross-check payload nested in the ``cross_check`` field; cost
  estimates sum.
"""

from __future__ import annotations

from typing import Any

import pytest

from designer.vision_evaluation import (
    CLAUDE_SONNET_4_6_MODEL_NAME,
    CROSS_CHECK_DISAGREEMENT_ANCHOR_DELTA,
    CrossCheckCandidate,
    PrincipleDisagreement,
    VisionEvaluationResult,
    VisualJudgment,
    VisualJudgmentPrinciple,
    _AssetReference,
    attach_cross_check_to_judgment,
    detect_principle_disagreements,
    run_sonnet_cross_check_pass,
    select_top_decile_for_cross_check,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _judgment(
    *,
    name: str = "candidate",
    overall_verdict: str = "yes",
    overall_confidence: float = 0.85,
    fallback_reason: str = "",
    score: int = 3,
    second_score: int | None = None,
    model: str = "gemini-2.5-pro",
) -> VisualJudgment:
    principles = [
        VisualJudgmentPrinciple(
            name="Visual hierarchy",
            score=score,
            anchor=["bad", "okay", "good", "excellent"][score],
            reasoning="x",
            image_ids=(0,),
        ),
    ]
    if second_score is not None:
        principles.append(
            VisualJudgmentPrinciple(
                name="Typographic refinement",
                score=second_score,
                anchor=["bad", "okay", "good", "excellent"][second_score],
                reasoning="y",
                image_ids=(1,),
            )
        )
    return VisualJudgment(
        model=model,
        principles=tuple(principles),
        overall_verdict=overall_verdict,
        overall_confidence=overall_confidence,
        fallback_reason=fallback_reason,
        cost_estimate_usd=0.01,
    )


# ---------------------------------------------------------------------------
# select_top_decile_for_cross_check
# ---------------------------------------------------------------------------


def test_select_ranks_by_confidence_and_verdict() -> None:
    candidates = [
        ("c-no-high", _judgment(overall_verdict="no", overall_confidence=0.95)),
        ("c-yes-high", _judgment(overall_verdict="yes", overall_confidence=0.90)),
        ("c-yes-mid", _judgment(overall_verdict="yes", overall_confidence=0.60)),
        ("c-borderline", _judgment(overall_verdict="borderline", overall_confidence=0.85)),
    ]
    selected = select_top_decile_for_cross_check(
        candidates, fraction=0.5, min_count=1, max_count=2
    )
    keys = [k for k, _ in selected]
    # yes high outranks borderline outranks no high (regardless of conf).
    assert keys[0] == "c-yes-high"


def test_select_excludes_fallback_judgments() -> None:
    candidates = [
        ("c-fallback", _judgment(fallback_reason="image_grounding")),
        ("c-clean", _judgment(overall_verdict="yes", overall_confidence=0.90)),
    ]
    selected = select_top_decile_for_cross_check(
        candidates, fraction=1.0, min_count=1, max_count=10
    )
    assert {k for k, _ in selected} == {"c-clean"}


def test_select_honors_min_count() -> None:
    """Even on a 5-candidate run, min_count guarantees ≥1 cross-check."""

    candidates = [
        (f"c{i}", _judgment(overall_verdict="yes", overall_confidence=0.5 + i * 0.1))
        for i in range(5)
    ]
    selected = select_top_decile_for_cross_check(
        candidates, fraction=0.10, min_count=1, max_count=10
    )
    assert len(selected) == 1


def test_select_honors_max_count() -> None:
    """Even on a 200-candidate run, max_count caps at 10."""

    candidates = [
        (f"c{i}", _judgment(overall_verdict="yes", overall_confidence=0.5))
        for i in range(200)
    ]
    selected = select_top_decile_for_cross_check(
        candidates, fraction=0.10, min_count=1, max_count=10
    )
    assert len(selected) == 10


def test_select_returns_empty_on_no_eligible_candidates() -> None:
    candidates = [
        ("c-fallback", _judgment(fallback_reason="x")),
    ]
    assert select_top_decile_for_cross_check(candidates) == []


# ---------------------------------------------------------------------------
# detect_principle_disagreements
# ---------------------------------------------------------------------------


def test_disagreement_returns_empty_on_byte_equal_scores() -> None:
    primary = _judgment(score=3, second_score=2)
    cross = _judgment(score=3, second_score=2)
    assert detect_principle_disagreements(primary=primary, cross_check=cross) == []


def test_disagreement_returns_empty_on_off_by_one() -> None:
    """Off-by-one is acceptable per spec §4.3."""

    primary = _judgment(score=3, second_score=2)
    cross = _judgment(score=2, second_score=1)  # off-by-one on both
    assert detect_principle_disagreements(primary=primary, cross_check=cross) == []


def test_disagreement_fires_on_anchor_delta_two() -> None:
    """Score delta of 2 = anchor delta of 2 → disagreement."""

    primary = _judgment(score=3)  # excellent
    cross = _judgment(score=1)  # okay
    disagreements = detect_principle_disagreements(primary=primary, cross_check=cross)
    assert len(disagreements) == 1
    d = disagreements[0]
    assert d.principle_name == "Visual hierarchy"
    assert d.primary_score == 3
    assert d.cross_check_score == 1
    assert d.delta == 2


def test_disagreement_threshold_default_matches_spec() -> None:
    assert CROSS_CHECK_DISAGREEMENT_ANCHOR_DELTA == 2


def test_disagreement_skips_principles_only_in_cross_check() -> None:
    """A principle only in cross-check (no primary counterpart) is
    skipped — the score-comparison is undefined."""

    primary = _judgment(score=3)  # only Visual hierarchy
    cross = _judgment(score=3, second_score=0)  # adds Typographic refinement
    disagreements = detect_principle_disagreements(primary=primary, cross_check=cross)
    # Visual hierarchy: equal scores → no disagreement.
    # Typographic refinement: only in cross — skipped.
    assert disagreements == []


# ---------------------------------------------------------------------------
# run_sonnet_cross_check_pass
# ---------------------------------------------------------------------------


def test_cross_check_pass_uses_claude_model_name() -> None:
    """The Sonnet pass labels the judgment with Claude's model name
    so the workspace surface can render which model produced the
    cross-check verdict."""

    asset_refs = (
        _AssetReference(image_id=0, asset_url="x", source="behance", project_title="t"),
    )
    selected = [
        CrossCheckCandidate(
            candidate_identity_key="c1",
            primary_judgment=_judgment(),
            asset_references=asset_refs,
        )
    ]

    def _fake_llm(model, system, user, images):
        # Return a well-formed Sonnet response with score=2.
        return {
            "principles": [
                {
                    "name": "Visual hierarchy",
                    "score": 2,
                    "reasoning": "Cross-check finds clear hierarchy across image_id 0.",
                    "image_ids": [0],
                }
            ],
            "overall_verdict": "yes",
            "overall_confidence": 0.7,
            "overall_reasoning": "Cross-check agrees overall.",
        }

    brief = {
        "design_rubric": {
            "principles": [
                {
                    "name": "Visual hierarchy",
                    "description": "x",
                    "anchors": {"bad": "n", "okay": "o", "good": "g", "excellent": "e"},
                }
            ]
        }
    }

    out = run_sonnet_cross_check_pass(
        brief=brief,
        selected_candidates=selected,
        image_bytes_lookup={"c1": [b"x"]},
        vision_llm_call=_fake_llm,
    )
    assert "c1" in out
    judgment = out["c1"].judgment
    assert judgment.model == CLAUDE_SONNET_4_6_MODEL_NAME
    # Cost estimate uses Claude pricing (~10x Gemini per spec §4.1).
    assert judgment.cost_estimate_usd > 0


# ---------------------------------------------------------------------------
# attach_cross_check_to_judgment
# ---------------------------------------------------------------------------


def test_attach_cross_check_nests_payload_and_sums_cost() -> None:
    primary = _judgment(score=3)
    cross = _judgment(score=2, model=CLAUDE_SONNET_4_6_MODEL_NAME)
    primary_with_cost = VisualJudgment(
        model=primary.model,
        principles=primary.principles,
        overall_verdict=primary.overall_verdict,
        overall_confidence=primary.overall_confidence,
        cost_estimate_usd=0.01,
    )
    cross_with_cost = VisualJudgment(
        model=cross.model,
        principles=cross.principles,
        overall_verdict=cross.overall_verdict,
        overall_confidence=cross.overall_confidence,
        cost_estimate_usd=0.10,
    )
    merged = attach_cross_check_to_judgment(
        primary=primary_with_cost, cross_check=cross_with_cost
    )
    assert merged.model == "gemini-2.5-pro"  # primary's model
    assert merged.cross_check is not None
    assert merged.cross_check["model"] == CLAUDE_SONNET_4_6_MODEL_NAME
    assert merged.cross_check["principles"][0]["score"] == 2
    # Cost estimates sum.
    assert merged.cost_estimate_usd == pytest.approx(0.11)


def test_attach_cross_check_payload_is_json_round_trippable() -> None:
    """The cross_check field lands in `terminal_payload_json` so it
    must round-trip through json.dumps/loads."""

    import json

    primary = _judgment(score=3)
    cross = _judgment(score=1, model=CLAUDE_SONNET_4_6_MODEL_NAME)
    merged = attach_cross_check_to_judgment(primary=primary, cross_check=cross)
    payload = {"cross_check": merged.cross_check}
    rountripped = json.loads(json.dumps(payload))
    assert rountripped["cross_check"]["model"] == CLAUDE_SONNET_4_6_MODEL_NAME
