"""Unit tests for designer/recommendation_pitch.py (D5b)."""

from designer.recommendation_pitch import assemble_recommendation_pitch


def _visual_judgment(
    *,
    principles: list[dict] | None = None,
    overall_verdict: str = "yes",
    overall_confidence: float = 0.85,
    fallback_reason: str = "",
) -> dict:
    if principles is None:
        principles = [
            {
                "name": "Visual hierarchy",
                "score": 3,
                "anchor": "excellent",
                "reasoning": "Strong hierarchy.",
                "image_ids": [0, 1],
                "anchor_consistency_pass": True,
            },
            {
                "name": "Typographic refinement",
                "score": 2,
                "anchor": "good",
                "reasoning": "Good type craft.",
                "image_ids": [1],
                "anchor_consistency_pass": True,
            },
            {
                "name": "Compositional balance",
                "score": 2,
                "anchor": "good",
                "reasoning": "Well-composed.",
                "image_ids": [0],
                "anchor_consistency_pass": True,
            },
            {
                "name": "Craft execution",
                "score": 1,
                "anchor": "okay",
                "reasoning": "Acceptable craft.",
                "image_ids": [],
                "anchor_consistency_pass": True,
            },
        ]
    return {
        "model": "gemini-3.1-pro-preview",
        "principles": principles,
        "overall_verdict": overall_verdict,
        "overall_confidence": overall_confidence,
        "fallback_reason": fallback_reason,
    }


def test_save_with_strong_principles_headline_references_role() -> None:
    payload = {
        "full_decision": {
            "decision": "SAVE",
            "rationale": "Strong portfolio showing design systems mastery.",
            "confidence": 0.88,
        },
        "visual_judgment": _visual_judgment(),
    }
    pitch = assemble_recommendation_pitch(payload, role_title="Senior Designer")
    assert pitch is not None
    assert "Strong fit for Senior Designer" in pitch["headline"]
    assert len(pitch["evidence_bullets"]) == 3  # 3 principles with score >= 2
    assert len(pitch["caveats"]) == 1  # 1 principle with score 1
    assert pitch["summary"] == "Strong portfolio showing design systems mastery."


def test_inferential_save_with_fallback_headline() -> None:
    payload = {
        "full_decision": {
            "decision": "INFERENTIAL_SAVE",
            "rationale": "Text signal is promising but images were unavailable.",
            "confidence": 0.6,
        },
        "visual_judgment": _visual_judgment(
            principles=[],
            overall_verdict="borderline",
            fallback_reason="no_images_acquired",
        ),
    }
    pitch = assemble_recommendation_pitch(payload, role_title="Product Designer")
    assert pitch is not None
    assert "visual evidence was limited" in pitch["headline"]
    assert len(pitch["evidence_bullets"]) == 0
    assert any("no_images_acquired" in c for c in pitch["caveats"])


def test_reject_returns_none() -> None:
    payload = {
        "full_decision": {"decision": "REJECT", "rationale": "No fit."},
        "visual_judgment": _visual_judgment(overall_verdict="no"),
    }
    pitch = assemble_recommendation_pitch(payload, role_title="Designer")
    assert pitch is None


def test_empty_visual_judgment_still_produces_pitch() -> None:
    payload = {
        "full_decision": {
            "decision": "SAVE",
            "rationale": "Text-only save.",
            "confidence": 0.7,
        },
        "visual_judgment": {
            "model": "test",
            "principles": [],
            "overall_verdict": "yes",
            "overall_confidence": 0.7,
            "fallback_reason": "",
        },
    }
    pitch = assemble_recommendation_pitch(payload)
    assert pitch is not None
    assert len(pitch["evidence_bullets"]) == 0
    assert "Visual review supports a conversation" in pitch["headline"]


def test_one_strong_principle_headline() -> None:
    payload = {
        "full_decision": {
            "decision": "SAVE",
            "rationale": "One strong principle.",
            "confidence": 0.75,
        },
        "visual_judgment": _visual_judgment(
            principles=[
                {
                    "name": "Conceptual strength",
                    "score": 3,
                    "anchor": "excellent",
                    "reasoning": "Original concept.",
                    "image_ids": [0],
                    "anchor_consistency_pass": True,
                },
                {
                    "name": "Craft execution",
                    "score": 1,
                    "anchor": "okay",
                    "reasoning": "Basic craft.",
                    "image_ids": [],
                    "anchor_consistency_pass": True,
                },
            ],
        ),
    }
    pitch = assemble_recommendation_pitch(payload, role_title="Designer")
    assert pitch is not None
    assert "Conceptual strength stands out" in pitch["headline"]


def test_anchor_drift_adds_caveat() -> None:
    payload = {
        "full_decision": {
            "decision": "SAVE",
            "rationale": "Anchor drift on one principle.",
            "confidence": 0.8,
        },
        "visual_judgment": _visual_judgment(
            principles=[
                {
                    "name": "Visual hierarchy",
                    "score": 3,
                    "anchor": "excellent",
                    "reasoning": "Strong.",
                    "image_ids": [0],
                    "anchor_consistency_pass": False,
                },
                {
                    "name": "Typographic refinement",
                    "score": 2,
                    "anchor": "good",
                    "reasoning": "Good.",
                    "image_ids": [1],
                    "anchor_consistency_pass": True,
                },
                {
                    "name": "Compositional balance",
                    "score": 2,
                    "anchor": "good",
                    "reasoning": "Fine.",
                    "image_ids": [],
                    "anchor_consistency_pass": True,
                },
            ],
        ),
    }
    pitch = assemble_recommendation_pitch(payload, role_title="Designer")
    assert pitch is not None
    assert any("anchor drift" in c for c in pitch["caveats"])
