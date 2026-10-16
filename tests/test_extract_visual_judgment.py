"""Designer Slice 6 — read-model helpers for the HITL visual review surface.

Pins:

- ``extract_surface_type`` returns the recognized string when present,
  None for legacy / missing payloads.
- ``extract_visual_judgment`` returns the structured payload when
  ``visual_judgment`` is a non-empty dict; None otherwise.
- The helpers are pure functions (no side effects).
- Backward-compat with the existing
  ``extract_save_reason_and_confidence`` contract: the visual-judgment
  helper does not interfere with the save_reason / confidence path.

These helpers are the wire contract Slice-6's
``CandidateDetail.svelte`` surface_type dispatch grounds itself in.
"""

from __future__ import annotations

from shared.runtime_state.read_models import (
    extract_save_reason_and_confidence,
    extract_surface_type,
    extract_visual_judgment,
)


# ---------------------------------------------------------------------------
# extract_surface_type
# ---------------------------------------------------------------------------


def test_extract_surface_type_returns_none_for_none_payload() -> None:
    assert extract_surface_type(None) is None


def test_extract_surface_type_returns_none_for_payload_without_field() -> None:
    payload = {
        "full_decision": {"rationale": "x", "confidence": 0.5},
    }
    assert extract_surface_type(payload) is None


def test_extract_surface_type_returns_recognized_value() -> None:
    payload = {"surface_type": "hitl_visual_review"}
    assert extract_surface_type(payload) == "hitl_visual_review"


def test_extract_surface_type_strips_whitespace() -> None:
    payload = {"surface_type": "  hitl_visual_review  "}
    assert extract_surface_type(payload) == "hitl_visual_review"


def test_extract_surface_type_returns_none_for_empty_string() -> None:
    payload = {"surface_type": ""}
    assert extract_surface_type(payload) is None


def test_extract_surface_type_returns_none_for_non_string_value() -> None:
    payload = {"surface_type": 42}
    assert extract_surface_type(payload) is None


# ---------------------------------------------------------------------------
# extract_visual_judgment
# ---------------------------------------------------------------------------


def test_extract_visual_judgment_returns_none_for_none_payload() -> None:
    assert extract_visual_judgment(None) is None


def test_extract_visual_judgment_returns_none_when_field_missing() -> None:
    payload = {"surface_type": "hitl_visual_review"}
    assert extract_visual_judgment(payload) is None


def test_extract_visual_judgment_returns_none_for_empty_dict() -> None:
    payload = {"visual_judgment": {}}
    assert extract_visual_judgment(payload) is None


def test_extract_visual_judgment_returns_dict_when_present() -> None:
    visual_judgment = {
        "model": "gemini-2.5-pro",
        "principles": [
            {
                "name": "Visual hierarchy",
                "score": 3,
                "anchor": "excellent",
                "reasoning": "Strong primary focal points across image_id 0 and 2.",
                "image_ids": [0, 2],
            }
        ],
        "overall_verdict": "yes",
        "overall_confidence": 0.85,
        "assets": [
            {
                "id": 0,
                "url": "https://example.com/a.jpg",
                "source": "behance",
                "project_title": "Acme",
            }
        ],
    }
    payload = {
        "surface_type": "hitl_visual_review",
        "visual_judgment": visual_judgment,
    }
    result = extract_visual_judgment(payload)
    assert result is not None
    assert result["model"] == "gemini-2.5-pro"
    assert result["overall_verdict"] == "yes"


def test_extract_visual_judgment_returns_none_for_non_dict_value() -> None:
    payload = {"visual_judgment": "not a dict"}
    assert extract_visual_judgment(payload) is None


# ---------------------------------------------------------------------------
# Co-existence with extract_save_reason_and_confidence
# ---------------------------------------------------------------------------


def test_extract_helpers_coexist_on_designer_payload() -> None:
    """A Designer-saved candidate carries BOTH the legacy
    full_decision.rationale + confidence (so the wire contract
    every other module reads stays uniform) AND the new
    visual_judgment payload."""

    payload = {
        "surface_type": "hitl_visual_review",
        "full_decision": {
            "rationale": "Strong product surface design with clear typography.",
            "confidence": 0.85,
        },
        "visual_judgment": {
            "model": "gemini-2.5-pro",
            "principles": [],
            "overall_verdict": "yes",
            "overall_confidence": 0.85,
            "assets": [],
        },
    }
    save_reason, confidence = extract_save_reason_and_confidence(payload)
    assert save_reason == "Strong product surface design with clear typography."
    assert confidence == 0.85
    assert extract_surface_type(payload) == "hitl_visual_review"
    visual_judgment = extract_visual_judgment(payload)
    assert visual_judgment is not None
    assert visual_judgment["overall_verdict"] == "yes"
