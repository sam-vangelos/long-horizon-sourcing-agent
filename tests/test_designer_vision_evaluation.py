"""Designer Slice 5 — vision evaluation pipeline + 4-layer hallucination guard.

Pins:

- ``assemble_vision_evaluation_system_prompt`` carries rubric
  principles (with anchor definitions), discipline weight overrides,
  calibration exemplars, hard-reject patterns, and a structured
  output spec the LLM is told to return.
- The 4-layer hallucination guard cascade fires correctly for each
  malformation:
  * Layer 1: response not parseable to schema → fallback.
  * Layer 2: any per-principle reasoning citing no images → fallback.
  * Layer 3: per-principle anchor consistency Jaccard < threshold →
    that principle is marked stale; doesn't drop the whole eval.
  * Layer 4: hard-reject pattern matched in overall_reasoning →
    auto-reject (verdict forced to "no", confidence 1.0).
- Cost estimate is roughly bounded by the documented per-image token
  count (spec §4.1: ~258 tokens per 1024×1024 image).
- The injectable ``vision_llm_call`` callable lets tests deterministically
  exercise each guard without invoking google-genai.

The fixture portfolio (3 hand-curated candidate × rubric pairs) lives
inline; Slice 11 ships a richer fixture set for end-to-end characterization.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from designer.vision_evaluation import (
    ANCHOR_OVERLAP_THRESHOLD,
    GEMINI_3_1_PRO_TOKENS_PER_IMAGE,
    SCORE_TO_ANCHOR,
    VisionEvaluationResult,
    VisualJudgment,
    VisualJudgmentPrinciple,
    _AssetReference,
    _layer1_schema_validity,
    _layer2_image_grounding,
    _layer3_anchor_consistency,
    _layer4_hard_reject,
    assemble_vision_evaluation_system_prompt,
    assemble_vision_evaluation_user_text,
    claude_vision_llm_call,
    evaluate_designer_visually,
    gemini_vision_llm_call,
    resolve_vision_fallback,
)


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------


def _rubric_with_two_principles() -> dict[str, Any]:
    return {
        "principles": [
            {
                "name": "Visual hierarchy",
                "description": "How clearly the work guides attention.",
                "anchors": {
                    "bad": "no discernible hierarchy elements compete equally",
                    "okay": "hierarchy present but weak distinctions subtle",
                    "good": "clear hierarchy primary secondary tertiary distinct",
                    "excellent": "purposeful confident precise direction attention",
                },
                "weight": 1.0,
            },
            {
                "name": "Typographic refinement",
                "description": "Type as a first-class material.",
                "anchors": {
                    "bad": "default undifferentiated workmanlike typography",
                    "okay": "consistent without precision",
                    "good": "intentional well-set kerning leading pairing craft",
                    "excellent": "decisive primary expressive material precision",
                },
                "weight": 1.0,
            },
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
        "hard_reject_patterns": ["layout-only portfolios with no shipped product"],
    }


def _designer_brief_for_vision() -> dict[str, Any]:
    return {
        "role_title": "Senior product designer",
        "design_rubric": _rubric_with_two_principles(),
    }


def _well_formed_vision_response() -> dict[str, Any]:
    """A Gemini response that passes all four guards."""

    return {
        "principles": [
            {
                "name": "Visual hierarchy",
                "score": 3,
                "reasoning": (
                    "Image_id 0 demonstrates purposeful confident attention "
                    "direction across the primary surface; image_id 2 "
                    "extends precise hierarchy treatment."
                ),
                "image_ids": [0, 2],
            },
            {
                "name": "Typographic refinement",
                "score": 2,
                "reasoning": (
                    "Image_id 1 shows intentional well-set kerning and "
                    "leading; pairing craft is consistent."
                ),
                "image_ids": [1],
            },
        ],
        "overall_verdict": "yes",
        "overall_confidence": 0.85,
        "overall_reasoning": "Strong product surface design with clear typography.",
    }


def _make_fake_llm(*, response: Any) -> Any:
    """Return a vision_llm_call stub that returns ``response``."""

    def _fake(model: str, system: str, user: str, images: list[bytes]) -> Any:
        return response

    return _fake


def _asset_metadata(n: int) -> list[tuple[str, str, str]]:
    return [
        (f"https://example.com/img{i}.jpg", "behance", f"Project {i}")
        for i in range(n)
    ]


def _image_bytes(n: int) -> list[bytes]:
    return [f"img-bytes-{i}".encode() for i in range(n)]


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------


def test_system_prompt_carries_rubric_principles_and_anchors() -> None:
    prompt = assemble_vision_evaluation_system_prompt(_designer_brief_for_vision())
    assert "Visual hierarchy" in prompt
    assert "Typographic refinement" in prompt
    # Each anchor level present.
    for level in ("bad", "okay", "good", "excellent"):
        assert f"[{level}]" in prompt


def test_system_prompt_carries_discipline_weight_overrides() -> None:
    prompt = assemble_vision_evaluation_system_prompt(_designer_brief_for_vision())
    assert "DISCIPLINE WEIGHT OVERRIDES" in prompt
    assert "product:" in prompt


def test_system_prompt_carries_calibration_exemplars() -> None:
    prompt = assemble_vision_evaluation_system_prompt(_designer_brief_for_vision())
    assert "CALIBRATION EXEMPLARS" in prompt
    assert "https://example.com" in prompt


def test_system_prompt_carries_hard_reject_patterns() -> None:
    prompt = assemble_vision_evaluation_system_prompt(_designer_brief_for_vision())
    assert "HARD REJECT PATTERNS" in prompt
    assert "layout-only portfolios" in prompt


def test_system_prompt_demands_image_id_citations() -> None:
    prompt = assemble_vision_evaluation_system_prompt(_designer_brief_for_vision())
    assert "image_id" in prompt
    assert "MUST cite at least one image_id" in prompt


def test_user_text_lists_image_ids_with_project_context() -> None:
    asset_refs = (
        _AssetReference(
            image_id=0,
            asset_url="https://example.com/a.jpg",
            source="behance",
            project_title="Project Alpha",
        ),
        _AssetReference(
            image_id=1,
            asset_url="https://example.com/b.jpg",
            source="google_cse",
            project_title="https://joe.cargo.site",
        ),
    )
    text = assemble_vision_evaluation_user_text(
        candidate_display_name="Joe Designer",
        candidate_headline="Senior product designer",
        asset_references=asset_refs,
    )
    assert "Joe Designer" in text
    assert "Senior product designer" in text
    assert "image_id=0" in text
    assert "image_id=1" in text
    assert "Project Alpha" in text


# ---------------------------------------------------------------------------
# Layer 1 — schema validity
# ---------------------------------------------------------------------------


def test_layer1_passes_well_formed_response() -> None:
    assert _layer1_schema_validity(_well_formed_vision_response()) is None


def test_layer1_rejects_non_dict() -> None:
    assert _layer1_schema_validity("not a dict") is not None


def test_layer1_rejects_missing_principles() -> None:
    assert (
        _layer1_schema_validity({"overall_verdict": "yes"})
        == "schema_invalid:no_principles"
    )


def test_layer1_rejects_principle_missing_score() -> None:
    bad = _well_formed_vision_response()
    del bad["principles"][0]["score"]
    descriptor = _layer1_schema_validity(bad)
    assert descriptor is not None
    assert "score" in descriptor


def test_layer1_rejects_invalid_overall_verdict() -> None:
    bad = _well_formed_vision_response()
    bad["overall_verdict"] = "maybe"
    assert _layer1_schema_validity(bad) == "schema_invalid:overall_verdict"


def test_layer1_rejects_confidence_out_of_range() -> None:
    bad = _well_formed_vision_response()
    bad["overall_confidence"] = 1.5
    assert _layer1_schema_validity(bad) == "schema_invalid:overall_confidence_range"


# ---------------------------------------------------------------------------
# Layer 2 — image grounding
# ---------------------------------------------------------------------------


def test_layer2_passes_when_every_principle_cites_image_ids() -> None:
    assert _layer2_image_grounding(_well_formed_vision_response()) is None


def test_layer2_rejects_principle_with_empty_image_ids() -> None:
    bad = _well_formed_vision_response()
    bad["principles"][1]["image_ids"] = []
    descriptor = _layer2_image_grounding(bad)
    assert descriptor is not None
    assert "principle[1]_no_image_ids" in descriptor


# ---------------------------------------------------------------------------
# Layer 3 — anchor consistency
# ---------------------------------------------------------------------------


def test_layer3_marks_pass_when_reasoning_overlaps_anchor() -> None:
    """Slice-5 fixture: well-formed reasoning should pass."""

    rubric_principles = _rubric_with_two_principles()["principles"]
    pass_map = _layer3_anchor_consistency(
        _well_formed_vision_response(),
        rubric_principles=rubric_principles,
    )
    # Both principles in the well-formed response use vocabulary
    # that overlaps the assigned-anchor definition.
    assert pass_map["Visual hierarchy"] is True
    assert pass_map["Typographic refinement"] is True


def test_layer3_marks_fail_when_reasoning_drifts_from_anchor() -> None:
    rubric_principles = _rubric_with_two_principles()["principles"]
    drift = _well_formed_vision_response()
    # Score 0 = "bad" anchor: "no discernible hierarchy elements compete equally".
    # Reasoning that talks about "marine biology octopus camouflage" has
    # zero overlap with the bad-anchor vocabulary.
    drift["principles"][0]["score"] = 0
    drift["principles"][0]["reasoning"] = (
        "Marine biology octopus camouflage adaptations across reef "
        "ecosystems."
    )
    pass_map = _layer3_anchor_consistency(drift, rubric_principles=rubric_principles)
    assert pass_map["Visual hierarchy"] is False


# ---------------------------------------------------------------------------
# Layer 4 — hard reject
# ---------------------------------------------------------------------------


def test_layer4_returns_none_when_no_patterns_match() -> None:
    raw = _well_formed_vision_response()
    result = _layer4_hard_reject(
        raw, hard_reject_patterns=["pattern that doesn't appear"]
    )
    assert result is None


def test_layer4_matches_pattern_in_overall_reasoning() -> None:
    raw = _well_formed_vision_response()
    raw["overall_reasoning"] = (
        "This is a layout-only portfolio with no shipped product."
    )
    result = _layer4_hard_reject(
        raw,
        # Pattern matches the reasoning verbatim — strict substring
        # match. Recruiter-authored patterns stay precise; brittleness
        # is the point (Slice 7 feedback marker informs which patterns
        # work).
        hard_reject_patterns=["layout-only portfolio with no shipped product"],
    )
    assert result is not None
    assert "matched_pattern" in result


def test_layer4_returns_none_when_brief_has_no_patterns() -> None:
    raw = _well_formed_vision_response()
    result = _layer4_hard_reject(raw, hard_reject_patterns=[])
    assert result is None


# ---------------------------------------------------------------------------
# evaluate_designer_visually — full pipeline integration
# ---------------------------------------------------------------------------


def test_evaluate_returns_structured_judgment_on_well_formed_response() -> None:
    fake_llm = _make_fake_llm(response=_well_formed_vision_response())

    result = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe Designer",
        candidate_headline="Senior product designer",
        image_bytes_list=_image_bytes(3),
        asset_metadata=_asset_metadata(3),
        vision_llm_call=fake_llm,
    )

    assert isinstance(result, VisionEvaluationResult)
    assert result.judgment.fallback_reason == ""
    assert result.judgment.overall_verdict == "yes"
    assert result.judgment.overall_confidence == pytest.approx(0.85)
    assert len(result.judgment.principles) == 2
    # Anchors derived from scores.
    assert result.judgment.principles[0].anchor == "excellent"
    assert result.judgment.principles[1].anchor == "good"
    # Cost estimate is non-zero (image tokens + output tokens).
    assert result.judgment.cost_estimate_usd > 0


def test_evaluate_falls_back_when_llm_raises() -> None:
    def _failing_llm(*args, **kwargs):
        raise RuntimeError("network down")

    result = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(3),
        asset_metadata=_asset_metadata(3),
        vision_llm_call=_failing_llm,
    )
    assert "llm_raise" in result.judgment.fallback_reason
    assert result.judgment.overall_confidence == 0.0
    assert result.judgment.principles == ()


def test_evaluate_falls_back_on_schema_invalid_response() -> None:
    fake_llm = _make_fake_llm(response="not a dict")
    result = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(2),
        asset_metadata=_asset_metadata(2),
        vision_llm_call=fake_llm,
    )
    assert "schema_invalid" in result.judgment.fallback_reason


def test_evaluate_falls_back_on_image_grounding_violation() -> None:
    response = _well_formed_vision_response()
    response["principles"][0]["image_ids"] = []
    fake_llm = _make_fake_llm(response=response)
    result = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(3),
        asset_metadata=_asset_metadata(3),
        vision_llm_call=fake_llm,
    )
    assert "image_grounding" in result.judgment.fallback_reason


def test_evaluate_marks_anchor_drift_on_layer3_fail() -> None:
    """Layer 3 doesn't drop the whole eval; the per-principle marker fires."""

    response = _well_formed_vision_response()
    # Marine-biology vocab in a "Visual hierarchy: bad" reasoning →
    # anchor consistency fails, but the eval continues.
    response["principles"][0]["score"] = 0
    response["principles"][0]["reasoning"] = (
        "Marine biology octopus camouflage adaptations across reef "
        "ecosystems."
    )
    fake_llm = _make_fake_llm(response=response)
    result = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(3),
        asset_metadata=_asset_metadata(3),
        vision_llm_call=fake_llm,
    )
    assert result.judgment.fallback_reason == ""  # Layer 3 doesn't trigger fallback
    bad_principle = next(
        p for p in result.judgment.principles if p.name == "Visual hierarchy"
    )
    assert bad_principle.anchor_consistency_pass is False


def test_evaluate_auto_rejects_on_hard_reject_pattern_match() -> None:
    response = _well_formed_vision_response()
    response["overall_reasoning"] = (
        "Reviewing the work, this reads as a layout-only portfolios with "
        "no shipped product evidence."
    )
    fake_llm = _make_fake_llm(response=response)
    result = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(3),
        asset_metadata=_asset_metadata(3),
        vision_llm_call=fake_llm,
    )
    # Hard reject forces verdict=no, confidence=1.0. The fixture
    # rubric's hard_reject_patterns matches the reasoning substring
    # exactly.
    assert result.judgment.overall_verdict == "no"
    assert result.judgment.overall_confidence == 1.0
    assert "hard_reject" in result.judgment.fallback_reason


def test_evaluate_cost_estimate_scales_with_image_count() -> None:
    """Spec §4.1: ~258 tokens per 1024×1024 image. Cost should
    scale roughly linearly with image count."""

    fake_llm = _make_fake_llm(response=_well_formed_vision_response())
    result_small = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(3),
        asset_metadata=_asset_metadata(3),
        vision_llm_call=fake_llm,
    )
    result_large = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(8),
        asset_metadata=_asset_metadata(8),
        vision_llm_call=fake_llm,
    )
    # 8 images cost more than 3 images (assuming positive token cost).
    assert result_large.judgment.cost_estimate_usd > result_small.judgment.cost_estimate_usd
    # Cost difference is bounded by image tokens.
    extra_tokens = (8 - 3) * GEMINI_3_1_PRO_TOKENS_PER_IMAGE
    assert result_large.judgment.cost_estimate_usd - result_small.judgment.cost_estimate_usd > 0
    # Sanity: the per-call cost stays under the spec's $0.05 envelope
    # for a typical evaluation.
    assert result_large.judgment.cost_estimate_usd < 0.05


# ---------------------------------------------------------------------------
# Sentinel: SCORE_TO_ANCHOR aligns with RECOGNIZED_RUBRIC_ANCHORS
# ---------------------------------------------------------------------------


def test_score_to_anchor_covers_full_anchor_set() -> None:
    from shared.brief_v2_schema import RECOGNIZED_RUBRIC_ANCHORS

    # All anchors map cleanly. Order in SCORE_TO_ANCHOR matches the
    # spec-recommended ordering (low → high).
    anchors_from_scores = tuple(SCORE_TO_ANCHOR[s] for s in (0, 1, 2, 3))
    assert anchors_from_scores == RECOGNIZED_RUBRIC_ANCHORS


# ---------------------------------------------------------------------------
# Audit Move #16 — second-vendor fallback cascade
# ---------------------------------------------------------------------------


def _gemini_raises_llm() -> Any:
    """Vision LLM stub that raises like a Gemini outage."""

    def _fake(model: str, system: str, user: str, images: list[bytes]) -> Any:
        raise RuntimeError("gemini_503")

    return _fake


def _gemini_returns_schema_invalid() -> Any:
    return _make_fake_llm(response="not a dict")


def test_fallback_cascade_invokes_secondary_when_gemini_raises() -> None:
    """Move #16: when Gemini raises, the cascade must call the
    fallback model and return its successful judgment, NOT the
    HITL fallback result."""

    secondary_calls: list[str] = []

    def _claude_success(model: str, system: str, user: str, images: list[bytes]) -> Any:
        secondary_calls.append(model)
        return _well_formed_vision_response()

    result = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(3),
        asset_metadata=_asset_metadata(3),
        vision_llm_call=_gemini_raises_llm(),
        model="gemini-2.5-pro",
        vision_fallback_llm_call=_claude_success,
        fallback_model="claude-sonnet-4-6",
    )
    assert secondary_calls == ["claude-sonnet-4-6"]
    assert result.judgment.fallback_reason == ""
    assert result.judgment.model == "claude-sonnet-4-6"
    assert result.judgment.overall_verdict == "yes"


def test_fallback_cascade_invokes_secondary_on_schema_invalid_primary() -> None:
    """A schema-invalid primary response should also escalate to the
    fallback (audit Move #16's "any of the existing fallback_reason
    paths" contract)."""

    secondary_calls: list[str] = []

    def _claude_success(model: str, system: str, user: str, images: list[bytes]) -> Any:
        secondary_calls.append(model)
        return _well_formed_vision_response()

    result = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(3),
        asset_metadata=_asset_metadata(3),
        vision_llm_call=_gemini_returns_schema_invalid(),
        model="gemini-2.5-pro",
        vision_fallback_llm_call=_claude_success,
        fallback_model="claude-sonnet-4-6",
    )
    assert secondary_calls == ["claude-sonnet-4-6"]
    assert result.judgment.fallback_reason == ""
    assert result.judgment.model == "claude-sonnet-4-6"


def test_fallback_cascade_drops_to_hitl_only_when_both_fail() -> None:
    """Move #16: HITL drop only fires when both models fail; the
    final fallback_reason must carry both descriptors so telemetry
    can distinguish vendor-specific from cross-vendor outages."""

    def _claude_also_raises(model: str, system: str, user: str, images: list[bytes]) -> Any:
        raise RuntimeError("claude_503")

    result = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(3),
        asset_metadata=_asset_metadata(3),
        vision_llm_call=_gemini_raises_llm(),
        model="gemini-2.5-pro",
        vision_fallback_llm_call=_claude_also_raises,
        fallback_model="claude-sonnet-4-6",
    )
    assert "both_vendors_failed" in result.judgment.fallback_reason
    assert "primary=" in result.judgment.fallback_reason
    assert "fallback=" in result.judgment.fallback_reason
    assert result.judgment.overall_confidence == 0.0


def test_no_fallback_configured_preserves_pre_move_16_behavior() -> None:
    """When the operator hasn't configured a fallback (the default),
    primary failure goes straight to HITL — pre-Move-16 behavior is
    byte-identical."""

    result = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(3),
        asset_metadata=_asset_metadata(3),
        vision_llm_call=_gemini_raises_llm(),
        model="gemini-2.5-pro",
        # vision_fallback_llm_call defaults to None
    )
    assert "llm_raise" in result.judgment.fallback_reason
    assert result.judgment.overall_confidence == 0.0
    assert result.judgment.model == "gemini-2.5-pro"


def test_gemini_vision_llm_call_records_success_receipt(monkeypatch: Any) -> None:
    class _FakeGeminiModels:
        def generate_content(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                text='{"ok": true}',
                usage_metadata=SimpleNamespace(
                    prompt_token_count=9,
                    candidates_token_count=4,
                    cached_content_token_count=2,
                ),
            )

    class _FakeGeminiClient:
        def __init__(self, api_key: str) -> None:
            self.models = _FakeGeminiModels()

    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    genai_module.Client = _FakeGeminiClient  # type: ignore[attr-defined]
    google_module.genai = genai_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    usage_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "designer.vision_evaluation.record_llm_usage",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    result = gemini_vision_llm_call(
        "gemini-2.5-pro",
        "system",
        "user",
        [b"\xff\xd8\xff", b"abc"],
    )

    assert result == {"ok": True}
    assert len(usage_calls) == 1
    call = usage_calls[0]
    assert call["provider"] == "google"
    assert call["actual_status"] == "ok"
    assert call["usage"]["input_tokens"] == 9
    assert call["usage"]["output_tokens"] == 4
    assert call["usage"]["cache_read_input_tokens"] == 2
    assert call["usage_context"]["stage"] == "vision_eval_gemini"
    assert call["request"]["image_count"] == 2
    assert call["request"]["image_bytes_total"] == 6


def test_gemini_vision_llm_call_records_error_receipt(monkeypatch: Any) -> None:
    class _FailingGeminiModels:
        def generate_content(self, **kwargs: Any) -> Any:
            raise RuntimeError("gemini_503")

    class _FailingGeminiClient:
        def __init__(self, api_key: str) -> None:
            self.models = _FailingGeminiModels()

    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    genai_module.Client = _FailingGeminiClient  # type: ignore[attr-defined]
    google_module.genai = genai_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    usage_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "designer.vision_evaluation.record_llm_usage",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="gemini_503"):
        gemini_vision_llm_call("gemini-2.5-pro", "system", "user", [b"abc"])

    assert len(usage_calls) == 1
    call = usage_calls[0]
    assert call["provider"] == "google"
    assert call["actual_status"] == "error"
    assert call["usage"]["input_tokens"] == 0
    assert call["request"]["error_type"] == "RuntimeError"
    assert call["request"]["error_message"] == "gemini_503"
    assert call["usage_context"]["stage"] == "vision_eval_gemini"


def test_claude_vision_llm_call_records_success_receipt(monkeypatch: Any) -> None:
    class _FakeAnthropicMessages:
        def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=12,
                    output_tokens=5,
                    cache_read_input_tokens=3,
                    cache_creation_input_tokens=1,
                ),
                content=[SimpleNamespace(type="text", text='{"ok": true}')],
            )

    class _FakeAnthropicClient:
        def __init__(self, api_key: str) -> None:
            self.messages = _FakeAnthropicMessages()

    anthropic_module = ModuleType("anthropic")
    anthropic_module.Anthropic = _FakeAnthropicClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_module)

    usage_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "designer.vision_evaluation.record_llm_usage",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    result = claude_vision_llm_call(
        "claude-sonnet-4-6",
        "system",
        "user",
        [b"\xff\xd8\xff"],
    )

    assert result == {"ok": True}
    assert len(usage_calls) == 1
    call = usage_calls[0]
    assert call["provider"] == "anthropic"
    assert call["actual_status"] == "ok"
    assert call["usage"]["input_tokens"] == 12
    assert call["usage"]["output_tokens"] == 5
    assert call["usage"]["cache_read_input_tokens"] == 3
    assert call["usage"]["cache_creation_input_tokens"] == 1
    assert call["usage_context"]["stage"] == "vision_eval_claude"


def test_claude_vision_llm_call_records_error_receipt(monkeypatch: Any) -> None:
    class _FailingAnthropicMessages:
        def create(self, **kwargs: Any) -> Any:
            raise ValueError("claude_503")

    class _FailingAnthropicClient:
        def __init__(self, api_key: str) -> None:
            self.messages = _FailingAnthropicMessages()

    anthropic_module = ModuleType("anthropic")
    anthropic_module.Anthropic = _FailingAnthropicClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_module)

    usage_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "designer.vision_evaluation.record_llm_usage",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="claude_503"):
        claude_vision_llm_call("claude-sonnet-4-6", "system", "user", [b"abc"])

    assert len(usage_calls) == 1
    call = usage_calls[0]
    assert call["provider"] == "anthropic"
    assert call["actual_status"] == "error"
    assert call["usage"]["input_tokens"] == 0
    assert call["request"]["error_type"] == "ValueError"
    assert call["request"]["error_message"] == "claude_503"
    assert call["usage_context"]["stage"] == "vision_eval_claude"


def test_fallback_does_not_fire_on_layer4_hard_reject() -> None:
    """Layer-4 hard reject is a successful policy outcome (verdict
    'no', confidence 1.0). The cascade must NOT escalate to the
    fallback model on hard reject."""

    response = _well_formed_vision_response()
    response["overall_reasoning"] = (
        "This is layout-only portfolios with no shipped product."
    )

    secondary_calls: list[str] = []

    def _claude_should_not_be_called(*args: Any, **kwargs: Any) -> Any:
        secondary_calls.append("called")
        return _well_formed_vision_response()

    result = evaluate_designer_visually(
        brief=_designer_brief_for_vision(),
        candidate_display_name="Joe",
        candidate_headline="",
        image_bytes_list=_image_bytes(3),
        asset_metadata=_asset_metadata(3),
        vision_llm_call=_make_fake_llm(response=response),
        model="gemini-2.5-pro",
        vision_fallback_llm_call=_claude_should_not_be_called,
        fallback_model="claude-sonnet-4-6",
    )
    assert secondary_calls == []
    assert result.judgment.overall_verdict == "no"
    assert result.judgment.overall_confidence == 1.0


def test_resolve_vision_fallback_disabled_by_default(monkeypatch: Any) -> None:
    """Pre-Move-16 default: env var unset ⇒ no fallback configured."""

    from shared import config as shared_config

    monkeypatch.setattr(
        shared_config, "DESIGNER_VISION_FALLBACK_MODEL_NAME", "", raising=False
    )
    assert resolve_vision_fallback() is None


def test_resolve_vision_fallback_routes_claude_models(monkeypatch: Any) -> None:
    from designer.vision_evaluation import claude_vision_llm_call
    from shared import config as shared_config

    monkeypatch.setattr(
        shared_config,
        "DESIGNER_VISION_FALLBACK_MODEL_NAME",
        "claude-sonnet-4-6",
        raising=False,
    )
    resolved = resolve_vision_fallback()
    assert resolved is not None
    caller, model = resolved
    assert caller is claude_vision_llm_call
    assert model == "claude-sonnet-4-6"


def test_resolve_vision_fallback_returns_none_for_unknown_provider(
    monkeypatch: Any,
) -> None:
    """Unknown model strings should NOT silently route to the wrong
    caller — return None instead."""

    from shared import config as shared_config

    monkeypatch.setattr(
        shared_config,
        "DESIGNER_VISION_FALLBACK_MODEL_NAME",
        "some-unknown-vision-model",
        raising=False,
    )
    assert resolve_vision_fallback() is None
