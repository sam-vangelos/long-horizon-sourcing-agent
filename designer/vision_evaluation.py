"""Designer module — vision-LLM evaluation against the brief-encoded rubric.

Designer Slice 5. The load-bearing differentiation: Gemini 3.1 Pro
reads the cached portfolio images (per :mod:`designer.image_acquisition`)
and produces a per-principle structured judgment grounded in the
:class:`shared.brief_v2_schema.BriefDesignRubric`.

Four-layer hallucination guard cascade (mirrors brief_polish.py's
named-cascade pattern):

1. Schema validity — the Gemini response must parse to the structured
   schema (per-principle score + reasoning + image_ids cited;
   overall verdict + confidence). Parse failure → fallback to text-
   only contextualization (the Slice-2 prompt's output).
2. Image-grounding — each per-principle reasoning must reference at
   least one ``image_id`` from the input set. If reasoning references
   no image, the model is talking from priors not from the
   candidate's work — fallback to lower confidence.
3. Anchor consistency — the per-principle reasoning text must
   lexically overlap (Jaccard ≥ ``ANCHOR_OVERLAP_THRESHOLD`` on
   ≥3-char alpha tokens) with the assigned anchor's definition.
   Mirrors :func:`market_intelligence.brief_polish._capability_area_overlap`'s
   pattern. If the model says "kerning precision" but its score is
   "okay" and the okay-anchor mentions "consistent without precision",
   that's anchor drift → lower confidence.
4. Hard-reject pattern check — if the brief carries
   ``hard_reject_patterns`` and any pattern appears (case-insensitive
   substring match) in the overall reasoning, the candidate is
   auto-rejected — NOT surfaced as a hitl_visual_review save.

The cross-check pass against Sonnet 4.6 on top-decile candidates
arrives in Slice 8 — this module's interface accepts a
``cross_check_model`` parameter that Slice 8 wires.

Cost telemetry per call lands in the ``visual_judgment.cost_estimate``
field so workspace surfaces can show per-customer monthly spend
without re-invoking the API.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from shared.brief_v2_schema import RECOGNIZED_RUBRIC_ANCHORS
from shared.llm_usage import anthropic_usage_dict, google_usage_dict, record_llm_usage
from shared.observability import observe


# Anchor consistency Jaccard threshold (mirrors HALLUCINATION_OVERLAP_THRESHOLD
# in brief_polish.py at 0.30 — chose lower (0.20) here because per-
# principle reasoning is shorter and naturally has less overlap with
# the anchor definitions). Tunable post-trial based on observed
# distribution.
ANCHOR_OVERLAP_THRESHOLD: float = 0.20

# Token regex matches the brief_polish overlap pattern.
_OVERLAP_TOKEN_RE = re.compile(r"[a-z]{3,}")

# Per-principle decision tier names — the structured output's score
# field uses these as integers (0/1/2/3) keyed to the four anchors
# (bad/okay/good/excellent).
SCORE_TO_ANCHOR: dict[int, str] = {
    0: "bad",
    1: "okay",
    2: "good",
    3: "excellent",
}


# Sentinels used in the visual_judgment payload's overall_verdict
# field. Match the recruiter-facing language the workspace renders.
VERDICT_VALUES = frozenset({"yes", "no", "borderline"})


@dataclass
class VisualJudgmentPrinciple:
    """Per-principle scoring outcome.

    ``image_ids`` references which images informed the score (input-
    indexed; the asset cache assigns a stable order at evaluation
    time). ``anchor_consistency_pass`` captures the Layer-3
    hallucination-guard outcome — when False, the workspace renders
    a "model anchor drift" caveat under the score chip.
    """

    name: str
    score: int  # 0-3, mapped to anchor via SCORE_TO_ANCHOR
    anchor: str  # bad | okay | good | excellent
    reasoning: str
    image_ids: tuple[int, ...]
    anchor_consistency_pass: bool = True


@dataclass
class VisualJudgment:
    """Complete vision-evaluation outcome for one candidate.

    The shape that lands in ``terminal_payload_json["visual_judgment"]``
    in Slice 6 (the read-model contract change). ``surface_type`` is
    set by the side-effects layer after Slice 6's rendering branches
    are wired.
    """

    model: str
    principles: tuple[VisualJudgmentPrinciple, ...]
    overall_verdict: str  # yes | no | borderline
    overall_confidence: float
    fallback_reason: str = ""  # non-empty when a guard layer fired
    cost_estimate_usd: float = 0.0
    cross_check: dict[str, Any] | None = None  # Slice 8 populates


@dataclass(frozen=True)
class _AssetReference:
    """The per-asset metadata the prompt references by id."""

    image_id: int
    asset_url: str
    source: str
    project_title: str


@dataclass
class VisionEvaluationResult:
    """Outcome of one ``evaluate_designer_visually`` call.

    Carries the structured judgment, the prompt that produced it
    (for debugging and for the visual-judgment workspace surface), and
    the asset-reference table the prompt grounded itself in.
    """

    judgment: VisualJudgment
    asset_references: tuple[_AssetReference, ...]
    raw_response: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gemini client signature — typed as a callable so tests inject fakes
# ---------------------------------------------------------------------------


# A vision LLM call takes ``(model_name, system_prompt, user_text,
# image_bytes_list)`` and returns parsed structured JSON. Production
# wires this to ``google.genai``'s vision-capable client; tests use
# a deterministic fake.
VisionLLMCall = Callable[[str, str, str, list[bytes]], dict[str, Any]]


def _sniff_image_mime(data: bytes) -> str:
    """Detect image MIME type from the magic bytes; default to JPEG.

    Portfolio CDNs (Behance, Google CSE thumbnails) serve PNG/WebP/GIF
    alongside JPEG. Sending the wrong mime to Gemini fails layer-1 with
    ``schema_invalid``; the fallback model fails for the same reason;
    every candidate silently degrades. Sniffing is a few bytes of
    insurance.
    """

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def _record_vision_llm_usage(
    *,
    provider: str,
    model: str,
    stage: str,
    system_prompt: str,
    user_text: str,
    image_bytes_list: list[bytes],
    usage: dict[str, Any] | None = None,
    actual_status: str = "ok",
    exc: Exception | None = None,
) -> None:
    """Best-effort typed LLM receipt for direct vision provider calls."""

    request = {
        "system_prompt_chars": len(system_prompt),
        "user_text_chars": len(user_text),
        "image_count": len(image_bytes_list),
        "image_bytes_total": sum(len(image_bytes) for image_bytes in image_bytes_list),
        "temperature": 0.1,
        "response_mime_type": "application/json",
    }
    if exc is not None:
        request["error_type"] = type(exc).__name__
        request["error_message"] = str(exc)[:240]

    try:
        record_llm_usage(
            provider=provider,
            model=model,
            usage=usage
            or {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            request=request,
            usage_context={"module": "designer", "stage": stage},
            actual_status=actual_status,
        )
    except Exception:  # noqa: BLE001 — telemetry must not break the pipeline
        pass


def gemini_vision_llm_call(
    model: str,
    system_prompt: str,
    user_text: str,
    image_bytes_list: list[bytes],
) -> dict[str, Any]:
    """Production vision-LLM call wired to ``google.genai``.

    Slice 5 ships this thin wrapper; the cross-check pass in Slice 8
    will plug Claude Sonnet 4.6 in via its own callable matching the
    same :data:`VisionLLMCall` shape.

    NOT exercised in unit tests — the surrounding pipeline tests inject
    a deterministic fake. Live integration tests covered by Slice 11.
    """

    parts: list[Any] = [system_prompt, user_text]
    for image_bytes in image_bytes_list:
        parts.append({"mime_type": _sniff_image_mime(image_bytes), "data": image_bytes})
    try:
        from google import genai  # noqa: PLC0415 — lazy import

        from shared import config

        client = genai.Client(api_key=getattr(config, "GOOGLE_API_KEY", ""))
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        )
    except Exception as exc:
        _record_vision_llm_usage(
            provider="google",
            model=model,
            stage="vision_eval_gemini",
            system_prompt=system_prompt,
            user_text=user_text,
            image_bytes_list=image_bytes_list,
            actual_status="error",
            exc=exc,
        )
        raise
    _record_vision_llm_usage(
        provider="google",
        model=model,
        stage="vision_eval_gemini",
        system_prompt=system_prompt,
        user_text=user_text,
        image_bytes_list=image_bytes_list,
        usage=google_usage_dict(response),
    )
    text = (response.text or "").strip()
    if text.startswith("```"):
        # Strip markdown code fences if present.
        lines = text.split("\n")
        lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def claude_vision_llm_call(
    model: str,
    system_prompt: str,
    user_text: str,
    image_bytes_list: list[bytes],
) -> dict[str, Any]:
    """Anthropic-vision call matching the :data:`VisionLLMCall` shape.

    Audit Move #16 — second-vendor fallback when Gemini fails. Lazy
    import of ``anthropic`` so the dependency is only resolved when
    the cascade actually fires. Returns the same JSON-decoded dict
    shape ``gemini_vision_llm_call`` returns so the four-layer guard
    cascade applies uniformly.

    NOT exercised in unit tests — production callers inject a
    deterministic fake. Live integration covered by the operator's
    smoke run before a customer demo.
    """

    import base64  # noqa: PLC0415 — lazy

    from shared import config

    content: list[dict[str, Any]] = []
    for image_bytes in image_bytes_list:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                },
            }
        )
    content.append({"type": "text", "text": user_text})
    try:
        import anthropic  # noqa: PLC0415 — lazy

        client = anthropic.Anthropic(api_key=getattr(config, "ANTHROPIC_API_KEY", ""))
        response = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=4096,
            temperature=0.1,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        _record_vision_llm_usage(
            provider="anthropic",
            model=model,
            stage="vision_eval_claude",
            system_prompt=system_prompt,
            user_text=user_text,
            image_bytes_list=image_bytes_list,
            actual_status="error",
            exc=exc,
        )
        raise
    _record_vision_llm_usage(
        provider="anthropic",
        model=model,
        stage="vision_eval_claude",
        system_prompt=system_prompt,
        user_text=user_text,
        image_bytes_list=image_bytes_list,
        usage=anthropic_usage_dict(response),
    )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def resolve_vision_fallback() -> tuple[VisionLLMCall, str] | None:
    """Resolve the configured (caller, model) tuple for the secondary
    vision provider, or ``None`` when no fallback is configured.

    Audit Move #16 — keyed off the env var
    ``DESIGNER_VISION_FALLBACK_MODEL_NAME`` (read via
    :mod:`shared.config`). Returns ``None`` when the var is unset
    (default — pre-Move-16 behavior). When the var names a Claude
    model, returns :func:`claude_vision_llm_call`. Future providers
    can be added here without touching :func:`evaluate_designer_visually`.
    """

    from shared import config  # noqa: PLC0415 — lazy

    fallback_model = getattr(config, "DESIGNER_VISION_FALLBACK_MODEL_NAME", "").strip()
    if not fallback_model:
        return None
    if fallback_model.startswith("claude-"):
        return claude_vision_llm_call, fallback_model
    # Future providers append here. Unknown model name ⇒ return None
    # rather than silently routing to the wrong caller.
    return None


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def assemble_vision_evaluation_system_prompt(brief: dict[str, Any]) -> str:
    """Build the Gemini 3.1 Pro system prompt for vision evaluation.

    Encodes the rubric (principles + anchors + weights), any
    discipline weight overrides, calibration exemplars (URLs +
    verdicts + per-principle reasoning the recruiter wrote), and the
    structured-output schema. Mirrors the editorial register of the
    Slice-2 text-based prompt.
    """

    rubric = brief.get("design_rubric") or {}
    if not isinstance(rubric, dict):
        rubric = {}

    principles_block = _principles_prompt_block(rubric.get("principles") or [])
    discipline_overrides_block = _discipline_overrides_block(
        rubric.get("discipline_weight_overrides") or {}
    )
    exemplars_block = _exemplars_prompt_block(
        rubric.get("calibration_exemplars") or []
    )
    hard_reject_block = _hard_reject_block(rubric.get("hard_reject_patterns") or [])

    return (
        "You are a senior design director critiquing a portfolio review. "
        "Substantive, specific, principle-grounded — no marketing copy, no "
        "hedging.\n\n"
        f"{principles_block}"
        f"{discipline_overrides_block}"
        f"{exemplars_block}"
        f"{hard_reject_block}"
        "OUTPUT FORMAT (JSON only — no preamble, no markdown fences):\n"
        "{\n"
        '  "principles": [\n'
        '    {"name": "<principle name>", "score": <0|1|2|3>, '
        '"reasoning": "<1-2 sentences referencing specific image_ids>", '
        '"image_ids": [<int>, ...]}, ...\n'
        "  ],\n"
        '  "overall_verdict": "<yes|no|borderline>",\n'
        '  "overall_confidence": <0.0-1.0>,\n'
        '  "overall_reasoning": "<1-2 sentence summary>"\n'
        "}\n\n"
        "Each principle MUST cite at least one image_id; reasoning that "
        "doesn't reference images is treated as hallucination and the "
        "evaluation falls back to text-only contextualization."
    )


def _principles_prompt_block(principles: list[Any]) -> str:
    if not principles:
        return ""
    lines = ["RUBRIC PRINCIPLES (score each 0=bad, 1=okay, 2=good, 3=excellent):"]
    for principle in principles:
        if not isinstance(principle, dict):
            continue
        name = str(principle.get("name") or "").strip()
        description = str(principle.get("description") or "").strip()
        anchors = principle.get("anchors") or {}
        if not name:
            continue
        lines.append(f"\n  - {name}")
        if description:
            lines.append(f"    {description}")
        if isinstance(anchors, dict):
            for level in RECOGNIZED_RUBRIC_ANCHORS:
                anchor_text = anchors.get(level)
                if isinstance(anchor_text, str) and anchor_text.strip():
                    lines.append(f"    [{level}] {anchor_text.strip()}")
    return "\n".join(lines) + "\n\n"


def _discipline_overrides_block(overrides: dict[str, Any]) -> str:
    if not overrides:
        return ""
    lines = ["DISCIPLINE WEIGHT OVERRIDES (multiply principle weight by the listed value):"]
    for discipline, weights in overrides.items():
        if not isinstance(weights, dict):
            continue
        weight_strs = [
            f"{name}: {weight}"
            for name, weight in weights.items()
            if isinstance(weight, (int, float))
        ]
        if weight_strs:
            lines.append(f"  {discipline}: {', '.join(weight_strs)}")
    return "\n".join(lines) + "\n\n"


def _exemplars_prompt_block(exemplars: list[Any]) -> str:
    if not exemplars:
        return ""
    lines = ["CALIBRATION EXEMPLARS (the recruiter's yes/no/borderline calls on prior portfolios):"]
    for ex in exemplars[:5]:  # cap at 5 to keep token cost bounded
        if not isinstance(ex, dict):
            continue
        url = str(ex.get("portfolio_url") or "")
        verdict = str(ex.get("verdict") or "")
        overall = str(ex.get("overall_reasoning") or "")
        lines.append(f"  - {url} [{verdict}] {overall}")
    return "\n".join(lines) + "\n\n"


def _hard_reject_block(patterns: list[Any]) -> str:
    if not patterns:
        return ""
    lines = ["HARD REJECT PATTERNS (auto-reject if observed in this portfolio):"]
    for p in patterns:
        if isinstance(p, str) and p.strip():
            lines.append(f"  - {p.strip()}")
    return "\n".join(lines) + "\n\n"


def assemble_vision_evaluation_user_text(
    *,
    candidate_display_name: str,
    candidate_headline: str,
    asset_references: tuple[_AssetReference, ...],
) -> str:
    """Build the user-facing text prompt that accompanies the image batch.

    The image bytes are passed separately to the LLM call (multimodal
    contents); the text prompt enumerates each image_id alongside the
    project context (which Behance project it's from, etc.) so the
    model's reasoning can reference image_ids meaningfully.
    """

    lines = [
        f"Candidate: {candidate_display_name}",
    ]
    if candidate_headline:
        lines.append(f"Headline: {candidate_headline}")
    lines.append("\nImages provided (cite by image_id in your reasoning):")
    for ref in asset_references:
        descriptor = ref.project_title or ref.asset_url
        lines.append(f"  image_id={ref.image_id}: [{ref.source}] {descriptor}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hallucination guard cascade
# ---------------------------------------------------------------------------


def _layer1_schema_validity(raw: Any) -> str | None:
    """Return a fallback descriptor when the response shape is invalid."""

    if not isinstance(raw, dict):
        return "schema_invalid:not_dict"
    principles = raw.get("principles")
    if not isinstance(principles, list) or not principles:
        return "schema_invalid:no_principles"
    for idx, principle in enumerate(principles):
        if not isinstance(principle, dict):
            return f"schema_invalid:principle[{idx}]_not_dict"
        if not isinstance(principle.get("name"), str):
            return f"schema_invalid:principle[{idx}]_name"
        if principle.get("score") not in (0, 1, 2, 3):
            return f"schema_invalid:principle[{idx}]_score"
        if not isinstance(principle.get("reasoning"), str):
            return f"schema_invalid:principle[{idx}]_reasoning"
        image_ids = principle.get("image_ids")
        if not isinstance(image_ids, list):
            return f"schema_invalid:principle[{idx}]_image_ids"
    if raw.get("overall_verdict") not in VERDICT_VALUES:
        return "schema_invalid:overall_verdict"
    overall_confidence = raw.get("overall_confidence")
    if not isinstance(overall_confidence, (int, float)):
        return "schema_invalid:overall_confidence"
    if not (0.0 <= float(overall_confidence) <= 1.0):
        return "schema_invalid:overall_confidence_range"
    return None


def _layer2_image_grounding(raw: dict[str, Any]) -> str | None:
    """Return a fallback descriptor when any principle cites no images."""

    principles = raw.get("principles") or []
    for idx, principle in enumerate(principles):
        if not isinstance(principle, dict):
            continue
        image_ids = principle.get("image_ids") or []
        if not isinstance(image_ids, list) or len(image_ids) == 0:
            return f"image_grounding:principle[{idx}]_no_image_ids"
    return None


def _jaccard_overlap(text_a: str, text_b: str) -> float:
    tokens_a = set(_OVERLAP_TOKEN_RE.findall(text_a.lower()))
    tokens_b = set(_OVERLAP_TOKEN_RE.findall(text_b.lower()))
    if not tokens_a or not tokens_b:
        # Can't measure → don't penalize. Mirrors brief_polish's
        # `_capability_area_overlap` posture for empty inputs.
        return 1.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _layer3_anchor_consistency(
    raw: dict[str, Any],
    *,
    rubric_principles: list[Any],
) -> dict[str, bool]:
    """Per-principle anchor-consistency map (name → pass/fail)."""

    anchor_lookup: dict[str, dict[str, str]] = {}
    for principle in rubric_principles:
        if not isinstance(principle, dict):
            continue
        name = principle.get("name")
        anchors = principle.get("anchors")
        if isinstance(name, str) and isinstance(anchors, dict):
            anchor_lookup[name] = {
                level: text
                for level, text in anchors.items()
                if isinstance(text, str)
            }

    pass_map: dict[str, bool] = {}
    for principle in raw.get("principles") or []:
        if not isinstance(principle, dict):
            continue
        name = principle.get("name") or ""
        score = principle.get("score")
        reasoning = principle.get("reasoning") or ""
        anchor = SCORE_TO_ANCHOR.get(int(score)) if isinstance(score, int) else None
        if anchor is None:
            pass_map[name] = False
            continue
        anchor_text = anchor_lookup.get(name, {}).get(anchor, "")
        if not anchor_text:
            # No anchor definition for this principle/level — pass
            # (we can't measure consistency without a reference).
            pass_map[name] = True
            continue
        pass_map[name] = (
            _jaccard_overlap(reasoning, anchor_text) >= ANCHOR_OVERLAP_THRESHOLD
        )
    return pass_map


def _layer4_hard_reject(
    raw: dict[str, Any],
    *,
    hard_reject_patterns: list[str],
) -> str | None:
    """Return descriptor of the matched hard-reject pattern, else None."""

    if not hard_reject_patterns:
        return None
    overall_text = (raw.get("overall_reasoning") or "").lower()
    for pattern in hard_reject_patterns:
        if not isinstance(pattern, str) or not pattern.strip():
            continue
        if pattern.lower() in overall_text:
            return f"hard_reject:matched_pattern={pattern[:60]!r}"
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


# Pricing estimate — Gemini 3.1 Pro Preview vision call.
# Hardcoded for the per-call cost estimate; revisit when
# Google adjusts pricing.
# TODO: update rates when Google publishes official 3.1 Pro pricing
GEMINI_3_1_PRO_INPUT_USD_PER_1M_TOKENS = 1.25
GEMINI_3_1_PRO_OUTPUT_USD_PER_1M_TOKENS = 10.0
GEMINI_3_1_PRO_TOKENS_PER_IMAGE = 258  # spec §4.1 measurement


# Slice 8: Sonnet 4.6 cross-check pricing. ~10x cost differential vs
# Gemini per spec §4.1 — chosen for cross-check on top-decile only.
CLAUDE_SONNET_4_6_INPUT_USD_PER_1M_TOKENS = 3.0
CLAUDE_SONNET_4_6_OUTPUT_USD_PER_1M_TOKENS = 15.0
CLAUDE_SONNET_4_6_TOKENS_PER_IMAGE = 1334  # spec §4.1 measurement

CLAUDE_SONNET_4_6_MODEL_NAME = "claude-sonnet-4-6"


# Slice 8: which fraction of top candidates get the cross-check pass.
# Per spec §4.3: "top-decile" = top 10% by overall_score × confidence.
DEFAULT_CROSS_CHECK_TOP_DECILE_FRACTION = 0.10
# Don't cross-check fewer than this number even when the run yields
# very few candidates — a 5-candidate run still wants 1 cross-check.
DEFAULT_MIN_CROSS_CHECK_COUNT = 1
# Don't cross-check more than this absolute count even on huge runs —
# protects per-run cost envelope ($0.10-0.20 per cross-check × 10
# = ~$2 cap on cross-check spend).
DEFAULT_MAX_CROSS_CHECK_COUNT = 10

# Disagreement threshold per spec §4.3: "Disagreement >1 anchor level
# on any principle". Anchors are 0-3; a delta of 2 or more is
# disagreement (off by one is acceptable).
CROSS_CHECK_DISAGREEMENT_ANCHOR_DELTA = 2


def _emit_vision_guard_attributes(result: "VisionEvaluationResult") -> None:
    """Phase 1 of Langfuse adoption: surface the 4-layer guard outcome.

    Emits ``vision.layer1_schema_validity`` / ``vision.layer2_image_grounding``
    / ``vision.layer3_anchor_consistency`` / ``vision.layer4_hard_reject``
    booleans plus ``vision.fallback_reason`` (verbatim from the
    judgment) as Langfuse span attributes on the active observation.
    The reason string convention encoded in :func:`_layer1_schema_validity`
    et al uses ``schema_invalid:...``, ``image_grounding:...``,
    ``hard_reject:...``, and ``llm_raise:<ExceptionClass>`` prefixes;
    the parser below matches those prefixes to derive the per-layer
    booleans. No-op when the Langfuse client is null / disabled /
    network-degraded.
    """

    judgment = result.judgment
    fallback_reason = judgment.fallback_reason or ""

    # Layer-3 (anchor consistency) is per-principle, not a fallback
    # trigger — read the per-principle pass count from the judgment's
    # principles tuple. Surface as a fraction so aggregate filters can
    # find runs where ``vision.layer3_anchor_consistency < 1.0``.
    if judgment.principles:
        passes = sum(
            1 for p in judgment.principles if p.anchor_consistency_pass
        )
        layer3_fraction = passes / len(judgment.principles)
    else:
        layer3_fraction = 1.0

    layer1_pass = not fallback_reason.startswith("schema_invalid")
    layer2_pass = not fallback_reason.startswith("image_grounding")
    # Layer 4 fires when fallback_reason starts with "hard_reject".
    # The hard-reject path is a SUCCESSFUL policy outcome (not an
    # error), so layer4=true means "rejected by policy."
    layer4_fired = fallback_reason.startswith("hard_reject")

    try:
        from shared.observability import update_current_observation

        update_current_observation(
            metadata={
                "vision.layer1_schema_validity": layer1_pass,
                "vision.layer2_image_grounding": layer2_pass,
                "vision.layer3_anchor_consistency": round(layer3_fraction, 3),
                "vision.layer4_hard_reject": layer4_fired,
                "vision.fallback_reason": fallback_reason,
                "vision.model": judgment.model,
            }
        )
    except Exception:  # noqa: BLE001 — Langfuse path is fail-soft
        pass


@observe(name="vision.evaluate")
def evaluate_designer_visually(
    *,
    brief: dict[str, Any],
    candidate_display_name: str,
    candidate_headline: str,
    image_bytes_list: list[bytes],
    asset_metadata: list[tuple[str, str, str]],  # (asset_url, source, project_title) per image
    vision_llm_call: VisionLLMCall = gemini_vision_llm_call,
    model: str = "gemini-3.1-pro-preview",
    vision_fallback_llm_call: VisionLLMCall | None = None,
    fallback_model: str = "",
) -> VisionEvaluationResult:
    """Run the vision-evaluation pipeline for one candidate.

    Returns a :class:`VisionEvaluationResult` whose ``judgment`` carries
    the structured outcome (per-principle scores + overall verdict +
    confidence + cost estimate). When any hallucination guard fires,
    ``judgment.fallback_reason`` is non-empty and ``judgment.overall_confidence``
    is downgraded to 0.0 — the workspace surface uses these signals
    to render the candidate in a "needs HITL review" lane.

    Audit Move #16 — second-vendor cascade:

    When the primary call (default Gemini 3.1 Pro) hits a recoverable
    failure path (``llm_raise`` / layer 1 schema-validity / layer 2
    image-grounding), the cascade retries against
    ``vision_fallback_llm_call`` + ``fallback_model`` before dropping
    the candidate to HITL. If both models hit recoverable failures,
    the final fallback_reason carries a ``primary=...,fallback=...``
    descriptor so telemetry can distinguish "Gemini transient" from
    "both vendors chronically failing."

    Layer-3 (anchor consistency) and layer-4 (hard reject) are NOT
    fallback triggers — layer 3 marks individual principles; layer 4
    is a successful policy outcome (auto-reject with confidence 1.0).

    Backward-compat: when ``vision_fallback_llm_call`` is None (the
    default), behavior is byte-identical to pre-Move-16 — primary
    failure goes straight to HITL. Production callers configure the
    cascade via :func:`resolve_vision_fallback`, which reads the
    ``DESIGNER_VISION_FALLBACK_MODEL_NAME`` env var.

    NOT a generator. Synchronous call returning a single result —
    the orchestrator parallelizes across candidates at a higher level.
    """

    rubric = brief.get("design_rubric") or {}
    if not isinstance(rubric, dict):
        rubric = {}
    rubric_principles = rubric.get("principles") or []
    hard_reject_patterns = rubric.get("hard_reject_patterns") or []
    if not isinstance(hard_reject_patterns, list):
        hard_reject_patterns = []

    asset_references = tuple(
        _AssetReference(
            image_id=idx,
            asset_url=meta[0],
            source=meta[1],
            project_title=meta[2],
        )
        for idx, meta in enumerate(asset_metadata)
    )

    system_prompt = assemble_vision_evaluation_system_prompt(brief)
    user_text = assemble_vision_evaluation_user_text(
        candidate_display_name=candidate_display_name,
        candidate_headline=candidate_headline,
        asset_references=asset_references,
    )

    primary_result, primary_fallback_reason = _attempt_vision_eval(
        vision_llm_call=vision_llm_call,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        image_bytes_list=image_bytes_list,
        asset_references=asset_references,
        rubric_principles=rubric_principles,
        hard_reject_patterns=hard_reject_patterns,
    )
    if primary_fallback_reason is None:
        _emit_vision_guard_attributes(primary_result)
        return primary_result

    # Primary cascaded out (layer 1 / layer 2 / llm_raise). If the
    # operator configured a second-vendor fallback, retry against it
    # before dropping to HITL.
    if vision_fallback_llm_call is None or not fallback_model:
        _emit_vision_guard_attributes(primary_result)
        return primary_result

    fallback_result, fallback_reason = _attempt_vision_eval(
        vision_llm_call=vision_fallback_llm_call,
        model=fallback_model,
        system_prompt=system_prompt,
        user_text=user_text,
        image_bytes_list=image_bytes_list,
        asset_references=asset_references,
        rubric_principles=rubric_principles,
        hard_reject_patterns=hard_reject_patterns,
    )
    if fallback_reason is None:
        _emit_vision_guard_attributes(fallback_result)
        return fallback_result

    # Both vendors hit recoverable failures. Build a fallback result
    # whose reason carries both descriptors so post-trial telemetry
    # can distinguish vendor-specific from cross-vendor outages.
    final_result = _fallback_result(
        model=fallback_model,
        asset_references=asset_references,
        fallback_reason=(
            f"both_vendors_failed primary={primary_fallback_reason} "
            f"fallback={fallback_reason}"
        ),
    )
    _emit_vision_guard_attributes(final_result)
    return final_result


@observe(name="vision.attempt")
def _attempt_vision_eval(
    *,
    vision_llm_call: VisionLLMCall,
    model: str,
    system_prompt: str,
    user_text: str,
    image_bytes_list: list[bytes],
    asset_references: tuple[_AssetReference, ...],
    rubric_principles: list[Any],
    hard_reject_patterns: list[Any],
) -> tuple[VisionEvaluationResult, str | None]:
    """Run one model's vision eval through the four-layer cascade.

    Returns ``(result, None)`` on a successful eval (including layer-4
    hard-reject which is a successful policy outcome). Returns
    ``(fallback_result, reason)`` when a recoverable failure path
    fires (llm_raise / layer 1 / layer 2). Layer 3 is not a failure.
    The caller decides whether to escalate to a fallback model
    based on ``reason``.
    """

    try:
        raw = vision_llm_call(model, system_prompt, user_text, image_bytes_list)
    except Exception as exc:
        reason = f"llm_raise:{exc.__class__.__name__}"
        return (
            _fallback_result(
                model=model,
                asset_references=asset_references,
                fallback_reason=reason,
            ),
            reason,
        )

    layer1 = _layer1_schema_validity(raw)
    if layer1 is not None:
        return (
            _fallback_result(
                model=model,
                asset_references=asset_references,
                fallback_reason=layer1,
                raw_response=raw if isinstance(raw, dict) else {},
            ),
            layer1,
        )

    layer2 = _layer2_image_grounding(raw)
    if layer2 is not None:
        return (
            _fallback_result(
                model=model,
                asset_references=asset_references,
                fallback_reason=layer2,
                raw_response=raw,
            ),
            layer2,
        )

    anchor_pass_map = _layer3_anchor_consistency(
        raw, rubric_principles=rubric_principles
    )

    layer4 = _layer4_hard_reject(raw, hard_reject_patterns=hard_reject_patterns)
    if layer4 is not None:
        principles = _principles_from_raw(raw, anchor_pass_map=anchor_pass_map)
        cost = _estimate_cost(image_count=len(asset_references))
        return (
            VisionEvaluationResult(
                judgment=VisualJudgment(
                    model=model,
                    principles=principles,
                    overall_verdict="no",
                    overall_confidence=1.0,
                    fallback_reason=layer4,
                    cost_estimate_usd=cost,
                ),
                asset_references=asset_references,
                raw_response=raw,
            ),
            None,  # layer 4 is a successful policy outcome — no escalation
        )

    principles = _principles_from_raw(raw, anchor_pass_map=anchor_pass_map)
    cost = _estimate_cost(image_count=len(asset_references))
    return (
        VisionEvaluationResult(
            judgment=VisualJudgment(
                model=model,
                principles=principles,
                overall_verdict=str(raw.get("overall_verdict")),
                overall_confidence=float(raw.get("overall_confidence", 0.0)),
                fallback_reason="",
                cost_estimate_usd=cost,
            ),
            asset_references=asset_references,
            raw_response=raw,
        ),
        None,
    )


def _principles_from_raw(
    raw: dict[str, Any], *, anchor_pass_map: dict[str, bool]
) -> tuple[VisualJudgmentPrinciple, ...]:
    out: list[VisualJudgmentPrinciple] = []
    for principle in raw.get("principles") or []:
        if not isinstance(principle, dict):
            continue
        name = str(principle.get("name") or "")
        score = int(principle.get("score") or 0)
        anchor = SCORE_TO_ANCHOR.get(score, "okay")
        reasoning = str(principle.get("reasoning") or "")
        image_ids_raw = principle.get("image_ids") or []
        image_ids = tuple(
            int(i) for i in image_ids_raw if isinstance(i, int) and i >= 0
        )
        out.append(
            VisualJudgmentPrinciple(
                name=name,
                score=score,
                anchor=anchor,
                reasoning=reasoning,
                image_ids=image_ids,
                anchor_consistency_pass=anchor_pass_map.get(name, True),
            )
        )
    return tuple(out)


def _fallback_result(
    *,
    model: str,
    asset_references: tuple[_AssetReference, ...],
    fallback_reason: str,
    raw_response: dict[str, Any] | None = None,
) -> VisionEvaluationResult:
    """Build the fallback :class:`VisionEvaluationResult` when a guard fires."""

    return VisionEvaluationResult(
        judgment=VisualJudgment(
            model=model,
            principles=(),
            overall_verdict="borderline",
            overall_confidence=0.0,
            fallback_reason=fallback_reason,
            cost_estimate_usd=_estimate_cost(image_count=len(asset_references)),
        ),
        asset_references=asset_references,
        raw_response=raw_response or {},
    )


def _estimate_cost(*, image_count: int) -> float:
    """Approximate USD cost for one Gemini Pro vision call.

    Estimate is dominated by image tokens (~258 per image at 1024px);
    output tokens (~500 typical) add a small fixed amount. System
    prompt token cost varies with rubric size; treated as ~1500
    tokens for the estimate.
    """

    estimated_input_tokens = image_count * GEMINI_3_1_PRO_TOKENS_PER_IMAGE + 1500
    estimated_output_tokens = 500

    input_cost = estimated_input_tokens * GEMINI_3_1_PRO_INPUT_USD_PER_1M_TOKENS / 1_000_000
    output_cost = estimated_output_tokens * GEMINI_3_1_PRO_OUTPUT_USD_PER_1M_TOKENS / 1_000_000
    return round(input_cost + output_cost, 5)


def _estimate_claude_cost(*, image_count: int) -> float:
    """Approximate USD cost for one Claude Sonnet 4.6 cross-check call.

    ~10x Gemini cost differential per spec §4.1; only run on top-decile.
    """

    estimated_input_tokens = image_count * CLAUDE_SONNET_4_6_TOKENS_PER_IMAGE + 1500
    estimated_output_tokens = 500
    input_cost = (
        estimated_input_tokens
        * CLAUDE_SONNET_4_6_INPUT_USD_PER_1M_TOKENS
        / 1_000_000
    )
    output_cost = (
        estimated_output_tokens
        * CLAUDE_SONNET_4_6_OUTPUT_USD_PER_1M_TOKENS
        / 1_000_000
    )
    return round(input_cost + output_cost, 5)


# ---------------------------------------------------------------------------
# Slice 8: Sonnet 4.6 cross-check on top-decile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossCheckCandidate:
    """One top-decile candidate selected for the Sonnet cross-check pass.

    Caller passes a list of these to :func:`run_sonnet_cross_check_pass`
    along with an injectable LLM call. The selection is pure
    (:func:`select_top_decile_for_cross_check`) so the orchestrator
    can replay it deterministically across resumes.
    """

    candidate_identity_key: str
    primary_judgment: VisualJudgment
    asset_references: tuple[_AssetReference, ...]


@dataclass(frozen=True)
class PrincipleDisagreement:
    """One per-principle disagreement between Gemini and Sonnet.

    Surfaced in the workspace cross-check disagreement marker
    (rendered by :file:`VisualReviewBeforeAfter.svelte` when the
    `crossCheck` prop is non-null).
    """

    principle_name: str
    primary_score: int
    primary_anchor: str
    cross_check_score: int
    cross_check_anchor: str
    cross_check_reasoning: str
    delta: int


def select_top_decile_for_cross_check(
    candidates: list[tuple[str, VisualJudgment]],
    *,
    fraction: float = DEFAULT_CROSS_CHECK_TOP_DECILE_FRACTION,
    min_count: int = DEFAULT_MIN_CROSS_CHECK_COUNT,
    max_count: int = DEFAULT_MAX_CROSS_CHECK_COUNT,
) -> list[tuple[str, VisualJudgment]]:
    """Pick the top-decile candidates for the Sonnet cross-check.

    Rank by ``overall_confidence × _verdict_score(overall_verdict)``
    so a high-confidence "yes" outranks a high-confidence
    "borderline" outranks a high-confidence "no". Take the top
    ``fraction`` of the ranked list, bounded by ``[min_count, max_count]``.

    Pure function — passed (candidate_identity_key, judgment) tuples
    in, returns the same shape ranked. No I/O.
    """

    rankable = [
        (key, judgment)
        for (key, judgment) in candidates
        if judgment.fallback_reason == ""
        and judgment.overall_verdict in VERDICT_VALUES
    ]
    rankable.sort(
        key=lambda item: (
            item[1].overall_confidence * _verdict_score(item[1].overall_verdict)
        ),
        reverse=True,
    )

    if not rankable:
        return []
    target = max(min_count, min(max_count, int(len(rankable) * fraction)))
    return rankable[:target]


def _verdict_score(verdict: str) -> float:
    """Map verdict to a sortable scalar so confidence × verdict ranks
    cleanly. yes > borderline > no."""

    return {"yes": 1.0, "borderline": 0.5, "no": 0.0}.get(verdict, 0.0)


def detect_principle_disagreements(
    *,
    primary: VisualJudgment,
    cross_check: VisualJudgment,
    delta_threshold: int = CROSS_CHECK_DISAGREEMENT_ANCHOR_DELTA,
) -> list[PrincipleDisagreement]:
    """Return per-principle disagreements between primary and cross-check.

    A principle disagrees when the absolute score delta between the two
    models is ``≥ delta_threshold`` (default: 2 — off-by-one is OK
    per spec §4.3). Returns an empty list when models agree.
    """

    primary_lookup = {p.name: p for p in primary.principles}
    out: list[PrincipleDisagreement] = []
    for cross_principle in cross_check.principles:
        primary_principle = primary_lookup.get(cross_principle.name)
        if primary_principle is None:
            continue
        delta = abs(primary_principle.score - cross_principle.score)
        if delta >= delta_threshold:
            out.append(
                PrincipleDisagreement(
                    principle_name=cross_principle.name,
                    primary_score=primary_principle.score,
                    primary_anchor=primary_principle.anchor,
                    cross_check_score=cross_principle.score,
                    cross_check_anchor=cross_principle.anchor,
                    cross_check_reasoning=cross_principle.reasoning,
                    delta=delta,
                )
            )
    return out


def run_sonnet_cross_check_pass(
    *,
    brief: dict[str, Any],
    selected_candidates: list[CrossCheckCandidate],
    image_bytes_lookup: dict[str, list[bytes]],
    vision_llm_call: VisionLLMCall = gemini_vision_llm_call,
    model: str = CLAUDE_SONNET_4_6_MODEL_NAME,
) -> dict[str, VisionEvaluationResult]:
    """Run the cross-check pass over the selected candidates.

    Returns ``{candidate_identity_key: VisionEvaluationResult}`` so
    the orchestrator can stitch each cross-check verdict into its
    candidate's ``visual_judgment.cross_check`` payload.

    The ``vision_llm_call`` parameter accepts the same callable shape
    as the primary pass — production wires it to a Sonnet 4.6 client
    (Slice 8 ships the abstraction; the client implementation can
    be a thin wrapper over Anthropic's vision-capable
    `messages.create` endpoint, mirroring
    :func:`gemini_vision_llm_call`'s structure). Tests inject a
    deterministic fake.

    The cross-check uses the SAME system prompt as the primary pass —
    the comparison is meaningful only when both models read the same
    rubric in the same words. Slice 8 explicitly does NOT special-
    case the prompt for Sonnet.
    """

    out: dict[str, VisionEvaluationResult] = {}

    for cross in selected_candidates:
        image_bytes = image_bytes_lookup.get(cross.candidate_identity_key, [])
        asset_metadata = [
            (ref.asset_url, ref.source, ref.project_title)
            for ref in cross.asset_references
        ]
        result = evaluate_designer_visually(
            brief=brief,
            candidate_display_name=cross.candidate_identity_key,
            candidate_headline="",
            image_bytes_list=image_bytes,
            asset_metadata=asset_metadata,
            vision_llm_call=vision_llm_call,
            model=model,
        )
        # Override the cost estimate to reflect Sonnet pricing.
        cross_check_judgment = VisualJudgment(
            model=result.judgment.model,
            principles=result.judgment.principles,
            overall_verdict=result.judgment.overall_verdict,
            overall_confidence=result.judgment.overall_confidence,
            fallback_reason=result.judgment.fallback_reason,
            cost_estimate_usd=_estimate_claude_cost(
                image_count=len(cross.asset_references)
            ),
        )
        out[cross.candidate_identity_key] = VisionEvaluationResult(
            judgment=cross_check_judgment,
            asset_references=result.asset_references,
            raw_response=result.raw_response,
        )

    return out


def attach_cross_check_to_judgment(
    *,
    primary: VisualJudgment,
    cross_check: VisualJudgment,
) -> VisualJudgment:
    """Return a copy of ``primary`` with ``cross_check`` attached.

    The ``cross_check`` field on :class:`VisualJudgment` is a dict
    (not a typed dataclass) so the wire payload to
    ``terminal_payload_json`` round-trips through JSON cleanly.
    """

    cross_check_payload = {
        "model": cross_check.model,
        "principles": [
            {
                "name": p.name,
                "score": p.score,
                "anchor": p.anchor,
                "reasoning": p.reasoning,
                "image_ids": list(p.image_ids),
            }
            for p in cross_check.principles
        ],
        "overall_verdict": cross_check.overall_verdict,
        "overall_confidence": cross_check.overall_confidence,
    }
    return VisualJudgment(
        model=primary.model,
        principles=primary.principles,
        overall_verdict=primary.overall_verdict,
        overall_confidence=primary.overall_confidence,
        fallback_reason=primary.fallback_reason,
        cost_estimate_usd=primary.cost_estimate_usd + cross_check.cost_estimate_usd,
        cross_check=cross_check_payload,
    )
