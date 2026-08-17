"""Identity-safe forced-tool contracts for LinkedIn V2 judgments.

The model-provided ``candidate_id`` is never trusted positionally.  Callers
generate an opaque, per-request ID for each candidate, validate the returned
set exactly, then restore the original input order.  These contracts are
LinkedIn V2-only; legacy briefs and the other source adapters retain their
existing response formats.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import re
import secrets
from typing import Any, Iterable, Mapping, Sequence

from shared.contracts import (
    CALIBER_VALUES,
    EVIDENCE_RECENCY_VALUES,
    FAILURE_DECISIONS,
    FULL_DECISIONS,
    LEVEL_ALIGNMENT_VALUES,
    OPPORTUNITY_COHERENCE_VALUES,
    OUTREACH_TIERS,
    REJECT_REASON_CODES,
    REVIEW_REASON_CODES,
    SAVE_DECISIONS,
)
from shared.llm_policy import FireworksToolContract


FACIAL_TOOL_NAME = "submit_linkedin_facial_judgments_v1"
FULL_TOOL_NAME = "submit_linkedin_full_evaluation_v2"
FACIAL_CONTRACT_VERSION = "linkedin_facial_tool_v1"
FULL_CONTRACT_VERSION = "linkedin_full_tool_v2"

_FACIAL_BINARY_DECISIONS = frozenset({"FACIAL_YES", "FACIAL_NO"})
_FACIAL_TERNARY_DECISIONS = _FACIAL_BINARY_DECISIONS | {
    "FACIAL_BORDERLINE"
}
_MODEL_FULL_DECISIONS = frozenset(FULL_DECISIONS) - frozenset(FAILURE_DECISIONS)
_MATCH_TYPES = frozenset({"DIRECT", "ADJACENT", "NONE"})
_DEPTH_TYPES = frozenset({"BUILDER", "USER", "UNKNOWN"})
_TRANSFERABILITY_TYPES = frozenset(
    {"TRANSFERABLE", "NOT_TRANSFERABLE", "N/A"}
)
_CONTROL_DELIMITER_RE = re.compile(
    r"</?(?:UNTRUSTED_CANDIDATE_DATA|UNTRUSTED_EXTERNAL_EVIDENCE|candidate_profile)>",
    re.IGNORECASE,
)


class JudgmentToolContractError(ValueError):
    """A model tool payload failed the local attribution/schema contract."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        message = reason if not detail else f"{reason}: {detail}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class FacialToolResult:
    candidate_id: str
    decision: str
    reason: str


@dataclass(frozen=True, slots=True)
class FullToolResult:
    candidate_id: str
    decision: str
    match_type: str
    capability_area: str | None
    capability_evidence: str
    depth: str
    depth_evidence: str
    transferability: str
    transferability_evidence: str
    evidence_recency: str
    level_alignment: str
    opportunity_coherence: str
    caliber: str
    outreach_tier: str | None
    reject_reason: str | None
    case_for: str
    case_against: str
    confidence: float
    post_save_modifier: str
    review_reason_code: str
    review_structural_evidence: tuple[str, ...]
    review_recommended_next_step: str
    summary: str


@dataclass(frozen=True, slots=True)
class FullEvaluationSemantics:
    """Normalized fields shared by the tool and legacy-text validators."""

    decision: str
    match_type: str
    capability_area: str | None
    depth: str
    transferability: str
    evidence_recency: str
    level_alignment: str
    opportunity_coherence: str
    caliber: str
    outreach_tier: str | None
    reject_reason: str | None
    confidence: float
    post_save_modifier: str
    review_reason_code: str
    review_structural_evidence: tuple[str, ...]
    review_recommended_next_step: str


def generate_opaque_candidate_ids(count: int) -> tuple[str, ...]:
    """Return non-PII, request-local candidate IDs.

    IDs intentionally contain neither source position nor a hash of candidate
    identity.  The mapping exists only in the caller for the life of one
    logical judgment request.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    ids: list[str] = []
    seen: set[str] = set()
    while len(ids) < count:
        candidate_id = f"cand_{secrets.token_hex(12)}"
        if candidate_id in seen:  # pragma: no cover - cryptographic collision guard
            continue
        seen.add(candidate_id)
        ids.append(candidate_id)
    return tuple(ids)


def _neutralize_control_delimiters(value: str) -> str:
    """Prevent scraped/model-sourced text from closing trusted boundaries."""

    return _CONTROL_DELIMITER_RE.sub(
        lambda match: "[escaped-delimiter:" + match.group(0)[1:-1] + "]",
        str(value),
    )


def render_facial_tool_user_message(
    candidate_texts: Sequence[str],
    candidate_ids: Sequence[str],
    *,
    prompt_prefix: str = "",
) -> str:
    if len(candidate_texts) != len(candidate_ids):
        raise ValueError("candidate text/id counts must match")
    records = []
    for candidate_id, candidate_text in zip(candidate_ids, candidate_texts):
        safe_candidate_text = _neutralize_control_delimiters(candidate_text)
        records.append(
            "\n".join(
                (
                    f"CANDIDATE_ID: {candidate_id}",
                    "<UNTRUSTED_CANDIDATE_DATA>",
                    safe_candidate_text,
                    "</UNTRUSTED_CANDIDATE_DATA>",
                )
            )
        )
    body = "\n\n".join(records)
    return f"{prompt_prefix}{body}" if prompt_prefix else body


def render_full_tool_user_message(
    candidate_text: str,
    candidate_id: str,
    *,
    external_evidence_block: str = "",
) -> str:
    safe_candidate_text = _neutralize_control_delimiters(candidate_text)
    parts = [
        f"CANDIDATE_ID: {candidate_id}",
        "<UNTRUSTED_CANDIDATE_DATA>",
        safe_candidate_text,
        "</UNTRUSTED_CANDIDATE_DATA>",
    ]
    if external_evidence_block:
        parts.extend(
            (
                "",
                "<UNTRUSTED_EXTERNAL_EVIDENCE>",
                _neutralize_control_delimiters(external_evidence_block),
                "</UNTRUSTED_EXTERNAL_EVIDENCE>",
            )
        )
    return "\n".join(parts)


def facial_tool_contract(*, allow_borderline: bool) -> FireworksToolContract:
    decisions = sorted(
        _FACIAL_TERNARY_DECISIONS
        if allow_borderline
        else _FACIAL_BINARY_DECISIONS
    )
    return FireworksToolContract(
        name=FACIAL_TOOL_NAME,
        description=(
            "Submit exactly one LinkedIn facial-triage judgment for every "
            "candidate ID supplied by the application. Copy each ID exactly."
        ),
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 25,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "candidate_id": {
                                "type": "string",
                                "pattern": r"^cand_[0-9a-f]{24}$",
                            },
                            "decision": {"type": "string", "enum": decisions},
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": ["candidate_id", "decision", "reason"],
                    },
                }
            },
            "required": ["results"],
        },
        strict=True,
    )


def full_tool_contract(
    *,
    capability_areas: Iterable[str],
    post_save_modifiers: Iterable[str],
) -> FireworksToolContract:
    exact_capability_areas = list(
        dict.fromkeys(
            str(area).strip()
            for area in capability_areas
            if str(area).strip()
        )
    )
    if not exact_capability_areas:
        raise ValueError("full tool contract requires capability-area names")
    exact_post_save_modifiers = list(
        dict.fromkeys(
            str(modifier).strip()
            for modifier in post_save_modifiers
            if str(modifier).strip() and str(modifier).strip() != "NONE"
        )
    )
    base_contract = FireworksToolContract(
        name=FULL_TOOL_NAME,
        description=(
            "Submit the terminal structured LinkedIn full-profile evaluation "
            "for the exact candidate ID supplied by the application."
        ),
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "pattern": r"^cand_[0-9a-f]{24}$",
                },
                "decision": {
                    "type": "string",
                    "enum": sorted(_MODEL_FULL_DECISIONS),
                },
                "match_type": {"type": "string", "enum": sorted(_MATCH_TYPES)},
                "capability_area": {
                    "type": ["string", "null"],
                    "enum": [None, *exact_capability_areas],
                    "description": (
                        "Exact brief capability-area name for DIRECT or ADJACENT; "
                        "JSON null when match_type is NONE."
                    ),
                },
                "capability_evidence": {"type": "string", "minLength": 1},
                "depth": {"type": "string", "enum": sorted(_DEPTH_TYPES)},
                "depth_evidence": {"type": "string", "minLength": 1},
                "transferability": {
                    "type": "string",
                    "enum": sorted(_TRANSFERABILITY_TYPES),
                },
                "transferability_evidence": {"type": "string", "minLength": 1},
                "evidence_recency": {
                    "type": "string",
                    "enum": sorted(EVIDENCE_RECENCY_VALUES),
                },
                "level_alignment": {
                    "type": "string",
                    "enum": sorted(LEVEL_ALIGNMENT_VALUES),
                },
                "opportunity_coherence": {
                    "type": "string",
                    "enum": sorted(OPPORTUNITY_COHERENCE_VALUES),
                },
                "caliber": {
                    "type": "string",
                    "enum": sorted(CALIBER_VALUES),
                },
                "outreach_tier": {
                    "type": ["string", "null"],
                    "enum": [None, *sorted(OUTREACH_TIERS)],
                },
                "reject_reason": {
                    "type": ["string", "null"],
                    "enum": [None, *sorted(REJECT_REASON_CODES)],
                },
                "case_for": {"type": "string", "minLength": 1},
                "case_against": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "post_save_modifier": {
                    "type": "string",
                    "enum": ["NONE", *exact_post_save_modifiers],
                    "minLength": 1,
                    "description": (
                        "Exact named modifier that fired, otherwise the string NONE."
                    ),
                },
                "review_reason_code": {
                    "type": ["string", "null"],
                    "enum": [None, *sorted(REVIEW_REASON_CODES)],
                    "description": (
                        "A bounded reason code for a REVIEW decision; JSON null for "
                        "every non-review decision."
                    ),
                },
                "review_structural_evidence": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "description": (
                        "At least two signals for REVIEW_INFERRED; otherwise an empty array."
                    ),
                },
                "review_recommended_next_step": {
                    "type": ["string", "null"],
                    "description": (
                        "Concrete next step for REVIEW_FLAGGED; JSON null otherwise."
                    ),
                },
                "summary": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Recruiter-readable summary, or the complete dossier rationale "
                        "when the rubric requests dossier output."
                    ),
                },
            },
            "required": [
                "candidate_id",
                "decision",
                "match_type",
                "capability_area",
                "capability_evidence",
                "depth",
                "depth_evidence",
                "transferability",
                "transferability_evidence",
                "evidence_recency",
                "level_alignment",
                "opportunity_coherence",
                "caliber",
                "outreach_tier",
                "reject_reason",
                "case_for",
                "case_against",
                "confidence",
                "post_save_modifier",
                "review_reason_code",
                "review_structural_evidence",
                "review_recommended_next_step",
                "summary",
            ],
        },
        strict=True,
    )

    root_properties = base_contract.parameters["properties"]
    root_required = base_contract.parameters["required"]

    def complete_object_branch(
        property_constraints: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        branch_properties = copy.deepcopy(root_properties)
        for field, constraint in property_constraints.items():
            branch_properties[field] = copy.deepcopy(dict(constraint))
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": branch_properties,
            "required": copy.deepcopy(root_required),
        }

    non_direct_transferability = ["NOT_TRANSFERABLE", "TRANSFERABLE"]
    null_review_fields = {
        "review_reason_code": {"type": "null", "enum": [None]},
        "review_structural_evidence": {"type": "array", "enum": [[]]},
        "review_recommended_next_step": {"type": "null", "enum": [None]},
    }
    null_currency_fields = {
        "outreach_tier": {"type": "null", "enum": [None]},
        "reject_reason": {"type": "null", "enum": [None]},
    }
    parameters = copy.deepcopy(dict(base_contract.parameters))
    parameters["allOf"] = [
        {
            "anyOf": [
                complete_object_branch(
                    {
                        "match_type": {"type": "string", "enum": ["DIRECT"]},
                        "capability_area": {
                            "type": "string",
                            "enum": exact_capability_areas,
                        },
                        "transferability": {
                            "type": "string",
                            "enum": ["N/A"],
                        },
                    }
                ),
                complete_object_branch(
                    {
                        "match_type": {"type": "string", "enum": ["ADJACENT"]},
                        "capability_area": {
                            "type": "string",
                            "enum": exact_capability_areas,
                        },
                        "transferability": {
                            "type": "string",
                            "enum": non_direct_transferability,
                        },
                    }
                ),
                complete_object_branch(
                    {
                        "match_type": {"type": "string", "enum": ["NONE"]},
                        "capability_area": {"type": "null", "enum": [None]},
                        "transferability": {
                            "type": "string",
                            "enum": non_direct_transferability,
                        },
                    }
                ),
            ]
        },
        {
            "anyOf": [
                complete_object_branch(
                    {
                        "decision": {
                            "type": "string",
                            "enum": sorted(SAVE_DECISIONS - {"INFERENTIAL_SAVE"}),
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["BUILDER", "UNKNOWN"],
                        },
                        "transferability": {
                            "type": "string",
                            "enum": ["N/A", "TRANSFERABLE"],
                        },
                        "post_save_modifier": {
                            "type": "string",
                            "enum": ["NONE", *exact_post_save_modifiers],
                        },
                        "outreach_tier": {
                            "type": "string",
                            "enum": sorted(OUTREACH_TIERS),
                        },
                        "reject_reason": {"type": "null", "enum": [None]},
                        **null_review_fields,
                    }
                ),
                complete_object_branch(
                    {
                        "decision": {
                            "type": "string",
                            "enum": ["INFERENTIAL_SAVE"],
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["BUILDER", "UNKNOWN"],
                        },
                        "transferability": {
                            "type": "string",
                            "enum": ["N/A", "TRANSFERABLE"],
                        },
                        "post_save_modifier": {
                            "type": "string",
                            "enum": ["NONE", *exact_post_save_modifiers],
                        },
                        "outreach_tier": {
                            "type": "string",
                            "enum": ["STANDARD"],
                        },
                        "reject_reason": {"type": "null", "enum": [None]},
                        **null_review_fields,
                    }
                ),
                complete_object_branch(
                    {
                        "decision": {"type": "string", "enum": ["REJECT"]},
                        "post_save_modifier": {
                            "type": "string",
                            "enum": ["NONE"],
                        },
                        "outreach_tier": {"type": "null", "enum": [None]},
                        "reject_reason": {
                            "type": "string",
                            "enum": sorted(REJECT_REASON_CODES),
                        },
                        **null_review_fields,
                    }
                ),
                complete_object_branch(
                    {
                        "decision": {
                            "type": "string",
                            "enum": ["REVIEW_INFERRED"],
                        },
                        "post_save_modifier": {
                            "type": "string",
                            "enum": ["NONE"],
                        },
                        "review_reason_code": {
                            "type": "string",
                            "enum": sorted(REVIEW_REASON_CODES),
                        },
                        "review_recommended_next_step": {
                            "type": "null",
                            "enum": [None],
                        },
                        **null_currency_fields,
                    }
                ),
                complete_object_branch(
                    {
                        "decision": {
                            "type": "string",
                            "enum": ["REVIEW_FLAGGED"],
                        },
                        "post_save_modifier": {
                            "type": "string",
                            "enum": ["NONE"],
                        },
                        "review_reason_code": {
                            "type": "string",
                            "enum": sorted(REVIEW_REASON_CODES),
                        },
                        "review_structural_evidence": {
                            "type": "array",
                            "enum": [[]],
                        },
                        "review_recommended_next_step": {"type": "string"},
                        **null_currency_fields,
                    }
                ),
            ]
        },
        {
            "anyOf": [
                complete_object_branch(
                    {
                        "outreach_tier": {
                            "type": "string",
                            "enum": ["PRIORITY"],
                        },
                        "match_type": {"type": "string", "enum": ["DIRECT"]},
                        "evidence_recency": {
                            "type": "string",
                            "enum": ["CURRENT"],
                        },
                        "caliber": {"type": "string", "enum": ["STRONG"]},
                    }
                ),
                complete_object_branch(
                    {
                        "outreach_tier": {
                            "type": "string",
                            "enum": ["STANDARD"],
                        }
                    }
                ),
                complete_object_branch(
                    {"outreach_tier": {"type": "null", "enum": [None]}}
                ),
            ]
        },
    ]
    return FireworksToolContract(
        name=base_contract.name,
        description=base_contract.description,
        parameters=parameters,
        strict=base_contract.strict,
    )


def validate_facial_tool_arguments(
    arguments: Mapping[str, Any],
    *,
    expected_ids: Sequence[str],
    allow_borderline: bool,
) -> tuple[FacialToolResult, ...]:
    _require_exact_keys(arguments, {"results"}, context="facial arguments")
    raw_results = arguments.get("results")
    if not isinstance(raw_results, list):
        raise JudgmentToolContractError("results_not_array")
    if len(raw_results) != len(expected_ids):
        raise JudgmentToolContractError(
            "cardinality_mismatch",
            f"expected={len(expected_ids)} actual={len(raw_results)}",
        )

    allowed = (
        _FACIAL_TERNARY_DECISIONS
        if allow_borderline
        else _FACIAL_BINARY_DECISIONS
    )
    by_id: dict[str, FacialToolResult] = {}
    expected_set = set(expected_ids)
    for index, raw in enumerate(raw_results):
        if not isinstance(raw, Mapping):
            raise JudgmentToolContractError("result_not_object", f"index={index}")
        _require_exact_keys(
            raw,
            {"candidate_id", "decision", "reason"},
            context=f"facial result {index}",
        )
        candidate_id = _required_string(raw.get("candidate_id"), "candidate_id")
        if candidate_id not in expected_set:
            raise JudgmentToolContractError("unknown_candidate_id", candidate_id)
        if candidate_id in by_id:
            raise JudgmentToolContractError("duplicate_candidate_id", candidate_id)
        decision = _required_string(raw.get("decision"), "decision").upper()
        if decision not in allowed:
            raise JudgmentToolContractError("invalid_facial_decision", decision)
        reason = _required_string(raw.get("reason"), "reason")
        by_id[candidate_id] = FacialToolResult(candidate_id, decision, reason)

    missing = [candidate_id for candidate_id in expected_ids if candidate_id not in by_id]
    if missing:
        raise JudgmentToolContractError("missing_candidate_ids", ",".join(missing))
    return tuple(by_id[candidate_id] for candidate_id in expected_ids)


def validate_full_evaluation_semantics(
    *,
    decision: object,
    match_type: object,
    capability_area: object,
    depth: object,
    transferability: object,
    evidence_recency: object,
    level_alignment: object,
    opportunity_coherence: object,
    caliber: object,
    outreach_tier: object,
    reject_reason: object,
    confidence: object,
    post_save_modifier: object,
    review_reason_code: object,
    review_structural_evidence: object,
    review_recommended_next_step: object,
    capability_areas: Iterable[str],
    post_save_modifiers: Iterable[str],
) -> FullEvaluationSemantics:
    """Validate normalized semantics shared by tool and legacy v2 paths.

    Callers remain responsible for transport-specific shape checks and prose
    evidence fields. Nullable currency fields must already be normalized to
    ``None``; the legacy parser maps its ``NONE``/``N/A`` wire tokens first.
    """

    normalized_decision = _required_string(decision, "decision").upper()
    if normalized_decision not in _MODEL_FULL_DECISIONS:
        raise JudgmentToolContractError(
            "invalid_full_decision", normalized_decision
        )
    normalized_match = _enum_string(match_type, "match_type", _MATCH_TYPES)
    normalized_depth = _enum_string(depth, "depth", _DEPTH_TYPES)
    normalized_transferability = _enum_string(
        transferability,
        "transferability",
        _TRANSFERABILITY_TYPES,
    )
    normalized_recency = _enum_string(
        evidence_recency,
        "evidence_recency",
        EVIDENCE_RECENCY_VALUES,
    )
    normalized_level = _enum_string(
        level_alignment,
        "level_alignment",
        LEVEL_ALIGNMENT_VALUES,
    )
    normalized_coherence = _enum_string(
        opportunity_coherence,
        "opportunity_coherence",
        OPPORTUNITY_COHERENCE_VALUES,
    )
    normalized_caliber = _enum_string(caliber, "caliber", CALIBER_VALUES)
    normalized_tier = _nullable_enum_string(
        outreach_tier,
        "outreach_tier",
        OUTREACH_TIERS,
    )
    normalized_reject_reason = _nullable_enum_string(
        reject_reason,
        "reject_reason",
        REJECT_REASON_CODES,
    )

    if capability_area is None:
        normalized_area = None
    else:
        normalized_area = _required_string(capability_area, "capability_area")
    known_areas = {
        str(area).strip() for area in capability_areas if str(area).strip()
    }
    if normalized_match in {"DIRECT", "ADJACENT"}:
        if normalized_area not in known_areas:
            raise JudgmentToolContractError(
                "invalid_capability_area", str(normalized_area)
            )
    elif normalized_area is not None:
        raise JudgmentToolContractError(
            "capability_area_for_none_match", normalized_area
        )
    if normalized_match == "DIRECT" and normalized_transferability != "N/A":
        raise JudgmentToolContractError(
            "invalid_direct_transferability", normalized_transferability
        )
    if normalized_match != "DIRECT" and normalized_transferability == "N/A":
        raise JudgmentToolContractError(
            "missing_non_direct_transferability", normalized_match
        )
    if normalized_decision in SAVE_DECISIONS:
        if normalized_depth == "USER":
            raise JudgmentToolContractError(
                "save_with_user_depth", normalized_decision
            )
        if normalized_transferability == "NOT_TRANSFERABLE":
            raise JudgmentToolContractError(
                "save_without_transferable_path", normalized_decision
            )

    if normalized_decision == "REJECT":
        if normalized_reject_reason is None:
            raise JudgmentToolContractError("reject_requires_reason")
        if normalized_tier is not None:
            raise JudgmentToolContractError("reject_forbids_tier")
    elif normalized_decision in SAVE_DECISIONS:
        if normalized_tier is None:
            raise JudgmentToolContractError("save_requires_tier")
        if normalized_reject_reason is not None:
            raise JudgmentToolContractError("save_forbids_reject_reason")
        if (
            normalized_decision == "INFERENTIAL_SAVE"
            and normalized_tier != "STANDARD"
        ):
            raise JudgmentToolContractError(
                "inferential_save_requires_standard"
            )
        if normalized_tier == "PRIORITY" and not (
            normalized_match == "DIRECT"
            and normalized_recency == "CURRENT"
            and normalized_caliber == "STRONG"
        ):
            raise JudgmentToolContractError(
                "priority_requires_strong_direct_current"
            )
    elif normalized_tier is not None or normalized_reject_reason is not None:
        raise JudgmentToolContractError("review_forbids_tier_or_reject_reason")

    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise JudgmentToolContractError("confidence_not_number")
    normalized_confidence = float(confidence)
    if not 0.0 <= normalized_confidence <= 1.0:
        raise JudgmentToolContractError(
            "confidence_out_of_range", str(normalized_confidence)
        )

    normalized_modifier = _required_string(
        post_save_modifier, "post_save_modifier"
    )
    allowed_modifiers = {
        str(modifier).strip()
        for modifier in post_save_modifiers
        if str(modifier).strip()
    }
    allowed_modifiers.add("NONE")
    if normalized_modifier not in allowed_modifiers:
        raise JudgmentToolContractError(
            "invalid_post_save_modifier", normalized_modifier
        )
    if (
        normalized_decision not in SAVE_DECISIONS
        and normalized_modifier != "NONE"
    ):
        raise JudgmentToolContractError(
            "modifier_on_non_save", normalized_modifier
        )

    normalized_review_reason = _nullable_string(
        review_reason_code, "review_reason_code"
    ).lower()
    if not isinstance(review_structural_evidence, (list, tuple)):
        raise JudgmentToolContractError(
            "review_structural_evidence_not_array"
        )
    normalized_review_evidence = tuple(
        _required_string(item, "review_structural_evidence item")
        for item in review_structural_evidence
    )
    normalized_next_step = _nullable_string(
        review_recommended_next_step,
        "review_recommended_next_step",
    )
    if normalized_decision in {"REVIEW_INFERRED", "REVIEW_FLAGGED"}:
        if normalized_review_reason not in REVIEW_REASON_CODES:
            raise JudgmentToolContractError(
                "invalid_review_reason_code", normalized_review_reason
            )
        if normalized_decision == "REVIEW_INFERRED":
            if len(normalized_review_evidence) < 2:
                raise JudgmentToolContractError(
                    "insufficient_structural_evidence"
                )
            if normalized_next_step:
                raise JudgmentToolContractError(
                    "recommended_next_step_on_inferred_review"
                )
        else:
            if normalized_review_evidence:
                raise JudgmentToolContractError(
                    "structural_evidence_on_flagged_review"
                )
            if not normalized_next_step:
                raise JudgmentToolContractError(
                    "missing_recommended_next_step"
                )
    elif (
        normalized_review_reason
        or normalized_review_evidence
        or normalized_next_step
    ):
        raise JudgmentToolContractError("review_fields_on_non_review")

    return FullEvaluationSemantics(
        decision=normalized_decision,
        match_type=normalized_match,
        capability_area=normalized_area,
        depth=normalized_depth,
        transferability=normalized_transferability,
        evidence_recency=normalized_recency,
        level_alignment=normalized_level,
        opportunity_coherence=normalized_coherence,
        caliber=normalized_caliber,
        outreach_tier=normalized_tier,
        reject_reason=normalized_reject_reason,
        confidence=normalized_confidence,
        post_save_modifier=normalized_modifier,
        review_reason_code=normalized_review_reason,
        review_structural_evidence=normalized_review_evidence,
        review_recommended_next_step=normalized_next_step,
    )


def validate_full_tool_arguments(
    arguments: Mapping[str, Any],
    *,
    expected_id: str,
    capability_areas: Iterable[str],
    post_save_modifiers: Iterable[str],
) -> FullToolResult:
    required = {
        "candidate_id",
        "decision",
        "match_type",
        "capability_area",
        "capability_evidence",
        "depth",
        "depth_evidence",
        "transferability",
        "transferability_evidence",
        "evidence_recency",
        "level_alignment",
        "opportunity_coherence",
        "caliber",
        "outreach_tier",
        "reject_reason",
        "case_for",
        "case_against",
        "confidence",
        "post_save_modifier",
        "review_reason_code",
        "review_structural_evidence",
        "review_recommended_next_step",
        "summary",
    }
    _require_exact_keys(arguments, required, context="full arguments")

    candidate_id = _required_string(arguments.get("candidate_id"), "candidate_id")
    if candidate_id != expected_id:
        raise JudgmentToolContractError(
            "candidate_id_mismatch", f"expected={expected_id} actual={candidate_id}"
        )
    semantics = validate_full_evaluation_semantics(
        decision=arguments.get("decision"),
        match_type=arguments.get("match_type"),
        capability_area=arguments.get("capability_area"),
        depth=arguments.get("depth"),
        transferability=arguments.get("transferability"),
        evidence_recency=arguments.get("evidence_recency"),
        level_alignment=arguments.get("level_alignment"),
        opportunity_coherence=arguments.get("opportunity_coherence"),
        caliber=arguments.get("caliber"),
        outreach_tier=arguments.get("outreach_tier"),
        reject_reason=arguments.get("reject_reason"),
        confidence=arguments.get("confidence"),
        post_save_modifier=arguments.get("post_save_modifier"),
        review_reason_code=arguments.get("review_reason_code"),
        review_structural_evidence=arguments.get("review_structural_evidence"),
        review_recommended_next_step=arguments.get(
            "review_recommended_next_step"
        ),
        capability_areas=capability_areas,
        post_save_modifiers=post_save_modifiers,
    )

    return FullToolResult(
        candidate_id=candidate_id,
        decision=semantics.decision,
        match_type=semantics.match_type,
        capability_area=semantics.capability_area,
        capability_evidence=_required_string(
            arguments.get("capability_evidence"), "capability_evidence"
        ),
        depth=semantics.depth,
        depth_evidence=_required_string(arguments.get("depth_evidence"), "depth_evidence"),
        transferability=semantics.transferability,
        transferability_evidence=_required_string(
            arguments.get("transferability_evidence"),
            "transferability_evidence",
        ),
        evidence_recency=semantics.evidence_recency,
        level_alignment=semantics.level_alignment,
        opportunity_coherence=semantics.opportunity_coherence,
        caliber=semantics.caliber,
        outreach_tier=semantics.outreach_tier,
        reject_reason=semantics.reject_reason,
        case_for=_required_string(arguments.get("case_for"), "case_for"),
        case_against=_required_string(arguments.get("case_against"), "case_against"),
        confidence=semantics.confidence,
        post_save_modifier=semantics.post_save_modifier,
        review_reason_code=semantics.review_reason_code,
        review_structural_evidence=semantics.review_structural_evidence,
        review_recommended_next_step=semantics.review_recommended_next_step,
        summary=_required_string(arguments.get("summary"), "summary"),
    )


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise JudgmentToolContractError(
            "field_set_mismatch",
            f"context={context} missing={missing} extra={extra}",
        )


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JudgmentToolContractError("invalid_string", field)
    return value.strip()


def _nullable_string(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise JudgmentToolContractError("invalid_nullable_string", field)
    return value.strip()


def _nullable_enum_string(
    value: object,
    field: str,
    allowed: frozenset[str],
) -> str | None:
    if value is None:
        return None
    return _enum_string(value, field, allowed)


def _enum_string(value: Any, field: str, allowed: frozenset[str]) -> str:
    normalized = _required_string(value, field).upper()
    if normalized not in allowed:
        raise JudgmentToolContractError(f"invalid_{field}", normalized)
    return normalized
