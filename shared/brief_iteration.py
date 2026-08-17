"""Bounded draft-brief generation from structured run debriefs."""

from __future__ import annotations

import copy
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from market_intelligence.engine import resolve_market_intel_artifact_path
from shared import config
from shared.brief_loader import load_brief
from shared.llm_usage import llm_usage_session
from shared.llm_clients import opus_llm
from shared.retrieval_design import (
    derive_legacy_search_views,
    retrieval_design_from_payload,
    summarize_retrieval_design,
    validate_retrieval_design,
)
from shared.run_report_schema import StructuredRunReport
from shared.search_memory import build_search_memory_summary, extract_dominant_anchors
from shared.storage import read_json, read_jsonl, write_json
from shared.strict_seniority import (
    is_strict_seniority_brief,
    looks_like_company_inventory,
    looks_like_title_inventory,
    mentions_risky_title_translation,
)


SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}
MUTABLE_FIELDS = {
    "instructions",
    "search_priorities",
    "additional_search_terms",
    "intake_notes",
    "depth_distinction",
    "non_fit_patterns",
    "minimum_bar_description",
    "facial_calibration",
    "employer_signal_rules",
    "calibration_examples",
    "notes",
    "version",
    "retrieval_design",
}
LOCKED_FIELDS = {
    "role_title",
    "role_level",
    "role_summary",
    "geography",
    "linkedin_project",
    "linkedin_project_id",
    "capability_areas",
    "minimum_years_experience",
    "market_density",
    "kit_url",
}
LIST_LIMITS = {
    "instructions": 16,
    "search_priorities": 12,
    "additional_search_terms": 120,
    "non_fit_patterns": 10,
    "employer_signal_rules": 10,
}
CALIBRATION_MAX_DELTA = 0.10
PROMPT_TEXT_PREVIEW_CHARS = 1200
PROMPT_ITEM_TEXT_CHARS = 220
PROMPT_LIST_LIMIT = 8
PROMPT_TIGHT_TEXT_PREVIEW_CHARS = 700
PROMPT_TIGHT_ITEM_TEXT_CHARS = 140
PROMPT_TIGHT_LIST_LIMIT = 5


@dataclass
class BriefIterationResult:
    draft_brief_path: Path
    rationale_path: Path
    draft_brief: dict
    rationale_markdown: str
    warnings: list[str]
    proposal: dict


def gate_planner_diff(diff: Any) -> tuple[Any, str | None]:
    """Gate a PlannerDiff against brief-iteration safety rules.

    Returns (diff, None) if the diff passes, or (modified_diff, warning)
    if the diff was downgraded to a validation question, or (None, warning)
    if the diff was rejected entirely.

    Rules:
    - Employer/title inventory in constraint payloads without internal evidence
      gets rejected or downgraded to a validation question.
    - Conflicting evidence (internal says one thing, external says another)
      produces a validation question, not a hard edit.
    """
    from shared.sourcing_lanes import PlannerDiff

    if not isinstance(diff, PlannerDiff):
        return None, "not a PlannerDiff instance"

    if not diff.is_valid():
        return None, f"invalid diff: {diff.diff_id}"

    payload = diff.payload or {}
    has_internal = bool(diff.internal_evidence)

    # Gate: employer/title inventory without internal evidence
    if diff.target_type == "constraint" and diff.action == "add":
        values = payload.get("values", [])
        values_text = ", ".join(str(v) for v in values) if values else ""
        dimension = str(payload.get("dimension", "")).lower()

        if dimension in ("employer", "company", "employer_target"):
            if values_text and looks_like_company_inventory(values_text):
                if not has_internal:
                    return (
                        PlannerDiff(
                            diff_id=diff.diff_id,
                            action="add",
                            target_type="validation_question",
                            target_id=diff.target_id,
                            payload={
                                "question": f"Should we target these employers? {values_text[:200]}",
                                "source": "external_research",
                                "original_diff_type": "constraint",
                            },
                            internal_evidence=list(diff.internal_evidence),
                            external_evidence=list(diff.external_evidence),
                            confidence=diff.confidence * 0.5,
                        ),
                        f"Employer inventory without internal evidence downgraded to validation question: {diff.diff_id}",
                    )

        if dimension in ("title", "title_family"):
            if values_text and looks_like_title_inventory(values_text):
                if not has_internal:
                    return (
                        PlannerDiff(
                            diff_id=diff.diff_id,
                            action="add",
                            target_type="validation_question",
                            target_id=diff.target_id,
                            payload={
                                "question": f"Should we add these title families? {values_text[:200]}",
                                "source": "external_research",
                                "original_diff_type": "constraint",
                            },
                            internal_evidence=list(diff.internal_evidence),
                            external_evidence=list(diff.external_evidence),
                            confidence=diff.confidence * 0.5,
                        ),
                        f"Title inventory without internal evidence downgraded to validation question: {diff.diff_id}",
                    )

    # Gate: conflicting evidence → validation question
    if has_internal and diff.external_evidence:
        # If there's both internal and external evidence with a retire action,
        # downgrade to validation to avoid auto-retiring based on external opinion
        if diff.action == "retire" and diff.target_type in ("hypothesis", "slice"):
            return (
                PlannerDiff(
                    diff_id=diff.diff_id,
                    action="add",
                    target_type="validation_question",
                    target_id=diff.target_id,
                    payload={
                        "question": f"Should we retire {diff.target_type} {diff.target_id}? Internal and external evidence present.",
                        "source": "mixed_evidence",
                        "original_diff_type": diff.target_type,
                        "original_action": "retire",
                    },
                    internal_evidence=list(diff.internal_evidence),
                    external_evidence=list(diff.external_evidence),
                    confidence=diff.confidence * 0.7,
                ),
                f"Retire with mixed evidence downgraded to validation question: {diff.diff_id}",
            )

    return diff, None


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _truncate_text(value: Any, limit: int) -> str:
    text = _normalize_text(value)
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}…"


def _dedupe_strings(values: list[Any], limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = _normalize_text(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit and len(out) >= limit:
            break
    return out


def _prompt_limits(*, tight: bool) -> dict[str, int]:
    return {
        "text": PROMPT_TIGHT_TEXT_PREVIEW_CHARS if tight else PROMPT_TEXT_PREVIEW_CHARS,
        "item": PROMPT_TIGHT_ITEM_TEXT_CHARS if tight else PROMPT_ITEM_TEXT_CHARS,
        "list": PROMPT_TIGHT_LIST_LIMIT if tight else PROMPT_LIST_LIMIT,
    }


def _preview_list(values: list[Any], *, item_limit: int, item_chars: int) -> list[str]:
    return [
        _truncate_text(item, item_chars)
        for item in _dedupe_strings(values, limit=item_limit)
    ]


def _raw_has_explicit_retrieval_design(raw: dict) -> bool:
    payload = raw.get("retrieval_design")
    if not isinstance(payload, dict) or not payload:
        return False
    design = retrieval_design_from_payload(payload)
    return design.is_explicit()


_INTERNAL_MARKET_INTEL_CHANGE_MARKERS = (
    "lane_intelligence",
    "market_thesis.external_context",
    "keep_sections",
    "draft regression",
    "section_generation_metadata",
    "technical appendix",
    "preserve narrative",
    "noise_patterns",
    "talent_pool_intelligence",
    "brief_recommendations",
)


def _filter_operator_actionable_market_changes(
    values: list[Any],
    limit: int | None = None,
) -> list[str]:
    out: list[str] = []
    for text in _dedupe_strings(values):
        lowered = text.lower()
        if lowered.startswith("critical:"):
            continue
        if any(marker in lowered for marker in _INTERNAL_MARKET_INTEL_CHANGE_MARKERS):
            continue
        out.append(text)
        if limit and len(out) >= limit:
            break
    return out


def _normalize_non_fit_patterns(values: Any, current: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for item in values or []:
        if not isinstance(item, dict):
            continue
        label = _normalize_text(item.get("label"))
        description = _normalize_text(item.get("description"))
        why_not = _normalize_text(item.get("why_not"))
        if not (label and description and why_not):
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "label": label,
                "description": description,
                "why_not": why_not,
                "examples": _dedupe_strings(item.get("examples", []), limit=5),
            }
        )
        if len(out) >= LIST_LIMITS["non_fit_patterns"]:
            break
    return out or copy.deepcopy(current)


def _normalize_employer_signal_rules(values: Any, current: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for item in values or []:
        if not isinstance(item, dict):
            continue
        tier = _normalize_text(item.get("tier"))
        evidence_required = _normalize_text(item.get("evidence_required"))
        if not (tier and evidence_required):
            continue
        key = tier.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "tier": tier,
                "employer_patterns": _dedupe_strings(item.get("employer_patterns", []), limit=20),
                "evidence_required": evidence_required,
                "save_on_employer_alone": False,
            }
        )
        if len(out) >= LIST_LIMITS["employer_signal_rules"]:
            break
    return out or copy.deepcopy(current)


def _normalize_calibration_examples(values: Any, current: dict) -> dict:
    if not isinstance(values, dict):
        return copy.deepcopy(current)

    def _bucket(name: str) -> list[dict]:
        out: list[dict] = []
        seen: set[str] = set()
        for item in values.get(name, []) or []:
            if not isinstance(item, dict):
                continue
            candidate = _normalize_text(item.get("name"))
            why = _normalize_text(item.get("why"))
            if not (candidate and why):
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": candidate, "why": why})
            if len(out) >= 6:
                break
        return out

    normalized = {
        "strong_saves": _bucket("strong_saves"),
        "incorrect_saves": _bucket("incorrect_saves"),
        "borderline_verify": _bucket("borderline_verify"),
    }
    for key, value in normalized.items():
        if not value and isinstance(current, dict):
            normalized[key] = copy.deepcopy(current.get(key, []))
    return normalized


def _normalize_depth_distinction(values: Any, current: dict) -> dict:
    if not isinstance(values, dict):
        return copy.deepcopy(current)
    normalized = {
        "builder_definition": _normalize_text(values.get("builder_definition")) or _normalize_text(current.get("builder_definition")),
        "user_definition": _normalize_text(values.get("user_definition")) or _normalize_text(current.get("user_definition")),
        "edge_case_guidance": _normalize_text(values.get("edge_case_guidance")) or _normalize_text(current.get("edge_case_guidance")),
    }
    return normalized


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_facial_calibration(values: Any, current: dict) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if not isinstance(values, dict):
        return copy.deepcopy(current), warnings

    current_low = float(current.get("expected_yes_rate_low", 0.0))
    current_high = float(current.get("expected_yes_rate_high", 1.0))
    proposed_low = float(values.get("expected_yes_rate_low", current_low))
    proposed_high = float(values.get("expected_yes_rate_high", current_high))
    clamped_low = _clamp(proposed_low, max(0.0, current_low - CALIBRATION_MAX_DELTA), min(1.0, current_low + CALIBRATION_MAX_DELTA))
    clamped_high = _clamp(proposed_high, max(0.0, current_high - CALIBRATION_MAX_DELTA), min(1.0, current_high + CALIBRATION_MAX_DELTA))
    if clamped_high < clamped_low:
        clamped_high = clamped_low
    if clamped_low != proposed_low or clamped_high != proposed_high:
        warnings.append(
            "Facial calibration deltas were clamped to prevent a single run from radically retuning pass-through expectations."
        )

    normalized = {
        "expected_yes_rate_low": round(clamped_low, 4),
        "expected_yes_rate_high": round(clamped_high, 4),
        "fast_exit_patterns": _dedupe_strings(values.get("fast_exit_patterns", current.get("fast_exit_patterns", [])), limit=12),
        "trajectory_yes_patterns": _dedupe_strings(values.get("trajectory_yes_patterns", current.get("trajectory_yes_patterns", [])), limit=12),
        "trajectory_ambiguous_patterns": _dedupe_strings(values.get("trajectory_ambiguous_patterns", current.get("trajectory_ambiguous_patterns", [])), limit=12),
        "trajectory_no_patterns": _dedupe_strings(values.get("trajectory_no_patterns", current.get("trajectory_no_patterns", [])), limit=12),
    }
    return normalized, warnings


def _derive_next_draft_version(current_version: str) -> str:
    version = _normalize_text(current_version)
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:-draft)?$", version)
    if not match:
        return "draft"
    major = int(match.group(1))
    minor = int(match.group(2) or 0) + 1
    return f"{major}.{minor}-draft"


def _draft_brief_path(brief_path: Path, draft_version: str) -> Path:
    stem = brief_path.stem
    if draft_version == "draft":
        return brief_path.with_name(f"{stem}-draft{brief_path.suffix}")
    version_token = f"v{draft_version}"
    if re.search(r"-v\d+(?:\.\d+)?(?:-draft)?$", stem):
        new_stem = re.sub(r"-v\d+(?:\.\d+)?(?:-draft)?$", f"-{version_token}", stem)
    else:
        new_stem = f"{stem}-{version_token}"
    return brief_path.with_name(f"{new_stem}{brief_path.suffix}")


def _backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.stem}.bak-{timestamp}{path.suffix}")
    path.rename(backup)
    return backup


def _summarize_final_judgments(path: Path | None, limit: int = 12) -> dict:
    if not path or not path.exists():
        return {}
    records = [row for row in read_jsonl(path) if isinstance(row, dict)]
    saves: list[dict] = []
    rejects: list[dict] = []
    for row in records:
        entry = {
            "candidate_name": row.get("candidate_name", ""),
            "decision": row.get("decision", ""),
            "path": row.get("path", ""),
            "confidence": row.get("confidence"),
            "rationale": _normalize_text(row.get("rationale", ""))[:280],
        }
        if row.get("decision") in SAVE_DECISIONS and len(saves) < limit:
            saves.append(entry)
        elif row.get("decision") == "REJECT" and len(rejects) < limit:
            rejects.append(entry)
    return {
        "total_records": len(records),
        "save_examples": saves,
        "reject_examples": rejects,
    }


def _resolve_optional_paths(
    brief_path: Path,
    report_path: str | None,
    search_memory_path: str | None,
    final_judgments_path: str | None,
    output_dir: str | None,
) -> tuple[Path, Path, Path | None, Path | None]:
    output_root = Path(output_dir) if output_dir else config.OUTPUT_DIR
    brief = load_brief(str(brief_path))
    project_id = brief.linkedin_project_id or brief_path.stem
    resolved_report = Path(report_path) if report_path else output_root / "run-report.json"
    resolved_search_memory = (
        Path(search_memory_path)
        if search_memory_path
        else output_root / f"search_memory-{project_id}.json"
    )
    resolved_final = (
        Path(final_judgments_path)
        if final_judgments_path
        else output_root / "final_judgments.jsonl"
    )
    return output_root, resolved_report, resolved_search_memory, resolved_final


def _load_market_intel_summary(brief_path: Path, output_root: Path) -> dict | None:
    artifact_path = resolve_market_intel_artifact_path(
        brief_path,
        output_dir=output_root,
    )
    if not artifact_path.exists():
        return None
    artifact = read_json(artifact_path)
    if not isinstance(artifact, dict):
        return None
    market_thesis = artifact.get("market_thesis", {})
    talent_pools = artifact.get("talent_pool_intelligence", [])
    employer_signals = artifact.get("employer_signal_intelligence", [])
    brief_recommendations = artifact.get("brief_recommendations", [])
    open_questions = artifact.get("open_questions", [])
    delta = artifact.get("delta_since_last_run", {})
    summary = {
        "market_key": artifact.get("market_identity", {}).get("market_key", ""),
        "artifact_path": str(artifact_path),
        "market_thesis_summary": _truncate_text(market_thesis.get("summary"), 1400),
        "next_run_changes": _filter_operator_actionable_market_changes(
            delta.get("next_run_changes", []),
            limit=5,
        ),
        "brief_recommendations": [
            {
                "target_field": _normalize_text(item.get("target_field")),
                "proposal": _truncate_text(item.get("proposal"), 320),
                "reason": _truncate_text(item.get("reason"), 380),
            }
            for item in brief_recommendations[:5]
            if isinstance(item, dict)
        ],
        "top_talent_pool_findings": [
            {
                "label": _normalize_text(item.get("label")),
                "evidence_summary": _truncate_text(item.get("evidence_summary"), 260),
            }
            for item in talent_pools[:3]
            if isinstance(item, dict)
        ],
        "top_employer_findings": [
            {
                "label": _normalize_text(item.get("label")),
                "evidence_summary": _truncate_text(item.get("evidence_summary"), 260),
            }
            for item in employer_signals[:3]
            if isinstance(item, dict)
        ],
        "open_questions": [
            {
                "question": _truncate_text(item.get("question"), 260),
                "priority": _normalize_text(item.get("priority")),
                "next_step": _truncate_text(item.get("next_step"), 220),
            }
            for item in open_questions[:4]
            if isinstance(item, dict)
        ],
        "retrieval_design_summary": artifact.get("retrieval_design_summary", {}),
    }
    return summary


def _summarize_mutable_fields_for_prompt(
    current_raw: dict,
    *,
    retrieval_design: Any,
    allow_retrieval_design_edits: bool,
    tight: bool,
) -> dict:
    limits = _prompt_limits(tight=tight)
    derived_search_priorities: list[str] = []
    derived_additional_search_terms: list[str] = []
    if allow_retrieval_design_edits:
        derived_search_priorities, derived_additional_search_terms = derive_legacy_search_views(
            retrieval_design
        )
    out: dict[str, Any] = {}
    for field in MUTABLE_FIELDS:
        if field not in current_raw:
            continue
        value = current_raw.get(field)
        if field in {"instructions", "search_priorities", "additional_search_terms"}:
            if allow_retrieval_design_edits and field == "search_priorities":
                value = derived_search_priorities
            elif allow_retrieval_design_edits and field == "additional_search_terms":
                value = derived_additional_search_terms
            field_limit = limits["list"]
            if field == "additional_search_terms":
                field_limit = 12 if tight else 20
            out[field] = _preview_list(value or [], item_limit=field_limit, item_chars=limits["item"])
        elif field in {"intake_notes", "minimum_bar_description", "notes"}:
            out[field] = _truncate_text(value, limits["text"])
        elif field == "depth_distinction" and isinstance(value, dict):
            out[field] = {
                key: _truncate_text(value.get(key), 320 if tight else 500)
                for key in ("builder_definition", "user_definition", "edge_case_guidance")
            }
        elif field == "non_fit_patterns" and isinstance(value, list):
            out[field] = [
                {
                    "label": _truncate_text(item.get("label"), 80),
                    "description": _truncate_text(item.get("description"), 180 if tight else 260),
                    "why_not": _truncate_text(item.get("why_not"), 180 if tight else 240),
                }
                for item in value[: limits["list"]]
                if isinstance(item, dict)
            ]
        elif field == "facial_calibration" and isinstance(value, dict):
            out[field] = {
                "expected_yes_rate_low": value.get("expected_yes_rate_low"),
                "expected_yes_rate_high": value.get("expected_yes_rate_high"),
                "fast_exit_patterns": _preview_list(value.get("fast_exit_patterns", []), item_limit=limits["list"], item_chars=limits["item"]),
                "trajectory_yes_patterns": _preview_list(value.get("trajectory_yes_patterns", []), item_limit=limits["list"], item_chars=limits["item"]),
                "trajectory_ambiguous_patterns": _preview_list(value.get("trajectory_ambiguous_patterns", []), item_limit=limits["list"], item_chars=limits["item"]),
                "trajectory_no_patterns": _preview_list(value.get("trajectory_no_patterns", []), item_limit=limits["list"], item_chars=limits["item"]),
            }
        elif field == "employer_signal_rules" and isinstance(value, list):
            out[field] = [
                {
                    "tier": _truncate_text(item.get("tier"), 80),
                    "employer_patterns": _preview_list(item.get("employer_patterns", []), item_limit=6 if tight else 10, item_chars=80),
                    "evidence_required": _truncate_text(item.get("evidence_required"), 180 if tight else 240),
                    "save_on_employer_alone": bool(item.get("save_on_employer_alone", False)),
                }
                for item in value[: limits["list"]]
                if isinstance(item, dict)
            ]
        elif field == "calibration_examples" and isinstance(value, dict):
            out[field] = {
                bucket: [
                    {
                        "name": _truncate_text(item.get("name"), 80),
                        "why": _truncate_text(item.get("why"), 180 if tight else 240),
                    }
                    for item in (value.get(bucket, []) or [])[: limits["list"]]
                    if isinstance(item, dict)
                ]
                for bucket in ("strong_saves", "incorrect_saves", "borderline_verify")
            }
        elif field == "retrieval_design" and allow_retrieval_design_edits:
            out[field] = summarize_retrieval_design(retrieval_design)
        else:
            out[field] = value
    return out


def _summarize_locked_fields_for_prompt(current_raw: dict, *, tight: bool) -> dict:
    limits = _prompt_limits(tight=tight)
    out: dict[str, Any] = {}
    for field in LOCKED_FIELDS:
        if field not in current_raw:
            continue
        value = current_raw.get(field)
        if isinstance(value, str):
            out[field] = _truncate_text(value, 320 if tight else 600)
        elif isinstance(value, list):
            out[field] = _preview_list(value, item_limit=limits["list"], item_chars=limits["item"])
        else:
            out[field] = value
    return out


def _summarize_run_report_for_prompt(
    report: StructuredRunReport,
    *,
    tight: bool,
) -> dict:
    limits = _prompt_limits(tight=tight)
    payload = report.to_dict()
    run_metadata = payload.get("run_metadata", {}) if isinstance(payload, dict) else {}
    metrics_summary = payload.get("metrics_summary", {}) if isinstance(payload, dict) else {}
    saved_patterns = payload.get("saved_candidate_patterns", {}) if isinstance(payload, dict) else {}
    adaptation = payload.get("adaptation_assessment", {}) if isinstance(payload, dict) else {}
    recommendations = payload.get("recommendations", {}) if isinstance(payload, dict) else {}
    hints = payload.get("brief_iteration_hints", {}) if isinstance(payload, dict) else {}

    ranked_string_performance = sorted(
        [
            item
            for item in (payload.get("string_performance", []) or [])
            if isinstance(item, dict)
        ],
        key=lambda item: (
            1 if _normalize_text(item.get("status")) == "done" else 0,
            float(item.get("saves", 0) or 0),
            float(item.get("pages_reviewed", 0) or 0),
            float(item.get("result_count", 0) or 0),
            float(item.get("save_rate", 0) or 0),
        ),
        reverse=True,
    )
    string_items = []
    for item in ranked_string_performance[: 4 if tight else 6]:
        if not isinstance(item, dict):
            continue
        string_items.append(
            {
                "name": _truncate_text(item.get("name"), 90),
                "status": _normalize_text(item.get("status")),
                "result_count": item.get("result_count"),
                "pages_reviewed": item.get("pages_reviewed"),
                "candidates_count": item.get("candidates_count"),
                "saves": item.get("saves"),
                "save_rate": item.get("save_rate"),
                "family_key": _normalize_text(item.get("family_key")),
                "novelty_bucket": _normalize_text(item.get("novelty_bucket")),
                "domain_lane": _normalize_text(item.get("domain_lane")),
                "notes": _truncate_text(item.get("notes"), 160 if tight else 220),
            }
        )

    def _lane_summary(items: list[Any], *, label_key: str) -> list[dict]:
        rows: list[dict] = []
        for item in (items or [])[: limits["list"]]:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    label_key: _truncate_text(item.get(label_key), 110),
                    "evidence": _truncate_text(item.get("evidence"), 180 if tight else 240),
                    "why": _truncate_text(
                        item.get("why_it_worked") or item.get("issue"),
                        180 if tight else 220,
                    ),
                    "recommended_action": _truncate_text(item.get("recommended_action"), 180 if tight else 220),
                }
            )
        return rows

    return {
        "run_metadata": {
            "role_title": _normalize_text(run_metadata.get("role_title")),
            "brief_version": _normalize_text(run_metadata.get("brief_version")),
            "linkedin_project_id": _normalize_text(run_metadata.get("linkedin_project_id")),
            "generated_at": _normalize_text(run_metadata.get("generated_at")),
            "overall_summary": _truncate_text(
                payload.get("overall_summary") or run_metadata.get("overall_summary"),
                280 if tight else 420,
            ),
        },
        "metrics_summary": {
            key: metrics_summary.get(key)
            for key in (
                "strings_executed",
                "strings_skipped",
                "total_results",
                "total_pages_reviewed",
                "candidates_evaluated",
                "facial_yes",
                "facial_no",
                "saved",
                "rejected",
                "overall_save_rate",
                "facial_yes_rate",
            )
            if key in metrics_summary
        },
        "top_string_performance": string_items,
        "winning_lanes": _lane_summary(payload.get("winning_lanes", []), label_key="lane"),
        "underperforming_lanes": _lane_summary(payload.get("underperforming_lanes", []), label_key="lane"),
        "coverage_gaps": [
            {
                "gap": _truncate_text(item.get("gap"), 110),
                "why_it_matters": _truncate_text(item.get("why_it_matters"), 180 if tight else 240),
                "suggested_search_strategy": _truncate_text(item.get("suggested_search_strategy"), 180 if tight else 240),
            }
            for item in (payload.get("coverage_gaps", []) or [])[: limits["list"]]
            if isinstance(item, dict)
        ],
        "noise_patterns": [
            {
                "pattern": _truncate_text(item.get("pattern"), 110),
                "evidence": _truncate_text(item.get("evidence"), 180 if tight else 240),
                "mitigation": _truncate_text(item.get("mitigation"), 180 if tight else 240),
            }
            for item in (payload.get("noise_patterns", []) or [])[: limits["list"]]
            if isinstance(item, dict)
        ],
        "saved_candidate_patterns": {
            "standout_candidates": [
                {
                    "name": _truncate_text(item.get("name"), 80),
                    "why": _truncate_text(item.get("why"), 180 if tight else 240),
                }
                for item in (saved_patterns.get("standout_candidates", []) or [])[: limits["list"]]
                if isinstance(item, dict)
            ],
            "common_employers": [
                {
                    "employer": _truncate_text(item.get("employer"), 80),
                    "count": item.get("count"),
                    "note": _truncate_text(item.get("note"), 140 if tight else 200),
                }
                for item in (saved_patterns.get("common_employers", []) or [])[: limits["list"]]
                if isinstance(item, dict)
            ],
            "common_titles": [
                {
                    "title_family": _truncate_text(item.get("title_family"), 80),
                    "count": item.get("count"),
                    "note": _truncate_text(item.get("note"), 140 if tight else 200),
                }
                for item in (saved_patterns.get("common_titles", []) or [])[: limits["list"]]
                if isinstance(item, dict)
            ],
            "archetype_distribution": [
                {
                    "archetype": _truncate_text(item.get("archetype"), 90),
                    "count": item.get("count"),
                    "note": _truncate_text(item.get("note"), 140 if tight else 200),
                }
                for item in (saved_patterns.get("archetype_distribution", []) or [])[: limits["list"]]
                if isinstance(item, dict)
            ],
            "seniority_notes": _preview_list(saved_patterns.get("seniority_notes", []), item_limit=limits["list"], item_chars=limits["item"]),
        },
        "adaptation_assessment": {
            "summary": _truncate_text(adaptation.get("summary"), 220 if tight else 320),
            "effective_refinements": _preview_list(adaptation.get("effective_refinements", []), item_limit=limits["list"], item_chars=limits["item"]),
            "questionable_or_skipped": _preview_list(adaptation.get("questionable_or_skipped", []), item_limit=limits["list"], item_chars=limits["item"]),
            "operational_notes": _preview_list(adaptation.get("operational_notes", []), item_limit=limits["list"], item_chars=limits["item"]),
        },
        "recommendations": {
            "try_next": _preview_list(recommendations.get("try_next", []), item_limit=limits["list"], item_chars=limits["item"]),
            "avoid_next": _preview_list(recommendations.get("avoid_next", []), item_limit=limits["list"], item_chars=limits["item"]),
            "prioritize_pipeline": _preview_list(recommendations.get("prioritize_pipeline", []), item_limit=limits["list"], item_chars=limits["item"]),
        },
        "brief_iteration_hints": {
            "instructions": _preview_list(hints.get("instructions", []), item_limit=limits["list"], item_chars=limits["item"]),
            "search_priorities": _preview_list(hints.get("search_priorities", []), item_limit=limits["list"], item_chars=limits["item"]),
            "additional_search_terms": _preview_list(hints.get("additional_search_terms", []), item_limit=12 if not tight else 8, item_chars=limits["item"]),
            "intake_notes": _truncate_text(hints.get("intake_notes"), 280 if tight else 420),
            "depth_distinction": {
                key: _truncate_text((hints.get("depth_distinction", {}) or {}).get(key), 220 if tight else 320)
                for key in ("builder_definition", "user_definition", "edge_case_guidance")
            }
            if isinstance(hints.get("depth_distinction"), dict)
            else None,
            "non_fit_patterns": [
                {
                    "label": _truncate_text(item.get("label"), 80),
                    "description": _truncate_text(item.get("description"), 160 if tight else 220),
                    "why_not": _truncate_text(item.get("why_not"), 160 if tight else 220),
                }
                for item in (hints.get("non_fit_patterns", []) or [])[: limits["list"]]
                if isinstance(item, dict)
            ],
            "minimum_bar_description": _truncate_text(hints.get("minimum_bar_description"), 220 if tight else 320),
            "facial_calibration": {
                "expected_yes_rate_low": (hints.get("facial_calibration", {}) or {}).get("expected_yes_rate_low"),
                "expected_yes_rate_high": (hints.get("facial_calibration", {}) or {}).get("expected_yes_rate_high"),
                "fast_exit_patterns": _preview_list((hints.get("facial_calibration", {}) or {}).get("fast_exit_patterns", []), item_limit=limits["list"], item_chars=limits["item"]),
                "trajectory_yes_patterns": _preview_list((hints.get("facial_calibration", {}) or {}).get("trajectory_yes_patterns", []), item_limit=limits["list"], item_chars=limits["item"]),
                "trajectory_ambiguous_patterns": _preview_list((hints.get("facial_calibration", {}) or {}).get("trajectory_ambiguous_patterns", []), item_limit=limits["list"], item_chars=limits["item"]),
                "trajectory_no_patterns": _preview_list((hints.get("facial_calibration", {}) or {}).get("trajectory_no_patterns", []), item_limit=limits["list"], item_chars=limits["item"]),
            }
            if isinstance(hints.get("facial_calibration"), dict)
            else None,
            "employer_signal_rules": [
                {
                    "tier": _truncate_text(item.get("tier"), 80),
                    "employer_patterns": _preview_list(item.get("employer_patterns", []), item_limit=6 if tight else 10, item_chars=80),
                    "evidence_required": _truncate_text(item.get("evidence_required"), 160 if tight else 220),
                    "save_on_employer_alone": bool(item.get("save_on_employer_alone", False)),
                }
                for item in (hints.get("employer_signal_rules", []) or [])[: limits["list"]]
                if isinstance(item, dict)
            ],
            "calibration_examples": {
                bucket: [
                    {
                        "name": _truncate_text(item.get("name"), 80),
                        "why": _truncate_text(item.get("why"), 160 if tight else 220),
                    }
                    for item in ((hints.get("calibration_examples", {}) or {}).get(bucket, []) or [])[: limits["list"]]
                    if isinstance(item, dict)
                ]
                for bucket in ("strong_saves", "incorrect_saves", "borderline_verify")
            }
            if isinstance(hints.get("calibration_examples"), dict)
            else None,
            "notes": _truncate_text(hints.get("notes"), 220 if tight else 320),
            "locked_field_cautions": _preview_list(hints.get("locked_field_cautions", []), item_limit=limits["list"], item_chars=limits["item"]),
        },
    }


def _build_iteration_system(*, allow_retrieval_design_edits: bool, strict_seniority_legacy: bool = False) -> str:
    retrieval_design_block = """
    "retrieval_design": {
      "families": [],
      "shared_layers": {},
      "edge_case_hypotheses": []
    },""" if allow_retrieval_design_edits else ""
    retrieval_rules = (
        "- retrieval_design is the canonical search-design surface for this brief. Prefer editing retrieval_design first, then let legacy search_priorities and additional_search_terms be derived from it.\n"
        "- Do NOT propose standalone search_priorities or additional_search_terms edits that are inconsistent with retrieval_design. If search behavior should change, express it through retrieval_design."
        if allow_retrieval_design_edits
        else "- Do NOT introduce a new retrieval_design for this brief. Keep edits on the existing legacy mutable fields only."
    )
    strict_rules = ""
    if strict_seniority_legacy:
        strict_rules = (
            "\n- This is a strict-seniority legacy brief. Preserve semantic search guidance and technical-authority framing; do NOT literalize employer clusters, title inventories, or company-first discoveries into long search_priorities/additional_search_terms lists.\n"  # VERTICAL-VOCAB(strict-seniority-legacy)
            "- Market intel can enrich notes, intake_notes, employer_signal_rules, calibration_examples, and later-phase guidance, but it must NOT broaden seniority translation on noisy evidence.\n"
            "- Prefer abstract search priorities like 'buy-side AI lab leaders with clearly in-range technical scope' over enumerating employer sets.\n"
            "- Do NOT loosen ED-equivalent calibration because a lane produced saves if the run metrics are inconsistent or the saved profiles may be above band.\n"
        )
    # VERTICAL-VOCAB(strict-seniority-legacy)
    template = """You are revising a sourcing brief after a completed run.

Return valid JSON only with this exact shape:
{
  "summary": "short summary string",
  "proposed_changes": {
    "instructions": ["..."],
    "search_priorities": ["..."],
    "additional_search_terms": ["..."],
    "intake_notes": "string",
    "depth_distinction": {
      "builder_definition": "string",
      "user_definition": "string",
      "edge_case_guidance": "string"
    },
    "non_fit_patterns": [
      {"label": "string", "description": "string", "why_not": "string", "examples": ["..."]}
    ],
    "minimum_bar_description": "string",
    "facial_calibration": {
      "expected_yes_rate_low": 0.0,
      "expected_yes_rate_high": 0.0,
      "fast_exit_patterns": ["..."],
      "trajectory_yes_patterns": ["..."],
      "trajectory_ambiguous_patterns": ["..."],
      "trajectory_no_patterns": ["..."]
    },
    "employer_signal_rules": [
      {"tier": "string", "employer_patterns": ["..."], "evidence_required": "string", "save_on_employer_alone": false}
    ],
    "calibration_examples": {
      "strong_saves": [{"name": "string", "why": "string"}],
      "incorrect_saves": [{"name": "string", "why": "string"}],
      "borderline_verify": [{"name": "string", "why": "string"}]
    },__RETRIEVAL_BLOCK__
    "notes": "string",
    "version": "string"
  },
  "changed_fields": [
    {
      "field": "field_name",
      "why": "why the change matters",
      "evidence": ["quoted or paraphrased evidence from the run report"],
      "expected_effects": ["downstream operational effects"]
    }
  ],
  "warnings": ["optional warning strings"]
}

Rules:
- Propose changes ONLY for mutable fields explicitly shown above.
- Do NOT propose changes to role identity, geography, LinkedIn project mapping, minimum years, capability areas, or market density.
- Keep hard gates intact: geography, years, BFSI domain, post-2022 GenAI builder evidence, executive-builder scope.
- Prefer replacing low-signal items over append-only growth.
- Keep lists concise and high-signal.
- __RETRIEVAL_RULES__
- Anti-employer-proxy rule (applies to every brief, not only strict-seniority): do NOT literalize employer clusters, prestige-company findings, or company-inventory text into search_priorities or additional_search_terms. Prestige employers are NOT confidence-boosting retrieval advice on their own; they need behavioral evidence (specific work, build patterns, capability area) to belong in retrieval. If employer signal is strong but behavioral evidence is thin, route the finding into employer_signal_rules as secondary classification logic, never into opening retrieval.
- Prefer abstract behavioral search guidance (the kind of work, the build pattern, the capability area) over employer or title inventories. Titles, employers, and keywords can support a pattern; they should not be the pattern.__STRICT_RULES__
- If you suggest employer signal rules, save_on_employer_alone must stay false.
- If you suggest facial calibration changes, make them small and evidence-based.
- If there is not enough evidence to change a field, omit it from proposed_changes."""
    return (
        template.replace("__RETRIEVAL_BLOCK__", retrieval_design_block)
        .replace("__RETRIEVAL_RULES__", retrieval_rules)
        .replace("__STRICT_RULES__", strict_rules)
    )


def _build_iteration_user_prompt(
    current_raw: dict,
    report: StructuredRunReport,
    search_memory: dict | None,
    final_judgments_summary: dict | None,
    market_intel_summary: dict | None,
    *,
    allow_retrieval_design_edits: bool,
    strict_seniority_legacy: bool = False,
    tight: bool = False,
) -> str:
    retrieval_design = retrieval_design_from_payload(
        current_raw.get("retrieval_design"),
        legacy_search_priorities=current_raw.get("search_priorities", []),
        legacy_additional_search_terms=current_raw.get("additional_search_terms", []),
        role_title=current_raw.get("role_title", ""),
    )
    mutable_snapshot = _summarize_mutable_fields_for_prompt(
        current_raw,
        retrieval_design=retrieval_design,
        allow_retrieval_design_edits=allow_retrieval_design_edits,
        tight=tight,
    )
    locked_snapshot = _summarize_locked_fields_for_prompt(current_raw, tight=tight)
    context = {
        "prompt_mode": "tight_retry" if tight else "standard",
        "note": (
            "Values below are compact previews of the current brief and supporting artifacts. Preserve their intent and constraints when proposing edits."
        ),
        "current_mutable_fields": mutable_snapshot,
        "locked_fields": locked_snapshot,
        "run_report": _summarize_run_report_for_prompt(report, tight=tight),
        "search_memory_summary": build_search_memory_summary(search_memory) if search_memory else None,
        "final_judgments_summary": final_judgments_summary or None,
        "market_intel_summary": market_intel_summary or None,
    }
    if strict_seniority_legacy:
        context["strict_seniority_guardrails"] = {
            "mode": "legacy_strict_seniority",
            "search_hint_policy": "Keep search_priorities and additional_search_terms abstract and semantic; do not literalize employer clusters or title inventories.",
            "seniority_policy": "Executive Director / clearly ED-analogous remains the anchor. Do not broaden buy-side MD or fintech VP translation unless strongly validated as in-band.",  # VERTICAL-VOCAB(strict-seniority-legacy)
            "run_metrics_consistency": "If the run metrics are inconsistent, do not loosen seniority or expand literal search-token lists based on that evidence.",
        }
    if allow_retrieval_design_edits:
        context["current_retrieval_design_summary"] = summarize_retrieval_design(retrieval_design)
    return (
        "Revise the brief using the run report and optional supporting artifacts.\n"
        "Use market-intel recommendations to improve the next sourcing run, but only propose bounded edits to mutable fields.\n\n"
        f"{json.dumps(context, indent=2)}"
    )


def _ensure_hard_gate_instruction(instructions: list[str], current_raw: dict) -> list[str]:
    existing = _dedupe_strings(instructions, limit=LIST_LIMITS["instructions"])
    if any("hard gates" in item.lower() for item in existing):
        return existing
    geography = current_raw.get("geography", "the target geography")
    years = current_raw.get("minimum_years_experience", 0)
    # VERTICAL-VOCAB(strict-seniority-legacy)
    fallback = (
        f"The evaluation bar does NOT move. Keep the hard gates: {geography}, "
        f"{years}+ years, BFSI domain depth, post-2022 applied GenAI build evidence, and executive-builder scope."
    )
    existing.append(fallback)
    return _dedupe_strings(existing, limit=LIST_LIMITS["instructions"])


def _ensure_minimum_bar_guardrail(description: str, current_raw: dict) -> str:
    text = _normalize_text(description)
    required_bits: list[str] = []
    geography = str(current_raw.get("geography", "")).strip()
    years = current_raw.get("minimum_years_experience")
    if geography and geography.lower() not in text.lower():
        required_bits.append(f"Current {geography} location remains non-negotiable.")
    if years and str(years) not in text and "fifteen" not in text.lower():
        required_bits.append(f"{years}+ years remains a hard floor.")
    if "genai" not in text.lower() and "llm" not in text.lower():  # VERTICAL-VOCAB(strict-seniority-legacy)
        required_bits.append("Post-2022 applied GenAI or LLM work remains a hard requirement.")  # VERTICAL-VOCAB(strict-seniority-legacy)
    if "bfsi" not in text.lower() and "financial" not in text.lower():  # VERTICAL-VOCAB(strict-seniority-legacy)
        required_bits.append("This remains a BFSI-first search.")  # VERTICAL-VOCAB(strict-seniority-legacy)
    if "builder" not in text.lower():
        required_bits.append("The role remains an executive-builder search, not strategy or product management.")
    if required_bits:
        text = f"{text} {' '.join(required_bits)}".strip()
    return text


def _collect_heuristic_hints() -> set[str]:
    from linkedin import strategy as strategy_mod
    from shared import search_memory as search_memory_mod

    hints = set()
    for source in (
        getattr(strategy_mod, "_EDGE_CASE_PATTERNS", ()),
        getattr(strategy_mod, "_EDGE_CASE_COMPANY_PATTERNS", ()),
        getattr(search_memory_mod, "_ANCHOR_PHRASES", ()),
        getattr(search_memory_mod, "_EDGE_CASE_TERMS", ()),
    ):
        hints.update(_normalize_text(item).lower() for item in source if _normalize_text(item))
    for values in getattr(search_memory_mod, "_DOMAIN_LANE_HINTS", {}).values():
        hints.update(_normalize_text(item).lower() for item in values if _normalize_text(item))
    return hints


def _find_heuristic_gap_warnings(current_raw: dict, draft_raw: dict) -> list[str]:
    heuristic_hints = _collect_heuristic_hints()
    warnings: list[str] = []

    current_terms = {_normalize_text(item).lower() for item in current_raw.get("additional_search_terms", [])}
    for term in draft_raw.get("additional_search_terms", []):
        normalized = _normalize_text(term).lower()
        if not normalized or normalized in current_terms:
            continue
        if not any(hint in normalized or normalized in hint for hint in heuristic_hints):
            warnings.append(
                f"New search term '{term}' is not recognized by current strategy/search-memory heuristics and may need follow-up heuristic support."
            )

    current_priorities = {_normalize_text(item).lower() for item in current_raw.get("search_priorities", [])}
    for priority in draft_raw.get("search_priorities", []):
        normalized = _normalize_text(priority).lower()
        if not normalized or normalized in current_priorities:
            continue
        anchors = [anchor.lower() for anchor in extract_dominant_anchors(priority, limit=6)]
        if anchors and not any(
            any(hint in anchor or anchor in hint for hint in heuristic_hints)
            for anchor in anchors
        ):
            warnings.append(
                f"New search priority '{priority}' introduces a lane that current strategy/search-memory heuristics may not recognize."
            )

    deduped: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        key = warning.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(warning)
    return deduped


def _run_metrics_are_internally_inconsistent(report: StructuredRunReport) -> bool:
    metrics_saved = int(report.metrics_summary.get("saved", 0) or 0)
    string_saved = sum(int(item.get("saves", 0) or 0) for item in report.string_performance)
    return metrics_saved == 0 and string_saved > 0


def _general_anti_employer_proxy_priority_rewrite(item: str) -> str | None:
    """General anti-employer-proxy rewrite for search_priorities.

    Applies to every brief, not only strict-seniority. The rule is
    behavior-first: if a proposed search priority reads as a company
    inventory (employer cluster prose), rewrite it to abstract behavioral
    guidance so prestige employers do not literalize into opening
    retrieval. Employer findings remain valid as secondary classification
    logic via employer_signal_rules.

    Returns the rewritten string, the original normalized string when no
    rewrite is needed, or None to drop the item entirely.
    """
    normalized = _normalize_text(item)
    if not normalized:
        return None
    if looks_like_company_inventory(normalized):
        return (
            "Behavioral search guidance: target the actual build pattern "
            "and capability area rather than the employer set. Use employer "
            "signals only as secondary classification."
        )
    return normalized


def _general_anti_employer_proxy_term_rewrite(item: str) -> str | None:
    """General anti-employer-proxy rewrite for additional_search_terms.

    Same intent as the priority rewrite: drop or reframe employer-inventory
    terms so they cannot become opening retrieval. Title-inventory terms
    are also reframed because title clusters are a form of identity proxy
    that should be a brief-secondary classifier, not a retrieval driver.
    """
    normalized = _normalize_text(item)
    if not normalized:
        return None
    if looks_like_company_inventory(normalized):
        return "behavioral build-pattern terms with capability-area context"
    if looks_like_title_inventory(normalized):
        return "bounded title variants tied to scoped capability work"
    return normalized


def _apply_general_anti_employer_proxy_guardrails(
    *,
    current_raw: dict,
    draft: dict,
    warnings: list[str],
) -> None:
    """Apply behavior-first anti-employer-proxy rewriting to every brief.

    Strict-seniority briefs receive a tighter rewrite via
    `_apply_strict_seniority_legacy_guardrails` after this. Briefs in
    explicit retrieval_design mode skip this step because their
    search_priorities and additional_search_terms are derived views.
    """
    if _raw_has_explicit_retrieval_design(current_raw):
        return

    rewritten_priorities = _dedupe_strings(
        [
            rewritten
            for item in draft.get("search_priorities", [])
            if (rewritten := _general_anti_employer_proxy_priority_rewrite(item))
        ],
        limit=LIST_LIMITS["search_priorities"],
    )
    rewritten_terms = _dedupe_strings(
        [
            rewritten
            for item in draft.get("additional_search_terms", [])
            if (rewritten := _general_anti_employer_proxy_term_rewrite(item))
        ],
        limit=LIST_LIMITS["additional_search_terms"],
    )

    if rewritten_priorities != draft.get("search_priorities", []):
        warnings.append(
            "Rewrote search_priorities to drop employer-cluster literalization "
            "and keep retrieval guidance behavior-first."
        )
        draft["search_priorities"] = rewritten_priorities
    if rewritten_terms != draft.get("additional_search_terms", []):
        warnings.append(
            "Rewrote additional_search_terms to drop employer/title-inventory "
            "literalization and keep retrieval guidance behavior-first."
        )
        draft["additional_search_terms"] = rewritten_terms


def _strict_seniority_priority_rewrite(item: str) -> str | None:
    normalized = _normalize_text(item)
    lowered = normalized.lower()
    if not normalized:
        return None
    if looks_like_company_inventory(normalized):
        if any(term in lowered for term in ("blackrock", "bridgewater", "citadel", "two sigma", "point72", "aqr", "buy-side", "buy side")):  # VERTICAL-VOCAB(strict-seniority-legacy)
            return "Buy-side AI lab leaders with clearly in-range technical scope and real BFSI workflow credibility"  # VERTICAL-VOCAB(strict-seniority-legacy)
        if any(term in lowered for term in ("stripe", "plaid", "revolut", "sofi", "affirm", "adyen", "klarna", "payments", "fintech")):  # VERTICAL-VOCAB(strict-seniority-legacy)
            return "Senior fintech and payments AI builders only when the scope clearly reads ED-equivalent and builder-led"  # VERTICAL-VOCAB(strict-seniority-legacy)
        if any(term in lowered for term in ("bloomberg", "s&p global", "dtcc", "factset", "market infrastructure", "tradeweb", "lseg", "broadridge")):
            return "AI leaders at market infrastructure and financial-data firms when the scope still reads one layer below broad enterprise executives"
        if any(term in lowered for term in ("jpmorgan", "goldman", "morgan stanley", "citi", "barclays", "hsbc", "ubs", "deutsche", "bank")):
            return "Senior technical owners at roughly Executive Director scope inside major financial institutions"
    return normalized


def _strict_seniority_term_rewrite(item: str) -> str | None:
    normalized = _normalize_text(item)
    lowered = normalized.lower()
    if not normalized:
        return None
    if looks_like_company_inventory(normalized):
        if any(term in lowered for term in ("buy-side", "buy side", "blackrock", "citadel", "two sigma", "point72", "aqr")):  # VERTICAL-VOCAB(strict-seniority-legacy)
            return "buy-side ai lab leaders with clearly in-range technical scope"  # VERTICAL-VOCAB(strict-seniority-legacy)
        if any(term in lowered for term in ("stripe", "plaid", "revolut", "sofi", "affirm", "adyen", "klarna", "payments", "fintech")):  # VERTICAL-VOCAB(strict-seniority-legacy)
            return "regulated-workflow ai builders with clearly ed-equivalent scope"
        if any(term in lowered for term in ("dtcc", "s&p global", "factset", "ice", "lseg", "broadridge", "finastra")):
            return "market-infrastructure ai builders with clear ed-equivalent technical ownership"
        return None
    if looks_like_title_inventory(normalized):
        return "bounded bfsi title variants that still imply true lab-head scope"  # VERTICAL-VOCAB(strict-seniority-legacy)
    return normalized


def _apply_strict_seniority_legacy_guardrails(
    *,
    current_raw: dict,
    draft: dict,
    report: StructuredRunReport,
    warnings: list[str],
) -> None:
    if not is_strict_seniority_brief(current_raw) or _raw_has_explicit_retrieval_design(current_raw):
        return

    rewritten_priorities = _dedupe_strings(
        [
            rewritten
            for item in draft.get("search_priorities", [])
            if (rewritten := _strict_seniority_priority_rewrite(item))
        ],
        limit=LIST_LIMITS["search_priorities"],
    )
    rewritten_terms = _dedupe_strings(
        [
            rewritten
            for item in draft.get("additional_search_terms", [])
            if (rewritten := _strict_seniority_term_rewrite(item))
        ],
        limit=LIST_LIMITS["additional_search_terms"],
    )

    if rewritten_priorities != draft.get("search_priorities", []):
        warnings.append(
            "Rewrote search_priorities to keep this strict-seniority brief abstract and semantic instead of company-list-first."
        )
        draft["search_priorities"] = (
            rewritten_priorities or current_raw.get("search_priorities", [])
        )
    if rewritten_terms != draft.get("additional_search_terms", []):
        warnings.append(
            "Rewrote additional_search_terms to remove literal employer/title inventories and preserve semantic guidance."
        )
        draft["additional_search_terms"] = (
            rewritten_terms or current_raw.get("additional_search_terms", [])
        )

    if _run_metrics_are_internally_inconsistent(report):
        risky_fields: list[str] = []
        if mentions_risky_title_translation(draft.get("minimum_bar_description", "")):
            draft["minimum_bar_description"] = current_raw.get("minimum_bar_description", "")
            risky_fields.append("minimum_bar_description")
        depth = draft.get("depth_distinction", {}) or {}
        current_depth = current_raw.get("depth_distinction", {}) or {}
        if mentions_risky_title_translation(depth.get("edge_case_guidance", "")):
            depth["edge_case_guidance"] = current_depth.get("edge_case_guidance", "")
            draft["depth_distinction"] = depth
            risky_fields.append("depth_distinction.edge_case_guidance")
        if mentions_risky_title_translation(draft.get("intake_notes", "")):
            draft["intake_notes"] = current_raw.get("intake_notes", "")
            risky_fields.append("intake_notes")
        if risky_fields:
            warnings.append(
                "Retained current seniority calibration for "
                + ", ".join(risky_fields)
                + " because the run metrics were internally inconsistent and did not justify loosening title translation."
            )


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _apply_iteration_proposal(
    current_raw: dict,
    proposal: dict,
    report_path: Path,
    report: StructuredRunReport,
) -> tuple[dict, list[str]]:
    draft = copy.deepcopy(current_raw)
    warnings: list[str] = []
    changes = proposal.get("proposed_changes", {}) if isinstance(proposal, dict) else {}
    allow_retrieval_design_edits = _raw_has_explicit_retrieval_design(current_raw)
    current_retrieval_design = retrieval_design_from_payload(
        current_raw.get("retrieval_design"),
        legacy_search_priorities=current_raw.get("search_priorities", []),
        legacy_additional_search_terms=current_raw.get("additional_search_terms", []),
        role_title=current_raw.get("role_title", ""),
    )

    explicit_retrieval_edit = "retrieval_design" in changes
    legacy_search_edits = (
        "search_priorities" in changes or "additional_search_terms" in changes
    )

    if "instructions" in changes:
        draft["instructions"] = _dedupe_strings(changes["instructions"], limit=LIST_LIMITS["instructions"])
    if "search_priorities" in changes and not (allow_retrieval_design_edits and not explicit_retrieval_edit):
        draft["search_priorities"] = _dedupe_strings(changes["search_priorities"], limit=LIST_LIMITS["search_priorities"])
    if "additional_search_terms" in changes and not (allow_retrieval_design_edits and not explicit_retrieval_edit):
        draft["additional_search_terms"] = _dedupe_strings(changes["additional_search_terms"], limit=LIST_LIMITS["additional_search_terms"])
    if "retrieval_design" in changes and allow_retrieval_design_edits:
        proposed_design = retrieval_design_from_payload(
            changes.get("retrieval_design"),
            legacy_search_priorities=draft.get("search_priorities", []),
            legacy_additional_search_terms=draft.get("additional_search_terms", []),
            role_title=current_raw.get("role_title", ""),
        )
        retrieval_warnings = validate_retrieval_design(proposed_design)
        warnings.extend(retrieval_warnings)
        draft["retrieval_design"] = proposed_design.to_dict()
    elif "retrieval_design" in changes and not allow_retrieval_design_edits:
        warnings.append(
            "Ignored retrieval_design proposal because this brief has not explicitly opted into layered retrieval editing."
        )
    if "intake_notes" in changes:
        draft["intake_notes"] = _normalize_text(changes["intake_notes"])
    if "depth_distinction" in changes:
        draft["depth_distinction"] = _normalize_depth_distinction(
            changes["depth_distinction"], current_raw.get("depth_distinction", {})
        )
    if "non_fit_patterns" in changes:
        draft["non_fit_patterns"] = _normalize_non_fit_patterns(
            changes["non_fit_patterns"], current_raw.get("non_fit_patterns", [])
        )
    if "minimum_bar_description" in changes:
        # Wave 3 slice 14 (correctness lens): SECOND call site of the
        # strict-seniority legacy guardrail — same gate as the fallback
        # site below, or a proposal that edits the minimum bar re-opens the
        # BFSI/GenAI injection on non-strict briefs.
        proposed_bar = _normalize_text(changes["minimum_bar_description"])
        if is_strict_seniority_brief(current_raw):
            proposed_bar = _ensure_minimum_bar_guardrail(proposed_bar, current_raw)
        draft["minimum_bar_description"] = proposed_bar
    if "facial_calibration" in changes:
        draft["facial_calibration"], clamp_warnings = _normalize_facial_calibration(
            changes["facial_calibration"], current_raw.get("facial_calibration", {})
        )
        warnings.extend(clamp_warnings)
    if "employer_signal_rules" in changes:
        draft["employer_signal_rules"] = _normalize_employer_signal_rules(
            changes["employer_signal_rules"], current_raw.get("employer_signal_rules", [])
        )
    if "calibration_examples" in changes:
        draft["calibration_examples"] = _normalize_calibration_examples(
            changes["calibration_examples"], current_raw.get("calibration_examples", {})
        )
    if "notes" in changes:
        draft["notes"] = _normalize_text(changes["notes"])

    if allow_retrieval_design_edits and not explicit_retrieval_edit:
        effective_retrieval_design = current_retrieval_design
        if legacy_search_edits:
            warnings.append(
                "Ignored direct search_priorities/additional_search_terms edits because this brief is in explicit retrieval_design mode; edit retrieval_design instead."
            )
    elif not explicit_retrieval_edit and legacy_search_edits:
        effective_retrieval_design = retrieval_design_from_payload(
            None,
            legacy_search_priorities=draft.get("search_priorities", []),
            legacy_additional_search_terms=draft.get("additional_search_terms", []),
            role_title=current_raw.get("role_title", ""),
        )
    else:
        effective_retrieval_design = retrieval_design_from_payload(
            draft.get("retrieval_design"),
            legacy_search_priorities=draft.get("search_priorities", []),
            legacy_additional_search_terms=draft.get("additional_search_terms", []),
            role_title=current_raw.get("role_title", ""),
        )
    if effective_retrieval_design.is_empty() and not current_retrieval_design.is_empty():
        effective_retrieval_design = current_retrieval_design
    if allow_retrieval_design_edits or "retrieval_design" in current_raw:
        draft["retrieval_design"] = effective_retrieval_design.to_dict()
    else:
        draft.pop("retrieval_design", None)
    if allow_retrieval_design_edits:
        derived_priorities, derived_terms = derive_legacy_search_views(
            effective_retrieval_design
        )
        draft["search_priorities"] = _dedupe_strings(
            derived_priorities,
            limit=LIST_LIMITS["search_priorities"],
        )
        draft["additional_search_terms"] = _dedupe_strings(
            derived_terms,
            limit=LIST_LIMITS["additional_search_terms"],
        )

    # Wave 3 slice 14: these two guardrails carry strict-seniority legacy
    # vocabulary (BFSI-first, post-2022 GenAI, executive-builder scope) in
    # their fallback text. They must fire ONLY on strict-seniority briefs —
    # unconditional application injected BFS/GenAI hard gates into EVERY
    # iterated brief (the audit's cross-vertical injection class, live).
    # Self-consistent under the gate: is_strict_seniority_brief requires
    # BFS text in the brief, so the BFS fallback describes the brief's own
    # market (same pattern as the strict-seniority lane-rank fallback).
    if is_strict_seniority_brief(current_raw):
        draft["instructions"] = _ensure_hard_gate_instruction(
            draft.get("instructions", current_raw.get("instructions", [])),
            current_raw,
        )
        draft["minimum_bar_description"] = _ensure_minimum_bar_guardrail(
            draft.get("minimum_bar_description", current_raw.get("minimum_bar_description", "")),
            current_raw,
        )

    draft["version"] = _derive_next_draft_version(str(current_raw.get("version", "")))
    generated_note = f"Draft iteration generated from {report_path.name}."
    if draft.get("notes"):
        if generated_note.lower() not in draft["notes"].lower():
            draft["notes"] = f"{draft['notes']} {generated_note}".strip()
    else:
        draft["notes"] = generated_note

    warnings.extend(_find_heuristic_gap_warnings(current_raw, draft))
    # Strict-seniority guardrail runs first when applicable: its rewrites
    # are tighter and its warnings name the seniority context. The general
    # anti-employer-proxy guardrail runs after to catch any remaining
    # employer-inventory items on non-strict briefs (the floor behavior).
    _apply_strict_seniority_legacy_guardrails(
        current_raw=current_raw,
        draft=draft,
        report=report,
        warnings=warnings,
    )
    _apply_general_anti_employer_proxy_guardrails(
        current_raw=current_raw,
        draft=draft,
        warnings=warnings,
    )
    return draft, _dedupe_strings(warnings)


def _validate_draft_brief(draft_raw: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(draft_raw, tmp, indent=2)
        temp_path = Path(tmp.name)
    try:
        load_brief(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


def _render_iteration_rationale(
    source_brief: Path,
    draft_brief: Path,
    report_path: Path,
    current_raw: dict,
    draft_raw: dict,
    proposal: dict,
    warnings: list[str],
) -> str:
    detail_map = {
        item.get("field"): item
        for item in proposal.get("changed_fields", [])
        if isinstance(item, dict) and item.get("field")
    }
    actual_changed = [
        field for field in MUTABLE_FIELDS
        if field in draft_raw and field in current_raw and not _json_equal(current_raw[field], draft_raw[field])
    ]
    lines = [
        f"# Brief Iteration Report: {source_brief.name} → {draft_brief.name}",
        "",
        f"- Source report: {report_path}",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Draft version: {draft_raw.get('version', '')}",
        "",
    ]
    summary = _normalize_text(proposal.get("summary"))
    if summary:
        lines.extend(["## Summary", summary, ""])

    lines.append("## Changed Fields")
    if not actual_changed:
        lines.append("- No mutable fields changed.")
    for field in actual_changed:
        detail = detail_map.get(field, {})
        lines.append(f"- **{field}**")
        why = _normalize_text(detail.get("why"))
        if why:
            lines.append(f"  Why: {why}")
        evidence = _dedupe_strings(detail.get("evidence", []), limit=5)
        if evidence:
            lines.append(f"  Evidence: {'; '.join(evidence)}")
        effects = _dedupe_strings(detail.get("expected_effects", []), limit=5)
        if effects:
            lines.append(f"  Expected effects: {'; '.join(effects)}")
    lines.append("")

    lines.append("## Locked Fields Preserved")
    for field in sorted(LOCKED_FIELDS):
        lines.append(f"- {field}")
    lines.append("")

    lines.append("## Guardrails Applied")
    lines.append("- Locked fields were preserved from the source brief.")
    lines.append("- Search terms and priorities were deduplicated and capped.")
    lines.append("- Employer signal rules were forced to keep `save_on_employer_alone = false`.")
    lines.append("- Facial calibration deltas were clamped before writing the draft.")
    lines.append("- The resulting draft was validated by round-tripping through `load_brief`.")
    lines.append("")

    lines.append("## Warnings")
    combined_warnings = _dedupe_strings(list(proposal.get("warnings", [])) + warnings)
    if not combined_warnings:
        lines.append("- None")
    else:
        for warning in combined_warnings:
            lines.append(f"- {warning}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def iterate_brief_draft(
    brief_path: str | Path,
    report_path: str | None = None,
    search_memory_path: str | None = None,
    final_judgments_path: str | None = None,
    output_dir: str | None = None,
) -> BriefIterationResult:
    brief_path = Path(brief_path)
    if not brief_path.exists():
        raise FileNotFoundError(f"Brief file not found: {brief_path}")

    output_root, resolved_report, resolved_search_memory, resolved_final = _resolve_optional_paths(
        brief_path,
        report_path,
        search_memory_path,
        final_judgments_path,
        output_dir,
    )
    if not resolved_report.exists():
        raise FileNotFoundError(f"Structured run report not found: {resolved_report}")

    current_raw = read_json(brief_path)
    allow_retrieval_design_edits = _raw_has_explicit_retrieval_design(current_raw)
    strict_seniority_legacy = is_strict_seniority_brief(current_raw) and not allow_retrieval_design_edits
    report = StructuredRunReport.from_dict(read_json(resolved_report))
    search_memory = read_json(resolved_search_memory) if resolved_search_memory and resolved_search_memory.exists() else None
    final_summary = _summarize_final_judgments(resolved_final) if resolved_final and resolved_final.exists() else None
    market_intel_summary = _load_market_intel_summary(brief_path, output_root)
    usage_log_path = output_root / "brief-iteration-token-cost-log.jsonl"

    with llm_usage_session(
        usage_log_path,
        pipeline="brief_iteration",
        brief_path=str(brief_path),
        draft_source_version=str(current_raw.get("version", "")),
    ):
        system_prompt = _build_iteration_system(
            allow_retrieval_design_edits=allow_retrieval_design_edits,
            strict_seniority_legacy=strict_seniority_legacy,
        )
        try:
            proposal = opus_llm(
                system_prompt,
                _build_iteration_user_prompt(
                    current_raw,
                    report,
                    search_memory,
                    final_summary,
                    market_intel_summary,
                    allow_retrieval_design_edits=allow_retrieval_design_edits,
                    strict_seniority_legacy=strict_seniority_legacy,
                    tight=False,
                ),
                expect_json=True,
                max_tokens=12000,
                usage_context={"stage": "brief_iteration_proposal", "attempt": "initial"},
            )
        except RuntimeError as exc:
            message = str(exc)
            if not (
                "stop_reason=max_tokens" in message
                or "finish_reason=length" in message
            ):
                raise
            proposal = opus_llm(
                system_prompt,
                _build_iteration_user_prompt(
                    current_raw,
                    report,
                    search_memory,
                    final_summary,
                    market_intel_summary,
                    allow_retrieval_design_edits=allow_retrieval_design_edits,
                    strict_seniority_legacy=strict_seniority_legacy,
                    tight=True,
                ),
                expect_json=True,
                max_tokens=16000,
                usage_context={"stage": "brief_iteration_proposal", "attempt": "tight_retry"},
            )
    if not isinstance(proposal, dict):
        raise ValueError("brief iteration proposal must be a dict")

    draft_raw, warnings = _apply_iteration_proposal(current_raw, proposal, resolved_report, report)
    _validate_draft_brief(draft_raw)

    draft_path = _draft_brief_path(brief_path, str(draft_raw.get("version", "draft")))
    rationale_path = output_root / f"brief-iteration-report-{draft_path.stem}.md"
    _backup_existing(draft_path)
    _backup_existing(rationale_path)
    write_json(draft_path, draft_raw)

    rationale_markdown = _render_iteration_rationale(
        brief_path,
        draft_path,
        resolved_report,
        current_raw,
        draft_raw,
        proposal,
        warnings,
    )
    rationale_path.parent.mkdir(parents=True, exist_ok=True)
    rationale_path.write_text(rationale_markdown)

    return BriefIterationResult(
        draft_brief_path=draft_path,
        rationale_path=rationale_path,
        draft_brief=draft_raw,
        rationale_markdown=rationale_markdown,
        warnings=warnings,
        proposal=proposal,
    )
