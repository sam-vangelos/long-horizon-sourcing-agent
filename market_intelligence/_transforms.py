"""Pure dict-in/dict-out transforms and narrative/lane merge helpers.

Extracted from :mod:`market_intelligence.engine` (Phase 4, slice P4-1) with no
behavior change. These functions have no I/O and no canonical-state contact;
they convert finding/implication records into artifact-shaped dicts and merge
narrative/lane collections. ``engine`` re-imports every name defined here so the
historical ``market_intelligence.engine.<name>`` import paths keep resolving.

The only non-stdlib dependency is on two pure predicates from
:mod:`market_intelligence.schema`; ``engine`` already imports from ``schema`` at
module load, so importing them here introduces no new import cycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from market_intelligence.schema import (
    market_thesis_summary_looks_like_review,
    sanitize_narrative_items,
)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _slugify(value: str) -> str:
    lowered = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    while "__" in lowered:
        lowered = lowered.replace("__", "_")
    return lowered.strip("_") or "unknown"


def _finding_to_external_context(item: dict) -> dict | None:
    claim = _normalize_text(item.get("summary"))
    if not claim:
        return None
    evidence_refs = [
        ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
    ]
    if not evidence_refs:
        return None
    return {
        "claim": claim,
        "label": _normalize_text(item.get("label", "")),
        "evidence_refs": evidence_refs,
        "confidence": float(item.get("confidence", 0.5) or 0.5),
    }


def _finding_to_talent_pool(item: dict) -> dict | None:
    if item.get("kind") not in {
        "title_variant",
        "talent_pool",
        "consulting_overlap",
        "adjacent_archetype",
    }:
        return None
    label = _normalize_text(item.get("label"))
    if not label:
        return None
    evidence_refs = [
        ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
    ]
    supporting_run_refs = [
        ref
        for ref in item.get("supporting_run_refs", [])
        if _normalize_text(ref)
    ]
    return {
        "pool_key": _slugify(label),
        "label": label,
        "status": "core_pool" if item.get("kind") == "talent_pool" else "adjacent_pool",
        "signal_strength": round(float(item.get("confidence", 0.6) or 0.6), 2),
        "supporting_run_refs": supporting_run_refs,
        "evidence_refs": evidence_refs,
        "evidence_summary": _normalize_text(item.get("summary"))
        or _normalize_text(item.get("why_it_matters")),
        "recommended_search_terms": [label],
    }


def _edge_case_submarket_to_external_context(item: dict) -> dict | None:
    label = _normalize_text(item.get("label"))
    summary = _normalize_text(item.get("summary"))
    why_missed = _normalize_text(item.get("why_it_is_easy_to_miss"))
    evidence_refs = [
        ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
    ]
    if not (label and summary and evidence_refs):
        return None
    claim = f"Hidden pool: {label}. {summary}"
    if why_missed:
        claim = f"{claim} Why this may be missed: {why_missed}"
    return {
        "claim": claim,
        "evidence_refs": evidence_refs,
        "confidence": float(item.get("confidence", 0.55) or 0.55),
    }


def _false_negative_hypothesis_to_external_context(item: dict) -> dict | None:
    statement = _normalize_text(item.get("statement"))
    why_it_matters = _normalize_text(item.get("why_it_matters"))
    evidence_refs = [
        ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
    ]
    if not (statement and why_it_matters and evidence_refs):
        return None
    return {
        "claim": f"{statement} {why_it_matters}",
        "evidence_refs": evidence_refs,
        "confidence": float(item.get("confidence", 0.5) or 0.5),
    }


def _edge_case_submarket_to_talent_pool(item: dict) -> dict | None:
    label = _normalize_text(item.get("label"))
    if not label:
        return None
    supporting_run_refs = [
        ref for ref in item.get("supporting_run_refs", []) if _normalize_text(ref)
    ]
    evidence_refs = [
        ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
    ]
    return {
        "pool_key": _normalize_text(item.get("submarket_key")) or _slugify(label),
        "label": label,
        "status": "adjacent_pool",
        "signal_strength": round(float(item.get("confidence", 0.55) or 0.55), 2),
        "supporting_run_refs": supporting_run_refs,
        "evidence_refs": evidence_refs,
        "evidence_summary": _normalize_text(item.get("summary"))
        or _normalize_text(item.get("why_it_is_easy_to_miss")),
        "recommended_search_terms": [label],
    }


def _title_mapping_to_talent_pool(item: dict) -> dict | None:
    title_family = _normalize_text(item.get("title_family"))
    likely_archetype = _normalize_text(item.get("likely_archetype"))
    caveats = _normalize_text(item.get("caveats"))
    if not (title_family and likely_archetype):
        return None
    supporting_run_refs = [
        ref for ref in item.get("supporting_run_refs", []) if _normalize_text(ref)
    ]
    evidence_refs = [
        ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
    ]
    summary = caveats if caveats else ""
    return {
        "pool_key": _normalize_text(item.get("mapping_key"))
        or _slugify(f"{title_family}-{likely_archetype}"),
        "label": title_family,
        "status": "adjacent_pool",
        "signal_strength": 0.58,
        "supporting_run_refs": supporting_run_refs,
        "evidence_refs": evidence_refs,
        "evidence_summary": summary,
        "recommended_search_terms": [title_family],
    }


def _self_presentation_pattern_to_talent_pool(item: dict) -> dict | None:
    label = _normalize_text(item.get("label"))
    pattern = _normalize_text(item.get("pattern"))
    why_false_negative = _normalize_text(item.get("why_it_causes_false_negatives"))
    if not (label and pattern):
        return None
    supporting_run_refs = [
        ref for ref in item.get("supporting_run_refs", []) if _normalize_text(ref)
    ]
    evidence_refs = [
        ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
    ]
    summary = pattern
    if why_false_negative:
        summary = f"{summary} {why_false_negative}"
    return {
        "pool_key": _normalize_text(item.get("pattern_key")) or _slugify(label),
        "label": label,
        "status": "adjacent_pool",
        "signal_strength": 0.52,
        "supporting_run_refs": supporting_run_refs,
        "evidence_refs": evidence_refs,
        "evidence_summary": summary,
        "recommended_search_terms": [label],
    }


def _finding_to_employer_signal(item: dict) -> dict | None:
    if item.get("kind") != "employer_cluster":
        return None
    label = _normalize_text(item.get("label"))
    if not label:
        return None
    evidence_refs = [
        ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
    ]
    supporting_run_refs = [
        ref
        for ref in item.get("supporting_run_refs", [])
        if _normalize_text(ref)
    ]
    return {
        "cluster_key": _slugify(label),
        "label": label,
        "status": "positive",
        "supporting_employers": [label],
        "supporting_run_refs": supporting_run_refs,
        "evidence_refs": evidence_refs,
        "evidence_summary": _normalize_text(item.get("summary"))
        or _normalize_text(item.get("why_it_matters")),
        "confidence": round(float(item.get("confidence", 0.6) or 0.6), 2),
    }


def _implication_to_brief_recommendation(item: dict) -> dict | None:
    category = _normalize_text(item.get("category"))
    recommendation = _normalize_text(item.get("recommendation"))
    rationale = _normalize_text(item.get("rationale"))
    if not (category and recommendation and rationale):
        return None
    target_field = _normalize_text(item.get("brief_target_field"))
    if not target_field:
        target_field = {
            "add_title_family": "additional_search_terms",
            "add_employer_target": "employer_signal_rules",
            "probe_adjacent_pool": "search_priorities",
            "relax_boolean": "instructions",
            "validate_hypothesis": "instructions",
            "instrumentation_followup": "notes",
        }.get(category, "instructions")
    suggested_values = [
        value for value in item.get("suggested_values", []) if _normalize_text(value)
    ]
    proposal = ", ".join(suggested_values) if suggested_values else recommendation
    retrieval_update = _implication_to_retrieval_update(item, target_field=target_field)
    recommendation_record = {
        "recommendation_id": _normalize_text(item.get("implication_id"))
        or _slugify(f"{category}-{proposal}"),
        "target_field": target_field,
        "proposal": proposal,
        "reason": rationale,
        "supporting_run_refs": [
            ref
            for ref in item.get("supporting_run_refs", [])
            if _normalize_text(ref)
        ],
        "evidence_refs": [
            ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
        ],
        "confidence": round(
            0.55
            + {
                "high": 0.25,
                "medium": 0.15,
                "low": 0.05,
            }.get(_normalize_text(item.get("priority")).lower(), 0.15),
            2,
        ),
    }
    if retrieval_update:
        recommendation_record["retrieval_update"] = retrieval_update
    return recommendation_record


def _implication_to_planner_diff(item: dict) -> dict | None:
    """Convert a sourcing implication into a PlannerDiff record.

    Returns None for implications that don't compile to a diff type.
    Non-compilable recommendations are dropped from planner output.
    """
    category = _normalize_text(item.get("category"))
    recommendation = _normalize_text(item.get("recommendation"))
    rationale = _normalize_text(item.get("rationale"))
    if not (category and recommendation):
        return None

    # Map category to diff action + target_type
    CATEGORY_MAP: dict[str, tuple[str, str]] = {
        "add_title_family": ("add", "constraint"),
        "add_employer_target": ("add", "constraint"),
        "probe_adjacent_pool": ("add", "hypothesis"),
        "relax_boolean": ("update", "constraint"),
        "validate_hypothesis": ("add", "validation_question"),
        "instrumentation_followup": ("add", "validation_question"),
    }
    mapped = CATEGORY_MAP.get(category)
    if not mapped:
        return None

    action, target_type = mapped
    suggested_values = [
        value for value in item.get("suggested_values", []) if _normalize_text(value)
    ]
    diff_id = (
        _normalize_text(item.get("implication_id"))
        or _slugify(f"{category}-{recommendation[:40]}")
    )
    supporting_run_refs = [
        ref for ref in item.get("supporting_run_refs", []) if _normalize_text(ref)
    ]
    evidence_refs = [
        ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
    ]
    confidence = round(
        0.55
        + {"high": 0.25, "medium": 0.15, "low": 0.05}.get(
            _normalize_text(item.get("priority")).lower(), 0.15
        ),
        2,
    )

    payload: dict[str, Any] = {"recommendation": recommendation}
    if suggested_values:
        payload["values"] = suggested_values
    if category in ("add_title_family",):
        payload["dimension"] = "title_family"
    elif category in ("add_employer_target",):
        payload["dimension"] = "employer_target"
    if target_type == "validation_question":
        payload["question"] = recommendation
    if rationale:
        payload["rationale"] = rationale

    return {
        "diff_id": diff_id,
        "action": action,
        "target_type": target_type,
        "target_id": "",
        "payload": payload,
        "internal_evidence": supporting_run_refs,
        "external_evidence": evidence_refs,
        "confidence": confidence,
    }


def _merge_planner_diffs(previous: list[dict], current: list[dict]) -> list[dict]:
    merged = {
        str(item.get("diff_id")): dict(item)
        for item in previous
        if isinstance(item, dict) and item.get("diff_id")
    }
    for item in current:
        if isinstance(item, dict) and item.get("diff_id"):
            merged[str(item["diff_id"])] = dict(item)
    return list(merged.values())


PLANNER_DIFF_EXPIRY_RUNS = 3


def _apply_planner_diff_lifecycle(
    previous: list[dict],
    current: list[dict],
    previously_archived: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """P3.3: merge planner diffs WITH a lifecycle instead of accreting forever.

    - A diff re-emitted this update (same diff_id in ``current``) refreshes:
      its unconsumed-run counter resets.
    - A previous diff not re-emitted and not consumed ages by one run; at
      PLANNER_DIFF_EXPIRY_RUNS it expires and moves to the archive (kept,
      never deleted).
    - Consumed diffs (marked by mark_planner_diffs_consumed) are retained one
      cycle for audit, then archived; they are never re-served (the loader
      filters them).

    Returns (active_diffs, archived_diffs).
    """

    archived: list[dict] = [
        dict(item) for item in (previously_archived or []) if isinstance(item, dict)
    ]
    current_by_id = {
        str(item.get("diff_id")): dict(item)
        for item in current
        if isinstance(item, dict) and item.get("diff_id")
    }
    active: dict[str, dict] = {}

    for item in previous:
        if not (isinstance(item, dict) and item.get("diff_id")):
            continue
        diff_id = str(item["diff_id"])
        entry = dict(item)
        if diff_id in current_by_id:
            # Refreshed: the new emission wins, counter resets; a consumed
            # marker survives the refresh (never re-serve a consumed diff).
            refreshed = current_by_id.pop(diff_id)
            if entry.get("consumed"):
                refreshed["consumed"] = True
                refreshed["consumed_at"] = entry.get("consumed_at")
            refreshed["runs_unconsumed"] = 0
            active[diff_id] = refreshed
            continue
        if entry.get("consumed"):
            entry["archived_reason"] = "consumed"
            archived.append(entry)
            continue
        entry["runs_unconsumed"] = int(entry.get("runs_unconsumed", 0) or 0) + 1
        if entry["runs_unconsumed"] >= PLANNER_DIFF_EXPIRY_RUNS:
            entry["archived_reason"] = "expired_unconsumed"
            archived.append(entry)
            continue
        active[diff_id] = entry

    for diff_id, entry in current_by_id.items():
        entry.setdefault("runs_unconsumed", 0)
        active[diff_id] = entry

    return list(active.values()), archived


def _implication_to_retrieval_update(
    item: dict,
    *,
    target_field: str,
) -> dict[str, Any] | None:
    category = _normalize_text(item.get("category"))
    suggested_values = [
        value for value in item.get("suggested_values", []) if _normalize_text(value)
    ]
    if not category:
        return None
    layer_name = {
        "add_title_family": "entry_signals",
        "probe_adjacent_pool": "capability_proxies",
        "relax_boolean": "reality_filters",
        "add_employer_target": "target_employers",
    }.get(category, "")
    update_type = {
        "validate_hypothesis": "hypothesis_validation",
        "instrumentation_followup": "instrumentation_followup",
    }.get(category, "layer_update")
    record = {
        "update_type": update_type,
        "category": category,
        "target_field": target_field,
        "layer_name": layer_name,
        "suggested_values": suggested_values,
        "reason": _normalize_text(item.get("rationale")),
        "expected_effect": _normalize_text(item.get("expected_effect")),
    }
    if category == "probe_adjacent_pool":
        record["edge_case_hypothesis"] = {
            "label": _normalize_text(item.get("recommendation")),
            "hidden_cohort": _normalize_text(item.get("recommendation")),
            "why_missed": _normalize_text(item.get("rationale")),
            "validation_rule": "Promote only after repeated signal across later runs.",
            "source": "market_intel",
        }
    return record


def _merge_lane_entries(
    *,
    deterministic_lanes: list[dict],
    generated_lanes: list[dict],
    previous_lanes: list[dict],
) -> list[dict]:
    generated_by_key = {
        lane.get("lane_key"): lane
        for lane in generated_lanes
        if isinstance(lane, dict)
    }
    previous_by_key = {
        lane.get("lane_key"): lane
        for lane in previous_lanes
        if isinstance(lane, dict)
    }
    merged: list[dict] = []
    for lane in deterministic_lanes:
        lane_key = lane["lane_key"]
        previous = previous_by_key.get(lane_key, {})
        generated = generated_by_key.get(lane_key, {})
        merged_lane = dict(lane)
        merged_lane["supporting_run_refs"] = sorted(
            set(previous.get("supporting_run_refs", []))
            | set(lane.get("supporting_run_refs", []))
        )
        merged_lane["why_it_works"] = _normalize_text(
            generated.get("why_it_works")
        ) or _normalize_text(previous.get("why_it_works")) or _default_lane_why(lane)
        merged_lane["recommended_action"] = _normalize_text(
            generated.get("recommended_action")
        ) or _normalize_text(previous.get("recommended_action")) or _default_lane_action(
            lane
        )
        merged_lane["confidence"] = float(
            generated.get("confidence")
            or previous.get("confidence")
            or _default_lane_confidence(lane)
        )
        merged.append(merged_lane)
    return merged


NARRATIVE_DECAY_RUNS = 5


def _merge_narrative_collection(
    *,
    key_field: str,
    current: list[dict],
    previous: list[dict],
    preserve_previous: bool,
) -> list[dict]:
    active, _archived = _merge_narrative_collection_with_decay(
        key_field=key_field,
        current=current,
        previous=previous,
        preserve_previous=preserve_previous,
    )
    return active


def _merge_narrative_collection_with_decay(
    *,
    key_field: str,
    current: list[dict],
    previous: list[dict],
    preserve_previous: bool,
    supported_run_ref: str = "",
) -> tuple[list[dict], list[dict]]:
    """P3.5: union-merge WITH decay instead of accreting forever.

    An entry touched by this update's evidence (present in ``current``) is
    stamped ``last_supported_run``/``last_supported_at`` and its
    ``runs_unsupported`` counter resets. A previous-only entry ages by one
    run; at NARRATIVE_DECAY_RUNS it moves to the archived block (excluded
    from prompt/context rendering, never deleted).
    """

    now = datetime.now(timezone.utc).isoformat()
    archived: list[dict] = []
    merged: dict[Any, dict] = {}
    current_keys = {
        item.get(key_field)
        for item in (current or [])
        if isinstance(item, dict) and item.get(key_field)
    }
    for item in previous:
        if not (isinstance(item, dict) and item.get(key_field)):
            continue
        key = item[key_field]
        if key in current_keys:
            continue  # the refreshed emission below wins
        entry = dict(item)
        entry["runs_unsupported"] = int(entry.get("runs_unsupported", 0) or 0) + 1
        if entry["runs_unsupported"] >= NARRATIVE_DECAY_RUNS:
            entry["archived_reason"] = "unsupported_decay"
            archived.append(entry)
            continue
        merged[key] = entry
    if current:
        for item in current:
            if isinstance(item, dict) and item.get(key_field):
                entry = dict(item)
                entry["runs_unsupported"] = 0
                entry["last_supported_at"] = now
                if supported_run_ref:
                    entry["last_supported_run"] = supported_run_ref
                merged[item[key_field]] = entry
    elif not preserve_previous:
        merged = {}
    return list(merged.values()), archived


def _merge_market_thesis(
    *,
    current: dict,
    previous: dict,
    preserve_previous: bool,
) -> dict:
    if current:
        merged = dict(current)
        summary = _normalize_text(merged.get("summary"))
        if market_thesis_summary_looks_like_review(summary):
            merged["summary"] = ""
        if previous:
            previous_summary = _normalize_text(previous.get("summary"))
            if (
                not _normalize_text(merged.get("summary"))
                or market_thesis_summary_looks_like_review(merged.get("summary"))
            ) and previous_summary and not market_thesis_summary_looks_like_review(previous_summary):
                merged["summary"] = previous_summary
            if not _normalize_text(merged.get("supply_assessment")) or _normalize_text(
                merged.get("supply_assessment")
            ) == "unknown":
                merged["supply_assessment"] = previous.get(
                    "supply_assessment",
                    "unknown",
                )
            if not _normalize_text(merged.get("competition_assessment")) or _normalize_text(
                merged.get("competition_assessment")
            ) == "unknown":
                merged["competition_assessment"] = previous.get(
                    "competition_assessment",
                    "unknown",
                )
            if not sanitize_narrative_items(
                "market_thesis.external_context",
                merged.get("external_context", []),
            ):
                merged["external_context"] = previous.get("external_context", [])
        return merged
    if preserve_previous and previous:
        return previous
    return {
        "summary": "Market thesis not yet synthesized.",
        "supply_assessment": "unknown",
        "competition_assessment": "unknown",
        "external_context": [],
    }


def _default_lane_why(lane: dict) -> str:
    if lane.get("status") == "winning":
        return "This lane is repeatedly producing saved candidates."
    if lane.get("status") == "noise":
        return "This lane is producing weak signal relative to review effort."
    if lane.get("status") == "saturated":
        return "This lane is generating high overlap and appears increasingly exhausted."
    return "This lane is producing mixed results so far."


def _default_lane_action(lane: dict) -> str:
    if lane.get("status") == "winning":
        return "Keep this lane active and expand nearby variants."
    if lane.get("status") == "noise":
        return "Tighten the lane or retire it."
    if lane.get("status") == "saturated":
        return "De-prioritize this lane and look for adjacent whitespace."
    return "Keep monitoring before promoting or retiring this lane."


def _default_lane_confidence(lane: dict) -> float:
    metrics = lane.get("metrics", {})
    if lane.get("status") == "winning":
        return 0.8 if metrics.get("saves", 0) >= 3 else 0.7
    if lane.get("status") == "noise":
        return 0.72
    if lane.get("status") == "saturated":
        return 0.75
    return 0.6
