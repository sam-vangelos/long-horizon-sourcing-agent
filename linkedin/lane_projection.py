"""Lane-projection helpers for queue building.

Extracted from :mod:`linkedin.orchestrator` (Phase 4, slice P4-3) with no
behavior change. These functions normalize lane aliases, lift compiler snapshots
onto flat work units, and match generated strings / coverage gaps to structured
hybrid lanes. ``orchestrator.Pipeline`` thin-delegates every name defined here.
"""

from __future__ import annotations

from typing import Any

from linkedin.search_intelligence import LinkedInStructuredFilters
from shared.schemas import ExecutionPlan
from shared.sourcing_lanes import SourcingLane, derive_search_posture, normalize_lane_id


def lane_projection_aliases(lane_id: str) -> set[str]:
    normalized = normalize_lane_id(lane_id)
    if not normalized:
        return set()
    aliases = {normalized}
    for prefix in ("fam_", "fde_"):
        if normalized.startswith(prefix):
            aliases.add(normalized[len(prefix):])
    return {alias for alias in aliases if alias}


def work_item_lane_key(item: dict[str, Any]) -> str:
    recipe = item.get("retrieval_recipe") if isinstance(item.get("retrieval_recipe"), dict) else {}
    return normalize_lane_id(
        str(
            item.get("lane_id")
            or item.get("family_key")
            or recipe.get("family_id")
            or ""
        )
    )


def lane_snapshot_filters(snapshot: dict[str, Any]) -> LinkedInStructuredFilters:
    query_payload = snapshot.get("query_payload")
    query_payload = query_payload if isinstance(query_payload, dict) else {}
    return LinkedInStructuredFilters.from_dict(query_payload.get("structured_filters"))


def current_lane_compiler_snapshot(lane_dict: dict[str, Any]) -> dict[str, Any]:
    snapshot = lane_dict.get("lane_compiler")
    snapshot = dict(snapshot) if isinstance(snapshot, dict) else {}
    query_payload = snapshot.get("query_payload")
    query_payload = query_payload if isinstance(query_payload, dict) else {}

    # Prefer an existing snapshot when it is complete. Older plans can carry a
    # hybrid snapshot with filters but no keyword Boolean; recompile those at
    # queue time so lane-first execution still keeps a keyword fallback.
    if snapshot and str(query_payload.get("boolean") or "").strip():
        return snapshot

    try:
        from linkedin.lane_compiler import LinkedInLaneCompiler

        lane = SourcingLane.from_dict(lane_dict)
        executable = LinkedInLaneCompiler().compile(lane)
        return {
            "acquisition_mode": executable.acquisition_mode,
            "search_posture": snapshot.get("search_posture")
            or derive_search_posture(lane.slice.constraints),
            "query_payload": executable.query_payload,
            "unsupported_dimensions": list(executable.unsupported_dimensions),
            "lint": list(snapshot.get("lint") or []),
        }
    except Exception:
        return snapshot


def structured_lane_projection_records(
    execution_plan: ExecutionPlan | None,
) -> list[dict[str, Any]]:
    if not execution_plan:
        return []
    records: list[dict[str, Any]] = []
    for lane_dict in execution_plan.sourcing_lanes or []:
        if not isinstance(lane_dict, dict):
            continue
        lane_id = normalize_lane_id(str(lane_dict.get("lane_id") or ""))
        if not lane_id:
            continue
        snapshot = current_lane_compiler_snapshot(lane_dict)
        filters = lane_snapshot_filters(snapshot)
        if snapshot.get("acquisition_mode") != "linkedin_hybrid" or filters.is_empty():
            continue
        lane_intent = str(
            lane_dict.get("lane_intent")
            or (lane_dict.get("slice") or {}).get("objective")
            or ""
        ).strip()
        records.append(
            {
                "lane_id": lane_id,
                "lane_name": str(lane_dict.get("lane_name") or lane_id),
                "lane_intent": lane_intent,
                "snapshot": snapshot,
                "aliases": lane_projection_aliases(lane_id),
            }
        )
    return records


def match_lane_projection(
    item: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    key = work_item_lane_key(item)
    if not key:
        return None
    for record in records:
        if key in record["aliases"]:
            return record
    for record in records:
        for alias in record["aliases"]:
            if key.startswith(f"{alias}_"):
                return record
    return None


def apply_lane_projection_to_work_item(
    item: dict[str, Any],
    record: dict[str, Any],
    *,
    boolean_key: str,
) -> dict[str, Any]:
    projected = dict(item)
    snapshot = dict(record["snapshot"])
    projected["lane_id"] = record["lane_id"]
    projected["lane_name"] = record["lane_name"]
    projected["lane_intent"] = record["lane_intent"]
    projected["acquisition_mode"] = snapshot.get("acquisition_mode") or projected.get(
        "acquisition_mode",
        "",
    )
    lane_snapshot = dict(projected.get("lane_snapshot") or {})
    lane_snapshot["compiler"] = snapshot
    projected["lane_snapshot"] = lane_snapshot
    if not projected.get("surface"):
        projected["surface"] = (
            "hybrid"
            if str(projected.get(boolean_key) or "").strip()
            else "structured_only"
        )
    return projected
