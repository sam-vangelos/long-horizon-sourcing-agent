"""Orchestrator-facing fallback discovery — P10 wiring."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from linkedin.fallback_search import (
    FallbackCandidate,
    FallbackQuery,
    FallbackSearchProvider,
    build_xray_query,
    fallback_result_to_candidate,
)
from shared.schemas import ExecutionPlan, SearchString
from shared.sourcing_lanes import FALLBACK_ACQUISITION_MODES, SourcingLane, normalize_lane_id


@runtime_checkable
class FallbackDiscoveryRecorder(Protocol):
    def __call__(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
    ) -> None: ...


def sourcing_lane_for_search_string(
    plan: ExecutionPlan | None,
    search_string: SearchString,
) -> SourcingLane | None:
    """Resolve a sourcing lane envelope for the active search string."""
    if plan is None:
        return None
    lane_id = normalize_lane_id(search_string.lane_id or search_string.family_key or "")
    if not lane_id:
        return None
    for lane_dict in getattr(plan, "sourcing_lanes", None) or []:
        if not isinstance(lane_dict, dict):
            continue
        if normalize_lane_id(str(lane_dict.get("lane_id", ""))) == lane_id:
            return SourcingLane.from_dict(lane_dict)
    return None


def fallback_mode_for_search_string(
    plan: ExecutionPlan | None,
    search_string: SearchString,
) -> str:
    mode = str(search_string.acquisition_mode or "").strip()
    if mode in FALLBACK_ACQUISITION_MODES:
        return mode
    lane = sourcing_lane_for_search_string(plan, search_string)
    if lane is None:
        return ""
    return str(lane.execution.acquisition_mode or "").strip()


def discover_fallback_candidates_for_string(
    plan: ExecutionPlan | None,
    search_string: SearchString,
    *,
    provider: FallbackSearchProvider | None = None,
    limit: int = 5,
) -> tuple[list[FallbackCandidate], FallbackQuery | None]:
    """Run fallback discovery when the lane uses a fallback acquisition mode."""
    lane = sourcing_lane_for_search_string(plan, search_string)
    if lane is None:
        return [], None

    mode = fallback_mode_for_search_string(plan, search_string)
    if mode not in FALLBACK_ACQUISITION_MODES:
        return [], None

    query = build_xray_query(lane)
    if mode == "linkedin_public":
        query = FallbackQuery(
            query_string=query.query_string,
            lane_id=query.lane_id,
            variant_id=query.variant_id,
            source_mode="linkedin_public",
            constraints_applied=query.constraints_applied,
        )

    if provider is None:
        return [], query

    variant_id = str(search_string.lane_snapshot.get("variant_id", "") or "").strip() or None
    results = provider.search(query, limit=limit)
    candidates = [
        fallback_result_to_candidate(
            result,
            lane_id=lane.lane_id,
            variant_id=variant_id,
        )
        for result in results
    ]
    return candidates, query


def record_fallback_discovery(
    *,
    candidates: list[FallbackCandidate],
    query: FallbackQuery | None,
    search_string: SearchString,
    trigger_reason: str,
    record_event: FallbackDiscoveryRecorder | None = None,
    append_candidate: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Persist fallback candidates and emit runtime events without save side effects."""
    base_payload = {
        "string_id": search_string.id,
        "lane_id": search_string.lane_id,
        "trigger_reason": trigger_reason,
        "candidate_count": len(candidates),
    }
    if query is not None:
        base_payload["query_string"] = query.query_string
        base_payload["source_mode"] = query.source_mode

    if record_event is not None:
        record_event(event_type="fallback_discovery_attempt", payload=base_payload)

    for candidate in candidates:
        if append_candidate is not None:
            append_candidate(candidate.to_persistence_dict())
        if record_event is not None:
            record_event(
                event_type="fallback_candidate_discovered",
                payload={
                    **base_payload,
                    "source_url": candidate.source_url,
                    "resolution_state": candidate.resolution_state,
                },
            )
