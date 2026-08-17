"""LinkedIn progress synchronization helpers."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from linkedin.search_intelligence import LinkedInExperimentState, bootstrap_experiment_state
from shared.runtime_state.lane_metrics import LEGACY_LANE_ID
from shared.schemas import Progress, SearchString

from .store import LINKEDIN_STRING_KIND, RuntimeStateStore


logger = logging.getLogger(__name__)


def lane_cost_from_usage_log(log_path: str | Path | None) -> dict[str, float]:
    """Sum ``estimated_cost_usd`` per ``lane_id`` from a usage JSONL log.

    The per-call usage records written by
    :func:`shared.llm_usage.record_llm_usage` already tag each row with
    the lane context (``lane_id`` / ``variant_id`` / ``stage``) and an
    ``estimated_cost_usd``. This rolls those rows up into a
    ``{lane_id: total_cost}`` map keyed by ``lane_id`` exactly as the
    work-unit write path attributes lanes — rows with an empty / missing
    ``lane_id`` collapse to :data:`LEGACY_LANE_ID`, matching how
    ``lane_metrics`` buckets unattributed work units.

    Fail-soft: a missing file, malformed line, or non-numeric cost
    contributes nothing and never raises. Cost is observability; its
    failure must not abort a progress sync.
    """

    if not log_path:
        return {}
    path = Path(log_path)
    if not path.exists():
        return {}
    by_lane: dict[str, float] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        cost = record.get("estimated_cost_usd")
        if not isinstance(cost, (int, float)):
            continue
        raw_lane = record.get("lane_id")
        lane_id = raw_lane.strip() if isinstance(raw_lane, str) and raw_lane.strip() else LEGACY_LANE_ID
        by_lane[lane_id] = by_lane.get(lane_id, 0.0) + float(cost)
    return by_lane


def sync_linkedin_progress(
    *,
    store: RuntimeStateStore,
    run_id: int,
    brief_id: str,
    progress: Progress,
    experiment_states: dict[int, LinkedInExperimentState] | None = None,
    rebuild_artifacts: Callable[[int], None],
    work_unit_metrics: Callable[[SearchString], dict[str, Any]],
    lane_cost_usd: Mapping[str, float] | None = None,
    timings: dict[str, float] | None = None,
    validate_status: bool = True,
) -> None:
    # Per-lane LLM cost is attributed to exactly one work unit per lane so
    # the read model (``lane_metrics._coerce_cost`` sums ``cost_usd`` across
    # every work unit in a lane) totals to the true per-lane cost rather
    # than multiplying it by the number of search strings sharing the lane.
    lane_cost_usd = dict(lane_cost_usd or {})
    cost_attributed_lanes: set[str] = set()

    keep_ids: set[str] = set()
    rewrite_started = time.monotonic()
    try:
        with store.connect() as conn:
            store.update_run_resume_state(
                run_id,
                _build_resume_state(progress),
                conn=conn,
            )
            for index, search_string in enumerate(progress.strings):
                keep_ids.add(str(search_string.id))
                experiment_state = (experiment_states or {}).get(search_string.id)
                if experiment_state is None:
                    experiment_state = bootstrap_experiment_state(search_string)
                experiment_state.apply_shadow(search_string)
                experiment_summary = experiment_state.metrics_summary()
                metrics = work_unit_metrics(search_string)
                metrics.update(
                    {
                        "experiment_summary": experiment_summary,
                        "variant_metrics": experiment_summary.get("variants", {}),
                    }
                )
                lane_key = (search_string.lane_id or "").strip() or LEGACY_LANE_ID
                if lane_key in lane_cost_usd and lane_key not in cost_attributed_lanes:
                    metrics["cost_usd"] = lane_cost_usd[lane_key]
                    cost_attributed_lanes.add(lane_key)
                counters = {
                    "result_count": search_string.result_count,
                    "candidates_discovered": search_string.candidates_count,
                    "facial_yes_count": search_string.facial_yes_count,
                    "facial_no_count": search_string.facial_no_count,
                    "facial_borderline_count": search_string.facial_borderline_count,
                    "saves_count": len(search_string.saves),
                    "rejected_count": search_string.full_reject_count,
                }
                store.upsert_work_unit(
                    run_id=run_id,
                    source="linkedin",
                    brief_id=brief_id,
                    kind=LINKEDIN_STRING_KIND,
                    source_unit_id=str(search_string.id),
                    display_name=search_string.name,
                    ordering_index=index,
                    status=search_string.status,
                    validate_status=validate_status,
                    payload={
                        **search_string.to_dict(),
                        "search_intent": experiment_state.intent.to_dict(),
                    },
                    checkpoint={
                        "pages_reviewed": search_string.pages_reviewed,
                        "duplicates_count": search_string.duplicates_count,
                        "phase": search_string.phase,
                        "refinement_stack": list(search_string.refinement_stack),
                        "experiment_state": experiment_state.to_dict(),
                    },
                    metrics=metrics,
                    family_key=search_string.family_key,
                    novelty_bucket=search_string.novelty_bucket,
                    domain_lane=search_string.domain_lane,
                    counters=counters,
                    notes=search_string.notes or "",
                    conn=conn,
                )
            store.delete_missing_work_units(
                run_id,
                kind=LINKEDIN_STRING_KIND,
                keep_source_unit_ids=keep_ids,
                conn=conn,
            )
    finally:
        if timings is not None:
            timings["work_unit_rewrite_ms"] = round(
                (time.monotonic() - rewrite_started) * 1000.0,
                3,
            )
    projection_started = time.monotonic()
    try:
        rebuild_artifacts(run_id)
    except Exception as exc:
        # JSON projections are compatibility artifacts. All authoritative
        # resume/work-unit rows above are already committed in SQLite, so a
        # projection failure cannot reverse the canonical checkpoint.
        logger.warning(
            "LinkedIn projection rebuild failed after canonical sync (%s)",
            type(exc).__name__,
        )
    finally:
        if timings is not None:
            timings["projection_rebuild_ms"] = round(
                (time.monotonic() - projection_started) * 1000.0,
                3,
            )


def _build_resume_state(progress: Progress) -> dict[str, Any]:
    return {
        "brief_name": progress.brief_name,
        "current_string_id": progress.current_string_id,
        "current_page": progress.current_page,
        "pending_block_name": progress.pending_block_name,
        "pending_block_string_ids": list(progress.pending_block_string_ids),
        "pending_block_ready": progress.pending_block_ready,
        "candidates_saved": progress.candidates_saved,
        "candidates_rejected": progress.candidates_rejected,
        "pivot_count": progress.pivot_count,
    }
