"""Apply LinkedIn lane compiler output to strategy plans — P9 wiring."""

from __future__ import annotations

from typing import Any

from linkedin.lane_compiler import LinkedInLaneCompiler
from shared.schemas import ExecutionPlan
from shared.sourcing_lanes import (
    SourcingLane,
    derive_search_posture,
    normalize_lane_id,
)


def apply_linkedin_lane_compiler_to_plan(plan: ExecutionPlan) -> int:
    """Compile sourcing lanes and attach acquisition metadata to generated strings.

    Returns the number of generated strings that received compiler metadata.
    """
    sourcing_lanes = getattr(plan, "sourcing_lanes", None) or []
    if not sourcing_lanes:
        return 0

    compiler = LinkedInLaneCompiler()
    lane_compiler_by_id: dict[str, dict[str, Any]] = {}

    for lane_dict in sourcing_lanes:
        if not isinstance(lane_dict, dict):
            continue
        lane = SourcingLane.from_dict(lane_dict)
        executable = compiler.compile(lane)
        lint_findings = compiler.lint(lane)
        snapshot = {
            "acquisition_mode": executable.acquisition_mode,
            # Slice A part 5: telemetry only — the derived posture records the lane's
            # surface composition. It MUST NOT feed acquisition_mode (set above straight
            # from the compiler, which derives it from structured.is_empty()).
            "search_posture": derive_search_posture(lane.slice.constraints),
            "query_payload": executable.query_payload,
            "unsupported_dimensions": list(executable.unsupported_dimensions),
            "lint": [
                {
                    "code": finding.code,
                    "severity": finding.severity,
                    "message": finding.message,
                    "dimension": finding.dimension,
                }
                for finding in lint_findings
            ],
        }
        lane_dict["lane_compiler"] = snapshot
        lane_compiler_by_id[normalize_lane_id(lane.lane_id)] = snapshot

        # Honesty: reconcile the stored execution view with the compiled truth.
        # lane_execution_from_retrieval_family (sourcing_lanes.py) freezes
        # acquisition_mode/search_posture at the mint defaults; the compiler is the
        # authority (it derives hybrid from folded structured filters). Without this,
        # a hybrid lane keeps execution.acquisition_mode='linkedin_boolean' and any
        # audit reading execution.* mis-reads it as Boolean. Consumers already run off
        # the snapshot (lane_projection.current_lane_compiler_snapshot), so this is an
        # observability fix, not a behavior change. A boolean-led lane keeps
        # 'linkedin_boolean'/'boolean_led' (byte-identical default).
        execution = lane_dict.get("execution")
        if isinstance(execution, dict):
            execution["acquisition_mode"] = snapshot["acquisition_mode"]
            execution["search_posture"] = snapshot["search_posture"]

    wired = 0
    for item in getattr(plan, "generated_strings", []) or []:
        if not isinstance(item, dict):
            continue
        lane_id = normalize_lane_id(str(item.get("lane_id") or item.get("family_key") or ""))
        snapshot = lane_compiler_by_id.get(lane_id)
        if not snapshot:
            continue
        item["acquisition_mode"] = snapshot.get("acquisition_mode") or item.get("acquisition_mode", "")
        item["search_posture"] = snapshot.get("search_posture") or item.get("search_posture", "")
        lane_snapshot = dict(item.get("lane_snapshot") or {})
        lane_snapshot["compiler"] = snapshot
        item["lane_snapshot"] = lane_snapshot
        wired += 1

    return wired
