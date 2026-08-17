#!/usr/bin/env python3
"""Fail-closed, PII-free report for a GLM judgment matrix artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


_FORBIDDEN_KEYS = {
    "candidate_text",
    "current_company",
    "headline",
    "name",
    "profile_url",
    "prompt",
    "rationale",
    "raw",
    "reasoning_content",
    "system_prompt",
    "user_prompt",
}
_HASH_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_PROMPT_HASH_FIELDS = (
    "system_prompt_sha256s",
    "candidate_prompt_sha256s",
    "tool_schema_sha256",
    "judgment_contract_version",
)
_EXECUTION_DESIGN = "deterministic-arm-block-interleaving-v1"
_EXECUTION_BLOCK_SIZE = 8
_WORKER_PROCESS_MODEL = "fresh-subprocess-per-arm-block"
_CACHE_CONTINUITY = "provider-side-affinity-key-across-worker-processes"
_TIMING_METRICS = {
    "facial_page_p50": ("facial_page", 0.50),
    "facial_page_p90": ("facial_page", 0.90),
    "full_profile_p50": ("full_profile", 0.50),
    "full_profile_p90": ("full_profile", 0.90),
}
_ALLOWED_THRESHOLD_KEYS = frozenset(
    {
        *(f"{name}_improvement_min" for name in _TIMING_METRICS),
        *(f"{name}_ci_lower_min" for name in _TIMING_METRICS),
        "deadline_success_min",
        "retryable_429_503_rate_max",
        "retryable_429_503_rate_max_increase",
        "postcondition_fallback_rate_max",
        "postcondition_fallback_rate_max_increase",
        "unrecovered_postcondition_rate_max",
        "max_attempts",
        "human_pass_rate_delta_ci_lower_min",
    }
)
_ALLOWED_HUMAN_DECISIONS = frozenset(
    {
        "FACIAL_YES",
        "FACIAL_NO",
        "FACIAL_BORDERLINE",
        "SAVE",
        "SIGNAL_SAVE",
        "INFERENTIAL_SAVE",
        "TRANSFERABLE_SAVE",
        "REJECT",
        "REVIEW_INFERRED",
        "REVIEW_FLAGGED",
    }
)
_LEVER_FIELDS = {
    "reasoning_effort": frozenset(
        {
            "env.FIREWORKS_FACIAL_REASONING_EFFORT",
            "env.FIREWORKS_FULL_REASONING_EFFORT",
        }
    ),
    "prompt_affinity": frozenset(
        {"env.FIREWORKS_PROMPT_AFFINITY_ENABLED"}
    ),
    "judgment_contract": frozenset(
        {
            "env.LINKEDIN_V2_FACIAL_CONTRACT",
            "env.LINKEDIN_V2_FULL_CONTRACT",
        }
    ),
    "serving_tier": frozenset(
        {"env.FACIAL_MODEL_NAME", "env.FULL_EVAL_MODEL_NAME"}
    ),
    "facial_concurrency": frozenset(
        {"facial_mode", "env.LINKEDIN_FACIAL_CONCURRENCY_ENABLED"}
    ),
}
_NONPROMOTABLE_LEVERS: frozenset[str] = frozenset()
_COMMON_REQUIRED_THRESHOLDS = frozenset(
    {
        "deadline_success_min",
        "retryable_429_503_rate_max_increase",
        "postcondition_fallback_rate_max_increase",
        "unrecovered_postcondition_rate_max",
        "max_attempts",
        "human_pass_rate_delta_ci_lower_min",
    }
)
_PROFILE_REQUIRED_THRESHOLDS = {
    "reasoning_effort": _COMMON_REQUIRED_THRESHOLDS,
    "prompt_affinity": _COMMON_REQUIRED_THRESHOLDS,
    "judgment_contract": _COMMON_REQUIRED_THRESHOLDS,
    "serving_tier": _COMMON_REQUIRED_THRESHOLDS
    | {
        f"{stage}_{percentile}_{suffix}"
        for stage in ("facial_page", "full_profile")
        for percentile in ("p50", "p90")
        for suffix in ("improvement_min", "ci_lower_min")
    },
    "facial_concurrency": _COMMON_REQUIRED_THRESHOLDS
    | {
        f"facial_page_{percentile}_{suffix}"
        for percentile in ("p50", "p90")
        for suffix in ("improvement_min", "ci_lower_min")
    },
}


class ExperimentReportError(ValueError):
    """Artifacts are incomplete, inconsistent, or contain forbidden PII."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentReportError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ExperimentReportError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ExperimentReportError(f"missing required artifact: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExperimentReportError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ExperimentReportError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _assert_no_forbidden_keys(value: object, *, location: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ExperimentReportError(f"forbidden PII/reasoning key {key!r} in {location}")
            _assert_no_forbidden_keys(nested, location=location)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_keys(nested, location=location)


def _call_id(row: dict[str, Any]) -> str:
    return str(row.get("logical_call_id") or row.get("call_id") or "")


def _nonnegative_number(row: dict[str, Any], key: str, *, location: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentReportError(f"{location} lacks numeric {key}") from exc
    if not math.isfinite(value) or value < 0:
        raise ExperimentReportError(f"{location} has invalid {key}")
    return value


def _nonnegative_int(row: dict[str, Any], key: str, *, location: str) -> int:
    raw = row.get(key)
    if isinstance(raw, bool):
        raise ExperimentReportError(f"{location} lacks integer {key}")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ExperimentReportError(f"{location} lacks integer {key}") from exc
    if value < 0 or value != raw:
        raise ExperimentReportError(f"{location} has invalid {key}")
    return value


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _round_optional(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _prompt_hash_manifest_sha256(calls: list[dict[str, Any]]) -> str:
    payload = [
        {
            "call_id": _call_id(row),
            **{field: row.get(field) for field in _PROMPT_HASH_FIELDS},
        }
        for row in sorted(calls, key=_call_id)
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _seed(base_seed: int, *parts: str) -> int:
    payload = "\0".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _scheduled_arm_order(
    arm_ids: list[str], *, random_seed: int, block_index: int
) -> list[str]:
    ordered = list(arm_ids)
    random.Random(_seed(random_seed, "arm-block", str(block_index))).shuffle(
        ordered
    )
    return ordered


def _schedule_sha256(schedule: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        schedule, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _validate_execution_schedule(
    metadata: dict[str, Any],
    *,
    arm_ids: list[str],
    expected_units: dict[str, list[str]],
    random_seed: int,
) -> tuple[str, int]:
    """Prove the completed subprocess order was paired block interleaving."""

    if metadata.get("status") not in {"reporting", "complete"}:
        raise ExperimentReportError(
            "experiment is not ready for fail-closed schedule validation"
        )
    if metadata.get("execution_design") != _EXECUTION_DESIGN:
        raise ExperimentReportError(
            "experiment execution design is missing or non-interleaved"
        )
    if metadata.get("execution_block_size") != _EXECUTION_BLOCK_SIZE:
        raise ExperimentReportError("experiment execution block size drifted")
    if metadata.get("worker_process_model") != _WORKER_PROCESS_MODEL:
        raise ExperimentReportError("experiment worker process model drifted")
    if metadata.get("cache_continuity") != _CACHE_CONTINUITY:
        raise ExperimentReportError("experiment cache-continuity design drifted")

    common_units = expected_units[arm_ids[0]]
    if any(expected_units[arm_id] != common_units for arm_id in arm_ids[1:]):
        raise ExperimentReportError(
            "experiment arms do not share exact paired execution-unit order"
        )
    expected_schedule = [
        {
            "block_index": block_index,
            "execution_unit_ids": common_units[
                start : start + _EXECUTION_BLOCK_SIZE
            ],
            "arm_order": _scheduled_arm_order(
                arm_ids,
                random_seed=random_seed,
                block_index=block_index,
            ),
        }
        for block_index, start in enumerate(
            range(0, len(common_units), _EXECUTION_BLOCK_SIZE)
        )
    ]
    schedule = metadata.get("execution_schedule")
    if schedule != expected_schedule:
        raise ExperimentReportError(
            "experiment execution schedule is missing, non-interleaved, or drifted"
        )
    schedule_hash = _schedule_sha256(expected_schedule)
    if metadata.get("execution_schedule_sha256") != schedule_hash:
        raise ExperimentReportError("experiment execution schedule hash mismatch")

    expected_dispatches = [
        {
            "block_index": int(block["block_index"]),
            "arm_id": arm_id,
            "status": "complete",
        }
        for block in expected_schedule
        for arm_id in block["arm_order"]
    ]
    if metadata.get("dispatch_log") != expected_dispatches:
        raise ExperimentReportError(
            "experiment subprocess dispatch log is missing or non-interleaved"
        )
    return schedule_hash, len(expected_schedule)


def _arm_summary(
    arm_id: str,
    expected: list[str],
    expected_units: list[str],
    expected_unit_calls: dict[str, list[str]],
    expected_unit_cache_phases: dict[str, str] | None,
    arm_dir: Path,
    *,
    experiment_id: str,
    manifest_hash: str,
    execution_schedule_sha256: str,
    expected_block_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arm_metadata = _read_json(arm_dir / "arm-metadata.json")
    if arm_metadata.get("status") != "complete":
        raise ExperimentReportError(f"arm {arm_id} is not complete")
    if arm_metadata.get("execution_design") != _EXECUTION_DESIGN:
        raise ExperimentReportError(f"arm {arm_id} execution design drifted")
    if arm_metadata.get("execution_block_size") != _EXECUTION_BLOCK_SIZE:
        raise ExperimentReportError(f"arm {arm_id} execution block size drifted")
    if arm_metadata.get("worker_process_model") != _WORKER_PROCESS_MODEL:
        raise ExperimentReportError(f"arm {arm_id} worker process model drifted")
    if arm_metadata.get("cache_continuity") != _CACHE_CONTINUITY:
        raise ExperimentReportError(f"arm {arm_id} cache continuity drifted")
    if arm_metadata.get("execution_schedule_sha256") != execution_schedule_sha256:
        raise ExperimentReportError(f"arm {arm_id} schedule hash mismatch")
    if arm_metadata.get("expected_block_count") != expected_block_count:
        raise ExperimentReportError(f"arm {arm_id} block count mismatch")
    if arm_metadata.get("completed_block_indices") != list(
        range(expected_block_count)
    ):
        raise ExperimentReportError(f"arm {arm_id} block completion is partial")
    block_offsets = arm_metadata.get("block_cost_offsets_usd")
    if (
        not isinstance(block_offsets, list)
        or len(block_offsets) != expected_block_count
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in block_offsets
        )
    ):
        raise ExperimentReportError(f"arm {arm_id} block spend offsets are invalid")
    if arm_metadata.get("experiment_id") != experiment_id:
        raise ExperimentReportError(f"arm {arm_id} experiment_id mismatch")
    if arm_metadata.get("manifest_hash") != manifest_hash:
        raise ExperimentReportError(f"arm {arm_id} manifest hash mismatch")
    if arm_metadata.get("arm_id") != arm_id:
        raise ExperimentReportError(f"arm directory {arm_id} contains different arm metadata")
    if arm_metadata.get("expected_call_ids") != expected:
        raise ExperimentReportError(f"arm {arm_id} expected-call metadata mismatch")
    unit_order = arm_metadata.get("execution_unit_order")
    if not isinstance(unit_order, list) or unit_order != expected_units:
        raise ExperimentReportError(
            f"arm {arm_id} execution-unit metadata/artifact order mismatch"
        )
    if set(expected_unit_calls) != set(expected_units):
        raise ExperimentReportError(f"arm {arm_id} expected unit/call attribution mismatch")

    calls = _read_jsonl(arm_dir / "calls.jsonl")
    units = _read_jsonl(arm_dir / "execution-units.jsonl")
    usage = _read_jsonl(arm_dir / "token-cost-log.jsonl")
    attempts = _read_jsonl(arm_dir / "llm-attempts.jsonl")
    for label, payload in (
        ("arm metadata", arm_metadata),
        ("calls", calls),
        ("execution units", units),
        ("usage", usage),
        ("attempts", attempts),
    ):
        _assert_no_forbidden_keys(payload, location=f"{arm_id} {label}")

    call_ids = [_call_id(row) for row in calls]
    if not all(call_ids):
        raise ExperimentReportError(f"arm {arm_id} has a call row without call_id")
    if len(call_ids) != len(set(call_ids)):
        raise ExperimentReportError(f"arm {arm_id} has duplicate call rows")
    if set(call_ids) != set(expected) or len(call_ids) != len(expected):
        missing = sorted(set(expected) - set(call_ids))
        extra = sorted(set(call_ids) - set(expected))
        raise ExperimentReportError(f"arm {arm_id} call coverage mismatch; missing={missing}, extra={extra}")
    arm_env = arm_metadata.get("env")
    if not isinstance(arm_env, dict):
        raise ExperimentReportError(f"arm {arm_id} metadata has no environment")
    cache_phases = arm_metadata.get("cache_phase_by_execution_unit")
    if expected_unit_cache_phases is not None:
        if cache_phases != expected_unit_cache_phases:
            raise ExperimentReportError(
                f"arm {arm_id} cache-phase metadata mismatch"
            )
    elif cache_phases is not None:
        if not isinstance(cache_phases, dict) or set(cache_phases) != set(expected_units):
            raise ExperimentReportError(f"arm {arm_id} has invalid cache phases")
    if cache_phases is not None:
        if any(phase not in {"warmup", "warm"} for phase in cache_phases.values()):
            raise ExperimentReportError(f"arm {arm_id} has invalid cache phase")
        ordered_phases = [cache_phases[unit_id] for unit_id in unit_order]
        if "warmup" not in ordered_phases or "warm" not in ordered_phases:
            raise ExperimentReportError(
                f"arm {arm_id} cache design lacks warmup or warm measurements"
            )
        first_warm = ordered_phases.index("warm")
        if any(phase != "warmup" for phase in ordered_phases[:first_warm]) or any(
            phase != "warm" for phase in ordered_phases[first_warm:]
        ):
            raise ExperimentReportError(
                f"arm {arm_id} cache phases do not form a warmup prefix and warm block"
            )
    for row in calls:
        if row.get("experiment_id") != experiment_id or row.get("arm_id") != arm_id:
            raise ExperimentReportError(
                f"arm {arm_id} has call with mismatched experiment attribution"
            )
        for field in ("system_prompt_sha256s", "candidate_prompt_sha256s"):
            hashes = row.get(field)
            if (
                not isinstance(hashes, list)
                or not hashes
                or any(not isinstance(value, str) or not _HASH_RE.fullmatch(value) for value in hashes)
                or len(set(hashes)) != len(hashes)
            ):
                raise ExperimentReportError(
                    f"arm {arm_id} call {_call_id(row)} has invalid {field}"
                )
        contract_version = row.get("judgment_contract_version")
        if not isinstance(contract_version, str) or not contract_version:
            raise ExperimentReportError(
                f"arm {arm_id} call {_call_id(row)} lacks judgment contract version"
            )
        contract_key = (
            "LINKEDIN_V2_FACIAL_CONTRACT"
            if row.get("stage") == "facial"
            else "LINKEDIN_V2_FULL_CONTRACT"
        )
        tool_hash = row.get("tool_schema_sha256")
        if str(arm_env.get(contract_key)) == "tool":
            if not isinstance(tool_hash, str) or not _HASH_RE.fullmatch(tool_hash):
                raise ExperimentReportError(
                    f"arm {arm_id} tool call {_call_id(row)} lacks tool schema hash"
                )
        elif tool_hash not in {None, ""}:
            raise ExperimentReportError(
                f"arm {arm_id} legacy call {_call_id(row)} unexpectedly has tool schema hash"
            )
    if arm_metadata.get("prompt_hash_call_count") != len(calls):
        raise ExperimentReportError(f"arm {arm_id} prompt hash count mismatch")
    if arm_metadata.get("prompt_hash_manifest_sha256") != _prompt_hash_manifest_sha256(calls):
        raise ExperimentReportError(f"arm {arm_id} prompt hash manifest mismatch")
    if arm_metadata.get("deterministic_replay_ids") != "call-id-position-sha256-v1":
        raise ExperimentReportError(f"arm {arm_id} deterministic replay ID scheme mismatch")
    unit_ids = [str(row.get("execution_unit_id") or "") for row in units]
    if not all(unit_ids) or len(unit_ids) != len(set(unit_ids)):
        raise ExperimentReportError(f"arm {arm_id} has missing/duplicate execution units")
    if set(unit_ids) != set(expected_units) or len(unit_ids) != len(expected_units):
        missing = sorted(set(expected_units) - set(unit_ids))
        extra = sorted(set(unit_ids) - set(expected_units))
        raise ExperimentReportError(
            f"arm {arm_id} execution-unit coverage mismatch; missing={missing}, extra={extra}"
        )
    if unit_ids != unit_order:
        raise ExperimentReportError(
            f"arm {arm_id} execution-unit artifact order does not match metadata"
        )
    unit_call_ids: list[str] = []
    for row in units:
        unit_id = str(row["execution_unit_id"])
        if row.get("experiment_id") != experiment_id or row.get("arm_id") != arm_id:
            raise ExperimentReportError(
                f"arm {arm_id} execution unit has mismatched experiment attribution"
            )
        contained = row.get("call_ids")
        if (
            not isinstance(contained, list)
            or not contained
            or any(not isinstance(value, str) or not value for value in contained)
        ):
            raise ExperimentReportError(f"arm {arm_id} execution unit has no call_ids")
        normalized_call_ids = list(contained)
        if normalized_call_ids != expected_unit_calls[unit_id]:
            raise ExperimentReportError(
                f"arm {arm_id} execution unit/call attribution mismatch"
            )
        unit_call_ids.extend(normalized_call_ids)
        stage = str(row.get("stage") or "")
        if stage not in {"facial_page", "full_profile"}:
            raise ExperimentReportError(f"arm {arm_id} execution unit has invalid stage")
        if (unit_id.startswith("page-") and stage != "facial_page") or (
            unit_id.startswith("profile-") and stage != "full_profile"
        ):
            raise ExperimentReportError(
                f"arm {arm_id} execution unit stage attribution mismatch"
            )
        if cache_phases is not None and row.get("cache_phase") != cache_phases[unit_id]:
            raise ExperimentReportError(
                f"arm {arm_id} execution unit {unit_id} cache-phase mismatch"
            )
        _nonnegative_number(
            row,
            "elapsed_ms",
            location=f"arm {arm_id} execution unit {unit_id}",
        )
    if Counter(unit_call_ids) != Counter(expected):
        raise ExperimentReportError(
            f"arm {arm_id} execution units do not partition expected calls exactly"
        )
    if cache_phases is not None:
        phases_by_stage = {
            stage: {
                cache_phases[str(row["execution_unit_id"])]
                for row in units
                if row.get("stage") == stage
            }
            for stage in ("facial_page", "full_profile")
        }
        if any(phases != {"warmup", "warm"} for phases in phases_by_stage.values()):
            raise ExperimentReportError(
                f"arm {arm_id} cache design requires warmup and warm units per stage"
            )
    for row in calls:
        samples = row.get("sample_ids")
        decisions = row.get("decisions")
        if not isinstance(samples, list) or not isinstance(decisions, list):
            raise ExperimentReportError(f"arm {arm_id} call lacks sample/decision arrays")
        if len(samples) != len(decisions):
            raise ExperimentReportError(
                f"arm {arm_id} {row.get('stage')} decision cardinality mismatch"
            )
        _nonnegative_number(
            row,
            "elapsed_ms",
            location=f"arm {arm_id} call {_call_id(row)}",
        )
        if cache_phases is not None:
            unit_id = next(
                (
                    expected_unit_id
                    for expected_unit_id, call_ids in expected_unit_calls.items()
                    if _call_id(row) in call_ids
                ),
                None,
            )
            if unit_id is None or row.get("cache_phase") != cache_phases[unit_id]:
                raise ExperimentReportError(
                    f"arm {arm_id} call {_call_id(row)} cache-phase mismatch"
                )

    expected_set = set(expected)
    fallback_parent_by_call: dict[str, str] = {}
    for row in usage:
        call_id = _call_id(row)
        if not call_id or call_id in expected_set:
            continue
        parent_id = str(row.get("parent_logical_call_id") or "")
        fallback_reason = str(row.get("fallback_reason") or "")
        if parent_id not in expected_set or not fallback_reason:
            raise ExperimentReportError(
                f"arm {arm_id} has unlinked usage receipt {call_id!r}"
            )
        fallback_parent_by_call[call_id] = parent_id
    allowed_call_ids = expected_set | set(fallback_parent_by_call)
    call_cache_phases = {
        _call_id(row): str(row.get("cache_phase") or "") for row in calls
    }
    fallback_counts = Counter(fallback_parent_by_call.values())
    for row in calls:
        if int(row.get("fallback_count") or 0) != fallback_counts[_call_id(row)]:
            raise ExperimentReportError(
                f"arm {arm_id} call {_call_id(row)} fallback attribution mismatch"
            )
    usage_ids = {_call_id(row) for row in usage if _call_id(row)}
    attempt_ids = {_call_id(row) for row in attempts if _call_id(row)}
    missing_usage = sorted(allowed_call_ids - usage_ids)
    missing_attempts = sorted(allowed_call_ids - attempt_ids)
    extra_usage = sorted(usage_ids - allowed_call_ids)
    extra_attempts = sorted(attempt_ids - allowed_call_ids)
    if missing_usage:
        raise ExperimentReportError(f"arm {arm_id} missing usage receipts for {missing_usage}")
    if missing_attempts:
        raise ExperimentReportError(f"arm {arm_id} missing attempt receipts for {missing_attempts}")
    if extra_usage:
        raise ExperimentReportError(f"arm {arm_id} has unplanned usage receipts for {extra_usage}")
    if extra_attempts:
        raise ExperimentReportError(f"arm {arm_id} has unplanned attempt receipts for {extra_attempts}")
    usage_counts = Counter(_call_id(row) for row in usage)
    if any(usage_counts[call_id] != 1 for call_id in allowed_call_ids):
        raise ExperimentReportError(
            f"arm {arm_id} must contain exactly one aggregate usage row per logical call"
        )
    attempts_by_id: dict[str, list[dict[str, Any]]] = {
        call_id: [] for call_id in allowed_call_ids
    }
    for row in attempts:
        attempts_by_id[_call_id(row)].append(row)
    for call_id, rows in attempts_by_id.items():
        try:
            numbers = [int(row["attempt_number"]) for row in rows]
            max_values = {int(row["max_attempts"]) for row in rows}
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentReportError(
                f"arm {arm_id} call {call_id} has invalid attempt metadata"
            ) from exc
        if numbers != list(range(1, len(rows) + 1)) or len(set(numbers)) != len(numbers):
            raise ExperimentReportError(
                f"arm {arm_id} call {call_id} attempt sequence is not contiguous"
            )
        if len(max_values) != 1 or len(rows) > next(iter(max_values)):
            raise ExperimentReportError(
                f"arm {arm_id} call {call_id} exceeds its declared attempt limit"
            )
        statuses_for_call = [str(row.get("status") or "") for row in rows]
        if any(status not in {"retryable_error", "response_received", "terminal_error"} for status in statuses_for_call):
            raise ExperimentReportError(
                f"arm {arm_id} call {call_id} has an invalid attempt status"
            )
        if statuses_for_call[-1] not in {"response_received", "terminal_error"}:
            raise ExperimentReportError(
                f"arm {arm_id} call {call_id} attempt sequence has no terminal row"
            )
        if any(status != "retryable_error" for status in statuses_for_call[:-1]):
            raise ExperimentReportError(
                f"arm {arm_id} call {call_id} continued after a terminal attempt"
            )
    for label, rows in (("usage", usage), ("attempt", attempts)):
        for row in rows:
            if row.get("experiment_id") != experiment_id or row.get("arm_id") != arm_id:
                raise ExperimentReportError(
                    f"arm {arm_id} has {label} receipt with mismatched experiment attribution"
                )
            if cache_phases is not None:
                call_id = _call_id(row)
                parent_id = fallback_parent_by_call.get(call_id, call_id)
                if row.get("cache_phase") != call_cache_phases.get(parent_id):
                    raise ExperimentReportError(
                        f"arm {arm_id} {label} receipt {call_id} cache-phase mismatch"
                    )

    usage_statuses = Counter(str(row.get("usage_status") or "unavailable") for row in usage)
    if any(status != "measured" for status in usage_statuses):
        raise ExperimentReportError(
            f"arm {arm_id} contains partial/unavailable usage; report cannot prove spend"
        )
    for row in usage:
        call_id = _call_id(row)
        if row.get("cost_completeness") != "complete":
            raise ExperimentReportError(
                f"arm {arm_id} call {call_id} usage cost is not complete"
            )
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            _nonnegative_int(
                row,
                key,
                location=f"arm {arm_id} call {call_id} usage",
            )
        _nonnegative_number(
            row,
            "estimated_cost_usd",
            location=f"arm {arm_id} call {call_id} usage",
        )
    input_tokens = sum(int(row["input_tokens"]) for row in usage)
    output_tokens = sum(int(row["output_tokens"]) for row in usage)
    cache_tokens = sum(int(row["cache_read_input_tokens"]) for row in usage)
    total_prompt_tokens = input_tokens + cache_tokens
    warm_usage = [
        row
        for row in usage
        if cache_phases is not None
        and call_cache_phases[
            fallback_parent_by_call.get(_call_id(row), _call_id(row))
        ]
        == "warm"
    ]
    warm_input_tokens = sum(int(row["input_tokens"]) for row in warm_usage)
    warm_cache_tokens = sum(
        int(row["cache_read_input_tokens"]) for row in warm_usage
    )
    warm_prompt_tokens = warm_input_tokens + warm_cache_tokens
    cost = sum(float(row["estimated_cost_usd"]) for row in usage)
    statuses = Counter(str(row.get("actual_status") or "unknown") for row in calls)
    decision_counts: Counter[str] = Counter()
    for row in calls:
        for decision in row.get("decisions") or []:
            decision_counts[str(decision)] += 1

    attempts_by_call = Counter(_call_id(row) for row in attempts if _call_id(row))
    retryable_parent_calls: set[str] = set()
    for row in attempts:
        try:
            status_code = int(row.get("http_status"))
        except (TypeError, ValueError):
            status_code = None
        if status_code in {429, 503}:
            call_id = _call_id(row)
            retryable_parent_calls.add(fallback_parent_by_call.get(call_id, call_id))
    deadline_successes = 0
    for row in calls:
        stage = str(row.get("stage") or "")
        deadline_key = (
            "FIREWORKS_FACIAL_TOTAL_DEADLINE_SECONDS"
            if stage == "facial"
            else "FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS"
        )
        try:
            deadline_ms = float(arm_env[deadline_key]) * 1000.0
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentReportError(
                f"arm {arm_id} lacks declared {stage} deadline"
            ) from exc
        call_attempts = attempts_by_id[_call_id(row)]
        if (
            call_attempts[-1].get("status") == "response_received"
            and float(row["elapsed_ms"]) <= deadline_ms
        ):
            deadline_successes += 1
    postcondition_or_fallback = {
        _call_id(row)
        for row in calls
        if row.get("actual_status") == "postcondition_fail"
        or fallback_counts[_call_id(row)] > 0
    }
    unrecovered_postconditions = {
        _call_id(row)
        for row in calls
        if row.get("actual_status") == "postcondition_fail"
    }
    call_latencies = [float(row["elapsed_ms"]) for row in calls]
    facial_page_latencies = [
        float(row["elapsed_ms"])
        for row in units
        if row.get("stage") == "facial_page"
    ]
    full_profile_latencies = [
        float(row["elapsed_ms"])
        for row in units
        if row.get("stage") == "full_profile"
    ]
    summary = {
        "arm_id": arm_id,
        "complete": True,
        "call_count": len(calls),
        "call_p50_ms": _round_optional(_percentile(call_latencies, 0.50)),
        "call_p90_ms": _round_optional(_percentile(call_latencies, 0.90)),
        "call_p95_ms": _round_optional(_percentile(call_latencies, 0.95)),
        "facial_page_timing_count": len(facial_page_latencies),
        "facial_page_p50_ms": _round_optional(_percentile(facial_page_latencies, 0.50)),
        "facial_page_p90_ms": _round_optional(_percentile(facial_page_latencies, 0.90)),
        "facial_page_p95_ms": _round_optional(_percentile(facial_page_latencies, 0.95)),
        "full_profile_timing_count": len(full_profile_latencies),
        "full_profile_p50_ms": _round_optional(_percentile(full_profile_latencies, 0.50)),
        "full_profile_p90_ms": _round_optional(_percentile(full_profile_latencies, 0.90)),
        "full_profile_p95_ms": _round_optional(_percentile(full_profile_latencies, 0.95)),
        "measured_input_tokens": int(input_tokens),
        "measured_output_tokens": int(output_tokens),
        "measured_cache_read_input_tokens": int(cache_tokens),
        "measured_cache_read_ratio": (
            round(cache_tokens / total_prompt_tokens, 4)
            if total_prompt_tokens
            else None
        ),
        "warm_measured_input_tokens": int(warm_input_tokens),
        "warm_measured_cache_read_input_tokens": int(warm_cache_tokens),
        "warm_measured_cache_read_ratio": (
            round(warm_cache_tokens / warm_prompt_tokens, 4)
            if warm_prompt_tokens
            else None
        ),
        "estimated_cost_lower_bound_usd": round(cost, 6),
        "usage_statuses": dict(sorted(usage_statuses.items())),
        "usage_complete": True,
        "max_attempts_per_call": max(attempts_by_call.values(), default=0),
        "deadline_success_rate": round(deadline_successes / len(calls), 6),
        "retryable_429_503_rate": round(
            len(retryable_parent_calls) / len(calls), 6
        ),
        "postcondition_fallback_rate": round(
            len(postcondition_or_fallback) / len(calls), 6
        ),
        "unrecovered_postcondition_rate": round(
            len(unrecovered_postconditions) / len(calls), 6
        ),
        "fallback_child_call_count": len(fallback_parent_by_call),
        "statuses": dict(sorted(statuses.items())),
        "decisions": dict(sorted(decision_counts.items())),
    }
    evidence = {
        "calls": calls,
        "units": units,
        "usage": usage,
        "attempts": attempts,
        "expected_unit_calls": expected_unit_calls,
        "arm_env": arm_env,
        "facial_mode": str(arm_metadata.get("facial_mode") or "pagewide"),
        "fallback_parent_by_call": fallback_parent_by_call,
        "cache_phase_by_call": call_cache_phases,
        "execution_unit_order": list(unit_order),
    }
    return summary, evidence


def _paired_timing_metric(
    control: dict[str, float],
    treatment: dict[str, float],
    *,
    percentile: float,
    iterations: int,
    seed: int,
    cluster_repetitions: bool,
) -> dict[str, Any]:
    if set(control) != set(treatment) or not control:
        raise ExperimentReportError("comparison timing units are not exactly paired")
    cluster_for = (
        (lambda unit_id: re.sub(r"-r\d{3}$", "", unit_id))
        if cluster_repetitions
        else (lambda unit_id: unit_id)
    )
    clusters: dict[str, list[str]] = {}
    for unit_id in sorted(control):
        clusters.setdefault(cluster_for(unit_id), []).append(unit_id)
    cluster_ids = sorted(clusters)

    def estimate(selected_clusters: list[str]) -> tuple[float, float, float]:
        unit_ids = [
            unit_id
            for cluster_id in selected_clusters
            for unit_id in clusters[cluster_id]
        ]
        control_value = _percentile([control[unit_id] for unit_id in unit_ids], percentile)
        treatment_value = _percentile(
            [treatment[unit_id] for unit_id in unit_ids], percentile
        )
        if control_value is None or treatment_value is None or control_value <= 0:
            raise ExperimentReportError("comparison timing metric has zero denominator")
        return (
            control_value,
            treatment_value,
            (control_value - treatment_value) / control_value,
        )

    control_value, treatment_value, improvement = estimate(cluster_ids)
    rng = random.Random(seed)
    bootstrap_improvements = []
    for _ in range(iterations):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        bootstrap_improvements.append(estimate(sampled)[2])
    low = _percentile(bootstrap_improvements, 0.025)
    high = _percentile(bootstrap_improvements, 0.975)
    return {
        "control_ms": round(control_value, 3),
        "treatment_ms": round(treatment_value, 3),
        "delta_ms": round(treatment_value - control_value, 3),
        "improvement_fraction": round(improvement, 6),
        "bootstrap_ci95_improvement": [
            _round_optional(low, 6),
            _round_optional(high, 6),
        ],
        "pair_count": len(control),
        "cluster_count": len(cluster_ids),
        "bootstrap_iterations": iterations,
    }


def _linked_complete_usage_by_parent(
    arm_id: str, evidence: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Group each exact aggregate receipt once under its planned parent call."""

    parent_ids = [_call_id(row) for row in evidence.get("calls", [])]
    if not parent_ids or not all(parent_ids) or len(set(parent_ids)) != len(parent_ids):
        raise ExperimentReportError(
            f"arm {arm_id} has invalid parent calls for paired usage"
        )
    fallback_parent_by_call = evidence.get("fallback_parent_by_call")
    if not isinstance(fallback_parent_by_call, dict) or any(
        not child_id
        or parent_id not in parent_ids
        or child_id in parent_ids
        for child_id, parent_id in fallback_parent_by_call.items()
    ):
        raise ExperimentReportError(
            f"arm {arm_id} has invalid fallback linkage for paired usage"
        )
    usage = evidence.get("usage")
    if not isinstance(usage, list) or not usage:
        raise ExperimentReportError(f"arm {arm_id} has no linked aggregate usage")
    allowed_ids = set(parent_ids) | set(fallback_parent_by_call)
    receipt_ids = [_call_id(row) for row in usage]
    if (
        not all(receipt_ids)
        or set(receipt_ids) != allowed_ids
        or len(receipt_ids) != len(set(receipt_ids))
    ):
        raise ExperimentReportError(
            f"arm {arm_id} linked aggregate usage is incomplete or duplicated"
        )
    cache_phase_by_call = evidence.get("cache_phase_by_call")
    if cache_phase_by_call is not None and (
        not isinstance(cache_phase_by_call, dict)
        or set(cache_phase_by_call) != set(parent_ids)
    ):
        raise ExperimentReportError(f"arm {arm_id} has invalid call cache phases")
    grouped = {parent_id: [] for parent_id in parent_ids}
    for row in usage:
        call_id = _call_id(row)
        parent_id = fallback_parent_by_call.get(call_id, call_id)
        if row.get("usage_status") != "measured" or row.get(
            "cost_completeness"
        ) != "complete":
            raise ExperimentReportError(
                f"arm {arm_id} paired usage is partial or unavailable"
            )
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            _nonnegative_int(
                row,
                key,
                location=f"arm {arm_id} call {call_id} paired usage",
            )
        if cache_phase_by_call is not None and row.get(
            "cache_phase"
        ) != cache_phase_by_call.get(parent_id):
            raise ExperimentReportError(
                f"arm {arm_id} usage receipt {call_id} cache-phase mismatch"
            )
        grouped[parent_id].append(row)
    return grouped


def _paired_output_token_metric(
    control_id: str,
    treatment_id: str,
    control: dict[str, Any],
    treatment: dict[str, Any],
) -> dict[str, Any]:
    control_usage = _linked_complete_usage_by_parent(control_id, control)
    treatment_usage = _linked_complete_usage_by_parent(treatment_id, treatment)
    if set(control_usage) != set(treatment_usage):
        raise ExperimentReportError(
            "reasoning output-token receipts are not exactly paired"
        )
    control_by_call = {
        call_id: sum(int(row["output_tokens"]) for row in rows)
        for call_id, rows in control_usage.items()
    }
    treatment_by_call = {
        call_id: sum(int(row["output_tokens"]) for row in rows)
        for call_id, rows in treatment_usage.items()
    }
    control_total = sum(control_by_call.values())
    treatment_total = sum(treatment_by_call.values())
    if control_total <= 0:
        raise ExperimentReportError(
            "reasoning output-token comparison has a zero control denominator"
        )
    return {
        "control_output_tokens": control_total,
        "treatment_output_tokens": treatment_total,
        "reduction_fraction": round(
            (control_total - treatment_total) / control_total, 6
        ),
        "pair_count": len(control_by_call),
        "linked_receipt_count": {
            "control": sum(len(rows) for rows in control_usage.values()),
            "treatment": sum(len(rows) for rows in treatment_usage.values()),
        },
    }


def _paired_relative_latency_metric(
    control: dict[str, Any],
    treatment: dict[str, Any],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    control_rows = control.get("calls", [])
    treatment_rows = treatment.get("calls", [])
    control_calls = {_call_id(row): row for row in control_rows}
    treatment_calls = {_call_id(row): row for row in treatment_rows}
    if (
        not control_calls
        or len(control_calls) != len(control_rows)
        or len(treatment_calls) != len(treatment_rows)
        or "" in control_calls
        or "" in treatment_calls
        or set(control_calls) != set(treatment_calls)
    ):
        raise ExperimentReportError(
            "reasoning latency calls are not exactly paired"
        )
    relative: dict[str, float] = {}
    stages: dict[str, list[float]] = {}
    for call_id in sorted(control_calls):
        control_row = control_calls[call_id]
        treatment_row = treatment_calls[call_id]
        if control_row.get("stage") != treatment_row.get("stage"):
            raise ExperimentReportError(
                "reasoning latency call stages are not exactly paired"
            )
        control_ms = _nonnegative_number(
            control_row,
            "elapsed_ms",
            location=f"control call {call_id}",
        )
        treatment_ms = _nonnegative_number(
            treatment_row,
            "elapsed_ms",
            location=f"treatment call {call_id}",
        )
        if control_ms <= 0:
            raise ExperimentReportError(
                "reasoning latency comparison has a zero control denominator"
            )
        improvement = (control_ms - treatment_ms) / control_ms
        relative[call_id] = improvement
        stages.setdefault(str(control_row.get("stage") or "unknown"), []).append(
            improvement
        )
    values = list(relative.values())
    point = _percentile(values, 0.50)
    rng = random.Random(seed)
    bootstrap = [
        _percentile([rng.choice(values) for _ in values], 0.50)
        for _ in range(iterations)
    ]
    low = _percentile([value for value in bootstrap if value is not None], 0.025)
    high = _percentile([value for value in bootstrap if value is not None], 0.975)
    return {
        "median_relative_improvement": _round_optional(point, 6),
        "bootstrap_ci95_median_relative_improvement": [
            _round_optional(low, 6),
            _round_optional(high, 6),
        ],
        "stage_median_relative_improvement": {
            stage: _round_optional(_percentile(stage_values, 0.50), 6)
            for stage, stage_values in sorted(stages.items())
        },
        "pair_count": len(values),
        "bootstrap_iterations": iterations,
    }


def _warm_cache_metric(arm_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    grouped = _linked_complete_usage_by_parent(arm_id, evidence)
    phases = evidence.get("cache_phase_by_call")
    if not isinstance(phases, dict) or set(phases.values()) != {"warmup", "warm"}:
        raise ExperimentReportError(
            f"arm {arm_id} affinity evidence lacks an explicit warm block"
        )
    warm_rows = [
        row
        for parent_id, rows in grouped.items()
        if phases[parent_id] == "warm"
        for row in rows
    ]
    if not warm_rows:
        raise ExperimentReportError(
            f"arm {arm_id} affinity evidence has no warm usage receipts"
        )
    uncached = sum(int(row["input_tokens"]) for row in warm_rows)
    cached = sum(int(row["cache_read_input_tokens"]) for row in warm_rows)
    denominator = uncached + cached
    if denominator <= 0:
        raise ExperimentReportError(
            f"arm {arm_id} warm cache ratio has a zero prompt-token denominator"
        )
    isolation_key_present = all(
        isinstance(row.get("prompt_cache_isolation_key"), str)
        and bool(str(row["prompt_cache_isolation_key"]).strip())
        for row in warm_rows
    )
    return {
        "phase": "warm",
        "token_weighted_cached_prompt_ratio": round(cached / denominator, 6),
        "cache_read_input_tokens": cached,
        "uncached_input_tokens": uncached,
        "prompt_token_denominator": denominator,
        "parent_call_count": sum(phase == "warm" for phase in phases.values()),
        "linked_receipt_count": len(warm_rows),
        "prompt_cache_isolation_key_present": isolation_key_present,
        "cold_cache_isolation_claimed": False,
        "cold_cache_isolation_reason": (
            "provider isolation is not verified by this report"
            if isolation_key_present
            else "prompt_cache_isolation_key is not recorded"
        ),
    }


def _decision_observations(evidence: dict[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    call_to_unit = {
        call_id: unit_id
        for unit_id, call_ids in evidence["expected_unit_calls"].items()
        for call_id in call_ids
    }
    observations: dict[tuple[str, str], dict[str, str]] = {}
    for row in evidence["calls"]:
        call_id = _call_id(row)
        unit_id = call_to_unit[call_id]
        stage = str(row["stage"])
        for sample_id, decision in zip(row["sample_ids"], row["decisions"]):
            key = (stage, str(sample_id))
            if unit_id in observations.setdefault(key, {}):
                raise ExperimentReportError("duplicate decision observation in execution unit")
            observations[key][unit_id] = str(decision)
    return observations


def _build_review_packet(
    *,
    experiment_id: str,
    manifest_hash: str,
    comparison_id: str,
    control: dict[str, Any],
    treatment: dict[str, Any],
    agreement_sample_size: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    control_observations = _decision_observations(control)
    treatment_observations = _decision_observations(treatment)
    if set(control_observations) != set(treatment_observations):
        raise ExperimentReportError("comparison decision samples are not exactly paired")
    internal: dict[str, dict[str, Any]] = {}
    disagreements: list[str] = []
    agreements: list[str] = []
    for stage, sample_id in sorted(control_observations):
        left = control_observations[(stage, sample_id)]
        right = treatment_observations[(stage, sample_id)]
        if set(left) != set(right):
            raise ExperimentReportError("comparison decision repeats are not exactly paired")
        unit_ids = sorted(left)
        control_decisions = [left[unit_id] for unit_id in unit_ids]
        treatment_decisions = [right[unit_id] for unit_id in unit_ids]
        review_id = "review-" + hashlib.sha256(
            f"{experiment_id}\0{comparison_id}\0{stage}\0{sample_id}".encode()
        ).hexdigest()[:24]
        internal[review_id] = {
            "stage": stage,
            "sample_id": sample_id,
            "control_decisions": control_decisions,
            "treatment_decisions": treatment_decisions,
        }
        (agreements if control_decisions == treatment_decisions else disagreements).append(
            review_id
        )
    # Preserve review power independently for the two materially different
    # judgments. A single global sample can be dominated by the much larger
    # facial corpus and hide a full-profile regression.
    selected_agreements: list[str] = []
    for stage in ("facial", "full"):
        stage_agreements = sorted(
            review_id
            for review_id in agreements
            if internal[review_id]["stage"] == stage
        )
        rng = random.Random(_seed(seed, stage, "agreement-sample"))
        selected_agreements.extend(
            rng.sample(
                stage_agreements,
                min(agreement_sample_size, len(stage_agreements)),
            )
        )
    selected = [(review_id, "disagreement") for review_id in sorted(disagreements)] + [
        (review_id, "agreement_sample") for review_id in sorted(selected_agreements)
    ]
    items: list[dict[str, Any]] = []
    for review_id, stratum in selected:
        row = internal[review_id]
        swap = _seed(seed, review_id, "blind-side") % 2 == 1
        side_a = row["treatment_decisions"] if swap else row["control_decisions"]
        side_b = row["control_decisions"] if swap else row["treatment_decisions"]
        items.append(
            {
                "review_id": review_id,
                "sample_id": row["sample_id"],
                "stage": row["stage"],
                "selection_stratum": stratum,
                "side_a_decisions": side_a,
                "side_b_decisions": side_b,
            }
        )
    packet_without_hash = {
        "schema_version": "glm-judgment-review-packet-v1",
        "experiment_id": experiment_id,
        "manifest_hash": manifest_hash,
        "comparison_id": comparison_id,
        "items": items,
    }
    packet_hash = "sha256:" + hashlib.sha256(
        json.dumps(packet_without_hash, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**packet_without_hash, "packet_hash": packet_hash}, {
        review_id: internal[review_id] for review_id, _stratum in selected
    }


def _comparison_arm_diff(control: dict[str, Any], treatment: dict[str, Any]) -> set[str]:
    changed = set()
    if control["facial_mode"] != treatment["facial_mode"]:
        changed.add("facial_mode")
    for key in set(control["arm_env"]) | set(treatment["arm_env"]):
        if control["arm_env"].get(key) != treatment["arm_env"].get(key):
            changed.add(f"env.{key}")
    return changed


def _evaluate_nonhuman_gates(
    thresholds: dict[str, float | int],
    *,
    metrics: dict[str, dict[str, Any]],
    control_arm: dict[str, Any],
    treatment_arm: dict[str, Any],
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for key, threshold in thresholds.items():
        if key == "human_pass_rate_delta_ci_lower_min":
            gates.append(
                {
                    "name": key,
                    "status": "pending",
                    "observed": None,
                    "threshold": threshold,
                    "comparator": ">=",
                }
            )
            continue
        comparator = ">="
        if key.endswith("_improvement_min"):
            metric_name = key.removesuffix("_improvement_min")
            observed = metrics[metric_name]["improvement_fraction"]
        elif key.endswith("_ci_lower_min"):
            metric_name = key.removesuffix("_ci_lower_min")
            observed = metrics[metric_name]["bootstrap_ci95_improvement"][0]
        elif key == "deadline_success_min":
            observed = treatment_arm["deadline_success_rate"]
        elif key == "retryable_429_503_rate_max":
            observed = treatment_arm["retryable_429_503_rate"]
            comparator = "<="
        elif key == "retryable_429_503_rate_max_increase":
            observed = (
                treatment_arm["retryable_429_503_rate"]
                - control_arm["retryable_429_503_rate"]
            )
            comparator = "<="
        elif key == "postcondition_fallback_rate_max":
            observed = treatment_arm["postcondition_fallback_rate"]
            comparator = "<="
        elif key == "postcondition_fallback_rate_max_increase":
            observed = (
                treatment_arm["postcondition_fallback_rate"]
                - control_arm["postcondition_fallback_rate"]
            )
            comparator = "<="
        elif key == "unrecovered_postcondition_rate_max":
            observed = treatment_arm["unrecovered_postcondition_rate"]
            comparator = "<="
        elif key == "max_attempts":
            observed = treatment_arm["max_attempts_per_call"]
            comparator = "<="
        else:
            raise ExperimentReportError(f"unknown comparison threshold: {key}")
        passed = observed >= threshold if comparator == ">=" else observed <= threshold
        gates.append(
            {
                "name": key,
                "status": "pass" if passed else "fail",
                "observed": _round_optional(float(observed), 6),
                "threshold": threshold,
                "comparator": comparator,
            }
        )
    return gates


def _build_comparison_report(
    comparison: dict[str, Any],
    *,
    summaries: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    experiment_id: str,
    manifest_hash: str,
    random_seed: int,
    bootstrap_iterations: int,
    agreement_sample_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    comparison_id = str(comparison.get("id") or "")
    lever_id = str(comparison.get("lever_id") or "")
    control_id = str(comparison.get("control_arm_id") or "")
    treatment_id = str(comparison.get("treatment_arm_id") or "")
    if not comparison_id or control_id not in summaries or treatment_id not in summaries:
        raise ExperimentReportError("comparison names unknown arms")
    if lever_id not in _LEVER_FIELDS:
        raise ExperimentReportError(f"comparison {comparison_id} has unknown lever")
    lever_fields = comparison.get("lever_fields")
    if not isinstance(lever_fields, list) or not lever_fields:
        raise ExperimentReportError(f"comparison {comparison_id} has no lever fields")
    if set(lever_fields) != set(_LEVER_FIELDS[lever_id]):
        raise ExperimentReportError(
            f"comparison {comparison_id} does not match its code-owned lever profile"
        )
    actual_diff = _comparison_arm_diff(evidence[control_id], evidence[treatment_id])
    if actual_diff != set(lever_fields):
        raise ExperimentReportError(
            f"comparison {comparison_id} arm diff does not match declared lever fields"
        )
    thresholds = comparison.get("thresholds", {})
    if not isinstance(thresholds, dict) or set(thresholds) - _ALLOWED_THRESHOLD_KEYS:
        raise ExperimentReportError(f"comparison {comparison_id} has invalid thresholds")
    missing_thresholds = _PROFILE_REQUIRED_THRESHOLDS[lever_id] - set(thresholds)
    if missing_thresholds:
        raise ExperimentReportError(
            f"comparison {comparison_id} omits required gate profile thresholds"
        )
    if (
        float(thresholds["deadline_success_min"]) < 0.99
        or not 0
        <= float(thresholds["retryable_429_503_rate_max_increase"])
        <= 0.02
        or not 0
        <= float(thresholds["postcondition_fallback_rate_max_increase"])
        <= 0.02
        or not 1 <= int(thresholds["max_attempts"]) <= 2
        or float(thresholds["human_pass_rate_delta_ci_lower_min"]) < -0.03
        or float(thresholds["unrecovered_postcondition_rate_max"]) != 0
    ):
        raise ExperimentReportError(
            f"comparison {comparison_id} gate profile weakens plan thresholds"
        )
    for key, value in thresholds.items():
        if key.endswith("_improvement_min") and float(value) < 0.30:
            raise ExperimentReportError(
                f"comparison {comparison_id} weakens point-improvement threshold"
            )
        if (
            key.endswith("_ci_lower_min")
            and key != "human_pass_rate_delta_ci_lower_min"
            and float(value) < 0.20
        ):
            raise ExperimentReportError(
                f"comparison {comparison_id} weakens bootstrap threshold"
            )

    unit_maps: dict[str, dict[str, tuple[str, float]]] = {}
    for arm_id in (control_id, treatment_id):
        unit_maps[arm_id] = {
            str(row["execution_unit_id"]): (
                str(row["stage"]),
                float(row["elapsed_ms"]),
            )
            for row in evidence[arm_id]["units"]
        }
    if set(unit_maps[control_id]) != set(unit_maps[treatment_id]):
        raise ExperimentReportError(
            f"comparison {comparison_id} execution units are not exactly paired"
        )
    if evidence[control_id].get("execution_unit_order") != evidence[treatment_id].get(
        "execution_unit_order"
    ):
        raise ExperimentReportError(
            f"comparison {comparison_id} execution-unit order is not exactly paired"
        )
    metrics: dict[str, dict[str, Any]] = {}
    for metric_name, (stage, percentile) in _TIMING_METRICS.items():
        control_values = {
            unit_id: value
            for unit_id, (unit_stage, value) in unit_maps[control_id].items()
            if unit_stage == stage
        }
        treatment_values = {
            unit_id: value
            for unit_id, (unit_stage, value) in unit_maps[treatment_id].items()
            if unit_stage == stage
        }
        metrics[metric_name] = _paired_timing_metric(
            control_values,
            treatment_values,
            percentile=percentile,
            iterations=bootstrap_iterations,
            seed=_seed(random_seed, comparison_id, metric_name),
            cluster_repetitions=stage == "facial_page",
        )
    if lever_id == "reasoning_effort":
        metrics["reasoning_output_tokens"] = _paired_output_token_metric(
            control_id,
            treatment_id,
            evidence[control_id],
            evidence[treatment_id],
        )
        metrics["reasoning_relative_latency"] = _paired_relative_latency_metric(
            evidence[control_id],
            evidence[treatment_id],
            iterations=bootstrap_iterations,
            seed=_seed(random_seed, comparison_id, "reasoning-relative-latency"),
        )
    elif lever_id == "prompt_affinity":
        metrics["warm_cache"] = _warm_cache_metric(
            treatment_id, evidence[treatment_id]
        )

    contract_levers = {
        "env.LINKEDIN_V2_FACIAL_CONTRACT",
        "env.LINKEDIN_V2_FULL_CONTRACT",
    }
    prompt_hashes_matched = True
    if not set(lever_fields) & contract_levers:
        control_calls = {_call_id(row): row for row in evidence[control_id]["calls"]}
        treatment_calls = {_call_id(row): row for row in evidence[treatment_id]["calls"]}
        for call_id in set(control_calls) & set(treatment_calls):
            if any(
                control_calls[call_id].get(field) != treatment_calls[call_id].get(field)
                for field in _PROMPT_HASH_FIELDS
            ):
                prompt_hashes_matched = False
                break
        if not prompt_hashes_matched:
            raise ExperimentReportError(
                f"comparison {comparison_id} changed prompt/contract hashes outside its lever"
            )

    packet, review_internal = _build_review_packet(
        experiment_id=experiment_id,
        manifest_hash=manifest_hash,
        comparison_id=comparison_id,
        control=evidence[control_id],
        treatment=evidence[treatment_id],
        agreement_sample_size=agreement_sample_size,
        seed=_seed(random_seed, comparison_id, "review"),
    )
    gates = _evaluate_nonhuman_gates(
        thresholds,
        metrics=metrics,
        control_arm=summaries[control_id],
        treatment_arm=summaries[treatment_id],
    )
    if lever_id == "reasoning_effort":
        output_reduction = metrics["reasoning_output_tokens"]["reduction_fraction"]
        latency_ci_lower = metrics["reasoning_relative_latency"][
            "bootstrap_ci95_median_relative_improvement"
        ][0]
        reasoning_passed = output_reduction >= 0.20 or latency_ci_lower >= 0.15
        gates.append(
            {
                "name": "reasoning_output_or_latency",
                "status": "pass" if reasoning_passed else "fail",
                "observed": {
                    "output_token_reduction": output_reduction,
                    "paired_relative_latency_ci_lower": latency_ci_lower,
                },
                "threshold": {
                    "output_token_reduction_min": 0.20,
                    "paired_relative_latency_ci_lower_min": 0.15,
                },
                "comparator": "OR",
            }
        )
    elif lever_id == "prompt_affinity":
        warm_ratio = metrics["warm_cache"][
            "token_weighted_cached_prompt_ratio"
        ]
        gates.append(
            {
                "name": "warm_token_weighted_cached_prompt_ratio",
                "status": "pass" if warm_ratio > 0.25 else "fail",
                "observed": warm_ratio,
                "threshold": 0.25,
                "comparator": ">",
            }
        )
        # Affinity is not a win if replica pinning buys cache hits by making
        # either operator-blocking stage materially slower or more expensive.
        latency_ci_lowers = {
            "facial_page_p90": metrics["facial_page_p90"][
                "bootstrap_ci95_improvement"
            ][0],
            "full_profile_p90": metrics["full_profile_p90"][
                "bootstrap_ci95_improvement"
            ][0],
        }
        gates.append(
            {
                "name": "affinity_latency_nonregression",
                "status": (
                    "pass"
                    if all(value >= -0.10 for value in latency_ci_lowers.values())
                    else "fail"
                ),
                "observed": latency_ci_lowers,
                "threshold": -0.10,
                "comparator": ">= per stage",
            }
        )
        control_cost = float(
            summaries[control_id]["estimated_cost_lower_bound_usd"]
        )
        treatment_cost = float(
            summaries[treatment_id]["estimated_cost_lower_bound_usd"]
        )
        cost_increase = (
            (treatment_cost - control_cost) / control_cost
            if control_cost > 0
            else (0.0 if treatment_cost == 0 else math.inf)
        )
        gates.append(
            {
                "name": "affinity_cost_nonregression",
                "status": "pass" if cost_increase <= 0.05 else "fail",
                "observed": _round_optional(cost_increase, 6),
                "threshold": 0.05,
                "comparator": "<=",
            }
        )
    report = {
        "comparison_id": comparison_id,
        "lever_id": lever_id,
        "depends_on": comparison.get("depends_on"),
        "control_arm_id": control_id,
        "treatment_arm_id": treatment_id,
        "lever_fields": list(lever_fields),
        "metrics": metrics,
        "observed": {
            "control": {
                key: summaries[control_id][key]
                for key in (
                    "deadline_success_rate",
                    "retryable_429_503_rate",
                    "postcondition_fallback_rate",
                    "unrecovered_postcondition_rate",
                    "max_attempts_per_call",
                )
            },
            "treatment": {
                key: summaries[treatment_id][key]
                for key in (
                    "deadline_success_rate",
                    "retryable_429_503_rate",
                    "postcondition_fallback_rate",
                    "unrecovered_postcondition_rate",
                    "max_attempts_per_call",
                )
            },
        },
        "prompt_hashes_matched_outside_lever": prompt_hashes_matched,
        "thresholds": thresholds,
        "promotion_supported": lever_id not in _NONPROMOTABLE_LEVERS,
        "promotion_support_blocker": None,
        "gates": gates,
        "review": {
            "packet_hash": packet["packet_hash"],
            "item_count": len(packet["items"]),
            "disagreement_count": sum(
                item["selection_stratum"] == "disagreement"
                for item in packet["items"]
            ),
            "labels_status": "missing",
        },
    }
    return report, packet, review_internal


def _human_stage_metrics(
    control_scores: list[float],
    treatment_scores: list[float],
    *,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    if not control_scores or len(control_scores) != len(treatment_scores):
        raise ExperimentReportError("human review stage sample is empty or unpaired")
    deltas_observed = [
        treatment - control
        for control, treatment in zip(control_scores, treatment_scores)
    ]
    point_delta = sum(deltas_observed) / len(deltas_observed)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(bootstrap_iterations):
        indexes = [rng.randrange(len(control_scores)) for _ in control_scores]
        deltas.append(sum(deltas_observed[index] for index in indexes) / len(indexes))
    return {
        "item_count": len(control_scores),
        "control_pass_rate": round(sum(control_scores) / len(control_scores), 6),
        "treatment_pass_rate": round(
            sum(treatment_scores) / len(treatment_scores), 6
        ),
        "pass_rate_delta": round(point_delta, 6),
        "pass_rate_delta_bootstrap_ci95": [
            _round_optional(_percentile(deltas, 0.025), 6),
            _round_optional(_percentile(deltas, 0.975), 6),
        ],
    }


def _apply_human_labels(
    labels_path: Path | None,
    *,
    experiment_id: str,
    manifest_hash: str,
    comparison_reports: list[dict[str, Any]],
    packets: dict[str, dict[str, Any]],
    review_internal: dict[str, dict[str, dict[str, Any]]],
    random_seed: int,
    bootstrap_iterations: int,
) -> str:
    if labels_path is None or not labels_path.is_file():
        return "missing"
    labels_payload = _read_json(labels_path)
    _assert_no_forbidden_keys(labels_payload, location="human review labels")
    if set(labels_payload) != {
        "schema_version",
        "experiment_id",
        "manifest_hash",
        "packet_hashes",
        "review_provenance",
        "labels",
    }:
        raise ExperimentReportError("human labels file has unknown/missing fields")
    if (
        labels_payload.get("schema_version") != "glm-judgment-review-labels-v2"
        or labels_payload.get("experiment_id") != experiment_id
        or labels_payload.get("manifest_hash") != manifest_hash
    ):
        raise ExperimentReportError("human labels identity/hash mismatch")
    expected_packet_hashes = {
        comparison_id: packet["packet_hash"]
        for comparison_id, packet in packets.items()
    }
    if labels_payload.get("packet_hashes") != expected_packet_hashes:
        raise ExperimentReportError("human labels packet hashes mismatch")
    provenance = labels_payload.get("review_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "reviewer_identity_sha256",
        "reviewed_at",
        "review_protocol",
    }:
        raise ExperimentReportError("human labels review provenance is invalid")
    if not _HASH_RE.fullmatch(str(provenance.get("reviewer_identity_sha256") or "")):
        raise ExperimentReportError("human labels reviewer identity hash is invalid")
    if provenance.get("review_protocol") != "blinded-local-evidence-v1":
        raise ExperimentReportError("human labels review protocol is invalid")
    reviewed_at = str(provenance.get("reviewed_at") or "").strip()
    try:
        reviewed_timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExperimentReportError("human labels reviewed_at is invalid") from exc
    if reviewed_timestamp.tzinfo is None or reviewed_timestamp.utcoffset() is None:
        raise ExperimentReportError("human labels reviewed_at must be timezone-aware")
    rows = labels_payload.get("labels")
    if not isinstance(rows, list):
        raise ExperimentReportError("human labels must be a list")
    expected_review_ids = {
        review_id
        for comparison_rows in review_internal.values()
        for review_id in comparison_rows
    }
    labels: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "review_id",
            "adjudicated_decision",
        }:
            raise ExperimentReportError("human label row has unknown/missing fields")
        review_id = str(row.get("review_id") or "")
        decision = str(row.get("adjudicated_decision") or "")
        if review_id in labels or decision not in _ALLOWED_HUMAN_DECISIONS:
            raise ExperimentReportError("human label row is duplicate or invalid")
        labels[review_id] = decision
    if set(labels) != expected_review_ids:
        raise ExperimentReportError("human labels do not cover the exact review packet")

    reports_by_id = {
        report["comparison_id"]: report for report in comparison_reports
    }
    for comparison_id, items in review_internal.items():
        control_scores: list[float] = []
        treatment_scores: list[float] = []
        stage_scores: dict[str, tuple[list[float], list[float]]] = {
            "facial": ([], []),
            "full": ([], []),
        }
        for review_id in sorted(items):
            adjudicated = labels[review_id]
            item = items[review_id]
            if item["stage"] == "facial" and not adjudicated.startswith("FACIAL_"):
                raise ExperimentReportError("human label decision has wrong stage")
            if item["stage"] == "full" and adjudicated.startswith("FACIAL_"):
                raise ExperimentReportError("human label decision has wrong stage")
            control_score = (
                sum(value == adjudicated for value in item["control_decisions"])
                / len(item["control_decisions"])
            )
            treatment_score = (
                sum(value == adjudicated for value in item["treatment_decisions"])
                / len(item["treatment_decisions"])
            )
            control_scores.append(control_score)
            treatment_scores.append(treatment_score)
            stage_scores[item["stage"]][0].append(control_score)
            stage_scores[item["stage"]][1].append(treatment_score)
        if not control_scores:
            raise ExperimentReportError("human review packet is empty")
        aggregate = _human_stage_metrics(
            control_scores,
            treatment_scores,
            seed=_seed(random_seed, comparison_id, "human-labels-aggregate"),
            bootstrap_iterations=bootstrap_iterations,
        )
        by_stage = {
            stage: _human_stage_metrics(
                controls,
                treatments,
                seed=_seed(random_seed, comparison_id, stage, "human-labels"),
                bootstrap_iterations=bootstrap_iterations,
            )
            for stage, (controls, treatments) in stage_scores.items()
        }
        report = reports_by_id[comparison_id]
        report["review"].update(
            {
                "labels_status": "validated",
                **aggregate,
                "by_stage": by_stage,
                "provenance": provenance,
            }
        )
        for gate in report["gates"]:
            if gate["name"] != "human_pass_rate_delta_ci_lower_min":
                continue
            observed = {
                stage: metrics["pass_rate_delta_bootstrap_ci95"][0]
                for stage, metrics in by_stage.items()
            }
            gate["observed"] = observed
            gate["status"] = (
                "pass"
                if all(
                    value >= float(gate["threshold"])
                    for value in observed.values()
                )
                else "fail"
            )
    return "validated"


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GLM judgment experiment report",
        "",
        f"Experiment: `{report['experiment_id']}`",
        "",
        "| Arm | Calls | Facial pages | Facial page p50/p90/p95 ms | Full profiles | Full profile p50/p90/p95 ms | Call p50/p90/p95 ms | Output tokens | Cache ratio | Cost USD | Max attempts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        cache = (
            "—"
            if arm["measured_cache_read_ratio"] is None
            else f"{arm['measured_cache_read_ratio']:.1%}"
        )
        lines.append(
            f"| {arm['arm_id']} | {arm['call_count']} | "
            f"{arm['facial_page_timing_count']} | "
            f"{arm['facial_page_p50_ms']}/{arm['facial_page_p90_ms']}/{arm['facial_page_p95_ms']} | "
            f"{arm['full_profile_timing_count']} | "
            f"{arm['full_profile_p50_ms']}/{arm['full_profile_p90_ms']}/{arm['full_profile_p95_ms']} | "
            f"{arm['call_p50_ms']}/{arm['call_p90_ms']}/{arm['call_p95_ms']} | "
            f"{arm['measured_output_tokens']} | "
            f"{cache} | {arm['estimated_cost_lower_bound_usd']:.6f} | {arm['max_attempts_per_call']} |"
        )
    lines.extend(["", f"Promotion ready: `{report['promotion_ready']}`", ""])
    for comparison in report["comparisons"]:
        lines.append(
            f"- `{comparison['comparison_id']}`: "
            f"{comparison['control_arm_id']} → {comparison['treatment_arm_id']}; "
            f"gates={comparison['gate_status']}; "
            f"review={comparison['review']['labels_status']}"
        )
    lines.extend(
        [
            "",
            "This report contains opaque sample IDs and aggregate metrics only; candidate evidence and model reasoning are excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    experiment_dir: Path,
    *,
    write: bool = True,
    labels_path: Path | None = None,
) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    metadata = _read_json(experiment_dir / "experiment-metadata.json")
    _assert_no_forbidden_keys(metadata, location="experiment metadata")
    experiment_id = str(metadata.get("experiment_id") or "")
    manifest_hash = str(metadata.get("manifest_hash") or "")
    arm_ids = metadata.get("arm_ids")
    expected_calls = metadata.get("expected_calls")
    expected_units = metadata.get("expected_units")
    expected_unit_calls = metadata.get("expected_unit_calls")
    expected_unit_cache_phases = metadata.get("expected_unit_cache_phases")
    comparisons = metadata.get("comparisons")
    review_config = metadata.get("review")
    random_seed = metadata.get("random_seed")
    bootstrap_iterations = metadata.get("bootstrap_iterations")
    if not experiment_id or not manifest_hash:
        raise ExperimentReportError("experiment metadata lacks identity/hash")
    if (
        not isinstance(arm_ids, list)
        or not arm_ids
        or any(not isinstance(value, str) or not value for value in arm_ids)
        or len(set(arm_ids)) != len(arm_ids)
    ):
        raise ExperimentReportError("experiment metadata has no arms")
    if not isinstance(expected_calls, dict) or set(expected_calls) != set(arm_ids):
        raise ExperimentReportError("experiment expected_calls does not match arm_ids")
    if not isinstance(expected_units, dict) or set(expected_units) != set(arm_ids):
        raise ExperimentReportError("experiment expected_units does not match arm_ids")
    if not isinstance(expected_unit_calls, dict) or set(expected_unit_calls) != set(arm_ids):
        raise ExperimentReportError("experiment expected_unit_calls does not match arm_ids")
    if not isinstance(comparisons, list):
        raise ExperimentReportError("experiment comparisons must be an ordered list")
    cache_phases_required = any(
        isinstance(comparison, dict)
        and comparison.get("lever_id") == "prompt_affinity"
        for comparison in comparisons
    )
    if cache_phases_required and (
        not isinstance(expected_unit_cache_phases, dict)
        or set(expected_unit_cache_phases) != set(arm_ids)
    ):
        raise ExperimentReportError(
            "affinity comparison requires pinned cache-phase metadata"
        )
    if (
        not isinstance(review_config, dict)
        or set(review_config) != {"agreement_sample_size"}
    ):
        raise ExperimentReportError("experiment review configuration is invalid")
    try:
        random_seed = int(random_seed)
        bootstrap_iterations = int(bootstrap_iterations)
        agreement_sample_size = int(review_config["agreement_sample_size"])
    except (TypeError, ValueError) as exc:
        raise ExperimentReportError("experiment analysis configuration is invalid") from exc
    if random_seed <= 0 or bootstrap_iterations < 100 or agreement_sample_size <= 0:
        raise ExperimentReportError("experiment analysis configuration is unsafe")

    normalized_calls: dict[str, list[str]] = {}
    normalized_units: dict[str, list[str]] = {}
    normalized_unit_calls: dict[str, dict[str, list[str]]] = {}
    normalized_cache_phases: dict[str, dict[str, str] | None] = {}
    for arm_id in arm_ids:
        calls_for_arm = expected_calls[arm_id]
        units_for_arm = expected_units[arm_id]
        unit_calls_for_arm = expected_unit_calls[arm_id]
        if (
            not isinstance(calls_for_arm, list)
            or not calls_for_arm
            or any(not isinstance(value, str) or not value for value in calls_for_arm)
            or len(set(calls_for_arm)) != len(calls_for_arm)
        ):
            raise ExperimentReportError(f"arm {arm_id} has invalid expected calls")
        if (
            not isinstance(units_for_arm, list)
            or not units_for_arm
            or any(not isinstance(value, str) or not value for value in units_for_arm)
            or len(set(units_for_arm)) != len(units_for_arm)
        ):
            raise ExperimentReportError(f"arm {arm_id} has invalid expected units")
        if not isinstance(unit_calls_for_arm, dict) or set(unit_calls_for_arm) != set(
            units_for_arm
        ):
            raise ExperimentReportError(
                f"arm {arm_id} expected unit/call attribution mismatch"
            )
        normalized_mapping: dict[str, list[str]] = {}
        for unit_id, call_ids in unit_calls_for_arm.items():
            if (
                not isinstance(call_ids, list)
                or not call_ids
                or any(not isinstance(value, str) or not value for value in call_ids)
            ):
                raise ExperimentReportError(
                    f"arm {arm_id} expected unit/call attribution mismatch"
                )
            normalized_mapping[unit_id] = list(call_ids)
        if Counter(
            call_id
            for call_ids in normalized_mapping.values()
            for call_id in call_ids
        ) != Counter(calls_for_arm):
            raise ExperimentReportError(
                f"arm {arm_id} expected units do not partition expected calls exactly"
            )
        normalized_calls[arm_id] = list(calls_for_arm)
        normalized_units[arm_id] = list(units_for_arm)
        normalized_unit_calls[arm_id] = normalized_mapping
        phases_for_arm = (
            expected_unit_cache_phases.get(arm_id)
            if isinstance(expected_unit_cache_phases, dict)
            else None
        )
        if phases_for_arm is not None and (
            not isinstance(phases_for_arm, dict)
            or set(phases_for_arm) != set(units_for_arm)
            or any(
                phase not in {"warmup", "warm"}
                for phase in phases_for_arm.values()
            )
        ):
            raise ExperimentReportError(
                f"arm {arm_id} has invalid expected cache phases"
            )
        normalized_cache_phases[arm_id] = phases_for_arm

    paired_phase_maps = [normalized_cache_phases[arm_id] for arm_id in arm_ids]
    if any(phases is not None for phases in paired_phase_maps) and (
        any(phases is None for phases in paired_phase_maps)
        or any(phases != paired_phase_maps[0] for phases in paired_phase_maps[1:])
    ):
        raise ExperimentReportError(
            "experiment arms do not share exact paired cache phases"
        )

    execution_schedule_sha256, expected_block_count = (
        _validate_execution_schedule(
            metadata,
            arm_ids=arm_ids,
            expected_units=normalized_units,
            random_seed=random_seed,
        )
    )
    arm_results = [
        _arm_summary(
            str(arm_id),
            normalized_calls[arm_id],
            normalized_units[arm_id],
            normalized_unit_calls[arm_id],
            normalized_cache_phases[arm_id],
            experiment_dir / "arms" / str(arm_id),
            experiment_id=experiment_id,
            manifest_hash=manifest_hash,
            execution_schedule_sha256=execution_schedule_sha256,
            expected_block_count=expected_block_count,
        )
        for arm_id in arm_ids
    ]
    arms = [summary for summary, _evidence in arm_results]
    summaries_by_id = {summary["arm_id"]: summary for summary in arms}
    evidence_by_id = {
        arm_id: result[1] for arm_id, result in zip(arm_ids, arm_results)
    }
    comparison_reports: list[dict[str, Any]] = []
    packets: dict[str, dict[str, Any]] = {}
    review_internal: dict[str, dict[str, dict[str, Any]]] = {}
    seen_comparison_ids: set[str] = set()
    previous_comparison: dict[str, Any] | None = None
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ExperimentReportError("comparison metadata row must be an object")
        comparison_id = str(comparison.get("id") or "")
        if not comparison_id or comparison_id in seen_comparison_ids:
            raise ExperimentReportError("comparison ids must be nonempty and unique")
        if previous_comparison is None:
            if comparison.get("depends_on") is not None:
                raise ExperimentReportError(
                    "first comparison must have depends_on=null"
                )
        elif (
            comparison.get("depends_on") != previous_comparison.get("id")
            or comparison.get("control_arm_id")
            != previous_comparison.get("treatment_arm_id")
        ):
            raise ExperimentReportError("comparison dependency chain is broken")
        seen_comparison_ids.add(comparison_id)
        comparison_report, packet, internal = _build_comparison_report(
            comparison,
            summaries=summaries_by_id,
            evidence=evidence_by_id,
            experiment_id=experiment_id,
            manifest_hash=manifest_hash,
            random_seed=random_seed,
            bootstrap_iterations=bootstrap_iterations,
            agreement_sample_size=agreement_sample_size,
        )
        comparison_reports.append(comparison_report)
        packets[comparison_id] = packet
        review_internal[comparison_id] = internal
        previous_comparison = comparison

    effective_labels_path = labels_path
    if effective_labels_path is None:
        candidate = experiment_dir / "review-labels.json"
        effective_labels_path = candidate if candidate.is_file() else None
    labels_status = _apply_human_labels(
        effective_labels_path,
        experiment_id=experiment_id,
        manifest_hash=manifest_hash,
        comparison_reports=comparison_reports,
        packets=packets,
        review_internal=review_internal,
        random_seed=random_seed,
        bootstrap_iterations=bootstrap_iterations,
    )
    for comparison in comparison_reports:
        statuses = [gate["status"] for gate in comparison["gates"]]
        comparison["gate_status"] = (
            "unconfigured"
            if not comparison["thresholds"]
            else "pending"
            if "pending" in statuses
            else "pass"
            if statuses and all(status == "pass" for status in statuses)
            else "fail"
        )
    promotion_ready = bool(comparison_reports) and labels_status == "validated" and all(
        comparison["gate_status"] == "pass"
        and comparison["promotion_supported"]
        for comparison in comparison_reports
    )
    report = {
        "schema_version": "glm-judgment-experiment-report-v3",
        "experiment_id": experiment_id,
        "manifest_hash": manifest_hash,
        "complete": True,
        "execution_design": _EXECUTION_DESIGN,
        "promotion_ready": promotion_ready,
        "promotion_blockers": [
            reason
            for reason, blocked in (
                ("no_declared_comparisons", not comparison_reports),
                ("human_review_labels_missing", labels_status != "validated"),
                (
                    "comparison_gates_not_passed",
                    any(
                        comparison["gate_status"] != "pass"
                        for comparison in comparison_reports
                    ),
                ),
                (
                    "lever_profile_not_machine_promotable",
                    any(
                        not comparison["promotion_supported"]
                        for comparison in comparison_reports
                    ),
                ),
            )
            if blocked
        ],
        "arms": arms,
        "comparisons": comparison_reports,
    }
    _assert_no_forbidden_keys(report, location="rendered report")
    for packet in packets.values():
        _assert_no_forbidden_keys(packet, location="review packet")
    if write:
        (experiment_dir / "experiment-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (experiment_dir / "experiment-report.md").write_text(
            _render_markdown(report), encoding="utf-8"
        )
        for comparison_id, packet in packets.items():
            (experiment_dir / f"review-packet-{comparison_id}.json").write_text(
                json.dumps(packet, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.experiment_dir, labels_path=args.labels)
    except ExperimentReportError as exc:
        print(f"experiment report failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"experiment report complete: {report['experiment_id']} "
        f"({len(report['arms'])} arm(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
