#!/usr/bin/env python3
"""Offline replay harness for the shadow-tier experiment (item 19).

The anti-whack-a-mole tool: every prior verification of the shadow seams
needed a live LinkedIn run. This harness constructs the REAL production
prompts (preflight via shared.preflight_v2, formation via
linkedin.strategy's builders, full-eval via linkedin.judgment_templates)
from on-disk briefs/profiles, then makes live calls through the REAL
client paths (opus_llm for the primary and the Fable shadow via
shared.strategy_shadow's dispatcher; shared.judger's _full_shadow_call for
GLM) with zero browser/LinkedIn dependency. Artifacts land in the exact
production shapes under --out and are rendered with tools/shadow_report.py
at the end — one command, human-readable model output.

Default run (formation + judge) costs roughly $1-3 in API calls: one
primary formation call (STRATEGY_MODEL_NAME), one Fable shadow formation
call (~$0.50-1, approved), one GLM full-eval call (cents).

  .venv/bin/python tools/shadow_replay.py
  .venv/bin/python tools/shadow_replay.py --stages preflight,formation,judge
  .venv/bin/python tools/shadow_replay.py --stages judge --profile-index 3

Inputs default to the 2026-07-05 SPL live run's real artifacts (the
generated v2 brief + captured profile summaries), so replayed prompts are
byte-representative of production. --seed-brief feeds the preflight stage;
--v2-brief feeds formation and judge. When preflight runs and --v2-brief
was not explicitly given, the freshly generated brief feeds the later
stages — the true production sequence.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import shared.config as config  # noqa: E402
from linkedin.page_allocator import PageObservation  # noqa: E402
from shared.contracts import SAVE_DECISIONS  # noqa: E402
from tools.glm_artifact_contract import (  # noqa: E402
    ArtifactContractError,
    resolve_external_artifact_path,
)

DEFAULT_SEED_BRIEF = REPO_ROOT / "config/spl-test/brief-spl-test-v1.json"
DEFAULT_STATE_SOURCE = REPO_ROOT / "output/state/linkedin/2078524586"
DEFAULT_V2_BRIEF = DEFAULT_STATE_SOURCE / "preflight_v2_brief.json"
DEFAULT_PROFILES = DEFAULT_STATE_SOURCE / "profile_summaries.jsonl"

# Byte-identical mirror of the primary preflight system prompt in
# linkedin/orchestrator.py::_run_preflight_v2 (named `preflight_system`
# there). Byte-identity is the experiment's contract; if the orchestrator's
# string changes, this must change with it.
PREFLIGHT_SYSTEM = (
    "You are generating structured evaluation criteria for an autonomous sourcing agent. "
    "Respond with ONLY the JSON object requested. No preamble."
)


# ---------------------------------------------------------------------------
# Manifest-driven GLM judgment matrix.  This is intentionally additive: the
# legacy strategist/shadow CLI below remains unchanged when --judgment-matrix
# is absent.
# ---------------------------------------------------------------------------

MATRIX_SCHEMA_VERSION = "glm-judgment-matrix-v2"
_MATRIX_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_MATRIX_ALLOWED_ENV = {
    "OPUS_MODEL_NAME",
    "FACIAL_MODEL_NAME",
    "FULL_EVAL_MODEL_NAME",
    "FIREWORKS_JUDGMENT_POLICY_ENABLED",
    "FIREWORKS_BASE_URL",
    "FIREWORKS_PRIMARY_MIN_MAX_TOKENS",
    "FIREWORKS_PRIMARY_MAX_COST_USD",
    "FIREWORKS_FACIAL_REASONING_EFFORT",
    "FIREWORKS_FULL_REASONING_EFFORT",
    "FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS",
    "FIREWORKS_FACIAL_TOTAL_DEADLINE_SECONDS",
    "FIREWORKS_FACIAL_MAX_ATTEMPTS",
    "FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS",
    "FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS",
    "FIREWORKS_FULL_MAX_ATTEMPTS",
    "FIREWORKS_PROMPT_AFFINITY_ENABLED",
    "SHADOW_FACIAL_MODEL_ENABLED",
    "SHADOW_STRATEGY_ENABLED",
    "SHADOW_ASYNC_ENABLED",
    "SHADOW_LLM_TIMEOUT_SECONDS",
    "LINKEDIN_V2_FACIAL_CONTRACT",
    "LINKEDIN_V2_FULL_CONTRACT",
    "LINKEDIN_FACIAL_CONCURRENCY_ENABLED",
    "LINKEDIN_FACIAL_MAX_CONCURRENCY",
    "LINKEDIN_FACIAL_TARGET_BATCH_SIZE",
    "LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
    "LINKEDIN_FACIAL_BORDERLINE_ENABLED",
    "LANGFUSE_DISABLE",
}
_MATRIX_REQUIRED_ENV = _MATRIX_ALLOWED_ENV - {"LANGFUSE_DISABLE"}
_MATRIX_MODES = {"pagewide", "partitioned_serial", "partitioned_concurrent"}
_MATRIX_RUNTIME_FILES = (
    "requirements.txt",
    "shared/config.py",
    "shared/brief_loader.py",
    "shared/brief_schema.py",
    "shared/contracts.py",
    "shared/failures.py",
    "shared/llm_clients.py",
    "shared/llm_policy.py",
    "shared/llm_spend_budget.py",
    "shared/llm_usage.py",
    "shared/observability/__init__.py",
    "shared/observability/langfuse_client.py",
    "shared/receipts.py",
    "shared/reconciliation_schemas.py",
    "shared/retrieval_design.py",
    "shared/schemas.py",
    "shared/storage.py",
    "shared/judger.py",
    "linkedin/judgment_templates.py",
    "linkedin/judgment_tool_contracts.py",
    "shared/judgment/templates.py",
    "shared/judgment/tool_contracts.py",
    "linkedin/facial_batching.py",
    "tools/glm_artifact_contract.py",
    "tools/shadow_replay.py",
    "tools/glm_experiment_report.py",
)
_MATRIX_RUNS_ROOT = (REPO_ROOT / "output/runs").resolve()
_MATRIX_MIN_INPUT_TOKENS = {"facial": 65536, "full": 131072}
_MATRIX_MIN_OUTPUT_TOKENS = {"facial": 16384, "full": 16384}
_MATRIX_MIN_FACIAL_CANDIDATES = 200
_MATRIX_MIN_FULL_CALLS = 60
_MATRIX_MIN_FACIAL_PAGE_TIMINGS = 40
_MATRIX_CACHE_WARMUP_EXECUTION_UNITS_PER_STAGE = 1
_MATRIX_INTERLEAVE_BLOCK_SIZE = 8
_MATRIX_EXECUTION_DESIGN = "deterministic-arm-block-interleaving-v1"
_MATRIX_WORKER_PROCESS_MODEL = "fresh-subprocess-per-arm-block"
_MATRIX_CACHE_CONTINUITY = "provider-side-affinity-key-across-worker-processes"
_MATRIX_EXECUTION_AUTHORIZATION_PHRASE = (
    "I AUTHORIZE PAID GLM JUDGMENT MATRIX EXECUTION"
)
_MATRIX_EXECUTION_AUTHORIZATION_MAX_AGE = timedelta(minutes=15)
_PRIVATE_PROSE_PII_ACK = (
    "I_ACKNOWLEDGE_MODEL_RATIONALES_MAY_CONTAIN_CANDIDATE_PII_IN_PRIVATE_LOCAL_ARTIFACTS"
)
_PRIVATE_PROSE_SCHEMA_VERSION = "glm-private-prose-v1"
_PRIVATE_PROSE_FILENAME = "private-pii-rationales.jsonl"
_PRIVATE_PROSE_ROW_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "manifest_hash",
        "arm_id",
        "call_id",
        "sample_id",
        "stage",
        "decision",
        "rationale",
        "row_hash",
    }
)
_MATRIX_OUTPUT_LOCK_FILENAME = ".matrix-execution.lock"
_MATRIX_WORKER_CAPABILITY_ENV = "CLORIS_MATRIX_WORKER_CAPABILITY"
_MATRIX_OUTPUT_OWNERS_GUARD = threading.Lock()
_MATRIX_OUTPUT_OWNERS: set[str] = set()
_MATRIX_BOOTSTRAP_ITERATIONS = 2_000
_MATRIX_ALLOWED_THRESHOLD_KEYS = frozenset(
    {
        "facial_page_p50_improvement_min",
        "facial_page_p50_ci_lower_min",
        "facial_page_p90_improvement_min",
        "facial_page_p90_ci_lower_min",
        "full_profile_p50_improvement_min",
        "full_profile_p50_ci_lower_min",
        "full_profile_p90_improvement_min",
        "full_profile_p90_ci_lower_min",
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
_MATRIX_LEVER_FIELDS = {
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
_MATRIX_COMMON_REQUIRED_THRESHOLDS = frozenset(
    {
        "deadline_success_min",
        "retryable_429_503_rate_max_increase",
        "postcondition_fallback_rate_max_increase",
        "unrecovered_postcondition_rate_max",
        "max_attempts",
        "human_pass_rate_delta_ci_lower_min",
    }
)
_MATRIX_PROFILE_REQUIRED_THRESHOLDS = {
    "reasoning_effort": _MATRIX_COMMON_REQUIRED_THRESHOLDS,
    "prompt_affinity": _MATRIX_COMMON_REQUIRED_THRESHOLDS,
    "judgment_contract": _MATRIX_COMMON_REQUIRED_THRESHOLDS,
    "serving_tier": _MATRIX_COMMON_REQUIRED_THRESHOLDS
    | {
        f"{stage}_{percentile}_{suffix}"
        for stage in ("facial_page", "full_profile")
        for percentile in ("p50", "p90")
        for suffix in ("improvement_min", "ci_lower_min")
    },
    "facial_concurrency": _MATRIX_COMMON_REQUIRED_THRESHOLDS
    | {
        f"facial_page_{percentile}_{suffix}"
        for percentile in ("p50", "p90")
        for suffix in ("improvement_min", "ci_lower_min")
    },
}
class MatrixValidationError(ValueError):
    """The matrix is unsafe, mutable, incomplete, or over its spend cap."""


class MatrixSpendCapExceeded(RuntimeError):
    """Measured spend cannot be proven below the authorized matrix cap."""


@dataclass(frozen=True)
class MatrixSpec:
    path: Path
    raw: dict[str, Any]
    manifest_hash: str
    experiment_id: str
    git_sha: str
    run_dir: Path
    run_id: int
    brief_json: dict[str, Any]
    file_hashes: dict[str, str]
    source_hashes: dict[str, str]
    facial_rows: tuple[tuple[int, dict[str, Any]], ...]
    full_rows: tuple[tuple[int, dict[str, Any]], ...]
    call_plans: dict[str, tuple[dict[str, Any], ...]]
    worst_case_cost_usd: float


@dataclass(frozen=True)
class MatrixExecutionAuthorization:
    phrase: str
    authorized_at: datetime


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _append_private_prose_row(path: Path, row: dict[str, Any]) -> None:
    """Append one private rationale row without following links or widening mode."""

    path = Path(path)
    if path.parent.is_symlink() or path.is_symlink():
        raise MatrixValidationError("private prose path must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        mode = stat.S_IMODE(path.stat().st_mode)
        if not path.is_file():
            raise MatrixValidationError("private prose path must be a regular file")
        if mode != 0o600:
            raise MatrixValidationError("private prose file mode must be exactly 0600")

    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise MatrixValidationError("private prose file cannot be opened safely") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise MatrixValidationError("private prose path must be a regular file")
        if stat.S_IMODE(file_stat.st_mode) != 0o600:
            raise MatrixValidationError("private prose file mode must be exactly 0600")
        encoded = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        with os.fdopen(descriptor, "ab", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _current_git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _require_matrix_id(value: object, field: str) -> str:
    text = str(value or "")
    if not _MATRIX_ID_RE.fullmatch(text):
        raise MatrixValidationError(f"{field} is missing or unsafe: {text!r}")
    return text


def _require_positive_int(value: object, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MatrixValidationError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise MatrixValidationError(f"{field} must be a positive integer")
    return parsed


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise MatrixValidationError(f"{field} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MatrixValidationError(f"{field} must be a non-negative integer") from exc
    if parsed < 0 or parsed != value:
        raise MatrixValidationError(f"{field} must be a non-negative integer")
    return parsed


def _require_matrix_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise MatrixValidationError(f"{field} must be an explicit boolean")


def _parse_timezone_aware_timestamp(value: object, field: str) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MatrixValidationError(
            f"{field} must be a valid timezone-aware ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MatrixValidationError(
            f"{field} must be a valid timezone-aware ISO-8601 timestamp"
        )
    return parsed.astimezone(timezone.utc)


def _validate_matrix_execution_authorization(
    *,
    phrase: object,
    authorized_at: object,
    now: datetime | None = None,
) -> MatrixExecutionAuthorization:
    if phrase != _MATRIX_EXECUTION_AUTHORIZATION_PHRASE:
        raise MatrixValidationError(
            "matrix execution authorization phrase does not match exactly"
        )
    parsed = _parse_timezone_aware_timestamp(
        authorized_at, "matrix execution authorization timestamp"
    )
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age = current - parsed
    if age < timedelta(0):
        raise MatrixValidationError(
            "matrix execution authorization timestamp cannot be in the future"
        )
    if age > _MATRIX_EXECUTION_AUTHORIZATION_MAX_AGE:
        raise MatrixValidationError(
            "matrix execution authorization timestamp is stale (maximum age is 15 minutes)"
        )
    return MatrixExecutionAuthorization(
        phrase=_MATRIX_EXECUTION_AUTHORIZATION_PHRASE,
        authorized_at=parsed,
    )


def _validate_private_prose_capture_request(
    *, enabled: bool, acknowledgement: object
) -> None:
    if enabled:
        if acknowledgement != _PRIVATE_PROSE_PII_ACK:
            raise MatrixValidationError(
                "private prose capture acknowledgement does not match exactly"
            )
    elif acknowledgement is not None:
        raise MatrixValidationError(
            "--ack-private-prose-pii requires --capture-private-prose"
        )


def _comparison_diff_fields(
    control: dict[str, Any], treatment: dict[str, Any]
) -> set[str]:
    changed: set[str] = set()
    if str(control.get("facial_mode") or "pagewide") != str(
        treatment.get("facial_mode") or "pagewide"
    ):
        changed.add("facial_mode")
    control_env = control["env"]
    treatment_env = treatment["env"]
    for key in sorted(set(control_env) | set(treatment_env)):
        if control_env.get(key) != treatment_env.get(key):
            changed.add(f"env.{key}")
    return changed


def _validate_threshold_profile(
    comparison_id: str,
    lever_id: str,
    thresholds: dict[str, float | int],
) -> None:
    required = _MATRIX_PROFILE_REQUIRED_THRESHOLDS[lever_id]
    missing = required - set(thresholds)
    if missing:
        raise MatrixValidationError(
            f"comparison {comparison_id} omits required {lever_id} thresholds: "
            f"{sorted(missing)}"
        )
    if float(thresholds["deadline_success_min"]) < 0.99:
        raise MatrixValidationError("deadline_success_min cannot be weaker than 0.99")
    if not 0 <= float(thresholds["retryable_429_503_rate_max_increase"]) <= 0.02:
        raise MatrixValidationError(
            "retryable_429_503_rate_max_increase cannot be weaker than 0.02"
        )
    if not 0 <= float(thresholds["postcondition_fallback_rate_max_increase"]) <= 0.02:
        raise MatrixValidationError(
            "postcondition_fallback_rate_max_increase cannot be weaker than 0.02"
        )
    if int(thresholds["max_attempts"]) > 2:
        raise MatrixValidationError("max_attempts cannot be weaker than 2")
    if float(thresholds["human_pass_rate_delta_ci_lower_min"]) < -0.03:
        raise MatrixValidationError(
            "human pass-rate noninferiority cannot be weaker than -0.03"
        )
    if float(thresholds["unrecovered_postcondition_rate_max"]) != 0:
        raise MatrixValidationError(
            "unrecovered postcondition failures require an absolute zero threshold"
        )
    for key, value in thresholds.items():
        if key.endswith("_improvement_min") and float(value) < 0.30:
            raise MatrixValidationError(
                f"{key} cannot be weaker than 0.30"
            )
        if key.endswith("_ci_lower_min") and key != "human_pass_rate_delta_ci_lower_min" and float(value) < 0.20:
            raise MatrixValidationError(
                f"{key} cannot be weaker than 0.20"
            )


def _validate_matrix_comparisons(
    raw: dict[str, Any],
    *,
    arms: list[dict[str, Any]],
    call_plans: dict[str, tuple[dict[str, Any], ...]],
) -> tuple[dict[str, Any], ...]:
    comparisons = raw.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise MatrixValidationError("comparisons must be a non-empty ordered list")
    arms_by_id = {str(arm["id"]): arm for arm in arms}
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            raise MatrixValidationError(f"comparisons[{index}] must be an object")
        unknown = set(comparison) - {
            "id",
            "lever_id",
            "depends_on",
            "control_arm_id",
            "treatment_arm_id",
            "lever_fields",
            "thresholds",
        }
        if unknown:
            raise MatrixValidationError(
                f"comparisons[{index}] has unknown fields: {sorted(unknown)}"
            )
        comparison_id = _require_matrix_id(comparison.get("id"), "comparison.id")
        if comparison_id in seen_ids:
            raise MatrixValidationError("comparison ids must be unique")
        seen_ids.add(comparison_id)
        lever_id = str(comparison.get("lever_id") or "")
        if lever_id not in _MATRIX_LEVER_FIELDS:
            raise MatrixValidationError(
                f"comparison {comparison_id} has unknown lever_id {lever_id!r}"
            )
        depends_on = comparison.get("depends_on")
        if index == 0:
            if depends_on is not None:
                raise MatrixValidationError(
                    "the first ordered comparison must have depends_on=null"
                )
        else:
            previous = normalized[-1]
            if depends_on != previous["id"]:
                raise MatrixValidationError(
                    f"comparison {comparison_id} must depend on immediately prior "
                    f"comparison {previous['id']}"
                )
        control_id = _require_matrix_id(
            comparison.get("control_arm_id"), "comparison.control_arm_id"
        )
        treatment_id = _require_matrix_id(
            comparison.get("treatment_arm_id"), "comparison.treatment_arm_id"
        )
        if control_id == treatment_id or control_id not in arms_by_id or treatment_id not in arms_by_id:
            raise MatrixValidationError(
                f"comparison {comparison_id} must name distinct existing arms"
            )
        if index and normalized[-1]["treatment_arm_id"] != control_id:
            raise MatrixValidationError(
                f"comparison {comparison_id} control arm must equal prior treatment arm"
            )
        lever_fields = comparison.get("lever_fields")
        if (
            not isinstance(lever_fields, list)
            or not lever_fields
            or any(not isinstance(value, str) or not value for value in lever_fields)
            or len(set(lever_fields)) != len(lever_fields)
        ):
            raise MatrixValidationError(
                f"comparison {comparison_id} lever_fields must be unique strings"
            )
        allowed_lever_fields = {"facial_mode"} | {
            f"env.{key}" for key in _MATRIX_ALLOWED_ENV
        }
        if not set(lever_fields).issubset(allowed_lever_fields):
            raise MatrixValidationError(
                f"comparison {comparison_id} declares unknown lever field(s)"
            )
        if set(lever_fields) != set(_MATRIX_LEVER_FIELDS[lever_id]):
            raise MatrixValidationError(
                f"comparison {comparison_id} lever_fields do not match code-owned "
                f"profile {lever_id}"
            )
        actual_diff = _comparison_diff_fields(
            arms_by_id[control_id], arms_by_id[treatment_id]
        )
        if actual_diff != set(lever_fields):
            raise MatrixValidationError(
                f"comparison {comparison_id} arm diff {sorted(actual_diff)} does not "
                f"equal declared lever_fields {sorted(lever_fields)}"
            )
        control_units = {
            plan["execution_unit_id"] for plan in call_plans[control_id]
        }
        treatment_units = {
            plan["execution_unit_id"] for plan in call_plans[treatment_id]
        }
        if control_units != treatment_units:
            raise MatrixValidationError(
                f"comparison {comparison_id} arms do not share exact execution units"
            )
        thresholds = comparison.get("thresholds", {})
        if not isinstance(thresholds, dict):
            raise MatrixValidationError(
                f"comparison {comparison_id} thresholds must be an object"
            )
        unknown_thresholds = set(thresholds) - _MATRIX_ALLOWED_THRESHOLD_KEYS
        if unknown_thresholds:
            raise MatrixValidationError(
                f"comparison {comparison_id} has unknown thresholds: "
                f"{sorted(unknown_thresholds)}"
            )
        normalized_thresholds: dict[str, float | int] = {}
        for key, raw_value in thresholds.items():
            if isinstance(raw_value, bool):
                raise MatrixValidationError(
                    f"comparison {comparison_id} threshold {key} must be numeric"
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise MatrixValidationError(
                    f"comparison {comparison_id} threshold {key} must be numeric"
                ) from exc
            if not math.isfinite(value):
                raise MatrixValidationError(
                    f"comparison {comparison_id} threshold {key} must be finite"
                )
            if key == "deadline_success_min" and not 0 <= value <= 1:
                raise MatrixValidationError(
                    f"comparison {comparison_id} threshold {key} must be in [0, 1]"
                )
            if (
                "rate" in key
                and key != "human_pass_rate_delta_ci_lower_min"
                and not 0 <= value <= 1
            ):
                raise MatrixValidationError(
                    f"comparison {comparison_id} threshold {key} must be in [0, 1]"
                )
            if (
                key.endswith("_improvement_min")
                or key.endswith("_ci_lower_min")
            ) and not -1 <= value <= 1:
                raise MatrixValidationError(
                    f"comparison {comparison_id} threshold {key} must be in [-1, 1]"
                )
            normalized_thresholds[key] = int(value) if key == "max_attempts" else value
            if key == "max_attempts" and (value < 1 or int(value) != value):
                raise MatrixValidationError(
                    f"comparison {comparison_id} max_attempts must be a positive integer"
                )
        _validate_threshold_profile(
            comparison_id, lever_id, normalized_thresholds
        )
        normalized.append(
            {
                "id": comparison_id,
                "lever_id": lever_id,
                "depends_on": depends_on,
                "control_arm_id": control_id,
                "treatment_arm_id": treatment_id,
                "lever_fields": list(lever_fields),
                "thresholds": normalized_thresholds,
            }
        )
    review = raw.get("review")
    if not isinstance(review, dict) or set(review) != {"agreement_sample_size"}:
        raise MatrixValidationError(
            "review must contain exactly agreement_sample_size"
        )
    _require_positive_int(
        review.get("agreement_sample_size"), "review.agreement_sample_size"
    )
    return tuple(normalized)


def _read_selected_jsonl(path: Path, line_numbers: object, field: str) -> tuple[tuple[int, dict], ...]:
    if not isinstance(line_numbers, list) or not line_numbers:
        raise MatrixValidationError(f"{field} must be a non-empty list of 1-based line numbers")
    selected = [_require_positive_int(value, field) for value in line_numbers]
    if len(set(selected)) != len(selected):
        raise MatrixValidationError(f"{field} contains duplicate line numbers")
    wanted = set(selected)
    found: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if number not in wanted:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MatrixValidationError(f"{path.name}:{number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise MatrixValidationError(f"{path.name}:{number} must contain a JSON object")
            found[number] = record
    missing = [number for number in selected if number not in found]
    if missing:
        raise MatrixValidationError(f"{field} references missing line(s): {missing}")
    return tuple((number, found[number]) for number in selected)


def _load_snapshot_brief(db_path: Path, run_id: int) -> dict[str, Any]:
    uri = f"file:{db_path.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            row = conn.execute(
                "SELECT brief_snapshot_json, brief_content_hash FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
    except sqlite3.Error as exc:
        raise MatrixValidationError("snapshot runtime SQLite is unreadable") from exc
    if not row or not row[0]:
        raise MatrixValidationError(f"snapshot run {run_id} has no pinned brief_snapshot_json")
    snapshot_value = row[0]
    try:
        brief = json.loads(snapshot_value)
    except json.JSONDecodeError as exc:
        raise MatrixValidationError("snapshot brief_snapshot_json is invalid") from exc
    if not isinstance(brief, dict):
        raise MatrixValidationError("snapshot brief must be a JSON object")
    # Runtime state hashes the exact serialized snapshot it persists.  Older
    # matrix fixtures (and potentially future normalized writers) pin the
    # canonical reserialization instead, so verify both representations while
    # still rejecting any value that matches neither.  Re-serializing alone
    # falsely rejected immutable production runs whose JSON whitespace/order
    # differs from this reader's canonical encoding.
    exact_bytes = (
        snapshot_value.encode("utf-8")
        if isinstance(snapshot_value, str)
        else bytes(snapshot_value)
    )
    canonical = json.dumps(brief, sort_keys=True, separators=(",", ":")).encode("utf-8")
    actual_hashes = {_sha256_bytes(exact_bytes), _sha256_bytes(canonical)}
    expected_hash = str(row[1] or "")
    accepted_hashes = actual_hashes | {
        value.removeprefix("sha256:") for value in actual_hashes
    }
    if expected_hash and expected_hash not in accepted_hashes:
        raise MatrixValidationError("snapshot brief content hash does not match SQLite")
    return brief


def _validate_finalized_run_manifest(path: Path, run_id: int) -> None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixValidationError("dataset run-manifest.json is unreadable") from exc
    if not isinstance(manifest, dict) or manifest.get("source") != "linkedin":
        raise MatrixValidationError("dataset run manifest is not a finalized LinkedIn run")
    try:
        manifest_run_id = int(manifest.get("run_id"))
    except (TypeError, ValueError) as exc:
        raise MatrixValidationError("dataset run manifest has no valid run_id") from exc
    if manifest_run_id != run_id:
        raise MatrixValidationError("dataset run_id does not match run-manifest.json")
    ended_at = str(manifest.get("ended_at") or "").strip()
    try:
        ended = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MatrixValidationError("dataset run manifest has no valid ended_at") from exc
    if ended.tzinfo is None:
        raise MatrixValidationError("dataset run manifest ended_at must be timezone-aware")
    artifacts = manifest.get("artifacts_present")
    required = {"runtime_state.sqlite3", "snippets.jsonl", "profile_summaries.jsonl"}
    if not isinstance(artifacts, list) or not required.issubset(map(str, artifacts)):
        raise MatrixValidationError(
            "dataset run manifest does not declare all required replay artifacts"
        )


def _sample_id(kind: str, file_hash: str, line_number: int) -> str:
    payload = f"{kind}\0{file_hash}\0{line_number}".encode("utf-8")
    return f"{kind}-" + hashlib.sha256(payload).hexdigest()[:20]


def _replay_candidate_ids(call_id: str, count: int) -> tuple[str, ...]:
    """Arm-independent opaque IDs for byte-comparable authorized replay."""

    return tuple(
        "cand_"
        + hashlib.sha256(f"{call_id}\0{position}".encode("utf-8")).hexdigest()[:24]
        for position in range(count)
    )


_PROMPT_HASH_FIELDS = (
    "system_prompt_sha256s",
    "candidate_prompt_sha256s",
    "tool_schema_sha256",
    "judgment_contract_version",
)


def _prompt_hash_manifest_sha256(calls: list[dict[str, Any]]) -> str:
    payload = [
        {
            "call_id": str(row.get("call_id") or ""),
            **{field: row.get(field) for field in _PROMPT_HASH_FIELDS},
        }
        for row in sorted(calls, key=lambda item: str(item.get("call_id") or ""))
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return _sha256_bytes(canonical)


def _decision_prompt_hashes(decisions: list[Any]) -> dict[str, list[str]]:
    system_hashes: set[str] = set()
    candidate_hashes: set[str] = set()
    for decision in decisions:
        capture = getattr(decision, "prompt_capture", None)
        if not isinstance(capture, dict):
            continue
        system_hash = str(capture.get("system_prompt_sha256") or "")
        if re.fullmatch(r"[0-9a-f]{64}", system_hash):
            system_hashes.add(system_hash)
        candidate_text = capture.get("candidate_text")
        if isinstance(candidate_text, str):
            candidate_hashes.add(hashlib.sha256(candidate_text.encode("utf-8")).hexdigest())
    return {
        "system_prompt_sha256s": sorted(system_hashes),
        "candidate_prompt_sha256s": sorted(candidate_hashes),
    }


def _group_facial_rows(rows: tuple[tuple[int, dict], ...]) -> list[list[tuple[int, dict]]]:
    groups: dict[tuple[object, object], list[tuple[int, dict]]] = {}
    order: list[tuple[object, object]] = []
    for line_number, record in rows:
        key = (record.get("source_string_id"), record.get("page"))
        if key == (None, None):
            key = ("line", line_number)
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append((line_number, record))
    return [groups[key] for key in order]


def _build_arm_call_plan(
    arm: dict[str, Any],
    *,
    facial_rows: tuple[tuple[int, dict], ...],
    full_rows: tuple[tuple[int, dict], ...],
    file_hashes: dict[str, str],
    facial_page_repetitions: int,
) -> tuple[dict[str, Any], ...]:
    arm_id = _require_matrix_id(arm.get("id"), "arm.id")
    unknown_arm_fields = set(arm) - {"id", "facial_mode", "env"}
    if unknown_arm_fields:
        raise MatrixValidationError(
            f"arm {arm_id} has unknown fields: {sorted(unknown_arm_fields)}"
        )
    mode = str(arm.get("facial_mode") or "pagewide")
    if mode not in _MATRIX_MODES:
        raise MatrixValidationError(f"arm {arm_id} has unknown facial_mode {mode!r}")
    env = arm.get("env")
    if not isinstance(env, dict):
        raise MatrixValidationError(f"arm {arm_id}.env must be an object")
    unknown = set(env) - _MATRIX_ALLOWED_ENV
    missing = _MATRIX_REQUIRED_ENV - set(env)
    if unknown:
        raise MatrixValidationError(f"arm {arm_id} contains forbidden env key(s): {sorted(unknown)}")
    if missing:
        raise MatrixValidationError(f"arm {arm_id} omits required env key(s): {sorted(missing)}")
    if any(not isinstance(value, (str, int, float, bool)) for value in env.values()):
        raise MatrixValidationError(f"arm {arm_id}.env values must be scalar")

    policy_enabled = _require_matrix_bool(
        env["FIREWORKS_JUDGMENT_POLICY_ENABLED"],
        f"arm {arm_id} FIREWORKS_JUDGMENT_POLICY_ENABLED",
    )
    concurrency_enabled = _require_matrix_bool(
        env["LINKEDIN_FACIAL_CONCURRENCY_ENABLED"],
        f"arm {arm_id} LINKEDIN_FACIAL_CONCURRENCY_ENABLED",
    )
    _require_matrix_bool(
        env["FIREWORKS_PROMPT_AFFINITY_ENABLED"],
        f"arm {arm_id} FIREWORKS_PROMPT_AFFINITY_ENABLED",
    )
    _require_matrix_bool(
        env["LINKEDIN_EXTERNAL_EVIDENCE_ENABLED"],
        f"arm {arm_id} LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
    )
    for disabled_key in (
        "SHADOW_FACIAL_MODEL_ENABLED",
        "SHADOW_STRATEGY_ENABLED",
        "SHADOW_ASYNC_ENABLED",
        "LINKEDIN_FACIAL_BORDERLINE_ENABLED",
    ):
        if _require_matrix_bool(env[disabled_key], f"arm {arm_id} {disabled_key}"):
            raise MatrixValidationError(
                f"arm {arm_id} requires {disabled_key}=false"
            )
    fireworks_base_url = str(env["FIREWORKS_BASE_URL"] or "").strip()
    if fireworks_base_url != "https://api.fireworks.ai/inference/v1":
        raise MatrixValidationError(
            f"arm {arm_id} FIREWORKS_BASE_URL must use the canonical Fireworks endpoint"
        )
    if _require_positive_int(
        env["FIREWORKS_PRIMARY_MIN_MAX_TOKENS"],
        f"arm {arm_id} FIREWORKS_PRIMARY_MIN_MAX_TOKENS",
    ) != 16384:
        raise MatrixValidationError(
            f"arm {arm_id} FIREWORKS_PRIMARY_MIN_MAX_TOKENS must equal 16384"
        )
    try:
        primary_cost_cap = float(env["FIREWORKS_PRIMARY_MAX_COST_USD"])
    except (TypeError, ValueError) as exc:
        raise MatrixValidationError(
            f"arm {arm_id} FIREWORKS_PRIMARY_MAX_COST_USD must equal 0"
        ) from exc
    if not math.isfinite(primary_cost_cap) or primary_cost_cap != 0:
        raise MatrixValidationError(
            f"arm {arm_id} FIREWORKS_PRIMARY_MAX_COST_USD must equal 0"
        )
    try:
        shadow_timeout = float(env["SHADOW_LLM_TIMEOUT_SECONDS"])
    except (TypeError, ValueError) as exc:
        raise MatrixValidationError(
            f"arm {arm_id} SHADOW_LLM_TIMEOUT_SECONDS must be positive"
        ) from exc
    if not math.isfinite(shadow_timeout) or shadow_timeout <= 0:
        raise MatrixValidationError(
            f"arm {arm_id} SHADOW_LLM_TIMEOUT_SECONDS must be positive"
        )
    for attempts_key in (
        "FIREWORKS_FACIAL_MAX_ATTEMPTS",
        "FIREWORKS_FULL_MAX_ATTEMPTS",
    ):
        attempts = _require_positive_int(
            env[attempts_key], f"arm {arm_id} {attempts_key}"
        )
        if attempts > 2:
            raise MatrixValidationError(
                f"arm {arm_id} {attempts_key} cannot exceed 2"
            )
    facial_contract = str(env["LINKEDIN_V2_FACIAL_CONTRACT"] or "").strip().lower()
    full_contract = str(env["LINKEDIN_V2_FULL_CONTRACT"] or "").strip().lower()
    if facial_contract not in {"legacy", "tool"} or full_contract not in {
        "legacy",
        "tool",
    }:
        raise MatrixValidationError(
            f"arm {arm_id} judgment contracts must be explicit legacy/tool values"
        )
    if "tool" in {facial_contract, full_contract} and not policy_enabled:
        raise MatrixValidationError(
            f"arm {arm_id} tool contracts require explicit Fireworks policy"
        )
    if policy_enabled:
        for model_key in ("FACIAL_MODEL_NAME", "FULL_EVAL_MODEL_NAME"):
            model = str(env[model_key] or "").strip()
            if not model.startswith("accounts/fireworks/"):
                raise MatrixValidationError(
                    f"arm {arm_id} policy requires a Fireworks {model_key}"
                )
        for effort_key in (
            "FIREWORKS_FACIAL_REASONING_EFFORT",
            "FIREWORKS_FULL_REASONING_EFFORT",
        ):
            effort = str(env[effort_key] or "").strip().lower()
            if effort not in {"high", "max"}:
                raise MatrixValidationError(
                    f"arm {arm_id} policy requires explicit high/max {effort_key}"
                )
        for stage in ("FACIAL", "FULL"):
            try:
                attempt_timeout = float(
                    env[f"FIREWORKS_{stage}_ATTEMPT_TIMEOUT_SECONDS"]
                )
                total_deadline = float(
                    env[f"FIREWORKS_{stage}_TOTAL_DEADLINE_SECONDS"]
                )
            except (TypeError, ValueError) as exc:
                raise MatrixValidationError(
                    f"arm {arm_id} {stage.lower()} policy timings must be numeric"
                ) from exc
            if (
                not math.isfinite(attempt_timeout)
                or not math.isfinite(total_deadline)
                or attempt_timeout <= 0
                or total_deadline < attempt_timeout
            ):
                raise MatrixValidationError(
                    f"arm {arm_id} {stage.lower()} policy timing envelope is invalid"
                )
    if mode == "partitioned_concurrent" and not concurrency_enabled:
        raise MatrixValidationError(f"arm {arm_id} concurrent mode requires its concurrency flag")
    if mode != "partitioned_concurrent" and concurrency_enabled:
        raise MatrixValidationError(f"arm {arm_id} enables concurrency outside concurrent mode")

    max_concurrency = _require_positive_int(
        env["LINKEDIN_FACIAL_MAX_CONCURRENCY"], f"arm {arm_id} max concurrency"
    )
    target_size = _require_positive_int(
        env["LINKEDIN_FACIAL_TARGET_BATCH_SIZE"], f"arm {arm_id} target batch size"
    )
    if max_concurrency > 3:
        raise MatrixValidationError(
            f"arm {arm_id} max concurrency cannot exceed 3"
        )
    if concurrency_enabled and max_concurrency > 1 and (
        not policy_enabled or facial_contract != "tool"
    ):
        raise MatrixValidationError(
            f"arm {arm_id} concurrency >1 requires Fireworks policy and facial tool contract"
        )

    from linkedin.facial_batching import partition_facial_batches

    plans: list[dict[str, Any]] = []
    facial_hash = file_hashes["snippets.jsonl"]
    for page_index, page_rows in enumerate(_group_facial_rows(facial_rows)):
        page_sample_ids = [
            _sample_id("facial", facial_hash, line_number)
            for line_number, _record in page_rows
        ]
        page_id = "page-" + hashlib.sha256(
            "\0".join(page_sample_ids).encode()
        ).hexdigest()[:20]
        for repeat_index in range(facial_page_repetitions):
            execution_unit_id = f"{page_id}-r{repeat_index + 1:03d}"
            if mode == "pagewide":
                slices = [(0, len(page_rows))]
            else:
                partitioned = partition_facial_batches(
                    page_rows,
                    max_concurrency=max_concurrency,
                    target_batch_size=target_size,
                )
                slices = [(batch.start, batch.stop) for batch in partitioned]
            for batch_index, (start, stop) in enumerate(slices):
                selected = page_rows[start:stop]
                sample_ids = [
                    _sample_id("facial", facial_hash, line_number)
                    for line_number, _record in selected
                ]
                call_key = "\0".join(
                    sample_ids + [execution_unit_id, str(batch_index)]
                )
                plans.append(
                    {
                        "call_id": "facial-"
                        + hashlib.sha256(call_key.encode()).hexdigest()[:20],
                        "stage": "facial",
                        "mode": mode,
                        "page_index": page_index,
                        "page_repeat_index": repeat_index,
                        "execution_unit_id": execution_unit_id,
                        "batch_index": batch_index,
                        "line_numbers": [line for line, _record in selected],
                        "sample_ids": sample_ids,
                    }
                )

    full_hash = file_hashes["profile_summaries.jsonl"]
    for line_number, _record in full_rows:
        sample_id = _sample_id("full", full_hash, line_number)
        plans.append(
            {
                "call_id": "full-" + hashlib.sha256(sample_id.encode()).hexdigest()[:20],
                "stage": "full",
                "mode": "serial",
                "execution_unit_id": "profile-" + hashlib.sha256(sample_id.encode()).hexdigest()[:20],
                "line_numbers": [line_number],
                "sample_ids": [sample_id],
            }
        )
    return tuple(plans)


def _cache_phased_execution_order(
    plans: tuple[dict[str, Any], ...],
    *,
    random_seed: int,
    warmup_units_per_stage: int,
) -> tuple[list[str], dict[str, str]]:
    """Return a paired warmup prefix followed by a randomized warm block."""

    units_by_stage: dict[str, list[str]] = {"facial": [], "full": []}
    unit_stage: dict[str, str] = {}
    for plan in plans:
        unit_id = str(plan["execution_unit_id"])
        stage = str(plan["stage"])
        prior_stage = unit_stage.setdefault(unit_id, stage)
        if prior_stage != stage or stage not in units_by_stage:
            raise MatrixValidationError(
                "execution units must belong to exactly one judgment stage"
            )
        if unit_id not in units_by_stage[stage]:
            units_by_stage[stage].append(unit_id)

    warmup: list[str] = []
    warm: list[str] = []
    for stage in ("facial", "full"):
        units = list(units_by_stage[stage])
        stage_seed = int.from_bytes(
            hashlib.sha256(
                f"{random_seed}\0cache-block\0{stage}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        random.Random(stage_seed).shuffle(units)
        if len(units) <= warmup_units_per_stage:
            raise MatrixValidationError(
                f"cache warm-block design requires at least one measured {stage} "
                "execution unit after warmup"
            )
        warmup.extend(units[:warmup_units_per_stage])
        warm.extend(units[warmup_units_per_stage:])

    warmup_seed = int.from_bytes(
        hashlib.sha256(f"{random_seed}\0cache-warmup".encode("utf-8")).digest()[:8],
        "big",
    )
    warm_seed = int.from_bytes(
        hashlib.sha256(f"{random_seed}\0cache-warm".encode("utf-8")).digest()[:8],
        "big",
    )
    random.Random(warmup_seed).shuffle(warmup)
    random.Random(warm_seed).shuffle(warm)
    phases = {
        **{unit_id: "warmup" for unit_id in warmup},
        **{unit_id: "warm" for unit_id in warm},
    }
    return warmup + warm, phases


def _matrix_block_arm_order(
    arm_ids: list[str], *, random_seed: int, block_index: int
) -> list[str]:
    """Return the reproducible arm order for one sequential provider block."""

    if not arm_ids or len(set(arm_ids)) != len(arm_ids):
        raise MatrixValidationError("block schedule requires unique arm ids")
    if block_index < 0:
        raise MatrixValidationError("block schedule index cannot be negative")
    seed = int.from_bytes(
        hashlib.sha256(
            f"{random_seed}\0arm-block\0{block_index}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    ordered = list(arm_ids)
    random.Random(seed).shuffle(ordered)
    return ordered


def _matrix_block_schedule(
    arm_ids: list[str],
    execution_units: list[str],
    *,
    random_seed: int,
    block_size: int = _MATRIX_INTERLEAVE_BLOCK_SIZE,
) -> list[dict[str, Any]]:
    """Partition paired units and interleave arms without concurrent load.

    Each arm keeps the exact same unit order.  A fresh process is used for an
    arm/block dispatch; cache continuity intentionally relies on the
    provider-side affinity key rather than process-local client state.
    """

    if block_size <= 1:
        raise MatrixValidationError("matrix interleave block size must be greater than 1")
    if not execution_units or len(set(execution_units)) != len(execution_units):
        raise MatrixValidationError(
            "block schedule requires unique execution units"
        )
    schedule: list[dict[str, Any]] = []
    for block_index, start in enumerate(range(0, len(execution_units), block_size)):
        schedule.append(
            {
                "block_index": block_index,
                "execution_unit_ids": execution_units[start : start + block_size],
                "arm_order": _matrix_block_arm_order(
                    arm_ids,
                    random_seed=random_seed,
                    block_index=block_index,
                ),
            }
        )
    return schedule


def _matrix_schedule_sha256(schedule: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        schedule, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _paired_matrix_execution_design(
    spec: MatrixSpec,
) -> tuple[
    dict[str, tuple[list[str], dict[str, str]]],
    list[dict[str, Any]],
]:
    """Build one exact unit order and block schedule shared by every arm."""

    warmup_units_per_stage = int(
        spec.raw["dataset"]["cache_warmup_execution_units_per_stage"]
    )
    phased_orders = {
        arm_id: _cache_phased_execution_order(
            plans,
            random_seed=int(spec.raw["random_seed"]),
            warmup_units_per_stage=warmup_units_per_stage,
        )
        for arm_id, plans in spec.call_plans.items()
    }
    arm_ids = list(phased_orders)
    first_order, first_phases = phased_orders[arm_ids[0]]
    for arm_id in arm_ids[1:]:
        order, phases = phased_orders[arm_id]
        if order != first_order or phases != first_phases:
            raise MatrixValidationError(
                f"matrix arm {arm_id} does not share the exact paired execution order"
            )
    schedule = _matrix_block_schedule(
        arm_ids,
        first_order,
        random_seed=int(spec.raw["random_seed"]),
    )
    return phased_orders, schedule


def _estimate_worst_case_cost(
    raw: dict[str, Any],
    call_plans: dict[str, tuple[dict[str, Any], ...]],
) -> float:
    from shared.llm_usage import estimate_usage_cost_usd

    limits = raw.get("limits")
    if not isinstance(limits, dict):
        raise MatrixValidationError("limits must be an object")
    stage_limits: dict[str, tuple[int, int]] = {}
    for stage in ("facial", "full"):
        input_limit = _require_positive_int(
            limits.get(f"{stage}_max_input_tokens"),
            f"limits.{stage}_max_input_tokens",
        )
        output_limit = _require_positive_int(
            limits.get(f"{stage}_max_output_tokens"),
            f"limits.{stage}_max_output_tokens",
        )
        if input_limit < _MATRIX_MIN_INPUT_TOKENS[stage]:
            raise MatrixValidationError(
                f"limits.{stage}_max_input_tokens must be at least "
                f"{_MATRIX_MIN_INPUT_TOKENS[stage]} (conservative rendered-prompt ceiling)"
            )
        if output_limit < _MATRIX_MIN_OUTPUT_TOKENS[stage]:
            raise MatrixValidationError(
                f"limits.{stage}_max_output_tokens must be at least "
                f"{_MATRIX_MIN_OUTPUT_TOKENS[stage]} (production Fireworks request floor)"
            )
        stage_limits[stage] = (input_limit, output_limit)
    totals = 0.0
    arms = {str(arm["id"]): arm for arm in raw["arms"]}
    for arm_id, plans in call_plans.items():
        env = arms[arm_id]["env"]
        for plan in plans:
            stage = plan["stage"]
            model_key = "FACIAL_MODEL_NAME" if stage == "facial" else "FULL_EVAL_MODEL_NAME"
            attempt_key = (
                "FIREWORKS_FACIAL_MAX_ATTEMPTS"
                if stage == "facial"
                else "FIREWORKS_FULL_MAX_ATTEMPTS"
            )
            input_tokens, output_tokens = stage_limits[stage]
            attempts = _require_positive_int(env[attempt_key], f"arm {arm_id} {attempt_key}")
            cost, _rate_source = estimate_usage_cost_usd(
                model=str(env[model_key]),
                input_tokens=input_tokens * attempts,
                output_tokens=output_tokens * attempts,
            )
            if cost is None:
                raise MatrixValidationError(
                    f"no price is registered for arm {arm_id} model {env[model_key]!r}"
                )
            totals += float(cost)
    return round(totals, 6)


def _load_and_validate_matrix_manifest(path: Path) -> MatrixSpec:
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixValidationError(f"cannot read matrix manifest: {path}") from exc
    if not isinstance(raw, dict):
        raise MatrixValidationError("matrix manifest must be a JSON object")
    if raw.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise MatrixValidationError(f"schema_version must be {MATRIX_SCHEMA_VERSION!r}")
    experiment_id = _require_matrix_id(raw.get("experiment_id"), "experiment_id")
    random_seed = _require_positive_int(raw.get("random_seed"), "random_seed")
    git_sha = str(raw.get("git_sha") or "")
    if git_sha != _current_git_sha():
        raise MatrixValidationError("manifest git_sha does not match the current checkout")
    declared_source_hashes = raw.get("source_hashes")
    if not isinstance(declared_source_hashes, dict) or set(declared_source_hashes) != set(
        _MATRIX_RUNTIME_FILES
    ):
        raise MatrixValidationError(
            f"source_hashes must contain exactly {list(_MATRIX_RUNTIME_FILES)}"
        )
    source_hashes: dict[str, str] = {}
    for relative in _MATRIX_RUNTIME_FILES:
        actual = _sha256_file(REPO_ROOT / relative)
        if str(declared_source_hashes[relative]) != actual:
            raise MatrixValidationError(f"runtime source hash mismatch for {relative}")
        source_hashes[relative] = actual

    authorization = raw.get("authorization")
    if not isinstance(authorization, dict) or authorization.get("approved") is not True:
        raise MatrixValidationError("matrix requires explicit authorization.approved=true")
    if not str(authorization.get("approved_by") or "").strip():
        raise MatrixValidationError("matrix authorization requires approved_by")
    _parse_timezone_aware_timestamp(
        authorization.get("approved_at"), "authorization.approved_at"
    )
    raw_dollar_cap = authorization.get("max_cost_usd")
    if isinstance(raw_dollar_cap, bool):
        raise MatrixValidationError(
            "authorization.max_cost_usd must be finite and positive"
        )
    try:
        dollar_cap = float(raw_dollar_cap)
    except (TypeError, ValueError) as exc:
        raise MatrixValidationError(
            "authorization.max_cost_usd must be finite and positive"
        ) from exc
    if not math.isfinite(dollar_cap) or dollar_cap <= 0:
        raise MatrixValidationError(
            "authorization.max_cost_usd must be finite and positive"
        )

    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        raise MatrixValidationError("dataset must be an object")
    run_dir = Path(str(dataset.get("run_dir") or "")).expanduser().resolve()
    try:
        run_relative = run_dir.relative_to(_MATRIX_RUNS_ROOT)
    except ValueError as exc:
        raise MatrixValidationError(
            f"dataset.run_dir must be inside this checkout's {_MATRIX_RUNS_ROOT}"
        ) from exc
    if len(run_relative.parts) < 3:
        raise MatrixValidationError("dataset.run_dir must identify source, state key, and finalized run")
    if not (run_dir / "run-manifest.json").is_file():
        raise MatrixValidationError("dataset.run_dir is missing run-manifest.json")
    run_id = _require_positive_int(dataset.get("run_id"), "dataset.run_id")
    _validate_finalized_run_manifest(run_dir / "run-manifest.json", run_id)

    hashes = dataset.get("hashes")
    required_files = ("runtime_state.sqlite3", "snippets.jsonl", "profile_summaries.jsonl")
    if not isinstance(hashes, dict) or set(hashes) != set(required_files):
        raise MatrixValidationError(f"dataset.hashes must contain exactly {list(required_files)}")
    file_hashes: dict[str, str] = {}
    for name in required_files:
        file_path = run_dir / name
        if not file_path.is_file():
            raise MatrixValidationError(f"snapshot is missing {name}")
        actual = _sha256_file(file_path)
        if str(hashes[name]) != actual:
            raise MatrixValidationError(f"snapshot hash mismatch for {name}")
        file_hashes[name] = actual

    brief_json = _load_snapshot_brief(run_dir / "runtime_state.sqlite3", run_id)
    facial_rows = _read_selected_jsonl(
        run_dir / "snippets.jsonl",
        dataset.get("facial_line_numbers"),
        "dataset.facial_line_numbers",
    )
    full_rows = _read_selected_jsonl(
        run_dir / "profile_summaries.jsonl",
        dataset.get("full_line_numbers"),
        "dataset.full_line_numbers",
    )
    facial_page_repetitions = _require_positive_int(
        dataset.get("facial_page_repetitions"),
        "dataset.facial_page_repetitions",
    )
    cache_warmup_units_per_stage = _require_nonnegative_int(
        dataset.get("cache_warmup_execution_units_per_stage"),
        "dataset.cache_warmup_execution_units_per_stage",
    )
    if (
        cache_warmup_units_per_stage
        != _MATRIX_CACHE_WARMUP_EXECUTION_UNITS_PER_STAGE
    ):
        raise MatrixValidationError(
            "dataset.cache_warmup_execution_units_per_stage must equal 1"
        )
    facial_page_count = len(_group_facial_rows(facial_rows))
    unique_facial_candidates = len(
        {str(record.get("profile_url") or "") for _line, record in facial_rows}
    )
    if unique_facial_candidates < _MATRIX_MIN_FACIAL_CANDIDATES:
        raise MatrixValidationError(
            "promotion-grade matrix requires at least "
            f"{_MATRIX_MIN_FACIAL_CANDIDATES} unique facial candidates"
        )
    if len(full_rows) < _MATRIX_MIN_FULL_CALLS:
        raise MatrixValidationError(
            "promotion-grade matrix requires at least "
            f"{_MATRIX_MIN_FULL_CALLS} full calls"
        )
    facial_timing_count = facial_page_count * facial_page_repetitions
    if facial_timing_count < _MATRIX_MIN_FACIAL_PAGE_TIMINGS:
        raise MatrixValidationError(
            "promotion-grade matrix requires at least "
            f"{_MATRIX_MIN_FACIAL_PAGE_TIMINGS} facial page timings per arm; "
            f"selected pages x facial_page_repetitions yields {facial_timing_count}"
        )
    from shared.schemas import CandidateProfileSummary, CandidateSnippet

    try:
        for _line_number, record in facial_rows:
            CandidateSnippet.from_dict(record)
        for _line_number, record in full_rows:
            CandidateProfileSummary.from_dict(record)
    except (KeyError, TypeError, ValueError) as exc:
        raise MatrixValidationError("selected dataset row does not match production schema") from exc

    arms = raw.get("arms")
    if not isinstance(arms, list) or not arms:
        raise MatrixValidationError("arms must be a non-empty list")
    arm_ids = [_require_matrix_id(arm.get("id"), "arm.id") for arm in arms if isinstance(arm, dict)]
    if len(arm_ids) != len(arms) or len(set(arm_ids)) != len(arm_ids):
        raise MatrixValidationError("arms must be objects with unique ids")
    call_plans = {
        arm_id: _build_arm_call_plan(
            arm,
            facial_rows=facial_rows,
            full_rows=full_rows,
            file_hashes=file_hashes,
            facial_page_repetitions=facial_page_repetitions,
        )
        for arm_id, arm in zip(arm_ids, arms)
    }
    for plans in call_plans.values():
        _cache_phased_execution_order(
            plans,
            random_seed=random_seed,
            warmup_units_per_stage=cache_warmup_units_per_stage,
        )
    raw["comparisons"] = list(
        _validate_matrix_comparisons(raw, arms=arms, call_plans=call_plans)
    )
    worst_case = _estimate_worst_case_cost(raw, call_plans)
    if worst_case > dollar_cap:
        raise MatrixValidationError(
            f"worst-case matrix cost ${worst_case:.2f} exceeds authorization cap ${dollar_cap:.2f}"
        )

    return MatrixSpec(
        path=path.resolve(),
        raw=raw,
        manifest_hash=_sha256_bytes(raw_bytes),
        experiment_id=experiment_id,
        git_sha=git_sha,
        run_dir=run_dir,
        run_id=run_id,
        brief_json=brief_json,
        file_hashes=file_hashes,
        source_hashes=source_hashes,
        facial_rows=facial_rows,
        full_rows=full_rows,
        call_plans=call_plans,
        worst_case_cost_usd=worst_case,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _measured_usage_cost(path: Path) -> float:
    """Return exact measured aggregate cost, failing closed on unknown usage."""

    if not path.exists():
        return 0.0
    total = 0.0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MatrixSpendCapExceeded(
                    f"invalid usage JSONL at {path}:{line_number}"
                ) from exc
            if row.get("usage_status") != "measured":
                raise MatrixSpendCapExceeded(
                    "matrix usage is partial/unavailable; exact spend cap cannot be enforced"
                )
            if row.get("cost_completeness") != "complete":
                raise MatrixSpendCapExceeded(
                    "matrix usage cost is not complete; exact spend cap cannot be enforced"
                )
            try:
                cost = float(row["estimated_cost_usd"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MatrixSpendCapExceeded(
                    "matrix usage row lacks measured estimated_cost_usd"
                ) from exc
            if not math.isfinite(cost) or cost < 0:
                raise MatrixSpendCapExceeded(
                    "matrix usage row has invalid estimated_cost_usd"
                )
            total += cost
    return total


def _assert_exact_usage_receipt(path: Path, logical_call_id: str) -> dict[str, Any]:
    """Stop replay immediately when the just-finished call lacks telemetry."""

    if not path.is_file():
        raise MatrixSpendCapExceeded(
            f"matrix call {logical_call_id} has no measured usage receipt"
        )
    matches: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MatrixSpendCapExceeded(
                    f"invalid usage JSONL at {path}:{line_number}"
                ) from exc
            if str(row.get("logical_call_id") or row.get("call_id") or "") == logical_call_id:
                matches.append(row)
                if row.get("usage_status") != "measured":
                    raise MatrixSpendCapExceeded(
                        f"matrix call {logical_call_id} usage is partial/unavailable"
                    )
    if len(matches) != 1:
        raise MatrixSpendCapExceeded(
            f"matrix call {logical_call_id} has {len(matches)} aggregate usage receipts; expected 1"
        )
    return matches[0]


def _fallback_receipt_count(path: Path, parent_logical_call_id: str) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                str(row.get("parent_logical_call_id") or "")
                == parent_logical_call_id
                and str(row.get("fallback_reason") or "")
            ):
                count += 1
    return count


def _completed_matrix_cost(out_dir: Path) -> float:
    return sum(
        _measured_usage_cost(path)
        for path in sorted((out_dir / "arms").glob("*/token-cost-log.jsonl"))
    )


def _assert_arm_usage_within_declared_limits(
    path: Path,
    *,
    spec: MatrixSpec,
    arm: dict[str, Any],
) -> None:
    if not path.exists():
        return
    limits = spec.raw["limits"]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            stage_word = str(row.get("stage") or "").lower()
            if "facial" in stage_word:
                stage = "facial"
            elif "full" in stage_word:
                stage = "full"
            else:
                raise MatrixSpendCapExceeded(
                    "matrix usage row has unknown judgment stage"
                )
            attempts_key = (
                "FIREWORKS_FACIAL_MAX_ATTEMPTS"
                if stage == "facial"
                else "FIREWORKS_FULL_MAX_ATTEMPTS"
            )
            attempts = int(arm["env"][attempts_key])
            input_ceiling = int(limits[f"{stage}_max_input_tokens"]) * attempts
            output_ceiling = int(limits[f"{stage}_max_output_tokens"]) * attempts
            try:
                token_values = {
                    key: int(row[key])
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise MatrixSpendCapExceeded(
                    "matrix measured usage row lacks complete token counts"
                ) from exc
            if any(value < 0 for value in token_values.values()):
                raise MatrixSpendCapExceeded(
                    "matrix measured usage row has negative token counts"
                )
            measured_prompt_tokens = sum(
                token_values[key]
                for key in (
                    "input_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
            )
            if measured_prompt_tokens > input_ceiling:
                raise MatrixSpendCapExceeded(
                    f"measured {stage} input exceeded the manifest worst-case ceiling"
                )
            if token_values["output_tokens"] > output_ceiling:
                raise MatrixSpendCapExceeded(
                    f"measured {stage} output exceeded the manifest worst-case ceiling"
                )


@contextmanager
def _matrix_output_ownership(out_dir: Path, *, resume: bool):
    """Own one matrix output exclusively across threads and processes."""

    resolved = out_dir.resolve()
    owner_key = str(resolved)
    with _MATRIX_OUTPUT_OWNERS_GUARD:
        if owner_key in _MATRIX_OUTPUT_OWNERS:
            raise MatrixValidationError(
                "matrix output is already owned by another parent"
            )
        _MATRIX_OUTPUT_OWNERS.add(owner_key)

    descriptor: int | None = None
    try:
        if resume:
            if not resolved.is_dir():
                raise MatrixValidationError(
                    "matrix resume requires an existing output directory"
                )
        else:
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.mkdir()
            except FileExistsError as exc:
                raise MatrixValidationError(
                    "matrix output already exists; use verified --matrix-resume"
                ) from exc
            except OSError as exc:
                raise MatrixValidationError(
                    "matrix output directory cannot be created"
                ) from exc

        lock_path = resolved / _MATRIX_OUTPUT_LOCK_FILENAME
        try:
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            raise MatrixValidationError(
                "matrix output ownership lock cannot be opened"
            ) from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MatrixValidationError(
                "matrix output is already owned by another parent"
            ) from exc
        owner_record = json.dumps(
            {
                "pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
        try:
            os.ftruncate(descriptor, 0)
            os.write(descriptor, owner_record)
            os.fsync(descriptor)
        except OSError as exc:
            raise MatrixValidationError(
                "matrix output ownership record cannot be persisted"
            ) from exc
        yield
    finally:
        try:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
        finally:
            with _MATRIX_OUTPUT_OWNERS_GUARD:
                _MATRIX_OUTPUT_OWNERS.discard(owner_key)


def _new_matrix_worker_capability(metadata: dict[str, Any]) -> str:
    token = secrets.token_urlsafe(32)
    metadata["worker_capability_sha256"] = _sha256_bytes(
        token.encode("utf-8")
    )
    return token


def _validate_internal_matrix_worker(arm_dir: Path) -> None:
    """Require a live owning parent and its process-local worker capability."""

    resolved_arm_dir = arm_dir.resolve()
    if resolved_arm_dir.parent.name != "arms":
        raise MatrixValidationError("matrix worker output layout is invalid")
    experiment_dir = resolved_arm_dir.parent.parent
    metadata = _read_matrix_metadata(
        experiment_dir / "experiment-metadata.json",
        label="parent metadata for internal worker",
    )
    token = os.environ.get(_MATRIX_WORKER_CAPABILITY_ENV, "")
    expected_hash = str(metadata.get("worker_capability_sha256") or "")
    actual_hash = _sha256_bytes(token.encode("utf-8")) if token else ""
    if (
        not token
        or len(expected_hash) != len("sha256:") + 64
        or not secrets.compare_digest(actual_hash, expected_hash)
    ):
        raise MatrixValidationError(
            "matrix worker is internal and lacks its parent capability"
        )

    lock_path = experiment_dir / _MATRIX_OUTPUT_LOCK_FILENAME
    try:
        descriptor = os.open(lock_path, os.O_RDWR)
    except OSError as exc:
        raise MatrixValidationError(
            "matrix worker has no live owning parent"
        ) from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            raise MatrixValidationError(
                "matrix worker has no live owning parent"
            )
    finally:
        os.close(descriptor)


def _build_matrix_parent_metadata(
    spec: MatrixSpec, *, capture_private_prose: bool = False
) -> dict[str, Any]:
    arms = {str(arm["id"]): arm for arm in spec.raw["arms"]}
    warmup_units_per_stage = int(
        spec.raw["dataset"]["cache_warmup_execution_units_per_stage"]
    )
    phased_orders, execution_schedule = _paired_matrix_execution_design(spec)
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "experiment_id": spec.experiment_id,
        "manifest_hash": spec.manifest_hash,
        "git_sha": spec.git_sha,
        "file_hashes": spec.file_hashes,
        "source_hashes": spec.source_hashes,
        "random_seed": int(spec.raw["random_seed"]),
        "worst_case_cost_usd": spec.worst_case_cost_usd,
        "authorization_cap_usd": float(
            spec.raw["authorization"]["max_cost_usd"]
        ),
        "arm_ids": list(arms),
        "comparisons": spec.raw["comparisons"],
        "review": spec.raw["review"],
        "bootstrap_iterations": _MATRIX_BOOTSTRAP_ITERATIONS,
        "execution_design": _MATRIX_EXECUTION_DESIGN,
        "execution_block_size": _MATRIX_INTERLEAVE_BLOCK_SIZE,
        "execution_schedule": execution_schedule,
        "execution_schedule_sha256": _matrix_schedule_sha256(
            execution_schedule
        ),
        "worker_process_model": _MATRIX_WORKER_PROCESS_MODEL,
        "cache_continuity": _MATRIX_CACHE_CONTINUITY,
        "private_prose_capture": {
            "enabled": bool(capture_private_prose),
            "schema_version": (
                _PRIVATE_PROSE_SCHEMA_VERSION if capture_private_prose else None
            ),
        },
        "dispatch_log": [],
        "expected_calls": {
            arm_id: [plan["call_id"] for plan in plans]
            for arm_id, plans in spec.call_plans.items()
        },
        "expected_units": {
            arm_id: phased_orders[arm_id][0]
            for arm_id in spec.call_plans
        },
        "expected_unit_cache_phases": {
            arm_id: phased_orders[arm_id][1]
            for arm_id in spec.call_plans
        },
        "expected_unit_calls": {
            arm_id: {
                unit_id: [
                    plan["call_id"]
                    for plan in plans
                    if plan["execution_unit_id"] == unit_id
                ]
                for unit_id in dict.fromkeys(
                    plan["execution_unit_id"] for plan in plans
                )
            }
            for arm_id, plans in spec.call_plans.items()
        },
        "sample_design": {
            "unique_facial_candidates": len(
                {
                    str(record.get("profile_url") or "")
                    for _line, record in spec.facial_rows
                }
            ),
            "full_calls": len(spec.full_rows),
            "facial_page_repetitions": int(
                spec.raw["dataset"]["facial_page_repetitions"]
            ),
            "cache_warmup_execution_units_per_stage": warmup_units_per_stage,
            "cache_block_design": "per-stage-warmup-prefix-v1",
            "facial_page_timing_units_per_arm": len(
                {
                    plan["execution_unit_id"]
                    for plan in next(iter(spec.call_plans.values()))
                    if plan["stage"] == "facial"
                }
            ),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "offline_replay": True,
    }


def _matrix_output_dir(spec: MatrixSpec, requested: Path | None) -> Path:
    del spec
    if requested is None:
        raise MatrixValidationError(
            "matrix parent requires an explicit --out outside protected paths"
        )
    try:
        return resolve_external_artifact_path(
            requested,
            label="matrix --out",
        )
    except ArtifactContractError as exc:
        raise MatrixValidationError(
            f"matrix --out must be outside the repository and Chrome profile: {exc}"
        ) from exc


def _matrix_dispatch_plan(
    execution_schedule: list[dict[str, Any]],
) -> list[tuple[int, str]]:
    return [
        (int(block["block_index"]), str(arm_id))
        for block in execution_schedule
        for arm_id in block["arm_order"]
    ]


def _read_matrix_metadata(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixValidationError(f"matrix resume lacks valid {label}") from exc
    if not isinstance(payload, dict):
        raise MatrixValidationError(f"matrix resume lacks valid {label}")
    return payload


def _verify_matrix_resume_boundary(
    spec: MatrixSpec,
    out_dir: Path,
    expected_metadata: dict[str, Any],
) -> tuple[dict[str, Any], int, float]:
    """Return a verified clean dispatch boundary; never infer partial success."""

    metadata = _read_matrix_metadata(
        out_dir / "experiment-metadata.json", label="experiment metadata"
    )
    pinned_keys = set(expected_metadata) - {"created_at", "dispatch_log"}
    if any(metadata.get(key) != expected_metadata[key] for key in pinned_keys):
        raise MatrixValidationError(
            "matrix resume metadata does not match the pinned manifest/source/schedule"
        )
    allowed_keys = set(expected_metadata) | {
        "status",
        "measured_cost_usd",
        "resume_count",
        "last_resumed_at",
        "worker_capability_sha256",
    }
    if set(metadata) - allowed_keys:
        raise MatrixValidationError("matrix resume metadata has unknown fields")
    if not re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        str(metadata.get("worker_capability_sha256") or ""),
    ):
        raise MatrixValidationError(
            "matrix resume metadata lacks a valid worker capability hash"
        )
    if metadata.get("status") not in {"running", "reporting"}:
        raise MatrixValidationError(
            "matrix resume refuses complete, failed, or spend-stopped experiments"
        )
    _parse_timezone_aware_timestamp(metadata.get("created_at"), "matrix created_at")
    resume_count = metadata.get("resume_count", 0)
    if (
        isinstance(resume_count, bool)
        or not isinstance(resume_count, int)
        or resume_count < 0
    ):
        raise MatrixValidationError("matrix resume_count is invalid")
    if "last_resumed_at" in metadata:
        _parse_timezone_aware_timestamp(
            metadata["last_resumed_at"], "matrix last_resumed_at"
        )

    dispatch_plan = _matrix_dispatch_plan(expected_metadata["execution_schedule"])
    dispatch_log = metadata.get("dispatch_log")
    if not isinstance(dispatch_log, list) or len(dispatch_log) > len(dispatch_plan):
        raise MatrixValidationError("matrix resume dispatch log is invalid")
    actual_dispatches: list[tuple[int, str]] = []
    for row in dispatch_log:
        if (
            not isinstance(row, dict)
            or set(row) != {"block_index", "arm_id", "status"}
            or row.get("status") != "complete"
        ):
            raise MatrixValidationError(
                "matrix resume refuses a partial, in-flight, or failed current block"
            )
        try:
            actual_dispatches.append(
                (int(row["block_index"]), str(row["arm_id"]))
            )
        except (TypeError, ValueError) as exc:
            raise MatrixValidationError(
                "matrix resume dispatch log is invalid"
            ) from exc
    if actual_dispatches != dispatch_plan[: len(actual_dispatches)]:
        raise MatrixValidationError(
            "matrix resume dispatch log does not match the pinned schedule prefix"
        )

    arms = {str(arm["id"]): arm for arm in spec.raw["arms"]}
    phased_orders, execution_schedule = _paired_matrix_execution_design(spec)
    completed_blocks_by_arm: dict[str, list[int]] = {
        arm_id: [] for arm_id in arms
    }
    for block_index, arm_id in actual_dispatches:
        completed_blocks_by_arm[arm_id].append(block_index)

    arms_root = out_dir / "arms"
    if arms_root.exists():
        unexpected_entries = {
            entry.name for entry in arms_root.iterdir() if entry.name not in arms
        }
        if unexpected_entries:
            raise MatrixValidationError(
                "matrix resume contains unexpected arm artifacts"
            )

    measured_by_arm: dict[str, float] = {}
    for arm_id, arm in arms.items():
        completed_blocks = completed_blocks_by_arm[arm_id]
        if completed_blocks != list(range(len(completed_blocks))):
            raise MatrixValidationError(
                f"matrix resume arm {arm_id} does not have a contiguous block prefix"
            )
        arm_dir = out_dir / "arms" / arm_id
        if not completed_blocks:
            if arm_dir.exists():
                raise MatrixValidationError(
                    f"matrix resume arm {arm_id} has uncommitted partial artifacts"
                )
            measured_by_arm[arm_id] = 0.0
            continue

        arm_metadata = _read_matrix_metadata(
            arm_dir / "arm-metadata.json", label=f"arm {arm_id} metadata"
        )
        effective_env = {
            key: (
                "1"
                if key == "LANGFUSE_DISABLE"
                else str(arm["env"].get(key, ""))
            )
            for key in sorted(_MATRIX_ALLOWED_ENV)
        }
        execution_units, cache_phases = phased_orders[arm_id]
        expected_arm_metadata = _build_matrix_arm_expected_metadata(
            spec,
            arm=arm,
            arm_id=arm_id,
            plans=spec.call_plans[arm_id],
            execution_units=execution_units,
            cache_phases=cache_phases,
            execution_schedule=execution_schedule,
            effective_env=effective_env,
            capture_private_prose=bool(
                expected_metadata["private_prose_capture"]["enabled"]
            ),
        )
        if any(
            arm_metadata.get(key) != value
            for key, value in expected_arm_metadata.items()
        ):
            raise MatrixValidationError(
                f"matrix resume arm {arm_id} metadata drifted"
            )
        completed_count = len(completed_blocks)
        expected_status = (
            "complete"
            if completed_count == len(execution_schedule)
            else "running"
        )
        if (
            arm_metadata.get("status") != expected_status
            or arm_metadata.get("completed_block_indices")
            != list(range(completed_count))
            or "current_block_index" in arm_metadata
        ):
            raise MatrixValidationError(
                f"matrix resume arm {arm_id} is not at a completed block boundary"
            )
        offsets = arm_metadata.get("block_cost_offsets_usd")
        if (
            not isinstance(offsets, list)
            or len(offsets) != completed_count
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in offsets
            )
        ):
            raise MatrixValidationError(
                f"matrix resume arm {arm_id} spend offsets are invalid"
            )
        completed_units = [
            unit_id
            for block in execution_schedule[:completed_count]
            for unit_id in block["execution_unit_ids"]
        ]
        _validate_matrix_worker_prefix(
            arm_dir,
            arm_id=arm_id,
            plans=spec.call_plans[arm_id],
            completed_units=completed_units,
            experiment_id=spec.experiment_id,
            manifest_hash=spec.manifest_hash,
            capture_private_prose=bool(
                expected_metadata["private_prose_capture"]["enabled"]
            ),
        )
        measured_by_arm[arm_id] = _validate_matrix_worker_receipts(
            arm_dir,
            arm_id=arm_id,
            arm=arm,
            spec=spec,
            plans=spec.call_plans[arm_id],
            completed_units=completed_units,
        )

    measured_total = sum(measured_by_arm.values())
    recomputed_total = _completed_matrix_cost(out_dir)
    if not math.isclose(measured_total, recomputed_total, abs_tol=1e-9):
        raise MatrixValidationError(
            "matrix resume measured spend does not reconcile across arms"
        )
    if metadata.get("status") == "reporting":
        try:
            reported_cost = float(metadata["measured_cost_usd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MatrixValidationError(
                "matrix resume reporting state lacks valid measured spend"
            ) from exc
        if (
            not math.isfinite(reported_cost)
            or len(actual_dispatches) != len(dispatch_plan)
            or not math.isclose(
                reported_cost,
                measured_total,
                abs_tol=1e-6,
            )
        ):
            raise MatrixValidationError(
                "matrix resume reporting state is incomplete or spend-drifted"
            )
    elif "measured_cost_usd" in metadata:
        raise MatrixValidationError(
            "matrix resume running state has unexpected finalized spend"
        )
    return metadata, len(actual_dispatches), measured_total


def _run_matrix_parent(
    spec: MatrixSpec,
    out_dir: Path,
    *,
    validate_only: bool,
    execution_authorization: MatrixExecutionAuthorization | None = None,
    resume: bool = False,
    capture_private_prose: bool = False,
    private_prose_ack: str | None = None,
) -> int:
    _validate_private_prose_capture_request(
        enabled=capture_private_prose,
        acknowledgement=private_prose_ack,
    )
    expected_metadata = _build_matrix_parent_metadata(
        spec, capture_private_prose=capture_private_prose
    )
    if validate_only:
        if resume:
            print("matrix validate-only cannot resume", file=sys.stderr)
            return 2
        print(
            f"matrix valid: {spec.experiment_id} — "
            f"{len(expected_metadata['arm_ids'])} arm(s), "
            f"worst case ${spec.worst_case_cost_usd:.2f}"
        )
        return 0
    if execution_authorization is None:
        print(
            "matrix execution requires fresh explicit runtime authorization",
            file=sys.stderr,
        )
        return 2
    try:
        _validate_matrix_execution_authorization(
            phrase=execution_authorization.phrase,
            authorized_at=execution_authorization.authorized_at.isoformat(),
        )
        with _matrix_output_ownership(out_dir, resume=resume):
            return _run_matrix_parent_owned(
                spec,
                out_dir,
                expected_metadata=expected_metadata,
                resume=resume,
            )
    except MatrixValidationError as exc:
        print(f"matrix execution refused: {exc}", file=sys.stderr)
        return 2
    except MatrixSpendCapExceeded as exc:
        print(f"matrix stopped: {exc}", file=sys.stderr)
        return 3


def _run_matrix_parent_owned(
    spec: MatrixSpec,
    out_dir: Path,
    *,
    expected_metadata: dict[str, Any],
    resume: bool,
) -> int:
    arms = {str(arm["id"]): arm for arm in spec.raw["arms"]}
    execution_schedule = expected_metadata["execution_schedule"]
    dispatch_plan = _matrix_dispatch_plan(execution_schedule)
    if resume:
        metadata, next_dispatch_index, _measured_total = (
            _verify_matrix_resume_boundary(spec, out_dir, expected_metadata)
        )
        metadata["resume_count"] = int(metadata.get("resume_count") or 0) + 1
        metadata["last_resumed_at"] = datetime.now(timezone.utc).isoformat()
        metadata["status"] = (
            "reporting"
            if next_dispatch_index == len(dispatch_plan)
            else "running"
        )
        worker_capability = _new_matrix_worker_capability(metadata)
        _write_json(out_dir / "experiment-metadata.json", metadata)
    else:
        metadata = expected_metadata
        metadata["status"] = "running"
        worker_capability = _new_matrix_worker_capability(metadata)
        _write_json(out_dir / "experiment-metadata.json", metadata)
        next_dispatch_index = 0

    authorized_cap = float(spec.raw["authorization"]["max_cost_usd"])
    for block_index, arm_id in dispatch_plan[next_dispatch_index:]:
        arm = arms[arm_id]
        arm_dir = out_dir / "arms" / arm_id
        try:
            measured_total = _completed_matrix_cost(out_dir)
            arm_cost = _measured_usage_cost(
                arm_dir / "token-cost-log.jsonl"
            )
        except MatrixSpendCapExceeded as exc:
            metadata["status"] = "spend_enforcement_failed"
            _write_json(out_dir / "experiment-metadata.json", metadata)
            print(f"matrix stopped: {exc}", file=sys.stderr)
            return 3
        if measured_total >= authorized_cap:
            metadata["status"] = "spend_stopped"
            _write_json(out_dir / "experiment-metadata.json", metadata)
            print(
                "matrix stopped: measured spend reached authorization cap",
                file=sys.stderr,
            )
            return 3

        # The worker re-reads this arm's cumulative JSONL. Pass only the cost
        # outside this arm so completed blocks are counted exactly once.
        if arm_cost > measured_total + 1e-9:
            metadata["status"] = "spend_enforcement_failed"
            _write_json(out_dir / "experiment-metadata.json", metadata)
            print(
                "matrix stopped: per-arm cost exceeds measured matrix total",
                file=sys.stderr,
            )
            return 3
        cost_excluding_arm = max(0.0, measured_total - arm_cost)
        dispatch = {
            "block_index": block_index,
            "arm_id": arm_id,
            "status": "running",
        }
        metadata["dispatch_log"].append(dispatch)
        _write_json(out_dir / "experiment-metadata.json", metadata)

        env = os.environ.copy()
        env.update({key: str(value) for key, value in arm["env"].items()})
        env["LANGFUSE_DISABLE"] = "1"
        env[_MATRIX_WORKER_CAPABILITY_ENV] = worker_capability
        worker_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--judgment-matrix",
            str(spec.path),
            "--matrix-worker",
            arm_id,
            "--matrix-worker-block",
            str(block_index),
            "--matrix-out",
            str(arm_dir),
            "--matrix-cost-offset",
            str(cost_excluding_arm),
        ]
        if expected_metadata["private_prose_capture"]["enabled"]:
            worker_command.append("--capture-private-prose")
        try:
            completed = subprocess.run(
                worker_command,
                cwd=REPO_ROOT,
                env=env,
            )
        except OSError as exc:
            dispatch["status"] = "failed"
            metadata["status"] = "failed"
            _write_json(out_dir / "experiment-metadata.json", metadata)
            print(
                f"matrix arm {arm_id} block {block_index} failed to start: "
                f"{type(exc).__name__}",
                file=sys.stderr,
            )
            return 1
        dispatch["status"] = (
            "complete" if completed.returncode == 0 else "failed"
        )
        if completed.returncode != 0:
            metadata["status"] = "failed"
            _write_json(out_dir / "experiment-metadata.json", metadata)
            print(
                f"matrix arm {arm_id} block {block_index} failed with exit "
                f"{completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode
        _write_json(out_dir / "experiment-metadata.json", metadata)
        try:
            measured_total = _completed_matrix_cost(out_dir)
        except MatrixSpendCapExceeded as exc:
            metadata["status"] = "spend_enforcement_failed"
            _write_json(out_dir / "experiment-metadata.json", metadata)
            print(f"matrix stopped: {exc}", file=sys.stderr)
            return 3
        if measured_total > authorized_cap:
            metadata["status"] = "spend_stopped"
            _write_json(out_dir / "experiment-metadata.json", metadata)
            print(
                "matrix stopped: measured spend exceeded authorization cap",
                file=sys.stderr,
            )
            return 3

    try:
        completed_cost = _completed_matrix_cost(out_dir)
    except MatrixSpendCapExceeded as exc:
        metadata["status"] = "spend_enforcement_failed"
        _write_json(out_dir / "experiment-metadata.json", metadata)
        print(f"matrix stopped: {exc}", file=sys.stderr)
        return 3
    if completed_cost > authorized_cap:
        metadata["status"] = "spend_stopped"
        _write_json(out_dir / "experiment-metadata.json", metadata)
        print(
            "matrix stopped: measured spend exceeded authorization cap",
            file=sys.stderr,
        )
        return 3
    metadata["measured_cost_usd"] = round(completed_cost, 6)
    metadata["status"] = "reporting"
    _write_json(out_dir / "experiment-metadata.json", metadata)
    report = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/glm_experiment_report.py"),
            str(out_dir),
        ],
        cwd=REPO_ROOT,
    )
    if report.returncode != 0:
        metadata["status"] = "report_failed"
        _write_json(out_dir / "experiment-metadata.json", metadata)
        print(
            "matrix failed closed: experiment report validation failed",
            file=sys.stderr,
        )
        return report.returncode
    metadata["status"] = "complete"
    _write_json(out_dir / "experiment-metadata.json", metadata)
    print(f"matrix artifacts: {out_dir}")
    return 0


@dataclass(frozen=True)
class _FacialReplayInput:
    line_number: int
    sample_id: str
    snippet: Any

    @property
    def profile_url(self) -> str:
        return str(self.snippet.profile_url)


def _read_matrix_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MatrixValidationError(
                    f"invalid worker JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise MatrixValidationError(
                    f"non-object worker JSONL at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def _validate_private_prose_prefix(
    arm_dir: Path,
    *,
    experiment_id: str,
    manifest_hash: str,
    arm_id: str,
    call_rows: list[dict[str, Any]],
    capture_private_prose: bool,
) -> None:
    """Validate the exact SAVE-rationale projection for a completed call prefix."""

    path = arm_dir / _PRIVATE_PROSE_FILENAME
    if not capture_private_prose:
        if path.exists() or path.is_symlink():
            raise MatrixValidationError(
                f"matrix arm {arm_id} has private prose while capture is disabled"
            )
        return

    expected: list[tuple[str, str, str]] = []
    for call in call_rows:
        if call.get("stage") != "full":
            continue
        call_id = str(call.get("call_id") or "")
        sample_ids = call.get("sample_ids")
        decisions = call.get("decisions")
        if (
            not call_id
            or not isinstance(sample_ids, list)
            or not isinstance(decisions, list)
            or len(sample_ids) != len(decisions)
        ):
            raise MatrixValidationError(
                f"matrix arm {arm_id} private prose source calls are malformed"
            )
        expected.extend(
            (call_id, str(sample_id), str(decision))
            for sample_id, decision in zip(sample_ids, decisions)
            if str(decision) in SAVE_DECISIONS
        )

    if not expected and not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        raise MatrixValidationError("private prose path must not be a symlink")
    if not path.is_file():
        raise MatrixValidationError(
            f"matrix arm {arm_id} private prose evidence is missing"
        )
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise MatrixValidationError("private prose file mode must be exactly 0600")

    rows = _read_matrix_jsonl(path)
    actual: list[tuple[str, str, str]] = []
    for row in rows:
        if set(row) != _PRIVATE_PROSE_ROW_KEYS:
            raise MatrixValidationError(
                f"matrix arm {arm_id} private prose row has unknown/missing fields"
            )
        row_without_hash = {
            key: value for key, value in row.items() if key != "row_hash"
        }
        if row.get("row_hash") != _canonical_sha256(row_without_hash):
            raise MatrixValidationError(
                f"matrix arm {arm_id} private prose row hash mismatch"
            )
        if (
            row.get("schema_version") != _PRIVATE_PROSE_SCHEMA_VERSION
            or row.get("experiment_id") != experiment_id
            or row.get("manifest_hash") != manifest_hash
            or row.get("arm_id") != arm_id
            or row.get("stage") != "full"
            or row.get("decision") not in SAVE_DECISIONS
            or not isinstance(row.get("rationale"), str)
            or not str(row["rationale"]).strip()
        ):
            raise MatrixValidationError(
                f"matrix arm {arm_id} private prose row identity/content mismatch"
            )
        actual.append(
            (
                str(row.get("call_id") or ""),
                str(row.get("sample_id") or ""),
                str(row.get("decision") or ""),
            )
        )
    if actual != expected or len(actual) != len(set(actual)):
        raise MatrixValidationError(
            f"matrix arm {arm_id} private prose coverage mismatch"
        )


def _build_matrix_arm_expected_metadata(
    spec: MatrixSpec,
    *,
    arm: dict[str, Any],
    arm_id: str,
    plans: tuple[dict[str, Any], ...],
    execution_units: list[str],
    cache_phases: dict[str, str],
    execution_schedule: list[dict[str, Any]],
    effective_env: dict[str, str],
    capture_private_prose: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "experiment_id": spec.experiment_id,
        "arm_id": arm_id,
        "manifest_hash": spec.manifest_hash,
        "git_sha": spec.git_sha,
        "source_hashes": spec.source_hashes,
        "random_seed": int(spec.raw["random_seed"]),
        "env": effective_env,
        "facial_mode": str(arm.get("facial_mode") or "pagewide"),
        "expected_call_ids": [plan["call_id"] for plan in plans],
        "execution_unit_order": execution_units,
        "cache_phase_by_execution_unit": cache_phases,
        "cache_warmup_execution_units_per_stage": int(
            spec.raw["dataset"]["cache_warmup_execution_units_per_stage"]
        ),
        "cache_block_design": "per-stage-warmup-prefix-v1",
        "execution_design": _MATRIX_EXECUTION_DESIGN,
        "execution_block_size": _MATRIX_INTERLEAVE_BLOCK_SIZE,
        "execution_schedule_sha256": _matrix_schedule_sha256(
            execution_schedule
        ),
        "expected_block_count": len(execution_schedule),
        "worker_process_model": _MATRIX_WORKER_PROCESS_MODEL,
        "cache_continuity": _MATRIX_CACHE_CONTINUITY,
        "private_prose_capture": {
            "enabled": bool(capture_private_prose),
            "schema_version": (
                _PRIVATE_PROSE_SCHEMA_VERSION if capture_private_prose else None
            ),
        },
        "offline_replay": True,
    }


def _validate_matrix_worker_prefix(
    arm_dir: Path,
    *,
    arm_id: str,
    plans: tuple[dict[str, Any], ...],
    completed_units: list[str],
    experiment_id: str,
    manifest_hash: str,
    capture_private_prose: bool = False,
) -> None:
    """Refuse to append a block onto missing, duplicated, or drifted evidence."""

    unit_rows = _read_matrix_jsonl(arm_dir / "execution-units.jsonl")
    unit_ids = [str(row.get("execution_unit_id") or "") for row in unit_rows]
    if unit_ids != completed_units:
        raise MatrixValidationError(
            f"matrix arm {arm_id} completed-unit prefix does not match its schedule"
        )
    expected_calls = {
        str(plan["call_id"])
        for plan in plans
        if str(plan["execution_unit_id"]) in set(completed_units)
    }
    call_rows = _read_matrix_jsonl(arm_dir / "calls.jsonl")
    call_ids = [str(row.get("call_id") or "") for row in call_rows]
    if (
        not all(call_ids)
        or len(call_ids) != len(set(call_ids))
        or set(call_ids) != expected_calls
    ):
        raise MatrixValidationError(
            f"matrix arm {arm_id} completed-call prefix does not match its schedule"
        )
    usage_path = arm_dir / "token-cost-log.jsonl"
    for call_id in expected_calls:
        _assert_exact_usage_receipt(usage_path, call_id)
    _validate_private_prose_prefix(
        arm_dir,
        experiment_id=experiment_id,
        manifest_hash=manifest_hash,
        arm_id=arm_id,
        call_rows=call_rows,
        capture_private_prose=capture_private_prose,
    )


def _validate_matrix_worker_receipts(
    arm_dir: Path,
    *,
    arm_id: str,
    arm: dict[str, Any],
    spec: MatrixSpec,
    plans: tuple[dict[str, Any], ...],
    completed_units: list[str],
) -> float:
    """Validate exact completed-call usage and attempt receipts for resume."""

    completed_unit_set = set(completed_units)
    expected_calls = {
        str(plan["call_id"])
        for plan in plans
        if str(plan["execution_unit_id"]) in completed_unit_set
    }
    usage_path = arm_dir / "token-cost-log.jsonl"
    usage = _read_matrix_jsonl(usage_path)
    attempts = _read_matrix_jsonl(arm_dir / "llm-attempts.jsonl")
    usage_ids = [
        str(row.get("logical_call_id") or row.get("call_id") or "")
        for row in usage
    ]
    if not all(usage_ids):
        raise MatrixValidationError(
            f"matrix arm {arm_id} has a usage receipt without a logical call id"
        )
    usage_counts = Counter(usage_ids)
    if any(usage_counts[call_id] != 1 for call_id in expected_calls):
        raise MatrixValidationError(
            f"matrix arm {arm_id} completed usage receipts are incomplete or duplicated"
        )
    fallback_ids: set[str] = set()
    for row, call_id in zip(usage, usage_ids):
        if call_id in expected_calls:
            continue
        parent_id = str(row.get("parent_logical_call_id") or "")
        if parent_id not in expected_calls or not str(
            row.get("fallback_reason") or ""
        ):
            raise MatrixValidationError(
                f"matrix arm {arm_id} has an unlinked or future usage receipt"
            )
        if call_id in fallback_ids:
            raise MatrixValidationError(
                f"matrix arm {arm_id} has duplicate fallback usage receipts"
            )
        fallback_ids.add(call_id)

    allowed_call_ids = expected_calls | fallback_ids
    attempt_ids = [
        str(row.get("logical_call_id") or row.get("call_id") or "")
        for row in attempts
    ]
    if not all(attempt_ids) or set(attempt_ids) != allowed_call_ids:
        raise MatrixValidationError(
            f"matrix arm {arm_id} attempt receipts do not match completed usage"
        )
    for call_id in allowed_call_ids:
        rows = [
            row
            for row, receipt_call_id in zip(attempts, attempt_ids)
            if receipt_call_id == call_id
        ]
        try:
            numbers = [int(row["attempt_number"]) for row in rows]
            max_attempts = {int(row["max_attempts"]) for row in rows}
        except (KeyError, TypeError, ValueError) as exc:
            raise MatrixValidationError(
                f"matrix arm {arm_id} has invalid attempt receipts"
            ) from exc
        statuses = [str(row.get("status") or "") for row in rows]
        if (
            numbers != list(range(1, len(rows) + 1))
            or len(max_attempts) != 1
            or len(rows) > next(iter(max_attempts))
            or not statuses
            or statuses[-1] not in {"response_received", "terminal_error"}
            or any(status != "retryable_error" for status in statuses[:-1])
        ):
            raise MatrixValidationError(
                f"matrix arm {arm_id} attempt receipts are not a complete sequence"
            )

    _assert_arm_usage_within_declared_limits(usage_path, spec=spec, arm=arm)
    return _measured_usage_cost(usage_path)


def _run_matrix_worker(
    spec: MatrixSpec,
    arm_id: str,
    arm_dir: Path,
    *,
    block_index: int,
    cost_offset_usd: float = 0.0,
    capture_private_prose: bool = False,
) -> int:
    from shared.brief_loader import load_brief
    from shared.judger import facial_judge_batch, full_judge, is_failure_decision
    from shared.llm_usage import llm_usage_session
    from shared.schemas import CandidateProfileSummary, CandidateSnippet
    from shared.storage import append_jsonl
    from linkedin.facial_batching import (
        FacialBatchFailureOutcome,
        run_facial_batches,
    )

    arm = next((item for item in spec.raw["arms"] if item["id"] == arm_id), None)
    if arm is None:
        raise MatrixValidationError(f"unknown matrix arm: {arm_id}")
    if block_index < 0:
        raise MatrixValidationError("matrix worker block cannot be negative")
    if not math.isfinite(cost_offset_usd) or cost_offset_usd < 0:
        raise MatrixValidationError("matrix cost offset must be finite and non-negative")
    plans = spec.call_plans[arm_id]
    warmup_units_per_stage = int(
        spec.raw["dataset"]["cache_warmup_execution_units_per_stage"]
    )
    execution_units, cache_phases = _cache_phased_execution_order(
        plans,
        random_seed=int(spec.raw["random_seed"]),
        warmup_units_per_stage=warmup_units_per_stage,
    )
    expected_env = {key: str(value) for key, value in arm["env"].items()}
    effective_env = {key: os.environ.get(key, "") for key in sorted(_MATRIX_ALLOWED_ENV)}
    if any(effective_env[key] != expected_env[key] for key in _MATRIX_REQUIRED_ENV):
        raise MatrixValidationError("worker environment does not match the arm manifest")

    _phased_orders, execution_schedule = _paired_matrix_execution_design(spec)
    if block_index >= len(execution_schedule):
        raise MatrixValidationError("matrix worker block is outside the pinned schedule")
    block_execution_units = list(
        execution_schedule[block_index]["execution_unit_ids"]
    )
    completed_units = [
        unit_id
        for block in execution_schedule[:block_index]
        for unit_id in block["execution_unit_ids"]
    ]
    metadata_path = arm_dir / "arm-metadata.json"
    expected_metadata = _build_matrix_arm_expected_metadata(
        spec,
        arm=arm,
        arm_id=arm_id,
        plans=plans,
        execution_units=execution_units,
        cache_phases=cache_phases,
        execution_schedule=execution_schedule,
        effective_env=effective_env,
        capture_private_prose=capture_private_prose,
    )
    if block_index == 0:
        if arm_dir.exists() and any(arm_dir.iterdir()):
            raise MatrixValidationError(
                f"matrix arm {arm_id} output is not empty before its first block"
            )
        arm_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            **expected_metadata,
            "completed_block_indices": [],
            "block_cost_offsets_usd": [],
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MatrixValidationError(
                f"matrix arm {arm_id} lacks valid metadata for block append"
            ) from exc
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in expected_metadata.items()
        ):
            raise MatrixValidationError(
                f"matrix arm {arm_id} metadata drifted before block append"
            )
        if metadata.get("status") != "running" or metadata.get(
            "completed_block_indices"
        ) != list(range(block_index)):
            raise MatrixValidationError(
                f"matrix arm {arm_id} blocks were not appended in order"
            )
        prior_offsets = metadata.get("block_cost_offsets_usd")
        if (
            not isinstance(prior_offsets, list)
            or len(prior_offsets) != block_index
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in prior_offsets
            )
        ):
            raise MatrixValidationError(
                f"matrix arm {arm_id} spend-offset prefix is invalid"
            )
        _validate_matrix_worker_prefix(
            arm_dir,
            arm_id=arm_id,
            plans=plans,
            completed_units=completed_units,
            experiment_id=spec.experiment_id,
            manifest_hash=spec.manifest_hash,
            capture_private_prose=capture_private_prose,
        )
    metadata["current_block_index"] = block_index
    metadata["block_cost_offsets_usd"].append(float(cost_offset_usd))
    _write_json(metadata_path, metadata)
    worker_seed = int.from_bytes(
        hashlib.sha256(
            f"{spec.raw['random_seed']}\0worker-block\0{block_index}".encode(
                "utf-8"
            )
        ).digest()[:8],
        "big",
    )
    random.seed(worker_seed)

    temporary_brief: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        ) as handle:
            json.dump(spec.brief_json, handle)
            temporary_brief = Path(handle.name)
        brief = load_brief(str(temporary_brief))
        if not brief.has_v2_schema:
            raise MatrixValidationError("snapshot brief is not V2; judgment matrix requires V2")

        facial_by_line = {
            number: _FacialReplayInput(
                number,
                _sample_id("facial", spec.file_hashes["snippets.jsonl"], number),
                CandidateSnippet.from_dict(record),
            )
            for number, record in spec.facial_rows
        }
        full_by_line = {
            number: CandidateProfileSummary.from_dict(record)
            for number, record in spec.full_rows
        }
        plan_by_unit_and_samples = {
            (plan["execution_unit_id"], tuple(plan["sample_ids"])): plan
            for plan in plans
        }
        calls_path = arm_dir / "calls.jsonl"
        units_path = arm_dir / "execution-units.jsonl"
        usage_path = arm_dir / "token-cost-log.jsonl"
        private_prose_path = arm_dir / _PRIVATE_PROSE_FILENAME
        calls_lock = threading.Lock()
        matrix_cache_lane_id = "matrix-" + hashlib.sha256(
            f"{spec.experiment_id}\0{arm_id}".encode("utf-8")
        ).hexdigest()[:24]

        def record_call(
            plan: dict[str, Any],
            decisions: list[str],
            elapsed_ms: float,
            status: str,
            prompt_hashes: dict[str, list[str]] | None = None,
            private_rationales: list[str] | None = None,
        ) -> None:
            row = {
                "experiment_id": spec.experiment_id,
                "arm_id": arm_id,
                "call_id": plan["call_id"],
                "stage": plan["stage"],
                "mode": plan["mode"],
                "sample_ids": plan["sample_ids"],
                "decisions": decisions,
                "elapsed_ms": round(elapsed_ms, 3),
                "actual_status": status,
                "offline_replay": True,
                "cache_phase": cache_phases[plan["execution_unit_id"]],
                **(prompt_hashes or {}),
            }
            with calls_lock:
                if capture_private_prose and plan["stage"] == "full":
                    if (
                        private_rationales is None
                        or len(private_rationales) != len(decisions)
                    ):
                        raise MatrixValidationError(
                            "full private prose capture lacks exact rationale cardinality"
                        )
                    for decision_word, rationale in zip(
                        decisions, private_rationales
                    ):
                        if decision_word in SAVE_DECISIONS and not str(rationale).strip():
                            raise MatrixValidationError(
                                "full SAVE private prose capture has an empty rationale"
                            )
                usage_receipt = _assert_exact_usage_receipt(
                    usage_path, plan["call_id"]
                )
                row.update(
                    {
                        "tool_schema_sha256": usage_receipt.get(
                            "tool_schema_sha256"
                        ),
                        "judgment_contract_version": usage_receipt.get(
                            "judgment_contract_version"
                        ),
                        "fallback_count": _fallback_receipt_count(
                            usage_path, plan["call_id"]
                        ),
                    }
                )
                append_jsonl(calls_path, row)
                if capture_private_prose and plan["stage"] == "full":
                    for sample_id, decision_word, rationale in zip(
                        plan["sample_ids"], decisions, private_rationales or []
                    ):
                        if decision_word not in SAVE_DECISIONS:
                            continue
                        private_row = {
                            "schema_version": _PRIVATE_PROSE_SCHEMA_VERSION,
                            "experiment_id": spec.experiment_id,
                            "manifest_hash": spec.manifest_hash,
                            "arm_id": arm_id,
                            "call_id": plan["call_id"],
                            "sample_id": sample_id,
                            "stage": "full",
                            "decision": decision_word,
                            "rationale": str(rationale).strip(),
                        }
                        _append_private_prose_row(
                            private_prose_path,
                            {
                                **private_row,
                                "row_hash": _canonical_sha256(private_row),
                            },
                        )
                measured = float(cost_offset_usd) + _measured_usage_cost(usage_path)
                _assert_arm_usage_within_declared_limits(
                    usage_path,
                    spec=spec,
                    arm=arm,
                )
                if measured > float(spec.raw["authorization"]["max_cost_usd"]):
                    raise MatrixSpendCapExceeded(
                        "measured matrix spend exceeded authorization cap"
                    )

        def facial_call(inputs: list[_FacialReplayInput], context: dict[str, Any]):
            sample_ids = tuple(item.sample_id for item in inputs)
            execution_unit_id = str(context.get("execution_unit_id") or "")
            try:
                plan = plan_by_unit_and_samples[(execution_unit_id, sample_ids)]
            except KeyError as exc:
                raise MatrixValidationError(
                    "facial execution unit/sample attribution does not match the pinned plan"
                ) from exc
            usage_context = {
                **context,
                "stage": "facial",
                "experiment_id": spec.experiment_id,
                "arm_id": arm_id,
                "logical_call_id": plan["call_id"],
                "offline_replay": True,
                "lane_id": matrix_cache_lane_id,
            }
            kwargs: dict[str, Any] = {}
            if "lane_context" in inspect.signature(facial_judge_batch).parameters:
                kwargs["lane_context"] = usage_context
            if "opaque_candidate_ids" in inspect.signature(
                facial_judge_batch
            ).parameters:
                kwargs["opaque_candidate_ids"] = _replay_candidate_ids(
                    plan["call_id"], len(inputs)
                )
            started = time.monotonic()
            try:
                decisions = facial_judge_batch(
                    [item.snippet for item in inputs], brief, **kwargs
                )
            except BaseException:
                record_call(plan, [], (time.monotonic() - started) * 1000, "error")
                raise
            if len(decisions) != len(inputs) or [
                decision.profile_url for decision in decisions
            ] != [item.profile_url for item in inputs]:
                record_call(
                    plan,
                    [],
                    (time.monotonic() - started) * 1000,
                    "postcondition_fail",
                )
                raise MatrixValidationError(
                    f"facial call {plan['call_id']} failed cardinality/identity validation"
                )
            words = [decision.decision for decision in decisions]
            status = "postcondition_fail" if any(is_failure_decision(word) for word in words) else "ok"
            record_call(
                plan,
                words,
                (time.monotonic() - started) * 1000,
                status,
                _decision_prompt_hashes(decisions),
            )
            if status == "postcondition_fail":
                raise MatrixValidationError(
                    f"facial call {plan['call_id']} returned an unrecovered "
                    "postcondition failure"
                )
            return decisions

        def record_unit(
            unit_id: str,
            unit_plans: list[dict[str, Any]],
            elapsed_ms: float,
            status: str,
        ) -> None:
            stage = (
                "facial_page"
                if unit_plans[0]["stage"] == "facial"
                else "full_profile"
            )
            append_jsonl(
                units_path,
                {
                    "experiment_id": spec.experiment_id,
                    "arm_id": arm_id,
                    "execution_unit_id": unit_id,
                    "stage": stage,
                    "mode": unit_plans[0]["mode"],
                    "call_ids": [plan["call_id"] for plan in unit_plans],
                    "sample_count": sum(
                        len(plan["sample_ids"]) for plan in unit_plans
                    ),
                    "elapsed_ms": round(elapsed_ms, 3),
                    "actual_status": status,
                    "offline_replay": True,
                    "cache_phase": cache_phases[unit_id],
                },
            )

        def full_call(plan: dict[str, Any]) -> None:
            line_number = plan["line_numbers"][0]
            context = {
                "stage": "full_eval",
                "experiment_id": spec.experiment_id,
                "arm_id": arm_id,
                "logical_call_id": plan["call_id"],
                "offline_replay": True,
                "execution_unit_id": plan["execution_unit_id"],
                "cache_phase": cache_phases[plan["execution_unit_id"]],
                "lane_id": matrix_cache_lane_id,
            }
            started = time.monotonic()
            try:
                kwargs: dict[str, Any] = {"lane_context": context}
                if "opaque_candidate_id" in inspect.signature(full_judge).parameters:
                    kwargs["opaque_candidate_id"] = _replay_candidate_ids(
                        plan["call_id"], 1
                    )[0]
                decision = full_judge(full_by_line[line_number], brief, **kwargs)
            except BaseException:
                record_call(
                    plan,
                    [],
                    (time.monotonic() - started) * 1000,
                    "error",
                    private_rationales=[],
                )
                raise
            status = "postcondition_fail" if is_failure_decision(decision.decision) else "ok"
            record_call(
                plan,
                [decision.decision],
                (time.monotonic() - started) * 1000,
                status,
                _decision_prompt_hashes([decision]),
                [decision.rationale],
            )
            if status == "postcondition_fail":
                raise MatrixValidationError(
                    f"full call {plan['call_id']} returned an unrecovered "
                    "postcondition failure"
                )

        with llm_usage_session(
            usage_path,
            experiment_id=spec.experiment_id,
            arm_id=arm_id,
            offline_replay=True,
        ):
            plans_by_unit = {
                unit_id: [plan for plan in plans if plan["execution_unit_id"] == unit_id]
                for unit_id in block_execution_units
            }
            for unit_id in block_execution_units:
                unit_plans = plans_by_unit[unit_id]
                unit_started = time.monotonic()
                try:
                    if unit_plans[0]["stage"] == "full":
                        full_call(unit_plans[0])
                    else:
                        page_index = unit_plans[0]["page_index"]
                        if arm.get("facial_mode", "pagewide") == "partitioned_concurrent":
                            page_inputs = [
                                facial_by_line[number]
                                for plan in unit_plans
                                for number in plan["line_numbers"]
                            ]
                            facial_outcomes = asyncio.run(
                                run_facial_batches(
                                    page_inputs,
                                    facial_call,
                                    max_concurrency=int(arm["env"]["LINKEDIN_FACIAL_MAX_CONCURRENCY"]),
                                    target_batch_size=int(arm["env"]["LINKEDIN_FACIAL_TARGET_BATCH_SIZE"]),
                                    base_context={
                                        "page_index": page_index,
                                        "execution_unit_id": unit_id,
                                        "cache_phase": cache_phases[unit_id],
                                    },
                                    input_identity=lambda item: item.profile_url,
                                    result_identity=lambda result: result.profile_url,
                                )
                            )
                            failure_outcome = next(
                                (
                                    outcome
                                    for outcome in facial_outcomes
                                    if isinstance(outcome, FacialBatchFailureOutcome)
                                ),
                                None,
                            )
                            if failure_outcome is not None:
                                raise failure_outcome.error
                        else:
                            for plan in unit_plans:
                                inputs = [
                                    facial_by_line[number]
                                    for number in plan["line_numbers"]
                                ]
                                facial_call(
                                    inputs,
                                    {
                                        "page_index": plan["page_index"],
                                        "batch_index": plan["batch_index"],
                                        "execution_unit_id": unit_id,
                                        "cache_phase": cache_phases[unit_id],
                                    },
                                )
                except BaseException:
                    record_unit(
                        unit_id,
                        unit_plans,
                        (time.monotonic() - unit_started) * 1000,
                        "error",
                    )
                    raise
                record_unit(
                    unit_id,
                    unit_plans,
                    (time.monotonic() - unit_started) * 1000,
                    "ok",
                )

        metadata["completed_block_indices"].append(block_index)
        metadata.pop("current_block_index", None)
        _validate_private_prose_prefix(
            arm_dir,
            experiment_id=spec.experiment_id,
            manifest_hash=spec.manifest_hash,
            arm_id=arm_id,
            call_rows=_read_matrix_jsonl(calls_path),
            capture_private_prose=capture_private_prose,
        )
        if block_index == len(execution_schedule) - 1:
            call_rows = _read_matrix_jsonl(calls_path)
            metadata["prompt_hash_call_count"] = len(call_rows)
            metadata["prompt_hash_manifest_sha256"] = _prompt_hash_manifest_sha256(
                call_rows
            )
            metadata["deterministic_replay_ids"] = "call-id-position-sha256-v1"
            metadata["status"] = "complete"
            metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        else:
            metadata["status"] = "running"
            metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(metadata_path, metadata)
        return 0
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error_type"] = type(exc).__name__
        metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(metadata_path, metadata)
        print(f"matrix arm {arm_id} failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        if temporary_brief is not None:
            temporary_brief.unlink(missing_ok=True)


def _load_shadow_report_module():
    spec = importlib.util.spec_from_file_location(
        "shadow_report", Path(__file__).resolve().parent / "shadow_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _drain_with_progress(label: str, timeout: float) -> bool:
    """Drain the strategy-shadow queue, printing progress while it thinks.

    Fable formation ran 113-173s on the 2026-07-05 live run; silence that
    long reads as a hang, so poll in short waits.
    """
    from shared.strategy_shadow import drain_strategy_shadows

    start = time.monotonic()
    while True:
        if drain_strategy_shadows(timeout=15.0):
            elapsed = time.monotonic() - start
            print(f"  [{label}] shadow drained after {elapsed:.0f}s")
            return True
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            print(
                f"  [{label}] TIMEOUT after {elapsed:.0f}s — shadow still "
                "in flight; artifact will be missing"
            )
            return False
        print(f"  [{label}] shadow still thinking… {elapsed:.0f}s", flush=True)


def _primary_call(system_prompt: str, user_prompt: str, *, stage: str, expect_json: bool):
    """One primary-tier call through the production client. Fail-soft:
    the shadow comparison is the point of the replay, so a primary failure
    is reported, not fatal."""
    from shared.llm_clients import opus_llm

    print(f"  [{stage}] primary call → {config.STRATEGY_MODEL_NAME} …", flush=True)
    start = time.monotonic()
    try:
        result = opus_llm(
            system_prompt,
            user_prompt,
            expect_json=expect_json,
            max_tokens=32768,
            usage_context={"stage": f"shadow_replay_{stage}"},
            model_name=config.STRATEGY_MODEL_NAME,
        )
        print(f"  [{stage}] primary done in {time.monotonic() - start:.0f}s")
        return result, None
    except Exception as exc:  # noqa: BLE001 — replay is observability, not gating
        print(f"  [{stage}] PRIMARY FAILED: {exc}")
        return None, str(exc)[:500]


def run_preflight(seed_brief_path: Path, out_dir: Path, *, skip_primary: bool, drain_timeout: float) -> Path | None:
    """Replay the preflight pair. Returns the generated v2 brief path (or
    None when the primary was skipped/failed so no brief exists)."""
    from shared.brief_loader import load_brief
    from shared.preflight_v2 import (
        format_confidence_notes,
        generate_preflight_prompt,
        parse_preflight_response,
        preflight_to_brief_json,
    )
    from shared.brief_lint import format_findings, lint_generated_brief
    from shared.storage import write_json
    from shared.strategy_shadow import dispatch_strategy_shadow, plan_metrics

    seed = load_brief(str(seed_brief_path))
    jd_text = seed.jd_text
    geography = (seed.permanent_filters or {}).get("Location", "")
    raw_instructions = getattr(seed, "instructions", None)
    if not isinstance(raw_instructions, (list, tuple)):
        raw_instructions = []
    operator_guidance = "\n".join(
        f"- {str(item).strip()}" for item in raw_instructions if str(item).strip()
    )
    prompt = generate_preflight_prompt(
        jd_text,
        geography or None,
        operator_guidance=operator_guidance or None,
    )

    raw_response, primary_error = (None, "skipped (--skip-primary)")
    if not skip_primary:
        raw_response, primary_error = _primary_call(
            PREFLIGHT_SYSTEM, prompt, stage="preflight", expect_json=False
        )

    primary_meta = {
        "primary_model": config.STRATEGY_MODEL_NAME,
        "metrics": (
            plan_metrics(
                raw_response,
                reference_text=PREFLIGHT_SYSTEM + "\n" + prompt,
                novelty_reference="system+user",
            )
            if raw_response is not None
            else None
        ),
        "raw_response": raw_response,
    }
    if primary_error:
        primary_meta["primary_error"] = primary_error

    dispatch_strategy_shadow(
        stage="linkedin_preflight_v2",
        system_prompt=PREFLIGHT_SYSTEM,
        user_prompt=prompt,
        max_tokens=32768,
        shadow_dir=out_dir / "shadow_strategy",
        primary_meta=primary_meta,
    )
    _drain_with_progress("preflight", drain_timeout)

    if raw_response is None:
        return None

    # Convert the primary's response to a v2 brief exactly the way the
    # orchestrator does (linkedin/orchestrator.py::_run_preflight_v2),
    # so the later stages can run on it. Lint findings print but never
    # block — the replay observes, it does not gate.
    try:
        data = parse_preflight_response(raw_response)
    except Exception as exc:  # noqa: BLE001
        print(f"  [preflight] primary response did not parse: {exc}")
        return None
    findings = lint_generated_brief(
        data,
        jd_text=jd_text,
        operator_instructions=raw_instructions,
        seed_blacklist=list(seed.employer_blacklist or []),
    )
    if findings:
        print(format_findings(findings))
    overrides = {}
    if seed.linkedin_project:
        overrides["linkedin_project"] = seed.linkedin_project
    if geography:
        overrides["geography"] = geography
    if seed.kit_url:
        overrides["kit_url"] = seed.kit_url
    if seed.employer_blacklist:
        overrides["employer_blacklist"] = seed.employer_blacklist
    brief_json = preflight_to_brief_json(data, overrides)
    brief_json["provenance"] = {
        "generated_by": "shadow_replay_preflight_v2",
        "reviewed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    generated_path = out_dir / "preflight_v2_brief.json"
    write_json(str(generated_path), brief_json)
    print(f"  [preflight] generated v2 brief → {generated_path}")
    notes = format_confidence_notes(data)
    if notes:
        print(notes)
    return generated_path


def run_formation(v2_brief_path: Path, out_dir: Path, *, skip_primary: bool, drain_timeout: float) -> None:
    """Replay the formation pair on the REAL strategy prompts."""
    from shared.brief_loader import load_brief
    from linkedin.strategy import (
        _build_strategy_system,
        _build_strategy_user,
        _explicit_design_from_brief,
    )
    from shared.strategy_shadow import dispatch_strategy_shadow, plan_metrics

    brief = load_brief(str(v2_brief_path))
    explicit_design = _explicit_design_from_brief(brief)
    use_layered_retrieval = explicit_design.is_explicit()
    # has_kit=False / no kit strings: the replay has no kit fetch, which is
    # also the documented production "JD context only" path.
    system = _build_strategy_system(
        brief, has_kit=False, use_layered_retrieval=use_layered_retrieval
    )
    user_prompt = _build_strategy_user(
        brief, [], None, use_layered_retrieval=use_layered_retrieval, lane_feedback=None
    )
    print(
        f"  [formation] prompts built from {v2_brief_path.name}: "
        f"system={len(system)} chars, user={len(user_prompt)} chars, "
        f"layered={use_layered_retrieval}"
    )

    result, primary_error = (None, "skipped (--skip-primary)")
    if not skip_primary:
        result, primary_error = _primary_call(
            system, user_prompt, stage="formation", expect_json=True
        )

    primary_meta = {
        "primary_model": config.STRATEGY_MODEL_NAME,
        "metrics": (
            plan_metrics(
                result,
                reference_text=system + "\n" + user_prompt,
                novelty_reference="system+user",
            ) if result is not None else None
        ),
        "raw_response": result,
    }
    if primary_error:
        primary_meta["primary_error"] = primary_error

    # Mirrors linkedin/strategy.py::form_strategy's dispatch contract: the
    # formation shadow runs under the FRESH-context contract (2026-07-05).
    # The replay builds its prompts with prior_run_data=None already, so
    # primary and shadow prompts coincide here — the stamp records the
    # contract, matching the production artifact shape.
    dispatch_strategy_shadow(
        stage="linkedin_strategy_form",
        system_prompt=system,
        user_prompt=user_prompt,
        max_tokens=32768,
        shadow_dir=out_dir / "shadow_strategy",
        primary_meta=primary_meta,
        shadow_prompt_context="fresh",
        primary_prompt_included_prior_run_data=False,
    )
    _drain_with_progress("formation", drain_timeout)


def run_judge(v2_brief_path: Path, profiles_path: Path, profile_index: int, out_dir: Path) -> None:
    """Replay ONE GLM full-eval judgment on a real captured profile,
    through the production judger seam, into the production capture file."""
    from shared.brief_loader import load_brief
    from shared.judger import (
        _full_shadow_call,
        _profile_to_text,
        _record_shadow_judgment,
        is_failure_decision,
    )
    from shared.schemas import CandidateProfileSummary
    from linkedin.judgment_templates import (
        assemble_full_evaluation_system,
        parse_full_evaluation_response,
    )

    brief = load_brief(str(v2_brief_path))
    lines = [
        line for line in profiles_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        print(f"  [judge] no profiles in {profiles_path}")
        return
    index = max(0, min(profile_index, len(lines) - 1))
    summary = CandidateProfileSummary.from_dict(json.loads(lines[index]))
    system = assemble_full_evaluation_system(brief._new_brief)
    profile_text = _profile_to_text(summary)
    print(
        f"  [judge] GLM full-eval on profile #{index} ({summary.name}) → "
        f"{config.SHADOW_FACIAL_MODEL_NAME} …",
        flush=True,
    )

    capture: dict = {}
    raw, latency_ms, error = _full_shadow_call(
        system_prompt=system,
        user_prompt=profile_text,
        max_tokens=8192,
        usage_context={"stage": "shadow_replay_judge"},
        capture=capture,
    )
    shadow_decision = None
    shadow_parse_failed = False
    if raw is not None:
        try:
            shadow_decision = parse_full_evaluation_response(raw).decision
        except Exception as parse_exc:  # noqa: BLE001
            print(f"  [judge] shadow parse failed: {parse_exc}")
            shadow_parse_failed = True
        else:
            shadow_parse_failed = is_failure_decision(shadow_decision)
    print(
        f"  [judge] done in {(latency_ms or 0) / 1000:.0f}s — "
        f"decision={shadow_decision or '—'} error={error or '—'}"
    )
    # Same payload shape as shared/judger.py::_run_full_shadow_single_sync;
    # primary_decision is None because the replay deliberately makes no
    # primary full-eval call (the GLM side is what needs verifying).
    _record_shadow_judgment(
        str(out_dir / "run_log.jsonl"),
        {
            "ts": time.time(),
            "stage": "full",
            "shadow_model": config.SHADOW_FACIAL_MODEL_NAME,
            "primary_decision": None,
            "shadow_decision": shadow_decision,
            "agrees": None,
            "shadow_parse_failed": shadow_parse_failed,
            "latency_ms": latency_ms,
            "shadow_error": error,
            "lane_context": {"stage": "shadow_replay_judge", "replay": True},
            "raw": raw,
            "reasoning_content": capture.get("reasoning_content"),
            "finish_reason": capture.get("finish_reason"),
            "user_prompt": profile_text,
        },
    )


_PAGE_ALLOCATOR_EVENTS = frozenset(
    {
        "page_allocator_shadow_checkpoint",
        "page_allocator_shadow_exhaustion",
        "page_allocator_shadow_poison",
    }
)


def _read_page_allocator_log(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return rows, {"line": line_number, "reason": "malformed_json"}
                if not isinstance(row, dict):
                    return rows, {
                        "line": line_number,
                        "reason": "non_object_jsonl_row",
                    }
                rows.append(row)
    except OSError as exc:
        return rows, {"line": None, "reason": f"log_unreadable: {exc}"}
    return rows, None


def summarize_page_allocator_replay(
    events: list[dict[str, Any]],
    *,
    source_issue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report only valid, on-policy page currency before first divergence."""

    valid_pages: list[dict[str, int]] = []
    currency: list[dict[str, int]] = []
    retained: list[dict[str, int]] = []
    avoided: list[dict[str, int]] = []
    observed_roots: set[int] = set()
    all_observed_roots: set[int] = set()
    recommended_roots: set[int] = set()
    first_divergence: dict[str, int] | None = None
    poison = source_issue
    previous_target: int | None = None
    previous_sequence: int | None = None
    invalid_excluded = 0
    off_policy_excluded = 0
    ignored_on_policy_labels = 0
    allocator_events_seen = 0

    for line_number, event in enumerate(events, 1):
        event_name = event.get("event")
        if event_name == "page_observation_gap":
            poison = {"line": line_number, "reason": "page_observation_gap"}
            break
        if event_name not in _PAGE_ALLOCATOR_EVENTS:
            continue
        allocator_events_seen += 1

        if event_name == "page_allocator_shadow_poison":
            root_id = event.get("root_string_id")
            page = event.get("page")
            reason = event.get("poison_reason")
            analysis_evaluable = event.get("analysis_evaluable")
            if (
                isinstance(root_id, bool)
                or not isinstance(root_id, int)
                or root_id <= 0
                or isinstance(page, bool)
                or not isinstance(page, int)
                or page <= 0
                or not isinstance(reason, str)
                or not reason
                or analysis_evaluable is not False
            ):
                poison = {
                    "line": line_number,
                    "reason": "malformed_allocator_poison",
                }
            else:
                poison = {
                    "line": line_number,
                    "reason": f"allocator_shadow_poison:{reason}",
                    "root_string_id": root_id,
                    "page": page,
                }
            break

        raw_sequence = event.get(
            "checkpoint_index", event.get("verdict_sequence")
        )
        expected_sequence = (
            1 if previous_sequence is None else previous_sequence + 1
        )
        if (
            isinstance(raw_sequence, bool)
            or not isinstance(raw_sequence, int)
            or raw_sequence != expected_sequence
        ):
            poison = {"line": line_number, "reason": "checkpoint_sequence_gap"}
            break
        previous_sequence = raw_sequence

        verdict = event.get("verdict")
        if not isinstance(verdict, dict):
            poison = {"line": line_number, "reason": "malformed_verdict"}
            break
        event_root_id = event.get("root_string_id")
        event_page = event.get("page")
        current_root_id = verdict.get("current_root_id")
        if (
            isinstance(event_root_id, bool)
            or not isinstance(event_root_id, int)
            or event_root_id <= 0
            or isinstance(event_page, bool)
            or not isinstance(event_page, int)
            or event_page <= 0
            or isinstance(current_root_id, bool)
            or not isinstance(current_root_id, int)
            or current_root_id <= 0
            or current_root_id != event_root_id
        ):
            poison = {"line": line_number, "reason": "checkpoint_identity_mismatch"}
            break
        selected = verdict.get("selected_root_id")
        if selected is not None and (
            isinstance(selected, bool)
            or not isinstance(selected, int)
            or selected <= 0
        ):
            poison = {"line": line_number, "reason": "malformed_selected_root"}
            break
        if selected is not None and first_divergence is None:
            recommended_roots.add(selected)

        if event_name == "page_allocator_shadow_exhaustion":
            root_id = current_root_id
            page = event_page
            explicit_divergence = event.get(
                "shadow_diverged", event.get("diverged", False)
            )
            if (
                isinstance(root_id, bool)
                or not isinstance(root_id, int)
                or root_id <= 0
                or isinstance(page, bool)
                or not isinstance(page, int)
                or page <= 0
                or not isinstance(explicit_divergence, bool)
            ):
                poison = {
                    "line": line_number,
                    "reason": "malformed_exhaustion_event",
                }
                break
            if (
                first_divergence is None
                and (
                    explicit_divergence
                    or (
                        previous_target is not None
                        and previous_target != root_id
                    )
                )
            ):
                first_divergence = {
                    "root_string_id": root_id,
                    "page": page,
                }
            previous_target = selected
            continue

        observation = event.get("observation")
        if not isinstance(observation, dict):
            poison = {"line": line_number, "reason": "malformed_observation"}
            break
        try:
            root_id = observation["root_string_id"]
            variant_id = observation["variant_id"]
            page = observation["page"]
            slots = observation["slots"]
            valid = observation["valid"]
            invalid_reasons = observation["invalid_reasons"]
            off_policy = observation["off_policy"]
            extracted = observation["extracted"]
            full_expected = observation["full_expected"]
            full_settled = observation["full_settled"]
            priority = observation["priority"]
            standard = observation["standard"]
            outreach = observation["outreach"]
            break_reason = observation["break_reason"]
            technical_interruption = observation["technical_interruption"]
        except KeyError as exc:
            poison = {
                "line": line_number,
                "reason": f"missing_observation_field: {exc.args[0]}",
            }
            break
        counters = (
            root_id,
            page,
            slots,
            extracted,
            full_expected,
            full_settled,
            priority,
            standard,
            outreach,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in counters
            )
            or root_id <= 0
            or page <= 0
            or not isinstance(variant_id, str)
            or not variant_id
            or not isinstance(break_reason, str)
            or not isinstance(technical_interruption, bool)
            or not isinstance(valid, bool)
            or not isinstance(off_policy, bool)
            or not isinstance(invalid_reasons, list)
            or any(not isinstance(reason, str) for reason in invalid_reasons)
        ):
            poison = {"line": line_number, "reason": "malformed_observation"}
            break
        reconstructed = PageObservation.from_dict(observation)
        if (
            reconstructed.valid != valid
            or list(reconstructed.invalid_reasons) != invalid_reasons
        ):
            poison = {
                "line": line_number,
                "reason": "observation_validity_mismatch",
            }
            break
        if root_id != event_root_id or page != event_page:
            poison = {
                "line": line_number,
                "reason": "checkpoint_identity_mismatch",
            }
            break

        page_ref = {"root_string_id": root_id, "page": page}
        all_observed_roots.add(root_id)
        explicit_divergence = event.get(
            "shadow_diverged", event.get("diverged", False)
        )
        divergence_after_observation = event.get(
            "divergence_after_observation", False
        )
        if not isinstance(explicit_divergence, bool):
            poison = {"line": line_number, "reason": "malformed_divergence_flag"}
            break
        if not isinstance(divergence_after_observation, bool):
            poison = {
                "line": line_number,
                "reason": "malformed_divergence_boundary",
            }
            break
        diverged_now = (
            off_policy
            or (explicit_divergence and not divergence_after_observation)
            or (previous_target is not None and previous_target != root_id)
        )
        if first_divergence is None and diverged_now:
            first_divergence = page_ref
            avoided.append(page_ref)
        if first_divergence is not None:
            off_policy_excluded += 1
            if not off_policy:
                ignored_on_policy_labels += 1
            previous_target = selected
            continue
        if not valid:
            invalid_excluded += 1
            previous_target = selected
            if divergence_after_observation and first_divergence is None:
                first_divergence = page_ref
            continue

        valid_pages.append(page_ref)
        observed_roots.add(root_id)
        currency.append(
            {
                **page_ref,
                "n": extracted,
                "p": priority,
                "e": priority + standard,
            }
        )
        if previous_target == root_id:
            retained.append(page_ref)
        previous_target = selected
        if divergence_after_observation and first_divergence is None:
            # The page itself was collected under the prior aligned frontier;
            # a legacy disposition diverged only after the observation settled.
            first_divergence = page_ref

    return {
        "evaluable": poison is None,
        "poison": poison,
        "allocator_events_seen": allocator_events_seen,
        "valid_on_policy_pages": valid_pages,
        "valid_on_policy_currency": currency,
        "currency_totals": {
            "n": sum(row["n"] for row in currency),
            "p": sum(row["p"] for row in currency),
            "e": sum(row["e"] for row in currency),
        },
        "observed_pages_retained": retained,
        "observed_pages_avoided": avoided,
        "first_divergence": first_divergence,
        "invalid_pages_excluded": invalid_excluded,
        "off_policy_pages_excluded": off_policy_excluded,
        "post_divergence_on_policy_labels_ignored": ignored_on_policy_labels,
        "observed_root_string_ids": sorted(observed_roots),
        "recommended_but_unobserved_root_string_ids": sorted(
            recommended_roots - all_observed_roots
        ),
        "claims": {
            "unobserved_sibling_yield": False,
            "fixed_uplift_percentage": False,
        },
    }


def page_allocator_replay_report(path: Path) -> dict[str, Any]:
    events, issue = _read_page_allocator_log(path)
    return summarize_page_allocator_replay(events, source_issue=issue)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stages",
        default="formation,judge",
        help="comma list from {preflight,formation,judge}; always executed in that order",
    )
    parser.add_argument("--seed-brief", type=Path, default=DEFAULT_SEED_BRIEF)
    parser.add_argument("--v2-brief", type=Path, default=None,
                        help=f"v2 brief for formation/judge (default: {DEFAULT_V2_BRIEF})")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--profile-index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="artifact dir (default: output/state/shadow_replay/<UTC ts>)")
    parser.add_argument("--skip-primary", action="store_true",
                        help="skip primary-tier calls (cheaper; shadow-only artifacts)")
    parser.add_argument("--drain-timeout", type=float, default=900.0)
    parser.add_argument(
        "--shadow-temperature",
        type=float,
        default=None,
        help="process-local GLM shadow temperature override (default: shared client setting)",
    )
    parser.add_argument(
        "--page-allocator-log",
        type=Path,
        default=None,
        help="summarize allocator events from a run-log JSONL without provider calls",
    )
    parser.add_argument(
        "--judgment-matrix",
        type=Path,
        default=None,
        help=(
            "run the manifest-driven primary judgment matrix instead of the "
            "legacy shadow replay"
        ),
    )
    parser.add_argument(
        "--matrix-validate-only",
        action="store_true",
        help="validate hashes, authorization, call plan, and worst-case spend without API calls",
    )
    parser.add_argument(
        "--matrix-execute",
        action="store_true",
        help="explicitly enable paid judgment-matrix execution",
    )
    parser.add_argument(
        "--matrix-resume",
        action="store_true",
        help="resume only from a verified completed arm/block boundary",
    )
    parser.add_argument(
        "--matrix-authorization-phrase",
        default=None,
        help=(
            "fresh paid-run confirmation; must exactly equal: "
            f"{_MATRIX_EXECUTION_AUTHORIZATION_PHRASE!r}"
        ),
    )
    parser.add_argument(
        "--matrix-authorization-at",
        default=None,
        help="timezone-aware ISO-8601 confirmation time, no more than 15 minutes old",
    )
    parser.add_argument(
        "--capture-private-prose",
        action="store_true",
        help=(
            "capture full SAVE rationales in a separate local 0600 PII artifact"
        ),
    )
    parser.add_argument(
        "--ack-private-prose-pii",
        default=None,
        metavar="ACK",
        help=f"must equal {_PRIVATE_PROSE_PII_ACK}",
    )
    parser.add_argument("--matrix-worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--matrix-worker-block", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--matrix-out", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--matrix-cost-offset", type=float, default=0.0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.page_allocator_log is not None:
        if args.judgment_matrix is not None:
            parser.error("--page-allocator-log cannot be combined with --judgment-matrix")
        report = page_allocator_replay_report(args.page_allocator_log)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["evaluable"] else 2

    if args.judgment_matrix is not None:
        try:
            spec = _load_and_validate_matrix_manifest(args.judgment_matrix)
            if args.matrix_worker:
                if (
                    args.matrix_validate_only
                    or args.matrix_execute
                    or args.matrix_resume
                    or args.matrix_authorization_phrase is not None
                    or args.matrix_authorization_at is not None
                    or args.ack_private_prose_pii is not None
                    or args.matrix_out is None
                    or args.matrix_worker_block is None
                ):
                    raise MatrixValidationError(
                        "matrix worker requires --matrix-out and "
                        "--matrix-worker-block and cannot validate-only"
                    )
                resolved_worker_out = args.matrix_out.resolve()
                if resolved_worker_out.name != args.matrix_worker:
                    raise MatrixValidationError(
                        "matrix worker output directory must match its arm id"
                    )
                _validate_internal_matrix_worker(resolved_worker_out)
                return _run_matrix_worker(
                    spec,
                    args.matrix_worker,
                    resolved_worker_out,
                    block_index=args.matrix_worker_block,
                    cost_offset_usd=args.matrix_cost_offset,
                    capture_private_prose=args.capture_private_prose,
                )
            if (
                args.matrix_worker_block is not None
                or args.matrix_out is not None
                or args.matrix_cost_offset != 0.0
            ):
                raise MatrixValidationError(
                    "matrix worker flags are internal and cannot be used by a parent"
                )
            out_dir = _matrix_output_dir(spec, args.out)
            _validate_private_prose_capture_request(
                enabled=args.capture_private_prose,
                acknowledgement=args.ack_private_prose_pii,
            )
            if args.matrix_validate_only:
                if (
                    args.matrix_execute
                    or args.matrix_resume
                    or args.matrix_authorization_phrase is not None
                    or args.matrix_authorization_at is not None
                ):
                    raise MatrixValidationError(
                        "matrix validate-only cannot be combined with execution flags"
                    )
                return _run_matrix_parent(
                    spec,
                    out_dir,
                    validate_only=True,
                    capture_private_prose=args.capture_private_prose,
                    private_prose_ack=args.ack_private_prose_pii,
                )
            if not args.matrix_execute:
                raise MatrixValidationError(
                    "paid matrix parent requires explicit --matrix-execute"
                )
            execution_authorization = _validate_matrix_execution_authorization(
                phrase=args.matrix_authorization_phrase,
                authorized_at=args.matrix_authorization_at,
            )
            return _run_matrix_parent(
                spec,
                out_dir,
                validate_only=False,
                execution_authorization=execution_authorization,
                resume=args.matrix_resume,
                capture_private_prose=args.capture_private_prose,
                private_prose_ack=args.ack_private_prose_pii,
            )
        except MatrixValidationError as exc:
            print(f"matrix validation failed: {exc}", file=sys.stderr)
            return 2
    if (
        args.matrix_validate_only
        or args.matrix_execute
        or args.matrix_resume
        or args.matrix_authorization_phrase is not None
        or args.matrix_authorization_at is not None
        or args.capture_private_prose
        or args.ack_private_prose_pii is not None
        or args.matrix_worker
        or args.matrix_worker_block is not None
        or args.matrix_out
        or args.matrix_cost_offset
    ):
        parser.error("matrix-only flags require --judgment-matrix")

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    unknown = stages - {"preflight", "formation", "judge"}
    if unknown:
        print(f"unknown stage(s): {sorted(unknown)}", file=sys.stderr)
        return 2

    out_dir = args.out or (
        REPO_ROOT
        / "output/state/shadow_replay"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # The replay IS a shadow tool: enable the dispatch flag for this process
    # regardless of .env, so the one command needs no env choreography.
    # (dispatch_strategy_shadow reads config.SHADOW_STRATEGY_ENABLED at call
    # time, so a module-attribute flip is sufficient and process-local.)
    config.SHADOW_STRATEGY_ENABLED = True
    if args.shadow_temperature is not None:
        import shared.llm_clients as llm_clients

        llm_clients.SHADOW_LLM_TEMPERATURE = args.shadow_temperature
        print(f"shadow temperature override: {args.shadow_temperature}")

    print(f"shadow replay → {out_dir}")
    print(
        f"  stages={sorted(stages)}  strategy_model={config.STRATEGY_MODEL_NAME}  "
        f"shadow_strategy={config.SHADOW_STRATEGY_MODEL_NAME}  "
        f"shadow_judge={config.SHADOW_FACIAL_MODEL_NAME}"
    )

    v2_brief = args.v2_brief
    if "preflight" in stages:
        generated = run_preflight(
            args.seed_brief,
            out_dir,
            skip_primary=args.skip_primary,
            drain_timeout=args.drain_timeout,
        )
        # Production sequence: a fresh preflight brief feeds formation/judge
        # unless the operator pinned one explicitly.
        if generated is not None and args.v2_brief is None:
            v2_brief = generated
    if v2_brief is None:
        v2_brief = DEFAULT_V2_BRIEF
    if stages & {"formation", "judge"} and not v2_brief.exists():
        print(
            f"v2 brief not found: {v2_brief} — run with --stages preflight,… "
            "to generate one, or pass --v2-brief",
            file=sys.stderr,
        )
        return 1

    if "formation" in stages:
        run_formation(
            v2_brief,
            out_dir,
            skip_primary=args.skip_primary,
            drain_timeout=args.drain_timeout,
        )
    if "judge" in stages:
        run_judge(v2_brief, args.profiles, args.profile_index, out_dir)

    report = _load_shadow_report_module()
    report.render_strategy_shadows(out_dir)
    report.render_judgments(out_dir, False)
    print(f"\nartifacts: {out_dir}")
    print(f"re-render anytime: .venv/bin/python tools/shadow_report.py {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
