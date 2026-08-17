from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import shared.config as shared_config
from tools import shadow_replay
from tools.glm_experiment_report import _arm_summary, _validate_execution_schedule
from shared.schemas import OpusDecision


@pytest.fixture(autouse=True)
def _pin_test_runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        shadow_replay,
        "_MATRIX_RUNS_ROOT",
        (tmp_path / "output/runs").resolve(),
    )
    monkeypatch.setattr(shadow_replay, "_MATRIX_MIN_FACIAL_CANDIDATES", 1)
    monkeypatch.setattr(shadow_replay, "_MATRIX_MIN_FULL_CALLS", 1)
    monkeypatch.setattr(shadow_replay, "_MATRIX_MIN_FACIAL_PAGE_TIMINGS", 1)


def _arm_env() -> dict[str, str]:
    return {
        "OPUS_MODEL_NAME": "accounts/fireworks/models/glm-5p2",
        "FACIAL_MODEL_NAME": "accounts/fireworks/models/glm-5p2",
        "FULL_EVAL_MODEL_NAME": "accounts/fireworks/models/glm-5p2",
        "FIREWORKS_JUDGMENT_POLICY_ENABLED": "true",
        "FIREWORKS_BASE_URL": "https://api.fireworks.ai/inference/v1",
        "FIREWORKS_PRIMARY_MIN_MAX_TOKENS": "16384",
        "FIREWORKS_PRIMARY_MAX_COST_USD": "0",
        "FIREWORKS_FACIAL_REASONING_EFFORT": "high",
        "FIREWORKS_FULL_REASONING_EFFORT": "high",
        "FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS": "120",
        "FIREWORKS_FACIAL_TOTAL_DEADLINE_SECONDS": "180",
        "FIREWORKS_FACIAL_MAX_ATTEMPTS": "2",
        "FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS": "240",
        "FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS": "360",
        "FIREWORKS_FULL_MAX_ATTEMPTS": "2",
        "FIREWORKS_PROMPT_AFFINITY_ENABLED": "true",
        "SHADOW_FACIAL_MODEL_ENABLED": "false",
        "SHADOW_STRATEGY_ENABLED": "false",
        "SHADOW_ASYNC_ENABLED": "false",
        "SHADOW_LLM_TIMEOUT_SECONDS": "300",
        "LINKEDIN_V2_FACIAL_CONTRACT": "legacy",
        "LINKEDIN_V2_FULL_CONTRACT": "legacy",
        "LINKEDIN_FACIAL_CONCURRENCY_ENABLED": "false",
        "LINKEDIN_FACIAL_MAX_CONCURRENCY": "1",
        "LINKEDIN_FACIAL_TARGET_BATCH_SIZE": "8",
        "LINKEDIN_EXTERNAL_EVIDENCE_ENABLED": "false",
        "LINKEDIN_FACIAL_BORDERLINE_ENABLED": "false",
    }


def _common_thresholds() -> dict[str, float | int]:
    return {
        "deadline_success_min": 0.99,
        "retryable_429_503_rate_max_increase": 0.02,
        "postcondition_fallback_rate_max_increase": 0.02,
        "unrecovered_postcondition_rate_max": 0.0,
        "max_attempts": 2,
        "human_pass_rate_delta_ci_lower_min": -0.03,
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _fresh_execution_authorization():
    return shadow_replay.MatrixExecutionAuthorization(
        phrase=shadow_replay._MATRIX_EXECUTION_AUTHORIZATION_PHRASE,
        authorized_at=datetime.now(timezone.utc),
    )


def _make_manifest(tmp_path: Path, *, tree: str = "runs") -> Path:
    run_dir = tmp_path / "output" / tree / "linkedin" / "project" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run-manifest.json").write_text(
        json.dumps(
            {
                "source": "linkedin",
                "run_id": 1,
                "ended_at": "2026-07-11T00:00:00+00:00",
                "artifacts_present": [
                    "runtime_state.sqlite3",
                    "snippets.jsonl",
                    "profile_summaries.jsonl",
                ],
            }
        )
    )
    (run_dir / "snippets.jsonl").write_text(
        json.dumps(
            {
                "name": "Private Person",
                "profile_url": "/talent/profile/private",
                "headline": "Builder",
                "current_title": "Engineer",
                "current_company": "Private Company",
                "location": "Remote",
                "education_snippet": "",
                "source_string_id": 7,
                "source_string_name": "test",
                "page": 1,
                "result_rank": 1,
                "card_index": 0,
            }
        )
        + "\n"
    )
    profile = {
        "name": "Private Person",
        "profile_url": "/talent/profile/private",
        "headline": "Builder",
        "experiences": [],
        "education": [],
        "skills_snippet": [],
    }
    (run_dir / "profile_summaries.jsonl").write_text(
        json.dumps(profile) + "\n" + json.dumps(profile) + "\n"
    )
    brief = {
        "role_title": "Canary",
        "role_summary": "Canary role",
        "minimum_bar_description": "Direct evidence",
        "capability_areas": [
            {
                "name": "Building",
                "description": "Built systems",
                "evidence_signals": ["owned delivery"],
            }
        ],
    }
    canonical = json.dumps(brief, sort_keys=True, separators=(",", ":"))
    with sqlite3.connect(run_dir / "runtime_state.sqlite3") as conn:
        conn.execute(
            "CREATE TABLE runs (id INTEGER PRIMARY KEY, brief_snapshot_json TEXT, brief_content_hash TEXT)"
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?)",
            (1, json.dumps(brief), shadow_replay._sha256_bytes(canonical.encode())),
        )

    manifest = {
        "schema_version": shadow_replay.MATRIX_SCHEMA_VERSION,
        "experiment_id": "matrix-test",
        "random_seed": 12345,
        "git_sha": _git_sha(),
        "source_hashes": {
            relative: shadow_replay._sha256_file(shadow_replay.REPO_ROOT / relative)
            for relative in shadow_replay._MATRIX_RUNTIME_FILES
        },
        "authorization": {
            "approved": True,
            "approved_by": "test operator",
            "approved_at": "2026-07-11T00:00:00Z",
            "max_cost_usd": 10.0,
        },
        "dataset": {
            "run_dir": str(run_dir),
            "run_id": 1,
            "hashes": {
                name: shadow_replay._sha256_file(run_dir / name)
                for name in (
                    "runtime_state.sqlite3",
                    "snippets.jsonl",
                    "profile_summaries.jsonl",
                )
            },
            "facial_line_numbers": [1],
            "full_line_numbers": [1, 2],
            "facial_page_repetitions": 2,
            "cache_warmup_execution_units_per_stage": 1,
        },
        "limits": {
            "facial_max_input_tokens": 65536,
            "facial_max_output_tokens": 16384,
            "full_max_input_tokens": 131072,
            "full_max_output_tokens": 16384,
        },
        "arms": [
            {
                "id": "standard-high",
                "facial_mode": "pagewide",
                "env": _arm_env(),
            },
            {
                "id": "standard-no-affinity",
                "facial_mode": "pagewide",
                "env": {
                    **_arm_env(),
                    "FIREWORKS_PROMPT_AFFINITY_ENABLED": "false",
                },
            },
        ],
        "comparisons": [
            {
                "id": "affinity",
                "lever_id": "prompt_affinity",
                "depends_on": None,
                "control_arm_id": "standard-no-affinity",
                "treatment_arm_id": "standard-high",
                "lever_fields": [
                    "env.FIREWORKS_PROMPT_AFFINITY_ENABLED"
                ],
                "thresholds": _common_thresholds(),
            }
        ],
        "review": {"agreement_sample_size": 1},
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def test_snapshot_brief_accepts_runtime_exact_serialized_hash(tmp_path: Path):
    db_path = tmp_path / "runtime_state.sqlite3"
    brief = {"z": [1, 2], "a": {"nested": True}}
    snapshot_json = json.dumps(brief, indent=2, sort_keys=False)
    exact_hash = shadow_replay._sha256_bytes(snapshot_json.encode("utf-8"))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE runs (id INTEGER PRIMARY KEY, brief_snapshot_json TEXT, brief_content_hash TEXT)"
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?)",
            (1, snapshot_json, exact_hash),
        )

    assert shadow_replay._load_snapshot_brief(db_path, 1) == brief


def _mutate_manifest(path: Path, mutate) -> None:
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2))


def test_matrix_validation_pins_immutable_inputs_and_builds_exact_calls(tmp_path: Path):
    path = _make_manifest(tmp_path)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    assert spec.experiment_id == "matrix-test"
    assert spec.worst_case_cost_usd > 0
    plans = spec.call_plans["standard-high"]
    assert [plan["stage"] for plan in plans] == [
        "facial",
        "facial",
        "full",
        "full",
    ]
    assert all(plan["call_id"] for plan in plans)
    assert all("Private Person" not in json.dumps(plan) for plan in plans)
    assert [plan["call_id"] for plan in plans] == [
        plan["call_id"] for plan in spec.call_plans["standard-no-affinity"]
    ]
    control_order, control_phases = shadow_replay._cache_phased_execution_order(
        plans,
        random_seed=spec.raw["random_seed"],
        warmup_units_per_stage=1,
    )
    treatment_order, treatment_phases = shadow_replay._cache_phased_execution_order(
        spec.call_plans["standard-no-affinity"],
        random_seed=spec.raw["random_seed"],
        warmup_units_per_stage=1,
    )
    assert control_order == treatment_order
    assert control_phases == treatment_phases
    assert [control_phases[unit_id] for unit_id in control_order] == [
        "warmup",
        "warmup",
        "warm",
        "warm",
    ]


def test_matrix_validation_rejects_live_state_tree(tmp_path: Path):
    path = _make_manifest(tmp_path, tree="state")
    with pytest.raises(shadow_replay.MatrixValidationError, match="inside this checkout"):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_validation_rejects_hash_drift(tmp_path: Path):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["dataset"]["hashes"].__setitem__(
            "snippets.jsonl", "sha256:" + "0" * 64
        ),
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match="hash mismatch"):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_validation_requires_a_finalized_linkedin_run_manifest(tmp_path: Path):
    path = _make_manifest(tmp_path)
    manifest_path = path.parent / "output/runs/linkedin/project/run-1/run-manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["ended_at"] = None
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(shadow_replay.MatrixValidationError, match="valid ended_at"):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_validation_requires_explicit_authorization(tmp_path: Path):
    path = _make_manifest(tmp_path)
    _mutate_manifest(path, lambda payload: payload["authorization"].update(approved=False))
    with pytest.raises(shadow_replay.MatrixValidationError, match="authorization"):
        shadow_replay._load_and_validate_matrix_manifest(path)


@pytest.mark.parametrize("max_cost", [float("inf"), float("nan")])
def test_matrix_validation_requires_finite_authorization_cap(
    tmp_path: Path, max_cost: float
):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["authorization"].update(max_cost_usd=max_cost),
    )
    with pytest.raises(
        shadow_replay.MatrixValidationError, match="finite and positive"
    ):
        shadow_replay._load_and_validate_matrix_manifest(path)


@pytest.mark.parametrize(
    "approved_at",
    ["2026-07-11T00:00:00", "not-a-timestamp"],
)
def test_matrix_validation_requires_timezone_aware_approval_timestamp(
    tmp_path: Path, approved_at: str
):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["authorization"].update(
            approved_at=approved_at
        ),
    )
    with pytest.raises(
        shadow_replay.MatrixValidationError, match="timezone-aware"
    ):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_runtime_execution_authorization_requires_exact_fresh_aware_flags():
    now = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
    valid = shadow_replay._validate_matrix_execution_authorization(
        phrase=shadow_replay._MATRIX_EXECUTION_AUTHORIZATION_PHRASE,
        authorized_at=(now - timedelta(minutes=15)).isoformat(),
        now=now,
    )
    assert valid.authorized_at == now - timedelta(minutes=15)

    with pytest.raises(
        shadow_replay.MatrixValidationError, match="phrase does not match exactly"
    ):
        shadow_replay._validate_matrix_execution_authorization(
            phrase="yes",
            authorized_at=now.isoformat(),
            now=now,
        )
    with pytest.raises(shadow_replay.MatrixValidationError, match="stale"):
        shadow_replay._validate_matrix_execution_authorization(
            phrase=shadow_replay._MATRIX_EXECUTION_AUTHORIZATION_PHRASE,
            authorized_at=(now - timedelta(minutes=15, seconds=1)).isoformat(),
            now=now,
        )
    with pytest.raises(
        shadow_replay.MatrixValidationError, match="timezone-aware"
    ):
        shadow_replay._validate_matrix_execution_authorization(
            phrase=shadow_replay._MATRIX_EXECUTION_AUTHORIZATION_PHRASE,
            authorized_at="2026-07-11T12:00:00",
            now=now,
        )


def test_matrix_validation_enforces_worst_case_dollar_cap(tmp_path: Path):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path, lambda payload: payload["authorization"].update(max_cost_usd=0.000001)
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match="exceeds authorization cap"):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_validation_enforces_promotion_grade_facial_sample_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    monkeypatch.setattr(shadow_replay, "_MATRIX_MIN_FACIAL_CANDIDATES", 2)
    with pytest.raises(
        shadow_replay.MatrixValidationError, match="at least 2 unique facial candidates"
    ):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_validation_enforces_promotion_grade_full_call_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    monkeypatch.setattr(shadow_replay, "_MATRIX_MIN_FULL_CALLS", 3)
    with pytest.raises(shadow_replay.MatrixValidationError, match="at least 3 full calls"):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_validation_requires_repeatable_page_timing_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    monkeypatch.setattr(shadow_replay, "_MATRIX_MIN_FACIAL_PAGE_TIMINGS", 3)
    with pytest.raises(
        shadow_replay.MatrixValidationError, match="at least 3 facial page timings"
    ):
        shadow_replay._load_and_validate_matrix_manifest(path)

    _mutate_manifest(
        path,
        lambda payload: payload["dataset"].update(facial_page_repetitions=3),
    )
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    facial_plans = [
        plan for plan in spec.call_plans["standard-high"] if plan["stage"] == "facial"
    ]
    assert len(facial_plans) == 3
    assert len({plan["execution_unit_id"] for plan in facial_plans}) == 3
    assert len({plan["call_id"] for plan in facial_plans}) == 3


@pytest.mark.parametrize("warmup_count", [0, 2])
def test_matrix_validation_requires_one_pinned_warmup_unit_per_stage(
    tmp_path: Path, warmup_count: int
):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["dataset"].update(
            cache_warmup_execution_units_per_stage=warmup_count
        ),
    )
    with pytest.raises(
        shadow_replay.MatrixValidationError,
        match="cache_warmup_execution_units_per_stage must equal 1",
    ):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_runtime_input_ceiling_includes_fireworks_cached_prompt_slice(tmp_path: Path):
    path = _make_manifest(tmp_path)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    usage_path = tmp_path / "token-cost-log.jsonl"
    usage_path.write_text(
        json.dumps(
            {
                "stage": "facial",
                "input_tokens": 100_000,
                "cache_read_input_tokens": 40_000,
                "cache_creation_input_tokens": 0,
                "output_tokens": 1,
            }
        )
        + "\n"
    )
    with pytest.raises(
        shadow_replay.MatrixSpendCapExceeded, match="input exceeded"
    ):
        shadow_replay._assert_arm_usage_within_declared_limits(
            usage_path,
            spec=spec,
            arm=spec.raw["arms"][0],
        )


def test_runtime_stops_immediately_when_finished_call_has_no_usage_receipt(
    tmp_path: Path,
):
    with pytest.raises(
        shadow_replay.MatrixSpendCapExceeded, match="no measured usage receipt"
    ):
        shadow_replay._assert_exact_usage_receipt(
            tmp_path / "missing-token-cost-log.jsonl",
            "call-1",
        )


def test_matrix_validation_rejects_limits_below_real_fireworks_floor(tmp_path: Path):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path, lambda payload: payload["limits"].update(facial_max_output_tokens=1)
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match="request floor"):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_validation_rejects_secret_or_unknown_arm_env(tmp_path: Path):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["arms"][0]["env"].update(FIREWORKS_API_KEY="secret"),
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match="forbidden env"):
        shadow_replay._load_and_validate_matrix_manifest(path)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        (
            {
                "LINKEDIN_V2_FACIAL_CONTRACT": "tool",
                "FIREWORKS_JUDGMENT_POLICY_ENABLED": "false",
            },
            "tool contracts require explicit Fireworks policy",
        ),
        (
            {"LINKEDIN_FACIAL_MAX_CONCURRENCY": "4"},
            "max concurrency cannot exceed 3",
        ),
        (
            {"FACIAL_MODEL_NAME": "claude-opus-4-6"},
            "policy requires a Fireworks FACIAL_MODEL_NAME",
        ),
        (
            {"FULL_EVAL_MODEL_NAME": "claude-opus-4-6"},
            "policy requires a Fireworks FULL_EVAL_MODEL_NAME",
        ),
        (
            {"FIREWORKS_FACIAL_REASONING_EFFORT": ""},
            "requires explicit high/max FIREWORKS_FACIAL_REASONING_EFFORT",
        ),
        (
            {"FIREWORKS_FULL_REASONING_EFFORT": "medium"},
            "requires explicit high/max FIREWORKS_FULL_REASONING_EFFORT",
        ),
    ],
)
def test_matrix_arm_rejects_unsafe_policy_contract_and_concurrency_combinations(
    tmp_path: Path,
    updates: dict[str, str],
    match: str,
):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["arms"][0]["env"].update(updates),
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match=match):
        shadow_replay._load_and_validate_matrix_manifest(path)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"SHADOW_FACIAL_MODEL_ENABLED": "true"}, "requires SHADOW_FACIAL_MODEL_ENABLED=false"),
        ({"SHADOW_STRATEGY_ENABLED": "true"}, "requires SHADOW_STRATEGY_ENABLED=false"),
        ({"SHADOW_ASYNC_ENABLED": "true"}, "requires SHADOW_ASYNC_ENABLED=false"),
        ({"LINKEDIN_FACIAL_BORDERLINE_ENABLED": "true"}, "requires LINKEDIN_FACIAL_BORDERLINE_ENABLED=false"),
        ({"FIREWORKS_BASE_URL": "https://example.invalid/v1"}, "canonical Fireworks endpoint"),
        ({"FIREWORKS_PRIMARY_MIN_MAX_TOKENS": "8192"}, "must equal 16384"),
        ({"FIREWORKS_PRIMARY_MAX_COST_USD": "1"}, "must equal 0"),
        ({"SHADOW_LLM_TIMEOUT_SECONDS": "0"}, "must be positive"),
        ({"FIREWORKS_FACIAL_MAX_ATTEMPTS": "3"}, "cannot exceed 2"),
        ({"FIREWORKS_FULL_MAX_ATTEMPTS": "3"}, "cannot exceed 2"),
    ],
)
def test_matrix_arm_rejects_ambient_prompt_transport_and_retry_drift(
    tmp_path: Path,
    updates: dict[str, str],
    match: str,
):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["arms"][0]["env"].update(updates),
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match=match):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_arm_rejects_effective_concurrency_without_policy_and_facial_tool(
    tmp_path: Path,
):
    path = _make_manifest(tmp_path)

    def mutate(payload):
        payload["arms"][0]["facial_mode"] = "partitioned_concurrent"
        payload["arms"][0]["env"].update(
            LINKEDIN_FACIAL_CONCURRENCY_ENABLED="true",
            LINKEDIN_FACIAL_MAX_CONCURRENCY="2",
        )

    _mutate_manifest(path, mutate)
    with pytest.raises(
        shadow_replay.MatrixValidationError,
        match="concurrency >1 requires Fireworks policy and facial tool contract",
    ):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_comparison_rejects_accidental_second_lever(tmp_path: Path):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["arms"][1]["env"].update(
            FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS="999"
        ),
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match="does not equal"):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_comparison_allows_declared_facial_mode_lever(tmp_path: Path):
    path = _make_manifest(tmp_path)

    def mutate(payload):
        payload["arms"][1]["facial_mode"] = "partitioned_concurrent"
        payload["arms"][1]["env"]["LINKEDIN_FACIAL_CONCURRENCY_ENABLED"] = "true"
        payload["arms"][1]["env"]["FIREWORKS_PROMPT_AFFINITY_ENABLED"] = "true"
        payload["comparisons"][0]["lever_id"] = "facial_concurrency"
        payload["comparisons"][0]["lever_fields"] = [
            "facial_mode",
            "env.LINKEDIN_FACIAL_CONCURRENCY_ENABLED",
        ]
        payload["comparisons"][0]["thresholds"].update(
            {
                "facial_page_p50_improvement_min": 0.30,
                "facial_page_p50_ci_lower_min": 0.20,
                "facial_page_p90_improvement_min": 0.30,
                "facial_page_p90_ci_lower_min": 0.20,
            }
        )

    _mutate_manifest(path, mutate)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    assert "facial_mode" in spec.raw["comparisons"][0]["lever_fields"]
    _mutate_manifest(
        path,
        lambda payload: payload["comparisons"][0]["thresholds"].update(
            facial_page_p50_improvement_min=0.29
        ),
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match="weaker than 0.30"):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_comparison_requires_full_nonweak_gate_profile(tmp_path: Path):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["comparisons"][0]["thresholds"].pop(
            "deadline_success_min"
        ),
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match="omits required"):
        shadow_replay._load_and_validate_matrix_manifest(path)

    _mutate_manifest(
        path,
        lambda payload: payload["comparisons"][0]["thresholds"].update(
            deadline_success_min=0.98
        ),
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match="weaker than 0.99"):
        shadow_replay._load_and_validate_matrix_manifest(path)

    _mutate_manifest(
        path,
        lambda payload: payload["comparisons"][0]["thresholds"].update(
            deadline_success_min=0.99,
            unrecovered_postcondition_rate_max=0.01,
        ),
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match="absolute zero"):
        shadow_replay._load_and_validate_matrix_manifest(path)


@pytest.mark.parametrize(
    ("key", "value", "domain"),
    [
        ("retryable_429_503_rate_max", 1.1, r"\[0, 1\]"),
        ("facial_page_p50_improvement_min", 1.1, r"\[-1, 1\]"),
        ("human_pass_rate_delta_ci_lower_min", -1.1, r"\[-1, 1\]"),
    ],
)
def test_matrix_threshold_numeric_domains_are_bounded(
    tmp_path: Path, key: str, value: float, domain: str
):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["comparisons"][0]["thresholds"].update(
            {key: value}
        ),
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match=domain):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_comparisons_form_an_ordered_treatment_chain(tmp_path: Path):
    path = _make_manifest(tmp_path)

    def add_second(payload):
        next_env = dict(payload["arms"][0]["env"])
        next_env["FIREWORKS_FACIAL_REASONING_EFFORT"] = "max"
        next_env["FIREWORKS_FULL_REASONING_EFFORT"] = "max"
        payload["arms"].append(
            {"id": "standard-max", "facial_mode": "pagewide", "env": next_env}
        )
        payload["comparisons"].append(
            {
                "id": "reasoning",
                "lever_id": "reasoning_effort",
                "depends_on": "affinity",
                "control_arm_id": "standard-high",
                "treatment_arm_id": "standard-max",
                "lever_fields": [
                    "env.FIREWORKS_FACIAL_REASONING_EFFORT",
                    "env.FIREWORKS_FULL_REASONING_EFFORT",
                ],
                "thresholds": _common_thresholds(),
            }
        )

    _mutate_manifest(path, add_second)
    assert len(shadow_replay._load_and_validate_matrix_manifest(path).raw["comparisons"]) == 2
    _mutate_manifest(
        path,
        lambda payload: payload["comparisons"][1].update(depends_on=None),
    )
    with pytest.raises(shadow_replay.MatrixValidationError, match="immediately prior"):
        shadow_replay._load_and_validate_matrix_manifest(path)


def test_matrix_validate_only_never_creates_artifacts_or_calls_provider(tmp_path: Path):
    path = _make_manifest(tmp_path)
    out = tmp_path / "matrix-output"
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    assert shadow_replay._run_matrix_parent(spec, out, validate_only=True) == 0
    assert not out.exists()


def test_matrix_output_requires_explicit_external_artifact_path(tmp_path: Path):
    spec = shadow_replay._load_and_validate_matrix_manifest(_make_manifest(tmp_path))

    with pytest.raises(shadow_replay.MatrixValidationError, match="explicit --out"):
        shadow_replay._matrix_output_dir(spec, None)
    with pytest.raises(shadow_replay.MatrixValidationError, match="outside"):
        shadow_replay._matrix_output_dir(
            spec,
            shadow_replay.REPO_ROOT / "output/debug/private-matrix",
        )

    external = tmp_path / "external-matrix"
    assert shadow_replay._matrix_output_dir(spec, external) == external.resolve()


def test_matrix_cli_requires_execute_and_fresh_runtime_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    out = tmp_path / "matrix-output"
    parent_calls: list[dict] = []

    def fake_parent(_spec, _out, **kwargs):
        parent_calls.append(kwargs)
        return 0

    monkeypatch.setattr(shadow_replay, "_run_matrix_parent", fake_parent)
    monkeypatch.setattr(
        "sys.argv",
        [
            "shadow_replay.py",
            "--judgment-matrix",
            str(path),
            "--out",
            str(out),
        ],
    )
    assert shadow_replay.main() == 2
    assert parent_calls == []

    monkeypatch.setattr(
        "sys.argv",
        [
            "shadow_replay.py",
            "--judgment-matrix",
            str(path),
            "--out",
            str(out),
            "--matrix-execute",
            "--matrix-authorization-phrase",
            shadow_replay._MATRIX_EXECUTION_AUTHORIZATION_PHRASE,
            "--matrix-authorization-at",
            (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat(),
        ],
    )
    assert shadow_replay.main() == 2
    assert parent_calls == []

    monkeypatch.setattr(
        "sys.argv",
        [
            "shadow_replay.py",
            "--judgment-matrix",
            str(path),
            "--out",
            str(out),
            "--matrix-execute",
            "--matrix-authorization-phrase",
            shadow_replay._MATRIX_EXECUTION_AUTHORIZATION_PHRASE,
            "--matrix-authorization-at",
            datetime.now(timezone.utc).isoformat(),
        ],
    )
    assert shadow_replay.main() == 0
    assert len(parent_calls) == 1
    assert parent_calls[0]["validate_only"] is False
    assert isinstance(
        parent_calls[0]["execution_authorization"],
        shadow_replay.MatrixExecutionAuthorization,
    )
    assert parent_calls[0]["capture_private_prose"] is False

    monkeypatch.setattr(
        "sys.argv",
        [
            "shadow_replay.py",
            "--judgment-matrix",
            str(path),
            "--out",
            str(out),
            "--matrix-execute",
            "--matrix-authorization-phrase",
            shadow_replay._MATRIX_EXECUTION_AUTHORIZATION_PHRASE,
            "--matrix-authorization-at",
            datetime.now(timezone.utc).isoformat(),
            "--capture-private-prose",
            "--ack-private-prose-pii",
            "wrong",
        ],
    )
    assert shadow_replay.main() == 2
    assert len(parent_calls) == 1

    monkeypatch.setattr(
        "sys.argv",
        [
            "shadow_replay.py",
            "--judgment-matrix",
            str(path),
            "--out",
            str(out),
            "--matrix-execute",
            "--matrix-authorization-phrase",
            shadow_replay._MATRIX_EXECUTION_AUTHORIZATION_PHRASE,
            "--matrix-authorization-at",
            datetime.now(timezone.utc).isoformat(),
            "--capture-private-prose",
            "--ack-private-prose-pii",
            shadow_replay._PRIVATE_PROSE_PII_ACK,
        ],
    )
    assert shadow_replay.main() == 0
    assert len(parent_calls) == 2
    assert parent_calls[1]["capture_private_prose"] is True


def test_matrix_worker_cli_refuses_calls_without_a_live_parent_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    worker_out = tmp_path / "matrix-output/arms/standard-high"
    monkeypatch.setattr(
        shadow_replay,
        "_run_matrix_worker",
        lambda *_args, **_kwargs: pytest.fail("unowned worker was invoked"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "shadow_replay.py",
            "--judgment-matrix",
            str(path),
            "--matrix-worker",
            "standard-high",
            "--matrix-worker-block",
            "0",
            "--matrix-out",
            str(worker_out),
        ],
    )
    assert shadow_replay.main() == 2


def test_matrix_block_schedule_is_deterministic_paired_and_bounded():
    arm_ids = ["arm-a", "arm-b", "arm-c"]
    units = [f"unit-{index:02d}" for index in range(21)]
    first = shadow_replay._matrix_block_schedule(
        arm_ids, units, random_seed=12345, block_size=8
    )
    second = shadow_replay._matrix_block_schedule(
        arm_ids, units, random_seed=12345, block_size=8
    )

    assert first == second
    assert [unit for block in first for unit in block["execution_unit_ids"]] == units
    assert [len(block["execution_unit_ids"]) for block in first] == [8, 8, 5]
    assert [block["block_index"] for block in first] == [0, 1, 2]
    assert all(set(block["arm_order"]) == set(arm_ids) for block in first)
    assert all(len(block["arm_order"]) == len(arm_ids) for block in first)


def test_two_matrix_parents_cannot_own_the_same_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    out = tmp_path / "matrix-output"
    entered = threading.Event()
    release = threading.Event()
    first_results: list[int] = []

    def hold_owned_parent(_spec, _out, **_kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return 0

    monkeypatch.setattr(
        shadow_replay, "_run_matrix_parent_owned", hold_owned_parent
    )

    thread = threading.Thread(
        target=lambda: first_results.append(
            shadow_replay._run_matrix_parent(
                spec,
                out,
                validate_only=False,
                execution_authorization=_fresh_execution_authorization(),
            )
        )
    )
    thread.start()
    assert entered.wait(timeout=5)
    try:
        second_result = shadow_replay._run_matrix_parent(
            spec,
            out,
            validate_only=False,
            execution_authorization=_fresh_execution_authorization(),
        )
    finally:
        release.set()
        thread.join(timeout=5)

    assert second_result == 2
    assert first_results == [0]
    assert not thread.is_alive()


def test_matrix_output_ownership_is_exclusive_across_processes(tmp_path: Path):
    out = tmp_path / "matrix-output"
    out.mkdir()
    lock_path = out / shadow_replay._MATRIX_OUTPUT_LOCK_FILENAME
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, os, sys; "
                "fd=os.open(sys.argv[1], os.O_RDWR|os.O_CREAT, 0o600); "
                "fcntl.flock(fd, fcntl.LOCK_EX); "
                "print('ready', flush=True); sys.stdin.readline()"
            ),
            str(lock_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(
            shadow_replay.MatrixValidationError, match="already owned"
        ):
            with shadow_replay._matrix_output_ownership(out, resume=True):
                pytest.fail("concurrent process acquired matrix output")
    finally:
        if child.stdin is not None:
            child.stdin.write("\n")
            child.stdin.flush()
        child.wait(timeout=5)


def test_internal_worker_capability_requires_the_live_owning_parent(
    tmp_path: Path,
):
    out = tmp_path / "matrix-output"
    arm_dir = out / "arms" / "arm-a"
    metadata: dict = {}
    with shadow_replay._matrix_output_ownership(out, resume=False):
        token = shadow_replay._new_matrix_worker_capability(metadata)
        shadow_replay._write_json(out / "experiment-metadata.json", metadata)
        env = dict(os.environ)
        env[shadow_replay._MATRIX_WORKER_CAPABILITY_ENV] = token
        command = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from tools.shadow_replay import _validate_internal_matrix_worker; "
                "_validate_internal_matrix_worker(Path(__import__('sys').argv[1]))"
            ),
            str(arm_dir),
        ]
        owned = subprocess.run(
            command,
            cwd=shadow_replay.REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert owned.returncode == 0, owned.stderr

    unowned = subprocess.run(
        command,
        cwd=shadow_replay.REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert unowned.returncode != 0
    assert "no live owning parent" in unowned.stderr


def _prepare_clean_resume_boundary(
    spec: shadow_replay.MatrixSpec, out: Path
) -> tuple[str, str, float]:
    expected = shadow_replay._build_matrix_parent_metadata(spec)
    dispatch_plan = shadow_replay._matrix_dispatch_plan(
        expected["execution_schedule"]
    )
    first_block, completed_arm_id = dispatch_plan[0]
    _next_block, next_arm_id = dispatch_plan[1]
    expected["status"] = "running"
    expected["worker_capability_sha256"] = "sha256:" + "a" * 64
    expected["dispatch_log"] = [
        {
            "block_index": first_block,
            "arm_id": completed_arm_id,
            "status": "complete",
        }
    ]
    shadow_replay._write_json(out / "experiment-metadata.json", expected)

    arms = {str(arm["id"]): arm for arm in spec.raw["arms"]}
    arm = arms[completed_arm_id]
    phased_orders, execution_schedule = (
        shadow_replay._paired_matrix_execution_design(spec)
    )
    execution_units, cache_phases = phased_orders[completed_arm_id]
    effective_env = {
        key: (
            "1"
            if key == "LANGFUSE_DISABLE"
            else str(arm["env"].get(key, ""))
        )
        for key in sorted(shadow_replay._MATRIX_ALLOWED_ENV)
    }
    arm_metadata = shadow_replay._build_matrix_arm_expected_metadata(
        spec,
        arm=arm,
        arm_id=completed_arm_id,
        plans=spec.call_plans[completed_arm_id],
        execution_units=execution_units,
        cache_phases=cache_phases,
        execution_schedule=execution_schedule,
        effective_env=effective_env,
    )
    arm_metadata.update(
        {
            "completed_block_indices": [0],
            "block_cost_offsets_usd": [0.0],
            "status": (
                "complete" if len(execution_schedule) == 1 else "running"
            ),
            "started_at": "2026-07-11T00:00:00+00:00",
        }
    )
    arm_dir = out / "arms" / completed_arm_id
    shadow_replay._write_json(arm_dir / "arm-metadata.json", arm_metadata)

    completed_units = list(execution_schedule[0]["execution_unit_ids"])
    plans = [
        plan
        for plan in spec.call_plans[completed_arm_id]
        if plan["execution_unit_id"] in set(completed_units)
    ]
    (arm_dir / "calls.jsonl").write_text(
        "".join(json.dumps({"call_id": plan["call_id"]}) + "\n" for plan in plans)
    )
    (arm_dir / "execution-units.jsonl").write_text(
        "".join(
            json.dumps({"execution_unit_id": unit_id}) + "\n"
            for unit_id in completed_units
        )
    )
    per_call_cost = 0.001
    (arm_dir / "token-cost-log.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "logical_call_id": plan["call_id"],
                    "stage": plan["stage"],
                    "usage_status": "measured",
                    "cost_completeness": "complete",
                    "estimated_cost_usd": per_call_cost,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            )
            + "\n"
            for plan in plans
        )
    )
    (arm_dir / "llm-attempts.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "logical_call_id": plan["call_id"],
                    "attempt_number": 1,
                    "max_attempts": 2,
                    "status": "response_received",
                }
            )
            + "\n"
            for plan in plans
        )
    )
    return completed_arm_id, next_arm_id, len(plans) * per_call_cost


def test_matrix_resume_continues_only_from_a_verified_clean_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    out = tmp_path / "matrix-output"
    _completed_arm, next_arm, prior_cost = _prepare_clean_resume_boundary(
        spec, out
    )
    worker_calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        if "--matrix-worker" in command:
            worker_calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shadow_replay.subprocess, "run", fake_run)
    assert (
        shadow_replay._run_matrix_parent(
            spec,
            out,
            validate_only=False,
            execution_authorization=_fresh_execution_authorization(),
            resume=True,
        )
        == 0
    )
    assert len(worker_calls) == 1
    command = worker_calls[0]
    assert command[command.index("--matrix-worker") + 1] == next_arm
    assert float(
        command[command.index("--matrix-cost-offset") + 1]
    ) == pytest.approx(prior_cost)
    metadata = json.loads((out / "experiment-metadata.json").read_text())
    assert metadata["resume_count"] == 1
    assert metadata["status"] == "complete"
    assert [row["status"] for row in metadata["dispatch_log"]] == [
        "complete",
        "complete",
    ]


@pytest.mark.parametrize("dispatch_status", ["running", "failed"])
def test_matrix_resume_refuses_inflight_or_failed_current_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dispatch_status: str,
):
    path = _make_manifest(tmp_path)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    out = tmp_path / "matrix-output"
    _prepare_clean_resume_boundary(spec, out)
    metadata_path = out / "experiment-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dispatch_log"][-1]["status"] = dispatch_status
    metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(
        shadow_replay.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("resume attempted a subprocess"),
    )

    assert (
        shadow_replay._run_matrix_parent(
            spec,
            out,
            validate_only=False,
            execution_authorization=_fresh_execution_authorization(),
            resume=True,
        )
        == 2
    )


@pytest.mark.parametrize("partial_kind", ["metadata", "receipts"])
def test_matrix_resume_refuses_partial_arm_metadata_or_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial_kind: str,
):
    path = _make_manifest(tmp_path)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    out = tmp_path / "matrix-output"
    completed_arm, _next_arm, _cost = _prepare_clean_resume_boundary(spec, out)
    arm_dir = out / "arms" / completed_arm
    if partial_kind == "metadata":
        arm_metadata_path = arm_dir / "arm-metadata.json"
        arm_metadata = json.loads(arm_metadata_path.read_text())
        arm_metadata["current_block_index"] = 0
        arm_metadata_path.write_text(json.dumps(arm_metadata))
    else:
        (arm_dir / "llm-attempts.jsonl").write_text("")
    monkeypatch.setattr(
        shadow_replay.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("resume attempted a subprocess"),
    )

    assert (
        shadow_replay._run_matrix_parent(
            spec,
            out,
            validate_only=False,
            execution_authorization=_fresh_execution_authorization(),
            resume=True,
        )
        == 2
    )


def test_matrix_resume_refuses_pinned_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    out = tmp_path / "matrix-output"
    _prepare_clean_resume_boundary(spec, out)
    metadata_path = out / "experiment-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    first_source = next(iter(metadata["source_hashes"]))
    metadata["source_hashes"][first_source] = "sha256:" + "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(
        shadow_replay.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("resume attempted a subprocess"),
    )

    assert (
        shadow_replay._run_matrix_parent(
            spec,
            out,
            validate_only=False,
            execution_authorization=_fresh_execution_authorization(),
            resume=True,
        )
        == 2
    )


def test_matrix_parent_dispatches_the_persisted_block_schedule_with_isolated_envs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["dataset"].update(facial_page_repetitions=8),
    )
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    out = tmp_path / "matrix-output"
    worker_calls: list[tuple[int, str, float, dict[str, str]]] = []
    report_calls = 0
    arm_costs = {arm["id"]: 0.0 for arm in spec.raw["arms"]}

    monkeypatch.setattr(
        shadow_replay,
        "_completed_matrix_cost",
        lambda _out: sum(arm_costs.values()),
    )
    monkeypatch.setattr(
        shadow_replay,
        "_measured_usage_cost",
        lambda usage_path: arm_costs[usage_path.parent.name],
    )

    def fake_run(command, **kwargs):
        nonlocal report_calls
        if "--matrix-worker" not in command:
            report_calls += 1
            return SimpleNamespace(returncode=0)
        arm_id = command[command.index("--matrix-worker") + 1]
        block_index = int(command[command.index("--matrix-worker-block") + 1])
        cost_offset = float(command[command.index("--matrix-cost-offset") + 1])
        worker_calls.append(
            (block_index, arm_id, cost_offset, dict(kwargs["env"]))
        )
        arm_costs[arm_id] += 1.0
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shadow_replay.subprocess, "run", fake_run)

    assert (
        shadow_replay._run_matrix_parent(
            spec,
            out,
            validate_only=False,
            execution_authorization=_fresh_execution_authorization(),
        )
        == 0
    )
    metadata = json.loads((out / "experiment-metadata.json").read_text())
    expected_dispatches = [
        (block["block_index"], arm_id)
        for block in metadata["execution_schedule"]
        for arm_id in block["arm_order"]
    ]
    assert [
        (block, arm) for block, arm, _offset, _env in worker_calls
    ] == expected_dispatches
    expected_offsets: list[float] = []
    arms = {arm["id"]: arm for arm in spec.raw["arms"]}
    simulated_costs = {arm_id: 0.0 for arm_id in arms}
    for _block, arm_id in expected_dispatches:
        expected_offsets.append(
            sum(cost for other, cost in simulated_costs.items() if other != arm_id)
        )
        simulated_costs[arm_id] += 1.0
    assert [offset for _block, _arm, offset, _env in worker_calls] == expected_offsets
    assert metadata["dispatch_log"] == [
        {"block_index": block, "arm_id": arm, "status": "complete"}
        for block, arm in expected_dispatches
    ]
    assert len(metadata["execution_schedule"]) == 2
    assert metadata["status"] == "complete"
    assert report_calls == 1
    schedule_hash, block_count = _validate_execution_schedule(
        metadata,
        arm_ids=metadata["arm_ids"],
        expected_units=metadata["expected_units"],
        random_seed=metadata["random_seed"],
    )
    assert schedule_hash == metadata["execution_schedule_sha256"]
    assert block_count == 2
    for _block, arm_id, _offset, env in worker_calls:
        assert env["LANGFUSE_DISABLE"] == "1"
        assert all(
            env[key] == str(value)
            for key, value in arms[arm_id]["env"].items()
        )


def test_matrix_parent_stops_on_block_failure_and_persists_attempt_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    out = tmp_path / "matrix-output"
    worker_calls = 0

    monkeypatch.setattr(shadow_replay, "_completed_matrix_cost", lambda _out: 0.0)

    def fake_run(command, **_kwargs):
        nonlocal worker_calls
        assert "--matrix-worker" in command
        worker_calls += 1
        return SimpleNamespace(returncode=0 if worker_calls == 1 else 7)

    monkeypatch.setattr(shadow_replay.subprocess, "run", fake_run)

    assert (
        shadow_replay._run_matrix_parent(
            spec,
            out,
            validate_only=False,
            execution_authorization=_fresh_execution_authorization(),
        )
        == 7
    )
    metadata = json.loads((out / "experiment-metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert [row["status"] for row in metadata["dispatch_log"]] == [
        "complete",
        "failed",
    ]
    assert worker_calls == 2


def test_matrix_parent_stops_immediately_after_measured_spend_exceeds_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    out = tmp_path / "matrix-output"
    measured = iter([0.0, 11.0])
    worker_calls = 0

    monkeypatch.setattr(
        shadow_replay, "_completed_matrix_cost", lambda _out: next(measured)
    )

    def fake_run(command, **_kwargs):
        nonlocal worker_calls
        assert "--matrix-worker" in command
        worker_calls += 1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shadow_replay.subprocess, "run", fake_run)

    assert (
        shadow_replay._run_matrix_parent(
            spec,
            out,
            validate_only=False,
            execution_authorization=_fresh_execution_authorization(),
        )
        == 3
    )
    metadata = json.loads((out / "experiment-metadata.json").read_text())
    assert metadata["status"] == "spend_stopped"
    assert metadata["dispatch_log"][-1]["status"] == "complete"
    assert worker_calls == 1


def test_matrix_worker_uses_primary_judger_seams_and_persists_only_opaque_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["dataset"].update(facial_page_repetitions=8),
    )
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    for key, value in _arm_env().items():
        monkeypatch.setenv(key, value)

    import shared.brief_loader
    import shared.judger
    import shared.llm_usage

    monkeypatch.setattr(
        shared.brief_loader,
        "load_brief",
        lambda _path: SimpleNamespace(has_v2_schema=True),
    )
    contexts: list[dict] = []
    replay_candidate_ids: list[tuple[str, ...]] = []

    def record_usage(context: dict) -> None:
        context.setdefault("judgment_contract_version", "legacy-v1")
        usage = {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        shared.llm_usage.record_llm_attempt(
            provider="fireworks",
            model="accounts/fireworks/models/glm-5p2",
            logical_call_id=context["logical_call_id"],
            attempt_number=1,
            max_attempts=2,
            status="response_received",
            usage=usage,
            usage_status="measured",
            usage_context=context,
        )
        shared.llm_usage.record_llm_usage(
            provider="fireworks",
            model="accounts/fireworks/models/glm-5p2",
            usage=usage,
            request={"logical_call_id": context["logical_call_id"]},
            usage_context=context,
            usage_status="measured",
        )

    def facial(
        snippets,
        _brief,
        prompt_prefix="",
        lane_context=None,
        opaque_candidate_ids=None,
    ):
        context = dict(lane_context or {})
        contexts.append(context)
        replay_candidate_ids.append(tuple(opaque_candidate_ids or ()))
        record_usage(context)
        return [
            OpusDecision(
                stage="facial",
                decision="FACIAL_YES",
                path="none",
                confidence=None,
                rationale="private rationale",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
                prompt_capture={
                    "system_prompt_sha256": "a" * 64,
                    "candidate_text": f"opaque-facial-{index}",
                },
            )
            for index, snippet in enumerate(snippets)
        ]

    def full(summary, _brief, lane_context=None, opaque_candidate_id=None):
        context = dict(lane_context or {})
        contexts.append(context)
        replay_candidate_ids.append((opaque_candidate_id,))
        record_usage(context)
        return OpusDecision(
            stage="full",
            decision="SAVE",
            path="none",
            confidence=0.8,
            rationale="private rationale",
            candidate_name=summary.name,
            profile_url=summary.profile_url,
            prompt_capture={
                "system_prompt_sha256": "a" * 64,
                "candidate_text": "opaque-full",
            },
        )

    monkeypatch.setattr(shared.judger, "facial_judge_batch", facial)
    monkeypatch.setattr(shared.judger, "full_judge", full)
    arm_dir = tmp_path / "worker"
    assert (
        shadow_replay._run_matrix_worker(
            spec, "standard-high", arm_dir, block_index=0
        )
        == 0
    )
    partial_metadata = json.loads((arm_dir / "arm-metadata.json").read_text())
    assert partial_metadata["status"] == "running"
    assert partial_metadata["completed_block_indices"] == [0]
    assert partial_metadata["expected_block_count"] == 2
    assert len(
        (arm_dir / "execution-units.jsonl").read_text().splitlines()
    ) == shadow_replay._MATRIX_INTERLEAVE_BLOCK_SIZE

    assert (
        shadow_replay._run_matrix_worker(
            spec, "standard-high", arm_dir, block_index=1
        )
        == 0
    )
    arm_metadata = json.loads((arm_dir / "arm-metadata.json").read_text())
    assert arm_metadata["status"] == "complete"
    assert arm_metadata["completed_block_indices"] == [0, 1]
    assert arm_metadata["block_cost_offsets_usd"] == [0.0, 0.0]
    calls_text = (arm_dir / "calls.jsonl").read_text()
    assert "Private Person" not in calls_text
    assert "/talent/profile/private" not in calls_text
    assert "private rationale" not in calls_text
    assert not (arm_dir / shadow_replay._PRIVATE_PROSE_FILENAME).exists()
    assert {row["stage"] for row in map(json.loads, calls_text.splitlines())} == {
        "facial",
        "full",
    }
    assert all(context["offline_replay"] is True for context in contexts)
    assert all(context["logical_call_id"] for context in contexts)
    assert len({context["lane_id"] for context in contexts}) == 1
    assert next(iter({context["lane_id"] for context in contexts})).startswith(
        "matrix-"
    )
    assert {context["cache_phase"] for context in contexts} == {"warmup", "warm"}
    assert all(
        all(isinstance(value, str) and value.startswith("cand_") and len(value) == 29 for value in values)
        for values in replay_candidate_ids
    )

    units_text = (arm_dir / "execution-units.jsonl").read_text()
    assert "Private Person" not in units_text
    assert "/talent/profile/private" not in units_text
    assert "private rationale" not in units_text
    units = [json.loads(line) for line in units_text.splitlines()]
    assert {row["stage"] for row in units} == {"facial_page", "full_profile"}
    assert {row["execution_unit_id"] for row in units} == set(
        arm_metadata["execution_unit_order"]
    )
    assert {call_id for row in units for call_id in row["call_ids"]} == {
        row["call_id"] for row in map(json.loads, calls_text.splitlines())
    }
    assert all(row["elapsed_ms"] >= 0 for row in units)
    assert [
        row["cache_phase"] for row in units
    ] == ["warmup", "warmup"] + ["warm"] * 8
    for artifact_name in ("token-cost-log.jsonl", "llm-attempts.jsonl"):
        rows = [
            json.loads(line)
            for line in (arm_dir / artifact_name).read_text().splitlines()
        ]
        assert {row["cache_phase"] for row in rows} == {"warmup", "warm"}

    expected_unit_calls = {
        unit_id: [
            plan["call_id"]
            for plan in spec.call_plans["standard-high"]
            if plan["execution_unit_id"] == unit_id
        ]
        for unit_id in arm_metadata["execution_unit_order"]
    }
    summary, _evidence = _arm_summary(
        "standard-high",
        [plan["call_id"] for plan in spec.call_plans["standard-high"]],
        arm_metadata["execution_unit_order"],
        expected_unit_calls,
        arm_metadata["cache_phase_by_execution_unit"],
        arm_dir,
        experiment_id=spec.experiment_id,
        manifest_hash=spec.manifest_hash,
        execution_schedule_sha256=arm_metadata["execution_schedule_sha256"],
        expected_block_count=arm_metadata["expected_block_count"],
    )
    assert summary["warm_measured_cache_read_ratio"] == 0.0


def test_matrix_worker_stops_immediately_on_unrecovered_postcondition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = _make_manifest(tmp_path)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    for key, value in _arm_env().items():
        monkeypatch.setenv(key, value)

    import shared.brief_loader
    import shared.judger
    import shared.llm_usage

    monkeypatch.setattr(
        shared.brief_loader,
        "load_brief",
        lambda _path: SimpleNamespace(has_v2_schema=True),
    )

    def record_usage(context: dict) -> None:
        context.setdefault("judgment_contract_version", "legacy-v1")
        usage = {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        shared.llm_usage.record_llm_attempt(
            provider="fireworks",
            model="accounts/fireworks/models/glm-5p2",
            logical_call_id=context["logical_call_id"],
            attempt_number=1,
            max_attempts=2,
            status="response_received",
            usage=usage,
            usage_status="measured",
            usage_context=context,
        )
        shared.llm_usage.record_llm_usage(
            provider="fireworks",
            model="accounts/fireworks/models/glm-5p2",
            usage=usage,
            request={"logical_call_id": context["logical_call_id"]},
            usage_context=context,
            usage_status="measured",
        )

    def facial(snippets, _brief, lane_context=None, **_kwargs):
        context = dict(lane_context or {})
        record_usage(context)
        return [
            OpusDecision(
                stage="facial",
                decision="FACIAL_YES",
                path="none",
                confidence=None,
                rationale="ok",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            )
            for snippet in snippets
        ]

    full_calls = 0

    def full(summary, _brief, lane_context=None, **_kwargs):
        nonlocal full_calls
        full_calls += 1
        context = dict(lane_context or {})
        record_usage(context)
        return OpusDecision(
            stage="full",
            decision="PARSE_FAILURE",
            path="parse_failure",
            confidence=0.0,
            rationale="invalid_capability_area",
            candidate_name=summary.name,
            profile_url=summary.profile_url,
        )

    monkeypatch.setattr(shared.judger, "facial_judge_batch", facial)
    monkeypatch.setattr(shared.judger, "full_judge", full)
    arm_dir = tmp_path / "worker-postcondition"

    assert (
        shadow_replay._run_matrix_worker(
            spec,
            "standard-high",
            arm_dir,
            block_index=0,
        )
        == 1
    )
    assert full_calls == 1
    metadata = json.loads((arm_dir / "arm-metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["error_type"] == "MatrixValidationError"
    assert metadata["current_block_index"] == 0
    assert metadata["completed_block_indices"] == []
    calls = [
        json.loads(line)
        for line in (arm_dir / "calls.jsonl").read_text().splitlines()
    ]
    assert [row["actual_status"] for row in calls].count("postcondition_fail") == 1
    assert next(
        row for row in calls if row["actual_status"] == "postcondition_fail"
    )["decisions"] == ["PARSE_FAILURE"]


def test_matrix_worker_records_facial_batch_failure_outcome_and_aborts_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = _make_manifest(tmp_path)
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    for key, value in _arm_env().items():
        monkeypatch.setenv(key, value)

    arm = next(item for item in spec.raw["arms"] if item["id"] == "standard-high")
    arm["facial_mode"] = "partitioned_concurrent"
    facial_unit_id = next(
        plan["execution_unit_id"]
        for plan in spec.call_plans["standard-high"]
        if plan["stage"] == "facial"
    )
    phased_orders, execution_schedule = shadow_replay._paired_matrix_execution_design(
        spec
    )
    facial_schedule = [
        {
            **execution_schedule[0],
            "execution_unit_ids": [facial_unit_id],
        }
    ]
    monkeypatch.setattr(
        shadow_replay,
        "_paired_matrix_execution_design",
        lambda _spec: (phased_orders, facial_schedule),
    )

    import linkedin.facial_batching as facial_batching
    import shared.brief_loader

    monkeypatch.setattr(
        shared.brief_loader,
        "load_brief",
        lambda _path: SimpleNamespace(has_v2_schema=True),
    )
    failure = RuntimeError("provider status 503")

    async def failed_batches(inputs, _judge_batch, **_kwargs):
        return [
            facial_batching.FacialBatchFailureOutcome(
                candidate=item,
                candidate_identity=item.profile_url,
                error=failure,
                batch_index=0,
            )
            for item in inputs
        ]

    monkeypatch.setattr(facial_batching, "run_facial_batches", failed_batches)
    arm_dir = tmp_path / "worker-facial-batch-failure"

    assert (
        shadow_replay._run_matrix_worker(
            spec,
            "standard-high",
            arm_dir,
            block_index=0,
        )
        == 1
    )
    metadata = json.loads((arm_dir / "arm-metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["error_type"] == "RuntimeError"
    assert metadata["completed_block_indices"] == []
    units = [
        json.loads(line)
        for line in (arm_dir / "execution-units.jsonl").read_text().splitlines()
    ]
    assert [row["actual_status"] for row in units] == ["error"]


def test_matrix_worker_private_prose_capture_is_save_only_mode_0600_and_resume_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _make_manifest(tmp_path)
    _mutate_manifest(
        path,
        lambda payload: payload["dataset"].update(facial_page_repetitions=8),
    )
    spec = shadow_replay._load_and_validate_matrix_manifest(path)
    for key, value in _arm_env().items():
        monkeypatch.setenv(key, value)

    import shared.brief_loader
    import shared.judger
    import shared.llm_usage

    monkeypatch.setattr(
        shared.brief_loader,
        "load_brief",
        lambda _path: SimpleNamespace(has_v2_schema=True),
    )

    def record_usage(context: dict) -> None:
        context.setdefault("judgment_contract_version", "legacy-v1")
        usage = {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        shared.llm_usage.record_llm_attempt(
            provider="fireworks",
            model="accounts/fireworks/models/glm-5p2",
            logical_call_id=context["logical_call_id"],
            attempt_number=1,
            max_attempts=2,
            status="response_received",
            usage=usage,
            usage_status="measured",
            usage_context=context,
        )
        shared.llm_usage.record_llm_usage(
            provider="fireworks",
            model="accounts/fireworks/models/glm-5p2",
            usage=usage,
            request={"logical_call_id": context["logical_call_id"]},
            usage_context=context,
            usage_status="measured",
        )

    def facial(snippets, _brief, prompt_prefix="", lane_context=None, **_kwargs):
        context = dict(lane_context or {})
        record_usage(context)
        return [
            OpusDecision(
                stage="facial",
                decision="FACIAL_YES",
                path="none",
                confidence=None,
                rationale="facial rationale must never persist",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            )
            for snippet in snippets
        ]

    full_calls = 0

    def full(summary, _brief, lane_context=None, **_kwargs):
        nonlocal full_calls
        context = dict(lane_context or {})
        record_usage(context)
        full_calls += 1
        decision = "SAVE" if full_calls == 1 else "REJECT"
        return OpusDecision(
            stage="full",
            decision=decision,
            path="none",
            confidence=0.8,
            rationale=f"private {decision.lower()} rationale for {summary.name}",
            candidate_name=summary.name,
            profile_url=summary.profile_url,
        )

    monkeypatch.setattr(shared.judger, "facial_judge_batch", facial)
    monkeypatch.setattr(shared.judger, "full_judge", full)
    arm_dir = tmp_path / "private-worker"
    assert (
        shadow_replay._run_matrix_worker(
            spec,
            "standard-high",
            arm_dir,
            block_index=0,
            capture_private_prose=True,
        )
        == 0
    )
    assert (
        shadow_replay._run_matrix_worker(
            spec,
            "standard-high",
            arm_dir,
            block_index=1,
            capture_private_prose=True,
        )
        == 0
    )

    private_path = arm_dir / shadow_replay._PRIVATE_PROSE_FILENAME
    assert private_path.stat().st_mode & 0o777 == 0o600
    rows = [json.loads(line) for line in private_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["decision"] == "SAVE"
    assert rows[0]["stage"] == "full"
    assert "private save rationale" in rows[0]["rationale"]
    assert "facial rationale" not in private_path.read_text()
    for ordinary in ("calls.jsonl", "execution-units.jsonl"):
        rendered = (arm_dir / ordinary).read_text()
        assert "private save rationale" not in rendered
        assert "Private Person" not in rendered

    tampered = dict(rows[0])
    tampered["rationale"] = "tampered"
    private_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    private_path.chmod(0o600)
    with pytest.raises(
        shadow_replay.MatrixValidationError,
        match="private prose.*hash",
    ):
        shadow_replay._validate_matrix_worker_prefix(
            arm_dir,
            arm_id="standard-high",
            plans=spec.call_plans["standard-high"],
            completed_units=json.loads(
                (arm_dir / "arm-metadata.json").read_text()
            )["execution_unit_order"],
            experiment_id=spec.experiment_id,
            manifest_hash=spec.manifest_hash,
            capture_private_prose=True,
        )


def test_private_prose_writer_refuses_symlink_and_non_private_mode(tmp_path: Path):
    target = tmp_path / "target.jsonl"
    target.write_text("", encoding="utf-8")
    target.chmod(0o600)
    symlink = tmp_path / shadow_replay._PRIVATE_PROSE_FILENAME
    symlink.symlink_to(target)
    with pytest.raises(shadow_replay.MatrixValidationError, match="symlink"):
        shadow_replay._append_private_prose_row(symlink, {"value": 1})

    symlink.unlink()
    symlink.write_text("", encoding="utf-8")
    symlink.chmod(0o644)
    with pytest.raises(shadow_replay.MatrixValidationError, match="0600"):
        shadow_replay._append_private_prose_row(symlink, {"value": 1})


def _allocator_checkpoint(
    sequence: int,
    *,
    root_id: int = 1,
    page: int = 1,
    selected_root_id: int = 1,
    valid: bool = True,
    off_policy: bool = False,
) -> dict:
    return {
        "event": "page_allocator_shadow_checkpoint",
        "checkpoint_index": sequence,
        "root_string_id": root_id,
        "page": page,
        "observation": {
            "root_string_id": root_id,
            "variant_id": "root",
            "page": page,
            "slots": 20,
            "valid": valid,
            "invalid_reasons": [] if valid else ["technical_interruption"],
            "off_policy": off_policy,
            "extracted": 20,
            "full_expected": 3,
            "full_settled": 3,
            "priority": 1,
            "standard": 2,
            "outreach": 3,
            "break_reason": "",
            "technical_interruption": not valid,
        },
        "verdict": {
            "current_root_id": root_id,
            "selected_root_id": selected_root_id,
        },
        "shadow_diverged": False,
    }


def test_page_allocator_mode_defaults_off_and_accepts_active(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("LINKEDIN_PAGE_ALLOCATOR_MODE", raising=False)
    assert shared_config._env_choice(
        "LINKEDIN_PAGE_ALLOCATOR_MODE", "off", {"active", "off", "shadow"}
    ) == "off"
    monkeypatch.setenv("LINKEDIN_PAGE_ALLOCATOR_MODE", " SHADOW ")
    assert shared_config._env_choice(
        "LINKEDIN_PAGE_ALLOCATOR_MODE", "off", {"active", "off", "shadow"}
    ) == "shadow"
    monkeypatch.setenv("LINKEDIN_PAGE_ALLOCATOR_MODE", " ACTIVE ")
    assert shared_config._env_choice(
        "LINKEDIN_PAGE_ALLOCATOR_MODE", "off", {"active", "off", "shadow"}
    ) == "active"
    monkeypatch.setenv("LINKEDIN_PAGE_ALLOCATOR_MODE", "unknown")
    with pytest.raises(ValueError, match="must be one of: active, off, shadow"):
        shared_config._env_choice(
            "LINKEDIN_PAGE_ALLOCATOR_MODE", "off", {"active", "off", "shadow"}
        )


def test_allocator_replay_keeps_page_when_divergence_occurs_after_observation():
    first = _allocator_checkpoint(1, selected_root_id=2)
    first["shadow_diverged"] = False
    first["divergence_after_observation"] = True
    later = _allocator_checkpoint(2, root_id=2, page=1, off_policy=True)

    report = shadow_replay.summarize_page_allocator_replay([first, later])

    assert report["currency_totals"] == {"n": 20, "p": 1, "e": 3}
    assert report["valid_on_policy_pages"] == [
        {"root_string_id": 1, "page": 1}
    ]
    assert report["first_divergence"] == {"root_string_id": 1, "page": 1}
    assert report["off_policy_pages_excluded"] == 1


def test_allocator_replay_latches_divergence_on_exhaustion_event():
    exhaustion = {
        "event": "page_allocator_shadow_exhaustion",
        "checkpoint_index": 1,
        "root_string_id": 4,
        "page": 3,
        "shadow_diverged": True,
        "verdict": {
            "action": "finish",
            "current_root_id": 4,
            "selected_root_id": None,
        },
    }

    report = shadow_replay.summarize_page_allocator_replay([exhaustion])

    assert report["first_divergence"] == {
        "root_string_id": 4,
        "page": 3,
    }


def test_allocator_replay_fails_closed_on_forged_validity():
    forged = _allocator_checkpoint(1)
    forged["observation"].update(
        {
            "full_expected": 10,
            "full_settled": 0,
            "outreach": 3,
        }
    )

    report = shadow_replay.summarize_page_allocator_replay([forged])

    assert report["evaluable"] is False
    assert report["currency_totals"] == {"n": 0, "p": 0, "e": 0}
    assert report["poison"] == {
        "line": 1,
        "reason": "observation_validity_mismatch",
    }


@pytest.mark.parametrize("event_kind", ["checkpoint", "exhaustion"])
def test_allocator_replay_fails_closed_on_cross_field_identity_mismatch(
    event_kind: str,
):
    if event_kind == "checkpoint":
        event = _allocator_checkpoint(1)
        event["verdict"]["current_root_id"] = 2
    else:
        event = {
            "event": "page_allocator_shadow_exhaustion",
            "checkpoint_index": 1,
            "root_string_id": 4,
            "page": 3,
            "shadow_diverged": False,
            "verdict": {
                "action": "finish",
                "current_root_id": 5,
                "selected_root_id": None,
            },
        }

    report = shadow_replay.summarize_page_allocator_replay([event])

    assert report["evaluable"] is False
    assert report["currency_totals"] == {"n": 0, "p": 0, "e": 0}
    assert report["poison"] == {
        "line": 1,
        "reason": "checkpoint_identity_mismatch",
    }


def test_allocator_replay_fails_closed_on_durable_shadow_poison():
    event = {
        "event": "page_allocator_shadow_poison",
        "mode": "shadow",
        "root_string_id": 4,
        "page": 2,
        "analysis_evaluable": False,
        "poison_reason": "checkpoint_apply:ValueError",
    }

    report = shadow_replay.summarize_page_allocator_replay([event])

    assert report["evaluable"] is False
    assert report["currency_totals"] == {"n": 0, "p": 0, "e": 0}
    assert report["poison"] == {
        "line": 1,
        "reason": "allocator_shadow_poison:checkpoint_apply:ValueError",
        "root_string_id": 4,
        "page": 2,
    }


@pytest.mark.parametrize("checkpoint_index", [None, 5])
def test_allocator_replay_requires_sequence_starting_at_one(
    checkpoint_index: int | None,
):
    event = _allocator_checkpoint(1)
    if checkpoint_index is None:
        event.pop("checkpoint_index")
    else:
        event["checkpoint_index"] = checkpoint_index

    report = shadow_replay.summarize_page_allocator_replay([event])

    assert report["evaluable"] is False
    assert report["currency_totals"] == {"n": 0, "p": 0, "e": 0}
    assert report["poison"] == {
        "line": 1,
        "reason": "checkpoint_sequence_gap",
    }


def test_allocator_log_cli_is_provider_free_and_reports_only_valid_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    events = [
        _allocator_checkpoint(1),
        _allocator_checkpoint(2, page=2, selected_root_id=2, valid=False),
        _allocator_checkpoint(3, page=3, selected_root_id=2),
        _allocator_checkpoint(4, root_id=2, selected_root_id=2),
    ]
    path = tmp_path / "run_log.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    for name in (
        "run_preflight",
        "run_formation",
        "run_judge",
        "_load_shadow_report_module",
    ):
        monkeypatch.setattr(
            shadow_replay,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"provider replay path called: {_name}"
            ),
        )
    monkeypatch.setattr(
        sys, "argv", ["shadow_replay.py", "--page-allocator-log", str(path)]
    )

    assert shadow_replay.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["currency_totals"] == {"n": 20, "p": 1, "e": 3}
    assert report["valid_on_policy_pages"] == [
        {"root_string_id": 1, "page": 1}
    ]
    assert report["invalid_pages_excluded"] == 1
    assert report["first_divergence"] == {"root_string_id": 1, "page": 3}
    assert report["off_policy_pages_excluded"] == 2
    assert report["post_divergence_on_policy_labels_ignored"] == 2
    assert report["claims"] == {
        "unobserved_sibling_yield": False,
        "fixed_uplift_percentage": False,
    }
    assert "projected_uplift" not in report


@pytest.mark.parametrize(
    ("lines", "reason"),
    [
        (["{not json\n"], "malformed_json"),
        (
            [
                json.dumps(_allocator_checkpoint(1)) + "\n",
                json.dumps(_allocator_checkpoint(3, page=2)) + "\n",
            ],
            "checkpoint_sequence_gap",
        ),
        (
            [
                json.dumps(_allocator_checkpoint(1)) + "\n",
                json.dumps({"event": "page_observation_gap"}) + "\n",
                json.dumps(_allocator_checkpoint(2, page=2)) + "\n",
            ],
            "page_observation_gap",
        ),
    ],
)
def test_allocator_log_cli_fails_closed_on_malformed_or_gapped_evidence(
    lines: list[str],
    reason: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    path = tmp_path / "run_log.jsonl"
    path.write_text("".join(lines))
    monkeypatch.setattr(
        sys, "argv", ["shadow_replay.py", "--page-allocator-log", str(path)]
    )

    assert shadow_replay.main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report["evaluable"] is False
    assert report["poison"]["reason"] == reason
