"""Tests for LLM cost attribution contracts (D5).

Pins:
- Usage records include lane_id/variant_id/stage when lane_context is provided.
- Per-lane cost rollup works from usage records.
- Unknown Fireworks cache splits stay nullable rather than becoming measured zero.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from shared.llm_usage import (
    MODEL_RATE_TABLE_USD_PER_MTOKEN,
    estimate_usage_cost_usd,
    llm_usage_session,
    record_llm_usage,
)


def test_lane_context_flows_into_usage_record(tmp_path: Path):
    log_path = tmp_path / "usage.jsonl"
    usage_context = {
        "lane_id": "ml-infra",
        "variant_id": "v1",
        "stage": "facial",
    }
    with llm_usage_session(log_path, brief_id="test-brief"):
        record_llm_usage(
            provider="anthropic",
            model="claude-opus",
            usage={"input_tokens": 1000, "output_tokens": 200},
            usage_context=usage_context,
        )
    records = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
    assert len(records) == 1
    r = records[0]
    assert r["lane_id"] == "ml-infra"
    assert r["variant_id"] == "v1"
    assert r["stage"] == "facial"
    assert r["brief_id"] == "test-brief"
    receipt = r["receipt"]
    assert receipt["receipt_type"] == "llm_call"
    assert receipt["stage"] == "llm:facial"
    assert receipt["actual_status"] == "ok"
    assert receipt["input_hash"].startswith("sha256:")
    assert receipt["actual_detail"]["input_tokens"] == 1000
    assert receipt["actual_detail"]["output_tokens"] == 200
    assert usage_context["_llm_receipt"]["receipt_id"] == receipt["receipt_id"]


def test_llm_usage_without_jsonl_session_still_returns_receipt_in_context():
    usage_context = {"stage": "chief_of_staff"}
    record_llm_usage(
        provider="anthropic",
        model="claude-opus",
        usage={"input_tokens": 20, "output_tokens": 5},
        usage_context=usage_context,
    )

    receipt = usage_context["_llm_receipt"]
    assert receipt["receipt_type"] == "llm_call"
    assert receipt["stage"] == "llm:chief_of_staff"
    assert receipt["actual_status"] == "ok"
    assert receipt["actual_detail"]["input_tokens"] == 20


# ---------------------------------------------------------------------------
# P10 actuate #4: created_at epoch pinning (shared/llm_usage.py, formerly
# hardcoded "1970-01-01T00:00:00+00:00" on every LLM receipt). Fix: stamp
# real time; receipt_id uniqueness for two calls with byte-identical content
# now comes from the real timestamp differing, not from a suppressed/pinned
# field forcing artificial collisions/non-collisions.
# ---------------------------------------------------------------------------


def test_llm_receipt_created_at_is_real_time_not_epoch_pinned():
    from datetime import datetime, timezone

    before = datetime.now(timezone.utc)
    usage_context = {"stage": "facial"}
    record_llm_usage(
        provider="anthropic",
        model="claude-opus",
        usage={"input_tokens": 10, "output_tokens": 2},
        usage_context=usage_context,
    )
    after = datetime.now(timezone.utc)

    receipt = usage_context["_llm_receipt"]
    assert receipt["created_at"] != "1970-01-01T00:00:00+00:00"
    stamped = datetime.fromisoformat(receipt["created_at"])
    assert before <= stamped <= after


def test_llm_receipt_ids_differ_across_calls_with_identical_content():
    """Two calls with byte-identical provider/model/usage/request/context still

    get distinct receipt_ids once real time is stamped — uniqueness comes
    from the real created_at, not from suppressing it to a fixed epoch.
    """
    import time

    payload = dict(
        provider="anthropic",
        model="claude-opus",
        usage={"input_tokens": 10, "output_tokens": 2},
    )

    ctx_a: dict = {"stage": "facial"}
    record_llm_usage(usage_context=ctx_a, **payload)
    time.sleep(0.01)
    ctx_b: dict = {"stage": "facial"}
    record_llm_usage(usage_context=ctx_b, **payload)

    receipt_a = ctx_a["_llm_receipt"]
    receipt_b = ctx_b["_llm_receipt"]
    assert receipt_a["created_at"] != receipt_b["created_at"]
    assert receipt_a["receipt_id"] != receipt_b["receipt_id"]


def test_production_llm_call_sites_pass_receipt_usage_context():
    """All production LLM wrapper calls must expose the LLM receipt hook."""

    production_roots = [
        Path("cloris"),
        Path("designer"),
        Path("exec_search"),
        Path("github"),
        Path("linkedin"),
        Path("market_intelligence"),
        Path("researcher"),
        Path("shared"),
    ]
    receipt_backed_call_names = {
        "cheap_llm",
        "facial_llm",
        "opus_llm",
        "opus_llm_cached",
        "opus_llm_cached_stream",
    }
    missing: list[str] = []
    for root in production_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.name == "llm_clients.py":
                continue
            missing.extend(_missing_usage_context_calls(path, receipt_backed_call_names))

    assert missing == []


def _missing_usage_context_calls(
    path: Path,
    call_names: set[str],
) -> list[str]:
    missing: list[str] = []
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:
        raise AssertionError(f"{path} failed to parse: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        if name not in call_names:
            continue
        if not any(keyword.arg == "usage_context" for keyword in node.keywords):
            missing.append(f"{path}:{node.lineno}:{name}")
    return missing


def test_cost_estimate_with_cache_tokens():
    cost, source = estimate_usage_cost_usd(
        model="claude-opus",
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=100,
    )
    assert cost is not None
    assert cost > 0
    assert source == "exact"


def test_cost_estimate_zero_cache():
    cost, source = estimate_usage_cost_usd(
        model="claude-opus",
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    assert cost is not None
    assert cost > 0


def test_fireworks_glm_fast_rate_is_exact_and_includes_cached_input():
    rates = MODEL_RATE_TABLE_USD_PER_MTOKEN[
        "accounts/fireworks/routers/glm-5p2-fast"
    ]
    assert rates == {
        "input": 2.10,
        "output": 6.60,
        "cache_read_input": 0.21,
    }
    cost, source = estimate_usage_cost_usd(
        model="accounts/fireworks/routers/glm-5p2-fast",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert source == "exact"
    assert cost == 2.10 + 6.60 + 0.21


def test_minimax_m3_rate_is_exact_and_includes_cached_input():
    rates = MODEL_RATE_TABLE_USD_PER_MTOKEN["MiniMax-M3"]
    assert rates == {
        "input": 0.30,
        "output": 1.20,
        "cache_read_input": 0.06,
    }
    cost, source = estimate_usage_cost_usd(
        model="MiniMax-M3",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert source == "exact"
    assert cost == 0.30 + 1.20 + 0.06


def test_minimax_usage_dict_subtracts_cached_tokens_once():
    from types import SimpleNamespace

    from shared.llm_usage import minimax_usage_dict

    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=800),
        )
    )
    assert minimax_usage_dict(response) == {
        "input_tokens": 200,
        "output_tokens": 50,
        "cache_read_input_tokens": 800,
        "cache_creation_input_tokens": 0,
    }


def test_explicit_unavailable_usage_records_null_tokens_and_no_cost(tmp_path: Path):
    log_path = tmp_path / "usage.jsonl"
    with llm_usage_session(log_path):
        record_llm_usage(
            provider="fireworks",
            model="accounts/fireworks/models/glm-5p2",
            usage={},
            usage_status="unavailable",
            actual_status="error",
        )

    row = json.loads(log_path.read_text())
    assert row["usage_status"] == "unavailable"
    assert row["cost_completeness"] == "unavailable"
    assert row["input_tokens"] is None
    assert row["output_tokens"] is None
    assert row["estimated_cost_usd"] is None


def test_per_lane_cost_rollup(tmp_path: Path):
    log_path = tmp_path / "usage.jsonl"
    with llm_usage_session(log_path):
        for lane_id, stage, inp, out in [
            ("ml-infra", "facial", 500, 100),
            ("ml-infra", "full_eval", 2000, 800),
            ("platform", "facial", 500, 100),
        ]:
            record_llm_usage(
                provider="anthropic",
                model="claude-opus",
                usage={"input_tokens": inp, "output_tokens": out},
                usage_context={"lane_id": lane_id, "stage": stage},
            )

    records = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
    by_lane: dict[str, float] = {}
    for r in records:
        lid = r.get("lane_id", "unknown")
        by_lane[lid] = by_lane.get(lid, 0) + (r.get("estimated_cost_usd") or 0)
    assert "ml-infra" in by_lane
    assert "platform" in by_lane
    assert by_lane["ml-infra"] > by_lane["platform"]


def test_unknown_model_cost_is_none():
    cost, source = estimate_usage_cost_usd(
        model="unknown-model",
        input_tokens=1000,
        output_tokens=500,
    )
    assert cost is None
    assert source == "unknown"


# ---------------------------------------------------------------------------
# P4.1: corrected claude-opus rate table (verified 2026-07-02 against Anthropic
# docs; Opus 4.x = $5/$25 per MTok in/out, cache write 1.25x, cache read 0.1x)
# ---------------------------------------------------------------------------

def test_claude_opus_rate_table_values_are_corrected():
    """Dict-level pin: claude-opus row must match Anthropic's published Opus
    4.x pricing, not the stale 15/75 (+18.75/1.5) rates."""
    rates = MODEL_RATE_TABLE_USD_PER_MTOKEN["claude-opus"]
    assert rates["input"] == 5.0
    assert rates["output"] == 25.0
    assert rates["cache_creation_input"] == 6.25
    assert rates["cache_read_input"] == 0.50


def test_claude_opus_cost_computation_uses_corrected_rates_end_to_end():
    """End-to-end through the public cost function (not just dict access):
    1M tokens of each kind should cost exactly the corrected per-MTok rates
    summed, proving estimate_usage_cost_usd actually reads the updated table
    rather than a cached/hardcoded value."""
    cost, source = estimate_usage_cost_usd(
        model="claude-opus",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    assert source == "exact"
    assert cost == 5.0 + 25.0 + 0.50 + 6.25


def test_claude_opus_prefix_match_still_resolves_real_client_model_string():
    """config.OPUS_MODEL_NAME defaults to 'claude-opus-4-6' — the actual
    string clients pass to record_llm_usage/estimate_usage_cost_usd. The
    'claude-opus' key must still prefix-match it after the rate edit."""
    cost, source = estimate_usage_cost_usd(
        model="claude-opus-4-6",
        input_tokens=1_000_000,
        output_tokens=0,
    )
    assert source == "prefix:claude-opus"
    assert cost == 5.0


def test_removed_dead_rate_rows_no_longer_resolve():
    """openai/gpt-5.2 and xai/grok-4-1-fast-non-reasoning have no live
    caller (verified: grep repo-wide — the former model_routing.py dead
    pin never fed a real LLM call; facial_llm() actually uses
    config.FACIAL_MODEL_NAME/OPUS_MODEL_NAME).
    Deleting these rows must produce the same miss behavior as any other
    unknown model: (None, "unknown")."""
    for dead_model in ("openai/gpt-5.2", "xai/grok-4-1-fast-non-reasoning"):
        cost, source = estimate_usage_cost_usd(
            model=dead_model,
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost is None
        assert source == "unknown"


def test_claude_fable_rate_table_values_are_pinned():
    """claude-fable row: the live .env's strategy tier (STRATEGY_MODEL_NAME=claude-fable-5);
    config default remains OPUS_MODEL_NAME. Verified 2026-07-31."""
    rates = MODEL_RATE_TABLE_USD_PER_MTOKEN["claude-fable"]
    assert rates == {
        "input": 10.0,
        "output": 50.0,
        "cache_creation_input": 12.50,
        "cache_read_input": 1.00,
    }


def test_claude_fable_cost_computation_uses_table_rates_end_to_end():
    cost, source = estimate_usage_cost_usd(
        model="claude-fable-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    assert source == "prefix:claude-fable"
    assert cost == 10.0 + 50.0 + 1.00 + 12.50


def test_claude_fable_prefix_match_resolves_versioned_client_model_string():
    cost, source = estimate_usage_cost_usd(
        model="claude-fable-5",
        input_tokens=1_000_000,
        output_tokens=0,
    )
    assert source == "prefix:claude-fable"
    assert cost == 10.0


def test_claude_haiku_production_model_string_resolves():
    """P4.1 correction: the LIVE .env runs CHEAP_MODEL_PROVIDER=anthropic with
    CHEAP_MODEL_NAME=claude-haiku-4-5-20251001 — the rate table must price it,
    or every production cheap-tier call is silently excluded from cost_usd.
    (The original P4.1 pass verified callers against config DEFAULTS, not the
    .env override — this locks the deployed string.)"""
    from shared.llm_usage import estimate_usage_cost_usd

    cost, match = estimate_usage_cost_usd(
        model="claude-haiku-4-5-20251001",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert match != "unknown"
    assert cost == 6.0  # $1 input + $5 output per MTok


# ---------------------------------------------------------------------------
# fireworks_shadow_usage_dict — GLM-5.2 shadow-judge cache-field mapping
# (Task B). Field name confirmed against Fireworks' public API reference:
# https://docs.fireworks.ai/api-reference/post-chatcompletions documents
# usage.prompt_tokens_details.cached_tokens (Chat Completions shape).
# ---------------------------------------------------------------------------


def test_fireworks_shadow_usage_dict_subtracts_cached_from_input_inclusive_convention():
    from types import SimpleNamespace

    from shared.llm_usage import fireworks_shadow_usage_dict

    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=800),
        )
    )
    usage = fireworks_shadow_usage_dict(response)
    # Inclusive: cached_tokens is a SUBSET of prompt_tokens, so billable
    # input is prompt_tokens - cached_tokens, not the raw prompt_tokens.
    assert usage["input_tokens"] == 200
    assert usage["cache_read_input_tokens"] == 800
    assert usage["output_tokens"] == 50
    assert usage["cache_creation_input_tokens"] == 0


def test_fireworks_shadow_usage_dict_no_cache_details_is_partial_not_fake_zero():
    from types import SimpleNamespace

    from shared.llm_usage import fireworks_shadow_usage_dict

    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=50)
    )
    usage = fireworks_shadow_usage_dict(response)
    assert usage["input_tokens"] is None
    assert usage["cache_read_input_tokens"] is None
    assert usage["output_tokens"] == 50


def test_fireworks_shadow_usage_dict_accepts_dict_shaped_prompt_tokens_details():
    """Some SDK/serialization paths surface usage as plain dicts rather
    than attribute objects — the dict fallback must work identically."""
    from types import SimpleNamespace

    from shared.llm_usage import fireworks_shadow_usage_dict

    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=400,
            completion_tokens=10,
            prompt_tokens_details={"cached_tokens": 300},
        )
    )
    usage = fireworks_shadow_usage_dict(response)
    assert usage["input_tokens"] == 100
    assert usage["cache_read_input_tokens"] == 300


def test_fireworks_shadow_usage_dict_missing_usage_returns_all_null():
    from types import SimpleNamespace

    from shared.llm_usage import fireworks_shadow_usage_dict

    usage = fireworks_shadow_usage_dict(SimpleNamespace())
    assert usage == {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
    }


def test_fireworks_shadow_usage_dict_null_cached_tokens_stays_unknown():
    """A null cache split cannot honestly be reported as a measured miss."""
    from types import SimpleNamespace

    from shared.llm_usage import fireworks_shadow_usage_dict

    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=None),
        )
    )
    usage = fireworks_shadow_usage_dict(response)
    assert usage["input_tokens"] is None
    assert usage["cache_read_input_tokens"] is None
    assert usage["output_tokens"] == 50


def test_fireworks_shadow_usage_dict_recovers_cache_split_from_raw_header():
    from types import SimpleNamespace

    from shared.llm_usage import fireworks_shadow_usage_dict

    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=None),
        )
    )
    usage = fireworks_shadow_usage_dict(
        response,
        provider_headers={"fireworks-cached-prompt-tokens": "800"},
    )
    assert usage["input_tokens"] == 200
    assert usage["cache_read_input_tokens"] == 800


def test_fireworks_shadow_usage_dict_floors_negative_input_at_zero():
    """Opus-review follow-up: a provider reporting cached > prompt (buggy or
    exclusive-convention drift) must floor billable input at 0, never
    negative-price."""
    from types import SimpleNamespace

    from shared.llm_usage import fireworks_shadow_usage_dict

    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=500,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=800),
        )
    )
    usage = fireworks_shadow_usage_dict(response)
    assert usage["input_tokens"] == 0
    assert usage["cache_read_input_tokens"] == 800
