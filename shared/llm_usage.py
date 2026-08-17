"""Helpers for token/cost logging across LLM-backed workflows."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from shared.receipts import ReceiptStatus, build_receipt
from shared.storage import append_jsonl


_USAGE_LOG_PATH: ContextVar[Path | None] = ContextVar("llm_usage_log_path", default=None)
_USAGE_BASE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "llm_usage_base_context",
    default={},
)

LLM_USAGE_SCHEMA_VERSION = "llm-usage-v2"
LLM_ATTEMPT_SCHEMA_VERSION = "llm-attempt-v1"
_USAGE_STATUSES = frozenset({"measured", "partial", "unavailable"})
_ATTEMPT_CONTEXT_KEYS = frozenset(
    {
        "arm_id",
        "batch_count",
        "batch_id",
        "batch_index",
        "batch_number",
        "batch_size",
        "batch_slot",
        "batch_start",
        "batch_stop",
        "brief_id",
        "cache_phase",
        "candidate_count",
        "contract_mode",
        "contract_version",
        "execution_unit_id",
        "experiment_id",
        "judgment_contract_mode",
        "judgment_contract_version",
        "lane_id",
        "module",
        "offline_replay",
        "page",
        "parent_logical_call_id",
        "run_id",
        "source",
        "stage",
        "string_id",
        "variant_id",
    }
)
_FIREWORKS_USAGE_CONTEXT_KEYS = _ATTEMPT_CONTEXT_KEYS | frozenset(
    {
        "fallback_index",
        "fallback_reason",
        "logical_call_id",
        "parent_logical_call_id",
        "retry_reason",
    }
)
_ATTEMPT_METADATA_KEYS = frozenset(
    {
        "attempt_timeout_seconds",
        "cooldown_wait_ms",
        "error_type",
        "failure_reason",
        "finish_reason",
        "http_status",
        "latency_ms",
        "policy_stage",
        "prompt_cache_key",
        "provider_headers",
        "provider_request_id",
        "reasoning_effort_effective",
        "reasoning_effort_requested",
        "response_transport",
        "response_id",
        "retry_decision",
        "retry_delay_seconds",
        "retry_delay_source",
        "serving_path",
        "stable_prefix_sha256",
        "stream_content_delta_count",
        "stream_event_count",
        "stream_first_content_ms",
        "stream_first_event_ms",
        "stream_reasoning_delta_count",
        "timeout_phase",
        "tool_schema_sha256",
        "total_deadline_seconds",
        "transport_timeout_type",
    }
)
_ATTEMPT_PROVIDER_HEADER_KEYS = frozenset(
    {
        "fireworks-cached-prompt-tokens",
        "fireworks-prompt-tokens",
        "fireworks-server-processing-time",
        "fireworks-server-time-to-first-token",
        "retry-after",
        "x-ratelimit-limit-tokens-cache-adjusted-prompt",
        "x-ratelimit-limit-tokens-generated",
        "x-ratelimit-limit-tokens-prompt",
        "x-request-id",
    }
)


# Price verification (P4.1): claude-opus row verified 2026-07-02 against
# Anthropic docs via the claude-api skill (model table cached 2026-06-24).
# Opus 4.x = $5.00 input / $25.00 output per MTok; cache write (5-min TTL) =
# 1.25x input = $6.25; cache read = 0.1x input = $0.50. The previous rates
# here (15/75, +18.75/1.5) were stale/incorrect.
#
# gpt-4o-mini and gemini-2.0-flash rows are UNCHANGED and NOT re-verified on
# 2026-07-02 — the claude-api skill covers Anthropic pricing only. OpenAI/
# Google rates below are the prior snapshot values pending operator sign-off
# (spec §14).
#
# Removed 2026-07-02: openai/gpt-5.2 and xai/grok-4-1-fast-non-reasoning rows.
# Neither model string is ever passed to record_llm_usage/estimate_usage_cost_usd
# by any production caller (verified via repo-wide grep). The only reference to
# the xai model string was shared/model_routing.py's ModelRoutePolicy.facial_model
# dataclass default, which that module's own docstring documents as "Pure data —
# no LLM calls"; the actual facial-triage call path (shared/llm_clients.py
# facial_llm) uses config.FACIAL_MODEL_NAME, which defaults to config.OPUS_MODEL_NAME
# ("claude-opus-4-6"), not the routing policy's facial_model string.
MODEL_RATE_TABLE_USD_PER_MTOKEN: dict[str, dict[str, float]] = {
    "claude-opus": {
        "input": 5.0,
        "output": 25.0,
        "cache_creation_input": 6.25,
        "cache_read_input": 0.50,
    },
    # Claude Fable — the live .env's strategy tier (STRATEGY_MODEL_NAME=claude-fable-5);
    # config default remains OPUS_MODEL_NAME. Verified 2026-07-31 against Anthropic docs: $10.00/$50.00 per MTok;
    # cache write 1.25x input = $12.50; cache read 0.1x input = $1.00.
    "claude-fable": {
        "input": 10.0,
        "output": 50.0,
        "cache_creation_input": 12.50,
        "cache_read_input": 1.00,
    },
    # Claude Haiku 4.5 — the LIVE .env's cheap tier (CHEAP_MODEL_PROVIDER=
    # anthropic, CHEAP_MODEL_NAME=claude-haiku-4-5-20251001). Missing from the
    # original P4.1 pass, which verified callers against config DEFAULTS
    # (gpt-4o-mini) rather than the deployed .env override — every production
    # cheap-tier call priced as "unknown" and fell out of cost_usd sums.
    # Verified 2026-07-03 against Anthropic docs via the claude-api skill:
    # $1.00/$5.00 per MTok; cache write 1.25x = $1.25; cache read 0.1x = $0.10.
    "claude-haiku": {
        "input": 1.0,
        "output": 5.0,
        "cache_creation_input": 1.25,
        "cache_read_input": 0.10,
    },
    # OpenAI GPT-4o-mini (platform pricing snapshot; aligns with CHEAP_MODEL_NAME default).
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    # Gemini 2.0 Flash (google.genai cheap path in llm_clients._call_google).
    "gemini-2.0-flash": {
        "input": 0.10,
        "output": 0.40,
    },
    # GLM-5.2 served by Fireworks (shadow-judge evaluation seam,
    # shared/llm_clients.shadow_facial_llm). Keyed on the full
    # "accounts/fireworks/models/glm-5p2" slug via the exact-match branch of
    # _lookup_model_rates; the prefix-match branch also covers any future
    # "-fp8"/dated suffix on the same base slug. Verified 2026-07-03 against
    # the public fireworks.ai model-library pricing card
    # (https://fireworks.ai/models/fireworks/glm-5p2): $1.40 input / $4.40
    # output per MTok, cached input $0.14/MTok. Not "unverified" — this is a
    # public pricing page read directly, not a prior research recollection.
    "accounts/fireworks/models/glm-5p2": {
        "input": 1.40,
        "output": 4.40,
        "cache_read_input": 0.14,
    },
    # GLM-5.2 Fast is the same model weights on Fireworks' latency-optimized
    # router. Verified 2026-07-11: $2.10 input / $6.60 output / $0.21 cached
    # input per MTok.
    "accounts/fireworks/routers/glm-5p2-fast": {
        "input": 2.10,
        "output": 6.60,
        "cache_read_input": 0.21,
    },
    # MiniMax M3 direct standard service, <=512K input. Verified 2026-07-14
    # against MiniMax's OpenAI-compatible API/pricing documentation. Extraction
    # calls are far below the long-context price boundary.
    "MiniMax-M3": {
        "input": 0.30,
        "output": 1.20,
        "cache_read_input": 0.06,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _lookup_model_rates(model: str) -> tuple[dict[str, float] | None, str]:
    normalized = _normalize_text(model)
    if not normalized:
        return None, "unknown"
    exact = MODEL_RATE_TABLE_USD_PER_MTOKEN.get(normalized)
    if exact:
        return exact, "exact"
    lowered = normalized.lower()
    for key, rates in MODEL_RATE_TABLE_USD_PER_MTOKEN.items():
        if lowered.startswith(key.lower()):
            return rates, f"prefix:{key}"
    return None, "unknown"


def estimate_usage_cost_usd(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> tuple[float | None, str]:
    rates, rate_source = _lookup_model_rates(model)
    if not rates:
        return None, rate_source
    cost = 0.0
    cost += (input_tokens / 1_000_000.0) * rates.get("input", 0.0)
    cost += (output_tokens / 1_000_000.0) * rates.get("output", 0.0)
    cost += (cache_read_input_tokens / 1_000_000.0) * rates.get(
        "cache_read_input",
        0.0,
    )
    cost += (cache_creation_input_tokens / 1_000_000.0) * rates.get(
        "cache_creation_input",
        0.0,
    )
    return round(cost, 6), rate_source


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _build_llm_receipt(
    *,
    provider: str,
    model: str,
    usage: dict[str, Any],
    request: dict[str, Any],
    usage_context: dict[str, Any],
    actual_status: ReceiptStatus | str,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_read_input_tokens: int | None,
    cache_creation_input_tokens: int | None,
    estimated_cost_usd: float | None,
    rate_source: str,
    usage_status: str,
    cost_completeness: str,
) -> dict[str, Any]:
    stage = str(usage_context.get("stage") or request.get("stage") or provider)
    actual_detail = {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "rate_source": rate_source,
        "usage_status": usage_status,
        "cost_completeness": cost_completeness,
    }
    if "error_type" in request or "error_message" in request:
        actual_detail["error"] = {
            "type": request.get("error_type"),
            "message": request.get("error_message"),
        }
    receipt = build_receipt(
        receipt_type="llm_call",
        stage=f"llm:{stage}",
        input_payload={
            "provider": provider,
            "model": model,
            "usage": _json_safe(usage),
            "request": _json_safe(request),
            "usage_context": _json_safe(usage_context),
        },
        actual_status=actual_status,
        intended_postcondition=(
            "LLM provider call returns token/cost telemetry or a typed error receipt"
        ),
        actual_detail=actual_detail,
        producer="shared.llm_usage",
        version_pins={"shared_llm_usage": "llm-receipts-v2"},
    )
    return receipt.to_dict()


def resolve_cost_log_run_id(
    log_path: str | Path | None,
    run_id: int | str | None,
) -> int | str | None:
    """Choose whether to scope a cost sum to ``run_id``.

    Once a log contains at least one row tagged with ``run_id``, scoped sums
    exclude legacy rows that lack the field. Until then, whole-file semantics
    are preserved so pre-migration rows still contribute to the active run.
    """

    if run_id is None:
        return None
    path = Path(log_path) if log_path else None
    if path is None or not path.exists():
        return run_id
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return run_id
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(record, dict) and record.get("run_id") is not None:
            return run_id
    return None


@contextmanager
def llm_usage_session(log_path: str | Path | None, **base_context: Any):
    path = Path(log_path) if log_path else None
    path_token = _USAGE_LOG_PATH.set(path)
    context_token = _USAGE_BASE_CONTEXT.set(dict(base_context))
    try:
        yield
    finally:
        _USAGE_LOG_PATH.reset(path_token)
        _USAGE_BASE_CONTEXT.reset(context_token)


def current_llm_usage_log_path() -> Path | None:
    return _USAGE_LOG_PATH.get()


def current_llm_attempt_log_path() -> Path | None:
    usage_path = current_llm_usage_log_path()
    if usage_path is None:
        return None
    return usage_path.parent / "llm-attempts.jsonl"


def _append_jsonl_fail_soft(path: Path, record: dict[str, Any]) -> None:
    """Diagnostic persistence must never influence a model decision."""

    try:
        append_jsonl(path, record)
    except Exception:
        pass


def _usage_values(
    usage: dict[str, Any],
    *,
    usage_status: str,
) -> tuple[int | None, int | None, int | None, int | None]:
    if usage_status == "unavailable":
        return None, None, None, None
    return (
        _safe_optional_int(usage.get("input_tokens")),
        _safe_optional_int(usage.get("output_tokens")),
        _safe_optional_int(usage.get("cache_read_input_tokens")),
        _safe_optional_int(usage.get("cache_creation_input_tokens")),
    )


def record_llm_attempt(
    *,
    provider: str,
    model: str,
    logical_call_id: str,
    attempt_number: int,
    max_attempts: int,
    status: str,
    usage: dict[str, Any] | None = None,
    usage_status: str = "unavailable",
    usage_context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one diagnostic attempt row beside the aggregate cost log.

    Attempt rows deliberately carry no cost field.  Cost readers continue to
    sum the single logical-call row in ``token-cost-log.jsonl`` and therefore
    cannot double-count retries.
    """

    path = current_llm_attempt_log_path()
    if path is None:
        return
    normalized_status = str(usage_status or "").strip().lower()
    if normalized_status not in _USAGE_STATUSES:
        normalized_status = "unavailable"
    usage_dict = dict(usage or {})
    input_tokens, output_tokens, cache_read, cache_creation = _usage_values(
        usage_dict,
        usage_status=normalized_status,
    )
    merged_context = dict(_USAGE_BASE_CONTEXT.get())
    merged_context.update(dict(usage_context or {}))
    safe_context = {
        key: _json_safe(value)
        for key, value in merged_context.items()
        if key in _ATTEMPT_CONTEXT_KEYS
    }
    safe_metadata = {
        key: _json_safe(value)
        for key, value in dict(metadata or {}).items()
        if key in _ATTEMPT_METADATA_KEYS and key != "provider_headers"
    }
    raw_headers = dict(metadata or {}).get("provider_headers")
    if isinstance(raw_headers, dict):
        normalized_headers = {
            str(key).lower(): str(value) for key, value in raw_headers.items()
        }
        safe_metadata["provider_headers"] = {
            key: normalized_headers[key]
            for key in sorted(_ATTEMPT_PROVIDER_HEADER_KEYS)
            if key in normalized_headers
        }
    row = {
        "schema_version": LLM_ATTEMPT_SCHEMA_VERSION,
        "timestamp": _utc_now(),
        "provider": str(provider),
        "model": str(model),
        "logical_call_id": str(logical_call_id),
        "attempt_number": int(attempt_number),
        "max_attempts": int(max_attempts),
        "status": str(status),
        "usage_status": normalized_status,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
        **safe_context,
        **safe_metadata,
    }
    _append_jsonl_fail_soft(path, row)


def record_llm_usage(
    *,
    provider: str,
    model: str,
    usage: dict[str, Any] | None = None,
    request: dict[str, Any] | None = None,
    usage_context: dict[str, Any] | None = None,
    actual_status: ReceiptStatus | str = ReceiptStatus.OK,
    usage_status: str | None = None,
    cost_completeness: str | None = None,
) -> None:
    """Record one LLM call's usage to the active session's JSONL sink.

    Phase 1 of Langfuse adoption: ALSO calls
    :func:`shared.observability.update_current_observation` so the
    in-flight ``@observe(as_type="generation")`` span on the LLM
    caller (``opus_llm`` / ``opus_llm_cached`` / ``facial_llm`` per
    ``shared/llm_clients.py``) gains the same cost / token / cache
    stats Langfuse's cost dashboard needs. The JSONL sink stays as
    the canonical record (``cost_rollup.py`` reads it); Langfuse
    becomes a parallel sink. Both must agree to within 1% per the
    Phase 1 verification's cost-source parity check.

    The ``update_current_observation`` call is a no-op when the
    Langfuse client is null / disabled / network-degraded — same
    posture as the keys-absent path. Existing JSONL semantics are
    byte-equivalent.

    Phase 2 relaxation: when no ``llm_usage_session`` is open (no
    JSONL log path), we still update Langfuse with cost telemetry.
    This recovers observability for every call that passes
    ``usage_context`` but runs outside a session (LinkedIn judger,
    Chief of Staff dispatch, search plans, Designer vision, etc.).
    JSONL still requires a session; Langfuse does not.
    """

    usage = dict(usage or {})
    request = dict(request or {})
    explicit_usage_status = usage_status is not None
    normalized_usage_status = str(usage_status or "measured").strip().lower()
    if normalized_usage_status not in _USAGE_STATUSES:
        raise ValueError(f"invalid usage_status: {usage_status!r}")
    caller_context = {
        key: value
        for key, value in dict(usage_context or {}).items()
        if key != "_llm_receipt"
    }
    log_path = _USAGE_LOG_PATH.get()
    merged_context = dict(_USAGE_BASE_CONTEXT.get()) if log_path else {}
    merged_context.update(caller_context)
    if str(provider).strip().lower() == "fireworks":
        # Fireworks judgment calls can carry candidate-rich lane_context
        # dictionaries. Aggregate usage, receipt hashes, and Langfuse metadata
        # retain only operational identifiers; raw candidate/profile/prompt or
        # reasoning fields are never persisted merely because a caller added
        # them to usage_context.
        merged_context = {
            key: value
            for key, value in merged_context.items()
            if key in _FIREWORKS_USAGE_CONTEXT_KEYS
        }

    if explicit_usage_status:
        (
            input_tokens,
            output_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
        ) = _usage_values(usage, usage_status=normalized_usage_status)
        if normalized_usage_status == "measured" and any(
            value is None
            for value in (
                input_tokens,
                output_tokens,
                cache_read_input_tokens,
                cache_creation_input_tokens,
            )
        ):
            normalized_usage_status = "partial"
    else:
        # Backward-compatible default for every existing provider caller.
        input_tokens = _safe_int(usage.get("input_tokens"))
        output_tokens = _safe_int(usage.get("output_tokens"))
        cache_read_input_tokens = _safe_int(usage.get("cache_read_input_tokens"))
        cache_creation_input_tokens = _safe_int(
            usage.get("cache_creation_input_tokens")
        )

    derived_completeness = {
        "measured": "complete",
        "partial": "lower_bound",
        "unavailable": "unavailable",
    }[normalized_usage_status]
    normalized_completeness = str(cost_completeness or derived_completeness)
    if normalized_completeness not in {"complete", "lower_bound", "unavailable"}:
        raise ValueError(f"invalid cost_completeness: {cost_completeness!r}")
    if normalized_usage_status == "unavailable":
        estimated_cost_usd = None
        _rates, rate_source = _lookup_model_rates(model)
    else:
        estimated_cost_usd, rate_source = estimate_usage_cost_usd(
            model=model,
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            cache_read_input_tokens=cache_read_input_tokens or 0,
            cache_creation_input_tokens=cache_creation_input_tokens or 0,
        )
    receipt = _build_llm_receipt(
        provider=provider,
        model=model,
        usage=usage,
        request=request,
        usage_context=merged_context,
        actual_status=actual_status,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        estimated_cost_usd=estimated_cost_usd,
        rate_source=rate_source,
        usage_status=normalized_usage_status,
        cost_completeness=normalized_completeness,
    )
    if usage_context is not None:
        usage_context["_llm_receipt"] = receipt

    if log_path:
        record = {
            "schema_version": LLM_USAGE_SCHEMA_VERSION,
            "timestamp": _utc_now(),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "usage_status": normalized_usage_status,
            "cost_completeness": normalized_completeness,
            "estimated_cost_usd": estimated_cost_usd,
            "rate_source": rate_source,
            "receipt": receipt,
            **merged_context,
            **request,
        }
        _append_jsonl_fail_soft(log_path, record)

    # Phase 1: parallel Langfuse sink. Lazy import keeps the JSONL
    # write path free from observability-layer coupling at module
    # import time. The helper is a no-op when no Langfuse client is
    # wired (keys absent / LANGFUSE_DISABLE=1 / SDK missing /
    # sticky-degraded post-network-failure), so the JSONL path stays
    # byte-equivalent to pre-Phase-1.
    try:
        from shared.observability import update_current_observation

        observation_usage: dict[str, int] = {}
        if input_tokens is not None:
            observation_usage["input"] = input_tokens
        if output_tokens is not None:
            observation_usage["output"] = output_tokens
        if input_tokens is not None and output_tokens is not None:
            observation_usage["total"] = input_tokens + output_tokens
        if cache_read_input_tokens is not None:
            observation_usage["cache_read_input_tokens"] = cache_read_input_tokens
        if cache_creation_input_tokens is not None:
            observation_usage["cache_creation_input_tokens"] = (
                cache_creation_input_tokens
            )

        update_current_observation(
            model=model,
            usage=observation_usage,
            metadata={
                "provider": provider,
                "estimated_cost_usd": estimated_cost_usd,
                "rate_source": rate_source,
                "usage_status": normalized_usage_status,
                "cost_completeness": normalized_completeness,
                "llm_receipt_id": receipt["receipt_id"],
                "llm_receipt_status": receipt["actual_status"],
                # merged_context carries caller attributes for Langfuse
                # filtering (Fireworks is restricted to the allowlist above).
                **{
                    k: v
                    for k, v in merged_context.items()
                    if isinstance(k, str) and not k.startswith("_")
                },
            },
        )
    except Exception:  # noqa: BLE001 — Langfuse path is fail-soft
        # The langfuse_client wrapper already self-degrades; this
        # outer guard catches anything else (an unexpected import
        # failure, etc.) so the JSONL sink path stays clean.
        pass


def anthropic_usage_dict(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage", None)
    return {
        "input_tokens": _safe_int(getattr(usage, "input_tokens", 0)),
        "output_tokens": _safe_int(getattr(usage, "output_tokens", 0)),
        "cache_read_input_tokens": _safe_int(
            getattr(usage, "cache_read_input_tokens", 0)
        ),
        "cache_creation_input_tokens": _safe_int(
            getattr(usage, "cache_creation_input_tokens", 0)
        ),
    }


def openai_usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    cached_tokens = 0
    if details is not None:
        cached_tokens = _safe_int(getattr(details, "cached_tokens", 0))
    if not cached_tokens and isinstance(details, dict):
        cached_tokens = _safe_int(details.get("cached_tokens", 0))
    return {
        "input_tokens": _safe_int(
            getattr(usage, "input_tokens", None)
            if usage is not None
            else 0,
            default=_safe_int(getattr(usage, "prompt_tokens", 0)),
        ),
        "output_tokens": _safe_int(
            getattr(usage, "output_tokens", None)
            if usage is not None
            else 0,
            default=_safe_int(getattr(usage, "completion_tokens", 0)),
        ),
        "cache_read_input_tokens": cached_tokens,
        "cache_creation_input_tokens": 0,
    }


def minimax_usage_dict(response: Any) -> dict[str, int | None]:
    """Normalize MiniMax's OpenAI-compatible usage without inventing cache misses.

    ``prompt_tokens`` is inclusive when ``cached_tokens`` is supplied, so the
    uncached input slice subtracts the cached slice exactly once. When MiniMax
    omits the cache detail, input/cache remain unknown and output still forms a
    truthful lower-bound receipt.
    """

    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
        }
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        details = getattr(usage, "input_tokens_details", None)
    cached_raw = (
        details.get("cached_tokens")
        if isinstance(details, dict)
        else getattr(details, "cached_tokens", None)
    )
    cached_tokens = _safe_optional_int(cached_raw)
    raw_prompt_tokens = _safe_optional_int(
        getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None))
    )
    output_tokens = _safe_optional_int(
        getattr(usage, "completion_tokens", getattr(usage, "output_tokens", None))
    )
    input_tokens = (
        max(raw_prompt_tokens - cached_tokens, 0)
        if raw_prompt_tokens is not None and cached_tokens is not None
        else None
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cached_tokens,
        "cache_creation_input_tokens": 0 if cached_tokens is not None else None,
    }


def fireworks_shadow_usage_dict(
    response: Any,
    *,
    provider_headers: Mapping[str, Any] | None = None,
) -> dict[str, int | None]:
    """Extract normalized token counts from a Fireworks (OpenAI-compatible
    Chat Completions) response, WITH automatic-prefix-cache accounting.

    Scoped to the Fireworks clients only
    (``shared.llm_clients._shadow_llm_call`` for both shadow tiers, and
    ``shared.llm_clients._fireworks_primary_chat`` since the item-19 GLM
    promotion) — deliberately NOT folded
    into :func:`openai_usage_dict`, which backs the real OpenAI cheap-tier
    path (``_call_openai``) and has its own (unverified, for Chat
    Completions) cache-field assumptions. Conflating the two would have
    made a shadow-only fix silently change real OpenAI cost accounting the
    moment OpenAI populates the field this function reads.

    Field name CONFIRMED 2026-07-03 against Fireworks' public API
    reference (https://docs.fireworks.ai/api-reference/post-chatcompletions):
    the documented "usage" response schema is ``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``, and
    ``prompt_tokens_details.cached_tokens`` (integer or null) — the same
    field name/shape the OpenAI Chat Completions API itself uses. This is
    NOT the ``input_tokens_details`` field :func:`openai_usage_dict` checks
    first — that field belongs to OpenAI's Responses API, a different
    surface ``chat.completions.create()`` does not populate.

    Fireworks' prompt-caching guide (https://docs.fireworks.ai/guides/
    prompt-caching) additionally documents cache info via
    ``fireworks-prompt-tokens`` / ``fireworks-cached-prompt-tokens`` HTTP
    response HEADERS. When the body cache detail is absent/null, the caller
    may pass the already-allowlisted raw response headers and this function
    uses ``fireworks-cached-prompt-tokens`` rather than inventing a zero.

    INCLUSIVE convention: the same prompt-caching guide frames the header
    pair as "the number of tokens in the prompt, OUT OF WHICH N are
    cached" — cached_tokens is a SUBSET of prompt_tokens, not additional
    (matches the standard OpenAI semantics for this field shape). So the
    ``input_tokens`` this function returns is ``prompt_tokens -
    cached_tokens`` (floored at 0) rather than the raw prompt-token count
    — pricing the full prompt-token count at the input rate AND
    cache_read_input_tokens again at the cache rate would double-price the
    cached slice in :func:`estimate_usage_cost_usd`.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
        }

    normalized_headers = {
        str(key).lower(): value for key, value in dict(provider_headers or {}).items()
    }

    def _nonnegative_optional_int(value: Any) -> int | None:
        parsed = _safe_optional_int(value)
        return parsed if parsed is not None and parsed >= 0 else None

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_raw: Any = None
    if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
        cached_raw = prompt_details.get("cached_tokens")
    elif prompt_details is not None and hasattr(prompt_details, "cached_tokens"):
        cached_raw = getattr(prompt_details, "cached_tokens", None)
    cached_tokens = _nonnegative_optional_int(cached_raw)
    if cached_tokens is None:
        cached_tokens = _nonnegative_optional_int(
            normalized_headers.get("fireworks-cached-prompt-tokens")
        )

    raw_prompt_tokens = _nonnegative_optional_int(
        getattr(usage, "prompt_tokens", None)
    )
    if raw_prompt_tokens is None:
        raw_prompt_tokens = _nonnegative_optional_int(
            normalized_headers.get("fireworks-prompt-tokens")
        )
    output_tokens = _nonnegative_optional_int(
        getattr(usage, "completion_tokens", None)
    )

    input_tokens = (
        max(raw_prompt_tokens - cached_tokens, 0)
        if raw_prompt_tokens is not None and cached_tokens is not None
        else None
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cached_tokens,
        "cache_creation_input_tokens": 0,
    }


def google_usage_dict(response: Any) -> dict[str, int]:
    """Extract normalized token counts from google.genai generate_content responses."""
    md = getattr(response, "usage_metadata", None)
    if md is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    if isinstance(md, dict):
        prompt_raw = md.get("prompt_token_count") or md.get("promptTokenCount") or 0
        cand_raw = (
            md.get("candidates_token_count")
            or md.get("candidatesTokenCount")
            or md.get("output_token_count")
        )
        cached_raw = md.get("cached_content_token_count") or md.get("cachedContentTokenCount") or 0
    else:
        prompt_raw = getattr(md, "prompt_token_count", 0)
        cand_raw = getattr(md, "candidates_token_count", None)
        if cand_raw is None:
            total_any = getattr(md, "total_token_count", None)
            if total_any is not None:
                cand_raw = max(_safe_int(total_any) - _safe_int(prompt_raw), 0)
            else:
                cand_raw = 0
        cached_raw = getattr(md, "cached_content_token_count", 0)

    inp = _safe_int(prompt_raw)
    out = _safe_int(cand_raw)
    if isinstance(md, dict):
        if cand_raw is None:
            total_alt = md.get("total_token_count")
            if total_alt is not None:
                out = max(_safe_int(total_alt) - inp, 0)

    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": _safe_int(cached_raw),
        "cache_creation_input_tokens": 0,
    }
