"""LLM client wrappers. Two functions: cheap_llm() for extraction, opus_llm() for judgment.

Phase 1 of Langfuse adoption (audit observability layer) wraps every
public LLM caller with ``@observe(as_type="generation")`` so each call
appears as a span in the Langfuse trace UI alongside the existing
JSONL ``token-cost-log.jsonl`` sink. The decorator is a passthrough
when Langfuse credentials are absent OR ``LANGFUSE_DISABLE=1``, so
behavior stays byte-equivalent to pre-Phase-1.

cheap_llm cost recording: every provider branch calls
``record_llm_usage`` after ``_retry_with_backoff`` succeeds, parallel to
the Anthropic-judge callers (``opus_llm`` / ``opus_llm_cached`` /
``facial_llm``), so typed LLM receipts are provider-independent.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Callable

import shared.config as config

from shared.failures import classify_runtime_failure
from shared.llm_policy import (
    FireworksDeadlineExceeded,
    FireworksReasoningEffort,
    FireworksStagePolicy,
    FireworksToolContract,
    effective_reasoning_effort,
    fireworks_serving_path,
    parse_retry_after_seconds,
)
from shared.llm_spend_budget import (
    reserve_fireworks_spend,
    settle_fireworks_spend,
)
from shared.llm_usage import (
    anthropic_usage_dict,
    fireworks_shadow_usage_dict,
    estimate_usage_cost_usd,
    google_usage_dict,
    minimax_usage_dict,
    openai_usage_dict,
    record_llm_attempt,
    record_llm_usage,
)
from shared.claude_cli_transport import claude_cli_chat
from shared.observability import (
    get_current_observation_id,
    get_current_trace_id,
    get_trace_url,
    observe,
)


# Model id passed to google.genai (must align with MODEL_RATE_TABLE keys in llm_usage).
GOOGLE_CHEAP_MODEL_NAME = "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Cached SDK clients
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def get_llm_client(
    provider: str,
    api_key: str,
    timeout: float,
    *,
    max_retries: int = 0,
    base_url: str | None = None,
):
    """Return a cached Anthropic or OpenAI-compatible client.

    The ``max_retries=0`` default is load-bearing: the explicit retry wrappers
    in this module are the SOLE retry owners. SDK-default retries (2, with
    backoff) silently STACK under the wrappers — up to 6 attempts x 60s per
    wrapped attempt — which is how a shadow call ballooned to 238s+ on the
    2026-07-05 SPL live run (genuine completions finish in 22-53s; the 119-349s
    tail was pure retry-storm). The streaming intake call is the sole exception
    because it has no replay wrapper.
    """
    if provider == "anthropic":
        import anthropic

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        return anthropic.Anthropic(**client_kwargs)

    from openai import OpenAI

    base_urls = {
        "fireworks": config.FIREWORKS_BASE_URL,
        "minimax": config.MINIMAX_BASE_URL,
        "perplexity": "https://api.perplexity.ai/v1",
        "shadow_fireworks": config.SHADOW_FACIAL_BASE_URL,
    }
    if provider != "openai" and provider not in base_urls:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    resolved_base_url = base_url if base_url is not None else base_urls.get(provider)
    if resolved_base_url is not None:
        kwargs["base_url"] = resolved_base_url
    return OpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a transient API error worth retrying."""
    return classify_runtime_failure(exc, source="llm").retryable


def _resolve_max_attempts(max_attempts: int | None) -> int:
    if max_attempts is None:
        return _MAX_RETRIES
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise ValueError("max_attempts must be an integer >= 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    return max_attempts


def _retry_with_backoff(
    fn,
    label: str = "LLM",
    on_attempt: Callable[..., None] | None = None,
    max_attempts: int | None = None,
    is_retryable: Callable[[Exception], bool] | None = None,
):
    """Call fn() with exponential backoff retry on transient errors."""
    attempt_limit = _resolve_max_attempts(max_attempts)
    for attempt in range(attempt_limit):
        started = time.monotonic()
        try:
            result = fn()
            if on_attempt is not None:
                try:
                    on_attempt(
                        attempt_number=attempt + 1,
                        result=result,
                        exc=None,
                        classification=None,
                        will_retry=False,
                        retry_delay_seconds=0.0,
                        latency_seconds=time.monotonic() - started,
                    )
                except Exception:
                    pass
            return result
        except Exception as e:
            classification = classify_runtime_failure(e, source="llm")
            retryable = (
                is_retryable(e) if is_retryable is not None else classification.retryable
            )
            will_retry = attempt < attempt_limit - 1 and retryable
            retry_after = _error_retry_after_seconds(e)
            wait = (
                # Protect the live sourcing loop from pathological provider headers.
                min(retry_after, 30.0)
                if will_retry and retry_after is not None
                else (2 ** attempt) + random.uniform(0, 1)
                if will_retry
                else 0.0
            )
            if on_attempt is not None:
                try:
                    on_attempt(
                        attempt_number=attempt + 1,
                        result=None,
                        exc=e,
                        classification=classification,
                        will_retry=will_retry,
                        retry_delay_seconds=wait,
                        latency_seconds=time.monotonic() - started,
                    )
                except Exception:
                    pass
            if will_retry:
                print(
                    f"    [RETRY] {label} {classification.kind.lower()}/"
                    f"{classification.domain}/{classification.reason} "
                    f"({e}), attempt {attempt + 1}/{attempt_limit}, waiting {wait:.1f}s"
                )
                time.sleep(wait)
            else:
                raise
        except BaseException as e:
            if on_attempt is not None:
                try:
                    on_attempt(
                        attempt_number=attempt + 1,
                        result=None,
                        exc=e,
                        classification={
                            "reason": "interrupted",
                            "status_code": None,
                        },
                        will_retry=False,
                        retry_delay_seconds=0.0,
                        latency_seconds=time.monotonic() - started,
                    )
                except Exception:
                    pass
            raise


def _stash_langfuse_context(usage_context: dict | None) -> None:
    """Attach active Langfuse ids to the mutable caller context, if any."""

    if usage_context is None:
        return
    trace_id = get_current_trace_id()
    observation_id = get_current_observation_id()
    trace_url = get_trace_url(trace_id=trace_id) if trace_id else None
    usage_context["_langfuse"] = {
        "trace_id": trace_id,
        "observation_id": observation_id,
        "trace_url": trace_url,
    }


# P4.3.3: finish-reason tokens (normalized lowercase, substring-matched — see
# _mark_if_truncated) that signal a cheap-model response was cut off before
# completion. Provider vocabularies differ: Anthropic's Message.stop_reason
# is "max_tokens"; OpenAI's choice.finish_reason is "length"; Google's
# candidate.finish_reason is a FinishReason enum whose stringified form
# contains "MAX_TOKENS".
_TRUNCATION_TOKENS = ("max_tokens", "length")


def _normalize_finish_reason(raw: object) -> str:
    if raw is None:
        return ""
    text = getattr(raw, "name", None) or str(raw)
    return text.strip().lower()


def _mark_if_truncated(
    *,
    provider: str,
    model: str,
    finish_reason: object,
    usage_context: dict | None,
) -> None:
    """Count + log a cheap-model response that stopped before completion.

    Unlike ``opus_llm``/``opus_llm_cached`` (which already raise loudly on
    a non-``end_turn`` stop_reason), the cheap-model path
    (``_call_anthropic_cheap`` / ``_call_openai`` / ``_call_google``) never
    checked finish reason at all — a truncated response fell straight into
    ``_parse_json_response``'s force-closed JSON repair with no trace. This
    doesn't change that control flow (the repair may still salvage a valid
    object), it just makes the truncation itself visible: a console log
    (matching this module's existing anomaly-logging convention — see the
    zero-usage warnings in ``_call_openai``/``_call_google``) plus an
    ``llm_truncated`` marker written into ``usage_context`` so it lands as a
    real column in the persisted ``token-cost-log.jsonl`` row. Never sets the
    marker when NOT truncated — its absence IS "not truncated" (no
    affirmative negative/zero).

    Mutates ``usage_context`` in place (sets the key directly on the dict
    the caller passed in) rather than returning a new dict; it never clears
    an already-set ``llm_truncated`` key either. Callers MUST build a fresh
    ``usage_context`` dict per LLM call — reusing/sharing one dict object
    across multiple calls (e.g. a retry loop or several calls stashed under
    one logical operation) will leak a `True` set on an earlier truncated
    call forward into later, non-truncated calls that reuse the same dict.
    """
    normalized = _normalize_finish_reason(finish_reason)
    if not any(token in normalized for token in _TRUNCATION_TOKENS):
        return
    print(
        f"    [truncation] {provider}/{model} cheap-model response stopped "
        f"early (finish_reason={finish_reason!r}); JSON repair may still "
        "salvage it, but the response was truncated.",
        flush=True,
    )
    if usage_context is not None:
        usage_context["llm_truncated"] = True


def _record_llm_error(
    *,
    provider: str,
    model: str,
    request: dict,
    usage_context: dict | None,
    exc: Exception,
) -> None:
    """Emit a typed error receipt without masking the provider exception."""

    error_request = {
        **request,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:240],
    }
    try:
        record_llm_usage(
            provider=provider,
            model=model,
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            request=error_request,
            usage_context=usage_context,
            actual_status="error",
        )
        _stash_langfuse_context(usage_context)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@observe(as_type="generation", name="cheap_llm")
def cheap_llm(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = True,
    usage_context: dict | None = None,
) -> str | dict | list:
    """Call the cheap model (GPT-4o-mini or Gemini Flash) for DOM extraction.

    If expect_json=True, parses the response as JSON and returns dict/list.
    Otherwise returns raw string.
    """
    if config.CHEAP_MODEL_PROVIDER == "openai":
        return _call_openai(
            system_prompt,
            user_prompt,
            expect_json,
            usage_context=usage_context,
        )
    elif config.CHEAP_MODEL_PROVIDER == "anthropic":
        return _call_anthropic_cheap(
            system_prompt,
            user_prompt,
            expect_json,
            usage_context=usage_context,
        )
    elif config.CHEAP_MODEL_PROVIDER == "google":
        return _call_google(
            system_prompt,
            user_prompt,
            expect_json,
            usage_context=usage_context,
        )
    elif config.CHEAP_MODEL_PROVIDER == "fireworks":
        # max_tokens=8192 mirrors _call_openai's cheap-tier cap. A reasoning
        # model here spends part of that budget on reasoning before content —
        # the truncation raise in _fireworks_primary_chat makes a clipped
        # extraction loud instead of feeding half-JSON to the repair path.
        return _fireworks_primary_chat(
            model=config.CHEAP_MODEL_NAME,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=expect_json,
            max_tokens=8192,
            usage_context=usage_context,
            capture=None,
            label="Cheap-fireworks",
            prompt_cache=False,
        )
    elif config.CHEAP_MODEL_PROVIDER == "minimax":
        try:
            return _call_minimax_cheap(
                system_prompt,
                user_prompt,
                expect_json,
                usage_context=usage_context,
            )
        except Exception as primary_exc:
            fallback_provider = config.CHEAP_MODEL_FALLBACK_PROVIDER
            fallback_model = config.CHEAP_MODEL_FALLBACK_NAME
            if not fallback_provider:
                raise
            if fallback_provider != "anthropic" or not fallback_model:
                raise RuntimeError(
                    "MiniMax cheap fallback must be anthropic with a non-empty model"
                ) from primary_exc
            if usage_context is not None:
                usage_context["cheap_fallback_used"] = True
                usage_context["cheap_fallback_from"] = "minimax"
            print(
                "    [fallback] MiniMax extraction unavailable; using Haiku",
                flush=True,
            )
            fallback_context = dict(usage_context or {})
            fallback_context.update(
                {
                    "provider_role": "fallback",
                    "fallback_from_provider": "minimax",
                }
            )
            fallback_result = _call_anthropic_cheap(
                system_prompt,
                user_prompt,
                expect_json,
                usage_context=fallback_context,
                model_name=fallback_model,
            )
            if usage_context is not None:
                fallback_receipt = fallback_context.get("_llm_receipt")
                if fallback_receipt is not None:
                    usage_context["_llm_receipt"] = fallback_receipt
                if usage_context.pop("llm_truncated", None):
                    usage_context["cheap_primary_truncated"] = True
            return fallback_result
    elif config.CHEAP_MODEL_PROVIDER == "claude_cli":
        model = _claude_cli_model_id(config.CHEAP_MODEL_NAME)
        return _call_claude_cli(
            model=f"claude-cli:{model}",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=expect_json,
            max_tokens=8192,
            usage_context=usage_context,
        )
    else:
        raise RuntimeError(f"Unknown CHEAP_MODEL_PROVIDER: {config.CHEAP_MODEL_PROVIDER}")


def _first_text_block(message) -> str:
    """Extract the first text block from an Anthropic response.

    ``content[0]`` is NOT always text: models with always-on thinking
    (claude-fable-5) lead with a ThinkingBlock, so indexing [0] raised
    ``'ThinkingBlock' object has no attribute 'text'`` on the first live
    Fable shadow call. Thinking blocks are skipped, never read — the
    judgment/plan contract lives in the text block.
    """
    blocks = getattr(message, "content", None) or []
    for block in blocks:
        if getattr(block, "type", "") == "text":
            return getattr(block, "text", "") or ""
    # Fallback for block objects that carry text without a type discriminator
    # (test doubles, SDK variations). Thinking blocks expose `.thinking`, not
    # `.text`, so they can never match either scan.
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    raise RuntimeError(
        "Anthropic response contained no text block "
        f"(stop_reason={getattr(message, 'stop_reason', None)!r})"
    )


def _is_fireworks_model(model: str) -> bool:
    """Fireworks model ids are account-scoped paths ("accounts/...")."""
    return model.startswith("accounts/")


def _is_claude_cli_model(model: str) -> bool:
    return model.startswith("claude-cli:")


def _claude_cli_model_id(model: str) -> str:
    return model.removeprefix("claude-cli:")


def _call_claude_cli(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    expect_json: bool,
    max_tokens: int | None = None,
    usage_context: dict | None = None,
    timeout_seconds: float = 600.0,
) -> str | dict | list:
    request: dict[str, Any] = {
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "expect_json": bool(expect_json),
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    return claude_cli_chat(
        model=_claude_cli_model_id(model),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expect_json=expect_json,
        usage_context=usage_context,
        request=request,
        timeout_seconds=timeout_seconds,
    )


def _resolve_tier_model(model_name: str | None) -> str:
    """Resolve a tier-routed model id (Anthropic or Fireworks).

    The tier envs (STRATEGY_MODEL_NAME / FULL_EVAL_MODEL_NAME) let a caller
    reassign its stage by env flip. A Fireworks id ("accounts/...") routes to
    ``_fireworks_primary_chat`` — the provider dispatch that lands with the
    GLM promotion decision (plans/sourcing-generality-hardening.md item 19,
    taken 2026-07-06: Fable on the strategy tier, GLM everywhere else).
    """
    return model_name or config.OPUS_MODEL_NAME


_LINKEDIN_STRATEGY_STAGE_PREFIXES = (
    "linkedin_preflight",
    "linkedin_strategy",
    "linkedin_plan_",
    "linkedin_adapt_",
    "linkedin_force_narrow_",
)


def _fireworks_strategy_reasoning_effort(
    usage_context: dict | None,
) -> FireworksReasoningEffort | None:
    """Return the explicit GLM effort for LinkedIn table-setting calls only."""

    stage = str((usage_context or {}).get("stage") or "").strip()
    if not stage.startswith(_LINKEDIN_STRATEGY_STAGE_PREFIXES):
        return None
    effort = config.FIREWORKS_STRATEGY_REASONING_EFFORT
    if not effort:
        return None
    if effort not in {"high", "max"}:
        raise ValueError(
            "FIREWORKS_STRATEGY_REASONING_EFFORT must be empty, 'high', or 'max'"
        )
    return effort


def _fireworks_strategy_stage_policy(
    usage_context: dict | None,
    explicit_policy: FireworksStagePolicy | None,
) -> FireworksStagePolicy | None:
    """Bound LinkedIn table-setting calls independently of shadow timing.

    Preflight already retries the complete generate/parse/lint operation once,
    so its provider policy owns exactly one attempt per outer pass. Other
    strategy stages retain the configured provider-aware retry allowance.
    Explicit caller policies always win.
    """

    if explicit_policy is not None:
        return explicit_policy
    stage = str((usage_context or {}).get("stage") or "").strip()
    if not stage.startswith(_LINKEDIN_STRATEGY_STAGE_PREFIXES):
        return None
    attempt_timeout = config.FIREWORKS_STRATEGY_ATTEMPT_TIMEOUT_SECONDS
    configured_attempts = config.FIREWORKS_STRATEGY_MAX_ATTEMPTS
    max_attempts = 1 if stage == "linkedin_preflight_v2" else configured_attempts
    total_deadline = config.FIREWORKS_STRATEGY_TOTAL_DEADLINE_SECONDS
    return FireworksStagePolicy(
        stage=stage,
        reasoning_effort=_fireworks_strategy_reasoning_effort(usage_context),
        attempt_timeout_seconds=attempt_timeout,
        total_deadline_seconds=total_deadline,
        max_attempts=max_attempts,
        response_transport=(
            "stream" if stage == "linkedin_preflight_v2" else "complete"
        ),
    )


# Models whose thinking is always on and whose thinking blocks return EMPTY
# text unless the request opts into the summarized display
# (thinking={"type": "adaptive", "display": "summarized"}). Scoped by model
# prefix, NEVER sent to Opus-family models: on claude-opus-4-7/4-8 an
# omitted `thinking` param means thinking OFF, so adding one would change
# primary behavior — on these models thinking runs regardless and `display`
# controls visibility only (billed identically), making the opt-in
# behavior-neutral. This is what makes the Fable shadow's reasoning readable
# in shadow_strategy artifacts (item 19).
_ALWAYS_THINKING_MODEL_PREFIXES = ("claude-fable", "claude-mythos")


def _thinking_request_kwargs(model: str) -> dict:
    """Request kwargs for readable thinking, empty for every other model."""
    if model.startswith(_ALWAYS_THINKING_MODEL_PREFIXES):
        return {"thinking": {"type": "adaptive", "display": "summarized"}}
    return {}


def _thinking_summary(message) -> str | None:
    """Join the summarized-thinking text of an Anthropic response, if any.

    Returns None when there are no thinking blocks OR the blocks carry empty
    text (the display="omitted" default) — absence and invisibility read the
    same to callers, which is correct: neither yields readable reasoning.
    """
    parts = [
        getattr(block, "thinking", "") or ""
        for block in (getattr(message, "content", None) or [])
        if getattr(block, "type", "") == "thinking"
    ]
    joined = "\n\n".join(part for part in parts if part.strip())
    return joined or None


def _cached_fireworks_policy_with_bounds(
    *,
    policy: FireworksStagePolicy | None,
    usage_context: dict | None,
    timeout_seconds: float | None,
    max_attempts: int | None,
) -> FireworksStagePolicy | None:
    """Map optional cached-call bounds onto the Fireworks policy surface."""

    if timeout_seconds is None and max_attempts is None:
        return policy
    if policy is not None:
        overrides: dict[str, Any] = {}
        if timeout_seconds is not None:
            overrides["attempt_timeout_seconds"] = timeout_seconds
        if max_attempts is not None:
            overrides["max_attempts"] = _resolve_max_attempts(max_attempts)
        return replace(policy, **overrides)

    attempt_limit = _resolve_max_attempts(max_attempts)
    attempt_timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else config.SHADOW_LLM_TIMEOUT_SECONDS
    )
    # Match the legacy wrapper's worst-case exponential+jitter waits while
    # still giving the policy runner a hard total wall-clock ceiling.
    backoff_budget = sum(
        (2**attempt) + 1.0 for attempt in range(attempt_limit - 1)
    )
    stage = str((usage_context or {}).get("stage") or "opus_llm_cached").strip()
    return FireworksStagePolicy(
        stage=stage or "opus_llm_cached",
        attempt_timeout_seconds=attempt_timeout,
        total_deadline_seconds=(attempt_timeout * attempt_limit) + backoff_budget,
        max_attempts=attempt_limit,
    )


@observe(as_type="generation", name="opus_llm")
def opus_llm(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = True,
    max_tokens: int = 8192,
    usage_context: dict | None = None,
    model_name: str | None = None,
    capture: dict | None = None,
    timeout_seconds: float | None = None,
    policy: FireworksStagePolicy | None = None,
    tool_contract: FireworksToolContract | None = None,
) -> str | dict:
    """Call Opus for candidate judgment. Returns parsed JSON or raw string.

    ``capture``, when provided, is filled in place with response metadata the
    string return value cannot carry: ``thinking_summary`` (the summarized
    reasoning of an always-thinking model, None elsewhere) and
    ``stop_reason``. Filled as soon as the response arrives — BEFORE the
    non-end_turn raise below — so a refusal/truncation still hands the caller
    whatever reasoning came back. Same idiom as ``_shadow_llm_call``'s
    capture dict. ``timeout_seconds`` overrides only the Fireworks-dispatched
    path; Anthropic keeps its fixed 300.0 client/request behavior.
    """
    model = _resolve_tier_model(model_name)
    if _is_claude_cli_model(model):
        if policy is not None or tool_contract is not None:
            raise ValueError("Fireworks policy/tool contract requires a Fireworks model")
        return _call_claude_cli(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=expect_json,
            max_tokens=max_tokens,
            usage_context=usage_context,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else 600.0,
        )
    if _is_fireworks_model(model):
        resolved_policy = _fireworks_strategy_stage_policy(usage_context, policy)
        return _fireworks_primary_chat(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=expect_json,
            max_tokens=max_tokens,
            usage_context=usage_context,
            capture=capture,
            label="Fireworks",
            prompt_cache=False,
            timeout_seconds=timeout_seconds,
            policy=resolved_policy,
            tool_contract=tool_contract,
            reasoning_effort=_fireworks_strategy_reasoning_effort(usage_context),
        )
    if policy is not None or tool_contract is not None:
        raise ValueError("Fireworks policy/tool contract requires a Fireworks model")
    client = get_llm_client("anthropic", config.ANTHROPIC_API_KEY, 300.0)

    def _call():
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            **_thinking_request_kwargs(model),
        )
        return message

    request = {
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "max_tokens": max_tokens,
        "expect_json": bool(expect_json),
    }
    try:
        message = _retry_with_backoff(_call, label="Opus")
    except Exception as exc:
        _record_llm_error(
            provider="anthropic",
            model=model,
            request=request,
            usage_context=usage_context,
            exc=exc,
        )
        raise
    if capture is not None:
        capture["thinking_summary"] = _thinking_summary(message)
        capture["stop_reason"] = getattr(message, "stop_reason", None)
    record_llm_usage(
        provider="anthropic",
        model=model,
        usage=anthropic_usage_dict(message),
        request={
            **request,
            "stop_reason": getattr(message, "stop_reason", None),
        },
        usage_context=usage_context,
    )
    _stash_langfuse_context(usage_context)
    if message.stop_reason != "end_turn":
        raise RuntimeError(
            f"Opus response truncated: stop_reason={message.stop_reason}. Increase max_tokens or reduce prompt size."
        )
    text = _first_text_block(message).strip()

    if expect_json:
        return _parse_json_response(text)
    return text


@observe(as_type="generation", name="opus_llm_cached")
def opus_llm_cached(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = True,
    max_tokens: int = 8192,
    usage_context: dict | None = None,
    model_name: str | None = None,
    capture: dict | None = None,
    policy: FireworksStagePolicy | None = None,
    tool_contract: FireworksToolContract | None = None,
    timeout_seconds: float | None = None,
    max_attempts: int | None = None,
) -> str | dict:
    """Call Opus with prompt caching on the system prompt.

    System prompt is sent as a content block with cache_control: {"type": "ephemeral"}.
    Cache write costs 1.25x, cache read costs 0.1x, TTL is 5 minutes (refreshed on hit).

    ``capture``, when provided, is filled in place with response metadata the
    string return value cannot carry: ``thinking_summary`` (the summarized
    reasoning of an always-thinking model, None elsewhere) and
    ``stop_reason``. Filled as soon as the response arrives — BEFORE the
    non-end_turn raise below — so a refusal/truncation still hands the caller
    whatever reasoning came back. Same idiom as ``_shadow_llm_call``'s
    capture dict.

    ``timeout_seconds`` and ``max_attempts`` are optional per-attempt bounds.
    The Anthropic SDK's hidden retries are always disabled so the explicit
    wrapper is the sole attempt owner; Fireworks receives the equivalent typed
    stage policy.
    """
    model = _resolve_tier_model(model_name)
    if _is_claude_cli_model(model):
        if policy is not None or tool_contract is not None:
            raise ValueError("Fireworks policy/tool contract requires a Fireworks model")
        return _call_claude_cli(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=expect_json,
            max_tokens=max_tokens,
            usage_context=usage_context,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else 600.0,
        )
    if _is_fireworks_model(model):
        strategy_policy = _fireworks_strategy_stage_policy(usage_context, policy)
        bounded_policy = _cached_fireworks_policy_with_bounds(
            policy=strategy_policy,
            usage_context=usage_context,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        )
        return _fireworks_primary_chat(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=expect_json,
            max_tokens=max_tokens,
            usage_context=usage_context,
            capture=capture,
            label="Fireworks-cached",
            prompt_cache=True,
            timeout_seconds=timeout_seconds,
            policy=bounded_policy,
            tool_contract=tool_contract,
            reasoning_effort=_fireworks_strategy_reasoning_effort(usage_context),
        )
    if policy is not None or tool_contract is not None:
        raise ValueError("Fireworks policy/tool contract requires a Fireworks model")
    client = get_llm_client(
        "anthropic",
        config.ANTHROPIC_API_KEY,
        timeout_seconds if timeout_seconds is not None else 300.0,
    )

    def _call():
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
            **_thinking_request_kwargs(model),
        )
        return message

    request = {
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "max_tokens": max_tokens,
        "expect_json": bool(expect_json),
        "prompt_cache": "ephemeral",
    }
    try:
        message = _retry_with_backoff(
            _call,
            label="Opus-cached",
            max_attempts=max_attempts,
        )
    except Exception as exc:
        _record_llm_error(
            provider="anthropic",
            model=model,
            request=request,
            usage_context=usage_context,
            exc=exc,
        )
        raise
    if capture is not None:
        capture["thinking_summary"] = _thinking_summary(message)
        capture["stop_reason"] = getattr(message, "stop_reason", None)
    record_llm_usage(
        provider="anthropic",
        model=model,
        usage=anthropic_usage_dict(message),
        request={
            **request,
            "stop_reason": getattr(message, "stop_reason", None),
        },
        usage_context=usage_context,
    )
    _stash_langfuse_context(usage_context)
    if message.stop_reason != "end_turn":
        raise RuntimeError(
            f"Opus response truncated: stop_reason={message.stop_reason}. Increase max_tokens or reduce prompt size."
        )
    text = _first_text_block(message).strip()

    if expect_json:
        return _parse_json_response(text)
    return text


@observe(as_type="generation", name="opus_llm_cached_stream")
def opus_llm_cached_stream(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 4096,
    usage_context: dict | None = None,
):
    """Stream Opus completion with prompt caching on the system block.

    Mirrors :func:`opus_llm_cached` lines 170-187: system prompt is sent
    as a content block with ``cache_control: {"type": "ephemeral"}`` so the
    expensive 4-6k-token orchestrator system prompt is paid for once per
    5-minute window and re-read at 0.1x cost on subsequent turns. Without
    this, the conversational intake's 5-15-turn cost projection collapses.

    Generator contract: yields tagged tuples so the caller can distinguish
    text deltas from end-of-stream usage telemetry without out-of-band
    state.

    - ``("delta", text_chunk)`` — for each token-delta the SDK emits.
    - ``("usage", usage_dict)`` — exactly once at end-of-stream, sourced
      from ``stream.get_final_message()`` and shaped via
      :func:`shared.llm_usage.anthropic_usage_dict`. Includes
      ``input_tokens``, ``output_tokens``, ``cache_read_input_tokens``,
      ``cache_creation_input_tokens`` so the C5 endpoint can compute
      per-turn cost in-stream and update
      ``state_json.conversation_meta.cost_usd_running_total`` without
      tailing the JSONL log.

    No retry wrapper. Mid-stream failures are best handled by the caller
    (the orchestrator yields a fallback delta + synthetic usage tuple so
    the SSE consumer's accounting stays single-shape). Retrying a stream
    means re-yielding partial text, which is worse than failing fast.
    """

    if _is_fireworks_model(config.OPUS_MODEL_NAME):
        # Only the conversational-intake surface streams; it has no
        # Fireworks route. Fail with the fix in the message rather than a
        # confusing Anthropic 404 mid-stream.
        raise RuntimeError(
            f"opus_llm_cached_stream has no Fireworks route, but "
            f"OPUS_MODEL_NAME={config.OPUS_MODEL_NAME!r}. The intake surface "
            "needs an Anthropic model id — unset the Fireworks override for "
            "this process or run intake separately."
        )
    # SDK retries cover pre-stream connection setup only; streams are never replayed.
    client = get_llm_client(
        "anthropic",
        config.ANTHROPIC_API_KEY,
        300.0,
        max_retries=2,
    )

    request = {
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "max_tokens": max_tokens,
        "prompt_cache": "ephemeral",
        "stream": True,
    }
    try:
        with client.messages.stream(
            model=config.OPUS_MODEL_NAME,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield ("delta", text)
            final = stream.get_final_message()
    except Exception as exc:
        _record_llm_error(
            provider="anthropic",
            model=config.OPUS_MODEL_NAME,
            request=request,
            usage_context=usage_context,
            exc=exc,
        )
        raise

    usage = anthropic_usage_dict(final)
    record_llm_usage(
        provider="anthropic",
        model=config.OPUS_MODEL_NAME,
        usage=usage,
        request={
            **request,
            "stop_reason": getattr(final, "stop_reason", None),
        },
        usage_context=usage_context,
    )
    _stash_langfuse_context(usage_context)
    yield ("usage", usage)


@observe(as_type="generation", name="facial_llm")
def facial_llm(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = True,
    max_tokens: int = 2048,
    usage_context: dict | None = None,
    policy: FireworksStagePolicy | None = None,
    tool_contract: FireworksToolContract | None = None,
) -> str | dict:
    """Call the facial triage model with prompt caching.

    Defaults to Opus (same as opus_llm_cached) but can be overridden to Sonnet
    via FACIAL_MODEL_NAME in .env for 5x cost reduction on facial calls.
    Lower default max_tokens since facial responses are short.
    """
    if _is_claude_cli_model(config.FACIAL_MODEL_NAME):
        if policy is not None or tool_contract is not None:
            raise ValueError("Fireworks policy/tool contract requires a Fireworks model")
        return _call_claude_cli(
            model=config.FACIAL_MODEL_NAME,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=expect_json,
            max_tokens=max_tokens,
            usage_context=usage_context,
        )
    if _is_fireworks_model(config.FACIAL_MODEL_NAME):
        return _fireworks_primary_chat(
            model=config.FACIAL_MODEL_NAME,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=expect_json,
            max_tokens=max_tokens,
            usage_context=usage_context,
            capture=None,
            label="Facial-fireworks",
            prompt_cache=True,
            policy=policy,
            tool_contract=tool_contract,
        )
    if policy is not None or tool_contract is not None:
        raise ValueError("Fireworks policy/tool contract requires a Fireworks model")
    client = get_llm_client("anthropic", config.ANTHROPIC_API_KEY, 300.0)

    def _call():
        message = client.messages.create(
            model=config.FACIAL_MODEL_NAME,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message

    request = {
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "max_tokens": max_tokens,
        "expect_json": bool(expect_json),
        "prompt_cache": "ephemeral",
    }
    try:
        message = _retry_with_backoff(_call, label="Facial")
    except Exception as exc:
        _record_llm_error(
            provider="anthropic",
            model=config.FACIAL_MODEL_NAME,
            request=request,
            usage_context=usage_context,
            exc=exc,
        )
        raise
    record_llm_usage(
        provider="anthropic",
        model=config.FACIAL_MODEL_NAME,
        usage=anthropic_usage_dict(message),
        request={
            **request,
            "stop_reason": getattr(message, "stop_reason", None),
        },
        usage_context=usage_context,
    )
    _stash_langfuse_context(usage_context)
    if message.stop_reason != "end_turn":
        raise RuntimeError(
            f"Facial model response truncated: stop_reason={message.stop_reason}."
        )
    text = _first_text_block(message).strip()

    if expect_json:
        return _parse_json_response(text)
    return text


# GLM-5.2 sampling is intentionally vendor-calibrated (applies to GLM as
# shadow AND as primary). Z.AI's GLM-5.2 guidance defaults to
# temperature=1.0 / top_p=0.95 and recommends tuning only ONE of them;
# Zhipu's own reasoning-benchmark evals also run at temperature=1.0.
# Off-calibration low temperature on reasoning models drives
# rumination/repetition, plausibly upstream of the observed 34-39K-char
# reasoning tails and finish_reason=length parse failures. Do NOT "fix" this
# back to 0.1. Legacy calls deliberately omit reasoning_effort; explicit
# FireworksStagePolicy calls may pin the experimental high/max posture.
SHADOW_LLM_TEMPERATURE = 1.0


@dataclass(slots=True)
class _FireworksResponseEnvelope:
    response: Any
    headers: dict[str, str]
    request_id: str | None
    stream_metadata: dict[str, Any] | None = None


_FIREWORKS_PROVIDER_HEADER_KEYS = frozenset(
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
_FIREWORKS_COOLDOWN_LOCK = threading.Lock()
_FIREWORKS_COOLDOWNS: dict[tuple[str, str, str], float] = {}


def _header_mapping(value: object) -> dict[str, str]:
    try:
        items = dict(value or {}).items()
    except (TypeError, ValueError):
        return {}
    return {str(key).lower(): str(item) for key, item in items}


def _fireworks_headers(value: object) -> dict[str, str]:
    headers = _header_mapping(value)
    return {
        key: headers[key]
        for key in sorted(_FIREWORKS_PROVIDER_HEADER_KEYS)
        if key in headers
    }


def _fireworks_error_headers(exc: BaseException) -> dict[str, str]:
    response = getattr(exc, "response", None)
    return _fireworks_headers(getattr(response, "headers", None))


def _error_retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = _header_mapping(getattr(response, "headers", None))
    return parse_retry_after_seconds(headers.get("retry-after"))


def _create_fireworks_completion(
    client: Any,
    request_kwargs: dict[str, Any],
) -> _FireworksResponseEnvelope:
    completions = client.chat.completions
    raw_surface = getattr(completions, "with_raw_response", None)
    if raw_surface is not None:
        raw = raw_surface.create(**request_kwargs)
        parser = getattr(raw, "parse", None)
        if callable(parser):
            response = parser()
            headers = _fireworks_headers(getattr(raw, "headers", None))
            request_id = getattr(raw, "request_id", None) or headers.get("x-request-id")
            return _FireworksResponseEnvelope(response, headers, request_id)
    response = completions.create(**request_kwargs)
    return _FireworksResponseEnvelope(
        response=response,
        headers={},
        request_id=getattr(response, "_request_id", None),
    )


def _create_fireworks_streaming_completion(
    client: Any,
    request_kwargs: dict[str, Any],
    *,
    deadline_at: float,
) -> _FireworksResponseEnvelope:
    """Consume one chat SSE response without persisting reasoning deltas.

    The OpenAI SDK's ordinary ``create`` surface eagerly reads the entire body
    before returning.  Preflight uses ``with_streaming_response`` so each SSE
    event resets the transport read-inactivity timer while this explicit timer
    still enforces the logical-call deadline.  Only final answer content and
    safe counters/timings leave this helper; provider reasoning is ignored.
    """

    completions = client.chat.completions
    streaming_surface = getattr(completions, "with_streaming_response", None)
    if streaming_surface is None:
        raise RuntimeError("Fireworks streaming response surface is unavailable")

    started = time.monotonic()
    if deadline_at <= started:
        raise FireworksDeadlineExceeded(
            "Fireworks total deadline expired before streaming response"
        )

    content_parts: list[str] = []
    tool_call_parts: dict[int, dict[str, object]] = {}
    event_count = 0
    content_delta_count = 0
    reasoning_delta_count = 0
    first_event_ms: float | None = None
    first_content_ms: float | None = None
    finish_reason: object = None
    response_id: str | None = None
    usage: object = None
    deadline_hit = threading.Event()
    finished = threading.Event()

    def _stream_metadata() -> dict[str, Any]:
        return {
            "response_transport": "stream",
            "stream_event_count": event_count,
            "stream_content_delta_count": content_delta_count,
            "stream_reasoning_delta_count": reasoning_delta_count,
            "stream_first_event_ms": first_event_ms,
            "stream_first_content_ms": first_content_ms,
        }

    response_manager = streaming_surface.create(
        **request_kwargs,
        stream=True,
        stream_options={"include_usage": True},
    )
    with response_manager as raw:
        headers = _fireworks_headers(getattr(raw, "headers", None))
        request_id = getattr(raw, "request_id", None) or headers.get("x-request-id")

        def _close_at_deadline() -> None:
            if finished.is_set():
                return
            deadline_hit.set()
            try:
                raw.close()
            except Exception:
                pass

        timer = threading.Timer(max(deadline_at - time.monotonic(), 0.0), _close_at_deadline)
        timer.daemon = True
        timer.start()
        try:
            stream = raw.parse()
            for chunk in stream:
                now = time.monotonic()
                if deadline_hit.is_set() or now >= deadline_at:
                    raise FireworksDeadlineExceeded(
                        "Fireworks total deadline expired while consuming stream"
                    )
                event_count += 1
                if first_event_ms is None:
                    first_event_ms = round((now - started) * 1000, 3)
                chunk_id = getattr(chunk, "id", None)
                if chunk_id and response_id is None:
                    response_id = str(chunk_id)
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                for choice in getattr(chunk, "choices", None) or []:
                    candidate_finish = getattr(choice, "finish_reason", None)
                    if candidate_finish is not None:
                        finish_reason = candidate_finish
                    delta = getattr(choice, "delta", None)
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning is None:
                        model_extra = getattr(delta, "model_extra", None)
                        if isinstance(model_extra, dict):
                            reasoning = model_extra.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_delta_count += 1
                    content = getattr(delta, "content", None)
                    if isinstance(content, str) and content:
                        content_parts.append(content)
                        content_delta_count += 1
                        if first_content_ms is None:
                            first_content_ms = round((now - started) * 1000, 3)
                    tool_calls = getattr(delta, "tool_calls", None)
                    if tool_calls is None:
                        if isinstance(delta, dict):
                            tool_calls = delta.get("tool_calls")
                        else:
                            model_extra = getattr(delta, "model_extra", None)
                            if isinstance(model_extra, dict):
                                tool_calls = model_extra.get("tool_calls")
                    for fragment in tool_calls or []:
                        if isinstance(fragment, dict):
                            index = fragment.get("index")
                            fragment_id = fragment.get("id")
                            function = fragment.get("function")
                        else:
                            index = getattr(fragment, "index", None)
                            fragment_id = getattr(fragment, "id", None)
                            function = getattr(fragment, "function", None)
                            fragment_extra = getattr(
                                fragment, "model_extra", None
                            )
                            if isinstance(fragment_extra, dict):
                                if index is None:
                                    index = fragment_extra.get("index")
                                if fragment_id is None:
                                    fragment_id = fragment_extra.get("id")
                                if function is None:
                                    function = fragment_extra.get("function")
                        if not isinstance(index, int):
                            continue
                        assembled = tool_call_parts.setdefault(
                            index,
                            {"id": None, "name": None, "arguments": []},
                        )
                        if isinstance(fragment_id, str) and fragment_id:
                            if not assembled["id"]:
                                assembled["id"] = fragment_id
                        if isinstance(function, dict):
                            function_name = function.get("name")
                            arguments = function.get("arguments")
                        else:
                            function_name = getattr(function, "name", None)
                            arguments = getattr(function, "arguments", None)
                            if function is not None:
                                function_extra = getattr(
                                    function, "model_extra", None
                                )
                                if isinstance(function_extra, dict):
                                    if function_name is None:
                                        function_name = function_extra.get("name")
                                    if arguments is None:
                                        arguments = function_extra.get("arguments")
                        if isinstance(function_name, str) and function_name:
                            if not assembled["name"]:
                                assembled["name"] = function_name
                        if isinstance(arguments, str):
                            argument_parts = assembled["arguments"]
                            if isinstance(argument_parts, list):
                                argument_parts.append(arguments)
            if deadline_hit.is_set() or time.monotonic() >= deadline_at:
                raise FireworksDeadlineExceeded(
                    "Fireworks total deadline expired while completing stream"
                )
        except BaseException as exc:
            if deadline_hit.is_set() and not isinstance(
                exc, FireworksDeadlineExceeded
            ):
                exc = FireworksDeadlineExceeded(
                    "Fireworks total deadline expired while consuming stream"
                )
            for name, value in (
                ("_cloris_stream_metadata", _stream_metadata()),
                ("_cloris_fireworks_headers", headers),
                ("_cloris_provider_request_id", request_id),
            ):
                try:
                    setattr(exc, name, value)
                except Exception:
                    pass
            raise exc
        finally:
            finished.set()
            timer.cancel()

    choices = []
    if event_count:
        message = SimpleNamespace(
            content="".join(content_parts),
            # Deliberately never aggregate or expose provider reasoning.
            reasoning_content=None,
        )
        if tool_call_parts:
            message.tool_calls = [
                SimpleNamespace(
                    id=parts["id"],
                    function=SimpleNamespace(
                        name=parts["name"],
                        arguments="".join(parts["arguments"]),
                    ),
                )
                for _index, parts in sorted(tool_call_parts.items())
            ]
        choices = [
            SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            )
        ]
    response = SimpleNamespace(
        id=response_id,
        choices=choices,
        usage=usage,
    )
    return _FireworksResponseEnvelope(
        response=response,
        headers=headers,
        request_id=request_id,
        stream_metadata=_stream_metadata(),
    )


def _fireworks_timeout_metadata(exc: BaseException) -> dict[str, str]:
    """Classify a wrapped transport timeout without logging exception text."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        phase = {
            "ConnectTimeout": "connect",
            "ReadTimeout": "read_inactivity",
            "WriteTimeout": "write",
            "PoolTimeout": "pool",
            "FireworksDeadlineExceeded": "logical_deadline",
        }.get(name)
        if phase is not None:
            return {
                "timeout_phase": phase,
                "transport_timeout_type": name,
            }
        current = current.__cause__ or current.__context__
    if type(exc).__name__ == "APITimeoutError":
        return {
            "timeout_phase": "transport_unknown",
            "transport_timeout_type": "APITimeoutError",
        }
    return {}


def _fireworks_response_finish_reason(response: Any) -> object:
    choices = getattr(response, "choices", None) or []
    return getattr(choices[0], "finish_reason", None) if choices else None


def _fireworks_response_id(response: Any) -> str | None:
    value = getattr(response, "id", None)
    return str(value) if value else None


def _fireworks_input_token_upper_bound(request_kwargs: dict[str, Any]) -> int:
    """Conservatively bound prompt tokens by serialized UTF-8 bytes.

    Provider tokenizers cannot emit more ordinary BPE tokens than the byte
    sequence they encode; 2,048 extra units cover chat/tool framing and future
    small transport fields. This intentionally over-reserves rather than
    estimating from words or characters.
    """

    prompt_surface = {
        key: request_kwargs[key]
        for key in ("messages", "tools", "tool_choice", "parallel_tool_calls")
        if key in request_kwargs
    }
    encoded = json.dumps(
        prompt_surface,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return len(encoded) + 2_048


def _measured_fireworks_cost(
    model: str,
    usage: dict[str, int | None] | None,
    usage_status: str,
) -> float | None:
    if usage_status != "measured" or usage is None:
        return None
    cost, _rate_source = estimate_usage_cost_usd(
        model=model,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cache_creation_input_tokens=int(
            usage.get("cache_creation_input_tokens") or 0
        ),
    )
    return cost


_TRANSPORT_TIMEOUT_CLASS_NAMES = frozenset(
    {"ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException"}
)
_TRANSPORT_FAILURE_CLASS_NAMES = frozenset(
    {
        "ConnectError",
        "ReadError",
        "WriteError",
        "NetworkError",
        "RemoteProtocolError",
        "ProtocolError",
        "ConnectionResetError",
        "ConnectionTerminated",
        "TransportError",
        "SSLError",
    }
)


def _transport_failure_reason(exc: BaseException) -> str | None:
    """Name-based cause-chain classification of raw httpx/httpcore failures.

    Streaming consumes the response body after the OpenAI SDK's ``create``
    returns, so mid-stream transport faults arrive as raw httpx classes
    instead of the SDK's ``APITimeoutError``/``APIConnectionError`` wrappers
    (measured live: two sessions killed by an unclassified
    ``httpx.ReadTimeout``, 2026-08-10/11). Classified by the type names on
    each node's MRO — exact-name matching defeated the base-class entries
    (an ``httpx.CloseError`` IS a ``NetworkError`` but matched neither,
    wave-1 review finding) — so this module keeps no direct httpx import. A
    ``FireworksDeadlineExceeded`` anywhere in the chain is the policy's own
    total budget and is never retryable.
    """

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        mro_names = {klass.__name__ for klass in type(current).__mro__}
        if "FireworksDeadlineExceeded" in mro_names:
            return None
        if mro_names & _TRANSPORT_TIMEOUT_CLASS_NAMES:
            return "timeout"
        if mro_names & _TRANSPORT_FAILURE_CLASS_NAMES:
            return "connection_error"
        current = current.__cause__ or current.__context__
    return None


def _fireworks_policy_error(
    exc: Exception,
    policy: FireworksStagePolicy,
) -> tuple[bool, int | None, str]:
    status_raw = getattr(exc, "status_code", None)
    try:
        status_code = int(status_raw) if status_raw is not None else None
    except (TypeError, ValueError):
        status_code = None
    class_name = type(exc).__name__
    # The logical deadline is the policy's own total budget — never retried.
    # Checked before the TimeoutError branch because FireworksDeadlineExceeded
    # subclasses TimeoutError.
    if isinstance(exc, FireworksDeadlineExceeded):
        return False, status_code, "logical_deadline"
    if class_name == "APITimeoutError" or isinstance(exc, TimeoutError):
        return True, status_code, "timeout"
    if class_name == "APIConnectionError" or isinstance(exc, ConnectionError):
        return True, status_code, "connection_error"
    if status_code in policy.retry_status_codes:
        return True, status_code, f"http_{status_code}"
    if status_code is None:
        # Only status-less failures consult the transport walk: a terminal
        # 4xx that happens to carry a transport fault in its context chain
        # must stay terminal (wave-1 review finding).
        transport_reason = _transport_failure_reason(exc)
        if transport_reason is not None:
            return True, status_code, transport_reason
    return False, status_code, f"http_{status_code}" if status_code else "terminal"


def _cooldown_key(model: str) -> tuple[str, str, str]:
    return (config.FIREWORKS_BASE_URL, model, fireworks_serving_path(model))


def _set_fireworks_cooldown(key: tuple[str, str, str], delay_seconds: float) -> None:
    delay = float(delay_seconds)
    if not math.isfinite(delay):
        raise ValueError("Fireworks cooldown delay must be finite")
    target = time.monotonic() + max(delay, 0.0)
    with _FIREWORKS_COOLDOWN_LOCK:
        _FIREWORKS_COOLDOWNS[key] = max(_FIREWORKS_COOLDOWNS.get(key, 0.0), target)


def _wait_for_fireworks_cooldown(
    key: tuple[str, str, str],
    *,
    deadline: float,
) -> float:
    waited = 0.0
    while True:
        now = time.monotonic()
        with _FIREWORKS_COOLDOWN_LOCK:
            remaining = max(_FIREWORKS_COOLDOWNS.get(key, 0.0) - now, 0.0)
        if remaining <= 0:
            return waited
        if now + remaining >= deadline:
            raise FireworksDeadlineExceeded(
                "Fireworks total deadline expired in shared cooldown"
            )
        time.sleep(remaining)
        waited += remaining
        # Another concurrent request may have received a later Retry-After
        # while this caller slept. Re-read the shared max target before any
        # provider attempt instead of assuming the first wakeup is sufficient.


def _run_fireworks_with_policy(
    call: Callable[[float], _FireworksResponseEnvelope],
    *,
    model: str,
    label: str,
    policy: FireworksStagePolicy,
    on_attempt: Callable[..., None],
) -> _FireworksResponseEnvelope:
    started_all = time.monotonic()
    deadline = started_all + policy.total_deadline_seconds
    key = _cooldown_key(model)
    last_exc: Exception | None = None
    for attempt_index in range(policy.max_attempts):
        cooldown_wait = _wait_for_fireworks_cooldown(key, deadline=deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FireworksDeadlineExceeded("Fireworks total deadline expired before attempt")
        attempt_timeout = min(policy.attempt_timeout_seconds, remaining)
        started = time.monotonic()
        try:
            result = call(attempt_timeout)
        except BaseException as exc:
            if not isinstance(exc, Exception):
                try:
                    on_attempt(
                        attempt_number=attempt_index + 1,
                        result=None,
                        exc=exc,
                        classification={
                            "reason": "interrupted",
                            "status_code": None,
                        },
                        will_retry=False,
                        retry_delay_seconds=0.0,
                        retry_delay_source=None,
                        latency_seconds=time.monotonic() - started,
                        attempt_timeout_seconds=attempt_timeout,
                        cooldown_wait_seconds=cooldown_wait,
                    )
                except Exception:
                    pass
                raise
            last_exc = exc
            retryable, status_code, reason = _fireworks_policy_error(exc, policy)
            retry_headers = _fireworks_error_headers(exc)
            retry_after = _error_retry_after_seconds(exc)
            retry_source = "retry-after" if retry_after is not None else "exponential"
            delay = (
                retry_after
                if retry_after is not None
                else (2 ** attempt_index) + random.uniform(0, 1)
            )
            remaining_after = deadline - time.monotonic()
            capacity_signal = status_code in policy.retry_status_codes
            if retryable and capacity_signal:
                # A provider capacity signal governs every worker sharing this
                # endpoint/model/serving path, even when this logical call has
                # exhausted its own attempts or deadline. Otherwise a sibling
                # immediately re-hammers the provider after a final-attempt 429.
                _set_fireworks_cooldown(key, delay)
            will_retry = (
                retryable
                and attempt_index < policy.max_attempts - 1
                and delay < remaining_after
            )
            try:
                on_attempt(
                    attempt_number=attempt_index + 1,
                    result=None,
                    exc=exc,
                    classification={
                        "reason": reason,
                        "status_code": status_code,
                    },
                    will_retry=will_retry,
                    retry_delay_seconds=delay if will_retry else 0.0,
                    retry_delay_source=retry_source if will_retry else None,
                    latency_seconds=time.monotonic() - started,
                    attempt_timeout_seconds=attempt_timeout,
                    cooldown_wait_seconds=cooldown_wait,
                )
            except Exception:
                pass
            if not will_retry:
                if retryable and attempt_index < policy.max_attempts - 1 and delay >= remaining_after:
                    raise FireworksDeadlineExceeded(
                        "Fireworks total deadline cannot accommodate retry"
                    ) from exc
                raise
            print(
                f"    [RETRY] {label} provider/{reason} ({type(exc).__name__}), "
                f"attempt {attempt_index + 1}/{policy.max_attempts}, waiting {delay:.1f}s"
            )
            if not capacity_signal:
                # HTTP capacity signals sleep through the shared cooldown at
                # the top of the next attempt so every worker observes one
                # max deadline.  Timeout/connection failures have no shared
                # provider signal, but the declared per-call backoff must still
                # elapse before this logical call retries.
                time.sleep(delay)
            continue
        try:
            on_attempt(
                attempt_number=attempt_index + 1,
                result=result,
                exc=None,
                classification=None,
                will_retry=False,
                retry_delay_seconds=0.0,
                retry_delay_source=None,
                latency_seconds=time.monotonic() - started,
                attempt_timeout_seconds=attempt_timeout,
                cooldown_wait_seconds=cooldown_wait,
            )
        except Exception:
            pass
        return result
    if last_exc is not None:  # pragma: no cover - loop always returns/raises
        raise last_exc
    raise FireworksDeadlineExceeded("Fireworks policy runner made no attempt")


def _tool_field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _decode_forced_tool_call(
    message: object,
    contract: FireworksToolContract,
) -> dict[str, Any]:
    calls = _tool_field(message, "tool_calls") or []
    if not isinstance(calls, (list, tuple)) or len(calls) != 1:
        raise RuntimeError(
            f"Fireworks tool contract expected exactly one call to {contract.name!r}"
        )
    function = _tool_field(calls[0], "function")
    name = _tool_field(function, "name")
    if name != contract.name:
        raise RuntimeError(
            f"Fireworks tool contract expected {contract.name!r}, received {name!r}"
        )
    arguments = _tool_field(function, "arguments")
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Fireworks tool arguments were not valid JSON") from exc
    elif isinstance(arguments, dict):
        decoded = dict(arguments)
    else:
        raise RuntimeError("Fireworks tool arguments were missing")
    if not isinstance(decoded, dict):
        raise RuntimeError("Fireworks tool arguments must decode to an object")
    return decoded


_FIREWORKS_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _fireworks_usage_status(usage: dict[str, int | None] | None) -> str:
    if not usage or not any(usage.get(key) is not None for key in _FIREWORKS_USAGE_KEYS):
        return "unavailable"
    if all(usage.get(key) is not None for key in _FIREWORKS_USAGE_KEYS):
        return "measured"
    return "partial"


def _aggregate_fireworks_attempt_usage(
    attempt_usages: list[dict[str, int | None] | None],
) -> tuple[dict[str, int | None], str]:
    """Sum each known token category once and retain unknowns as null.

    A failed attempt with no provider usage makes the logical-call total a
    lower bound, but must not erase token counts measured on other attempts.
    Likewise, an unknown cache split makes only the affected input/cache
    categories null; known output tokens still contribute to the lower bound.
    """

    aggregate: dict[str, int | None] = {}
    for key in _FIREWORKS_USAGE_KEYS:
        known_values = [
            int(item[key])
            for item in attempt_usages
            if item is not None and item.get(key) is not None
        ]
        aggregate[key] = sum(known_values) if known_values else None

    if not any(value is not None for value in aggregate.values()):
        return aggregate, "unavailable"
    if attempt_usages and all(
        item is not None and _fireworks_usage_status(item) == "measured"
        for item in attempt_usages
    ):
        return aggregate, "measured"
    return aggregate, "partial"


@observe(as_type="generation", name="fireworks_primary")
def _fireworks_primary_chat(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    expect_json: bool,
    max_tokens: int,
    usage_context: dict | None,
    capture: dict | None,
    label: str,
    prompt_cache: bool,
    timeout_seconds: float | None = None,
    policy: FireworksStagePolicy | None = None,
    tool_contract: FireworksToolContract | None = None,
    reasoning_effort: FireworksReasoningEffort | None = None,
) -> str | dict:
    """Primary-grade Fireworks (OpenAI-compatible) chat call.

    The provider-dispatch half of the GLM promotion (item 19): tier envs may
    name a Fireworks model id ("accounts/...") and the ``opus_llm`` family,
    ``facial_llm``, and ``cheap_llm`` route here instead of raising. Mirrors
    ``_shadow_llm_call``'s wire format (same vendor temperature calibration,
    same reasoning_content capture, same typed fireworks usage receipts with
    cached-token pricing) but with PRIMARY posture, which differs in exactly
    two ways:

    - Full ``_retry_with_backoff`` (the 5-attempt primary doctrine) instead
      of the shadow's at-most-one-retry. The SDK's own retries stay disabled
      (``max_retries=0``) so attempts can never stack under the wrapper —
      the 2026-07-05 238s+ retry-storm class.
    - A loud truncation raise mirroring the Anthropic primaries'
      non-``end_turn`` raise: a primary must fail into its caller's
      retry/abort path, never hand back silently-clipped output.

    Anthropic ``cache_control`` has no request-side Fireworks equivalent and
    needs none: Fireworks prefix-caches automatically on byte-identical
    prompt prefixes (see ``_shadow_llm_call``'s prefix-cache note), so the
    cached and uncached entry points converge here; ``prompt_cache`` only
    stamps the usage receipt.

    ``capture`` is filled with the SAME keys the Anthropic primaries fill
    (``thinking_summary``/``stop_reason``) so consumers stay
    provider-agnostic: reasoning_content is the OpenAI-compat analog of the
    summarized-thinking channel, finish_reason of stop_reason. Filled before
    the truncation raise, same as ``opus_llm``.
    """
    # Reasoning headroom floor: callers pass Anthropic-calibrated caps, but
    # Fireworks counts chain-of-thought against max_tokens — see the config
    # knob's note (five finish_reason=length failures on the first
    # GLM-primary session). Floored before the request so the usage receipt
    # records what was actually sent.
    max_tokens = max(max_tokens, config.FIREWORKS_PRIMARY_MIN_MAX_TOKENS)

    request_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else config.SHADOW_LLM_TIMEOUT_SECONDS
    )

    client = get_llm_client("fireworks", config.FIREWORKS_API_KEY, request_timeout)

    request_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": SHADOW_LLM_TEMPERATURE,
    }
    requested_reasoning_effort = (
        policy.reasoning_effort
        if policy is not None and policy.reasoning_effort is not None
        else reasoning_effort
    )
    if policy is not None:
        if policy.prompt_cache_key:
            request_kwargs["prompt_cache_key"] = policy.prompt_cache_key
    if requested_reasoning_effort is not None:
        # Fireworks supports literal "max" but OpenAI 2.44's public type
        # alias does not. extra_body preserves the provider's literal wire
        # value rather than translating it to the SDK's "xhigh" spelling.
        request_kwargs["extra_body"] = {
            "reasoning_effort": requested_reasoning_effort
        }
    if tool_contract is not None:
        request_kwargs["tools"] = [tool_contract.tool_spec()]
        request_kwargs["tool_choice"] = tool_contract.forced_choice()
        request_kwargs["parallel_tool_calls"] = False

    policy_deadline_at: float | None = None

    def _call(attempt_timeout: float) -> _FireworksResponseEnvelope:
        bounded_request = {**request_kwargs, "timeout": attempt_timeout}
        if policy is not None and policy.response_transport == "stream":
            if policy_deadline_at is None:  # pragma: no cover - runner invariant
                raise RuntimeError("streaming policy deadline was not initialized")
            return _create_fireworks_streaming_completion(
                client,
                bounded_request,
                deadline_at=policy_deadline_at,
            )
        return _create_fireworks_completion(client, bounded_request)

    request = {
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "max_tokens": max_tokens,
        "expect_json": bool(expect_json),
        "temperature": SHADOW_LLM_TEMPERATURE,
        "stable_prefix_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "serving_path": fireworks_serving_path(model),
        "response_transport": (
            policy.response_transport if policy is not None else "complete"
        ),
    }
    if prompt_cache:
        request["prompt_cache"] = "fireworks-auto"

    if requested_reasoning_effort is not None:
        request.update(
            {
                "reasoning_effort_requested": requested_reasoning_effort,
                "reasoning_effort_effective": effective_reasoning_effort(
                    model, requested_reasoning_effort
                ),
            }
        )

    if policy is not None:
        request.update(
            {
                "policy_stage": policy.stage,
                "attempt_timeout_seconds": policy.attempt_timeout_seconds,
                "total_deadline_seconds": policy.total_deadline_seconds,
                "max_attempts": policy.max_attempts,
                "prompt_cache_key": policy.prompt_cache_key,
            }
        )
    if tool_contract is not None:
        request["tool_contract"] = tool_contract.name
        request["tool_schema_sha256"] = hashlib.sha256(
            json.dumps(
                tool_contract.tool_spec(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    call_context = usage_context if usage_context is not None else {}
    logical_call_id = str(call_context.get("logical_call_id") or "").strip()
    if not logical_call_id:
        logical_call_id = f"llm_{uuid.uuid4().hex}"
        call_context["logical_call_id"] = logical_call_id
    if policy is not None:
        call_context.setdefault("stage", policy.stage)

    attempt_usages: list[dict[str, int | None] | None] = []
    input_token_upper_bound = _fireworks_input_token_upper_bound(request_kwargs)
    from shared.llm_spend_budget import (
        FIREWORKS_SPEND_COHORT_CONTEXT_KEY,
        synchronize_fireworks_spend_cohort,
    )

    spend_cohort = call_context.get(FIREWORKS_SPEND_COHORT_CONTEXT_KEY)
    if spend_cohort is not None and not isinstance(spend_cohort, threading.Barrier):
        raise RuntimeError("invalid Fireworks spend reservation cohort")
    try:
        spend_reservation = reserve_fireworks_spend(
            cap_usd=config.FIREWORKS_PRIMARY_MAX_COST_USD,
            model=model,
            input_token_upper_bound=input_token_upper_bound,
            max_output_tokens=max_tokens,
            max_attempts=policy.max_attempts if policy else _MAX_RETRIES,
        )
    except BaseException:
        if spend_cohort is not None:
            try:
                spend_cohort.abort()
            except Exception:
                pass
        raise
    synchronize_fireworks_spend_cohort(
        spend_reservation,
        spend_cohort,
        timeout_seconds=min(
            30.0,
            policy.total_deadline_seconds if policy is not None else request_timeout,
        ),
    )
    if spend_reservation is not None:
        request.update(
            {
                "spend_cap_usd": spend_reservation.cap_usd,
                "spend_reserved_usd": spend_reservation.reserved_usd,
                "input_token_upper_bound": input_token_upper_bound,
            }
        )

    def _attempt_observer(**event: Any) -> None:
        envelope = event.get("result")
        exc = event.get("exc")
        response_obj = envelope.response if isinstance(envelope, _FireworksResponseEnvelope) else None
        has_usage = response_obj is not None and getattr(response_obj, "usage", None) is not None
        headers = (
            envelope.headers
            if isinstance(envelope, _FireworksResponseEnvelope)
            else (
                _fireworks_error_headers(exc)
                or getattr(exc, "_cloris_fireworks_headers", {})
            )
            if isinstance(exc, BaseException)
            else {}
        )
        stream_metadata = (
            envelope.stream_metadata
            if isinstance(envelope, _FireworksResponseEnvelope)
            else getattr(exc, "_cloris_stream_metadata", {})
            if isinstance(exc, BaseException)
            else {}
        ) or {}
        normalized_usage = (
            fireworks_shadow_usage_dict(response_obj, provider_headers=headers)
            if has_usage
            else None
        )
        attempt_usages.append(normalized_usage)
        classification = event.get("classification")
        status_code = (
            classification.get("status_code")
            if isinstance(classification, dict)
            else getattr(classification, "status_code", None)
        )
        reason = (
            classification.get("reason")
            if isinstance(classification, dict)
            else getattr(classification, "reason", None)
        )
        metadata = {
            "serving_path": fireworks_serving_path(model),
            "policy_stage": policy.stage if policy else None,
            "reasoning_effort_requested": requested_reasoning_effort,
            "reasoning_effort_effective": effective_reasoning_effort(
                model, requested_reasoning_effort
            ),
            "response_transport": request["response_transport"],
            "attempt_timeout_seconds": event.get(
                "attempt_timeout_seconds", request_timeout
            ),
            "total_deadline_seconds": policy.total_deadline_seconds if policy else None,
            "latency_ms": round(float(event.get("latency_seconds") or 0.0) * 1000, 3),
            "cooldown_wait_ms": round(
                float(event.get("cooldown_wait_seconds") or 0.0) * 1000, 3
            ),
            "http_status": status_code,
            "failure_reason": reason,
            "error_type": type(exc).__name__ if isinstance(exc, BaseException) else None,
            "retry_decision": "retry" if event.get("will_retry") else "stop",
            "retry_delay_seconds": event.get("retry_delay_seconds") or 0.0,
            "retry_delay_source": event.get("retry_delay_source"),
            "provider_request_id": (
                envelope.request_id
                if isinstance(envelope, _FireworksResponseEnvelope)
                else (
                    getattr(exc, "request_id", None)
                    or getattr(exc, "_cloris_provider_request_id", None)
                )
            ),
            "response_id": _fireworks_response_id(response_obj),
            "finish_reason": _fireworks_response_finish_reason(response_obj),
            "provider_headers": headers,
            "stable_prefix_sha256": request["stable_prefix_sha256"],
            "prompt_cache_key": policy.prompt_cache_key if policy else None,
            "tool_schema_sha256": request.get("tool_schema_sha256"),
            **stream_metadata,
            **(
                _fireworks_timeout_metadata(exc)
                if isinstance(exc, BaseException)
                else {}
            ),
        }
        record_llm_attempt(
            provider="fireworks",
            model=model,
            logical_call_id=logical_call_id,
            attempt_number=int(event["attempt_number"]),
            max_attempts=policy.max_attempts if policy else _MAX_RETRIES,
            status="response_received" if response_obj is not None else (
                "retryable_error" if event.get("will_retry") else "terminal_error"
            ),
            usage=normalized_usage,
            usage_status=_fireworks_usage_status(normalized_usage),
            usage_context=call_context,
            metadata=metadata,
        )

    try:
        if policy is None:
            envelope = _retry_with_backoff(
                lambda: _call(request_timeout),
                label=label,
                on_attempt=_attempt_observer,
            )
        else:
            policy_deadline_at = time.monotonic() + policy.total_deadline_seconds
            envelope = _run_fireworks_with_policy(
                _call,
                model=model,
                label=label,
                policy=policy,
                on_attempt=_attempt_observer,
            )
    except BaseException as exc:
        observed = [item for item in attempt_usages if item is not None]
        aggregate, aggregate_status = _aggregate_fireworks_attempt_usage(
            attempt_usages
        )
        settle_fireworks_spend(
            spend_reservation,
            measured_cost_usd=_measured_fireworks_cost(
                model, aggregate, aggregate_status
            ),
            usage_complete=aggregate_status == "measured",
        )
        request.update(
            {
                "logical_call_id": logical_call_id,
                "attempt_count": len(attempt_usages),
                "measured_attempt_count": len(observed),
                "error_type": type(exc).__name__,
                **(getattr(exc, "_cloris_stream_metadata", {}) or {}),
                **_fireworks_timeout_metadata(exc),
            }
        )
        record_llm_usage(
            provider="fireworks",
            model=model,
            usage=aggregate,
            request=request,
            usage_context=call_context,
            actual_status="error",
            usage_status=aggregate_status,
        )
        _stash_langfuse_context(call_context)
        raise

    response = envelope.response
    request.update(envelope.stream_metadata or {})
    usage = fireworks_shadow_usage_dict(
        response,
        provider_headers=envelope.headers,
    )
    choices = getattr(response, "choices", None) or []
    finish_reason = (
        getattr(choices[0], "finish_reason", None) if choices else None
    )
    request["finish_reason"] = finish_reason
    request["stop_reason"] = finish_reason
    request["logical_call_id"] = logical_call_id
    request["provider_request_id"] = envelope.request_id
    request["response_id"] = _fireworks_response_id(response)
    request["provider_headers"] = envelope.headers

    raw_content = ""
    reasoning_content = None
    if choices:
        msg = getattr(choices[0], "message", None)
        raw_content = getattr(msg, "content", "") or ""
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content is None:
            extra = getattr(msg, "model_extra", None)
            if isinstance(extra, dict):
                reasoning_content = extra.get("reasoning_content")
    if capture is not None:
        # Legacy calls retain their established capture contract. Explicit
        # judgment-policy/tool experiments never expose provider chain of
        # thought to callers, logs, replay artifacts, or the console.
        capture["thinking_summary"] = (
            reasoning_content
            if policy is None
            and tool_contract is None
            and isinstance(reasoning_content, str)
            else None
        )
        capture["stop_reason"] = finish_reason

    actual_status = "ok"
    try:
        if not choices:
            raise RuntimeError(f"{label} response contained no choices ({model}).")
        expected_finish = "tool_calls" if tool_contract is not None else "stop"
        if (
            tool_contract is not None
            and str(finish_reason) != expected_finish
        ) or (
            tool_contract is None
            and finish_reason is not None
            and str(finish_reason) != expected_finish
        ):
            raise RuntimeError(
                f"{label} response truncated: finish_reason={finish_reason}. "
                "Increase max_tokens or reduce prompt size."
            )
        message = getattr(choices[0], "message", None)
        if tool_contract is not None:
            result: str | dict = _decode_forced_tool_call(message, tool_contract)
        else:
            text = (
                raw_content if isinstance(raw_content, str) else str(raw_content)
            ).strip()
            result = _parse_json_response(text) if expect_json else text
    except BaseException as exc:
        actual_status = (
            "postcondition_fail" if isinstance(exc, Exception) else "error"
        )
        raise
    finally:
        observed = [item for item in attempt_usages if item is not None]
        aggregate_usage, aggregate_status = _aggregate_fireworks_attempt_usage(
            attempt_usages
        )
        settle_fireworks_spend(
            spend_reservation,
            measured_cost_usd=_measured_fireworks_cost(
                model, aggregate_usage or usage, aggregate_status
            ),
            usage_complete=aggregate_status == "measured",
        )
        request["attempt_count"] = len(attempt_usages)
        request["measured_attempt_count"] = len(observed)
        record_llm_usage(
            provider="fireworks",
            model=model,
            usage=aggregate_usage or usage,
            request=request,
            usage_context=call_context,
            actual_status=actual_status,
            usage_status=aggregate_status,
        )
        _stash_langfuse_context(call_context)
    return result


def _shadow_llm_call(
    *,
    model: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    usage_context: dict | None,
    capture: dict | None = None,
) -> str:
    """Shared call layer for the GLM-5.2 shadow-judge models (Fireworks).

    Both ``shadow_facial_llm`` and ``shadow_full_llm`` are thin wrappers
    around this function — one experiment (SHADOW_FACIAL_MODEL_ENABLED),
    two judgment tiers reading the SAME model/base_url config
    (``SHADOW_FACIAL_MODEL_NAME`` / ``SHADOW_FACIAL_BASE_URL`` — names are
    facial-era but apply to both tiers; no second config knob was added).

    SHADOW ONLY: this function's return value is compared against, and
    never influences, the primary decision. Callers (shared/judger.py) are
    the ones enforcing that boundary and must catch every exception this
    raises — this function itself does not fail-soft.

    Mirrors ``_call_openai``'s structure (OpenAI-compatible client, same
    request/usage-record shape) but is deliberately NOT routed through
    ``_retry_with_backoff`` — a shadow call must never meaningfully slow
    the real run, so it gets at most one retry (two attempts total) with
    no exponential backoff sleep, instead of the primary path's 5-attempt
    backoff loop.

    No JSON mode: the live facial/full-eval contracts
    (assemble_facial_system/parse_facial_response,
    assemble_full_evaluation_system/parse_full_evaluation_response) are
    plain structured text, not JSON — forcing
    ``response_format={"type": "json_object"}`` here would change the wire
    format relative to what the primary Opus call actually receives,
    contaminating the parse-failure-rate instrument this seam exists to
    measure. Returns the raw stripped text; the caller parses it with the
    exact same parser used for the primary verdict.

    Prefix-cache prerequisite (verified 2026-07-03): Fireworks' automatic
    prefix caching (enabled by default, no request-side opt-in) only hits
    when successive calls share a byte-identical prompt PREFIX. Both
    ``assemble_facial_system`` and ``assemble_full_evaluation_system``
    (shared/judgment/templates.py) are assembled entirely from static
    ``Brief`` fields — no timestamps, candidate names, or other
    per-candidate content leaks into either system prompt (confirmed by
    inspection: neither judgment_templates.py nor shared/brief_schema.py
    references ``datetime``/``time.time``/``uuid``/``random``). The system
    prompt is therefore byte-identical across every candidate judged
    against the same brief within a run, on both tiers, which is exactly
    the invariant Fireworks' cache needs to actually hit repeatedly (the
    same precondition Anthropic's own ``cache_control: ephemeral`` system
    block relies on in ``opus_llm_cached``/``facial_llm``). Only the user
    prompt (candidate snippet / profile text) varies per call, as
    expected — caching a system-prompt-only prefix.

    Always recorded via ``record_llm_usage(provider="fireworks", ...)``,
    using :func:`shared.llm_usage.fireworks_shadow_usage_dict` (NOT
    :func:`shared.llm_usage.openai_usage_dict`) so a cached-token count
    Fireworks reports is (a) actually read and (b) priced at the cached
    rate instead of being silently absorbed into the full-price input
    count — see that function's docstring for the confirmed field name/
    doc citations. Usage lands in token-cost-log.jsonl in the same place
    as every other typed LLM receipt, on both the success and error paths.
    """
    client = get_llm_client(
        "shadow_fireworks",
        config.FIREWORKS_API_KEY,
        config.SHADOW_LLM_TIMEOUT_SECONDS,
        base_url=base_url,
    )

    def _call():
        return client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=SHADOW_LLM_TEMPERATURE,
            timeout=config.SHADOW_LLM_TIMEOUT_SECONDS,
        )

    request = {
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "max_tokens": max_tokens,
        "temperature": SHADOW_LLM_TEMPERATURE,
    }

    def _call_with_one_retry():
        try:
            return _call()
        except Exception as first_exc:
            if not classify_runtime_failure(first_exc, source="llm").retryable:
                raise
            return _call()

    try:
        response = _call_with_one_retry()
    except Exception as exc:
        _record_llm_error(
            provider="fireworks",
            model=model,
            request=request,
            usage_context=usage_context,
            exc=exc,
        )
        raise

    usage = fireworks_shadow_usage_dict(response)
    choices = getattr(response, "choices", None) or []
    # Parse-failure post-mortems need the provider's own stop signal: a
    # finish_reason of "length" separates "GLM couldn't hold the format"
    # from "we cut it off mid-response" without re-running anything
    # (2026-07-04 SPL live capture: full-eval PARSE_FAILUREs were
    # undiagnosable because neither the raw text nor the stop signal was
    # persisted). Stamped onto the request dict because record_llm_usage
    # spreads **request flat into the token-cost-log row.
    request["finish_reason"] = (
        getattr(choices[0], "finish_reason", None) if choices else None
    )
    record_llm_usage(
        provider="fireworks",
        model=model,
        usage=usage,
        request=request,
        usage_context=usage_context,
        usage_status=_fireworks_usage_status(usage),
    )
    _stash_langfuse_context(usage_context)

    raw_content = ""
    reasoning_content = None
    if choices:
        msg = getattr(choices[0], "message", None)
        raw_content = getattr(msg, "content", "") or ""
        # Reasoning models on the OpenAI-compat surface (GLM/DeepSeek style)
        # return their chain-of-thought in a separate reasoning field the
        # judgment parser never sees. Capture it for the monitoring channel;
        # absent on non-reasoning models, harmless either way.
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content is None:
            extra = getattr(msg, "model_extra", None)
            if isinstance(extra, dict):
                reasoning_content = extra.get("reasoning_content")
    if capture is not None:
        capture["reasoning_content"] = (
            reasoning_content if isinstance(reasoning_content, str) else None
        )
        capture["finish_reason"] = request.get("finish_reason")
    return raw_content.strip() if isinstance(raw_content, str) else str(raw_content).strip()


@observe(as_type="generation", name="shadow_facial_llm")
def shadow_facial_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    usage_context: dict | None = None,
    capture: dict | None = None,
) -> str:
    """Call the GLM-5.2 shadow-judge model (served by Fireworks) for A/B
    comparison against the real FACIAL verdict. See ``_shadow_llm_call``
    for the shared implementation/doctrine both shadow tiers share.
    """
    return _shadow_llm_call(
        model=config.SHADOW_FACIAL_MODEL_NAME,
        base_url=config.SHADOW_FACIAL_BASE_URL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        usage_context=usage_context,
        capture=capture,
    )


@observe(as_type="generation", name="shadow_full_llm")
def shadow_full_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 8192,
    usage_context: dict | None = None,
    capture: dict | None = None,
) -> str:
    """Call the GLM-5.2 shadow-judge model (served by Fireworks) for A/B
    comparison against the real FULL-EVALUATION verdict. Sibling to
    ``shadow_facial_llm`` — same model/base_url/flag/one-retry doctrine
    (see ``_shadow_llm_call``), default ``max_tokens`` raised to 8192 to
    mirror ``full_judge``'s primary call (``opus_llm_cached`` with its
    default ``max_tokens=8192``) since full-eval rationale runs longer
    than facial's short verdict.
    """
    return _shadow_llm_call(
        model=config.SHADOW_FACIAL_MODEL_NAME,
        base_url=config.SHADOW_FACIAL_BASE_URL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        usage_context=usage_context,
        capture=capture,
    )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_anthropic_cheap(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool,
    usage_context: dict | None = None,
    model_name: str | None = None,
) -> str | dict | list:
    model = model_name or config.CHEAP_MODEL_NAME
    client = get_llm_client("anthropic", config.ANTHROPIC_API_KEY, 120.0)

    def _call():
        message = client.messages.create(
            model=model,
            max_tokens=config.CHEAP_MODEL_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message

    request = {
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "max_tokens": config.CHEAP_MODEL_MAX_TOKENS,
        "expect_json": bool(expect_json),
    }
    try:
        message = _retry_with_backoff(_call, label="Anthropic-cheap")
    except Exception as exc:
        _record_llm_error(
            provider="anthropic",
            model=model,
            request=request,
            usage_context=usage_context,
            exc=exc,
        )
        raise
    # P4.3.3: mark truncation into usage_context BEFORE record_llm_usage
    # reads it — record_llm_usage snapshots usage_context into the
    # persisted JSONL row at call time, so the marker must land first or
    # it never reaches the log.
    _mark_if_truncated(
        provider="anthropic",
        model=model,
        finish_reason=getattr(message, "stop_reason", None),
        usage_context=usage_context,
    )
    record_llm_usage(
        provider="anthropic",
        model=model,
        usage=anthropic_usage_dict(message),
        request=request,
        usage_context=usage_context,
    )
    _stash_langfuse_context(usage_context)
    text = _first_text_block(message).strip()

    if expect_json:
        return _parse_json_response(text)
    return text


def _call_minimax_cheap(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool,
    usage_context: dict | None = None,
) -> str | dict | list:
    """Call MiniMax M3 for extraction with reasoning explicitly disabled."""

    if not config.MINIMAX_API_KEY.strip():
        raise RuntimeError(
            "MiniMax extraction requires MINIMAX_M3_API_KEY "
            "(or the MINIMAX_API_KEY alias)"
        )
    client = get_llm_client("minimax", config.MINIMAX_API_KEY, 60.0)

    def _call():
        return client.chat.completions.create(
            model=config.CHEAP_MODEL_NAME,
            max_completion_tokens=8192,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            extra_body={
                "thinking": {"type": "disabled"},
                "service_tier": "standard",
            },
        )

    request = {
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "max_tokens": 8192,
        "temperature": 0.1,
        "expect_json": bool(expect_json),
        "thinking": "disabled",
        "service_tier": "standard",
        "max_attempts": config.MINIMAX_CHEAP_MAX_ATTEMPTS,
    }
    try:
        response = _retry_with_backoff(
            _call,
            label="MiniMax-cheap",
            max_attempts=config.MINIMAX_CHEAP_MAX_ATTEMPTS,
        )
    except Exception as exc:
        _record_llm_error(
            provider="minimax",
            model=config.CHEAP_MODEL_NAME,
            request=request,
            usage_context=usage_context,
            exc=exc,
        )
        raise

    choices = getattr(response, "choices", None) or []
    finish_reason = (
        getattr(choices[0], "finish_reason", None) if choices else None
    )
    _mark_if_truncated(
        provider="minimax",
        model=config.CHEAP_MODEL_NAME,
        finish_reason=finish_reason,
        usage_context=usage_context,
    )
    record_llm_usage(
        provider="minimax",
        model=config.CHEAP_MODEL_NAME,
        usage=minimax_usage_dict(response),
        request={**request, "finish_reason": finish_reason},
        usage_context=usage_context,
    )
    _stash_langfuse_context(usage_context)
    if str(finish_reason or "").lower() in _TRUNCATION_TOKENS:
        raise RuntimeError(
            f"MiniMax extraction truncated: finish_reason={finish_reason}"
        )

    message = getattr(choices[0], "message", None) if choices else None
    raw_content = getattr(message, "content", "") or ""
    text = raw_content.strip() if isinstance(raw_content, str) else str(raw_content).strip()
    if expect_json:
        return _parse_json_response(text)
    return text


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool,
    usage_context: dict | None = None,
) -> str | dict | list:
    client = get_llm_client("openai", config.OPENAI_API_KEY, 60.0)
    kwargs = {}
    if expect_json:
        kwargs["response_format"] = {"type": "json_object"}

    def _call():
        response = client.chat.completions.create(
            model=config.CHEAP_MODEL_NAME,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            timeout=120.0,
            **kwargs,
        )
        return response

    request = {
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "max_tokens": 8192,
        "temperature": 0.1,
        "expect_json": bool(expect_json),
    }
    try:
        response = _retry_with_backoff(_call, label="OpenAI")
    except Exception as exc:
        _record_llm_error(
            provider="openai",
            model=config.CHEAP_MODEL_NAME,
            request=request,
            usage_context=usage_context,
            exc=exc,
        )
        raise

    usage = openai_usage_dict(response)
    if usage["input_tokens"] == 0 and usage["output_tokens"] == 0:
        print(
            "    [cost] OpenAI response missing token counts (usage); cost estimate may be $0.",
            flush=True,
        )
    choices = getattr(response, "choices", None) or []
    # P4.3.3: mark truncation into usage_context BEFORE record_llm_usage
    # reads it — see _call_anthropic_cheap for why the ordering matters.
    _mark_if_truncated(
        provider="openai",
        model=config.CHEAP_MODEL_NAME,
        finish_reason=getattr(choices[0], "finish_reason", None) if choices else None,
        usage_context=usage_context,
    )
    record_llm_usage(
        provider="openai",
        model=config.CHEAP_MODEL_NAME,
        usage=usage,
        request=request,
        usage_context=usage_context,
    )
    _stash_langfuse_context(usage_context)

    raw_content = ""
    if choices:
        msg = getattr(choices[0], "message", None)
        raw_content = getattr(msg, "content", "") or ""
    text = raw_content.strip() if isinstance(raw_content, str) else str(raw_content).strip()

    if expect_json:
        return _parse_json_response(text)
    return text


def _call_google(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool,
    usage_context: dict | None = None,
) -> str | dict | list:
    from google import genai

    client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def _call():
        response = client.models.generate_content(
            model=GOOGLE_CHEAP_MODEL_NAME,
            contents=f"{system_prompt}\n\n{user_prompt}",
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json" if expect_json else "text/plain",
            },
        )
        return response

    request = {
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "temperature": 0.1,
        "expect_json": bool(expect_json),
    }
    try:
        response = _retry_with_backoff(_call, label="Google")
    except Exception as exc:
        _record_llm_error(
            provider="google",
            model=GOOGLE_CHEAP_MODEL_NAME,
            request=request,
            usage_context=usage_context,
            exc=exc,
        )
        raise

    usage = google_usage_dict(response)
    if usage["input_tokens"] == 0 and usage["output_tokens"] == 0:
        print(
            "    [cost] Google response missing token counts (usage_metadata); cost estimate may be $0.",
            flush=True,
        )
    candidates = getattr(response, "candidates", None) or []
    # P4.3.3: mark truncation into usage_context BEFORE record_llm_usage
    # reads it — see _call_anthropic_cheap for why the ordering matters.
    _mark_if_truncated(
        provider="google",
        model=GOOGLE_CHEAP_MODEL_NAME,
        finish_reason=getattr(candidates[0], "finish_reason", None) if candidates else None,
        usage_context=usage_context,
    )
    record_llm_usage(
        provider="google",
        model=GOOGLE_CHEAP_MODEL_NAME,
        usage=usage,
        request=request,
        usage_context=usage_context,
    )
    _stash_langfuse_context(usage_context)

    raw_content = getattr(response, "text", "") or ""
    text = raw_content.strip() if isinstance(raw_content, str) else str(raw_content).strip()

    if expect_json:
        return _parse_json_response(text)
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> dict | list:
    """Parse JSON from LLM response, handling markdown code fences."""
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to find JSON object or array in the text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        # Last resort: try to close truncated JSON by finding the last complete key-value
        # and closing any open braces/brackets
        if text.strip().startswith("{"):
            # Truncate to last complete string value (ends with ")
            last_quote = text.rfind('"')
            if last_quote > 0:
                truncated = text[:last_quote + 1]
                # Count open/close braces to close properly
                open_braces = truncated.count("{") - truncated.count("}")
                open_brackets = truncated.count("[") - truncated.count("]")
                truncated += "}" * open_braces + "]" * open_brackets
                try:
                    return json.loads(truncated)
                except json.JSONDecodeError:
                    pass
        raise RuntimeError(f"Could not parse JSON from LLM response: {text[:500]}") from e
