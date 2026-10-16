"""Judgment runtime configuration and its stable diagnostic profile."""
from __future__ import annotations
import hashlib
import json
import math
import os
from shared import config

def _validate_judgment_runtime_configuration() -> None:
    """Fail before Chrome work when judgment knobs form an unsafe profile."""

    if int(config.LINKEDIN_TOTAL_PAGE_CAP) < 0:
        raise RuntimeError("LINKEDIN_TOTAL_PAGE_CAP must be >= 0")
    if (
        config.LINKEDIN_PAGE_ALLOCATOR_MODE == "active"
        and int(config.LINKEDIN_TOTAL_PAGE_CAP) <= 0
    ):
        raise RuntimeError(
            "active page allocation requires LINKEDIN_TOTAL_PAGE_CAP > 0"
        )

    facial_contract = str(config.LINKEDIN_V2_FACIAL_CONTRACT).lower()
    full_contract = str(config.LINKEDIN_V2_FULL_CONTRACT).lower()
    for name, value in (
        ("LINKEDIN_V2_FACIAL_CONTRACT", facial_contract),
        ("LINKEDIN_V2_FULL_CONTRACT", full_contract),
    ):
        if value not in {"legacy", "tool"}:
            raise RuntimeError(f"{name} must be 'legacy' or 'tool'; got {value!r}")

    if config.LINKEDIN_FACIAL_MAX_CONCURRENCY < 1:
        raise RuntimeError("LINKEDIN_FACIAL_MAX_CONCURRENCY must be >= 1")
    if config.LINKEDIN_FACIAL_MAX_CONCURRENCY > 3:
        raise RuntimeError("LINKEDIN_FACIAL_MAX_CONCURRENCY must be <= 3")
    if config.LINKEDIN_FACIAL_TARGET_BATCH_SIZE < 1:
        raise RuntimeError("LINKEDIN_FACIAL_TARGET_BATCH_SIZE must be >= 1")

    policy_enabled = bool(config.FIREWORKS_JUDGMENT_POLICY_ENABLED)
    if (facial_contract == "tool" or full_contract == "tool") and not policy_enabled:
        raise RuntimeError(
            "LinkedIn tool judgment contracts require "
            "FIREWORKS_JUDGMENT_POLICY_ENABLED=true"
        )
    if config.FIREWORKS_PROMPT_AFFINITY_ENABLED and not policy_enabled:
        raise RuntimeError(
            "FIREWORKS_PROMPT_AFFINITY_ENABLED requires the explicit "
            "Fireworks judgment policy"
        )
    if (
        config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED
        and config.LINKEDIN_FACIAL_MAX_CONCURRENCY > 1
    ):
        if facial_contract != "tool":
            raise RuntimeError(
                "facial concurrency >1 requires LINKEDIN_V2_FACIAL_CONTRACT=tool"
            )
        if not policy_enabled:
            raise RuntimeError(
                "facial concurrency >1 requires FIREWORKS_JUDGMENT_POLICY_ENABLED=true"
            )

    if facial_contract == "tool" and not config.FACIAL_MODEL_NAME.startswith("accounts/"):
        raise RuntimeError("facial tool contract requires a Fireworks model")
    if full_contract == "tool" and not config.FULL_EVAL_MODEL_NAME.startswith("accounts/"):
        raise RuntimeError("full tool contract requires a Fireworks model")

    if os.environ.get(
        "CLORIS_SKIP_STARTUP_VALIDATION", ""
    ).strip().lower() not in {"1", "true", "yes", "on"}:
        provider_keys = {
            "anthropic": ("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY),
            "fireworks": ("FIREWORKS_API_KEY", config.FIREWORKS_API_KEY),
            "google": ("GOOGLE_API_KEY", config.GOOGLE_API_KEY),
            "minimax": ("MINIMAX_API_KEY", config.MINIMAX_API_KEY),
            "openai": ("OPENAI_API_KEY", config.OPENAI_API_KEY),
        }
        required_providers = {
            str(config.CHEAP_MODEL_PROVIDER).strip().lower()
        }
        if (
            config.CHEAP_MODEL_PROVIDER == "minimax"
            and config.CHEAP_MODEL_FALLBACK_PROVIDER
        ):
            required_providers.add(config.CHEAP_MODEL_FALLBACK_PROVIDER)
        for model_name in (
            config.STRATEGY_MODEL_NAME,
            config.FACIAL_MODEL_NAME,
            config.FULL_EVAL_MODEL_NAME,
            config.OPUS_MODEL_NAME,
        ):
            required_providers.add(
                "fireworks"
                if str(model_name).startswith("accounts/")
                else "anthropic"
            )
        if config.SHADOW_STRATEGY_ENABLED:
            required_providers.add(
                "fireworks"
                if str(config.SHADOW_STRATEGY_MODEL_NAME).startswith(
                    "accounts/"
                )
                else "anthropic"
            )
        if config.SHADOW_FACIAL_MODEL_ENABLED:
            required_providers.add("fireworks")

        unsupported = sorted(
            provider
            for provider in required_providers
            if provider not in provider_keys
        )
        if unsupported:
            raise RuntimeError(
                "LinkedIn cannot start: unsupported model provider(s): "
                f"{', '.join(unsupported)}"
            )
        missing = sorted(
            provider_keys[provider][0]
            for provider in required_providers
            if not str(provider_keys[provider][1]).strip()
        )
        if missing:
            raise config.MissingRequiredKeyError(missing)

    if not policy_enabled:
        return
    if not config.FACIAL_MODEL_NAME.startswith("accounts/"):
        raise RuntimeError("Fireworks facial policy requires a Fireworks model")
    if not config.FULL_EVAL_MODEL_NAME.startswith("accounts/"):
        raise RuntimeError("Fireworks full policy requires a Fireworks model")

    for name, effort in (
        (
            "FIREWORKS_FACIAL_REASONING_EFFORT",
            config.FIREWORKS_FACIAL_REASONING_EFFORT,
        ),
        (
            "FIREWORKS_FULL_REASONING_EFFORT",
            config.FIREWORKS_FULL_REASONING_EFFORT,
        ),
    ):
        if str(effort).lower() not in {"high", "max"}:
            raise RuntimeError(f"{name} must explicitly be 'high' or 'max'")

    for name, value in (
        (
            "FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS",
            config.FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS,
        ),
        (
            "FIREWORKS_FACIAL_TOTAL_DEADLINE_SECONDS",
            config.FIREWORKS_FACIAL_TOTAL_DEADLINE_SECONDS,
        ),
        (
            "FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS",
            config.FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS,
        ),
        (
            "FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS",
            config.FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS,
        ),
    ):
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise RuntimeError(f"{name} must be finite and > 0")
    if not config.FIREWORKS_JUDGMENT_STREAM_ENABLED:
        for name, value in (
            (
                "FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS",
                config.FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS,
            ),
            (
                "FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS",
                config.FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS,
            ),
        ):
            if float(value) < 300.0:
                raise RuntimeError(
                    f"{name} must be >= 300.0 when "
                    "FIREWORKS_JUDGMENT_STREAM_ENABLED=false: non-streaming "
                    "spends the attempt timeout as wall clock, and the observed "
                    "transport tail exceeds 90s"
                )
    for name, value, maximum in (
        (
            "FIREWORKS_FACIAL_MAX_ATTEMPTS",
            config.FIREWORKS_FACIAL_MAX_ATTEMPTS,
            4,
        ),
        (
            "FIREWORKS_FULL_MAX_ATTEMPTS",
            config.FIREWORKS_FULL_MAX_ATTEMPTS,
            2,
        ),
    ):
        if int(value) < 1:
            raise RuntimeError(f"{name} must be >= 1")
        if int(value) > maximum:
            raise RuntimeError(
                f"{name} must be <= {maximum} for judgment policy calls"
            )


def _judgment_runtime_profile(brief_obj: object | None = None) -> dict[str, object]:
    """Non-secret, hash-stable model/runtime posture for run receipts.

    A live timing result is not attributable unless the state artifact can
    prove which model endpoints, contracts, deadlines, and concurrency
    controls were effective at process start.  Keep this compact and free
    of keys, prompts, candidate data, and ambient environment values.
    """

    _validate_judgment_runtime_configuration()
    new_brief = getattr(brief_obj, "_new_brief", None)
    ambiguity_posture = str(
        getattr(new_brief, "facial_ambiguity_posture", "") or ""
    ).strip().lower()
    if ambiguity_posture == "ternary":
        effective_ternary = True
    elif ambiguity_posture == "binary":
        effective_ternary = False
    else:
        effective_ternary = bool(config.LINKEDIN_FACIAL_BORDERLINE_ENABLED)
    profile: dict[str, object] = {
        "schema_version": "linkedin-judgment-runtime-v1",
        "fireworks_base_url": config.FIREWORKS_BASE_URL.rstrip("/"),
        "cheap_model_provider": config.CHEAP_MODEL_PROVIDER,
        "cheap_model": config.CHEAP_MODEL_NAME,
        "strategy_provider": (
            "fireworks"
            if config.STRATEGY_MODEL_NAME.startswith("accounts/")
            else "anthropic"
        ),
        "strategy_model": config.STRATEGY_MODEL_NAME,
        "opus_provider": (
            "fireworks"
            if config.OPUS_MODEL_NAME.startswith("accounts/")
            else "anthropic"
        ),
        "opus_model": config.OPUS_MODEL_NAME,
        "facial_provider": (
            "fireworks"
            if config.FACIAL_MODEL_NAME.startswith("accounts/")
            else "anthropic"
        ),
        "facial_model": config.FACIAL_MODEL_NAME,
        "full_eval_provider": (
            "fireworks"
            if config.FULL_EVAL_MODEL_NAME.startswith("accounts/")
            else "anthropic"
        ),
        "full_eval_model": config.FULL_EVAL_MODEL_NAME,
        "shadow_strategy_enabled": config.SHADOW_STRATEGY_ENABLED,
        "shadow_facial_enabled": config.SHADOW_FACIAL_MODEL_ENABLED,
        "fireworks_primary_spend_cap_usd": config.FIREWORKS_PRIMARY_MAX_COST_USD,
        "policy_enabled": config.FIREWORKS_JUDGMENT_POLICY_ENABLED,
        "facial_reasoning_effort": config.FIREWORKS_FACIAL_REASONING_EFFORT,
        "full_reasoning_effort": config.FIREWORKS_FULL_REASONING_EFFORT,
        "facial_attempt_timeout_seconds": config.FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS,
        "facial_total_deadline_seconds": config.FIREWORKS_FACIAL_TOTAL_DEADLINE_SECONDS,
        "facial_max_attempts": config.FIREWORKS_FACIAL_MAX_ATTEMPTS,
        "full_attempt_timeout_seconds": config.FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS,
        "full_total_deadline_seconds": config.FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS,
        "full_max_attempts": config.FIREWORKS_FULL_MAX_ATTEMPTS,
        "prompt_affinity_enabled": config.FIREWORKS_PROMPT_AFFINITY_ENABLED,
        "facial_contract": config.LINKEDIN_V2_FACIAL_CONTRACT,
        "full_contract": config.LINKEDIN_V2_FULL_CONTRACT,
        "facial_concurrency_enabled": config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED,
        "facial_max_concurrency": config.LINKEDIN_FACIAL_MAX_CONCURRENCY,
        "facial_target_batch_size": config.LINKEDIN_FACIAL_TARGET_BATCH_SIZE,
        "facial_ambiguity_posture": ambiguity_posture,
        "facial_ternary_effective": effective_ternary,
        "page_allocator_mode": config.LINKEDIN_PAGE_ALLOCATOR_MODE,
        "total_page_cap": config.LINKEDIN_TOTAL_PAGE_CAP,
        "full_eval_pipeline_enabled": config.FULL_EVAL_PIPELINE_ENABLED,
        "external_evidence_enabled": config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED,
        "remote_observability_disabled": str(
            os.environ.get("LANGFUSE_DISABLE", "") or ""
        ).strip().lower() in {"1", "true", "yes", "on"},
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    profile["fingerprint_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return profile
