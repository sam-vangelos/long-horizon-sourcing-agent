"""Configuration loader. Reads .env and provides typed access to settings.

The writable user-data .env takes precedence over project-root defaults.
Frozen applications place runtime output in the writable user-data directory."""

import os
from pathlib import Path
from dotenv import load_dotenv

from shared.user_data_dir import (
    cloris_user_data_dir,
    should_use_user_data_dir,
)

# Load user-data credentials first; project defaults must not overwrite them.
_PROJECT_ROOT_ENV: Path = Path(__file__).parent.parent / ".env"
if should_use_user_data_dir():
    _user_env_path = cloris_user_data_dir() / ".env"
    if _user_env_path.exists():
        load_dotenv(_user_env_path, override=False)
load_dotenv(_PROJECT_ROOT_ENV, override=False)


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_flag(key: str, default: str = "false") -> bool:
    return _optional(key, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_choice(key: str, default: str, allowed: set[str]) -> str:
    value = _optional(key, default).strip().lower()
    if value not in allowed:
        raise ValueError(f"{key} must be one of: {', '.join(sorted(allowed))}")
    return value


# --- API Keys (lazy — validated at call time in llm_clients, not at import) ---
# Per-agent keys (LINKEDIN_ or GITHUB_ prefix) override shared keys.
# Prefix is set by each agent's entry point before config is imported.
_AGENT_PREFIX: str = os.getenv("AGENT_KEY_PREFIX", "")

def _agent_key(key: str) -> str:
    """Return agent-prefixed key if set, otherwise fall back to shared key."""
    if _AGENT_PREFIX:
        prefixed = os.getenv(f"{_AGENT_PREFIX}_{key}", "")
        if prefixed:
            return prefixed
    return _optional(key, "")

# --- Which Anthropic account pays ---
#
# A run can bill one of two Anthropic accounts. Pasting a different key over
# ANTHROPIC_API_KEY would make the switch a key-shaped edit nobody can eyeball,
# and nothing downstream could say WHICH account paid for a run. So the source
# is selected BY NAME:
#
#   ANTHROPIC_API_KEY_SOURCE=company   (default) — _agent_key("ANTHROPIC_API_KEY"),
#       so LINKEDIN_ANTHROPIC_API_KEY still outranks the shared key when
#       AGENT_KEY_PREFIX=LINKEDIN.
#   ANTHROPIC_API_KEY_SOURCE=personal — PERSONAL_ANTHROPIC_API_KEY, read FLAT:
#       no agent-prefix lookup, and NO fallback to the default key. A silent
#       fallback would let spend land on the wrong account unnoticed (a
#       prefixed key once kept a campaign on the other account until an .env
#       prune caught it), so an empty personal key fails startup validation
#       loudly — see validate_startup_keys — rather than quietly billing the
#       other account.
#
# Every receipt and the launcher preflight carry the SOURCE, never the key:
# anthropic_key_fingerprint() below renders "<source>:…<last4>" and nothing more.
ANTHROPIC_API_KEY_SOURCE: str = _optional(
    "ANTHROPIC_API_KEY_SOURCE", "company"
).strip().lower()
if ANTHROPIC_API_KEY_SOURCE not in {"company", "personal"}:
    raise ValueError(
        "ANTHROPIC_API_KEY_SOURCE must be 'company' or 'personal'; got "
        f"{ANTHROPIC_API_KEY_SOURCE!r}"
    )

ANTHROPIC_API_KEY: str = (
    _optional("PERSONAL_ANTHROPIC_API_KEY", "")
    if ANTHROPIC_API_KEY_SOURCE == "personal"
    else _agent_key("ANTHROPIC_API_KEY")
)


def anthropic_key_fingerprint() -> str:
    """Name the Anthropic key in force without exposing it.

    Returns ``"<source>:…<last4>"``, or ``"<source>:MISSING"`` when the
    resolved key is empty. NEVER returns more than the last four characters —
    it is built for preflight output, run consoles, and receipts.
    """

    key = ANTHROPIC_API_KEY.strip()
    if not key:
        return f"{ANTHROPIC_API_KEY_SOURCE}:MISSING"
    return f"{ANTHROPIC_API_KEY_SOURCE}:…{key[-4:]}"


OPENAI_API_KEY: str = _agent_key("OPENAI_API_KEY")
GOOGLE_API_KEY: str = _agent_key("GOOGLE_API_KEY")
PERPLEXITY_API_KEY: str = _agent_key("PERPLEXITY_API_KEY")
EXA_API_KEY: str = _agent_key("EXA_API_KEY")
# Per-run hard cap on Exa API spend, enforced pre-dispatch by
# ``exa.cost_meter.ExaCostMeter``. 0 disables the guard for callers that
# carry their own governor — the same idiom as the Fireworks spend cap
# (``shared/llm_spend_budget.py:44-60``).
EXA_RUN_BUDGET_USD: float = float(_optional("EXA_RUN_BUDGET_USD", "5.00"))
MINIMAX_API_KEY: str = (
    _agent_key("MINIMAX_M3_API_KEY") or _agent_key("MINIMAX_API_KEY")
)
MINIMAX_BASE_URL: str = _optional("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
SUPABASE_ANON_KEY: str = _optional("SUPABASE_ANON_KEY", "")

CHEAP_MODEL_PROVIDER: str = _optional("CHEAP_MODEL_PROVIDER", "openai")
CHEAP_MODEL_FALLBACK_PROVIDER: str = _optional(
    "CHEAP_MODEL_FALLBACK_PROVIDER", ""
).strip().lower()
CHEAP_MODEL_FALLBACK_NAME: str = _optional(
    "CHEAP_MODEL_FALLBACK_NAME", ""
).strip()
MINIMAX_CHEAP_MAX_ATTEMPTS: int = int(
    _optional("MINIMAX_CHEAP_MAX_ATTEMPTS", "2")
)
# Output ceiling for cheap-tier extraction; large profiles need room for complete JSON.
CHEAP_MODEL_MAX_TOKENS: int = int(_optional("CHEAP_MODEL_MAX_TOKENS", "16384"))

# --- Startup key validation ---

_REQUIRED_AT_STARTUP: tuple[str, ...] = ("ANTHROPIC_API_KEY",)


class MissingRequiredKeyError(RuntimeError):
    """Raised by validate_startup_keys() when a required env key is absent."""

    def __init__(self, missing: list[str], *, detail: str = "") -> None:
        self.missing = missing
        self.detail = detail
        message = (
            f"Cloris cannot start: required env key(s) missing: "
            f"{', '.join(missing)}. Set them in .env or the environment."
        )
        if detail:
            message = f"{message} {detail}."
        super().__init__(message)


def validate_startup_keys() -> None:
    """Raise MissingRequiredKeyError if any required API key is unset.

    Skip when CLORIS_SKIP_STARTUP_VALIDATION=1 (test environments that
    don't set live API keys).
    """
    if os.getenv("CLORIS_SKIP_STARTUP_VALIDATION", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        return
    required = _REQUIRED_AT_STARTUP
    if ANTHROPIC_API_KEY_SOURCE == "personal":
        # The env sweep below reads os.getenv("ANTHROPIC_API_KEY"), which under
        # the secondary account selector is the wrong variable — if the default
        # variable is populated in .env it would report "present" while the
        # resolved key is empty, and if unset it would report the wrong variable
        # name. Under this selector the requirement is PERSONAL_ANTHROPIC_API_KEY,
        # checked against the resolved value and named plainly in the message.
        if not ANTHROPIC_API_KEY.strip():
            raise MissingRequiredKeyError(
                ["PERSONAL_ANTHROPIC_API_KEY"],
                detail=(
                    "ANTHROPIC_API_KEY_SOURCE=personal but "
                    "PERSONAL_ANTHROPIC_API_KEY is empty"
                ),
            )
        required = tuple(k for k in required if k != "ANTHROPIC_API_KEY")
    missing = [k for k in required if not os.getenv(k, "").strip()]
    if missing:
        raise MissingRequiredKeyError(missing)

# --- Model Names ---
CHEAP_MODEL_NAME: str = _optional("CHEAP_MODEL_NAME", "gpt-4o-mini")
OPUS_MODEL_NAME: str = _optional("OPUS_MODEL_NAME", "claude-opus-4-6")
FACIAL_MODEL_NAME: str = _optional("FACIAL_MODEL_NAME", OPUS_MODEL_NAME)
# Strategy selects plans; full evaluation judges individual candidates.
STRATEGY_MODEL_NAME: str = _optional("STRATEGY_MODEL_NAME", OPUS_MODEL_NAME)
# Which strategy system prompt to render. "live" is production and must stay
# byte-identical; "register" swaps one craft paragraph for the register-and-
# shape section under measurement (plans/boolean-postprocessor.md).
STRATEGY_PROMPT_VARIANT: str = _optional("STRATEGY_PROMPT_VARIANT", "live")
FULL_EVAL_MODEL_NAME: str = _optional("FULL_EVAL_MODEL_NAME", OPUS_MODEL_NAME)
# Outreach tier: stored GitHub recruiter copy (one candidate at a time).
OUTREACH_MODEL_NAME: str = _optional("OUTREACH_MODEL_NAME", OPUS_MODEL_NAME)
# Judgment pipelining (window 2). Only the serial path is live-verified.
# gate: CLO-161 by 2026-10-15
FULL_EVAL_PIPELINE_ENABLED: bool = _env_flag("FULL_EVAL_PIPELINE_ENABLED", "false")

# GLM-5.2 judgment-runtime experiment. Model constants are explicit so
# launchers can route only facial/full stages without repointing
# OPUS_MODEL_NAME and silently changing other calls.
FIREWORKS_STANDARD_MODEL_NAME: str = "accounts/fireworks/models/glm-5p2"
FIREWORKS_FAST_MODEL_NAME: str = "accounts/fireworks/routers/glm-5p2-fast"
# Which campaigns route judgment through the Fireworks provider policy.
# gate: by-design — per-campaign configuration
FIREWORKS_JUDGMENT_POLICY_ENABLED: bool = _env_flag(
    "FIREWORKS_JUDGMENT_POLICY_ENABLED", "false"
)
# Kill switch: default on since 2026-09-20; off restores non-streamed judgment
# responses, whose attempt budget is a wall clock rather than an inactivity
# window. On in every active launcher since 2026-07-31.
FIREWORKS_JUDGMENT_STREAM_ENABLED: bool = _env_flag(
    "FIREWORKS_JUDGMENT_STREAM_ENABLED", "true"
)
FIREWORKS_STRATEGY_REASONING_EFFORT: str = _optional(
    "FIREWORKS_STRATEGY_REASONING_EFFORT", ""
).strip().lower()
FIREWORKS_STRATEGY_ATTEMPT_TIMEOUT_SECONDS: float = float(
    _optional("FIREWORKS_STRATEGY_ATTEMPT_TIMEOUT_SECONDS", "300")
)
FIREWORKS_STRATEGY_TOTAL_DEADLINE_SECONDS: float = float(
    _optional("FIREWORKS_STRATEGY_TOTAL_DEADLINE_SECONDS", "630")
)
FIREWORKS_STRATEGY_MAX_ATTEMPTS: int = int(
    _optional("FIREWORKS_STRATEGY_MAX_ATTEMPTS", "2")
)
FIREWORKS_FACIAL_REASONING_EFFORT: str = _optional(
    "FIREWORKS_FACIAL_REASONING_EFFORT", ""
).strip().lower()
FIREWORKS_FULL_REASONING_EFFORT: str = _optional(
    "FIREWORKS_FULL_REASONING_EFFORT", ""
).strip().lower()
# DeepSeek-family models on Fireworks think by DEFAULT (a live 2026-09-16 probe
# of accounts/fireworks/models/deepseek-v4p1-flash returned reasoning_content
# and non-zero usage.completion_tokens_details.reasoning_tokens with no
# reasoning_effort sent). "none" disables thinking outright — the posture the
# cheap/extraction tier wants, where chain-of-thought only burns the max_tokens
# budget before the JSON; "low"/"high"/"max" set the tier instead. Empty (the
# default) sends nothing and leaves the provider default in force, so neither
# knob changes any wire byte until an operator sets it. Allowed values are
# "", "none", "low", "high", "max"; anything else raises ValueError naming the
# knob where the value is consumed (shared/llm_clients._validated_fireworks_
# reasoning_effort).
FIREWORKS_CHEAP_REASONING_EFFORT: str = _optional("FIREWORKS_CHEAP_REASONING_EFFORT", "")
FIREWORKS_SHADOW_REASONING_EFFORT: str = _optional("FIREWORKS_SHADOW_REASONING_EFFORT", "")
# Completion-budget FLOOR for the zero-influence shadow seat. Callers hardcode
# an Anthropic-calibrated max_tokens (shared/judger.py: 8192 single facial,
# 16384 facial batch and full eval), but DeepSeek-family shadows think by
# default and Fireworks counts that chain-of-thought against max_tokens — so
# the caller's cap is spent on reasoning before the verdict is written. Two
# measurements, not guesses (2026-09-16):
#
#   - Offline replay, full eval at effort "high": 35K characters of reasoning,
#     finish_reason=length against the 16384 cap, verdict UNPARSED. That is a
#     CEILING failure, not an effort failure — the only reason the live run
#     was pinned to "low".
#   - Live run, facial BATCH (25 candidates) at effort "low": parsed cleanly
#     (finish_reason=stop, 25/25 agreement with Opus) but on 43,379 characters
#     of reasoning — roughly 11-13K tokens, sitting just under the same 16384
#     cap with no headroom left.
#
# max_tokens is a CEILING, not spend — raising it only pays for tokens actually
# generated — so the floor buys back the headroom that lets the shadow run at
# the effort a promotion decision would actually use. Mirrors
# FIREWORKS_PRIMARY_MIN_MAX_TOKENS (the same reasoning-headroom floor on the
# primary Fireworks path); 0 (the default) means NO floor, leaving every shadow
# request byte-identical to today.
SHADOW_LLM_MIN_MAX_TOKENS: int = int(_optional("SHADOW_LLM_MIN_MAX_TOKENS", "0"))
if SHADOW_LLM_MIN_MAX_TOKENS < 0:
    raise ValueError(
        "SHADOW_LLM_MIN_MAX_TOKENS must be >= 0 (0 disables the floor); got "
        f"{SHADOW_LLM_MIN_MAX_TOKENS!r}"
    )
# Worker count for the shadow-judge executor (shared/judger.py
# ``_get_shadow_executor``). WHY it is a knob at all, from the 2026-09-16
# DeepSeek live run: a single worker could not keep up. Each shadow call took
# 40s-7min (one hit the SHADOW_LLM_TIMEOUT_SECONDS wall-clock timeout below and
# was retried) while the PRIMARY produced a verdict every 2-3 minutes, so the
# one-worker queue grew monotonically and the shadow report ran 10+ minutes
# behind the run it was supposed to be shadowing.
#
# Concurrency here CANNOT affect a verdict. Shadow comparisons are
# zero-influence by construction — the result is recorded and discarded, never
# read by a primary path — and fail-soft end to end, so a worker that dies
# takes nothing with it. What the extra workers buy is queue drain rate, i.e.
# how fresh the tally in tools/shadow_report.py is; the provider side is not
# the constraint either (DeepSeek on Fireworks admits concurrency far above any
# value that makes sense here). 1, the default, is byte-identical to today:
# same single worker, same submit order, same inline-fallback behavior under
# SHADOW_ASYNC_ENABLED=false.
SHADOW_JUDGE_WORKERS: int = int(_optional("SHADOW_JUDGE_WORKERS", "1"))
if SHADOW_JUDGE_WORKERS < 1:
    raise ValueError(
        "SHADOW_JUDGE_WORKERS must be >= 1 (use SHADOW_ASYNC_ENABLED=false to "
        f"run shadow comparisons inline instead); got {SHADOW_JUDGE_WORKERS!r}"
    )
FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS: float = float(
    _optional("FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS", "120")
)
FIREWORKS_FACIAL_TOTAL_DEADLINE_SECONDS: float = float(
    _optional("FIREWORKS_FACIAL_TOTAL_DEADLINE_SECONDS", "180")
)
FIREWORKS_FACIAL_MAX_ATTEMPTS: int = int(
    _optional("FIREWORKS_FACIAL_MAX_ATTEMPTS", "2")
)
FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS: float = float(
    _optional("FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS", "240")
)
FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS: float = float(
    _optional("FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS", "360")
)
FIREWORKS_FULL_MAX_ATTEMPTS: int = int(
    _optional("FIREWORKS_FULL_MAX_ATTEMPTS", "2")
)
# Keys a Fireworks prompt-cache slot per facial batch; judgment_runtime.py
# refuses it without FIREWORKS_JUDGMENT_POLICY_ENABLED. A cost and latency
# lever that needs one measured session on a Fireworks-judge campaign.
# gate: by-design — per-campaign configuration
FIREWORKS_PROMPT_AFFINITY_ENABLED: bool = _env_flag(
    "FIREWORKS_PROMPT_AFFINITY_ENABLED", "false"
)
LINKEDIN_V2_FACIAL_CONTRACT: str = _optional(
    "LINKEDIN_V2_FACIAL_CONTRACT", "legacy"
).strip().lower()
LINKEDIN_V2_FULL_CONTRACT: str = _optional(
    "LINKEDIN_V2_FULL_CONTRACT", "legacy"
).strip().lower()
# Facial tightening raises the positive-signal bar within a string.
# Keep it off when a dense probe could be mistaken for judge bias; telemetry stays active.
# gate: by-design — a dense probe would read as judge bias
LINKEDIN_FACIAL_TIGHTENING_ENABLED: bool = _env_flag(
    "LINKEDIN_FACIAL_TIGHTENING_ENABLED", "false"
)
# Mid-campaign kill switch for the per-string sniper retrieval prior.
LINKEDIN_SNIPER_PRIOR_ENABLED: bool = _env_flag(
    "LINKEDIN_SNIPER_PRIOR_ENABLED", "true"
)
# Kill switch: default on since 2026-09-20; off restores saving without a
# judgment note. The note is an IRREVERSIBLE MUTATION — it writes visible text
# on a real candidate's profile in the recruiter's account — but every active
# campaign launcher has exported it true since 2026-08-22, so the default now
# matches what live runs already do. Turn it off for any run whose saves should
# leave no recruiter-visible trace.
LINKEDIN_SAVE_NOTE_ENABLED: bool = _env_flag("LINKEDIN_SAVE_NOTE_ENABLED", "true")
# Bound note typing time while preserving room for rationale and evidence.
LINKEDIN_NOTE_MAX_CHARS: int = int(_optional("LINKEDIN_NOTE_MAX_CHARS", "900"))
# Trim quoted evidence before the recruiter-facing rationale reaches the note ceiling.
LINKEDIN_NOTE_EVIDENCE_MAX_CHARS: int = int(
    _optional("LINKEDIN_NOTE_EVIDENCE_MAX_CHARS", "420")
)

# Kill switch: default on since 2026-09-20; off restores a campaign that EXITS
# at a closed governor window instead of sleeping through it. Persistence never
# raises volume caps. On in every active launcher since 2026-07-29.
LINKEDIN_CAMPAIGN_PERSIST: bool = _env_flag("LINKEDIN_CAMPAIGN_PERSIST", "true")
# Consecutive UNCLASSIFIED session errors ("error: <Type>") the persistent
# cycle may absorb by sleeping a dormant period and resuming, before giving up
# and raising. Deterministic regime errors (geography_regime_error,
# constraint_manifest_error, preflight_regime_error) and operator interrupts
# are NEVER retried — retrying a config error loops forever. A clean
# session-duration-cap session resets the counter. 0 = today's behavior.
LINKEDIN_SESSION_ERROR_RETRIES: int = int(
    _optional("LINKEDIN_SESSION_ERROR_RETRIES", "0")
)
# Clear companies/titles/keywords at every string start, then re-add.
# gate: CLO-240 by 2026-10-04
LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED: bool = _env_flag(
    "LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", "false"
)
# gate: by-design — per-campaign configuration
LINKEDIN_CONTAMINATION_RESET_ENABLED: bool = _env_flag(
    "LINKEDIN_CONTAMINATION_RESET_ENABLED", "false"
)
# Run detect_recruiter_health on the every-loop happy path
# (_ensure_browser_healthy) and trip force_backoff on
# blocked_or_rate_limited. Also short-circuits the SWG recovery ladder
# after one try-again so a blocked page is not re-probed via reload/goto.
# Flag OFF preserves today's disconnect-only classifier path and full SWG
# escalation.
# gate: CLO-72 by 2026-11-03
LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED: bool = _env_flag(
    "LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED", "false"
)
# Settle unmatched pending-full candidates with a receipt and continue recovery.
# Pagination failures and unsettled work still raise.
# Kill switch: default on since 2026-09-20; off restores a resume that RAISES
# on an unmatched pending-full card. On in every active launcher since
# 2026-08-01.
LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED: bool = _env_flag(
    "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", "true"
)
# Re-ask a full evaluation ONCE when the model echoes the wrong opaque
# candidate ID. The mismatch check inside validate_full_tool_arguments is a
# security boundary, not a formatting check — profile text is attacker
# controlled, so a displaced ID may be injection rather than model sloppiness.
# Today every mismatch becomes a PARSE_FAILURE that throws away a fully
# opened, scrolled, and read profile (~75s of browser work plus the full-eval
# call) and spends one of the two strikes the full-contract corruption breaker
# allows; measured rate is 2 of 218 full attempts (0.92%). Flag ON records and
# classifies the mismatch FIRST (``full_candidate_id_mismatch``), then re-asks
# the same profile text under a FRESH opaque ID and its own child logical
# call. A failed re-ask surfaces exactly as today, keyed to the child call, so
# the breaker still sees one strike per corrupted call. Recoveries are capped
# per Pipeline (see ``_FULL_ID_MISMATCH_RECOVERY_CEILING``) so an injection
# campaign cannot buy unlimited free probes, and the contract validator itself
# is untouched in both flag states.
# gate: CLO-240 by 2026-09-27
LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED: bool = _env_flag(
    "LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED", "false"
)
# Dormant-gap shape between sessions (log-normal, minutes). Defaults reproduce
# the historical median ~110 clamped 75-180. The 24h open cap — not gap length —
# is the binding daily-volume limit, so shortening gaps compresses the workday
# without raising volume; it is an operator signature-shape call.
LINKEDIN_DORMANT_MEDIAN_MINUTES: float = float(
    _optional("LINKEDIN_DORMANT_MEDIAN_MINUTES", "110")
)
LINKEDIN_DORMANT_MIN_MINUTES: float = float(
    _optional("LINKEDIN_DORMANT_MIN_MINUTES", "75")
)
LINKEDIN_DORMANT_MAX_MINUTES: float = float(
    _optional("LINKEDIN_DORMANT_MAX_MINUTES", "180")
)

# Add domain vocabulary before string formation, using one strategy-tier call.
# Failure yields no vocabulary and leaves the ordinary formation path intact.
# gate: by-design — per-campaign configuration
LINKEDIN_VOCABULARY_ENUMERATION_ENABLED: bool = _env_flag(
    "LINKEDIN_VOCABULARY_ENUMERATION_ENABLED", "false"
)
# Union external research with the model-generated vocabulary.
# gate: by-design — per-campaign configuration
LINKEDIN_VOCABULARY_ENUMERATION_RESEARCH_ENABLED: bool = _env_flag(
    "LINKEDIN_VOCABULARY_ENUMERATION_RESEARCH_ENABLED", "false"
)
# Deterministic hyphen-space twins in the Boolean compiler.
# gate: CLO-240 by 2026-09-27
LINKEDIN_SURFACE_VARIANT_EXPANSION_ENABLED: bool = _env_flag(
    "LINKEDIN_SURFACE_VARIANT_EXPANSION_ENABLED", "false"
)

# Lexicon-driven Boolean enrichment at strategy formation and on
# adaptation-generated strings (plans/boolean-postprocessor.md). Off means
# every plan serializes exactly as it did before the enricher existed; the
# queue-time normalizer stays in place either way as the safety net.
# gate: CLO-304 by 2026-11-03
LINKEDIN_LEXICON_ENRICHMENT_ENABLED: bool = _env_flag(
    "LINKEDIN_LEXICON_ENRICHMENT_ENABLED", "false"
)
# Which role lexicon the enricher reads. Empty means the package default.
LINKEDIN_BOOLEAN_LEXICON_PATH: str = _optional("LINKEDIN_BOOLEAN_LEXICON_PATH", "")

# Ask the local enricher sidecar (a trained model served over HTTP) for
# richer OR groups and missing anchor groups, after the lexicon enricher, at
# strategy formation and on adaptation strings. A proposal is kept only when
# the scorer rates the whole string higher. Off means every plan serializes
# exactly as it does without the sidecar; a dead sidecar yields no proposals.
# gate: CLO-333 by 2026-10-15
LINKEDIN_ENRICHER_SIDECAR_ENABLED: bool = _env_flag(
    "LINKEDIN_ENRICHER_SIDECAR_ENABLED", "false"
)
LINKEDIN_ENRICHER_SIDECAR_URL: str = _optional(
    "LINKEDIN_ENRICHER_SIDECAR_URL", "http://127.0.0.1:8765"
)
LINKEDIN_ENRICHER_SIDECAR_TIMEOUT_SECONDS: float = float(
    _optional("LINKEDIN_ENRICHER_SIDECAR_TIMEOUT_SECONDS", "90")
)
LINKEDIN_ENRICHER_SIDECAR_SAMPLES: int = int(
    _optional("LINKEDIN_ENRICHER_SIDECAR_SAMPLES", "8")
)

# gate: by-design — requires the tool facial contract; every launcher pins legacy
LINKEDIN_FACIAL_CONCURRENCY_ENABLED: bool = _env_flag(
    "LINKEDIN_FACIAL_CONCURRENCY_ENABLED", "false"
)
# Kill switch: default on since 2026-09-20; off restores a session that DIES on
# a facial provider fault instead of abandoning the page. On in every active
# launcher since 2026-07-31.
LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED: bool = _env_flag(
    "LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED", "true"
)
# Contain provider faults as re-meetable JUDGMENT_FAILURE outcomes.
# The consecutive-fault ceiling still ends a sustained outage.
# Kill switch: default on since 2026-09-20; off restores a session that DIES on
# a full-eval provider fault. On in every active launcher since 2026-08-12.
LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED: bool = _env_flag(
    "LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED", "true"
)
LINKEDIN_FULL_EVAL_MAX_CONSECUTIVE_FAULTS: int = int(
    _optional("LINKEDIN_FULL_EVAL_MAX_CONSECUTIVE_FAULTS", "3")
)
LINKEDIN_FACIAL_MAX_CONCURRENCY: int = int(
    _optional("LINKEDIN_FACIAL_MAX_CONCURRENCY", "1")
)
LINKEDIN_FACIAL_TARGET_BATCH_SIZE: int = int(
    _optional("LINKEDIN_FACIAL_TARGET_BATCH_SIZE", "8")
)
# Process-local safety cap for deliberately bounded live canaries.  This is
# separate from MAX_PAGES_PER_STRING: it counts fully reviewed pages across
# every string/variant in the invocation.  Zero preserves the ordinary
# unbounded-by-session behavior; launchers must opt in explicitly.
LINKEDIN_TOTAL_PAGE_CAP: int = int(
    _optional("LINKEDIN_TOTAL_PAGE_CAP", "0")
)
LINKEDIN_PAGE_ALLOCATOR_MODE: str = _env_choice(
    "LINKEDIN_PAGE_ALLOCATOR_MODE", "off", {"active", "off", "shadow"}
)

# --- Trial build posture ---
# pilot scope is deliberately LinkedIn-only. Dev defaults stay broad so
# non-trial tests and module work are not silently hidden; packaged trial
# entrypoints set CLORIS_TRIAL_MODE=true before config import.
# gate: by-design — product-era relic; delete with the frontend
CLORIS_TRIAL_MODE: bool = _env_flag("CLORIS_TRIAL_MODE", "false")
CLORIS_TRIAL_ALLOWED_MODULES: tuple[str, ...] = tuple(
    part.strip()
    for part in _optional("CLORIS_TRIAL_ALLOWED_MODULES", "linkedin").split(",")
    if part.strip()
)
ANTHROPIC_HEALTH_CACHE_SECONDS: float = float(
    _optional("ANTHROPIC_HEALTH_CACHE_SECONDS", "600")
)

# --- Designer vision-LLM model + fallback (audit Move #16) ---
# Primary multimodal portfolio evaluation model. The orchestrator reads
# this instead of hardcoding a model id so provider-side renames can be
# handled through env/config.
DESIGNER_VISION_MODEL_NAME: str = _optional(
    "DESIGNER_VISION_MODEL_NAME", "gemini-3.1-pro-preview"
)
# When Gemini fails (schema-validity / image-grounding /
# llm_raise paths), the cascade retries against this secondary vision
# provider before dropping the candidate to HITL. Default empty
# disables the cascade — pre-Move-16 behavior. Recommended fallback
# for the trial:
#   DESIGNER_VISION_FALLBACK_MODEL_NAME=claude-sonnet-4-6
# The fallback caller for "claude-*" models is
# `designer.vision_evaluation.claude_vision_llm_call`.
DESIGNER_VISION_FALLBACK_MODEL_NAME: str = _optional(
    "DESIGNER_VISION_FALLBACK_MODEL_NAME", ""
)
MARKET_INTEL_EXTERNAL_RESEARCH_PROVIDER: str = _optional(
    "MARKET_INTEL_EXTERNAL_RESEARCH_PROVIDER",
    "auto",
)
MARKET_INTEL_EXTERNAL_RESEARCH_MODEL: str = _optional(
    "MARKET_INTEL_EXTERNAL_RESEARCH_MODEL",
    "",
)
MARKET_INTEL_PERPLEXITY_PRESET: str = _optional(
    "MARKET_INTEL_PERPLEXITY_PRESET",
    "medium",
)
MARKET_INTEL_EXTERNAL_RESEARCH_TIMEOUT_SECONDS: float = float(
    _optional("MARKET_INTEL_EXTERNAL_RESEARCH_TIMEOUT_SECONDS", "300")
)
# Require every strategy work unit to finish before publishing market intelligence.
# A clean session exit alone does not establish complete strategy coverage.
# Kill switch: default on since 2026-09-20; off restores publishing market
# intelligence on PARTIAL strategy coverage. On in every active launcher since
# 2026-07-29.
MARKET_INTEL_REQUIRE_STRATEGY_COMPLETE: bool = _env_flag(
    "MARKET_INTEL_REQUIRE_STRATEGY_COMPLETE", "true"
)

# LinkedIn external evidence augmentation; disabled unless explicitly enabled.
LINKEDIN_EXTERNAL_EVIDENCE_ENABLED: bool = _optional(
    "LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
LINKEDIN_EXTERNAL_EVIDENCE_MODEL: str = _optional("LINKEDIN_EXTERNAL_EVIDENCE_MODEL", "")
LINKEDIN_EXTERNAL_EVIDENCE_TIMEOUT_SECONDS: float = float(
    _optional("LINKEDIN_EXTERNAL_EVIDENCE_TIMEOUT_SECONDS", "90")
)
LINKEDIN_EXTERNAL_EVIDENCE_MAX_OUTPUT_TOKENS: int = int(
    _optional("LINKEDIN_EXTERNAL_EVIDENCE_MAX_OUTPUT_TOKENS", "4096")
)
LINKEDIN_EXTERNAL_EVIDENCE_MIN_CITATIONS: int = int(
    _optional("LINKEDIN_EXTERNAL_EVIDENCE_MIN_CITATIONS", "2")
)
LINKEDIN_EXTERNAL_EVIDENCE_MIN_IDENTITY_CONFIDENCE: float = float(
    _optional("LINKEDIN_EXTERNAL_EVIDENCE_MIN_IDENTITY_CONFIDENCE", "0.5")
)
# Intentionally NOT defaulted to the market-intel "medium" research preset
# and the candidate-evidence path runs under a tighter time budget.
LINKEDIN_EXTERNAL_EVIDENCE_PERPLEXITY_PRESET: str = _optional(
    "LINKEDIN_EXTERNAL_EVIDENCE_PERPLEXITY_PRESET", ""
)

# Enable three-class facial triage (YES/BORDERLINE/NO).
LINKEDIN_FACIAL_BORDERLINE_ENABLED: bool = _optional(
    "LINKEDIN_FACIAL_BORDERLINE_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}

# --- GLM-5.2 shadow judge (Fireworks) ---
# Evaluates replacing Opus as the facial judgment model. SHADOW ONLY: the
# shadow verdict is recorded and compared (see shared/judger.py's
# facial_judge / facial_judge_batch shadow hooks) but never influences any
# real decision. Off by default; opt-in per env. The Fireworks account is
# US-hosted and OpenAI-API-compatible.
FIREWORKS_API_KEY: str = _optional("FIREWORKS_API_KEY", "")
# Latency note: with SHADOW_ASYNC_ENABLED (below, default on) the shadow
# call runs on a background worker and does not block the judge path. With
# it OFF the call is synchronous inline: on the batch path that adds at
# most ~120s (60s timeout x one retry) per PAGE; on the sequential path
# (legacy briefs, or an untrustworthy-batch fallback) the worst case is
# ~120s per CANDIDATE — do not leave that combination enabled on
# latency-sensitive runs.
# gate: by-design — shadow experiment, cost
SHADOW_FACIAL_MODEL_ENABLED: bool = _env_flag("SHADOW_FACIAL_MODEL_ENABLED", "false")
SHADOW_FACIAL_MODEL_NAME: str = _optional(
    "SHADOW_FACIAL_MODEL_NAME", "accounts/fireworks/models/glm-5p2"
)
SHADOW_FACIAL_BASE_URL: str = _optional(
    "SHADOW_FACIAL_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
# Primary-path Fireworks endpoint (provider dispatch for non-Anthropic
# primaries — the GLM promotion, item 19). Defaults to the shadow knob so
# the two paths can't silently point at different hosts; the general name
# exists because primaries reading a SHADOW_* knob is a confusion trap.
FIREWORKS_BASE_URL: str = _optional("FIREWORKS_BASE_URL", SHADOW_FACIAL_BASE_URL)
# Fireworks includes reasoning tokens in the completion budget.
# This floor leaves room for both reasoning and the answer.
FIREWORKS_PRIMARY_MIN_MAX_TOKENS: int = int(
    _optional("FIREWORKS_PRIMARY_MIN_MAX_TOKENS", "16384")
)
# Optional process-local primary Fireworks spend ceiling. Zero keeps ordinary
# and offline paths unchanged; the isolated live glmopt launcher requires and
# pins a positive operator cap.
FIREWORKS_PRIMARY_MAX_COST_USD: float = float(
    _optional("FIREWORKS_PRIMARY_MAX_COST_USD", "0")
)
# Dispatch shadow comparisons on a background worker; no live decision consumes them.
# Disabling this flag makes shadow calls synchronous.
SHADOW_ASYNC_ENABLED: bool = _env_flag("SHADOW_ASYNC_ENABLED", "true")
# Per-attempt shadow timeout; the final drain has a separate bound.
SHADOW_LLM_TIMEOUT_SECONDS: float = float(
    _optional("SHADOW_LLM_TIMEOUT_SECONDS", "300")
)

# --- Shadow strategist (plans/sourcing-generality-hardening.md item 19) ---
# Evaluates replacing Opus at the table-setting tier (preflight v2 +
# strategy formation). SHADOW ONLY: when on, those calls also fire a
# non-blocking call to the model below with byte-identical prompts
# (shared/strategy_shadow.py); the artifact + deterministic comparison
# metrics land under <state_dir>/shadow_strategy/ for offline judging and
# never influence the run. Always async (single background worker) — no
# inline fallback, unlike SHADOW_ASYNC_ENABLED above.
# gate: by-design — shadow experiment, cost
SHADOW_STRATEGY_ENABLED: bool = _env_flag("SHADOW_STRATEGY_ENABLED", "false")
# claude-fable-5 rejects sampling params; opus_llm and opus_llm_cached pass
# neither. Both route thinking kwargs through _thinking_request_kwargs, whose
# guard sends summarized thinking only to Fable/Mythos and NO thinking param
# to Opus-family models.
SHADOW_STRATEGY_MODEL_NAME: str = _optional(
    "SHADOW_STRATEGY_MODEL_NAME", "claude-fable-5-1"
)

# --- Browser ---
CDP_URL: str = _optional("CDP_URL", "http://127.0.0.1:9222")

# Behavior limits: zero MAX_PAGES_PER_STRING explicitly disables its cap.
MAX_PAGES_PER_STRING: int = int(_optional("MAX_PAGES_PER_STRING", "25"))
PAGE_DELAY_SECONDS: float = float(_optional("PAGE_DELAY_SECONDS", "3"))
PROFILE_DELAY_SECONDS: float = float(_optional("PROFILE_DELAY_SECONDS", "2"))

# --- Cadence pause (anti-detection) ---
# After this many minutes of continuous activity, pause for a human-like break.
# Both values are jittered ±20% at runtime to avoid metronomic patterns.
CADENCE_INTERVAL_MINUTES: float = float(_optional("CADENCE_INTERVAL_MINUTES", "30"))
CADENCE_PAUSE_SECONDS: float = float(_optional("CADENCE_PAUSE_SECONDS", "120"))
# Kill switch: default on since 2026-09-20; off restores a run that records no
# timing telemetry at all. Recording only; it changes no decision. On in every
# active launcher since 2026-07-31.
LINKEDIN_TIMING_TELEMETRY_ENABLED: bool = _env_flag(
    "LINKEDIN_TIMING_TELEMETRY_ENABLED", "true"
)

# --- Search-control tuning ---
# The first compound block is intentionally smaller so the run can exploit early.
OPENING_BLOCK_SIZE: int = int(_optional("OPENING_BLOCK_SIZE", "3"))
# Large/noisy strings get a bounded number of pre-commit rescue attempts before stop.
PRECOMMIT_MAX_RECOVERY_ATTEMPTS: int = int(_optional("PRECOMMIT_MAX_RECOVERY_ATTEMPTS", "2"))
# Once a variant is committed, stop after this many consecutive zero-signal pages.
COMMITTED_ZERO_SIGNAL_STOP_STREAK: int = int(_optional("COMMITTED_ZERO_SIGNAL_STOP_STREAK", "2"))
# After a failed drift rescue, allow only this many additional zero-signal committed pages.
POST_DRIFT_ZERO_SIGNAL_STOP_STREAK: int = int(_optional("POST_DRIFT_ZERO_SIGNAL_STOP_STREAK", "1"))

# --- Legacy pagination floors (deprecated) ---
# These remain for compatibility with prior runs / architecture metadata, but the
# orchestrator now prefers phase-aware recovery + signal-decay controls instead.
MIN_PAGES_BY_RESULT_COUNT: list[tuple[int, int]] = [
    (500, 3),
    (100, 2),
    (30, 1),
    (0, 1),
]

# --- Glance assessment (page-level pre-filter) ---
GLANCE_NOISE_TITLE_THRESHOLD = 0.7   # >70% sharing a non-fit title family
GLANCE_KEYWORD_MISS_THRESHOLD = 0    # 0 snippets have any relevant key_term
GLANCE_MIN_SNIPPETS = 8              # Skip glance if fewer than 8 snippets

# --- Mid-page early exit ---
EARLY_EXIT_MIN_CANDIDATES = 5        # Evaluate at least N before checking
EARLY_EXIT_FACIAL_NO_RATE = 0.95     # Fallback when market density is unknown
# Sparse markets pay the most wall-clock per page, so they abandon a dead page
# SOONEST. The retired brief-derived formula (1 - expected_yes_low * 0.5)
# inverted this: the sparser the brief's expected yes rate, the more facial_no
# a page had to show before the agent left it.
EARLY_EXIT_FACIAL_NO_RATE_BY_DENSITY: dict[str, float] = {
    "sparse": 0.85,
    "moderate": 0.90,
    "dense": 0.95,
}

# --- LinkedIn search experimentation ---
SEARCH_EXPERIMENT_MAX_PLANNED_VARIANTS: int = int(_optional("SEARCH_EXPERIMENT_MAX_PLANNED_VARIANTS", "3"))
SEARCH_EXPERIMENT_MAX_EXECUTED_SIBLINGS: int = int(_optional("SEARCH_EXPERIMENT_MAX_EXECUTED_SIBLINGS", "2"))
SEARCH_EXPERIMENT_MAX_CONSECUTIVE_REWRITES: int = int(_optional("SEARCH_EXPERIMENT_MAX_CONSECUTIVE_REWRITES", "2"))
SEARCH_EXPERIMENT_MUTATION_BUDGET: int = int(_optional("SEARCH_EXPERIMENT_MUTATION_BUDGET", "8"))
SEARCH_EXPERIMENT_MAX_DRIFT_ATTEMPTS_PER_VARIANT: int = int(
    _optional("SEARCH_EXPERIMENT_MAX_DRIFT_ATTEMPTS_PER_VARIANT", "1")
)
SEARCH_EXPERIMENT_DRIFT_BUDGET: int = int(_optional("SEARCH_EXPERIMENT_DRIFT_BUDGET", "1"))
# After repeated structured demotions, close promotion in both planning and parsing.
# The deterministic lifecycle gate remains independent of the search posture.
SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT: int = int(
    _optional("SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT", "2")
)
# Minimum distinct title families required for a title-filter promotion.
# Below the floor, keep keyword-only search; zero disables the floor.
LINKEDIN_TITLE_FILTER_MIN_FAMILIES: int = int(
    _optional("LINKEDIN_TITLE_FILTER_MIN_FAMILIES", "6")
)
# Block-adaptation SPRT/cooldown gate, configurable via env. Defaults
# match default_adaptation_gate_config() exactly (min 1/1/1, no SPRT bounds,
# no cooldown) so the gate is inert until deliberately tuned.
ADAPTATION_GATE_MIN_STRINGS: int = int(_optional("ADAPTATION_GATE_MIN_STRINGS", "1"))
ADAPTATION_GATE_MIN_CANDIDATES_SEEN: int = int(
    _optional("ADAPTATION_GATE_MIN_CANDIDATES_SEEN", "1")
)
ADAPTATION_GATE_MIN_RESULTS_SEEN: int = int(
    _optional("ADAPTATION_GATE_MIN_RESULTS_SEEN", "1")
)
ADAPTATION_GATE_COOLDOWN_BLOCKS: int = int(
    _optional("ADAPTATION_GATE_COOLDOWN_BLOCKS", "0")
)
_adaptation_gate_sprt_lower = _optional("ADAPTATION_GATE_SPRT_LOWER", "")
ADAPTATION_GATE_SPRT_LOWER: float | None = (
    float(_adaptation_gate_sprt_lower) if _adaptation_gate_sprt_lower else None
)
_adaptation_gate_sprt_upper = _optional("ADAPTATION_GATE_SPRT_UPPER", "")
ADAPTATION_GATE_SPRT_UPPER: float | None = (
    float(_adaptation_gate_sprt_upper) if _adaptation_gate_sprt_upper else None
)
# Scale the result window when constructing filter-bearing variants.
# Keep the lifecycle decision function independent of posture; Boolean-only windows stay unscaled.
SEARCH_EXPERIMENT_FILTER_LED_WINDOW_FACTOR: float = float(
    _optional("SEARCH_EXPERIMENT_FILTER_LED_WINDOW_FACTOR", "0.3")
)
SEARCH_INTELLIGENCE_EXPLOIT_PROMOTION_LIMIT: int = int(
    _optional("SEARCH_INTELLIGENCE_EXPLOIT_PROMOTION_LIMIT", "3")
)
SEARCH_INTELLIGENCE_EXPLOIT_DEMOTION_LIMIT: int = int(
    _optional("SEARCH_INTELLIGENCE_EXPLOIT_DEMOTION_LIMIT", "3")
)

# Count a daily session slot from the first profile open rather than only on a
# clean completion. The old rule made 133 of 162 historical sessions invisible
# to the cap, because a session that opened profiles and then errored consumed
# nothing — the account was touched and the budget did not record it. Sessions
# the operator stops are still free: those are development churn, and a launch
# that never opens a profile is free under either rule.
# Kill switch: default on since 2026-09-20; off restores the old rule, under
# which an errored session consumed no daily slot. On in every active launcher
# since 2026-08-03.
LINKEDIN_SLOT_ON_FIRST_OPEN: bool = _env_flag("LINKEDIN_SLOT_ON_FIRST_OPEN", "true")

# --- LinkedIn cadence trim knobs ---
# Breaks the cadence clockwork: 64.0% of intervals land on exactly 24.0 or
# 42.0 minutes with this off.
# gate: CLO-240 by 2026-09-27
LINKEDIN_CADENCE_READ_FIX_ENABLED: bool = _env_flag(
    "LINKEDIN_CADENCE_READ_FIX_ENABLED", "false"
)
# Kill switch: default on since 2026-09-20; off restores the pre-Wave-2 profile
# read (whole-page scroll, no section targeting). On in every active launcher
# since 2026-08-02.
LINKEDIN_SECTION_DIRECTED_READ_ENABLED: bool = _env_flag(
    "LINKEDIN_SECTION_DIRECTED_READ_ENABLED", "true"
)
# Off dispatches every keystroke with zero key overlap.
# gate: CLO-240 by 2026-09-27
LINKEDIN_TYPING_DWELL_ENABLED: bool = _env_flag("LINKEDIN_TYPING_DWELL_ENABLED", "false")
# On, every ghost click traces a Bezier mouse path by aliasing the playwright
# module names to rebrowser; off, the pointer jumps straight to each target.
# gate: CLO-322 by 2026-11-03
LINKEDIN_GHOST_CURSOR_ENABLED: bool = _env_flag("LINKEDIN_GHOST_CURSOR_ENABLED", "false")
LINKEDIN_SEARCH_TYPING_CHAR_MIN_SECONDS: float = float(
    _optional("LINKEDIN_SEARCH_TYPING_CHAR_MIN_SECONDS", "0.055")
)
LINKEDIN_SEARCH_TYPING_CHAR_MAX_SECONDS: float = float(
    _optional("LINKEDIN_SEARCH_TYPING_CHAR_MAX_SECONDS", "0.11")
)
LINKEDIN_SEARCH_TYPING_OPERATOR_PAUSE_MIN_SECONDS: float = float(
    _optional("LINKEDIN_SEARCH_TYPING_OPERATOR_PAUSE_MIN_SECONDS", "0.18")
)
LINKEDIN_SEARCH_TYPING_OPERATOR_PAUSE_MAX_SECONDS: float = float(
    _optional("LINKEDIN_SEARCH_TYPING_OPERATOR_PAUSE_MAX_SECONDS", "0.45")
)
LINKEDIN_SEARCH_TYPING_THOUGHT_PAUSE_MIN_SECONDS: float = float(
    _optional("LINKEDIN_SEARCH_TYPING_THOUGHT_PAUSE_MIN_SECONDS", "0.30")
)
LINKEDIN_SEARCH_TYPING_THOUGHT_PAUSE_MAX_SECONDS: float = float(
    _optional("LINKEDIN_SEARCH_TYPING_THOUGHT_PAUSE_MAX_SECONDS", "0.90")
)
LINKEDIN_SEARCH_TYPING_PRE_SUBMIT_MIN_SECONDS: float = float(
    _optional("LINKEDIN_SEARCH_TYPING_PRE_SUBMIT_MIN_SECONDS", "0.40")
)
LINKEDIN_SEARCH_TYPING_PRE_SUBMIT_MAX_SECONDS: float = float(
    _optional("LINKEDIN_SEARCH_TYPING_PRE_SUBMIT_MAX_SECONDS", "1.20")
)
LINKEDIN_SEARCH_TYPING_MEDIUM_TYPO_PROBABILITY: float = float(
    _optional("LINKEDIN_SEARCH_TYPING_MEDIUM_TYPO_PROBABILITY", "0.15")
)
LINKEDIN_SEARCH_TYPING_LONG_TYPO_PROBABILITY: float = float(
    _optional("LINKEDIN_SEARCH_TYPING_LONG_TYPO_PROBABILITY", "0.30")
)
LINKEDIN_SEARCH_TYPING_SECOND_TYPO_PROBABILITY: float = float(
    _optional("LINKEDIN_SEARCH_TYPING_SECOND_TYPO_PROBABILITY", "0.10")
)
LINKEDIN_SEARCH_TYPING_MAX_TYPOS: int = int(
    _optional("LINKEDIN_SEARCH_TYPING_MAX_TYPOS", "2")
)
# Candidate-note cadence. A note char's delay is the key-down to next-key-down
# interval with the key hold inside it, unlike the search knobs above, where
# the hold is added on top. With dwell on, a 500-char note averages ~53s at
# these defaults. The holds alone come to ~44s per 500 chars, so no setting
# below can take a note under that.
LINKEDIN_NOTE_TYPING_CHAR_MIN_SECONDS: float = float(
    _optional("LINKEDIN_NOTE_TYPING_CHAR_MIN_SECONDS", "0.05")
)
LINKEDIN_NOTE_TYPING_CHAR_MAX_SECONDS: float = float(
    _optional("LINKEDIN_NOTE_TYPING_CHAR_MAX_SECONDS", "0.10")
)
LINKEDIN_NOTE_TYPING_WORD_PAUSE_MIN_SECONDS: float = float(
    _optional("LINKEDIN_NOTE_TYPING_WORD_PAUSE_MIN_SECONDS", "0.02")
)
LINKEDIN_NOTE_TYPING_WORD_PAUSE_MAX_SECONDS: float = float(
    _optional("LINKEDIN_NOTE_TYPING_WORD_PAUSE_MAX_SECONDS", "0.06")
)
LINKEDIN_NOTE_TYPING_SENTENCE_PAUSE_MIN_SECONDS: float = float(
    _optional("LINKEDIN_NOTE_TYPING_SENTENCE_PAUSE_MIN_SECONDS", "0.30")
)
LINKEDIN_NOTE_TYPING_SENTENCE_PAUSE_MAX_SECONDS: float = float(
    _optional("LINKEDIN_NOTE_TYPING_SENTENCE_PAUSE_MAX_SECONDS", "0.60")
)
LINKEDIN_PROFILE_EXPAND_CLICK_DWELL_SECONDS: float = float(
    _optional("LINKEDIN_PROFILE_EXPAND_CLICK_DWELL_SECONDS", "0.25")
)
LINKEDIN_PROFILE_EXPAND_SETTLE_SECONDS: float = float(
    _optional("LINKEDIN_PROFILE_EXPAND_SETTLE_SECONDS", "0.5")
)
LINKEDIN_SAVE_LINGER_BASE_SECONDS: float = float(_optional("LINKEDIN_SAVE_LINGER_BASE_SECONDS", "3.6"))
LINKEDIN_SAVE_LINGER_MIN_SECONDS: float = float(_optional("LINKEDIN_SAVE_LINGER_MIN_SECONDS", "2.0"))
LINKEDIN_SAVE_LINGER_MAX_SECONDS: float = float(_optional("LINKEDIN_SAVE_LINGER_MAX_SECONDS", "6.0"))
LINKEDIN_SAVE_LINGER_MIN_CHUNKS_BACK: int = int(_optional("LINKEDIN_SAVE_LINGER_MIN_CHUNKS_BACK", "1"))
LINKEDIN_SAVE_LINGER_MAX_CHUNKS_BACK: int = int(_optional("LINKEDIN_SAVE_LINGER_MAX_CHUNKS_BACK", "2"))
LINKEDIN_REJECT_CLOSE_BASE_SECONDS: float = float(_optional("LINKEDIN_REJECT_CLOSE_BASE_SECONDS", "0.35"))
LINKEDIN_REJECT_CLOSE_MIN_SECONDS: float = float(_optional("LINKEDIN_REJECT_CLOSE_MIN_SECONDS", "0.2"))
LINKEDIN_REJECT_CLOSE_MAX_SECONDS: float = float(_optional("LINKEDIN_REJECT_CLOSE_MAX_SECONDS", "1.2"))
LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS: float = float(_optional("LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS", "0.5"))

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).parent.parent
# Writable runtime-data root: user-data directory for frozen apps or explicit overrides;
# otherwise PROJECT_ROOT/output. See shared/user_data_dir.py for resolution.
from shared.user_data_dir import output_dir as _resolve_output_dir
OUTPUT_DIR: Path = _resolve_output_dir()
