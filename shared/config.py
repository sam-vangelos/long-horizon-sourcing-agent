"""Configuration loader. Reads .env and provides typed access to all settings.

Note: For Acme corporate proxy, you may need:
    export NODE_TLS_REJECT_UNAUTHORIZED=0

Frozen-app deployment (Phase 0 ``userdata`` slice): when Cloris runs
inside a signed .app bundle, the bundle is read-only — PROJECT_ROOT
resolves to a path inside ``Cloris.app/Contents/Resources/`` after
PyInstaller extraction and cannot be written to. ``shared.user_data_dir``
detects that case and we layer ``.env`` loads accordingly: the
recipient's writable ``~/Library/Application Support/Cloris/.env``
takes priority, with the project-root ``.env`` falling through for
dev. ``OUTPUT_DIR`` resolves the same way.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from shared.user_data_dir import (
    cloris_user_data_dir,
    should_use_user_data_dir,
)

# .env layering. The user-data ``.env`` (written by the in-product API
# key entry on first launch) is loaded first so its values seed the
# environment. ``override=False`` on the second load means the
# project-root ``.env`` provides defaults for any keys the user-data
# ``.env`` did not set, without ever clobbering recipient-entered
# credentials. Dev workflow is unchanged: when the user-data path is
# disabled, we just load PROJECT_ROOT/.env directly the way we always
# have.
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

ANTHROPIC_API_KEY: str = _agent_key("ANTHROPIC_API_KEY")
OPENAI_API_KEY: str = _agent_key("OPENAI_API_KEY")
GOOGLE_API_KEY: str = _agent_key("GOOGLE_API_KEY")
PERPLEXITY_API_KEY: str = _agent_key("PERPLEXITY_API_KEY")
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
# Output ceiling for the Anthropic cheap tier (profile/snippet extraction).
# Raised from a hardcoded 8192 on 2026-08-11: a 43KB profile's extraction
# JSON exceeded 8192 output tokens (finish_reason=max_tokens), the truncated
# JSON failed to parse, and the candidate wedged in pending-full recovery —
# every subsequent session died re-attempting her (PAE campaign, CLO-147).
# A cap is not a target: raising it only pays for tokens actually generated.
CHEAP_MODEL_MAX_TOKENS: int = int(_optional("CHEAP_MODEL_MAX_TOKENS", "16384"))

# --- Startup key validation ---

_REQUIRED_AT_STARTUP: tuple[str, ...] = ("ANTHROPIC_API_KEY",)


class MissingRequiredKeyError(RuntimeError):
    """Raised by validate_startup_keys() when a required env key is absent."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            f"Cloris cannot start: required env key(s) missing: "
            f"{', '.join(missing)}. Set them in .env or the environment."
        )


def validate_startup_keys() -> None:
    """Raise MissingRequiredKeyError if any required API key is unset.

    Skip when CLORIS_SKIP_STARTUP_VALIDATION=1 (test environments that
    don't set live API keys).
    """
    if os.getenv("CLORIS_SKIP_STARTUP_VALIDATION", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        return
    missing = [k for k in _REQUIRED_AT_STARTUP if not os.getenv(k, "").strip()]
    if missing:
        raise MissingRequiredKeyError(missing)

# --- Model Names ---
CHEAP_MODEL_NAME: str = _optional("CHEAP_MODEL_NAME", "gpt-4o-mini")
OPUS_MODEL_NAME: str = _optional("OPUS_MODEL_NAME", "claude-opus-4-6")
FACIAL_MODEL_NAME: str = _optional("FACIAL_MODEL_NAME", OPUS_MODEL_NAME)
# --- Model tiers (leverage-tiered routing; defaults preserve today's behavior) ---
# STRATEGY tier: the plan-authoring calls (preflight, formation, block
# adaptation, drift/variant/narrow planners) — a handful of calls per run,
# each setting the table for hours of downstream execution. FULL_EVAL tier:
# the per-candidate volume stage (facial has carried its own seam above since
# the 5x-cost note). Tier reassignment happens by env flip AFTER the shadow
# re-measure (plans/sourcing-generality-hardening.md item 19), never by
# changing these defaults. Non-Anthropic ids are rejected at the client until
# provider dispatch lands with the promotion itself.
STRATEGY_MODEL_NAME: str = _optional("STRATEGY_MODEL_NAME", OPUS_MODEL_NAME)
FULL_EVAL_MODEL_NAME: str = _optional("FULL_EVAL_MODEL_NAME", OPUS_MODEL_NAME)
# Outreach tier: stored GitHub recruiter copy (one candidate at a time).
OUTREACH_MODEL_NAME: str = _optional("OUTREACH_MODEL_NAME", OPUS_MODEL_NAME)
FULL_EVAL_PIPELINE_ENABLED: bool = _env_flag("FULL_EVAL_PIPELINE_ENABLED", "false")

# GLM-5.2 judgment-runtime experiment. Every behavior-changing switch defaults
# off; model constants are explicit so launchers can route only facial/full
# stages without repointing OPUS_MODEL_NAME and silently changing other calls.
FIREWORKS_STANDARD_MODEL_NAME: str = "accounts/fireworks/models/glm-5p2"
FIREWORKS_FAST_MODEL_NAME: str = "accounts/fireworks/routers/glm-5p2-fast"
FIREWORKS_JUDGMENT_POLICY_ENABLED: bool = _env_flag(
    "FIREWORKS_JUDGMENT_POLICY_ENABLED", "false"
)
FIREWORKS_JUDGMENT_STREAM_ENABLED: bool = _env_flag(
    "FIREWORKS_JUDGMENT_STREAM_ENABLED", "false"
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
FIREWORKS_PROMPT_AFFINITY_ENABLED: bool = _env_flag(
    "FIREWORKS_PROMPT_AFFINITY_ENABLED", "false"
)
LINKEDIN_V2_FACIAL_CONTRACT: str = _optional(
    "LINKEDIN_V2_FACIAL_CONTRACT", "legacy"
).strip().lower()
LINKEDIN_V2_FULL_CONTRACT: str = _optional(
    "LINKEDIN_V2_FULL_CONTRACT", "legacy"
).strip().lower()
# Deterministic HYPHENATION expansion in the Boolean compiler.
#
# Why: the strategy doctrine already mandates surface variants per OR group (the
# "real variants" clause in the strategy system prompt), and _check_morphology
# already warns when they are absent. Neither enforces. Measured 2026-07-27 on
# the Principal-Research-Engineer brief, counted plan-wide over distinct
# hyphenated terms: claude-fable-5 emitted 0 of 18 with their spaced twin,
# gpt-5.6-sol 11 of 22 — both at max effort, both against a rule stated in the
# prompt. Hyphen<->space is orthographic and therefore mechanically derivable,
# so compliance moved out of the generation and into
# linkedin/boolean_compiler.derive_surface_variants.
#
# Scope is ONE axis on purpose. A number axis was built and removed the same
# day: pluralising is morphological, needs a lexicon, and rules alone produced
# "kubernete", "mlop", "devop", "verls", "jaxes", "numpies" and "datas" from
# real tooling vocabulary. The brief field that would have exempted them,
# domain_depth_objects, is wired to no producer, so the do-not-vary set is
# always empty in production and those guesses would have shipped. See the
# derive_surface_variants docstring.
#
# Default false, and NOT set by tools/launch_prre_code.sh — the 2026-07-27
# campaign runs without it. It widens every executed OR group and has never run
# live, so switching it on is a deliberate export, not a launcher default.
# Facial triage tightening — the bias monitor's one VERDICT-affecting path.
# When a string's facial YES rate exceeds 2x the brief's expected_yes_rate_high,
# the orchestrator injects "require TWO strong positive signals instead of one"
# into the facial prompt for the rest of the string. Default OFF (2026-07-30):
# the band is a brief-level preflight GUESS while precision is per-string, so a
# deliberately dense probe (a page of 25 on-target forward-deployed engineers,
# measured live) reads as judge bias and gets silently stricter triage —
# punishing exactly the strings that work. The 2026-07-04 telemetry demotion
# already routed the dense-vein-vs-loosened-judge call to the adaptation model
# via the block report; this flag completes it. All monitor telemetry
# (facial_rate_anomaly alerts, bias_expected_band, string_context) stays live.
LINKEDIN_FACIAL_TIGHTENING_ENABLED: bool = _env_flag(
    "LINKEDIN_FACIAL_TIGHTENING_ENABLED", "false"
)

# --- campaign persistence (2026-07-27, operator request) ---
# The multi-session day cycle historically EXITED the process whenever the
# governor refused a session (daily session cap, 24h open cap, forced backoff)
# and on any session error — so an unattended campaign formally died at end of
# day and needed a human relaunch. Operator ruling: the process must REMAIN
# ACTIVE — sleep through governor windows and resume when they reopen; short
# dormant gaps are acceptable. None of this raises any volume cap: sessions/day,
# opens/session, and opens/24h are enforced exactly as before — persistence
# changes when the process exits, never how much it sources.
LINKEDIN_CAMPAIGN_PERSIST: bool = _env_flag("LINKEDIN_CAMPAIGN_PERSIST", "false")
# Consecutive UNCLASSIFIED session errors ("error: <Type>") the persistent
# cycle may absorb by sleeping a dormant period and resuming, before giving up
# and raising. Deterministic regime errors (geography_regime_error,
# constraint_manifest_error, preflight_regime_error) and operator interrupts
# are NEVER retried — retrying a config error loops forever. A clean
# session-duration-cap session resets the counter. 0 = today's behavior.
LINKEDIN_SESSION_ERROR_RETRIES: int = int(
    _optional("LINKEDIN_SESSION_ERROR_RETRIES", "0")
)
LINKEDIN_BROWSER_CRASH_RESUME_ENABLED: bool = _env_flag(
    "LINKEDIN_BROWSER_CRASH_RESUME_ENABLED", "false"
)
# Run detect_recruiter_health on the every-loop happy path
# (_ensure_browser_healthy) and trip force_backoff on
# blocked_or_rate_limited. Also short-circuits the SWG recovery ladder
# after one try-again so a blocked page is not re-probed via reload/goto.
# Flag OFF preserves today's disconnect-only classifier path and full SWG
# escalation.
LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED: bool = _env_flag(
    "LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED", "false"
)
# Contain the two per-candidate re-match failures inside owner pending-full
# recovery instead of raising. LinkedIn reorders Recruiter results between
# sessions, so the resume re-match by profile URL can miss the card a facial
# YES/BORDERLINE was captured against — and that raise is deterministic across
# retries: it killed every resume attempt of the 2026-07-31 campaign (10:30,
# 10:47, 21:47 plus two absorbed retries, all "0 profile opens"). Flag ON
# settles the unmatched candidate terminally on canonical state, receipts it as
# ``pending_full_recovery_abandoned`` so an operator can re-run deliberately,
# and lets the remaining owned snippets recover. Pagination failures and the
# end-of-loop unsettled check keep raising in BOTH states.
LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED: bool = _env_flag(
    "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", "false"
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

# Enumerate the domain's named artifacts as strategy vocabulary before forming
# strings. See shared/vocabulary_enumeration.py for the measurement that
# motivated it: the 2026-07-27 strategy step produced 72 strings carrying nine
# benchmark names, all head-of-distribution, while the SAME model asked directly
# with the SAME capability areas returned 292 named artifacts including 98
# benchmarks. Satisficing, not missing knowledge — the strategy call asks for
# search strings and gets good search strings, and nothing ever asks for the
# thirtieth benchmark.
#
# Costs one extra strategy-tier call per fresh run, before string formation.
# Fail-soft: any failure yields no vocabulary, which is byte-identical to the
# current no-kit path.
LINKEDIN_VOCABULARY_ENUMERATION_ENABLED: bool = _env_flag(
    "LINKEDIN_VOCABULARY_ENUMERATION_ENABLED", "false"
)
# Union an external-research pass into the enumeration. Measured the same day:
# Perplexity deep-research returned 222 artifacts for $0.058 with only 106
# overlapping Fable's 292, so the providers are substantially complementary —
# but it found FEWER of the operator's named misses (5 of 10 against Fable's 6),
# and neither found the newest three. Worth its cost as a union; not a
# replacement, and not the thing that closes the bleeding edge.
LINKEDIN_VOCABULARY_ENUMERATION_RESEARCH_ENABLED: bool = _env_flag(
    "LINKEDIN_VOCABULARY_ENUMERATION_RESEARCH_ENABLED", "false"
)
LINKEDIN_SURFACE_VARIANT_EXPANSION_ENABLED: bool = _env_flag(
    "LINKEDIN_SURFACE_VARIANT_EXPANSION_ENABLED", "false"
)

LINKEDIN_FACIAL_CONCURRENCY_ENABLED: bool = _env_flag(
    "LINKEDIN_FACIAL_CONCURRENCY_ENABLED", "false"
)
LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED: bool = _env_flag(
    "LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED", "false"
)
# CLO-177: contain a provider/transport fault during a FULL evaluation the way
# CLO-69 contains facial page faults — settle that one candidate as
# JUDGMENT_FAILURE (re-meetable later; not dedup-blocking) and keep the string
# alive, instead of the raise unwinding into the per-string handler and ending
# the session. Bounded by the consecutive-faults ceiling below: a sustained
# provider outage still ends the session legibly instead of grinding through
# every candidate on the page.
LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED: bool = _env_flag(
    "LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED", "false"
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
# Northwind trial scope is deliberately LinkedIn-only. Dev defaults stay broad so
# non-trial tests and module work are not silently hidden; packaged trial
# entrypoints set CLORIS_TRIAL_MODE=true before config import.
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
# When true, market intelligence is refreshed only after the ENTIRE sourcing
# strategy has been executed — every string work unit for the run in a terminal
# status — rather than after any individually clean-exiting run.
#
# Why: `honest_completion` (linkedin/orchestrator.py) gates enrichment on a
# clean per-RUN exit, which a multi-session campaign satisfies repeatedly. The
# SPL campaign at output/market_intelligence/strategic_project_lead__* shows
# run_count 5 and 3 — the artifact was rewritten once per completing run, so the
# operator saw a sequence of partial reports rather than one settled one. A run
# that exits cleanly having executed 1 of 26 strings is honest about the exit
# and useless as a market read.
#
# Default false preserves existing behaviour for every other brief; the launcher
# opts a specific campaign in.
MARKET_INTEL_REQUIRE_STRATEGY_COMPLETE: bool = _env_flag(
    "MARKET_INTEL_REQUIRE_STRATEGY_COMPLETE", "false"
)

# --- LinkedIn external evidence augmentation (Perplexity-backed) ---
# Slice 1 of the perplexity-evidence-augmentation feature. All defaults are
# safe / disabled: the feature is gated off until slice 2 wires it in.
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

# Step B of the FACIAL_BORDERLINE promotion plan (slice 13). When True, the
# facial-triage prompt offers a three-class output (YES/BORDERLINE/NO),
# the parser recognizes BORDERLINE, and the orchestrator translates
# BORDERLINE -> FACIAL_YES at the parser-output boundary (alias-to-YES).
# Persistence and counters stay binary at Step B; canonical state never
# observes BORDERLINE. Step C is where BORDERLINE becomes a real third
# state. Default off; production behavior under flag-off is byte-identical
# to pre-Step-B.
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
SHADOW_FACIAL_MODEL_ENABLED: bool = _env_flag("SHADOW_FACIAL_MODEL_ENABLED", "false")
# Slug CONFIRMED LIVE 2026-07-03: accounts/fireworks/models/glm-5p2 returned
# a real chat completion against this account's Fireworks key (verified
# against the fireworks.ai model-library pages, which list "GLM 5.2" at
# https://fireworks.ai/models/fireworks/glm-5p2 with API id
# "accounts/fireworks/models/glm-5p2"; also cross-checked via a live
# /chat/completions call). Not a guess.
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
# Minimum completion budget for Fireworks primary calls. Fireworks counts a
# reasoning model's chain-of-thought against max_tokens (GLM ~70 tok/s,
# reasoning tails observed to ~39K chars), so Anthropic-calibrated caller
# caps starve generation: the first GLM-primary session (2026-07-07) hit
# five finish_reason=length failures across the glance (4096),
# profile-extraction (8192), and full-eval (8192) tiers. A cap is not a
# target — raising it only pays for tokens actually generated (~$0.06 at a
# fully-used 16384 on GLM pricing). 16384 is the shadow-era proven number
# (full-eval shadow cap fix, 2026-07-05).
FIREWORKS_PRIMARY_MIN_MAX_TOKENS: int = int(
    _optional("FIREWORKS_PRIMARY_MIN_MAX_TOKENS", "16384")
)
# Optional process-local primary Fireworks spend ceiling. Zero keeps ordinary
# and offline paths unchanged; the isolated live glmopt launcher requires and
# pins a positive operator cap.
FIREWORKS_PRIMARY_MAX_COST_USD: float = float(
    _optional("FIREWORKS_PRIMARY_MAX_COST_USD", "0")
)
# Fire-and-forget dispatch for the shadow comparison. The GLM call has
# ~58s mean latency with outliers to ~240s; on the 2026-07-05 live run 33
# synchronous shadow calls blocked roughly 30-60 minutes of a 166-minute
# session. Nothing in the run consumes the comparison synchronously (it is
# offline analytics: one run-log event + one token-cost row), so when this
# flag is on (default) shared/judger.py runs the comparison on a single
# background worker thread instead of inline in the judge path. When off,
# behavior is byte-identical to the pre-executor synchronous inline call —
# the test/CI escape hatch and the rollback lever.
SHADOW_ASYNC_ENABLED: bool = _env_flag("SHADOW_ASYNC_ENABLED", "true")
# Per-attempt timeout for the Fireworks shadow calls. GLM's mean full-eval
# latency is ~58s, so the old hardwired 60s ceiling cut the legitimate tail —
# first live batch-facial shadow timed out at exactly 2×60s (both attempts).
# 120 was the next value; measured against it (2026-07-05 SPL live run +
# replay): 2 of 21 full-evals timed out at exactly 2×120s, and the replay's
# GLM call succeeded only via retry (fail @120 + success @87 = 207s total).
# GLM generates ~70 tok/s (8,192 tokens in ~118s, captured), so a full
# 16,384-token generation — the full-eval shadow's cap, see shared/judger.py
# — legitimately needs ~234s. 300 covers that with margin. Shadows run off
# the hot path; only the end-of-run judge drain (360s bound,
# linkedin/orchestrator.py report build) interacts with it.
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
SHADOW_STRATEGY_ENABLED: bool = _env_flag("SHADOW_STRATEGY_ENABLED", "false")
# claude-fable-5 rejects sampling params; opus_llm and opus_llm_cached pass
# neither. Both route thinking kwargs through _thinking_request_kwargs, whose
# guard sends summarized thinking only to Fable/Mythos and NO thinking param
# to Opus-family models.
SHADOW_STRATEGY_MODEL_NAME: str = _optional(
    "SHADOW_STRATEGY_MODEL_NAME", "claude-fable-5"
)

# --- Browser ---
CDP_URL: str = _optional("CDP_URL", "http://127.0.0.1:9222")

# --- Behavior ---
# Default 25 approximates the deepest productive run observed; 0 remains the
# explicit unbounded escape hatch (Sam's 2026-07-07 call).
MAX_PAGES_PER_STRING: int = int(_optional("MAX_PAGES_PER_STRING", "25"))
PAGE_DELAY_SECONDS: float = float(_optional("PAGE_DELAY_SECONDS", "3"))
PROFILE_DELAY_SECONDS: float = float(_optional("PROFILE_DELAY_SECONDS", "2"))

# --- Cadence pause (anti-detection) ---
# After this many minutes of continuous activity, pause for a human-like break.
# Both values are jittered ±20% at runtime to avoid metronomic patterns.
CADENCE_INTERVAL_MINUTES: float = float(_optional("CADENCE_INTERVAL_MINUTES", "30"))
CADENCE_PAUSE_SECONDS: float = float(_optional("CADENCE_PAUSE_SECONDS", "120"))
LINKEDIN_TIMING_TELEMETRY_ENABLED: bool = _env_flag(
    "LINKEDIN_TIMING_TELEMETRY_ENABLED", "false"
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
# Phase 2 hop 4 (slice E): deterministic circuit-breaker on structured demotions.
# After this many structured-control demote-and-proceed events on a lane (a hybrid /
# filter_led / boolean lane whose structured dim dropped at apply while the keyword
# landed), the variant/drift planners STOP offering the promote/structured lever AND
# the deterministic parse layer (_resolve_structured_controls) ENFORCES the closure:
# even a disobeying model that still emits surface:"hybrid" + structured_controls has
# its promote stripped and surface coerced to "boolean", so the lane cannot re-promote.
# The execution metric CONSTRAINS what the LLM may propose AND what the parser will
# build. It does NOT override the deterministic gate, and decide_variant_lifecycle
# stays untouched.
SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT: int = int(
    _optional("SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT", "2")
)
# Minimum distinct title FAMILIES a title filter must span before it is allowed
# to bound a live search. Sam's ruling 2026-08-05 (CLO-73): "IF we're gonna be
# using titles as a filter, then it needs to be more expansive/comprehensive
# than this. We're missing out on entire talent pools otherwise; I'd rather just
# use keyword filters if this does not happen."
#
# The defect this closes: a title PROMOTE was graduated on one test only — does
# the label appear in the results-rail clusters — and the rail ranks by VOLUME,
# so the qualifying labels are definitionally the pool's MODAL titles. On the
# PRRE campaign that produced titles=["Software Engineer", "Applied Scientist"]
# on a Principal-Research-Engineer search, cutting the pool from ~1,200 results
# to ~840 and excluding Research Engineer, Research Scientist, Member of
# Technical Staff and every other non-modal title in one move.
#
# Below the floor the whole promotion is dropped and the variant stays
# keyword-only — the fail-closed direction Sam named. Set 0 to disable the
# floor (restores the pre-2026-08-05 behavior; not recommended).
LINKEDIN_TITLE_FILTER_MIN_FAMILIES: int = int(
    _optional("LINKEDIN_TITLE_FILTER_MIN_FAMILIES", "6")
)
# P3.7: block-adaptation SPRT/cooldown gate, configurable via env. Defaults
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
# Phase 2 hop 4 (slice F): posture-aware lifecycle windows. A filter-led /
# structured search is legitimately NARROWER than a keyword search, so the
# keyword-tuned lifecycle gate (classify_result_window -> decide_variant_lifecycle)
# would mis-classify a good structured probe as too_narrow and ABANDON it. At
# variant CONSTRUCTION a filter-bearing variant (surface in {hybrid,
# structured_only} OR non-empty structured_filters) has its healthy target window
# scaled DOWN by this factor, so the decision function reads an already-scaled
# window and stays a PURE, byte-stable function — no posture leaks into the gate.
# A boolean variant uses the UNSCALED window (factor NOT applied), so the default
# path is byte-identical.
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
LINKEDIN_SLOT_ON_FIRST_OPEN: bool = _env_flag("LINKEDIN_SLOT_ON_FIRST_OPEN", "false")

# --- LinkedIn cadence trim knobs ---
LINKEDIN_CADENCE_READ_FIX_ENABLED: bool = _env_flag(
    "LINKEDIN_CADENCE_READ_FIX_ENABLED", "false"
)
LINKEDIN_SECTION_DIRECTED_READ_ENABLED: bool = _env_flag(
    "LINKEDIN_SECTION_DIRECTED_READ_ENABLED", "false"
)
LINKEDIN_TYPING_DWELL_ENABLED: bool = _env_flag("LINKEDIN_TYPING_DWELL_ENABLED", "false")
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
# OUTPUT_DIR is the writable root for all per-state runtime data
# (state dirs, runtime_state.sqlite3, projection JSONLs). When Cloris
# runs as a frozen .app, this relocates under
# ``~/Library/Application Support/Cloris/output/`` because the bundle
# is read-only. Dev (running from the repo) preserves the historical
# ``PROJECT_ROOT/output`` layout — see ``shared/user_data_dir.py`` for
# the resolution rules and the ``CLORIS_USER_DATA_DIR`` opt-in for
# tests / power users.
from shared.user_data_dir import output_dir as _resolve_output_dir
OUTPUT_DIR: Path = _resolve_output_dir()
