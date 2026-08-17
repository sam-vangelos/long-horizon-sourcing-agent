"""Frozen contracts for the Phase 0 baseline.

These constants intentionally document the current execution contracts without
changing runtime behavior yet. Later phases can wire code onto these contracts,
but Phase 0 is about making the current meanings explicit and testable.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Brief / policy contracts
# ---------------------------------------------------------------------------

BRIEF_FAMILIES = frozenset({"legacy", "v2"})

NORMALIZED_BRIEF_REQUIRED_FIELDS = frozenset({
    "id",
    "role_title",
    "role_description",
    "minimum_bar",
    "archetypes",
    "noise_archetypes",
    "permanent_filters",
    "raw",
})

V2_BRIEF_REQUIRED_FIELDS = frozenset({
    "role_title",
    "capability_areas",
    "depth_distinction",
    "non_fit_patterns",
    "employer_signal_rules",
    "facial_calibration",
    "bias_controls",
})


# ---------------------------------------------------------------------------
# Decision contracts
# ---------------------------------------------------------------------------

FAILURE_DECISIONS = frozenset({"PARSE_FAILURE", "JUDGMENT_FAILURE"})

# FACIAL_BORDERLINE was added at Step A of the slice 12 promotion plan
# (see plans/perplexity-evidence-augmentation.md and the
# execution-boundary audit report). At Step A the constant is a *type-system
# widening* only -- no parser, validator, orchestrator, runtime-state, or
# persistence layer recognizes it yet. Step B will widen the parser and
# orchestrator behind a feature flag with FACIAL_BORDERLINE aliasing to
# FACIAL_YES at the orchestrator boundary. Step C wires it as a real third
# state. Until then this constant is intentionally dark.
ACTIVE_FACIAL_DECISIONS = frozenset({"FACIAL_YES", "FACIAL_NO", "FACIAL_BORDERLINE"})
COMPAT_FACIAL_DECISIONS = frozenset({"FACIAL_SKIP"})
FACIAL_DECISIONS = ACTIVE_FACIAL_DECISIONS | COMPAT_FACIAL_DECISIONS | FAILURE_DECISIONS

FULL_DECISIONS = frozenset({
    "SAVE",
    "REJECT",
    "INFERENTIAL_SAVE",
    "TRANSFERABLE_SAVE",
    "SIGNAL_SAVE",
    "REVIEW_INFERRED",
    "REVIEW_FLAGGED",
}) | FAILURE_DECISIONS

SAVE_DECISIONS = frozenset({
    "SAVE",
    "INFERENTIAL_SAVE",
    "TRANSFERABLE_SAVE",
    "SIGNAL_SAVE",
})

# P4: bounded non-save review outcomes for ambiguous high-priority
# candidates. These are full-stage terminal decisions, distinct from
# FACIAL_BORDERLINE (which remains a facial-stage state). They MUST NOT
# trigger LinkedIn save side effects and MUST NOT inflate save counts.
# Internal reason codes carry the nuance; the visible taxonomy is bounded
# to the two outcomes below. Lane/run review budgets are enforced at the
# orchestrator dispatch site (structural-evidence guard); per-lane caps
# land with lane metrics in P5.
NON_SAVE_REVIEW_DECISIONS = frozenset({
    "REVIEW_INFERRED",
    "REVIEW_FLAGGED",
})

REVIEW_REASON_CODES = frozenset({
    "spot_check",
    "inferred_high_priority",
    "needs_more_evidence",
    "identity_unclear",
    "source_gap",
})

# LinkedIn full-evaluation judgment currency (TUR-13). These bounded values
# are shared by the forced-tool and legacy-text contracts so provider routing
# cannot create a second taxonomy for allocator-facing semantics.
EVIDENCE_RECENCY_VALUES = frozenset({"CURRENT", "RECENT", "STALE"})
LEVEL_ALIGNMENT_VALUES = frozenset({"ALIGNED", "ABOVE", "BELOW", "UNCLEAR"})
OPPORTUNITY_COHERENCE_VALUES = frozenset({"COHERENT", "INCOHERENT", "UNCLEAR"})
CALIBER_VALUES = frozenset({"STRONG", "SOLID", "WEAK", "UNKNOWN"})
OUTREACH_TIERS = frozenset({"PRIORITY", "STANDARD"})
REJECT_REASON_CODES = frozenset({
    "HARD_GATE",
    "NON_FIT",
    "CAPABILITY_INSUFFICIENT",
    "EVIDENCE_STALE",
    "OVER_LEVEL",
    "UNDER_LEVEL",
    "INCOHERENT_MOVE",
    "DEPTH_CONSUMER",
    "BAR_ORDINARY",
})


# ---------------------------------------------------------------------------
# Current execution status contracts
# ---------------------------------------------------------------------------

LINKEDIN_STRING_STATUSES = frozenset({
    "queued",
    "in_progress",
    "done",
    "skipped",
    "error",
})

GITHUB_QUERY_STATUSES = frozenset({
    "queued",
    "in_progress",
    "done",
    "skipped",
    "error",
})


# ---------------------------------------------------------------------------
# Target candidate lifecycle contract (Phase 2 target, frozen in Phase 0)
# ---------------------------------------------------------------------------

TARGET_CANDIDATE_LIFECYCLE = (
    "discovered",
    "snippet_extracted",
    "facial_started",
    "facial_terminal",
    "full_started",
    "full_terminal",
    "failed_retryable",
    "failed_terminal",
)


# ---------------------------------------------------------------------------
# Event vocabulary currently emitted via shared.storage.log_event()
# ---------------------------------------------------------------------------

RUN_LOG_EVENTS = frozenset({
    "adaptation_checkpoint_cadence",
    "adaptation_decision",
    "adaptation_error",
    "activity_saturation_context",
    "api_budget_exhausted",
    "architecture_pivot",
    "bias_alert",
    "block_adaptation",
    "cadence_pause",
    "cadence_pause_timing",
    "card_focus_timing",
    "card_snapshot_timing",
    "card_extract_error",
    "candidate_opened",
    "candidate_dedup_blocked",
    "candidate_saved",
    "circuit_breaker",
    "checkpoint_progress_timing",
    "budget_exhausted",
    "candidate_review_recorded",
    "designer_run_end",
    "early_exit",
    "execution_started",
    "external_evidence_enriched_judge_failed",
    "external_evidence_failed",
    "external_evidence_fetched",
    "external_evidence_shadow_unhandled_exception",
    "external_evidence_skipped",
    "external_evidence_unavailable",
    "fallback_candidate_discovered",
    "fallback_discovery_attempt",
    "facial_error",
    "facial_page_judgment_timing",
    "facial_shadow_comparison",
    "full_page_judgment_timing",
    "full_shadow_comparison",
    "full_pipeline_abort_cleanup_failed",
    "final_error",
    "forced_narrow",
    "forced_narrow_failed",
    "github_auth_failed",
    "go_back_error",
    "glance_assess",
    "glance_llm_error",
    "insufficient_data",
    "investigation_failed",
    "linkedin_block_exploitation",
    "linkedin_search_assess",
    "linkedin_search_mutation_applied",
    "linkedin_search_mutation_attempt",
    "linkedin_search_plan_failed",
    "linkedin_search_plan_variant_skipped",
    "market_intel_deferred",
    "market_intel_update_failed",
    "market_intel_updated",
    "page_allocator_shadow_checkpoint",
    "page_allocator_shadow_exhaustion",
    "page_allocator_shadow_poison",
    "page_allocator_active_checkpoint",
    "page_allocator_active_exhaustion",
    "page_allocator_active_actuation",
    "page_abandoned",
    "page_cap_reached",
    "page_observation_gap",
    "page_render_zero_slots",
    "panel_close_browser_disconnect",
    "panel_recovered",
    "panel_recovery_started",
    "panel_stuck",
    "pagination_exhausted",
    "pipeline_end",
    "pipeline_error",
    "pipeline_start",
    "pivot_blocked",
    "posture_report",
    "provider_unavailable",
    "profile_browser_disconnect",
    "profile_expand_timing",
    "profile_innertext_timing",
    "profile_open_timing",
    "profile_read_timing",
    "profile_activity_enrichment_failed",
    "profile_error",
    "report_completed",
    "report_started",
    "run_degraded",
    "run_report_error",
    "run_report_generated",
    "run_snapshot_finalized",
    "resume_committed_fastforward_exhausted",
    "resume_fastforward_exhausted",
    "save",
    "search_string_lint_blocked",
    "session_location_applied",
    "session_location_reasserted",
    "session_location_resolved",
    "strategy_completed",
    "strategy_started",
    "surface_applied",
    "surface_intended",
    "string_complete",
    "string_error",
    "string_executed",
    "string_results",
    "string_resumed",
    "string_started",
    "total_page_cap_reached",
    "zero_results_after_prior_results",
})
