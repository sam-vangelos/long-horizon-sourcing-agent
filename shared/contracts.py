"""Frozen execution contracts.

These constants document the current execution contracts so their meanings
are explicit and testable.
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

# FACIAL_BORDERLINE is the third facial-stage state (see
# plans/perplexity-evidence-augmentation.md). Under the ternary posture the
# candidate reviewer persists it as-is; under the binary posture a returned
# FACIAL_BORDERLINE becomes PARSE_FAILURE. It is never aliased to FACIAL_YES.
ACTIVE_FACIAL_DECISIONS = frozenset({"FACIAL_YES", "FACIAL_NO", "FACIAL_BORDERLINE"})
COMPAT_FACIAL_DECISIONS = frozenset({"FACIAL_SKIP"})
FACIAL_DECISIONS = ACTIVE_FACIAL_DECISIONS | COMPAT_FACIAL_DECISIONS | FAILURE_DECISIONS

FULL_DECISIONS = frozenset({
    "SAVE",
    "REJECT",
    "INFERENTIAL_SAVE",
    "TRANSFERABLE_SAVE",
    "SIGNAL_SAVE",
}) | FAILURE_DECISIONS

SAVE_DECISIONS = frozenset({
    "SAVE",
    "INFERENTIAL_SAVE",
    "TRANSFERABLE_SAVE",
    "SIGNAL_SAVE",
})

# RETIRED 2026-09-16 (Sam's ruling: the agent IS the reviewer). The P4
# non-save review outcomes REVIEW_INFERRED / REVIEW_FLAGGED are gone from
# the live judgment contract: a full evaluation now ends either in the
# recruiter's pipeline (SAVE and its typed variants) or out of it (REJECT).
# A strong case carrying one conversation-resolvable question goes through
# INFERENTIAL_SAVE, which reaches the pipeline; nothing is parked for a
# human spot check, because there is no human spot check.
#
# The two names survive HERE ONLY as a read vocabulary. Live SQLite stores
# hold terminal rows that carry them, so every reader that hydrates,
# resumes, reports on, or re-judges historical rows keys on this set. No
# writer may produce them: ``FULL_DECISIONS`` above is the write contract.
LEGACY_REVIEW_DECISIONS = frozenset({
    "REVIEW_INFERRED",
    "REVIEW_FLAGGED",
})

# Legacy alongside the decisions above: the reason codes those historical
# rows carry. Kept so a stored review payload still renders and re-judges;
# no live verdict can set one.
REVIEW_REASON_CODES = frozenset({
    "spot_check",
    "inferred_high_priority",
    "needs_more_evidence",
    "identity_unclear",
    "source_gap",
})

# LinkedIn full-evaluation judgment currency. These bounded values
# are shared by the forced-tool and legacy-text contracts so provider routing
# cannot create a second taxonomy for allocator-facing semantics.
EVIDENCE_RECENCY_VALUES = frozenset({"CURRENT", "RECENT", "STALE"})
LEVEL_ALIGNMENT_VALUES = frozenset({"ALIGNED", "ABOVE", "BELOW", "UNCLEAR"})
OPPORTUNITY_COHERENCE_VALUES = frozenset({"COHERENT", "INCOHERENT", "UNCLEAR"})
CALIBER_VALUES = frozenset({"STRONG", "SOLID", "WEAK", "UNKNOWN"})
OUTREACH_TIERS = frozenset({"PRIORITY", "STANDARD"})
# The bounded set a JUDGE may choose from. IDENTITY_UNRESOLVED landed with
# the review retirement (2026-09-16): an unreadable profile or contradictory
# identity evidence used to be parked for a human, and now resolves as the
# rejection it always was.
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
    "IDENTITY_UNRESOLVED",
})

# Codes the SYSTEM stamps, never offered to a model. REVIEW_RETIRED marks a
# verdict that named a retired review outcome and was mapped to REJECT at
# the parse boundary — countable, not silently absorbed into a parse
# failure. Kept out of REJECT_REASON_CODES so it never enters a tool schema
# enum or the template's REJECT_REASON list.
SYSTEM_REJECT_REASON_CODES = frozenset({
    "REVIEW_RETIRED",
})

# Every reject reason a persisted row may legitimately carry.
ALL_REJECT_REASON_CODES = REJECT_REASON_CODES | SYSTEM_REJECT_REASON_CODES


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
# Target candidate lifecycle contract
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
    "alumni_blacklist_check_failed",
    "alumni_blacklist_rejected",
    "competitor_facet_demote_refused",
    "competitor_facet_restored",
    "competitor_facet_skip",
    "competitor_facet_stamped",
    "employer_scope_check_failed",
    "employer_scope_rejected",
    "api_budget_exhausted",
    "architecture_pivot",
    "bias_alert",
    "block_adaptation",
    "cadence_pause",
    "cadence_pause_timing",
    "card_focus_timing",
    "card_snapshot_timing",
    "card_extract_error",
    "candidate_dedup_blocked",
    "candidate_note_failed",
    "candidate_note_written",
    "candidate_opened",
    "candidate_review_recorded",
    "candidate_saved",
    "contamination_reset_boundary",
    "circuit_breaker",
    "checkpoint_progress_timing",
    "budget_exhausted",
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
    "linkedin_facet_reconciliation",
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
    "panel_close_strategy",
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
    "profile_panel_unavailable",
    "report_completed",
    "report_started",
    "run_degraded",
    "run_report_error",
    "run_report_generated",
    "run_snapshot_finalized",
    "resume_committed_fastforward_exhausted",
    "resume_fastforward_exhausted",
    "save",
    "search_qualification_removed",
    "search_string_lint_blocked",
    "search_surface_reanchor_failed",
    "search_surface_reanchored",
    "session_location_applied",
    "session_location_reasserted",
    "session_location_resolved",
    "strategy_completed",
    "strategy_started",
    "surface_applied",
    "surface_intended",
    "surface_ownership_lost",
    "string_complete",
    "string_error",
    "string_executed",
    "string_results",
    "string_resumed",
    "string_started",
    "total_page_cap_reached",
    "zero_results_after_prior_results",
})
