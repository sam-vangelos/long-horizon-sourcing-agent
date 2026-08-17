"""Internal LinkedIn sourcing pipeline.

Production sourcing enters through ``python -m linkedin.session_orchestrator``;
``linkedin.run`` is browser-free rejudging only.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import math
import os
import random
import signal
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from shared.human_timing import human_delay, human_delay_correlated
from shared.output_paths import resolve_linkedin_state_dir, source_archive_root

from shared.schemas import (
    CandidateSnippet, CandidateProfileSummary, OpusDecision,
    SearchString, Progress, KitString, BlockReport, AdaptationResponse,
    ExecutionPlan, GlanceResult, TriggerDecision,
)
from linkedin.acquisition import (
    LinkedInAcquisitionDeps,
    LinkedInAcquisitionService,
    _is_browser_disconnect_error as _acquisition_is_browser_disconnect_error,
)
from linkedin.browser import (
    LinkedInBrowser,
    normalize_facet_value_for_compare,
    recruiter_project_search_url,
    # F4: imported rather than redefined. Both layers classify a page's project
    # — the browser to pick a tab, the orchestrator to refuse a save — and two
    # copies of that rule would let a bind and a guard disagree about which
    # pipeline the run is in.
    _recruiter_page_project_id,
)
from linkedin.timing_telemetry import RunLogTimingRecorder, emit_timing_event
from linkedin.page_allocator import (
    AllocationAction,
    AllocationVerdict,
    AllocatorArm,
    AllocatorPolicyError,
    PageObservation,
    allocate_page,
)
from linkedin.search_intelligence import (
    seed_structured_filters_onto_variants,
    LinkedInDriftAssessment,
    LinkedInExperimentState,
    LinkedInPageInsights,
    LinkedInSearchVariant,
    LinkedInStructuredFilters,
    LinkedInVariantSnapshot,
    _copy_filters,
    _drop_one_filter,
    bootstrap_experiment_state,
    result_window_for_count,
    scale_window_for_surface,
    spawn_rescue_variant_from_hint,
)
from linkedin.fallback_acquisition import (
    discover_fallback_candidates_for_string,
    fallback_mode_for_search_string,
    record_fallback_discovery,
)
from linkedin.fallback_search import FallbackSearchProvider
from linkedin.facial_batching import (
    FacialBatchContractError,
    FacialBatchFailureOutcome,
)
from linkedin.judgment_tool_contracts import (
    JudgmentToolContractError,
    generate_opaque_candidate_ids,
)
from linkedin.recruiter_recovery import capture_recovery_snapshot
from linkedin.search_mutation import LinkedInSearchMutationDeps, LinkedInSearchMutationExecutor
from linkedin.evaluation_cascade import CascadePolicy, ProfileProbe
from linkedin.side_effects import LinkedInSideEffectsDeps, LinkedInSideEffectsService
from linkedin.work_units import LinkedInWorkUnitDeps, LinkedInWorkUnitService
from linkedin import lane_projection as _lane_projection
from linkedin import surface_receipt as _surface_receipt
from linkedin.boolean_compiler import (
    BooleanNormalizationError,
    UbiquitousAndGateError,
    boolean_lint_context_from_brief,
    lint_generated_string,
    normalize_execution_work_item_boolean,
    proper_nouns_from_brief,
    summarize_kit_lint,
    ubiquitous_terms_from_brief,
)
from linkedin.matching_contract import render_adaptation_matching_guidance
from linkedin.adaptation_signal_state import (
    AdaptationGateConfig,
    AdaptationGateDecision,
    SearchSignalState,
    default_adaptation_gate_config,
    evaluate_adaptation_gate,
)
from shared.extractors import (
    extract_snippet_from_card_innertext,
    extract_profile_from_dom,
)
from shared.failures import (
    ApiBudgetExhaustedError,
    classify_runtime_failure,
    is_api_budget_exhausted_error,
    judgment_failure_decision,
    parse_failure_decision,
)
from shared.contracts import (
    FAILURE_DECISIONS,
    NON_SAVE_REVIEW_DECISIONS,
    OUTREACH_TIERS,
    REVIEW_REASON_CODES,
)
from shared.judger import (
    SAVE_FAMILY_DECISIONS,
    facial_judge,
    full_id_mismatch_retry_context,
    full_judge,
    full_judge_with_external_evidence,
    init_judger,
    is_failure_decision,
)
from shared.external_evidence import (
    fetch_external_candidate_evidence,
    should_request_external_evidence,
)
from shared.external_evidence.shadow_writer import (
    ShadowFullJudgmentRecord,
    compute_judgment_diff,
    record_shadow_full_judgment,
)
from shared.schemas import ExternalCandidateEvidence, ExternalEvidenceFailure
from shared.runtime_state import LinkedInRuntimeStateBridge, RuntimeStateLock, RuntimeStateStore
from shared.runtime_state.store import (
    LINKEDIN_STRING_KIND,
    TERMINAL_WORK_UNIT_STATUSES,
)

# Strategy completion uses the SUCCESSFUL-terminal set, not every terminal
# status: "error" is terminal for a work unit but is NOT executed coverage,
# and canonical resume (read_models.has_pending_work) agrees by counting
# anything outside done/skipped as pending.
_STRATEGY_COMPLETE_WORK_UNIT_STATUSES = frozenset({"done", "skipped"})
from shared.safety import LinkedInRecoveryService, RunSafetyCoordinator, RunStopReason
from shared.storage import append_jsonl, read_jsonl, read_jsonl_set, log_event, write_json, read_json
from shared.brief_loader import load_brief, Brief
from shared.bias_controls import BiasMonitor, DecisionRecord
from shared.search_memory import (
    build_search_memory_summary,
    extract_dominant_anchors,
    infer_domain_lane,
    normalize_family_key,
    normalize_novelty_bucket,
    update_search_memory,
)
from shared.identity_resolution import (
    classify_recruiter_activity_pressure,
    infer_reachout_status,
)
from shared.reconciliation_schemas import RecruiterActivitySnapshot
from shared.strict_seniority import (
    classify_search_string_seniority,
    is_strict_seniority_brief,
    profile_reads_above_band,
    recommended_yoe_window,
)
from shared.sourcing_lanes import (
    apply_lane_fields_to_search_string,
    lane_fields_from_work_unit_item,
    normalize_lane_id,
)
from linkedin.allocator_state import AllocatorStateDeps, AllocatorStateService
from linkedin.block_adaptation import BlockAdaptationDeps, BlockAdaptationService
from linkedin.geography_gate import GeographyGateDeps, GeographyGateService
from linkedin.run_report import (
    RUN_REPORT_ANALYSIS_SYSTEM,
    RunReportDeps,
    RunReportService,
    enrich_linkedin_run_snapshot,
    freeze_linkedin_run_snapshot,
    generate_run_report,
    _PageReport,
)
from linkedin.runtime_attempts import RuntimeAttemptDeps, RuntimeAttemptService
from shared import config
from shared.cost_rollup import (
    _cost_per_save_usd,
    _sum_token_cost_log_usd,
    aggregate_cost_for_run,
    write_cost_rollup_sidecar,
)
from shared.governor import (
    GovernorLimitReached,
    OperatorStopRequested,
    SessionExpired,
    SessionGovernor,
)

import re as _re

logger = logging.getLogger(__name__)

# --- Glance assessment: title normalization ---

_SENIORITY_PREFIXES = _re.compile(
    r'^(senior|sr\.?|lead|principal|staff|junior|jr\.?|associate|chief|head of|director of|vp of)\s+',
    _re.IGNORECASE,
)
_TRAILING_LEVELS = _re.compile(r'\s+(i{1,4}|iv|[1-6])$', _re.IGNORECASE)

# Wave 3 slice 14 (P1 discharge): the ML title synonyms ("ml engineer" /
# "ai engineer" → "machine learning engineer") moved out — deterministic
# code carries no vertical vocabulary. Cost: those titles now cluster as
# distinct families in the glance assessment (cosmetic; clustering only).
_TITLE_SYNONYMS: dict[str, str] = {
    "software developer": "software engineer",
    "swe": "software engineer",
    "dev ops": "devops engineer",
    "data science": "data scientist",
    "programme manager": "program manager",
}


class GeographyRegimeError(RuntimeError):
    """Raised when the brief's session geography cannot be applied AND verified.

    P3a (plans/sourcing-rigor-hardening.md, decided 2026-07-03): geography is a
    fail-closed precondition of searching, not a monitored outcome. The prior
    fail-soft (log one line, proceed boolean-only) shipped a live run where
    every save came back off-geography. Same doctrine as PreflightRegimeError:
    a run on the wrong pool is worse than no run.
    """

    def __init__(self, *args, retryable: bool = False):
        super().__init__(*args)
        self.retryable = retryable


class PageRenderFailedError(RuntimeError):
    """Raised when a results page failed to render reviewable card slots."""


class PanelRecoveryError(RuntimeError):
    """Raised when bounded profile-panel recovery cannot restore results state."""


class TransientPaginationError(RuntimeError):
    """Raised when result-count evidence says a missing next page is retryable."""


class ProjectContextMismatchError(RuntimeError):
    """Raised when the live Recruiter page belongs to a different project.

    E4: a Recruiter save click files the candidate into whichever project the
    visible page belongs to, so a page bound from another project's tab lands
    physical saves in the wrong pipeline. Same fail-closed doctrine as
    GeographyRegimeError: a save into the wrong pipeline is worse than no save.
    """


_PANEL_RECOVERY_MAX_ATTEMPTS = 2
# Full-evaluation candidate-ID mismatches a single Pipeline may RECOVER by
# re-asking (see LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED). A mismatch is a
# security signal first — profile text is attacker-controlled — so recovery is
# a bounded courtesy: past the ceiling every mismatch is still recorded and
# then surfaces as the PARSE_FAILURE it is, which keeps an injection campaign
# from buying unlimited free probes off a candidate it controls.
_FULL_ID_MISMATCH_RECOVERY_CEILING = 3
# Clamp on the echoed ID as it lands in the runtime EVENT payload. It is model
# output, so the receipt carries a bounded slice of it and no console line or
# prompt ever interpolates it. This is not a clamp on the ID everywhere: the
# judge's prompt-capture record keeps an unbounded copy, which rides the same
# durable attempt payload as ``candidate_text`` and is that pre-existing
# exposure class, unchanged by this slice.
_FULL_ID_MISMATCH_ACTUAL_ID_MAX_CHARS = 64


def _normalize_title_family(title: str) -> str:
    """Collapse a job title to a canonical family for clustering."""
    t = title.strip().lower()
    t = _SENIORITY_PREFIXES.sub('', t)
    t = _TRAILING_LEVELS.sub('', t)
    t = t.strip()
    for pattern, canonical in _TITLE_SYNONYMS.items():
        if t == pattern:
            t = canonical
            break
    return t


_PARENS_SUFFIX = _re.compile(r"\s*\([^)]*\)")
_NON_ALNUM = _re.compile(r"[^a-z0-9]+")
_TRANSITIONAL_OUTREACH_TIER_TAG = _re.compile(
    r"\[TIER:\s*(PRIORITY|STANDARD)\s*\]",
    _re.IGNORECASE,
)


def _persisted_outreach_tier(full_payload: dict[str, Any]) -> str:
    """Read structured currency, falling back only for historical rows."""

    if "outreach_tier" in full_payload:
        tier = str(full_payload.get("outreach_tier") or "").strip().upper()
        return tier if tier in OUTREACH_TIERS else ""
    matches = list(
        _TRANSITIONAL_OUTREACH_TIER_TAG.finditer(
            str(full_payload.get("rationale") or "")
        )
    )
    return matches[-1].group(1).upper() if matches else ""


_COMPLETED_RUN_STRING_STATUSES = {"done", "skipped"}
_QUEUE_HONESTY_STATUS_ORDER = ("error", "in_progress", "queued")
_LINKEDIN_RESULTS_PAGE_SIZE = 25
_FACIAL_READ_INTEREST = {"FACIAL_YES": 0.9, "FACIAL_BORDERLINE": 0.35}
_DEFAULT_READ_INTEREST = 0.5


def _normalize_candidate_name_key(name: str) -> str:
    """Collapse minor punctuation/parenthetical variants when matching saved profiles."""
    t = _PARENS_SUFFIX.sub("", (name or "").lower())
    t = t.replace(".", " ")
    t = _NON_ALNUM.sub(" ", t)
    return " ".join(t.split())


def _linkedin_persist_cost_rollup_sidecar(output_dir: Path) -> None:
    """Move #10 / O5: persist per-module cost rollup next to run artifacts.

    Fail-soft — this is observability, its failure must not affect the run.
    """
    try:
        rollup = aggregate_cost_for_run({"linkedin": output_dir})
        write_cost_rollup_sidecar(rollup, run_dir=output_dir)
    except Exception as exc:
        logger.debug("cost rollup sidecar write failed: %s", exc)


def _is_browser_disconnect_error(error: BaseException | str) -> bool:
    return _acquisition_is_browser_disconnect_error(
        error,
        include_render_failures=True,
    )


def _is_containable_facial_page_error(error: BaseException) -> bool:
    """True only for facial provider/transport and attribution-contract faults."""

    if not isinstance(error, Exception):
        return False
    if isinstance(
        error,
        (
            ApiBudgetExhaustedError,
            GovernorLimitReached,
            OperatorStopRequested,
            SessionExpired,
            GeographyRegimeError,
            PageRenderFailedError,
            PanelRecoveryError,
            TransientPaginationError,
            ProjectContextMismatchError,
            AllocatorPolicyError,
        ),
    ):
        return False
    if _is_browser_disconnect_error(error):
        return False
    if is_api_budget_exhausted_error(error):
        return False
    if isinstance(error, (FacialBatchContractError, JudgmentToolContractError)):
        return True
    if str(error).startswith("Fireworks tool contract "):
        return True
    classification = classify_runtime_failure(error, source="judgment")
    return classification.domain in {"network", "provider"}


# URL-structure classifiers (not vertical vocabulary): a Recruiter SEARCH
# view carries the filter sidebar; a profile detail page nested under the
# discover path (…/discover/recruiterSearch/profile/…) does not.
_PROFILE_PAGE_RE = _re.compile(r"/profile/")
_SEARCH_PAGE_RE = _re.compile(r"/discover/|/talent/search")


def _is_recruiter_search_page(url: str) -> bool:
    """True only for a Recruiter SEARCH view — the page with the filter sidebar.

    Treating a profile-under-discover page as a search page skips run-start
    navigation and every filter apply silently gets an empty typeahead
    (live abort, 2026-07-05 Fable/GLM run).
    """
    text = str(url or "")
    if _PROFILE_PAGE_RE.search(text):
        return False
    return bool(_SEARCH_PAGE_RE.search(text))


def _is_foreign_project_page(url: str, expected_project_id: object) -> bool:
    """True when the page is not PROVABLY the brief's Recruiter project.

    E4: `_bind_existing_recruiter_page` (linkedin/browser.py) binds the first
    healthy Recruiter tab without matching the brief, and a save click files the
    candidate into whichever project the page belongs to — so a tab from another
    project silently saves into the wrong pipeline.

    The rule is asymmetric, and the asymmetry is the correction to E4's original
    "absence of either id is not a mismatch", which was a bypass rather than a
    carve-out: a page with no project id in its URL (the global /talent/search
    view, a bare /talent/profile/<id> page) answered False, so run-start skipped
    navigation and the pre-save boundary passed a projectless page through to
    the click.

    - Brief pins no project → nothing can be violated; every page is accepted,
      which is the one place the permissive carve-out was ever justified.
    - Brief pins a project → the page must NAME that project. An absent or
      unparseable page project is UNVERIFIED, and unverified is a mismatch:
      navigate at run-start, refuse at the pre-save boundary.
    """
    expected = str(expected_project_id or "").strip()
    if not expected:
        return False
    actual = _recruiter_page_project_id(url)
    if not actual:
        return True
    return actual != expected


def review_decision_demotion_reason(decision: OpusDecision) -> str:
    """P4 structural-evidence guard.

    Returns the demotion reason code when a non-save review decision
    fails the evidence threshold, else an empty string when no demotion
    is required.

    Rules:
    - ``REVIEW_INFERRED`` requires at least two structural signals.
    - ``REVIEW_FLAGGED`` requires a non-empty ``recommended_next_step``.
    - Any other decision returns ``""`` (no demotion).

    Lane/run review budgets land with lane metrics in P5; this helper is
    the per-candidate guard only.
    """

    if decision.decision in NON_SAVE_REVIEW_DECISIONS:
        code = (decision.review_reason_code or "").strip().lower()
        if not code or code not in REVIEW_REASON_CODES:
            return "invalid_review_reason_code"

    if decision.decision == "REVIEW_INFERRED":
        if len(decision.review_structural_evidence) < 2:
            return "insufficient_structural_evidence"
    elif decision.decision == "REVIEW_FLAGGED":
        if not (decision.review_recommended_next_step or "").strip():
            return "missing_recommended_next_step"
    return ""


def _clear_review_evidence(decision: OpusDecision) -> None:
    """Clear P4 review_* fields on an ``OpusDecision`` in place.

    Used by the orchestrator on the demotion path so the canonical row
    never carries half-populated review metadata after the decision
    flips to ``REJECT``.
    """

    decision.review_reason_code = ""
    decision.review_structural_evidence = []
    decision.review_recommended_next_step = ""
    decision.review_ambiguity_reason = ""


class Pipeline:
    """Orchestrates the multi-model sourcing pipeline."""

    def __init__(
        self,
        brief_path: str,
        output_dir: Optional[str] = None,
        test_mode: bool = False,
        input_mode: str = "concurrent",
        governor: Optional[SessionGovernor] = None,
    ):
        # P8.1: governance attaches to the browser at construction. A caller
        # (session_orchestrator's day-cycle) may hand in the shared governor
        # it already created; absent that, Pipeline stands up its own so the
        # browser is never constructed ungoverned. This differs from
        # LinkedInBrowser's own constructor, which raises on a missing
        # governor — Pipeline is the production execution engine invoked by
        # session_orchestrator, so a sensible real-governor default is not a
        # bypass.
        self._governor = governor or SessionGovernor()
        self._validate_judgment_runtime_configuration()
        self.brief_path = str(brief_path)
        self.brief_obj = load_brief(brief_path)
        self.state_dir = resolve_linkedin_state_dir(
            brief_path=self.brief_path,
            brief=self.brief_obj,
            state_dir=output_dir,
        )
        self.output_dir = self.state_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_mode = test_mode
        self.input_mode = input_mode

        # Initialize judger with the Brief dataclass
        init_judger(self.brief_obj)

        # Output file paths
        self.snippets_path = self.output_dir / "snippets.jsonl"
        self.facial_path = self.output_dir / "facial_judgments.jsonl"
        self.profiles_path = self.output_dir / "profile_summaries.jsonl"
        self.final_path = self.output_dir / "final_judgments.jsonl"
        self.progress_path = self.output_dir / "progress.json"
        self.log_path = self.output_dir / "run_log.jsonl"
        self._fallback_candidates_path = self.output_dir / "fallback_candidates.jsonl"
        self._fallback_provider: FallbackSearchProvider | None = None
        self.runtime_db_path = self.output_dir / "runtime_state.sqlite3"

        # Browser
        self._timing_recorder = (
            RunLogTimingRecorder(self.log_path)
            if config.LINKEDIN_TIMING_TELEMETRY_ENABLED
            else None
        )
        self.browser = LinkedInBrowser(
            input_mode=input_mode,
            governor=self._governor,
            timing_recorder=self._timing_recorder,
            state_dir=Path(self.output_dir) if self.output_dir else None,
        )
        # F4: pin the authorized project before anything can bind a tab. Set here
        # rather than at run start because `connect()` is not the only binder —
        # sidebar ops and crash recovery rebind too, and a preference applied
        # after the first bind would leave exactly those paths unguided.
        self.browser.set_required_project_id(self.brief_obj.linkedin_project_id)

        # Brief identifier for cross-session file scoping
        self._brief_id = self.brief_obj.linkedin_project_id or Path(brief_path).stem

        # Cross-session candidate history (brief-scoped, never archived)
        self.history_path = self.output_dir / f"candidate_history-{self._brief_id}.jsonl"

        # Cross-session noise discoveries (brief-scoped, never archived)
        self.noise_path = self.output_dir / f"noise_discoveries-{self._brief_id}.jsonl"
        self.search_memory_path = self.output_dir / f"search_memory-{self._brief_id}.json"
        self._search_memory: dict = {}

        # Dedup cache (loaded once, updated in memory — avoids O(n^2) file reads)
        self._seen_urls: set[str] = set()       # terminal outcomes only
        # P3.2: prior-session suppression set (snapshot at history load) —
        # distinguishes "duplicate because suppressed from history" from
        # within-run overlap for family-exhaustion accounting.
        self._prior_session_urls: set[str] = set()
        self._in_flight_urls: set[str] = set()  # currently being evaluated, NOT persisted

        # Prior session outcomes — URL → most recent outcome (for skip-on-prior-reject)
        self._prior_outcomes: dict[str, str] = {}

        # Save dedup — tracks profile URLs already saved in this session
        self._saved_urls: set[str] = set()

        # Wave 3.2: strings that returned results earlier this run (observation only).
        self._string_ids_with_results: set[int] = set()

        # Stats
        self.stats = {
            "snippets_extracted": 0,
            "facial_yes": 0,
            "facial_borderline": 0,
            "facial_no": 0,
            # Settled full-profile funnel. These semantic outcomes are kept
            # separate from facial opens and physical LinkedIn save clicks.
            "full_reviewed": 0,
            "full_outreach": 0,
            "full_review": 0,
            "full_reject": 0,
            "outreach_tier_counts": {},
            "saved": 0,
            "save_attempts": 0,
            "rejected": 0,
            "high_pressure_candidates_seen": 0,
            "activity_saturated_preview_skips": 0,
            "high_fit_low_novelty_saves": 0,
            # P4: bounded non-save review outcomes. Tracked separately so
            # save counts never absorb candidates routed to spot check.
            "reviewed": 0,
            "reviewed_inferred": 0,
            "reviewed_flagged": 0,
            "reviewed_demoted": 0,
            "facial_contract_corruptions": 0,
            "full_contract_corruptions": 0,
            "consecutive_facial_provider_failures": 0,
        }
        self._full_contract_corruption_call_ids: set[str] = set()
        self._full_id_mismatch_recovered_count: int = 0

        # Progress ref for Ctrl+C handler
        self._progress: Optional[Progress] = None
        self._resume_pending_full_decisions: dict[str, str] = {}
        # Profile-read interest hint, keyed by profile_url. In memory and per-run by
        # design: CandidateSnippet is serialized into canonical run state, so a field
        # there would change the persisted shape and break the to_dict() golden tests.
        # Stamped from the RAW facial verdict, upstream of
        # _normalize_facial_decision_for_persistence, so the hint does not depend on
        # which ambiguity posture the brief selected. Under ternary the verdict
        # survives normalization and either side would read the same; under binary a
        # returned BORDERLINE becomes a PARSE_FAILURE that never reaches a profile
        # read at all. Stamping upstream is the one placement correct under both.
        self._profile_read_interest: dict[str, float] = {}
        self._resume_pending_full_snippets: dict[str, CandidateSnippet] = {}
        self._resume_pending_full_owner_ids: dict[str, int] = {}
        self._runtime_state = RuntimeStateStore(self.runtime_db_path)
        self._runtime_lock = RuntimeStateLock(self.output_dir)
        self._runtime_run_id: int | None = None
        self._completed_pages_this_process: int = 0
        self._runtime_bridge = LinkedInRuntimeStateBridge(
            store=self._runtime_state,
            output_dir=self.output_dir,
            brief_id=self._brief_id,
            brief_name=self.brief_obj.id,
            # Phase 3: brief_path so the bridge can pin brief identity
            # on every start_run. brief_path is the on-disk JSON the
            # orchestrator was loaded from; the canonical hash + snapshot
            # come from compute_brief_identity at run-start.
            brief_path=self.brief_path,
        )
        self._safety = RunSafetyCoordinator(
            store=self._runtime_state,
            output_dir=self.output_dir,
            source="linkedin",
            brief_id=self._brief_id,
        )
        self._recovery_service = LinkedInRecoveryService(
            coordinator=self._safety,
            browser=self.browser,
        )
        self._experiment_states: dict[int, LinkedInExperimentState] = {}
        self._profile_probe = ProfileProbe(CascadePolicy())
        self._latest_page_preview_snippets: list[CandidateSnippet] = []
        self._latest_page_observed: dict[str, int | str] = self._empty_page_observation()
        # Allocator evidence is process-local until the ordinary cursor-N+1
        # checkpoint commits it. Shadow never owns execution; active mode uses
        # a separate, redo-safe transition checkpoint after that verdict.
        self._allocator_page_identity: tuple[int, str, int] | None = None
        self._allocator_page_expected_keys: set[str] = set()
        self._allocator_page_settled_keys: set[str] = set()
        self._allocator_page_off_policy: bool = False
        self._pending_allocator_checkpoint: dict[str, Any] | None = None
        self._pending_allocator_poison: dict[str, Any] | None = None
        self._active_allocator_validated_segments: set[str] = set()
        self._current_variant_candidates: list[dict] = []
        self._search_mutation_budget_used: int = 0
        self._work_unit_service: LinkedInWorkUnitService | None = None
        self._acquisition_service: LinkedInAcquisitionService | None = None
        self._search_mutation_executor: LinkedInSearchMutationExecutor | None = None
        self._side_effects_service: LinkedInSideEffectsService | None = None
        self._ensure_services()

        # Kit strings (populated by run_full)
        self._kit_strings: list[KitString] = []
        self._execution_plan: Optional[ExecutionPlan] = None

        # P5 (Wave 2): strings refused at queue build — error-severity lint
        # findings and ubiquity-gate hits. Each record carries codes and
        # repair hints; rendered in the run report and logged as
        # search_string_lint_blocked. In-memory only: a resumed session
        # rebuilds nothing here (the durable trace is the run log).
        self._lint_blocked_strings: list[dict] = []

        # Bias monitor (V2 briefs only — uses brief's BiasControls + FacialCalibration)
        self._bias_monitor: Optional[BiasMonitor] = None
        if self.brief_obj.has_v2_schema:
            # Brief-scoped bias checkpoint (survives across sessions)
            self.bias_checkpoint_path = self.output_dir / f"bias_monitor-{self._brief_id}.json"
            if self.bias_checkpoint_path.exists():
                self._bias_monitor = BiasMonitor.from_brief(self.brief_obj._new_brief)
                self._bias_monitor.load_checkpoint(str(self.bias_checkpoint_path))
                print(f"  [bias] Loaded bias monitor from prior session ({len(self._bias_monitor._decisions)} decisions)")
            else:
                self._bias_monitor = BiasMonitor.from_brief(self.brief_obj._new_brief)
        else:
            self.bias_checkpoint_path = self.output_dir / f"bias_monitor-{self._brief_id}.json"

        # Cadence pause — anti-detection idle breaks
        self._last_pause_time: float = time.time()

        # URL snapshot — last known good URL for recovery fallback.
        # INVARIANT (F4): only ever written through _record_last_good_url, which
        # refuses any page that is not provably this brief's Recruiter project.
        self._last_good_url: str = ""

        # P3b (Wave 2): constraint-ownership manifest (built at run start once
        # the brief is final).
        self._constraint_manifest: dict = {}

        self._allocator_service = AllocatorStateService(
            AllocatorStateDeps(
                get_experiment_states=lambda: self._experiment_states,
                get_allocator_page_identity=lambda: self._allocator_page_identity,
                get_pending_allocator_checkpoint=lambda: self._pending_allocator_checkpoint,
            )
        )

        self._geography_service = GeographyGateService(
            GeographyGateDeps(
                get_browser=lambda: self.browser,
                get_brief_obj=lambda: self.brief_obj,
                log_path=self.log_path,
                stats=self.stats,
            )
        )

        self._run_report_service = RunReportService(
            RunReportDeps(
                get_brief_obj=lambda: self.brief_obj,
                brief_path=self.brief_path,
                output_dir=self.output_dir,
                final_path=self.final_path,
                log_path=self.log_path,
                profiles_path=self.profiles_path,
                get_runtime_db_path=lambda: self.runtime_db_path,
                stats=self.stats,
                get_search_memory=lambda: self._search_memory,
                get_constraint_manifest=lambda: self._constraint_manifest,
                get_experiment_states=lambda: self._experiment_states,
                get_runtime_bridge=lambda: self._runtime_bridge,
                get_runtime_run_id=lambda: self._runtime_run_id,
                get_session_geography_receipt=lambda: self._geography_service.session_geography_receipt,
                get_bias_monitor=lambda: self._bias_monitor,
                get_lint_blocked_strings=lambda: self._lint_blocked_strings,
                _adaptation_roi_summary=lambda *a, **kw: self._adaptation_roi_summary(*a, **kw),
                _shadow_cache_hit_rate=lambda *a, **kw: self._shadow_cache_hit_rate(*a, **kw),
                _string_has_seniority_contamination=lambda *a, **kw: self._string_has_seniority_contamination(*a, **kw),
            )
        )

        self._runtime_attempt_service = RuntimeAttemptService(
            RuntimeAttemptDeps(
                get_runtime_bridge=lambda: self._runtime_bridge,
                get_runtime_run_id=lambda: self._runtime_run_id,
                get_runtime_state=lambda: self._runtime_state,
                get_in_flight_urls=lambda: self._in_flight_urls,
                get_resume_pending_full_decisions=lambda: self._resume_pending_full_decisions,
                get_resume_pending_full_snippets=lambda: self._resume_pending_full_snippets,
                get_resume_pending_full_owner_ids=lambda: self._resume_pending_full_owner_ids,
                funnel_candidate_key=lambda *a, **kw: self._funnel_candidate_key(*a, **kw),
                note_page_full_review_settled=lambda *a, **kw: self._note_page_full_review_settled(*a, **kw),
                record_outreach_tier_outcome=lambda *a, **kw: self._record_outreach_tier_outcome(*a, **kw),
                variant_id_for_search_string=lambda *a, **kw: self._variant_id_for_search_string(*a, **kw),
            )
        )

        self._block_adaptation_service = BlockAdaptationService(
            BlockAdaptationDeps(
                log_path=self.log_path,
                get_brief_obj=lambda: self.brief_obj,
                get_lint_blocked_strings=lambda: self._lint_blocked_strings,
                ensure_services=lambda: self._ensure_services(),
                get_work_unit_service=lambda: self._work_unit_service,
                set_search_memory=lambda memory: self._set_search_memory(memory),
                normalize_candidate_name_key=_normalize_candidate_name_key,
            )
        )

    # ------------------------------------------------------------------
    # Cross-session history
    # ------------------------------------------------------------------

    def _load_candidate_history(self) -> None:
        """Load cross-session candidate history into dedup set and prior outcomes dict.

        Called after per-session dedup loading in all three init paths.
        Additive — merges history URLs into _seen_urls alongside current-session files.

        Executive Search Slice 10: merges
        ``brief.prior_search.ruled_out_urls`` into ``_seen_urls`` so
        candidates the recruiter has already approached or formally
        ruled out never reach the evaluation pipeline. The exclusion
        applies regardless of whether the brief is an exec-search
        brief — `prior_search` is harmless on classic LinkedIn
        briefs (defaults to empty) and load-bearing on exec briefs.
        """

        if self._runtime_bridge and self._runtime_bridge.has_runtime_state():
            blocked_urls, prior_outcomes, saved_urls = self._runtime_bridge.load_history()
            self._seen_urls = set(blocked_urls)
            # P3.2: snapshot the PRIOR-SESSION suppression set before the run
            # adds its own seen urls, so duplicate counting can distinguish
            # "suppressed from history" from within-run overlap.
            self._prior_session_urls = set(blocked_urls)
            self._prior_outcomes = dict(prior_outcomes)
            self._saved_urls.update(saved_urls)
            saves = sum(
                1
                for outcome in self._prior_outcomes.values()
                if outcome in SAVE_FAMILY_DECISIONS
            )
            rejects = sum(1 for outcome in self._prior_outcomes.values() if outcome == "REJECT")
            print(f"  [dedup] Runtime history: {len(self._prior_outcomes)} candidates ({saves} saves, {rejects} rejects)")
            self._merge_prior_search_exclusion()
            return
        if not self.history_path.exists():
            self._merge_prior_search_exclusion()
            return
        for entry in read_jsonl(self.history_path):
            url = entry.get("profile_url", "")
            if url:
                self._seen_urls.add(url)
                self._prior_session_urls.add(url)
                self._prior_outcomes[url] = entry.get("outcome", "")
        saves = sum(
            1 for o in self._prior_outcomes.values() if o in SAVE_FAMILY_DECISIONS
        )
        rejects = sum(1 for o in self._prior_outcomes.values() if o == "REJECT")
        print(f"  [dedup] Cross-session history: {len(self._prior_outcomes)} candidates ({saves} saves, {rejects} rejects)")
        self._merge_prior_search_exclusion()

    @staticmethod
    def _runtime_json_object(raw: object) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str) or not raw:
            return {}
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _runtime_attempt_string_id(cls, row: dict[str, Any]) -> int | None:
        payload = cls._runtime_json_object(row.get("payload_json"))
        cursor = cls._runtime_json_object(row.get("source_cursor_json"))
        payload_cursor = payload.get("cursor")
        if not isinstance(payload_cursor, dict):
            payload_cursor = {}
        raw = (
            payload.get("source_string_id")
            or payload_cursor.get("source_string_id")
            or cursor.get("source_string_id")
            or row.get("source_unit_id")
        )
        try:
            return int(raw) if raw is not None and str(raw).strip() else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _runtime_attempt_decision(
        cls,
        row: dict[str, Any],
        *,
        stage: str,
    ) -> str:
        payload = cls._runtime_json_object(row.get("payload_json"))
        decision = payload.get(f"{stage}_decision")
        if not isinstance(decision, dict):
            return ""
        return str(decision.get("decision") or "").strip()

    @classmethod
    def _runtime_attempt_snippet(
        cls,
        row: dict[str, Any],
        *,
        string_id: int | None,
    ) -> CandidateSnippet | None:
        """Rehydrate the immutable snippet captured with a stage attempt."""

        payload = cls._runtime_json_object(row.get("payload_json"))
        raw_snippet = payload.get("snippet")
        if isinstance(raw_snippet, dict):
            try:
                return CandidateSnippet.from_dict(raw_snippet)
            except (KeyError, TypeError, ValueError):
                pass

        profile_url = str(row.get("profile_url") or "").strip()
        if not profile_url or string_id is None:
            return None
        cursor = cls._runtime_json_object(row.get("source_cursor_json"))

        def cursor_int(name: str, default: int) -> int:
            try:
                return int(cursor.get(name, default))
            except (TypeError, ValueError):
                return default

        return CandidateSnippet(
            name=str(row.get("display_name") or "Unknown candidate"),
            headline="",
            current_title="",
            current_company="",
            location="",
            education_snippet="",
            profile_url=profile_url,
            source_string_id=string_id,
            source_string_name=str(cursor.get("source_string_name") or ""),
            page=max(1, cursor_int("page", 1)),
            result_rank=max(0, cursor_int("result_rank", 0)),
        )

    def _hydrate_resume_funnel_from_runtime(self, progress: Progress) -> None:
        """Rebuild resume-only funnel truth from this run's canonical ancestry.

        Resuming clones work units into a new run, but candidate attempts remain
        on the interrupted ancestor.  Read only that explicit ancestry chain so
        unrelated completed runs for the same brief cannot leak into the new
        process.  Successful stage attempts own semantic outcome truth; cloned
        SearchString counters remain a compatibility fallback only for a string
        that has no attributable attempts in the chain.
        """

        if not self._runtime_run_id:
            return

        chain_cte = """
            WITH RECURSIVE run_chain(id, resumed_from_run_id) AS (
                SELECT id, resumed_from_run_id
                FROM runs
                WHERE id = ? AND source = 'linkedin' AND brief_id = ?
                UNION
                SELECT r.id, r.resumed_from_run_id
                FROM runs r
                JOIN run_chain child ON r.id = child.resumed_from_run_id
                WHERE r.source = 'linkedin' AND r.brief_id = ?
            )
        """
        with self._runtime_state.connect() as conn:
            attempt_rows = [
                dict(row)
                for row in conn.execute(
                    chain_cte
                    + """
                    SELECT ca.id, ca.run_id, ca.stage, ca.status,
                           ca.payload_json, ca.source_cursor_json,
                           c.identity_key, c.display_name, c.profile_url,
                           wu.source_unit_id
                    FROM candidate_attempts ca
                    JOIN run_chain chain ON chain.id = ca.run_id
                    JOIN candidates c ON c.id = ca.candidate_id
                    LEFT JOIN work_units wu ON wu.id = ca.work_unit_id
                    WHERE c.source = 'linkedin' AND c.brief_id = ?
                      AND ca.status = 'succeeded'
                    ORDER BY ca.id ASC
                    """,
                    (
                        self._runtime_run_id,
                        self._brief_id,
                        self._brief_id,
                        self._brief_id,
                    ),
                ).fetchall()
            ]
            side_effect_rows = [
                dict(row)
                for row in conn.execute(
                    chain_cte
                    + """
                    SELECT se.id, se.run_id, se.status, se.payload_json,
                           c.identity_key, c.display_name, c.profile_url,
                           ca.payload_json AS attempt_payload_json,
                           ca.source_cursor_json, wu.source_unit_id
                    FROM side_effects se
                    JOIN run_chain chain ON chain.id = se.run_id
                    JOIN candidates c ON c.id = se.candidate_id
                    LEFT JOIN candidate_attempts ca ON ca.id = se.attempt_id
                    LEFT JOIN work_units wu ON wu.id = ca.work_unit_id
                    WHERE c.source = 'linkedin' AND c.brief_id = ?
                      AND se.effect_type = 'linkedin_save'
                    ORDER BY se.id ASC
                    """,
                    (
                        self._runtime_run_id,
                        self._brief_id,
                        self._brief_id,
                        self._brief_id,
                    ),
                ).fetchall()
            ]

        candidate_rows: dict[str, tuple[int | None, str]] = {}
        facial_rows: dict[str, tuple[str, int | None]] = {}
        facial_snippets: dict[str, CandidateSnippet] = {}
        full_rows: dict[str, tuple[str, int | None]] = {}
        full_outreach_tiers: dict[str, str] = {}
        attributable_string_ids: set[int] = set()
        for row in attempt_rows:
            key = str(row.get("identity_key") or row.get("profile_url") or "").strip()
            if not key:
                continue
            string_id = self._runtime_attempt_string_id(row)
            if string_id is not None:
                attributable_string_ids.add(string_id)
            stage = str(row.get("stage") or "")
            if stage == "snippet" and key not in candidate_rows:
                candidate_rows[key] = (
                    string_id,
                    str(row.get("display_name") or ""),
                )
            elif key not in candidate_rows:
                # Compatibility for early runtime rows that predate the
                # explicit snippet attempt but still carry a stage attempt.
                candidate_rows[key] = (
                    string_id,
                    str(row.get("display_name") or ""),
                )
            if stage == "facial" and key not in facial_rows:
                decision = self._runtime_attempt_decision(row, stage="facial")
                if decision:
                    facial_rows[key] = (decision, string_id)
                    snippet = self._runtime_attempt_snippet(
                        row,
                        string_id=string_id,
                    )
                    if snippet is not None:
                        facial_snippets[key] = snippet
            elif stage == "full":
                decision = self._runtime_attempt_decision(row, stage="full")
                # Full rows are first-succeeded-wins EXCEPT that a
                # failure-family row yields to a later real verdict. A
                # contained resume skip settles the candidate with a synthetic
                # succeeded JUDGMENT_FAILURE (see
                # _abandon_unrecoverable_pending_full); under plain first-wins
                # that placeholder would shadow the real decision forever if
                # the person is met and evaluated again, hiding the review from
                # the counters and — for a SAVE-family re-meet — from the
                # pending derivation that drives actuation. Two real decisions
                # keep first-wins. Facial stays first-wins unconditionally: no
                # failure decision ever lands in a succeeded facial row, and
                # the SQL twin in shared/runtime_state/read_models.py orders
                # the same way.
                stored = full_rows.get(key)
                displaces = (
                    stored is not None
                    and stored[0] in FAILURE_DECISIONS
                    and decision not in FAILURE_DECISIONS
                )
                if decision and (stored is None or displaces):
                    full_rows[key] = (decision, string_id)
                    payload = self._runtime_json_object(row.get("payload_json"))
                    full_payload = payload.get("full_decision")
                    tier = (
                        _persisted_outreach_tier(full_payload)
                        if isinstance(full_payload, dict)
                        else ""
                    )
                    if tier:
                        full_outreach_tiers[key] = tier
                    elif displaces:
                        full_outreach_tiers.pop(key, None)

        saved_rows: dict[str, tuple[int | None, str]] = {}
        already_present_keys: set[str] = set()
        save_attempted_keys: set[str] = set()
        for row in side_effect_rows:
            key = str(row.get("identity_key") or row.get("profile_url") or "").strip()
            if not key:
                continue
            save_attempted_keys.add(key)
            if str(row.get("status") or "") != "succeeded":
                continue
            payload = self._runtime_json_object(row.get("payload_json"))
            if payload.get("already_present") is True and not payload.get(
                "reconciled_self_save"
            ):
                already_present_keys.add(key)
            attempt_row = dict(row)
            attempt_row["payload_json"] = row.get("attempt_payload_json")
            saved_rows.setdefault(
                key,
                (
                    self._runtime_attempt_string_id(attempt_row),
                    str(row.get("display_name") or ""),
                ),
            )

        self._candidate_funnel_counted = set(candidate_rows)
        self._facial_funnel_counted = set(facial_rows)
        self._full_funnel_counted = set(full_rows)
        self._outreach_tier_counted = set(full_outreach_tiers)
        self._resume_pending_full_decisions = {}
        self._resume_pending_full_snippets = {}
        self._resume_pending_full_owner_ids = {}
        for identity_key, (decision, string_id) in facial_rows.items():
            full_decision = (
                full_rows.get(identity_key, ("", None))[0]
                if identity_key in full_rows
                else ""
            )
            full_is_terminal = (
                identity_key in full_rows
                and (
                    full_decision not in SAVE_FAMILY_DECISIONS
                    or identity_key in saved_rows
                )
            )
            if (
                decision not in {"FACIAL_YES", "FACIAL_BORDERLINE"}
                or full_is_terminal
            ):
                continue
            snippet = facial_snippets.get(identity_key)
            pending_key = (
                self._funnel_candidate_key(snippet)
                if snippet is not None
                else identity_key
            )
            self._resume_pending_full_decisions[pending_key] = decision
            self._resume_pending_full_owner_ids[pending_key] = string_id
            if snippet is not None:
                self._resume_pending_full_snippets[pending_key] = snippet

        per_string: dict[int, dict[str, Any]] = {}

        def counters_for(string_id: int) -> dict[str, Any]:
            counters = per_string.get(string_id)
            if counters is None:
                counters = self._fresh_string_stats()
                counters["saved_names"] = []
                per_string[string_id] = counters
            return counters

        for _key, (string_id, _name) in candidate_rows.items():
            if string_id is not None:
                counters_for(string_id)["candidates"] += 1
        for _key, (decision, string_id) in facial_rows.items():
            if string_id is None:
                continue
            counter = {
                "FACIAL_YES": "facial_yes",
                "FACIAL_BORDERLINE": "facial_borderline",
                "FACIAL_NO": "facial_no",
            }.get(decision)
            if counter:
                counters_for(string_id)[counter] += 1
        for _key, (decision, string_id) in full_rows.items():
            if string_id is None or (
                decision not in SAVE_FAMILY_DECISIONS
                and decision not in NON_SAVE_REVIEW_DECISIONS
                and decision != "REJECT"
            ):
                continue
            counters = counters_for(string_id)
            counters["full_reviewed"] += 1
            if decision in SAVE_FAMILY_DECISIONS:
                counters["full_outreach"] += 1
            elif decision in NON_SAVE_REVIEW_DECISIONS:
                counters["full_review"] += 1
            else:
                counters["full_reject"] += 1
                counters["rejects"] += 1
        for key, (string_id, name) in saved_rows.items():
            if string_id is None or key in already_present_keys:
                continue
            counters = counters_for(string_id)
            counters["saves"] += 1
            if name:
                counters["saved_names"].append(name)

        for search_string in progress.strings:
            if search_string.id not in attributable_string_ids:
                continue
            counters = counters_for(search_string.id)
            search_string.candidates_count = int(counters["candidates"])
            search_string.facial_yes_count = int(counters["facial_yes"])
            search_string.facial_borderline_count = int(
                counters["facial_borderline"]
            )
            search_string.facial_no_count = int(counters["facial_no"])
            search_string.full_reviewed_count = int(counters["full_reviewed"])
            search_string.full_outreach_count = int(counters["full_outreach"])
            search_string.full_review_count = int(counters["full_review"])
            search_string.full_reject_count = int(counters["full_reject"])
            search_string.saves = list(counters["saved_names"])

        facial_counts = {
            decision: sum(1 for value, _sid in facial_rows.values() if value == decision)
            for decision in ("FACIAL_YES", "FACIAL_BORDERLINE", "FACIAL_NO")
        }
        full_decisions = [decision for decision, _sid in full_rows.values()]
        self.stats["snippets_extracted"] = len(candidate_rows)
        self.stats["facial_yes"] = facial_counts["FACIAL_YES"]
        self.stats["facial_borderline"] = facial_counts["FACIAL_BORDERLINE"]
        self.stats["facial_no"] = facial_counts["FACIAL_NO"]
        self.stats["full_outreach"] = sum(
            1 for decision in full_decisions if decision in SAVE_FAMILY_DECISIONS
        )
        self.stats["full_review"] = sum(
            1 for decision in full_decisions if decision in NON_SAVE_REVIEW_DECISIONS
        )
        self.stats["full_reject"] = sum(
            1 for decision in full_decisions if decision == "REJECT"
        )
        self.stats["full_reviewed"] = (
            self.stats["full_outreach"]
            + self.stats["full_review"]
            + self.stats["full_reject"]
        )
        outreach_tier_counts: dict[str, int] = {}
        for identity_key, (decision, _string_id) in full_rows.items():
            if decision not in SAVE_FAMILY_DECISIONS:
                continue
            tier = full_outreach_tiers.get(identity_key, "")
            if tier:
                outreach_tier_counts[tier] = (
                    outreach_tier_counts.get(tier, 0) + 1
                )
        self.stats["outreach_tier_counts"] = outreach_tier_counts
        self.stats["saved"] = len(saved_rows) - len(already_present_keys)
        self.stats["already_present"] = len(already_present_keys)
        self.stats["save_attempts"] = len(save_attempted_keys)
        self.stats["rejected"] = self.stats["full_reject"]
        self.stats["reviewed"] = self.stats["full_review"]
        self.stats["reviewed_inferred"] = sum(
            1 for decision in full_decisions if decision == "REVIEW_INFERRED"
        )
        self.stats["reviewed_flagged"] = sum(
            1 for decision in full_decisions if decision == "REVIEW_FLAGGED"
        )
        progress.candidates_saved = self.stats["saved"]
        progress.candidates_rejected = self.stats["rejected"]

    def _merge_prior_search_exclusion(self) -> None:
        """Executive Search Slice 10: fold ``brief.prior_search.ruled_out_urls``
        into the dedup set.

        Defensive against:
        - ``self.brief`` not yet set (some init paths call
          ``_load_candidate_history`` before ``self.brief`` is bound;
          we no-op rather than crash so the existing init-order
          contract is preserved).
        - Missing ``prior_search`` field (classic briefs without the
          Slice 1 schema extension).
        - Malformed ``ruled_out_urls`` (string instead of list).

        Idempotent: re-running is a no-op since URLs go into a set.
        """

        brief = getattr(self, "brief", None)
        if brief is None:
            return
        prior_search = getattr(brief, "prior_search", None)
        if prior_search is None:
            return
        ruled_out_urls = getattr(prior_search, "ruled_out_urls", None) or []
        if not isinstance(ruled_out_urls, list):
            return
        added = 0
        for url in ruled_out_urls:
            if isinstance(url, str) and url:
                if url not in self._seen_urls:
                    self._seen_urls.add(url)
                    added += 1
        if added > 0:
            print(
                f"  [dedup] Prior-search exclusion: merged {added} URL(s) "
                f"from brief.prior_search.ruled_out_urls into _seen_urls"
            )

    def _mark_terminal(self, url: str):
        """Promote URL from in-flight to permanent dedup."""
        self._in_flight_urls.discard(url)
        if url:
            self._seen_urls.add(url)

    def _load_search_memory(self) -> None:
        """Load brief-scoped search family memory if present."""
        self._ensure_services()
        self._search_memory = self._work_unit_service.load_search_memory()

    def _save_search_memory(self) -> None:
        """Persist brief-scoped search family memory."""
        self._ensure_services()
        self._search_memory = self._work_unit_service.save_search_memory()

    def _update_search_memory_from_block(self, block_strings: list[SearchString]) -> None:
        """Delegates to BlockAdaptationService."""
        self._block_adaptation_service._update_search_memory_from_block(block_strings)

    def _hydrate_search_string_metadata(self, search_string: SearchString) -> None:
        """Delegates to BlockAdaptationService."""
        self._block_adaptation_service._hydrate_search_string_metadata(search_string)

    def _checkpoint_progress(
        self,
        progress: Progress | None,
        search_string: SearchString | None = None,
        page_num: int | None = None,
        *,
        completed_page: bool = False,
    ) -> None:
        started = time.monotonic()
        recorder = getattr(self, "_timing_recorder", None)
        timings = (
            {
                "lane_cost_reparse_ms": 0.0,
                "work_unit_rewrite_ms": 0.0,
                "projection_rebuild_ms": 0.0,
                "search_memory_reload_ms": 0.0,
            }
            if recorder is not None
            else None
        )
        try:
            self._checkpoint_progress_impl(
                progress,
                search_string=search_string,
                page_num=page_num,
                completed_page=completed_page,
                _timings=timings,
            )
        finally:
            if timings is not None:
                emit_timing_event(
                    recorder,
                    "checkpoint_progress_timing",
                    elapsed_ms=round((time.monotonic() - started) * 1000.0, 3),
                    **timings,
                )

    def _checkpoint_progress_impl(
        self,
        progress: Progress | None,
        search_string: SearchString | None = None,
        page_num: int | None = None,
        *,
        completed_page: bool = False,
        _timings: dict[str, float] | None = None,
    ) -> None:
        """Persist current run state without waiting for a full page to finish."""
        self._ensure_services()
        pending: dict[str, Any] | None = None
        restored_state: LinkedInExperimentState | None = None
        state_object: LinkedInExperimentState | None = None
        event: tuple[str, SearchString, dict[str, Any]] | None = None
        try:
            # Allocator readiness, identity parsing, snapshotting, and
            # application share the ordinary canonical checkpoint. Shadow is
            # fail-soft; active mode is fail-closed because a lost verdict
            # would let browser spend outrun canonical scheduling state.
            pending = self._allocator_checkpoint_ready()
            if pending is not None:
                state_object = self._experiment_states.get(
                    int(pending.get("root_string_id", 0) or 0)
                )
                if state_object is not None:
                    restored_state = LinkedInExperimentState.from_dict(
                        state_object.to_dict()
                    )
                event = self._apply_pending_allocator_checkpoint(
                    pending,
                    progress=progress,
                )
        except Exception as exc:
            if self._allocator_active_enabled():
                if restored_state is not None and state_object is not None:
                    for field_name in restored_state.__dataclass_fields__:
                        setattr(
                            state_object,
                            field_name,
                            getattr(restored_state, field_name),
                        )
                raise
            # Shadow computation is fail-soft for sourcing, but the trace is
            # fail-closed for analysis from this point forward.
            logger.warning(
                "Page allocator shadow state failed (%s)",
                type(exc).__name__,
            )
            if restored_state is not None and state_object is not None:
                for field_name in restored_state.__dataclass_fields__:
                    setattr(
                        state_object,
                        field_name,
                        getattr(restored_state, field_name),
                    )
            self._poison_allocator_trace(
                search_string=search_string,
                reason=f"checkpoint_apply:{type(exc).__name__}",
            )
            self._pending_allocator_checkpoint = None
            pending = None
            event = None
        poison_event = getattr(self, "_pending_allocator_poison", None)
        owner_ids = set(
            getattr(self, "_resume_pending_full_owner_ids", {}).values()
        )
        if progress is not None and owner_ids:
            for item in progress.strings:
                if item.status in {"done", "skipped"} and item.id in owner_ids:
                    item.status = "in_progress"
        try:
            checkpoint_kwargs = {
                "search_string": search_string,
                "page_num": page_num,
            }
            if _timings is not None:
                checkpoint_kwargs["timings"] = _timings
            self._work_unit_service.checkpoint_progress(
                progress,
                **checkpoint_kwargs,
            )
        except BaseException:
            if restored_state is not None and state_object is not None:
                for field_name in restored_state.__dataclass_fields__:
                    setattr(
                        state_object,
                        field_name,
                        getattr(restored_state, field_name),
                    )
            pending_poison = getattr(self, "_pending_allocator_poison", None)
            if isinstance(pending_poison, dict):
                self._poison_allocator_trace(
                    search_string=search_string,
                    reason=str(
                        pending_poison.get("poison_reason", "shadow_sync_failure")
                    ),
                )
            raise
        if pending is not None:
            self._pending_allocator_checkpoint = None
        if completed_page and search_string is not None:
            # The cursor/action and any allocator observation crossed the same
            # canonical durability boundary. Clear the page rollback inside
            # that boundary so an exception immediately after this method
            # returns cannot downgrade an already-committed N+1 checkpoint.
            self._discard_incomplete_page_rollback(search_string.id)
        if isinstance(poison_event, dict):
            try:
                poison_root_id = int(
                    poison_event.get("root_string_id", 0) or 0
                )
                poison_string = (
                    search_string
                    if (
                        search_string is not None
                        and search_string.id == poison_root_id
                    )
                    else next(
                        (
                            item
                            for item in (
                                progress.strings if progress is not None else []
                            )
                            if item.id == poison_root_id
                        ),
                        None,
                    )
                )
                if poison_string is not None:
                    self._pending_allocator_poison = None
                    self._emit_allocator_event_after_sync(
                        event_type="page_allocator_shadow_poison",
                        search_string=poison_string,
                        payload=dict(poison_event),
                    )
            except Exception as exc:
                logger.warning(
                    "Page allocator poison event failed (%s)",
                    type(exc).__name__,
                )
        if event is not None:
            event_type, event_string, payload = event
            self._emit_allocator_event_after_sync(
                event_type=event_type,
                search_string=event_string,
                payload=payload,
            )

    def _arm_incomplete_page_rollback(
        self,
        search_string: SearchString,
        experiment_state: LinkedInExperimentState,
    ) -> None:
        """Remember the cursor-N state that recovery checkpoints may persist."""
        rollbacks = getattr(self, "_incomplete_page_rollbacks", None)
        if not isinstance(rollbacks, dict):
            rollbacks = {}
            self._incomplete_page_rollbacks = rollbacks
        rollbacks[search_string.id] = {
            "state_object": experiment_state,
            "experiment_state": experiment_state.to_dict(),
            "search_string": search_string.to_dict(),
        }

    def _discard_incomplete_page_rollback(self, string_id: int) -> None:
        rollbacks = getattr(self, "_incomplete_page_rollbacks", None)
        if isinstance(rollbacks, dict):
            rollbacks.pop(string_id, None)

    def _restore_incomplete_page_rollback(self, search_string: SearchString) -> None:
        """Restore pre-decision state before an outer error checkpoint runs."""
        rollbacks = getattr(self, "_incomplete_page_rollbacks", None)
        if not isinstance(rollbacks, dict):
            return
        snapshot = rollbacks.pop(search_string.id, None)
        if not isinstance(snapshot, dict):
            return

        restored_state = LinkedInExperimentState.from_dict(
            snapshot.get("experiment_state")
        )
        state_object = snapshot.get("state_object")
        if restored_state is None or not isinstance(
            state_object, LinkedInExperimentState
        ):
            raise RuntimeError("invalid incomplete-page rollback snapshot")
        # Preserve object identity for callers that already hold the state while
        # replacing every dataclass field with the durable cursor-N image. Keep
        # any instance-level test probes or diagnostics that are not state fields.
        for field_name in restored_state.__dataclass_fields__:
            setattr(state_object, field_name, getattr(restored_state, field_name))
        self._experiment_states[search_string.id] = state_object

        restored_string = SearchString.from_dict(snapshot.get("search_string", {}))
        # Candidate outcomes remain canonical in runtime state and may have
        # settled during the partial page. Roll back only page-control fields;
        # resume hydration reconstructs the candidate counters independently.
        for field_name in (
            "boolean",
            "status",
            "result_count",
            "pages_reviewed",
            "notes",
            "phase",
            "original_boolean",
            "refinement_stack",
            "acquisition_mode",
            "structured_filters",
            "surface",
            "surface_receipt",
        ):
            setattr(search_string, field_name, getattr(restored_string, field_name))

    @staticmethod
    def _empty_page_observation(slots: int = 0) -> dict[str, int | str]:
        return {
            "slots": int(slots or 0),
            "extracted": 0,
            "judged": 0,
            "errored": 0,
            "skipped_dup": 0,
            "skipped_blacklist": 0,
            "skipped_missing_url": 0,
            "full_expected": 0,
            "full_settled": 0,
            "priority": 0,
            "standard": 0,
            "outreach": 0,
            "break_reason": "",
        }

    def _reset_page_observation(
        self,
        slots: int = 0,
        *,
        search_string: SearchString | None = None,
        page_num: int | None = None,
    ) -> None:
        self._latest_page_observed = self._empty_page_observation(slots)
        self._allocator_page_expected_keys = set()
        self._allocator_page_settled_keys = set()
        self._allocator_page_off_policy = (
            self._allocator_run_diverged()
            if self._allocator_shadow_enabled()
            else False
        )
        state = (
            self._experiment_states.get(search_string.id)
            if search_string is not None
            else None
        )
        self._allocator_page_identity = (
            (
                search_string.id,
                state.active_variant_id if state is not None else "",
                int(page_num),
            )
            if search_string is not None and page_num is not None
            else None
        )

    def _page_observation(self) -> dict[str, int | str]:
        observed = getattr(self, "_latest_page_observed", None)
        if not isinstance(observed, dict):
            observed = self._empty_page_observation()
            self._latest_page_observed = observed
        return dict(observed)

    def _note_page_observation(self, key: str, amount: int = 1) -> None:
        observed = getattr(self, "_latest_page_observed", None)
        if not isinstance(observed, dict):
            observed = self._empty_page_observation()
            self._latest_page_observed = observed
        observed[key] = int(observed.get(key, 0) or 0) + int(amount)

    def _note_string_results_seen(
        self, search_string: SearchString, result_count: int
    ) -> None:
        if result_count > 0:
            self._string_ids_with_results.add(search_string.id)

    def _set_page_break_reason(self, reason: str, *, force: bool = False) -> None:
        if reason not in {
            "",
            "glance_reformulate",
            "early_exit",
            "panel_stuck",
            "operator_stop",
            "session_expired",
        }:
            return
        observed = getattr(self, "_latest_page_observed", None)
        if not isinstance(observed, dict):
            observed = self._empty_page_observation()
            self._latest_page_observed = observed
        if force or not observed.get("break_reason"):
            observed["break_reason"] = reason

    @staticmethod
    def _allocator_terminal_full_decision(decision: OpusDecision) -> bool:
        return AllocatorStateService._allocator_terminal_full_decision(decision)

    def _allocator_page_matches(self, snippet: CandidateSnippet) -> bool:
        return self._allocator_service._allocator_page_matches(snippet)

    def _note_page_full_review_expected(self, snippet: CandidateSnippet) -> None:
        """Count a page-local facial positive when full review is scheduled."""

        if not self._allocator_page_matches(snippet):
            return
        key = self._funnel_candidate_key(snippet)
        if key in self._allocator_page_expected_keys:
            return
        self._allocator_page_expected_keys.add(key)
        self._note_page_observation("full_expected")

    def _track_full_review_obligation(
        self,
        snippet: CandidateSnippet,
        decision: str,
    ) -> None:
        key = self._funnel_candidate_key(snippet)
        self._resume_pending_full_owner_ids[key] = snippet.source_string_id
        self._resume_pending_full_snippets[key] = snippet
        self._resume_pending_full_decisions[key] = decision

    def _note_page_full_review_settled(
        self,
        *,
        snippet: CandidateSnippet,
        decision: OpusDecision,
    ) -> None:
        """Count canonical terminal currency for the exact open page only."""

        if (
            not self._allocator_page_matches(snippet)
            or not self._allocator_terminal_full_decision(decision)
        ):
            return
        key = self._funnel_candidate_key(snippet)
        if key in self._allocator_page_settled_keys:
            return
        self._allocator_page_settled_keys.add(key)
        self._note_page_observation("full_settled")
        if decision.decision not in SAVE_FAMILY_DECISIONS:
            return
        self._note_page_observation("outreach")
        tier = str(getattr(decision, "outreach_tier", "") or "").upper()
        if tier == "PRIORITY":
            self._note_page_observation("priority")
        elif tier == "STANDARD":
            self._note_page_observation("standard")

    def _assert_page_full_reviews_settled(
        self,
        *,
        progress: Progress,
        search_string: SearchString,
        page_num: int,
    ) -> None:
        missing = (
            self._allocator_page_expected_keys
            - self._allocator_page_settled_keys
        )
        if not missing:
            return
        print(
            f"    [full-review] Carrying {len(missing)} unsettled review(s) "
            f"forward for string #{search_string.id}"
        )

    @staticmethod
    def _allocator_shadow_enabled() -> bool:
        return AllocatorStateService._allocator_shadow_enabled()

    @staticmethod
    def _allocator_active_enabled() -> bool:
        return AllocatorStateService._allocator_active_enabled()

    @staticmethod
    def _allocator_tracking_enabled() -> bool:
        return AllocatorStateService._allocator_tracking_enabled()

    @staticmethod
    def _allocator_verdict_requires_actuation(
        verdict: AllocationVerdict,
    ) -> bool:
        return AllocatorStateService._allocator_verdict_requires_actuation(verdict)

    def _allocator_run_diverged(self) -> bool:
        return self._allocator_service._allocator_run_diverged()

    def _poison_allocator_trace(
        self,
        *,
        search_string: SearchString | None,
        reason: str,
    ) -> None:
        if not self._allocator_shadow_enabled():
            return
        state = (
            self._experiment_states.get(search_string.id)
            if search_string is not None
            else None
        )
        if state is None and self._experiment_states:
            state = next(iter(self._experiment_states.values()))
        root_string_id = (
            state.root_string_id
            if state is not None
            else (search_string.id if search_string is not None else 0)
        )
        if (
            root_string_id > 0
            and not isinstance(
                getattr(self, "_pending_allocator_poison", None),
                dict,
            )
        ):
            page = (
                max(1, state.active_allocator_page_cursor())
                if state is not None
                else 1
            )
            self._pending_allocator_poison = {
                "mode": "shadow",
                "root_string_id": root_string_id,
                "page": page,
                "analysis_evaluable": False,
                "poison_reason": str(reason),
            }
        if state is None:
            return
        if state.allocator_shadow_diverged:
            trace_poison_reason = str(
                state.allocator_causality.get("trace_poison_reason", "")
                or reason
            )
            state.allocator_causality = {
                **state.allocator_causality,
                "analysis_evaluable": False,
                "trace_poison_reason": trace_poison_reason,
            }
            return
        state.allocator_shadow_diverged = True
        state.allocator_causality = {
            "aligned": False,
            "reason": str(reason),
            "analysis_evaluable": False,
            "trace_poison_reason": str(reason),
        }

    @staticmethod
    def _allocator_terminal_status(status: str) -> bool:
        return AllocatorStateService._allocator_terminal_status(status)

    def _allocator_contiguous_segment(
        self,
        progress: Progress,
        current: SearchString,
    ) -> list[tuple[int, SearchString]]:
        return self._allocator_service._allocator_contiguous_segment(progress, current)

    def _allocator_segment_identity(
        self,
        progress: Progress,
        current: SearchString,
    ) -> dict[str, Any]:
        return self._allocator_service._allocator_segment_identity(progress, current)

    def _allocator_state_for_arm(
        self,
        search_string: SearchString,
    ) -> LinkedInExperimentState:
        return self._allocator_service._allocator_state_for_arm(search_string)

    def _allocator_arms(
        self,
        *,
        progress: Progress,
        current: SearchString,
        prospective_observation: PageObservation | None = None,
        exhausted_root_id: int | None = None,
    ) -> list[AllocatorArm]:
        return self._allocator_service._allocator_arms(
            progress=progress,
            current=current,
            prospective_observation=prospective_observation,
            exhausted_root_id=exhausted_root_id,
        )

    @staticmethod
    def _allocator_post_verdict_order(
        root_ids: list[int],
        verdict: AllocationVerdict,
    ) -> list[int]:
        return AllocatorStateService._allocator_post_verdict_order(root_ids, verdict)

    def _allocator_expected_statuses(
        self,
        *,
        progress: Progress,
        current: SearchString,
        arms: list[AllocatorArm],
        verdict: AllocationVerdict,
    ) -> dict[str, str]:
        return self._allocator_service._allocator_expected_statuses(
            progress=progress,
            current=current,
            arms=arms,
            verdict=verdict,
        )

    def _allocator_expectation(
        self,
        *,
        progress: Progress,
        current: SearchString,
        arms: list[AllocatorArm],
        verdict: AllocationVerdict,
        sequence: int,
    ) -> dict[str, Any]:
        return self._allocator_service._allocator_expectation(
            progress=progress,
            current=current,
            arms=arms,
            verdict=verdict,
            sequence=sequence,
        )

    def _allocator_frontier_alignment(
        self,
        *,
        progress: Progress,
        current: SearchString,
        expectation: dict[str, Any],
        require_selected: bool,
    ) -> tuple[bool, str]:
        return self._allocator_service._allocator_frontier_alignment(
            progress=progress,
            current=current,
            expectation=expectation,
            require_selected=require_selected,
        )

    def _latest_allocator_frontier_expectation(self) -> dict[str, Any]:
        expectations = [
            getattr(state, "allocator_frontier_expectation", {})
            for state in self._experiment_states.values()
        ]
        expectations = [item for item in expectations if isinstance(item, dict) and item]
        if not expectations:
            return {}
        return max(
            expectations,
            key=lambda item: int(item.get("sequence", 0) or 0),
        )

    def _next_allocator_sequence(self) -> int:
        sequences = [
            int(
                (
                    getattr(state, "allocator_last_verdict", {}) or {}
                ).get("checkpoint_index", 0)
                or 0
            )
            for state in self._experiment_states.values()
            if isinstance(getattr(state, "allocator_last_verdict", {}), dict)
        ]
        return max(sequences, default=0) + 1

    def _check_allocator_pre_spend(
        self,
        *,
        progress: Progress,
        current: SearchString,
    ) -> None:
        """Verify active authority or latch shadow divergence before spend."""

        if not self._allocator_tracking_enabled():
            return
        if self._allocator_active_enabled():
            if progress.current_string_id != current.id:
                raise AllocatorPolicyError(
                    "active allocator current root disagrees with progress"
                )
            if current.status != "in_progress":
                raise AllocatorPolicyError(
                    "active allocator selected root is not in progress"
                )
            arms = self._allocator_arms(progress=progress, current=current)
            verdict = allocate_page(current_root_id=current.id, arms=arms)
            if (
                verdict.action is not AllocationAction.CONTINUE
                or verdict.selected_root_id != current.id
                or verdict.floored_root_ids
            ):
                raise AllocatorPolicyError(
                    "active allocator dispatch was not actuated before spend"
                )
            return
        state = self._experiment_states.get(current.id)
        if state is None:
            return
        if self._allocator_run_diverged():
            if not state.allocator_shadow_diverged:
                state.allocator_shadow_diverged = True
                state.allocator_causality = {
                    "aligned": False,
                    "reason": "prior_run_divergence",
                    "analysis_evaluable": False,
                }
            return
        try:
            expectation = self._latest_allocator_frontier_expectation()
            if expectation:
                aligned, reason = self._allocator_frontier_alignment(
                    progress=progress,
                    current=current,
                    expectation=expectation,
                    require_selected=True,
                )
                if (
                    aligned
                    and reason == "completed_segment_transition"
                ):
                    expectation = {}
            if not expectation:
                arms = self._allocator_arms(progress=progress, current=current)
                bootstrap = allocate_page(
                    current_root_id=current.id,
                    arms=arms,
                )
                aligned = bootstrap.selected_root_id == current.id
                reason = "bootstrap_aligned" if aligned else "bootstrap_mismatch"
                if bootstrap.action in {
                    AllocationAction.FINISH,
                    AllocationAction.FLOOR,
                }:
                    aligned = False
                    reason = "bootstrap_terminal_before_spend"
            state.allocator_causality = {
                "aligned": bool(aligned),
                "reason": reason,
                "analysis_evaluable": bool(aligned),
                "actual_root_id": current.id,
                "expected_root_id": (
                    expectation.get("selected_root_id")
                    if expectation
                    else bootstrap.selected_root_id
                ),
            }
            if not aligned:
                state.allocator_shadow_diverged = True
        except Exception as exc:
            logger.warning(
                "Page allocator pre-spend check failed (%s)",
                type(exc).__name__,
            )
            self._poison_allocator_trace(
                search_string=current,
                reason=f"pre_spend:{type(exc).__name__}",
            )

    def _stage_allocator_page_checkpoint(
        self,
        *,
        progress: Progress,
        search_string: SearchString,
        experiment_state: LinkedInExperimentState,
        page_num: int,
        page_observed: dict[str, int | str],
    ) -> None:
        if not self._allocator_tracking_enabled():
            return
        try:
            identity = self._allocator_page_identity
            if identity != (
                search_string.id,
                experiment_state.active_variant_id,
                int(page_num),
            ):
                raise AllocatorPolicyError("page observation identity changed")
            break_reason = str(page_observed.get("break_reason", "") or "")
            observation = PageObservation(
                root_string_id=search_string.id,
                variant_id=experiment_state.active_variant_id,
                page=int(page_num),
                slots=int(page_observed.get("slots", 0) or 0),
                extracted=int(page_observed.get("extracted", 0) or 0),
                full_expected=int(page_observed.get("full_expected", 0) or 0),
                full_settled=int(page_observed.get("full_settled", 0) or 0),
                priority=int(page_observed.get("priority", 0) or 0),
                standard=int(page_observed.get("standard", 0) or 0),
                outreach=int(page_observed.get("outreach", 0) or 0),
                break_reason=break_reason,
                technical_interruption=break_reason not in {
                    "",
                    "early_exit",
                    "glance_reformulate",
                },
                off_policy=(
                    bool(
                        self._allocator_page_off_policy
                        or self._allocator_run_diverged()
                    )
                    if self._allocator_shadow_enabled()
                    else False
                ),
            )
            arms = self._allocator_arms(
                progress=progress,
                current=search_string,
                prospective_observation=observation,
            )
            verdict = allocate_page(
                current_root_id=search_string.id,
                arms=arms,
            )
            sequence = self._next_allocator_sequence()
            self._pending_allocator_checkpoint = {
                "kind": "page",
                "root_string_id": search_string.id,
                "variant_id": observation.variant_id,
                "page": int(page_num),
                "sequence": sequence,
                "observation": observation,
                "verdict": verdict,
                "expectation": self._allocator_expectation(
                    progress=progress,
                    current=search_string,
                    arms=arms,
                    verdict=verdict,
                    sequence=sequence,
                ),
            }
        except Exception as exc:
            if self._allocator_active_enabled():
                raise
            logger.warning(
                "Page allocator observation failed (%s)",
                type(exc).__name__,
            )
            self._poison_allocator_trace(
                search_string=search_string,
                reason=f"page_stage:{type(exc).__name__}",
            )

    def _stage_allocator_exhaustion(
        self,
        *,
        progress: Progress,
        search_string: SearchString,
        page_num: int,
    ) -> None:
        if not self._allocator_tracking_enabled():
            return
        try:
            if self._pending_allocator_checkpoint is not None:
                raise AllocatorPolicyError(
                    "completed page must checkpoint before exhaustion revision"
                )
            arms = self._allocator_arms(
                progress=progress,
                current=search_string,
                exhausted_root_id=search_string.id,
            )
            verdict = allocate_page(
                current_root_id=search_string.id,
                arms=arms,
            )
            sequence = self._next_allocator_sequence()
            self._pending_allocator_checkpoint = {
                "kind": "exhaustion",
                "root_string_id": search_string.id,
                "variant_id": self._allocator_state_for_arm(
                    search_string
                ).active_variant_id,
                "page": int(page_num),
                "sequence": sequence,
                "observation": None,
                "verdict": verdict,
                "expectation": self._allocator_expectation(
                    progress=progress,
                    current=search_string,
                    arms=arms,
                    verdict=verdict,
                    sequence=sequence,
                ),
            }
        except Exception as exc:
            if self._allocator_active_enabled():
                raise
            logger.warning(
                "Page allocator exhaustion failed (%s)",
                type(exc).__name__,
            )
            self._poison_allocator_trace(
                search_string=search_string,
                reason=f"exhaustion:{type(exc).__name__}",
            )

    def _allocator_checkpoint_ready(self) -> dict[str, Any] | None:
        return self._allocator_service._allocator_checkpoint_ready()

    def _apply_pending_allocator_checkpoint(
        self,
        pending: dict[str, Any],
        *,
        progress: Progress | None,
    ) -> tuple[str, SearchString, dict[str, Any]]:
        state = self._experiment_states[
            int(pending["root_string_id"])
        ]
        observation = pending.get("observation")
        if observation is not None:
            if not isinstance(observation, PageObservation):
                raise AllocatorPolicyError("malformed pending observation")
            state.record_allocator_observation(observation)
        verdict = pending.get("verdict")
        if not isinstance(verdict, AllocationVerdict):
            raise AllocatorPolicyError("malformed pending verdict")
        expectation = pending.get("expectation")
        if not isinstance(expectation, dict):
            raise AllocatorPolicyError("malformed pending expectation")
        sequence = int(pending["sequence"])
        mode = "active" if self._allocator_active_enabled() else "shadow"
        actuation_required = (
            self._allocator_verdict_requires_actuation(verdict)
            if mode == "active"
            else False
        )
        verdict_payload = {
            **verdict.to_dict(),
            "checkpoint_index": sequence,
            "page": int(pending.get("page", 0) or 0),
            "active_variant_id": str(pending.get("variant_id", "")),
            "mode": mode,
            "actuation_required": actuation_required,
            "actuated": not actuation_required,
        }
        state.allocator_last_verdict = verdict_payload
        state.allocator_frontier_expectation = dict(expectation)
        search_string = next(
            (
                item
                for item in (progress.strings if progress is not None else [])
                if item.id == state.root_string_id
            ),
            None,
        )
        if search_string is None:
            raise AllocatorPolicyError("allocator root missing at commit")
        divergence_after_observation = bool(
            mode == "shadow"
            and observation is not None
            and self._allocator_run_diverged()
            and not observation.off_policy
        )
        if mode == "shadow" and progress is not None:
            aligned, reason = self._allocator_frontier_alignment(
                progress=progress,
                current=search_string,
                expectation=expectation,
                require_selected=False,
            )
            if not aligned:
                divergence_after_observation = (
                    divergence_after_observation
                    or bool(observation is not None and not observation.off_policy)
                )
                already_diverged = state.allocator_shadow_diverged
                state.allocator_shadow_diverged = True
                if not already_diverged:
                    state.allocator_causality = {
                        "aligned": False,
                        "reason": reason,
                        "analysis_evaluable": False,
                        "diverged_after_observation": (
                            divergence_after_observation
                        ),
                    }
        payload: dict[str, Any] = {
            "checkpoint_index": sequence,
            "verdict_sequence": sequence,
            "mode": mode,
            "root_string_id": state.root_string_id,
            "page": int(pending.get("page", 0) or 0),
            "observation": (
                observation.to_dict() if observation is not None else None
            ),
            "verdict": verdict.to_dict(),
            "frontier_expectation": dict(expectation),
            "actuation_required": actuation_required,
            "actuated": not actuation_required,
            # For page events this flag describes whether the observation was
            # already off-policy. A disposition mismatch discovered only after
            # the page must not retroactively erase that page's currency.
            "shadow_diverged": (
                bool(observation.off_policy)
                if observation is not None
                else self._allocator_run_diverged()
            ),
            "divergence_after_observation": divergence_after_observation,
            "divergence_reason": str(
                (state.allocator_causality or {}).get("reason", "")
            ),
        }
        event_prefix = f"page_allocator_{mode}"
        event_type = (
            f"{event_prefix}_checkpoint"
            if pending.get("kind") in {"dispatch", "page"}
            else f"{event_prefix}_exhaustion"
        )
        return event_type, search_string, payload

    def _emit_allocator_event_after_sync(
        self,
        *,
        event_type: str,
        search_string: SearchString,
        payload: dict[str, Any],
    ) -> None:
        try:
            self._record_runtime_event(
                search_string=search_string,
                event_type=event_type,
                payload=payload,
            )
        except Exception:
            pass
        try:
            if event_type == "page_allocator_shadow_checkpoint":
                log_event(
                    self.log_path,
                    "page_allocator_shadow_checkpoint",
                    **payload,
                )
            elif event_type == "page_allocator_shadow_exhaustion":
                log_event(
                    self.log_path,
                    "page_allocator_shadow_exhaustion",
                    **payload,
                )
            elif event_type == "page_allocator_shadow_poison":
                log_event(
                    self.log_path,
                    "page_allocator_shadow_poison",
                    **payload,
                )
            elif event_type == "page_allocator_active_checkpoint":
                log_event(
                    self.log_path,
                    "page_allocator_active_checkpoint",
                    **payload,
                )
            elif event_type == "page_allocator_active_exhaustion":
                log_event(
                    self.log_path,
                    "page_allocator_active_exhaustion",
                    **payload,
                )
            elif event_type == "page_allocator_active_actuation":
                log_event(
                    self.log_path,
                    "page_allocator_active_actuation",
                    **payload,
                )
        except Exception:
            pass

    @staticmethod
    def _allocator_verdict_from_payload(
        payload: dict[str, Any],
    ) -> AllocationVerdict:
        return AllocatorStateService._allocator_verdict_from_payload(payload)

    def _active_unactuated_allocator_transition(
        self,
    ) -> tuple[
        LinkedInExperimentState,
        dict[str, Any],
        dict[str, Any],
        AllocationVerdict,
    ] | None:
        pending: list[
            tuple[
                LinkedInExperimentState,
                dict[str, Any],
                dict[str, Any],
                AllocationVerdict,
            ]
        ] = []
        for state in self._experiment_states.values():
            payload = getattr(state, "allocator_last_verdict", {})
            if not isinstance(payload, dict) or not payload:
                continue
            if payload.get("mode") != "active":
                continue
            if payload.get("actuation_required") is not True:
                continue
            if payload.get("actuated") is not False:
                continue
            expectation = getattr(state, "allocator_frontier_expectation", {})
            if not isinstance(expectation, dict) or not expectation:
                raise AllocatorPolicyError(
                    "unactuated allocator verdict has no frontier expectation"
                )
            verdict = self._allocator_verdict_from_payload(payload)
            if not self._allocator_verdict_requires_actuation(verdict):
                raise AllocatorPolicyError(
                    "durable actuation flag disagrees with allocator verdict"
                )
            expected_segment = expectation.get("segment")
            if not isinstance(expected_segment, dict):
                raise AllocatorPolicyError(
                    "unactuated allocator verdict has no segment"
                )
            expected_root_ids = {
                int(root_id)
                for root_id in expected_segment.get("root_ids", [])
            }
            controlled_root_ids = {
                verdict.current_root_id,
                *verdict.paused_root_ids,
                *verdict.floored_root_ids,
            }
            if verdict.selected_root_id is not None:
                controlled_root_ids.add(verdict.selected_root_id)
            if int(state.root_string_id) != verdict.current_root_id or not (
                controlled_root_ids.issubset(expected_root_ids)
            ):
                raise AllocatorPolicyError(
                    "allocator verdict roots disagree with their owner segment"
                )
            if int(payload.get("checkpoint_index", 0) or 0) != int(
                expectation.get("sequence", 0) or 0
            ):
                raise AllocatorPolicyError(
                    "allocator verdict and expectation sequence disagree"
                )
            pending.append((state, payload, expectation, verdict))
        if len(pending) > 1:
            raise AllocatorPolicyError(
                "multiple unactuated allocator verdicts require operator repair"
            )
        return pending[0] if pending else None

    def _validate_active_allocator_canary(
        self,
        *,
        progress: Progress,
        current: SearchString,
    ) -> None:
        segment = self._allocator_segment_identity(progress, current)
        stable_key = (
            f"{segment['block']}\x1f{segment['start_index']}:"
            f"{segment['stop_index']}\x1f"
            + ",".join(
                str(root_id)
                for root_id in sorted(int(item) for item in segment["root_ids"])
            )
        )
        if stable_key in self._active_allocator_validated_segments:
            return
        live_frontier = sum(
            not arm.terminal
            for arm in self._allocator_arms(
                progress=progress,
                current=current,
            )
        )
        page_cap = int(config.LINKEDIN_TOTAL_PAGE_CAP)
        if live_frontier <= 0:
            raise AllocatorPolicyError("active allocator frontier has no live roots")
        if page_cap <= 0 or page_cap > 2 * live_frontier:
            raise AllocatorPolicyError(
                "active allocator requires 0 < LINKEDIN_TOTAL_PAGE_CAP "
                "<= twice the live frontier"
            )
        max_pages = int(config.MAX_PAGES_PER_STRING or 0)
        if max_pages > 0 and page_cap > max_pages:
            raise AllocatorPolicyError(
                "active allocator canary cap exceeds MAX_PAGES_PER_STRING"
            )
        self._active_allocator_validated_segments.add(stable_key)

    def _stage_active_allocator_dispatch(
        self,
        *,
        progress: Progress,
        current: SearchString,
        arms: list[AllocatorArm],
        verdict: AllocationVerdict,
    ) -> None:
        if not self._allocator_active_enabled():
            return
        if self._pending_allocator_checkpoint is not None:
            raise AllocatorPolicyError(
                "allocator dispatch cannot replace a pending page verdict"
            )
        sequence = self._next_allocator_sequence()
        state = self._experiment_states.get(current.id)
        if state is None:
            state = bootstrap_experiment_state(current)
            self._experiment_states[current.id] = state
        self._pending_allocator_checkpoint = {
            "kind": "dispatch",
            "root_string_id": current.id,
            "variant_id": state.active_variant_id,
            "page": max(1, state.active_allocator_page_cursor()),
            "sequence": sequence,
            "observation": None,
            "verdict": verdict,
            "expectation": self._allocator_expectation(
                progress=progress,
                current=current,
                arms=arms,
                verdict=verdict,
                sequence=sequence,
            ),
        }

    def _honor_active_allocator_stop_boundary(self) -> None:
        """Let external stop authority interpose before queue actuation."""

        if not self._allocator_active_enabled():
            return
        if (
            hasattr(self, "_operator_stop_event")
            and self._operator_stop_event.is_set()
        ):
            raise OperatorStopRequested()
        if hasattr(self, "_session_expired") and self._session_expired.is_set():
            raise SessionExpired("session_duration_cap")
        governor_reason = self._governor.check_limits()
        if governor_reason:
            raise GovernorLimitReached(governor_reason)

    def _honor_irreversible_side_effect_boundary(self) -> None:
        """Recheck all stop authority immediately before a save click."""
        if (
            hasattr(self, "_operator_stop_event")
            and self._operator_stop_event.is_set()
        ):
            raise OperatorStopRequested()
        if hasattr(self, "_session_expired") and self._session_expired.is_set():
            raise SessionExpired("session_duration_cap")
        governor_reason = self._governor.check_limits()
        if governor_reason:
            raise GovernorLimitReached(governor_reason)
        self._assert_brief_project_context()

    def _assert_brief_project_context(self) -> None:
        """Refuse a save the live page cannot prove belongs to the brief's project.

        E4: run-start pins the brief's project, but the bound page can move
        mid-run (operator tab switch, recovery re-bind onto a different Recruiter
        tab). A Recruiter save lands in whatever project the page belongs to, so
        the project is revalidated on the same boundary as profile identity.

        F1: a URL with NO project id is refused too when the brief pins one. It
        is not evidence of the right pipeline, it is the absence of evidence, and
        treating it as a pass let a projectless profile page reach the click.
        When the brief pins no project there is nothing to violate and every page
        passes, exactly as before.

        Called on two boundaries: `_honor_irreversible_side_effect_boundary`
        (immediately before the click) and, via
        `LinkedInSideEffectsDeps.assert_project_context`, either side of the
        already-saved probe — which short-circuits the click entirely and so
        never reaches the first boundary.
        """
        # Fail CLOSED on an unreadable URL. This guard's whole doctrine is
        # "unverified is a mismatch"; returning here on an exception would make
        # "could not read the page" the one unverified state that passes, right
        # before an irreversible save. Every sibling boundary (assert_expected_identity,
        # _guard_identity) already fails closed on the same read. If the brief
        # pins no project there is nothing to verify, so an unreadable URL is
        # harmless — match that carve-out explicitly rather than swallowing all.
        try:
            current_url = self.browser.page.url
        except Exception as exc:
            if str(self.brief_obj.linkedin_project_id or "").strip():
                raise ProjectContextMismatchError(
                    "project unverified before save: could not read the page URL "
                    f"({type(exc).__name__})"
                ) from exc
            return
        if _is_foreign_project_page(current_url, self.brief_obj.linkedin_project_id):
            page_project = _recruiter_page_project_id(current_url)
            expected = str(self.brief_obj.linkedin_project_id or "")
            if page_project:
                raise ProjectContextMismatchError(
                    "project mismatch before save: page belongs to Recruiter "
                    f"project {page_project!r}, brief expects {expected!r}"
                )
            raise ProjectContextMismatchError(
                "project unverified before save: page carries no Recruiter "
                f"project id, brief expects {expected!r}"
            )

    def _finish_completed_page_allocator_boundary(
        self,
        *,
        progress: Progress,
        search_string: SearchString,
        page_num: int,
    ) -> AllocationVerdict | None:
        """Interpose canary/stops, then apply any durable active transition."""

        self._honor_total_page_cap_at_checkpoint(
            search_string=search_string,
            page_num=page_num,
            progress=progress,
        )
        self._honor_active_allocator_stop_boundary()
        return self._resume_active_allocator_actuation(progress)

    def _checkpoint_allocator_exhaustion_transition(
        self,
        *,
        progress: Progress,
        search_string: SearchString,
        page_num: int,
        legacy_terminal_status: str = "done",
        completed_page: bool = False,
    ) -> AllocationVerdict | None:
        """Persist physical exhaustion, then actuate only in active mode."""

        self._stage_allocator_exhaustion(
            progress=progress,
            search_string=search_string,
            page_num=page_num,
        )
        if not self._allocator_active_enabled():
            search_string.status = legacy_terminal_status
            self._checkpoint_progress(
                progress,
                search_string=search_string,
                page_num=page_num if completed_page else None,
                completed_page=completed_page,
            )
            return None
        # Phase one deliberately preserves the live frontier pre-image. The
        # persisted verdict is the redo record for the second transaction.
        self._checkpoint_progress(
            progress,
            completed_page=completed_page,
        )
        if completed_page:
            self._discard_incomplete_page_rollback(search_string.id)
        self._honor_active_allocator_stop_boundary()
        verdict = self._resume_active_allocator_actuation(progress)
        if verdict is None:
            raise AllocatorPolicyError(
                "active physical-exhaustion verdict was not actuated"
            )
        return verdict

    def _actuate_active_allocator_transition(
        self,
        *,
        progress: Progress,
        owner_state: LinkedInExperimentState,
        verdict_payload: dict[str, Any],
        expectation: dict[str, Any],
        verdict: AllocationVerdict,
    ) -> AllocationVerdict:
        """Apply one persisted verdict as an atomic, replayable transition."""

        current = next(
            (
                item
                for item in progress.strings
                if item.id == verdict.current_root_id
            ),
            None,
        )
        if current is None:
            raise AllocatorPolicyError("allocator current root is missing at actuation")
        segment = self._allocator_contiguous_segment(progress, current)
        indexes = [index for index, _item in segment]
        segment_items = [item for _index, item in segment]
        segment_by_id = {item.id: item for item in segment_items}
        if len(segment_by_id) != len(segment_items):
            raise AllocatorPolicyError("allocator segment contains duplicate roots")

        expected_segment = expectation.get("segment")
        expected_statuses = expectation.get("expected_status_by_root")
        pre_root_ids = expectation.get("pre_root_ids")
        pre_statuses = expectation.get("pre_status_by_root")
        if not isinstance(expected_segment, dict) or not isinstance(
            expected_statuses, dict
        ):
            raise AllocatorPolicyError("malformed allocator actuation expectation")
        if not isinstance(pre_root_ids, list) or not isinstance(pre_statuses, dict):
            raise AllocatorPolicyError("allocator expectation has no durable pre-image")
        actual_ids = [item.id for item in segment_items]
        expected_ids = [int(root_id) for root_id in expected_segment.get("root_ids", [])]
        durable_pre_ids = [int(root_id) for root_id in pre_root_ids]
        same_boundary = (
            current.block == expected_segment.get("block")
            and indexes[0] == int(expected_segment.get("start_index", -1))
            and indexes[-1] + 1 == int(expected_segment.get("stop_index", -1))
        )
        if (
            not same_boundary
            or actual_ids != durable_pre_ids
            or set(expected_ids) != set(actual_ids)
            or len(expected_ids) != len(actual_ids)
        ):
            raise AllocatorPolicyError("allocator actuation pre-image changed")
        if progress.current_string_id != expectation.get("pre_current_string_id"):
            raise AllocatorPolicyError("allocator current root changed before actuation")
        if {
            str(root_id): segment_by_id[root_id].status for root_id in actual_ids
        } != {str(root_id): str(pre_statuses.get(str(root_id), "")) for root_id in actual_ids}:
            raise AllocatorPolicyError("allocator root statuses changed before actuation")
        if set(expected_statuses) != {str(root_id) for root_id in actual_ids}:
            raise AllocatorPolicyError("allocator expected statuses changed roots")

        original_order = list(progress.strings)
        original_progress = progress.to_dict()
        original_states = dict(self._experiment_states)
        state_snapshots = {
            root_id: state.to_dict()
            for root_id, state in original_states.items()
        }

        def restore_preimage() -> None:
            progress.strings[:] = original_order
            restored_progress = Progress.from_dict(original_progress)
            for field_name in progress.__dataclass_fields__:
                if field_name != "strings":
                    setattr(
                        progress,
                        field_name,
                        getattr(restored_progress, field_name),
                    )
            restored_by_id = {
                item.id: item for item in restored_progress.strings
            }
            for item in original_order:
                restored_item = restored_by_id[item.id]
                for field_name in item.__dataclass_fields__:
                    setattr(item, field_name, getattr(restored_item, field_name))
            self._experiment_states = original_states
            for root_id, state in original_states.items():
                restored_state = LinkedInExperimentState.from_dict(
                    state_snapshots[root_id]
                )
                if restored_state is None:
                    raise RuntimeError("allocator state rollback failed")
                for field_name in state.__dataclass_fields__:
                    setattr(
                        state,
                        field_name,
                        getattr(restored_state, field_name),
                    )

        try:
            progress.strings[indexes[0] : indexes[-1] + 1] = [
                segment_by_id[root_id] for root_id in expected_ids
            ]
            for root_id in expected_ids:
                item = segment_by_id[root_id]
                expected_status = str(expected_statuses[str(root_id)])
                if expected_status == "terminal":
                    if not self._allocator_terminal_status(item.status):
                        assert not any(
                            owner == item.id
                            for owner in self._resume_pending_full_owner_ids.values()
                        )
                        item.status = "done"
                        note = f" Allocator terminal: {verdict.reason}."
                        if note not in (item.notes or ""):
                            item.notes = (item.notes or "") + note
                elif expected_status in {"in_progress", "queued"}:
                    item.status = expected_status
                else:
                    raise AllocatorPolicyError(
                        "allocator expectation contains an invalid status"
                    )

            if verdict.selected_root_id is None:
                progress.current_string_id = None
                progress.current_page = 0
            else:
                selected = segment_by_id.get(verdict.selected_root_id)
                if selected is None or selected.status != "in_progress":
                    raise AllocatorPolicyError(
                        "allocator selected root is not runnable after actuation"
                    )
                selected_state = self._experiment_states.get(selected.id)
                if selected_state is None:
                    selected_state = bootstrap_experiment_state(selected)
                    self._experiment_states[selected.id] = selected_state
                selected_page = selected_state.active_allocator_page_cursor()
                if selected_page <= 0:
                    selected_page = max(1, int(selected.pages_reviewed or 1))
                    selected_state.set_active_allocator_page_cursor(selected_page)
                progress.current_string_id = selected.id
                progress.current_page = selected_page

            terminal_items = [
                segment_by_id[root_id]
                for root_id in expected_ids
                if self._allocator_terminal_status(segment_by_id[root_id].status)
            ]
            live_items = [
                segment_by_id[root_id]
                for root_id in expected_ids
                if not self._allocator_terminal_status(segment_by_id[root_id].status)
            ]
            if terminal_items:
                self._set_pending_block_adaptation(
                    progress,
                    current.block,
                    terminal_items,
                    ready=not live_items,
                )

            owner_payload = owner_state.allocator_last_verdict
            if (
                not isinstance(owner_payload, dict)
                or int(owner_payload.get("checkpoint_index", 0) or 0)
                != int(verdict_payload.get("checkpoint_index", 0) or 0)
                or owner_payload.get("actuated") is not False
            ):
                raise AllocatorPolicyError(
                    "allocator verdict changed before actuation commit"
                )
            owner_state.allocator_last_verdict = {
                **owner_payload,
                "actuated": True,
            }
            self._ensure_services()
            self._work_unit_service.checkpoint_progress(
                progress,
                search_string=None,
            )
        except BaseException:
            # CLO-152: the preimage restore must not replace the actuation
            # error it is unwinding; the run is aborting either way.
            try:
                restore_preimage()
            except Exception as rollback_error:
                print(
                    "  [warn] allocator state rollback failed "
                    f"({type(rollback_error).__name__}: {rollback_error}); "
                    "propagating the original error."
                )
            raise

        payload = {
            "checkpoint_index": int(
                verdict_payload.get("checkpoint_index", 0) or 0
            ),
            "mode": "active",
            "root_string_id": verdict.current_root_id,
            "page": int(verdict_payload.get("page", 0) or 0),
            "action": verdict.action.value,
            "reason": verdict.reason,
            "selected_root_id": verdict.selected_root_id,
            "paused_root_ids": list(verdict.paused_root_ids),
            "floored_root_ids": list(verdict.floored_root_ids),
            "actuated": True,
            "current_string_id": progress.current_string_id,
            "current_page": int(progress.current_page or 0),
            "segment_root_ids": expected_ids,
            "status_by_root": {
                str(root_id): segment_by_id[root_id].status
                for root_id in expected_ids
            },
        }
        self._emit_allocator_event_after_sync(
            event_type="page_allocator_active_actuation",
            search_string=current,
            payload=payload,
        )
        return verdict

    def _resume_active_allocator_actuation(
        self,
        progress: Progress,
    ) -> AllocationVerdict | None:
        if not self._allocator_active_enabled():
            return None
        pending = self._active_unactuated_allocator_transition()
        if pending is None:
            return None
        owner_state, verdict_payload, expectation, verdict = pending
        return self._actuate_active_allocator_transition(
            progress=progress,
            owner_state=owner_state,
            verdict_payload=verdict_payload,
            expectation=expectation,
            verdict=verdict,
        )

    def _prepare_active_allocator_dispatch(
        self,
        *,
        progress: Progress,
        current: SearchString,
    ) -> AllocationVerdict | None:
        if not self._allocator_active_enabled():
            return None
        self._honor_active_allocator_stop_boundary()
        replayed = self._resume_active_allocator_actuation(progress)
        if replayed is not None and (
            replayed.action is not AllocationAction.CONTINUE
            or replayed.selected_root_id != current.id
            or self._allocator_terminal_status(current.status)
        ):
            return replayed
        self._validate_active_allocator_canary(
            progress=progress,
            current=current,
        )
        arms = self._allocator_arms(progress=progress, current=current)
        verdict = allocate_page(current_root_id=current.id, arms=arms)
        if not self._allocator_verdict_requires_actuation(verdict):
            return None
        self._stage_active_allocator_dispatch(
            progress=progress,
            current=current,
            arms=arms,
            verdict=verdict,
        )
        self._checkpoint_progress(progress)
        self._honor_active_allocator_stop_boundary()
        actuated = self._resume_active_allocator_actuation(progress)
        if actuated is None:
            raise AllocatorPolicyError("active allocator dispatch was not actuated")
        return (
            actuated
            if (
                actuated.action is not AllocationAction.CONTINUE
                or actuated.selected_root_id != current.id
            )
            else None
        )

    async def _recover_stuck_profile_panel(
        self,
        *,
        candidate_name: str,
        page_num: int,
        decision: OpusDecision | None = None,
        progress: Progress | None = None,
        search_string: SearchString | None = None,
    ) -> None:
        """Boundedly restore results state or stop with durable progress.

        A facial-positive candidate must never authorize a page-loop ``break``.
        If closing its profile panel fails, retry the supported results-return
        operation.  Successful recovery clears the transient marker so the
        caller can keep draining its already-acquired review queue.  Exhausted
        recovery checkpoints the page and raises instead of silently dropping
        the current or remaining candidates.
        """

        def record_recovery_event(status: str, **details: Any) -> None:
            try:
                if status == "started":
                    log_event(
                        self.log_path,
                        "panel_recovery_started",
                        name=candidate_name,
                        page=page_num,
                        recovery_status=status,
                        **details,
                    )
                elif status == "succeeded":
                    log_event(
                        self.log_path,
                        "panel_recovered",
                        name=candidate_name,
                        page=page_num,
                        recovery_status=status,
                        **details,
                    )
                else:
                    log_event(
                        self.log_path,
                        "panel_stuck",
                        name=candidate_name,
                        page=page_num,
                        recovery_status=status,
                        **details,
                    )
            except Exception as event_error:
                # Observability cannot turn a restored browser state back into
                # a control-flow failure. Keep the console diagnostic bounded.
                logger.warning(
                    "Panel recovery event write failed (%s)",
                    type(event_error).__name__,
                )

        record_recovery_event("started", recovery_attempted=True)
        last_error: Exception | None = None
        attempts_used = 0
        for attempt in range(1, _PANEL_RECOVERY_MAX_ATTEMPTS + 1):
            attempts_used = attempt
            try:
                await self.browser.go_back_to_results()
                await asyncio.sleep(
                    human_delay_correlated(
                        config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS,
                        channel="panel_close",
                    )
                )
                if decision is not None:
                    decision._panel_stuck = False
                record_recovery_event("succeeded", attempts=attempt)
                print(
                    f"    [recovered] Profile panel restored after "
                    f"{candidate_name}; continuing queued full reviews"
                )
                return
            except Exception as exc:
                last_error = exc
                if _is_browser_disconnect_error(exc):
                    break

        # A fatal browser-state failure is authoritative even if pagination had
        # already recorded an advisory stop reason such as ``early_exit``.
        self._set_page_break_reason("panel_stuck", force=True)
        if progress is not None:
            self._checkpoint_progress(
                progress,
                search_string=search_string,
                page_num=page_num,
            )
        record_recovery_event(
            "failed",
            attempts=attempts_used,
            error_type=type(last_error).__name__ if last_error else "unknown",
        )
        recovery_stop_detail = (
            "progress checkpointed and review stopped"
            if progress is not None
            else "review stopped; run handler will checkpoint"
        )
        print(
            f"    [ERROR] Profile panel recovery failed after {candidate_name}; "
            f"{recovery_stop_detail}"
        )
        if last_error is not None and _is_browser_disconnect_error(last_error):
            raise last_error
        raise PanelRecoveryError(
            f"profile panel recovery failed after {candidate_name} "
            f"({attempts_used} attempts)"
        ) from last_error

    def _note_page_judgment(self, decision: OpusDecision | None) -> None:
        if decision is None:
            return
        if is_failure_decision(getattr(decision, "decision", "")):
            self._note_page_observation("errored")
        else:
            self._note_page_observation("judged")

    async def _probe_card_slot_count_after_hydration(self) -> int:
        slot_count = await self.browser.get_card_slot_count()
        if slot_count == 0:
            fallback_count = await self.browser.scroll_to_load_all_results()
            slot_count = fallback_count or await self.browser.get_card_count()
        return int(slot_count or 0)

    async def _card_slot_count_or_raise(
        self,
        *,
        search_string: SearchString,
        page_num: int,
        result_count: int,
        mode: str,
    ) -> int:
        slot_count = await self._probe_card_slot_count_after_hydration()
        if slot_count:
            return slot_count

        variant_id = self._variant_id_for_search_string(search_string)
        retry_key = (search_string.id, variant_id, page_num, mode)
        attempted = getattr(self, "_zero_slot_reload_attempted_pages", None)
        if not isinstance(attempted, set):
            attempted = set()
            self._zero_slot_reload_attempted_pages = attempted

        if retry_key not in attempted:
            attempted.add(retry_key)
            try:
                await self.browser.page.reload(
                    wait_until="domcontentloaded",
                    timeout=30000,
                )
                await self.browser.page.wait_for_timeout(4000)
            except Exception as exc:
                if _is_browser_disconnect_error(exc):
                    raise
                raise PageRenderFailedError(
                    f"page render failed during zero-slot reload on page {page_num}: {exc!r}"
                ) from exc
            slot_count = await self._probe_card_slot_count_after_hydration()
            if slot_count:
                return slot_count

        self._latest_page_preview_snippets = []
        self._reset_page_observation(0)
        try:
            log_event(
                self.log_path,
                "page_render_zero_slots",
                string_id=search_string.id,
                page=page_num,
                result_count=result_count,
                slots=0,
                mode=mode,
                retry_exhausted=True,
            )
        except Exception as e:
            print(f"  [warn] page_render_zero_slots event failed: {e}")
        raise PageRenderFailedError(
            f"page render failed: zero card slots after reload on page {page_num}"
        )

    def _pages_remaining_by_result_count(self, result_count: int, page_num: int) -> bool:
        return int(result_count or 0) > int(page_num or 0) * _LINKEDIN_RESULTS_PAGE_SIZE

    async def _go_to_next_page_with_transient_retry(
        self,
        *,
        result_count: int,
        page_num: int,
    ) -> tuple[bool, bool]:
        has_next = await self.browser.go_to_next_page()
        if has_next:
            return True, False
        if not self._pages_remaining_by_result_count(result_count, page_num):
            return False, False
        await asyncio.sleep(3)
        has_next = await self.browser.go_to_next_page()
        return has_next, not has_next

    def _ensure_runtime_state(self) -> None:
        output_dir = Path(self.output_dir)
        if not hasattr(self, "runtime_db_path") or self.runtime_db_path is None:
            self.runtime_db_path = output_dir / "runtime_state.sqlite3"
        if not hasattr(self, "_runtime_state") or self._runtime_state is None:
            self._runtime_state = RuntimeStateStore(self.runtime_db_path)
        if not hasattr(self, "_runtime_lock") or self._runtime_lock is None:
            self._runtime_lock = RuntimeStateLock(output_dir)
        if not hasattr(self, "_runtime_bridge") or self._runtime_bridge is None:
            self._runtime_bridge = LinkedInRuntimeStateBridge(
                store=self._runtime_state,
                output_dir=output_dir,
                brief_id=self._brief_id,
                brief_name=self.brief_obj.id,
                brief_path=self.brief_path,
            )
        if not hasattr(self, "_runtime_run_id"):
            self._runtime_run_id = None
        if not hasattr(self, "_safety") or self._safety is None:
            self._safety = RunSafetyCoordinator(
                store=self._runtime_state,
                output_dir=output_dir,
                source="linkedin",
                brief_id=self._brief_id,
            )
        if not hasattr(self, "_recovery_service") or self._recovery_service is None:
            self._recovery_service = LinkedInRecoveryService(
                coordinator=self._safety,
                browser=self.browser,
            )
        self._ensure_services()

    def _record_safety_event(self, event_type: str, payload: dict) -> None:
        if self._runtime_run_id:
            self._runtime_state.record_event(
                run_id=self._runtime_run_id,
                event_type=event_type,
                payload=payload,
            )

    def _ensure_services(self) -> None:
        if not hasattr(self, "_work_unit_service") or self._work_unit_service is None:
            self._work_unit_service = LinkedInWorkUnitService(
                LinkedInWorkUnitDeps(
                    ensure_runtime_state=self._ensure_runtime_state,
                    get_runtime_bridge=lambda: self._runtime_bridge,
                    get_runtime_run_id=lambda: self._runtime_run_id,
                    set_runtime_run_id=self._set_runtime_run_id,
                    get_progress=lambda: self._progress,
                    set_progress=self._set_progress,
                    stats=self.stats,
                    get_experiment_states=lambda: self._experiment_states,
                    set_experiment_states=self._set_experiment_states,
                    get_search_memory=lambda: self._search_memory,
                    set_search_memory=self._set_search_memory,
                    reset_restart_history=self._reset_restart_history,
                )
            )
        if not hasattr(self, "_acquisition_service") or self._acquisition_service is None:
            self._acquisition_service = LinkedInAcquisitionService(
                LinkedInAcquisitionDeps(
                    browser=self.browser,
                    log_path=self.log_path,
                    ensure_browser_healthy=lambda: self._ensure_browser_healthy(),
                )
            )
        if not hasattr(self, "_search_mutation_executor") or self._search_mutation_executor is None:
            self._search_mutation_executor = LinkedInSearchMutationExecutor(
                LinkedInSearchMutationDeps(
                    browser=self.browser,
                    log_path=self.log_path,
                    get_input_mode=lambda: self.input_mode,
                    get_runtime_run_id=lambda: self._runtime_run_id,
                    get_runtime_state=lambda: self._runtime_state,
                    get_search_mutation_budget_used=lambda: self._search_mutation_budget_used,
                    set_search_mutation_budget_used=self._set_search_mutation_budget_used,
                )
            )
        if not hasattr(self, "_side_effects_service") or self._side_effects_service is None:
            self._side_effects_service = LinkedInSideEffectsService(
                LinkedInSideEffectsDeps(
                    browser=self.browser,
                    stats=self.stats,
                    saved_urls=self._saved_urls,
                    log_path=self.log_path,
                    get_test_mode=lambda: self.test_mode,
                    get_runtime_bridge=lambda: self._runtime_bridge,
                    get_runtime_run_id=lambda: self._runtime_run_id,
                    before_irreversible_side_effect=(
                        self._honor_irreversible_side_effect_boundary
                    ),
                    assert_project_context=self._assert_brief_project_context,
                )
            )

    def _set_runtime_run_id(self, runtime_run_id: int | None) -> None:
        self._runtime_run_id = runtime_run_id

    def _set_progress(self, progress: Progress | None) -> None:
        self._progress = progress

    def _set_experiment_states(self, states: dict[int, LinkedInExperimentState]) -> None:
        self._experiment_states = states

    def _set_search_memory(self, search_memory: dict) -> None:
        self._search_memory = search_memory

    def _set_search_mutation_budget_used(self, budget_used: int) -> None:
        self._search_mutation_budget_used = budget_used

    def _reset_restart_history(self) -> None:
        self._seen_urls = set()
        self._in_flight_urls = set()
        self._prior_outcomes = {}
        self._load_candidate_history()

    def _experiment_state_for(self, search_string: SearchString) -> LinkedInExperimentState:
        state = self._experiment_states.get(search_string.id)
        if state is None:
            state = bootstrap_experiment_state(search_string)
            self._experiment_states[search_string.id] = state
        state.apply_shadow(search_string)
        return state

    def _record_runtime_event(
        self,
        *,
        search_string: SearchString | None,
        event_type: str,
        payload: dict,
    ) -> None:
        return self._runtime_attempt_service._record_runtime_event(
            search_string=search_string,
            event_type=event_type,
            payload=payload,
        )

    def _record_judgment_runtime_profile_event(self, *, resumed: bool) -> None:
        """Persist the process-local judgment posture on the canonical run."""

        self._record_runtime_event(
            search_string=None,
            event_type="judgment_runtime_profile",
            payload={
                "resumed_process": bool(resumed),
                "profile": self._judgment_runtime_profile(self.brief_obj),
            },
        )

    def _honor_total_page_cap_at_checkpoint(
        self,
        *,
        search_string: SearchString,
        page_num: int,
        progress: Progress,
    ) -> None:
        """Cooperatively stop a bounded canary after a durable page checkpoint.

        The caller owns the ordering contract: page judgment and metric updates
        have completed, then ``_checkpoint_progress`` has synchronously written
        canonical runtime state, and only then may this method record the stop
        receipt and raise.  No browser navigation or adaptation happens here.
        """

        page_cap = int(config.LINKEDIN_TOTAL_PAGE_CAP)
        if page_cap <= 0:
            return

        self._completed_pages_this_process = (
            int(getattr(self, "_completed_pages_this_process", 0)) + 1
        )
        if self._completed_pages_this_process < page_cap:
            return

        payload = {
            "page_cap": page_cap,
            "completed_pages_this_process": self._completed_pages_this_process,
            "string_id": search_string.id,
            "page": int(page_num),
            "checkpoint_current_page": int(progress.current_page or 0),
            "candidates": int(search_string.candidates_count),
            "duplicates": int(search_string.duplicates_count),
            "facial_yes": int(search_string.facial_yes_count),
            "facial_no": int(search_string.facial_no_count),
        }
        self._record_runtime_event(
            search_string=search_string,
            event_type="total_page_cap_reached",
            payload=payload,
        )
        log_event(self.log_path, "total_page_cap_reached", **payload)
        print(
            f"  [canary] Total page cap {page_cap} reached after durable page "
            f"checkpoint; stopping safely."
        )
        raise OperatorStopRequested("total_page_cap_reached")

    @staticmethod
    def _sync_bounded_page_stats_for_checkpoint(
        search_string: SearchString,
        string_stats: dict[str, int],
    ) -> None:
        """Persist cumulative page counters before a bounded-canary stop.

        Ordinary runs retain their existing end-of-string write timing. A
        total-page cap can raise before that footer, so the canary path must
        copy the already-settled counters onto the SearchString before its
        canonical checkpoint or its aggregate work-unit receipt would lag the
        candidate attempts written during the page.
        """

        search_string.facial_yes_count = int(string_stats["facial_yes"])
        search_string.facial_borderline_count = int(
            string_stats.get("facial_borderline", 0)
        )
        search_string.facial_no_count = int(string_stats["facial_no"])
        search_string.full_reviewed_count = int(
            string_stats.get("full_reviewed", 0)
        )
        search_string.full_outreach_count = int(
            string_stats.get("full_outreach", 0)
        )
        search_string.full_review_count = int(string_stats.get("full_review", 0))
        search_string.full_reject_count = int(string_stats.get("full_reject", 0))
        search_string.candidates_count = int(string_stats["candidates"])
        search_string.duplicates_count = int(string_stats["duplicates"])
        search_string.suppressed_prior_session_count = int(
            string_stats.get("suppressed_prior_session", 0)
        )

    async def _capture_recovery_snapshot(
        self,
        search_string: SearchString,
        *,
        page_num: int | None = None,
    ):
        work_unit_id = None
        if self._runtime_run_id:
            work_unit_id = self._runtime_state.get_work_unit_id(
                self._runtime_run_id,
                kind=LINKEDIN_STRING_KIND,
                source_unit_id=str(search_string.id),
            )
        advanced_controls: dict[str, Any] = {}
        try:
            advanced_controls = await self.browser.snapshot_advanced_search_controls()
        except Exception:
            advanced_controls = {}
        # The snapshot's top-level keyword_boolean is the field that
        # compile_recovery_plan_from_snapshot (advanced_search.py:~304) actually
        # reads to re-add a keyword control on replay. Default it to the
        # compat boolean, but a structured_only active variant must ZERO it so the
        # crash-recovery replay never re-adds the keyword the surface is defined to
        # suppress. This gate is surface-scoped and UNCONDITIONAL on the active
        # variant — it must fire whether or not snapshot_advanced_search_controls()
        # already returned live controls. (apply_location_filter re-populates
        # browser._last_search_snapshot with a controls-bearing, keyword-empty dict;
        # if we only zeroed inside the empty-controls compile branch below, that
        # live snapshot would skip the branch and the keyword would leak back in on
        # replay — the fifth keyword re-entry pathway via replay_search_context.)
        snapshot_keyword_boolean = search_string.boolean or ""
        experiment_state = self._experiment_states.get(search_string.id)
        if experiment_state is not None:
            active = experiment_state.active_variant
            if active.surface == "structured_only":
                snapshot_keyword_boolean = ""
            if not advanced_controls.get("controls") and not active.structured_filters.is_empty():
                from linkedin.advanced_search import (
                    compile_structured_filters_to_plan,
                    snapshot_controls_from_plan,
                )

                plan = compile_structured_filters_to_plan(
                    active.structured_filters,
                    keyword_boolean=search_string.boolean or active.boolean,
                    acquisition_mode=search_string.acquisition_mode or "linkedin_boolean",
                    # Slice D: mirror apply_variant — a structured_only active
                    # variant drops the keyword from the recovery snapshot so the
                    # replay does not re-add an unintended keyword search.
                    include_keyword=(active.surface != "structured_only"),
                )
                advanced_controls = snapshot_controls_from_plan(plan)
                # plan.keyword_boolean is '' for a structured_only surface and the
                # real boolean otherwise — keep the snapshot's replayed keyword in
                # lockstep with the gated plan instead of search_string.boolean.
                snapshot_keyword_boolean = plan.keyword_boolean
        return capture_recovery_snapshot(
            self.browser,
            run_id=self._runtime_run_id,
            work_unit_id=work_unit_id,
            lane_id=getattr(search_string, "lane_id", "") or "",
            keyword_boolean=snapshot_keyword_boolean,
            current_page=page_num or 0,
            search_url=self._last_good_url or self.browser.get_current_search_url() or "",
            advanced_search_controls=advanced_controls,
        )

    def _evaluate_variant_lifecycle(
        self,
        *,
        search_string: SearchString,
        experiment_state: LinkedInExperimentState,
        allow_terminal: bool = True,
    ) -> str | None:
        """Run deterministic lifecycle rules after probe metrics and return decision override."""
        if experiment_state.mode != "experiment":
            return None
        active = experiment_state.active_variant
        if active.variant_id == "root":
            return None
        if active.probe_pages_used < 1:
            return None

        prior_status = active.status
        prior_reason = active.lifecycle_reason
        prior_last_decision = dict(experiment_state.last_variant_decision)
        output = self._search_mutation_executor.evaluate_lane_variant_lifecycle(
            search_string=search_string,
            experiment_state=experiment_state,
            variant=active,
        )

        if output.action == "commit":
            return "commit"
        if output.action == "abandon":
            if not allow_terminal:
                active.status = prior_status
                active.lifecycle_reason = prior_reason
                experiment_state.last_variant_decision = prior_last_decision
                return None
            self._maybe_discover_fallback_candidates(
                search_string,
                trigger_reason=output.reason or "variant_abandoned",
            )
            return "stop"
        if output.action in {"rescue", "split"}:
            rescue_variant = spawn_rescue_variant_from_hint(
                active,
                hint=output.next_variant_hint,
                root_string_id=search_string.id,
            )
            if rescue_variant is None:
                if not allow_terminal:
                    active.status = prior_status
                    active.lifecycle_reason = prior_reason
                    experiment_state.last_variant_decision = prior_last_decision
                    return None
                return "stop"
            experiment_state.variants[rescue_variant.variant_id] = rescue_variant
            if rescue_variant.variant_id not in experiment_state.planned_variant_ids:
                experiment_state.planned_variant_ids.append(rescue_variant.variant_id)
            return "experiment"
        return None

    def _variant_id_for_search_string(self, search_string: SearchString) -> str:
        state = self._experiment_states.get(search_string.id)
        if state is None:
            return ""
        active = state.active_variant
        if active.variant_id and active.variant_id != "root":
            return active.variant_id
        return state.committed_variant_id or state.active_variant_id or ""

    def _lane_context_for_stage(
        self,
        search_string: SearchString,
        *,
        stage: str,
    ) -> dict[str, str]:
        return self._runtime_attempt_service._lane_context_for_stage(
            search_string,
            stage=stage,
        )

    @staticmethod
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

    @staticmethod
    def _judgment_runtime_profile(brief_obj: object | None = None) -> dict[str, object]:
        """Non-secret, hash-stable model/runtime posture for run receipts.

        A live timing result is not attributable unless the state artifact can
        prove which model endpoints, contracts, deadlines, and concurrency
        controls were effective at process start.  Keep this compact and free
        of keys, prompts, candidate data, and ambient environment values.
        """

        Pipeline._validate_judgment_runtime_configuration()
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

    def _brief_signals_for_profile_probe(self) -> dict[str, list[str]]:
        # The shadow probe needs the V2 capability-area NAMES and non-fit LABELS.
        # Those live on self.brief_obj._new_brief (shared/brief_schema.Brief), NOT
        # on the compat Brief — reading them off self.brief_obj directly via
        # getattr silently yields empties and forces every candidate to
        # REVIEW_FLAGGED. Use the typed, fail-loud accessor instead.
        return self.brief_obj.sourcing_signals()

    def _maybe_discover_fallback_candidates(
        self,
        search_string: SearchString,
        *,
        trigger_reason: str,
    ) -> None:
        """Discover public/x-ray candidates for fallback lanes without save side effects."""
        if not fallback_mode_for_search_string(self._execution_plan, search_string):
            return

        candidates, query = discover_fallback_candidates_for_string(
            self._execution_plan,
            search_string,
            provider=self._fallback_provider,
        )

        def _record_event(*, event_type: str, payload: dict[str, Any]) -> None:
            event_payload = {
                **payload,
                # The current work unit is authoritative if a provider or a
                # stale wrapper payload happens to repeat these keys.
                "string_id": search_string.id,
                "lane_id": search_string.lane_id,
            }
            log_event(
                self.log_path,
                event_type,
                **event_payload,
            )

        record_fallback_discovery(
            candidates=candidates,
            query=query,
            search_string=search_string,
            trigger_reason=trigger_reason,
            record_event=_record_event,
            append_candidate=lambda payload: append_jsonl(self._fallback_candidates_path, payload),
        )

    def _clear_worker_sidecar_if_current(self) -> None:
        """Release Cloris's sidecar lock if this process still owns it."""

        sidecar_path = self.output_dir / "worker.json"
        try:
            payload = json.loads(sidecar_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if payload.get("pid") != os.getpid():
            return
        try:
            sidecar_path.unlink()
        except FileNotFoundError:
            pass

    async def _cleanup_post_run_resources(self, *, lock_acquired: bool) -> None:
        """Release run-owned resources before post-snapshot enrichment.

        The nested ``finally`` blocks make lock release and browser teardown
        survive a ``BaseException`` from any earlier cleanup step. Provider-backed
        market-intelligence work runs only after this method returns.
        """

        try:
            self._clear_worker_sidecar_if_current()
        finally:
            try:
                if lock_acquired:
                    self._runtime_lock.release()
            finally:
                await self.browser.disconnect()

    async def _finalize_setup_abort(
        self,
        *,
        primary_error: BaseException,
        progress: Progress | None,
        snapshotter: Any,
        include_bias_checkpoint: bool,
        lock_acquired: bool,
    ) -> None:
        """Close an already-started run and release resources after setup aborts."""

        try:
            self._finish_runtime_record_and_freeze(
                run_status="error",
                stop_reason=RunStopReason.FATAL_RUNTIME_ERROR,
                progress=progress,
                snapshotter=snapshotter,
                include_bias_checkpoint=include_bias_checkpoint,
                primary_error=primary_error,
            )
        except BaseException as finalization_error:
            detail = (
                str(finalization_error).splitlines()[0]
                if str(finalization_error)
                else type(finalization_error).__name__
            )
            print(f"  [warn] setup-abort finalization failed: {detail}")
        try:
            await self._cleanup_post_run_resources(lock_acquired=lock_acquired)
        except BaseException as cleanup_error:
            detail = (
                str(cleanup_error).splitlines()[0]
                if str(cleanup_error)
                else type(cleanup_error).__name__
            )
            print(f"  [warn] setup-abort cleanup failed: {detail}")

    def _finish_runtime_record_and_freeze(
        self,
        *,
        run_status: str,
        stop_reason: RunStopReason,
        progress: Progress | None,
        snapshotter: Any,
        include_bias_checkpoint: bool,
        primary_error: BaseException | None = None,
    ) -> tuple[str, RunStopReason, Path | None, BaseException | None]:
        """Attempt every finalization layer without letting projections block truth."""

        first_error: BaseException | None = None

        if primary_error is not None and run_status == "completed":
            run_status = "error"
            stop_reason = RunStopReason.FATAL_RUNTIME_ERROR

        def attempt(label: str, action: Any) -> Any:
            nonlocal first_error
            try:
                return action()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                print(f"  [warn] {label} failed: {detail}")
                return None

        if hasattr(self, "_llm_usage_cm"):
            attempt(
                "LLM usage finalization",
                lambda: self._llm_usage_cm.__exit__(None, None, None),
            )
        if progress is not None:
            attempt(
                "progress checkpoint",
                lambda: self._checkpoint_progress(progress),
            )
        if include_bias_checkpoint and getattr(self, "_bias_monitor", None):
            attempt(
                "bias checkpoint",
                lambda: self._bias_monitor.save_checkpoint(
                    str(self.bias_checkpoint_path)
                ),
            )

        if first_error is not None and run_status == "completed":
            run_status = "error"
            stop_reason = RunStopReason.FATAL_RUNTIME_ERROR

        finalized_run_dir: Path | None = None
        if self._runtime_run_id:
            completion_was_provisional = run_status == "completed"
            try:
                run_status, stop_reason = self._apply_completed_run_honesty_gate(
                    run_status=run_status,
                    stop_reason=stop_reason,
                    progress=progress,
                )
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                print(f"  [warn] completion honesty gate failed: {detail}")
                run_status = "error"
                stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
            if (
                completion_was_provisional
                and run_status != "completed"
                and first_error is None
            ):
                first_error = RuntimeError(
                    f"LinkedIn completion rejected: {stop_reason}"
                )

            attempt(
                "canonical run finalization",
                lambda: self._safety.finish_run(
                    run_id=self._runtime_run_id,
                    status=run_status,
                    stop_reason=stop_reason,
                ),
            )
            finalized = attempt(
                "immutable run snapshot",
                snapshotter,
            )
            if isinstance(finalized, Path):
                finalized_run_dir = finalized
            else:
                if first_error is None:
                    first_error = RuntimeError(
                        "immutable LinkedIn run snapshot did not return a path"
                    )
                run_status = "error"
                stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
                attempt(
                    "canonical run failure finalization",
                    lambda: self._safety.finish_run(
                        run_id=self._runtime_run_id,
                        status=run_status,
                        stop_reason=stop_reason,
                    ),
                )

        return run_status, stop_reason, finalized_run_dir, first_error

    def _completed_run_honesty_downgrade_detail(
        self,
        progress: Progress | None,
    ) -> str | None:
        """Return a downgrade reason when a nominally completed run did not earn it."""

        if progress and progress.strings:
            status_counts: dict[str, int] = {}
            for search_string in progress.strings:
                status = str(search_string.status or "unknown")
                if status in _COMPLETED_RUN_STRING_STATUSES:
                    continue
                status_counts[status] = status_counts.get(status, 0) + 1
            if status_counts:
                ordered_statuses = [
                    status
                    for status in _QUEUE_HONESTY_STATUS_ORDER
                    if status in status_counts
                ]
                ordered_statuses.extend(
                    sorted(
                        status
                        for status in status_counts
                        if status not in _QUEUE_HONESTY_STATUS_ORDER
                    )
                )
                detail = ", ".join(
                    f"{status_counts[status]} {status}" for status in ordered_statuses
                )
                return f"queue_not_drained: {detail}"
            if all(item.status == "skipped" for item in progress.strings):
                return None

        run_health = self._run_health_summary()
        if run_health.get("green_but_useless"):
            return "green_but_useless"
        return None

    def _apply_completed_run_honesty_gate(
        self,
        *,
        run_status: str,
        stop_reason: str,
        progress: Progress | None,
    ) -> tuple[str, str]:
        if run_status != "completed":
            return run_status, stop_reason
        detail = self._completed_run_honesty_downgrade_detail(progress)
        if not detail:
            return run_status, stop_reason
        return "error", f"{RunStopReason.FATAL_RUNTIME_ERROR}: {detail}"

    @staticmethod
    def _employer_blacklist_match(
        snippet: CandidateSnippet,
        employer_blacklist: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, str, str] | None:
        if not employer_blacklist:
            return None

        # headline+current only; history-screening is a pool-scope call, see plan.
        fields = (
            ("current_company", snippet.current_company or ""),
            ("headline", snippet.headline or ""),
        )
        for blocked in employer_blacklist:
            blocked_text = str(blocked or "").strip()
            if not blocked_text:
                continue
            pattern = r"\b" + _re.escape(blocked_text) + r"\b"
            for field_name, field_value in fields:
                if not field_value:
                    continue
                for match in _re.finditer(pattern, field_value, _re.IGNORECASE):
                    if Pipeline._blacklist_match_is_non_employer_context(field_value, match):
                        continue
                    # CLO-175: the exclusion ruling (CLO-115) keys on the
                    # CURRENT employer. A headline names an employer only in
                    # "@ X" / "at X" patterns; a bare mention is overwhelmingly
                    # a tool, cert, or skill ("Azure OpenAI", "Anthropic
                    # Claude", "Anthropic CCAR-F") — 29 of PAE's 31 exclusions
                    # were headline tool-mentions of exactly-corridor people.
                    if field_name == "headline" and not (
                        Pipeline._headline_match_is_employer_context(field_value, match)
                    ):
                        continue
                    return blocked_text, field_name, field_value
        return None

    @staticmethod
    def _blacklist_match_is_non_employer_context(field_value: str, match: _re.Match[str]) -> bool:
        remainder = field_value[match.end():]
        next_word = _re.match(r"\W+([A-Za-z]+)\b", remainder)
        if not next_word:
            return False
        return next_word.group(1).lower() in {"school", "award"}

    def _full_judgment_fault_disposition(self, error: Exception) -> str:
        """CLO-177: how the tool-contract full-judgment handler treats a fault.

        'contain' — settle THIS candidate as JUDGMENT_FAILURE and keep the
        string alive. 'ceiling' — too many consecutive provider faults;
        treat the provider as down and end the session legibly. 'raise' —
        not a containable provider/transport fault, or containment is off:
        keep the fail-fast behavior. Only containable faults advance the
        counter; a successful judgment resets it, so the ceiling measures
        CONSECUTIVE faults, exactly like CLO-69's facial page containment.
        """
        if not config.LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED:
            return "raise"
        if not _is_containable_facial_page_error(error):
            return "raise"
        count = getattr(self, "_full_eval_consecutive_faults", 0) + 1
        self._full_eval_consecutive_faults = count
        ceiling = max(1, int(config.LINKEDIN_FULL_EVAL_MAX_CONSECUTIVE_FAULTS))
        return "ceiling" if count >= ceiling else "contain"

    @staticmethod
    def _headline_match_is_employer_context(field_value: str, match: _re.Match[str]) -> bool:
        prefix = field_value[: match.start()].rstrip()
        if prefix.endswith("@"):
            anchor_len = 1
        elif _re.search(r"(?i)\bat$", prefix):
            anchor_len = 2
        else:
            return False
        # "previously at OpenAI" / "ex @ Anthropic" names a PAST employer;
        # alumni are in-pool by ruling (the exclusion keys on current only).
        lookback = prefix[: len(prefix) - anchor_len][-32:]
        return not _re.search(
            r"(?i)\b(ex|former|formerly|previously|prev|past|alumni|alumnus|alum)\b[\s\-.:]*$",
            lookback,
        )

    def _record_runtime_snippet(self, search_string: SearchString, snippet: CandidateSnippet) -> None:
        if self._runtime_bridge and self._runtime_run_id:
            self._runtime_bridge.record_snippet_extracted(
                run_id=self._runtime_run_id,
                search_string=search_string,
                snippet=snippet,
            )

    def _start_runtime_stage_attempt(
        self,
        *,
        search_string: SearchString,
        snippet: CandidateSnippet,
        stage: str,
        payload: dict | None = None,
    ) -> int | None:
        return self._runtime_attempt_service._start_runtime_stage_attempt(
            search_string=search_string,
            snippet=snippet,
            stage=stage,
            payload=payload,
        )

    def _finish_runtime_stage_success(
        self,
        *,
        attempt_id: int | None,
        stage: str,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        profile_summary: CandidateProfileSummary | None = None,
        extra_payload: dict | None = None,
    ) -> None:
        return self._runtime_attempt_service._finish_runtime_stage_success(
            attempt_id=attempt_id,
            stage=stage,
            snippet=snippet,
            decision=decision,
            profile_summary=profile_summary,
            extra_payload=extra_payload,
        )

    def _record_outreach_tier_outcome(
        self,
        *,
        snippet: CandidateSnippet,
        decision: OpusDecision,
    ) -> None:
        """Count semantic save currency once, before physical actuation."""

        if (
            decision.stage != "full"
            or decision.decision not in SAVE_FAMILY_DECISIONS
        ):
            return
        tier = str(getattr(decision, "outreach_tier", "") or "").strip().upper()
        if tier not in OUTREACH_TIERS:
            return
        seen = getattr(self, "_outreach_tier_counted", None)
        if not isinstance(seen, set):
            seen = set()
            self._outreach_tier_counted = seen
        key = self._funnel_candidate_key(snippet)
        if key in seen:
            return
        seen.add(key)
        counts = self.stats.setdefault("outreach_tier_counts", {})
        counts[tier] = int(counts.get(tier, 0)) + 1

    def _finish_runtime_stage_failure(
        self,
        *,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        error: Exception,
        stage: str = "full",
        payload: dict | None = None,
    ) -> None:
        return self._runtime_attempt_service._finish_runtime_stage_failure(
            attempt_id=attempt_id,
            snippet=snippet,
            error=error,
            stage=stage,
            payload=payload,
        )

    def _abort_runtime_stage_attempt(
        self,
        *,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        error: BaseException,
        payload: dict | None = None,
    ) -> None:
        return self._runtime_attempt_service._abort_runtime_stage_attempt(
            attempt_id=attempt_id,
            snippet=snippet,
            error=error,
            payload=payload,
        )

    def _propagate_profile_activity_abort(
        self,
        *,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        error: Exception,
        payload: dict,
    ) -> None:
        """Propagate run-level controls from optional profile enrichment."""

        normalized_error: Exception = error
        if isinstance(error, OperatorStopRequested):
            abort_reason = "operator_stop"
        elif isinstance(error, SessionExpired):
            abort_reason = "session_expired"
        elif isinstance(error, GovernorLimitReached):
            abort_reason = "governor_limit"
        elif isinstance(error, GeographyRegimeError):
            abort_reason = "geography_regime"
        elif _is_browser_disconnect_error(error):
            abort_reason = "browser_disconnect"
        elif is_api_budget_exhausted_error(error):
            abort_reason = "api_budget_exhausted"
            if not isinstance(error, ApiBudgetExhaustedError):
                normalized_error = ApiBudgetExhaustedError(str(error))
        else:
            return

        self._abort_runtime_stage_attempt(
            attempt_id=attempt_id,
            snippet=snippet,
            error=normalized_error,
            payload={
                **payload,
                "run_abort": abort_reason,
                "stage": "full",
            },
        )
        if normalized_error is not error:
            raise normalized_error from error
        raise error

    def _finish_runtime_failure_decision(
        self,
        *,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        payload: dict | None = None,
    ) -> None:
        return self._runtime_attempt_service._finish_runtime_failure_decision(
            attempt_id=attempt_id,
            snippet=snippet,
            decision=decision,
            payload=payload,
        )

    def _facial_ternary_enabled(self) -> bool:
        """Whether this run triages on the ternary facial templates.

        The ambiguity posture is brief content
        (``Brief.facial_ambiguity_posture``, preflight-set), with the
        ``LINKEDIN_FACIAL_BORDERLINE_ENABLED`` config flag as the fallback for
        briefs that do not set it — the same resolution the template selector
        uses, so BORDERLINE handling here can never disagree with the prompt
        that was actually assembled.
        """
        from linkedin.judgment_templates import _facial_ternary_selected

        new_brief = getattr(self.brief_obj, "_new_brief", None)
        if new_brief is not None:
            return _facial_ternary_selected(new_brief)
        return bool(config.LINKEDIN_FACIAL_BORDERLINE_ENABLED)

    def _stamp_read_interest(self, snippet, raw_decision: str) -> None:
        """Record the read-budget hint from the RAW facial verdict."""
        interest = _FACIAL_READ_INTEREST.get(raw_decision)
        if interest is not None and snippet.profile_url:
            self._profile_read_interest[snippet.profile_url] = interest

    def _read_interest_for(self, snippet) -> float:
        """Interest hint for a profile read; medium when unstamped (e.g. resume)."""
        return self._profile_read_interest.get(
            snippet.profile_url, _DEFAULT_READ_INTEREST
        )

    def _normalize_facial_decision_for_persistence(self, facial: OpusDecision) -> OpusDecision:
        """Preserve ternary BORDERLINE; fail loud under a binary posture.

        When the ternary posture is active for this brief, a returned
        ``FACIAL_BORDERLINE`` is canonical evidence that the snippet could
        not resolve fit and therefore demands full-profile review. Keep it
        distinct through persistence, counters, and adaptation telemetry.

        Under the binary posture, a returned ``FACIAL_BORDERLINE`` is a
        structural surprise (the binary prompt should not produce
        BORDERLINE). Convert to ``PARSE_FAILURE`` and let the standard
        non-terminal failure path handle it. Fail loud, do not silently
        coerce to YES or NO.

        For all non-borderline decisions this is the identity function so
        callers can wrap unconditionally.
        """
        if facial.decision != "FACIAL_BORDERLINE":
            return facial

        if self._facial_ternary_enabled():
            return facial

        failure = parse_failure_decision(
            stage=facial.stage,
            candidate_name=facial.candidate_name,
            profile_url=facial.profile_url,
            detail=(
                "Facial parser emitted FACIAL_BORDERLINE under the binary "
                "ambiguity posture. This is a structural surprise (the "
                "binary prompt should not produce BORDERLINE). "
                f"Original rationale: {facial.rationale}"
            ),
            reason="facial_borderline_under_flag_off",
            path=facial.path,
        )
        failure.prompt_capture = dict(facial.prompt_capture or {})
        return failure

    @staticmethod
    def _is_recoverable_provider_failure_decision(
        decision: OpusDecision | None,
    ) -> bool:
        """Recognize standardized retryable provider/network exhaustion."""

        return bool(
            decision is not None
            and decision.decision == "JUDGMENT_FAILURE"
            and "[judgment_failure: recoverable_error/"
            in str(decision.rationale or "").lower()
        )

    def _set_pending_block_adaptation(
        self,
        progress: Progress | None,
        block_name: str,
        block_strings: list[SearchString],
        *,
        ready: bool,
    ) -> None:
        """Persist the current block context across resume boundaries."""
        self._ensure_services()
        self._work_unit_service.set_pending_block_adaptation(
            progress,
            block_name,
            block_strings,
            ready=ready,
        )

    def _clear_pending_block_adaptation(self, progress: Progress | None) -> None:
        """Delegates to BlockAdaptationService."""
        self._block_adaptation_service._clear_pending_block_adaptation(progress)

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    async def rejudge_from_file(self, snippets_path: str) -> None:
        """Re-run judgments on existing snippet extractions. No browser needed."""
        from shared.storage import read_jsonl

        print("=" * 60)
        print("RE-JUDGE MODE: Processing existing snippets with brief-driven prompts")
        print(f"Brief: {self.brief_obj.id}")
        print("=" * 60)

        snippets_data = read_jsonl(snippets_path)
        print(f"  Loaded {len(snippets_data)} snippets from {snippets_path}")

        rejudge_path = self.output_dir / "rejudged_facial.jsonl"
        stats = {"total": 0, "facial_yes": 0, "facial_no": 0, "errors": 0}

        for i, data in enumerate(snippets_data, 1):
            snippet = CandidateSnippet.from_dict(data)
            stats["total"] += 1

            print(f"\n  [{i}/{len(snippets_data)}] {snippet.name}")
            print(f"    {snippet.current_title} at {snippet.current_company}")

            try:
                decision = facial_judge(snippet, brief=self.brief_obj)
                append_jsonl(rejudge_path, decision.to_dict())

                if decision.decision == "FACIAL_YES":
                    print(f"    [FACIAL_YES] {decision.rationale}")
                    stats["facial_yes"] += 1
                else:
                    print(f"    [FACIAL_NO] {decision.rationale}")
                    stats["facial_no"] += 1
            except Exception as e:
                print(f"    [ERROR] {e}")
                stats["errors"] += 1

        print(f"\n{'=' * 60}")
        print(f"  Re-judge Summary")
        print(f"  Total: {stats['total']}, "
              f"YES: {stats['facial_yes']}, NO: {stats['facial_no']}, Errors: {stats['errors']}")
        print(f"  Results written to: {rejudge_path}")
        print(f"{'=' * 60}")

    async def run_full(
        self,
        resume: bool = False,
        restart_string_id: int | None = None,
        restart_string_ids: list[int] | None = None,
    ) -> None:
        """Autonomous search evolution: extract kit → strategy → execute → adapt."""
        from shared.kit_extractor import extract_kit_strings
        from linkedin.strategy import form_strategy, adapt_after_block

        print("=" * 60)
        print(f"FULL RUN: Autonomous Search Evolution{' (RESUMING)' if resume else ''}")
        print(f"Brief: {self.brief_obj.id}")
        print(f"Kit URL: {self.brief_obj.kit_url}")
        print("=" * 60)

        self._ensure_runtime_state()
        lock_acquired = False
        run_status = "completed"
        stop_reason = RunStopReason.NORMAL
        try:
            self._runtime_lock.acquire()
            lock_acquired = True
        except RuntimeError as exc:
            stop_reason = RunStopReason.LOCK_CONFLICT
            self._runtime_state.record_event(
                event_type="runtime_lock_conflict",
                payload={"error": str(exc), "traceback": traceback.format_exc(), "output_dir": str(self.output_dir)},
            )
            raise

        progress: Progress | None = None
        had_runtime_state = bool(
            self._runtime_bridge and self._runtime_bridge.has_runtime_state()
        )
        resume_existing_state = bool(
            resume
            and self._runtime_bridge
            and (
                had_runtime_state
                or self._runtime_bridge.has_legacy_state()
            )
        )
        try:
            # Bind the canonical run before touching Recruiter so setup
            # failures can still close it honestly and freeze diagnostics.
            latest_before_start = self._runtime_state.get_latest_run(
                source="linkedin",
                brief_id=self._brief_id,
            )
            try:
                if resume_existing_state:
                    progress = self._load_or_create_progress()
                    self._progress = progress
                else:
                    progress = Progress(brief_name=self.brief_obj.id)
                    self._runtime_run_id, progress = (
                        self._runtime_bridge.start_or_resume_run(
                            resume=False,
                            initial_progress=progress,
                        )
                    )
                    self._progress = progress
            except BaseException:
                latest_after_start = self._runtime_state.get_latest_run(
                    source="linkedin",
                    brief_id=self._brief_id,
                )
                prior_run_id = (
                    int(latest_before_start["id"])
                    if latest_before_start
                    else None
                )
                if (
                    self._runtime_run_id is None
                    and latest_after_start
                    and latest_after_start["status"] == "running"
                    and int(latest_after_start["id"]) != prior_run_id
                ):
                    self._runtime_run_id = int(latest_after_start["id"])
                raise

            if resume_existing_state:
                self._hydrate_resume_funnel_from_runtime(progress)
                self._checkpoint_progress(progress)

            await self.browser.connect()
            log_event(
                self.log_path,
                "pipeline_start",
                mode="full_run_resume" if resume else "full_run",
                judgment_runtime_profile=self._judgment_runtime_profile(self.brief_obj),
            )

            # P4.2: run_full() (the autonomous-search entry point invoked by
            # session_orchestrator) previously never opened a usage session at all, so
            # token-cost-log.jsonl was never written for real runs and pipeline_end
            # cost_usd could never be anything but absent.
            from shared.llm_usage import llm_usage_session

            self._llm_usage_cm = llm_usage_session(
                self.output_dir / "token-cost-log.jsonl",
                module="linkedin",
                brief_id=self.brief_obj.id,
                run_id=self._runtime_run_id,
            )
            self._llm_usage_cm.__enter__()

            # Navigate to the project search page if we're not already there.
            # Must be on a search page (not a profile page) for the Keywords sidebar.
            current_url = self.browser.page.url
            project_url = self._get_project_url()
            # E4: project-awareness at the decision point. A Recruiter search
            # page that is not provably THIS brief's project counts exactly like
            # not being on a search page, so the existing navigation path
            # corrects it instead of the run inheriting a foreign pipeline for
            # every save. F1: that includes a page carrying no project id at all
            # (the global /talent/search view) — it proves nothing, and the old
            # predicate accepted it, so run-start skipped navigation entirely.
            foreign_project_page = _is_foreign_project_page(
                current_url, self.brief_obj.linkedin_project_id
            )
            page_project_id = _recruiter_page_project_id(current_url)
            on_search_page = (
                _is_recruiter_search_page(current_url) and not foreign_project_page
            )
            if project_url and not on_search_page:
                if foreign_project_page and page_project_id:
                    print(
                        "  On another project's Recruiter search page — "
                        "navigating to this brief's project..."
                    )
                elif foreign_project_page:
                    print(
                        "  Recruiter page carries no project context — "
                        "navigating to this brief's project..."
                    )
                else:
                    print(f"  Navigating to project search page...")
                await self.browser.navigate_to_search(project_url)
            elif not on_search_page:
                print(f"  Navigating to LinkedIn Recruiter search...")
                await self.browser.navigate_to_search("https://www.linkedin.com/talent/search")

            # Capture initial good URL for error recovery
            try:
                self._record_last_good_url(self.browser.page.url)
            except Exception:
                pass

            self._seen_urls = set()
            self._in_flight_urls = set()
            self._prior_outcomes = {}
            if not resume and had_runtime_state:
                self._load_candidate_history()
                self._load_search_memory()

            # Ctrl+C handler — only install if not running under session_orchestrator
            def _sigint_handler(sig, frame):
                print("\n\n  [!] Interrupted. Saving progress...")
                if self._progress:
                    self._checkpoint_progress(self._progress)
                raise KeyboardInterrupt

            if not hasattr(self, '_session_expired'):
                signal.signal(signal.SIGINT, _sigint_handler)
        except BaseException as primary_error:
            await self._finalize_setup_abort(
                primary_error=primary_error,
                progress=self._progress or progress,
                snapshotter=self._finalize_run_snapshot,
                include_bias_checkpoint=True,
                lock_acquired=lock_acquired,
            )
            raise

        try:
            # Apply the brief's session-level location filter on the fresh sidebar,
            # before the keyword playbook starts (hop-4 producer). It persists across
            # keyword strings. MUST live inside this try: the P3a gate makes it raise
            # GeographyRegimeError on a facet miss, and a raise before this try would
            # skip the finally at the bottom — leaking the runtime lock (wedging every
            # subsequent day-cycle session with LOCK_CONFLICT), the browser CDP
            # connection, and the llm-usage session (contract-break lens, Wave 1).
            await self._apply_session_location_filter()

            if resume_existing_state:
                # --- Resume: skip kit extraction, strategy, queue building ---
                print("\n--- Resuming from runtime_state ---")

                # Resume must judge with the SAME evaluation criteria the run
                # started with. A preflight-born run's seed brief is hollow
                # (no capability areas / non-fit patterns / calibration), so
                # resuming on it would silently swap the whole evaluation
                # regime to the legacy template path — the exact failure
                # class P9.2 removed from the fresh path, entering through
                # the resume door. Reload the state dir's generated brief;
                # refuse to resume without it (a run on the wrong regime is
                # worse than no run).
                if self.brief_obj.needs_preflight():
                    generated_path = self.output_dir / "preflight_v2_brief.json"
                    if not generated_path.exists():
                        raise RuntimeError(
                            "resume requires the run's generated evaluation "
                            f"criteria ({generated_path}) — not found; "
                            "start a fresh run instead"
                        )
                    from shared.brief_loader import _load_v2_brief
                    self.brief_obj = _load_v2_brief(read_json(str(generated_path)))
                    init_judger(self.brief_obj)
                    if self.brief_obj.has_v2_schema:
                        self._bias_monitor = BiasMonitor.from_brief(self.brief_obj._new_brief)
                    print(
                        "  Reloaded generated V2 brief from state dir "
                        "(same criteria as the original run)"
                    )

                # Restore progress and its checkpointed experiment state as one
                # operation before any later sync can write stale in-memory state.
                if progress is None:
                    progress = self._load_or_create_progress()
                self._record_judgment_runtime_profile_event(resumed=True)
                self._progress = progress

                # Preserve persisted queue order. Adaptive strings may be inserted
                # "next in queue", which is not always the same as numeric ID order.

                restart_ids: list[int] = []
                if restart_string_ids:
                    restart_ids.extend(restart_string_ids)
                elif restart_string_id is not None:
                    restart_ids.append(restart_string_id)

                if restart_ids:
                    self._restart_strings(progress, restart_ids)

                # Rebuild dedup set from cross-session history only
                self._seen_urls = set()
                self._in_flight_urls = set()
                self._prior_outcomes = {}
                self._load_candidate_history()
                self._load_search_memory()
                # Load kit strings for adaptation vocabulary
                kit_path = self.output_dir / "kit_strings.json"
                if kit_path.exists():
                    kit_data = read_json(str(kit_path))
                    self._kit_strings = [KitString.from_dict(ks) for ks in kit_data]
                    print(f"  Loaded {len(self._kit_strings)} kit vocabulary strings")

                # Load execution plan for reference
                plan_path = self.output_dir / "execution_plan.json"
                if plan_path.exists():
                    self._execution_plan = ExecutionPlan.from_dict(read_json(str(plan_path)))

                # Load bias monitor checkpoint
                if self._bias_monitor and self.bias_checkpoint_path.exists():
                    self._bias_monitor.load_checkpoint(str(self.bias_checkpoint_path))
                    print(f"  Resumed bias monitor from checkpoint")

                for search_string in progress.strings:
                    self._hydrate_search_string_metadata(search_string)

                done_count = sum(1 for s in progress.strings if s.status == "done")
                total_count = len(progress.strings)
                print(f"  {done_count}/{total_count} strings already complete")

                # P3b: re-enforce the constraint manifest on the reloaded
                # brief — a between-session brief edit that added a
                # zero-owner constraint must abort here, not silently drop.
                self._enforce_constraint_manifest()
            else:
                # --- Fresh run: archive stale output files ---
                self._archive_stale_outputs()

                # --- Phase 0: Sourcing Preflight (if JD provided without full eval criteria) ---
                if self.brief_obj.needs_preflight():
                    print("\n--- Phase 0: Sourcing Preflight (V2 structured) ---")
                    self._run_preflight_v2()

                # --- Phase 1: Extract kit strings (if kit URL provided) ---
                print("\n--- Phase 1: Kit Extraction ---")
                if not self.brief_obj.kit_url:
                    # An empty vocabulary channel is what produced nine
                    # benchmark names across 72 strings on 2026-07-27. Fill it
                    # by enumeration when enabled; otherwise behave exactly as
                    # before. An operator-supplied kit is never displaced —
                    # that is their explicit vocabulary and outranks a
                    # generated one.
                    self._kit_strings = self._enumerate_vocabulary()
                    if not self._kit_strings:
                        print("  No kit URL provided — strategy will generate strings from JD context only.")
                else:
                    self._kit_strings = extract_kit_strings(self.brief_obj.kit_url)
                    if not self._kit_strings:
                        print("  [warn] No strings extracted from kit — proceeding with JD context only.")
                    else:
                        # P5 (Wave 2): advisory craft check on kit vocabulary.
                        # Kit strings are vocabulary — never queued — so a
                        # defect here degrades only via the model copying it
                        # into a compound; a summary print is the right weight.
                        kit_lint_summary = summarize_kit_lint(self._kit_strings)
                        if kit_lint_summary:
                            print(f"  [kit-lint] {kit_lint_summary}")

                # Save extracted kit for reference
                if self._kit_strings:
                    kit_data = [ks.to_dict() for ks in self._kit_strings]
                    write_json(self.output_dir / "kit_strings.json", kit_data)
                    print(f"  Kit strings saved to {self.output_dir / 'kit_strings.json'}")

                # P3b: the brief is final (preflight ran above if needed) —
                # build/persist the constraint manifest and abort on any
                # stated constraint with zero owners, before spending a
                # strategy call on a run that cannot honor its intake.
                self._enforce_constraint_manifest()

                # --- Phase 2: Strategy formation ---
                print(f"\n--- Phase 2: Strategy Formation ({config.STRATEGY_MODEL_NAME.rsplit('/', 1)[-1]}) ---")
                log_event(self.log_path, "strategy_started")
                prior_data = None
                if self.progress_path.exists():
                    prior_data = read_json(str(self.progress_path))

                # Load noise discoveries from prior sessions
                if self.noise_path.exists():
                    noise_entries = read_jsonl(self.noise_path)
                    if noise_entries:
                        if prior_data is None:
                            prior_data = {}
                        prior_data["noise_discoveries"] = noise_entries

                if self._search_memory:
                    if prior_data is None:
                        prior_data = {}
                    prior_data["search_memory_summary"] = build_search_memory_summary(
                        self._search_memory
                    )

                from market_intelligence.engine import load_lane_feedback_for_strategy

                lane_feedback = load_lane_feedback_for_strategy(self.brief_path)
                self._execution_plan = form_strategy(
                    self.brief_obj,
                    self._kit_strings,
                    prior_data,
                    lane_feedback=lane_feedback or None,
                    # Shadow strategist (item 19): same artifact dir as the
                    # preflight shadow; no-op unless SHADOW_STRATEGY_ENABLED.
                    shadow_dir=self.output_dir / "shadow_strategy",
                )
                # P3.3: close the consumption loop — diffs the strategist
                # consumed are marked in the artifact and never re-served.
                if self._execution_plan.consumed_feedback_ids:
                    try:
                        from market_intelligence.engine import (
                            mark_planner_diffs_consumed,
                        )

                        marked = mark_planner_diffs_consumed(
                            self.brief_path,
                            self._execution_plan.consumed_feedback_ids,
                        )
                        if marked:
                            print(f"  [strategy] Marked {marked} planner diff(s) consumed")
                    except Exception as consume_exc:
                        print(f"  [warn] planner-diff consumption marking failed: {consume_exc}")
                write_json(self.output_dir / "execution_plan.json", self._execution_plan.to_dict())
                log_event(
                    self.log_path,
                    "strategy_completed",
                    generated_string_count=len(self._execution_plan.generated_strings),
                    coverage_gap_count=len(self._execution_plan.coverage_gaps),
                    architecture=self._execution_plan.architecture,
                )
                print(f"  Strategy: {self._execution_plan.strategy_rationale[:120]}...")
                source = "kit vocabulary" if self._kit_strings else "JD context"
                print(f"  {len(self._execution_plan.generated_strings)} compound strings synthesized from {source}")
                if self._execution_plan.coverage_gaps:
                    gap_with_boolean = sum(1 for g in self._execution_plan.coverage_gaps if g.get("suggested_boolean"))
                    print(f"  {len(self._execution_plan.coverage_gaps)} coverage gaps identified ({gap_with_boolean} with executable strings)")

                # --- Phase 2b: Verbose strategy logging ---
                self._print_strategy_details()

                # --- Phase 3: Build execution order from plan ---
                search_strings = self._build_ordered_search_strings()
                for search_string in search_strings:
                    self._hydrate_search_string_metadata(search_string)

                progress = Progress(
                    brief_name=self.brief_obj.id,
                    strings=search_strings,
                )
                self._progress = progress
                self._checkpoint_progress(progress)
                self._record_judgment_runtime_profile_event(resumed=False)

            # --- Execution ---
            print(f"\n--- Execution ({len(progress.strings)} strings) ---")
            log_event(self.log_path, "execution_started", string_count=len(progress.strings))
            current_block = ""
            block_strings: list[SearchString] = []

            if resume:
                for s in progress.strings:
                    if s.status not in {"done", "skipped"}:
                        print(f"\n  Resuming interrupted string #{s.id}: {s.name[:60]}")

                if progress.pending_block_name and progress.pending_block_string_ids:
                    pending_by_id = {s.id: s for s in progress.strings}
                    pending_block_strings = [
                        pending_by_id[sid]
                        for sid in progress.pending_block_string_ids
                        if sid in pending_by_id and pending_by_id[sid].status == "done"
                    ]
                    if pending_block_strings:
                        if progress.pending_block_ready:
                            print(
                                f"\n  Resuming pending adaptation for "
                                f"{progress.pending_block_name} ({len(pending_block_strings)} strings)"
                            )
                            await self._run_block_adaptation(
                                progress.pending_block_name,
                                pending_block_strings,
                                progress,
                                adapt_after_block,
                            )
                        else:
                            current_block = progress.pending_block_name
                            block_strings = pending_block_strings
                            print(
                                f"\n  Restored block context for {current_block} "
                                f"({len(block_strings)} completed strings)"
                            )
                    else:
                        self._clear_pending_block_adaptation(progress)
                        self._checkpoint_progress(progress)

            string_index = 0
            while string_index < len(progress.strings):
                if progress and hasattr(self, "_operator_stop_event") and self._operator_stop_event.is_set():
                    self._checkpoint_progress(progress)
                    print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                    raise OperatorStopRequested()
                search_string = progress.strings[string_index]
                if search_string.status == "done":
                    print(f"\n  [skip] String #{search_string.id}: {search_string.name} (already done)")
                    string_index += 1
                    continue
                if search_string.status == "skipped":
                    print(f"\n  [skip] String #{search_string.id}: {search_string.name} (skipped by strategy)")
                    string_index += 1
                    continue
                if (
                    search_string.status == "error"
                    and self._allocator_active_enabled()
                ):
                    print(f"\n  [skip] String #{search_string.id}: {search_string.name} (failed previously)")
                    string_index += 1
                    continue

                # Block transition → adaptation
                if search_string.block and search_string.block != current_block:
                    if current_block and block_strings:
                        await self._run_block_adaptation(
                            current_block, block_strings, progress, adapt_after_block
                        )
                        block_strings = []
                        # Re-evaluate this position because adaptation may have
                        # inserted replacement strings or re-ordered the queue.
                        continue
                    current_block = search_string.block
                    block_strings = []

                if progress and hasattr(self, "_operator_stop_event") and self._operator_stop_event.is_set():
                    self._checkpoint_progress(progress)
                    print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                    raise OperatorStopRequested()

                print(f"\n{'─' * 60}")
                print(f"  String #{search_string.id}: {search_string.name}")
                print(f"  [{search_string.block} / {search_string.subblock} / {search_string.string_type}]")
                print(f"{'─' * 60}")
                log_event(
                    self.log_path,
                    "string_started",
                    string_id=search_string.id,
                    name=search_string.name,
                    block=search_string.block,
                    rationale=search_string.name,
                )

                search_string.status = "in_progress"
                progress.current_string_id = search_string.id
                allocator_verdict: AllocationVerdict | None = None
                self._checkpoint_progress(progress, search_string=search_string)
                recovery_attempted = False
                original_disconnect_error: Exception | None = None
                while True:
                    try:
                        allocator_verdict = await self._process_string(
                            search_string,
                            progress,
                        )
                        break
                    except SessionExpired:
                        stop_reason = RunStopReason.SESSION_EXPIRED
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        if self._bias_monitor:
                            self._bias_monitor.save_checkpoint(str(self.bias_checkpoint_path))
                        raise
                    except GovernorLimitReached as e:
                        print(f"\n  [GOVERNOR] Session limit reached: {e.reason}")
                        stop_reason = RunStopReason.GOVERNOR_LIMIT
                        if self._runtime_run_id:
                            self._safety.record_governor_limit(
                                run_id=self._runtime_run_id,
                                reason=e.reason,
                                payload={"string_id": search_string.id},
                            )
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        raise
                    except ApiBudgetExhaustedError as e:
                        stop_reason = RunStopReason.API_BUDGET_EXHAUSTED
                        print(
                            "\n  [BUDGET] API credits exhausted during "
                            f"string #{search_string.id}: {e}"
                        )
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        log_event(
                            self.log_path,
                            "api_budget_exhausted",
                            string_id=search_string.id,
                            page=progress.current_page or None,
                            error=str(e),
                            traceback=traceback.format_exc(),
                        )
                        self._record_runtime_event(
                            search_string=search_string,
                            event_type="api_budget_exhausted",
                            payload={
                                "string_id": search_string.id,
                                "page": progress.current_page or None,
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        raise
                    except TransientPaginationError:
                        stop_reason = RunStopReason.BROWSER_DISCONNECT_UNRECOVERED
                        search_string.status = "in_progress"
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        raise
                    except GeographyRegimeError:
                        # P3a: a regime error is a run-level abort, never a
                        # per-string error. Without this clause the generic
                        # handler below would mark string #1 "error" and
                        # continue to string #2, which hits the same gate —
                        # the run would limp through every string erroring
                        # instead of stopping crisply on the wrong-pool
                        # condition (the exact wrapper-swallow failure mode
                        # the gate exists to prevent).
                        raise
                    except OperatorStopRequested:
                        stop_reason = RunStopReason.OPERATOR_STOP
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        raise
                    except PanelRecoveryError:
                        # Profile-panel recovery exhaustion is a run-level,
                        # retryable browser-state failure.  Never let the generic
                        # per-string handler turn it into a terminal ``error`` and
                        # advance past canonical facial positives still awaiting
                        # full review.
                        search_string.status = "in_progress"
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        raise
                    except AllocatorPolicyError:
                        # Active allocation is a run-level authority. A policy,
                        # pre-image, or durability fault must remain retryable;
                        # never terminalize one root and spend on the next.
                        search_string.status = "in_progress"
                        self._checkpoint_progress(progress)
                        raise
                    except Exception as e:
                        if _is_browser_disconnect_error(e):
                            if original_disconnect_error is None:
                                original_disconnect_error = e
                            search_string.status = "in_progress"
                            if recovery_attempted:
                                stop_reason = (
                                    RunStopReason.BROWSER_DISCONNECT_UNRECOVERED
                                )
                                original_disconnect_error.add_note(
                                    "Browser recovery retry also failed: "
                                    f"{type(e).__name__}: {e}"
                                )
                                self._checkpoint_progress(
                                    progress,
                                    search_string=search_string,
                                    page_num=progress.current_page or None,
                                )
                                raise original_disconnect_error from None

                            recovery_attempted = True
                            try:
                                recovery_snapshot = (
                                    await self._capture_recovery_snapshot(
                                        search_string,
                                        page_num=progress.current_page or None,
                                    )
                                )
                            except Exception as capture_error:
                                stop_reason = (
                                    RunStopReason.BROWSER_DISCONNECT_UNRECOVERED
                                )
                                original_disconnect_error.add_note(
                                    "Recovery snapshot capture failed: "
                                    f"{type(capture_error).__name__}: {capture_error}"
                                )
                                self._checkpoint_progress(
                                    progress,
                                    search_string=search_string,
                                    page_num=progress.current_page or None,
                                )
                                raise original_disconnect_error from None

                            self._checkpoint_progress(
                                progress,
                                search_string=search_string,
                                page_num=progress.current_page or None,
                            )
                            recovered = False
                            try:
                                recovered = await self._recovery_service.recover(
                                    run_id=self._runtime_run_id,
                                    snapshot=recovery_snapshot,
                                )
                                if not recovered:
                                    raise RuntimeError("browser recovery failed")
                                await self._reassert_session_location_after_recovery()
                                active_string_id = search_string.id
                                block_string_ids = {
                                    item.id for item in block_strings
                                }
                                reloaded_progress = (
                                    self._runtime_bridge.load_progress(
                                        self._runtime_run_id
                                    )
                                )
                                reloaded_states = (
                                    self._runtime_bridge.load_experiment_states(
                                        self._runtime_run_id,
                                        progress=reloaded_progress,
                                    )
                                )
                                self._hydrate_resume_funnel_from_runtime(
                                    reloaded_progress
                                )
                                reloaded_by_id = {
                                    item.id: item
                                    for item in reloaded_progress.strings
                                }
                                reloaded_search_string = reloaded_by_id[
                                    active_string_id
                                ]
                                reloaded_index = next(
                                    index
                                    for index, item in enumerate(
                                        reloaded_progress.strings
                                    )
                                    if item.id == active_string_id
                                )
                                reloaded_block_strings = [
                                    item
                                    for item in reloaded_progress.strings
                                    if item.id in block_string_ids
                                ]
                            except GeographyRegimeError as geo_error:
                                original_disconnect_error.add_note(
                                    "Geography-regime failure occurred during "
                                    "post-recovery reassertion: "
                                    f"{type(geo_error).__name__}: {geo_error}"
                                )
                                raise
                            except Exception as recovery_error:
                                stop_reason = (
                                    RunStopReason.BROWSER_DISCONNECT_UNRECOVERED
                                    if not recovered
                                    or _is_browser_disconnect_error(recovery_error)
                                    else RunStopReason.FATAL_RUNTIME_ERROR
                                )
                                original_disconnect_error.add_note(
                                    "Browser recovery/resume failed: "
                                    f"{type(recovery_error).__name__}: "
                                    f"{recovery_error}"
                                )
                                raise original_disconnect_error from None

                            progress = reloaded_progress
                            search_string = reloaded_search_string
                            string_index = reloaded_index
                            block_strings = reloaded_block_strings
                            self._progress = progress
                            self._experiment_states = reloaded_states
                            self._in_flight_urls.clear()
                            continue

                        if self._allocator_active_enabled():
                            # The bounded active canary is fail-closed across
                            # the whole control path, including raw sync or
                            # rehydration errors. Finalization checkpoints the
                            # restored pre-image without a stale root override.
                            search_string.status = "in_progress"
                            raise

                        # The standard lifecycle is string-local. An unexpected
                        # owner failure stays resumable and aborts the run; later
                        # strings never acquire browser authority from the error.
                        print(f"\n  [!] String #{search_string.id} failed: {e}")
                        search_string.status = "in_progress"
                        search_string.notes = (search_string.notes or "") + f" Error: {e}"
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        log_event(self.log_path, "string_error", string_id=search_string.id, error=str(e), traceback=traceback.format_exc())
                        raise

                if (
                    self._allocator_active_enabled()
                    and allocator_verdict is not None
                ):
                    pending_by_id = {item.id: item for item in progress.strings}
                    block_strings = [
                        pending_by_id[root_id]
                        for root_id in progress.pending_block_string_ids
                        if root_id in pending_by_id
                        and self._allocator_terminal_status(
                            pending_by_id[root_id].status
                        )
                    ]
                    selected_root_id = allocator_verdict.selected_root_id
                    if selected_root_id is not None:
                        selected_index = next(
                            (
                                index
                                for index, item in enumerate(progress.strings)
                                if item.id == selected_root_id
                            ),
                            -1,
                        )
                        if selected_index < 0:
                            raise AllocatorPolicyError(
                                "actuated allocator selection is missing from queue"
                            )
                        string_index = selected_index
                    else:
                        current_index = next(
                            (
                                index
                                for index, item in enumerate(progress.strings)
                                if item.id == search_string.id
                            ),
                            string_index,
                        )
                        string_index = current_index
                    # Phase-two actuation already checkpointed canonical state.
                    # In particular, never checkpoint with the stale root here.
                    continue

                if search_string.status == "in_progress":
                    search_string.status = "done"
                if (
                    search_string.status in {"done", "skipped"}
                    and any(
                        owner == search_string.id
                        for owner in self._resume_pending_full_owner_ids.values()
                    )
                ):
                    terminal_status = search_string.status
                    search_string.status = "in_progress"
                    self._checkpoint_progress(
                        progress,
                        search_string=search_string,
                    )
                    experiment_state = self._experiment_state_for(search_string)
                    await self._apply_opening_search(
                        search_string,
                        experiment_state,
                        experiment_state.current_boolean(),
                    )
                    page_cursor = (
                        experiment_state.active_allocator_page_cursor()
                    )
                    first_incomplete_page = (
                        page_cursor
                        if page_cursor > 0
                        else max(1, int(search_string.pages_reviewed or 1))
                    )
                    recovery_error: RuntimeError | None = None
                    try:
                        await self._recover_owner_pending_full_evaluations(
                            progress=progress,
                            search_string=search_string,
                            first_incomplete_page=first_incomplete_page,
                            string_stats=self._string_stats_for_processing(
                                search_string,
                                resuming=True,
                            ),
                        )
                    except RuntimeError as exc:
                        recovery_error = exc
                    outstanding_keys = [
                        key
                        for key in self._resume_pending_full_decisions
                        if self._resume_pending_full_owner_ids.get(key)
                        == search_string.id
                    ]
                    if outstanding_keys:
                        search_string.status = "in_progress"
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                        )
                        outstanding = ", ".join(
                            (
                                f"{snippet.name} ({snippet.profile_url})"
                                if snippet is not None
                                else key
                            )
                            for key in outstanding_keys
                            for snippet in [
                                self._resume_pending_full_snippets.get(key)
                            ]
                        )
                        outstanding_error = RuntimeError(
                            "outstanding full review(s) for string "
                            f"#{search_string.id}: {outstanding}"
                        )
                        # CLO-152: a recovery failure that ALSO left
                        # outstanding keys used to vanish here — chain it so
                        # the traceback carries both faults.
                        if recovery_error is not None:
                            raise outstanding_error from recovery_error
                        raise outstanding_error
                    if recovery_error is not None:
                        raise recovery_error
                    search_string.status = terminal_status
                block_strings.append(search_string)
                next_active = next(
                    (
                        candidate
                        for candidate in progress.strings[string_index + 1:]
                        if candidate.status not in {"done", "skipped"}
                    ),
                    None,
                )
                self._set_pending_block_adaptation(
                    progress,
                    current_block or search_string.block,
                    block_strings,
                    ready=next_active is None or next_active.block != (current_block or search_string.block),
                )
                self._checkpoint_progress(progress, search_string=search_string)
                if self._bias_monitor:
                    self._bias_monitor.save_checkpoint(str(self.bias_checkpoint_path))
                log_event(
                    self.log_path,
                    "string_complete",
                    **{**self.stats, "string_id": search_string.id},
                )
                self._print_session_summary(progress)
                string_index += 1

            if progress and hasattr(self, "_operator_stop_event") and self._operator_stop_event.is_set():
                self._checkpoint_progress(progress)
                print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                raise OperatorStopRequested()

            # Final block adaptation
            if current_block and block_strings:
                await self._run_block_adaptation(
                    current_block, block_strings, progress, adapt_after_block
                )
            if progress and hasattr(self, "_operator_stop_event") and self._operator_stop_event.is_set():
                self._checkpoint_progress(progress)
                print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                raise OperatorStopRequested()

        except SessionExpired:
            print("\n\n  [!] Session duration cap reached. Progress saved.")
            run_status = "interrupted"
            stop_reason = RunStopReason.SESSION_EXPIRED
            raise
        except GovernorLimitReached as e:
            print(f"\n\n  [!] Governor limit reached: {e.reason}. Progress saved.")
            run_status = "governor_limit_reached"
            stop_reason = RunStopReason.GOVERNOR_LIMIT
            raise
        except ApiBudgetExhaustedError:
            print("\n\n  [!] API budget exhausted. Progress saved; top up credits and resume.")
            run_status = "interrupted"
            stop_reason = RunStopReason.API_BUDGET_EXHAUSTED
            raise
        except OperatorStopRequested:
            run_status = "interrupted"
            stop_reason = RunStopReason.OPERATOR_STOP
            raise
        except asyncio.CancelledError:
            run_status = "interrupted"
            stop_reason = RunStopReason.OPERATOR_STOP
            raise
        except KeyboardInterrupt:
            print("\n\n  [!] Interrupted. Progress saved.")
            run_status = "interrupted"
            stop_reason = RunStopReason.OPERATOR_STOP
            raise
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            log_event(self.log_path, "pipeline_error", error=str(e), traceback=traceback.format_exc())
            if is_api_budget_exhausted_error(e):
                run_status = "interrupted"
                stop_reason = RunStopReason.API_BUDGET_EXHAUSTED
            else:
                run_status = (
                    "interrupted"
                    if stop_reason == RunStopReason.BROWSER_DISCONNECT_UNRECOVERED
                    else "error"
                )
                if stop_reason == RunStopReason.NORMAL:
                    stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
            raise
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_error: BaseException | None = None
            finalized_run_dir: Path | None = None
            finalization_error: BaseException | None = None
            try:
                try:
                    (
                        run_status,
                        stop_reason,
                        finalized_run_dir,
                        finalization_error,
                    ) = self._finish_runtime_record_and_freeze(
                        run_status=run_status,
                        stop_reason=stop_reason,
                        progress=self._progress,
                        snapshotter=self._finalize_run_snapshot,
                        include_bias_checkpoint=True,
                        primary_error=primary_error,
                    )
                except BaseException as freeze_error:
                    # CLO-152: the freezer is defensively written; if it still
                    # raises, record it as the finalization error (surfaced at
                    # the end of this finally when the run was otherwise clean)
                    # instead of letting it replace primary_error on the way
                    # out.
                    finalization_error = freeze_error
                    detail = (
                        str(freeze_error).splitlines()[0]
                        if str(freeze_error)
                        else type(freeze_error).__name__
                    )
                    print(
                        "  [warn] runtime-record finalization raised: "
                        f"{type(freeze_error).__name__}: {detail}"
                    )
            finally:
                try:
                    await self._cleanup_post_run_resources(
                        lock_acquired=lock_acquired
                    )
                except BaseException as exc:
                    cleanup_error = exc
            if cleanup_error is not None:
                run_status = "error"
                stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
                try:
                    self._safety.finish_run(
                        run_id=self._runtime_run_id,
                        status=run_status,
                        stop_reason=stop_reason,
                    )
                except BaseException as finish_error:
                    detail = (
                        str(finish_error).splitlines()[0]
                        if str(finish_error)
                        else type(finish_error).__name__
                    )
                    print(
                        "  [warn] canonical cleanup-failure finalization "
                        f"failed: {detail}"
                    )
                if primary_error is None and finalization_error is None:
                    finalization_error = cleanup_error
                else:
                    detail = (
                        str(cleanup_error).splitlines()[0]
                        if str(cleanup_error)
                        else type(cleanup_error).__name__
                    )
                    print(f"  [warn] post-run cleanup failed: {detail}")
            try:
                try:
                    honest_completion = (
                        primary_error is None
                        and
                        finalization_error is None
                        and
                        cleanup_error is None
                        and
                        run_status == "completed"
                        and stop_reason == RunStopReason.NORMAL
                        and config.LINKEDIN_TOTAL_PAGE_CAP <= 0
                        and isinstance(finalized_run_dir, Path)
                    )
                except Exception as gate_error:
                    # CLO-152: this expression runs even while unwinding a
                    # primary error — it must not be able to replace it.
                    honest_completion = False
                    print(
                        "  [warn] completion-gate evaluation failed: "
                        f"{type(gate_error).__name__}: {gate_error}"
                    )
                # Market intelligence is gated a second time, on whether the
                # STRATEGY finished — not just whether this run exited cleanly.
                # A multi-session campaign satisfies honest_completion once per
                # session, which rewrote the SPL artifact five separate times and
                # gave the operator a sequence of partial market reads instead of
                # one settled report. The run report below is deliberately NOT
                # gated this way: per-session visibility is wanted.
                if honest_completion and self._strategy_fully_executed():
                    self._enrich_run_snapshot(finalized_run_dir)
                try:
                    self._print_summary()
                except Exception as e:
                    print(f"  [warn] Summary print failed: {e}")
                if (
                    honest_completion
                    and
                    self._progress
                ):
                    try:
                        log_event(self.log_path, "report_started")
                        self._generate_run_report(self._progress)
                        log_event(self.log_path, "report_completed")
                    except Exception as e:
                        print(f"  [warn] Run report generation failed: {e}")
                        try:
                            log_event(
                                self.log_path,
                                "run_report_error",
                                error=str(e),
                                traceback=traceback.format_exc(),
                            )
                        except Exception as log_error:
                            print(
                                "  [warn] Run report error logging failed: "
                                f"{log_error}"
                            )
            finally:
                try:
                    log_event(
                        self.log_path,
                        "pipeline_end",
                        **self._pipeline_end_stats(),
                    )
                except Exception as e:
                    print(f"  [warn] Pipeline-end logging failed: {e}")
                self._persist_cost_rollup_sidecar()
            if primary_error is None and finalization_error is not None:
                raise finalization_error

    # ------------------------------------------------------------------
    # Browser crash recovery
    # ------------------------------------------------------------------

    async def _attempt_reconnect(
        self,
        recovery_url: str | None = None,
        max_attempts: int = 6,
        wait_seconds: int = 10,
    ) -> bool:
        """Try to reconnect to Chrome after a crash.

        Waits for the user to refresh LinkedIn Recruiter, then reconnects.
        Tries up to max_attempts times with wait_seconds between each.
        """
        import asyncio
        for attempt in range(max_attempts):
            print(f"  [reconnect] Attempt {attempt + 1}/{max_attempts} — waiting {wait_seconds}s for Chrome...")
            await asyncio.sleep(wait_seconds)
            try:
                await self.browser.disconnect()
            except Exception:
                pass
            try:
                await self.browser.connect()
                if recovery_url:
                    try:
                        await self.browser.navigate_to_search(recovery_url)
                    except Exception as nav_error:
                        print(f"  [reconnect] Reconnected but could not restore search page: {nav_error}")
                print(f"  [reconnect] Success — reconnected to LinkedIn Recruiter.")
                return True
            except Exception as e:
                print(f"  [reconnect] Failed: {e}")
        return False

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _enforce_constraint_manifest(self) -> None:
        """P3b: build, persist, and enforce the constraint-ownership manifest.

        Runs once the brief is FINAL (post-preflight on fresh runs; post-reload
        on resume). Persists constraint_manifest.json beside the run state and
        aborts with ConstraintManifestError when a stated constraint has zero
        owners — the drop becomes a decision, never an accident. The error
        propagates through run_full's except-and-re-raise and classifies as a
        stable day-cycle break (session_orchestrator), mirroring the
        geography gate.
        """
        from shared.constraint_manifest import (
            MANIFEST_FILENAME,
            assert_constraint_manifest_runnable,
            build_constraint_manifest,
        )

        manifest = build_constraint_manifest(self.brief_obj)
        self._constraint_manifest = manifest
        write_json(self.output_dir / MANIFEST_FILENAME, manifest)
        owned = sum(
            1 for entry in manifest["classes"].values() if entry["status"] == "owned"
        )
        print(
            f"  [constraints] Manifest: {owned} owned / "
            f"{len(manifest['classes'])} classes → {self.output_dir / MANIFEST_FILENAME}"
        )
        assert_constraint_manifest_runnable(manifest)

    @property
    def _session_location_applied(self) -> bool:
        return self._geography_service._session_location_applied

    @_session_location_applied.setter
    def _session_location_applied(self, value: bool) -> None:
        self._geography_service._session_location_applied = value

    @property
    def _session_geography_receipt(self) -> dict:
        return self._geography_service._session_geography_receipt

    @_session_geography_receipt.setter
    def _session_geography_receipt(self, value: dict) -> None:
        self._geography_service._session_geography_receipt = value

    def _session_geography_values(self) -> list[str]:
        return self._geography_service._session_geography_values()

    async def _apply_session_location_filter(self) -> None:
        return await self._geography_service._apply_session_location_filter()

    async def _resolve_and_reapply_geography(
        self, geo_values: list[str]
    ) -> list[str] | None:
        return await self._geography_service._resolve_and_reapply_geography(geo_values)

    async def _verify_session_geography_chips(self) -> None:
        return await self._geography_service._verify_session_geography_chips()

    async def _reassert_session_location_after_recovery(self) -> None:
        return await self._geography_service._reassert_session_location_after_recovery()

    async def _apply_opening_search(
        self,
        search_string: SearchString,
        experiment_state: LinkedInExperimentState,
        current_boolean: str,
    ) -> None:
        """Apply a string's OPENING search — keyword-led or filter-led.

        Phase 2 hop 4 (slice B, parts 2-4). A filter-led lane
        (acquisition_mode == 'linkedin_hybrid' AND the active variant carries
        non-empty structured filters, seeded from the checkpointed SearchString by
        bootstrap_experiment_state) applies its compiled structured plan; every
        other lane keeps the bare keyword entry byte-for-byte.

        The opening is NOT a rewrite, so it goes through
        browser.apply_advanced_search_plan — which applies keywords + controls and
        writes the R5 applied-only snapshot in one call, BELOW apply_variant's
        budget layer. It therefore never consumes the mutation/consecutive-rewrite
        budget nor emits a mutation event: opening structure is free, only
        mid-string rewrites spend budget.

        Location caution (part 4): the plan carries ONLY the lane's own
        structured_filters (sidebar locations included). The brief's session
        geography stays on _apply_session_location_filter (idempotent), so the two
        location paths never double-apply at browser.apply_location_filter.

        P3a pre-string invariant: BEFORE any opening search is entered — keyword
        or filter-led, fresh or resume — the session geography chips must be
        verified present on the live sidebar. This is the single choke point both
        opening paths flow through, so an unenumerated navigation race that
        dropped the chips is caught here rather than shipping an off-geo string.
        """
        await self._verify_session_geography_chips()
        active = experiment_state.active_variant
        if (
            search_string.acquisition_mode == "linkedin_hybrid"
            and not active.structured_filters.is_empty()
        ):
            from linkedin.advanced_search import compile_structured_filters_to_plan

            plan = compile_structured_filters_to_plan(
                active.structured_filters,
                keyword_boolean=current_boolean,
                acquisition_mode=search_string.acquisition_mode,
                # Slice D: the third compile call site that can carry a
                # structured_only active variant (alongside apply_variant at
                # search_mutation.py:~192 and the recovery snapshot at ~:717).
                # A structured_only variant drops the keyword from the OPENING
                # apply so a non-empty boolean is not entered as a keyword
                # search the filters are meant to carry instead.
                include_keyword=(active.surface != "structured_only"),
            )
            result = await self.browser.apply_advanced_search_plan(plan)
            try:
                log_event(
                    self.log_path,
                    "string_executed",
                    string_id=search_string.id,
                    executed_boolean=plan.keyword_boolean,
                    execution_surface="advanced",
                )
            except Exception as e:
                print(f"  [warn] provenance event failed: {e}")
            # Surface receipt (applied): record what actually landed on the live
            # sidebar vs fell back to keyword. This is the INTENDED-vs-APPLIED gap the
            # unlock exists to close, made visible per string. Fail-soft.
            try:
                fields = _surface_receipt.apply_receipt_fields(search_string, result)
                # P2.2: the receipt no longer dies in run_log.jsonl — persist
                # the last apply receipt on the SearchString so block reports,
                # adaptation, and the run report see actuator health.
                search_string.surface_receipt = fields
                print(_surface_receipt.format_apply_receipt(fields))
                if getattr(self, "log_path", None):
                    log_event(self.log_path, "surface_applied", **fields)
            except Exception as e:
                print(f"  [warn] provenance event failed: {e}")
        else:
            await self.browser.enter_search_string(current_boolean)
            try:
                log_event(
                    self.log_path,
                    "string_executed",
                    string_id=search_string.id,
                    executed_boolean=current_boolean,
                    execution_surface="keyword",
                )
            except Exception as e:
                print(f"  [warn] provenance event failed: {e}")

    async def _process_string(
        self,
        search_string: SearchString,
        progress: Progress,
    ) -> AllocationVerdict | None:
        """Process a string while keeping cursor-N recovery checkpoints non-teaching."""
        self._discard_incomplete_page_rollback(search_string.id)
        try:
            return await self._process_string_impl(search_string, progress)
        except BaseException:
            # CLO-152: a rollback failure on the unwind path must not replace
            # the in-flight exception it is cleaning up after.
            try:
                self._restore_incomplete_page_rollback(search_string)
            except Exception as rollback_error:
                print(
                    "  [warn] incomplete-page rollback restore failed "
                    f"({type(rollback_error).__name__}: {rollback_error}); "
                    "propagating the original error."
                )
            raise
        finally:
            self._discard_incomplete_page_rollback(search_string.id)
            pending = getattr(self, "_pending_allocator_checkpoint", None)
            if (
                isinstance(pending, dict)
                and int(pending.get("root_string_id", 0) or 0)
                == search_string.id
            ):
                self._pending_allocator_checkpoint = None
            self._allocator_page_identity = None

    async def _process_string_impl(
        self,
        search_string: SearchString,
        progress: Progress,
    ) -> AllocationVerdict | None:
        # Reset triage tightening state per string
        self._triage_tightened = False
        self._tightening_prefix = ""
        experiment_state = self._experiment_state_for(search_string)

        dispatch_verdict = self._prepare_active_allocator_dispatch(
            progress=progress,
            current=search_string,
        )
        if dispatch_verdict is not None:
            return dispatch_verdict

        # Classify this first actual spend against the persisted counterfactual
        # frontier before any page evidence can be opened.
        self._check_allocator_pre_spend(
            progress=progress,
            current=search_string,
        )

        # Emergency recovery check before starting
        await self._ensure_browser_healthy()

        current_boolean = experiment_state.current_boolean()
        page_cursor = experiment_state.active_allocator_page_cursor()
        first_incomplete_page = (
            page_cursor
            if page_cursor > 0
            else max(1, int(search_string.pages_reviewed or 1))
        )
        resuming = search_string.pages_reviewed > 0 or first_incomplete_page > 1

        if resuming:
            print(f"  Resuming string on first incomplete page {first_incomplete_page}")

            # Dismiss any open profile panel
            try:
                await self.browser.go_back_to_results()
            except Exception:
                pass

            # Ensure we're on a search page — this brief's, not merely someone's.
            # F4: `/talent/hire/` present was the whole test, so ANY project's
            # Recruiter page satisfied resume and the owner's Boolean was
            # re-entered on it. Resume is exactly where this bites: the page was
            # bound by a rebind or a crash recovery that had no idea which
            # project the run owns. `_is_foreign_project_page` applies F1's
            # asymmetry — a page that does not NAME the brief's project is
            # unverified, and unverified is wrong — so the navigation below
            # corrects it instead of the run inheriting a foreign pipeline for
            # every review and save on the page. The two original clauses stay
            # because with no project pinned the predicate is False by design.
            current_page_url = self.browser.page.url
            if (
                "/manage/" in current_page_url
                or "/talent/hire/" not in current_page_url
                or _is_foreign_project_page(
                    current_page_url, self.brief_obj.linkedin_project_id
                )
            ):
                project_url = self._get_project_url()
                if project_url:
                    print(f"  Wrong page ({current_page_url[:60]}...). Navigating to search...")
                    await self.browser.navigate_to_search(project_url)
                else:
                    print(f"  Wrong page and no project URL. Reloading...")
                    await self.browser.page.reload(wait_until="domcontentloaded", timeout=30000)
                    await self.browser.page.wait_for_timeout(4000)
                # The (re)navigation dropped the sidebar; re-assert the session location.
                self._geography_service.reset_location_applied()
                await self._apply_session_location_filter()

            # Always re-enter keywords — browser state is not trustworthy on resume.
            # Phase 2 hop 4 (slice B, part 3): a hybrid lane re-applies its persisted
            # structured filters (seeded onto the active variant from the checkpointed
            # SearchString) through the same applied-only snapshot path as the opening,
            # so cross-process resume reconstructs the structured search rather than
            # dropping to keyword-only. Boolean lanes keep the bare keyword re-entry.
            print(f"  Re-entering Boolean: {current_boolean[:80]}...")
            await self._apply_opening_search(search_string, experiment_state, current_boolean)
            result_count_text = await self.browser.get_results_count_text()
            result_count = await self.browser.get_results_count()
            search_string.result_count = result_count
            self._note_string_results_seen(search_string, result_count)
            print(f"  Results: {result_count_text or 'unknown'}")

            log_event(self.log_path, "string_resumed", string_id=search_string.id,
                      pages_done=search_string.pages_reviewed,
                      result_count=result_count, result_count_text=result_count_text)
        else:
            # Fresh string — enter the boolean (or the reasoned structured surface
            # for a hybrid lane, slice B part 2).
            print(f"  Entering Boolean: {current_boolean[:80]}...")
            await self._apply_opening_search(search_string, experiment_state, current_boolean)

            # Snapshot URL after search entry — known good state for recovery
            try:
                self._record_last_good_url(self.browser.page.url)
            except Exception:
                pass

            result_count_text = await self.browser.get_results_count_text()
            result_count = await self.browser.get_results_count()
            search_string.result_count = result_count
            self._note_string_results_seen(search_string, result_count)
            print(f"  Results: {result_count_text or 'unknown'} (parsed: {result_count})")
            log_event(self.log_path, "string_results", string_id=search_string.id,
                      result_count=result_count, result_count_text=result_count_text)

        # Validate canonical pending work only after this owner's exact search
        # surface is active. Foreign work stays inert until its own turn.
        owned_pending = (
            []
            if self._allocator_active_enabled()
            else self._validated_owner_pending_full_snippets(
                progress=progress,
                search_string=search_string,
            )
        )
        if any(
            max(1, int(snippet.page or 1)) > first_incomplete_page
            for _key, snippet in owned_pending
        ):
            search_string.status = "in_progress"
            self._checkpoint_progress(progress, search_string=search_string)
            raise RuntimeError(
                "canonical pending full review is ahead of the active owner cursor"
            )

        # Guard: don't proceed with invalid result count.
        if result_count <= 0:
            if owned_pending:
                search_string.status = "in_progress"
                self._checkpoint_progress(progress, search_string=search_string)
                raise RuntimeError(
                    "active owner has canonical pending full reviews but its "
                    "search surface returned no results"
                )
            if search_string.id in self._string_ids_with_results:
                try:
                    log_event(
                        self.log_path,
                        "zero_results_after_prior_results",
                        string_id=search_string.id,
                        result_count=result_count,
                    )
                except Exception:
                    pass
            print(f"  No results detected (result_count={result_count}). Skipping string.")
            search_string.notes = (search_string.notes or "") + " Skipped: no results on entry."
            return self._checkpoint_allocator_exhaustion_transition(
                progress=progress,
                search_string=search_string,
                page_num=first_incomplete_page,
                legacy_terminal_status="skipped",
            )

        # Older-page pending reviews belong to this string and may resume only
        # after its exact Boolean/filter surface owns the browser. Current-page
        # pending reviews remain on the normal page-local path below.
        string_stats = self._string_stats_for_processing(
            search_string,
            resuming=resuming,
        )
        resume_surface_page = 1
        if (
            not self._allocator_active_enabled()
            and resuming
            and first_incomplete_page > 1
        ):
            resume_surface_page = await self._recover_owner_pending_full_evaluations(
                progress=progress,
                search_string=search_string,
                first_incomplete_page=first_incomplete_page,
                string_stats=string_stats,
            )

        if not resuming or not experiment_state.mode:
            experiment_state.mode = self._initial_search_mode(result_count)
            if (
                experiment_state.mode == "paginate"
                and not experiment_state.committed_variant_id
            ):
                # Direct pagination (sub-RECON pool) IS a commitment to the
                # root variant. Without it the committed-path accounting
                # never arms — committed_zero_signal_streak stays 0 and the
                # stop rule is unreachable, so a sub-500 pool paginates
                # until it physically runs out (2026-07-06 SPL-MM live run:
                # 386 results, 16 pages, 13 consecutive zero-signal).
                experiment_state.commit_variant()
        experiment_state.active_variant.result_count = result_count
        experiment_state.apply_shadow(search_string)
        if experiment_state.mode == "recon":
            print(f"  [phase] RECON — {result_count} results, evaluating page 1 before committing")
        elif experiment_state.mode == "experiment":
            print(f"  [phase] EXPERIMENT — resuming variant exploration for {result_count} results")
        else:
            print(f"  [phase] PAGINATE — proceeding with committed variant")

        # Accumulating context for page-level adaptation
        all_candidates: list[dict] = []
        page_num = 1
        max_pages = config.MAX_PAGES_PER_STRING or 999
        if (
            self._allocator_active_enabled()
            and first_incomplete_page > max_pages
        ):
            raise AllocatorPolicyError(
                "active allocator cursor exceeds MAX_PAGES_PER_STRING"
            )

        # On resume, skip to the interrupted page (re-process it; dupe filter handles already-seen)
        if resuming and first_incomplete_page > 1:
            target_page = first_incomplete_page
            print(f"  Advancing to page {target_page}...")
            for page_at_exhaustion in range(
                resume_surface_page,
                target_page,
            ):
                (
                    has_next,
                    transient_suspected,
                ) = await self._go_to_next_page_with_transient_retry(
                    result_count=result_count,
                    page_num=page_at_exhaustion,
                )
                if not has_next:
                    print("  No more pages after resume point.")
                    pages_remaining_by_math = self._pages_remaining_by_result_count(
                        result_count,
                        page_at_exhaustion,
                    )
                    try:
                        log_event(
                            self.log_path,
                            "resume_fastforward_exhausted",
                            string_id=search_string.id,
                            page_num=page_at_exhaustion,
                            target_page=target_page,
                            result_count=result_count,
                            pages_remaining_by_math=pages_remaining_by_math,
                            transient_suspected=transient_suspected,
                        )
                    except Exception as e:
                        print(f"  [warn] resume_fastforward_exhausted event failed: {e}")
                    if transient_suspected:
                        search_string.status = "in_progress"
                        search_string.notes = (
                            (search_string.notes or "")
                            + " transient resume fast-forward suspected; "
                            "result-count math indicated pages remaining."
                        )
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                        )
                    else:
                        active_transition = (
                            self._checkpoint_allocator_exhaustion_transition(
                                progress=progress,
                                search_string=search_string,
                                page_num=page_at_exhaustion,
                            )
                        )
                        if active_transition is not None:
                            return active_transition
                    if transient_suspected:
                        raise TransientPaginationError(
                            "transient pagination during resume fast-forward"
                        )
                    return None
            page_num = target_page

        while page_num <= max_pages:
            self._check_allocator_pre_spend(
                progress=progress,
                current=search_string,
            )
            # Emergency recovery check before each page
            await self._ensure_browser_healthy()

            phase_label = search_string.phase.upper()
            refinement_depth = len(search_string.refinement_stack)
            print(f"\n  --- Page {page_num} [{phase_label}] (refinement depth: {refinement_depth}) ---")
            progress.current_page = page_num
            # Cursor N means page N has started but is not yet complete. Any
            # checkpoint during review therefore retries N after a crash.
            experiment_state.set_active_allocator_page_cursor(page_num)
            self._checkpoint_progress(
                progress,
                search_string=search_string,
                page_num=page_num,
            )
            self._arm_incomplete_page_rollback(search_string, experiment_state)

            # Check for 0-result page before trying to scroll/extract
            no_results_visible = False
            try:
                no_results = self.browser.page.locator('text="No search results"').first
                no_results_visible = await no_results.is_visible(timeout=2000)
            except Exception:
                pass
            if no_results_visible:
                if any(
                    max(1, int(snippet.page or 1)) == page_num
                    for _key, snippet in owned_pending
                ):
                    search_string.status = "in_progress"
                    self._checkpoint_progress(
                        progress,
                        search_string=search_string,
                        page_num=page_num,
                    )
                    raise RuntimeError(
                        "active owner page has canonical pending full reviews "
                        "but the browser reported no search results"
                    )
                print("  No search results for this string. Skipping.")
                active_transition = (
                    self._checkpoint_allocator_exhaustion_transition(
                        progress=progress,
                        search_string=search_string,
                        page_num=page_num,
                        completed_page=True,
                    )
                )
                self._discard_incomplete_page_rollback(search_string.id)
                if active_transition is not None:
                    return active_transition
                break

            # Review the page top-to-bottom, card by card.
            page_report = _PageReport(
                string_id=search_string.id,
                string_name=search_string.name,
                page=page_num,
                result_count=result_count,
            )
            stats_before_page = dict(string_stats)

            glance_result = await self._review_page_sequentially(
                search_string=search_string,
                page_num=page_num,
                result_count=result_count,
                page_report=page_report,
                all_candidates=all_candidates,
                string_stats=string_stats,
                progress=progress,
            )

            self._assert_page_full_reviews_settled(
                progress=progress,
                search_string=search_string,
                page_num=page_num,
            )
            page_observed = self._page_observation()
            # Compute immutable shadow evidence before legacy page metrics can
            # make this just-observed page look like pre-existing depth.
            self._stage_allocator_page_checkpoint(
                progress=progress,
                search_string=search_string,
                experiment_state=experiment_state,
                page_num=page_num,
                page_observed=page_observed,
            )
            allocator_page_verdict = None
            if self._allocator_active_enabled():
                pending_allocator = self._pending_allocator_checkpoint
                if not isinstance(pending_allocator, dict) or not isinstance(
                    pending_allocator.get("verdict"), AllocationVerdict
                ):
                    raise AllocatorPolicyError(
                        "active allocator did not stage a completed-page verdict"
                    )
                allocator_page_verdict = pending_allocator["verdict"]
            allocator_controls_page = bool(
                allocator_page_verdict is not None
                and (
                    allocator_page_verdict.action
                    is not AllocationAction.CONTINUE
                    or allocator_page_verdict.reason == "opening_probe"
                )
            )
            if (
                int(page_observed.get("extracted", 0) or 0)
                < int(page_observed.get("slots", 0) or 0)
                and not page_observed.get("break_reason")
            ):
                try:
                    log_event(
                        self.log_path,
                        "page_observation_gap",
                        string_id=search_string.id,
                        page=page_num,
                        slots=int(page_observed.get("slots", 0) or 0),
                        extracted=int(page_observed.get("extracted", 0) or 0),
                    )
                except Exception as e:
                    print(f"  [warn] page_observation_gap event failed: {e}")

            string_stats["pages"] = page_num

            page_stats = self._page_stat_delta(stats_before_page, string_stats)
            self._current_variant_candidates = list(all_candidates)
            page_insights = self._build_page_insights(
                page_num=page_num,
                result_count=result_count,
                preview_snippets=list(self._latest_page_preview_snippets),
                all_candidates=all_candidates,
                glance_result=glance_result,
            )
            experiment_state.record_variant_metrics(
                page_num=page_num,
                result_count=result_count,
                page_stats=page_stats,
                page_insights=page_insights,
            )
            experiment_state.record_family_page_metrics(
                page_num=page_num,
                result_count=result_count,
                page_stats=page_stats,
                page_insights=page_insights,
            )
            experiment_state.note_page_review()
            completed_variant = experiment_state.active_variant
            experiment_state.apply_shadow(search_string)
            if int(config.LINKEDIN_TOTAL_PAGE_CAP) > 0:
                self._sync_bounded_page_stats_for_checkpoint(
                    search_string,
                    string_stats,
                )
            page_report.print_report(self.stats)

            if allocator_controls_page:
                lifecycle_decision = None
                assessment = {
                    "decision": "continue",
                    "rationale": (
                        "allocator owns the opening probe or cross-root transition"
                    ),
                    "page": page_num,
                }
                decision = "continue"
            else:
                lifecycle_decision = self._evaluate_variant_lifecycle(
                    search_string=search_string,
                    experiment_state=experiment_state,
                    allow_terminal=not self._allocator_active_enabled(),
                )

                assessment = await self._assess_string_state(
                    search_string=search_string,
                    experiment_state=experiment_state,
                    page_num=page_num,
                    result_count=result_count,
                    string_stats=string_stats,
                    page_stats=page_stats,
                    page_insights=page_insights,
                    remaining_queued_strings=self._remaining_queued_strings(
                        progress,
                        current_string_id=search_string.id,
                    ),
                )
                decision = assessment["decision"]
                if lifecycle_decision:
                    decision = lifecycle_decision
            if experiment_state.mode == "recon":
                scout_bucket = assessment.get("scout_gate_bucket") or "recon"
                print(f"  [adapt] Scout gate ({scout_bucket}): {assessment['rationale']}")
            else:
                # One verdict line per reviewed page — the paginate-mode
                # counterpart of the scout-gate line above. Before this,
                # every non-RECON page decided continue/stop silently (the
                # operator watched 16 pages scroll with zero decision
                # visibility, 2026-07-06 SPL-MM).
                print(
                    f"  [adapt] Page {page_num} verdict: {decision} — "
                    f"{assessment['rationale']} (signal={assessment.get('page_signal')}, "
                    f"zero-signal streak={assessment.get('committed_zero_signal_streak')})"
                )

            if (
                self._allocator_active_enabled()
                and decision == "resume_committed"
            ):
                # Restoring a prior variant fast-forwards the browser before
                # the completed-page interlocks. Active authority keeps the
                # selected root on its current variant at this boundary.
                decision = "continue"

            if decision in {"refine_committed", "spawn_recall_sibling"}:
                drift_variant, drift_summary = await self._plan_drift_refinement(
                    search_string=search_string,
                    experiment_state=experiment_state,
                    current_boolean=current_boolean,
                    result_count=result_count,
                    result_count_text=result_count_text,
                    page_insights=page_insights,
                    page_stats=page_stats,
                )
                if drift_variant is not None:
                    # Hop 3: a hybrid lane's structured filters ride onto the drift
                    # variant so apply_variant takes the structured path (no-op for
                    # boolean lanes).
                    seed_structured_filters_onto_variants(
                        search_string.structured_filters, [drift_variant]
                    )
                    experiment_state.variants[drift_variant.variant_id] = drift_variant
                    mutation = await self._search_mutation_executor.apply_variant(
                        search_string=search_string,
                        experiment_state=experiment_state,
                        variant=drift_variant,
                        mutation_kind="drift",
                        mutation_summary=drift_summary,
                        # Slice C (part 6): a mid-run promote (drift carries structured
                        # filters) executes as hybrid even on a compiled-boolean lane,
                        # else apply_variant rejects it as unsupported.
                        acquisition_mode=self._acquisition_mode_for_variant(
                            search_string, drift_variant
                        ),
                    )
                    if mutation.applied:
                        # The completed page and accepted query transition are
                        # checkpointed together. Until this point the durable
                        # cursor remains N, so a crash retries the undecided page.
                        completed_variant.allocator_page_cursor = page_num + 1
                        experiment_state.set_active_allocator_page_cursor(1)
                        current_boolean = experiment_state.current_boolean()
                        result_count = mutation.result_count
                        result_count_text = mutation.result_count_text
                        search_string.result_count = result_count
                        experiment_state.apply_shadow(search_string)
                        search_string.notes = (
                            (search_string.notes or "")
                            + f" Drift rescue {drift_variant.variant_kind} applied on page {page_num}."
                        )
                        if int(config.LINKEDIN_TOTAL_PAGE_CAP) > 0:
                            self._sync_bounded_page_stats_for_checkpoint(
                                search_string,
                                string_stats,
                            )
                        completed_page_num = page_num
                        search_string.pages_reviewed = 0
                        page_num = 1
                        all_candidates.clear()
                        string_stats = self._fresh_string_stats()
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=1,
                            completed_page=True,
                        )
                        self._discard_incomplete_page_rollback(search_string.id)
                        active_transition = (
                            self._finish_completed_page_allocator_boundary(
                                progress=progress,
                                search_string=search_string,
                                page_num=completed_page_num,
                            )
                        )
                        if (
                            active_transition is not None
                            and active_transition.action
                            is not AllocationAction.CONTINUE
                        ):
                            return active_transition
                        continue
                decision = "continue"

            if decision == "resume_committed":
                failed_drift_page = page_num
                experiment_state.resume_committed_after_failed_drift()
                current_boolean = experiment_state.current_boolean()
                committed_variant = experiment_state.active_variant
                committed_page = experiment_state.active_allocator_page_cursor()
                if committed_page <= 0:
                    # Legacy checkpoints still carry trustworthy completed-page
                    # metrics on each variant even though they have no cursor.
                    committed_page = max(1, committed_variant.pages_reviewed + 1)
                    experiment_state.set_active_allocator_page_cursor(committed_page)

                experiment_state.apply_shadow(search_string)
                search_string.pages_reviewed = int(committed_variant.pages_reviewed)
                print(
                    f"  [adapt] Re-entering committed variant on page {committed_page}: "
                    f"{current_boolean[:80]}..."
                )
                await self._apply_opening_search(
                    search_string,
                    experiment_state,
                    current_boolean,
                )
                result_count_text = await self.browser.get_results_count_text()
                result_count = await self.browser.get_results_count()
                search_string.result_count = result_count
                committed_variant.result_count = result_count

                for fastforward_index in range(committed_page - 1):
                    page_at_exhaustion = fastforward_index + 1
                    (
                        has_next,
                        transient_suspected,
                    ) = await self._go_to_next_page_with_transient_retry(
                        result_count=result_count,
                        page_num=page_at_exhaustion,
                    )
                    if has_next:
                        continue
                    print("  No more pages while restoring committed variant.")
                    pages_remaining_by_math = self._pages_remaining_by_result_count(
                        result_count,
                        page_at_exhaustion,
                    )
                    try:
                        log_event(
                            self.log_path,
                            "resume_committed_fastforward_exhausted",
                            string_id=search_string.id,
                            failed_drift_page=failed_drift_page,
                            page_num=page_at_exhaustion,
                            target_page=committed_page,
                            result_count=result_count,
                            pages_remaining_by_math=pages_remaining_by_math,
                            transient_suspected=transient_suspected,
                        )
                    except Exception as e:
                        print(
                            "  [warn] resume_committed_fastforward_exhausted "
                            f"event failed: {e}"
                        )
                    if transient_suspected:
                        search_string.status = "in_progress"
                        search_string.notes = (
                            (search_string.notes or "")
                            + " transient committed-variant fast-forward suspected; "
                            "result-count math indicated pages remaining."
                        )
                        raise TransientPaginationError(
                            "transient pagination while restoring committed variant"
                        )
                    completed_variant.allocator_page_cursor = failed_drift_page + 1
                    # First persist the completed drift page with its unchanged
                    # legacy restore action.  Physical exhaustion is a second,
                    # observation-free allocator revision.
                    self._checkpoint_progress(
                        progress,
                        search_string=search_string,
                        completed_page=True,
                    )
                    self._discard_incomplete_page_rollback(search_string.id)
                    active_transition = (
                        self._finish_completed_page_allocator_boundary(
                            progress=progress,
                            search_string=search_string,
                            page_num=failed_drift_page,
                        )
                    )
                    if (
                        active_transition is not None
                        and active_transition.action
                        is not AllocationAction.CONTINUE
                    ):
                        return active_transition
                    return self._checkpoint_allocator_exhaustion_transition(
                        progress=progress,
                        search_string=search_string,
                        page_num=page_at_exhaustion,
                    )

                completed_variant.allocator_page_cursor = failed_drift_page + 1
                page_num = committed_page
                progress.current_page = committed_page
                all_candidates.clear()
                string_stats = self._fresh_string_stats()
                search_string.notes = (
                    (search_string.notes or "")
                    + f" Drift rescue failed on page {failed_drift_page}; "
                    f"resumed committed variant on page {committed_page}."
                )
                # Persist the accepted restore only after the browser surface,
                # query identity, and variant-local cursor agree.
                self._checkpoint_progress(
                    progress,
                    search_string=search_string,
                    completed_page=True,
                )
                self._discard_incomplete_page_rollback(search_string.id)
                continue

            if decision == "experiment":
                if experiment_state.mode != "experiment" or experiment_state.next_planned_variant() is None:
                    planned_variants = await self._plan_variant_experiments(
                        search_string=search_string,
                        experiment_state=experiment_state,
                        current_boolean=current_boolean,
                        result_count=result_count,
                        result_count_text=result_count_text,
                        page_insights=page_insights,
                        all_candidates=all_candidates,
                        string_stats=string_stats,
                    )
                    if planned_variants:
                        # Hop 3: seed the hybrid lane's structured filters onto the
                        # planned variants before they enter the experiment round, so
                        # next_planned_variant hands apply_variant a filter-bearing
                        # variant (no-op for boolean lanes).
                        seed_structured_filters_onto_variants(
                            search_string.structured_filters, planned_variants
                        )
                        experiment_state.begin_experiment_round(planned_variants)
                        print(f"  [adapt] Planning {len(planned_variants)} sibling experiment(s)")
                    else:
                        decision = "continue" if experiment_state.mode == "paginate" else "commit"

                if decision == "experiment":
                    next_variant = experiment_state.next_planned_variant()
                    if next_variant is None:
                        best_variant = experiment_state.best_variant()
                        if best_variant.variant_id == experiment_state.active_variant_id and best_variant.variant_id != "root":
                            decision = "commit"
                        else:
                            decision = "stop"
                    else:
                        mutation = await self._search_mutation_executor.apply_variant(
                            search_string=search_string,
                            experiment_state=experiment_state,
                            variant=next_variant,
                            # Slice C (part 6): a mid-run promote (variant carries
                            # structured filters) executes as hybrid even on a
                            # compiled-boolean lane, else apply_variant rejects it.
                            acquisition_mode=self._acquisition_mode_for_variant(
                                search_string, next_variant
                            ),
                        )
                        if mutation.applied:
                            completed_variant.allocator_page_cursor = page_num + 1
                            experiment_state.set_active_allocator_page_cursor(1)
                            current_boolean = experiment_state.current_boolean()
                            result_count = mutation.result_count
                            result_count_text = mutation.result_count_text
                            search_string.result_count = result_count
                            experiment_state.apply_shadow(search_string)
                            search_string.notes = (
                                (search_string.notes or "")
                                + f" Variant {next_variant.variant_kind} applied on page {page_num}."
                            )
                            if int(config.LINKEDIN_TOTAL_PAGE_CAP) > 0:
                                self._sync_bounded_page_stats_for_checkpoint(
                                    search_string,
                                    string_stats,
                                )
                            completed_page_num = page_num
                            search_string.pages_reviewed = 0
                            page_num = 1
                            all_candidates.clear()
                            string_stats = self._fresh_string_stats()
                            self._checkpoint_progress(
                                progress,
                                search_string=search_string,
                                page_num=1,
                                completed_page=True,
                            )
                            self._discard_incomplete_page_rollback(search_string.id)
                            active_transition = (
                                self._finish_completed_page_allocator_boundary(
                                    progress=progress,
                                    search_string=search_string,
                                    page_num=completed_page_num,
                                )
                            )
                            if (
                                active_transition is not None
                                and active_transition.action
                                is not AllocationAction.CONTINUE
                            ):
                                return active_transition
                            continue
                        best_variant = experiment_state.best_variant()
                        if best_variant.variant_id == experiment_state.active_variant_id and best_variant.variant_id != "root":
                            decision = "commit"
                        else:
                            decision = "stop"

            if self._allocator_active_enabled() and decision == "stop":
                # In active mode the local policy may rewrite or commit the
                # selected root, but it cannot terminate allocator-owned work.
                decision = "continue"

            if decision == "commit":
                committed = experiment_state.commit_variant()
                if assessment.get("bootstrap_early_snapshot"):
                    experiment_state.early_signal_snapshot = LinkedInVariantSnapshot.from_page(
                        page_num=page_num,
                        result_count=result_count,
                        page_insights=page_insights,
                        page_stats=page_stats,
                    )
                current_boolean = committed.boolean
                search_string.result_count = committed.result_count or result_count
                experiment_state.apply_shadow(search_string)
                print(f"  [adapt] COMMIT → {committed.variant_kind} variant")
                search_string.notes = (search_string.notes or "") + f" Committed {committed.variant_kind} variant on page {page_num}."

            if decision == "stop":
                self._maybe_discover_fallback_candidates(
                    search_string,
                    trigger_reason="experiment_stop",
                )
                print("  [adapt] STOP — signal exhausted for this root family.")
                search_string.notes = (search_string.notes or "") + f" Stopped after page {page_num}."
                search_string.status = "done"

            # The page becomes durably complete only after its authoritative
            # local decision is reflected in state. A crash anywhere above
            # retries N; this checkpoint resumes at N+1 with the same action.
            completed_variant.allocator_page_cursor = page_num + 1
            experiment_state.apply_shadow(search_string)
            if int(config.LINKEDIN_TOTAL_PAGE_CAP) > 0:
                self._sync_bounded_page_stats_for_checkpoint(
                    search_string,
                    string_stats,
                )
            self._checkpoint_progress(
                progress,
                search_string=search_string,
                page_num=page_num,
                completed_page=True,
            )
            self._discard_incomplete_page_rollback(search_string.id)
            active_transition = self._finish_completed_page_allocator_boundary(
                progress=progress,
                search_string=search_string,
                page_num=page_num,
            )
            if (
                active_transition is not None
                and active_transition.action is not AllocationAction.CONTINUE
            ):
                return active_transition

            if decision == "stop":
                break

            # "continue" or "paginate" — proceed to next page
            if page_num < max_pages:
                has_next, transient_suspected = await self._go_to_next_page_with_transient_retry(
                    result_count=result_count,
                    page_num=page_num,
                )
                if not has_next:
                    print("  No more pages.")
                    pages_remaining_by_math = self._pages_remaining_by_result_count(
                        result_count,
                        page_num,
                    )
                    if transient_suspected:
                        search_string.notes = (
                            (search_string.notes or "")
                            + " transient pagination suspected; "
                            "result-count math indicated pages remaining."
                        )
                        search_string.status = "in_progress"
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                        )
                        raise TransientPaginationError(
                            "transient pagination before next results page"
                        )
                    try:
                        log_event(
                            self.log_path,
                            "pagination_exhausted",
                            string_id=search_string.id,
                            page_num=page_num,
                            result_count=result_count,
                            pages_remaining_by_math=pages_remaining_by_math,
                            transient_suspected=transient_suspected,
                        )
                    except Exception as e:
                        print(f"  [warn] pagination_exhausted event failed: {e}")
                    active_transition = self._checkpoint_allocator_exhaustion_transition(
                        progress=progress,
                        search_string=search_string,
                        page_num=page_num,
                    )
                    if active_transition is not None:
                        return active_transition
                    break
                page_num += 1
                await asyncio.sleep(human_delay_correlated(config.PAGE_DELAY_SECONDS, channel="page_turn"))
            else:
                try:
                    log_event(
                        self.log_path,
                        "page_cap_reached",
                        string_id=search_string.id,
                        page_num=page_num,
                        result_count=result_count,
                        max_pages=max_pages,
                    )
                except Exception as e:
                    print(f"  [warn] page_cap_reached event failed: {e}")
                if self._allocator_active_enabled():
                    search_string.status = "in_progress"
                    self._checkpoint_progress(progress)
                    raise OperatorStopRequested("max_pages_per_string")
                search_string.status = "done"
                self._checkpoint_progress(
                    progress,
                    search_string=search_string,
                )
                break

        # Persist cumulative funnel truth for block-level aggregation. On a
        # resume, string_stats starts from the canonical run-chain baseline.
        self._sync_bounded_page_stats_for_checkpoint(search_string, string_stats)
        self._hydrate_search_string_metadata(search_string)

    async def _extract_card_snippet(
        self,
        search_string: SearchString,
        page_num: int,
        card_index: int,
    ) -> CandidateSnippet | None:
        self._ensure_services()
        result = await self._acquisition_service.extract_card_snippet(
            search_string,
            page_num,
            card_index,
        )
        self._last_card_extraction_error = (
            self._acquisition_service.last_card_extraction_error
        )
        return result.snippet if result else None

    async def _preview_skip_pause(self, reason: str) -> None:
        """Pause on visible skips so the review flow does not feel bursty."""
        if reason == "facial_no":
            dwell = human_delay_correlated(random.uniform(0.8, 2.0), channel="preview_no")
        elif reason == "facial_skip":
            dwell = human_delay_correlated(random.uniform(0.7, 1.6), channel="preview_skip")
        elif reason in {"already_saved", "duplicate"}:
            dwell = human_delay_correlated(random.uniform(0.2, 0.6), channel="preview_skip")
        else:
            dwell = human_delay_correlated(random.uniform(0.2, 0.5), channel="preview_skip")
        await asyncio.sleep(dwell)

    async def _review_page_sequentially(
        self,
        search_string: SearchString,
        page_num: int,
        result_count: int,
        page_report: "_PageReport",
        all_candidates: list[dict],
        string_stats: dict,
        progress: Progress | None = None,
    ) -> GlanceResult | None:
        """Review the current results page top-to-bottom, card by card."""
        # V2 briefs: use batch facial triage (one LLM call for all candidates on page)
        if self.brief_obj and self.brief_obj.has_v2_schema:
            return await self._review_page_batch(
                search_string, page_num, result_count, page_report,
                all_candidates, string_stats, progress,
            )

        slot_count = await self._card_slot_count_or_raise(
            search_string=search_string,
            page_num=page_num,
            result_count=result_count,
            mode="sequential",
        )
        self._reset_page_observation(
            slot_count,
            search_string=search_string,
            page_num=page_num,
        )
        self._seed_page_pending_full_reviews(
            search_string=search_string,
            page_num=page_num,
        )
        print(f"  Reviewing {slot_count} card slots sequentially", flush=True)

        glance_result = None
        preview_snippets: list[CandidateSnippet] = []
        self._latest_page_preview_snippets = []
        consecutive_api_errors = 0
        consecutive_card_extraction_failures = 0
        page_evaluated = 0
        page_facial_no = 0

        for card_index in range(slot_count):
            snippet = await self._extract_card_snippet(search_string, page_num, card_index)
            extraction_error = getattr(self, "_last_card_extraction_error", None)
            if extraction_error is not None:
                consecutive_card_extraction_failures += 1
                if consecutive_card_extraction_failures >= 5:
                    raise RuntimeError(
                        "5 consecutive card extraction failures on page"
                    ) from extraction_error
            else:
                consecutive_card_extraction_failures = 0
            if not snippet:
                continue

            self._note_page_observation("extracted")
            preview_snippets.append(snippet)
            self._latest_page_preview_snippets = list(preview_snippets)
            print(
                f"    [card {card_index + 1}/{slot_count}] {snippet.name} — "
                f"{snippet.current_title or snippet.headline or 'preview only'}"
            )

            # Drop anyone we cannot durably IDENTIFY, before spending any
            # judgment on them. Sam's call 2026-07-27: an unidentifiable card
            # ("LinkedIn Member" / out-of-network) is an instant drop, not a
            # candidate to reason about — consistent with the high-bar rule that
            # an uncertain candidate is dropped rather than surfaced.
            #
            # The test is the identity FRAGMENT, not a non-empty string. A card
            # can carry an anchor that is not a Recruiter profile link at all
            # ('#', a search URL, a public /in/ link); those satisfied the old
            # `not snippet.profile_url` check, so the candidate was facially
            # judged and possibly fully evaluated — real spend — and then the
            # save aborted the whole run on an identity the save path could
            # never resolve. Same class as everything else this wave: refuse
            # early and cheaply rather than late and expensively.
            if not LinkedInBrowser._profile_url_fragment(snippet.profile_url):
                print(f"    [skip] {snippet.name} — no usable Recruiter identity")
                self._note_page_observation("skipped_missing_url")
                if page_report:
                    page_report.add_skip_preview(snippet.name, "missing_profile_url")
                if self._runtime_bridge and self._runtime_run_id:
                    self._runtime_bridge.record_missing_identity(
                        run_id=self._runtime_run_id,
                        search_string=search_string,
                        snippet=snippet,
                    )
                continue

            pending_facial = self._resume_pending_full_decision(snippet)
            if pending_facial is None:
                print(
                    f"    [defer] {snippet.name} — pending full review belongs "
                    "to another string"
                )
                page_report.add_skip_preview(snippet.name, "foreign_pending")
                continue

            url = snippet.profile_url
            if (
                not pending_facial
                and url
                and (url in self._seen_urls or url in self._in_flight_urls)
            ):
                if url in self._seen_urls:
                    prior = self._prior_outcomes.get(url, "")
                    if prior in SAVE_FAMILY_DECISIONS:
                        print(f"    [dup] {snippet.name} — saved in prior session")
                    elif prior == "REJECT":
                        print(f"    [dup] {snippet.name} — rejected in prior session")
                    elif prior in ("FACIAL_NO", "FACIAL_SKIP"):
                        print(f"    [dup] {snippet.name} — {prior} in prior session")
                    elif prior in ("FACIAL_YES", "FACIAL_BORDERLINE"):
                        print(
                            f"    [re-eval] {snippet.name} — {prior} in prior "
                            "session, completing evaluation"
                        )
                        self._seen_urls.discard(url)
                    else:
                        print(f"    [dup] {snippet.name} — already processed")
                if url in self._seen_urls or url in self._in_flight_urls:
                    page_report.add_skip_preview(snippet.name, "duplicate")
                    self._note_page_observation("skipped_dup")
                    string_stats["duplicates"] += 1
                    # P3.2: bucket prior-session suppressions separately —
                    # they must not feed same-epoch exhaustion accounting.
                    if url in self._prior_session_urls:
                        string_stats["suppressed_prior_session"] = (
                            string_stats.get("suppressed_prior_session", 0) + 1
                        )
                    await self._preview_skip_pause("duplicate")
                    continue

            if snippet.already_saved and not pending_facial:
                print(f"    [skip] {snippet.name} — already saved in LinkedIn pipeline")
                page_report.add_skip_preview(snippet.name, "already_saved")
                string_stats.setdefault("already_saved_skips", 0)
                string_stats["already_saved_skips"] += 1
                await self._preview_skip_pause("already_saved")
                continue

            if consecutive_api_errors >= 5:
                print(f"    [CIRCUIT BREAKER] {consecutive_api_errors} consecutive API failures — pausing 60s")
                log_event(self.log_path, "circuit_breaker", consecutive_errors=consecutive_api_errors)
                await asyncio.sleep(60)
                consecutive_api_errors = 0

            if snippet.profile_url:
                self._in_flight_urls.add(snippet.profile_url)
            if self._record_candidate_funnel_discovery(
                snippet=snippet,
                string_stats=string_stats,
            ):
                self._record_runtime_snippet(search_string, snippet)
                self._record_snippet_activity(snippet, string_stats)

            decision = await self._evaluate_snippet(
                snippet,
                page_report,
                search_string,
                string_stats=string_stats,
            )
            if (
                decision
                and decision.stage == "facial"
                and decision.path == "employer_blacklist"
            ):
                self._note_page_observation("skipped_blacklist")
            else:
                self._note_page_judgment(decision)

            if self._is_recoverable_provider_failure_decision(decision):
                consecutive_api_errors += 1
            else:
                consecutive_api_errors = 0

            outcome = "error"
            if decision:
                self._record_full_funnel_outcome(
                    snippet=snippet,
                    decision=decision,
                    search_string=search_string,
                    string_stats=string_stats,
                )
                if decision.stage == "facial" and decision.decision == "FACIAL_NO":
                    outcome = "facial_no"
                elif decision.stage == "facial" and decision.decision == "FACIAL_SKIP":
                    outcome = "facial_skip"
                    string_stats.setdefault("facial_skip", 0)
                    string_stats["facial_skip"] += 1
                elif decision.decision in SAVE_FAMILY_DECISIONS:
                    # P1.2: a judge SAVE whose side effect physically failed
                    # is a save_failed, not a save — string stats feed
                    # exhaustion, queue scoring, and block reports, so a
                    # broken actuator must not read as productivity.
                    save_outcome = getattr(decision, "save_outcome", None) or {}
                    if save_outcome and not (
                        save_outcome.get("persisted")
                        or save_outcome.get("already_present")
                    ):
                        outcome = "save_failed"
                        string_stats["save_failed"] = (
                            string_stats.get("save_failed", 0) + 1
                        )
                    else:
                        outcome = "save"
                        if not save_outcome.get("already_present") or save_outcome.get(
                            "reconciled_self_save"
                        ):
                            string_stats["saves"] += 1
                            search_string.saves.append(snippet.name)
                            # RC2: capture who the save actually was, so memory
                            # records the discovered pocket, not just the
                            # formation-time family label.
                            if len(search_string.save_exemplars) < 8:
                                search_string.save_exemplars.append({
                                    "title": str(
                                        getattr(snippet, "current_title", "")
                                        or getattr(snippet, "headline", "")
                                        or ""
                                    ),
                                    "company": str(
                                        getattr(snippet, "current_company", "") or ""
                                    ),
                                })
                        self._warn_if_off_geo_save(snippet)
                elif decision.decision == "REJECT":
                    outcome = "reject"
                elif decision.decision in NON_SAVE_REVIEW_DECISIONS:
                    outcome = "review"
                elif decision.stage == "facial" and decision.decision in {
                    "FACIAL_YES",
                    "FACIAL_BORDERLINE",
                }:
                    outcome = (
                        "facial_borderline"
                        if decision.decision == "FACIAL_BORDERLINE"
                        else "facial_yes"
                    )

            all_candidates.append({
                "name": snippet.name,
                "title": snippet.current_title,
                "company": snippet.current_company,
                "headline": snippet.headline,
                "profile_url": snippet.profile_url,
                "outcome": outcome,
                "rationale": decision.rationale if decision else "",
                "page": page_num,
            })

            if outcome == "facial_no":
                page_facial_no += 1
                await self._preview_skip_pause("facial_no")
            elif outcome == "facial_skip":
                await self._preview_skip_pause("facial_skip")

            if outcome in (
                "facial_no",
                "save",
                "reject",
                "review",
                "facial_yes",
                "facial_borderline",
            ):
                page_evaluated += 1

            self._checkpoint_progress(progress, search_string=search_string, page_num=page_num)

            if decision and getattr(decision, "_panel_stuck", False):
                await self._recover_stuck_profile_panel(
                    candidate_name=snippet.name,
                    page_num=page_num,
                    decision=decision,
                    progress=progress,
                    search_string=search_string,
                )

            if glance_result is None and len(preview_snippets) >= config.GLANCE_MIN_SNIPPETS:
                glance_result = self._glance_assess(preview_snippets)
                log_event(
                    self.log_path,
                    "glance_assess",
                    string_id=search_string.id,
                    page=page_num,
                    action=glance_result.action,
                    confidence=glance_result.confidence,
                    signals=glance_result.signals,
                )
                print(
                    f"  [glance] {glance_result.action} ({glance_result.confidence:.2f}): "
                    f"{glance_result.summary}"
                )
                if glance_result.action == "reformulate":
                    print("    [glance] Sequential review found strong reformulation signal — breaking page")
                    self._set_page_break_reason("glance_reformulate")
                    break

            if page_evaluated >= config.EARLY_EXIT_MIN_CANDIDATES:
                page_no_rate = page_facial_no / page_evaluated
                if page_no_rate >= self._get_early_exit_rate():
                    print(
                        f"    [early-exit] {page_facial_no}/{page_evaluated} "
                        f"facial_no ({page_no_rate:.0%}) — breaking page"
                    )
                    log_event(
                        self.log_path,
                        "early_exit",
                        string_id=search_string.id,
                        page=page_num,
                        evaluated=page_evaluated,
                        facial_no=page_facial_no,
                        rate=round(page_no_rate, 2),
                    )
                    self._set_page_break_reason("early_exit")
                    break

            if progress and hasattr(self, "_session_expired") and self._session_expired.is_set():
                self._checkpoint_progress(progress)
                self._set_page_break_reason("session_expired")
                raise SessionExpired("session_duration_cap")
            if progress and hasattr(self, "_operator_stop_event") and self._operator_stop_event.is_set():
                self._checkpoint_progress(progress)
                print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                self._set_page_break_reason("operator_stop")
                raise OperatorStopRequested()

            if progress and hasattr(self, "_pause_requested") and self._pause_requested.is_set():
                self._pause_requested.clear()
                self._checkpoint_progress(progress)
                try:
                    await asyncio.wait_for(self._resume_event.wait(), timeout=300)
                except asyncio.TimeoutError:
                    print("  [!] Resume timeout (5 min) — continuing without decoy burst.")
                    self._resume_event.set()

        return glance_result

    async def _review_page_batch(
        self,
        search_string: SearchString,
        page_num: int,
        result_count: int,
        page_report: "_PageReport",
        all_candidates: list[dict],
        string_stats: dict,
        progress: Progress | None = None,
    ) -> GlanceResult | None:
        """Batch-mode page review for V2 briefs.

        Three phases:
          1. Extract all card snippets (browser interaction, no LLM calls)
          2. Batch facial triage (one LLM call for all eligible snippets)
          3. Full evaluation for FACIAL_YES candidates (sequential profile opens)
        """
        from shared.judger import facial_judge_batch, is_failure_decision

        self._validate_judgment_runtime_configuration()
        slot_count = await self._card_slot_count_or_raise(
            search_string=search_string,
            page_num=page_num,
            result_count=result_count,
            mode="batch",
        )
        self._reset_page_observation(
            slot_count,
            search_string=search_string,
            page_num=page_num,
        )
        self._seed_page_pending_full_reviews(
            search_string=search_string,
            page_num=page_num,
        )
        print(f"  Reviewing {slot_count} card slots (batch mode)", flush=True)

        # ── Phase 1: Extract all card snippets ──────────────────────────
        glance_result = None
        preview_snippets: list[CandidateSnippet] = []
        glance_sample_snippets: list[CandidateSnippet] = []
        self._latest_page_preview_snippets = []
        eligible_snippets: list[CandidateSnippet] = []
        resumed_full_snippets: list[CandidateSnippet] = []
        consecutive_card_extraction_failures = 0

        for card_index in range(slot_count):
            snippet = await self._extract_card_snippet(search_string, page_num, card_index)
            extraction_error = getattr(self, "_last_card_extraction_error", None)
            if extraction_error is not None:
                consecutive_card_extraction_failures += 1
                if consecutive_card_extraction_failures >= 5:
                    raise RuntimeError(
                        "5 consecutive card extraction failures on page"
                    ) from extraction_error
            else:
                consecutive_card_extraction_failures = 0
            if not snippet:
                continue

            self._note_page_observation("extracted")
            preview_snippets.append(snippet)
            self._latest_page_preview_snippets = list(preview_snippets)
            print(
                f"    [card {card_index + 1}/{slot_count}] {snippet.name} — "
                f"{snippet.current_title or snippet.headline or 'preview only'}"
            )

            # Drop anyone we cannot durably IDENTIFY, before spending any
            # judgment on them. Sam's call 2026-07-27: an unidentifiable card
            # ("LinkedIn Member" / out-of-network) is an instant drop, not a
            # candidate to reason about — consistent with the high-bar rule that
            # an uncertain candidate is dropped rather than surfaced.
            #
            # The test is the identity FRAGMENT, not a non-empty string. A card
            # can carry an anchor that is not a Recruiter profile link at all
            # ('#', a search URL, a public /in/ link); those satisfied the old
            # `not snippet.profile_url` check, so the candidate was facially
            # judged and possibly fully evaluated — real spend — and then the
            # save aborted the whole run on an identity the save path could
            # never resolve. Same class as everything else this wave: refuse
            # early and cheaply rather than late and expensively.
            if not LinkedInBrowser._profile_url_fragment(snippet.profile_url):
                print(f"    [skip] {snippet.name} — no usable Recruiter identity")
                self._note_page_observation("skipped_missing_url")
                if page_report:
                    page_report.add_skip_preview(snippet.name, "missing_profile_url")
                if self._runtime_bridge and self._runtime_run_id:
                    self._runtime_bridge.record_missing_identity(
                        run_id=self._runtime_run_id,
                        search_string=search_string,
                        snippet=snippet,
                    )
                continue

            pending_facial = self._resume_pending_full_decision(snippet)
            if pending_facial is None:
                print(
                    f"    [defer] {snippet.name} — pending full review belongs "
                    "to another string"
                )
                page_report.add_skip_preview(snippet.name, "foreign_pending")
                continue
            if pending_facial:
                print(
                    f"    [resume-full] {snippet.name} — {pending_facial} already "
                    "settled; completing full review"
                )
                self._in_flight_urls.add(snippet.profile_url)
                resumed_full_snippets.append(snippet)
                all_candidates.append(
                    {
                        "name": snippet.name,
                        "title": snippet.current_title,
                        "company": snippet.current_company,
                        "headline": snippet.headline,
                        "profile_url": snippet.profile_url,
                        "outcome": (
                            "facial_borderline"
                            if pending_facial == "FACIAL_BORDERLINE"
                            else "facial_yes"
                        ),
                        "rationale": "resumed canonical facial result",
                        "page": page_num,
                    }
                )
                continue

            # Dedup check
            url = snippet.profile_url
            if url and (url in self._seen_urls or url in self._in_flight_urls):
                if url in self._seen_urls:
                    prior = self._prior_outcomes.get(url, "")
                    if prior in SAVE_FAMILY_DECISIONS:
                        print(f"    [dup] {snippet.name} — saved in prior session")
                    elif prior == "REJECT":
                        print(f"    [dup] {snippet.name} — rejected in prior session")
                    elif prior in ("FACIAL_NO", "FACIAL_SKIP"):
                        print(f"    [dup] {snippet.name} — {prior} in prior session")
                    elif prior in ("FACIAL_YES", "FACIAL_BORDERLINE"):
                        print(
                            f"    [re-eval] {snippet.name} — {prior} in prior "
                            "session, completing evaluation"
                        )
                        self._seen_urls.discard(url)
                    else:
                        print(f"    [dup] {snippet.name} — already processed")
                if url in self._seen_urls or url in self._in_flight_urls:
                    page_report.add_skip_preview(snippet.name, "duplicate")
                    self._note_page_observation("skipped_dup")
                    string_stats["duplicates"] += 1
                    # P3.2: bucket prior-session suppressions separately —
                    # they must not feed same-epoch exhaustion accounting.
                    if url in self._prior_session_urls:
                        string_stats["suppressed_prior_session"] = (
                            string_stats.get("suppressed_prior_session", 0) + 1
                        )
                    await self._preview_skip_pause("duplicate")
                    continue

            if snippet.already_saved:
                print(f"    [skip] {snippet.name} — already saved in LinkedIn pipeline")
                page_report.add_skip_preview(snippet.name, "already_saved")
                string_stats.setdefault("already_saved_skips", 0)
                string_stats["already_saved_skips"] += 1
                await self._preview_skip_pause("already_saved")
                continue

            # Employer blacklist check (no LLM call)
            blacklisted = False
            blacklist_match = self._employer_blacklist_match(
                snippet,
                self.brief_obj.employer_blacklist,
            )
            if blacklist_match:
                blocked, matched_field, matched_value = blacklist_match
                print(
                    f"    [BLACKLIST] {snippet.name} — "
                    f"'{matched_value}' matches '{blocked}' ({matched_field})"
                )
                self.stats.setdefault("blacklist_skips", 0)
                self.stats["blacklist_skips"] += 1
                if page_report:
                    page_report.add_skip_preview(snippet.name, f"BLACKLIST: {blocked}")
                bl_decision = OpusDecision(
                    stage="facial", decision="FACIAL_NO", path="employer_blacklist",
                    confidence=1.0, rationale=f"Employer blacklist: {blocked} ({matched_field})",
                    candidate_name=snippet.name, profile_url=snippet.profile_url,
                )
                # Stage writes hard-require a registered candidate row
                # (store does NOT auto-create; the eligible path
                # registers below at its own _record_runtime_snippet).
                # Without this, the first blacklisted card killed its
                # whole string: "candidate not found" (2026-07-06
                # SPL-MM live run — first pool saturated with the
                # hiring company's own contractors).
                self._record_runtime_snippet(search_string, snippet)
                facial_attempt_id = self._start_runtime_stage_attempt(
                    search_string=search_string,
                    snippet=snippet,
                    stage="facial",
                )
                self._finish_runtime_stage_success(
                    attempt_id=facial_attempt_id,
                    stage="facial",
                    snippet=snippet,
                    decision=bl_decision,
                )
                self._prior_outcomes[snippet.profile_url] = "FACIAL_NO"
                self._mark_terminal(snippet.profile_url)
                all_candidates.append({
                    "name": snippet.name, "title": snippet.current_title,
                    "company": snippet.current_company, "headline": snippet.headline,
                    "profile_url": snippet.profile_url,
                    "outcome": "facial_no", "rationale": bl_decision.rationale,
                    "page": page_num,
                })
                self._record_facial_funnel_outcome(
                    snippet=snippet,
                    decision=bl_decision.decision,
                    search_string=search_string,
                    string_stats=string_stats,
                )
                blacklisted = True
                self._note_page_observation("skipped_blacklist")
            if blacklisted:
                continue

            # Eligible for batch facial
            if snippet.profile_url:
                self._in_flight_urls.add(snippet.profile_url)
            if self._record_candidate_funnel_discovery(
                snippet=snippet,
                string_stats=string_stats,
            ):
                self._record_runtime_snippet(search_string, snippet)
                self._record_snippet_activity(snippet, string_stats)
            eligible_snippets.append(snippet)
            glance_sample_snippets.append(snippet)

            # Glance assessment
            if glance_result is None and len(glance_sample_snippets) >= config.GLANCE_MIN_SNIPPETS:
                glance_result = self._glance_assess(glance_sample_snippets)
                log_event(
                    self.log_path, "glance_assess", string_id=search_string.id,
                    page=page_num, action=glance_result.action,
                    confidence=glance_result.confidence, signals=glance_result.signals,
                )
                print(
                    f"  [glance] {glance_result.action} ({glance_result.confidence:.2f}): "
                    f"{glance_result.summary}"
                )

            if progress and hasattr(self, "_session_expired") and self._session_expired.is_set():
                self._checkpoint_progress(progress)
                self._set_page_break_reason("session_expired")
                raise SessionExpired("session_duration_cap")
            if progress and hasattr(self, "_operator_stop_event") and self._operator_stop_event.is_set():
                self._checkpoint_progress(progress)
                print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                self._set_page_break_reason("operator_stop")
                raise OperatorStopRequested()

        if resumed_full_snippets:
            resume_panel_stuck = await self._process_resumed_pending_full_evaluations(
                snippets=resumed_full_snippets,
                page_report=page_report,
                search_string=search_string,
                all_candidates=all_candidates,
                string_stats=string_stats,
                progress=progress,
                page_num=page_num,
            )
            if resume_panel_stuck:
                return glance_result

        if not eligible_snippets:
            return glance_result

        # ── Phase 2: Batch facial triage ────────────────────────────────
        from linkedin.facial_batching import (
            partition_facial_batches,
            run_facial_batches,
        )

        effective_concurrency = (
            config.LINKEDIN_FACIAL_MAX_CONCURRENCY
            if config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED
            else 1
        )
        batches = partition_facial_batches(
            eligible_snippets,
            max_concurrency=effective_concurrency,
            target_batch_size=config.LINKEDIN_FACIAL_TARGET_BATCH_SIZE,
        )
        from shared.llm_spend_budget import FIREWORKS_SPEND_COHORT_CONTEXT_KEY

        spend_cohort = (
            threading.Barrier(len(batches))
            if config.FIREWORKS_PRIMARY_MAX_COST_USD > 0
            and config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED
            and len(batches) > 1
            else None
        )
        batch_call_ids = [f"judge-{uuid.uuid4().hex}" for _batch in batches]
        attempt_payloads: list[dict[str, object]] = [
            {} for _snippet in eligible_snippets
        ]
        for batch, logical_call_id in zip(batches, batch_call_ids):
            batch_payload: dict[str, object] = {
                "logical_call_id": logical_call_id,
                "stage": "facial",
                "page": page_num,
                "dispatch": "page_batch",
                "batch_index": batch.index,
                "batch_number": batch.index + 1,
                "batch_count": len(batches),
                "batch_size": batch.size,
                "batch_slot": batch.index,
                "batch_start": batch.start,
                "batch_stop": batch.stop,
            }
            for position in range(batch.start, batch.stop):
                attempt_payloads[position] = dict(batch_payload)

        # Start canonical attempts BEFORE model dispatch so their measured
        # duration includes the operator-visible GLM wait and a fatal dispatch
        # cannot leave the page with no attempt evidence.
        facial_attempt_ids: list[int | None] = []
        open_facial_attempt_positions: set[int] = set()
        base_batch_context: dict[str, object] = {
            **self._lane_context_for_stage(search_string, stage="facial"),
            "string_id": search_string.id,
            "page": page_num,
            "candidate_count": len(eligible_snippets),
        }
        if spend_cohort is not None:
            base_batch_context[FIREWORKS_SPEND_COHORT_CONTEXT_KEY] = spend_cohort

        def abort_open_page_facial_attempts(
            error: BaseException,
            *,
            failure_marker: str,
            positions: set[int] | None = None,
            force_terminal: bool = False,
        ) -> None:
            """Abort only page-facial attempts that have not already closed."""

            cleanup_error: BaseException | None = None
            target_positions = (
                set(open_facial_attempt_positions)
                if positions is None
                else open_facial_attempt_positions.intersection(positions)
            )
            for position in sorted(target_positions):
                try:
                    payload = {
                        **attempt_payloads[position],
                        failure_marker: True,
                    }
                    if force_terminal:
                        failure = (
                            error
                            if isinstance(error, Exception)
                            else RuntimeError(type(error).__name__)
                        )
                        self._finish_runtime_stage_failure(
                            attempt_id=facial_attempt_ids[position],
                            snippet=eligible_snippets[position],
                            error=failure,
                            stage="facial",
                            payload={
                                **payload,
                                "failure_kind_override": "page_abandoned",
                                "force_terminal": True,
                            },
                        )
                        profile_url = eligible_snippets[position].profile_url
                        self._prior_outcomes[profile_url] = "PAGE_ABANDONED"
                        self._mark_terminal(profile_url)
                    else:
                        self._abort_runtime_stage_attempt(
                            attempt_id=facial_attempt_ids[position],
                            snippet=eligible_snippets[position],
                            error=error,
                            payload=payload,
                        )
                except BaseException as attempt_cleanup_error:
                    # Keep sweeping the rest of the page even if one
                    # canonical close fails. Surface the first cleanup error
                    # after every other attempt and URL has been handled.
                    if cleanup_error is None:
                        cleanup_error = attempt_cleanup_error
                else:
                    open_facial_attempt_positions.discard(position)
                finally:
                    self._in_flight_urls.discard(
                        eligible_snippets[position].profile_url
                    )
            if not force_terminal:
                # A page-level abort must release every URL, including one
                # whose ordinary bookkeeping failed before it could be
                # promoted or discarded from the in-process set.
                for page_snippet in eligible_snippets:
                    self._in_flight_urls.discard(page_snippet.profile_url)
            if cleanup_error is not None:
                raise cleanup_error

        def judge_batch_slice(
            batch_snippets: list[CandidateSnippet],
            batch_context: dict[str, object],
        ) -> list[OpusDecision]:
            batch_index = int(batch_context.get("batch_index", 0))
            batch_context = {
                **batch_context,
                "logical_call_id": batch_call_ids[batch_index],
            }
            try:
                return facial_judge_batch(
                    batch_snippets,
                    brief=self.brief_obj,
                    prompt_prefix=self._tightening_prefix,
                    lane_context=batch_context,
                )
            except BaseException:
                if spend_cohort is not None:
                    try:
                        spend_cohort.abort()
                    except Exception:
                        pass
                raise

        try:
            for position, (snippet, attempt_payload) in enumerate(zip(
                eligible_snippets,
                attempt_payloads,
            )):
                facial_attempt_ids.append(
                    self._start_runtime_stage_attempt(
                        search_string=search_string,
                        snippet=snippet,
                        stage="facial",
                        payload=attempt_payload,
                    )
                )
                open_facial_attempt_positions.add(position)

            facial_dispatch_started = time.monotonic()
            if (
                config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED
                or config.LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED
            ):
                batch_sizes = "/".join(str(batch.size) for batch in batches)
                print(
                    f"  Batch facial triage: {len(eligible_snippets)} candidates "
                    f"in {len(batches)} bounded call(s) ({batch_sizes})",
                    flush=True,
                )
                decisions = await run_facial_batches(
                    eligible_snippets,
                    judge_batch_slice,
                    max_concurrency=effective_concurrency,
                    target_batch_size=config.LINKEDIN_FACIAL_TARGET_BATCH_SIZE,
                    base_context=base_batch_context,
                )
            else:
                print(
                    f"  Batch facial triage: {len(eligible_snippets)} candidates in one call",
                    flush=True,
                )
                serial_context = {
                    **base_batch_context,
                    "batch_index": 0,
                    "batch_number": 1,
                    "batch_count": 1,
                    "batch_size": len(eligible_snippets),
                    "batch_slot": 0,
                    "batch_start": 0,
                    "batch_stop": len(eligible_snippets),
                    "logical_call_id": batch_call_ids[0],
                }
                decisions = judge_batch_slice(eligible_snippets, serial_context)

            if len(decisions) != len(eligible_snippets):
                raise RuntimeError(
                    "facial page dispatch returned "
                    f"{len(decisions)} decisions for {len(eligible_snippets)} candidates"
                )
            failure_outcomes = [
                (position, outcome)
                for position, outcome in enumerate(decisions)
                if isinstance(outcome, FacialBatchFailureOutcome)
            ]
            uncontained = next(
                (
                    outcome
                    for _position, outcome in failure_outcomes
                    if not config.LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED
                    or not _is_containable_facial_page_error(outcome.error)
                ),
                None,
            )
            if uncontained is not None:
                raise uncontained.error
            facial_dispatch_elapsed_ms = round(
                (time.monotonic() - facial_dispatch_started) * 1000.0,
                3,
            )
            timing_payload = {
                "string_id": search_string.id,
                "page": page_num,
                "candidate_count": len(eligible_snippets),
                "batch_count": len(batches),
                "batch_sizes": [batch.size for batch in batches],
                "max_concurrency": effective_concurrency,
                "elapsed_ms": facial_dispatch_elapsed_ms,
            }
            # Measurement is diagnostic and must never change a verdict or
            # strand attempts if its JSONL/canonical event sink is degraded.
            try:
                self._record_runtime_event(
                    search_string=search_string,
                    event_type="facial_page_judgment_timing",
                    payload=timing_payload,
                )
                log_event(
                    self.log_path,
                    "facial_page_judgment_timing",
                    **timing_payload,
                )
            except Exception:
                pass
        except BaseException as exc:
            abort_open_page_facial_attempts(
                exc,
                failure_marker="facial_dispatch_failed",
            )
            self._checkpoint_progress(
                progress,
                search_string=search_string,
                page_num=page_num,
            )
            raise

        page_abandoned_positions: set[int] = set()
        page_abandoned_identities: list[str] = []
        if failure_outcomes:
            try:
                for position, outcome in failure_outcomes:
                    abort_open_page_facial_attempts(
                        outcome.error,
                        failure_marker="page_abandoned",
                        positions={position},
                        force_terminal=True,
                    )
                    page_abandoned_positions.add(position)
                    page_abandoned_identities.append(
                        str(outcome.candidate_identity or "")
                    )
            except BaseException as exc:
                abort_open_page_facial_attempts(
                    exc,
                    failure_marker="facial_postdispatch_failed",
                )
                self._checkpoint_progress(
                    progress,
                    search_string=search_string,
                    page_num=page_num,
                )
                raise

            converted_decisions: list[OpusDecision] = []
            for position, outcome in enumerate(decisions):
                if not isinstance(outcome, FacialBatchFailureOutcome):
                    converted_decisions.append(outcome)
                    continue
                snippet = eligible_snippets[position]
                failure_decision = judgment_failure_decision(
                    stage="facial",
                    candidate_name=snippet.name,
                    profile_url=snippet.profile_url,
                    error=(
                        outcome.error
                        if isinstance(outcome.error, Exception)
                        else RuntimeError(type(outcome.error).__name__)
                    ),
                    path="page_abandoned",
                    source="judgment",
                )
                failure_decision.prompt_capture = {
                    "logical_call_id": batch_call_ids[outcome.batch_index],
                    "page_abandoned": True,
                }
                converted_decisions.append(failure_decision)
            decisions = converted_decisions

        def finalize_page_containment() -> None:
            """Emit the abandonment receipt and restore cursor-N before return."""

            if not page_abandoned_positions:
                return
            try:
                log_event(
                    self.log_path,
                    "page_abandoned",
                    string_id=search_string.id,
                    page=page_num,
                    candidate_identities=page_abandoned_identities,
                )
            finally:
                self._restore_incomplete_page_rollback(search_string)

        try:
            try:
                # Preserve raw borderline observability while also retaining the
                # verdict itself in canonical state below.
                ternary_observable = (
                    self._facial_ternary_enabled()
                    and self._bias_monitor is not None
                )
                for raw_snippet, raw_decision in zip(eligible_snippets, decisions):
                    self._stamp_read_interest(raw_snippet, raw_decision.decision)
                    if (
                        ternary_observable
                        and raw_decision.decision == "FACIAL_BORDERLINE"
                    ):
                        self._bias_monitor.record_facial_borderline_seen(
                            string_id=str(raw_snippet.source_string_id),
                        )
                decisions = [
                    self._normalize_facial_decision_for_persistence(d) for d in decisions
                ]
                contract_corruption_call_ids: set[str] = set()
                if config.LINKEDIN_V2_FACIAL_CONTRACT == "tool":
                    for decision in decisions:
                        if decision.decision != "PARSE_FAILURE":
                            continue
                        logical_call_id = str(
                            (decision.prompt_capture or {}).get("logical_call_id") or ""
                        )
                        contract_corruption_call_ids.add(
                            logical_call_id or f"page-{page_num}-unknown-call"
                        )
            except BaseException as exc:
                abort_open_page_facial_attempts(
                    exc,
                    failure_marker="facial_postdispatch_failed",
                )
                self._checkpoint_progress(
                    progress,
                    search_string=search_string,
                    page_num=page_num,
                )
                raise

            facial_yes_snippets: list[CandidateSnippet] = []
            page_evaluated = 0
            page_facial_no = 0

            for position, (
                snippet,
                facial,
                facial_attempt_id,
                facial_attempt_payload,
            ) in enumerate(zip(
                eligible_snippets,
                decisions,
                facial_attempt_ids,
                attempt_payloads,
            )):
                try:
                    self._note_page_judgment(facial)

                    if position in page_abandoned_positions:
                        self.stats.setdefault("page_abandoned_candidates", 0)
                        self.stats["page_abandoned_candidates"] += 1
                        string_stats.setdefault("page_abandoned_candidates", 0)
                        string_stats["page_abandoned_candidates"] += 1
                        if self._bias_monitor:
                            self._bias_monitor.record_decision(DecisionRecord(
                                candidate_id=(
                                    f"{snippet.source_string_id}_p{snippet.page}_"
                                    f"r{snippet.result_rank}"
                                ),
                                string_id=str(snippet.source_string_id),
                                stage="facial",
                                decision=facial.decision,
                                confidence=facial.confidence,
                                capability_area=None,
                            ))
                        if page_report:
                            page_report.add_skip_preview(
                                snippet.name,
                                f"PAGE_ABANDONED: {facial.rationale}",
                            )
                        all_candidates.append({
                            "name": snippet.name,
                            "title": snippet.current_title,
                            "company": snippet.current_company,
                            "headline": snippet.headline,
                            "profile_url": snippet.profile_url,
                            "outcome": "page_abandoned",
                            "rationale": facial.rationale,
                            "page": page_num,
                        })
                        continue

                    # Handle parse/judgment failures. Close the canonical row
                    # before optional observability or report bookkeeping: either
                    # may raise, but neither may strand this attempt.
                    if is_failure_decision(facial.decision):
                        self._finish_runtime_failure_decision(
                            attempt_id=facial_attempt_id,
                            snippet=snippet,
                            decision=facial,
                            payload=facial_attempt_payload,
                        )
                        open_facial_attempt_positions.discard(position)
                        self._in_flight_urls.discard(snippet.profile_url)
                        print(f"    [PARSE_FAILURE] {snippet.name}: {facial.rationale}")
                        self.stats.setdefault("parse_failures", 0)
                        self.stats["parse_failures"] += 1
                        if self._bias_monitor:
                            self._bias_monitor.record_decision(DecisionRecord(
                                candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                                string_id=str(snippet.source_string_id),
                                stage="facial", decision=facial.decision,
                                confidence=facial.confidence, capability_area=None,
                            ))
                        all_candidates.append({
                            "name": snippet.name, "title": snippet.current_title,
                            "company": snippet.current_company, "headline": snippet.headline,
                            "outcome": "error", "rationale": facial.rationale, "page": page_num,
                        })
                        continue

                    self._finish_runtime_stage_success(
                        attempt_id=facial_attempt_id,
                        stage="facial",
                        snippet=snippet,
                        decision=facial,
                        extra_payload=facial_attempt_payload,
                    )
                    open_facial_attempt_positions.discard(position)
                    self._prior_outcomes[snippet.profile_url] = facial.decision
                    self._mark_terminal(snippet.profile_url)

                    # Bias monitoring
                    if self._bias_monitor:
                        self._bias_monitor.record_decision(DecisionRecord(
                            candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                            string_id=str(snippet.source_string_id),
                            stage="facial", decision=facial.decision,
                            confidence=facial.confidence, capability_area=None,
                        ))

                    if facial.decision in ("FACIAL_NO", "FACIAL_SKIP"):
                        tag = "FACIAL_NO" if facial.decision == "FACIAL_NO" else "FACIAL_SKIP"
                        print(f"    [{tag}] {snippet.name}: {facial.rationale}")
                        if facial.decision == "FACIAL_NO":
                            self._record_facial_funnel_outcome(
                                snippet=snippet,
                                decision=facial.decision,
                                search_string=search_string,
                                string_stats=string_stats,
                            )
                            page_facial_no += 1
                        else:
                            self.stats.setdefault("facial_skip", 0)
                            self.stats["facial_skip"] += 1
                            string_stats.setdefault("facial_skip", 0)
                            string_stats["facial_skip"] += 1
                        if page_report:
                            page_report.add_skip_preview(snippet.name, f"{tag}: {facial.rationale}")
                        all_candidates.append({
                            "name": snippet.name, "title": snippet.current_title,
                            "company": snippet.current_company, "headline": snippet.headline,
                            "profile_url": snippet.profile_url,
                            "outcome": "facial_no" if facial.decision == "FACIAL_NO" else "facial_skip",
                            "rationale": facial.rationale, "page": page_num,
                        })
                        page_evaluated += 1
                    else:
                        # Both ternary positive classes require full review.
                        tag = facial.decision
                        print(f"    [{tag}] {snippet.name}: {facial.rationale}")
                        self._record_facial_funnel_outcome(
                            snippet=snippet,
                            decision=facial.decision,
                            search_string=search_string,
                            string_stats=string_stats,
                        )
                        if self._should_skip_full_eval_for_activity(snippet, facial):
                            self._record_activity_saturation_context(
                                snippet=snippet,
                                search_string=search_string,
                                facial_decision=facial,
                            )
                        facial_yes_snippets.append(snippet)
                        self._track_full_review_obligation(
                            snippet,
                            facial.decision,
                        )
                        self._note_page_full_review_expected(snippet)
                        all_candidates.append({
                            "name": snippet.name, "title": snippet.current_title,
                            "company": snippet.current_company, "headline": snippet.headline,
                            "profile_url": snippet.profile_url,
                            "outcome": (
                                "facial_borderline"
                                if facial.decision == "FACIAL_BORDERLINE"
                                else "facial_yes"
                            ),
                            "rationale": facial.rationale,
                            "page": page_num,
                        })
                        page_evaluated += 1
                except BaseException as exc:
                    abort_open_page_facial_attempts(
                        exc,
                        failure_marker="facial_postdispatch_failed",
                    )
                    self._checkpoint_progress(
                        progress,
                        search_string=search_string,
                        page_num=page_num,
                    )
                    raise

            try:
                if contract_corruption_call_ids:
                    self.stats["facial_contract_corruptions"] += len(
                        contract_corruption_call_ids
                    )
                    if self.stats["facial_contract_corruptions"] >= 2:
                        print(
                            "    [CIRCUIT BREAKER] two facial tool-contract corruptions "
                            "— stopping optimized run",
                            flush=True,
                        )
                        log_event(
                            self.log_path,
                            "circuit_breaker",
                            stage="facial_tool_contract",
                            consecutive_errors=self.stats["facial_contract_corruptions"],
                        )
                        self._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=page_num,
                        )
                        raise RuntimeError(
                            "facial tool-contract corruption threshold reached"
                        )

                if decisions and all(
                    self._is_recoverable_provider_failure_decision(decision)
                    for decision in decisions
                ):
                    recoverable_call_ids = {
                        str((decision.prompt_capture or {}).get("logical_call_id") or "")
                        for decision in decisions
                    }
                    recoverable_call_ids.discard("")
                    self.stats["consecutive_facial_provider_failures"] += max(
                        1,
                        len(recoverable_call_ids),
                    )
                else:
                    self.stats["consecutive_facial_provider_failures"] = 0
                if self.stats["consecutive_facial_provider_failures"] >= 3:
                    print(
                        "    [CIRCUIT BREAKER] repeated facial provider failures "
                        "— stopping run",
                        flush=True,
                    )
                    log_event(
                        self.log_path,
                        "circuit_breaker",
                        stage="facial_provider",
                        consecutive_errors=self.stats[
                            "consecutive_facial_provider_failures"
                        ],
                    )
                    self._checkpoint_progress(
                        progress,
                        search_string=search_string,
                        page_num=page_num,
                    )
                    raise RuntimeError("facial provider failure threshold reached")
            except BaseException as exc:
                abort_open_page_facial_attempts(
                    exc,
                    failure_marker="facial_postdispatch_failed",
                )
                raise

            try:
                # Tightening check (applies to NEXT page — batch processes current page at once).
                # Flag-gated OFF by default (2026-07-30): this was the one bias-monitor
                # path that changed VERDICTS rather than telemetry — it injected
                # "require TWO strong positive signals instead of one" into the facial
                # prompt whenever a string's YES rate ran above 2x a band preflight
                # GUESSED at brief level. Bands are brief-scoped; precision is
                # per-string, so a deliberately dense probe (25 Palantir deployment
                # strategists on one page, 2026-07-30 live) reads as "bias" and the
                # judge is silently made stricter mid-string — punishing exactly the
                # strings that work. The 2026-07-04 telemetry demotion moved the
                # dense-vein-vs-loosened-judge call to the adaptation model via the
                # block report; this gate completes that demotion. Telemetry
                # (facial_rate_anomaly alerts, block-report band, string_context) is
                # unaffected.
                if (
                    config.LINKEDIN_FACIAL_TIGHTENING_ENABLED
                    and not self._triage_tightened
                    and self._bias_monitor
                ):
                    tightening = self._bias_monitor.get_tightening_status(str(snippet.source_string_id))
                    if tightening:
                        self._triage_tightened = True
                        self._tightening_prefix = (
                            f"⚠ TRIAGE TIGHTENING ACTIVE: The facial YES rate on this search string is running "
                            f"{tightening['actual_rate']:.0%}, which is {tightening['multiplier']:.1f}x above the expected "
                            f"maximum of {tightening['expected_high']:.0%}. Apply stricter filtering: require TWO strong "
                            f"positive signals for FACIAL_YES instead of one.\n\n"
                        )
                        print(f"    [bias] Tightening facial criteria for next page "
                              f"(YES rate: {tightening['actual_rate']:.0%}, expected max: {tightening['expected_high']:.0%})")

                # Early exit check (after all batch results are known)
                if page_evaluated >= config.EARLY_EXIT_MIN_CANDIDATES:
                    page_no_rate = page_facial_no / page_evaluated
                    if page_no_rate >= self._get_early_exit_rate():
                        print(
                            f"    [early-exit] {page_facial_no}/{page_evaluated} "
                            f"facial_no ({page_no_rate:.0%}) — stopping pagination "
                            "after queued full reviews"
                        )
                        log_event(
                            self.log_path, "early_exit", string_id=search_string.id,
                            page=page_num, evaluated=page_evaluated,
                            facial_no=page_facial_no, rate=round(page_no_rate, 2),
                        )
                        self._set_page_break_reason("early_exit")

                self._checkpoint_progress(progress, search_string=search_string, page_num=page_num)
            except BaseException as exc:
                abort_open_page_facial_attempts(
                    exc,
                    failure_marker="facial_postdispatch_failed",
                )
                raise

            # ── Phase 3: Full evaluation for FACIAL_YES candidates ──────────
            if config.FULL_EVAL_PIPELINE_ENABLED:
                await self._process_pipelined_full_evaluations(
                    facial_yes_snippets=facial_yes_snippets,
                    page_report=page_report,
                    search_string=search_string,
                    all_candidates=all_candidates,
                    string_stats=string_stats,
                    progress=progress,
                    page_num=page_num,
                )
                return glance_result

            consecutive_api_errors = 0

            for snippet in facial_yes_snippets:
                if progress and hasattr(self, "_session_expired") and self._session_expired.is_set():
                    self._checkpoint_progress(progress)
                    self._set_page_break_reason("session_expired")
                    raise SessionExpired("session_duration_cap")
                if progress and hasattr(self, "_operator_stop_event") and self._operator_stop_event.is_set():
                    self._checkpoint_progress(progress)
                    print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                    self._set_page_break_reason("operator_stop")
                    raise OperatorStopRequested()

                if consecutive_api_errors >= 5:
                    print(f"    [CIRCUIT BREAKER] {consecutive_api_errors} consecutive API failures — pausing 60s")
                    log_event(self.log_path, "circuit_breaker", consecutive_errors=consecutive_api_errors)
                    await asyncio.sleep(60)
                    consecutive_api_errors = 0

                decision = await self._full_evaluate(snippet, page_report, search_string)
                self._clear_resume_pending_full_if_settled(
                    snippet=snippet,
                    decision=decision,
                )

                if self._is_recoverable_provider_failure_decision(decision):
                    consecutive_api_errors += 1
                else:
                    consecutive_api_errors = 0

                # Update the candidate entry in all_candidates with full eval outcome
                if decision:
                    self._record_full_funnel_outcome(
                        snippet=snippet,
                        decision=decision,
                        search_string=search_string,
                        string_stats=string_stats,
                    )
                    if decision.decision in SAVE_FAMILY_DECISIONS:
                        # P1.2: same actuator-truth gate as the sequential loop.
                        save_outcome = getattr(decision, "save_outcome", None) or {}
                        if save_outcome and not (
                            save_outcome.get("persisted")
                            or save_outcome.get("already_present")
                        ):
                            outcome = "save_failed"
                            string_stats["save_failed"] = (
                                string_stats.get("save_failed", 0) + 1
                            )
                        else:
                            outcome = "save"
                            if not save_outcome.get("already_present") or save_outcome.get(
                                "reconciled_self_save"
                            ):
                                string_stats["saves"] += 1
                                search_string.saves.append(snippet.name)
                                # RC2: capture who the save actually was, so memory
                                # records the discovered pocket, not just the
                                # formation-time family label.
                                if len(search_string.save_exemplars) < 8:
                                    search_string.save_exemplars.append({
                                        "title": str(
                                            getattr(snippet, "current_title", "")
                                            or getattr(snippet, "headline", "")
                                            or ""
                                        ),
                                        "company": str(
                                            getattr(snippet, "current_company", "") or ""
                                        ),
                                    })
                            self._warn_if_off_geo_save(snippet)
                    elif decision.decision == "REJECT":
                        outcome = "reject"
                    elif decision.decision in NON_SAVE_REVIEW_DECISIONS:
                        outcome = "review"
                    elif decision.stage == "full" and is_failure_decision(decision.decision):
                        outcome = "error"
                        self._note_page_observation("errored")
                    else:
                        outcome = "facial_yes"

                    for c in all_candidates:
                        if (
                            c.get("profile_url") == snippet.profile_url
                            and c["page"] == page_num
                            and c["outcome"] in {"facial_yes", "facial_borderline"}
                        ):
                            c["outcome"] = outcome
                            c["rationale"] = decision.rationale
                            break

                self._checkpoint_progress(progress, search_string=search_string, page_num=page_num)

                if decision and getattr(decision, "_panel_stuck", False):
                    await self._recover_stuck_profile_panel(
                        candidate_name=snippet.name,
                        page_num=page_num,
                        decision=decision,
                        progress=progress,
                        search_string=search_string,
                    )

                if progress and hasattr(self, "_pause_requested") and self._pause_requested.is_set():
                    self._pause_requested.clear()
                    self._checkpoint_progress(progress)
                    try:
                        await asyncio.wait_for(self._resume_event.wait(), timeout=300)
                    except asyncio.TimeoutError:
                        print("  [!] Resume timeout (5 min) — continuing without decoy burst.")
                        self._resume_event.set()

            return glance_result
        finally:
            finalize_page_containment()

    @staticmethod
    def _fresh_string_stats() -> dict[str, int]:
        return {
            "pages": 0,
            "candidates": 0,
            "duplicates": 0,
            "facial_yes": 0,
            "facial_borderline": 0,
            "facial_no": 0,
            "full_reviewed": 0,
            "full_outreach": 0,
            "full_review": 0,
            "full_reject": 0,
            "saves": 0,
            "rejects": 0,
            "high_pressure_candidates_seen": 0,
            "activity_saturated_preview_skips": 0,
            "high_fit_low_novelty_saves": 0,
        }

    @classmethod
    def _string_stats_for_processing(
        cls,
        search_string: SearchString,
        *,
        resuming: bool,
    ) -> dict[str, int]:
        """Return a cumulative checkpoint baseline for an interrupted string."""

        stats = cls._fresh_string_stats()
        if not resuming:
            return stats
        stats.update(
            {
                "pages": int(search_string.pages_reviewed),
                "candidates": int(search_string.candidates_count),
                "duplicates": int(search_string.duplicates_count),
                "suppressed_prior_session": int(
                    search_string.suppressed_prior_session_count
                ),
                "facial_yes": int(search_string.facial_yes_count),
                "facial_borderline": int(
                    search_string.facial_borderline_count
                ),
                "facial_no": int(search_string.facial_no_count),
                "full_reviewed": int(search_string.full_reviewed_count),
                "full_outreach": int(search_string.full_outreach_count),
                "full_review": int(search_string.full_review_count),
                "full_reject": int(search_string.full_reject_count),
                "saves": len(search_string.saves),
                "rejects": int(search_string.full_reject_count),
            }
        )
        return stats

    def _record_candidate_funnel_discovery(
        self,
        *,
        snippet: CandidateSnippet,
        string_stats: dict[str, int],
    ) -> bool:
        """Count a discovered identity once across a fresh or resumed process."""

        seen = getattr(self, "_candidate_funnel_counted", None)
        if not isinstance(seen, set):
            seen = set()
            self._candidate_funnel_counted = seen
        key = self._funnel_candidate_key(snippet)
        if key in seen:
            return False
        seen.add(key)
        self.stats["snippets_extracted"] = int(
            self.stats.get("snippets_extracted", 0)
        ) + 1
        string_stats["candidates"] = int(string_stats.get("candidates", 0)) + 1
        return True

    def _resume_pending_full_decision(
        self,
        snippet: CandidateSnippet,
    ) -> str | None:
        """Return owned decision, ``None`` for foreign work, or blank if absent."""

        pending = getattr(self, "_resume_pending_full_decisions", None)
        if not isinstance(pending, dict):
            return ""
        key = self._funnel_candidate_key(snippet)
        decision = str(pending.get(key) or "")
        if not decision:
            return ""
        owner_ids = getattr(self, "_resume_pending_full_owner_ids", None)
        if not isinstance(owner_ids, dict) or owner_ids.get(key) is None:
            raise RuntimeError("canonical pending full review lacks owning string")
        if owner_ids[key] != snippet.source_string_id:
            return None
        return decision

    def _validated_owner_pending_full_snippets(
        self,
        *,
        progress: Progress,
        search_string: SearchString,
    ) -> list[tuple[str, CandidateSnippet]]:
        pending = getattr(self, "_resume_pending_full_decisions", None)
        snippets = getattr(self, "_resume_pending_full_snippets", None)
        owner_ids = getattr(self, "_resume_pending_full_owner_ids", None)
        if not isinstance(pending, dict) or not pending:
            return []

        def abort(message: str) -> None:
            search_string.status = "in_progress"
            self._checkpoint_progress(progress, search_string=search_string)
            raise RuntimeError(message)

        if not isinstance(snippets, dict) or not isinstance(owner_ids, dict):
            abort("canonical pending full review metadata is incomplete")
        if any(owner_ids.get(key) is None for key in pending):
            abort("canonical pending full review lacks owning string")
        queue_ids = {item.id for item in progress.strings}
        if any(owner_ids.get(key) not in queue_ids for key in pending):
            abort("canonical pending full review owner is absent from the queue")

        owned: list[tuple[str, CandidateSnippet]] = []
        for key, decision in pending.items():
            if owner_ids.get(key) != search_string.id:
                continue
            snippet = snippets.get(key)
            if snippet is None:
                abort("canonical pending full review lacks stored snippet")
            if snippet.source_string_id != search_string.id:
                abort(
                    "canonical pending full review has conflicting string ownership"
                )
            profile_url = str(snippet.profile_url or "").strip()
            if (
                not profile_url
                or self._funnel_candidate_key(snippet) != key
                or not LinkedInBrowser._profile_url_fragment(profile_url)
            ):
                abort("canonical pending full review lacks exact Recruiter identity")
            if str(decision or "") not in {
                "FACIAL_YES",
                "FACIAL_BORDERLINE",
            }:
                abort("canonical pending full review has invalid facial decision")
            owned.append((key, snippet))
        return sorted(
            owned,
            key=lambda item: (
                max(1, int(item[1].page or 1)),
                max(0, int(item[1].result_rank or 0)),
            ),
        )

    def _seed_page_pending_full_reviews(
        self,
        *,
        search_string: SearchString,
        page_num: int,
    ) -> None:
        pending = getattr(self, "_resume_pending_full_decisions", None)
        snippets = getattr(self, "_resume_pending_full_snippets", None)
        owner_ids = getattr(self, "_resume_pending_full_owner_ids", None)
        if not all(
            isinstance(value, dict)
            for value in (pending, snippets, owner_ids)
        ):
            return
        for key, snippet in snippets.items():
            if (
                key in pending
                and owner_ids.get(key) == search_string.id
                and snippet.source_string_id == search_string.id
                and max(1, int(snippet.page or 1)) == page_num
            ):
                self._note_page_full_review_expected(snippet)

    def _clear_resume_pending_full_if_settled(
        self,
        *,
        snippet: CandidateSnippet,
        decision: OpusDecision | None,
    ) -> None:
        if (
            decision is None
            or decision.stage != "full"
            or is_failure_decision(decision.decision)
        ):
            return
        pending = getattr(self, "_resume_pending_full_decisions", None)
        if isinstance(pending, dict):
            key = self._funnel_candidate_key(snippet)
            pending.pop(key, None)
            pending_snippets = getattr(self, "_resume_pending_full_snippets", None)
            if isinstance(pending_snippets, dict):
                pending_snippets.pop(key, None)
            pending_owner_ids = getattr(
                self,
                "_resume_pending_full_owner_ids",
                None,
            )
            if isinstance(pending_owner_ids, dict):
                pending_owner_ids.pop(key, None)

    # CLO-147: a pending-full candidate whose evaluation keeps failing must be
    # abandoned, not retried forever — each retry re-opens the profile (one
    # governed open, which also makes the session non-absorbable) and a
    # deterministic failure (an oversized profile, a contract wedge) otherwise
    # livelocks every subsequent resume into the same crash. Two non-succeeded
    # full attempts is the signal: the original orphaning plus one recovery
    # retry.
    _PENDING_FULL_RECOVERY_MAX_FAILED_ATTEMPTS = 2

    def _failed_full_attempt_counts(
        self, profile_urls: list[str]
    ) -> dict[str, int]:
        """Non-succeeded full-stage attempts per profile URL across the run chain.

        Resume hydration deliberately reads succeeded attempts only, so a
        candidate whose full evaluations keep FAILING looks eternally pending
        to it. This is the failure-side complement: how many full attempts have
        already started and not succeeded for each candidate — the number of
        evaluations (each usually a governed profile open) the run chain has
        already burned on them.
        """

        urls = sorted({str(url) for url in profile_urls if url})
        if not urls or not self._runtime_run_id:
            return {}
        placeholders = ",".join("?" for _ in urls)
        # Run-level aborts (operator stop, session expiry, governor limit,
        # provider outage) write force_retryable=True — the process's fault,
        # not the candidate's — and are excluded so two unlucky crashes
        # during her evaluation never abandon a healthy candidate (wave-2
        # review finding).
        query = f"""
            WITH RECURSIVE run_chain(id, resumed_from_run_id) AS (
                SELECT id, resumed_from_run_id
                FROM runs
                WHERE id = ? AND source = 'linkedin' AND brief_id = ?
                UNION
                SELECT r.id, r.resumed_from_run_id
                FROM runs r
                JOIN run_chain child ON r.id = child.resumed_from_run_id
                WHERE r.source = 'linkedin' AND r.brief_id = ?
            )
            SELECT c.profile_url, COUNT(*)
            FROM candidate_attempts ca
            JOIN run_chain chain ON chain.id = ca.run_id
            JOIN candidates c ON c.id = ca.candidate_id
            WHERE c.source = 'linkedin' AND c.brief_id = ?
              AND ca.stage = 'full'
              AND ca.status != 'succeeded'
              AND COALESCE(
                  json_extract(ca.payload_json, '$.force_retryable'), 0
              ) != 1
              AND c.profile_url IN ({placeholders})
            GROUP BY c.profile_url
        """
        with self._runtime_state.connect() as conn:
            rows = conn.execute(
                query,
                (
                    self._runtime_run_id,
                    self._brief_id,
                    self._brief_id,
                    self._brief_id,
                    *urls,
                ),
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def _abandon_unrecoverable_pending_full(
        self,
        *,
        key: str,
        snippet: CandidateSnippet,
        search_string: SearchString,
        reason: str,
    ) -> None:
        """Terminally skip a pending full review the live surface cannot re-match.

        LinkedIn reorders Recruiter results between sessions, so a resume can
        fail to relocate the exact card a facial YES/BORDERLINE was captured
        against.  Raising there is deterministic across retries and killed every
        resume attempt of the 2026-07-31 campaign.  Operator ruling: no requeue
        and no retry layer — settle the candidate on canonical state so
        rehydration stops listing it as pending, receipt the skip so it can be
        re-run deliberately, and let the remaining owned snippets recover.

        Settlement rides the same writer the live full-evaluation path uses for
        every non-save terminal full decision: a succeeded ``full`` attempt
        whose payload carries ``full_decision`` is exactly the row
        ``_hydrate_resume_funnel_from_runtime`` reads to decide the review is
        over.  JUDGMENT_FAILURE keeps it out of the reviewed/reject counters
        (nothing was reviewed) and out of ``DEDUP_BLOCKING_DECISIONS``, so a
        later deliberate run may still meet the person.  A pending row with no
        profile URL has no canonical identity to settle against and the writer
        no-ops on it; its containment is in-process plus the receipt.
        """

        abandoned = judgment_failure_decision(
            stage="full",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
            error=RuntimeError(f"pending full recovery abandoned: {reason}"),
            path="pending_full_recovery_abandoned",
            source="judgment",
        )
        abandon_payload = {
            "stage": "full",
            "pending_full_recovery_abandoned": True,
            "abandon_reason": reason,
        }
        abandon_attempt_id = self._start_runtime_stage_attempt(
            search_string=search_string,
            snippet=snippet,
            stage="full",
            payload=abandon_payload,
        )
        self._finish_runtime_stage_success(
            attempt_id=abandon_attempt_id,
            stage="full",
            snippet=snippet,
            decision=abandoned,
            extra_payload=dict(abandon_payload),
        )
        self._record_runtime_event(
            search_string=search_string,
            event_type="pending_full_recovery_abandoned",
            payload={
                "string_id": search_string.id,
                "page": max(1, int(snippet.page or 1)),
                "candidate_name": snippet.name,
                "profile_url": snippet.profile_url,
                "reason": reason,
                # A skip is a negative outcome, not a completion. Say so in the
                # payload so any reader grading receipts by status reads this
                # as failed rather than as a healthy settle.
                "status": "failed",
            },
        )
        for pending_map in (
            getattr(self, "_resume_pending_full_decisions", None),
            getattr(self, "_resume_pending_full_snippets", None),
            getattr(self, "_resume_pending_full_owner_ids", None),
        ):
            if isinstance(pending_map, dict):
                pending_map.pop(key, None)
        print(
            f"    [recover-full] SKIPPED {snippet.name} — {reason}; exact "
            f"string #{search_string.id} page {snippet.page}"
        )

    async def _recover_owner_pending_full_evaluations(
        self,
        *,
        progress: Progress,
        search_string: SearchString,
        first_incomplete_page: int,
        string_stats: dict[str, int],
    ) -> int:
        """Resume older pending reviews only under their owning search surface."""

        pending = getattr(self, "_resume_pending_full_decisions", None)
        owned = [
            (key, snippet)
            for key, snippet in self._validated_owner_pending_full_snippets(
                progress=progress,
                search_string=search_string,
            )
            if int(snippet.page or 1) < first_incomplete_page
        ]
        if not owned:
            return 1

        def contain(
            *,
            key: str,
            snippet: CandidateSnippet,
            reason: str,
        ) -> None:
            """Keep containment inside this function's checkpoint-then-raise rule.

            The helper writes canonical state and emits the receipt; if any of
            that blows up, the string must still be persisted resumable before
            the exception leaves, exactly as every other failure exit here does.
            """

            try:
                self._abandon_unrecoverable_pending_full(
                    key=key,
                    snippet=snippet,
                    search_string=search_string,
                    reason=reason,
                )
            except BaseException:
                search_string.status = "in_progress"
                self._checkpoint_progress(progress)
                raise

        containment_enabled = bool(
            config.LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED
        )
        failed_counts: dict[str, int] = {}
        if containment_enabled:
            # Checked BEFORE any navigation or re-open: a candidate the run
            # chain has already burned the attempt budget on is settled with a
            # receipt instead of spending another governed profile open on a
            # deterministic failure.
            try:
                failed_counts = self._failed_full_attempt_counts(
                    [str(snippet.profile_url or "") for _key, snippet in owned]
                )
            except Exception as count_error:
                # The budget read is advisory; a store hiccup must not become
                # a new resume-killer inside the function that exists to end
                # them (wave-2 review finding). No counts, no abandonment
                # this pass.
                print(
                    "    [recover-full] attempt-budget read failed "
                    f"({type(count_error).__name__}); skipping abandonment "
                    "this pass."
                )
                failed_counts = {}
            surviving: list[tuple[str, CandidateSnippet]] = []
            for key, snippet in owned:
                if (
                    failed_counts.get(str(snippet.profile_url or ""), 0)
                    >= self._PENDING_FULL_RECOVERY_MAX_FAILED_ATTEMPTS
                ):
                    contain(
                        key=key,
                        snippet=snippet,
                        reason="recovery_attempts_exhausted",
                    )
                    continue
                surviving.append((key, snippet))
            owned = surviving
            if not owned:
                return 1

        unsettled: list[str] = []
        rendered_page = 1
        for key, snippet in owned:
            target_page = max(1, int(snippet.page or 1))
            while rendered_page < target_page:
                has_next, transient_suspected = (
                    await self._go_to_next_page_with_transient_retry(
                        result_count=int(search_string.result_count or -1),
                        page_num=rendered_page,
                    )
                )
                if not has_next:
                    search_string.status = "in_progress"
                    self._checkpoint_progress(progress)
                    failure_kind = (
                        "transient pagination"
                        if transient_suspected
                        else "pagination exhaustion"
                    )
                    raise RuntimeError(
                        f"{failure_kind} before pending full review page "
                        f"{target_page}"
                    )
                rendered_page += 1
            if not snippet.profile_url:
                if config.LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED:
                    contain(key=key, snippet=snippet, reason="missing_profile_url")
                    continue
                search_string.status = "in_progress"
                self._checkpoint_progress(progress)
                raise RuntimeError(
                    "pending full recovery requires a stable Recruiter profile URL"
                )
            card_index = await self.browser.find_result_slot_by_profile_url(
                snippet.profile_url
            )
            if card_index is None:
                if config.LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED:
                    contain(key=key, snippet=snippet, reason="unmatched_profile")
                    continue
                search_string.status = "in_progress"
                self._checkpoint_progress(progress)
                raise RuntimeError(
                    "pending full recovery could not match the exact Recruiter profile"
                )
            snippet.card_index = card_index
            print(
                f"    [recover-full] {snippet.name} — exact string "
                f"#{search_string.id} page {snippet.page}"
            )
            self._in_flight_urls.add(snippet.profile_url)

            saved_observation = dict(self._latest_page_observed)
            saved_expected = set(self._allocator_page_expected_keys)
            saved_settled = set(self._allocator_page_settled_keys)
            saved_identity = self._allocator_page_identity
            saved_off_policy = self._allocator_page_off_policy
            try:
                await self._process_resumed_pending_full_evaluations(
                    snippets=[snippet],
                    page_report=None,
                    search_string=search_string,
                    all_candidates=[],
                    string_stats=string_stats,
                    progress=progress,
                    page_num=max(1, int(snippet.page or 1)),
                    preserve_progress_cursor=True,
                )
            except BaseException:
                search_string.status = "in_progress"
                self._checkpoint_progress(progress)
                raise
            finally:
                self._latest_page_observed = saved_observation
                self._allocator_page_expected_keys = saved_expected
                self._allocator_page_settled_keys = saved_settled
                self._allocator_page_identity = saved_identity
                self._allocator_page_off_policy = saved_off_policy

            self._sync_bounded_page_stats_for_checkpoint(
                search_string,
                string_stats,
            )
            self._checkpoint_progress(progress)
            if key in pending:
                # The evaluation completed without settling the canonical
                # pending decision — the just-burned attempt counts against the
                # same budget the pre-check reads, so a repeat wedge is
                # abandoned here instead of raised into a livelock.
                burned = failed_counts.get(str(snippet.profile_url or ""), 0) + 1
                if (
                    containment_enabled
                    and burned >= self._PENDING_FULL_RECOVERY_MAX_FAILED_ATTEMPTS
                ):
                    contain(
                        key=key,
                        snippet=snippet,
                        reason="recovery_attempts_exhausted",
                    )
                else:
                    unsettled.append(snippet.name or snippet.profile_url)

        if unsettled:
            search_string.status = "in_progress"
            self._checkpoint_progress(progress)
            raise RuntimeError(
                "canonical pending full review remains unsettled for active owner"
            )
        return rendered_page

    @staticmethod
    def _page_stat_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        keys = (
            "candidates",
            "duplicates",
            "facial_yes",
            "facial_borderline",
            "facial_no",
            "full_reviewed",
            "full_outreach",
            "full_review",
            "full_reject",
            "saves",
            "rejects",
            "high_pressure_candidates_seen",
            "activity_saturated_preview_skips",
            "high_fit_low_novelty_saves",
        )
        return {key: max(0, int(after.get(key, 0)) - int(before.get(key, 0))) for key in keys}

    @staticmethod
    def _funnel_candidate_key(snippet: CandidateSnippet) -> str:
        """Stable identity for exactly-once in-process funnel accounting."""

        return str(snippet.profile_url or "").strip() or (
            f"{snippet.source_string_id}:{snippet.page}:{snippet.result_rank}:"
            f"{snippet.name}"
        )

    def _record_facial_funnel_outcome(
        self,
        *,
        snippet: CandidateSnippet,
        decision: str,
        search_string: SearchString,
        string_stats: dict[str, int] | None,
    ) -> None:
        """Count one canonical facial verdict per candidate for this run.

        Facial opens are stage-boundary metrics. They are never reconstructed
        from later full outcomes, which previously double-counted retries and
        erased the distinction between YES and BORDERLINE. The first retrieval
        string owns attribution if the same profile URL resurfaces later.
        """

        counter_by_decision = {
            "FACIAL_YES": "facial_yes",
            "FACIAL_BORDERLINE": "facial_borderline",
            "FACIAL_NO": "facial_no",
        }
        counter = counter_by_decision.get(decision)
        if counter is None:
            return
        seen = getattr(self, "_facial_funnel_counted", None)
        if not isinstance(seen, set):
            seen = set()
            self._facial_funnel_counted = seen
        key = self._funnel_candidate_key(snippet)
        if key in seen:
            return
        seen.add(key)
        self.stats[counter] = int(self.stats.get(counter, 0)) + 1
        if string_stats is not None:
            string_stats[counter] = int(string_stats.get(counter, 0)) + 1

    def _record_full_funnel_outcome(
        self,
        *,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        search_string: SearchString,
        string_stats: dict[str, int],
    ) -> None:
        """Count one settled full-profile disposition per candidate for this run.

        SAVE-family means the evidence is strong enough for outreach even if
        the later LinkedIn click fails. Human REVIEW is weak positive signal;
        REJECT is negative signal. Provider/parse failures are not settled.
        """

        if decision.stage != "full" or is_failure_decision(decision.decision):
            return
        if (
            decision.decision not in SAVE_FAMILY_DECISIONS
            and decision.decision not in NON_SAVE_REVIEW_DECISIONS
            and decision.decision != "REJECT"
        ):
            return
        seen = getattr(self, "_full_funnel_counted", None)
        if not isinstance(seen, set):
            seen = set()
            self._full_funnel_counted = seen
        key = self._funnel_candidate_key(snippet)
        if key in seen:
            return
        seen.add(key)

        self.stats["full_reviewed"] = int(self.stats.get("full_reviewed", 0)) + 1
        string_stats["full_reviewed"] = int(
            string_stats.get("full_reviewed", 0)
        ) + 1
        if decision.decision in SAVE_FAMILY_DECISIONS:
            self.stats["full_outreach"] = int(self.stats.get("full_outreach", 0)) + 1
            string_stats["full_outreach"] = int(
                string_stats.get("full_outreach", 0)
            ) + 1
        elif decision.decision in NON_SAVE_REVIEW_DECISIONS:
            self.stats["full_review"] = int(self.stats.get("full_review", 0)) + 1
            string_stats["full_review"] = int(string_stats.get("full_review", 0)) + 1
        else:
            self.stats["full_reject"] = int(self.stats.get("full_reject", 0)) + 1
            string_stats["full_reject"] = int(string_stats.get("full_reject", 0)) + 1
            # Compatibility alias for existing reports while readers migrate.
            string_stats["rejects"] = int(string_stats.get("rejects", 0)) + 1

    @staticmethod
    def _remaining_queued_strings(progress: Progress, *, current_string_id: int) -> int:
        return sum(
            1
            for string in progress.strings
            if string.id != current_string_id
            and string.status not in {"done", "skipped"}
        )

    def _initial_search_mode(self, result_count: int) -> str:
        return "recon" if result_count >= 500 else "paginate"

    @staticmethod
    def _page_candidates(all_candidates: list[dict], page_num: int) -> list[dict]:
        return [candidate for candidate in all_candidates if candidate.get("page") == page_num]

    @staticmethod
    def _activity_snapshot_from_sources(
        snippet: CandidateSnippet,
        profile_status_summary: dict | None = None,
    ) -> RecruiterActivitySnapshot | None:
        if isinstance(profile_status_summary, dict) and profile_status_summary:
            return RecruiterActivitySnapshot.from_dict(profile_status_summary)
        return snippet.recruiter_activity

    @staticmethod
    def _activity_summary_text(activity: RecruiterActivitySnapshot | None) -> str:
        if not activity:
            return ""
        parts: list[str] = []
        if activity.message_count:
            parts.append(f"{activity.message_count} messages")
        if activity.project_count:
            parts.append(f"{activity.project_count} projects")
        if activity.view_count:
            parts.append(f"{activity.view_count} views")
        if activity.last_outbound_contact:
            parts.append(f"last outbound {activity.last_outbound_contact}")
        if activity.saved_by:
            parts.append(f"saved by {activity.saved_by}")
        return ", ".join(parts)

    def _snippet_is_clearly_compelling(self, snippet: CandidateSnippet) -> bool:
        """Conservative heuristic for candidates worth opening despite high saturation.

        Wave 3 slice 14 (P1 discharge): the AI/ML vocabulary moved out.
        Structural seniority markers stay in code; role-specific vocabulary
        comes from the brief's calibration mirrors (canonical title/framework
        patterns + per-area key terms) via ``_brief_compelling_terms``. A
        brief carrying no mirrors gets structural signals only — fewer forced
        opens, never a built-in vertical prior.
        """
        text = " ".join(
            part
            for part in [
                snippet.current_title,
                snippet.headline,
                " ".join(snippet.experience_entries[:3]),
            ]
            if part
        ).lower()
        if not text:
            return False

        structural_signals = (
            "executive director",
            "chief architect",
            "principal architect",
            "technology fellow",
            "distinguished engineer",
            "principal scientist",
            "research director",
            "research manager",
        )
        if any(signal in text for signal in structural_signals):
            return True

        brief_terms = self._brief_compelling_terms()
        if not brief_terms:
            return False
        leadership_terms = ("head", "chief", "director", "vp", "vice president", "principal", "fellow", "architect")
        return any(term in text for term in brief_terms) and any(
            term in text for term in leadership_terms
        )

    def _brief_compelling_terms(self) -> tuple[str, ...]:
        """Brief-supplied vocabulary the compellingness heuristic may consume.

        Mock-safe by design: non-list attributes (MagicMock briefs in tests,
        legacy briefs without mirrors) contribute nothing.
        """
        terms: list[str] = []
        brief = self.brief_obj
        for attr in (
            "canonical_title_patterns",
            "canonical_framework_patterns",
            "canonical_broad_patterns",
        ):
            values = getattr(brief, attr, None)
            if isinstance(values, (list, tuple)):
                terms.extend(
                    str(v).strip().lower() for v in values if str(v or "").strip()
                )
        key_terms = getattr(brief, "key_terms_by_area", None)
        if isinstance(key_terms, dict):
            for area_terms in key_terms.values():
                if isinstance(area_terms, (list, tuple)):
                    terms.extend(
                        str(v).strip().lower()
                        for v in area_terms
                        if str(v or "").strip()
                    )
        return tuple(dict.fromkeys(terms))

    def _warn_if_off_geo_save(self, snippet: CandidateSnippet) -> None:
        return self._geography_service._warn_if_off_geo_save(snippet)

    def _candidate_location_contained(
        self, location: str, geo_values: list[str]
    ) -> bool | None:
        return self._geography_service._candidate_location_contained(location, geo_values)

    def _record_snippet_activity(
        self,
        snippet: CandidateSnippet,
        string_stats: dict[str, int],
    ) -> None:
        pressure = snippet.novelty_pressure or classify_recruiter_activity_pressure(
            snippet.recruiter_activity
        )
        snippet.novelty_pressure = pressure
        if pressure == "high":
            self.stats["high_pressure_candidates_seen"] += 1
            string_stats.setdefault("high_pressure_candidates_seen", 0)
            string_stats["high_pressure_candidates_seen"] += 1

    def _should_skip_full_eval_for_activity(
        self,
        snippet: CandidateSnippet,
        facial_decision: OpusDecision,
    ) -> bool:
        """Whether high recruiter activity merits a low-novelty annotation.

        P5.4: this used to gate on ``facial_decision.confidence >= 0.85``,
        but the V2 facial contract has no real confidence signal — every
        valid V2 facial verdict was fabricated to 1.0 (shared/judger.py),
        which made this branch always true and permanently blocked the
        rationale-pattern checks below it. The confidence gate is removed;
        the decision now rests entirely on the signals this function
        actually has: novelty pressure, the facial rationale text, and the
        snippet-compellingness heuristic.
        """
        if (snippet.novelty_pressure or "").lower() != "high":
            return False
        rationale = (facial_decision.rationale or "").lower()
        if "strong" in rationale or "clear" in rationale:
            return False
        return not self._snippet_is_clearly_compelling(snippet)

    def _record_activity_saturation_context(
        self,
        *,
        snippet: CandidateSnippet,
        search_string: SearchString,
        facial_decision: OpusDecision,
    ) -> None:
        """Annotate recruiter saturation without suppressing full review."""

        activity = self._activity_snapshot_from_sources(snippet)
        activity_summary = self._activity_summary_text(activity) or "high recruiter activity"
        facial_decision.novelty_value = "low"
        facial_decision.value_rationale = f"Low novelty due to recruiter saturation: {activity_summary}"
        log_event(
            self.log_path,
            "activity_saturation_context",
            string_id=search_string.id,
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
            novelty_pressure=snippet.novelty_pressure,
            activity_summary=activity_summary,
            facial_confidence=facial_decision.confidence,
        )

    def _derive_novelty_value(
        self,
        snippet: CandidateSnippet,
        *,
        profile_status_summary: dict | None = None,
    ) -> tuple[str, str]:
        activity = self._activity_snapshot_from_sources(snippet, profile_status_summary)
        pressure = classify_recruiter_activity_pressure(activity)
        reachout_status = infer_reachout_status(activity)
        summary = self._activity_summary_text(activity)
        if pressure == "high":
            return "low", f"Low novelty due to visible recruiter saturation ({summary or reachout_status or 'high activity'})."
        if pressure == "medium":
            return "medium", f"Moderate novelty because the candidate already shows recruiter activity ({summary or reachout_status or 'some activity'})."
        return "high", "High novelty: limited visible recruiter activity."

    def _build_page_insights(
        self,
        *,
        page_num: int,
        result_count: int,
        preview_snippets: list[CandidateSnippet],
        all_candidates: list[dict],
        glance_result: GlanceResult | None,
    ) -> LinkedInPageInsights:
        title_counts: dict[str, int] = {}
        company_counts: dict[str, int] = {}
        for snippet in preview_snippets:
            title = _normalize_title_family(snippet.current_title or snippet.headline or "")
            if title:
                title_counts[title] = title_counts.get(title, 0) + 1
            company = (snippet.current_company or "").strip()
            if company:
                company_counts[company] = company_counts.get(company, 0) + 1

        page_candidates = self._page_candidates(all_candidates, page_num)
        title_signal_counts: dict[str, int] = {}
        company_signal_counts: dict[str, int] = {}
        settled_positive_outcomes = {"save", "save_failed", "review"}
        for candidate in page_candidates:
            if candidate.get("outcome") not in settled_positive_outcomes:
                continue
            title = _normalize_title_family(str(candidate.get("title") or ""))
            if title:
                title_signal_counts[title] = title_signal_counts.get(title, 0) + 1
            company = self._normalize_company_rail_label(str(candidate.get("company") or ""))
            if company:
                company_signal_counts[company] = company_signal_counts.get(company, 0) + 1
        signal_anchors = [
            f"{candidate.get('title') or 'Unknown'} at {candidate.get('company') or 'Unknown'}"
            for candidate in page_candidates
            if candidate.get("outcome") in settled_positive_outcomes
        ][:5]
        noise_anchors = [
            f"{candidate.get('title') or 'Unknown'} at {candidate.get('company') or 'Unknown'}"
            for candidate in page_candidates
            if candidate.get("outcome") in {"facial_no", "reject"}
        ][:5]
        dominant_non_fit_patterns: list[str] = []
        if glance_result and glance_result.action == "reformulate":
            dominant_non_fit_patterns.append(glance_result.summary)
        if noise_anchors and not dominant_non_fit_patterns:
            dominant_non_fit_patterns.append("preview signal dominated by facial-no candidates")

        window = result_window_for_count(result_count)
        result_window = f"{window[0]}-{window[1]}" if window else "direct_paginate"
        return LinkedInPageInsights(
            page=page_num,
            result_count=result_count,
            result_window=result_window,
            title_clusters=[
                {
                    "label": label,
                    "title": label,
                    "count": count,
                    "signal_count": title_signal_counts.get(label, 0),
                }
                for label, count in sorted(title_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
            ],
            company_clusters=[
                {
                    "label": label,
                    "company": label,
                    "count": count,
                    "signal_count": company_signal_counts.get(
                        self._normalize_company_rail_label(label),
                        0,
                    ),
                }
                for label, count in sorted(company_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
            ],
            signal_anchors=signal_anchors,
            noise_anchors=noise_anchors,
            dominant_non_fit_patterns=dominant_non_fit_patterns,
            glance_action=glance_result.action if glance_result else "",
            glance_summary=glance_result.summary if glance_result else "",
        )

    @staticmethod
    def _normalize_company_rail_label(company: str) -> str:
        return " ".join(str(company or "").strip().lower().split())

    @staticmethod
    def _cluster_signal_count(cluster: dict[str, Any]) -> int:
        try:
            return int(cluster.get("signal_count", 0) or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _thresholded_structured_promotions(
        cls,
        insights: Any | None,
    ) -> tuple[set[str], set[str]]:
        title_clusters = getattr(insights, "title_clusters", []) or []
        company_clusters = getattr(insights, "company_clusters", []) or []
        title_labels = {
            _normalize_title_family(
                str(cluster.get("label") or cluster.get("title") or "")
            )
            for cluster in title_clusters
            if isinstance(cluster, dict) and cls._cluster_signal_count(cluster) > 0
        }
        company_labels = {
            cls._normalize_company_rail_label(
                str(cluster.get("label") or cluster.get("company") or "")
            )
            for cluster in company_clusters
            if isinstance(cluster, dict) and cls._cluster_signal_count(cluster) > 0
        }
        return (
            {label for label in title_labels if label},
            {label for label in company_labels if label},
        )

    @staticmethod
    def _scout_metrics(
        *,
        page_stats: dict[str, int],
        page_insights: LinkedInPageInsights,
    ) -> dict[str, object]:
        full_outreach = int(
            page_stats.get("full_outreach", page_stats.get("saves", 0))
        )
        full_review = int(page_stats.get("full_review", 0))
        full_reject = int(
            page_stats.get("full_reject", page_stats.get("rejects", 0))
        )
        full_reviewed = int(
            page_stats.get(
                "full_reviewed",
                full_outreach + full_review + full_reject,
            )
        )
        facial_no = int(page_stats.get("facial_no", 0))
        page_signal = full_outreach + full_review
        noise_dominant = any(
            (
                page_insights.glance_action == "reformulate",
                bool(page_insights.dominant_non_fit_patterns),
                facial_no + full_reject >= max(8, page_signal * 2),
                len(page_insights.noise_anchors) >= len(page_insights.signal_anchors) + 2,
            )
        )
        # A SAVE-family full judgment is strong directional signal. Human
        # REVIEW is weak signal. Full rejects are negative, never productive.
        real_signal = full_outreach > 0
        weak_but_real_signal = full_review > 0 and not real_signal
        return {
            "page_signal": page_signal,
            "full_reviewed": full_reviewed,
            "full_outreach": full_outreach,
            "full_review": full_review,
            "full_reject": full_reject,
            # Compatibility telemetry only; neither field drives signal.
            "saves": int(page_stats.get("saves", 0)),
            "facial_yes": int(page_stats.get("facial_yes", 0)),
            "rejects": full_reject,
            "facial_no": facial_no,
            "noise_dominant": noise_dominant,
            "real_signal": real_signal,
            "strong_scout_signal": real_signal,
            "weak_but_real_signal": weak_but_real_signal,
        }

    @staticmethod
    def _scout_quality_for_recon(
        *,
        result_count: int,
        scout_metrics: dict[str, object],
        precommit_recovery_attempts_used: int,
    ) -> tuple[str, str, str]:
        page_signal = int(scout_metrics["page_signal"])
        noise_dominant = bool(scout_metrics["noise_dominant"])
        real_signal = bool(scout_metrics["real_signal"])
        weak_but_real_signal = bool(scout_metrics["weak_but_real_signal"])
        recovery_budget_remaining = max(
            0,
            config.PRECOMMIT_MAX_RECOVERY_ATTEMPTS - precommit_recovery_attempts_used,
        )

        if result_count < 500:
            if page_signal == 0 and noise_dominant:
                return (
                    "stop",
                    "small pool is already dead/noisy on page 1, so it should stop rather than spend rescue budget",
                    "small_pool_dead_stop",
                )
            return (
                "commit",
                "result pool is already small enough that direct pagination is cheaper than pre-commit rescue branching",
                "small_pool_direct",
            )
        if result_count < 5000:
            if page_signal > 0:
                return (
                    "commit",
                    "mid-sized pool already shows enough directional scout signal to paginate without pre-commit rescue branching",
                    "mid_pool_signal_commit",
                )
            if noise_dominant and recovery_budget_remaining > 0:
                return (
                    "experiment",
                    "mid-sized pool is dead/noisy on the scout page, so one bounded rescue attempt is justified before stop",
                    "mid_pool_dead_noisy_recovery",
                )
            return (
                "stop",
                "mid-sized pool is not surfacing signal and lacks a coherent scout-time rescue path, so stop cleanly",
                "mid_pool_unsalvageable_stop",
            )

        if page_signal == 0 and noise_dominant:
            if recovery_budget_remaining > 0:
                return (
                    "experiment",
                    "large/noisy scout page is dead, so spend a bounded pre-commit rescue attempt before abandoning the family",
                    "precommit_dead_noisy_recovery",
                )
            return (
                "stop",
                "pre-commit rescue budget is exhausted and the large pool is still dead/noisy, so stop instead of paying more pagination tax",
                "precommit_recovery_exhausted_stop",
            )
        if real_signal and noise_dominant:
            if recovery_budget_remaining > 0:
                return (
                    "experiment",
                    "large pool shows real signal, but dominant noise means it should get one bounded rescue attempt before direct pagination",
                    "precommit_real_signal_noisy_recovery",
                )
            return (
                "commit",
                "large pool still shows real signal and rescue budget is exhausted, so fall back to direct pagination rather than stopping a live lane",
                "precommit_real_signal_budget_exhausted_commit",
            )
        if real_signal:
            return (
                "commit",
                "root scout already shows real signal without dominant noise, so it has earned direct pagination",
                "root_real_signal_commit",
            )
        if weak_but_real_signal:
            if recovery_budget_remaining > 0:
                return (
                    "experiment",
                    "scout page has only weak signal, so it should survive a bounded rescue attempt before the root query earns pagination",
                    "precommit_weak_signal_recovery",
                )
            return (
                "stop",
                "pre-commit rescue budget is exhausted and the large pool still only shows weak scout signal, so stop instead of paying more pagination tax",
                "precommit_recovery_exhausted_stop",
            )
        return (
            "stop",
            "large pool is not surfacing signal and the scout did not produce a coherent rescue path, so stop cleanly",
            "precommit_unsalvageable_stop",
        )

    @staticmethod
    def _variant_has_earned_commit(variant: LinkedInSearchVariant) -> bool:
        # Facial positives only justify opening a profile. A variant earns
        # pagination after at least one settled full judgment clears outreach.
        return int(getattr(variant, "full_outreach", 0) or 0) > 0

    async def _assess_string_state(
        self,
        *,
        search_string: SearchString,
        experiment_state: LinkedInExperimentState,
        page_num: int,
        result_count: int,
        string_stats: dict[str, int],
        page_stats: dict[str, int],
        page_insights: LinkedInPageInsights,
        remaining_queued_strings: int,
    ) -> dict[str, object]:
        scout_metrics = self._scout_metrics(page_stats=page_stats, page_insights=page_insights)
        page_signal = int(scout_metrics["page_signal"])
        active_variant = experiment_state.active_variant
        scout_gate_bucket = ""
        real_signal = bool(scout_metrics["real_signal"])
        post_drift_stop_limit = (
            config.POST_DRIFT_ZERO_SIGNAL_STOP_STREAK
            if experiment_state.last_drift_refinement_summary.get("outcome") == "not_rescued"
            else config.COMMITTED_ZERO_SIGNAL_STOP_STREAK
        )

        if experiment_state.mode == "recon":
            decision, rationale, scout_gate_bucket = self._scout_quality_for_recon(
                result_count=result_count,
                scout_metrics=scout_metrics,
                precommit_recovery_attempts_used=experiment_state.precommit_recovery_attempts_used,
            )
        elif experiment_state.mode == "experiment":
            best_variant = experiment_state.best_variant()
            variant_earned_commit = self._variant_has_earned_commit(active_variant)
            best_variant_earned_commit = self._variant_has_earned_commit(best_variant)
            if variant_earned_commit:
                decision = "commit"
                rationale = "active rescue variant surfaced enough real signal density to earn commit"
            elif (
                experiment_state.executed_sibling_count < config.SEARCH_EXPERIMENT_MAX_EXECUTED_SIBLINGS
                and experiment_state.next_planned_variant() is not None
            ):
                decision = "experiment"
                rationale = "current rescue variant did not earn commit, but another planned sibling is still available"
            elif best_variant.variant_id != "root" and best_variant_earned_commit:
                decision = "commit"
                rationale = "best explored sibling earned commit and should be promoted over the root path"
            else:
                decision = "stop"
                rationale = "pre-commit rescue attempts are exhausted and no explored sibling earned commit"
        elif experiment_state.mode == "drift":
            if self._variant_has_earned_commit(active_variant):
                decision = "commit"
                rationale = "drift rescue recovered a higher-signal slice and should become the new committed path"
                experiment_state.last_drift_refinement_summary = {
                    **experiment_state.last_drift_refinement_summary,
                    "outcome": "rescued",
                    "reviewed_page": page_num,
                }
            else:
                decision = "resume_committed"
                rationale = "drift rescue did not recover signal; resume the committed path and stop after one more zero-signal page unless signal returns"
                experiment_state.last_drift_refinement_summary = {
                    **experiment_state.last_drift_refinement_summary,
                    "outcome": "not_rescued",
                    "reviewed_page": page_num,
                }
        else:
            drift_assessment = self._assess_pagination_drift(
                experiment_state=experiment_state,
                page_num=page_num,
                result_count=result_count,
                page_stats=page_stats,
                page_insights=page_insights,
                remaining_queued_strings=remaining_queued_strings,
            )
            if drift_assessment.decision in {"refine_committed", "spawn_recall_sibling"}:
                decision = drift_assessment.decision
                rationale = drift_assessment.rationale
                experiment_state.last_drift_refinement_summary = {
                    **experiment_state.last_drift_refinement_summary,
                    **drift_assessment.to_dict(),
                    "parent_variant_id": experiment_state.committed_variant_id,
                    "page": page_num,
                    "result_count": result_count,
                    "outcome": "pending",
                }
            elif (
                experiment_state.committed_pages_reviewed >= 1
                and experiment_state.committed_zero_signal_streak >= post_drift_stop_limit
                and page_signal == 0
            ):
                decision = "stop"
                if post_drift_stop_limit == config.POST_DRIFT_ZERO_SIGNAL_STOP_STREAK:
                    rationale = "committed path stayed empty after a failed drift rescue, so stop after the first subsequent zero-signal page"
                else:
                    rationale = "committed path has hit the zero-signal decay limit, so stop instead of paying more pagination tax"
            else:
                decision = "continue"
                rationale = "keep paginating the committed variant"

        payload = {
            "decision": decision,
            "rationale": rationale,
            "page": page_num,
            "result_count": result_count,
            "mode": experiment_state.mode,
            "active_variant_id": experiment_state.active_variant_id,
            "remaining_queued_strings": remaining_queued_strings,
            "page_signal": scout_metrics["page_signal"],
            "real_signal": real_signal,
            "full_reviewed": scout_metrics["full_reviewed"],
            "full_outreach": scout_metrics["full_outreach"],
            "full_review": scout_metrics["full_review"],
            "full_reject": scout_metrics["full_reject"],
            "saves": scout_metrics["saves"],
            "facial_yes": scout_metrics["facial_yes"],
            "rejects": scout_metrics["rejects"],
            "facial_no": scout_metrics["facial_no"],
            "noise_dominant": scout_metrics["noise_dominant"],
            "strong_scout_signal": scout_metrics["strong_scout_signal"],
            "scout_gate_bucket": scout_gate_bucket,
            "precommit_recovery_attempts_used": experiment_state.precommit_recovery_attempts_used,
            "committed_pages_reviewed": experiment_state.committed_pages_reviewed,
            "committed_zero_signal_streak": experiment_state.committed_zero_signal_streak,
            "page_observed": self._page_observation(),
        }
        if decision == "commit" and page_signal > 0:
            payload["bootstrap_early_snapshot"] = True
        if decision in {"refine_committed", "spawn_recall_sibling"}:
            payload["drift_rescue_summary"] = dict(experiment_state.last_drift_refinement_summary)
        log_event(self.log_path, "linkedin_search_assess", string_id=search_string.id, **payload)
        self._record_runtime_event(
            search_string=search_string,
            event_type="linkedin_search_assess",
            payload=payload,
        )
        return payload

    def _assess_pagination_drift(
        self,
        *,
        experiment_state: LinkedInExperimentState,
        page_num: int,
        result_count: int,
        page_stats: dict[str, int],
        page_insights: LinkedInPageInsights,
        remaining_queued_strings: int,
    ) -> LinkedInDriftAssessment:
        if experiment_state.committed_variant_id is None:
            return LinkedInDriftAssessment(decision="continue", rationale="", eligible=False)
        if experiment_state.active_variant_id != experiment_state.committed_variant_id:
            return LinkedInDriftAssessment(decision="continue", rationale="", eligible=False)
        if experiment_state.drift_attempt_count >= config.SEARCH_EXPERIMENT_DRIFT_BUDGET:
            return LinkedInDriftAssessment(
                decision="continue",
                rationale="drift budget already used for this committed variant",
                eligible=False,
            )
        if experiment_state.pages_since_last_mutation < 1:
            return LinkedInDriftAssessment(
                decision="continue",
                rationale="need at least one fully reviewed page between search mutations",
                eligible=False,
            )
        active_variant = experiment_state.active_variant
        if active_variant.pages_reviewed < 2 or page_num < 2 or result_count < 500:
            return LinkedInDriftAssessment(decision="continue", rationale="", eligible=False)
        if not experiment_state.real_signal_seen():
            return LinkedInDriftAssessment(
                decision="continue",
                rationale="committed variant has not shown enough real signal to justify rescue",
                eligible=False,
            )

        page_signal = int(
            page_stats.get("full_outreach", page_stats.get("saves", 0))
        ) + int(page_stats.get("full_review", 0))
        noisy_page = page_signal == 0 and (
            page_insights.glance_action == "reformulate"
            or len(page_insights.noise_anchors) >= max(2, len(page_insights.signal_anchors) + 1)
            or bool(page_insights.dominant_non_fit_patterns)
        )
        if not noisy_page:
            return LinkedInDriftAssessment(decision="continue", rationale="", eligible=False)

        opportunity_cost_ok = (
            experiment_state.family_outreach_total > 0
            or remaining_queued_strings <= 3
        )
        if not opportunity_cost_ok:
            return LinkedInDriftAssessment(
                decision="continue",
                rationale="other untouched families still have higher expected value than rescuing this one",
                eligible=False,
            )

        early = experiment_state.early_signal_snapshot
        recent = experiment_state.recent_noise_snapshot
        early_titles = [cluster.get("label", "") for cluster in (early.title_clusters if early else []) if cluster.get("label")]
        recent_titles = [cluster.get("label", "") for cluster in (recent.title_clusters if recent else []) if cluster.get("label")]
        overfit_risk = "high" if early and len(early.signal_anchors) <= 1 and remaining_queued_strings > 1 else "medium"
        keyword_hypothesis = ""
        if early_titles:
            keyword_hypothesis = f"tighten around early signal titles like {early_titles[0]}"
        elif recent_titles:
            keyword_hypothesis = f"exclude late-page leakage titles like {recent_titles[0]}"
        future_filter_hypothesis = ""
        if recent_titles:
            future_filter_hypothesis = f"title filter could exclude {recent_titles[0]}"
        elif recent and recent.company_clusters:
            future_filter_hypothesis = f"company filter could demote {recent.company_clusters[0].get('label', '')}"

        if overfit_risk == "high":
            decision = "spawn_recall_sibling"
            rationale = "later pages are drifting, but the early signal sample is narrow enough that rescue should stay recall-friendly"
        else:
            decision = "refine_committed"
            rationale = "later pages are leaking noise and the committed variant has enough prior value to justify one bounded keyword rescue"
        return LinkedInDriftAssessment(
            decision=decision,
            rationale=rationale,
            eligible=True,
            overfit_risk=overfit_risk,
            keyword_hypothesis=keyword_hypothesis,
            future_filter_hypothesis=future_filter_hypothesis,
        )

    @staticmethod
    def _acquisition_mode_for_variant(
        search_string: SearchString,
        variant: LinkedInSearchVariant,
    ) -> str:
        """Phase 2 hop 4 (slice C, part 6): resolve the acquisition mode apply_variant
        must run under for THIS variant.

        The product invariant is `structured_filters present <=> linkedin_hybrid`, forced
        at lane-compile time (strategy_lane_compiler). A mid-run PROMOTE adds structured
        filters to a variant AFTER compile, on a lane whose SearchString is still frozen
        to linkedin_boolean — and apply_variant rejects a non-empty-filter variant unless
        the mode is linkedin_hybrid (search_mutation.py:60-73). Without this upgrade every
        mid-run promote on a boolean lane is silently DENIED: a dead planner call whose
        structured plan never executes.

        So: a variant carrying non-empty structured filters runs as linkedin_hybrid, and
        the SearchString is upgraded in place to keep the invariant true on the persisted
        record (mirrors the compile-time forcing; a later deliberate demote-to-boolean
        clears the checkpointed filters via apply_shadow while the mode stays hybrid —
        the established hybrid+empty==keyword-only contract). A keyword/empty-filter
        variant keeps the lane's own mode (no-op for the dominant boolean lane).
        """
        if not variant.structured_filters.is_empty():
            if search_string.acquisition_mode != "linkedin_hybrid":
                search_string.acquisition_mode = "linkedin_hybrid"
            return "linkedin_hybrid"
        return search_string.acquisition_mode or "linkedin_boolean"

    @staticmethod
    def _resolve_structured_controls(
        raw_variant: dict[str, Any],
        inherited: LinkedInStructuredFilters,
        *,
        structured_lever_open: bool = True,
        allowed_promoted_titles: set[str] | None = None,
        allowed_promoted_companies: set[str] | None = None,
    ) -> tuple[LinkedInStructuredFilters, str, str, bool]:
        """Phase 2 hop 4 (slice C, parts 3+4): turn a proposal's surface +
        structured_controls into the variant's structured_filters, surface, and kind.

        Shared by _plan_variant_experiments and _plan_drift_refinement so the two live
        LLM contracts apply identical promote/demote/structured_only semantics:
          - start from a COPY of the parent/active variant's filters (inherited),
          - ADD promoted titles/companies,
          - DROP demoted dims (_drop_one_filter, dimension-targeted),
          - resolve surface (default "boolean"); mark variant_kind "structured_filter"
            when structured_controls is non-empty (the deliberate-move provenance the
            seeding gate keys on).

        Returns (structured_filters, surface, variant_kind, structured_changed).
        structured_changed is True when the built filters differ from what was
        inherited — the signal the no-op guard adds to the boolean-unchanged check.

        Phase 2 hop 4 (slice E, part 2 — deterministic enforcement): when
        ``structured_lever_open`` is False (this lane has demoted structured controls
        >= K times) the circuit-breaker is not merely withheld from the prompt — it is
        ENFORCED here, in the deterministic parse layer, so a DISOBEYING model that
        ignores the keyword-only instruction and still emits surface:"hybrid" +
        structured_controls cannot re-promote and defeat the breaker. With the lever
        closed the parser refuses the only RE-PROMOTE vector: promoted titles/companies
        are dropped and the surface is coerced to "boolean", so a closed lane cannot add
        a structured dim or leave a non-boolean surface. DEMOTES are still honored —
        shedding is the breaker's own direction, so a model that keeps trying to drop a
        filter is obeyed (a demote that empties the filters keeps its surface=="boolean"
        + variant_kind=="structured_filter" deliberate-demotion marker for the seeding
        gate). The lever-open path is byte-identical to slice C.

        Results-rail graduation gate: title/company PROMOTEs are honored only when
        their literal label appears in the thresholded page/snapshot evidence supplied
        by the caller. A stripped weak promotion must not still wear structured_filter
        provenance just because raw JSON contained ``structured_controls``.
        """
        # Harden the unpack: a disobeying model can hand back a non-dict truthy
        # structured_controls (list/str/int) — `or {}` only rescues FALSY values, so
        # an isinstance gate is what keeps controls.get(...) from raising
        # (AttributeError would escape the drift planner's try/except, aborting the run).
        raw_controls = raw_variant.get("structured_controls")
        controls = raw_controls if isinstance(raw_controls, dict) else {}
        # Closed breaker: coerce the surface to "boolean" — the lever offers no
        # hybrid/structured_only, so a disobeying model's surface is overridden, not
        # trusted. (Open path keeps the model's surface, default "boolean".)
        if structured_lever_open:
            surface = str(raw_variant.get("surface") or "boolean").strip() or "boolean"
        else:
            surface = "boolean"
        filters = _copy_filters(inherited)

        # PROMOTE is the only RE-PROMOTE vector, so it is the one the closed breaker
        # refuses: with the lever closed, titles/companies adds are dropped entirely
        # (the inherited filters carry forward unchanged; demotes below still run).
        # Each dimension is likewise list-guarded: a str value (e.g. "NotAList")
        # otherwise iterates per-character into garbage titles/companies.
        ran_structured_control = False
        if structured_lever_open:
            raw_titles = controls.get("titles")
            promoted_titles = [
                str(t).strip()
                for t in (raw_titles if isinstance(raw_titles, list) else [])
                if str(t).strip()
            ]
            graduated_titles = [
                title
                for title in promoted_titles
                if allowed_promoted_titles is None
                or _normalize_title_family(title) in allowed_promoted_titles
            ]
            # Breadth floor (Sam's ruling 2026-08-05, CLO-73): a title filter
            # either spans the role's title families or it does not run at all.
            #
            # The rail-graduation gate above asks only "was this label observed
            # on the results rail", and the rail ranks by VOLUME — so every
            # label that can pass it is a MODAL title of the pool. That is how a
            # Principal-Research-Engineer search ended up bounded to ["Software
            # Engineer", "Applied Scientist"], losing ~360 of ~1,200 results and
            # every Research Engineer / Research Scientist / Member of Technical
            # Staff in the pool. Evidence-gating the labels individually cannot
            # catch that: each label IS evidenced; the SET is what is too narrow.
            #
            # Counted on families, not strings, so ["Software Engineer", "Senior
            # Software Engineer"] reads as the one family it is. Counted on the
            # PROSPECTIVE set (inherited + promoted) so a promotion that widens
            # an already-broad filter is judged on the filter it produces. Below
            # the floor the promotion is dropped WHOLE and the variant stays
            # keyword-only — Sam's stated fallback, and the fail-closed
            # direction: a search that is too broad wastes reads, a search that
            # is silently too narrow never meets the people at all.
            prospective_families = {
                family
                for family in (
                    _normalize_title_family(title)
                    for title in (*filters.titles, *graduated_titles)
                )
                if family
            }
            if (
                graduated_titles
                and len(prospective_families) < config.LINKEDIN_TITLE_FILTER_MIN_FAMILIES
            ):
                graduated_titles = []
            for title in graduated_titles:
                if title not in filters.titles:
                    filters.titles.append(title)
                    ran_structured_control = True
            raw_companies = controls.get("companies")
            promoted_companies = [
                str(c).strip()
                for c in (raw_companies if isinstance(raw_companies, list) else [])
                if str(c).strip()
            ]
            for company in promoted_companies:
                if (
                    allowed_promoted_companies is not None
                    and Pipeline._normalize_company_rail_label(company)
                    not in allowed_promoted_companies
                ):
                    continue
                if company not in filters.companies:
                    filters.companies.append(company)
                    ran_structured_control = True

        # DEMOTE: drop the named dimension(s) entirely (the over-narrowing/leaking
        # filter the loop is retiring). Matches _drop_one_filter's dimension set.
        raw_demote = controls.get("demote")
        demote_dims = [
            str(d).strip()
            for d in (raw_demote if isinstance(raw_demote, list) else [])
            if str(d).strip()
        ]
        for dim in demote_dims:
            if dim in {"titles", "companies", "skills", "assessments"}:
                ran_structured_control = True
                getattr(filters, dim).clear()
            elif dim in filters.sidebar_filters:
                ran_structured_control = True
                filters.sidebar_filters.pop(dim, None)
            elif dim in filters.advanced_filters:
                ran_structured_control = True
                filters.advanced_filters.pop(dim, None)

        structured_changed = filters.to_dict() != inherited.to_dict()
        # Make the "structured_filter" marker PROVENANCE-true, not LABEL-true: the kind
        # is keyed on by is_deliberate_boolean_demotion (seeding gate), so only a variant
        # that actually ran a structured control may wear it. A real structured move is
        # one that either carries non-empty controls (a promote, or a demote that named a
        # dimension — full demote-to-boolean included) OR changed the structured state.
        # An EMPTY-controls proposal that merely self-labels "structured_filter" while
        # changing nothing structural is a keyword variant; fall through to its keyword
        # kind so it is NOT mistaken for a deliberate demote and silently denied the
        # lane's structured-filter seed on a hybrid lane.
        #
        # Closed breaker: drop the bool(controls) term. With the lever closed a promote
        # was already stripped above, so a promote-only payload's still-truthy `controls`
        # had NO effect; keying the marker on it would mint a structured_filter kind for
        # a variant that ran no structured move and re-promote nothing — exactly the
        # disobeying-model re-promote the breaker exists to deny. Only a demote that
        # actually changed the filters (structured_changed) is a structured move when
        # closed, so a demote-to-empty keeps its deliberate-demotion provenance while a
        # stripped promote falls through to a plain keyword kind.
        ran_structured_move = ran_structured_control or structured_changed
        if ran_structured_move:
            variant_kind = "structured_filter"
        else:
            raw_kind = str(raw_variant.get("variant_kind") or "precision")
            variant_kind = "precision" if raw_kind == "structured_filter" else raw_kind
        # A structured move that emptied the filters is a deliberate boolean demotion;
        # normalize its surface so the marker (surface=="boolean" + kind=="structured_filter")
        # is unambiguous for the seeding gate.
        if ran_structured_move and filters.is_empty():
            surface = "boolean"
        elif not ran_structured_move and filters.is_empty():
            surface = "boolean"
        return filters, surface, variant_kind, structured_changed

    async def _plan_variant_experiments(
        self,
        *,
        search_string: SearchString,
        experiment_state: LinkedInExperimentState,
        current_boolean: str,
        result_count: int,
        result_count_text: str,
        page_insights: LinkedInPageInsights,
        all_candidates: list[dict],
        string_stats: dict[str, int],
    ) -> list[LinkedInSearchVariant]:
        from shared.llm_clients import opus_llm_cached

        target_window = result_window_for_count(result_count)
        candidates_text = "\n".join(
            f"- {candidate['name']} | {candidate['title']} at {candidate['company']} | {candidate['outcome']}"
            for candidate in all_candidates[-10:]
        )
        active_filters = experiment_state.active_variant.structured_filters
        current_filters_text = json.dumps(active_filters.to_dict(), indent=2)
        # Deterministic circuit-breaker (slice E, part 2): after K structured
        # demote-and-proceed events on this lane, the structured lever is CLOSED — the
        # promote/structured option is dropped from the instructions AND the schema, so
        # the LLM may only propose keyword variants. The execution metric CONSTRAINS the
        # proposal space; it does not touch the deterministic gate or the lifecycle
        # decision. A lane that keeps shedding structured controls stops being offered
        # them rather than re-proposing a filter the sidebar will not accept.
        structured_lever_open = (
            experiment_state.structured_demotions
            < config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT
        )
        allowed_promoted_titles, allowed_promoted_companies = (
            self._thresholded_structured_promotions(page_insights)
        )
        if structured_lever_open:
            structured_intro = """You may propose two kinds of move, and they are ADDITIVE — keyword variants remain
the backbone; structured moves are an extra lever, never a replacement.

KEYWORD variants (the backbone, unchanged):"""
            # f-string (unlike its sibling fragments) so the live breadth floor
            # reaches the model. There are no literal braces in this block, so
            # nothing needs doubling; keep it that way when editing.
            min_title_families = int(config.LINKEDIN_TITLE_FILTER_MIN_FAMILIES or 0)
            title_breadth_rule = (
                f"Fewer than {min_title_families} distinct title families is "
                "rejected outright and your variant runs keyword-only, so promote "
                "a full set or promote no titles at all."
                if min_title_families > 1
                else "Promote a full set or promote no titles at all."
            )
            structured_moves_block = f"""

STRUCTURED moves (additive — propose only when the page evidence is clean):
- PROMOTE: when page insights show a tight, LITERAL title or company cluster
  (an exact title/company string, not a fuzzy/semantic concept), promote it to a
  structured filter via structured_controls.titles / structured_controls.companies.
  Set surface "hybrid" (keep the Boolean) or "structured_only" (drop the Boolean and
  let the filter carry the search).
  TITLE PROMOTIONS MUST BE COMPREHENSIVE. A title filter deletes everyone who does
  not carry a listed title, so listing the two or three titles you happen to see on
  this page silently removes entire talent pools — the same people the Boolean was
  written to find, wearing a title you did not enumerate. When you promote titles,
  enumerate EVERY title family a qualified person for THIS role plausibly carries,
  including the ones absent from this page's clusters: the research-side titles, the
  engineering-side titles, the individual-contributor and the lead/principal variants,
  and the company-specific idioms (e.g. "Member of Technical Staff").
  {title_breadth_rule}
- DEMOTE: when an existing structured filter is over-narrowing or leaking, demote that
  dimension back to keyword via structured_controls.demote (e.g. ["titles"]). A full
  demote that empties the filters must set surface "boolean".

Surface judgment (same rule the lane compiler uses): a CLEAN LITERAL title/company maps
to a structured filter; a FUZZY or SEMANTIC concept stays a keyword. Do not promote a
loose phrase or a multi-concept idea — those belong in the Boolean."""
            structured_rule = "\n- Prefer a mix of keyword variants; add a structured move only when the evidence is clean."
            no_op_rule = """- Do not return a no-op: a variant whose Boolean equals the current Boolean AND whose
  structured_controls is empty changes nothing — omit it."""
            variant_kind_enum = "precision|recall|noise_exclusion|recovery|structured_filter"
            # NOTE: these fragments are PLAIN strings spliced into the outer f-string via
            # {surface_schema} / {keyword_footer}; they are inserted verbatim, NOT
            # re-evaluated, so braces here are SINGLE (only literals written directly in
            # the f-string body get doubled).
            surface_schema = """
      "surface": "boolean|hybrid|structured_only",
      "structured_controls": {"titles": ["..."], "companies": ["..."], "demote": ["titles|companies|skills|assessments"]},"""
            keyword_footer = 'For a keyword variant, omit structured_controls or pass {} and set surface "boolean".'
            keyword_surface_note = '\nFor these, surface is "boolean" and structured_controls is {}.'
        else:
            structured_intro = """This lane has shed its structured filters repeatedly, so structured moves are
DISABLED for it — propose KEYWORD variants only.

KEYWORD variants:"""
            structured_moves_block = ""
            structured_rule = ""
            no_op_rule = """- Do not return a no-op: a variant whose Boolean equals the current Boolean changes
  nothing — omit it."""
            variant_kind_enum = "precision|recall|noise_exclusion|recovery"
            surface_schema = ""
            keyword_footer = "Every variant is a keyword variant; there is no structured option."
            keyword_surface_note = ""
        system = f"""You are a senior LinkedIn Recruiter operator modernizing a live search.

Role: {self.brief_obj.role_title}
{self.brief_obj.role_description}

Goal: propose up to {config.SEARCH_EXPERIMENT_MAX_PLANNED_VARIANTS} conservative sibling variants for ONE root search family.

{structured_intro}
- precision: hyper-specific/high-signal Boolean
- recall: broader but still targeted Boolean
- noise_exclusion: removes the dominant noise pattern{keyword_surface_note}{structured_moves_block}

Rules:
- Keep the search human-like: no frantic rewrites, no throwaway variants, no cosmetic rewrites.{structured_rule}
{no_op_rule}
- Each variant must state a short hypothesis and a target result window.

Return JSON:
{{
  "variants": [
    {{
      "variant_id": "optional",
      "variant_kind": "{variant_kind_enum}",
      "hypothesis": "short explanation",
      "boolean": "LinkedIn Boolean (may be empty ONLY when surface is structured_only)",{surface_schema}
      "target_result_min": {target_window[0] if target_window else 75},
      "target_result_max": {target_window[1] if target_window else 400}
    }}
  ]
}}
{keyword_footer}"""
        target_window_text = f"{target_window[0]}-{target_window[1]}" if target_window else "75-400"
        user_prompt = f"""Current Boolean:
{current_boolean}

Current structured filters (inherited by any structured move; empty means keyword-only today):
{current_filters_text}

Current result count:
{result_count_text}

Target result window:
{target_window_text}

Page insights:
{json.dumps(page_insights.to_dict(), indent=2)}

Recent candidate outcomes:
{candidates_text or "- none"}

Current stats:
{json.dumps(string_stats, indent=2)}
"""
        variants: list[LinkedInSearchVariant] = []
        try:
            usage_context = {
                "stage": "linkedin_plan_variant_experiments",
                "brief_id": self.brief_obj.id,
                "string_id": search_string.id,
                "active_variant_id": experiment_state.active_variant_id,
            }
            result = opus_llm_cached(
                system,
                user_prompt,
                expect_json=True,
                max_tokens=16384,
                usage_context=usage_context,
                model_name=config.STRATEGY_MODEL_NAME,
            )
            for index, raw_variant in enumerate(result.get("variants", []), start=1):
                # Per-item parse is isolated: one malformed variant must drop ONLY
                # itself, never abort the batch. The loop-level except (below) would
                # otherwise discard every sibling parsed after a bad item — and with a
                # bad-FIRST ordering, return an empty list while well-formed later
                # variants are silently lost (rc<500 returns that empty list directly).
                try:
                    if not isinstance(raw_variant, dict):
                        continue
                    boolean = (raw_variant.get("boolean") or "").strip()
                    structured_filters, surface, variant_kind, structured_changed = (
                        self._resolve_structured_controls(
                            raw_variant,
                            active_filters,
                            structured_lever_open=structured_lever_open,
                            allowed_promoted_titles=allowed_promoted_titles,
                            allowed_promoted_companies=allowed_promoted_companies,
                        )
                    )
                    # No-op guard (slice C, part 3): the legacy boolean==current check now
                    # also requires NO structured change — a proposal that neither moves the
                    # Boolean nor changes the filters has nothing to run.
                    if boolean == current_boolean and not structured_changed:
                        continue
                    # An empty Boolean is only runnable when the filters carry it
                    # (surface=="structured_only"); otherwise there is nothing to execute.
                    if not boolean:
                        if surface != "structured_only" or structured_filters.is_empty():
                            continue
                    # Slice F: scale the healthy window DOWN at construction for a
                    # filter-led / structured variant so the keyword-tuned lifecycle gate
                    # does not mis-read a legitimately-narrower structured probe as
                    # too_narrow. A boolean variant is returned unscaled (byte-identical).
                    unscaled_min = int(raw_variant.get("target_result_min") or (target_window[0] if target_window else 75))
                    unscaled_max = int(raw_variant.get("target_result_max") or (target_window[1] if target_window else 400))
                    scaled_min, scaled_max = scale_window_for_surface(
                        unscaled_min,
                        unscaled_max,
                        surface=surface,
                        structured_filters=structured_filters,
                    )
                    variant = LinkedInSearchVariant(
                        variant_id=str(raw_variant.get("variant_id") or f"round-{experiment_state.experiment_round + 1}-{index}"),
                        parent_variant_id=experiment_state.active_variant_id,
                        root_string_id=search_string.id,
                        boolean=boolean,
                        variant_kind=variant_kind,
                        hypothesis=str(raw_variant.get("hypothesis") or ""),
                        target_result_min=scaled_min,
                        target_result_max=scaled_max,
                        structured_filters=structured_filters,
                        surface=surface,
                    )
                except Exception as item_exc:
                    log_event(
                        self.log_path,
                        "linkedin_search_plan_variant_skipped",
                        string_id=search_string.id,
                        error=str(item_exc),
                        traceback=traceback.format_exc(),
                    )
                    continue
                variants.append(variant)
                if len(variants) >= config.SEARCH_EXPERIMENT_MAX_PLANNED_VARIANTS:
                    break
        except Exception as exc:
            log_event(
                self.log_path,
                "linkedin_search_plan_failed",
                string_id=search_string.id,
                error=str(exc),
                traceback=traceback.format_exc(),
            )

        if not variants and result_count >= 500:
            fallback = await self._force_narrow_adapt(
                search_string,
                current_boolean,
                result_count_text,
                all_candidates,
                string_stats,
            )
            if fallback and fallback.startswith("narrow:"):
                boolean = fallback.split("narrow:", 1)[1]
                target_min, target_max = target_window or (75, 400)
                # Slice F: the forced-narrow fallback is a pure KEYWORD variant
                # (default surface, empty filters), so scaling is a no-op — routed
                # through the same helper to keep every build site consistent.
                target_min, target_max = scale_window_for_surface(
                    target_min,
                    target_max,
                    surface="",
                    structured_filters=LinkedInStructuredFilters(),
                )
                variants.append(
                    LinkedInSearchVariant(
                        variant_id=f"round-{experiment_state.experiment_round + 1}-fallback",
                        parent_variant_id=experiment_state.active_variant_id,
                        root_string_id=search_string.id,
                        boolean=boolean,
                        variant_kind="precision",
                        hypothesis="fallback forced narrow from scout/planner failure",
                        target_result_min=target_min,
                        target_result_max=target_max,
                    )
                )

        self._record_runtime_event(
            search_string=search_string,
            event_type="linkedin_search_plan_variants",
            payload={
                "active_variant_id": experiment_state.active_variant_id,
                "planned_count": len(variants),
                "variants": [variant.to_dict() for variant in variants],
            },
        )
        return variants

    async def _plan_drift_refinement(
        self,
        *,
        search_string: SearchString,
        experiment_state: LinkedInExperimentState,
        current_boolean: str,
        result_count: int,
        result_count_text: str,
        page_insights: LinkedInPageInsights,
        page_stats: dict[str, int],
    ) -> tuple[LinkedInSearchVariant | None, dict[str, Any]]:
        from shared.llm_clients import opus_llm_cached

        summary = dict(experiment_state.last_drift_refinement_summary)
        decision = summary.get("decision", "refine_committed")
        target_window = result_window_for_count(result_count) or (75, 400)
        early_snapshot = experiment_state.early_signal_snapshot
        recent_snapshot = experiment_state.recent_noise_snapshot
        promotion_evidence = early_snapshot or page_insights
        allowed_promoted_titles, allowed_promoted_companies = (
            self._thresholded_structured_promotions(promotion_evidence)
        )
        variant_kind = "recall" if decision == "spawn_recall_sibling" else "precision"
        active_filters = experiment_state.active_variant.structured_filters
        current_filters_text = json.dumps(active_filters.to_dict(), indent=2)
        # Deterministic circuit-breaker (slice E, part 2), mirroring
        # _plan_variant_experiments: once this lane has demoted structured controls K
        # times, the promote/structured lever is dropped from the drift prompt too. The
        # rescue stays a keyword rescue; the gate and lifecycle decision are untouched.
        structured_lever_open = (
            experiment_state.structured_demotions
            < config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT
        )
        if structured_lever_open:
            # f-string so the live breadth floor reaches the model; no literal
            # braces in this block, so nothing needs doubling.
            min_title_families = int(config.LINKEDIN_TITLE_FILTER_MIN_FAMILIES or 0)
            title_breadth_rule = (
                f"A title promotion must span at least {min_title_families} distinct "
                "title families or it is rejected outright and the rescue runs "
                "keyword-only."
                if min_title_families > 1
                else "Promote a full title set or promote no titles at all."
            )
            structured_moves_block = f"""

Drift is the STRONGEST promote signal: a title/company cluster that was stable and
productive on the early pages but degrades later is the prime candidate to lock behind
a structured filter rather than chase with more keywords. You may now ALSO:
- PROMOTE that stable early cluster, if it is a clean LITERAL title/company string, to a
  structured filter via structured_controls.titles / structured_controls.companies; set
  surface "hybrid" (keep the Boolean) or "structured_only" (let the filter carry it).
  A TITLE promotion must be COMPREHENSIVE, not a transcript of the clusters in front of
  you: the filter deletes everyone whose title you did not list, so enumerate every
  title family a qualified person for this role plausibly carries — research-side and
  engineering-side, IC and lead/principal, plus company-specific idioms such as
  "Member of Technical Staff" — including families absent from these pages.
  {title_breadth_rule}
- DEMOTE a structured filter that is now over-narrowing back to keyword via
  structured_controls.demote (a full demote that empties the filters sets surface "boolean").
A FUZZY or SEMANTIC anchor stays a keyword — record it as future_filter_hypothesis, not a
filter. Keyword rescue (precision/recall) remains the default; structured is additive."""
            no_op_rule = """- Do not return a no-op: a rescue whose Boolean equals the current Boolean AND whose
  structured_controls is empty changes nothing."""
            variant_kind_enum = "precision|recall|structured_filter"
            # PLAIN-string fragments spliced via {surface_schema} / {keyword_footer}:
            # inserted verbatim, so braces are SINGLE here (see _plan_variant_experiments).
            surface_schema = """
  "surface": "boolean|hybrid|structured_only",
  "structured_controls": {"titles": ["..."], "companies": ["..."], "demote": ["titles|companies|skills|assessments"]},"""
            keyword_footer = 'For a keyword rescue, omit structured_controls or pass {} and set surface "boolean".'
        else:
            structured_moves_block = """

This lane has shed its structured filters repeatedly, so structured promotes are
DISABLED for it — return a KEYWORD rescue only. A stable early cluster is recorded as a
future_filter_hypothesis, never promoted to a filter."""
            no_op_rule = """- Do not return a no-op: a rescue whose Boolean equals the current Boolean changes
  nothing."""
            variant_kind_enum = "precision|recall"
            surface_schema = ""
            keyword_footer = "This is a keyword rescue."
        system = f"""You are a senior LinkedIn Recruiter operator rescuing a search that started strong and then degraded mid-pagination.

Role: {self.brief_obj.role_title}
{self.brief_obj.role_description}

Task:
- Compare early productive pages against recent noisy pages.
- Return exactly ONE conservative rescue variant.
- If overfit risk is high, keep it recall-friendly. Otherwise tighten around the early-good anchors and exclude later leakage.{structured_moves_block}

{no_op_rule}

Return JSON:
{{
  "boolean": "LinkedIn Boolean (may be empty ONLY when surface is structured_only)",
  "variant_kind": "{variant_kind_enum}",{surface_schema}
  "hypothesis": "short explanation",
  "keyword_hypothesis": "what the rescue is trying to preserve/exclude",
  "future_filter_hypothesis": "optional future title/company/skill filter idea (fuzzy anchors only)",
  "target_result_min": {target_window[0]},
  "target_result_max": {target_window[1]}
}}
{keyword_footer}"""
        user_prompt = f"""Current Boolean:
{current_boolean}

Current structured filters (inherited by any structured move; empty means keyword-only today):
{current_filters_text}

Current result count:
{result_count_text}

Drift decision:
{json.dumps(summary, indent=2)}

Early signal snapshot:
{json.dumps(early_snapshot.to_dict() if early_snapshot else {}, indent=2)}

Recent noise snapshot:
{json.dumps(recent_snapshot.to_dict() if recent_snapshot else {}, indent=2)}

Current page insights:
{json.dumps(page_insights.to_dict(), indent=2)}

Current page stats:
{json.dumps(page_stats, indent=2)}
"""
        payload: dict[str, Any] = {}
        try:
            usage_context = {
                "stage": "linkedin_plan_drift_refinement",
                "brief_id": self.brief_obj.id,
                "string_id": search_string.id,
                "active_variant_id": experiment_state.active_variant_id,
            }
            payload = opus_llm_cached(
                system,
                user_prompt,
                expect_json=True,
                max_tokens=16384,
                usage_context=usage_context,
                model_name=config.STRATEGY_MODEL_NAME,
            )
        except Exception as exc:
            log_event(
                self.log_path,
                "linkedin_search_plan_failed",
                string_id=search_string.id,
                planner="drift",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        # A disobeying model can return non-dict JSON (a list/str); coerce so the
        # payload.get(...) reads below cannot raise (an AttributeError here escapes the
        # planner -> _process_string and, at the `run` driver, aborts the ENTIRE run).
        if not isinstance(payload, dict):
            payload = {}

        boolean = str(payload.get("boolean", "")).strip()
        # Slice C (part 4): resolve any structured promote/demote from the proposal,
        # inheriting the committed/active variant's filters. A hybrid promote keeps the
        # Boolean unchanged but carries a structured change, so the keyword-fallback
        # below must only fire for a PURE keyword rescue (no structured change).
        # Guarded: even though _resolve_structured_controls is now total, a parse fault
        # must degrade this string to the keyword fallback, never escape the planner.
        try:
            structured_filters, surface, resolved_kind, structured_changed = (
                self._resolve_structured_controls(
                    payload,
                    active_filters,
                    structured_lever_open=structured_lever_open,
                    allowed_promoted_titles=allowed_promoted_titles,
                    allowed_promoted_companies=allowed_promoted_companies,
                )
            )
        except Exception as exc:
            log_event(
                self.log_path,
                "linkedin_search_plan_failed",
                string_id=search_string.id,
                planner="drift_structured_resolve",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            structured_filters, surface, resolved_kind, structured_changed = (
                _copy_filters(active_filters),
                "boolean",
                variant_kind,
                False,
            )
        if (not boolean or boolean == current_boolean) and not structured_changed:
            boolean = self._fallback_drift_boolean(
                current_boolean=current_boolean,
                decision=decision,
                early_snapshot=early_snapshot,
                recent_snapshot=recent_snapshot,
            )

        keyword_hypothesis = str(payload.get("keyword_hypothesis") or summary.get("keyword_hypothesis") or "")
        future_filter_hypothesis = str(
            payload.get("future_filter_hypothesis") or summary.get("future_filter_hypothesis") or ""
        )
        drift_summary = {
            **summary,
            "keyword_hypothesis": keyword_hypothesis,
            "future_filter_hypothesis": future_filter_hypothesis,
        }
        # No-op guard (extended for slice C): reject only when neither the Boolean nor
        # the structured filters changed. A structured_only move (empty Boolean) is
        # runnable when its filters carry the search.
        boolean_unchanged = (not boolean or boolean == current_boolean)
        if boolean_unchanged and not structured_changed:
            return None, drift_summary
        if not boolean and (surface != "structured_only" or structured_filters.is_empty()):
            return None, drift_summary

        # variant_kind authority is the RESOLVED kind, not the raw model label — the
        # same provenance-over-label rule _resolve_structured_controls enforces (slice C)
        # and the variant planner already honors (it builds from the resolved kind
        # directly, :3759). A structured move uses resolved_kind; a model-supplied
        # KEYWORD kind (recall/precision/...) is preserved; but a "structured_filter"
        # label the resolver REFUSED (a label-only mislabel, or — under the slice-E
        # closed breaker — a disobeying promote whose structured controls were stripped)
        # must NOT survive: it falls through to the drift-decision kind so a closed lane
        # cannot wear the structured_filter provenance and re-promote.
        raw_drift_kind = str(payload.get("variant_kind") or "")
        if structured_changed:
            resolved_variant_kind = resolved_kind
        elif raw_drift_kind and raw_drift_kind != "structured_filter":
            resolved_variant_kind = raw_drift_kind
        else:
            resolved_variant_kind = variant_kind
        # Slice F: scale the healthy window DOWN at construction for a filter-led /
        # structured rescue (same posture-into-window bake as _plan_variant_experiments),
        # so the keyword-tuned lifecycle gate does not abandon a narrower structured
        # rescue. A boolean rescue is returned unscaled (byte-identical).
        unscaled_min = int(payload.get("target_result_min") or target_window[0])
        unscaled_max = int(payload.get("target_result_max") or target_window[1])
        scaled_min, scaled_max = scale_window_for_surface(
            unscaled_min,
            unscaled_max,
            surface=surface,
            structured_filters=structured_filters,
        )
        variant = LinkedInSearchVariant(
            variant_id=f"drift-{experiment_state.root_string_id}-{experiment_state.drift_attempt_count + 1}",
            parent_variant_id=experiment_state.committed_variant_id or experiment_state.active_variant_id,
            root_string_id=search_string.id,
            boolean=boolean,
            variant_kind=resolved_variant_kind,
            hypothesis=str(payload.get("hypothesis") or drift_summary.get("keyword_hypothesis") or "mid-pagination rescue"),
            target_result_min=scaled_min,
            target_result_max=scaled_max,
            structured_filters=structured_filters,
            surface=surface,
        )
        return variant, drift_summary

    def _fallback_drift_boolean(
        self,
        *,
        current_boolean: str,
        decision: str,
        early_snapshot: LinkedInVariantSnapshot | None,
        recent_snapshot: LinkedInVariantSnapshot | None,
    ) -> str:
        if decision == "spawn_recall_sibling":
            if recent_snapshot and recent_snapshot.title_clusters:
                label = str(recent_snapshot.title_clusters[0].get("label", "")).strip()
                if label:
                    return f"({current_boolean}) NOT (\"{label}\")"
            return ""
        if early_snapshot and early_snapshot.title_clusters:
            label = str(early_snapshot.title_clusters[0].get("label", "")).strip()
            if label and label.lower() not in current_boolean.lower():
                return f"({current_boolean}) AND (\"{label}\")"
        if recent_snapshot and recent_snapshot.title_clusters:
            label = str(recent_snapshot.title_clusters[0].get("label", "")).strip()
            if label:
                return f"({current_boolean}) NOT (\"{label}\")"
        return ""

    def _should_force_narrow_in_scout(
        self,
        *,
        search_string: SearchString,
        page_num: int,
        result_count: int,
        string_stats: dict,
        glance_result: GlanceResult | None,
    ) -> bool:
        """Allow forced narrow only for page-1 scout strings that show zero signal."""
        return (
            search_string.phase == "scout"
            and page_num == 1
            and result_count >= 500
            and not search_string.refinement_stack
            and string_stats.get("full_outreach", 0) == 0
            and string_stats.get("full_review", 0) == 0
            and glance_result is not None
            and glance_result.action == "reformulate"
        )

    def _get_early_exit_rate(self) -> float:
        """Get facial_no rate threshold for mid-page early exit.

        Derives from the brief's market_density: sparse markets pay the most
        wall-clock per page, so they abandon a dead page at a LOWER facial_no
        rate. (The retired brief-derived formula ``1 - expected_yes_low * 0.5``
        inverted this — the sparser the expected yes rate, the more facial_no a
        page had to show before the agent left it — and the per-architecture
        overrides it fed rode a shape taxonomy that no longer prescribes
        behavior.)

        The observed denominator retains FACIAL_YES and FACIAL_BORDERLINE as
        distinct full-review-eligible outcomes. BORDERLINE never enters the
        FACIAL_NO numerator, so ternary mode cannot silently turn uncertainty
        into negative evidence.
        """
        density = str(getattr(self.brief_obj, "market_density", "") or "").strip().lower()
        return config.EARLY_EXIT_FACIAL_NO_RATE_BY_DENSITY.get(
            density, config.EARLY_EXIT_FACIAL_NO_RATE
        )

    def _record_last_good_url(self, url: object) -> None:
        """Remember a recovery URL only when it is PROVABLY the brief's project.

        F4. `_ensure_browser_healthy` re-navigates to `_last_good_url` after an
        error or crash recovery, and the rebind that precedes it
        (`_bind_existing_recruiter_page`) chooses among all open Recruiter tabs.
        Recording whatever tab it landed on turns one stray rebind into a run
        that keeps steering itself back into the wrong project: every save then
        aborts at `_assert_brief_project_context`, so the run burns rather than
        misfiles — but it burns for the rest of the session.

        Refusing here rather than at the read site keeps the invariant true for
        all three snapshot points and any added later.
        """
        current_url = str(url or "")
        if "linkedin.com/talent" not in current_url:
            return
        if "/login" in current_url or "/manage/" in current_url:
            return
        if _is_foreign_project_page(
            current_url, self.brief_obj.linkedin_project_id
        ):
            return
        self._last_good_url = current_url

    async def _ensure_browser_healthy(self) -> None:
        """Check for error states, attempt recovery, and enforce cadence pauses."""
        # --- Cadence pause: human-like idle break ---
        await self._maybe_cadence_pause()

        # --- Snapshot current URL as known-good before checking for errors ---
        try:
            self._record_last_good_url(self.browser.page.url)
        except Exception:
            pass

        # --- Error recovery ---
        recovered = await self.browser.check_and_recover()
        if recovered:
            # Prefer the last known-good URL: it carries the live search position
            # and, by _record_last_good_url's invariant, is already this brief's
            # project. Fall back to the bare project search page. Re-checked here
            # anyway — this is the navigation that decides which pipeline the run
            # resumes against, and a read-site check costs nothing.
            recovery_url = self._last_good_url or self._get_project_url()
            if _is_foreign_project_page(
                recovery_url, self.brief_obj.linkedin_project_id
            ):
                recovery_url = self._get_project_url()
            if recovery_url:
                print(f"  [recovery] Re-navigating to: {recovery_url[:60]}...")
                await self.browser.navigate_to_search(recovery_url)
                # The recovery re-navigation dropped the sidebar; re-assert location.
                self._geography_service.reset_location_applied()
                await self._apply_session_location_filter()
            else:
                print("  [recovery] No recovery URL available — continuing from current page.")

        if config.LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED:
            try:
                from linkedin.recruiter_recovery import detect_recruiter_health
                health = await detect_recruiter_health(self.browser)
            except Exception:
                health = None
            if health == "blocked_or_rate_limited":
                print("  [health] Live health classifier: blocked_or_rate_limited — tripping governor backoff.")
                self._governor.force_backoff("blocked_or_rate_limited")

    def _get_project_url(self) -> str:
        """Construct LinkedIn Recruiter project URL from brief or auto-detected browser URL."""
        pid = self.brief_obj.linkedin_project_id
        if not pid and hasattr(self.browser, '_project_id'):
            pid = self.browser._project_id
        return recruiter_project_search_url(pid)

    def _sample_cadence_interval(self) -> float:
        if config.LINKEDIN_CADENCE_READ_FIX_ENABLED:
            return human_delay_correlated(
                config.CADENCE_INTERVAL_MINUTES, channel="cadence_interval"
            )
        return human_delay(
            config.CADENCE_INTERVAL_MINUTES * 0.8,
            config.CADENCE_INTERVAL_MINUTES * 1.4,
        )

    def _sample_cadence_pause(self) -> float:
        if config.LINKEDIN_CADENCE_READ_FIX_ENABLED:
            return human_delay_correlated(
                config.CADENCE_PAUSE_SECONDS, channel="cadence_pause"
            )
        return human_delay(
            config.CADENCE_PAUSE_SECONDS * 0.6,
            config.CADENCE_PAUSE_SECONDS * 2.0,
        )

    async def _maybe_cadence_pause(self) -> None:
        """Pause if enough continuous activity time has elapsed (anti-detection).

        When running under session_orchestrator, decoy interleave bursts serve as
        cadence breaks — skip the independent timer to avoid double-pausing.
        """
        # Decoy interleaving replaces cadence pauses when session_orchestrator is active
        if hasattr(self, '_pause_requested'):
            return

        if config.CADENCE_INTERVAL_MINUTES <= 0:
            return

        elapsed = (time.time() - self._last_pause_time) / 60.0
        # Jitter the interval ±20% so it's not metronomic
        jittered_interval = self._sample_cadence_interval()

        if elapsed >= jittered_interval:
            # Jitter the pause duration ±25%
            pause_secs = self._sample_cadence_pause()
            pause_started = time.monotonic()
            try:
                print(f"\n  [cadence] {elapsed:.0f}min of activity — pausing {pause_secs:.0f}s to look human...")
                log_event(self.log_path, "cadence_pause", elapsed_minutes=round(elapsed, 1),
                          pause_seconds=round(pause_secs, 1))
                await asyncio.sleep(pause_secs)
                self._last_pause_time = time.time()
                print(f"  [cadence] Resuming.")
            finally:
                emit_timing_event(
                    getattr(self, "_timing_recorder", None),
                    "cadence_pause_timing",
                    elapsed_ms=round(
                        (time.monotonic() - pause_started) * 1000.0,
                        3,
                    ),
                    pause_seconds=round(pause_secs, 3),
                )

    # ------------------------------------------------------------------
    # Glance assessment — page-level pre-filter
    # ------------------------------------------------------------------

    def _get_glance_key_terms(self) -> list[str]:
        """Extract key terms from the brief for glance keyword scanning."""
        nb = self.brief_obj._new_brief
        if nb is not None:
            # V2 brief: collect key_terms from all capability_areas
            terms = []
            for ca in nb.capability_areas:
                terms.extend(t.lower() for t in ca.key_terms)
            return terms
        # Old brief: extract from archetypes save_signals
        terms = []
        for arch in self.brief_obj.archetypes:
            for sig in arch.get("save_signals", []):
                # Each save_signal is a short phrase — use as-is
                terms.append(sig.lower())
        return terms

    def _glance_assess(self, snippets: list[CandidateSnippet]) -> GlanceResult:
        """Fast page-level assessment from snippet metadata. No per-candidate LLM calls."""
        signals = {}
        noise_count = 0

        # --- Signal 1: Title clustering ---
        title_families: dict[str, int] = {}
        for s in snippets:
            family = _normalize_title_family(s.current_title) if s.current_title else ""
            if family:
                title_families[family] = title_families.get(family, 0) + 1

        if title_families:
            top_family = max(title_families, key=title_families.get)
            top_ratio = title_families[top_family] / len(snippets)
            signals["title_cluster"] = {
                "top_family": top_family,
                "ratio": round(top_ratio, 2),
            }
            if top_ratio >= config.GLANCE_NOISE_TITLE_THRESHOLD:
                # Check if top family matches a known non-fit pattern
                is_noise_title = False
                nb = self.brief_obj._new_brief
                if nb is not None:
                    for nf in nb.non_fit_patterns:
                        nf_lower = [ex.lower() for ex in nf.examples]
                        if any(top_family in ex or ex in top_family for ex in nf_lower):
                            is_noise_title = True
                            break
                        if top_family in nf.label.lower() or top_family in nf.description.lower():
                            is_noise_title = True
                            break
                else:
                    for na in self.brief_obj.noise_archetypes:
                        na_name = na.get("name", "").lower()
                        na_desc = na.get("description", "").lower()
                        na_signals = [s.lower() for s in na.get("signals", [])]
                        if (top_family in na_name or top_family in na_desc
                                or any(top_family in sig or sig in top_family for sig in na_signals)):
                            is_noise_title = True
                            break
                if is_noise_title:
                    signals["title_cluster"]["noise"] = True
                    noise_count += 1

        # --- Signal 2: Keyword scan ---
        key_terms = self._get_glance_key_terms()
        if key_terms:
            hits = 0
            for s in snippets:
                text = " ".join([
                    s.headline or "",
                    s.current_title or "",
                    " ".join(s.experience_entries),
                ]).lower()
                if any(term in text for term in key_terms):
                    hits += 1
            signals["keyword_scan"] = {"hits": hits, "total": len(snippets)}
            if hits <= config.GLANCE_KEYWORD_MISS_THRESHOLD:
                signals["keyword_scan"]["noise"] = True
                noise_count += 1

        # --- Signal 3: Non-fit pattern scan ---
        nb = self.brief_obj._new_brief
        nf_examples: list[str] = []
        if nb is not None:
            for nf in nb.non_fit_patterns:
                nf_examples.extend(ex.lower() for ex in nf.examples)
        else:
            for na in self.brief_obj.noise_archetypes:
                nf_examples.extend(s.lower() for s in na.get("signals", []))

        if nf_examples:
            nf_matches = 0
            for s in snippets:
                text = " ".join([
                    s.headline or "",
                    s.current_title or "",
                    " ".join(s.experience_entries),
                ]).lower()
                if any(ex in text for ex in nf_examples):
                    nf_matches += 1
            nf_ratio = nf_matches / len(snippets) if snippets else 0
            signals["non_fit_scan"] = {
                "matches": nf_matches,
                "total": len(snippets),
                "ratio": round(nf_ratio, 2),
            }
            if nf_ratio > 0.5:
                signals["non_fit_scan"]["noise"] = True
                noise_count += 1

        # --- Decision ---
        if noise_count >= 3:
            titles_summary = ", ".join(
                f"{f} ({c})" for f, c in sorted(title_families.items(), key=lambda x: -x[1])[:5]
            )
            return GlanceResult(
                action="reformulate",
                summary=f"3/3 noise signals. Top titles: {titles_summary}. "
                        f"0/{len(snippets)} keyword hits." if key_terms else f"3/3 noise signals. Top titles: {titles_summary}.",
                confidence=0.9,
                signals=signals,
            )
        elif noise_count == 0:
            return GlanceResult(
                action="proceed",
                summary="No noise signals detected.",
                confidence=0.95,
                signals=signals,
            )
        else:
            # 1-2 signals: use cheap LLM to disambiguate
            llm_verdict = self._glance_llm_check(snippets, signals)
            if llm_verdict == "noise":
                titles_summary = ", ".join(
                    f"{f} ({c})" for f, c in sorted(title_families.items(), key=lambda x: -x[1])[:5]
                )
                return GlanceResult(
                    action="reformulate",
                    summary=f"{noise_count}/3 noise signals + LLM confirms noise. Top titles: {titles_summary}.",
                    confidence=0.7,
                    signals=signals,
                )
            else:
                return GlanceResult(
                    action="proceed",
                    summary=f"{noise_count}/3 noise signals but LLM says proceed.",
                    confidence=0.6,
                    signals=signals,
                )

    def _glance_llm_check(self, snippets: list[CandidateSnippet], signal_details: dict) -> str:
        """Cheap LLM call to disambiguate ambiguous glance signals. Returns 'noise' or 'proceed'."""
        from shared.llm_clients import cheap_llm

        # Format first 10 snippets
        snippet_lines = []
        for s in snippets[:10]:
            company = f" at {s.current_company}" if s.current_company else ""
            snippet_lines.append(f"- {s.name} | {s.current_title}{company} | {s.headline}")
        snippets_text = "\n".join(snippet_lines)

        # Format signal details
        signal_lines = []
        for name, details in signal_details.items():
            if details.get("noise"):
                signal_lines.append(f"  {name}: NOISE — {details}")
            else:
                signal_lines.append(f"  {name}: OK — {details}")
        signals_text = "\n".join(signal_lines)

        system = f"""You are a sourcing quality checker. Decide whether a LinkedIn search results page is hitting the right population or is noise.

Role: {self.brief_obj.role_title}
{self.brief_obj.role_description}

Minimum bar: {self.brief_obj.minimum_bar}

Respond with ONLY the word "noise" or "proceed". Nothing else."""

        user = f"""Here are the first {len(snippets[:10])} search result snippets from this page:
{snippets_text}

Automated signal analysis:
{signals_text}

Is this page mostly noise (wrong population) or does it have plausible candidates? Reply "noise" or "proceed"."""

        usage_context = {
            "stage": "linkedin_glance_llm_check",
            "source": "linkedin",
            "brief_id": getattr(self.brief_obj, "id", None) or self._brief_id,
            "snippet_count": len(snippets[:10]),
        }

        try:
            result = cheap_llm(
                system,
                user,
                expect_json=False,
                usage_context=usage_context,
            )
            verdict = result.strip() if isinstance(result, str) else str(result).strip()
            first_token = verdict.split(maxsplit=1)[0].strip('.,:;!?"\'`()[]{}').lower() if verdict else ""
            return "noise" if first_token == "noise" else "proceed"
        except Exception as e:
            print(f"    [glance-llm] Error: {e} — defaulting to proceed")
            log_event(
                self.log_path,
                "glance_llm_error",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return "proceed"

    async def _force_narrow_adapt(
        self,
        search_string: SearchString,
        current_boolean: str,
        result_count_text: str,
        all_candidates: list[dict],
        string_stats: dict,
    ) -> str | None:
        """Called when abandon/stop is blocked by min-pages. Forces Opus to attempt a narrow.

        Returns:
            "narrow:<boolean>" if Opus provides a narrowed Boolean, None if it can't.
        """
        from shared.llm_clients import opus_llm_cached

        candidate_lines = []
        for c in all_candidates:
            candidate_lines.append(
                f"  p{c['page']} | {c['outcome']:10s} | {c['name']} | {c['title']} at {c['company']} | {c['rationale'][:80]}"
            )
        candidates_text = "\n".join(candidate_lines)

        system = f"""You are a senior sourcing strategist. Your previous recommendation to abandon this search was blocked because minimum pagination requirements haven't been met.

Role: {self.brief_obj.role_title}
{self.brief_obj.role_description}

## Minimum Bar
{self.brief_obj.minimum_bar}

## Your task
Instead of abandoning, you MUST provide a narrowed Boolean that filters out the dominant noise pattern. The current string has {result_count_text} results — there may be signal buried under noise if you add the right AND clauses.

Analyze the candidate outcomes below and identify the dominant noise pattern (wrong domain? wrong seniority? wrong function?). Then add AND terms to exclude that noise.

{render_adaptation_matching_guidance()}

Return JSON:
- "action": must be "narrow"
- "rationale": What noise pattern you identified and what AND terms you're adding to exclude it
- "refined_boolean": The narrowed Boolean string"""

        user_prompt = f"""## Current Boolean
{current_boolean}

## Result Count
{result_count_text}

## Stats ({string_stats['pages']} pages)
- Candidates: {string_stats['candidates']} | Full reviewed: {string_stats.get('full_reviewed', 0)} | Outreach-positive: {string_stats.get('full_outreach', 0)} | Human review: {string_stats.get('full_review', 0)} | Full rejects: {string_stats.get('full_reject', 0)} | Facial YES/BORDERLINE: {string_stats['facial_yes']}/{string_stats.get('facial_borderline', 0)} | Facial NO: {string_stats['facial_no']}

## Candidates
{candidates_text}

Provide a narrowed Boolean."""

        try:
            usage_context = {
                "stage": "linkedin_force_narrow_adapt",
                "brief_id": self.brief_obj.id,
                "string_id": search_string.id,
            }
            result = opus_llm_cached(
                system,
                user_prompt,
                expect_json=True,
                max_tokens=16384,
                usage_context=usage_context,
                model_name=config.STRATEGY_MODEL_NAME,
            )
            new_boolean = result.get("refined_boolean", "")
            rationale = result.get("rationale", "")
            if new_boolean:
                print(f"  [adapt] Forced narrow: {rationale}")
                log_event(self.log_path, "forced_narrow", string_id=search_string.id,
                          rationale=rationale, new_boolean=new_boolean[:100])
                return f"narrow:{new_boolean}"
            else:
                print(f"  [adapt] Forced narrow returned no Boolean — honoring abandon.")
                try:
                    log_event(
                        self.log_path,
                        "forced_narrow_failed",
                        string_id=search_string.id,
                        reason="no_boolean",
                        rationale=rationale,
                    )
                except Exception as event_error:
                    print(f"  [warn] forced_narrow_failed event failed: {event_error}")
                return None
        except Exception as e:
            print(f"  [adapt] Forced narrow failed ({e}) — honoring abandon.")
            try:
                log_event(
                    self.log_path,
                    "forced_narrow_failed",
                    string_id=search_string.id,
                    reason="exception",
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
            except Exception as event_error:
                print(f"  [warn] forced_narrow_failed event failed: {event_error}")
            return None

    async def _evaluate_snippet(
        self,
        snippet: CandidateSnippet,
        page_report: _PageReport | None = None,
        search_string: SearchString | None = None,
        string_stats: dict[str, int] | None = None,
    ) -> Optional[OpusDecision]:
        """Run snippet through facial judgment -> full profile -> final judgment."""
        runtime_search_string = search_string or SearchString(
            id=snippet.source_string_id,
            name=snippet.source_string_name,
            boolean="",
        )
        facial_attempt_id: int | None = None

        pending_facial = self._resume_pending_full_decision(snippet)
        if pending_facial is None:
            raise RuntimeError(
                "foreign pending full review reached the active string evaluator"
            )
        if pending_facial:
            print(
                f"    [resume-full] {snippet.name} — {pending_facial} already "
                "settled; completing full review"
            )
            if snippet.profile_url:
                self._in_flight_urls.add(snippet.profile_url)
            self._note_page_full_review_expected(snippet)
            resumed_full = await self._full_evaluate(
                snippet,
                page_report,
                runtime_search_string,
            )
            self._clear_resume_pending_full_if_settled(
                snippet=snippet,
                decision=resumed_full,
            )
            return resumed_full

        # --- Employer blacklist check (no LLM call) ---
        blacklist_match = self._employer_blacklist_match(
            snippet,
            self.brief_obj.employer_blacklist,
        )
        if blacklist_match:
            blocked, matched_field, matched_value = blacklist_match
            print(
                f"    [BLACKLIST] {snippet.name} — "
                f"'{matched_value}' matches '{blocked}' ({matched_field})"
            )
            self.stats.setdefault("blacklist_skips", 0)
            self.stats["blacklist_skips"] += 1
            if page_report:
                page_report.add_skip_preview(snippet.name, f"BLACKLIST: {blocked}")
            blacklist_decision = OpusDecision(
                stage="facial", decision="FACIAL_NO", path="employer_blacklist",
                confidence=1.0, rationale=f"Employer blacklist: {blocked} ({matched_field})",
                candidate_name=snippet.name, profile_url=snippet.profile_url,
            )
            # Same registration-before-stage-write requirement as the
            # batch path's blacklist branch (see there for the RCA).
            self._record_runtime_snippet(runtime_search_string, snippet)
            facial_attempt_id = self._start_runtime_stage_attempt(
                search_string=runtime_search_string,
                snippet=snippet,
                stage="facial",
            )
            self._finish_runtime_stage_success(
                attempt_id=facial_attempt_id,
                stage="facial",
                snippet=snippet,
                decision=blacklist_decision,
            )
            self._record_facial_funnel_outcome(
                snippet=snippet,
                decision=blacklist_decision.decision,
                search_string=runtime_search_string,
                string_stats=string_stats,
            )
            self._prior_outcomes[snippet.profile_url] = "FACIAL_NO"
            self._mark_terminal(snippet.profile_url)
            return blacklist_decision

        # --- Facial judgment (facial tier) ---
        print(f"    Facial judgment ({config.FACIAL_MODEL_NAME.rsplit('/', 1)[-1]})...")
        facial_call_id = f"judge-{uuid.uuid4().hex}"
        facial_attempt_payload = {
            "logical_call_id": facial_call_id,
            "stage": "facial",
            "batch_index": 0,
            "batch_count": 1,
            "batch_size": 1,
            "batch_slot": 0,
        }
        facial_attempt_id = self._start_runtime_stage_attempt(
            search_string=runtime_search_string,
            snippet=snippet,
            stage="facial",
            payload=facial_attempt_payload,
        )
        _lane_ctx = {
            **self._lane_context_for_stage(runtime_search_string, stage="facial"),
            "logical_call_id": facial_call_id,
            "string_id": runtime_search_string.id,
            "page": snippet.page,
            "batch_index": 0,
            "batch_count": 1,
            "batch_size": 1,
            "batch_slot": 0,
        }
        try:
            facial = facial_judge(snippet, brief=self.brief_obj, prompt_prefix=self._tightening_prefix, lane_context=_lane_ctx)
        except Exception as e:
            if is_api_budget_exhausted_error(e):
                budget_error = ApiBudgetExhaustedError(str(e))
                self._abort_runtime_stage_attempt(
                    attempt_id=facial_attempt_id,
                    snippet=snippet,
                    error=budget_error,
                    payload={
                        **facial_attempt_payload,
                        "api_budget_exhausted": True,
                        "stage": "facial",
                    },
                )
                raise budget_error from e
            print(f"    [ERROR] Facial judgment failed: {e}")
            log_event(self.log_path, "facial_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            facial = judgment_failure_decision(
                stage="facial",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
                error=e,
                source="judgment",
            )

        # Preserve raw borderline observability while retaining the verdict
        # itself in canonical state below.
        if (
            self._facial_ternary_enabled()
            and self._bias_monitor is not None
            and facial.decision == "FACIAL_BORDERLINE"
        ):
            self._bias_monitor.record_facial_borderline_seen(
                string_id=str(snippet.source_string_id),
            )

        self._stamp_read_interest(snippet, facial.decision)

        facial = self._normalize_facial_decision_for_persistence(facial)

        # Intercept parse/judgment failures — log but do NOT persist to cross-session history
        if is_failure_decision(facial.decision):
            print(f"    [PARSE_FAILURE] {facial.rationale}")
            self.stats.setdefault("parse_failures", 0)
            self.stats["parse_failures"] += 1
            if self._bias_monitor:
                self._bias_monitor.record_decision(DecisionRecord(
                    candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                    string_id=str(snippet.source_string_id),
                    stage="facial",
                    decision=facial.decision,
                    confidence=facial.confidence,
                    capability_area=None,
                ))
            # Non-terminal: allow retry later in this session or on resume
            self._finish_runtime_failure_decision(
                attempt_id=facial_attempt_id,
                snippet=snippet,
                decision=facial,
                payload=facial_attempt_payload,
            )
            self._in_flight_urls.discard(snippet.profile_url)
            return facial

        self._prior_outcomes[snippet.profile_url] = facial.decision
        self._mark_terminal(snippet.profile_url)
        self._finish_runtime_stage_success(
            attempt_id=facial_attempt_id,
            stage="facial",
            snippet=snippet,
            decision=facial,
            extra_payload=facial_attempt_payload,
        )

        # Record facial decision for bias monitoring
        if self._bias_monitor:
            self._bias_monitor.record_decision(DecisionRecord(
                candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                string_id=str(snippet.source_string_id),
                stage="facial",
                decision=facial.decision,
                confidence=facial.confidence,
                capability_area=None,
            ))
            # Check if triage should be tightened for this string. Flag-gated OFF
            # by default — see the sibling site in the batch path for the full
            # rationale (2026-07-30: verdict-affecting injection keyed to a
            # brief-level guessed band, punishing per-string precision probes).
            if (
                config.LINKEDIN_FACIAL_TIGHTENING_ENABLED
                and not self._triage_tightened
                and self._bias_monitor  # V1 briefs leave this None
            ):
                tightening = self._bias_monitor.get_tightening_status(str(snippet.source_string_id))
                if tightening:
                    self._triage_tightened = True
                    self._tightening_prefix = (
                        f"⚠ TRIAGE TIGHTENING ACTIVE: The facial YES rate on this search string is running "
                        f"{tightening['actual_rate']:.0%}, which is {tightening['multiplier']:.1f}x above the expected "
                        f"maximum of {tightening['expected_high']:.0%}. Apply stricter filtering: require TWO strong "
                        f"positive signals for FACIAL_YES instead of one. Generic seniority + AI keywords is insufficient. "
                        f"Example of insufficient: 'VP of Digital Transformation at Accenture' — senior title + consulting firm "
                        f"mentioning AI, but no specific ML/data work visible. Example of sufficient: 'ML Engineer at DeepMind' "
                        f"+ 'Research Scientist at Google Brain' — two positions with direct capability area connections.\n\n"
                    )
                    print(f"    [bias] Tightening facial criteria for remaining candidates on this string "
                          f"(YES rate: {tightening['actual_rate']:.0%}, expected max: {tightening['expected_high']:.0%})")

        if facial.decision in ("FACIAL_NO", "FACIAL_SKIP"):
            tag = "FACIAL_NO" if facial.decision == "FACIAL_NO" else "FACIAL_SKIP"
            print(f"    [{tag}] {facial.rationale}")
            if facial.decision == "FACIAL_NO":
                self._record_facial_funnel_outcome(
                    snippet=snippet,
                    decision=facial.decision,
                    search_string=runtime_search_string,
                    string_stats=string_stats,
                )
            else:
                self.stats.setdefault("facial_skip", 0)
                self.stats["facial_skip"] += 1
            if page_report:
                page_report.add_skip_preview(snippet.name, f"{tag}: {facial.rationale}")
            return facial

        print(f"    [{facial.decision}] {facial.rationale}")
        self._record_facial_funnel_outcome(
            snippet=snippet,
            decision=facial.decision,
            search_string=runtime_search_string,
            string_stats=string_stats,
        )

        if self._should_skip_full_eval_for_activity(snippet, facial):
            self._record_activity_saturation_context(
                snippet=snippet,
                search_string=runtime_search_string,
                facial_decision=facial,
            )

        self._track_full_review_obligation(snippet, facial.decision)
        self._note_page_full_review_expected(snippet)
        full_decision = await self._full_evaluate(
            snippet,
            page_report,
            runtime_search_string,
        )
        self._clear_resume_pending_full_if_settled(
            snippet=snippet,
            decision=full_decision,
        )
        return full_decision

    def _full_candidate_id_mismatch_facts(
        self,
        decision: OpusDecision,
    ) -> dict | None:
        """Machine-typed mismatch facts, or None for any other outcome.

        Reads ONLY the structured record ``full_judge`` attaches to the prompt
        capture. The rationale is deliberately not consulted: its detail
        carries the model-returned ID, and profile text is attacker
        controlled, so classifying on prose would hand an injection the very
        channel this check exists to defend.
        """

        if decision.decision != "PARSE_FAILURE":
            return None
        capture = decision.prompt_capture or {}
        failure = capture.get("judgment_contract_failure")
        if not isinstance(failure, dict):
            return None
        if failure.get("reason") != "candidate_id_mismatch":
            return None
        return failure

    @staticmethod
    def _full_id_mismatch_event_payload(
        *,
        facts: dict,
        snippet: CandidateSnippet,
        logical_call_id: str,
        retry_logical_call_id: str,
        parent_logical_call_id: str,
        retry_scheduled: bool,
        is_retry_result: bool,
        recovered_so_far: int,
    ) -> dict:
        """One payload shape for both mismatch receipts.

        A reader joining the two emissions of a double mismatch must not have
        to branch on which one it holds, so the first (parent) and second
        (re-ask) receipts carry identical keys and differ only in values.
        """

        return {
            "candidate_name": snippet.name,
            "profile_url": snippet.profile_url,
            "expected_id": str(facts.get("expected_candidate_id") or ""),
            # Model output: clamped here, and printed nowhere.
            "actual_id": str(facts.get("actual_candidate_id") or "")[
                :_FULL_ID_MISMATCH_ACTUAL_ID_MAX_CHARS
            ],
            "logical_call_id": logical_call_id,
            "retry_logical_call_id": retry_logical_call_id,
            "parent_logical_call_id": parent_logical_call_id,
            "retry_scheduled": retry_scheduled,
            "is_retry_result": is_retry_result,
            "recovered_mismatches_so_far": recovered_so_far,
        }

    def _resolve_full_candidate_id_mismatch(
        self,
        *,
        decision: OpusDecision,
        summary: CandidateProfileSummary,
        snippet: CandidateSnippet,
        search_string: SearchString,
        lane_context: dict,
        logical_call_id: str,
        trace: dict,
    ) -> tuple[OpusDecision, str]:
        """Re-ask ONCE for a full evaluation the model mis-attributed.

        Returns the decision that flows onward and the logical call ID the
        corruption breaker must key on. Every mismatch is recorded and
        classified BEFORE anything is re-asked — including the ones the
        recovery ceiling refuses — because the receipt is the security signal
        and the re-ask is only a throughput recovery layered on top of it.
        The re-ask carries a FRESH opaque ID and its own child call, so a
        second mismatch is a genuinely independent observation rather than a
        replay of the first, and it earns its own receipt: two mis-attributions
        under two unrelated IDs is the loudest injection signal this design
        can produce, so it is classified rather than merely struck. A re-ask
        that fails surfaces exactly as the original would have, keyed to the
        child call, and only a re-ask that produced a verdict spends budget.

        This method NEVER touches the canonical attempt. It stamps ``trace``
        with the phase it is in and lets every failure escape, so the caller
        owns the single close site and no path can double-close or leak.
        """

        facts = self._full_candidate_id_mismatch_facts(decision)
        if facts is None:
            return decision, logical_call_id

        recovered_so_far = int(
            getattr(self, "_full_id_mismatch_recovered_count", 0)
        )
        retry_scheduled = recovered_so_far < _FULL_ID_MISMATCH_RECOVERY_CEILING
        # Mint the child call BEFORE the receipt so the receipt can name the
        # call that will carry the surviving verdict; runtime state is then
        # joinable parent -> child without reading provider logs.
        retry_context = (
            full_id_mismatch_retry_context(
                lane_context,
                parent_logical_call_id=logical_call_id,
            )
            if retry_scheduled
            else None
        )
        retry_call_id = (
            str(retry_context.get("logical_call_id") or "")
            if retry_context is not None
            else ""
        )
        self._record_runtime_event(
            search_string=search_string,
            event_type="full_candidate_id_mismatch",
            payload=self._full_id_mismatch_event_payload(
                facts=facts,
                snippet=snippet,
                logical_call_id=logical_call_id,
                retry_logical_call_id=retry_call_id,
                parent_logical_call_id="",
                retry_scheduled=retry_scheduled,
                is_retry_result=False,
                recovered_so_far=recovered_so_far,
            ),
        )
        if retry_context is None:
            print(
                f"    [full-id-mismatch] {snippet.name} — recovery ceiling "
                f"{_FULL_ID_MISMATCH_RECOVERY_CEILING} reached; surfacing the "
                "contract failure"
            )
            return decision, logical_call_id

        print(
            f"    [full-id-mismatch] {snippet.name} — re-asking once under a "
            "fresh candidate ID"
        )
        trace["phase"] = "re_ask"
        retried = full_judge(
            summary,
            self.brief_obj,
            lane_context=retry_context,
            opaque_candidate_id=generate_opaque_candidate_ids(1)[0],
        )
        trace["phase"] = "settle"

        retried_facts = self._full_candidate_id_mismatch_facts(retried)
        if retried_facts is not None:
            # Never re-ask a re-ask: this receipt is the terminal
            # classification of a candidate that missed two unrelated IDs.
            self._record_runtime_event(
                search_string=search_string,
                event_type="full_candidate_id_mismatch",
                payload=self._full_id_mismatch_event_payload(
                    facts=retried_facts,
                    snippet=snippet,
                    logical_call_id=retry_call_id,
                    retry_logical_call_id="",
                    parent_logical_call_id=logical_call_id,
                    retry_scheduled=False,
                    is_retry_result=True,
                    recovered_so_far=recovered_so_far,
                ),
            )
        elif not is_failure_decision(retried.decision):
            self._full_id_mismatch_recovered_count = recovered_so_far + 1
        return retried, retry_call_id

    def _raise_if_full_tool_contract_corruption(
        self,
        *,
        decision: OpusDecision,
        logical_call_id: str,
    ) -> None:
        """Stop the optimized run after two distinct malformed full calls.

        Tool mode has no prose parser or automatic legacy fallback: every
        ``PARSE_FAILURE`` at this boundary is a failed forced-tool contract.
        Count logical calls rather than decision objects so future accounting
        refactors cannot accidentally count the same provider call twice.
        Callers invoke this only after the canonical attempt is closed.
        """

        if (
            config.LINKEDIN_V2_FULL_CONTRACT != "tool"
            or decision.decision != "PARSE_FAILURE"
        ):
            return

        call_id = str(logical_call_id or "").strip()
        if not call_id:
            call_id = str(
                (decision.prompt_capture or {}).get("logical_call_id") or ""
            ).strip()
        if not call_id:
            # Every production full call is assigned an ID before its attempt
            # starts. Keep a bounded fail-safe for tests/older imported calls.
            call_id = f"unattributed-{id(decision)}"

        seen = getattr(self, "_full_contract_corruption_call_ids", None)
        if not isinstance(seen, set):
            seen = set()
            self._full_contract_corruption_call_ids = seen
        if call_id in seen:
            return
        seen.add(call_id)
        self.stats["full_contract_corruptions"] = len(seen)

        if len(seen) < 2:
            return

        console_line = (
            "    [CIRCUIT BREAKER] two full tool-contract corruptions "
            "— stopping optimized run"
        )
        print(console_line, flush=True)
        log_event(
            self.log_path,
            "circuit_breaker",
            stage="full_tool_contract",
            consecutive_errors=len(seen),
            logical_call_id=call_id,
        )
        raise RuntimeError("full tool-contract corruption threshold reached")

    async def _full_evaluate(
        self,
        snippet: CandidateSnippet,
        page_report: "_PageReport | None" = None,
        search_string: SearchString | None = None,
    ) -> Optional[OpusDecision]:
        """Open profile, extract, and run full evaluation. Called after facial triage passes.

        TWIN NOTE: the flag-on pipelined path (_process_pipelined_full_evaluations →
        _open_and_extract / _judge_summary / _complete_full_evaluation) duplicates this
        function's accounting on purpose (A/B staging — plans/c5-judgment-pipelining.md
        addendum). Changes to decision accounting, terminal writes, demotion, or save
        handling must land in BOTH paths until the unification slice retires one.
        """
        runtime_search_string = search_string or SearchString(
            id=snippet.source_string_id,
            name=snippet.source_string_name,
            boolean="",
        )
        full_call_id = f"judge-{uuid.uuid4().hex}"
        # The corruption breaker keys on whichever call produced the decision
        # that actually surfaces; a candidate-ID mismatch re-issue is its own
        # child call and is counted as itself, never as its parent.
        full_breaker_call_id = full_call_id
        full_attempt_payload = {
            "logical_call_id": full_call_id,
            "stage": "full",
        }
        full_attempt_id = self._start_runtime_stage_attempt(
            search_string=runtime_search_string,
            snippet=snippet,
            stage="full",
            payload=full_attempt_payload,
        )

        try:
            self._ensure_services()
            log_event(
                self.log_path,
                "candidate_opened",
                name=snippet.name,
                profile_url=snippet.profile_url,
                string_id=runtime_search_string.id,
            )
            acquisition = await self._acquisition_service.extract_profile_summary(
                snippet, interest=self._read_interest_for(snippet)
            )
            summary = acquisition.profile_summary
        except (OperatorStopRequested, SessionExpired) as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": (
                        "operator_stop"
                        if isinstance(e, OperatorStopRequested)
                        else "session_expired"
                    ),
                    "stage": "full",
                },
            )
            raise
        except GovernorLimitReached as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "governor_limit",
                    "stage": "full",
                },
            )
            raise
        except GeographyRegimeError as e:
            # P3a: extract_profile_summary -> deps.ensure_browser_healthy ->
            # _ensure_browser_healthy can re-navigate after a mid-profile
            # recovery and re-assert the session geography, which is
            # fail-closed. A regime error is a RUN-level abort — converting it
            # to a soft judgment_failure_decision below would leave the rest
            # of this string paging/saving off-geography (the wrapper-swallow
            # mode the gate exists to prevent; correctness lens, Wave 1).
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "geography_regime",
                    "stage": "full",
                },
            )
            raise
        except Exception as e:
            if _is_browser_disconnect_error(e):
                print(f"    [ERROR] Browser session dropped during profile extraction: {e}")
                log_event(self.log_path, "profile_browser_disconnect", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=e,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "browser_disconnect",
                        "stage": "full",
                    },
                )
                raise
            if is_api_budget_exhausted_error(e):
                budget_error = ApiBudgetExhaustedError(str(e))
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=budget_error,
                    payload={
                        **full_attempt_payload,
                        "api_budget_exhausted": True,
                        "stage": "full",
                    },
                )
                raise budget_error from e
            print(f"    [ERROR] Profile extraction failed: {e}")
            log_event(self.log_path, "profile_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            self._finish_runtime_stage_failure(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "profile_extraction_failed": True,
                },
            )
            self._in_flight_urls.discard(snippet.profile_url)
            try:
                await self.browser.go_back_to_results()
            except Exception as cleanup_error:
                if _is_browser_disconnect_error(cleanup_error):
                    raise
            return judgment_failure_decision(
                stage="full",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
                error=e,
                source="profile_extraction",
            )
        except BaseException as e:
            # asyncio cancellation, KeyboardInterrupt, and SystemExit do not
            # inherit from Exception. Close the canonical attempt before the
            # process-level control signal escapes.
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "base_exception_during_profile_extraction",
                    "stage": "full",
                },
            )
            raise

        profile_status_summary: dict = {}
        try:
            profile_status_summary = await self.browser.get_profile_status_summary()
        except Exception as e:
            self._propagate_profile_activity_abort(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload=full_attempt_payload,
            )
            log_event(
                self.log_path,
                "profile_activity_enrichment_failed",
                name=snippet.name,
                error=str(e),
                traceback=traceback.format_exc(),
            )
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "base_exception_during_profile_enrichment",
                    "stage": "full",
                },
            )
            raise

        # --- Shadow profile probe (P12) — records decision; full eval always runs ---
        try:
            probe_decision = self._profile_probe.evaluate(
                summary.to_dict(),
                self._brief_signals_for_profile_probe(),
            )
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "profile_probe_failed",
                    "stage": "full",
                },
            )
            raise

        # --- Final judgment (full-eval tier) ---
        print(f"    Final judgment ({config.FULL_EVAL_MODEL_NAME.rsplit('/', 1)[-1]})...")
        try:
            _lane_ctx_full = {
                **self._lane_context_for_stage(
                    runtime_search_string,
                    stage="full_eval",
                ),
                "logical_call_id": full_call_id,
                "string_id": runtime_search_string.id,
                "page": snippet.page,
            }
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "full_lane_context_failed",
                    "stage": "full",
                },
            )
            raise
        try:
            final = full_judge(summary, brief=self.brief_obj, lane_context=_lane_ctx_full)
            # CLO-177: a successful judgment proves the provider healthy —
            # the containment ceiling counts CONSECUTIVE faults only.
            self._full_eval_consecutive_faults = 0
        except BaseException as e:
            if not isinstance(e, Exception):
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=e,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "base_exception_during_full_judgment",
                        "stage": "full",
                    },
                )
                raise
            if is_api_budget_exhausted_error(e):
                budget_error = ApiBudgetExhaustedError(str(e))
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=budget_error,
                    payload={
                        **full_attempt_payload,
                        "api_budget_exhausted": True,
                        "stage": "full",
                    },
                )
                raise budget_error from e
            if config.LINKEDIN_V2_FULL_CONTRACT == "tool":
                # CLO-177: the tool-contract fail-fast was written for
                # CONTRACT violations, but it also swallowed plain provider
                # timeouts — one hung judgment killed string + session (~13
                # sessions across AIEL/PAE/CISO). A containable
                # provider/transport fault now settles this one candidate as
                # JUDGMENT_FAILURE (fall-through below, same as the legacy
                # contract has always done) and the string continues; the
                # consecutive-fault ceiling keeps a sustained outage legible.
                disposition = self._full_judgment_fault_disposition(e)
                if disposition != "contain":
                    self._abort_runtime_stage_attempt(
                        attempt_id=full_attempt_id,
                        snippet=snippet,
                        error=e,
                        payload={
                            **full_attempt_payload,
                            "full_tool_transport_failed": True,
                            "stage": "full",
                            **(
                                {"full_eval_containment_ceiling": True}
                                if disposition == "ceiling"
                                else {}
                            ),
                        },
                    )
                    if disposition == "ceiling":
                        raise RuntimeError(
                            "full-eval containment ceiling: "
                            f"{config.LINKEDIN_FULL_EVAL_MAX_CONSECUTIVE_FAULTS}"
                            " consecutive judgment provider faults — treating "
                            f"the provider as down (last: {e})"
                        ) from e
                    raise
                print(
                    "    [CONTAINED] full-judgment provider fault "
                    f"({self._full_eval_consecutive_faults}/"
                    f"{config.LINKEDIN_FULL_EVAL_MAX_CONSECUTIVE_FAULTS}): "
                    "settling this candidate as JUDGMENT_FAILURE and "
                    "continuing the string (CLO-177)"
                )
            print(f"    [ERROR] Final judgment failed: {e}")
            log_event(self.log_path, "final_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            final = judgment_failure_decision(
                stage="full",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
                error=e,
                source="judgment",
            )

        # A mis-attributed verdict is intercepted here, before any downstream
        # bookkeeping reads `final`, so only the decision that actually
        # survives reaches the shadow record, the canonical attempt, and the
        # corruption breaker.
        if config.LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED:
            mismatch_trace: dict = {"phase": "classify"}
            try:
                final, full_breaker_call_id = (
                    self._resolve_full_candidate_id_mismatch(
                        decision=final,
                        summary=summary,
                        snippet=snippet,
                        search_string=runtime_search_string,
                        lane_context=_lane_ctx_full,
                        logical_call_id=full_call_id,
                        trace=mismatch_trace,
                    )
                )
            except BaseException as e:
                # The single close site for the whole interception. The
                # resolver deliberately closes nothing, so every failure —
                # a receipt write, the re-ask itself, the second receipt —
                # lands here exactly once and the canonical attempt cannot
                # be stranded in `started` or closed twice.
                abort_payload = {**full_attempt_payload, "stage": "full"}
                budget_exhausted = isinstance(
                    e, Exception
                ) and is_api_budget_exhausted_error(e)
                if budget_exhausted:
                    abort_payload["api_budget_exhausted"] = True
                elif mismatch_trace.get("phase") == "re_ask":
                    abort_payload["full_id_mismatch_retry_failed"] = True
                else:
                    abort_payload["run_abort"] = (
                        "full_id_mismatch_interception_failed"
                    )
                abort_error = (
                    ApiBudgetExhaustedError(str(e)) if budget_exhausted else e
                )
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=abort_error,
                    payload=abort_payload,
                )
                if budget_exhausted:
                    raise abort_error from e
                raise
            if full_breaker_call_id != full_call_id:
                # The surviving verdict came from the re-issue. The attempt's
                # opening call ID stays truthful; the child is named beside
                # it so every close from here on can be joined to the call
                # that actually produced the decision.
                full_attempt_payload["verdict_logical_call_id"] = (
                    full_breaker_call_id
                )

        try:
            shadow_record = self._profile_probe.record_shadow_outcome(
                snippet.name,
                snippet.profile_url,
                runtime_search_string.lane_id or "",
                probe_decision,
                final.decision,
                final.confidence,
            )
            # P10 actuate #2: make the shadow evidence real by persisting it to the
            # runtime DB events channel at this checkpoint. No reader is wired yet
            # (cascade activation stays OFF) — same unwired-shadow-evidence class as
            # the "shadow_full_judge_completed" event above; ad-hoc query against
            # runtime_state.sqlite3 is the only consumer until D8 activation lands.
            # Recovery replay re-judges the interrupted candidate and appends a
            # second cascade_shadow_recorded row, so any future activation feed
            # must dedup on profile_url before computing AuditSampler metrics.
            self._record_runtime_event(
                search_string=runtime_search_string,
                event_type="cascade_shadow_recorded",
                payload=shadow_record.to_dict(),
            )

            novelty_value, value_rationale = self._derive_novelty_value(
                snippet,
                profile_status_summary=profile_status_summary,
            )
            final.novelty_value = novelty_value
            final.value_rationale = value_rationale
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "full_precommit_bookkeeping_failed",
                    "stage": "full",
                },
            )
            raise

        # Intercept parse/judgment failures — do NOT persist to cross-session history
        if is_failure_decision(final.decision):
            print(f"    [{final.decision}] {final.rationale}")
            self.stats.setdefault("parse_failures", 0)
            self.stats["parse_failures"] += 1
            # Canonical closure precedes optional bias bookkeeping so an
            # observability defect cannot strand a paid full-eval attempt.
            self._finish_runtime_failure_decision(
                attempt_id=full_attempt_id,
                snippet=snippet,
                decision=final,
                payload={
                    **full_attempt_payload,
                    "profile_summary": summary.to_dict(),
                },
            )
            self._in_flight_urls.discard(snippet.profile_url)
            if self._bias_monitor:
                self._bias_monitor.record_decision(DecisionRecord(
                    candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                    string_id=str(snippet.source_string_id),
                    stage="full",
                    decision=final.decision,
                    confidence=final.confidence,
                    capability_area=None,
                ))
            try:
                await self.browser.go_back_to_results()
                await asyncio.sleep(
                    human_delay_correlated(
                        config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS,
                        channel="panel_close",
                    )
                )
            except Exception:
                final._panel_stuck = True
            self._raise_if_full_tool_contract_corruption(
                decision=final,
                logical_call_id=full_breaker_call_id,
            )
            return final

        # P4: structural-evidence guard for bounded non-save review
        # outcomes. Must run BEFORE the canonical terminal write so the
        # SQLite row reflects the demoted decision; the dispatch site
        # below only handles side-effects, stats, and logging.
        review_demotion_reason = review_decision_demotion_reason(final)
        if review_demotion_reason:
            original_review_decision = final.decision
            log_event(
                self.log_path,
                "candidate_review_recorded",
                name=snippet.name,
                profile_url=snippet.profile_url,
                string_id=runtime_search_string.id,
                original_decision=original_review_decision,
                decision="REJECT",
                demoted=True,
                reason=review_demotion_reason,
                lane_id=runtime_search_string.lane_id,
            )
            print(
                f"    [REVIEW_DEMOTED] {original_review_decision} -> REJECT "
                f"({review_demotion_reason})"
            )
            self.stats["reviewed_demoted"] += 1
            _clear_review_evidence(final)
            final.decision = "REJECT"
            final.outreach_tier = ""
            final.reject_reason = "CAPABILITY_INSUFFICIENT"

        save_decision = final.decision in SAVE_FAMILY_DECISIONS
        if not save_decision:
            self._prior_outcomes[snippet.profile_url] = final.decision
            self._mark_terminal(snippet.profile_url)
            # REVIEW outcomes carry lane attribution into terminal payloads.
            # SAVE terminalization is delayed until its side effect settles.
            finish_extra_payload: dict | None = dict(full_attempt_payload)
            if final.decision in NON_SAVE_REVIEW_DECISIONS:
                lane_payload = {
                    "lane_id": runtime_search_string.lane_id,
                    "lane_name": runtime_search_string.lane_name,
                    "lane_intent": runtime_search_string.lane_intent,
                }
                lane_payload = {k: v for k, v in lane_payload.items() if v}
                if lane_payload:
                    finish_extra_payload["lane"] = lane_payload
            self._finish_runtime_stage_success(
                attempt_id=full_attempt_id,
                stage="full",
                snippet=snippet,
                decision=final,
                profile_summary=summary,
                extra_payload=finish_extra_payload,
            )

        # Record full eval decision and check bias alerts
        if self._bias_monitor:
            self._bias_monitor.record_decision(DecisionRecord(
                candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                string_id=str(snippet.source_string_id),
                stage="full",
                decision=final.decision,
                confidence=final.confidence,
                capability_area=final.path if final.path != "none" else None,
            ))
            alerts = self._bias_monitor.check_alerts(str(snippet.source_string_id))
            for alert in alerts:
                # Telemetry demotion (2026-07-04 SPL run): bias alerts are
                # observations, never control flow. Every severity is
                # printed AND persisted uniformly; the adaptation model
                # receives the per-string context via the block report
                # (string_context) and owns any skip/pivot decision.
                symbol = {"flag": "⚡", "info": "ℹ"}.get(alert.severity, "⚡")
                print(f"    {symbol} BIAS {alert.severity.upper()}: {alert.message}")
                log_event(self.log_path, "bias_alert", severity=alert.severity,
                          alert_type=alert.alert_type, message=alert.message,
                          string_id=alert.string_id)

        # --- Shadow: external evidence augmentation (analytical-debug only) ---
        # Placed AFTER baseline canonical persistence (_finish_runtime_stage_success,
        # _mark_terminal, _prior_outcomes write, bias_monitor record) so that any
        # failure here is mechanically incapable of affecting baseline truth.
        # Disabled by default via config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED.
        try:
            if config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED and not is_failure_decision(final.decision):
                trigger = should_request_external_evidence(summary=summary, brief=self.brief_obj)
                external_evidence_status = ""
                evidence: ExternalCandidateEvidence | None = None
                enriched: OpusDecision | None = None

                if not trigger.should_run:
                    external_evidence_status = "skipped_no_trigger"
                    log_event(
                        self.log_path,
                        "external_evidence_skipped",
                        name=snippet.name,
                        profile_url=snippet.profile_url,
                        skip_reason=trigger.skip_reason,
                        signals=trigger.signals,
                    )
                    self._record_runtime_event(
                        search_string=runtime_search_string,
                        event_type="external_evidence_skipped",
                        payload={
                            "profile_url": snippet.profile_url,
                            "skip_reason": trigger.skip_reason,
                            "signals": trigger.signals,
                        },
                    )
                else:
                    identity_hints = {
                        "name": snippet.name,
                        "current_company": snippet.current_company,
                        "current_title": snippet.current_title,
                        "headline": snippet.headline,
                        "education_snippet": snippet.education_snippet,
                        "profile_url": snippet.profile_url,
                    }
                    result = fetch_external_candidate_evidence(
                        summary=summary,
                        brief=self.brief_obj,
                        trigger=trigger,
                        identity_hints=identity_hints,
                    )
                    if isinstance(result, ExternalCandidateEvidence):
                        evidence = result
                        external_evidence_status = "evidence_present"
                        log_event(
                            self.log_path,
                            "external_evidence_fetched",
                            name=snippet.name,
                            profile_url=snippet.profile_url,
                            trigger_reason=trigger.reason,
                            identity_confidence=evidence.identity_confidence,
                            fact_blocks=len(evidence.external_fact_blocks),
                            inferences=len(evidence.external_inferences),
                        )
                        self._record_runtime_event(
                            search_string=runtime_search_string,
                            event_type="external_evidence_fetched",
                            payload={
                                "profile_url": snippet.profile_url,
                                "trigger_reason": trigger.reason,
                                "identity_confidence": evidence.identity_confidence,
                            },
                        )
                        try:
                            enriched = full_judge_with_external_evidence(
                                summary,
                                evidence,
                                brief=self.brief_obj,
                                lane_context=_lane_ctx_full,
                            )
                        except Exception as enrich_exc:
                            enriched = None
                            log_event(
                                self.log_path,
                                "external_evidence_enriched_judge_failed",
                                name=snippet.name,
                                error=str(enrich_exc),
                                traceback=traceback.format_exc(),
                            )
                    else:  # ExternalEvidenceFailure
                        external_evidence_status = result.reason
                        log_event(
                            self.log_path,
                            "external_evidence_failed",
                            name=snippet.name,
                            profile_url=snippet.profile_url,
                            reason=result.reason,
                            detail=result.detail,
                            http_status=result.http_status,
                        )
                        self._record_runtime_event(
                            search_string=runtime_search_string,
                            event_type="external_evidence_failed",
                            payload={
                                "profile_url": snippet.profile_url,
                                "reason": result.reason,
                                "http_status": result.http_status,
                            },
                        )

                diff = compute_judgment_diff(
                    final,
                    enriched,
                    skip_reason=external_evidence_status if enriched is None else "",
                )
                evidence_refs_count = 0
                identity_confidence = None
                if evidence is not None:
                    identity_confidence = evidence.identity_confidence
                    evidence_refs_count = sum(len(b.evidence_refs) for b in evidence.external_fact_blocks)
                    evidence_refs_count += sum(len(i.basis_refs) for i in evidence.external_inferences)

                record_shadow_full_judgment(
                    output_dir=self.output_dir,
                    record=ShadowFullJudgmentRecord(
                        candidate_name=snippet.name,
                        profile_url=snippet.profile_url,
                        source_string_id=snippet.source_string_id,
                        page=snippet.page,
                        result_rank=snippet.result_rank,
                        trigger_reason=trigger.reason if trigger.should_run else "",
                        external_evidence_status=external_evidence_status,
                        identity_confidence=identity_confidence,
                        evidence_refs_count=evidence_refs_count,
                        baseline=final.to_dict(),
                        enriched=enriched.to_dict() if enriched is not None else None,
                        diff=diff,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                )

                if enriched is not None:
                    self._record_runtime_event(
                        search_string=runtime_search_string,
                        event_type="shadow_full_judge_completed",
                        payload={
                            "profile_url": snippet.profile_url,
                            "decision_changed": diff.get("decision_changed"),
                            "path_changed": diff.get("path_changed"),
                            "rationale_changed": diff.get("rationale_changed"),
                            "confidence_delta": diff.get("confidence_delta"),
                            "evidence_refs_count": evidence_refs_count,
                        },
                    )
        except Exception as shadow_exc:
            # The whole shadow path is best-effort. Nothing here may affect baseline.
            print(f"    [WARN] shadow external-evidence eval failed: {shadow_exc}")
            log_event(
                self.log_path,
                "external_evidence_shadow_unhandled_exception",
                name=snippet.name,
                error=str(shadow_exc),
                traceback=traceback.format_exc(),
            )

        if final.decision in SAVE_FAMILY_DECISIONS:
            tag = final.decision if final.decision in ("INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE") else "SAVE"
            print(f"    [{tag}] {final.rationale}")
            self.stats["save_attempts"] += 1
            if final.novelty_value == "low":
                self.stats["high_fit_low_novelty_saves"] += 1
            self._record_outreach_tier_outcome(
                snippet=snippet,
                decision=final,
            )
            # P1.2: consume the side-effect outcome instead of discarding
            # it. handle_save_decision emits the one honest candidate_saved
            # event (linkedin_save flag + failure_reason); the flag-less
            # duplicate that used to be emitted here shadowed it for every
            # consumer (live feed, monitors, report).
            try:
                await self._reopen_profile_for_full_eval_save(snippet)
                outcome = await self._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=runtime_search_string,
                    attempt_id=full_attempt_id,
                )
            except BaseException as exc:
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=exc,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "save_side_effect_interrupted",
                        "stage": "full",
                    },
                )
                raise
            outcome_payload = dict(outcome.payload or {})
            skip_reason = str(outcome_payload.get("skip_reason") or "")
            persisted = outcome.status == "succeeded"
            already_present = bool(outcome_payload.get("already_present")) or (
                outcome.status == "skipped"
                and skip_reason == "existing_succeeded"
            )
            final.save_outcome = {
                "status": outcome.status,
                "persisted": persisted,
                "already_present": already_present,
                "failure_reason": outcome_payload.get("failure_reason")
                or (skip_reason or None if not persisted and not already_present else None),
            }
            if outcome_payload.get("reconciled_self_save"):
                final.save_outcome["reconciled_self_save"] = True
            if not (persisted or already_present):
                save_error = RuntimeError(
                    "LinkedIn save was not durably confirmed"
                )
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=save_error,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "save_not_confirmed",
                        "save_outcome": final.save_outcome,
                        "stage": "full",
                    },
                )
                raise save_error

            self._prior_outcomes[snippet.profile_url] = final.decision
            self._mark_terminal(snippet.profile_url)
            self._finish_runtime_stage_success(
                attempt_id=full_attempt_id,
                stage="full",
                snippet=snippet,
                decision=final,
                profile_summary=summary,
                extra_payload=dict(full_attempt_payload),
            )

            if page_report:
                page_report.add_saved(snippet, final)
        elif final.decision in NON_SAVE_REVIEW_DECISIONS:
            # P4: bounded non-save review outcome. Cloris is preserving a
            # candidate a strong sourcer would not discard, without
            # recommending a save. MUST NOT call handle_save_decision
            # (no LinkedIn Save-to-pipeline click) and MUST NOT increment
            # save-class counters. The structural-evidence guard already
            # ran upstream of the canonical write; demoted candidates
            # land in the ``else`` REJECT branch below with the demotion
            # event already recorded by the upstream guard.
            print(f"    [{final.decision}] {final.rationale}")
            self.stats["reviewed"] += 1
            if final.decision == "REVIEW_INFERRED":
                self.stats["reviewed_inferred"] += 1
            else:
                self.stats["reviewed_flagged"] += 1
            # Reuse the skipped-opened page-report bucket for now;
            # a dedicated ``add_review()`` lane-level surface is P5
            # work (lane metrics).
            if page_report:
                page_report.add_skipped_opened(snippet, final)
            log_event(
                self.log_path,
                "candidate_review_recorded",
                name=snippet.name,
                profile_url=snippet.profile_url,
                string_id=runtime_search_string.id,
                decision=final.decision,
                review_reason_code=final.review_reason_code,
                lane_id=runtime_search_string.lane_id,
                structural_evidence_count=len(
                    final.review_structural_evidence
                ),
            )
            review_dwell = max(
                config.LINKEDIN_REJECT_CLOSE_MIN_SECONDS,
                min(
                    config.LINKEDIN_REJECT_CLOSE_MAX_SECONDS,
                    human_delay_correlated(
                        config.LINKEDIN_REJECT_CLOSE_BASE_SECONDS,
                        channel="reject_close",
                    ),
                ),
            )
            await asyncio.sleep(review_dwell)
            print(
                f"    [profile-read] {final.decision} verdict → "
                f"closing in {review_dwell:.1f}s (no save click)"
            )
        else:
            print(f"    [REJECT] {final.rationale}")
            self.stats["rejected"] += 1
            if page_report:
                page_report.add_skipped_opened(snippet, final)
            # Quick exit — saw enough, moving on
            reject_dwell = max(
                config.LINKEDIN_REJECT_CLOSE_MIN_SECONDS,
                min(
                    config.LINKEDIN_REJECT_CLOSE_MAX_SECONDS,
                    human_delay_correlated(
                        config.LINKEDIN_REJECT_CLOSE_BASE_SECONDS,
                        channel="reject_close",
                    ),
                ),
            )
            await asyncio.sleep(reject_dwell)
            print(f"    [profile-read] REJECT verdict → closing in {reject_dwell:.1f}s")

        try:
            await self.browser.go_back_to_results()
            await asyncio.sleep(
                human_delay_correlated(
                    config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS,
                    channel="panel_close",
                )
            )
        except Exception as e:
            if _is_browser_disconnect_error(e):
                print(f"    [ERROR] Browser session dropped while closing profile: {e}")
                log_event(self.log_path, "panel_close_browser_disconnect", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                raise
            print("    [WARN] Profile panel close missed; attempting recovery")
            log_event(self.log_path, "go_back_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            # Tag the result so the page loop knows state is corrupted
            final._panel_stuck = True

        return final

    async def _process_resumed_pending_full_evaluations(
        self,
        *,
        snippets: list[CandidateSnippet],
        page_report: "_PageReport | None",
        search_string: SearchString,
        all_candidates: list[dict],
        string_stats: dict[str, int],
        progress: Progress | None,
        page_num: int,
        preserve_progress_cursor: bool = False,
    ) -> bool:
        """Finish canonical facial positives without issuing another facial call."""

        checkpoint_kwargs = (
            {}
            if preserve_progress_cursor
            else {
                "search_string": search_string,
                "page_num": page_num,
            }
        )
        checkpoint_search_string = (
            None if preserve_progress_cursor else search_string
        )
        for snippet in snippets:
            self._note_page_full_review_expected(snippet)

        if config.FULL_EVAL_PIPELINE_ENABLED:
            decisions = await self._process_pipelined_full_evaluations(
                facial_yes_snippets=snippets,
                page_report=page_report,
                search_string=search_string,
                all_candidates=all_candidates,
                string_stats=string_stats,
                progress=progress,
                page_num=page_num,
                preserve_progress_cursor=preserve_progress_cursor,
            )
            # The pipelined processor either recovers every transient marker or
            # raises PanelRecoveryError; a returned queue is fully drained.
            return False

        for snippet in snippets:
            decision = await self._full_evaluate(
                snippet,
                page_report,
                search_string,
            )
            self._apply_pipelined_full_eval_page_outcome(
                decision=decision,
                snippet=snippet,
                page_num=page_num,
                all_candidates=all_candidates,
                string_stats=string_stats,
                search_string=search_string,
            )
            self._checkpoint_progress(progress, **checkpoint_kwargs)
            if getattr(decision, "_panel_stuck", False):
                await self._recover_stuck_profile_panel(
                    candidate_name=snippet.name,
                    page_num=page_num,
                    decision=decision,
                    progress=progress,
                    search_string=checkpoint_search_string,
                )
        return False

    async def _process_pipelined_full_evaluations(
        self,
        *,
        facial_yes_snippets: list[CandidateSnippet],
        page_report: "_PageReport | None",
        search_string: SearchString,
        all_candidates: list[dict],
        string_stats: dict[str, int],
        progress: Progress | None,
        page_num: int,
        preserve_progress_cursor: bool = False,
    ) -> list[OpusDecision]:
        """Run full eval with one browser-phase lookahead and FIFO completion."""
        pending: dict[str, Any] | None = None
        decisions: list[OpusDecision] = []
        full_timing_samples: list[dict[str, float]] = []
        consecutive_api_errors = 0
        checkpoint_kwargs = (
            {}
            if preserve_progress_cursor
            else {
                "search_string": search_string,
                "page_num": page_num,
            }
        )
        checkpoint_search_string = (
            None if preserve_progress_cursor else search_string
        )

        async def complete_pending() -> OpusDecision:
            nonlocal pending, consecutive_api_errors
            assert pending is not None
            completed = pending
            # Transfer ownership to this completion call before awaiting. Every
            # exception path inside _complete_full_evaluation now closes the
            # canonical attempt, so retaining the dict would invite a second
            # finish attempt during outer cleanup.
            pending = None
            decision = await self._complete_full_evaluation(
                eval_context=completed["context"],
                judgment_task=completed["task"],
                page_report=page_report,
                timing_sink=full_timing_samples,
            )
            decisions.append(decision)
            consecutive_api_errors = (
                consecutive_api_errors + 1
                if self._is_recoverable_provider_failure_decision(decision)
                else 0
            )
            self._apply_pipelined_full_eval_page_outcome(
                decision=decision,
                snippet=completed["context"]["snippet"],
                page_num=page_num,
                all_candidates=all_candidates,
                string_stats=string_stats,
                search_string=search_string,
            )
            self._checkpoint_progress(progress, **checkpoint_kwargs)
            return decision

        def stop_requested() -> bool:
            return bool(
                progress
                and hasattr(self, "_operator_stop_event")
                and self._operator_stop_event.is_set()
            )

        def session_expired() -> bool:
            return bool(
                progress
                and hasattr(self, "_session_expired")
                and self._session_expired.is_set()
            )

        async def drain_before_stop(exc: Exception) -> None:
            if pending is not None:
                await complete_pending()
            self._checkpoint_progress(progress)
            if isinstance(exc, OperatorStopRequested):
                print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                self._set_page_break_reason("operator_stop")
            elif isinstance(exc, SessionExpired):
                self._set_page_break_reason("session_expired")
            raise exc

        async def cleanup_iteration_abort(
            *,
            opened_context: dict[str, Any] | None,
            cause: BaseException,
        ) -> None:
            """Close unspawned work and settle any paid worker before escape."""

            cleanup_errors: list[BaseException] = []
            if opened_context is not None:
                try:
                    self._finish_unspawned_full_eval_context(
                        opened_context,
                        reason=f"iteration_abort:{type(cause).__name__}",
                    )
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)

            if pending is not None:
                try:
                    await complete_pending()
                except BaseException as cleanup_error:
                    # _complete_full_evaluation closes its attempt before
                    # raising. Preserve the initiating failure while recording
                    # compact cleanup diagnostics; never print raw provider data.
                    cleanup_errors.append(cleanup_error)

            for cleanup_error in cleanup_errors:
                log_event(
                    self.log_path,
                    "full_pipeline_abort_cleanup_failed",
                    cause_type=type(cause).__name__,
                    cleanup_error_type=type(cleanup_error).__name__,
                )

        for snippet in facial_yes_snippets:
            if session_expired():
                await drain_before_stop(SessionExpired("session_duration_cap"))
            if stop_requested():
                await drain_before_stop(OperatorStopRequested())

            opened_context: dict[str, Any] | None = None
            try:
                opened = await self._open_and_extract(
                    snippet,
                    page_report,
                    search_string,
                )
                opened_context = opened.get("context")

                if pending is not None:
                    previous = await complete_pending()
                    if getattr(previous, "_panel_stuck", False):
                        await self._recover_stuck_profile_panel(
                            candidate_name=previous.candidate_name,
                            page_num=page_num,
                            decision=previous,
                            progress=progress,
                            search_string=checkpoint_search_string,
                        )

                if "decision" in opened:
                    # _open_and_extract already closed this candidate's attempt.
                    opened_context = None
                    decision = opened["decision"]
                    decisions.append(decision)
                    consecutive_api_errors = (
                        consecutive_api_errors + 1
                        if self._is_recoverable_provider_failure_decision(decision)
                        else 0
                    )
                    self._apply_pipelined_full_eval_page_outcome(
                        decision=decision,
                        snippet=snippet,
                        page_num=page_num,
                        all_candidates=all_candidates,
                        string_stats=string_stats,
                        search_string=search_string,
                    )
                    self._checkpoint_progress(progress, **checkpoint_kwargs)
                    if getattr(decision, "_panel_stuck", False):
                        await self._recover_stuck_profile_panel(
                            candidate_name=snippet.name,
                            page_num=page_num,
                            decision=decision,
                            progress=progress,
                            search_string=checkpoint_search_string,
                        )
                    # Same breaker as the pre-spawn site: without it, a run of
                    # extraction-phase API failures `continue`s past that check
                    # forever and the 60s pause never fires.
                    if consecutive_api_errors >= 5:
                        print(
                            f"    [CIRCUIT BREAKER] {consecutive_api_errors} "
                            "consecutive API failures — pausing 60s"
                        )
                        log_event(
                            self.log_path,
                            "circuit_breaker",
                            consecutive_errors=consecutive_api_errors,
                        )
                        await asyncio.sleep(60)
                        consecutive_api_errors = 0
                    continue

                # Fresh-count breaker, deliberately NOT at loop top: pipelining
                # completes judgment K-1 mid-iteration K, so a top-of-loop check
                # reads a count one completion stale and can miss the threshold
                # entirely at queue end. Here the count is current and the pause
                # lands before the next paid judgment call.
                if consecutive_api_errors >= 5:
                    print(
                        f"    [CIRCUIT BREAKER] {consecutive_api_errors} "
                        "consecutive API failures — pausing 60s"
                    )
                    log_event(
                        self.log_path,
                        "circuit_breaker",
                        consecutive_errors=consecutive_api_errors,
                    )
                    await asyncio.sleep(60)
                    consecutive_api_errors = 0

                assert opened_context is not None
                pending = {
                    "context": opened_context,
                    "task": asyncio.create_task(
                        asyncio.to_thread(
                            self._judge_summary,
                            summary=opened_context["summary"],
                            snippet=snippet,
                            lane_context=opened_context["lane_context"],
                        )
                    ),
                }
                # The pending task now owns this attempt. Outer cleanup must not
                # also mark it as an unspawned context.
                opened_context = None

                if (
                    progress
                    and hasattr(self, "_pause_requested")
                    and self._pause_requested.is_set()
                ):
                    self._pause_requested.clear()
                    self._checkpoint_progress(progress)
                    try:
                        await asyncio.wait_for(
                            self._resume_event.wait(),
                            timeout=300,
                        )
                    except asyncio.TimeoutError:
                        print(
                            "  [!] Resume timeout (5 min) — continuing without "
                            "decoy burst."
                        )
                        self._resume_event.set()
            except BaseException as exc:
                await cleanup_iteration_abort(
                    opened_context=opened_context,
                    cause=exc,
                )
                raise

        if pending is not None:
            decision = await complete_pending()
            if getattr(decision, "_panel_stuck", False):
                await self._recover_stuck_profile_panel(
                    candidate_name=decision.candidate_name,
                    page_num=page_num,
                    decision=decision,
                    progress=progress,
                    search_string=checkpoint_search_string,
                )

        if full_timing_samples:
            logical_call_elapsed_ms = [
                round(sample["logical_call_elapsed_ms"], 3)
                for sample in full_timing_samples
            ]
            full_timing_payload = {
                "string_id": search_string.id,
                "page": page_num,
                "full_call_count": len(full_timing_samples),
                "logical_call_elapsed_ms": logical_call_elapsed_ms,
                "logical_call_elapsed_total_ms": round(
                    sum(logical_call_elapsed_ms), 3
                ),
                "operator_blocking_elapsed_ms": round(
                    sum(
                        sample["operator_blocking_elapsed_ms"]
                        for sample in full_timing_samples
                    ),
                    3,
                ),
                "pipeline": "lookahead_one",
                "browser_extraction_excluded": True,
            }
            # Timing is diagnostic and cannot change a verdict. A bounded
            # canary whose sinks lose this receipt fails postflight instead of
            # turning observability into live control flow.
            try:
                self._record_runtime_event(
                    search_string=search_string,
                    event_type="full_page_judgment_timing",
                    payload=full_timing_payload,
                )
                log_event(
                    self.log_path,
                    "full_page_judgment_timing",
                    **full_timing_payload,
                )
            except Exception:
                pass

        if session_expired():
            self._checkpoint_progress(progress)
            self._set_page_break_reason("session_expired")
            raise SessionExpired("session_duration_cap")
        if stop_requested():
            self._checkpoint_progress(progress)
            print("  [!] Operator stop honored at string/page boundary. Progress saved.")
            self._set_page_break_reason("operator_stop")
            raise OperatorStopRequested()

        return decisions

    async def _open_and_extract(
        self,
        snippet: CandidateSnippet,
        page_report: "_PageReport | None" = None,
        search_string: SearchString | None = None,
    ) -> dict[str, Any]:
        runtime_search_string = search_string or SearchString(
            id=snippet.source_string_id,
            name=snippet.source_string_name,
            boolean="",
        )
        full_call_id = f"judge-{uuid.uuid4().hex}"
        full_attempt_payload = {
            "logical_call_id": full_call_id,
            "stage": "full",
        }
        full_attempt_id = self._start_runtime_stage_attempt(
            search_string=runtime_search_string,
            snippet=snippet,
            stage="full",
            payload=full_attempt_payload,
        )

        try:
            self._ensure_services()
            log_event(
                self.log_path,
                "candidate_opened",
                name=snippet.name,
                profile_url=snippet.profile_url,
                string_id=runtime_search_string.id,
            )
            acquisition = await self._acquisition_service.extract_profile_summary(
                snippet, interest=self._read_interest_for(snippet)
            )
            summary = acquisition.profile_summary
        except (OperatorStopRequested, SessionExpired) as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": (
                        "operator_stop"
                        if isinstance(e, OperatorStopRequested)
                        else "session_expired"
                    ),
                    "stage": "full",
                },
            )
            raise
        except GovernorLimitReached as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "governor_limit",
                    "stage": "full",
                },
            )
            raise
        except GeographyRegimeError as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "geography_regime",
                    "stage": "full",
                },
            )
            raise
        except Exception as e:
            if _is_browser_disconnect_error(e):
                print(f"    [ERROR] Browser session dropped during profile extraction: {e}")
                log_event(self.log_path, "profile_browser_disconnect", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=e,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "browser_disconnect",
                        "stage": "full",
                    },
                )
                raise
            if is_api_budget_exhausted_error(e):
                budget_error = ApiBudgetExhaustedError(str(e))
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=budget_error,
                    payload={
                        **full_attempt_payload,
                        "api_budget_exhausted": True,
                        "stage": "full",
                    },
                )
                raise budget_error from e
            print(f"    [ERROR] Profile extraction failed: {e}")
            log_event(self.log_path, "profile_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            self._finish_runtime_stage_failure(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "profile_extraction_failed": True,
                },
            )
            self._in_flight_urls.discard(snippet.profile_url)
            try:
                await self.browser.go_back_to_results()
            except Exception as cleanup_error:
                if _is_browser_disconnect_error(cleanup_error):
                    raise
            return {
                "decision": judgment_failure_decision(
                    stage="full",
                    candidate_name=snippet.name,
                    profile_url=snippet.profile_url,
                    error=e,
                    source="profile_extraction",
                )
            }
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "base_exception_during_profile_extraction",
                    "stage": "full",
                },
            )
            raise

        profile_status_summary: dict = {}
        try:
            profile_status_summary = await self.browser.get_profile_status_summary()
        except Exception as e:
            self._propagate_profile_activity_abort(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload=full_attempt_payload,
            )
            log_event(
                self.log_path,
                "profile_activity_enrichment_failed",
                name=snippet.name,
                error=str(e),
                traceback=traceback.format_exc(),
            )
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "base_exception_during_profile_enrichment",
                    "stage": "full",
                },
            )
            raise

        try:
            probe_decision = self._profile_probe.evaluate(
                summary.to_dict(),
                self._brief_signals_for_profile_probe(),
            )
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "profile_probe_failed",
                    "stage": "full",
                },
            )
            raise
        try:
            lane_context = {
                **self._lane_context_for_stage(
                    runtime_search_string,
                    stage="full_eval",
                ),
                "logical_call_id": full_call_id,
                "string_id": runtime_search_string.id,
                "page": snippet.page,
            }
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "full_lane_context_failed",
                    "stage": "full",
                },
            )
            raise

        try:
            await self.browser.go_back_to_results()
            await asyncio.sleep(
                human_delay_correlated(
                    config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS,
                    channel="panel_close",
                )
            )
        except Exception as e:
            if _is_browser_disconnect_error(e):
                print(f"    [ERROR] Browser session dropped while closing profile: {e}")
                log_event(self.log_path, "panel_close_browser_disconnect", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                self._finish_runtime_stage_failure(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=e,
                    payload={
                        **full_attempt_payload,
                        "panel_close_failed": True,
                    },
                )
                self._in_flight_urls.discard(snippet.profile_url)
                raise
            print("    [WARN] Profile panel close missed; attempting recovery")
            log_event(self.log_path, "go_back_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            try:
                await self._recover_stuck_profile_panel(
                    candidate_name=snippet.name,
                    page_num=snippet.page,
                )
            except BaseException as recovery_error:
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=recovery_error,
                    payload={
                        **full_attempt_payload,
                        "panel_close_failed": True,
                        "run_abort": "panel_recovery_failed",
                        "stage": "full",
                    },
                )
                raise
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "base_exception_during_panel_close",
                    "stage": "full",
                },
            )
            raise

        return {
            "context": {
                "snippet": snippet,
                "runtime_search_string": runtime_search_string,
                "attempt_id": full_attempt_id,
                "attempt_payload": full_attempt_payload,
                "summary": summary,
                "profile_status_summary": profile_status_summary,
                "probe_decision": probe_decision,
                "lane_context": lane_context,
            }
        }

    def _judge_summary(
        self,
        *,
        summary: CandidateProfileSummary,
        snippet: CandidateSnippet,
        lane_context: dict | None,
    ) -> dict[str, Any]:
        judgment_started = time.monotonic()
        final = full_judge(summary, brief=self.brief_obj, lane_context=lane_context)
        judgment_elapsed_ms = round(
            (time.monotonic() - judgment_started) * 1000.0,
            3,
        )
        external_shadow: dict[str, Any] | None = None
        external_shadow_error: dict[str, str] | None = None

        if config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED and not is_failure_decision(final.decision):
            try:
                trigger = should_request_external_evidence(summary=summary, brief=self.brief_obj)
                external_evidence_status = ""
                evidence: ExternalCandidateEvidence | None = None
                enriched: OpusDecision | None = None
                enrich_error: Exception | None = None
                result: ExternalCandidateEvidence | ExternalEvidenceFailure | None = None

                if not trigger.should_run:
                    external_evidence_status = "skipped_no_trigger"
                else:
                    identity_hints = {
                        "name": snippet.name,
                        "current_company": snippet.current_company,
                        "current_title": snippet.current_title,
                        "headline": snippet.headline,
                        "education_snippet": snippet.education_snippet,
                        "profile_url": snippet.profile_url,
                    }
                    result = fetch_external_candidate_evidence(
                        summary=summary,
                        brief=self.brief_obj,
                        trigger=trigger,
                        identity_hints=identity_hints,
                    )
                    if isinstance(result, ExternalCandidateEvidence):
                        evidence = result
                        external_evidence_status = "evidence_present"
                        try:
                            enriched = full_judge_with_external_evidence(
                                summary,
                                evidence,
                                brief=self.brief_obj,
                                lane_context=lane_context,
                            )
                        except Exception as exc:
                            # Augmentation is analytical shadow work. Even auth,
                            # budget, or provider defects here must not replace a
                            # completed baseline judgment.
                            enrich_error = exc
                    else:
                        external_evidence_status = result.reason

                diff = compute_judgment_diff(
                    final,
                    enriched,
                    skip_reason=external_evidence_status if enriched is None else "",
                )
                evidence_refs_count = 0
                identity_confidence = None
                if evidence is not None:
                    identity_confidence = evidence.identity_confidence
                    evidence_refs_count = sum(len(b.evidence_refs) for b in evidence.external_fact_blocks)
                    evidence_refs_count += sum(len(i.basis_refs) for i in evidence.external_inferences)

                external_shadow = {
                    "trigger": trigger,
                    "status": external_evidence_status,
                    "result": result,
                    "evidence": evidence,
                    "enriched": enriched,
                    "enrich_error": enrich_error,
                    "diff": diff,
                    "identity_confidence": identity_confidence,
                    "evidence_refs_count": evidence_refs_count,
                }
            except Exception as exc:
                # The baseline is already valid. Keep all optional evidence
                # setup/fetch/diff failures on the shadow side of the boundary
                # and hand a compact diagnostic back to the event-loop thread.
                external_shadow_error = {
                    "type": type(exc).__name__,
                    "message": str(exc)[:240],
                }

        return {
            "final": final,
            "judgment_elapsed_ms": judgment_elapsed_ms,
            "external_shadow": external_shadow,
            "external_shadow_error": external_shadow_error,
        }

    async def _complete_full_evaluation(
        self,
        *,
        eval_context: dict[str, Any],
        judgment_task: "asyncio.Task[dict[str, Any]]",
        page_report: "_PageReport | None" = None,
        timing_sink: list[dict[str, float]] | None = None,
    ) -> OpusDecision:
        # TWIN of _full_evaluate's post-judge accounting (serial path). The two
        # copies are deliberate A/B staging until the pipelined path proves a
        # live session (plans/c5-judgment-pipelining.md addendum) — any change
        # to decision accounting, terminal writes, demotion, or save handling
        # must land in BOTH until the unification slice retires one.
        snippet: CandidateSnippet = eval_context["snippet"]
        runtime_search_string: SearchString = eval_context["runtime_search_string"]
        full_attempt_id: int | None = eval_context["attempt_id"]
        full_attempt_payload = dict(eval_context.get("attempt_payload") or {})
        summary: CandidateProfileSummary = eval_context["summary"]
        profile_status_summary: dict = eval_context["profile_status_summary"]
        probe_decision = eval_context["probe_decision"]

        print(f"    Final judgment ({config.FULL_EVAL_MODEL_NAME.rsplit('/', 1)[-1]})...")
        blocking_wait_started = time.monotonic()
        try:
            # Shield the to_thread task: cancelling the coordinator cannot stop
            # an in-flight provider request, so cancelling the wrapper would
            # merely hide a still-billable worker. The cancellation handler
            # below drains the real task before closing canonical state.
            judged = await asyncio.shield(judgment_task)
            final: OpusDecision = judged["final"]
            external_shadow = judged.get("external_shadow")
            external_shadow_error = judged.get("external_shadow_error")
            if timing_sink is not None:
                try:
                    judgment_elapsed_ms = float(judged["judgment_elapsed_ms"])
                    blocking_wait_elapsed_ms = (
                        time.monotonic() - blocking_wait_started
                    ) * 1000.0
                    if (
                        math.isfinite(judgment_elapsed_ms)
                        and judgment_elapsed_ms > 0
                        and math.isfinite(blocking_wait_elapsed_ms)
                        and blocking_wait_elapsed_ms >= 0
                    ):
                        timing_sink.append(
                            {
                                "logical_call_elapsed_ms": judgment_elapsed_ms,
                                "operator_blocking_elapsed_ms": (
                                    blocking_wait_elapsed_ms
                                ),
                            }
                        )
                except (KeyError, TypeError, ValueError):
                    pass
        except asyncio.CancelledError as e:
            while not judgment_task.done():
                try:
                    await asyncio.shield(judgment_task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if judgment_task.done():
                try:
                    judgment_task.result()
                except BaseException:
                    # The coordinator cancellation remains authoritative; this
                    # retrieval only consumes the worker outcome.
                    pass
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "full_judgment_cancelled",
                    "pipeline": "lookahead_one",
                    "stage": "full",
                },
            )
            raise
        except BaseException as e:
            if not isinstance(e, Exception):
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=e,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "base_exception_during_full_judgment",
                        "pipeline": "lookahead_one",
                        "stage": "full",
                    },
                )
                raise
            if is_api_budget_exhausted_error(e):
                budget_error = ApiBudgetExhaustedError(str(e))
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=budget_error,
                    payload={
                        **full_attempt_payload,
                        "api_budget_exhausted": True,
                        "pipeline": "lookahead_one",
                    },
                )
                raise budget_error from e
            if config.LINKEDIN_V2_FULL_CONTRACT == "tool":
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=e,
                    payload={
                        **full_attempt_payload,
                        "full_tool_transport_failed": True,
                        "pipeline": "lookahead_one",
                        "stage": "full",
                    },
                )
                raise
            print(f"    [ERROR] Final judgment failed: {e}")
            log_event(self.log_path, "final_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            final = judgment_failure_decision(
                stage="full",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
                error=e,
                source="judgment",
            )
            external_shadow = None
            external_shadow_error = None

        try:
            if external_shadow_error:
                print(
                    "    [WARN] shadow external-evidence setup failed: "
                    f"{external_shadow_error['type']}",
                    flush=True,
                )
                log_event(
                    self.log_path,
                    "external_evidence_shadow_unhandled_exception",
                    name=snippet.name,
                    error=external_shadow_error["message"],
                )

            shadow_record = self._profile_probe.record_shadow_outcome(
                snippet.name,
                snippet.profile_url,
                runtime_search_string.lane_id or "",
                probe_decision,
                final.decision,
                final.confidence,
            )
            self._record_runtime_event(
                search_string=runtime_search_string,
                event_type="cascade_shadow_recorded",
                payload=shadow_record.to_dict(),
            )

            novelty_value, value_rationale = self._derive_novelty_value(
                snippet,
                profile_status_summary=profile_status_summary,
            )
            final.novelty_value = novelty_value
            final.value_rationale = value_rationale
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "full_precommit_bookkeeping_failed",
                    "pipeline": "lookahead_one",
                    "stage": "full",
                },
            )
            raise

        if is_failure_decision(final.decision):
            print(f"    [{final.decision}] {final.rationale}")
            self.stats.setdefault("parse_failures", 0)
            self.stats["parse_failures"] += 1
            # Persist the retryable failure before optional bias bookkeeping;
            # otherwise a monitor exception leaves the canonical attempt open.
            self._finish_runtime_failure_decision(
                attempt_id=full_attempt_id,
                snippet=snippet,
                decision=final,
                payload={
                    **full_attempt_payload,
                    "profile_summary": summary.to_dict(),
                },
            )
            self._in_flight_urls.discard(snippet.profile_url)
            if self._bias_monitor:
                self._bias_monitor.record_decision(DecisionRecord(
                    candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                    string_id=str(snippet.source_string_id),
                    stage="full",
                    decision=final.decision,
                    confidence=final.confidence,
                    capability_area=None,
                ))
            self._raise_if_full_tool_contract_corruption(
                decision=final,
                logical_call_id=str(
                    full_attempt_payload.get("logical_call_id") or ""
                ),
            )
            return final

        review_demotion_reason = review_decision_demotion_reason(final)
        if review_demotion_reason:
            original_review_decision = final.decision
            log_event(
                self.log_path,
                "candidate_review_recorded",
                name=snippet.name,
                profile_url=snippet.profile_url,
                string_id=runtime_search_string.id,
                original_decision=original_review_decision,
                decision="REJECT",
                demoted=True,
                reason=review_demotion_reason,
                lane_id=runtime_search_string.lane_id,
            )
            print(
                f"    [REVIEW_DEMOTED] {original_review_decision} -> REJECT "
                f"({review_demotion_reason})"
            )
            self.stats["reviewed_demoted"] += 1
            _clear_review_evidence(final)
            final.decision = "REJECT"
            final.outreach_tier = ""
            final.reject_reason = "CAPABILITY_INSUFFICIENT"

        save_decision = final.decision in SAVE_FAMILY_DECISIONS
        if not save_decision:
            self._prior_outcomes[snippet.profile_url] = final.decision
            self._mark_terminal(snippet.profile_url)

            finish_extra_payload: dict | None = dict(full_attempt_payload)
            if final.decision in NON_SAVE_REVIEW_DECISIONS:
                lane_payload = {
                    "lane_id": runtime_search_string.lane_id,
                    "lane_name": runtime_search_string.lane_name,
                    "lane_intent": runtime_search_string.lane_intent,
                }
                lane_payload = {k: v for k, v in lane_payload.items() if v}
                if lane_payload:
                    finish_extra_payload["lane"] = lane_payload
            self._finish_runtime_stage_success(
                attempt_id=full_attempt_id,
                stage="full",
                snippet=snippet,
                decision=final,
                profile_summary=summary,
                extra_payload=finish_extra_payload,
            )

        if self._bias_monitor:
            self._bias_monitor.record_decision(DecisionRecord(
                candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                string_id=str(snippet.source_string_id),
                stage="full",
                decision=final.decision,
                confidence=final.confidence,
                capability_area=final.path if final.path != "none" else None,
            ))
            alerts = self._bias_monitor.check_alerts(str(snippet.source_string_id))
            for alert in alerts:
                symbol = {"flag": "⚡", "info": "ℹ"}.get(alert.severity, "⚡")
                print(f"    {symbol} BIAS {alert.severity.upper()}: {alert.message}")
                log_event(self.log_path, "bias_alert", severity=alert.severity,
                          alert_type=alert.alert_type, message=alert.message,
                          string_id=alert.string_id)

        self._record_external_evidence_shadow_from_thread(
            external_shadow=external_shadow,
            snippet=snippet,
            summary=summary,
            final=final,
            runtime_search_string=runtime_search_string,
        )

        if final.decision in SAVE_FAMILY_DECISIONS:
            tag = final.decision if final.decision != "SAVE" else "SAVE"
            print(f"    [{tag}] {final.rationale}")
            self.stats["save_attempts"] += 1
            if final.novelty_value == "low":
                self.stats["high_fit_low_novelty_saves"] += 1
            self._record_outreach_tier_outcome(
                snippet=snippet,
                decision=final,
            )
            try:
                await self._reopen_profile_for_full_eval_save(snippet)
                outcome = await self._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=runtime_search_string,
                    attempt_id=full_attempt_id,
                )
            except BaseException as exc:
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=exc,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "save_side_effect_interrupted",
                        "pipeline": "lookahead_one",
                        "stage": "full",
                    },
                )
                raise
            outcome_payload = dict(outcome.payload or {})
            skip_reason = str(outcome_payload.get("skip_reason") or "")
            persisted = outcome.status == "succeeded"
            already_present = bool(outcome_payload.get("already_present")) or (
                outcome.status == "skipped"
                and skip_reason == "existing_succeeded"
            )
            final.save_outcome = {
                "status": outcome.status,
                "persisted": persisted,
                "already_present": already_present,
                "failure_reason": outcome_payload.get("failure_reason")
                or (skip_reason or None if not persisted and not already_present else None),
            }
            if outcome_payload.get("reconciled_self_save"):
                final.save_outcome["reconciled_self_save"] = True
            if not (persisted or already_present):
                save_error = RuntimeError(
                    "LinkedIn save was not durably confirmed"
                )
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=save_error,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "save_not_confirmed",
                        "pipeline": "lookahead_one",
                        "save_outcome": final.save_outcome,
                        "stage": "full",
                    },
                )
                raise save_error

            self._prior_outcomes[snippet.profile_url] = final.decision
            self._mark_terminal(snippet.profile_url)
            self._finish_runtime_stage_success(
                attempt_id=full_attempt_id,
                stage="full",
                snippet=snippet,
                decision=final,
                profile_summary=summary,
                extra_payload=dict(full_attempt_payload),
            )

            if page_report:
                page_report.add_saved(snippet, final)
            try:
                await self.browser.go_back_to_results()
                await asyncio.sleep(
                    human_delay_correlated(
                        config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS,
                        channel="panel_close",
                    )
                )
            except Exception as e:
                if _is_browser_disconnect_error(e):
                    print(f"    [ERROR] Browser session dropped while closing profile: {e}")
                    log_event(self.log_path, "panel_close_browser_disconnect", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                    raise
                print("    [WARN] Profile panel close missed; attempting recovery")
                log_event(self.log_path, "go_back_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                final._panel_stuck = True
        elif final.decision in NON_SAVE_REVIEW_DECISIONS:
            print(f"    [{final.decision}] {final.rationale}")
            self.stats["reviewed"] += 1
            if final.decision == "REVIEW_INFERRED":
                self.stats["reviewed_inferred"] += 1
            else:
                self.stats["reviewed_flagged"] += 1
            if page_report:
                page_report.add_skipped_opened(snippet, final)
            log_event(
                self.log_path,
                "candidate_review_recorded",
                name=snippet.name,
                profile_url=snippet.profile_url,
                string_id=runtime_search_string.id,
                decision=final.decision,
                review_reason_code=final.review_reason_code,
                lane_id=runtime_search_string.lane_id,
                structural_evidence_count=len(final.review_structural_evidence),
            )
        else:
            print(f"    [REJECT] {final.rationale}")
            self.stats["rejected"] += 1
            if page_report:
                page_report.add_skipped_opened(snippet, final)

        return final

    async def _reopen_profile_for_full_eval_save(self, snippet: CandidateSnippet) -> None:
        expected_identity = LinkedInBrowser._profile_url_fragment(
            snippet.profile_url
        )
        if (
            expected_identity
            and self.browser.current_profile_identity_fragment()
            == expected_identity
        ):
            return

        print("    Re-opening profile for save...")
        if snippet.card_index >= 0:
            await self.browser.ensure_card_rendered(snippet.card_index)

        if snippet.profile_url:
            await self.browser.open_profile_by_url(snippet.profile_url)
        else:
            await self.browser.open_profile(snippet.name)
        if (
            not expected_identity
            or self.browser.current_profile_identity_fragment()
            != expected_identity
        ):
            raise PanelRecoveryError(
                "profile identity mismatch after save reopen"
            )

    def _record_external_evidence_shadow_from_thread(
        self,
        *,
        external_shadow: dict[str, Any] | None,
        snippet: CandidateSnippet,
        summary: CandidateProfileSummary,
        final: OpusDecision,
        runtime_search_string: SearchString,
    ) -> None:
        if not external_shadow:
            return

        trigger: TriggerDecision = external_shadow["trigger"]
        result = external_shadow.get("result")
        evidence = external_shadow.get("evidence")
        enriched = external_shadow.get("enriched")
        enrich_error = external_shadow.get("enrich_error")
        external_evidence_status = external_shadow.get("status", "")
        diff = external_shadow.get("diff") or {}
        evidence_refs_count = int(external_shadow.get("evidence_refs_count") or 0)
        identity_confidence = external_shadow.get("identity_confidence")

        try:
            if not trigger.should_run:
                log_event(
                    self.log_path,
                    "external_evidence_skipped",
                    name=snippet.name,
                    profile_url=snippet.profile_url,
                    skip_reason=trigger.skip_reason,
                    signals=trigger.signals,
                )
                self._record_runtime_event(
                    search_string=runtime_search_string,
                    event_type="external_evidence_skipped",
                    payload={
                        "profile_url": snippet.profile_url,
                        "skip_reason": trigger.skip_reason,
                        "signals": trigger.signals,
                    },
                )
            elif isinstance(result, ExternalCandidateEvidence):
                log_event(
                    self.log_path,
                    "external_evidence_fetched",
                    name=snippet.name,
                    profile_url=snippet.profile_url,
                    trigger_reason=trigger.reason,
                    identity_confidence=result.identity_confidence,
                    fact_blocks=len(result.external_fact_blocks),
                    inferences=len(result.external_inferences),
                )
                self._record_runtime_event(
                    search_string=runtime_search_string,
                    event_type="external_evidence_fetched",
                    payload={
                        "profile_url": snippet.profile_url,
                        "trigger_reason": trigger.reason,
                        "identity_confidence": result.identity_confidence,
                    },
                )
                if enrich_error is not None:
                    log_event(
                        self.log_path,
                        "external_evidence_enriched_judge_failed",
                        name=snippet.name,
                        error=str(enrich_error),
                        # Stored exception logged outside its handler, so
                        # format_exc() would grab the wrong stack.
                        traceback="".join(
                            traceback.format_exception(enrich_error)
                        ),
                    )
            elif isinstance(result, ExternalEvidenceFailure):
                log_event(
                    self.log_path,
                    "external_evidence_failed",
                    name=snippet.name,
                    profile_url=snippet.profile_url,
                    reason=result.reason,
                    detail=result.detail,
                    http_status=result.http_status,
                )
                self._record_runtime_event(
                    search_string=runtime_search_string,
                    event_type="external_evidence_failed",
                    payload={
                        "profile_url": snippet.profile_url,
                        "reason": result.reason,
                        "http_status": result.http_status,
                    },
                )

            record_shadow_full_judgment(
                output_dir=self.output_dir,
                record=ShadowFullJudgmentRecord(
                    candidate_name=snippet.name,
                    profile_url=snippet.profile_url,
                    source_string_id=snippet.source_string_id,
                    page=snippet.page,
                    result_rank=snippet.result_rank,
                    trigger_reason=trigger.reason if trigger.should_run else "",
                    external_evidence_status=external_evidence_status,
                    identity_confidence=identity_confidence,
                    evidence_refs_count=evidence_refs_count,
                    baseline=final.to_dict(),
                    enriched=enriched.to_dict() if enriched is not None else None,
                    diff=diff,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
            )

            if enriched is not None:
                self._record_runtime_event(
                    search_string=runtime_search_string,
                    event_type="shadow_full_judge_completed",
                    payload={
                        "profile_url": snippet.profile_url,
                        "decision_changed": diff.get("decision_changed"),
                        "path_changed": diff.get("path_changed"),
                        "rationale_changed": diff.get("rationale_changed"),
                        "confidence_delta": diff.get("confidence_delta"),
                        "evidence_refs_count": evidence_refs_count,
                    },
                )
        except Exception as shadow_exc:
            print(f"    [WARN] shadow external-evidence eval failed: {shadow_exc}")
            log_event(
                self.log_path,
                "external_evidence_shadow_unhandled_exception",
                name=snippet.name,
                error=str(shadow_exc),
                traceback=traceback.format_exc(),
            )

    def _apply_pipelined_full_eval_page_outcome(
        self,
        *,
        decision: OpusDecision,
        snippet: CandidateSnippet,
        page_num: int,
        all_candidates: list[dict],
        string_stats: dict[str, int],
        search_string: SearchString,
    ) -> None:
        self._clear_resume_pending_full_if_settled(
            snippet=snippet,
            decision=decision,
        )
        preview_outcome = next(
            (
                candidate.get("outcome")
                for candidate in all_candidates
                if (
                    candidate.get("profile_url") == snippet.profile_url
                    or (
                        not candidate.get("profile_url")
                        and candidate.get("name") == snippet.name
                    )
                )
                and candidate.get("page") == page_num
            ),
            None,
        )
        if preview_outcome in {"facial_yes", "facial_borderline"}:
            self._record_facial_funnel_outcome(
                snippet=snippet,
                decision=(
                    "FACIAL_BORDERLINE"
                    if preview_outcome == "facial_borderline"
                    else "FACIAL_YES"
                ),
                search_string=search_string,
                string_stats=string_stats,
            )
        self._record_full_funnel_outcome(
            snippet=snippet,
            decision=decision,
            search_string=search_string,
            string_stats=string_stats,
        )
        if decision.decision in SAVE_FAMILY_DECISIONS:
            save_outcome = getattr(decision, "save_outcome", None) or {}
            if save_outcome and not (
                save_outcome.get("persisted")
                or save_outcome.get("already_present")
            ):
                outcome = "save_failed"
                string_stats["save_failed"] = string_stats.get("save_failed", 0) + 1
            else:
                outcome = "save"
                if not save_outcome.get("already_present") or save_outcome.get(
                    "reconciled_self_save"
                ):
                    string_stats["saves"] += 1
                    search_string.saves.append(snippet.name)
                    if len(search_string.save_exemplars) < 8:
                        search_string.save_exemplars.append({
                            "title": str(
                                getattr(snippet, "current_title", "")
                                or getattr(snippet, "headline", "")
                                or ""
                            ),
                            "company": str(getattr(snippet, "current_company", "") or ""),
                        })
                self._warn_if_off_geo_save(snippet)
        elif decision.decision == "REJECT":
            outcome = "reject"
        elif decision.decision in NON_SAVE_REVIEW_DECISIONS:
            outcome = "review"
        elif decision.stage == "full" and is_failure_decision(decision.decision):
            outcome = "error"
            self._note_page_observation("errored")
        else:
            outcome = "facial_yes"

        for candidate in all_candidates:
            if (
                (
                    candidate.get("profile_url") == snippet.profile_url
                    or (
                        not candidate.get("profile_url")
                        and candidate.get("name") == snippet.name
                    )
                )
                and candidate["page"] == page_num
                and candidate["outcome"] in {"facial_yes", "facial_borderline"}
            ):
                candidate["outcome"] = outcome
                candidate["rationale"] = decision.rationale
                break

    def _finish_unspawned_full_eval_context(
        self,
        eval_context: dict[str, Any],
        *,
        reason: str,
    ) -> OpusDecision:
        snippet: CandidateSnippet = eval_context["snippet"]
        summary: CandidateProfileSummary = eval_context["summary"]
        error = RuntimeError(f"Full evaluation skipped before judgment: {reason}")
        final = judgment_failure_decision(
            stage="full",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
            error=error,
            source="judgment",
        )
        self._finish_runtime_failure_decision(
            attempt_id=eval_context["attempt_id"],
            snippet=snippet,
            decision=final,
            payload={
                **dict(eval_context.get("attempt_payload") or {}),
                "profile_summary": summary.to_dict(),
                "skipped_reason": reason,
            },
        )
        self._in_flight_urls.discard(snippet.profile_url)
        return final

    # ------------------------------------------------------------------
    # Full run: build execution order from strategy
    # ------------------------------------------------------------------

    @staticmethod
    def _lane_projection_aliases(lane_id: str) -> set[str]:
        return _lane_projection.lane_projection_aliases(lane_id)

    @staticmethod
    def _work_item_lane_key(item: dict[str, Any]) -> str:
        return _lane_projection.work_item_lane_key(item)

    @staticmethod
    def _lane_snapshot_filters(snapshot: dict[str, Any]) -> LinkedInStructuredFilters:
        return _lane_projection.lane_snapshot_filters(snapshot)

    def _current_lane_compiler_snapshot(self, lane_dict: dict[str, Any]) -> dict[str, Any]:
        return _lane_projection.current_lane_compiler_snapshot(lane_dict)

    def _structured_lane_projection_records(self) -> list[dict[str, Any]]:
        return _lane_projection.structured_lane_projection_records(self._execution_plan)

    @staticmethod
    def _match_lane_projection(
        item: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        return _lane_projection.match_lane_projection(item, records)

    @staticmethod
    def _apply_lane_projection_to_work_item(
        item: dict[str, Any],
        record: dict[str, Any],
        *,
        boolean_key: str,
    ) -> dict[str, Any]:
        return _lane_projection.apply_lane_projection_to_work_item(
            item,
            record,
            boolean_key=boolean_key,
        )

    def _record_lint_blocked(
        self,
        item: dict[str, Any],
        *,
        boolean_key: str,
        source: str,
        codes: list[str],
        messages: list[str],
        repair_hints: list[str],
    ) -> None:
        """Delegates to BlockAdaptationService."""
        self._block_adaptation_service._record_lint_blocked(
            item,
            boolean_key=boolean_key,
            source=source,
            codes=codes,
            messages=messages,
            repair_hints=repair_hints,
        )

    def _queue_lint_gate(
        self,
        work_item: dict[str, Any],
        *,
        boolean_key: str,
        source: str,
        lint_context,
    ) -> dict | None:
        """Delegates to BlockAdaptationService."""
        return self._block_adaptation_service._queue_lint_gate(
            work_item,
            boolean_key=boolean_key,
            source=source,
            lint_context=lint_context,
        )

    def _lane_only_search_string(
        self,
        record: dict[str, Any],
        *,
        next_id: int,
        batch_num: int,
    ) -> SearchString | None:
        snapshot = dict(record["snapshot"])
        query_payload = snapshot.get("query_payload")
        query_payload = query_payload if isinstance(query_payload, dict) else {}
        filters = LinkedInStructuredFilters.from_dict(query_payload.get("structured_filters"))
        if filters.is_empty():
            return None
        boolean = str(query_payload.get("boolean") or "").strip()
        item = {
            "boolean": boolean,
            "rationale": record["lane_intent"] or record["lane_name"],
            "lane_id": record["lane_id"],
            "family_key": record["lane_id"],
            "lane_name": record["lane_name"],
            "lane_intent": record["lane_intent"],
            "acquisition_mode": snapshot.get("acquisition_mode") or "linkedin_hybrid",
            "lane_snapshot": {"compiler": snapshot},
            "surface": "hybrid" if boolean else "structured_only",
        }
        lane_fields = lane_fields_from_work_unit_item(item)
        try:
            item = normalize_execution_work_item_boolean(
                item,
                boolean_key="boolean",
                structured_filters=lane_fields.get("structured_filters"),
                ubiquitous_terms=ubiquitous_terms_from_brief(self.brief_obj),
                expand_surface_variants=(
                    config.LINKEDIN_SURFACE_VARIANT_EXPANSION_ENABLED
                ),
                proper_nouns=proper_nouns_from_brief(self.brief_obj),
            )
        except BooleanNormalizationError as exc:
            self._record_lint_blocked(
                item,
                boolean_key="boolean",
                source="lane_only",
                codes=[
                    "ubiquitous_and_gate"
                    if isinstance(exc, UbiquitousAndGateError)
                    else "boolean_normalization_error"
                ],
                messages=[str(exc)],
                repair_hints=["Rebuild the lane boolean around at least one specific term."],
            )
            return None
        lane_fields = lane_fields_from_work_unit_item(item)
        lint_payload = self._queue_lint_gate(
            item,
            boolean_key="boolean",
            source="lane_only",
            lint_context=boolean_lint_context_from_brief(self.brief_obj),
        )
        if lint_payload is None:
            return None
        ss = SearchString(
            id=next_id,
            name=f"Lane / {record['lane_name'][:60]}",
            boolean=str(item.get("boolean") or ""),
            block=f"Lane Batch {batch_num}",
            subblock="Structured Lane",
            string_type="Precision",
            family_key=item["family_key"],
            **lane_fields,
            surface=item["surface"],
            boolean_lint=lint_payload,
        )
        self._hydrate_search_string_metadata(ss)
        apply_lane_fields_to_search_string(ss)
        return ss

    def _build_ordered_search_strings(self) -> list[SearchString]:
        """Build execution queue from strategy work units.

        Kit strings are vocabulary — they NEVER appear in the execution queue.
        Generated compounds and coverage gaps remain the legacy projection. When
        lane compiler snapshots carry structured filters, those flat work units
        are projected through the matching lane; structured lanes without any
        flat projection still queue one lane-native SearchString.
        The opening block is intentionally smaller so the run can exploit earlier.
        """
        next_id = 1
        ordered: list[SearchString] = []
        opening_block_size = max(1, config.OPENING_BLOCK_SIZE)
        later_block_size = 5

        if not self._execution_plan:
            return ordered

        lane_records = self._structured_lane_projection_records()
        represented_lane_ids: set[str] = set()

        # P5 (Wave 2): the lint is wired — error-severity findings block
        # queueing (fail-closed per string; healthy siblings still run), and
        # the ubiquity gate runs on a live feed (brief blacklist + structural
        # terms) instead of a term set no producer ever supplied.
        lint_context = boolean_lint_context_from_brief(self.brief_obj)
        ubiquitous_terms = ubiquitous_terms_from_brief(self.brief_obj)
        # Do-not-vary set for deterministic surface expansion: brief-declared
        # named artifacts keep their one correct spelling.
        proper_noun_terms = proper_nouns_from_brief(self.brief_obj)

        # --- Generated compound strings (in priority order from Opus) ---
        compound_idx = 0
        for gs in self._execution_plan.generated_strings:
            if not isinstance(gs, dict):
                continue
            lane_record = self._match_lane_projection(gs, lane_records)
            work_item = (
                self._apply_lane_projection_to_work_item(
                    gs,
                    lane_record,
                    boolean_key="boolean",
                )
                if lane_record
                else gs
            )
            boolean = work_item.get("boolean", "")
            if not boolean:
                continue
            lane_fields = lane_fields_from_work_unit_item(work_item)
            try:
                work_item = normalize_execution_work_item_boolean(
                    work_item,
                    boolean_key="boolean",
                    structured_filters=lane_fields.get("structured_filters"),
                    ubiquitous_terms=ubiquitous_terms,
                    expand_surface_variants=(
                        config.LINKEDIN_SURFACE_VARIANT_EXPANSION_ENABLED
                    ),
                    proper_nouns=proper_noun_terms,
                )
            except BooleanNormalizationError as exc:
                self._record_lint_blocked(
                    work_item,
                    boolean_key="boolean",
                    source="generated",
                    codes=[
                        "ubiquitous_and_gate"
                        if isinstance(exc, UbiquitousAndGateError)
                        else "boolean_normalization_error"
                    ],
                    messages=[str(exc)],
                    repair_hints=[
                        "Anchor at least one AND group on a specific capability or domain term."
                    ],
                )
                continue
            lane_fields = lane_fields_from_work_unit_item(work_item)
            boolean = work_item.get("boolean", "")
            lint_payload = self._queue_lint_gate(
                work_item,
                boolean_key="boolean",
                source="generated",
                lint_context=lint_context,
            )
            if lint_payload is None:
                continue
            compound_idx += 1
            if compound_idx <= opening_block_size:
                batch_num = 1
            else:
                batch_num = 2 + ((compound_idx - opening_block_size - 1) // later_block_size)
            rationale = work_item.get("rationale", "")
            ss = SearchString(
                id=next_id,
                name=f"Compound / {rationale[:60]}" if rationale else "Compound",
                boolean=boolean,
                block=f"Compound Batch {batch_num}",
                subblock="Compound",
                string_type="Precision",
                family_key=work_item.get("family_key", ""),
                novelty_bucket=work_item.get("novelty_bucket", ""),
                domain_lane=work_item.get("domain_lane", ""),
                domain_lane_raw=work_item.get("domain_lane_raw", ""),
                undeclared_lane=bool(work_item.get("undeclared_lane", False)),
                seniority_risk=work_item.get("seniority_risk", ""),
                title_bucket_risk=work_item.get("title_bucket_risk", ""),
                opening_eligible=work_item.get("opening_eligible"),
                retrieval_recipe=work_item.get("retrieval_recipe", {}),
                retrieval_hypothesis_ids=list(work_item.get("retrieval_hypothesis_ids", [])),
                **lane_fields,
                surface=work_item.get("surface", ""),
                boolean_lint=lint_payload,
            )
            self._hydrate_search_string_metadata(ss)
            apply_lane_fields_to_search_string(ss)
            ordered.append(ss)
            if lane_record:
                represented_lane_ids.add(lane_record["lane_id"])
            next_id += 1

        compound_count = len(ordered)

        # --- Coverage gap strings ---
        for gap in self._execution_plan.coverage_gaps:
            if not isinstance(gap, dict):
                continue
            lane_record = self._match_lane_projection(gap, lane_records)
            work_item = (
                self._apply_lane_projection_to_work_item(
                    gap,
                    lane_record,
                    boolean_key="suggested_boolean",
                )
                if lane_record
                else gap
            )
            boolean = work_item.get("suggested_boolean")
            if not boolean:
                continue
            lane_fields = lane_fields_from_work_unit_item(work_item)
            try:
                work_item = normalize_execution_work_item_boolean(
                    work_item,
                    boolean_key="suggested_boolean",
                    structured_filters=lane_fields.get("structured_filters"),
                    ubiquitous_terms=ubiquitous_terms,
                )
            except BooleanNormalizationError as exc:
                self._record_lint_blocked(
                    work_item,
                    boolean_key="suggested_boolean",
                    source="coverage_gap",
                    codes=[
                        "ubiquitous_and_gate"
                        if isinstance(exc, UbiquitousAndGateError)
                        else "boolean_normalization_error"
                    ],
                    messages=[str(exc)],
                    repair_hints=[
                        "Anchor at least one AND group on a specific capability or domain term."
                    ],
                )
                continue
            lane_fields = lane_fields_from_work_unit_item(work_item)
            boolean = work_item.get("suggested_boolean")
            lint_payload = self._queue_lint_gate(
                work_item,
                boolean_key="suggested_boolean",
                source="coverage_gap",
                lint_context=lint_context,
            )
            if lint_payload is None:
                continue
            gap_desc = work_item.get("gap", "coverage gap")
            ss = SearchString(
                id=next_id,
                name=f"Coverage Gap / {gap_desc[:60]}",
                boolean=boolean,
                block="Coverage Gaps",
                subblock="Coverage Gap",
                string_type="Recall",
                boolean_lint=lint_payload,
                family_key=work_item.get("family_key", ""),
                novelty_bucket=work_item.get("novelty_bucket", ""),
                domain_lane=work_item.get("domain_lane", ""),
                domain_lane_raw=work_item.get("domain_lane_raw", ""),
                undeclared_lane=bool(work_item.get("undeclared_lane", False)),
                seniority_risk=work_item.get("seniority_risk", ""),
                title_bucket_risk=work_item.get("title_bucket_risk", ""),
                opening_eligible=work_item.get("opening_eligible"),
                retrieval_recipe=work_item.get("retrieval_recipe", {}),
                retrieval_hypothesis_ids=list(work_item.get("retrieval_hypothesis_ids", [])),
                **lane_fields,
                surface=work_item.get("surface", ""),
            )
            self._hydrate_search_string_metadata(ss)
            apply_lane_fields_to_search_string(ss)
            ordered.append(ss)
            if lane_record:
                represented_lane_ids.add(lane_record["lane_id"])
            next_id += 1

        gap_count = len(ordered) - compound_count
        lane_only_count = 0
        for record in lane_records:
            if record["lane_id"] in represented_lane_ids:
                continue
            ss = self._lane_only_search_string(
                record,
                next_id=next_id,
                batch_num=1 + (lane_only_count // later_block_size),
            )
            if ss is None:
                continue
            ordered.append(ss)
            represented_lane_ids.add(record["lane_id"])
            next_id += 1
            lane_only_count += 1

        if lane_only_count:
            print(
                f"  {compound_count} compound strings + {gap_count} coverage gap strings "
                f"+ {lane_only_count} structured lane strings queued"
            )
        else:
            print(f"  {compound_count} compound strings + {gap_count} coverage gap strings queued")

        # Surface receipt (intended): make structured-filter materialization visible at
        # queue time instead of eyeballed across the Boolean list. Fail-soft — the
        # receipt must never break queue building.
        try:
            summary = _surface_receipt.summarize_intended_surfaces(ordered)
            print(_surface_receipt.format_intended_summary(summary))
            if getattr(self, "log_path", None):
                log_event(
                    self.log_path,
                    "surface_intended",
                    total_strings=summary["total_strings"],
                    hybrid_strings=summary["hybrid_strings"],
                    dimension_counts=summary["dimension_counts"],
                    normalization_strings_with_findings=summary[
                        "normalization_strings_with_findings"
                    ],
                    normalization_finding_counts=summary[
                        "normalization_finding_counts"
                    ],
                    normalization_guard_counts=summary[
                        "normalization_guard_counts"
                    ],
                )
        except Exception:
            pass

        return ordered

    def _load_profile_index_for_adaptation(self) -> dict[str, dict]:
        return self._run_report_service._load_profile_index_for_adaptation()

    def _saved_profile_snapshots(
        self,
        saved_names: list[str],
        profile_index: dict[str, dict],
    ) -> list[dict]:
        """Delegates to BlockAdaptationService."""
        return self._block_adaptation_service._saved_profile_snapshots(
            saved_names,
            profile_index,
        )

    def _string_has_seniority_contamination(
        self,
        search_string: SearchString,
        profile_index: dict[str, dict] | None = None,
    ) -> bool:
        self._hydrate_search_string_metadata(search_string)
        if search_string.seniority_risk == "high" or search_string.title_bucket_risk == "high":
            return True
        if not profile_index or not search_string.saves:
            return False
        for profile in self._saved_profile_snapshots(search_string.saves[:5], profile_index):
            if profile_reads_above_band(profile):
                return True
        return False

    @staticmethod
    def _checkpoint_mode_for_block(
        block_name: str,
        block_strings: list[SearchString] | None = None,
    ) -> str:
        if block_name != "Compound Batch 1":
            return "normal_block_checkpoint"
        if block_strings and any(
            (search_string.string_type or "").lower() == "adaptive"
            for search_string in block_strings
        ):
            return "normal_block_checkpoint"
        return "opening_checkpoint"

    def _search_intelligence_detail_for_string(self, search_string: SearchString) -> dict[str, Any]:
        return self._run_report_service._search_intelligence_detail_for_string(search_string)

    def _search_intelligence_aggregate(
        self,
        strings: list[SearchString],
        *,
        profile_index: dict[str, dict] | None = None,
    ) -> dict[str, Any]:
        return self._run_report_service._search_intelligence_aggregate(
            strings, profile_index=profile_index
        )

    def _apply_exploitation_bias_to_adaptation(
        self,
        *,
        adaptation: AdaptationResponse,
        remaining: list[SearchString],
        block_summary: dict[str, Any],
        checkpoint_mode: str,
    ) -> dict[str, Any]:
        """Delegates to BlockAdaptationService."""
        return self._block_adaptation_service._apply_exploitation_bias_to_adaptation(
            adaptation=adaptation,
            remaining=remaining,
            block_summary=block_summary,
            checkpoint_mode=checkpoint_mode,
        )

    @staticmethod
    def _apply_reorder_actions(progress: Progress, reorder_actions: list[dict[str, Any]]) -> None:
        """Delegates to BlockAdaptationService."""
        BlockAdaptationService._apply_reorder_actions(progress, reorder_actions)

    def _opening_checkpoint_all_dead_and_coherent(self, block_strings: list[SearchString]) -> bool:
        if not block_strings or any(
            search_string.full_outreach_count > 0
            or search_string.full_review_count > 0
            for search_string in block_strings
        ):
            return False

        pattern_counts: dict[str, int] = {}
        inspected = 0
        for search_string in block_strings:
            state = self._experiment_states.get(search_string.id)
            insights = state.last_page_insights if state else None
            if insights is None:
                continue
            inspected += 1
            patterns = list(insights.dominant_non_fit_patterns)
            if not patterns and insights.glance_summary:
                patterns = [insights.glance_summary]
            for pattern in patterns[:1]:
                normalized = " ".join(pattern.strip().lower().split())
                if not normalized:
                    continue
                pattern_counts[normalized] = pattern_counts.get(normalized, 0) + 1

        if inspected < 2 or not pattern_counts:
            return False
        return max(pattern_counts.values()) >= max(2, inspected - 1)

    # ------------------------------------------------------------------
    # Full run: block-level adaptation
    # ------------------------------------------------------------------

    async def _run_block_adaptation(
        self,
        block_name: str,
        block_strings: list[SearchString],
        progress: Progress,
        adapt_fn,
    ) -> None:
        """After a batch of strings completes, send summary to Opus and apply adaptations."""
        checkpoint_mode = self._checkpoint_mode_for_block(block_name, block_strings)
        print(f"\n{'═' * 60}")
        print(f"  Adaptation checkpoint (after {len(block_strings)} strings)")
        print(f"{'═' * 60}")

        profile_index = self._load_profile_index_for_adaptation()
        for search_string in block_strings:
            self._hydrate_search_string_metadata(search_string)
        block_search_intelligence_summary = self._search_intelligence_aggregate(
            block_strings,
            profile_index=profile_index,
        )

        # Build block report
        strings_with_saves = [
            s for s in block_strings if s.full_outreach_count > 0
        ]
        # P7 Stage B: plan-level deterministic warnings (lane collapse) ride
        # every block report so adaptation sees them, not just the console.
        plan_warning_messages = [
            str(w.get("message", "") or "")
            for w in (getattr(self._execution_plan, "plan_warnings", []) or [])
            if isinstance(w, dict) and w.get("message")
        ]
        report = BlockReport(
            block_name=block_name,
            strings_run=len(block_strings),
            strings_with_saves=len(strings_with_saves),
            plan_warnings=plan_warning_messages,
            bias_expected_band=(
                [
                    self._bias_monitor.expected_facial_yes_low,
                    self._bias_monitor.expected_facial_yes_high,
                ]
                if self._bias_monitor
                else None
            ),
            total_results=sum(s.result_count for s in block_strings if s.result_count > 0),
            total_saves=sum(s.full_outreach_count for s in block_strings),
            top_performers=[
                {
                    "string_id": s.id,
                    "name": s.name,
                    "saves": s.full_outreach_count,
                    "physical_saves": len(s.saves),
                    "results": s.result_count,
                    "family_key": s.family_key,
                    "novelty_bucket": s.novelty_bucket,
                    "domain_lane": s.domain_lane,
                }
                for s in sorted(
                    strings_with_saves,
                    key=lambda x: x.full_outreach_count,
                    reverse=True,
                )[:3]
            ],
            zero_save_string_ids=[
                s.id for s in block_strings if s.full_outreach_count == 0
            ],
            string_details=[
                {
                    "string_id": s.id,
                    "name": s.name,
                    "boolean": s.boolean,
                    "original_boolean": s.original_boolean or s.boolean,
                    "result_count": s.result_count,
                    "pages_reviewed": s.pages_reviewed,
                    "candidates": s.candidates_count,
                    "duplicates": s.duplicates_count,
                    "saves": s.full_outreach_count,
                    "physical_saves": len(s.saves),
                    "save_names": s.saves[:5],
                    "saved_profiles": self._saved_profile_snapshots(s.saves[:5], profile_index),
                    "facial_yes": s.facial_yes_count,
                    "facial_borderline": s.facial_borderline_count,
                    "facial_no": s.facial_no_count,
                    "full_reviewed": s.full_reviewed_count,
                    "full_outreach": s.full_outreach_count,
                    "full_review": s.full_review_count,
                    "full_reject": s.full_reject_count,
                    "family_key": s.family_key,
                    "novelty_bucket": s.novelty_bucket,
                    "domain_lane": s.domain_lane,
                    # Codex review (Wave 3): keep-but-flag markers ride the
                    # surfaces adaptation reads, or lane drift looks
                    # first-class mid-run.
                    "domain_lane_raw": s.domain_lane_raw,
                    "undeclared_lane": s.undeclared_lane,
                    "notes": s.notes,
                    # Telemetry demotion (2026-07-04): the per-string bias
                    # context replaces the deleted mid-string pause — the
                    # adaptation model weighs "dense vein vs loosened judge"
                    # itself. Monitor keys are str(source_string_id); s.id is
                    # int — the coercion is load-bearing.
                    "bias": (
                        self._bias_monitor.string_context(str(s.id))
                        if self._bias_monitor
                        else None
                    ),
                    "surface_receipt": s.surface_receipt,
                    "search_intelligence": self._search_intelligence_detail_for_string(s),
                    # P5 (Wave 2): craft health — warning-severity lint
                    # findings ride into the block report so adaptation can
                    # see them (errors never reach execution; they were
                    # blocked at queue build).
                    "lint_warnings": sum(
                        1
                        for f in (s.boolean_lint.get("findings") or [])
                        if f.get("severity") == "warning"
                    ),
                    "lint_warning_codes": sorted(
                        {
                            str(f.get("code"))
                            for f in (s.boolean_lint.get("findings") or [])
                            if f.get("severity") == "warning"
                        }
                    ),
                }
                for s in block_strings
            ],
            search_intelligence_summary=block_search_intelligence_summary,
        )

        print(f"  {report.to_summary_text()}")
        self._update_search_memory_from_block(block_strings)

        # Get remaining unexecuted search strings
        remaining = [s for s in progress.strings if s.status == "queued"]
        for search_string in remaining:
            self._hydrate_search_string_metadata(search_string)

        if not remaining:
            self._clear_pending_block_adaptation(progress)
            self._checkpoint_progress(progress)
            print("  No remaining strings — skipping adaptation.")
            return

        signal_state = SearchSignalState.from_block_report(report)
        gate_config = getattr(self, "_adaptation_gate_config", None)
        if not isinstance(gate_config, AdaptationGateConfig):
            # P3.7: the SPRT/cooldown gate is configurable from shared.config
            # env vars; defaults match default_adaptation_gate_config()
            # exactly, so this is inert until deliberately tuned.
            gate_config = AdaptationGateConfig(
                min_strings=config.ADAPTATION_GATE_MIN_STRINGS,
                min_candidates_seen=config.ADAPTATION_GATE_MIN_CANDIDATES_SEEN,
                min_results_seen=config.ADAPTATION_GATE_MIN_RESULTS_SEEN,
                cooldown_blocks_remaining=config.ADAPTATION_GATE_COOLDOWN_BLOCKS,
                sprt_lower=config.ADAPTATION_GATE_SPRT_LOWER,
                sprt_upper=config.ADAPTATION_GATE_SPRT_UPPER,
            )
        gate_result = evaluate_adaptation_gate(signal_state, gate_config)
        if gate_result.decision != AdaptationGateDecision.ADAPT:
            payload = {
                "block": block_name,
                "checkpoint_mode": checkpoint_mode,
                "decision": gate_result.decision.value,
                "reasons": list(gate_result.reasons),
                "gate_config": gate_config.to_dict(),
                "search_signal_state": signal_state.to_dict(),
            }
            print(
                "  [adapt] Gate deferred adaptation: "
                f"{gate_result.decision.value} ({'; '.join(gate_result.reasons)})"
            )
            log_event(self.log_path, "adaptation_decision", **payload)
            self._record_runtime_event(
                search_string=None,
                event_type="adaptation_decision",
                payload=payload,
            )
            self._clear_pending_block_adaptation(progress)
            self._checkpoint_progress(progress)
            return

        market_intel_advisory_context = ""
        try:
            from market_intelligence.live_advisory import (
                record_block_checkpoint_and_get_context,
            )

            market_intel_advisory_context = record_block_checkpoint_and_get_context(
                brief_path=self.brief_path,
                state_dir=self.state_dir,
                brief_id=self.brief_obj.linkedin_project_id
                or self.brief_obj.id
                or Path(self.brief_path).stem,
                block_name=block_name,
                block_report=report,
                search_memory_summary=build_search_memory_summary(self._search_memory),
            )
        except Exception:
            market_intel_advisory_context = ""

        try:
            adaptation = adapt_fn(
                self.brief_obj, report, remaining,
                kit_vocabulary=self._kit_strings,
                execution_plan=self._execution_plan,
                pivot_count=progress.pivot_count,
                search_memory_summary=build_search_memory_summary(self._search_memory),
                checkpoint_mode=checkpoint_mode,
                market_intel_advisory_context=market_intel_advisory_context,
            )

            # P5 (Wave 2): ubiquity-gate drops from the adapted-string
            # firewall are per-string refusals, recorded like every other
            # queue-time block so the run report sees them.
            firewall_payload = getattr(adaptation, "adapted_string_firewall", None) or {}
            for dropped in firewall_payload.get("dropped") or []:
                self._record_lint_blocked(
                    dropped,
                    boolean_key="boolean",
                    source="adaptive",
                    codes=[str(dropped.get("code") or "ubiquitous_and_gate")],
                    messages=[str(dropped.get("message") or "")],
                    repair_hints=[
                        "Anchor at least one AND group on a specific capability or domain term."
                    ],
                )

            if getattr(adaptation, "no_change", False):
                # P11.1: an explicit decline is a valid, logged decision —
                # not a failure. Deterministic: no strings inserted, no
                # skips/reorders/noise updates/pivot applied, queue order
                # untouched. Short-circuits before exploitation bias and the
                # rest of the "Apply adaptations" section, which would
                # otherwise be no-ops anyway (adapt_after_block already
                # rebuilt a clean, empty AdaptationResponse for this case).
                print("  [adapt] Model declined to adapt (no_change) — queue untouched.")
                self._clear_pending_block_adaptation(progress)
                self._checkpoint_progress(progress)
                log_event(
                    self.log_path,
                    "block_adaptation",
                    block=block_name,
                    report=report.to_dict(),
                    no_change=True,
                    inserted_string_ids=[],
                    displaced_string_ids=[],
                )
                return

            exploitation_overlay = self._apply_exploitation_bias_to_adaptation(
                adaptation=adaptation,
                remaining=remaining,
                block_summary=block_search_intelligence_summary,
                checkpoint_mode=checkpoint_mode,
            )
            if (
                block_search_intelligence_summary.get("proven_family_keys")
                or exploitation_overlay["promoted_string_ids"]
                or exploitation_overlay["demoted_string_ids"]
            ):
                payload = {
                    "block_name": block_name,
                    "checkpoint_mode": checkpoint_mode,
                    **block_search_intelligence_summary,
                    **exploitation_overlay,
                }
                log_event(self.log_path, "linkedin_block_exploitation", **payload)
                self._record_runtime_event(
                    search_string=None,
                    event_type="linkedin_block_exploitation",
                    payload=payload,
                )
                if exploitation_overlay["promoted_string_ids"]:
                    print(
                        "    [adapt] Exploitation bias promoted: "
                        + ", ".join(
                            f"#{string_id}" for string_id in exploitation_overlay["promoted_string_ids"]
                        )
                    )
                if exploitation_overlay["demoted_string_ids"]:
                    print(
                        "    [adapt] Exploitation bias demoted: "
                        + ", ".join(
                            f"#{string_id}" for string_id in exploitation_overlay["demoted_string_ids"]
                        )
                    )

            # Apply adaptations
            # P4.5: hoisted (not just declared inside the if-block) so the
            # block_adaptation log_event below can always reference it —
            # displaced_string_ids is the "strings this adaptation skipped"
            # side of the adaptation-ROI accounting.
            skip_ids: set[int] = set()
            if adaptation.skip_remaining:
                skip_ids = {s["string_id"] for s in adaptation.skip_remaining}
                for ss in progress.strings:
                    if ss.id in skip_ids and ss.status == "queued":
                        assert not any(
                            owner == ss.id
                            for owner in self._resume_pending_full_owner_ids.values()
                        )
                        ss.status = "skipped"
                        ss.notes = f"Skipped by adaptation: {next((s['reason'] for s in adaptation.skip_remaining if s['string_id'] == ss.id), '')}"
                        print(f"    [adapt] Skipping #{ss.id}: {ss.notes}")

            inserted_ids: set[int] = set()

            if adaptation.new_strings:
                max_id = max(s.id for s in progress.strings) if progress.strings else 0
                # Insert adaptive strings at front of queue (before first queued string)
                insert_idx = next(
                    (i for i, s in enumerate(progress.strings) if s.status == "queued"),
                    len(progress.strings),
                )
                adaptive_lint_context = boolean_lint_context_from_brief(self.brief_obj)
                for ns in adaptation.new_strings:
                    # P5 (Wave 2): adaptive strings are the second queueing
                    # path — the same craft-lint gate as the initial queue
                    # build (the ubiquity gate ran earlier, per string,
                    # inside apply_adapted_string_firewall; its drops are
                    # recorded below from the firewall trace).
                    lint_payload = self._queue_lint_gate(
                        ns,
                        boolean_key="boolean",
                        source="adaptive",
                        lint_context=adaptive_lint_context,
                    )
                    if lint_payload is None:
                        continue
                    max_id += 1
                    new_ss = SearchString(
                        id=max_id,
                        name=f"Adaptive / {ns.get('rationale', 'new')}",
                        boolean=ns["boolean"],
                        block=block_name,
                        string_type="Adaptive",
                        boolean_lint=lint_payload,
                        family_key=ns.get("family_key", ""),
                        novelty_bucket=ns.get("novelty_bucket", ""),
                        domain_lane=ns.get("domain_lane", ""),
                        domain_lane_raw=ns.get("domain_lane_raw", ""),
                        undeclared_lane=bool(ns.get("undeclared_lane", False)),
                        seniority_risk=ns.get("seniority_risk", ""),
                        title_bucket_risk=ns.get("title_bucket_risk", ""),
                        opening_eligible=ns.get("opening_eligible"),
                        retrieval_recipe=ns.get("retrieval_recipe", {}),
                        retrieval_hypothesis_ids=list(ns.get("retrieval_hypothesis_ids", [])),
                        **lane_fields_from_work_unit_item(ns),
                    )
                    self._hydrate_search_string_metadata(new_ss)
                    apply_lane_fields_to_search_string(new_ss)
                    progress.strings.insert(insert_idx, new_ss)
                    inserted_ids.add(max_id)
                    insert_idx += 1
                    print(f"    [adapt] Inserted new string #{max_id} (next in queue): {ns['boolean'][:60]}...")

            if adaptation.reorder:
                self._apply_reorder_actions(progress, adaptation.reorder)
                for ro in adaptation.reorder:
                    print(f"    [adapt] Reordered #{ro['string_id']} to {ro.get('move_to', 'last')}")

            if adaptation.noise_updates:
                for nu in adaptation.noise_updates:
                    print(f"    [adapt] Noise update: {nu['term']} → {nu['status']}: {nu.get('note', '')}")
                    if nu['status'] in ("confirmed_noise", "confirmed_signal"):
                        append_jsonl(self.noise_path, {
                            "term": nu.get("term", ""),
                            "status": nu["status"],
                            "note": nu.get("note", ""),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

            # Architecture pivot handling
            if adaptation.pivot_to_architecture:
                if (
                    checkpoint_mode == "opening_checkpoint"
                    and not self._opening_checkpoint_all_dead_and_coherent(block_strings)
                ):
                    print(
                        "  [adapt] Opening checkpoint blocked architecture pivot "
                        f"to {adaptation.pivot_to_architecture}: early block is not uniformly dead/coherent enough."
                    )
                    log_event(
                        self.log_path,
                        "pivot_blocked",
                        reason="opening_checkpoint_guard",
                        recommended=adaptation.pivot_to_architecture,
                        block=block_name,
                    )
                    adaptation.pivot_to_architecture = ""
                    adaptation.pivot_rationale = ""

            if adaptation.pivot_to_architecture:
                # Titration gets 2 pivots (recon-then-commit is its design), others get 1
                max_pivots = 2 if (self._execution_plan and
                                   self._execution_plan.original_architecture == "titration") else 1

                if progress.pivot_count < max_pivots:
                    old_arch = self._execution_plan.architecture if self._execution_plan else "unknown"
                    new_arch = adaptation.pivot_to_architecture
                    print(f"\n  {'!' * 40}")
                    print(f"  ARCHITECTURE PIVOT: {old_arch} → {new_arch}")
                    print(f"  Rationale: {adaptation.pivot_rationale}")
                    print(f"  {'!' * 40}")

                    if self._execution_plan:
                        self._execution_plan.architecture = new_arch
                        self._execution_plan.architecture_rationale = adaptation.pivot_rationale
                        write_json(self.output_dir / "execution_plan.json", self._execution_plan.to_dict())

                    # Clear remaining queued strings but keep the replacement strings
                    # we just injected for the new architecture.
                    for ss in progress.strings:
                        if ss.status == "queued" and ss.id not in inserted_ids:
                            assert not any(
                                owner == ss.id
                                for owner in self._resume_pending_full_owner_ids.values()
                            )
                            ss.status = "skipped"
                            ss.notes = f"Skipped by architecture pivot: {old_arch} → {new_arch}"

                    progress.pivot_count += 1
                    log_event(self.log_path, "architecture_pivot",
                              old=old_arch, new=new_arch,
                              rationale=adaptation.pivot_rationale, block=block_name)
                else:
                    print(f"  [adapt] Pivot to {adaptation.pivot_to_architecture} recommended "
                          f"but max pivots ({max_pivots}) reached — ignoring.")
                    log_event(self.log_path, "pivot_blocked", reason="max_pivots_reached",
                              recommended=adaptation.pivot_to_architecture)

            # P11.2: apply the model-requested adaptation cadence (already
            # clamped to [2, 8] in adapt_after_block) to the FINAL queue
            # state — after skips/reorders/pivot — so a pivot's bulk-skip
            # naturally takes precedence over a stale checkpoint request.
            # Absent/invalid requests leave next_checkpoint_after None and
            # this is a no-op: block segmentation stays exactly as before.
            requested_checkpoint = getattr(adaptation, "next_checkpoint_after_requested", None)
            applied_checkpoint = getattr(adaptation, "next_checkpoint_after", None)
            if applied_checkpoint:
                upcoming = [s for s in progress.strings if s.status == "queued"][:applied_checkpoint]
                if upcoming:
                    checkpoint_block_name = f"Adaptive Checkpoint @{upcoming[0].id}"
                    for ss in upcoming:
                        ss.block = checkpoint_block_name
                print(
                    "    [adapt] next_checkpoint_after: requested="
                    f"{requested_checkpoint} applied={applied_checkpoint} "
                    f"({len(upcoming)} strings relabeled)"
                )
                log_event(
                    self.log_path,
                    "adaptation_checkpoint_cadence",
                    block=block_name,
                    requested=requested_checkpoint,
                    applied=applied_checkpoint,
                    relabeled_string_ids=[ss.id for ss in upcoming],
                )

            self._clear_pending_block_adaptation(progress)
            self._checkpoint_progress(progress)
            log_event(
                self.log_path,
                "block_adaptation",
                block=block_name,
                report=report.to_dict(),
                # P4.5: adaptation ROI accounting reads these back at report
                # time (_adaptation_roi_summary) to compare the inserted
                # strings' realized saves against what was displaced.
                # displaced_string_ids is skip_ids only (adaptation.skip_remaining)
                # — the architecture-pivot bulk-skip below (queued strings not in
                # inserted_ids, on pivot_to_architecture) is a different mechanism
                # and is deliberately excluded here; it's separately logged as its
                # own "architecture_pivot" event.
                inserted_string_ids=sorted(inserted_ids),
                displaced_string_ids=sorted(skip_ids),
            )

        except Exception as e:
            print(f"  [warn] Adaptation failed: {e} — continuing without adaptation")
            log_event(self.log_path, "adaptation_error", block=block_name, error=str(e), traceback=traceback.format_exc())

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def _run_preflight_v2(self) -> None:
        """Run V2 structured preflight: Opus answers specific questions → V2 brief JSON.

        Generates a V2-compatible brief from the JD, saves it, and reloads the brief
        so the pipeline uses structural templates + bias controls instead of freeform archetypes.
        """
        from shared.preflight_v2 import (
            PREFLIGHT_MAX_TOKENS,
            PREFLIGHT_STAGE,
            PreflightRegimeError,
            finalize_preflight_v2,
            format_confidence_notes,
            format_for_review,
            generate_preflight_v2_once,
        )
        from shared.strategy_shadow import dispatch_strategy_shadow, plan_metrics
        import sys

        raw_intake = getattr(self.brief_obj, "intake_notes", "")
        intake_notes = raw_intake.strip() if isinstance(raw_intake, str) else ""
        print(f"  Preflight V2... ({config.STRATEGY_MODEL_NAME.rsplit('/', 1)[-1]} generating structured eval criteria from {'JD + intake notes' if intake_notes else 'JD'})")

        parent_logical_call_id = f"preflight_{uuid.uuid4().hex}"
        base_usage_context = {
            "stage": PREFLIGHT_STAGE,
            "brief_id": self.brief_obj.id,
            "parent_logical_call_id": parent_logical_call_id,
        }

        def _dispatch_shadow(
            raw_response: str,
            system_prompt: str,
            user_prompt: str,
        ) -> None:
            # Shadow strategist (plans/sourcing-generality-hardening.md item
            # 19): fire-and-forget on the exact prompts the primary just saw,
            # before parse/lint so a lint-failed attempt still yields its
            # comparison artifact. No-op unless SHADOW_STRATEGY_ENABLED;
            # never raises (the P9.2 retry/abort path stays untouched).
            dispatch_strategy_shadow(
                stage=PREFLIGHT_STAGE,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=PREFLIGHT_MAX_TOKENS,
                shadow_dir=self.output_dir / "shadow_strategy",
                primary_meta={
                    "primary_model": config.STRATEGY_MODEL_NAME,
                    "metrics": plan_metrics(
                        raw_response,
                        reference_text=system_prompt + "\n" + user_prompt,
                        novelty_reference="system+user",
                    ),
                    # The primary's actual response, so the artifact
                    # renders both sides without the run log.
                    "raw_response": raw_response,
                },
            )

        def _generate_and_validate():
            return generate_preflight_v2_once(
                self.brief_obj,
                model_name=config.STRATEGY_MODEL_NAME,
                usage_context=base_usage_context,
                on_raw_response=_dispatch_shadow,
                on_findings=print,
            )

        # P9.2(a): a v2 preflight failure retries once, then aborts the run.
        # It never falls back to the legacy `shared/preflight.py` regime —
        # that regime's hardcoded ML archetype reinstates the exact
        # permissiveness failure v2 was built to fix. A run on the wrong
        # evaluation regime is worse than no run.
        try:
            generation = _generate_and_validate()
        except Exception as first_exc:
            print(
                f"  [warn] Preflight V2 failed ({first_exc}) — retrying once",
                file=sys.stderr,
            )
            try:
                generation = _generate_and_validate()
            except Exception as retry_exc:
                raise PreflightRegimeError(
                    "Preflight V2 failed twice in a row; aborting the run rather than "
                    "falling back to the legacy evaluation regime (a run on the wrong "
                    f"evaluation regime is worse than no run). First error: {first_exc!r}. "
                    f"Retry error: {retry_exc!r}."
                ) from retry_exc

        # Preserve the historical retry boundary: override/provenance/typed
        # loading happen once after provider+parse+lint succeeds. A local
        # loader defect must never spend a second provider call.
        execution = finalize_preflight_v2(self.brief_obj, generation)
        preflight_data = generation.data
        brief_json = execution.brief_json

        # P9.3: format_for_review previously had zero callers and the
        # generated brief went live with no human-readable rendering
        # anywhere. Print it now so an operator watching the run (or
        # reading the log after the fact) sees the eval criteria before
        # any candidate is judged against them.
        print(format_for_review(preflight_data))

        # Save the generated V2 brief for operator review and debugging
        generated_path = self.output_dir / "preflight_v2_brief.json"
        write_json(str(generated_path), brief_json)
        print(f"  Preflight V2 brief saved to: {generated_path}")
        print(f"  Capability areas: {len(brief_json.get('capability_areas', []))}")
        print(f"  Non-fit patterns: {len(brief_json.get('non_fit_patterns', []))}")

        # The shared seam already proved typed V2 loading before returning.
        self.brief_obj = execution.brief
        init_judger(self.brief_obj)

        # Initialize bias monitor now that we have a V2 brief
        if self.brief_obj.has_v2_schema:
            self._bias_monitor = BiasMonitor.from_brief(self.brief_obj._new_brief)

        print("  Preflight V2 complete — pipeline will use structural templates + bias controls")
        # Re-print the confidence notes LAST: they are preflight's own open
        # questions for the operator (the SPL/Senior-SPL band ambiguity went
        # unread on the 2026-07-04 run because the full review rendering
        # scrolled away). Answers flow back through the seed brief's
        # `instructions` — the operator-calibration channel.
        notes_block = format_confidence_notes(preflight_data)
        if notes_block:
            print(notes_block)

    def _archive_stale_outputs(self) -> None:
        """Move stale per-run artifacts out of the mutable state_dir before a fresh run."""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_dir = source_archive_root(
            "linkedin",
            self._brief_id,
            output_root=self.output_dir,
        ) / "state-resets" / ts
        for path in [
            self.snippets_path,
            self.facial_path,
            self.profiles_path,
            self.final_path,
            self.output_dir / "shadow_judgments.jsonl",
        ]:
            if path.exists() and path.stat().st_size > 0:
                archive_dir.mkdir(parents=True, exist_ok=True)
                backup = archive_dir / path.name
                path.rename(backup)
                print(f"  Archived {path.name} → {backup}")
        shadow_strategy_dir = self.output_dir / "shadow_strategy"
        if shadow_strategy_dir.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
            backup = archive_dir / shadow_strategy_dir.name
            shadow_strategy_dir.rename(backup)
            print(f"  Archived {shadow_strategy_dir.name} → {backup}")

    def _prune_old_archives(self, stem: str, keep: int = 5) -> None:
        """Keep only the N most recent archived versions of a file."""
        pattern = list(self.output_dir.glob(f"{stem}-*.jsonl"))
        # Exclude brief-scoped persistent files (candidate_history-*, noise_discoveries-*, bias_monitor-*)
        archives = sorted(
            [p for p in pattern if not any(p.stem.startswith(pref) for pref in
             ("candidate_history", "noise_discoveries", "bias_monitor"))],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_file in archives[keep:]:
            old_file.unlink()
            print(f"  [cleanup] Removed old archive: {old_file.name}")

    # ------------------------------------------------------------------
    # String restart
    # ------------------------------------------------------------------

    def _restart_string(self, progress: Progress, string_id: int) -> None:
        self._ensure_services()
        self._work_unit_service.restart_string(progress, string_id)

    def _restart_strings(self, progress: Progress, string_ids: list[int]) -> None:
        self._ensure_services()
        self._work_unit_service.restart_strings(progress, string_ids)

    @staticmethod
    def _rewrite_jsonl(path, records: list[dict]) -> None:
        """Rewrite a JSONL file with the given records."""
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Progress management
    # ------------------------------------------------------------------

    def _load_or_create_progress(self) -> Progress:
        self._ensure_services()
        return self._work_unit_service.load_or_create_progress()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _print_strategy_details(self) -> None:
        """Print every generated compound string and coverage gap with rationale."""
        if not self._execution_plan:
            return

        gs = self._execution_plan.generated_strings
        if gs:
            print(f"\n  Compound strings ({len(gs)}):")
            for i, s in enumerate(gs, 1):
                boolean = s.get("boolean", "")
                rationale = s.get("rationale", "")[:100]
                print(f"    #{i}: {boolean}")
                print(f"        → {rationale}")

        gaps = self._execution_plan.coverage_gaps
        if gaps:
            executable = [g for g in gaps if g.get("suggested_boolean")]
            if executable:
                print(f"\n  Coverage gaps ({len(executable)}):")
                for i, g in enumerate(executable, 1):
                    boolean = g.get("suggested_boolean", "")
                    gap_desc = g.get("gap", "")[:100]
                    print(f"    #{i}: {boolean}")
                    print(f"        → Gap: {gap_desc}")

    def _print_session_summary(self, progress: Progress) -> None:
        """Print running session summary after each string completes."""
        done = sum(1 for s in progress.strings if s.status == "done")
        total = len(progress.strings)
        refined = sum(1 for s in progress.strings if s.notes and "Refined" in s.notes)
        print(f"\n  Session: {done}/{total} strings | "
              f"{self.stats['snippets_extracted']} evaluated | "
              f"{self.stats['facial_yes']} facial YES | "
              f"{self.stats['saved']} saves | "
              f"{self.stats.get('activity_saturated_preview_skips', 0)} activity-sat skips | "
              f"{refined} refined")

    def _pipeline_end_stats(self) -> dict:
        """P4.2: ``self.stats`` plus run-level cost, when the run's usage
        JSONL actually has cost data. Never adds ``cost_usd`` as an
        affirmative zero — omitted entirely when the log is absent/empty/
        all-unknown-rate, matching :func:`_sum_token_cost_log_usd`."""
        from shared.llm_usage import resolve_cost_log_run_id

        payload = dict(self.stats)
        log_path = self.output_dir / "token-cost-log.jsonl"
        cost_usd = _sum_token_cost_log_usd(
            log_path,
            run_id=resolve_cost_log_run_id(log_path, self._runtime_run_id),
            exclude_rows_with=("shadow_stage",),
        )
        if cost_usd is not None:
            payload["cost_usd"] = cost_usd
            cost_per_save = _cost_per_save_usd(cost_usd, self.stats.get("saved", 0))
            if cost_per_save is not None:
                payload["cost_per_save_usd"] = cost_per_save
        return payload

    @staticmethod
    def _facial_rate_metrics(stats: dict[str, Any]) -> dict[str, Any]:
        return RunReportService._facial_rate_metrics(stats)

    def _print_summary(self) -> None:
        facial_metrics = self._facial_rate_metrics(self.stats)
        print(f"\n{'=' * 60}")
        print("  Run Summary")
        print(f"{'=' * 60}")
        print(f"  Snippets extracted:  {self.stats['snippets_extracted']}")
        print(f"  Facial YES:          {self.stats['facial_yes']}")
        print(f"  Facial BORDERLINE:   {self.stats.get('facial_borderline', 0)}")
        print(f"  Facial NO:           {self.stats['facial_no']}")
        if facial_metrics["facial_rate_denominator_count"]:
            facial_rate_line = (
                "  Facial rates: "
                f"strict YES {facial_metrics['facial_strict_yes_rate']:.1%} | "
                f"BORDERLINE {facial_metrics['facial_borderline_rate']:.1%} | "
                f"open {facial_metrics['facial_open_rate']:.1%} | "
                "denominator "
                f"{facial_metrics['facial_rate_denominator_count']} "
                "(YES+BORDERLINE+NO)"
            )
            print(facial_rate_line)
        print(f"  SAVED:               {self.stats['saved']}")
        if self.stats.get("save_attempts", 0) != self.stats["saved"]:
            print(f"  Save attempts:       {self.stats['save_attempts']}")
        print(f"  REJECTED:            {self.stats['rejected']}")
        print(f"  High-pressure seen:  {self.stats.get('high_pressure_candidates_seen', 0)}")
        print(f"  Activity skips:      {self.stats.get('activity_saturated_preview_skips', 0)}")
        print(f"  Low-novelty saves:   {self.stats.get('high_fit_low_novelty_saves', 0)}")
        if self._bias_monitor:
            summary = self._bias_monitor.session_summary()
            if summary.get("total_decisions", 0) > 0:
                print(f"  ---")
                print(f"  Full save rate:      {summary.get('save_rate', 0):.1%}")
                print(f"  Parse failures:      {summary.get('parse_failures', 0)} ({summary.get('parse_failure_rate', 0):.1%})")
                print(f"  Bias alerts fired:   {len(summary.get('alerts_fired', []))}")
        print(f"{'=' * 60}")

    def _bias_summary_for_report(self) -> str:
        return self._run_report_service._bias_summary_for_report()

    def _load_run_report_decisions(
        self,
        decision_filter: set[str],
        limit: int = 20,
    ) -> list[dict]:
        return self._run_report_service._load_run_report_decisions(decision_filter, limit=limit)

    def _cost_summary_for_report(self) -> dict:
        return self._run_report_service._cost_summary_for_report()

    def _persist_cost_rollup_sidecar(self) -> None:
        _linkedin_persist_cost_rollup_sidecar(self.output_dir)

    def _run_health_summary(self) -> dict:
        return self._run_report_service._run_health_summary()

    def _shadow_facial_summary(self) -> dict | None:
        return self._run_report_service._shadow_facial_summary()

    def _shadow_cache_hit_rate(self, *, shadow_stage: str) -> float | None:
        """Mean per-call prefix-cache hit rate for one shadow tier.

        Reads token-cost-log.jsonl rows with ``provider="fireworks"`` and
        ``shadow_stage`` matching the given tier, computing
        ``cache_read_input_tokens / (cache_read_input_tokens +
        input_tokens)`` per row (only rows carrying real int token counts
        with a positive denominator), then the mean across those rows.

        Returns ``None`` — never an affirmative ``0.0`` — when there are no
        usable rows: token-cost-log.jsonl is absent (e.g. only run_log.jsonl
        was written in a test), no row for this stage exists yet, or every
        such row is missing token counts entirely. Mirrors
        ``_sum_token_cost_log_usd``'s "no signal, not zero" discipline.
        """
        path = self.output_dir / "token-cost-log.jsonl"
        if not path.exists():
            return None
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return None
        rates: list[float] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            if str(record.get("provider") or "") != "fireworks":
                continue
            if record.get("shadow_stage") != shadow_stage:
                continue
            cached = record.get("cache_read_input_tokens")
            input_tokens = record.get("input_tokens")
            if not isinstance(cached, (int, float)) or isinstance(cached, bool):
                continue
            if not isinstance(input_tokens, (int, float)) or isinstance(input_tokens, bool):
                continue
            denom = cached + input_tokens
            if denom <= 0:
                continue
            rates.append(cached / denom)
        if not rates:
            return None
        return round(sum(rates) / len(rates), 4)

    def _shadow_full_summary(self) -> dict | None:
        return self._run_report_service._shadow_full_summary()

    def _adaptation_roi_summary(self, progress: Progress) -> dict:
        """P4.5: pure per-string-stat accounting for adaptation ROI.

        For every ``block_adaptation`` event this run logged, compares the
        realized save count of the strings the adaptation INSERTED against
        the strings it SKIPPED in that same call (queued strings the
        adaptation displaced instead of running). Both sides read directly
        off ``SearchString.saves`` recorded during the run — no new metric,
        no invented weighting, no fabricated non-zero for strings that
        never ran (a skipped string's realized contribution is honestly 0,
        not estimated).
        """
        string_by_id = {s.id: s for s in progress.strings}

        def _mean_saves(ids: list[int]) -> tuple[float | None, int, int]:
            saves = [len(string_by_id[i].saves) for i in ids if i in string_by_id]
            if not saves:
                return None, 0, 0
            return round(sum(saves) / len(saves), 4), sum(saves), len(saves)

        events: list[dict] = []
        for record in read_jsonl(self.log_path):
            if not isinstance(record, dict) or record.get("event") != "block_adaptation":
                continue
            inserted_ids = [
                i for i in (record.get("inserted_string_ids") or []) if isinstance(i, int)
            ]
            displaced_ids = [
                i for i in (record.get("displaced_string_ids") or []) if isinstance(i, int)
            ]
            if not inserted_ids and not displaced_ids:
                continue
            inserted_mean, inserted_total, inserted_n = _mean_saves(inserted_ids)
            displaced_mean, displaced_total, displaced_n = _mean_saves(displaced_ids)
            events.append(
                {
                    "block": record.get("block", ""),
                    "inserted_string_ids": inserted_ids,
                    "inserted_mean_saves": inserted_mean,
                    "inserted_total_saves": inserted_total,
                    "inserted_count": inserted_n,
                    "displaced_string_ids": displaced_ids,
                    "displaced_mean_saves": displaced_mean,
                    "displaced_total_saves": displaced_total,
                    "displaced_count": displaced_n,
                }
            )

        if not events:
            return {"status": "no_adaptation_events"}
        total_inserted = sum(e["inserted_total_saves"] for e in events)
        total_displaced = sum(e["displaced_total_saves"] for e in events)
        return {
            "status": "ok",
            "events": events,
            "total_inserted_saves": total_inserted,
            "total_displaced_saves": total_displaced,
            "net_saves_gained": total_inserted - total_displaced,
        }

    def _build_run_report_snapshot(self, progress: Progress) -> dict:
        return self._run_report_service._build_run_report_snapshot(progress)

    def _run_report_analysis_system(self) -> str:
        return self._run_report_service._run_report_analysis_system()

    def _generate_run_report(self, progress: Progress | None) -> None:
        """Generate structured and markdown end-of-run debrief artifacts."""
        if not progress or not progress.strings:
            return
        snapshot = self._build_run_report_snapshot(progress)
        run_health = (snapshot.get("metrics_summary") or {}).get("run_health") or {}
        if run_health.get("degraded"):
            log_event(
                self.log_path,
                "run_degraded",
                run_id=self._runtime_run_id,
                reasons=run_health.get("degraded_reasons", []),
                green_but_useless=run_health.get("green_but_useless"),
                judge_parse_failure_rate=run_health.get("judge_parse_failure_rate"),
                baseline_judge_parse_failure_rate=run_health.get(
                    "baseline_judge_parse_failure_rate"
                ),
            )
        generate_run_report(
            snapshot=snapshot,
            output_dir=self.output_dir,
            log_path=self.log_path,
        )

    def _finalize_run_snapshot(self) -> Path | None:
        return freeze_linkedin_run_snapshot(
            runtime_run_id=self._runtime_run_id,
            brief_path=self.brief_path,
            state_dir=self.state_dir,
            log_path=self.log_path,
        )

    def _enumerate_vocabulary(self) -> list[KitString]:
        """Named-artifact vocabulary for the strategy call, or [] when off.

        Runs on the fresh path only, before ``form_strategy``, and costs one
        strategy-tier call. Fail-soft by construction: the helper swallows its
        own provider errors, and this wrapper catches anything it does not, so
        the worst case is the empty channel the run already tolerates.
        """
        if not config.LINKEDIN_VOCABULARY_ENUMERATION_ENABLED:
            return []
        try:
            from shared.vocabulary_enumeration import (
                default_research_call,
                enumerate_domain_vocabulary,
            )

            research_call = (
                default_research_call
                if config.LINKEDIN_VOCABULARY_ENUMERATION_RESEARCH_ENABLED
                else None
            )

            print("  Enumerating domain vocabulary (named artifacts)...")
            kit = enumerate_domain_vocabulary(
                self.brief_obj._new_brief
                if getattr(self.brief_obj, "has_v2_schema", False)
                else self.brief_obj,
                research_call=research_call,
                artifact_dir=self.output_dir,
            )
        except Exception as exc:  # noqa: BLE001 — enrichment never breaks a run
            print(f"  [warn] vocabulary enumeration unavailable: {exc}")
            return []

        if kit:
            terms = sum(ks.boolean.count(" OR ") + 1 for ks in kit)
            print(
                f"  Enumerated {terms} named artifacts into "
                f"{len(kit)} vocabulary groups → {self.output_dir / 'vocabulary_enumeration.json'}"
            )
        return kit

    def _strategy_fully_executed(self) -> bool:
        """True when every search string for this run reached a terminal status.

        Work units are re-created per run (26 distinct strings across 4 runs =
        104 rows in the 3000000001 store), so the question is scoped to THIS
        run's units, not the brief's lifetime.

        Fails OPEN — returns True — when the flag is off, when there is no run
        id, or when the store cannot be read. This gate exists to suppress a
        premature report, not to become a new way for the run to end without
        producing one; an unreadable store should not silently cost the operator
        their market intelligence.
        """
        if not config.MARKET_INTEL_REQUIRE_STRATEGY_COMPLETE:
            return True
        if not self._runtime_run_id or not self._runtime_bridge:
            return True
        try:
            units = self._runtime_bridge.store.list_work_units(
                int(self._runtime_run_id), kind=LINKEDIN_STRING_KIND
            )
        except Exception as exc:
            print(f"  [warn] strategy-completion check failed, allowing: {exc}")
            return True
        if not units:
            return True
        # Use the SAME predicate canonical resume uses. TERMINAL_WORK_UNIT_STATUSES
        # includes "error" (shared/runtime_state/store.py:60), but
        # has_pending_work() counts anything outside done/skipped as pending
        # (shared/runtime_state/read_models.py:1001). Keying this gate to the
        # wider set let a [done, error] strategy publish market intelligence
        # while resume still said work remained — the exact partial-coverage
        # report this gate exists to prevent.
        pending = [
            u for u in units
            if str(u.get("status")) not in _STRATEGY_COMPLETE_WORK_UNIT_STATUSES
        ]
        if pending:
            print(
                f"  Market intelligence deferred: {len(pending)} of {len(units)} "
                f"search strings still pending. It refreshes once the strategy "
                f"is fully executed."
            )
            # Fail-soft: this gate must never become a new way for a run to end
            # without its report. log_event opens/writes a file and can raise.
            try:
                log_event(
                    self.log_path,
                    "market_intel_deferred",
                    pending_strings=len(pending),
                    total_strings=len(units),
                    run_id=int(self._runtime_run_id),
                )
            except Exception as exc:  # noqa: BLE001 — a receipt must not abort finalization
                print(f"  [warn] market_intel_deferred receipt failed: {exc}")
            return False
        return True

    def _enrich_run_snapshot(self, run_dir: Path) -> None:
        enrich_linkedin_run_snapshot(
            runtime_run_id=self._runtime_run_id,
            brief_path=self.brief_path,
            run_dir=run_dir,
            log_path=self.log_path,
        )


# ---------------------------------------------------------------------------
# Page report (structured console output per protocol.md format)
# ---------------------------------------------------------------------------

# _PageReport moved to linkedin.run_report
