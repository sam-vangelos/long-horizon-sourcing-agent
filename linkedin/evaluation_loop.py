"""LinkedIn run lifecycle and sequencing of the concrete execution owners.

Interface: Pipeline constructs EvaluationLoop(EvaluationRunState()), calls
_initialize with the existing constructor arguments, then awaits run_full
with the existing resume arguments. Initialization must precede execution.
Pipeline supplies late-bound day-cycle events; no Pipeline object or callback
bag enters here.
The state holder is in-memory working state. SQLite and its existing writers
remain canonical. Browser work and candidate settlement have one owner.
"""

from __future__ import annotations

from linkedin import judgment_runtime as _judgment_runtime
# Compatibility exports consumed by linkedin.orchestrator remain here.
from linkedin.evaluation_support import (
    _clear_review_evidence,
    _is_containable_facial_page_error,
)
from linkedin.evaluation_support import (
    _PANEL_DOWN_COOLDOWN_SECONDS,
    _PANEL_MOUNT_RELOAD_AT,
    _PANEL_MOUNT_TRIP_AT,
    _is_browser_disconnect_error,
    _is_foreign_project_page,
    _is_recruiter_search_page,
)
from linkedin.search_policy import _normalize_title_family
from shared.employer_allowlist import employer_allowlist_from_brief
import asyncio
import json
import os
import shutil
import signal
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from linkedin.candidate_ledger import _persisted_outreach_tier
from linkedin.evaluation_state import EvaluationRunState
from linkedin.page_review import empty_page_observation
from shared.output_paths import resolve_linkedin_state_dir, source_archive_root
from shared.schemas import CandidateSnippet, SearchString, Progress, KitString, ExecutionPlan
from linkedin.acquisition import LinkedInAcquisitionService
from linkedin.auth_wall import AuthWallError, auth_wall_variant_of
from linkedin.browser import (
    FacetReconciliationError,
    LinkedInBrowser,
    _recruiter_page_project_id,
)
from linkedin.timing_telemetry import RunLogTimingRecorder
from linkedin.page_allocator import AllocationVerdict, AllocatorPolicyError
from linkedin.search_intelligence import LinkedInExperimentState
from linkedin.fallback_search import FallbackSearchProvider
from linkedin.search_mutation import LinkedInSearchMutationDeps, LinkedInSearchMutationExecutor
from linkedin.evaluation_cascade import CascadePolicy, ProfileProbe
from linkedin.side_effects import LinkedInSideEffectsService
from linkedin.work_units import LinkedInWorkUnitDeps, LinkedInWorkUnitService
from linkedin.boolean_compiler import summarize_kit_lint
from shared.failures import ApiBudgetExhaustedError, is_api_budget_exhausted_error
from shared.judger import SAVE_FAMILY_DECISIONS, facial_judge, init_judger
from shared.runtime_state import LinkedInRuntimeStateBridge, RuntimeStateLock, RuntimeStateStore
from shared.runtime_state.store import LINKEDIN_STRING_KIND
from shared.safety import LinkedInRecoveryService, RunSafetyCoordinator, RunStopReason
from shared.safety.stop_reasons import normalize_stop_reason
from shared.storage import append_jsonl, read_jsonl, log_event, write_json, read_json
from shared.brief_loader import load_brief
from shared.bias_controls import BiasMonitor
from shared.search_memory import build_search_memory_summary
from linkedin.block_adaptation import BlockAdaptationDeps, BlockAdaptationService
from linkedin.competitor_scope import company_facet_scope
from linkedin.geography_gate import GeographyGateDeps, GeographyGateService
from linkedin.run_report import (
    RunReportDeps,
    RunReportService,
    enrich_linkedin_run_snapshot,
    freeze_linkedin_run_snapshot,
    _PageReport,
)
from shared import config
from shared.cost_rollup import _cost_per_save_usd, _sum_token_cost_log_usd
from shared.governor import (
    CAP_SHAPED_SHUTDOWN_PREFIXES,
    GovernorLimitReached,
    OperatorStopRequested,
    SessionExpired,
    SessionGovernor,
)
import re as _re
from linkedin.evaluation_errors import (
    GeographyRegimeError,
    PageRenderFailedError,
    PanelRecoveryError,
    TransientPaginationError,
    SurfaceOwnershipLostError,
    ProjectContextMismatchError,
)

# Strategy completion uses the SUCCESSFUL-terminal set, not every terminal
# status: "error" is terminal for a work unit but is NOT executed coverage,
# and canonical resume (read_models.has_pending_work) agrees by counting
# anything outside done/skipped as pending.
_STRATEGY_COMPLETE_WORK_UNIT_STATUSES = frozenset({"done", "skipped"})


_PARENS_SUFFIX = _re.compile(r"\s*\([^)]*\)")
_NON_ALNUM = _re.compile(r"[^a-z0-9]+")


# How many strings in a row may fail facet reconciliation before the run
# stops. One failure belongs to that string; three in a row belongs to the
# rail, and a rail the run cannot own is a named, detected-dangerous state —
# the only kind of condition permitted to end a run.
_FACET_UNRECONCILABLE_STRING_CEILING = 3
_FACET_UNRECONCILABLE_STOP_REASON = "facet_surface_unreconcilable"
# Facet gates the opening raises BEFORE it touches the browser: the compile
# filter that rejects a per-string location, and the two that refuse a
# desired set the strategist built wrong. Nothing on the page caused them and
# nothing on the page can answer them, so they cost the string and nothing
# else — a re-anchor would discard a live result surface, and a strike would
# stop the run over a strategy defect. Every other gate READ the live page
# and keeps both. The observed-set twins (`*_observed_*`) are live reads and
# are deliberately absent.
_FACET_COMPILE_TIME_GATES = frozenset(
    {
        "per_string_location_conflict",
        "companies_desired_blank_value",
        "companies_desired_normalized_collision",
        "titles_desired_blank_value",
        "titles_desired_normalized_collision",
    }
)


_COMPLETED_RUN_STRING_STATUSES = {"done", "skipped"}
_QUEUE_HONESTY_STATUS_ORDER = ("error", "in_progress", "queued")


def _normalize_candidate_name_key(name: str) -> str:
    """Collapse minor punctuation/parenthetical variants when matching saved profiles."""
    t = _PARENS_SUFFIX.sub("", (name or "").lower())
    t = t.replace(".", " ")
    t = _NON_ALNUM.sub(" ", t)
    return " ".join(t.split())


def _exception_message_line(exc: BaseException) -> str | None:
    """First non-blank line of an exception's message, or None when empty."""
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 — a broken __str__ must not mask the run's end
        return None
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def _located_stop_detail(where: str, exc: BaseException) -> str:
    """``<where>: <first line>``, or ``<where>: <ExcType>`` for an empty message.

    This is what a FATAL run records as ``stop_reason_detail``, so a dead run
    says why without anyone reading its stdout.
    """
    return f"{where}: {_exception_message_line(exc) or type(exc).__name__}"


def _runtime_stop_detail(exc: BaseException) -> str:
    """``runtime: <ExcType>: <first line>`` for a failure inside the run body.

    When the exception carries notes, the first line of the last one is
    appended after `` | ``: the post-recovery branch re-raises the original
    disconnect with the resume failure that made the run fatal as a note.
    """
    message = _exception_message_line(exc)
    name = type(exc).__name__
    detail = f"runtime: {name}: {message}" if message else f"runtime: {name}"
    notes = getattr(exc, "__notes__", None)
    if isinstance(notes, list) and notes:
        note_line = next(
            (line.strip() for line in str(notes[-1]).splitlines() if line.strip()),
            None,
        )
        if note_line:
            detail = f"{detail} | {note_line}"
    return detail


def _append_stop_detail(existing: str | None, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


def _setup_abort_outcome(error: BaseException) -> tuple[str, RunStopReason]:
    """``(run_status, stop_reason)`` for a run that ended during setup.

    The same mapping the run body uses, so a stop that is routine there (the
    operator's own Ctrl+C or cancel, the session cap, a governor limit, an
    exhausted API budget) is not recorded as a crash just because it landed
    before the first string. Anything else is a fatal runtime error.
    """
    if isinstance(
        error, (KeyboardInterrupt, asyncio.CancelledError, OperatorStopRequested)
    ):
        return "interrupted", RunStopReason.OPERATOR_STOP
    if isinstance(error, SessionExpired):
        return "interrupted", RunStopReason.SESSION_EXPIRED
    if isinstance(error, GovernorLimitReached):
        return "governor_limit_reached", RunStopReason.GOVERNOR_LIMIT
    if isinstance(error, ApiBudgetExhaustedError) or is_api_budget_exhausted_error(
        error
    ):
        return "interrupted", RunStopReason.API_BUDGET_EXHAUSTED
    return "error", RunStopReason.FATAL_RUNTIME_ERROR


class EvaluationLoop:
    """Owns one session's execution against explicitly held working state."""

    def __init__(self, state: EvaluationRunState):
        self.state = state

    def _initialize(
        self,
        brief_path: str,
        output_dir: Optional[str] = None,
        test_mode: bool = False,
        input_mode: str = "concurrent",
        governor: Optional[SessionGovernor] = None,
    ):
        # Governance attaches to the browser at construction. A caller
        # (session_orchestrator's day-cycle) may hand in the shared governor
        # it already created; absent that, Pipeline stands up its own so the
        # browser is never constructed ungoverned. This differs from
        # LinkedInBrowser's own constructor, which raises on a missing
        # governor — Pipeline is the production execution engine invoked by
        # session_orchestrator, so a sensible real-governor default is not a
        # bypass.
        self.state.context._governor = governor or SessionGovernor()
        _judgment_runtime._validate_judgment_runtime_configuration()
        self.state.brief_path = str(brief_path)
        self.state.context.brief_obj = load_brief(brief_path)
        self.state.state_dir = resolve_linkedin_state_dir(
            brief_path=self.state.brief_path,
            brief=self.state.context.brief_obj,
            state_dir=output_dir,
        )
        self.state.context.output_dir = self.state.state_dir
        self.state.context.output_dir.mkdir(parents=True, exist_ok=True)
        self.state.context.test_mode = test_mode
        self.state.input_mode = input_mode

        # Initialize judger with the Brief dataclass
        init_judger(self.state.context.brief_obj)

        # Output file paths
        self.state.snippets_path = self.state.context.output_dir / "snippets.jsonl"
        self.state.facial_path = self.state.context.output_dir / "facial_judgments.jsonl"
        self.state.profiles_path = self.state.context.output_dir / "profile_summaries.jsonl"
        self.state.final_path = self.state.context.output_dir / "final_judgments.jsonl"
        self.state.progress_path = self.state.context.output_dir / "progress.json"
        self.state.context.log_path = self.state.context.output_dir / "run_log.jsonl"
        self.state.planner._fallback_candidates_path = self.state.context.output_dir / "fallback_candidates.jsonl"
        self.state.planner._fallback_provider: FallbackSearchProvider | None = None
        self.state.runtime_db_path = self.state.context.output_dir / "runtime_state.sqlite3"

        # Browser
        self.state.context._timing_recorder = (
            RunLogTimingRecorder(self.state.context.log_path)
            if config.LINKEDIN_TIMING_TELEMETRY_ENABLED
            else None
        )
        self.state.context.browser = LinkedInBrowser(
            input_mode=input_mode,
            governor=self.state.context._governor,
            timing_recorder=self.state.context._timing_recorder,
            state_dir=Path(self.state.context.output_dir) if self.state.context.output_dir else None,
        )
        # Consecutive profile-panel MOUNT failures (open click lands, panel never appears).
        # Reset on any successful extraction; policy in _note_profile_extraction_failure.
        self.state.surface._panel_mount_failstreak = 0
        self.state.surface._panel_mount_reload_used = False
        # F4: pin the authorized project before anything can bind a tab. Set here
        # rather than at run start because `connect()` is not the only binder —
        # sidebar ops and crash recovery rebind too, and a preference applied
        # after the first bind would leave exactly those paths unguided.
        self.state.context.browser.set_required_project_id(self.state.context.brief_obj.linkedin_project_id)
        # An allowlisted campaign bounds people to CURRENT employers, so every
        # company chip any path reconciles carries the Current scope.
        self.state.context.browser.set_company_facet_scope(
            company_facet_scope(self.state.context.brief_obj)
        )

        # Brief identifier for cross-session file scoping
        self.state.context._brief_id = self.state.context.brief_obj.linkedin_project_id or Path(brief_path).stem

        # Cross-session candidate history (brief-scoped, never archived)
        self.state.history_path = self.state.context.output_dir / f"candidate_history-{self.state.context._brief_id}.jsonl"

        # Cross-session noise discoveries (brief-scoped, never archived)
        self.state.noise_path = self.state.context.output_dir / f"noise_discoveries-{self.state.context._brief_id}.jsonl"
        self.state.search_memory_path = self.state.context.output_dir / f"search_memory-{self.state.context._brief_id}.json"
        self.state.planner.brief_path = self.state.brief_path
        self.state.planner.state_dir = self.state.state_dir
        self.state.planner.noise_path = self.state.noise_path
        self.state.planner._search_memory: dict = {}

        # Dedup cache (loaded once, updated in memory — avoids O(n^2) file reads)
        self.state.candidates._seen_urls: set[str] = set()       # terminal outcomes only
        # Prior-session suppression set (snapshot at history load) —
        # distinguishes "duplicate because suppressed from history" from
        # within-run overlap for family-exhaustion accounting.
        self.state.candidates._prior_session_urls: set[str] = set()
        self.state.candidates._in_flight_urls: set[str] = set()  # currently being evaluated, NOT persisted

        # Prior session outcomes — URL → most recent outcome (for skip-on-prior-reject)
        self.state.candidates._prior_outcomes: dict[str, str] = {}

        # Save dedup — tracks profile URLs already saved in this session
        self.state.candidates._saved_urls: set[str] = set()

        # Strings that returned results earlier this run (observation only).
        self.state.driver._string_ids_with_results: set[int] = set()

        # Stats
        self.state.context.stats = {
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
            # Bounded non-save review outcomes. Tracked separately so
            # save counts never absorb candidates routed to spot check.
            # Retired 2026-09-16: these three are recomputed from
            # HISTORICAL rows on resume and never grow on a fresh run.
            "reviewed": 0,
            "reviewed_inferred": 0,
            "reviewed_flagged": 0,
            # Verdicts that named a retired review outcome and were mapped
            # to REJECT / REVIEW_RETIRED. Non-zero means something upstream
            # is still speaking the old vocabulary.
            "review_retired_mappings": 0,
            "facial_contract_corruptions": 0,
            "full_contract_corruptions": 0,
            "consecutive_facial_provider_failures": 0,
        }
        self.state.reviewer._full_contract_corruption_call_ids: set[str] = set()
        self.state.reviewer._full_id_mismatch_recovered_count: int = 0

        # Progress ref for Ctrl+C handler
        self.state._progress: Optional[Progress] = None
        self.state.candidates._resume_pending_full_decisions: dict[str, str] = {}
        # Profile-read interest hint, keyed by profile_url. In memory and per-run by
        # design: CandidateSnippet is serialized into canonical run state, so a field
        # there would change the persisted shape and break the to_dict() golden tests.
        # Stamped from the RAW facial verdict, upstream of
        # _normalize_facial_decision_for_persistence, so the hint does not depend on
        # which ambiguity posture the brief selected. Under ternary the verdict
        # survives normalization and either side would read the same; under binary a
        # returned BORDERLINE becomes a PARSE_FAILURE that never reaches a profile
        # read at all. Stamping upstream is the one placement correct under both.
        self.state.reviewer._profile_read_interest: dict[str, float] = {}
        self.state.candidates._resume_pending_full_snippets: dict[str, CandidateSnippet] = {}
        self.state.candidates._resume_pending_full_owner_ids: dict[str, int] = {}
        self.state.context._runtime_state = RuntimeStateStore(self.state.runtime_db_path)
        self.state._runtime_lock = RuntimeStateLock(self.state.context.output_dir)
        self.state.context._runtime_run_id: int | None = None
        self.state.driver._completed_pages_this_process: int = 0
        self.state.context._runtime_bridge = LinkedInRuntimeStateBridge(
            store=self.state.context._runtime_state,
            output_dir=self.state.context.output_dir,
            brief_id=self.state.context._brief_id,
            brief_name=self.state.context.brief_obj.id,
            # Phase 3: brief_path so the bridge can pin brief identity
            # on every start_run. brief_path is the on-disk JSON the
            # orchestrator was loaded from; the canonical hash + snapshot
            # come from compute_brief_identity at run-start.
            brief_path=self.state.brief_path,
        )
        self.state._safety = RunSafetyCoordinator(
            store=self.state.context._runtime_state,
            output_dir=self.state.context.output_dir,
            source="linkedin",
            brief_id=self.state.context._brief_id,
        )
        self.state._recovery_service = LinkedInRecoveryService(
            coordinator=self.state._safety,
            browser=self.state.context.browser,
        )
        self.state._experiment_states: dict[int, LinkedInExperimentState] = {}
        self.state.reviewer._profile_probe = ProfileProbe(CascadePolicy())
        self.state.page._latest_page_preview_snippets: list[CandidateSnippet] = []
        self.state.page.review._latest_page_observed: dict[str, int | str] = empty_page_observation()
        # Allocator evidence is process-local until the ordinary cursor-N+1
        # checkpoint commits it. Shadow never owns execution; active mode uses
        # a separate, redo-safe transition checkpoint after that verdict.
        self.state.page.review._allocator_page_identity: tuple[int, str, int] | None = None
        self.state.page.review._allocator_page_expected_keys: set[str] = set()
        self.state.page.review._allocator_page_settled_keys: set[str] = set()
        self.state.page.review._allocator_page_off_policy: bool = False
        self.state.driver._search_mutation_budget_used: int = 0
        # These services bind initialized statistics, browser and lifecycle callbacks.
        self.state._allocator_service.work_units = None
        self.state.reviewer.acquisition: LinkedInAcquisitionService | None = None
        self.state.driver._search_mutation_executor: LinkedInSearchMutationExecutor | None = None
        self.state.reviewer.side_effects: LinkedInSideEffectsService | None = None
        self._ensure_services()

        # Kit strings (populated by run_full)
        self.state.planner._kit_strings: list[KitString] = []
        self.state.planner._execution_plan: Optional[ExecutionPlan] = None

        # Strings refused at queue build — error-severity lint findings and ubiquity-gate hits.
        # Each record carries codes and repair hints; rendered in the run report and logged as
        # search_string_lint_blocked. In-memory only: resumed sessions rebuild nothing here
        # because the durable trace is the run log.
        self.state.planner._lint_blocked_strings: list[dict] = []

        # Bias monitor (V2 briefs only — uses brief's BiasControls + FacialCalibration)
        self.state.context._bias_monitor: Optional[BiasMonitor] = None
        if self.state.context.brief_obj.has_v2_schema:
            # Brief-scoped bias checkpoint (survives across sessions)
            self.state.bias_checkpoint_path = self.state.context.output_dir / f"bias_monitor-{self.state.context._brief_id}.json"
            if self.state.bias_checkpoint_path.exists():
                self.state.context._bias_monitor = BiasMonitor.from_brief(self.state.context.brief_obj._new_brief)
                self.state.context._bias_monitor.load_checkpoint(str(self.state.bias_checkpoint_path))
                print(f"  [bias] Loaded bias monitor from prior session ({len(self.state.context._bias_monitor._decisions)} decisions)")
            else:
                self.state.context._bias_monitor = BiasMonitor.from_brief(self.state.context.brief_obj._new_brief)
        else:
            self.state.bias_checkpoint_path = self.state.context.output_dir / f"bias_monitor-{self.state.context._brief_id}.json"

        # Cadence pause — anti-detection idle breaks
        self.state.surface._last_pause_time: float = time.time()

        # URL snapshot — last known good URL for recovery fallback.
        # INVARIANT (F4): only ever written through _record_last_good_url, which
        # refuses any page that is not provably this brief's Recruiter project.
        self.state.surface._last_good_url: str = ""

        # Constraint-ownership manifest, built at run start once the brief is final.
        self.state._constraint_manifest: dict = {}


        self.state.surface.geography = GeographyGateService(
            GeographyGateDeps(
                get_browser=lambda: self.state.context.browser,
                get_brief_obj=lambda: self.state.context.brief_obj,
                log_path=self.state.context.log_path,
                stats=self.state.context.stats,
            )
        )

        self.state._run_report_service = RunReportService(
            RunReportDeps(
                get_brief_obj=lambda: self.state.context.brief_obj,
                brief_path=self.state.brief_path,
                output_dir=self.state.context.output_dir,
                final_path=self.state.final_path,
                log_path=self.state.context.log_path,
                profiles_path=self.state.profiles_path,
                get_runtime_db_path=lambda: self.state.runtime_db_path,
                stats=self.state.context.stats,
                get_search_memory=lambda: self.state.planner._search_memory,
                get_constraint_manifest=lambda: self.state._constraint_manifest,
                get_experiment_states=lambda: self.state._experiment_states,
                get_runtime_bridge=lambda: self.state.context._runtime_bridge,
                get_runtime_run_id=lambda: self.state.context._runtime_run_id,
                get_session_geography_receipt=lambda: self.state.surface.geography.session_geography_receipt,
                get_bias_monitor=lambda: self.state.context._bias_monitor,
                get_lint_blocked_strings=lambda: self.state.planner._lint_blocked_strings,
                _string_has_seniority_contamination=lambda *a, **kw: self.state.planner._string_has_seniority_contamination(*a, **kw),
            )
        )

        self.state.planner.reports = self.state._run_report_service

        self.state.planner._block_adaptation_service = BlockAdaptationService(
            BlockAdaptationDeps(
                log_path=self.state.context.log_path,
                get_brief_obj=lambda: self.state.context.brief_obj,
                get_lint_blocked_strings=lambda: self.state.planner._lint_blocked_strings,
                ensure_services=lambda: self._ensure_services(),
                get_work_unit_service=lambda: self.state._allocator_service.work_units,
                set_search_memory=lambda memory: self.state.planner._set_search_memory(memory),
                normalize_candidate_name_key=_normalize_candidate_name_key,
            )
        )

    def _load_candidate_history(self) -> None:
        """Load cross-session candidate history into dedup set and prior outcomes dict.

        Called after per-session dedup loading in all three init paths. Additive: merges
        history URLs into ``_seen_urls`` alongside current-session files.

        When the brief carries ``prior_search.ruled_out_urls``, merge them into
        ``_seen_urls`` so candidates already approached or formally ruled out never reach
        the evaluation pipeline. The exclusion applies to every brief; the empty default
        keeps it harmless for ordinary LinkedIn searches.
        """

        if self.state.context._runtime_bridge and self.state.context._runtime_bridge.has_runtime_state():
            blocked_urls, prior_outcomes, saved_urls = self.state.context._runtime_bridge.load_history()
            self.state.candidates._seen_urls = set(blocked_urls)
            # Snapshot the PRIOR-SESSION suppression set before the run
            # adds its own seen urls, so duplicate counting can distinguish
            # "suppressed from history" from within-run overlap.
            self.state.candidates._prior_session_urls = set(blocked_urls)
            self.state.candidates._prior_outcomes = dict(prior_outcomes)
            self.state.candidates._saved_urls.update(saved_urls)
            saves = sum(
                1
                for outcome in self.state.candidates._prior_outcomes.values()
                if outcome in SAVE_FAMILY_DECISIONS
            )
            rejects = sum(1 for outcome in self.state.candidates._prior_outcomes.values() if outcome == "REJECT")
            print(f"  [dedup] Runtime history: {len(self.state.candidates._prior_outcomes)} candidates ({saves} saves, {rejects} rejects)")
            self.state.candidates._merge_prior_search_exclusion(self.state.context.brief_obj)
            return
        if not self.state.history_path.exists():
            self.state.candidates._merge_prior_search_exclusion(self.state.context.brief_obj)
            return
        for entry in read_jsonl(self.state.history_path):
            url = entry.get("profile_url", "")
            if url:
                self.state.candidates._seen_urls.add(url)
                self.state.candidates._prior_session_urls.add(url)
                self.state.candidates._prior_outcomes[url] = entry.get("outcome", "")
        saves = sum(
            1 for o in self.state.candidates._prior_outcomes.values() if o in SAVE_FAMILY_DECISIONS
        )
        rejects = sum(1 for o in self.state.candidates._prior_outcomes.values() if o == "REJECT")
        print(f"  [dedup] Cross-session history: {len(self.state.candidates._prior_outcomes)} candidates ({saves} saves, {rejects} rejects)")
        self.state.candidates._merge_prior_search_exclusion(self.state.context.brief_obj)

    def _ensure_runtime_state(self) -> None:
        output_dir = Path(self.state.context.output_dir)
        if not hasattr(self.state, "runtime_db_path") or self.state.runtime_db_path is None:
            self.state.runtime_db_path = output_dir / "runtime_state.sqlite3"
        if not hasattr(self.state.context, "_runtime_state") or self.state.context._runtime_state is None:
            self.state.context._runtime_state = RuntimeStateStore(self.state.runtime_db_path)
        if not hasattr(self.state, "_runtime_lock") or self.state._runtime_lock is None:
            self.state._runtime_lock = RuntimeStateLock(output_dir)
        if not hasattr(self.state.context, "_runtime_bridge") or self.state.context._runtime_bridge is None:
            self.state.context._runtime_bridge = LinkedInRuntimeStateBridge(
                store=self.state.context._runtime_state,
                output_dir=output_dir,
                brief_id=self.state.context._brief_id,
                brief_name=self.state.context.brief_obj.id,
                brief_path=self.state.brief_path,
            )
        if not hasattr(self.state.context, "_runtime_run_id"):
            self.state.context._runtime_run_id = None
        if not hasattr(self.state, "_safety") or self.state._safety is None:
            self.state._safety = RunSafetyCoordinator(
                store=self.state.context._runtime_state,
                output_dir=output_dir,
                source="linkedin",
                brief_id=self.state.context._brief_id,
            )
        if not hasattr(self.state, "_recovery_service") or self.state._recovery_service is None:
            self.state._recovery_service = LinkedInRecoveryService(
                coordinator=self.state._safety,
                browser=self.state.context.browser,
            )
        self._ensure_services()

    def _record_safety_event(self, event_type: str, payload: dict) -> None:
        if self.state.context._runtime_run_id:
            self.state.context._runtime_state.record_event(
                run_id=self.state.context._runtime_run_id,
                event_type=event_type,
                payload=payload,
            )

    def _ensure_services(self) -> None:
        if self.state._allocator_service.work_units is None:
            self.state._allocator_service.work_units = LinkedInWorkUnitService(
                LinkedInWorkUnitDeps(
                    ensure_runtime_state=self._ensure_runtime_state,
                    get_runtime_bridge=lambda: self.state.context._runtime_bridge,
                    get_runtime_run_id=lambda: self.state.context._runtime_run_id,
                    set_runtime_run_id=self._set_runtime_run_id,
                    get_progress=lambda: self.state._progress,
                    set_progress=self._set_progress,
                    stats=self.state.context.stats,
                    get_experiment_states=lambda: self.state._experiment_states,
                    set_experiment_states=self._set_experiment_states,
                    get_search_memory=lambda: self.state.planner._search_memory,
                    set_search_memory=self.state.planner._set_search_memory,
                    reset_restart_history=self._reset_restart_history,
                )
            )
        self.state.reviewer.ensure_acquisition()
        if not hasattr(self.state.driver, "_search_mutation_executor") or self.state.driver._search_mutation_executor is None:
            self.state.driver._search_mutation_executor = LinkedInSearchMutationExecutor(
                LinkedInSearchMutationDeps(
                    browser=self.state.context.browser,
                    log_path=self.state.context.log_path,
                    get_input_mode=lambda: self.state.input_mode,
                    get_runtime_run_id=lambda: self.state.context._runtime_run_id,
                    get_runtime_state=lambda: self.state.context._runtime_state,
                    get_search_mutation_budget_used=lambda: self.state.driver._search_mutation_budget_used,
                    set_search_mutation_budget_used=self.state.driver._set_search_mutation_budget_used,
                    get_session_geography=lambda: self.state.surface._session_geography_values(),
                    get_company_scope=lambda: employer_allowlist_from_brief(
                        self.state.context.brief_obj
                    ),
                )
            )
        self.state.reviewer.ensure_side_effects()

    def _set_runtime_run_id(self, runtime_run_id: int | None) -> None:
        self.state.context._runtime_run_id = runtime_run_id

    def _set_progress(self, progress: Progress | None) -> None:
        self.state._progress = progress

    def _set_experiment_states(self, states: dict[int, LinkedInExperimentState]) -> None:
        self.state._experiment_states = states

    def _reset_restart_history(self) -> None:
        self.state.candidates._seen_urls = set()
        self.state.candidates._in_flight_urls = set()
        self.state.candidates._prior_outcomes = {}
        self._load_candidate_history()

    def _record_judgment_runtime_profile_event(self, *, resumed: bool) -> None:
        """Persist the process-local judgment posture on the canonical run."""

        self.state.reviewer.attempts._record_runtime_event(
            search_string=None,
            event_type="judgment_runtime_profile",
            payload={
                "resumed_process": bool(resumed),
                "profile": _judgment_runtime._judgment_runtime_profile(self.state.context.brief_obj),
            },
        )

    def _clear_worker_sidecar_if_current(self) -> None:
        """Release Cloris's sidecar lock if this process still owns it."""

        sidecar_path = self.state.context.output_dir / "worker.json"
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
                    self.state._runtime_lock.release()
            finally:
                await self.state.context.browser.disconnect()

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

        run_status, stop_reason = _setup_abort_outcome(primary_error)
        try:
            self._finish_runtime_record_and_freeze(
                run_status=run_status,
                stop_reason=stop_reason,
                progress=progress,
                snapshotter=snapshotter,
                include_bias_checkpoint=include_bias_checkpoint,
                primary_error=primary_error,
                stop_reason_detail=_located_stop_detail(
                    "setup_abort", primary_error
                ),
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
        stop_reason_detail: str | None = None,
    ) -> tuple[str, RunStopReason, Path | None, BaseException | None, str | None]:
        """Attempt every finalization layer without letting projections block truth.

        ``stop_reason_detail`` is the caller's account of why the run ended;
        every downgrade to FATAL below that has no account yet supplies its
        own, so no fatal run is recorded without a reason. The detail
        actually recorded is returned last.
        """

        first_error: BaseException | None = None
        first_error_label: str | None = None
        last_error: BaseException | None = None

        if primary_error is not None and run_status == "completed":
            run_status = "error"
            stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
            if stop_reason_detail is None:
                stop_reason_detail = _runtime_stop_detail(primary_error)

        def attempt(label: str, action: Any) -> Any:
            nonlocal first_error, first_error_label, last_error
            last_error = None
            try:
                return action()
            except BaseException as exc:
                last_error = exc
                if first_error is None:
                    first_error = exc
                    first_error_label = label
                detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                print(f"  [warn] {label} failed: {detail}")
                return None

        if hasattr(self.state, "_llm_usage_cm"):
            attempt(
                "LLM usage finalization",
                lambda: self.state._llm_usage_cm.__exit__(None, None, None),
            )
        if progress is not None:
            attempt(
                "progress checkpoint",
                lambda: self.state._allocator_service._checkpoint_progress(progress),
            )
        if include_bias_checkpoint and getattr(self.state.context, "_bias_monitor", None):
            attempt(
                "bias checkpoint",
                lambda: self.state.context._bias_monitor.save_checkpoint(
                    str(self.state.bias_checkpoint_path)
                ),
            )

        if first_error is not None and run_status == "completed":
            run_status = "error"
            stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
            if stop_reason_detail is None:
                stop_reason_detail = _located_stop_detail(
                    f"finalization: {first_error_label}", first_error
                )

        finalized_run_dir: Path | None = None
        if self.state.context._runtime_run_id:
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
                if stop_reason_detail is None:
                    stop_reason_detail = _located_stop_detail(
                        "finalization: completion honesty gate", exc
                    )
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
                lambda: self.state._safety.finish_run(
                    run_id=self.state.context._runtime_run_id,
                    status=run_status,
                    stop_reason=stop_reason,
                    stop_reason_detail=stop_reason_detail,
                ),
            )
            finalized = attempt(
                "immutable run snapshot",
                snapshotter,
            )
            if isinstance(finalized, Path):
                finalized_run_dir = finalized
            else:
                snapshot_error = last_error or RuntimeError(
                    "immutable LinkedIn run snapshot did not return a path"
                )
                if first_error is None:
                    first_error = snapshot_error
                snapshot_detail = (
                    _located_stop_detail(
                        "finalization: immutable run snapshot", snapshot_error
                    )
                    if last_error is not None
                    else "finalization: immutable LinkedIn run snapshot "
                    "did not return a path"
                )
                # Same rule as a post-run cleanup failure: the snapshot is
                # frozen after the run already ended, so an earlier non-normal
                # reason (session cap, governor, operator, an earlier fatal or
                # honesty downgrade) stays the answer to "why did the run end"
                # and the snapshot failure is appended to the detail. Only an
                # otherwise-normal end becomes fatal.
                if normalize_stop_reason(stop_reason) == RunStopReason.NORMAL:
                    run_status = "error"
                    stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
                    stop_reason_detail = snapshot_detail
                else:
                    stop_reason_detail = _append_stop_detail(
                        stop_reason_detail, snapshot_detail
                    )
                attempt(
                    "canonical run failure finalization",
                    lambda: self.state._safety.finish_run(
                        run_id=self.state.context._runtime_run_id,
                        status=run_status,
                        stop_reason=stop_reason,
                        stop_reason_detail=stop_reason_detail,
                    ),
                )

        return (
            run_status,
            stop_reason,
            finalized_run_dir,
            first_error,
            stop_reason_detail,
        )

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

        run_health = self.state._run_report_service._run_health_summary()
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

    async def rejudge_from_file(self, snippets_path: str) -> None:
        """Re-run judgments on existing snippet extractions. No browser needed."""
        from shared.storage import read_jsonl

        print("=" * 60)
        print("RE-JUDGE MODE: Processing existing snippets with brief-driven prompts")
        print(f"Brief: {self.state.context.brief_obj.id}")
        print("=" * 60)

        snippets_data = read_jsonl(snippets_path)
        print(f"  Loaded {len(snippets_data)} snippets from {snippets_path}")

        rejudge_path = self.state.context.output_dir / "rejudged_facial.jsonl"
        stats = {"total": 0, "facial_yes": 0, "facial_no": 0, "errors": 0}

        for i, data in enumerate(snippets_data, 1):
            snippet = CandidateSnippet.from_dict(data)
            stats["total"] += 1

            print(f"\n  [{i}/{len(snippets_data)}] {snippet.name}")
            print(f"    {snippet.current_title} at {snippet.current_company}")

            try:
                decision = facial_judge(snippet, brief=self.state.context.brief_obj)
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
        print(f"Brief: {self.state.context.brief_obj.id}")
        print(f"Kit URL: {self.state.context.brief_obj.kit_url}")
        print("=" * 60)

        self._ensure_runtime_state()
        lock_acquired = False
        run_status = "completed"
        stop_reason = RunStopReason.NORMAL
        governor_stop_detail: str | None = None
        stop_reason_detail: str | None = None
        try:
            self.state._runtime_lock.acquire()
            lock_acquired = True
        except RuntimeError as exc:
            stop_reason = RunStopReason.LOCK_CONFLICT
            self.state.context._runtime_state.record_event(
                event_type="runtime_lock_conflict",
                payload={"error": str(exc), "traceback": traceback.format_exc(), "output_dir": str(self.state.context.output_dir)},
            )
            raise

        progress: Progress | None = None
        had_runtime_state = bool(
            self.state.context._runtime_bridge and self.state.context._runtime_bridge.has_runtime_state()
        )
        resume_existing_state = bool(
            resume
            and self.state.context._runtime_bridge
            and (
                had_runtime_state
                or self.state.context._runtime_bridge.has_legacy_state()
            )
        )
        try:
            # Bind the canonical run before touching Recruiter so setup
            # failures can still close it honestly and freeze diagnostics.
            latest_before_start = self.state.context._runtime_state.get_latest_run(
                source="linkedin",
                brief_id=self.state.context._brief_id,
            )
            try:
                if resume_existing_state:
                    progress = self._load_or_create_progress()
                    self.state._progress = progress
                else:
                    progress = Progress(brief_name=self.state.context.brief_obj.id)
                    self.state.context._runtime_run_id, progress = (
                        self.state.context._runtime_bridge.start_or_resume_run(
                            resume=False,
                            initial_progress=progress,
                        )
                    )
                    self.state._progress = progress
            except BaseException:
                latest_after_start = self.state.context._runtime_state.get_latest_run(
                    source="linkedin",
                    brief_id=self.state.context._brief_id,
                )
                prior_run_id = (
                    int(latest_before_start["id"])
                    if latest_before_start
                    else None
                )
                if (
                    self.state.context._runtime_run_id is None
                    and latest_after_start
                    and latest_after_start["status"] == "running"
                    and int(latest_after_start["id"]) != prior_run_id
                ):
                    self.state.context._runtime_run_id = int(latest_after_start["id"])
                raise

            if resume_existing_state:
                self.state.candidates._hydrate_resume_funnel_from_runtime(progress, store=self.state.context._runtime_state, run_id=self.state.context._runtime_run_id, brief_id=self.state.context._brief_id, stats=self.state.context.stats)
                self.state._allocator_service._checkpoint_progress(progress)

            await self.state.context.browser.connect()
            log_event(
                self.state.context.log_path,
                "pipeline_start",
                mode="full_run_resume" if resume else "full_run",
                judgment_runtime_profile=_judgment_runtime._judgment_runtime_profile(self.state.context.brief_obj),
            )

            # Open an LLM usage session for every full run so token-cost-log.jsonl records
            # model usage and pipeline_end can report its cost.
            from shared.llm_usage import llm_usage_session

            self.state._llm_usage_cm = llm_usage_session(
                self.state.context.output_dir / "token-cost-log.jsonl",
                module="linkedin",
                brief_id=self.state.context.brief_obj.id,
                run_id=self.state.context._runtime_run_id,
            )
            self.state._llm_usage_cm.__enter__()

            # Navigate to the project search page if we're not already there.
            # Must be on a search page (not a profile page) for the Keywords sidebar.
            current_url = self.state.context.browser.page.url
            project_url = self.state.surface._get_project_url()
            # Project-awareness at the decision point. A Recruiter search
            # page that is not provably THIS brief's project counts exactly like
            # not being on a search page, so the existing navigation path
            # corrects it instead of the run inheriting a foreign pipeline for
            # every save. F1: that includes a page carrying no project id at all
            # (the global /talent/search view) — it proves nothing, and the old
            # predicate accepted it, so run-start skipped navigation entirely.
            foreign_project_page = _is_foreign_project_page(
                current_url, self.state.context.brief_obj.linkedin_project_id
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
                await self.state.context.browser.navigate_to_search(project_url)
            elif not on_search_page:
                print(f"  Navigating to LinkedIn Recruiter search...")
                await self.state.context.browser.navigate_to_search("https://www.linkedin.com/talent/search")

            # Capture initial good URL for error recovery
            try:
                self.state.surface._record_last_good_url(self.state.context.browser.page.url)
            except Exception:
                pass

            self.state.candidates._seen_urls = set()
            self.state.candidates._in_flight_urls = set()
            self.state.candidates._prior_outcomes = {}
            if not resume and had_runtime_state:
                self._load_candidate_history()
                self.state.planner._load_search_memory()

            # Ctrl+C handler — only install if not running under session_orchestrator
            def _sigint_handler(sig, frame):
                print("\n\n  [!] Interrupted. Saving progress...")
                if self.state._progress:
                    self.state._allocator_service._checkpoint_progress(self.state._progress)
                raise KeyboardInterrupt

            if not hasattr(self.state.context, '_session_expired'):
                signal.signal(signal.SIGINT, _sigint_handler)
        except BaseException as primary_error:
            await self._finalize_setup_abort(
                primary_error=primary_error,
                progress=self.state._progress or progress,
                snapshotter=self._finalize_run_snapshot,
                include_bias_checkpoint=True,
                lock_acquired=lock_acquired,
            )
            raise

        try:
            # Apply the brief's session-level location filter on the fresh sidebar before
            # the keyword playbook starts. It persists across keyword strings.
            # MUST live inside this try: a facet miss raises GeographyRegimeError, and a
            # raise before this try would skip the finally below, leaking the runtime lock,
            # browser connection, and LLM usage session.
            await self.state.surface._apply_session_location_filter()

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
                if self.state.context.brief_obj.needs_preflight():
                    generated_path = self.state.context.output_dir / "preflight_v2_brief.json"
                    if not generated_path.exists():
                        raise RuntimeError(
                            "resume requires the run's generated evaluation "
                            f"criteria ({generated_path}) — not found; "
                            "start a fresh run instead"
                        )
                    from shared.brief_loader import _load_v2_brief
                    from shared.preflight_v2 import overlay_seed_onto_resumed_brief

                    # What the operator owns comes from the seed, not the
                    # frozen brief: the alumni blacklist, and the experience
                    # and level fields the seed carries. A change written into
                    # the seed after this campaign generated its brief must
                    # still reach every gate and the judge on resume.
                    frozen_brief = read_json(str(generated_path))
                    generated_brief = overlay_seed_onto_resumed_brief(
                        frozen_brief,
                        self.state.context.brief_obj,
                    )
                    criteria_note = (
                        "(same criteria as the original run)"
                        if generated_brief == frozen_brief
                        else "(criteria updated from the seed, as listed above)"
                    )
                    self.state.context.brief_obj = _load_v2_brief(generated_brief)
                    init_judger(self.state.context.brief_obj)
                    if self.state.context.brief_obj.has_v2_schema:
                        self.state.context._bias_monitor = BiasMonitor.from_brief(self.state.context.brief_obj._new_brief)
                    print(
                        f"  Reloaded generated V2 brief from state dir {criteria_note}"
                    )

                # Restore progress and its checkpointed experiment state as one
                # operation before any later sync can write stale in-memory state.
                if progress is None:
                    progress = self._load_or_create_progress()
                self._record_judgment_runtime_profile_event(resumed=True)
                self.state._progress = progress

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
                self.state.candidates._seen_urls = set()
                self.state.candidates._in_flight_urls = set()
                self.state.candidates._prior_outcomes = {}
                self._load_candidate_history()
                self.state.planner._load_search_memory()
                # Load kit strings for adaptation vocabulary
                kit_path = self.state.context.output_dir / "kit_strings.json"
                if kit_path.exists():
                    kit_data = read_json(str(kit_path))
                    self.state.planner._kit_strings = [KitString.from_dict(ks) for ks in kit_data]
                    print(f"  Loaded {len(self.state.planner._kit_strings)} kit vocabulary strings")

                # Load execution plan for reference
                plan_path = self.state.context.output_dir / "execution_plan.json"
                if plan_path.exists():
                    self.state.planner._execution_plan = ExecutionPlan.from_dict(read_json(str(plan_path)))

                # Load bias monitor checkpoint
                if self.state.context._bias_monitor and self.state.bias_checkpoint_path.exists():
                    self.state.context._bias_monitor.load_checkpoint(str(self.state.bias_checkpoint_path))
                    print(f"  Resumed bias monitor from checkpoint")

                for search_string in progress.strings:
                    self.state.planner._hydrate_search_string_metadata(search_string)

                done_count = sum(1 for s in progress.strings if s.status == "done")
                total_count = len(progress.strings)
                print(f"  {done_count}/{total_count} strings already complete")

                # Re-enforce the constraint manifest on the reloaded
                # brief — a between-session brief edit that added a
                # zero-owner constraint must abort here, not silently drop.
                self._enforce_constraint_manifest()
            else:
                # --- Fresh run: archive stale output files ---
                self._archive_stale_outputs()

                # --- Phase 0: Sourcing Preflight (if JD provided without full eval criteria) ---
                if self.state.context.brief_obj.needs_preflight():
                    print("\n--- Phase 0: Sourcing Preflight (V2 structured) ---")
                    self._run_preflight_v2()

                # --- Phase 1: Extract kit strings (if kit URL provided) ---
                print("\n--- Phase 1: Kit Extraction ---")
                if not self.state.context.brief_obj.kit_url:
                    # An empty vocabulary channel needs enumeration when enabled; otherwise
                    # preserve the normal behavior. An operator-supplied kit is never displaced:
                    # it is explicit vocabulary and outranks a generated one.
                    self.state.planner._kit_strings = self.state.planner._enumerate_vocabulary()
                    if not self.state.planner._kit_strings:
                        print("  No kit URL provided — strategy will generate strings from JD context only.")
                else:
                    self.state.planner._kit_strings = extract_kit_strings(self.state.context.brief_obj.kit_url)
                    if not self.state.planner._kit_strings:
                        print("  [warn] No strings extracted from kit — proceeding with JD context only.")
                    else:
                        # Advisory craft check on kit vocabulary. Kit strings are vocabulary, never
                        # queued, so a defect here only affects copied terms; a summary print is
                        # the appropriate weight.
                        kit_lint_summary = summarize_kit_lint(self.state.planner._kit_strings)
                        if kit_lint_summary:
                            print(f"  [kit-lint] {kit_lint_summary}")

                # Save extracted kit for reference
                if self.state.planner._kit_strings:
                    kit_data = [ks.to_dict() for ks in self.state.planner._kit_strings]
                    write_json(self.state.context.output_dir / "kit_strings.json", kit_data)
                    print(f"  Kit strings saved to {self.state.context.output_dir / 'kit_strings.json'}")

                # The brief is final (preflight ran above if needed) —
                # build/persist the constraint manifest and abort on any
                # stated constraint with zero owners, before spending a
                # strategy call on a run that cannot honor its intake.
                self._enforce_constraint_manifest()

                # --- Phase 2: Strategy formation ---
                print(f"\n--- Phase 2: Strategy Formation ({config.STRATEGY_MODEL_NAME.rsplit('/', 1)[-1]}) ---")
                log_event(self.state.context.log_path, "strategy_started")
                prior_data = None
                if self.state.progress_path.exists():
                    prior_data = read_json(str(self.state.progress_path))

                # Load noise discoveries from prior sessions
                if self.state.noise_path.exists():
                    noise_entries = read_jsonl(self.state.noise_path)
                    if noise_entries:
                        if prior_data is None:
                            prior_data = {}
                        prior_data["noise_discoveries"] = noise_entries

                if self.state.planner._search_memory:
                    if prior_data is None:
                        prior_data = {}
                    prior_data["search_memory_summary"] = build_search_memory_summary(
                        self.state.planner._search_memory
                    )

                from market_intelligence.engine import load_lane_feedback_for_strategy

                lane_feedback = load_lane_feedback_for_strategy(self.state.brief_path)
                self.state.planner._execution_plan = form_strategy(
                    self.state.context.brief_obj,
                    self.state.planner._kit_strings,
                    prior_data,
                    lane_feedback=lane_feedback or None,
                    # Shadow strategist (item 19): same artifact dir as the
                    # preflight shadow; no-op unless SHADOW_STRATEGY_ENABLED.
                    shadow_dir=self.state.context.output_dir / "shadow_strategy",
                    # An unparseable formation reply is saved under
                    # <state_dir>/forensics/strategy/ before the one retry.
                    state_dir=self.state.state_dir,
                )
                # Close the consumption loop — diffs the strategist
                # consumed are marked in the artifact and never re-served.
                if self.state.planner._execution_plan.consumed_feedback_ids:
                    try:
                        from market_intelligence.engine import (
                            mark_planner_diffs_consumed,
                        )

                        marked = mark_planner_diffs_consumed(
                            self.state.brief_path,
                            self.state.planner._execution_plan.consumed_feedback_ids,
                        )
                        if marked:
                            print(f"  [strategy] Marked {marked} planner diff(s) consumed")
                    except Exception as consume_exc:
                        print(f"  [warn] planner-diff consumption marking failed: {consume_exc}")
                write_json(self.state.context.output_dir / "execution_plan.json", self.state.planner._execution_plan.to_dict())
                log_event(
                    self.state.context.log_path,
                    "strategy_completed",
                    generated_string_count=len(self.state.planner._execution_plan.generated_strings),
                    coverage_gap_count=len(self.state.planner._execution_plan.coverage_gaps),
                    architecture=self.state.planner._execution_plan.architecture,
                )
                print(f"  Strategy: {self.state.planner._execution_plan.strategy_rationale[:120]}...")
                source = "kit vocabulary" if self.state.planner._kit_strings else "JD context"
                print(f"  {len(self.state.planner._execution_plan.generated_strings)} compound strings synthesized from {source}")
                if self.state.planner._execution_plan.coverage_gaps:
                    gap_with_boolean = sum(1 for g in self.state.planner._execution_plan.coverage_gaps if g.get("suggested_boolean"))
                    print(f"  {len(self.state.planner._execution_plan.coverage_gaps)} coverage gaps identified ({gap_with_boolean} with executable strings)")

                # --- Phase 2b: Verbose strategy logging ---
                self.state.planner._print_strategy_details()

                # --- Phase 3: Build execution order from plan ---
                search_strings = self.state.planner._build_ordered_search_strings()
                for search_string in search_strings:
                    self.state.planner._hydrate_search_string_metadata(search_string)

                progress = Progress(
                    brief_name=self.state.context.brief_obj.id,
                    strings=search_strings,
                )
                self.state._progress = progress
                self.state._allocator_service._checkpoint_progress(progress)
                self._record_judgment_runtime_profile_event(resumed=False)

            # --- Execution ---
            print(f"\n--- Execution ({len(progress.strings)} strings) ---")
            log_event(self.state.context.log_path, "execution_started", string_count=len(progress.strings))
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
                            await self.state.planner._run_block_adaptation(
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
                        self.state.planner._clear_pending_block_adaptation(progress)
                        self.state._allocator_service._checkpoint_progress(progress)

            string_index = 0
            consecutive_facet_failures = 0
            while string_index < len(progress.strings):
                if progress and hasattr(self.state.context, "_operator_stop_event") and self.state.context._operator_stop_event.is_set():
                    self.state._allocator_service._checkpoint_progress(progress)
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
                    and self.state._allocator_service._allocator_active_enabled()
                ):
                    print(f"\n  [skip] String #{search_string.id}: {search_string.name} (failed previously)")
                    string_index += 1
                    continue

                # Block transition → adaptation
                if search_string.block and search_string.block != current_block:
                    if current_block and block_strings:
                        await self.state.planner._run_block_adaptation(
                            current_block, block_strings, progress, adapt_after_block
                        )
                        block_strings = []
                        # Re-evaluate this position because adaptation may have
                        # inserted replacement strings or re-ordered the queue.
                        continue
                    current_block = search_string.block
                    block_strings = []

                if progress and hasattr(self.state.context, "_operator_stop_event") and self.state.context._operator_stop_event.is_set():
                    self.state._allocator_service._checkpoint_progress(progress)
                    print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                    raise OperatorStopRequested()

                print(f"\n{'─' * 60}")
                print(f"  String #{search_string.id}: {search_string.name}")
                print(f"  [{search_string.block} / {search_string.subblock} / {search_string.string_type}]")
                print(f"{'─' * 60}")
                log_event(
                    self.state.context.log_path,
                    "string_started",
                    string_id=search_string.id,
                    name=search_string.name,
                    block=search_string.block,
                    rationale=search_string.name,
                )

                search_string.status = "in_progress"
                progress.current_string_id = search_string.id
                allocator_verdict: AllocationVerdict | None = None
                self.state._allocator_service._checkpoint_progress(progress, search_string=search_string)
                recovery_attempted = False
                original_disconnect_error: Exception | None = None
                facet_surface_lost = False
                facet_reanchor_attempted = False
                while True:
                    try:
                        allocator_verdict = await self.state.driver._process_string(
                            search_string,
                            progress,
                        )
                        consecutive_facet_failures = 0
                        break
                    except FacetReconciliationError as e:
                        # A sidebar this string cannot own is THIS STRING's
                        # failure, never the run's: keep-alive absorbs an
                        # unclassified surface fault and the queue moves on.
                        # Removal itself stays fail-closed — the string never
                        # searches inside foreign pills — and only a rail that
                        # refuses three strings in a row, each on a page
                        # re-anchored for it, stops the run, by name.
                        if not config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
                            # Flag off, this is unreachable and identical to
                            # the generic handler below if it ever is reached.
                            raise
                        gate = str(getattr(e, "gate", "") or "unknown")
                        if gate in _FACET_COMPILE_TIME_GATES:
                            print(
                                f"\n  [!] String #{search_string.id}: {gate}; "
                                "the variant's own filters, not the rail — "
                                "marking it errored and moving on."
                            )
                            search_string.status = "error"
                            search_string.notes = (
                                search_string.notes or ""
                            ) + f" Error: {e}"
                            self.state._allocator_service._checkpoint_progress(
                                progress,
                                search_string=search_string,
                                page_num=progress.current_page or None,
                            )
                            log_event(
                                self.state.context.log_path,
                                "string_error",
                                string_id=search_string.id,
                                error=str(e),
                                traceback=traceback.format_exc(),
                            )
                            facet_surface_lost = True
                            break
                        if not facet_reanchor_attempted:
                            # A blocked string enters no search and runs no
                            # recovery, so nothing changes between strikes:
                            # the page that refused string #38 was the page
                            # that refused #39 and #40, and the ceiling was
                            # reached against one dead surface (Addendum I3).
                            # Capture it, re-anchor once, and let this string
                            # try again. Only a failure that survives a fresh
                            # page is evidence about the rail, and three of
                            # those still stop the run by the same name.
                            facet_reanchor_attempted = True
                            try:
                                from linkedin.page_capture import capture_page_state

                                await capture_page_state(
                                    self.state.context.browser,
                                    self.state.state_dir,
                                    reason=f"facet_gate_{gate}",
                                    run_id=self.state.context._runtime_run_id,
                                )
                            except Exception:
                                pass
                            reanchored = (
                                await self.state.surface.reanchor_search_surface(
                                    f"facet_gate_{gate}"
                                )
                            )
                            # One line, and it says which of the two happened:
                            # a retry announced as re-anchored when the
                            # navigation failed is the kind of console line
                            # that made this RCA take a log diff to settle.
                            print(
                                f"  [!] String #{search_string.id}: {gate}; "
                                + (
                                    "re-anchored the search surface"
                                    if reanchored
                                    else "could not re-anchor the search surface"
                                )
                                + ", retrying once."
                            )
                            continue
                        consecutive_facet_failures += 1
                        print(
                            f"\n  [!] String #{search_string.id} could not own its "
                            f"facets ({gate}); marking it errored and moving on "
                            f"({consecutive_facet_failures} in a row)."
                        )
                        search_string.status = "error"
                        search_string.notes = (search_string.notes or "") + f" Error: {e}"
                        self.state._allocator_service._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        log_event(
                            self.state.context.log_path,
                            "string_error",
                            string_id=search_string.id,
                            error=str(e),
                            traceback=traceback.format_exc(),
                        )
                        if (
                            consecutive_facet_failures
                            >= _FACET_UNRECONCILABLE_STRING_CEILING
                        ):
                            stop_reason = RunStopReason.GOVERNOR_LIMIT
                            if self.state.context._runtime_run_id:
                                self.state._safety.record_governor_limit(
                                    run_id=self.state.context._runtime_run_id,
                                    reason=_FACET_UNRECONCILABLE_STOP_REASON,
                                    payload={
                                        "string_id": search_string.id,
                                        "consecutive_strings": consecutive_facet_failures,
                                        "gate": gate,
                                    },
                                )
                            raise GovernorLimitReached(
                                _FACET_UNRECONCILABLE_STOP_REASON
                            ) from e
                        facet_surface_lost = True
                        break
                    except SessionExpired:
                        stop_reason = RunStopReason.SESSION_EXPIRED
                        self.state._allocator_service._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        if self.state.context._bias_monitor:
                            self.state.context._bias_monitor.save_checkpoint(str(self.state.bias_checkpoint_path))
                        raise
                    except GovernorLimitReached as e:
                        print(f"\n  [GOVERNOR] Session limit reached: {e.reason}")
                        stop_reason = RunStopReason.GOVERNOR_LIMIT
                        if self.state.context._runtime_run_id:
                            self.state._safety.record_governor_limit(
                                run_id=self.state.context._runtime_run_id,
                                reason=e.reason,
                                payload={"string_id": search_string.id},
                            )
                        self.state._allocator_service._checkpoint_progress(
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
                        self.state._allocator_service._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        log_event(
                            self.state.context.log_path,
                            "api_budget_exhausted",
                            string_id=search_string.id,
                            page=progress.current_page or None,
                            error=str(e),
                            traceback=traceback.format_exc(),
                        )
                        self.state.reviewer.attempts._record_runtime_event(
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
                        self.state._allocator_service._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        raise
                    except GeographyRegimeError:
                        # A regime error is a run-level abort, never a
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
                        self.state._allocator_service._checkpoint_progress(
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
                        self.state._allocator_service._checkpoint_progress(
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
                        self.state._allocator_service._checkpoint_progress(progress)
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
                                self.state._allocator_service._checkpoint_progress(
                                    progress,
                                    search_string=search_string,
                                    page_num=progress.current_page or None,
                                )
                                raise original_disconnect_error from None

                            recovery_attempted = True
                            try:
                                recovery_snapshot = (
                                    await self.state.surface._capture_recovery_snapshot(
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
                                self.state._allocator_service._checkpoint_progress(
                                    progress,
                                    search_string=search_string,
                                    page_num=progress.current_page or None,
                                )
                                raise original_disconnect_error from None

                            self.state._allocator_service._checkpoint_progress(
                                progress,
                                search_string=search_string,
                                page_num=progress.current_page or None,
                            )
                            recovered = False
                            try:
                                if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
                                    recovered = await self.state._recovery_service.recover(
                                        run_id=self.state.context._runtime_run_id,
                                        snapshot=recovery_snapshot,
                                        restore_location=self.state.surface._verify_session_geography_chips,
                                    )
                                else:
                                    recovered = await self.state._recovery_service.recover(
                                        run_id=self.state.context._runtime_run_id,
                                        snapshot=recovery_snapshot,
                                    )
                                if not recovered:
                                    raise RuntimeError("browser recovery failed")
                                if not config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
                                    await self.state.surface._reassert_session_location_after_recovery()
                                active_string_id = search_string.id
                                block_string_ids = {
                                    item.id for item in block_strings
                                }
                                reloaded_progress = (
                                    self.state.context._runtime_bridge.load_progress(
                                        self.state.context._runtime_run_id
                                    )
                                )
                                reloaded_states = (
                                    self.state.context._runtime_bridge.load_experiment_states(
                                        self.state.context._runtime_run_id,
                                        progress=reloaded_progress,
                                    )
                                )
                                self.state.candidates._hydrate_resume_funnel_from_runtime(
                                    reloaded_progress
                                , store=self.state.context._runtime_state, run_id=self.state.context._runtime_run_id, brief_id=self.state.context._brief_id, stats=self.state.context.stats)
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
                                wall_variant = auth_wall_variant_of(recovery_error)
                                if wall_variant is not None:
                                    # Recovery met a LinkedIn sign-in wall. That,
                                    # not the disconnect, is why the run ends:
                                    # surface the wall itself so the run records
                                    # it as its fatal detail and the day cycle
                                    # stops as recruiter_reauth_required.
                                    stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
                                    if isinstance(recovery_error, AuthWallError):
                                        raise
                                    raise AuthWallError(
                                        wall_variant
                                    ) from recovery_error
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
                            self.state._progress = progress
                            self.state._experiment_states = reloaded_states
                            self.state.candidates._in_flight_urls.clear()
                            continue

                        if self.state._allocator_service._allocator_active_enabled():
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
                        self.state._allocator_service._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=progress.current_page or None,
                        )
                        log_event(self.state.context.log_path, "string_error", string_id=search_string.id, error=str(e), traceback=traceback.format_exc())
                        raise

                if facet_surface_lost:
                    # Already errored and checkpointed above. It reviewed no
                    # page, so it teaches no block adaptation.
                    string_index += 1
                    continue

                if (
                    self.state._allocator_service._allocator_active_enabled()
                    and allocator_verdict is not None
                ):
                    pending_by_id = {item.id: item for item in progress.strings}
                    block_strings = [
                        pending_by_id[root_id]
                        for root_id in progress.pending_block_string_ids
                        if root_id in pending_by_id
                        and self.state._allocator_service._allocator_terminal_status(
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

                await self.state.driver.finish_pending_reviews(search_string, progress)
                block_strings.append(search_string)
                next_active = next(
                    (
                        candidate
                        for candidate in progress.strings[string_index + 1:]
                        if candidate.status not in {"done", "skipped"}
                    ),
                    None,
                )
                self.state.planner._set_pending_block_adaptation(
                    progress,
                    current_block or search_string.block,
                    block_strings,
                    ready=next_active is None or next_active.block != (current_block or search_string.block),
                )
                self.state._allocator_service._checkpoint_progress(progress, search_string=search_string)
                if self.state.context._bias_monitor:
                    self.state.context._bias_monitor.save_checkpoint(str(self.state.bias_checkpoint_path))
                log_event(
                    self.state.context.log_path,
                    "string_complete",
                    **{**self.state.context.stats, "string_id": search_string.id},
                )
                self.state._run_report_service._print_session_summary(progress)
                string_index += 1

            if progress and hasattr(self.state.context, "_operator_stop_event") and self.state.context._operator_stop_event.is_set():
                self.state._allocator_service._checkpoint_progress(progress)
                print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                raise OperatorStopRequested()

            # Final block adaptation
            if current_block and block_strings:
                await self.state.planner._run_block_adaptation(
                    current_block, block_strings, progress, adapt_after_block
                )
            if progress and hasattr(self.state.context, "_operator_stop_event") and self.state.context._operator_stop_event.is_set():
                self.state._allocator_service._checkpoint_progress(progress)
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
            governor_stop_detail = str(getattr(e, "reason", None) or e)
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
            log_event(self.state.context.log_path, "pipeline_error", error=str(e), traceback=traceback.format_exc())
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
                if stop_reason == RunStopReason.FATAL_RUNTIME_ERROR:
                    stop_reason_detail = _runtime_stop_detail(e)
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
                        stop_reason_detail,
                    ) = self._finish_runtime_record_and_freeze(
                        run_status=run_status,
                        stop_reason=stop_reason,
                        progress=self.state._progress,
                        snapshotter=self._finalize_run_snapshot,
                        include_bias_checkpoint=True,
                        primary_error=primary_error,
                        stop_reason_detail=stop_reason_detail,
                    )
                except BaseException as freeze_error:
                    # If finalization still raises, record it as finalization_error (surfaced at
                    # the end of this finally when the run was otherwise clean) rather than
                    # letting it replace primary_error while unwinding.
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
                # A failed teardown (sidecar, lock, browser) happens AFTER the
                # run already ended for some reason. When that reason is
                # non-normal (session cap, governor stop, operator stop, an
                # earlier fatal error) it stays the answer to "why did the
                # run end" and the cleanup failure is appended to the detail;
                # overwriting it with fatal_runtime_error hid the real cause.
                # Only an otherwise-normal end becomes fatal. Market-intel
                # eligibility is unaffected: both gates below already require
                # cleanup_error is None.
                cleanup_detail = _located_stop_detail("cleanup", cleanup_error)
                if normalize_stop_reason(stop_reason) == RunStopReason.NORMAL:
                    run_status = "error"
                    stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
                    stop_reason_detail = cleanup_detail
                else:
                    stop_reason_detail = _append_stop_detail(
                        stop_reason_detail, cleanup_detail
                    )
                try:
                    self.state._safety.finish_run(
                        run_id=self.state.context._runtime_run_id,
                        status=run_status,
                        stop_reason=stop_reason,
                        stop_reason_detail=stop_reason_detail,
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
                    # If this expression raises while unwinding a primary error, do not let it
                    # replace that error.
                    honest_completion = False
                    print(
                        "  [warn] completion-gate evaluation failed: "
                        f"{type(gate_error).__name__}: {gate_error}"
                    )
                try:
                    _detail = governor_stop_detail or getattr(
                        getattr(self.state.context, "_governor", None),
                        "shutdown_reason",
                        None,
                    )
                    # A cap exception is the primary error while its preserved
                    # status pair is the eligibility truth source. Unknown and
                    # forced-backoff governor reasons fail closed.
                    cap_shaped_end = (
                        finalization_error is None
                        and cleanup_error is None
                        and config.LINKEDIN_TOTAL_PAGE_CAP <= 0
                        and isinstance(finalized_run_dir, Path)
                        and (
                            (
                                stop_reason == RunStopReason.SESSION_EXPIRED
                                and run_status == "interrupted"
                            )
                            or (
                                stop_reason == RunStopReason.GOVERNOR_LIMIT
                                and run_status == "governor_limit_reached"
                                and _detail is not None
                                and str(_detail).startswith(
                                    CAP_SHAPED_SHUTDOWN_PREFIXES
                                )
                            )
                        )
                    )
                except Exception as gate_error:
                    cap_shaped_end = False
                    print(
                        "  [warn] market-intel eligibility evaluation failed: "
                        f"{type(gate_error).__name__}: {gate_error}"
                    )
                mi_session_eligible = honest_completion or cap_shaped_end
                try:
                    if mi_session_eligible:
                        if self._strategy_fully_executed():
                            self._enrich_run_snapshot(
                                finalized_run_dir,
                                with_external_research=True,
                            )
                        else:
                            self._enrich_run_snapshot(
                                finalized_run_dir,
                                with_external_research=False,
                            )
                    else:
                        deferred_line = (
                            "  Market intel deferred: session ineligible "
                            f"(reason={stop_reason}"
                        )
                        deferred_receipt = {
                            "reason": stop_reason,
                            "run_id": self.state.context._runtime_run_id,
                        }
                        if stop_reason == RunStopReason.GOVERNOR_LIMIT:
                            deferred_line += f"; detail={governor_stop_detail}"
                            deferred_receipt["detail"] = governor_stop_detail
                        deferred_line += ")."
                        print(deferred_line)
                        log_event(
                            self.state.context.log_path,
                            "market_intel_deferred",
                            **deferred_receipt,
                        )
                except Exception as enrich_error:
                    # KeyboardInterrupt intentionally still escapes this guard.
                    print(f"  [warn] Market intel enrichment failed: {enrich_error}")
                try:
                    self.state._run_report_service._print_summary()
                except Exception as e:
                    print(f"  [warn] Summary print failed: {e}")
                if (
                    honest_completion
                    and
                    self.state._progress
                ):
                    try:
                        log_event(self.state.context.log_path, "report_started")
                        self.state._run_report_service._generate_run_report(self.state._progress)
                        log_event(self.state.context.log_path, "report_completed")
                    except Exception as e:
                        print(f"  [warn] Run report generation failed: {e}")
                        try:
                            log_event(
                                self.state.context.log_path,
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
                        self.state.context.log_path,
                        "pipeline_end",
                        **self.state._run_report_service._pipeline_end_stats(),
                    )
                except Exception as e:
                    print(f"  [warn] Pipeline-end logging failed: {e}")
                self.state._run_report_service._persist_cost_rollup_sidecar()
            if primary_error is None and finalization_error is not None:
                raise finalization_error

    def _enforce_constraint_manifest(self) -> None:
        """Build, persist, and enforce the constraint-ownership manifest.

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

        manifest = build_constraint_manifest(self.state.context.brief_obj)
        self.state._constraint_manifest = manifest
        write_json(self.state.context.output_dir / MANIFEST_FILENAME, manifest)
        owned = sum(
            1 for entry in manifest["classes"].values() if entry["status"] == "owned"
        )
        print(
            f"  [constraints] Manifest: {owned} owned / "
            f"{len(manifest['classes'])} classes → {self.state.context.output_dir / MANIFEST_FILENAME}"
        )
        assert_constraint_manifest_runnable(manifest)

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

        raw_intake = getattr(self.state.context.brief_obj, "intake_notes", "")
        intake_notes = raw_intake.strip() if isinstance(raw_intake, str) else ""
        print(f"  Preflight V2... ({config.STRATEGY_MODEL_NAME.rsplit('/', 1)[-1]} generating structured eval criteria from {'JD + intake notes' if intake_notes else 'JD'})")

        parent_logical_call_id = f"preflight_{uuid.uuid4().hex}"
        base_usage_context = {
            "stage": PREFLIGHT_STAGE,
            "brief_id": self.state.context.brief_obj.id,
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
                shadow_dir=self.state.context.output_dir / "shadow_strategy",
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
                self.state.context.brief_obj,
                model_name=config.STRATEGY_MODEL_NAME,
                usage_context=base_usage_context,
                on_raw_response=_dispatch_shadow,
                on_findings=print,
            )

        # A V2 preflight failure retries once, then aborts the run.
        # It never falls back to a legacy preflight regime: that regime's hardcoded
        # archetype would restore the permissive evaluation path V2 was designed to prevent.
        # A run on the wrong evaluation regime is worse than no run.
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

        # Keep this retry boundary: override, provenance, and typed loading happen once
        # after provider, parse, and lint succeed. A loader defect must never spend a
        # second provider call.
        execution = finalize_preflight_v2(self.state.context.brief_obj, generation)
        preflight_data = generation.data
        brief_json = execution.brief_json

        # Print the review rendering so an operator watching the run or reading the log
        # sees the evaluation criteria before any candidate is judged against them.
        print(format_for_review(preflight_data))

        # Save the generated V2 brief for operator review and debugging
        generated_path = self.state.context.output_dir / "preflight_v2_brief.json"
        write_json(str(generated_path), brief_json)
        print(f"  Preflight V2 brief saved to: {generated_path}")
        print(f"  Capability areas: {len(brief_json.get('capability_areas', []))}")
        print(f"  Non-fit patterns: {len(brief_json.get('non_fit_patterns', []))}")

        # The shared seam already proved typed V2 loading before returning.
        self.state.context.brief_obj = execution.brief
        init_judger(self.state.context.brief_obj)

        # Initialize bias monitor now that we have a V2 brief
        if self.state.context.brief_obj.has_v2_schema:
            self.state.context._bias_monitor = BiasMonitor.from_brief(self.state.context.brief_obj._new_brief)

        print("  Preflight V2 complete — pipeline will use structural templates + bias controls")
        # Re-print confidence notes last: they are preflight's open questions for the operator.
        # Answers flow back through the seed brief's ``instructions`` calibration channel.
        notes_block = format_confidence_notes(preflight_data)
        if notes_block:
            print(notes_block)

    def _archive_stale_outputs(self) -> None:
        """Move stale per-run artifacts out of the mutable state_dir before a fresh run."""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_dir = source_archive_root(
            "linkedin",
            self.state.context._brief_id,
            output_root=self.state.context.output_dir,
        ) / "state-resets" / ts
        for path in [
            self.state.snippets_path,
            self.state.facial_path,
            self.state.profiles_path,
            self.state.final_path,
            self.state.context.output_dir / "shadow_judgments.jsonl",
        ]:
            if path.exists() and path.stat().st_size > 0:
                archive_dir.mkdir(parents=True, exist_ok=True)
                backup = archive_dir / path.name
                path.rename(backup)
                print(f"  Archived {path.name} → {backup}")
        shadow_strategy_dir = self.state.context.output_dir / "shadow_strategy"
        if shadow_strategy_dir.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
            backup = archive_dir / shadow_strategy_dir.name
            shadow_strategy_dir.rename(backup)
            print(f"  Archived {shadow_strategy_dir.name} → {backup}")
        # The criteria behind every verdict above. Preflight and the manifest
        # overwrite these in place on the fresh run, and no snapshot or log
        # keeps them, so the reset copies them (a copy, never a move: nothing
        # here may change what the state dir holds before preflight runs).
        for name in ("preflight_v2_brief.json", "constraint_manifest.json"):
            path = self.state.context.output_dir / name
            if path.exists() and path.stat().st_size > 0:
                archive_dir.mkdir(parents=True, exist_ok=True)
                backup = archive_dir / name
                shutil.copy2(path, backup)
                print(f"  Archived a copy of {name} → {backup}")

    def _restart_string(self, progress: Progress, string_id: int) -> None:
        self._ensure_services()
        self.state._allocator_service.work_units.restart_string(progress, string_id)

    def _restart_strings(self, progress: Progress, string_ids: list[int]) -> None:
        self._ensure_services()
        self.state._allocator_service.work_units.restart_strings(progress, string_ids)

    def _load_or_create_progress(self) -> Progress:
        self._ensure_services()
        return self.state._allocator_service.work_units.load_or_create_progress()

    def _finalize_run_snapshot(self) -> Path | None:
        return freeze_linkedin_run_snapshot(
            runtime_run_id=self.state.context._runtime_run_id,
            brief_path=self.state.brief_path,
            state_dir=self.state.state_dir,
            log_path=self.state.context.log_path,
        )

    def _strategy_fully_executed(self) -> bool:
        """True when every search string for this run reached a terminal status.

        Work units are re-created per run (26 distinct strings across 4 runs =
        104 rows in the 9000000005 store), so the question is scoped to THIS
        run's units, not the brief's lifetime.

        Fails OPEN — returns True — when the flag is off, when there is no run
        id, or when the store cannot be read. This gate exists to suppress a
        premature report, not to become a new way for the run to end without
        producing one; an unreadable store should not silently cost the operator
        their market intelligence.
        """
        if not config.MARKET_INTEL_REQUIRE_STRATEGY_COMPLETE:
            return True
        if not self.state.context._runtime_run_id or not self.state.context._runtime_bridge:
            return True
        try:
            units = self.state.context._runtime_bridge.store.list_work_units(
                int(self.state.context._runtime_run_id), kind=LINKEDIN_STRING_KIND
            )
        except Exception as exc:
            print(f"  [warn] strategy-completion check failed, allowing: {exc}")
            return True
        if not units:
            return True
        # Use the same predicate canonical resume uses: only ``done`` and ``skipped`` are
        # complete. ``error`` is terminal status but remains pending for resume. A wider
        # terminal set could publish market intelligence from partial coverage.
        pending = [
            u for u in units
            if str(u.get("status")) not in _STRATEGY_COMPLETE_WORK_UNIT_STATUSES
        ]
        if pending:
            print(
                "  Market intel external research deferred: "
                f"{len(pending)} of {len(units)} search strings pending; "
                "session evidence folded without external refresh."
            )
            # Fail-soft: this gate must never become a new way for a run to end
            # without its report. log_event opens/writes a file and can raise.
            try:
                log_event(
                    self.state.context.log_path,
                    "market_intel_deferred",
                    pending_strings=len(pending),
                    total_strings=len(units),
                    run_id=int(self.state.context._runtime_run_id),
                    reason="strategy_pending",
                )
            except Exception as exc:  # noqa: BLE001 — a receipt must not abort finalization
                print(f"  [warn] market_intel_deferred receipt failed: {exc}")
            return False
        return True

    def _enrich_run_snapshot(
        self,
        run_dir: Path,
        *,
        with_external_research: bool,
    ) -> None:
        enrich_linkedin_run_snapshot(
            runtime_run_id=self.state.context._runtime_run_id,
            brief_path=self.state.brief_path,
            run_dir=run_dir,
            log_path=self.state.context.log_path,
            with_external_research=with_external_research,
        )
