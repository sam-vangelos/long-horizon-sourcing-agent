"""Declared process-local loop state; SQLite remains the canonical store.

Fields are initialized at their original lifecycle points. In particular,
init=False preserves absent day-cycle controls and lazy caches, whose absence
has different semantics from an inactive event or an empty collection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from linkedin.page_review import PageReviewContext
from linkedin.candidate_ledger import CandidateLedger
from linkedin.evaluation_context import EvaluationContext
from linkedin.search_surface import SearchSurface
from linkedin.candidate_review import CandidateReviewer
from linkedin.page_review import PageReviewer
from linkedin.string_driver import StringDriver
from linkedin.queue_planner import QueuePlanner
from linkedin.allocator_state import AllocatorStateService
from linkedin.runtime_attempts import RuntimeAttemptDeps, RuntimeAttemptService
from pathlib import Path
from typing import Any, ContextManager, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.brief_loader import Brief
    from shared.schemas import CandidateSnippet, CandidateProfileSummary, OpusDecision, SearchString, Progress, KitString, ExecutionPlan
    from shared.runtime_state import RuntimeStateStore, RuntimeStateLock, LinkedInRuntimeStateBridge
    from shared.safety import RunSafetyCoordinator, LinkedInRecoveryService
    from shared.governor import SessionGovernor
    from shared.bias_controls import BiasMonitor
    from linkedin.browser import LinkedInBrowser
    from linkedin.search_intelligence import LinkedInExperimentState
    from linkedin.evaluation_cascade import ProfileProbe, CascadeDecision
    from linkedin.timing_telemetry import RunLogTimingRecorder
    from linkedin.adaptation_signal_state import AdaptationGateConfig
    from linkedin.fallback_search import FallbackSearchProvider
    from linkedin.acquisition import LinkedInAcquisitionService
    from linkedin.work_units import LinkedInWorkUnitService
    from linkedin.side_effects import LinkedInSideEffectsService
    from linkedin.search_mutation import LinkedInSearchMutationExecutor
    from linkedin.allocator_state import AllocatorStateService
    from linkedin.geography_gate import GeographyGateService
    from linkedin.run_report import RunReportService
    from linkedin.runtime_attempts import RuntimeAttemptService
    from linkedin.block_adaptation import BlockAdaptationService


@dataclass
class PageContext:
    """Page observations and previews; commit and recovery state have separate owners."""
    review: PageReviewContext = field(default_factory=PageReviewContext)
    _last_card_extraction_error: Exception | None = field(init=False, repr=False)
    _latest_page_preview_snippets: list[CandidateSnippet] = field(init=False, repr=False)


@dataclass
class EvaluationRunState:
    """Stable holder for live run state and concrete execution collaborators."""

    pages: PageReviewer = field(init=False, repr=False)
    driver: StringDriver = field(init=False, repr=False)
    planner: QueuePlanner = field(init=False, repr=False)
    reviewer: CandidateReviewer = field(init=False, repr=False)
    surface: SearchSurface = field(init=False, repr=False)
    context: EvaluationContext = field(default_factory=EvaluationContext)
    candidates: CandidateLedger = field(default_factory=CandidateLedger)
    page: PageContext = field(default_factory=PageContext)
    _allocator_service: AllocatorStateService = field(init=False, repr=False)
    _constraint_manifest: dict[str, Any] = field(init=False, repr=False)
    _experiment_states: dict[int, LinkedInExperimentState] = field(init=False, repr=False)
    _llm_usage_cm: ContextManager[None] = field(init=False, repr=False)
    _progress: Progress | None = field(init=False, repr=False)
    _recovery_service: LinkedInRecoveryService = field(init=False, repr=False)
    _run_report_service: RunReportService | None = field(init=False, repr=False)
    _runtime_lock: RuntimeStateLock = field(init=False, repr=False)
    _safety: RunSafetyCoordinator = field(init=False, repr=False)
    bias_checkpoint_path: Path = field(init=False, repr=False)
    brief: Any = field(init=False, repr=False)
    brief_path: str = field(init=False, repr=False)
    facial_path: Path = field(init=False, repr=False)
    final_path: Path = field(init=False, repr=False)
    history_path: Path = field(init=False, repr=False)
    input_mode: str = field(init=False, repr=False)
    noise_path: Path = field(init=False, repr=False)
    profiles_path: Path = field(init=False, repr=False)
    progress_path: Path = field(init=False, repr=False)
    runtime_db_path: Path = field(init=False, repr=False)
    search_memory_path: Path = field(init=False, repr=False)
    snippets_path: Path = field(init=False, repr=False)
    state_dir: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # These owners do no I/O at construction. Runtime bindings are read only
        # when called, after EvaluationLoop initialization and run creation.
        attempts = RuntimeAttemptService(
            RuntimeAttemptDeps(
                get_runtime_bridge=lambda: self.context._runtime_bridge,
                get_runtime_run_id=lambda: self.context._runtime_run_id,
                get_runtime_state=lambda: self.context._runtime_state,
            )
        )
        self._allocator_service = AllocatorStateService(
            context=self.context,
            candidates=self.candidates,
            events=attempts,
            get_experiment_states=lambda: self._experiment_states,
        )
        self.surface = SearchSurface(self.context, self.page, self._allocator_service)
        self.reviewer = CandidateReviewer(
            self.context,
            self.candidates,
            self.surface,
            self._allocator_service,
            self.page,
        )
        self.pages = PageReviewer(
            self.context,
            self.candidates,
            self.surface,
            self.reviewer,
            self._allocator_service,
            self.page,
        )
        self.planner = QueuePlanner(self.context, self.candidates, self._allocator_service)
        self.driver = StringDriver(
            self.context,
            self.candidates,
            self.surface,
            self.reviewer,
            self.pages,
            self._allocator_service,
            self.planner,
            self.page,
        )
        self.reviewer.attempts = attempts
        self.surface.attempts = attempts
        self.planner.events = attempts
