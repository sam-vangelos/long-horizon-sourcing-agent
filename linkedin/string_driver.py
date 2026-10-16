"""StringDriver implementation with explicit concrete collaborators."""
from __future__ import annotations
from linkedin.evaluation_support import _is_foreign_project_page
from linkedin import search_policy as _search_policy
import asyncio
import json
import traceback
from typing import Any
from shared.human_timing import human_delay_correlated
from shared.schemas import CandidateSnippet, SearchString, Progress
from linkedin.page_allocator import (
    AllocationAction,
    AllocationVerdict,
    AllocatorPolicyError,
    allocate_page,
)
from linkedin.search_intelligence import (
    seed_structured_filters_onto_variants,
    LinkedInExperimentState,
    LinkedInPageInsights,
    LinkedInSearchVariant,
    LinkedInStructuredFilters,
    LinkedInVariantSnapshot,
    _copy_filters,
    bootstrap_experiment_state,
    result_window_for_count,
    scale_window_for_surface,
    spawn_rescue_variant_from_hint,
)
from linkedin.matching_contract import render_adaptation_matching_guidance
from shared.failures import judgment_failure_decision
from shared.storage import log_event
from linkedin.run_report import _PageReport
from shared import config
from shared.governor import GovernorLimitReached, OperatorStopRequested, SessionExpired
from linkedin.evaluation_errors import TransientPaginationError
from linkedin.page_review import PageReviewContext
from linkedin.competitor_scope import (
    competitor_facet_skip_reason,
    competitor_scope_prompt_block,
    stamp_competitor_facet_for_brief,
)
from shared.employer_allowlist import employer_allowlist_from_brief


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linkedin.evaluation_context import EvaluationContext
    from linkedin.candidate_ledger import CandidateLedger
    from linkedin.search_surface import SearchSurface
    from linkedin.candidate_review import CandidateReviewer
    from linkedin.page_review import PageReviewer
    from linkedin.allocator_state import AllocatorStateService
    from linkedin.queue_planner import QueuePlanner
    from linkedin.evaluation_state import PageContext
    from linkedin.search_mutation import LinkedInSearchMutationExecutor


class StringDriver:
    """One string execution frame and its lifecycle transitions."""

    _PENDING_FULL_RECOVERY_MAX_FAILED_ATTEMPTS = 2

    _completed_pages_this_process: int
    _string_ids_with_results: set[int]
    _search_mutation_budget_used: int
    _search_mutation_executor: LinkedInSearchMutationExecutor | None

    def __init__(
        self,
        context: EvaluationContext,
        ledger: CandidateLedger,
        surface: SearchSurface,
        reviewer: CandidateReviewer,
        pages: PageReviewer,
        allocator: AllocatorStateService,
        planner: QueuePlanner,
        page: PageContext,
    ) -> None:
        self.context, self.ledger = context, ledger
        self.surface, self.reviewer, self.pages = surface, reviewer, pages
        self.allocator, self.planner, self.page = allocator, planner, page
        self._search_mutation_executor = None

    def _note_string_results_seen(
        self, search_string: SearchString, result_count: int
    ) -> None:
        if result_count > 0:
            self._string_ids_with_results.add(search_string.id)


    def _honor_active_allocator_stop_boundary(self) -> None:
        """Let external stop authority interpose before queue actuation."""

        if not self.allocator._allocator_active_enabled():
            return
        if (
            hasattr(self.context, "_operator_stop_event")
            and self.context._operator_stop_event.is_set()
        ):
            raise OperatorStopRequested()
        if hasattr(self.context, "_session_expired") and self.context._session_expired.is_set():
            raise SessionExpired("session_duration_cap")
        governor_reason = self.context._governor.check_limits()
        if governor_reason:
            raise GovernorLimitReached(governor_reason)


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
        return self.allocator._resume_active_allocator_actuation(progress, work_units=self.allocator.work_units, resumable_string_ids=self.ledger.resumable_string_ids())


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

        self.allocator._stage_allocator_exhaustion(
            progress=progress,
            search_string=search_string,
            page_num=page_num,
        )
        if not self.allocator._allocator_active_enabled():
            search_string.status = legacy_terminal_status
            self.allocator._checkpoint_progress(
                progress,
                search_string=search_string,
                page_num=page_num if completed_page else None,
                completed_page=completed_page,
            )
            return None
        # Phase one deliberately preserves the live frontier pre-image. The
        # persisted verdict is the redo record for the second transaction.
        self.allocator._checkpoint_progress(
            progress,
            completed_page=completed_page,
        )
        if completed_page:
            self.allocator._discard_incomplete_page_rollback(search_string.id)
        self._honor_active_allocator_stop_boundary()
        verdict = self.allocator._resume_active_allocator_actuation(progress, work_units=self.allocator.work_units, resumable_string_ids=self.ledger.resumable_string_ids())
        if verdict is None:
            raise AllocatorPolicyError(
                "active physical-exhaustion verdict was not actuated"
            )
        return verdict


    def _prepare_active_allocator_dispatch(
        self,
        *,
        progress: Progress,
        current: SearchString,
    ) -> AllocationVerdict | None:
        if not self.allocator._allocator_active_enabled():
            return None
        self._honor_active_allocator_stop_boundary()
        replayed = self.allocator._resume_active_allocator_actuation(progress, work_units=self.allocator.work_units, resumable_string_ids=self.ledger.resumable_string_ids())
        if replayed is not None and (
            replayed.action is not AllocationAction.CONTINUE
            or replayed.selected_root_id != current.id
            or self.allocator._allocator_terminal_status(current.status)
        ):
            return replayed
        self.allocator._validate_active_allocator_canary(
            progress=progress,
            current=current,
        )
        arms = self.allocator._allocator_arms(progress=progress, current=current)
        verdict = allocate_page(current_root_id=current.id, arms=arms)
        if not self.allocator._allocator_verdict_requires_actuation(verdict):
            return None
        self.allocator._stage_active_allocator_dispatch(
            progress=progress,
            current=current,
            arms=arms,
            verdict=verdict,
        )
        self.allocator._checkpoint_progress(progress)
        self._honor_active_allocator_stop_boundary()
        actuated = self.allocator._resume_active_allocator_actuation(progress, work_units=self.allocator.work_units, resumable_string_ids=self.ledger.resumable_string_ids())
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


    def _stamp_competitor_facet(
        self, search_string: SearchString, *, queue: list[SearchString] | None = None
    ) -> None:
        stamp_competitor_facet_for_brief(
            search_string,
            self.context.brief_obj,
            queue=list(queue or []),
            log_path=getattr(self.context, "log_path", None),
        )


    def _competitor_facet_skip(
        self,
        search_string: SearchString,
        experiment_state: LinkedInExperimentState,
    ) -> str | None:
        """Why this string must not run on its live surface, or None.

        An allowlisted campaign never runs a string whose competitor facet did
        not land: without it the string searches the whole Location filter on
        keywords alone. Records the reason on the string and the run log.
        """

        allowlist = employer_allowlist_from_brief(self.context.brief_obj)
        if allowlist is None:
            return None
        filters = experiment_state.active_variant.structured_filters
        reason = competitor_facet_skip_reason(
            list(filters.companies or []),
            getattr(self.surface, "last_opening_facet_receipt", None),
            allowlist,
            requested_titles=list(getattr(filters, "titles", None) or []),
        )
        if not reason:
            return None
        print(f"  [scope] String #{search_string.id} skipped: {reason}")
        search_string.notes = (
            (search_string.notes or "") + f" Skipped: {reason}."
        ).strip()
        if getattr(self.context, "log_path", None):
            log_event(
                self.context.log_path,
                "competitor_facet_skip",
                string_id=search_string.id,
                reason=reason,
            )
        return reason


    def _record_competitor_facet_refusals(
        self, search_string: SearchString, refused: list[str]
    ) -> None:
        """One console line and run-log event per refused companies move."""

        if not refused:
            return
        print(
            f"    [scope] String #{search_string.id}: refused "
            f"{', '.join(refused)} (the competitor facet stays)"
        )
        if getattr(self.context, "log_path", None):
            log_event(
                self.context.log_path,
                "competitor_facet_demote_refused",
                string_id=search_string.id,
                refused=list(refused),
            )


    def _set_search_mutation_budget_used(self, budget_used: int) -> None:
        self._search_mutation_budget_used = budget_used


    def _experiment_state_for(self, search_string: SearchString) -> LinkedInExperimentState:
        state = self.allocator.experiment(search_string.id)
        if state is None:
            # Last stop before a string's facet is frozen into its root
            # variant: any path that queued it without a competitor facet
            # (inert unless the brief carries an employer_allowlist).
            self._stamp_competitor_facet(search_string)
            state = bootstrap_experiment_state(search_string)
            self.allocator.register_experiment(state)
        state.apply_shadow(search_string)
        return state


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
        self.reviewer.attempts._record_runtime_event(
            search_string=search_string,
            event_type="total_page_cap_reached",
            payload=payload,
        )
        log_event(self.context.log_path, "total_page_cap_reached", **payload)
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
            self.planner._maybe_discover_fallback_candidates(
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


    async def _process_string(
        self,
        search_string: SearchString,
        progress: Progress,
    ) -> AllocationVerdict | None:
        """Process a string while keeping cursor-N recovery checkpoints non-teaching."""
        self.allocator._discard_incomplete_page_rollback(search_string.id)
        try:
            return await self._process_string_impl(search_string, progress)
        except BaseException:
            # Preserve the in-flight exception if rollback fails during unwind.
            try:
                self.allocator._restore_incomplete_page_rollback(search_string)
            except Exception as rollback_error:
                print(
                    "  [warn] incomplete-page rollback restore failed "
                    f"({type(rollback_error).__name__}: {rollback_error}); "
                    "propagating the original error."
                )
            raise
        finally:
            self.allocator._discard_incomplete_page_rollback(search_string.id)
            pending = getattr(self.allocator, "_pending_allocator_checkpoint", None)
            if (
                isinstance(pending, dict)
                and int(pending.get("root_string_id", 0) or 0)
                == search_string.id
            ):
                self.allocator._pending_allocator_checkpoint = None
            self.page.review._allocator_page_identity = None


    async def _process_string_impl(
        self,
        search_string: SearchString,
        progress: Progress,
    ) -> AllocationVerdict | None:
        # Reset triage tightening state per string
        self.reviewer._triage_tightened = False
        self.reviewer._tightening_prefix = ""
        if self.allocator.experiment(search_string.id) is None:
            self._stamp_competitor_facet(search_string, queue=list(progress.strings))
        experiment_state = self._experiment_state_for(search_string)

        dispatch_verdict = self._prepare_active_allocator_dispatch(
            progress=progress,
            current=search_string,
        )
        if dispatch_verdict is not None:
            return dispatch_verdict

        # Classify this first actual spend against the persisted counterfactual
        # frontier before any page evidence can be opened.
        self.allocator._check_allocator_pre_spend(
            progress=progress,
            current=search_string,
        )

        # Emergency recovery check before starting. The opening transition
        # below re-establishes the full owner surface, so a recovery
        # navigation here needs no mid-string stop (F11).
        await self.surface._ensure_browser_healthy(owner_restore_follows=True)

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
                await self.context.browser.go_back_to_results()
            except Exception:
                pass

            # Ensure we're on a search page — this brief's, not merely someone's.
            # `/talent/hire/` present was the whole test, so ANY project's
            # Recruiter page satisfied resume and the owner's Boolean was
            # re-entered on it. Resume is exactly where this bites: the page was
            # bound by a rebind or a crash recovery that had no idea which
            # project the run owns. `_is_foreign_project_page` applies F1's
            # asymmetry — a page that does not NAME the brief's project is
            # unverified, and unverified is wrong — so the navigation below
            # corrects it instead of the run inheriting a foreign pipeline for
            # every review and save on the page. The two original clauses stay
            # because with no project pinned the predicate is False by design.
            current_page_url = self.context.browser.page.url
            if (
                "/manage/" in current_page_url
                or "/talent/hire/" not in current_page_url
                or _is_foreign_project_page(
                    current_page_url, self.context.brief_obj.linkedin_project_id
                )
            ):
                project_url = self.surface._get_project_url()
                if project_url:
                    print(f"  Wrong page ({current_page_url[:60]}...). Navigating to search...")
                    await self.context.browser.navigate_to_search(project_url)
                else:
                    print(f"  Wrong page and no project URL. Reloading...")
                    await self.context.browser.page.reload(wait_until="domcontentloaded", timeout=30000)
                    await self.context.browser.page.wait_for_timeout(4000)
                # The (re)navigation dropped the sidebar; re-assert the session location.
                self.surface.geography.reset_location_applied()
                await self.surface._apply_session_location_filter()

            # Always re-enter keywords; browser state is not trustworthy on resume.
            # A hybrid lane also re-applies its persisted structured filters through
            # the applied-only snapshot path; Boolean lanes keep bare keyword entry.
            print(f"  Re-entering Boolean: {current_boolean[:80]}...")
            await self.surface._apply_opening_search(search_string, experiment_state, current_boolean)
            result_count_text = await self.context.browser.get_results_count_text()
            result_count = await self.context.browser.get_results_count()
            search_string.result_count = result_count
            self._note_string_results_seen(search_string, result_count)
            print(f"  Results: {result_count_text or 'unknown'}")

            log_event(self.context.log_path, "string_resumed", string_id=search_string.id,
                      pages_done=search_string.pages_reviewed,
                      result_count=result_count, result_count_text=result_count_text)
        else:
            # Fresh string — enter the Boolean or the lane's structured surface.
            print(f"  Entering Boolean: {current_boolean[:80]}...")
            await self.surface._apply_opening_search(search_string, experiment_state, current_boolean)

            # Snapshot URL after search entry — known good state for recovery
            try:
                self.surface._record_last_good_url(self.context.browser.page.url)
            except Exception:
                pass

            result_count_text = await self.context.browser.get_results_count_text()
            result_count = await self.context.browser.get_results_count()
            search_string.result_count = result_count
            self._note_string_results_seen(search_string, result_count)
            print(f"  Results: {result_count_text or 'unknown'} (parsed: {result_count})")
            log_event(self.context.log_path, "string_results", string_id=search_string.id,
                      result_count=result_count, result_count_text=result_count_text)

        # Validate canonical pending work only after this owner's exact search
        # surface is active. Foreign work stays inert until its own turn.
        owned_pending = (
            []
            if self.allocator._allocator_active_enabled()
            else self._validated_owner_pending_full_snippets(
                progress=progress,
                search_string=search_string,
            )
        )
        owned_pending = self._resolve_ahead_of_cursor_pending(
            progress=progress,
            search_string=search_string,
            experiment_state=experiment_state,
            owned_pending=owned_pending,
            first_incomplete_page=first_incomplete_page,
        )

        # An allowlisted campaign never runs a string keyword-only across the
        # country: a competitor facet that did not land skips the string. A
        # resumed string with canonical pending reviews keeps its surface so
        # those reviews settle; the cards still pass the scope gate.
        # ``facet_skip_reason`` set below means the string runs only the page
        # that settles its pending reviews, then stops as skipped.
        facet_skip_reason = self._competitor_facet_skip(search_string, experiment_state)
        if facet_skip_reason:
            if not owned_pending:
                return self._checkpoint_allocator_exhaustion_transition(
                    progress=progress,
                    search_string=search_string,
                    page_num=first_incomplete_page,
                    legacy_terminal_status="skipped",
                )

        # Guard: don't proceed with invalid result count.
        if result_count <= 0:
            if owned_pending:
                search_string.status = "in_progress"
                self.allocator._checkpoint_progress(progress, search_string=search_string)
                raise RuntimeError(
                    "active owner has canonical pending full reviews but its "
                    "search surface returned no results"
                )
            if search_string.id in self._string_ids_with_results:
                try:
                    log_event(
                        self.context.log_path,
                        "zero_results_after_prior_results",
                        string_id=search_string.id,
                        result_count=result_count,
                    )
                except Exception:
                    pass
            print(f"  No results detected (result_count={result_count}). Skipping string.")
            # This return leaves the browser on LinkedIn's no-results page,
            # which has no result summary and no cards. The next string's
            # opening gates read the page BEFORE typing their own Boolean, so
            # without this mark the dead page becomes every following string's
            # surface (Addendum I2). Zero results are routine — 2 of 13, 3 of
            # 24, 1 of 7 across the last three runs — so the next opening
            # re-anchors rather than the run discovering it three gate
            # failures later. Only the flag-on opening reads the mark, so the
            # flag-off path does not set it — a flag-off path is either
            # byte-identical state or it is not proven identical.
            if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
                self.surface.mark_surface_for_reanchor("zero_results")
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
        string_stats = self.ledger._string_stats_for_processing(
            search_string,
            resuming=resuming,
        )
        resume_surface_page = 1
        if (
            not self.allocator._allocator_active_enabled()
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
            experiment_state.mode = _search_policy._initial_search_mode(result_count)
            if (
                experiment_state.mode == "paginate"
                and not experiment_state.committed_variant_id
            ):
                # Direct pagination in a sub-RECON pool commits the root variant.
                # Without it committed-path accounting never arms, so the stop rule
                # cannot fire and a small pool paginates until it is exhausted.
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
            self.allocator._allocator_active_enabled()
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
                ) = await self.surface._go_to_next_page_with_transient_retry(
                    search_string=search_string,
                    result_count=result_count,
                    page_num=page_at_exhaustion,
                )
                if not has_next:
                    print("  No more pages after resume point.")
                    pages_remaining_by_math = self.surface._pages_remaining_by_result_count(
                        result_count,
                        page_at_exhaustion,
                    )
                    try:
                        log_event(
                            self.context.log_path,
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
                        self.allocator._checkpoint_progress(
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
            self.allocator._check_allocator_pre_spend(
                progress=progress,
                current=search_string,
            )
            # Emergency recovery check before each page
            await self.surface._ensure_browser_healthy()

            phase_label = search_string.phase.upper()
            refinement_depth = len(search_string.refinement_stack)
            print(f"\n  --- Page {page_num} [{phase_label}] (refinement depth: {refinement_depth}) ---")
            progress.current_page = page_num
            # Cursor N means page N has started but is not yet complete. Any
            # checkpoint during review therefore retries N after a crash.
            self.allocator._checkpoint_progress(
                progress,
                search_string=search_string,
                page_num=page_num,
                cursor_updates=((search_string.id, experiment_state.active_variant_id, page_num),),
            )
            self.allocator._arm_incomplete_page_rollback(search_string, experiment_state)

            # Check for 0-result page before trying to scroll/extract
            no_results_visible = False
            try:
                no_results = self.context.browser.page.locator('text="No search results"').first
                no_results_visible = await no_results.is_visible(timeout=2000)
            except Exception:
                pass
            if no_results_visible:
                if any(
                    max(1, int(snippet.page or 1)) == page_num
                    for _key, snippet in owned_pending
                ):
                    search_string.status = "in_progress"
                    self.allocator._checkpoint_progress(
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
                self.allocator._discard_incomplete_page_rollback(search_string.id)
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

            glance_result = await self.pages._review_page_sequentially(
                search_string=search_string,
                page_num=page_num,
                result_count=result_count,
                page_report=page_report,
                all_candidates=all_candidates,
                string_stats=string_stats,
                progress=progress,
            )

            page_context = self.page.review
            self.pages._assert_page_full_reviews_settled(
                progress=progress,
                search_string=search_string,
                page_num=page_num,
                page_context=page_context,
            )
            page_observed = self.pages._page_observation(page_context)
            # Compute immutable shadow evidence before legacy page metrics can
            # make this just-observed page look like pre-existing depth.
            self.allocator._stage_allocator_page_checkpoint(
                progress=progress,
                search_string=search_string,
                experiment_state=experiment_state,
                page_num=page_num,
                page_observed=page_observed,
                page_context=page_context,
            )
            allocator_controls_page = self.allocator.bypasses_local_page_policy()
            if (
                int(page_observed.get("extracted", 0) or 0)
                < int(page_observed.get("slots", 0) or 0)
                and not page_observed.get("break_reason")
            ):
                try:
                    log_event(
                        self.context.log_path,
                        "page_observation_gap",
                        string_id=search_string.id,
                        page=page_num,
                        slots=int(page_observed.get("slots", 0) or 0),
                        extracted=int(page_observed.get("extracted", 0) or 0),
                    )
                except Exception as e:
                    print(f"  [warn] page_observation_gap event failed: {e}")

            string_stats["pages"] = page_num

            page_stats = self.ledger._page_stat_delta(stats_before_page, string_stats)
            page_insights = _search_policy._build_page_insights(
                page_num=page_num,
                result_count=result_count,
                preview_snippets=list(self.page._latest_page_preview_snippets),
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
            page_report.print_report(self.context.stats)

            if facet_skip_reason:
                # Its competitor facet did not land: this page settled the
                # string's pending reviews, and nothing more runs keyword-only.
                lifecycle_decision = None
                assessment = {
                    "decision": "stop",
                    "rationale": "pending reviews settled; the competitor facet did not land",
                    "page": page_num,
                }
                decision = "stop"
            elif allocator_controls_page:
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
                    allow_terminal=not self.allocator._allocator_active_enabled(),
                )

                assessment = await self._assess_string_state(
                    search_string=search_string,
                    experiment_state=experiment_state,
                    page_num=page_num,
                    result_count=result_count,
                    string_stats=string_stats,
                    page_stats=page_stats,
                    page_insights=page_insights,
                    remaining_queued_strings=_search_policy._remaining_queued_strings(
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
                # Emit one verdict line per reviewed page so paginate mode exposes
                # each continue/stop decision.
                print(
                    f"  [adapt] Page {page_num} verdict: {decision} — "
                    f"{assessment['rationale']} (signal={assessment.get('page_signal')}, "
                    f"zero-signal streak={assessment.get('committed_zero_signal_streak')})"
                )

            if (
                self.allocator._allocator_active_enabled()
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
                        # A mid-run promote carrying structured filters executes as
                        # hybrid even on a compiled-Boolean lane; otherwise apply_variant
                        # rejects it as unsupported.
                        acquisition_mode=_search_policy._acquisition_mode_for_variant(
                            search_string, drift_variant
                        ),
                    )
                    if mutation.applied:
                        # The completed page and accepted query transition are
                        # checkpointed together. Until this point the durable
                        # cursor remains N, so a crash retries the undecided page.
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
                        self._settle_owner_pending_full_before_surface_change(
                            search_string=search_string,
                            reason="surface_replaced_by_drift",
                        )
                        completed_page_num = page_num
                        search_string.pages_reviewed = 0
                        page_num = 1
                        all_candidates.clear()
                        string_stats = self.ledger._fresh_string_stats()
                        self.allocator._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=1,
                            completed_page=True,
                            cursor_updates=((search_string.id, completed_variant.variant_id, completed_page_num + 1), (search_string.id, experiment_state.active_variant_id, 1)),
                        )
                        self.allocator._discard_incomplete_page_rollback(search_string.id)
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

                experiment_state.apply_shadow(search_string)
                search_string.pages_reviewed = int(committed_variant.pages_reviewed)
                print(
                    f"  [adapt] Re-entering committed variant on page {committed_page}: "
                    f"{current_boolean[:80]}..."
                )
                await self.surface._apply_opening_search(
                    search_string,
                    experiment_state,
                    current_boolean,
                )
                if self._competitor_facet_skip(search_string, experiment_state):
                    # Never fast-forward a keyword-only result set: persist the
                    # completed drift page, then skip the string.
                    self.allocator._checkpoint_progress(
                        progress,
                        search_string=search_string,
                        completed_page=True,
                        cursor_updates=((search_string.id, completed_variant.variant_id, failed_drift_page + 1), (search_string.id, experiment_state.active_variant_id, committed_page)),
                    )
                    self.allocator._discard_incomplete_page_rollback(search_string.id)
                    active_transition = self._finish_completed_page_allocator_boundary(
                        progress=progress,
                        search_string=search_string,
                        page_num=failed_drift_page,
                    )
                    if (
                        active_transition is not None
                        and active_transition.action is not AllocationAction.CONTINUE
                    ):
                        return active_transition
                    return self._checkpoint_allocator_exhaustion_transition(
                        progress=progress,
                        search_string=search_string,
                        page_num=committed_page,
                        legacy_terminal_status="skipped",
                    )
                result_count_text = await self.context.browser.get_results_count_text()
                result_count = await self.context.browser.get_results_count()
                search_string.result_count = result_count
                committed_variant.result_count = result_count

                for fastforward_index in range(committed_page - 1):
                    page_at_exhaustion = fastforward_index + 1
                    (
                        has_next,
                        transient_suspected,
                    ) = await self.surface._go_to_next_page_with_transient_retry(
                        search_string=search_string,
                        result_count=result_count,
                        page_num=page_at_exhaustion,
                    )
                    if has_next:
                        continue
                    print("  No more pages while restoring committed variant.")
                    pages_remaining_by_math = self.surface._pages_remaining_by_result_count(
                        result_count,
                        page_at_exhaustion,
                    )
                    try:
                        log_event(
                            self.context.log_path,
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
                    # First persist the completed drift page with its unchanged
                    # legacy restore action.  Physical exhaustion is a second,
                    # observation-free allocator revision.
                    self.allocator._checkpoint_progress(
                        progress,
                        search_string=search_string,
                        completed_page=True,
                        cursor_updates=((search_string.id, completed_variant.variant_id, failed_drift_page + 1), (search_string.id, experiment_state.active_variant_id, committed_page)),
                    )
                    self.allocator._discard_incomplete_page_rollback(search_string.id)
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

                page_num = committed_page
                progress.current_page = committed_page
                all_candidates.clear()
                string_stats = self.ledger._fresh_string_stats()
                search_string.notes = (
                    (search_string.notes or "")
                    + f" Drift rescue failed on page {failed_drift_page}; "
                    f"resumed committed variant on page {committed_page}."
                )
                # Persist the accepted restore only after the browser surface,
                # query identity, and variant-local cursor agree.
                self.allocator._checkpoint_progress(
                    progress,
                    search_string=search_string,
                    completed_page=True,
                    cursor_updates=((search_string.id, completed_variant.variant_id, failed_drift_page + 1), (search_string.id, experiment_state.active_variant_id, committed_page)),
                )
                self.allocator._discard_incomplete_page_rollback(search_string.id)
                active_transition = self._finish_completed_page_allocator_boundary(
                    progress=progress, search_string=search_string,
                    page_num=failed_drift_page,
                )
                if (active_transition is not None
                        and active_transition.action is not AllocationAction.CONTINUE):
                    return active_transition
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
                            # A mid-run promote carrying structured filters executes as
                            # hybrid even on a compiled-Boolean lane; otherwise apply_variant
                            # rejects it.
                            acquisition_mode=_search_policy._acquisition_mode_for_variant(
                                search_string, next_variant
                            ),
                        )
                        if mutation.applied:
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
                            self._settle_owner_pending_full_before_surface_change(
                                search_string=search_string,
                                reason="surface_replaced_by_variant",
                            )
                            completed_page_num = page_num
                            search_string.pages_reviewed = 0
                            page_num = 1
                            all_candidates.clear()
                            string_stats = self.ledger._fresh_string_stats()
                            self.allocator._checkpoint_progress(
                                progress,
                                search_string=search_string,
                                page_num=1,
                                completed_page=True,
                                cursor_updates=((search_string.id, completed_variant.variant_id, completed_page_num + 1), (search_string.id, experiment_state.active_variant_id, 1)),
                            )
                            self.allocator._discard_incomplete_page_rollback(search_string.id)
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

            if self.allocator._allocator_active_enabled() and decision == "stop":
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

            if decision == "stop" and facet_skip_reason:
                print(
                    f"  [scope] String #{search_string.id} stopped after page {page_num}: "
                    "its pending reviews settled"
                )
                search_string.status = "skipped"
            elif decision == "stop":
                self.planner._maybe_discover_fallback_candidates(
                    search_string,
                    trigger_reason="experiment_stop",
                )
                print("  [adapt] STOP — signal exhausted for this root family.")
                search_string.notes = (search_string.notes or "") + f" Stopped after page {page_num}."
                search_string.status = "done"

            # The page becomes durably complete only after its authoritative
            # local decision is reflected in state. A crash anywhere above
            # retries N; this checkpoint resumes at N+1 with the same action.
            experiment_state.apply_shadow(search_string)
            if int(config.LINKEDIN_TOTAL_PAGE_CAP) > 0:
                self._sync_bounded_page_stats_for_checkpoint(
                    search_string,
                    string_stats,
                )
            self.allocator._checkpoint_progress(
                progress,
                search_string=search_string,
                page_num=page_num,
                completed_page=True,
                cursor_updates=((search_string.id, completed_variant.variant_id, page_num + 1),),
            )
            self.allocator._discard_incomplete_page_rollback(search_string.id)
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
                has_next, transient_suspected = await self.surface._go_to_next_page_with_transient_retry(
                    search_string=search_string,
                    result_count=result_count,
                    page_num=page_num,
                )
                if not has_next:
                    print("  No more pages.")
                    pages_remaining_by_math = self.surface._pages_remaining_by_result_count(
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
                        self.allocator._checkpoint_progress(
                            progress,
                            search_string=search_string,
                        )
                        raise TransientPaginationError(
                            "transient pagination before next results page"
                        )
                    try:
                        log_event(
                            self.context.log_path,
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
                        self.context.log_path,
                        "page_cap_reached",
                        string_id=search_string.id,
                        page_num=page_num,
                        result_count=result_count,
                        max_pages=max_pages,
                    )
                except Exception as e:
                    print(f"  [warn] page_cap_reached event failed: {e}")
                if self.allocator._allocator_active_enabled():
                    search_string.status = "in_progress"
                    self.allocator._checkpoint_progress(progress)
                    raise OperatorStopRequested("max_pages_per_string")
                search_string.status = "done"
                self.allocator._checkpoint_progress(
                    progress,
                    search_string=search_string,
                )
                break

        # Persist cumulative funnel truth for block-level aggregation. On a
        # resume, string_stats starts from the canonical run-chain baseline.
        self._sync_bounded_page_stats_for_checkpoint(search_string, string_stats)
        self.planner._hydrate_search_string_metadata(search_string)


    def _validated_owner_pending_full_snippets(
        self,
        *,
        progress: Progress,
        search_string: SearchString,
    ) -> list[tuple[str, CandidateSnippet]]:
        try:
            return self.ledger.validated_pending(
                owner_id=search_string.id, queue_ids={item.id for item in progress.strings},
            )
        except RuntimeError:
            search_string.status = "in_progress"
            self.allocator._checkpoint_progress(progress, search_string=search_string)
            raise


    def _resolve_ahead_of_cursor_pending(
        self,
        *,
        progress: Progress,
        search_string: SearchString,
        experiment_state: LinkedInExperimentState,
        owned_pending: list[tuple[str, CandidateSnippet]],
        first_incomplete_page: int,
    ) -> list[tuple[str, CandidateSnippet]]:
        """Split owner-pending fulls recorded ahead of the resume cursor.

        On a stable sequential surface, pending work ahead of the cursor means
        cursor rollback and raising is correct. A committed mutation resets the
        cursor to page 1 of a new result set while pending metadata lacks variant
        identity, so snippets from the outgoing surface compare as ahead forever
        and cannot be revisited by navigation. When containment is enabled,
        settle those snippets through the receipted abandonment writer and return
        the behind-cursor remainder for normal recovery.
        """

        ahead = [
            (key, snippet)
            for key, snippet in owned_pending
            if max(1, int(snippet.page or 1)) > first_incomplete_page
        ]
        if not ahead:
            return owned_pending
        surface_changed = (
            str(getattr(experiment_state, "active_variant_id", "root") or "root")
            != "root"
        )
        if not (
            surface_changed
            and config.LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED
        ):
            search_string.status = "in_progress"
            self.allocator._checkpoint_progress(progress, search_string=search_string)
            raise RuntimeError(
                "canonical pending full review is ahead of the active owner cursor"
            )
        for key, snippet in ahead:
            try:
                self._abandon_unrecoverable_pending_full(
                    key=key,
                    snippet=snippet,
                    search_string=search_string,
                    reason="surface_changed_ahead_of_cursor",
                )
            except BaseException:
                # Same checkpoint-then-raise rule as recovery containment: the
                # string must persist resumable before the exception leaves.
                search_string.status = "in_progress"
                self.allocator._checkpoint_progress(progress, search_string=search_string)
                raise
        return [
            (key, snippet)
            for key, snippet in owned_pending
            if max(1, int(snippet.page or 1)) <= first_incomplete_page
        ]


    def _settle_owner_pending_full_before_surface_change(
        self,
        *,
        search_string: SearchString,
        reason: str,
    ) -> None:
        """Settle this owner's resume-hydrated pending fulls at a surface change.

        A committed mutation replaces the string's result set; any still-
        unsettled pending full review recorded against the outgoing surface
        (the "Carrying N unsettled review(s) forward" population) becomes
        unreachable the moment the cursor resets, and left in place it wedges
        the next resume as ahead-of-cursor work. Settle each with the receipted
        abandonment writer BEFORE the reset so the class cannot form. Flag-off
        preserves today's behavior (nothing settles; the resume-side guard
        raises as before).
        """

        if not config.LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED:
            return
        for key, snippet in self.ledger.pending_for_surface_change(search_string.id):
            if snippet is None:
                # No stored snippet means no canonical identity to settle
                # against; drop the in-process entry so it cannot re-wedge.
                self.ledger.forget_pending(key)
                continue
            self._abandon_unrecoverable_pending_full(
                key=key,
                snippet=snippet,
                search_string=search_string,
                reason=reason,
            )


    def _abandon_unrecoverable_pending_full(
        self,
        *,
        key: str,
        snippet: CandidateSnippet,
        search_string: SearchString,
        reason: str,
    ) -> None:
        """Terminally skip a pending full review the live surface cannot re-match.

        Recruiter results can reorder between sessions, so resume may not locate
        the card that produced a facial YES/BORDERLINE. Do not requeue or retry;
        settle the candidate on canonical state, receipt the skip, and let the
        remaining owned snippets recover.

        Use the same writer as other non-save terminal full decisions. A
        successful ``full`` attempt carrying ``full_decision`` ends the review;
        ``JUDGMENT_FAILURE`` stays out of reviewed/reject counters and
        ``DEDUP_BLOCKING_DECISIONS``. A pending row without a profile URL has no
        canonical identity and is a writer no-op; containment is in-process plus
        the receipt.
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
        abandon_attempt_id = self.reviewer.attempts._start_runtime_stage_attempt(
            search_string=search_string,
            snippet=snippet,
            stage="full",
            payload=abandon_payload,
        )
        self.reviewer._finish_runtime_stage_success(
            attempt_id=abandon_attempt_id,
            stage="full",
            snippet=snippet,
            decision=abandoned,
            extra_payload=dict(abandon_payload),
        )
        self.reviewer.attempts._record_runtime_event(
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
        self.ledger.forget_pending(key)
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
                self.allocator._checkpoint_progress(progress)
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
                failed_counts = self.ledger._failed_full_attempt_counts(
                    [str(snippet.profile_url or "") for _key, snippet in owned],
                    store=self.context._runtime_state,
                    run_id=self.context._runtime_run_id,
                    brief_id=self.context._brief_id,
                )
            except Exception as count_error:
                # The budget read is advisory; a store hiccup must not become a
                # resume-killer. No counts or abandonment happen in this pass.
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
                    await self.surface._go_to_next_page_with_transient_retry(
                        search_string=search_string,
                        result_count=int(search_string.result_count or -1),
                        page_num=rendered_page,
                    )
                )
                if not has_next:
                    search_string.status = "in_progress"
                    self.allocator._checkpoint_progress(progress)
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
                self.allocator._checkpoint_progress(progress)
                raise RuntimeError(
                    "pending full recovery requires a stable Recruiter profile URL"
                )
            card_index = await self.context.browser.find_result_slot_by_profile_url(
                snippet.profile_url
            )
            if card_index is None:
                if config.LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED:
                    contain(key=key, snippet=snippet, reason="unmatched_profile")
                    continue
                search_string.status = "in_progress"
                self.allocator._checkpoint_progress(progress)
                raise RuntimeError(
                    "pending full recovery could not match the exact Recruiter profile"
                )
            snippet.card_index = card_index
            print(
                f"    [recover-full] {snippet.name} — exact string "
                f"#{search_string.id} page {snippet.page}"
            )
            self.ledger.start_review(snippet.profile_url)

            page_context = PageReviewContext(
                _allocator_page_identity=(
                    search_string.id, self.surface._variant_id_for_search_string(search_string),
                    max(1, int(snippet.page or 1)),
                ),
                _allocator_page_off_policy=(
                    self.allocator._allocator_run_diverged() if self.allocator._allocator_shadow_enabled() else False
                ),
            )
            try:
                await self.reviewer._process_resumed_pending_full_evaluations(
                    snippets=[snippet],
                    page_report=None,
                    search_string=search_string,
                    all_candidates=[],
                    string_stats=string_stats,
                    progress=progress,
                    page_num=max(1, int(snippet.page or 1)),
                    preserve_progress_cursor=True,
                    page_context=page_context,
                )
            except BaseException:
                search_string.status = "in_progress"
                self.allocator._checkpoint_progress(progress)
                raise

            self._sync_bounded_page_stats_for_checkpoint(
                search_string,
                string_stats,
            )
            self.allocator._checkpoint_progress(progress)
            if self.ledger.has_pending(key):
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
            self.allocator._checkpoint_progress(progress)
            raise RuntimeError(
                "canonical pending full review remains unsettled for active owner"
            )
        return rendered_page


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
        scout_metrics = _search_policy._scout_metrics(page_stats=page_stats, page_insights=page_insights)
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
            decision, rationale, scout_gate_bucket = _search_policy._scout_quality_for_recon(
                result_count=result_count,
                scout_metrics=scout_metrics,
                precommit_recovery_attempts_used=experiment_state.precommit_recovery_attempts_used,
            )
        elif experiment_state.mode == "experiment":
            best_variant = experiment_state.best_variant()
            variant_earned_commit = _search_policy._variant_has_earned_commit(active_variant)
            best_variant_earned_commit = _search_policy._variant_has_earned_commit(best_variant)
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
            if _search_policy._variant_has_earned_commit(active_variant):
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
            drift_assessment = _search_policy._assess_pagination_drift(
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
            "page_observed": self.pages._page_observation(),
        }
        if decision == "commit" and page_signal > 0:
            payload["bootstrap_early_snapshot"] = True
        if decision in {"refine_committed", "spawn_recall_sibling"}:
            payload["drift_rescue_summary"] = dict(experiment_state.last_drift_refinement_summary)
        log_event(self.context.log_path, "linkedin_search_assess", string_id=search_string.id, **payload)
        self.reviewer.attempts._record_runtime_event(
            search_string=search_string,
            event_type="linkedin_search_assess",
            payload=payload,
        )
        return payload


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
        # After K structured demote-and-proceed events on a lane, close the
        # structured lever: remove the promote option from instructions and schema
        # so only keyword variants can be proposed. The execution metric constrains
        # proposals without changing the deterministic gate or lifecycle decision.
        structured_lever_open = (
            experiment_state.structured_demotions
            < config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT
        )
        allowed_promoted_titles, allowed_promoted_companies = (
            _search_policy._thresholded_structured_promotions(page_insights)
        )
        company_scope = employer_allowlist_from_brief(self.context.brief_obj)
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

Role: {self.context.brief_obj.role_title}
{self.context.brief_obj.role_description}{competitor_scope_prompt_block(self.context.brief_obj)}

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
                "brief_id": self.context.brief_obj.id,
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
                    refused: list[str] = []
                    structured_filters, surface, variant_kind, structured_changed = (
                        _search_policy._resolve_structured_controls(
                            raw_variant,
                            active_filters,
                            structured_lever_open=structured_lever_open,
                            allowed_promoted_titles=allowed_promoted_titles,
                            allowed_promoted_companies=allowed_promoted_companies,
                            company_scope=company_scope,
                            refused=refused,
                        )
                    )
                    self._record_competitor_facet_refusals(search_string, refused)
                    # A proposal with no Boolean or structured-filter change has
                    # nothing to execute.
                    if boolean == current_boolean and not structured_changed:
                        continue
                    # An empty Boolean is only runnable when the filters carry it
                    # (surface=="structured_only"); otherwise there is nothing to execute.
                    if not boolean:
                        if surface != "structured_only" or structured_filters.is_empty():
                            continue
                    # Scale the healthy window down for filter-led/structured variants
                    # so the keyword-tuned lifecycle gate accepts a narrower probe;
                    # Boolean variants remain unscaled.
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
                        self.context.log_path,
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
                self.context.log_path,
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
                # The forced-narrow fallback is a pure keyword variant with empty
                # filters, so scaling through the shared helper is a no-op.
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

        self.reviewer.attempts._record_runtime_event(
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
            _search_policy._thresholded_structured_promotions(promotion_evidence)
        )
        company_scope = employer_allowlist_from_brief(self.context.brief_obj)
        variant_kind = "recall" if decision == "spawn_recall_sibling" else "precision"
        active_filters = experiment_state.active_variant.structured_filters
        current_filters_text = json.dumps(active_filters.to_dict(), indent=2)
        # After K structured demotions on a lane, remove the promote/structured
        # option from the drift prompt. The rescue stays keyword-only; the gate and
        # lifecycle decision are unchanged.
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

Role: {self.context.brief_obj.role_title}
{self.context.brief_obj.role_description}{competitor_scope_prompt_block(self.context.brief_obj)}

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
                "brief_id": self.context.brief_obj.id,
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
                self.context.log_path,
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
        # Resolve structured promote/demote output using the active variant's filters.
        # A hybrid promote may keep the Boolean and add filters, so keyword fallback
        # applies only to a pure keyword rescue. A parse fault must degrade this
        # string to keyword fallback and never escape the planner.
        try:
            refused: list[str] = []
            structured_filters, surface, resolved_kind, structured_changed = (
                _search_policy._resolve_structured_controls(
                    payload,
                    active_filters,
                    structured_lever_open=structured_lever_open,
                    allowed_promoted_titles=allowed_promoted_titles,
                    allowed_promoted_companies=allowed_promoted_companies,
                    company_scope=company_scope,
                    refused=refused,
                )
            )
            self._record_competitor_facet_refusals(search_string, refused)
        except Exception as exc:
            log_event(
                self.context.log_path,
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
            boolean = _search_policy._fallback_drift_boolean(
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
        # Reject only when neither the Boolean nor structured filters changed. An
        # empty Boolean is runnable when ``surface=="structured_only"`` and filters carry it.
        boolean_unchanged = (not boolean or boolean == current_boolean)
        if boolean_unchanged and not structured_changed:
            return None, drift_summary
        if not boolean and (surface != "structured_only" or structured_filters.is_empty()):
            return None, drift_summary

        # ``variant_kind`` follows the resolved kind, not the raw model label. A
        # structured move uses ``resolved_kind``; a model-supplied KEYWORD kind is
        # preserved. A refused ``structured_filter`` label falls through to the
        # drift-decision kind so a closed lane cannot re-promote.
        raw_drift_kind = str(payload.get("variant_kind") or "")
        if structured_changed:
            resolved_variant_kind = resolved_kind
        elif raw_drift_kind and raw_drift_kind != "structured_filter":
            resolved_variant_kind = raw_drift_kind
        else:
            resolved_variant_kind = variant_kind
        # Scale the healthy window down for a filter-led/structured rescue so the
        # keyword-tuned lifecycle gate accepts a narrower probe. Boolean rescues
        # remain unscaled.
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

Role: {self.context.brief_obj.role_title}
{self.context.brief_obj.role_description}

## Minimum Bar
{self.context.brief_obj.minimum_bar}

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
                "brief_id": self.context.brief_obj.id,
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
                log_event(self.context.log_path, "forced_narrow", string_id=search_string.id,
                          rationale=rationale, new_boolean=new_boolean[:100])
                return f"narrow:{new_boolean}"
            else:
                print(f"  [adapt] Forced narrow returned no Boolean — honoring abandon.")
                try:
                    log_event(
                        self.context.log_path,
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
                    self.context.log_path,
                    "forced_narrow_failed",
                    string_id=search_string.id,
                    reason="exception",
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
            except Exception as event_error:
                print(f"  [warn] forced_narrow_failed event failed: {event_error}")
            return None

    async def finish_pending_reviews(self, search_string: SearchString, progress: Progress) -> None:
        """A string cannot finish while its canonical candidate work is pending."""
        if search_string.status == "in_progress":
            search_string.status = "done"
        if (
            search_string.status in {"done", "skipped"}
            and search_string.id in self.ledger.resumable_string_ids()
        ):
            terminal_status = search_string.status
            search_string.status = "in_progress"
            self.allocator._checkpoint_progress(
                progress,
                search_string=search_string,
            )
            experiment_state = self._experiment_state_for(search_string)
            await self.surface._apply_opening_search(
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
                    string_stats=self.ledger._string_stats_for_processing(
                        search_string,
                        resuming=True,
                    ),
                )
            except RuntimeError as exc:
                recovery_error = exc
            outstanding_keys = self.ledger.pending_keys(search_string.id)
            if outstanding_keys:
                search_string.status = "in_progress"
                self.allocator._checkpoint_progress(
                    progress,
                    search_string=search_string,
                )
                outstanding = self.ledger.describe_pending(outstanding_keys)
                outstanding_error = RuntimeError(
                    "outstanding full review(s) for string "
                    f"#{search_string.id}: {outstanding}"
                )
                # Chain recovery failure with outstanding keys so the traceback
                # carries both faults.
                if recovery_error is not None:
                    raise outstanding_error from recovery_error
                raise outstanding_error
            if recovery_error is not None:
                raise recovery_error
            search_string.status = terminal_status
