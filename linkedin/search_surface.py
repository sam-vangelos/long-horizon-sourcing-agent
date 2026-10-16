"""Physical search-surface ownership, restoration and recovery budgets."""
from __future__ import annotations
import asyncio
import time
from typing import Any
from linkedin.page_review import PageReviewContext
from rebrowser_playwright.async_api import (
    Error as BrowserError,
    TimeoutError as BrowserTimeoutError,
)
from shared.human_timing import human_delay, human_delay_correlated
from shared.schemas import CandidateSnippet, OpusDecision, SearchString, Progress
from linkedin.acquisition import (
    _is_browser_disconnect_error as _acquisition_is_browser_disconnect_error,
)
from linkedin.auth_wall import AuthWallError
from linkedin.browser import (
    FacetReconciliationError,
    LinkedInBrowser,
    facet_misses,
    facet_name_key,
    facet_target_landed,
    format_facet_reconciliation_console_line,
    format_facet_substituted_console_line,
    format_facet_unapplied_console_line,
    normalize_facet_value_for_compare,
    recruiter_project_search_url,
    _recruiter_page_project_id,
)
from linkedin.timing_telemetry import emit_timing_event
from linkedin.page_allocator import AllocatorPolicyError
from linkedin.search_intelligence import LinkedInExperimentState
from linkedin.recruiter_recovery import capture_recovery_snapshot
from linkedin.search_mutation import enforce_per_string_location_ownership
from linkedin.competitor_scope import competitor_facet_skip_reason
from shared.employer_allowlist import employer_allowlist_from_brief
from linkedin import surface_receipt as _surface_receipt
from shared.failures import ApiBudgetExhaustedError, is_api_budget_exhausted_error
from shared.runtime_state.store import LINKEDIN_STRING_KIND
from shared.storage import log_event
from shared import config
from shared.governor import GovernorLimitReached, OperatorStopRequested, SessionExpired
from linkedin.evaluation_errors import (
    GeographyRegimeError,
    PageRenderFailedError,
    PanelRecoveryError,
    TransientPaginationError,
    SurfaceOwnershipLostError,
    ProjectContextMismatchError,
)
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from linkedin.geography_gate import GeographyGateService
from linkedin.evaluation_context import EvaluationContext
from linkedin.allocator_state import AllocatorStateService
if TYPE_CHECKING:
    from linkedin.evaluation_state import PageContext
    from linkedin.runtime_attempts import RuntimeAttemptService

from linkedin.evaluation_support import (
    _LINKEDIN_RESULTS_PAGE_SIZE,
    _PANEL_DOWN_COOLDOWN_SECONDS,
    _PANEL_MOUNT_RELOAD_AT,
    _PANEL_MOUNT_TIMEOUT_RE,
    _PANEL_MOUNT_TRIP_AT,
    _PANEL_RECOVERY_MAX_ATTEMPTS,
    _is_browser_disconnect_error,
    _is_foreign_project_page,
    _is_recruiter_search_page,
    logger,
)

@dataclass
class PageRecoveryBudget:
    """Shared process-local ceiling; successful retries also spend an attempt."""
    enabled: bool = False
    per_page_limit: int = 2
    session_limit: int = 6
    used: int = 0
    pages: dict[tuple[int, str, int], int] = field(default_factory=dict)

    def take(self, key: tuple[int, str, int]) -> bool:
        used_here = self.pages.get(key, 0)
        if not self.enabled or used_here >= self.per_page_limit or self.used >= self.session_limit:
            return False
        self.pages[key] = used_here + 1
        self.used += 1
        return True


class SearchSurface:
    """Physical position is distinct from the committer's published cursor."""

    _last_good_url: str
    _last_pause_time: float
    _panel_mount_failstreak: int
    _panel_mount_reload_used: bool
    _zero_slot_reload_attempted_pages: set[tuple[int, str, int, str]]

    def __init__(
        self, context: EvaluationContext, page: PageContext, allocator: AllocatorStateService,
    ):
        self.context = context
        self.page = page
        self.allocator = allocator
        self.budget = PageRecoveryBudget()
        self.recovery_key: tuple[int, str, int] | None = None
        self.recovery_top_identity = ""
        self.geography: GeographyGateService | None = None
        self.attempts: RuntimeAttemptService
        # Why the next opening must re-anchor before it inspects the surface;
        # empty means the live page is the one the last string left in a state
        # the next one can use.
        self._surface_reanchor_reason: str = ""

    def _honor_irreversible_side_effect_boundary(self) -> None:
        """Recheck all stop authority immediately before a save click."""
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
        self._assert_brief_project_context()


    def _assert_brief_project_context(self) -> None:
        """Refuse a save the live page cannot prove belongs to the brief's project.

        Run-start pins the brief's project, but the bound page can move
        mid-run (operator tab switch, recovery re-bind onto a different Recruiter
        tab). A Recruiter save lands in whatever project the page belongs to, so
        the project is revalidated on the same boundary as profile identity.

        A URL with NO project id is refused too when the brief pins one. It
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
            current_url = self.context.browser.page.url
        except Exception as exc:
            if str(self.context.brief_obj.linkedin_project_id or "").strip():
                raise ProjectContextMismatchError(
                    "project unverified before save: could not read the page URL "
                    f"({type(exc).__name__})"
                ) from exc
            return
        if _is_foreign_project_page(current_url, self.context.brief_obj.linkedin_project_id):
            page_project = _recruiter_page_project_id(current_url)
            expected = str(self.context.brief_obj.linkedin_project_id or "")
            if page_project:
                raise ProjectContextMismatchError(
                    "project mismatch before save: page belongs to Recruiter "
                    f"project {page_project!r}, brief expects {expected!r}"
                )
            raise ProjectContextMismatchError(
                "project unverified before save: page carries no Recruiter "
                f"project id, brief expects {expected!r}"
            )


    async def _recover_stuck_profile_panel(
        self,
        *,
        candidate_name: str,
        page_num: int,
        decision: OpusDecision | None = None,
        progress: Progress | None = None,
        search_string: SearchString | None = None,
        recovery_owner: SearchString | None = None,
        page_context: PageReviewContext | None = None,
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
                        self.context.log_path,
                        "panel_recovery_started",
                        name=candidate_name,
                        page=page_num,
                        recovery_status=status,
                        **details,
                    )
                elif status == "succeeded":
                    log_event(
                        self.context.log_path,
                        "panel_recovered",
                        name=candidate_name,
                        page=page_num,
                        recovery_status=status,
                        **details,
                    )
                else:
                    log_event(
                        self.context.log_path,
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

        owner = recovery_owner or search_string
        record_recovery_event("started", recovery_attempted=True)
        last_error: Exception | None = None
        attempts_used = 0
        for attempt in range(1, _PANEL_RECOVERY_MAX_ATTEMPTS + 1):
            attempts_used = attempt
            try:
                if self.budget.enabled:
                    await self._check_page_recovery_health()
                await self.context.browser.go_back_to_results()
                await asyncio.sleep(
                    human_delay_correlated(
                        config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS,
                        channel="panel_close",
                    )
                )
                if self.budget.enabled and owner is not None:
                    if not await self._recovery_page_is_verified(owner, page_num):
                        last_error = PanelRecoveryError("results page unverified after panel dismissal")
                        break
                if decision is not None:
                    decision._panel_stuck = False
                record_recovery_event("succeeded", attempts=attempt)
                print(
                    f"    [recovered] Profile panel restored after "
                    f"{candidate_name}; continuing queued full reviews"
                )
                return
            except Exception as exc:
                self._raise_page_recovery_danger(exc)
                last_error = exc
                if _is_browser_disconnect_error(exc):
                    break

        if await self._try_page_recovery(
            owner=owner, page_num=page_num, fault="panel", error=last_error,
        ):
            if decision is not None:
                decision._panel_stuck = False
            record_recovery_event("succeeded", attempts=attempts_used, mode="page_local")
            return

        # A fatal browser-state failure is authoritative even if pagination had
        # already recorded an advisory stop reason such as ``early_exit``.
        (page_context or self.page.review).set_break_reason("panel_stuck", force=True)
        if progress is not None:
            self.allocator._checkpoint_progress(
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


    @staticmethod
    def _is_panel_mount_timeout(error: BaseException) -> bool:
        """A profile open whose click landed but whose panel never appeared.

        Regex rather than substring gates: the role-agnosticism ratchet's
        text-gate scan must not read DOM-error classification as
        brief-vocabulary gating (its baseline may never grow).
        """
        return bool(_PANEL_MOUNT_TIMEOUT_RE.search(str(error)))


    def _note_profile_extraction_success(self) -> None:
        """Any successful extraction proves the panel mounts; clear the streak."""
        self._panel_mount_failstreak = 0
        self._panel_mount_reload_used = False


    async def _note_profile_extraction_failure(
        self, snippet: CandidateSnippet, error: Exception
    ) -> None:
        """Contain a profile-panel mount wedge without extending the failure path.

        Capture forensics at the configured streak points, attempt one document
        reload at ``_PANEL_MOUNT_RELOAD_AT``, and trip the governor at
        ``_PANEL_MOUNT_TRIP_AT`` so the existing governed-shutdown path ends the
        session cleanly. A reload is used because the URL and applied facets
        survive it. This failure handler is fail-soft and never raises.
        """
        try:
            browser = getattr(self.context, "browser", None)
            if browser is None:
                return
            if not self._is_panel_mount_timeout(error):
                # Non-mount extraction failure (read/expand stage) — rare;
                # keep a picture of the page, no streak semantics.
                await browser.capture_failure_forensics(
                    f"profile-error-{snippet.name}"
                )
                return
            self._panel_mount_failstreak += 1
            streak = self._panel_mount_failstreak
            if streak in (1, _PANEL_MOUNT_RELOAD_AT) or streak >= _PANEL_MOUNT_TRIP_AT:
                await browser.capture_failure_forensics(f"panel-mount-{streak}")
            if streak >= _PANEL_MOUNT_TRIP_AT:
                print(
                    f"    [panel-down] profile panel unavailable ({streak} "
                    "consecutive mount failures, reload ineffective) — ending session"
                )
                try:
                    log_event(
                        self.context.log_path,
                        "profile_panel_unavailable",
                        name=snippet.name,
                        failstreak=streak,
                        reload_used=self._panel_mount_reload_used,
                    )
                except Exception:
                    logger.warning("profile_panel_unavailable event write failed")
                self.context._governor.force_backoff(
                    f"profile_panel_unavailable ({streak} consecutive mount failures)",
                    cooldown_seconds=_PANEL_DOWN_COOLDOWN_SECONDS,
                )
                return
            if streak >= _PANEL_MOUNT_RELOAD_AT and not self._panel_mount_reload_used:
                self._panel_mount_reload_used = True
                print(
                    f"    [panel-recovery] {streak} consecutive panel failures — "
                    "reloading search page"
                )
                try:
                    log_event(
                        self.context.log_path,
                        "panel_recovery_started",
                        name=snippet.name,
                        recovery_status="started",
                        mode="mount_wedge",
                        failstreak=streak,
                    )
                except Exception:
                    logger.warning("panel_recovery_started event write failed")
                if await browser.reload_search_page():
                    print("    [panel-recovery] reload complete — resuming")
                    try:
                        log_event(
                            self.context.log_path,
                            "panel_recovered",
                            name=snippet.name,
                            recovery_status="succeeded",
                            mode="mount_wedge",
                        )
                    except Exception:
                        logger.warning("panel_recovered event write failed")
                else:
                    print(
                        "    [panel-recovery] reload failed — stopping at the "
                        "failure ceiling"
                    )
                    try:
                        log_event(
                            self.context.log_path,
                            "panel_stuck",
                            name=snippet.name,
                            recovery_status="reload_failed",
                            mode="mount_wedge",
                        )
                    except Exception:
                        logger.warning("panel_stuck event write failed")
        except Exception as note_error:
            # Containment must fail open — the extraction failure it is
            # annotating already has an owner in the caller's handler.
            logger.warning(
                "Panel wedge containment failed open (%s)",
                type(note_error).__name__,
            )


    async def _probe_card_slot_count_after_hydration(self) -> int:
        slot_count = await self.context.browser.get_card_slot_count()
        if slot_count == 0:
            if self.budget.enabled:
                await self._check_page_recovery_health()
            fallback_count = await self.context.browser.scroll_to_load_all_results()
            slot_count = fallback_count or await self.context.browser.get_card_count()
        return int(slot_count or 0)


    def _owned_values_are_accounted_for(
        self,
        owned: set[str],
        requested: set[str],
        dimension: str,
    ) -> bool:
        """Whether every owned facet value belongs to ``requested``.

        A value belongs when it IS a requested value, or when it is the name
        LinkedIn applied for one — per the last reconcile receipt, or per the
        resolution sidecar that seeded it. Compared on the typeahead's own
        fold, so an "Itaú" pill accounts for a requested "Itau".

        A request the receipt lists under ``unapplied`` is accounted for by
        nothing: the rail's word is that no pill was proved for it, so its
        cached spelling must not bless a pill either. A wrong typeahead take
        writes that spelling to the sidecar, and the entry outlives the string
        that made it.
        """
        receipt = getattr(
            self.context.browser, "last_facet_reconciliation", None
        )
        receipt = receipt if isinstance(receipt, dict) else {}
        accounted = set(requested)
        accounted |= facet_target_landed(requested, receipt, dimension)
        refused = {
            facet_name_key(value)
            for value in (receipt.get("unapplied") or {}).get(dimension) or []
        }
        cache_get = getattr(
            self.context.browser, "_facet_resolution_cache_get", None
        )
        if callable(cache_get):
            for value in requested:
                if facet_name_key(value) in refused:
                    continue
                try:
                    cached = cache_get(dimension, value)
                except Exception:
                    cached = None
                if cached:
                    accounted.add(cached)
        allowed = {facet_name_key(value) for value in accounted}
        return {facet_name_key(value) for value in owned} <= allowed


    async def _verify_owned_facets_after_reload(
        self,
        *,
        search_string: SearchString,
        boundary: str,
    ) -> None:
        """F11: read-only owned-facet verification after a raw page reload.

        A document reload restores LinkedIn's saved-search state, which is
        assumed — not proven — to equal the active variant's reconciled sets.
        Verify before any post-reload candidate work; a mismatch records the
        canonical receipt and raises (typed, resumable). Never mutates: a
        mid-string removal would re-run the search and corrupt the page
        cursor, so restore belongs to the resume path.

        The target is the set the string's own reconcile PROVED on the rail,
        which is the variant's request minus any value the typeahead refused
        and under the spelling LinkedIn actually applied. The browser's owned
        target is preferred whenever every value in it is accounted for by
        this string's request — itself, its landed spelling on the reconcile
        receipt, or the resolution cache. A value nothing accounts for is
        another string's evidence, so the request stands and the mismatch
        raises.

        Accounting for the landed spelling is the whole point: a requested
        "Mercado Livre" that lands as "Mercado Livre Brasil" is not a subset
        of the request, and `verify_only` never substitutes, so a raw subset
        guard fell back to the request and refused a correct page — after
        which the strike handler navigated off a live result surface to
        repair nothing. `reload_search_page` verifies the owned target
        directly and never had this failure.
        """
        state = self.allocator.experiment(search_string.id)
        if state is None:
            return
        active = state.active_variant
        companies = set(active.structured_filters.companies or [])
        titles = set(active.structured_filters.titles or [])
        owned = getattr(self.context.browser, "_owned_facet_target", None)
        if isinstance(owned, dict):
            owned_companies = set(owned.get("companies") or [])
            owned_titles = set(owned.get("titles") or [])
            if self._owned_values_are_accounted_for(
                owned_companies, companies, "companies"
            ) and self._owned_values_are_accounted_for(
                owned_titles, titles, "titles"
            ):
                companies, titles = owned_companies, owned_titles
        try:
            await self.context.browser.reconcile_facets(
                companies=companies,
                titles=titles,
                verify_only=True,
            )
        except FacetReconciliationError as exc:
            receipt = self._facet_failure_receipt(
                companies=companies,
                titles=titles,
                gate="post_reload_mismatch",
                error=exc,
            )
            self._record_facet_reconciliation(
                search_string=search_string,
                boundary=boundary,
                receipt=receipt,
            )
            raise


    async def _card_slot_count_attempt(
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
                if self.budget.enabled:
                    await self._check_page_recovery_health()
                await self.context.browser.page.reload(
                    wait_until="domcontentloaded",
                    timeout=30000,
                )
                await self.context.browser.page.wait_for_timeout(4000)
                if self.budget.enabled:
                    await self._check_page_recovery_health()
            except Exception as exc:
                self._raise_page_recovery_danger(exc)
                if _is_browser_disconnect_error(exc):
                    raise
                raise PageRenderFailedError(
                    f"page render failed during zero-slot reload on page {page_num}: {exc!r}"
                ) from exc
            slot_count = await self._probe_card_slot_count_after_hydration()
            if slot_count:
                if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
                    # F11: never accept post-reload cards on an unverified
                    # sidebar — the reload restored saved-search state, not a
                    # proven copy of the active variant's sets.
                    await self._verify_owned_facets_after_reload(
                        search_string=search_string,
                        boundary="zero_slot_reload",
                    )
                return slot_count

        self.page._latest_page_preview_snippets = []
        self.page.review = PageReviewContext()
        try:
            log_event(
                self.context.log_path,
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


    async def _attempt_page_advance(
        self,
        *,
        result_count: int,
        page_num: int,
    ) -> tuple[bool, bool]:
        """Advance to the next results page, escalating before declaring a transient failure.

        Retry after 3 seconds, scroll the results list and retry, capture
        forensics, then perform one document reload before giving up. The
        post-reload retry is position-verified only when the top result card
        proves the page did not move; otherwise the checkpointed transient
        raise lets resume re-seek the page.
        """
        if self.budget.enabled:
            await self._check_page_recovery_health()
        has_next = await self.context.browser.go_to_next_page()
        if self.budget.enabled:
            await self._check_page_recovery_health()
        if has_next:
            return True, False
        if not self._pages_remaining_by_result_count(result_count, page_num):
            return False, False
        await asyncio.sleep(3)
        if self.budget.enabled:
            await self._check_page_recovery_health()
        has_next = await self.context.browser.go_to_next_page()
        if self.budget.enabled:
            await self._check_page_recovery_health()
        if has_next:
            return True, False
        try:
            if self.budget.enabled:
                await self._check_page_recovery_health()
            await self.context.browser.scroll_to_load_all_results()
        except Exception as exc:
            self._raise_page_recovery_danger(exc)
            pass
        await asyncio.sleep(5)
        if self.budget.enabled:
            await self._check_page_recovery_health()
        has_next = await self.context.browser.go_to_next_page()
        if self.budget.enabled:
            await self._check_page_recovery_health()
        if has_next:
            return True, False
        try:
            await self.context.browser.capture_failure_forensics(
                f"pagination-p{page_num}"
            )
        except Exception as exc:
            self._raise_page_recovery_danger(exc)
            pass
        try:
            signature_before = await self.context.browser._peek_top_card_signature()
            if self.budget.enabled:
                await self._check_page_recovery_health()
            if await self.context.browser.reload_search_page():
                if self.budget.enabled:
                    await self._check_page_recovery_health()
                signature_after = await self.context.browser._peek_top_card_signature()
                if signature_before != ("", "") and (
                    signature_after == signature_before
                ):
                    print(
                        "    [pagination] reload kept the page — retrying next"
                    )
                    if self.budget.enabled:
                        await self._check_page_recovery_health()
                    has_next = await self.context.browser.go_to_next_page()
                    if self.budget.enabled:
                        await self._check_page_recovery_health()
        except Exception as exc:
            self._raise_page_recovery_danger(exc)
            pass
        return has_next, not has_next


    def _raise_page_recovery_danger(self, error: BaseException) -> None:
        """Never erase stop/ownership authority inside a recoverable wrapper."""
        if not self.budget.enabled:
            return
        seen: set[int] = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            if (
                not isinstance(error, Exception)
                or isinstance(error, (
                    OperatorStopRequested, SessionExpired, GovernorLimitReached,
                    GeographyRegimeError, SurfaceOwnershipLostError,
                    ProjectContextMismatchError, FacetReconciliationError,
                    AllocatorPolicyError, ApiBudgetExhaustedError,
                ))
                or _acquisition_is_browser_disconnect_error(error)
                or is_api_budget_exhausted_error(error)
            ):
                raise error
            error = error.__cause__ or error.__context__


    async def _check_page_recovery_health(self) -> None:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError
        # This classifier reads only; _ensure_browser_healthy can navigate
        # before detecting a challenge and is not a recovery admission check.
        from linkedin.recruiter_recovery import detect_recruiter_health

        health = await detect_recruiter_health(self.context.browser)
        if health == "blocked_or_rate_limited":
            self.context._governor.force_backoff("blocked_or_rate_limited")
            raise GovernorLimitReached("blocked_or_rate_limited")
        self._honor_irreversible_side_effect_boundary()
        if health != "healthy":
            raise SurfaceOwnershipLostError(
                f"page-local recovery refused unhealthy surface: {health}"
            )


    async def _verify_recovery_geography(self) -> None:
        """Read-only comparison; the ordinary geography verifier can re-search."""
        try:
            chips = await self.context.browser.read_applied_location_chips()
        except Exception as error:
            self._raise_page_recovery_danger(error)
            raise GeographyRegimeError("geography unreadable during page-local recovery") from error
        desired = {normalize_facet_value_for_compare(v) for v in self._session_geography_values()}
        observed = [normalize_facet_value_for_compare(v) for v in chips] if chips is not None else None
        if (observed is None or not all(observed) or len(observed) != len(set(observed))
                or set(observed) != desired):
            raise GeographyRegimeError("geography differs during page-local recovery")


    async def _check_page_recovery_authority(self, owner: SearchString) -> None:
        await self._check_page_recovery_health()
        await self._verify_recovery_geography()
        await self._verify_owned_facets_after_reload(
            search_string=owner, boundary="page_local_recovery",
        )
        self._honor_irreversible_side_effect_boundary()


    async def _recovery_top_identity(self) -> str:
        signature = await self.context.browser._peek_top_card_signature()
        if not signature or len(signature) != 2 or not all(signature):
            return ""
        return LinkedInBrowser._profile_url_fragment(signature[1])


    def _page_recovery_key(self, owner: SearchString, page_num: int) -> tuple[int, str, int]:
        return owner.id, self._variant_id_for_search_string(owner), page_num


    async def _remember_recovery_page(self, owner: SearchString, page_num: int) -> None:
        self.recovery_key = self._page_recovery_key(owner, page_num)
        self.recovery_top_identity = await self._recovery_top_identity()


    async def _recovery_page_is_verified(self, owner: SearchString, page_num: int) -> bool:
        await self._check_page_recovery_authority(owner)
        expected = self.recovery_top_identity
        return bool(
            self.recovery_key == self._page_recovery_key(owner, page_num)
            and expected
            and _is_recruiter_search_page(self.context.browser.page.url)
            and await self._recovery_top_identity() == expected
        )


    async def _advance_recovery_page(self, owner: SearchString) -> bool:
        """Nonrecursive seek: a native bool alone never proves advancement."""
        await self._check_page_recovery_authority(owner)
        before = await self._recovery_top_identity()
        if not before:
            return False
        advanced = await self.context.browser.go_to_next_page()
        await self._check_page_recovery_authority(owner)
        after = await self._recovery_top_identity()
        return bool(advanced and after and after != before)


    async def _reanchor_recovery_page(
        self, owner: SearchString, page_num: int, variant_id: str,
    ) -> bool:
        state = self.allocator.require_experiment(owner.id)
        try:
            await self.context.browser.navigate_to_search(self._get_project_url())
        except AuthWallError:
            # A sign-in wall is a RuntimeError, but it is not a navigation
            # that can be retried: only the operator can clear it.
            raise
        except (RuntimeError, BrowserError) as failure:
            # Navigation reports sidebar hydration and browser transport faults.
            # This catch surrounds only that browser operation, never writers.
            self._raise_page_recovery_danger(failure)
            await self._check_page_recovery_health()
            return False
        await self._check_page_recovery_health()
        self.geography.reset_location_applied()
        await self.geography._apply_session_location_filter(allow_resolution=False)
        await self._check_page_recovery_health()
        await self._apply_opening_search(owner, state, state.current_boolean(), for_recovery=True)
        await self._check_page_recovery_authority(owner)
        if not _is_recruiter_search_page(self.context.browser.page.url):
            return False
        # An allowlisted campaign never restores a page keyword-only: a
        # competitor facet that did not land fails the re-anchor.
        unlanded = competitor_facet_skip_reason(
            list(state.active_variant.structured_filters.companies or []),
            getattr(self, "last_opening_facet_receipt", None),
            employer_allowlist_from_brief(self.context.brief_obj),
            requested_titles=list(
                getattr(state.active_variant.structured_filters, "titles", None) or []
            ),
        )
        if unlanded:
            print(f"    [scope] String #{owner.id}: page recovery refused; {unlanded}")
            return False
        if not await self._probe_card_slot_count_after_hydration():
            return False
        # Page one is proved by the URL: pagination writes start=<offset> and
        # a fresh search carries none. A disabled Previous button proves
        # nothing, because the "Spotlights" carousel above the results always
        # shows one. An unreadable offset is not page one either.
        if await self.context.browser.results_page_offset() != 0:
            return False
        if not await self._recovery_top_identity():
            return False
        for _ in range(1, page_num):
            if not await self._advance_recovery_page(owner):
                return False
        if self._variant_id_for_search_string(owner) != variant_id:
            raise SurfaceOwnershipLostError("active variant changed during page-local recovery")
        await self._check_page_recovery_authority(owner)
        await self._remember_recovery_page(owner, page_num)
        return bool(self.recovery_top_identity)


    async def _try_page_recovery(
        self, *, owner: SearchString | None, page_num: int,
        fault: str, error: BaseException | None = None,
    ) -> bool:
        budget = self.budget
        if not budget.enabled or owner is None:
            return False
        if error is not None:
            self._raise_page_recovery_danger(error)
        await self._check_page_recovery_authority(owner)
        state = self.allocator.experiment(owner.id)
        if state is None:
            return False
        filters = state.active_variant.structured_filters
        # Only these dimensions have an exact live verifier in this checkout.
        # An unsupported control cannot silently become a broader search.
        if (filters.skills or filters.assessments or filters.advanced_filters
                or set(filters.sidebar_filters) - {"locations"}):
            return False
        enforce_per_string_location_ownership(filters, self._session_geography_values())
        key = self._page_recovery_key(owner, page_num)

        def record(status: str) -> None:
            used = budget.pages.get(key, 0)
            line = (
                f"    [page-recovery] {fault} string #{owner.id} page {page_num}: "
                f"{status}; page {used}/{budget.per_page_limit}, "
                f"session {budget.used}/{budget.session_limit}"
            )
            print(line)
            self.attempts._record_runtime_event(
                search_string=owner, event_type="page_local_recovery",
                payload={"fault": fault, "status": status, "page": page_num,
                         "variant_id": key[1], "page_attempts": used,
                         "session_attempts": budget.used,
                         "page_limit": budget.per_page_limit,
                         "session_limit": budget.session_limit},
            )

        while budget.take(key):
            record("reanchoring")
            try:
                restored = await self._reanchor_recovery_page(owner, page_num, key[1])
            except (PageRenderFailedError, PanelRecoveryError, TransientPaginationError,
                    TimeoutError, BrowserTimeoutError) as failure:
                self._raise_page_recovery_danger(failure)
                restored = False
            if restored:
                record("reanchored")
                return True
            # A failed navigation may have exposed a challenge or lost owner.
            await self._check_page_recovery_authority(owner)
        record("limit reached")
        return False


    async def _card_slot_count_or_raise(
        self, *, search_string: SearchString, page_num: int,
        result_count: int, mode: str,
    ) -> int:
        while True:
            try:
                count = await self._card_slot_count_attempt(
                    search_string=search_string, page_num=page_num,
                    result_count=result_count, mode=mode,
                )
            except PageRenderFailedError as error:
                if await self._try_page_recovery(
                    owner=search_string, page_num=page_num, fault="render", error=error,
                ):
                    continue
                raise
            if self.budget.enabled:
                await self._remember_recovery_page(search_string, page_num)
            return count


    async def _go_to_next_page_with_transient_retry(
        self, *, result_count: int, page_num: int,
        search_string: SearchString | None = None,
    ) -> tuple[bool, bool]:
        advanced, transient = await self._attempt_page_advance(
            result_count=result_count, page_num=page_num,
        )
        while transient and await self._try_page_recovery(
            owner=search_string, page_num=page_num, fault="pagination",
        ):
            if await self._advance_recovery_page(search_string):
                return True, False
        # Each caller retains its original checkpoint, message, and exception.
        return advanced, transient


    def _record_facet_reconciliation(
        self,
        *,
        search_string: SearchString,
        boundary: str,
        receipt: dict[str, Any],
    ) -> None:
        payload = {
            **receipt,
            "boundary": boundary,
            "string_id": search_string.id,
        }
        self.attempts._record_runtime_event(
            search_string=search_string,
            event_type="linkedin_facet_reconciliation",
            payload=payload,
        )
        log_event(self.context.log_path, "linkedin_facet_reconciliation", **payload)
        print(format_facet_reconciliation_console_line(search_string.id, payload))
        unapplied_line = format_facet_unapplied_console_line(
            search_string.id, payload
        )
        if unapplied_line:
            print(unapplied_line)
        substituted_line = format_facet_substituted_console_line(
            search_string.id, payload
        )
        if substituted_line:
            print(substituted_line)


    def _facet_failure_receipt(
        self,
        *,
        companies: set[str],
        titles: set[str],
        gate: str,
        error: BaseException,
        before: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        observed = getattr(error, "receipt", None)
        receipt = dict(observed) if isinstance(observed, dict) else {}
        if before:
            initial = dict(before)
            if isinstance(observed, dict):
                initial.update(
                    {
                        key: value
                        for key, value in observed.items()
                        if key not in {"before", "desired"}
                    }
                )
            receipt = initial
        receipt.setdefault("before", {"companies": None, "titles": None})
        receipt.setdefault(
            "desired",
            {
                "companies": sorted(companies, key=str.casefold),
                "titles": sorted(titles, key=str.casefold),
            },
        )
        receipt.setdefault("after", {"companies": None, "titles": None})
        receipt["failure_gate"] = str(getattr(error, "gate", "") or gate)
        return receipt


    async def _capture_recovery_snapshot(
        self,
        search_string: SearchString,
        *,
        page_num: int | None = None,
    ):
        work_unit_id = None
        if self.context._runtime_run_id:
            work_unit_id = self.context._runtime_state.get_work_unit_id(
                self.context._runtime_run_id,
                kind=LINKEDIN_STRING_KIND,
                source_unit_id=str(search_string.id),
            )
        advanced_controls: dict[str, Any] = {}
        try:
            advanced_controls = await self.context.browser.snapshot_advanced_search_controls()
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
        experiment_state = self.allocator.experiment(search_string.id)
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
                    # A structured-only active variant omits the keyword from the
                    # recovery snapshot so replay does not re-add an unintended search.
                    include_keyword=(active.surface != "structured_only"),
                )
                advanced_controls = snapshot_controls_from_plan(plan)
                # plan.keyword_boolean is '' for a structured_only surface and the
                # real boolean otherwise — keep the snapshot's replayed keyword in
                # lockstep with the gated plan instead of search_string.boolean.
                snapshot_keyword_boolean = plan.keyword_boolean
        return capture_recovery_snapshot(
            self.context.browser,
            run_id=self.context._runtime_run_id,
            work_unit_id=work_unit_id,
            lane_id=getattr(search_string, "lane_id", "") or "",
            keyword_boolean=snapshot_keyword_boolean,
            current_page=page_num or 0,
            search_url=self._last_good_url or self.context.browser.get_current_search_url() or "",
            advanced_search_controls=advanced_controls,
            string_id=(
                search_string.id
                if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED
                else 0
            ),
        )


    def _variant_id_for_search_string(self, search_string: SearchString) -> str:
        state = self.allocator.experiment(search_string.id)
        if state is None:
            return ""
        active = state.active_variant
        if active.variant_id and active.variant_id != "root":
            return active.variant_id
        return state.committed_variant_id or state.active_variant_id or ""


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
                await self.context.browser.disconnect()
            except Exception:
                pass
            try:
                await self.context.browser.connect()
                if recovery_url:
                    try:
                        await self.context.browser.navigate_to_search(recovery_url)
                    except AuthWallError:
                        raise
                    except Exception as nav_error:
                        print(f"  [reconnect] Reconnected but could not restore search page: {nav_error}")
                print(f"  [reconnect] Success — reconnected to LinkedIn Recruiter.")
                return True
            except AuthWallError:
                # Another attempt would only navigate into the same wall.
                raise
            except Exception as e:
                print(f"  [reconnect] Failed: {e}")
        return False


    @property
    def _session_location_applied(self) -> bool:
        return self.geography._session_location_applied


    @_session_location_applied.setter
    def _session_location_applied(self, value: bool) -> None:
        self.geography._session_location_applied = value


    @property
    def _session_geography_receipt(self) -> dict:
        return self.geography._session_geography_receipt


    @_session_geography_receipt.setter
    def _session_geography_receipt(self, value: dict) -> None:
        self.geography._session_geography_receipt = value


    def _session_geography_values(self) -> list[str]:
        return self.geography._session_geography_values()


    async def _apply_session_location_filter(self) -> None:
        return await self.geography._apply_session_location_filter()


    async def _resolve_and_reapply_geography(
        self, geo_values: list[str]
    ) -> list[str] | None:
        return await self.geography._resolve_and_reapply_geography(geo_values)


    async def _verify_session_geography_chips(self) -> None:
        return await self.geography._verify_session_geography_chips()


    async def _reassert_session_location_after_recovery(self) -> None:
        return await self.geography._reassert_session_location_after_recovery()


    def mark_surface_for_reanchor(self, reason: str) -> None:
        """Record that no following string should inherit the live page.

        A zero-result string returns without touching the browser, so the run
        stays on LinkedIn's no-results page — rail and Keywords box present,
        result summary and cards absent. Nothing downstream re-navigates, so
        the next string's opening inspected the dead page instead of its own
        (Addendum I2). The next opening reads this flag and re-anchors first,
        rather than waiting for a gate to refuse three strings in a row.
        """
        self._surface_reanchor_reason = str(reason or "unknown")


    def surface_reanchor_pending(self) -> str:
        """Why a re-anchor is owed, or ``""`` when none is."""
        return getattr(self, "_surface_reanchor_reason", "")


    async def reanchor_search_surface(self, reason: str) -> bool:
        """Put the live page back on this project's search view.

        The same sequence the resume path already uses for a wrong page
        (`string_driver._process_string`): navigate to the project search URL,
        or reload when the brief pins no project, then re-assert the session
        geography the navigation dropped.

        Fail-soft on purpose, and it bypasses no fail-closed gate. Every call
        site is immediately followed by the readiness gate and
        `_verify_session_geography_chips`, which re-read the live rail and
        refuse on their own evidence — a re-anchor that could not navigate
        leaves them to decide, exactly as if it had never run. Swallowing here
        is what keeps a repair attempt from becoming a second failure mode.

        A LinkedIn sign-in wall (``AuthWallError``) is the one exception: it
        is not a repair that failed but a stop only the operator can clear.
        Swallowed, the next gate refuses generically, the loop strikes the
        string, and the run re-anchors into the same wall for the next one.
        """
        # Cleared first: a raising re-anchor must not leave the flag set and
        # ask every later string to try the same broken navigation.
        self._surface_reanchor_reason = ""
        try:
            project_url = self._get_project_url()
            if project_url:
                await self.context.browser.navigate_to_search(project_url)
            else:
                await self.context.browser.page.reload(
                    wait_until="domcontentloaded", timeout=30000
                )
                await self.context.browser.page.wait_for_timeout(4000)
            self.geography.reset_location_applied()
            await self._apply_session_location_filter()
        except AuthWallError:
            raise
        except Exception as exc:
            try:
                log_event(
                    self.context.log_path,
                    "search_surface_reanchor_failed",
                    reason=str(reason or ""),
                    error=str(exc),
                )
            except Exception:
                pass
            return False
        try:
            log_event(
                self.context.log_path,
                "search_surface_reanchored",
                reason=str(reason or ""),
            )
        except Exception:
            pass
        return True


    def _log_removed_qualifications(self, *, search_string_id: int) -> None:
        """One run-log event per qualification the readiness gate removed."""
        removed = getattr(self.context.browser, "last_removed_qualifications", None)
        if not isinstance(removed, list):
            return
        for text in removed:
            log_event(
                self.context.log_path,
                "search_qualification_removed",
                string_id=search_string_id,
                qualification=str(text),
            )


    async def _apply_opening_search(
        self,
        search_string: SearchString,
        experiment_state: LinkedInExperimentState,
        current_boolean: str,
        *, for_recovery: bool = False,
    ) -> None:
        """Apply a string's opening search, keyword-led or filter-led.

        A hybrid lane with non-empty active structured filters applies the
        compiled structured plan; every other lane enters the bare keyword.
        Opening structure is not a rewrite: the browser applies keywords and
        controls in one call, writes the applied-only snapshot, and does not
        consume the mutation budget or emit a mutation event.

        The plan carries only the lane's structured filters; session geography
        is applied separately so the two location paths do not double-apply.
        Before either path, verify the session geography chips on the live
        sidebar so a navigation race cannot send an off-geo string.
        """
        opening_receipt: dict[str, Any] | None = None
        companies: set[str] = set()
        titles: set[str] = set()
        redundant_locations: list[str] = []
        # The verified receipt of this opening, read by the competitor-scope
        # skip (linkedin/competitor_scope.py); None until a reconcile proves it.
        self.last_opening_facet_receipt = None
        if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
            active_target = experiment_state.active_variant
            companies = set(active_target.structured_filters.companies or [])
            titles = set(active_target.structured_filters.titles or [])
            gate = "per_string_location_conflict"
            try:
                # F8: reject-or-omit a lane-compiled per-string location
                # BEFORE any browser interaction. Session-identical values are
                # returned here (and recorded on the receipt) so the compile
                # filter below drops only proven-redundant controls; anything
                # else raises with this gate.
                redundant_locations = enforce_per_string_location_ownership(
                    active_target.structured_filters,
                    self._session_geography_values(),
                )
                # A surface the last string left unusable is re-anchored
                # before this string's gates read it — after F8, which is a
                # pure compile-time block and touches no browser. The
                # re-anchor serves the readiness step, so a sign-in wall it
                # raises is filed under that step's gate.
                gate = "search_surface_unready"
                reanchor_reason = self.surface_reanchor_pending()
                if reanchor_reason and await self.reanchor_search_surface(
                    reanchor_reason
                ):
                    print(
                        "  [surface] re-anchored to project search "
                        f"({reanchor_reason})"
                    )
                await self.context.browser.assert_search_surface_ready()
                self._log_removed_qualifications(search_string_id=search_string.id)
                gate = "session_location"
                if for_recovery:
                    await self._verify_recovery_geography()
                else:
                    await self._verify_session_geography_chips()
                gate = "facet_reconciliation"
                await self.context.browser.reconcile_facets(
                    companies=companies,
                    titles=titles,
                )
                opening_receipt = dict(self.context.browser.last_facet_reconciliation)
                if redundant_locations:
                    opening_receipt["redundant_locations_omitted"] = (
                        redundant_locations
                    )
            except Exception as exc:
                receipt = self._facet_failure_receipt(
                    companies=companies,
                    titles=titles,
                    gate=gate,
                    error=exc,
                )
                self._record_facet_reconciliation(
                    search_string=search_string,
                    boundary="opening_search",
                    receipt=receipt,
                )
                raise
        else:
            if for_recovery:
                await self._verify_recovery_geography()
            else:
                await self._verify_session_geography_chips()
        active = experiment_state.active_variant
        executed_boolean = current_boolean
        execution_surface = "keyword"
        if (
            search_string.acquisition_mode == "linkedin_hybrid"
            and not active.structured_filters.is_empty()
        ):
            from linkedin.advanced_search import compile_structured_filters_to_plan

            plan = compile_structured_filters_to_plan(
                active.structured_filters,
                keyword_boolean=current_boolean,
                acquisition_mode=search_string.acquisition_mode,
                # A structured-only active variant omits the keyword from the opening
                # apply so the filters carry the search instead.
                include_keyword=(active.surface != "structured_only"),
            )
            if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
                # Only session-identical redundant values can reach this
                # filter — the F8 ownership gate at the top of this method
                # rejected any conflicting per-string location already.
                plan.controls = [
                    control
                    for control in plan.controls
                    if control.dimension != "locations"
                ]
                # The reconcile above already typed every company and title,
                # one value per call. The controls stay on the plan for the
                # recovery snapshot; their outcome comes from its receipt.
                plan.reconciled_facet_misses = facet_misses(opening_receipt or {})
            result = await self.context.browser.apply_advanced_search_plan(plan)
            executed_boolean = plan.keyword_boolean
            execution_surface = "advanced"
            if not config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
                try:
                    log_event(
                        self.context.log_path,
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
                # The receipt no longer dies in run_log.jsonl — persist
                # the last apply receipt on the SearchString so block reports,
                # adaptation, and the run report see actuator health.
                search_string.surface_receipt = fields
                print(_surface_receipt.format_apply_receipt(fields))
                if getattr(self.context, "log_path", None):
                    log_event(self.context.log_path, "surface_applied", **fields)
            except Exception as e:
                print(f"  [warn] provenance event failed: {e}")
        else:
            await self.context.browser.enter_search_string(current_boolean)
            if not config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
                try:
                    log_event(
                        self.context.log_path,
                        "string_executed",
                        string_id=search_string.id,
                        executed_boolean=current_boolean,
                        execution_surface="keyword",
                    )
                except Exception as e:
                    print(f"  [warn] provenance event failed: {e}")

        if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
            # Verify what LANDED. A value the typeahead does not offer
            # verbatim is recorded on the opening receipt, not applied, so
            # re-asserting the full request here would fail every string that
            # carries one.
            landed_companies = facet_target_landed(
                companies, opening_receipt or {}, "companies"
            )
            landed_titles = facet_target_landed(
                titles, opening_receipt or {}, "titles"
            )
            try:
                await self.context.browser.reconcile_facets(
                    companies=landed_companies,
                    titles=landed_titles,
                    verify_only=True,
                )
            except FacetReconciliationError as exc:
                receipt = self._facet_failure_receipt(
                    companies=companies,
                    titles=titles,
                    gate="postcondition_mismatch",
                    error=exc,
                    before=opening_receipt,
                )
                self._record_facet_reconciliation(
                    search_string=search_string,
                    boundary="opening_search",
                    receipt=receipt,
                )
                raise
            receipt = dict(opening_receipt or {})
            receipt["after"] = dict(
                self.context.browser.last_facet_reconciliation.get("after", {})
            )
            receipt["failure_gate"] = ""
            self.last_opening_facet_receipt = receipt
            self._record_facet_reconciliation(
                search_string=search_string,
                boundary="opening_search",
                receipt=receipt,
            )
            log_event(
                self.context.log_path,
                "string_executed",
                string_id=search_string.id,
                executed_boolean=executed_boolean,
                execution_surface=execution_surface,
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
            current_url, self.context.brief_obj.linkedin_project_id
        ):
            return
        self._last_good_url = current_url


    async def _ensure_browser_healthy(
        self, *, owner_restore_follows: bool = False
    ) -> None:
        """Check for error states, attempt recovery, and enforce cadence pauses.

        ``owner_restore_follows`` (F11): True only at the string-start call
        site, where _apply_opening_search immediately re-establishes the full
        owner surface (geography + facet reconcile + keyword + cursor). Every
        other caller runs mid-string, so under
        LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED a recovery that touched the
        surface raises SurfaceOwnershipLostError after the legacy healing
        steps — a named, resumable stop — instead of letting candidate work
        continue on an unverified pool and position.
        """
        # --- Cadence pause: human-like idle break ---
        await self._maybe_cadence_pause()

        # --- Snapshot current URL as known-good before checking for errors ---
        try:
            self._record_last_good_url(self.context.browser.page.url)
        except Exception:
            pass

        # --- Error recovery ---
        recovered = await self.context.browser.check_and_recover()
        if recovered:
            # Prefer the last known-good URL: it carries the live search position
            # and, by _record_last_good_url's invariant, is already this brief's
            # project. Fall back to the bare project search page. Re-checked here
            # anyway — this is the navigation that decides which pipeline the run
            # resumes against, and a read-site check costs nothing.
            recovery_url = self._last_good_url or self._get_project_url()
            if _is_foreign_project_page(
                recovery_url, self.context.brief_obj.linkedin_project_id
            ):
                recovery_url = self._get_project_url()
            if recovery_url:
                print(f"  [recovery] Re-navigating to: {recovery_url[:60]}...")
                await self.context.browser.navigate_to_search(recovery_url)
                # The recovery re-navigation dropped the sidebar; re-assert location.
                self.geography.reset_location_applied()
                await self._apply_session_location_filter()
            else:
                print("  [recovery] No recovery URL available — continuing from current page.")
            if (
                config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED
                and not owner_restore_follows
            ):
                print(
                    "  [recovery] Surface ownership lost mid-string — "
                    "stopping for resume restore (facet lifecycle)."
                )
                try:
                    log_event(
                        self.context.log_path,
                        "surface_ownership_lost",
                        recovered=True,
                        recovery_url=str(recovery_url or ""),
                    )
                except Exception:
                    pass
                raise SurfaceOwnershipLostError(
                    "browser recovery re-navigated the search surface "
                    "mid-string; owner state (facets, keyword, page cursor) "
                    "is unverified — stopping for resume restore"
                )

        if config.LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED:
            try:
                from linkedin.recruiter_recovery import detect_recruiter_health
                health = await detect_recruiter_health(self.context.browser)
            except Exception:
                health = None
            if health == "blocked_or_rate_limited":
                print("  [health] Live health classifier: blocked_or_rate_limited — tripping governor backoff.")
                self.context._governor.force_backoff("blocked_or_rate_limited")


    def _get_project_url(self) -> str:
        """Construct LinkedIn Recruiter project URL from brief or auto-detected browser URL."""
        pid = self.context.brief_obj.linkedin_project_id
        if not pid and hasattr(self.context.browser, '_project_id'):
            pid = self.context.browser._project_id
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
        if hasattr(self.context, '_pause_requested'):
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
                log_event(self.context.log_path, "cadence_pause", elapsed_minutes=round(elapsed, 1),
                          pause_seconds=round(pause_secs, 1))
                await asyncio.sleep(pause_secs)
                self._last_pause_time = time.time()
                print(f"  [cadence] Resuming.")
            finally:
                emit_timing_event(
                    getattr(self.context, "_timing_recorder", None),
                    "cadence_pause_timing",
                    elapsed_ms=round(
                        (time.monotonic() - pause_started) * 1000.0,
                        3,
                    ),
                    pause_seconds=round(pause_secs, 3),
                )


    async def _reopen_profile_for_full_eval_save(self, snippet: CandidateSnippet) -> None:
        expected_identity = LinkedInBrowser._profile_url_fragment(
            snippet.profile_url
        )
        if (
            expected_identity
            and self.context.browser.current_profile_identity_fragment()
            == expected_identity
        ):
            return

        print("    Re-opening profile for save...")
        if snippet.card_index >= 0:
            await self.context.browser.ensure_card_rendered(snippet.card_index)

        if snippet.profile_url:
            await self.context.browser.open_profile_by_url(snippet.profile_url)
        else:
            await self.context.browser.open_profile(snippet.name)
        if (
            not expected_identity
            or self.context.browser.current_profile_identity_fragment()
            != expected_identity
        ):
            raise PanelRecoveryError(
                "profile identity mismatch after save reopen"
            )
