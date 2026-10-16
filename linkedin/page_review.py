"""Process-local observation and settlement for one exact results page."""

from __future__ import annotations

from dataclasses import dataclass, field
from shared.contracts import LEGACY_REVIEW_DECISIONS
from shared.judger import SAVE_FAMILY_DECISIONS
from shared.schemas import CandidateSnippet, OpusDecision
from linkedin.evaluation_support import _is_containable_facial_page_error
from linkedin.search_policy import _normalize_title_family
import asyncio
import random
import threading
import time
import traceback
import uuid
from shared.human_timing import human_delay_correlated
from shared.schemas import SearchString, Progress, GlanceResult
from linkedin.browser import LinkedInBrowser
from linkedin.facial_batching import FacialBatchFailureOutcome
from shared.failures import judgment_failure_decision
from shared.judger import is_failure_decision
from shared.storage import log_event
from shared.bias_controls import DecisionRecord
from shared import config
from shared.governor import OperatorStopRequested, SessionExpired
from linkedin import judgment_runtime as _judgment_runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linkedin.evaluation_context import EvaluationContext
    from linkedin.candidate_ledger import CandidateLedger
    from linkedin.search_surface import SearchSurface
    from linkedin.candidate_review import CandidateReviewer
    from linkedin.allocator_state import AllocatorStateService
    from linkedin.evaluation_state import PageContext


def empty_page_observation(slots: int = 0) -> dict[str, int | str]:
    return {
        "slots": int(slots or 0), "extracted": 0, "judged": 0, "errored": 0,
        "skipped_dup": 0, "skipped_blacklist": 0, "skipped_missing_url": 0,
        "full_expected": 0, "full_settled": 0, "priority": 0,
        "standard": 0, "outreach": 0, "break_reason": "",
    }


def terminal_full_decision(decision: OpusDecision) -> bool:
    return decision.stage == "full" and (
        decision.decision in SAVE_FAMILY_DECISIONS
        or decision.decision in LEGACY_REVIEW_DECISIONS
        or decision.decision == "REJECT"
    )


@dataclass
class PageReviewContext:
    """Five owned fields; historical work never borrows the live page."""

    _latest_page_observed: dict[str, int | str] = field(default_factory=empty_page_observation)
    _allocator_page_expected_keys: set[str] = field(default_factory=set)
    _allocator_page_settled_keys: set[str] = field(default_factory=set)
    _allocator_page_identity: tuple[int, str, int] | None = None
    _allocator_page_off_policy: bool = False

    def observation(self) -> dict[str, int | str]:
        return dict(self._observed())

    def _observed(self) -> dict[str, int | str]:
        if not isinstance(self._latest_page_observed, dict):
            self._latest_page_observed = empty_page_observation()
        return self._latest_page_observed

    def note(self, key: str, amount: int = 1) -> None:
        observed = self._observed()
        observed[key] = int(observed.get(key, 0) or 0) + int(amount)

    def matches(self, snippet: CandidateSnippet) -> bool:
        if self._allocator_page_identity is None:
            return False
        root_string_id, _variant_id, page = self._allocator_page_identity
        return (int(snippet.source_string_id) == root_string_id
                and max(1, int(snippet.page or 1)) == page)

    def set_break_reason(self, reason: str, *, force: bool = False) -> None:
        if reason not in {"", "glance_reformulate", "early_exit", "panel_stuck",
                          "operator_stop", "session_expired"}:
            return
        observed = self._observed()
        if force or not observed.get("break_reason"):
            observed["break_reason"] = reason

    def expect(self, snippet: CandidateSnippet, key: str) -> None:
        if not self.matches(snippet) or key in self._allocator_page_expected_keys:
            return
        self._allocator_page_expected_keys.add(key)
        self.note("full_expected")

    def settle(self, snippet: CandidateSnippet, decision: OpusDecision, key: str) -> None:
        if not self.matches(snippet) or not terminal_full_decision(decision):
            return
        if key in self._allocator_page_settled_keys:
            return
        self._allocator_page_settled_keys.add(key)
        self.note("full_settled")
        if decision.decision not in SAVE_FAMILY_DECISIONS:
            return
        self.note("outreach")
        tier = str(getattr(decision, "outreach_tier", "") or "").upper()
        if tier == "PRIORITY":
            self.note("priority")
        elif tier == "STANDARD":
            self.note("standard")


class PageReviewer:
    """Page enumeration, facial review and ordered candidate publication."""

    def __init__(
        self,
        context: EvaluationContext,
        ledger: CandidateLedger,
        surface: SearchSurface,
        reviewer: CandidateReviewer,
        allocator: AllocatorStateService,
        page: PageContext,
    ) -> None:
        self.context, self.ledger = context, ledger
        self.surface, self.reviewer = surface, reviewer
        self.allocator, self.page = allocator, page


    def _reset_page_observation(
        self,
        slots: int = 0,
        *,
        search_string: SearchString | None = None,
        page_num: int | None = None,
    ) -> PageReviewContext:
        state = (
            self.allocator.experiment(search_string.id)
            if search_string is not None
            else None
        )
        identity = (
            (
                search_string.id,
                state.active_variant_id if state is not None else "",
                int(page_num),
            )
            if search_string is not None and page_num is not None
            else None
        )

        self.page.review = PageReviewContext(
            _latest_page_observed=empty_page_observation(slots),
            _allocator_page_identity=identity,
            _allocator_page_off_policy=(
                self.allocator._allocator_run_diverged() if self.allocator._allocator_shadow_enabled() else False
            ),
        )
        return self.page.review


    def _page_observation(self, page_context: PageReviewContext | None = None) -> dict[str, int | str]:
        return (page_context or self.page.review).observation()


    def _note_page_observation(self, key: str, amount: int = 1, *, page_context: PageReviewContext | None = None) -> None:
        (page_context or self.page.review).note(key, amount)


    def _set_page_break_reason(self, reason: str, *, force: bool = False, page_context: PageReviewContext | None = None) -> None:
        (page_context or self.page.review).set_break_reason(reason, force=force)


    def _assert_page_full_reviews_settled(
        self,
        *,
        progress: Progress,
        search_string: SearchString,
        page_num: int,
        page_context: PageReviewContext | None = None,
    ) -> None:
        page_context = page_context or self.page.review
        missing = (
            page_context._allocator_page_expected_keys
            - page_context._allocator_page_settled_keys
        )
        if not missing:
            return
        print(
            f"    [full-review] Carrying {len(missing)} unsettled review(s) "
            f"forward for string #{search_string.id}"
        )


    def _note_page_judgment(self, decision: OpusDecision | None, *, page_context: PageReviewContext | None = None) -> None:
        if decision is None:
            return
        if is_failure_decision(getattr(decision, "decision", "")):
            self._note_page_observation("errored", page_context=page_context)
        else:
            self._note_page_observation("judged", page_context=page_context)


    async def _extract_card_snippet(
        self,
        search_string: SearchString,
        page_num: int,
        card_index: int,
    ) -> CandidateSnippet | None:
        self.reviewer.ensure_acquisition()
        result = await self.reviewer.acquisition.extract_card_snippet(
            search_string,
            page_num,
            card_index,
        )
        self.page._last_card_extraction_error = (
            self.reviewer.acquisition.last_card_extraction_error
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
        if self.context.brief_obj and self.context.brief_obj.has_v2_schema:
            return await self._review_page_batch(
                search_string, page_num, result_count, page_report,
                all_candidates, string_stats, progress,
            )

        slot_count = await self.surface._card_slot_count_or_raise(
            search_string=search_string,
            page_num=page_num,
            result_count=result_count,
            mode="sequential",
        )
        page_context = self._reset_page_observation(
            slot_count,
            search_string=search_string,
            page_num=page_num,
        )
        self._seed_page_pending_full_reviews(
            search_string=search_string,
            page_num=page_num,
            page_context=page_context,
        )
        print(f"  Reviewing {slot_count} card slots sequentially", flush=True)

        glance_result = None
        preview_snippets: list[CandidateSnippet] = []
        self.page._latest_page_preview_snippets = []
        consecutive_api_errors = 0
        consecutive_card_extraction_failures = 0
        page_evaluated = 0
        page_facial_no = 0

        for card_index in range(slot_count):
            snippet = await self._extract_card_snippet(search_string, page_num, card_index)
            extraction_error = getattr(self.page, "_last_card_extraction_error", None)
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

            self._note_page_observation("extracted", page_context=page_context)
            preview_snippets.append(snippet)
            self.page._latest_page_preview_snippets = list(preview_snippets)
            print(
                f"    [card {card_index + 1}/{slot_count}] {snippet.name} — "
                f"{snippet.current_title or snippet.headline or 'preview only'}"
            )

            # Drop cards without a durable Recruiter identity fragment before any
            # judgment spend. The fragment must be a Recruiter profile link;
            # ``#``, a search URL, and a public ``/in/`` link do not qualify.
            # Check the fragment rather than string truthiness so an unusable
            # anchor cannot reach evaluation and fail at save time.
            if not LinkedInBrowser._profile_url_fragment(snippet.profile_url):
                print(f"    [skip] {snippet.name} — no usable Recruiter identity")
                self._note_page_observation("skipped_missing_url", page_context=page_context)
                if page_report:
                    page_report.add_skip_preview(snippet.name, "missing_profile_url")
                if self.context._runtime_bridge and self.context._runtime_run_id:
                    self.context._runtime_bridge.record_missing_identity(
                        run_id=self.context._runtime_run_id,
                        search_string=search_string,
                        snippet=snippet,
                    )
                continue

            pending_facial = self.ledger._resume_pending_full_decision(snippet)
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
                and (self.ledger.is_terminal(url) or self.ledger.is_in_flight(url))
            ):
                if self.ledger.is_terminal(url):
                    prior = self.ledger.prior_outcome(url)
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
                        self.ledger.allow_review(url)
                    else:
                        print(f"    [dup] {snippet.name} — already processed")
                if self.ledger.is_terminal(url) or self.ledger.is_in_flight(url):
                    page_report.add_skip_preview(snippet.name, "duplicate")
                    self._note_page_observation("skipped_dup", page_context=page_context)
                    string_stats["duplicates"] += 1
                    # Bucket prior-session suppressions separately —
                    # they must not feed same-epoch exhaustion accounting.
                    if self.ledger.was_seen_before_run(url):
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
                print(f"    [CIRCUIT BREAKER] {consecutive_api_errors} consecutive full-eval failures — pausing 60s")
                log_event(self.context.log_path, "circuit_breaker", consecutive_errors=consecutive_api_errors)
                await asyncio.sleep(60)
                consecutive_api_errors = 0

            if snippet.profile_url:
                self.ledger.start_review(snippet.profile_url)
            if self.ledger._record_candidate_funnel_discovery(
                snippet=snippet,
                string_stats=string_stats,
                stats=self.context.stats,
            ):
                self.reviewer.attempts._record_runtime_snippet(search_string, snippet)
                self.reviewer._record_snippet_activity(snippet, string_stats)

            decision = await self.reviewer._evaluate_snippet(
                snippet,
                page_report,
                search_string,
                string_stats=string_stats,
                page_context=page_context,
            )
            from linkedin.candidate_review import (
                EMPLOYER_GATE_PATHS,
                gate_skip_outcome,
            )

            if (
                decision
                and decision.stage == "facial"
                and decision.path in EMPLOYER_GATE_PATHS
            ):
                self._note_page_observation("skipped_blacklist", page_context=page_context)
            else:
                self._note_page_judgment(decision, page_context=page_context)

            if self.reviewer._is_recoverable_provider_failure_decision(decision):
                consecutive_api_errors += 1
            else:
                consecutive_api_errors = 0

            outcome = "error"
            # A territory or scope skip (card or profile) is not a verdict on
            # fit: its own page outcome, and no facial or full funnel count.
            skip_outcome = gate_skip_outcome(decision.path) if decision else None
            if decision:
                if skip_outcome is None:
                    self.ledger._record_full_funnel_outcome(
                        snippet=snippet,
                        decision=decision,
                        search_string=search_string,
                        string_stats=string_stats,
                        stats=self.context.stats,
                    )
                if skip_outcome is not None:
                    outcome = skip_outcome
                elif decision.stage == "facial" and decision.decision == "FACIAL_NO":
                    outcome = "facial_no"
                elif decision.stage == "facial" and decision.decision == "FACIAL_SKIP":
                    outcome = "facial_skip"
                    string_stats.setdefault("facial_skip", 0)
                    string_stats["facial_skip"] += 1
                elif decision.decision in SAVE_FAMILY_DECISIONS:
                    # A judge SAVE whose side effect physically failed
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
                            # Capture who the save actually was, so memory
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
                        self.surface.geography._warn_if_off_geo_save(snippet)
                elif decision.decision == "REJECT":
                    outcome = "reject"
                elif decision.decision in LEGACY_REVIEW_DECISIONS:
                    # Historical rows only — retired 2026-09-16.
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
            elif skip_outcome is not None and decision.stage == "facial":
                # Same dwell the card had as a facial_no, so pacing is unchanged.
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

            self.allocator._checkpoint_progress(progress, search_string=search_string, page_num=page_num)

            if decision and getattr(decision, "_panel_stuck", False):
                await self.surface._recover_stuck_profile_panel(
                    candidate_name=snippet.name,
                    page_num=page_num,
                    decision=decision,
                    progress=progress,
                    search_string=search_string,
                    page_context=page_context,
                )

            if glance_result is None and len(preview_snippets) >= config.GLANCE_MIN_SNIPPETS:
                glance_result = self._glance_assess(preview_snippets)
                log_event(
                    self.context.log_path,
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
                    self._set_page_break_reason("glance_reformulate", page_context=page_context)
                    break

            if page_evaluated >= config.EARLY_EXIT_MIN_CANDIDATES:
                page_no_rate = page_facial_no / page_evaluated
                if page_no_rate >= self._get_early_exit_rate():
                    print(
                        f"    [early-exit] {page_facial_no}/{page_evaluated} "
                        f"facial_no ({page_no_rate:.0%}) — breaking page"
                    )
                    log_event(
                        self.context.log_path,
                        "early_exit",
                        string_id=search_string.id,
                        page=page_num,
                        evaluated=page_evaluated,
                        facial_no=page_facial_no,
                        rate=round(page_no_rate, 2),
                    )
                    self._set_page_break_reason("early_exit", page_context=page_context)
                    break

            if progress and hasattr(self.context, "_session_expired") and self.context._session_expired.is_set():
                self.allocator._checkpoint_progress(progress)
                self._set_page_break_reason("session_expired", page_context=page_context)
                raise SessionExpired("session_duration_cap")
            if progress and hasattr(self.context, "_operator_stop_event") and self.context._operator_stop_event.is_set():
                self.allocator._checkpoint_progress(progress)
                print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                self._set_page_break_reason("operator_stop", page_context=page_context)
                raise OperatorStopRequested()

            if progress and hasattr(self.context, "_pause_requested") and self.context._pause_requested.is_set():
                self.context._pause_requested.clear()
                self.allocator._checkpoint_progress(progress)
                try:
                    await asyncio.wait_for(self.context._resume_event.wait(), timeout=300)
                except asyncio.TimeoutError:
                    print("  [!] Resume timeout (5 min) — continuing without decoy burst.")
                    self.context._resume_event.set()

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

        _judgment_runtime._validate_judgment_runtime_configuration()
        slot_count = await self.surface._card_slot_count_or_raise(
            search_string=search_string,
            page_num=page_num,
            result_count=result_count,
            mode="batch",
        )
        page_context = self._reset_page_observation(
            slot_count,
            search_string=search_string,
            page_num=page_num,
        )
        self._seed_page_pending_full_reviews(
            search_string=search_string,
            page_num=page_num,
            page_context=page_context,
        )
        print(f"  Reviewing {slot_count} card slots (batch mode)", flush=True)

        # ── Phase 1: Extract all card snippets ──────────────────────────
        glance_result = None
        preview_snippets: list[CandidateSnippet] = []
        glance_sample_snippets: list[CandidateSnippet] = []
        self.page._latest_page_preview_snippets = []
        eligible_snippets: list[CandidateSnippet] = []
        resumed_full_snippets: list[CandidateSnippet] = []
        consecutive_card_extraction_failures = 0

        for card_index in range(slot_count):
            snippet = await self._extract_card_snippet(search_string, page_num, card_index)
            extraction_error = getattr(self.page, "_last_card_extraction_error", None)
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

            self._note_page_observation("extracted", page_context=page_context)
            preview_snippets.append(snippet)
            self.page._latest_page_preview_snippets = list(preview_snippets)
            print(
                f"    [card {card_index + 1}/{slot_count}] {snippet.name} — "
                f"{snippet.current_title or snippet.headline or 'preview only'}"
            )

            # Drop cards without a durable Recruiter identity fragment before any
            # judgment spend. The fragment must be a Recruiter profile link;
            # ``#``, a search URL, and a public ``/in/`` link do not qualify.
            # Check the fragment rather than string truthiness so an unusable
            # anchor cannot reach evaluation and fail at save time.
            if not LinkedInBrowser._profile_url_fragment(snippet.profile_url):
                print(f"    [skip] {snippet.name} — no usable Recruiter identity")
                self._note_page_observation("skipped_missing_url", page_context=page_context)
                if page_report:
                    page_report.add_skip_preview(snippet.name, "missing_profile_url")
                if self.context._runtime_bridge and self.context._runtime_run_id:
                    self.context._runtime_bridge.record_missing_identity(
                        run_id=self.context._runtime_run_id,
                        search_string=search_string,
                        snippet=snippet,
                    )
                continue

            pending_facial = self.ledger._resume_pending_full_decision(snippet)
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
                self.ledger.start_review(snippet.profile_url)
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
            if url and (self.ledger.is_terminal(url) or self.ledger.is_in_flight(url)):
                if self.ledger.is_terminal(url):
                    prior = self.ledger.prior_outcome(url)
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
                        self.ledger.allow_review(url)
                    else:
                        print(f"    [dup] {snippet.name} — already processed")
                if self.ledger.is_terminal(url) or self.ledger.is_in_flight(url):
                    page_report.add_skip_preview(snippet.name, "duplicate")
                    self._note_page_observation("skipped_dup", page_context=page_context)
                    string_stats["duplicates"] += 1
                    # Bucket prior-session suppressions separately —
                    # they must not feed same-epoch exhaustion accounting.
                    if self.ledger.was_seen_before_run(url):
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

            # Employer, alumni, and territory checks (no LLM call)
            from linkedin.candidate_review import (
                facial_gate_console_line,
                facial_gate_labels,
                gate_skip_outcome,
            )

            blacklisted = False
            employer_gate = self.reviewer._facial_employer_gate(snippet)
            if employer_gate:
                gate_path, (blocked, matched_field, matched_value) = employer_gate
                gate_tag, gate_label, gate_stat = facial_gate_labels(gate_path)
                print(
                    facial_gate_console_line(
                        gate_path, snippet.name, blocked, matched_field, matched_value
                    )
                )
                self.context.stats.setdefault(gate_stat, 0)
                self.context.stats[gate_stat] += 1
                if page_report:
                    page_report.add_skip_preview(snippet.name, f"{gate_tag}: {blocked}")
                bl_decision = OpusDecision(
                    stage="facial", decision="FACIAL_NO", path=gate_path,
                    confidence=1.0, rationale=f"{gate_label}: {blocked} ({matched_field})",
                    candidate_name=snippet.name, profile_url=snippet.profile_url,
                )
                # Stage writes require a registered candidate row; the store does
                # not auto-create one. The eligible path registers the row in
                # ``_record_runtime_snippet`` before this stage write, so a
                # blacklisted card can record FACIAL_NO without aborting its
                # search string.
                self.reviewer.attempts._record_runtime_snippet(search_string, snippet)
                facial_attempt_id = self.reviewer.attempts._start_runtime_stage_attempt(
                    search_string=search_string,
                    snippet=snippet,
                    stage="facial",
                )
                self.reviewer._finish_runtime_stage_success(
                    attempt_id=facial_attempt_id,
                    stage="facial",
                    snippet=snippet,
                    decision=bl_decision,
                )
                self.ledger.record_prior_outcome(snippet.profile_url, "FACIAL_NO")
                self.ledger._mark_terminal(snippet.profile_url)
                all_candidates.append({
                    "name": snippet.name, "title": snippet.current_title,
                    "company": snippet.current_company, "headline": snippet.headline,
                    "profile_url": snippet.profile_url,
                    "outcome": gate_skip_outcome(gate_path) or "facial_no",
                    "rationale": bl_decision.rationale,
                    "page": page_num,
                })
                # A territory or scope skip stays out of the facial_no counters,
                # which feed exhaustion and family yield (NON_VERDICT_GATE_PATHS).
                if gate_skip_outcome(gate_path) is None:
                    self.ledger._record_facial_funnel_outcome(
                        snippet=snippet,
                        decision=bl_decision.decision,
                        search_string=search_string,
                        string_stats=string_stats,
                        stats=self.context.stats,
                    )
                blacklisted = True
                self._note_page_observation("skipped_blacklist", page_context=page_context)
            if blacklisted:
                continue

            # Eligible for batch facial
            if snippet.profile_url:
                self.ledger.start_review(snippet.profile_url)
            if self.ledger._record_candidate_funnel_discovery(
                snippet=snippet,
                string_stats=string_stats,
                stats=self.context.stats,
            ):
                self.reviewer.attempts._record_runtime_snippet(search_string, snippet)
                self.reviewer._record_snippet_activity(snippet, string_stats)
            eligible_snippets.append(snippet)
            glance_sample_snippets.append(snippet)

            # Glance assessment
            if glance_result is None and len(glance_sample_snippets) >= config.GLANCE_MIN_SNIPPETS:
                glance_result = self._glance_assess(glance_sample_snippets)
                log_event(
                    self.context.log_path, "glance_assess", string_id=search_string.id,
                    page=page_num, action=glance_result.action,
                    confidence=glance_result.confidence, signals=glance_result.signals,
                )
                print(
                    f"  [glance] {glance_result.action} ({glance_result.confidence:.2f}): "
                    f"{glance_result.summary}"
                )

            if progress and hasattr(self.context, "_session_expired") and self.context._session_expired.is_set():
                self.allocator._checkpoint_progress(progress)
                self._set_page_break_reason("session_expired", page_context=page_context)
                raise SessionExpired("session_duration_cap")
            if progress and hasattr(self.context, "_operator_stop_event") and self.context._operator_stop_event.is_set():
                self.allocator._checkpoint_progress(progress)
                print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                self._set_page_break_reason("operator_stop", page_context=page_context)
                raise OperatorStopRequested()

        if resumed_full_snippets:
            resume_panel_stuck = await self.reviewer._process_resumed_pending_full_evaluations(
                snippets=resumed_full_snippets,
                page_report=page_report,
                search_string=search_string,
                all_candidates=all_candidates,
                string_stats=string_stats,
                progress=progress,
                page_num=page_num,
                page_context=page_context,
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
            **self.reviewer.attempts._lane_context_for_stage(search_string, stage="facial", variant_id=self.surface._variant_id_for_search_string(search_string)),
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
                        self.reviewer._finish_runtime_stage_failure(
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
                        self.ledger.record_prior_outcome(profile_url, "PAGE_ABANDONED")
                        self.ledger._mark_terminal(profile_url)
                    else:
                        self.reviewer._abort_runtime_stage_attempt(
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
                    self.ledger.release_review(
                        eligible_snippets[position].profile_url
                    )
            if not force_terminal:
                # A page-level abort must release every URL, including one
                # whose ordinary bookkeeping failed before it could be
                # promoted or discarded from the in-process set.
                for page_snippet in eligible_snippets:
                    self.ledger.release_review(page_snippet.profile_url)
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
                # Every batch on this page belongs to this one search string.
                return facial_judge_batch(
                    batch_snippets,
                    brief=self.context.brief_obj,
                    prompt_prefix=self.reviewer._facial_prefix_for_string(search_string),
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
                    self.reviewer.attempts._start_runtime_stage_attempt(
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
                self.reviewer.attempts._record_runtime_event(
                    search_string=search_string,
                    event_type="facial_page_judgment_timing",
                    payload=timing_payload,
                )
                log_event(
                    self.context.log_path,
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
            self.allocator._checkpoint_progress(
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
                self.allocator._checkpoint_progress(
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
                    self.context.log_path,
                    "page_abandoned",
                    string_id=search_string.id,
                    page=page_num,
                    candidate_identities=page_abandoned_identities,
                )
            finally:
                self.allocator._restore_incomplete_page_rollback(search_string)

        try:
            try:
                # Preserve raw borderline observability while also retaining the
                # verdict itself in canonical state below.
                ternary_observable = (
                    self.reviewer._facial_ternary_enabled()
                    and self.context._bias_monitor is not None
                )
                for raw_snippet, raw_decision in zip(eligible_snippets, decisions):
                    self.reviewer._stamp_read_interest(raw_snippet, raw_decision.decision)
                    if (
                        ternary_observable
                        and raw_decision.decision == "FACIAL_BORDERLINE"
                    ):
                        self.context._bias_monitor.record_facial_borderline_seen(
                            string_id=str(raw_snippet.source_string_id),
                        )
                decisions = [
                    self.reviewer._normalize_facial_decision_for_persistence(d) for d in decisions
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
                self.allocator._checkpoint_progress(
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
                    self._note_page_judgment(facial, page_context=page_context)

                    if position in page_abandoned_positions:
                        self.context.stats.setdefault("page_abandoned_candidates", 0)
                        self.context.stats["page_abandoned_candidates"] += 1
                        string_stats.setdefault("page_abandoned_candidates", 0)
                        string_stats["page_abandoned_candidates"] += 1
                        if self.context._bias_monitor:
                            self.context._bias_monitor.record_decision(DecisionRecord(
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
                        self.reviewer.attempts._finish_runtime_failure_decision(
                            attempt_id=facial_attempt_id,
                            snippet=snippet,
                            decision=facial,
                            payload=facial_attempt_payload,
                        )
                        open_facial_attempt_positions.discard(position)
                        self.ledger.release_review(snippet.profile_url)
                        print(f"    [PARSE_FAILURE] {snippet.name}: {facial.rationale}")
                        self.context.stats.setdefault("parse_failures", 0)
                        self.context.stats["parse_failures"] += 1
                        if self.context._bias_monitor:
                            self.context._bias_monitor.record_decision(DecisionRecord(
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

                    self.reviewer._finish_runtime_stage_success(
                        attempt_id=facial_attempt_id,
                        stage="facial",
                        snippet=snippet,
                        decision=facial,
                        extra_payload=facial_attempt_payload,
                    )
                    open_facial_attempt_positions.discard(position)
                    self.ledger.record_prior_outcome(snippet.profile_url, facial.decision)
                    self.ledger._mark_terminal(snippet.profile_url)

                    # Bias monitoring
                    if self.context._bias_monitor:
                        self.context._bias_monitor.record_decision(DecisionRecord(
                            candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                            string_id=str(snippet.source_string_id),
                            stage="facial", decision=facial.decision,
                            confidence=facial.confidence, capability_area=None,
                        ))

                    if facial.decision in ("FACIAL_NO", "FACIAL_SKIP"):
                        tag = "FACIAL_NO" if facial.decision == "FACIAL_NO" else "FACIAL_SKIP"
                        print(f"    [{tag}] {snippet.name}: {facial.rationale}")
                        if facial.decision == "FACIAL_NO":
                            self.ledger._record_facial_funnel_outcome(
                                snippet=snippet,
                                decision=facial.decision,
                                search_string=search_string,
                                string_stats=string_stats,
                                stats=self.context.stats,
                            )
                            page_facial_no += 1
                        else:
                            self.context.stats.setdefault("facial_skip", 0)
                            self.context.stats["facial_skip"] += 1
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
                        self.ledger._record_facial_funnel_outcome(
                            snippet=snippet,
                            decision=facial.decision,
                            search_string=search_string,
                            string_stats=string_stats,
                            stats=self.context.stats,
                        )
                        if self.reviewer._should_skip_full_eval_for_activity(snippet, facial):
                            self.reviewer._record_activity_saturation_context(
                                snippet=snippet,
                                search_string=search_string,
                                facial_decision=facial,
                            )
                        facial_yes_snippets.append(snippet)
                        self.ledger._track_full_review_obligation(
                            snippet,
                            facial.decision,
                        )
                        self.reviewer._note_page_full_review_expected(snippet, page_context=page_context)
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
                    self.allocator._checkpoint_progress(
                        progress,
                        search_string=search_string,
                        page_num=page_num,
                    )
                    raise

            try:
                if contract_corruption_call_ids:
                    self.context.stats["facial_contract_corruptions"] += len(
                        contract_corruption_call_ids
                    )
                    if self.context.stats["facial_contract_corruptions"] >= 2:
                        print(
                            "    [CIRCUIT BREAKER] two facial tool-contract corruptions "
                            "— stopping optimized run",
                            flush=True,
                        )
                        log_event(
                            self.context.log_path,
                            "circuit_breaker",
                            stage="facial_tool_contract",
                            consecutive_errors=self.context.stats["facial_contract_corruptions"],
                        )
                        self.allocator._checkpoint_progress(
                            progress,
                            search_string=search_string,
                            page_num=page_num,
                        )
                        raise RuntimeError(
                            "facial tool-contract corruption threshold reached"
                        )

                if decisions and all(
                    self.reviewer._is_recoverable_provider_failure_decision(decision)
                    for decision in decisions
                ):
                    recoverable_call_ids = {
                        str((decision.prompt_capture or {}).get("logical_call_id") or "")
                        for decision in decisions
                    }
                    recoverable_call_ids.discard("")
                    self.context.stats["consecutive_facial_provider_failures"] += max(
                        1,
                        len(recoverable_call_ids),
                    )
                else:
                    self.context.stats["consecutive_facial_provider_failures"] = 0
                if self.context.stats["consecutive_facial_provider_failures"] >= 3:
                    print(
                        "    [CIRCUIT BREAKER] repeated facial provider failures "
                        "— stopping run",
                        flush=True,
                    )
                    log_event(
                        self.context.log_path,
                        "circuit_breaker",
                        stage="facial_provider",
                        consecutive_errors=self.context.stats[
                            "consecutive_facial_provider_failures"
                        ],
                    )
                    self.allocator._checkpoint_progress(
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
                # Tightening applies to the NEXT page; this batch has already
                # dispatched. The flag is OFF by default. When enabled, a
                # per-string YES rate above 2x the brief-level expected-high
                # band sets the next-page facial prompt to require TWO strong
                # positive signals instead of one. The band is brief-scoped
                # while the observed rate is per-string. Facial anomaly alerts,
                # the block-report band, and string_context telemetry are
                # unchanged.
                if (
                    config.LINKEDIN_FACIAL_TIGHTENING_ENABLED
                    and not self.reviewer._triage_tightened
                    and self.context._bias_monitor
                ):
                    tightening = self.context._bias_monitor.get_tightening_status(str(snippet.source_string_id))
                    if tightening:
                        self.reviewer._triage_tightened = True
                        self.reviewer._tightening_prefix = (
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
                            self.context.log_path, "early_exit", string_id=search_string.id,
                            page=page_num, evaluated=page_evaluated,
                            facial_no=page_facial_no, rate=round(page_no_rate, 2),
                        )
                        self._set_page_break_reason("early_exit", page_context=page_context)

                self.allocator._checkpoint_progress(progress, search_string=search_string, page_num=page_num)
            except BaseException as exc:
                abort_open_page_facial_attempts(
                    exc,
                    failure_marker="facial_postdispatch_failed",
                )
                raise

            # Full review uses one queue; the flag selects its candidate window.
            await self.reviewer._run_full_queue(
                snippets=facial_yes_snippets,
                window=2 if config.FULL_EVAL_PIPELINE_ENABLED else 1,
                origin="page", page_report=page_report,
                search_string=search_string, all_candidates=all_candidates,
                string_stats=string_stats, progress=progress, page_num=page_num,
                page_context=page_context,
            )
            return glance_result
        finally:
            finalize_page_containment()


    def _seed_page_pending_full_reviews(
        self,
        *,
        search_string: SearchString,
        page_num: int,
        page_context: PageReviewContext | None = None,
    ) -> None:
        for snippet in self.ledger.pending_page(search_string.id, page_num):
            self.reviewer._note_page_full_review_expected(snippet, page_context=page_context)


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
        density = str(getattr(self.context.brief_obj, "market_density", "") or "").strip().lower()
        return config.EARLY_EXIT_FACIAL_NO_RATE_BY_DENSITY.get(
            density, config.EARLY_EXIT_FACIAL_NO_RATE
        )


    def _get_glance_key_terms(self) -> list[str]:
        """Extract key terms from the brief for glance keyword scanning."""
        nb = self.context.brief_obj._new_brief
        if nb is not None:
            # V2 brief: collect key_terms from all capability_areas
            terms = []
            for ca in nb.capability_areas:
                terms.extend(t.lower() for t in ca.key_terms)
            return terms
        # Old brief: extract from archetypes save_signals
        terms = []
        for arch in self.context.brief_obj.archetypes:
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
                nb = self.context.brief_obj._new_brief
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
                    for na in self.context.brief_obj.noise_archetypes:
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
        nb = self.context.brief_obj._new_brief
        nf_examples: list[str] = []
        if nb is not None:
            for nf in nb.non_fit_patterns:
                nf_examples.extend(ex.lower() for ex in nf.examples)
        else:
            for na in self.context.brief_obj.noise_archetypes:
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
        from shared.judgment.tool_contracts import (
            fence_untrusted,
            untrusted_data_instruction,
        )
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

Role: {self.context.brief_obj.role_title}
{self.context.brief_obj.role_description}

Minimum bar: {self.context.brief_obj.minimum_bar}

{untrusted_data_instruction()}

Respond with ONLY the word "noise" or "proceed". Nothing else."""

        user = f"""Here are the first {len(snippets[:10])} search result snippets from this page:
{fence_untrusted(snippets_text)}

Automated signal analysis:
{signals_text}

Is this page mostly noise (wrong population) or does it have plausible candidates? Reply "noise" or "proceed"."""

        usage_context = {
            "stage": "linkedin_glance_llm_check",
            "source": "linkedin",
            "brief_id": getattr(self.context.brief_obj, "id", None) or self.context._brief_id,
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
                self.context.log_path,
                "glance_llm_error",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return "proceed"
