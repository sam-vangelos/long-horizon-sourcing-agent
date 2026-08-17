"""Humanized LinkedIn search-mutation executor."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from shared import config
from shared.human_timing import human_delay_correlated
from shared.runtime_state.store import LINKEDIN_STRING_KIND
from shared.storage import log_event

if TYPE_CHECKING:
    from linkedin.lane_variant_decisions import VariantDecisionOutput
    from linkedin.search_intelligence import LinkedInExperimentState, LinkedInSearchVariant
    from shared.schemas import SearchString


@dataclass
class SearchMutationResult:
    applied: bool
    result_count: int = 0
    result_count_text: str = ""
    blocked_reason: str = ""
    top_card_snapshot: dict[str, Any] | None = None
    structured_apply: dict[str, Any] | None = None
    hybrid_partial: bool = False


@dataclass
class LinkedInSearchMutationDeps:
    browser: Any
    log_path: Path
    get_input_mode: Callable[[], str]
    get_runtime_run_id: Callable[[], int | None]
    get_runtime_state: Callable[[], Any]
    get_search_mutation_budget_used: Callable[[], int]
    set_search_mutation_budget_used: Callable[[int], None]


def _clear_filter_list(target: Any, name: str, is_dict: bool) -> None:
    """Empty a list-valued structured dimension on either a LinkedInStructuredFilters
    dataclass (attribute) or a checkpoint dict (key). Used by the demote-and-proceed
    clear so the variant and the checkpointed search_string.structured_filters strip
    the same dropped dim through one code path."""
    if is_dict:
        if isinstance(target.get(name), list):
            target[name] = []
    else:
        getattr(target, name).clear()


def _pop_filter_bucket_key(target: Any, bucket: str, key: str, is_dict: bool) -> None:
    """Drop a single key from a bucket-valued structured dimension (sidebar_filters /
    advanced_filters) on either the dataclass or the checkpoint dict."""
    if is_dict:
        container = target.get(bucket)
        if isinstance(container, dict):
            container.pop(key, None)
    else:
        getattr(target, bucket).pop(key, None)


class LinkedInSearchMutationExecutor:
    """Applies keyword-only search mutations using the existing sidebar workflow."""

    def __init__(self, deps: LinkedInSearchMutationDeps):
        self.deps = deps

    async def apply_variant(
        self,
        *,
        search_string: "SearchString",
        experiment_state: "LinkedInExperimentState",
        variant: "LinkedInSearchVariant",
        mutation_kind: str = "experiment",
        mutation_summary: dict[str, Any] | None = None,
        acquisition_mode: str = "linkedin_boolean",
    ) -> SearchMutationResult:
        # Slice D defense-in-depth: a structured_only variant must carry the
        # structured filters meant to bound the search. With no filters (and the
        # keyword dropped) it has nothing to constrain on — dispatching it would run
        # a keyword-less, control-less whole-population search. The orchestrator
        # ingestion guards block this today; reject here too so apply_variant is safe
        # in isolation against any future caller or refactor.
        if variant.surface == "structured_only" and variant.structured_filters.is_empty():
            self._record_event(
                search_string=search_string,
                event_type="linkedin_search_mutation_rejected",
                payload={
                    "reason": "structured_only_without_filters",
                    "variant_id": variant.variant_id,
                },
            )
            return SearchMutationResult(
                applied=False,
                blocked_reason="structured_only_without_filters",
            )
        if not variant.structured_filters.is_empty():
            if acquisition_mode != "linkedin_hybrid":
                self._record_event(
                    search_string=search_string,
                    event_type="linkedin_search_mutation_rejected",
                    payload={
                        "reason": "experimental_structured_filters_not_supported",
                        "variant_id": variant.variant_id,
                    },
                )
                return SearchMutationResult(
                    applied=False,
                    blocked_reason="experimental_structured_filters_not_supported",
                )
        if experiment_state.consecutive_mutations >= config.SEARCH_EXPERIMENT_MAX_CONSECUTIVE_REWRITES:
            self._record_event(
                search_string=search_string,
                event_type="linkedin_search_mutation_blocked",
                payload={
                    "reason": "consecutive_rewrite_limit",
                    "variant_id": variant.variant_id,
                    "mutation_kind": mutation_kind,
                },
            )
            return SearchMutationResult(applied=False, blocked_reason="consecutive_rewrite_limit")
        if (
            self.deps.get_search_mutation_budget_used()
            >= config.SEARCH_EXPERIMENT_MUTATION_BUDGET
        ):
            self._record_event(
                search_string=search_string,
                event_type="linkedin_search_mutation_blocked",
                payload={
                    "reason": "session_humanization_budget",
                    "variant_id": variant.variant_id,
                    "mutation_kind": mutation_kind,
                },
            )
            return SearchMutationResult(applied=False, blocked_reason="session_humanization_budget")
        if mutation_kind == "drift":
            if (
                experiment_state.drift_attempt_count
                >= config.SEARCH_EXPERIMENT_MAX_DRIFT_ATTEMPTS_PER_VARIANT
            ):
                self._record_event(
                    search_string=search_string,
                    event_type="linkedin_search_mutation_blocked",
                    payload={
                        "reason": "drift_attempt_limit",
                        "variant_id": variant.variant_id,
                        "mutation_kind": mutation_kind,
                    },
                )
                return SearchMutationResult(applied=False, blocked_reason="drift_attempt_limit")
            if experiment_state.drift_attempt_count >= config.SEARCH_EXPERIMENT_DRIFT_BUDGET:
                self._record_event(
                    search_string=search_string,
                    event_type="linkedin_search_mutation_blocked",
                    payload={
                        "reason": "string_drift_budget",
                        "variant_id": variant.variant_id,
                        "mutation_kind": mutation_kind,
                    },
                )
                return SearchMutationResult(applied=False, blocked_reason="string_drift_budget")
            if experiment_state.pages_since_last_mutation < 1:
                self._record_event(
                    search_string=search_string,
                    event_type="linkedin_search_mutation_blocked",
                    payload={
                        "reason": "drift_cooldown",
                        "variant_id": variant.variant_id,
                        "mutation_kind": mutation_kind,
                    },
                )
                return SearchMutationResult(applied=False, blocked_reason="drift_cooldown")

        self._record_event(
            search_string=search_string,
            event_type="linkedin_search_mutation_attempt",
            payload={
                "variant_id": variant.variant_id,
                "variant_kind": variant.variant_kind,
                "hypothesis": variant.hypothesis,
                "target_result_min": variant.target_result_min,
                "target_result_max": variant.target_result_max,
                "mutation_kind": mutation_kind,
                "input_mode": self.deps.get_input_mode(),
                "typing_transport": None,
                "typing_duration_ms": None,
                "typo_count": None,
                "used_correction": None,
                "fallback_char_count": None,
                "results_wait_ms": None,
            },
        )
        log_event(
            self.deps.log_path,
            "linkedin_search_mutation_attempt",
            string_id=search_string.id,
            variant_id=variant.variant_id,
            variant_kind=variant.variant_kind,
            hypothesis=variant.hypothesis,
            mutation_kind=mutation_kind,
            input_mode=self.deps.get_input_mode(),
        )

        try:
            if mutation_kind == "drift":
                experiment_state.mark_pending_drift(
                    variant_id=variant.variant_id,
                    parent_variant_id=experiment_state.committed_variant_id or experiment_state.active_variant_id,
                    summary=mutation_summary,
                )
            await self.deps.browser.go_back_to_results()
            await asyncio.sleep(human_delay_correlated(0.45, channel="search_mutation"))
            structured_apply: dict[str, Any] | None = None
            hybrid_partial = False
            used_hybrid = (
                acquisition_mode == "linkedin_hybrid"
                and not variant.structured_filters.is_empty()
            )
            if used_hybrid:
                from linkedin.advanced_search import compile_structured_filters_to_plan

                plan = compile_structured_filters_to_plan(
                    variant.structured_filters,
                    keyword_boolean=variant.boolean,
                    acquisition_mode=acquisition_mode,
                    # Slice D: a structured_only variant drops the keyword at
                    # compile — no keyword control, and plan.keyword_boolean is
                    # zeroed so neither apply nor recovery re-adds it.
                    include_keyword=(variant.surface != "structured_only"),
                )
                apply_result = await self.deps.browser.apply_advanced_search_plan(plan)
                try:
                    log_event(
                        self.deps.log_path,
                        "string_executed",
                        string_id=search_string.id,
                        executed_boolean=plan.keyword_boolean,
                        execution_surface="advanced",
                    )
                except Exception:
                    pass
                structured_apply = apply_result.to_dict()
                requested_non_keyword = [
                    ctrl.dimension
                    for ctrl in plan.controls
                    if ctrl.dimension != "keywords"
                ]
                # Posture-aware honesty (slice E, audit #2): a hybrid/filter_led lane
                # whose structured control DROPPED — whether classified unsupported OR
                # failed at verification — but whose keyword (or a partial structured
                # set) landed is a PARTIAL, never a full success. The pre-E code only
                # flagged the unsupported case; a verification FAILURE that left the
                # keyword running was silently reported full. Both axes are partials.
                dropped_non_keyword = [
                    dim
                    for dim in (
                        list(apply_result.unsupported_controls)
                        + list(apply_result.failed_controls)
                    )
                    if dim in requested_non_keyword
                ]
                if requested_non_keyword and dropped_non_keyword:
                    hybrid_partial = True
                if requested_non_keyword and not apply_result.applied_controls:
                    self._record_event(
                        search_string=search_string,
                        event_type="linkedin_search_mutation_rejected",
                        payload={
                            "reason": "structured_controls_unsupported_no_boolean_fallback",
                            "variant_id": variant.variant_id,
                            "structured_apply": structured_apply,
                        },
                    )
                    return SearchMutationResult(
                        applied=False,
                        blocked_reason="structured_controls_unsupported_no_boolean_fallback",
                        structured_apply=structured_apply,
                        hybrid_partial=hybrid_partial,
                    )
                if apply_result.failed_controls and not apply_result.applied_controls:
                    self._record_event(
                        search_string=search_string,
                        event_type="linkedin_search_mutation_rejected",
                        payload={
                            "reason": "structured_controls_not_applied",
                            "variant_id": variant.variant_id,
                            "structured_apply": structured_apply,
                        },
                    )
                    return SearchMutationResult(
                        applied=False,
                        blocked_reason="structured_controls_not_applied",
                        structured_apply=structured_apply,
                        hybrid_partial=hybrid_partial,
                    )
                # Surface-aware fallback (slice E, part 1). We are past the two abandon
                # guards, so SOMETHING landed (applied_controls is non-empty). When a
                # requested structured dim dropped here, the response depends on the lane
                # surface:
                #   - structured_only NEVER demotes to keyword. It proceeds on whatever
                #     structured controls DID land (the re-entry below takes its synthetic,
                #     no-keyword path); the all-structured-dropped case already abandoned
                #     in the guards above (slice D). So no demotion is recorded for it.
                #   - boolean / filter_led / hybrid DEMOTE-AND-PROCEED: the keyword (or a
                #     partial structured set) carries the search. Emit a
                #     linkedin_structured_demotion event consuming the gate reason, CLEAR
                #     the dropped dims off variant.structured_filters so the next probe and
                #     the recovery snapshot agree on what actually landed, and tick the
                #     deterministic circuit-breaker counter (part 2).
                if (
                    variant.surface != "structured_only"
                    and dropped_non_keyword
                    and not apply_result.plan_fully_applied
                ):
                    # Strip the dropped dims off BOTH the variant (same-process: next
                    # probe + recovery snapshot agree) AND the checkpointed
                    # search_string.structured_filters (cross-process: a resume's
                    # bootstrap_experiment_state must not re-seed a dim the sidebar
                    # rejected on a PARTIAL demote that apply_shadow's full-demote clear
                    # never reaches).
                    self._clear_dropped_structured_dims(
                        variant,
                        dropped_non_keyword,
                        checkpoint=search_string.structured_filters,
                    )
                    experiment_state.structured_demotions += 1
                    self._record_event(
                        search_string=search_string,
                        event_type="linkedin_structured_demotion",
                        payload={
                            "variant_id": variant.variant_id,
                            "surface": variant.surface,
                            "reason": apply_result.reason,
                            "dropped_dimensions": list(dropped_non_keyword),
                            "structured_demotions": experiment_state.structured_demotions,
                            "structured_apply": structured_apply,
                        },
                    )
                # Slice D: this re-entry reads variant.boolean DIRECTLY, so the
                # compile-time keyword suppression does not reach it — gate it on
                # the surface. A structured_only variant must NOT re-enter the
                # keyword here; its structured controls already landed (the
                # all-unsupported case rejected above), so it takes the synthetic
                # SearchEntryResult path with no keyword search.
                if (
                    variant.surface != "structured_only"
                    and "keywords" not in apply_result.applied_controls
                ):
                    entry_result = await self.deps.browser.enter_search_string(variant.boolean)
                    try:
                        log_event(
                            self.deps.log_path,
                            "string_executed",
                            string_id=search_string.id,
                            executed_boolean=variant.boolean,
                            execution_surface="keyword",
                        )
                    except Exception:
                        pass
                else:
                    from linkedin.browser import SearchEntryResult
                    from linkedin.input_backends import TypingResult

                    entry_result = SearchEntryResult(
                        typing_result=TypingResult(
                            transport="advanced_search_plan",
                            duration_ms=0,
                            typo_count=0,
                            used_correction=False,
                            fallback_char_count=0,
                        ),
                        results_wait_ms=0,
                    )
            else:
                entry_result = await self.deps.browser.enter_search_string(variant.boolean)
                try:
                    log_event(
                        self.deps.log_path,
                        "string_executed",
                        string_id=search_string.id,
                        executed_boolean=variant.boolean,
                        execution_surface="keyword",
                    )
                except Exception:
                    pass
            await asyncio.sleep(human_delay_correlated(random.uniform(0.35, 0.75), channel="search_mutation"))
            result_count_text = await self.deps.browser.get_results_count_text()
            result_count = await self.deps.browser.get_results_count()
            top_card_snapshot = None
            try:
                top_card_snapshot = await self.deps.browser.get_card_snapshot(0)
            except Exception:
                top_card_snapshot = None
        except Exception:
            if mutation_kind == "drift":
                experiment_state.rollback_pending_drift()
            raise

        typing_result = getattr(entry_result, "typing_result", None)
        results_wait_ms = getattr(entry_result, "results_wait_ms", 0)
        self.deps.set_search_mutation_budget_used(self.deps.get_search_mutation_budget_used() + 1)
        experiment_state.activate_variant(variant.variant_id)

        payload = {
            "variant_id": variant.variant_id,
            "variant_kind": variant.variant_kind,
            "result_count": result_count,
            "result_count_text": result_count_text,
            "top_card_snapshot": top_card_snapshot or {},
            "mutation_kind": mutation_kind,
            "input_mode": self.deps.get_input_mode(),
            "typing_transport": getattr(typing_result, "transport", None),
            "typing_duration_ms": getattr(typing_result, "duration_ms", None),
            "typo_count": getattr(typing_result, "typo_count", None),
            "used_correction": getattr(typing_result, "used_correction", None),
            "fallback_char_count": getattr(typing_result, "fallback_char_count", None),
            "results_wait_ms": results_wait_ms,
            "acquisition_mode": acquisition_mode,
        }
        if structured_apply is not None:
            payload["structured_apply"] = structured_apply
            payload["hybrid_partial"] = hybrid_partial
        self._record_event(
            search_string=search_string,
            event_type="linkedin_search_mutation_applied",
            payload=payload,
        )
        log_event(
            self.deps.log_path,
            "linkedin_search_mutation_applied",
            string_id=search_string.id,
            variant_id=variant.variant_id,
            result_count=result_count,
            result_count_text=result_count_text,
            mutation_kind=mutation_kind,
            input_mode=self.deps.get_input_mode(),
            typing_transport=getattr(typing_result, "transport", None),
            typing_duration_ms=getattr(typing_result, "duration_ms", None),
            typo_count=getattr(typing_result, "typo_count", None),
            used_correction=getattr(typing_result, "used_correction", None),
            fallback_char_count=getattr(typing_result, "fallback_char_count", None),
            results_wait_ms=results_wait_ms,
        )
        return SearchMutationResult(
            applied=True,
            result_count=result_count,
            result_count_text=result_count_text,
            top_card_snapshot=top_card_snapshot,
            structured_apply=structured_apply,
            hybrid_partial=hybrid_partial,
        )

    def evaluate_lane_variant_lifecycle(
        self,
        *,
        search_string: "SearchString",
        experiment_state: "LinkedInExperimentState",
        variant: "LinkedInSearchVariant",
    ) -> "VariantDecisionOutput":
        """Run deterministic lifecycle decision and persist events/checkpoint metadata."""
        from linkedin.lane_variant_decisions import VariantDecisionInput, decide_variant_lifecycle

        decision = decide_variant_lifecycle(
            VariantDecisionInput(
                variant=variant,
                experiment_state=experiment_state,
            )
        )

        experiment_state.last_variant_decision = decision.to_dict()
        experiment_state.last_variant_decision["variant_id"] = variant.variant_id
        experiment_state.last_variant_decision["lane_id"] = variant.lane_id

        self._record_event(
            search_string=search_string,
            event_type="lane_variant_decision",
            payload={
                "variant_id": variant.variant_id,
                "lane_id": variant.lane_id,
                "action": decision.action,
                "reason": decision.reason,
                "result_window_health": variant.result_window_health,
                "result_count": variant.result_count,
                "saves": variant.saves,
                "facial_yes": variant.facial_yes,
                "probe_pages_used": variant.probe_pages_used,
                "probe_page_budget": variant.probe_page_budget,
                **({"next_variant_hint": decision.next_variant_hint} if decision.next_variant_hint else {}),
            },
        )

        if decision.action == "commit":
            variant.status = "committed"
            variant.lifecycle_reason = decision.reason
        elif decision.action == "abandon":
            variant.status = "abandoned"
            variant.lifecycle_reason = decision.reason
        elif decision.action in ("rescue", "split"):
            variant.status = "exhausted"
            variant.lifecycle_reason = decision.reason

        return decision

    def apply_lane_variant(
        self,
        *,
        search_string: "SearchString",
        experiment_state: "LinkedInExperimentState",
        variant: "LinkedInSearchVariant",
        acquisition_mode: str = "linkedin_boolean",
    ) -> "VariantDecisionOutput":
        """Backward-compatible alias for lifecycle evaluation."""
        return self.evaluate_lane_variant_lifecycle(
            search_string=search_string,
            experiment_state=experiment_state,
            variant=variant,
        )

    @staticmethod
    def _clear_dropped_structured_dims(
        variant: "LinkedInSearchVariant",
        dropped_dimensions: list[str],
        *,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        """Clear demoted PLAN dimensions off the variant's structured_filters.

        The demote-and-proceed path (slice E, part 1) fell back to keyword because a
        structured control dropped. Stripping the dropped dim off the variant means the
        next probe re-compiles WITHOUT it and the applied-only recovery snapshot agrees
        on what actually landed — neither replays a control the sidebar never accepted.

        ``dropped_dimensions`` are PLAN dimensions (the names
        ``compile_structured_filters_to_plan`` emits: ``job_titles`` / ``companies`` /
        ``locations`` / ``fields_of_study``); map each back to its structured_filters
        home before clearing.

        Slice E (medium-gap closure): a PARTIAL demote (one of N dims dropped, surface
        stays hybrid, surviving filters non-empty) satisfies neither apply_shadow's
        full-demote checkpoint-clear (search_intelligence.py:555, which needs
        structured_filters.is_empty()) nor is_deliberate_boolean_demotion. Left alone,
        the checkpointed ``search_string.structured_filters`` keeps the dropped dim, and
        a CROSS-PROCESS resume's bootstrap_experiment_state re-seeds it onto a fresh
        legacy variant — re-applying a control the sidebar already rejected. Passing the
        live ``checkpoint`` dict (``search_string.structured_filters``) strips the SAME
        dropped dims from it in lockstep, so the post-demote checkpoint reflects only
        what landed and the resume cannot re-seed the dropped dim. The dim→home mapping
        is shared with the variant clear above so the two never diverge.
        """
        targets: list[Any] = [variant.structured_filters]
        if checkpoint is not None:
            targets.append(checkpoint)
        for target in targets:
            is_dict = isinstance(target, dict)
            for dimension in dropped_dimensions:
                if dimension == "job_titles":
                    _clear_filter_list(target, "titles", is_dict)
                elif dimension == "companies":
                    _clear_filter_list(target, "companies", is_dict)
                elif dimension == "locations":
                    _pop_filter_bucket_key(target, "sidebar_filters", "locations", is_dict)
                elif dimension == "fields_of_study":
                    _pop_filter_bucket_key(
                        target, "advanced_filters", "fields_of_study", is_dict
                    )
                elif dimension in {"titles", "skills", "assessments"}:
                    # Defensive: a future plan that emits the structured_filters field
                    # name directly (not the plan alias) still clears cleanly.
                    _clear_filter_list(target, dimension, is_dict)

    def _record_event(self, *, search_string: "SearchString", event_type: str, payload: dict[str, Any]) -> None:
        run_id = self.deps.get_runtime_run_id()
        if not run_id:
            return
        runtime_state = self.deps.get_runtime_state()
        work_unit_id = runtime_state.get_work_unit_id(
            run_id,
            kind=LINKEDIN_STRING_KIND,
            source_unit_id=str(search_string.id),
        )
        runtime_state.record_event(
            run_id=run_id,
            work_unit_id=work_unit_id,
            event_type=event_type,
            payload=payload,
        )
