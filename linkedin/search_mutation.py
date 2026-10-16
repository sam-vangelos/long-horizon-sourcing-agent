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
from linkedin.browser import (
    FacetReconciliationError,
    facet_misses,
    facet_target_landed,
    format_facet_reconciliation_console_line,
    format_facet_substituted_console_line,
    format_facet_unapplied_console_line,
    normalize_facet_value_for_compare,
)

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
    # F8: session-geography reader for the per-string location ownership gate.
    # None (legacy construction) fails closed — a location-bearing variant is
    # rejected because its identity with the session set cannot be proved.
    get_session_geography: Callable[[], list[str]] | None = None
    # The brief's employer allowlist (shared/employer_allowlist.py), or None.
    # Under one, every variant keeps a listed competitor on its company facet.
    get_company_scope: Callable[[], Any] | None = None


class PerStringLocationConflictError(FacetReconciliationError):
    """Reject per-string location filters that conflict with session geography.

    Strategy formation forbids per-string locations, but the lane compiler can
    still route one into ``sidebar_filters["locations"]``. Delegate the
    reject-or-omit decision to ``enforce_per_string_location_ownership``.
    """

    def __init__(self, *, per_string: list[str], session: list[str]):
        super().__init__("per_string_location_conflict")
        self.per_string = sorted(per_string, key=str.casefold)
        self.session = sorted(session, key=str.casefold)


def enforce_per_string_location_ownership(
    structured_filters: Any,
    session_geography: list[str] | None,
) -> list[str]:
    """Reject-or-omit a lane-compiled per-string location set (F8).

    Location is session-owned. Under the facet-lifecycle flag the rule is:

    - no per-string locations -> ``[]`` (the dominant case);
    - normalized set EQUAL to the session geography -> redundant; return the
      raw values so the caller omits their controls and records the receipt;
    - anything else (subset, superset, disjoint, or an unprovable session
      set) -> :class:`PerStringLocationConflictError`. Running the lane
      inside the session's pool while attributing its result count to a
      different geographic definition silently corrupts adaptive evidence.
    """
    sidebar = getattr(structured_filters, "sidebar_filters", None)
    raw_locations = sidebar.get("locations") if isinstance(sidebar, dict) else None
    per_string = [
        str(value).strip() for value in (raw_locations or []) if str(value).strip()
    ]
    if not per_string:
        return []
    per_string_normalized = {
        normalize_facet_value_for_compare(value) for value in per_string
    }
    session_values = [
        str(value).strip()
        for value in (session_geography or [])
        if str(value).strip()
    ]
    session_normalized = {
        normalize_facet_value_for_compare(value) for value in session_values
    }
    if session_normalized and per_string_normalized == session_normalized:
        return sorted(per_string, key=str.casefold)
    raise PerStringLocationConflictError(
        per_string=per_string, session=session_values
    )


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

    def _record_facet_reconciliation(
        self,
        *,
        search_string: "SearchString",
        receipt: dict[str, Any],
    ) -> None:
        payload = {
            **receipt,
            "boundary": "search_mutation",
            "string_id": search_string.id,
        }
        self._record_event(
            search_string=search_string,
            event_type="linkedin_facet_reconciliation",
            payload=payload,
        )
        log_event(self.deps.log_path, "linkedin_facet_reconciliation", **payload)
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
        get_company_scope = getattr(self.deps, "get_company_scope", None)
        company_scope = get_company_scope() if get_company_scope is not None else None
        if company_scope is not None:
            acquisition_mode = self._keep_competitor_facet(
                search_string=search_string,
                variant=variant,
                company_scope=company_scope,
                acquisition_mode=acquisition_mode,
            )
        # A structured-only variant must carry filters to constrain the search.
        # With neither filters nor keyword it would run a whole-population search;
        # reject it here so apply_variant is safe for every caller.
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
        redundant_locations: list[str] = []
        if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
            # F8 plan-integrity gate, sibling of the two structural rejections
            # above: runs before any budget spend, event, or browser touch.
            get_session_geography = getattr(self.deps, "get_session_geography", None)
            try:
                redundant_locations = enforce_per_string_location_ownership(
                    variant.structured_filters,
                    get_session_geography()
                    if get_session_geography is not None
                    else None,
                )
            except PerStringLocationConflictError as exc:
                receipt = self._facet_failure_receipt(
                    companies=set(variant.structured_filters.companies or []),
                    titles=set(variant.structured_filters.titles or []),
                    gate="per_string_location_conflict",
                    error=exc,
                )
                receipt["per_string_locations"] = exc.per_string
                receipt["session_geography"] = exc.session
                self._record_facet_reconciliation(
                    search_string=search_string,
                    receipt=receipt,
                )
                self._record_event(
                    search_string=search_string,
                    event_type="linkedin_search_mutation_rejected",
                    payload={
                        "reason": "per_string_location_conflict",
                        "variant_id": variant.variant_id,
                        "per_string_locations": exc.per_string,
                        "session_geography": exc.session,
                    },
                )
                return SearchMutationResult(
                    applied=False,
                    blocked_reason="per_string_location_conflict",
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
            companies = set(variant.structured_filters.companies or [])
            titles = set(variant.structured_filters.titles or [])
            facet_receipt: dict[str, Any] | None = None
            if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
                gate = "search_surface_unready"
                try:
                    await self.deps.browser.assert_search_surface_ready()
                    gate = "facet_reconciliation"
                    await self.deps.browser.reconcile_facets(
                        companies=companies,
                        titles=titles,
                    )
                    facet_receipt = dict(
                        self.deps.browser.last_facet_reconciliation
                    )
                    if redundant_locations:
                        facet_receipt["redundant_locations_omitted"] = (
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
                        receipt=receipt,
                    )
                    raise
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
                    # Structured-only drops the keyword at compile time; the plan
                    # carries no keyword to apply or re-add during recovery.
                    include_keyword=(variant.surface != "structured_only"),
                )
                if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
                    # Only session-identical redundant values can reach this
                    # filter — the F8 ownership gate above rejected any
                    # conflicting per-string location before browser work.
                    plan.controls = [
                        control
                        for control in plan.controls
                        if control.dimension != "locations"
                    ]
                    # Companies and titles were typed by the reconcile above,
                    # one value per call; the controls stay on the plan for the
                    # recovery snapshot, their outcome comes from its receipt.
                    plan.reconciled_facet_misses = facet_misses(facet_receipt or {})
                apply_result = await self.deps.browser.apply_advanced_search_plan(plan)
                if not config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
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
                # A dropped structured control with a keyword or partial structured
                # set is a partial result, whether unsupported or verification-failed;
                # never report it as full success.
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
                    if mutation_kind == "drift":
                        experiment_state.rollback_pending_drift()
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
                    if mutation_kind == "drift":
                        experiment_state.rollback_pending_drift()
                    return SearchMutationResult(
                        applied=False,
                        blocked_reason="structured_controls_not_applied",
                        structured_apply=structured_apply,
                        hybrid_partial=hybrid_partial,
                    )
                # Surface-aware fallback. We are past the two abandon guards, so SOMETHING
                # landed (applied_controls is non-empty). When a requested structured dim
                # dropped here, the response depends on the lane surface:
                #   - structured_only NEVER demotes to keyword. It proceeds on whatever
                #     structured controls DID land (the re-entry below takes its synthetic,
                #     no-keyword path); the all-structured-dropped case already abandoned
                #     in the guards above. So no demotion is recorded for it.
                #   - boolean / filter_led / hybrid DEMOTE-AND-PROCEED: the keyword (or a
                #     partial structured set) carries the search. Emit a
                #     linkedin_structured_demotion event consuming the gate reason, CLEAR
                #     the dropped dims off variant.structured_filters so the next probe and
                #     the recovery snapshot agree on what actually landed, and tick the
                #     deterministic circuit-breaker counter.
                if (
                    company_scope is not None
                    and "companies" in dropped_non_keyword
                    and "companies" not in apply_result.applied_controls
                    and variant.surface != "structured_only"
                ):
                    # An allowlisted campaign never demotes its competitor
                    # facet to keyword: the variant is refused and keeps its
                    # companies, so the string stays on its committed surface.
                    self._record_event(
                        search_string=search_string,
                        event_type="linkedin_search_mutation_rejected",
                        payload={
                            "reason": "competitor_facet_not_applied",
                            "variant_id": variant.variant_id,
                            "structured_apply": structured_apply,
                        },
                    )
                    log_event(
                        self.deps.log_path,
                        "competitor_facet_demote_refused",
                        string_id=search_string.id,
                        variant_id=variant.variant_id,
                        refused=["demote_and_proceed:companies"],
                    )
                    print(
                        f"    [scope] String #{search_string.id}: variant "
                        f"{variant.variant_id} refused — its competitor facet did not "
                        "apply, and the facet is never demoted to keyword"
                    )
                    if mutation_kind == "drift":
                        experiment_state.rollback_pending_drift()
                    return SearchMutationResult(
                        applied=False,
                        blocked_reason="competitor_facet_not_applied",
                        structured_apply=structured_apply,
                        hybrid_partial=hybrid_partial,
                    )
                if (
                    company_scope is not None
                    and "job_titles" in dropped_non_keyword
                    and "job_titles" not in apply_result.applied_controls
                    and any(
                        company_scope.is_restricted(company_scope.canonical_for(company))
                        for company in variant.structured_filters.companies or []
                    )
                ):
                    # A client-side string (a title-restricted employer on the
                    # facet) never runs on its company facet alone: that would
                    # page through the client's engineers. Refused like the
                    # competitor facet above; the titles stay on the variant.
                    self._record_event(
                        search_string=search_string,
                        event_type="linkedin_search_mutation_rejected",
                        payload={
                            "reason": "client_title_facet_not_applied",
                            "variant_id": variant.variant_id,
                            "structured_apply": structured_apply,
                        },
                    )
                    log_event(
                        self.deps.log_path,
                        "competitor_facet_demote_refused",
                        string_id=search_string.id,
                        variant_id=variant.variant_id,
                        refused=["demote_and_proceed:titles"],
                    )
                    print(
                        f"    [scope] String #{search_string.id}: variant "
                        f"{variant.variant_id} refused — its client title facet did not "
                        "apply, and a client string never runs without it"
                    )
                    if mutation_kind == "drift":
                        experiment_state.rollback_pending_drift()
                    return SearchMutationResult(
                        applied=False,
                        blocked_reason="client_title_facet_not_applied",
                        structured_apply=structured_apply,
                        hybrid_partial=hybrid_partial,
                    )
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
                # This re-entry reads variant.boolean DIRECTLY, so the compile-time
                # keyword suppression does not reach it — gate it on the surface. A
                # structured_only variant must NOT re-enter the keyword here; its
                # structured controls already landed (the all-unsupported case rejected
                # above), so it takes the synthetic SearchEntryResult path with no
                # keyword search.
                if (
                    variant.surface != "structured_only"
                    and "keywords" not in apply_result.applied_controls
                ):
                    entry_result = await self.deps.browser.enter_search_string(variant.boolean)
                    if not config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
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
                if not config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
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
            if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
                # Verify what LANDED: a value the typeahead never offered
                # verbatim was recorded on the reconcile receipt, not applied.
                try:
                    await self.deps.browser.reconcile_facets(
                        companies=facet_target_landed(
                            companies, facet_receipt or {}, "companies"
                        ),
                        titles=facet_target_landed(
                            titles, facet_receipt or {}, "titles"
                        ),
                        verify_only=True,
                    )
                except FacetReconciliationError as exc:
                    receipt = self._facet_failure_receipt(
                        companies=companies,
                        titles=titles,
                        gate="postcondition_mismatch",
                        error=exc,
                        before=facet_receipt,
                    )
                    self._record_facet_reconciliation(
                        search_string=search_string,
                        receipt=receipt,
                    )
                    raise
                receipt = dict(facet_receipt or {})
                receipt["after"] = dict(
                    self.deps.browser.last_facet_reconciliation.get("after", {})
                )
                receipt["failure_gate"] = ""
                self._record_facet_reconciliation(
                    search_string=search_string,
                    receipt=receipt,
                )
                log_event(
                    self.deps.log_path,
                    "string_executed",
                    string_id=search_string.id,
                    executed_boolean=variant.boolean,
                    execution_surface="advanced" if used_hybrid else "keyword",
                )
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

    def _keep_competitor_facet(
        self,
        *,
        search_string: "SearchString",
        variant: "LinkedInSearchVariant",
        company_scope: Any,
        acquisition_mode: str,
    ) -> str:
        """Give a variant with no listed competitor the string's competitor facet.

        A rescue that dropped a filter, a forced-narrow fallback, or a keyword
        variant can each reach here with an empty company facet; under the
        brief's allowlist it runs inside the string's stamped companies
        instead. Returns the acquisition mode the variant now needs.
        """

        from linkedin.competitor_scope import scoped_facet_companies

        if scoped_facet_companies(variant.structured_filters.companies or [], company_scope):
            return acquisition_mode
        root = (search_string.structured_filters or {}).get("companies") or []
        restored = scoped_facet_companies(root, company_scope)
        if not restored:
            return acquisition_mode
        variant.structured_filters.companies = list(restored)
        if variant.surface in ("", "boolean"):
            variant.surface = "hybrid"
        log_event(
            self.deps.log_path,
            "competitor_facet_restored",
            string_id=search_string.id,
            variant_id=variant.variant_id,
            companies=list(restored),
        )
        print(
            f"    [scope] String #{search_string.id}: variant {variant.variant_id} "
            f"keeps the competitor facet ({', '.join(restored)})"
        )
        if search_string.acquisition_mode != "linkedin_hybrid":
            search_string.acquisition_mode = "linkedin_hybrid"
        return "linkedin_hybrid"

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

        The demote-and-proceed path fell back to keyword because a structured
        control dropped. Stripping the dropped dim off the variant means the next
        probe re-compiles WITHOUT it and the applied-only recovery snapshot agrees
        on what actually landed — neither replays a control the sidebar never
        accepted.

        ``dropped_dimensions`` are PLAN dimensions (the names
        ``compile_structured_filters_to_plan`` emits: ``job_titles`` /
        ``companies`` / ``locations`` / ``fields_of_study``); map each back to its
        structured_filters home before clearing.

        A PARTIAL demote (one of N dims dropped, surface stays hybrid, surviving
        filters non-empty) satisfies neither apply_shadow's full-demote
        checkpoint-clear (which needs ``structured_filters.is_empty()``) nor
        ``is_deliberate_boolean_demotion``. Left alone, the checkpointed
        ``search_string.structured_filters`` keeps the dropped dim, and a
        CROSS-PROCESS resume's ``bootstrap_experiment_state`` re-seeds it onto a
        fresh variant — re-applying a control the sidebar already rejected. Passing
        the live ``checkpoint`` dict (``search_string.structured_filters``) strips
        the SAME dropped dims from it in lockstep, so the post-demote checkpoint
        reflects only what landed and the resume cannot re-seed the dropped dim. The
        dim→home mapping is shared with the variant clear above so the two never
        diverge.
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
