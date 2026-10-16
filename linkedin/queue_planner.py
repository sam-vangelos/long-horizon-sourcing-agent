"""QueuePlanner implementation with explicit concrete collaborators."""
from __future__ import annotations
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from shared.schemas import SearchString, Progress, KitString, BlockReport, AdaptationResponse
from linkedin.search_intelligence import LinkedInStructuredFilters
from linkedin.fallback_acquisition import (
    discover_fallback_candidates_for_string,
    fallback_mode_for_search_string,
    record_fallback_discovery,
)
from linkedin import lane_projection as _lane_projection
from linkedin import surface_receipt as _surface_receipt
from linkedin.boolean_compiler import (
    BooleanNormalizationError,
    UbiquitousAndGateError,
    boolean_lint_context_from_brief,
    normalize_execution_work_item_boolean,
    proper_nouns_from_brief,
    ubiquitous_terms_from_brief,
)
from linkedin.adaptation_signal_state import (
    AdaptationGateConfig,
    AdaptationGateDecision,
    SearchSignalState,
    evaluate_adaptation_gate,
)
from shared.storage import append_jsonl, log_event, write_json
from shared.search_memory import build_search_memory_summary
from shared.strict_seniority import profile_reads_above_band
from shared.sourcing_lanes import (
    apply_lane_fields_to_search_string,
    lane_fields_from_work_unit_item,
)
from linkedin.block_adaptation import BlockAdaptationService
from linkedin.competitor_scope import stamp_competitor_facets
from shared.employer_allowlist import employer_allowlist_from_brief
from shared import config


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linkedin.evaluation_context import EvaluationContext
    from linkedin.candidate_ledger import CandidateLedger
    from linkedin.allocator_state import AllocatorStateService
    from shared.schemas import ExecutionPlan
    from linkedin.fallback_search import FallbackSearchProvider


class QueuePlanner:
    """Queue construction, block adaptation and search-memory ownership."""

    _execution_plan: ExecutionPlan | None
    _kit_strings: list[KitString]
    _lint_blocked_strings: list[dict[str, Any]]
    _search_memory: dict[str, Any]
    _fallback_provider: FallbackSearchProvider | None
    _fallback_candidates_path: Path
    _block_adaptation_service: BlockAdaptationService | None
    _adaptation_gate_config: AdaptationGateConfig

    def __init__(
        self,
        context: EvaluationContext,
        ledger: CandidateLedger,
        allocator: AllocatorStateService,
    ) -> None:
        self.context, self.ledger = context, ledger
        self.allocator = allocator
        self.events = None
        self.reports = None
        self._block_adaptation_service = None
        self.brief_path: str
        self.state_dir: Path
        self.noise_path: Path

    def _load_search_memory(self) -> None:
        """Load brief-scoped search family memory if present."""
        self._search_memory = self.allocator.work_units.load_search_memory()


    def _save_search_memory(self) -> None:
        """Persist brief-scoped search family memory."""
        self._search_memory = self.allocator.work_units.save_search_memory()


    def _update_search_memory_from_block(self, block_strings: list[SearchString]) -> None:
        """Delegates to BlockAdaptationService."""
        self._block_adaptation_service._update_search_memory_from_block(block_strings)


    def _hydrate_search_string_metadata(self, search_string: SearchString) -> None:
        """Delegates to BlockAdaptationService."""
        self._block_adaptation_service._hydrate_search_string_metadata(search_string)


    def _set_search_memory(self, search_memory: dict) -> None:
        self._search_memory = search_memory


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
                self.context.log_path,
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


    def _set_pending_block_adaptation(
        self,
        progress: Progress | None,
        block_name: str,
        block_strings: list[SearchString],
        *,
        ready: bool,
    ) -> None:
        """Persist the current block context across resume boundaries."""
        self.allocator.work_units.set_pending_block_adaptation(
            progress,
            block_name,
            block_strings,
            ready=ready,
        )


    def _clear_pending_block_adaptation(self, progress: Progress | None) -> None:
        """Delegates to BlockAdaptationService."""
        self._block_adaptation_service._clear_pending_block_adaptation(progress)


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
                ubiquitous_terms=ubiquitous_terms_from_brief(self.context.brief_obj),
                expand_surface_variants=(
                    config.LINKEDIN_SURFACE_VARIANT_EXPANSION_ENABLED
                ),
                proper_nouns=proper_nouns_from_brief(self.context.brief_obj),
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
            lint_context=boolean_lint_context_from_brief(self.context.brief_obj),
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
            precision_class=item.get("precision_class", ""),
            evidence_implication=item.get("evidence_implication", ""),
            **lane_fields,
            surface=item["surface"],
            boolean_lint=lint_payload,
        )
        self._hydrate_search_string_metadata(ss)
        apply_lane_fields_to_search_string(ss)
        return ss


    def _enforce_competitor_facets(
        self,
        strings: list[SearchString],
        *,
        queue: list[SearchString],
    ) -> list[dict[str, Any]]:
        """Stamp a listed competitor facet onto strings that lack one.

        Inert unless the brief carries an ``employer_allowlist``
        (linkedin/competitor_scope.py). One console line and one run-log event
        per string the stamp changed.
        """

        allowlist = employer_allowlist_from_brief(self.context.brief_obj)
        records = stamp_competitor_facets(strings, allowlist, queue=queue)
        for record in records:
            print(
                f"    [scope] String #{record['string_id']}: competitor facet "
                f"{record['action']} -> {', '.join(record['after'])}"
            )
            log_path = getattr(self.context, "log_path", None)
            if log_path:
                log_event(log_path, "competitor_facet_stamped", **record)
        return records


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

        lane_records = _lane_projection.structured_lane_projection_records(self._execution_plan)
        represented_lane_ids: set[str] = set()

        # Error-severity lint findings block queueing per string; healthy siblings
        # still run. The ubiquity gate uses the brief blacklist and structural terms.
        lint_context = boolean_lint_context_from_brief(self.context.brief_obj)
        ubiquitous_terms = ubiquitous_terms_from_brief(self.context.brief_obj)
        # Do-not-vary set for deterministic surface expansion: brief-declared
        # named artifacts keep their one correct spelling.
        proper_noun_terms = proper_nouns_from_brief(self.context.brief_obj)

        # --- Generated compound strings (in priority order from Opus) ---
        compound_idx = 0
        for gs in self._execution_plan.generated_strings:
            if not isinstance(gs, dict):
                continue
            lane_record = _lane_projection.match_lane_projection(gs, lane_records)
            work_item = (
                _lane_projection.apply_lane_projection_to_work_item(
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
                precision_class=work_item.get("precision_class", ""),
                evidence_implication=work_item.get("evidence_implication", ""),
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
            lane_record = _lane_projection.match_lane_projection(gap, lane_records)
            work_item = (
                _lane_projection.apply_lane_projection_to_work_item(
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
                precision_class=work_item.get("precision_class", ""),
                evidence_implication=work_item.get("evidence_implication", ""),
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

        # An allowlisted campaign bounds every string by a competitor facet,
        # stamped here so the plan, progress.json, and the receipt below all
        # show the facet each string will run with.
        self._enforce_competitor_facets(ordered, queue=ordered)

        # Surface receipt (intended): make structured-filter materialization visible at
        # queue time instead of eyeballed across the Boolean list. Fail-soft — the
        # receipt must never break queue building.
        try:
            summary = _surface_receipt.summarize_intended_surfaces(ordered)
            print(_surface_receipt.format_intended_summary(summary))
            if getattr(self.context, "log_path", None):
                log_event(
                    self.context.log_path,
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
        return self.reports._load_profile_index_for_adaptation()


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
        return self.reports._search_intelligence_detail_for_string(search_string)


    def _search_intelligence_aggregate(
        self,
        strings: list[SearchString],
        *,
        profile_index: dict[str, dict] | None = None,
    ) -> dict[str, Any]:
        return self.reports._search_intelligence_aggregate(
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
            state = self.allocator.experiment(search_string.id)
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
        # Plan-level deterministic warnings (lane collapse) ride
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
                    self.context._bias_monitor.expected_facial_yes_low,
                    self.context._bias_monitor.expected_facial_yes_high,
                ]
                if self.context._bias_monitor
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
                    # Keep-but-flag markers ride the surfaces adaptation reads, or
                    # lane drift looks first-class mid-run.
                    "domain_lane_raw": s.domain_lane_raw,
                    "undeclared_lane": s.undeclared_lane,
                    "notes": s.notes,
                    # Per-string bias context lets the adaptation model weigh
                    # "dense vein vs loosened judge" itself. Monitor keys are
                    # str(source_string_id); s.id is int — the coercion is
                    # load-bearing.
                    "bias": (
                        self.context._bias_monitor.string_context(str(s.id))
                        if self.context._bias_monitor
                        else None
                    ),
                    "surface_receipt": s.surface_receipt,
                    "search_intelligence": self._search_intelligence_detail_for_string(s),
                    # Craft health: warning-severity lint findings ride into the
                    # block report so adaptation can see them (errors never reach
                    # execution; they were blocked at queue build).
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
            self.allocator._checkpoint_progress(progress)
            print("  No remaining strings — skipping adaptation.")
            return

        signal_state = SearchSignalState.from_block_report(report)
        gate_config = getattr(self, "_adaptation_gate_config", None)
        if not isinstance(gate_config, AdaptationGateConfig):
            # The SPRT/cooldown gate is configurable from shared.config
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
            log_event(self.context.log_path, "adaptation_decision", **payload)
            self.events._record_runtime_event(
                search_string=None,
                event_type="adaptation_decision",
                payload=payload,
            )
            self._clear_pending_block_adaptation(progress)
            self.allocator._checkpoint_progress(progress)
            return

        market_intel_advisory_context = ""
        try:
            from market_intelligence.live_advisory import (
                record_block_checkpoint_and_get_context,
            )

            market_intel_advisory_context = record_block_checkpoint_and_get_context(
                brief_path=self.brief_path,
                state_dir=self.state_dir,
                brief_id=self.context.brief_obj.linkedin_project_id
                or self.context.brief_obj.id
                or Path(self.brief_path).stem,
                block_name=block_name,
                block_report=report,
                search_memory_summary=build_search_memory_summary(self._search_memory),
            )
        except Exception:
            market_intel_advisory_context = ""

        try:
            adaptation = adapt_fn(
                self.context.brief_obj, report, remaining,
                kit_vocabulary=self._kit_strings,
                execution_plan=self._execution_plan,
                pivot_count=progress.pivot_count,
                search_memory_summary=build_search_memory_summary(self._search_memory),
                checkpoint_mode=checkpoint_mode,
                market_intel_advisory_context=market_intel_advisory_context,
                # Where an unparseable adaptation reply is saved for diagnosis.
                state_dir=getattr(self, "state_dir", None),
            )

            # Ubiquity-gate drops from the adapted-string firewall are
            # per-string refusals, recorded like every other queue-time block
            # so the run report sees them.
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
                # An explicit decline is a valid, logged decision —
                # not a failure. Deterministic: no strings inserted, no
                # skips/reorders/noise updates/pivot applied, queue order
                # untouched. Short-circuits before exploitation bias and the
                # rest of the "Apply adaptations" section, which would
                # otherwise be no-ops anyway (adapt_after_block already
                # rebuilt a clean, empty AdaptationResponse for this case).
                print("  [adapt] Model declined to adapt (no_change) — queue untouched.")
                self._clear_pending_block_adaptation(progress)
                self.allocator._checkpoint_progress(progress)
                log_event(
                    self.context.log_path,
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
                log_event(self.context.log_path, "linkedin_block_exploitation", **payload)
                self.events._record_runtime_event(
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
            # Hoisted (not just declared inside the if-block) so the
            # block_adaptation log_event below can always reference it —
            # displaced_string_ids is the "strings this adaptation skipped"
            # side of the adaptation-ROI accounting.
            skip_ids: set[int] = set()
            if adaptation.skip_remaining:
                skip_ids = {s["string_id"] for s in adaptation.skip_remaining}
                for ss in progress.strings:
                    if ss.id in skip_ids and ss.status == "queued":
                        assert ss.id not in self.ledger.resumable_string_ids()
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
                adaptive_lint_context = boolean_lint_context_from_brief(self.context.brief_obj)
                for ns in adaptation.new_strings:
                    # Adaptive strings are the second queueing path and take the
                    # same craft-lint gate as the initial queue build. The
                    # ubiquity gate ran earlier, per string, inside
                    # apply_adapted_string_firewall; its drops are recorded below
                    # from the firewall trace.
                    lint_payload = self._queue_lint_gate(
                        ns,
                        boolean_key="boolean",
                        source="adaptive",
                        lint_context=adaptive_lint_context,
                    )
                    if lint_payload is None:
                        continue
                    max_id += 1
                    if "precision_class" in ns:
                        precision_class = ns.get("precision_class") or ""
                        evidence_implication = ns.get("evidence_implication") or ""
                    else:
                        family_key = str(ns.get("family_key") or "")
                        # Adaptive siblings share the retrieval vein, so the
                        # parent's membership implication holds.
                        precision_parent = next(
                            (
                                search_string
                                for search_string in progress.strings
                                if family_key
                                and search_string.family_key == family_key
                                and search_string.precision_class == "sniper"
                            ),
                            None,
                        )
                        precision_class = (
                            precision_parent.precision_class if precision_parent else ""
                        )
                        evidence_implication = (
                            precision_parent.evidence_implication
                            if precision_parent
                            else ""
                        )
                    new_ss = SearchString(
                        id=max_id,
                        name=f"Adaptive / {ns.get('rationale', 'new')}",
                        boolean=ns["boolean"],
                        block=block_name,
                        string_type="Adaptive",
                        boolean_lint=lint_payload,
                        family_key=ns.get("family_key", ""),
                        precision_class=precision_class,
                        evidence_implication=evidence_implication,
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
                    # Adaptation strings come from a prompt that may leave the
                    # facet off; an allowlisted campaign stamps one here.
                    self._enforce_competitor_facets(
                        [new_ss], queue=[*progress.strings, new_ss]
                    )
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
                        self.context.log_path,
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
                        write_json(self.context.output_dir / "execution_plan.json", self._execution_plan.to_dict())

                    # Clear remaining queued strings but keep the replacement strings
                    # we just injected for the new architecture.
                    for ss in progress.strings:
                        if ss.status == "queued" and ss.id not in inserted_ids:
                            assert ss.id not in self.ledger.resumable_string_ids()
                            ss.status = "skipped"
                            ss.notes = f"Skipped by architecture pivot: {old_arch} → {new_arch}"

                    progress.pivot_count += 1
                    log_event(self.context.log_path, "architecture_pivot",
                              old=old_arch, new=new_arch,
                              rationale=adaptation.pivot_rationale, block=block_name)
                else:
                    print(f"  [adapt] Pivot to {adaptation.pivot_to_architecture} recommended "
                          f"but max pivots ({max_pivots}) reached — ignoring.")
                    log_event(self.context.log_path, "pivot_blocked", reason="max_pivots_reached",
                              recommended=adaptation.pivot_to_architecture)

            # Apply the model-requested adaptation cadence (already
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
                    self.context.log_path,
                    "adaptation_checkpoint_cadence",
                    block=block_name,
                    requested=requested_checkpoint,
                    applied=applied_checkpoint,
                    relabeled_string_ids=[ss.id for ss in upcoming],
                )

            self._clear_pending_block_adaptation(progress)
            self.allocator._checkpoint_progress(progress)
            log_event(
                self.context.log_path,
                "block_adaptation",
                block=block_name,
                report=report.to_dict(),
                # Adaptation ROI accounting reads these back at report
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
            log_event(self.context.log_path, "adaptation_error", block=block_name, error=str(e), traceback=traceback.format_exc())


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
                self.context.brief_obj._new_brief
                if getattr(self.context.brief_obj, "has_v2_schema", False)
                else self.context.brief_obj,
                research_call=research_call,
                artifact_dir=self.context.output_dir,
            )
        except Exception as exc:  # noqa: BLE001 — enrichment never breaks a run
            print(f"  [warn] vocabulary enumeration unavailable: {exc}")
            return []

        if kit:
            terms = sum(ks.boolean.count(" OR ") + 1 for ks in kit)
            print(
                f"  Enumerated {terms} named artifacts into "
                f"{len(kit)} vocabulary groups → {self.context.output_dir / 'vocabulary_enumeration.json'}"
            )
        return kit
