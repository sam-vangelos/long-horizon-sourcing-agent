"""Page allocator evidence, canonical checkpoints and cursor publication.

The ledger supplies a frozen pending-owner snapshot for ordinary checkpoints.
Work-unit publication and candidate attempt writes retain their existing owners.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING
import logging
import time
from linkedin.timing_telemetry import emit_timing_event
from shared.storage import log_event
from linkedin.page_review import PageReviewContext

if TYPE_CHECKING:
    from linkedin.evaluation_context import EvaluationContext
    from linkedin.candidate_ledger import CandidateLedger
    from linkedin.work_units import LinkedInWorkUnitService
    from linkedin.runtime_attempts import RuntimeAttemptService

logger = logging.getLogger("linkedin.orchestrator")

from shared.schemas import OpusDecision, Progress, SearchString
from linkedin.page_review import terminal_full_decision

from linkedin.page_allocator import (
    AllocationAction,
    AllocationVerdict,
    AllocatorArm,
    AllocatorPolicyError,
    PageObservation,
    allocate_page,
)
from linkedin.search_intelligence import LinkedInExperimentState, bootstrap_experiment_state
from linkedin.competitor_scope import stamp_competitor_facet_for_brief
from shared import config


class AllocatorStateService:
    """Owns allocator decisions, rollback and publication of proposed cursors."""

    def __init__(
        self,
        *,
        context: EvaluationContext,
        candidates: CandidateLedger,
        events: RuntimeAttemptService,
        get_experiment_states: Callable[[], dict[int, LinkedInExperimentState]],
    ) -> None:
        self.context = context
        self.candidates = candidates
        self.events = events
        self._get_experiment_states = get_experiment_states
        # Bound by EvaluationLoop._ensure_services after runtime initialization.
        self.work_units: LinkedInWorkUnitService | None = None
        self._pending_allocator_checkpoint: dict[str, Any] | None = None
        self._pending_allocator_poison: dict[str, Any] | None = None
        self._incomplete_page_rollbacks: dict[int, dict[str, Any]] = {}
        self._active_allocator_validated_segments: set[str] = set()

    @staticmethod
    def _allocator_terminal_full_decision(decision: OpusDecision) -> bool:
        return terminal_full_decision(decision)

    @staticmethod
    def _allocator_shadow_enabled() -> bool:
        return config.LINKEDIN_PAGE_ALLOCATOR_MODE == "shadow"

    @staticmethod
    def _allocator_active_enabled() -> bool:
        return config.LINKEDIN_PAGE_ALLOCATOR_MODE == "active"

    @staticmethod
    def _allocator_tracking_enabled() -> bool:
        return config.LINKEDIN_PAGE_ALLOCATOR_MODE in {"active", "shadow"}

    @staticmethod
    def _allocator_verdict_requires_actuation(
        verdict: AllocationVerdict,
    ) -> bool:
        return (
            verdict.action is not AllocationAction.CONTINUE
            or bool(verdict.floored_root_ids)
        )

    def _allocator_run_diverged(self) -> bool:
        return any(
            bool(getattr(state, "allocator_shadow_diverged", False))
            for state in self._get_experiment_states().values()
        )

    @staticmethod
    def _allocator_terminal_status(status: str) -> bool:
        return status in {"done", "skipped", "error"}

    def _allocator_contiguous_segment(
        self,
        progress: Progress,
        current: SearchString,
    ) -> list[tuple[int, SearchString]]:
        current_index = next(
            (
                index
                for index, item in enumerate(progress.strings)
                if item is current or item.id == current.id
            ),
            -1,
        )
        if current_index < 0:
            raise AllocatorPolicyError("current root is missing from progress")
        start = current_index
        while (
            start > 0
            and progress.strings[start - 1].block == current.block
        ):
            start -= 1
        stop = current_index + 1
        while (
            stop < len(progress.strings)
            and progress.strings[stop].block == current.block
        ):
            stop += 1
        return list(enumerate(progress.strings[start:stop], start=start))

    def _allocator_segment_identity(
        self,
        progress: Progress,
        current: SearchString,
    ) -> dict[str, Any]:
        segment = self._allocator_contiguous_segment(progress, current)
        indexes = [index for index, _item in segment]
        ids = [item.id for _index, item in segment]
        return {
            "block": current.block,
            "start_index": indexes[0],
            "stop_index": indexes[-1] + 1,
            "root_ids": ids,
            "key": f"{current.block}\x1f{indexes[0]}:{indexes[-1] + 1}\x1f"
            + ",".join(str(root_id) for root_id in ids),
        }

    def _bootstrap_registered_state(
        self,
        search_string: SearchString,
        progress: Progress,
    ) -> LinkedInExperimentState:
        """Bootstrap and register a root the allocator selects before it runs.

        The root variant freezes the string's facet, so an allowlisted
        campaign stamps a competitor facet first, as the driver does
        (linkedin/competitor_scope.py; inert without an employer_allowlist).
        """

        stamp_competitor_facet_for_brief(
            search_string,
            getattr(self.context, "brief_obj", None),
            queue=list(progress.strings),
            log_path=getattr(self.context, "log_path", None),
        )
        state = bootstrap_experiment_state(search_string)
        self._get_experiment_states()[search_string.id] = state
        return state

    def _allocator_state_for_arm(
        self,
        search_string: SearchString,
    ) -> LinkedInExperimentState:
        state = self._get_experiment_states().get(search_string.id)
        return state if state is not None else bootstrap_experiment_state(search_string)

    def _allocator_arms(
        self,
        *,
        progress: Progress,
        current: SearchString,
        prospective_observation: PageObservation | None = None,
        exhausted_root_id: int | None = None,
    ) -> list[AllocatorArm]:
        segment_identity = self._allocator_segment_identity(progress, current)
        arms: list[AllocatorArm] = []
        for queue_priority, item in self._allocator_contiguous_segment(
            progress, current
        ):
            state = self._allocator_state_for_arm(item)
            variant = state.active_variant
            observations = list(
                getattr(variant, "allocator_observations", []) or []
            )
            valid_page_count = int(
                getattr(variant, "allocator_valid_page_count", 0) or 0
            )
            root_has_valid_probe = any(
                int(getattr(candidate, "allocator_valid_page_count", 0) or 0)
                > 0
                for candidate in state.variants.values()
            )
            if (
                prospective_observation is not None
                and item.id == prospective_observation.root_string_id
                and variant.variant_id == prospective_observation.variant_id
            ):
                observations.append(prospective_observation)
                if prospective_observation.teaches_policy:
                    valid_page_count += 1
                    root_has_valid_probe = True
            completed_count = int(
                getattr(
                    variant,
                    "allocator_completed_observation_count",
                    0,
                )
                or 0
            )
            arms.append(
                AllocatorArm(
                    root_string_id=item.id,
                    block=str(segment_identity["key"]),
                    queue_priority=queue_priority,
                    active_variant_id=variant.variant_id,
                    observations=tuple(observations),
                    active_valid_page_count=valid_page_count,
                    root_has_valid_probe=root_has_valid_probe,
                    legacy_unobserved_pages=max(
                        0,
                        max(
                            int(variant.pages_reviewed),
                            max(
                                0,
                                int(item.pages_reviewed)
                                - int(
                                    prospective_observation is not None
                                    and item.id
                                    == prospective_observation.root_string_id
                                ),
                            ),
                        )
                        - completed_count,
                    ),
                    physically_exhausted=item.id == exhausted_root_id,
                    terminal=self._allocator_terminal_status(item.status),
                )
            )
        return arms

    @staticmethod
    def _allocator_post_verdict_order(
        root_ids: list[int],
        verdict: AllocationVerdict,
    ) -> list[int]:
        """Project the selected-first/paused-tail order without mutation."""

        ordered = list(root_ids)
        selected = verdict.selected_root_id
        if (
            verdict.action is AllocationAction.SWITCH
            and selected is not None
            and selected in ordered
            and verdict.current_root_id in ordered
        ):
            ordered.remove(selected)
            selected_index = ordered.index(verdict.current_root_id)
            ordered.insert(selected_index, selected)
            for paused_root_id in verdict.paused_root_ids:
                if paused_root_id in ordered:
                    ordered.remove(paused_root_id)
                    ordered.append(paused_root_id)
        return ordered

    def _allocator_expected_statuses(
        self,
        *,
        progress: Progress,
        current: SearchString,
        arms: list[AllocatorArm],
        verdict: AllocationVerdict,
    ) -> dict[str, str]:
        arm_by_id = {arm.root_string_id: arm for arm in arms}
        statuses: dict[str, str] = {}
        for _index, item in self._allocator_contiguous_segment(
            progress, current
        ):
            arm = arm_by_id[item.id]
            statuses[str(item.id)] = (
                "terminal" if arm.terminal else "queued"
            )
        if verdict.selected_root_id is not None:
            statuses[str(verdict.selected_root_id)] = "in_progress"
        for paused_root_id in verdict.paused_root_ids:
            statuses[str(paused_root_id)] = "queued"
        for floored_root_id in verdict.floored_root_ids:
            statuses[str(floored_root_id)] = "terminal"
        if verdict.action is AllocationAction.FINISH:
            statuses[str(verdict.current_root_id)] = "terminal"
        elif (
            verdict.action is AllocationAction.SWITCH
            and verdict.current_root_id not in verdict.paused_root_ids
        ):
            statuses[str(verdict.current_root_id)] = "terminal"
        return statuses

    def _allocator_expectation(
        self,
        *,
        progress: Progress,
        current: SearchString,
        arms: list[AllocatorArm],
        verdict: AllocationVerdict,
        sequence: int,
    ) -> dict[str, Any]:
        segment = self._allocator_segment_identity(progress, current)
        pre_root_ids = [int(root_id) for root_id in segment["root_ids"]]
        expected_order = self._allocator_post_verdict_order(
            pre_root_ids,
            verdict,
        )
        expected_statuses = self._allocator_expected_statuses(
            progress=progress,
            current=current,
            arms=arms,
            verdict=verdict,
        )
        segment = {**segment, "root_ids": expected_order}
        return {
            "sequence": int(sequence),
            "segment": segment,
            "pre_root_ids": pre_root_ids,
            "pre_status_by_root": {
                str(item.id): item.status
                for _index, item in self._allocator_contiguous_segment(
                    progress, current
                )
            },
            "pre_current_string_id": progress.current_string_id,
            "pre_current_page": int(progress.current_page or 0),
            "action": verdict.action.value,
            "selected_root_id": verdict.selected_root_id,
            "expected_paused_root_ids": list(verdict.paused_root_ids),
            "expected_live_root_ids": [
                root_id
                for root_id in expected_order
                if expected_statuses.get(str(root_id)) != "terminal"
            ],
            "expected_status_by_root": expected_statuses,
            "floored_root_ids": list(verdict.floored_root_ids),
            "allow_new_segment": verdict.action
            in {AllocationAction.FINISH, AllocationAction.FLOOR},
        }

    def _allocator_frontier_alignment(
        self,
        *,
        progress: Progress,
        current: SearchString,
        expectation: dict[str, Any],
        require_selected: bool,
    ) -> tuple[bool, str]:
        segment = self._allocator_segment_identity(progress, current)
        expected_segment = expectation.get("segment")
        if not isinstance(expected_segment, dict):
            return False, "malformed_frontier_expectation"
        same_boundary = (
            segment.get("block") == expected_segment.get("block")
            and segment.get("start_index")
            == expected_segment.get("start_index")
            and segment.get("stop_index") == expected_segment.get("stop_index")
        )
        expected_root_ids = [
            int(root_id) for root_id in expected_segment.get("root_ids", [])
        ]
        if (
            not same_boundary
            or segment.get("root_ids") != expected_root_ids
        ):
            if expectation.get("allow_new_segment"):
                return True, "completed_segment_transition"
            return False, "segment_signature_changed"
        expected_statuses = expectation.get("expected_status_by_root")
        if not isinstance(expected_statuses, dict):
            return False, "malformed_frontier_expectation"
        for _index, item in self._allocator_contiguous_segment(
            progress, current
        ):
            expected_status = expected_statuses.get(str(item.id))
            if expected_status == "terminal":
                aligned_status = self._allocator_terminal_status(item.status)
            else:
                aligned_status = item.status == expected_status
            if not aligned_status:
                return False, "frontier_disposition_changed"
        if require_selected:
            selected = expectation.get("selected_root_id")
            if selected is None or int(selected) != current.id:
                return False, "selected_root_mismatch"
        return True, "aligned"

    def _allocator_checkpoint_ready(self) -> dict[str, Any] | None:
        pending = self._pending_allocator_checkpoint
        if not isinstance(pending, dict):
            return None
        if pending.get("kind") in {"dispatch", "exhaustion"}:
            return pending
        state = self._get_experiment_states().get(
            int(pending.get("root_string_id", 0) or 0)
        )
        if state is None:
            return None
        variant = state.variants.get(str(pending.get("variant_id", "")))
        if variant is None:
            return None
        return (
            pending
            if int(variant.allocator_page_cursor) > int(pending.get("page", 0))
            else None
        )

    @staticmethod
    def _allocator_verdict_from_payload(
        payload: dict[str, Any],
    ) -> AllocationVerdict:
        """Rehydrate the control fields needed to replay a durable verdict."""

        try:
            action = AllocationAction(str(payload["action"]))
            current_root_id = int(payload["current_root_id"])
            selected_value = payload.get("selected_root_id")
            selected_root_id = (
                int(selected_value) if selected_value is not None else None
            )
            paused_root_ids = tuple(
                int(root_id) for root_id in payload.get("paused_root_ids", [])
            )
            floored_root_ids = tuple(
                int(root_id) for root_id in payload.get("floored_root_ids", [])
            )
            ranked_root_ids = tuple(
                int(root_id) for root_id in payload.get("ranked_root_ids", [])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AllocatorPolicyError("malformed durable allocator verdict") from exc
        if current_root_id <= 0 or (
            selected_root_id is not None and selected_root_id <= 0
        ):
            raise AllocatorPolicyError("durable allocator verdict has invalid roots")
        return AllocationVerdict(
            action=action,
            current_root_id=current_root_id,
            selected_root_id=selected_root_id,
            reason=str(payload.get("reason", "") or ""),
            paused_root_ids=paused_root_ids,
            floored_root_ids=floored_root_ids,
            ranked_root_ids=ranked_root_ids,
        )


    def _checkpoint_progress_impl(
        self,
        progress: Progress | None,
        search_string: SearchString | None = None,
        page_num: int | None = None,
        *,
        completed_page: bool = False,
        _timings: dict[str, float] | None = None,
        work_units: LinkedInWorkUnitService,
        resumable_string_ids: frozenset[int],
        cursor_updates: tuple[tuple[int, str, int], ...] = (),
    ) -> None:
        """Apply proposed positions only within the canonical checkpoint."""
        prior_cursors = {
            (root_id, variant_id): self._get_experiment_states()[root_id].variants[variant_id].allocator_page_cursor
            for root_id, variant_id, _next_page in cursor_updates
        }
        committed = False
        try:
            for root_id, variant_id, next_page in cursor_updates:
                self._get_experiment_states()[root_id].variants[variant_id].allocator_page_cursor = next_page
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
                    state_object = self._get_experiment_states().get(
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
            owner_ids = resumable_string_ids
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
                work_units.checkpoint_progress(
                    progress,
                    **checkpoint_kwargs,
                )
                committed = True
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

        finally:
            if not committed:
                for (root_id, variant_id), prior_page in prior_cursors.items():
                    # Allocator rollback may replace variant objects. Resolve by
                    # identity rather than writing through an obsolete alias.
                    self._get_experiment_states()[root_id].variants[variant_id].allocator_page_cursor = prior_page


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
        self._get_experiment_states()[search_string.id] = state_object

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


    def _poison_allocator_trace(
        self,
        *,
        search_string: SearchString | None,
        reason: str,
    ) -> None:
        if not self._allocator_shadow_enabled():
            return
        state = (
            self._get_experiment_states().get(search_string.id)
            if search_string is not None
            else None
        )
        if state is None and self._get_experiment_states():
            state = next(iter(self._get_experiment_states().values()))
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


    def _latest_allocator_frontier_expectation(self) -> dict[str, Any]:
        expectations = [
            getattr(state, "allocator_frontier_expectation", {})
            for state in self._get_experiment_states().values()
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
            for state in self._get_experiment_states().values()
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
        state = self._get_experiment_states().get(current.id)
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
        page_context: PageReviewContext | None = None,
    ) -> None:
        if not self._allocator_tracking_enabled():
            return
        try:
            if page_context is None:
                raise AllocatorPolicyError("page observation context missing")
            identity = page_context._allocator_page_identity
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
                        page_context._allocator_page_off_policy
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


    def bypasses_local_page_policy(self) -> bool:
        """Read completed-page authority after staging, before legacy metrics."""
        if not self._allocator_active_enabled():
            return False
        pending = self._pending_allocator_checkpoint
        if not isinstance(pending, dict) or not isinstance(
            pending.get("verdict"), AllocationVerdict
        ):
            raise AllocatorPolicyError(
                "active allocator did not stage a completed-page verdict"
            )
        verdict = pending["verdict"]
        return bool(
            verdict.action is not AllocationAction.CONTINUE
            or verdict.reason == "opening_probe"
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


    def _apply_pending_allocator_checkpoint(
        self,
        pending: dict[str, Any],
        *,
        progress: Progress | None,
    ) -> tuple[str, SearchString, dict[str, Any]]:
        state = self._get_experiment_states()[
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
            self.events._record_runtime_event(
                search_string=search_string,
                event_type=event_type,
                payload=payload,
            )
        except Exception:
            pass
        try:
            if event_type == "page_allocator_shadow_checkpoint":
                log_event(
                    self.context.log_path,
                    "page_allocator_shadow_checkpoint",
                    **payload,
                )
            elif event_type == "page_allocator_shadow_exhaustion":
                log_event(
                    self.context.log_path,
                    "page_allocator_shadow_exhaustion",
                    **payload,
                )
            elif event_type == "page_allocator_shadow_poison":
                log_event(
                    self.context.log_path,
                    "page_allocator_shadow_poison",
                    **payload,
                )
            elif event_type == "page_allocator_active_checkpoint":
                log_event(
                    self.context.log_path,
                    "page_allocator_active_checkpoint",
                    **payload,
                )
            elif event_type == "page_allocator_active_exhaustion":
                log_event(
                    self.context.log_path,
                    "page_allocator_active_exhaustion",
                    **payload,
                )
            elif event_type == "page_allocator_active_actuation":
                log_event(
                    self.context.log_path,
                    "page_allocator_active_actuation",
                    **payload,
                )
        except Exception:
            pass


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
        for state in self._get_experiment_states().values():
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
        state = self._get_experiment_states().get(current.id)
        if state is None:
            state = self._bootstrap_registered_state(current, progress)
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


    def _actuate_active_allocator_transition(
        self,
        *,
        progress: Progress,
        owner_state: LinkedInExperimentState,
        verdict_payload: dict[str, Any],
        expectation: dict[str, Any],
        verdict: AllocationVerdict,
        work_units: LinkedInWorkUnitService,
        resumable_string_ids: frozenset[int],
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
        original_states = dict(self._get_experiment_states())
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
            work_units.deps.set_experiment_states(original_states)
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
                        assert item.id not in resumable_string_ids
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
                selected_state = self._get_experiment_states().get(selected.id)
                if selected_state is None:
                    selected_state = self._bootstrap_registered_state(selected, progress)
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
                work_units.set_pending_block_adaptation(
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
            work_units.checkpoint_progress(
                progress,
                search_string=None,
            )
        except BaseException:
            # The preimage restore must not replace the actuation
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
        *,
        work_units: LinkedInWorkUnitService,
        resumable_string_ids: frozenset[int],
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
            work_units=work_units,
            resumable_string_ids=resumable_string_ids,
        )

    def _checkpoint_progress(
        self,
        progress: Progress | None,
        search_string: SearchString | None = None,
        page_num: int | None = None,
        *,
        completed_page: bool = False,
        cursor_updates: tuple[tuple[int, str, int], ...] = (),
    ) -> None:
        started = time.monotonic()
        recorder = getattr(self.context, "_timing_recorder", None)
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
                cursor_updates=cursor_updates,
                _timings=timings,
                work_units=self.work_units,
                resumable_string_ids=self.candidates.resumable_string_ids(),
            )
        finally:
            if timings is not None:
                emit_timing_event(
                    recorder,
                    "checkpoint_progress_timing",
                    elapsed_ms=round((time.monotonic() - started) * 1000.0, 3),
                    **timings,
                )

    def experiment(self, root_id: int) -> LinkedInExperimentState | None:
        """Read one root's working lifecycle; cursor publication stays here."""
        return self._get_experiment_states().get(root_id)

    def register_experiment(self, state: LinkedInExperimentState) -> None:
        self._get_experiment_states()[state.root_string_id] = state

    def require_experiment(self, root_id: int) -> LinkedInExperimentState:
        return self._get_experiment_states()[root_id]
