"""Page allocator state and expectation cluster for LinkedIn sourcing runs.

Owns the pure read/computation helpers for page-allocator tracking,
expectation projection, and verdict rehydration. ``Pipeline`` delegates to
``AllocatorStateService``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from shared.contracts import NON_SAVE_REVIEW_DECISIONS
from shared.judger import SAVE_FAMILY_DECISIONS
from shared.schemas import CandidateSnippet, OpusDecision, Progress, SearchString

from linkedin.page_allocator import (
    AllocationAction,
    AllocationVerdict,
    AllocatorArm,
    AllocatorPolicyError,
    PageObservation,
)
from linkedin.search_intelligence import (
    LinkedInExperimentState,
    bootstrap_experiment_state,
)
from shared import config


@dataclass(frozen=True)
class AllocatorStateDeps:
    get_experiment_states: Callable[[], dict[int, LinkedInExperimentState]]
    get_allocator_page_identity: Callable[[], Any]
    get_pending_allocator_checkpoint: Callable[[], Any]


class AllocatorStateService:
    """Owns page-allocator state reads and expectation computation."""

    def __init__(self, deps: AllocatorStateDeps):
        self.deps = deps

    @staticmethod
    def _allocator_terminal_full_decision(decision: OpusDecision) -> bool:
        return (
            decision.stage == "full"
            and (
                decision.decision in SAVE_FAMILY_DECISIONS
                or decision.decision in NON_SAVE_REVIEW_DECISIONS
                or decision.decision == "REJECT"
            )
        )

    def _allocator_page_matches(self, snippet: CandidateSnippet) -> bool:
        identity = self.deps.get_allocator_page_identity()
        if identity is None:
            return False
        root_string_id, _variant_id, page = identity
        return (
            int(snippet.source_string_id) == root_string_id
            and max(1, int(snippet.page or 1)) == page
        )

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
            for state in self.deps.get_experiment_states().values()
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

    def _allocator_state_for_arm(
        self,
        search_string: SearchString,
    ) -> LinkedInExperimentState:
        state = self.deps.get_experiment_states().get(search_string.id)
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
        """Project TUR-15's selected-first/paused-tail order without mutation."""

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
        pending = self.deps.get_pending_allocator_checkpoint()
        if not isinstance(pending, dict):
            return None
        if pending.get("kind") in {"dispatch", "exhaustion"}:
            return pending
        state = self.deps.get_experiment_states().get(
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
