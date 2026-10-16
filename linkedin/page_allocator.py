"""Pure deterministic page allocation across LinkedIn root search strings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from linkedin.adaptation_signal_state import WilsonInterval


_INTENTIONAL_PARTIAL_REASONS = frozenset({"early_exit", "glance_reformulate"})
_ELIGIBLE_FLOOR_PERCENT = 8


class AllocatorPolicyError(ValueError):
    """Raised when a frontier cannot be scheduled as one root-string block."""


class AllocationAction(str, Enum):
    CONTINUE = "continue"
    SWITCH = "switch"
    FLOOR = "floor"
    FINISH = "finish"


@dataclass(frozen=True)
class PageObservation:
    """Currency and completeness evidence for one canonically completed page."""

    root_string_id: int
    variant_id: str
    page: int
    slots: int
    extracted: int
    full_expected: int
    full_settled: int
    priority: int
    standard: int
    outreach: int
    break_reason: str = ""
    technical_interruption: bool = False
    off_policy: bool = False

    @property
    def n(self) -> int:
        return self.extracted

    @property
    def p(self) -> int:
        return self.priority

    @property
    def e(self) -> int:
        return self.priority + self.standard

    @property
    def intentional_partial(self) -> bool:
        return self.break_reason in _INTENTIONAL_PARTIAL_REASONS

    @property
    def invalid_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        counters = (
            self.root_string_id,
            self.page,
            self.slots,
            self.extracted,
            self.full_expected,
            self.full_settled,
            self.priority,
            self.standard,
            self.outreach,
        )
        if any(value < 0 for value in counters):
            reasons.append("negative_counter")
        if self.root_string_id <= 0:
            reasons.append("invalid_root_string_id")
        if not self.variant_id:
            reasons.append("missing_variant_id")
        if self.page <= 0:
            reasons.append("invalid_page")
        if self.extracted <= 0:
            reasons.append("empty_extraction")
        if self.extracted > self.slots:
            reasons.append("extraction_exceeds_slots")
        if self.break_reason and not self.intentional_partial:
            reasons.append("unsupported_break_reason")
        if not self.intentional_partial and self.extracted * 5 < self.slots * 4:
            reasons.append("incomplete_extraction")
        if self.full_expected > self.extracted:
            reasons.append("full_reviews_exceed_extraction")
        if self.full_settled > self.full_expected:
            reasons.append("full_settled_exceeds_expected")
        if self.full_expected > 0 and self.full_settled * 5 < self.full_expected * 4:
            reasons.append("incomplete_full_reviews")
        if self.e != self.outreach:
            reasons.append("tier_outreach_mismatch")
        if self.outreach > self.full_settled:
            reasons.append("outreach_exceeds_full_settled")
        if self.technical_interruption:
            reasons.append("technical_interruption")
        return tuple(dict.fromkeys(reasons))

    @property
    def valid(self) -> bool:
        return not self.invalid_reasons

    @property
    def teaches_policy(self) -> bool:
        return self.valid and not self.off_policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_string_id": self.root_string_id,
            "variant_id": self.variant_id,
            "page": self.page,
            "slots": self.slots,
            "extracted": self.extracted,
            "full_expected": self.full_expected,
            "full_settled": self.full_settled,
            "priority": self.priority,
            "standard": self.standard,
            "outreach": self.outreach,
            "break_reason": self.break_reason,
            "technical_interruption": self.technical_interruption,
            "off_policy": self.off_policy,
            "valid": self.valid,
            "invalid_reasons": list(self.invalid_reasons),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PageObservation":
        return cls(
            root_string_id=int(payload.get("root_string_id", 0) or 0),
            variant_id=str(payload.get("variant_id", "") or ""),
            page=int(payload.get("page", 0) or 0),
            slots=int(payload.get("slots", 0) or 0),
            extracted=int(payload.get("extracted", 0) or 0),
            full_expected=int(payload.get("full_expected", 0) or 0),
            full_settled=int(payload.get("full_settled", 0) or 0),
            priority=int(payload.get("priority", 0) or 0),
            standard=int(payload.get("standard", 0) or 0),
            outreach=int(payload.get("outreach", 0) or 0),
            break_reason=str(payload.get("break_reason", "") or ""),
            technical_interruption=bool(payload.get("technical_interruption", False)),
            off_policy=bool(payload.get("off_policy", False)),
        )


@dataclass(frozen=True)
class AllocatorArm:
    """One root search string and the bounded window of its active variant."""

    root_string_id: int
    block: str
    queue_priority: int
    active_variant_id: str
    observations: tuple[PageObservation, ...] = ()
    active_valid_page_count: int = 0
    root_has_valid_probe: bool = False
    legacy_unobserved_pages: int = 0
    physically_exhausted: bool = False
    terminal: bool = False


@dataclass(frozen=True)
class ArmScore:
    root_string_id: int
    queue_priority: int
    n: int
    priority: int
    eligible: int
    priority_upper: float
    eligible_upper: float

    @property
    def viable(self) -> bool:
        return self.priority >= 1 or (
            self.n > 0 and self.eligible * 100 >= self.n * _ELIGIBLE_FLOOR_PERCENT
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_string_id": self.root_string_id,
            "queue_priority": self.queue_priority,
            "n": self.n,
            "priority": self.priority,
            "eligible": self.eligible,
            "priority_upper": self.priority_upper,
            "eligible_upper": self.eligible_upper,
            "viable": self.viable,
        }


@dataclass(frozen=True)
class AllocationVerdict:
    action: AllocationAction
    current_root_id: int
    selected_root_id: int | None
    reason: str
    paused_root_ids: tuple[int, ...] = ()
    floored_root_ids: tuple[int, ...] = ()
    ranked_root_ids: tuple[int, ...] = ()
    scores: tuple[ArmScore, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "current_root_id": self.current_root_id,
            "selected_root_id": self.selected_root_id,
            "reason": self.reason,
            "paused_root_ids": list(self.paused_root_ids),
            "floored_root_ids": list(self.floored_root_ids),
            "ranked_root_ids": list(self.ranked_root_ids),
            "scores": [score.to_dict() for score in self.scores],
        }


def pool_arm(arm: AllocatorArm) -> ArmScore:
    """Pool the active variant's last two valid, on-policy page observations."""

    observations = [
        observation
        for observation in arm.observations
        if observation.root_string_id == arm.root_string_id
        and observation.variant_id == arm.active_variant_id
        and observation.teaches_policy
    ][-2:]
    n = sum(observation.n for observation in observations)
    priority = sum(observation.p for observation in observations)
    eligible = sum(observation.e for observation in observations)
    return ArmScore(
        root_string_id=arm.root_string_id,
        queue_priority=arm.queue_priority,
        n=n,
        priority=priority,
        eligible=eligible,
        priority_upper=WilsonInterval(priority, n, 1.96).upper,
        eligible_upper=WilsonInterval(eligible, n, 1.96).upper,
    )


def challenger_clears_friction(current: ArmScore, challenger: ArmScore) -> bool:
    """Return whether the best challenger strictly clears switching friction."""

    if current.n <= 0 or challenger.n <= 0:
        return False
    h = max(1 / current.n, 1 / challenger.n)
    if challenger.priority_upper > current.priority_upper + h:
        return True
    return (
        current.priority_upper - h
        < challenger.priority_upper
        < current.priority_upper + h
        and challenger.eligible_upper > current.eligible_upper + 2 * h
    )


def _rank(scores: Iterable[ArmScore]) -> list[ArmScore]:
    return sorted(
        scores,
        key=lambda score: (
            -score.priority_upper,
            -score.eligible_upper,
            score.queue_priority,
            score.root_string_id,
        ),
    )


def _probe_target(arms: Sequence[AllocatorArm]) -> AllocatorArm | None:
    unprobed = [arm for arm in arms if not arm.root_has_valid_probe]
    if not unprobed:
        return None
    return min(
        unprobed,
        key=lambda arm: (
            arm.legacy_unobserved_pages > 0,
            arm.queue_priority,
            arm.root_string_id,
        ),
    )


def _switch_verdict(
    *,
    current_root_id: int,
    target_root_id: int,
    reason: str,
    scores: Sequence[ArmScore],
    floored_root_ids: tuple[int, ...] = (),
    pause_current: bool = True,
) -> AllocationVerdict:
    return AllocationVerdict(
        action=AllocationAction.SWITCH,
        current_root_id=current_root_id,
        selected_root_id=target_root_id,
        reason=reason,
        paused_root_ids=(current_root_id,) if pause_current else (),
        floored_root_ids=floored_root_ids,
        ranked_root_ids=tuple(score.root_string_id for score in scores),
        scores=tuple(scores),
    )


def _finish_verdict(
    *,
    current_root_id: int,
    reason: str,
    scores: Sequence[ArmScore],
) -> AllocationVerdict:
    return AllocationVerdict(
        action=AllocationAction.FINISH,
        current_root_id=current_root_id,
        selected_root_id=None,
        reason=reason,
        ranked_root_ids=tuple(score.root_string_id for score in scores),
        scores=tuple(scores),
    )


def allocate_page(
    *,
    current_root_id: int,
    arms: Sequence[AllocatorArm],
) -> AllocationVerdict:
    """Choose the next same-block root without mutating execution state."""

    if not arms:
        raise AllocatorPolicyError("allocator frontier cannot be empty")
    if len({arm.block for arm in arms}) != 1:
        raise AllocatorPolicyError("allocator frontier must be one contiguous block")
    ids = [arm.root_string_id for arm in arms]
    if len(ids) != len(set(ids)):
        raise AllocatorPolicyError("allocator frontier contains duplicate roots")
    if any(root_id <= 0 for root_id in ids):
        raise AllocatorPolicyError("allocator frontier contains an invalid root")
    priorities = [arm.queue_priority for arm in arms]
    if len(priorities) != len(set(priorities)):
        raise AllocatorPolicyError(
            "allocator frontier contains duplicate queue priorities"
        )
    for arm in arms:
        if any(
            observation.root_string_id != arm.root_string_id
            for observation in arm.observations
        ):
            raise AllocatorPolicyError(
                "allocator arm contains mismatched observations"
            )

    by_id = {arm.root_string_id: arm for arm in arms}
    current = by_id.get(current_root_id)
    if current is None:
        raise AllocatorPolicyError("current root is outside allocator frontier")
    if current.terminal:
        raise AllocatorPolicyError("current root is terminal")

    live = [arm for arm in arms if not arm.terminal]
    all_scores = _rank(pool_arm(arm) for arm in live)
    if current.physically_exhausted:
        remaining = [
            arm
            for arm in live
            if arm.root_string_id != current_root_id and not arm.physically_exhausted
        ]
        if not remaining:
            return _finish_verdict(
                current_root_id=current_root_id,
                reason="physical_exhaustion",
                scores=all_scores,
            )
        target = _probe_target(remaining)
        if target is None:
            target = by_id[_rank(pool_arm(arm) for arm in remaining)[0].root_string_id]
        return _switch_verdict(
            current_root_id=current_root_id,
            target_root_id=target.root_string_id,
            reason="physical_exhaustion",
            scores=all_scores,
            pause_current=False,
        )

    frontier = [arm for arm in live if not arm.physically_exhausted]
    probe_target = _probe_target(frontier)
    if probe_target is not None:
        if probe_target.root_string_id != current_root_id:
            return _switch_verdict(
                current_root_id=current_root_id,
                target_root_id=probe_target.root_string_id,
                reason="opening_probe",
                scores=all_scores,
            )
        return AllocationVerdict(
            action=AllocationAction.CONTINUE,
            current_root_id=current_root_id,
            selected_root_id=current_root_id,
            reason="opening_probe",
            ranked_root_ids=tuple(score.root_string_id for score in all_scores),
            scores=tuple(all_scores),
        )

    floored_root_ids: tuple[int, ...] = ()
    if all(arm.active_valid_page_count >= 2 for arm in frontier):
        score_by_id = {score.root_string_id: score for score in all_scores}
        floored_root_ids = tuple(
            arm.root_string_id
            for arm in sorted(frontier, key=lambda item: (item.queue_priority, item.root_string_id))
            if not score_by_id[arm.root_string_id].viable
        )
        if floored_root_ids:
            viable = [arm for arm in frontier if arm.root_string_id not in floored_root_ids]
            if not viable:
                return AllocationVerdict(
                    action=AllocationAction.FLOOR,
                    current_root_id=current_root_id,
                    selected_root_id=None,
                    reason="allocation_floor",
                    floored_root_ids=floored_root_ids,
                    ranked_root_ids=tuple(score.root_string_id for score in all_scores),
                    scores=tuple(all_scores),
                )
            if current_root_id in floored_root_ids:
                target = _rank(pool_arm(arm) for arm in viable)[0]
                return _switch_verdict(
                    current_root_id=current_root_id,
                    target_root_id=target.root_string_id,
                    reason="allocation_floor",
                    scores=all_scores,
                    floored_root_ids=floored_root_ids,
                    pause_current=False,
                )
            frontier = viable

    scores = _rank(pool_arm(arm) for arm in frontier)
    current_score = next(score for score in scores if score.root_string_id == current_root_id)
    challengers = [score for score in scores if score.root_string_id != current_root_id]
    if challengers and challenger_clears_friction(current_score, challengers[0]):
        return _switch_verdict(
            current_root_id=current_root_id,
            target_root_id=challengers[0].root_string_id,
            reason="relative_underperformance",
            scores=scores,
            floored_root_ids=floored_root_ids,
        )
    return AllocationVerdict(
        action=AllocationAction.CONTINUE,
        current_root_id=current_root_id,
        selected_root_id=current_root_id,
        reason="friction_hold" if challengers else "only_frontier_arm",
        floored_root_ids=floored_root_ids,
        ranked_root_ids=tuple(score.root_string_id for score in scores),
        scores=tuple(scores),
    )
