"""Pure, deterministic scheduling policy for LinkedIn facial positives.

``FACIAL_BORDERLINE`` is a snippet-uncertainty verdict, not a weaker final
qualification tier.  This module decides only when a facial-positive candidate
is ready for full-profile review.  It never changes the full judgment contract.

The controller deliberately accepts no candidate name, photo, education, age,
or other demographic/proxy fields.  Ordering uses only canonical scheduling
identity and retrieval provenance: lane, string, page, and result rank.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Collection, Mapping, Sequence


FACIAL_YES = "FACIAL_YES"
FACIAL_BORDERLINE = "FACIAL_BORDERLINE"

SAVE_FAMILY_DECISIONS = frozenset(
    {
        "SAVE",
        "INFERENTIAL_SAVE",
        "TRANSFERABLE_SAVE",
        "SIGNAL_SAVE",
    }
)


class FacialRecallPolicyError(ValueError):
    """Raised when an explicit recall-policy value is structurally invalid."""


class FacialRecallMode(str, Enum):
    PRECISION_FIRST = "precision_first"
    BALANCED = "balanced"
    RECALL_FIRST = "recall_first"


class QualifiedTalentSupply(str, Enum):
    ABUNDANT = "abundant"
    MODERATE = "moderate"
    SCARCE = "scarce"
    UNKNOWN = "unknown"


class SnippetObservability(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FacialRecallPolicy:
    """Role-level instructions for spending BORDERLINE review effort."""

    mode: FacialRecallMode
    qualified_talent_supply: QualifiedTalentSupply = QualifiedTalentSupply.UNKNOWN
    snippet_observability: SnippetObservability = SnippetObservability.UNKNOWN
    initial_borderline_audit: int = 0
    borderline_wave_size: int = 4
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.mode, FacialRecallMode):
            raise FacialRecallPolicyError("mode must be a FacialRecallMode")
        if not isinstance(self.qualified_talent_supply, QualifiedTalentSupply):
            raise FacialRecallPolicyError(
                "qualified_talent_supply must be a QualifiedTalentSupply"
            )
        if not isinstance(self.snippet_observability, SnippetObservability):
            raise FacialRecallPolicyError(
                "snippet_observability must be a SnippetObservability"
            )
        _nonnegative_int(
            self.initial_borderline_audit,
            field="initial_borderline_audit",
        )
        _positive_int(self.borderline_wave_size, field="borderline_wave_size")
        if not isinstance(self.rationale, str):
            raise FacialRecallPolicyError("rationale must be a string")
        if (
            self.mode is FacialRecallMode.PRECISION_FIRST
            and self.initial_borderline_audit
        ):
            raise FacialRecallPolicyError(
                "precision_first requires initial_borderline_audit=0"
            )

    @classmethod
    def recall_first_default(cls) -> "FacialRecallPolicy":
        """Safe legacy fallback: review every facial-positive candidate."""

        return cls(
            mode=FacialRecallMode.RECALL_FIRST,
            initial_borderline_audit=0,
            borderline_wave_size=4,
            rationale="safe legacy fallback",
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FacialRecallPolicy":
        if not isinstance(payload, Mapping):
            raise FacialRecallPolicyError("facial_recall_policy must be an object")

        mode = _enum_value(
            FacialRecallMode,
            payload.get("mode"),
            field="facial_recall_policy.mode",
        )
        supply = _enum_value(
            QualifiedTalentSupply,
            payload.get("qualified_talent_supply", "unknown"),
            field="facial_recall_policy.qualified_talent_supply",
        )
        observability = _enum_value(
            SnippetObservability,
            payload.get("snippet_observability", "unknown"),
            field="facial_recall_policy.snippet_observability",
        )
        audit_default = 4 if mode is FacialRecallMode.BALANCED else 0
        audit = _nonnegative_int(
            payload.get("initial_borderline_audit", audit_default),
            field="facial_recall_policy.initial_borderline_audit",
        )
        wave = _positive_int(
            payload.get("borderline_wave_size", 4),
            field="facial_recall_policy.borderline_wave_size",
        )
        rationale = payload.get("rationale", "")
        if not isinstance(rationale, str):
            raise FacialRecallPolicyError(
                "facial_recall_policy.rationale must be a string"
            )
        if mode is FacialRecallMode.PRECISION_FIRST and audit:
            raise FacialRecallPolicyError(
                "precision_first requires initial_borderline_audit=0"
            )

        return cls(
            mode=mode,
            qualified_talent_supply=supply,
            snippet_observability=observability,
            initial_borderline_audit=audit,
            borderline_wave_size=wave,
            rationale=rationale.strip(),
        )


@dataclass(frozen=True, slots=True)
class OutreachSufficiency:
    """Operator-confirmed definition of a useful outreach-ready pool."""

    minimum_total: int
    minimum_distinct_lanes: int = 0
    required_lane_ids: tuple[str, ...] = ()
    operator_confirmed: bool = False

    def __post_init__(self) -> None:
        _positive_int(self.minimum_total, field="minimum_total")
        _nonnegative_int(
            self.minimum_distinct_lanes,
            field="minimum_distinct_lanes",
        )
        if not isinstance(self.required_lane_ids, tuple):
            raise FacialRecallPolicyError("required_lane_ids must be a tuple")
        if any(
            not isinstance(lane, str) or not lane.strip()
            for lane in self.required_lane_ids
        ):
            raise FacialRecallPolicyError(
                "required_lane_ids must contain non-empty strings"
            )
        if len(set(self.required_lane_ids)) != len(self.required_lane_ids):
            raise FacialRecallPolicyError("required_lane_ids must be unique")
        if not isinstance(self.operator_confirmed, bool):
            raise FacialRecallPolicyError("operator_confirmed must be a boolean")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OutreachSufficiency":
        if not isinstance(payload, Mapping):
            raise FacialRecallPolicyError("outreach_sufficiency must be an object")

        minimum_total = _positive_int(
            payload.get("minimum_total"),
            field="outreach_sufficiency.minimum_total",
        )
        minimum_distinct_lanes = _nonnegative_int(
            payload.get("minimum_distinct_lanes", 0),
            field="outreach_sufficiency.minimum_distinct_lanes",
        )
        raw_lanes = payload.get("required_lane_ids", ())
        if isinstance(raw_lanes, (str, bytes)) or not isinstance(raw_lanes, Sequence):
            raise FacialRecallPolicyError(
                "outreach_sufficiency.required_lane_ids must be a list of strings"
            )
        lanes: list[str] = []
        seen: set[str] = set()
        for value in raw_lanes:
            if not isinstance(value, str) or not value.strip():
                raise FacialRecallPolicyError(
                    "outreach_sufficiency.required_lane_ids must contain "
                    "non-empty strings"
                )
            lane = value.strip()
            if lane not in seen:
                seen.add(lane)
                lanes.append(lane)

        confirmed = payload.get("operator_confirmed", False)
        if not isinstance(confirmed, bool):
            raise FacialRecallPolicyError(
                "outreach_sufficiency.operator_confirmed must be a boolean"
            )

        return cls(
            minimum_total=minimum_total,
            minimum_distinct_lanes=minimum_distinct_lanes,
            required_lane_ids=tuple(lanes),
            operator_confirmed=confirmed,
        )


@dataclass(frozen=True, slots=True)
class FacialReviewCandidate:
    """The complete candidate input accepted by the policy controller.

    Candidate-controlled or demographic/profile attributes are intentionally
    absent. ``candidate_id`` must be the canonical scheduling identity.
    """

    candidate_id: str
    facial_decision: str
    lane_id: str = "general"
    string_id: str = ""
    page: int = 0
    result_rank: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise FacialRecallPolicyError("candidate_id must be a non-empty string")
        if self.facial_decision not in {FACIAL_YES, FACIAL_BORDERLINE}:
            raise FacialRecallPolicyError(
                "facial_decision must be FACIAL_YES or FACIAL_BORDERLINE"
            )
        if not isinstance(self.lane_id, str) or not isinstance(self.string_id, str):
            raise FacialRecallPolicyError("lane_id and string_id must be strings")
        _nonnegative_int(self.page, field="candidate.page")
        _nonnegative_int(self.result_rank, field="candidate.result_rank")


@dataclass(frozen=True, slots=True)
class SettledFullOutcome:
    """One canonical settled full-profile outcome used for sufficiency."""

    candidate_id: str
    decision: str
    lane_id: str = "general"

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise FacialRecallPolicyError(
                "settled outcome candidate_id must be a non-empty string"
            )
        if not isinstance(self.decision, str) or not self.decision.strip():
            raise FacialRecallPolicyError(
                "settled outcome decision must be a non-empty string"
            )
        if not isinstance(self.lane_id, str):
            raise FacialRecallPolicyError("settled outcome lane_id must be a string")


@dataclass(frozen=True, slots=True)
class OutreachSufficiencyStatus:
    outreach_total: int
    outreach_lane_ids: tuple[str, ...]
    missing_total: int
    missing_distinct_lanes: int
    missing_required_lane_ids: tuple[str, ...]
    conflicting_candidate_ids: tuple[str, ...]
    met: bool


@dataclass(frozen=True, slots=True)
class ResolvedFacialRecallPolicy:
    policy: FacialRecallPolicy
    sufficiency: OutreachSufficiency | None
    used_safe_fallback: bool
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FacialReviewPlan:
    """Deterministic selection result for one scheduler evaluation."""

    effective_mode: FacialRecallMode
    strict_ready_ids: tuple[str, ...]
    borderline_ready_ids: tuple[str, ...]
    deferred_borderline_ids: tuple[str, ...]
    already_activated_borderline_ids: tuple[str, ...]
    reason: str
    sufficiency_status: OutreachSufficiencyStatus | None
    used_safe_fallback: bool = False
    fallback_reason: str | None = None
    review_all_override: bool = False

    @property
    def ready_candidate_ids(self) -> tuple[str, ...]:
        """Strict candidates always precede BORDERLINE candidates."""

        return self.strict_ready_ids + self.borderline_ready_ids


def resolve_facial_recall_policy(
    policy: FacialRecallPolicy | Mapping[str, Any] | None,
    sufficiency: OutreachSufficiency | Mapping[str, Any] | None,
) -> ResolvedFacialRecallPolicy:
    """Resolve explicit input or fail safely to current recall-first behavior."""

    try:
        parsed_policy = _coerce_policy(policy)
    except (FacialRecallPolicyError, TypeError, ValueError):
        return _safe_fallback("policy_malformed")
    if parsed_policy is None:
        return _safe_fallback("policy_absent")

    try:
        parsed_sufficiency = _coerce_sufficiency(sufficiency)
    except (FacialRecallPolicyError, TypeError, ValueError):
        return _safe_fallback("sufficiency_malformed")
    if parsed_sufficiency is None:
        return _safe_fallback("sufficiency_absent")
    if not parsed_sufficiency.operator_confirmed:
        return _safe_fallback("sufficiency_unconfirmed")

    return ResolvedFacialRecallPolicy(
        policy=parsed_policy,
        sufficiency=parsed_sufficiency,
        used_safe_fallback=False,
    )


def evaluate_outreach_sufficiency(
    sufficiency: OutreachSufficiency,
    outcomes: Sequence[SettledFullOutcome],
) -> OutreachSufficiencyStatus:
    """Count unique, unambiguous SAVE-family outcomes only.

    Conflicting duplicate terminal outcomes are excluded.  The conservative
    result can cause extra review, but can never suppress review on corrupted
    evidence.
    """

    grouped: dict[str, list[SettledFullOutcome]] = defaultdict(list)
    for outcome in outcomes:
        if not isinstance(outcome, SettledFullOutcome):
            raise FacialRecallPolicyError(
                "outcomes must contain SettledFullOutcome values"
            )
        grouped[outcome.candidate_id.strip()].append(outcome)

    counted: dict[str, str] = {}
    conflicts: list[str] = []
    for candidate_id in sorted(grouped):
        candidate_outcomes = grouped[candidate_id]
        decisions = {outcome.decision.strip().upper() for outcome in candidate_outcomes}
        if len(decisions) != 1:
            conflicts.append(candidate_id)
            continue
        decision = next(iter(decisions))
        if decision not in SAVE_FAMILY_DECISIONS:
            continue
        lanes = sorted(
            {
                _normalized_lane(outcome.lane_id)
                for outcome in candidate_outcomes
                if _normalized_lane(outcome.lane_id)
            }
        )
        counted[candidate_id] = lanes[0] if lanes else "general"

    outreach_lanes = tuple(sorted(set(counted.values())))
    missing_required = tuple(
        lane for lane in sufficiency.required_lane_ids if lane not in outreach_lanes
    )
    missing_total = max(0, sufficiency.minimum_total - len(counted))
    missing_distinct = max(
        0,
        sufficiency.minimum_distinct_lanes - len(outreach_lanes),
    )
    met = not missing_total and not missing_distinct and not missing_required
    return OutreachSufficiencyStatus(
        outreach_total=len(counted),
        outreach_lane_ids=outreach_lanes,
        missing_total=missing_total,
        missing_distinct_lanes=missing_distinct,
        missing_required_lane_ids=missing_required,
        conflicting_candidate_ids=tuple(conflicts),
        met=met,
    )


def plan_facial_reviews(
    candidates: Sequence[FacialReviewCandidate],
    *,
    policy: FacialRecallPolicy | Mapping[str, Any] | None,
    sufficiency: OutreachSufficiency | Mapping[str, Any] | None,
    settled_outcomes: Sequence[SettledFullOutcome] = (),
    already_activated_borderline_ids: Collection[str] = (),
    activate_deferred_wave: bool = False,
    review_all: bool = False,
) -> FacialReviewPlan:
    """Return strict-ready, BORDERLINE-ready, and deferred identities.

    ``activate_deferred_wave`` requests at most one bounded wave.  In balanced
    mode, an unfinished initial audit is selected before any later wave.  The
    caller persists activation and invokes the controller again only after that
    selection settles.
    """

    ordered_candidates = _validated_candidates(candidates)
    strict = tuple(
        candidate
        for candidate in ordered_candidates
        if candidate.facial_decision == FACIAL_YES
    )
    borderline = tuple(
        candidate
        for candidate in ordered_candidates
        if candidate.facial_decision == FACIAL_BORDERLINE
    )
    activated = frozenset(_validated_ids(already_activated_borderline_ids))

    if review_all:
        all_ordered_borderline = _round_robin_borderlines(borderline, ())
        eligible_borderline = tuple(
            candidate
            for candidate in all_ordered_borderline
            if candidate.candidate_id not in activated
        )
        return FacialReviewPlan(
            effective_mode=FacialRecallMode.RECALL_FIRST,
            strict_ready_ids=tuple(candidate.candidate_id for candidate in strict),
            borderline_ready_ids=tuple(
                candidate.candidate_id for candidate in eligible_borderline
            ),
            deferred_borderline_ids=(),
            already_activated_borderline_ids=tuple(
                candidate.candidate_id
                for candidate in all_ordered_borderline
                if candidate.candidate_id in activated
            ),
            reason="review_all_override",
            sufficiency_status=None,
            review_all_override=True,
        )

    resolved = resolve_facial_recall_policy(policy, sufficiency)
    parsed_sufficiency = resolved.sufficiency
    status = (
        evaluate_outreach_sufficiency(parsed_sufficiency, settled_outcomes)
        if parsed_sufficiency is not None
        else None
    )
    all_ordered_borderline = _round_robin_borderlines(
        borderline,
        status.missing_required_lane_ids if status is not None else (),
    )
    eligible_borderline = tuple(
        candidate
        for candidate in all_ordered_borderline
        if candidate.candidate_id not in activated
    )
    activated_present = tuple(
        candidate.candidate_id
        for candidate in all_ordered_borderline
        if candidate.candidate_id in activated
    )

    selected: tuple[FacialReviewCandidate, ...]
    reason: str
    mode = resolved.policy.mode
    if mode is FacialRecallMode.RECALL_FIRST:
        selected = eligible_borderline
        reason = resolved.fallback_reason or "recall_first"
    elif status is not None and status.met:
        selected = ()
        reason = "sufficiency_met"
    elif mode is FacialRecallMode.BALANCED:
        audit_prefix = all_ordered_borderline[
            : resolved.policy.initial_borderline_audit
        ]
        unactivated_audit = tuple(
            candidate
            for candidate in audit_prefix
            if candidate.candidate_id not in activated
        )
        if unactivated_audit:
            selected = unactivated_audit
            reason = "balanced_initial_audit"
        elif activate_deferred_wave:
            selected = eligible_borderline[: resolved.policy.borderline_wave_size]
            reason = "balanced_deferred_wave"
        else:
            selected = ()
            reason = "balanced_deferred"
    elif activate_deferred_wave:
        selected = eligible_borderline[: resolved.policy.borderline_wave_size]
        reason = "precision_first_deferred_wave"
    else:
        selected = ()
        reason = "precision_first_deferred"

    selected_ids = tuple(candidate.candidate_id for candidate in selected)
    selected_set = frozenset(selected_ids)
    deferred_ids = tuple(
        candidate.candidate_id
        for candidate in eligible_borderline
        if candidate.candidate_id not in selected_set
    )
    return FacialReviewPlan(
        effective_mode=mode,
        strict_ready_ids=tuple(candidate.candidate_id for candidate in strict),
        borderline_ready_ids=selected_ids,
        deferred_borderline_ids=deferred_ids,
        already_activated_borderline_ids=activated_present,
        reason=reason,
        sufficiency_status=status,
        used_safe_fallback=resolved.used_safe_fallback,
        fallback_reason=resolved.fallback_reason,
    )


def _safe_fallback(reason: str) -> ResolvedFacialRecallPolicy:
    return ResolvedFacialRecallPolicy(
        policy=FacialRecallPolicy.recall_first_default(),
        sufficiency=None,
        used_safe_fallback=True,
        fallback_reason=reason,
    )


def _coerce_policy(
    value: FacialRecallPolicy | Mapping[str, Any] | None,
) -> FacialRecallPolicy | None:
    if value is None:
        return None
    if isinstance(value, FacialRecallPolicy):
        return value
    if isinstance(value, Mapping):
        return FacialRecallPolicy.from_mapping(value)
    raise FacialRecallPolicyError("facial_recall_policy must be an object")


def _coerce_sufficiency(
    value: OutreachSufficiency | Mapping[str, Any] | None,
) -> OutreachSufficiency | None:
    if value is None:
        return None
    if isinstance(value, OutreachSufficiency):
        return value
    if isinstance(value, Mapping):
        return OutreachSufficiency.from_mapping(value)
    raise FacialRecallPolicyError("outreach_sufficiency must be an object")


def _validated_candidates(
    candidates: Sequence[FacialReviewCandidate],
) -> tuple[FacialReviewCandidate, ...]:
    seen: set[str] = set()
    values: list[FacialReviewCandidate] = []
    for candidate in candidates:
        if not isinstance(candidate, FacialReviewCandidate):
            raise FacialRecallPolicyError(
                "candidates must contain FacialReviewCandidate values"
            )
        candidate_id = candidate.candidate_id.strip()
        if candidate_id in seen:
            raise FacialRecallPolicyError(
                f"duplicate facial review candidate_id: {candidate_id}"
            )
        seen.add(candidate_id)
        values.append(candidate)
    return tuple(sorted(values, key=_candidate_order_key))


def _validated_ids(values: Collection[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise FacialRecallPolicyError(
            "already_activated_borderline_ids must be a collection of strings"
        )
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise FacialRecallPolicyError(
                "already_activated_borderline_ids must contain non-empty strings"
            )
        result.append(value.strip())
    return tuple(result)


def _round_robin_borderlines(
    candidates: Sequence[FacialReviewCandidate],
    missing_required_lanes: Sequence[str],
) -> tuple[FacialReviewCandidate, ...]:
    """Gap lanes first, then stable lane/string bucket round-robin.

    Candidates within a bucket retain original page/result-rank order.  Bucket
    keys are sorted so selection is independent of input iteration order.
    """

    missing = frozenset(missing_required_lanes)
    gap_candidates = tuple(
        candidate
        for candidate in candidates
        if _normalized_lane(candidate.lane_id) in missing
    )
    remaining_candidates = tuple(
        candidate
        for candidate in candidates
        if _normalized_lane(candidate.lane_id) not in missing
    )
    return _bucket_round_robin(gap_candidates) + _bucket_round_robin(
        remaining_candidates
    )


def _bucket_round_robin(
    candidates: Sequence[FacialReviewCandidate],
) -> tuple[FacialReviewCandidate, ...]:
    buckets: dict[tuple[str, str], deque[FacialReviewCandidate]] = defaultdict(deque)
    for candidate in sorted(candidates, key=_candidate_order_key):
        buckets[
            (_normalized_lane(candidate.lane_id), candidate.string_id.strip())
        ].append(candidate)

    ordered: list[FacialReviewCandidate] = []
    bucket_keys = sorted(buckets)
    while bucket_keys:
        next_keys: list[tuple[str, str]] = []
        for key in bucket_keys:
            bucket = buckets[key]
            ordered.append(bucket.popleft())
            if bucket:
                next_keys.append(key)
        bucket_keys = next_keys
    return tuple(ordered)


def _candidate_order_key(
    candidate: FacialReviewCandidate,
) -> tuple[int, int, str, str, str]:
    return (
        candidate.page,
        candidate.result_rank,
        _normalized_lane(candidate.lane_id),
        candidate.string_id.strip(),
        candidate.candidate_id,
    )


def _normalized_lane(value: str) -> str:
    return value.strip() or "general"


def _enum_value(enum_type: type[Enum], value: Any, *, field: str):
    if not isinstance(value, str):
        raise FacialRecallPolicyError(f"{field} must be a string")
    try:
        return enum_type(value.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise FacialRecallPolicyError(
            f"{field} must be one of: {allowed}"
        ) from exc


def _nonnegative_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FacialRecallPolicyError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FacialRecallPolicyError(f"{field} must be a positive integer")
    return value


__all__ = [
    "FACIAL_BORDERLINE",
    "FACIAL_YES",
    "SAVE_FAMILY_DECISIONS",
    "FacialRecallMode",
    "FacialRecallPolicy",
    "FacialRecallPolicyError",
    "FacialReviewCandidate",
    "FacialReviewPlan",
    "OutreachSufficiency",
    "OutreachSufficiencyStatus",
    "QualifiedTalentSupply",
    "ResolvedFacialRecallPolicy",
    "SettledFullOutcome",
    "SnippetObservability",
    "evaluate_outreach_sufficiency",
    "plan_facial_reviews",
    "resolve_facial_recall_policy",
]
