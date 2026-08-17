"""Typed adaptation signal state and action validation for LinkedIn runs."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence

from linkedin.boolean_compiler import (
    BooleanNormalizationError,
    BooleanNormalizationReport,
    normalize_boolean_for_linkedin,
)
from shared.schemas import BlockReport


class AdaptationGateDecision(str, Enum):
    ADAPT = "adapt"
    COLLECT_MORE_SIGNAL = "collect_more_signal"
    COOLDOWN = "cooldown"
    RESET_BLOCKED = "reset_blocked"


class ProfileIdAvailabilityStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"


class AdaptationValidationError(ValueError):
    """Raised when an adaptation action or adapted string fails validation."""


@dataclass(frozen=True)
class WilsonInterval:
    successes: int
    total: int
    confidence_z: float = 1.96

    @property
    def rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def lower(self) -> float:
        return self._bounds()[0]

    @property
    def upper(self) -> float:
        return self._bounds()[1]

    def _bounds(self) -> tuple[float, float]:
        if self.total <= 0:
            return (0.0, 0.0)
        z = self.confidence_z
        n = self.total
        p = self.rate
        denominator = 1 + z * z / n
        centre = p + z * z / (2 * n)
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        return ((centre - margin) / denominator, (centre + margin) / denominator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "successes": self.successes,
            "total": self.total,
            "rate": self.rate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_z": self.confidence_z,
        }


@dataclass(frozen=True)
class ProfileIdAvailabilityContract:
    status: ProfileIdAvailabilityStatus
    evidence: Mapping[str, Any] = field(default_factory=dict)
    verified_at: str | None = None

    @classmethod
    def unverified(cls) -> "ProfileIdAvailabilityContract":
        return cls(status=ProfileIdAvailabilityStatus.UNVERIFIED)

    @classmethod
    def verified(
        cls,
        *,
        evidence: Mapping[str, Any],
        verified_at: str,
    ) -> "ProfileIdAvailabilityContract":
        if not evidence:
            raise AdaptationValidationError(
                "profile-ID availability verification requires evidence"
            )
        if not verified_at:
            raise AdaptationValidationError(
                "profile-ID availability verification requires verified_at"
            )
        _require_iso_timestamp(verified_at, field="verified_at")
        return cls(
            status=ProfileIdAvailabilityStatus.VERIFIED,
            evidence=dict(evidence),
            verified_at=verified_at,
        )

    @property
    def is_verified(self) -> bool:
        return self.status == ProfileIdAvailabilityStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "evidence": dict(self.evidence),
            "verified_at": self.verified_at,
        }


UNVERIFIED_PROFILE_ID_AVAILABILITY_CONTRACT = (
    ProfileIdAvailabilityContract.unverified()
)


@dataclass(frozen=True)
class MarketSignal:
    signal_type: str
    recommendation: str
    evidence_ref_ids: tuple[str, ...] = ()
    confidence: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MarketSignal":
        signal_type = _optional_string_field(
            payload,
            ("signal_type", "type"),
            default="note",
            field="market signal signal_type",
        )
        recommendation = _required_string_field(
            payload,
            ("recommendation", "message"),
            field="market signal recommendation",
            missing_message="market signal requires a recommendation",
        )
        evidence = _optional_string_sequence_field(
            _first_present_value(payload, ("evidence_ref_ids", "evidence_refs")),
            field="market signal evidence_ref_ids",
        )
        confidence_value = _optional_confidence_value(payload.get("confidence"))
        return cls(
            signal_type=signal_type or "note",
            recommendation=recommendation,
            evidence_ref_ids=evidence,
            confidence=confidence_value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "recommendation": self.recommendation,
            "evidence_ref_ids": list(self.evidence_ref_ids),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class MarketSignalPrior:
    source: str
    signals: tuple[MarketSignal, ...] = ()
    context_hash: str = ""
    generated_at: str | None = None

    @classmethod
    def empty(cls) -> "MarketSignalPrior":
        return cls(source="none")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MarketSignalPrior":
        raw_signals = payload.get("signals") or ()
        if not isinstance(raw_signals, Sequence) or isinstance(raw_signals, (str, bytes)):
            raise AdaptationValidationError("market prior signals must be an array")
        signals: list[MarketSignal] = []
        for index, signal in enumerate(raw_signals):
            if not isinstance(signal, Mapping):
                raise AdaptationValidationError(
                    f"market prior signals[{index}] must be an object"
                )
            signals.append(MarketSignal.from_mapping(signal))
        return cls(
            source=_optional_string_field(
                payload,
                ("source",),
                default="market_intelligence",
                field="market prior source",
            ),
            signals=tuple(signals),
            context_hash=_optional_string_field(
                payload,
                ("context_hash",),
                default="",
                field="market prior context_hash",
            ),
            generated_at=_optional_string_field(
                payload,
                ("generated_at",),
                default=None,
                field="market prior generated_at",
            ),
        )

    @classmethod
    def from_advisory_context(cls, context: str) -> "MarketSignalPrior":
        text = str(context or "").strip()
        if not text:
            return cls.empty()
        signals: list[MarketSignal] = []
        for line in text.splitlines():
            parsed = _parse_market_advisory_line(line)
            if parsed is not None:
                signals.append(parsed)
        return cls(
            source="market_intelligence.live_advisory",
            signals=tuple(signals),
            context_hash="sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    @property
    def is_empty(self) -> bool:
        return not self.signals

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "signals": [signal.to_dict() for signal in self.signals],
            "context_hash": self.context_hash,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class SearchSignalState:
    block_name: str
    strings_run: int
    total_results: int
    total_saves: int
    candidates_seen: int
    duplicates_seen: int
    facial_yes: int
    facial_no: int
    edge_case_saves: int
    canonical_saves: int
    zero_save_string_ids: tuple[int, ...] = ()
    employer_counts: Mapping[str, int] = field(default_factory=dict)
    overlap_available: bool = False
    overlap_rate: float | None = None
    profile_id_sample_size: int = 0
    profile_id_availability: ProfileIdAvailabilityContract = (
        UNVERIFIED_PROFILE_ID_AVAILABILITY_CONTRACT
    )
    # P2.2 (actuator honesty): per-dimension facet-value counts aggregated from
    # the strings' surface receipts, so the adaptation prompt sees actuator
    # health — a lane whose facets never landed must not be judged as a lane
    # that ran. strings_fell_back counts strings that ran keyword-only despite
    # requesting structured filters.
    facet_values_requested: Mapping[str, int] = field(default_factory=dict)
    facet_values_applied: Mapping[str, int] = field(default_factory=dict)
    strings_fell_back: int = 0

    @classmethod
    def from_block_report(
        cls,
        report: BlockReport,
        *,
        profile_id_contract: ProfileIdAvailabilityContract | None = None,
    ) -> "SearchSignalState":
        profile_id_contract = (
            profile_id_contract or UNVERIFIED_PROFILE_ID_AVAILABILITY_CONTRACT
        )
        candidates_seen = 0
        duplicates_seen = 0
        facial_yes = 0
        facial_no = 0
        edge_case_saves = 0
        canonical_saves = 0
        employer_counts: dict[str, int] = {}
        profile_ids: set[str] = set()
        facet_values_requested: dict[str, int] = {}
        facet_values_applied: dict[str, int] = {}
        strings_fell_back = 0

        for detail in report.string_details or []:
            receipt = detail.get("surface_receipt")
            if isinstance(receipt, Mapping) and receipt:
                for dim, count in (receipt.get("requested_value_counts") or {}).items():
                    facet_values_requested[str(dim)] = (
                        facet_values_requested.get(str(dim), 0) + _int_value(count)
                    )
                for dim, count in (receipt.get("applied_value_counts") or {}).items():
                    facet_values_applied[str(dim)] = (
                        facet_values_applied.get(str(dim), 0) + _int_value(count)
                    )
                if receipt.get("fell_back_to_keyword"):
                    strings_fell_back += 1
            candidates_seen += _int_value(detail.get("candidates"))
            duplicates_seen += _int_value(detail.get("duplicates"))
            yes = _int_value(detail.get("facial_yes"))
            no = _int_value(detail.get("facial_no"))
            facial_yes += yes
            facial_no += no
            saves = _int_value(detail.get("saves"))
            if detail.get("novelty_bucket") == "edge_case":
                edge_case_saves += saves
            elif detail.get("novelty_bucket") == "canonical":
                canonical_saves += saves
            for profile in detail.get("saved_profiles") or ():
                if isinstance(profile, dict):
                    company = str(profile.get("company") or "").strip()
                    if company:
                        employer_counts[company] = employer_counts.get(company, 0) + 1
                    profile_id = str(
                        profile.get("profile_id")
                        or profile.get("member_id")
                        or profile.get("urn")
                        or ""
                    ).strip()
                    if profile_id:
                        profile_ids.add(profile_id)

        if not candidates_seen:
            candidates_seen = facial_yes + facial_no

        return cls(
            block_name=report.block_name,
            strings_run=report.strings_run,
            total_results=report.total_results,
            total_saves=report.total_saves,
            candidates_seen=candidates_seen,
            duplicates_seen=duplicates_seen,
            facial_yes=facial_yes,
            facial_no=facial_no,
            edge_case_saves=edge_case_saves,
            canonical_saves=canonical_saves,
            zero_save_string_ids=tuple(report.zero_save_string_ids or ()),
            employer_counts=employer_counts,
            overlap_available=profile_id_contract.is_verified and bool(profile_ids),
            overlap_rate=None,
            profile_id_sample_size=len(profile_ids),
            profile_id_availability=profile_id_contract,
            facet_values_requested=facet_values_requested,
            facet_values_applied=facet_values_applied,
            strings_fell_back=strings_fell_back,
        )

    @property
    def triage_pass_interval(self) -> WilsonInterval:
        total = self.facial_yes + self.facial_no
        return WilsonInterval(successes=self.facial_yes, total=total)

    @property
    def novelty_mix(self) -> dict[str, int]:
        return {
            "edge_case_saves": self.edge_case_saves,
            "canonical_saves": self.canonical_saves,
        }

    @property
    def employer_concentration(self) -> float:
        if not self.employer_counts:
            return 0.0
        total = sum(self.employer_counts.values())
        if total <= 0:
            return 0.0
        return max(self.employer_counts.values()) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_name": self.block_name,
            "strings_run": self.strings_run,
            "total_results": self.total_results,
            "total_saves": self.total_saves,
            "candidates_seen": self.candidates_seen,
            "duplicates_seen": self.duplicates_seen,
            "facial_yes": self.facial_yes,
            "facial_no": self.facial_no,
            "triage_pass_rate": self.triage_pass_interval.to_dict(),
            "novelty_mix": self.novelty_mix,
            "zero_save_string_ids": list(self.zero_save_string_ids),
            "employer_counts": dict(self.employer_counts),
            "employer_concentration": self.employer_concentration,
            "overlap": {
                "available": self.overlap_available,
                "rate": self.overlap_rate,
                "profile_id_sample_size": self.profile_id_sample_size,
                "profile_id_availability": self.profile_id_availability.to_dict(),
            },
            "structured_actuator": {
                "facet_values_requested": dict(self.facet_values_requested),
                "facet_values_applied": dict(self.facet_values_applied),
                "strings_fell_back_to_keyword": self.strings_fell_back,
            },
        }


@dataclass(frozen=True)
class AdaptationGateConfig:
    min_strings: int
    min_candidates_seen: int
    min_results_seen: int
    cooldown_blocks_remaining: int = 0
    allow_autonomous_reset: bool = False
    sprt_lower: float | None = None
    sprt_upper: float | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.min_strings, field="min_strings")
        _require_non_negative_int(
            self.min_candidates_seen,
            field="min_candidates_seen",
        )
        _require_non_negative_int(self.min_results_seen, field="min_results_seen")
        _require_non_negative_int(
            self.cooldown_blocks_remaining,
            field="cooldown_blocks_remaining",
        )
        if not isinstance(self.allow_autonomous_reset, bool):
            raise AdaptationValidationError(
                "allow_autonomous_reset must be a boolean"
            )
        _optional_probability(self.sprt_lower, field="sprt_lower")
        _optional_probability(self.sprt_upper, field="sprt_upper")
        if (
            self.sprt_lower is not None
            and self.sprt_upper is not None
            and self.sprt_lower > self.sprt_upper
        ):
            raise AdaptationValidationError(
                "sprt_lower cannot exceed sprt_upper"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_strings": self.min_strings,
            "min_candidates_seen": self.min_candidates_seen,
            "min_results_seen": self.min_results_seen,
            "cooldown_blocks_remaining": self.cooldown_blocks_remaining,
            "allow_autonomous_reset": self.allow_autonomous_reset,
            "sprt_lower": self.sprt_lower,
            "sprt_upper": self.sprt_upper,
        }


def default_adaptation_gate_config() -> AdaptationGateConfig:
    """Return the local, uncalibrated M3 adaptation gate defaults.

    This only asserts that some signal exists before spending an adaptation call.
    SPRT thresholds and real sufficiency calibration remain explicit inputs.
    """

    return AdaptationGateConfig(
        min_strings=1,
        min_candidates_seen=1,
        min_results_seen=1,
    )


@dataclass(frozen=True)
class AdaptationGateResult:
    decision: AdaptationGateDecision
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class AdaptedStringFirewallTrace:
    reports: tuple[BooleanNormalizationReport, ...]
    # P5 (Wave 2): strings the ubiquity gate refused, removed from the batch
    # per-item instead of failing the whole adaptation. A batch-wide raise
    # here reopened the 2026-06-18 regression class (one bad new_string
    # voided skip_remaining/reorder/pivot from the same checkpoint) the
    # moment the gate got a live term feed.
    dropped: tuple[dict[str, Any], ...] = ()

    @property
    def passed(self) -> bool:
        return not any(
            finding.code == "ubiquitous_and_gate"
            for report in self.reports
            for finding in report.findings
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reports": [report.to_dict() for report in self.reports],
            "dropped": [dict(item) for item in self.dropped],
        }


def render_signal_state_for_prompt(state: SearchSignalState) -> str:
    return "## Typed SearchSignalState\n" + _jsonish(state.to_dict())


def coerce_market_signal_prior(value: Any) -> MarketSignalPrior:
    if value is None:
        return MarketSignalPrior.empty()
    if isinstance(value, MarketSignalPrior):
        return value
    if isinstance(value, Mapping):
        return MarketSignalPrior.from_mapping(value)
    return MarketSignalPrior.from_advisory_context(str(value))


def render_market_signal_prior_for_prompt(prior: MarketSignalPrior) -> str:
    if prior.is_empty:
        return ""
    return "## Typed MarketSignalPrior\n" + _jsonish(prior.to_dict())


def evaluate_adaptation_gate(
    state: SearchSignalState,
    config: AdaptationGateConfig,
) -> AdaptationGateResult:
    reasons: list[str] = []
    if config.cooldown_blocks_remaining > 0:
        return AdaptationGateResult(
            decision=AdaptationGateDecision.COOLDOWN,
            reasons=(f"cooldown_blocks_remaining={config.cooldown_blocks_remaining}",),
        )
    if state.strings_run < config.min_strings:
        reasons.append(f"strings_run {state.strings_run} < min_strings {config.min_strings}")
    if state.candidates_seen < config.min_candidates_seen:
        reasons.append(
            f"candidates_seen {state.candidates_seen} < min_candidates_seen {config.min_candidates_seen}"
        )
    if state.total_results < config.min_results_seen:
        reasons.append(
            f"total_results {state.total_results} < min_results_seen {config.min_results_seen}"
        )
    if reasons:
        return AdaptationGateResult(
            decision=AdaptationGateDecision.COLLECT_MORE_SIGNAL,
            reasons=tuple(reasons),
        )
    interval = state.triage_pass_interval
    if config.sprt_lower is not None and interval.upper < config.sprt_lower:
        if not config.allow_autonomous_reset:
            return AdaptationGateResult(
                decision=AdaptationGateDecision.RESET_BLOCKED,
                reasons=("autonomous reset requires product approval",),
            )
    if config.sprt_upper is not None and interval.lower > config.sprt_upper:
        reasons.append("triage pass interval exceeds upper threshold")
    return AdaptationGateResult(
        decision=AdaptationGateDecision.ADAPT,
        reasons=tuple(reasons) or ("sufficiency thresholds met",),
    )


def apply_adapted_string_firewall(
    new_strings: list[dict[str, Any]],
    *,
    structured_filters: dict[str, Any] | None = None,
    ubiquitous_terms: set[str] | frozenset[str] | None = None,
    enable_token_subset_pruning: bool = True,
) -> AdaptedStringFirewallTrace:
    if structured_filters is not None and not isinstance(structured_filters, Mapping):
        raise AdaptationValidationError("structured_filters must be an object")
    reports: list[BooleanNormalizationReport] = []
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for item in new_strings:
        boolean = str(item.get("boolean") or "")
        if structured_filters is not None:
            item_structured_filters = structured_filters
        elif "structured_filters" in item:
            raw_structured_filters = item.get("structured_filters")
            if raw_structured_filters is None:
                item_structured_filters = None
            elif isinstance(raw_structured_filters, Mapping):
                item_structured_filters = raw_structured_filters
            else:
                raise AdaptationValidationError(
                    "adapted string structured_filters must be an object"
                )
        else:
            item_structured_filters = None
        try:
            report = normalize_boolean_for_linkedin(
                boolean,
                structured_filters=item_structured_filters,
                ubiquitous_terms=ubiquitous_terms,
                enable_token_subset_pruning=enable_token_subset_pruning,
            )
        except BooleanNormalizationError as exc:
            raise AdaptationValidationError(
                f"Adapted string failed the M1C normalizer: {exc}"
            ) from exc
        if any(finding.code == "ubiquitous_and_gate" for finding in report.findings):
            # Fail closed PER STRING: refuse this string, keep the rest of
            # the adaptation decision (skips, reorders, healthy siblings).
            # The old batch-wide raise here was structurally dead until the
            # gate got a live term feed; live, it voided whole checkpoints.
            dropped.append(
                {
                    "boolean": boolean,
                    "rationale": str(item.get("rationale") or ""),
                    "family_key": str(item.get("family_key") or ""),
                    "code": "ubiquitous_and_gate",
                    "message": "AND clause is composed entirely of ubiquitous terms.",
                }
            )
            continue
        item["boolean"] = report.normalized_boolean
        item["boolean_normalization"] = report.to_dict()
        kept.append(item)
        reports.append(report)
    new_strings[:] = kept
    return AdaptedStringFirewallTrace(reports=tuple(reports), dropped=tuple(dropped))


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


_FIELD_MISSING = object()


def _first_present_value(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return _FIELD_MISSING


def _optional_string_field(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: str | None,
    field: str,
) -> str | None:
    value = _first_present_value(payload, keys)
    if value is _FIELD_MISSING or value is None:
        return default
    if not isinstance(value, str):
        raise AdaptationValidationError(f"{field} must be a string")
    stripped = value.strip()
    return stripped or default


def _required_string_field(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    *,
    field: str,
    missing_message: str,
) -> str:
    value = _first_present_value(payload, keys)
    if value is _FIELD_MISSING or value is None:
        raise AdaptationValidationError(missing_message)
    if not isinstance(value, str):
        raise AdaptationValidationError(f"{field} must be a string")
    stripped = value.strip()
    if not stripped:
        raise AdaptationValidationError(missing_message)
    return stripped


def _optional_string_sequence_field(value: Any, *, field: str) -> tuple[str, ...]:
    if value is _FIELD_MISSING or value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        raise AdaptationValidationError(f"{field} must be an array or string")
    refs: list[str] = []
    for ref in value:
        if not isinstance(ref, str):
            raise AdaptationValidationError(f"{field} must contain strings")
        stripped = ref.strip()
        if stripped:
            refs.append(stripped)
    return tuple(refs)


def _optional_confidence_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        raise AdaptationValidationError("market signal confidence must be numeric")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptationValidationError("market signal confidence must be numeric")
    confidence_value = float(value)
    if not math.isfinite(confidence_value):
        raise AdaptationValidationError("market signal confidence must be numeric")
    return max(0.0, min(1.0, confidence_value))


def _require_non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdaptationValidationError(f"{field} must be a non-negative integer")
    return value


def _optional_probability(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptationValidationError(f"{field} must be a number between 0 and 1")
    probability = float(value)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise AdaptationValidationError(f"{field} must be a number between 0 and 1")
    return probability


def _require_iso_timestamp(value: str, *, field: str) -> None:
    try:
        datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdaptationValidationError(f"{field} must be an ISO timestamp") from exc


def _jsonish(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


def _parse_market_advisory_line(line: str) -> MarketSignal | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("-"):
        text = text[1:].strip()
    signal_type = "note"
    if text.startswith("[") and "]" in text:
        raw_type, rest = text[1:].split("]", 1)
        signal_type = raw_type.strip() or "note"
        text = rest.strip(" :-")
    if not text:
        return None
    return MarketSignal(signal_type=signal_type, recommendation=text)
