"""Fixture-friendly continuous verification for LinkedIn connector contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from linkedin.matching_contract import (
    LinkedInMatchingContract,
    UNVERIFIED_LINKEDIN_MATCHING_CONTRACT,
)


DEFAULT_PROBE_CADENCE_DAYS = 7
ALLOWED_COLD_PATH_RUNNER_OUTCOME_KEYS = frozenset(
    ("status", "evidence_ref", "error")
)
ALLOWED_COLD_PATH_RUNNER_OUTCOME_STATUSES = frozenset(("ok", "error"))


@dataclass(frozen=True)
class AccessibilityNodeSnapshot:
    selector: str
    role: str
    name: str

    @classmethod
    def from_value(
        cls,
        selector: str,
        value: "AccessibilityNodeSnapshot | Mapping[str, Any]",
    ) -> "AccessibilityNodeSnapshot":
        if isinstance(value, AccessibilityNodeSnapshot):
            return cls(
                selector=selector,
                role=_optional_snapshot_string(value.role, field="role"),
                name=_optional_snapshot_string(value.name, field="name"),
            )
        if not isinstance(value, Mapping):
            raise ValueError("accessibility snapshot must be an object")
        return cls(
            selector=selector,
            role=_optional_snapshot_string(value.get("role"), field="role"),
            name=_optional_snapshot_string(value.get("name"), field="name"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"selector": self.selector, "role": self.role, "name": self.name}


@dataclass(frozen=True)
class AccessibilityDrift:
    selector: str
    status: str
    baseline: AccessibilityNodeSnapshot | None
    current: AccessibilityNodeSnapshot | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "status": self.status,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "current": self.current.to_dict() if self.current else None,
        }


@dataclass(frozen=True)
class AccessibilityDriftReport:
    generated_at: str
    cadence_days: int
    drifts: tuple[AccessibilityDrift, ...]
    alert_only: bool = True

    @property
    def status(self) -> str:
        return "alert" if self.drifts else "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "generated_at": self.generated_at,
            "cadence_days": self.cadence_days,
            "alert_only": self.alert_only,
            "drifts": [drift.to_dict() for drift in self.drifts],
        }


@dataclass(frozen=True)
class ColdPathProbe:
    name: str
    max_silence_days: int
    last_ran_at: str | None = None
    requires_live_seat: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_silence_days": self.max_silence_days,
            "last_ran_at": self.last_ran_at,
            "requires_live_seat": self.requires_live_seat,
        }


ColdPathRunner = Callable[[ColdPathProbe], Mapping[str, Any]]


@dataclass(frozen=True)
class ColdPathProbeResult:
    name: str
    status: str
    days_since_last_run: int | None
    max_silence_days: int
    requires_live_seat: bool
    was_due: bool = False
    ran_at: str | None = None
    evidence_ref: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "days_since_last_run": self.days_since_last_run,
            "max_silence_days": self.max_silence_days,
            "requires_live_seat": self.requires_live_seat,
            "was_due": self.was_due,
            "ran_at": self.ran_at,
            "evidence_ref": self.evidence_ref,
            "error": self.error,
        }


@dataclass(frozen=True)
class ColdPathRegistryReport:
    generated_at: str
    results: tuple[ColdPathProbeResult, ...]
    alert_only: bool = True

    @property
    def status(self) -> str:
        return (
            "alert"
            if any(result.status != "ok" for result in self.results)
            else "ok"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "generated_at": self.generated_at,
            "alert_only": self.alert_only,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class MatchingContractFreshnessReport:
    generated_at: str
    status: str
    last_empirically_verified: str | None
    max_age_days: int
    age_days: int | None
    alert_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "last_empirically_verified": self.last_empirically_verified,
            "max_age_days": self.max_age_days,
            "age_days": self.age_days,
            "alert_only": self.alert_only,
        }


@dataclass(frozen=True)
class _ValidatedColdPathProbe:
    name: str
    max_silence_days: int
    last_ran_at: datetime | None
    requires_live_seat: bool
    error: str | None = None


def diff_accessibility_tree(
    baseline: Mapping[str, AccessibilityNodeSnapshot | Mapping[str, Any]],
    current: Mapping[str, AccessibilityNodeSnapshot | Mapping[str, Any]],
    *,
    generated_at: datetime | str | None = None,
    cadence_days: int = DEFAULT_PROBE_CADENCE_DAYS,
) -> AccessibilityDriftReport:
    """Compare selector snapshots and alert on role/name drift."""

    generated = _format_datetime(generated_at, field="generated_at")
    cadence_days = _require_non_negative_int(cadence_days, field="cadence_days")
    normalized_baseline = _normalize_accessibility_snapshot_map(
        baseline,
        field="baseline",
    )
    normalized_current = _normalize_accessibility_snapshot_map(
        current,
        field="current",
    )
    drifts: list[AccessibilityDrift] = []
    for selector in sorted(normalized_baseline):
        baseline_node = normalized_baseline[selector]
        current_node = normalized_current.get(selector)
        if current_node is None:
            drifts.append(
                AccessibilityDrift(
                    selector=selector,
                    status="missing",
                    baseline=baseline_node,
                    current=None,
                )
            )
            continue
        if baseline_node.role != current_node.role:
            drifts.append(
                AccessibilityDrift(
                    selector=selector,
                    status="role_changed",
                    baseline=baseline_node,
                    current=current_node,
                )
            )
            continue
        if baseline_node.name != current_node.name:
            drifts.append(
                AccessibilityDrift(
                    selector=selector,
                    status="name_changed",
                    baseline=baseline_node,
                    current=current_node,
                )
            )
    return AccessibilityDriftReport(
        generated_at=generated,
        cadence_days=cadence_days,
        drifts=tuple(drifts),
    )


def _normalize_accessibility_snapshot_map(
    snapshots: Mapping[str, AccessibilityNodeSnapshot | Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, AccessibilityNodeSnapshot]:
    if not isinstance(snapshots, Mapping):
        raise ValueError(f"accessibility {field} snapshot map must be an object")
    normalized: dict[str, AccessibilityNodeSnapshot] = {}
    for raw_selector, value in snapshots.items():
        if not isinstance(raw_selector, str):
            raise ValueError(f"accessibility {field} selector must be a string")
        selector = raw_selector.strip()
        if not selector:
            raise ValueError(f"accessibility {field} selector must be non-empty")
        normalized[selector] = AccessibilityNodeSnapshot.from_value(selector, value)
    return normalized


def default_cold_path_probe_schedule(
    *,
    last_ran_at: str | None = None,
    max_silence_days: int = DEFAULT_PROBE_CADENCE_DAYS,
) -> tuple[ColdPathProbe, ...]:
    return (
        ColdPathProbe("save", max_silence_days, last_ran_at),
        ColdPathProbe("drawer", max_silence_days, last_ran_at),
        ColdPathProbe("pagination", max_silence_days, last_ran_at),
        ColdPathProbe("fallback", max_silence_days, last_ran_at),
    )


def evaluate_cold_path_registry(
    probes: Sequence[ColdPathProbe],
    *,
    now: datetime | str | None = None,
    alert_only: bool = True,
) -> ColdPathRegistryReport:
    generated = _coerce_datetime(now, field="now")
    alert_only = _require_bool(alert_only, field="alert_only")
    results: list[ColdPathProbeResult] = []
    for probe in probes:
        validated = _validate_cold_path_probe(probe)
        if validated.error:
            results.append(
                ColdPathProbeResult(
                    name=validated.name,
                    status="invalid_probe",
                    days_since_last_run=None,
                    max_silence_days=validated.max_silence_days,
                    requires_live_seat=validated.requires_live_seat,
                    was_due=True,
                    error=validated.error,
                )
            )
            continue
        last_ran = validated.last_ran_at
        days_since = (generated - last_ran).days if last_ran else None
        if validated.requires_live_seat:
            status = "pending_live_seat"
        elif days_since is None:
            status = "never_run"
        elif days_since > validated.max_silence_days:
            status = "stale"
        else:
            status = "ok"
        results.append(
            ColdPathProbeResult(
                name=validated.name,
                status=status,
                days_since_last_run=days_since,
                max_silence_days=validated.max_silence_days,
                requires_live_seat=validated.requires_live_seat,
                was_due=status in {"never_run", "stale", "pending_live_seat"},
            )
        )
    return ColdPathRegistryReport(
        generated_at=_format_datetime(generated),
        results=tuple(results),
        alert_only=alert_only,
    )


def run_due_cold_path_probes(
    probes: Sequence[ColdPathProbe],
    runners: Mapping[str, ColdPathRunner],
    *,
    now: datetime | str | None = None,
    alert_only: bool = True,
) -> ColdPathRegistryReport:
    """Run due cold-path probes against supplied fixtures.

    This does not drive LinkedIn. Live-seat probes stay pending until the final
    live bucket supplies external evidence. Fixture runners must return an
    ``evidence_ref`` for an OK result so a green report remains evidence-backed.
    """

    generated = _coerce_datetime(now, field="now")
    alert_only = _require_bool(alert_only, field="alert_only")
    generated_at = _format_datetime(generated)
    results: list[ColdPathProbeResult] = []
    for probe in probes:
        validated = _validate_cold_path_probe(probe)
        if validated.error:
            results.append(
                ColdPathProbeResult(
                    name=validated.name,
                    status="invalid_probe",
                    days_since_last_run=None,
                    max_silence_days=validated.max_silence_days,
                    requires_live_seat=validated.requires_live_seat,
                    was_due=True,
                    ran_at=generated_at,
                    error=validated.error,
                )
            )
            continue
        last_ran = validated.last_ran_at
        days_since = (generated - last_ran).days if last_ran else None
        was_due = days_since is None or days_since > validated.max_silence_days
        if validated.requires_live_seat:
            status = "pending_live_seat" if was_due else "ok"
            results.append(
                ColdPathProbeResult(
                    name=validated.name,
                    status=status,
                    days_since_last_run=days_since,
                    max_silence_days=validated.max_silence_days,
                    requires_live_seat=True,
                    was_due=was_due,
                )
            )
            continue
        if not was_due:
            results.append(
                ColdPathProbeResult(
                    name=validated.name,
                    status="ok",
                    days_since_last_run=days_since,
                    max_silence_days=validated.max_silence_days,
                    requires_live_seat=False,
                    was_due=False,
                )
            )
            continue

        runner = runners.get(validated.name)
        if runner is None:
            results.append(
                ColdPathProbeResult(
                    name=validated.name,
                    status="missing_runner",
                    days_since_last_run=days_since,
                    max_silence_days=validated.max_silence_days,
                    requires_live_seat=False,
                    was_due=True,
                    ran_at=generated_at,
                )
            )
            continue

        try:
            outcome = _coerce_cold_path_runner_outcome(runner(probe))
        except Exception as exc:  # pragma: no cover - exercised by unit test.
            results.append(
                ColdPathProbeResult(
                    name=validated.name,
                    status="error",
                    days_since_last_run=days_since,
                    max_silence_days=validated.max_silence_days,
                    requires_live_seat=False,
                    was_due=True,
                    ran_at=generated_at,
                    error=exc.__class__.__name__,
                )
            )
            continue

        status = outcome["status"]
        evidence_ref = outcome["evidence_ref"]
        if status == "ok" and not evidence_ref:
            status = "missing_evidence"
        results.append(
            ColdPathProbeResult(
                name=validated.name,
                status=status,
                days_since_last_run=days_since,
                max_silence_days=validated.max_silence_days,
                requires_live_seat=False,
                was_due=True,
                ran_at=generated_at,
                evidence_ref=evidence_ref or None,
                error=outcome["error"] or None,
            )
        )

    return ColdPathRegistryReport(
        generated_at=generated_at,
        results=tuple(results),
        alert_only=alert_only,
    )


def _validate_cold_path_probe(probe: ColdPathProbe) -> _ValidatedColdPathProbe:
    errors: list[str] = []

    if isinstance(probe.name, str) and probe.name.strip():
        name = probe.name.strip()
    else:
        name = "<invalid>"
        errors.append("name must be a non-empty string")

    if (
        isinstance(probe.max_silence_days, int)
        and not isinstance(probe.max_silence_days, bool)
        and probe.max_silence_days >= 0
    ):
        max_silence_days = probe.max_silence_days
    else:
        max_silence_days = 0
        errors.append("max_silence_days must be a non-negative integer")

    if isinstance(probe.requires_live_seat, bool):
        requires_live_seat = probe.requires_live_seat
    else:
        requires_live_seat = False
        errors.append("requires_live_seat must be a boolean")

    last_ran_at: datetime | None = None
    if probe.last_ran_at is not None:
        if not isinstance(probe.last_ran_at, str):
            errors.append("last_ran_at must be a string ISO timestamp")
        else:
            try:
                last_ran_at = _coerce_datetime(
                    probe.last_ran_at,
                    field="last_ran_at",
                )
            except ValueError:
                errors.append("last_ran_at must be an ISO timestamp")

    return _ValidatedColdPathProbe(
        name=name,
        max_silence_days=max_silence_days,
        last_ran_at=last_ran_at,
        requires_live_seat=requires_live_seat,
        error="; ".join(errors) or None,
    )


def evaluate_matching_contract_freshness(
    contract: LinkedInMatchingContract | Mapping[str, Any] | None = None,
    *,
    now: datetime | str | None = None,
    max_age_days: int = 30,
    alert_only: bool = True,
) -> MatchingContractFreshnessReport:
    checked_at = _coerce_datetime(now, field="now")
    max_age_days = _require_non_negative_int(max_age_days, field="max_age_days")
    alert_only = _require_bool(alert_only, field="alert_only")
    last_verified = _last_empirically_verified(
        contract or UNVERIFIED_LINKEDIN_MATCHING_CONTRACT
    )
    if not last_verified:
        return MatchingContractFreshnessReport(
            generated_at=_format_datetime(checked_at),
            status="unverified",
            last_empirically_verified=None,
            max_age_days=max_age_days,
            age_days=None,
            alert_only=alert_only,
        )

    try:
        verified_at = _coerce_datetime(
            last_verified,
            field="last_empirically_verified",
        )
    except (AttributeError, ValueError):
        return MatchingContractFreshnessReport(
            generated_at=_format_datetime(checked_at),
            status="invalid",
            last_empirically_verified=last_verified,
            max_age_days=max_age_days,
            age_days=None,
            alert_only=alert_only,
        )
    age_days = (checked_at - verified_at).days
    status = "stale" if age_days > max_age_days else "fresh"
    return MatchingContractFreshnessReport(
        generated_at=_format_datetime(checked_at),
        status=status,
        last_empirically_verified=last_verified,
        max_age_days=max_age_days,
        age_days=age_days,
        alert_only=alert_only,
    )


def _last_empirically_verified(
    contract: LinkedInMatchingContract | Mapping[str, Any],
) -> str | None:
    if isinstance(contract, LinkedInMatchingContract):
        return contract.last_empirically_verified
    value = contract.get("last_empirically_verified")
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _coerce_cold_path_runner_outcome(
    value: Any,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return _invalid_cold_path_runner_outcome("outcome must be an object")
    unexpected_keys: list[str] = []
    for key in value:
        if not isinstance(key, str):
            return _invalid_cold_path_runner_outcome(
                f"outcome key {key!r} must be a string"
            )
        if key not in ALLOWED_COLD_PATH_RUNNER_OUTCOME_KEYS:
            unexpected_keys.append(key)
    if unexpected_keys:
        return _invalid_cold_path_runner_outcome(
            "outcome has unexpected keys: " + ", ".join(sorted(unexpected_keys))
        )
    if "status" not in value:
        return _invalid_cold_path_runner_outcome("status is required")
    status_value = value["status"]
    if isinstance(status_value, str):
        status = status_value.strip()
    else:
        return _invalid_cold_path_runner_outcome("status must be a string")
    if not status:
        return _invalid_cold_path_runner_outcome("status must be a non-empty string")
    if status not in ALLOWED_COLD_PATH_RUNNER_OUTCOME_STATUSES:
        return _invalid_cold_path_runner_outcome(
            "status must be one of: "
            + ", ".join(sorted(ALLOWED_COLD_PATH_RUNNER_OUTCOME_STATUSES))
        )

    evidence_value = value.get("evidence_ref")
    if evidence_value is None:
        evidence_ref = ""
    elif isinstance(evidence_value, str):
        evidence_ref = evidence_value.strip()
    else:
        return _invalid_cold_path_runner_outcome("evidence_ref must be a string")

    error_value = value.get("error")
    if error_value is None:
        error = ""
    elif isinstance(error_value, str):
        error = error_value.strip()
    else:
        return _invalid_cold_path_runner_outcome("error must be a string")
    return {"status": status, "evidence_ref": evidence_ref, "error": error}


def _invalid_cold_path_runner_outcome(error: str) -> dict[str, str]:
    return {"status": "invalid_outcome", "evidence_ref": "", "error": error}


def _optional_snapshot_string(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"accessibility snapshot {field} must be a string")
    return value.strip()


def _require_non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError(f"{field} must be a non-negative integer")


def _require_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a boolean")


def _coerce_datetime(
    value: datetime | str | None,
    *,
    field: str = "timestamp",
) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string ISO timestamp or datetime")
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _format_datetime(
    value: datetime | str | None,
    *,
    field: str = "timestamp",
) -> str:
    return _coerce_datetime(value, field=field).astimezone(timezone.utc).isoformat()
