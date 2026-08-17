"""Final LinkedIn live-evidence gate for the sourcing quality kernel.

This module does not run LinkedIn or infer external behavior. It validates
evidence supplied after live Recruiter seat tests and reports exactly which
goal gates remain pending.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from linkedin.adaptation_signal_state import (
    ProfileIdAvailabilityContract,
)
from linkedin.matching_contract import (
    SeatTestEvidenceError,
    build_verified_contract_from_seat_test_counts,
    required_linkedin_matching_seat_tests,
)


FINAL_LIVE_BUCKET_SCHEMA_VERSION = "linkedin.final_live_bucket.v1"
REQUIRED_COLD_PATHS = ("save", "drawer", "pagination", "fallback")
STALE_MATCHING_CONTRACT_POLICIES = ("alert_only", "deploy_blocking")
PLACEHOLDER_EVIDENCE_REF_VALUES = frozenset(
    (
        "live evidence artifact id",
        "placeholder",
        "todo",
        "tbd",
        "n/a",
        "na",
        "none",
        "null",
        "example",
        "sample",
        "dummy",
        "fake",
        "missing",
        "missing evidence",
        "not applicable",
        "pending",
        "unknown",
    )
)
PLACEHOLDER_EVIDENCE_REF_PREFIXES = (
    "dummy",
    "example",
    "fake",
    "missing",
    "placeholder",
    "sample",
    "todo",
    "tbd",
)
PLACEHOLDER_EVIDENCE_REF_SEPARATORS = frozenset((" ", "-", "_", ":", "/", ".", "#"))
ALLOWED_COLD_PATH_RESULT_STATUSES = frozenset(
    (
        "ok",
        "error",
        "invalid_outcome",
        "invalid_probe",
        "missing_evidence",
        "missing_runner",
        "never_run",
        "pending_live_seat",
        "stale",
    )
)
ALLOWED_FINAL_LIVE_BUCKET_KEYS = frozenset(
    (
        "schema_version",
        "verified_at",
        "matching_counts",
        "matching_queries",
        "matching_counts_evidence_ref",
        "profile_id_probe",
        "cold_path_results",
        "stale_matching_contract_policy",
    )
)
ALLOWED_COLD_PATH_RESULT_KEYS = frozenset(
    ("name", "status", "ran_at", "evidence_ref")
)
ALLOWED_PROFILE_ID_PROBE_KEYS = frozenset(
    ("stable_profile_id_seen", "sample_size", "evidence_ref", "verified_at")
)


class FinalLiveBucketError(ValueError):
    """Raised when final live evidence is invalid or incomplete."""


@dataclass(frozen=True)
class FinalLiveBucketReport:
    status: str
    pending_gates: tuple[str, ...]
    errors: tuple[str, ...]
    matching_contract: Mapping[str, Any] | None
    profile_id_availability: Mapping[str, Any] | None
    cold_path_results: tuple[Mapping[str, Any], ...]
    stale_matching_contract_policy: str | None
    schema_version: str = FINAL_LIVE_BUCKET_SCHEMA_VERSION

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "pending_gates": list(self.pending_gates),
            "errors": list(self.errors),
            "matching_contract": (
                dict(self.matching_contract) if self.matching_contract else None
            ),
            "profile_id_availability": (
                dict(self.profile_id_availability)
                if self.profile_id_availability
                else None
            ),
            "cold_path_results": [
                dict(result) for result in self.cold_path_results
            ],
            "stale_matching_contract_policy": self.stale_matching_contract_policy,
        }


def validate_final_linkedin_live_bucket(
    payload: Any,
) -> FinalLiveBucketReport:
    """Validate supplied live evidence for the final Sourcing Quality Kernel gate.

    Missing evidence is reported as ``pending``. Incoherent evidence is reported
    as ``invalid``. Only complete, coherent evidence returns ``complete``.
    """

    if not isinstance(payload, Mapping):
        return FinalLiveBucketReport(
            status="invalid",
            pending_gates=(),
            errors=("payload must be an object",),
            matching_contract=None,
            profile_id_availability=None,
            cold_path_results=(),
            stale_matching_contract_policy=None,
        )

    pending: list[str] = []
    errors: list[str] = []
    envelope_errors_before = len(errors)
    _validate_allowed_keys(
        payload,
        allowed=ALLOWED_FINAL_LIVE_BUCKET_KEYS,
        field="payload",
        errors=errors,
    )
    _validate_schema_version(payload, pending=pending, errors=errors)
    if len(errors) > envelope_errors_before:
        return FinalLiveBucketReport(
            status="invalid",
            pending_gates=(),
            errors=tuple(errors),
            matching_contract=None,
            profile_id_availability=None,
            cold_path_results=(),
            stale_matching_contract_policy=None,
        )
    if "schema_version" in pending:
        return FinalLiveBucketReport(
            status="pending",
            pending_gates=tuple(sorted(set(pending))),
            errors=(),
            matching_contract=None,
            profile_id_availability=None,
            cold_path_results=(),
            stale_matching_contract_policy=None,
        )
    verified_at = _optional_timestamp(
        payload.get("verified_at"),
        field="verified_at",
        errors=errors,
    )

    matching_contract = _validate_matching_counts(
        payload,
        verified_at=verified_at,
        pending=pending,
        errors=errors,
    )
    profile_id_availability = _validate_profile_id_probe(
        payload,
        verified_at=verified_at,
        pending=pending,
        errors=errors,
    )
    cold_path_results = _validate_cold_path_results(
        payload,
        pending=pending,
        errors=errors,
    )
    stale_policy = _validate_stale_policy(
        payload,
        pending=pending,
        errors=errors,
    )

    if errors:
        return FinalLiveBucketReport(
            status="invalid",
            pending_gates=(),
            errors=tuple(errors),
            matching_contract=None,
            profile_id_availability=None,
            cold_path_results=(),
            stale_matching_contract_policy=None,
        )
    if pending:
        status = "pending"
    else:
        status = "complete"
    return FinalLiveBucketReport(
        status=status,
        pending_gates=tuple(sorted(set(pending))),
        errors=tuple(errors),
        matching_contract=matching_contract,
        profile_id_availability=profile_id_availability,
        cold_path_results=tuple(cold_path_results),
        stale_matching_contract_policy=stale_policy,
    )


def require_final_linkedin_live_bucket_complete(
    payload: Any,
) -> FinalLiveBucketReport:
    """Return the report or raise if any final live gate is not proven."""

    report = validate_final_linkedin_live_bucket(payload)
    if report.is_complete:
        return report
    reasons = list(report.errors) or [
        f"pending final LinkedIn live gate: {gate}"
        for gate in report.pending_gates
    ]
    raise FinalLiveBucketError("; ".join(reasons))


def final_linkedin_live_bucket_payload_template() -> dict[str, Any]:
    """Return the canonical JSON-shaped payload template for live-seat evidence."""

    return {
        "schema_version": FINAL_LIVE_BUCKET_SCHEMA_VERSION,
        "verified_at": "<live verification timestamp>",
        "matching_counts": {
            spec.query_key: "<Recruiter result count>"
            for spec in required_linkedin_matching_seat_tests()
        },
        "matching_queries": {
            spec.query_key: spec.query
            for spec in required_linkedin_matching_seat_tests()
        },
        "matching_counts_evidence_ref": "<live evidence artifact id>",
        "profile_id_probe": {
            "stable_profile_id_seen": True,
            "sample_size": "<positive inspected-result count>",
            "evidence_ref": "<live evidence artifact id>",
        },
        "cold_path_results": [
            {
                "name": name,
                "status": "ok",
                "ran_at": "<live probe timestamp>",
                "evidence_ref": "<live evidence artifact id>",
            }
            for name in REQUIRED_COLD_PATHS
        ],
        "stale_matching_contract_policy": " | ".join(
            STALE_MATCHING_CONTRACT_POLICIES
        ),
    }


def _validate_matching_counts(
    payload: Mapping[str, Any],
    *,
    verified_at: str,
    pending: list[str],
    errors: list[str],
) -> Mapping[str, Any] | None:
    counts = payload.get("matching_counts")
    if counts is None:
        if "matching_queries" in payload:
            _validate_matching_queries(
                payload,
                pending=pending,
                errors=errors,
            )
        if "matching_counts_evidence_ref" in payload:
            evidence_ref = _optional_evidence_ref(
                payload.get("matching_counts_evidence_ref"),
                field="matching_counts_evidence_ref",
                errors=errors,
            )
            if not evidence_ref:
                pending.append("matching_counts_evidence_ref")
        pending.append("matching_counts")
        return None
    if not isinstance(counts, Mapping):
        errors.append("matching_counts must be an object")
        return None
    try:
        build_verified_contract_from_seat_test_counts(
            counts,
            verified_at="1970-01-01T00:00:00Z",
        )
    except (SeatTestEvidenceError, ValueError) as exc:
        errors.append(f"matching_counts invalid: {exc}")
        return None
    matching_queries = _validate_matching_queries(
        payload,
        pending=pending,
        errors=errors,
    )
    if matching_queries is None:
        return None
    evidence_ref = _optional_evidence_ref(
        payload.get("matching_counts_evidence_ref"),
        field="matching_counts_evidence_ref",
        errors=errors,
    )
    if not evidence_ref:
        pending.append("matching_counts_evidence_ref")
        return None
    if not verified_at:
        pending.append("verified_at")
        return None
    contract = build_verified_contract_from_seat_test_counts(
        counts,
        verified_at=verified_at,
    )
    payload = contract.to_dict()
    payload["evidence_ref"] = evidence_ref
    payload["matching_queries"] = dict(matching_queries)
    return payload


def _validate_matching_queries(
    payload: Mapping[str, Any],
    *,
    pending: list[str],
    errors: list[str],
) -> Mapping[str, str] | None:
    queries = payload.get("matching_queries")
    if queries is None:
        pending.append("matching_queries")
        return None
    if not isinstance(queries, Mapping):
        errors.append("matching_queries must be an object")
        return None
    required_queries = {
        spec.query_key: spec.query
        for spec in required_linkedin_matching_seat_tests()
    }
    for key in queries:
        if not isinstance(key, str):
            errors.append(f"matching_queries key {key!r} must be a string")
            return None
    missing = sorted(set(required_queries).difference(queries))
    if missing:
        pending.extend(f"matching_queries.{key}" for key in missing)
        return None
    unexpected = sorted(set(queries).difference(required_queries))
    if unexpected:
        errors.append("Unexpected matching query keys: " + ", ".join(unexpected))
        return None
    query_errors: list[str] = []
    for key, expected in required_queries.items():
        supplied = queries[key]
        if not isinstance(supplied, str):
            query_errors.append(f"matching_queries.{key} must be a string")
            continue
        if supplied.strip() != expected:
            query_errors.append(
                f"matching_queries.{key} must be {expected!r}"
            )
    if query_errors:
        errors.extend(query_errors)
        return None
    return required_queries


def _validate_profile_id_probe(
    payload: Mapping[str, Any],
    *,
    verified_at: str,
    pending: list[str],
    errors: list[str],
) -> Mapping[str, Any] | None:
    probe = payload.get("profile_id_probe")
    if probe is None:
        pending.append("profile_id_probe")
        return None
    if not isinstance(probe, Mapping):
        errors.append("profile_id_probe must be an object")
        return None
    if not _validate_allowed_keys(
        probe,
        allowed=ALLOWED_PROFILE_ID_PROBE_KEYS,
        field="profile_id_probe",
        errors=errors,
    ):
        return None

    sample_size = probe.get("sample_size")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        errors.append("profile_id_probe.sample_size must be an integer")
        return None
    if sample_size <= 0:
        errors.append("profile_id_probe.sample_size must be positive")
        return None

    stable_profile_id_seen = probe.get("stable_profile_id_seen")
    if not isinstance(stable_profile_id_seen, bool):
        errors.append("profile_id_probe.stable_profile_id_seen must be a boolean")
        return None
    if stable_profile_id_seen is not True:
        pending.append("profile_id_probe.stable_profile_id_seen")
        return None
    evidence_ref = _optional_evidence_ref(
        probe.get("evidence_ref"),
        field="profile_id_probe.evidence_ref",
        errors=errors,
    )
    if not evidence_ref:
        pending.append("profile_id_probe.evidence_ref")
        return None

    errors_before_timestamp = len(errors)
    probe_verified_at = _optional_timestamp(
        probe.get("verified_at"),
        field="profile_id_probe.verified_at",
        errors=errors,
    )
    if len(errors) > errors_before_timestamp:
        return None
    effective_verified_at = probe_verified_at or verified_at
    if not effective_verified_at:
        pending.append("verified_at")
        return None

    evidence = {
        "stable_profile_id_seen": stable_profile_id_seen,
        "sample_size": sample_size,
        "evidence_ref": evidence_ref,
    }
    if probe_verified_at:
        evidence["verified_at"] = probe_verified_at
    contract = ProfileIdAvailabilityContract.verified(
        evidence=evidence,
        verified_at=effective_verified_at,
    )
    return contract.to_dict()


def _validate_cold_path_results(
    payload: Mapping[str, Any],
    *,
    pending: list[str],
    errors: list[str],
) -> list[Mapping[str, Any]]:
    raw_results = payload.get("cold_path_results")
    if raw_results is None:
        pending.append("cold_path_results")
        return []
    if not isinstance(raw_results, list):
        errors.append("cold_path_results must be a list")
        return []

    by_name: dict[str, Mapping[str, Any]] = {}
    duplicate_names: set[str] = set()
    required_names = set(REQUIRED_COLD_PATHS)
    for index, result in enumerate(raw_results):
        if not isinstance(result, Mapping):
            errors.append(f"cold_path_results[{index}] must be an object")
            continue
        keys_ok = _validate_allowed_keys(
            result,
            allowed=ALLOWED_COLD_PATH_RESULT_KEYS,
            field=f"cold_path_results[{index}]",
            errors=errors,
        )
        name = _required_string(
            result.get("name"),
            field=f"cold_path_results[{index}].name",
            errors=errors,
        )
        if not name:
            errors.append(f"cold_path_results[{index}].name is required")
            continue
        if name not in required_names:
            errors.append(f"cold_path_results.{name} is not a required cold path")
            continue
        if not keys_ok:
            continue
        if name in duplicate_names:
            continue
        if name in by_name:
            errors.append(f"cold_path_results.{name} appears more than once")
            duplicate_names.add(name)
            by_name.pop(name, None)
            continue
        by_name[name] = dict(result)

    valid_results: list[Mapping[str, Any]] = []
    for name in REQUIRED_COLD_PATHS:
        if name in duplicate_names:
            continue
        result = by_name.get(name)
        if result is None:
            pending.append(f"cold_path_results.{name}")
            continue
        result_valid = True
        status = _required_string(
            result.get("status"),
            field=f"cold_path_results.{name}.status",
            errors=errors,
        )
        if status and status not in ALLOWED_COLD_PATH_RESULT_STATUSES:
            errors.append(
                f"cold_path_results.{name}.status must be one of: "
                + ", ".join(sorted(ALLOWED_COLD_PATH_RESULT_STATUSES))
            )
            result_valid = False
        if status != "ok":
            pending.append(f"cold_path_results.{name}.status")
            result_valid = False
        ran_at = _optional_timestamp(
            result.get("ran_at"),
            field=f"cold_path_results.{name}.ran_at",
            errors=errors,
        )
        if not ran_at:
            pending.append(f"cold_path_results.{name}.ran_at")
            result_valid = False
        evidence_ref = _optional_evidence_ref(
            result.get("evidence_ref"),
            field=f"cold_path_results.{name}.evidence_ref",
            errors=errors,
        )
        if not evidence_ref:
            pending.append(f"cold_path_results.{name}.evidence_ref")
            result_valid = False
        if result_valid:
            valid_results.append(
                {
                    "name": name,
                    "status": status,
                    "ran_at": ran_at,
                    "evidence_ref": evidence_ref,
                }
            )

    return valid_results


def _validate_stale_policy(
    payload: Mapping[str, Any],
    *,
    pending: list[str],
    errors: list[str],
) -> str | None:
    policy = _required_string(
        payload.get("stale_matching_contract_policy"),
        field="stale_matching_contract_policy",
        errors=errors,
    )
    if not policy:
        pending.append("stale_matching_contract_policy")
        return None
    if policy not in STALE_MATCHING_CONTRACT_POLICIES:
        errors.append(
            "stale_matching_contract_policy must be one of: "
            + ", ".join(STALE_MATCHING_CONTRACT_POLICIES)
        )
        return None
    return policy


def _validate_schema_version(
    payload: Mapping[str, Any],
    *,
    pending: list[str],
    errors: list[str],
) -> None:
    value = payload.get("schema_version")
    if value is None:
        if payload:
            pending.append("schema_version")
        return
    if not isinstance(value, str):
        errors.append("schema_version must be a string")
        return
    if value.strip() != FINAL_LIVE_BUCKET_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {FINAL_LIVE_BUCKET_SCHEMA_VERSION}"
        )


def _validate_allowed_keys(
    payload: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    field: str,
    errors: list[str],
) -> bool:
    unexpected: list[str] = []
    has_error = False
    for key in payload:
        if not isinstance(key, str):
            errors.append(f"{field} key {key!r} must be a string")
            has_error = True
            continue
        if key not in allowed:
            unexpected.append(key)
    if unexpected:
        errors.append(
            f"{field} has unexpected keys: {', '.join(sorted(unexpected))}"
        )
        has_error = True
    return not has_error


def _optional_string(value: Any, *, field: str, errors: list[str]) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        errors.append(f"{field} must be a string")
        return ""
    return value.strip()


def _optional_evidence_ref(value: Any, *, field: str, errors: list[str]) -> str:
    text = _optional_string(value, field=field, errors=errors)
    if text.startswith("<") and text.endswith(">"):
        return ""
    lower_text = text.lower()
    if lower_text in PLACEHOLDER_EVIDENCE_REF_VALUES:
        return ""
    if _starts_with_placeholder_evidence_ref_token(lower_text):
        return ""
    return text


def _starts_with_placeholder_evidence_ref_token(text: str) -> bool:
    for token in PLACEHOLDER_EVIDENCE_REF_PREFIXES:
        if text == token:
            return True
        if (
            text.startswith(token)
            and len(text) > len(token)
            and text[len(token)] in PLACEHOLDER_EVIDENCE_REF_SEPARATORS
        ):
            return True
    return False


def _required_string(value: Any, *, field: str, errors: list[str]) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        errors.append(f"{field} must be a string")
        return ""
    return value.strip()


def _optional_timestamp(
    value: Any,
    *,
    field: str,
    errors: list[str],
) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        errors.append(f"{field} must be a string ISO timestamp")
        return ""
    text = value.strip()
    if not text:
        return ""
    if "T" not in text:
        errors.append(f"{field} must include date and time")
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{field} must be an ISO timestamp")
        return ""
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        errors.append(f"{field} must include timezone")
        return ""
    parsed_utc = parsed.astimezone(timezone.utc)
    if parsed_utc > datetime.now(timezone.utc):
        errors.append(f"{field} must not be in the future")
        return ""
    return text
