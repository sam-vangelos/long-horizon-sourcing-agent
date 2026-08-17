"""Typed receipt primitives for sourcing runtime observability.

Receipts are the small, content-addressed records that make intended-vs-actual
execution visible. They are deliberately plain dataclasses so source adapters can
emit them without depending on a runtime-state store.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


RECEIPT_SCHEMA_VERSION = "receipt.v1"
RECEIPT_ID_HASH_VERSION = "receipt-id-sha256-v1"
CANONICAL_JSON_VERSION = "canonical-json-v1"


class ReceiptError(ValueError):
    """Base class for receipt validation errors."""


class ReceiptValidationError(ReceiptError):
    """Raised when a receipt is missing or malformed."""


class EventLogIntegrityError(ReceiptError):
    """Raised when an append-only event log fails receipt/hash verification."""


class ReceiptStatus(str, Enum):
    """Typed stage status per INV-8.

    Keep this enum intentionally small. Callers that need more detail should put
    it in ``actual_detail`` instead of inventing a second status vocabulary.
    """

    OK = "ok"
    PARSE_FAIL = "parse_fail"
    REFUSED = "refused"
    ABSTAIN = "abstain"
    EMPTY = "empty"
    ERROR = "error"
    POSTCONDITION_FAIL = "postcondition_fail"


RECEIPT_STATUS_VALUES = frozenset(status.value for status in ReceiptStatus)


@dataclass(frozen=True)
class Receipt:
    """A typed intended-vs-actual execution receipt."""

    receipt_id: str
    receipt_type: str
    stage: str
    input_hash: str
    actual_status: ReceiptStatus
    intended_postcondition: str
    actual_detail: dict[str, Any] = field(default_factory=dict)
    producer: str = ""
    schema_version: str = RECEIPT_SCHEMA_VERSION
    version_pins: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["actual_status"] = self.actual_status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Receipt":
        if not isinstance(data, dict):
            raise ReceiptValidationError("receipt must be a JSON object")
        required = {
            "receipt_id",
            "receipt_type",
            "stage",
            "input_hash",
            "actual_status",
            "intended_postcondition",
            "created_at",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ReceiptValidationError(
                f"receipt missing required field(s): {', '.join(missing)}"
            )
        status = _coerce_status(data["actual_status"])
        actual_detail = data.get("actual_detail", {})
        if not isinstance(actual_detail, dict):
            raise ReceiptValidationError("receipt.actual_detail must be an object")
        version_pins = data.get("version_pins", {})
        if not isinstance(version_pins, dict):
            raise ReceiptValidationError("receipt.version_pins must be an object")
        schema_version = _require_string(
            data.get("schema_version"),
            field="receipt.schema_version",
        )
        if schema_version != RECEIPT_SCHEMA_VERSION:
            raise ReceiptValidationError(
                f"unsupported receipt schema_version: {schema_version!r}"
            )
        return cls(
            receipt_id=_require_string(data["receipt_id"], field="receipt.receipt_id"),
            receipt_type=_require_string(
                data["receipt_type"],
                field="receipt.receipt_type",
            ),
            stage=_require_string(data["stage"], field="receipt.stage"),
            input_hash=_require_string(data["input_hash"], field="receipt.input_hash"),
            actual_status=status,
            intended_postcondition=_require_string(
                data["intended_postcondition"],
                field="receipt.intended_postcondition",
            ),
            actual_detail=dict(actual_detail),
            producer=_optional_string(
                data.get("producer"),
                field="receipt.producer",
            ),
            schema_version=schema_version,
            version_pins=_validate_version_pins(version_pins),
            created_at=_require_string(data["created_at"], field="receipt.created_at"),
        )


def build_receipt(
    *,
    receipt_type: str,
    stage: str,
    input_payload: Any,
    actual_status: ReceiptStatus | str,
    intended_postcondition: str,
    actual_detail: dict[str, Any] | None = None,
    producer: str = "",
    version_pins: dict[str, str] | None = None,
    created_at: str | None = None,
) -> Receipt:
    """Build a content-addressed typed receipt.

    ``actual_status`` must be one of :class:`ReceiptStatus`; booleans are rejected
    even though ``bool`` is an ``int`` subclass in Python. That keeps the INV-8
    boundary explicit.
    """

    status = _coerce_status(actual_status)
    if actual_detail is not None and not isinstance(actual_detail, dict):
        raise ReceiptValidationError("receipt.actual_detail must be an object")
    if version_pins is not None and not isinstance(version_pins, dict):
        raise ReceiptValidationError("receipt.version_pins must be an object")
    timestamp = created_at or _utc_now()
    body = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": _require_string(receipt_type, field="receipt.receipt_type"),
        "stage": _require_string(stage, field="receipt.stage"),
        "input_hash": canonical_json_hash(input_payload),
        "actual_status": status.value,
        "intended_postcondition": _require_string(
            intended_postcondition,
            field="receipt.intended_postcondition",
        ),
        "actual_detail": actual_detail or {},
        "producer": _optional_string(producer, field="receipt.producer"),
        "version_pins": _validate_version_pins(version_pins or {}),
        "created_at": _require_string(timestamp, field="receipt.created_at"),
    }
    receipt_id = canonical_json_hash(
        {
            "hash_version": RECEIPT_ID_HASH_VERSION,
            "receipt": body,
        }
    )
    receipt_fields = dict(body)
    receipt_fields["actual_status"] = status
    return Receipt(receipt_id=receipt_id, **receipt_fields)


def receipt_from_json(raw: str | bytes | None) -> Receipt:
    """Parse a receipt JSON string and validate its typed shape."""

    if not raw:
        raise ReceiptValidationError("missing receipt JSON")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReceiptValidationError("receipt JSON is malformed") from exc
    return Receipt.from_dict(parsed)


def canonical_json_dumps(data: Any) -> str:
    """Return the canonical JSON form used for receipt and event hashes."""

    return json.dumps(
        _jsonable(data),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_json_hash(data: Any) -> str:
    """Return a versioned sha256 hash over canonical JSON."""

    digest = hashlib.sha256(canonical_json_dumps(data).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _coerce_status(status: ReceiptStatus | str) -> ReceiptStatus:
    if isinstance(status, bool):
        raise ReceiptValidationError("receipt status must be typed, not boolean")
    if isinstance(status, ReceiptStatus):
        return status
    if isinstance(status, str):
        try:
            return ReceiptStatus(status)
        except ValueError as exc:
            raise ReceiptValidationError(f"unknown receipt status: {status!r}") from exc
    raise ReceiptValidationError(
        f"receipt status must be a ReceiptStatus or string, got {type(status).__name__}"
    )


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ReceiptValidationError(f"{field} must be a string")
    if not value:
        raise ReceiptValidationError(f"{field} must be non-empty")
    return value


def _optional_string(value: Any, *, field: str) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise ReceiptValidationError(f"{field} must be a string")
    return value


def _validate_version_pins(version_pins: dict[Any, Any]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for key, value in version_pins.items():
        if not isinstance(key, str):
            raise ReceiptValidationError("receipt.version_pins keys must be strings")
        if not key:
            raise ReceiptValidationError(
                "receipt.version_pins keys must be non-empty"
            )
        if not isinstance(value, str):
            raise ReceiptValidationError("receipt.version_pins values must be strings")
        if not value:
            raise ReceiptValidationError(
                "receipt.version_pins values must be non-empty"
            )
        pins[key] = value
    return pins


def _jsonable(value: Any) -> Any:
    if isinstance(value, ReceiptStatus):
        return value.value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
