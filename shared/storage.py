"""JSONL file helpers: append, read, deduplicate. Each stage writes to its own JSONL file."""

from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Any

from shared.receipts import Receipt, ReceiptStatus, build_receipt


_JSONL_LOCKS_GUARD = threading.Lock()
_JSONL_LOCKS: dict[str, threading.Lock] = {}


def _jsonl_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _JSONL_LOCKS_GUARD:
        lock = _JSONL_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _JSONL_LOCKS[key] = lock
        return lock


class RunLogEventContractError(ValueError):
    """Raised when a JSONL run-log event is not in the frozen contract."""


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one complete JSON line, serialized per path across threads."""
    path = Path(path)
    encoded = json.dumps(record) + "\n"
    with _jsonl_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(encoded)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read all records from a JSONL file. Returns empty list if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_jsonl_set(path: str | Path, key: str = "profile_url") -> set[str]:
    """Read a JSONL file and return a set of values for deduplication."""
    return {r[key] for r in read_jsonl(path) if key in r}


def write_json(path: str | Path, data: Any) -> None:
    """Write a JSON file (pretty-printed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)


def log_event(path: str | Path, event: str, **kwargs) -> None:
    """Append a timestamped event to the run log."""
    from datetime import datetime, timezone
    from shared.contracts import RUN_LOG_EVENTS

    if event not in RUN_LOG_EVENTS:
        raise RunLogEventContractError(
            f"run log event {event!r} is not registered in RUN_LOG_EVENTS"
        )
    timestamp = datetime.now(timezone.utc).isoformat()
    record = {"timestamp": timestamp, "event": event, **kwargs}
    if "receipt" not in record:
        receipt = build_receipt(
            receipt_type="pipeline_stage",
            stage=event,
            input_payload=record,
            actual_status=_receipt_status_for_event(event, kwargs),
            intended_postcondition=(
                f"run log event {event!r} is emitted with typed receipt metadata"
            ),
            actual_detail=dict(kwargs),
            producer="shared.storage.log_event",
            version_pins={"shared_storage": "run-log-receipts-v1"},
            created_at=timestamp,
        )
        record["receipt"] = receipt.to_dict()
    else:
        record["receipt"] = Receipt.from_dict(record["receipt"]).to_dict()
    append_jsonl(path, record)


def _receipt_status_for_event(event: str, payload: dict[str, Any]) -> ReceiptStatus:
    event_name = str(event)
    payload_status = str(payload.get("status") or "").strip().lower()
    if payload_status in {"error", "failed", "failure"}:
        return ReceiptStatus.ERROR
    if payload.get("fell_back_to_keyword") is True:
        return ReceiptStatus.POSTCONDITION_FAIL
    if (
        event_name.endswith("_error")
        or event_name.endswith("_failed")
        or event_name.endswith("_failure")
        or event_name.endswith("_unhandled_exception")
    ):
        return ReceiptStatus.ERROR
    return ReceiptStatus.OK
