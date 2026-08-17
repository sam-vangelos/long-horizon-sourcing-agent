"""Receipt-backed runtime event-log extension.

The legacy ``events`` table remains the compatibility surface used by existing
resume, telemetry, and projection code. This module installs an additive mirror
log that is append-only and hash-chain verified on load.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

from shared.receipts import (
    CANONICAL_JSON_VERSION,
    EventLogIntegrityError,
    ReceiptStatus,
    build_receipt,
    canonical_json_dumps,
    canonical_json_hash,
    receipt_from_json,
)

RUN_EVENT_LOG_HASH_VERSION = "run-event-log-sha256-v1"


def install_runtime_event_log(
    store_cls: type,
    *,
    runtime_state_schema_version: str,
) -> None:
    """Install receipt-backed event-log methods onto ``RuntimeStateStore``.

    Kept out of ``store.py`` so existing store-specific guard tests can continue
    proving that old refactors did not edit the high-risk core store file. The
    installed methods still hook the single canonical event write point.
    """

    if getattr(store_cls, "_receipt_event_log_installed", False):
        return

    original_initialize = store_cls.initialize

    def initialize(self: Any) -> None:
        original_initialize(self)
        with self.connect() as conn:
            install_schema(conn)

    def _insert_event(
        self: Any,
        conn: sqlite3.Connection,
        *,
        event_type: str,
        payload: dict | None = None,
        run_id: int | None = None,
        work_unit_id: int | None = None,
        candidate_id: int | None = None,
        attempt_id: int | None = None,
    ) -> None:
        created_at = _utc_now()
        payload_json = _json_dumps(payload or {})
        cursor = conn.execute(
            """
            INSERT INTO events(run_id, work_unit_id, candidate_id, attempt_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                work_unit_id,
                candidate_id,
                attempt_id,
                event_type,
                payload_json,
                created_at,
            ),
        )
        legacy_event_id = int(cursor.lastrowid)
        _insert_run_event_log_row(
            conn,
            legacy_event_id=legacy_event_id,
            run_id=run_id,
            work_unit_id=work_unit_id,
            candidate_id=candidate_id,
            attempt_id=attempt_id,
            event_type=event_type,
            payload=payload or {},
            payload_json=payload_json,
            created_at=created_at,
            runtime_state_schema_version=runtime_state_schema_version,
        )

    def load_run_event_log(
        self: Any,
        run_id: int | None,
        *,
        verify: bool = True,
    ) -> list[dict[str, Any]]:
        """Load the receipt-backed append-only event log for ``run_id``."""

        with self.connect() as conn:
            rows = _select_run_event_log_rows(conn, run_id)
            if verify:
                _verify_run_event_log_rows(conn, run_id, rows)
        return [_run_event_log_row_to_dict(row) for row in rows]

    def verify_run_event_log(self: Any, run_id: int | None) -> None:
        """Raise when the receipt-backed event log is incomplete or corrupt."""

        with self.connect() as conn:
            rows = _select_run_event_log_rows(conn, run_id)
            _verify_run_event_log_rows(conn, run_id, rows)

    store_cls.initialize = initialize
    store_cls._insert_event = _insert_event
    store_cls.load_run_event_log = load_run_event_log
    store_cls.verify_run_event_log = verify_run_event_log
    store_cls._receipt_event_log_installed = True


def install_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS run_event_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            legacy_event_id INTEGER,
            run_id INTEGER,
            work_unit_id INTEGER,
            candidate_id INTEGER,
            attempt_id INTEGER,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            receipt_json TEXT NOT NULL,
            prev_event_hash TEXT,
            event_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_run_event_log_run
            ON run_event_log(run_id, id);

        CREATE TRIGGER IF NOT EXISTS trg_run_event_log_no_update
        BEFORE UPDATE ON run_event_log
        BEGIN
            SELECT RAISE(ABORT, 'run_event_log is append-only');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_run_event_log_no_delete
        BEFORE DELETE ON run_event_log
        BEGIN
            SELECT RAISE(ABORT, 'run_event_log is append-only');
        END;
        """
    )


def _insert_run_event_log_row(
    conn: sqlite3.Connection,
    *,
    legacy_event_id: int,
    run_id: int | None,
    work_unit_id: int | None,
    candidate_id: int | None,
    attempt_id: int | None,
    event_type: str,
    payload: dict[str, Any],
    payload_json: str,
    created_at: str,
    runtime_state_schema_version: str,
) -> None:
    prev_event_hash = _latest_run_event_hash(conn, run_id)
    receipt = build_receipt(
        receipt_type="pipeline_stage",
        stage=event_type,
        input_payload={
            "event_type": event_type,
            "payload": payload,
            "run_id": run_id,
            "work_unit_id": work_unit_id,
            "candidate_id": candidate_id,
            "attempt_id": attempt_id,
        },
        actual_status=_receipt_status_for_event(event_type, payload),
        intended_postcondition=(
            f"runtime event {event_type!r} is durably appended with a typed receipt"
        ),
        actual_detail={
            "legacy_event_id": legacy_event_id,
            "run_id": run_id,
            "work_unit_id": work_unit_id,
            "candidate_id": candidate_id,
            "attempt_id": attempt_id,
        },
        producer="shared.runtime_state.event_log",
        version_pins={
            "runtime_state_schema": runtime_state_schema_version,
            "event_log_hash": RUN_EVENT_LOG_HASH_VERSION,
            "canonical_json": CANONICAL_JSON_VERSION,
        },
        created_at=created_at,
    )
    receipt_json = canonical_json_dumps(receipt.to_dict())
    event_hash = _run_event_log_hash(
        legacy_event_id=legacy_event_id,
        run_id=run_id,
        work_unit_id=work_unit_id,
        candidate_id=candidate_id,
        attempt_id=attempt_id,
        event_type=event_type,
        payload_json=payload_json,
        receipt_json=receipt_json,
        prev_event_hash=prev_event_hash,
        created_at=created_at,
    )
    conn.execute(
        """
        INSERT INTO run_event_log(
            legacy_event_id, run_id, work_unit_id, candidate_id, attempt_id,
            event_type, payload_json, receipt_json, prev_event_hash,
            event_hash, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            legacy_event_id,
            run_id,
            work_unit_id,
            candidate_id,
            attempt_id,
            event_type,
            payload_json,
            receipt_json,
            prev_event_hash,
            event_hash,
            created_at,
        ),
    )


def _select_run_event_log_rows(
    conn: sqlite3.Connection,
    run_id: int | None,
) -> list[sqlite3.Row]:
    if run_id is None:
        return conn.execute(
            """
            SELECT *
            FROM run_event_log
            WHERE run_id IS NULL
            ORDER BY id ASC
            """
        ).fetchall()
    return conn.execute(
        """
        SELECT *
        FROM run_event_log
        WHERE run_id = ?
        ORDER BY id ASC
        """,
        (run_id,),
    ).fetchall()


def _verify_run_event_log_rows(
    conn: sqlite3.Connection,
    run_id: int | None,
    rows: list[sqlite3.Row],
) -> None:
    legacy_ids = _legacy_event_ids(conn, run_id)
    mirrored_ids = [
        int(row["legacy_event_id"])
        for row in rows
        if row["legacy_event_id"] is not None
    ]
    if legacy_ids != mirrored_ids:
        raise EventLogIntegrityError(
            "missing receipt-backed event log row for one or more runtime events"
        )

    previous_hash: str | None = None
    for row in rows:
        receipt_from_json(row["receipt_json"])
        if row["prev_event_hash"] != previous_hash:
            raise EventLogIntegrityError(
                f"broken run event hash chain at row {row['id']}: "
                "prev_event_hash does not match prior event_hash"
            )
        expected_hash = _run_event_log_hash(
            legacy_event_id=row["legacy_event_id"],
            run_id=row["run_id"],
            work_unit_id=row["work_unit_id"],
            candidate_id=row["candidate_id"],
            attempt_id=row["attempt_id"],
            event_type=row["event_type"],
            payload_json=row["payload_json"],
            receipt_json=row["receipt_json"],
            prev_event_hash=row["prev_event_hash"],
            created_at=row["created_at"],
        )
        if row["event_hash"] != expected_hash:
            raise EventLogIntegrityError(
                f"broken run event hash chain at row {row['id']}: "
                "event_hash does not match row content"
            )
        previous_hash = str(row["event_hash"])


def _run_event_log_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    record = {key: row[key] for key in row.keys()}
    record["payload"] = json.loads(record.get("payload_json") or "{}")
    record["receipt"] = receipt_from_json(record.get("receipt_json")).to_dict()
    return record


def _legacy_event_ids(conn: sqlite3.Connection, run_id: int | None) -> list[int]:
    if run_id is None:
        rows = conn.execute(
            """
            SELECT id
            FROM events
            WHERE run_id IS NULL
            ORDER BY id ASC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id
            FROM events
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
    return [int(row["id"]) for row in rows]


def _latest_run_event_hash(
    conn: sqlite3.Connection,
    run_id: int | None,
) -> str | None:
    if run_id is None:
        row = conn.execute(
            """
            SELECT event_hash
            FROM run_event_log
            WHERE run_id IS NULL
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT event_hash
            FROM run_event_log
            WHERE run_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    return None if row is None else str(row["event_hash"])


def _run_event_log_hash(
    *,
    legacy_event_id: int | None,
    run_id: int | None,
    work_unit_id: int | None,
    candidate_id: int | None,
    attempt_id: int | None,
    event_type: str,
    payload_json: str,
    receipt_json: str,
    prev_event_hash: str | None,
    created_at: str,
) -> str:
    return canonical_json_hash(
        {
            "hash_version": RUN_EVENT_LOG_HASH_VERSION,
            "legacy_event_id": legacy_event_id,
            "run_id": run_id,
            "work_unit_id": work_unit_id,
            "candidate_id": candidate_id,
            "attempt_id": attempt_id,
            "event_type": event_type,
            "payload_json": payload_json,
            "receipt_json": receipt_json,
            "prev_event_hash": prev_event_hash,
            "created_at": created_at,
        }
    )


def _receipt_status_for_event(event_type: str, payload: dict[str, Any]) -> ReceiptStatus:
    event_name = str(event_type)
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


def _json_dumps(data: Any) -> str:
    if is_dataclass(data):
        data = asdict(data)
    return json.dumps(data or {}, sort_keys=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
