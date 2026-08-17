"""Tests for typed receipts and the append-only runtime event log substrate."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from linkedin import surface_receipt as surface
from shared.contracts import RUN_LOG_EVENTS
from shared.receipts import (
    EventLogIntegrityError,
    Receipt,
    ReceiptStatus,
    ReceiptValidationError,
    build_receipt,
    canonical_json_dumps,
)
from shared.runtime_state.store import RuntimeStateStore
from shared.storage import RunLogEventContractError, log_event, read_jsonl


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


def _start_run(store: RuntimeStateStore, tmp_path: Path) -> int:
    return store.start_run(
        source="linkedin",
        brief_id="brief-receipts",
        output_dir=str(tmp_path),
        mode="fresh",
    )


def test_receipt_status_is_typed_not_boolean() -> None:
    with pytest.raises(ReceiptValidationError, match="not boolean"):
        build_receipt(
            receipt_type="pipeline_stage",
            stage="bad_status",
            input_payload={"x": 1},
            actual_status=True,
            intended_postcondition="status must be typed",
        )


def test_receipt_from_dict_rejects_stringified_envelope_fields() -> None:
    receipt = build_receipt(
        receipt_type="pipeline_stage",
        stage="surface_intended",
        input_payload={"event": "surface_intended"},
        actual_status=ReceiptStatus.OK,
        intended_postcondition="caller supplied receipt is typed",
        version_pins={"test": "v1"},
    )
    payload = receipt.to_dict()

    with pytest.raises(ReceiptValidationError, match="receipt.stage must be a string"):
        Receipt.from_dict({**payload, "stage": 123})

    with pytest.raises(
        ReceiptValidationError,
        match="receipt.schema_version must be a string",
    ):
        Receipt.from_dict({**payload, "schema_version": 1})

    with pytest.raises(
        ReceiptValidationError,
        match="receipt.version_pins values must be strings",
    ):
        Receipt.from_dict({**payload, "version_pins": {"test": 1}})

    with pytest.raises(
        ReceiptValidationError,
        match="receipt.version_pins keys must be strings",
    ):
        Receipt.from_dict({**payload, "version_pins": {1: "v1"}})


def test_build_receipt_rejects_stringified_envelope_fields() -> None:
    with pytest.raises(ReceiptValidationError, match="receipt.stage must be a string"):
        build_receipt(
            receipt_type="pipeline_stage",
            stage=123,
            input_payload={"x": 1},
            actual_status=ReceiptStatus.OK,
            intended_postcondition="stage must be typed",
        )

    with pytest.raises(
        ReceiptValidationError,
        match="receipt.version_pins values must be strings",
    ):
        build_receipt(
            receipt_type="pipeline_stage",
            stage="surface_intended",
            input_payload={"x": 1},
            actual_status=ReceiptStatus.OK,
            intended_postcondition="version pins must be typed",
            version_pins={"test": 1},
        )


def test_surface_receipts_emit_typed_statuses() -> None:
    summary = surface.summarize_intended_surfaces(
        [
            SimpleNamespace(
                id=1,
                name="hybrid",
                acquisition_mode="linkedin_hybrid",
                surface="",
                boolean="",
                structured_filters={"companies": ["Nubank"]},
            )
        ]
    )
    intended = surface.intended_surface_receipt(summary)
    assert intended["actual_status"] == ReceiptStatus.OK.value
    assert intended["input_hash"].startswith("sha256:")
    assert intended["intended_postcondition"]

    applied = surface.applied_surface_receipt(
        {
            "string_id": 1,
            "acquisition_mode": "linkedin_hybrid",
            "structured_applied": [],
            "plan_fully_applied": False,
            "fell_back_to_keyword": True,
        }
    )
    assert applied["actual_status"] == ReceiptStatus.POSTCONDITION_FAIL.value


def test_receipt_backed_run_log_events_are_in_contract() -> None:
    assert {
        "pipeline_start",
        "pipeline_end",
        "surface_intended",
        "surface_applied",
    }.issubset(RUN_LOG_EVENTS)


def test_jsonl_log_event_attaches_typed_receipt(tmp_path: Path) -> None:
    log_path = tmp_path / "run_log.jsonl"

    log_event(log_path, "pipeline_start", mode="full")
    log_event(log_path, "pipeline_error", error="boom")
    log_event(log_path, "surface_applied", fell_back_to_keyword=True)

    rows = read_jsonl(log_path)
    assert [row["event"] for row in rows] == [
        "pipeline_start",
        "pipeline_error",
        "surface_applied",
    ]
    statuses = [Receipt.from_dict(row["receipt"]).actual_status for row in rows]
    assert statuses == [
        ReceiptStatus.OK,
        ReceiptStatus.ERROR,
        ReceiptStatus.POSTCONDITION_FAIL,
    ]
    assert rows[0]["receipt"]["stage"] == "pipeline_start"
    assert rows[0]["receipt"]["input_hash"].startswith("sha256:")
    assert rows[0]["receipt"]["intended_postcondition"]


def test_jsonl_log_event_validates_caller_supplied_receipt(tmp_path: Path) -> None:
    log_path = tmp_path / "run_log.jsonl"
    receipt = build_receipt(
        receipt_type="pipeline_stage",
        stage="surface_intended",
        input_payload={"event": "surface_intended"},
        actual_status=ReceiptStatus.OK,
        intended_postcondition="caller supplied receipt is typed",
    )

    log_event(log_path, "surface_intended", receipt=receipt.to_dict())

    rows = read_jsonl(log_path)
    assert rows[0]["receipt"]["receipt_id"] == receipt.receipt_id

    with pytest.raises(ReceiptValidationError, match="not boolean"):
        log_event(
            log_path,
            "surface_applied",
            receipt={
                **receipt.to_dict(),
                "actual_status": True,
            },
        )

    assert len(read_jsonl(log_path)) == 1


def test_jsonl_log_event_rejects_unregistered_events(tmp_path: Path) -> None:
    log_path = tmp_path / "run_log.jsonl"

    with pytest.raises(RunLogEventContractError, match="RUN_LOG_EVENTS"):
        log_event(log_path, "unregistered_event")

    assert read_jsonl(log_path) == []


def test_runtime_events_get_receipts_and_hash_chain(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    store.record_event(
        run_id=run_id,
        event_type="pipeline_start",
        payload={"phase": "receipt-test"},
    )

    rows = store.load_run_event_log(run_id)
    assert [row["event_type"] for row in rows] == ["run_started", "pipeline_start"]
    assert rows[0]["prev_event_hash"] is None
    assert rows[1]["prev_event_hash"] == rows[0]["event_hash"]
    assert rows[1]["receipt"]["actual_status"] == ReceiptStatus.OK.value
    assert rows[1]["receipt"]["input_hash"].startswith("sha256:")
    assert rows[1]["receipt"]["intended_postcondition"]


def test_runtime_event_log_receipts_preserve_error_and_postcondition_statuses(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    store.record_event(
        run_id=run_id,
        event_type="pipeline_error",
        payload={"error": "boom"},
    )
    store.record_event(
        run_id=run_id,
        event_type="surface_applied",
        payload={"fell_back_to_keyword": True},
    )
    store.record_event(
        run_id=run_id,
        event_type="pipeline_end",
        payload={"status": "failed"},
    )

    rows = store.load_run_event_log(run_id)
    statuses = {
        row["event_type"]: row["receipt"]["actual_status"]
        for row in rows
        if row["event_type"] in {"pipeline_error", "surface_applied", "pipeline_end"}
    }
    assert statuses == {
        "pipeline_error": ReceiptStatus.ERROR.value,
        "surface_applied": ReceiptStatus.POSTCONDITION_FAIL.value,
        "pipeline_end": ReceiptStatus.ERROR.value,
    }


def test_missing_receipt_mirror_is_detected(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO events(
                run_id, work_unit_id, candidate_id, attempt_id,
                event_type, payload_json, created_at
            )
            VALUES (?, NULL, NULL, NULL, ?, '{}', ?)
            """,
            (run_id, "pipeline_start", "2026-06-17T00:00:00+00:00"),
        )

    with pytest.raises(EventLogIntegrityError, match="missing receipt-backed"):
        store.load_run_event_log(run_id)


def test_event_log_hash_chain_detects_bad_row_hash(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    created_at = "2026-06-17T00:01:00+00:00"
    with store.connect() as conn:
        legacy_event_id = int(
            conn.execute(
                """
                INSERT INTO events(
                    run_id, work_unit_id, candidate_id, attempt_id,
                    event_type, payload_json, created_at
                )
                VALUES (?, NULL, NULL, NULL, ?, '{}', ?)
                """,
                (run_id, "pipeline_start", created_at),
            ).lastrowid
        )
        previous_hash = conn.execute(
            """
            SELECT event_hash
            FROM run_event_log
            WHERE run_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()["event_hash"]
        receipt = build_receipt(
            receipt_type="pipeline_stage",
            stage="pipeline_start",
            input_payload={"event_type": "pipeline_start"},
            actual_status=ReceiptStatus.OK,
            intended_postcondition="test receipt is present",
            created_at=created_at,
        )
        conn.execute(
            """
            INSERT INTO run_event_log(
                legacy_event_id, run_id, work_unit_id, candidate_id, attempt_id,
                event_type, payload_json, receipt_json, prev_event_hash,
                event_hash, created_at
            )
            VALUES (?, ?, NULL, NULL, NULL, ?, '{}', ?, ?, ?, ?)
            """,
            (
                legacy_event_id,
                run_id,
                "pipeline_start",
                canonical_json_dumps(receipt.to_dict()),
                previous_hash,
                "sha256:not-the-real-hash",
                created_at,
            ),
        )

    with pytest.raises(EventLogIntegrityError, match="event_hash"):
        store.verify_run_event_log(run_id)


def test_event_log_mirror_is_append_only(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)
    row_id = store.load_run_event_log(run_id)[0]["id"]

    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        with store.connect() as conn:
            conn.execute(
                "UPDATE run_event_log SET event_type = 'changed' WHERE id = ?",
                (row_id,),
            )

    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        with store.connect() as conn:
            conn.execute("DELETE FROM run_event_log WHERE id = ?", (row_id,))
