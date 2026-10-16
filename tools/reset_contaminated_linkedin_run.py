#!/usr/bin/env python3
"""Dry-run and apply a reviewed LinkedIn contamination-reset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import config  # noqa: E402
from shared.runtime_state.admin import rebuild_compat_projections  # noqa: E402
from shared.runtime_state.linkedin_contamination_reset import (  # noqa: E402
    ContaminationResetPlan,
    append_contamination_reset_boundary,
    apply_linkedin_contamination_reset,
    plan_linkedin_contamination_reset,
    record_contamination_reset_completed,
)
from shared.runtime_state.lock import RuntimeStateLock  # noqa: E402
from shared.runtime_state.store import RuntimeStateStore  # noqa: E402


RECEIPT_VERSION = "linkedin-contamination-reset-v1"
BACKUP_METADATA_VERSION = "linkedin-contamination-reset-backup-v1"
_RECEIPT_PREFIX = "facet-contamination-reset-"
_PROJECTION_FILENAMES = (
    "progress.json",
    "candidate_history-{brief_id}.jsonl",
    "search_memory-{brief_id}.json",
    "snippets.jsonl",
    "facial_judgments.jsonl",
    "profile_summaries.jsonl",
    "final_judgments.jsonl",
)


class ResetToolError(RuntimeError):
    """The operator reset could not safely proceed."""


@dataclass(frozen=True)
class ReviewedManifest:
    path: Path
    payload: dict[str, Any]
    pairs: tuple[tuple[int, str], ...]
    sha256: str


@dataclass(frozen=True)
class ResetContext:
    plan: ContaminationResetPlan
    brief_id: str
    latest_run_id: int
    latest_run_status: str
    confirmation: str
    reset_id: str


def load_reviewed_manifest(path: str | Path) -> ReviewedManifest:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ResetToolError(f"manifest not found: {manifest_path}")
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResetToolError(f"manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ResetToolError("manifest root must be an object")
    if payload.get("reviewed") is not True:
        raise ResetToolError("manifest must declare reviewed: true")
    executions = payload.get("executions")
    if not isinstance(executions, list) or not executions:
        raise ResetToolError("manifest.executions must be a non-empty array")
    if payload.get("ambiguous") or payload.get("ambiguities"):
        raise ResetToolError("manifest contains unresolved ambiguity")

    pairs: list[tuple[int, str]] = []
    for index, execution in enumerate(executions):
        if not isinstance(execution, dict):
            raise ResetToolError(f"manifest execution #{index + 1} must be an object")
        classification = str(execution.get("classification") or "").strip().lower()
        if classification != "contaminated":
            raise ResetToolError(
                f"manifest execution #{index + 1} must be classified contaminated"
            )
        run_id = execution.get("run_id")
        source_unit_id = execution.get("source_unit_id")
        if isinstance(run_id, bool) or not isinstance(run_id, int):
            raise ResetToolError(
                f"manifest execution #{index + 1} run_id must be an integer"
            )
        if source_unit_id is None or not str(source_unit_id).strip():
            raise ResetToolError(
                f"manifest execution #{index + 1} source_unit_id is required"
            )
        pairs.append((run_id, str(source_unit_id).strip()))

    return ReviewedManifest(
        path=manifest_path,
        payload=payload,
        pairs=tuple(pairs),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def expected_confirmation(
    *,
    brief_id: str,
    manifest_sha256: str,
    executed_count: int,
) -> str:
    return f"{brief_id}:{manifest_sha256}:{executed_count}"


def run_reset_tool(
    *,
    state_dir: str | Path,
    manifest: ReviewedManifest,
    apply: bool,
    confirmation: str | None,
) -> int:
    resolved_state_dir = Path(state_dir).resolve()
    db_path = resolved_state_dir / "runtime_state.sqlite3"
    if not resolved_state_dir.is_dir() or not db_path.is_file():
        raise ResetToolError(f"runtime state DB not found: {db_path}")

    lock_path = resolved_state_dir / "runtime_state.lock"
    if not lock_path.exists():
        raise ResetToolError(
            f"runtime-state lock file is missing; refuse to create it during reset: {lock_path}"
        )
    with RuntimeStateLock(resolved_state_dir):
        _assert_sqlite_integrity(db_path)
        context = _read_context(db_path, manifest)
        _print_plan(context)
        if not apply:
            if confirmation is not None:
                raise ResetToolError("--confirm is valid only with --apply")
            print(f"[reset] DRY-RUN confirmation={context.confirmation}")
            return 0

        if confirmation is None:
            raise ResetToolError("--apply requires --confirm")
        if confirmation != context.confirmation:
            raise ResetToolError(
                f"confirmation mismatch; expected {context.confirmation}"
            )
        if not config.LINKEDIN_CONTAMINATION_RESET_ENABLED:
            raise ResetToolError(
                "LINKEDIN_CONTAMINATION_RESET_ENABLED must be true before apply"
            )

        receipt_path = _receipt_path(resolved_state_dir, context.reset_id)
        if receipt_path.exists():
            _validate_completed_receipt(receipt_path, manifest, context)
            if not context.plan.is_noop:
                raise ResetToolError(
                    "completed receipt exists but canonical reset actions remain"
                )
            # A crash between the receipt write and the completion event
            # leaves run start blocked (reset_incomplete); the idempotent
            # rerun closes that window here. Read-only probe first so the
            # normal NO-OP rerun leaves the DB byte-identical.
            if not _completion_event_exists(db_path, reset_id=context.reset_id):
                record_contamination_reset_completed(
                    RuntimeStateStore(db_path),
                    reset_id=context.reset_id,
                    payload={"receipt": receipt_path.name},
                )
            print(
                f"[reset] NO-OP reset={context.reset_id} receipt={receipt_path.name}"
            )
            return 0

        backup_path = _backup_path(resolved_state_dir, context.reset_id)
        if context.plan.is_noop and not backup_path.exists():
            raise ResetToolError(
                "canonical reset is already applied but its pre-reset backup is missing"
            )
        if backup_path.exists():
            _assert_sqlite_integrity(backup_path)
        else:
            _backup_sqlite(db_path, backup_path)
        backup_context = _read_context(backup_path, manifest)
        if backup_context.confirmation != context.confirmation:
            raise ResetToolError("pre-reset backup does not match the reviewed manifest")
        if backup_context.plan.is_noop:
            raise ResetToolError("pre-reset backup contains no reset actions")
        already_applied = _classify_canonical_reset_phase(
            db_path=db_path,
            backup_path=backup_path,
            reset_id=context.reset_id,
            action_plan=backup_context.plan,
        )
        backup_metadata = _load_or_create_backup_metadata(
            state_dir=resolved_state_dir,
            db_path=db_path,
            backup_path=backup_path,
            context=backup_context,
            manifest=manifest,
            allow_create=not already_applied,
        )

        store = RuntimeStateStore(db_path)
        if not already_applied:
            apply_linkedin_contamination_reset(
                store,
                manifest=manifest.pairs,
                reset_id=context.reset_id,
                manifest_digest=manifest.sha256,
            )
        _assert_expected_post_reset_state(
            db_path=db_path,
            backup_path=backup_path,
            reset_id=context.reset_id,
            action_plan=backup_context.plan,
        )
        rebuild_compat_projections(
            store,
            run_id=context.latest_run_id,
            output_dir=resolved_state_dir,
        )
        # F6 artifact phase: reset the noninvertible false-lesson sidecars
        # from a per-reset backup, then mark the run-log boundary so
        # boundary-aware readers never re-ingest pre-reset events.
        sidecar_dispositions = _reset_sidecar_artifacts(
            state_dir=resolved_state_dir,
            brief_id=context.brief_id,
            reset_id=context.reset_id,
        )
        append_contamination_reset_boundary(
            resolved_state_dir,
            reset_id=context.reset_id,
            sidecars=[
                name
                for name, disposition in sidecar_dispositions.items()
                if disposition != "absent"
            ],
        )
        after_hashes = _reset_hashes(
            state_dir=resolved_state_dir,
            brief_id=context.brief_id,
            sqlite_snapshot=db_path,
        )
        receipt = _build_receipt(
            context=context,
            manifest=manifest,
            backup_path=backup_path,
            before_hashes=backup_metadata["before_hashes"],
            after_hashes=after_hashes,
            action_plan=backup_context.plan,
            sidecars=sidecar_dispositions,
        )
        _write_json_atomic(receipt_path, receipt)
        # The completion event is the LAST write: run start stays blocked
        # (reset_incomplete) until every artifact phase above has finished,
        # which is what makes a crashed artifact phase resumable.
        record_contamination_reset_completed(
            store,
            reset_id=context.reset_id,
            payload={"receipt": receipt_path.name},
        )

    print(
        f"[reset] APPLIED reset={context.reset_id} receipt={receipt_path.name}"
    )
    return 0


def _read_context(db_path: Path, manifest: ReviewedManifest) -> ResetContext:
    with closing(_open_readonly(db_path)) as conn:
        plan = plan_linkedin_contamination_reset(conn, manifest.pairs)
        brief_ids = {work_unit.brief_id for work_unit in plan.work_units}
        if len(brief_ids) != 1:
            raise ResetToolError("manifest executions must belong to one brief")
        brief_id = next(iter(brief_ids))
        declared_brief_id = manifest.payload.get("brief_id")
        if declared_brief_id is not None and str(declared_brief_id) != brief_id:
            raise ResetToolError(
                f"manifest brief_id {declared_brief_id!r} does not match {brief_id!r}"
            )
        latest_run = conn.execute(
            """
            SELECT id, status
            FROM runs
            WHERE source = 'linkedin' AND brief_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (brief_id,),
        ).fetchone()
        if latest_run is None:
            raise ResetToolError(f"no LinkedIn run found for brief {brief_id}")
    context = ResetContext(
        plan=plan,
        brief_id=brief_id,
        latest_run_id=int(latest_run["id"]),
        latest_run_status=str(latest_run["status"]),
        confirmation=expected_confirmation(
            brief_id=brief_id,
            manifest_sha256=manifest.sha256,
            executed_count=len(plan.work_units),
        ),
        reset_id=manifest.sha256[:20],
    )
    _assert_not_running(context)
    return context


def _assert_not_running(context: ResetContext) -> None:
    if context.latest_run_status == "running":
        raise ResetToolError(
            f"latest LinkedIn run {context.latest_run_id} is still running"
        )


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    wal_path = db_path.with_name(db_path.name + "-wal")
    immutable = not wal_path.exists() or wal_path.stat().st_size == 0
    query = "mode=ro&immutable=1" if immutable else "mode=ro"
    uri = f"{db_path.resolve().as_uri()}?{query}"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _assert_sqlite_integrity(db_path: Path) -> None:
    with closing(_open_readonly(db_path)) as conn:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
    messages = [str(row[0]) for row in rows]
    if messages != ["ok"]:
        raise ResetToolError(
            f"SQLite integrity_check failed for {db_path}: {'; '.join(messages)}"
        )


def _backup_sqlite(db_path: Path, backup_path: Path) -> None:
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=backup_path.name + ".",
        suffix=".tmp",
        dir=str(backup_path.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with (
            closing(_open_readonly(db_path)) as source,
            closing(sqlite3.connect(str(temp_path))) as destination,
        ):
            source.backup(destination)
            destination.commit()
        _assert_sqlite_integrity(temp_path)
        os.replace(temp_path, backup_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _reset_hashes(
    *,
    state_dir: Path,
    brief_id: str,
    sqlite_snapshot: Path,
) -> dict[str, Any]:
    if sqlite_snapshot == state_dir / "runtime_state.sqlite3":
        sqlite_hash = _live_sqlite_sha256(sqlite_snapshot)
    else:
        sqlite_hash = _sha256_file(sqlite_snapshot)
    projections: dict[str, str | None] = {}
    for template in _PROJECTION_FILENAMES:
        filename = template.format(brief_id=brief_id)
        path = state_dir / filename
        projections[filename] = _sha256_file(path) if path.is_file() else None
    return {
        "sqlite_snapshot_sha256": sqlite_hash,
        "compatibility_projections": projections,
    }


def _live_sqlite_sha256(db_path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="cloris-reset-hash-") as temp_dir:
        snapshot = Path(temp_dir) / "runtime_state.sqlite3"
        _backup_sqlite(db_path, snapshot)
        return _sha256_file(snapshot)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_path(state_dir: Path, reset_id: str) -> Path:
    return state_dir / "backups" / f"{_RECEIPT_PREFIX}{reset_id}.sqlite3"


def _backup_metadata_path(state_dir: Path, reset_id: str) -> Path:
    return state_dir / "backups" / f"{_RECEIPT_PREFIX}{reset_id}.json"


def _receipt_path(state_dir: Path, reset_id: str) -> Path:
    return state_dir / f"{_RECEIPT_PREFIX}{reset_id}.json"


def _load_or_create_backup_metadata(
    *,
    state_dir: Path,
    db_path: Path,
    backup_path: Path,
    context: ResetContext,
    manifest: ReviewedManifest,
    allow_create: bool,
) -> dict[str, Any]:
    metadata_path = _backup_metadata_path(state_dir, context.reset_id)
    expected_backup = {
        "path": str(backup_path.relative_to(state_dir)),
        "sha256": _sha256_file(backup_path),
    }
    expected_actions = _plan_receipt(context.plan)
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResetToolError(
                f"pre-reset backup metadata is unreadable: {metadata_path}"
            ) from exc
    else:
        if not allow_create:
            raise ResetToolError(
                "canonical reset is already applied but pre-reset backup metadata is missing"
            )
        _assert_exact_canonical_state(db_path=db_path, backup_path=backup_path)
        metadata = {
            "version": BACKUP_METADATA_VERSION,
            "reset_id": context.reset_id,
            "brief_id": context.brief_id,
            "manifest_sha256": manifest.sha256,
            "executed_count": len(context.plan.work_units),
            "backup": expected_backup,
            "before_hashes": _reset_hashes(
                state_dir=state_dir,
                brief_id=context.brief_id,
                sqlite_snapshot=backup_path,
            ),
            "actions": expected_actions,
        }
        _write_json_atomic(metadata_path, metadata)

    expected = {
        "version": BACKUP_METADATA_VERSION,
        "reset_id": context.reset_id,
        "brief_id": context.brief_id,
        "manifest_sha256": manifest.sha256,
        "executed_count": len(context.plan.work_units),
        "backup": expected_backup,
        "actions": expected_actions,
    }
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != value for key, value in expected.items()
    ):
        raise ResetToolError(
            f"pre-reset backup metadata does not match: {metadata_path}"
        )
    before_hashes = metadata.get("before_hashes")
    if not isinstance(before_hashes, dict) or before_hashes.get(
        "sqlite_snapshot_sha256"
    ) != expected_backup["sha256"]:
        raise ResetToolError(
            f"pre-reset backup hashes do not match: {metadata_path}"
        )
    projection_hashes = before_hashes.get("compatibility_projections")
    expected_projection_names = {
        template.format(brief_id=context.brief_id)
        for template in _PROJECTION_FILENAMES
    }
    if not isinstance(projection_hashes, dict) or set(projection_hashes) != (
        expected_projection_names
    ):
        raise ResetToolError(
            f"pre-reset projection hashes do not match: {metadata_path}"
        )
    return metadata


def _build_receipt(
    *,
    context: ResetContext,
    manifest: ReviewedManifest,
    backup_path: Path,
    before_hashes: dict[str, Any],
    after_hashes: dict[str, Any],
    action_plan: ContaminationResetPlan,
    sidecars: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "version": RECEIPT_VERSION,
        "status": "completed",
        "reset_id": context.reset_id,
        "brief_id": context.brief_id,
        "manifest_sha256": manifest.sha256,
        "executed_count": len(context.plan.work_units),
        "manifest": manifest.payload,
        "backup": {
            "path": str(backup_path.relative_to(backup_path.parents[1])),
            "sha256": _sha256_file(backup_path),
        },
        "before_hashes": before_hashes,
        "after_hashes": after_hashes,
        "actions": _plan_receipt(action_plan),
        "sidecars": dict(sidecars or {}),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _plan_receipt(plan: ContaminationResetPlan) -> dict[str, Any]:
    return {
        "counts": plan.counts(),
        "invalidated_work_unit_ids": [
            row.work_unit_id for row in plan.work_units_to_invalidate
        ],
        "released_candidates": [asdict(row) for row in plan.candidates_to_release],
        "abandoned_attempts": [asdict(row) for row in plan.attempts_to_abandon],
        "untouched_candidates": [asdict(row) for row in plan.untouched_candidates],
    }


def _classify_canonical_reset_phase(
    *,
    db_path: Path,
    backup_path: Path,
    reset_id: str,
    action_plan: ContaminationResetPlan,
) -> bool:
    """Return whether the canonical reset is fully applied, refusing drift."""

    if not _reset_event_rows(db_path, reset_id=reset_id):
        _assert_exact_canonical_state(db_path=db_path, backup_path=backup_path)
        return False
    _assert_expected_post_reset_state(
        db_path=db_path,
        backup_path=backup_path,
        reset_id=reset_id,
        action_plan=action_plan,
    )
    return True


def _assert_exact_canonical_state(*, db_path: Path, backup_path: Path) -> None:
    with (
        closing(_open_readonly(backup_path)) as before,
        closing(_open_readonly(db_path)) as after,
    ):
        before_tables = _user_table_names(before)
        after_tables = _user_table_names(after)
        if before_tables != after_tables:
            raise ResetToolError(
                "canonical schema changed after the pre-reset backup was created"
            )
        for table_name in sorted(before_tables):
            if _table_fingerprint(before, table_name) != _table_fingerprint(
                after, table_name
            ):
                raise ResetToolError(
                    f"canonical table {table_name} changed after the pre-reset backup"
                )


def _assert_expected_post_reset_state(
    *,
    db_path: Path,
    backup_path: Path,
    reset_id: str,
    action_plan: ContaminationResetPlan,
) -> None:
    """Prove live SQLite equals the reset-only transformation of its backup."""

    targeted_tables = {
        "candidates",
        "candidate_attempts",
        "events",
        "run_event_log",
    }
    with (
        closing(_open_readonly(backup_path)) as before,
        closing(_open_readonly(db_path)) as after,
    ):
        before_tables = _user_table_names(before)
        after_tables = _user_table_names(after)
        if before_tables != after_tables:
            raise ResetToolError(
                "canonical schema changed after the pre-reset backup was created"
            )
        for table_name in sorted(before_tables - targeted_tables):
            if _table_fingerprint(before, table_name) != _table_fingerprint(
                after, table_name
            ):
                raise ResetToolError(
                    f"canonical table {table_name} changed outside the reset plan"
                )
        _assert_expected_candidate_rows(before, after, action_plan)
        _assert_expected_attempt_rows(before, after, action_plan)
        _assert_expected_event_rows(
            before,
            after,
            reset_id=reset_id,
            action_plan=action_plan,
        )
    _assert_reset_events(db_path, reset_id=reset_id, action_plan=action_plan)


def _user_table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
    }


def _table_fingerprint(conn: sqlite3.Connection, table_name: str) -> str:
    quoted_name = '"' + table_name.replace('"', '""') + '"'
    columns = [
        tuple(row)
        for row in conn.execute(f"PRAGMA table_info({quoted_name})").fetchall()
    ]
    encoded_rows = [
        json.dumps(
            [_encode_sqlite_value(value) for value in tuple(row)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in conn.execute(f"SELECT * FROM {quoted_name}").fetchall()
    ]
    encoded_rows.sort()
    material = json.dumps(
        {
            "columns": [
                [_encode_sqlite_value(value) for value in row] for row in columns
            ],
            "rows": encoded_rows,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _encode_sqlite_value(value: Any) -> list[Any]:
    if value is None:
        return ["null", None]
    if isinstance(value, bytes):
        return ["blob", value.hex()]
    if isinstance(value, int):
        return ["integer", value]
    if isinstance(value, float):
        return ["real", value.hex()]
    return ["text", str(value)]


def _rows_by_id(conn: sqlite3.Connection, table_name: str) -> dict[int, dict[str, Any]]:
    quoted_name = '"' + table_name.replace('"', '""') + '"'
    return {
        int(row["id"]): dict(row)
        for row in conn.execute(f"SELECT * FROM {quoted_name}").fetchall()
    }


def _assert_expected_candidate_rows(
    before: sqlite3.Connection,
    after: sqlite3.Connection,
    action_plan: ContaminationResetPlan,
) -> None:
    before_rows = _rows_by_id(before, "candidates")
    after_rows = _rows_by_id(after, "candidates")
    if before_rows.keys() != after_rows.keys():
        raise ResetToolError("canonical candidates changed outside the reset plan")
    released_ids = {row.candidate_id for row in action_plan.candidates_to_release}
    for candidate_id, before_row in before_rows.items():
        expected = dict(before_row)
        if candidate_id in released_ids:
            expected.update(
                terminal_decision=None,
                terminal_payload_json="{}",
                current_lifecycle_state="failed_retryable",
            )
        if after_rows[candidate_id] != expected:
            raise ResetToolError(
                f"canonical candidate {candidate_id} changed outside the reset plan"
            )


def _assert_expected_attempt_rows(
    before: sqlite3.Connection,
    after: sqlite3.Connection,
    action_plan: ContaminationResetPlan,
) -> None:
    before_rows = _rows_by_id(before, "candidate_attempts")
    after_rows = _rows_by_id(after, "candidate_attempts")
    if before_rows.keys() != after_rows.keys():
        raise ResetToolError("canonical attempts changed outside the reset plan")
    abandoned_ids = {row.attempt_id for row in action_plan.attempts_to_abandon}
    for attempt_id, before_row in before_rows.items():
        expected = dict(before_row)
        if attempt_id in abandoned_ids:
            ended_at = after_rows[attempt_id].get("ended_at")
            if not isinstance(ended_at, str) or not ended_at:
                raise ResetToolError(
                    f"reset attempt {attempt_id} has no completion timestamp"
                )
            expected.update(
                status="failed",
                failure_kind="contamination_reset",
                ended_at=ended_at,
            )
        if after_rows[attempt_id] != expected:
            raise ResetToolError(
                f"canonical attempt {attempt_id} changed outside the reset plan"
            )


def _assert_expected_event_rows(
    before: sqlite3.Connection,
    after: sqlite3.Connection,
    *,
    reset_id: str,
    action_plan: ContaminationResetPlan,
) -> None:
    before_rows = _rows_by_id(before, "events")
    after_rows = _rows_by_id(after, "events")
    for event_id, before_row in before_rows.items():
        if after_rows.get(event_id) != before_row:
            raise ResetToolError(
                f"canonical event {event_id} changed outside the reset plan"
            )
    added_rows = [
        row for event_id, row in after_rows.items() if event_id not in before_rows
    ]
    action_count = (
        len(action_plan.work_units_to_invalidate)
        + len(action_plan.candidates_to_release)
        + len(action_plan.attempts_to_abandon)
    )
    root_rows = [
        row
        for row in added_rows
        if row.get("event_type") == "linkedin_contamination_reset"
    ]
    completed_rows = [
        row
        for row in added_rows
        if row.get("event_type") == "linkedin_contamination_reset_completed"
    ]
    # The canonical apply writes exactly one root receipt alongside the
    # actions; the completion event exists only once the artifact phase
    # finished (an artifact-phase resume re-checks this state with it present).
    if len(root_rows) != 1 or len(completed_rows) > 1:
        raise ResetToolError("canonical events changed outside the reset plan")
    if len(added_rows) != action_count + len(root_rows) + len(completed_rows):
        raise ResetToolError("canonical events changed outside the reset plan")
    reset_event_types = {
        "linkedin_execution_invalidated",
        "linkedin_candidate_released_for_contamination",
        "linkedin_attempt_abandoned_for_contamination",
        "linkedin_contamination_reset",
        "linkedin_contamination_reset_completed",
    }
    for row in added_rows:
        try:
            payload = json.loads(row.get("payload_json") or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ResetToolError("canonical reset event payload is unreadable") from exc
        if (
            row.get("event_type") not in reset_event_types
            or not isinstance(payload, dict)
            or payload.get("reset_id") != reset_id
        ):
            raise ResetToolError("canonical events changed outside the reset plan")
    _assert_expected_run_event_log_rows(
        before,
        after,
        added_event_rows=added_rows,
    )


def _assert_expected_run_event_log_rows(
    before: sqlite3.Connection,
    after: sqlite3.Connection,
    *,
    added_event_rows: list[dict[str, Any]],
) -> None:
    before_rows = _rows_by_id(before, "run_event_log")
    after_rows = _rows_by_id(after, "run_event_log")
    for event_id, before_row in before_rows.items():
        if after_rows.get(event_id) != before_row:
            raise ResetToolError(
                f"canonical run-event receipt {event_id} changed outside the reset plan"
            )
    added_rows = [
        row for event_id, row in after_rows.items() if event_id not in before_rows
    ]
    if len(added_rows) != len(added_event_rows):
        raise ResetToolError("canonical run-event log changed outside the reset plan")
    legacy_events = {int(row["id"]): row for row in added_event_rows}
    mirrored_fields = (
        "run_id",
        "work_unit_id",
        "candidate_id",
        "attempt_id",
        "event_type",
        "payload_json",
        "created_at",
    )
    for row in added_rows:
        legacy_event_id = row.get("legacy_event_id")
        legacy_row = (
            legacy_events.get(int(legacy_event_id))
            if legacy_event_id is not None
            else None
        )
        if legacy_row is None or any(
            row.get(field) != legacy_row.get(field) for field in mirrored_fields
        ):
            raise ResetToolError("canonical run-event log changed outside the reset plan")


def _reset_event_rows(db_path: Path, *, reset_id: str) -> list[sqlite3.Row]:
    event_types = (
        "linkedin_execution_invalidated",
        "linkedin_candidate_released_for_contamination",
        "linkedin_attempt_abandoned_for_contamination",
        "linkedin_contamination_reset",
        "linkedin_contamination_reset_completed",
    )
    with closing(_open_readonly(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM events
            WHERE event_type IN (?, ?, ?, ?, ?)
            ORDER BY id ASC
            """,
            event_types,
        ).fetchall()
    matching: list[sqlite3.Row] = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("reset_id") == reset_id:
            matching.append(row)
    return matching


def _assert_reset_events(
    db_path: Path,
    *,
    reset_id: str,
    action_plan: ContaminationResetPlan,
) -> None:
    event_types = {
        "linkedin_execution_invalidated",
        "linkedin_candidate_released_for_contamination",
        "linkedin_attempt_abandoned_for_contamination",
    }
    observed = {event_type: set() for event_type in event_types}
    with closing(_open_readonly(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT event_type, run_id, work_unit_id, candidate_id, attempt_id, payload_json
            FROM events
            WHERE event_type IN (
                'linkedin_execution_invalidated',
                'linkedin_candidate_released_for_contamination',
                'linkedin_attempt_abandoned_for_contamination'
            )
            """
        ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("reset_id") != reset_id:
            continue
        observed[row["event_type"]].add(
            (
                int(row["run_id"]),
                int(row["work_unit_id"]),
                int(row["candidate_id"]) if row["candidate_id"] is not None else None,
                int(row["attempt_id"]) if row["attempt_id"] is not None else None,
            )
        )

    expected = {
        "linkedin_execution_invalidated": {
            (row.run_id, row.work_unit_id, None, None)
            for row in action_plan.work_units_to_invalidate
        },
        "linkedin_candidate_released_for_contamination": {
            (row.run_id, row.work_unit_id, row.candidate_id, row.attempt_id)
            for row in action_plan.candidates_to_release
        },
        "linkedin_attempt_abandoned_for_contamination": {
            (row.run_id, row.work_unit_id, row.candidate_id, row.attempt_id)
            for row in action_plan.attempts_to_abandon
        },
    }
    if observed != expected:
        raise ResetToolError("canonical reset events do not match the pre-reset plan")
    _assert_reset_root_event(db_path, reset_id=reset_id)


# File sidecars that teach across sessions but carry no per-execution
# provenance. None is surgically invertible, so the reset backs each up under
# the reset's own backup directory and starts it empty.
_SIDECAR_FILE_TEMPLATES = (
    "bias_monitor-{brief_id}.json",
    "noise_discoveries-{brief_id}.jsonl",
    "execution_plan.json",
)
_SIDECAR_DIR_NAMES = ("market_intel",)


def _sidecar_backup_dir(state_dir: Path, reset_id: str) -> Path:
    return state_dir / "backups" / f"{_RECEIPT_PREFIX}{reset_id}-sidecars"


def _reset_sidecar_artifacts(
    *,
    state_dir: Path,
    brief_id: str,
    reset_id: str,
) -> dict[str, str]:
    """Back up and reset the noninvertible sidecars; return dispositions.

    Idempotent per reset: a sidecar already moved by a prior (crashed)
    artifact phase reports ``already_reset``; one that never existed reports
    ``absent``. A backup-name collision refuses rather than overwrites.
    """

    backup_dir = _sidecar_backup_dir(state_dir, reset_id)
    targets = [
        state_dir / template.format(brief_id=brief_id)
        for template in _SIDECAR_FILE_TEMPLATES
    ]
    targets += [state_dir / name for name in _SIDECAR_DIR_NAMES]
    dispositions: dict[str, str] = {}
    for path in targets:
        destination = backup_dir / path.name
        if path.exists():
            if destination.exists():
                raise ResetToolError(
                    f"sidecar backup already holds {path.name}; refusing to overwrite"
                )
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            dispositions[path.name] = "reset"
        elif destination.exists():
            dispositions[path.name] = "already_reset"
        else:
            dispositions[path.name] = "absent"
    return dispositions


def _completion_event_exists(db_path: Path, *, reset_id: str) -> bool:
    with closing(_open_readonly(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT payload_json
            FROM events
            WHERE event_type = 'linkedin_contamination_reset_completed'
            """
        ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("reset_id") == reset_id:
            return True
    return False


def _assert_reset_root_event(db_path: Path, *, reset_id: str) -> None:
    with closing(_open_readonly(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT payload_json
            FROM events
            WHERE event_type = 'linkedin_contamination_reset'
            """
        ).fetchall()
    matching = 0
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("reset_id") == reset_id:
            matching += 1
    if matching != 1:
        raise ResetToolError(
            "canonical root reset event is missing or duplicated"
        )


def _validate_completed_receipt(
    receipt_path: Path,
    manifest: ReviewedManifest,
    context: ResetContext,
) -> None:
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResetToolError(f"existing reset receipt is unreadable: {receipt_path}") from exc
    expected = {
        "version": RECEIPT_VERSION,
        "status": "completed",
        "reset_id": context.reset_id,
        "brief_id": context.brief_id,
        "manifest_sha256": manifest.sha256,
        "executed_count": len(context.plan.work_units),
    }
    if not isinstance(receipt, dict) or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise ResetToolError(f"existing reset receipt does not match: {receipt_path}")
    if receipt.get("manifest") != manifest.payload:
        raise ResetToolError(
            f"existing reset receipt manifest does not match: {receipt_path}"
        )
    backup_path = _backup_path(receipt_path.parent, context.reset_id)
    expected_backup = {
        "path": str(backup_path.relative_to(receipt_path.parent)),
        "sha256": _sha256_file(backup_path) if backup_path.is_file() else None,
    }
    if receipt.get("backup") != expected_backup:
        raise ResetToolError(
            f"existing reset receipt backup does not match: {receipt_path}"
        )
    _assert_sqlite_integrity(backup_path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _print_plan(context: ResetContext) -> None:
    for work_unit in context.plan.work_units:
        status = "already-invalidated" if work_unit.already_invalidated else "invalidate"
        print(
            f"[reset] execution run={work_unit.run_id} string={work_unit.source_unit_id} "
            f"work_unit={work_unit.work_unit_id} action={status}"
        )
    counts = context.plan.counts()
    print(
        f"[reset] plan brief={context.brief_id} executed={counts['executions']} "
        f"invalidate={counts['executions_to_invalidate']} "
        f"release={counts['candidates_to_release']} "
        f"abandon={counts['attempts_to_abandon']}"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely reset reviewed contaminated LinkedIn executions",
    )
    parser.add_argument(
        "--state-dir",
        required=True,
        help="LinkedIn state directory containing runtime_state.sqlite3",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Reviewed JSON manifest of (run_id, source_unit_id) executions",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the reset; omitted means read-only dry-run",
    )
    parser.add_argument(
        "--confirm",
        help="Exact '<brief-id>:<manifest-sha256>:<executed-count>' confirmation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = load_reviewed_manifest(args.manifest)
        return run_reset_tool(
            state_dir=args.state_dir,
            manifest=manifest,
            apply=bool(args.apply),
            confirmation=args.confirm,
        )
    except Exception as exc:
        print(f"[reset] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
