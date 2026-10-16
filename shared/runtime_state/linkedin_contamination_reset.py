"""Guarded reset of manifest-proved contaminated LinkedIn executions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from shared import config
from shared.contracts import SAVE_DECISIONS
from shared.runtime_state.execution_validity import (
    RESET_COMPLETED_EVENT_TYPE,
    RESET_ROOT_EVENT_TYPE,
)
from shared.runtime_state.store import LINKEDIN_STRING_KIND, RuntimeStateStore


INVALIDATION_EVENT_TYPE = "linkedin_execution_invalidated"
RELEASE_EVENT_TYPE = "linkedin_candidate_released_for_contamination"
ABANDON_EVENT_TYPE = "linkedin_attempt_abandoned_for_contamination"
_STARTED_LIFECYCLE_STATES = frozenset({"facial_started", "full_started"})
_RESET_REASON = "facet_contamination"


class ContaminationResetError(RuntimeError):
    """Base error for a rejected or stale contamination reset."""


class ContaminationManifestError(ContaminationResetError):
    """The reviewed manifest does not resolve to exact executed strings."""


class ContaminationResetDisabledError(ContaminationResetError):
    """Apply was attempted while reset-aware behavior is disabled."""


class StaleContaminationResetError(ContaminationResetError):
    """Canonical state changed between reset planning and guarded mutation."""


@dataclass(frozen=True)
class ManifestWorkUnit:
    run_id: int
    source_unit_id: str
    work_unit_id: int
    brief_id: str
    status: str
    started_at: str
    ended_at: str | None
    payload_json: str
    checkpoint_json: str
    metrics_json: str
    counters: tuple[tuple[str, int], ...]
    already_invalidated: bool


@dataclass(frozen=True)
class CandidateRelease:
    candidate_id: int
    identity_key: str
    lifecycle_before: str
    terminal_decision_before: str | None
    run_id: int
    work_unit_id: int
    attempt_id: int | None
    decision_event_id: int | None
    provenance: str


@dataclass(frozen=True)
class AttemptAbandonment:
    attempt_id: int
    candidate_id: int
    identity_key: str
    run_id: int
    work_unit_id: int
    stage: str
    lifecycle_before: str


@dataclass(frozen=True)
class CandidateDisposition:
    candidate_id: int
    identity_key: str
    reason: str


@dataclass(frozen=True)
class ContaminationResetPlan:
    work_units: tuple[ManifestWorkUnit, ...]
    candidates_to_release: tuple[CandidateRelease, ...]
    attempts_to_abandon: tuple[AttemptAbandonment, ...]
    untouched_candidates: tuple[CandidateDisposition, ...]

    @property
    def work_units_to_invalidate(self) -> tuple[ManifestWorkUnit, ...]:
        return tuple(row for row in self.work_units if not row.already_invalidated)

    @property
    def is_noop(self) -> bool:
        return not (
            self.work_units_to_invalidate
            or self.candidates_to_release
            or self.attempts_to_abandon
        )

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {
            "executions": len(self.work_units),
            "executions_to_invalidate": len(self.work_units_to_invalidate),
            "candidates_to_release": len(self.candidates_to_release),
            "attempts_to_abandon": len(self.attempts_to_abandon),
        }
        for candidate in self.untouched_candidates:
            key = f"untouched_{candidate.reason}"
            counts[key] = counts.get(key, 0) + 1
        return counts


def plan_linkedin_contamination_reset(
    conn: sqlite3.Connection,
    manifest: Iterable[tuple[int, str]],
) -> ContaminationResetPlan:
    """Plan a reset without mutating canonical state.

    The caller owns ``conn``. It may be a SQLite ``mode=ro`` connection.
    """

    conn.row_factory = sqlite3.Row
    reviewed_pairs = _normalize_manifest(manifest)
    work_units = tuple(
        _resolve_manifest_work_unit(conn, run_id, source_unit_id)
        for run_id, source_unit_id in reviewed_pairs
    )
    manifest_work_unit_ids = {row.work_unit_id for row in work_units}
    candidates = _load_manifest_candidate_cohort(conn, manifest_work_unit_ids)

    releases: list[CandidateRelease] = []
    abandonments: list[AttemptAbandonment] = []
    untouched: list[CandidateDisposition] = []
    for candidate in candidates:
        disposition, release, abandonment = _classify_candidate(
            conn,
            candidate,
            manifest_work_unit_ids=manifest_work_unit_ids,
        )
        if release is not None:
            releases.append(release)
        if abandonment is not None:
            abandonments.append(abandonment)
        if disposition is not None:
            untouched.append(disposition)

    return ContaminationResetPlan(
        work_units=work_units,
        candidates_to_release=tuple(releases),
        attempts_to_abandon=tuple(abandonments),
        untouched_candidates=tuple(untouched),
    )


def apply_linkedin_contamination_reset(
    store: RuntimeStateStore,
    *,
    manifest: Iterable[tuple[int, str]],
    reset_id: str,
    manifest_digest: str | None = None,
) -> ContaminationResetPlan:
    """Apply one manifest reset atomically and return its pre-write plan."""

    if not config.LINKEDIN_CONTAMINATION_RESET_ENABLED:
        raise ContaminationResetDisabledError(
            "LINKEDIN_CONTAMINATION_RESET_ENABLED must be true before apply"
        )
    normalized_reset_id = str(reset_id).strip()
    if not normalized_reset_id:
        raise ValueError("reset_id must be non-empty")
    reviewed_pairs = _normalize_manifest(manifest)
    reset_at = _utc_now()

    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            plan = plan_linkedin_contamination_reset(conn, reviewed_pairs)
            if not plan.is_noop:
                # The root receipt anchors the reset in canonical SQLite:
                # run-start protection (F5) keys off it, and its absence of a
                # matching completion event keeps the campaign blocked while
                # the artifact phase is unfinished.
                store.record_event(
                    conn=conn,
                    event_type=RESET_ROOT_EVENT_TYPE,
                    payload={
                        "reset_id": normalized_reset_id,
                        "manifest_digest": manifest_digest,
                        "reset_at": reset_at,
                        "counts": plan.counts(),
                    },
                )
            _write_invalidations(
                store,
                conn,
                plan.work_units_to_invalidate,
                reset_id=normalized_reset_id,
            )
            _write_attempt_abandonments(
                store,
                conn,
                plan.attempts_to_abandon,
                reset_id=normalized_reset_id,
                reset_at=reset_at,
            )
            _write_candidate_releases(
                store,
                conn,
                plan.candidates_to_release,
                reset_id=normalized_reset_id,
            )
        except Exception:
            conn.rollback()
            raise
    return plan


def _normalize_manifest(
    manifest: Iterable[tuple[int, str]],
) -> tuple[tuple[int, str], ...]:
    normalized: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for raw_pair in manifest:
        if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
            raise ContaminationManifestError(
                "manifest entries must be (run_id, source_unit_id) pairs"
            )
        raw_run_id, raw_source_unit_id = raw_pair
        if isinstance(raw_run_id, bool) or not isinstance(raw_run_id, int):
            raise ContaminationManifestError("manifest run_id must be an integer")
        run_id = raw_run_id
        source_unit_id = str(raw_source_unit_id).strip()
        if run_id <= 0 or not source_unit_id:
            raise ContaminationManifestError(
                "manifest run_id and source_unit_id must be non-empty"
            )
        pair = (run_id, source_unit_id)
        if pair in seen:
            raise ContaminationManifestError(
                f"duplicate manifest execution: {run_id}:{source_unit_id}"
            )
        seen.add(pair)
        normalized.append(pair)
    if not normalized:
        raise ContaminationManifestError("manifest must contain at least one execution")
    return tuple(sorted(normalized))


def _resolve_manifest_work_unit(
    conn: sqlite3.Connection,
    run_id: int,
    source_unit_id: str,
) -> ManifestWorkUnit:
    rows = conn.execute(
        """
        SELECT wu.*, r.source AS run_source, r.brief_id AS run_brief_id
        FROM work_units AS wu
        JOIN runs AS r ON r.id = wu.run_id
        WHERE wu.run_id = ? AND wu.source_unit_id = ?
        ORDER BY wu.id ASC
        """,
        (run_id, source_unit_id),
    ).fetchall()
    if len(rows) != 1:
        raise ContaminationManifestError(
            f"manifest execution {run_id}:{source_unit_id} resolved to {len(rows)} work units"
        )
    row = rows[0]
    if (
        row["source"] != "linkedin"
        or row["run_source"] != "linkedin"
        or row["kind"] != LINKEDIN_STRING_KIND
        or row["brief_id"] != row["run_brief_id"]
        or row["started_at"] is None
    ):
        raise ContaminationManifestError(
            f"manifest execution {run_id}:{source_unit_id} is not an executed LinkedIn string"
        )
    already_invalidated = conn.execute(
        """
        SELECT 1
        FROM events
        WHERE work_unit_id = ? AND event_type = ?
        LIMIT 1
        """,
        (int(row["id"]), INVALIDATION_EVENT_TYPE),
    ).fetchone()
    counter_names = (
        "result_count",
        "candidates_discovered",
        "candidates_enriched",
        "candidates_insufficient",
        "facial_yes_count",
        "facial_no_count",
        "facial_borderline_count",
        "saves_count",
        "rejected_count",
    )
    return ManifestWorkUnit(
        run_id=run_id,
        source_unit_id=source_unit_id,
        work_unit_id=int(row["id"]),
        brief_id=str(row["brief_id"]),
        status=str(row["status"]),
        started_at=str(row["started_at"]),
        ended_at=row["ended_at"],
        payload_json=str(row["payload_json"]),
        checkpoint_json=str(row["checkpoint_json"]),
        metrics_json=str(row["metrics_json"]),
        counters=tuple((name, int(row[name] or 0)) for name in counter_names),
        already_invalidated=already_invalidated is not None,
    )


def _load_manifest_candidate_cohort(
    conn: sqlite3.Connection,
    work_unit_ids: set[int],
) -> tuple[sqlite3.Row, ...]:
    placeholders = ",".join("?" for _ in work_unit_ids)
    ordered_ids = sorted(work_unit_ids)
    rows = conn.execute(
        f"""
        SELECT c.*
        FROM candidates AS c
        WHERE c.source = 'linkedin'
          AND c.id IN (
              SELECT e.candidate_id
              FROM events AS e
              WHERE e.work_unit_id IN ({placeholders})
                AND e.event_type = 'candidate_discovered'
                AND e.candidate_id IS NOT NULL
              UNION
              SELECT ca.candidate_id
              FROM candidate_attempts AS ca
              WHERE ca.work_unit_id IN ({placeholders})
          )
        ORDER BY c.id ASC
        """,
        (*ordered_ids, *ordered_ids),
    ).fetchall()
    return tuple(rows)


def _classify_candidate(
    conn: sqlite3.Connection,
    candidate: sqlite3.Row,
    *,
    manifest_work_unit_ids: set[int],
) -> tuple[
    CandidateDisposition | None,
    CandidateRelease | None,
    AttemptAbandonment | None,
]:
    candidate_id = int(candidate["id"])
    identity_key = str(candidate["identity_key"])
    if candidate["user_status"] is not None:
        return CandidateDisposition(candidate_id, identity_key, "recruiter_touched"), None, None
    if candidate["terminal_decision"] in SAVE_DECISIONS:
        return CandidateDisposition(candidate_id, identity_key, "save_family"), None, None

    terminal_decision = candidate["terminal_decision"]
    if terminal_decision is not None:
        proof = _prove_current_terminal_decision(conn, candidate)
        if proof is None:
            return CandidateDisposition(candidate_id, identity_key, "ambiguous"), None, None
        event, attempt = proof
        work_unit_id = int(event["work_unit_id"])
        if work_unit_id not in manifest_work_unit_ids:
            return CandidateDisposition(candidate_id, identity_key, "later_clean"), None, None
        return (
            None,
            CandidateRelease(
                candidate_id=candidate_id,
                identity_key=identity_key,
                lifecycle_before=str(candidate["current_lifecycle_state"]),
                terminal_decision_before=str(terminal_decision),
                run_id=int(event["run_id"]),
                work_unit_id=work_unit_id,
                attempt_id=(int(attempt["id"]) if attempt is not None else None),
                decision_event_id=int(event["id"]),
                provenance=str(event["event_type"]),
            ),
            None,
        )

    lifecycle = str(candidate["current_lifecycle_state"])
    if lifecycle not in _STARTED_LIFECYCLE_STATES:
        return CandidateDisposition(candidate_id, identity_key, "nonblocking"), None, None
    attempt = _prove_current_open_attempt(conn, candidate)
    if attempt is None:
        return CandidateDisposition(candidate_id, identity_key, "ambiguous"), None, None
    work_unit_id = int(attempt["work_unit_id"])
    if work_unit_id not in manifest_work_unit_ids:
        return CandidateDisposition(candidate_id, identity_key, "later_clean"), None, None
    release = CandidateRelease(
        candidate_id=candidate_id,
        identity_key=identity_key,
        lifecycle_before=lifecycle,
        terminal_decision_before=None,
        run_id=int(attempt["run_id"]),
        work_unit_id=work_unit_id,
        attempt_id=int(attempt["id"]),
        decision_event_id=None,
        provenance="open_attempt",
    )
    abandonment = AttemptAbandonment(
        attempt_id=int(attempt["id"]),
        candidate_id=candidate_id,
        identity_key=identity_key,
        run_id=int(attempt["run_id"]),
        work_unit_id=work_unit_id,
        stage=str(attempt["stage"]),
        lifecycle_before=lifecycle,
    )
    return None, release, abandonment


def _prove_current_terminal_decision(
    conn: sqlite3.Connection,
    candidate: sqlite3.Row,
) -> tuple[sqlite3.Row, sqlite3.Row | None] | None:
    events = conn.execute(
        """
        SELECT *
        FROM events
        WHERE candidate_id = ?
          AND event_type IN ('attempt_succeeded', 'candidate_state_transition')
        ORDER BY id DESC
        """,
        (int(candidate["id"]),),
    ).fetchall()
    decision_event: sqlite3.Row | None = None
    for event in events:
        payload = _json_object(event["payload_json"])
        if payload.get("terminal_decision") is not None:
            decision_event = event
            break
    if decision_event is None:
        return None
    payload = _json_object(decision_event["payload_json"])
    if payload.get("terminal_decision") != candidate["terminal_decision"]:
        return None
    if decision_event["run_id"] is None or decision_event["work_unit_id"] is None:
        return None
    work_unit = _load_executed_linkedin_work_unit(
        conn,
        int(decision_event["work_unit_id"]),
    )
    if work_unit is None:
        return None
    if (
        int(decision_event["run_id"]) != int(work_unit["run_id"])
        or str(candidate["brief_id"]) != str(work_unit["brief_id"])
    ):
        return None

    attempt: sqlite3.Row | None = None
    if decision_event["event_type"] == "attempt_succeeded":
        if decision_event["attempt_id"] is None:
            return None
        attempt = conn.execute(
            "SELECT * FROM candidate_attempts WHERE id = ?",
            (int(decision_event["attempt_id"]),),
        ).fetchone()
        if not _succeeded_attempt_matches(
            attempt,
            candidate_id=int(candidate["id"]),
            run_id=int(decision_event["run_id"]),
            work_unit_id=int(decision_event["work_unit_id"]),
            last_attempt_id=candidate["last_attempt_id"],
        ):
            return None
        if (
            payload.get("new_state") != candidate["current_lifecycle_state"]
            or payload.get("stage") != attempt["stage"]
        ):
            return None
    elif decision_event["event_type"] == "candidate_state_transition":
        if decision_event["attempt_id"] is not None:
            return None
        if payload.get("to_state") != candidate["current_lifecycle_state"]:
            return None
        if candidate["last_attempt_id"] is not None:
            attempt = conn.execute(
                "SELECT * FROM candidate_attempts WHERE id = ?",
                (int(candidate["last_attempt_id"]),),
            ).fetchone()
            if not _succeeded_attempt_matches(
                attempt,
                candidate_id=int(candidate["id"]),
                run_id=int(decision_event["run_id"]),
                work_unit_id=int(decision_event["work_unit_id"]),
                last_attempt_id=candidate["last_attempt_id"],
            ):
                return None
    else:
        return None
    return decision_event, attempt


def _succeeded_attempt_matches(
    attempt: sqlite3.Row | None,
    *,
    candidate_id: int,
    run_id: int,
    work_unit_id: int,
    last_attempt_id: int | None,
) -> bool:
    return bool(
        attempt is not None
        and last_attempt_id is not None
        and int(attempt["id"]) == int(last_attempt_id)
        and int(attempt["candidate_id"]) == candidate_id
        and int(attempt["run_id"]) == run_id
        and attempt["work_unit_id"] is not None
        and int(attempt["work_unit_id"]) == work_unit_id
        and attempt["status"] == "succeeded"
    )


def _prove_current_open_attempt(
    conn: sqlite3.Connection,
    candidate: sqlite3.Row,
) -> sqlite3.Row | None:
    if candidate["last_attempt_id"] is None:
        return None
    attempt = conn.execute(
        "SELECT * FROM candidate_attempts WHERE id = ?",
        (int(candidate["last_attempt_id"]),),
    ).fetchone()
    if (
        attempt is None
        or int(attempt["candidate_id"]) != int(candidate["id"])
        or attempt["status"] != "started"
        or attempt["stage"] not in {"facial", "full"}
        or attempt["work_unit_id"] is None
    ):
        return None
    work_unit = _load_executed_linkedin_work_unit(conn, int(attempt["work_unit_id"]))
    if work_unit is None:
        return None
    if (
        int(attempt["run_id"]) != int(work_unit["run_id"])
        or str(candidate["brief_id"]) != str(work_unit["brief_id"])
    ):
        return None
    return attempt


def _load_executed_linkedin_work_unit(
    conn: sqlite3.Connection,
    work_unit_id: int,
) -> sqlite3.Row | None:
    row = conn.execute(
        """
        SELECT wu.*
        FROM work_units AS wu
        JOIN runs AS r ON r.id = wu.run_id
        WHERE wu.id = ?
          AND wu.source = 'linkedin'
          AND r.source = 'linkedin'
          AND wu.kind = ?
          AND wu.started_at IS NOT NULL
          AND wu.brief_id = r.brief_id
        """,
        (work_unit_id, LINKEDIN_STRING_KIND),
    ).fetchone()
    return row


def _write_invalidations(
    store: RuntimeStateStore,
    conn: sqlite3.Connection,
    work_units: tuple[ManifestWorkUnit, ...],
    *,
    reset_id: str,
) -> None:
    for work_unit in work_units:
        store.record_event(
            conn=conn,
            event_type=INVALIDATION_EVENT_TYPE,
            run_id=work_unit.run_id,
            work_unit_id=work_unit.work_unit_id,
            payload={
                "reset_id": reset_id,
                "reason": _RESET_REASON,
                "source_unit_id": work_unit.source_unit_id,
                "status_before": work_unit.status,
                "started_at": work_unit.started_at,
                "ended_at": work_unit.ended_at,
                "counters": dict(work_unit.counters),
                "payload_sha256": _text_sha256(work_unit.payload_json),
                "checkpoint_sha256": _text_sha256(work_unit.checkpoint_json),
                "metrics_sha256": _text_sha256(work_unit.metrics_json),
            },
        )


def _write_attempt_abandonments(
    store: RuntimeStateStore,
    conn: sqlite3.Connection,
    attempts: tuple[AttemptAbandonment, ...],
    *,
    reset_id: str,
    reset_at: str,
) -> None:
    for attempt in attempts:
        cursor = conn.execute(
            """
            UPDATE candidate_attempts
            SET status = 'failed', failure_kind = 'contamination_reset', ended_at = ?
            WHERE id = ?
              AND candidate_id = ?
              AND run_id = ?
              AND work_unit_id = ?
              AND stage = ?
              AND status = 'started'
            """,
            (
                reset_at,
                attempt.attempt_id,
                attempt.candidate_id,
                attempt.run_id,
                attempt.work_unit_id,
                attempt.stage,
            ),
        )
        if cursor.rowcount != 1:
            raise StaleContaminationResetError(
                f"open attempt {attempt.attempt_id} changed during reset"
            )
        store.record_event(
            conn=conn,
            event_type=ABANDON_EVENT_TYPE,
            run_id=attempt.run_id,
            work_unit_id=attempt.work_unit_id,
            candidate_id=attempt.candidate_id,
            attempt_id=attempt.attempt_id,
            payload={
                "reset_id": reset_id,
                "reason": _RESET_REASON,
                "status_before": "started",
                "stage": attempt.stage,
                "lifecycle_before": attempt.lifecycle_before,
            },
        )


def _write_candidate_releases(
    store: RuntimeStateStore,
    conn: sqlite3.Connection,
    candidates: tuple[CandidateRelease, ...],
    *,
    reset_id: str,
) -> None:
    for candidate in candidates:
        if candidate.terminal_decision_before is None:
            terminal_predicate = "terminal_decision IS NULL"
            parameters: tuple[object, ...] = (
                candidate.candidate_id,
                candidate.lifecycle_before,
                candidate.attempt_id,
            )
        else:
            terminal_predicate = "terminal_decision = ?"
            parameters = (
                candidate.candidate_id,
                candidate.lifecycle_before,
                candidate.attempt_id,
                candidate.terminal_decision_before,
            )
        cursor = conn.execute(
            f"""
            UPDATE candidates
            SET terminal_decision = NULL,
                terminal_payload_json = '{{}}',
                current_lifecycle_state = 'failed_retryable'
            WHERE id = ?
              AND current_lifecycle_state = ?
              AND last_attempt_id IS ?
              AND user_status IS NULL
              AND {terminal_predicate}
            """,
            parameters,
        )
        if cursor.rowcount != 1:
            raise StaleContaminationResetError(
                f"candidate {candidate.candidate_id} changed during reset"
            )
        store.record_event(
            conn=conn,
            event_type=RELEASE_EVENT_TYPE,
            run_id=candidate.run_id,
            work_unit_id=candidate.work_unit_id,
            candidate_id=candidate.candidate_id,
            attempt_id=candidate.attempt_id,
            payload={
                "reset_id": reset_id,
                "reason": _RESET_REASON,
                "lifecycle_before": candidate.lifecycle_before,
                "terminal_decision_before": candidate.terminal_decision_before,
                "decision_event_id": candidate.decision_event_id,
                "provenance": candidate.provenance,
            },
        )


def _json_object(raw: object) -> dict:
    try:
        value = json.loads(str(raw or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_contamination_reset_completed(
    store: RuntimeStateStore,
    *,
    reset_id: str,
    payload: dict | None = None,
) -> None:
    """Append the reset's completion event in a final short transaction.

    Written only after every artifact phase (projections, sidecar resets,
    receipt) has finished and verified. Its absence keeps run start blocked
    (``reset_incomplete``), which is what makes a crashed artifact phase
    resumable rather than silently half-applied. Idempotent per reset_id.
    """

    if not config.LINKEDIN_CONTAMINATION_RESET_ENABLED:
        raise ContaminationResetDisabledError(
            "LINKEDIN_CONTAMINATION_RESET_ENABLED must be true before apply"
        )
    normalized_reset_id = str(reset_id).strip()
    if not normalized_reset_id:
        raise ValueError("reset_id must be non-empty")
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT payload_json FROM events WHERE event_type = ?",
            (RESET_COMPLETED_EVENT_TYPE,),
        ).fetchall()
        for row in rows:
            try:
                existing = json.loads(row[0] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if (
                isinstance(existing, dict)
                and str(existing.get("reset_id") or "") == normalized_reset_id
            ):
                return
        store.record_event(
            conn=conn,
            event_type=RESET_COMPLETED_EVENT_TYPE,
            payload={
                "reset_id": normalized_reset_id,
                "completed_at": _utc_now(),
                **(payload or {}),
            },
        )


def append_contamination_reset_boundary(
    state_dir: "str | Path",
    *,
    reset_id: str,
    sidecars: list[str],
) -> None:
    """Append ONE contamination_reset_boundary row per reset to run_log.jsonl.

    Every existing row is preserved (forensic history); boundary-aware
    readers (the live market-intel advisory) use only rows after the latest
    boundary, so the freshly reset sidecar never re-ingests contaminated
    events. Idempotent per reset_id.
    """

    from pathlib import Path as _Path

    from shared.storage import log_event, read_jsonl

    run_log_path = _Path(state_dir) / "run_log.jsonl"
    for row in read_jsonl(run_log_path):
        if (
            isinstance(row, dict)
            and row.get("event") == "contamination_reset_boundary"
            and row.get("reset_id") == reset_id
        ):
            return
    log_event(
        run_log_path,
        "contamination_reset_boundary",
        reset_id=reset_id,
        sidecars=sorted(sidecars),
    )


__all__ = [
    "ABANDON_EVENT_TYPE",
    "INVALIDATION_EVENT_TYPE",
    "RELEASE_EVENT_TYPE",
    "AttemptAbandonment",
    "CandidateDisposition",
    "CandidateRelease",
    "ContaminationManifestError",
    "ContaminationResetDisabledError",
    "ContaminationResetError",
    "ContaminationResetPlan",
    "ManifestWorkUnit",
    "StaleContaminationResetError",
    "append_contamination_reset_boundary",
    "apply_linkedin_contamination_reset",
    "plan_linkedin_contamination_reset",
    "record_contamination_reset_completed",
]
