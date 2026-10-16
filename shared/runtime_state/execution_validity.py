"""Shared SQL predicate for contamination-aware LinkedIn learning reads."""

from __future__ import annotations

import json
import sqlite3

from shared import config


INVALIDATION_EVENT_TYPE = "linkedin_execution_invalidated"
RESET_ROOT_EVENT_TYPE = "linkedin_contamination_reset"
RESET_COMPLETED_EVENT_TYPE = "linkedin_contamination_reset_completed"
_ANCESTRY_WALK_LIMIT = 100


class ContaminationRunStartRefused(RuntimeError):
    """A LinkedIn run start is refused by the contamination-reset contract.

    ``reason`` is one of:

    - ``reset_incomplete`` — the latest reset root event has no completion
      event; the artifact phase is unfinished and the campaign stays blocked.
    - ``invalidated_ancestry`` — the requested resume/clone ancestry contains
      an invalidated execution; form a fresh strategy from the approved brief
      instead of resuming the contaminated queue (F5).
    - ``reset_flag_required`` — the store carries a contamination reset but
      LINKEDIN_CONTAMINATION_RESET_ENABLED is off. The scrub is canonical
      state, not a read-time preference: running flag-off would count
      invalidated history again, so the repaired campaign refuses.
    """

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


def _latest_reset_root(conn: sqlite3.Connection) -> tuple[int, str] | None:
    row = conn.execute(
        """
        SELECT id, payload_json
        FROM events
        WHERE event_type = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (RESET_ROOT_EVENT_TYPE,),
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row[1] or "{}")
    except (TypeError, json.JSONDecodeError):
        payload = {}
    reset_id = str(payload.get("reset_id") or "") if isinstance(payload, dict) else ""
    return int(row[0]), reset_id


def _reset_completed(conn: sqlite3.Connection, reset_id: str) -> bool:
    rows = conn.execute(
        """
        SELECT payload_json
        FROM events
        WHERE event_type = ?
        """,
        (RESET_COMPLETED_EVENT_TYPE,),
    ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row[0] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and str(payload.get("reset_id") or "") == reset_id:
            return True
    return False


def _resume_ancestry(conn: sqlite3.Connection, seed_run_id: int) -> set[int]:
    ancestry: set[int] = set()
    current: int | None = int(seed_run_id)
    for _ in range(_ANCESTRY_WALK_LIMIT):
        if current is None or current in ancestry:
            break
        ancestry.add(current)
        row = conn.execute(
            "SELECT resumed_from_run_id FROM runs WHERE id = ?",
            (current,),
        ).fetchone()
        current = int(row[0]) if row is not None and row[0] is not None else None
    return ancestry


def assert_linkedin_run_start_permitted(
    conn: sqlite3.Connection,
    *,
    resumed_from_run_id: int | None = None,
    clone_work_units_from_run_id: int | None = None,
) -> None:
    """Refuse a LinkedIn run start that violates the reset contract (F5).

    Called by ``RuntimeStateStore.start_run`` for ``source == 'linkedin'``
    before any write. A store with no reset history is unaffected beyond one
    indexed-miss SELECT, flag on or off.
    """

    latest_root = _latest_reset_root(conn)
    if latest_root is None:
        if not config.LINKEDIN_CONTAMINATION_RESET_ENABLED:
            return
    else:
        if not config.LINKEDIN_CONTAMINATION_RESET_ENABLED:
            raise ContaminationRunStartRefused(
                "reset_flag_required",
                "this store carries a LinkedIn contamination reset; "
                "LINKEDIN_CONTAMINATION_RESET_ENABLED must stay on for the "
                "repaired campaign — running flag-off would re-admit "
                "invalidated history",
            )
        _root_event_id, reset_id = latest_root
        if not _reset_completed(conn, reset_id):
            raise ContaminationRunStartRefused(
                "reset_incomplete",
                f"contamination reset {reset_id or '<unknown>'} has no "
                "completion event; rerun the reset tool to finish its "
                "artifact phase before starting a run",
            )

    ancestor_ids: set[int] = set()
    for seed in (resumed_from_run_id, clone_work_units_from_run_id):
        if seed:
            ancestor_ids |= _resume_ancestry(conn, int(seed))
    if not ancestor_ids:
        return
    placeholders = ",".join("?" for _ in ancestor_ids)
    tainted = conn.execute(
        f"""
        SELECT 1
        FROM events AS invalidation
        JOIN work_units AS work_unit
          ON work_unit.id = invalidation.work_unit_id
        WHERE invalidation.event_type = ?
          AND work_unit.run_id IN ({placeholders})
        LIMIT 1
        """,
        (INVALIDATION_EVENT_TYPE, *sorted(ancestor_ids)),
    ).fetchone()
    if tainted is not None:
        raise ContaminationRunStartRefused(
            "invalidated_ancestry",
            "the requested resume/clone ancestry contains an invalidated "
            "LinkedIn execution; form a fresh strategy from the approved "
            "brief instead of resuming the contaminated queue",
        )


def valid_execution_sql(
    work_unit_alias: str,
    *,
    source_expression: str | None = None,
    run_alias: str | None = None,
) -> str:
    """Return the flag-aware SQL expression for a valid learning execution.

    Non-LinkedIn sources retain their existing behavior. A LinkedIn execution
    is valid only when the work unit actually started and no invalidation event
    targets that exact work-unit row.

    Under a LEFT JOIN a row with no work unit joins to an all-NULL alias, so
    ``started_at IS NOT NULL`` is false and ``source != 'linkedin'`` is NULL and
    the row vanishes whatever its source. That strictness is right where every
    production row does carry a work unit — a candidate attempt always does, so
    a NULL one there is unproved provenance and stays excluded. It is wrong for
    adaptation decisions, whose sole production writer
    (``linkedin/orchestrator.py``'s gate-deferred branch, ``search_string=None``)
    records them with no work unit at all: judging that as invalid empties the
    LinkedIn aggregate outright rather than filtering it. Such a query passes
    ``run_alias`` to opt in, and its unlinked rows are then admitted only while
    nothing in the owning run has been invalidated — so a reset still scrubs the
    false lessons of the run that reasoned from contaminated counts.
    """

    if not config.LINKEDIN_CONTAMINATION_RESET_ENABLED:
        return "1 = 1"
    source = source_expression or f"{work_unit_alias}.source"
    unlinked = (
        f"""(
                {work_unit_alias}.id IS NULL
                AND NOT EXISTS (
                    SELECT 1
                    FROM events AS run_invalidation
                    JOIN work_units AS run_invalidation_unit
                      ON run_invalidation_unit.id = run_invalidation.work_unit_id
                    WHERE run_invalidation.event_type = '{INVALIDATION_EVENT_TYPE}'
                      AND run_invalidation_unit.run_id = {run_alias}.id
                )
            )
            OR """
        if run_alias
        else ""
    )
    return f"""
        (
            {unlinked}{source} != 'linkedin'
            OR (
                {work_unit_alias}.started_at IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1
                    FROM events AS execution_invalidation
                    WHERE execution_invalidation.work_unit_id = {work_unit_alias}.id
                      AND execution_invalidation.event_type = '{INVALIDATION_EVENT_TYPE}'
                )
            )
        )
    """


__all__ = [
    "INVALIDATION_EVENT_TYPE",
    "RESET_COMPLETED_EVENT_TYPE",
    "RESET_ROOT_EVENT_TYPE",
    "ContaminationRunStartRefused",
    "assert_linkedin_run_start_permitted",
    "valid_execution_sql",
]
