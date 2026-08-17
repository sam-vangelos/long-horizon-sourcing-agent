"""Read-side outcome monitors for sourcing runs.

M2 keeps this pure: it reads runtime SQLite state, emits one monitor record per
run, and computes baseline rates. It does not mutate pipeline state or
compatibility projections.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from shared.receipts import ReceiptStatus, receipt_from_json


SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}
FAILURE_DECISIONS = {"PARSE_FAILURE", "JUDGMENT_FAILURE"}
NEGATIVE_EVENT_TOKENS = ("error", "failed", "failure")


@dataclass(frozen=True)
class OutcomeMonitorRecord:
    """One emitted monitor row for a runtime run."""

    db_path: str
    run_id: int
    source: str
    brief_id: str
    status: str
    started_at: str
    ended_at: str | None
    all_spans_ok: bool
    candidates_saved: int
    green_but_useless: bool
    judge_decisions: int
    judge_parse_failures: int
    judge_parse_failure_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": self.db_path,
            "run_id": self.run_id,
            "source": self.source,
            "brief_id": self.brief_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "all_spans_ok": self.all_spans_ok,
            "candidates_saved": self.candidates_saved,
            "green_but_useless": self.green_but_useless,
            "judge_decisions": self.judge_decisions,
            "judge_parse_failures": self.judge_parse_failures,
            "judge_parse_failure_rate": self.judge_parse_failure_rate,
        }


@dataclass(frozen=True)
class BaselineRates:
    """Aggregate M2 baseline rates over recent run monitor records."""

    runs_measured: int
    green_but_useless_runs: int
    green_but_useless_rate: float
    judge_decisions: int
    judge_parse_failures: int
    judge_parse_failure_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs_measured": self.runs_measured,
            "green_but_useless_runs": self.green_but_useless_runs,
            "green_but_useless_rate": self.green_but_useless_rate,
            "judge_decisions": self.judge_decisions,
            "judge_parse_failures": self.judge_parse_failures,
            "judge_parse_failure_rate": self.judge_parse_failure_rate,
        }


def discover_runtime_state_dbs(root: str | Path = "output") -> list[Path]:
    """Return runtime_state.sqlite3 files under ``root`` in deterministic order."""

    base = Path(root)
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("runtime_state.sqlite3") if path.is_file())


def emit_outcome_monitors(
    db_paths: Iterable[str | Path],
    *,
    recent_limit: int | None = None,
) -> list[OutcomeMonitorRecord]:
    """Emit one read-side outcome monitor per run."""

    rows: list[tuple[str, int, str, str, str, str, str | None]] = []
    for db_path in db_paths:
        rows.extend(_run_rows(Path(db_path)))
    rows.sort(key=lambda row: row[5] or "", reverse=True)
    if recent_limit is not None:
        rows = rows[:recent_limit]

    records: list[OutcomeMonitorRecord] = []
    for db_path, run_id, source, brief_id, status, started_at, ended_at in rows:
        path = Path(db_path)
        with _open_readonly(path) as conn:
            if conn is None:
                continue
            all_spans_ok = _all_spans_ok(conn, run_id)
            candidates_saved = _candidates_saved(conn, run_id)
            judge_decisions, judge_parse_failures = _judge_parse_counts(conn, run_id)
        records.append(
            OutcomeMonitorRecord(
                db_path=str(path),
                run_id=run_id,
                source=source,
                brief_id=brief_id,
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                all_spans_ok=all_spans_ok,
                candidates_saved=candidates_saved,
                green_but_useless=all_spans_ok and candidates_saved == 0,
                judge_decisions=judge_decisions,
                judge_parse_failures=judge_parse_failures,
                judge_parse_failure_rate=(
                    judge_parse_failures / judge_decisions if judge_decisions else 0.0
                ),
            )
        )
    return records


def baseline_rates(records: Iterable[OutcomeMonitorRecord]) -> BaselineRates:
    records_list = list(records)
    for index, record in enumerate(records_list):
        _validate_baseline_record(record, index=index)
    green = sum(1 for record in records_list if record.green_but_useless)
    decisions = sum(record.judge_decisions for record in records_list)
    parse_failures = sum(record.judge_parse_failures for record in records_list)
    return BaselineRates(
        runs_measured=len(records_list),
        green_but_useless_runs=green,
        green_but_useless_rate=green / len(records_list) if records_list else 0.0,
        judge_decisions=decisions,
        judge_parse_failures=parse_failures,
        judge_parse_failure_rate=parse_failures / decisions if decisions else 0.0,
    )


def _validate_baseline_record(record: OutcomeMonitorRecord, *, index: int) -> None:
    if not isinstance(record, OutcomeMonitorRecord):
        raise ValueError(f"records[{index}] must be an OutcomeMonitorRecord")
    if not isinstance(record.green_but_useless, bool):
        raise ValueError(f"records[{index}].green_but_useless must be a boolean")
    if (
        not isinstance(record.judge_decisions, int)
        or isinstance(record.judge_decisions, bool)
        or record.judge_decisions < 0
    ):
        raise ValueError(f"records[{index}].judge_decisions must be a non-negative integer")
    if (
        not isinstance(record.judge_parse_failures, int)
        or isinstance(record.judge_parse_failures, bool)
        or record.judge_parse_failures < 0
    ):
        raise ValueError(
            f"records[{index}].judge_parse_failures must be a non-negative integer"
        )
    if record.judge_parse_failures > record.judge_decisions:
        raise ValueError(
            f"records[{index}].judge_parse_failures cannot exceed judge_decisions"
        )


@dataclass(frozen=True)
class RunHealth:
    """P4.3: this run's outcome record judged against its own db's
    historical baseline — the wiring ``green_but_useless`` and the judge
    parse-failure baseline needed to actually gate a run as degraded."""

    run_id: int
    green_but_useless: bool
    judge_decisions: int
    judge_parse_failures: int
    judge_parse_failure_rate: float
    baseline_judge_parse_failure_rate: float
    baseline_runs_measured: int
    degraded: bool
    degraded_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "green_but_useless": self.green_but_useless,
            "judge_decisions": self.judge_decisions,
            "judge_parse_failures": self.judge_parse_failures,
            "judge_parse_failure_rate": self.judge_parse_failure_rate,
            "baseline_judge_parse_failure_rate": self.baseline_judge_parse_failure_rate,
            "baseline_runs_measured": self.baseline_runs_measured,
            "degraded": self.degraded,
            "degraded_reasons": list(self.degraded_reasons),
        }


def compute_run_health(
    db_path: str | Path,
    run_id: int,
    *,
    parse_failure_multiplier: float = 1.5,
    parse_failure_floor: float = 0.10,
    min_decisions_for_baseline_check: int = 5,
) -> RunHealth | None:
    """Judge one run against the historical baseline from the same db.

    Two independent degradation signals, matching the monitors this
    function wires together:

    - ``green_but_useless``: every span in the run reports OK, but zero
      candidates were saved — a run that looks healthy by exception-count
      alone but produced nothing.
    - judge parse-failure rate materially above this brief's own
      historical baseline (computed from every OTHER run in the same db,
      so a single noisy run doesn't baseline against itself). The
      threshold is ``max(baseline * multiplier, floor)`` so a baseline of
      0 doesn't make any nonzero rate "infinitely degraded", and runs with
      too few judge decisions to be meaningful are not flagged.

    Returns ``None`` when ``run_id`` has no monitor record in ``db_path``
    (e.g. runtime state absent) — fail-soft, the caller omits the "Run
    health" block rather than emit a false verdict.
    """

    records = emit_outcome_monitors([db_path])
    current = next((record for record in records if record.run_id == run_id), None)
    if current is None:
        return None
    history = [record for record in records if record.run_id != run_id]
    baseline = baseline_rates(history)

    reasons: list[str] = []
    if current.green_but_useless:
        reasons.append("green_but_useless")

    threshold = max(
        baseline.judge_parse_failure_rate * parse_failure_multiplier,
        parse_failure_floor,
    )
    if (
        current.judge_decisions >= min_decisions_for_baseline_check
        and current.judge_parse_failure_rate > threshold
    ):
        reasons.append("judge_parse_failure_rate_above_baseline")

    return RunHealth(
        run_id=run_id,
        green_but_useless=current.green_but_useless,
        judge_decisions=current.judge_decisions,
        judge_parse_failures=current.judge_parse_failures,
        judge_parse_failure_rate=current.judge_parse_failure_rate,
        baseline_judge_parse_failure_rate=baseline.judge_parse_failure_rate,
        baseline_runs_measured=baseline.runs_measured,
        degraded=bool(reasons),
        degraded_reasons=tuple(reasons),
    )


def current_baseline_rates(
    root: str | Path = "output",
    *,
    recent_limit: int = 50,
) -> BaselineRates:
    """Measure M2 baseline rates over the latest runtime DB runs under ``root``."""

    return baseline_rates(
        emit_outcome_monitors(
            discover_runtime_state_dbs(root),
            recent_limit=recent_limit,
        )
    )


def _run_rows(db_path: Path) -> list[tuple[str, int, str, str, str, str, str | None]]:
    with _open_readonly(db_path) as conn:
        if conn is None or not _table_exists(conn, "runs"):
            return []
        rows = conn.execute(
            """
            SELECT id, source, brief_id, status, started_at, ended_at
            FROM runs
            ORDER BY started_at DESC, id DESC
            """
        ).fetchall()
    return [
        (
            str(db_path),
            int(row["id"]),
            str(row["source"]),
            str(row["brief_id"]),
            str(row["status"]),
            str(row["started_at"]),
            str(row["ended_at"]) if row["ended_at"] is not None else None,
        )
        for row in rows
    ]


def _all_spans_ok(conn: sqlite3.Connection, run_id: int) -> bool:
    if _table_exists(conn, "run_event_log"):
        mirror_rows = conn.execute(
            """
            SELECT legacy_event_id, receipt_json
            FROM run_event_log
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
        legacy_event_count = _count_rows(conn, "events", "run_id = ?", (run_id,))
        if mirror_rows:
            if len(mirror_rows) != legacy_event_count:
                return False
            for row in mirror_rows:
                try:
                    receipt = receipt_from_json(row["receipt_json"])
                except Exception:
                    return False
                if receipt.actual_status != ReceiptStatus.OK:
                    return False
            return True

    failed_attempts = _count_rows(
        conn,
        "candidate_attempts",
        "run_id = ? AND status IN ('failed', 'reconciled')",
        (run_id,),
    )
    errored_units = _count_rows(
        conn,
        "work_units",
        "run_id = ? AND status = 'error'",
        (run_id,),
    )
    negative_events = 0
    if _table_exists(conn, "events"):
        event_rows = conn.execute(
            "SELECT event_type FROM events WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        for row in event_rows:
            event_type = str(row["event_type"]).lower()
            if any(token in event_type for token in NEGATIVE_EVENT_TOKENS):
                negative_events += 1
    return failed_attempts == 0 and errored_units == 0 and negative_events == 0


def _candidates_saved(conn: sqlite3.Connection, run_id: int) -> int:
    run_saved = 0
    row = conn.execute(
        "SELECT resume_state_json FROM runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    if row is not None:
        resume_state = _safe_json(row["resume_state_json"])
        run_saved = _coerce_int(resume_state.get("candidates_saved"))

    work_unit_saved = 0
    if _table_exists(conn, "work_units"):
        row = conn.execute(
            "SELECT COALESCE(SUM(saves_count), 0) AS saves FROM work_units WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        work_unit_saved = _coerce_int(row["saves"] if row else 0)

    # P1.2: count saves by side-effect ledger status ('succeeded'), not by
    # judge terminal decision — a judge SAVE whose pipeline click failed is
    # not a save. When the run has save side-effect rows, the ledger is
    # authoritative; the judge-decision count remains only as the legacy
    # fallback for DBs/runs that predate the ledger.
    candidate_saved = 0
    if _table_exists(conn, "side_effects"):
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS attempted,
                COALESCE(SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END), 0) AS succeeded
            FROM side_effects
            WHERE run_id = ?
              AND effect_type LIKE '%save%'
            """,
            (run_id,),
        ).fetchone()
        if row and _coerce_int(row["attempted"]) > 0:
            # The ledger is authoritative when present — no max() with the
            # softer sources, which can overcount (judge decisions,
            # pre-fix stats that included already-present skips).
            return _coerce_int(row["succeeded"])
    if _table_exists(conn, "candidates") and _table_exists(conn, "work_units"):
        placeholders = ",".join("?" for _ in SAVE_DECISIONS)
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS saves
            FROM candidates c
            JOIN work_units wu ON wu.id = c.last_work_unit_id
            WHERE wu.run_id = ?
              AND c.terminal_decision IN ({placeholders})
            """,
            (run_id, *sorted(SAVE_DECISIONS)),
        ).fetchone()
        candidate_saved = _coerce_int(row["saves"] if row else 0)
    return max(run_saved, work_unit_saved, candidate_saved)


def _judge_parse_counts(conn: sqlite3.Connection, run_id: int) -> tuple[int, int]:
    attempt_total, attempt_parse_failures = _attempt_parse_counts(conn, run_id)
    candidate_total, candidate_parse_failures = _candidate_parse_counts(conn, run_id)
    if attempt_total:
        return attempt_total, attempt_parse_failures
    return candidate_total, candidate_parse_failures


def _is_abandoned_recovery_payload(payload: Any) -> bool:
    """True for the synthetic settle a contained resume skip writes.

    ``Pipeline._abandon_unrecoverable_pending_full`` records a JUDGMENT_FAILURE
    to settle a pending review the live Recruiter surface could not re-match.
    No judge ever ran, so grading it as a parse failure would report a provider
    problem that did not happen — and the marker rides both the attempt payload
    and the candidate terminal payload, so both readers here must honor it.
    """

    return bool(
        isinstance(payload, dict)
        and payload.get("pending_full_recovery_abandoned")
    )


def _attempt_parse_counts(conn: sqlite3.Connection, run_id: int) -> tuple[int, int]:
    if not _table_exists(conn, "candidate_attempts"):
        return 0, 0
    rows = conn.execute(
        """
        SELECT stage, status, failure_kind, failure_reason, payload_json
        FROM candidate_attempts
        WHERE run_id = ?
          AND stage IN ('facial', 'full')
        """,
        (run_id,),
    ).fetchall()
    total = 0
    parse_failures = 0
    for row in rows:
        column_haystack = " ".join(
            str(row[key] or "")
            for key in ("status", "failure_kind", "failure_reason")
        )
        payload = _safe_json(row["payload_json"])
        if _is_abandoned_recovery_payload(payload):
            continue
        decision = _extract_decision(payload)
        if (
            decision
            or "PARSE_FAILURE" in column_haystack
            or "JUDGMENT_FAILURE" in column_haystack
        ):
            total += 1
        if decision in FAILURE_DECISIONS or "PARSE_FAILURE" in column_haystack:
            parse_failures += 1
    return total, parse_failures


def _candidate_parse_counts(conn: sqlite3.Connection, run_id: int) -> tuple[int, int]:
    if not (_table_exists(conn, "candidates") and _table_exists(conn, "work_units")):
        return 0, 0
    rows = conn.execute(
        """
        SELECT c.terminal_decision, c.terminal_payload_json
        FROM candidates c
        JOIN work_units wu ON wu.id = c.last_work_unit_id
        WHERE wu.run_id = ?
          AND c.terminal_decision IS NOT NULL
        """,
        (run_id,),
    ).fetchall()
    total = 0
    parse_failures = 0
    for row in rows:
        payload = _safe_json(row["terminal_payload_json"])
        if _is_abandoned_recovery_payload(payload):
            continue
        payload_decision = _extract_decision(payload)
        decision = str(row["terminal_decision"] or "")
        total += 1
        if decision in FAILURE_DECISIONS or payload_decision in FAILURE_DECISIONS:
            parse_failures += 1
    return total, parse_failures


def _extract_decision(payload: Any) -> str:
    if isinstance(payload, dict):
        raw = payload.get("decision")
        if isinstance(raw, str):
            return raw
        for value in payload.values():
            decision = _extract_decision(value)
            if decision:
                return decision
    if isinstance(payload, list):
        for value in payload:
            decision = _extract_decision(value)
            if decision:
                return decision
    return ""


def _count_rows(
    conn: sqlite3.Connection,
    table: str,
    where_sql: str,
    params: tuple[Any, ...],
) -> int:
    if not _table_exists(conn, table):
        return 0
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM {table} WHERE {where_sql}",
        params,
    ).fetchone()
    return _coerce_int(row["n"] if row else 0)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _safe_json(raw: Any) -> Any:
    if raw is None:
        return {}
    try:
        return json.loads(str(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _open_readonly(path: Path):
    if not path.exists():
        return _NullContext(None)
    uri = f"file:{path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return _NullContext(None)
    conn.row_factory = sqlite3.Row
    return conn


class _NullContext:
    def __init__(self, value: Any):
        self.value = value

    def __enter__(self) -> Any:
        return self.value

    def __exit__(self, *_exc: object) -> None:
        return None
