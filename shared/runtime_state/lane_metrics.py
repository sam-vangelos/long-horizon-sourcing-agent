"""P5 — lane-attributed read model over canonical runtime state.

Aggregates per-lane sourcing-quality metrics for a run from the
``runtime_state.sqlite3`` tables that already exist; no new tables.
Source-of-truth invariant: this module reads SQLite read-only and never
consults JSON/JSONL projections. Diverging projections cannot influence
the aggregated counts.

Aggregation seams (all already populated upstream):

- ``work_units.payload_json`` carries ``lane_id`` / ``lane_name`` /
  ``lane_intent`` / ``acquisition_mode`` (from ``SearchString.to_dict()``
  via ``shared.runtime_state.linkedin_progress_sync.sync_linkedin_progress``).
- ``candidates.terminal_payload_json["lane"]["lane_id"]`` is written by
  ``linkedin/orchestrator.py`` for REVIEW outcomes only (P4) — wins on
  attribution when present.
- ``candidates.terminal_payload_json["full_decision"]`` is the
  serialized ``OpusDecision.to_dict()``; carries
  ``review_reason_code`` for REVIEW decisions (P4).
- ``candidate_attempts.stage`` / ``status`` + ``run_id`` drive
  per-run open / evaluated counts.

The reader keeps the same layering as ``shared.runtime_state.read_models``:
no import of ``shared.runtime_state.store`` (the writer's ``__init__``
runs DDL on instantiation; the read path deliberately avoids that).

Lane caps and lifecycle decisions land alongside P7 / variant execution.
This module is a read primitive only.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from shared.contracts import NON_SAVE_REVIEW_DECISIONS, SAVE_DECISIONS


LEGACY_LANE_ID = "legacy"
UNSPECIFIED_REVIEW_REASON = "unspecified"


@dataclass(frozen=True)
class LaneMetricsRow:
    """Per-lane aggregated metrics for one run.

    ``lane_id == "legacy"`` is the catch-all bucket for candidates whose
    canonical work unit / terminal payload carry no ``lane_id``. The
    ``legacy`` flag is the boolean form of the same signal for callers
    that prefer not to string-compare. ``cost_usd`` is ``None`` when no
    canonical cost write has happened yet (P5 leaves cost as best-effort
    pass-through; future writers can populate ``work_units.metrics_json``
    ``cost_usd`` or ``total_cost`` keys without changing the read model).
    """

    lane_id: str
    lane_name: str = ""
    lane_intent: str = ""
    acquisition_mode: str = ""
    result_count: int = 0
    candidates_seen: int = 0
    opened_count: int = 0
    evaluated_count: int = 0
    facial_yes_count: int = 0
    facial_no_count: int = 0
    facial_borderline_count: int = 0
    save_count: int = 0
    reject_count: int = 0
    review_count: int = 0
    review_by_reason: dict[str, int] = field(default_factory=dict)
    work_unit_source_ids: tuple[str, ...] = field(default_factory=tuple)
    cost_usd: float | None = None
    legacy: bool = False


def candidate_lane_attribution(
    candidate_terminal_payload: dict | None,
    work_unit_payload: dict | None,
) -> str:
    """Return the canonical lane_id for a candidate.

    Precedence:

    1. ``candidate_terminal_payload["lane"]["lane_id"]`` (P4 writes this
       for REVIEW outcomes; future slices may extend to SAVE / REJECT).
    2. ``work_unit_payload["lane_id"]`` (set by ``SearchString.lane_id``
       at work-unit write time).
    3. ``LEGACY_LANE_ID`` fallback.

    Pure helper — no I/O. Caller threads the parsed payload dicts.
    """

    if isinstance(candidate_terminal_payload, dict):
        lane = candidate_terminal_payload.get("lane")
        if isinstance(lane, dict):
            value = lane.get("lane_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(work_unit_payload, dict):
        value = work_unit_payload.get("lane_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return LEGACY_LANE_ID


@contextmanager
def _open_readonly(db_path: Path) -> Iterator[sqlite3.Connection | None]:
    """Open ``db_path`` read-only or yield ``None`` on missing / corrupt.

    Mirrors ``shared.runtime_state.read_models._open_readonly`` rather
    than importing it: keeps the layering rule visible at the import
    surface (no transitive pull into the writer). The two helpers should
    track each other; behavior is intentionally identical.
    """

    if not db_path.exists():
        yield None
        return
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        yield conn
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        yield None
    finally:
        if conn is not None:
            conn.close()


def _safe_json_loads(raw: Any) -> dict | None:
    """Parse a SQLite JSON column safely.

    Empty / ``"{}"`` / malformed / non-dict values collapse to ``None``
    so callers can treat them as absent without a try/except dance.
    """

    if not isinstance(raw, str) or not raw or raw == "{}":
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _coerce_cost(metrics: dict | None) -> float | None:
    """Best-effort cost extraction from a work-unit ``metrics_json``.

    Looks for ``cost_usd`` first, then ``total_cost``. Returns ``None``
    if neither is a numeric value; the read model leaves ``cost_usd`` as
    ``None`` at the lane level when no work unit contributed a number.
    """

    if not isinstance(metrics, dict):
        return None
    for key in ("cost_usd", "total_cost"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _normalize_review_reason(payload: dict | None) -> str:
    """Extract the review reason code, or ``UNSPECIFIED_REVIEW_REASON``.

    P4 serializes the code under ``terminal_payload_json["full_decision"]
    ["review_reason_code"]`` via ``OpusDecision.to_dict()``. Empty /
    missing / non-string values fall back so REVIEW outcomes never drop
    out of the breakdown silently.
    """

    if not isinstance(payload, dict):
        return UNSPECIFIED_REVIEW_REASON
    full_decision = payload.get("full_decision")
    if isinstance(full_decision, dict):
        value = full_decision.get("review_reason_code")
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Some downstream consumers may stash the reason at the top level
    # of the terminal payload (older writers / future writers); accept
    # that as a fallback so we don't punt to "unspecified" prematurely.
    value = payload.get("review_reason_code")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return UNSPECIFIED_REVIEW_REASON


@dataclass
class _LaneAccumulator:
    """Mutable per-lane state assembled during aggregation.

    Frozen ``LaneMetricsRow`` is the output type. The accumulator keeps
    the rolling counts / sets / dicts the aggregator needs while it
    walks the run's rows, then materializes into the frozen row.
    """

    lane_id: str
    lane_name: str = ""
    lane_intent: str = ""
    acquisition_mode: str = ""
    result_count: int = 0
    candidates_seen: int = 0
    facial_yes_count: int = 0
    facial_no_count: int = 0
    facial_borderline_count: int = 0
    work_unit_saves_count: int = 0
    work_unit_rejected_count: int = 0
    candidate_save_count: int = 0
    candidate_reject_count: int = 0
    candidate_review_count: int = 0
    review_by_reason: dict[str, int] = field(default_factory=dict)
    work_unit_source_ids: set[str] = field(default_factory=set)
    opened_candidate_ids: set[int] = field(default_factory=set)
    evaluated_candidate_ids: set[int] = field(default_factory=set)
    cost_usd_sum: float = 0.0
    saw_cost: bool = False

    def absorb_label(
        self,
        *,
        lane_name: str,
        lane_intent: str,
        acquisition_mode: str,
    ) -> None:
        # Labels are best-effort: keep the first non-empty value we see
        # so a single mislabeled work unit (e.g. a future legacy migration)
        # doesn't blank out the human-readable lane name.
        if lane_name and not self.lane_name:
            self.lane_name = lane_name
        if lane_intent and not self.lane_intent:
            self.lane_intent = lane_intent
        if acquisition_mode and not self.acquisition_mode:
            self.acquisition_mode = acquisition_mode

    def to_row(self) -> LaneMetricsRow:
        # Candidate-level save/reject totals win over work-unit columns
        # so REVIEW outcomes that pre-P4 work-unit counters never knew
        # about don't end up double-counted. The work-unit columns stay
        # available via the ``review_count`` separation invariant.
        save_count = self.candidate_save_count
        reject_count = self.candidate_reject_count
        return LaneMetricsRow(
            lane_id=self.lane_id,
            lane_name=self.lane_name,
            lane_intent=self.lane_intent,
            acquisition_mode=self.acquisition_mode,
            result_count=self.result_count,
            candidates_seen=self.candidates_seen,
            opened_count=len(self.opened_candidate_ids),
            evaluated_count=len(self.evaluated_candidate_ids),
            facial_yes_count=self.facial_yes_count,
            facial_no_count=self.facial_no_count,
            facial_borderline_count=self.facial_borderline_count,
            save_count=save_count,
            reject_count=reject_count,
            review_count=self.candidate_review_count,
            review_by_reason=dict(self.review_by_reason),
            work_unit_source_ids=tuple(sorted(self.work_unit_source_ids)),
            cost_usd=self.cost_usd_sum if self.saw_cost else None,
            legacy=self.lane_id == LEGACY_LANE_ID,
        )


def _is_abandoned_recovery_attempt(row: Any) -> bool:
    """True for the synthetic full attempt a contained resume skip writes.

    ``Pipeline._abandon_unrecoverable_pending_full`` settles a pending review
    the live Recruiter surface could not re-match by writing a succeeded full
    attempt carrying this marker. Nobody opened the profile and no judge saw
    it, so the row is a skip receipt — counting it as an open or an evaluation
    would report work that never happened.
    """

    try:
        payload = _safe_json_loads(row["payload_json"])
    except (IndexError, KeyError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("pending_full_recovery_abandoned")
    )


def lane_metrics_for_run(
    db_path: Path, *, run_id: int
) -> tuple[LaneMetricsRow, ...]:
    """Return ``LaneMetricsRow`` per lane present in ``run_id``.

    Rows are sorted with the ``legacy`` bucket last and non-legacy
    lanes ordered alphabetically by ``lane_id`` so the wire shape is
    deterministic across calls. Returns an empty tuple for a missing
    or corrupt DB.
    """

    with _open_readonly(db_path) as conn:
        if conn is None:
            return tuple()
        try:
            work_unit_rows = conn.execute(
                "SELECT id, source_unit_id, payload_json, metrics_json, "
                "result_count, candidates_discovered, "
                "facial_yes_count, facial_no_count, facial_borderline_count, "
                "saves_count, rejected_count "
                "FROM work_units WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return tuple()
        try:
            candidate_rows = conn.execute(
                "SELECT DISTINCT c.id, c.terminal_decision, "
                "c.terminal_payload_json, c.last_work_unit_id "
                "FROM candidates c "
                "JOIN candidate_attempts ca ON ca.candidate_id = c.id "
                "WHERE ca.run_id = ?",
                (run_id,),
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            candidate_rows = []
        try:
            attempt_rows = conn.execute(
                "SELECT candidate_id, work_unit_id, stage, status, payload_json "
                "FROM candidate_attempts WHERE run_id = ? AND stage = 'full'",
                (run_id,),
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            attempt_rows = []

    work_unit_payloads: dict[int, dict | None] = {}
    work_unit_lane_for_id: dict[int, str] = {}
    accumulators: dict[str, _LaneAccumulator] = {}

    def _bucket(lane_id: str) -> _LaneAccumulator:
        return accumulators.setdefault(lane_id, _LaneAccumulator(lane_id=lane_id))

    # Work-unit pass: typed counters + lane labels + cost roll-up.
    for row in work_unit_rows:
        payload = _safe_json_loads(row["payload_json"])
        work_unit_payloads[int(row["id"])] = payload
        lane_id = candidate_lane_attribution(None, payload)
        work_unit_lane_for_id[int(row["id"])] = lane_id
        bucket = _bucket(lane_id)
        bucket.work_unit_source_ids.add(str(row["source_unit_id"]))
        if isinstance(payload, dict):
            bucket.absorb_label(
                lane_name=str(payload.get("lane_name") or ""),
                lane_intent=str(payload.get("lane_intent") or ""),
                acquisition_mode=str(payload.get("acquisition_mode") or ""),
            )
        bucket.result_count += int(row["result_count"] or 0)
        bucket.candidates_seen += int(row["candidates_discovered"] or 0)
        bucket.facial_yes_count += int(row["facial_yes_count"] or 0)
        bucket.facial_no_count += int(row["facial_no_count"] or 0)
        bucket.facial_borderline_count += int(row["facial_borderline_count"] or 0)
        bucket.work_unit_saves_count += int(row["saves_count"] or 0)
        bucket.work_unit_rejected_count += int(row["rejected_count"] or 0)
        cost = _coerce_cost(_safe_json_loads(row["metrics_json"]))
        if cost is not None:
            bucket.cost_usd_sum += cost
            bucket.saw_cost = True

    # Candidate pass: attribute via P4 terminal payload first, else the
    # candidate's last work unit. Candidate counts are authoritative for
    # save/reject/review separation.
    for row in candidate_rows:
        candidate_payload = _safe_json_loads(row["terminal_payload_json"])
        work_unit_id = row["last_work_unit_id"]
        work_unit_payload = (
            work_unit_payloads.get(int(work_unit_id))
            if work_unit_id is not None
            else None
        )
        lane_id = candidate_lane_attribution(candidate_payload, work_unit_payload)
        bucket = _bucket(lane_id)
        terminal = row["terminal_decision"]
        if terminal in SAVE_DECISIONS:
            bucket.candidate_save_count += 1
        elif terminal == "REJECT":
            bucket.candidate_reject_count += 1
        elif terminal in NON_SAVE_REVIEW_DECISIONS:
            bucket.candidate_review_count += 1
            reason = _normalize_review_reason(candidate_payload)
            bucket.review_by_reason[reason] = (
                bucket.review_by_reason.get(reason, 0) + 1
            )

    # Attempt pass: opens (any full attempt) and evaluated (succeeded
    # full attempt). Attribute by work unit when available; fall back to
    # the candidate's last work unit otherwise.
    candidate_to_last_wu: dict[int, Any] = {
        int(row["id"]): row["last_work_unit_id"] for row in candidate_rows
    }
    for row in attempt_rows:
        if _is_abandoned_recovery_attempt(row):
            continue
        wu_id = row["work_unit_id"] if row["work_unit_id"] is not None else (
            candidate_to_last_wu.get(int(row["candidate_id"]))
        )
        if wu_id is None:
            lane_id = LEGACY_LANE_ID
        else:
            lane_id = work_unit_lane_for_id.get(int(wu_id), LEGACY_LANE_ID)
        bucket = _bucket(lane_id)
        bucket.opened_candidate_ids.add(int(row["candidate_id"]))
        if row["status"] == "succeeded":
            bucket.evaluated_candidate_ids.add(int(row["candidate_id"]))

    rows = [acc.to_row() for acc in accumulators.values()]
    rows.sort(key=lambda r: (1 if r.legacy else 0, r.lane_id))
    return tuple(rows)


__all__ = [
    "LEGACY_LANE_ID",
    "UNSPECIFIED_REVIEW_REASON",
    "LaneMetricsRow",
    "candidate_lane_attribution",
    "lane_metrics_for_run",
]
