"""Read-only primitives over canonical runtime state.

Each primitive accepts a ``Path`` and, where useful, a pre-opened
``sqlite3.Connection``. It opens SQLite with URI ``mode=ro`` and never imports
the writer store, whose construction performs DDL and metadata writes.
Missing, corrupt, or not-yet-readable databases collapse to ``None`` or an
empty result; primitives that need to distinguish an absent run use their
``NotFound`` variant. Timestamp windows use ``datetime.fromisoformat`` rather
than string ordering.

``has_pending_work`` prefers canonical SQLite and falls back to the
``progress.json`` compatibility projection; neither path constructs the
writer store. The layering boundary is enforced by ``tests/test_read_models.py``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from shared.contracts import SAVE_DECISIONS


_logger = logging.getLogger(__name__)

_LATEST_RUN_QUERY = (
    "SELECT id, source, brief_id, mode, status, stop_reason, started_at, ended_at, "
    "is_archived, intake_session_id "
    "FROM runs ORDER BY id DESC LIMIT 1"
)
# Legacy schemas may lack the brief-taxonomy columns. The fallback keeps the
# read path usable and lets those fields take their dataclass defaults.
_LATEST_RUN_QUERY_LEGACY = (
    "SELECT id, source, brief_id, mode, status, stop_reason, started_at, ended_at "
    "FROM runs ORDER BY id DESC LIMIT 1"
)
_RUN_BY_ID_QUERY = (
    "SELECT id, source, brief_id, output_dir, mode, status, stop_reason, "
    "started_at, ended_at, resumed_from_run_id, "
    "brief_path_at_launch, brief_content_hash, brief_snapshot_json "
    "FROM runs WHERE id = ?"
)
_PROGRESS_JSON_FILENAME = "progress.json"
# Keep this filename local so importing the reader cannot pull in the writer;
# constructing the writer performs DDL and metadata writes.
_RUNTIME_DB_FILENAME = "runtime_state.sqlite3"


# --- types ------------------------------------------------------------------


@dataclass(frozen=True)
class RunSummary:
    """Latest-run snapshot for read-only surfaces.

    Its field set mirrors ``cloris.models.RunSummary`` so the aggregator can
    copy values without interpretation; that Pydantic model is the wire shape
    and this dataclass is the canonical-read shape.
    """

    id: int | None = None
    status: str | None = None
    stop_reason: str | None = None
    mode: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    # Brief-taxonomy fields: is_archived flips when the user
    # files away a brief (or the reconciler auto-archives a stale orphan).
    # intake_session_id is non-null when this run was launched out of the
    # onboarding authoring flow — the aggregator uses it to distinguish
    # "authored brief" from "filesystem state-dir artifact." Both default
    # safely on legacy DBs that don't have the columns yet.
    is_archived: bool = False
    intake_session_id: int | None = None


@dataclass(frozen=True)
class FailureKindCount:
    """One entry of an attempt-health failure-kind histogram."""

    kind: str
    count: int


@dataclass(frozen=True)
class AttemptHealth:
    """Recent attempt outcomes for detecting a stalled run.

    A run is stalled when ``last_success_age_s`` is large and recent failures
    are dominated by retryable HTTP-style failure kinds: it is alive but not
    making progress, typically because the provider is degraded.
    """

    total_attempts_in_window: int = 0
    succeeded_in_window: int = 0
    failed_in_window: int = 0
    last_success_age_s: float | None = None
    recent_failures: tuple[FailureKindCount, ...] = field(default_factory=tuple)
    dominant_failure_kind: str | None = None


@dataclass(frozen=True)
class RunDetail:
    """Full ``runs`` row needed by the run-report surface.

    Superset of :class:`RunSummary` adding identity columns
    (``brief_id``, ``output_dir``, ``brief_path_at_launch``,
    ``brief_content_hash``, ``brief_snapshot_json``,
    ``resumed_from_run_id``). The status aggregator uses ``RunSummary``
    because it only needs the latest-run snapshot; the per-run report
    needs the brief-identity fields too so it can render the role
    title, drift detection, and the resume-chain.
    """

    id: int | None = None
    source: str | None = None
    brief_id: str | None = None
    output_dir: str | None = None
    mode: str | None = None
    status: str | None = None
    stop_reason: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    resumed_from_run_id: int | None = None
    brief_path_at_launch: str | None = None
    brief_content_hash: str | None = None
    brief_snapshot_json: str | None = None


@dataclass(frozen=True)
class CandidateDecision:
    """Per-candidate terminal-decision row for the run report.

    Decisions live on the ``candidates`` row (``terminal_decision`` +
    ``terminal_payload_json``), not on per-run rows — a candidate's
    final outcome is brief-wide, not run-scoped. The run-scoping comes
    from joining through ``candidate_attempts`` to filter to candidates
    that had at least one attempt in this run.
    """

    candidate_id: int
    identity_key: str
    display_name: str
    profile_url: str
    terminal_decision: str | None
    confidence: float | None


@dataclass(frozen=True)
class CandidateNote:
    """One recruiter-authored note on a candidate.

    Notes are append-only — a candidate accumulates notes across review
    sessions and across runs. The schema stores them as a JSON array on
    ``candidates.notes``; this primitive parses the array into a tuple
    of frozen records so consumers don't have to re-parse the JSON.
    """

    body: str
    created_at: str


@dataclass(frozen=True)
class CandidateRecord:
    """Full ``candidates`` row needed by the candidate-detail surface.

    Superset of :class:`CandidateDecision` returning every field a detail
    view needs to render: source/brief identity, terminal payload (parsed
    via :func:`candidate_terminal_payload` for the save reason / confidence
    / any source-specific judgment fields), and timestamps. The terminal
    payload stays as a raw ``str`` here so the Pydantic layer above can
    decide which keys the wire surfaces.

    ``notes`` (parsed from the JSON column) and ``user_status``
    (recruiter-overridden status; ``None`` = use Cloris's judgment).

    ``judgment_accuracy`` (recruiter
    calibration signal — distinct from ``user_status``) and
    ``judgment_accuracy_at`` (timestamp the signal was set). NULLs by
    default so legacy rows pass through without touching either column.
    """

    candidate_id: int
    source: str
    brief_id: str
    identity_key: str
    display_name: str
    profile_url: str
    current_lifecycle_state: str
    terminal_decision: str | None
    terminal_payload_json: str
    first_seen_at: str
    last_seen_at: str
    notes: tuple[CandidateNote, ...] = field(default_factory=tuple)
    user_status: str | None = None
    judgment_accuracy: str | None = None
    judgment_accuracy_at: str | None = None


@dataclass(frozen=True)
class CandidateCardRecord:
    """Per-card row for the Workspace grid.

    Trimmed shape of :class:`CandidateRecord` — the Workspace surface
    only needs the fields that fit on a card. Save reason and confidence
    arrive parsed (the Pydantic layer doesn't have to re-parse the
    terminal_payload_json from the wire). ``last_seen_at`` drives the
    sort order and the "Last touched" stamp on each card. ``user_status``
    is the recruiter override ("shortlist" / "contacted" / etc.; ``None``
    means use Cloris's terminal_decision).
    """

    candidate_id: int
    identity_key: str
    display_name: str
    profile_url: str
    terminal_decision: str
    save_reason: str | None
    confidence: float | None
    last_seen_at: str
    first_seen_at: str
    user_status: str | None = None


@dataclass(frozen=True)
class WorkUnitProgress:
    """Tagged-union return for work-unit status counts.

    Per critique B3: "run doesn't exist" / "run exists but work_units
    empty" / "run has counts" are semantically distinct and callers
    must be able to disambiguate. ``kind`` discriminates; ``counts``
    is populated only when ``kind == "counts"``.
    """

    kind: Literal["not_found", "empty", "counts"]
    queued: int = 0
    in_progress: int = 0
    done: int = 0
    skipped: int = 0
    error: int = 0


# --- shared connection helper -----------------------------------------------


@contextmanager
def _open_readonly(db_path: Path) -> Iterator[sqlite3.Connection | None]:
    """Yield a read-only connection or ``None`` if the file is missing
    or unreadable.

    A freshly-created DB whose ``-wal``/``-shm`` files do not exist yet
    cannot be opened with ``mode=ro``. The
    resulting ``OperationalError`` is treated as "not yet readable" —
    indistinguishable from "missing" at this layer. Callers that need
    the distinction must check ``Path.exists`` themselves before
    invoking the read primitive.
    """

    if not db_path.exists():
        yield None
        return
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        yield conn
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
        # Keep the None-collapse contract, with a debug-only trace for
        # lock, WAL, and corruption races; never write this to the console.
        _logger.debug("read-only open failed for %s: %s", db_path, exc)
        yield None
    finally:
        if conn is not None:
            conn.close()


# --- primitives -------------------------------------------------------------


def latest_run_summary(db_path: Path) -> RunSummary | None:
    """Return the latest ``runs`` row as :class:`RunSummary`, or ``None``.

    ``None`` collapses three cases: file missing, file corrupt/unreadable,
    and ``runs`` table empty. This matches the aggregator's existing
    semantics (one bad state dir must not break the whole response).
    """

    with _open_readonly(db_path) as conn:
        if conn is None:
            return None
        try:
            row = conn.execute(_LATEST_RUN_QUERY).fetchone()
        except sqlite3.OperationalError:
            # Legacy DB without is_archived / intake_session_id columns.
            # Fall back to the pre-v6 query; the new fields collapse to
            # their dataclass defaults so callers continue to work.
            try:
                row = conn.execute(_LATEST_RUN_QUERY_LEGACY).fetchone()
            except sqlite3.DatabaseError:
                return None
        except sqlite3.DatabaseError:
            return None
    if row is None:
        return None
    keys = row.keys()
    return RunSummary(
        id=row["id"] if "id" in keys else None,
        status=row["status"] if "status" in keys else None,
        stop_reason=row["stop_reason"] if "stop_reason" in keys else None,
        mode=row["mode"] if "mode" in keys else None,
        started_at=row["started_at"] if "started_at" in keys else None,
        ended_at=row["ended_at"] if "ended_at" in keys else None,
        is_archived=bool(row["is_archived"]) if "is_archived" in keys else False,
        intake_session_id=(
            row["intake_session_id"] if "intake_session_id" in keys else None
        ),
    )


# --- per-source run-summary read helpers ------------------------------------
#
# The launcher registry exposes one named ``summarize_run_fn`` per source.
# Each helper delegates to :func:`latest_run_summary` against that source's
# canonical ``runtime_state.sqlite3``. Separate names keep the registry
# explicit.
#
# All helpers remain read-only through :func:`_open_readonly`; missing,
# corrupt, or empty databases return ``RunSummary()`` so callers use
# ``summary.id is None`` as the no-run sentinel.


def summarize_linkedin_run(run_dir: Path) -> RunSummary:
    """Read LinkedIn's latest canonical run summary from ``run_dir``."""

    summary = latest_run_summary(run_dir / _RUNTIME_DB_FILENAME)
    return summary if summary is not None else RunSummary()


def summarize_github_run(run_dir: Path) -> RunSummary:
    """Per-source run-summary read for GitHub.

    Mirrors :func:`summarize_linkedin_run`; see that docstring for the
    read-only / empty-fallback contract.
    """

    summary = latest_run_summary(run_dir / _RUNTIME_DB_FILENAME)
    return summary if summary is not None else RunSummary()


def summarize_researcher_run(run_dir: Path) -> RunSummary:
    """Per-source run-summary read for Researcher.

    Mirrors :func:`summarize_linkedin_run`.
    """

    summary = latest_run_summary(run_dir / _RUNTIME_DB_FILENAME)
    return summary if summary is not None else RunSummary()


def summarize_designer_run(run_dir: Path) -> RunSummary:
    """Per-source run-summary read for Designer.

    Mirrors :func:`summarize_linkedin_run`.
    """

    summary = latest_run_summary(run_dir / _RUNTIME_DB_FILENAME)
    return summary if summary is not None else RunSummary()


def summarize_exec_search_run(run_dir: Path) -> RunSummary:
    """Per-source run-summary read for Executive Search.

    Mirrors :func:`summarize_linkedin_run`.
    """

    summary = latest_run_summary(run_dir / _RUNTIME_DB_FILENAME)
    return summary if summary is not None else RunSummary()


def summarize_exa_run(run_dir: Path) -> RunSummary:
    """Read Exa's latest session summary from its canonical campaign DB."""

    summary = latest_run_summary(run_dir / _RUNTIME_DB_FILENAME)
    return summary if summary is not None else RunSummary()


def run_by_id(db_path: Path, *, run_id: int) -> RunDetail | None:
    """Return the ``runs`` row identified by ``run_id`` as :class:`RunDetail`.

    ``None`` collapses three cases: file missing, file corrupt/unreadable,
    and the run id is not present in the ``runs`` table. Brief-identity
    columns may be ``None`` for older schema rows; the read
    primitive falls back to a stripped query in that case so a legacy
    DB still yields a usable :class:`RunDetail`.
    """

    with _open_readonly(db_path) as conn:
        if conn is None:
            return None
        try:
            row = conn.execute(_RUN_BY_ID_QUERY, (run_id,)).fetchone()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            # Legacy schemas lack brief-identity columns; project the available
            # row onto a RunDetail with those fields left at their defaults.
            try:
                row = conn.execute(
                    "SELECT id, source, brief_id, output_dir, mode, status, "
                    "stop_reason, started_at, ended_at, resumed_from_run_id, "
                    "NULL AS brief_path_at_launch, NULL AS brief_content_hash, "
                    "NULL AS brief_snapshot_json FROM runs WHERE id = ?",
                    (run_id,),
                ).fetchone()
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                return None
    if row is None:
        return None
    return RunDetail(
        id=row["id"],
        source=row["source"],
        brief_id=row["brief_id"],
        output_dir=row["output_dir"] if "output_dir" in row.keys() else None,
        mode=row["mode"],
        status=row["status"],
        stop_reason=row["stop_reason"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        resumed_from_run_id=row["resumed_from_run_id"]
        if "resumed_from_run_id" in row.keys()
        else None,
        brief_path_at_launch=row["brief_path_at_launch"]
        if "brief_path_at_launch" in row.keys()
        else None,
        brief_content_hash=row["brief_content_hash"]
        if "brief_content_hash" in row.keys()
        else None,
        brief_snapshot_json=row["brief_snapshot_json"]
        if "brief_snapshot_json" in row.keys()
        else None,
    )


def run_decisions(
    db_path: Path,
    *,
    run_id: int,
    limit: int = 200,
) -> tuple[CandidateDecision, ...]:
    """Return up to ``limit`` candidates that had attempts in ``run_id``.

    Joins ``candidate_attempts`` to ``candidates`` so the run-report
    surface lists who Cloris touched in this run. Each candidate's
    ``terminal_decision`` is the brief-wide outcome (a candidate's
    final decision can be set in a later run for the same brief), but
    that's the right value to surface — recruiters care about the
    final state, not the intermediate one.

    Confidence parses safely from ``terminal_payload_json``; malformed
    JSON or missing ``confidence`` keys collapse to ``None``.

    Empty tuple when DB is missing/corrupt or no attempts exist for
    this run.
    """

    with _open_readonly(db_path) as conn:
        if conn is None:
            return tuple()
        try:
            rows = conn.execute(
                "SELECT DISTINCT c.id, c.identity_key, c.display_name, "
                "c.profile_url, c.terminal_decision, c.terminal_payload_json, "
                "c.last_seen_at "
                "FROM candidates c "
                "JOIN candidate_attempts ca ON ca.candidate_id = c.id "
                "WHERE ca.run_id = ? "
                "ORDER BY c.last_seen_at DESC, c.id DESC "
                "LIMIT ?",
                (run_id, limit),
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return tuple()

    decisions: list[CandidateDecision] = []
    for row in rows:
        confidence: float | None = None
        payload_raw = row["terminal_payload_json"]
        if isinstance(payload_raw, str) and payload_raw and payload_raw != "{}":
            try:
                payload = json.loads(payload_raw)
                value = payload.get("confidence")
                if isinstance(value, (int, float)):
                    confidence = float(value)
            except (json.JSONDecodeError, TypeError, AttributeError):
                confidence = None
        decisions.append(
            CandidateDecision(
                candidate_id=row["id"],
                identity_key=row["identity_key"] or "",
                display_name=row["display_name"] or "",
                profile_url=row["profile_url"] or "",
                terminal_decision=row["terminal_decision"],
                confidence=confidence,
            )
        )
    return tuple(decisions)


def candidate_by_id(
    db_path: Path, *, candidate_id: int
) -> CandidateRecord | None:
    """Return a single candidate row by its primary-key id.

    The candidate-detail primitive. ``None``
    collapses three cases: file missing, file corrupt/unreadable, and
    the candidate id is not present in the ``candidates`` table.
    Source-agnostic — the row carries its own ``source`` and ``brief_id``.
    """

    with _open_readonly(db_path) as conn:
        if conn is None:
            return None
        # Try the richest query first, then older schemas. Each fallback aliases
        # absent fields to NULL for a uniform row shape for callers.
        try:
            row = conn.execute(
                "SELECT id, source, brief_id, identity_key, display_name, "
                "profile_url, current_lifecycle_state, terminal_decision, "
                "terminal_payload_json, first_seen_at, last_seen_at, "
                "notes, user_status, judgment_accuracy, judgment_accuracy_at "
                "FROM candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            try:
                row = conn.execute(
                    "SELECT id, source, brief_id, identity_key, display_name, "
                    "profile_url, current_lifecycle_state, terminal_decision, "
                    "terminal_payload_json, first_seen_at, last_seen_at, "
                    "notes, user_status, NULL AS judgment_accuracy, "
                    "NULL AS judgment_accuracy_at "
                    "FROM candidates WHERE id = ?",
                    (candidate_id,),
                ).fetchone()
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                try:
                    row = conn.execute(
                        "SELECT id, source, brief_id, identity_key, display_name, "
                        "profile_url, current_lifecycle_state, terminal_decision, "
                        "terminal_payload_json, first_seen_at, last_seen_at, "
                        "'[]' AS notes, NULL AS user_status, "
                        "NULL AS judgment_accuracy, NULL AS judgment_accuracy_at "
                        "FROM candidates WHERE id = ?",
                        (candidate_id,),
                    ).fetchone()
                except (sqlite3.OperationalError, sqlite3.DatabaseError):
                    return None
    if row is None:
        return None

    notes_raw = row["notes"] if "notes" in row.keys() else "[]"
    notes_parsed: list[CandidateNote] = []
    if isinstance(notes_raw, str) and notes_raw and notes_raw != "[]":
        try:
            entries = json.loads(notes_raw)
        except (json.JSONDecodeError, TypeError):
            entries = []
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                body = entry.get("body")
                created_at = entry.get("created_at")
                if isinstance(body, str) and isinstance(created_at, str):
                    notes_parsed.append(
                        CandidateNote(body=body, created_at=created_at)
                    )

    user_status_raw = row["user_status"] if "user_status" in row.keys() else None
    judgment_accuracy_raw = (
        row["judgment_accuracy"] if "judgment_accuracy" in row.keys() else None
    )
    judgment_accuracy_at_raw = (
        row["judgment_accuracy_at"]
        if "judgment_accuracy_at" in row.keys()
        else None
    )

    return CandidateRecord(
        candidate_id=row["id"],
        source=row["source"] or "",
        brief_id=row["brief_id"] or "",
        identity_key=row["identity_key"] or "",
        display_name=row["display_name"] or "",
        profile_url=row["profile_url"] or "",
        current_lifecycle_state=row["current_lifecycle_state"] or "",
        terminal_decision=row["terminal_decision"],
        terminal_payload_json=row["terminal_payload_json"] or "{}",
        first_seen_at=row["first_seen_at"] or "",
        last_seen_at=row["last_seen_at"] or "",
        notes=tuple(notes_parsed),
        user_status=user_status_raw if isinstance(user_status_raw, str) and user_status_raw else None,
        judgment_accuracy=(
            judgment_accuracy_raw
            if isinstance(judgment_accuracy_raw, str) and judgment_accuracy_raw
            else None
        ),
        judgment_accuracy_at=(
            judgment_accuracy_at_raw
            if isinstance(judgment_accuracy_at_raw, str) and judgment_accuracy_at_raw
            else None
        ),
    )


def candidate_recent_run_id(
    db_path: Path, *, candidate_id: int
) -> int | None:
    """Return the most-recent ``run_id`` that had an attempt on this candidate.

    The candidate-detail page renders a back-link to "the run where Cloris
    found this person." A candidate may have attempts across multiple runs
    (judgement re-runs, retries); the highest run_id wins. ``None`` when
    there are no attempts (the candidate exists but has never been touched
    by any run — uncommon but possible for legacy/imported rows).
    """

    with _open_readonly(db_path) as conn:
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT MAX(run_id) AS run_id FROM candidate_attempts "
                "WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return None
    if row is None or row["run_id"] is None:
        return None
    return int(row["run_id"])


def brief_saves(
    db_path: Path,
    *,
    source: str,
    brief_id: str,
    limit: int = 200,
) -> tuple[CandidateCardRecord, ...]:
    """Return all SAVE-class candidates for ``(source, brief_id)``.

    The Workspace surface aggregates saves *across all runs* of a brief,
    not just the latest run, so the join goes against ``candidates``
    directly rather than through ``candidate_attempts``. The terminal
    decision must be in the ``SAVE`` family (the orchestrator writes
    SAVE / INFERENTIAL_SAVE / TRANSFERABLE_SAVE / SIGNAL_SAVE for
    different judgment shapes — all are recruiter-actionable).

    Save reason and confidence are parsed safely from
    ``terminal_payload_json``; malformed JSON or missing keys collapse
    to ``None`` per :func:`candidate_terminal_payload`.

    Empty tuple when DB is missing / corrupt or no saves exist for
    this brief.
    """

    save_decisions = (
        "SAVE",
        "INFERENTIAL_SAVE",
        "TRANSFERABLE_SAVE",
        "SIGNAL_SAVE",
    )
    placeholders = ",".join("?" for _ in save_decisions)

    with _open_readonly(db_path) as conn:
        if conn is None:
            return tuple()
        # Fall back to the legacy projection when `user_status` is
        # unavailable; the value collapses to NULL for those rows.
        try:
            rows = conn.execute(
                f"SELECT id, identity_key, display_name, profile_url, "
                f"terminal_decision, terminal_payload_json, "
                f"first_seen_at, last_seen_at, user_status "
                f"FROM candidates "
                f"WHERE source = ? AND brief_id = ? "
                f"  AND terminal_decision IN ({placeholders}) "
                f"  AND (current_lifecycle_state IS NULL OR current_lifecycle_state NOT LIKE 'failed_%') "
                f"ORDER BY last_seen_at DESC, id DESC "
                f"LIMIT ?",
                (source, brief_id, *save_decisions, limit),
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            try:
                rows = conn.execute(
                    f"SELECT id, identity_key, display_name, profile_url, "
                    f"terminal_decision, terminal_payload_json, "
                    f"first_seen_at, last_seen_at, NULL AS user_status "
                    f"FROM candidates "
                    f"WHERE source = ? AND brief_id = ? "
                    f"  AND terminal_decision IN ({placeholders}) "
                    f"ORDER BY last_seen_at DESC, id DESC "
                    f"LIMIT ?",
                    (source, brief_id, *save_decisions, limit),
                ).fetchall()
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                return tuple()

    cards: list[CandidateCardRecord] = []
    for row in rows:
        payload = candidate_terminal_payload(row["terminal_payload_json"] or "{}")
        save_reason, confidence = extract_save_reason_and_confidence(payload)
        user_status_raw = row["user_status"] if "user_status" in row.keys() else None
        cards.append(
            CandidateCardRecord(
                candidate_id=row["id"],
                identity_key=row["identity_key"] or "",
                display_name=row["display_name"] or "",
                profile_url=row["profile_url"] or "",
                terminal_decision=row["terminal_decision"] or "SAVE",
                save_reason=save_reason,
                confidence=confidence,
                last_seen_at=row["last_seen_at"] or "",
                first_seen_at=row["first_seen_at"] or "",
                user_status=user_status_raw if isinstance(user_status_raw, str) and user_status_raw else None,
            )
        )
    return tuple(cards)


def latest_run_in_state_dir(db_path: Path) -> int | None:
    """Return the id of the most-recent run in this state dir's DB.

    Used by the Workspace aggregator to find the latest_run_id for the
    "View latest run report" link, plus to pick a brief_id when the URL
    only carries (source, state_key). ``None`` if the DB is missing /
    corrupt / has no runs.
    """

    with _open_readonly(db_path) as conn:
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT id FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return None
    if row is None:
        return None
    return int(row["id"])


def candidate_terminal_payload(
    terminal_payload_json: str,
) -> dict | None:
    """Parse ``candidates.terminal_payload_json`` into a dict, safely.

    Returns ``None`` for empty / ``"{}"`` / malformed inputs so callers
    can ``if payload is None: skip`` rather than threading try/except.
    Save reason, confidence, and any source-specific judgment fields
    (e.g. linkedin's ``judgment_reason``) live in this blob; the wire
    surface decides which keys to publish.
    """

    if not terminal_payload_json or terminal_payload_json == "{}":
        return None
    try:
        parsed = json.loads(terminal_payload_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def extract_save_reason_and_confidence(
    payload: dict | None,
) -> tuple[str | None, float | None]:
    """Extract recruiter-facing save reason + confidence from a parsed
    ``terminal_payload``, with provenance-aware priority.

    The full-stage judgment is stored under
    ``payload["full_decision"]["rationale"]`` and ``...["confidence"]``.
    Top-level ``save_reason`` / ``reason`` and ``confidence`` remain fallback
    paths for older or test-built payloads.

    Priority on read:

    - reason text: ``full_decision.rationale`` →
      ``save_reason`` (top-level) → ``reason`` (top-level legacy).
    - confidence: ``full_decision.confidence`` →
      ``confidence`` (top-level legacy).

    Returns ``(None, None)`` for ``payload is None`` or for any
    missing / malformed values. Callers that want a ``"No save
    reason recorded"`` fallback at the surface layer should treat
    ``None`` as the trigger — the substrate now passes through
    truthful judgment whenever it exists.
    """

    if payload is None:
        return (None, None)

    save_reason: str | None = None
    confidence: float | None = None

    full_decision = payload.get("full_decision")
    if isinstance(full_decision, dict):
        candidate_text = full_decision.get("rationale")
        if isinstance(candidate_text, str) and candidate_text.strip():
            save_reason = candidate_text.strip()
        raw_conf = full_decision.get("confidence")
        if isinstance(raw_conf, (int, float)):
            confidence = float(raw_conf)

    if save_reason is None:
        for key in ("save_reason", "reason"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                save_reason = value.strip()
                break

    if confidence is None:
        raw_conf = payload.get("confidence")
        if isinstance(raw_conf, (int, float)):
            confidence = float(raw_conf)

    return (save_reason, confidence)


def extract_surface_type(payload: dict | None) -> str | None:
    """Return a saved candidate's recruiter-facing ``surface_type``.

    The field is top-level in ``terminal_payload_json``. Missing or empty
    values return ``None``; non-empty values pass through for surface dispatch.
    """

    if payload is None:
        return None
    surface_type = payload.get("surface_type")
    if isinstance(surface_type, str) and surface_type.strip():
        return surface_type.strip()
    return None


def extract_visual_judgment(payload: dict | None) -> dict | None:
    """Return the structured ``visual_judgment`` payload, or ``None``.

    The payload lives at
    ``payload["visual_judgment"]`` and carries:

    - ``model``: which vision LLM produced the eval (e.g.,
      ``"gemini-2.5-pro"``).
    - ``principles``: list of per-principle scores + reasoning +
      ``image_ids`` cited from the input image set.
    - ``overall_verdict``: ``yes | no | borderline``.
    - ``overall_confidence``: 0.0-1.0.
    - ``cross_check``: optional secondary review data present only for
      top-decile candidates that receive the secondary review pass.
    - ``assets``: list of {id, url, source, project_title} so the
      frontend can render the thumbnail-grid alongside the per-
      principle reasoning.

    Returns ``None`` when ``payload`` is ``None``, when the field is missing,
    or when it is not a dict. This pure extraction has no side effects.
    """

    if payload is None:
        return None
    visual_judgment = payload.get("visual_judgment")
    if isinstance(visual_judgment, dict) and visual_judgment:
        return visual_judgment
    return None


def extract_recommendation_pitch(payload: dict | None) -> dict | None:
    """Return the ``recommendation_pitch`` payload, or None.

    D5b. Shape: {headline, summary, evidence_bullets, caveats}.
    Present on non-REJECT Designer terminal payloads. Pure extraction.
    """

    if payload is None:
        return None
    pitch = payload.get("recommendation_pitch")
    if isinstance(pitch, dict) and pitch:
        return pitch
    return None


def has_pending_work(state_dir: Path) -> bool | None:
    """Return whether the latest LinkedIn run has nonterminal work.

    Canonical SQLite wins when it can answer. ``progress.json`` is only a
    compatibility fallback. ``None`` means neither source was readable.
    """

    db_path = state_dir / _RUNTIME_DB_FILENAME
    with _open_readonly(db_path) as conn:
        if conn is not None:
            try:
                run = conn.execute(
                    "SELECT id, resume_state_json FROM runs "
                    "WHERE source = 'linkedin' ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if run is not None:
                    pending_string = conn.execute(
                        "SELECT 1 FROM work_units "
                        "WHERE run_id = ? AND source = 'linkedin' "
                        "AND kind = 'linkedin_string' "
                        "AND status NOT IN ('done', 'skipped') LIMIT 1",
                        (run["id"],),
                    ).fetchone()
                    if pending_string is not None:
                        return True
                    unresolved_full_review = None
                    try:
                        run_brief = conn.execute(
                            "SELECT brief_id FROM runs WHERE id = ?",
                            (run["id"],),
                        ).fetchone()
                        save_placeholders = ", ".join(
                            "?" for _ in SAVE_DECISIONS
                        )
                        unresolved_full_review = conn.execute(
                            f"""
                        WITH RECURSIVE run_chain(id, resumed_from_run_id) AS (
                            SELECT id, resumed_from_run_id
                            FROM runs
                            WHERE id = ? AND source = 'linkedin'
                              AND brief_id = ?
                            UNION
                            SELECT parent.id, parent.resumed_from_run_id
                            FROM runs parent
                            JOIN run_chain child
                              ON parent.id = child.resumed_from_run_id
                            WHERE parent.source = 'linkedin'
                              AND parent.brief_id = ?
                        ),
                        stage_attempts AS (
                            SELECT
                                ca.candidate_id,
                                ca.stage,
                                json_extract(
                                    ca.payload_json,
                                    '$.' || ca.stage || '_decision.decision'
                                ) AS decision,
                                -- Sequence 1 is the authoritative row per
                                -- stage. For the FULL stage a failure-family
                                -- decision sorts last so it can never shadow a
                                -- real verdict: a contained resume skip settles
                                -- the candidate with a synthetic succeeded
                                -- JUDGMENT_FAILURE, and if the person is later
                                -- met and really evaluated, that verdict owns
                                -- the answer. Scoped to 'full' by the CASE, so
                                -- facial ordering is byte-identical. Mirrors
                                -- the Python twin in
                                -- Pipeline._hydrate_resume_funnel_from_runtime.
                                ROW_NUMBER() OVER (
                                    PARTITION BY ca.candidate_id, ca.stage
                                    ORDER BY
                                        CASE
                                            WHEN ca.stage = 'full'
                                             AND json_extract(
                                                     ca.payload_json,
                                                     '$.full_decision.decision'
                                                 ) IN (
                                                     'PARSE_FAILURE',
                                                     'JUDGMENT_FAILURE'
                                                 )
                                            THEN 1
                                            ELSE 0
                                        END,
                                        ca.id
                                ) AS sequence
                            FROM candidate_attempts ca
                            JOIN run_chain chain ON chain.id = ca.run_id
                            JOIN candidates candidate
                              ON candidate.id = ca.candidate_id
                            WHERE ca.status = 'succeeded'
                              AND ca.stage IN ('facial', 'full')
                              AND candidate.source = 'linkedin'
                              AND candidate.brief_id = ?
                              AND COALESCE(
                                  json_extract(
                                      ca.payload_json,
                                      '$.' || ca.stage || '_decision.decision'
                                  ),
                                  ''
                              ) <> ''
                        )
                        SELECT 1
                        FROM stage_attempts facial
                        LEFT JOIN stage_attempts full_attempt
                          ON full_attempt.candidate_id = facial.candidate_id
                         AND full_attempt.stage = 'full'
                         AND full_attempt.sequence = 1
                        WHERE facial.stage = 'facial'
                          AND facial.sequence = 1
                          AND facial.decision IN (
                              'FACIAL_YES', 'FACIAL_BORDERLINE'
                          )
                          AND (
                              full_attempt.candidate_id IS NULL
                              OR (
                                  full_attempt.decision IN ({save_placeholders})
                                  AND NOT EXISTS (
                                      SELECT 1
                                      FROM side_effects effect
                                      JOIN run_chain effect_chain
                                        ON effect_chain.id = effect.run_id
                                      WHERE effect.candidate_id =
                                            facial.candidate_id
                                        AND effect.effect_type = 'linkedin_save'
                                        AND effect.status = 'succeeded'
                                  )
                              )
                          )
                        LIMIT 1
                        """,
                            (
                                run["id"],
                                run_brief["brief_id"],
                                run_brief["brief_id"],
                                run_brief["brief_id"],
                                *sorted(SAVE_DECISIONS),
                            ),
                        ).fetchone()
                    except sqlite3.DatabaseError:
                        # Legacy read-only schemas can still answer from their
                        # canonical work-unit and resume-state columns.
                        pass
                    if unresolved_full_review is not None:
                        return True
                    resume_state = json.loads(run["resume_state_json"])
                    if isinstance(resume_state, dict):
                        return bool(
                            resume_state.get("pending_block_string_ids")
                        )
            except (
                sqlite3.DatabaseError,
                json.JSONDecodeError,
                TypeError,
                UnicodeDecodeError,
            ):
                pass

    progress_path = state_dir / _PROGRESS_JSON_FILENAME
    if not progress_path.exists():
        return None
    try:
        progress = json.loads(progress_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(progress, dict):
        return None
    if progress.get("pending_block_string_ids"):
        return True
    strings = progress.get("strings", [])
    if not isinstance(strings, list):
        return None
    if not strings:
        return False
    if not all(isinstance(item, dict) for item in strings):
        return None
    return any(
        item.get("status") not in {"done", "skipped"}
        for item in strings
    )


def attempt_health(
    db_path: Path,
    *,
    run_id: int,
    window_minutes: int = 5,
) -> AttemptHealth:
    """Summarize recent ``candidate_attempts`` outcomes for ``run_id``.

    The window is "rows whose started_at is within the last
    ``window_minutes`` minutes." String-comparison on ISO-8601 would
    work for the canonical fixed-width format the writer emits, but
    we parse via ``datetime.fromisoformat`` to be robust to format
    drift — per critique B2.

    Returns an empty :class:`AttemptHealth` rather than ``None`` when
    the DB is missing/corrupt or the run has no attempts; that keeps
    the consuming aggregator simple (no Optional unwrapping at the
    call site).
    """

    empty = AttemptHealth()
    with _open_readonly(db_path) as conn:
        if conn is None:
            return empty
        try:
            rows = conn.execute(
                "SELECT status, failure_kind, started_at, ended_at "
                "FROM candidate_attempts WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        except sqlite3.DatabaseError:
            return empty

    cutoff = datetime.now(timezone.utc).timestamp() - window_minutes * 60.0
    in_window: list[sqlite3.Row] = []
    last_success_at: float | None = None

    for row in rows:
        started = _parse_iso_to_epoch(row["started_at"])
        if started is None:
            continue
        if row["status"] == "succeeded":
            ended = _parse_iso_to_epoch(row["ended_at"]) or started
            if last_success_at is None or ended > last_success_at:
                last_success_at = ended
        if started >= cutoff:
            in_window.append(row)

    failure_kinds: dict[str, int] = {}
    succeeded = 0
    failed = 0
    for row in in_window:
        if row["status"] == "succeeded":
            succeeded += 1
        else:
            failed += 1
            kind = row["failure_kind"] or "unknown"
            failure_kinds[kind] = failure_kinds.get(kind, 0) + 1

    if last_success_at is None:
        last_success_age_s: float | None = None
    else:
        last_success_age_s = max(
            0.0, datetime.now(timezone.utc).timestamp() - last_success_at
        )

    histogram = tuple(
        FailureKindCount(kind=kind, count=count)
        for kind, count in sorted(
            failure_kinds.items(), key=lambda kv: (-kv[1], kv[0])
        )
    )
    dominant = histogram[0].kind if histogram else None

    return AttemptHealth(
        total_attempts_in_window=len(in_window),
        succeeded_in_window=succeeded,
        failed_in_window=failed,
        last_success_age_s=last_success_age_s,
        recent_failures=histogram,
        dominant_failure_kind=dominant,
    )


def work_unit_progress(
    db_path: Path,
    *,
    run_id: int,
    kind: str,
) -> WorkUnitProgress:
    """Return queued/in_progress/done/skipped/error counts for ``run_id``.

    Three-state result discriminated by ``kind``:

    - ``"not_found"``: no run exists with this id (or DB is missing/corrupt).
    - ``"empty"``: run exists but has no work_units of the given kind.
    - ``"counts"``: at least one row; counts are populated.
    """

    with _open_readonly(db_path) as conn:
        if conn is None:
            return WorkUnitProgress(kind="not_found")
        try:
            run_row = conn.execute(
                "SELECT id FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                return WorkUnitProgress(kind="not_found")
            rows = conn.execute(
                "SELECT status, COUNT(*) as n FROM work_units "
                "WHERE run_id = ? AND kind = ? GROUP BY status",
                (run_id, kind),
            ).fetchall()
        except sqlite3.DatabaseError:
            return WorkUnitProgress(kind="not_found")

    if not rows:
        return WorkUnitProgress(kind="empty")

    counts = {"queued": 0, "in_progress": 0, "done": 0, "skipped": 0, "error": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + int(row["n"])
    return WorkUnitProgress(
        kind="counts",
        queued=counts["queued"],
        in_progress=counts["in_progress"],
        done=counts["done"],
        skipped=counts["skipped"],
        error=counts["error"],
    )


def work_unit_progress_multi(
    db_path: Path,
    *,
    run_id: int,
    kinds: tuple[str, ...],
) -> WorkUnitProgress:
    """Like :func:`work_unit_progress` but counts across multiple kinds.

    Designer uses both ``designer_behance_query`` and
    ``designer_cse_query`` work-unit kinds. This variant aggregates across
    all provided kinds so CSE-only, Behance-only, and mixed runs show
    accurate progress.
    """

    if not kinds:
        return WorkUnitProgress(kind="empty")

    placeholders = ", ".join("?" for _ in kinds)
    with _open_readonly(db_path) as conn:
        if conn is None:
            return WorkUnitProgress(kind="not_found")
        try:
            run_row = conn.execute(
                "SELECT id FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                return WorkUnitProgress(kind="not_found")
            rows = conn.execute(
                "SELECT status, COUNT(*) as n FROM work_units "
                f"WHERE run_id = ? AND kind IN ({placeholders}) GROUP BY status",
                (run_id, *kinds),
            ).fetchall()
        except sqlite3.DatabaseError:
            return WorkUnitProgress(kind="not_found")

    if not rows:
        return WorkUnitProgress(kind="empty")

    counts = {"queued": 0, "in_progress": 0, "done": 0, "skipped": 0, "error": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + int(row["n"])
    return WorkUnitProgress(
        kind="counts",
        queued=counts["queued"],
        in_progress=counts["in_progress"],
        done=counts["done"],
        skipped=counts["skipped"],
        error=counts["error"],
    )


# --- orchestration SQLite read helpers --------------------------------------
#
# These primitives read the orchestration database through the same
# ``mode=ro`` boundary as per-source state. Missing or corrupt files collapse
# to ``None`` or an empty tuple; no writer-store DDL or metadata writes occur.


@dataclass(frozen=True)
class ChiefOfStaffRunRecord:
    """One ``chief_of_staff_runs`` row.

    Brief-grain cross-source (one orchestration run merges per-source runs
    into one synthesis). Payload columns stay as raw JSON strings so
    the Pydantic / API layer above this primitive decides which keys
    to surface — this matches the convention set by
    :class:`CandidateRecord` (raw ``terminal_payload_json``) and keeps the
    reader independent of writer-side payload evolution.
    """

    id: int
    brief_id: str
    principal_id: str
    status: str
    dispatch_plan_json: str
    invocation_order_json: str
    handoff_payloads_json: str
    synthesis_output_json: str
    started_at: str
    ended_at: str | None


@dataclass(frozen=True)
class CrossBriefObservation:
    """One ``cross_brief_playbook_observations`` row.

    Per-principal × per-market × per-role-shape grain. Consumers group by
    (``principal_id``, ``market_key``, ``role_shape``) and read
    ``observation_json`` for the per-row payload.
    """

    id: int
    principal_id: str
    market_key: str
    role_shape: str
    brief_id: str
    observation_json: str
    created_at: str


def chief_of_staff_run_by_brief(
    db_path: Path, *, brief_id: str
) -> ChiefOfStaffRunRecord | None:
    """Return the latest ``chief_of_staff_runs`` row for ``brief_id``.

    ``None`` collapses three cases (matching the per-source primitives'
    convention): file missing, file corrupt/unreadable, no row in
    ``chief_of_staff_runs`` for ``brief_id``. Latest-wins ordering is
    by ``started_at DESC, id DESC``. The writer inserts one row per CoS run
    rather than mutating in place, so this yields the most-recent attempt.
    """

    with _open_readonly(db_path) as conn:
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT id, brief_id, principal_id, status, "
                "dispatch_plan_json, invocation_order_json, "
                "handoff_payloads_json, synthesis_output_json, "
                "started_at, ended_at "
                "FROM chief_of_staff_runs "
                "WHERE brief_id = ? "
                "ORDER BY started_at DESC, id DESC "
                "LIMIT 1",
                (brief_id,),
            ).fetchone()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return None
    if row is None:
        return None
    return ChiefOfStaffRunRecord(
        id=int(row["id"]),
        brief_id=row["brief_id"] or "",
        principal_id=row["principal_id"] or "",
        status=row["status"] or "",
        dispatch_plan_json=row["dispatch_plan_json"] or "{}",
        invocation_order_json=row["invocation_order_json"] or "[]",
        handoff_payloads_json=row["handoff_payloads_json"] or "{}",
        synthesis_output_json=row["synthesis_output_json"] or "{}",
        started_at=row["started_at"] or "",
        ended_at=row["ended_at"],
    )


def chief_of_staff_runs_for_brief(
    db_path: Path, *, brief_id: str
) -> list[ChiefOfStaffRunRecord]:
    """Return every ``chief_of_staff_runs`` row for ``brief_id``.

    Ordered by ``started_at DESC, id DESC`` (most recent first). Empty
    list if the DB is missing/unreadable or there are no rows — unlike
    :func:`chief_of_staff_run_by_brief`, absent rows yield ``[]`` not
    ``None`` so list endpoints can return an honest empty payload.
    """

    with _open_readonly(db_path) as conn:
        if conn is None:
            return []
        try:
            rows = conn.execute(
                "SELECT id, brief_id, principal_id, status, "
                "dispatch_plan_json, invocation_order_json, "
                "handoff_payloads_json, synthesis_output_json, "
                "started_at, ended_at "
                "FROM chief_of_staff_runs "
                "WHERE brief_id = ? "
                "ORDER BY started_at DESC, id DESC",
                (brief_id,),
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return []
    return [
        ChiefOfStaffRunRecord(
            id=int(row["id"]),
            brief_id=row["brief_id"] or "",
            principal_id=row["principal_id"] or "",
            status=row["status"] or "",
            dispatch_plan_json=row["dispatch_plan_json"] or "{}",
            invocation_order_json=row["invocation_order_json"] or "[]",
            handoff_payloads_json=row["handoff_payloads_json"] or "{}",
            synthesis_output_json=row["synthesis_output_json"] or "{}",
            started_at=row["started_at"] or "",
            ended_at=row["ended_at"],
        )
        for row in rows
    ]


def cross_brief_observations_for_principal(
    db_path: Path,
    *,
    principal_id: str,
    market_key: str | None = None,
    role_shape: str | None = None,
    limit: int = 200,
) -> tuple[CrossBriefObservation, ...]:
    """Return cross-brief observations for ``principal_id``.

    Optional ``market_key`` / ``role_shape`` filters narrow the query
    to the calibration grain the aggregator wants ("for principal X,
    frontier-AI briefs in NYC"). Sorted by ``created_at DESC, id DESC``
    so the most recent observations come first; ``limit`` caps the
    return so a long-lived principal doesn't blow up the read.

    Empty tuple collapses three cases: file missing, file corrupt, or
    no matching rows.
    """

    where_clauses = ["principal_id = ?"]
    params: list[object] = [principal_id]
    if market_key is not None:
        where_clauses.append("market_key = ?")
        params.append(market_key)
    if role_shape is not None:
        where_clauses.append("role_shape = ?")
        params.append(role_shape)
    params.append(int(limit))
    sql = (
        "SELECT id, principal_id, market_key, role_shape, brief_id, "
        "observation_json, created_at "
        "FROM cross_brief_playbook_observations "
        f"WHERE {' AND '.join(where_clauses)} "
        "ORDER BY created_at DESC, id DESC "
        "LIMIT ?"
    )
    with _open_readonly(db_path) as conn:
        if conn is None:
            return tuple()
        try:
            rows = conn.execute(sql, params).fetchall()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return tuple()
    return tuple(
        CrossBriefObservation(
            id=int(row["id"]),
            principal_id=row["principal_id"] or "",
            market_key=row["market_key"] or "",
            role_shape=row["role_shape"] or "",
            brief_id=row["brief_id"] or "",
            observation_json=row["observation_json"] or "{}",
            created_at=row["created_at"] or "",
        )
        for row in rows
    )


# --- intake_sessions read primitives ----------------------------------------
#
# Parallel to ``cloris.intake_sessions.list_intake_sessions`` /
# ``get_intake_session`` but opened via ``_open_readonly`` so the
# polled GET endpoints in ``cloris.api`` can answer without
# instantiating ``RuntimeStateStore`` (whose ``__init__`` runs DDL
# plus ``INSERT OR REPLACE INTO meta`` on every call). Returned dict
# shape matches ``cloris.intake_sessions._row_to_session`` so the
# wire model (``cloris.models.IntakeSession``) consumes either path
# without branching.


def _row_to_intake_session(row: sqlite3.Row) -> dict:
    """Mirror of ``cloris.intake_sessions._row_to_session`` for the read path.

    Kept private so the read-models layering rule (no imports back into
    ``cloris.*`` or ``shared.runtime_state.store``) holds; see
    ``tests/test_read_models.py``.
    """

    raw_state = row["state_json"]
    if raw_state:
        try:
            parsed_state = json.loads(raw_state)
        except (json.JSONDecodeError, TypeError):
            parsed_state = {}
    else:
        parsed_state = {}
    if not isinstance(parsed_state, dict):
        parsed_state = {}
    return {
        "id": row["id"],
        "brief_id_draft": row["brief_id_draft"],
        "role_title": row["role_title"],
        "current_step": row["current_step"],
        "state_json": parsed_state,
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
        "archived_at": row["archived_at"],
    }


def list_intake_sessions(db_path: str | Path) -> list[dict]:
    """Read-only list of active (non-archived) intake_sessions rows.

    Parallel to :func:`cloris.intake_sessions.list_intake_sessions` but
    opens the SQLite via ``mode=ro`` URI (through :func:`_open_readonly`)
    so it can be called on the read endpoint without instantiating the
    writer ``RuntimeStateStore`` (whose ``__init__`` runs DDL on every
    call). Same WHERE/ORDER BY as the writer version
    (``WHERE archived_at IS NULL ORDER BY updated_at DESC``). Same dict
    shape per row so callers don't need to branch.

    Empty list collapses three cases (matching the per-source
    primitives' convention): file missing, file corrupt/unreadable,
    no rows. The intake schema lives behind a writer-side migration;
    a missing file simply means "no intake activity yet."
    """

    path = db_path if isinstance(db_path, Path) else Path(db_path)
    with _open_readonly(path) as conn:
        if conn is None:
            return []
        try:
            rows = conn.execute(
                """
                SELECT id, brief_id_draft, role_title, current_step,
                       state_json, started_at, updated_at,
                       completed_at, archived_at
                FROM intake_sessions
                WHERE archived_at IS NULL
                ORDER BY updated_at DESC
                """
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return []
    return [_row_to_intake_session(row) for row in rows]


def get_intake_session(
    db_path: str | Path, *, session_id: int
) -> dict | None:
    """Read-only fetch of one intake_sessions row by id.

    Parallel to :func:`cloris.intake_sessions.get_intake_session` —
    same dict shape, same "return regardless of archived state"
    contract (the GET endpoint is used for direct deep-links and
    resume flows where a recruiter may want to inspect or unarchive
    an old session). Returns ``None`` for missing/corrupt DB or
    unknown ``session_id``.
    """

    path = db_path if isinstance(db_path, Path) else Path(db_path)
    with _open_readonly(path) as conn:
        if conn is None:
            return None
        try:
            row = conn.execute(
                """
                SELECT id, brief_id_draft, role_title, current_step,
                       state_json, started_at, updated_at,
                       completed_at, archived_at
                FROM intake_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return None
    if row is None:
        return None
    return _row_to_intake_session(row)


# --- reflection_sessions read primitives ------------------------------------
#
# Parallel to ``shared.runtime_state.reflection.{get_reflection_session,
# get_active_reflection_for_brief}`` but opened via ``_open_readonly``
# so the polled GET endpoints in ``cloris.api`` can answer without
# instantiating ``RuntimeStateStore`` (whose ``__init__`` runs DDL +
# ``INSERT OR REPLACE INTO meta`` on every call). Returned dict shape
# matches ``shared.runtime_state.reflection._row_to_session`` so the
# wire model (``cloris.models.ReflectionSession``) consumes either path
# without branching. Reflection sessions colocate with intake sessions
# in the same SQLite file per the ``_reflection_store_factory =
# _intake_store`` aliasing in ``cloris.api``.


def _row_to_reflection_session(row: sqlite3.Row) -> dict:
    """Mirror of ``shared.runtime_state.reflection._row_to_session`` for
    the read path. Kept private here so the layering rule (no imports
    back into ``shared.runtime_state.{store,reflection}``) holds —
    enforced by ``tests/test_read_models.py``.
    """

    raw_state = row["state_json"]
    if raw_state:
        try:
            parsed_state = json.loads(raw_state)
        except (json.JSONDecodeError, TypeError):
            parsed_state = {}
    else:
        parsed_state = {}
    if not isinstance(parsed_state, dict):
        parsed_state = {}
    return {
        "id": row["id"],
        "brief_id": row["brief_id"],
        "source_run_id": row["source_run_id"],
        "current_phase": row["current_phase"],
        "state_json": parsed_state,
        "steering_iterations": row["steering_iterations"],
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
        "discarded_at": row["discarded_at"],
        "brief_version_committed": row["brief_version_committed"],
        "research_error": row["research_error"],
    }


def get_reflection_session(
    db_path: str | Path, *, session_id: int
) -> dict | None:
    """Read-only fetch of one reflection_sessions row by id.

    Parallel to :func:`shared.runtime_state.reflection.get_reflection_session`
    but opens the SQLite via ``mode=ro`` URI. Returns the row regardless
    of completed / discarded state (the GET endpoint serves both
    in-flight resume and post-mortem inspection). ``None`` collapses
    missing-DB / corrupt-DB / unknown-id.
    """

    path = db_path if isinstance(db_path, Path) else Path(db_path)
    with _open_readonly(path) as conn:
        if conn is None:
            return None
        try:
            row = conn.execute(
                """
                SELECT id, brief_id, source_run_id, current_phase,
                       state_json, steering_iterations, started_at,
                       updated_at, completed_at, discarded_at,
                       brief_version_committed, research_error
                FROM reflection_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return None
    if row is None:
        return None
    return _row_to_reflection_session(row)


def get_active_reflection_for_brief(
    db_path: str | Path, *, brief_id: str
) -> dict | None:
    """Read-only fetch of the active (non-terminal) reflection for a brief.

    Parallel to :func:`shared.runtime_state.reflection.get_active_reflection_for_brief`.
    Active means ``completed_at IS NULL AND discarded_at IS NULL``.
    Returns the most recently updated row when (despite the API guard)
    more than one happens to be active. ``None`` collapses missing-DB /
    corrupt-DB / no-active-row.
    """

    path = db_path if isinstance(db_path, Path) else Path(db_path)
    with _open_readonly(path) as conn:
        if conn is None:
            return None
        try:
            row = conn.execute(
                """
                SELECT id, brief_id, source_run_id, current_phase,
                       state_json, steering_iterations, started_at,
                       updated_at, completed_at, discarded_at,
                       brief_version_committed, research_error
                FROM reflection_sessions
                WHERE brief_id = ?
                  AND completed_at IS NULL
                  AND discarded_at IS NULL
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (brief_id,),
            ).fetchone()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return None
    if row is None:
        return None
    return _row_to_reflection_session(row)


# --- run telemetry (Live Monitor GET .../telemetry) --------------------------

# Wire layer: ``cloris.api._monolith.api_run_telemetry`` maps these rows to
# Pydantic models. Read path stays in read_models with ``mode=ro`` URIs.


@dataclass(frozen=True)
class TelemetryAttempt:
    """One ``candidate_attempts`` row for a run (monitor window)."""

    id: int
    candidate_id: int
    work_unit_id: int | None
    stage: str
    attempt_number: int
    status: str
    failure_kind: str | None
    failure_reason: str | None
    started_at: str
    ended_at: str | None


@dataclass(frozen=True)
class TelemetryEvent:
    """One ``events`` row for a run (monitor window)."""

    id: int
    event_type: str
    candidate_id: int | None
    attempt_id: int | None
    payload_json: str | None
    created_at: str


@dataclass(frozen=True)
class RunTelemetry:
    """Bounded attempt + event slices plus totals for one ``run_id``."""

    attempts: list[TelemetryAttempt]
    events: list[TelemetryEvent]
    last_event_at: str | None
    attempts_total: int
    events_total: int


def run_telemetry(
    db_path: Path | str,
    *,
    run_id: int,
    attempts_limit: int = 50,
    events_limit: int = 30,
) -> RunTelemetry:
    """Return recent attempts and events for ``run_id`` (most-recent first).

    Opens ``db_path`` read-only. On missing DB / unreadable file / schema
    errors, returns empty lists and zero totals.
    """

    path = db_path if isinstance(db_path, Path) else Path(db_path)
    empty = RunTelemetry(
        attempts=[], events=[], last_event_at=None, attempts_total=0, events_total=0
    )
    with _open_readonly(path) as conn:
        if conn is None:
            return empty
        try:
            attempts_total = int(
                conn.execute(
                    "SELECT COUNT(*) FROM candidate_attempts WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0]
            )
            events_total = int(
                conn.execute(
                    "SELECT COUNT(*) FROM events WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0]
            )
            last_row = conn.execute(
                "SELECT MAX(created_at) AS m FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            last_event_at = str(last_row["m"]) if last_row and last_row["m"] else None

            attempt_rows = conn.execute(
                """
                SELECT id, candidate_id, work_unit_id, stage, attempt_number,
                       status, failure_kind, failure_reason, started_at, ended_at
                FROM candidate_attempts
                WHERE run_id = ?
                ORDER BY attempt_number DESC
                LIMIT ?
                """,
                (run_id, attempts_limit),
            ).fetchall()
            event_rows = conn.execute(
                """
                SELECT id, event_type, candidate_id, attempt_id, payload_json, created_at
                FROM events
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (run_id, events_limit),
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return empty

    attempts = [
        TelemetryAttempt(
            id=int(r["id"]),
            candidate_id=int(r["candidate_id"]),
            work_unit_id=int(r["work_unit_id"]) if r["work_unit_id"] is not None else None,
            stage=str(r["stage"]),
            attempt_number=int(r["attempt_number"]),
            status=str(r["status"]),
            failure_kind=str(r["failure_kind"]) if r["failure_kind"] is not None else None,
            failure_reason=str(r["failure_reason"]) if r["failure_reason"] is not None else None,
            started_at=str(r["started_at"]),
            ended_at=str(r["ended_at"]) if r["ended_at"] is not None else None,
        )
        for r in attempt_rows
    ]
    events = [
        TelemetryEvent(
            id=int(r["id"]),
            event_type=str(r["event_type"]),
            candidate_id=int(r["candidate_id"]) if r["candidate_id"] is not None else None,
            attempt_id=int(r["attempt_id"]) if r["attempt_id"] is not None else None,
            payload_json=str(r["payload_json"]) if r["payload_json"] is not None else None,
            created_at=str(r["created_at"]),
        )
        for r in event_rows
    ]
    return RunTelemetry(
        attempts=attempts,
        events=events,
        last_event_at=last_event_at,
        attempts_total=attempts_total,
        events_total=events_total,
    )


# --- helpers ----------------------------------------------------------------


def _parse_iso_to_epoch(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()
