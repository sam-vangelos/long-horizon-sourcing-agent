"""Cloris status aggregator.

Pure read-only aggregation over per-state-dir canonical SQLite stores plus
the per-state-dir ``worker.json`` sidecar plus a small ``progress.json``
peek for LinkedIn resumability. This is the single seam between Cloris and
the canonical runtime-state files; it has no FastAPI imports, no pywebview
imports, and **no** import of the canonical runtime-state store class in
production paths.

Why not the canonical store class? Its constructor runs unconditional DDL
plus ``INSERT OR REPLACE INTO meta`` on every instantiation
(``shared/runtime_state/store.py:56-213``), which would make the API
process silently writable against active runtime state. We open the file
directly via ``sqlite3.connect(f"file:{path}?mode=ro", uri=True)`` so the
read path is honestly read-only.

Public surfaces:

- :func:`enumerate_state_dirs` — list discovered state dirs across LinkedIn
  and GitHub.
- :func:`read_latest_run_readonly` — open one canonical SQLite read-only and
  return the latest ``runs`` row as a dict, or ``None``.
- :func:`read_worker_sidecar` — thin wrapper over :func:`cloris.worker.read_sidecar`.
- :func:`linkedin_resumable` — Slice 4 read-only resumability oracle that
  mirrors the semantics of ``linkedin.session_orchestrator._resume_has_pending_work``
  by reading ``progress.json`` directly. Returns ``True``/``False``/``None``;
  ``None`` means "unknown" (file missing or unreadable). Reimplemented here
  rather than imported so this module stays decoupled from
  ``linkedin/session_orchestrator.py`` and never triggers a ``mkdir``.
- :func:`aggregate_status` — orchestrate the above and return a
  :class:`cloris.models.StatusResponse`. Slice 4 enriches each
  :class:`cloris.models.StateDirEntry` with worker-sidecar provenance fields
  and a ``resumable`` hint (LinkedIn only).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cloris.models import (
    AttemptHealthSummary,
    BriefCounts,
    BriefInfo,
    CandidateCardSummary,
    CandidateDecisionSummary,
    CandidateDetailResponse,
    CandidateNoteEntry,
    CrossSourceLink,
    DecisionCounts,
    EntryKind,
    FailureKindCount,
    LaneMetricsSummary,
    LatestRunRef,
    ModuleStatus,
    ProductLifecycle,
    RunDetail,
    RunReportResponse,
    RunSummary,
    StateDirEntry,
    StatusResponse,
    WorkspaceResponse,
    WorkUnitProgressSummary,
)
from cloris.launchers import LAUNCHERS, known_sources
from cloris.worker import is_pid_alive
from shared import config
from shared.output_paths import enumerate_state_dirs
from shared.run_status_constants import (
    TERMINAL_RUN_STATUSES,
    derive_attention_state,
    live_signal_eligible,
)
from shared.runtime_state import read_models
from shared.runtime_state.lane_metrics import lane_metrics_for_run


log = logging.getLogger(__name__)

# Phase C, slice C2: rolling-window for "saves this week" stats on the
# Workspace surface. 7 days is the recruiter-natural review cadence.
_ONE_WEEK = timedelta(days=7)


# Phase 4: stalled classifier thresholds.
#
# A worker that's been silent on the success path for 5 minutes AND has
# at least 3 recent retryable failures dominating the histogram is
# almost always blocked on provider degradation (rate limits, 5xx
# storms). Lower thresholds would cause false alarms on long-but-normal
# operations (captcha waits, slow page loads); higher thresholds delay
# actionable signal too long.
_STALLED_NO_SUCCESS_THRESHOLD_S = 300.0
_STALLED_MIN_FAILURES = 3
_STALLED_DOMINATION_RATIO = 0.8
# Failure kinds that indicate "the orchestrator is alive but stuck on
# the provider side" rather than "the orchestrator hit a terminal
# error." Drawn from shared/failures.py:21 retryable HTTP set.
_RETRYABLE_FAILURE_KINDS: frozenset[str] = frozenset(
    {
        "rate_limit",
        "timeout",
        "capacity",
        "browser_disconnect",
        "http_408",
        "http_409",
        "http_425",
        "http_429",
        "http_500",
        "http_502",
        "http_503",
        "http_504",
        "http_529",
    }
)


def _trial_visible_sources() -> set[str]:
    return set(config.CLORIS_TRIAL_ALLOWED_MODULES or ("linkedin",))


def _module_statuses() -> list[ModuleStatus]:
    trial_visible = _trial_visible_sources()
    out: list[ModuleStatus] = []
    for source in known_sources():
        launcher = LAUNCHERS[source]
        production = launcher.pipeline_state == "production"
        visible = source in trial_visible if config.CLORIS_TRIAL_MODE else True
        launchable = production and (source in trial_visible if config.CLORIS_TRIAL_MODE else True)
        # Reopen P7.1 (spec §8): the registry's ``launchable``/``sunset``
        # flags are the administrative-retirement gate, distinct from
        # ``pipeline_state``'s maturity signal computed above. A registry
        # entry with ``launchable=False`` (sunset modules today; any future
        # administrative gate tomorrow) must force the payload's
        # ``launchable`` false regardless of pipeline_state/trial
        # visibility — the single spawn choke point
        # (``_spawn_worker_for_source``) enforces the same gate
        # unconditionally, so the status payload must not promise a
        # launch the launch endpoint will refuse.
        launchable = launchable and launcher.launchable
        out.append(
            ModuleStatus(
                source=source,  # type: ignore[arg-type]
                pipeline_state=launcher.pipeline_state,
                launchable=launchable,
                visible=visible,
                sunset=launcher.sunset,
            )
        )
    return out


def _product_run_contract_fields(
    *,
    latest_run: RunSummary | None,
    worker_state: str,
    lifecycle: ProductLifecycle,
    attention_state: str,
) -> tuple[bool, str | None, bool]:
    """Derive recruiter-facing lifecycle DTO fields for one state dir."""

    status = latest_run.status if latest_run is not None else None
    projection_disagreement = status in TERMINAL_RUN_STATUSES and worker_state in {
        "alive",
        "alive_silent",
    }
    # Terminal canonical runs may still project writing_report while the
    # sidecar is alive; recruiters must not see active=True in that case.
    if status in TERMINAL_RUN_STATUSES or attention_state == "terminal":
        active = False
    else:
        active = attention_state in {"live", "stalled", "recovering"} or lifecycle in {
            "preparing",
            "strategizing",
            "searching",
            "reviewing",
            "writing_report",
        }
    terminal_reason: str | None = None
    if status in TERMINAL_RUN_STATUSES:
        terminal_reason = (
            latest_run.stop_reason if latest_run is not None and latest_run.stop_reason
            else status
        )
    return active, terminal_reason, projection_disagreement


def _product_lifecycle(
    *,
    latest_run: RunSummary | None,
    worker_state: str,
    work_unit_progress: WorkUnitProgressSummary | None,
) -> ProductLifecycle:
    """Collapse runtime truth into the product lifecycle vocabulary."""

    status = latest_run.status if latest_run is not None else None
    worker_live = worker_state in {"alive", "alive_silent"}

    if worker_state == "stale":
        return "recovering"
    if status == "running" and not worker_live:
        return "recovering"
    if worker_live and latest_run is None:
        return "preparing"
    if worker_live and status in TERMINAL_RUN_STATUSES:
        return "writing_report"
    if worker_live and status == "running":
        progress = work_unit_progress
        if progress is None or progress.kind != "counts":
            return "strategizing"
        if progress.done == 0 and progress.in_progress == 0:
            return "strategizing"
        if progress.in_progress > 0:
            return "reviewing"
        return "searching"
    if status in {"completed", "succeeded"}:
        return "finished"
    if status in TERMINAL_RUN_STATUSES:
        return "ready"
    return "ready"


def _classify_stalled(
    *,
    worker_state: str,
    health: read_models.AttemptHealth,
) -> tuple[bool, str | None]:
    """Decide whether the run is stalled and which failure_kind dominates.

    Stalled = alive worker + no successful attempt in N seconds + recent
    failures dominate around a retryable kind. Returns ``(False, None)``
    when the worker is not alive, the success age is fresh, or failures
    don't dominate enough.
    """

    if worker_state != "alive":
        return False, None
    if health.dominant_failure_kind is None:
        return False, None
    if health.failed_in_window < _STALLED_MIN_FAILURES:
        return False, None
    last_success = health.last_success_age_s
    if last_success is not None and last_success < _STALLED_NO_SUCCESS_THRESHOLD_S:
        return False, None
    total = health.total_attempts_in_window
    if total == 0:
        return False, None
    domination = health.failed_in_window / total
    if domination < _STALLED_DOMINATION_RATIO:
        return False, None
    if health.dominant_failure_kind not in _RETRYABLE_FAILURE_KINDS:
        return False, None
    return True, health.dominant_failure_kind


def _summarize_attempt_health(
    health: read_models.AttemptHealth,
) -> AttemptHealthSummary:
    return AttemptHealthSummary(
        total_attempts_in_window=health.total_attempts_in_window,
        succeeded_in_window=health.succeeded_in_window,
        failed_in_window=health.failed_in_window,
        last_success_age_s=health.last_success_age_s,
        recent_failures=[
            FailureKindCount(kind=fc.kind, count=fc.count)
            for fc in health.recent_failures
        ],
        dominant_failure_kind=health.dominant_failure_kind,
    )


def _summarize_work_unit_progress(
    progress: read_models.WorkUnitProgress,
) -> WorkUnitProgressSummary:
    return WorkUnitProgressSummary(
        kind=progress.kind,
        queued=progress.queued,
        in_progress=progress.in_progress,
        done=progress.done,
        skipped=progress.skipped,
        error=progress.error,
    )


# Phase 1.6: an alive worker whose heartbeat hasn't been bumped in this many
# seconds gets demoted to ``worker_state="alive_silent"`` so the UI can
# promote it to the attention lane. 5 minutes is wide enough to avoid noise
# from long-but-normal operations (captcha waits, slow page loads) and tight
# enough to catch sleep, hang, and stalled rate-limit retries.
ALIVE_SILENT_THRESHOLD_S = 300.0


_RUNTIME_DB_FILENAME = "runtime_state.sqlite3"
_PROGRESS_JSON_FILENAME = "progress.json"
_LATEST_RUN_QUERY = (
    "SELECT id, source, brief_id, mode, status, stop_reason, started_at, ended_at "
    "FROM runs ORDER BY id DESC LIMIT 1"
)
_LATEST_RUN_IDENTITY_QUERY = (
    "SELECT brief_id, brief_path_at_launch, brief_content_hash, brief_snapshot_json "
    "FROM runs ORDER BY id DESC LIMIT 1"
)


def read_latest_run_readonly(db_path: Path) -> dict | None:
    """Return the latest ``runs`` row from ``db_path`` as a dict, or ``None``.

    Phase 2: thin wrapper that delegates to
    :func:`shared.runtime_state.read_models.latest_run_summary`. The dict
    return shape is preserved for tests and any external caller that
    imports this symbol; new code should consume the read model directly.

    The function previously hard-coded the ``runs`` SELECT shape inline;
    moving that SQL into ``read_models`` is the foundation that lets
    Run Review and Authoring Loop ship without each reinventing it.
    """

    summary = read_models.latest_run_summary(db_path)
    if summary is None:
        return None
    # Preserve the historical dict-with-brief_id shape so callers that
    # consumed the row directly (older tests, the aggregator's
    # brief_id_from_run pull below) keep working unchanged. brief_id is
    # not on RunSummary because the wire model doesn't expose it; the
    # aggregator pulls it via a separate read below.
    return {
        "id": summary.id,
        "status": summary.status,
        "stop_reason": summary.stop_reason,
        "mode": summary.mode,
        "started_at": summary.started_at,
        "ended_at": summary.ended_at,
    }


def _read_latest_run_brief_id(db_path: Path) -> str | None:
    """Pull just the brief_id of the latest run, used by the aggregator.

    Kept here rather than in ``read_models`` because brief_id is the
    raw column value with no transformation; surfacing it through
    ``RunSummary`` would push wire-shape concerns into the read model.
    """

    identity = _read_latest_run_identity(db_path)
    return identity.brief_id if identity else None


@dataclass(frozen=True)
class _LatestRunIdentity:
    """Phase 3: brief-identity columns from the latest runs row.

    Kept private to control_plane because the wire-side model
    (``StateDirEntry``) decomposes these into recruiter-facing fields
    (``brief_role_title``, ``brief_drift_since_last_run``). Future
    surfaces that need the raw values can promote this to a public read
    model in shared/runtime_state/read_models.py.
    """

    brief_id: str | None
    brief_path_at_launch: str | None
    brief_content_hash: str | None
    brief_snapshot_json: str | None


def _read_latest_run_identity(db_path: Path) -> _LatestRunIdentity | None:
    """Pull the brief-identity columns from the latest run.

    Returns None for missing/corrupt DB or when the brief-identity
    columns aren't present (pre-Phase-3 schema). The latter is also
    rendered as None at the wire shape so the UI gracefully falls back
    to state_key. Avoids the `'column' in row.keys()` overhead by
    relying on schema gating: if the migration ran, the columns exist.
    """

    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return None
    conn.row_factory = sqlite3.Row
    try:
        try:
            row = conn.execute(_LATEST_RUN_IDENTITY_QUERY).fetchone()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            # Columns not yet migrated (read against a partially-written
            # DB or one this Cloris version is too new for) — collapse to
            # plain brief_id only.
            try:
                row = conn.execute(
                    "SELECT brief_id, NULL AS brief_path_at_launch, "
                    "NULL AS brief_content_hash, NULL AS brief_snapshot_json "
                    "FROM runs ORDER BY id DESC LIMIT 1"
                ).fetchone()
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                return None
        if row is None:
            return None
        return _LatestRunIdentity(
            brief_id=row["brief_id"],
            brief_path_at_launch=row["brief_path_at_launch"],
            brief_content_hash=row["brief_content_hash"],
            brief_snapshot_json=row["brief_snapshot_json"],
        )
    finally:
        conn.close()


def _extract_brief_role_title(snapshot_json: str | None) -> str | None:
    """Pull ``role_title`` out of the brief snapshot JSON, if present."""

    if not snapshot_json or snapshot_json == "{}":
        return None
    try:
        snapshot = json.loads(snapshot_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(snapshot, dict):
        return None
    value = snapshot.get("role_title")
    return value if isinstance(value, str) and value else None


def _extract_brief_linkedin_project(snapshot_json: str | None) -> str | None:
    if not snapshot_json or snapshot_json == "{}":
        return None
    try:
        snapshot = json.loads(snapshot_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(snapshot, dict):
        return None
    value = snapshot.get("linkedin_project")
    return value if isinstance(value, str) and value else None


def _extract_brief_target_modules(snapshot_json: str | None) -> list[str]:
    """Pull ``target_modules`` out of the brief snapshot JSON, if present."""
    if not snapshot_json or snapshot_json == "{}":
        return []
    try:
        snapshot = json.loads(snapshot_json)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(snapshot, dict):
        return []
    value = snapshot.get("target_modules")
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    return []


def _detect_brief_drift(
    brief_path_at_launch: str | None,
    brief_content_hash: str | None,
) -> bool | None:
    """Return whether the on-disk brief differs from the pinned hash.

    None when the comparison cannot be made (no pinning, missing/
    unreadable file). True when hashes differ — the recruiter modified
    the brief after the run started. False when hashes match.
    """

    if not brief_path_at_launch or not brief_content_hash:
        return None
    from shared.brief_identity import hash_current_brief_on_disk

    current = hash_current_brief_on_disk(brief_path_at_launch)
    if current is None:
        return None
    return current != brief_content_hash


def is_runtime_state_corrupt(db_path: Path) -> bool:
    """Return True iff ``db_path`` exists but cannot be opened/read.

    Distinguishes "DB exists but is corrupt" (truncated file, not a SQLite
    DB, schema gone) from "DB missing" (file not on disk) and "DB readable
    but no runs" (empty ``runs`` table). The aggregator surfaces this as a
    boolean on :class:`StateDirEntry` so the UI can render a distinct
    state-row line ("runtime state unreadable" vs "no run recorded" vs
    "runtime DB missing").

    Returns ``False`` when:
    - the file does not exist (caller should already render "DB missing")
    - the file exists and a trivial read against ``sqlite_master`` succeeds

    Returns ``True`` only when the file exists but
    :class:`sqlite3.OperationalError` or :class:`sqlite3.DatabaseError`
    fires during open or the probe query.

    Phase 2 will fold this into ``shared/runtime_state/read_models.py``;
    until then it lives here so Phase 1 can ship the distinction without
    blocking on the read-model extraction.
    """

    if not db_path.exists():
        return False

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        # Probe sqlite_master rather than the runs table so the result
        # doesn't depend on whether the schema has been migrated yet.
        # Any well-formed SQLite DB has sqlite_master.
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        return False
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return True
    finally:
        if conn is not None:
            conn.close()


def read_worker_sidecar(state_dir: Path) -> dict | None:
    """Return the parsed ``worker.json`` for ``state_dir``, or ``None``.

    Thin wrapper around :func:`cloris.worker.read_sidecar` so the control
    plane stays the single seam between Cloris and per-state-dir disk
    artifacts (canonical SQLite + the ``worker.json`` sidecar). Slice 4
    uses this to enrich ``GET /api/status`` with worker provenance and to
    classify ``worker_state`` per :class:`cloris.models.WorkerState`.
    """

    from cloris.worker import read_sidecar

    return read_sidecar(state_dir)


def _heartbeat_age_seconds(heartbeat_at: object) -> float | None:
    """Return seconds since ``heartbeat_at`` (ISO-8601 string), or None.

    Phase 1.6 stores ISO-8601 timestamps via ``datetime.now(timezone.utc).isoformat()``
    so we parse with ``datetime.fromisoformat`` (round-trip stable since
    Python 3.11). Anything unparseable collapses to None — the aggregator
    must not fail one state dir over a single bad sidecar.
    """

    if not isinstance(heartbeat_at, str):
        return None
    try:
        ts = datetime.fromisoformat(heartbeat_at)
    except ValueError:
        return None
    if ts.tzinfo is None:
        # Treat naive timestamps as UTC. The worker's _now always emits
        # tz-aware ISO-8601, so this branch only fires on edge-case sidecars
        # written by an older or external tool; we still want a sane number.
        ts = ts.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    if age < 0:
        # Clock skew (sidecar written by a host whose clock is ahead of
        # ours). Floor to 0 rather than reporting a negative age.
        return 0.0
    return age


def linkedin_resumable(state_dir: Path) -> bool | None:
    """Return whether the LinkedIn run in ``state_dir`` has pending work.

    Phase 2: thin wrapper that delegates to
    :func:`shared.runtime_state.read_models.has_pending_work`. The two
    used to be parallel implementations; now they share one. The
    intentional divergence from
    ``linkedin/session_orchestrator._resume_has_pending_work`` is
    preserved (active worker biases toward "attempt resume" on missing
    data; passive read model returns "unknown") — see the read model's
    docstring for the full rationale.
    """

    return read_models.has_pending_work(state_dir)


def _classify_entry(
    *,
    latest_run: RunSummary | None,
    is_archived: bool,
) -> EntryKind:
    """Phase 1C: classify a state-dir into the brief-taxonomy bucket.

    Order matters — ``is_archived`` wins over presence of a run because
    the user explicitly filed it away. ``orphaned_state_dir`` is the
    fallback when there's no run row at all (legacy artifacts, aborted
    CLI launches, manually-created folders). ``authored_brief`` is the
    common case: a real run exists and the user hasn't archived it.

    ``intake_only`` (intake session exists but no run yet) is reserved
    for Phase 4 — this v0 classifier doesn't read the intake_sessions
    table because the aggregator's read-only contract pins it to the
    canonical SQLite without crossing into the intake DB. When Phase 4
    wires the onboarding flow, the classifier will be extended; for now
    intake-only state dirs collapse to ``orphaned_state_dir``, which is
    safe (under-counts authored briefs by at most a small number).
    """

    if is_archived:
        return "archived"
    if latest_run is None or latest_run.id is None:
        return "orphaned_state_dir"
    return "authored_brief"


def _compute_brief_counts(entries: list[StateDirEntry]) -> BriefCounts:
    """Phase 1C: roll up the per-entry kind + run status into masthead counts.

    Active = working ∪ paused. The split lets the UI render either a
    single "5 active briefs" line or a more detailed "1 working, 4
    paused" breakdown depending on the surface. Finished / lost /
    archived / orphaned are tracked separately so the homescreen can
    accurately report what's available without conflating recently-done
    work with abandoned / orphan rot.

    P7.4: ``interrupted`` / ``governor_limit_reached`` used to bucket to
    ``paused`` unconditionally, ignoring ``resumable`` (see
    :func:`linkedin_resumable` / :func:`shared.runtime_state.read_models.has_pending_work`).
    That was the standing "98% of paused runs are actually finished" bug —
    a run that got interrupted (or hit its governor limit) *after* draining
    its work queue has nothing left to resume; it belongs in ``finished``,
    not in a bucket that promises the user a resume affordance. The rule
    now keys off ``resumable`` directly:

    - ``resumable is True`` (positive evidence of queued/in-progress work,
      LinkedIn-only today) → ``paused``.
    - ``resumable is False`` (positive evidence the queue is drained) →
      ``finished`` — this is the "actually finished" case the bug
      description names.
    - ``resumable is None`` (unknown — non-LinkedIn sources have no
      progress.json projection at all, or the sidecar/progress file is
      missing/malformed) → ``lost``. We have no positive evidence the run
      completed its work, so it does not get the "finished" label; fail
      closed the same way ``lost`` already treats ``abandoned`` / ``error``
      — surface it as something that may need a rerun rather than quietly
      implying it's done.

    ``failed`` (a real, distinct terminal status — see
    ``shared.reconciler`` / ``shared.runtime_state.store.SIDE_EFFECT_RETRYABLE_STATUSES``)
    previously fell through to the "unknown status" catch-all, which
    bucketed it as ``paused``; it now buckets with ``lost`` alongside
    ``abandoned`` / ``error``. The literal status ``"paused"`` has never
    had a producer anywhere in the codebase and is removed — nothing sets
    ``runs.status = "paused"``.
    """

    archived = 0
    orphaned = 0
    working = 0
    paused = 0
    finished = 0
    lost = 0
    for e in entries:
        if e.kind == "archived":
            archived += 1
            continue
        if e.kind in ("orphaned_state_dir", "intake_only"):
            orphaned += 1
            continue
        # authored_brief — bucket by run status.
        status = (e.latest_run.status if e.latest_run else None) or ""
        if status == "running":
            working += 1
        elif status in ("interrupted", "governor_limit_reached"):
            if e.resumable is True:
                paused += 1
            elif e.resumable is False:
                finished += 1
            else:
                lost += 1
        elif status in ("completed", "succeeded"):
            finished += 1
        elif status in ("abandoned", "error", "failed"):
            lost += 1
        else:
            # Unknown / null status on an authored brief. Same
            # resumable-gated rule as above — only claim "paused" when
            # there's positive evidence of pending work; otherwise this
            # is unclassifiable so it lands in "lost" rather than the
            # false-promise "paused" bucket.
            if e.resumable is True:
                paused += 1
            else:
                lost += 1
    return BriefCounts(
        active=working + paused,
        working=working,
        paused=paused,
        finished=finished,
        lost=lost,
        archived=archived,
        orphaned=orphaned,
    )


def aggregate_status(state_root: Path | None = None) -> StatusResponse:
    """Build a :class:`StatusResponse` for ``GET /api/status``.

    Pure function of disk: walks every discovered state dir, reads the
    latest ``runs`` row read-only when a DB is present, reads the optional
    ``worker.json`` sidecar to classify ``worker_state``, and (LinkedIn
    only) reads ``progress.json`` to compute ``resumable``. Returns a
    stable response sorted by ``(source, state_key)`` so tests and clients
    see deterministic ordering.

    Slice 4 enrichment per :class:`cloris.models.StateDirEntry`:

    - ``worker_json_present`` — whether ``worker.json`` exists and parses.
    - ``worker_pid`` — int when sidecar's ``pid`` is an int, else ``None``.
    - ``worker_alive`` — :func:`cloris.worker.is_pid_alive` result when
      ``worker_pid`` is set; ``None`` otherwise.
    - ``worker_mode`` / ``worker_input_mode`` / ``brief_path_from_worker`` —
      forwarded verbatim from the sidecar.
    - ``worker_state`` ∈ ``{"missing", "alive", "stale"}`` — derived:
      ``"missing"`` when no parseable sidecar; ``"alive"`` when sidecar
      has an int PID currently alive; ``"stale"`` when sidecar exists but
      its PID is missing/non-int or dead.
    - ``resumable`` — :func:`linkedin_resumable` for LinkedIn, ``None``
      for GitHub (no analogous progress.json gate).
    """

    entries: list[StateDirEntry] = []
    for source, state_dir in enumerate_state_dirs(state_root):
        db_path = state_dir / _RUNTIME_DB_FILENAME
        runtime_state_present = db_path.exists()
        runtime_state_corrupt = (
            is_runtime_state_corrupt(db_path) if runtime_state_present else False
        )

        latest_run: RunSummary | None = None
        brief_id_from_run: str | None = None
        brief_role_title: str | None = None
        brief_linkedin_project: str | None = None
        brief_drift_since_last_run: bool | None = None
        attempt_health_summary: AttemptHealthSummary | None = None
        work_unit_progress_summary: WorkUnitProgressSummary | None = None
        latest_run_id: int | None = None
        # Phase 1C: brief-taxonomy fields read off the run row (defaults
        # apply when there is no run / no migration applied yet).
        is_archived: bool = False
        if runtime_state_present and not runtime_state_corrupt:
            summary = read_models.latest_run_summary(db_path)
            if summary is not None:
                latest_run = RunSummary(
                    id=summary.id,
                    status=summary.status,
                    stop_reason=summary.stop_reason,
                    mode=summary.mode,
                    started_at=summary.started_at,
                    ended_at=summary.ended_at,
                )
                latest_run_id = summary.id
                is_archived = summary.is_archived
            identity = _read_latest_run_identity(db_path)
            if identity is not None:
                brief_id_from_run = identity.brief_id
                brief_role_title = _extract_brief_role_title(
                    identity.brief_snapshot_json
                )
                brief_linkedin_project = _extract_brief_linkedin_project(
                    identity.brief_snapshot_json
                )
                brief_drift_since_last_run = _detect_brief_drift(
                    identity.brief_path_at_launch,
                    identity.brief_content_hash,
                )
            # Phase 4: attempt health + work_unit progress for the
            # latest run. Skipped when there's no run yet (latest_run_id
            # is None) so the read_models functions don't have to
            # invent an "empty health" path for the no-run case.
            if latest_run_id is not None:
                attempt_health_summary = _summarize_attempt_health(
                    read_models.attempt_health(db_path, run_id=latest_run_id)
                )
                progress_kinds = LAUNCHERS[source].progress_kinds
                progress_kind = LAUNCHERS[source].progress_kind
                if progress_kinds:
                    work_unit_progress_summary = _summarize_work_unit_progress(
                        read_models.work_unit_progress_multi(
                            db_path, run_id=latest_run_id, kinds=progress_kinds
                        )
                    )
                elif progress_kind:
                    work_unit_progress_summary = _summarize_work_unit_progress(
                        read_models.work_unit_progress(
                            db_path, run_id=latest_run_id, kind=progress_kind
                        )
                    )

        sidecar = read_worker_sidecar(state_dir)
        worker_json_present = sidecar is not None
        worker_pid: int | None = None
        worker_alive: bool | None = None
        worker_mode: str | None = None
        worker_input_mode: str | None = None
        brief_path_from_worker: str | None = None
        worker_state: str = "missing"
        heartbeat_age_s: float | None = None

        if sidecar is not None:
            pid_raw = sidecar.get("pid")
            mode_raw = sidecar.get("mode")
            input_mode_raw = sidecar.get("input_mode")
            brief_path_raw = sidecar.get("brief_path")
            heartbeat_raw = sidecar.get("heartbeat_at")
            worker_mode = mode_raw if isinstance(mode_raw, str) else None
            worker_input_mode = (
                input_mode_raw if isinstance(input_mode_raw, str) else None
            )
            brief_path_from_worker = (
                brief_path_raw if isinstance(brief_path_raw, str) else None
            )
            heartbeat_age_s = _heartbeat_age_seconds(heartbeat_raw)
            if isinstance(pid_raw, int) and not isinstance(pid_raw, bool):
                worker_pid = pid_raw
                worker_alive = is_pid_alive(pid_raw)
                if not worker_alive:
                    worker_state = "stale"
                elif (
                    heartbeat_age_s is not None
                    and heartbeat_age_s > ALIVE_SILENT_THRESHOLD_S
                ):
                    # PID alive but heartbeat hasn't been bumped recently —
                    # likely sleep, hang, or stalled retry. UI promotes
                    # this to the attention lane.
                    worker_state = "alive_silent"
                else:
                    worker_state = "alive"
            else:
                worker_pid = None
                worker_alive = None
                worker_state = "stale"

        if source == "linkedin":
            resumable = linkedin_resumable(state_dir)
        else:
            resumable = None

        # Phase 4: classify run_stalled now that we have both worker_state
        # and attempt_health populated. Done outside the latest_run_id
        # block because the classifier is a pure function and the result
        # always lives on StateDirEntry (False/None for state dirs that
        # don't have an alive worker).
        if (
            attempt_health_summary is not None
            and latest_run_id is not None
            and not runtime_state_corrupt
        ):
            stalled, stall_kind = _classify_stalled(
                worker_state=worker_state,
                health=read_models.attempt_health(
                    db_path, run_id=latest_run_id
                ),
            )
        else:
            stalled, stall_kind = False, None

        kind: EntryKind = _classify_entry(
            latest_run=latest_run, is_archived=is_archived
        )
        lifecycle = _product_lifecycle(
            latest_run=latest_run,
            worker_state=worker_state,
            work_unit_progress=work_unit_progress_summary,
        )
        latest_status = latest_run.status if latest_run is not None else None
        attention = derive_attention_state(
            worker_state=worker_state,
            latest_run_status=latest_status,
            run_stalled=stalled,
            lifecycle=lifecycle,
        )
        signal_eligible = live_signal_eligible(
            worker_state=worker_state,
            latest_run_status=latest_status,
        )
        product_active, terminal_reason, projection_disagreement = (
            _product_run_contract_fields(
                latest_run=latest_run,
                worker_state=worker_state,
                lifecycle=lifecycle,
                attention_state=attention,
            )
        )
        if projection_disagreement:
            log.warning(
                "status projection_disagreement source=%s state_key=%s "
                "canonical_status=%s worker_state=%s",
                source,
                state_dir.name,
                latest_status,
                worker_state,
            )
        entries.append(
            StateDirEntry(
                source=source,  # type: ignore[arg-type]
                state_key=state_dir.name,
                runtime_state_present=runtime_state_present,
                runtime_state_corrupt=runtime_state_corrupt,
                latest_run=latest_run,
                brief_id_from_run=brief_id_from_run,
                brief_path_from_worker=brief_path_from_worker,
                worker_json_present=worker_json_present,
                worker_pid=worker_pid,
                worker_alive=worker_alive,
                worker_mode=worker_mode,
                worker_input_mode=worker_input_mode,
                resumable=resumable,
                worker_state=worker_state,  # type: ignore[arg-type]
                heartbeat_age_s=heartbeat_age_s,
                brief_role_title=brief_role_title,
                brief_linkedin_project=brief_linkedin_project,
                brief_drift_since_last_run=brief_drift_since_last_run,
                attempt_health=attempt_health_summary,
                work_unit_progress=work_unit_progress_summary,
                run_stalled=stalled,
                stall_failure_kind=stall_kind,
                lifecycle=lifecycle,
                attention_state=attention,
                live_signal_eligible=signal_eligible,
                active=product_active,
                terminal_reason=terminal_reason,
                projection_disagreement=projection_disagreement,
                kind=kind,
            )
        )

    entries.sort(key=lambda e: (e.source, e.state_key))
    return StatusResponse(
        slice="v0-shell-slice-4",
        entries=entries,
        counts=_compute_brief_counts(entries),
        briefs=_group_entries_by_brief(entries),
        trial_mode=config.CLORIS_TRIAL_MODE,
        modules=_module_statuses(),
    )


def _group_entries_by_brief(
    entries: list[StateDirEntry],
) -> list["BriefStatusGroup"]:
    """Group ``StateDirEntry`` rows into one :class:`BriefStatusGroup`
    per ``brief_id_from_run`` (Phase F Slice F7 / Ledger L11 + L22).

    Orphaned state dirs (no ``brief_id_from_run``) collect under a single
    group with ``brief_id=None``; the frontend renders that bucket
    distinctly so the recruiter sees them without surfacing N anonymous
    "unknown brief" cards.

    Within each group, ``modules`` is sorted by source then state_key so
    LinkedIn renders before GitHub deterministically.

    Group ordering uses the most recent ``latest_run.started_at`` across
    the group's modules so the freshest brief leads — same recruiter
    affordance as a brief library, but for live state.
    """

    from cloris.models import BriefStatusGroup

    groups: dict[str | None, list[StateDirEntry]] = {}
    for entry in entries:
        key = entry.brief_id_from_run or None
        groups.setdefault(key, []).append(entry)

    result: list[BriefStatusGroup] = []
    for brief_id, modules in groups.items():
        modules.sort(key=lambda m: (m.source, m.state_key))
        # Pull role title + project from whichever module has the
        # freshest run; that's the same heuristic aggregate_workspace
        # uses for the merged brief context.
        primary = max(
            modules,
            key=lambda m: (
                m.latest_run.started_at if m.latest_run else "",
            ),
        )
        result.append(
            BriefStatusGroup(
                brief_id=brief_id,
                brief_role_title=primary.brief_role_title,
                brief_linkedin_project=primary.brief_linkedin_project,
                modules=modules,
            )
        )

    # Sort in two passes — Python's sort is stable so the second pass
    # preserves the first pass's relative order within ties:
    #   1. Most-recently-active first (max latest_run.started_at across modules).
    #   2. Briefs with a brief_id ahead of orphans.
    def _latest_started_at(g: "BriefStatusGroup") -> str:
        return max(
            (m.latest_run.started_at or "" for m in g.modules if m.latest_run),
            default="",
        )

    result.sort(key=_latest_started_at, reverse=True)
    result.sort(key=lambda g: 0 if g.brief_id is not None else 1)
    return result


# Phase B: per-run report aggregator.
# ----------------------------------
# Mirrors `aggregate_status`'s read-only contract: opens the per-state-dir
# SQLite via the read_models primitives, never touches the canonical
# write-side store class, and forwards data verbatim with minimal
# interpretation. The route
# layer (cloris/api.py) is responsible for resolving (source, state_key)
# to a state_dir and turning a missing state dir / missing run into
# HTTP 404.

CANDIDATE_LIMIT_DEFAULT = 200


def aggregate_run_report(
    state_dir: Path,
    *,
    source: str,
    state_key: str,
    run_id: int,
    candidate_limit: int = CANDIDATE_LIMIT_DEFAULT,
) -> RunReportResponse | None:
    """Return a :class:`RunReportResponse` for ``run_id`` in ``state_dir``.

    Returns ``None`` if the run does not exist (the route translates to
    HTTP 404). The DB itself being missing or corrupt also yields
    ``None`` — at the route boundary we can't distinguish "no run" from
    "no DB" usefully without leaking implementation detail; both are
    "we don't have that report."

    The function is pure aside from its read primitives and the
    on-disk brief-content-hash comparison (drift detection).
    """

    db_path = state_dir / "runtime_state.sqlite3"
    detail = read_models.run_by_id(db_path, run_id=run_id)
    if detail is None:
        return None

    # Brief-derived fields. Same helpers as aggregate_status uses for
    # the StateDirEntry's brief identity columns; reused here so the
    # Run Report renders the same role title and drift signal.
    brief_role_title = _extract_brief_role_title(detail.brief_snapshot_json)
    brief_linkedin_project = _extract_brief_linkedin_project(
        detail.brief_snapshot_json
    )
    brief_drift = _detect_brief_drift(
        detail.brief_path_at_launch, detail.brief_content_hash
    )

    # Per-run progress + attempt health. Use the same kind selector the
    # status aggregator uses so a state dir's homescreen card and its
    # run report agree on the work-unit picture. Slice 1.5 routes both
    # callsites through the launcher registry — the registry is the
    # single source-of-truth for per-source progress kinds.
    progress_kinds = LAUNCHERS[source].progress_kinds
    progress_kind = LAUNCHERS[source].progress_kind
    if progress_kinds:
        progress = read_models.work_unit_progress_multi(
            db_path, run_id=run_id, kinds=progress_kinds
        )
    elif progress_kind:
        progress = read_models.work_unit_progress(
            db_path, run_id=run_id, kind=progress_kind
        )
    else:
        progress = read_models.WorkUnitProgress(kind="not_found"
    )
    attempts = read_models.attempt_health(db_path, run_id=run_id)

    # Decisions: list of candidates touched in this run (capped) plus
    # full counts by terminal_decision. The list is ordered most-recent
    # first via the read primitive; counts are computed across the full
    # population (no cap) so a truncated list still has accurate totals.
    candidate_rows = read_models.run_decisions(
        db_path, run_id=run_id, limit=candidate_limit + 1
    )
    truncated = len(candidate_rows) > candidate_limit
    visible = candidate_rows[:candidate_limit]
    counts = _decision_counts_from_db(db_path, run_id=run_id)

    # Sourcing-judgment kernel P5: per-lane aggregated metrics from
    # canonical SQLite only (no projection reads). Legacy runs with no
    # ``lane_id`` populated on work units / terminal payloads land in
    # the ``"legacy"`` bucket; the wire shape stays additive (empty list
    # for legacy runs that have no attempts yet).
    lane_rows = lane_metrics_for_run(db_path, run_id=run_id)
    lane_metrics = [
        LaneMetricsSummary(
            lane_id=row.lane_id,
            lane_name=row.lane_name,
            lane_intent=row.lane_intent,
            acquisition_mode=row.acquisition_mode,
            result_count=row.result_count,
            candidates_seen=row.candidates_seen,
            opened_count=row.opened_count,
            evaluated_count=row.evaluated_count,
            facial_yes_count=row.facial_yes_count,
            facial_no_count=row.facial_no_count,
            facial_borderline_count=row.facial_borderline_count,
            save_count=row.save_count,
            reject_count=row.reject_count,
            review_count=row.review_count,
            review_by_reason=dict(row.review_by_reason),
            work_unit_source_ids=list(row.work_unit_source_ids),
            cost_usd=row.cost_usd,
            legacy=row.legacy,
        )
        for row in lane_rows
    ]

    return RunReportResponse(
        slice="v0-shell-slice-b1",
        source=source,  # type: ignore[arg-type]
        state_key=state_key,
        run=RunDetail(
            id=detail.id or run_id,
            source=source,  # type: ignore[arg-type]
            brief_id=detail.brief_id,
            mode=detail.mode,
            status=detail.status,
            stop_reason=detail.stop_reason,
            started_at=detail.started_at,
            ended_at=detail.ended_at,
            resumed_from_run_id=detail.resumed_from_run_id,
            brief_role_title=brief_role_title,
            brief_linkedin_project=brief_linkedin_project,
            brief_drift_since_run=brief_drift,
        ),
        work_unit_progress=_summarize_work_unit_progress(progress),
        attempt_health=_summarize_attempt_health(attempts),
        decisions=counts,
        candidates=[
            CandidateDecisionSummary(
                candidate_id=row.candidate_id,
                display_name=row.display_name,
                profile_url=row.profile_url,
                terminal_decision=row.terminal_decision,
                confidence=row.confidence,
            )
            for row in visible
        ],
        candidates_truncated=truncated,
        lane_metrics=lane_metrics,
    )


def aggregate_briefs(
    *,
    state_root: Path | None = None,
    config_dir: Path | None = None,
    decorate_runs: bool = True,
) -> list[BriefInfo]:
    """Phase D Slice D1: brief library aggregator.

    Walks every authored brief under ``config_dir`` (via
    :func:`cloris.api._scan_authored_briefs`) and decorates each with
    its runtime-state metadata: ``brief_id`` (computed via
    ``derive_brief_id(path)``, the same hash :func:`start_run`
    writes into ``runs.brief_id``), latest run summary, total runs,
    total saves.

    ``decorate_runs=False`` returns the picker shape (no run metadata)
    so the existing BriefPicker callers don't pay the per-brief
    runtime-state lookup cost.

    Sorting matches the picker: most-recently-modified first.
    """

    # Late imports keep this module from pulling cloris/api at module
    # load time (api.py imports control_plane.py for the aggregators).
    from cloris.api import _CONFIG_DIR, _CONFIG_PARENT, _scan_authored_briefs
    from shared.output_paths import derive_brief_id

    if config_dir is None:
        config_dir = _CONFIG_DIR

    raw = _scan_authored_briefs(config_dir)
    if not decorate_runs:
        return raw

    decorated: list[BriefInfo] = []
    for brief in raw:
        brief_abs_path = _CONFIG_PARENT / brief.path
        try:
            brief_id = derive_brief_id(brief_path=str(brief_abs_path))
        except Exception:
            # Defensive: a malformed brief path shouldn't fail the whole
            # library. Surface the brief without run metadata; the
            # frontend renders an empty-runs state per-row.
            decorated.append(brief)
            continue

        # Walk every state_dir whose latest run carries this brief_id.
        # In Phase D this is typically 0-1 dirs per brief (one source).
        # Phase F's identity-resolution makes 2+ dirs per brief common
        # (multi-module brief running across linkedin + github); the
        # aggregator already iterates so widening is free.
        matches = state_dirs_for_brief_id(brief_id, state_root=state_root)
        if not matches:
            # No runs yet — common for newly-authored briefs.
            decorated.append(
                brief.model_copy(
                    update={
                        "brief_id": brief_id,
                        "total_runs": 0,
                        "total_saves": 0,
                    }
                )
            )
            continue

        latest_run_id: int | None = None
        latest_run_at: str | None = None
        latest_run_status: str | None = None
        latest_run_source: str | None = None
        total_runs = 0
        total_saves = 0

        for source, state_dir in matches:
            db_path = state_dir / "runtime_state.sqlite3"
            try:
                conn = sqlite3.connect(
                    f"file:{db_path}?mode=ro", uri=True
                )
                conn.row_factory = sqlite3.Row
            except sqlite3.OperationalError:
                continue
            try:
                # Per-state-dir run count.
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM runs WHERE brief_id = ?",
                    (brief_id,),
                ).fetchone()
                total_runs += int(row["n"]) if row else 0

                # Per-state-dir saves count via brief_saves; cheaper to
                # call the read_models helper but it returns the full
                # row list. For the library card we just want the count.
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM candidates "
                    "WHERE brief_id = ? "
                    "AND terminal_decision IN ('SAVE', 'INFERENTIAL_SAVE') "
                    "AND current_lifecycle_state NOT LIKE 'failed_%'",
                    (brief_id,),
                ).fetchone()
                total_saves += int(row["n"]) if row else 0

                # Latest run across all matching state_dirs. Prefer the
                # state_dir's latest run by run_id, then keep the
                # max-started across dirs.
                row = conn.execute(
                    "SELECT id, status, started_at FROM runs "
                    "WHERE brief_id = ? "
                    "ORDER BY started_at DESC LIMIT 1",
                    (brief_id,),
                ).fetchone()
                if row is not None:
                    started = row["started_at"] or ""
                    if latest_run_at is None or started > latest_run_at:
                        latest_run_id = int(row["id"])
                        latest_run_at = started
                        latest_run_status = row["status"]
                        latest_run_source = source
            finally:
                conn.close()

        decorated.append(
            _apply_confidentiality_to_brief_aggregate(
                brief.model_copy(
                    update={
                        "brief_id": brief_id,
                        "last_run_id": latest_run_id,
                        "last_run_at": latest_run_at,
                        "last_run_status": latest_run_status,
                        "last_run_source": latest_run_source,
                        "total_runs": total_runs,
                        "total_saves": total_saves,
                    }
                )
            )
        )

    return decorated


def _apply_confidentiality_to_brief_aggregate(brief: BriefInfo) -> BriefInfo:
    """Apply confidentiality redaction at the cross-brief aggregator boundary.

    Executive Search Slice 6 wiring. Routes through
    :func:`shared.confidentiality.aggregator_visibility` so any future
    surface that aggregates across briefs (``aggregate_status``,
    market intel narratives, run reports) gets consistent redaction
    semantics from one helper.

    Behavior per class:

    - ``open`` — unchanged (passthrough).
    - ``referenceable`` — title + role visible; save count visible.
      Cross-brief surfaces don't currently emit candidate names or
      save reasons through this aggregator, so no further filtering
      is needed at this seam (the dossier surface and reflection
      surface have their own gates).
    - ``blind`` — mask ``role_title`` to
      :data:`shared.confidentiality.BLIND_TITLE_MASK`; zero out
      ``total_saves`` (the wire shape is ``int``, so the frontend
      renders the saves-count cell as the blind mask token when it
      sees ``confidentiality_class == "blind"``).
    """

    from shared.confidentiality import (
        BLIND_TITLE_MASK,
        SurfaceKind,
        aggregator_visibility,
    )

    visibility = aggregator_visibility(brief, SurfaceKind.BRIEF_AGGREGATOR)
    if visibility == "full":
        return brief
    if visibility == "masked":
        return brief.model_copy(
            update={
                "role_title": BLIND_TITLE_MASK,
                "total_saves": 0,
            }
        )
    # "redacted" — title + count stay; this aggregator doesn't emit
    # candidate-bearing detail so no further filtering at this seam.
    return brief


def state_dirs_for_brief_id(
    brief_id: str,
    *,
    state_root: Path | None = None,
) -> list[tuple[str, Path]]:
    """Return ``[(source, state_dir), ...]`` whose latest run carries this brief_id.

    Phase C-bis 0.1: brief-first workspace + candidate routes pivot on
    ``brief_id``, but the canonical SQLite is still partitioned per state
    dir. This helper enumerates every discovered state dir, opens its DB
    read-only, reads the latest run's ``brief_id``, and returns the
    pairs that match. Returns ``[]`` when no state dir matches — the
    route layer translates that to a clean 404.

    Forward-compat for Phase F: today the same brief usually lives in
    one state_dir per source (often only one source total). When
    multi-module orchestration ships, the same brief_id may live in
    multiple state_dirs (e.g. linkedin + github). The aggregator iterates
    the returned list, so adding more matches is purely additive — no
    contract change.
    """

    if not brief_id:
        return []
    matches: list[tuple[str, Path]] = []
    for source, state_dir in enumerate_state_dirs(state_root):
        db_path = state_dir / "runtime_state.sqlite3"
        latest_run_id = read_models.latest_run_in_state_dir(db_path)
        if latest_run_id is None:
            continue
        detail = read_models.run_by_id(db_path, run_id=latest_run_id)
        if detail is None:
            continue
        if detail.brief_id == brief_id:
            matches.append((source, state_dir))
    return matches


def resolve_legacy_workspace(
    source: str,
    state_key: str,
    *,
    state_root: Path | None = None,
) -> str | None:
    """Resolve a legacy ``(source, state_key)`` pair to its current brief_id.

    Used by the legacy redirect endpoint ``GET
    /api/resolve-legacy/workspace/{source}/{state_key}``. Returns the
    brief_id of the latest run in that state dir, or ``None`` when the
    state dir doesn't exist or has no runs (route → 404).
    """

    for discovered_source, discovered_state_dir in enumerate_state_dirs(state_root):
        if discovered_source != source or discovered_state_dir.name != state_key:
            continue
        db_path = discovered_state_dir / "runtime_state.sqlite3"
        latest_run_id = read_models.latest_run_in_state_dir(db_path)
        if latest_run_id is None:
            return None
        detail = read_models.run_by_id(db_path, run_id=latest_run_id)
        if detail is None or not detail.brief_id:
            return None
        return detail.brief_id
    return None


def aggregate_candidate_detail(
    *,
    brief_id: str,
    candidate_id: int,
    state_root: Path | None = None,
) -> CandidateDetailResponse | None:
    """Return a :class:`CandidateDetailResponse` keyed by brief_id + candidate_id.

    Phase C-bis 0.1: the candidate-detail route is now brief-first
    (``/api/candidate/{brief_id}/{candidate_id}``). The aggregator
    enumerates all state_dirs whose latest run carries ``brief_id``,
    finds which one holds the candidate row, and returns the detail.
    Returns ``None`` when no state dir holds the candidate — the route
    translates that to a 404.

    Cross-brief safety: even if a candidate id incidentally exists in
    another state_dir (different brief), this function only matches
    state_dirs whose latest run is the requested brief_id, so a candidate
    from a different brief cannot leak under this URL.

    ``source_run`` carries the (source, state_key, run_id) tuple needed
    by the back-link to construct ``#/run/<source>/<state_key>/<run_id>``.
    The run report stays per-run / per-source for now — only the
    workspace + candidate-detail routes are brief-first in this phase.
    """

    matches = state_dirs_for_brief_id(brief_id, state_root=state_root)
    if not matches:
        return None
    for source, state_dir in matches:
        db_path = state_dir / "runtime_state.sqlite3"
        record = read_models.candidate_by_id(db_path, candidate_id=candidate_id)
        if record is None:
            continue
        if record.brief_id != brief_id or record.source != source:
            # Defensive: a candidate row exists with this id but its brief_id
            # or source doesn't match the URL context. Skip to the next match.
            continue

        payload = read_models.candidate_terminal_payload(
            record.terminal_payload_json
        )
        save_reason, confidence = read_models.extract_save_reason_and_confidence(
            payload
        )
        surface_type = read_models.extract_surface_type(payload)
        visual_judgment = read_models.extract_visual_judgment(payload)
        recommendation_pitch = read_models.extract_recommendation_pitch(payload)

        source_run_id = read_models.candidate_recent_run_id(
            db_path, candidate_id=candidate_id
        )
        brief_role_title: str | None = None
        brief_linkedin_project: str | None = None
        source_run: LatestRunRef | None = None
        if source_run_id is not None:
            run_detail = read_models.run_by_id(db_path, run_id=source_run_id)
            if run_detail is not None:
                brief_role_title = _extract_brief_role_title(
                    run_detail.brief_snapshot_json
                )
                brief_linkedin_project = _extract_brief_linkedin_project(
                    run_detail.brief_snapshot_json
                )
            source_run = LatestRunRef(
                source=source,  # type: ignore[arg-type]
                state_key=state_dir.name,
                run_id=source_run_id,
            )

        is_failed_state = (
            record.current_lifecycle_state or ""
        ).startswith("failed_")

        # D6: read persisted HITL feedback from annotations.sqlite3.
        excluded_asset_urls: list[str] = []
        principle_feedback_markers: list[dict[str, Any]] = []
        annotations_db = state_dir / "annotations.sqlite3"
        if annotations_db.exists():
            try:
                from designer.recruiter_annotations import (
                    ExcludedAssetStore,
                    PrincipleFeedbackStore,
                )

                exc_store = ExcludedAssetStore(annotations_db)
                exclusions = exc_store.active_exclusions_for_candidate(
                    record.identity_key
                )
                excluded_asset_urls = [e.asset_url for e in exclusions]

                fb_store = PrincipleFeedbackStore(annotations_db)
                markers = fb_store.markers_for_candidate(record.identity_key)
                principle_feedback_markers = [
                    {
                        "marker_id": m.marker_id,
                        "candidate_identity_key": m.candidate_identity_key,
                        "principle_name": m.principle_name,
                        "marker": m.marker,
                        "note": m.note,
                        "marked_at": m.marked_at,
                    }
                    for m in markers
                ]
            except Exception:  # noqa: BLE001 — annotations DB may be corrupt
                pass

        return CandidateDetailResponse(
            slice="v0-shell-slice-c5",
            source=source,  # type: ignore[arg-type]
            brief_id=brief_id,
            candidate_id=record.candidate_id,
            identity_key=record.identity_key,
            display_name=record.display_name,
            profile_url=record.profile_url,
            terminal_decision=record.terminal_decision,
            confidence=confidence,
            save_reason=save_reason,
            current_lifecycle_state=record.current_lifecycle_state or None,
            first_seen_at=record.first_seen_at or None,
            last_seen_at=record.last_seen_at or None,
            source_run=source_run,
            brief_role_title=brief_role_title,
            brief_linkedin_project=brief_linkedin_project,
            notes=[
                CandidateNoteEntry(body=n.body, created_at=n.created_at)
                for n in record.notes
            ],
            user_status=record.user_status,
            is_failed_state=is_failed_state,
            judgment_accuracy=record.judgment_accuracy,
            judgment_accuracy_at=record.judgment_accuracy_at,
            surface_type=surface_type,
            visual_judgment=visual_judgment,
            recommendation_pitch=recommendation_pitch,
            excluded_asset_urls=excluded_asset_urls,
            principle_feedback_markers=principle_feedback_markers,
        )
    return None


def aggregate_workspace(
    *,
    brief_id: str,
    state_root: Path | None = None,
) -> WorkspaceResponse | None:
    """Return a :class:`WorkspaceResponse` keyed by brief_id.

    Phase C-bis 0.1: workspace is brief-first. The aggregator enumerates
    every state_dir whose latest run carries this brief_id (typically one
    today; possibly several in Phase F multi-module operation), queries
    each for SAVE-class candidates, and merges the results. Brief context
    (role title, linkedin project) is pulled from the most-recently-active
    state dir's run snapshot.

    Returns ``None`` when no state_dir matches — the route translates to
    a 404. A brief that has runs but no SAVE-class candidates yet returns
    a populated :class:`WorkspaceResponse` with an empty candidate list.

    ``saves_this_week`` is a 7-day rolling count from each candidate's
    ``last_seen_at`` (ISO-8601 with timezone). ``latest_run`` is the most
    recent run across all matched state_dirs, used by the "View latest
    run report" link.
    """

    matches = state_dirs_for_brief_id(brief_id, state_root=state_root)
    if not matches:
        return None

    one_week_ago = datetime.now(timezone.utc) - _ONE_WEEK
    # F6: carry state_key alongside each card so we can join into the
    # global identity DB's (source, state_key, candidate_id) primary key.
    # CandidateCardRecord doesn't expose state_key on the read model, so
    # the aggregator stamps it here from state_dir.name.
    all_cards: list[tuple[str, str, Any]] = []  # (source, state_key, card)
    sources: list[str] = []
    target_modules: list[str] = []
    brief_role_title: str | None = None
    brief_linkedin_project: str | None = None
    latest_run: LatestRunRef | None = None
    latest_run_started_at: str | None = None

    for source, state_dir in matches:
        db_path = state_dir / "runtime_state.sqlite3"
        latest_run_id = read_models.latest_run_in_state_dir(db_path)
        if latest_run_id is None:
            continue
        detail = read_models.run_by_id(db_path, run_id=latest_run_id)
        if detail is None or not detail.brief_id:
            continue

        # Track per-source contribution.
        if source not in sources:
            sources.append(source)

        # Brief metadata + most-recent run pointer come from whichever state
        # dir has the latest started_at — Phase F can refine this if a brief
        # spans modules with disagreeing started_at semantics.
        if (
            latest_run_started_at is None
            or (detail.started_at or "") > latest_run_started_at
        ):
            latest_run_started_at = detail.started_at or ""
            brief_role_title = _extract_brief_role_title(
                detail.brief_snapshot_json
            )
            brief_linkedin_project = _extract_brief_linkedin_project(
                detail.brief_snapshot_json
            )
            target_modules = _extract_brief_target_modules(
                detail.brief_snapshot_json
            )
            latest_run = LatestRunRef(
                source=source,  # type: ignore[arg-type]
                state_key=state_dir.name,
                run_id=latest_run_id,
            )

        cards = read_models.brief_saves(
            db_path, source=source, brief_id=brief_id, limit=200
        )
        for card in cards:
            all_cards.append((source, state_dir.name, card))

    # Sort by last_seen_at descending so the freshest saves render first
    # across all sources. Cap at 200 to keep the response bounded; Phase F
    # can introduce paging when multi-module aggregation pushes past this.
    all_cards.sort(
        key=lambda triple: triple[2].last_seen_at or "",
        reverse=True,
    )
    capped = all_cards[:200]

    saves_this_week = 0
    last_save_at: str | None = None
    for _src, _sk, card in capped:
        if card.last_seen_at:
            try:
                seen = datetime.fromisoformat(card.last_seen_at)
            except ValueError:
                seen = None
            if seen is not None:
                if seen.tzinfo is None:
                    seen = seen.replace(tzinfo=timezone.utc)
                if seen >= one_week_ago:
                    saves_this_week += 1
                if last_save_at is None or card.last_seen_at > last_save_at:
                    last_save_at = card.last_seen_at

    shortlisted_count = sum(
        1 for _src, _sk, c in capped if c.user_status == "shortlist"
    )

    # Phase F Slice F6: ensure F3's identity-resolver has run for this
    # brief, then build cross-source links per card. Skip this read-path
    # work for single-source workspaces: the response field is
    # cross_source_links, and those cannot exist without source diversity.
    # Resolution failures (e.g., identity DB unwritable in tests that
    # don't seed it) are non-fatal: cards still render, just without
    # cross-source pills.
    cross_links_by_pkey: dict[
        tuple[str, str, int], list[CrossSourceLink]
    ] = {}
    secondary_pkeys: set[tuple[str, str, int]] = set()
    has_cross_source_cards = len({src for src, _sk, _card in capped}) > 1
    if has_cross_source_cards:
        try:
            from shared.identity_resolution_service import (
                brief_persons_with_evidence,
                describe_merge_signal,
                resolve_persons_for_brief,
            )

            resolve_persons_for_brief(brief_id, state_root=state_root)
            # Reopen Stage 3a: accrue recruiter↔person sightings from the
            # persons the resolver just wrote. Idempotent (ledger-gated), so
            # re-resolution on every read doesn't inflate times_surfaced.
            from shared.runtime_state.recruiter_sighting import (
                record_sightings_for_brief,
            )

            record_sightings_for_brief(brief_id)
            persons = brief_persons_with_evidence(brief_id)
            capped_by_pkey = {
                (src, sk, card.candidate_id): card for src, sk, card in capped
            }
            for person in persons:
                if len(person.sources) < 2:
                    continue
                present_links = [
                    link
                    for link in person.sources
                    if (link.source, link.state_key, link.candidate_id)
                    in capped_by_pkey
                ]
                if len(present_links) < 2:
                    continue
                primary = _pick_primary_link(present_links, capped_by_pkey)
                primary_pkey = (
                    primary.source,
                    primary.state_key,
                    primary.candidate_id,
                )
                others: list[CrossSourceLink] = []
                for link in present_links:
                    pkey = (link.source, link.state_key, link.candidate_id)
                    if pkey == primary_pkey:
                        continue
                    secondary_pkeys.add(pkey)
                    card = capped_by_pkey[pkey]
                    others.append(
                        CrossSourceLink(
                            source=link.source,  # type: ignore[arg-type]
                            state_key=link.state_key,
                            candidate_id=link.candidate_id,
                            profile_url=card.profile_url,
                            display_name=card.display_name,
                            link_kind=link.link_kind,  # type: ignore[arg-type]
                            describe=describe_merge_signal(
                                link.link_kind, link.match_signal
                            ),
                        )
                    )
                cross_links_by_pkey[primary_pkey] = others
        except Exception as exc:
            log.debug(
                "Workspace identity cross-links skipped (%s): %s",
                type(exc).__name__,
                exc,
            )
            cross_links_by_pkey = {}
            secondary_pkeys = set()

    candidate_summaries: list[CandidateCardSummary] = []
    for src, sk, c in capped:
        pkey = (src, sk, c.candidate_id)
        if pkey in secondary_pkeys:
            # Folded into a primary card's cross_source_links — skip.
            continue
        cross_links = cross_links_by_pkey.get(pkey, [])
        candidate_summaries.append(
            CandidateCardSummary(
                candidate_id=c.candidate_id,
                source=src,  # type: ignore[arg-type]
                identity_key=c.identity_key,
                display_name=c.display_name,
                profile_url=c.profile_url,
                terminal_decision=c.terminal_decision,
                save_reason=c.save_reason,
                confidence=c.confidence,
                first_seen_at=c.first_seen_at or None,
                last_seen_at=c.last_seen_at or None,
                user_status=c.user_status,
                cross_source_links=cross_links,
            )
        )

    return WorkspaceResponse(
        slice="v0-shell-slice-c5",
        brief_id=brief_id,
        sources=sources,  # type: ignore[arg-type]
        target_modules=target_modules,
        brief_role_title=brief_role_title,
        brief_linkedin_project=brief_linkedin_project,
        latest_run=latest_run,
        total_saves=len(candidate_summaries),
        saves_this_week=saves_this_week,
        shortlisted_count=shortlisted_count,
        last_save_at=last_save_at,
        candidates=candidate_summaries,
    )


def _pick_primary_link(
    present_links: list[Any],
    capped_by_pkey: dict[tuple[str, str, int], Any],
) -> Any:
    """Pick the primary card representation for a multi-source person.

    Heuristic (load-bearing for F6 + F7 dedup behavior):
      1. Prefer LinkedIn (primary public-handle module).
      2. Then highest non-null confidence.
      3. Then most-recently-seen.
    """

    def sort_key(link: Any) -> tuple:
        pkey = (link.source, link.state_key, link.candidate_id)
        card = capped_by_pkey[pkey]
        confidence = float(card.confidence or 0.0)
        last_seen = str(card.last_seen_at or "")
        # LinkedIn ranks first via 0; everything else 1.
        source_rank = 0 if link.source == "linkedin" else 1
        return (source_rank, -confidence, last_seen)

    return sorted(present_links, key=sort_key)[0]


def _decision_counts_from_db(db_path: Path, *, run_id: int) -> DecisionCounts:
    """Count candidates by terminal decision for a given run.

    Uniqueness is enforced via DISTINCT — the same candidate can have
    multiple attempts in a run; we count each candidate once. Returns
    an empty DecisionCounts if the DB is missing/corrupt.
    """

    if not db_path.exists():
        return DecisionCounts()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return DecisionCounts()
    conn.row_factory = sqlite3.Row
    try:
        try:
            rows = conn.execute(
                "SELECT terminal_decision, COUNT(*) as n FROM ("
                "  SELECT DISTINCT c.id, c.terminal_decision "
                "  FROM candidates c "
                "  JOIN candidate_attempts ca ON ca.candidate_id = c.id "
                "  WHERE ca.run_id = ? AND c.terminal_decision IS NOT NULL"
                ") GROUP BY terminal_decision",
                (run_id,),
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return DecisionCounts()
    finally:
        conn.close()

    by_decision: dict[str, int] = {}
    total = 0
    for row in rows:
        decision = row["terminal_decision"]
        if not isinstance(decision, str):
            continue
        count = int(row["n"])
        by_decision[decision] = count
        total += count
    return DecisionCounts(total=total, by_decision=by_decision)
