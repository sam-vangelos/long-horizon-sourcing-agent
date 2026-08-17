"""Cloris HTTP response models.

Pydantic v2 models for the Cloris HTTP surface. Slice 2 introduced read-only
status aggregation (``GET /api/status``); Slice 3 added the launch payload
contract (``POST /api/launch/linkedin``). Slice 4 enriches ``StateDirEntry``
with worker-sidecar provenance plus a ``resumable`` hint, and adds
:class:`StopResponse` and :class:`ResumeResponse` for ``POST /api/stop/...``
and ``POST /api/resume/linkedin``. Slice 4 also introduces
``StopResponseState`` alongside ``WorkerState`` so the stop response can
describe the result of a stop action (``stopping`` / ``missing`` / ``stale``)
without conflating that with the steady-state aggregator enum
(``WorkerState``).

Models stay deliberately small and explicit: no validators, no derived
fields, no semantic shaping beyond the ``WorkerState`` and
``StopResponseState`` literals. Anything that requires further
interpretation (normalized stop reasons, queue semantics, pause state) is
out of scope.

Slice tag conventions:

- :class:`StatusResponse.slice` is bumped to ``"v0-shell-slice-4"`` because
  Slice 4 enriches the payload shape.
- :class:`LaunchResponse.slice` stays ``"v0-shell-slice-3"`` — the launch
  contract did not change.
- :class:`StopResponse.slice` and :class:`ResumeResponse.slice` are
  ``"v0-shell-slice-4"`` (new payloads, new slice).
- ``GET /healthz`` continues to advertise ``"v0-shell-slice-1"`` because it
  is a stable readiness probe, not a slice-version banner.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cloris.launchers import Source


WorkerState = Literal["missing", "alive", "alive_silent", "stale"]
StopResponseState = Literal["stopping", "missing", "stale"]
PipelineState = Literal["production", "partial", "stub"]
ProductLifecycle = Literal[
    "ready",
    "preparing",
    "strategizing",
    "searching",
    "reviewing",
    "writing_report",
    "finished",
    "recovering",
]
AttentionState = Literal[
    "live",
    "silent",
    "stalled",
    "terminal",
    "idle",
    "recovering",
]

# Phase 1C: brief-vs-state-dir taxonomy. Every state directory the
# aggregator discovers gets exactly one of these kinds:
#   - ``authored_brief`` — has a ``runs`` row that the user actually
#     authored (or a legacy run that pre-dates the intake-session FK,
#     treated as authored for backwards compatibility). The headline
#     count for the homescreen masthead derives from this kind.
#   - ``archived`` — the user has explicitly filed this brief away
#     (``runs.is_archived = 1``), or the reconciler has auto-archived
#     a long-stale orphan.
#   - ``intake_only`` — has an open intake-authoring session but no
#     completed run yet. Surfaced separately so the user can resume
#     authoring without losing track of in-progress drafts.
#   - ``orphaned_state_dir`` — a filesystem state directory with no
#     associated run history (an artifact left behind by a deleted
#     brief, an aborted CLI launch, or a manually-created folder).
#     These were the bulk of the meaningless "282 BRIEFS" header.
EntryKind = Literal[
    "authored_brief",
    "archived",
    "intake_only",
    "orphaned_state_dir",
]


class FailureKindCount(BaseModel):
    """Phase 4: one entry of an attempt-health failure-kind histogram.

    Mirrors :class:`shared.runtime_state.read_models.FailureKindCount`
    for the wire shape. The aggregator populates these from the read
    model so the UI can display "12 × http_429" style descriptions
    without duplicating SQL knowledge.
    """

    kind: str
    count: int


class AttemptHealthSummary(BaseModel):
    """Phase 4: attempt outcomes within the recent activity window.

    The aggregator computes this for the latest run of every state dir
    so the UI can surface stalled runs (alive worker, no recent
    success, retryable failures piling up — almost always provider
    degradation that the user needs to know about).
    """

    total_attempts_in_window: int = 0
    succeeded_in_window: int = 0
    failed_in_window: int = 0
    last_success_age_s: float | None = None
    recent_failures: list[FailureKindCount] = Field(default_factory=list)
    dominant_failure_kind: str | None = None


class WorkUnitProgressSummary(BaseModel):
    """Phase 4: queued / in_progress / done counts for the latest run.

    ``kind`` discriminates: ``"not_found"`` (no run / corrupt DB),
    ``"empty"`` (run exists but no work_units of the relevant kind),
    or ``"counts"`` (counts populated). The UI renders a "32 of 78"
    style progress fact only when ``kind == "counts"``.
    """

    kind: Literal["not_found", "empty", "counts"]
    queued: int = 0
    in_progress: int = 0
    done: int = 0
    skipped: int = 0
    error: int = 0


class RunSummary(BaseModel):
    """Verbatim subset of a ``runs`` row exposed by the status aggregator.

    All fields are optional because a state directory may be missing its
    ``runtime_state.sqlite3`` entirely, may have an empty ``runs`` table, or
    may have been written by an older schema variant. The aggregator never
    invents values; it only forwards what canonical SQLite returns.
    """

    id: int | None = None
    status: str | None = None
    stop_reason: str | None = None
    mode: str | None = None
    started_at: str | None = None
    ended_at: str | None = None


class StateDirEntry(BaseModel):
    """One entry per discovered ``output/state/<source>/<state_key>`` dir.

    The aggregator yields an entry for every state directory it discovers,
    including those without a ``runtime_state.sqlite3``. ``state_key`` is the
    on-disk directory name; ``brief_id_from_run`` is the ``runs.brief_id``
    column read from canonical SQLite (which can disagree with ``state_key``,
    especially for LinkedIn).

    Slice 4 adds worker-sidecar provenance fields. They all default in a way
    that preserves backward shape for callers that built ``StateDirEntry``
    manually before Slice 4: ``worker_state`` defaults to ``"missing"`` and
    every other worker_* field defaults to ``None`` / ``False``. The
    ``resumable`` field is ``None`` for GitHub entries (no analogous
    progress.json gate) and ``True``/``False``/``None`` for LinkedIn per
    :func:`cloris.control_plane.linkedin_resumable` semantics.
    """

    source: Source
    state_key: str
    # Phase 1B: state_dir was an absolute /Users/... path leaked into 282/282
    # entries on every status poll. Dropped — recruiter-facing surfaces never
    # need a filesystem path, and developer-facing surfaces (Reference Slip)
    # can compose `<source>/<state_key>` if they need an unambiguous handle.
    runtime_state_present: bool
    runtime_state_corrupt: bool = False
    latest_run: RunSummary | None = None
    brief_id_from_run: str | None = None
    brief_path_from_worker: str | None = None
    worker_json_present: bool = False
    worker_pid: int | None = None
    worker_alive: bool | None = None
    worker_mode: str | None = None
    worker_input_mode: str | None = None
    resumable: bool | None = None
    worker_state: WorkerState = "missing"
    # Phase 1.6: heartbeat_age_s is None when no sidecar / no heartbeat_at /
    # unparseable. >0 when alive but the most recent canonical write was
    # heartbeat_age_s seconds ago. The aggregator promotes worker_state to
    # "alive_silent" when this exceeds ALIVE_SILENT_THRESHOLD_S and the PID
    # is alive — typically meaning the machine slept, the worker is hung in
    # a non-cancellable section, or rate-limit retries have stalled work.
    heartbeat_age_s: float | None = None
    # Phase 3: brief identity. ``brief_role_title`` and
    # ``brief_linkedin_project`` are extracted from the snapshot stored in
    # ``runs.brief_snapshot_json`` so the UI can render the recruiter-meaningful
    # role title as the row heading instead of the directory slug
    # (state_key). All three default to None for legacy rows that pre-date
    # Phase 3 — the UI falls back to state_key in that case.
    # ``brief_drift_since_last_run`` is True when the on-disk brief at
    # ``runs.brief_path_at_launch`` no longer hashes to the stored
    # ``runs.brief_content_hash``; False when they match; None when the
    # comparison cannot be made (legacy row, file moved, etc.).
    brief_role_title: str | None = None
    brief_linkedin_project: str | None = None
    brief_drift_since_last_run: bool | None = None
    # Phase 4: status enrichment. The aggregator surfaces attempt-health
    # and work-unit-progress on every state dir entry so the UI can
    # render progress facts ("32 of 78 strings") and promote stalled
    # runs to the attention lane. ``run_stalled`` is True when the
    # worker is alive but recent failures cluster around retryable
    # HTTP-style errors with no recent success — almost always provider
    # degradation. ``stall_failure_kind`` carries the dominant kind
    # (e.g. "http_429") so the UI can render the operational reason
    # without consulting the histogram.
    attempt_health: AttemptHealthSummary | None = None
    work_unit_progress: WorkUnitProgressSummary | None = None
    run_stalled: bool = False
    stall_failure_kind: str | None = None
    # Northwind trial hardening: one recruiter-facing lifecycle. UI should prefer
    # this over inferring from raw worker/status fields.
    lifecycle: ProductLifecycle = "ready"
    # Runtime attention truth (domain Slice 5): derived from worker_state,
    # latest_run.status, run_stalled, and lifecycle. Prefer these over
    # re-deriving attention in the frontend.
    attention_state: AttentionState = "idle"
    live_signal_eligible: bool = False
    # Product lifecycle contract (launch/resume truth): recruiter-facing
    # active flag, terminal explanation, and projection drift.
    active: bool = False
    terminal_reason: str | None = None
    projection_disagreement: bool = False
    # Phase 1C: brief-taxonomy classifier. Computed by
    # :func:`cloris.control_plane._classify_entry`. Defaults to
    # ``orphaned_state_dir`` so a partially-constructed entry (e.g., legacy
    # test fixture) collapses to the safest bucket.
    kind: EntryKind = "orphaned_state_dir"


class RunDetail(BaseModel):
    """Phase B: the per-run report subject.

    Superset of :class:`RunSummary` adding identity columns
    (``brief_id``, ``output_dir``, ``brief_path_at_launch``,
    ``resumed_from_run_id``) and brief-derived fields the UI needs to
    render the role title and drift indicator. Wire shape mirrors
    :class:`shared.runtime_state.read_models.RunDetail`.
    """

    id: int
    source: Source
    brief_id: str | None = None
    # Phase 1B: output_dir dropped from the wire shape (R9 — no absolute
    # filesystem paths in API surfaces). It's still stored canonical-side
    # in `runs.output_dir` for forensic purposes; the wire payload
    # composes `<source>/<state_key>` when the frontend needs a handle.
    mode: str | None = None
    status: str | None = None
    stop_reason: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    resumed_from_run_id: int | None = None
    brief_role_title: str | None = None
    brief_linkedin_project: str | None = None
    brief_drift_since_run: bool | None = None


class CandidateDecisionSummary(BaseModel):
    """Phase B: one candidate row in the run-report decisions list.

    A "thin" view: name, profile URL, terminal decision, confidence.
    Evidence panels (snippet, profile summary, decision rationale,
    external evidence) are deferred to Phase C's Candidate Card —
    Phase B only needs to show *who* turned up and *what* the outcome
    was.
    """

    candidate_id: int
    display_name: str
    profile_url: str
    terminal_decision: str | None = None
    confidence: float | None = None


class DecisionCounts(BaseModel):
    """Phase B: counts of candidates by terminal decision in this run.

    The histogram complements the candidate list: even when the list is
    capped, the counts reflect the full population. Keys are the raw
    decision strings (``"SAVE"``, ``"REJECT"``, ``"FACIAL_NO"``, etc.);
    the UI maps these to product copy.
    """

    total: int = 0
    by_decision: dict[str, int] = Field(default_factory=dict)


class LaneMetricsSummary(BaseModel):
    """Sourcing-judgment kernel P5: per-lane metrics for one run.

    Mirrors :class:`shared.runtime_state.lane_metrics.LaneMetricsRow` on
    the wire. ``lane_id == "legacy"`` is the catch-all bucket for
    candidates whose canonical work unit / terminal payload carry no
    ``lane_id``; ``legacy`` is the boolean form of the same signal for
    consumers that prefer not to string-compare. ``cost_usd`` is ``None``
    when no canonical cost write has happened yet; future writers may
    sum into ``work_units.metrics_json`` without changing this shape.

    REVIEW outcomes (P4 ``REVIEW_INFERRED`` / ``REVIEW_FLAGGED``) are
    counted under ``review_count`` and broken out by reason code under
    ``review_by_reason``. They MUST NOT appear under ``save_count``;
    the read-model invariant is pinned in ``tests/test_lane_metrics.py``.
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
    review_by_reason: dict[str, int] = Field(default_factory=dict)
    work_unit_source_ids: list[str] = Field(default_factory=list)
    cost_usd: float | None = None
    legacy: bool = False


class RunReportResponse(BaseModel):
    """Top-level payload for ``GET /api/run/{source}/{state_key}/{run_id}``.

    Slice tag ``"v0-shell-slice-b1"`` (Phase B) — bumped from the
    status response slice to mark this as a new contract surface that
    can evolve independently of the aggregator.

    P5 adds ``lane_metrics`` as a default-empty list of
    :class:`LaneMetricsSummary`. Legacy consumers see ``[]``; new
    consumers iterate without a ``None`` check.
    """

    slice: Literal["v0-shell-slice-b1"] = Field(default="v0-shell-slice-b1")
    source: Source
    state_key: str
    # Phase 1B: state_dir dropped (R9 — no absolute paths on the wire).
    run: RunDetail
    work_unit_progress: WorkUnitProgressSummary
    attempt_health: AttemptHealthSummary
    decisions: DecisionCounts
    candidates: list[CandidateDecisionSummary] = Field(default_factory=list)
    candidates_truncated: bool = False
    lane_metrics: list[LaneMetricsSummary] = Field(default_factory=list)


class LatestRunRef(BaseModel):
    """Phase C-bis 0.1: a (source, state_key, run_id) triple suitable for
    constructing a ``#/run/<source>/<state_key>/<run_id>`` link.

    The run report stays per-source / per-state-dir even after the
    workspace and candidate-detail routes pivot to brief-first, so this
    triple is what the new responses carry to keep "View latest run report"
    and the candidate-detail back-link working.
    """

    source: Source
    state_key: str
    run_id: int


class CrossSourceLink(BaseModel):
    """Phase F Slice F6: one of the OTHER sources a person aggregates.

    Renders as a multi-source pill on a workspace card and as a row in
    candidate-detail's "Cross-source evidence" section. The describe
    field carries the editorial prose
    (`shared.identity_resolution_service.describe_merge_signal`) so
    the frontend never sees a raw `link_kind` enum.
    """

    source: Source
    state_key: str
    candidate_id: int
    profile_url: str
    display_name: str
    link_kind: Literal["auto_strong", "auto_medium", "manual"]
    describe: str


class IdentityCandidateLink(BaseModel):
    """Phase G Slice G2: one candidate row backing a person in the
    identity reconciliation surface.

    Wire-distinct from the F6 ``CrossSourceLink`` because the identity
    surface needs the source/state_key/candidate_id triple even for the
    primary link, plus the editorial ``describe`` prose for each row.
    The frontend never sees raw ``link_kind`` enums or confidence floats.
    """

    source: Source
    state_key: str
    candidate_id: int
    link_kind: Literal["auto_strong", "auto_medium", "manual"]
    recruiter_locked: bool
    describe: str


class IdentityPerson(BaseModel):
    """Phase G Slice G2: a canonical person from the global identity
    store, enriched with all candidate links visible under one brief.
    """

    person_id: int
    canonical_name: str
    canonical_handle: str
    sources: list[IdentityCandidateLink]


class IdentityPendingDecision(BaseModel):
    """Phase G Slice G2: one unresolved merge decision for a brief.

    Carries side-by-side person evidence and Cloris-voice
    ``signal_summary`` prose. Confidence floats are intentionally
    omitted — the editorial summary carries the meaning so the
    recruiter doesn't see raw probability output.
    """

    decision_id: int
    person_a: IdentityPerson
    person_b: IdentityPerson
    signal_summary: str
    created_at: str


class IdentityPendingResponse(BaseModel):
    """GET /api/brief/{brief_id}/identity/pending response shape."""

    slice: Literal["v0-identity-pending-1"] = "v0-identity-pending-1"
    brief_id: str
    persons_total: int
    decisions: list[IdentityPendingDecision]


class IdentityDecisionRequest(BaseModel):
    """POST /api/brief/{brief_id}/identity/decision request body."""

    model_config = ConfigDict(extra="forbid")

    decision_id: int
    choice: Literal["merge", "keep_separate"]


class IdentityUnlinkRequest(BaseModel):
    """POST /api/brief/{brief_id}/identity/unlink request body."""

    model_config = ConfigDict(extra="forbid")

    source: Source
    state_key: str
    candidate_id: int


# ---------------------------------------------------------------------------
# Phase G Slice G3: Live Monitor wire models. Operational register — raw
# enums + dense per-attempt rows are correct here, NOT editorial. Recruiters
# come to Monitor when they want depth Run Report deliberately doesn't show.
# ---------------------------------------------------------------------------


class ActiveRunSummary(BaseModel):
    """One row on the Live Monitor index — a run currently in motion.

    Carried fields are a thin recruiter-friendly subset of StateDirEntry so
    the index page can render brief title + source + run state without
    fetching the full status payload again.
    """

    source: Source
    state_key: str
    run_id: int | None
    run_status: str | None
    stop_reason: str | None
    started_at: str | None
    ended_at: str | None
    brief_id: str | None
    brief_role_title: str | None
    worker_pid: int | None
    lifecycle: ProductLifecycle = "ready"


class MonitorIndexResponse(BaseModel):
    """GET /api/monitor/index response shape."""

    slice: Literal["v0-monitor-index-1"] = "v0-monitor-index-1"
    active_runs: list[ActiveRunSummary]


class TelemetryAttemptRow(BaseModel):
    """One candidate_attempts row, raw enums preserved for operator view."""

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


class TelemetryEventRow(BaseModel):
    """One events row — raw event log."""

    id: int
    event_type: str
    candidate_id: int | None
    attempt_id: int | None
    payload_summary: str | None
    created_at: str


class RunTelemetryResponse(BaseModel):
    """GET /api/run/{source}/{state_key}/{run_id}/telemetry response shape.

    Bounded: at most 50 attempts + 30 events (most-recent first). The
    Monitor view is a window into the live run, not a full audit trail —
    the runtime_state DB itself holds everything; recruiters who need
    forensics use sqlite directly.
    """

    slice: Literal["v0-run-telemetry-1"] = "v0-run-telemetry-1"
    source: Source
    state_key: str
    run_id: int
    attempts: list[TelemetryAttemptRow]
    events: list[TelemetryEventRow]
    last_event_at: str | None
    attempts_total: int
    events_total: int


# ---------------------------------------------------------------------------
# Live run signal. Recruiter-readable, read-only activity summary for the
# homescreen. This is deliberately higher-level than Monitor telemetry.
# ---------------------------------------------------------------------------


RunSignalPhase = Literal[
    "idle",
    "starting",
    "preparing",
    "strategizing",
    "strategy_ready",
    "searching",
    "reviewing",
    "adapting",
    "writing_report",
    "working",
    "completed",
    "finished",
    "recovering",
]


class RunSignalStringPreview(BaseModel):
    """One planned search string, compacted for the homescreen signal panel."""

    id: int | None = None
    label: str
    rationale: str | None = None
    boolean: str | None = None
    domain_lane: str | None = None
    novelty_bucket: str | None = None


class RunSignalEvent(BaseModel):
    """One recruiter-readable recent activity line."""

    kind: str
    label: str
    detail: str | None = None
    timestamp: str | None = None


class RunSignalResponse(BaseModel):
    """GET /api/run-signal/{source}/{state_key} response shape.

    Reads projections/artifacts (`live-console.log`, `run_log.jsonl`,
    `execution_plan.json`) only. It is not canonical control state and must
    not be used for launch/stop decisions.
    """

    slice: Literal["v0-run-signal-1"] = "v0-run-signal-1"
    source: Source
    state_key: str
    active: bool
    phase: RunSignalPhase
    lifecycle: ProductLifecycle = "ready"
    headline: str
    detail: str | None = None
    strategy_rationale: str | None = None
    strategy_architecture: str | None = None
    strategy_architecture_rationale: str | None = None
    generated_string_count: int | None = None
    coverage_gap_count: int | None = None
    strategy_strings: list[RunSignalStringPreview] = Field(default_factory=list)
    recent_events: list[RunSignalEvent] = Field(default_factory=list)
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# Phase G Slice G4: Tools index + execution wire models.
# ---------------------------------------------------------------------------


class ToolEntry(BaseModel):
    """One tool in the catalog. ``schema_fields`` is a hand-crafted
    summary of the args schema (field name + type label) so the frontend
    can render a form without reflecting Pydantic at runtime.
    """

    tool_id: str
    tier: Literal["A", "B", "C"]
    label: str
    pitch: str
    cli_command: str
    execution_model: Literal["sync", "async", "cli_only"]
    schema_fields: list[dict] = Field(default_factory=list)


class ToolsIndexResponse(BaseModel):
    slice: Literal["v0-tools-index-1"] = "v0-tools-index-1"
    tools: list[ToolEntry]


class ToolRunRequest(BaseModel):
    """POST /api/tools/{tool_id} body — args validated per-tool."""

    model_config = ConfigDict(extra="forbid")

    args: dict = Field(default_factory=dict)


class ToolRunSyncWire(BaseModel):
    slice: Literal["v0-tool-sync-1"] = "v0-tool-sync-1"
    tool_id: str
    exit_code: int
    stdout_tail: str
    stderr_tail: str


class ToolRunAsyncWire(BaseModel):
    slice: Literal["v0-tool-async-1"] = "v0-tool-async-1"
    tool_id: str
    job_id: str


class ToolJobStatusWire(BaseModel):
    slice: Literal["v0-tool-job-1"] = "v0-tool-job-1"
    job_id: str
    tool_id: str
    status: Literal["queued", "running", "succeeded", "failed", "purged"]
    started_at: float
    finished_at: float | None
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    error_message: str | None


# ---------------------------------------------------------------------------
# Phase G Slice G5: Settings transparency surface. Read-only — Cloris's
# operational config is hard-coded by deliberate engineering decision; this
# surface lets the recruiter see what's set without being able to break
# anything. Credentials surface as ✓/✗ booleans, NEVER values.
# ---------------------------------------------------------------------------


class SettingsCredential(BaseModel):
    key: str
    label: str
    present: bool
    pitch: str  # editorial Cloris-voice line


class SettingsBriefSaveSummary(BaseModel):
    brief_id: str
    role_title: str | None
    target_modules: list[str]
    linkedin_project_id: str | None


class SettingsGovernorLimit(BaseModel):
    name: str
    label: str
    value: int | str
    explainer: str  # Cloris-voice explanation of why this isn't editable


class SettingsResponse(BaseModel):
    slice: Literal["v0-settings-1"] = "v0-settings-1"
    credentials: list[SettingsCredential]
    save_destinations: list[SettingsBriefSaveSummary]
    governor: list[SettingsGovernorLimit]
    cdp_url: str


class CandidateCardSummary(BaseModel):
    """Phase C, slice C2 (extended in C4 + C-bis 0.1 + F6): one card on the Workspace surface.

    Trimmed view: name, profile URL, save reason, decision, confidence,
    timestamps. The full :class:`CandidateDetailResponse` is fetched
    when the user clicks through. Unlike
    :class:`CandidateDecisionSummary` (run-report scoped), this row is
    brief-scoped — saves across all runs of the same brief.

    C4 extension: ``user_status`` surfaces the recruiter override.
    C-bis 0.1: ``source`` is part of the wire shape so the workspace
    grid can render source-provenance on each card.
    F6: ``cross_source_links`` lists the OTHER (source, candidate_id)
    pairs aggregated under the same canonical person. Empty for a
    person observed on a single source.
    """

    candidate_id: int
    source: Source
    identity_key: str
    display_name: str
    profile_url: str
    terminal_decision: str
    save_reason: str | None = None
    confidence: float | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    user_status: str | None = None
    cross_source_links: list[CrossSourceLink] = Field(default_factory=list)


class WorkspaceResponse(BaseModel):
    """Phase C-bis 0.1: per-brief workspace payload, brief-first.

    Wire shape for ``GET /api/workspace/{brief_id}``. Aggregates every
    SAVE-class candidate for the brief across every state_dir whose
    latest run carries that brief_id — typically one source today, but
    Phase F multi-module operation will fan out across LinkedIn +
    GitHub + Researcher etc.

    Recipe-card stats (``total_saves``, ``saves_this_week``,
    ``shortlisted_count``, ``last_save_at``) roll up across all matched
    state_dirs. ``sources`` lists which source modules contributed.
    ``latest_run`` is the most recent run across all matches, used by
    "View latest run report".

    Slice tag ``v0-shell-slice-c5`` marks the brief-first contract
    revision; the prior c4 source-siloed shape is gone.
    """

    slice: Literal["v0-shell-slice-c5"] = Field(default="v0-shell-slice-c5")
    brief_id: str
    sources: list[Source] = Field(default_factory=list)
    target_modules: list[str] = Field(default_factory=list)
    brief_role_title: str | None = None
    brief_linkedin_project: str | None = None
    latest_run: LatestRunRef | None = None
    total_saves: int = 0
    saves_this_week: int = 0
    shortlisted_count: int = 0
    last_save_at: str | None = None
    candidates: list[CandidateCardSummary] = Field(default_factory=list)


# Executive Search Slice 7 (saves-shape alarm threshold). The
# recruiter-facing banner threshold per the spec's Risks section
# ("This run produced more than 25 saves — that's a high-volume
# pattern, not an exec search"). Distinct from Slice 5's cost cap
# (fires on cost) and Slice 5's eval-count alarm (fires on
# evaluations); this fires on saves. Centralized so the shortlist
# API and the frontend agree on the same number.
EXEC_SEARCH_SAVES_SHAPE_THRESHOLD: int = 25


class ShortlistResponse(BaseModel):
    """Executive Search Slice 7: shortlist surface read shape.

    Wire shape for ``GET /api/shortlist/{brief_id}``. Slice 7 ships a
    read-side projection on top of the existing per-source candidates
    tables (the Cloris-native `shortlist_entries` table + the
    AbstractSaveDestination it would write through depend on
    multi-module-foundation Slices 6-7, which are NOT shipped — so
    Slice 7's scope is downgraded to a read view per the spec's
    "downgrade or absorb" rule).

    Recruiter signals surfaced beyond the workspace shape:

    - ``saves_shape_alarm`` — true when ``len(candidates) >
      EXEC_SEARCH_SAVES_SHAPE_THRESHOLD``. The frontend renders an
      editorial banner ("This run produced more than 25 saves —
      that's a high-volume pattern, not an exec search. The brief
      calibration may be too broad. [Review brief criteria]").
    - ``saves_shape_alarm_threshold`` — the threshold value, surfaced
      so the banner copy reads "more than N saves" without the
      frontend hard-coding the number.

    Slice 7b (the actual save destination + write path) waits for
    multi-module-foundation Slices 6-7 to ship.
    """

    slice: Literal["v0-shell-slice-c5"] = Field(default="v0-shell-slice-c5")
    brief_id: str
    sources: list[
        Source
    ] = Field(default_factory=list)
    brief_role_title: str | None = None
    brief_linkedin_project: str | None = None
    latest_run: LatestRunRef | None = None
    total_saves: int = 0
    saves_this_week: int = 0
    last_save_at: str | None = None
    candidates: list[CandidateCardSummary] = Field(default_factory=list)
    saves_shape_alarm: bool = False
    saves_shape_alarm_threshold: int = EXEC_SEARCH_SAVES_SHAPE_THRESHOLD


class CandidateNoteEntry(BaseModel):
    """Phase C, slice C3: one recruiter-authored note on a candidate."""

    body: str
    created_at: str


class CandidateDetailResponse(BaseModel):
    """Phase C-bis 0.1: candidate-detail surface payload, brief-first.

    Wire shape for ``GET /api/candidate/{brief_id}/{candidate_id}``. The
    URL no longer carries ``source`` or ``state_key`` — both are present
    in the response for rendering (source eyebrow, run back-link) but
    they're metadata, not identity.

    ``source_run`` bundles ``(source, state_key, run_id)`` so the
    candidate-detail page can construct ``#/run/<source>/<state_key>/<run_id>``
    for the back-link without round-tripping through the URL params.

    Slice tag ``v0-shell-slice-c5`` marks the brief-first contract
    revision; the prior c3 source-siloed shape is gone.
    """

    slice: Literal["v0-shell-slice-c5"] = Field(default="v0-shell-slice-c5")
    source: Source
    brief_id: str
    candidate_id: int
    identity_key: str
    display_name: str
    profile_url: str
    terminal_decision: str | None = None
    confidence: float | None = None
    save_reason: str | None = None
    current_lifecycle_state: str | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    # Brief context from the most recent run that touched this candidate.
    source_run: LatestRunRef | None = None
    brief_role_title: str | None = None
    brief_linkedin_project: str | None = None
    # C3: recruiter-authored fields.
    notes: list[CandidateNoteEntry] = Field(default_factory=list)
    user_status: str | None = None
    # Phase C-bis 0.3: when the candidate's lifecycle state is in the
    # failed_* family, the workspace already filters it out — but a
    # recruiter who lands on the candidate detail directly (deep link)
    # needs the page to disable pipeline-action affordances (status
    # toggle, notes compose) until the next run resolves the failure.
    is_failed_state: bool = False
    # Phase C-bis 0.5: closed-loop feedback substrate. Distinct from
    # ``user_status`` — this is the recruiter's calibration signal on
    # whether Cloris's *judgment* was right, not a pipeline action.
    # Both fields ride on the wire so the future Next Run Learning
    # surface can render the calibration history without an extra
    # round-trip. NULL = no signal yet.
    judgment_accuracy: str | None = None
    judgment_accuracy_at: str | None = None
    # Phase F Slice F6: per-candidate cross-source links. Empty when
    # this person was only observed on a single source. Populated when
    # F3's identity resolver merged this candidate with one or more
    # candidates from other sources for the same brief. The
    # candidate-detail page renders these as a "Cross-source evidence"
    # section.
    cross_source_links: list[CrossSourceLink] = Field(default_factory=list)
    # Designer go-live D1: surface_type discriminates rendering branches
    # in CandidateDetail.svelte. "hitl_visual_review" for Designer saves;
    # None for legacy/non-Designer candidates.
    surface_type: str | None = None
    # Designer go-live D1: structured visual-judgment payload. Present
    # when surface_type == "hitl_visual_review". Shape varies by module;
    # the frontend does its own type narrowing via VisualJudgmentPayload.
    visual_judgment: dict[str, Any] | None = None
    # Designer go-live D5b: recruiter-facing recommendation pitch.
    # Present on non-REJECT Designer candidates. Deterministic assembly
    # from full_decision + visual_judgment evidence.
    recommendation_pitch: dict[str, Any] | None = None
    # Designer D6: persisted HITL feedback state. Populated from
    # annotations.sqlite3 so the UI can initialize toggles on load.
    excluded_asset_urls: list[str] = Field(default_factory=list)
    principle_feedback_markers: list[dict[str, Any]] = Field(default_factory=list)
    # Reopen Stage 3 (R3, the "remembers" half): cross-brief presence for
    # this person, fed by the recruiter spine's accretion log
    # (``recruiter_candidate_history``). Present ONLY when this person has
    # been seen in at least one OTHER brief — the inline "you've seen them
    # before" marker. None when first-seen-here (nothing to render) or when
    # the recruiter spine can't resolve (fail-soft; the candidate page never
    # 500s on this enrichment). Deliberately verdict-FREE (invariant D-B):
    # a count + where, NEVER a flattened "you liked them".
    cross_brief_presence: "CrossBriefPresence | None" = None


class CrossBriefPresence(BaseModel):
    """Reopen Stage 3 (R3): one person's cross-brief recurrence, for the
    inline candidate-detail memory marker.

    A thin recruiter-readable projection of one ``recruiter_candidate_history``
    row — the accretion log the spine bumps each time a person is sighted on a
    brief. ``other_briefs_count`` is ``times_surfaced`` minus the current brief
    (the honest "elsewhere" number); the field is only attached to a candidate
    when this is >= 1, so the marker means "seen in N OTHER briefs", never
    "seen here, count 1".

    Verdict-free by contract (invariant D-B): ``last_lifecycle_state`` is the
    pipeline state Cloris last observed (e.g. ``screened``), NOT the recruiter's
    judgment. The marker shows *where a person keeps showing up*, not a call on
    them — the per-(person, role) verdict view is a separate, later surface.
    """

    times_surfaced: int
    other_briefs_count: int
    first_seen_brief: str
    last_seen_brief: str
    last_lifecycle_state: str


class LegacyResolveResponse(BaseModel):
    """Phase C-bis 0.1: response for the legacy URL resolver endpoints.

    The frontend hits these when it parses an old-shape hash like
    ``#/workspace/<source>/<state_key>`` so it can rewrite to the
    brief-first equivalent. Same shape works for both workspace and
    candidate redirects — the candidate_id stays unchanged across the
    rewrite, so only ``brief_id`` is needed.
    """

    brief_id: str


class CandidateNoteRequest(BaseModel):
    """Phase C, slice C3: request body for ``POST /api/candidate/.../note``."""

    model_config = ConfigDict(extra="forbid")
    body: str


class CandidateStatusPatchRequest(BaseModel):
    """Phase C, slice C3: request body for ``PATCH /api/candidate/...``.

    A ``user_status`` of ``None`` clears the recruiter override and
    falls back to Cloris's terminal_decision. Non-empty strings set the
    override; the API layer validates the allowed set.
    """

    model_config = ConfigDict(extra="forbid")
    user_status: str | None = None


class CandidateJudgmentAccuracyPatchRequest(BaseModel):
    """Phase C-bis 0.5: request body for the judgment-accuracy PATCH.

    ``judgment_accuracy`` is the recruiter's calibration signal on
    Cloris's terminal decision — explicitly distinct from
    ``user_status`` (a pipeline action). ``None`` clears the signal.
    Non-null values must be in the allowed set; the API layer
    validates and returns 422 on unknown.
    """

    model_config = ConfigDict(extra="forbid")
    judgment_accuracy: str | None = None


class PrincipleFeedbackRequest(BaseModel):
    """Designer D6: per-principle feedback from the recruiter."""

    model_config = ConfigDict(extra="forbid")
    principle_name: str
    marker: str
    note: str = ""


class ExcludeAssetRequest(BaseModel):
    """Designer D6: recruiter excludes an asset from visual judgment."""

    model_config = ConfigDict(extra="forbid")
    asset_url: str
    reason: str = ""


class RevokeExcludedAssetRequest(BaseModel):
    """Designer D6: recruiter un-excludes a previously excluded asset."""

    model_config = ConfigDict(extra="forbid")
    asset_url: str


class BriefCounts(BaseModel):
    """Phase 1C: roll-up of authored briefs by activity bucket.

    Replaces the meaningless 282 / 170 / 112 ribbon ("BRIEFS / ACTIVE /
    PAUSED") on the homescreen masthead — those numbers counted state
    directories on disk, of which 275/282 had no run history at all.
    These counts answer recruiter-facing questions:

      - ``active``: how many briefs are *live* — currently running, paused
        on a limit, or interrupted but resumable. The user might want
        to look at any of them today.
      - ``working``: how many of those are progressing right now (worker
        alive, run status='running'). Subset of ``active``.
      - ``paused``: those that hit a limit or were interrupted and now
        wait on the user to nudge them. Subset of ``active``.
      - ``finished``: completed cleanly; effectively done.
      - ``lost``: abandoned (zombie reconciled) or errored out. Need
        attention but not progressing.
      - ``archived``: explicitly filed away by the user (or auto-archived).
      - ``orphaned``: filesystem state directories with no run history,
        kept around for diagnostic purposes but not surfaced as briefs.
    """

    active: int = 0
    working: int = 0
    paused: int = 0
    finished: int = 0
    lost: int = 0
    archived: int = 0
    orphaned: int = 0


class BriefStatusGroup(BaseModel):
    """Phase F Slice F7: home + filed group state-dir entries by brief_id.

    A single brief can spawn N state dirs (one per discovery module).
    Pre-F7, each state dir rendered as its own card on home + filed
    surfaces — duplicate visual entries for one logical brief. F7
    groups by ``brief_id`` so the recruiter sees ONE card per brief
    with a multi-module status indicator inside.

    ``brief_id`` is None for orphaned state dirs (no run, no
    ``brief_id_from_run``) so they still surface in the response as a
    diagnostic group rather than being silently dropped.
    """

    brief_id: str | None = None
    brief_role_title: str | None = None
    brief_linkedin_project: str | None = None
    modules: list[StateDirEntry] = Field(default_factory=list)


class ModuleStatus(BaseModel):
    """One registered discovery module and its product maturity state."""

    source: Source
    pipeline_state: PipelineState
    launchable: bool
    visible: bool
    # Reopen P7.1 (spec §8): mirrors ``cloris.launchers.LauncherEntry.sunset``.
    # ``True`` means the module was administratively retired by product
    # decision (paused, not broken) — distinct from ``pipeline_state``,
    # which describes build maturity. The frontend renders sunset modules
    # with "Paused for now" instead of the generic unavailable/disabled
    # copy. Defaults False so payloads built without reading the registry
    # (tests, older callers) collapse to the non-sunset case.
    sunset: bool = False


class StatusResponse(BaseModel):
    """Top-level payload for ``GET /api/status``.

    ``slice`` is pinned to ``"v0-shell-slice-4"`` so callers can detect the
    contract version without re-typing the literal at every construction
    site. The bump from ``"v0-shell-slice-2"`` to ``"v0-shell-slice-4"``
    reflects the enriched :class:`StateDirEntry` shape Slice 4 ships.

    Phase 1C adds ``counts`` — a roll-up of the brief-taxonomy buckets so
    the masthead can render recruiter-facing numbers without computing them
    client-side from a partially-classified list.

    Phase F Slice F7 adds ``briefs`` — the same state-dir entries
    grouped by ``brief_id`` so the home + filed surfaces can render one
    card per brief instead of one per (brief × module). ``entries``
    stays for backward compat: existing callers that read it keep
    working. The frontend home + filed surfaces consume ``briefs``.
    """

    slice: Literal["v0-shell-slice-4"] = Field(default="v0-shell-slice-4")
    entries: list[StateDirEntry]
    counts: BriefCounts = Field(default_factory=BriefCounts)
    briefs: list[BriefStatusGroup] = Field(default_factory=list)
    trial_mode: bool = False
    modules: list[ModuleStatus] = Field(default_factory=list)


class LaunchResponse(BaseModel):
    """Response payload for ``POST /api/launch/{source}`` (Phase F Slice F1)
    and the legacy ``POST /api/launch/linkedin`` synonym.

    ``slice`` stays at ``"v0-shell-slice-3"`` — F1 widens ``source`` from
    ``Literal["linkedin"]`` to ``Source`` (additive)
    and adds ``mode`` so resume launches can carry their truth without
    needing the separate :class:`ResumeResponse`. The slice tag is the
    skew-detection signal for *shape* breaks; widening a Literal is
    additive, so the tag does not bump.

    ``pid`` is the spawned worker process's PID at the moment ``Popen``
    returns; after ``cloris.worker`` ``execvp``s into the per-source
    orchestrator, the same PID belongs to the orchestrator process, so
    this value remains the truthful process handle for later
    stop/probe operations.
    """

    slice: Literal["v0-shell-slice-3"] = Field(default="v0-shell-slice-3")
    source: Source
    input_mode: Literal["concurrent"]
    mode: Literal["fresh", "resume"] = Field(default="fresh")
    pid: int
    state_dir: str
    worker_json_path: str


class StopResponse(BaseModel):
    """Response payload for ``POST /api/stop/{source}/{state_key}``.

    The HTTP status code conveys the action: 202 when SIGTERM was
    dispatched against an alive PID, 200 when there was nothing to signal
    (``worker_state`` ∈ ``{"missing", "stale"}``). The body conveys the
    truth — clients should branch on ``worker_state``, not on the status
    code, because the body is the durable contract.

    ``worker_state`` here is ``StopResponseState`` (``stopping`` /
    ``missing`` / ``stale``), intentionally distinct from
    :class:`StatusResponse`'s ``WorkerState`` because the stop response
    describes the **result of the stop action**, not a steady-state
    observation.
    """

    slice: Literal["v0-shell-slice-4"] = Field(default="v0-shell-slice-4")
    source: Source
    state_key: str
    state_dir: str
    worker_state: StopResponseState
    pid: int | None = None


class ResumeResponse(BaseModel):
    """Response payload for the legacy ``POST /api/resume/linkedin`` synonym.

    Phase F Slice F1 collapses launch + resume into a single endpoint
    (``POST /api/launch/{source}`` with ``mode="resume"``) and returns
    :class:`LaunchResponse`. This shape stays around for backward compat
    with clients still hitting the old route. The ``source`` Literal
    widens from ``Literal["linkedin"]`` to
    ``Source`` so the legacy resume route can
    redirect transparently to the F1 path without re-shaping the wire.
    """

    slice: Literal["v0-shell-slice-4"] = Field(default="v0-shell-slice-4")
    source: Source
    mode: Literal["resume"] = Field(default="resume")
    input_mode: Literal["concurrent"]
    pid: int
    state_dir: str
    worker_json_path: str


class LaunchRequest(BaseModel):
    """Request body for the generic ``POST /api/launch/{source}`` endpoint.

    Phase F Slice F1. The recruiter picks a brief by id (the same hash
    runtime_state tracks); the body carries no path. The dispatch
    layer in :mod:`cloris.api` resolves the brief_id to disk via
    :func:`_resolve_brief_by_id` and dispatches to the per-source
    spawn function via :data:`cloris.launchers.LAUNCHERS`.

    ``mode`` is strict ``Literal["fresh", "resume"]`` so an unrecognized
    value (e.g., a future ``"refresh"`` mode added by Phase E) is
    caught at the request boundary, forcing an explicit slice bump.

    ``force`` skips the launch-readiness probe (Phase D Slice D9). It
    does NOT skip ``BriefPathNotFoundError``,
    ``WorkerAlreadyRunningError``, ``NoPendingWorkError``, or the
    fresh-over-resumable-state guard — those are pre-flight integrity
    checks, not soft readiness signals. The recruiter mental model is
    "I know auth is iffy but try anyway."

    ``force_fresh`` is narrower: it is explicit consent to launch in
    fresh mode even when the state dir already carries resumable
    generated artifacts. It does not skip readiness probes.
    """

    model_config = ConfigDict(extra="forbid")
    brief_id: str
    mode: Literal["fresh", "resume"] = "fresh"
    force: bool = False
    force_fresh: bool = False


class LaunchMultiRequest(BaseModel):
    """Request body for ``POST /api/launch/multi`` (audit Move #8).

    Atomic multi-module launch: the recruiter picks a brief + a list
    of modules; the API spawns one worker per module server-side
    rather than the frontend looping over the per-source endpoint.
    Closes the browser-crash-mid-launch failure mode where a half-
    started run had no backend resume affordance.

    ``modules`` is the list of sources to launch. Order is taken
    as the dispatch order — the frontend's chief-of-staff dispatch
    UI passes modules in the recommended order, and the API spawns
    them sequentially in the same order so per-module spawn errors
    surface in dispatch-plan order.

    ``mode``, ``force``, and ``force_fresh`` apply uniformly across
    every module — matching :class:`LaunchRequest`'s semantics.
    """

    model_config = ConfigDict(extra="forbid")
    brief_id: str
    modules: list[
        Source
    ] = Field(default_factory=list)
    mode: Literal["fresh", "resume"] = "fresh"
    force: bool = False
    force_fresh: bool = False


class LaunchMultiPerSourceError(BaseModel):
    """Per-module spawn error envelope inside :class:`LaunchMultiResponse`.

    ``kind`` is the typed error class name as the API layer maps it
    today — e.g., ``"launch_not_ready"``, ``"worker_already_running"``,
    ``"brief_path_not_found"``. The frontend renders each kind with
    the same prose it uses for the per-source endpoint's error
    responses; this preserves the recruiter's existing mental model
    when one module readiness-blocks while another spawns.
    """

    kind: str
    detail: dict


class LaunchMultiPerSourceResult(BaseModel):
    """One module's outcome inside :class:`LaunchMultiResponse`.

    Either ``launch`` is set (spawn succeeded) OR ``error`` is set
    (typed error from the spawn path). The frontend branches on
    presence — never both populated, never neither.
    """

    source: Source
    launch: LaunchResponse | None = None
    error: LaunchMultiPerSourceError | None = None


class LaunchMultiResponse(BaseModel):
    """Response payload for ``POST /api/launch/multi`` (audit Move #8).

    Order of ``results`` mirrors the request's ``modules`` order so
    the frontend renders status in dispatch-plan order. ``slice``
    follows the v0-shell-* convention used by sibling launch
    payloads.

    HTTP status semantics:
    - 201 — at least one module spawned successfully (partial-
      failure rendering is the frontend's job).
    - 422 — every module's spawn raised a typed error; the response
      body's ``results`` carries each error envelope.
    """

    slice: Literal["v0-launch-multi-1"] = Field(default="v0-launch-multi-1")
    brief_id: str
    results: list[LaunchMultiPerSourceResult]


class BriefInfo(BaseModel):
    """One authored brief discovered under ``config/``.

    Two consumers, two surface use-cases:
    - **LaunchForm BriefPicker** (Phase 4) — pick by role to launch. Uses
      ``path``, ``role_title``, ``linkedin_project*``, ``modified_at``.
    - **Phase D Slice D1 — Brief library at ``#/briefs``** — list every
      authored brief with run metadata. Uses the picker fields PLUS the
      ``brief_id`` + ``last_run_*`` + ``total_*`` fields below.

    The library-only fields are nullable / default 0 so picker callers
    don't break — the same model serves both surfaces additively. The
    aggregator in :mod:`cloris.control_plane.aggregate_briefs` populates
    the library fields by walking ``state_dirs_for_brief_id()`` for each
    authored brief; the picker endpoint can skip the run-metadata
    decoration if it doesn't need it (cheaper).
    """

    path: str
    role_title: str | None = None
    linkedin_project: str | None = None
    linkedin_project_id: str | None = None
    modified_at: str  # ISO-8601 UTC; "" if stat failed

    # Phase D Slice D1: library-only run metadata. Nullable so picker
    # callers see the same shape they always have. ``brief_id`` is the
    # ``derive_brief_id(path)`` hash that ``runs.brief_id`` carries.
    brief_id: str | None = None
    last_run_id: int | None = None
    last_run_at: str | None = None
    last_run_status: str | None = None
    last_run_source: Source | None = None
    total_runs: int = 0
    total_saves: int = 0

    # Executive Search Slice 6: brief confidentiality posture, propagated
    # from the V2 brief (or "open" default). The aggregator
    # (:func:`cloris.control_plane.aggregate_briefs`) routes this through
    # :mod:`shared.confidentiality` to mask titles + redact save counts
    # for ``"blind"`` briefs in cross-brief surfaces. The frontend reads
    # this verbatim so it can render the confidentiality pill on the
    # brief library row without re-parsing.
    confidentiality_class: Literal["open", "referenceable", "blind"] = "open"

    # Phase F Slice F5: which discovery modules this brief targets.
    # Legacy briefs without this key default to ["linkedin"] at the
    # frontend (mirrors BriefDetail.svelte's destinationModules helper);
    # the backend sends None so the picker can distinguish "not set"
    # from "explicitly empty list" (which would mean "ask the recruiter
    # to pick at launch time").
    target_modules: list[str] | None = None


class BriefsListResponse(BaseModel):
    """Response payload for ``GET /api/briefs``.

    Sorted most-recently-modified first so the picker leads with what
    the user has been working on lately.
    """

    slice: Literal["v0-briefs-list-1"] = Field(default="v0-briefs-list-1")
    briefs: list[BriefInfo]


class BriefDetailResponse(BaseModel):
    """Response payload for ``GET /api/brief/{brief_id}``. Phase D Slice D2.

    Mirrors the partition that
    :class:`shared.brief_v2_schema.MergedBrief` already returns. The
    architectural-fit critique caught: if we only send ``deprecated_keys``
    (names) without ``preserved_legacy`` (values), the frontend can't
    render the legacy values in the deprecation drawer, AND the PUT
    handler would have to re-read disk to know what to keep. By sending
    the full partition over the wire, PUT becomes pure: rebuild =
    ``v2_data ∪ (preserved_legacy − dropped_legacy_keys)``.

    ``v2_data`` is intentionally typed as ``dict[str, Any]`` (not a
    Pydantic sub-model). The V2 schema has 60+ optional sub-shapes
    (capability_areas, depth_distinction, non_fit_patterns,
    facial_calibration, …); typing each is premature for D2. The frontend
    treats it as opaque + renders the fields it knows. The envelope
    stays Pydantic-typed for stable client expectations.

    ``was_flat`` tells the frontend the brief is currently a flat
    ``config/<name>.json`` and will migrate to nested
    ``config/<name>/brief.json`` on first edit (Fork C).
    """

    slice: Literal["v0-brief-detail-1"] = Field(default="v0-brief-detail-1")
    brief_id: str
    path: str
    role_title: str | None = None
    v2_data: dict[str, Any]
    preserved_legacy: dict[str, Any]
    deprecated_keys: list[str]
    unknown_keys: list[str]
    last_modified: str
    version_count: int = 0
    was_flat: bool = False


class BriefVersionEntry(BaseModel):
    """One row in the version history list. Phase D Slice D5."""

    version_id: str  # filename stem, e.g. "2026-05-01T16-32-12.567+00-00"
    created_at: str  # ISO-8601 (decoded from filename)
    size_bytes: int


class BriefVersionsResponse(BaseModel):
    """Response for ``GET /api/brief/{brief_id}/versions``. Phase D Slice D5."""

    slice: Literal["v0-brief-versions-1"] = Field(default="v0-brief-versions-1")
    brief_id: str
    versions: list[BriefVersionEntry]


class ConversationCitationDebug(BaseModel):
    """Structured pointer for companion debug mode (never default UI)."""

    source: str
    state_key: str
    signal_ref: str


class ConversationQueryRequest(BaseModel):
    """Body for ``POST /api/conversation/{brief_id}/query``."""

    message: str = Field(..., min_length=1, max_length=12000)


class ConversationQueryResponse(BaseModel):
    """Wire shape for recruiter chat replies."""

    slice: Literal["v0-conversation-query-1"] = Field(
        default="v0-conversation-query-1",
    )
    assistant_text: str
    kind: Literal["ok", "degraded"]
    degraded_reason: str | None = None
    citations_debug: list[ConversationCitationDebug] | None = None


class ConversationMuteRequest(BaseModel):
    """Ambient narration mute toggle (persisted server-side)."""

    ambient_muted: bool


class ConversationMuteResponse(BaseModel):
    """Echo stored mute preference."""

    slice: Literal["v0-conversation-mute-1"] = Field(
        default="v0-conversation-mute-1",
    )
    brief_id: str
    ambient_muted: bool


# Phase E Slice E1: Market viewer wire shapes. Distinct from
# `market_intelligence.schema.MarketIntelArtifact` so the on-disk
# artifact format can evolve without breaking the wire (and so the
# detail payload can flatten + trim the artifact to recruiter-facing
# fields rather than shipping the full 60-lane raw JSON).


class MarketSummary(BaseModel):
    """One row in the market viewer's catalog list."""

    market_key: str
    role_title: str
    role_level: str
    geography: str
    brief_ids_seen: list[str] = Field(default_factory=list)
    last_updated_at: str
    run_count: int
    saved_count: int
    aggregate_save_rate: float | None = None


class MarketsListResponse(BaseModel):
    """Response payload for ``GET /api/markets``. Most-recently-updated
    first so the list reads as "what Cloris has been studying lately."""

    slice: Literal["v0-markets-list-1"] = Field(default="v0-markets-list-1")
    markets: list[MarketSummary]


class MarketLane(BaseModel):
    """One lane row on the market detail page.

    Trimmed shape — ``why_it_works`` + ``recommended_action`` carry the
    recruiter-readable prose; metrics roll up the per-lane volumes.
    Drops the supporting_run_refs + dominant_anchors machinery (that
    detail belongs in a future Reference Slip / Cloris-internal view).
    """

    lane_key: str
    domain_lane: str
    novelty_bucket: str
    status: str
    candidates_seen: int
    saves: int
    save_rate: float | None = None
    why_it_works: str | None = None
    recommended_action: str | None = None


class MarketTalentPool(BaseModel):
    """One talent-pool row on the market detail page."""

    pool_key: str
    label: str
    signal_strength: str
    status: str
    evidence_summary: str | None = None


class ExternalContextClaim(BaseModel):
    """One Perplexity-sourced claim in the market thesis."""

    claim: str
    label: str = ""
    evidence_refs: list[str] = []
    confidence: float = 0.0


class MarketThesis(BaseModel):
    """The "Cloris's read" stanza copy."""

    summary: str
    supply_assessment: str = ""
    competition_assessment: str = ""
    external_context: list[ExternalContextClaim] = []


class MarketDetailResponse(BaseModel):
    """Response payload for ``GET /api/market/{market_key}``."""

    slice: Literal["v0-market-detail-1"] = Field(default="v0-market-detail-1")
    market_key: str
    role_title: str
    role_level: str
    geography: str
    brief_ids_seen: list[str] = Field(default_factory=list)
    last_updated_at: str
    run_count: int
    candidates_seen: int = 0
    saved_count: int
    rejected_count: int = 0
    aggregate_save_rate: float | None = None
    facial_yes_rate: float | None = None
    lanes: list[MarketLane] = Field(default_factory=list)
    talent_pools: list[MarketTalentPool] = Field(default_factory=list)
    market_thesis: MarketThesis
    # Engine's structured brief-edit proposals from the market intel
    # artifact. The frontend's computeBriefDiff() walks this as a fourth
    # source alongside lanes/thesis/talent_pools so RefreshBrief surfaces
    # the wider field set (additional_search_terms,
    # employer_signal_rules, search_priorities, instructions, notes)
    # the same way Reflection's HunkCard surface does. Default-empty
    # so any consumer built before the field was added handles it as
    # "no recommendations" — forward-compatible.
    brief_recommendations: list[dict] = Field(default_factory=list)


class BriefEditRequest(BaseModel):
    """Request body for ``PUT /api/brief/{brief_id}``. Phase D Slice D2.

    The frontend rebuilds the full brief client-side and ships:
    - ``v2_data`` — the V2-schema fields the recruiter edited (or kept).
    - ``preserved_legacy`` — every legacy/unknown key the recruiter
      decided to keep. The backend writes ``v2_data ∪ preserved_legacy``
      verbatim. Anything not in either dict is dropped.
    - ``dropped_legacy_keys`` — optional, audit-only list of which
      deprecated/unknown keys the recruiter dropped this edit. Captured
      for telemetry; the actual dropping is just "key absent from
      ``preserved_legacy``."
    """

    model_config = ConfigDict(extra="forbid")
    v2_data: dict[str, Any]
    preserved_legacy: dict[str, Any] = Field(default_factory=dict)
    dropped_legacy_keys: list[str] = Field(default_factory=list)
    last_modified: str | None = Field(
        default=None,
        description=(
            "Echo of last_modified from the preceding GET /api/brief/{id}. "
            "When present, the server checks the file's current mtime and "
            "returns 409 if they differ (stale edit). None skips the check."
        ),
    )


class LaunchReadinessBlocker(BaseModel):
    """One reason a launch isn't ready. Phase D Slice D9 (Ledger L4).

    Each blocker carries an editorial remediation string the LaunchForm
    renders as italic prose — never a red form-error chip. The ``kind``
    discriminator lets the frontend tint each remediation per category
    (auth = peach-deep, config = blue-pencil, net = wood) without a
    per-source switch.
    """

    kind: Literal["auth", "config", "net"]
    message: str
    remediation: str
    code: str = ""


class LaunchReadinessResponse(BaseModel):
    """Response payload for ``GET /api/launch-readiness/{source}/{brief_id}``.

    Phase D Slice D9 (Ledger L4). ``ready`` is true iff ``blockers`` is
    empty. The ``source`` and ``brief_id`` echo the request so the
    frontend can verify it got an answer for what it asked. The
    ``brief_id`` slot is captured but not used by Phase D's checks
    (which are all source-level: browser session, token scope). Phase
    F's per-brief save-destination check will start consulting it.

    Slice tag ``v0-launch-readiness-1`` is the wire-shape version. It
    bumps when the response gains new fields (e.g. Phase F's brief-
    specific blockers), not when the underlying probe logic shifts.
    """

    slice: Literal["v0-launch-readiness-1"] = Field(default="v0-launch-readiness-1")
    source: Source
    brief_id: str
    ready: bool
    blockers: list[LaunchReadinessBlocker]


class OnboardingStatusResponse(BaseModel):
    """Response payload for ``GET /api/onboarding/status``.

    Phase 0 ``apikey-ui`` slice. The frontend gate in App.svelte
    fetches this on mount and, when ``welcome_complete`` is false,
    renders Welcome.svelte instead of the route table. The other
    fields let the welcome surface reflect partial progress (key
    entered but acknowledgment missing, or vice versa) without
    losing what the recipient already typed.

    ``env_path`` and ``acknowledgment_path`` are surfaced to the wire
    so the welcome copy can name the exact paths Cloris will write to
    on the recipient's Mac. This is the relational-disclosure layer
    the IT-defensibility README leans on.
    """

    slice: Literal["v0-onboarding-status-1"] = Field(
        default="v0-onboarding-status-1"
    )
    welcome_complete: bool
    anthropic_present: bool
    acknowledged: bool
    acknowledged_at: str | None
    env_path: str
    acknowledgment_path: str


class CredentialUpsertRequest(BaseModel):
    """Request body for ``POST /api/onboarding/credential``.

    Phase 0 ``apikey-ui`` slice. ``key`` mirrors the wire-side
    credential key surface in ``cloris.api._CREDENTIAL_LABELS``
    (``anthropic_api_key`` / ``openai_api_key`` / ``google_api_key``
    / ``perplexity_api_key``). ``value`` is the raw secret; the
    backend chmod 600's the .env file on every write.

    ``extra="forbid"`` rejects unknown fields at the request boundary
    so a typo in the welcome / Settings UI surfaces clearly.
    """

    model_config = ConfigDict(extra="forbid")
    key: str
    value: str


class AcknowledgmentRequest(BaseModel):
    """Request body for ``POST /api/onboarding/acknowledge``.

    Phase 0 ``apikey-ui`` / ``disclosure`` slices. ``acknowledged``
    must be ``true`` — the welcome surface only sends this on the
    explicit checkbox-checked + Continue flow. Sending ``false`` is
    rejected at the route layer.
    """

    model_config = ConfigDict(extra="forbid")
    acknowledged: bool


class ChromeStatusResponse(BaseModel):
    """Response payload for ``GET /api/chrome-status`` and
    ``POST /api/chrome-relaunch`` / ``POST /api/chrome-open-linkedin``.

    Phase 0 ``chrome-launcher`` slice. Wire shape for the welcome
    surface's polling loop and the Settings "Re-open Chrome" control.
    Mirrors :class:`cloris.chrome_launcher.ChromeStatus` exactly; the
    Pydantic shell is just for FastAPI response validation.

    ``state`` values:

    - ``healthy`` — CDP is reachable; the recipient can sign into
      LinkedIn Recruiter and start a search.
    - ``spawning`` — Cloris kicked Chrome off; not healthy yet. The
      welcome surface should keep polling.
    - ``unhealthy`` — Chrome isn't up and Cloris hasn't relaunched
      it. The welcome surface offers the "Re-open Chrome" control.
    - ``missing_chrome`` — Google Chrome isn't installed at the
      expected path. ``message`` carries the install URL prompt.
    - ``unsupported_platform`` — non-macOS host. The .app is
      macOS-only; the wire response makes the failure mode explicit.

    ``message`` is a recruiter-readable one-sentence summary that the
    welcome surface can render verbatim.
    """

    slice: Literal["v0-chrome-status-1"] = Field(default="v0-chrome-status-1")
    state: Literal[
        "healthy",
        "spawning",
        "unhealthy",
        "missing_chrome",
        "unsupported_platform",
    ]
    cdp_url: str
    profile_dir: str
    message: str


class AnthropicHealthResponse(BaseModel):
    """Cached Anthropic readiness probe used by launch pre-flight."""

    slice: Literal["v0-anthropic-health-1"] = "v0-anthropic-health-1"
    state: Literal["healthy", "missing", "unhealthy", "api_budget_exhausted"]
    message: str
    checked_at: str | None = None
    cache_age_s: float | None = None


class ReconciledRun(BaseModel):
    """One run that the reconciler marked abandoned.

    Returned in :class:`ReconcileResponse`. Frontend uses these for a
    one-time "Cloris noticed N runs lost track" toast on app startup so
    the user has forensic context for status pills that just changed.
    """

    source: Source
    state_key: str
    run_id: int
    new_status: str
    stop_reason: str
    reason: str  # missing_sidecar / bad_sidecar / pid_dead


class ReconcileResponse(BaseModel):
    """Response payload for ``POST /api/reconcile``.

    Returned synchronously after the reconciler walks every state dir,
    emits mutations for runs whose worker process is gone, and applies
    them through the canonical write path. ``applied`` is the count
    actually written; ``mutations`` are the per-run records for the
    UI to surface (and for tests to inspect).
    """

    slice: Literal["v0-reconciler-slice-1"] = Field(default="v0-reconciler-slice-1")
    applied: int
    mutations: list[ReconciledRun]


# --- Onboarding flow intake sessions (Northwind trial plan, Slice 1B) ---
#
# Authoring state for the brief-authoring conversation. Per the plan's A1
# decision, intake sessions live in a dedicated SQLite table that colocates
# with the runtime-state DB but is distinct from run-lifecycle state. The
# slice tag ``"v0-onboarding-slice-1"`` is new (no relation to the
# v0-shell-slice tags) so onboarding-flow versioning evolves independently
# of the launch/status surface.


class FilingReadinessWire(BaseModel):
    """Server-evaluated filing gate for intake session GET responses."""

    can_file: bool
    blocking_codes: list[str] = Field(default_factory=list)
    valid_v2_draft: bool = False
    missing_keys: list[str] = Field(default_factory=list)
    invalid_keys: list[str] = Field(default_factory=list)
    in_flight_synthesis: bool = False
    in_flight_compose: bool = False
    insight_deficits: list[dict[str, str]] = Field(default_factory=list)


class IntakeSession(BaseModel):
    """One brief-authoring conversation, resumable across reloads.

    ``current_step`` is the literal step name used by the onboarding flow
    state machine. The full set of legal values:

    ``"welcome"``, ``"role_basics"``, ``"role_framing"``,
    ``"good_looks_like"``, ``"lookalikes"``, ``"exemplars"``,
    ``"search_stance"``, ``"anything_else"``, ``"synthesis"``, ``"review"``,
    ``"completed"``.

    Slice 1B does not enforce these as a Literal because the synthesis
    endpoint (Slice 5) may want to introduce new transitional steps without
    a wire-shape break; the column is a free-form ``TEXT`` and the model
    surface accepts any string. ``state_json`` is the parsed dict (the DB
    column is TEXT and stores JSON).
    """

    id: int
    brief_id_draft: str | None
    role_title: str | None
    current_step: str
    state_json: dict
    started_at: str
    updated_at: str
    completed_at: str | None
    archived_at: str | None
    filing_readiness: FilingReadinessWire | None = None


class IntakeSessionCreateRequest(BaseModel):
    """Request body for ``POST /api/intake/sessions``.

    ``role_title`` is an optional initial hint surfaced by the welcome
    step; everything else is set server-side at creation time.
    """

    model_config = ConfigDict(extra="forbid")
    role_title: str | None = None


class IntakeSessionPatchRequest(BaseModel):
    """Request body for ``PATCH /api/intake/sessions/{session_id}``.

    All fields optional — the patch endpoint is a partial update. Sending
    no body is a no-op patch that still bumps ``updated_at`` so the UI's
    last-seen timestamp ticks even when only a heartbeat-style ping is
    issued.
    """

    model_config = ConfigDict(extra="forbid")
    current_step: str | None = None
    state_json: dict | None = None
    role_title: str | None = None


class IntakeSourcePacketRequest(BaseModel):
    """Request body for source-packet brief synthesis."""

    model_config = ConfigDict(extra="forbid")
    job_description_text: str = ""
    intake_notes_text: str = ""
    geography: str | None = None


class IntakeGapAnswerRequest(BaseModel):
    """Natural-language answer to one or more Cloris gap questions."""

    model_config = ConfigDict(extra="forbid")
    answer_text: str = Field(..., min_length=1)
    answered_question_ids: list[str] = Field(default_factory=list)


class IntakeCritiqueRequest(BaseModel):
    """Natural-language critique of the drafted brief read-back."""

    model_config = ConfigDict(extra="forbid")
    critique_text: str = Field(..., min_length=1)
    modality: Literal["text", "voice"] = "text"


class IntakeCritiqueCommitRequest(BaseModel):
    """Approve a subset of pending critique edits."""

    model_config = ConfigDict(extra="forbid")
    approved_edit_indices: list[int] | None = None
    newly_affirmed_fields: list[str] = Field(default_factory=list)
    released_locks: list[str] = Field(default_factory=list)


class RecruiterPreferencesRequest(BaseModel):
    """Patch recruiter-level brief-authoring preferences."""

    model_config = ConfigDict(extra="forbid")
    summary: str | None = None
    active_voice: bool | None = None
    summaries: dict[str, str] | None = None


class RecruiterPreferencesResponse(BaseModel):
    """Recruiter-level correction memory and preference response."""

    slice: Literal["v0-preferences-1"] = "v0-preferences-1"
    preferences: dict[str, Any]


class RecruiterPresenceEntry(BaseModel):
    """One person's cross-brief PRESENCE for a recruiter.

    Reopen Stage 3 (R3.0). Sourced from the ``recruiter_candidate_history``
    ACCRETION log — count + first/last brief — NOT the flattened current-state
    authority. Deliberately carries no terminal_decision / verdict: presence is
    "this person has shown up N times across these briefs and was last in this
    lifecycle state," not "this is the recruiter's resolved call on them." The
    resolved-call surface (recruiter_candidates) is a separate, later read.
    """

    person_id: int
    times_surfaced: int
    first_seen_brief: str
    last_seen_brief: str
    last_lifecycle_state: str


class RecruiterCalibrationEntry(BaseModel):
    """One brief's calibration-marker rollup for a recruiter (R3.2).

    Reopen Stage 3: R3.0 shipped the placeholder shape (``domain`` / ``drift``)
    so the panel could render empty; R3.2 fills ``calibration_drift`` with one
    entry per brief the recruiter owns. The real signal is the
    ``judgment_accuracy`` marker rollup merged across every state dir a brief
    spans (LinkedIn + GitHub today; more under Phase F multi-module), mirroring
    ``cloris.control_plane.aggregate_workspace``'s per-dir merge. Reading a
    single ``db_path`` would under-read a multi-source brief, so the producer
    fans ``state_dirs_for_brief_id`` and sums.

    The original ``domain`` / ``drift`` fields are retained (additive widening
    so no existing constructor or wire consumer breaks) but carry placeholder
    defaults — the load-bearing fields are ``brief_id``, ``total_markers``, and
    the per-marker / per-area breakdowns. ``domain`` defaults to the brief_id
    and ``drift`` to the high-confidence weighted ``wrong`` + ``off_rubric``
    share so the legacy two-field reader still sees a coherent (brief, signal)
    pair rather than a constant.
    """

    brief_id: str = ""
    total_markers: int = 0
    by_marker_value: dict[str, int] = Field(default_factory=dict)
    by_capability_area: dict[str, int] = Field(default_factory=dict)
    weighted_markers_by_area: dict[str, int] = Field(default_factory=dict)
    source_state_dirs: int = 0
    domain: str = ""
    drift: float = 0.0


class RecruiterReflectionEntry(BaseModel):
    """One brief's active reflection session for a recruiter (R3.3).

    Reopen Stage 3: R3.0 shipped the placeholder shape (``reflection_id`` /
    ``summary`` / ``created_at``) so the panel could render empty; R3.3 fills
    ``reflection_trail`` with one entry per brief that has an *active*
    reflection — ``completed_at IS NULL AND discarded_at IS NULL`` — read from
    the SINGLE intake DB (``resolve_intake_db_path``) where reflection rows
    actually live. Reflection sessions are authored before any (source,
    state_key) commitment, so a per-state-dir read would hit an always-empty
    table and ship a dead panel; the producer reads the intake DB read-only.

    ``reflection_id`` / ``summary`` / ``created_at`` are retained (additive)
    and back-filled from the real row: ``reflection_id`` is the session id,
    ``created_at`` is ``started_at``, ``summary`` a short ``brief_id @ phase``
    string. The structured fields ``brief_id`` / ``current_phase`` /
    ``started_at`` / ``updated_at`` / ``steering_iterations`` are the real
    surface.
    """

    reflection_id: int = 0
    brief_id: str = ""
    current_phase: str = ""
    started_at: str = ""
    updated_at: str = ""
    steering_iterations: int = 0
    summary: str = ""
    created_at: str = ""


class RecruiterDashboardResponse(BaseModel):
    """Cross-brief recruiter dashboard surface (Reopen Stage 3, R3.1).

    ``presence`` is built from the recruiter_candidate_history accretion log.
    ``calibration_drift`` and ``reflection_trail`` are forward markers — shape
    present, content ``[]`` — filled by R3.2 / R3.3 respectively.
    """

    slice: Literal["v0-recruiter-slice-1"] = "v0-recruiter-slice-1"
    recruiter_id: int
    recruiter_handle: str
    presence: list[RecruiterPresenceEntry] = Field(default_factory=list)
    calibration_drift: list[RecruiterCalibrationEntry] = Field(
        default_factory=list
    )
    reflection_trail: list[RecruiterReflectionEntry] = Field(
        default_factory=list
    )


class IntakeSessionListResponse(BaseModel):
    """Response payload for ``GET /api/intake/sessions``."""

    slice: Literal["v0-onboarding-slice-1"] = Field(
        default="v0-onboarding-slice-1"
    )
    sessions: list[IntakeSession]


class IntakeSessionResponse(BaseModel):
    """Response payload for ``POST/GET/PATCH /api/intake/sessions[...]``."""

    slice: Literal["v0-onboarding-slice-1"] = Field(
        default="v0-onboarding-slice-1"
    )
    session: IntakeSession


ComposeJobStatus = Literal["idle", "composing", "ready", "failed"]


class ComposeJobResult(BaseModel):
    """Composition outcome when ``conversation_compose.status`` is ``ready``."""

    model_config = ConfigDict(extra="forbid")
    compose_status: Literal["composed", "deficits"]
    deficits: list[dict[str, str]] = Field(default_factory=list)
    missing_keys: list[str] = Field(default_factory=list)
    invalid_keys: list[str] = Field(default_factory=list)
    insight_deficits: list[dict[str, str]] = Field(default_factory=list)


class ConversationComposeJob(BaseModel):
    """``state_json.conversation_compose`` job block exposed on the wire."""

    model_config = ConfigDict(extra="forbid")
    status: ComposeJobStatus = "idle"
    revision: int = 0
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    result: ComposeJobResult | None = None


class IntakeComposeJobResponse(BaseModel):
    """Response for compose job routes (``POST .../compose_jobs``, etc.)."""

    slice: Literal["v0-intake-compose-job-1"] = Field(
        default="v0-intake-compose-job-1"
    )
    session: IntakeSession
    job: ConversationComposeJob


class IntakeComposeFromConversationResponse(BaseModel):
    """Legacy sync compose response — superseded by :class:`IntakeComposeJobResponse`.

    Breaking change (Slice 1): ``POST .../compose_from_conversation`` now
    schedules :class:`ConversationComposeJob` work and returns
    :class:`IntakeComposeJobResponse` instead of this shape. Kept for
    type references in older tests/docs; do not use for new routes.
    """

    slice: Literal["v0-intake-compose-from-conversation-1"] = Field(
        default="v0-intake-compose-from-conversation-1"
    )
    session: IntakeSession
    status: Literal["composed", "deficits"]
    deficits: list[dict[str, str]] = Field(default_factory=list)
    missing_keys: list[str] = Field(default_factory=list)
    invalid_keys: list[str] = Field(default_factory=list)
    # ``insight_deficits`` is parallel to ``deficits`` / ``missing_keys`` /
    # ``invalid_keys``: the brief schema can be valid while the
    # hiring-manager picture is still missing. Surfaced separately so the
    # frontend CTA recovery path can distinguish "the brief itself is
    # incomplete" from "the brief is complete but the picture isn't yet."
    insight_deficits: list[dict[str, str]] = Field(default_factory=list)


class IntakeSessionDeleteResponse(BaseModel):
    """Response payload for ``DELETE /api/intake/sessions/{session_id}``."""

    slice: Literal["v0-onboarding-slice-1"] = Field(
        default="v0-onboarding-slice-1"
    )
    deleted: bool
    id: int


class IntakeSessionCompleteResponse(BaseModel):
    """Response payload for ``POST /api/intake/sessions/{session_id}/complete``.

    Phase D Slice D3. The intake wizard's terminal call writes the
    drafted V2 brief to disk and stamps the session as completed; this
    response carries both the freshly-completed session and the new
    ``brief_id`` / ``brief_path`` so the wizard can navigate directly
    to ``#/brief/<brief_id>`` without a second round-trip.
    """

    slice: Literal["v0-onboarding-slice-1"] = Field(
        default="v0-onboarding-slice-1"
    )
    session: IntakeSession
    brief_id: str
    brief_path: str


# --- The Reflection — HITL market intelligence flow ---
#
# Wire models for the two-gate HITL flow that pauses around the market
# intelligence engine. Slice tag ``"v0-reflection-slice-1"`` is new and
# evolves independently of the onboarding/intake versioning.
#
# Wire-shape note: the ``state_json`` and ``hunks`` payloads carry the
# engine's structured outputs (planner result, editorial briefing,
# proposed hunks). The Pydantic surface keeps them as opaque dicts/lists
# at the BaseModel boundary; the engine module owns their schema.
# Keeping the wire types loose is deliberate — if a follow-up enriches
# the hunk shape, the API doesn't need a model bump.


class ReflectionSession(BaseModel):
    """One reflection session, persisted across the full HITL flow.

    ``current_phase`` walks: ``planning`` → ``plan_approved`` →
    ``researching`` → ``awaiting_diff`` → ``committed`` (terminal) |
    ``discarded`` (terminal).

    ``state_json`` is the engine's phase-output bag: keys
    ``phase_outputs.plan``, ``phase_outputs.research``,
    ``phase_outputs.propose``, plus a top-level ``context`` block and
    ``steering_history`` list. The frontend reads structured sub-keys
    off the dict; the wire type is permissive on purpose.

    ``research_error`` is non-null when the research phase hit a fatal
    error (Perplexity timeout, network blip). The frontend surfaces it
    as an editorial recovery prompt; the user can re-trigger research
    or proceed to propose using internal evidence only.
    """

    id: int
    brief_id: str
    source_run_id: int | None
    current_phase: str
    state_json: dict
    steering_iterations: int
    started_at: str
    updated_at: str
    completed_at: str | None
    discarded_at: str | None
    brief_version_committed: str | None
    research_error: str | None


class ReflectionCreateRequest(BaseModel):
    """Request body for ``POST /api/reflection/sessions``.

    ``brief_id`` identifies the brief the recruiter wants Cloris to
    reflect on. ``source_run_id`` optionally biases the reflection
    toward a specific run (otherwise the planner uses whatever
    evidence it can find for the brief). ``run_dir`` is an explicit
    override for the snapshot directory; in practice the API resolves
    it from ``source_run_id`` when not provided.
    """

    model_config = ConfigDict(extra="forbid")
    brief_id: str
    source_run_id: int | None = None
    run_dir: str | None = None


class ReflectionSteeringRequest(BaseModel):
    """Request body for ``PATCH /api/reflection/sessions/{id}/steering``.

    ``note`` is the recruiter's natural-language steering input.
    Empty strings degenerate to a no-op (the API echoes the existing
    state without bumping the iteration counter). The 3-iteration cap
    is enforced server-side; over the cap the call returns 409.
    """

    model_config = ConfigDict(extra="forbid")
    note: str


class ReflectionStartResearchRequest(BaseModel):
    """Request body for ``POST /api/reflection/sessions/{id}/start_research``.

    No fields today — Gate 1 approval is just a state transition. The
    request body exists so the endpoint can grow extension fields
    (e.g., ``with_external_research`` override) without a wire break.
    """

    model_config = ConfigDict(extra="forbid")


class ReflectionCommitRequest(BaseModel):
    """Request body for ``POST /api/reflection/sessions/{id}/commit``.

    ``accepted_hunk_ids`` is the list of hunk ids the recruiter
    approved at Gate 2. ``edited_hunks`` is an optional map of
    ``hunk_id -> {"after": "<edited value>"}`` for hunks the recruiter
    edited inline. Hunks not in ``accepted_hunk_ids`` are dropped on
    the floor; the brief is committed with only accepted (and possibly
    edited) hunks applied.
    """

    model_config = ConfigDict(extra="forbid")
    accepted_hunk_ids: list[str]
    edited_hunks: dict[str, dict] | None = None


class ReflectionDiscardRequest(BaseModel):
    """Request body for ``POST /api/reflection/sessions/{id}/discard``.

    No fields. Discarding is unconditional — the recruiter has decided
    to walk away from this reflection entirely. The brief stays
    untouched.
    """

    model_config = ConfigDict(extra="forbid")


class ReflectionResponse(BaseModel):
    """Response payload for most reflection endpoints (POST/GET/PATCH)."""

    slice: Literal["v0-reflection-slice-1"] = Field(
        default="v0-reflection-slice-1"
    )
    session: ReflectionSession


class ReflectionCommitResponse(BaseModel):
    """Response payload for ``POST /api/reflection/sessions/{id}/commit``.

    Carries the freshly-committed session plus the new brief version
    path so the frontend can surface it (and link to the brief detail
    surface for inspection).
    """

    slice: Literal["v0-reflection-slice-1"] = Field(
        default="v0-reflection-slice-1"
    )
    session: ReflectionSession
    brief_version_path: str
    applied_hunks: list[dict]


class ReflectionActiveResponse(BaseModel):
    """Response payload for ``GET /api/reflection/sessions/active?brief_id=...``.

    Returns the active (non-terminal) reflection session for a brief,
    or ``session=None`` when there isn't one. Used by the workspace
    surface to decide whether to render the "review what Cloris read"
    pickup card.
    """

    slice: Literal["v0-reflection-slice-1"] = Field(
        default="v0-reflection-slice-1"
    )
    session: ReflectionSession | None


# --- Multi-agent orchestrator (Phase 2.7 heuristic dispatch) ---------


class OrchestratorDispatchStepWire(BaseModel):
    """One dispatch step surfaced to the frontend."""

    model_config = ConfigDict(extra="forbid")

    module_name: str
    handoff_condition: str | None = None


class OrchestratorDispatchPlanWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[OrchestratorDispatchStepWire]


class OrchestratorDecideRequest(BaseModel):
    """POST ``/api/orchestrator/decide`` — chief-of-staff dispatch preview."""

    model_config = ConfigDict(extra="forbid")

    brief_path: str | None = None
    brief_id: str | None = None
    partial_brief: dict[str, Any] | None = None


class OrchestratorDecideResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice: Literal["orchestrator-decide-v1"] = "orchestrator-decide-v1"
    dispatch_plan: OrchestratorDispatchPlanWire
    order_context: str = Field(
        default="",
        description=(
            "Short editor-facing explanation of why this module order was "
            "recommended (chief of staff vs. brief list, optional gates)."
        ),
    )


class OrchestratorSynthesizeRequest(BaseModel):
    """POST ``/api/orchestrator/synthesize`` — re-run chief-of-staff synthesis."""

    model_config = ConfigDict(extra="forbid")

    reflection_session_id: int


class OrchestratorSynthesizeResponse(BaseModel):
    """Wire shape aligned with :meth:`ChiefOfStaffSynthesis.to_dict`."""

    model_config = ConfigDict(extra="forbid")

    slice: Literal["orchestrator-synthesize-v1"] = "orchestrator-synthesize-v1"
    paragraph: str
    per_specialist_weight: dict[str, dict[str, Any]]
    priority_for_principal: str
    confidence: float
    source: str


class OrchestratorRunRecord(BaseModel):
    """One ``chief_of_staff_runs`` row for orchestrator history."""

    model_config = ConfigDict(extra="forbid")

    id: int
    brief_id: str
    principal_id: str
    status: str
    dispatch_plan: dict[str, Any]
    invocation_order: list[str]
    handoff_payloads: dict[str, Any]
    synthesis_output: dict[str, Any]
    started_at: str
    ended_at: str | None


class OrchestratorRunsResponse(BaseModel):
    """GET ``/api/orchestrator/{brief_id}/runs`` — dispatch / CoS history."""

    model_config = ConfigDict(extra="forbid")

    slice: Literal["orchestrator-runs-v1"] = "orchestrator-runs-v1"
    runs: list[OrchestratorRunRecord]
