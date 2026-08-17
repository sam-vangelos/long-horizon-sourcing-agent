"""Cloris HTTP surface (FastAPI router).

Canonical reads enumerate state dirs, open each ``runtime_state.sqlite3`` via
:mod:`shared.runtime_state.read_models` **read-only**, and attach worker /
progress sidecars through :mod:`cloris.control_plane`. SQLite is authoritative;
JSON/JSONL projections disagreeing with SQLite are wrong.

Coordination-heavy HTTP plane — onboarding, intake, workspace,
launch readiness/spawn, COS/conversation, reflection — touches adapters plus
global orchestration state beyond per-project runtime SQLite.

Legacy slice-by-slice notes:

Slice 1 endpoints (route declarations byte-identical here; only the body
``GET /`` returns has changed in Slice 5 — see below):

- ``GET /healthz`` — readiness probe used by :func:`cloris.app.run_app` and by
  external smoke checks. Stable JSON contract: ``status``, ``slice``,
  ``version``. The slice tag stays ``"v0-shell-slice-1"`` because this is a
  readiness probe, not a slice-version banner.
- ``GET /`` — returns the built Cloris UI's ``index.html``. See the Slice 5
  paragraph below.

Slice 2 endpoint (route body byte-identical, payload bumped to slice-4 by the
aggregator):

- ``GET /api/status`` — read-only aggregation across discovered
  ``output/state/<source>/*`` directories. Slice 4 enriches the payload with
  worker-sidecar provenance and a ``resumable`` hint, and bumps the slice
  tag in :class:`cloris.models.StatusResponse` to ``"v0-shell-slice-4"``.

Slice 3 endpoint (route body byte-identical):

- ``POST /api/launch/linkedin`` — spawn a detached
  ``python -m cloris.worker ...`` subprocess that writes ``worker.json``
  and ``execvp``s into ``linkedin.session_orchestrator``. LinkedIn-only,
  concurrent-only, fresh-only. The :class:`cloris.models.LaunchResponse`
  slice tag stays ``"v0-shell-slice-3"`` — the launch contract did not
  change in Slice 4.

Slice 4 endpoints:

- ``POST /api/stop/{source}/{state_key}`` — resolve the state dir via
  :func:`cloris.control_plane.enumerate_state_dirs` (path-traversal-safe
  lookup; raw URL segments cannot escape the discovered set), read
  ``worker.json``, and:

  - Missing sidecar ⇒ HTTP 200, ``worker_state="missing"``.
  - Stale sidecar (PID malformed / non-int / dead) ⇒ HTTP 200,
    ``worker_state="stale"``. The stale sidecar is **not** deleted; the
    next launch overwrites it via the existing Slice-3 stale-overwrite
    policy.
  - Alive PID ⇒ HTTP 202, ``worker_state="stopping"``. Send
    ``signal.SIGTERM`` exactly once and return immediately. **No** wait
    for exit. **No** SIGKILL escalation.
  - Unknown ``(source, state_key)`` ⇒ HTTP 404 with
    ``error="state_dir_not_found"``.

- ``POST /api/resume/linkedin`` — same request body as launch
  (:class:`LaunchLinkedInRequest`, ``brief_path``-only with
  ``extra="forbid"``). Spawns a detached worker with ``--mode resume``,
  which threads ``--resume`` into the orchestrator argv. Returns
  :class:`cloris.models.ResumeResponse` (HTTP 201) with ``slice``
  ``"v0-shell-slice-4"`` and ``mode="resume"``. The launch's stale-sidecar
  overwrite policy and ``WorkerAlreadyRunningError`` / ``BriefPathNotFoundError``
  shapes are reused identically.

Pause is still out of scope per ``docs/cloris-control-plane-spec.md`` §7.

Slice 5 surface change (no API contract changes):

- ``GET /`` now returns the built Svelte SPA from
  ``cloris/frontend/dist/index.html``. The Slice 1 inline placeholder is
  gone; the entry HTML is emitted by Vite at build time.
- :func:`mount_static` mounts ``cloris/frontend/dist/assets/`` at
  ``/assets/`` via :class:`fastapi.staticfiles.StaticFiles`. Vite emits
  hashed JS/CSS filenames there (e.g. ``index-DJcGuPNc.js``), plus the
  bundled OFL-licensed fonts under ``/assets/fonts/``. The mount is added
  by :func:`cloris.app.create_app` calling :func:`mount_static` after the
  router is included; existing API routes are unaffected. No slice tag in
  any payload bumps for Slice 5.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

log = logging.getLogger("cloris.api")

from fastapi import HTTPException, Response
from .briefs import _resolve_brief_by_id, api_briefs
from .intake import _intake_db_path, _intake_store
from pydantic import BaseModel, ConfigDict, Field

from cloris.api import _paths
from .routing import router
from cloris.control_plane import (
    aggregate_candidate_detail,
    aggregate_run_report,
    aggregate_status,
    aggregate_workspace,
    enumerate_state_dirs,
    resolve_legacy_workspace,
    state_dirs_for_brief_id,
)
from shared.runtime_state import read_models as _read_models
from cloris.launch_lock import (
    DEFAULT_LAUNCH_LOCK_TIMEOUT_S,
    DEFAULT_SIDECAR_WAIT_TIMEOUT_S,
    LaunchLockTimeoutError,
    state_dir_launch_lock,
    wait_for_sidecar,
)
from cloris.models import (
    ActiveRunSummary,
    OrchestratorDecideRequest,
    OrchestratorDecideResponse,
    OrchestratorDispatchPlanWire,
    OrchestratorRunRecord,
    OrchestratorRunsResponse,
    OrchestratorSynthesizeRequest,
    OrchestratorSynthesizeResponse,
    IdentityCandidateLink,
    IdentityDecisionRequest,
    IdentityPendingDecision,
    IdentityPendingResponse,
    IdentityPerson,
    IdentityUnlinkRequest,
    LaunchMultiPerSourceError,
    LaunchMultiPerSourceResult,
    LaunchMultiRequest,
    LaunchMultiResponse,
    LaunchReadinessBlocker,
    LaunchReadinessResponse,
    LaunchRequest,
    LaunchResponse,
    LegacyResolveResponse,
    ExternalContextClaim,
    MarketDetailResponse,
    MarketLane,
    MarketsListResponse,
    MarketSummary,
    MarketTalentPool,
    MarketThesis,
    MonitorIndexResponse,
    RecruiterCalibrationEntry,
    RecruiterDashboardResponse,
    RecruiterPresenceEntry,
    RecruiterReflectionEntry,
    ReconciledRun,
    ReconcileResponse,
    ReflectionActiveResponse,
    ReflectionCommitRequest,
    ReflectionCommitResponse,
    ReflectionCreateRequest,
    ReflectionDiscardRequest,
    ReflectionResponse,
    ReflectionSession,
    ReflectionStartResearchRequest,
    ReflectionSteeringRequest,
    ResumeResponse,
    RunReportResponse,
    RunSignalResponse,
    RunTelemetryResponse,
    SettingsBriefSaveSummary,
    SettingsCredential,
    SettingsGovernorLimit,
    SettingsResponse,
    StopResponse,
    TelemetryAttemptRow,
    TelemetryEventRow,
    ToolEntry,
    ToolJobStatusWire,
    ToolRunAsyncWire,
    ToolRunRequest,
    ToolRunSyncWire,
    ToolsIndexResponse,
    WorkspaceResponse,
)
from cloris.live_signal import build_run_signal
from cloris import reconciler
from cloris.worker import (
    BriefPathNotFoundError,
    WorkerAlreadyRunningError,
    is_pid_alive,
    read_sidecar,
)
from shared.runtime_state import read_models
from shared.runtime_state.store import RuntimeStateStore


class NoPendingWorkError(Exception):
    """Raised by ``_spawn_linkedin_worker(mode="resume")`` when the brief's
    state dir has no queued or in-progress work to resume.

    Phase 1.3 fix for the lying-success bug: previously ``POST /api/resume/linkedin``
    spawned a worker even when there was nothing to resume. The worker
    exited cleanly seconds later, but the API had already returned 201
    and the UI rendered "Resumed. PID 12345..." — a fake success.

    Now the route layer maps this to HTTP 422 so the UI can render an
    accurate "nothing to resume" message instead.
    """

    def __init__(self, state_dir: str) -> None:
        super().__init__(f"no pending work to resume in {state_dir}")
        self.state_dir = state_dir


class FreshOverResumableStateError(Exception):
    """Raised when a fresh launch would discard resumable generated work.

    Mirrors :class:`NoPendingWorkError` on the opposite launch direction:
    the API must reject a silent fresh launch when the state dir already
    carries a generated brief and pending work, unless the caller supplies
    explicit fresh consent.
    """

    def __init__(self, state_dir: str) -> None:
        super().__init__(
            f"fresh launch over generated brief and resumable work in {state_dir}"
        )
        self.state_dir = state_dir


class WorkerDidNotStartError(Exception):
    """Raised by ``_spawn_worker_for_source`` when :func:`wait_for_sidecar`
    times out after ``Popen`` returns — the spawned process never wrote its
    ``worker.json`` sidecar within the wait window, so we do not actually
    know a live worker exists (Reopen P7.3).

    Same class of bug as :class:`NoPendingWorkError`: an endpoint reporting
    success it never observed. Previously this case was silently ignored
    and the route returned HTTP 201 with the ``Popen`` pid regardless of
    whether the worker was still alive by the time the response went out.
    The route maps this to HTTP 502 ("worker did not start") instead of a
    201-with-pid.
    """

    def __init__(self, source: str, state_dir: str, pid: int) -> None:
        super().__init__(
            f"worker did not start for source={source}: no sidecar observed "
            f"for pid={pid} at {state_dir} within the wait window"
        )
        self.source = source
        self.state_dir = state_dir
        self.pid = pid


def _brief_for_orchestrator_decide(body: OrchestratorDecideRequest):
    """Resolve a :class:`shared.brief_loader.Brief` for dispatch preview."""

    from shared.brief_loader import Brief, load_brief

    if body.brief_path:
        try:
            abs_path = _paths.resolve_brief_path_contained(body.brief_path)
            if not abs_path.is_file():
                raise BriefPathNotFoundError(body.brief_path)
        except BriefPathNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "brief_path_not_found",
                    "path": str(body.brief_path),
                },
            ) from exc
        return load_brief(str(abs_path))

    partial = dict(body.partial_brief or {})
    modules = partial.get("target_modules")
    if not isinstance(modules, list) or len(modules) == 0:
        modules_list = ["linkedin"]
    else:
        modules_list = [
            str(m).strip()
            for m in modules
            if isinstance(m, str) and str(m).strip()
        ]
        if not modules_list:
            modules_list = ["linkedin"]

    brief_id = body.brief_id
    if not isinstance(brief_id, str) or not brief_id.strip():
        bid = partial.get("brief_id")
        brief_id = bid if isinstance(bid, str) and bid.strip() else None
    if not brief_id:
        bid2 = partial.get("id")
        brief_id = bid2 if isinstance(bid2, str) and bid2.strip() else None
    if not brief_id:
        brief_id = "intake-draft"

    role_title = partial.get("role_title")
    role_title_str = role_title if isinstance(role_title, str) else ""

    raw: dict[str, Any] = {**partial}
    raw["target_modules"] = modules_list

    return Brief(
        id=str(brief_id).strip(),
        role_title=role_title_str,
        role_description="",
        kit_url="",
        linkedin_project="",
        linkedin_project_id="",
        minimum_bar="",
        archetypes=[],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
        target_modules=list(modules_list),
        raw=raw,
    )


def _send_sigterm(pid: int) -> None:
    """Module-level seam for the single SIGTERM dispatch in :func:`stop_worker`.

    Tests monkeypatch this symbol to a recorder so the SIGTERM dispatch
    can be observed without disturbing the unrelated ``os.kill(pid, 0)``
    liveness probe inside :func:`cloris.worker.is_pid_alive`. Production
    behavior is a single ``os.kill(pid, signal.SIGTERM)`` and nothing
    else.
    """

    os.kill(pid, signal.SIGTERM)


class StateDirNotFoundError(Exception):
    """Raised when ``stop_worker`` cannot resolve the requested state dir.

    The route maps this to HTTP 404 with
    ``{"error": "state_dir_not_found", "source": ..., "state_key": ...}``.
    Lives in :mod:`cloris.api` rather than :mod:`cloris.worker` because it
    is purely a routing-layer error: the worker module has no notion of
    URL-borne ``(source, state_key)`` lookups.
    """

    def __init__(self, source: str, state_key: str) -> None:
        super().__init__(
            f"state dir not found (source={source}, state_key={state_key})"
        )
        self.source = source
        self.state_key = state_key


class UnknownSourceError(Exception):
    """Raised by ``_spawn_worker_for_source`` when the source path
    parameter doesn't appear in :data:`cloris.launchers.LAUNCHERS`.

    The route maps this to HTTP 422 with
    ``{"error": "unknown_source", "source": ..., "allowed": [...]}``.
    Phase F Slice F1.
    """

    def __init__(self, source: str, allowed: tuple[str, ...]) -> None:
        super().__init__(
            f"unknown launch source '{source}'; allowed={list(allowed)}"
        )
        self.source = source
        self.allowed = allowed


class DomainPausedError(Exception):
    """Raised by ``_spawn_worker_for_source`` when launches for a source
    are administratively paused via the ``CLORIS_PAUSE_LAUNCHES_<SOURCE>``
    environment gate (Reopen Y.5.2).

    A clean operator kill-switch: set ``CLORIS_PAUSE_LAUNCHES_DESIGNER=1``
    (case-insensitive truthy: ``{"1","true","yes","on"}``) to refuse every
    launch/resume of that one source while leaving every other source — and
    the brief-keyed candidate path — untouched. The gate fires before any
    state dir or ``runs`` row is created, so a paused launch leaves no
    sidecar and no partial run behind.

    Every launch entry's typed-allowlist except chain maps this to HTTP 409
    (temporarily unavailable / retry once the operator unsets the flag),
    mirroring :class:`WorkerAlreadyRunningError`. Without that mapping the
    raise would surface as a raw 500 — so the 409 handler is part of the
    contract, not optional.
    """

    def __init__(self, source: str) -> None:
        super().__init__(
            f"launches for {source} are paused; "
            f"unset CLORIS_PAUSE_LAUNCHES_{source.upper()} to resume"
        )
        self.source = source


class SourceSunsetError(Exception):
    """Raised by ``_spawn_worker_for_source`` when the source's launcher
    registry entry is marked ``launchable=False`` / ``sunset=True``
    (Reopen P7.1 — designer and exec_search, gated off by product
    decision, not by a bug in their internals).

    Unlike :class:`DomainPausedError` (a temporary operator kill-switch),
    this is a standing product state: the subagent is retired, not
    transiently unavailable. ``force=true`` never bypasses it — force
    only skips readiness PROBES at the ``_launch_for_source_impl`` layer,
    and this check lives one layer deeper, at the single spawn choke
    point, so it fires regardless of caller. Every launch entry's typed-
    allowlist except chain maps this to HTTP 409 with a message the
    recruiter can see, so the copy must say "subagent," never "module."
    """

    def __init__(self, source: str) -> None:
        super().__init__(
            f"the {source} subagent is paused for now and cannot be launched"
        )
        self.source = source


class LaunchNotReadyError(Exception):
    """Raised when ``POST /api/launch/{source}`` is called without
    ``force=true`` and the launch-readiness probe (Phase D Slice D9)
    surfaces blockers. The route maps this to HTTP 422 with
    ``{"error": "launch_not_ready", "source": ..., "blockers": [...]}``.
    """

    def __init__(self, source: str, blockers: list) -> None:
        super().__init__(
            f"launch not ready for source={source}; "
            f"blockers={[b.kind for b in blockers]}"
        )
        self.source = source
        self.blockers = blockers


class BriefIdNotFoundError(Exception):
    """Raised when ``POST /api/launch/{source}`` receives a brief_id
    that doesn't resolve to any brief in the catalog.

    Distinct from :class:`BriefPathNotFoundError` (which fires on a
    legacy ``brief_path`` request payload) because the brief_id flow
    cannot meaningfully report the missing path. Route maps to HTTP 404.
    """

    def __init__(self, brief_id: str) -> None:
        super().__init__(f"no brief in catalog hashes to id '{brief_id}'")
        self.brief_id = brief_id


class LaunchLinkedInRequest(BaseModel):
    """Request body for ``POST /api/launch/linkedin`` and
    ``POST /api/resume/linkedin``.

    Slice 4 reuses this exact model for resume because the user-approved
    contract is "same request body shape as launch" — a single
    ``brief_path`` string. ``extra="forbid"`` rejects any unknown field
    (e.g. ``input_mode``, ``mode``) at the request boundary, which is how
    ``away`` and stray fields are rejected without ever reaching the
    helper.

    ``force_fresh`` is explicit consent for the fresh launch route to
    discard/rebuild generated artifacts when resumable state already
    exists. It is distinct from generic ``force`` on :class:`LaunchRequest`,
    which only skips readiness probes.
    """

    model_config = ConfigDict(extra="forbid")
    brief_path: str
    force_fresh: bool = False


# Phase E Slice E1: market viewer endpoints. The on-disk
# `MarketIntelArtifact` is rich (60 lanes, full evidence index,
# section-generation metadata); the wire shapes are trimmed to the
# recruiter-facing fields the viewer renders. Detail responses build
# from `market_intelligence.engine.load_artifact()`; the catalog list
# from `list_market_records()`.


@router.get("/api/markets", response_model=MarketsListResponse)
def api_markets_list() -> MarketsListResponse:
    """Catalog of every market with a parseable artifact on disk.

    Sorted most-recently-updated first by `freshness.artifact_updated_at`.
    Empty when no artifacts exist (no error — the route layer renders
    a Cloris-voice empty state).
    """

    from market_intelligence.engine import list_market_records

    records = list_market_records()
    return MarketsListResponse(
        markets=[
            MarketSummary(
                market_key=r.market_key,
                role_title=r.role_title,
                role_level=r.role_level,
                geography=r.geography,
                brief_ids_seen=r.brief_ids_seen,
                last_updated_at=r.last_updated_at,
                run_count=r.run_count,
                saved_count=r.saved_count,
                aggregate_save_rate=r.aggregate_save_rate,
            )
            for r in records
        ]
    )


@router.get(
    "/api/market/{market_key}",
    response_model=MarketDetailResponse,
)
def api_market_detail(market_key: str) -> MarketDetailResponse:
    """Per-market detail payload for the `#/market/<key>` viewer.

    404 ``market_not_found`` when no parseable artifact exists at the
    canonical path. The wire shape flattens the artifact to the
    recruiter-facing fields and drops the cloris-internal machinery
    (evidence_index, section_generation_metadata, etc.).
    """

    from market_intelligence.engine import load_artifact

    artifact = load_artifact(market_key)
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "market_not_found", "market_key": market_key},
        )

    identity = artifact.market_identity
    freshness = artifact.freshness or {}
    aggregate = artifact.aggregate_metrics or {}

    def _opt_float(value: object) -> float | None:
        try:
            return float(value) if value is not None else None  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def _str_field(payload: dict, key: str) -> str:
        value = payload.get(key)
        return str(value) if isinstance(value, str) else ""

    lanes: list[MarketLane] = []
    for lane in artifact.lane_intelligence or []:
        if not isinstance(lane, dict):
            continue
        metrics = lane.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        candidates_seen = int(metrics.get("candidates_seen") or 0)
        saves = int(metrics.get("saves") or 0)
        # R6: omit zero-evidence lanes from the detail wire so the
        # frontend never sees empty rows it would have to suppress.
        if candidates_seen == 0 and saves == 0:
            continue
        lanes.append(
            MarketLane(
                lane_key=str(lane.get("lane_key") or "").strip(),
                domain_lane=str(lane.get("domain_lane") or "").strip(),
                novelty_bucket=str(lane.get("novelty_bucket") or "").strip(),
                status=str(lane.get("status") or "").strip(),
                candidates_seen=candidates_seen,
                saves=saves,
                save_rate=_opt_float(metrics.get("save_rate")),
                why_it_works=(
                    str(lane.get("why_it_works")).strip()
                    if isinstance(lane.get("why_it_works"), str)
                    else None
                ),
                recommended_action=(
                    str(lane.get("recommended_action")).strip()
                    if isinstance(lane.get("recommended_action"), str)
                    else None
                ),
            )
        )

    talent_pools: list[MarketTalentPool] = []
    for pool in artifact.talent_pool_intelligence or []:
        if not isinstance(pool, dict):
            continue
        talent_pools.append(
            MarketTalentPool(
                pool_key=str(pool.get("pool_key") or "").strip(),
                label=str(pool.get("label") or "").strip(),
                signal_strength=str(pool.get("signal_strength") or "").strip(),
                status=str(pool.get("status") or "").strip(),
                evidence_summary=(
                    str(pool.get("evidence_summary")).strip()
                    if isinstance(pool.get("evidence_summary"), str)
                    else None
                ),
            )
        )

    thesis_data = artifact.market_thesis or {}
    _raw_ec = thesis_data.get("external_context") or []
    thesis = MarketThesis(
        summary=_str_field(thesis_data, "summary"),
        supply_assessment=_str_field(thesis_data, "supply_assessment"),
        competition_assessment=_str_field(thesis_data, "competition_assessment"),
        external_context=[
            ExternalContextClaim(
                claim=str(c.get("claim", "")),
                label=c.get("label") or "",
                evidence_refs=[r for r in (c.get("evidence_refs") or []) if isinstance(r, str)],
                confidence=float(c.get("confidence") or 0.0),
            )
            for c in (_raw_ec if isinstance(_raw_ec, list) else [])
            if isinstance(c, dict) and c.get("claim")
        ],
    )

    # Engine emits brief_recommendations as a list of dicts on the
    # artifact. Surface it verbatim — the frontend's computeBriefDiff()
    # walks it as a fourth diff source. Filter out malformed entries
    # so the wire stays well-typed.
    brief_recommendations: list[dict] = [
        rec
        for rec in (artifact.brief_recommendations or [])
        if isinstance(rec, dict)
    ]

    return MarketDetailResponse(
        market_key=identity.market_key,
        role_title=identity.role_title,
        role_level=identity.role_level,
        geography=identity.geography,
        brief_ids_seen=identity.brief_ids_seen,
        last_updated_at=str(freshness.get("artifact_updated_at") or "").strip(),
        run_count=int(aggregate.get("run_count") or 0),
        candidates_seen=int(aggregate.get("candidates_seen") or 0),
        saved_count=int(aggregate.get("saved_count") or 0),
        rejected_count=int(aggregate.get("rejected_count") or 0),
        aggregate_save_rate=_opt_float(aggregate.get("save_rate")),
        facial_yes_rate=_opt_float(aggregate.get("facial_yes_rate")),
        lanes=lanes,
        talent_pools=talent_pools,
        market_thesis=thesis,
        brief_recommendations=brief_recommendations,
    )


@router.post("/api/reconcile", response_model=ReconcileResponse)
def api_reconcile() -> ReconcileResponse:
    """Reconcile runs marked ``status='running'`` whose workers have died.

    The aggregator (``GET /api/status``) is read-only and faithfully
    reports any contradiction it finds — a run can stay marked ``running``
    forever after the orchestrator process is killed by Mac sleep, an
    OOM, ``kill -9``, or a host reboot. This endpoint walks every state
    dir, classifies the worker as ``missing_sidecar`` / ``bad_sidecar`` /
    ``pid_dead``, and finalizes those runs as ``status='abandoned'`` with
    ``stop_reason='worker_missing'`` so the UI shows a "Lost track" pill
    instead of a lying "Working" one.

    The endpoint is intentionally explicit — the frontend calls it on
    app mount and on a slow timer — so test suites that monkeypatch
    ``aggregate_status`` are not perturbed by side effects on read.

    Idempotent: a re-run with no zombies returns ``applied=0`` and an
    empty mutation list.
    """

    applied, mutations = reconciler.reconcile_and_apply()
    return ReconcileResponse(
        applied=applied,
        mutations=[
            ReconciledRun(
                source=m.source,  # type: ignore[arg-type]
                state_key=m.state_key,
                run_id=m.run_id,
                new_status=m.new_status,
                stop_reason=m.stop_reason,
                reason=m.reason,
            )
            for m in mutations
        ],
    )


@router.post("/api/operator/recover-stale-locks", response_model=ReconcileResponse)
def api_operator_recover_stale_locks() -> ReconcileResponse:
    """Hidden support action: run the same safe recovery sweep as the UI.

    This endpoint deliberately does not accept paths or raw PIDs. It only
    clears locks the reconciler can prove are stale from canonical state.
    """

    return api_reconcile()


@router.get("/api/operator/support-bundle")
def api_operator_support_bundle() -> dict[str, Any]:
    """Redacted support snapshot for trial triage.

    Candidate rows, secrets, filesystem paths, PIDs, and console logs are
    omitted. Canonical runtime state remains the source for lifecycle fields.
    """

    status = aggregate_status()
    entries: list[dict[str, Any]] = []
    for entry in status.entries:
        latest = entry.latest_run
        entries.append(
            {
                "source": entry.source,
                "state_key": entry.state_key,
                "lifecycle": entry.lifecycle,
                "worker_state": entry.worker_state,
                "worker_alive": entry.worker_alive,
                "heartbeat_age_s": entry.heartbeat_age_s,
                "run_status": latest.status if latest is not None else None,
                "stop_reason": latest.stop_reason if latest is not None else None,
                "run_started_at": latest.started_at if latest is not None else None,
                "run_ended_at": latest.ended_at if latest is not None else None,
            }
        )
    return {
        "slice": "v0-operator-support-bundle-1",
        "generated_at_unix": time.time(),
        "trial_mode": status.trial_mode,
        "counts": status.counts.model_dump(),
        "modules": [m.model_dump() for m in status.modules],
        "entries": entries,
    }


@router.get(
    "/api/run/{source}/{state_key}/{run_id}",
    response_model=RunReportResponse,
)
def api_run_report(
    source: str, state_key: str, run_id: int
) -> RunReportResponse:
    """Per-run report (Phase B).

    Aggregates run lifecycle, work-unit progress, attempt-health, and
    candidate decisions for a single ``run_id`` in the discovered
    ``(source, state_key)`` state dir.

    Errors:
      * 404 ``state_dir_not_found`` — no enumerated state dir matches
        ``(source, state_key)``.
      * 404 ``run_not_found`` — state dir resolves but the canonical
        SQLite has no row with this ``run_id`` (or the DB is missing /
        unreadable; we don't distinguish at the wire).
    """

    state_dir = _resolve_state_dir(source, state_key)
    if state_dir is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "state_dir_not_found",
                "source": source,
                "state_key": state_key,
            },
        )
    report = aggregate_run_report(
        state_dir, source=source, state_key=state_key, run_id=run_id
    )
    if report is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "run_not_found",
                "source": source,
                "state_key": state_key,
                "run_id": run_id,
            },
        )
    return report


# ---------------------------------------------------------------------------
# Phase G Slice G3: Live Monitor endpoints. Operational depth for runs in
# motion. Polled by Monitor.svelte at 1s while the worker is alive, 5s
# otherwise. The /telemetry endpoint surfaces raw candidate_attempts +
# events for the operator view — recruiters who want forensics drop into
# sqlite directly; this is a window, not a full audit trail.
# ---------------------------------------------------------------------------


@router.get(
    "/api/monitor/index",
    response_model=MonitorIndexResponse,
)
def api_monitor_index() -> MonitorIndexResponse:
    """List runs with a currently-alive worker process.

    Derived from ``aggregate_status``: filter to entries where
    ``worker_alive=True``. Cheap; no new query work over /api/status.
    """

    status = aggregate_status()
    active: list[ActiveRunSummary] = []
    for entry in status.entries:
        if entry.worker_alive is not True:
            continue
        latest = entry.latest_run
        active.append(
            ActiveRunSummary(
                source=entry.source,
                state_key=entry.state_key,
                run_id=latest.id if latest is not None else None,
                run_status=latest.status if latest is not None else None,
                stop_reason=latest.stop_reason if latest is not None else None,
                started_at=latest.started_at if latest is not None else None,
                ended_at=latest.ended_at if latest is not None else None,
                brief_id=entry.brief_id_from_run,
                brief_role_title=entry.brief_role_title,
                worker_pid=entry.worker_pid,
                lifecycle=entry.lifecycle,
            )
        )
    # Most-recently-started first within the active set.
    active.sort(key=lambda r: r.started_at or "", reverse=True)
    return MonitorIndexResponse(slice="v0-monitor-index-1", active_runs=active)


_TELEMETRY_ATTEMPTS_LIMIT = 50
_TELEMETRY_EVENTS_LIMIT = 30


@router.get(
    "/api/run/{source}/{state_key}/{run_id}/telemetry",
    response_model=RunTelemetryResponse,
)
def api_run_telemetry(
    source: str, state_key: str, run_id: int
) -> RunTelemetryResponse:
    """Per-run operational telemetry: recent attempts + events.

    Bounded windowed view: 50 most-recent attempts, 30 most-recent events.
    Recruiters who need the full history use sqlite directly; this is a
    live monitor window, not a forensics surface.

    Errors:
      * 404 ``state_dir_not_found`` — unknown ``(source, state_key)``.
    """

    state_dir = _resolve_state_dir(source, state_key)
    if state_dir is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "state_dir_not_found",
                "source": source,
                "state_key": state_key,
            },
        )
    db_path = state_dir / "runtime_state.sqlite3"
    if not db_path.exists():
        return RunTelemetryResponse(
            slice="v0-run-telemetry-1",
            source=source,  # type: ignore[arg-type]
            state_key=state_key,
            run_id=run_id,
            attempts=[],
            events=[],
            last_event_at=None,
            attempts_total=0,
            events_total=0,
        )

    # Read-only path: route through ``read_models.run_telemetry`` instead
    # of instantiating ``RuntimeStateStore``. The store's __init__ runs
    # unconditional DDL plus an ``INSERT OR REPLACE INTO meta`` on every
    # call (shared/runtime_state/store.py:87) — using it here would make
    # this polling endpoint a writer against active runtime state. The
    # control-plane docstring (cloris/control_plane.py:10-15) explicitly
    # warns against this pattern; ``read_models.run_telemetry`` opens
    # the DB via the URI ``mode=ro`` pattern so the kernel refuses any
    # accidental write.
    from shared.runtime_state.read_models import run_telemetry

    telemetry = run_telemetry(
        db_path,
        run_id=run_id,
        attempts_limit=_TELEMETRY_ATTEMPTS_LIMIT,
        events_limit=_TELEMETRY_EVENTS_LIMIT,
    )

    attempts = [
        TelemetryAttemptRow(
            id=a.id,
            candidate_id=a.candidate_id,
            work_unit_id=a.work_unit_id,
            stage=a.stage,
            attempt_number=a.attempt_number,
            status=a.status,
            failure_kind=a.failure_kind,
            failure_reason=a.failure_reason,
            started_at=a.started_at,
            ended_at=a.ended_at,
        )
        for a in telemetry.attempts
    ]
    events = [
        TelemetryEventRow(
            id=e.id,
            event_type=e.event_type,
            candidate_id=e.candidate_id,
            attempt_id=e.attempt_id,
            payload_summary=_truncate_payload_for_telemetry(e.payload_json),
            created_at=e.created_at,
        )
        for e in telemetry.events
    ]
    return RunTelemetryResponse(
        slice="v0-run-telemetry-1",
        source=source,  # type: ignore[arg-type]
        state_key=state_key,
        run_id=run_id,
        attempts=attempts,
        events=events,
        last_event_at=telemetry.last_event_at,
        attempts_total=telemetry.attempts_total,
        events_total=telemetry.events_total,
    )


def _truncate_payload_for_telemetry(raw: object, *, max_chars: int = 240) -> str | None:
    """Compact stringification of an event payload for the Monitor row.

    The full payload_json may be megabytes (LLM transcripts, stale page
    snapshots). Telemetry rows just need a quick "what kind of payload"
    glance. We return the first ``max_chars`` of the raw string to keep
    the wire payload bounded.
    """

    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


@router.get(
    "/api/run-signal/{source}/{state_key}",
    response_model=RunSignalResponse,
)
def api_run_signal(source: str, state_key: str) -> RunSignalResponse:
    """Recruiter-readable live signal for a state dir.

    This is a read-only summary over worker artifacts and projections. It is
    intentionally less raw than Monitor telemetry and must not be used for
    launch/stop decisions; SQLite and sidecars remain canonical for control.
    """

    state_dir = _resolve_state_dir(source, state_key)
    if state_dir is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "state_dir_not_found",
                "source": source,
                "state_key": state_key,
            },
        )
    return RunSignalResponse(
        **build_run_signal(state_dir, source=source, state_key=state_key)
    )


@router.get(
    "/api/workspace/{brief_id}",
    response_model=WorkspaceResponse,
)
def api_workspace(brief_id: str) -> WorkspaceResponse:
    """Per-brief candidate workspace (Phase C-bis 0.1, brief-first).

    Aggregates every SAVE-class candidate for ``brief_id`` across every
    state_dir whose latest run carries that brief_id. Today that's
    typically one source; Phase F multi-module operation will fan out
    across LinkedIn + GitHub + Researcher in the same response.

    Errors:
      * 404 ``workspace_not_found`` — no state_dir's latest run carries
        this brief_id (or no runs exist at all).
    """

    workspace = aggregate_workspace(brief_id=brief_id)
    if workspace is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "workspace_not_found",
                "brief_id": brief_id,
            },
        )
    return workspace


class ExecSearchInvestigateRequest(BaseModel):
    """Executive Search Slice 9: pre-launch investigation request body."""

    brief_path: str = Field(
        description="Project-relative path to the brief JSON file."
    )
    persist: bool = Field(
        default=True,
        description=(
            "When False, returns a preview without writing the packet "
            "to the canonical investigation path. Default writes."
        ),
    )


class ExecSearchInvestigateResponse(BaseModel):
    """Executive Search Slice 9: pre-launch investigation response.

    Wire shape for ``POST /api/exec-search/investigate``. Returns the
    investigation packet on success, or a 422 with a typed failure
    detail. The recruiter reviews this in the Cloris UI before
    launching; edits at this point feed back into strategy formation
    per the old spec's Stage 0.
    """

    slice: Literal["v0-exec-search-slice-9"] = Field(
        default="v0-exec-search-slice-9"
    )
    packet: dict


@router.post(
    "/api/exec-search/investigate",
    response_model=ExecSearchInvestigateResponse,
    status_code=200,
)
def api_exec_search_investigate(
    req: ExecSearchInvestigateRequest,
) -> ExecSearchInvestigateResponse:
    """Executive Search pre-launch investigation (Slice 9).

    Net-new entry per spec amendment B (the pre-launch market
    intelligence surface didn't exist; ``update_market_intel`` is
    post-run only). Wraps :func:`market_intelligence.pre_launch.run_pre_launch_investigation`.

    Errors:
      * 404 ``brief_not_found`` — the brief JSON file is missing.
      * 422 ``brief_load_error`` — the brief JSON failed to parse.
      * 500 ``persistence_error`` / ``investigation_failed`` —
        unexpected backend failure.
    """

    from market_intelligence.pre_launch import (
        InvestigationFailure,
        run_pre_launch_investigation,
    )

    raw_path = req.brief_path
    if not raw_path:
        raise HTTPException(
            status_code=422,
            detail={"error": "missing_brief_path"},
        )
    try:
        brief_path = _paths.resolve_brief_path_contained(raw_path)
    except BriefPathNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "brief_not_found",
                "detail": f"Brief file not found at {raw_path}",
                "brief_path": str(req.brief_path),
            },
        ) from exc

    result = run_pre_launch_investigation(
        brief_path=brief_path,
        prior_search_context=None,
        research_backend=None,
        persist=req.persist,
    )
    if isinstance(result, InvestigationFailure):
        if result.reason == "brief_not_found":
            raise HTTPException(
                status_code=404,
                detail={
                    "error": result.reason,
                    "detail": result.detail,
                    "brief_path": str(req.brief_path),
                },
            )
        if result.reason == "brief_load_error":
            raise HTTPException(
                status_code=422,
                detail={
                    "error": result.reason,
                    "detail": result.detail,
                    "brief_path": str(req.brief_path),
                },
            )
        raise HTTPException(
            status_code=500,
            detail={"error": result.reason, "detail": result.detail},
        )
    return ExecSearchInvestigateResponse(packet=result.to_dict())


# ---------------------------------------------------------------------------
# Candidate read + annotation routes (shortlist, candidate detail, note,
# user_status, judgment-accuracy, Designer principle-feedback / asset
# exclude+revoke) were carved into ``cloris.api.candidate_routes`` (Phase 4,
# slice P4-4). They register on the shared router when that module is imported
# by ``cloris.api``. ``_cross_brief_presence_for_candidate`` is re-exported
# below so existing ``cloris.api._monolith`` import paths keep resolving.
# ---------------------------------------------------------------------------
from cloris.api.candidate_routes import (  # noqa: E402,F401
    _cross_brief_presence_for_candidate,
)


# ---------------------------------------------------------------------------
# Legacy URL resolvers (Phase C-bis 0.1).
#
# Old bookmarks pointing at the source-siloed URLs hit these endpoints to
# learn the brief_id. The frontend then rewrites the hash to the new
# brief-first URL. Cheap reads; no mutations. Both endpoints return the
# same shape (just `brief_id`); the candidate variant exists separately
# only because the legacy URL also carries a `candidate_id` segment that
# the frontend already has — so it doesn't need to be echoed back.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase G Slice G2: identity reconciliation endpoints. Recruiter-driven merge
# / keep-separate decisions over the F3 backend's pending_merge_decisions.
#
# Routes are brief-scoped because the identity service is brief-scoped: a
# pending decision is meaningful only in the context of a specific brief's
# saves. The UI lives at #/workspace/<brief_id>/identity.
# ---------------------------------------------------------------------------


def _identity_person_to_wire(
    person: "object",  # PersonWithEvidence — string-quoted to defer import
) -> IdentityPerson:
    """Convert a service-layer PersonWithEvidence to the wire IdentityPerson.

    Routes the link_kind enum through ``describe_merge_signal`` so the
    frontend never sees raw enums. Each candidate link gets its own
    editorial ``describe`` line.
    """

    from shared.identity_resolution_service import describe_merge_signal

    sources_wire: list[IdentityCandidateLink] = []
    for link in person.sources:
        sources_wire.append(
            IdentityCandidateLink(
                source=link.source,
                state_key=link.state_key,
                candidate_id=link.candidate_id,
                link_kind=link.link_kind,
                recruiter_locked=link.recruiter_locked,
                describe=describe_merge_signal(link.link_kind, link.match_signal),
            )
        )
    return IdentityPerson(
        person_id=person.person_id,
        canonical_name=person.canonical_name,
        canonical_handle=person.canonical_handle,
        sources=sources_wire,
    )


@router.get(
    "/api/brief/{brief_id}/identity/pending",
    response_model=IdentityPendingResponse,
)
def api_identity_pending(brief_id: str) -> IdentityPendingResponse:
    """List unresolved merge decisions for a brief, with person evidence
    and Cloris-voice signal_summary prose.

    Side effect: re-runs ``resolve_persons_for_brief`` on read so the
    pending list reflects any candidates added since the last launch.
    Idempotent — already-linked candidates and recruiter-locked rows
    are untouched.
    """

    from shared.identity_resolution_service import (
        auto_resolve_anonymous_pending,
        brief_person_count,
        pending_decisions_for_brief,
        resolve_persons_for_brief,
    )

    resolve_persons_for_brief(brief_id)
    # Reopen Stage 3a: accrue recruiter↔person sightings from the persons
    # the resolver just wrote. Idempotent (ledger-gated) and fail-soft —
    # this read re-runs resolution every call, so a naive sighting would
    # inflate times_surfaced; record_sightings_for_brief is a no-op on a
    # re-resolution of an already-sighted brief and never raises.
    from shared.runtime_state.recruiter_sighting import record_sightings_for_brief

    record_sightings_for_brief(brief_id)
    auto_resolve_anonymous_pending(brief_id=brief_id)
    persons_total = brief_person_count(brief_id)
    decisions = pending_decisions_for_brief(brief_id)

    decisions_wire = [
        IdentityPendingDecision(
            decision_id=d.decision_id,
            person_a=_identity_person_to_wire(d.person_a),
            person_b=_identity_person_to_wire(d.person_b),
            signal_summary=d.signal_summary,
            created_at=d.created_at,
        )
        for d in decisions
    ]
    return IdentityPendingResponse(
        slice="v0-identity-pending-1",
        brief_id=brief_id,
        persons_total=persons_total,
        decisions=decisions_wire,
    )


@router.post(
    "/api/brief/{brief_id}/identity/decision",
    status_code=204,
)
def api_identity_decision(brief_id: str, req: IdentityDecisionRequest) -> Response:
    """Resolve one pending merge decision.

    422 cases:
      - decision_id doesn't exist for this brief
      - decision was already resolved (terminal)
      - choice is not in {"merge","keep_separate"} (caught by Pydantic)
    """

    from shared.identity_resolution_service import record_decision_by_id

    try:
        record_decision_by_id(
            brief_id=brief_id,
            decision_id=req.decision_id,
            decision=req.choice,
        )
    except LookupError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "identity_decision_not_found",
                "brief_id": brief_id,
                "decision_id": req.decision_id,
            },
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "identity_decision_already_resolved",
                "brief_id": brief_id,
                "decision_id": req.decision_id,
                "message": str(exc),
            },
        )
    return Response(status_code=204)


@router.post(
    "/api/brief/{brief_id}/identity/unlink",
    status_code=204,
)
def api_identity_unlink(brief_id: str, req: IdentityUnlinkRequest) -> Response:
    """Split a candidate off into its own person row.

    Locks the new link so auto-resolution doesn't merge it back.
    Idempotent: if the candidate is unknown, the call is a no-op rather
    than a 404 — the recruiter's intent is "make sure this is its own
    person," and the no-record case satisfies that.
    """

    from shared.identity_resolution_service import record_recruiter_unlink

    record_recruiter_unlink(
        source=req.source,
        state_key=req.state_key,
        candidate_id=req.candidate_id,
    )
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Phase G Slice G4: Tools index + execution.
# ---------------------------------------------------------------------------


def _tool_entry_for_wire(tool) -> ToolEntry:
    """Project a registry ToolDefinition into the wire ToolEntry shape.

    Includes a hand-crafted ``schema_fields`` summary so the frontend can
    render a form without re-introspecting Pydantic at runtime.
    """

    schema_fields: list[dict] = []
    if tool.args_schema is not None:
        model_schema = tool.args_schema.model_json_schema()
        properties = model_schema.get("properties", {}) or {}
        required = set(model_schema.get("required", []) or [])
        for name, prop in properties.items():
            field_type = prop.get("type") or prop.get("enum") or "string"
            schema_fields.append(
                {
                    "name": name,
                    "type": field_type if isinstance(field_type, str) else "enum",
                    "required": name in required,
                    "default": prop.get("default"),
                    "description": prop.get("description") or "",
                }
            )
    return ToolEntry(
        tool_id=tool.tool_id,
        tier=tool.tier,
        label=tool.label,
        pitch=tool.pitch,
        cli_command=tool.cli_command,
        execution_model=tool.execution_model,
        schema_fields=schema_fields,
    )


# ---------------------------------------------------------------------------
# Phase G Slice G5: Settings transparency surface. Read-only — credentials
# as ✓/✗ booleans, governor limits as constants with editorial explainers,
# brief save destinations summarized from V2 source_config.
# ---------------------------------------------------------------------------


_CREDENTIAL_LABELS: list[tuple[str, str, str]] = [
    (
        "anthropic_api_key",
        "Anthropic (Claude)",
        "Cloris uses Claude for the heavy reading and writing — judgment, brief drafting, market reads. Without it, runs can't think.",
    ),
    (
        "openai_api_key",
        "OpenAI",
        "OpenAI is a fallback for some judgment paths. Optional but recommended.",
    ),
    (
        "google_api_key",
        "Google (Gemini)",
        "Used for the design-consult workflow. Optional unless you're running the audit ensemble.",
    ),
    (
        "perplexity_api_key",
        "Perplexity",
        "Used for the external-evidence workflow during candidate research.",
    ),
    (
        "linkedin_cdp",
        "LinkedIn (Chrome session)",
        "Cloris uses its own Chrome window for LinkedIn. Open LinkedIn there before starting a search.",
    ),
]


def _credential_present(key: str) -> bool:
    """Return True iff the credential is set + non-empty.

    NEVER returns the value itself; the wire shape is boolean-only by
    design (sensitive operational state must be boolean-only).
    """

    import shared.config as cfg
    import os

    def _val_set(attr: str) -> bool:
        raw = os.getenv(attr) or getattr(cfg, attr, "")
        return isinstance(raw, str) and raw.strip() != ""

    if key == "anthropic_api_key":
        return _val_set("ANTHROPIC_API_KEY")
    if key == "openai_api_key":
        return _val_set("OPENAI_API_KEY")
    if key == "google_api_key":
        return _val_set("GOOGLE_API_KEY")
    if key == "perplexity_api_key":
        return _val_set("PERPLEXITY_API_KEY")
    if key == "linkedin_cdp":
        # URL presence only. Reachability is the launch-readiness probe's job.
        return _val_set("CDP_URL")
    return False


@router.get("/api/settings", response_model=SettingsResponse)
def api_settings() -> SettingsResponse:
    """Read-only operational snapshot. Credentials boolean-only; governor
    limits read as constants with editorial explainers; save destinations
    summarized from V2 ``source_config`` per brief."""

    credentials = [
        SettingsCredential(
            key=key,
            label=label,
            present=_credential_present(key),
            pitch=pitch,
        )
        for key, label, pitch in _CREDENTIAL_LABELS
    ]

    # Save destinations: walk the brief catalog and summarize per-brief
    # source_config. Reuses the brief loader from D1.
    #
    # No outer try/except: api_briefs() failing is a system-level fault
    # that should surface as 500, not silently degrade save_destinations
    # to []. The previous outer broad-except hid a NameError on a typo'd
    # function call (api_list_briefs vs api_briefs) for an unknown
    # period — every /api/settings response carried an empty
    # save_destinations until the typo was caught by code review. The
    # inner try/except below covers the only real degradation case
    # (per-brief loader/schema failure for one bad brief).
    #
    # Briefs whose ``brief_id`` is None are deliberately skipped:
    # ``aggregate_briefs`` returns brief_id=None when ``derive_brief_id``
    # raises (malformed schema, unparseable JSON, etc.). Those briefs are
    # broken at a deeper level than save-destination configuration, and
    # surfacing them on the settings page with a missing id would mislead
    # the recruiter into thinking the brief is ready to save against.
    save_destinations: list[SettingsBriefSaveSummary] = []
    briefs_response = api_briefs()
    for b in briefs_response.briefs:
        if b.brief_id is None:
            continue
        target_modules = list(b.target_modules or [])
        linkedin_project_id = None
        try:
            from shared.brief_loader import load_brief

            data = load_brief(b.path)
            from shared.brief_v2_schema import linkedin_project_id_from_brief

            linkedin_project_id = linkedin_project_id_from_brief(data)
        except Exception:
            linkedin_project_id = None
        save_destinations.append(
            SettingsBriefSaveSummary(
                brief_id=b.brief_id,
                role_title=b.role_title,
                target_modules=target_modules,
                linkedin_project_id=linkedin_project_id,
            )
        )

    # Governor: read constants directly. Per shared/governor.py:
    # "do not make these configurable."
    import shared.governor as gov

    governor = [
        SettingsGovernorLimit(
            name="MAX_PROFILE_OPENS_PER_SESSION",
            label="Profiles per run",
            value=int(gov.MAX_PROFILE_OPENS_PER_SESSION),
            explainer="Tuned for safe LinkedIn cadence. Cloris won't open more than this during one run.",
        ),
        SettingsGovernorLimit(
            name="MAX_PROFILE_OPENS_PER_24H",
            label="Profiles per 24h",
            value=int(gov.MAX_PROFILE_OPENS_PER_24H),
            explainer="Daily ceiling across all runs. Cloris pauses until the rolling window clears.",
        ),
        SettingsGovernorLimit(
            name="MAX_SESSION_DURATION_SECONDS",
            label="Max run length",
            # LinkedIn's active session band lives in
            # linkedin.session_orchestrator._sample_session_duration(); avoid
            # importing the launcher stack from the settings surface.
            value="randomized 4-5h per run (+10 min governor backstop)",
            explainer="Wall-clock cap with jitter. Randomization is part of the safety profile; Cloris doesn't expose the exact value.",
        ),
    ]

    import shared.config as cfg

    return SettingsResponse(
        slice="v0-settings-1",
        credentials=credentials,
        save_destinations=save_destinations,
        governor=governor,
        cdp_url=str(getattr(cfg, "CDP_URL", "") or ""),
    )


@router.get("/api/tools", response_model=ToolsIndexResponse)
def api_tools_index() -> ToolsIndexResponse:
    """Return the catalog of tools — Tier A/B with UI runners + Tier B/C
    documentation entries (CLI only)."""

    from cloris.tools_registry import list_tools
    import shared.config as cfg

    tools = list_tools()
    if cfg.CLORIS_TRIAL_MODE:
        trial_tool_ids = {
            "iterate_brief",
            "backfill_clean_linkedin_urls",
        }
        tools = [t for t in tools if t.tool_id in trial_tool_ids]
    return ToolsIndexResponse(
        slice="v0-tools-index-1",
        tools=[_tool_entry_for_wire(t) for t in tools],
    )


@router.post("/api/tools/{tool_id}")
async def api_tools_run(tool_id: str, req: ToolRunRequest):
    """Execute a tool. Sync tools return ``ToolRunSyncWire`` immediately;
    async tools return ``ToolRunAsyncWire`` with a job_id; cli_only tools
    return 422 (the catalog already shows them).
    """

    from cloris.tools_registry import find_tool
    from cloris.tools_runtime import execute_async, execute_sync

    tool = find_tool(tool_id)
    if tool is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "tool_not_found", "tool_id": tool_id},
        )
    if tool.execution_model == "cli_only":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "tool_cli_only",
                "tool_id": tool_id,
                "cli_command": tool.cli_command,
            },
        )
    if tool.args_schema is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "tool_misconfigured", "tool_id": tool_id},
        )
    try:
        args_model = tool.args_schema.model_validate(req.args)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "tool_args_invalid",
                "tool_id": tool_id,
                "message": str(exc),
            },
        )

    if tool.execution_model == "sync":
        result = await execute_sync(tool, args_model)
        return ToolRunSyncWire(
            slice="v0-tool-sync-1",
            tool_id=result.tool_id,
            exit_code=result.exit_code,
            stdout_tail=result.stdout_tail,
            stderr_tail=result.stderr_tail,
        )
    # async
    result = await execute_async(tool, args_model)
    return ToolRunAsyncWire(
        slice="v0-tool-async-1",
        tool_id=result.tool_id,
        job_id=result.job_id,
    )


@router.get(
    "/api/tools/jobs/{job_id}",
    response_model=ToolJobStatusWire,
)
async def api_tools_job_status(job_id: str) -> ToolJobStatusWire:
    """Poll for the status of an async tool job."""

    from cloris.tools_runtime import get_job_status

    status = await get_job_status(job_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "tool_job_not_found", "job_id": job_id},
        )
    return ToolJobStatusWire(
        slice="v0-tool-job-1",
        job_id=status.job_id,
        tool_id=status.tool_id,
        status=status.status,
        started_at=status.started_at,
        finished_at=status.finished_at,
        exit_code=status.exit_code,
        stdout_tail=status.stdout_tail,
        stderr_tail=status.stderr_tail,
        error_message=status.error_message,
    )


@router.get(
    "/api/resolve-legacy/workspace/{source}/{state_key}",
    response_model=LegacyResolveResponse,
)
def api_resolve_legacy_workspace(
    source: str, state_key: str
) -> LegacyResolveResponse:
    """Resolve a legacy workspace URL to its current brief_id."""

    brief_id = resolve_legacy_workspace(source, state_key)
    if brief_id is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "legacy_workspace_not_resolvable",
                "source": source,
                "state_key": state_key,
            },
        )
    return LegacyResolveResponse(brief_id=brief_id)


@router.get(
    "/api/resolve-legacy/candidate/{source}/{state_key}/{candidate_id}",
    response_model=LegacyResolveResponse,
)
def api_resolve_legacy_candidate(
    source: str, state_key: str, candidate_id: int
) -> LegacyResolveResponse:
    """Resolve a legacy candidate URL to its current brief_id.

    Same shape as the workspace resolver — the candidate_id stays
    unchanged across the rewrite, so the frontend just needs the
    brief_id to construct ``#/candidate/<brief_id>/<candidate_id>``.
    The ``candidate_id`` path segment is consumed for symmetry with the
    legacy URL and to enable a future variant that re-keys the id under
    a cross-module identity layer.
    """

    brief_id = resolve_legacy_workspace(source, state_key)
    if brief_id is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "legacy_candidate_not_resolvable",
                "source": source,
                "state_key": state_key,
                "candidate_id": candidate_id,
            },
        )
    return LegacyResolveResponse(brief_id=brief_id)


@dataclass(frozen=True)
class _SpawnResult:
    """Outcome of :func:`_spawn_linkedin_worker`.

    Decoupled from the response models because the route layer is what knows
    which response type to box this into (LaunchResponse vs ResumeResponse).
    Keeping the helper response-shape-agnostic means future modes can be added
    without forcing a new response type into the helper signature.
    """

    pid: int
    state_dir: Path
    worker_json_path: Path


def _frozen_worker_binary_path() -> Path | None:
    """Return the path to the frozen ``cloris-worker`` sibling binary,
    or ``None`` when not running inside a frozen .app.

    Phase 0 ``worker-binary`` slice. PyInstaller bundles the .app as
    ``Cloris.app`` with the main entry binary at
    ``Cloris.app/Contents/MacOS/Cloris``. The worker ships as a
    sibling binary at ``Cloris.app/Contents/MacOS/cloris-worker``
    (built from a separate PyInstaller spec that pulls in every
    orchestrator + dependency the worker needs at runtime).

    The api process can't ``python -m cloris.worker`` from a frozen
    .app because ``sys.executable`` is the main entry binary, not a
    Python interpreter — invoking it with ``-m`` would re-launch the
    UI. The sibling-binary path keeps the spawn pattern intact while
    using the right binary for each role.
    """

    if not getattr(sys, "frozen", False):
        return None
    candidate = Path(sys.executable).parent / "cloris-worker"
    if candidate.exists():
        return candidate
    log.warning(
        "cloris.api: frozen app detected but sibling cloris-worker not "
        "found at %s; falling back to python -m invocation. The launch "
        "will likely fail because the .app has no python interpreter.",
        candidate,
    )
    return None


def _build_worker_argv(
    *,
    source: str = "linkedin",
    brief_path: str,
    brief_id: str,
    state_dir: Path,
    mode: Literal["fresh", "resume"],
    fresh: bool = False,
) -> list[str]:
    """Compose argv for the detached worker.

    Two shapes:

    - Frozen .app (Phase 0 ``worker-binary`` slice): invoke the
      sibling ``cloris-worker`` binary directly. ``sys.executable`` is
      the UI binary, not a python interpreter.
    - Dev / source install (the long-standing path):
      ``[sys.executable, "-m", "cloris.worker", ...]``. Test fixtures
      that monkeypatch ``subprocess.Popen`` continue to observe this
      shape because ``getattr(sys, 'frozen', False)`` is false outside
      the frozen .app.

    Phase F Slice F1: ``--source`` threads the source name into the
    worker so the wrapper can dispatch to the right per-source
    orchestrator argv builder via :data:`cloris.launchers.LAUNCHERS`.
    Slice 3 callers without ``--source`` continue to default to
    LinkedIn at the wrapper boundary.
    """

    worker_bin = _frozen_worker_binary_path()
    if worker_bin is not None:
        argv = [str(worker_bin)]
    else:
        argv = [sys.executable, "-m", "cloris.worker"]

    argv.extend(
        [
            "--source",
            source,
            "--brief",
            brief_path,
            "--brief-id",
            brief_id,
            "--state-dir",
            str(state_dir),
        ]
    )
    if mode == "resume":
        argv.extend(["--mode", "resume"])
    if fresh:
        argv.append("--fresh")
    return argv


def _worker_stderr_log_path(state_dir: Path) -> Path:
    """Return the detached worker stderr log path for a state dir."""

    return state_dir / "worker.stderr.log"


def _spawn_worker_for_source(
    *,
    source: str,
    brief_path: Path,
    mode: Literal["fresh", "resume"],
    force_fresh: bool = False,
) -> _SpawnResult:
    """Spawn a detached worker for the given brief + source.

    Phase F Slice F1. Single source of truth for both launch and
    resume across every registered source. The per-source seam is the
    :data:`cloris.launchers.LAUNCHERS` registry, which maps
    ``source`` → ``(state_key_fn, state_dir_fn, orchestrator_argv_fn)``.

    Steps:

    1. Validate that ``brief_path`` exists on disk; raise
       :class:`BriefPathNotFoundError` if not (route maps to HTTP 400).
    2. Resolve the per-source state directory + brief id via the
       registry so the sidecar carries a truthful ``brief_id`` even
       before the orchestrator inserts a ``runs`` row.
    3. Pre-flight resume and fresh-over-resumable launches against the
       read model. Spawning a worker for "resume" when there's no
       pending work, or a flagless fresh worker over generated
       resumable state, would surface success in the UI before the
       worker silently exits.
    4. Probe any existing ``worker.json``: a present sidecar with an
       ``int`` ``pid`` field that is currently alive raises
       :class:`WorkerAlreadyRunningError` (route maps to HTTP 409). A
       missing sidecar, malformed sidecar, non-int ``pid``, or dead
       ``pid`` is treated as stale and silently overwritten by the new
       worker.
    5. Spawn ``python -m cloris.worker --source <source> ...`` with
       ``start_new_session=True`` so the worker survives the API
       process exiting and is its own process-group leader. Stdio is
       fully detached.
    6. Return a :class:`_SpawnResult` with ``pid`` from ``Popen.pid`` —
       the same PID will belong to the per-source orchestrator after
       the worker ``execvp``s.
    """

    from cloris.launchers import LAUNCHERS

    if source not in LAUNCHERS:
        raise UnknownSourceError(source=source, allowed=tuple(sorted(LAUNCHERS.keys())))

    # Reopen P7.1 — sunset/launchability gate. Fires before the pause
    # kill-switch and before any state dir / runs row is created. Unlike
    # the pause kill-switch below, ``force=true`` never bypasses this: a
    # retired subagent is not a transient unavailability the caller can
    # retry past. This is the single spawn choke point every launch and
    # resume path funnels through, so gating here (rather than only at
    # the route layer) closes it for every current and future caller.
    launcher_entry = LAUNCHERS[source]
    if not launcher_entry.launchable or launcher_entry.sunset:
        raise SourceSunsetError(source=source)

    # Reopen Y.5.2 + Y.5.6 — per-domain hard-pause kill-switch. Fires BEFORE
    # any state dir / runs row is created so a paused launch leaves nothing
    # behind. Additive: no-op when neither arm is set. Mirrors the
    # CLORIS_CERTIFY_STUB_RUNNERS truthy parse below. The raise propagates
    # to a clean 409 via each launch entry's typed-allowlist except chain.
    #
    # Two arms, OR'd (Y.5.6 / F1):
    #   1. ENV arm (Y.5.2, FIRST) — ``CLORIS_PAUSE_LAUNCHES_<SOURCE>`` truthy.
    #      Process-env only; armable where the API server's env lives.
    #   2. PERSISTED arm (Y.5.6) — a durable ``source_pause`` row in the
    #      orchestration DB, armed OUT-OF-PROCESS by an operator (CLI/admin).
    #      The in-process gate reads it on the NEXT spawn, so a pause armed
    #      after the server booted is honored without a restart or env edit.
    #
    # The env arm is evaluated FIRST and short-circuits (Y.5.2's contract is
    # byte-for-byte preserved: env truthy => raise, the persisted store is not
    # even opened). The persisted read fails closed: if the pause state cannot
    # be read, the launch is blocked with an operator-facing 503.
    paused = os.getenv(
        f"CLORIS_PAUSE_LAUNCHES_{source.upper()}", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not paused:
        try:
            from shared.output_paths import resolve_orchestration_db_path
            from shared.runtime_state.orchestration_store import (
                OrchestrationStateStore,
            )

            paused = OrchestrationStateStore(
                resolve_orchestration_db_path()
            ).is_source_paused(source)
        except Exception as exc:  # noqa: BLE001 — kill switch must fail closed
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "pause_state_unavailable",
                    "source": source,
                    "message": (
                        "Pause state could not be read; check the orchestration "
                        "DB and retry."
                    ),
                },
            ) from exc
    if paused:
        raise DomainPausedError(source=source)

    if not brief_path.exists():
        raise BriefPathNotFoundError(str(brief_path))

    launcher = LAUNCHERS[source]
    state_dir = launcher.state_dir_fn(str(brief_path))
    brief_id = launcher.state_key_fn(str(brief_path))

    # Phase 1.3: pre-flight resume against the read model before the
    # certification and live paths diverge.
    if mode == "resume":
        pending = read_models.has_pending_work(state_dir)
        if pending is False:
            raise NoPendingWorkError(state_dir=str(state_dir))
    elif not force_fresh:
        generated_artifacts = (
            state_dir / "preflight_v2_brief.json",
            state_dir / "execution_plan.json",
        )
        if any(path.exists() for path in generated_artifacts):
            pending = read_models.has_pending_work(state_dir)
            if pending is not False:
                raise FreshOverResumableStateError(state_dir=str(state_dir))

    if os.getenv("CLORIS_CERTIFY_STUB_RUNNERS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        from cloris.worker import build_sidecar, write_sidecar
        from shared.runtime_state.store import RuntimeStateStore

        state_dir.mkdir(parents=True, exist_ok=True)
        store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
        run_id = store.start_run(
            source=source,
            brief_id=brief_id,
            output_dir=str(state_dir),
            mode=mode,
            resume_state={"brief_name": brief_id},
            brief_path_at_launch=str(brief_path),
        )
        worker_json_path = write_sidecar(
            state_dir,
            build_sidecar(
                source=source,
                brief_id=brief_id,
                brief_path=str(brief_path),
                output_dir=str(state_dir),
                mode=mode,
                input_mode="certification",
                started_at=datetime.now(timezone.utc).isoformat(),
                pid=0,
                run_id=run_id,
            ),
        )
        return _SpawnResult(
            pid=0,
            state_dir=state_dir,
            worker_json_path=worker_json_path,
        )

    # Phase 1.1: serialize read-sidecar + spawn across Cloris UI processes
    # on the same machine so two simultaneous launches cannot race-spawn two
    # workers for the same state dir. The lock release waits until the
    # spawned worker has written its sidecar (wait_for_sidecar), closing
    # the residual race between Popen-return and the worker's first write.
    with state_dir_launch_lock(state_dir, timeout=DEFAULT_LAUNCH_LOCK_TIMEOUT_S):
        existing = read_sidecar(state_dir)
        if existing is not None:
            existing_pid = existing.get("pid")
            if (
                isinstance(existing_pid, int)
                and not isinstance(existing_pid, bool)
                and is_pid_alive(existing_pid)
            ):
                # Terminal runs can leave a live sidecar while final report or
                # failed recovery cleanup is draining. Canonical SQLite has
                # already ended the launchable work, so clear the sidecar
                # before deciding whether this is truly active.
                if reconciler.cleanup_drainable_terminal_lock(state_dir):
                    existing = read_sidecar(state_dir)
                    existing_pid = existing.get("pid") if existing else None
                if (
                    isinstance(existing_pid, int)
                    and not isinstance(existing_pid, bool)
                    and is_pid_alive(existing_pid)
                ):
                    existing_brief_id = existing.get("brief_id") if existing else None
                    if existing_brief_id == brief_id:
                        return _SpawnResult(
                            pid=existing_pid,
                            state_dir=state_dir,
                            worker_json_path=state_dir / "worker.json",
                        )
                    raise WorkerAlreadyRunningError(
                        pid=existing_pid,
                        state_dir=str(state_dir),
                    )

        argv = _build_worker_argv(
            source=source,
            brief_path=str(brief_path),
            brief_id=brief_id,
            state_dir=state_dir,
            mode=mode,
            fresh=(mode == "fresh" and force_fresh),
        )
        stderr_log_path = _worker_stderr_log_path(state_dir)
        with stderr_log_path.open("ab") as stderr_log:
            process = subprocess.Popen(
                argv,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=stderr_log,
                stderr=stderr_log,
                close_fds=True,
            )

        # Wait for the worker's first sidecar write before releasing the
        # lock. Bounded: if the worker fails before writing, the next
        # launcher sees no sidecar and proceeds normally. Reopen P7.3 —
        # the return value is load-bearing, not decorative: a ``False``
        # here means we never observed the worker actually start, so the
        # caller must not report a 201-with-pid success it never saw.
        sidecar_observed = wait_for_sidecar(
            state_dir,
            expected_pid=process.pid,
            timeout=DEFAULT_SIDECAR_WAIT_TIMEOUT_S,
        )
        if not sidecar_observed:
            raise WorkerDidNotStartError(
                source=source, state_dir=str(state_dir), pid=process.pid
            )

    return _SpawnResult(
        pid=process.pid,
        state_dir=state_dir,
        worker_json_path=state_dir / "worker.json",
    )


def _spawn_linkedin_worker(
    req: LaunchLinkedInRequest,
    *,
    mode: Literal["fresh", "resume"],
) -> _SpawnResult:
    """Backward-compat shim around :func:`_spawn_worker_for_source`.

    Phase F Slice F1 generalized the spawn helper. Existing test
    fixtures import ``_spawn_linkedin_worker`` directly; this shim
    keeps them working while the new code path is what production
    actually uses.
    """

    brief_path = _paths.resolve_brief_path_contained(req.brief_path)
    return _spawn_worker_for_source(
        source="linkedin",
        brief_path=brief_path,
        mode=mode,
        force_fresh=req.force_fresh,
    )


def launch_linkedin_worker(req: LaunchLinkedInRequest) -> LaunchResponse:
    """Spawn a detached LinkedIn worker in fresh mode (legacy synonym).

    Phase F Slice F1 keeps this helper as a thin wrapper over the
    generalized :func:`_spawn_worker_for_source` so the legacy
    ``POST /api/launch/linkedin`` endpoint keeps its byte-for-byte
    contract while the new ``POST /api/launch/{source}`` endpoint
    consumes the same spawner.
    """

    brief_path = _paths.resolve_brief_path_contained(req.brief_path)
    result = _spawn_worker_for_source(
        source="linkedin",
        brief_path=brief_path,
        mode="fresh",
        force_fresh=req.force_fresh,
    )
    return LaunchResponse(
        source="linkedin",
        input_mode="concurrent",
        mode="fresh",
        pid=result.pid,
        state_dir=str(result.state_dir),
        worker_json_path=str(result.worker_json_path),
    )


def resume_linkedin_worker(req: LaunchLinkedInRequest) -> ResumeResponse:
    """Spawn a detached LinkedIn worker in resume mode (legacy synonym).

    See :func:`launch_linkedin_worker`. Same generalized spawner;
    different mode + response shape so the legacy
    ``POST /api/resume/linkedin`` endpoint keeps its existing contract.
    """

    brief_path = _paths.resolve_brief_path_contained(req.brief_path)
    result = _spawn_worker_for_source(
        source="linkedin",
        brief_path=brief_path,
        mode="resume",
        force_fresh=req.force_fresh,
    )
    return ResumeResponse(
        source="linkedin",
        input_mode="concurrent",
        pid=result.pid,
        state_dir=str(result.state_dir),
        worker_json_path=str(result.worker_json_path),
    )


def _resolve_state_dir(source: str, state_key: str) -> Optional[Path]:
    """Return the discovered state dir matching ``(source, state_key)``.

    Iterates :func:`cloris.control_plane.enumerate_state_dirs` rather than
    naively joining ``STATE_ROOT / source / state_key``. This is the
    path-traversal-safe lookup: any ``state_key`` that doesn't appear in
    the discovered set returns ``None``, which the route translates to
    HTTP 404.
    """

    for discovered_source, discovered_state_dir in enumerate_state_dirs():
        if discovered_source == source and discovered_state_dir.name == state_key:
            return discovered_state_dir
    return None


def stop_worker(source: str, state_key: str) -> StopResponse:
    """Resolve the state dir, classify the worker, optionally SIGTERM.

    Pure helper — no FastAPI imports beyond the type contract. Returns a
    :class:`cloris.models.StopResponse`; the route is responsible for
    setting the HTTP status code (202 when ``worker_state == "stopping"``,
    200 otherwise).

    Steps:

    1. Resolve ``state_dir`` via :func:`_resolve_state_dir`. If not
       found, raise :class:`StateDirNotFoundError`.
    2. Read ``worker.json`` via :func:`cloris.worker.read_sidecar`.
    3. Missing sidecar ⇒ ``worker_state="missing"``, ``pid=None``. **No
       signal sent.**
    4. Sidecar present, but ``pid`` is not a valid int OR
       :func:`cloris.worker.is_pid_alive` returns ``False`` ⇒
       ``worker_state="stale"``. **No signal sent. Sidecar not deleted.**
    5. Alive PID ⇒ ``os.kill(pid, signal.SIGTERM)`` exactly once and
       return ``worker_state="stopping"``. ``ProcessLookupError`` (race:
       process died between probe and signal) ⇒ treat as stale.
       ``PermissionError`` (we cannot signal it but it exists) ⇒ also
       treat as stale; the user-facing contract is "stop is best-effort
       against the worker we own".

    Slice 4 deliberately does **not** wait for SIGTERM to take effect
    and does **not** escalate to SIGKILL. Both are explicit non-goals
    per the operative plan.
    """

    state_dir = _resolve_state_dir(source, state_key)
    if state_dir is None:
        raise StateDirNotFoundError(source, state_key)

    sidecar = read_sidecar(state_dir)
    if sidecar is None:
        return StopResponse(
            source=source,  # type: ignore[arg-type]
            state_key=state_key,
            state_dir=str(state_dir),
            worker_state="missing",
            pid=None,
        )

    pid_raw = sidecar.get("pid")
    if not isinstance(pid_raw, int) or isinstance(pid_raw, bool):
        return StopResponse(
            source=source,  # type: ignore[arg-type]
            state_key=state_key,
            state_dir=str(state_dir),
            worker_state="stale",
            pid=None,
        )

    if not is_pid_alive(pid_raw):
        return StopResponse(
            source=source,  # type: ignore[arg-type]
            state_key=state_key,
            state_dir=str(state_dir),
            worker_state="stale",
            pid=pid_raw,
        )

    try:
        _send_sigterm(pid_raw)
    except ProcessLookupError:
        return StopResponse(
            source=source,  # type: ignore[arg-type]
            state_key=state_key,
            state_dir=str(state_dir),
            worker_state="stale",
            pid=pid_raw,
        )
    except PermissionError:
        return StopResponse(
            source=source,  # type: ignore[arg-type]
            state_key=state_key,
            state_dir=str(state_dir),
            worker_state="stale",
            pid=pid_raw,
        )

    return StopResponse(
        source=source,  # type: ignore[arg-type]
        state_key=state_key,
        state_dir=str(state_dir),
        worker_state="stopping",
        pid=pid_raw,
    )


def _orchestrator_order_context(brief: object, plan) -> str:
    """Human-readable dispatch explanation for the launch surface."""

    from cloris.chief_of_staff.decision import DispatchPlan

    if not isinstance(plan, DispatchPlan):
        return ""

    tm = getattr(brief, "target_modules", None)
    declared: list[str] = []
    if isinstance(tm, list):
        declared = [
            str(m).strip() for m in tm if isinstance(m, str) and str(m).strip()
        ]
    planned = [s.module_name for s in plan.steps]
    if declared and planned == declared:
        return (
            "Same order as the modules listed in your brief. "
            "You can still turn modules off or match your brief order only."
        )
    gates = [
        str(s.handoff_condition).strip()
        for s in plan.steps
        if s.handoff_condition and str(s.handoff_condition).strip()
    ]
    if gates:
        return (
            "Chief of staff suggested this sequence. "
            "Gates between steps: " + "; ".join(gates)
        )
    return (
        "Chief of staff suggested this run order "
        "(can differ from your brief list). Reorder with the chips, "
        "or click Use brief order."
    )


@router.post(
    "/api/orchestrator/decide",
    response_model=OrchestratorDecideResponse,
)
def api_orchestrator_decide(
    body: OrchestratorDecideRequest,
) -> OrchestratorDecideResponse:
    """Heuristic chief-of-staff dispatch preview (Slice 2.7 + 4.1 / 4.2).

    Uses :meth:`ChiefOfStaffAgent.dispatch` without persisting orchestration
    rows so intake / launch surfaces can poll recommendations without filling
    ``chief_of_staff_runs``.
    """

    from cloris.chief_of_staff.agent import ChiefOfStaffAgent

    if body.brief_path is None and body.partial_brief is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "orchestrator_decide_payload_required",
                "message": "Provide brief_path or partial_brief.",
            },
        )

    brief = _brief_for_orchestrator_decide(body)
    plan = ChiefOfStaffAgent().dispatch(brief, prior_runs=[], persist=False)
    return OrchestratorDecideResponse(
        dispatch_plan=OrchestratorDispatchPlanWire.model_validate(plan.to_dict()),
        order_context=_orchestrator_order_context(brief, plan),
    )


def _chief_of_staff_synthesis_for_reflection_session(
    *, reflection_session_id: int
):
    """Re-run v1 chief-of-staff synthesis from a reflection session's state.

    Returns a :class:`~cloris.chief_of_staff.agent.ChiefOfStaffSynthesis`, or
    ``None`` if the session id does not exist. Raises :class:`HTTPException`
    with status 422 when stored state cannot satisfy synthesis inputs
    (missing paths, fewer than two candidate-producing sources, etc.).
    """

    from dataclasses import asdict, fields

    from cloris.chief_of_staff.agent import ChiefOfStaffAgent
    from market_intelligence.agent_backends import PlannerResult
    from market_intelligence.briefing_polish import (
        BriefingPolishBackend,
        HeuristicBriefingBackend,
    )
    from market_intelligence.engine import (
        _build_deterministic_summary,
        _collect_evidence_batches,
        _load_previous_artifact,
        _load_previous_agent_state,
        resolve_market_intel_agent_state_path,
        resolve_market_intel_artifact_path,
    )
    from market_intelligence.reflection import (
        _contributing_sources_count,
        _per_source_signals_from_batches,
    )
    from market_intelligence.schema import MarketIdentity
    from shared.brief_loader import load_brief
    from shared.runtime_state import reflection as reflection_store
    from shared.storage import read_json

    store = _intake_store()
    session = reflection_store.get_reflection_session(
        store, session_id=reflection_session_id
    )
    if session is None:
        return None

    state_json = session.get("state_json") or {}
    if not isinstance(state_json, dict):
        state_json = {}

    context = state_json.get("context") or {}
    if not isinstance(context, dict):
        context = {}

    brief_path_s = context.get("brief_path")
    if not isinstance(brief_path_s, str) or not brief_path_s.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "synthesis_preconditions_not_met",
                "message": "Reflection state lacks brief_path in context.",
            },
        )

    brief_path = Path(brief_path_s)
    raw = read_json(brief_path)
    brief = load_brief(str(brief_path))

    run_dir_raw = context.get("run_dir")
    run_dir_path: Path | None = None
    if isinstance(run_dir_raw, str) and run_dir_raw.strip():
        run_dir_path = Path(run_dir_raw)

    mode = context.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        mode = "post_run"

    mi = context.get("market_identity")
    if not isinstance(mi, dict):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "synthesis_preconditions_not_met",
                "message": "Reflection state lacks market_identity in context.",
            },
        )
    market_identity = MarketIdentity.from_dict(mi)

    artifact_path = resolve_market_intel_artifact_path(
        brief_path, output_dir=run_dir_path
    )
    agent_state_path = resolve_market_intel_agent_state_path(
        brief_path, output_dir=run_dir_path
    )
    previous_artifact = _load_previous_artifact(artifact_path)
    _ = _load_previous_agent_state(agent_state_path)

    evidence_batches = _collect_evidence_batches(
        brief_path=brief_path,
        brief=brief,
        raw=raw,
        mode=mode,
        run_dir=run_dir_path,
        report_path=None,
        previous_artifact=previous_artifact,
        reconstruct_report_analysis=(mode == "backfill"),
    )

    if _contributing_sources_count(evidence_batches) < 2:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "synthesis_preconditions_not_met",
                "message": (
                    "Synthesis requires at least two candidate-producing "
                    "sources for this brief."
                ),
            },
        )

    per_source_signals = _per_source_signals_from_batches(evidence_batches)

    deterministic_summary = _build_deterministic_summary(
        market_identity=market_identity,
        evidence_batches=evidence_batches,
        previous_artifact=previous_artifact,
    )

    plan_block = (state_json.get("phase_outputs") or {}).get("plan") or {}
    if not isinstance(plan_block, dict):
        plan_block = {}
    planner_dict = plan_block.get("planner_result") or {}
    if not isinstance(planner_dict, dict):
        planner_dict = {}

    base_planner = asdict(PlannerResult())
    merged_planner = {**base_planner, **planner_dict}
    planner_result = PlannerResult(
        **{f.name: merged_planner[f.name] for f in fields(PlannerResult)}
    )

    steering_notes: list[str] = []
    for entry in state_json.get("steering_history") or []:
        if isinstance(entry, dict) and isinstance(entry.get("note"), str):
            steering_notes.append(entry["note"])

    briefing_backend = BriefingPolishBackend(fallback=HeuristicBriefingBackend())
    editorial = briefing_backend.polish(
        market_identity=market_identity,
        deterministic_summary=deterministic_summary,
        planner_result=planner_result,
        steering_notes=steering_notes,
    )

    agent = ChiefOfStaffAgent()
    return agent.synthesize(
        market_identity=market_identity,
        per_source_signals=per_source_signals,
        briefing_paragraph=editorial.paragraph,
        deterministic_summary=deterministic_summary,
    )


@router.post(
    "/api/orchestrator/synthesize",
    response_model=OrchestratorSynthesizeResponse,
)
def api_orchestrator_synthesize(
    body: OrchestratorSynthesizeRequest,
) -> OrchestratorSynthesizeResponse:
    """Re-run v1 chief-of-staff synthesis for a persisted reflection session."""

    synthesis = _chief_of_staff_synthesis_for_reflection_session(
        reflection_session_id=body.reflection_session_id,
    )
    if synthesis is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "reflection_session_not_found",
                "reflection_session_id": body.reflection_session_id,
            },
        )
    payload = synthesis.to_dict()
    return OrchestratorSynthesizeResponse(
        paragraph=str(payload.get("paragraph", "")),
        per_specialist_weight=dict(payload.get("per_specialist_weight") or {}),
        priority_for_principal=str(payload.get("priority_for_principal", "")),
        confidence=float(payload.get("confidence", 0.0)),
        source=str(payload.get("source", "")),
    )


@router.get(
    "/api/orchestrator/{brief_id}/runs",
    response_model=OrchestratorRunsResponse,
)
def api_orchestrator_runs(brief_id: str) -> OrchestratorRunsResponse:
    """List persisted chief-of-staff runs for a brief (most recent first)."""

    import json

    from shared.output_paths import resolve_orchestration_db_path

    records = read_models.chief_of_staff_runs_for_brief(
        resolve_orchestration_db_path(),
        brief_id=brief_id,
    )
    out: list[OrchestratorRunRecord] = []
    for rec in records:
        try:
            dispatch_plan = json.loads(rec.dispatch_plan_json)
        except (json.JSONDecodeError, TypeError):
            dispatch_plan = {}
        if not isinstance(dispatch_plan, dict):
            dispatch_plan = {}
        try:
            invocation_order = json.loads(rec.invocation_order_json)
        except (json.JSONDecodeError, TypeError):
            invocation_order = []
        if not isinstance(invocation_order, list):
            invocation_order = []
        invocation_order = [str(x) for x in invocation_order]
        try:
            handoff_payloads = json.loads(rec.handoff_payloads_json)
        except (json.JSONDecodeError, TypeError):
            handoff_payloads = {}
        if not isinstance(handoff_payloads, dict):
            handoff_payloads = {}
        try:
            synthesis_output = json.loads(rec.synthesis_output_json)
        except (json.JSONDecodeError, TypeError):
            synthesis_output = {}
        if not isinstance(synthesis_output, dict):
            synthesis_output = {}
        out.append(
            OrchestratorRunRecord(
                id=rec.id,
                brief_id=rec.brief_id,
                principal_id=rec.principal_id,
                status=rec.status,
                dispatch_plan=dispatch_plan,
                invocation_order=invocation_order,
                handoff_payloads=handoff_payloads,
                synthesis_output=synthesis_output,
                started_at=rec.started_at,
                ended_at=rec.ended_at,
            )
        )
    return OrchestratorRunsResponse(runs=out)


@router.get(
    "/api/launch-readiness/{source}/{brief_id:path}",
    response_model=LaunchReadinessResponse,
)
def api_launch_readiness(source: str, brief_id: str) -> LaunchReadinessResponse:
    """Phase D Slice D9 (Ledger L4). Per-source launch-readiness probe.

    Surfaces blockers BEFORE worker spawn (LinkedIn browser session,
    GitHub token scope, missing per-brief save destinations). The
    recruiter sees specific remediation — "open linkedin.com/talent in
    a tab", "fill in the LinkedIn project ID under Where Cloris
    saves" — instead of generic launch failure after the worker dies.

    Phase F Slice F2 layered brief-level readiness on top of the
    source-level probe via the same ``_readiness_blockers`` aggregator
    F1's launch path uses, so this read endpoint and the launch path
    share one truth.

    Errors:
      * 422 ``unknown_source`` — ``source`` is not one of the
        registered modules (currently linkedin / github).
    """

    from cloris.launchers import LAUNCHERS

    if source not in LAUNCHERS:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unknown_source",
                "source": source,
                "allowed": sorted(LAUNCHERS.keys()),
            },
        )

    blockers = _readiness_blockers(source, brief_id)
    return LaunchReadinessResponse(
        source=source,  # type: ignore[arg-type]
        brief_id=brief_id,
        ready=len(blockers) == 0,
        blockers=[
            LaunchReadinessBlocker(
                kind=b.kind,
                message=b.message,
                remediation=b.remediation,
                code=getattr(b, "code", ""),
            )
            for b in blockers
        ],
    )


# ---------------------------------------------------------------------------
# Phase F Slice F1 — Generic POST /api/launch/{source} + collapsed resume.
# ---------------------------------------------------------------------------
#
# The canonical launch endpoint as of Phase F. Accepts ``{brief_id, mode,
# force?}`` and dispatches to the per-source spawn function via
# :data:`cloris.launchers.LAUNCHERS`. Mode "resume" is folded into the same
# route so the legacy ``POST /api/resume/linkedin`` endpoint is just a
# synonym. The legacy ``POST /api/launch/linkedin`` likewise translates
# ``brief_path → brief_id`` and calls this same code path.


def _resolve_brief_path_or_raise(brief_id: str) -> Path:
    """Resolve a brief_id to its on-disk path or raise BriefIdNotFoundError.

    Phase F Slice F1. The brief_id is the universal Cloris brief
    identifier produced by the existing ``derive_brief_id`` hash
    (which reads brief CONTENT, not path, so the value stays stable
    across flat→nested migrations regardless of source).
    """

    import sys

    api_pkg = sys.modules.get("cloris.api")
    if api_pkg is not None:
        override = getattr(api_pkg, "_resolve_brief_path_or_raise", None)
        if override is not None and override is not _resolve_brief_path_or_raise:
            return override(brief_id)

    resolved = _resolve_brief_by_id(brief_id)
    if resolved is None:
        raise BriefIdNotFoundError(brief_id)
    abs_path, _was_flat = resolved
    return abs_path


def _readiness_blockers(source: str, brief_id: str) -> list:
    """Aggregate launch-readiness blockers across two layers.

    Phase D Slice D9 introduced source-level readiness probes (auth /
    config / net) at :mod:`linkedin.health` and :mod:`github.health`.
    Phase F Slice F2 layers brief-level readiness on top via
    :data:`cloris.launchers.LAUNCHERS[source].save_destination_blocker_fn`.

    Both layers are aggregated AND-style: any blocker on either layer
    blocks the launch (unless ``force=true`` at the caller). The two
    layers are kept separate so source-level probes stay brief-agnostic
    (they just check "can we connect at all?") and brief-level checks
    stay source-agnostic at the registry boundary.

    Returns an empty list when both layers are clear. Sources unknown
    to the probe return an empty list (probe doesn't know the source
    ⇒ caller handled UnknownSourceError already; defensive).
    """

    blockers: list = []

    from cloris.launchers import LAUNCHERS
    from shared import config

    if source == "linkedin" and config.CLORIS_TRIAL_MODE:
        from cloris.anthropic_health import launch_readiness_blocker

        anthropic_blocker = launch_readiness_blocker()
        if anthropic_blocker is not None:
            blockers.append(anthropic_blocker)

    # Layer 1 — source-level readiness (Phase D D9). Multi-agent
    # execution Phase 1 Slice 1.1 routed linkedin / github through
    # ``LAUNCHERS[source].readiness_probe_fn``; Phase 2.2 wired the
    # remaining three sources (researcher / designer / exec_search)
    # via per-module ``probe_<source>_readiness`` functions. The
    # ``probe_fn is None`` short-circuit is retained defensively for
    # the unknown-source path (``LAUNCHERS.get(source)`` returns
    # ``None``); every registered source now supplies a probe so the
    # aggregator surfaces real blockers uniformly.
    launcher = LAUNCHERS.get(source)
    probe_fn = launcher.readiness_probe_fn if launcher is not None else None
    report = probe_fn() if probe_fn is not None else None

    if report is not None and not report.ready:
        blockers.extend(report.blockers)

    # Layer 2 — brief-level readiness (Phase F F2). Resolve the brief
    # path if possible; if it can't be resolved (bogus brief_id), let
    # the layer-2 check pass through — the launch handler will surface
    # a 404 separately.
    if launcher is not None:
        try:
            resolved = _resolve_brief_by_id(brief_id)
        except Exception:
            resolved = None
        if resolved is not None:
            abs_path, _was_flat = resolved
            try:
                brief_blocker = launcher.save_destination_blocker_fn(
                    str(abs_path)
                )
            except Exception:
                brief_blocker = None
            if brief_blocker is not None:
                blockers.append(brief_blocker)

            if source == "linkedin":
                from cloris.launchers import linkedin_permanent_filter_automation_blockers

                blockers.extend(
                    linkedin_permanent_filter_automation_blockers(str(abs_path))
                )

    return blockers


def _link_recruiter_brief_fail_soft(brief_id: str | None) -> None:
    """Record acting-recruiter ownership of a brief; never raise.

    Reopen Stage 2 (Part III). Called after a successful worker spawn so
    the recruiter — Cloris's durable entity — accretes brief membership.
    Fail-soft by construction: every error is swallowed because the
    launch has already succeeded by the time this runs, and a
    recruiter-store problem must not surface as a launch failure. The
    recruiter_id flows through the ``get_current_recruiter_id`` seam, and
    ``link_brief`` is idempotent.
    """

    if not brief_id:
        return
    try:
        from shared.output_paths import resolve_recruiter_db_path
        from shared.recruiter_context import get_current_recruiter_id
        from shared.runtime_state.recruiter_store import RecruiterStore

        store = RecruiterStore(resolve_recruiter_db_path())
        store.link_brief(get_current_recruiter_id(), brief_id)
    except Exception:  # noqa: BLE001 — fail-soft; launch already succeeded
        return


def _launch_for_source_impl(source: str, req: LaunchRequest) -> LaunchResponse:
    """Phase F Slice F1 — generic launch dispatch (handler implementation).

    Extracted from the route decorator so the route registration can be
    moved AFTER the legacy literal routes (Starlette matches in route-
    declaration order; a path-param route declared first would shadow
    ``/api/launch/linkedin``). The actual ``@router.post`` for this
    handler lives at the bottom of the file, after the legacy routes.

    See route docstring (``launch_for_source``) for the full contract.
    """

    try:
        from cloris.launchers import LAUNCHERS

        if source not in LAUNCHERS:
            raise UnknownSourceError(
                source=source, allowed=tuple(sorted(LAUNCHERS.keys()))
            )

        brief_path = _resolve_brief_path_or_raise(req.brief_id)

        if not req.force:
            blockers = _readiness_blockers(source, req.brief_id)
            if blockers:
                raise LaunchNotReadyError(source=source, blockers=blockers)

        result = _spawn_worker_for_source(
            source=source,
            brief_path=brief_path,
            mode=req.mode,
            force_fresh=req.force_fresh,
        )

        # Reopen Stage 2: record that the acting recruiter owns this brief.
        # Additive + fail-soft — a recruiter-store hiccup must never turn a
        # successful worker spawn into a launch failure. ``link_brief`` is
        # idempotent (INSERT OR IGNORE), and the recruiter_id flows through
        # the ``get_current_recruiter_id`` seam (NOT a hardcoded 1) so
        # Phase-2 auth swaps one function. The brief membership key is
        # ``req.brief_id`` — the same universal brief id the launch
        # resolved against.
        _link_recruiter_brief_fail_soft(req.brief_id)
    except UnknownSourceError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unknown_source",
                "source": exc.source,
                "allowed": list(exc.allowed),
            },
        ) from exc
    except BriefIdNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "brief_id_not_found", "brief_id": exc.brief_id},
        ) from exc
    except LaunchNotReadyError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "launch_not_ready",
                "source": exc.source,
                "blockers": [
                    {
                        "kind": b.kind,
                        "message": b.message,
                        "remediation": b.remediation,
                        "code": getattr(b, "code", ""),
                    }
                    for b in exc.blockers
                ],
            },
        ) from exc
    except BriefPathNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "brief_path_not_found", "brief_path": str(exc)},
        ) from exc
    except WorkerAlreadyRunningError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "worker_already_running",
                "pid": exc.pid,
                "state_dir": exc.state_dir,
            },
        ) from exc
    except DomainPausedError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "domain_paused",
                "source": exc.source,
                "message": str(exc),
            },
        ) from exc
    except SourceSunsetError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "source_sunset",
                "source": exc.source,
                "message": str(exc),
            },
        ) from exc
    except NoPendingWorkError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "no_pending_work", "state_dir": exc.state_dir},
        ) from exc
    except FreshOverResumableStateError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "fresh_over_resumable_state",
                "state_dir": exc.state_dir,
                "message": (
                    "State dir carries a generated brief and resumable "
                    "pending work. Resume the existing run, or retry the "
                    "fresh launch with force_fresh=true."
                ),
            },
        ) from exc
    except LaunchLockTimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "launch_lock_timeout",
                "state_dir": exc.state_dir,
                "timeout_s": exc.timeout,
            },
        ) from exc
    except WorkerDidNotStartError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "worker_did_not_start",
                "source": exc.source,
                "state_dir": exc.state_dir,
                "pid": exc.pid,
                "message": "worker did not start",
            },
        ) from exc

    return LaunchResponse(
        source=source,  # type: ignore[arg-type]
        input_mode="concurrent",
        mode=req.mode,
        pid=result.pid,
        state_dir=str(result.state_dir),
        worker_json_path=str(result.worker_json_path),
    )


@router.post("/api/launch/linkedin", status_code=201, response_model=LaunchResponse)
def launch_linkedin(req: LaunchLinkedInRequest) -> LaunchResponse:
    """Spawn a detached LinkedIn worker; map typed errors to HTTP codes.

    Phase F Slice F1: legacy synonym route. Kept for backward compat
    with clients posting ``{brief_path}``. Internally calls the same
    spawn helper as the canonical ``POST /api/launch/{source}``.
    Deprecation window: ~6 months from F1 ship.

    - :class:`BriefPathNotFoundError` → HTTP 400 with
      ``{"error": "brief_path_not_found", "brief_path": "..."}``.
    - :class:`WorkerAlreadyRunningError` → HTTP 409 with
      ``{"error": "worker_already_running", "pid": ..., "state_dir": "..."}``.
    - :class:`FreshOverResumableStateError` → HTTP 422 with
      remediation guidance to resume or retry with ``force_fresh=true``.
    """

    try:
        return launch_linkedin_worker(req)
    except BriefPathNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "brief_path_not_found", "brief_path": str(exc)},
        ) from exc
    except WorkerAlreadyRunningError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "worker_already_running",
                "pid": exc.pid,
                "state_dir": exc.state_dir,
            },
        ) from exc
    except FreshOverResumableStateError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "fresh_over_resumable_state",
                "state_dir": exc.state_dir,
                "message": (
                    "State dir carries a generated brief and resumable "
                    "pending work. Resume the existing run, or retry the "
                    "fresh launch with force_fresh=true."
                ),
            },
        ) from exc
    except DomainPausedError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "domain_paused",
                "source": exc.source,
                "message": str(exc),
            },
        ) from exc
    except LaunchLockTimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "launch_lock_timeout",
                "state_dir": exc.state_dir,
                "timeout_s": exc.timeout,
            },
        ) from exc
    except WorkerDidNotStartError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "worker_did_not_start",
                "source": exc.source,
                "state_dir": exc.state_dir,
                "pid": exc.pid,
                "message": "worker did not start",
            },
        ) from exc


@router.post(
    "/api/resume/linkedin", status_code=201, response_model=ResumeResponse
)
def resume_linkedin(req: LaunchLinkedInRequest) -> ResumeResponse:
    """Spawn a detached LinkedIn worker in resume mode.

    Error mapping (Phase 1.3 adds the 422 case):

    - :class:`BriefPathNotFoundError` → HTTP 400.
    - :class:`WorkerAlreadyRunningError` → HTTP 409.
    - :class:`NoPendingWorkError` → HTTP 422 (no pending work — pre-flight
      from the canonical read model rejected this resume before any
      worker was spawned).
    - :class:`LaunchLockTimeoutError` → HTTP 503.
    """

    try:
        return resume_linkedin_worker(req)
    except BriefPathNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "brief_path_not_found", "brief_path": str(exc)},
        ) from exc
    except WorkerAlreadyRunningError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "worker_already_running",
                "pid": exc.pid,
                "state_dir": exc.state_dir,
            },
        ) from exc
    except DomainPausedError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "domain_paused",
                "source": exc.source,
                "message": str(exc),
            },
        ) from exc
    except NoPendingWorkError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "no_pending_work",
                "state_dir": exc.state_dir,
            },
        ) from exc
    except LaunchLockTimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "launch_lock_timeout",
                "state_dir": exc.state_dir,
                "timeout_s": exc.timeout,
            },
        ) from exc
    except WorkerDidNotStartError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "worker_did_not_start",
                "source": exc.source,
                "state_dir": exc.state_dir,
                "pid": exc.pid,
                "message": "worker did not start",
            },
        ) from exc


@router.post(
    "/api/launch/multi",
    status_code=201,
    response_model=LaunchMultiResponse,
)
def launch_multi(req: LaunchMultiRequest, response: Response) -> LaunchMultiResponse:
    """Audit Move #8 — atomic multi-module launch endpoint.

    Spawns one worker per ``modules`` entry, server-side, in the
    declared order. Closes the browser-crash-mid-launch failure mode
    where the frontend's per-source loop could leave a half-started
    run with no backend resume affordance.

    Per-module spawn errors do NOT abort the rest — each module's
    outcome lands in ``results`` independently:

    - Successful spawn → ``LaunchMultiPerSourceResult`` with
      ``launch`` populated (the same :class:`LaunchResponse` shape
      the per-source endpoint returns).
    - Typed spawn error (``UnknownSourceError``,
      ``BriefIdNotFoundError``, ``LaunchNotReadyError``,
      ``BriefPathNotFoundError``, ``WorkerAlreadyRunningError``,
      ``NoPendingWorkError``, ``FreshOverResumableStateError``,
      ``LaunchLockTimeoutError``) →
      ``LaunchMultiPerSourceResult`` with ``error`` populated. The
      error envelope's ``kind`` matches the per-source endpoint's
      ``error`` value (``"launch_not_ready"``,
      ``"worker_already_running"``, etc.) so the frontend renders
      it with the same prose.

    HTTP status code:
    - 201 when at least one module spawned successfully.
    - 422 when every module's spawn raised a typed error (the body
      still carries the per-module error envelopes).
    - 400 when ``modules`` is empty (no work to do).

    Route registered BEFORE ``/api/launch/{source}`` (the path-param
    route) so Starlette matches the literal ``multi`` path first.
    Same precedence trick :func:`launch_linkedin` uses to keep its
    legacy literal route from being shadowed.
    """

    if not req.modules:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "modules_required",
                "message": (
                    "POST /api/launch/multi requires at least one module"
                ),
            },
        )

    results: list[LaunchMultiPerSourceResult] = []
    success_count = 0

    for source in req.modules:
        per_source_request = LaunchRequest(
            brief_id=req.brief_id,
            mode=req.mode,
            force=req.force,
            force_fresh=req.force_fresh,
        )
        try:
            launch_response = _launch_for_source_impl(source, per_source_request)
        except HTTPException as exc:
            error_kind, error_detail = _multi_launch_error_envelope(exc)
            results.append(
                LaunchMultiPerSourceResult(
                    source=source,  # type: ignore[arg-type]
                    error=LaunchMultiPerSourceError(
                        kind=error_kind,
                        detail=error_detail,
                    ),
                )
            )
            continue

        success_count += 1
        results.append(
            LaunchMultiPerSourceResult(
                source=source,  # type: ignore[arg-type]
                launch=launch_response,
            )
        )

    if success_count == 0:
        # Every module failed. Return the per-module envelopes via 422
        # so the frontend can render the per-source readiness blockers
        # inline.
        response.status_code = 422

    return LaunchMultiResponse(
        brief_id=req.brief_id,
        results=results,
    )


def _multi_launch_error_envelope(exc: HTTPException) -> tuple[str, dict]:
    """Project a per-source HTTPException into the multi-launch shape.

    The per-source endpoint raises HTTPException with ``detail =
    {"error": "<kind>", ...payload}``. We pull the ``error`` out as
    ``kind`` and pass the rest of the dict through as ``detail`` so
    the frontend renders the same prose it does today for the
    per-source endpoint.
    """

    detail = exc.detail
    if isinstance(detail, dict):
        kind = str(detail.get("error") or "unknown")
        payload = {k: v for k, v in detail.items() if k != "error"}
        return kind, payload
    return "unknown", {"detail": detail, "status_code": exc.status_code}


@router.post(
    "/api/launch/{source}", status_code=201, response_model=LaunchResponse
)
def launch_for_source(source: str, req: LaunchRequest) -> LaunchResponse:
    """Phase F Slice F1. Generic per-source launch endpoint.

    Resolves ``brief_id`` to a brief on disk, runs the launch-readiness
    probe (skip if ``force=true``), then dispatches to the per-source
    spawn function via :data:`cloris.launchers.LAUNCHERS`. Returns
    :class:`LaunchResponse` with ``source``, ``mode``, and the spawned
    worker's ``pid`` / ``state_dir`` / ``worker_json_path``.

    The route is registered AFTER the legacy ``/api/launch/linkedin``
    and ``/api/resume/linkedin`` routes so Starlette matches the
    literal paths first; the path-param route catches all other
    sources (``github``, future Researcher, etc.).

    Error mapping:

    - :class:`UnknownSourceError` → HTTP 422 with allowed-sources list.
    - :class:`BriefIdNotFoundError` → HTTP 404.
    - :class:`LaunchNotReadyError` → HTTP 422 with structured blocker list
      (only when ``force=false``).
    - :class:`BriefPathNotFoundError` → HTTP 400 (defensive — the brief
      was resolved but disappeared between resolve and spawn).
    - :class:`WorkerAlreadyRunningError` → HTTP 409.
    - :class:`NoPendingWorkError` → HTTP 422 (only fires for ``mode="resume"``).
    - :class:`FreshOverResumableStateError` → HTTP 422 (fresh over generated
      resumable state without ``force_fresh`` consent).
    - :class:`LaunchLockTimeoutError` → HTTP 503.
    """

    return _launch_for_source_impl(source, req)


@router.post("/api/stop/{source}/{state_key}", response_model=StopResponse)
def stop(source: str, state_key: str, response: Response) -> StopResponse:
    """Send SIGTERM to the worker for ``(source, state_key)`` if alive.

    Status code is dynamic:

    - 202 when the helper signaled an alive PID.
    - 200 when there was nothing to signal (``missing`` or ``stale``).
    - 404 when the state dir isn't in the discovered set.

    The body always carries the truthful ``worker_state``; clients should
    branch on the body, not on the status code.
    """

    try:
        result = stop_worker(source, state_key)
    except StateDirNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "state_dir_not_found",
                "source": exc.source,
                "state_key": exc.state_key,
            },
        ) from exc

    if result.worker_state == "stopping":
        response.status_code = 202
    return result


# ---------------------------------------------------------------------------
# The Reflection — HITL Market Intelligence
# ---------------------------------------------------------------------------
#
# Two HITL gates around the market-intel pipeline:
#   Gate 1 — The Read   :: planner result + user steering (PATCH /steering)
#   Gate 2 — The Diff   :: proposed brief hunks (POST /commit | /discard)
#
# Long-running phase (research) executes in a background thread; the
# frontend polls GET /api/reflection/sessions/{id} for state transitions.
# Threads are sufficient for trial scope: one Cloris worker, one user,
# one active reflection per brief.
#
# Phase persistence lives in ``reflection_sessions.state_json`` (per the
# CRUD module). The engine phase functions in
# ``market_intelligence.reflection`` are pure with respect to the DB —
# the API layer owns the read/patch/transition cycle.


import threading

from market_intelligence import reflection as reflection_engine
from shared.runtime_state import reflection as reflection_store


# Reflection sessions live next to intake sessions in the same SQLite
# DB; reuse the same store factory so they share the migration path.
def _reflection_store_factory() -> RuntimeStateStore:
    """Compatibility seam for reflection tests/control-plane callers.

    The API package re-exports this name. If a legacy caller monkeypatches
    ``cloris.api._reflection_store_factory`` instead of this monolith module,
    honor that override while keeping the route implementation here.
    """

    import sys

    api_pkg = sys.modules.get("cloris.api")
    if api_pkg is not None:
        override = getattr(api_pkg, "_reflection_store_factory", None)
        if override is not None and override is not _reflection_store_factory:
            return override()
    return _intake_store()


def _reflection_session_response(session: dict) -> ReflectionResponse:
    return ReflectionResponse(
        session=ReflectionSession.model_validate(session)
    )


def _resolve_run_dir_for_run_id(run_id: int, *, brief_id: str) -> Path | None:
    """Best-effort lookup of the finalized run snapshot directory for a run.

    The runs table carries ``output_dir`` (the live state_dir) plus the
    finalized ``output_dir`` once the run completes. For trial scope we
    accept either: ``_resolve_market_intel_run_dir`` in the engine
    knows how to reconcile both. Returns ``None`` if the run id doesn't
    resolve; the engine will then raise on missing run_dir which surfaces
    as a 422 to the frontend.

    Run ids are per-state-dir SQLite autoincrements, not globally unique,
    so the lookup is scoped by ``brief_id``: iterate the state dirs whose
    latest run carries this brief (the same resolution the brief-first
    workspace routes use via ``state_dirs_for_brief_id``) and require the
    run row itself to carry the same brief_id, so a same-numbered run
    from another brief can't leak in.

    Routed through ``read_models.run_by_id`` rather than instantiating
    ``RuntimeStateStore``: this is a SELECT-only lookup, and even
    though the only caller today (``create_reflection_session_endpoint``)
    is a POST, mixing the read with a writer instantiation triggers
    DDL + meta-write before the actual write path runs. The read helper
    keeps the lookup honest.
    """

    try:
        for _source, state_dir in state_dirs_for_brief_id(brief_id):
            run = read_models.run_by_id(
                state_dir / "runtime_state.sqlite3", run_id=run_id
            )
            if run is None or not run.output_dir:
                continue
            if run.brief_id and run.brief_id != brief_id:
                continue
            return Path(run.output_dir)
    except Exception as exc:
        log.warning(
            "reflection: run lookup failed for run_id=%s brief_id=%s: %s",
            run_id,
            brief_id,
            exc,
        )
        return None
    return None


@router.post(
    "/api/reflection/sessions",
    status_code=201,
    response_model=ReflectionResponse,
)
def create_reflection_session_endpoint(
    req: ReflectionCreateRequest,
) -> ReflectionResponse:
    """Boot a new reflection session and run the planner phase synchronously.

    Resolves brief_id → brief_path, optionally maps source_run_id →
    run_dir (or accepts an explicit run_dir override), then runs the
    plan phase in-band so the response carries the editorial briefing
    + intentions the recruiter sees at Gate 1.

    Errors:
      * 404 ``brief_id_not_found`` — brief_id doesn't resolve
      * 409 ``reflection_already_active`` — there's already an
        in-flight reflection for this brief
      * 422 ``reflection_no_evidence`` — engine couldn't resolve a
        run_dir or the snapshot is empty
    """

    # Reject if there's already an active reflection for this brief.
    # The frontend should normally catch this via GET /active before
    # POSTing, but the API enforces the invariant so two tabs can't
    # both create.
    existing = reflection_store.get_active_reflection_for_brief(
        store=_reflection_store_factory(), brief_id=req.brief_id
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "reflection_already_active",
                "session_id": existing["id"],
                "current_phase": existing["current_phase"],
            },
        )

    try:
        brief_path = _resolve_brief_path_or_raise(req.brief_id)
    except BriefIdNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "brief_id_not_found", "brief_id": req.brief_id},
        ) from exc

    run_dir: Path | None = None
    if req.run_dir:
        run_dir = Path(req.run_dir)
    elif req.source_run_id is not None:
        run_dir = _resolve_run_dir_for_run_id(
            req.source_run_id, brief_id=req.brief_id
        )

    try:
        plan_state = reflection_engine.reflection_phase_plan(
            brief_path=brief_path,
            run_dir=run_dir,
            mode="post_run",
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "reflection_no_evidence",
                "message": str(exc),
            },
        ) from exc

    session = reflection_store.create_reflection_session(
        store=_reflection_store_factory(),
        brief_id=req.brief_id,
        source_run_id=req.source_run_id,
        initial_state=plan_state,
    )
    return _reflection_session_response(session)


# NOTE: /active must be declared BEFORE /{session_id} so the literal
# path matches before the int-typed catch-all.
@router.get(
    "/api/reflection/sessions/active",
    response_model=ReflectionActiveResponse,
)
def get_active_reflection_endpoint(brief_id: str) -> ReflectionActiveResponse:
    """Return the active (non-terminal) reflection for a brief, if any.

    Used by the workspace surface to decide whether to render the
    "Cloris read the market — review what she'd change" pickup card.
    Returns ``session=None`` when there's no active reflection
    (rather than 404) so the frontend doesn't have to distinguish
    error codes from absence.

    Read-only path: routes through
    ``read_models.get_active_reflection_for_brief`` against the
    intake DB (reflection sessions colocate with intake sessions per
    the ``_reflection_store_factory`` compatibility seam). See the
    intake list endpoint for the writer-on-read rationale.
    """

    from shared.runtime_state import read_models

    session = read_models.get_active_reflection_for_brief(
        _intake_db_path(), brief_id=brief_id
    )
    if session is None:
        return ReflectionActiveResponse(session=None)
    return ReflectionActiveResponse(
        session=ReflectionSession.model_validate(session)
    )


@router.get(
    "/api/reflection/sessions/{session_id}",
    response_model=ReflectionResponse,
)
def get_reflection_session_endpoint(session_id: int) -> ReflectionResponse:
    """Return one reflection session by id, or 404 if missing.

    Used both for in-flight resume (recruiter closes tab and comes
    back) and for short-interval polling during the research phase.
    The endpoint is cheap (single SQLite read) so polling at 2-3s
    intervals is fine — but this is also exactly what makes the
    writer-on-read pattern especially toxic here. Each poll formerly
    rewrote the schema_version meta row; routing through
    ``read_models.get_reflection_session`` keeps the polling honest.
    """

    from shared.runtime_state import read_models

    session = read_models.get_reflection_session(
        _intake_db_path(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "reflection_session_not_found", "id": session_id},
        )
    return _reflection_session_response(session)


@router.patch(
    "/api/reflection/sessions/{session_id}/steering",
    response_model=ReflectionResponse,
)
def patch_reflection_steering_endpoint(
    session_id: int, req: ReflectionSteeringRequest
) -> ReflectionResponse:
    """Add a steering note and re-run the planner phase.

    Each call bumps ``steering_iterations`` by 1. The 3-iteration cap
    is enforced server-side: the 4th attempt returns 409 with
    structured detail so the frontend can surface the
    "you've refined three times — trust the plan or discard" message.

    Empty / whitespace-only notes degenerate to a no-op (the cap
    isn't bumped, the planner isn't re-run). This protects against
    accidental empty-submit clicks.
    """

    session = reflection_store.get_reflection_session(
        store=_reflection_store_factory(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "reflection_session_not_found", "id": session_id},
        )
    if session["current_phase"] != "planning":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "reflection_phase_locked",
                "current_phase": session["current_phase"],
                "message": (
                    "Steering only applies during the planning gate; "
                    "this session has already moved past Gate 1."
                ),
            },
        )

    note = (req.note or "").strip()
    if not note:
        return _reflection_session_response(session)

    if session["steering_iterations"] >= reflection_engine.MAX_STEERING_ITERATIONS:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "reflection_steering_capped",
                "max_iterations": reflection_engine.MAX_STEERING_ITERATIONS,
                "message": (
                    "You've refined this plan three times. "
                    "Trust the plan and start reading, or discard and try again later."
                ),
            },
        )

    state = session["state_json"] or {}
    context = state.get("context") or {}
    history = list(state.get("steering_history") or [])
    notes = [item["note"] for item in history if isinstance(item, dict) and item.get("note")]
    notes.append(note)

    try:
        new_state = reflection_engine.reflection_phase_plan(
            brief_path=context.get("brief_path"),
            run_dir=context.get("run_dir"),
            mode=context.get("mode", "post_run"),
            steering_notes=notes,
        )
    except Exception as exc:
        log.exception(
            "reflection: re-plan failed for session=%s after steering: %s",
            session_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "reflection_plan_failed",
                "message": "I lost my train of thought — start the reflection over.",
            },
        ) from exc

    updated = reflection_store.patch_reflection_state(
        store=_reflection_store_factory(),
        session_id=session_id,
        state_json=new_state,
        bump_steering=True,
    )
    if updated is None:
        raise HTTPException(
            status_code=410,
            detail={
                "error": "reflection_session_gone",
                "id": session_id,
            },
        )
    return _reflection_session_response(updated)


def _run_research_in_background(session_id: int) -> None:
    """Background-thread worker for the research + propose phases.

    Reads the session, runs research → propose, persists the result,
    transitions to ``awaiting_diff``. On error, persists
    ``research_error`` and leaves the session in ``researching`` so the
    user can retry (or discard).

    Defensive: if the session was discarded mid-flight, exits without
    further work (the engine call is wasted but the result is dropped).
    Mirrors the plan's edge case 2 ("recruiter discards mid-research").
    """

    store = _reflection_store_factory()
    session = reflection_store.get_reflection_session(
        store=store, session_id=session_id
    )
    if session is None:
        log.warning(
            "reflection: research worker found no session id=%s", session_id
        )
        return

    try:
        researched_state = reflection_engine.reflection_phase_research(
            state=session["state_json"]
        )
    except Exception as exc:
        log.exception(
            "reflection: research phase failed for session=%s: %s",
            session_id,
            exc,
        )
        try:
            reflection_store.patch_reflection_state(
                store=store,
                session_id=session_id,
                research_error=(
                    "Cloris had trouble reaching her sources. "
                    "Try again, or skip the research and propose changes "
                    "from what's already on disk."
                ),
            )
        except ValueError:
            pass  # session went terminal mid-flight
        return

    # Re-read the session — the user may have discarded while research
    # was running. The discard endpoint is idempotent and the patch
    # below will raise ValueError on a terminal session, which we
    # swallow because the discard already won the race.
    current = reflection_store.get_reflection_session(
        store=store, session_id=session_id
    )
    if current is None or current["current_phase"] in {"committed", "discarded"}:
        log.info(
            "reflection: research finished but session id=%s is terminal "
            "(phase=%s); dropping result",
            session_id,
            current["current_phase"] if current else "missing",
        )
        return

    try:
        proposed_state = reflection_engine.reflection_phase_propose(
            state=researched_state
        )
    except Exception as exc:
        log.exception(
            "reflection: propose phase failed for session=%s: %s",
            session_id,
            exc,
        )
        try:
            reflection_store.patch_reflection_state(
                store=store,
                session_id=session_id,
                research_error=(
                    "Cloris read the market but couldn't synthesize the "
                    "findings. Try again or discard."
                ),
            )
        except ValueError:
            pass
        return

    try:
        reflection_store.patch_reflection_state(
            store=store,
            session_id=session_id,
            state_json=proposed_state,
            current_phase="awaiting_diff",
            clear_research_error=True,
        )
    except ValueError:
        # Session went terminal between phases; drop result silently.
        pass


@router.post(
    "/api/reflection/sessions/{session_id}/start_research",
    response_model=ReflectionResponse,
)
def start_reflection_research_endpoint(
    session_id: int, req: ReflectionStartResearchRequest
) -> ReflectionResponse:
    """Approve the plan; kick off research in a background thread.

    State transition: ``planning`` → ``plan_approved`` → (immediately)
    ``researching``. The intermediate ``plan_approved`` state is
    momentary — we transition straight to ``researching`` and spawn
    the worker thread. The frontend polls GET /sessions/{id} every
    2-3s and pivots to Gate 2 when the phase becomes ``awaiting_diff``.
    """

    session = reflection_store.get_reflection_session(
        store=_reflection_store_factory(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "reflection_session_not_found", "id": session_id},
        )
    if session["current_phase"] != "planning":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "reflection_phase_locked",
                "current_phase": session["current_phase"],
                "message": (
                    "Research can only start from the planning gate; "
                    "this session is already past Gate 1."
                ),
            },
        )

    updated = reflection_store.patch_reflection_state(
        store=_reflection_store_factory(),
        session_id=session_id,
        current_phase="researching",
        clear_research_error=True,
    )
    if updated is None:
        raise HTTPException(
            status_code=410,
            detail={"error": "reflection_session_gone", "id": session_id},
        )

    # Spawn the research worker. Daemon thread so it doesn't block
    # process exit if the worker is mid-call when Cloris shuts down
    # (acceptable: the result would be dropped anyway).
    worker = threading.Thread(
        target=_run_research_in_background,
        args=(session_id,),
        name=f"reflection-research-{session_id}",
        daemon=True,
    )
    worker.start()

    return _reflection_session_response(updated)


@router.post(
    "/api/reflection/sessions/{session_id}/commit",
    response_model=ReflectionCommitResponse,
)
def commit_reflection_endpoint(
    session_id: int, req: ReflectionCommitRequest
) -> ReflectionCommitResponse:
    """Apply accepted hunks to the brief; tombstone the session.

    Errors:
      * 404 ``reflection_session_not_found``
      * 409 ``reflection_phase_locked`` — not in awaiting_diff
      * 422 ``reflection_commit_no_hunks`` — accepted_hunk_ids empty
        (the frontend disables the CTA in this case but server enforces)
      * 500 ``reflection_commit_failed`` — write_brief_atomic blew up
    """

    session = reflection_store.get_reflection_session(
        store=_reflection_store_factory(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "reflection_session_not_found", "id": session_id},
        )
    if session["current_phase"] != "awaiting_diff":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "reflection_phase_locked",
                "current_phase": session["current_phase"],
                "message": (
                    "Commit can only happen from the diff gate; "
                    "this session isn't ready for changes yet."
                ),
            },
        )
    if not req.accepted_hunk_ids:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "reflection_commit_no_hunks",
                "message": (
                    "No changes accepted. Discard the reflection or "
                    "approve at least one change before filing."
                ),
            },
        )

    try:
        commit_result = reflection_engine.reflection_commit(
            state=session["state_json"],
            accepted_hunk_ids=req.accepted_hunk_ids,
            edited_hunks=req.edited_hunks or {},
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "reflection_commit_failed",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        log.exception(
            "reflection: commit failed for session=%s: %s", session_id, exc
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "reflection_commit_failed",
                "message": (
                    "I couldn't file the new brief. The previous brief is "
                    "still in place. Try again or discard."
                ),
            },
        ) from exc

    final_state = dict(session["state_json"] or {})
    final_state["commit_result"] = {
        "accepted_hunk_ids": list(req.accepted_hunk_ids),
        "edited_hunk_ids": list((req.edited_hunks or {}).keys()),
        "applied_hunks": commit_result["applied_hunks"],
    }
    committed = reflection_store.commit_reflection(
        store=_reflection_store_factory(),
        session_id=session_id,
        brief_version_path=commit_result["brief_version_path"],
        final_state=final_state,
    )
    if committed is None:
        raise HTTPException(
            status_code=410,
            detail={"error": "reflection_session_gone", "id": session_id},
        )
    return ReflectionCommitResponse(
        session=ReflectionSession.model_validate(committed),
        brief_version_path=commit_result["brief_version_path"],
        applied_hunks=commit_result["applied_hunks"],
    )


@router.post(
    "/api/reflection/sessions/{session_id}/discard",
    response_model=ReflectionResponse,
)
def discard_reflection_endpoint(
    session_id: int, req: ReflectionDiscardRequest
) -> ReflectionResponse:
    """Tombstone the session; brief untouched.

    Idempotent: discarding an already-discarded session returns the
    existing tombstone. Discarding a committed session is also
    idempotent — discard is a no-op against terminal rows. The
    frontend uses this to silently transition out of the reflection
    surface back to the workspace.
    """

    session = reflection_store.get_reflection_session(
        store=_reflection_store_factory(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "reflection_session_not_found", "id": session_id},
        )
    discarded = reflection_store.discard_reflection(
        store=_reflection_store_factory(), session_id=session_id
    )
    if discarded is None:
        raise HTTPException(
            status_code=410,
            detail={"error": "reflection_session_gone", "id": session_id},
        )
    return _reflection_session_response(discarded)


# ---------------------------------------------------------------------------
# Reopen Stage 2 (Part VII): recruiter-independent query endpoint.
#
# This is the adversarial-ledger flaw "reopen-coherence" fix (the critical
# one): every other recruiter read/write in Stage 2 is brief-triggered,
# which deepens brief-as-spine and makes the Stage-3 inversion (brief
# becomes a query over the recruiter) a from-scratch rewrite. This
# endpoint resolves the recruiter and returns their cross-brief state with
# NO brief context required — proving the recruiter is a first-class
# durable entity, not a brief-scoped sidecar, and giving Stage 3 a seam to
# build on. It is deliberately read-only and brief-independent.
# ---------------------------------------------------------------------------


class RecruiterTasteSignalSummary(BaseModel):
    """One active (non-superseded) recruiter taste signal."""

    id: int
    signal_kind: str
    domain: str
    source_brief_id: str | None = None
    confidence: float
    created_at: str
    payload: dict[str, Any] = Field(default_factory=dict)


class RecruiterResponse(BaseModel):
    """GET /api/recruiter — the acting recruiter as a first-class entity."""

    recruiter_id: int
    canonical_handle: str
    display_name: str
    briefs_count: int
    active_signals: list[RecruiterTasteSignalSummary] = Field(default_factory=list)


@router.get("/api/recruiter", response_model=RecruiterResponse)
def api_recruiter() -> RecruiterResponse:
    """Return the acting recruiter + cross-brief state. No brief context.

    Reopen Stage 2 Part VII. Resolves the recruiter through the
    ``get_current_recruiter_id`` seam (Stage 1: the implicit recruiter,
    Sam → id 1; Phase 2: the authenticated principal) and reads their
    durable cross-brief state directly from the recruiter store: brief
    membership count and active taste signals. Nothing here is keyed by a
    brief — that is the point.
    """

    from shared.output_paths import resolve_recruiter_db_path
    from shared.recruiter_context import get_current_recruiter_id
    from shared.runtime_state.recruiter_store import RecruiterStore

    recruiter_id = get_current_recruiter_id()
    store = RecruiterStore(resolve_recruiter_db_path())
    recruiter = store.get_recruiter(recruiter_id)
    if recruiter is None:
        # The resolver's default path get-or-creates the recruiter, so a
        # missing row here means a custom (Phase-2) resolver returned an id
        # that hasn't been provisioned. Surface it honestly rather than
        # fabricating an empty recruiter.
        raise HTTPException(
            status_code=404,
            detail={"error": "recruiter_not_found", "recruiter_id": recruiter_id},
        )

    briefs = store.briefs_for_recruiter(recruiter_id)
    signals = store.active_taste_signals(recruiter_id)
    return RecruiterResponse(
        recruiter_id=recruiter_id,
        canonical_handle=str(recruiter.get("canonical_handle", "")),
        display_name=str(recruiter.get("display_name", "")),
        briefs_count=len(briefs),
        active_signals=[
            RecruiterTasteSignalSummary(
                id=int(s["id"]),
                signal_kind=str(s["signal_kind"]),
                domain=str(s["domain"]),
                source_brief_id=s.get("source_brief_id"),
                confidence=float(s.get("confidence", 0.5)),
                created_at=str(s.get("created_at", "")),
                payload=s.get("payload") if isinstance(s.get("payload"), dict) else {},
            )
            for s in signals
        ],
    )


def _brief_state_dirs_map(
    brief_ids: set[str],
) -> dict[str, list[tuple[str, Path]]]:
    """Resolve ``{brief_id: [(source, state_dir), ...]}`` in ONE enumerate pass.

    R3.2 perf seam. ``cloris.control_plane.state_dirs_for_brief_id`` walks
    every discovered state dir and opens each DB read-only *per call*; the
    recruiter dashboard fans over N briefs, so calling it once per brief is
    N full walks. This helper inverts the loop — it enumerates the state-dir
    tree exactly once, reads each dir's latest-run ``brief_id`` (the same
    match key ``state_dirs_for_brief_id`` uses; see control_plane.py:1338-1347),
    and buckets the dir under that brief when it is one the caller cares about.

    The result is the per-request brief→dirs map the calibration fan-out
    consumes. A brief with no matching state dir is simply absent from the
    map (the caller treats absence as "zero markers, fail-soft"). Mirrors
    ``aggregate_workspace``'s ``state_dir / "runtime_state.sqlite3"`` resolution
    so the merged read targets the right per-source DBs, not a single one.
    """

    out: dict[str, list[tuple[str, Path]]] = {}
    if not brief_ids:
        return out
    for source, state_dir in enumerate_state_dirs():
        db_path = state_dir / "runtime_state.sqlite3"
        latest_run_id = _read_models.latest_run_in_state_dir(db_path)
        if latest_run_id is None:
            continue
        detail = _read_models.run_by_id(db_path, run_id=latest_run_id)
        if detail is None:
            continue
        brief_id = detail.brief_id
        if brief_id in brief_ids:
            out.setdefault(brief_id, []).append((source, state_dir))
    return out


def _active_reflections_by_brief(
    brief_ids: set[str],
) -> dict[str, sqlite3.Row]:
    """Read active reflection sessions for ``brief_ids`` from the intake DB.

    R3.3 read seam. Reflection rows live ONLY in the single intake DB
    (``resolve_intake_db_path`` / ``_intake_db_path``) — they are authored
    before any (source, state_key) commitment, so per-state-dir
    ``runtime_state.sqlite3`` files carry an always-empty
    ``reflection_sessions`` table. Reading per-state-dir would ship a dead
    panel; this reader fans by ``brief_id`` against that one DB.

    Read-only by construction: opens the DB via the
    ``calibration._open_readonly`` ``mode=ro`` pattern and runs raw SQL. It
    constructs NO ``RuntimeStateStore`` — instantiating one runs mkdir +
    ``initialize()`` DDL (write-on-read), which a GET must never do. A
    missing/unreadable intake DB, or one predating the reflection schema,
    collapses to an empty map (no 500, no DB creation).

    "Active" mirrors
    ``shared.runtime_state.reflection.get_active_reflection_for_brief``
    (reflection.py:167-191): ``completed_at IS NULL AND discarded_at IS NULL``,
    most-recently-updated row per brief. Returns ``{brief_id: row}`` for the
    briefs that have an active session; briefs without one are absent.
    """

    from shared.runtime_state.calibration import _open_readonly

    out: dict[str, sqlite3.Row] = {}
    if not brief_ids:
        return out

    intake_db = _intake_db_path()
    sql = (
        "SELECT id, brief_id, source_run_id, current_phase, "
        "steering_iterations, started_at, updated_at, completed_at, "
        "discarded_at "
        "FROM reflection_sessions "
        "WHERE completed_at IS NULL AND discarded_at IS NULL "
        "ORDER BY updated_at ASC"
    )
    with _open_readonly(intake_db) as conn:
        if conn is None:
            return out
        try:
            rows = list(conn.execute(sql).fetchall())
        except sqlite3.OperationalError:
            # Intake DB predates the reflection_sessions schema. Same
            # fail-soft posture as the calibration aggregator's
            # pre-Phase-C-bis branch — a passive read never crashes.
            return out

    # ``ORDER BY updated_at ASC`` means the last write for a given brief_id
    # wins this dict assignment, matching get_active_reflection_for_brief's
    # "most recently updated row when more than one is active" tie-break.
    for row in rows:
        brief_id = row["brief_id"]
        if brief_id in brief_ids:
            out[brief_id] = row
    return out


@router.get(
    "/api/recruiter/{recruiter_id}/dashboard",
    response_model=RecruiterDashboardResponse,
)
def api_recruiter_dashboard(recruiter_id: int) -> RecruiterDashboardResponse:
    """Cross-brief recruiter dashboard surface (Reopen Stage 3, R3.1).

    Builds the PRESENCE panel from the ``recruiter_candidate_history``
    accretion log — for each person this recruiter has seen, the count and the
    first/last brief and last lifecycle state. Resolves the recruiter through
    the same recruiter store / 404 contract as ``GET /api/recruiter``, but keyed
    on the path ``recruiter_id`` (the addressed recruiter) rather than the
    acting-principal seam — a missing row is surfaced honestly, not fabricated.

    ``calibration_drift`` (R3.2) is one entry per brief the recruiter owns,
    carrying the ``judgment_accuracy`` marker rollup MERGED across every state
    dir the brief spans (a brief may live in linkedin + github; reading one
    db_path under-reads it). ``reflection_trail`` (R3.3) is one entry per brief
    with an *active* reflection session, read from the SINGLE intake DB
    (read-only — the GET creates no DB and runs no DDL). Both fan over
    ``briefs_for_recruiter``. Presence deliberately carries no
    terminal_decision: it is the accretion log, not the flattened current-state
    authority, and R3.2/R3.3 leave it byte-identical.
    """

    from shared.runtime_state.calibration import aggregate_calibration_markers
    from shared.output_paths import resolve_recruiter_db_path
    from shared.runtime_state.recruiter_store import RecruiterStore

    store = RecruiterStore(resolve_recruiter_db_path())
    recruiter = store.get_recruiter(recruiter_id)
    if recruiter is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "recruiter_not_found", "recruiter_id": recruiter_id},
        )

    history = store.candidate_history_for_recruiter(recruiter_id)

    # The brief set both forward panels fan over (recruiter_store.py:330).
    brief_ids = store.briefs_for_recruiter(recruiter_id)
    brief_id_set = set(brief_ids)

    # R3.2 — calibration drift. Resolve every brief's state dirs in ONE
    # enumerate pass (shared across the fan), then for each brief MERGE the
    # per-dir rollups exactly like aggregate_workspace: sum total_markers and
    # merge the per-axis counters across sources. A brief with no state dir
    # contributes zero (fail-soft), never a 500.
    state_dirs_by_brief = _brief_state_dirs_map(brief_id_set)
    calibration_drift: list[RecruiterCalibrationEntry] = []
    for brief_id in brief_ids:
        merged_total = 0
        merged_by_marker: dict[str, int] = {}
        merged_by_area: dict[str, int] = {}
        merged_weighted: dict[str, int] = {}
        matched_dirs = state_dirs_by_brief.get(brief_id, [])
        for _source, state_dir in matched_dirs:
            db_path = state_dir / "runtime_state.sqlite3"
            rollup = aggregate_calibration_markers(db_path, brief_id=brief_id)
            merged_total += rollup.total_markers
            for marker, count in rollup.by_marker_value.items():
                merged_by_marker[marker] = merged_by_marker.get(marker, 0) + count
            for area, count in rollup.by_capability_area.items():
                # ``None`` (unattributed) area: bucket under a stable string
                # key so the wire dict stays str-keyed. The rollup itself
                # keeps the None bucket; the panel collapses it to "unknown".
                area_key = area if area is not None else "unknown"
                merged_by_area[area_key] = merged_by_area.get(area_key, 0) + count
            for area, weight in rollup.weighted_markers_by_area.items():
                area_key = area if area is not None else "unknown"
                merged_weighted[area_key] = merged_weighted.get(area_key, 0) + weight

        # Legacy two-field surface: ``drift`` = share of markers the recruiter
        # flagged as miscalibrated (wrong + off_rubric), 0.0 when no markers.
        miscalibrated = merged_by_marker.get("wrong", 0) + merged_by_marker.get(
            "off_rubric", 0
        )
        drift = (miscalibrated / merged_total) if merged_total else 0.0
        calibration_drift.append(
            RecruiterCalibrationEntry(
                brief_id=brief_id,
                total_markers=merged_total,
                by_marker_value=merged_by_marker,
                by_capability_area=merged_by_area,
                weighted_markers_by_area=merged_weighted,
                source_state_dirs=len(matched_dirs),
                domain=brief_id,
                drift=drift,
            )
        )

    # R3.3 — reflection trail. One read of the SINGLE intake DB, fanned by
    # brief_id (zero extra state-dir walks). Read-only: no RuntimeStateStore,
    # no DDL, no DB creation on this GET.
    active_reflections = _active_reflections_by_brief(brief_id_set)
    reflection_trail: list[RecruiterReflectionEntry] = []
    for brief_id in brief_ids:
        row = active_reflections.get(brief_id)
        if row is None:
            continue
        phase = str(row["current_phase"] or "")
        started_at = str(row["started_at"] or "")
        reflection_trail.append(
            RecruiterReflectionEntry(
                reflection_id=int(row["id"]),
                brief_id=brief_id,
                current_phase=phase,
                started_at=started_at,
                updated_at=str(row["updated_at"] or ""),
                steering_iterations=int(row["steering_iterations"] or 0),
                summary=f"{brief_id} @ {phase}" if phase else brief_id,
                created_at=started_at,
            )
        )

    return RecruiterDashboardResponse(
        recruiter_id=recruiter_id,
        recruiter_handle=str(recruiter.get("canonical_handle", "")),
        presence=[
            RecruiterPresenceEntry(
                person_id=int(row["person_id"]),
                times_surfaced=int(row["times_surfaced"]),
                first_seen_brief=str(row["first_seen_brief"]),
                last_seen_brief=str(row["last_seen_brief"]),
                last_lifecycle_state=str(row.get("last_lifecycle_state", "")),
            )
            for row in history
        ],
        calibration_drift=calibration_drift,
        reflection_trail=reflection_trail,
    )
