"""Candidate read + annotation HTTP routes.

Carved out of :mod:`cloris.api._monolith` (Phase 4, slice P4-4) with no behavior
change. These are the brief-first candidate-detail / shortlist reads and the
recruiter annotation mutations (note / user_status / judgment-accuracy /
Designer principle-feedback / asset exclude+revoke). They register on the shared
``router`` from :mod:`cloris.api.routing`, identical to their prior home; the
module is imported by :mod:`cloris.api` so the decorators run at app assembly.

Deliberately excludes the launch / reflection / COS routers, which carry open
control-plane bugs and must not be split this phase.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException

from cloris.control_plane import (
    aggregate_candidate_detail,
    aggregate_workspace,
    state_dirs_for_brief_id,
)
from cloris.models import (
    CandidateDetailResponse,
    CandidateJudgmentAccuracyPatchRequest,
    CandidateNoteRequest,
    CandidateStatusPatchRequest,
    CrossBriefPresence,
    EXEC_SEARCH_SAVES_SHAPE_THRESHOLD,
    ExcludeAssetRequest,
    PrincipleFeedbackRequest,
    RevokeExcludedAssetRequest,
    ShortlistResponse,
)
from shared.runtime_state import read_models as _read_models
from shared.runtime_state.store import RuntimeStateStore

from .routing import router

log = logging.getLogger("cloris.api")


@router.get(
    "/api/shortlist/{brief_id}",
    response_model=ShortlistResponse,
)
def api_shortlist(brief_id: str) -> ShortlistResponse:
    """Executive Search shortlist surface (Slice 7, downgraded scope).

    Read-side projection over the existing per-source candidates
    tables. The Cloris-native ``shortlist_entries`` table + the
    ``AbstractSaveDestination`` write path depend on
    multi-module-foundation Slices 6-7 (workspace tables +
    AbstractSaveDestination), which are NOT shipped — so Slice 7's
    scope is downgraded to a read view per the spec's "downgrade or
    absorb" rule.

    Surfaces a saves-shape alarm when the SAVE-class candidate count
    exceeds :data:`EXEC_SEARCH_SAVES_SHAPE_THRESHOLD` (25). Distinct
    from Slice 5's cost cap circuit-breaker (fires on cost) and
    eval-count alarm (fires on evaluations); this alarm fires on
    saves and tells the recruiter the brief calibration may be too
    broad.

    Errors:
      * 404 ``shortlist_not_found`` — no state_dir's latest run
        carries this brief_id (or no runs exist at all).
    """

    workspace = aggregate_workspace(brief_id=brief_id)
    if workspace is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "shortlist_not_found",
                "brief_id": brief_id,
            },
        )
    saves_shape_alarm_fired = (
        len(workspace.candidates) > EXEC_SEARCH_SAVES_SHAPE_THRESHOLD
    )
    return ShortlistResponse(
        brief_id=workspace.brief_id,
        sources=workspace.sources,
        brief_role_title=workspace.brief_role_title,
        brief_linkedin_project=workspace.brief_linkedin_project,
        latest_run=workspace.latest_run,
        total_saves=workspace.total_saves,
        saves_this_week=getattr(workspace, "saves_this_week", 0),
        last_save_at=getattr(workspace, "last_save_at", None),
        candidates=workspace.candidates,
        saves_shape_alarm=saves_shape_alarm_fired,
        saves_shape_alarm_threshold=EXEC_SEARCH_SAVES_SHAPE_THRESHOLD,
    )


@router.get(
    "/api/candidate/{brief_id}/{candidate_id}",
    response_model=CandidateDetailResponse,
)
def api_candidate_detail(
    brief_id: str, candidate_id: int
) -> CandidateDetailResponse:
    """Per-candidate detail (Phase C-bis 0.1, brief-first).

    The aggregator finds which state_dir under this brief_id holds the
    candidate row and returns the detail. ``source`` and ``source_run``
    in the response carry the metadata the page needs (source eyebrow,
    run-report back-link) without polluting the URL contract.

    Errors:
      * 404 ``candidate_not_found`` — no state_dir under this brief_id
        contains a candidate with the given id (or the candidate's
        ``brief_id``/``source`` doesn't match the resolved state_dir's,
        which is a cross-source guard that prevents leaking foreign rows).
    """

    detail = aggregate_candidate_detail(
        brief_id=brief_id, candidate_id=candidate_id
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "candidate_not_found",
                "brief_id": brief_id,
                "candidate_id": candidate_id,
            },
        )
    # Reopen Stage 3 (R3, "remembers"): enrich with cross-brief presence —
    # the inline "you've seen them before" marker — AT THE API LAYER, not in
    # the aggregator. ``control_plane.aggregate_candidate_detail`` is
    # deliberately read-only and free of the cross-DB recruiter seam (see
    # ``recruiter_sighting`` module docstring); the recruiter-spine join lives
    # here, where the seam is already imported and ``record_sightings_for_brief``
    # already fires. Fail-soft: never 500 the candidate page on this enrichment.
    detail.cross_brief_presence = _cross_brief_presence_for_candidate(detail)
    return detail


def _cross_brief_presence_for_candidate(
    detail: CandidateDetailResponse,
) -> CrossBriefPresence | None:
    """Resolve one person's cross-brief recurrence for the memory marker.

    Reopen Stage 3 (R3). Read-only join across the identity DB
    (``candidate_persons`` → person_id) and the recruiter spine
    (``recruiter_for_brief`` → recruiter_id, then ``candidate_history`` →
    the accretion row). Returns a :class:`CrossBriefPresence` ONLY when this
    person has been seen in at least one OTHER brief (``other_briefs_count``
    >= 1); ``None`` otherwise (first-seen-here → nothing to render) or on any
    resolution failure (the spine is a calibration side channel, not part of
    the candidate-detail contract — a failure here must never break the read).

    ``other_briefs_count`` = ``times_surfaced`` minus 1 when the current brief
    is among the sighted briefs (the steady state on a candidate the recruiter
    is actively looking at — they've been sighted here). Clamped at 0.
    """

    # The candidate→person link is keyed (source, state_key, candidate_id).
    # ``state_key`` rides on source_run; without it we cannot resolve the
    # person unambiguously, so degrade to no marker.
    if detail.source_run is None:
        return None
    state_key = detail.source_run.state_key

    try:
        from shared.output_paths import (
            resolve_identity_db_path,
            resolve_recruiter_db_path,
        )
        from shared.runtime_state.identity_store import IdentityStore
        from shared.runtime_state.recruiter_store import RecruiterStore

        identity_store = IdentityStore(resolve_identity_db_path())
        with identity_store.connect() as conn:
            row = conn.execute(
                "SELECT person_id FROM candidate_persons "
                "WHERE source = ? AND state_key = ? AND candidate_id = ?",
                (detail.source, state_key, detail.candidate_id),
            ).fetchone()
        if row is None:
            return None
        person_id = int(row["person_id"])

        recruiter_store = RecruiterStore(resolve_recruiter_db_path())
        recruiter_id = recruiter_store.recruiter_for_brief(detail.brief_id)
        if recruiter_id is None:
            return None

        history = recruiter_store.candidate_history(recruiter_id, person_id)
        if history is None:
            return None

        times_surfaced = int(history.get("times_surfaced", 0) or 0)
        first_seen = str(history.get("first_seen_brief", "") or "")
        last_seen = str(history.get("last_seen_brief", "") or "")
        last_state = str(history.get("last_lifecycle_state", "") or "")

        # "Elsewhere" count: subtract the current brief if it's in the trail
        # (it is, for a candidate the recruiter is viewing — they were sighted
        # here). Only attach the marker when the person genuinely recurs.
        seen_here = detail.brief_id in {first_seen, last_seen}
        other_briefs_count = times_surfaced - 1 if seen_here else times_surfaced
        if other_briefs_count < 1:
            return None

        return CrossBriefPresence(
            times_surfaced=times_surfaced,
            other_briefs_count=other_briefs_count,
            first_seen_brief=first_seen,
            last_seen_brief=last_seen,
            last_lifecycle_state=last_state,
        )
    except Exception:  # noqa: BLE001 — calibration side channel, never break the read
        log.debug(
            "cross-brief presence enrichment failed for candidate %s/%s",
            detail.brief_id,
            detail.candidate_id,
            exc_info=True,
        )
        return None


# Phase C, slice C3: validated user_status values. NULL clears the override
# (Cloris's terminal_decision is the displayed status). Non-null must be in
# this set; any other value gets a 422.
_ALLOWED_USER_STATUSES: frozenset[str] = frozenset({
    "shortlist",
    "parked",
    "contacted",
    "hidden",
})


def _find_state_dir_for_candidate(
    brief_id: str, candidate_id: int
) -> Path | None:
    """Find the state_dir holding a (brief_id, candidate_id) pair, for
    mutation endpoints that need to instantiate :class:`RuntimeStateStore`
    on the right SQLite file. Returns ``None`` if no state_dir matches.
    """

    for source, state_dir in state_dirs_for_brief_id(brief_id):
        db_path = state_dir / "runtime_state.sqlite3"
        record = _read_models.candidate_by_id(db_path, candidate_id=candidate_id)
        if record is None:
            continue
        if record.brief_id == brief_id and record.source == source:
            return state_dir
    return None


@router.post(
    "/api/candidate/{brief_id}/{candidate_id}/note",
    response_model=CandidateDetailResponse,
)
def api_candidate_append_note(
    brief_id: str,
    candidate_id: int,
    request: CandidateNoteRequest,
) -> CandidateDetailResponse:
    """Append a recruiter-authored note to a candidate (brief-first).

    Notes are append-only; each note is timestamped at write time and
    rendered in reverse-chrono on the candidate-detail page. The route
    returns the full updated :class:`CandidateDetailResponse` so the
    client can re-render without a follow-up GET.

    Errors:
      * 404 ``candidate_not_found`` — no state_dir under this brief_id
        contains a candidate with the given id.
      * 422 ``empty_note_body`` — body is empty / whitespace-only.
    """

    body = request.body.strip()
    if not body:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_note_body"},
        )
    state_dir = _find_state_dir_for_candidate(brief_id, candidate_id)
    if state_dir is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "candidate_not_found",
                "brief_id": brief_id,
                "candidate_id": candidate_id,
            },
        )
    db_path = state_dir / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    try:
        store.append_candidate_note(candidate_id, body)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "candidate_not_found",
                "brief_id": brief_id,
                "candidate_id": candidate_id,
            },
        )
    # Reopen Y.5.3: the brief-keyed setter above succeeded — re-sync the
    # recruiter current-state authority for this person. Fail-soft: a broken
    # authority must NEVER break the mutation (the note already committed), so
    # swallow + debug-log. See recruiter_mutation_sync (mirrors the
    # recruiter_sighting fail-soft contract).
    try:
        from shared.runtime_state.recruiter_mutation_sync import (
            sync_candidate_mutation,
        )

        sync_candidate_mutation(brief_id, candidate_id, state_dir)
    except Exception as exc:  # noqa: BLE001 — never break the mutation
        log.debug(
            "Recruiter authority sync skipped after note on candidate %s "
            "(brief %r, %s): %s",
            candidate_id,
            brief_id,
            type(exc).__name__,
            exc,
        )
    detail = aggregate_candidate_detail(
        brief_id=brief_id, candidate_id=candidate_id
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "candidate_not_found"},
        )
    return detail


@router.patch(
    "/api/candidate/{brief_id}/{candidate_id}",
    response_model=CandidateDetailResponse,
)
def api_candidate_update_status(
    brief_id: str,
    candidate_id: int,
    request: CandidateStatusPatchRequest,
) -> CandidateDetailResponse:
    """Set or clear the recruiter-overridden status on a candidate (brief-first).

    Errors:
      * 404 ``candidate_not_found`` — no state_dir under this brief_id
        contains a candidate with the given id.
      * 422 ``invalid_user_status`` — the requested status isn't in the
        allowed set (``shortlist`` / ``parked`` / ``contacted`` /
        ``hidden`` / ``null``).
    """

    user_status = request.user_status
    if user_status is not None and user_status not in _ALLOWED_USER_STATUSES:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_user_status",
                "allowed": sorted(_ALLOWED_USER_STATUSES),
            },
        )
    state_dir = _find_state_dir_for_candidate(brief_id, candidate_id)
    if state_dir is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "candidate_not_found",
                "brief_id": brief_id,
                "candidate_id": candidate_id,
            },
        )
    db_path = state_dir / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    try:
        store.set_candidate_user_status(candidate_id, user_status)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "candidate_not_found",
                "brief_id": brief_id,
                "candidate_id": candidate_id,
            },
        )
    # Reopen Y.5.3: re-sync the recruiter current-state authority after the
    # brief-keyed user_status write. Fail-soft (the write already committed) —
    # see recruiter_mutation_sync.
    try:
        from shared.runtime_state.recruiter_mutation_sync import (
            sync_candidate_mutation,
        )

        sync_candidate_mutation(brief_id, candidate_id, state_dir)
    except Exception as exc:  # noqa: BLE001 — never break the mutation
        log.debug(
            "Recruiter authority sync skipped after user_status on candidate %s "
            "(brief %r, %s): %s",
            candidate_id,
            brief_id,
            type(exc).__name__,
            exc,
        )
    detail = aggregate_candidate_detail(
        brief_id=brief_id, candidate_id=candidate_id
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "candidate_not_found"},
        )
    return detail


# Phase C-bis 0.5: closed-loop feedback substrate. Allowed values for
# the recruiter's calibration signal — kept in sync with
# RuntimeStateStore.set_candidate_judgment_accuracy. NULL clears.
_ALLOWED_JUDGMENT_ACCURACIES: frozenset[str] = frozenset({
    "useful",
    "wrong",
    "off_rubric",
    "overstated_depth",
    "understated_depth",
})


@router.patch(
    "/api/candidate/{brief_id}/{candidate_id}/judgment-accuracy",
    response_model=CandidateDetailResponse,
)
def api_candidate_update_judgment_accuracy(
    brief_id: str,
    candidate_id: int,
    request: CandidateJudgmentAccuracyPatchRequest,
) -> CandidateDetailResponse:
    """Set or clear the recruiter's judgment-accuracy signal (brief-first).

    Phase C-bis Slice 0.5. Distinct from the user_status PATCH —
    judgment_accuracy captures whether Cloris's *judgment* was useful
    or off, not what pipeline action the recruiter is taking. Both
    columns coexist on the candidate row.

    Errors:
      * 404 ``candidate_not_found`` — no state_dir under this brief_id
        contains a candidate with the given id.
      * 422 ``invalid_judgment_accuracy`` — the requested value isn't
        in the allowed set.
    """

    judgment_accuracy = request.judgment_accuracy
    if (
        judgment_accuracy is not None
        and judgment_accuracy not in _ALLOWED_JUDGMENT_ACCURACIES
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_judgment_accuracy",
                "allowed": sorted(_ALLOWED_JUDGMENT_ACCURACIES),
            },
        )
    state_dir = _find_state_dir_for_candidate(brief_id, candidate_id)
    if state_dir is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "candidate_not_found",
                "brief_id": brief_id,
                "candidate_id": candidate_id,
            },
        )
    db_path = state_dir / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    try:
        store.set_candidate_judgment_accuracy(candidate_id, judgment_accuracy)
    except ValueError as exc:
        # The store raises ValueError for two distinct cases — unknown
        # candidate_id and unknown accuracy value. The accuracy values
        # are pre-validated above, so a ValueError here is the
        # candidate-not-found case. Mirror the user_status pattern.
        message = str(exc)
        if "invalid judgment_accuracy" in message:
            # Defense-in-depth: store-side validation also fired. Surface
            # as 422 like the API-layer check.
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_judgment_accuracy",
                    "allowed": sorted(_ALLOWED_JUDGMENT_ACCURACIES),
                },
            )
        raise HTTPException(
            status_code=404,
            detail={
                "error": "candidate_not_found",
                "brief_id": brief_id,
                "candidate_id": candidate_id,
            },
        )
    # Reopen Y.5.3: re-sync the recruiter current-state authority after the
    # brief-keyed judgment_accuracy write. Fail-soft (the write already
    # committed) — see recruiter_mutation_sync.
    try:
        from shared.runtime_state.recruiter_mutation_sync import (
            sync_candidate_mutation,
        )

        sync_candidate_mutation(brief_id, candidate_id, state_dir)
    except Exception as exc:  # noqa: BLE001 — never break the mutation
        log.debug(
            "Recruiter authority sync skipped after judgment_accuracy on "
            "candidate %s (brief %r, %s): %s",
            candidate_id,
            brief_id,
            type(exc).__name__,
            exc,
        )
    detail = aggregate_candidate_detail(
        brief_id=brief_id, candidate_id=candidate_id
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "candidate_not_found"},
        )
    return detail


# ---------------------------------------------------------------------------
# Designer D6: HITL feedback endpoints. Per-principle markers + asset
# exclusion/revocation. The two-store bridge writes to annotations.sqlite3
# (Designer-specific) and mirrors to runtime_state.sqlite3 (canonical).
# ---------------------------------------------------------------------------


@router.post(
    "/api/candidate/{brief_id}/{candidate_id}/principle-feedback",
    response_model=CandidateDetailResponse,
)
def api_candidate_principle_feedback(
    brief_id: str,
    candidate_id: int,
    request: PrincipleFeedbackRequest,
) -> CandidateDetailResponse:
    """Record per-principle feedback from the recruiter (Designer D6).

    Writes to both the per-state-dir annotations.sqlite3 and the canonical
    runtime_state.sqlite3 via the two-store bridge. Returns the updated
    CandidateDetailResponse so the frontend can re-render without a GET.

    Errors:
      * 404 ``candidate_not_found``
      * 422 ``invalid_feedback_marker``
    """

    state_dir = _find_state_dir_for_candidate(brief_id, candidate_id)
    if state_dir is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "candidate_not_found",
                "brief_id": brief_id,
                "candidate_id": candidate_id,
            },
        )
    db_path = state_dir / "runtime_state.sqlite3"
    record = _read_models.candidate_by_id(db_path, candidate_id=candidate_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "candidate_not_found"},
        )

    from designer.recruiter_annotations import (
        PrincipleFeedbackStore,
        record_designer_principle_feedback,
    )
    from designer.run_end import annotations_db_path

    annotations_db = annotations_db_path(state_dir)
    principle_store = PrincipleFeedbackStore(annotations_db)
    runtime_store = RuntimeStateStore(db_path)

    try:
        record_designer_principle_feedback(
            runtime_state_store=runtime_store,
            principle_feedback_store=principle_store,
            source="designer",
            brief_id=brief_id,
            identity_key=record.identity_key,
            principle_name=request.principle_name,
            marker=request.marker,
            note=request.note,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_feedback_marker",
                "message": str(exc),
            },
        )

    detail = aggregate_candidate_detail(
        brief_id=brief_id, candidate_id=candidate_id
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "candidate_not_found"},
        )
    return detail


@router.post(
    "/api/candidate/{brief_id}/{candidate_id}/excluded-asset",
    response_model=CandidateDetailResponse,
)
def api_candidate_exclude_asset(
    brief_id: str,
    candidate_id: int,
    request: ExcludeAssetRequest,
) -> CandidateDetailResponse:
    """Exclude a portfolio asset from visual judgment (Designer D6).

    Writes to the per-state-dir annotations.sqlite3. Returns the updated
    CandidateDetailResponse.

    Errors:
      * 404 ``candidate_not_found``
    """

    state_dir = _find_state_dir_for_candidate(brief_id, candidate_id)
    if state_dir is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "candidate_not_found",
                "brief_id": brief_id,
                "candidate_id": candidate_id,
            },
        )
    db_path = state_dir / "runtime_state.sqlite3"
    record = _read_models.candidate_by_id(db_path, candidate_id=candidate_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "candidate_not_found"},
        )

    from designer.recruiter_annotations import ExcludedAssetStore
    from designer.run_end import annotations_db_path

    annotations_db = annotations_db_path(state_dir)
    store = ExcludedAssetStore(annotations_db)
    store.exclude(
        candidate_identity_key=record.identity_key,
        asset_url=request.asset_url,
        reason=request.reason,
    )

    detail = aggregate_candidate_detail(
        brief_id=brief_id, candidate_id=candidate_id
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "candidate_not_found"},
        )
    return detail


@router.delete(
    "/api/candidate/{brief_id}/{candidate_id}/excluded-asset",
    response_model=CandidateDetailResponse,
)
def api_candidate_revoke_excluded_asset(
    brief_id: str,
    candidate_id: int,
    request: RevokeExcludedAssetRequest,
) -> CandidateDetailResponse:
    """Un-exclude a previously excluded asset (Designer D6).

    Soft-deletes the active exclusion row in annotations.sqlite3. Returns
    the updated CandidateDetailResponse.

    Errors:
      * 404 ``candidate_not_found``
    """

    state_dir = _find_state_dir_for_candidate(brief_id, candidate_id)
    if state_dir is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "candidate_not_found",
                "brief_id": brief_id,
                "candidate_id": candidate_id,
            },
        )
    db_path = state_dir / "runtime_state.sqlite3"
    record = _read_models.candidate_by_id(db_path, candidate_id=candidate_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "candidate_not_found"},
        )

    from designer.recruiter_annotations import ExcludedAssetStore
    from designer.run_end import annotations_db_path

    annotations_db = annotations_db_path(state_dir)
    store = ExcludedAssetStore(annotations_db)
    store.revoke(
        candidate_identity_key=record.identity_key,
        asset_url=request.asset_url,
    )

    detail = aggregate_candidate_detail(
        brief_id=brief_id, candidate_id=candidate_id
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "candidate_not_found"},
        )
    return detail
