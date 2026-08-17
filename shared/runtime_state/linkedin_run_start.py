"""LinkedIn run-start orchestration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from linkedin.search_intelligence import LinkedInExperimentState
from shared.schemas import Progress
from shared.safety.stop_reasons import RunStopReason

from .projections import project_linkedin_progress
from .store import RuntimeStateStore


def _identity_kwargs(brief_identity: dict | None) -> dict:
    """Return start_run kwargs from a brief_identity dict, or empty.

    Centralizes the dict-to-kwargs unpacking so future fields can be
    added without touching every start_run call site.
    """

    if not brief_identity:
        return {}
    out = {}
    for key in ("brief_path_at_launch", "brief_content_hash", "brief_snapshot_json"):
        if key in brief_identity:
            out[key] = brief_identity[key]
    return out


def _resolve_recruiter_id() -> int | None:
    """Resolve the acting recruiter for a run-start, fail-soft to None.

    reopen Stage 2 (R5a-3): stamps ``runs.recruiter_id`` so the read-only
    taste aggregator (R5a-4) can attribute adaptation decisions. The
    resolver is the single auth seam (``shared.recruiter_context``); we
    catch broadly because a run launch must never die on recruiter
    resolution — a None recruiter_id is a clean "unknown" (the aggregator
    skips it), whereas a raised exception here would abort the run.
    """

    try:
        from shared.recruiter_context import get_current_recruiter_id

        return get_current_recruiter_id()
    except Exception:  # noqa: BLE001 — resolution must never break a run launch
        return None


def start_or_resume_linkedin_run(
    *,
    store: RuntimeStateStore,
    output_dir: str | Path,
    brief_id: str,
    brief_name: str,
    resume: bool,
    initial_progress: Progress | None,
    experiment_states: dict[int, LinkedInExperimentState] | None,
    legacy_state_exists: bool,
    import_legacy_state: Callable[[int], None],
    sync_progress: Callable[[int, Progress, dict[int, LinkedInExperimentState] | None], None],
    rebuild_artifacts: Callable[[int], None],
    brief_identity: dict | None = None,
) -> tuple[int, Progress]:
    """Phase 3: ``brief_identity`` is an optional dict carrying
    ``brief_path_at_launch``, ``brief_content_hash``, and
    ``brief_snapshot_json``. When present, the new run row pins those
    values; when absent (legacy callers / tests), the columns stay
    NULL/empty and the aggregator falls back to state_key for the UI
    heading."""

    pinning_kwargs = _identity_kwargs(brief_identity)
    recruiter_id = _resolve_recruiter_id()
    latest_run = store.get_latest_run(source="linkedin", brief_id=brief_id)
    had_runtime_before = bool(
        latest_run or store.has_candidates(source="linkedin", brief_id=brief_id)
    )
    if latest_run and latest_run.get("status") == "running":
        store.finish_run(
            int(latest_run["id"]),
            "interrupted",
            stop_reason=RunStopReason.WORKER_MISSING,
        )

    if resume and latest_run and store.has_work_units(int(latest_run["id"])):
        run_id = store.start_run(
            source="linkedin",
            brief_id=brief_id,
            output_dir=str(output_dir),
            mode="resume",
            resume_state=store.get_run_resume_state(int(latest_run["id"])),
            resumed_from_run_id=int(latest_run["id"]),
            clone_work_units_from_run_id=int(latest_run["id"]),
            recruiter_id=recruiter_id,
            **pinning_kwargs,
        )
        rebuild_artifacts(run_id)
        return run_id, project_linkedin_progress(store, run_id)

    run_id = store.start_run(
        source="linkedin",
        brief_id=brief_id,
        output_dir=str(output_dir),
        mode="resume" if resume else "fresh",
        resume_state={"brief_name": brief_name},
        resumed_from_run_id=int(latest_run["id"]) if resume and latest_run else None,
        recruiter_id=recruiter_id,
        **pinning_kwargs,
    )

    if resume and not had_runtime_before and legacy_state_exists:
        import_legacy_state(run_id)
        rebuild_artifacts(run_id)
        return run_id, project_linkedin_progress(store, run_id)

    if initial_progress is not None:
        sync_progress(run_id, initial_progress, experiment_states=experiment_states)
        return run_id, project_linkedin_progress(store, run_id)

    progress = Progress(brief_name=brief_name)
    sync_progress(run_id, progress, experiment_states=experiment_states)
    return run_id, progress
