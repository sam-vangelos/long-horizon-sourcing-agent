"""Admin helpers for runtime_state repair and projection maintenance."""

from __future__ import annotations

from pathlib import Path

from .projections import (
    write_github_stage_projections,
    write_github_progress_projection,
    write_linkedin_candidate_history_projection,
    write_linkedin_progress_projection,
    write_linkedin_search_memory_projection,
    write_linkedin_stage_projections,
)
from .store import RuntimeStateStore


def rebuild_compat_projections(
    store: RuntimeStateStore,
    *,
    run_id: int,
    output_dir: str | Path,
) -> None:
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"run not found: {run_id}")
    output_dir = Path(output_dir)
    source = run["source"]
    brief_id = run["brief_id"]
    if source == "github":
        write_github_progress_projection(store, run_id, output_dir / "progress.json")
        write_github_stage_projections(
            store,
            brief_id=brief_id,
            output_dir=output_dir,
        )
    elif source == "linkedin":
        write_linkedin_progress_projection(store, run_id, output_dir / "progress.json")
        write_linkedin_candidate_history_projection(
            store,
            brief_id=brief_id,
            path=output_dir / f"candidate_history-{brief_id}.jsonl",
        )
        write_linkedin_search_memory_projection(
            store,
            brief_id=brief_id,
            path=output_dir / f"search_memory-{brief_id}.json",
        )
        write_linkedin_stage_projections(
            store,
            brief_id=brief_id,
            output_dir=output_dir,
        )


def inspect_orphaned_attempts(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
) -> list[dict]:
    return store.list_orphaned_attempts(source=source, brief_id=brief_id)


def requeue_work_unit(
    store: RuntimeStateStore,
    *,
    run_id: int,
    kind: str,
    source_unit_id: str,
    output_dir: str | Path,
) -> None:
    store.requeue_work_unit(run_id, kind=kind, source_unit_id=source_unit_id)
    rebuild_compat_projections(store, run_id=run_id, output_dir=output_dir)


def clear_candidate_terminal_state(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    identity_key: str,
) -> None:
    store.clear_candidate_terminal_state(source=source, brief_id=brief_id, identity_key=identity_key)
    store.invalidate_candidate_side_effects(
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
    )


def inspect_candidate_side_effects(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    status: str | None = None,
    identity_key: str | None = None,
) -> list[dict]:
    return store.list_candidate_side_effects(
        source=source,
        brief_id=brief_id,
        status=status,
        identity_key=identity_key,
    )


def replay_candidate_side_effect(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    identity_key: str,
    effect_type: str,
) -> int:
    return store.invalidate_candidate_side_effects(
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        effect_type=effect_type,
    )
