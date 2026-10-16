"""LinkedIn runtime-state artifact and projection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Collection

from shared.runtime_state.admin import rebuild_compat_projections
from shared.runtime_state.projections import (
    project_linkedin_candidate_history,
    project_linkedin_progress,
    project_linkedin_search_memory,
)
from shared.runtime_state.store import RuntimeStateStore
from shared.schemas import Progress


def load_linkedin_progress(store: RuntimeStateStore, run_id: int) -> Progress:
    return project_linkedin_progress(store, run_id)


def load_linkedin_search_memory(store: RuntimeStateStore, *, brief_id: str) -> dict:
    return project_linkedin_search_memory(store, brief_id=brief_id)


def load_linkedin_history(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    save_decisions: Collection[str],
) -> tuple[set[str], dict[str, str], set[str]]:
    blocked_urls = set(store.list_terminal_identity_keys(source="linkedin", brief_id=brief_id))
    prior_outcomes: dict[str, str] = {}
    saved_urls: set[str] = set()
    for record in project_linkedin_candidate_history(store, brief_id=brief_id):
        url = record.get("profile_url", "")
        outcome = record.get("outcome", "")
        if url:
            prior_outcomes[url] = outcome
            if outcome in save_decisions:
                saved_urls.add(url)
    return blocked_urls, prior_outcomes, saved_urls


def rebuild_linkedin_artifacts(
    store: RuntimeStateStore,
    *,
    run_id: int,
    output_dir: str | Path,
) -> None:
    rebuild_compat_projections(
        store,
        run_id=run_id,
        output_dir=output_dir,
    )
