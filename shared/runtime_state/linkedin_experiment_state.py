"""LinkedIn experiment-state loading helpers."""

from __future__ import annotations

import json

from linkedin.search_intelligence import (
    LinkedInExperimentState,
    bootstrap_experiment_state,
    seed_structured_filters_onto_variants,
)
from shared.schemas import Progress, SearchString

from .store import LINKEDIN_STRING_KIND, RuntimeStateStore


def load_linkedin_experiment_states(
    *,
    store: RuntimeStateStore,
    run_id: int,
    progress: Progress | None = None,
) -> dict[int, LinkedInExperimentState]:
    states: dict[int, LinkedInExperimentState] = {}
    progress_lookup = {item.id: item for item in (progress.strings if progress else [])}
    for row in store.list_work_units(run_id, kind=LINKEDIN_STRING_KIND):
        payload = _json_loads(row["payload_json"])
        checkpoint = _json_loads(row["checkpoint_json"])
        search_string = progress_lookup.get(int(payload.get("id") or row["source_unit_id"]))
        if search_string is None:
            search_string = SearchString.from_dict(payload)
        state = LinkedInExperimentState.from_dict(checkpoint.get("experiment_state"))
        if state is None:
            state = bootstrap_experiment_state(search_string)
        else:
            _align_checkpoint_state_with_search_string(state, search_string)
        state.apply_shadow(search_string)
        states[search_string.id] = state
    return states


def _align_checkpoint_state_with_search_string(
    state: LinkedInExperimentState,
    search_string: SearchString,
) -> None:
    """Mirror bootstrap's compat re-seeding on persisted checkpoint loads."""
    surface = str(getattr(search_string, "surface", "") or "").strip()
    if surface:
        root = state.variants.get("root")
        if root is not None and not root.surface:
            root.surface = surface
        active = state.variants.get(state.active_variant_id)
        if active is not None and not active.surface:
            active.surface = surface

    structured_filters = dict(getattr(search_string, "structured_filters", {}) or {})
    if structured_filters:
        seed_structured_filters_onto_variants(
            structured_filters,
            list(state.variants.values()),
        )


def _json_loads(raw: str | None) -> dict:
    if not raw:
        return {}
    return json.loads(raw)
