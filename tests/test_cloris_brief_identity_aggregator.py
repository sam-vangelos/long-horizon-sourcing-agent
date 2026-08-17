"""Phase 3 aggregator tests: brief_role_title, brief_drift, fallback.

Validates that ``cloris.control_plane.aggregate_status``:

- Surfaces ``brief_role_title`` from ``runs.brief_snapshot_json`` when
  present.
- Surfaces ``brief_drift_since_last_run`` by comparing the on-disk
  hash to ``runs.brief_content_hash``.
- Falls back gracefully (None on all three new fields) for legacy
  rows that pre-date Phase 3 — no NULL handling in the UI required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloris.control_plane import aggregate_status
from shared.brief_identity import compute_brief_identity
from shared.runtime_state.store import RuntimeStateStore


def _build_state_dir(state_root: Path, source: str, key: str) -> Path:
    state_dir = state_root / source / key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _start_run_with_brief(
    store: RuntimeStateStore, brief_path: Path, *, brief_id: str = "brief-3"
) -> int:
    identity = compute_brief_identity(brief_path)
    assert identity is not None
    return store.start_run(
        source="linkedin",
        brief_id=brief_id,
        output_dir=str(brief_path.parent),
        mode="fresh",
        brief_path_at_launch=identity["brief_path_at_launch"],
        brief_content_hash=identity["brief_content_hash"],
        brief_snapshot_json=identity["brief_snapshot_json"],
    )


def test_aggregator_surfaces_brief_role_title_from_snapshot(
    tmp_path: Path,
) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "principal-eng-nyc")
    brief_path = state_dir / "brief.json"
    brief_path.write_text(
        json.dumps(
            {"role_title": "Principal Engineer NYC", "linkedin_project": "PE-NYC"}
        )
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    _start_run_with_brief(store, brief_path)

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.brief_role_title == "Principal Engineer NYC"
    assert entry.brief_linkedin_project == "PE-NYC"


def test_aggregator_brief_drift_false_when_unchanged(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "no-drift")
    brief_path = state_dir / "brief.json"
    brief_path.write_text(json.dumps({"role_title": "Stable"}))
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    _start_run_with_brief(store, brief_path)

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.brief_drift_since_last_run is False


def test_aggregator_brief_drift_true_when_brief_modified(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "drift")
    brief_path = state_dir / "brief.json"
    brief_path.write_text(json.dumps({"role_title": "Original"}))
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    _start_run_with_brief(store, brief_path)

    # Recruiter edits the brief mid-run.
    brief_path.write_text(json.dumps({"role_title": "Modified"}))

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.brief_drift_since_last_run is True


def test_aggregator_brief_drift_none_when_brief_missing(tmp_path: Path) -> None:
    """If the brief file moved or was deleted between runs, drift
    detection cannot decide — return None rather than guessing."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "moved")
    brief_path = state_dir / "brief.json"
    brief_path.write_text(json.dumps({"role_title": "Movable"}))
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    _start_run_with_brief(store, brief_path)

    brief_path.unlink()  # File no longer at the pinned path.

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.brief_drift_since_last_run is None
    # role_title still readable from the stored snapshot — the on-disk
    # file going missing does not invalidate Run Review.
    assert entry.brief_role_title == "Movable"


def test_aggregator_legacy_row_falls_back_to_state_key(tmp_path: Path) -> None:
    """A run row that pre-dates Phase 3 has NULL pinning columns. The
    aggregator must not crash; brief_role_title is None and the UI
    falls back to state_key."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "legacy-key")
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    # No brief_path/hash/snapshot — legacy caller path.
    store.start_run(
        source="linkedin",
        brief_id="legacy-brief",
        output_dir=str(state_dir),
        mode="fresh",
    )

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.brief_role_title is None
    assert entry.brief_linkedin_project is None
    assert entry.brief_drift_since_last_run is None
