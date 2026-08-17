"""Tests for the Phase D Slice D1 brief library aggregator
(``cloris.control_plane.aggregate_briefs``). Pins the contract:

- Walks `_scan_authored_briefs` and decorates each with a
  ``brief_id`` from ``derive_brief_id(path)``.
- Per-brief run metadata: ``last_run_*``, ``total_runs``,
  ``total_saves`` from ``runtime_state.sqlite3``.
- Briefs with no runs come back with ``total_runs=0``,
  ``total_saves=0``, ``last_run_*=None`` (newly authored briefs).
- ``decorate_runs=False`` skips the runtime-state lookup and returns
  the picker shape (no ``brief_id``, no run fields).
- Failed-state candidates are filtered from the saves count
  (matches the workspace aggregator's filter from C-bis 0.3).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from cloris.control_plane import aggregate_briefs
from shared.runtime_state.store import RuntimeStateStore


@pytest.fixture(autouse=True)
def _isolate_project_root(tmp_path, monkeypatch):
    """``_scan_authored_briefs`` calls ``path.relative_to(_PROJECT_ROOT)``
    so a test config_dir outside the project tree raises. Patch the
    sentinel to tmp_path so test fixtures stay isolated.

    Also stub ``derive_brief_id`` to return a deterministic hash from
    the brief filename — we test the aggregator's wiring, not the brief
    loader (which insists on a richer V2 schema than the minimal test
    fixtures provide)."""

    import cloris.api as cloris_api
    import shared.output_paths as output_paths

    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)

    def _fake_state_key(*, brief_path, **_kwargs):
        return Path(brief_path).stem  # e.g. brief-alpha.json → brief-alpha

    monkeypatch.setattr(output_paths, "derive_brief_id", _fake_state_key)


def _seed_brief(config_dir: Path, name: str, role_title: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"brief-{name}.json"
    payload = {
        "id": name,
        "role_title": role_title,
        "role_summary": "Test role.",
        "geography": "Test",
        # Minimal V2 schema keys so the loader doesn't reject;
        # the aggregator ignores everything except role_title.
    }
    path.write_text(json.dumps(payload))
    return path


def _seed_run(state_dir: Path, *, brief_id: str, source: str = "linkedin") -> int:
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    return store.start_run(
        source=source,
        brief_id=brief_id,
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": brief_id},
    )


def test_aggregate_briefs_returns_one_entry_per_authored_brief(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    _seed_brief(config_dir, "alpha", "Alpha Role")
    _seed_brief(config_dir, "beta", "Beta Role")

    briefs = aggregate_briefs(config_dir=config_dir, state_root=tmp_path / "state")

    assert len(briefs) == 2
    titles = {b.role_title for b in briefs}
    assert titles == {"Alpha Role", "Beta Role"}


def test_aggregate_briefs_decorates_brief_id_via_derive_brief_id(
    tmp_path: Path,
) -> None:
    """The brief_id field must equal the same hash that runs.brief_id
    carries — that's the whole point. Without this guarantee the
    library cards can't find their runs."""

    config_dir = tmp_path / "config"
    _seed_brief(config_dir, "alpha", "Alpha Role")

    briefs = aggregate_briefs(config_dir=config_dir, state_root=tmp_path / "state")

    # Stubbed derive_brief_id returns the path stem, e.g. brief-alpha.
    assert briefs[0].brief_id == "brief-alpha"


def test_aggregate_briefs_returns_zero_runs_for_unrun_brief(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    _seed_brief(config_dir, "fresh", "Just Authored")

    briefs = aggregate_briefs(config_dir=config_dir, state_root=tmp_path / "state")

    assert len(briefs) == 1
    b = briefs[0]
    assert b.total_runs == 0
    assert b.total_saves == 0
    assert b.last_run_id is None
    assert b.last_run_at is None
    assert b.last_run_status is None
    assert b.last_run_source is None


def test_aggregate_briefs_decorates_run_metadata_when_runs_exist(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    path = _seed_brief(config_dir, "alpha", "Alpha Role")
    brief_id = path.stem  # matches the stubbed derive_brief_id

    state_root = tmp_path / "state"
    state_dir = state_root / "linkedin" / brief_id
    state_dir.mkdir(parents=True)
    run_id = _seed_run(state_dir, brief_id=brief_id, source="linkedin")

    briefs = aggregate_briefs(config_dir=config_dir, state_root=state_root)

    assert len(briefs) == 1
    b = briefs[0]
    assert b.total_runs == 1
    assert b.last_run_id == run_id
    assert b.last_run_status == "running"
    assert b.last_run_source == "linkedin"
    assert b.last_run_at is not None


def test_aggregate_briefs_counts_saves_filtering_failed_state(
    tmp_path: Path,
) -> None:
    """Mirrors C-bis 0.3 — failed-state candidates with terminal_decision
    SAVE don't count toward the library's save count."""

    config_dir = tmp_path / "config"
    path = _seed_brief(config_dir, "alpha", "Alpha Role")
    brief_id = path.stem  # matches the stubbed derive_brief_id

    state_root = tmp_path / "state"
    state_dir = state_root / "linkedin" / brief_id
    state_dir.mkdir(parents=True)
    run_id = _seed_run(state_dir, brief_id=brief_id)

    # Inject 2 healthy SAVEs + 1 failed-state SAVE directly into the
    # candidates table. The aggregator should count 2.
    db_path = state_dir / "runtime_state.sqlite3"
    conn = sqlite3.connect(str(db_path))
    try:
        for i, lifecycle in enumerate([
            "full_terminal",
            "full_terminal",
            "failed_terminal",
        ]):
            conn.execute(
                "INSERT INTO candidates "
                "(source, brief_id, identity_key, display_name, profile_url, "
                "current_lifecycle_state, terminal_decision, "
                "terminal_payload_json, first_seen_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "linkedin", brief_id, f"li-{i}", f"Pat {i}",
                    f"https://example.com/{i}",
                    lifecycle, "SAVE", "{}",
                    "2026-04-29T00:00:00+00:00",
                    "2026-04-29T00:00:00+00:00",
                ),
            )
        conn.commit()
    finally:
        conn.close()

    briefs = aggregate_briefs(config_dir=config_dir, state_root=state_root)

    assert briefs[0].total_saves == 2  # Failed-state row excluded.


def test_aggregate_briefs_decorate_runs_false_returns_picker_shape(
    tmp_path: Path,
) -> None:
    """When the picker doesn't need run metadata, the aggregator
    returns the cheaper picker shape (no brief_id, no run fields)."""

    config_dir = tmp_path / "config"
    _seed_brief(config_dir, "alpha", "Alpha Role")

    briefs = aggregate_briefs(
        config_dir=config_dir,
        state_root=tmp_path / "state",
        decorate_runs=False,
    )

    assert len(briefs) == 1
    b = briefs[0]
    assert b.role_title == "Alpha Role"
    assert b.brief_id is None
    assert b.total_runs == 0
    assert b.last_run_id is None


def test_aggregate_briefs_walks_nested_layout(tmp_path: Path) -> None:
    """Hybrid catalog layout — _scan_authored_briefs uses rglob, so a
    brief at `config/<dir>/brief-fde.json` is picked up alongside
    flat briefs."""

    config_dir = tmp_path / "config"
    _seed_brief(config_dir, "flat", "Flat Brief")
    nested_dir = config_dir / "nested-role"
    _seed_brief(nested_dir, "nested", "Nested Brief")

    briefs = aggregate_briefs(config_dir=config_dir, state_root=tmp_path / "state")

    titles = {b.role_title for b in briefs}
    assert titles == {"Flat Brief", "Nested Brief"}
