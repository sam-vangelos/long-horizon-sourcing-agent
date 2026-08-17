"""Tests for the Phase C, slice C2 workspace aggregator
(`cloris.control_plane.aggregate_workspace`).

Pins the contract of ``GET /api/workspace/{source}/{state_key}``:

- Missing DB / no runs collapses to ``None`` so the route returns 404.
- A state dir with runs but no SAVE candidates returns a populated
  :class:`WorkspaceResponse` with ``total_saves=0`` and an empty list.
- ``brief_id`` is pulled from the most-recent run; brief context (role
  title, project) reads from that run's ``brief_snapshot_json``.
- Save reason and confidence parse from the candidate's terminal payload.
- ``saves_this_week`` is a 7-day rolling count.
- Cross-source candidates with the same brief_id are correctly filtered.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cloris.control_plane import aggregate_workspace
from shared.runtime_state.store import RuntimeStateStore


def _build_state_dir(tmp_path: Path, source: str, key: str) -> Path:
    state_dir = tmp_path / source / key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _seed_run(
    state_dir: Path,
    *,
    source: str = "linkedin",
    brief_id: str = "brief-1",
    brief_snapshot: dict | None = None,
    mode: str = "fresh",
) -> tuple[RuntimeStateStore, int]:
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    snapshot_json = (
        json.dumps(brief_snapshot) if brief_snapshot is not None else None
    )
    run_id = store.start_run(
        source=source,
        brief_id=brief_id,
        output_dir=str(state_dir),
        mode=mode,
        resume_state={"brief_name": brief_id},
        brief_snapshot_json=snapshot_json,
    )
    return store, run_id


def _save_candidate(
    store: RuntimeStateStore,
    *,
    run_id: int,
    source: str,
    brief_id: str,
    identity_key: str,
    display_name: str,
    profile_url: str,
    decision: str,
    payload: dict | None = None,
) -> int:
    """Walk a candidate to full_terminal with a SAVE-class decision."""

    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=display_name,
        profile_url=profile_url,
    )

    snippet_id = store.start_attempt(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        stage="snippet",
        display_name=display_name,
        profile_url=profile_url,
    )
    store.finish_attempt_success(
        attempt_id=snippet_id,
        new_state="snippet_extracted",
        payload={},
        run_id=run_id,
    )

    store.set_candidate_state(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="facial_started",
    )
    facial_id = store.start_attempt(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        stage="facial",
    )
    store.finish_attempt_success(
        attempt_id=facial_id,
        new_state="facial_terminal",
        payload={},
        run_id=run_id,
    )

    store.set_candidate_state(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="full_started",
    )
    full_id = store.start_attempt(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        stage="full",
    )
    store.finish_attempt_success(
        attempt_id=full_id,
        new_state="full_terminal",
        terminal_decision=decision,
        payload=payload or {},
        run_id=run_id,
    )

    import sqlite3

    conn = sqlite3.connect(
        f"file:{store.db_path}?mode=ro", uri=True
    )
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id FROM candidates WHERE source=? AND brief_id=? AND identity_key=?",
            (source, brief_id, identity_key),
        ).fetchone()
        assert row is not None
        return int(row["id"])
    finally:
        conn.close()


def test_returns_none_when_no_state_dirs_match_brief_id(tmp_path: Path) -> None:
    # No state_dirs under tmp_path → no matches → 404.
    workspace = aggregate_workspace(
        brief_id="brief-1", state_root=tmp_path
    )

    assert workspace is None


def test_returns_none_when_state_dir_has_no_runs(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    # Touch the DB by instantiating the store without seeding a run.
    RuntimeStateStore(state_dir / "runtime_state.sqlite3")

    workspace = aggregate_workspace(
        brief_id="brief-1", state_root=tmp_path
    )

    assert workspace is None


def test_returns_empty_workspace_when_runs_exist_but_no_saves(
    tmp_path: Path,
) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(
        state_dir,
        brief_snapshot={"role_title": "Test Role"},
    )
    # A REJECT candidate — should NOT show up in the workspace.
    _save_candidate(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-rejected",
        display_name="Rejected User",
        profile_url="https://linkedin.com/in/rejected",
        decision="REJECT",
    )

    workspace = aggregate_workspace(
        brief_id="brief-1", state_root=tmp_path
    )

    assert workspace is not None
    assert workspace.total_saves == 0
    assert workspace.candidates == []
    assert workspace.brief_role_title == "Test Role"
    assert workspace.latest_run is not None
    assert workspace.latest_run.run_id == run_id
    assert workspace.latest_run.source == "linkedin"
    assert workspace.latest_run.state_key == "key"
    assert workspace.sources == ["linkedin"]


def test_returns_full_workspace_with_saves(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(
        state_dir,
        brief_snapshot={
            "role_title": "Senior Forward Deployed Engineer",
            "linkedin_project": "FDE NYC",
        },
    )
    _save_candidate(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-pat",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
        decision="SAVE",
        payload={"confidence": 0.92, "save_reason": "Strong systems-design background"},
    )
    _save_candidate(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-alex",
        display_name="Alex Smith",
        profile_url="https://linkedin.com/in/alex",
        decision="INFERENTIAL_SAVE",
        payload={"confidence": 0.78, "save_reason": "Adjacent senior staff at peer co"},
    )

    workspace = aggregate_workspace(
        brief_id="brief-1", state_root=tmp_path
    )

    assert workspace is not None
    assert workspace.brief_id == "brief-1"
    assert workspace.sources == ["linkedin"]
    assert workspace.brief_role_title == "Senior Forward Deployed Engineer"
    assert workspace.brief_linkedin_project == "FDE NYC"
    assert workspace.latest_run is not None
    assert workspace.latest_run.run_id == run_id
    assert workspace.latest_run.source == "linkedin"
    assert workspace.total_saves == 2
    assert workspace.saves_this_week == 2  # both touched today
    assert workspace.last_save_at is not None

    names = {c.display_name for c in workspace.candidates}
    assert names == {"Pat Doe", "Alex Smith"}

    pat = next(c for c in workspace.candidates if c.display_name == "Pat Doe")
    assert pat.source == "linkedin"
    assert pat.terminal_decision == "SAVE"
    assert pat.confidence == 0.92
    assert pat.save_reason == "Strong systems-design background"

    alex = next(c for c in workspace.candidates if c.display_name == "Alex Smith")
    assert alex.source == "linkedin"
    assert alex.terminal_decision == "INFERENTIAL_SAVE"
    assert alex.save_reason == "Adjacent senior staff at peer co"


def test_failed_state_candidates_are_filtered_from_workspace(
    tmp_path: Path,
) -> None:
    """Phase C-bis 0.3: candidates with current_lifecycle_state in the
    failed_* family don't appear in the workspace list, even if their
    terminal_decision is in the SAVE family. The aggregator filters at
    the SQL level so the response is shorter and the UI doesn't have
    to render dead-state rows."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-1")
    # A "good" save — should appear.
    _save_candidate(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-good",
        display_name="Good Save",
        profile_url="https://linkedin.com/in/good",
        decision="SAVE",
    )
    # A failed-state row with terminal_decision=SAVE — the lifecycle
    # filter (not the decision filter) should exclude it from the
    # workspace. We seed the row via discovery, then UPDATE both
    # current_lifecycle_state and terminal_decision directly via SQL —
    # the store's transition guard prevents this combination via the
    # set_candidate_state / finish_attempt_success paths.
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-failed",
        display_name="Failed Save",
        profile_url="https://linkedin.com/in/failed",
    )
    import sqlite3
    conn = sqlite3.connect(str(store.db_path))
    try:
        conn.execute(
            "UPDATE candidates SET terminal_decision='SAVE', "
            "current_lifecycle_state='failed_terminal' "
            "WHERE source='linkedin' AND brief_id='brief-1' AND identity_key='li-failed'"
        )
        conn.commit()
    finally:
        conn.close()

    workspace = aggregate_workspace(brief_id="brief-1", state_root=tmp_path)

    assert workspace is not None
    assert workspace.total_saves == 1
    names = [c.display_name for c in workspace.candidates]
    assert names == ["Good Save"]


def test_aggregates_saves_across_multiple_runs(tmp_path: Path) -> None:
    """A brief that's been re-run should accumulate saves across runs."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id_first = _seed_run(state_dir)
    _save_candidate(
        store,
        run_id=run_id_first,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        display_name="First Save",
        profile_url="https://linkedin.com/in/first",
        decision="SAVE",
    )

    # Second run, second save.
    run_id_second = store.start_run(
        source="linkedin",
        brief_id="brief-1",
        output_dir=str(state_dir),
        mode="resume",
        resume_state={"brief_name": "brief-1"},
    )
    _save_candidate(
        store,
        run_id=run_id_second,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-2",
        display_name="Second Save",
        profile_url="https://linkedin.com/in/second",
        decision="SAVE",
    )

    workspace = aggregate_workspace(
        brief_id="brief-1", state_root=tmp_path
    )

    assert workspace is not None
    assert workspace.total_saves == 2
    assert workspace.latest_run is not None
    assert workspace.latest_run.run_id == run_id_second


# Phase F Slice F6 tests. Cross-source identity resolution (F3) feeds
# the workspace aggregator so a person aggregated under LinkedIn AND
# GitHub renders ONE card with cross_source_links surfacing the
# secondary source. The test patches IDENTITY_ROOT so each test gets
# its own identity.sqlite3 isolated to tmp_path.


def _isolate_identity_db(monkeypatch, tmp_path: Path) -> None:
    identity_root = tmp_path / "_identity"
    identity_root.mkdir(parents=True, exist_ok=True)
    import shared.output_paths as output_paths

    monkeypatch.setattr(output_paths, "IDENTITY_ROOT", identity_root)


def test_workspace_merges_cross_source_via_handle_match(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_identity_db(monkeypatch, tmp_path)

    li_dir = _build_state_dir(tmp_path, "linkedin", "li-key-f6")
    li_store, li_run_id = _seed_run(
        li_dir,
        source="linkedin",
        brief_id="brief-f6",
        brief_snapshot={"role_title": "Multi-source Role"},
    )
    _save_candidate(
        li_store,
        run_id=li_run_id,
        source="linkedin",
        brief_id="brief-f6",
        identity_key="li-merged",
        display_name="Eri Barrett",
        profile_url="https://www.linkedin.com/in/eri-barrett/",
        decision="SAVE",
    )

    gh_dir = _build_state_dir(tmp_path, "github", "gh-key-f6")
    gh_store, gh_run_id = _seed_run(
        gh_dir,
        source="github",
        brief_id="brief-f6",
        brief_snapshot={"role_title": "Multi-source Role"},
    )
    _save_candidate(
        gh_store,
        run_id=gh_run_id,
        source="github",
        brief_id="brief-f6",
        identity_key="erosika",
        display_name="erosika",
        profile_url="https://github.com/erosika",
        decision="SAVE",
        payload={
            "candidate_record": {
                "user": {"name": "Eri Barrett"},
                "contact": {"linkedin_url": "https://www.linkedin.com/in/eri-barrett/"},
            }
        },
    )

    workspace = aggregate_workspace(brief_id="brief-f6", state_root=tmp_path)

    assert workspace is not None
    # Two saves seeded, but they're the SAME human → one card with a
    # cross_source_links entry pointing at the secondary source.
    assert workspace.total_saves == 1
    primary = workspace.candidates[0]
    # LinkedIn ranks first as the primary card by _pick_primary_link heuristic.
    assert primary.source == "linkedin"
    assert len(primary.cross_source_links) == 1
    other = primary.cross_source_links[0]
    assert other.source == "github"
    assert other.link_kind == "auto_strong"
    # describe_merge_signal output is editorial prose, not a raw enum.
    assert "LinkedIn handle" in other.describe


def test_workspace_renders_unmerged_when_no_match(
    tmp_path: Path, monkeypatch
) -> None:
    """Two distinct people on different sources stay as TWO cards."""

    _isolate_identity_db(monkeypatch, tmp_path)

    li_dir = _build_state_dir(tmp_path, "linkedin", "li-distinct")
    li_store, li_run_id = _seed_run(
        li_dir,
        source="linkedin",
        brief_id="brief-distinct",
        brief_snapshot={"role_title": "Distinct Role"},
    )
    _save_candidate(
        li_store,
        run_id=li_run_id,
        source="linkedin",
        brief_id="brief-distinct",
        identity_key="li-alice",
        display_name="Alice Anderson",
        profile_url="https://www.linkedin.com/in/alice-anderson/",
        decision="SAVE",
    )

    gh_dir = _build_state_dir(tmp_path, "github", "gh-distinct")
    gh_store, gh_run_id = _seed_run(
        gh_dir,
        source="github",
        brief_id="brief-distinct",
        brief_snapshot={"role_title": "Distinct Role"},
    )
    _save_candidate(
        gh_store,
        run_id=gh_run_id,
        source="github",
        brief_id="brief-distinct",
        identity_key="bobcoder",
        display_name="bobcoder",
        profile_url="https://github.com/bobcoder",
        decision="SAVE",
        payload={
            "candidate_record": {
                "user": {"name": "Bob Brown"},
                "contact": {"linkedin_url": ""},
            }
        },
    )

    workspace = aggregate_workspace(
        brief_id="brief-distinct", state_root=tmp_path
    )

    assert workspace is not None
    # Two distinct people → two cards, no cross-source links.
    assert workspace.total_saves == 2
    for card in workspace.candidates:
        assert card.cross_source_links == []
