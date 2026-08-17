"""Tests for the Cloris status aggregator (Slice 2).

These tests pin the read-only contract of :mod:`cloris.control_plane`:

- The aggregator opens canonical SQLite read-only and never instantiates
  :class:`shared.runtime_state.store.RuntimeStateStore` in production paths.
  Test fixtures *may* use ``RuntimeStateStore`` to build a real DB — that is
  the cleanest way to exercise the read path.
- An empty state root returns an empty list, not an error.
- A state dir with no DB shows ``runtime_state_present=False`` and null run
  fields.
- A real ``runs`` row is forwarded verbatim (no semantic shaping).
- The aggregator is source-symmetric across LinkedIn and GitHub.
- Repeated polls do not mutate the underlying DB (the canonical read-only
  invariant Slice 2 must hold).
- A corrupt DB collapses cleanly to ``latest_run=None`` instead of crashing
  the whole response.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

from cloris import control_plane
from cloris.control_plane import aggregate_status
from shared.runtime_state.store import RuntimeStateStore


assert "RuntimeStateStore" not in inspect.getsource(control_plane), (
    "cloris/control_plane.py must not import or reference RuntimeStateStore "
    "in production paths (Slice 2 read-only contract)."
)


def _build_state_dir(state_root: Path, source: str, key: str) -> Path:
    state_dir = state_root / source / key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def test_aggregate_empty_state_root(tmp_path: Path) -> None:
    response = aggregate_status(tmp_path)

    assert response.slice == "v0-shell-slice-4"
    assert response.entries == []


def test_aggregate_state_dir_without_db(tmp_path: Path) -> None:
    _build_state_dir(tmp_path, "linkedin", "some-key")

    response = aggregate_status(tmp_path)

    assert response.slice == "v0-shell-slice-4"
    assert len(response.entries) == 1

    entry = response.entries[0]
    assert entry.source == "linkedin"
    assert entry.state_key == "some-key"
    assert entry.runtime_state_present is False
    assert entry.latest_run is None
    assert entry.brief_id_from_run is None


def test_aggregate_state_dir_with_run_row(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    db_path = state_dir / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-1",
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": "brief-1"},
    )
    store.finish_run(run_id, "completed")

    response = aggregate_status(tmp_path)

    assert len(response.entries) == 1
    entry = response.entries[0]
    assert entry.source == "linkedin"
    assert entry.state_key == "key"
    assert entry.runtime_state_present is True
    assert entry.brief_id_from_run == "brief-1"
    assert entry.latest_run is not None
    assert entry.latest_run.status == "completed"
    assert entry.latest_run.id == run_id
    assert entry.latest_run.mode == "fresh"
    assert entry.attention_state == "terminal"
    assert entry.live_signal_eligible is False


def test_aggregate_across_linkedin_and_github(tmp_path: Path) -> None:
    linkedin_dir = _build_state_dir(tmp_path, "linkedin", "li-key")
    li_store = RuntimeStateStore(linkedin_dir / "runtime_state.sqlite3")
    li_store.start_run(
        source="linkedin",
        brief_id="brief-li",
        output_dir=str(linkedin_dir),
        mode="fresh",
        resume_state={"brief_name": "brief-li"},
    )

    github_dir = _build_state_dir(tmp_path, "github", "gh-key")
    gh_store = RuntimeStateStore(github_dir / "runtime_state.sqlite3")
    gh_run_id = gh_store.start_run(
        source="github",
        brief_id="brief-gh",
        output_dir=str(github_dir),
        mode="fresh",
        resume_state={"brief_name": "brief-gh"},
    )
    gh_store.finish_run(gh_run_id, "interrupted")

    response = aggregate_status(tmp_path)

    by_source = {entry.source: entry for entry in response.entries}
    assert set(by_source) == {"linkedin", "github"}

    li_entry = by_source["linkedin"]
    assert li_entry.state_key == "li-key"
    assert li_entry.runtime_state_present is True
    assert li_entry.latest_run is not None
    assert li_entry.latest_run.status == "running"
    assert li_entry.brief_id_from_run == "brief-li"
    assert li_entry.attention_state == "recovering"
    assert li_entry.live_signal_eligible is False

    gh_entry = by_source["github"]
    assert gh_entry.state_key == "gh-key"
    assert gh_entry.runtime_state_present is True
    assert gh_entry.latest_run is not None
    assert gh_entry.latest_run.status == "interrupted"
    assert gh_entry.brief_id_from_run == "brief-gh"


def test_aggregator_does_not_write_to_db(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "ro-key")
    db_path = state_dir / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-ro",
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": "brief-ro"},
    )
    store.finish_run(run_id, "completed")

    pinned_mtime_ns = 1_700_000_000_000_000_000
    os.utime(db_path, ns=(pinned_mtime_ns, pinned_mtime_ns))
    baseline_mtime = db_path.stat().st_mtime_ns

    for _ in range(3):
        aggregate_status(tmp_path)
        assert db_path.stat().st_mtime_ns == baseline_mtime, (
            "aggregate_status must be a pure read; db mtime changed across polls"
        )


def test_aggregator_handles_corrupt_db_gracefully(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "corrupt-key")
    db_path = state_dir / "runtime_state.sqlite3"
    db_path.write_bytes(b"not a sqlite database")

    response = aggregate_status(tmp_path)

    assert len(response.entries) == 1
    entry = response.entries[0]
    assert entry.runtime_state_present is True
    # Phase 1.4: corrupt DB now sets a distinct flag so the UI can render
    # "runtime state unreadable" instead of indistinguishable "no run".
    assert entry.runtime_state_corrupt is True
    assert entry.latest_run is None
    assert entry.brief_id_from_run is None


def test_aggregator_empty_db_is_not_corrupt(tmp_path: Path) -> None:
    """An empty but readable DB (no runs row yet) must NOT be classified
    as corrupt — the user just hasn't run anything for this brief."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "empty-key")
    db_path = state_dir / "runtime_state.sqlite3"
    # Initialize a real schema-shaped DB but never insert a run.
    RuntimeStateStore(db_path)

    response = aggregate_status(tmp_path)

    assert len(response.entries) == 1
    entry = response.entries[0]
    assert entry.runtime_state_present is True
    assert entry.runtime_state_corrupt is False
    assert entry.latest_run is None


def test_aggregator_missing_db_is_not_corrupt(tmp_path: Path) -> None:
    """A state dir without runtime_state.sqlite3 must report
    runtime_state_corrupt=False — corrupt is reserved for "file exists but
    cannot be read." The UI distinguishes "DB missing" from "DB unreadable"
    via separate flags so the remediation guidance differs."""

    _build_state_dir(tmp_path, "linkedin", "no-db-key")

    response = aggregate_status(tmp_path)

    assert len(response.entries) == 1
    entry = response.entries[0]
    assert entry.runtime_state_present is False
    assert entry.runtime_state_corrupt is False


def test_is_runtime_state_corrupt_helper(tmp_path: Path) -> None:
    """Direct unit coverage for the corruption-probe helper. The aggregator
    consumes it once per state dir per poll; bugs here would surface as
    silent miscategorization in the UI."""

    from cloris.control_plane import is_runtime_state_corrupt

    # Missing file: not corrupt (just absent).
    missing = tmp_path / "missing.sqlite3"
    assert is_runtime_state_corrupt(missing) is False

    # Truncated/garbage file: corrupt.
    bad = tmp_path / "bad.sqlite3"
    bad.write_bytes(b"this is not a sqlite db")
    assert is_runtime_state_corrupt(bad) is True

    # Real, schema-initialized DB: not corrupt.
    good = tmp_path / "good.sqlite3"
    RuntimeStateStore(good)
    assert is_runtime_state_corrupt(good) is False


# --- Slice 4: enriched StateDirEntry + linkedin_resumable -----------------


import json
import subprocess
import sys

from cloris.control_plane import linkedin_resumable
from cloris.worker import WORKER_SIDECAR_FILENAME, build_sidecar, write_sidecar


def _write_worker_sidecar(state_dir: Path, **overrides) -> Path:
    """Write a worker.json with sensible defaults plus any overrides.

    Centralizes the build_sidecar plumbing so each Slice-4 test only has
    to express what it cares about (typically: pid, mode).

    Phase 1.6 note: started_at and heartbeat_at default to "now" rather
    than a fixed past date so the aggregator's alive_silent classifier
    (heartbeat older than 5 minutes ⇒ silent) doesn't fire on tests that
    only care about alive/stale, not staleness duration. Tests that
    deliberately want an old heartbeat pass started_at via overrides.
    """

    from datetime import datetime, timezone

    fresh_now = datetime.now(timezone.utc).isoformat()
    payload = build_sidecar(
        source="linkedin",
        brief_id="brief-4",
        brief_path=str(state_dir / "brief.json"),
        output_dir=str(state_dir),
        mode="fresh",
        input_mode="concurrent",
        started_at=fresh_now,
        pid=os.getpid(),
        run_id=None,
    )
    payload.update(overrides)
    return write_sidecar(state_dir, payload)


def test_aggregate_status_slice_tag_bumped(tmp_path: Path) -> None:
    """Slice 4 bumps the StatusResponse slice literal to v0-shell-slice-4."""

    response = aggregate_status(tmp_path)

    assert response.slice == "v0-shell-slice-4"


def test_aggregate_enriches_with_alive_worker_sidecar(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "alive-key")
    db_path = state_dir / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    store.start_run(
        source="linkedin",
        brief_id="brief-4",
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": "brief-4"},
    )

    _write_worker_sidecar(state_dir, pid=os.getpid())

    response = aggregate_status(tmp_path)

    assert len(response.entries) == 1
    entry = response.entries[0]
    assert entry.worker_json_present is True
    assert entry.worker_pid == os.getpid()
    assert entry.worker_alive is True
    assert entry.worker_state == "alive"
    assert entry.worker_mode == "fresh"
    assert entry.worker_input_mode == "concurrent"
    assert entry.brief_path_from_worker == str(state_dir / "brief.json")


def test_aggregate_enriches_with_stale_worker_sidecar_dead_pid(
    tmp_path: Path,
) -> None:
    """A sidecar pointing at a dead PID is classified as stale."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "dead-pid-key")

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    dead_pid = child.pid

    _write_worker_sidecar(state_dir, pid=dead_pid)

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.worker_json_present is True
    assert entry.worker_pid == dead_pid
    assert entry.worker_alive is False
    assert entry.worker_state == "stale"


def test_aggregate_enriches_with_stale_worker_sidecar_malformed_pid(
    tmp_path: Path,
) -> None:
    """Sidecar with a non-int pid is classified as stale, pid forwarded as None."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "bad-pid-key")
    sidecar_path = state_dir / WORKER_SIDECAR_FILENAME
    sidecar_path.write_text(
        json.dumps(
            {
                "pid": "not-an-int",
                "source": "linkedin",
                "brief_id": "brief-4",
                "brief_path": str(state_dir / "brief.json"),
                "output_dir": str(state_dir),
                "run_id": None,
                "started_at": "2026-04-27T18:00:00+00:00",
                "heartbeat_at": "2026-04-27T18:00:00+00:00",
                "mode": "fresh",
                "input_mode": "concurrent",
                "launcher_version": "cloris-v0-slice-4",
            },
            indent=2,
            sort_keys=True,
        )
    )

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.worker_json_present is True
    assert entry.worker_pid is None
    assert entry.worker_alive is None
    assert entry.worker_state == "stale"


def test_aggregate_missing_worker_sidecar_classified_as_missing(
    tmp_path: Path,
) -> None:
    """No worker.json means worker_state == 'missing' and worker_* defaults hold."""

    _build_state_dir(tmp_path, "linkedin", "missing-key")

    response = aggregate_status(tmp_path)
    entry = response.entries[0]
    assert entry.worker_state == "missing"
    assert entry.worker_json_present is False
    assert entry.worker_pid is None
    assert entry.worker_alive is None
    assert entry.worker_mode is None
    assert entry.worker_input_mode is None
    assert entry.brief_path_from_worker is None


def test_linkedin_resumable_pending_strings(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "queued-key")
    (state_dir / "progress.json").write_text(
        json.dumps({"strings": [{"id": 1, "status": "queued"}]})
    )

    assert linkedin_resumable(state_dir) is True


def test_linkedin_resumable_pending_block_ids(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "pending-blocks-key")
    (state_dir / "progress.json").write_text(
        json.dumps({"pending_block_string_ids": [1, 2, 3]})
    )

    assert linkedin_resumable(state_dir) is True


def test_linkedin_resumable_no_pending_work(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "done-key")
    (state_dir / "progress.json").write_text(
        json.dumps(
            {
                "strings": [
                    {"id": 1, "status": "done"},
                    {"id": 2, "status": "done"},
                ]
            }
        )
    )

    assert linkedin_resumable(state_dir) is False


def test_linkedin_resumable_missing_progress_json_returns_none(
    tmp_path: Path,
) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "no-progress-key")

    assert linkedin_resumable(state_dir) is None


def test_linkedin_resumable_malformed_returns_none(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "malformed-progress-key")
    (state_dir / "progress.json").write_bytes(b"not json")

    assert linkedin_resumable(state_dir) is None


def test_aggregate_resumable_populated_for_linkedin_only(tmp_path: Path) -> None:
    """LinkedIn entries get resumable from progress.json; GitHub entries are None."""

    li_dir = _build_state_dir(tmp_path, "linkedin", "li-resumable-key")
    (li_dir / "progress.json").write_text(
        json.dumps({"strings": [{"id": 1, "status": "queued"}]})
    )

    gh_dir = _build_state_dir(tmp_path, "github", "gh-key")
    (gh_dir / "progress.json").write_text(
        json.dumps({"strings": [{"id": 1, "status": "queued"}]})
    )

    response = aggregate_status(tmp_path)
    by_source = {entry.source: entry for entry in response.entries}

    assert by_source["linkedin"].resumable is True
    assert by_source["github"].resumable is None


# ---------------------------------------------------------------------------
# P4.4 — state-dir hygiene. enumerate_state_dirs must skip test-debris dirs
# (the "*_tmp*" naming pattern left behind by
# tests/test_linkedin_session_orchestrator.py calling
# resolve_linkedin_state_dir() without an explicit state_dir override) while
# still surfacing every legitimately-named state dir.
# ---------------------------------------------------------------------------


def test_enumerate_state_dirs_excludes_test_debris_pattern(tmp_path: Path) -> None:
    from cloris.control_plane import enumerate_state_dirs

    _build_state_dir(tmp_path, "linkedin", "exhausted_tmp0rrw7snk")
    _build_state_dir(tmp_path, "linkedin", "in_progress_tmpa0zvg23a")
    _build_state_dir(tmp_path, "linkedin", "missing_tmp47_rwjk3")
    _build_state_dir(tmp_path, "linkedin", "pending_block_tmp3pt46hhe")
    _build_state_dir(tmp_path, "linkedin", "queued_tmp2qzcioca")

    assert list(enumerate_state_dirs(tmp_path)) == []


def test_enumerate_state_dirs_includes_legit_dirs_alongside_debris(
    tmp_path: Path,
) -> None:
    from cloris.control_plane import enumerate_state_dirs

    _build_state_dir(tmp_path, "linkedin", "3000000005")
    _build_state_dir(tmp_path, "linkedin", "head_of_applied_ai_fixture")
    _build_state_dir(tmp_path, "linkedin", "unknown")
    _build_state_dir(tmp_path, "linkedin", "exhausted_tmp0rrw7snk")

    discovered = {key for _source, path in enumerate_state_dirs(tmp_path) for key in [path.name]}

    assert discovered == {"3000000005", "head_of_applied_ai_fixture", "unknown"}
    assert "exhausted_tmp0rrw7snk" not in discovered


def test_aggregate_status_excludes_test_debris_from_counts(tmp_path: Path) -> None:
    """End-to-end: a debris dir must not show up in /api/status entries or
    counts, so recruiter-facing surfaces never see the 74-dir noise."""

    _build_state_dir(tmp_path, "linkedin", "queued_tmp9db4l5z3")
    _build_state_dir(tmp_path, "linkedin", "real-brief-key")

    response = aggregate_status(tmp_path)

    assert len(response.entries) == 1
    assert response.entries[0].state_key == "real-brief-key"
    assert response.counts.orphaned == 1
