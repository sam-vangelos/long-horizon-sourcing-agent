"""Tests for the Phase B run-report aggregator (`cloris.control_plane.aggregate_run_report`).

Pins the contract of the new ``GET /api/run/{source}/{state_key}/{run_id}``
read path:

- Missing state dir / missing DB / missing run all collapse to ``None`` so
  the route can return a clean 404.
- Run lifecycle fields are forwarded verbatim.
- Per-run attempt-health and work-unit-progress mirror the same
  primitives the homescreen uses, applied to a specific run id.
- Decisions are aggregated from the ``candidates`` table joined to
  ``candidate_attempts`` so the report only includes candidates that
  Cloris actually touched in this run.
- Decision counts reflect the full population even when the candidate
  list is truncated.
- Same read-only contract as ``aggregate_status``: no
  RuntimeStateStore in the production source.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from cloris import control_plane
from cloris.control_plane import aggregate_run_report
from shared.runtime_state.store import RuntimeStateStore


assert "RuntimeStateStore" not in inspect.getsource(control_plane), (
    "cloris/control_plane.py must not import or reference the canonical "
    "write-side store class in production paths."
)


def _build_state_dir(tmp_path: Path, source: str, key: str) -> Path:
    state_dir = tmp_path / source / key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _seed_run(state_dir: Path, *, brief_id: str = "brief-1") -> tuple[RuntimeStateStore, int]:
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id=brief_id,
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": brief_id},
    )
    return store, run_id


def _save_candidate_through_lifecycle(
    store: RuntimeStateStore,
    *,
    run_id: int,
    brief_id: str,
    identity_key: str,
    display_name: str,
    profile_url: str,
    decision: str,
    confidence: float,
) -> None:
    """Walk a candidate from discovery through full_terminal with a decision.

    Mirrors production-shaped writes (the lifecycle guard rejects any
    shortcut). Uses ``set_candidate_state`` to advance the lifecycle
    between attempts the same way the orchestrator does.
    """

    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=display_name,
        profile_url=profile_url,
    )

    snippet_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
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
        source="linkedin",
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="facial_started",
    )
    facial_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
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
        source="linkedin",
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="full_started",
    )
    full_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key=identity_key,
        stage="full",
    )
    store.finish_attempt_success(
        attempt_id=full_id,
        new_state="full_terminal",
        terminal_decision=decision,
        payload={"confidence": confidence},
        run_id=run_id,
    )


def test_returns_none_when_state_dir_has_no_db(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")

    report = aggregate_run_report(
        state_dir, source="linkedin", state_key="key", run_id=1
    )

    assert report is None


def test_returns_none_when_run_id_does_not_exist(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    _seed_run(state_dir)

    # run_id 999 does not exist.
    report = aggregate_run_report(
        state_dir, source="linkedin", state_key="key", run_id=999
    )

    assert report is None


def test_basic_run_report_shape(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-alpha")
    store.finish_run(run_id, "completed")

    report = aggregate_run_report(
        state_dir, source="linkedin", state_key="key", run_id=run_id
    )

    assert report is not None
    assert report.slice == "v0-shell-slice-b1"
    assert report.source == "linkedin"
    assert report.state_key == "key"
    # Phase 1B: state_dir + output_dir removed from the wire shape (R9 — no
    # absolute filesystem paths in API responses).
    assert not hasattr(report, "state_dir")
    assert not hasattr(report.run, "output_dir")
    assert report.run.id == run_id
    assert report.run.brief_id == "brief-alpha"
    assert report.run.status == "completed"
    assert report.run.mode == "fresh"
    assert report.run.started_at is not None
    assert report.run.ended_at is not None
    assert report.candidates == []
    assert report.candidates_truncated is False
    assert report.decisions.total == 0
    assert report.decisions.by_decision == {}


def test_run_report_lists_candidates_touched_in_this_run(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-alpha")

    _save_candidate_through_lifecycle(
        store,
        run_id=run_id,
        brief_id="brief-alpha",
        identity_key="alice-id",
        display_name="Alice Apple",
        profile_url="https://www.linkedin.com/in/alice",
        decision="SAVE",
        confidence=0.91,
    )

    report = aggregate_run_report(
        state_dir, source="linkedin", state_key="key", run_id=run_id
    )

    assert report is not None
    assert len(report.candidates) == 1
    cand = report.candidates[0]
    assert cand.display_name == "Alice Apple"
    assert cand.profile_url == "https://www.linkedin.com/in/alice"
    assert cand.terminal_decision == "SAVE"
    assert cand.confidence == 0.91
    assert report.decisions.total == 1
    assert report.decisions.by_decision == {"SAVE": 1}


def test_run_report_excludes_candidates_from_other_runs(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id_a = _seed_run(state_dir, brief_id="brief-alpha")
    store.finish_run(run_id_a, "completed")

    # Second run touches a different candidate. The report for run A must
    # NOT show the candidate touched in run B.
    run_id_b = store.start_run(
        source="linkedin",
        brief_id="brief-alpha",
        output_dir=str(state_dir),
        mode="resume",
        resume_state={"brief_name": "brief-alpha"},
    )

    _save_candidate_through_lifecycle(
        store,
        run_id=run_id_b,
        brief_id="brief-alpha",
        identity_key="bob-id",
        display_name="Bob Berry",
        profile_url="https://www.linkedin.com/in/bob",
        decision="REJECT",
        confidence=0.4,
    )

    report_a = aggregate_run_report(
        state_dir, source="linkedin", state_key="key", run_id=run_id_a
    )
    report_b = aggregate_run_report(
        state_dir, source="linkedin", state_key="key", run_id=run_id_b
    )

    assert report_a is not None
    assert report_a.candidates == []
    assert report_a.decisions.total == 0

    assert report_b is not None
    assert len(report_b.candidates) == 1
    assert report_b.candidates[0].display_name == "Bob Berry"
    assert report_b.decisions.by_decision == {"REJECT": 1}


def test_run_report_decision_counts_aggregate_across_population(
    tmp_path: Path,
) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-alpha")

    fixtures = [
        ("alice-id", "Alice", "SAVE", 0.92),
        ("bob-id", "Bob", "SAVE", 0.81),
        ("carol-id", "Carol", "REJECT", 0.21),
        ("dan-id", "Dan", "FACIAL_NO", 0.18),
    ]
    for identity_key, name, decision, conf in fixtures:
        _save_candidate_through_lifecycle(
            store,
            run_id=run_id,
            brief_id="brief-alpha",
            identity_key=identity_key,
            display_name=name,
            profile_url=f"https://example.com/{identity_key}",
            decision=decision,
            confidence=conf,
        )

    report = aggregate_run_report(
        state_dir, source="linkedin", state_key="key", run_id=run_id
    )

    assert report is not None
    assert report.decisions.total == 4
    assert report.decisions.by_decision == {
        "SAVE": 2,
        "REJECT": 1,
        "FACIAL_NO": 1,
    }
    # All four candidates are listed (well below the 200 cap).
    names = {c.display_name for c in report.candidates}
    assert names == {"Alice", "Bob", "Carol", "Dan"}
    assert report.candidates_truncated is False


def test_run_report_truncates_candidate_list_at_limit(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-alpha")

    for i in range(7):
        identity_key = f"cand-{i:02d}"
        _save_candidate_through_lifecycle(
            store,
            run_id=run_id,
            brief_id="brief-alpha",
            identity_key=identity_key,
            display_name=f"Candidate {i}",
            profile_url=f"https://example.com/{identity_key}",
            decision="SAVE",
            confidence=0.85,
        )

    report = aggregate_run_report(
        state_dir,
        source="linkedin",
        state_key="key",
        run_id=run_id,
        candidate_limit=3,
    )

    assert report is not None
    assert len(report.candidates) == 3
    assert report.candidates_truncated is True
    # Counts span the full population, not just the truncated list.
    assert report.decisions.total == 7
    assert report.decisions.by_decision == {"SAVE": 7}


# Sourcing-judgment kernel P5: the run-report response surface carries
# a ``lane_metrics`` list. Legacy runs (no ``lane_id`` populated on any
# work unit or terminal payload) collapse into a single ``"legacy"``
# bucket; the wire shape stays additive — never absent — so frontend
# consumers can iterate without a ``None`` check.


def test_run_report_lane_metrics_present_with_legacy_bucket_for_legacy_runs(
    tmp_path: Path,
):
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-legacy-lane")

    _save_candidate_through_lifecycle(
        store,
        run_id=run_id,
        brief_id="brief-legacy-lane",
        identity_key="cand-legacy",
        display_name="Legacy Candidate",
        profile_url="https://example.com/legacy",
        decision="SAVE",
        confidence=0.8,
    )

    report = aggregate_run_report(
        state_dir,
        source="linkedin",
        state_key="key",
        run_id=run_id,
    )
    assert report is not None
    # The helper does not seed a work unit, so the lane attribution
    # chain falls back to the ``"legacy"`` bucket and aggregates the
    # save count under it. The wire still carries a list (never None).
    assert isinstance(report.lane_metrics, list)
    assert len(report.lane_metrics) == 1
    legacy_row = report.lane_metrics[0]
    assert legacy_row.lane_id == "legacy"
    assert legacy_row.legacy is True
    assert legacy_row.save_count == 1
    assert legacy_row.review_count == 0


def test_run_report_lane_metrics_empty_when_no_attempts(tmp_path: Path):
    """An empty run (no candidate attempts) returns ``[]`` on the wire,
    not ``None``. Legacy consumers see an empty list and skip rendering;
    new consumers iterate without a guard.
    """

    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    _seed_run(state_dir, brief_id="brief-empty")

    report = aggregate_run_report(
        state_dir,
        source="linkedin",
        state_key="key",
        run_id=1,
    )
    assert report is not None
    assert report.lane_metrics == []
