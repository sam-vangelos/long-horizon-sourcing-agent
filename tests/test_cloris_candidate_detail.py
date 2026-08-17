"""Tests for the Phase C, slice C1 candidate-detail aggregator
(`cloris.control_plane.aggregate_candidate_detail`).

Pins the contract of ``GET /api/candidate/{source}/{state_key}/{candidate_id}``:

- Missing state dir / missing DB / missing candidate id all collapse to
  ``None`` so the route returns a clean 404.
- Cross-source lookups (a github candidate accessed via a linkedin URL,
  or vice versa) collapse to ``None`` rather than leaking the foreign-source
  row.
- Save reason and confidence parse safely from
  ``candidates.terminal_payload_json`` — malformed / missing keys collapse
  to ``None`` rather than raising.
- Brief context (role title, linkedin project) is pulled from the most
  recent run that touched the candidate.
- Same read-only contract as ``aggregate_run_report``: no
  ``RuntimeStateStore`` import in the production module.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from cloris import control_plane
from cloris.control_plane import aggregate_candidate_detail
from shared.runtime_state.store import RuntimeStateStore


assert "RuntimeStateStore" not in inspect.getsource(control_plane), (
    "cloris/control_plane.py must not import or reference the canonical "
    "write-side store class in production paths."
)


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
) -> tuple[RuntimeStateStore, int]:
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    snapshot_json = (
        json.dumps(brief_snapshot) if brief_snapshot is not None else None
    )
    run_id = store.start_run(
        source=source,
        brief_id=brief_id,
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": brief_id},
        brief_snapshot_json=snapshot_json,
    )
    return store, run_id


def _save_candidate_through_lifecycle(
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
    """Walk a candidate to full_terminal and return its db row id.

    Mirrors the production write sequence the orchestrator uses; the
    lifecycle guard would reject any shortcut. The returned id is the
    URL-parameter for the candidate-detail endpoint.
    """

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

    db_path = state_dir_for(store)
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id FROM candidates WHERE source=? AND brief_id=? AND identity_key=?",
            (source, brief_id, identity_key),
        ).fetchone()
        assert row is not None, "candidate row should exist after seeding"
        return int(row["id"])
    finally:
        conn.close()


def state_dir_for(store: RuntimeStateStore) -> Path:
    """Recover the underlying DB path from a store. Helper kept here to
    avoid hard-coding the runtime_state.sqlite3 filename in every test."""

    return Path(store.db_path)


def test_returns_none_when_no_state_dirs_match_brief_id(tmp_path: Path) -> None:
    # No state_dirs under tmp_path → no matching brief → 404.
    detail = aggregate_candidate_detail(
        brief_id="brief-1", candidate_id=1, state_root=tmp_path
    )

    assert detail is None


def test_returns_none_when_candidate_id_does_not_exist(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    _seed_run(state_dir, brief_id="brief-1")

    detail = aggregate_candidate_detail(
        brief_id="brief-1", candidate_id=999, state_root=tmp_path
    )

    assert detail is None


def test_returns_none_when_brief_id_does_not_match_candidate_brief(
    tmp_path: Path,
) -> None:
    """Cross-brief safety: a candidate id from a different brief collapses to
    not-found. With brief-first routing, the URL only carries (brief_id,
    candidate_id) — the aggregator must verify the candidate's own brief_id
    matches the requested one before returning the detail."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(
        state_dir, source="linkedin", brief_id="brief-A"
    )
    candidate_id = _save_candidate_through_lifecycle(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-A",
        identity_key="li-1",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
        decision="SAVE",
    )

    # Request the candidate under a different brief id — must return None.
    detail = aggregate_candidate_detail(
        brief_id="brief-B", candidate_id=candidate_id, state_root=tmp_path
    )

    assert detail is None


def test_returns_full_detail_for_valid_candidate(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(
        state_dir,
        source="linkedin",
        brief_id="brief-1",
        brief_snapshot={
            "role_title": "Senior Forward Deployed Engineer",
            "linkedin_project": "FDE NYC",
        },
    )
    candidate_id = _save_candidate_through_lifecycle(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
        decision="SAVE",
        payload={"confidence": 0.91, "save_reason": "Strong systems-design background"},
    )

    detail = aggregate_candidate_detail(
        brief_id="brief-1", candidate_id=candidate_id, state_root=tmp_path
    )

    assert detail is not None
    assert detail.source == "linkedin"
    assert detail.brief_id == "brief-1"
    assert detail.candidate_id == candidate_id
    assert detail.identity_key == "li-1"
    assert detail.display_name == "Pat Doe"
    assert detail.profile_url == "https://linkedin.com/in/pat"
    assert detail.terminal_decision == "SAVE"
    assert detail.confidence == 0.91
    assert detail.save_reason == "Strong systems-design background"
    assert detail.source_run is not None
    assert detail.source_run.run_id == run_id
    assert detail.source_run.source == "linkedin"
    assert detail.source_run.state_key == "key"
    assert detail.brief_role_title == "Senior Forward Deployed Engineer"
    assert detail.brief_linkedin_project == "FDE NYC"


def test_save_reason_falls_back_to_legacy_reason_key(tmp_path: Path) -> None:
    """Legacy linkedin payloads stored the prose under ``reason``; the
    aggregator should accept either key with ``save_reason`` winning when
    both are present."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-1")
    candidate_id = _save_candidate_through_lifecycle(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
        decision="SAVE",
        payload={"reason": "Legacy reason field"},
    )

    detail = aggregate_candidate_detail(
        brief_id="brief-1", candidate_id=candidate_id, state_root=tmp_path
    )

    assert detail is not None
    assert detail.save_reason == "Legacy reason field"


def test_save_reason_reads_full_decision_rationale(tmp_path: Path) -> None:
    """Trial-walk wiring fix: the orchestrator writes the recruiter-facing
    judgment under ``terminal_payload['full_decision']['rationale']``, not
    at the top level. Across the trial-walk audit dataset (114/114 SAVE-
    class candidates on the flagship brief), every save carried
    substantive rationale at that path. The candidate-detail aggregator
    must surface it as ``save_reason`` on the wire so the candidate-detail
    surface stops rendering "No save reason recorded" on every save.

    Same shape on confidence: ``full_decision.confidence`` is the
    canonical write path."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-1")
    candidate_id = _save_candidate_through_lifecycle(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
        decision="SAVE",
        payload={
            "full_decision": {
                "decision": "SAVE",
                "rationale": (
                    "Solid enterprise GenAI builder at Mastercard with "
                    "production RAG/agentic workflow ownership."
                ),
                "confidence": 0.78,
            }
        },
    )

    detail = aggregate_candidate_detail(
        brief_id="brief-1", candidate_id=candidate_id, state_root=tmp_path
    )

    assert detail is not None
    assert detail.save_reason == (
        "Solid enterprise GenAI builder at Mastercard with "
        "production RAG/agentic workflow ownership."
    )
    assert detail.confidence == 0.78


def test_save_reason_full_decision_wins_over_top_level(tmp_path: Path) -> None:
    """When both the canonical ``full_decision.rationale`` AND a top-level
    ``save_reason`` / ``reason`` are present, the canonical (deeper)
    write wins. Top-level keys are legacy-fallback territory; the
    orchestrator's full-stage judgment is the authoritative signal."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-1")
    candidate_id = _save_candidate_through_lifecycle(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
        decision="SAVE",
        payload={
            "save_reason": "stale top-level summary",
            "confidence": 0.40,
            "full_decision": {
                "rationale": "fresh full-decision rationale",
                "confidence": 0.91,
            },
        },
    )

    detail = aggregate_candidate_detail(
        brief_id="brief-1", candidate_id=candidate_id, state_root=tmp_path
    )

    assert detail is not None
    assert detail.save_reason == "fresh full-decision rationale"
    assert detail.confidence == 0.91


def test_missing_payload_keys_collapse_to_null(tmp_path: Path) -> None:
    """A candidate with no save_reason / confidence should return null
    fields rather than raise."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-1")
    candidate_id = _save_candidate_through_lifecycle(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
        decision="REJECT",
        payload={},
    )

    detail = aggregate_candidate_detail(
        brief_id="brief-1", candidate_id=candidate_id, state_root=tmp_path
    )

    assert detail is not None
    assert detail.terminal_decision == "REJECT"
    assert detail.confidence is None
    assert detail.save_reason is None


def test_is_failed_state_set_when_lifecycle_in_failed_family(
    tmp_path: Path,
) -> None:
    """Phase C-bis 0.3: a candidate with current_lifecycle_state in the
    failed_* family surfaces ``is_failed_state=True`` so the
    candidate-detail page can disable pipeline-action affordances
    (status toggle, notes compose). The candidate is still queryable —
    only the workspace list filters them out."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-1")
    candidate_id = _save_candidate_through_lifecycle(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
        decision="SAVE",
    )
    # Force the lifecycle state directly via SQL (the store's transition
    # guard prevents full_terminal -> failed_terminal, but failed states
    # are reachable in production via mid-stage failure paths). The test
    # is verifying the read-side surface, not the transition rules.
    import sqlite3
    conn = sqlite3.connect(str(store.db_path))
    try:
        conn.execute(
            "UPDATE candidates SET current_lifecycle_state='failed_terminal' "
            "WHERE id=?",
            (candidate_id,),
        )
        conn.commit()
    finally:
        conn.close()

    detail = aggregate_candidate_detail(
        brief_id="brief-1", candidate_id=candidate_id, state_root=tmp_path
    )

    assert detail is not None
    assert detail.is_failed_state is True
    assert detail.current_lifecycle_state == "failed_terminal"


def test_is_failed_state_false_for_normal_lifecycle(tmp_path: Path) -> None:
    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id = _seed_run(state_dir, brief_id="brief-1")
    candidate_id = _save_candidate_through_lifecycle(
        store,
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
        decision="SAVE",
    )

    detail = aggregate_candidate_detail(
        brief_id="brief-1", candidate_id=candidate_id, state_root=tmp_path
    )

    assert detail is not None
    assert detail.is_failed_state is False


def test_source_run_uses_most_recent_attempt(tmp_path: Path) -> None:
    """When a candidate has attempts across multiple runs, the back-link
    points at the most recent (highest run_id) — the run the recruiter
    is most likely thinking about."""

    state_dir = _build_state_dir(tmp_path, "linkedin", "key")
    store, run_id_first = _seed_run(state_dir, brief_id="brief-1")
    candidate_id = _save_candidate_through_lifecycle(
        store,
        run_id=run_id_first,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
        decision="SAVE",
    )

    # Second run touches the same candidate (a re-judge / facial-rerun).
    run_id_second = store.start_run(
        source="linkedin",
        brief_id="brief-1",
        output_dir=str(state_dir),
        mode="resume",
        resume_state={"brief_name": "brief-1"},
    )
    rejudge_id = store.start_attempt(
        run_id=run_id_second,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        stage="full",
    )
    store.finish_attempt_success(
        attempt_id=rejudge_id,
        new_state="full_terminal",
        terminal_decision="SAVE",
        payload={"confidence": 0.95},
        run_id=run_id_second,
    )

    detail = aggregate_candidate_detail(
        brief_id="brief-1", candidate_id=candidate_id, state_root=tmp_path
    )

    assert detail is not None
    assert detail.source_run is not None
    assert detail.source_run.run_id == run_id_second
    assert run_id_second > run_id_first
