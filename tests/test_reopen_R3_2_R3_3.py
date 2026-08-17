"""Reopen Stage 3 R3.2 (calibration drift) + R3.3 (reflection trail).

Verification for the two forward panels the dashboard handler
(``cloris.api._monolith.api_recruiter_dashboard``) fills:

- R3.2: ``calibration_drift`` MERGES the ``judgment_accuracy`` rollup
  across every state dir a brief spans (linkedin + github), so a
  multi-source brief is not under-read by a single-db_path read.
- R3.3: ``reflection_trail`` reads ACTIVE reflection sessions from the
  SINGLE intake DB (read-only — no RuntimeStateStore, no DDL on the GET),
  fanned by brief_id. Per-state-dir ``runtime_state.sqlite3`` files carry
  an always-empty reflection table; a per-state-dir read would ship a
  dead panel, so this test pins the intake-DB target with a negative
  per-state-dir check.

Seed idiom mirrors
``tests/test_reflection_calibration_integration.py::_seed_marker`` and the
``client`` fixture in ``tests/test_reopen_stage2.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _make_store(state_dir: Path):
    from shared.runtime_state.store import RuntimeStateStore

    state_dir.mkdir(parents=True, exist_ok=True)
    return RuntimeStateStore(state_dir / "runtime_state.sqlite3")


def _seed_marker(
    store,
    *,
    source: str,
    brief_id: str,
    identity_key: str,
    capability_area: str | None,
    confidence: float | None,
    terminal_decision: str,
    judgment_accuracy: str,
) -> None:
    """Walk a candidate discovered → full_terminal and stamp a marker.

    Inlined copy of the integration-test fixture so this verification is
    self-contained.
    """

    runs = store.list_runs(source=source, brief_id=brief_id)
    if runs:
        run_id = int(runs[0]["id"])
    else:
        run_id = store.start_run(
            source=source,
            brief_id=brief_id,
            output_dir=str(store.db_path.parent),
            mode="fresh",
            resume_state={"brief_name": brief_id},
        )

    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=f"Candidate {identity_key}",
        profile_url=f"https://example.test/{identity_key}",
    )
    for new_state in (
        "snippet_extracted",
        "facial_started",
        "facial_terminal",
        "full_started",
    ):
        store.set_candidate_state(
            run_id=run_id,
            source=source,
            brief_id=brief_id,
            identity_key=identity_key,
            new_state=new_state,
        )

    full_decision: dict[str, Any] = {
        "decision": terminal_decision,
        "rationale": f"rationale for {identity_key}",
    }
    if confidence is not None:
        full_decision["confidence"] = confidence
    if capability_area is not None:
        full_decision["capability_area"] = capability_area

    store.set_candidate_state(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="full_terminal",
        terminal_decision=terminal_decision,
        terminal_payload={"full_decision": full_decision},
    )

    candidate = store.get_candidate(
        source=source, brief_id=brief_id, identity_key=identity_key
    )
    assert candidate is not None
    store.set_candidate_judgment_accuracy(int(candidate["id"]), judgment_accuracy)


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point every output root at tmp + return the TestClient + roots.

    Mirrors ``tests/test_reopen_stage2.py::client`` and additionally
    patches ``INTAKE_ROOT`` (R3.3 needs the intake DB seedable in tmp).
    """

    from fastapi.testclient import TestClient

    from cloris import api as cloris_api
    from cloris.app import create_app

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)

    output_dir = tmp_path / "output"
    state_root = output_dir / "state"
    (state_root / "_recruiter").mkdir(parents=True, exist_ok=True)
    intake_root = output_dir / "intake"

    monkeypatch.setattr("shared.config.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("shared.output_paths.OUTPUT_ROOT", output_dir)
    monkeypatch.setattr("shared.output_paths.STATE_ROOT", state_root)
    monkeypatch.setattr(
        "shared.output_paths.RECRUITER_ROOT", state_root / "_recruiter"
    )
    monkeypatch.setattr("shared.output_paths.INTAKE_ROOT", intake_root)

    client = TestClient(create_app())
    return {
        "client": client,
        "output": output_dir,
        "state_root": state_root,
        "intake_root": intake_root,
    }


def _link_recruiter_with_briefs(brief_ids: list[str]) -> int:
    from shared.output_paths import resolve_recruiter_db_path
    from shared.runtime_state.recruiter_store import RecruiterStore

    store = RecruiterStore(resolve_recruiter_db_path())
    rid = store.upsert_recruiter("operator@example.com")  # id 1
    for brief_id in brief_ids:
        store.link_brief(rid, brief_id)
    return rid


# ---------------------------------------------------------------------------
# R3.2 — calibration merge across state dirs
# ---------------------------------------------------------------------------


def test_R3_2_calibration_merges_across_two_state_dirs(env) -> None:
    """A brief living in linkedin + github has its marker rollup MERGED.

    Seeds 3 ``wrong`` markers in ``Distributed Systems`` on the linkedin
    state dir + 3 ``off_rubric`` on the github state dir, SAME brief_id.
    ``calibration_drift`` for that brief must show ``total_markers == 6``
    (the SUM), proving the merge — a single-db_path read would show 3.
    A second linked brief with NO state dir contributes zero (fail-soft).
    """

    client = env["client"]
    state_root = env["state_root"]

    b1 = "brief-multi-source"
    b2 = "brief-no-runs"  # linked but never run → no state dir → zero
    _link_recruiter_with_briefs([b1, b2])

    li_store = _make_store(state_root / "linkedin" / "state-li")
    gh_store = _make_store(state_root / "github" / "state-gh")
    for ident in ("li-1", "li-2", "li-3"):
        _seed_marker(
            li_store,
            source="linkedin",
            brief_id=b1,
            identity_key=ident,
            capability_area="Distributed Systems",
            confidence=0.6,
            terminal_decision="REJECT",
            judgment_accuracy="wrong",
        )
    for ident in ("gh-1", "gh-2", "gh-3"):
        _seed_marker(
            gh_store,
            source="github",
            brief_id=b1,
            identity_key=ident,
            capability_area="Distributed Systems",
            confidence=0.85,
            terminal_decision="SAVE",
            judgment_accuracy="off_rubric",
        )

    resp = client.get("/api/recruiter/1/dashboard")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    drift_by_brief = {e["brief_id"]: e for e in body["calibration_drift"]}

    # Both linked briefs appear (one entry per brief the recruiter owns).
    assert set(drift_by_brief) == {b1, b2}

    merged = drift_by_brief[b1]
    # THE MERGE ASSERTION: 3 (linkedin) + 3 (github) = 6, not 3.
    assert merged["total_markers"] == 6, merged
    assert merged["source_state_dirs"] == 2
    assert merged["by_marker_value"] == {"wrong": 3, "off_rubric": 3}
    # Same capability area on both sources → counts merge under one key.
    assert merged["by_capability_area"] == {"Distributed Systems": 6}
    # drift = miscalibrated share = (3 wrong + 3 off_rubric) / 6 = 1.0
    assert merged["drift"] == 1.0

    # Fail-soft: a brief with no state dir contributes zero, no 500.
    empty = drift_by_brief[b2]
    assert empty["total_markers"] == 0
    assert empty["source_state_dirs"] == 0
    assert empty["by_marker_value"] == {}
    assert empty["drift"] == 0.0


def test_R3_2_single_db_path_read_would_have_under_read(env) -> None:
    """Direct proof the merge is load-bearing: a single-source read of
    one state dir yields 3, while the handler's merged panel yields 6.

    Guards against a regression to ``aggregate_calibration_markers(one_db)``
    — if someone replaced the fan with a single read, this asserts the
    gap the merge closes.
    """

    from shared.runtime_state.calibration import aggregate_calibration_markers

    client = env["client"]
    state_root = env["state_root"]

    b1 = "brief-multi-source"
    _link_recruiter_with_briefs([b1])

    li_dir = state_root / "linkedin" / "state-li"
    gh_dir = state_root / "github" / "state-gh"
    li_store = _make_store(li_dir)
    gh_store = _make_store(gh_dir)
    for ident in ("li-1", "li-2", "li-3"):
        _seed_marker(
            li_store, source="linkedin", brief_id=b1, identity_key=ident,
            capability_area="Distributed Systems", confidence=0.6,
            terminal_decision="REJECT", judgment_accuracy="wrong",
        )
    for ident in ("gh-1", "gh-2", "gh-3"):
        _seed_marker(
            gh_store, source="github", brief_id=b1, identity_key=ident,
            capability_area="Distributed Systems", confidence=0.85,
            terminal_decision="SAVE", judgment_accuracy="off_rubric",
        )

    # A single-db_path read (the BUG shape) sees only one source.
    single = aggregate_calibration_markers(
        li_dir / "runtime_state.sqlite3", brief_id=b1
    )
    assert single.total_markers == 3  # under-reads the multi-source brief

    # The handler's merged panel sees both.
    body = client.get("/api/recruiter/1/dashboard").json()
    merged = {e["brief_id"]: e for e in body["calibration_drift"]}[b1]
    assert merged["total_markers"] == 6
    assert merged["total_markers"] > single.total_markers


# ---------------------------------------------------------------------------
# R3.3 — reflection trail reads the SINGLE intake DB, read-only
# ---------------------------------------------------------------------------


def test_R3_3_reflection_trail_reads_intake_db(env) -> None:
    """An active reflection session created against the intake DB (the
    same store ``_reflection_store_factory`` writes through) surfaces in
    ``reflection_trail``. A brief with no session is absent.

    NEGATIVE CHECK: a per-state-dir ``runtime_state.sqlite3`` carries ZERO
    reflection rows — so a future per-state-dir read would be caught here.
    """

    from cloris.api._monolith import _reflection_store_factory
    from shared.runtime_state.reflection import create_reflection_session

    client = env["client"]
    state_root = env["state_root"]

    b_active = "brief-reflecting"
    b_idle = "brief-idle"
    _link_recruiter_with_briefs([b_active, b_idle])

    # Give b_active a state dir too (proves reflection does NOT come from
    # the per-state-dir DB even when one exists for the brief).
    _make_store(state_root / "linkedin" / "state-reflect")

    # Create the active reflection THROUGH the same factory the live
    # endpoint uses — i.e. against the intake DB, not a state dir.
    store = _reflection_store_factory()
    session = create_reflection_session(store, brief_id=b_active)
    assert session["current_phase"] == "planning"

    resp = client.get("/api/recruiter/1/dashboard")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    trail_by_brief = {e["brief_id"]: e for e in body["reflection_trail"]}
    # Only the brief with an active session appears.
    assert set(trail_by_brief) == {b_active}
    entry = trail_by_brief[b_active]
    assert entry["current_phase"] == "planning"
    assert entry["reflection_id"] == int(session["id"])
    assert entry["started_at"]  # non-empty timestamp from the real row

    # NEGATIVE: the per-state-dir runtime_state.sqlite3 has ZERO reflection
    # rows. If R3.3 ever read per-state-dir, the panel would be dead.
    import sqlite3

    per_state_db = state_root / "linkedin" / "state-reflect" / "runtime_state.sqlite3"
    assert per_state_db.exists()
    conn = sqlite3.connect(f"file:{per_state_db}?mode=ro", uri=True)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM reflection_sessions"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 0, (
        "per-state-dir reflection_sessions must be empty; reflection rows "
        "live in the intake DB"
    )


def test_R3_3_completed_and_discarded_sessions_are_excluded(env) -> None:
    """Only sessions with ``completed_at IS NULL AND discarded_at IS NULL``
    surface. A committed session and a discarded session are excluded.
    """

    from cloris.api._monolith import _reflection_store_factory
    from shared.runtime_state.reflection import (
        commit_reflection,
        create_reflection_session,
        discard_reflection,
    )

    client = env["client"]
    b_active = "brief-active"
    b_committed = "brief-committed"
    b_discarded = "brief-discarded"
    _link_recruiter_with_briefs([b_active, b_committed, b_discarded])

    store = _reflection_store_factory()
    create_reflection_session(store, brief_id=b_active)
    committed = create_reflection_session(store, brief_id=b_committed)
    commit_reflection(
        store, session_id=int(committed["id"]), brief_version_path="v2.json"
    )
    discarded = create_reflection_session(store, brief_id=b_discarded)
    discard_reflection(store, session_id=int(discarded["id"]))

    body = client.get("/api/recruiter/1/dashboard").json()
    trail = {e["brief_id"] for e in body["reflection_trail"]}
    assert trail == {b_active}


def test_R3_3_get_is_read_only_creates_no_intake_db(env) -> None:
    """The dashboard GET creates NO intake DB and runs NO DDL.

    With briefs linked but no reflection ever written, the intake DB file
    must not exist before OR after the GET — proving the handler opens it
    ``mode=ro`` and never constructs a RuntimeStateStore (which would
    mkdir + initialize() the file).
    """

    from shared.output_paths import resolve_intake_db_path

    client = env["client"]
    _link_recruiter_with_briefs(["brief-x", "brief-y"])

    intake_db = resolve_intake_db_path()
    assert not intake_db.exists(), "precondition: no intake DB yet"

    resp = client.get("/api/recruiter/1/dashboard")
    assert resp.status_code == 200, resp.text
    assert resp.json()["reflection_trail"] == []

    # The READ must not have created the DB file (no write-on-read DDL).
    assert not intake_db.exists(), (
        "dashboard GET created the intake DB — it must read-only, never "
        "construct a RuntimeStateStore"
    )


# ---------------------------------------------------------------------------
# PRESENCE panel unchanged by R3.2 / R3.3
# ---------------------------------------------------------------------------


def test_presence_panel_unchanged_alongside_new_panels(env) -> None:
    """R3.2/R3.3 leave the PRESENCE panel byte-identical.

    Seeds a candidate sighting on the recruiter spine, then asserts the
    presence row is shaped exactly as R3.1 emitted it (no verdict /
    terminal_decision leak) AND coexists with a populated calibration +
    reflection panel.
    """

    from cloris.api._monolith import _reflection_store_factory
    from shared.output_paths import resolve_recruiter_db_path
    from shared.runtime_state.recruiter_store import RecruiterStore
    from shared.runtime_state.reflection import create_reflection_session

    client = env["client"]
    state_root = env["state_root"]

    b1 = "brief-presence"
    rid = _link_recruiter_with_briefs([b1])

    # A sighting so the PRESENCE accretion log has a row.
    rstore = RecruiterStore(resolve_recruiter_db_path())
    rstore.record_candidate_sighting(
        recruiter_id=rid,
        person_id=4242,
        brief_id=b1,
        lifecycle_state="full_terminal",
    )

    # Populate the two new panels.
    li_store = _make_store(state_root / "linkedin" / "state-p")
    for ident in ("p1", "p2", "p3", "p4", "p5"):
        _seed_marker(
            li_store, source="linkedin", brief_id=b1, identity_key=ident,
            capability_area="ML", confidence=0.6,
            terminal_decision="REJECT", judgment_accuracy="wrong",
        )
    create_reflection_session(_reflection_store_factory(), brief_id=b1)

    body = client.get("/api/recruiter/1/dashboard").json()

    # PRESENCE: exactly the R3.1 shape — five keys, no verdict.
    assert len(body["presence"]) == 1
    row = body["presence"][0]
    assert set(row.keys()) == {
        "person_id",
        "times_surfaced",
        "first_seen_brief",
        "last_seen_brief",
        "last_lifecycle_state",
    }
    assert row["person_id"] == 4242
    assert row["times_surfaced"] == 1
    assert row["first_seen_brief"] == b1
    assert row["last_seen_brief"] == b1
    assert row["last_lifecycle_state"] == "full_terminal"
    assert "terminal_decision" not in row
    assert "verdict" not in row

    # The new panels are populated, confirming presence is unaffected by
    # their presence.
    assert any(e["brief_id"] == b1 for e in body["calibration_drift"])
    assert any(e["brief_id"] == b1 for e in body["reflection_trail"])
