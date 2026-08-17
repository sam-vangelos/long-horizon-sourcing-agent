"""Behavioral tests for reopen Stage 2 (wire the recruiter primitive).

Covers the Part IX verification checklist of
``plans/reopen-stage2-hardened.md``:

- resolver swap + init guard (Part II)
- intention-then-signal sequence, incl. a simulated recruiter-store
  failure leaving a recoverable intention (Part IV)
- brief_polish returns priors SEPARATELY; v2_draft stays unpolluted
  (Part VI / Decision 3)
- GET /api/recruiter returns 200 with no brief context (Part VII)
- backfill idempotent: run twice = same state (Part VIII)

Each test isolates the global recruiter DB (and, where needed, the
per-state-dir state tree + config dir) to ``tmp_path`` and resets the
recruiter-id resolver so nothing leaks across tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.runtime_state.recruiter_store import (
    RecruiterIdMismatchError,
    RecruiterStore,
)
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def recruiter_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the global recruiter DB resolver to a tmp path.

    ``resolve_recruiter_db_path`` derives from the live ``RECRUITER_ROOT``
    module constant, so patch that (mirrors how identity/state tests patch
    their roots). Returns the resolved db path for direct assertions.
    """

    recruiter_root = tmp_path / "state" / "_recruiter"
    recruiter_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("shared.output_paths.RECRUITER_ROOT", recruiter_root)
    return recruiter_root / "recruiter.sqlite3"


@pytest.fixture(autouse=True)
def _reset_resolver() -> None:
    """Ensure every test starts + ends on the Stage-1 default resolver."""

    from shared.recruiter_context import reset_recruiter_id_resolver

    reset_recruiter_id_resolver()
    yield
    reset_recruiter_id_resolver()


# ---------------------------------------------------------------------------
# Part II — pluggable resolver + init guard
# ---------------------------------------------------------------------------


def test_default_resolver_bootstraps_stage1_recruiter(recruiter_db: Path) -> None:
    from shared.recruiter_context import get_current_recruiter_id

    # First resolve get-or-creates the Stage-1 recruiter (id 1 on a fresh
    # store) and is idempotent across calls.
    rid = get_current_recruiter_id()
    assert rid == 1
    assert get_current_recruiter_id() == 1

    store = RecruiterStore(recruiter_db)
    rec = store.get_recruiter(1)
    assert rec is not None
    assert rec["canonical_handle"] == "operator@example.com"


def test_set_recruiter_id_resolver_swaps_the_id(recruiter_db: Path) -> None:
    from shared.recruiter_context import (
        get_current_recruiter_id,
        set_recruiter_id_resolver,
    )

    set_recruiter_id_resolver(lambda: 7)
    assert get_current_recruiter_id() == 7


def test_init_guard_fires_on_recruiter_id_mismatch(recruiter_db: Path) -> None:
    # Seed a store with recruiter id 1.
    store = RecruiterStore(recruiter_db)
    assert store.upsert_recruiter("operator@example.com", display_name="Sam") == 1

    # Opening the SAME store while claiming a different acting principal
    # (id 2, absent from the table) must refuse with a migration pointer.
    with pytest.raises(RecruiterIdMismatchError) as exc:
        RecruiterStore(recruiter_db, expected_recruiter_id=2)
    assert "migrate_recruiter_id_stage1_to_phase2" in str(exc.value)
    assert exc.value.existing_ids == [1]


def test_init_guard_allows_matching_id(recruiter_db: Path) -> None:
    store = RecruiterStore(recruiter_db)
    assert store.upsert_recruiter("operator@example.com") == 1
    # Re-open claiming id 1 (the row that exists) — no refusal.
    reopened = RecruiterStore(recruiter_db, expected_recruiter_id=1)
    assert reopened.get_recruiter(1) is not None


def test_init_guard_no_check_when_expected_id_absent(recruiter_db: Path) -> None:
    # Default construction (no expected id) never runs the guard — this is
    # what keeps the Stage-1 callers + the self-bootstrapping resolver from
    # a chicken-and-egg recursion.
    store = RecruiterStore(recruiter_db)
    store.upsert_recruiter("a@b.com")  # id 1
    # A *different* id exists than a hypothetical resolver, but with no
    # expected_recruiter_id passed the open is unconditionally allowed.
    assert RecruiterStore(recruiter_db).get_recruiter(1) is not None


# ---------------------------------------------------------------------------
# Part IV — intention ledger + designer principle-feedback double-write
# ---------------------------------------------------------------------------


def _designer_candidate(state_dir: Path, brief_id: str, identity_key: str) -> RuntimeStateStore:
    """Create a per-state-dir store with one discovered designer candidate.

    The DB is rooted under ``<state_dir>/state/designer/<identity_key>/`` —
    the layout ``enumerate_state_dirs`` discovers (``<root>/<source>/<key>/
    runtime_state.sqlite3``) — so an intention left behind by a simulated
    recruiter-store failure is reachable by ``replay_write_intentions``
    pointed at ``<state_dir>/state``.
    """

    per_state_dir = state_dir / "state" / "designer" / identity_key
    per_state_dir.mkdir(parents=True, exist_ok=True)
    store = RuntimeStateStore(per_state_dir / "runtime_state.sqlite3")
    store.ensure_candidate(
        source="designer",
        brief_id=brief_id,
        identity_key=identity_key,
        display_name="Test Designer",
    )
    return store


def test_intention_ledger_records_and_lists_incomplete(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    iid = store.record_write_intention(
        signal_kind="principle_feedback",
        domain="designer",
        dedup_key="k1",
        payload={"principle_name": "Typography", "marker": "useful_guidance"},
        source_brief_id="brief-1",
    )
    incomplete = store.list_incomplete_write_intentions()
    assert [i["id"] for i in incomplete] == [iid]
    assert incomplete[0]["payload"]["principle_name"] == "Typography"

    store.mark_write_intention_complete(iid)
    assert store.list_incomplete_write_intentions() == []


def test_record_write_intention_is_idempotent_on_dedup_key(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    a = store.record_write_intention(
        signal_kind="principle_feedback", domain="designer", dedup_key="dup"
    )
    b = store.record_write_intention(
        signal_kind="principle_feedback", domain="designer", dedup_key="dup"
    )
    assert a == b  # one row, not two
    assert len(store.list_incomplete_write_intentions()) == 1


def test_principle_feedback_writes_intention_and_signal(
    tmp_path: Path, recruiter_db: Path
) -> None:
    from designer.recruiter_annotations import (
        PrincipleFeedbackStore,
        record_designer_principle_feedback,
    )

    brief_id = "brief-designer-1"
    identity_key = "designer-candidate-1"
    runtime_store = _designer_candidate(tmp_path, brief_id, identity_key)
    principle_store = PrincipleFeedbackStore(tmp_path / "annotations.sqlite3")

    record_designer_principle_feedback(
        runtime_state_store=runtime_store,
        principle_feedback_store=principle_store,
        source="designer",
        brief_id=brief_id,
        identity_key=identity_key,
        principle_name="Typographic refinement",
        marker="useful_guidance",
        note="Strong type hierarchy.",
    )

    # Intention was recorded AND marked complete (the recruiter write
    # succeeded through the self-bootstrapping resolver).
    assert runtime_store.list_incomplete_write_intentions() == []

    # The recruiter taste signal landed under the Stage-1 recruiter, in
    # the designer domain.
    recruiter_store = RecruiterStore(recruiter_db)
    signals = recruiter_store.active_taste_signals(1, domain="designer")
    assert len(signals) == 1
    assert signals[0]["signal_kind"] == "principle_feedback"
    assert signals[0]["payload"]["principle_name"] == "Typographic refinement"
    assert signals[0]["payload"]["judgment_accuracy"] == "useful"


def test_principle_feedback_failure_leaves_recoverable_intention(
    tmp_path: Path, recruiter_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recruiter-store failure must NOT lose the signal — the committed
    intention stays incomplete for the backfill to replay, and the
    recruiter's primary action (the per-state-dir write) still succeeds."""

    from designer.recruiter_annotations import (
        PrincipleFeedbackStore,
        record_designer_principle_feedback,
    )

    brief_id = "brief-designer-2"
    identity_key = "designer-candidate-2"
    runtime_store = _designer_candidate(tmp_path, brief_id, identity_key)
    principle_store = PrincipleFeedbackStore(tmp_path / "annotations.sqlite3")

    # Simulate the global recruiter store being unavailable mid-write.
    # Save the real method explicitly (rather than relying on
    # monkeypatch.undo(), which would also revert the recruiter_db
    # fixture's RECRUITER_ROOT patch — same function-scoped monkeypatch)
    # so the recovery step below can restore it surgically.
    real_record_taste_signal = RecruiterStore.record_taste_signal

    def _boom(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("recruiter store unavailable")

    monkeypatch.setattr(RecruiterStore, "record_taste_signal", _boom)

    # The call still succeeds (fail-soft) and returns the per-state-dir marker.
    marker = record_designer_principle_feedback(
        runtime_state_store=runtime_store,
        principle_feedback_store=principle_store,
        source="designer",
        brief_id=brief_id,
        identity_key=identity_key,
        principle_name="Color discipline",
        marker="off_rubric",
    )
    assert marker.principle_name == "Color discipline"

    # The canonical per-state-dir mirror still happened.
    candidate = runtime_store.get_candidate(
        source="designer", brief_id=brief_id, identity_key=identity_key
    )
    assert candidate["judgment_accuracy"] == "off_rubric"

    # The intention is left INCOMPLETE — recoverable by the backfill.
    incomplete = runtime_store.list_incomplete_write_intentions()
    assert len(incomplete) == 1
    assert incomplete[0]["domain"] == "designer"
    assert incomplete[0]["payload"]["marker"] == "off_rubric"

    # PROVE recovery end-to-end: the recruiter store comes back (restore
    # the real record_taste_signal), the backfill scans the per-state-dir
    # tree, finds THIS failure path's incomplete intention, and replays it
    # into the global recruiter store.
    from tools.backfill_recruiter_store import replay_write_intentions

    monkeypatch.setattr(
        RecruiterStore, "record_taste_signal", real_record_taste_signal
    )
    backfill = replay_write_intentions(
        state_root=tmp_path / "state", db_path=recruiter_db, recruiter_id=1
    )
    assert backfill.intentions_seen == 1
    assert backfill.intentions_replayed == 1

    # The signal now exists in the recruiter store, and the intention is
    # marked complete — the full failure → recover cycle closed.
    recruiter_store = RecruiterStore(recruiter_db)
    signals = recruiter_store.active_taste_signals(1, domain="designer")
    assert len(signals) == 1
    assert signals[0]["signal_kind"] == "principle_feedback"
    assert signals[0]["payload"]["marker"] == "off_rubric"
    assert runtime_store.list_incomplete_write_intentions() == []


# ---------------------------------------------------------------------------
# Part VI — brief_polish priors overlay SEPARATE from v2_draft
# ---------------------------------------------------------------------------


def test_brief_polish_returns_priors_separately(recruiter_db: Path) -> None:
    from market_intelligence.brief_polish import BriefPolishBackend

    # Seed an active taste signal for the designer domain.
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("operator@example.com")
    store.record_taste_signal(
        rid,
        signal_kind="principle_feedback",
        domain="designer",
        payload={"principle": "Typography", "delta": -1},
        source_brief_id="brief-x",
    )

    backend = BriefPolishBackend()
    # Single-module draft → domain resolves to "designer". No LLM access in
    # tests → heuristic path; that's fine, the overlay is attached
    # post-core regardless of which polish path ran.
    result = backend.polish(
        chapter_captures={
            "role": {"title": "Senior Product Designer"},
            "good_looks": {"prose": "Owns end-to-end product design with strong craft."},
            "where_to_look": {"target_modules": ["designer"]},
        },
        recruiter_id=rid,
    )

    # Priors are present, SEPARATE, and carry the active signal.
    assert result.recruiter_priors_overlay is not None
    assert result.recruiter_priors_overlay["domain"] == "designer"
    assert len(result.recruiter_priors_overlay["signals"]) == 1

    # v2_draft is UNPOLLUTED: no recruiter-priors key leaked in, and the
    # overlay is not serialized into the polish meta dict (which is what
    # intake persists alongside the brief).
    assert "recruiter_priors_overlay" not in result.v2_draft
    assert "recruiter_priors" not in result.v2_draft
    assert "recruiter_priors_overlay" not in result.to_meta_dict()


def test_brief_polish_no_overlay_when_no_recruiter_id(recruiter_db: Path) -> None:
    from market_intelligence.brief_polish import BriefPolishBackend

    backend = BriefPolishBackend()
    result = backend.polish(
        chapter_captures={
            "role": {"title": "Senior Product Designer"},
            "good_looks": {"prose": "Owns end-to-end product design."},
            "where_to_look": {"target_modules": ["designer"]},
        },
    )
    # No recruiter_id passed → no overlay attached (every existing caller).
    assert result.recruiter_priors_overlay is None


def test_brief_polish_no_overlay_when_multiple_modules(recruiter_db: Path) -> None:
    from market_intelligence.brief_polish import BriefPolishBackend

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("operator@example.com")
    store.record_taste_signal(
        rid, signal_kind="principle_feedback", domain="designer"
    )

    backend = BriefPolishBackend()
    # Two modules → no single domain → no overlay (rather than mixing
    # domains / writing an unfilterable read).
    result = backend.polish(
        chapter_captures={
            "role": {"title": "Hybrid role"},
            "good_looks": {"prose": "Designs and ships."},
            "where_to_look": {"target_modules": ["designer", "linkedin"]},
        },
        recruiter_id=rid,
    )
    assert result.recruiter_priors_overlay is None


# ---------------------------------------------------------------------------
# Part VII — recruiter-independent query endpoint
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    from cloris import api as cloris_api
    from cloris.app import create_app

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)

    output_dir = tmp_path / "output"
    (output_dir / "state" / "_recruiter").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("shared.config.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("shared.output_paths.OUTPUT_ROOT", output_dir)
    monkeypatch.setattr("shared.output_paths.STATE_ROOT", output_dir / "state")
    monkeypatch.setattr(
        "shared.output_paths.RECRUITER_ROOT", output_dir / "state" / "_recruiter"
    )
    return TestClient(create_app())


def test_get_api_recruiter_200_no_brief_context(client) -> None:
    # No brief created, no brief_id in the request — the recruiter is a
    # first-class entity, queryable on its own.
    resp = client.get("/api/recruiter")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["recruiter_id"] == 1
    assert body["canonical_handle"] == "operator@example.com"
    assert body["briefs_count"] == 0
    assert body["active_signals"] == []


def test_get_api_recruiter_reflects_signals_and_briefs(
    client, tmp_path: Path
) -> None:
    from shared.output_paths import resolve_recruiter_db_path

    store = RecruiterStore(resolve_recruiter_db_path())
    rid = store.upsert_recruiter("operator@example.com")
    store.link_brief(rid, "brief-a")
    store.link_brief(rid, "brief-b")
    store.record_taste_signal(
        rid, signal_kind="archetype_preference", domain="designer"
    )

    resp = client.get("/api/recruiter")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["briefs_count"] == 2
    assert len(body["active_signals"]) == 1
    assert body["active_signals"][0]["signal_kind"] == "archetype_preference"


def test_recruiter_dashboard_is_no_longer_a_501_placeholder(client) -> None:
    # Reopen Stage 3 (R3.0/R3.1) filled this endpoint — it was a 501
    # placeholder under Stage 2. It now resolves the recruiter on the PATH
    # recruiter_id (a deliberate deviation from GET /api/recruiter, which
    # uses the acting-principal seam — see api_recruiter_dashboard docstring).
    # This fixture's recruiter DB is a fresh tmp DB with no recruiter
    # provisioned (the get-or-create only fires when the resolver SEAM is
    # called; a path-param get_recruiter lookup does not create), so id 1 is
    # absent and the endpoint returns a clean 404 — the honest "not found",
    # not the old 501 "not implemented".
    resp = client.get("/api/recruiter/1/dashboard")
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"]["error"] == "recruiter_not_found"


def test_recruiter_dashboard_returns_triad_when_recruiter_exists(
    client, recruiter_db: Path
) -> None:
    # With the recruiter provisioned, the dashboard returns the persistence
    # triad. PRESENCE comes from the recruiter_candidate_history accretion
    # log (count + briefs, NO verdict per D-B); calibration_drift and
    # reflection_trail are present as panels (R3.2/R3.3), empty here since
    # this recruiter has no markers/reflections in the fixture.
    store = RecruiterStore(recruiter_db)
    store.upsert_recruiter(canonical_handle="sam", display_name="Sam")

    resp = client.get("/api/recruiter/1/dashboard")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["slice"] == "v0-recruiter-slice-1"
    assert body["recruiter_id"] == 1
    assert body["presence"] == []
    assert body["calibration_drift"] == []
    assert body["reflection_trail"] == []


# ---------------------------------------------------------------------------
# Part VIII — idempotent backfill (run twice == same state)
# ---------------------------------------------------------------------------


def _write_brief(path: Path, role_title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "role_title": role_title,
                "capability_areas": [{"name": "A", "description": "x"}],
                "depth_distinction": {
                    "builder_definition": "",
                    "user_definition": "",
                    "edge_case_guidance": "",
                },
                "non_fit_patterns": [],
                "target_modules": ["linkedin"],
            }
        )
    )


def test_backfill_briefs_idempotent(
    tmp_path: Path, recruiter_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cloris import api as cloris_api
    from tools.backfill_recruiter_store import backfill_recruiter_briefs

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)
    _write_brief(config_dir / "role-one" / "brief.json", "Role One")
    _write_brief(config_dir / "role-two" / "brief.json", "Role Two")

    store = RecruiterStore(recruiter_db)
    store.upsert_recruiter("operator@example.com")  # id 1

    first = backfill_recruiter_briefs(
        config_dir=config_dir, db_path=recruiter_db, recruiter_id=1
    )
    assert first.briefs_seen == 2
    assert first.briefs_linked == 2

    briefs_after_first = sorted(store.briefs_for_recruiter(1))
    assert len(briefs_after_first) == 2

    # Second run: same input → no new links, identical end state.
    second = backfill_recruiter_briefs(
        config_dir=config_dir, db_path=recruiter_db, recruiter_id=1
    )
    assert second.briefs_seen == 2
    assert second.briefs_linked == 0
    assert sorted(store.briefs_for_recruiter(1)) == briefs_after_first


def test_backfill_intentions_replay_idempotent(
    tmp_path: Path, recruiter_db: Path
) -> None:
    from tools.backfill_recruiter_store import replay_write_intentions

    # Build a per-state-dir DB under state/designer/<key> with one
    # incomplete intention (the shape a transient recruiter-store failure
    # would leave behind).
    state_root = tmp_path / "state"
    state_dir = state_root / "designer" / "brief-key"
    state_dir.mkdir(parents=True, exist_ok=True)
    per_state = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    per_state.record_write_intention(
        signal_kind="principle_feedback",
        domain="designer",
        dedup_key="designer:principle_feedback:brief-key:cand:Typography:useful_guidance:t0",
        payload={"principle_name": "Typography", "marker": "useful_guidance"},
        source_brief_id="brief-key",
    )

    store = RecruiterStore(recruiter_db)
    store.upsert_recruiter("operator@example.com")  # id 1

    first = replay_write_intentions(
        state_root=state_root, db_path=recruiter_db, recruiter_id=1
    )
    assert first.intentions_seen == 1
    assert first.intentions_replayed == 1

    signals_after_first = store.active_taste_signals(1, domain="designer")
    assert len(signals_after_first) == 1
    assert per_state.list_incomplete_write_intentions() == []  # marked complete

    # Second run: the intention is already complete → nothing re-read, no
    # duplicate signal written.
    second = replay_write_intentions(
        state_root=state_root, db_path=recruiter_db, recruiter_id=1
    )
    assert second.intentions_seen == 0
    assert second.intentions_replayed == 0
    assert len(store.active_taste_signals(1, domain="designer")) == 1


def test_backfill_intentions_replay_dedups_partial_failure(
    tmp_path: Path, recruiter_db: Path
) -> None:
    """If a prior run wrote the signal but crashed before marking the
    intention complete, the replay must mark-complete WITHOUT writing a
    duplicate signal (dedup by signal_kind/domain/payload)."""

    from tools.backfill_recruiter_store import replay_write_intentions

    state_root = tmp_path / "state"
    state_dir = state_root / "designer" / "brief-key"
    state_dir.mkdir(parents=True, exist_ok=True)
    per_state = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    payload = {"principle_name": "Spacing", "marker": "useful_guidance"}
    per_state.record_write_intention(
        signal_kind="principle_feedback",
        domain="designer",
        dedup_key="k-partial",
        payload=payload,
        source_brief_id="brief-key",
    )

    store = RecruiterStore(recruiter_db)
    store.upsert_recruiter("operator@example.com")  # id 1
    # Pre-existing signal identical to what the replay would write (the
    # "wrote but didn't mark" crash state). The backfill fingerprints on
    # (signal_kind, domain, payload), so the stored payload must match the
    # intention's payload for the dedup-skip to fire.
    store.record_taste_signal(
        1,
        signal_kind="principle_feedback",
        domain="designer",
        payload=payload,
        source_brief_id="brief-key",
    )

    result = replay_write_intentions(
        state_root=state_root, db_path=recruiter_db, recruiter_id=1
    )
    assert result.intentions_seen == 1
    assert result.intentions_replayed == 0
    assert result.intentions_skipped_duplicate == 1
    # Still exactly one signal — no duplicate.
    assert len(store.active_taste_signals(1, domain="designer")) == 1
    assert per_state.list_incomplete_write_intentions() == []
