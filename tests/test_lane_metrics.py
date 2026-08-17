"""P5 — lane-attributed metrics read model regression tests.

Pins the source-of-truth invariant (canonical SQLite, never JSON/JSONL
projections), the legacy fallback bucket, the REVIEW-vs-SAVE separation
(including review-reason breakdown), opens/evaluated semantics, the
attribution precedence chain on the helper, and the missing-DB
fallback.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from shared.runtime_state import lane_metrics as lm
from shared.runtime_state.lane_metrics import (
    LEGACY_LANE_ID,
    UNSPECIFIED_REVIEW_REASON,
    LaneMetricsRow,
    candidate_lane_attribution,
    lane_metrics_for_run,
)
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Fixture helpers — small canonical-state seeding via the writer store.
# ---------------------------------------------------------------------------


def _start_run(store: RuntimeStateStore, *, brief_id: str = "brief-test") -> int:
    """Seed a run row and return its id. Mirrors test_linkedin_runtime_state.py."""

    return store.start_run(
        source="linkedin",
        brief_id=brief_id,
        output_dir=str(store.db_path.parent),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )


def _upsert_lane_work_unit(
    store: RuntimeStateStore,
    *,
    run_id: int,
    brief_id: str,
    source_unit_id: str,
    lane_id: str,
    lane_name: str = "",
    lane_intent: str = "",
    acquisition_mode: str = "",
    metrics: dict | None = None,
    counters: dict | None = None,
) -> int:
    """Upsert a work-unit row carrying lane attribution in payload_json.

    Mirrors the production write path at
    ``shared.runtime_state.linkedin_progress_sync.sync_linkedin_progress``
    which serializes ``SearchString.to_dict()`` into payload_json.
    """

    payload = {
        "id": int(source_unit_id),
        "name": f"lane:{lane_id}",
        "boolean": "ml",
        "lane_id": lane_id,
        "lane_name": lane_name,
        "lane_intent": lane_intent,
        "acquisition_mode": acquisition_mode,
    }
    return store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        kind="linkedin_string",
        source_unit_id=source_unit_id,
        display_name=f"lane:{lane_id}",
        ordering_index=int(source_unit_id),
        status="done",
        payload=payload,
        checkpoint={"pages_reviewed": 1},
        metrics=metrics or {},
        family_key="",
        novelty_bucket="",
        domain_lane="",
        counters=counters or {},
        notes="",
    )


def _seed_candidate_terminal(
    store: RuntimeStateStore,
    *,
    run_id: int,
    brief_id: str,
    identity_key: str,
    work_unit_id: int,
    terminal_decision: str,
    terminal_payload: dict,
    full_attempt_status: str = "succeeded",
    full_attempt_payload: dict | None = None,
) -> int:
    """Drive a candidate through facial+full attempts to a terminal state.

    Returns the candidate id. The orchestrator transitions
    ``discovered -> snippet_extracted -> facial_started -> facial_terminal
    -> full_started -> full_terminal``; we replay that via the writer
    primitives so the test exercises real SQLite shape.
    """

    candidate_id = store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=identity_key,
        profile_url=identity_key,
    )
    # Walk lifecycle transitions in lockstep with the orchestrator:
    # discovered -> snippet_extracted -> facial_started ->
    # facial_terminal -> full_started -> full_terminal. The store's
    # ``_guard_transition`` enforces each step, so the test exercises
    # the same shape as the production write path.
    for state in ("snippet_extracted", "facial_started"):
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id=brief_id,
            identity_key=identity_key,
            new_state=state,
            last_work_unit_id=work_unit_id,
        )
    facial_attempt = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key=identity_key,
        stage="facial",
        work_unit_id=work_unit_id,
    )
    store.finish_attempt_success(
        attempt_id=facial_attempt,
        new_state="facial_terminal",
        terminal_decision=None,
        payload={"facial_decision": {"decision": "FACIAL_YES"}},
        run_id=run_id,
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="full_started",
        last_work_unit_id=work_unit_id,
    )
    full_attempt = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key=identity_key,
        stage="full",
        work_unit_id=work_unit_id,
    )
    if full_attempt_status == "succeeded":
        store.finish_attempt_success(
            attempt_id=full_attempt,
            new_state="full_terminal",
            terminal_decision=terminal_decision,
            payload=full_attempt_payload or {},
            terminal_payload=terminal_payload,
            run_id=run_id,
        )
    # else: leave the attempt in 'started' so opens > evaluated by one.
    return candidate_id


# ---------------------------------------------------------------------------
# candidate_lane_attribution — pure helper precedence chain.
# ---------------------------------------------------------------------------


def test_candidate_lane_attribution_prefers_candidate_payload():
    result = candidate_lane_attribution(
        {"lane": {"lane_id": "from_candidate"}},
        {"lane_id": "from_work_unit"},
    )
    assert result == "from_candidate"


def test_candidate_lane_attribution_falls_back_to_work_unit():
    result = candidate_lane_attribution(None, {"lane_id": "from_work_unit"})
    assert result == "from_work_unit"


def test_candidate_lane_attribution_falls_back_to_legacy():
    assert candidate_lane_attribution(None, None) == LEGACY_LANE_ID
    assert candidate_lane_attribution({}, {}) == LEGACY_LANE_ID
    assert (
        candidate_lane_attribution(
            {"lane": {"lane_id": ""}}, {"lane_id": "   "}
        )
        == LEGACY_LANE_ID
    )


def test_candidate_lane_attribution_ignores_malformed_lane_block():
    """A non-dict ``lane`` field on terminal_payload must not crash; the
    helper falls through to the work-unit payload.
    """

    assert (
        candidate_lane_attribution(
            {"lane": "not a dict"}, {"lane_id": "from_work_unit"}
        )
        == "from_work_unit"
    )


# ---------------------------------------------------------------------------
# Missing / corrupt DB fallbacks.
# ---------------------------------------------------------------------------


def test_lane_metrics_for_run_missing_db_returns_empty(tmp_path: Path):
    assert lane_metrics_for_run(tmp_path / "missing.sqlite3", run_id=1) == tuple()


def test_lane_metrics_for_run_corrupt_db_returns_empty(tmp_path: Path):
    db_path = tmp_path / "runtime_state.sqlite3"
    db_path.write_text("not a sqlite file")
    assert lane_metrics_for_run(db_path, run_id=1) == tuple()


def test_lane_metrics_for_run_empty_run_returns_empty(tmp_path: Path):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = _start_run(store)
    # No work units / candidates / attempts seeded.
    assert lane_metrics_for_run(store.db_path, run_id=run_id) == tuple()


# ---------------------------------------------------------------------------
# Multi-lane aggregation correctness.
# ---------------------------------------------------------------------------


def test_lane_metrics_aggregates_two_lanes_plus_legacy(tmp_path: Path):
    """A run with lane A (save), lane B (review), and a legacy work unit
    (no lane_id) returns three rows with the legacy bucket last.
    """

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    brief_id = "brief-multi"
    run_id = _start_run(store, brief_id=brief_id)

    wu_a = _upsert_lane_work_unit(
        store,
        run_id=run_id,
        brief_id=brief_id,
        source_unit_id="11",
        lane_id="lane_a",
        lane_name="Lane Alpha",
        lane_intent="builders",
        acquisition_mode="linkedin_boolean",
        counters={
            "result_count": 100,
            "candidates_discovered": 5,
            "facial_yes_count": 2,
            "facial_no_count": 3,
            "saves_count": 1,
            "rejected_count": 0,
        },
    )
    wu_b = _upsert_lane_work_unit(
        store,
        run_id=run_id,
        brief_id=brief_id,
        source_unit_id="22",
        lane_id="lane_b",
        lane_name="Lane Beta",
        counters={
            "result_count": 50,
            "candidates_discovered": 3,
            "facial_yes_count": 1,
            "facial_no_count": 2,
            "saves_count": 0,
            "rejected_count": 0,
        },
    )
    # Legacy work unit: no lane_id in payload.
    wu_legacy = store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        kind="linkedin_string",
        source_unit_id="99",
        display_name="legacy:99",
        ordering_index=99,
        status="done",
        payload={"id": 99, "name": "legacy", "boolean": "ml"},
        checkpoint={},
        metrics={},
        counters={"result_count": 25, "candidates_discovered": 1},
    )

    _seed_candidate_terminal(
        store,
        run_id=run_id,
        brief_id=brief_id,
        identity_key="cand-a-save",
        work_unit_id=wu_a,
        terminal_decision="SAVE",
        terminal_payload={"full_decision": {"decision": "SAVE"}},
    )
    _seed_candidate_terminal(
        store,
        run_id=run_id,
        brief_id=brief_id,
        identity_key="cand-b-review",
        work_unit_id=wu_b,
        terminal_decision="REVIEW_INFERRED",
        terminal_payload={
            "lane": {"lane_id": "lane_b", "lane_name": "Lane Beta"},
            "full_decision": {
                "decision": "REVIEW_INFERRED",
                "review_reason_code": "inferred_high_priority",
            },
        },
    )
    _seed_candidate_terminal(
        store,
        run_id=run_id,
        brief_id=brief_id,
        identity_key="cand-legacy-reject",
        work_unit_id=wu_legacy,
        terminal_decision="REJECT",
        terminal_payload={"full_decision": {"decision": "REJECT"}},
    )

    rows = lane_metrics_for_run(store.db_path, run_id=run_id)
    by_lane = {row.lane_id: row for row in rows}

    assert set(by_lane.keys()) == {"lane_a", "lane_b", LEGACY_LANE_ID}
    # Sorted: non-legacy alpha then legacy last.
    assert [r.lane_id for r in rows] == ["lane_a", "lane_b", LEGACY_LANE_ID]

    a = by_lane["lane_a"]
    assert a.lane_name == "Lane Alpha"
    assert a.lane_intent == "builders"
    assert a.acquisition_mode == "linkedin_boolean"
    assert a.result_count == 100
    assert a.candidates_seen == 5
    assert a.facial_yes_count == 2
    assert a.facial_no_count == 3
    assert a.save_count == 1
    assert a.reject_count == 0
    assert a.review_count == 0
    assert a.work_unit_source_ids == ("11",)
    assert a.cost_usd is None
    assert a.legacy is False

    b = by_lane["lane_b"]
    assert b.save_count == 0
    assert b.reject_count == 0
    assert b.review_count == 1
    assert b.review_by_reason == {"inferred_high_priority": 1}
    assert b.legacy is False

    legacy_row = by_lane[LEGACY_LANE_ID]
    assert legacy_row.legacy is True
    assert legacy_row.save_count == 0
    assert legacy_row.reject_count == 1
    assert legacy_row.review_count == 0


def test_review_outcomes_never_inflate_save_count(tmp_path: Path):
    """The P4 / P5 separation invariant — REVIEW counts under
    ``review_count`` only, never under ``save_count``.
    """

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    brief_id = "brief-review-separation"
    run_id = _start_run(store, brief_id=brief_id)
    wu = _upsert_lane_work_unit(
        store,
        run_id=run_id,
        brief_id=brief_id,
        source_unit_id="1",
        lane_id="lane_sep",
    )
    # One SAVE + one REVIEW_INFERRED + one REVIEW_FLAGGED + one REJECT.
    for identity_key, terminal, reason in [
        ("c1", "SAVE", None),
        (
            "c2",
            "REVIEW_INFERRED",
            "inferred_high_priority",
        ),
        ("c3", "REVIEW_FLAGGED", "needs_more_evidence"),
        ("c4", "REJECT", None),
    ]:
        payload: dict = {"full_decision": {"decision": terminal}}
        if reason:
            payload["full_decision"]["review_reason_code"] = reason
            payload["lane"] = {"lane_id": "lane_sep"}
        _seed_candidate_terminal(
            store,
            run_id=run_id,
            brief_id=brief_id,
            identity_key=identity_key,
            work_unit_id=wu,
            terminal_decision=terminal,
            terminal_payload=payload,
        )

    rows = lane_metrics_for_run(store.db_path, run_id=run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.lane_id == "lane_sep"
    assert row.save_count == 1
    assert row.reject_count == 1
    assert row.review_count == 2
    assert row.review_by_reason == {
        "inferred_high_priority": 1,
        "needs_more_evidence": 1,
    }


def test_review_with_empty_reason_code_falls_back_to_unspecified(
    tmp_path: Path,
):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    brief_id = "brief-review-unspec"
    run_id = _start_run(store, brief_id=brief_id)
    wu = _upsert_lane_work_unit(
        store,
        run_id=run_id,
        brief_id=brief_id,
        source_unit_id="1",
        lane_id="lane_unspec",
    )
    _seed_candidate_terminal(
        store,
        run_id=run_id,
        brief_id=brief_id,
        identity_key="c1",
        work_unit_id=wu,
        terminal_decision="REVIEW_INFERRED",
        terminal_payload={
            # No review_reason_code anywhere — should land under
            # ``unspecified``.
            "lane": {"lane_id": "lane_unspec"},
            "full_decision": {"decision": "REVIEW_INFERRED"},
        },
    )
    rows = lane_metrics_for_run(store.db_path, run_id=run_id)
    assert rows[0].review_by_reason == {UNSPECIFIED_REVIEW_REASON: 1}


def test_review_terminal_payload_lane_overrides_work_unit_lane(tmp_path: Path):
    """A REVIEW candidate whose ``terminal_payload_json["lane"]["lane_id"]``
    is set wins over the candidate's ``last_work_unit_id`` work-unit
    payload (P4 attribution path).
    """

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    brief_id = "brief-attrib"
    run_id = _start_run(store, brief_id=brief_id)
    # Work unit has lane_id = "work_unit_lane"; candidate terminal
    # payload claims "candidate_lane". The candidate must attribute to
    # the candidate-side value.
    wu = _upsert_lane_work_unit(
        store,
        run_id=run_id,
        brief_id=brief_id,
        source_unit_id="1",
        lane_id="work_unit_lane",
    )
    _seed_candidate_terminal(
        store,
        run_id=run_id,
        brief_id=brief_id,
        identity_key="c1",
        work_unit_id=wu,
        terminal_decision="REVIEW_INFERRED",
        terminal_payload={
            "lane": {"lane_id": "candidate_lane"},
            "full_decision": {
                "decision": "REVIEW_INFERRED",
                "review_reason_code": "spot_check",
            },
        },
    )
    rows = lane_metrics_for_run(store.db_path, run_id=run_id)
    by_lane = {row.lane_id: row for row in rows}
    # The candidate is attributed to ``candidate_lane``; the work-unit
    # row still aggregates under ``work_unit_lane`` (typed counters /
    # source_unit_ids). Both buckets exist; saves/reviews are not
    # double-counted.
    assert "candidate_lane" in by_lane
    assert "work_unit_lane" in by_lane
    assert by_lane["candidate_lane"].review_count == 1
    assert by_lane["work_unit_lane"].review_count == 0
    assert by_lane["candidate_lane"].review_by_reason == {"spot_check": 1}


def test_opens_and_evaluated_semantics(tmp_path: Path):
    """``opened_count`` = distinct candidates with any full attempt;
    ``evaluated_count`` = distinct candidates whose full attempt
    succeeded.
    """

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    brief_id = "brief-opens"
    run_id = _start_run(store, brief_id=brief_id)
    wu = _upsert_lane_work_unit(
        store,
        run_id=run_id,
        brief_id=brief_id,
        source_unit_id="1",
        lane_id="lane_opens",
    )
    # One succeeded full attempt (counts as opened+evaluated).
    _seed_candidate_terminal(
        store,
        run_id=run_id,
        brief_id=brief_id,
        identity_key="c-eval",
        work_unit_id=wu,
        terminal_decision="SAVE",
        terminal_payload={"full_decision": {"decision": "SAVE"}},
    )
    # One opened but in-progress full attempt (opened, not evaluated).
    _seed_candidate_terminal(
        store,
        run_id=run_id,
        brief_id=brief_id,
        identity_key="c-open",
        work_unit_id=wu,
        terminal_decision="SAVE",  # unused: full_attempt_status != "succeeded"
        terminal_payload={},
        full_attempt_status="started",
    )

    rows = lane_metrics_for_run(store.db_path, run_id=run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.opened_count == 2
    assert row.evaluated_count == 1


def test_contained_resume_skip_is_neither_opened_nor_evaluated(tmp_path: Path):
    """A resume abandon is a skip receipt, not an open and not an evaluation.

    ``Pipeline._abandon_unrecoverable_pending_full`` settles a pending review
    the live Recruiter surface could not re-match by writing a succeeded full
    attempt. No profile was opened and no judge ran, so counting it would
    inflate opens and evaluations on exactly the lane whose surface went stale.
    """

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    brief_id = "brief-abandon"
    run_id = _start_run(store, brief_id=brief_id)
    wu = _upsert_lane_work_unit(
        store,
        run_id=run_id,
        brief_id=brief_id,
        source_unit_id="1",
        lane_id="lane_abandon",
    )
    # A genuine evaluation on the same lane, so the assertion reads "1, not 2".
    _seed_candidate_terminal(
        store,
        run_id=run_id,
        brief_id=brief_id,
        identity_key="c-real",
        work_unit_id=wu,
        terminal_decision="SAVE",
        terminal_payload={"full_decision": {"decision": "SAVE"}},
    )
    _seed_candidate_terminal(
        store,
        run_id=run_id,
        brief_id=brief_id,
        identity_key="c-abandoned",
        work_unit_id=wu,
        terminal_decision="JUDGMENT_FAILURE",
        terminal_payload={"full_decision": {"decision": "JUDGMENT_FAILURE"}},
        full_attempt_payload={
            "full_decision": {"decision": "JUDGMENT_FAILURE"},
            "pending_full_recovery_abandoned": True,
            "abandon_reason": "unmatched_profile",
        },
    )

    rows = lane_metrics_for_run(store.db_path, run_id=run_id)
    assert len(rows) == 1
    assert rows[0].opened_count == 1
    assert rows[0].evaluated_count == 1


def test_lane_metrics_reads_canonical_sqlite_not_projections(tmp_path: Path):
    """Source-of-truth invariant: even when a divergent projection
    artifact is written alongside the runtime SQLite, the read model
    aggregates SQLite, not the projection.
    """

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    brief_id = "brief-source-of-truth"
    run_id = _start_run(store, brief_id=brief_id)
    wu = _upsert_lane_work_unit(
        store,
        run_id=run_id,
        brief_id=brief_id,
        source_unit_id="1",
        lane_id="lane_canonical",
    )
    _seed_candidate_terminal(
        store,
        run_id=run_id,
        brief_id=brief_id,
        identity_key="c-save",
        work_unit_id=wu,
        terminal_decision="SAVE",
        terminal_payload={"full_decision": {"decision": "SAVE"}},
    )

    # Drop a divergent projection alongside the DB. If the aggregator
    # ever started reading projections, the assertion would flip.
    projection_path = tmp_path / "progress.json"
    projection_path.write_text(
        json.dumps(
            {
                "lane_metrics": [
                    {
                        "lane_id": "lane_canonical",
                        "save_count": 99,  # wildly wrong on purpose
                        "review_count": 7,
                    }
                ]
            }
        )
    )
    fake_saves_path = tmp_path / "lane_metrics.json"
    fake_saves_path.write_text(
        json.dumps({"lane_canonical": {"save_count": 99}})
    )

    rows = lane_metrics_for_run(store.db_path, run_id=run_id)
    assert len(rows) == 1
    assert rows[0].save_count == 1  # canonical SQLite value
    assert rows[0].review_count == 0


def test_lane_metrics_module_does_not_import_runtime_state_store():
    """Layering rule pin: ``lane_metrics.py`` must not import the writer
    store class (the writer's ``__init__`` runs DDL on instantiation, a
    hazard the read path is built to avoid). Mirrors the existing pin
    on ``read_models.py`` in ``tests/test_read_models.py``.
    """

    import ast

    source = lm.__file__
    assert source is not None
    tree = ast.parse(Path(source).read_text())

    forbidden = "RuntimeStateStore"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.endswith("runtime_state.store") or module == (
                "shared.runtime_state.store"
            ):
                names = [alias.name for alias in node.names]
                assert forbidden not in names, (
                    f"lane_metrics must not import {forbidden} from {module}"
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "shared.runtime_state.store", (
                    "lane_metrics must not import the writer module."
                )


def test_lane_metrics_cost_summed_when_metrics_json_carries_it(tmp_path: Path):
    """Best-effort cost pass-through: when at least one work unit in a
    lane carries ``cost_usd`` in ``metrics_json``, the row reports the
    sum. When no work unit carries a numeric cost, the row stays at
    ``None`` so consumers can distinguish "no data" from "$0".
    """

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    brief_id = "brief-cost"
    run_id = _start_run(store, brief_id=brief_id)
    _upsert_lane_work_unit(
        store,
        run_id=run_id,
        brief_id=brief_id,
        source_unit_id="1",
        lane_id="lane_cost",
        metrics={"cost_usd": 1.25},
    )
    _upsert_lane_work_unit(
        store,
        run_id=run_id,
        brief_id=brief_id,
        source_unit_id="2",
        lane_id="lane_cost",
        metrics={"cost_usd": 0.75},
    )
    _upsert_lane_work_unit(
        store,
        run_id=run_id,
        brief_id=brief_id,
        source_unit_id="3",
        lane_id="lane_no_cost",
    )

    rows = lane_metrics_for_run(store.db_path, run_id=run_id)
    by_lane = {row.lane_id: row for row in rows}
    assert by_lane["lane_cost"].cost_usd == pytest.approx(2.0)
    assert by_lane["lane_no_cost"].cost_usd is None


# ---------------------------------------------------------------------------
# FIX 2 seam — usage-log cost actually reaches metrics_json at sync time.
# ---------------------------------------------------------------------------


def test_sync_writes_usage_log_cost_into_work_unit_metrics(tmp_path: Path):
    """End-to-end seam for FIX 2: a run that incurred LLM cost (recorded in
    the per-call usage JSONL the way ``record_llm_usage`` writes it) must
    surface a non-None ``cost_usd`` for that lane in ``lane_metrics``.

    Pre-fix this asserts ``None`` because ``sync_linkedin_progress`` never
    wrote ``cost_usd`` into ``work_units.metrics_json`` — the read side's
    ``_coerce_cost`` had nothing to coerce. Post-fix the sync path rolls
    the usage log up per lane and attributes it to one work unit per lane.
    """

    from shared.llm_usage import llm_usage_session, record_llm_usage
    from shared.runtime_state.linkedin_progress_sync import (
        lane_cost_from_usage_log,
        sync_linkedin_progress,
    )
    from shared.runtime_state.linkedin import LinkedInRuntimeStateBridge
    from shared.schemas import Progress, SearchString

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    brief_id = "brief-cost-seam"
    run_id = _start_run(store, brief_id=brief_id)

    # Two search strings share lane "ml-infra"; one is in lane "platform".
    # Two judge calls land against ml-infra and one against platform — the
    # usage log is keyed by lane_id, NOT by search-string/work-unit id.
    usage_log = tmp_path / "token-cost-log.jsonl"
    with llm_usage_session(usage_log, module="linkedin", brief_id=brief_id):
        record_llm_usage(
            provider="anthropic",
            model="claude-opus",
            usage={"input_tokens": 1000, "output_tokens": 500},
            usage_context={"lane_id": "ml-infra", "stage": "facial"},
        )
        record_llm_usage(
            provider="anthropic",
            model="claude-opus",
            usage={"input_tokens": 2000, "output_tokens": 800},
            usage_context={"lane_id": "ml-infra", "stage": "full_eval"},
        )
        record_llm_usage(
            provider="anthropic",
            model="claude-opus",
            usage={"input_tokens": 500, "output_tokens": 100},
            usage_context={"lane_id": "platform", "stage": "facial"},
        )

    lane_cost = lane_cost_from_usage_log(usage_log)
    assert "ml-infra" in lane_cost and "platform" in lane_cost
    assert lane_cost["ml-infra"] > lane_cost["platform"] > 0

    progress = Progress(
        brief_name="test",
        strings=[
            SearchString(id=1, name="s1", boolean="a", lane_id="ml-infra", lane_name="ML Infra"),
            SearchString(id=2, name="s2", boolean="b", lane_id="ml-infra", lane_name="ML Infra"),
            SearchString(id=3, name="s3", boolean="c", lane_id="platform", lane_name="Platform"),
        ],
    )

    sync_linkedin_progress(
        store=store,
        run_id=run_id,
        brief_id=brief_id,
        progress=progress,
        rebuild_artifacts=lambda _run_id: None,
        work_unit_metrics=LinkedInRuntimeStateBridge._work_unit_metrics,
        lane_cost_usd=lane_cost,
    )

    rows = lane_metrics_for_run(store.db_path, run_id=run_id)
    by_lane = {row.lane_id: row for row in rows}

    # The corrected behavior: each lane reports its true usage-log cost.
    assert by_lane["ml-infra"].cost_usd is not None
    assert by_lane["platform"].cost_usd is not None
    assert by_lane["ml-infra"].cost_usd == pytest.approx(lane_cost["ml-infra"])
    assert by_lane["platform"].cost_usd == pytest.approx(lane_cost["platform"])
    # Two work units share ml-infra; the lane total must NOT be doubled.
    assert by_lane["ml-infra"].cost_usd < lane_cost["ml-infra"] * 2
