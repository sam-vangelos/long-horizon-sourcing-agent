"""Langfuse dataset export — Phase 2 tests.

Pins the contract for :func:`shared.runtime_state.calibration.build_langfuse_dataset_rows`
+ the sync tool's discovery / idempotency invariants.

Tests run without Langfuse credentials (the export helper is pure
read; the sync tool's network calls live behind the singleton's
no-op posture).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from shared.runtime_state.calibration import (
    LangfuseDatasetRow,
    build_langfuse_dataset_rows,
)
import shared.runtime_state.calibration as calibration_module
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Fixture builders (mirror tests/test_calibration_aggregator.py)
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    db_path = tmp_path / "runtime_state.sqlite3"
    return RuntimeStateStore(db_path)


def _ensure_run(
    store: RuntimeStateStore, *, source: str, brief_id: str
) -> int:
    runs = store.list_runs(source=source, brief_id=brief_id)
    if runs:
        return int(runs[0]["id"])
    return store.start_run(
        source=source,
        brief_id=brief_id,
        output_dir=str(store.db_path.parent),
        mode="fresh",
        resume_state={"brief_name": brief_id},
    )


def _candidate_id(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    identity_key: str,
) -> int:
    candidate = store.get_candidate(
        source=source, brief_id=brief_id, identity_key=identity_key
    )
    assert candidate is not None
    return int(candidate["id"])


def _seed_marked_candidate(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    identity_key: str,
    judgment_accuracy: str,
    rationale: str,
    confidence: float,
    terminal_decision: str = "SAVE",
) -> int:
    """Walk a candidate through to terminal + stamp the marker."""

    run_id = _ensure_run(store, source=source, brief_id=brief_id)
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
    store.set_candidate_state(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="full_terminal",
        terminal_decision=terminal_decision,
        terminal_payload={
            "full_decision": {
                "decision": terminal_decision,
                "rationale": rationale,
                "confidence": confidence,
                "capability_area": "Capability A",
            }
        },
    )
    cid = _candidate_id(
        store, source=source, brief_id=brief_id, identity_key=identity_key
    )
    store.set_candidate_judgment_accuracy(cid, judgment_accuracy)
    return cid


def _brief_dict() -> dict:
    return {
        "role_title": "Senior FDE",
        "capability_areas": [
            {"name": "Capability A", "description": "First capability."},
            {"name": "Capability B", "description": "Second capability."},
        ],
        "depth_distinction": {
            "builder_definition": "Strong builder.",
            "user_definition": "Strong consumer.",
        },
    }


def _seed_marked_candidate_with_prompt_capture(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    identity_key: str,
    judgment_accuracy: str,
    rationale: str,
    confidence: float,
    candidate_text: str,
    trace_id: str,
    observation_id: str,
    trace_url: str,
    terminal_decision: str = "SAVE",
) -> int:
    run_id = _ensure_run(store, source=source, brief_id=brief_id)
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
    attempt_id = store.start_attempt(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        stage="full",
        payload={},
        source_cursor={"source_string_id": 1},
        display_name=f"Candidate {identity_key}",
        profile_url=f"https://example.test/{identity_key}",
    )
    prompt_capture = {
        "schema_version": 1,
        "stage": "full",
        "source": source,
        "render_route": f"{source}.full.v2_structural",
        "llm_caller": "opus_llm_cached",
        "expect_json": False,
        "candidate_text": candidate_text,
        "system_prompt_sha256": "abc123",
        "trace_id": trace_id,
        "observation_id": observation_id,
        "trace_url": trace_url,
    }
    terminal_payload = {
        "full_decision": {
            "decision": terminal_decision,
            "rationale": rationale,
            "confidence": confidence,
            "capability_area": "Capability A",
        },
        "observability": {
            "trace_id": trace_id,
            "observation_id": observation_id,
            "trace_url": trace_url,
            "prompt_capture_schema_version": 1,
        },
    }
    store.finish_attempt_success(
        attempt_id=attempt_id,
        new_state="full_terminal",
        terminal_decision=terminal_decision,
        payload={
            "full_decision": terminal_payload["full_decision"],
            "prompt_capture": prompt_capture,
        },
        terminal_payload=terminal_payload,
        run_id=run_id,
    )
    cid = _candidate_id(
        store, source=source, brief_id=brief_id, identity_key=identity_key
    )
    store.set_candidate_judgment_accuracy(cid, judgment_accuracy)
    return cid


# ---------------------------------------------------------------------------
# build_langfuse_dataset_rows — schema + idempotency
# ---------------------------------------------------------------------------


def test_missing_db_returns_empty_list(tmp_path: Path) -> None:
    rows = build_langfuse_dataset_rows(
        tmp_path / "no.sqlite3", brief_id="brief-x"
    )
    assert rows == []


def test_export_emits_one_row_per_marked_candidate(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-A",
        judgment_accuracy="useful",
        rationale="Strong FDE candidate.",
        confidence=0.9,
    )
    _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-B",
        judgment_accuracy="wrong",
        rationale="Off-thesis read.",
        confidence=0.3,
    )

    rows = build_langfuse_dataset_rows(
        store.db_path, brief_id="brief-1", brief_dict=_brief_dict()
    )

    assert len(rows) == 2
    by_key = {r.identity_key: r for r in rows}
    assert "li-A" in by_key
    assert "li-B" in by_key


def test_export_row_carries_pinned_schema(tmp_path: Path) -> None:
    """The row schema is pinned in the plan body. Each row must
    carry input{brief_id, candidate_summary, candidate_text,
    capability_areas, depth_distinction, source},
    expected_output{judgment_accuracy, full_decision_rationale,
    recruiter_marker_set_at}, metadata {identity_key,
    confidence_at_eval, trace_id, observation_id, trace_url,
    capture_mode, cascade_route_hit}."""

    store = _make_store(tmp_path)
    _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-pin",
        identity_key="li-pin",
        judgment_accuracy="useful",
        rationale="Top-decile builder; cited capability A.",
        confidence=0.95,
    )

    rows = build_langfuse_dataset_rows(
        store.db_path, brief_id="brief-pin", brief_dict=_brief_dict()
    )
    assert len(rows) == 1
    row = rows[0]

    # input keys.
    assert set(row.input.keys()) == {
        "brief_id",
        "candidate_summary",
        "candidate_text",
        "capability_areas",
        "depth_distinction",
        "source",
    }
    assert row.input["brief_id"] == "brief-pin"
    assert row.input["source"] == "linkedin"
    assert "Top-decile builder" in row.input["candidate_summary"]
    assert row.input["candidate_text"] == row.input["candidate_summary"]
    assert len(row.input["capability_areas"]) == 2
    assert row.input["capability_areas"][0]["name"] == "Capability A"
    assert row.input["depth_distinction"]["builder_definition"] == "Strong builder."

    # expected_output keys.
    assert set(row.expected_output.keys()) == {
        "judgment_accuracy",
        "full_decision_rationale",
        "recruiter_marker_set_at",
    }
    assert row.expected_output["judgment_accuracy"] == "useful"
    assert "Top-decile builder" in row.expected_output["full_decision_rationale"]
    # judgment_accuracy_at is set at marker time → ISO timestamp.
    assert row.expected_output["recruiter_marker_set_at"]

    # metadata keys.
    assert set(row.metadata.keys()) == {
        "identity_key",
        "confidence_at_eval",
        "trace_id",
        "observation_id",
        "trace_url",
        "capture_mode",
        "cascade_route_hit",
    }
    assert row.metadata["identity_key"] == "li-pin"
    assert row.metadata["confidence_at_eval"] == 0.95
    assert row.metadata["trace_id"] is None
    assert row.metadata["observation_id"] is None
    assert row.metadata["trace_url"] is None
    assert row.metadata["capture_mode"] == "legacy_summary_fallback"
    assert row.metadata["cascade_route_hit"] is None


def test_export_filters_unmarked_candidates(tmp_path: Path) -> None:
    """A candidate without a recruiter marker doesn't appear in the
    dataset export — the dataset is recruiter-feedback-only by
    contract."""

    store = _make_store(tmp_path)
    _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-mix",
        identity_key="li-marked",
        judgment_accuracy="useful",
        rationale="Strong.",
        confidence=0.8,
    )
    # An unmarked candidate (no judgment_accuracy stamp).
    run_id = _ensure_run(store, source="linkedin", brief_id="brief-mix")
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-mix",
        identity_key="li-unmarked",
        display_name="Unmarked",
        profile_url="https://example.test/unmarked",
    )

    rows = build_langfuse_dataset_rows(
        store.db_path, brief_id="brief-mix", brief_dict=_brief_dict()
    )
    assert len(rows) == 1
    assert rows[0].identity_key == "li-marked"


def test_export_filters_by_source(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-multi",
        identity_key="li-1",
        judgment_accuracy="useful",
        rationale="LinkedIn read.",
        confidence=0.85,
    )
    _seed_marked_candidate(
        store,
        source="github",
        brief_id="brief-multi",
        identity_key="gh-1",
        judgment_accuracy="off_rubric",
        rationale="GitHub read.",
        confidence=0.4,
    )

    li_rows = build_langfuse_dataset_rows(
        store.db_path,
        brief_id="brief-multi",
        brief_dict=_brief_dict(),
        source="linkedin",
    )
    assert len(li_rows) == 1
    assert li_rows[0].input["source"] == "linkedin"

    gh_rows = build_langfuse_dataset_rows(
        store.db_path,
        brief_id="brief-multi",
        brief_dict=_brief_dict(),
        source="github",
    )
    assert len(gh_rows) == 1
    assert gh_rows[0].input["source"] == "github"


def test_export_drops_rows_with_unknown_marker(tmp_path: Path) -> None:
    """Defensive: a legacy row with a marker outside the writer-
    validated set is dropped silently — same posture as the
    aggregator at calibration.py."""

    store = _make_store(tmp_path)
    _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-legacy",
        identity_key="li-good",
        judgment_accuracy="useful",
        rationale="Real marker.",
        confidence=0.7,
    )
    # Bypass the writer's validation gate by writing directly.
    legacy_id = _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-legacy",
        identity_key="li-legacy",
        judgment_accuracy="useful",
        rationale="To be hijacked.",
        confidence=0.5,
    )
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE candidates SET judgment_accuracy = ? WHERE id = ?",
            ("legacy_unknown_marker", legacy_id),
        )
        conn.commit()

    rows = build_langfuse_dataset_rows(
        store.db_path,
        brief_id="brief-legacy",
        brief_dict=_brief_dict(),
    )
    assert len(rows) == 1
    assert rows[0].identity_key == "li-good"


def test_export_handles_missing_brief_dict(tmp_path: Path) -> None:
    """When brief_dict is None, capability_areas + depth_distinction
    fall back to empty placeholders so unit tests / scripts don't
    have to load a real brief."""

    store = _make_store(tmp_path)
    _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-nobrief",
        identity_key="li-1",
        judgment_accuracy="useful",
        rationale="Test.",
        confidence=0.5,
    )
    rows = build_langfuse_dataset_rows(
        store.db_path, brief_id="brief-nobrief", brief_dict=None
    )
    assert len(rows) == 1
    assert rows[0].input["capability_areas"] == []
    assert rows[0].input["depth_distinction"] == {}


def test_export_prefers_captured_prompt_input_and_trace_metadata(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_marked_candidate_with_prompt_capture(
        store,
        source="linkedin",
        brief_id="brief-capture",
        identity_key="li-capture",
        judgment_accuracy="useful",
        rationale="Strong FDE candidate.",
        confidence=0.88,
        candidate_text="Exact prompt body from eval time",
        trace_id="trace-1",
        observation_id="obs-1",
        trace_url="https://langfuse.test/trace-1",
    )

    class _Observation:
        def __init__(self, obs_id: str, metadata: dict[str, object]) -> None:
            self.id = obs_id
            self.metadata = metadata

    class _Trace:
        def __init__(self) -> None:
            self.observations = [_Observation("obs-1", {"cascade.fallback_reason": "schema_invalid"})]

    def _fake_fetch(trace_id: str):
        assert trace_id == "trace-1"
        return _Trace()

    original = calibration_module._fetch_trace_details
    calibration_module._fetch_trace_details = _fake_fetch
    try:
        rows = build_langfuse_dataset_rows(
            store.db_path, brief_id="brief-capture", brief_dict=_brief_dict()
        )
    finally:
        calibration_module._fetch_trace_details = original

    assert len(rows) == 1
    row = rows[0]
    assert row.input["candidate_text"] == "Exact prompt body from eval time"
    assert row.metadata["trace_id"] == "trace-1"
    assert row.metadata["observation_id"] == "obs-1"
    assert row.metadata["trace_url"] == "https://langfuse.test/trace-1"
    assert row.metadata["capture_mode"] == "captured_prompt"
    assert row.metadata["cascade_route_hit"] == "schema_invalid"


def test_export_caches_trace_fetches_per_trace_id(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_marked_candidate_with_prompt_capture(
        store,
        source="linkedin",
        brief_id="brief-cache",
        identity_key="li-cache-a",
        judgment_accuracy="useful",
        rationale="Row A.",
        confidence=0.91,
        candidate_text="Prompt A",
        trace_id="shared-trace",
        observation_id="obs-a",
        trace_url="https://langfuse.test/shared-trace",
    )
    _seed_marked_candidate_with_prompt_capture(
        store,
        source="linkedin",
        brief_id="brief-cache",
        identity_key="li-cache-b",
        judgment_accuracy="wrong",
        rationale="Row B.",
        confidence=0.33,
        candidate_text="Prompt B",
        trace_id="shared-trace",
        observation_id="obs-b",
        trace_url="https://langfuse.test/shared-trace",
    )

    fetch_calls: list[str] = []

    class _Observation:
        def __init__(self, obs_id: str, metadata: dict[str, object]) -> None:
            self.id = obs_id
            self.metadata = metadata

    class _Trace:
        observations = [
            _Observation("obs-a", {"cascade.fallback_reason": "schema_invalid"}),
            _Observation("obs-b", {}),
        ]

    def _fake_fetch(trace_id: str):
        fetch_calls.append(trace_id)
        return _Trace()

    original = calibration_module._fetch_trace_details
    calibration_module._fetch_trace_details = _fake_fetch
    try:
        rows = build_langfuse_dataset_rows(
            store.db_path, brief_id="brief-cache", brief_dict=_brief_dict()
        )
    finally:
        calibration_module._fetch_trace_details = original

    assert fetch_calls == ["shared-trace"]
    by_key = {row.identity_key: row for row in rows}
    assert by_key["li-cache-a"].metadata["cascade_route_hit"] == "schema_invalid"
    assert by_key["li-cache-b"].metadata["cascade_route_hit"] == "clean"


def test_dataset_row_to_dataset_item_shape() -> None:
    """``LangfuseDatasetRow.to_dataset_item`` returns a JSON-
    serializable dict ready for the Langfuse API."""

    row = LangfuseDatasetRow(
        input={"brief_id": "b", "source": "linkedin"},
        expected_output={"judgment_accuracy": "useful"},
        metadata={"identity_key": "li-x"},
        identity_key="li-x",
    )
    item = row.to_dataset_item()
    assert set(item.keys()) == {"input", "expected_output", "metadata"}
    # Roundtrip through json.dumps to confirm serializability.
    serialized = json.dumps(item, sort_keys=True)
    assert json.loads(serialized) == item


# ---------------------------------------------------------------------------
# Sync tool — dataset name + state-dir discovery
# ---------------------------------------------------------------------------


def test_dataset_name_for_is_stable() -> None:
    from tools.sync_judgment_datasets import dataset_name_for

    assert dataset_name_for(source="linkedin", brief_id="brief-xyz") == (
        "judgment-accuracy-linkedin-brief-xyz"
    )


def test_discover_state_dirs_walks_output_state_root(tmp_path: Path) -> None:
    from tools.sync_judgment_datasets import _discover_state_dirs

    # Build output/state/<source>/<state_key>/runtime_state.sqlite3
    for source in ("linkedin", "github"):
        for key in ("brief-a", "brief-b"):
            state_dir = tmp_path / "state" / source / key
            state_dir.mkdir(parents=True)
            (state_dir / "runtime_state.sqlite3").write_text("stub")

    found = _discover_state_dirs(output_root=tmp_path)
    sources_seen = {source for source, _ in found}
    assert sources_seen == {"linkedin", "github"}
    assert len(found) == 4

    # Filter to one source.
    li_only = _discover_state_dirs(output_root=tmp_path, sources=["linkedin"])
    assert {source for source, _ in li_only} == {"linkedin"}
    assert len(li_only) == 2


def test_discover_state_dirs_skips_dirs_without_runtime_state(tmp_path: Path) -> None:
    from tools.sync_judgment_datasets import _discover_state_dirs

    state_dir = tmp_path / "state" / "linkedin" / "no-db"
    state_dir.mkdir(parents=True)
    # No runtime_state.sqlite3 inside.

    found = _discover_state_dirs(output_root=tmp_path)
    assert found == []


def test_brief_ids_in_state_dir_lists_distinct_marked_briefs(
    tmp_path: Path,
) -> None:
    from tools.sync_judgment_datasets import _brief_ids_in_state_dir

    state_dir = tmp_path / "state" / "linkedin" / "key-a"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        judgment_accuracy="useful",
        rationale="r1",
        confidence=0.5,
    )
    _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-2",
        identity_key="li-2",
        judgment_accuracy="wrong",
        rationale="r2",
        confidence=0.4,
    )

    found = _brief_ids_in_state_dir(state_dir)
    assert set(found) == {"brief-1", "brief-2"}


# ---------------------------------------------------------------------------
# Sync tool — degraded-client guard
# ---------------------------------------------------------------------------


def test_sync_one_raises_when_langfuse_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sync tool's whole point is the Langfuse push. No-op on
    degraded would silently look like success — must raise."""

    monkeypatch.setenv("LANGFUSE_DISABLE", "1")
    from shared.observability.langfuse_client import reset_for_testing
    reset_for_testing()

    state_dir = tmp_path / "state" / "linkedin" / "key-a"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-x",
        identity_key="li-1",
        judgment_accuracy="useful",
        rationale="r",
        confidence=0.5,
    )

    from tools.sync_judgment_datasets import sync_one

    with pytest.raises(RuntimeError, match="Langfuse client is null"):
        sync_one(
            state_dir=state_dir,
            source="linkedin",
            brief_id="brief-x",
            dry_run=False,
        )


def test_sync_one_dry_run_skips_langfuse_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--dry-run`` builds rows + reports counts without pushing.
    Useful for operators verifying the row shape before the first
    sync."""

    monkeypatch.setenv("LANGFUSE_DISABLE", "1")
    from shared.observability.langfuse_client import reset_for_testing
    reset_for_testing()

    state_dir = tmp_path / "state" / "linkedin" / "key-a"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    _seed_marked_candidate(
        store,
        source="linkedin",
        brief_id="brief-dry",
        identity_key="li-dry",
        judgment_accuracy="useful",
        rationale="r",
        confidence=0.6,
    )

    from tools.sync_judgment_datasets import sync_one

    result = sync_one(
        state_dir=state_dir,
        source="linkedin",
        brief_id="brief-dry",
        dry_run=True,
    )
    assert result.dry_run is True
    assert result.rows_built == 1
    assert result.rows_pushed == 0
    assert result.failed_count == 0
    assert result.dataset_name == "judgment-accuracy-linkedin-brief-dry"
