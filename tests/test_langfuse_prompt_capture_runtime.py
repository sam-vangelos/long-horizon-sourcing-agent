from __future__ import annotations

import json
from pathlib import Path

from shared.execution.runtime import SharedExecutionRuntime
from shared.execution.types import CandidateExecutionEnvelope
from shared.runtime_state.store import RuntimeStateStore
from shared.schemas import CandidateSnippet, OpusDecision


def _make_store(tmp_path: Path) -> tuple[RuntimeStateStore, int]:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-x",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "brief-x"},
    )
    return store, run_id


def _make_runtime(store: RuntimeStateStore, tmp_path: Path) -> SharedExecutionRuntime:
    return SharedExecutionRuntime(
        store=store,
        output_dir=str(tmp_path),
        brief_id="brief-x",
        source="linkedin",
    )


def _make_envelope(run_id: int) -> CandidateExecutionEnvelope:
    snippet = CandidateSnippet(
        name="Ada Lovelace",
        headline="ML Engineer",
        current_title="ML Engineer",
        current_company="Analytical Engines",
        location="NYC",
        education_snippet="",
        profile_url="/talent/profile/ada",
        source_string_id=1,
        source_string_name="builders",
        page=1,
        result_rank=1,
    )
    return CandidateExecutionEnvelope(
        source="linkedin",
        brief_id="brief-x",
        run_id=run_id,
        work_unit_kind="linkedin_string",
        work_unit_source_id="1",
        identity_key=snippet.profile_url,
        display_name=snippet.name,
        profile_url=snippet.profile_url,
        snippet=snippet,
        source_cursor={"source_string_id": 1, "page": 1},
    )


def _prompt_capture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "facial",
        "source": "linkedin",
        "render_route": "linkedin.facial.v2_structural",
        "llm_caller": "facial_llm",
        "expect_json": False,
        "candidate_text": "compiled prompt body",
        "system_prompt_sha256": "abc123",
        "trace_id": "trace-1",
        "observation_id": "obs-1",
        "trace_url": "https://langfuse.test/trace-1",
    }


def test_runtime_success_splits_attempt_payload_from_terminal_payload(tmp_path: Path) -> None:
    store, run_id = _make_store(tmp_path)
    runtime = _make_runtime(store, tmp_path)
    envelope = _make_envelope(run_id)

    runtime.record_discovery(envelope)
    runtime.record_snippet_extracted(
        envelope,
        payload={"snippet": envelope.snippet.to_dict()},
    )
    attempt_id = runtime.start_stage(envelope, stage="facial")
    decision = OpusDecision(
        stage="facial",
        decision="FACIAL_NO",
        path="none",
        confidence=0.91,
        rationale="clear non-fit",
        candidate_name=envelope.display_name,
        profile_url=envelope.profile_url,
        prompt_capture=_prompt_capture(),
    )

    runtime.finish_stage_success(
        attempt_id=attempt_id,
        envelope=envelope,
        stage="facial",
        decision=decision,
    )

    with store.connect() as conn:
        attempt_row = conn.execute(
            "SELECT payload_json FROM candidate_attempts WHERE id = ?",
            (attempt_id,),
        ).fetchone()
        candidate_row = conn.execute(
            "SELECT terminal_payload_json FROM candidates WHERE identity_key = ?",
            (envelope.identity_key,),
        ).fetchone()
        event_row = conn.execute(
            "SELECT payload_json FROM events WHERE attempt_id = ? AND event_type = 'attempt_succeeded' ORDER BY id DESC LIMIT 1",
            (attempt_id,),
        ).fetchone()

    attempt_payload = json.loads(attempt_row["payload_json"])
    terminal_payload = json.loads(candidate_row["terminal_payload_json"])
    event_payload = json.loads(event_row["payload_json"])

    assert attempt_payload["prompt_capture"]["candidate_text"] == "compiled prompt body"
    assert terminal_payload["facial_decision"]["decision"] == "FACIAL_NO"
    assert terminal_payload["observability"] == {
        "trace_id": "trace-1",
        "observation_id": "obs-1",
        "trace_url": "https://langfuse.test/trace-1",
        "prompt_capture_schema_version": 1,
    }
    assert "prompt_capture" not in terminal_payload
    assert event_payload["trace_id"] == "trace-1"
    assert event_payload["observation_id"] == "obs-1"


def test_runtime_failure_event_carries_trace_ids_when_failure_decision_has_prompt_capture(
    tmp_path: Path,
) -> None:
    store, run_id = _make_store(tmp_path)
    runtime = _make_runtime(store, tmp_path)
    envelope = _make_envelope(run_id)

    runtime.record_discovery(envelope)
    runtime.record_snippet_extracted(
        envelope,
        payload={"snippet": envelope.snippet.to_dict()},
    )
    attempt_id = runtime.start_stage(envelope, stage="facial")
    failure_decision = OpusDecision(
        stage="facial",
        decision="JUDGMENT_FAILURE",
        path="none",
        confidence=0.0,
        rationale="parse failed",
        candidate_name=envelope.display_name,
        profile_url=envelope.profile_url,
        prompt_capture=_prompt_capture(),
    )

    runtime.finish_stage_failure(
        attempt_id=attempt_id,
        envelope=envelope,
        stage="facial",
        error_or_failure_decision=failure_decision,
    )

    with store.connect() as conn:
        event_row = conn.execute(
            "SELECT payload_json FROM events WHERE attempt_id = ? AND event_type = 'attempt_failed' ORDER BY id DESC LIMIT 1",
            (attempt_id,),
        ).fetchone()
    event_payload = json.loads(event_row["payload_json"])
    assert event_payload["trace_id"] == "trace-1"
    assert event_payload["observation_id"] == "obs-1"


def test_store_finish_attempt_success_legacy_payload_mirroring_still_works(tmp_path: Path) -> None:
    store, run_id = _make_store(tmp_path)
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-x",
        identity_key="legacy-candidate",
        display_name="Legacy Candidate",
        profile_url="https://example.test/legacy",
    )
    attempt_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-x",
        identity_key="legacy-candidate",
        stage="snippet",
        payload={"source": "legacy"},
        source_cursor={"page": 1},
        display_name="Legacy Candidate",
        profile_url="https://example.test/legacy",
    )

    store.finish_attempt_success(
        attempt_id=attempt_id,
        new_state="snippet_extracted",
        payload={"snippet": {"name": "Legacy Candidate"}},
        run_id=run_id,
    )

    candidate = store.get_candidate(
        source="linkedin",
        brief_id="brief-x",
        identity_key="legacy-candidate",
    )
    assert json.loads(candidate["terminal_payload_json"]) == {
        "snippet": {"name": "Legacy Candidate"}
    }
