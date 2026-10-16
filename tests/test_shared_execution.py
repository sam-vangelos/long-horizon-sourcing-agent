"""Shared candidate execution engine tests."""

from __future__ import annotations

import json

from shared.execution import CandidateExecutionEngine
from shared.runtime_state import RuntimeStateStore
from shared.schemas import CandidateSnippet, OpusDecision, SearchString


def _make_store(tmp_path):
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


def _snippet(url: str = "https://linkedin.com/in/ada") -> CandidateSnippet:
    return CandidateSnippet(
        name="Ada Lovelace",
        headline="ML Engineer",
        current_title="ML Engineer",
        current_company="Analytical Engines",
        location="NYC",
        education_snippet="",
        profile_url=url,
        source_string_id=1,
        source_string_name="builders",
        page=1,
        result_rank=1,
    )


def _setup_linkedin_engine(tmp_path):
    store = _make_store(tmp_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-1",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "brief-1"},
    )
    search_string = SearchString(id=1, name="builders", boolean="ml", status="queued")
    store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        kind="linkedin_string",
        source_unit_id="1",
        display_name=search_string.name,
        ordering_index=0,
        status="queued",
        payload=search_string.to_dict(),
    )
    engine = CandidateExecutionEngine(
        store=store,
        output_dir=str(tmp_path),
        brief_id="brief-1",
        source="linkedin",
    )
    return store, run_id, engine, search_string


def test_shared_engine_facial_yes_then_full_save_blocks_only_after_full_terminal(tmp_path):
    store, run_id, engine, search_string = _setup_linkedin_engine(tmp_path)
    snippet = _snippet()
    envelope = engine.envelope(
        source="linkedin",
        brief_id="brief-1",
        run_id=run_id,
        work_unit_kind="linkedin_string",
        work_unit_source_id="1",
        identity_key=snippet.profile_url,
        display_name=snippet.name,
        profile_url=snippet.profile_url,
        snippet=snippet,
        source_cursor={"source_string_id": search_string.id, "page": 1, "result_rank": 1},
    )
    engine.runtime.record_discovery(envelope, payload=envelope.source_cursor)
    engine.runtime.record_snippet_extracted(
        envelope,
        payload={"cursor": envelope.source_cursor, "snippet": snippet.to_dict()},
    )
    facial_attempt = engine.runtime.start_stage(envelope, stage="facial", payload={"cursor": envelope.source_cursor})
    engine.runtime.finish_stage_success(
        attempt_id=facial_attempt,
        envelope=envelope,
        stage="facial",
        decision=OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="none",
            confidence=0.8,
            rationale="worth opening",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        ),
    )
    assert store.is_dedup_blocked(source="linkedin", brief_id="brief-1", identity_key=snippet.profile_url) is False

    full_attempt = engine.runtime.start_stage(envelope, stage="full", payload={"cursor": envelope.source_cursor})
    engine.runtime.finish_stage_success(
        attempt_id=full_attempt,
        envelope=envelope,
        stage="full",
        decision=OpusDecision(
            stage="full",
            decision="SAVE",
            path="direct_experience",
            confidence=0.92,
            rationale="direct fit",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        ),
    )
    engine.runtime.flush_projections_if_needed(run_id=run_id, force_artifacts=True)

    assert store.is_dedup_blocked(source="linkedin", brief_id="brief-1", identity_key=snippet.profile_url) is True
    history_lines = (tmp_path / "candidate_history-brief-1.jsonl").read_text().strip().splitlines()
    assert any(json.loads(line)["outcome"] == "SAVE" for line in history_lines)


def test_shared_engine_failure_is_retryable_and_side_effect_does_not_change_decision(tmp_path):
    store, run_id, engine, search_string = _setup_linkedin_engine(tmp_path)
    snippet = _snippet("https://linkedin.com/in/grace")
    envelope = engine.envelope(
        source="linkedin",
        brief_id="brief-1",
        run_id=run_id,
        work_unit_kind="linkedin_string",
        work_unit_source_id="1",
        identity_key=snippet.profile_url,
        display_name=snippet.name,
        profile_url=snippet.profile_url,
        snippet=snippet,
        source_cursor={"source_string_id": search_string.id, "page": 1, "result_rank": 1},
    )
    engine.runtime.record_discovery(envelope, payload=envelope.source_cursor)
    engine.runtime.record_snippet_extracted(
        envelope,
        payload={"cursor": envelope.source_cursor, "snippet": snippet.to_dict()},
    )
    facial_attempt = engine.runtime.start_stage(envelope, stage="facial", payload={"cursor": envelope.source_cursor})
    engine.runtime.finish_stage_success(
        attempt_id=facial_attempt,
        envelope=envelope,
        stage="facial",
        decision=OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="promising_profile",
            confidence=0.88,
            rationale="worth full review",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        ),
    )
    full_attempt = engine.runtime.start_stage(envelope, stage="full", payload={"cursor": envelope.source_cursor})
    engine.runtime.finish_stage_failure(
        attempt_id=full_attempt,
        envelope=envelope,
        stage="full",
        error_or_failure_decision=RuntimeError("profile extraction exploded"),
        extra_payload={"profile_extraction_failed": True},
    )
    candidate = store.get_candidate(source="linkedin", brief_id="brief-1", identity_key=snippet.profile_url)
    assert candidate["current_lifecycle_state"] == "failed_retryable"

    retry_attempt = engine.runtime.start_stage(envelope, stage="full", payload={"cursor": envelope.source_cursor})
    engine.runtime.finish_stage_success(
        attempt_id=retry_attempt,
        envelope=envelope,
        stage="full",
        decision=OpusDecision(
            stage="full",
            decision="SAVE",
            path="direct_experience",
            confidence=0.95,
            rationale="strong builder",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        ),
    )
    engine.runtime.record_side_effect_result(
        envelope=envelope,
        attempt_id=retry_attempt,
        effect_type="linkedin_save",
        status="failed",
        payload={"reason": "click timeout"},
    )
    candidate = store.get_candidate(source="linkedin", brief_id="brief-1", identity_key=snippet.profile_url)
    assert candidate["terminal_decision"] == "SAVE"
    assert candidate["current_lifecycle_state"] == "full_terminal"
