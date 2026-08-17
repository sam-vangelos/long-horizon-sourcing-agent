"""Tests for compatibility projections derived from runtime_state."""

from __future__ import annotations

import json

from github.schemas import GitHubProgress, GitHubSearchQuery
from shared.runtime_state import RuntimeStateStore
from shared.runtime_state.projections import (
    project_github_facial_judgments,
    project_github_final_judgments,
    project_github_profile_summaries,
    project_github_progress,
    project_github_snippets,
    project_linkedin_candidate_history,
    project_linkedin_final_judgments,
    project_linkedin_progress,
    project_linkedin_search_memory,
    write_github_progress_projection,
    write_linkedin_candidate_history_projection,
    write_linkedin_progress_projection,
    write_linkedin_search_memory_projection,
)
from shared.runtime_state.store import LINKEDIN_STRING_KIND
from shared.schemas import SearchString


def _make_store(tmp_path):
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


def test_github_progress_projection_round_trip(tmp_path):
    store = _make_store(tmp_path)
    run_id = store.start_run(
        source="github",
        brief_id="brief-1",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "brief-1"},
    )
    progress = GitHubProgress(
        brief_name="brief-1",
        queries=[
            GitHubSearchQuery(id=1, name="q1", query="language:python", channel="user_search", status="done"),
            GitHubSearchQuery(id=2, name="q2", query="lang:rust", channel="code_search", status="queued"),
        ],
        candidates_discovered=12,
        candidates_enriched=4,
        candidates_saved=1,
        current_query_id=2,
        discovered_usernames=["alice"],
        graph_expansion_queue=[
            {
                "username": "seed-user",
                "reason": "SAVE",
                "confidence": 0.91,
                "capability_area": "infra",
            }
        ],
        graph_expansion_processed=["done-user"],
    )

    store.sync_github_progress(run_id, progress)
    projected = project_github_progress(store, run_id)

    assert [query.id for query in projected.queries] == [1, 2]
    assert projected.current_query_id == 2
    assert "alice" in projected.discovered_usernames
    assert projected.graph_expansion_queue[0]["username"] == "seed-user"
    assert projected.graph_expansion_processed == ["done-user"]

    path = tmp_path / "progress.json"
    write_github_progress_projection(store, run_id, path)
    first = path.read_text()
    write_github_progress_projection(store, run_id, path)
    second = path.read_text()
    assert json.loads(first) == json.loads(second)


def test_load_github_progress_round_trip_after_sync(tmp_path):
    """Pin store.load_github_progress field-for-field (Phase 4 P4-2 guard).

    Exercises the runtime construction sites for GitHubSearchQuery / GitHubProgress
    inside load_github_progress so the lazy-import refactor is provably inert.
    """
    store = _make_store(tmp_path)
    run_id = store.start_run(
        source="github",
        brief_id="brief-1",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "brief-1"},
    )
    progress = GitHubProgress(
        brief_name="brief-1",
        queries=[
            GitHubSearchQuery(id=1, name="q1", query="language:python", channel="user_search", status="done"),
            GitHubSearchQuery(id=2, name="q2", query="lang:rust", channel="code_search", status="queued"),
        ],
        candidates_discovered=12,
        candidates_enriched=4,
        candidates_saved=1,
        candidates_rejected=2,
        candidates_insufficient=3,
        current_query_id=2,
        discovered_usernames=["alice"],
        mined_repos=["org/repo"],
        api_calls_made=7,
        graph_expansion_queue=[
            {"username": "seed-user", "reason": "SAVE", "confidence": 0.91, "capability_area": "infra"}
        ],
        graph_expansion_processed=["done-user"],
    )

    store.sync_github_progress(run_id, progress)
    loaded = store.load_github_progress(run_id)

    assert loaded.brief_name == "brief-1"
    assert [q.id for q in loaded.queries] == [1, 2]
    assert [q.name for q in loaded.queries] == ["q1", "q2"]
    assert [q.status for q in loaded.queries] == ["done", "queued"]
    assert loaded.candidates_discovered == 12
    assert loaded.candidates_enriched == 4
    assert loaded.candidates_saved == 1
    assert loaded.candidates_rejected == 2
    assert loaded.candidates_insufficient == 3
    assert loaded.current_query_id == 2
    assert loaded.mined_repos == ["org/repo"]
    assert loaded.api_calls_made == 7
    assert "alice" in loaded.discovered_usernames
    assert loaded.graph_expansion_queue[0]["username"] == "seed-user"
    assert loaded.graph_expansion_processed == ["done-user"]


def test_github_stage_projections_round_trip(tmp_path):
    store = _make_store(tmp_path)
    run_id = store.start_run(
        source="github",
        brief_id="brief-1",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "brief-1"},
    )
    store.sync_github_progress(run_id, GitHubProgress(brief_name="brief-1", queries=[
        GitHubSearchQuery(id=1, name="q1", query="language:python", channel="user_search", status="done"),
    ]))
    work_unit_id = store.get_work_unit_id(run_id, kind="github_query", source_unit_id="1")
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        display_name="Alice",
        profile_url="https://github.com/alice",
        payload={"query_id": 1},
    )
    snippet_attempt_id = store.start_attempt(
        run_id=run_id,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        stage="preparation",
        work_unit_id=work_unit_id,
        payload={"cursor": {"query_id": 1}},
        source_cursor={"query_id": 1},
        display_name="Alice",
        profile_url="https://github.com/alice",
    )
    store.finish_attempt_success(
        attempt_id=snippet_attempt_id,
        new_state="snippet_extracted",
        payload={"cursor": {"query_id": 1}},
        run_id=run_id,
    )
    store.set_candidate_state(
        run_id=run_id,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        new_state="facial_started",
        last_work_unit_id=work_unit_id,
    )
    facial_attempt_id = store.start_attempt(
        run_id=run_id,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        stage="facial",
        work_unit_id=work_unit_id,
        payload={
            "snippet": {"candidate_name": "Alice", "profile_url": "https://github.com/alice"},
            "facial_decision": {"decision": "FACIAL_YES", "profile_url": "https://github.com/alice"},
        },
        source_cursor={"query_id": 1},
        display_name="Alice",
        profile_url="https://github.com/alice",
    )
    store.finish_attempt_success(
        attempt_id=facial_attempt_id,
        new_state="facial_terminal",
        payload={
            "snippet": {"candidate_name": "Alice", "profile_url": "https://github.com/alice"},
            "facial_decision": {"decision": "FACIAL_YES", "profile_url": "https://github.com/alice"},
        },
        run_id=run_id,
    )
    store.set_candidate_state(
        run_id=run_id,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        new_state="full_started",
        last_work_unit_id=work_unit_id,
    )
    full_attempt_id = store.start_attempt(
        run_id=run_id,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        stage="full",
        work_unit_id=work_unit_id,
        payload={
            "profile_summary": {"name": "Alice", "profile_url": "https://github.com/alice"},
            "full_decision": {"decision": "SAVE", "profile_url": "https://github.com/alice"},
        },
        source_cursor={"query_id": 1},
        display_name="Alice",
        profile_url="https://github.com/alice",
    )
    store.finish_attempt_success(
        attempt_id=full_attempt_id,
        new_state="full_terminal",
        terminal_decision="SAVE",
        payload={
            "profile_summary": {"name": "Alice", "profile_url": "https://github.com/alice"},
            "full_decision": {"decision": "SAVE", "profile_url": "https://github.com/alice"},
        },
        run_id=run_id,
    )

    assert project_github_snippets(store, brief_id="brief-1") == [
        {"candidate_name": "Alice", "profile_url": "https://github.com/alice"}
    ]
    assert project_github_facial_judgments(store, brief_id="brief-1") == [
        {"decision": "FACIAL_YES", "profile_url": "https://github.com/alice"}
    ]
    assert project_github_profile_summaries(store, brief_id="brief-1") == [
        {"name": "Alice", "profile_url": "https://github.com/alice"}
    ]
    assert project_github_final_judgments(store, brief_id="brief-1") == [
        {"decision": "SAVE", "profile_url": "https://github.com/alice"}
    ]


def test_linkedin_projections_cover_progress_history_and_search_memory(tmp_path):
    store = _make_store(tmp_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-1",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "brief-1"},
    )
    store.update_run_resume_state(
        run_id,
        {
            "brief_name": "brief-1",
            "current_string_id": 1,
            "current_page": 3,
            "pending_block_name": "Builders",
            "pending_block_string_ids": [1],
            "pending_block_ready": True,
            "candidates_saved": 1,
            "candidates_rejected": 2,
        },
    )

    done_string = SearchString(
        id=1,
        name="Payments edge case",
        boolean="payments AND fednow",
        status="done",
        result_count=11,
        pages_reviewed=3,
        saves=["Alice"],
        block="Payments",
        family_key="payments",
        novelty_bucket="edge_case",
        domain_lane="payments",
        candidates_count=8,
        duplicates_count=2,
        facial_yes_count=2,
        facial_no_count=5,
    )
    queued_string = SearchString(
        id=2,
        name="Capital markets canonical",
        boolean="capital markets AND workflow",
        status="queued",
        block="Markets",
        family_key="capital_markets",
        novelty_bucket="canonical",
        domain_lane="capital_markets",
    )

    store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        kind=LINKEDIN_STRING_KIND,
        source_unit_id=str(done_string.id),
        display_name=done_string.name,
        ordering_index=0,
        status="done",
        payload=done_string.to_dict(),
        checkpoint={"pages_reviewed": 3, "duplicates_count": 2},
        family_key=done_string.family_key,
        novelty_bucket=done_string.novelty_bucket,
        domain_lane=done_string.domain_lane,
        counters={
            "result_count": done_string.result_count,
            "candidates_discovered": done_string.candidates_count,
            "facial_yes_count": done_string.facial_yes_count,
            "facial_no_count": done_string.facial_no_count,
            "saves_count": len(done_string.saves),
        },
    )
    store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        kind=LINKEDIN_STRING_KIND,
        source_unit_id=str(queued_string.id),
        display_name=queued_string.name,
        ordering_index=1,
        status="queued",
        payload=queued_string.to_dict(),
        family_key=queued_string.family_key,
        novelty_bucket=queued_string.novelty_bucket,
        domain_lane=queued_string.domain_lane,
    )

    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-1",
        identity_key="https://linkedin.com/in/alice",
        display_name="Alice",
        profile_url="https://linkedin.com/in/alice",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="https://linkedin.com/in/alice",
        new_state="failed_terminal",
        terminal_decision="SAVE",
        terminal_payload={
            "confidence": 0.92,
            "source_string_id": 1,
            "timestamp": "2026-04-06T00:00:00+00:00",
        },
    )

    progress = project_linkedin_progress(store, run_id)
    history = project_linkedin_candidate_history(store, brief_id="brief-1")
    memory = project_linkedin_search_memory(store, brief_id="brief-1")

    assert progress.current_string_id == 1
    assert progress.current_page == 3
    assert progress.pending_block_ready is True
    assert [string.id for string in progress.strings] == [1, 2]

    assert history == [
        {
            "profile_url": "https://linkedin.com/in/alice",
            "candidate_name": "Alice",
            "outcome": "SAVE",
            "confidence": 0.92,
            "source_string_id": 1,
            "timestamp": "2026-04-06T00:00:00+00:00",
        }
    ]

    assert memory["project_id"] == "brief-1"
    assert memory["overall"]["strings_seen"] == 1
    assert "payments" in memory["families"]

    progress_path = tmp_path / "progress.json"
    history_path = tmp_path / "candidate_history-brief-1.jsonl"
    memory_path = tmp_path / "search_memory-brief-1.json"

    write_linkedin_progress_projection(store, run_id, progress_path)
    write_linkedin_candidate_history_projection(store, brief_id="brief-1", path=history_path)
    write_linkedin_search_memory_projection(store, brief_id="brief-1", path=memory_path)

    assert json.loads(progress_path.read_text())["current_page"] == 3
    assert "Alice" in history_path.read_text()
    assert json.loads(memory_path.read_text())["project_id"] == "brief-1"


def test_linkedin_final_judgments_projection_round_trip(tmp_path):
    """Canonical SQLite stores full-stage decisions under ``full_decision``.

    This test pins that the LinkedIn projection reads the correct canonical
    key. Without this test, the LinkedIn projection's payload-key drift can
    silently produce 0-byte ``final_judgments.jsonl`` across finalized runs
    (slice 18, runtime-state audit). Mirrors the GitHub finals slice in
    ``test_github_stage_projections_round_trip``.
    """

    store = _make_store(tmp_path)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-1",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "brief-1"},
    )
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-1",
        identity_key="https://linkedin.com/in/ada",
        display_name="Ada",
        profile_url="https://linkedin.com/in/ada",
        payload={"source_string_id": 1},
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="https://linkedin.com/in/ada",
        new_state="snippet_extracted",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="https://linkedin.com/in/ada",
        new_state="facial_started",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="https://linkedin.com/in/ada",
        new_state="facial_terminal",
        terminal_decision="FACIAL_YES",
        terminal_payload={"source_string_id": 1},
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="https://linkedin.com/in/ada",
        new_state="full_started",
    )
    full_attempt_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="https://linkedin.com/in/ada",
        stage="full",
        work_unit_id=None,
        payload={
            "profile_summary": {"name": "Ada", "profile_url": "https://linkedin.com/in/ada"},
            "full_decision": {"decision": "SAVE", "profile_url": "https://linkedin.com/in/ada"},
        },
        source_cursor={"source_string_id": 1},
        display_name="Ada",
        profile_url="https://linkedin.com/in/ada",
    )
    store.finish_attempt_success(
        attempt_id=full_attempt_id,
        new_state="full_terminal",
        terminal_decision="SAVE",
        payload={
            "profile_summary": {"name": "Ada", "profile_url": "https://linkedin.com/in/ada"},
            "full_decision": {"decision": "SAVE", "profile_url": "https://linkedin.com/in/ada"},
        },
        run_id=run_id,
    )

    assert project_linkedin_final_judgments(store, brief_id="brief-1") == [
        {"decision": "SAVE", "profile_url": "https://linkedin.com/in/ada"}
    ]


def test_projection_module_does_not_depend_on_lane_metrics():
    """P5: ``shared.runtime_state.projections`` is the compatibility
    projection layer; ``shared.runtime_state.lane_metrics`` is a
    canonical-SQLite read primitive. Projections must never import the
    aggregator (no cycle, no projection-derived control state). Pinned
    here so a drive-by edit does not invert the dependency direction.
    """

    import ast
    from pathlib import Path

    from shared.runtime_state import projections

    source = projections.__file__
    assert source is not None
    tree = ast.parse(Path(source).read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module != "shared.runtime_state.lane_metrics", (
                "projections must not import the lane_metrics aggregator; "
                "canonical SQLite is the source of truth for lane metrics, "
                "projections are non-authoritative."
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "shared.runtime_state.lane_metrics", (
                    "projections must not import lane_metrics"
                )
