"""LinkedIn runtime-state bridge regressions."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin.page_allocator import PageObservation
from linkedin.search_intelligence import (
    LinkedInPageInsights,
    LinkedInSearchVariant,
    LinkedInStructuredFilters,
    bootstrap_experiment_state,
)
from linkedin.timing_telemetry import RunLogTimingRecorder
from shared.runtime_state import LinkedInRuntimeStateBridge, RuntimeStateStore
from shared.failures import ApiBudgetExhaustedError
from shared.governor import OperatorStopRequested
from shared.schemas import (
    CandidateProfileSummary,
    CandidateSnippet,
    OpusDecision,
    Progress,
    SearchString,
)
from shared.safety import RunStopReason
from shared.storage import append_jsonl, read_jsonl


# Where a live run actually sits: the brief's OWN Recruiter project view.
# The global `https://www.linkedin.com/talent/search` view names no project,
# so with a project-pinned brief it is the F1 "unverified page" condition
# (run-start navigates off it, the pre-save boundary refuses from it) rather
# than neutral scenery.
_PROJECT_SEARCH_URL = (
    "https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
)


def _make_pipeline(output_dir: str):
    with patch("linkedin.orchestrator.load_brief") as mock_brief, \
         patch("linkedin.orchestrator.init_judger"), \
         patch("linkedin.orchestrator.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        brief.kit_url = ""
        # A truthy bare-Mock permanent_filters.get("Location") would read as
        # a phantom geography and trip the P3a fail-closed gate; real briefs
        # carry a dict.
        brief.permanent_filters = {}
        brief.needs_preflight = MagicMock(return_value=False)
        mock_brief.return_value = brief

        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline

        return Pipeline(brief_path=str(brief_path), output_dir=output_dir)


def _snippet(**kwargs) -> CandidateSnippet:
    defaults = {
        "name": "Ada Lovelace",
        "headline": "ML Engineer",
        "current_title": "ML Engineer",
        "current_company": "Analytical Engines",
        "location": "NYC",
        "education_snippet": "",
        "profile_url": "/talent/profile/ada",
        "source_string_id": 1,
        "source_string_name": "builders",
        "page": 1,
        "result_rank": 1,
    }
    defaults.update(kwargs)
    return CandidateSnippet(**defaults)


def _start_pipeline_runtime(p, search_string: SearchString) -> Progress:
    progress = Progress(brief_name="test", strings=[search_string])
    p._runtime_run_id, _ = p._runtime_bridge.start_or_resume_run(
        resume=False,
        initial_progress=progress,
    )
    return progress


def _facial_decision_payload(p, profile_url: str) -> dict:
    with p._runtime_state.connect() as conn:
        row = conn.execute(
            """
            SELECT ca.payload_json
            FROM candidate_attempts ca
            JOIN candidates c ON c.id = ca.candidate_id
            WHERE c.identity_key = ? AND ca.stage = 'facial'
            ORDER BY ca.id DESC
            LIMIT 1
            """,
            (profile_url,),
        ).fetchone()
    assert row is not None
    return json.loads(row["payload_json"])["facial_decision"]


def test_checkpoint_progress_emits_end_to_end_component_timings_fail_soft(
    tmp_path,
):
    pipeline = _make_pipeline(str(tmp_path))
    search_string = SearchString(id=1, name="timed", boolean="ml")
    progress = _start_pipeline_runtime(pipeline, search_string)
    pipeline._experiment_states[search_string.id] = bootstrap_experiment_state(
        search_string
    )
    pipeline._timing_recorder = RunLogTimingRecorder(pipeline.log_path)

    pipeline._checkpoint_progress(
        progress,
        search_string=search_string,
        page_num=1,
    )

    timing_records = [
        record
        for record in read_jsonl(pipeline.log_path)
        if record["event"] == "checkpoint_progress_timing"
    ]
    assert len(timing_records) == 1
    timing = timing_records[0]
    for field in (
        "elapsed_ms",
        "lane_cost_reparse_ms",
        "work_unit_rewrite_ms",
        "projection_rebuild_ms",
        "search_memory_reload_ms",
    ):
        assert timing[field] >= 0
    assert timing["elapsed_ms"] >= sum(
        timing[field]
        for field in (
            "lane_cost_reparse_ms",
            "work_unit_rewrite_ms",
            "projection_rebuild_ms",
            "search_memory_reload_ms",
        )
    ) - 1.0

    pipeline._timing_recorder = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("timing recorder failed")
    )
    pipeline._checkpoint_progress(progress, search_string=search_string, page_num=1)


def test_legacy_import_is_idempotent(tmp_path):
    output_dir = tmp_path / "linkedin-output"
    output_dir.mkdir()

    progress = Progress(
        brief_name="test",
        strings=[SearchString(id=1, name="builders", boolean="ml", status="done", pages_reviewed=1)],
        current_string_id=1,
        current_page=1,
    )
    progress.save(str(output_dir / "progress.json"))
    append_jsonl(output_dir / "snippets.jsonl", _snippet().to_dict())
    append_jsonl(
        output_dir / "facial_judgments.jsonl",
        OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="none",
            confidence=0.9,
            rationale="interesting builder",
            candidate_name="Ada Lovelace",
            profile_url="/talent/profile/ada",
        ).to_dict(),
    )
    append_jsonl(
        output_dir / "final_judgments.jsonl",
        OpusDecision(
            stage="full",
            decision="SAVE",
            path="direct_experience",
            confidence=0.93,
            rationale="strong direct fit",
            candidate_name="Ada Lovelace",
            profile_url="/talent/profile/ada",
        ).to_dict(),
    )
    append_jsonl(
        output_dir / "candidate_history-test-project.jsonl",
        {
            "profile_url": "/talent/profile/ada",
            "candidate_name": "Ada Lovelace",
            "outcome": "SAVE",
            "confidence": 0.93,
            "source_string_id": 1,
            "timestamp": "2026-04-07T00:00:00+00:00",
        },
    )

    store = RuntimeStateStore(output_dir / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=output_dir,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(output_dir),
        mode="resume",
        resume_state={"brief_name": "test"},
    )

    bridge.import_legacy_state(run_id)
    with store.connect() as conn:
        first_attempts = conn.execute("SELECT COUNT(*) AS count FROM candidate_attempts").fetchone()["count"]

    bridge.import_legacy_state(run_id)
    with store.connect() as conn:
        second_attempts = conn.execute("SELECT COUNT(*) AS count FROM candidate_attempts").fetchone()["count"]

    assert first_attempts == second_attempts


def test_full_semantic_evidence_and_about_persist_in_canonical_terminal_payload(
    tmp_path,
):
    """Structured GLM evidence and complete About text survive the SQLite write."""

    p = _make_pipeline(str(tmp_path))
    search_string = SearchString(id=1, name="builders", boolean="ml")
    _start_pipeline_runtime(p, search_string)
    snippet = _snippet(profile_url="/talent/profile/semantic-evidence")
    p._record_runtime_snippet(search_string, snippet)

    facial_attempt = p._start_runtime_stage_attempt(
        search_string=search_string,
        snippet=snippet,
        stage="facial",
    )
    p._finish_runtime_stage_success(
        attempt_id=facial_attempt,
        stage="facial",
        snippet=snippet,
        decision=OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="none",
            confidence=0.9,
            rationale="credible signal",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        ),
    )

    semantic_evidence = {
        "match_type": "ADJACENT",
        "capability_area": "Synthetic capability",
        "capability_evidence": "Owned a neighboring operating system.",
        "depth": "UNKNOWN",
        "depth_evidence": "Profile does not resolve implementation ownership.",
        "transferability": "TRANSFERABLE",
        "transferability_evidence": "Comparable scale and stakeholder work.",
        "case_for": "Strong enough to justify outreach.",
        "case_against": "Confirm builder depth in screen.",
    }
    summary = CandidateProfileSummary(
        name=snippet.name,
        profile_url=snippet.profile_url,
        headline=snippet.headline,
        about="Complete candidate-authored About text.\nSecond paragraph preserved.",
    )
    full_attempt = p._start_runtime_stage_attempt(
        search_string=search_string,
        snippet=snippet,
        stage="full",
    )
    p._finish_runtime_stage_success(
        attempt_id=full_attempt,
        stage="full",
        snippet=snippet,
        decision=OpusDecision(
            stage="full",
            decision="TRANSFERABLE_SAVE",
            path="ADJACENT:Synthetic capability|TRANSFERABLE",
            confidence=0.8,
            rationale="Outreach-positive adjacent case.",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
            semantic_evidence=semantic_evidence,
        ),
        profile_summary=summary,
    )

    with p._runtime_state.connect() as conn:
        candidate_row = conn.execute(
            "SELECT terminal_payload_json FROM candidates WHERE identity_key = ?",
            (snippet.profile_url,),
        ).fetchone()
        attempt_row = conn.execute(
            "SELECT payload_json FROM candidate_attempts "
            "WHERE candidate_id = (SELECT id FROM candidates WHERE identity_key = ?) "
            "AND stage = 'full' ORDER BY id DESC LIMIT 1",
            (snippet.profile_url,),
        ).fetchone()

    terminal_payload = json.loads(candidate_row["terminal_payload_json"])
    attempt_payload = json.loads(attempt_row["payload_json"])
    assert terminal_payload["full_decision"]["semantic_evidence"] == semantic_evidence
    assert terminal_payload["profile_summary"]["about"] == summary.about
    assert attempt_payload["full_decision"]["semantic_evidence"] == semantic_evidence
    assert attempt_payload["profile_summary"]["about"] == summary.about


def test_missing_profile_url_never_enters_dedup_state(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )
    bridge.sync_progress(
        run_id,
        Progress(brief_name="test", strings=[SearchString(id=1, name="builders", boolean="ml")]),
    )

    snippet = _snippet(profile_url="")
    bridge.record_snippet_extracted(
        run_id=run_id,
        search_string=SearchString(id=1, name="builders", boolean="ml"),
        snippet=snippet,
    )

    assert store.has_candidates(source="linkedin", brief_id="test-project") is False


def test_start_or_resume_run_imports_legacy_progress_when_runtime_absent(tmp_path):
    output_dir = tmp_path / "linkedin-output"
    output_dir.mkdir()
    legacy_progress = Progress(
        brief_name="test",
        strings=[SearchString(id=1, name="legacy", boolean="ml", status="done", pages_reviewed=2)],
        current_string_id=1,
        current_page=2,
    )
    legacy_progress.save(str(output_dir / "progress.json"))

    store = RuntimeStateStore(output_dir / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=output_dir,
        brief_id="test-project",
        brief_name="test",
    )

    run_id, progress = bridge.start_or_resume_run(resume=True)

    assert progress.current_string_id == 1
    assert progress.current_page == 2
    assert [(item.id, item.name, item.status, item.pages_reviewed) for item in progress.strings] == [
        (1, "legacy", "done", 2)
    ]
    rows = store.list_work_units(run_id, kind="linkedin_string")
    assert len(rows) == 1
    assert rows[0]["source_unit_id"] == "1"


def test_resume_comes_from_db_when_compat_files_are_stale(tmp_path):
    p = _make_pipeline(str(tmp_path))
    progress = Progress(
        brief_name="test",
        strings=[
            SearchString(id=2, name="second", boolean="two"),
            SearchString(id=1, name="first", boolean="one"),
        ],
    )
    p._runtime_run_id, progress = p._runtime_bridge.start_or_resume_run(
        resume=False,
        initial_progress=progress,
    )
    stale_progress = {
        "brief_name": "test",
        "strings": [{"id": 99, "name": "stale", "boolean": "stale", "status": "queued"}],
    }
    (tmp_path / "progress.json").write_text(json.dumps(stale_progress))
    history_path = tmp_path / "candidate_history-test-project.jsonl"
    memory_path = tmp_path / "search_memory-test-project.json"
    history_path.write_text("")
    if memory_path.exists():
        memory_path.unlink()

    p.browser.connect = AsyncMock()
    p.browser.disconnect = AsyncMock()
    p.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
    p._print_session_summary = MagicMock()
    p._print_summary = MagicMock()
    p._generate_run_report = MagicMock()
    p._finalize_run_snapshot = MagicMock(return_value=tmp_path / "frozen-run")
    p._enrich_run_snapshot = MagicMock()
    p._run_health_summary = MagicMock(
        return_value={"status": "ok", "green_but_useless": False}
    )
    p._session_expired = MagicMock()

    processed_ids: list[int] = []

    async def fake_process(search_string, progress):
        processed_ids.append(search_string.id)

    p._process_string = fake_process

    asyncio.run(p.run_full(resume=True))

    assert processed_ids == [2, 1]


def test_sync_progress_roundtrips_experiment_state(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )

    search_string = SearchString(id=1, name="builders", boolean="foo")
    state = bootstrap_experiment_state(search_string)
    state.begin_experiment_round(
        [
            LinkedInSearchVariant(
                variant_id="precision-1",
                parent_variant_id="root",
                root_string_id=1,
                boolean="foo AND bar",
                variant_kind="precision",
                target_result_min=75,
                target_result_max=400,
            )
        ]
    )
    state.activate_variant("precision-1")
    state.commit_variant("precision-1")
    observation = PageObservation(
        root_string_id=1,
        variant_id="precision-1",
        page=2,
        slots=5,
        extracted=5,
        full_expected=1,
        full_settled=1,
        priority=1,
        standard=0,
        outreach=1,
    )
    state.record_allocator_observation(observation)
    state.allocator_last_verdict = {"action": "switch", "selected_root_id": 2}
    state.allocator_shadow_diverged = True
    state.allocator_causality = {"spend_sequence": 3}
    state.allocator_frontier_expectation = {
        "root_ids": [1, 2],
        "dispositions": {"1": "queued", "2": "in_progress"},
    }
    state.apply_shadow(search_string)

    progress = Progress(brief_name="test", strings=[search_string], current_string_id=1, current_page=2)
    bridge.sync_progress(run_id, progress, experiment_states={1: state})

    loaded_progress = bridge.load_progress(run_id)
    loaded_states = bridge.load_experiment_states(run_id, progress=loaded_progress)

    assert loaded_progress.strings[0].boolean == "foo AND bar"
    assert loaded_progress.strings[0].refinement_stack == ["foo"]
    assert loaded_states[1].active_variant_id == "precision-1"
    assert loaded_states[1].committed_variant_id == "precision-1"
    assert loaded_states[1].allocator_last_observation == observation
    assert loaded_states[1].active_variant.allocator_valid_page_count == 1
    assert loaded_states[1].active_variant.allocator_completed_observation_count == 1
    assert loaded_states[1].allocator_last_verdict["action"] == "switch"
    assert loaded_states[1].allocator_shadow_diverged is True
    assert loaded_states[1].allocator_causality == {"spend_sequence": 3}
    assert loaded_states[1].allocator_frontier_expectation["root_ids"] == [1, 2]


def test_sync_progress_keeps_sqlite_commit_when_projection_rebuild_fails(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )
    search_string = SearchString(
        id=1,
        name="builders",
        boolean="foo",
        status="done",
    )
    progress = Progress(
        brief_name="test",
        strings=[search_string],
        current_string_id=1,
        current_page=2,
    )

    with patch.object(
        bridge,
        "rebuild_artifacts",
        side_effect=RuntimeError("injected projection failure"),
    ):
        bridge.sync_progress(run_id, progress)

    loaded = bridge.load_progress(run_id)
    assert loaded.current_string_id == 1
    assert loaded.current_page == 2
    assert loaded.strings[0].status == "done"


def test_sync_progress_surface_and_filters_agree_across_both_resume_flavors(tmp_path):
    """Slice G part 5: the active variant's surface + CURRENT structured_filters
    round-trip CONSISTENTLY through BOTH resume flavors —

      * the in-memory checkpoint (experiment_state.to_dict, preserved by the
        from_dict load path), and
      * the bootstrap-reconstruct path (the compat SearchString, used when the
        in-memory checkpoint is absent on a worker-death / cross-process resume).

    SCOPE (intentional): this exercises a PRISTINE-ROOT lane — the active variant is
    "root" (refinement_stack==[]), so both flavors round-trip the SAME variant's surface
    AND its own filters identically. The companion test below covers the mid-run
    refined hybrid lane that previously diverged and now reseeds filters on the
    in-memory resume path too.
    """
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )

    search_string = SearchString(
        id=1,
        name="structured only lane",
        boolean='"Staff Engineer"',
        acquisition_mode="linkedin_hybrid",
    )
    state = bootstrap_experiment_state(search_string)
    active = state.active_variant
    active.surface = "structured_only"
    active.structured_filters = LinkedInStructuredFilters(titles=["Staff Engineer"])
    state.apply_shadow(search_string)

    progress = Progress(brief_name="test", strings=[search_string], current_string_id=1, current_page=1)
    bridge.sync_progress(run_id, progress, experiment_states={1: state})

    # Flavor 1: in-memory checkpoint present -> from_dict restores surface + filters.
    loaded_progress = bridge.load_progress(run_id)
    in_memory_states = bridge.load_experiment_states(run_id, progress=loaded_progress)
    in_memory_active = in_memory_states[1].active_variant
    assert in_memory_active.surface == "structured_only"
    assert in_memory_active.structured_filters.titles == ["Staff Engineer"]

    # The compat SearchString that the sync persisted carries surface + filters too,
    # so the bootstrap path (in-memory absent) reconstructs the SAME state.
    persisted_string = loaded_progress.strings[0]
    assert persisted_string.surface == "structured_only"
    assert persisted_string.structured_filters["titles"] == ["Staff Engineer"]

    # Flavor 2: bootstrap-reconstruct from the persisted compat SearchString alone.
    bootstrap_active = bootstrap_experiment_state(persisted_string).active_variant
    assert bootstrap_active.surface == in_memory_active.surface == "structured_only"
    assert (
        bootstrap_active.structured_filters.to_dict()
        == in_memory_active.structured_filters.to_dict()
    )


def test_resume_flavors_agree_on_filters_for_mid_run_refined_hybrid_lane(tmp_path):
    """A mid-run refined hybrid lane must keep filters in both resume flavors.

    The lane is a hybrid that committed a keyword refinement mid-run: the producer-time
    SearchString carries the lane geography, and the committed precision variant is
    minted with empty own-filters (a keyword refinement does not author its own filters).
    Bootstrap already re-seeds those filters onto the active legacy variant. The
    in-memory checkpoint loader must do the same so resume does not depend on whether
    the checkpoint row was available.
    """
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )

    # Hybrid lane carrying its geography; commit a keyword refinement (empty own filters).
    search_string = SearchString(
        id=1,
        name="mid-run refined hybrid",
        boolean='"director" AND payments',
        acquisition_mode="linkedin_hybrid",
        structured_filters={"sidebar_filters": {"locations": ["New York City"]}},
    )
    state = bootstrap_experiment_state(search_string)
    state.begin_experiment_round(
        [
            LinkedInSearchVariant(
                variant_id="precision-1",
                parent_variant_id="root",
                root_string_id=1,
                boolean='"director" AND payments AND fintech',
                variant_kind="precision",
            )
        ]
    )
    state.activate_variant("precision-1")
    state.commit_variant("precision-1")
    state.apply_shadow(search_string)

    progress = Progress(brief_name="test", strings=[search_string], current_string_id=1, current_page=2)
    bridge.sync_progress(run_id, progress, experiment_states={1: state})

    loaded_progress = bridge.load_progress(run_id)
    persisted_string = loaded_progress.strings[0]

    # Both flavors agree on surface (a keyword refinement leaves the active surface "").
    in_memory_active = bridge.load_experiment_states(run_id, progress=loaded_progress)[1].active_variant
    bootstrap_active = bootstrap_experiment_state(persisted_string).active_variant
    assert in_memory_active.surface == bootstrap_active.surface == ""

    # Both flavors must also agree on filters: the in-memory loader re-seeds the lane
    # geography onto the active precision variant just like bootstrap does.
    assert not in_memory_active.structured_filters.is_empty()
    assert not bootstrap_active.structured_filters.is_empty()
    assert (
        in_memory_active.structured_filters.to_dict()
        == bootstrap_active.structured_filters.to_dict()
    )
    assert bootstrap_active.structured_filters.sidebar_filters["locations"] == [
        "New York City"
    ]


def test_sync_progress_roundtrips_pending_drift_and_family_metrics(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )

    search_string = SearchString(id=1, name="builders", boolean="foo")
    state = bootstrap_experiment_state(search_string)
    state.commit_variant("root")
    page = LinkedInPageInsights(
        page=1,
        result_count=1800,
        result_window="150-800",
        signal_anchors=["ML engineer at OpenAI", "Research engineer at Anthropic"],
        title_clusters=[{"label": "machine learning engineer", "count": 4}],
    )
    state.record_variant_metrics(page_num=1, result_count=1800, page_stats={"candidates": 4, "facial_yes": 2, "saves": 1}, page_insights=page)
    state.record_family_page_metrics(page_num=1, result_count=1800, page_stats={"candidates": 4, "facial_yes": 2, "saves": 1}, page_insights=page)
    state.precommit_recovery_attempts_used = 2
    state.committed_pages_reviewed = 1
    state.committed_zero_signal_streak = 0
    drift_variant = LinkedInSearchVariant(
        variant_id="drift-1",
        parent_variant_id="root",
        root_string_id=1,
        boolean="foo AND bar",
        variant_kind="precision",
    )
    state.variants["drift-1"] = drift_variant
    state.mark_pending_drift(
        variant_id="drift-1",
        parent_variant_id="root",
        summary={"decision": "refine_committed", "keyword_hypothesis": "tighten around ML engineer"},
    )
    state.activate_variant("drift-1")
    state.apply_shadow(search_string)

    progress = Progress(brief_name="test", strings=[search_string], current_string_id=1, current_page=1)
    bridge.sync_progress(run_id, progress, experiment_states={1: state})

    loaded_progress = bridge.load_progress(run_id)
    loaded_states = bridge.load_experiment_states(run_id, progress=loaded_progress)
    row = store.get_work_unit_by_source_id(run_id, kind="linkedin_string", source_unit_id="1")
    metrics = json.loads(row["metrics_json"])

    assert loaded_states[1].pending_drift_variant_id == "drift-1"
    assert loaded_states[1].pending_drift_parent_variant_id == "root"
    assert loaded_states[1].drift_attempt_count == 1
    assert loaded_states[1].precommit_recovery_attempts_used == 2
    assert loaded_states[1].committed_pages_reviewed == 1
    assert loaded_states[1].committed_zero_signal_streak == 0
    assert metrics["experiment_summary"]["family_pages_reviewed_total"] == 1
    assert metrics["experiment_summary"]["active_variant_pages_reviewed"] == 0
    assert metrics["experiment_summary"]["precommit_recovery_attempts_used"] == 2
    assert metrics["experiment_summary"]["drift_rescue_summary"]["decision"] == "refine_committed"


def test_profile_extraction_failure_becomes_failed_retryable(tmp_path):
    p = _make_pipeline(str(tmp_path))
    search_string = SearchString(id=1, name="builders", boolean="ml")
    progress = Progress(brief_name="test", strings=[search_string])
    p._runtime_run_id, _ = p._runtime_bridge.start_or_resume_run(resume=False, initial_progress=progress)

    snippet = _snippet()
    p._record_runtime_snippet(search_string, snippet)
    facial_attempt_id = p._start_runtime_stage_attempt(
        search_string=search_string,
        snippet=snippet,
        stage="facial",
    )
    facial_yes = OpusDecision(
        stage="facial",
        decision="FACIAL_YES",
        path="none",
        confidence=0.82,
        rationale="worth opening",
        candidate_name=snippet.name,
        profile_url=snippet.profile_url,
    )
    p._finish_runtime_stage_success(
        attempt_id=facial_attempt_id,
        stage="facial",
        snippet=snippet,
        decision=facial_yes,
    )

    p.browser.ensure_card_rendered = AsyncMock()
    p.browser.open_profile_by_url = AsyncMock(return_value=None)
    p.browser.simulate_profile_read = AsyncMock(return_value=None)
    p.browser.get_profile_innertext = AsyncMock(side_effect=RuntimeError("extract failed"))
    p.browser.go_back_to_results = AsyncMock(return_value=None)
    p._ensure_browser_healthy = AsyncMock()

    decision = asyncio.run(p._full_evaluate(snippet, None, search_string))

    assert decision is not None
    assert decision.decision == "JUDGMENT_FAILURE"
    candidate = p._runtime_state.get_candidate(
        source="linkedin",
        brief_id="test-project",
        identity_key=snippet.profile_url,
    )
    assert candidate["current_lifecycle_state"] == "failed_retryable"
    assert p._runtime_state.is_dedup_blocked(
        source="linkedin",
        brief_id="test-project",
        identity_key=snippet.profile_url,
    ) is False


def test_batch_blacklist_registers_candidate_before_runtime_stage_write(tmp_path):
    p = _make_pipeline(str(tmp_path))
    p.brief_obj.has_v2_schema = True
    p.brief_obj.employer_blacklist = ["Blocked Corp"]
    p._bias_monitor = None
    p._triage_tightened = False
    p._tightening_prefix = ""

    search_string = SearchString(id=1, name="batch", boolean="ml")
    progress = Progress(brief_name="test", strings=[search_string])
    p._runtime_run_id, _ = p._runtime_bridge.start_or_resume_run(
        resume=False,
        initial_progress=progress,
    )

    blacklisted = _snippet(
        name="Blair Blocked",
        current_company="Blocked Corp Consulting",
        profile_url="/talent/profile/blocked",
        result_rank=1,
    )
    eligible = _snippet(
        name="Elliot Eligible",
        current_company="Open Market Labs",
        profile_url="/talent/profile/eligible",
        result_rank=2,
    )
    p.browser.get_card_slot_count = AsyncMock(return_value=2)
    p._extract_card_snippet = AsyncMock(side_effect=[blacklisted, eligible])
    p._checkpoint_progress = MagicMock()

    batch_seen: list[str] = []

    def fake_facial_judge_batch(snippets, brief, prompt_prefix="", lane_context=None):
        batch_seen.extend(snippet.profile_url for snippet in snippets)
        return [
            OpusDecision(
                stage="facial",
                decision="FACIAL_NO",
                path="none",
                confidence=1.0,
                rationale="eligible candidate evaluated by batch facial",
                candidate_name=eligible.name,
                profile_url=eligible.profile_url,
            )
        ]

    all_candidates: list[dict] = []
    string_stats = {
        "pages": 1,
        "candidates": 0,
        "duplicates": 0,
        "facial_yes": 0,
        "facial_no": 0,
        "saves": 0,
        "rejects": 0,
    }

    with patch("shared.judger.facial_judge_batch", side_effect=fake_facial_judge_batch):
        asyncio.run(
            p._review_page_batch(
                search_string,
                1,
                386,
                MagicMock(),
                all_candidates,
                string_stats,
                progress,
            )
        )

    blacklisted_row = p._runtime_state.get_candidate(
        source="linkedin",
        brief_id="test-project",
        identity_key=blacklisted.profile_url,
    )
    eligible_row = p._runtime_state.get_candidate(
        source="linkedin",
        brief_id="test-project",
        identity_key=eligible.profile_url,
    )

    assert blacklisted_row["current_lifecycle_state"] == "facial_terminal"
    assert blacklisted_row["terminal_decision"] == "FACIAL_NO"
    assert batch_seen == [eligible.profile_url]
    assert eligible_row["terminal_decision"] == "FACIAL_NO"
    assert [candidate["name"] for candidate in all_candidates] == [
        "Blair Blocked",
        "Elliot Eligible",
    ]


def test_batch_blacklist_screens_headline_before_facial_judge(tmp_path):
    p = _make_pipeline(str(tmp_path))
    p.brief_obj.has_v2_schema = True
    p.brief_obj.employer_blacklist = ["Acme"]
    p._bias_monitor = None
    p._triage_tightened = False
    p._tightening_prefix = ""

    search_string = SearchString(id=1, name="batch", boolean="ml")
    progress = _start_pipeline_runtime(p, search_string)

    blacklisted = _snippet(
        name="Hana Headline",
        headline="Team Lead | Business & AI Operations at Acme",
        current_company="Nova Energy",
        profile_url="/talent/profile/headline-blacklist",
        result_rank=1,
    )
    eligible = _snippet(
        name="Elliot Eligible",
        headline="ML platform leader at Open Market Labs",
        current_company="Open Market Labs",
        profile_url="/talent/profile/eligible-headline-test",
        result_rank=2,
    )
    p.browser.get_card_slot_count = AsyncMock(return_value=2)
    p._extract_card_snippet = AsyncMock(side_effect=[blacklisted, eligible])
    p._checkpoint_progress = MagicMock()

    batch_seen: list[str] = []

    def fake_facial_judge_batch(snippets, brief, prompt_prefix="", lane_context=None):
        batch_seen.extend(snippet.profile_url for snippet in snippets)
        return [
            OpusDecision(
                stage="facial",
                decision="FACIAL_NO",
                path="none",
                confidence=1.0,
                rationale="eligible candidate evaluated by batch facial",
                candidate_name=eligible.name,
                profile_url=eligible.profile_url,
            )
        ]

    all_candidates: list[dict] = []
    string_stats = {
        "pages": 1,
        "candidates": 0,
        "duplicates": 0,
        "facial_yes": 0,
        "facial_no": 0,
        "saves": 0,
        "rejects": 0,
    }

    with patch("shared.judger.facial_judge_batch", side_effect=fake_facial_judge_batch):
        asyncio.run(
            p._review_page_batch(
                search_string,
                1,
                386,
                MagicMock(),
                all_candidates,
                string_stats,
                progress,
            )
        )

    blacklisted_row = p._runtime_state.get_candidate(
        source="linkedin",
        brief_id="test-project",
        identity_key=blacklisted.profile_url,
    )
    blacklisted_payload = _facial_decision_payload(p, blacklisted.profile_url)

    assert blacklisted_row["current_lifecycle_state"] == "facial_terminal"
    assert blacklisted_row["terminal_decision"] == "FACIAL_NO"
    assert blacklisted_payload["path"] == "employer_blacklist"
    assert blacklisted_payload["rationale"] == "Employer blacklist: Acme (headline)"
    assert batch_seen == [eligible.profile_url]


def test_batch_blacklist_word_boundary_does_not_match_headline_school_context(tmp_path):
    p = _make_pipeline(str(tmp_path))
    p.brief_obj.has_v2_schema = True
    p.brief_obj.employer_blacklist = ["Acme"]
    p._bias_monitor = None
    p._triage_tightened = False
    p._tightening_prefix = ""

    search_string = SearchString(id=1, name="batch", boolean="ml")
    progress = _start_pipeline_runtime(p, search_string)

    boundary = _snippet(
        name="Casey Boundary",
        headline="Acme School of Software & Design alum",
        current_company="Northwind",
        profile_url="/talent/profile/acme-school",
        result_rank=1,
    )
    p.browser.get_card_slot_count = AsyncMock(return_value=1)
    p._extract_card_snippet = AsyncMock(return_value=boundary)
    p._checkpoint_progress = MagicMock()

    batch_seen: list[str] = []

    def fake_facial_judge_batch(snippets, brief, prompt_prefix="", lane_context=None):
        batch_seen.extend(snippet.profile_url for snippet in snippets)
        return [
            OpusDecision(
                stage="facial",
                decision="FACIAL_NO",
                path="normal_eligibility",
                confidence=1.0,
                rationale="boundary case reached normal facial eligibility",
                candidate_name=boundary.name,
                profile_url=boundary.profile_url,
            )
        ]

    all_candidates: list[dict] = []
    string_stats = {
        "pages": 1,
        "candidates": 0,
        "duplicates": 0,
        "facial_yes": 0,
        "facial_no": 0,
        "saves": 0,
        "rejects": 0,
    }

    with patch("shared.judger.facial_judge_batch", side_effect=fake_facial_judge_batch):
        asyncio.run(
            p._review_page_batch(
                search_string,
                1,
                386,
                MagicMock(),
                all_candidates,
                string_stats,
                progress,
            )
        )

    boundary_row = p._runtime_state.get_candidate(
        source="linkedin",
        brief_id="test-project",
        identity_key=boundary.profile_url,
    )
    boundary_payload = _facial_decision_payload(p, boundary.profile_url)

    assert batch_seen == [boundary.profile_url]
    assert p.stats.get("blacklist_skips", 0) == 0
    assert boundary_row["current_lifecycle_state"] == "facial_terminal"
    assert boundary_row["terminal_decision"] == "FACIAL_NO"
    assert boundary_payload["path"] == "normal_eligibility"


def test_judgment_runtime_profile_is_stable_and_secret_free(monkeypatch):
    from linkedin.orchestrator import Pipeline
    from shared import config

    for name, value in {
        "CHEAP_MODEL_PROVIDER": "anthropic",
        "CHEAP_MODEL_NAME": "claude-haiku-4-5-20251001",
        "STRATEGY_MODEL_NAME": "claude-fable-5",
        "FACIAL_MODEL_NAME": "accounts/fireworks/models/glm-5p2",
        "FULL_EVAL_MODEL_NAME": "accounts/fireworks/models/glm-5p2",
        "OPUS_MODEL_NAME": "claude-opus-4-8",
        "SHADOW_STRATEGY_ENABLED": False,
        "SHADOW_FACIAL_MODEL_ENABLED": False,
        "LINKEDIN_PAGE_ALLOCATOR_MODE": "off",
        "LINKEDIN_TOTAL_PAGE_CAP": 0,
        "FIREWORKS_API_KEY": "super-secret",
    }.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setenv("LANGFUSE_DISABLE", "1")
    first = Pipeline._judgment_runtime_profile()
    second = Pipeline._judgment_runtime_profile()

    assert first == second
    assert len(first["fingerprint_sha256"]) == 64
    assert first["schema_version"] == "linkedin-judgment-runtime-v1"
    assert first["remote_observability_disabled"] is True
    assert first["fireworks_base_url"].startswith("https://")
    assert first["cheap_model_provider"] == config.CHEAP_MODEL_PROVIDER
    assert first["cheap_model"] == config.CHEAP_MODEL_NAME
    assert first["strategy_provider"] == "anthropic"
    assert first["strategy_model"] == config.STRATEGY_MODEL_NAME
    assert first["facial_provider"] == "fireworks"
    assert first["facial_model"] == config.FACIAL_MODEL_NAME
    assert first["full_eval_provider"] == "fireworks"
    assert first["full_eval_model"] == config.FULL_EVAL_MODEL_NAME
    assert first["opus_provider"] == "anthropic"
    assert first["opus_model"] == config.OPUS_MODEL_NAME
    assert first["shadow_strategy_enabled"] is False
    assert first["shadow_facial_enabled"] is False
    assert first["page_allocator_mode"] == config.LINKEDIN_PAGE_ALLOCATOR_MODE
    assert first["total_page_cap"] == config.LINKEDIN_TOTAL_PAGE_CAP
    assert first["facial_ambiguity_posture"] == ""
    assert first["facial_ternary_effective"] is False
    serialized = json.dumps(first, sort_keys=True)
    assert "super-secret" not in serialized
    assert "api_key" not in serialized.lower()


@pytest.mark.parametrize(
    ("missing_attr", "missing_name"),
    [
        ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        ("FIREWORKS_API_KEY", "FIREWORKS_API_KEY"),
    ],
)
def test_judgment_runtime_requires_keys_for_effective_model_roles(
    monkeypatch,
    missing_attr,
    missing_name,
):
    from linkedin.orchestrator import Pipeline
    from shared import config

    monkeypatch.setenv("CLORIS_SKIP_STARTUP_VALIDATION", "0")
    monkeypatch.setenv("SSL_CERT_FILE", "")
    for name, value in {
        "CHEAP_MODEL_PROVIDER": "anthropic",
        "CHEAP_MODEL_NAME": "claude-haiku-4-5-20251001",
        "STRATEGY_MODEL_NAME": "claude-fable-5",
        "FACIAL_MODEL_NAME": "accounts/fireworks/models/glm-5p2",
        "FULL_EVAL_MODEL_NAME": "accounts/fireworks/models/glm-5p2",
        "OPUS_MODEL_NAME": "claude-opus-4-8",
        "SHADOW_STRATEGY_ENABLED": False,
        "SHADOW_FACIAL_MODEL_ENABLED": False,
        "FIREWORKS_JUDGMENT_POLICY_ENABLED": False,
        "LINKEDIN_V2_FACIAL_CONTRACT": "legacy",
        "LINKEDIN_V2_FULL_CONTRACT": "legacy",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "FIREWORKS_API_KEY": "fireworks-secret",
        "OPENAI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "MINIMAX_API_KEY": "",
    }.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(config, missing_attr, "")

    with pytest.raises(RuntimeError, match=missing_name) as captured:
        Pipeline._judgment_runtime_profile()

    assert "anthropic-secret" not in str(captured.value)
    assert "fireworks-secret" not in str(captured.value)
    monkeypatch.setattr(config, missing_attr, "restored")
    Pipeline._judgment_runtime_profile()


def test_missing_effective_model_key_stops_before_browser_construction(
    monkeypatch,
    tmp_path,
):
    from linkedin.orchestrator import Pipeline
    from shared import config

    monkeypatch.setenv("CLORIS_SKIP_STARTUP_VALIDATION", "0")
    monkeypatch.setenv("SSL_CERT_FILE", "")
    monkeypatch.setattr(config, "CHEAP_MODEL_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "STRATEGY_MODEL_NAME", "claude-fable-5")
    monkeypatch.setattr(
        config,
        "FACIAL_MODEL_NAME",
        "accounts/fireworks/models/glm-5p2",
    )
    monkeypatch.setattr(
        config,
        "FULL_EVAL_MODEL_NAME",
        "accounts/fireworks/models/glm-5p2",
    )
    monkeypatch.setattr(config, "OPUS_MODEL_NAME", "claude-opus-4-8")
    monkeypatch.setattr(config, "SHADOW_STRATEGY_ENABLED", False)
    monkeypatch.setattr(config, "SHADOW_FACIAL_MODEL_ENABLED", False)
    monkeypatch.setattr(config, "FIREWORKS_JUDGMENT_POLICY_ENABLED", False)
    monkeypatch.setattr(config, "LINKEDIN_V2_FACIAL_CONTRACT", "legacy")
    monkeypatch.setattr(config, "LINKEDIN_V2_FULL_CONTRACT", "legacy")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "present")
    monkeypatch.setattr(config, "FIREWORKS_API_KEY", "")

    with patch("linkedin.orchestrator.LinkedInBrowser") as browser_type:
        with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
            Pipeline(brief_path=str(tmp_path / "brief.json"))

    browser_type.assert_not_called()


def test_judgment_runtime_profile_records_brief_owned_ternary_posture():
    from linkedin.orchestrator import Pipeline

    brief = MagicMock()
    brief._new_brief.facial_ambiguity_posture = "ternary"

    profile = Pipeline._judgment_runtime_profile(brief)

    assert profile["facial_ambiguity_posture"] == "ternary"
    assert profile["facial_ternary_effective"] is True


def test_judgment_runtime_profile_rejects_unsafe_legacy_concurrency(monkeypatch):
    from linkedin.orchestrator import Pipeline

    monkeypatch.setattr("shared.config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED", True)
    monkeypatch.setattr("shared.config.LINKEDIN_FACIAL_MAX_CONCURRENCY", 3)
    monkeypatch.setattr("shared.config.LINKEDIN_V2_FACIAL_CONTRACT", "legacy")
    monkeypatch.setattr("shared.config.FIREWORKS_JUDGMENT_POLICY_ENABLED", True)

    with pytest.raises(RuntimeError, match="requires.*FACIAL_CONTRACT=tool"):
        Pipeline._judgment_runtime_profile()


def test_judgment_runtime_profile_rejects_tool_contract_without_policy(monkeypatch):
    from linkedin.orchestrator import Pipeline

    monkeypatch.setattr("shared.config.LINKEDIN_V2_FACIAL_CONTRACT", "tool")
    monkeypatch.setattr("shared.config.FIREWORKS_JUDGMENT_POLICY_ENABLED", False)

    with pytest.raises(RuntimeError, match="tool judgment contracts require"):
        Pipeline._judgment_runtime_profile()


def test_judgment_runtime_profile_caps_facial_concurrency_at_three(monkeypatch):
    from linkedin.orchestrator import Pipeline

    monkeypatch.setattr("shared.config.LINKEDIN_FACIAL_MAX_CONCURRENCY", 4)

    with pytest.raises(RuntimeError, match="must be <= 3"):
        Pipeline._judgment_runtime_profile()


@pytest.mark.parametrize(
    ("config_name", "invalid_value", "maximum"),
    [
        ("FIREWORKS_FACIAL_MAX_ATTEMPTS", 5, 4),
        ("FIREWORKS_FULL_MAX_ATTEMPTS", 3, 2),
    ],
)
def test_judgment_runtime_profile_caps_policy_attempts(
    monkeypatch,
    config_name,
    invalid_value,
    maximum,
):
    from linkedin.orchestrator import Pipeline

    monkeypatch.setattr("shared.config.FIREWORKS_JUDGMENT_POLICY_ENABLED", True)
    monkeypatch.setattr("shared.config.FIREWORKS_JUDGMENT_STREAM_ENABLED", True)
    monkeypatch.setattr("shared.config.FIREWORKS_FACIAL_REASONING_EFFORT", "high")
    monkeypatch.setattr("shared.config.FIREWORKS_FULL_REASONING_EFFORT", "high")
    monkeypatch.setattr(
        "shared.config.FACIAL_MODEL_NAME",
        "accounts/fireworks/models/glm-5p2",
    )
    monkeypatch.setattr(
        "shared.config.FULL_EVAL_MODEL_NAME",
        "accounts/fireworks/models/glm-5p2",
    )
    monkeypatch.setattr("shared.config.FIREWORKS_FACIAL_MAX_ATTEMPTS", 4)
    Pipeline._judgment_runtime_profile()

    monkeypatch.setattr(f"shared.config.{config_name}", invalid_value)

    with pytest.raises(
        RuntimeError,
        match=f"{config_name} must be <= {maximum}",
    ):
        Pipeline._judgment_runtime_profile()


def test_single_facial_budget_abort_closes_canonical_attempt(tmp_path):
    p = _make_pipeline(str(tmp_path))
    search_string = SearchString(id=1, name="facial", boolean="ml")
    _start_pipeline_runtime(p, search_string)
    snippet = _snippet(profile_url="/talent/profile/facial-budget")
    p._record_runtime_snippet(search_string, snippet)
    p._tightening_prefix = ""
    p._in_flight_urls.add(snippet.profile_url)

    with patch(
        "linkedin.orchestrator.facial_judge",
        side_effect=ApiBudgetExhaustedError("provider credits exhausted"),
    ):
        with pytest.raises(ApiBudgetExhaustedError):
            asyncio.run(
                p._evaluate_snippet(
                    snippet,
                    search_string=search_string,
                )
            )

    assert p._runtime_state.list_orphaned_attempts(
        source="linkedin",
        brief_id="test-project",
    ) == []
    assert snippet.profile_url not in p._in_flight_urls
    with p._runtime_state.connect() as conn:
        row = conn.execute(
            "SELECT stage, status, payload_json FROM candidate_attempts "
            "WHERE stage = 'facial'"
        ).fetchone()
    payload = json.loads(row["payload_json"])
    assert row["stage"] == "facial"
    assert row["status"] == "failed"
    assert payload["logical_call_id"].startswith("judge-")
    assert payload["force_retryable"] is True


def test_serial_full_budget_abort_closes_canonical_attempt(tmp_path):
    p = _make_pipeline(str(tmp_path))
    search_string = SearchString(id=1, name="full", boolean="ml")
    _start_pipeline_runtime(p, search_string)
    snippet = _snippet(profile_url="/talent/profile/full-budget")
    p._record_runtime_snippet(search_string, snippet)
    summary = CandidateProfileSummary(
        name=snippet.name,
        profile_url=snippet.profile_url,
        headline=snippet.headline,
    )
    p._acquisition_service = MagicMock()
    p._acquisition_service.extract_profile_summary = AsyncMock(
        return_value=SimpleNamespace(profile_summary=summary)
    )
    p.browser.get_profile_status_summary = AsyncMock(return_value={})
    p._profile_probe = MagicMock()
    p._profile_probe.evaluate.return_value = "probe"
    p._in_flight_urls.add(snippet.profile_url)

    with patch(
        "linkedin.orchestrator.full_judge",
        side_effect=ApiBudgetExhaustedError("provider credits exhausted"),
    ):
        with pytest.raises(ApiBudgetExhaustedError):
            asyncio.run(p._full_evaluate(snippet, None, search_string))

    assert p._runtime_state.list_orphaned_attempts(
        source="linkedin",
        brief_id="test-project",
    ) == []
    assert snippet.profile_url not in p._in_flight_urls
    with p._runtime_state.connect() as conn:
        row = conn.execute(
            "SELECT stage, status, payload_json FROM candidate_attempts "
            "WHERE stage = 'full'"
        ).fetchone()
    payload = json.loads(row["payload_json"])
    assert row["stage"] == "full"
    assert row["status"] == "failed"
    assert payload["logical_call_id"].startswith("judge-")
    assert payload["force_retryable"] is True


def test_batch_partial_attempt_start_failure_closes_started_attempts(tmp_path):
    p = _make_pipeline(str(tmp_path))
    p.brief_obj.has_v2_schema = True
    p._bias_monitor = None
    p._triage_tightened = False
    p._tightening_prefix = ""

    search_string = SearchString(id=1, name="batch-start-failure", boolean="ml")
    progress = _start_pipeline_runtime(p, search_string)
    snippets = [
        _snippet(
            name=f"Candidate {index}",
            profile_url=f"/talent/profile/start-failure-{index}",
            result_rank=index,
        )
        for index in range(1, 4)
    ]
    p.browser.get_card_slot_count = AsyncMock(return_value=len(snippets))
    p._extract_card_snippet = AsyncMock(side_effect=snippets)

    real_start = p._start_runtime_stage_attempt
    start_count = 0

    def fail_third_start(**kwargs):
        nonlocal start_count
        start_count += 1
        if start_count == 3:
            raise RuntimeError("synthetic third attempt start failure")
        return real_start(**kwargs)

    p._start_runtime_stage_attempt = MagicMock(side_effect=fail_third_start)
    string_stats = {
        "pages": 1,
        "candidates": 0,
        "duplicates": 0,
        "facial_yes": 0,
        "facial_no": 0,
        "saves": 0,
        "rejects": 0,
    }

    with pytest.raises(RuntimeError, match="third attempt start failure"):
        asyncio.run(
            p._review_page_batch(
                search_string,
                1,
                len(snippets),
                MagicMock(),
                [],
                string_stats,
                progress,
            )
        )

    assert p._runtime_state.list_orphaned_attempts(
        source="linkedin",
        brief_id="test-project",
    ) == []
    assert p._in_flight_urls.isdisjoint(
        {snippet.profile_url for snippet in snippets}
    )
    with p._runtime_state.connect() as conn:
        rows = conn.execute(
            "SELECT stage, status, payload_json FROM candidate_attempts "
            "WHERE stage = 'facial' "
            "ORDER BY id"
        ).fetchall()
    assert [(row["stage"], row["status"]) for row in rows] == [
        ("facial", "failed"),
        ("facial", "failed"),
    ]
    for row in rows:
        payload = json.loads(row["payload_json"])
        assert payload["facial_dispatch_failed"] is True
        assert payload["force_retryable"] is True


def test_batch_postdispatch_normalization_failure_aborts_every_open_attempt(
    tmp_path,
    monkeypatch,
):
    p = _make_pipeline(str(tmp_path))
    p.brief_obj.has_v2_schema = True
    p._bias_monitor = None
    p._triage_tightened = False
    p._tightening_prefix = ""

    search_string = SearchString(id=1, name="normalize-failure", boolean="ml")
    progress = _start_pipeline_runtime(p, search_string)
    snippets = [
        _snippet(
            name=f"Candidate {index}",
            profile_url=f"/talent/profile/normalize-failure-{index}",
            result_rank=index,
        )
        for index in range(1, 4)
    ]
    p.browser.get_card_slot_count = AsyncMock(return_value=len(snippets))
    p._extract_card_snippet = AsyncMock(side_effect=snippets)
    p._checkpoint_progress = MagicMock()
    p._normalize_facial_decision_for_persistence = MagicMock(
        side_effect=RuntimeError("synthetic normalization failure")
    )
    monkeypatch.setattr("shared.config.GLANCE_MIN_SNIPPETS", 100)

    decisions = [
        OpusDecision(
            stage="facial",
            decision="FACIAL_NO",
            path="normal_eligibility",
            confidence=1.0,
            rationale="not a fit",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        for snippet in snippets
    ]
    string_stats = {
        "pages": 1,
        "candidates": 0,
        "duplicates": 0,
        "facial_yes": 0,
        "facial_no": 0,
        "saves": 0,
        "rejects": 0,
    }

    with patch("shared.judger.facial_judge_batch", return_value=decisions):
        with pytest.raises(RuntimeError, match="normalization failure"):
            asyncio.run(
                p._review_page_batch(
                    search_string,
                    1,
                    len(snippets),
                    MagicMock(),
                    [],
                    string_stats,
                    progress,
                )
            )

    assert p._runtime_state.list_orphaned_attempts(
        source="linkedin",
        brief_id="test-project",
    ) == []
    assert p._in_flight_urls.isdisjoint(
        {snippet.profile_url for snippet in snippets}
    )
    with p._runtime_state.connect() as conn:
        rows = conn.execute(
            "SELECT c.identity_key, ca.status, ca.payload_json "
            "FROM candidate_attempts ca "
            "JOIN candidates c ON c.id = ca.candidate_id "
            "WHERE ca.stage = 'facial' ORDER BY ca.id"
        ).fetchall()
    assert [row["identity_key"] for row in rows] == [
        snippet.profile_url for snippet in snippets
    ]
    assert [row["status"] for row in rows] == ["failed", "failed", "failed"]
    for row in rows:
        payload = json.loads(row["payload_json"])
        assert payload["facial_postdispatch_failed"] is True
        assert payload["force_retryable"] is True
    assert [
        p._runtime_state.get_candidate(
            source="linkedin",
            brief_id="test-project",
            identity_key=snippet.profile_url,
        )["current_lifecycle_state"]
        for snippet in snippets
    ] == ["failed_retryable", "failed_retryable", "failed_retryable"]


def test_batch_midloop_bias_failure_preserves_closed_rows_and_aborts_remainder(
    tmp_path,
    monkeypatch,
):
    class SyntheticPostdispatchAbort(BaseException):
        pass

    p = _make_pipeline(str(tmp_path))
    p.brief_obj.has_v2_schema = True
    p._bias_monitor = MagicMock()
    p._bias_monitor.record_decision.side_effect = [
        None,
        SyntheticPostdispatchAbort("synthetic bias bookkeeping failure"),
    ]
    p._triage_tightened = False
    p._tightening_prefix = ""

    search_string = SearchString(id=1, name="midloop-failure", boolean="ml")
    progress = _start_pipeline_runtime(p, search_string)
    snippets = [
        _snippet(
            name=f"Candidate {index}",
            profile_url=f"/talent/profile/midloop-failure-{index}",
            result_rank=index,
        )
        for index in range(1, 4)
    ]
    p.browser.get_card_slot_count = AsyncMock(return_value=len(snippets))
    p._extract_card_snippet = AsyncMock(side_effect=snippets)
    p._checkpoint_progress = MagicMock()
    monkeypatch.setattr("shared.config.GLANCE_MIN_SNIPPETS", 100)

    decisions = [
        OpusDecision(
            stage="facial",
            decision="FACIAL_NO",
            path="normal_eligibility",
            confidence=1.0,
            rationale="not a fit",
            candidate_name=snippets[0].name,
            profile_url=snippets[0].profile_url,
        ),
        OpusDecision(
            stage="facial",
            decision="PARSE_FAILURE",
            path="error",
            confidence=0.0,
            rationale="synthetic malformed response",
            candidate_name=snippets[1].name,
            profile_url=snippets[1].profile_url,
        ),
        OpusDecision(
            stage="facial",
            decision="FACIAL_NO",
            path="normal_eligibility",
            confidence=1.0,
            rationale="not a fit",
            candidate_name=snippets[2].name,
            profile_url=snippets[2].profile_url,
        ),
    ]
    string_stats = {
        "pages": 1,
        "candidates": 0,
        "duplicates": 0,
        "facial_yes": 0,
        "facial_no": 0,
        "saves": 0,
        "rejects": 0,
    }

    with patch("shared.judger.facial_judge_batch", return_value=decisions):
        with pytest.raises(SyntheticPostdispatchAbort, match="bias bookkeeping failure"):
            asyncio.run(
                p._review_page_batch(
                    search_string,
                    1,
                    len(snippets),
                    MagicMock(),
                    [],
                    string_stats,
                    progress,
                )
            )

    assert p._runtime_state.list_orphaned_attempts(
        source="linkedin",
        brief_id="test-project",
    ) == []
    assert p._in_flight_urls.isdisjoint(
        {snippet.profile_url for snippet in snippets}
    )
    with p._runtime_state.connect() as conn:
        rows = conn.execute(
            "SELECT c.identity_key, ca.status, ca.payload_json "
            "FROM candidate_attempts ca "
            "JOIN candidates c ON c.id = ca.candidate_id "
            "WHERE ca.stage = 'facial' ORDER BY ca.id"
        ).fetchall()
    assert [row["identity_key"] for row in rows] == [
        snippet.profile_url for snippet in snippets
    ]
    assert [row["status"] for row in rows] == ["succeeded", "failed", "failed"]
    payloads = [json.loads(row["payload_json"]) for row in rows]
    assert "facial_postdispatch_failed" not in payloads[0]
    assert payloads[1]["facial_decision"]["decision"] == "PARSE_FAILURE"
    assert "facial_postdispatch_failed" not in payloads[1]
    assert payloads[2]["facial_postdispatch_failed"] is True
    assert payloads[2]["force_retryable"] is True
    assert [
        p._runtime_state.get_candidate(
            source="linkedin",
            brief_id="test-project",
            identity_key=snippet.profile_url,
        )["current_lifecycle_state"]
        for snippet in snippets
    ] == ["facial_terminal", "failed_retryable", "failed_retryable"]


def test_concurrent_page_facial_25_merges_in_order_without_orphans(
    tmp_path,
    monkeypatch,
):
    p = _make_pipeline(str(tmp_path))
    p.brief_obj.has_v2_schema = True
    p._bias_monitor = None
    p._triage_tightened = False
    p._tightening_prefix = ""

    search_string = SearchString(id=1, name="concurrent-page", boolean="ml")
    progress = _start_pipeline_runtime(p, search_string)
    snippets = [
        _snippet(
            name=f"Candidate {index:02d}",
            profile_url=f"/talent/profile/concurrent-{index:02d}",
            result_rank=index,
        )
        for index in range(1, 26)
    ]
    p.browser.get_card_slot_count = AsyncMock(return_value=len(snippets))
    p._extract_card_snippet = AsyncMock(side_effect=snippets)

    monkeypatch.setattr("shared.config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED", True)
    monkeypatch.setattr("shared.config.LINKEDIN_FACIAL_MAX_CONCURRENCY", 3)
    monkeypatch.setattr("shared.config.LINKEDIN_FACIAL_TARGET_BATCH_SIZE", 8)
    monkeypatch.setattr("shared.config.LINKEDIN_V2_FACIAL_CONTRACT", "tool")
    monkeypatch.setattr("shared.config.FIREWORKS_JUDGMENT_POLICY_ENABLED", True)
    monkeypatch.setattr("shared.config.FIREWORKS_JUDGMENT_STREAM_ENABLED", True)
    monkeypatch.setattr(
        "shared.config.FIREWORKS_FACIAL_REASONING_EFFORT",
        "high",
    )
    monkeypatch.setattr(
        "shared.config.FIREWORKS_FULL_REASONING_EFFORT",
        "high",
    )
    monkeypatch.setattr(
        "shared.config.FACIAL_MODEL_NAME",
        "accounts/fireworks/models/glm-5p2",
    )
    monkeypatch.setattr(
        "shared.config.FULL_EVAL_MODEL_NAME",
        "accounts/fireworks/models/glm-5p2",
    )
    monkeypatch.setattr("shared.config.GLANCE_MIN_SNIPPETS", 100)

    barrier = threading.Barrier(3)
    worker_thread_ids: set[int] = set()
    observed_batches: list[list[str]] = []
    observation_lock = threading.Lock()

    def fake_facial_judge_batch(
        batch_snippets,
        brief,
        prompt_prefix="",
        lane_context=None,
    ):
        del brief, prompt_prefix
        with observation_lock:
            worker_thread_ids.add(threading.get_ident())
            observed_batches.append(
                [snippet.profile_url for snippet in batch_snippets]
            )
        barrier.wait(timeout=2)
        return [
            OpusDecision(
                stage="facial",
                decision="FACIAL_NO",
                path="normal_eligibility",
                confidence=1.0,
                rationale=f"batch {lane_context['batch_index']}",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            )
            for snippet in batch_snippets
        ]

    all_candidates: list[dict] = []
    string_stats = {
        "pages": 1,
        "candidates": 0,
        "duplicates": 0,
        "facial_yes": 0,
        "facial_no": 0,
        "saves": 0,
        "rejects": 0,
    }
    main_thread_id = threading.get_ident()

    with patch(
        "shared.judger.facial_judge_batch",
        side_effect=fake_facial_judge_batch,
    ):
        asyncio.run(
            p._review_page_batch(
                search_string,
                1,
                len(snippets),
                MagicMock(),
                all_candidates,
                string_stats,
                progress,
            )
        )

    assert sorted(len(batch) for batch in observed_batches) == [8, 8, 9]
    assert len(worker_thread_ids) == 3
    assert main_thread_id not in worker_thread_ids
    assert [candidate["name"] for candidate in all_candidates] == [
        snippet.name for snippet in snippets
    ]
    assert p._runtime_state.list_orphaned_attempts(
        source="linkedin",
        brief_id="test-project",
    ) == []
    monkeypatch.setattr("shared.config.LINKEDIN_TOTAL_PAGE_CAP", 1)
    p._sync_bounded_page_stats_for_checkpoint(search_string, string_stats)
    progress.current_page = 1
    p._checkpoint_progress(progress, search_string=search_string, page_num=1)
    with pytest.raises(OperatorStopRequested, match="total_page_cap_reached"):
        p._honor_total_page_cap_at_checkpoint(
            search_string=search_string,
            page_num=1,
            progress=progress,
        )
    with p._runtime_state.connect() as conn:
        rows = conn.execute(
            "SELECT status, payload_json FROM candidate_attempts "
            "WHERE stage = 'facial' ORDER BY id"
        ).fetchall()
        timing_event = conn.execute(
            "SELECT payload_json FROM events "
            "WHERE event_type = 'facial_page_judgment_timing' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        cap_event = conn.execute(
            "SELECT payload_json FROM events "
            "WHERE event_type = 'total_page_cap_reached' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert len(rows) == 25
    assert {row["status"] for row in rows} == {"succeeded"}
    payloads = [json.loads(row["payload_json"]) for row in rows]
    assert len({payload["logical_call_id"] for payload in payloads}) == 3
    assert sorted({payload["batch_size"] for payload in payloads}) == [8, 9]
    assert string_stats["facial_no"] == 25
    assert timing_event is not None
    timing = json.loads(timing_event["payload_json"])
    assert timing["candidate_count"] == 25
    assert timing["batch_count"] == 3
    assert timing["batch_sizes"] == [8, 8, 9]
    assert timing["max_concurrency"] == 3
    assert timing["elapsed_ms"] >= 0
    assert cap_event is not None
    assert json.loads(cap_event["payload_json"])["facial_no"] == 25
    unit = p._runtime_state.get_work_unit_by_source_id(
        p._runtime_run_id,
        kind="linkedin_string",
        source_unit_id="1",
    )
    unit_metrics = json.loads(unit["metrics_json"])
    assert unit_metrics["profiles_processed"] == 25
    assert unit_metrics["facial_no"] == 25


def test_sequential_blacklist_still_screens_current_company_before_facial_judge(tmp_path):
    p = _make_pipeline(str(tmp_path))
    p.brief_obj.has_v2_schema = False
    p.brief_obj.employer_blacklist = ["Acme"]
    p._bias_monitor = None
    p._triage_tightened = False
    p._tightening_prefix = ""

    search_string = SearchString(id=1, name="sequential", boolean="ml")
    _start_pipeline_runtime(p, search_string)

    snippet = _snippet(
        name="Connie Current",
        headline="AI operations leader",
        current_company="Acme",
        profile_url="/talent/profile/current-company-blacklist",
        result_rank=1,
    )

    with patch("linkedin.orchestrator.facial_judge") as facial_judge_stub:
        decision = asyncio.run(
            p._evaluate_snippet(snippet, search_string=search_string)
        )

    payload = _facial_decision_payload(p, snippet.profile_url)

    assert decision is not None
    assert decision.decision == "FACIAL_NO"
    assert decision.path == "employer_blacklist"
    assert decision.rationale == "Employer blacklist: Acme (current_company)"
    facial_judge_stub.assert_not_called()
    assert payload["path"] == "employer_blacklist"


def test_restart_string_rolls_back_all_canonical_changes_on_failure(tmp_path, monkeypatch):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    progress = Progress(
        brief_name="test",
        strings=[SearchString(id=1, name="one", boolean="a", status="done", saves=["Ada"])],
        current_string_id=1,
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="resume",
        resume_state={"brief_name": "test"},
    )
    bridge.sync_progress(run_id, progress)

    url = "/talent/profile/ada"
    work_unit_id = store.get_work_unit_id(run_id, kind="linkedin_string", source_unit_id="1")
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
        display_name="Ada",
        profile_url=url,
        payload={"source_string_id": 1},
    )
    store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
        stage="full",
        work_unit_id=work_unit_id,
        payload={"source_string_id": 1},
        source_cursor={"source_string_id": 1},
    )
    work_unit_before = store.get_work_unit_by_source_id(
        run_id, kind="linkedin_string", source_unit_id="1"
    )

    def crash_mid_restart(**_kwargs):
        raise RuntimeError("synthetic restart failure")

    monkeypatch.setattr(store, "invalidate_candidate_side_effects", crash_mid_restart)

    with pytest.raises(RuntimeError, match="synthetic restart failure"):
        bridge.restart_string(run_id=run_id, progress=progress, string_id=1)

    with store.connect() as conn:
        attempts_after = conn.execute(
            "SELECT COUNT(*) FROM candidate_attempts WHERE work_unit_id = ?",
            (work_unit_id,),
        ).fetchone()[0]
    assert attempts_after == 1
    assert store.get_work_unit_by_source_id(
        run_id, kind="linkedin_string", source_unit_id="1"
    ) == work_unit_before


def test_restart_string_clears_only_targeted_runtime_state(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    progress = Progress(
        brief_name="test",
        strings=[
            SearchString(id=1, name="one", boolean="a", status="done", saves=["Ada"]),
            SearchString(id=2, name="two", boolean="b", status="done", saves=["Grace"]),
        ],
        current_string_id=1,
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="resume",
        resume_state={"brief_name": "test"},
    )
    bridge.sync_progress(run_id, progress)

    for string_id, name, url in (
        (1, "Ada", "/talent/profile/ada"),
        (2, "Grace", "/talent/profile/grace"),
    ):
        store.record_candidate_discovery(
            run_id=run_id,
            work_unit_id=store.get_work_unit_id(run_id, kind="linkedin_string", source_unit_id=str(string_id)),
            source="linkedin",
            brief_id="test-project",
            identity_key=url,
            display_name=name,
            profile_url=url,
            payload={"source_string_id": string_id},
        )
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id="test-project",
            identity_key=url,
            new_state="snippet_extracted",
        )
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id="test-project",
            identity_key=url,
            new_state="facial_started",
        )
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id="test-project",
            identity_key=url,
            new_state="facial_terminal",
            terminal_decision="FACIAL_YES",
            terminal_payload={"source_string_id": string_id},
        )
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id="test-project",
            identity_key=url,
            new_state="full_started",
        )
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id="test-project",
            identity_key=url,
            new_state="full_terminal",
            terminal_decision="SAVE",
            terminal_payload={"source_string_id": string_id, "timestamp": "2026-04-07T00:00:00+00:00"},
        )
        attempt_id = store.start_attempt(
            run_id=run_id,
            source="linkedin",
            brief_id="test-project",
            identity_key=url,
            stage="full",
            work_unit_id=store.get_work_unit_id(run_id, kind="linkedin_string", source_unit_id=str(string_id)),
            payload={"source_string_id": string_id, "full_decision": {"decision": "SAVE"}},
            source_cursor={"source_string_id": string_id},
            display_name=name,
            profile_url=url,
        )
        store.finish_attempt_success(
            attempt_id=attempt_id,
            new_state="full_terminal",
            terminal_decision="SAVE",
            payload={"source_string_id": string_id, "full_decision": {"decision": "SAVE"}},
            run_id=run_id,
        )

    bridge.restart_string(run_id=run_id, progress=progress, string_id=1)

    assert store.is_dedup_blocked(
        source="linkedin",
        brief_id="test-project",
        identity_key="/talent/profile/ada",
    ) is False
    assert store.is_dedup_blocked(
        source="linkedin",
        brief_id="test-project",
        identity_key="/talent/profile/grace",
    ) is True
    restarted = progress.strings[0]
    assert restarted.status == "queued"
    assert restarted.pages_reviewed == 1
    assert restarted.saves == []
    restarted_unit = store.get_work_unit_by_source_id(
        run_id, kind="linkedin_string", source_unit_id="1"
    )
    assert restarted_unit["status"] == "queued"
    assert json.loads(restarted_unit["payload_json"])["saves"] == []
    assert restarted_unit["result_count"] == 0
    assert restarted_unit["saves_count"] == 0
    history = read_jsonl(tmp_path / "candidate_history-test-project.jsonl")
    assert [row["profile_url"] for row in history] == ["/talent/profile/grace"]


def test_restart_string_does_not_leak_attempt_payload_into_work_unit(tmp_path, monkeypatch):
    """P10 actuate #5 regression.

    restart_string used to reuse the loop variable name ``payload`` for both
    the work-unit payload (read before the candidate_attempts loop) and each
    attempt row's own payload inside the loop. After the loop, ``payload``
    held whichever attempt was iterated LAST, so the subsequent
    ``payload.update({...target.to_dict(), "status": "queued", ...})``
    mutated the clobbered dict and durably wrote it via ``upsert_work_unit``
    in its own transaction — leaking the attempt's stray keys (e.g.
    ``full_decision``) into the restarted work unit.

    ``restart_string`` unconditionally calls ``self.sync_progress(...)``
    right after that write, which re-derives a clean payload from the
    in-memory ``SearchString`` and overwrites the row again — masking the
    bug from any test that only inspects the FINAL persisted state. The
    corrupted intermediate write is still durably committed the instant it
    happens, though: a crash/interrupt between the two writes (process
    kill, OOM, machine crash) leaves the leaked payload as the on-disk
    truth. This test freezes that window by no-op'ing the trailing
    ``sync_progress`` call so the assertion targets the actually-buggy write.
    """
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    progress = Progress(
        brief_name="test",
        strings=[SearchString(id=1, name="one", boolean="a", status="done", saves=["Ada"])],
        current_string_id=1,
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="resume",
        resume_state={"brief_name": "test"},
    )
    bridge.sync_progress(run_id, progress)

    url = "/talent/profile/ada"
    work_unit_id = store.get_work_unit_id(run_id, kind="linkedin_string", source_unit_id="1")
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
        display_name="Ada",
        profile_url=url,
        payload={"source_string_id": 1},
    )
    for state in ("snippet_extracted", "facial_started"):
        store.set_candidate_state(
            run_id=run_id, source="linkedin", brief_id="test-project",
            identity_key=url, new_state=state,
        )
    store.set_candidate_state(
        run_id=run_id, source="linkedin", brief_id="test-project",
        identity_key=url, new_state="facial_terminal",
        terminal_decision="FACIAL_YES", terminal_payload={"source_string_id": 1},
    )
    store.set_candidate_state(
        run_id=run_id, source="linkedin", brief_id="test-project",
        identity_key=url, new_state="full_started",
    )
    store.set_candidate_state(
        run_id=run_id, source="linkedin", brief_id="test-project",
        identity_key=url, new_state="full_terminal",
        terminal_decision="SAVE",
        terminal_payload={"source_string_id": 1, "timestamp": "2026-04-07T00:00:00+00:00"},
    )
    # The attempt's payload carries keys that must NEVER show up on the
    # restarted work unit's payload — a sentinel proving no cross-contamination.
    attempt_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
        stage="full",
        work_unit_id=work_unit_id,
        payload={
            "source_string_id": 1,
            "full_decision": {"decision": "SAVE"},
            "sentinel_attempt_only_key": "leaked-if-clobbered",
        },
        source_cursor={"source_string_id": 1},
        display_name="Ada",
        profile_url=url,
    )
    store.finish_attempt_success(
        attempt_id=attempt_id,
        new_state="full_terminal",
        terminal_decision="SAVE",
        payload={
            "source_string_id": 1,
            "full_decision": {"decision": "SAVE"},
            "sentinel_attempt_only_key": "leaked-if-clobbered",
        },
        run_id=run_id,
    )

    # Freeze the crash window: restart_string's own upsert_work_unit write is
    # the one under test; the trailing sync_progress resync (which would mask
    # the bug by re-deriving a clean payload) is disabled here.
    monkeypatch.setattr(bridge, "sync_progress", lambda *a, **kw: None)

    bridge.restart_string(run_id=run_id, progress=progress, string_id=1)

    restarted_unit = store.get_work_unit_by_source_id(
        run_id, kind="linkedin_string", source_unit_id="1"
    )
    restarted_payload = json.loads(restarted_unit["payload_json"])
    assert "sentinel_attempt_only_key" not in restarted_payload
    assert "full_decision" not in restarted_payload
    assert restarted_payload["status"] == "queued"
    assert restarted_payload["id"] == 1


def test_restart_string_resets_variant_execution_state(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )

    search_string = SearchString(id=1, name="builders", boolean="foo", status="done")
    state = bootstrap_experiment_state(search_string)
    state.begin_experiment_round(
        [
            LinkedInSearchVariant(
                variant_id="precision-1",
                parent_variant_id="root",
                root_string_id=1,
                boolean="foo AND bar",
                variant_kind="precision",
            )
        ]
    )
    state.activate_variant("precision-1")
    state.commit_variant("precision-1")
    state.apply_shadow(search_string)
    progress = Progress(brief_name="test", strings=[search_string], current_string_id=1, current_page=2)
    bridge.sync_progress(run_id, progress, experiment_states={1: state})

    bridge.restart_string(run_id=run_id, progress=progress, string_id=1)
    reloaded_progress = bridge.load_progress(run_id)
    reloaded_states = bridge.load_experiment_states(run_id, progress=reloaded_progress)

    assert reloaded_progress.strings[0].boolean == "foo"
    assert reloaded_progress.strings[0].refinement_stack == []
    assert reloaded_states[1].active_variant_id == "root"
    assert reloaded_states[1].committed_variant_id is None


def test_restart_string_clears_pending_drift_state(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )
    search_string = SearchString(id=1, name="builders", boolean="foo", status="in_progress")
    state = bootstrap_experiment_state(search_string)
    state.commit_variant("root")
    state.variants["drift-1"] = LinkedInSearchVariant(
        variant_id="drift-1",
        parent_variant_id="root",
        root_string_id=1,
        boolean="foo AND bar",
        variant_kind="precision",
    )
    state.mark_pending_drift(
        variant_id="drift-1",
        parent_variant_id="root",
        summary={"decision": "refine_committed"},
    )
    state.activate_variant("drift-1")
    state.apply_shadow(search_string)
    progress = Progress(brief_name="test", strings=[search_string], current_string_id=1, current_page=1)
    bridge.sync_progress(run_id, progress, experiment_states={1: state})

    bridge.restart_string(run_id=run_id, progress=progress, string_id=1)
    reloaded_progress = bridge.load_progress(run_id)
    reloaded_states = bridge.load_experiment_states(run_id, progress=reloaded_progress)

    assert reloaded_progress.strings[0].boolean == "foo"
    assert reloaded_states[1].pending_drift_variant_id is None
    assert reloaded_states[1].pending_drift_parent_variant_id is None
    assert reloaded_states[1].drift_attempt_count == 0


def test_sync_progress_delete_missing_work_units_when_progress_subset(tmp_path):
    """Subset progress.strings removes other linkedin_string work_units for the same run_id."""
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )
    two = Progress(
        brief_name="test",
        strings=[
            SearchString(id=1, name="one", boolean="a", status="queued"),
            SearchString(id=2, name="two", boolean="b", status="queued"),
        ],
    )
    bridge.sync_progress(run_id, two)
    rows = store.list_work_units(run_id, kind="linkedin_string")
    assert {row["source_unit_id"] for row in rows} == {"1", "2"}

    one_only = Progress(
        brief_name="test",
        strings=[SearchString(id=1, name="one", boolean="a", status="in_progress")],
        current_string_id=1,
    )
    bridge.sync_progress(run_id, one_only)
    rows_after = store.list_work_units(run_id, kind="linkedin_string")
    assert len(rows_after) == 1
    assert rows_after[0]["source_unit_id"] == "1"


def test_start_or_resume_run_reconciles_open_attempt_and_pending_side_effect(tmp_path):
    """Bridge entry reconciles orphaned LinkedIn attempts and pending side effects."""
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )
    progress = Progress(
        brief_name="test",
        strings=[SearchString(id=1, name="s", boolean="x", status="in_progress")],
        current_string_id=1,
    )
    bridge.sync_progress(run_id, progress)
    work_unit_id = store.get_work_unit_id(run_id, kind="linkedin_string", source_unit_id="1")
    assert work_unit_id is not None

    url = "/talent/profile/reconcile-me"
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
        display_name="Test",
        profile_url=url,
        payload={},
    )
    attempt_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
        stage="facial",
        work_unit_id=work_unit_id,
        payload={},
        source_cursor={},
        display_name="Test",
        profile_url=url,
    )
    started = store.begin_candidate_side_effect(
        run_id=run_id,
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
        attempt_id=None,
        effect_type="linkedin_save",
        idempotency_key="save-1",
        payload={"search_string_id": 1},
    )
    side_effect_id = int(started["side_effect"]["id"])

    new_run_id, _progress = bridge.start_or_resume_run(resume=True)
    assert new_run_id != run_id

    with store.connect() as conn:
        attempt = conn.execute(
            "SELECT status, failure_kind FROM candidate_attempts WHERE id = ?",
            (attempt_id,),
        ).fetchone()
        assert attempt["status"] == "reconciled"
        assert attempt["failure_kind"] == "orphaned_attempt"

        # P1.1: crash-interrupted pending side effects reconcile to
        # 'interrupted' (retryable without consuming an attempt), not
        # 'failed' (which would burn one of the three attempts).
        side_effect = conn.execute(
            "SELECT status FROM side_effects WHERE id = ?",
            (side_effect_id,),
        ).fetchone()
        assert side_effect["status"] == "interrupted"


def test_load_experiment_states_bootstraps_from_payload_when_progress_lookup_misses(tmp_path):
    """When progress lookup misses and checkpoint state is absent, load from payload and bootstrap."""
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )
    progress = Progress(
        brief_name="test",
        strings=[SearchString(id=1, name="builders", boolean="foo", status="in_progress")],
        current_string_id=1,
        current_page=1,
    )
    bridge.sync_progress(run_id, progress)

    with store.connect() as conn:
        conn.execute(
            """
            UPDATE work_units
            SET checkpoint_json = '{}'
            WHERE run_id = ? AND kind = ? AND source_unit_id = ?
            """,
            (run_id, "linkedin_string", "1"),
        )

    loaded_states = bridge.load_experiment_states(
        run_id,
        progress=Progress(
            brief_name="test",
            strings=[SearchString(id=99, name="other", boolean="bar")],
        ),
    )

    assert list(loaded_states) == [1]
    state = loaded_states[1]
    assert state.intent.root_boolean == "foo"
    assert state.active_variant_id == "root"
    assert state.committed_variant_id is None
    assert state.active_variant.boolean == "foo"


def test_run_full_resume_uses_db_page_cursor_instead_of_stale_progress_file(tmp_path):
    """resume=True should hand _process_string the DB-backed page cursor and pages_reviewed."""
    pipeline = _make_pipeline(str(tmp_path))
    initial_progress = Progress(
        brief_name="test",
        strings=[
            SearchString(
                id=7,
                name="resume me",
                boolean="foo",
                status="in_progress",
                pages_reviewed=3,
            )
        ],
        current_string_id=7,
        current_page=3,
    )
    canonical_state = bootstrap_experiment_state(initial_progress.strings[0])
    canonical_state.active_variant.allocator_page_cursor = 4
    pipeline._runtime_run_id, _ = pipeline._runtime_bridge.start_or_resume_run(
        resume=False,
        initial_progress=initial_progress,
        experiment_states={7: canonical_state},
    )

    stale_state = bootstrap_experiment_state(initial_progress.strings[0])
    stale_state.active_variant.allocator_page_cursor = 99
    pipeline._experiment_states = {7: stale_state}

    stale_progress = {
        "brief_name": "test",
        "current_string_id": 999,
        "current_page": 99,
        "strings": [
            {
                "id": 7,
                "name": "stale",
                "boolean": "stale",
                "status": "queued",
                "pages_reviewed": 99,
            }
        ],
    }
    (tmp_path / "progress.json").write_text(json.dumps(stale_progress))

    pipeline.browser.connect = AsyncMock()
    pipeline.browser.disconnect = AsyncMock()
    pipeline.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
    pipeline._print_session_summary = MagicMock()
    pipeline._print_summary = MagicMock()
    pipeline._generate_run_report = MagicMock()
    pipeline._finalize_run_snapshot = MagicMock(
        return_value=tmp_path / "frozen-run"
    )
    pipeline._enrich_run_snapshot = MagicMock()
    pipeline._run_health_summary = MagicMock(
        return_value={"status": "ok", "green_but_useless": False}
    )
    pipeline._session_expired = MagicMock()

    observed: dict[str, int] = {}

    async def fake_process(search_string, progress):
        observed["string_id"] = search_string.id
        observed["pages_reviewed"] = search_string.pages_reviewed
        observed["current_string_id"] = progress.current_string_id
        observed["current_page"] = progress.current_page
        observed["allocator_page_cursor"] = pipeline._experiment_states[
            search_string.id
        ].active_variant.allocator_page_cursor

    pipeline._process_string = fake_process

    asyncio.run(pipeline.run_full(resume=True))

    assert observed == {
        "string_id": 7,
        "pages_reviewed": 3,
        "current_string_id": 7,
        "current_page": 3,
        "allocator_page_cursor": 4,
    }




def test_resume_first_sync_preserves_complete_canonical_non_root_state(tmp_path):
    pipeline = _make_pipeline(str(tmp_path))
    search_string = SearchString(
        id=17,
        name="canonical",
        boolean="foo",
        status="in_progress",
        pages_reviewed=3,
    )
    progress = Progress(
        brief_name="test",
        strings=[search_string],
        current_string_id=search_string.id,
        current_page=3,
    )
    canonical = bootstrap_experiment_state(search_string)
    canonical.lane_id = "lane-sentinel"
    canonical.experiment_round = 4
    canonical.commit_variant("root")
    canonical.set_active_allocator_page_cursor(4)
    drift = LinkedInSearchVariant(
        variant_id="drift-17-1",
        parent_variant_id="root",
        root_string_id=search_string.id,
        boolean="foo NOT product",
        variant_kind="precision",
        experiment_round=5,
    )
    canonical.variants[drift.variant_id] = drift
    canonical.mark_pending_drift(
        variant_id=drift.variant_id,
        parent_variant_id="root",
        summary={"decision": "refine_committed", "sentinel": 17},
    )
    canonical.activate_variant(drift.variant_id)
    canonical.set_active_allocator_page_cursor(2)
    expected = canonical.to_dict()

    pipeline._runtime_bridge.start_or_resume_run(
        resume=False,
        initial_progress=progress,
        experiment_states={search_string.id: canonical},
    )

    stale = bootstrap_experiment_state(search_string)
    stale.set_active_allocator_page_cursor(99)
    pipeline._experiment_states = {search_string.id: stale}

    resumed = pipeline._load_or_create_progress()

    restored = pipeline._experiment_states[search_string.id]
    assert restored is not stale
    assert restored.to_dict() == expected

    pipeline._checkpoint_progress(
        resumed,
        search_string=resumed.strings[0],
        page_num=resumed.current_page,
    )
    reloaded = pipeline._runtime_bridge.load_experiment_states(
        pipeline._runtime_run_id,
        progress=resumed,
    )
    assert reloaded[search_string.id].to_dict() == expected


@pytest.mark.parametrize("split_after_page", [1, 2])
def test_split_at_each_completed_page_matches_uninterrupted_state_and_decisions(
    tmp_path,
    split_after_page,
):
    decisions = {1: "continue", 2: "commit", 3: "stop"}

    def configure(pipeline, trace):
        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results
        # Model where a live run actually sits: THIS brief's own project discover
        # view. The brief pins "test-project" (below), so the old
        # "/talent/hire/123/search" was a foreign project's URL — previously inert
        # scenery, but the project guard now correctly reclassifies it and fires a
        # corrective navigation. That navigation is incidental to this test (which
        # is about state-split equivalence) and it awaited a non-async mock.
        pipeline.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        pipeline.browser.page.locator.return_value = locator
        pipeline.browser.enter_search_string = AsyncMock()
        pipeline.browser.get_results_count_text = AsyncMock(return_value="100")
        pipeline.browser.get_results_count = AsyncMock(return_value=100)
        pipeline.browser.go_to_next_page = AsyncMock(return_value=True)
        pipeline._ensure_browser_healthy = AsyncMock()
        pipeline._review_page_sequentially = AsyncMock(return_value=None)
        pipeline._evaluate_variant_lifecycle = MagicMock(return_value=None)

        async def assess(**kwargs):
            page = kwargs["page_num"]
            decision = decisions[page]
            trace.append((page, decision))
            return {
                "decision": decision,
                "rationale": f"decision-{page}",
                "page": page,
            }

        pipeline._assess_string_state = AsyncMock(side_effect=assess)

    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted_dir.mkdir()
    uninterrupted = _make_pipeline(str(uninterrupted_dir))
    uninterrupted_string = SearchString(
        id=18,
        name="equivalence",
        boolean="foo",
        status="in_progress",
    )
    uninterrupted_progress = Progress(
        brief_name="test",
        strings=[uninterrupted_string],
        current_string_id=uninterrupted_string.id,
    )
    uninterrupted_trace = []
    configure(uninterrupted, uninterrupted_trace)
    with patch("linkedin.orchestrator.human_delay_correlated", return_value=0):
        asyncio.run(
            uninterrupted._process_string(
                uninterrupted_string,
                uninterrupted_progress,
            )
        )
    expected_state = uninterrupted._experiment_states[
        uninterrupted_string.id
    ].to_dict()
    expected_string = uninterrupted_string.to_dict()

    split_dir = tmp_path / f"split-{split_after_page}"
    split_dir.mkdir()
    process_a = _make_pipeline(str(split_dir))
    split_string = SearchString(
        id=18,
        name="equivalence",
        boolean="foo",
        status="in_progress",
    )
    split_progress = Progress(
        brief_name="test",
        strings=[split_string],
        current_string_id=split_string.id,
    )
    split_trace = []
    configure(process_a, split_trace)
    durable_checkpoint = process_a._checkpoint_progress

    def crash_after_completed_checkpoint(*args, **kwargs):
        durable_checkpoint(*args, **kwargs)
        state = process_a._experiment_states[split_string.id]
        if (
            kwargs.get("page_num") == split_after_page
            and state.active_allocator_page_cursor() == split_after_page + 1
        ):
            raise RuntimeError(f"split after page {split_after_page}")

    process_a._checkpoint_progress = MagicMock(
        side_effect=crash_after_completed_checkpoint
    )
    with (
        patch("linkedin.orchestrator.human_delay_correlated", return_value=0),
        pytest.raises(RuntimeError, match=f"split after page {split_after_page}"),
    ):
        asyncio.run(process_a._process_string(split_string, split_progress))

    process_b = _make_pipeline(str(split_dir))
    resumed_progress = process_b._load_or_create_progress()
    resumed_string = resumed_progress.strings[0]
    configure(process_b, split_trace)
    with patch("linkedin.orchestrator.human_delay_correlated", return_value=0):
        asyncio.run(process_b._process_string(resumed_string, resumed_progress))

    assert split_trace == uninterrupted_trace
    assert split_trace[split_after_page] == (
        split_after_page + 1,
        decisions[split_after_page + 1],
    )
    assert process_b._experiment_states[resumed_string.id].to_dict() == expected_state
    assert resumed_string.to_dict() == expected_string


def test_run_full_recovery_checkpoint_does_not_persist_undecided_page_metrics(
    tmp_path,
):
    """A browser failure after assessment metrics must leave cursor N non-teaching."""

    def configure_page(pipeline, *, decision):
        no_results = MagicMock()
        no_results.is_visible = AsyncMock(return_value=False)
        locator = MagicMock()
        locator.first = no_results
        # Project id matches the fixture brief ("test-project") so run-start
        # takes the already-on-the-right-project path (E4) instead of
        # navigating; this test is about checkpoint metrics, not navigation.
        pipeline.browser.page = MagicMock(
            url=(
                "https://www.linkedin.com/talent/hire/test-project/discover/"
                "recruiterSearch"
            )
        )
        pipeline.browser.page.locator.return_value = locator
        pipeline.browser.enter_search_string = AsyncMock()
        pipeline.browser.get_results_count_text = AsyncMock(return_value="100")
        pipeline.browser.get_results_count = AsyncMock(return_value=100)
        pipeline._ensure_browser_healthy = AsyncMock()

        async def review_page(**kwargs):
            string_stats = kwargs["string_stats"]
            string_stats["candidates"] += 4
            string_stats["saves"] += 1
            string_stats["full_reviewed"] += 2
            string_stats["full_outreach"] += 1
            string_stats["full_review"] += 1

        pipeline._review_page_sequentially = AsyncMock(side_effect=review_page)
        pipeline._evaluate_variant_lifecycle = MagicMock(return_value=None)
        pipeline._assess_string_state = AsyncMock(
            return_value={
                "decision": decision,
                "rationale": f"decision-{decision}",
                "page": 1,
            }
        )

    interrupted_dir = tmp_path / "interrupted-middle"
    interrupted_dir.mkdir()
    process_a = _make_pipeline(str(interrupted_dir))
    interrupted_string = SearchString(
        id=19,
        name="middle-window",
        boolean="foo",
        status="in_progress",
    )
    interrupted_progress = Progress(
        brief_name="test",
        strings=[interrupted_string],
        current_string_id=interrupted_string.id,
    )
    process_a._experiment_states = {
        interrupted_string.id: bootstrap_experiment_state(interrupted_string)
    }
    process_a._checkpoint_progress(
        interrupted_progress,
        search_string=interrupted_string,
    )

    configure_page(process_a, decision="experiment")
    process_a.browser.connect = AsyncMock()
    process_a.browser.disconnect = AsyncMock()
    process_a._plan_variant_experiments = AsyncMock(
        return_value=[
            LinkedInSearchVariant(
                variant_id="precision-19-1",
                parent_variant_id="root",
                root_string_id=interrupted_string.id,
                boolean="bar",
                variant_kind="precision",
            )
        ]
    )
    process_a._search_mutation_executor.apply_variant = AsyncMock(
        side_effect=RuntimeError(
            "Page.evaluate: Target page, context or browser has been closed"
        )
    )
    process_a._capture_recovery_snapshot = AsyncMock(return_value=MagicMock())
    process_a._recovery_service.recover = AsyncMock(return_value=False)
    process_a._print_session_summary = MagicMock()
    process_a._print_summary = MagicMock()
    process_a._generate_run_report = MagicMock()
    process_a._finalize_run_snapshot = MagicMock(
        return_value=interrupted_dir / "frozen-run"
    )

    with (
        patch("linkedin.orchestrator.human_delay_correlated", return_value=0),
        patch("linkedin.orchestrator.config.MAX_PAGES_PER_STRING", 1),
        pytest.raises(RuntimeError, match="Target page"),
    ):
        asyncio.run(process_a.run_full(resume=True))

    process_b = _make_pipeline(str(interrupted_dir))
    resumed_progress = process_b._load_or_create_progress()
    resumed_string = resumed_progress.strings[0]
    resumed_state = process_b._experiment_states[resumed_string.id]

    assert resumed_state.active_allocator_page_cursor() == 1
    assert resumed_state.family_pages_reviewed_total == 0
    assert resumed_state.active_variant.pages_reviewed == 0
    assert resumed_string.status == "in_progress"

    configure_page(process_b, decision="stop")
    with (
        patch("linkedin.orchestrator.human_delay_correlated", return_value=0),
        patch("linkedin.orchestrator.config.MAX_PAGES_PER_STRING", 1),
    ):
        asyncio.run(process_b._process_string(resumed_string, resumed_progress))

    uninterrupted_dir = tmp_path / "uninterrupted-middle"
    uninterrupted_dir.mkdir()
    uninterrupted = _make_pipeline(str(uninterrupted_dir))
    uninterrupted_string = SearchString(
        id=19,
        name="middle-window",
        boolean="foo",
        status="in_progress",
    )
    uninterrupted_progress = Progress(
        brief_name="test",
        strings=[uninterrupted_string],
        current_string_id=uninterrupted_string.id,
    )
    configure_page(uninterrupted, decision="stop")
    with (
        patch("linkedin.orchestrator.human_delay_correlated", return_value=0),
        patch("linkedin.orchestrator.config.MAX_PAGES_PER_STRING", 1),
    ):
        asyncio.run(
            uninterrupted._process_string(
                uninterrupted_string,
                uninterrupted_progress,
            )
        )

    assert resumed_state.family_pages_reviewed_total == 1
    assert resumed_state.family_candidates_total == 4
    assert resumed_state.family_outreach_total == 1
    assert resumed_state.family_review_total == 1
    assert resumed_state.active_variant.pages_reviewed == 1
    assert resumed_state.active_allocator_page_cursor() == 2
    assert resumed_state.to_dict() == uninterrupted._experiment_states[
        uninterrupted_string.id
    ].to_dict()
    assert resumed_string.to_dict() == uninterrupted_string.to_dict()


def test_run_full_final_checkpoint_keeps_durable_completed_page_after_cancel(
    tmp_path,
):
    """A post-SQLite checkpoint failure must not downgrade N+1 back to N."""
    output_dir = tmp_path / "post-commit-cancel"
    output_dir.mkdir()
    pipeline = _make_pipeline(str(output_dir))
    search_string = SearchString(
        id=20,
        name="accepted-page",
        boolean="foo",
        status="in_progress",
    )
    progress = Progress(
        brief_name="test",
        strings=[search_string],
        current_string_id=search_string.id,
    )
    pipeline._experiment_states = {
        search_string.id: bootstrap_experiment_state(search_string)
    }
    pipeline._checkpoint_progress(progress, search_string=search_string)

    no_results = MagicMock()
    no_results.is_visible = AsyncMock(return_value=False)
    locator = MagicMock()
    locator.first = no_results
    # Project id matches the fixture brief ("test-project") so run-start takes
    # the already-on-the-right-project path (E4) instead of navigating.
    pipeline.browser.page = MagicMock(
        url=(
            "https://www.linkedin.com/talent/hire/test-project/discover/"
            "recruiterSearch"
        )
    )
    pipeline.browser.page.locator.return_value = locator
    pipeline.browser.connect = AsyncMock()
    pipeline.browser.disconnect = AsyncMock()
    pipeline.browser.enter_search_string = AsyncMock()
    pipeline.browser.get_results_count_text = AsyncMock(return_value="100")
    pipeline.browser.get_results_count = AsyncMock(return_value=100)
    pipeline._ensure_browser_healthy = AsyncMock()

    async def review_page(**kwargs):
        string_stats = kwargs["string_stats"]
        string_stats["candidates"] += 3
        string_stats["full_reviewed"] += 1
        string_stats["full_outreach"] += 1

    pipeline._review_page_sequentially = AsyncMock(side_effect=review_page)
    pipeline._evaluate_variant_lifecycle = MagicMock(return_value=None)
    pipeline._assess_string_state = AsyncMock(
        return_value={"decision": "stop", "rationale": "done", "page": 1}
    )
    pipeline._print_session_summary = MagicMock()
    pipeline._print_summary = MagicMock()
    pipeline._generate_run_report = MagicMock()
    pipeline._finalize_run_snapshot = MagicMock(
        return_value=output_dir / "frozen-run"
    )

    durable_checkpoint = pipeline._checkpoint_progress
    injected = {"raised": False}

    def cancel_after_durable_completed_checkpoint(*args, **kwargs):
        durable_checkpoint(*args, **kwargs)
        state = pipeline._experiment_states.get(search_string.id)
        if (
            not injected["raised"]
            and kwargs.get("search_string") is not None
            and kwargs.get("page_num") == 1
            and state is not None
            and state.active_allocator_page_cursor() == 2
        ):
            injected["raised"] = True
            raise asyncio.CancelledError()

    pipeline._checkpoint_progress = MagicMock(
        side_effect=cancel_after_durable_completed_checkpoint
    )

    with (
        patch("linkedin.orchestrator.human_delay_correlated", return_value=0),
        patch("linkedin.orchestrator.config.MAX_PAGES_PER_STRING", 1),
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(pipeline.run_full(resume=True))

    assert injected["raised"] is True
    resumed = _make_pipeline(str(output_dir))
    resumed_progress = resumed._load_or_create_progress()
    resumed_string = resumed_progress.strings[0]
    resumed_state = resumed._experiment_states[resumed_string.id]

    assert resumed_state.active_allocator_page_cursor() == 2
    assert resumed_state.family_pages_reviewed_total == 1
    assert resumed_state.family_candidates_total == 3
    assert resumed_state.family_outreach_total == 1
    assert resumed_string.status == "done"


def test_resume_clone_keeps_candidate_linked_to_old_run_until_retouched(tmp_path):
    """Cloned work units get a new run_id, but candidate linkage stays on the prior run until updated."""
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    initial_progress = Progress(
        brief_name="test",
        strings=[SearchString(id=1, name="builders", boolean="foo", status="in_progress")],
        current_string_id=1,
        current_page=1,
    )
    run_id_1, _ = bridge.start_or_resume_run(resume=False, initial_progress=initial_progress)
    work_unit_id_1 = store.get_work_unit_id(run_id_1, kind="linkedin_string", source_unit_id="1")
    assert work_unit_id_1 is not None

    url = "/talent/profile/cloned-linkage"
    store.record_candidate_discovery(
        run_id=run_id_1,
        work_unit_id=work_unit_id_1,
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
        display_name="Test",
        profile_url=url,
        payload={"source_string_id": 1},
    )
    attempt_id_1 = store.start_attempt(
        run_id=run_id_1,
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
        stage="facial",
        work_unit_id=work_unit_id_1,
        payload={"source_string_id": 1},
        source_cursor={"source_string_id": 1},
        display_name="Test",
        profile_url=url,
    )
    store.finish_attempt_failure(
        attempt_id=attempt_id_1,
        failure_kind="interrupted_test",
        failure_reason="test retryable failure",
        retryable=True,
        payload={"source_string_id": 1},
        run_id=run_id_1,
    )

    run_id_2, _ = bridge.start_or_resume_run(resume=True)
    assert run_id_2 != run_id_1
    work_unit_id_2 = store.get_work_unit_id(run_id_2, kind="linkedin_string", source_unit_id="1")
    assert work_unit_id_2 is not None
    assert work_unit_id_2 != work_unit_id_1

    candidate_before = store.get_candidate(
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
    )
    assert candidate_before is not None
    assert candidate_before["last_work_unit_id"] == work_unit_id_1
    assert candidate_before["last_attempt_id"] == attempt_id_1

    with store.connect() as conn:
        work_unit_rows = conn.execute(
            "SELECT id, run_id FROM work_units WHERE id IN (?, ?) ORDER BY id ASC",
            (work_unit_id_1, work_unit_id_2),
        ).fetchall()
        attempt_row = conn.execute(
            "SELECT run_id FROM candidate_attempts WHERE id = ?",
            (attempt_id_1,),
        ).fetchone()

    assert {row["id"]: row["run_id"] for row in work_unit_rows} == {
        work_unit_id_1: run_id_1,
        work_unit_id_2: run_id_2,
    }
    assert attempt_row["run_id"] == run_id_1

    store.record_candidate_discovery(
        run_id=run_id_2,
        work_unit_id=work_unit_id_2,
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
        display_name="Test",
        profile_url=url,
        payload={"source_string_id": 1},
    )
    attempt_id_2 = store.start_attempt(
        run_id=run_id_2,
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
        stage="facial",
        work_unit_id=work_unit_id_2,
        payload={"source_string_id": 1},
        source_cursor={"source_string_id": 1},
        display_name="Test",
        profile_url=url,
    )

    candidate_after = store.get_candidate(
        source="linkedin",
        brief_id="test-project",
        identity_key=url,
    )
    assert candidate_after is not None
    assert candidate_after["last_work_unit_id"] == work_unit_id_2
    assert candidate_after["last_attempt_id"] == attempt_id_2

    with store.connect() as conn:
        attempt_row_2 = conn.execute(
            "SELECT run_id FROM candidate_attempts WHERE id = ?",
            (attempt_id_2,),
        ).fetchone()
    assert attempt_row_2["run_id"] == run_id_2


def test_resume_finalizes_stale_running_run_before_cloning_work_units(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    initial_progress = Progress(
        brief_name="test",
        strings=[SearchString(id=1, name="builders", boolean="foo", status="in_progress")],
        current_string_id=1,
        current_page=3,
    )
    stale_run_id, _ = bridge.start_or_resume_run(
        resume=False,
        initial_progress=initial_progress,
    )
    stale_run_before = store.get_run(stale_run_id)
    assert stale_run_before["status"] == "running"

    new_run_id, resumed_progress = bridge.start_or_resume_run(resume=True)

    stale_run_after = store.get_run(stale_run_id)
    new_run = store.get_run(new_run_id)
    cloned_units = store.list_work_units(new_run_id, kind="linkedin_string")
    assert stale_run_after["status"] == "interrupted"
    assert stale_run_after["stop_reason"] == RunStopReason.WORKER_MISSING
    assert stale_run_after["ended_at"]
    assert new_run_id != stale_run_id
    assert new_run["resumed_from_run_id"] == stale_run_id
    assert len(cloned_units) == 1
    assert cloned_units[0]["source_unit_id"] == "1"
    assert resumed_progress.current_string_id == 1
    assert resumed_progress.current_page == 3


def test_progress_sync_rolls_back_resume_and_all_work_units_together(
    tmp_path,
    monkeypatch,
):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="test-project",
        brief_name="test",
    )
    first = SearchString(
        id=1,
        name="first",
        boolean="first",
        status="in_progress",
        block="Compound Batch 1",
    )
    second = SearchString(
        id=2,
        name="second",
        boolean="second",
        status="queued",
        block="Compound Batch 1",
    )
    progress = Progress(
        brief_name="test",
        strings=[first, second],
        current_string_id=first.id,
        current_page=2,
    )
    run_id, _ = bridge.start_or_resume_run(
        resume=False,
        initial_progress=progress,
    )

    progress.strings[:] = [second, first]
    first.status = "queued"
    second.status = "in_progress"
    progress.current_string_id = second.id
    progress.current_page = 1

    real_upsert = store.upsert_work_unit
    calls = 0

    def fail_second_upsert(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second work-unit failure")
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(store, "upsert_work_unit", fail_second_upsert)
    with pytest.raises(RuntimeError, match="injected second work-unit failure"):
        bridge.sync_progress(run_id, progress)

    loaded = bridge.load_progress(run_id)
    assert [item.id for item in loaded.strings] == [first.id, second.id]
    assert [item.status for item in loaded.strings] == ["in_progress", "queued"]
    assert loaded.current_string_id == first.id
    assert loaded.current_page == 2


# ---------------------------------------------------------------------------
# Sourcing-judgment kernel P5: lane attribution round-trip from canonical
# runtime state. A REVIEW_INFERRED candidate written by the P4 dispatch
# path must surface via ``lane_metrics_for_run`` under the correct
# lane_id without re-reading any projection file.
# ---------------------------------------------------------------------------


def test_lane_metrics_attribute_p4_review_outcome_from_canonical_state(
    tmp_path,
):
    from shared.runtime_state.lane_metrics import lane_metrics_for_run

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    brief_id = "brief-lane-attribution"
    run_id = store.start_run(
        source="linkedin",
        brief_id=brief_id,
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )

    # Production write path: SearchString.to_dict() lands in
    # work_units.payload_json with the lane_* fields populated.
    search_string = SearchString(
        id=1,
        name="bfs senior applied ai",
        boolean="ml",
        lane_id="bfs_senior_applied_ai",
        lane_name="BFS Senior Applied AI",
        lane_intent="senior leaders driving applied-AI delivery",
        acquisition_mode="linkedin_boolean",
    )
    work_unit_id = store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        kind="linkedin_string",
        source_unit_id=str(search_string.id),
        display_name=search_string.name,
        ordering_index=1,
        status="done",
        payload=search_string.to_dict(),
        checkpoint={},
        metrics={},
        family_key="",
        novelty_bucket="",
        domain_lane="",
        counters={
            "result_count": 75,
            "candidates_discovered": 3,
            "facial_yes_count": 1,
            "facial_no_count": 2,
        },
        notes="",
    )

    # Walk a candidate through the lifecycle to full_terminal with a
    # REVIEW_INFERRED decision. Mirrors P4's dispatch.
    identity_key = "/talent/profile/ada"
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key=identity_key,
        display_name="Ada Lovelace",
        profile_url=identity_key,
    )
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
    store.finish_attempt_success(
        attempt_id=full_attempt,
        new_state="full_terminal",
        terminal_decision="REVIEW_INFERRED",
        payload={},
        terminal_payload={
            # P4 orchestrator writes both ``lane`` (attribution) and
            # ``full_decision`` (OpusDecision.to_dict() with the
            # review_reason_code populated) into terminal_payload_json.
            "lane": {
                "lane_id": search_string.lane_id,
                "lane_name": search_string.lane_name,
                "lane_intent": search_string.lane_intent,
            },
            "full_decision": {
                "decision": "REVIEW_INFERRED",
                "review_reason_code": "inferred_high_priority",
                "review_structural_evidence": [
                    "senior bank title",
                    "CS PhD",
                    "relevant org scope",
                ],
            },
        },
        run_id=run_id,
    )

    rows = lane_metrics_for_run(store.db_path, run_id=run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.lane_id == "bfs_senior_applied_ai"
    assert row.lane_name == "BFS Senior Applied AI"
    assert row.acquisition_mode == "linkedin_boolean"
    assert row.review_count == 1
    assert row.review_by_reason == {"inferred_high_priority": 1}
    assert row.save_count == 0
    assert row.reject_count == 0
    assert row.opened_count == 1
    assert row.evaluated_count == 1
    assert row.facial_yes_count == 1
    assert row.facial_no_count == 2
    assert row.legacy is False


def test_hydration_counts_reconciled_self_save_as_saved(tmp_path):
    """Reconciled self-saves count as saved; foreign already_present does not."""
    p = _make_pipeline(str(tmp_path))
    search_string = SearchString(
        id=7,
        name="saves-test",
        boolean="ml",
        status="in_progress",
    )
    progress = Progress(
        brief_name="test",
        strings=[search_string],
        current_string_id=7,
    )
    run_id, progress = p._runtime_bridge.start_or_resume_run(
        resume=False,
        initial_progress=progress,
    )
    p._runtime_run_id = run_id
    search_string = progress.strings[0]

    def record_save_candidate(
        slug: str,
        *,
        rank: int,
        save_payload: dict,
    ) -> CandidateSnippet:
        snippet = _snippet(
            name=slug.title(),
            profile_url=f"/talent/profile/{slug}",
            source_string_id=search_string.id,
            source_string_name=search_string.name,
            result_rank=rank,
        )
        p._runtime_bridge.record_snippet_extracted(
            run_id=run_id,
            search_string=search_string,
            snippet=snippet,
        )
        facial_attempt_id = p._runtime_bridge.start_stage_attempt(
            run_id=run_id,
            search_string=search_string,
            snippet=snippet,
            stage="facial",
        )
        facial = OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="facial",
            confidence=0.8,
            rationale="facial yes",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        p._runtime_bridge.finish_stage_success(
            run_id=run_id,
            attempt_id=facial_attempt_id,
            stage="facial",
            snippet=snippet,
            decision=facial,
        )
        full_attempt_id = p._runtime_bridge.start_stage_attempt(
            run_id=run_id,
            search_string=search_string,
            snippet=snippet,
            stage="full",
        )
        full = OpusDecision(
            stage="full",
            decision="SAVE",
            path="DIRECT:runtime truth",
            confidence=0.82,
            rationale="save candidate",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        p._runtime_bridge.finish_stage_success(
            run_id=run_id,
            attempt_id=full_attempt_id,
            stage="full",
            snippet=snippet,
            decision=full,
        )
        save_start = p._runtime_bridge.begin_candidate_side_effect(
            run_id=run_id,
            search_string=search_string,
            snippet=snippet,
            attempt_id=full_attempt_id,
            effect_type="linkedin_save",
            idempotency_key="save",
            payload={"search_string_id": search_string.id},
        )
        p._runtime_bridge.complete_candidate_side_effect(
            side_effect_id=int(save_start["side_effect"]["id"]),
            status="succeeded",
            payload=save_payload,
        )
        return snippet

    plain = record_save_candidate(
        "plain-save",
        rank=1,
        save_payload={"test_mode": True},
    )
    reconciled = record_save_candidate(
        "reconciled-self-save",
        rank=2,
        save_payload={
            "already_present": True,
            "reconciled_self_save": True,
        },
    )
    record_save_candidate(
        "foreign-already-present",
        rank=3,
        save_payload={"already_present": True},
    )

    p._hydrate_resume_funnel_from_runtime(progress)

    assert p.stats["saved"] == 2
    assert p.stats["already_present"] == 1
    hydrated_string = progress.strings[0]
    assert plain.name in hydrated_string.saves
    assert reconciled.name in hydrated_string.saves
    assert hydrated_string.saves.count(reconciled.name) == 1
