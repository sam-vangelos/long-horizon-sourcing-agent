"""Tests for GitHub pipeline interruption safety and governor enforcement.

Covers: governor lifecycle, query error handling, deferred dedup, and resume semantics.

Run with: python -m pytest tests/test_github_pipeline.py -v
"""

import asyncio
import importlib
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from github.schemas import ContactInfo, GitHubCandidate, GitHubUser
from shared.execution import CandidateExecutionEngine
from shared.runtime_state import GitHubRuntimeStateBridge, RuntimeStateLock, RuntimeStateStore
from shared.storage import append_jsonl


class _FakeExhaustionState:
    def __init__(self):
        self.channels = {}

    def record_query_result(self, **kw):
        pass

    def to_adaptation_context(self):
        return ""


def _import_pipeline_with_stubs():
    """Import github.orchestrator under temporary stubs without polluting later tests."""
    stub_names = (
        "github.client",
        "github.enricher",
        "github.strategy",
        "github.query_validator",
        "github.observability",
        "github.outreach",
        "github.export",
        "shared.contact_discovery",
        "shared.judger",
    )
    originals = {name: sys.modules.get(name) for name in stub_names}

    try:
        for mod_name in stub_names:
            sys.modules[mod_name] = types.ModuleType(mod_name)

        client_mod = sys.modules["github.client"]
        client_mod.GitHubClient = MagicMock
        client_mod.GitHubAuthError = type("GitHubAuthError", (RuntimeError,), {"status_code": 401})

        enricher_mod = sys.modules["github.enricher"]
        enricher_mod.GitHubEnricher = MagicMock

        strategy_mod = sys.modules["github.strategy"]
        strategy_mod.form_github_strategy = MagicMock(return_value=([], ""))
        strategy_mod.adapt_after_batch = MagicMock()
        strategy_mod.build_github_adaptation_decision = MagicMock()

        qv_mod = sys.modules["github.query_validator"]
        qv_mod.ExhaustionState = _FakeExhaustionState

        obs_mod = sys.modules["github.observability"]
        obs_mod.SessionObserver = MagicMock

        contact_mod = sys.modules["shared.contact_discovery"]
        contact_mod.merge_profile_contact = MagicMock()

        judger_mod = sys.modules["shared.judger"]
        for name in (
            "facial_judge",
            "full_judge",
            "init_judger",
            "github_facial_judge",
            "github_facial_judge_batch",
            "github_full_judge",
            "extract_priority_rank",
        ):
            setattr(judger_mod, name, MagicMock())

        judger_mod.is_failure_decision = (
            lambda decision: decision in ("PARSE_FAILURE", "JUDGMENT_FAILURE")
        )

        outreach_mod = sys.modules["github.outreach"]
        outreach_mod.generate_outreach = AsyncMock(return_value={"message": "hi"})

        from github.governor import GitHubGovernor, GitHubGovernorLimitReached
        from github.schemas import GitHubProgress, GitHubSearchQuery

        orchestrator_mod = importlib.import_module("github.orchestrator")
        return (
            GitHubGovernor,
            GitHubGovernorLimitReached,
            GitHubProgress,
            GitHubSearchQuery,
            orchestrator_mod.GitHubPipeline,
            orchestrator_mod,
        )
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


(
    GitHubGovernor,
    GitHubGovernorLimitReached,
    GitHubProgress,
    GitHubSearchQuery,
    GitHubPipeline,
    github_orchestrator,
) = _import_pipeline_with_stubs()


# ---------------------------------------------------------------------------
# Minimal pipeline fixture
# ---------------------------------------------------------------------------

def _make_pipeline(**overrides):
    """Build a GitHubPipeline with heavy deps stubbed out."""
    brief = MagicMock()
    brief.id = "test-brief"
    brief.has_v2_schema = True
    brief._new_brief = {}

    with patch.object(GitHubPipeline, "__init__", lambda self: None):
        p = GitHubPipeline()

    p.brief_path = "fake.json"
    p.brief_obj = brief
    p.output_dir = "/tmp/test_gh_pipeline"
    p.progress_path = Path(tempfile.mkdtemp(prefix="gh_pipeline_test_")) / "progress.json"
    p.candidates_path = "/tmp/test_gh_pipeline/candidates.jsonl"
    p.snippets_path = "/tmp/test_gh_pipeline/snippets.jsonl"
    p.facial_path = "/tmp/test_gh_pipeline/facial.jsonl"
    p.profiles_path = "/tmp/test_gh_pipeline/profiles.jsonl"
    p.final_path = "/tmp/test_gh_pipeline/final.jsonl"
    p.saves_path = "/tmp/test_gh_pipeline/saves.jsonl"
    p.outreach_path = "/tmp/test_gh_pipeline/outreach.jsonl"
    p.log_path = "/tmp/test_gh_pipeline/log.jsonl"
    p.bias_path = Path("/tmp/test_gh_pipeline/bias_monitor.json")
    p.stats = {
        "candidates_discovered": 0,
        "candidates_enriched": 0,
        "facial_yes": 0,
        "facial_no": 0,
        "saved": 0,
        "rejected": 0,
        "insufficient": 0,
    }
    p._governor = GitHubGovernor()
    p._governor.start_session()
    p._bias_monitor = None
    # P6.4: pause severity stops the current query — reset per query.
    p._bias_pause_active = False
    p._progress = None
    p._client = None
    p._observer = MagicMock()
    p._shutdown_requested = False
    p._seen_usernames = set()
    p._in_flight_usernames = set()
    p._exhaustion = _FakeExhaustionState()
    # P6.3: batch-scoped accumulators for _build_batch_report honesty.
    p._batch_baseline_stats = dict(p.stats)
    p._batch_save_candidates = []

    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def _make_query(**overrides) -> GitHubSearchQuery:
    defaults = dict(
        id=1, name="test query", query="language:python", channel="user_search",
    )
    defaults.update(overrides)
    return GitHubSearchQuery(**defaults)


def _make_candidate(username: str, name: str) -> GitHubCandidate:
    return GitHubCandidate(
        user=GitHubUser(
            username=username,
            name=name,
            profile_url=f"https://github.com/{username}",
        ),
        contact=ContactInfo(),
        portfolio_summary={"profile_summary": f"{name} builds ML systems"},
    )


def _attach_runtime_state(pipeline, base_dir: Path):
    pipeline.output_dir = Path(base_dir)
    pipeline.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.progress_path = pipeline.output_dir / "progress.json"
    pipeline.runtime_db_path = pipeline.output_dir / "runtime_state.sqlite3"
    pipeline._runtime_state = RuntimeStateStore(pipeline.runtime_db_path)
    pipeline._runtime_lock = RuntimeStateLock(pipeline.output_dir)
    pipeline._runtime_bridge = GitHubRuntimeStateBridge(
        store=pipeline._runtime_state,
        output_dir=pipeline.output_dir,
        brief_id=pipeline.brief_obj.id,
        brief_name=pipeline.brief_obj.id,
    )
    pipeline._execution_engine = CandidateExecutionEngine(
        store=pipeline._runtime_state,
        output_dir=str(pipeline.output_dir),
        brief_id=pipeline.brief_obj.id,
        source="github",
    )
    pipeline._runtime_run_id = None
    return pipeline


# ---------------------------------------------------------------------------
# Fix 1: Governor lifecycle
# ---------------------------------------------------------------------------

class TestGovernorLifecycle:
    def test_governor_starts_in_run(self):
        """Governor._active is True after start_session() called."""
        gov = GitHubGovernor()
        assert not gov._active
        gov.start_session()
        assert gov._active

    def test_governor_ends_in_finally(self):
        """Governor._active is False after end_session()."""
        gov = GitHubGovernor()
        gov.start_session()
        assert gov._active
        gov.end_session()
        assert not gov._active


# ---------------------------------------------------------------------------
# Fix 2: Query error handling
# ---------------------------------------------------------------------------

class TestQueryErrorHandling:
    def test_query_done_on_success(self):
        """Successful query → status='done'."""
        pipeline = _make_pipeline()
        query = _make_query()
        progress = GitHubProgress(brief_name="test")
        progress.queries = [query]

        client = MagicMock()
        client.limiter.remaining.return_value = 5000
        enricher = MagicMock()

        pipeline._execute_single_query = AsyncMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._get_executed_query_strings = MagicMock(return_value=set())

        asyncio.run(pipeline._execute_queries(client, enricher, progress))
        assert query.status == "done"

    def test_query_error_on_failure(self):
        """Failed query → status='error'."""
        pipeline = _make_pipeline()
        query = _make_query()
        progress = GitHubProgress(brief_name="test")
        progress.queries = [query]

        client = MagicMock()
        client.limiter.remaining.return_value = 5000
        enricher = MagicMock()

        pipeline._execute_single_query = AsyncMock(side_effect=RuntimeError("boom"))
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})

        asyncio.run(pipeline._execute_queries(client, enricher, progress))
        assert query.status == "error"
        assert "boom" in query.notes

    def test_auth_error_stops_query_loop(self):
        """Credential failures are fatal, not per-query misses."""
        pipeline = _make_pipeline()
        query = _make_query()
        progress = GitHubProgress(brief_name="test")
        progress.queries = [query]

        client = MagicMock()
        client.limiter.remaining.return_value = 5000
        enricher = MagicMock()

        auth_error = github_orchestrator.GitHubAuthError("bad credentials")
        pipeline._execute_single_query = AsyncMock(side_effect=auth_error)
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})

        with pytest.raises(github_orchestrator.GitHubAuthError):
            asyncio.run(pipeline._execute_queries(client, enricher, progress))

        assert query.status == "in_progress"

    def test_error_query_retried_on_resume(self):
        """'error' queries are NOT in the skip set ('done', 'skipped')."""
        query = _make_query(status="error")
        assert query.status not in ("done", "skipped")

    def test_error_query_not_in_batch_stats(self):
        """Errored query does not appear in batch_stats — only done query does."""
        pipeline = _make_pipeline()
        q1 = _make_query(id=1)
        q2 = _make_query(id=2)
        progress = GitHubProgress(brief_name="test")
        progress.queries = [q1, q2]

        client = MagicMock()
        client.limiter.remaining.return_value = 5000
        enricher = MagicMock()

        call_count = 0
        async def _side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail first")

        pipeline._execute_single_query = AsyncMock(side_effect=_side_effect)
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})

        asyncio.run(pipeline._execute_queries(client, enricher, progress))

        # q1 errored, q2 succeeded
        assert q1.status == "error"
        assert q2.status == "done"

    def test_resume_skips_do_not_trigger_early_adaptation(self):
        """Resume batches should count newly executed queries, not raw loop index."""
        pipeline = _make_pipeline()
        q1 = _make_query(id=1, status="done")
        q2 = _make_query(id=2, status="skipped")
        q3 = _make_query(id=3)
        progress = GitHubProgress(brief_name="test")
        progress.queries = [q1, q2, q3]

        client = MagicMock()
        client.limiter.remaining.return_value = 5000
        enricher = MagicMock()

        pipeline._execute_single_query = AsyncMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._get_executed_query_strings = MagicMock(return_value=set())

        with patch.object(github_orchestrator, "_ADAPTATION_BATCH_SIZE", 2), patch.object(
            github_orchestrator, "adapt_after_batch", return_value=([], "", [])
        ) as adapt_mock:
            asyncio.run(pipeline._execute_queries(client, enricher, progress))

        adapt_mock.assert_not_called()

    def test_adapted_queries_insert_with_source_native_priority(self):
        pipeline = _make_pipeline()
        queries = [
            _make_query(id=1, status="done", channel="repo_mining"),
            _make_query(id=2, status="queued", channel="code_search"),
        ]
        adapted = [
            _make_query(id=10, channel="repo_mining"),
            _make_query(id=11, channel="graph_expansion"),
            _make_query(id=12, channel="user_search"),
        ]

        pipeline._insert_queries_by_priority(queries, adapted, current_index=0)

        assert [query.id for query in queries] == [1, 12, 11, 2, 10]

    def test_v2_execute_single_query_batches_facial_triage(self):
        """V2 GitHub flow should call the batch facial helper once per collected batch."""
        pipeline = _make_pipeline()
        query = _make_query()
        progress = GitHubProgress(brief_name="test")

        client = MagicMock()
        enricher = MagicMock()

        alice = _make_candidate("alice", "Alice")
        bob = _make_candidate("bob", "Bob")

        pipeline._search_users = AsyncMock(return_value=(2, ["alice", "bob"]))
        pipeline._prepare_candidate_for_evaluation = AsyncMock(side_effect=[alice, bob])

        with patch.object(
            github_orchestrator,
            "github_facial_judge_batch",
            return_value=[
                github_orchestrator.OpusDecision(
                    stage="facial", decision="FACIAL_NO", path="none",
                    confidence=1.0, rationale="out of scope",
                    candidate_name="Alice", profile_url="https://github.com/alice",
                ),
                github_orchestrator.OpusDecision(
                    stage="facial", decision="FACIAL_NO", path="none",
                    confidence=1.0, rationale="out of scope",
                    candidate_name="Bob", profile_url="https://github.com/bob",
                ),
            ],
        ) as batch_mock, patch.object(
            github_orchestrator,
            "github_facial_judge",
            side_effect=AssertionError("single-candidate GitHub facial should not run"),
        ):
            asyncio.run(pipeline._execute_single_query(client, enricher, query, progress))

        batch_mock.assert_called_once()
        assert pipeline.stats["facial_no"] == 2


# ---------------------------------------------------------------------------
# Fix 3: Deferred dedup
# ---------------------------------------------------------------------------

class TestDeferredDedup:
    def test_dedup_terminal_permanent(self):
        """Username with terminal outcome stays in _seen_usernames."""
        pipeline = _make_pipeline()
        pipeline._in_flight_usernames.add("alice")
        pipeline._mark_terminal("alice")
        assert "alice" in pipeline._seen_usernames
        assert "alice" not in pipeline._in_flight_usernames

    def test_dedup_inflight_not_persisted(self):
        """_in_flight_usernames not in progress.discovered_usernames."""
        pipeline = _make_pipeline()
        progress = GitHubProgress(brief_name="test")
        pipeline._progress = progress
        pipeline._client = MagicMock()
        pipeline._client.limiter.total_calls = 0

        # Simulate in-flight and terminal
        pipeline._in_flight_usernames = {"inflight_user"}
        pipeline._seen_usernames = {"terminal_user"}

        pipeline._save_progress()

        assert "terminal_user" in progress.discovered_usernames
        assert "inflight_user" not in progress.discovered_usernames

    def test_light_enrich_failure_not_terminal(self):
        """light_enrich returning None → username NOT in _seen_usernames."""
        pipeline = _make_pipeline()
        result = pipeline._dedup_usernames(["bob"])
        assert result == ["bob"]
        assert "bob" in pipeline._in_flight_usernames

        # Simulate light_enrich returning None — no _mark_terminal called
        assert "bob" not in pipeline._seen_usernames
        assert "bob" in pipeline._in_flight_usernames

    def test_parse_failure_not_terminal(self):
        """PARSE_FAILURE → username NOT in _seen_usernames (non-terminal)."""
        pipeline = _make_pipeline()
        pipeline._dedup_usernames(["charlie"])
        # PARSE_FAILURE exits without calling _mark_terminal
        assert "charlie" not in pipeline._seen_usernames
        assert "charlie" in pipeline._in_flight_usernames

    def test_candidates_jsonl_not_dedup_source(self):
        """_seen_usernames not loaded from candidates.jsonl."""
        pipeline = _make_pipeline()
        assert pipeline._seen_usernames == set()

    def test_dedup_blocks_seen_and_inflight(self):
        """_dedup_usernames filters both _seen and _in_flight."""
        pipeline = _make_pipeline()
        pipeline._seen_usernames = {"alice"}
        pipeline._in_flight_usernames = {"bob"}

        result = pipeline._dedup_usernames(["alice", "bob", "charlie"])
        assert result == ["charlie"]
        assert "charlie" in pipeline._in_flight_usernames

    def test_runtime_state_blocks_terminal_candidates(self, tmp_path):
        """Canonical candidate state should block dedup even without progress.json."""
        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        progress = pipeline._load_or_create_progress(resume=False)
        pipeline._progress = progress

        pipeline._runtime_state.record_candidate_discovery(
            run_id=pipeline._runtime_run_id,
            work_unit_id=None,
            source="github",
            brief_id=pipeline.brief_obj.id,
            identity_key="alice",
            display_name="Alice",
            profile_url="https://github.com/alice",
        )
        pipeline._runtime_state.set_candidate_state(
            run_id=pipeline._runtime_run_id,
            source="github",
            brief_id=pipeline.brief_obj.id,
            identity_key="alice",
            new_state="failed_terminal",
            terminal_decision="SAVE",
        )

        result = pipeline._dedup_usernames(["alice", "bob"])
        assert result == ["bob"]

    def test_resume_uses_runtime_state_when_progress_json_is_missing(self, tmp_path):
        """Resume should be deterministic from runtime_state even if progress.json was deleted."""
        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        progress = pipeline._load_or_create_progress(resume=False)
        progress.queries = [
            _make_query(id=1, status="done"),
            _make_query(id=2, status="queued"),
        ]
        pipeline._progress = progress
        pipeline._seen_usernames = {"alice"}
        pipeline._save_progress()

        pipeline.progress_path.unlink()

        resumed = _attach_runtime_state(_make_pipeline(), tmp_path)
        loaded = resumed._load_or_create_progress(resume=True)

        assert [q.id for q in loaded.queries] == [1, 2]
        assert loaded.queries[0].status == "done"
        assert loaded.queries[1].status == "queued"
        assert "alice" in loaded.discovered_usernames

    def test_resume_reconciles_orphaned_attempts(self, tmp_path):
        """Open attempts from an interrupted run should become failed_retryable on resume."""
        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline._load_or_create_progress(resume=False)

        pipeline._runtime_state.record_candidate_discovery(
            run_id=pipeline._runtime_run_id,
            work_unit_id=None,
            source="github",
            brief_id=pipeline.brief_obj.id,
            identity_key="alice",
            display_name="Alice",
            profile_url="https://github.com/alice",
        )
        attempt_id = pipeline._runtime_state.start_attempt(
            run_id=pipeline._runtime_run_id,
            source="github",
            brief_id=pipeline.brief_obj.id,
            identity_key="alice",
            stage="facial",
            payload={},
            source_cursor={},
            display_name="Alice",
            profile_url="https://github.com/alice",
        )
        assert attempt_id > 0

        resumed = _attach_runtime_state(_make_pipeline(), tmp_path)
        resumed._load_or_create_progress(resume=True)

        candidate = resumed._runtime_state.get_candidate(
            source="github",
            brief_id=resumed.brief_obj.id,
            identity_key="alice",
        )
        assert candidate["current_lifecycle_state"] == "failed_retryable"
        assert resumed._runtime_state.list_orphaned_attempts(
            source="github",
            brief_id=resumed.brief_obj.id,
        ) == []


# ---------------------------------------------------------------------------
# P4.3.1 — run health wired into finalize (shared.observability_monitors)
# ---------------------------------------------------------------------------


def test_run_health_summary_flags_green_but_useless(tmp_path):
    pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
    run_id = pipeline._runtime_state.start_run(
        source="github",
        brief_id=pipeline.brief_obj.id,
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={},
    )
    pipeline._runtime_state.record_event(run_id=run_id, event_type="pipeline_start")
    pipeline._runtime_state.record_event(run_id=run_id, event_type="pipeline_end")
    pipeline._runtime_state.finish_run(run_id, "completed")
    pipeline._runtime_run_id = run_id

    health = pipeline._run_health_summary()

    assert health["status"] == "ok"
    assert health["degraded"] is True
    assert "green_but_useless" in health["degraded_reasons"]


def test_run_health_summary_no_runtime_state_before_run_starts(tmp_path):
    pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)

    health = pipeline._run_health_summary()

    assert health == {"status": "no_runtime_state"}


# ---------------------------------------------------------------------------
# P4.2 — orchestrator-level cost wiring on pipeline_end.
#
# Unlike LinkedIn (which has an isolated, directly-testable
# ``_pipeline_end_stats()`` method), GitHub's cost_usd/cost_per_save_usd
# summing lives inline in ``run()``'s ``finally:`` block. Exercise it
# through a real ``run()`` invocation (with the query-execution surface
# empty, so nothing beyond strategy/finalize runs) rather than duplicating
# the inline computation in a test.
# ---------------------------------------------------------------------------


class _FakeGitHubApiLimiter:
    total_calls = 0

    def remaining(self, *_args, **_kwargs):
        return 5000


class _FakeGitHubClient:
    """Async-context-manager stand-in for github.client.GitHubClient."""

    def __init__(self, *_args, **_kwargs):
        self.limiter = _FakeGitHubApiLimiter()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info):
        return False

    async def validate_credentials(self):
        return None


def _run_github_pipeline_to_completion(pipeline) -> dict:
    # The module-level `SessionObserver = MagicMock` stub (see
    # _import_pipeline_with_stubs above) only works when instantiated with
    # zero args, as existing tests do (`p._observer = MagicMock()`). The
    # real `run()` constructs it as `SessionObserver(session_id,
    # output_dir, brief_obj)` — passing those positionally into MagicMock's
    # own constructor sets `spec=session_id`, which produces a str-spec'd
    # mock with none of the observer's real methods. Swap in a permissive
    # factory for the duration of a real `run()` invocation.
    with patch.object(github_orchestrator, "GitHubClient", _FakeGitHubClient), \
         patch.object(github_orchestrator, "SessionObserver", lambda *a, **kw: MagicMock()):
        return asyncio.run(pipeline.run(resume=False))


def _read_events(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


class TestPipelineEndCostOrchestratorLevel:
    def test_pipeline_end_omits_cost_when_jsonl_absent(self, tmp_path):
        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline.state_dir = pipeline.output_dir
        pipeline.log_path = pipeline.output_dir / "log.jsonl"

        stats = _run_github_pipeline_to_completion(pipeline)

        assert "cost_usd" not in stats
        assert "cost_per_save_usd" not in stats

        events = _read_events(pipeline.log_path)
        pipeline_end = next(e for e in events if e.get("event") == "pipeline_end")
        assert "cost_usd" not in pipeline_end
        assert "cost_per_save_usd" not in pipeline_end

    def test_pipeline_end_emits_correct_sum_when_jsonl_present(self, tmp_path):
        pipeline = _attach_runtime_state(_make_pipeline(), tmp_path)
        pipeline.state_dir = pipeline.output_dir
        pipeline.log_path = pipeline.output_dir / "log.jsonl"
        # saved stays 0 — this test is scoped to the cost sum, not the
        # saved-candidate CSV export side effect that a nonzero saved
        # count would trigger via the real (unstubbed) github.export.
        cost_log = pipeline.output_dir / "token-cost-log.jsonl"
        append_jsonl(cost_log, {"provider": "anthropic", "estimated_cost_usd": 0.5})
        append_jsonl(cost_log, {"provider": "anthropic", "estimated_cost_usd": 1.0})

        stats = _run_github_pipeline_to_completion(pipeline)

        assert stats["cost_usd"] == 1.5
        assert "cost_per_save_usd" not in stats  # no saves — omitted, not $0

        events = _read_events(pipeline.log_path)
        pipeline_end = next(e for e in events if e.get("event") == "pipeline_end")
        assert pipeline_end["cost_usd"] == 1.5
        assert "cost_per_save_usd" not in pipeline_end


# ---------------------------------------------------------------------------
# O4: typed enrichment-failure events + live failure counter
# ---------------------------------------------------------------------------

def test_enricher_website_fetch_failure_emits_typed_event():
    from github.enricher import GitHubEnricher

    recorded: list[tuple[str, dict]] = []

    def _recorder(event_type: str, payload: dict) -> None:
        recorded.append((event_type, payload))

    client = MagicMock()
    enricher = GitHubEnricher(client, safety_event_recorder=_recorder)

    candidate = _make_candidate("alice", "Alice")
    candidate.user.blog = "https://myblog.example.com"

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    async def _run():
        with patch("github.enricher.validate_public_url", new=AsyncMock(return_value=(True, ""))):
            with patch("github.enricher.fetch_text_if_safe", new=AsyncMock(side_effect=RuntimeError("fetch boom"))):
                with patch("aiohttp.ClientSession", return_value=session_ctx):
                    await enricher._crawl_website_and_papers(candidate)

    asyncio.run(_run())

    enrichment_events = [(et, pl) for et, pl in recorded if et == "enrichment_failure"]
    assert len(enrichment_events) == 1
    _, payload = enrichment_events[0]
    assert payload["kind"] == "website_fetch"
    assert payload["identity_key"] == "alice"
    assert payload.get("url")


def test_observer_on_enrichment_failure_increments_report_counter(tmp_path):
    from github.observability.observer import SessionObserver

    brief = MagicMock()
    brief.id = "test-brief"
    observer = SessionObserver("sess-o4", tmp_path, brief)

    observer.on_enrichment_failure("website_fetch")
    observer.on_enrichment_failure("website_fetch")
    observer.on_enrichment_failure("arxiv_fetch")

    assert observer.metrics._enrichment_failures == 3
    assert observer.metrics._enrichment_failures_by_kind == {
        "website_fetch": 2,
        "arxiv_fetch": 1,
    }

    observer.metrics.write_checkpoint({"rest": 5000}, {})
    checkpoint = json.loads(observer._metrics_path.read_text().strip().splitlines()[-1])
    assert checkpoint["enrichment_failures_by_kind"] == {
        "website_fetch": 2,
        "arxiv_fetch": 1,
    }


def test_orchestrator_safety_event_dispatches_enrichment_failure_to_observer():
    pipeline = _make_pipeline()
    stub_observer = MagicMock()
    pipeline._observer = stub_observer

    pipeline._record_safety_event("enrichment_failure", {"kind": "arxiv_fetch"})

    stub_observer.on_enrichment_failure.assert_called_once_with("arxiv_fetch")
