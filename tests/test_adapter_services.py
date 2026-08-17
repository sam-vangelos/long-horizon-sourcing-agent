import asyncio
import importlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from github.schemas import GitHubProgress
from linkedin.browser import LinkedInBrowser
from linkedin.timing_telemetry import TIMING_EVENT_SCHEMAS
from shared.governor import (
    GovernorLimitReached,
    OperatorStopRequested,
    SessionExpired,
    UNGOVERNED_FOR_TESTS,
)
from shared.schemas import OpusDecision, Progress, SearchString
from tests.test_github_pipeline import _make_candidate, _make_pipeline as _make_github_pipeline, _make_query
from tests.test_linkedin_pipeline import _make_snippet

# Every LinkedIn fixture below is built on a brief whose linkedin_project_id is
# "test-project". A live run reaches a save from that project's own Recruiter
# page (`open_profile_by_url` opens a slide-in over the project search view), so
# that is what the default fixture page models. A page with no project id in its
# URL is NOT the neutral default it looks like — it is the F1 bypass condition,
# and only the tests that mean to exercise it set one.
_PROJECT_SEARCH_URL = (
    "https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
)
_PROJECT_PAGE_URL = f"{_PROJECT_SEARCH_URL}/profile/ada"
# Identity-drift fixtures swap the panel to a different person WITHOUT leaving
# the project, so the identity guard is the only thing that can explain a
# refusal in those tests.
_PROJECT_PAGE_URL_DRIFTED = f"{_PROJECT_SEARCH_URL}/profile/grace"
_PROJECTLESS_PAGE_URL = "https://www.linkedin.com/talent/recruiterSearch/profile/ada"
_FOREIGN_PROJECT_PAGE_URL = (
    "https://www.linkedin.com/talent/hire/999999999/discover/"
    "recruiterSearch/profile/ada"
)


def _make_linkedin_pipeline(output_dir: str):
    sys.modules.pop("linkedin.orchestrator", None)
    stubbed_llm = sys.modules.get("shared.llm_clients")
    if stubbed_llm is not None and not hasattr(stubbed_llm, "opus_llm"):
        sys.modules.pop("shared.llm_clients", None)
        importlib.import_module("shared.llm_clients")
    orchestrator_mod = importlib.import_module("linkedin.orchestrator")
    with patch.object(orchestrator_mod, "load_brief") as mock_brief, \
         patch.object(orchestrator_mod, "init_judger"), \
         patch.object(orchestrator_mod, "LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        brief.kit_url = ""
        brief.needs_preflight = MagicMock(return_value=False)
        mock_brief.return_value = brief

        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        pipeline = orchestrator_mod.Pipeline(
            brief_path=str(brief_path), output_dir=output_dir
        )
        # Park the fake browser on the brief's OWN project page. Left as a bare
        # MagicMock attribute, `browser.page.url` stringifies to "<MagicMock …>",
        # which names no Recruiter project — the F1 bypass condition, not a
        # neutral default. Tests that mean to exercise a projectless or foreign
        # page set `browser.page` themselves.
        pipeline.browser.page = MagicMock(url=_PROJECT_PAGE_URL)
        return pipeline


def test_github_acquisition_service_returns_terminal_geo_filter_result():
    pipeline = _make_github_pipeline()
    pipeline._runtime_run_id = None
    pipeline._ensure_services()
    pipeline._passes_geography_check = MagicMock(return_value=False)
    pipeline._prescreen_light = MagicMock(return_value="continue")
    pipeline._finish_preparation_terminal = MagicMock()
    pipeline._mark_terminal = MagicMock()
    pipeline._observer = MagicMock()

    candidate = _make_candidate("ada", "Ada Lovelace")
    query = _make_query(channel="code_search")
    progress = GitHubProgress(brief_name="test")
    enricher = MagicMock()
    enricher.light_enrich = AsyncMock(return_value=candidate)
    enricher.full_enrich = AsyncMock()

    result = asyncio.run(
        pipeline._acquisition_service.prepare_candidate_for_evaluation(
            enricher,
            "ada",
            query,
            progress,
        )
    )

    assert result.terminal_decision == "GEO_FILTERED"
    assert result.skip_reason == "light geography filter"
    pipeline._finish_preparation_terminal.assert_called_once()
    pipeline._mark_terminal.assert_called_once_with("ada")
    enricher.full_enrich.assert_not_awaited()


def test_github_work_unit_service_processes_graph_expansion_queue():
    pipeline = _make_github_pipeline()
    pipeline._runtime_run_id = None
    pipeline._ensure_services()
    pipeline._observer = MagicMock()
    query = _make_query(id=1)
    progress = GitHubProgress(
        brief_name="test",
        queries=[query],
        graph_expansion_queue=[
            {
                "username": "seed-user",
                "reason": "SAVE",
                "confidence": 0.92,
                "capability_area": "research",
                "added_at": "2026-01-01T00:00:00+00:00",
            }
        ],
        graph_expansion_processed=[],
    )

    asyncio.run(pipeline._work_unit_service.process_graph_expansion_queue(progress, progress.queries))

    assert len(progress.queries) == 2
    assert progress.queries[1].channel == "graph_expansion"
    pipeline._observer.on_graph_expansion_processed.assert_called_once()


def test_github_side_effects_service_handles_save_and_outreach():
    pipeline = _make_github_pipeline()
    pipeline._runtime_run_id = None
    pipeline._ensure_services()
    pipeline._observer = MagicMock()
    candidate = _make_candidate("ada", "Ada Lovelace")
    query = _make_query(name="Seed Query", channel="user_search")
    progress = GitHubProgress(brief_name="test")
    decision = OpusDecision(
        stage="full",
        decision="SAVE",
        path="research_builder",
        confidence=0.95,
        rationale="Strong fit",
        candidate_name="Ada Lovelace",
        profile_url="https://github.com/ada",
    )
    envelope = pipeline._execution_envelope(
        username="ada",
        query=query,
        result_rank=1,
        candidate=candidate,
        metadata={"candidate_record": {"username": "ada"}},
    )

    with patch("github.side_effects.generate_outreach", AsyncMock(return_value={"message": "hello"})), patch(
        "github.side_effects.extract_priority_rank",
        return_value=1,
    ):
        outcome = asyncio.run(
            pipeline._side_effects_service.handle_full_decision(
                username="ada",
                candidate=candidate,
                query=query,
                progress=progress,
                full_decision=decision,
                envelope=envelope,
                full_attempt_id=None,
            )
        )

    assert outcome.status == "succeeded"
    assert pipeline.stats["saved"] == 1
    assert progress.candidates_saved == 1
    assert query.saves == ["Ada Lovelace"]
    assert len(progress.graph_expansion_queue) == 1


def test_github_side_effects_service_skips_duplicate_outreach():
    pipeline = _make_github_pipeline()
    pipeline._runtime_run_id = None
    pipeline._ensure_services()
    pipeline._observer = MagicMock()
    pipeline._execution_engine = MagicMock()
    pipeline._execution_engine.runtime = MagicMock()
    pipeline._execution_engine.runtime.begin_candidate_side_effect = MagicMock(
        return_value={"should_execute": False, "side_effect": {"status": "succeeded"}}
    )
    pipeline._execution_engine.runtime.record_side_effect_result = MagicMock()
    candidate = _make_candidate("ada", "Ada Lovelace")
    query = _make_query(name="Seed Query", channel="user_search")
    progress = GitHubProgress(brief_name="test")
    decision = OpusDecision(
        stage="full",
        decision="SAVE",
        path="research_builder",
        confidence=0.95,
        rationale="Strong fit",
        candidate_name="Ada Lovelace",
        profile_url="https://github.com/ada",
    )
    envelope = pipeline._execution_envelope(
        username="ada",
        query=query,
        result_rank=1,
        candidate=candidate,
        metadata={"candidate_record": {"username": "ada"}},
    )
    envelope = type(envelope)(**{**envelope.__dict__, "run_id": 7})

    with patch("github.side_effects.generate_outreach", AsyncMock(return_value={"message": "hello"})), patch(
        "github.side_effects.extract_priority_rank",
        return_value=1,
    ):
        outcome = asyncio.run(
            pipeline._side_effects_service.handle_full_decision(
                username="ada",
                candidate=candidate,
                query=query,
                progress=progress,
                full_decision=decision,
                envelope=envelope,
                full_attempt_id=11,
            )
        )

    assert outcome.status == "succeeded"
    pipeline._execution_engine.runtime.record_side_effect_result.assert_called_once()


def test_linkedin_acquisition_service_extracts_dom_enriched_snippet():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline._ensure_services()
        pipeline.browser.focus_card_for_review = AsyncMock()
        pipeline.browser.get_card_snapshot = AsyncMock(
            return_value={
                "innertext": "Select Ada Lovelace\nAda Lovelace\nML Engineer\nActivity 9 messages · In 3 projects · 3 views\nSaved by Sam Vangelos on April 11, 2026",
                "name": "Ada Lovelace",
                "url": "/talent/profile/ada",
                "already_saved": True,
                "recruiter_activity": {
                    "message_count": 9,
                    "project_count": 3,
                    "view_count": 3,
                    "saved_by": "Sam Vangelos",
                    "raw_activity_text": "Activity 9 messages · In 3 projects · 3 views | Saved by Sam Vangelos on April 11, 2026",
                },
            }
        )
        search_string = SearchString(id=9, name="seq", boolean="(test)")

        with patch(
            "linkedin.acquisition.extract_snippet_from_card_innertext",
            return_value=_make_snippet(
                name="Ada Lovelace",
                profile_url="",
                source_string_id=9,
                source_string_name="seq",
                page=2,
                result_rank=3,
            ),
        ), patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
            result = asyncio.run(
                pipeline._acquisition_service.extract_card_snippet(search_string, page_num=2, card_index=2)
            )

        assert result is not None
        assert result.snippet.name == "Ada Lovelace"
        assert result.snippet.profile_url == "/talent/profile/ada"
        assert result.snippet.already_saved is True
        assert result.snippet.recruiter_activity is not None
        assert result.snippet.recruiter_activity.message_count == 9
        assert result.snippet.novelty_pressure == "high"


def test_linkedin_save_rechecks_profile_identity_at_browser_click_boundary():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline.test_mode = False
        pipeline._ensure_services()
        snippet = _make_snippet(profile_url="/talent/profile/ada")
        search_string = SearchString(id=1, name="test", boolean="foo")
        pipeline.browser.page = MagicMock(url=_PROJECT_PAGE_URL)
        pipeline.browser.current_profile_identity_fragment.side_effect = (
            lambda: LinkedInBrowser._profile_url_fragment(
                pipeline.browser.page.url
            )
        )

        async def probe(_identity_fragment):
            return False

        pipeline.browser.is_already_saved_on_card = AsyncMock(
            side_effect=probe
        )

        async def click_after_identity_drift(
            *,
            before_click,
            **_kwargs,
        ):
            pipeline.browser.page.url = _PROJECT_PAGE_URL_DRIFTED
            before_click()
            return True

        pipeline.browser.save_candidate = AsyncMock(
            side_effect=click_after_identity_drift
        )
        pipeline.browser.scroll_for_linger = AsyncMock(return_value=0)
        pipeline.browser.scroll_restore = AsyncMock()

        with patch(
            "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MIN_SECONDS",
            0,
        ), patch(
            "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MAX_SECONDS",
            0,
        ), patch(
            "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_BASE_SECONDS",
            0,
        ), patch(
            "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MIN_CHUNKS_BACK",
            0,
        ), patch(
            "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MAX_CHUNKS_BACK",
            0,
        ):
            with pytest.raises(RuntimeError, match="profile identity"):
                asyncio.run(
                    pipeline._side_effects_service.handle_save_decision(
                        snippet=snippet,
                        runtime_search_string=search_string,
                        attempt_id=None,
                    )
                )

        pipeline.browser.save_candidate.assert_awaited_once()


def test_linkedin_acquisition_service_delegates_profile_open_to_browser():
    # P8.1: governance (check + count) now lives inside
    # LinkedInBrowser.open_profile_by_url() itself, not at this call site —
    # acquisition just has to delegate to the browser. pipeline.browser here
    # is a mock (LinkedInBrowser is patched in _make_linkedin_pipeline), so
    # the real governed implementation isn't exercised by this test; that is
    # covered at the browser level (tests/test_linkedin_browser_governor.py).
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline._ensure_services()
        snippet = _make_snippet()
        snippet.card_index = 0
        pipeline.browser.ensure_card_rendered = AsyncMock()
        pipeline.browser.open_profile_by_url = AsyncMock(return_value=None)
        pipeline.browser.simulate_profile_read = AsyncMock(return_value=None)
        pipeline.browser.get_profile_innertext = AsyncMock(return_value="experience text")
        pipeline._ensure_browser_healthy = AsyncMock()

        with patch("linkedin.acquisition.extract_profile_from_dom", return_value=MagicMock()):
            asyncio.run(pipeline._acquisition_service.extract_profile_summary(snippet))

        pipeline.browser.open_profile_by_url.assert_awaited_once_with(snippet.profile_url)


def test_expansion_precedes_the_profile_read():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline._ensure_services()
        snippet = _make_snippet()
        snippet.card_index = 0
        pipeline.browser.ensure_card_rendered = AsyncMock()
        pipeline.browser.open_profile_by_url = AsyncMock(return_value=None)
        call_order = []

        async def track_expand():
            call_order.append("expand")

        async def track_read(interest: float = 0.5):
            call_order.append("read")

        async def track_innertext():
            call_order.append("innertext")
            return "experience text"

        pipeline.browser.expand_profile_sections = AsyncMock(side_effect=track_expand)
        pipeline.browser.simulate_profile_read = AsyncMock(side_effect=track_read)
        pipeline.browser.get_profile_innertext = AsyncMock(side_effect=track_innertext)
        pipeline._ensure_browser_healthy = AsyncMock()

        with patch(
            "linkedin.acquisition.config.LINKEDIN_SECTION_DIRECTED_READ_ENABLED", True
        ), patch(
            "linkedin.acquisition.extract_profile_from_dom", return_value=MagicMock()
        ), patch("linkedin.acquisition.asyncio.sleep", new=AsyncMock()):
            asyncio.run(pipeline._acquisition_service.extract_profile_summary(snippet))

        assert call_order.index("expand") < call_order.index("read")
        pipeline.browser.expand_profile_sections.assert_awaited_once()


def test_acquisition_forwards_interest_to_the_profile_read():
    """The interest hint must survive the acquisition hop, not just be accepted.

    Without this, dropping the argument at the `simulate_profile_read` call
    leaves every band green while the read budget silently reverts to a
    constant — the orchestrator-level tests assert only against a mocked
    acquisition service and never see the browser.
    """
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline._ensure_services()
        snippet = _make_snippet()
        snippet.card_index = 0
        pipeline.browser.ensure_card_rendered = AsyncMock()
        pipeline.browser.open_profile_by_url = AsyncMock(return_value=None)
        pipeline.browser.expand_profile_sections = AsyncMock()
        pipeline.browser.simulate_profile_read = AsyncMock(return_value=None)
        pipeline.browser.get_profile_innertext = AsyncMock(return_value="experience")
        pipeline._ensure_browser_healthy = AsyncMock()

        with patch(
            "linkedin.acquisition.config.LINKEDIN_SECTION_DIRECTED_READ_ENABLED", True
        ), patch(
            "linkedin.acquisition.extract_profile_from_dom", return_value=MagicMock()
        ), patch("linkedin.acquisition.asyncio.sleep", new=AsyncMock()):
            asyncio.run(
                pipeline._acquisition_service.extract_profile_summary(
                    snippet, interest=0.9
                )
            )

        pipeline.browser.simulate_profile_read.assert_awaited_once_with(0.9)


def test_expansion_skipped_when_section_directed_read_disabled():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline._ensure_services()
        snippet = _make_snippet()
        snippet.card_index = 0
        pipeline.browser.ensure_card_rendered = AsyncMock()
        pipeline.browser.open_profile_by_url = AsyncMock(return_value=None)
        call_order = []

        async def track_read(interest: float = 0.5):
            call_order.append("read")

        async def track_innertext():
            call_order.append("innertext")
            return "experience text"

        pipeline.browser.expand_profile_sections = AsyncMock()
        pipeline.browser.simulate_profile_read = AsyncMock(side_effect=track_read)
        pipeline.browser.get_profile_innertext = AsyncMock(side_effect=track_innertext)
        pipeline._ensure_browser_healthy = AsyncMock()

        with patch(
            "linkedin.acquisition.config.LINKEDIN_SECTION_DIRECTED_READ_ENABLED", False
        ), patch(
            "linkedin.acquisition.extract_profile_from_dom", return_value=MagicMock()
        ), patch("linkedin.acquisition.asyncio.sleep", new=AsyncMock()):
            asyncio.run(pipeline._acquisition_service.extract_profile_summary(snippet))

        pipeline.browser.expand_profile_sections.assert_not_awaited()
        assert call_order == ["read", "innertext"]


def test_linkedin_acquisition_service_reraises_governor_limit_without_name_fallback():
    # P8.1: when the real governed browser.open_profile_by_url() raises
    # GovernorLimitReached, acquisition must propagate it rather than treat
    # it as a URL-open failure and fall back to open_profile(name) — the
    # acquisition-layer except/raise structure is the thing being preserved
    # even though the check/record calls that used to sit here are gone.
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline._ensure_services()
        snippet = _make_snippet()
        snippet.card_index = 0
        pipeline.browser.ensure_card_rendered = AsyncMock()
        pipeline.browser.open_profile_by_url = AsyncMock(
            side_effect=GovernorLimitReached("24h_profile_cap (400/400)")
        )
        pipeline.browser.open_profile = AsyncMock()
        pipeline._ensure_browser_healthy = AsyncMock()

        raised = None
        try:
            asyncio.run(pipeline._acquisition_service.extract_profile_summary(snippet))
        except GovernorLimitReached as exc:
            raised = exc

        assert raised is not None
        pipeline.browser.open_profile.assert_not_called()


def test_linkedin_acquisition_service_never_name_falls_back_from_stable_url():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline._ensure_services()
        snippet = _make_snippet()
        snippet.card_index = 0
        pipeline.browser.ensure_card_rendered = AsyncMock()
        pipeline.browser.open_profile_by_url = AsyncMock(
            side_effect=RuntimeError("exact Recruiter identity not found")
        )
        pipeline.browser.open_profile = AsyncMock()
        pipeline._ensure_browser_healthy = AsyncMock()

        with pytest.raises(RuntimeError, match="exact Recruiter identity"):
            asyncio.run(
                pipeline._acquisition_service.extract_profile_summary(snippet)
            )

        pipeline.browser.open_profile.assert_not_awaited()


def test_linkedin_work_unit_service_restarts_runtime_string_and_reloads_state():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline._ensure_services()
        pipeline._runtime_run_id = 42
        pipeline._runtime_bridge = MagicMock()
        pipeline._runtime_bridge.load_search_memory.return_value = {"families": {}}
        pipeline._seen_urls = {"a"}
        pipeline._in_flight_urls = {"b"}
        pipeline._prior_outcomes = {"a": "SAVE"}
        pipeline._load_candidate_history = MagicMock()

        progress = Progress(brief_name="test", strings=[SearchString(id=5, name="test", boolean="foo")])
        pipeline._work_unit_service.restart_string(progress, 5)

        pipeline._runtime_bridge.restart_string.assert_called_once()
        pipeline._load_candidate_history.assert_called_once()
        assert pipeline._seen_urls == set()
        assert pipeline._in_flight_urls == set()
        assert pipeline._prior_outcomes == {}


def test_linkedin_side_effects_service_records_test_mode_save():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline.test_mode = True
        pipeline._ensure_services()
        pipeline._runtime_run_id = 7
        pipeline._runtime_bridge = MagicMock()
        snippet = _make_snippet()
        search_string = SearchString(id=1, name="test", boolean="foo")

        outcome = asyncio.run(
            pipeline._side_effects_service.handle_save_decision(
                snippet=snippet,
                runtime_search_string=search_string,
                attempt_id=11,
            )
        )

        assert outcome.status == "succeeded"
        assert pipeline.stats["saved"] == 1
        pipeline._runtime_bridge.record_side_effect_result.assert_called_once()


def test_linkedin_failed_save_is_honest_end_to_end():
    """P1 exit gate: judge SAVE + browser save returns False.

    The ledger row must be failed (attempt 1) carrying the browser's
    failure classification; stats must bucket save_failed (not saved);
    the honest candidate_saved event must map to the "Save failed" live
    label; and the next begin (resume/rediscovery) must retry.
    """

    from cloris.live_signal import _event_summary
    from shared.runtime_state.store import RuntimeStateStore

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline.test_mode = False
        pipeline._ensure_services()

        store = RuntimeStateStore(Path(td) / "runtime_state.sqlite3")
        run_id = store.start_run(
            source="linkedin",
            brief_id="test-project",
            output_dir=td,
            mode="fresh",
        )
        store.record_candidate_discovery(
            run_id=run_id,
            work_unit_id=None,
            source="linkedin",
            brief_id="test-project",
            identity_key="/talent/profile/ada",
            display_name="Ada",
            profile_url="/talent/profile/ada",
        )

        class _StoreBridge:
            """Minimal real-ledger bridge exposing the service's call shape."""

            def begin_candidate_side_effect(self, *, run_id, search_string, snippet, attempt_id, effect_type, idempotency_key, payload=None):
                return store.begin_candidate_side_effect(
                    run_id=run_id,
                    source="linkedin",
                    brief_id="test-project",
                    identity_key=snippet.profile_url,
                    attempt_id=attempt_id,
                    effect_type=effect_type,
                    idempotency_key=idempotency_key,
                    payload=payload,
                )

            def complete_candidate_side_effect(self, *, side_effect_id, status, payload=None):
                store.complete_candidate_side_effect(
                    side_effect_id=side_effect_id, status=status, payload=payload
                )

            def record_side_effect_result(self, **kwargs):
                pass

        pipeline._runtime_run_id = run_id
        pipeline._runtime_bridge = _StoreBridge()
        pipeline.browser.current_profile_identity_fragment.return_value = "ada"
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=False)

        async def failed_save_candidate(**_kwargs):
            pipeline.browser._last_save_failure_reason = "save_not_persisted"
            return False

        pipeline.browser.save_candidate = AsyncMock(
            side_effect=failed_save_candidate
        )
        pipeline.browser.scroll_for_linger = AsyncMock(return_value=0)
        pipeline.browser.scroll_restore = AsyncMock()
        snippet = _make_snippet()
        snippet.profile_url = "/talent/profile/ada"
        search_string = SearchString(id=1, name="test", boolean="foo")

        with patch("linkedin.side_effects.asyncio.sleep", new=AsyncMock()), \
             patch("linkedin.side_effects.human_delay_correlated", return_value=0.0):
            outcome = asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=search_string,
                    attempt_id=None,
                )
            )

        # Outcome is honest.
        assert outcome.status == "failed"
        assert outcome.payload["failure_reason"] == "save_not_persisted"

        # Stats bucket save_failed, never saved.
        assert pipeline.stats.get("saved", 0) == 0
        assert pipeline.stats["save_failed"] == 1

        # Ledger row: failed, attempt 1, failure reason persisted.
        import json as _json

        rows = store.list_candidate_side_effects(
            source="linkedin", brief_id="test-project"
        )
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert int(rows[0]["attempt_count"]) == 1
        assert _json.loads(rows[0]["payload_json"])["failure_reason"] == (
            "save_not_persisted"
        )

        # Live-signal label: the honest event maps to the failed label.
        summary = _event_summary(
            "candidate_saved",
            {"name": "Ada", "linkedin_save": False, "ts": "2026-07-02T00:00:00Z"},
        )
        assert summary["label"] == "Save failed — will retry"
        succeeded_summary = _event_summary(
            "candidate_saved",
            {"name": "Ada", "linkedin_save": True, "ts": "2026-07-02T00:00:00Z"},
        )
        assert succeeded_summary["label"] == "Saved a candidate"

        # On resume/rediscovery the save is retried (attempt 2).
        retry = store.begin_candidate_side_effect(
            run_id=run_id,
            source="linkedin",
            brief_id="test-project",
            identity_key="/talent/profile/ada",
            attempt_id=None,
            effect_type="linkedin_save",
            idempotency_key="save",
            payload={"search_string_id": 1},
        )
        assert retry["should_execute"] is True
        assert int(retry["side_effect"]["attempt_count"]) == 2


def _real_save_ledger_pipeline(td, *profile_urls):
    from shared.runtime_state.store import RuntimeStateStore

    pipeline = _make_linkedin_pipeline(td)
    pipeline.test_mode = False
    pipeline._ensure_services()
    store = RuntimeStateStore(Path(td) / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-project",
        output_dir=td,
        mode="fresh",
    )
    for profile_url in profile_urls:
        store.record_candidate_discovery(
            run_id=run_id,
            work_unit_id=None,
            source="linkedin",
            brief_id="test-project",
            identity_key=profile_url,
            display_name=profile_url.rsplit("/", 1)[-1],
            profile_url=profile_url,
        )

    class _StoreBridge:
        def begin_candidate_side_effect(self, *, run_id, search_string, snippet, attempt_id, effect_type, idempotency_key, payload=None):
            return store.begin_candidate_side_effect(
                run_id=run_id,
                source="linkedin",
                brief_id="test-project",
                identity_key=snippet.profile_url,
                attempt_id=attempt_id,
                effect_type=effect_type,
                idempotency_key=idempotency_key,
                payload=payload,
            )

        def complete_candidate_side_effect(self, *, side_effect_id, status, payload=None):
            store.complete_candidate_side_effect(
                side_effect_id=side_effect_id, status=status, payload=payload
            )

        def record_side_effect_result(self, **kwargs):
            pass

    pipeline._runtime_run_id = run_id
    pipeline._runtime_bridge = _StoreBridge()
    pipeline.browser.current_profile_identity_fragment.return_value = (
        LinkedInBrowser._profile_url_fragment(profile_urls[0])
    )
    return pipeline, store


@pytest.mark.parametrize(
    "stop_error",
    [
        OperatorStopRequested(),
        SessionExpired(),
        GovernorLimitReached("24h_profile_cap"),
    ],
    ids=["operator_stop", "session_expired", "governor_limit"],
)
def test_linkedin_stop_before_save_leaves_pending_receipt(stop_error):
    import json as _json

    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        pipeline, store = _real_save_ledger_pipeline(td, profile_url)
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=False)
        pipeline.browser.save_candidate = AsyncMock()
        stats_before = dict(pipeline.stats)
        stop_check = MagicMock(side_effect=stop_error)
        pipeline._side_effects_service.deps.before_irreversible_side_effect = (
            stop_check
        )
        snippet = _make_snippet()
        snippet.profile_url = profile_url

        with patch("linkedin.side_effects.log_event") as event_log, \
             pytest.raises(type(stop_error)) as raised:
            asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=SearchString(
                        id=1, name="test", boolean="foo"
                    ),
                    attempt_id=None,
                )
            )

        assert raised.value is stop_error
        stop_check.assert_called_once_with()
        pipeline.browser.save_candidate.assert_not_awaited()
        rows = store.list_candidate_side_effects(
            source="linkedin", brief_id="test-project"
        )
        assert len(rows) == 1
        assert rows[0]["status"] == "pending"
        assert int(rows[0]["attempt_count"]) == 1
        assert "failure_reason" not in _json.loads(rows[0]["payload_json"])
        assert pipeline.stats == stats_before
        assert pipeline.stats.get("save_failed", 0) == 0
        event_log.assert_not_called()


def test_linkedin_save_refuses_a_foreign_project_page_before_the_click():
    """E4: the pre-save boundary revalidates the Recruiter PROJECT, not just the
    profile identity. A mid-run project change (operator tab switch, recovery
    re-bind onto another project's tab) must abort before the irreversible save
    click instead of filing the candidate into the wrong pipeline.

    Drives the real production wiring: Pipeline._ensure_services installs
    Pipeline._honor_irreversible_side_effect_boundary as
    LinkedInSideEffectsDeps.before_irreversible_side_effect, and
    handle_save_decision calls it from before_save_click().
    """
    import json as _json

    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        pipeline, store = _real_save_ledger_pipeline(td, profile_url)
        # _make_linkedin_pipeline re-imports linkedin.orchestrator, so the class
        # must be read from the module the pipeline under test actually uses.
        ProjectContextMismatchError = sys.modules[
            "linkedin.orchestrator"
        ].ProjectContextMismatchError
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=False)
        pipeline.browser.save_candidate = AsyncMock()
        # Brief project id is "test-project"; the live page is another project.
        pipeline.browser.page = MagicMock(
            url=(
                "https://www.linkedin.com/talent/hire/9999999999/discover/"
                "recruiterSearch/profile/AEMAAA?start=25"
            )
        )
        snippet = _make_snippet()
        snippet.profile_url = profile_url

        with pytest.raises(ProjectContextMismatchError):
            asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=SearchString(
                        id=1, name="test", boolean="foo"
                    ),
                    attempt_id=None,
                )
            )

        pipeline.browser.save_candidate.assert_not_awaited()
        assert pipeline.stats.get("saved", 0) == 0
        rows = store.list_candidate_side_effects(
            source="linkedin", brief_id="test-project"
        )
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert "project" in _json.loads(rows[0]["payload_json"])["failure_reason"]


def test_linkedin_save_proceeds_on_the_brief_project_page():
    """E4 counterpart: the brief's own project page still saves — the new
    project revalidation must not refuse the normal path.
    """
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        pipeline, store = _real_save_ledger_pipeline(td, profile_url)
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=False)
        pipeline.browser.save_candidate = AsyncMock(return_value=True)
        pipeline.browser.scroll_for_linger = AsyncMock(return_value=0)
        pipeline.browser.scroll_restore = AsyncMock()
        pipeline.browser.page = MagicMock(
            url=(
                "https://www.linkedin.com/talent/hire/test-project/discover/"
                "recruiterSearch/profile/AEMAAA?start=25"
            )
        )
        snippet = _make_snippet()
        snippet.profile_url = profile_url

        outcome = asyncio.run(
            pipeline._side_effects_service.handle_save_decision(
                snippet=snippet,
                runtime_search_string=SearchString(id=1, name="test", boolean="foo"),
                attempt_id=None,
            )
        )

        assert outcome.status == "succeeded"
        pipeline.browser.save_candidate.assert_awaited_once()


def test_linkedin_probe_failure_ignores_previous_candidates_save_reason():
    import json as _json

    with tempfile.TemporaryDirectory() as td:
        profile_a = "/talent/profile/ada"
        profile_b = "/talent/profile/grace"
        pipeline, store = _real_save_ledger_pipeline(td, profile_a, profile_b)
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=False)

        async def save_after_internal_retry(**kwargs):
            pipeline.browser._last_save_failure_reason = "save_trigger_not_found"
            return True

        pipeline.browser.save_candidate = AsyncMock(
            side_effect=save_after_internal_retry
        )
        pipeline.browser.scroll_for_linger = AsyncMock(return_value=0)
        pipeline.browser.scroll_restore = AsyncMock()
        search_string = SearchString(id=1, name="test", boolean="foo")
        snippet_a = _make_snippet()
        snippet_a.profile_url = profile_a

        with patch("linkedin.side_effects.asyncio.sleep", new=AsyncMock()), \
             patch("linkedin.side_effects.human_delay_correlated", return_value=0.0):
            outcome = asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet_a,
                    runtime_search_string=search_string,
                    attempt_id=None,
                )
            )

        assert outcome.status == "succeeded"
        assert pipeline.browser._last_save_failure_reason == "save_trigger_not_found"

        pipeline.browser.is_already_saved_on_card = AsyncMock(
            side_effect=RuntimeError("browser context closed")
        )
        snippet_b = _make_snippet()
        snippet_b.profile_url = profile_b
        pipeline.browser.current_profile_identity_fragment.return_value = "grace"
        with pytest.raises(RuntimeError, match="browser context closed"):
            asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet_b,
                    runtime_search_string=search_string,
                    attempt_id=None,
                )
            )

        rows = store.list_candidate_side_effects(
            source="linkedin", brief_id="test-project"
        )
        assert [row["status"] for row in rows] == ["succeeded", "failed"]
        failure_reason = _json.loads(rows[1]["payload_json"])["failure_reason"]
        assert failure_reason == "RuntimeError: browser context closed"
        assert failure_reason != "save_trigger_not_found"


def test_linkedin_already_present_counts_as_already_present_not_saved():
    """P1.2: 'already in pipeline' must not inflate stats['saved']."""

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline.test_mode = False
        pipeline._ensure_services()
        pipeline._runtime_run_id = None
        pipeline._runtime_bridge = None
        pipeline.browser.current_profile_identity_fragment.return_value = "test123"
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=True)
        snippet = _make_snippet()
        search_string = SearchString(id=1, name="test", boolean="foo")

        outcome = asyncio.run(
            pipeline._side_effects_service.handle_save_decision(
                snippet=snippet,
                runtime_search_string=search_string,
                attempt_id=None,
            )
        )

        assert outcome.status == "succeeded"
        assert outcome.payload["already_present"] is True
        assert pipeline.stats.get("saved", 0) == 0
        assert pipeline.stats["already_present"] == 1


def test_linkedin_already_present_probe_rejects_identity_drift():
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        pipeline, store = _real_save_ledger_pipeline(td, profile_url)
        pipeline.browser.page = MagicMock(url=_PROJECT_PAGE_URL)
        pipeline.browser.current_profile_identity_fragment.side_effect = (
            lambda: LinkedInBrowser._profile_url_fragment(
                pipeline.browser.page.url
            )
        )

        async def drift_during_probe(_identity_fragment):
            pipeline.browser.page.url = _PROJECT_PAGE_URL_DRIFTED
            return True

        pipeline.browser.is_already_saved_on_card = AsyncMock(
            side_effect=drift_during_probe
        )
        pipeline.browser.save_candidate = AsyncMock()
        snippet = _make_snippet()
        snippet.profile_url = profile_url

        with pytest.raises(RuntimeError, match="profile identity"):
            asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=SearchString(
                        id=1,
                        name="test",
                        boolean="foo",
                    ),
                    attempt_id=None,
                )
            )

        pipeline.browser.save_candidate.assert_not_awaited()
        rows = store.list_candidate_side_effects(
            source="linkedin",
            brief_id="test-project",
        )
        assert [row["status"] for row in rows] == ["failed"]


@pytest.mark.parametrize("confirmation_poll", [1, 2], ids=["first", "retry"])
def test_save_candidate_rejects_identity_drift_during_confirmation(
    confirmation_poll,
):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock(
        url="https://www.linkedin.com/talent/recruiterSearch/profile/ada"
    )
    page.wait_for_timeout = AsyncMock()
    browser._page = page
    events = []

    async def transaction(_identity_fragment, *, dry_run):
        events.append("resolve" if dry_run else "dispatch")
        return {"ok": True, "dispatched": not dry_run}

    browser._card_save_transaction = AsyncMock(side_effect=transaction)
    probe_count = 0

    async def probe(_identity_fragment):
        nonlocal probe_count
        probe_count += 1
        if probe_count == confirmation_poll + 1:
            page.url = (
                "https://www.linkedin.com/talent/recruiterSearch/profile/grace"
            )
            return True
        return False

    browser.is_already_saved_on_card = AsyncMock(side_effect=probe)

    async def retry_once(coro_fn, *_args, **_kwargs):
        return await coro_fn()

    with patch("linkedin.browser._retry", side_effect=retry_once):
        with pytest.raises(RuntimeError, match="profile identity"):
            asyncio.run(browser.save_candidate())

    assert events == ["resolve", "dispatch"]

def test_linkedin_save_probe_failure_propagates_before_click():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline.test_mode = False
        pipeline._ensure_services()
        pipeline._runtime_run_id = 7
        pipeline._runtime_bridge = MagicMock()
        pipeline._runtime_bridge.begin_candidate_side_effect.return_value = {
            "should_execute": True,
            "side_effect": {"id": 11, "status": "pending"},
        }
        pipeline.browser.is_already_saved_on_card = AsyncMock(
            side_effect=RuntimeError("browser context closed")
        )
        pipeline.browser.save_candidate = AsyncMock()
        pipeline.browser.current_profile_identity_fragment.return_value = "test123"
        pipeline.browser._last_save_failure_reason = None

        with pytest.raises(RuntimeError, match="browser context closed"):
            asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=_make_snippet(),
                    runtime_search_string=SearchString(
                        id=1,
                        name="test",
                        boolean="foo",
                    ),
                    attempt_id=None,
                )
            )

        pipeline.browser.save_candidate.assert_not_awaited()
        pipeline._runtime_bridge.complete_candidate_side_effect.assert_called_once()
        complete_call = (
            pipeline._runtime_bridge.complete_candidate_side_effect.call_args.kwargs
        )
        assert complete_call["status"] == "failed"
        assert "browser context closed" in complete_call["payload"]["failure_reason"]
        record_call = (
            pipeline._runtime_bridge.record_side_effect_result.call_args.kwargs
        )
        assert record_call["status"] == "failed"
        assert "browser context closed" in record_call["payload"]["failure_reason"]


def test_linkedin_stop_authority_is_rechecked_immediately_before_save_click():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline.test_mode = False
        pipeline._ensure_services()
        pipeline._runtime_run_id = None
        pipeline._runtime_bridge = None
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=False)
        pipeline.browser.current_profile_identity_fragment.return_value = "test123"
        pipeline.browser.save_candidate = AsyncMock(return_value=True)
        stop_check = MagicMock(side_effect=OperatorStopRequested())
        pipeline._side_effects_service.deps.before_irreversible_side_effect = (
            stop_check
        )

        with pytest.raises(OperatorStopRequested):
            asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=_make_snippet(),
                    runtime_search_string=SearchString(
                        id=1,
                        name="test",
                        boolean="foo",
                    ),
                    attempt_id=None,
                )
            )

        stop_check.assert_called_once_with()
        pipeline.browser.save_candidate.assert_not_awaited()


def test_linkedin_save_rechecks_stop_after_trigger_resolution_before_click():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock(
        url="https://www.linkedin.com/talent/recruiterSearch/profile/test123"
    )
    page.wait_for_timeout = AsyncMock()
    browser._page = page
    browser.is_already_saved_on_card = AsyncMock(return_value=False)
    stop_state = {"requested": False}
    events = []

    async def transaction(_identity_fragment, *, dry_run):
        if dry_run:
            events.append("resolve")
            stop_state["requested"] = True
            return {"ok": True, "dispatched": False}
        events.append("dispatch")
        return {"ok": True, "dispatched": True}

    def stop_check():
        events.append("guard")
        if stop_state["requested"]:
            raise OperatorStopRequested()

    browser._card_save_transaction = AsyncMock(side_effect=transaction)

    with pytest.raises(BaseException) as exc_info:
        asyncio.run(browser.save_candidate(before_click=stop_check))

    assert isinstance(exc_info.value.cause, OperatorStopRequested)
    assert events == ["resolve", "guard"]

def test_linkedin_side_effects_service_skips_duplicate_ledger_entry():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline.test_mode = False
        pipeline._ensure_services()
        pipeline._runtime_run_id = 7
        pipeline._runtime_bridge = MagicMock()
        pipeline._runtime_bridge.begin_candidate_side_effect.return_value = {
            "should_execute": False,
            "side_effect": {"status": "succeeded"},
        }
        snippet = _make_snippet()
        search_string = SearchString(id=1, name="test", boolean="foo")

        outcome = asyncio.run(
            pipeline._side_effects_service.handle_save_decision(
                snippet=snippet,
                runtime_search_string=search_string,
                attempt_id=11,
            )
        )

        assert outcome.status == "skipped"
        pipeline.browser.is_already_saved_on_card.assert_not_called()


@pytest.mark.parametrize("save_result", [False, True], ids=["failed", "succeeded"])
def test_replayed_failed_save_receipt_does_not_use_saved_history_shortcut(
    save_result,
):
    """Terminal SAVE history cannot suppress retrying its failed receipt."""
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        pipeline, store = _real_save_ledger_pipeline(td, profile_url)
        snippet = _make_snippet(profile_url=profile_url)
        search_string = SearchString(id=1, name="test", boolean="foo")
        started = store.begin_candidate_side_effect(
            run_id=pipeline._runtime_run_id,
            source="linkedin",
            brief_id="test-project",
            identity_key=profile_url,
            attempt_id=None,
            effect_type="linkedin_save",
            idempotency_key="save",
            payload={"search_string_id": 1},
        )
        store.complete_candidate_side_effect(
            side_effect_id=int(started["side_effect"]["id"]),
            status="failed",
            payload={"failure_reason": "save_not_persisted"},
        )
        pipeline._saved_urls.add(profile_url)
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=False)
        pipeline.browser.save_candidate = AsyncMock(return_value=save_result)
        pipeline.browser.scroll_for_linger = AsyncMock(return_value=0)
        pipeline.browser.scroll_restore = AsyncMock()

        with patch("linkedin.side_effects.asyncio.sleep", new=AsyncMock()), \
             patch("linkedin.side_effects.human_delay_correlated", return_value=0):
            outcome = asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=search_string,
                    attempt_id=None,
                )
            )

        pipeline.browser.is_already_saved_on_card.assert_awaited_once()
        pipeline.browser.save_candidate.assert_awaited_once()
        assert outcome.status == ("succeeded" if save_result else "failed")
        receipt = store.list_candidate_side_effects(
            source="linkedin",
            brief_id="test-project",
        )[0]
        assert receipt["status"] == outcome.status


def _card_snapshot_browser_with_crash(probe_position):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    article = MagicMock()
    article.wait_for = AsyncMock()
    article.inner_text = AsyncMock(return_value="Ada\nEngineer")
    li = MagicMock()
    li.locator.return_value.first = article
    browser._wait_for_result_slot = AsyncMock(return_value=li)

    primary = MagicMock()
    primary.inner_text = AsyncMock(return_value="Ada")
    primary.get_attribute = AsyncMock(return_value="/talent/profile/ada")
    fallback = MagicMock()
    fallback.inner_text = AsyncMock(return_value="Ada")
    fallback.get_attribute = AsyncMock(return_value="/talent/profile/ada")
    saved = MagicMock()
    saved.is_visible = AsyncMock(return_value=False)
    if probe_position == "primary_identity":
        primary.get_attribute.side_effect = RuntimeError("Page crashed")
    elif probe_position == "fallback_identity":
        primary.inner_text.side_effect = RuntimeError("ordinary primary miss")
        fallback.inner_text.side_effect = RuntimeError("Page crashed")
    else:
        saved.is_visible.side_effect = RuntimeError("Page crashed")

    def locate(selector):
        located = MagicMock()
        if "lockup__title" in selector:
            located.first = primary
        elif "/talent/profile/" in selector:
            located.first = fallback
        else:
            located.first = saved
        return located

    article.locator.side_effect = locate
    return browser


@pytest.mark.parametrize(
    "probe_position",
    ["primary_identity", "fallback_identity", "saved_status"],
)
def test_get_card_snapshot_propagates_page_crash_from_inner_probe(
    probe_position,
):
    browser = _card_snapshot_browser_with_crash(probe_position)
    async def retry_once(coro_fn, *_args, **_kwargs):
        return await coro_fn()

    with patch("linkedin.browser._retry", side_effect=retry_once), \
         pytest.raises(RuntimeError, match="Page crashed"):
        asyncio.run(browser.get_card_snapshot(0))


@pytest.mark.parametrize(
    "probe_position",
    ["primary_identity", "fallback_identity", "saved_status"],
)
def test_inner_card_probe_crash_preserves_top_level_string_ownership(
    probe_position,
):
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_linkedin_pipeline(td)
        pipeline.brief_obj.has_v2_schema = True
        pipeline._tightening_prefix = ""
        pipeline._triage_tightened = False
        owner = SearchString(
            id=1,
            name="owner",
            boolean="one",
            status="in_progress",
        )
        later = SearchString(
            id=2,
            name="later",
            boolean="two",
            status="queued",
        )
        Progress(brief_name="test", strings=[owner, later]).save(
            str(pipeline.progress_path)
        )
        inner_browser = _card_snapshot_browser_with_crash(probe_position)
        pipeline.browser.connect = AsyncMock()
        pipeline.browser.disconnect = AsyncMock()
        pipeline.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        pipeline._session_expired = MagicMock()
        pipeline._session_expired.is_set.return_value = False
        pipeline.browser.focus_card_for_review = AsyncMock()
        pipeline.browser.get_card_slot_count = AsyncMock(return_value=5)
        pipeline.browser.get_card_snapshot = AsyncMock(
            side_effect=inner_browser.get_card_snapshot
        )
        pipeline._apply_session_location_filter = AsyncMock()
        pipeline._print_session_summary = MagicMock()
        pipeline._print_summary = MagicMock()
        pipeline._generate_run_report = MagicMock()
        pipeline._finalize_run_snapshot = MagicMock(
            return_value=Path(td, "frozen")
        )
        touched: list[int] = []

        async def process(search_string, progress):
            touched.append(search_string.id)
            await pipeline._review_page_batch(
                search_string,
                1,
                5,
                MagicMock(),
                [],
                pipeline._fresh_string_stats(),
                progress,
            )
            if search_string.id == later.id:
                raise AssertionError("later string acquired browser authority")

        pipeline._process_string = process
        facial_no = OpusDecision(
            stage="facial",
            decision="FACIAL_NO",
            path="none",
            confidence=1.0,
            rationale="not a fit",
            candidate_name="Ada",
            profile_url="/talent/profile/ada",
        )
        snippet = _make_snippet(
            name="Ada",
            profile_url="/talent/profile/ada",
        )

        async def retry_once(coro_fn, *_args, **_kwargs):
            return await coro_fn()

        with patch(
            "linkedin.acquisition.extract_snippet_from_card_innertext",
            return_value=snippet,
        ), patch(
            "linkedin.acquisition.human_delay_correlated",
            return_value=0,
        ), patch(
            "shared.judger.facial_judge_batch",
            return_value=[facial_no],
        ), patch(
            "linkedin.browser._retry",
            side_effect=retry_once,
        ), pytest.raises(RuntimeError, match="Page crashed"):
            asyncio.run(pipeline.run_full(resume=True))

        assert touched == [owner.id]
        latest = pipeline._runtime_state.get_latest_run(
            source="linkedin",
            brief_id=pipeline.brief_obj.linkedin_project_id,
        )
        statuses = [
            row["status"]
            for row in pipeline._runtime_state.list_work_units(
                int(latest["id"]),
                kind="linkedin_string",
            )
        ]
        assert statuses == ["in_progress", "queued"]


@pytest.mark.parametrize(
    ("panel_text", "expected_name", "expected"),
    [
        ("Grace Hopper\nEngineer", "Biao Zhang", False),
        (RuntimeError("panel unreadable"), "Biao Zhang", True),
        ("Biao Z.\nEngineer", "Biao Zhang", True),
        ("Biao Zhang\nEngineer", "Biao Z.", True),
        # Truth table for the name matcher. A loose prefix comparison confirmed
        # the WRONG person (Ann Li read as Ann Lin, John Smith as John Smithson),
        # and a raw comparison rejected the SAME person whenever the panel
        # rendered diacritics the result card dropped (Jose Garcia vs José
        # García) — that one aborts live runs, because the post-click probe
        # fails identically and reports save_not_persisted.
        ("Ann Lin\nEngineer", "Ann Li", False),
        ("Ann Li\nEngineer", "Ann Lin", False),
        ("John Smithson\nEngineer", "John Smith", False),
        ("José García\nEngineer", "Jose Garcia", True),
        ("Jose Garcia\nEngineer", "José García", True),
        ("Ada Lovelace\nEngineer", "Ada Lovelace", True),
        ("Biao Zh…\nEngineer", "Biao Zhang", True),
    ],
    ids=[
        "mismatch",
        "unreadable",
        "panel-truncated",
        "snippet-truncated",
        "expected-is-prefix-of-panel",
        "panel-is-prefix-of-expected",
        "surname-superstring",
        "accent-folding-panel-side",
        "accent-folding-expected-side",
        "exact-match",
        "panel-ellipsis",
    ],
)
def test_already_saved_cross_checks_expected_name_against_panel_text(
    panel_text,
    expected_name,
    expected,
):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    change_stage = MagicMock()
    change_stage.is_visible = AsyncMock(return_value=True)
    panel = MagicMock()
    panel.inner_text = AsyncMock(
        side_effect=panel_text if isinstance(panel_text, Exception) else None,
        return_value=None if isinstance(panel_text, Exception) else panel_text,
    )
    page = MagicMock()

    def locate(selector):
        located = MagicMock()
        located.first = (
            panel if "profile__main-container" in selector else change_stage
        )
        return located

    page.locator.side_effect = locate
    browser._page = page
    assert asyncio.run(
        browser.is_already_saved(expected_name=expected_name)
    ) is expected


def test_identity_drift_after_pointer_move_fails_receipt_before_mouse_down():
    """Drift after the commit guard yields a failed receipt without dispatch."""
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        (pipeline, store, browser, page, transaction, events, snippet) = (
            _production_save_click_harness(td, profile_url)
        )
        browser.is_already_saved_on_card = AsyncMock(return_value=False)

        async def drift_at_commit(_identity_fragment, *, dry_run):
            if dry_run:
                events.append("resolve")
                return {"ok": True, "dispatched": False}
            events.append("refused")
            page.url = _PROJECT_PAGE_URL_DRIFTED
            return {"ok": False, "reason": "card_not_found"}

        transaction.side_effect = drift_at_commit
        with pytest.raises(RuntimeError, match="profile identity"):
            _run_save_decision(pipeline, snippet)
        assert events == ["guard", "resolve", "guard", "refused"]
        assert sum(call.kwargs["dry_run"] is False for call in transaction.await_args_list) == 1
        receipt = store.list_candidate_side_effects(source="linkedin", brief_id="test-project")[0]
        assert receipt["status"] == "failed"
        candidate = store.get_candidate(source="linkedin", brief_id="test-project", identity_key=profile_url)
        assert candidate["current_lifecycle_state"] != "full_terminal"


def _production_save_click_harness(td, profile_url, *, name="Ada Lovelace"):
    """Wire the real Python save path at the atomic browser transaction seam."""
    pipeline, store = _real_save_ledger_pipeline(td, profile_url)
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    events: list[str] = []
    page = MagicMock(url=_PROJECT_PAGE_URL)
    page.wait_for_timeout = AsyncMock()
    browser._page = page

    async def card_save_transaction(_identity_fragment, *, dry_run):
        events.append("resolve" if dry_run else "dispatch")
        return {"ok": True, "dispatched": not dry_run}

    transaction = AsyncMock(side_effect=card_save_transaction)
    browser._card_save_transaction = transaction
    browser.set_required_project_id(pipeline.brief_obj.linkedin_project_id)
    browser.scroll_for_linger = AsyncMock(return_value=0)
    browser.scroll_restore = AsyncMock()
    pipeline.browser = browser
    pipeline._side_effects_service.deps.browser = browser
    pipeline._side_effects_service.deps.before_irreversible_side_effect = lambda: events.append("guard")
    snippet = _make_snippet(name=name, profile_url=profile_url)
    return pipeline, store, browser, page, transaction, events, snippet

def _run_save_decision(pipeline, snippet):
    with patch(
        "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MIN_SECONDS", 0
    ), patch(
        "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MAX_SECONDS", 0
    ), patch(
        "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_BASE_SECONDS", 0
    ), patch(
        "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MIN_CHUNKS_BACK", 0
    ), patch(
        "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MAX_CHUNKS_BACK", 0
    ), patch("linkedin.side_effects.asyncio.sleep", new=AsyncMock()):
        return asyncio.run(
            pipeline._side_effects_service.handle_save_decision(
                snippet=snippet,
                runtime_search_string=SearchString(id=1, name="test", boolean="foo"),
                attempt_id=None,
            )
        )


def test_save_click_positions_before_guard_then_presses_once():
    """The commit guard runs immediately before the sole atomic dispatch."""
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        (pipeline, store, browser, page, transaction, events, snippet) = (
            _production_save_click_harness(td, profile_url)
        )
        browser.is_already_saved_on_card = AsyncMock(side_effect=[False, False, True])
        outcome = _run_save_decision(pipeline, snippet)
        assert outcome.status == "succeeded"
        assert events == ["guard", "resolve", "guard", "dispatch"]
        assert sum(call.kwargs["dry_run"] is False for call in transaction.await_args_list) == 1
        receipt = store.list_candidate_side_effects(source="linkedin", brief_id="test-project")[0]
        assert receipt["status"] == "succeeded"


def test_ambiguous_landed_save_is_confirmed_without_a_second_dispatch():
    import json as _json

    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        (pipeline, store, browser, _page, transaction, _events, snippet) = (
            _production_save_click_harness(td, profile_url)
        )
        landed = False

        async def probe(_identity_fragment):
            return landed

        async def land_then_disconnect(_identity_fragment, *, dry_run):
            nonlocal landed
            if dry_run:
                return {"ok": True, "dispatched": False}
            landed = True
            raise RuntimeError("Target crashed")

        browser.is_already_saved_on_card = AsyncMock(side_effect=probe)
        transaction.side_effect = land_then_disconnect

        with pytest.raises(RuntimeError, match="Target crashed"):
            _run_save_decision(pipeline, snippet)

        failed = store.list_candidate_side_effects(
            source="linkedin",
            brief_id="test-project",
        )[0]
        assert failed["status"] == "failed"
        assert _json.loads(failed["payload_json"])["failure_reason"] == (
            "save_not_confirmed"
        )

        outcome = _run_save_decision(pipeline, snippet)

        assert outcome.status == "succeeded"
        assert outcome.payload["already_present"] is True
        assert sum(
            call.kwargs["dry_run"] is False
            for call in transaction.await_args_list
        ) == 1
        receipt = store.list_candidate_side_effects(
            source="linkedin",
            brief_id="test-project",
        )[0]
        assert receipt["status"] == "succeeded"
        assert int(receipt["attempt_count"]) == 2


def test_identity_drift_during_positioning_blocks_the_press():
    """Drift during retryable resolution prevents any commit transaction."""
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        (pipeline, store, browser, page, transaction, events, snippet) = (
            _production_save_click_harness(td, profile_url)
        )
        browser.is_already_saved_on_card = AsyncMock(return_value=False)

        async def drift_during_resolution(_identity_fragment, *, dry_run):
            assert dry_run is True
            events.append("resolve")
            page.url = _PROJECT_PAGE_URL_DRIFTED
            return {"ok": False, "reason": "card_not_found"}

        transaction.side_effect = drift_during_resolution
        with pytest.raises(RuntimeError, match="profile identity"):
            _run_save_decision(pipeline, snippet)

        assert events == ["guard", "resolve"]
        assert not any(call.kwargs["dry_run"] is False for call in transaction.await_args_list)
        receipt = store.list_candidate_side_effects(source="linkedin", brief_id="test-project")[0]
        assert receipt["status"] == "failed"


def test_mismatched_panel_cannot_succeed_canonical_save_receipt():
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        pipeline, store = _real_save_ledger_pipeline(td, profile_url)
        browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
        page = MagicMock(url=_PROJECT_PAGE_URL)
        change_stage = MagicMock()
        change_stage.is_visible = AsyncMock(return_value=True)
        panel = MagicMock()
        panel.inner_text = AsyncMock(return_value="Grace Hopper\nEngineer")

        def locate(selector):
            located = MagicMock()
            located.first = (
                panel if "profile__main-container" in selector else change_stage
            )
            return located

        page.locator.side_effect = locate
        browser._page = page
        browser.is_already_saved_on_card = AsyncMock(return_value=None)
        browser.save_candidate = AsyncMock(return_value=False)
        pipeline.browser = browser
        pipeline._side_effects_service.deps.browser = browser
        snippet = _make_snippet(
            name="Ada Lovelace",
            profile_url=profile_url,
        )

        with patch(
            "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MIN_SECONDS",
            0,
        ), patch(
            "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MAX_SECONDS",
            0,
        ), patch(
            "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_BASE_SECONDS",
            0,
        ), patch(
            "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MIN_CHUNKS_BACK",
            0,
        ), patch(
            "linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MAX_CHUNKS_BACK",
            0,
        ), patch("linkedin.side_effects.asyncio.sleep", new=AsyncMock()):
            outcome = asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=SearchString(
                        id=1,
                        name="test",
                        boolean="foo",
                    ),
                    attempt_id=None,
                )
            )

        assert outcome.status == "failed"
        receipt = store.list_candidate_side_effects(
            source="linkedin",
            brief_id="test-project",
        )[0]
        assert receipt["status"] == "failed"
        candidate = store.get_candidate(
            source="linkedin",
            brief_id="test-project",
            identity_key=profile_url,
        )
        assert candidate["current_lifecycle_state"] != "full_terminal"


# ---------------------------------------------------------------------------
# F1 / F2 — the project-context carve-out was a BYPASS.
#
# Wave E wrote both project guards so an ABSENT id on EITHER side was "not a
# mismatch". A Recruiter page with no project id in its URL (the global
# /talent/search view, a bare /talent/profile/<id> page) therefore satisfied a
# brief that pins a project: run-start skipped navigation and the pre-save
# boundary let the click through. The correct rule is asymmetric — when the
# brief HAS a project, a page that cannot PROVE it belongs to that project is
# UNVERIFIED, and unverified is a mismatch. When the brief has no project,
# nothing is a mismatch (the carve-out survives exactly there).
# ---------------------------------------------------------------------------

def _project_mismatch_error():
    return sys.modules["linkedin.orchestrator"].ProjectContextMismatchError


def _assert_no_save_landed(store, profile_url, *, transaction=None, browser=None):
    """No dispatch, no succeeded receipt, no terminal full stage."""
    if transaction is not None:
        assert not any(call.kwargs["dry_run"] is False for call in transaction.await_args_list)
    if browser is not None:
        browser.save_candidate.assert_not_awaited()
    receipts = store.list_candidate_side_effects(source="linkedin", brief_id="test-project")
    assert len(receipts) == 1
    assert receipts[0]["status"] == "failed"
    candidate = store.get_candidate(source="linkedin", brief_id="test-project", identity_key=profile_url)
    assert candidate["current_lifecycle_state"] != "full_terminal"

def test_save_from_a_projectless_page_is_refused_when_the_brief_pins_a_project():
    """A pinned brief refuses an unverified project before save resolution."""
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        (pipeline, store, browser, page, transaction, events, snippet) = _production_save_click_harness(td, profile_url)
        pipeline._side_effects_service.deps.before_irreversible_side_effect = pipeline._honor_irreversible_side_effect_boundary
        page.url = _PROJECTLESS_PAGE_URL
        browser.is_already_saved_on_card = AsyncMock(return_value=False)
        with pytest.raises(_project_mismatch_error()):
            _run_save_decision(pipeline, snippet)
        assert events == []
        transaction.assert_not_awaited()
        _assert_no_save_landed(store, profile_url, transaction=transaction)

def test_project_drops_out_of_the_url_while_positioning_blocks_the_press():
    """The atomic commit refuses when its live location lost the project."""
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        (pipeline, store, browser, page, transaction, events, snippet) = _production_save_click_harness(td, profile_url)
        pipeline._side_effects_service.deps.before_irreversible_side_effect = pipeline._honor_irreversible_side_effect_boundary
        page.url = _PROJECT_PAGE_URL
        browser.is_already_saved_on_card = AsyncMock(return_value=False)

        async def project_mismatch_at_commit(_identity_fragment, *, dry_run):
            if dry_run:
                events.append("resolve")
                return {"ok": True, "dispatched": False}
            events.append("refused")
            page.url = _PROJECTLESS_PAGE_URL
            return {"ok": False, "reason": "project_mismatch", "saw": None}

        transaction.side_effect = project_mismatch_at_commit
        outcome = _run_save_decision(pipeline, snippet)
        assert outcome.status == "failed"
        assert events == ["resolve", "refused"]
        assert browser._last_save_failure_reason == "save_project_mismatch"
        assert sum(call.kwargs["dry_run"] is False for call in transaction.await_args_list) == 1
        assert "if (project && pageProject !== project)" in LinkedInBrowser._CARD_SAVE_TRANSACTION_JS
        receipts = store.list_candidate_side_effects(source="linkedin", brief_id="test-project")
        assert receipts[0]["status"] == "failed"
        candidate = store.get_candidate(source="linkedin", brief_id="test-project", identity_key=profile_url)
        assert candidate["current_lifecycle_state"] != "full_terminal"

def test_projectless_brief_still_saves_from_a_projectless_page():
    """A projectless brief leaves the transaction's project pin unset."""
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        (pipeline, store, browser, page, transaction, events, snippet) = _production_save_click_harness(td, profile_url)
        pipeline._side_effects_service.deps.before_irreversible_side_effect = pipeline._honor_irreversible_side_effect_boundary
        pipeline.brief_obj.linkedin_project_id = None
        browser.set_required_project_id(None)
        page.url = _PROJECTLESS_PAGE_URL
        browser.is_already_saved_on_card = AsyncMock(side_effect=[False, False, True])
        outcome = _run_save_decision(pipeline, snippet)
        assert outcome.status == "succeeded"
        assert browser._required_project_id is None
        assert events == ["resolve", "dispatch"]
        assert sum(call.kwargs["dry_run"] is False for call in transaction.await_args_list) == 1
        receipt = store.list_candidate_side_effects(source="linkedin", brief_id="test-project")[0]
        assert receipt["status"] == "succeeded"

def test_already_saved_probe_on_a_foreign_project_cannot_report_success():
    """F2: `is_already_saved` True short-circuited the whole guard chain —
    `before_save_click` (the only place the project assertion lived) is never
    invoked on that branch, so being present in project 999 satisfied a SAVE
    owed to the brief's project and the caller marked the full stage terminal.
    """
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        pipeline, store = _real_save_ledger_pipeline(td, profile_url)
        pipeline.browser.page = MagicMock(url=_FOREIGN_PROJECT_PAGE_URL)
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=True)
        pipeline.browser.save_candidate = AsyncMock()
        snippet = _make_snippet(profile_url=profile_url)

        with pytest.raises(_project_mismatch_error()):
            asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=SearchString(
                        id=1, name="test", boolean="foo"
                    ),
                    attempt_id=None,
                )
            )

        assert pipeline.stats.get("already_present", 0) == 0
        _assert_no_save_landed(store, profile_url, browser=pipeline.browser)


def test_already_saved_probe_on_a_projectless_page_cannot_report_success():
    """F2 x F1: the same short-circuit reached through the projectless bypass
    rather than an explicit foreign project id.
    """
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        pipeline, store = _real_save_ledger_pipeline(td, profile_url)
        pipeline.browser.page = MagicMock(url=_PROJECTLESS_PAGE_URL)
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=True)
        pipeline.browser.save_candidate = AsyncMock()
        snippet = _make_snippet(profile_url=profile_url)

        with pytest.raises(_project_mismatch_error()):
            asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=SearchString(
                        id=1, name="test", boolean="foo"
                    ),
                    attempt_id=None,
                )
            )

        assert pipeline.stats.get("already_present", 0) == 0
        _assert_no_save_landed(store, profile_url, browser=pipeline.browser)


def test_project_changes_during_the_already_saved_probe_cannot_report_success():
    """F2 twin: the page is the brief's project when the probe starts and a
    different project when it returns. Only the AFTER assertion catches this —
    a single pre-probe check would pass it through as already_present.
    """
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        pipeline, store = _real_save_ledger_pipeline(td, profile_url)
        page = MagicMock(url=_PROJECT_PAGE_URL)
        pipeline.browser.page = page

        async def _probe_and_switch_project(_identity_fragment):
            page.url = _FOREIGN_PROJECT_PAGE_URL
            return True

        pipeline.browser.is_already_saved_on_card = AsyncMock(
            side_effect=_probe_and_switch_project
        )
        pipeline.browser.save_candidate = AsyncMock()
        snippet = _make_snippet(profile_url=profile_url)

        with pytest.raises(_project_mismatch_error()):
            asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=SearchString(
                        id=1, name="test", boolean="foo"
                    ),
                    attempt_id=None,
                )
            )

        assert pipeline.stats.get("already_present", 0) == 0
        _assert_no_save_landed(store, profile_url, browser=pipeline.browser)


def test_already_saved_on_the_brief_project_still_reports_already_present():
    """F2 counterpart: the correct-project already-saved path is untouched."""
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        pipeline, store = _real_save_ledger_pipeline(td, profile_url)
        pipeline.browser.page = MagicMock(url=_PROJECT_PAGE_URL)
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=True)
        pipeline.browser.save_candidate = AsyncMock()
        snippet = _make_snippet(profile_url=profile_url)

        outcome = asyncio.run(
            pipeline._side_effects_service.handle_save_decision(
                snippet=snippet,
                runtime_search_string=SearchString(id=1, name="test", boolean="foo"),
                attempt_id=None,
            )
        )

        assert outcome.status == "succeeded"
        assert outcome.payload["already_present"] is True
        assert pipeline.stats.get("already_present", 0) == 1
        assert pipeline.stats.get("saved", 0) == 0
        pipeline.browser.save_candidate.assert_not_awaited()
        receipt = store.list_candidate_side_effects(
            source="linkedin", brief_id="test-project"
        )[0]
        assert receipt["status"] == "succeeded"


def test_already_saved_probe_is_ungated_when_the_brief_pins_no_project():
    """F2 x F1 acceptance (3): with no brief project, the already-saved branch
    behaves exactly as it does today even on a projectless page.
    """
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        pipeline, store = _real_save_ledger_pipeline(td, profile_url)
        pipeline.brief_obj.linkedin_project_id = None
        pipeline.browser.page = MagicMock(url=_PROJECTLESS_PAGE_URL)
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=True)
        pipeline.browser.save_candidate = AsyncMock()
        snippet = _make_snippet(profile_url=profile_url)

        outcome = asyncio.run(
            pipeline._side_effects_service.handle_save_decision(
                snippet=snippet,
                runtime_search_string=SearchString(id=1, name="test", boolean="foo"),
                attempt_id=None,
            )
        )

        assert outcome.status == "succeeded"
        assert outcome.payload["already_present"] is True
        pipeline.browser.save_candidate.assert_not_awaited()


# ---------------------------------------------------------------------------
# F2, the rest of the population. Bracketing `browser.is_already_saved` in
# side_effects.py closes the probe the SERVICE runs. It does not close the
# probes `LinkedInBrowser.save_candidate` runs INSIDE itself
# (linkedin/browser.py `_do` -> `_probe_saved`, three call sites plus the
# post-exception fallback). Each of those returns True out of save_candidate
# WITHOUT ever reaching `before_click`, which is the only thing that carries
# the click-time project assertion — so `saved = True`, `stats["saved"] += 1`,
# a SUCCEEDED receipt, and a terminal full stage, all on a page that moved to
# another project after the service's own probe cleared. Same defect as F2,
# one layer down; the guard therefore belongs on the boundary where the
# success is consumed, not on any single branch inside the browser.
# ---------------------------------------------------------------------------

def test_save_candidates_own_probe_cannot_report_success_on_a_foreign_project():
    """A foreign-project transaction is unreadable, never already saved."""
    probe_browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    probe_browser._card_save_transaction = AsyncMock(return_value={"ok": False, "reason": "project_mismatch", "saw": "999"})
    assert asyncio.run(probe_browser.is_already_saved_on_card("ada")) is None

    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        (pipeline, store, browser, page, transaction, events, snippet) = _production_save_click_harness(td, profile_url)
        pipeline._side_effects_service.deps.before_irreversible_side_effect = pipeline._honor_irreversible_side_effect_boundary
        page.url = _PROJECT_PAGE_URL
        calls = 0

        async def foreign_after_service_probe(_identity_fragment, *, dry_run):
            nonlocal calls
            assert dry_run is True
            calls += 1
            if calls == 1:
                events.append("brief-project-probe")
                return {"ok": True, "dispatched": False}
            page.url = _FOREIGN_PROJECT_PAGE_URL
            events.append("foreign-project-refusal")
            return {"ok": False, "reason": "project_mismatch", "saw": "999999999"}

        transaction.side_effect = foreign_after_service_probe
        outcome = _run_save_decision(pipeline, snippet)
        assert outcome.status == "failed"
        assert calls > 1
        assert "foreign-project-refusal" in events
        assert not any(call.kwargs["dry_run"] is False for call in transaction.await_args_list)
        assert pipeline.stats.get("saved", 0) == 0
        assert pipeline.stats.get("already_present", 0) == 0
        _assert_no_save_landed(store, profile_url, transaction=transaction)

def test_interrupted_save_replay_stamps_reconciled_self_save():
    """Attempt 1 lands the click then disconnects; attempt 2 probe finds saved."""
    import json as _json

    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        (pipeline, store, browser, _page, transaction, _events, snippet) = (
            _production_save_click_harness(td, profile_url)
        )
        landed = False

        async def probe(_identity_fragment):
            return landed

        async def land_then_disconnect(_identity_fragment, *, dry_run):
            nonlocal landed
            if dry_run:
                return {"ok": True, "dispatched": False}
            landed = True
            raise RuntimeError("Target crashed")

        browser.is_already_saved_on_card = AsyncMock(side_effect=probe)
        transaction.side_effect = land_then_disconnect

        with pytest.raises(RuntimeError, match="Target crashed"):
            _run_save_decision(pipeline, snippet)

        outcome = _run_save_decision(pipeline, snippet)

        assert outcome.status == "succeeded"
        assert outcome.payload["already_present"] is True
        assert outcome.payload["reconciled_self_save"] is True
        # Live global counter: a reconciled self-save is OUR save, so the
        # running tally credits "saved", not "already_present" — otherwise the
        # run under-reports until the next startup rebuild.
        assert pipeline.stats["saved"] == 1
        assert pipeline.stats.get("already_present", 0) == 0
        receipt = store.list_candidate_side_effects(
            source="linkedin",
            brief_id="test-project",
        )[0]
        assert receipt["status"] == "succeeded"
        assert int(receipt["attempt_count"]) == 2
        assert _json.loads(receipt["payload_json"])["reconciled_self_save"] is True


def test_foreign_already_present_first_attempt_has_no_self_save_stamp():
    """First-attempt probe finds already-saved — foreign, not reconciled."""
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        (pipeline, store, browser, _page, transaction, _events, snippet) = (
            _production_save_click_harness(td, profile_url)
        )
        browser.is_already_saved_on_card = AsyncMock(return_value=True)
        outcome = _run_save_decision(pipeline, snippet)
        assert outcome.status == "succeeded"
        assert outcome.payload["already_present"] is True
        assert "reconciled_self_save" not in outcome.payload
        receipt = store.list_candidate_side_effects(
            source="linkedin",
            brief_id="test-project",
        )[0]
        assert receipt["status"] == "succeeded"
        assert int(receipt["attempt_count"]) == 1
        assert not transaction.await_args_list


def test_confirmed_save_on_the_brief_project_still_succeeds():
    """A confirmed atomic dispatch commits one successful save receipt."""
    with tempfile.TemporaryDirectory() as td:
        profile_url = "/talent/profile/ada"
        (pipeline, store, browser, page, transaction, events, snippet) = _production_save_click_harness(td, profile_url)
        pipeline._side_effects_service.deps.before_irreversible_side_effect = pipeline._honor_irreversible_side_effect_boundary
        page.url = _PROJECT_PAGE_URL
        browser.is_already_saved_on_card = AsyncMock(side_effect=[False, False, True])
        outcome = _run_save_decision(pipeline, snippet)
        assert outcome.status == "succeeded"
        assert outcome.payload.get("already_present") is None
        assert events == ["resolve", "dispatch"]
        assert sum(call.kwargs["dry_run"] is False for call in transaction.await_args_list) == 1
        assert pipeline.stats.get("saved", 0) == 1
        receipt = store.list_candidate_side_effects(source="linkedin", brief_id="test-project")[0]
        assert receipt["status"] == "succeeded"


def test_browser_card_timing_events_are_registered_and_fail_soft():
    events = []
    browser = LinkedInBrowser(
        governor=UNGOVERNED_FOR_TESTS,
        timing_recorder=lambda event, payload: events.append((event, payload)),
    )
    page = MagicMock()
    page.evaluate = AsyncMock(return_value=900)
    page.wait_for_timeout = AsyncMock()
    browser._page = page

    article = MagicMock()
    article.wait_for = AsyncMock()
    article.inner_text = AsyncMock(return_value="Ada\nEngineer at Acme")
    name = MagicMock()
    name.inner_text = AsyncMock(return_value="Ada")
    name.get_attribute = AsyncMock(return_value="/talent/profile/ada")
    change = MagicMock()
    change.is_visible = AsyncMock(return_value=False)

    def article_locator(selector):
        result = MagicMock()
        result.first = change if "Change stage" in selector else name
        return result

    article.locator.side_effect = article_locator
    li = MagicMock()
    li.evaluate = AsyncMock(return_value=None)
    li.locator.return_value.first = article
    browser._wait_for_result_slot = AsyncMock(return_value=li)

    asyncio.run(browser.focus_card_for_review(2))
    snapshot = asyncio.run(browser.get_card_snapshot(2))

    assert snapshot["name"] == "Ada"
    assert [event for event, _payload in events] == [
        "card_focus_timing",
        "card_snapshot_timing",
    ]
    assert events[0][1]["card_index"] == 2
    assert events[0][1]["succeeded"] is True
    assert events[1][1]["text_chars"] == len("Ada\nEngineer at Acme")
    assert all(event in TIMING_EVENT_SCHEMAS for event, _payload in events)

    browser._timing_recorder = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("telemetry sink failed")
    )
    asyncio.run(browser.focus_card_for_review(2))


def test_browser_profile_open_expand_and_innertext_emit_registered_timings():
    events = []
    browser = LinkedInBrowser(
        governor=UNGOVERNED_FOR_TESTS,
        timing_recorder=lambda event, payload: events.append((event, payload)),
    )
    page = MagicMock()
    page.wait_for_timeout = AsyncMock()
    browser._page = page
    browser._ghost_click_locator = AsyncMock()

    link = MagicMock()
    link.get_attribute = AsyncMock(return_value="/talent/profile/ada")
    links = MagicMock()
    links.count = AsyncMock(return_value=1)
    links.nth.return_value = link
    panel = MagicMock()
    panel.wait_for = AsyncMock()

    control = MagicMock()
    control.inner_text = AsyncMock(return_value="See more")
    control.is_visible = AsyncMock(return_value=True)
    controls = MagicMock()
    controls.all = AsyncMock(return_value=[control])
    container = MagicMock()
    container.wait_for = AsyncMock()
    container.inner_text = AsyncMock(
        return_value="Ada\nEngineer\nExperience\nAcme\nEducation\nMIT"
    )
    container.locator.return_value = controls

    def page_locator(selector):
        if selector == 'ol.profile-list a[href*="/talent/profile/"]':
            return links
        if selector == "div.profile-slidein__container":
            return panel
        result = MagicMock()
        result.first = container
        return result

    page.locator.side_effect = page_locator
    with patch("linkedin.browser.asyncio.sleep", new=AsyncMock()):
        asyncio.run(browser.open_profile_by_url("/talent/profile/ada"))
        text = asyncio.run(browser.get_profile_innertext())

    by_name = {event: payload for event, payload in events}
    assert text.startswith("Ada\nEngineer")
    assert by_name["profile_open_timing"]["succeeded"] is True
    assert by_name["profile_expand_timing"]["elements_walked"] == 1
    assert by_name["profile_expand_timing"]["clicks_made"] == 1
    assert by_name["profile_innertext_timing"]["text_chars"] == len(text)


def test_profile_read_timing_counts_actual_backend_wheel_events(capsys):
    events = []
    browser = LinkedInBrowser(
        governor=UNGOVERNED_FOR_TESTS,
        timing_recorder=lambda event, payload: events.append((event, payload)),
    )
    container = MagicMock()
    container.wait_for = AsyncMock()
    container.evaluate = AsyncMock(side_effect=[1200, 400])
    page = MagicMock()
    page.locator.return_value.first = container
    browser._page = page
    browser._input_backend.scroll = AsyncMock(return_value=3)

    async def focused(_scrollable):
        await browser._human_scroll(400, channel="profile_read")
        await browser._human_scroll(-400, channel="profile_read_return")

    browser._read_focused = AsyncMock(side_effect=focused)
    with patch("linkedin.browser.random.choices", return_value=["focused_reader"]):
        asyncio.run(browser.simulate_profile_read())

    event, payload = events[-1]
    assert event == "profile_read_timing"
    assert payload["pattern"] == "focused_reader"
    assert payload["chunk_count"] == 2
    assert payload["wheel_events"] == 6
    assert "timing" not in capsys.readouterr().out.lower()


def test_all_nonstream_timing_events_have_explicit_schemas():
    assert set(TIMING_EVENT_SCHEMAS) == {
        "card_focus_timing",
        "card_snapshot_timing",
        "profile_open_timing",
        "profile_read_timing",
        "profile_innertext_timing",
        "profile_expand_timing",
        "checkpoint_progress_timing",
        "cadence_pause_timing",
    }
