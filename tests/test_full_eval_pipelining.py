from __future__ import annotations

from linkedin.orchestrator import Pipeline as PipelineFacade

import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin.browser import LinkedInBrowser
from linkedin.side_effects import LinkedInSideEffectsDeps, LinkedInSideEffectsService
from shared.execution import SideEffectOutcome
from shared.failures import ApiBudgetExhaustedError
from shared.governor import (
    GovernorLimitReached,
    OperatorStopRequested,
    SessionExpired,
)
from shared.runtime_state import LinkedInRuntimeStateBridge, RuntimeStateStore
from shared.schemas import (
    CandidateProfileSummary,
    CandidateSnippet,
    ExternalCandidateEvidence,
    OpusDecision,
    Progress,
    SearchString,
    TriggerDecision,
)
from shared.storage import read_jsonl


def _make_pipeline(output_dir: str):
    with patch("linkedin.evaluation_loop.load_brief") as mock_brief, \
         patch("linkedin.evaluation_loop.init_judger"), \
         patch("linkedin.evaluation_loop.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        brief.permanent_filters = {}
        brief.needs_preflight.return_value = False
        mock_brief.return_value = brief

        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline

        pipeline = PipelineFacade(brief_path=str(brief_path), output_dir=output_dir)._loop
        pipeline._ensure_services = MagicMock()
        pipeline.state.context._runtime_bridge = None
        pipeline.state.context._runtime_run_id = None
        pipeline.state.context._runtime_state = None
        pipeline.state.context._bias_monitor = None
        pipeline.state.reviewer.attempts._record_runtime_event = MagicMock()
        pipeline.state.reviewer._derive_novelty_value = MagicMock(return_value=("high", "test rationale"))
        pipeline.state.reviewer._profile_probe = MagicMock()
        pipeline.state.reviewer._profile_probe.evaluate.return_value = "probe"
        pipeline.state.reviewer._profile_probe.record_shadow_outcome.return_value = SimpleNamespace(
            to_dict=lambda: {"probe": "recorded"}
        )
        pipeline.state._allocator_service._checkpoint_progress = MagicMock()
        return pipeline


def _snippet(name: str, rank: int) -> CandidateSnippet:
    slug = name.lower().replace(" ", "-")
    return CandidateSnippet(
        name=name,
        headline="Builder",
        current_title="Engineer",
        current_company="Acme",
        location="Remote",
        education_snippet="",
        profile_url=f"/talent/profile/{slug}",
        source_string_id=7,
        source_string_name="test string",
        page=1,
        result_rank=rank,
        card_index=rank - 1,
    )


def _summary(snippet: CandidateSnippet) -> CandidateProfileSummary:
    return CandidateProfileSummary(
        name=snippet.name,
        profile_url=snippet.profile_url,
        headline=snippet.headline,
    )


def _decision(snippet: CandidateSnippet, decision: str, rationale: str | None = None) -> OpusDecision:
    return OpusDecision(
        stage="full",
        decision=decision,
        path="DIRECT:test" if decision in {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE"} else "none",
        confidence=0.8,
        rationale=rationale or f"{decision} rationale for {snippet.name}",
        candidate_name=snippet.name,
        profile_url=snippet.profile_url,
    )


class FakeBrowser:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.page = SimpleNamespace(url="https://www.linkedin.com/talent/search")
        self._governor = MagicMock()
        self._governor.profile_opens = []
        self._governor.record_profile_open.side_effect = (
            lambda identity: self._governor.profile_opens.append(identity)
        )

    def current_profile_identity_fragment(self) -> str:
        return LinkedInBrowser._profile_url_fragment(self.page.url)

    def show_profile(self, profile_url: str) -> None:
        identity = LinkedInBrowser._profile_url_fragment(profile_url)
        self.page.url = (
            f"https://www.linkedin.com/talent/recruiterSearch/profile/{identity}"
        )
        self._governor.record_profile_open(identity)

    async def get_profile_status_summary(self) -> dict:
        return {}

    async def go_back_to_results(self) -> None:
        self.events.append(("back", "results"))
        self.page.url = "https://www.linkedin.com/talent/search"

    async def ensure_card_rendered(self, card_index: int) -> None:
        self.events.append(("ensure_card", str(card_index)))

    async def open_profile_by_url(self, profile_url: str) -> None:
        self.events.append(("reopen_by_url", profile_url))
        self.show_profile(profile_url)

    async def open_profile(self, candidate_name: str) -> None:
        self.events.append(("reopen_by_name", candidate_name))

    async def is_already_saved_on_card(self, _identity_fragment: str) -> bool:
        self.events.append(("is_already_saved_on_card", ""))
        return False

    async def save_candidate(
        self,
        *,
        before_click=None,
        **_kwargs,
    ) -> bool:
        if before_click is not None:
            before_click()
        self.events.append(
            ("save_candidate", self.current_profile_identity_fragment())
        )
        return True

    async def write_candidate_note(
        self,
        note_text: str,
        *,
        expected_name: str,
        before_commit=None,
    ) -> bool:
        if before_commit is not None:
            before_commit()
        self.events.append(("write_candidate_note", expected_name))
        return True

    async def scroll_for_linger(self, chunks_back: int) -> int:
        self.events.append(("scroll_for_linger", str(chunks_back)))
        return 0

    async def scroll_restore(self, px: int) -> None:
        self.events.append(("scroll_restore", str(px)))


class FakePageReport:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def add_saved(self, snippet: CandidateSnippet, decision: OpusDecision) -> None:
        self.events.append(("saved", snippet.name))

    def add_save_failed(
        self,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        reason: str,
    ) -> None:
        self.events.append(("save_failed", snippet.name))

    def add_skipped_opened(self, snippet: CandidateSnippet, decision: OpusDecision) -> None:
        self.events.append((decision.decision, snippet.name))


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.linkedin.com/talent/profile/ada?tracking=one", "ada"),
        (
            "https://www.linkedin.com/talent/recruiterSearch/profile/ada#details",
            "ada",
        ),
        (
            "https://www.linkedin.com/talent/hire/9000000010/manage/all/profile/ada",
            "ada",
        ),
        ("https://www.linkedin.com/talent/search", ""),
        ("https://www.linkedin.com/company/profile/ada", ""),
    ],
)
def test_recruiter_profile_identity_fragment_supports_page_and_slidein_urls(
    url,
    expected,
):
    assert LinkedInBrowser._profile_url_fragment(url) == expected


def test_pipeline_and_search_drawers_share_exact_candidate_identity():
    pipeline_url = (
        "https://www.linkedin.com/talent/hire/9000000010/manage/all/profile/ada"
    )
    search_url = "https://www.linkedin.com/talent/recruiterSearch/profile/ada"
    neighbor_url = "https://www.linkedin.com/talent/recruiterSearch/profile/grace"

    # Named values, not the function compared with itself: an extractor that
    # returned the same thing for every URL would make every open panel pass
    # the identity check at the save boundary.
    assert LinkedInBrowser._profile_url_fragment(pipeline_url) == "ada"
    assert LinkedInBrowser._profile_url_fragment(search_url) == "ada"
    assert LinkedInBrowser._profile_url_fragment(neighbor_url) == "grace"


def _candidate_rows(snippets: list[CandidateSnippet]) -> list[dict]:
    return [
        {
            "name": snippet.name,
            "profile_url": snippet.profile_url,
            "page": snippet.page,
            "outcome": "facial_yes",
            "rationale": "facial yes",
        }
        for snippet in snippets
    ]


def _string_stats() -> dict[str, int]:
    return {
        "facial_yes": 0,
        "saves": 0,
        "rejects": 0,
        "save_failed": 0,
    }


def _wire_common_pipeline(
    pipeline,
    *,
    browser_latency: float = 0.0,
    stop_after_extract: str | None = None,
) -> list[tuple[str, str, float]]:
    events: list[tuple[str, str, float]] = []
    pipeline.state.context.browser = FakeBrowser()

    async def extract_profile_summary(snippet: CandidateSnippet, *, interest: float = 0.5):
        events.append(("browser_start", snippet.name, time.perf_counter()))
        pipeline.state.context.browser.show_profile(snippet.profile_url)
        if browser_latency:
            await asyncio.sleep(browser_latency)
        events.append(("browser_end", snippet.name, time.perf_counter()))
        if stop_after_extract == snippet.name:
            pipeline.state.context._operator_stop_event.set()
        return SimpleNamespace(profile_summary=_summary(snippet))

    pipeline.state.reviewer.acquisition = MagicMock()
    pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
        side_effect=extract_profile_summary
    )
    pipeline.state.reviewer.side_effects = MagicMock()
    pipeline.state.reviewer.side_effects.handle_save_decision = AsyncMock(
        return_value=SideEffectOutcome(
            effect_type="linkedin_save",
            status="succeeded",
            payload={"test_mode": True},
        )
    )
    return events


def _wire_real_runtime(
    pipeline,
    *,
    output_dir: str,
    snippets: list[CandidateSnippet],
) -> tuple[SearchString, Progress]:
    """Attach canonical SQLite state and register full-eval candidates."""

    store = RuntimeStateStore(Path(output_dir) / "runtime_state.sqlite3")
    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=output_dir,
        brief_id="test-project",
        brief_name="test",
    )
    search_string = SearchString(id=7, name="test string", boolean="engineer")
    progress = Progress(brief_name="test", strings=[search_string])
    run_id, _ = bridge.start_or_resume_run(
        resume=False,
        initial_progress=progress,
    )
    pipeline.state.context._runtime_state = store
    pipeline.state.context._runtime_bridge = bridge
    pipeline.state.context._runtime_run_id = run_id
    pipeline.state._progress = progress
    for snippet in snippets:
        pipeline.state.reviewer.attempts._record_runtime_snippet(search_string, snippet)
        pipeline.state.candidates._in_flight_urls.add(snippet.profile_url)
    return search_string, progress


def _full_attempt_rows(pipeline) -> list[dict]:
    with pipeline.state.context._runtime_state.connect() as conn:
        rows = conn.execute(
            """
            SELECT c.identity_key, c.current_lifecycle_state,
                   ca.status, ca.failure_kind, ca.payload_json
            FROM candidate_attempts ca
            JOIN candidates c ON c.id = ca.candidate_id
            WHERE ca.stage = 'full'
            ORDER BY ca.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


async def _run_pipelined(
    pipeline,
    snippets: list[CandidateSnippet],
    *,
    page_report: FakePageReport | None = None,
    progress: Progress | None = None,
):
    search_string = SearchString(id=7, name="test string", boolean="engineer")
    all_candidates = _candidate_rows(snippets)
    stats = _string_stats()
    for snippet in snippets:
        pipeline.state.candidates._record_facial_funnel_outcome(
            snippet=snippet, decision="FACIAL_YES", search_string=search_string, string_stats=stats,
         stats=pipeline.state.context.stats)
    decisions = await pipeline.state.reviewer._run_full_queue(
        snippets=snippets, window=2, origin="page",
        page_report=page_report,
        search_string=search_string,
        all_candidates=all_candidates,
        string_stats=stats,
        progress=progress,
        page_num=1,
    )
    return decisions, all_candidates, stats


def test_batch_early_exit_drains_yes_and_borderline_into_pipelined_full_review():
    """The page may stop paginating only after every acquired positive is drained."""

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.has_v2_schema = True
        pipeline.state.context.brief_obj._new_brief = None
        pipeline.state.reviewer._tightening_prefix = ""
        pipeline.state.reviewer._triage_tightened = False
        snippets = [
            _snippet("No One", 1),
            _snippet("No Two", 2),
            _snippet("Yes Three", 3),
            _snippet("Borderline Four", 4),
        ]
        decisions = [
            OpusDecision(
                stage="facial",
                decision="FACIAL_NO",
                path="none",
                confidence=1.0,
                rationale="not relevant",
                candidate_name=snippets[0].name,
                profile_url=snippets[0].profile_url,
            ),
            OpusDecision(
                stage="facial",
                decision="FACIAL_NO",
                path="none",
                confidence=1.0,
                rationale="not relevant",
                candidate_name=snippets[1].name,
                profile_url=snippets[1].profile_url,
            ),
            OpusDecision(
                stage="facial",
                decision="FACIAL_YES",
                path="none",
                confidence=1.0,
                rationale="clear signal",
                candidate_name=snippets[2].name,
                profile_url=snippets[2].profile_url,
            ),
            OpusDecision(
                stage="facial",
                decision="FACIAL_BORDERLINE",
                path="none",
                confidence=1.0,
                rationale="ambiguous but plausible signal",
                candidate_name=snippets[3].name,
                profile_url=snippets[3].profile_url,
            ),
        ]
        pipeline.state.context.browser.get_card_slot_count = AsyncMock(return_value=len(snippets))
        pipeline.state.pages._extract_card_snippet = AsyncMock(side_effect=snippets)
        pipeline.state.reviewer._run_full_queue = AsyncMock(return_value=[])
        pipeline.state.pages._get_early_exit_rate = MagicMock(return_value=0.50)
        page_report = MagicMock()
        all_candidates: list[dict] = []
        string_stats = pipeline.state.candidates._fresh_string_stats()
        search_string = SearchString(
            id=7,
            name="test string",
            boolean="engineer",
        )

        with patch(
            "shared.judger.facial_judge_batch",
            return_value=decisions,
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED",
            True,
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED",
            False,
        ), patch(
            "linkedin.evaluation_loop.config.FULL_EVAL_PIPELINE_ENABLED",
            True,
        ), patch(
            "linkedin.evaluation_loop.config.EARLY_EXIT_MIN_CANDIDATES",
            len(snippets),
        ):
            asyncio.run(
                pipeline.state.pages._review_page_batch(
                    search_string,
                    1,
                    0,
                    page_report,
                    all_candidates,
                    string_stats,
                    None,
                )
            )

        full_review = (
            pipeline.state.reviewer._run_full_queue.await_args.kwargs[
                "snippets"
            ]
        )
        assert [snippet.profile_url for snippet in full_review] == [
            snippets[2].profile_url,
            snippets[3].profile_url,
        ]
        assert pipeline.state.candidates._prior_outcomes[snippets[2].profile_url] == "FACIAL_YES"
        assert (
            pipeline.state.candidates._prior_outcomes[snippets[3].profile_url]
            == "FACIAL_BORDERLINE"
        )
        assert pipeline.state.context.stats["facial_yes"] == 1
        assert pipeline.state.context.stats["facial_borderline"] == 1
        assert string_stats["facial_yes"] == 1
        assert string_stats["facial_borderline"] == 1
        assert pipeline.state.pages._page_observation()["break_reason"] == "early_exit"


def test_serial_full_review_recovers_first_stuck_panel_and_drains_page():
    """A recoverable close failure must not drop later facial positives."""

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.has_v2_schema = True
        pipeline.state.context.brief_obj._new_brief = None
        pipeline.state.reviewer._tightening_prefix = ""
        pipeline.state.reviewer._triage_tightened = False
        snippets = [
            _snippet("A One", 1),
            _snippet("B Two", 2),
            _snippet("C Three", 3),
        ]
        facial_decisions = [
            OpusDecision(
                stage="facial",
                decision="FACIAL_YES",
                path="none",
                confidence=1.0,
                rationale="clear signal",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            )
            for snippet in snippets
        ]
        full_decisions = [_decision(snippet, "REJECT") for snippet in snippets]
        full_decisions[0]._panel_stuck = True
        pipeline.state.context.browser.get_card_slot_count = AsyncMock(return_value=len(snippets))
        pipeline.state.context.browser.go_back_to_results = AsyncMock()
        pipeline.state.pages._extract_card_snippet = AsyncMock(side_effect=snippets)
        pipeline.state.reviewer._full_evaluate = AsyncMock(side_effect=full_decisions)
        pipeline.state.reviewer._prepare_full_candidate = AsyncMock(
            side_effect=lambda snippet, *args, **kwargs: SimpleNamespace(snippet=snippet)
        )
        page_report = MagicMock()
        all_candidates: list[dict] = []
        string_stats = pipeline.state.candidates._fresh_string_stats()
        search_string = SearchString(id=7, name="test string", boolean="engineer")

        with patch(
            "shared.judger.facial_judge_batch",
            return_value=facial_decisions,
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED",
            False,
        ), patch(
            "linkedin.evaluation_loop.config.FULL_EVAL_PIPELINE_ENABLED",
            False,
        ), patch(
            "linkedin.evaluation_loop.config.EARLY_EXIT_MIN_CANDIDATES",
            99,
        ), patch(
            "linkedin.candidate_review.human_delay_correlated",
            return_value=0.0,
        ):
            asyncio.run(
                pipeline.state.pages._review_page_batch(
                    search_string,
                    1,
                    0,
                    page_report,
                    all_candidates,
                    string_stats,
                    Progress(brief_name="test", strings=[search_string]),
                )
            )

        assert [call.args[0].name for call in pipeline.state.reviewer._full_evaluate.await_args_list] == [
            "A One",
            "B Two",
            "C Three",
        ]
        pipeline.state.context.browser.go_back_to_results.assert_awaited_once()
        assert [candidate["outcome"] for candidate in all_candidates] == [
            "reject",
            "reject",
            "reject",
        ]
        assert pipeline.state.pages._page_observation()["break_reason"] != "panel_stuck"


def test_pipelined_full_eval_overlaps_browser_and_judgment_wall_time():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [_snippet("A One", 1), _snippet("B Two", 2), _snippet("C Three", 3)]
        events = _wire_common_pipeline(pipeline, browser_latency=0.08)
        judge_latency = 0.18

        def fake_full_judge(summary, brief, lane_context=None):
            events.append(("judge_start", summary.name, time.perf_counter()))
            time.sleep(judge_latency)
            events.append(("judge_end", summary.name, time.perf_counter()))
            return _decision(snippets[[s.name for s in snippets].index(summary.name)], "REJECT")

        with patch("linkedin.candidate_review.full_judge", side_effect=fake_full_judge), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
             patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            started = time.perf_counter()
            asyncio.run(_run_pipelined(pipeline, snippets))
            elapsed = time.perf_counter() - started

        serial_sum = len(snippets) * (0.08 + judge_latency)
        assert elapsed < serial_sum * 0.9

        by_key = {(kind, name): ts for kind, name, ts in events}
        assert by_key[("judge_start", "A One")] < by_key[("browser_end", "B Two")]
        assert by_key[("judge_start", "B Two")] < by_key[("browser_end", "C Three")]

        timing_calls = [
            call
            for call in pipeline.state.reviewer.attempts._record_runtime_event.call_args_list
            if call.kwargs.get("event_type") == "full_page_judgment_timing"
        ]
        assert len(timing_calls) == 1
        timing = timing_calls[0].kwargs["payload"]
        assert timing["string_id"] == 7
        assert timing["page"] == 1
        assert timing["full_call_count"] == 3
        assert timing["pipeline"] == "lookahead_one"
        assert timing["browser_extraction_excluded"] is True
        assert len(timing["logical_call_elapsed_ms"]) == 3
        assert all(value > 0 for value in timing["logical_call_elapsed_ms"])
        assert timing["logical_call_elapsed_total_ms"] == pytest.approx(
            sum(timing["logical_call_elapsed_ms"]), abs=0.01
        )
        assert timing["operator_blocking_elapsed_ms"] > 0


def test_pipelined_first_extraction_close_failure_retries_and_reviews_candidate():
    """A close miss after extraction retries in place instead of failing the candidate."""

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [_snippet("A One", 1), _snippet("B Two", 2)]
        _wire_common_pipeline(pipeline)
        original_back = pipeline.state.context.browser.go_back_to_results
        close_calls = 0

        async def fail_first_close():
            nonlocal close_calls
            close_calls += 1
            if close_calls == 1:
                raise RuntimeError("transient panel close failure")
            await original_back()

        pipeline.state.context.browser.go_back_to_results = AsyncMock(side_effect=fail_first_close)

        def fake_full_judge(summary, brief, lane_context=None):
            snippet = snippets[[item.name for item in snippets].index(summary.name)]
            return _decision(snippet, "REJECT")

        with patch("linkedin.candidate_review.full_judge", side_effect=fake_full_judge), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
             patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            decisions, _, _ = asyncio.run(_run_pipelined(pipeline, snippets))

        assert [decision.candidate_name for decision in decisions] == ["A One", "B Two"]
        assert [decision.decision for decision in decisions] == ["REJECT", "REJECT"]
        assert close_calls == 3


def test_unrecoverable_extraction_close_fails_loudly_and_stays_retryable():
    """Exhausted close recovery aborts the open attempt with retryable truth."""

    from linkedin.orchestrator import PanelRecoveryError

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        search_string, _ = _wire_real_runtime(
            pipeline,
            output_dir=td,
            snippets=[snippet],
        )
        pipeline.state.context.browser.go_back_to_results = AsyncMock(
            side_effect=RuntimeError("panel cannot close")
        )
        pipeline.state.pages._set_page_break_reason("early_exit")

        with patch(
            "linkedin.candidate_review.human_delay_correlated",
            return_value=0.0,
        ):
            with pytest.raises(PanelRecoveryError, match="panel recovery failed"):
                asyncio.run(
                    pipeline.state.reviewer._prepare_full_candidate(
                        snippet,
                        search_string=search_string,
                    )
                )

        assert pipeline.state.context.browser.go_back_to_results.await_count == 3
        assert pipeline.state.context._runtime_state.list_orphaned_attempts(
            source="linkedin",
            brief_id="test-project",
        ) == []
        rows = _full_attempt_rows(pipeline)
        assert [(row["identity_key"], row["status"]) for row in rows] == [
            (snippet.profile_url, "failed")
        ]
        payload = json.loads(rows[0]["payload_json"])
        assert payload["force_retryable"] is True
        assert payload["run_abort"] == "panel_recovery_failed"
        assert snippet.profile_url not in pipeline.state.candidates._in_flight_urls
        assert pipeline.state.pages._page_observation()["break_reason"] == "panel_stuck"


def test_pipelined_middle_save_close_failure_recovers_and_drains_page():
    """A stuck panel from a middle save cannot discard the opened successor."""

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [
            _snippet("A One", 1),
            _snippet("B Two", 2),
            _snippet("C Three", 3),
        ]
        _wire_common_pipeline(pipeline)
        original_back = pipeline.state.context.browser.go_back_to_results
        close_calls = 0

        async def fail_middle_save_close():
            nonlocal close_calls
            close_calls += 1
            # Three extraction closes precede the second candidate's save close.
            if close_calls == 4:
                raise RuntimeError("transient panel close failure")
            await original_back()

        pipeline.state.context.browser.go_back_to_results = AsyncMock(
            side_effect=fail_middle_save_close
        )
        page_report = FakePageReport()

        def fake_full_judge(summary, brief, lane_context=None):
            index = [item.name for item in snippets].index(summary.name)
            return _decision(snippets[index], "SAVE" if index == 1 else "REJECT")

        with patch("linkedin.candidate_review.full_judge", side_effect=fake_full_judge), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS", 0), \
             patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            decisions, _, _ = asyncio.run(
                _run_pipelined(pipeline, snippets, page_report=page_report)
            )

        assert [decision.candidate_name for decision in decisions] == [
            "A One",
            "B Two",
            "C Three",
        ]
        assert page_report.events == [
            ("REJECT", "A One"),
            ("saved", "B Two"),
            ("REJECT", "C Three"),
        ]
        assert (
            pipeline.state.reviewer.side_effects.handle_save_decision.await_args.kwargs[
                "decision"
            ]
            is decisions[1]
        )
        assert close_calls == 5
        assert pipeline.state.pages._page_observation()["break_reason"] != "panel_stuck"


def test_pipelined_full_eval_completion_is_fifo():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [_snippet("A One", 1), _snippet("B Two", 2), _snippet("C Three", 3)]
        _wire_common_pipeline(pipeline, browser_latency=0.01)
        page_report = FakePageReport()

        def fake_full_judge(summary, brief, lane_context=None):
            if summary.name == "A One":
                time.sleep(0.08)
            return _decision(snippets[[s.name for s in snippets].index(summary.name)], "REJECT")

        with patch("linkedin.candidate_review.full_judge", side_effect=fake_full_judge), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
             patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            asyncio.run(_run_pipelined(pipeline, snippets, page_report=page_report))

        assert page_report.events == [
            ("REJECT", "A One"),
            ("REJECT", "B Two"),
            ("REJECT", "C Three"),
        ]


@pytest.mark.parametrize("outreach_tier", ["STANDARD", "PRIORITY"])
def test_pipelined_save_reopens_profile_and_uses_save_actuator(outreach_tier):
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        pipeline.state.reviewer.side_effects = LinkedInSideEffectsService(
            LinkedInSideEffectsDeps(
                browser=pipeline.state.context.browser,
                stats=pipeline.state.context.stats,
                saved_urls=set(),
                log_path=pipeline.state.context.log_path,
                get_test_mode=lambda: False,
                get_runtime_bridge=lambda: None,
                get_runtime_run_id=lambda: None,
            )
        )
        page_report = FakePageReport()

        decision = _decision(snippet, "SAVE")
        decision.outreach_tier = outreach_tier
        expected_tier_counts = {outreach_tier: 1}
        original_reopen = pipeline.state.context.browser.open_profile_by_url
        original_save = pipeline.state.context.browser.save_candidate

        async def assert_tier_counted_before_reopen(profile_url):
            assert pipeline.state.context.stats["outreach_tier_counts"] == expected_tier_counts
            await original_reopen(profile_url)

        async def assert_tier_counted_before_save(**kwargs):
            assert pipeline.state.context.stats["outreach_tier_counts"] == expected_tier_counts
            return await original_save(**kwargs)

        pipeline.state.context.browser.open_profile_by_url = assert_tier_counted_before_reopen
        pipeline.state.context.browser.save_candidate = assert_tier_counted_before_save

        with patch("linkedin.candidate_review.full_judge", return_value=decision), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
             patch("linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MIN_SECONDS", 0), \
             patch("linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MAX_SECONDS", 0), \
             patch("linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_BASE_SECONDS", 0), \
             patch("linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MIN_CHUNKS_BACK", 0), \
             patch("linkedin.side_effects.config.LINKEDIN_SAVE_LINGER_MAX_CHUNKS_BACK", 0), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS", 0), \
             patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0), \
             patch("linkedin.side_effects.human_delay_correlated", return_value=0.0):
            decisions, _, stats = asyncio.run(
                _run_pipelined(pipeline, [snippet], page_report=page_report)
            )

        assert ("reopen_by_url", snippet.profile_url) in pipeline.state.context.browser.events
        assert ("save_candidate", "a-one") in pipeline.state.context.browser.events
        assert page_report.events == [("saved", snippet.name)]
        assert decisions[0].save_outcome["persisted"] is True
        assert stats["saves"] == 1
        assert pipeline.state.context.stats["outreach_tier_counts"] == expected_tier_counts


def test_serial_save_reuses_identity_verified_panel_and_one_governed_open():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        pipeline.state.reviewer.side_effects = LinkedInSideEffectsService(
            LinkedInSideEffectsDeps(
                browser=pipeline.state.context.browser,
                stats=pipeline.state.context.stats,
                saved_urls=set(),
                log_path=pipeline.state.context.log_path,
                get_test_mode=lambda: False,
                get_runtime_bridge=lambda: None,
                get_runtime_run_id=lambda: None,
            )
        )

        with patch(
            "linkedin.candidate_review.full_judge",
            return_value=_decision(snippet, "SAVE"),
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ), patch(
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
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS",
            0,
        ), patch(
            "linkedin.candidate_review.human_delay_correlated",
            return_value=0.0,
        ), patch(
            "linkedin.side_effects.human_delay_correlated",
            return_value=0.0,
        ):
            decision = asyncio.run(
                pipeline.state.reviewer._full_evaluate(
                    snippet,
                    None,
                    SearchString(id=7, name="test string", boolean="engineer"),
                )
            )

        assert decision.save_outcome["persisted"] is True
        assert pipeline.state.context.browser._governor.profile_opens == ["a-one"]
        assert ("reopen_by_url", snippet.profile_url) not in pipeline.state.context.browser.events
        assert ("save_candidate", "a-one") in pipeline.state.context.browser.events


@pytest.mark.parametrize(
    "page_url",
    [
        "https://www.linkedin.com/talent/recruiterSearch/profile/someone-else",
        "https://www.linkedin.com/talent/search",
    ],
)
def test_save_reopens_when_open_panel_identity_does_not_match(page_url):
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        pipeline.state.context.browser.page.url = page_url

        asyncio.run(pipeline.state.surface._reopen_profile_for_full_eval_save(snippet))

        assert ("reopen_by_url", snippet.profile_url) in pipeline.state.context.browser.events
        assert pipeline.state.context.browser.current_profile_identity_fragment() == "a-one"


def test_save_aborts_before_probe_when_reopen_returns_wrong_identity():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        search_string, _ = _wire_real_runtime(
            pipeline,
            output_dir=td,
            snippets=[snippet],
        )
        pipeline.state.context.browser.page.url = (
            "https://www.linkedin.com/talent/recruiterSearch/profile/b-two"
        )

        async def stale_reopen(profile_url):
            pipeline.state.context.browser.events.append(("reopen_by_url", profile_url))

        pipeline.state.context.browser.open_profile_by_url = AsyncMock(side_effect=stale_reopen)
        pipeline.state.context.browser.is_already_saved_on_card = AsyncMock()
        pipeline.state.context.browser.save_candidate = AsyncMock()
        pipeline.state.reviewer.side_effects = LinkedInSideEffectsService(
            LinkedInSideEffectsDeps(
                browser=pipeline.state.context.browser,
                stats=pipeline.state.context.stats,
                saved_urls=set(),
                log_path=pipeline.state.context.log_path,
                get_test_mode=lambda: False,
                get_runtime_bridge=lambda: pipeline.state.context._runtime_bridge,
                get_runtime_run_id=lambda: pipeline.state.context._runtime_run_id,
            )
        )

        def decide_after_panel_drift(*_args, **_kwargs):
            pipeline.state.context.browser.page.url = (
                "https://www.linkedin.com/talent/recruiterSearch/profile/b-two"
            )
            return _decision(snippet, "SAVE")

        with patch(
            "linkedin.candidate_review.full_judge",
            side_effect=decide_after_panel_drift,
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ), patch(
            "linkedin.candidate_review.human_delay_correlated",
            return_value=0.0,
        ):
            with pytest.raises(RuntimeError, match="profile identity"):
                asyncio.run(
                    pipeline.state.reviewer._full_evaluate(snippet, None, search_string)
                )

        pipeline.state.context.browser.is_already_saved_on_card.assert_not_awaited()
        pipeline.state.context.browser.save_candidate.assert_not_awaited()
        rows = pipeline.state.context._runtime_state.list_candidate_side_effects(
            source="linkedin",
            brief_id="test-project",
        )
        assert rows == []
        assert snippet.profile_url not in pipeline.state.candidates._seen_urls
        full_rows = _full_attempt_rows(pipeline)
        assert full_rows[0]["status"] == "failed"
        assert full_rows[0]["current_lifecycle_state"] == "failed_retryable"


@pytest.mark.parametrize("execution_mode", ["serial", "pipelined"])
@pytest.mark.parametrize(
    "probe_phase",
    ["initial", "post_click", "fallback"],
)
def test_save_probe_identity_drift_stays_retryable(
    execution_mode,
    probe_phase,
    monkeypatch,
):
    # The subject is the drift receipt, not the note. LINKEDIN_SAVE_NOTE_ENABLED
    # defaults ON since 2026-09-20 and the harness browser has no note surface,
    # so a note-write receipt would land alongside the save one; pin it off.
    monkeypatch.setattr("shared.config.LINKEDIN_SAVE_NOTE_ENABLED", False)
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        search_string, _ = _wire_real_runtime(
            pipeline,
            output_dir=td,
            snippets=[snippet],
        )
        pipeline.state.reviewer.side_effects = LinkedInSideEffectsService(
            LinkedInSideEffectsDeps(
                browser=pipeline.state.context.browser,
                stats=pipeline.state.context.stats,
                saved_urls=set(),
                log_path=pipeline.state.context.log_path,
                get_test_mode=lambda: False,
                get_runtime_bridge=lambda: pipeline.state.context._runtime_bridge,
                get_runtime_run_id=lambda: pipeline.state.context._runtime_run_id,
            )
        )

        async def initial_probe(_identity_fragment):
            if probe_phase == "initial":
                pipeline.state.context.browser.page.url = (
                    "https://www.linkedin.com/talent/"
                    "recruiterSearch/profile/b-two"
                )
                return True
            return False

        async def save_candidate(*, before_click=None, **_kwargs):
            if before_click is not None:
                before_click()
            pipeline.state.context.browser.page.url = (
                "https://www.linkedin.com/talent/"
                "recruiterSearch/profile/b-two"
            )
            return True

        pipeline.state.context.browser.is_already_saved_on_card = AsyncMock(
            side_effect=initial_probe
        )
        pipeline.state.context.browser.save_candidate = AsyncMock(
            side_effect=save_candidate
        )
        page_report = FakePageReport()

        with patch(
            "linkedin.candidate_review.full_judge",
            return_value=_decision(snippet, "SAVE"),
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ), patch(
            "linkedin.side_effects.human_delay_correlated",
            return_value=0.0,
        ):
            with pytest.raises(RuntimeError, match="profile identity"):
                if execution_mode == "serial":
                    asyncio.run(
                        pipeline.state.reviewer._full_evaluate(
                            snippet,
                            None,
                            search_string,
                        )
                    )
                else:
                    asyncio.run(
                        _run_pipelined(
                            pipeline,
                            [snippet],
                            page_report=page_report,
                        )
                    )

        side_effects = pipeline.state.context._runtime_state.list_candidate_side_effects(
            source="linkedin",
            brief_id="test-project",
        )
        # The RECEIPT must describe what physically happened, and the REVIEW
        # must stay retryable. Those are two different questions, and this test
        # used to answer both with "failed".
        #
        # When drift happens on the initial probe, no click was ever issued, so
        # nothing was saved -> failed. When `save_candidate` already returned
        # True, the candidate IS in the Recruiter pipeline; recording that as
        # `failed` is the ledger disagreeing with reality in the unrecoverable
        # direction — nothing downstream knows to go looking for a save it
        # believes never happened, and the retry ledger re-executes it forever.
        # The run still aborts on the drift either way; it just no longer lies
        # about the side effect it already committed.
        expected_receipt = "failed" if probe_phase == "initial" else "succeeded"
        assert [row["status"] for row in side_effects] == [expected_receipt]

        full_rows = _full_attempt_rows(pipeline)
        assert [row["status"] for row in full_rows] == ["failed"]
        assert full_rows[0]["current_lifecycle_state"] == "failed_retryable"
        assert snippet.profile_url not in pipeline.state.candidates._seen_urls
        assert snippet.profile_url not in pipeline.state.candidates._prior_outcomes
        assert page_report.events == []


def test_pipelined_failed_save_stays_retryable_and_nonterminal():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        pipeline.state.context.browser.save_candidate = AsyncMock(return_value=False)
        pipeline.state.context.browser._last_save_failure_reason = "save_not_persisted"
        pipeline.state.reviewer.side_effects = LinkedInSideEffectsService(
            LinkedInSideEffectsDeps(
                browser=pipeline.state.context.browser,
                stats=pipeline.state.context.stats,
                saved_urls=set(),
                log_path=pipeline.state.context.log_path,
                get_test_mode=lambda: False,
                get_runtime_bridge=lambda: None,
                get_runtime_run_id=lambda: None,
            )
        )
        pipeline.state.candidates._in_flight_urls.add(snippet.profile_url)
        page_report = FakePageReport()

        with patch(
            "linkedin.candidate_review.full_judge",
            return_value=_decision(snippet, "SAVE"),
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ), patch(
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
        ), patch(
            "linkedin.side_effects.human_delay_correlated",
            return_value=0.0,
        ):
            with pytest.raises(
                RuntimeError,
                match="save was not durably confirmed",
            ):
                asyncio.run(
                    _run_pipelined(
                        pipeline,
                        [snippet],
                        page_report=page_report,
                    )
                )

        assert snippet.profile_url not in pipeline.state.candidates._seen_urls
        assert snippet.profile_url not in pipeline.state.candidates._prior_outcomes
        assert snippet.profile_url not in pipeline.state.candidates._in_flight_urls
        assert page_report.events == []


def test_pipelined_stop_drains_in_flight_judgment_before_raising():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context._operator_stop_event = asyncio.Event()
        snippets = [_snippet("A One", 1), _snippet("B Two", 2)]
        _wire_common_pipeline(
            pipeline,
            browser_latency=0.01,
            stop_after_extract="A One",
        )
        page_report = FakePageReport()

        def fake_full_judge(summary, brief, lane_context=None):
            time.sleep(0.05)
            return _decision(snippet=snippets[0], decision="REJECT")

        with patch("linkedin.candidate_review.full_judge", side_effect=fake_full_judge), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
             patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            with pytest.raises(OperatorStopRequested):
                asyncio.run(
                    _run_pipelined(
                        pipeline,
                        snippets,
                        page_report=page_report,
                        progress=Progress(
                            brief_name="test",
                            strings=[SearchString(id=7, name="test string", boolean="engineer")],
                        ),
                    )
                )

        assert page_report.events == [("REJECT", "A One")]
        assert pipeline.state.reviewer.acquisition.extract_profile_summary.await_count == 1
        assert pipeline.state._allocator_service._checkpoint_progress.call_count >= 1


def test_pipelined_circuit_breaker_counts_completion_order(monkeypatch):
    from linkedin import evaluation_loop as orchestrator

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [_snippet(f"Person {idx}", idx) for idx in range(1, 7)]
        _wire_common_pipeline(pipeline)
        sleep_calls: list[float] = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        monkeypatch.setattr(orchestrator.asyncio, "sleep", fake_sleep)

        def fake_full_judge(summary, brief, lane_context=None):
            snippet = snippets[[s.name for s in snippets].index(summary.name)]
            return _decision(
                snippet,
                "JUDGMENT_FAILURE",
                rationale=(
                    "[JUDGMENT_FAILURE: recoverable_error/provider/rate_limit]"
                ),
            )

        with patch("linkedin.candidate_review.full_judge", side_effect=fake_full_judge), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
             patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            asyncio.run(_run_pipelined(pipeline, snippets))

        assert 60 in sleep_calls


def test_pipelined_thread_exception_finishes_attempt_once_as_failure():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        pipeline.state.reviewer.attempts._start_runtime_stage_attempt = MagicMock(return_value=42)
        pipeline.state.reviewer.attempts._finish_runtime_failure_decision = MagicMock()
        pipeline.state.reviewer._finish_runtime_stage_success = MagicMock()

        with patch("linkedin.candidate_review.full_judge", side_effect=RuntimeError("provider boom")), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
             patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            decisions, _, _ = asyncio.run(_run_pipelined(pipeline, [snippet]))

        assert decisions[0].decision == "JUDGMENT_FAILURE"
        pipeline.state.reviewer.attempts._finish_runtime_failure_decision.assert_called_once()
        assert pipeline.state.reviewer.attempts._finish_runtime_failure_decision.call_args.kwargs["attempt_id"] == 42
        pipeline.state.reviewer._finish_runtime_stage_success.assert_not_called()


def test_serial_full_tool_provider_failure_propagates_and_closes_attempt(monkeypatch):
    # Propagation IS the kill-switch side of LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED,
    # default ON since 2026-09-20; the contained path is covered by
    # tests/test_full_eval_containment.py.
    monkeypatch.setattr("shared.config.LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED", False)
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        pipeline.state.reviewer.attempts._start_runtime_stage_attempt = MagicMock(return_value=42)
        pipeline.state.reviewer._abort_runtime_stage_attempt = MagicMock()
        provider_error = RuntimeError("provider status 401")
        provider_error.status_code = 401

        with patch(
            "linkedin.candidate_review.full_judge",
            side_effect=provider_error,
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_V2_FULL_CONTRACT", "tool"
        ):
            with pytest.raises(RuntimeError, match="401"):
                asyncio.run(
                    pipeline.state.reviewer._full_evaluate(
                        snippet,
                        search_string=SearchString(
                            id=7,
                            name="test string",
                            boolean="engineer",
                        ),
                    )
                )

        pipeline.state.reviewer._abort_runtime_stage_attempt.assert_called_once()
        aborted = pipeline.state.reviewer._abort_runtime_stage_attempt.call_args.kwargs
        assert aborted["attempt_id"] == 42
        assert aborted["payload"]["full_tool_transport_failed"] is True
        assert snippet.profile_url not in pipeline.state.candidates._in_flight_urls


def test_pipelined_full_tool_provider_failure_propagates_and_closes_attempt(monkeypatch):
    monkeypatch.setattr("shared.config.LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED", False)
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        pipeline.state.reviewer.attempts._start_runtime_stage_attempt = MagicMock(return_value=42)
        pipeline.state.reviewer._abort_runtime_stage_attempt = MagicMock()
        pipeline.state.reviewer.attempts._finish_runtime_failure_decision = MagicMock()
        provider_error = RuntimeError("provider status 401")
        provider_error.status_code = 401

        with patch(
            "linkedin.candidate_review.full_judge",
            side_effect=provider_error,
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_V2_FULL_CONTRACT", "tool"
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ):
            with pytest.raises(RuntimeError, match="401"):
                asyncio.run(_run_pipelined(pipeline, [snippet]))

        pipeline.state.reviewer._abort_runtime_stage_attempt.assert_called_once()
        aborted = pipeline.state.reviewer._abort_runtime_stage_attempt.call_args.kwargs
        assert aborted["attempt_id"] == 42
        assert aborted["payload"]["full_tool_transport_failed"] is True
        pipeline.state.reviewer.attempts._finish_runtime_failure_decision.assert_not_called()


def test_pipelined_budget_exhaustion_propagates_and_closes_runtime_attempt():
    """The lookahead worker must not turn exhausted provider credit into a
    candidate-level JUDGMENT_FAILURE and continue spending on later profiles."""
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        pipeline.state.reviewer.attempts._start_runtime_stage_attempt = MagicMock(return_value=42)
        # A1 slice 5: this private moved onto RuntimeAttemptService, so the
        # test moves with it (spec A1: "tests moving with the code"). Mocking
        # the Pipeline delegator would force the service to round-trip back
        # out through Pipeline for an intra-cluster call.
        pipeline.state.reviewer.attempts._finish_runtime_stage_failure = MagicMock()

        with patch(
            "linkedin.candidate_review.full_judge",
            side_effect=ApiBudgetExhaustedError("provider credits exhausted"),
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ), patch(
            "linkedin.candidate_review.human_delay_correlated", return_value=0.0
        ):
            with pytest.raises(ApiBudgetExhaustedError, match="credits exhausted"):
                asyncio.run(_run_pipelined(pipeline, [snippet]))

        pipeline.state.reviewer.attempts._finish_runtime_stage_failure.assert_called_once()
        failure = pipeline.state.reviewer.attempts._finish_runtime_stage_failure.call_args.kwargs
        assert failure["attempt_id"] == 42
        assert failure["payload"]["api_budget_exhausted"] is True


def test_serial_full_tool_contract_stops_after_two_closed_calls():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [_snippet("A One", 1), _snippet("B Two", 2)]
        _wire_common_pipeline(pipeline)
        pipeline.state.reviewer.attempts._start_runtime_stage_attempt = MagicMock(side_effect=[41, 42])
        pipeline.state.reviewer.attempts._finish_runtime_failure_decision = MagicMock()

        def malformed_tool_result(summary, brief, lane_context=None):
            del brief, lane_context
            snippet = next(item for item in snippets if item.name == summary.name)
            return _decision(
                snippet,
                "PARSE_FAILURE",
                rationale="[PARSE_FAILURE: terminal_error/parse/cardinality_mismatch]",
            )

        patches = (
            patch(
                "linkedin.candidate_review.full_judge",
                side_effect=malformed_tool_result,
            ),
            patch(
                "linkedin.evaluation_loop.config.LINKEDIN_V2_FULL_CONTRACT",
                "tool",
            ),
            patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0),
        )
        with patches[0], patches[1], patches[2]:
            first = asyncio.run(
                pipeline.state.reviewer._full_evaluate(
                    snippets[0],
                    search_string=SearchString(
                        id=7,
                        name="test string",
                        boolean="engineer",
                    ),
                )
            )
            assert first.decision == "PARSE_FAILURE"
            with pytest.raises(
                RuntimeError,
                match="full tool-contract corruption threshold",
            ):
                asyncio.run(
                    pipeline.state.reviewer._full_evaluate(
                        snippets[1],
                        search_string=SearchString(
                            id=7,
                            name="test string",
                            boolean="engineer",
                        ),
                    )
                )

        assert pipeline.state.context.stats["full_contract_corruptions"] == 2
        assert pipeline.state.reviewer.attempts._finish_runtime_failure_decision.call_count == 2


def test_pipelined_full_tool_contract_stops_after_two_closed_calls():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [_snippet("A One", 1), _snippet("B Two", 2)]
        _wire_common_pipeline(pipeline)
        _wire_real_runtime(pipeline, output_dir=td, snippets=snippets)

        def malformed_tool_result(summary, brief, lane_context=None):
            del brief, lane_context
            snippet = next(item for item in snippets if item.name == summary.name)
            return _decision(
                snippet,
                "PARSE_FAILURE",
                rationale="[PARSE_FAILURE: terminal_error/parse/cardinality_mismatch]",
            )

        with patch(
            "linkedin.candidate_review.full_judge",
            side_effect=malformed_tool_result,
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_V2_FULL_CONTRACT", "tool"
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ):
            with pytest.raises(
                RuntimeError,
                match="full tool-contract corruption threshold",
            ):
                asyncio.run(_run_pipelined(pipeline, snippets))

        assert pipeline.state.context.stats["full_contract_corruptions"] == 2
        assert pipeline.state.context._runtime_state.list_orphaned_attempts(
            source="linkedin",
            brief_id="test-project",
        ) == []
        assert [row["status"] for row in _full_attempt_rows(pipeline)] == [
            "failed",
            "failed",
        ]


def test_two_candidate_open_failure_drains_prior_worker_without_orphans():
    """Opening K may fail while K-1 is paid/in flight; both attempts must close."""

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [_snippet("A One", 1), _snippet("B Two", 2)]
        _wire_common_pipeline(pipeline)
        _wire_real_runtime(pipeline, output_dir=td, snippets=snippets)
        worker_finished = threading.Event()

        async def extract(snippet, *, interest: float = 0.5):
            if snippet.name == "B Two":
                raise GovernorLimitReached("synthetic profile-open cap")
            return SimpleNamespace(profile_summary=_summary(snippet))

        pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
            side_effect=extract
        )

        def fake_full_judge(summary, brief, lane_context=None):
            del brief, lane_context
            time.sleep(0.05)
            worker_finished.set()
            return _decision(snippets[0], "REJECT")

        with patch(
            "linkedin.candidate_review.full_judge",
            side_effect=fake_full_judge,
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ), patch(
            "linkedin.candidate_review.human_delay_correlated", return_value=0.0
        ):
            with pytest.raises(GovernorLimitReached, match="profile-open cap"):
                asyncio.run(_run_pipelined(pipeline, snippets))

        assert worker_finished.is_set()
        assert pipeline.state.context._runtime_state.list_orphaned_attempts(
            source="linkedin",
            brief_id="test-project",
        ) == []
        rows = _full_attempt_rows(pipeline)
        assert [(row["identity_key"], row["status"]) for row in rows] == [
            (snippets[0].profile_url, "succeeded"),
            (snippets[1].profile_url, "failed"),
        ]


@pytest.mark.parametrize(
    ("method_name", "signal_type"),
    [
        ("_full_evaluate", OperatorStopRequested),
        ("_full_evaluate", SessionExpired),
        ("_prepare_full_candidate", OperatorStopRequested),
        ("_prepare_full_candidate", SessionExpired),
    ],
)
def test_profile_extraction_control_signals_abort_and_stay_retryable(
    method_name,
    signal_type,
):
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        search_string, _ = _wire_real_runtime(
            pipeline,
            output_dir=td,
            snippets=[snippet],
        )
        signal = signal_type()
        pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
            side_effect=signal
        )

        with pytest.raises(signal_type):
            asyncio.run(
                getattr(pipeline.state.reviewer, method_name)(
                    snippet,
                    search_string=search_string,
                )
            )

        assert pipeline.state.context._runtime_state.list_orphaned_attempts(
            source="linkedin",
            brief_id="test-project",
        ) == []
        rows = _full_attempt_rows(pipeline)
        assert [(row["identity_key"], row["status"]) for row in rows] == [
            (snippet.profile_url, "failed"),
        ]
        assert json.loads(rows[0]["payload_json"])["force_retryable"] is True
        assert snippet.profile_url not in pipeline.state.candidates._in_flight_urls


def test_two_candidate_prior_budget_failure_closes_opened_successor():
    """A K-1 abort after K opens must fail K's unspawned canonical context."""

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [_snippet("A One", 1), _snippet("B Two", 2)]
        _wire_common_pipeline(pipeline)
        _wire_real_runtime(pipeline, output_dir=td, snippets=snippets)
        worker_finished = threading.Event()

        def exhausted_judge(summary, brief, lane_context=None):
            del summary, brief, lane_context
            worker_finished.set()
            raise ApiBudgetExhaustedError("provider credits exhausted")

        with patch(
            "linkedin.candidate_review.full_judge",
            side_effect=exhausted_judge,
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ), patch(
            "linkedin.candidate_review.human_delay_correlated", return_value=0.0
        ):
            with pytest.raises(ApiBudgetExhaustedError, match="credits exhausted"):
                asyncio.run(_run_pipelined(pipeline, snippets))

        assert worker_finished.is_set()
        assert pipeline.state.context._runtime_state.list_orphaned_attempts(
            source="linkedin",
            brief_id="test-project",
        ) == []
        rows = _full_attempt_rows(pipeline)
        assert [(row["identity_key"], row["status"]) for row in rows] == [
            (snippets[0].profile_url, "failed"),
            (snippets[1].profile_url, "failed"),
        ]
        successor_payload = next(
            row["payload_json"]
            for row in rows
            if row["identity_key"] == snippets[1].profile_url
        )
        assert "logical_call_id" in successor_payload
        assert "iteration_abort:ApiBudgetExhaustedError" in successor_payload


@pytest.mark.parametrize("verdict", ["SAVE", "REJECT"])
def test_pipelined_cancellation_drains_paid_worker_and_closes_attempt(verdict):
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        _wire_real_runtime(pipeline, output_dir=td, snippets=[snippet])
        worker_started = threading.Event()
        release_worker = threading.Event()
        worker_finished = threading.Event()

        def slow_judge(summary, brief, lane_context=None):
            del summary, brief, lane_context
            worker_started.set()
            release_worker.wait(timeout=2)
            worker_finished.set()
            return _decision(snippet, verdict)

        async def cancel_while_paid() -> None:
            task = asyncio.create_task(_run_pipelined(pipeline, [snippet]))
            while not worker_started.is_set():
                await asyncio.sleep(0.001)
            asyncio.get_running_loop().call_later(0.05, release_worker.set)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        with patch(
            "linkedin.candidate_review.full_judge",
            side_effect=slow_judge,
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            False,
        ), patch(
            "linkedin.candidate_review.human_delay_correlated", return_value=0.0
        ):
            asyncio.run(cancel_while_paid())

        assert worker_finished.is_set()
        assert pipeline.state.context._runtime_state.list_orphaned_attempts(
            source="linkedin",
            brief_id="test-project",
        ) == []
        assert [row["status"] for row in _full_attempt_rows(pipeline)] == ["failed"]


        pipeline.state.reviewer.side_effects.handle_save_decision.assert_not_awaited()
        import json
        payload = json.loads(_full_attempt_rows(pipeline)[0]["payload_json"])
        assert payload["paid_judgment"]["decision"] == verdict
        assert payload["profile_summary"]

def test_serial_full_cancellation_during_open_closes_attempt():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)
        search_string, _ = _wire_real_runtime(
            pipeline,
            output_dir=td,
            snippets=[snippet],
        )
        extraction_started = asyncio.Event()

        async def never_finishes(snippet_arg, *, interest: float = 0.5):
            del snippet_arg
            extraction_started.set()
            await asyncio.Event().wait()

        pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
            side_effect=never_finishes
        )

        async def cancel_open() -> None:
            task = asyncio.create_task(
                pipeline.state.reviewer._full_evaluate(
                    snippet,
                    search_string=search_string,
                )
            )
            await extraction_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_open())

        assert pipeline.state.context._runtime_state.list_orphaned_attempts(
            source="linkedin",
            brief_id="test-project",
        ) == []
        assert [row["status"] for row in _full_attempt_rows(pipeline)] == ["failed"]


def test_pipelined_external_shadow_setup_failure_preserves_baseline():
    """Optional evidence setup is analytical shadow work and cannot replace a
    valid baseline decision when its trigger/fetch/diff path raises."""
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)

        with patch(
            "linkedin.candidate_review.full_judge",
            return_value=_decision(snippet, "REJECT"),
        ), patch(
            "linkedin.candidate_review.should_request_external_evidence",
            side_effect=RuntimeError("shadow gate unavailable"),
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            True,
        ), patch(
            "linkedin.candidate_review.human_delay_correlated", return_value=0.0
        ):
            decisions, _, _ = asyncio.run(_run_pipelined(pipeline, [snippet]))

        assert decisions[0].decision == "REJECT"
        events = read_jsonl(Path(td) / "run_log.jsonl")
        assert any(
            row.get("event") == "external_evidence_shadow_unhandled_exception"
            for row in events
        )


def test_pipelined_external_evidence_keeps_peak_full_judgment_concurrency_one():
    """Baseline and augmented judgments remain serial inside the single
    pending profile task even when external evidence is enabled."""
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [_snippet("A One", 1), _snippet("B Two", 2), _snippet("C Three", 3)]
        _wire_common_pipeline(pipeline, browser_latency=0.005)
        active = 0
        peak = 0
        lock = threading.Lock()

        def timed_decision(summary, *_args, **_kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.02)
                snippet = snippets[[s.name for s in snippets].index(summary.name)]
                return _decision(snippet, "REJECT")
            finally:
                with lock:
                    active -= 1

        evidence = ExternalCandidateEvidence(
            trigger_reason="test",
            identity_confidence=1.0,
        )
        trigger = TriggerDecision(should_run=True, reason="test")

        with patch(
            "linkedin.candidate_review.full_judge", side_effect=timed_decision
        ), patch(
            "linkedin.candidate_review.full_judge_with_external_evidence",
            side_effect=timed_decision,
        ), patch(
            "linkedin.candidate_review.should_request_external_evidence",
            return_value=trigger,
        ), patch(
            "linkedin.candidate_review.fetch_external_candidate_evidence",
            return_value=evidence,
        ), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
            True,
        ), patch(
            "linkedin.candidate_review.human_delay_correlated", return_value=0.0
        ):
            decisions, _, _ = asyncio.run(_run_pipelined(pipeline, snippets))

        assert [decision.candidate_name for decision in decisions] == [
            snippet.name for snippet in snippets
        ]
        assert peak == 1


@pytest.mark.parametrize(
    ("save_outcome", "expected_saves", "expected_names"),
    [
        (
            {
                "status": "succeeded",
                "persisted": True,
                "already_present": False,
                "failure_reason": None,
            },
            1,
            ["Signal Save"],
        ),
        (
            {
                "status": "succeeded",
                "persisted": False,
                "already_present": True,
                "failure_reason": None,
            },
            0,
            [],
        ),
        (
            {
                "status": "succeeded",
                "persisted": False,
                "already_present": True,
                "reconciled_self_save": True,
                "failure_reason": None,
            },
            1,
            ["Signal Save"],
        ),
    ],
)
def test_signal_save_physical_metrics_in_pipelined_page_outcome(
    save_outcome,
    expected_saves,
    expected_names,
):
    """The structured full-judgment contract can emit SIGNAL_SAVE; page
    accounting must agree with the save actuator and canonical runtime state."""

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("Signal Save", 1)
        decision = _decision(snippet, "SIGNAL_SAVE")
        decision.save_outcome = save_outcome
        candidates = _candidate_rows([snippet])
        stats = _string_stats()
        search_string = SearchString(id=7, name="test string", boolean="engineer")

        pipeline.state.reviewer._apply_full_eval_page_outcome(
            restore_facial=True, decision=decision,
            snippet=snippet,
            page_num=1,
            all_candidates=candidates,
            string_stats=stats,
            search_string=search_string,
        )

        assert stats["saves"] == expected_saves
        assert stats["facial_yes"] == 1
        assert candidates[0]["outcome"] == "save"
        assert search_string.saves == expected_names


def test_full_eval_pipeline_flag_defaults_off_and_serial_path_uses_no_threads():
    from shared import config

    assert config.FULL_EVAL_PIPELINE_ENABLED is False

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet("A One", 1)
        _wire_common_pipeline(pipeline)

        with patch("linkedin.candidate_review.full_judge", return_value=_decision(snippet, "REJECT")), \
             patch("linkedin.evaluation_loop.asyncio.to_thread", new=AsyncMock()) as to_thread, \
             patch("linkedin.evaluation_loop.config.FULL_EVAL_PIPELINE_ENABLED", False), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_REJECT_CLOSE_MIN_SECONDS", 0), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_REJECT_CLOSE_MAX_SECONDS", 0), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_REJECT_CLOSE_BASE_SECONDS", 0), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS", 0), \
             patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            decision = asyncio.run(
                pipeline.state.reviewer._full_evaluate(
                    snippet,
                    None,
                    SearchString(id=7, name="test string", boolean="engineer"),
                )
            )

        assert decision.decision == "REJECT"
        to_thread.assert_not_awaited()
        assert not read_jsonl(Path(td) / "shadow_final_judgments.jsonl")


def test_pipelined_circuit_breaker_fires_on_inline_extraction_failures(monkeypatch):
    """Extraction-phase failures take the inline-decision arm, which `continue`s
    past the pre-spawn breaker check — the inline arm needs its own check or a
    failing extraction API is hammered without the 60s pause ever firing."""
    from linkedin import evaluation_loop as orchestrator

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [_snippet(f"Person {idx}", idx) for idx in range(1, 7)]
        _wire_common_pipeline(pipeline)
        sleep_calls: list[float] = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        monkeypatch.setattr(orchestrator.asyncio, "sleep", fake_sleep)

        async def failing_extract(snippet, *, interest: float = 0.5):
            raise RuntimeError("extraction API down")

        pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
            side_effect=failing_extract
        )

        def fake_failure_decision(**kwargs):
            name = kwargs.get("candidate_name", "")
            snippet = snippets[[s.name for s in snippets].index(name)]
            return _decision(
                snippet,
                "JUDGMENT_FAILURE",
                rationale=(
                    "[JUDGMENT_FAILURE: recoverable_error/network/timeout]"
                ),
            )

        with patch("linkedin.candidate_review.judgment_failure_decision", side_effect=fake_failure_decision), \
             patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            asyncio.run(_run_pipelined(pipeline, snippets))

        assert 60 in sleep_calls


@pytest.mark.parametrize("cancel", [False, True])
def test_cleanup_diagnostic_cannot_replace_the_original_failure(tmp_path, monkeypatch, cancel):
    import linkedin.candidate_review as review_module

    loop = _make_pipeline(str(tmp_path))
    _wire_common_pipeline(loop)
    snippets = [_snippet("Predecessor", 1), _snippet("Successor", 2)]
    original = asyncio.CancelledError("cancelled extraction") if cancel else RuntimeError("failed extraction")
    prepare = loop.state.reviewer._prepare_full_candidate

    async def fail_successor(snippet, *args, **kwargs):
        if snippet is snippets[1]:
            raise original
        return await prepare(snippet, *args, **kwargs)

    monkeypatch.setattr(loop.state.reviewer, "_prepare_full_candidate", fail_successor)
    monkeypatch.setattr(loop.state.reviewer, "_judge_full_profile",
                        lambda **kwargs: _decision(snippets[0], "REJECT"))
    monkeypatch.setattr(loop.state.reviewer, "_full_evaluate",
                        AsyncMock(side_effect=RuntimeError("cleanup settlement failed")))
    actual_log_event = review_module.log_event
    diagnostics = []

    def fail_cleanup_diagnostic(path, event_type, **kwargs):
        if event_type == "full_pipeline_abort_cleanup_failed":
            diagnostics.append(kwargs)
            raise OSError("cleanup diagnostic failed")
        return actual_log_event(path, event_type, **kwargs)

    monkeypatch.setattr(review_module, "log_event", fail_cleanup_diagnostic)
    with pytest.raises(type(original)) as caught:
        asyncio.run(_run_pipelined(loop, snippets))
    assert caught.value is original
    assert diagnostics == [{"cause_type": type(original).__name__,
                            "cleanup_error_type": "RuntimeError"}]
    loop.state.reviewer._full_evaluate.assert_awaited_once()


@pytest.mark.parametrize("cancel_mode", ["task", "extraction"])
def test_two_candidate_cancellation_retains_judgment_without_starting_save(cancel_mode):
    """Cancellation drains paid work but cannot authorize a new save."""
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippets = [_snippet("Paid Predecessor", 1), _snippet("Opening Successor", 2)]
        _wire_common_pipeline(pipeline)
        _wire_real_runtime(pipeline, output_dir=td, snippets=snippets)
        worker_started = threading.Event()
        release_worker = threading.Event()
        worker_finished = threading.Event()
        successor_started = asyncio.Event()
        original_extract = pipeline.state.reviewer.acquisition.extract_profile_summary

        async def extract(snippet, *, interest=0.5):
            if snippet is snippets[1]:
                successor_started.set()
                if cancel_mode == "extraction":
                    while not worker_started.is_set():
                        await asyncio.sleep(0.001)
                    release_worker.set()
                    raise asyncio.CancelledError
                await asyncio.Event().wait()
            return await original_extract(snippet, interest=interest)

        pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(side_effect=extract)

        def paid_save(summary, brief, lane_context=None):
            worker_started.set()
            release_worker.wait(timeout=2)
            worker_finished.set()
            return _decision(snippets[0], "SAVE")

        async def cancel_successor():
            task = asyncio.create_task(_run_pipelined(pipeline, snippets))
            await successor_started.wait()
            while not worker_started.is_set():
                await asyncio.sleep(0.001)
            asyncio.get_running_loop().call_later(0.05, release_worker.set)
            if cancel_mode == "task":
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        with patch("linkedin.candidate_review.full_judge", side_effect=paid_save), \
             patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
             patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            asyncio.run(cancel_successor())

        assert worker_finished.is_set()
        pipeline.state.reviewer.side_effects.handle_save_decision.assert_not_awaited()
        assert [row["status"] for row in _full_attempt_rows(pipeline)] == ["failed", "failed"]
        assert pipeline.state.context._runtime_state.list_orphaned_attempts(
            source="linkedin", brief_id="test-project",
        ) == []

        import json
        payload = json.loads(_full_attempt_rows(pipeline)[0]["payload_json"])
        assert payload["paid_judgment"]["decision"] == "SAVE"
        assert payload["profile_summary"]
