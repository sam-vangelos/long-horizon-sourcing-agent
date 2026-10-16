"""Boundary behavior of the one candidate queue, exercised through real stages."""
from tests.evaluation_test_adapters import patch_evaluation_delay
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.test_full_eval_pipelining import (
    _decision, _full_attempt_rows, _make_pipeline, _run_pipelined,
    _snippet, _wire_common_pipeline, _wire_real_runtime,
)


def test_five_inline_failures_at_queue_end_do_not_pause_for_a_sixth(tmp_path):
    loop = _make_pipeline(str(tmp_path))
    snippets = [_snippet(f"Candidate {i}", i) for i in range(1, 6)]
    _wire_common_pipeline(loop)
    loop.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
        side_effect=RuntimeError("recorded acquisition API failure"),
    )
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    def failure(**kwargs):
        snippet = next(s for s in snippets if s.name == kwargs["candidate_name"])
        return _decision(snippet, "JUDGMENT_FAILURE",
                         "[JUDGMENT_FAILURE: recoverable_error/network/timeout]")

    with patch("linkedin.evaluation_loop.asyncio.sleep", sleep), \
         patch("linkedin.candidate_review.judgment_failure_decision", side_effect=failure), \
         patch_evaluation_delay(return_value=0):
        decisions, _, _ = asyncio.run(_run_pipelined(loop, snippets))
    assert len(decisions) == 5
    assert 60 not in sleeps, "queue paused after its last candidate instead of before the next"


def test_pause_follows_canonical_settlement_of_the_current_candidate(tmp_path):
    loop = _make_pipeline(str(tmp_path))
    snippet = _snippet("Pause Boundary", 1)
    _wire_common_pipeline(loop)
    _, progress = _wire_real_runtime(loop, output_dir=str(tmp_path), snippets=[snippet])
    loop.state.context._pause_requested = asyncio.Event()
    original_extract = loop.state.reviewer.acquisition.extract_profile_summary
    observed = []

    async def extract(*args, **kwargs):
        result = await original_extract(*args, **kwargs)
        loop.state.context._pause_requested.set()
        return result

    async def resume():
        observed.append([row["status"] for row in _full_attempt_rows(loop)])

    loop.state.reviewer.acquisition.extract_profile_summary = AsyncMock(side_effect=extract)
    loop.state.context._resume_event = SimpleNamespace(wait=resume)
    with patch("linkedin.candidate_review.full_judge", return_value=_decision(snippet, "REJECT")), \
         patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
         patch_evaluation_delay(return_value=0):
        asyncio.run(_run_pipelined(loop, [snippet], progress=progress))
    assert observed == [["succeeded"]], "pause exposed an unsettled paid candidate"
    assert [row["status"] for row in _full_attempt_rows(loop)] == ["succeeded"]


def test_stop_during_fifth_failure_drain_prevents_the_sixth_open(tmp_path):
    from shared.governor import OperatorStopRequested

    loop = _make_pipeline(str(tmp_path))
    snippets = [_snippet(f"Candidate {i}", i) for i in range(1, 7)]
    _wire_common_pipeline(loop)
    _, progress = _wire_real_runtime(loop, output_dir=str(tmp_path), snippets=snippets)
    loop.state.context._operator_stop_event = asyncio.Event()
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    async def in_process_worker(function, *args, **kwargs):
        return function(*args, **kwargs)

    def judge(summary, **kwargs):
        snippet = next(s for s in snippets if s.name == summary.name)
        if snippet is snippets[4]:
            loop.state.context._operator_stop_event.set()
        return _decision(snippet, "JUDGMENT_FAILURE",
                         "[JUDGMENT_FAILURE: recoverable_error/network/timeout]")

    escaped = None
    with patch("linkedin.evaluation_loop.asyncio.sleep", sleep), \
         patch("linkedin.evaluation_loop.asyncio.to_thread", in_process_worker), \
         patch("linkedin.candidate_review.full_judge", side_effect=judge), \
         patch_evaluation_delay(return_value=0):
        try:
            asyncio.run(_run_pipelined(loop, snippets, progress=progress))
        except OperatorStopRequested as exc:
            escaped = exc
    assert loop.state.reviewer.acquisition.extract_profile_summary.await_count == 5
    assert isinstance(escaped, OperatorStopRequested)
    assert 60 not in sleeps
    assert [row["status"] for row in _full_attempt_rows(loop)] == ["failed"] * 5


def test_stop_during_pause_does_not_start_the_prepared_successors_judgment(tmp_path):
    from shared.governor import OperatorStopRequested

    loop = _make_pipeline(str(tmp_path))
    snippets = [_snippet("Settled", 1), _snippet("Prepared", 2)]
    _wire_common_pipeline(loop)
    _, progress = _wire_real_runtime(loop, output_dir=str(tmp_path), snippets=snippets)
    loop.state.context._operator_stop_event = asyncio.Event()
    loop.state.context._pause_requested = asyncio.Event()
    original_extract = loop.state.reviewer.acquisition.extract_profile_summary
    judged = []

    async def extract(snippet, **kwargs):
        result = await original_extract(snippet, **kwargs)
        if snippet is snippets[1]:
            loop.state.context._pause_requested.set()
        return result

    async def resume():
        loop.state.context._operator_stop_event.set()

    async def in_process_worker(function, *args, **kwargs):
        return function(*args, **kwargs)

    def judge(summary, **kwargs):
        judged.append(summary.name)
        return _decision(next(s for s in snippets if s.name == summary.name), "REJECT")

    loop.state.reviewer.acquisition.extract_profile_summary = AsyncMock(side_effect=extract)
    loop.state.context._resume_event = SimpleNamespace(wait=resume)
    escaped = None
    with patch("linkedin.evaluation_loop.asyncio.to_thread", in_process_worker), \
         patch("linkedin.candidate_review.full_judge", side_effect=judge), \
         patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
         patch_evaluation_delay(return_value=0):
        try:
            asyncio.run(_run_pipelined(loop, snippets, progress=progress))
        except OperatorStopRequested as exc:
            escaped = exc
    assert judged == ["Settled"], "stop during pause admitted another paid judgment"
    assert isinstance(escaped, OperatorStopRequested)
    assert [row["status"] for row in _full_attempt_rows(loop)] == ["succeeded", "failed"]


def test_window_two_resume_drain_honors_a_pre_set_operator_stop(tmp_path):
    """Baseline's pipelined resume drain checked stop authority before every
    open (orchestrator.py@17568fa:12969-12973); window two keeps that contract."""
    import pytest
    from shared.governor import OperatorStopRequested
    from shared.schemas import Progress, SearchString
    from tests.test_full_eval_pipelining import _candidate_rows, _string_stats

    loop = _make_pipeline(str(tmp_path))
    snippets = [_snippet(f"Resumed {i}", i) for i in range(1, 4)]
    _wire_common_pipeline(loop)
    loop.state.context._operator_stop_event = asyncio.Event()
    loop.state.context._operator_stop_event.set()
    search_string = SearchString(id=7, name="test string", boolean="engineer")
    progress = Progress(brief_name="test", strings=[search_string])

    def judge(summary, brief, lane_context=None):
        return _decision(next(s for s in snippets if s.name == summary.name), "REJECT")

    with patch("linkedin.candidate_review.full_judge", side_effect=judge), \
         patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
         patch("linkedin.evaluation_loop.config.FULL_EVAL_PIPELINE_ENABLED", True), \
         patch_evaluation_delay(return_value=0):
        with pytest.raises(OperatorStopRequested):
            asyncio.run(loop.state.reviewer._process_resumed_pending_full_evaluations(
                snippets=snippets, page_report=None, search_string=search_string,
                all_candidates=_candidate_rows(snippets), string_stats=_string_stats(),
                progress=progress, page_num=1,
            ))
    assert loop.state.reviewer.acquisition.extract_profile_summary.await_count == 0


def test_window_two_honors_a_stop_that_arrives_during_the_last_judgment(tmp_path):
    """Baseline re-checked stop/expiry after the trailing drain and raised
    (orchestrator.py@17568fa:13152-13161); the paid result is still settled first."""
    import pytest
    from shared.governor import OperatorStopRequested

    loop = _make_pipeline(str(tmp_path))
    snippet = _snippet("Only One", 1)
    _wire_common_pipeline(loop)
    _, progress = _wire_real_runtime(loop, output_dir=str(tmp_path), snippets=[snippet])
    loop.state.context._operator_stop_event = asyncio.Event()

    def judge(summary, brief, lane_context=None):
        loop.state.context._operator_stop_event.set()
        return _decision(snippet, "REJECT")

    with patch("linkedin.candidate_review.full_judge", side_effect=judge), \
         patch("linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False), \
         patch_evaluation_delay(return_value=0):
        with pytest.raises(OperatorStopRequested):
            asyncio.run(_run_pipelined(loop, [snippet], progress=progress))
    assert [row["status"] for row in _full_attempt_rows(loop)] == ["succeeded"]
