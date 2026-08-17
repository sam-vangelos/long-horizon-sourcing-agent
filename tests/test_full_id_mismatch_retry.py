"""Flag-gated re-issue of a full evaluation the model mis-attributed.

The candidate-ID check inside ``validate_full_tool_arguments`` is a security
boundary — profile text is attacker-controlled, so a displaced opaque ID may be
injection rather than sloppiness. These tests lock the shape of the recovery
that sits on top of it: record and classify FIRST, re-ask at most once under a
fresh ID and a child logical call, cap recoveries per session, and leave the
two-strike corruption breaker seeing exactly one strike per corrupted call.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import shared.config as config
import shared.judger as judger
from linkedin.judgment_tool_contracts import JudgmentToolContractError
from shared.execution import SideEffectOutcome
from shared.failures import ApiBudgetExhaustedError, parse_failure_decision
from shared.schemas import (
    CandidateProfileSummary,
    CandidateSnippet,
    OpusDecision,
    SearchString,
)

_OPAQUE_ID_RE = re.compile(r"^cand_[0-9a-f]{24}$")
_SEARCH_STRING = SearchString(id=7, name="test string", boolean="engineer")


def _make_pipeline(output_dir: str):
    with patch("linkedin.orchestrator.load_brief") as mock_brief, \
         patch("linkedin.orchestrator.init_judger"), \
         patch("linkedin.orchestrator.LinkedInBrowser"):
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

        pipeline = Pipeline(brief_path=str(brief_path), output_dir=output_dir)
        pipeline._ensure_services = MagicMock()
        pipeline._runtime_bridge = None
        pipeline._runtime_run_id = None
        pipeline._runtime_state = None
        pipeline._bias_monitor = None
        pipeline._derive_novelty_value = MagicMock(
            return_value=("high", "test rationale")
        )
        pipeline._profile_probe = MagicMock()
        pipeline._profile_probe.evaluate.return_value = "probe"
        pipeline._profile_probe.record_shadow_outcome.return_value = SimpleNamespace(
            to_dict=lambda: {"probe": "recorded"}
        )
        pipeline._checkpoint_progress = MagicMock()
        pipeline._start_runtime_stage_attempt = MagicMock(return_value=41)
        pipeline._abort_runtime_stage_attempt = MagicMock()
        pipeline._finish_runtime_failure_decision = MagicMock()
        return pipeline


def _snippet(name: str = "A One", rank: int = 1) -> CandidateSnippet:
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
        about="I built and owned the core reliability platform.",
    )


def _decision(snippet: CandidateSnippet, decision: str) -> OpusDecision:
    return OpusDecision(
        stage="full",
        decision=decision,
        path="none",
        confidence=0.8,
        rationale=f"{decision} rationale for {snippet.name}",
        candidate_name=snippet.name,
        profile_url=snippet.profile_url,
    )


def _contract_failure_decision(
    snippet: CandidateSnippet,
    *,
    reason: str,
    expected_id: str,
    actual_id: str,
    logical_call_id: str,
) -> OpusDecision:
    """Build the decision ``full_judge`` returns for a failed full contract.

    The structured record is produced by the judger's own helper, so these
    tests break if the channel the orchestrator classifies on ever changes
    shape — they never hand-roll it.
    """

    decision = parse_failure_decision(
        stage="full",
        candidate_name=snippet.name,
        profile_url=snippet.profile_url,
        reason=reason,
        detail=f"expected={expected_id} actual={actual_id}",
    )
    decision.prompt_capture = {
        "logical_call_id": logical_call_id,
        "judgment_contract_failure": judger._full_contract_failure_record(
            JudgmentToolContractError(
                reason,
                f"expected={expected_id} actual={actual_id}",
            ),
            expected_candidate_id=expected_id,
            arguments={"candidate_id": actual_id},
        ),
    }
    return decision


def _wire_browser(pipeline, snippet: CandidateSnippet) -> None:
    browser = MagicMock()
    browser.get_profile_status_summary = AsyncMock(return_value={})
    browser.go_back_to_results = AsyncMock(return_value=None)
    pipeline.browser = browser
    pipeline._acquisition_service = MagicMock()
    pipeline._acquisition_service.extract_profile_summary = AsyncMock(
        return_value=SimpleNamespace(profile_summary=_summary(snippet))
    )
    pipeline._side_effects_service = MagicMock()
    pipeline._side_effects_service.handle_save_decision = AsyncMock(
        return_value=SideEffectOutcome(
            effect_type="linkedin_save",
            status="succeeded",
            payload={"test_mode": True},
        )
    )


def _recorded_events(pipeline) -> list[tuple[str, dict]]:
    return [
        (call.kwargs["event_type"], call.kwargs["payload"])
        for call in pipeline._record_runtime_event.call_args_list
    ]


def _mismatch_payloads(pipeline) -> list[dict]:
    return [
        payload
        for event_type, payload in _recorded_events(pipeline)
        if event_type == "full_candidate_id_mismatch"
    ]


def _run_full_evaluate(pipeline, snippet, judge, *, retry_enabled: bool):
    """Drive the serial full-evaluation path with a scripted judge."""

    with patch("linkedin.orchestrator.full_judge", side_effect=judge), \
         patch("linkedin.orchestrator.config.LINKEDIN_V2_FULL_CONTRACT", "tool"), \
         patch(
             "linkedin.orchestrator.config.LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED",
             retry_enabled,
         ), \
         patch(
             "linkedin.orchestrator.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED",
             False,
         ), \
         patch("linkedin.orchestrator.config.LINKEDIN_REJECT_CLOSE_MIN_SECONDS", 0), \
         patch("linkedin.orchestrator.config.LINKEDIN_REJECT_CLOSE_MAX_SECONDS", 0), \
         patch("linkedin.orchestrator.config.LINKEDIN_REJECT_CLOSE_BASE_SECONDS", 0), \
         patch("linkedin.orchestrator.config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS", 0), \
         patch("linkedin.orchestrator.human_delay_correlated", return_value=0.0):
        return asyncio.run(
            pipeline._full_evaluate(snippet, None, _SEARCH_STRING)
        )


class _ScriptedJudge:
    """Stand in for ``full_judge``, recording how each call was addressed."""

    def __init__(self, responses, timeline: list[tuple[str, object]]) -> None:
        self._responses = list(responses)
        self._timeline = timeline
        self.calls: list[dict] = []

    def __call__(
        self,
        summary,
        brief,
        lane_context=None,
        opaque_candidate_id=None,
    ):
        del summary, brief
        self.calls.append(
            {
                "lane_context": dict(lane_context or {}),
                "opaque_candidate_id": opaque_candidate_id,
            }
        )
        self._timeline.append(("judge_call", len(self.calls)))
        return self._responses[len(self.calls) - 1](self.calls[-1])

    @property
    def parent_call_id(self) -> str:
        return str(self.calls[0]["lane_context"]["logical_call_id"])

    @property
    def child_call_id(self) -> str:
        return str(self.calls[1]["lane_context"]["logical_call_id"])


def _timeline_recorder(pipeline, timeline: list[tuple[str, object]]) -> None:
    def record(*, search_string, event_type, payload):
        del search_string
        timeline.append(("event", event_type))
        return None

    pipeline._record_runtime_event = MagicMock(side_effect=record)


def test_full_id_mismatch_retry_disabled_by_default_surfaces_parse_failure():
    assert config.LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED is False

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline._record_runtime_event = MagicMock()
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        timeline: list[tuple[str, object]] = []
        judge = _ScriptedJudge(
            [
                lambda call: _contract_failure_decision(
                    snippet,
                    reason="candidate_id_mismatch",
                    expected_id="cand_" + "a" * 24,
                    actual_id="cand_" + "b" * 24,
                    logical_call_id=str(
                        call["lane_context"]["logical_call_id"]
                    ),
                )
            ],
            timeline,
        )

        decision = _run_full_evaluate(
            pipeline,
            snippet,
            judge,
            retry_enabled=config.LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED,
        )

        assert decision.decision == "PARSE_FAILURE"
        assert len(judge.calls) == 1
        assert _mismatch_payloads(pipeline) == []
        pipeline._finish_runtime_failure_decision.assert_called_once()
        assert pipeline._full_contract_corruption_call_ids == {
            judge.parent_call_id
        }
        assert pipeline.stats["full_contract_corruptions"] == 1
        assert pipeline._full_id_mismatch_recovered_count == 0


def test_full_id_mismatch_recorded_then_retried_with_fresh_id_and_child_call_id():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        timeline: list[tuple[str, object]] = []
        _timeline_recorder(pipeline, timeline)
        expected_id = "cand_" + "a" * 24
        actual_id = "cand_" + "b" * 24
        judge = _ScriptedJudge(
            [
                lambda call: _contract_failure_decision(
                    snippet,
                    reason="candidate_id_mismatch",
                    expected_id=expected_id,
                    actual_id=actual_id,
                    logical_call_id=str(
                        call["lane_context"]["logical_call_id"]
                    ),
                ),
                lambda call: _decision(snippet, "REJECT"),
            ],
            timeline,
        )

        decision = _run_full_evaluate(
            pipeline, snippet, judge, retry_enabled=True
        )

        # The recovered verdict is what the profile open bought.
        assert decision.decision == "REJECT"
        assert len(judge.calls) == 2

        # Recorded and classified BEFORE the re-ask executed.
        assert timeline.index(("event", "full_candidate_id_mismatch")) < (
            timeline.index(("judge_call", 2))
        )

        payloads = [
            call.kwargs["payload"]
            for call in pipeline._record_runtime_event.call_args_list
            if call.kwargs["event_type"] == "full_candidate_id_mismatch"
        ]
        assert payloads == [
            {
                "candidate_name": snippet.name,
                "profile_url": snippet.profile_url,
                "expected_id": expected_id,
                "actual_id": actual_id,
                "logical_call_id": judge.parent_call_id,
                "retry_logical_call_id": judge.child_call_id,
                "parent_logical_call_id": "",
                "retry_scheduled": True,
                "is_retry_result": False,
                "recovered_mismatches_so_far": 0,
            }
        ]

        # A fresh opaque ID, not the one the model failed to echo.
        retry_id = judge.calls[1]["opaque_candidate_id"]
        assert judge.calls[0]["opaque_candidate_id"] is None
        assert _OPAQUE_ID_RE.match(str(retry_id))
        assert retry_id != expected_id
        assert retry_id != actual_id

        # A distinct child call, parented to the call it replaces.
        assert judge.calls[1]["lane_context"]["parent_logical_call_id"] == (
            judge.parent_call_id
        )
        assert judge.child_call_id != judge.parent_call_id
        assert judge.calls[1]["lane_context"]["retry_reason"] == (
            "candidate_id_mismatch"
        )

        # Nothing reached the failure accounting or the breaker.
        pipeline._finish_runtime_failure_decision.assert_not_called()
        assert pipeline._full_contract_corruption_call_ids == set()
        assert pipeline.stats["full_contract_corruptions"] == 0
        assert pipeline.stats.get("parse_failures", 0) == 0
        assert pipeline._full_id_mismatch_recovered_count == 1


def test_full_id_mismatch_retry_failure_surfaces_parse_failure_with_child_call_id():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        timeline: list[tuple[str, object]] = []
        _timeline_recorder(pipeline, timeline)
        judge = _ScriptedJudge(
            [
                lambda call: _contract_failure_decision(
                    snippet,
                    reason="candidate_id_mismatch",
                    expected_id="cand_" + "a" * 24,
                    actual_id="cand_" + "b" * 24,
                    logical_call_id=str(
                        call["lane_context"]["logical_call_id"]
                    ),
                ),
                lambda call: _contract_failure_decision(
                    snippet,
                    reason="candidate_id_mismatch",
                    expected_id="cand_" + "c" * 24,
                    actual_id="cand_" + "d" * 24,
                    logical_call_id=str(
                        call["lane_context"]["logical_call_id"]
                    ),
                ),
            ],
            timeline,
        )

        decision = _run_full_evaluate(
            pipeline, snippet, judge, retry_enabled=True
        )

        assert decision.decision == "PARSE_FAILURE"
        # Bounded: the failed re-ask is never itself re-asked, and both
        # mis-attributions are classified.
        assert len(judge.calls) == 2
        assert len(_mismatch_payloads(pipeline)) == 2
        pipeline._finish_runtime_failure_decision.assert_called_once()
        # The failure row names the call that actually produced the verdict
        # without disturbing the call the attempt was opened under.
        closed = pipeline._finish_runtime_failure_decision.call_args.kwargs
        assert closed["payload"]["logical_call_id"] == judge.parent_call_id
        assert closed["payload"]["verdict_logical_call_id"] == (
            judge.child_call_id
        )
        pipeline._abort_runtime_stage_attempt.assert_not_called()
        assert pipeline._full_contract_corruption_call_ids == {
            judge.child_call_id
        }
        assert pipeline.stats["full_contract_corruptions"] == 1
        assert pipeline._full_id_mismatch_recovered_count == 0


def test_full_id_mismatch_retry_budget_exhausted_surfaces_directly():
    from linkedin import orchestrator

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline._full_id_mismatch_recovered_count = (
            orchestrator._FULL_ID_MISMATCH_RECOVERY_CEILING
        )
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        timeline: list[tuple[str, object]] = []
        _timeline_recorder(pipeline, timeline)
        judge = _ScriptedJudge(
            [
                lambda call: _contract_failure_decision(
                    snippet,
                    reason="candidate_id_mismatch",
                    expected_id="cand_" + "a" * 24,
                    actual_id="cand_" + "b" * 24,
                    logical_call_id=str(
                        call["lane_context"]["logical_call_id"]
                    ),
                )
            ],
            timeline,
        )

        decision = _run_full_evaluate(
            pipeline, snippet, judge, retry_enabled=True
        )

        assert decision.decision == "PARSE_FAILURE"
        assert len(judge.calls) == 1
        payloads = _mismatch_payloads(pipeline)
        assert len(payloads) == 1
        assert payloads[0]["retry_scheduled"] is False
        assert payloads[0]["retry_logical_call_id"] == ""
        assert payloads[0]["is_retry_result"] is False
        assert payloads[0]["recovered_mismatches_so_far"] == 3
        assert payloads[0]["logical_call_id"] == judge.parent_call_id
        assert pipeline._full_contract_corruption_call_ids == {
            judge.parent_call_id
        }
        assert pipeline._full_id_mismatch_recovered_count == 3


def test_full_non_mismatch_parse_failures_never_retry():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        timeline: list[tuple[str, object]] = []
        _timeline_recorder(pipeline, timeline)
        judge = _ScriptedJudge(
            [
                lambda call: _contract_failure_decision(
                    snippet,
                    reason="missing_keys",
                    expected_id="cand_" + "a" * 24,
                    actual_id="cand_" + "a" * 24,
                    logical_call_id=str(
                        call["lane_context"]["logical_call_id"]
                    ),
                )
            ],
            timeline,
        )

        decision = _run_full_evaluate(
            pipeline, snippet, judge, retry_enabled=True
        )

        assert decision.decision == "PARSE_FAILURE"
        assert len(judge.calls) == 1
        assert _mismatch_payloads(pipeline) == []
        pipeline._finish_runtime_failure_decision.assert_called_once()
        assert pipeline._full_contract_corruption_call_ids == {
            judge.parent_call_id
        }
        assert pipeline._full_id_mismatch_recovered_count == 0


def test_full_id_mismatch_event_truncates_the_untrusted_echoed_id():
    """The echoed ID is model output; the receipt stores a bounded slice."""

    from linkedin import orchestrator

    hostile_actual = "cand_" + "e" * 240
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        timeline: list[tuple[str, object]] = []
        _timeline_recorder(pipeline, timeline)
        judge = _ScriptedJudge(
            [
                lambda call: _contract_failure_decision(
                    snippet,
                    reason="candidate_id_mismatch",
                    expected_id="cand_" + "a" * 24,
                    actual_id=hostile_actual,
                    logical_call_id=str(
                        call["lane_context"]["logical_call_id"]
                    ),
                ),
                lambda call: _decision(snippet, "REJECT"),
            ],
            timeline,
        )

        _run_full_evaluate(pipeline, snippet, judge, retry_enabled=True)

        payloads = _mismatch_payloads(pipeline)
        assert len(payloads) == 1
        stored = payloads[0]["actual_id"]
        assert len(stored) == orchestrator._FULL_ID_MISMATCH_ACTUAL_ID_MAX_CHARS
        assert stored == hostile_actual[
            : orchestrator._FULL_ID_MISMATCH_ACTUAL_ID_MAX_CHARS
        ]


class _HardStop(BaseException):
    """A control-flow signal that does not inherit from Exception."""


def _mismatch_first_then(snippet, timeline, second):
    """Script: mis-attributed first call, caller-supplied second response."""

    return _ScriptedJudge(
        [
            lambda call: _contract_failure_decision(
                snippet,
                reason="candidate_id_mismatch",
                expected_id="cand_" + "a" * 24,
                actual_id="cand_" + "b" * 24,
                logical_call_id=str(call["lane_context"]["logical_call_id"]),
            ),
            second,
        ],
        timeline,
    )


def test_full_id_mismatch_receipt_write_failure_closes_the_attempt_once():
    """The receipt is not fail-soft; a raising write must not strand the row."""

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        pipeline._record_runtime_event = MagicMock(
            side_effect=RuntimeError("runtime state unavailable")
        )
        timeline: list[tuple[str, object]] = []
        judge = _mismatch_first_then(
            snippet, timeline, lambda call: _decision(snippet, "REJECT")
        )

        with pytest.raises(RuntimeError, match="runtime state unavailable"):
            _run_full_evaluate(pipeline, snippet, judge, retry_enabled=True)

        # Never re-asked, closed exactly once, and closed as an abort.
        assert len(judge.calls) == 1
        pipeline._abort_runtime_stage_attempt.assert_called_once()
        aborted = pipeline._abort_runtime_stage_attempt.call_args.kwargs
        assert aborted["attempt_id"] == 41
        assert aborted["payload"]["run_abort"] == (
            "full_id_mismatch_interception_failed"
        )
        assert aborted["payload"]["logical_call_id"] == judge.parent_call_id
        pipeline._finish_runtime_failure_decision.assert_not_called()


def test_full_id_mismatch_retry_budget_exhaustion_closes_the_attempt_once():
    def _exhausted(call):
        raise ApiBudgetExhaustedError("provider credits exhausted")

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        timeline: list[tuple[str, object]] = []
        _timeline_recorder(pipeline, timeline)
        judge = _mismatch_first_then(snippet, timeline, _exhausted)

        with pytest.raises(ApiBudgetExhaustedError, match="credits exhausted"):
            _run_full_evaluate(pipeline, snippet, judge, retry_enabled=True)

        assert len(judge.calls) == 2
        pipeline._abort_runtime_stage_attempt.assert_called_once()
        aborted = pipeline._abort_runtime_stage_attempt.call_args.kwargs
        assert aborted["payload"]["api_budget_exhausted"] is True
        assert "full_id_mismatch_retry_failed" not in aborted["payload"]
        assert isinstance(aborted["error"], ApiBudgetExhaustedError)
        pipeline._finish_runtime_failure_decision.assert_not_called()


def test_full_id_mismatch_retry_transport_failure_closes_the_attempt_once():
    def _transport_error(call):
        raise RuntimeError("provider status 500")

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        timeline: list[tuple[str, object]] = []
        _timeline_recorder(pipeline, timeline)
        judge = _mismatch_first_then(snippet, timeline, _transport_error)

        with pytest.raises(RuntimeError, match="provider status 500"):
            _run_full_evaluate(pipeline, snippet, judge, retry_enabled=True)

        assert len(judge.calls) == 2
        pipeline._abort_runtime_stage_attempt.assert_called_once()
        aborted = pipeline._abort_runtime_stage_attempt.call_args.kwargs
        assert aborted["payload"]["full_id_mismatch_retry_failed"] is True
        assert "run_abort" not in aborted["payload"]
        pipeline._finish_runtime_failure_decision.assert_not_called()


def test_full_id_mismatch_retry_base_exception_closes_the_attempt_once():
    def _hard_stop(call):
        raise _HardStop("cancelled mid re-ask")

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        timeline: list[tuple[str, object]] = []
        _timeline_recorder(pipeline, timeline)
        judge = _mismatch_first_then(snippet, timeline, _hard_stop)

        with pytest.raises(_HardStop):
            _run_full_evaluate(pipeline, snippet, judge, retry_enabled=True)

        assert len(judge.calls) == 2
        pipeline._abort_runtime_stage_attempt.assert_called_once()
        aborted = pipeline._abort_runtime_stage_attempt.call_args.kwargs
        assert aborted["payload"]["full_id_mismatch_retry_failed"] is True
        assert isinstance(aborted["error"], _HardStop)
        pipeline._finish_runtime_failure_decision.assert_not_called()


def test_full_double_mismatch_is_classified_as_a_retry_result():
    """Two misses under two unrelated IDs is the loudest injection signal."""

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        timeline: list[tuple[str, object]] = []
        _timeline_recorder(pipeline, timeline)
        second_expected = "cand_" + "c" * 24
        second_actual = "cand_" + "d" * 24
        judge = _mismatch_first_then(
            snippet,
            timeline,
            lambda call: _contract_failure_decision(
                snippet,
                reason="candidate_id_mismatch",
                expected_id=second_expected,
                actual_id=second_actual,
                logical_call_id=str(call["lane_context"]["logical_call_id"]),
            ),
        )

        decision = _run_full_evaluate(
            pipeline, snippet, judge, retry_enabled=True
        )

        assert decision.decision == "PARSE_FAILURE"
        payloads = _mismatch_payloads(pipeline)
        assert len(payloads) == 2
        assert payloads[1] == {
            "candidate_name": snippet.name,
            "profile_url": snippet.profile_url,
            "expected_id": second_expected,
            "actual_id": second_actual,
            "logical_call_id": judge.child_call_id,
            "retry_logical_call_id": "",
            "parent_logical_call_id": judge.parent_call_id,
            "retry_scheduled": False,
            "is_retry_result": True,
            "recovered_mismatches_so_far": 0,
        }
        # The second receipt is written as soon as the re-ask returns, before
        # the decision is handed back to the accounting path.
        second_call = timeline.index(("judge_call", 2))
        assert timeline[second_call + 1] == (
            "event",
            "full_candidate_id_mismatch",
        )
        pipeline._abort_runtime_stage_attempt.assert_not_called()
        pipeline._finish_runtime_failure_decision.assert_called_once()
        assert pipeline._full_id_mismatch_recovered_count == 0


def test_recovered_verdict_attempt_row_names_the_child_call():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        snippet = _snippet()
        _wire_browser(pipeline, snippet)
        pipeline._finish_runtime_stage_success = MagicMock()
        timeline: list[tuple[str, object]] = []
        _timeline_recorder(pipeline, timeline)
        judge = _mismatch_first_then(
            snippet, timeline, lambda call: _decision(snippet, "REJECT")
        )

        decision = _run_full_evaluate(
            pipeline, snippet, judge, retry_enabled=True
        )

        assert decision.decision == "REJECT"
        pipeline._finish_runtime_stage_success.assert_called_once()
        closed = pipeline._finish_runtime_stage_success.call_args.kwargs
        # The opening call ID stays truthful; the child is named beside it.
        assert closed["extra_payload"]["logical_call_id"] == (
            judge.parent_call_id
        )
        assert closed["extra_payload"]["verdict_logical_call_id"] == (
            judge.child_call_id
        )
        pipeline._abort_runtime_stage_attempt.assert_not_called()
        pipeline._finish_runtime_failure_decision.assert_not_called()


def _tool_brief():
    inner = MagicMock()
    inner.facial_ambiguity_posture = "binary"
    inner.dossier_mode = False
    inner.role_title = "Test Role"
    inner.capability_area_names.return_value = ["Core Systems"]
    inner.post_save_modifiers = []
    brief = MagicMock()
    brief.has_v2_schema = True
    brief._new_brief = inner
    brief.id = "test-brief"
    return brief


def _tool_payload(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "decision": "SAVE",
        "match_type": "DIRECT",
        "capability_area": "Core Systems",
        "capability_evidence": "Built the target system.",
        "depth": "BUILDER",
        "depth_evidence": "Designed and owned it.",
        "transferability": "N/A",
        "transferability_evidence": "N/A for direct match.",
        "evidence_recency": "CURRENT",
        "level_alignment": "ALIGNED",
        "opportunity_coherence": "COHERENT",
        "caliber": "STRONG",
        "outreach_tier": "STANDARD",
        "reject_reason": None,
        "case_for": "Direct evidence.",
        "case_against": "Limited scale detail.",
        "confidence": 0.81,
        "post_save_modifier": "NONE",
        "review_reason_code": None,
        "review_structural_evidence": [],
        "review_recommended_next_step": None,
        "summary": "Strong direct fit.",
    }


def test_full_judge_publishes_the_structured_contract_failure_reason(monkeypatch):
    """The classification channel is produced by the production judge path.

    Without this, the orchestrator tests above would only prove the retry
    works against a record the tests themselves invented.
    """

    expected_id = "cand_" + "a" * 24
    actual_id = "cand_" + "b" * 24
    monkeypatch.setattr(config, "LINKEDIN_V2_FULL_CONTRACT", "tool")
    monkeypatch.setattr(config, "FIREWORKS_JUDGMENT_POLICY_ENABLED", False)
    monkeypatch.setattr(
        judger,
        "generate_opaque_candidate_ids",
        lambda count: (expected_id,),
    )

    with patch.object(
        judger,
        "assemble_full_evaluation_tool_system",
        return_value="STABLE SYSTEM",
    ), patch.object(
        judger,
        "opus_llm_cached",
        return_value=_tool_payload(actual_id),
    ):
        decision = judger.full_judge(
            _summary(_snippet()),
            _tool_brief(),
            lane_context={"logical_call_id": "parent-call"},
        )

    assert decision.decision == "PARSE_FAILURE"
    assert decision.prompt_capture["judgment_contract_failure"] == {
        "reason": "candidate_id_mismatch",
        "expected_candidate_id": expected_id,
        "actual_candidate_id": actual_id,
    }
    # The reason token never travels as prose: it is the validator's own.
    assert decision.rationale.startswith(
        "[PARSE_FAILURE: terminal_error/parse/candidate_id_mismatch]"
    )


def test_full_judge_structured_failure_distinguishes_other_contract_breaks(
    monkeypatch,
):
    expected_id = "cand_" + "a" * 24
    monkeypatch.setattr(config, "LINKEDIN_V2_FULL_CONTRACT", "tool")
    monkeypatch.setattr(config, "FIREWORKS_JUDGMENT_POLICY_ENABLED", False)
    monkeypatch.setattr(
        judger,
        "generate_opaque_candidate_ids",
        lambda count: (expected_id,),
    )
    payload = _tool_payload(expected_id)
    payload.pop("summary")

    with patch.object(
        judger,
        "assemble_full_evaluation_tool_system",
        return_value="STABLE SYSTEM",
    ), patch.object(
        judger,
        "opus_llm_cached",
        return_value=payload,
    ):
        decision = judger.full_judge(
            _summary(_snippet()),
            _tool_brief(),
            lane_context={"logical_call_id": "parent-call"},
        )

    assert decision.decision == "PARSE_FAILURE"
    failure = decision.prompt_capture["judgment_contract_failure"]
    assert failure["reason"] != "candidate_id_mismatch"
    assert failure["expected_candidate_id"] == expected_id


def test_full_id_mismatch_retry_context_parents_a_fresh_call():
    context = judger.full_id_mismatch_retry_context(
        {"logical_call_id": "parent-call", "lane_id": "lane-1"},
        parent_logical_call_id="parent-call",
    )

    assert context["parent_logical_call_id"] == "parent-call"
    assert context["logical_call_id"] != "parent-call"
    assert context["logical_call_id"].startswith("judge-")
    assert context["retry_reason"] == "candidate_id_mismatch"
    # Lane attribution survives so the re-ask is billed to the same lane.
    assert context["lane_id"] == "lane-1"
