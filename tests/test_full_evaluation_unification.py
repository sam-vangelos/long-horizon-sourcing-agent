"""The two candidate windows share the serial judgment policy and settlement."""
import asyncio
from unittest.mock import patch

import httpx
import pytest

from shared import config
from tests.test_full_eval_pipelining import (
    _decision, _full_attempt_rows, _make_pipeline, _run_pipelined,
    _snippet, _wire_common_pipeline, _wire_real_runtime,
)
from tests.test_full_id_mismatch_retry import _contract_failure_decision, _ScriptedJudge


@pytest.fixture
def judgment_policy(monkeypatch):
    for name, value in {
        "LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED": True,
        "LINKEDIN_FULL_EVAL_MAX_CONSECUTIVE_FAULTS": 3,
        "LINKEDIN_EXTERNAL_EVIDENCE_ENABLED": False,
        "LINKEDIN_REJECT_CLOSE_MIN_SECONDS": 0,
        "LINKEDIN_REJECT_CLOSE_MAX_SECONDS": 0,
    }.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr("linkedin.candidate_review.human_delay_correlated", lambda *args, **kwargs: 0)


def test_lookahead_contains_provider_fault_then_resets_after_success(tmp_path, judgment_policy, monkeypatch):
    loop = _make_pipeline(str(tmp_path))
    monkeypatch.setattr(config, "LINKEDIN_V2_FULL_CONTRACT", "tool")
    snippets = [_snippet("Fault", 1), _snippet("Healthy", 2)]
    _wire_common_pipeline(loop)
    _wire_real_runtime(loop, output_dir=str(tmp_path), snippets=snippets)
    escaped = None
    with patch("linkedin.candidate_review.full_judge", side_effect=[
        httpx.ReadTimeout("recorded provider timeout"), _decision(snippets[1], "REJECT"),
    ]):
        try:
            decisions, _, _ = asyncio.run(_run_pipelined(loop, snippets))
        except Exception as exc:
            escaped = exc
    assert escaped is None, "lookahead bypassed the serial containment policy"
    assert [item.decision for item in decisions] == ["JUDGMENT_FAILURE", "REJECT"]
    assert [row["status"] for row in _full_attempt_rows(loop)] == ["failed", "succeeded"]
    assert loop.state.reviewer._full_eval_consecutive_faults == 0
    assert loop.state.context._runtime_state.list_orphaned_attempts(source="linkedin", brief_id="test-project") == []


def test_lookahead_reissues_candidate_id_mismatch_before_terminal_write(tmp_path, judgment_policy, monkeypatch):
    loop = _make_pipeline(str(tmp_path))
    monkeypatch.setattr(config, "LINKEDIN_V2_FULL_CONTRACT", "tool")
    monkeypatch.setattr(config, "LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED", True)
    snippet = _snippet("Mismatch", 1)
    _wire_common_pipeline(loop)
    _wire_real_runtime(loop, output_dir=str(tmp_path), snippets=[snippet])
    judge = _ScriptedJudge([
        lambda call: _contract_failure_decision(
            snippet, reason="candidate_id_mismatch",
            expected_id="cand_" + "a" * 24, actual_id="cand_" + "b" * 24,
            logical_call_id=call["lane_context"]["logical_call_id"],
        ),
        lambda call: _decision(snippet, "REJECT"),
    ], [])
    with patch("linkedin.candidate_review.full_judge", side_effect=judge):
        decisions, _, _ = asyncio.run(_run_pipelined(loop, [snippet]))
    assert len(judge.calls) == 2, "lookahead bypassed the serial ID-mismatch reissue"
    assert decisions[0].decision == "REJECT"
    rows = _full_attempt_rows(loop)
    assert [row["status"] for row in rows] == ["succeeded"]
    assert judge.child_call_id != judge.parent_call_id
    assert loop.state.reviewer._full_id_mismatch_recovered_count == 1
