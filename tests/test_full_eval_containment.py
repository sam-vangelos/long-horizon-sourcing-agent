"""CLO-177 regression lock: a containable provider fault during a FULL
evaluation settles that one candidate and keeps the session alive, bounded by
a consecutive-fault ceiling. Pre-fix, the tool-contract handler re-raised
every judgment exception — one hung Fireworks stream killed string + session
(~13 sessions across AIEL/PAE/CISO).
"""

from linkedin.evaluation_loop import EvaluationLoop
from linkedin.evaluation_state import EvaluationRunState

import asyncio
from unittest.mock import patch

import httpx
import httpcore
import pytest

from shared import config
from tests.test_full_eval_pipelining import (
    _decision,
    _make_pipeline,
    _run_pipelined,
    _snippet,
    _wire_common_pipeline,
    _wire_real_runtime,
)
from linkedin.orchestrator import (
    Pipeline,
    ProjectContextMismatchError,
    _is_containable_facial_page_error,
)


@pytest.fixture
def pipe():
    # __init__ is heavyweight; the disposition helper touches only
    # config + one instance counter, so a bare instance is the honest unit.
    return EvaluationLoop(EvaluationRunState())


@pytest.fixture
def containment_on(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED", True)
    monkeypatch.setattr(config, "LINKEDIN_FULL_EVAL_MAX_CONSECUTIVE_FAULTS", 3)


READ_TIMEOUT = httpx.ReadTimeout("The read operation timed out")


@pytest.mark.structural
def test_flag_defaults_on():
    # Promoted 2026-09-20: every active launcher already exported it true, so
    # the default now matches what campaigns run. Off is the kill switch.
    assert config.LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED is True


def test_read_timeout_is_containable():
    # The exact exception class from the 2026-08-12 CISO session death.
    assert _is_containable_facial_page_error(READ_TIMEOUT) is True
    assert (
        _is_containable_facial_page_error(
            httpcore.ReadTimeout("The read operation timed out")
        )
        is True
    )


def test_flag_off_keeps_fail_fast(pipe, monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED", False)
    assert pipe.state.reviewer._full_judgment_fault_disposition(READ_TIMEOUT) == "raise"
    assert getattr(pipe.state.reviewer, "_full_eval_consecutive_faults", 0) == 0


def test_contain_then_ceiling(pipe, containment_on):
    assert pipe.state.reviewer._full_judgment_fault_disposition(READ_TIMEOUT) == "contain"
    assert pipe.state.reviewer._full_judgment_fault_disposition(READ_TIMEOUT) == "contain"
    assert pipe.state.reviewer._full_judgment_fault_disposition(READ_TIMEOUT) == "ceiling"
    assert pipe.state.reviewer._full_eval_consecutive_faults == 3


def test_success_resets_the_streak(tmp_path, containment_on, monkeypatch):
    # The ceiling counts CONSECUTIVE faults. Two provider faults, one healthy
    # judgment, then another fault: with a ceiling of three the fourth
    # candidate is still contained and the string goes on. Were the success not
    # to reset the streak, that fault would be the third and end the session.
    loop = _make_pipeline(str(tmp_path))
    # The ceiling lives on the tool contract's fault path. Set after
    # construction: the constructor refuses that contract without the
    # Fireworks policy, which this test does not need.
    for name, value in {
        "LINKEDIN_V2_FULL_CONTRACT": "tool",
        "LINKEDIN_EXTERNAL_EVIDENCE_ENABLED": False,
        "LINKEDIN_REJECT_CLOSE_MIN_SECONDS": 0,
        "LINKEDIN_REJECT_CLOSE_MAX_SECONDS": 0,
    }.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr("linkedin.candidate_review.human_delay_correlated", lambda *a, **k: 0)
    snippets = [_snippet(name, rank) for rank, name in enumerate(["A", "B", "Healthy", "D"], 1)]
    _wire_common_pipeline(loop)
    _wire_real_runtime(loop, output_dir=str(tmp_path), snippets=snippets)
    judgments = [
        httpx.ReadTimeout("provider timed out"),
        httpx.ReadTimeout("provider timed out"),
        _decision(snippets[2], "REJECT"),
        httpx.ReadTimeout("provider timed out"),
    ]

    ended_the_session = None
    with patch("linkedin.candidate_review.full_judge", side_effect=judgments):
        try:
            decisions, _, _ = asyncio.run(_run_pipelined(loop, snippets))
        except RuntimeError as exc:
            ended_the_session = exc

    assert ended_the_session is None, "a healthy judgment did not reset the fault streak"
    assert [item.decision for item in decisions] == [
        "JUDGMENT_FAILURE",
        "JUDGMENT_FAILURE",
        "REJECT",
        "JUDGMENT_FAILURE",
    ]
    assert loop.state.reviewer._full_eval_consecutive_faults == 1


def test_non_containable_faults_still_raise_and_do_not_count(pipe, containment_on):
    err = ProjectContextMismatchError("wrong project")
    assert pipe.state.reviewer._full_judgment_fault_disposition(err) == "raise"
    assert getattr(pipe.state.reviewer, "_full_eval_consecutive_faults", 0) == 0


def test_ceiling_floor_is_one(pipe, monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED", True)
    monkeypatch.setattr(config, "LINKEDIN_FULL_EVAL_MAX_CONSECUTIVE_FAULTS", 0)
    # A zero/negative ceiling must not mean "contain forever": floor at 1
    # makes the very first fault terminal.
    assert pipe.state.reviewer._full_judgment_fault_disposition(READ_TIMEOUT) == "ceiling"
