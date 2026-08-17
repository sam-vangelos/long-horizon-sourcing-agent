"""Tests for planner consumption of lane feedback diffs (D4).

Pins:
- Lane feedback diffs appear in the strategy prompt.
- consumed_feedback_ids field exists on ExecutionPlan.
- Unsupported feedback preserved as warning, not silently dropped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.schemas import ExecutionPlan
from linkedin.strategy import _build_strategy_user


def _minimal_brief():
    """Return a duck-typed brief for prompt assembly."""
    from types import SimpleNamespace

    return SimpleNamespace(
        role_title="ML Engineer",
        role_level="Senior",
        role_summary="Build ML systems",
        geography="NYC",
        linkedin_project="ML Eng",
        linkedin_project_id="123",
        jd_text="",
        intake_notes="",
        search_priorities=[],
        additional_search_terms=[],
        instructions=[],
        has_v2_schema=False,
        raw={},
    )


# -- ExecutionPlan consumed_feedback_ids --

def test_execution_plan_has_consumed_feedback_ids():
    plan = ExecutionPlan(strategy_rationale="test")
    assert hasattr(plan, "consumed_feedback_ids")
    assert plan.consumed_feedback_ids == []


def test_execution_plan_round_trips_consumed_ids():
    plan = ExecutionPlan(
        strategy_rationale="test",
        consumed_feedback_ids=["d1", "d2"],
    )
    d = plan.to_dict()
    restored = ExecutionPlan.from_dict(d)
    assert restored.consumed_feedback_ids == ["d1", "d2"]


# -- Lane feedback in strategy prompt --

def test_lane_feedback_appears_in_prompt():
    brief = _minimal_brief()
    feedback = [
        {
            "diff_id": "d1",
            "action": "add",
            "target_type": "hypothesis",
            "payload": {"label": "ML Platform"},
            "internal_evidence": ["run:1"],
        },
        {
            "diff_id": "d2",
            "action": "retire",
            "target_type": "slice",
            "target_id": "old-slice",
            "payload": {},
        },
    ]
    prompt = _build_strategy_user(brief, [], lane_feedback=feedback)
    assert "Lane Feedback Diffs" in prompt
    assert '"d1"' in prompt
    assert '"d2"' in prompt
    assert "consumed_feedback_ids" in prompt
    assert "retire" in prompt.lower()
    assert "add" in prompt.lower()


def test_no_lane_feedback_section_when_empty():
    brief = _minimal_brief()
    prompt = _build_strategy_user(brief, [], lane_feedback=None)
    assert "Lane Feedback Diffs" not in prompt
    prompt2 = _build_strategy_user(brief, [], lane_feedback=[])
    assert "Lane Feedback Diffs" not in prompt2


def test_prompt_instructs_to_record_consumed_ids():
    brief = _minimal_brief()
    feedback = [{"diff_id": "d1", "action": "add", "target_type": "hypothesis", "payload": {}}]
    prompt = _build_strategy_user(brief, [], lane_feedback=feedback)
    assert "consumed_feedback_ids" in prompt
    assert "silently drop" in prompt.lower()


def test_prompt_includes_validation_question_handling():
    brief = _minimal_brief()
    feedback = [
        {
            "diff_id": "vq1",
            "action": "add",
            "target_type": "validation_question",
            "payload": {"question": "Should we target MLOps?"},
        }
    ]
    prompt = _build_strategy_user(brief, [], lane_feedback=feedback)
    assert "validation_question" in prompt
    assert "strategy rationale" in prompt.lower()
