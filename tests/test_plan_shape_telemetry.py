"""Primary-plan structural telemetry warning coverage."""

from __future__ import annotations

from linkedin.strategy import (
    _PLAN_SHAPE_WARNING_THRESHOLDS,
    _attach_plan_shape_telemetry,
    _finalize_execution_plan,
)
from shared.brief_loader import Brief
from shared.schemas import ExecutionPlan


def _brief() -> Brief:
    return Brief(
        id="plan-shape-test",
        role_title="AI Data Operations Lead",
        role_description="Owns data operations for AI training and evaluation.",
        kit_url="",
        linkedin_project="",
        linkedin_project_id="",
        minimum_bar="Has led production-grade AI data workflows.",
        archetypes=[{"name": "AI data operations leader"}],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
    )


def _plan(booleans: list[str]) -> ExecutionPlan:
    return ExecutionPlan(
        strategy_rationale="test",
        generated_strings=[
            {
                "boolean": boolean,
                "rationale": "Find qualified AI data operations leaders.",
            }
            for boolean in booleans
        ],
    )


def _plan_shape_warnings(plan: ExecutionPlan) -> list[dict]:
    return [
        warning
        for warning in plan.plan_warnings
        if warning.get("code") == "plan_shape_telemetry"
    ]


def _monoculture_booleans() -> list[str]:
    return [
        '("a" OR "b") AND ("c" OR "d")',
        '("e" OR "f") AND ("g" OR "h")',
        '("i" OR "j") AND ("k" OR "l")',
        '("m" OR "n") AND ("o" OR "p")',
        '("q" OR "r") AND ("s" OR "t")',
    ]


def test_live_shaped_monoculture_plan_warns_once_and_prints_one_line(capsys):
    assert _PLAN_SHAPE_WARNING_THRESHOLDS == {
        "max_skeleton_share": 0.5,
        "distinct_skeletons_min": 3,
        "not_usage_rate": 0.6,
        "min_strings": 5,
    }
    plan = _plan(_monoculture_booleans())

    finalized = _finalize_execution_plan(_brief(), plan, prior_run_data=None)

    warnings = _plan_shape_warnings(finalized)
    assert len(warnings) == 1
    message = warnings[0]["message"]
    assert "max_skeleton_share=1.0>0.5" in message
    assert "distinct_skeletons=1<3" in message
    assert "not_usage_rate" not in message
    assert "n_strings" not in message
    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines == [f"  [plan-shape] {message}"]
    assert "{" not in output_lines[0]
    assert "}" not in output_lines[0]


def test_diverse_plan_does_not_warn_or_print(capsys):
    plan = _plan(
        [
            '"a"',
            '"b" AND "c"',
            '("d" OR "e")',
            '("f" OR "g") AND "h"',
            '"i" OR "j"',
        ]
    )

    _attach_plan_shape_telemetry(plan)

    assert _plan_shape_warnings(plan) == []
    assert capsys.readouterr().out == ""


def test_tiny_monoculture_plan_does_not_warn(capsys):
    plan = _plan(
        [
            '("a" OR "b") AND ("c" OR "d")',
            '("e" OR "f") AND ("g" OR "h")',
        ]
    )

    _attach_plan_shape_telemetry(plan)

    assert _plan_shape_warnings(plan) == []
    assert capsys.readouterr().out == ""


def test_plan_shape_warning_dedupes_on_reentry(capsys):
    plan = _plan(_monoculture_booleans())

    _attach_plan_shape_telemetry(plan)
    _attach_plan_shape_telemetry(plan)

    assert len(_plan_shape_warnings(plan)) == 1
    assert len(capsys.readouterr().out.splitlines()) == 2
