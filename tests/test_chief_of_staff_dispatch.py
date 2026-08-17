"""Tests for chief-of-staff dispatch heuristic backend (Slice 2.5)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import shared.output_paths as output_paths
from cloris.chief_of_staff.agent import ChiefOfStaffAgent
from cloris.chief_of_staff.decision import DispatchPlan
from shared.brief_loader import Brief
from shared.runtime_state.read_models import chief_of_staff_run_by_brief


@pytest.fixture()
def isolated_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    root = tmp_path / "output"
    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", root)
    return root


def _compat_brief(
    *,
    brief_id: str,
    target_modules: list[str],
    principal_id: str = "",
) -> Brief:
    raw: dict[str, object] = {"target_modules": list(target_modules)}
    if principal_id:
        raw["principal_id"] = principal_id
    return Brief(
        id=brief_id,
        role_title="Principal Applied AI Engineer",
        role_description="Dispatch heuristic fixture brief.",
        kit_url="",
        linkedin_project="",
        linkedin_project_id="",
        minimum_bar="",
        archetypes=[],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
        target_modules=list(target_modules),
        raw=raw,
    )


@pytest.mark.parametrize(
    ("target_modules", "expected"),
    [
        (["linkedin"], ["linkedin"]),
        (["linkedin", "researcher"], ["linkedin", "researcher"]),
        (
            ["linkedin", "github", "exec_search"],
            ["linkedin", "github", "exec_search"],
        ),
    ],
)
def test_dispatch_heuristic_preserves_declared_target_module_order(
    target_modules: list[str],
    expected: list[str],
    isolated_output_root: Path,
) -> None:
    del isolated_output_root  # fixture side effect only
    brief = _compat_brief(
        brief_id=f"brief-{'-'.join(target_modules)}",
        target_modules=target_modules,
    )

    plan = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])

    assert isinstance(plan, DispatchPlan)
    assert [step.module_name for step in plan.steps] == expected
    assert all(step.handoff_condition in (None, "") for step in plan.steps)


def test_dispatch_persists_run_row_and_round_trips_dispatch_plan(
    isolated_output_root: Path,
) -> None:
    del isolated_output_root  # fixture side effect only
    brief = _compat_brief(
        brief_id="brief-dispatch-round-trip",
        target_modules=["linkedin", "github", "researcher"],
        principal_id="principal-northwind",
    )

    plan = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])

    record = chief_of_staff_run_by_brief(
        output_paths.resolve_orchestration_db_path(),
        brief_id=brief.id,
    )
    assert record is not None
    assert record.brief_id == brief.id
    assert record.principal_id == "principal-northwind"
    assert record.status == "running"
    assert json.loads(record.dispatch_plan_json) == plan.to_dict()
    assert json.loads(record.invocation_order_json) == [
        "linkedin",
        "github",
        "researcher",
    ]
    assert json.loads(record.handoff_payloads_json) == {}
    assert json.loads(record.synthesis_output_json) == {}
    assert record.started_at
    assert record.ended_at is None

    # Slice 2.5 contract: each dispatch call inserts a new provenance row.
    _ = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])
    with sqlite3.connect(str(output_paths.resolve_orchestration_db_path())) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM chief_of_staff_runs WHERE brief_id = ?",
            (brief.id,),
        ).fetchone()[0]
    assert count == 2
