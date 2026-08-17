"""Dispatch LLM cascade tests (Slice 2.6) — mirror TestSynthesisCascade."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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
        role_description="Dispatch cascade fixture brief.",
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


def _heuristic_plan_for(brief: Brief) -> DispatchPlan:
    return ChiefOfStaffAgent()._heuristic_dispatch(brief, [])  # noqa: SLF001


def _good_llm_dispatch(target_modules: list[str]) -> dict:
    return {
        "steps": [
            {"module_name": m, "handoff_condition": None}
            for m in target_modules
        ]
    }


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)


# ---------------------------------------------------------------------------
# Cascade — six routes converge on the heuristic
# ---------------------------------------------------------------------------


class TestDispatchCascade:
    """Each failure mode routes to :meth:`ChiefOfStaffAgent._heuristic_dispatch`."""

    @pytest.fixture(autouse=True)
    def _enable_llm_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._has_llm_access", lambda: True
        )

    def _brief(self) -> Brief:
        return _compat_brief(
            brief_id="brief-cascade-dispatch",
            target_modules=["linkedin", "github"],
        )

    def test_route_no_llm_access(
        self, monkeypatch: pytest.MonkeyPatch, isolated_output_root: Path
    ) -> None:
        del isolated_output_root
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._has_llm_access", lambda: False
        )
        recorder = _Recorder()
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._emit_stage", recorder
        )
        brief = self._brief()
        plan = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])
        assert plan == _heuristic_plan_for(brief)
        joined = "\n".join(recorder.messages)
        assert "dispatch:fallback reason=no_llm_access" in joined

    def test_route_llm_raises(
        self, monkeypatch: pytest.MonkeyPatch, isolated_output_root: Path
    ) -> None:
        del isolated_output_root
        recorder = _Recorder()
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._emit_stage", recorder
        )
        brief = self._brief()
        with patch(
            "cloris.chief_of_staff.agent.opus_llm",
            side_effect=RuntimeError("network down"),
        ):
            plan = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])
        assert plan == _heuristic_plan_for(brief)
        assert "dispatch:fallback reason=llm_raise" in "\n".join(
            recorder.messages
        )

    def test_route_schema_invalid_not_dict(self, isolated_output_root: Path) -> None:
        del isolated_output_root
        brief = self._brief()
        with patch(
            "cloris.chief_of_staff.agent.opus_llm",
            return_value="not a dict",
        ):
            plan = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])
        assert plan == _heuristic_plan_for(brief)

    def test_route_schema_invalid_empty_steps(
        self, isolated_output_root: Path
    ) -> None:
        del isolated_output_root
        brief = self._brief()
        with patch(
            "cloris.chief_of_staff.agent.opus_llm",
            return_value={"steps": []},
        ):
            plan = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])
        assert plan == _heuristic_plan_for(brief)

    def test_route_unknown_source_proposed(self, isolated_output_root: Path) -> None:
        del isolated_output_root
        brief = self._brief()
        bad = _good_llm_dispatch(["linkedin", "github"])
        bad["steps"][0]["module_name"] = "not_a_real_launcher"
        with patch("cloris.chief_of_staff.agent.opus_llm", return_value=bad):
            plan = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])
        assert plan == _heuristic_plan_for(brief)

    def test_route_mfm_dependency_unsatisfied(
        self, monkeypatch: pytest.MonkeyPatch, isolated_output_root: Path
    ) -> None:
        del isolated_output_root
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._exec_search_mfm_ready", lambda: False
        )
        brief = _compat_brief(
            brief_id="brief-mfm",
            target_modules=["linkedin", "exec_search"],
        )
        payload = _good_llm_dispatch(["linkedin", "exec_search"])
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=payload
        ):
            plan = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])
        assert plan == _heuristic_plan_for(brief)

    def test_route_dispatch_loops_back(self, isolated_output_root: Path) -> None:
        del isolated_output_root
        brief = self._brief()
        bad = _good_llm_dispatch(["linkedin", "github"])
        bad["steps"][0]["handoff_condition"] = "when linkedin finishes"
        with patch("cloris.chief_of_staff.agent.opus_llm", return_value=bad):
            plan = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])
        assert plan == _heuristic_plan_for(brief)

    def test_dispatch_persistence_on_fallback_with_persist_true(
        self, isolated_output_root: Path
    ) -> None:
        del isolated_output_root
        brief = _compat_brief(
            brief_id="brief-dispatch-persist-fallback",
            target_modules=["linkedin"],
            principal_id="p1",
        )
        with patch(
            "cloris.chief_of_staff.agent.opus_llm",
            return_value={"steps": []},
        ):
            plan = ChiefOfStaffAgent().dispatch(brief, prior_runs=[], persist=True)
        record = chief_of_staff_run_by_brief(
            output_paths.resolve_orchestration_db_path(),
            brief_id=brief.id,
        )
        assert record is not None
        assert json.loads(record.dispatch_plan_json) == plan.to_dict()


class TestDispatchCascadeIntegrationEquivalence:
    """Heuristic fallback matches explicit :meth:`_heuristic_dispatch` for each route."""

    @pytest.fixture(autouse=True)
    def _enable_llm_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._has_llm_access", lambda: True
        )

    def _brief(self) -> Brief:
        return _compat_brief(
            brief_id="brief-equiv",
            target_modules=["linkedin", "researcher"],
        )

    def test_equivalence_schema_invalid_vs_heuristic(self) -> None:
        brief = self._brief()
        expected = ChiefOfStaffAgent()._heuristic_dispatch(brief, [])  # noqa: SLF001
        with patch(
            "cloris.chief_of_staff.agent.opus_llm",
            return_value={"no": "shape"},
        ):
            got = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])
        assert got.to_dict() == expected.to_dict()

    def test_equivalence_unknown_source_vs_heuristic(self) -> None:
        brief = self._brief()
        expected = ChiefOfStaffAgent()._heuristic_dispatch(brief, [])  # noqa: SLF001
        bad = _good_llm_dispatch(["linkedin", "researcher"])
        bad["steps"][0]["module_name"] = "phantom"
        with patch("cloris.chief_of_staff.agent.opus_llm", return_value=bad):
            got = ChiefOfStaffAgent().dispatch(brief, prior_runs=[])
        assert got.to_dict() == expected.to_dict()
