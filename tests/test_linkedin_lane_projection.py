"""Characterization tests for the linkedin.orchestrator lane-projection cluster.

These pin CURRENT behavior of the seven lane-projection helpers so the Phase 4
P4-3 extraction into ``linkedin/lane_projection.py`` is provably
behavior-preserving. They call through ``Pipeline`` (the names' historical home);
after extraction orchestrator thin-delegates to the sibling module, so these must
stay green before and after the move.
"""

from __future__ import annotations
from linkedin import lane_projection as _lane_projection

from linkedin.evaluation_loop import EvaluationLoop
from linkedin.orchestrator import Pipeline as PipelineFacade

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from linkedin.orchestrator import Pipeline
from linkedin.strategy_lane_compiler import apply_linkedin_lane_compiler_to_plan
from shared.schemas import ExecutionPlan
from shared.sourcing_lanes import (
    LaneExecution,
    SearchConstraint,
    SearchHypothesis,
    SearchSlice,
    SourcingLane,
)


def _make_pipeline(output_dir: str) -> EvaluationLoop:
    with patch("linkedin.evaluation_loop.load_brief") as mock_brief, \
         patch("linkedin.evaluation_loop.init_judger"), \
         patch("linkedin.evaluation_loop.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        mock_brief.return_value = brief

        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        return PipelineFacade(brief_path=str(brief_path), output_dir=output_dir)._loop


def _hybrid_lane_dict(*, lane_id: str = "stripe_alumni") -> dict:
    lane = SourcingLane(
        lane_id=lane_id,
        lane_name="Stripe alumni",
        hypothesis=SearchHypothesis(
            hypothesis_id="h1",
            label="Stripe",
            target_archetype="payments",
            why_this_pool_may_exist="employer anchor",
        ),
        slice=SearchSlice(
            slice_id="s1",
            hypothesis_id="h1",
            label="Slice",
            objective="find stripe alumni",
            constraints=[
                SearchConstraint(
                    dimension="company",
                    values=["Stripe"],
                    execution_surface="linkedin_company_filter",
                    operator="prefer",
                )
            ],
        ),
        execution=LaneExecution(
            lane_id=lane_id,
            source="linkedin",
            acquisition_mode="linkedin_hybrid",
            boolean_strategy={"root_boolean": '"payments" AND engineer'},
            structured_filters={"companies": ["Stripe"]},
        ),
    )
    return lane.to_dict()


# ---------------------------------------------------------------------------
# _lane_projection_aliases
# ---------------------------------------------------------------------------


def test_lane_projection_aliases_strips_fam_prefix():
    aliases = _lane_projection.lane_projection_aliases("fam_stripe_alumni")
    assert aliases == {"fam_stripe_alumni", "stripe_alumni"}


def test_lane_projection_aliases_strips_fde_prefix():
    aliases = _lane_projection.lane_projection_aliases("fde_senior_pool")
    assert aliases == {"fde_senior_pool", "senior_pool"}


def test_lane_projection_aliases_empty_for_blank_lane_id():
    assert _lane_projection.lane_projection_aliases("") == set()
    assert _lane_projection.lane_projection_aliases("   ") == set()


# ---------------------------------------------------------------------------
# _work_item_lane_key
# ---------------------------------------------------------------------------


def test_work_item_lane_key_prefers_lane_id():
    key = _lane_projection.work_item_lane_key(
        {
            "lane_id": "fam_stripe_alumni",
            "family_key": "other",
            "retrieval_recipe": {"family_id": "recipe"},
        }
    )
    assert key == "fam_stripe_alumni"


def test_work_item_lane_key_falls_back_to_family_key():
    key = _lane_projection.work_item_lane_key(
        {"family_key": "stripe_alumni", "retrieval_recipe": {"family_id": "recipe"}}
    )
    assert key == "stripe_alumni"


def test_work_item_lane_key_falls_back_to_retrieval_recipe_family_id():
    key = _lane_projection.work_item_lane_key({"retrieval_recipe": {"family_id": "recipe_lane"}})
    assert key == "recipe_lane"


# ---------------------------------------------------------------------------
# _lane_snapshot_filters
# ---------------------------------------------------------------------------


def test_lane_snapshot_filters_extracts_structured_filters():
    filters = _lane_projection.lane_snapshot_filters(
        {
            "query_payload": {
                "structured_filters": {
                    "companies": ["Stripe"],
                    "titles": [],
                    "skills": [],
                    "assessments": [],
                    "sidebar_filters": {},
                    "advanced_filters": {},
                }
            }
        }
    )
    assert filters.companies == ["Stripe"]


def test_lane_snapshot_filters_empty_when_query_payload_missing():
    filters = _lane_projection.lane_snapshot_filters({})
    assert filters.is_empty()


# ---------------------------------------------------------------------------
# _match_lane_projection
# ---------------------------------------------------------------------------


def test_match_lane_projection_exact_alias_hit():
    records = [
        {
            "lane_id": "stripe_alumni",
            "aliases": {"stripe_alumni", "fam_stripe_alumni"},
            "snapshot": {},
        }
    ]
    item = {"lane_id": "fam_stripe_alumni"}
    matched = _lane_projection.match_lane_projection(item, records)
    assert matched is records[0]


def test_match_lane_projection_prefix_alias_hit():
    records = [
        {
            "lane_id": "stripe_alumni",
            "aliases": {"stripe_alumni"},
            "snapshot": {},
        }
    ]
    item = {"lane_id": "stripe_alumni_recall"}
    matched = _lane_projection.match_lane_projection(item, records)
    assert matched is records[0]


def test_match_lane_projection_none_when_no_key():
    records = [{"lane_id": "stripe_alumni", "aliases": {"stripe_alumni"}, "snapshot": {}}]
    assert _lane_projection.match_lane_projection({}, records) is None


# ---------------------------------------------------------------------------
# _apply_lane_projection_to_work_item
# ---------------------------------------------------------------------------


def test_apply_lane_projection_sets_hybrid_surface_when_boolean_present():
    record = {
        "lane_id": "stripe_alumni",
        "lane_name": "Stripe alumni",
        "lane_intent": "employer anchor",
        "snapshot": {
            "acquisition_mode": "linkedin_hybrid",
            "query_payload": {"boolean": '"payments" AND engineer'},
        },
    }
    projected = _lane_projection.apply_lane_projection_to_work_item(
        {"boolean": '"payments" AND engineer', "family_key": "stripe_alumni"},
        record,
        boolean_key="boolean",
    )
    assert projected["lane_id"] == "stripe_alumni"
    assert projected["lane_name"] == "Stripe alumni"
    assert projected["lane_intent"] == "employer anchor"
    assert projected["acquisition_mode"] == "linkedin_hybrid"
    assert projected["surface"] == "hybrid"
    assert projected["lane_snapshot"]["compiler"] == record["snapshot"]


def test_apply_lane_projection_sets_structured_only_surface_without_boolean():
    record = {
        "lane_id": "stripe_alumni",
        "lane_name": "Stripe alumni",
        "lane_intent": "employer anchor",
        "snapshot": {"acquisition_mode": "linkedin_hybrid", "query_payload": {}},
    }
    projected = _lane_projection.apply_lane_projection_to_work_item(
        {"suggested_boolean": "", "family_key": "stripe_alumni"},
        record,
        boolean_key="suggested_boolean",
    )
    assert projected["surface"] == "structured_only"


def test_apply_lane_projection_preserves_existing_surface():
    record = {
        "lane_id": "stripe_alumni",
        "lane_name": "Stripe alumni",
        "lane_intent": "",
        "snapshot": {"acquisition_mode": "linkedin_hybrid", "query_payload": {}},
    }
    projected = _lane_projection.apply_lane_projection_to_work_item(
        {"boolean": "x", "surface": "boolean"},
        record,
        boolean_key="boolean",
    )
    assert projected["surface"] == "boolean"


# ---------------------------------------------------------------------------
# _current_lane_compiler_snapshot / _structured_lane_projection_records
# ---------------------------------------------------------------------------


def test_current_lane_compiler_snapshot_returns_complete_snapshot_as_is():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        lane_dict = _hybrid_lane_dict()
        apply_linkedin_lane_compiler_to_plan(
            ExecutionPlan(strategy_rationale="test", sourcing_lanes=[lane_dict])
        )
        snapshot = lane_dict["lane_compiler"]
        assert str(snapshot["query_payload"]["boolean"]).strip()
        assert _lane_projection.current_lane_compiler_snapshot(lane_dict) == snapshot


def test_structured_lane_projection_records_hybrid_lane_with_filters():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        lane_dict = _hybrid_lane_dict(lane_id="fam_stripe_alumni")
        plan = ExecutionPlan(
            strategy_rationale="test",
            sourcing_lanes=[lane_dict],
        )
        apply_linkedin_lane_compiler_to_plan(plan)
        p.state.planner._execution_plan = plan

        records = _lane_projection.structured_lane_projection_records(p.state.planner._execution_plan)

        assert len(records) == 1
        record = records[0]
        assert record["lane_id"] == "fam_stripe_alumni"
        assert record["lane_name"] == "Stripe alumni"
        assert record["lane_intent"] == "find stripe alumni"
        assert "fam_stripe_alumni" in record["aliases"]
        assert "stripe_alumni" in record["aliases"]
        assert record["snapshot"]["acquisition_mode"] == "linkedin_hybrid"
        assert not _lane_projection.lane_snapshot_filters(record["snapshot"]).is_empty()


def test_structured_lane_projection_records_skips_boolean_only_lane():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        lane = SourcingLane(
            lane_id="keyword_only",
            lane_name="Keyword only",
            hypothesis=SearchHypothesis(
                hypothesis_id="h1",
                label="Keyword",
                target_archetype="generic",
                why_this_pool_may_exist="broad",
            ),
            slice=SearchSlice(
                slice_id="s1",
                hypothesis_id="h1",
                label="Slice",
                objective="broad keyword",
                constraints=[],
            ),
            execution=LaneExecution(
                lane_id="keyword_only",
                source="linkedin",
                acquisition_mode="linkedin_boolean",
                boolean_strategy={"root_boolean": '"ml" AND research'},
                structured_filters={},
            ),
        )
        lane_dict = lane.to_dict()
        plan = ExecutionPlan(
            strategy_rationale="test",
            sourcing_lanes=[lane_dict],
        )
        apply_linkedin_lane_compiler_to_plan(plan)
        p.state.planner._execution_plan = plan

        assert _lane_projection.structured_lane_projection_records(p.state.planner._execution_plan) == []


def test_structured_lane_projection_records_empty_without_execution_plan():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.state.planner._execution_plan = None
        assert _lane_projection.structured_lane_projection_records(p.state.planner._execution_plan) == []
