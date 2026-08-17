"""Plan-finalization employer blacklist scrub coverage.

Standalone from tests/test_linkedin_strategy.py because that module is
fixture-gated locally; these assertions must always run.
"""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

from linkedin.strategy import (
    _finalize_execution_plan,
    _scrub_blacklisted_employers,
    adapt_after_block,
)
from shared.brief_loader import Brief
from shared.schemas import BlockReport, ExecutionPlan, SearchString


def _brief(
    *,
    employer_blacklist: list[str] | None = None,
    retrieval_design: dict | None = None,
) -> Brief:
    return Brief(
        id="plan-blacklist-test",
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
        employer_blacklist=list(employer_blacklist or []),
        retrieval_design=dict(retrieval_design or {}),
    )


def _block_report() -> BlockReport:
    return BlockReport(
        block_name="Opening",
        strings_run=1,
        strings_with_saves=0,
        total_results=20,
        total_saves=0,
    )


def _retrieval_family() -> dict:
    return {
        "family_id": "ai_data_vendors",
        "label": "AI data vendors",
        "objective": "Find leaders from AI data and labeling vendors.",
        "priority": 90,
        "enabled": True,
        "variants_to_emit": 1,
        "target_employers": [
            "Scale AI",
            " Acme ",
            "Surge AI",
            "Alan Turing Institute",
        ],
        "entry_signals": [
            {
                "item_id": "entry_1",
                "label": "AI data operations",
                "terms": ["AI data operations", "RLHF operations"],
            }
        ],
        "capability_proxies": [
            {
                "item_id": "cap_1",
                "label": "Labeling workflows",
                "terms": ["labeling workflow", "data quality"],
            }
        ],
        "reality_filters": [
            {
                "item_id": "real_1",
                "label": "Production",
                "terms": ["production", "scaled"],
            }
        ],
        "context_constraints": [
            {
                "item_id": "ctx_companies",
                "label": "AI data vendors",
                "terms": ["Scale AI", "ACME", "Surge AI", "Labelbox"],
                "structured_surface": "linkedin_company_filter",
            }
        ],
        "anti_noise": [
            {
                "item_id": "anti_1",
                "label": "Hiring company exclusion",
                "terms": ["Acme"],
                "structured_surface": "linkedin_company_filter",
            }
        ],
    }


def _generated_string() -> dict:
    return {
        "boolean": '("AI data operations" OR "RLHF operations")',
        "rationale": "Bound canonical AI-data vendor pool by company facet.",
        "family_key": "ai_data_vendors",
        "novelty_bucket": "canonical",
        "domain_lane": "ai_data_vendors",
        "structured_filters": {
            "companies": [
                "Scale AI",
                "Acme",
                "Surge AI",
                "Alan Turing Institute",
            ]
        },
    }


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        strategy_rationale="test",
        retrieval_families=[_retrieval_family()],
        generated_strings=[_generated_string()],
    )


def _normalized(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _blacklist_warnings(plan: ExecutionPlan) -> list[dict]:
    return [
        warning
        for warning in plan.plan_warnings
        if warning.get("code") == "blacklisted_employer_scrubbed"
    ]


def _assert_no_turing_outside_allowed_paths(value: object, path: tuple[object, ...] = ()) -> None:
    if path and path[0] == "plan_warnings":
        return
    if "anti_noise" in path:
        return
    if "hidden_pool_risks" in path:
        return
    if isinstance(value, dict) and value.get("dimension") == "anti_noise":
        return
    if isinstance(value, str):
        assert _normalized(value) != "acme", f"unexpected Acme at {path}"
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_no_turing_outside_allowed_paths(child, (*path, key))
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_turing_outside_allowed_paths(child, (*path, index))


def test_live_shaped_plan_scrubs_blacklisted_employer_from_all_source_locations(capsys):
    plan = _plan()

    _scrub_blacklisted_employers(_brief(employer_blacklist=["Acme"]), plan)

    assert plan.retrieval_families[0]["target_employers"] == [
        "Scale AI",
        "Surge AI",
        "Alan Turing Institute",
    ]
    assert plan.retrieval_families[0]["context_constraints"][0]["terms"] == [
        "Scale AI",
        "Surge AI",
        "Labelbox",
    ]
    assert plan.retrieval_families[0]["anti_noise"][0]["terms"] == ["Acme"]
    assert plan.generated_strings[0]["structured_filters"]["companies"] == [
        "Scale AI",
        "Surge AI",
        "Alan Turing Institute",
    ]
    warnings = _blacklist_warnings(plan)
    assert len(warnings) == 3
    warning_messages = [warning["message"] for warning in warnings]
    assert any("retrieval_families[0].target_employers" in msg for msg in warning_messages)
    assert any(
        "retrieval_families[0].context_constraints[0].terms" in msg
        for msg in warning_messages
    )
    assert any(
        "generated_strings[0].structured_filters.companies" in msg
        for msg in warning_messages
    )
    assert "Scale AI" in plan.retrieval_families[0]["context_constraints"][0]["terms"]
    assert "Surge AI" in plan.retrieval_families[0]["context_constraints"][0]["terms"]
    assert "Alan Turing Institute" in plan.retrieval_families[0]["target_employers"]
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_empty_blacklist_is_strict_noop(capsys):
    plan = _plan()
    before = deepcopy(plan.to_dict())

    _scrub_blacklisted_employers(_brief(employer_blacklist=[]), plan)

    assert plan.to_dict() == before
    assert plan.plan_warnings == []
    assert capsys.readouterr().out == ""


def test_finalize_scrubs_before_lane_projection_recursive_sweep(capsys):
    plan = _plan()

    finalized = _finalize_execution_plan(
        _brief(employer_blacklist=["Acme"]),
        plan,
        prior_run_data=None,
    )

    assert finalized.search_hypotheses
    assert finalized.search_slices
    assert finalized.sourcing_lanes
    assert _blacklist_warnings(finalized)
    _assert_no_turing_outside_allowed_paths(finalized.to_dict())
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_finalize_scrubs_brief_shared_layer_company_filter_after_materialize(capsys):
    retrieval_design = {
        "families": [],
        "shared_layers": {
            "context_constraints": [
                {
                    "item_id": "shared_companies",
                    "label": "Legit AI data vendors",
                    "terms": ["Scale AI", "Acme", "Surge AI", "Labelbox"],
                    "structured_surface": "linkedin_company_filter",
                }
            ],
        },
        "edge_case_hypotheses": [],
    }
    plan = ExecutionPlan(
        strategy_rationale="test",
        retrieval_families=[
            {
                "family_id": "ai_data_vendors",
                "label": "AI data vendors",
                "objective": "Find leaders from AI data and labeling vendors.",
                "priority": 90,
                "enabled": True,
                "variants_to_emit": 1,
                "target_employers": ["Scale AI", "Surge AI"],
                "entry_signals": [
                    {
                        "item_id": "entry_1",
                        "label": "AI data operations",
                        "terms": ["AI data operations"],
                    }
                ],
                "capability_proxies": [
                    {
                        "item_id": "cap_1",
                        "label": "Labeling workflows",
                        "terms": ["labeling workflow"],
                    }
                ],
            }
        ],
        generated_strings=[],
    )

    finalized = _finalize_execution_plan(
        _brief(employer_blacklist=["Acme"], retrieval_design=retrieval_design),
        plan,
        prior_run_data=None,
    )

    assert finalized.generated_strings
    companies = finalized.generated_strings[0]["structured_filters"]["companies"]
    assert companies == ["Scale AI", "Surge AI", "Labelbox"]
    assert finalized.retrieval_families[0]["target_employers"] == [
        "Scale AI",
        "Surge AI",
    ]
    assert finalized.sourcing_lanes
    assert _blacklist_warnings(finalized)
    _assert_no_turing_outside_allowed_paths(finalized.to_dict())
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_adaptation_path_hook_scrubs_new_families_and_new_strings(capsys):
    execution_plan = ExecutionPlan(strategy_rationale="current plan")

    def fake_opus(
        system_prompt: str,
        user_prompt: str,
        expect_json: bool = True,
        max_tokens: int = 8192,
        usage_context: dict | None = None,
        model_name: str | None = None,
    ):
        return {
            "new_retrieval_families": [_retrieval_family()],
            "new_strings": [_generated_string()],
            "skip_remaining": [],
            "reorder": [],
            "noise_updates": [],
        }

    with patch("linkedin.strategy.opus_llm", side_effect=fake_opus):
        adaptation = adapt_after_block(
            _brief(employer_blacklist=["Acme"]),
            _block_report(),
            [SearchString(id=20, name="Queued", boolean='("queued")')],
            execution_plan=execution_plan,
        )

    assert adaptation.new_retrieval_families[0]["target_employers"] == [
        "Scale AI",
        "Surge AI",
        "Alan Turing Institute",
    ]
    assert adaptation.new_retrieval_families[0]["context_constraints"][0]["terms"] == [
        "Scale AI",
        "Surge AI",
        "Labelbox",
    ]
    assert adaptation.new_strings[0]["structured_filters"]["companies"] == [
        "Scale AI",
        "Surge AI",
        "Alan Turing Institute",
    ]
    assert len(_blacklist_warnings(execution_plan)) == 3
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_scrub_console_contract_is_one_compact_line(capsys):
    plan = _plan()

    _scrub_blacklisted_employers(_brief(employer_blacklist=["Acme"]), plan)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert "[blacklist-scrub]" in lines[0]
    assert "Acme" in lines[0]
    assert "3 plan location(s)" in lines[0]
