"""Every string in an allowlisted campaign runs inside a competitor company facet.

The strategist is told to facet every string, but three paths ran without
one (review of 2026-10-08): adaptation strings come from a prompt that never
saw the instruction, a company typeahead miss runs the string keyword-only,
and a variant or rescue may DEMOTE the companies dimension. Under a brief's
``employer_allowlist`` (inert without one):

- a deterministic stamp puts a listed facet group on every string that lacks
  one, maps aliases to their parent, drops unlisted companies, and caps the
  facet at the per-facet value limit;
- a string whose competitor companies all fail to land is skipped, never run
  keyword-only across the country;
- a variant or rescue may not demote, or empty, the companies dimension, and
  may not promote an unlisted company into it.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.employer_allowlist import parse_employer_allowlist
from shared.schemas import SearchString

ALLOWLIST_FIELD = {
    "facet_groups": [
        ["Mercor"],
        ["Handshake"],
        ["Surge AI", "AfterQuery"],
        ["Scale AI"],
        ["Snorkel AI", "Labelbox", "Toloka", "Encord"],
        ["Appen", "TELUS Digital", "Uber AI Solutions", "Innodata"],
    ],
    "aliases": {"Outlier": "Scale AI", "Handshake AI": "Handshake", "Alignerr": "Labelbox"},
}
ALLOWLIST = parse_employer_allowlist(ALLOWLIST_FIELD)


def _string(sid: int, boolean: str = '("program manager" OR "engagement manager")', **filters) -> SearchString:
    return SearchString(id=sid, name=f"s{sid}", boolean=boolean, structured_filters=dict(filters))


# ---------------------------------------------------------------------------
# The stamp
# ---------------------------------------------------------------------------


def test_an_unfaceted_string_is_stamped_with_a_listed_group() -> None:
    from linkedin.competitor_scope import stamp_competitor_facet

    ss = _string(1)
    record = stamp_competitor_facet(ss, ALLOWLIST, queue=[ss])

    assert record is not None and record["action"] == "stamped"
    assert ss.structured_filters["companies"] == ["Mercor"]
    assert ss.acquisition_mode == "linkedin_hybrid"
    assert "Competitor facet stamped: Mercor" in ss.notes


def test_the_stamp_follows_a_competitor_the_boolean_already_names() -> None:
    from linkedin.competitor_scope import stamp_competitor_facet

    ss = _string(1, '("Scale AI" OR "Outlier") AND ("queue manager")')
    stamp_competitor_facet(ss, ALLOWLIST, queue=[ss])

    assert ss.structured_filters["companies"] == ["Scale AI"]


def test_stamps_spread_across_the_groups_least_used_first() -> None:
    from linkedin.competitor_scope import stamp_competitor_facets

    queue = [_string(i) for i in range(1, 8)]
    stamp_competitor_facets(queue, ALLOWLIST, queue=queue)

    groups = [tuple(ss.structured_filters["companies"]) for ss in queue]
    assert groups[:6] == [tuple(group) for group in ALLOWLIST.facet_groups]
    assert groups[6] == ("Mercor",)


def test_a_facet_of_unlisted_companies_is_replaced() -> None:
    from linkedin.competitor_scope import stamp_competitor_facet

    ss = _string(1, companies=["xAI", "GitHub"])
    record = stamp_competitor_facet(ss, ALLOWLIST, queue=[ss])

    assert record["action"] == "stamped"
    assert record["before"] == ["xAI", "GitHub"]
    assert ss.structured_filters["companies"] == ["Mercor"]


def test_unlisted_companies_leave_a_mixed_facet_and_aliases_map_to_their_parent() -> None:
    from linkedin.competitor_scope import stamp_competitor_facet

    ss = _string(1, companies=["Handshake AI", "xAI", "Outlier", "Scale AI"])
    record = stamp_competitor_facet(ss, ALLOWLIST, queue=[ss])

    assert record["action"] == "scoped"
    assert ss.structured_filters["companies"] == ["Handshake", "Scale AI"]


def test_a_facet_over_the_value_cap_is_trimmed_to_it() -> None:
    from linkedin.competitor_scope import stamp_competitor_facet
    from shared.employer_allowlist import FACET_COMPANY_VALUE_CAP

    big = parse_employer_allowlist(
        {"facet_groups": [[f"Vendor {i}" for i in range(12)], [f"Vendor {i}" for i in range(12, 16)]]}
    )
    ss = _string(1, companies=[f"Vendor {i}" for i in range(16)])
    stamp_competitor_facet(ss, big, queue=[ss])

    assert len(ss.structured_filters["companies"]) == FACET_COMPANY_VALUE_CAP == 12


def test_a_string_already_inside_the_scope_is_untouched() -> None:
    from linkedin.competitor_scope import stamp_competitor_facet

    ss = _string(1, companies=["Mercor"], titles=["Engagement Manager"])
    before = ss.to_dict()

    assert stamp_competitor_facet(ss, ALLOWLIST, queue=[ss]) is None
    assert ss.to_dict() == before


def test_without_an_allowlist_the_stamp_is_inert() -> None:
    from linkedin.competitor_scope import stamp_competitor_facets

    ss = _string(1)
    before = ss.to_dict()

    assert stamp_competitor_facets([ss], None, queue=[ss]) == []
    assert ss.to_dict() == before


def test_the_initial_queue_build_stamps_every_string() -> None:
    from linkedin.queue_planner import QueuePlanner

    planner = object.__new__(QueuePlanner)
    planner.context = SimpleNamespace(
        brief_obj=SimpleNamespace(raw={"employer_allowlist": ALLOWLIST_FIELD}), log_path=None
    )
    ordered = [_string(1), _string(2, companies=["xAI"])]

    planner._enforce_competitor_facets(ordered, queue=ordered)

    assert ordered[0].structured_filters["companies"] == ["Mercor"]
    assert ordered[1].structured_filters["companies"] == ["Handshake"]


# ---------------------------------------------------------------------------
# A string whose competitor facet never landed does not run
# ---------------------------------------------------------------------------


def _receipt(desired, after, unapplied=(), substituted=None):
    return {
        "desired": {"companies": list(desired), "titles": []},
        "after": {"companies": list(after), "titles": []},
        "unapplied": {"companies": list(unapplied), "titles": []},
        "substituted": {"companies": dict(substituted or {}), "titles": {}},
        "failure_gate": "",
    }


def test_a_string_whose_competitors_all_missed_the_typeahead_is_skipped() -> None:
    from linkedin.competitor_scope import competitor_facet_skip_reason

    reason = competitor_facet_skip_reason(
        ["Mercor", "Handshake"], _receipt(["Mercor", "Handshake"], [], ["Mercor", "Handshake"]), ALLOWLIST
    )

    assert reason and "Mercor" in reason


def test_a_near_miss_onto_an_unlisted_company_counts_as_not_landed() -> None:
    from linkedin.competitor_scope import competitor_facet_skip_reason

    reason = competitor_facet_skip_reason(
        ["Uber AI Solutions"],
        _receipt(["Uber AI Solutions"], ["Uber"], substituted={"Uber AI Solutions": "Uber"}),
        ALLOWLIST,
    )

    assert reason


def test_a_string_with_no_company_facet_is_skipped() -> None:
    from linkedin.competitor_scope import competitor_facet_skip_reason

    assert competitor_facet_skip_reason([], _receipt([], []), ALLOWLIST)


def test_one_landed_competitor_is_enough() -> None:
    from linkedin.competitor_scope import competitor_facet_skip_reason

    assert competitor_facet_skip_reason(
        ["Mercor", "Handshake"], _receipt(["Mercor", "Handshake"], ["Mercor"], ["Handshake"]), ALLOWLIST
    ) is None


def test_the_skip_is_inert_without_an_allowlist_or_a_receipt() -> None:
    from linkedin.competitor_scope import competitor_facet_skip_reason

    assert competitor_facet_skip_reason([], _receipt([], []), None) is None
    assert competitor_facet_skip_reason(["Mercor"], None, ALLOWLIST) is None


def test_the_string_driver_skips_a_string_whose_facet_did_not_land(capsys) -> None:
    from linkedin.string_driver import StringDriver

    driver = object.__new__(StringDriver)
    driver.context = SimpleNamespace(
        brief_obj=SimpleNamespace(raw={"employer_allowlist": ALLOWLIST_FIELD}), log_path=None
    )
    driver.surface = SimpleNamespace(last_opening_facet_receipt=_receipt(["Mercor"], [], ["Mercor"]))
    state = SimpleNamespace(active_variant=SimpleNamespace(structured_filters=SimpleNamespace(companies=["Mercor"])))
    ss = _string(3, companies=["Mercor"])

    reason = driver._competitor_facet_skip(ss, state)

    assert reason
    assert "Skipped: no listed competitor landed on the company facet" in ss.notes
    assert "[scope] String #3 skipped" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# No variant or rescue demotes the companies dimension
# ---------------------------------------------------------------------------


def _resolve(raw_variant, inherited, **kwargs):
    from linkedin import search_policy

    return search_policy._resolve_structured_controls(raw_variant, inherited, **kwargs)


def test_a_companies_demote_is_refused_under_the_allowlist() -> None:
    from linkedin.search_intelligence import LinkedInStructuredFilters

    inherited = LinkedInStructuredFilters(companies=["Mercor"], titles=["TPM"])
    refused: list[str] = []
    filters, surface, kind, changed = _resolve(
        {"surface": "boolean", "structured_controls": {"demote": ["companies", "titles"]}},
        inherited,
        company_scope=ALLOWLIST,
        refused=refused,
    )

    assert filters.companies == ["Mercor"]
    assert filters.titles == []
    assert refused == ["demote:companies"]
    assert surface != "boolean" or not filters.is_empty()


def test_an_unlisted_company_promotion_is_refused_under_the_allowlist() -> None:
    from linkedin.search_intelligence import LinkedInStructuredFilters

    refused: list[str] = []
    filters, *_ = _resolve(
        {"surface": "hybrid", "structured_controls": {"companies": ["xAI", "Handshake"]}},
        LinkedInStructuredFilters(companies=["Mercor"]),
        company_scope=ALLOWLIST,
        refused=refused,
    )

    assert filters.companies == ["Mercor", "Handshake"]
    assert refused == ["promote:xAI"]


def test_without_the_allowlist_a_companies_demote_still_clears() -> None:
    from linkedin.search_intelligence import LinkedInStructuredFilters

    filters, *_ = _resolve(
        {"surface": "boolean", "structured_controls": {"demote": ["companies"]}},
        LinkedInStructuredFilters(companies=["Mercor"]),
    )

    assert filters.companies == []


def _browser(apply_result):
    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    browser.apply_advanced_search_plan = AsyncMock(return_value=apply_result)
    browser.enter_search_string = AsyncMock()
    browser.get_results_count_text = AsyncMock(return_value="80")
    browser.get_results_count = AsyncMock(return_value=80)
    browser.get_card_snapshot = AsyncMock(return_value={})
    return browser


def _executor(browser, scope):
    from linkedin.search_mutation import LinkedInSearchMutationDeps, LinkedInSearchMutationExecutor

    return LinkedInSearchMutationExecutor(
        LinkedInSearchMutationDeps(
            browser=browser,
            log_path=Path("test.log"),
            get_input_mode=lambda: "concurrent",
            get_runtime_run_id=lambda: None,
            get_runtime_state=lambda: MagicMock(),
            get_search_mutation_budget_used=lambda: 0,
            set_search_mutation_budget_used=lambda _v: None,
            get_company_scope=lambda: scope,
        )
    )


def _state_and_variant(companies):
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
    )

    state = LinkedInExperimentState(
        root_string_id=1, intent=LinkedInSearchIntent(root_boolean="x"), mode="experiment"
    )
    variant = LinkedInSearchVariant(
        variant_id="v1",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"program manager"',
        surface="hybrid" if companies else "boolean",
        structured_filters=LinkedInStructuredFilters(companies=list(companies)),
    )
    state.variants[variant.variant_id] = variant
    return state, variant


def test_a_rescue_variant_that_lost_its_competitor_facet_gets_it_back() -> None:
    from linkedin.advanced_search import ControlApplicationResult

    browser = _browser(ControlApplicationResult(success=True, applied_controls=["keywords", "companies"]))
    state, variant = _state_and_variant([])
    ss = _string(1, companies=["Mercor"])
    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0), patch(
        "linkedin.search_mutation.log_event"
    ):
        result = asyncio.run(
            _executor(browser, ALLOWLIST).apply_variant(
                search_string=ss, experiment_state=state, variant=variant, acquisition_mode="linkedin_boolean"
            )
        )

    assert result.applied is True
    assert variant.structured_filters.companies == ["Mercor"]
    plan = browser.apply_advanced_search_plan.await_args.args[0]
    assert any(control.dimension == "companies" for control in plan.controls)


def test_demote_and_proceed_never_drops_the_companies_dimension() -> None:
    from linkedin.advanced_search import ControlApplicationResult

    browser = _browser(
        ControlApplicationResult(
            success=True,
            applied_controls=["keywords"],
            failed_controls=["companies"],
            reason="structured_controls_dropped_keyword_only",
            plan_fully_applied=False,
        )
    )
    state, variant = _state_and_variant(["Mercor"])
    ss = _string(1, companies=["Mercor"])
    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0), patch(
        "linkedin.search_mutation.log_event"
    ):
        result = asyncio.run(
            _executor(browser, ALLOWLIST).apply_variant(
                search_string=ss, experiment_state=state, variant=variant, acquisition_mode="linkedin_hybrid"
            )
        )

    assert result.applied is False
    assert result.blocked_reason == "competitor_facet_not_applied"
    assert variant.structured_filters.companies == ["Mercor"]
    assert ss.structured_filters["companies"] == ["Mercor"]
    browser.enter_search_string.assert_not_awaited()


def test_without_the_allowlist_demote_and_proceed_is_unchanged() -> None:
    from linkedin.advanced_search import ControlApplicationResult

    browser = _browser(
        ControlApplicationResult(
            success=True,
            applied_controls=["keywords"],
            failed_controls=["companies"],
            reason="structured_controls_dropped_keyword_only",
            plan_fully_applied=False,
        )
    )
    state, variant = _state_and_variant(["Mercor"])
    ss = _string(1, companies=["Mercor"])
    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0), patch(
        "linkedin.search_mutation.log_event"
    ):
        result = asyncio.run(
            _executor(browser, None).apply_variant(
                search_string=ss, experiment_state=state, variant=variant, acquisition_mode="linkedin_hybrid"
            )
        )

    assert result.applied is True
    assert variant.structured_filters.companies == []


# ---------------------------------------------------------------------------
# The adaptation and variant prompts see the scope
# ---------------------------------------------------------------------------


def test_the_scope_prompt_block_names_the_groups_and_forbids_a_demote() -> None:
    from linkedin.competitor_scope import competitor_scope_prompt_block

    block = competitor_scope_prompt_block(SimpleNamespace(raw={"employer_allowlist": ALLOWLIST_FIELD}))

    assert "COMPETITOR SCOPE" in block
    assert "Surge AI, AfterQuery" in block
    assert "never demote" in block.lower()
    assert competitor_scope_prompt_block(SimpleNamespace(raw={})) == ""


def test_the_adaptation_prompt_carries_the_scope_block_only_under_an_allowlist() -> None:
    from linkedin.strategy import adapt_after_block
    from tests.test_adaptation_signal_state import _block_report, _brief

    for raw, expected in (({"employer_allowlist": ALLOWLIST_FIELD}, True), ({}, False)):
        captured: dict = {}

        def fake_opus(system_prompt, *args, **kwargs):
            captured["system"] = system_prompt
            return {"no_change": True}

        brief = _brief()
        brief.raw = raw
        with patch("linkedin.strategy.opus_llm", side_effect=fake_opus):
            adapt_after_block(brief, _block_report(), [_string(20, companies=["Mercor"])])

        assert ("COMPETITOR SCOPE (deterministic" in captured["system"]) is expected


def test_process_string_skips_a_string_whose_competitor_facet_did_not_land(tmp_path) -> None:
    from shared.schemas import Progress
    from tests.test_linkedin_zero_results_signal import _make_pipeline

    pipeline = _make_pipeline(tmp_path)
    pipeline.state.context.brief_obj.raw = {"employer_allowlist": ALLOWLIST_FIELD}
    search_string = _string(7, companies=["Mercor"])
    driver = pipeline.state.driver
    driver._prepare_active_allocator_dispatch = MagicMock(return_value=None)
    pipeline.state._allocator_service._check_allocator_pre_spend = MagicMock()
    pipeline.state.surface._ensure_browser_healthy = AsyncMock()
    pipeline.state._allocator_service._allocator_active_enabled = MagicMock(return_value=False)
    driver._checkpoint_allocator_exhaustion_transition = MagicMock(return_value=None)
    experiment_state = MagicMock()
    experiment_state.current_boolean.return_value = search_string.boolean
    experiment_state.active_allocator_page_cursor.return_value = 0
    experiment_state.active_variant.structured_filters.companies = ["Mercor"]
    driver._experiment_state_for = MagicMock(return_value=experiment_state)
    driver._validated_owner_pending_full_snippets = MagicMock(return_value=[])
    driver._resolve_ahead_of_cursor_pending = MagicMock(return_value=[])

    async def opening(*args, **kwargs):
        pipeline.state.surface.last_opening_facet_receipt = _receipt(["Mercor"], [], ["Mercor"])

    pipeline.state.surface._apply_opening_search = AsyncMock(side_effect=opening)
    pipeline.state.surface._record_last_good_url = MagicMock()
    pipeline.state.context.browser.page = MagicMock()
    pipeline.state.context.browser.page.url = "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    pipeline.state.context.browser.get_results_count_text = AsyncMock(return_value="80")
    pipeline.state.context.browser.get_results_count = AsyncMock(return_value=80)

    with patch("linkedin.string_driver.log_event"):
        asyncio.run(driver._process_string_impl(search_string, Progress(brief_name="test")))

    driver._checkpoint_allocator_exhaustion_transition.assert_called_once()
    assert driver._checkpoint_allocator_exhaustion_transition.call_args.kwargs["legacy_terminal_status"] == "skipped"
    assert "Skipped: no listed competitor landed on the company facet" in search_string.notes


def test_the_real_queue_build_stamps_and_keeps_a_facet_only_string() -> None:
    """Instruction 8's facet-only strings: a generated string carrying
    ``"surface": "structured_only"`` queues with that surface, so the opening
    enters the company facet alone and never types the keyword Boolean."""
    import tempfile

    from linkedin.advanced_search import compile_structured_filters_to_plan
    from linkedin.search_intelligence import bootstrap_experiment_state
    from shared.schemas import ExecutionPlan
    from tests.test_seam_strategy_execution import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.state.context.brief_obj.raw = {"employer_allowlist": ALLOWLIST_FIELD}
        p.state.planner._execution_plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {
                    "boolean": '("program manager" OR "engagement manager")',
                    "rationale": "AfterQuery facet only",
                    "family_key": "core_afterquery",
                    "surface": "structured_only",
                    "structured_filters": {"companies": ["AfterQuery"]},
                },
                {
                    "boolean": '("human data" OR "data operations")',
                    "rationale": "unfaceted adaptation-shaped string",
                    "family_key": "seat",
                },
            ],
            coverage_gaps=[],
        )

        facet_only, stamped = p.state.planner._build_ordered_search_strings()

    assert facet_only.surface == "structured_only"
    assert facet_only.acquisition_mode == "linkedin_hybrid"
    assert facet_only.structured_filters["companies"] == ["AfterQuery"]
    root = bootstrap_experiment_state(facet_only).root_variant
    plan = compile_structured_filters_to_plan(
        root.structured_filters,
        keyword_boolean=root.boolean,
        acquisition_mode=facet_only.acquisition_mode,
        include_keyword=(root.surface != "structured_only"),
    )
    assert [control.dimension for control in plan.controls] == ["companies"]
    assert stamped.structured_filters["companies"] == ["Mercor"]
    assert stamped.acquisition_mode == "linkedin_hybrid"


def test_an_adaptation_string_without_a_facet_is_stamped_when_it_is_queued(tmp_path) -> None:
    from linkedin.adaptation_signal_state import AdaptationGateConfig
    from shared.schemas import AdaptationResponse, Progress
    from tests.test_linkedin_pipeline import _make_pipeline

    pipeline = _make_pipeline(str(tmp_path))
    pipeline.state.context.brief_obj.raw = {"employer_allowlist": ALLOWLIST_FIELD}
    pipeline.state.planner._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)
    done = SearchString(id=1, name="done", boolean='"x"', status="done", block="Block A",
                        structured_filters={"companies": ["Mercor"]})
    progress = Progress(brief_name="test", strings=[done, SearchString(id=2, name="q", boolean='"q"')])
    response = AdaptationResponse(new_strings=[
        {"boolean": '("human data" OR "queue manager")', "rationale": "adapted", "family_key": "seat"},
    ])

    asyncio.run(
        pipeline.state.planner._run_block_adaptation(
            "Block A", [done], progress, lambda *args, **kwargs: response
        )
    )

    adaptive = [s for s in progress.strings if s.string_type == "Adaptive"]
    assert len(adaptive) == 1
    assert adaptive[0].structured_filters["companies"] == ["Handshake"]
    assert adaptive[0].acquisition_mode == "linkedin_hybrid"


def test_the_driver_stamps_a_string_no_queue_path_faceted_before_its_first_page(tmp_path) -> None:
    from shared.schemas import Progress
    from tests.test_linkedin_zero_results_signal import _make_pipeline

    pipeline = _make_pipeline(tmp_path)
    pipeline.state.context.brief_obj.raw = {"employer_allowlist": ALLOWLIST_FIELD}
    search_string = _string(9)
    driver = pipeline.state.driver
    driver._prepare_active_allocator_dispatch = MagicMock(return_value=None)
    pipeline.state._allocator_service._check_allocator_pre_spend = MagicMock()
    pipeline.state.surface._ensure_browser_healthy = AsyncMock()
    pipeline.state._allocator_service._allocator_active_enabled = MagicMock(return_value=False)
    driver._checkpoint_allocator_exhaustion_transition = MagicMock(return_value=None)
    pipeline.state.surface._apply_opening_search = AsyncMock()
    pipeline.state.surface._record_last_good_url = MagicMock()
    pipeline.state.context.browser.page = MagicMock()
    pipeline.state.context.browser.page.url = "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    pipeline.state.context.browser.get_results_count_text = AsyncMock(return_value="0")
    pipeline.state.context.browser.get_results_count = AsyncMock(return_value=0)
    driver._experiment_state_for = MagicMock(side_effect=lambda ss: MagicMock(
        current_boolean=MagicMock(return_value=ss.boolean),
        active_allocator_page_cursor=MagicMock(return_value=0),
    ))

    with patch("linkedin.string_driver.log_event"):
        asyncio.run(driver._process_string_impl(search_string, Progress(brief_name="test", strings=[search_string])))

    assert search_string.structured_filters["companies"] == ["Mercor"]


def test_experiment_bootstrap_freezes_a_stamped_facet_into_the_root_variant() -> None:
    from linkedin.string_driver import StringDriver

    driver = object.__new__(StringDriver)
    driver.context = SimpleNamespace(brief_obj=SimpleNamespace(raw={"employer_allowlist": ALLOWLIST_FIELD}), log_path=None)
    registered = []
    driver.allocator = SimpleNamespace(experiment=lambda _id: None, register_experiment=registered.append)
    ss = _string(4)

    state = driver._experiment_state_for(ss)

    assert state.root_variant.structured_filters.companies == ["Mercor"]


def _seam_pipeline(td):
    from tests.test_seam_strategy_execution import _make_pipeline

    p = _make_pipeline(td)
    p.state.context.brief_obj.raw = {"employer_allowlist": ALLOWLIST_FIELD}
    p.state.context.brief_obj.role_title = "SPL"
    p.state.context.brief_obj.role_description = "desc"
    return p


def test_the_variant_planner_keeps_the_competitor_facet_and_sees_the_scope() -> None:
    import tempfile

    from tests.test_seam_strategy_execution import (
        _experiment_state_with_active,
        _page_insights_with_company_cluster,
    )

    with tempfile.TemporaryDirectory() as td:
        p = _seam_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=7, root_boolean='"pm"', active_boolean='"pm"',
            active_filters={"companies": ["Mercor"]},
        )
        proposal = {"variants": [{
            "variant_id": "demote-1", "variant_kind": "structured_filter", "hypothesis": "drop the facet",
            "boolean": '"pm" OR "tpm"', "surface": "boolean",
            "structured_controls": {"demote": ["companies"]},
            "target_result_min": 75, "target_result_max": 400,
        }]}
        planner = MagicMock(return_value=proposal)
        with patch("shared.llm_clients.opus_llm_cached", planner), patch("linkedin.string_driver.log_event"):
            variants = asyncio.run(p.state.driver._plan_variant_experiments(
                search_string=MagicMock(id=7), experiment_state=state, current_boolean='"pm"',
                result_count=600, result_count_text="600 results",
                page_insights=_page_insights_with_company_cluster("Mercor"),
                all_candidates=[], string_stats={},
            ))

    assert "COMPETITOR SCOPE (deterministic" in planner.call_args.args[0]
    assert variants and variants[0].structured_filters.companies == ["Mercor"]


def test_the_drift_planner_keeps_the_competitor_facet_and_sees_the_scope() -> None:
    import tempfile

    from tests.test_seam_strategy_execution import (
        _experiment_state_with_active,
        _page_insights_with_title_cluster,
    )

    with tempfile.TemporaryDirectory() as td:
        p = _seam_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=15, root_boolean='"pm"', active_boolean='"pm"',
            active_filters={"companies": ["Mercor"]},
        )
        state.last_drift_refinement_summary = {"decision": "refine_committed"}
        proposal = {
            "boolean": '"pm" AND coding', "variant_kind": "structured_filter", "surface": "boolean",
            "hypothesis": "demote the facet", "structured_controls": {"demote": ["companies"]},
            "target_result_min": 75, "target_result_max": 400,
        }
        planner = MagicMock(return_value=proposal)
        with patch("shared.llm_clients.opus_llm_cached", planner), patch("linkedin.string_driver.log_event"):
            variant, _summary = asyncio.run(p.state.driver._plan_drift_refinement(
                search_string=MagicMock(id=15), experiment_state=state, current_boolean='"pm"',
                result_count=600, result_count_text="600 results",
                page_insights=_page_insights_with_title_cluster("TPM"), page_stats={},
            ))

    assert "COMPETITOR SCOPE (deterministic" in planner.call_args.args[0]
    assert variant is not None and variant.structured_filters.companies == ["Mercor"]


def test_a_near_miss_onto_a_namesake_company_counts_as_not_landed() -> None:
    from linkedin.competitor_scope import competitor_facet_skip_reason

    reason = competitor_facet_skip_reason(
        ["Handshake"],
        _receipt(["Handshake"], ["Handshake Partners"], substituted={"Handshake": "Handshake Partners"}),
        ALLOWLIST,
    )

    assert reason and "Handshake -> Handshake Partners" in reason


def test_the_opening_keeps_its_verified_facet_receipt_for_the_skip() -> None:
    import tempfile

    from shared import config
    from tests.test_linkedin_pipeline import _facet_containment_pipeline

    with tempfile.TemporaryDirectory() as td:
        p = _facet_containment_pipeline(td, string_count=1)
        browser = p.state.context.browser
        browser.assert_search_surface_ready = AsyncMock()
        browser.enter_search_string = AsyncMock()
        receipt = _receipt(["Mercor"], ["Mercor"])

        async def reconcile(**kwargs):
            browser.last_facet_reconciliation = dict(receipt)

        browser.reconcile_facets = AsyncMock(side_effect=reconcile)
        p.state.surface._verify_session_geography_chips = AsyncMock()
        p.state.surface._record_facet_reconciliation = MagicMock()
        filters = MagicMock(companies=["Mercor"], titles=[], sidebar_filters={})
        filters.is_empty.return_value = False
        experiment_state = MagicMock()
        experiment_state.active_variant.structured_filters = filters
        search_string = MagicMock(id=5, acquisition_mode="linkedin_boolean")

        with patch.object(config, "LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", True), patch(
            "linkedin.search_surface.log_event"
        ):
            asyncio.run(p.state.surface._apply_opening_search(search_string, experiment_state, '"pm"'))

        assert p.state.surface.last_opening_facet_receipt["after"]["companies"] == ["Mercor"]


# ---------------------------------------------------------------------------
# The facet-landed skip on every path that re-applies a string's search, and
# the stamp on every path that bootstraps a string's experiment state
# ---------------------------------------------------------------------------


def _driver_harness(tmp_path, search_string, *, receipts, decision, owned_pending=(), cursors=(0,)):
    """A pipeline whose driver runs ``_process_string_impl`` against mocks.

    ``receipts`` are the opening receipts the successive ``_apply_opening_search``
    calls leave; ``decision`` is every page's local assessment.
    """
    from tests.test_linkedin_zero_results_signal import _make_pipeline

    pipeline = _make_pipeline(tmp_path)
    pipeline.state.context.brief_obj.raw = {"employer_allowlist": ALLOWLIST_FIELD}
    driver = pipeline.state.driver
    allocator = pipeline.state._allocator_service
    surface = pipeline.state.surface
    driver._prepare_active_allocator_dispatch = MagicMock(return_value=None)
    allocator._check_allocator_pre_spend = MagicMock()
    allocator._allocator_active_enabled = MagicMock(return_value=False)
    allocator._checkpoint_progress = MagicMock()
    allocator._arm_incomplete_page_rollback = MagicMock()
    allocator._discard_incomplete_page_rollback = MagicMock()
    allocator._stage_allocator_page_checkpoint = MagicMock()
    allocator.bypasses_local_page_policy = MagicMock(return_value=False)
    surface._ensure_browser_healthy = AsyncMock()
    surface._record_last_good_url = MagicMock()
    surface._go_to_next_page_with_transient_retry = AsyncMock(return_value=(False, False))
    driver._checkpoint_allocator_exhaustion_transition = MagicMock(return_value=None)
    driver._finish_completed_page_allocator_boundary = MagicMock(return_value=None)
    driver._evaluate_variant_lifecycle = MagicMock(return_value=None)
    driver._assess_string_state = AsyncMock(
        return_value={"decision": decision, "rationale": decision, "page": 1}
    )
    driver._validated_owner_pending_full_snippets = MagicMock(return_value=list(owned_pending))
    driver._resolve_ahead_of_cursor_pending = MagicMock(return_value=list(owned_pending))
    pipeline.state.planner._maybe_discover_fallback_candidates = MagicMock()
    pipeline.state.pages._review_page_sequentially = AsyncMock(return_value=None)

    experiment_state = MagicMock()
    experiment_state.current_boolean.return_value = search_string.boolean
    # Successive page cursors: the opening's, then a committed variant's.
    cursor_values = list(cursors)
    experiment_state.active_allocator_page_cursor.side_effect = (
        lambda: cursor_values.pop(0) if len(cursor_values) > 1 else cursor_values[0]
    )
    experiment_state.active_variant.structured_filters.companies = ["Mercor"]
    experiment_state.active_variant.pages_reviewed = 1
    experiment_state.active_variant.variant_id = "root"
    driver._experiment_state_for = MagicMock(return_value=experiment_state)

    queue = list(receipts)

    async def opening(*args, **kwargs):
        surface.last_opening_facet_receipt = queue.pop(0)

    surface._apply_opening_search = AsyncMock(side_effect=opening)
    no_results = MagicMock()
    no_results.is_visible = AsyncMock(return_value=False)
    browser = pipeline.state.context.browser
    browser.page = MagicMock()
    browser.page.url = "https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
    browser.page.locator.return_value.first = no_results
    browser.go_back_to_results = AsyncMock()
    browser.get_results_count_text = AsyncMock(return_value="80")
    browser.get_results_count = AsyncMock(return_value=80)
    return pipeline, driver


def test_a_string_kept_open_for_pending_reviews_stops_after_they_settle(tmp_path) -> None:
    """Resume with canonical pending reviews and a facet that did not land: the
    string keeps its surface for the page that settles them, then stops."""
    from shared.schemas import CandidateSnippet, Progress

    search_string = _string(7, companies=["Mercor"])
    pending = CandidateSnippet(
        name="Pending", headline="", current_title="", current_company="Mercor",
        location="", education_snippet="", profile_url="/talent/profile/pending",
        source_string_id=7, source_string_name="s7", page=1, result_rank=1,
    )
    pipeline, driver = _driver_harness(
        tmp_path,
        search_string,
        receipts=[_receipt(["Mercor"], [], ["Mercor"])],
        decision="continue",
        owned_pending=[("pending-key", pending)],
    )

    with patch("linkedin.string_driver.log_event"):
        asyncio.run(driver._process_string_impl(search_string, Progress(brief_name="test", strings=[search_string])))

    pipeline.state.pages._review_page_sequentially.assert_awaited_once()
    pipeline.state.surface._go_to_next_page_with_transient_retry.assert_not_awaited()
    driver._assess_string_state.assert_not_awaited()
    pipeline.state.planner._maybe_discover_fallback_candidates.assert_not_called()
    assert search_string.status == "skipped"
    assert "Skipped: no listed competitor landed on the company facet" in search_string.notes


def test_re_entering_a_committed_variant_whose_facet_did_not_land_skips_the_string(tmp_path) -> None:
    from shared.schemas import Progress

    search_string = _string(8, companies=["Mercor"])
    pipeline, driver = _driver_harness(
        tmp_path,
        search_string,
        receipts=[_receipt(["Mercor"], ["Mercor"]), _receipt(["Mercor"], [], ["Mercor"])],
        decision="resume_committed",
        cursors=(0, 2),
    )

    with patch("linkedin.string_driver.log_event"):
        asyncio.run(driver._process_string_impl(search_string, Progress(brief_name="test", strings=[search_string])))

    assert pipeline.state.surface._apply_opening_search.await_count == 2
    # No fast-forward through a keyword-only result set.
    pipeline.state.surface._go_to_next_page_with_transient_retry.assert_not_awaited()
    driver._checkpoint_allocator_exhaustion_transition.assert_called_once()
    assert driver._checkpoint_allocator_exhaustion_transition.call_args.kwargs["legacy_terminal_status"] == "skipped"
    # The completed drift page is persisted before the skip.
    completed = [
        c for c in pipeline.state._allocator_service._checkpoint_progress.call_args_list
        if c.kwargs.get("completed_page")
    ]
    assert completed
    assert "Skipped: no listed competitor landed on the company facet" in search_string.notes


def _recovery_surface(receipt):
    from linkedin.search_surface import SearchSurface

    surface = object.__new__(SearchSurface)
    owner = _string(5, companies=["Mercor"])
    state = SimpleNamespace(
        current_boolean=lambda: owner.boolean,
        active_variant=SimpleNamespace(structured_filters=SimpleNamespace(companies=["Mercor"])),
    )
    surface.allocator = SimpleNamespace(require_experiment=lambda _id: state)
    page = MagicMock(url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch")
    surface.context = SimpleNamespace(
        brief_obj=SimpleNamespace(raw={"employer_allowlist": ALLOWLIST_FIELD}, linkedin_project_id="test-project"),
        browser=SimpleNamespace(
            navigate_to_search=AsyncMock(),
            page=page,
            results_page_offset=AsyncMock(return_value=0),
        ),
        log_path=None,
    )
    surface.geography = SimpleNamespace(
        reset_location_applied=MagicMock(), _apply_session_location_filter=AsyncMock()
    )
    surface._get_project_url = MagicMock(return_value="https://www.linkedin.com/talent/hire/test-project")
    surface._check_page_recovery_health = AsyncMock()
    surface._check_page_recovery_authority = AsyncMock()
    surface._probe_card_slot_count_after_hydration = AsyncMock(return_value=25)
    surface._recovery_top_identity = AsyncMock(return_value="top")
    surface._variant_id_for_search_string = MagicMock(return_value="root")

    async def remember(_owner, _page):
        surface.recovery_top_identity = "top"

    surface._remember_recovery_page = AsyncMock(side_effect=remember)

    async def opening(*args, **kwargs):
        surface.last_opening_facet_receipt = receipt

    surface._apply_opening_search = AsyncMock(side_effect=opening)
    return surface, owner


@pytest.mark.parametrize(
    "receipt, restored",
    [(_receipt(["Mercor"], ["Mercor"]), True), (_receipt(["Mercor"], [], ["Mercor"]), False)],
)
def test_page_local_recovery_never_restores_a_page_whose_facet_did_not_land(receipt, restored, capsys) -> None:
    surface, owner = _recovery_surface(receipt)

    assert asyncio.run(surface._reanchor_recovery_page(owner, 1, "root")) is restored
    if not restored:
        assert "[scope] String #5: page recovery refused" in capsys.readouterr().out


def _active_switch(tmp_path, monkeypatch):
    import linkedin.evaluation_loop as loop_module
    from tests.test_linkedin_page_allocator_integration import _make_pipeline, _stage_active_opening_switch

    monkeypatch.setattr(loop_module.config, "LINKEDIN_PAGE_ALLOCATOR_MODE", "active")
    monkeypatch.setattr(loop_module.config, "LINKEDIN_TOTAL_PAGE_CAP", 1)
    pipeline = _make_pipeline(tmp_path)
    pipeline.state.context.brief_obj.raw = {"employer_allowlist": ALLOWLIST_FIELD}
    progress, current, sibling, *_rest = _stage_active_opening_switch(pipeline)
    for item in progress.strings:
        item.structured_filters = {}
    current.structured_filters = {"companies": ["Mercor"]}
    return pipeline, progress, current, sibling


def test_an_allocator_switch_stamps_the_root_it_bootstraps(tmp_path, monkeypatch) -> None:
    pipeline, progress, current, sibling = _active_switch(tmp_path, monkeypatch)
    allocator = pipeline.state._allocator_service
    allocator.work_units = MagicMock()
    allocator._emit_allocator_event_after_sync = MagicMock()
    allocator._checkpoint_progress(progress, search_string=current, page_num=1)
    # The selected root has never run: the actuation bootstraps its state.
    pipeline.state._experiment_states.pop(sibling.id)

    verdict = allocator._resume_active_allocator_actuation(
        progress, work_units=allocator.work_units,
        resumable_string_ids=pipeline.state.candidates.resumable_string_ids(),
    )

    assert verdict.selected_root_id == sibling.id
    stamped = sibling.structured_filters["companies"]
    assert stamped and all(ALLOWLIST.canonical_for(name) for name in stamped)
    root = pipeline.state._experiment_states[sibling.id].root_variant
    assert root.structured_filters.companies == stamped


def test_an_allocator_dispatch_stamps_the_root_it_bootstraps(tmp_path, monkeypatch) -> None:
    from linkedin.page_allocator import allocate_page

    pipeline, progress, current, sibling = _active_switch(tmp_path, monkeypatch)
    allocator = pipeline.state._allocator_service
    allocator._pending_allocator_checkpoint = None
    pipeline.state._experiment_states.pop(sibling.id)
    arms = allocator._allocator_arms(progress=progress, current=sibling)
    verdict = allocate_page(current_root_id=sibling.id, arms=arms)

    allocator._stage_active_allocator_dispatch(
        progress=progress, current=sibling, arms=arms, verdict=verdict
    )

    stamped = sibling.structured_filters["companies"]
    assert stamped and all(ALLOWLIST.canonical_for(name) for name in stamped)
    assert pipeline.state._experiment_states[sibling.id].root_variant.structured_filters.companies == stamped
