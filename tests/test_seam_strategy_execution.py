"""Phase 1 seam-contract tests: strategy -> execution wiring.

Each test pins the PRODUCER -> CONSUMER edge of one seam in the strategy
compile -> execution-queue -> runtime-variant pipeline so a refactor cannot
silently sever it. Seams that are not yet wired for *execution* are written as
strict xfails: the test runs the real producer and asserts the real consumer
observes the produced value, and fails AT the seam assertion until Phase 2
lifts the value onto the execution path.

External boundaries are the only things mocked: load_brief / init_judger /
LinkedInBrowser at Pipeline construction (so the orchestrator can be built
offline), and the fake browser for the search-mutation executor. The seam
producers and consumers themselves are never mocked.

Run with: .venv/bin/python -m pytest tests/test_seam_strategy_execution.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin.boolean_compiler import (
    BooleanNormalizationError,
    attach_constraint_lint_to_plan,
    compile_constraint,
)
from linkedin.search_intelligence import (
    LinkedInExperimentState,
    LinkedInPageInsights,
    LinkedInSearchIntent,
    LinkedInSearchVariant,
    LinkedInStructuredFilters,
    seed_structured_filters_onto_variants,
)
from linkedin.strategy_lane_compiler import apply_linkedin_lane_compiler_to_plan
from shared.schemas import CandidateSnippet, ExecutionPlan, Progress, SearchString
from shared.storage import read_jsonl
from shared.sourcing_lanes import (
    LaneExecution,
    SearchConstraint,
    SearchHypothesis,
    SearchSlice,
    SourcingLane,
)


# ---------------------------------------------------------------------------
# Shared builders (match tests/test_linkedin_pipeline.py::_make_pipeline and
# tests/test_linkedin_strategy_lane_compiler.py::_lane_dict patterns)
# ---------------------------------------------------------------------------


def _make_pipeline(output_dir: str):
    """Construct a Pipeline with external boundaries mocked (offline)."""
    with patch("linkedin.orchestrator.load_brief") as mock_brief, \
         patch("linkedin.orchestrator.init_judger"), \
         patch("linkedin.orchestrator.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        # Truthy bare-Mock permanent_filters.get("Location") would read as a
        # phantom geography and trip the P3a fail-closed gate; real briefs
        # carry a dict. Tests that exercise geography set Location explicitly.
        brief.permanent_filters = {}
        # Truthy bare-Mock needs_preflight() would trip the resume
        # regime-guard; these fixtures model already-complete briefs.
        brief.needs_preflight.return_value = False
        mock_brief.return_value = brief

        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline

        return Pipeline(brief_path=str(brief_path), output_dir=output_dir)


def _title_lane(
    *,
    lane_id: str = "senior-pool",
    acquisition_mode: str,
    titles: list[str],
    lane_structured_filters: dict | None = None,
    constraint_surface: str = "linkedin_title_filter",
    constraint_operator: str = "require",
) -> SourcingLane:
    """A SourcingLane whose slice carries a title constraint.

    `lane_structured_filters` defaults to {} so the constraint is the *only*
    title carrier (used to isolate Seam 0.0). Pass titles explicitly to also
    populate lane.execution.structured_filters (used by Seam 0.1).
    """
    return SourcingLane(
        lane_id=lane_id,
        lane_name="Senior Pool",
        hypothesis=SearchHypothesis(
            hypothesis_id="h1",
            label="Senior",
            target_archetype="leader",
            why_this_pool_may_exist="banks",
        ),
        slice=SearchSlice(
            slice_id="s1",
            hypothesis_id="h1",
            label="Slice",
            objective="find leaders",
            constraints=[
                SearchConstraint(
                    dimension="title",
                    values=list(titles),
                    execution_surface=constraint_surface,
                    operator=constraint_operator,
                )
            ],
        ),
        execution=LaneExecution(
            lane_id=lane_id,
            source="linkedin",
            acquisition_mode=acquisition_mode,
            boolean_strategy={"root_boolean": '"VP" AND engineering'},
            structured_filters=(
                lane_structured_filters
                if lane_structured_filters is not None
                else {}
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Seam 0.4 (PIN) — ExecutionPlan.generated_strings -> _build_ordered_search_strings
#
# Producer: Opus-built generated_strings on ExecutionPlan (strategy.py:703),
#           annotated by the lane helpers (sourcing_lanes.py:723).
# Consumer: orchestrator._build_ordered_search_strings (orchestrator.py:4862),
#           iterates generated_strings (:4879), drops empty boolean (:4881),
#           builds SearchString from gs['boolean'] (:4892) / gs['family_key']
#           (:4896) / **lane_fields_from_work_unit_item(gs) (:4904) which sets
#           acquisition_mode (sourcing_lanes.py:723,728).
# ---------------------------------------------------------------------------


def test_generated_strings_field_level_consumption_into_queue():
    """Real generated_string fields cross into the executed SearchString.

    Signatures found:
      ExecutionPlan.generated_strings: list[dict]  (shared/schemas.py:39)
      Pipeline._build_ordered_search_strings(self) -> list[SearchString]
        (linkedin/orchestrator.py:4862); reads gs.get("boolean") (:4880),
        `if not boolean: continue` (:4881-4882), family_key=gs.get("family_key")
        (:4896), **lane_fields_from_work_unit_item(gs) (:4904) ->
        {"acquisition_mode": ...} (shared/sourcing_lanes.py:728).
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._execution_plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {
                    "boolean": '"ml" AND research',
                    "rationale": "r",
                    "family_key": "fam",
                    "acquisition_mode": "linkedin_boolean",
                },
                # empty boolean must be dropped by the :4881 guard
                {"boolean": "", "rationale": "skip me", "family_key": "fam2"},
                {"rationale": "no boolean key at all", "family_key": "fam3"},
            ],
            coverage_gaps=[],
        )

        result = p._build_ordered_search_strings()

        # Two of three generated_strings have empty/absent boolean -> dropped.
        assert len(result) == 1
        ss = result[0]
        assert ss.boolean == '"ml" AND research'
        assert ss.family_key == "fam"
        assert ss.acquisition_mode == "linkedin_boolean"


def test_execution_queue_normalizes_generated_and_coverage_gap_booleans():
    """M1C firewall: initial generated strings and coverage gaps are normalized
    before SearchString execution, not merely able to carry preexisting metadata."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._execution_plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {
                    "boolean": (
                        '("Nubank" OR "fintech") AND '
                        '("reward model" OR "reward model development")'
                    ),
                    "rationale": "structured conflict plus redundant phrase",
                    "family_key": "fam",
                    "structured_filters": {"companies": ["Nubank"]},
                }
            ],
            coverage_gaps=[
                {
                    "suggested_boolean": '("ML Engineer" OR "platform")',
                    "gap": "title conflict",
                    "family_key": "gap",
                    "structured_filters": {"titles": ["ML Engineer"]},
                }
            ],
        )

        result = p._build_ordered_search_strings()

        assert [ss.boolean for ss in result] == [
            '("fintech") AND ("reward model")',
            '("platform")',
        ]
        compound_findings = [
            finding["code"]
            for finding in result[0].boolean_normalization["findings"]
        ]
        assert compound_findings == [
            "surface_conflict_stripped",
            "token_subset_superstring_pruned",
        ]
        assert result[1].boolean_normalization["findings"][0]["code"] == (
            "surface_conflict_stripped"
        )
        events = read_jsonl(p.log_path)
        surface_events = [event for event in events if event.get("event") == "surface_intended"]
        assert surface_events[-1]["normalization_guard_counts"] == {
            "ubiquitous_and_gate": 0,
            "token_subset_superstring_pruned": 1,
        }
        assert surface_events[-1]["normalization_strings_with_findings"] == 2


def test_execution_queue_blocks_explicit_ubiquitous_and_gate():
    """P5 (Wave 2): the ubiquity gate is fail-closed PER STRING at queue build.

    The offending string is blocked from the queue — recorded and logged, not
    executed — while healthy siblings still queue. (Previously the raise
    propagated and killed the whole queue build; error-blocks/warning-informs
    doctrine makes the block per-string.)
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._execution_plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {
                    "boolean": '("Python") AND ("PyTorch")',
                    "rationale": "too broad",
                    "family_key": "generic",
                    "ubiquitous_terms": ["python", "pytorch"],
                },
                {
                    "boolean": '("agent platform" OR "workflow orchestration")',
                    "rationale": "healthy",
                    "family_key": "healthy",
                },
            ],
            coverage_gaps=[],
        )

        result = p._build_ordered_search_strings()

        assert [ss.family_key for ss in result] == ["healthy"]
        blocked = p._lint_blocked_strings
        assert len(blocked) == 1
        assert blocked[0]["family_key"] == "generic"
        assert "ubiquitous_and_gate" in blocked[0]["codes"]
        events = read_jsonl(p.log_path)
        lint_events = [
            e for e in events if e.get("event") == "search_string_lint_blocked"
        ]
        assert len(lint_events) == 1
        assert "ubiquitous_and_gate" in lint_events[0]["codes"]


def test_lint_error_findings_block_queueing_through_production_path():
    """P5 (Wave 2): error-severity lint findings block queueing.

    Through the production path (_build_ordered_search_strings), a string with
    an error finding is skipped, its repair hint recorded, and a
    search_string_lint_blocked event logged; healthy strings still queue.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._execution_plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {
                    "boolean": '("agent platform" OR "workflow orchestration")',
                    "rationale": "healthy",
                    "family_key": "healthy",
                },
                {
                    "boolean": '("distributed training" OR "inference"',
                    "rationale": "unbalanced",
                    "family_key": "unbalanced",
                },
                {
                    "boolean": 'NOT ("recruiter" AND "agency")',
                    "rationale": "AND inside NOT",
                    "family_key": "not_and",
                },
            ],
            coverage_gaps=[
                {
                    "suggested_boolean": '("staffing" OR ) AND "platform"',
                    "gap": "empty operand",
                    "family_key": "gap_bad",
                },
            ],
        )

        result = p._build_ordered_search_strings()

        assert [ss.family_key for ss in result] == ["healthy"]
        codes_by_family = {b["family_key"]: b["codes"] for b in p._lint_blocked_strings}
        assert "unbalanced_parenthesis" in codes_by_family["unbalanced"]
        assert "not_group_contains_and" in codes_by_family["not_and"]
        assert "empty_or_group" in codes_by_family["gap_bad"]
        for blocked in p._lint_blocked_strings:
            assert any(hint for hint in blocked["repair_hints"])
        events = [
            e
            for e in read_jsonl(p.log_path)
            if e.get("event") == "search_string_lint_blocked"
        ]
        assert len(events) == 3


def test_lint_warning_findings_queue_and_attach_to_search_string():
    """P5 (Wave 2): warnings inform, never block — the string queues with its
    lint report attached so block reports can surface craft health."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._execution_plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {
                    "boolean": '("$1,000M budget" OR "fraud analytics") and ("payments")',
                    "rationale": "noop tokens",
                    "family_key": "warned",
                },
            ],
            coverage_gaps=[],
        )

        result = p._build_ordered_search_strings()

        assert len(result) == 1
        assert p._lint_blocked_strings == []
        findings = result[0].boolean_lint["findings"]
        codes = {f["code"] for f in findings}
        assert {"noop_special_character", "noop_comma_numeral", "lowercase_operator"} <= codes
        assert all(f["severity"] == "warning" for f in findings)


def test_structural_ubiquity_default_fires_at_execution_seam():
    """P5 (Wave 2): with no brief blacklist at all, the structural default set
    still refuses an AND gate composed entirely of ubiquitous terms — the gate
    can actually fire with the live default feed."""
    from types import SimpleNamespace

    from linkedin.boolean_compiler import (
        normalize_execution_work_item_boolean,
        ubiquitous_terms_from_brief,
    )

    terms = ubiquitous_terms_from_brief(SimpleNamespace(term_blacklist_categories=[]))
    item = {"boolean": '("AI") AND ("Engineer")'}
    with pytest.raises(BooleanNormalizationError, match="ubiquitous"):
        normalize_execution_work_item_boolean(
            item,
            boolean_key="boolean",
            ubiquitous_terms=terms,
        )


def test_brief_blacklist_terms_feed_ubiquity_gate_through_queue_build():
    """P5 (Wave 2): brief term_blacklist_categories feed the ubiquity gate at
    the queue-build call sites — an all-blacklist AND gate is blocked while a
    half-specific sibling queues."""
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.term_blacklist_categories = [
            SimpleNamespace(label="generic stack", rationale="", terms=["python", "pytorch"]),
        ]
        p._execution_plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {
                    "boolean": '("Python") AND ("PyTorch")',
                    "rationale": "all-blacklist",
                    "family_key": "blocked",
                },
                {
                    "boolean": '("Python") AND ("reward modeling")',
                    "rationale": "half-specific",
                    "family_key": "kept",
                },
            ],
            coverage_gaps=[],
        )

        result = p._build_ordered_search_strings()

        assert [ss.family_key for ss in result] == ["kept"]
        assert len(p._lint_blocked_strings) == 1
        assert "ubiquitous_and_gate" in p._lint_blocked_strings[0]["codes"]


def _company_lane(
    *,
    lane_id: str = "fam_employer_anchor",
    target_employers: list[str] | None = None,
) -> SourcingLane:
    target_employers = target_employers or [
        "Palantir Technologies",
        "Scale AI",
    ]
    return SourcingLane(
        lane_id=lane_id,
        lane_name="Employer anchor",
        hypothesis=SearchHypothesis(
            hypothesis_id=lane_id,
            label="Employer anchor",
            target_archetype="employer-bound engineers",
            why_this_pool_may_exist="Known employers carry the target motion.",
        ),
        slice=SearchSlice(
            slice_id=f"{lane_id}_slice",
            hypothesis_id=lane_id,
            label="Employer anchor",
            objective="Search inside the employer set with engineer keywords.",
            constraints=[
                SearchConstraint(
                    dimension="entry_signal",
                    values=["engineer", "engineering"],
                    execution_surface="boolean_keyword",
                    operator="prefer",
                ),
                SearchConstraint(
                    dimension="capability",
                    values=["LLM", "generative AI", "agentic", "RAG"],
                    execution_surface="boolean_keyword",
                    operator="prefer",
                ),
                SearchConstraint(
                    dimension="context",
                    values=["Palantir Technologies", "Scale AI"],
                    execution_surface="linkedin_company_filter",
                    operator="prefer",
                ),
            ],
        ),
        execution=LaneExecution(
            lane_id=lane_id,
            source="linkedin",
            acquisition_mode="linkedin_boolean",
            structured_filters={
                "target_employers": list(target_employers),
                "target_markets": ["general"],
            },
        ),
    )


def test_generated_string_projection_uses_matching_structured_lane_snapshot():
    """A flat generated string whose key is a lane variant must execute through the
    structured lane, not as a keyword-only orphan.

    This mirrors the FDE captured shape: lane_id ``fam_employer_anchor`` while
    generated_strings use keys like ``employer_anchor_palantir_scale``.
    """
    fde_companies = [
        "Palantir Technologies",
        "Scale AI",
        "OpenAI",
        "Hebbia",
        "Writer",
        "Deloitte",
        "EY",
        "Databricks",
        "Ramp",
        "Kalepa",
        "Arcesium",
    ]
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {
                    "boolean": (
                        '("engineer" OR "engineering") AND '
                        '("LLM" OR "generative AI")'
                    ),
                    "rationale": "employer anchor projection",
                    "family_key": "employer_anchor_palantir_scale",
                }
            ],
            sourcing_lanes=[
                _company_lane(target_employers=fde_companies).to_dict()
            ],
        )
        apply_linkedin_lane_compiler_to_plan(plan)

        p._execution_plan = plan
        result = p._build_ordered_search_strings()

        assert len(result) == 1
        ss = result[0]
        assert ss.boolean == (
            '("engineer" OR "engineering") AND ("LLM" OR "generative AI")'
        )
        assert ss.family_key == "employer_anchor_palantir_scale"
        assert ss.lane_id == "fam_employer_anchor"
        assert ss.acquisition_mode == "linkedin_hybrid"
        assert ss.surface == "hybrid"
        assert ss.structured_filters["companies"] == fde_companies


def test_lane_without_flat_projection_still_queues_structured_lane():
    """A structured lane with no generated_string projection must not strand."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[],
            sourcing_lanes=[_company_lane().to_dict()],
        )
        apply_linkedin_lane_compiler_to_plan(plan)

        p._execution_plan = plan
        result = p._build_ordered_search_strings()

        assert len(result) == 1
        ss = result[0]
        assert ss.lane_id == "fam_employer_anchor"
        assert ss.family_key == "fam_employer_anchor"
        assert ss.boolean == (
            '("engineer" OR "engineering") AND '
            '("LLM" OR "generative AI" OR "agentic" OR "RAG")'
        )
        assert ss.acquisition_mode == "linkedin_hybrid"
        assert ss.surface == "hybrid"
        assert ss.structured_filters["companies"] == [
            "Palantir Technologies",
            "Scale AI",
        ]


def test_all_keyword_lane_plan_keeps_legacy_queue_shape():
    """All-keyword lanes are still projected through the legacy queue shape."""
    generated = [
        {
            "boolean": '"deployment engineer" AND "workflow orchestration"',
            "rationale": "delivery builders",
            "family_key": "delivery_builders",
        },
        {
            "boolean": '"platform engineer" AND production',
            "rationale": "platform builders",
            "family_key": "platform_builders",
        },
    ]
    keyword_lane = SourcingLane(
        lane_id="fam_delivery_builders",
        lane_name="Delivery builders",
        hypothesis=SearchHypothesis(
            hypothesis_id="fam_delivery_builders",
            label="Delivery builders",
            target_archetype="delivery engineers",
            why_this_pool_may_exist="They do the work under adjacent titles.",
        ),
        slice=SearchSlice(
            slice_id="fam_delivery_builders_slice",
            hypothesis_id="fam_delivery_builders",
            label="Delivery builders",
            objective="Keyword-only adjacent pool.",
            constraints=[
                SearchConstraint(
                    dimension="entry_signal",
                    values=["deployment engineer"],
                    execution_surface="boolean_keyword",
                )
            ],
        ),
        execution=LaneExecution(
            lane_id="fam_delivery_builders",
            source="linkedin",
            acquisition_mode="linkedin_boolean",
        ),
    )

    with tempfile.TemporaryDirectory() as td:
        legacy_pipeline = _make_pipeline(td)
        legacy_pipeline._execution_plan = ExecutionPlan(
            strategy_rationale="legacy",
            generated_strings=[dict(item) for item in generated],
        )
        legacy_queue = legacy_pipeline._build_ordered_search_strings()

    with tempfile.TemporaryDirectory() as td:
        lane_pipeline = _make_pipeline(td)
        plan = ExecutionPlan(
            strategy_rationale="with lane",
            generated_strings=[dict(item) for item in generated],
            sourcing_lanes=[keyword_lane.to_dict()],
        )
        apply_linkedin_lane_compiler_to_plan(plan)
        lane_pipeline._execution_plan = plan
        lane_queue = lane_pipeline._build_ordered_search_strings()

    assert [item.to_dict() for item in lane_queue] == [
        item.to_dict() for item in legacy_queue
    ]


# ---------------------------------------------------------------------------
# Seam 0.1 (SPLIT) — LinkedInLaneCompiler.compile() query_payload + acquisition_mode
#   -> apply_linkedin_lane_compiler_to_plan -> _build_ordered_search_strings
#
# PASS-half: acquisition_mode IS consumed end-to-end. The compiler writes
#   item['acquisition_mode'] (strategy_lane_compiler.py:55); the builder reads it
#   via lane_fields_from_work_unit_item (sourcing_lanes.py:723) ->
#   SearchString.acquisition_mode (orchestrator.py:4904).
# XFAIL-half (now wired, Phase 2 hop 2): compiler query_payload.structured_filters
#   was written only into lane_snapshot['compiler']; lane_fields_from_work_unit_item
#   (sourcing_lanes.py) now lifts it onto the new SearchString.structured_filters
#   field (shared/schemas.py), so execution can read it. (advanced_search_plan is
#   still snapshot-only.)
# ---------------------------------------------------------------------------


def test_lane_compiler_acquisition_mode_reaches_executed_search_string():
    """PIN-half: the compiler's acquisition_mode crosses onto the queued SearchString.

    Signatures found:
      apply_linkedin_lane_compiler_to_plan(plan) -> int (strategy_lane_compiler.py:12);
        sets item["acquisition_mode"] = snapshot.get("acquisition_mode") (:55).
      LinkedInLaneCompiler.compile(...).acquisition_mode = lane.execution.acquisition_mode
        (lane_compiler.py:37,83).
      Consumer: lane_fields_from_work_unit_item(item)["acquisition_mode"]
        (sourcing_lanes.py:723,728) -> SearchString(**...) (orchestrator.py:4904).
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        lane = _title_lane(
            acquisition_mode="linkedin_hybrid",
            titles=["VP Engineering"],
            lane_structured_filters={"titles": ["VP Engineering"]},
        )
        plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {
                    "boolean": '"VP" AND engineering',
                    "rationale": "senior leaders",
                    "lane_id": "senior-pool",
                }
            ],
            sourcing_lanes=[lane.to_dict()],
        )
        # Real producer: compile lanes and annotate generated_strings.
        wired = apply_linkedin_lane_compiler_to_plan(plan)
        assert wired == 1  # producer actually fed the generated_string

        p._execution_plan = plan
        result = p._build_ordered_search_strings()

        assert len(result) == 1
        # The compiler's acquisition_mode crossed the seam onto the live queue.
        assert result[0].acquisition_mode == "linkedin_hybrid"


def test_lane_compiler_structured_filters_reach_executed_search_string():
    """PIN (wired in Phase 2 hop 2): the compiler's structured_filters reach execution.

    Producer: LinkedInLaneCompiler.compile(...).query_payload["structured_filters"]
      (lane_compiler.py:85-89) carries titles=["VP Engineering"].
    Consumer (intended): the executed SearchString exposes those filters where
      runtime mutation reads them. SearchString has no such field today, so the
      compiled filters never reach execution.

    Wired in Phase 2 hop 2: lane_fields_from_work_unit_item lifts
    query_payload['structured_filters'] onto SearchString.structured_filters (a plain
    dict), so getattr(ss, "structured_filters")["titles"] == ["VP Engineering"]. Goes
    red if that lift is removed or SearchString loses the field.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        lane = _title_lane(
            acquisition_mode="linkedin_hybrid",
            titles=["VP Engineering"],
            lane_structured_filters={"titles": ["VP Engineering"]},
        )
        plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {
                    "boolean": '"VP" AND engineering',
                    "rationale": "senior leaders",
                    "lane_id": "senior-pool",
                }
            ],
            sourcing_lanes=[lane.to_dict()],
        )
        apply_linkedin_lane_compiler_to_plan(plan)

        p._execution_plan = plan
        ss = p._build_ordered_search_strings()[0]

        # Execution would read structured filters from a first-class field on the
        # SearchString (not the diagnostic lane_snapshot['compiler'] dict, which
        # has no execution consumer). That field does not exist yet -> seam fail.
        executable_filters = getattr(ss, "structured_filters", None)
        titles = []
        if executable_filters is not None:
            titles = getattr(executable_filters, "titles", None) or (
                executable_filters.get("titles", [])
                if isinstance(executable_filters, dict)
                else []
            )
        assert titles == ["VP Engineering"]


# ---------------------------------------------------------------------------
# Seam 0.0 (PIN — wired in Phase 1 hop 1) — compile_constraint().structured_control
#   -> query_payload
#
# Producer: compile_constraint(SearchConstraint(execution_surface=
#   'linkedin_title_filter', operator='require', values=['VP Engineering']))
#   -> ExecutableConstraint.structured_control={'dimension':'title','values':
#   ['VP Engineering'],...} (boolean_compiler.py:726-732).
# Consumer: LinkedInLaneCompiler.compile (lane_compiler.py) now folds the slice's
#   structured constraints into query_payload['structured_filters'] via
#   _merge_slice_constraints_into_filters, so the constraint's title control reaches
#   the queued SearchString's lane_snapshot['compiler'] even when
#   lane.execution.structured_filters is empty. Before hop 1 the sole caller of
#   compile_constraint was attach_constraint_lint_to_plan (boolean_compiler.py:837),
#   which fed only the linter.
# ---------------------------------------------------------------------------


def test_compile_constraint_produces_title_structured_control():
    """Sanity: the producer genuinely emits the title structured_control.

    This is NOT the seam assertion (it stays inside the producer); it guards the
    xfail below against silently degrading into a producer-side no-op.

    Signature found:
      compile_constraint(constraint, *, source="linkedin") -> ExecutableConstraint
        (boolean_compiler.py:701); for linkedin_title_filter sets
        structured_control={"source","dimension":"title","values","operator",
        "temporal_scope"} (:726-732).
    """
    compiled = compile_constraint(
        SearchConstraint(
            dimension="title",
            values=["VP Engineering"],
            execution_surface="linkedin_title_filter",
            operator="require",
        )
    )
    assert compiled.structured_control["dimension"] == "title"
    assert compiled.structured_control["values"] == ["VP Engineering"]


def test_constraint_structured_control_reaches_executed_search_string():
    """PIN (wired in Phase 1 hop 1): a title constraint's structured_control reaches the queue.

    Producer drives a SourcingLane whose slice has a linkedin_title_filter
    constraint for ['VP Engineering'] but whose lane.execution.structured_filters
    is EMPTY (so the constraint is the only title carrier). After the real strategy
    compile path (attach_constraint_lint_to_plan + apply_linkedin_lane_compiler_to_plan)
    and _build_ordered_search_strings, the queued SearchString must carry a title
    control derived from that constraint.

    Wired in Phase 1 hop 1: LinkedInLaneCompiler.compile folds the slice's structured
    constraints into query_payload['structured_filters'] via
    _merge_slice_constraints_into_filters (lane_compiler.py), so the constraint's title
    control reaches ss.lane_snapshot['compiler']['query_payload']['structured_filters']
    ['titles'] even with an empty lane.execution.structured_filters. Goes red if that
    merge is removed or compile_constraint stops emitting the title structured_control.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        lane = _title_lane(
            acquisition_mode="linkedin_boolean",
            titles=["VP Engineering"],          # constraint values
            lane_structured_filters={},          # lane carries NO structured filters
        )
        plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {
                    "boolean": '"VP" AND engineering',
                    "rationale": "senior leaders",
                    "lane_id": "senior-pool",
                }
            ],
            sourcing_lanes=[lane.to_dict()],
        )
        # Real strategy compile path (the only callers of compile_constraint and
        # the lane compiler).
        attach_constraint_lint_to_plan(plan)
        apply_linkedin_lane_compiler_to_plan(plan)

        p._execution_plan = plan
        ss = p._build_ordered_search_strings()[0]

        compiler_snapshot = ss.lane_snapshot.get("compiler", {})
        query_payload = (
            compiler_snapshot.get("query_payload", {})
            if isinstance(compiler_snapshot, dict)
            else {}
        )
        titles = (query_payload.get("structured_filters") or {}).get("titles", [])
        assert titles == ["VP Engineering"]


# ---------------------------------------------------------------------------
# R3 (PIN — wired in Phase 1 R3) — a slice/per-lane location constraint reaches a
#   'locations' control.
#
# Producer: a SourcingLane whose slice carries a linkedin_location_filter constraint.
#   compile_constraint emits structured_control={'dimension':'location',...}
#   (boolean_compiler.py:720-732).
# Consumer: LinkedInLaneCompiler.compile -> compile_structured_filters_to_plan, whose
#   sole location reader is sidebar_filters['locations'] (advanced_search.py:268-271).
#   (The session-location path — Build-1 / _apply_session_location_filter — is a SEPARATE
#   direct-apply route off permanent_filters that bypasses compile; it never enters this
#   bucket. Slice and session locations converge only at browser.apply_location_filter.)
#
# Pre-fix the lane compiler parked the location in advanced_filters['locations']
# (_merge_slice_constraints_into_filters, lane_compiler.py), which compile never reads
# for locations, so the plan dimmed to ['keywords'] and the lane searched geographically
# unbounded. R3 routes a slice location to sidebar_filters['locations'], the key compile
# reads — NOT a shared bucket with the session path (which direct-applies, above).
# ---------------------------------------------------------------------------


def _location_lane(
    *,
    lane_id: str = "geo-pool",
    locations: list[str] | None,
    operator: str = "require",
) -> SourcingLane:
    """A SourcingLane whose slice carries a linkedin_location_filter constraint.

    ``locations=None`` builds a slice with NO location constraint (the negative
    guard); the lane otherwise matches the title-lane shape so only the constraint
    dimension differs.
    """
    constraints = []
    if locations is not None:
        constraints.append(
            SearchConstraint(
                dimension="location",
                values=list(locations),
                execution_surface="linkedin_location_filter",
                operator=operator,
            )
        )
    return SourcingLane(
        lane_id=lane_id,
        lane_name="Geo Pool",
        hypothesis=SearchHypothesis(
            hypothesis_id="h1",
            label="Geo",
            target_archetype="leader",
            why_this_pool_may_exist="banks",
        ),
        slice=SearchSlice(
            slice_id="s1",
            hypothesis_id="h1",
            label="Slice",
            objective="find leaders in a geography",
            constraints=constraints,
        ),
        execution=LaneExecution(
            lane_id=lane_id,
            source="linkedin",
            acquisition_mode="linkedin_boolean",
            boolean_strategy={"root_boolean": '"VP" AND engineering'},
            structured_filters={},
        ),
    )


def test_slice_location_constraint_reaches_locations_control():
    """PIN (R3): a slice linkedin_location_filter constraint yields a 'locations' control.

    Drives the real LinkedInLaneCompiler.compile on a lane whose slice carries a
    linkedin_location_filter constraint for ['New York City'] and whose
    lane.execution.structured_filters is EMPTY (so the constraint is the only location
    carrier). The compiled plan must expose a 'locations' control with that value —
    dims include 'locations', not ['keywords'] alone.

    Pre-fix FAILS: _merge_slice_constraints_into_filters parked the location in
    advanced_filters['locations'], which compile_structured_filters_to_plan never reads
    for locations, so the location dead-ended and dims were ['keywords'] only. R3 parks a
    slice location in sidebar_filters['locations'] — the key compile reads — so it reaches
    the single location reader (advanced_search.py:268-271). (The session path
    direct-applies off permanent_filters and does not share this bucket.)
    Goes red if that route is removed or compile stops reading sidebar_filters['locations'].
    """
    from linkedin.lane_compiler import LinkedInLaneCompiler

    lane = _location_lane(locations=["New York City"])
    exe = LinkedInLaneCompiler().compile(lane)

    plan = exe.query_payload["advanced_search_plan"]
    controls = plan["controls"]
    dims = [c["dimension"] for c in controls]
    assert "locations" in dims, dims

    location_controls = [c for c in controls if c["dimension"] == "locations"]
    assert len(location_controls) == 1, location_controls
    assert location_controls[0]["values"] == ["New York City"]

    # Parked in sidebar_filters (the key compile reads), not advanced_filters (which has
    # no location reader). The session path is separate (direct-apply off permanent_filters).
    structured = exe.query_payload["structured_filters"]
    assert structured.get("sidebar_filters", {}).get("locations") == ["New York City"]
    assert "locations" not in structured.get("advanced_filters", {})


def test_slice_without_location_constraint_emits_no_locations_control():
    """Contrast guard (R3): a slice with NO location constraint emits no 'locations' control.

    The R3 route must not fabricate a geography. A lane whose slice carries no
    linkedin_location_filter constraint (and whose execution.structured_filters is empty)
    compiles to a plan with no 'locations' control — which would otherwise silently narrow
    an intentionally-unbounded search. Goes red if the location-parking route emits a
    control unconditionally."""
    from linkedin.lane_compiler import LinkedInLaneCompiler

    lane = _location_lane(locations=None)
    exe = LinkedInLaneCompiler().compile(lane)

    plan = exe.query_payload["advanced_search_plan"]
    dims = [c["dimension"] for c in plan["controls"]]
    assert "locations" not in dims, dims
    structured = exe.query_payload["structured_filters"]
    assert "locations" not in structured.get("sidebar_filters", {})


# ---------------------------------------------------------------------------
# Seam 0.2 / hop 3 (PIN — wired in Phase 2 hop 3) — hybrid+structured variant
#   reaches apply_variant
#
# Consumer: LinkedInSearchMutationExecutor.apply_variant (search_mutation.py:50);
#   used_hybrid = acquisition_mode=='linkedin_hybrid' and not
#   variant.structured_filters.is_empty() (:178-181) -> apply_advanced_search_plan.
# Trigger (product rule): a lane carrying any structured filter compiles to
#   acquisition_mode='linkedin_hybrid' (lane_compiler.py), because the apply_variant
#   gate rejects a boolean variant with structured filters (search_mutation.py:60-73).
# Seeding: seed_structured_filters_onto_variants (search_intelligence.py), called by
#   the orchestrator before begin_experiment_round (orchestrator.py:~1983) and the
#   drift apply (orchestrator.py:~1931), copies the SearchString's structured filters
#   onto each runtime variant whose own filters are empty — so next_planned_variant
#   hands apply_variant a hybrid + non-empty-filter variant. Live variants are still
#   Opus-built with empty filters until seeded.
#
# (Seam 0.2 PIN-half -- structured filters + linkedin_boolean ->
#  blocked_reason "experimental_structured_filters_not_supported" -- is already
#  pinned by tests/test_linkedin_search_intelligence.py:89-111; not duplicated here.)
# ---------------------------------------------------------------------------


def test_structured_filters_force_hybrid_acquisition_mode():
    """PIN (hop-3 trigger): a lane carrying structured filters compiles to hybrid.

    Even a lane explicitly declared linkedin_boolean is forced to linkedin_hybrid when
    it carries structured filters, because the apply_variant gate
    (search_mutation.py:60-73) rejects a boolean variant with non-empty filters. Pins
    the product decision: structured_filters present <=> hybrid mode. Goes red if the
    trigger in LinkedInLaneCompiler.compile stops forcing hybrid.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        lane = _title_lane(
            acquisition_mode="linkedin_boolean",  # explicitly boolean...
            titles=["VP Engineering"],
            lane_structured_filters={"titles": ["VP Engineering"]},  # ...but has filters
        )
        plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {"boolean": '"VP"', "rationale": "r", "lane_id": "senior-pool"}
            ],
            sourcing_lanes=[lane.to_dict()],
        )
        apply_linkedin_lane_compiler_to_plan(plan)
        p._execution_plan = plan
        ss = p._build_ordered_search_strings()[0]

        # The trigger overrode the declared boolean mode.
        assert ss.acquisition_mode == "linkedin_hybrid"
        assert ss.structured_filters.get("titles") == ["VP Engineering"]


def test_runtime_feeds_hybrid_structured_variant_into_apply_variant():
    """PIN (wired in Phase 2 hop 3): the runtime hands a hybrid+structured variant to apply_variant.

    Drives the real path: a hybrid SearchString built through the lane compiler (which
    carries structured_filters via hop 2 and acquisition_mode='linkedin_hybrid' via the
    hop-3 trigger), an Opus-style runtime variant with empty filters (as
    _plan_variant_experiments builds, orchestrator.py:3367-3376), then the real
    seed_structured_filters_onto_variants + begin_experiment_round + next_planned_variant
    sequence the orchestrator runs (orchestrator.py:~1983-2001).

    Goes red if the trigger stops forcing hybrid or the seeding helper stops copying
    the SearchString's filters onto the empty runtime variant.

    Signatures found:
      seed_structured_filters_onto_variants(structured_filters: dict, variants) -> None
        (search_intelligence.py); seeds only variants whose own filters are empty.
      LinkedInExperimentState.begin_experiment_round(self, variants) (search_intelligence.py:548)
        — does not touch structured_filters, so a seeded variant survives selection.
      LinkedInExperimentState.next_planned_variant(self) (search_intelligence.py:564).
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        lane = _title_lane(
            acquisition_mode="linkedin_hybrid",
            titles=["VP Engineering"],
            lane_structured_filters={"titles": ["VP Engineering"]},
        )
        plan = ExecutionPlan(
            strategy_rationale="test",
            generated_strings=[
                {"boolean": '"VP" AND engineering', "rationale": "r", "lane_id": "senior-pool"}
            ],
            sourcing_lanes=[lane.to_dict()],
        )
        apply_linkedin_lane_compiler_to_plan(plan)
        p._execution_plan = plan
        search_string = p._build_ordered_search_strings()[0]

        # The runtime builds Opus variants with empty structured_filters
        # (orchestrator.py:3367-3376); begin_experiment_round is the real entry.
        runtime_variant = LinkedInSearchVariant(
            variant_id="round-1-1",
            parent_variant_id="root",
            root_string_id=search_string.id,
            boolean='"VP" AND senior',
            variant_kind="precision",
        )
        assert runtime_variant.structured_filters.is_empty()  # as the runtime builds it

        # The real orchestrator step (orchestrator.py:~1983): seed, then begin round.
        seed_structured_filters_onto_variants(
            search_string.structured_filters, [runtime_variant]
        )
        state = LinkedInExperimentState(
            root_string_id=search_string.id,
            intent=LinkedInSearchIntent(root_boolean='"VP" AND engineering'),
        )
        state.begin_experiment_round([runtime_variant])

        next_variant = state.next_planned_variant()
        resolved_mode = search_string.acquisition_mode or "linkedin_boolean"

        # The seam: the runtime hands apply_variant a hybrid + non-empty-filter variant.
        assert resolved_mode == "linkedin_hybrid"
        assert next_variant is not None
        assert not next_variant.structured_filters.is_empty()
        assert next_variant.structured_filters.titles == ["VP Engineering"]


# ---------------------------------------------------------------------------
# Hop-4 producer (PIN) — session-level location filter from the brief
#
# Location is a session fact (one geography per run, from the brief), applied once on
# the freshly-navigated sidebar by Pipeline._apply_session_location_filter and
# re-asserted after recovery re-navigations — NOT threaded through the per-lane
# constraint grammar. browser.apply_location_filter is the live stable_now apply (hop 4).
# ---------------------------------------------------------------------------


def test_session_location_filter_applies_brief_geography():
    """PIN: the brief's geography is applied once via apply_location_filter with the
    current-or-past scope, idempotent within a session (no re-apply without a flag reset)."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {"Location": "New York City Metropolitan Area"}
        p.browser.apply_location_filter = AsyncMock(return_value=True)

        asyncio.run(p._apply_session_location_filter())

        p.browser.apply_location_filter.assert_awaited_once_with(
            ["New York City Metropolitan Area"], temporal_scope="current_or_past"
        )
        assert p._session_location_applied is True
        # Idempotent: a second call without a flag reset does not re-apply.
        asyncio.run(p._apply_session_location_filter())
        p.browser.apply_location_filter.assert_awaited_once()


def test_session_location_filter_noop_without_geography():
    """No geography on the brief -> the producer never touches the browser."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {}
        p.browser.apply_location_filter = AsyncMock(return_value=True)
        asyncio.run(p._apply_session_location_filter())
        p.browser.apply_location_filter.assert_not_awaited()


def test_session_location_filter_fails_closed():
    """P3a (decided 2026-07-03): a miss (apply returns False) ABORTS the run with
    GeographyRegimeError — never a boolean-only proceed. The prior fail-soft here
    shipped a live run where every save came back off-geography."""
    from linkedin.orchestrator import GeographyRegimeError

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {"Location": "Brazil"}
        p.browser.apply_location_filter = AsyncMock(return_value=False)
        with pytest.raises(GeographyRegimeError):
            asyncio.run(p._apply_session_location_filter())
        assert p._session_location_applied is False


# ---------------------------------------------------------------------------
# R4 — crash-recovery re-asserts the session location
#
# The P6 browser-disconnect crash-recovery flow (orchestrator run_full crash block
# -> _recovery_service.recover -> recover_recruiter_context) re-binds the tab and
# re-navigates the search surface, dropping the sidebar location chip. The session
# geography rides apply_location_filter directly (Build-1 / _apply_session_location_filter),
# which zeroes _last_search_snapshot, so the recovery snapshot has NO location and the
# replay plan dims to keywords-only (test_seam_recovery_browser.py seam 4.1). Without an
# independent re-assert the resumed search loses its geography and returns an over-broad,
# location-unbounded result set the recruiter never asked to widen.
#
# Consumer of the fix: Pipeline._reassert_session_location_after_recovery, called from the
# crash-recovery block after recovery succeeds, mirroring the legacy check_and_recover
# re-assert in _ensure_browser_healthy (orchestrator.py). It resets _session_location_applied
# then reuses _apply_session_location_filter (idempotent + fail-soft) to re-apply the brief
# geography on the re-bound sidebar.
# ---------------------------------------------------------------------------


def test_recovery_reasserts_session_location_after_browser_crash():
    """PIN (R4): after a browser-crash recovery, the brief's session location is re-applied.

    Arrange the post-initial state of a session-location-only run: brief Location set,
    _session_location_applied=True (already applied on the fresh sidebar at Build-1), the
    browser re-bound onto a talent/hire search URL (as recover_recruiter_context leaves it).
    Run the orchestrator's post-recovery re-assert. The session location must cross back onto
    the live sidebar via apply_location_filter(['New York City']) with the 'current' scope —
    even though _session_location_applied was True, because the re-assert resets that flag.

    Goes red pre-fix: the crash-recovery block never re-asserts, _session_location_applied
    stays True, and the recovery snapshot carries no location (it rode apply_location_filter,
    which zeroes the snapshot), so the resumed search is location-unbounded.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {"Location": "New York City"}
        # The browser is re-bound on the search surface after recover_recruiter_context.
        p.browser.page = MagicMock()
        type(p.browser.page).url = MagicMock(
            return_value="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
        )
        p.browser.apply_location_filter = AsyncMock(return_value=True)
        # Post-initial-apply state: the session location was already applied at Build-1.
        p._session_location_applied = True

        asyncio.run(p._reassert_session_location_after_recovery())

        # The session geography is re-applied on the re-bound sidebar.
        p.browser.apply_location_filter.assert_awaited_once_with(
            ["New York City"], temporal_scope="current_or_past"
        )
        assert p._session_location_applied is True


def test_recovery_does_not_reassert_location_without_brief_geography():
    """Contrast guard (R4): a recovery for a brief with NO Location must not touch
    apply_location_filter — the re-assert must never fabricate a geography the brief
    never carried, which would silently narrow an intentionally-unbounded search."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {}  # no geography on the brief
        p.browser.apply_location_filter = AsyncMock(return_value=True)
        p._session_location_applied = False

        asyncio.run(p._reassert_session_location_after_recovery())

        p.browser.apply_location_filter.assert_not_awaited()


def test_recovery_location_reassert_fails_closed():
    """P3a: a re-assert miss PROPAGATES — a recovery that cannot restore the session
    geography must not resume an unbounded search. (Inverts the pre-P3a fail-soft.)"""
    from linkedin.orchestrator import GeographyRegimeError

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.permanent_filters = {"Location": "New York City"}
        p.browser.apply_location_filter = AsyncMock(side_effect=RuntimeError("sidebar gone"))
        p._session_location_applied = True

        with pytest.raises(GeographyRegimeError):
            asyncio.run(p._reassert_session_location_after_recovery())
        assert p._session_location_applied is False


def _crash_recovery_pipeline(td: str, *, location: str | None):
    """Stage one resumable string for run_full finalization tests."""
    p = _make_pipeline(td)
    p.brief_obj.permanent_filters = {"Location": location} if location else {}

    progress = Progress(
        brief_name="test",
        strings=[
            SearchString(
                id=5, name="interrupted", boolean="one", status="queued", block="Block A"
            )
        ],
    )
    progress.save(str(p.progress_path))

    p.browser.connect = AsyncMock()
    p.browser.disconnect = AsyncMock()
    # The brief's OWN project search view — where a live run sits once run-start
    # has navigated. The global /talent/search view names no project, so with a
    # project-bearing brief it is the F1 "unverified page" condition and run-start
    # navigates off it; tests that mean to exercise that set it explicitly.
    p.browser.page = MagicMock(
        url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
    )
    # Recovery succeeds without exercising the live recover_recruiter_context machinery;
    # the snapshot branch is what the orchestrator always takes (snapshot is non-None).
    p._recovery_service.recover = AsyncMock(return_value=True)
    p._attempt_reconnect = AsyncMock(return_value=False)
    p._print_session_summary = MagicMock()
    p._print_summary = MagicMock()
    p._generate_run_report = MagicMock()
    return p


def _seed_run_full_resume_progress(p, strings: list[SearchString]) -> None:
    Progress(brief_name="test", strings=strings).save(str(p.progress_path))


def _seed_runtime_judge_decisions(p, decisions: list[str]) -> None:
    work_unit_id = p._runtime_state.upsert_work_unit(
        run_id=p._runtime_run_id,
        source="linkedin",
        brief_id=p.brief_obj.linkedin_project_id,
        kind="linkedin_string",
        source_unit_id="1",
        display_name="String 1",
        ordering_index=0,
        status="done",
    )
    for index, decision in enumerate(decisions):
        identity_key = f"candidate-{p._runtime_run_id}-{index}"
        p._runtime_state.record_candidate_discovery(
            run_id=p._runtime_run_id,
            work_unit_id=work_unit_id,
            source="linkedin",
            brief_id=p.brief_obj.linkedin_project_id,
            identity_key=identity_key,
            display_name=identity_key,
            profile_url=f"https://example.test/{identity_key}",
        )
        p._runtime_state.set_candidate_state(
            run_id=p._runtime_run_id,
            source="linkedin",
            brief_id=p.brief_obj.linkedin_project_id,
            identity_key=identity_key,
            new_state="failed_terminal",
            terminal_decision=decision,
            terminal_payload={"full_decision": {"decision": decision}},
            last_work_unit_id=work_unit_id,
        )




# ---------------------------------------------------------------------------
# SLICE A — producer reasons execution_surface per constraint.
#
# The downstream pipe is already verified by the seams above: a layer item's
# structured_surface -> SearchConstraint.execution_surface
# (search_constraints_from_layer_items, sourcing_lanes.py:780-793) ->
# compile_constraint -> _merge_slice_constraints_into_filters folds it into
# structured_filters + flips acquisition_mode=linkedin_hybrid (lane_compiler.py:141)
# -> lifted onto SearchString.structured_filters (lane_fields_from_work_unit_item,
# sourcing_lanes.py:716-739). Slice A drives that full producer path from a real
# RetrievalFamily and pins:
#   (a) a retrieval-family item carrying structured_surface="linkedin_company_filter"
#       reaches SearchString.structured_filters["companies"] + linkedin_hybrid.
#   (b) byte-identical default: an all-keyword family keeps every queued string on
#       empty filters + linkedin_boolean.
#   (c) injection gate: an unknown structured_surface collapses to "" (keyword) at the
#       allow-list (retrieval_design.py:101), so no structured control is produced.
#   (d) repair_constraint_surfaces flips ONLY a title-like, boolean_keyword,
#       current-temporal constraint carrying the temporal_scope_mismatch finding; a
#       non-title constraint with the same finding is left advisory.
# ---------------------------------------------------------------------------


def _filters_are_empty(structured_filters: dict | None) -> bool:
    """A queued SearchString carries no live structured facet. The compiler always
    materializes the 6-key LinkedInStructuredFilters.to_dict() shape (titles/companies/
    skills/assessments/sidebar_filters/advanced_filters), so 'no control' means every
    one of those is falsy — NOT a literal {} (which the keyword path never produced,
    before or after Slice A)."""
    sf = structured_filters or {}
    return not any(bool(v) for v in sf.values())


def _family_plan(family_payload: dict) -> ExecutionPlan:
    """An ExecutionPlan seeded from a single retrieval family, run through the REAL
    family -> sourcing-lane producer (populate_execution_plan_lane_payloads) + the
    real lane compiler. The generated_string keys to the family's lane by lane_id so
    the queue builder (_build_ordered_search_strings) lifts the compiled fields."""
    from shared.sourcing_lanes import (
        normalize_lane_id,
        populate_execution_plan_lane_payloads,
    )

    lane_id = normalize_lane_id(family_payload["family_id"])
    plan = ExecutionPlan(
        strategy_rationale="test",
        retrieval_families=[family_payload],
        generated_strings=[
            {
                "boolean": '"VP" AND engineering',
                "rationale": "from the family",
                "lane_id": lane_id,
            }
        ],
    )
    # Real producer: families -> sourcing_lanes (this is what _materialize_retrieval_plan
    # runs in form_strategy before the lane compiler).
    populate_execution_plan_lane_payloads(plan)
    apply_linkedin_lane_compiler_to_plan(plan)
    return plan


def test_family_company_surface_reaches_structured_filters_and_hybrid_mode():
    """(a) A layer item opting into linkedin_company_filter reaches the executed
    SearchString's structured_filters['companies'] and forces linkedin_hybrid.

    Drives the real family -> lane -> compile -> queue path. The company surface on an
    entry_signal (a positive 'prefer' constraint) compiles to a {dimension:'company'}
    structured_control (boolean_compiler.py:726-732), folds into structured_filters
    ['companies'] (_merge_slice_constraints_into_filters via _STRUCTURED_DIMENSION_FIELD,
    lane_compiler.py:31,88-97), trips the not-empty -> linkedin_hybrid trigger
    (lane_compiler.py:141), and is lifted onto SearchString.structured_filters
    (sourcing_lanes.py:716-739). Goes red if the surface stops flowing through any link.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        plan = _family_plan(
            {
                "family_id": "stripe_alumni",
                "label": "Stripe payments leaders",
                "objective": "Bound the pool to a real employer LinkedIn indexes.",
                "entry_signals": [
                    {
                        "item_id": "e1",
                        "label": "Employer: Stripe",
                        "terms": ["Stripe"],
                        "structured_surface": "linkedin_company_filter",
                    }
                ],
            }
        )
        p._execution_plan = plan
        ss = p._build_ordered_search_strings()[0]

        companies = (getattr(ss, "structured_filters", None) or {}).get("companies", [])
        assert companies == ["Stripe"], ss.structured_filters
        assert ss.acquisition_mode == "linkedin_hybrid"
        assert ss.search_posture == "structured_only"

        # Part 5 (telemetry): the lane snapshot records the derived search posture. The
        # slice's only constraint is the company filter -> structured_only. This proves
        # posture is INDEPENDENT of acquisition_mode: the posture is 'structured_only'
        # while the mode is 'linkedin_hybrid' (lane_compiler derives mode from
        # structured.is_empty(), never from the posture).
        snapshot = plan.sourcing_lanes[0]["lane_compiler"]
        assert snapshot["search_posture"] == "structured_only", snapshot.get("search_posture")


def test_compiled_execution_view_reports_hybrid_mode_honestly():
    """Fix 2 (honesty): after apply_linkedin_lane_compiler_to_plan, the stored
    lane.execution view reports the COMPILED acquisition_mode/posture, not the frozen
    mint default. Before this fix a hybrid lane kept execution.acquisition_mode=
    'linkedin_boolean' (lane_execution_from_retrieval_family mints the default) while
    its own lane_compiler snapshot said 'linkedin_hybrid' — any audit reading
    execution.* mis-read the lane as Boolean. Goes red if the execution view stops
    being reconciled to the snapshot.
    """
    plan = _family_plan(
        {
            "family_id": "stripe_alumni",
            "label": "Stripe payments leaders",
            "objective": "Bound the pool to a real employer LinkedIn indexes.",
            "entry_signals": [
                {
                    "item_id": "e1",
                    "label": "Employer: Stripe",
                    "terms": ["Stripe"],
                    "structured_surface": "linkedin_company_filter",
                }
            ],
        }
    )
    lane = plan.sourcing_lanes[0]
    assert lane["lane_compiler"]["acquisition_mode"] == "linkedin_hybrid"
    assert lane["execution"]["acquisition_mode"] == "linkedin_hybrid", lane["execution"]
    assert lane["execution"]["search_posture"] == lane["lane_compiler"]["search_posture"]


def test_compiled_execution_view_stays_boolean_for_all_keyword_family():
    """Fix 2 byte-identical default: an all-keyword family keeps execution.acquisition_mode
    'linkedin_boolean' / posture 'boolean_led' after compile — the reconcile is a no-op
    when the compiled snapshot is Boolean.
    """
    plan = _family_plan(
        {
            "family_id": "ml_research",
            "label": "ML research engineers",
            "objective": "Plain keyword family, no structured facets.",
            "entry_signals": [
                {"item_id": "e1", "label": "ml engineer", "terms": ["ml engineer"]}
            ],
        }
    )
    lane = plan.sourcing_lanes[0]
    assert lane["lane_compiler"]["acquisition_mode"] == "linkedin_boolean"
    assert lane["execution"]["acquisition_mode"] == "linkedin_boolean", lane["execution"]
    assert lane["execution"]["search_posture"] == "boolean_led"


def test_all_keyword_family_keeps_empty_filters_and_boolean_mode():
    """(b) Byte-identical default: a family with NO structured_surface anywhere queues a
    SearchString with empty structured_filters and acquisition_mode 'linkedin_boolean'.

    This is the regression guard for the live keyword path — every existing family
    omits structured_surface, so it must keep flowing exactly as before (empty filters
    -> the not-empty trigger never fires -> linkedin_boolean). Goes red if a default
    surface ever starts emitting a structured control.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        plan = _family_plan(
            {
                "family_id": "ml_research",
                "label": "ML research engineers",
                "objective": "Plain keyword family, no structured facets.",
                "entry_signals": [
                    {"item_id": "e1", "label": "ml engineer", "terms": ["ml engineer"]}
                ],
                "capability_proxies": [
                    {"item_id": "c1", "label": "rlhf", "terms": ["rlhf", "alignment"]}
                ],
                "reality_filters": [
                    {"item_id": "r1", "label": "production", "terms": ["production"]}
                ],
            }
        )
        p._execution_plan = plan
        results = p._build_ordered_search_strings()

        assert results, "expected at least one queued string"
        for ss in results:
            assert _filters_are_empty(getattr(ss, "structured_filters", None))
            assert ss.acquisition_mode == "linkedin_boolean"
            assert ss.search_posture == "boolean_led"

        # Part 5 (telemetry): an all-keyword slice reads as boolean_led.
        snapshot = plan.sourcing_lanes[0]["lane_compiler"]
        assert snapshot["search_posture"] == "boolean_led", snapshot.get("search_posture")


def test_injected_structured_surface_collapses_to_keyword():
    """(c) Injection gate: an unknown/garbage structured_surface collapses to "" at the
    allow-list (retrieval_design.py:101, _STRUCTURED_LAYER_SURFACES), so the item stays a
    keyword and produces NO structured control — no acquisition-mode flip.

    A layer item declaring linkedin_seniority_filter (not on the allow-list) must not
    smuggle a structured facet through. Goes red if parsing widens to honor arbitrary
    surfaces. Contrast with (a): same shape, an off-list surface, no structured output.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        plan = _family_plan(
            {
                "family_id": "injected_pool",
                "label": "Injected seniority filter",
                "objective": "An off-allow-list surface must not become a control.",
                "entry_signals": [
                    {
                        "item_id": "e1",
                        "label": "Senior leaders",
                        "terms": ["senior director"],
                        "structured_surface": "linkedin_seniority_filter",
                    }
                ],
                "capability_proxies": [
                    {"item_id": "c1", "label": "ai", "terms": ["applied ai"]}
                ],
            }
        )
        p._execution_plan = plan
        ss = p._build_ordered_search_strings()[0]

        assert _filters_are_empty(getattr(ss, "structured_filters", None))
        assert ss.acquisition_mode == "linkedin_boolean"


def _temporal_mismatch_lane(*, dimension: str, lane_id: str) -> SourcingLane:
    """A SourcingLane whose slice carries one require/prefer, boolean_keyword,
    temporal_scope='current' constraint of the given dimension — the exact shape that
    trips lint_constraint_compile's temporal_scope_mismatch finding
    (boolean_compiler.py:766-781)."""
    return SourcingLane(
        lane_id=lane_id,
        lane_name=lane_id,
        hypothesis=SearchHypothesis(
            hypothesis_id="h1",
            label="H",
            target_archetype="leader",
            why_this_pool_may_exist="banks",
        ),
        slice=SearchSlice(
            slice_id="s1",
            hypothesis_id="h1",
            label="Slice",
            objective="find leaders",
            constraints=[
                SearchConstraint(
                    dimension=dimension,
                    values=["head of applied ai"]
                    if dimension in {"title", "seniority", "role", "job_title"}
                    else ["banking"],
                    operator="require",
                    execution_surface="boolean_keyword",
                    temporal_scope="current",
                )
            ],
        ),
        execution=LaneExecution(
            lane_id=lane_id,
            source="linkedin",
            acquisition_mode="linkedin_boolean",
            boolean_strategy={"root_boolean": '"head of applied ai"'},
            structured_filters={},
        ),
    )


def test_repair_flips_title_constraint_on_temporal_mismatch_and_reaches_title_control():
    """(d) repair_constraint_surfaces flips a title-like, boolean_keyword,
    current-temporal constraint carrying the temporal_scope_mismatch finding to
    linkedin_title_filter — and the flipped surface then compiles into a title control —
    while leaving a non-title constraint with the same finding untouched.

    The repair MUST mutate lane_dict["slice"]["constraints"][i]["execution_surface"] in
    place (attach_constraint_lint_to_plan builds a from_dict COPY, so a copy mutation
    would be invisible to the compiler that runs next). After repair, driving the real
    lane compiler folds the now-linkedin_title_filter constraint into a title control.
    Goes red if the repair widens (flips non-title) or narrows (misses the title flip).
    """
    from linkedin.boolean_compiler import repair_constraint_surfaces
    from linkedin.lane_compiler import LinkedInLaneCompiler

    title_lane = _temporal_mismatch_lane(dimension="title", lane_id="title-pool")
    domain_lane = _temporal_mismatch_lane(dimension="domain", lane_id="domain-pool")
    plan = ExecutionPlan(
        strategy_rationale="test",
        sourcing_lanes=[title_lane.to_dict(), domain_lane.to_dict()],
    )

    # Sanity: both constraints genuinely carry the temporal_scope_mismatch finding pre-repair
    # (so the repair's selectivity, not a missing finding, is what spares the domain one).
    for ld in plan.sourcing_lanes:
        c = ld["slice"]["constraints"][0]
        compiled = compile_constraint(SearchConstraint.from_dict(c))
        codes = {f.code for f in __import__("linkedin.boolean_compiler", fromlist=["lint_constraint_compile"]).lint_constraint_compile(SearchConstraint.from_dict(c), compiled)}
        assert "temporal_scope_mismatch" in codes

    repairs = repair_constraint_surfaces(plan)

    # The TITLE constraint flipped in place; the DOMAIN constraint did not.
    title_dict = plan.sourcing_lanes[0]["slice"]["constraints"][0]
    domain_dict = plan.sourcing_lanes[1]["slice"]["constraints"][0]
    assert title_dict["execution_surface"] == "linkedin_title_filter"
    assert domain_dict["execution_surface"] == "boolean_keyword"

    # A repair record was appended to the title lane only, in place.
    assert len(repairs) == 1, repairs
    assert plan.sourcing_lanes[0].get("constraint_repairs"), "title lane missing repair record"
    assert not plan.sourcing_lanes[1].get("constraint_repairs"), "domain lane should not be repaired"

    # The flipped surface compiles into a real title control (it reaches execution, not
    # just a different label): rebuild the lane from the mutated dict and compile.
    repaired_title_lane = SourcingLane.from_dict(plan.sourcing_lanes[0])
    exe = LinkedInLaneCompiler().compile(repaired_title_lane)
    assert exe.query_payload["structured_filters"]["titles"] == ["head of applied ai"]


# ---------------------------------------------------------------------------
# SLICE B — the OPENING search honors the reasoned surface (start-in-pool).
#
# Slice A's producer SETS the surface (acquisition_mode='linkedin_hybrid' +
# SearchString.structured_filters); slice B makes the opening HONOR it.
#
# Consumer: orchestrator._process_string opening dispatch — the resume entry
#   (orchestrator.py:~1844) and the fresh entry (~:1856). bootstrap_experiment_state
#   seeds the root variant from the checkpointed SearchString.structured_filters
#   (slice B part 1), and _apply_opening_search routes a hybrid lane with non-empty
#   active filters through browser.apply_advanced_search_plan (the applied-only
#   snapshot path BELOW apply_variant's budget layer) instead of the bare
#   enter_search_string.
#
# Load-bearing assertion (part 2): the opening is NOT a mutation — no
#   mutation/consecutive-rewrite budget is consumed and no mutation event fires.
# Location caution (part 4): the opening plan carries ONLY the lane's own
#   structured locations, never the brief's session geography.
# ---------------------------------------------------------------------------


def _hybrid_location_search_string(
    p, *, locations: list[str], boolean: str = '"VP" AND engineering'
):
    """Build a hybrid SearchString carrying a sidebar location filter through the
    REAL lane compiler + queue builder (the slice-A producer path)."""
    lane = _location_lane(locations=locations)
    plan = ExecutionPlan(
        strategy_rationale="test",
        generated_strings=[
            {"boolean": boolean, "rationale": "r", "lane_id": "geo-pool"}
        ],
        sourcing_lanes=[lane.to_dict()],
    )
    apply_linkedin_lane_compiler_to_plan(plan)
    p._execution_plan = plan
    return p._build_ordered_search_strings()[0]


def _stage_browser_for_opening(p, *, result_count: int = 0):
    """Stub the browser surface _process_string touches up to the opening apply,
    then bail at the result_count<=0 guard. enter_search_string and
    apply_advanced_search_plan are the two dispatch targets the opening chooses
    between; both are spies."""
    p.browser.page = MagicMock(url="https://www.linkedin.com/talent/hire/test/discover/recruiterSearch")
    p.browser.check_and_recover = AsyncMock(return_value=False)
    p.browser.go_back_to_results = AsyncMock()
    p.browser.navigate_to_search = AsyncMock()
    p.browser.enter_search_string = AsyncMock()
    p.browser.apply_advanced_search_plan = AsyncMock()
    p.browser.apply_location_filter = AsyncMock(return_value=True)
    p.browser.get_results_count_text = AsyncMock(return_value=str(result_count))
    p.browser.get_results_count = AsyncMock(return_value=result_count)


def _new_progress(p, search_string):
    progress = Progress(brief_name="test", strings=[search_string])
    progress.save(str(p.progress_path))
    p._progress = progress
    return progress


def _stage_browser_for_review_loop(p, *, result_count: int = 80):
    _stage_browser_for_opening(p, result_count=result_count)
    no_results = MagicMock()
    no_results.is_visible = AsyncMock(return_value=False)
    locator = MagicMock()
    locator.first = no_results
    p.browser.page.locator.return_value = locator
    p.browser.go_to_next_page = AsyncMock(return_value=False)
    p._ensure_browser_healthy = AsyncMock()
    p._review_page_sequentially = AsyncMock(return_value=None)
    p._assess_string_state = AsyncMock(
        return_value={
            "decision": "continue",
            "rationale": "keep paginating",
            "page_signal": 0,
            "committed_zero_signal_streak": 0,
        }
    )


def test_filter_led_opening_applies_structured_plan_without_spending_budget():
    """(b) A hybrid lane with structured filters opens via apply_advanced_search_plan
    with the compiled plan — and the opening is NOT a mutation: the mutation/rewrite
    budget counter is not decremented and no mutation event is emitted.

    This is the load-bearing slice-B assertion. The opening goes through
    browser.apply_advanced_search_plan (applies keywords + controls + snapshot in one
    call, below apply_variant's budget layer), so consecutive_mutations / mutations_used
    / _search_mutation_budget_used all stay 0 and no linkedin_search_mutation_* event
    fires. Goes red if the opening is ever routed through apply_variant.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        boolean = '"VP" AND engineering NOT ("recruiter" OR "sourcer")'
        ss = _hybrid_location_search_string(
            p, locations=["New York City"], boolean=boolean
        )
        assert ss.acquisition_mode == "linkedin_hybrid"  # producer set the surface
        _stage_browser_for_opening(p)
        progress = _new_progress(p, ss)

        asyncio.run(p._process_string(ss, progress))

        # The filter-led path was taken: the compiled plan reached the browser.
        p.browser.apply_advanced_search_plan.assert_awaited_once()
        p.browser.enter_search_string.assert_not_awaited()
        (plan_arg,), _ = p.browser.apply_advanced_search_plan.await_args
        dims = [c.dimension for c in plan_arg.controls]
        assert "locations" in dims
        assert plan_arg.acquisition_mode == "linkedin_hybrid"
        keyword_ctrl = [c for c in plan_arg.controls if c.dimension == "keywords"]
        assert keyword_ctrl and keyword_ctrl[0].values == [boolean]

        # The opening did NOT spend the rewrite budget — it is not a mutation.
        state = p._experiment_states[ss.id]
        assert state.consecutive_mutations == 0
        assert state.mutations_used == 0
        assert p._search_mutation_budget_used == 0
        events = read_jsonl(p.log_path)
        mutation_events = [e for e in events if str(e.get("event", "")).startswith("linkedin_search_mutation")]
        assert mutation_events == [], mutation_events
        executed_events = [e for e in events if e.get("event") == "string_executed"]
        assert len(executed_events) == 1
        assert executed_events[0]["executed_boolean"] == boolean
        assert " NOT " in executed_events[0]["executed_boolean"]
        assert executed_events[0]["execution_surface"] == "advanced"


def test_boolean_led_opening_is_byte_identical_keyword_entry():
    """(c) A boolean lane (empty filters / non-hybrid) opens via enter_search_string
    only — apply_advanced_search_plan is never called. The keyword path is unchanged.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        boolean = '"VP" AND engineering NOT ("recruiter" OR "sourcer")'
        ss = SearchString(
            id=1, name="builders", boolean=boolean, status="queued", block="Block A"
        )
        _stage_browser_for_opening(p)
        progress = _new_progress(p, ss)

        asyncio.run(p._process_string(ss, progress))

        p.browser.enter_search_string.assert_awaited_once_with(boolean)
        p.browser.apply_advanced_search_plan.assert_not_awaited()
        events = read_jsonl(p.log_path)
        executed_events = [e for e in events if e.get("event") == "string_executed"]
        assert len(executed_events) == 1
        assert executed_events[0]["executed_boolean"] == boolean
        assert " NOT " in executed_events[0]["executed_boolean"]
        assert executed_events[0]["execution_surface"] == "keyword"


def test_process_string_retries_pagination_false_when_pages_remain_then_marks_transient():
    from linkedin.orchestrator import TransientPaginationError

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = SearchString(id=1, name="builders", boolean='"engineer"', status="queued")
        _stage_browser_for_review_loop(p, result_count=60)
        p.browser.go_to_next_page = AsyncMock(side_effect=[False, False])
        progress = _new_progress(p, ss)

        with (
            patch(
                "linkedin.orchestrator.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep_mock,
            pytest.raises(TransientPaginationError),
        ):
            asyncio.run(p._process_string(ss, progress))

        events = read_jsonl(p.log_path)
        exhausted = [
            event for event in events if event.get("event") == "pagination_exhausted"
        ]
        assert exhausted == []
        assert p.browser.go_to_next_page.await_count == 2
        sleep_mock.assert_awaited_once_with(3)
        assert ss.status == "in_progress"
        assert p._experiment_states[ss.id].active_allocator_page_cursor() == 2
        assert "transient pagination suspected" in (ss.notes or "")


def test_process_string_does_not_retry_pagination_false_at_genuine_end():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = SearchString(id=1, name="builders", boolean='"engineer"', status="queued")
        _stage_browser_for_review_loop(p, result_count=25)
        progress = _new_progress(p, ss)

        with patch("linkedin.orchestrator.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            asyncio.run(p._process_string(ss, progress))

        events = read_jsonl(p.log_path)
        exhausted = [
            event for event in events if event.get("event") == "pagination_exhausted"
        ]
        assert len(exhausted) == 1
        assert exhausted[0]["string_id"] == 1
        assert exhausted[0]["page_num"] == 1
        assert exhausted[0]["result_count"] == 25
        assert exhausted[0]["pages_remaining_by_math"] is False
        assert exhausted[0]["transient_suspected"] is False
        p.browser.go_to_next_page.assert_awaited_once()
        sleep_mock.assert_not_awaited()


def test_process_string_logs_page_cap_reached_at_max_pages_break():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = SearchString(id=1, name="builders", boolean='"engineer"', status="queued")
        _stage_browser_for_review_loop(p, result_count=60)
        progress = _new_progress(p, ss)

        with patch("linkedin.orchestrator.config.MAX_PAGES_PER_STRING", 1):
            asyncio.run(p._process_string(ss, progress))

        events = read_jsonl(p.log_path)
        cap_events = [event for event in events if event.get("event") == "page_cap_reached"]
        assert len(cap_events) == 1
        assert cap_events[0]["string_id"] == 1
        assert cap_events[0]["page_num"] == 1
        assert cap_events[0]["result_count"] == 60
        assert cap_events[0]["max_pages"] == 1
        p.browser.go_to_next_page.assert_not_awaited()


def test_process_string_logs_resume_fastforward_exhausted():
    from linkedin.orchestrator import TransientPaginationError

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = SearchString(
            id=1,
            name="builders",
            boolean='"engineer"',
            status="queued",
            pages_reviewed=3,
        )
        _stage_browser_for_opening(p, result_count=100)
        p.browser.go_to_next_page = AsyncMock(side_effect=[False, False])
        progress = _new_progress(p, ss)

        with (
            patch(
                "linkedin.orchestrator.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep_mock,
            pytest.raises(TransientPaginationError),
        ):
            asyncio.run(p._process_string(ss, progress))

        events = read_jsonl(p.log_path)
        exhausted = [
            event
            for event in events
            if event.get("event") == "resume_fastforward_exhausted"
        ]
        assert len(exhausted) == 1
        assert exhausted[0]["string_id"] == 1
        assert exhausted[0]["page_num"] == 1
        assert exhausted[0]["target_page"] == 3
        assert exhausted[0]["result_count"] == 100
        assert exhausted[0]["pages_remaining_by_math"] is True
        assert exhausted[0]["transient_suspected"] is True
        assert p.browser.go_to_next_page.await_count == 2
        sleep_mock.assert_awaited_once_with(3)
        assert ss.status == "in_progress"
        assert ss.pages_reviewed == 3
        assert p._experiment_states[ss.id].active_allocator_page_cursor() == 0
        assert "transient resume fast-forward suspected" in (ss.notes or "")


def test_force_narrow_failed_events_cover_no_boolean_and_exception():
    stats = {
        "pages": 1,
        "candidates": 0,
        "saves": 0,
        "facial_yes": 0,
        "facial_no": 0,
    }
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = SearchString(id=1, name="builders", boolean='"engineer"', status="queued")

        with patch(
            "shared.llm_clients.opus_llm_cached",
            return_value={"action": "narrow", "rationale": "no usable boolean"},
        ):
            result = asyncio.run(
                p._force_narrow_adapt(ss, ss.boolean, "100", [], stats)
            )
        assert result is None

        with patch(
            "shared.llm_clients.opus_llm_cached",
            side_effect=RuntimeError("model unavailable"),
        ):
            result = asyncio.run(
                p._force_narrow_adapt(ss, ss.boolean, "100", [], stats)
            )
        assert result is None

        failed = [
            event
            for event in read_jsonl(p.log_path)
            if event.get("event") == "forced_narrow_failed"
        ]
        assert [event["reason"] for event in failed] == ["no_boolean", "exception"]
        assert failed[0]["string_id"] == 1
        assert failed[1]["error"] == "model unavailable"


def test_hybrid_mode_with_empty_filters_stays_keyword_only():
    """(guard) A lane declared linkedin_hybrid but carrying NO structured filters must
    still open via enter_search_string only. hybrid+empty is constructible — the lane
    compiler force-SETS hybrid when filters exist (lane_compiler.py:141) but never CLEARS
    a pre-declared hybrid mode when they are empty. Pins that the opening guard's SECOND
    conjunct (not active.structured_filters.is_empty()), not just the mode check, gates
    the structured path: dropping it to a bare `== "linkedin_hybrid"` goes red here.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = SearchString(
            id=1,
            name="hybrid-empty",
            boolean='"VP" AND engineering',
            status="queued",
            block="Block A",
            acquisition_mode="linkedin_hybrid",
        )
        assert ss.acquisition_mode == "linkedin_hybrid"
        assert not ss.structured_filters  # hybrid declared, but no filters carried
        _stage_browser_for_opening(p)
        progress = _new_progress(p, ss)

        asyncio.run(p._process_string(ss, progress))

        p.browser.enter_search_string.assert_awaited_once_with('"VP" AND engineering')
        p.browser.apply_advanced_search_plan.assert_not_awaited()


def test_resume_reapplies_persisted_structured_filters_via_structured_path():
    """(d) Crash -> cross-process resume on a hybrid lane re-applies the persisted
    filters via the structured path, not keyword-only.

    A resumed string (pages_reviewed > 0) with NO in-memory experiment state forces
    bootstrap_experiment_state, which re-seeds the root variant from the checkpointed
    SearchString.structured_filters (slice B part 1). The resume branch then drives
    _apply_opening_search off that seeded active variant, so the structured plan is
    re-applied rather than dropping to a bare keyword re-entry. Goes red if resume
    stops reconstructing the structured search.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = _hybrid_location_search_string(p, locations=["New York City"])
        ss.pages_reviewed = 3  # interrupted mid-string -> resume branch
        _stage_browser_for_opening(p)
        progress = _new_progress(p, ss)
        assert ss.id not in p._experiment_states  # cross-process: state not yet hydrated

        asyncio.run(p._process_string(ss, progress))

        # Resume re-applied the structured plan, not a keyword-only re-entry.
        p.browser.apply_advanced_search_plan.assert_awaited_once()
        p.browser.enter_search_string.assert_not_awaited()
        (plan_arg,), _ = p.browser.apply_advanced_search_plan.await_args
        location_ctrl = [c for c in plan_arg.controls if c.dimension == "locations"]
        assert location_ctrl and location_ctrl[0].values == ["New York City"]


def test_resume_reapplies_filters_when_lane_committed_a_refinement_before_crash():
    """(d2) The MUTATED-lane resume case: a hybrid lane that committed a precision
    refinement BEFORE the crash still re-applies its structured filters on
    cross-process resume — not a bare keyword re-entry that loses geography.

    Test (d) only exercises the pristine lane: its checkpoint has refinement_stack==[]
    so bootstrap keeps active_variant_id=='root' and the root seed carries the location.
    A mid-run lane is different: apply_shadow checkpoints a NON-EMPTY refinement_stack
    (search_intelligence.py:527 writes refinement_stack = compat_refinement_stack()),
    which on resume drives bootstrap_experiment_state down the legacy-chain branch and
    points active_variant_id at the LAST legacy variant, not 'root'. Before the fix the
    legacy variants were minted with empty structured_filters, so _apply_opening_search
    read the active variant's empty filters and dropped to enter_search_string, losing
    the lane's New York City geography. bootstrap now seeds every legacy variant whose
    own filters are empty, so the active variant on resume reconstructs the structured
    search. Goes red if bootstrap stops seeding the active (non-root) variant.
    """
    from linkedin.search_intelligence import bootstrap_experiment_state

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = _hybrid_location_search_string(p, locations=["New York City"])

        # Faithfully reproduce the checkpoint a real mid-run lane writes: bootstrap the
        # pristine state, commit a precision refinement, then apply_shadow it back onto
        # the SearchString exactly as the orchestrator does — producing a non-empty
        # refinement_stack via the real runtime rather than a fabricated field.
        live = bootstrap_experiment_state(ss)
        refined = LinkedInSearchVariant(
            variant_id="precision-1",
            parent_variant_id="root",
            root_string_id=ss.id,
            boolean='"VP" AND engineering AND platform',
            variant_kind="precision",
        )
        live.variants[refined.variant_id] = refined
        live.commit_variant(refined.variant_id)
        live.apply_shadow(ss)
        assert ss.refinement_stack, "mid-run lane must checkpoint a non-empty refinement_stack"

        ss.pages_reviewed = 3  # interrupted mid-string -> resume branch
        _stage_browser_for_opening(p)
        progress = _new_progress(p, ss)
        assert ss.id not in p._experiment_states  # cross-process: state not yet hydrated

        # The bootstrap that resume will perform points active at the legacy variant,
        # NOT root — the precondition that broke the old code path.
        booted = bootstrap_experiment_state(ss)
        assert booted.active_variant_id != "root"

        asyncio.run(p._process_string(ss, progress))

        # Resume re-applied the structured plan off the active legacy variant, not a
        # keyword-only re-entry: the lane's geography survived the crash.
        p.browser.apply_advanced_search_plan.assert_awaited_once()
        p.browser.enter_search_string.assert_not_awaited()
        (plan_arg,), _ = p.browser.apply_advanced_search_plan.await_args
        location_ctrl = [c for c in plan_arg.controls if c.dimension == "locations"]
        assert location_ctrl and location_ctrl[0].values == ["New York City"]


def test_opening_plan_carries_lane_locations_not_brief_session_geography():
    """(e) Location caution: the opening plan carries the LANE's own structured
    locations, never the brief's session geography — the two paths never double-apply
    at apply_location_filter.

    The lane's slice location is 'New York City'; the brief's session geography is a
    DIFFERENT place ('Brazil'). The opening structured plan must carry only the lane's
    location; the brief geography rides _apply_session_location_filter (a separate,
    idempotent direct apply off permanent_filters), never the opening plan. Goes red if
    session geography is ever injected into the opening plan.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = _hybrid_location_search_string(p, locations=["New York City"])
        p.brief_obj.permanent_filters = {"Location": "Brazil"}  # session geo: different place
        _stage_browser_for_opening(p)
        # P3a invariant staging: the session geography was applied at Build-1 and
        # its chip is live on the sidebar — this test is about the OPENING PLAN's
        # contents, not the gate (the gate has its own tests).
        p._session_location_applied = True
        p.browser.read_applied_location_chips = AsyncMock(return_value=["Brazil"])
        progress = _new_progress(p, ss)

        asyncio.run(p._process_string(ss, progress))

        (plan_arg,), _ = p.browser.apply_advanced_search_plan.await_args
        location_values = [v for c in plan_arg.controls if c.dimension == "locations" for v in c.values]
        assert location_values == ["New York City"], location_values
        assert "Brazil" not in location_values  # session geography never entered the plan


def _seed_structured_only_active(p, ss, *, boolean: str, locations: list[str]):
    """Pre-seed the experiment state so the ACTIVE variant is structured_only with a
    NON-EMPTY boolean and a location filter.

    The accidental-safety boundary (advanced_search.py:268 'if keyword_boolean and
    include_keyword') means an EMPTY-boolean structured_only variant is safe whether
    or not the gate fires — so the regression MUST carry a non-empty boolean to
    actually exercise the leak. _process_string / _capture_recovery_snapshot reuse a
    pre-seeded _experiment_states entry (orchestrator.py:653), so this controls the
    surface the opening and recovery paths compile against.
    """
    state = LinkedInExperimentState(
        root_string_id=ss.id,
        intent=LinkedInSearchIntent(root_boolean=boolean),
        mode="experiment",
    )
    active = LinkedInSearchVariant(
        variant_id="structured-only-active",
        parent_variant_id="root",
        root_string_id=ss.id,
        boolean=boolean,
        surface="structured_only",
        variant_kind="structured_filter",
        structured_filters=LinkedInStructuredFilters(
            sidebar_filters={"locations": list(locations)}
        ),
    )
    state.variants[active.variant_id] = active
    state.active_variant_id = active.variant_id
    state.committed_variant_id = active.variant_id
    p._experiment_states[ss.id] = state
    return state


def test_structured_only_active_opening_does_not_enter_keyword():
    """(D) The OPENING apply (orchestrator.py:~1846, _apply_opening_search) is the
    THIRD compile call site that can carry a structured_only active variant. A
    structured_only variant with a non-empty boolean must open with the keyword
    SUPPRESSED — no keywords control reaches apply_advanced_search_plan and the
    plan's keyword_boolean is empty — not entered as the keyword search the filters
    are defined to carry instead. Goes red if the include_keyword gate is dropped
    from the opening compile (it defaults True, advanced_search.py:255).
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = _hybrid_location_search_string(p, locations=["New York City"])
        _seed_structured_only_active(
            p, ss, boolean='"ML" AND "engineer"', locations=["New York City"]
        )
        _stage_browser_for_opening(p)
        progress = _new_progress(p, ss)

        asyncio.run(p._process_string(ss, progress))

        # The structured path ran (locations carried it), but the keyword did NOT.
        p.browser.apply_advanced_search_plan.assert_awaited_once()
        p.browser.enter_search_string.assert_not_awaited()
        (plan_arg,), _ = p.browser.apply_advanced_search_plan.await_args
        dims = [c.dimension for c in plan_arg.controls]
        assert "locations" in dims
        assert "keywords" not in dims, dims
        assert plan_arg.keyword_boolean == "", plan_arg.keyword_boolean


def test_structured_only_active_opening_with_empty_boolean_also_safe():
    """(D guard) The accidental-safety case: a structured_only variant with an EMPTY
    boolean is already safe (the 'if keyword_boolean' half of the compile guard fails
    on the empty string regardless of include_keyword). Pins that the gate does not
    REGRESS that case — still no keyword control, still empty keyword_boolean.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = _hybrid_location_search_string(p, locations=["New York City"])
        _seed_structured_only_active(p, ss, boolean="", locations=["New York City"])
        _stage_browser_for_opening(p)
        progress = _new_progress(p, ss)

        asyncio.run(p._process_string(ss, progress))

        p.browser.apply_advanced_search_plan.assert_awaited_once()
        (plan_arg,), _ = p.browser.apply_advanced_search_plan.await_args
        dims = [c.dimension for c in plan_arg.controls]
        assert "keywords" not in dims, dims
        assert plan_arg.keyword_boolean == ""


def test_recovery_snapshot_for_structured_only_active_does_not_replay_keyword():
    """(D) The recovery-snapshot compile (orchestrator.py:~710,
    _capture_recovery_snapshot) is the SECOND structured_only compile call site. Its
    include_keyword-zeroed plan.keyword_boolean must reach the snapshot's TOP-LEVEL
    keyword_boolean field — the field compile_recovery_plan_from_snapshot
    (advanced_search.py:~304) actually reads to re-add a keyword on replay — not be
    overwritten by search_string.boolean.

    Before the fix the gate zeroed only the (discarded) advanced_controls copy while
    the snapshot field was sourced unconditionally from search_string.boolean, so a
    browser crash mid-pagination would replay the keyword a structured_only surface
    is defined to suppress. Builds the snapshot the way the LIVE orchestrator does
    (search_string carries the boolean; the active variant is structured_only) and
    asserts the replay does NOT re-add the keyword.
    """
    from linkedin.advanced_search import compile_recovery_plan_from_snapshot

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = _hybrid_location_search_string(p, locations=["New York City"])
        # The live orchestrator keeps SearchString.boolean populated even for a
        # structured_only variant (apply_shadow copies current_boolean onto it);
        # that is precisely the value that used to leak into the snapshot field.
        ss.boolean = '"ML" AND "engineer"'
        _seed_structured_only_active(
            p, ss, boolean='"ML" AND "engineer"', locations=["New York City"]
        )
        # No live controls captured -> the compile branch (orchestrator.py:704) fires.
        p.browser.snapshot_advanced_search_controls = AsyncMock(return_value={})
        p.browser.get_current_search_url = MagicMock(return_value="")

        snapshot = asyncio.run(p._capture_recovery_snapshot(ss, page_num=2))

        # The snapshot field the replay guard reads was zeroed by the gated plan,
        # NOT left as search_string.boolean.
        assert snapshot.keyword_boolean == "", snapshot.keyword_boolean
        recovered = compile_recovery_plan_from_snapshot(snapshot)
        assert all(c.dimension != "keywords" for c in recovered.controls), [
            c.dimension for c in recovered.controls
        ]
        assert recovered.keyword_boolean == ""
        # The structured control still survives the round-trip — only the keyword dropped.
        assert "locations" in [c.dimension for c in recovered.controls]


def test_recovery_snapshot_for_keyword_led_active_still_replays_keyword():
    """(D regression) The contrast case proving the gate is surface-scoped, not a
    blanket zero: a keyword-led (default surface) active variant's recovery snapshot
    still carries the boolean into the top-level field, so the replay re-adds the
    keyword exactly as before. Goes red if the fix zeroes the keyword unconditionally.
    """
    from linkedin.advanced_search import compile_recovery_plan_from_snapshot

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = _hybrid_location_search_string(p, locations=["New York City"])
        ss.boolean = '"ML" AND "engineer"'
        state = _seed_structured_only_active(
            p, ss, boolean='"ML" AND "engineer"', locations=["New York City"]
        )
        # Flip the active variant back to keyword-led (default surface).
        state.active_variant.surface = ""
        p.browser.snapshot_advanced_search_controls = AsyncMock(return_value={})
        p.browser.get_current_search_url = MagicMock(return_value="")

        snapshot = asyncio.run(p._capture_recovery_snapshot(ss, page_num=2))

        assert snapshot.keyword_boolean == '"ML" AND "engineer"'
        recovered = compile_recovery_plan_from_snapshot(snapshot)
        assert any(c.dimension == "keywords" for c in recovered.controls)
        assert recovered.keyword_boolean == '"ML" AND "engineer"'


def test_structured_only_recovery_after_worker_death_resume_suppresses_keyword_end_to_end():
    """SLICE G part 4 (end-to-end close): the ORIGINAL item-a bug — a resumed
    structured_only variant re-adding the keyword on the crash-recovery replay — is
    closed by surface durability (slice G) + the slice-D snapshot zero TOGETHER, with
    the surface arriving via bootstrap_experiment_state (the worker-death path), NOT a
    pre-seeded in-memory state.

    The other recovery tests above pre-seed _experiment_states so the surface is in
    memory. This one reproduces a true cross-process resume: the producer persists a
    structured_only variant onto the compat SearchString via apply_shadow, the worker
    dies (in-memory state gone), and the resume hydrates _experiment_states purely from
    the SearchString through bootstrap_experiment_state. WITHOUT slice G the bootstrap
    mints surface="" and _capture_recovery_snapshot's include_keyword gate would NOT
    fire — the keyword would leak back into the replay. WITH slice G the surface is
    durable, so the snapshot's keyword_boolean is zeroed and the replay re-adds no
    keyword. Goes red if SearchString.surface stops persisting or bootstrap stops
    reconstructing it.
    """
    from linkedin.advanced_search import compile_recovery_plan_from_snapshot
    from linkedin.search_intelligence import bootstrap_experiment_state

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = _hybrid_location_search_string(p, locations=["New York City"])
        ss.boolean = '"ML" AND "engineer"'

        # Producer side: a structured_only active variant, persisted onto the compat
        # SearchString via the REAL apply_shadow (surface + filters land on ss).
        producer_state = _seed_structured_only_active(
            p, ss, boolean='"ML" AND "engineer"', locations=["New York City"]
        )
        producer_state.apply_shadow(ss)
        assert ss.surface == "structured_only"

        # Worker death: serialize the compat record and drop ALL in-memory state.
        resumed_string = SearchString.from_dict(ss.to_dict())
        p._experiment_states = {}

        # Resume hydrates the experiment state purely from the SearchString (the
        # bootstrap path orchestrator._experiment_state_for takes when the in-memory
        # entry is absent), reconstructing surface=structured_only.
        resumed_state = bootstrap_experiment_state(resumed_string)
        assert resumed_state.active_variant.surface == "structured_only"
        p._experiment_states[resumed_string.id] = resumed_state

        p.browser.snapshot_advanced_search_controls = AsyncMock(return_value={})
        p.browser.get_current_search_url = MagicMock(return_value="")

        snapshot = asyncio.run(p._capture_recovery_snapshot(resumed_string, page_num=2))

        # The replay guard reads keyword_boolean; the durable surface zeroed it.
        assert snapshot.keyword_boolean == "", snapshot.keyword_boolean
        recovered = compile_recovery_plan_from_snapshot(snapshot)
        assert all(c.dimension != "keywords" for c in recovered.controls), [
            c.dimension for c in recovered.controls
        ]
        assert recovered.keyword_boolean == ""
        # The structured control survives the resume + recovery round-trip.
        assert "locations" in [c.dimension for c in recovered.controls]


# ---------------------------------------------------------------------------
# Slice C (the heart) — adaptive structured proposals MID-RUN.
#
# The variant/drift planners (orchestrator._plan_variant_experiments,
# _plan_drift_refinement) used to FORBID structured filters: their system prompts
# said "Return ONLY keyword Boolean variants" / "Do not use structured filters",
# and the parse built keyword-only LinkedInSearchVariant (empty filters). Slice C
# extends the live variant-proposal + drift LLM contracts so Cloris can, from
# observed page results, PROMOTE a clean literal title/company cluster to a
# structured filter, DEMOTE a failing dimension back to keyword, or go
# structured_only mid-run — while keeping the precision/recall/noise_exclusion
# keyword path fully intact (structured is ADDITIVE).
#
# Only the external LLM boundary (shared.llm_clients.opus_llm_cached, imported
# inside each planner) is mocked; the real prompt build, parse, filter
# inheritance (_copy_filters/_drop_one_filter), surface marker, and seeding gate
# all run. Patterns mirror the variant/experiment seam tests above and the
# opus_llm_cached patch idiom in tests/test_seam_judgment.py.
# ---------------------------------------------------------------------------


def _page_insights_with_title_cluster(label: str, *, signal: bool = True):
    """A LinkedInPageInsights carrying one tight literal title cluster — the
    evidence a promote reasons from (built ~orchestrator.py:2956-3010)."""
    from linkedin.search_intelligence import LinkedInPageInsights

    return LinkedInPageInsights(
        page=1,
        result_count=600,
        result_window="too_broad",
        title_clusters=[
            {"label": label, "title": label, "signal_count": 4 if signal else 0}
        ],
        company_clusters=[],
        signal_anchors=[label] if signal else [],
    )


def _page_insights_with_company_cluster(label: str, *, signal: bool = True):
    """A LinkedInPageInsights carrying one tight literal company cluster."""
    from linkedin.search_intelligence import LinkedInPageInsights

    return LinkedInPageInsights(
        page=1,
        result_count=600,
        result_window="too_broad",
        title_clusters=[],
        company_clusters=[
            {"label": label, "company": label, "signal_count": 4 if signal else 0}
        ],
        signal_anchors=[label] if signal else [],
    )


def _experiment_state_with_active(
    *,
    root_string_id: int,
    root_boolean: str,
    active_boolean: str | None = None,
    active_filters: dict | None = None,
):
    """An experiment state whose active variant is a committed refinement (so the
    planner inherits the active variant's structured_filters, not just root's)."""
    state = LinkedInExperimentState(
        root_string_id=root_string_id,
        intent=LinkedInSearchIntent(root_boolean=root_boolean),
    )
    if active_boolean is not None:
        active = LinkedInSearchVariant(
            variant_id="active-1",
            parent_variant_id="root",
            root_string_id=root_string_id,
            boolean=active_boolean,
            variant_kind="precision",
            structured_filters=LinkedInStructuredFilters.from_dict(active_filters or {}),
        )
        state.variants[active.variant_id] = active
        state.active_variant_id = active.variant_id
        state.committed_variant_id = active.variant_id
    return state


def test_slice_c_promote_yields_structured_filter_variant_with_surface():
    """(a) PROMOTE: a tight literal title-cluster page-insight lets
    _plan_variant_experiments yield a proposable structured_filter variant whose
    structured_filters.titles carries the title AND whose surface is set.

    The model (mocked) proposes one structured promote alongside the usual keyword
    variants. The real parse must turn structured_controls.titles into
    variant.structured_filters.titles, set variant.surface to a hybrid/structured
    surface, and mark variant_kind='structured_filter'. Goes red if the parse drops
    structured_controls or the prompt re-forbids structured proposals.
    """
    from linkedin.search_intelligence import LinkedInPageInsights

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=7, root_boolean='"VP" AND engineering'
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        proposal = {
            "variants": [
                {
                    "variant_id": "promote-1",
                    "variant_kind": "structured_filter",
                    "hypothesis": "promote the dominant clean title cluster to a filter",
                    "boolean": '"VP" AND engineering',
                    "surface": "hybrid",
                    "structured_controls": {"titles": ["Staff Software Engineer"]},
                    "target_result_min": 75,
                    "target_result_max": 400,
                }
            ]
        }

        planner = MagicMock(return_value=proposal)
        with patch("shared.llm_clients.opus_llm_cached", planner):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=7),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=600,
                    result_count_text="600 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        assert planner.call_args.kwargs["max_tokens"] == 16384
        assert variants, "promote proposal must yield a proposable variant"
        promoted = next(
            (v for v in variants if not v.structured_filters.is_empty()), None
        )
        assert promoted is not None, "a structured_filter variant must be proposable"
        assert promoted.structured_filters.titles == ["Staff Software Engineer"]
        assert promoted.surface in {"hybrid", "structured_only"}
        assert promoted.variant_kind == "structured_filter"


def test_slice_c_promote_rejects_title_without_signal_threshold():
    """A title cluster can graduate to a filter only after thresholded rail signal.

    The model can still propose the move, but the deterministic parser must strip it
    when the cluster has no signal_count. With an unchanged Boolean that leaves a
    no-op, so no variant should be scheduled.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=32, root_boolean='"VP" AND engineering'
        )
        page_insights = _page_insights_with_title_cluster(
            "Staff Software Engineer", signal=False
        )

        proposal = {
            "variants": [
                {
                    "variant_id": "weak-title-promote",
                    "variant_kind": "structured_filter",
                    "hypothesis": "promote a raw title cluster without signal",
                    "boolean": '"VP" AND engineering',
                    "surface": "hybrid",
                    "structured_controls": {"titles": ["Staff Software Engineer"]},
                    "target_result_min": 75,
                    "target_result_max": 400,
                }
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=32),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=120,
                    result_count_text="120 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        assert variants == []


def test_slice_c_promote_rejects_company_without_signal_threshold():
    """Company clusters follow the same thresholded graduation rule as titles."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=33, root_boolean='"VP" AND engineering'
        )
        page_insights = _page_insights_with_company_cluster("Acme AI", signal=False)

        proposal = {
            "variants": [
                {
                    "variant_id": "weak-company-promote",
                    "variant_kind": "structured_filter",
                    "hypothesis": "promote a raw company cluster without signal",
                    "boolean": '"VP" AND engineering',
                    "surface": "hybrid",
                    "structured_controls": {"companies": ["Acme AI"]},
                    "target_result_min": 75,
                    "target_result_max": 400,
                }
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=33),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=120,
                    result_count_text="120 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        assert variants == []


def test_build_page_insights_marks_title_company_signal_counts_from_results_rail():
    """Saved/facial-yes result-rail candidates should graduate cluster signal."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        snippets = [
            CandidateSnippet(
                name="Ada",
                headline="Staff Software Engineer at Acme AI",
                current_title="Staff Software Engineer",
                current_company="Acme AI",
                location="New York",
                education_snippet="",
                profile_url="/talent/profile/ada",
                source_string_id=32,
                source_string_name="test",
                page=1,
                result_rank=1,
            ),
            CandidateSnippet(
                name="Grace",
                headline="Staff Software Engineer at Acme AI",
                current_title="Staff Software Engineer",
                current_company="Acme AI",
                location="New York",
                education_snippet="",
                profile_url="/talent/profile/grace",
                source_string_id=32,
                source_string_name="test",
                page=1,
                result_rank=2,
            ),
        ]
        insights = p._build_page_insights(
            page_num=1,
            result_count=600,
            preview_snippets=snippets,
            all_candidates=[
                {
                    "page": 1,
                    "title": "Staff Software Engineer",
                    "company": "Acme AI",
                    "outcome": "save",
                },
                {
                    "page": 1,
                    "title": "Staff Software Engineer",
                    "company": "Acme AI",
                    "outcome": "facial_no",
                },
            ],
            glance_result=None,
        )

        title_cluster = insights.title_clusters[0]
        company_cluster = insights.company_clusters[0]
        assert title_cluster["label"] == "software engineer"
        assert title_cluster["signal_count"] == 1
        assert company_cluster["label"] == "Acme AI"
        assert company_cluster["signal_count"] == 1


def test_slice_c_promote_keeps_keyword_variants_intact():
    """(a, additive) PROMOTE is ADDITIVE: a mixed proposal (one keyword
    precision/recall/noise_exclusion variant + one structured promote) yields BOTH,
    and the keyword one stays surface='boolean' with empty filters.

    Goes red if the rewrite makes structured a REPLACEMENT for the keyword path
    instead of an addition.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=8, root_boolean='"VP" AND engineering'
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        proposal = {
            "variants": [
                {
                    "variant_kind": "precision",
                    "hypothesis": "tighten",
                    "boolean": '"VP" AND engineering AND platform',
                    "surface": "boolean",
                    "structured_controls": {},
                    "target_result_min": 75,
                    "target_result_max": 400,
                },
                {
                    "variant_kind": "structured_filter",
                    "hypothesis": "promote",
                    "boolean": '"VP" AND engineering',
                    "surface": "hybrid",
                    "structured_controls": {"titles": ["Staff Software Engineer"]},
                    "target_result_min": 75,
                    "target_result_max": 400,
                },
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=8),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=600,
                    result_count_text="600 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        keyword = [v for v in variants if v.structured_filters.is_empty()]
        structured = [v for v in variants if not v.structured_filters.is_empty()]
        assert keyword, "the keyword precision variant must survive"
        assert structured, "the structured promote must survive alongside it"
        assert keyword[0].boolean == '"VP" AND engineering AND platform'
        assert keyword[0].surface == "boolean"
        assert keyword[0].variant_kind == "precision"


def test_slice_c_demote_drops_inherited_dimension():
    """(b) DEMOTE: a demote proposal inherits the active variant's filters minus the
    demoted dimension. A title+company active variant that demotes 'titles' yields a
    variant whose structured_filters keeps companies, drops titles.

    Goes red if the parse stops inheriting from the active variant or stops applying
    the demote via _drop_one_filter semantics.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=9,
            root_boolean='"VP"',
            active_boolean='"VP" AND engineering',
            active_filters={
                "titles": ["Staff Software Engineer"],
                "companies": ["Stripe"],
            },
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer", signal=False)

        proposal = {
            "variants": [
                {
                    "variant_kind": "structured_filter",
                    "hypothesis": "the title filter is over-narrowing; demote it",
                    "boolean": '"VP" AND engineering',
                    "surface": "hybrid",
                    "structured_controls": {"demote": ["titles"]},
                    "target_result_min": 75,
                    "target_result_max": 400,
                }
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=9),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=10,
                    result_count_text="10 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        assert variants, "demote proposal must yield a variant"
        demoted = variants[0]
        assert demoted.structured_filters.companies == ["Stripe"]
        assert demoted.structured_filters.titles == []  # the demoted dimension


def test_slice_c_full_demote_to_boolean_sets_surface_and_empties_filters():
    """(b, full) A full demote-to-boolean: the only inherited dimension is demoted,
    leaving empty filters and surface='boolean'. This is the deliberate-demote marker
    the seeding gate keys on (part 5).

    Goes red if a demote-to-empty leaves filters non-empty or fails to mark surface.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=10,
            root_boolean='"VP"',
            active_boolean='"VP" AND engineering',
            active_filters={"titles": ["Staff Software Engineer"]},
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer", signal=False)

        proposal = {
            "variants": [
                {
                    "variant_kind": "structured_filter",
                    "hypothesis": "title filter killed recall; drop to keyword",
                    "boolean": '"VP" AND engineering AND "Staff Software Engineer"',
                    "surface": "boolean",
                    "structured_controls": {"demote": ["titles"]},
                    "target_result_min": 75,
                    "target_result_max": 400,
                }
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=10),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=8,
                    result_count_text="8 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        assert variants, "full demote must still yield a runnable keyword variant"
        demoted = variants[0]
        assert demoted.structured_filters.is_empty()
        assert demoted.surface == "boolean"
        assert demoted.variant_kind == "structured_filter"  # provenance: ran the structured planner


def test_slice_c_demote_to_boolean_not_reseeded_by_runtime_or_bootstrap():
    """(c) SEEDING GATE: a deliberate demote-to-boolean variant is NOT re-seeded with
    the lane's structured filters — neither by the runtime seed calls
    (orchestrator.py:2058/2116) NOR by bootstrap_experiment_state's legacy-variant
    seeding (slice B's HIGH fix). The surface marker distinguishes 'deliberately
    demoted to boolean' (do NOT re-seed) from 'never had filters' (seed it).

    Goes red if the gate stops keying on the surface marker and a demote variant gets
    its filters silently restored by the carryover.
    """
    from linkedin.search_intelligence import (
        bootstrap_experiment_state,
        seed_structured_filters_onto_variants,
    )

    lane_filters = {
        "titles": ["Staff Software Engineer"],
        "sidebar_filters": {"locations": ["New York City"]},
    }

    # (c.1) RUNTIME seed call: a demote-to-boolean variant marked surface='boolean'
    # + variant_kind='structured_filter' must be skipped by the carryover seed.
    demote_variant = LinkedInSearchVariant(
        variant_id="demote-1",
        parent_variant_id="active-1",
        root_string_id=11,
        boolean='"VP" AND engineering',
        variant_kind="structured_filter",
        surface="boolean",  # deliberate demotion marker
    )
    assert demote_variant.structured_filters.is_empty()

    # A genuinely-empty keyword variant (never had filters) in the same batch MUST
    # still be seeded — proving the gate discriminates rather than blanket-skipping.
    fresh_keyword = LinkedInSearchVariant(
        variant_id="recall-1",
        parent_variant_id="active-1",
        root_string_id=11,
        boolean='"VP" AND engineering OR director',
        variant_kind="recall",
        surface="boolean",
    )

    seed_structured_filters_onto_variants(lane_filters, [demote_variant, fresh_keyword])

    assert demote_variant.structured_filters.is_empty(), (
        "deliberate demote-to-boolean must NOT be re-seeded with the lane's filters"
    )
    assert fresh_keyword.structured_filters.titles == ["Staff Software Engineer"], (
        "a never-had-filters keyword variant must still be seeded"
    )

    # (c.2) BOOTSTRAP legacy-variant seeding: a resumed hybrid lane whose committed
    # refinement was a demote-to-boolean must not have geography re-seeded onto that
    # legacy variant by bootstrap_experiment_state. The demote is carried across the
    # crash through the REAL runtime: apply_shadow clears the checkpointed
    # structured_filters when the active variant is a deliberate boolean demotion
    # (no new SearchString schema), so the resume bootstrap finds an empty filter dict
    # and seeds nothing — faithfully reproducing the checkpoint a mid-run demote writes
    # (mirrors the slice-B resume test's real-runtime discipline, :1403-1418).
    ss = SearchString(
        id=12,
        name="geo leaders",
        boolean='"VP" AND engineering AND "Staff Software Engineer"',
        original_boolean='"VP" AND engineering',
        acquisition_mode="linkedin_hybrid",
        structured_filters=dict(lane_filters),
    )
    # Bootstrap the pristine hybrid state, then commit a deliberate demote-to-boolean
    # variant and apply_shadow it back onto the SearchString exactly as the runtime does.
    live = bootstrap_experiment_state(ss)
    assert ss.structured_filters, "pristine hybrid lane checkpoints its filters"
    demoted_commit = LinkedInSearchVariant(
        variant_id="demote-commit",
        parent_variant_id="root",
        root_string_id=ss.id,
        boolean='"VP" AND engineering AND "Staff Software Engineer"',
        variant_kind="structured_filter",
        surface="boolean",  # deliberate demotion marker
    )
    live.variants[demoted_commit.variant_id] = demoted_commit
    live.commit_variant(demoted_commit.variant_id)
    live.apply_shadow(ss)

    # apply_shadow cleared the stale checkpointed filters on the demote.
    assert not ss.structured_filters, (
        "apply_shadow must clear checkpointed filters on a deliberate boolean demote"
    )
    assert ss.refinement_stack, "the demote checkpoints a non-empty refinement_stack"

    # Cross-process resume: bootstrap drives the legacy-chain branch (active != root)
    # and must NOT re-seed geography onto the demoted legacy variant.
    booted = bootstrap_experiment_state(ss)
    active = booted.active_variant
    assert active.variant_id != "root"  # legacy-chain branch
    assert active.structured_filters.is_empty(), (
        "bootstrap must NOT re-seed a deliberately-demoted legacy variant"
    )


def test_slice_c_no_op_proposal_still_rejected():
    """(d) NO-OP: a proposal with boolean==current and NO structured change is
    rejected — the existing no-op guard (boolean==current_boolean) must extend to
    'AND no structured change'. Nothing to run -> not proposed.

    Goes red if adding the structured branch lets a do-nothing proposal slip past the
    no-op guard.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=13, root_boolean='"VP" AND engineering'
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        proposal = {
            "variants": [
                {
                    "variant_kind": "precision",
                    "hypothesis": "no real change",
                    "boolean": '"VP" AND engineering',  # == current
                    "surface": "boolean",
                    "structured_controls": {},  # no structured change either
                    "target_result_min": 75,
                    "target_result_max": 400,
                }
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=13),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=120,  # < 500 so the force-narrow fallback never fires
                    result_count_text="120 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        assert variants == [], "a boolean-unchanged, structured-unchanged proposal is a no-op"


def test_slice_c_pure_keyword_proposal_parses_exactly_as_before():
    """(e) REGRESSION: a pure-keyword proposal (no structured_controls key at all)
    parses to a keyword variant — surface 'boolean', empty filters, kind preserved —
    EXACTLY as the precision/recall/noise_exclusion path did before slice C.

    This is the live-contract guard: the rewrite is additive, so a model that emits
    the OLD keyword-only shape (no surface, no structured_controls) must still parse.
    Goes red if the new schema makes structured_controls/surface mandatory.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=14, root_boolean='"VP" AND engineering'
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        # The OLD shape: variant_kind + boolean + hypothesis + window. No surface,
        # no structured_controls — exactly what the pre-slice-C prompt elicited.
        proposal = {
            "variants": [
                {
                    "variant_kind": "recall",
                    "hypothesis": "broaden to directors",
                    "boolean": '"VP" OR director',
                    "target_result_min": 75,
                    "target_result_max": 400,
                }
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=14),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=600,
                    result_count_text="600 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        assert len(variants) == 1
        kw = variants[0]
        assert kw.boolean == '"VP" OR director'
        assert kw.variant_kind == "recall"
        assert kw.structured_filters.is_empty()
        assert kw.surface == "boolean"  # default surface for a keyword variant


def test_slice_c_drift_promote_yields_structured_filter_variant():
    """(a, drift twin) PROMOTE via drift: drift is the strongest promote signal — a
    stable early title cluster that later degrades. _plan_drift_refinement, given a
    structured promote proposal, builds a single rescue variant carrying the promoted
    title in structured_filters with a hybrid/structured surface.

    Goes red if the drift prompt re-forbids structured filters or the drift build
    drops the structured_controls.
    """
    from linkedin.search_intelligence import LinkedInVariantSnapshot

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=15,
            root_boolean='"VP" AND engineering',
            active_boolean='"VP" AND engineering',
        )
        state.last_drift_refinement_summary = {
            "decision": "refine_committed",
            "keyword_hypothesis": "preserve the early Staff Software Engineer signal",
            "future_filter_hypothesis": "title filter could lock onto Staff Software Engineer",
        }
        state.early_signal_snapshot = LinkedInVariantSnapshot(
            page_start=1,
            page_end=2,
            result_count=600,
            result_window="healthy",
            title_clusters=[{"label": "Staff Software Engineer", "signal_count": 5}],
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        proposal = {
            "boolean": '"VP" AND engineering',
            "variant_kind": "structured_filter",
            "hypothesis": "lock the degrading-but-early title cluster behind a filter",
            "surface": "hybrid",
            "structured_controls": {"titles": ["Staff Software Engineer"]},
            "keyword_hypothesis": "preserve early signal",
            "future_filter_hypothesis": "",
            "target_result_min": 75,
            "target_result_max": 400,
        }

        planner = MagicMock(return_value=proposal)
        with patch("shared.llm_clients.opus_llm_cached", planner):
            variant, summary = asyncio.run(
                p._plan_drift_refinement(
                    search_string=MagicMock(id=15),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=600,
                    result_count_text="600 results",
                    page_insights=page_insights,
                    page_stats={},
                )
            )

        assert planner.call_args.kwargs["max_tokens"] == 16384
        assert variant is not None, "drift promote must yield a rescue variant"
        assert variant.structured_filters.titles == ["Staff Software Engineer"]
        assert variant.surface in {"hybrid", "structured_only"}
        assert variant.variant_kind == "structured_filter"


def test_force_narrow_adapt_passes_fable_token_headroom():
    """The forced-narrow fallback gives Fable room for thinking + JSON output."""
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.role_title = "Engineering Leader"
        p.brief_obj.role_description = "Builds applied AI systems."
        p.brief_obj.minimum_bar = "Strong engineering leadership signal."

        planner = MagicMock(
            return_value={
                "refined_boolean": '"VP" AND engineering AND "applied AI"',
                "rationale": "tighten around applied AI leaders",
            }
        )

        with patch("shared.llm_clients.opus_llm_cached", planner):
            result = asyncio.run(
                p._force_narrow_adapt(
                    search_string=SearchString(
                        id=42,
                        name="test",
                        boolean='"VP" AND engineering',
                    ),
                    current_boolean='"VP" AND engineering',
                    result_count_text="900 results",
                    all_candidates=[
                        {
                            "page": 1,
                            "outcome": "REJECT",
                            "name": "A Candidate",
                            "title": "Recruiter",
                            "company": "Agency",
                            "rationale": "wrong function",
                        }
                    ],
                    string_stats={
                        "pages": 2,
                        "candidates": 10,
                        "saves": 0,
                        "facial_yes": 0,
                        "facial_no": 10,
                    },
                )
            )

        assert planner.call_args.kwargs["max_tokens"] == 16384
        assert result == 'narrow:"VP" AND engineering AND "applied AI"'


# ---------------------------------------------------------------------------
# Slice E (part 2) — deterministic circuit-breaker on the proposal prompt.
# After K (config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT) structured demotions
# on a lane, _plan_variant_experiments / _plan_drift_refinement STOP offering the
# promote/structured lever — the option is dropped from the instructions AND the
# JSON schema. Only the external LLM boundary is mocked; the real prompt build runs.
# ---------------------------------------------------------------------------


def _capture_planner_system_prompt(plan_coro_factory):
    """Run a planner with opus_llm_cached spied; return the system prompt it built.

    The proposal returned is a benign keyword variant so the >=500 narrow-adapt
    fallback (_force_narrow_adapt) is not triggered.
    """
    captured: dict = {}

    def _spy(
        system,
        user_prompt,
        expect_json=True,
        max_tokens=None,
        usage_context=None,
        model_name=None,
    ):
        captured["system"] = system
        return {
            "variants": [
                {
                    "variant_kind": "precision",
                    "boolean": '"VP" AND engineering AND staff',
                    "surface": "boolean",
                    "hypothesis": "tighten",
                }
            ],
            # drift planner reads a flat dict; harmless extra keys are ignored by experiments
            "boolean": '"VP" AND engineering AND staff',
            "variant_kind": "precision",
            "hypothesis": "tighten",
        }

    with patch("shared.llm_clients.opus_llm_cached", side_effect=_spy):
        asyncio.run(plan_coro_factory())
    return captured["system"]


def test_slice_e_circuit_breaker_drops_structured_lever_from_variant_prompt():
    """Slice E test (c): once structured_demotions reaches the config limit,
    _plan_variant_experiments no longer offers the structured lever — the PROMOTE
    instruction, the surface key, the structured_controls schema field, and the
    'structured_filter' variant_kind option all disappear. Below the limit they are
    present. Goes red if the breaker stops gating either the instructions or the schema.
    """
    from shared import config

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        def _run(demotions: int):
            state = _experiment_state_with_active(
                root_string_id=7, root_boolean='"VP" AND engineering'
            )
            state.structured_demotions = demotions
            return _capture_planner_system_prompt(
                lambda: p._plan_variant_experiments(
                    search_string=MagicMock(id=7),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=600,
                    result_count_text="600 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={"pages": 1},
                )
            )

        # Below the limit: structured lever is OFFERED.
        open_prompt = _run(config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT - 1)
        assert "PROMOTE" in open_prompt
        assert "structured_controls" in open_prompt
        assert '"surface"' in open_prompt
        assert "structured_filter" in open_prompt

        # At the limit: structured lever is WITHDRAWN from instructions AND schema.
        closed_prompt = _run(config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT)
        assert "PROMOTE" not in closed_prompt
        assert "structured_controls" not in closed_prompt
        assert '"surface"' not in closed_prompt
        assert "structured_filter" not in closed_prompt
        # Keyword variants remain the backbone.
        assert "precision" in closed_prompt


def test_slice_e_circuit_breaker_drops_structured_lever_from_drift_prompt():
    """Slice E test (c, drift twin): the same breaker withdraws the structured lever
    from _plan_drift_refinement's prompt at the limit. The drift rescue stays a keyword
    rescue; the deterministic gate and lifecycle decision are untouched (asserted by
    their own suites).
    """
    from shared import config

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        def _run(demotions: int):
            state = _experiment_state_with_active(
                root_string_id=15,
                root_boolean='"VP" AND engineering',
                active_boolean='"VP" AND engineering',
            )
            state.structured_demotions = demotions
            state.last_drift_refinement_summary = {"decision": "refine_committed"}
            return _capture_planner_system_prompt(
                lambda: p._plan_drift_refinement(
                    search_string=MagicMock(id=15),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=600,
                    result_count_text="600 results",
                    page_insights=page_insights,
                    page_stats={"pages": 1},
                )
            )

        open_prompt = _run(config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT - 1)
        assert "PROMOTE" in open_prompt
        assert "structured_controls" in open_prompt
        assert '"surface"' in open_prompt

        closed_prompt = _run(config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT)
        assert "PROMOTE" not in closed_prompt
        assert "structured_controls" not in closed_prompt
        assert '"surface"' not in closed_prompt


def test_slice_e_closed_breaker_strips_disobeying_promote_in_variant_parser():
    """(gap, HIGH) The circuit-breaker is enforced in the DETERMINISTIC parse layer,
    not just the prompt. A disobeying model that ignores the keyword-only instruction
    and still emits surface='hybrid' + structured_controls (a PROMOTE) on a
    closed-breaker lane must NOT produce a structured variant — otherwise the lane
    re-promotes on that round and the breaker is defeated.

    With the breaker closed, _resolve_structured_controls (via
    _plan_variant_experiments) drops the promoted titles/companies and coerces the
    surface to 'boolean', so the built variant is KEYWORD-ONLY. The same payload below
    the limit (lever open) IS honored as a promote — pinning that the CLOSURE is what
    strips it, not a generic parse break.

    Goes red if the parser honors a promote on a closed-breaker lane.
    """
    from shared import config

    proposal = {
        "variants": [
            {
                "variant_kind": "structured_filter",
                "hypothesis": "model ignores the keyword-only instruction",
                # CHANGED boolean so the variant survives the no-op guard and we can
                # inspect the built (keyword-only) result rather than have it dropped.
                "boolean": '"VP" AND engineering AND staff',
                "surface": "hybrid",
                "structured_controls": {"titles": ["Staff Software Engineer"]},
                "target_result_min": 75,
                "target_result_max": 400,
            }
        ]
    }

    def _run(demotions: int):
        with tempfile.TemporaryDirectory() as td:
            p = _make_pipeline(td)
            state = _experiment_state_with_active(
                root_string_id=30, root_boolean='"VP" AND engineering'
            )
            state.structured_demotions = demotions
            page_insights = _page_insights_with_title_cluster("Staff Software Engineer")
            with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
                return asyncio.run(
                    p._plan_variant_experiments(
                        search_string=MagicMock(id=30),
                        experiment_state=state,
                        current_boolean='"VP" AND engineering',
                        result_count=600,
                        result_count_text="600 results",
                        page_insights=page_insights,
                        all_candidates=[],
                        string_stats={},
                    )
                )

    # Lever OPEN: the promote is honored (control test).
    open_variants = _run(config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT - 1)
    promoted = next(
        (v for v in open_variants if not v.structured_filters.is_empty()), None
    )
    assert promoted is not None, "below the limit, the promote must be honored"
    assert promoted.structured_filters.titles == ["Staff Software Engineer"]
    assert promoted.surface in {"hybrid", "structured_only"}

    # Lever CLOSED: the disobeying promote is stripped at parse time -> keyword-only.
    closed_variants = _run(config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT)
    assert closed_variants, "a changed-boolean variant must still be runnable as keyword"
    for v in closed_variants:
        assert v.structured_filters.is_empty(), (
            "a closed-breaker lane must not carry a promoted structured dim"
        )
        assert v.surface == "boolean", (
            "a closed-breaker lane's surface must be coerced to boolean"
        )
        assert v.variant_kind != "structured_filter", (
            "a stripped promote must fall through to a keyword kind, not wear the "
            "structured_filter provenance marker"
        )


def test_slice_e_closed_breaker_strips_disobeying_promote_in_drift_parser():
    """(gap, HIGH — drift twin) The same deterministic enforcement holds in
    _plan_drift_refinement: a disobeying drift rescue that emits surface='hybrid' +
    structured_controls on a closed-breaker lane is parsed keyword-only.

    Goes red if the drift parser honors a promote on a closed-breaker lane.
    """
    from shared import config

    payload = {
        "boolean": '"VP" AND engineering AND staff',  # changed -> survives no-op guard
        "variant_kind": "structured_filter",
        "surface": "hybrid",
        "structured_controls": {"titles": ["Staff Software Engineer"]},
        "hypothesis": "model ignores the keyword-only rescue instruction",
        "target_result_min": 75,
        "target_result_max": 400,
    }

    def _run(demotions: int):
        with tempfile.TemporaryDirectory() as td:
            p = _make_pipeline(td)
            state = _experiment_state_with_active(
                root_string_id=31,
                root_boolean='"VP" AND engineering',
                active_boolean='"VP" AND engineering',
            )
            state.structured_demotions = demotions
            state.last_drift_refinement_summary = {"decision": "refine_committed"}
            page_insights = _page_insights_with_title_cluster("Staff Software Engineer")
            with patch("shared.llm_clients.opus_llm_cached", return_value=payload):
                return asyncio.run(
                    p._plan_drift_refinement(
                        search_string=MagicMock(id=31),
                        experiment_state=state,
                        current_boolean='"VP" AND engineering',
                        result_count=600,
                        result_count_text="600 results",
                        page_insights=page_insights,
                        page_stats={"pages": 1},
                    )
                )

    open_variant, _ = _run(config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT - 1)
    assert open_variant is not None
    assert open_variant.structured_filters.titles == ["Staff Software Engineer"]
    assert open_variant.surface in {"hybrid", "structured_only"}

    closed_variant, _ = _run(config.SEARCH_EXPERIMENT_STRUCTURED_FAILURE_LIMIT)
    assert closed_variant is not None, "the changed-boolean rescue must still run"
    assert closed_variant.structured_filters.is_empty(), (
        "a closed-breaker drift rescue must not carry a promoted structured dim"
    )
    assert closed_variant.surface == "boolean"
    assert closed_variant.variant_kind != "structured_filter"


# ---------------------------------------------------------------------------
# Slice C adversarial-gap closures (HIGH/MEDIUM): the eight planner-output tests
# above exercise only the PLAN/parse boundary on well-formed input. These pin the
# edges the adversarial lenses found — mislabel provenance, malformed-input
# totality, partial-parse isolation, and the promote EXECUTION seam (real
# apply_variant), which no slice-C test drove before.
# ---------------------------------------------------------------------------


def test_slice_c_mislabel_empty_controls_is_not_a_demotion_and_still_seeds():
    """(gap, medium) A proposal that SELF-LABELS variant_kind='structured_filter'
    with surface='boolean' + EMPTY structured_controls + a CHANGED Boolean is a
    keyword variant, not a deliberate demote. Provenance, not the model's label,
    must decide: with no structured control run, the variant keeps its keyword kind
    so is_deliberate_boolean_demotion is False and the lane's structured-filter seed
    is NOT denied (which would silently drop geography for that one variant on a
    hybrid lane).

    Goes red if _resolve_structured_controls keeps a label-only 'structured_filter'
    kind on an empty-controls/unchanged-filters proposal.
    """
    from linkedin.search_intelligence import is_deliberate_boolean_demotion

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=20, root_boolean='"VP" AND engineering'
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        proposal = {
            "variants": [
                {
                    "variant_kind": "structured_filter",  # model mislabels...
                    "hypothesis": "really just a keyword tighten",
                    "boolean": '"VP" AND engineering AND platform',  # changed
                    "surface": "boolean",
                    "structured_controls": {},  # ...but ran NO structured control
                    "target_result_min": 75,
                    "target_result_max": 400,
                }
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=20),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=600,
                    result_count_text="600 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        assert len(variants) == 1
        v = variants[0]
        assert v.structured_filters.is_empty()
        assert v.variant_kind != "structured_filter", (
            "an empty-controls, structurally-unchanged proposal must NOT wear the "
            "structured_filter provenance marker"
        )
        assert not is_deliberate_boolean_demotion(v), (
            "a mislabeled keyword variant must not read as a deliberate demote"
        )

        # The seeding gate must therefore SEED it (it never had filters), not skip it.
        seed_structured_filters_onto_variants(
            {"sidebar_filters": {"locations": ["New York City"]}}, [v]
        )
        assert v.structured_filters.sidebar_filters.get("locations") == [
            "New York City"
        ], "a mislabeled keyword variant must still receive the lane's seed"


def test_slice_c_malformed_structured_controls_parse_safely_in_experiments():
    """(gap, high/medium) Malformed structured_controls must parse SAFELY, never
    crash. A non-dict controls value (list) and a str title value are the two
    disobeying-model shapes the adversarial lens reproduced (str iterated
    per-character into garbage titles; list raised AttributeError). Both must
    degrade to a clean keyword parse.

    Goes red if _resolve_structured_controls drops its isinstance guards.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=21, root_boolean='"VP" AND engineering'
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        proposal = {
            "variants": [
                {
                    "variant_kind": "precision",
                    "hypothesis": "controls is a list, not a dict",
                    "boolean": '"VP" AND engineering AND list',
                    "structured_controls": ["titles"],  # wrong type (truthy non-dict)
                    "target_result_min": 75,
                    "target_result_max": 400,
                },
                {
                    "variant_kind": "structured_filter",
                    "hypothesis": "titles is a str, not a list",
                    "boolean": '"VP" AND engineering AND str',
                    "surface": "hybrid",
                    "structured_controls": {"titles": "NotAList"},  # wrong type
                    "target_result_min": 75,
                    "target_result_max": 400,
                },
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=21),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=120,
                    result_count_text="120 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        # Both items parse (no crash) as keyword variants with empty filters.
        assert len(variants) == 2
        for v in variants:
            assert v.structured_filters.is_empty(), (
                "malformed structured_controls must not populate filters"
            )
            assert v.structured_filters.titles == [], (
                "a str title value must NOT iterate per-character into garbage titles"
            )


def test_slice_c_experiments_bad_first_item_does_not_drop_siblings():
    """(gap, medium) PARTIAL-PARSE ISOLATION: a malformed FIRST variant must drop
    only itself, never the well-formed siblings parsed after it. The pre-fix
    loop-level except aborted the whole batch on a bad-first ordering and returned
    an EMPTY list (rc<500 returns it directly), silently losing the good variant.

    A target_result_min that cannot int() is the residual per-item throw after the
    parse is hardened. The good second variant must survive.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=22, root_boolean='"VP" AND engineering'
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        proposal = {
            "variants": [
                {
                    "variant_kind": "precision",
                    "hypothesis": "bad first: target window cannot int()",
                    "boolean": '"VP" AND engineering AND bad',
                    "target_result_min": "not-a-number",  # raises in int(...)
                    "target_result_max": 400,
                },
                {
                    "variant_kind": "recall",
                    "hypothesis": "good second: must survive the bad first",
                    "boolean": '"VP" OR director',
                    "target_result_min": 75,
                    "target_result_max": 400,
                },
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=22),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=120,  # < 500 so no force-narrow rescue masks the loss
                    result_count_text="120 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        assert len(variants) == 1, "one bad item must drop only itself"
        assert variants[0].boolean == '"VP" OR director', (
            "the well-formed sibling after a bad-first item must survive"
        )


def test_slice_c_drift_malformed_payload_degrades_without_aborting_run():
    """(gap, high) The drift planner must not let a malformed LLM payload escape: a
    non-dict structured_controls (list) — and a non-dict payload entirely — must
    degrade to the keyword rescue, never raise AttributeError out of the planner
    (which, at the run driver, aborts the ENTIRE run).

    Goes red if _resolve_structured_controls/the payload coerce/the resolve guard is
    removed from _plan_drift_refinement.
    """
    from linkedin.search_intelligence import LinkedInVariantSnapshot

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)

        def _state():
            s = _experiment_state_with_active(
                root_string_id=23,
                root_boolean='"VP" AND engineering',
                active_boolean='"VP" AND engineering',
            )
            s.last_drift_refinement_summary = {"decision": "refine_committed"}
            s.early_signal_snapshot = LinkedInVariantSnapshot(
                page_start=1,
                page_end=2,
                result_count=600,
                result_window="healthy",
                title_clusters=[{"label": "Staff Software Engineer", "signal_count": 5}],
            )
            return s

        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        # (1) non-dict structured_controls inside an otherwise-valid payload.
        bad_controls = {
            "boolean": '"VP" AND engineering AND rescue',
            "variant_kind": "precision",
            "structured_controls": ["titles"],  # wrong type
        }
        with patch("shared.llm_clients.opus_llm_cached", return_value=bad_controls):
            variant, _summary = asyncio.run(
                p._plan_drift_refinement(
                    search_string=MagicMock(id=23),
                    experiment_state=_state(),
                    current_boolean='"VP" AND engineering',
                    result_count=600,
                    result_count_text="600 results",
                    page_insights=page_insights,
                    page_stats={},
                )
            )
        assert variant is not None, "a malformed-controls drift payload must still rescue"
        assert variant.structured_filters.is_empty()

        # (2) a wholly non-dict payload (model returned a JSON list).
        with patch("shared.llm_clients.opus_llm_cached", return_value=["not", "a", "dict"]):
            variant2, _summary2 = asyncio.run(
                p._plan_drift_refinement(
                    search_string=MagicMock(id=23),
                    experiment_state=_state(),
                    current_boolean='"VP" AND engineering',
                    result_count=600,
                    result_count_text="600 results",
                    page_insights=page_insights,
                    page_stats={},
                )
            )
        # No crash; the keyword fallback decides whether a rescue is runnable. Either a
        # keyword rescue or None is acceptable — the contract is "did not raise".
        if variant2 is not None:
            assert variant2.structured_filters.is_empty()


def test_slice_c_boolean_lane_promote_actually_applies_via_apply_variant():
    """(gap, high) THE EXECUTION SEAM: a mid-run PROMOTE on a compiled linkedin_boolean
    lane must actually EXECUTE — not be silently rejected by apply_variant's
    'experimental_structured_filters_not_supported' gate. The orchestrator resolves
    the variant's acquisition mode to linkedin_hybrid (because it carries structured
    filters) and upgrades the SearchString to keep the 'filters present <=> hybrid'
    invariant, so the real apply_variant runs the structured plan.

    The eight planner-output slice-C tests stop at PLAN/parse; this drives the real
    apply_variant. Goes red if a promote on a boolean lane is rejected or routed
    through bare keyword entry instead of apply_advanced_search_plan.
    """
    from linkedin.advanced_search import ControlApplicationResult
    from linkedin.browser import SearchEntryResult
    from linkedin.input_backends import TypingResult
    from linkedin.search_intelligence import bootstrap_experiment_state
    from linkedin.search_mutation import (
        LinkedInSearchMutationDeps,
        LinkedInSearchMutationExecutor,
    )

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.go_back_to_results = AsyncMock()
        p.browser.apply_advanced_search_plan = AsyncMock(
            return_value=ControlApplicationResult(
                success=True,
                applied_controls=["keywords", "job_titles"],
                failed_controls=[],
                unsupported_controls=[],
            )
        )
        p.browser.enter_search_string = AsyncMock(
            return_value=SearchEntryResult(
                typing_result=TypingResult(
                    transport="advanced_search_plan",
                    duration_ms=0,
                    typo_count=0,
                    used_correction=False,
                    fallback_char_count=0,
                ),
                results_wait_ms=0,
            )
        )
        p.browser.get_results_count_text = AsyncMock(return_value="180")
        p.browser.get_results_count = AsyncMock(return_value=180)
        p.browser.get_card_snapshot = AsyncMock(return_value={"name": "Ada"})

        # A lane compiled boolean (no producer-time filters) — the dominant lane type.
        ss = SearchString(
            id=30, name="boolean lane", boolean="foo", acquisition_mode="linkedin_boolean"
        )
        state = bootstrap_experiment_state(ss)
        promote = LinkedInSearchVariant(
            variant_id="promote-1",
            parent_variant_id="root",
            root_string_id=30,
            boolean="foo",
            variant_kind="structured_filter",
            surface="hybrid",
            structured_filters=LinkedInStructuredFilters(titles=["Staff Software Engineer"]),
        )
        state.begin_experiment_round([promote])

        def _set_budget(value: int) -> None:
            p._search_mutation_budget_used = value

        executor = LinkedInSearchMutationExecutor(
            LinkedInSearchMutationDeps(
                browser=p.browser,
                log_path=p.log_path,
                get_input_mode=lambda: p.input_mode,
                get_runtime_run_id=lambda: p._runtime_run_id,
                get_runtime_state=lambda: p._runtime_state,
                get_search_mutation_budget_used=lambda: p._search_mutation_budget_used,
                set_search_mutation_budget_used=_set_budget,
            )
        )

        # The orchestrator's mode resolution (the fix): a filter-bearing variant runs
        # hybrid and the SearchString is upgraded to preserve the invariant.
        resolved_mode = type(p)._acquisition_mode_for_variant(ss, promote)
        assert resolved_mode == "linkedin_hybrid"
        assert ss.acquisition_mode == "linkedin_hybrid", (
            "the boolean lane is upgraded to hybrid so filters-present <=> hybrid holds"
        )

        with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
            result = asyncio.run(
                executor.apply_variant(
                    search_string=ss,
                    experiment_state=state,
                    variant=promote,
                    acquisition_mode=resolved_mode,
                )
            )

        assert result.applied is True, (
            "a mid-run promote on a boolean lane must EXECUTE, not be rejected"
        )
        assert result.blocked_reason == ""
        # It ran the STRUCTURED plan (apply_advanced_search_plan), not bare keyword entry.
        p.browser.apply_advanced_search_plan.assert_awaited_once()
        (plan_arg,), _ = p.browser.apply_advanced_search_plan.await_args
        assert "job_titles" in [c.dimension for c in plan_arg.controls]
        assert plan_arg.keyword_boolean == "foo"


# ---------------------------------------------------------------------------
# SLICE F — posture-aware lifecycle windows.
#
# A filter-led / structured search is legitimately NARROWER than a keyword
# search. The deterministic lifecycle gate (classify_result_window ->
# decide_variant_lifecycle) is tuned for keyword breadth, so a good structured
# probe would classify too_narrow and ABANDON. Fix: scale a filter-led variant's
# healthy result-window DOWN by config.SEARCH_EXPERIMENT_FILTER_LED_WINDOW_FACTOR
# AT CONSTRUCTION (the build sites), so the decision function reads an
# already-scaled window and stays a PURE, byte-stable function. A boolean variant
# uses the UNSCALED window — the default path is byte-identical.
# ---------------------------------------------------------------------------


def test_slice_f_filter_led_variant_window_scaled_down_at_construction():
    """(F.2) A filter-led promote built by _plan_variant_experiments carries a window
    scaled DOWN by the config factor — target_result_min/max are the helper's scaled
    output, NOT the model-proposed keyword window. Goes red if the build site stops
    scaling a structured variant's window.
    """
    from shared import config
    from linkedin.search_intelligence import scale_window_for_surface

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=40, root_boolean='"VP" AND engineering'
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        proposal = {
            "variants": [
                {
                    "variant_id": "promote-1",
                    "variant_kind": "structured_filter",
                    "hypothesis": "promote the clean title cluster to a filter",
                    "boolean": '"VP" AND engineering',
                    "surface": "hybrid",
                    "structured_controls": {"titles": ["Staff Software Engineer"]},
                    "target_result_min": 75,
                    "target_result_max": 400,
                }
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=40),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=600,
                    result_count_text="600 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        promoted = next(
            (v for v in variants if not v.structured_filters.is_empty()), None
        )
        assert promoted is not None, "the structured promote must be proposable"
        expected_min, expected_max = scale_window_for_surface(
            75, 400, surface=promoted.surface,
            structured_filters=promoted.structured_filters,
        )
        assert promoted.target_result_min == expected_min
        assert promoted.target_result_max == expected_max
        # The scaled window is strictly narrower than the proposed keyword window.
        assert promoted.target_result_min < 75
        assert promoted.target_result_max < 400
        # The factor actually drove the scaling (not some unrelated default).
        assert promoted.target_result_min == max(
            1, round(75 * config.SEARCH_EXPERIMENT_FILTER_LED_WINDOW_FACTOR)
        )


def test_slice_f_drift_filter_led_variant_window_scaled_down():
    """(F.2, drift twin) The drift build site scales a structured rescue's window too."""
    from linkedin.search_intelligence import (
        LinkedInVariantSnapshot,
        scale_window_for_surface,
    )

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=41,
            root_boolean='"VP" AND engineering',
            active_boolean='"VP" AND engineering',
        )
        state.early_signal_snapshot = LinkedInVariantSnapshot(
            page_start=1, page_end=2, result_count=600, result_window="healthy",
            title_clusters=[{"label": "Staff Software Engineer", "signal_count": 5}],
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        proposal = {
            "boolean": '"VP" AND engineering',
            "variant_kind": "structured_filter",
            "hypothesis": "lock the early title cluster behind a filter",
            "surface": "hybrid",
            "structured_controls": {"titles": ["Staff Software Engineer"]},
            "target_result_min": 150,
            "target_result_max": 800,
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variant, _summary = asyncio.run(
                p._plan_drift_refinement(
                    search_string=MagicMock(id=41),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=1600,
                    result_count_text="1600 results",
                    page_insights=page_insights,
                    page_stats={},
                )
            )

        assert variant is not None and not variant.structured_filters.is_empty()
        expected_min, expected_max = scale_window_for_surface(
            150, 800, surface=variant.surface,
            structured_filters=variant.structured_filters,
        )
        assert variant.target_result_min == expected_min
        assert variant.target_result_max == expected_max
        assert variant.target_result_min < 150 and variant.target_result_max < 800


def test_slice_f_boolean_variant_window_is_byte_identical_unscaled():
    """(F.c regression) A boolean keyword variant uses the UNSCALED window — its
    target_result_min/max are byte-identical to the model-proposed keyword window, and
    its classification + lifecycle decision are unchanged from pre-F. Goes red if the
    factor ever leaks onto the default (boolean) path.
    """
    from linkedin.lane_variant_decisions import (
        VariantDecisionInput,
        decide_variant_lifecycle,
    )

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        state = _experiment_state_with_active(
            root_string_id=42, root_boolean='"VP" AND engineering'
        )
        page_insights = _page_insights_with_title_cluster("Staff Software Engineer")

        proposal = {
            "variants": [
                {
                    "variant_kind": "precision",
                    "hypothesis": "tighten the boolean",
                    "boolean": '"VP" AND engineering AND platform',
                    "surface": "boolean",
                    "structured_controls": {},
                    "target_result_min": 75,
                    "target_result_max": 400,
                }
            ]
        }

        with patch("shared.llm_clients.opus_llm_cached", return_value=proposal):
            variants = asyncio.run(
                p._plan_variant_experiments(
                    search_string=MagicMock(id=42),
                    experiment_state=state,
                    current_boolean='"VP" AND engineering',
                    result_count=600,
                    result_count_text="600 results",
                    page_insights=page_insights,
                    all_candidates=[],
                    string_stats={},
                )
            )

        keyword = [v for v in variants if v.structured_filters.is_empty()]
        assert keyword, "the boolean variant must be proposable"
        kw = keyword[0]
        # UNSCALED — byte-identical to the proposed keyword window.
        assert kw.target_result_min == 75
        assert kw.target_result_max == 400
        assert kw.surface == "boolean"

        # A keyword count of 150 (inside [75,400]) classifies healthy exactly as pre-F.
        kw.result_count = 150
        kw.pages_reviewed = 1
        kw.saves = 1
        assert kw.classify_result_window() == "healthy"
        out = decide_variant_lifecycle(
            VariantDecisionInput(variant=kw, experiment_state=state)
        )
        assert out.action == "commit"


def test_slice_f_dead_end_closed_narrow_filter_variant_commits_not_rescues():
    """(F.d) The as-built dead-end (cloris-as-built-architecture.md:129): a too_narrow
    filter-led variant -> rescue/broaden -> _drop_one_filter drops the ONLY filter ->
    un-broadenable -> the rescue degenerates (spawn_rescue_variant_from_hint returns
    None when there is nothing left to drop, or — the step before — a hollow variant
    that has shed its only filter AND its boolean) -> the orchestrator stops with no
    runnable structured probe.

    With the window scaled at construction the same count classifies HEALTHY and the
    lifecycle COMMITS, so the dead-end never opens. The counterfactual (UNSCALED window)
    is asserted to prove what the fix closed: too_narrow -> rescue -> the only filter is
    dropped, leaving an un-runnable variant; and one filter further along (zero filters,
    bare boolean) the spawner returns None outright.
    """
    from linkedin.lane_variant_decisions import (
        VariantDecisionInput,
        decide_variant_lifecycle,
    )
    from linkedin.search_intelligence import (
        scale_window_for_surface,
        spawn_rescue_variant_from_hint,
    )

    # A filter-led variant with a SINGLE filter and a count below the keyword min.
    # surface="structured_only" + empty boolean: the filter alone carries the search,
    # so there is no Boolean left to broaden — broadening can only drop the filter.
    single_filter = LinkedInStructuredFilters(titles=["Staff Software Engineer"])
    state = LinkedInExperimentState(
        root_string_id=43, intent=LinkedInSearchIntent(root_boolean='"VP"')
    )

    def _filter_variant(target_min, target_max):
        return LinkedInSearchVariant(
            variant_id="filter-led-1",
            parent_variant_id="root",
            root_string_id=43,
            boolean="",
            variant_kind="structured_filter",
            surface="structured_only",
            structured_filters=LinkedInStructuredFilters.from_dict(single_filter.to_dict()),
            result_count=30,
            target_result_min=target_min,
            target_result_max=target_max,
            saves=1,
            pages_reviewed=1,
            probe_page_budget=1,
            probe_pages_used=1,
        )

    # FIXED path: scaled window at construction -> healthy -> commit.
    scaled_min, scaled_max = scale_window_for_surface(
        75, 400, surface="structured_only", structured_filters=single_filter
    )
    fixed = _filter_variant(scaled_min, scaled_max)
    assert fixed.classify_result_window() == "healthy"
    fixed_out = decide_variant_lifecycle(
        VariantDecisionInput(variant=fixed, experiment_state=state)
    )
    assert fixed_out.action == "commit", fixed_out.to_dict()

    # COUNTERFACTUAL (pre-F, UNSCALED keyword window): too_narrow -> rescue -> broaden.
    unscaled = _filter_variant(75, 400)
    assert unscaled.classify_result_window() == "too_narrow"
    unscaled_out = decide_variant_lifecycle(
        VariantDecisionInput(variant=unscaled, experiment_state=state)
    )
    assert unscaled_out.action == "rescue"
    assert unscaled_out.next_variant_hint == {"variant_kind": "recall", "action": "broaden"}

    # The broaden spawn drops the ONLY filter — the spawned variant has lost its
    # structured identity AND has no Boolean to fall back on: un-runnable. That is the
    # dead-end the scaled-window commit avoids.
    spawned = spawn_rescue_variant_from_hint(
        unscaled, hint=unscaled_out.next_variant_hint, root_string_id=43
    )
    assert spawned is not None  # the step before the terminal None
    assert spawned.structured_filters.is_empty(), (
        "broaden dropped the only structured filter — the probe lost its identity"
    )
    assert spawned.boolean == "", (
        "no Boolean to broaden into: the dropped-filter rescue is un-runnable"
    )

    # One step further along (a variant with ZERO filters and a bare un-broadenable
    # Boolean) the spawner returns None outright — the terminal orchestrator stop the
    # as-built doc names.
    bare = LinkedInSearchVariant(
        variant_id="bare",
        parent_variant_id="root",
        root_string_id=43,
        boolean="engineer",
        variant_kind="recall",
        structured_filters=LinkedInStructuredFilters(),
        result_count=30,
        target_result_min=75,
        target_result_max=400,
    )
    assert (
        spawn_rescue_variant_from_hint(
            bare, hint={"variant_kind": "recall", "action": "broaden"}, root_string_id=43
        )
        is None
    )


def test_precision_rescue_uses_noise_anchors_not_dominant_summary_sentence():
    from linkedin.search_intelligence import (
        _noise_terms,
        spawn_rescue_variant_from_hint,
    )

    summary_sentence = (
        "glance thinks this page is dominated by product managers and program managers"
    )
    parent = LinkedInSearchVariant(
        variant_id="v1",
        parent_variant_id="root",
        root_string_id=43,
        boolean='"ML" AND "engineer"',
        variant_kind="precision",
        last_page_insights=LinkedInPageInsights(
            page=1,
            result_count=5000,
            result_window="too_broad",
            noise_anchors=["Product Manager at BankCorp"],
            dominant_non_fit_patterns=[summary_sentence],
            glance_action="reformulate",
        ),
        probe_pages_used=1,
    )

    assert _noise_terms(parent) == ["Product Manager at BankCorp"]
    spawned = spawn_rescue_variant_from_hint(
        parent,
        hint={"variant_kind": "precision", "action": "narrow"},
        root_string_id=43,
    )

    assert spawned is not None
    assert 'NOT ("Product Manager at BankCorp")' in spawned.boolean
    assert summary_sentence not in spawned.boolean


def test_slice_f_hybrid_parent_broaden_reprojects_inherited_window_to_keyword():
    """(F.d2) The runnable-keyword-child case test (d) does not exercise, and the
    directional INVERSE of the dead-end it closed.

    A HYBRID parent (non-empty boolean + one structured filter) carries a window scaled
    DOWN for structured precision. A recall/broaden rescue drops the last filter but the
    boolean survives, so the child is a RUNNABLE KEYWORD variant. spawn_rescue inherits the
    parent's window; pre-fix it inherited the structured-narrow window RAW onto a now-boolean
    child, so the keyword-tuned gate (classify_result_window) mis-read a healthy keyword count
    as over-max. The spawner is NOT one of the four scale build sites and the orchestrator
    registers the rescue child straight into experiment_state (orchestrator.py:788-791) without
    re-planning, so the inherited window is exactly what the next probe is judged against.

    The fix resolves the child's surface from its actual (boolean, filters) and re-projects
    the window across the posture flip: filter-led parent -> boolean child un-scales the window
    back to keyword bounds. The same healthy count then classifies healthy, not noisy/too_broad.
    """
    from shared import config
    from linkedin.lane_variant_decisions import (
        VariantDecisionInput,
        decide_variant_lifecycle,
    )
    from linkedin.search_intelligence import (
        scale_window_for_surface,
        spawn_rescue_variant_from_hint,
        unscale_window_from_surface,
    )

    state = LinkedInExperimentState(
        root_string_id=44, intent=LinkedInSearchIntent(root_boolean='"VP"')
    )

    # Hybrid parent: keyword AND a title filter, window scaled for structured precision
    # at construction (exactly what the build sites bake in).
    parent_filters = LinkedInStructuredFilters(titles=["ML Engineer"])
    scaled_min, scaled_max = scale_window_for_surface(
        75, 400, surface="hybrid", structured_filters=parent_filters
    )
    assert scaled_min < 75 and scaled_max < 400  # the narrow structured window
    parent = LinkedInSearchVariant(
        variant_id="hybrid-1",
        parent_variant_id="root",
        root_string_id=44,
        boolean='"ML" AND "engineer" AND platform',
        variant_kind="structured_filter",
        surface="hybrid",
        structured_filters=LinkedInStructuredFilters.from_dict(parent_filters.to_dict()),
        target_result_min=scaled_min,
        target_result_max=scaled_max,
        probe_pages_used=1,
    )

    spawned = spawn_rescue_variant_from_hint(
        parent, hint={"variant_kind": "recall", "action": "broaden"}, root_string_id=44
    )
    # The broaden dropped the only filter but the boolean survives: a RUNNABLE keyword child.
    assert spawned is not None
    assert spawned.structured_filters.is_empty()
    assert spawned.boolean and spawned.boolean != parent.boolean
    # The spawner resolves the child's flipped posture instead of defaulting surface="".
    assert spawned.surface == "boolean"
    # The inherited narrow window is re-projected UP to keyword bounds (not copied raw).
    expected_min, expected_max = unscale_window_from_surface(scaled_min, scaled_max)
    assert (spawned.target_result_min, spawned.target_result_max) == (expected_min, expected_max)
    assert spawned.target_result_min >= 70 and spawned.target_result_max == 400, (
        "the keyword child must be judged on keyword-class bounds, not the structured window"
    )

    # A keyword count of 300 is healthy for the broadened keyword child.
    spawned.result_count = 300
    spawned.pages_reviewed = 1
    assert spawned.classify_result_window() == "healthy"
    out = decide_variant_lifecycle(
        VariantDecisionInput(variant=spawned, experiment_state=state)
    )
    assert out.action != "rescue", out.to_dict()

    # COUNTERFACTUAL (pre-fix): the same child carrying the RAW inherited structured-narrow
    # window mis-judges the healthy 300 as over-max (noisy/too_broad) and spuriously rescues.
    spawned.target_result_min, spawned.target_result_max = scaled_min, scaled_max
    assert spawned.classify_result_window() in {"noisy", "too_broad"}
    mis_out = decide_variant_lifecycle(
        VariantDecisionInput(variant=spawned, experiment_state=state)
    )
    assert mis_out.action == "rescue"


def test_slice_f_same_posture_rescue_inherits_window_unchanged():
    """(F.d3) A SAME-posture rescue must not double-scale. A structured parent whose rescue
    keeps a structured filter (variant_kind="structured_filter") inherits the already-scaled
    window byte-for-byte — re-projection only fires on a posture FLIP, so the same-posture
    path the commit/dead-end behavior depends on stays byte-stable.
    """
    from linkedin.search_intelligence import (
        scale_window_for_surface,
        spawn_rescue_variant_from_hint,
    )

    parent_filters = LinkedInStructuredFilters(titles=["ML Engineer"], companies=["Acme"])
    scaled_min, scaled_max = scale_window_for_surface(
        75, 400, surface="structured_only", structured_filters=parent_filters
    )
    parent = LinkedInSearchVariant(
        variant_id="structured-1",
        parent_variant_id="root",
        root_string_id=45,
        boolean="",
        variant_kind="structured_filter",
        surface="structured_only",
        structured_filters=LinkedInStructuredFilters.from_dict(parent_filters.to_dict()),
        target_result_min=scaled_min,
        target_result_max=scaled_max,
        probe_pages_used=1,
    )

    # structured_filter rescue keeps a non-empty filter set -> child stays filter-led.
    spawned = spawn_rescue_variant_from_hint(
        parent, hint={"variant_kind": "structured_filter"}, root_string_id=45
    )
    assert spawned is not None
    assert not spawned.structured_filters.is_empty()
    assert spawned.surface in {"hybrid", "structured_only"}
    # Same posture -> inherited window is byte-identical (no double-scale, no rounding drift).
    assert spawned.target_result_min == scaled_min
    assert spawned.target_result_max == scaled_max


def test_run_full_aborts_run_when_opening_geography_gate_fires():
    """P3a run-level abort (locks orchestrator.py's `except GeographyRegimeError:
    raise` clause in the run_full string loop): a GeographyRegimeError raised from
    string processing must abort the RUN — never be marked a per-string 'error'
    with the loop limping on to string #2 (each string would hit the same gate;
    the wrapper-swallow mode feedback_failclosed_swallowed_by_wrapper warns about).

    Goes red if the explicit re-raise clause is deleted: the generic per-string
    `except Exception` handler would then mark string #1 'error' and CONTINUE —
    calls would reach 2 and no exception would propagate.
    """
    from linkedin.orchestrator import GeographyRegimeError

    with tempfile.TemporaryDirectory() as td:
        p = _crash_recovery_pipeline(td, location="New York City")
        # The run-start session apply succeeds; the gate under test fires
        # INSIDE string processing (the pre-string invariant path).
        p.browser.apply_location_filter = AsyncMock(return_value=True)
        # Two queued strings: the abort must stop the run BEFORE string #2.
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="one", boolean="a", status="queued", block="Block A"),
                SearchString(id=2, name="two", boolean="b", status="queued", block="Block A"),
            ],
        )
        progress.save(str(p.progress_path))

        calls = {"count": 0}

        async def gate_fires(search_string, progress):
            calls["count"] += 1
            raise GeographyRegimeError("geography chips absent after re-assert")

        p._process_string = gate_fires

        with pytest.raises(GeographyRegimeError):
            asyncio.run(p.run_full(resume=True))

        # The run aborted on string #1; string #2 was never started and was
        # never marked done/error by the per-string handler.
        assert calls["count"] == 1
        saved = json.loads(Path(td, "progress.json").read_text())
        assert saved["strings"][1]["status"] == "queued"


def test_run_full_session_start_geography_failure_releases_lock_and_disconnects():
    """Contract-break regression (Wave 1 Opus lens): the session-START apply sits
    inside run_full's cleanup guard. When the gate fires there (facet miss on the
    fresh sidebar — the live SPL condition), the raise must still run the finally:
    browser disconnected, runtime lock RELEASED (a leaked flock wedges every
    subsequent day-cycle session with LOCK_CONFLICT), usage session closed.

    Goes red if _apply_session_location_filter is ever moved back above the try.
    """
    from linkedin.orchestrator import GeographyRegimeError

    with tempfile.TemporaryDirectory() as td:
        p = _crash_recovery_pipeline(td, location="Atlantis Metro Area")
        p.browser.apply_location_filter = AsyncMock(return_value=False)  # facet miss

        with pytest.raises(GeographyRegimeError):
            asyncio.run(p.run_full(resume=True))

        # Cleanup ran: browser disconnected and the runtime lock is re-acquirable
        # (a leaked lock raises RuntimeError/LOCK_CONFLICT here).
        p.browser.disconnect.assert_awaited()
        p._runtime_lock.acquire()
        p._runtime_lock.release()


def test_recruiter_search_page_predicate_rejects_profile_under_discover():
    """Run-start navigation guard: a profile detail page nested under the
    discover path has no filter sidebar and must NOT count as a search page
    (live abort, 2026-07-05: navigation skipped -> empty typeahead ->
    geography gate abort)."""
    from linkedin.orchestrator import _is_recruiter_search_page

    assert _is_recruiter_search_page(
        "https://www.linkedin.com/talent/hire/2078524586/discover/recruiterSearch?project=2078524586"
    )
    assert _is_recruiter_search_page("https://www.linkedin.com/talent/search?x=1")
    assert not _is_recruiter_search_page(
        "https://www.linkedin.com/talent/hire/2078524586/discover/recruiterSearch/profile/AEMAAA?start=25"
    )
    assert not _is_recruiter_search_page("https://www.linkedin.com/feed/")
    assert not _is_recruiter_search_page("")


def test_run_full_navigates_off_a_foreign_project_search_page():
    """E4: a Recruiter SEARCH view belonging to a DIFFERENT project must be
    treated exactly like not being on a search page — run-start navigates to the
    brief's project URL. Before the project-aware decision, `on_search_page` was
    True for any /discover/ URL, so a tab bound from another project's search
    (browser._bind_existing_recruiter_page picks any healthy Recruiter tab) was
    accepted and every save physically landed in the wrong pipeline.
    """
    from linkedin.orchestrator import GeographyRegimeError

    with tempfile.TemporaryDirectory() as td:
        p = _crash_recovery_pipeline(td, location="Atlantis Metro Area")
        # Brief project id is "test-project" (see _make_pipeline); the live tab
        # is another project's search view.
        p.browser.page = MagicMock(
            url=(
                "https://www.linkedin.com/talent/hire/9999999999/discover/"
                "recruiterSearch?project=9999999999"
            )
        )
        p.browser.navigate_to_search = AsyncMock()
        p.browser.apply_location_filter = AsyncMock(return_value=False)  # facet miss

        with pytest.raises(GeographyRegimeError):
            asyncio.run(p.run_full(resume=True))

        p.browser.navigate_to_search.assert_awaited_once_with(
            "https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )


def test_run_full_does_not_navigate_off_the_brief_project_search_page():
    """E4 counterpart: the brief's OWN project search page is still accepted —
    the project-awareness must not turn run-start into an unconditional
    re-navigation (that would drop an operator-prepared sidebar every run).
    """
    from linkedin.orchestrator import GeographyRegimeError

    with tempfile.TemporaryDirectory() as td:
        p = _crash_recovery_pipeline(td, location="Atlantis Metro Area")
        p.browser.page = MagicMock(
            url=(
                "https://www.linkedin.com/talent/hire/test-project/discover/"
                "recruiterSearch?project=test-project"
            )
        )
        p.browser.navigate_to_search = AsyncMock()
        p.browser.apply_location_filter = AsyncMock(return_value=False)

        with pytest.raises(GeographyRegimeError):
            asyncio.run(p.run_full(resume=True))

        p.browser.navigate_to_search.assert_not_awaited()


_BRIEF_PROJECT_SEARCH_URL = (
    "https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
)


@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume"])
def test_run_full_navigates_off_a_projectless_search_page(resume):
    """F1: the E4 carve-out ("absence of either id is NOT a mismatch") was a
    bypass, not a carve-out. `https://www.linkedin.com/talent/search` carries no
    project id, so `_recruiter_page_project_id` returned None, the predicate
    answered False, `on_search_page` was True — and run-start made ZERO
    navigation calls even though the brief pins `test-project`. The run then
    filtered and saved inside whatever pipeline that global view was pointed at.

    When the brief HAS a project, a page that cannot PROVE it belongs to that
    project is UNVERIFIED, and unverified navigates. Both entry regimes are
    pinned: `run_full(resume=False)` and `run_full(resume=True)` share the
    decision but not the setup path above it.
    """
    from linkedin.orchestrator import GeographyRegimeError

    with tempfile.TemporaryDirectory() as td:
        p = _crash_recovery_pipeline(td, location="Atlantis Metro Area")
        # The global Recruiter search view: a real Recruiter page, healthy
        # enough for `_bind_existing_recruiter_page` to adopt, with no project
        # id anywhere in its URL.
        p.browser.page = MagicMock(url="https://www.linkedin.com/talent/search")
        p.browser.navigate_to_search = AsyncMock()
        p.browser.apply_location_filter = AsyncMock(return_value=False)

        with pytest.raises(GeographyRegimeError):
            asyncio.run(p.run_full(resume=resume))

        p.browser.navigate_to_search.assert_awaited_once_with(
            _BRIEF_PROJECT_SEARCH_URL
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://www.linkedin.com/talent/search",
        "about:blank",
        "",
    ],
    ids=["global_search", "about_blank", "empty"],
)
def test_run_full_leaves_a_projectless_brief_on_a_projectless_page(url):
    """F1 acceptance (3): with NO brief project the permissive carve-out is
    retained exactly as-is — `_get_project_url()` returns "" so the run-start
    decision keeps the legacy `/talent/search` fallback, and a page already on
    a Recruiter search view is not re-navigated.
    """
    from linkedin.orchestrator import GeographyRegimeError

    with tempfile.TemporaryDirectory() as td:
        p = _crash_recovery_pipeline(td, location="Atlantis Metro Area")
        p.brief_obj.linkedin_project_id = None
        del p.browser._project_id  # _get_project_url()'s auto-detect fallback
        p.browser.page = MagicMock(url=url)
        p.browser.navigate_to_search = AsyncMock()
        p.browser.apply_location_filter = AsyncMock(return_value=False)

        with pytest.raises(GeographyRegimeError):
            asyncio.run(p.run_full(resume=True))

        if url == "https://www.linkedin.com/talent/search":
            p.browser.navigate_to_search.assert_not_awaited()
        else:
            p.browser.navigate_to_search.assert_awaited_once_with(
                "https://www.linkedin.com/talent/search"
            )


def test_project_page_predicate_truth_table_over_its_input_space():
    """The predicate's contract derived from the INPUT SPACE it accepts —
    (brief project present / absent / blank) x (page project same / different /
    absent / unparseable / query-string-only / case-shifted / non-ASCII) — not
    from the two example URLs the E4 wave happened to use.

    The asymmetry is the whole point: an absent PAGE project is a mismatch
    (nothing proves the page is the brief's project); an absent BRIEF project
    is not (there is nothing to violate).
    """
    from linkedin.orchestrator import (
        _is_foreign_project_page,
        _recruiter_page_project_id,
    )

    same = "2079249138"
    other = "9999999999"

    # --- id extraction: only the PATH names the project the page belongs to ---
    assert _recruiter_page_project_id(
        f"https://www.linkedin.com/talent/hire/{same}/discover/recruiterSearch"
    ) == same
    assert _recruiter_page_project_id(
        f"https://www.linkedin.com/talent/hire/{same}?foo=bar"
    ) == same
    assert _recruiter_page_project_id("https://www.linkedin.com/talent/search") is None
    assert _recruiter_page_project_id("https://www.linkedin.com/talent/hire//x") is None
    assert _recruiter_page_project_id(None) is None
    # A /talent/hire/<id> that lives in the QUERY STRING is a redirect target,
    # not the page's project — reading one would let a projectless page claim
    # the brief's project and re-open the bypass from the other direction.
    assert _recruiter_page_project_id(
        f"https://www.linkedin.com/talent/search?redirect=/talent/hire/{same}/discover"
    ) is None

    verified = [
        f"https://www.linkedin.com/talent/hire/{same}/discover/recruiterSearch",
        f"https://www.linkedin.com/talent/hire/{same}/discover/recruiterSearch"
        "/profile/AEMAAA?start=25",
        f"https://www.linkedin.com/talent/hire/{same}",
        f"https://www.linkedin.com/talent/hire/{same}?project={same}",
        f"https://www.linkedin.com/talent/hire/{same}#tab",
    ]
    unverified = [
        # different project — the case E4 already handled
        f"https://www.linkedin.com/talent/hire/{other}/discover/recruiterSearch",
        # no project id at all — the case E4 waved through
        "https://www.linkedin.com/talent/search",
        "https://www.linkedin.com/talent/search?x=1",
        "https://www.linkedin.com/talent/profile/AEMAAA",
        "https://www.linkedin.com/talent/recruiterSearch/profile/AEMAAA",
        "https://www.linkedin.com/feed/",
        "about:blank",
        "",
        None,
        # unparseable / decorative project segments
        "https://www.linkedin.com/talent/hire//discover/recruiterSearch",
        f"https://www.linkedin.com/talent/search?redirect=/talent/hire/{same}/x",
        # no percent-decoding: an encoded id is not proof of the same id
        "https://www.linkedin.com/talent/hire/2079%32%33/discover",
    ]

    for url in verified:
        assert _is_foreign_project_page(url, same) is False, url
        assert _is_foreign_project_page(url, int(same)) is False, url
        assert _is_foreign_project_page(url, f"  {same}  ") is False, url
    for url in unverified:
        assert _is_foreign_project_page(url, same) is True, url
        assert _is_foreign_project_page(url, int(same)) is True, url

    # Comparison is exact: case-shifted and non-ASCII ids fail closed.
    assert _is_foreign_project_page(
        "https://www.linkedin.com/talent/hire/TEST-PROJECT/discover", "test-project"
    ) is True
    assert _is_foreign_project_page(
        "https://www.linkedin.com/talent/hire/プロジェクト/discover", "プロジェクト"
    ) is False
    assert _is_foreign_project_page(
        "https://www.linkedin.com/talent/hire/项目/discover", "プロジェクト"
    ) is True

    # An absent BRIEF project keeps the permissive carve-out for every page.
    for expected in (None, "", "   ", 0, False):
        for url in verified + unverified:
            assert _is_foreign_project_page(url, expected) is False, (expected, url)


def test_run_full_disconnect_failure_is_abnormal():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock(side_effect=RuntimeError("disconnect failed"))
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()
        Path(td, "worker.json").write_text(json.dumps({"pid": os.getpid()}))

        async def process_with_save(search_string, _progress):
            _seed_runtime_judge_decisions(p, ["SAVE"])
            search_string.saves.append("candidate-save")

        p._process_string = process_with_save

        with pytest.raises(RuntimeError, match="disconnect failed"):
            asyncio.run(p.run_full(resume=True))

        p.browser.disconnect.assert_awaited_once()
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()
        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] is None
        assert run["ended_at"]
        assert not Path(td, "worker.json").exists()
        p._runtime_lock.acquire()
        p._runtime_lock.release()


def test_run_full_cleans_up_before_interruptible_market_intel_enrichment():
    """Freeze the run while locked, then release run resources before enrichment."""

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        Path(td, "worker.json").write_text(json.dumps({"pid": os.getpid()}))
        p.browser.connect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()

        order: list[str] = []
        frozen_run_dir = Path(td, "frozen-run")
        original_clear = p._clear_worker_sidecar_if_current
        original_release = p._runtime_lock.release

        def freeze_snapshot() -> Path:
            assert p._runtime_lock._handle is not None
            order.append("freeze_snapshot")
            return frozen_run_dir

        def clear_sidecar() -> None:
            order.append("clear_sidecar")
            original_clear()

        def release_lock() -> None:
            order.append("release_lock")
            original_release()

        async def disconnect_browser() -> None:
            order.append("disconnect_browser")

        def interrupt_enrichment(run_dir: Path) -> None:
            assert run_dir == frozen_run_dir
            assert p._runtime_lock._handle is None
            assert not Path(td, "worker.json").exists()
            run = p._runtime_state.get_run(p._runtime_run_id)
            assert run["ended_at"]
            order.append("enrich_market_intel")
            raise KeyboardInterrupt()

        p._finalize_run_snapshot = MagicMock(side_effect=freeze_snapshot)
        p._clear_worker_sidecar_if_current = MagicMock(side_effect=clear_sidecar)
        p._runtime_lock.release = MagicMock(side_effect=release_lock)
        p.browser.disconnect = AsyncMock(side_effect=disconnect_browser)
        p._enrich_run_snapshot = MagicMock(side_effect=interrupt_enrichment)

        async def process_with_save(search_string, _progress):
            _seed_runtime_judge_decisions(p, ["SAVE"])
            search_string.saves.append("candidate-save")

        p._process_string = process_with_save

        with pytest.raises(KeyboardInterrupt):
            asyncio.run(p.run_full(resume=True))

        assert order == [
            "freeze_snapshot",
            "clear_sidecar",
            "release_lock",
            "disconnect_browser",
            "enrich_market_intel",
        ]
        p.browser.disconnect.assert_awaited_once()
        pipeline_end_events = [
            row for row in read_jsonl(p.log_path) if row.get("event") == "pipeline_end"
        ]
        assert len(pipeline_end_events) == 1
        p._runtime_lock.release = original_release
        p._runtime_lock.acquire()
        p._runtime_lock.release()



def test_bounded_run_full_skips_paid_debrief_after_page_cap_stop():
    """The one-page GLM canary must end with judgment receipts, not a new
    unmeasured Standard-model report call after its usage session closes."""

    from shared.governor import OperatorStopRequested

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()

        async def stop_at_page_cap(_search_string, _progress):
            raise OperatorStopRequested("total_page_cap_reached")

        p._process_string = stop_at_page_cap
        with patch("linkedin.orchestrator.config.LINKEDIN_TOTAL_PAGE_CAP", 1):
            with pytest.raises(OperatorStopRequested):
                asyncio.run(p.run_full(resume=True))

        p._generate_run_report.assert_not_called()
        p._enrich_run_snapshot.assert_not_called()
        p._finalize_run_snapshot.assert_called_once()
        events = [
            row.get("event")
            for row in read_jsonl(p.log_path)
            if row.get("event") in {
                "report_started",
                "run_report_generated",
                "run_report_error",
            }
        ]
        assert events == []


def test_run_full_runtime_error_freezes_diagnostics_without_post_run_models():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [
                SearchString(id=1, name="one", boolean="a", status="queued"),
                SearchString(id=2, name="two", boolean="b", status="queued"),
            ],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._process_string = AsyncMock(side_effect=RuntimeError("string failed"))
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()

        with pytest.raises(RuntimeError, match="string failed"):
            asyncio.run(p.run_full(resume=True))

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] is None
        assert run["ended_at"]
        assert p._progress.strings[0].status == "in_progress"
        assert p._progress.strings[1].status == "queued"
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()


def test_run_full_connect_abort_closes_resumed_run_and_freezes_diagnostics():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="in_progress")],
        )
        p.browser.connect = AsyncMock(side_effect=RuntimeError("connect failed"))
        p.browser.disconnect = AsyncMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()
        p._generate_run_report = MagicMock()

        with pytest.raises(RuntimeError, match="connect failed"):
            asyncio.run(p.run_full(resume=True))

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["ended_at"]
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()
        p.browser.disconnect.assert_awaited_once()
        p._runtime_lock.acquire()
        p._runtime_lock.release()


def test_run_full_fresh_connect_abort_starts_and_closes_canonical_run():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.connect = AsyncMock(side_effect=RuntimeError("fresh connect failed"))
        p.browser.disconnect = AsyncMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()
        p._generate_run_report = MagicMock()

        with pytest.raises(RuntimeError, match="fresh connect failed"):
            asyncio.run(p.run_full(resume=False))

        assert p._runtime_run_id is not None
        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["mode"] == "fresh"
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["ended_at"]
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()
        p.browser.disconnect.assert_awaited_once()
        p._runtime_lock.acquire()
        p._runtime_lock.release()


def test_run_full_fresh_sync_abort_recovers_new_run_id_and_freezes_diagnostics():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()
        p._generate_run_report = MagicMock()

        with patch(
            "shared.runtime_state.linkedin.sync_linkedin_progress",
            side_effect=RuntimeError("initial progress sync failed"),
        ):
            with pytest.raises(RuntimeError, match="initial progress sync failed"):
                asyncio.run(p.run_full(resume=False))

        latest_run = p._runtime_state.get_latest_run(
            source="linkedin",
            brief_id=p.brief_obj.linkedin_project_id,
        )
        assert latest_run["id"] == p._runtime_run_id
        assert latest_run["mode"] == "fresh"
        assert latest_run["status"] == "error"
        assert latest_run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert latest_run["ended_at"]
        p.browser.connect.assert_not_awaited()
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()
        p.browser.disconnect.assert_awaited_once()
        p._runtime_lock.acquire()
        p._runtime_lock.release()


def test_run_full_resume_rebuild_abort_recovers_new_run_id_and_freezes_diagnostics():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        seed = _make_pipeline(td)
        initial_progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1,
                    name="owner",
                    boolean="one",
                    status="in_progress",
                )
            ],
        )
        prior_run_id, _ = seed._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=initial_progress,
        )

        p = _make_pipeline(td)
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()
        p._generate_run_report = MagicMock()

        with patch(
            "shared.runtime_state.linkedin.rebuild_linkedin_artifacts",
            side_effect=RuntimeError("resume artifact rebuild failed"),
        ):
            with pytest.raises(RuntimeError, match="resume artifact rebuild failed"):
                asyncio.run(p.run_full(resume=True))

        latest_run = p._runtime_state.get_latest_run(
            source="linkedin",
            brief_id=p.brief_obj.linkedin_project_id,
        )
        assert latest_run["id"] == p._runtime_run_id
        assert latest_run["id"] != prior_run_id
        assert latest_run["resumed_from_run_id"] == prior_run_id
        assert latest_run["mode"] == "resume"
        assert latest_run["status"] == "error"
        assert latest_run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert latest_run["ended_at"]
        p.browser.connect.assert_not_awaited()
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()
        p.browser.disconnect.assert_awaited_once()
        p._runtime_lock.acquire()
        p._runtime_lock.release()


def test_run_full_uncaught_base_exception_is_abnormal_and_skips_paid_work():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        primary_error = SystemExit("synthetic process exit")
        p._process_string = AsyncMock(side_effect=primary_error)
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()

        with pytest.raises(SystemExit) as captured:
            asyncio.run(p.run_full(resume=True))

        assert captured.value is primary_error
        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["ended_at"]
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()


def test_run_full_secondary_checkpoint_failure_cannot_block_canonical_finish():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(id=1, name="owner", boolean="one"),
                SearchString(id=2, name="later", boolean="two"),
            ],
        )
        progress.save(str(p.progress_path))
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._load_candidate_history = MagicMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()
        p._bias_monitor = MagicMock()
        p._bias_monitor.save_checkpoint.side_effect = OSError(
            "secondary bias checkpoint failure"
        )
        p._process_string = AsyncMock(
            side_effect=RuntimeError("primary string failure")
        )

        with pytest.raises(RuntimeError, match="primary string failure"):
            asyncio.run(p.run_full(resume=True))

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["ended_at"]
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()


def test_run_full_final_checkpoint_failure_is_abnormal():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()

        processing_finished = False
        original_checkpoint = p._checkpoint_progress

        def checkpoint(progress, search_string=None, page_num=None, **kwargs):
            if processing_finished and search_string is None:
                raise OSError("final checkpoint failed")
            return original_checkpoint(
                progress,
                search_string=search_string,
                page_num=page_num,
                **kwargs,
            )

        async def process_with_save(search_string, _progress):
            nonlocal processing_finished
            _seed_runtime_judge_decisions(p, ["SAVE"])
            search_string.saves.append("candidate-save")
            processing_finished = True

        p._checkpoint_progress = MagicMock(side_effect=checkpoint)
        p._process_string = process_with_save

        with pytest.raises(OSError, match="final checkpoint failed"):
            asyncio.run(p.run_full(resume=True))

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()


def test_run_full_snapshot_exception_refinalizes_canonical_run():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        snapshot_error = RuntimeError("snapshot exploded")
        p._finalize_run_snapshot = MagicMock(side_effect=snapshot_error)
        p._enrich_run_snapshot = MagicMock()
        original_finish = p._safety.finish_run
        p._safety.finish_run = MagicMock(wraps=original_finish)

        async def process_with_save(search_string, _progress):
            _seed_runtime_judge_decisions(p, ["SAVE"])
            search_string.saves.append("candidate-save")

        p._process_string = process_with_save

        with pytest.raises(RuntimeError) as captured:
            asyncio.run(p.run_full(resume=True))

        assert captured.value is snapshot_error
        assert [
            call.kwargs["status"]
            for call in p._safety.finish_run.call_args_list
        ] == ["completed", "error"]
        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()


def test_run_full_snapshot_none_refinalizes_canonical_run():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._finalize_run_snapshot = MagicMock(return_value=None)
        p._enrich_run_snapshot = MagicMock()
        original_finish = p._safety.finish_run
        p._safety.finish_run = MagicMock(wraps=original_finish)

        async def process_with_save(search_string, _progress):
            _seed_runtime_judge_decisions(p, ["SAVE"])
            search_string.saves.append("candidate-save")

        p._process_string = process_with_save

        with pytest.raises(
            RuntimeError,
            match="immutable LinkedIn run snapshot did not return a path",
        ):
            asyncio.run(p.run_full(resume=True))

        assert [
            call.kwargs["status"]
            for call in p._safety.finish_run.call_args_list
        ] == ["completed", "error"]
        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()


def test_run_full_healthy_drain_keeps_completed_normal():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()

        async def process_with_save(search_string, _progress):
            _seed_runtime_judge_decisions(p, ["SAVE"])
            search_string.saves.append("candidate-save")

        p._process_string = process_with_save

        asyncio.run(p.run_full(resume=True))

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "completed"
        assert run["stop_reason"] == RunStopReason.NORMAL
        assert run["stop_reason_detail"] is None
        assert run["ended_at"]
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_called_once_with(frozen_run_dir)
        p._generate_run_report.assert_called_once_with(p._progress)


def test_run_full_late_diagnostic_failures_do_not_invalidate_completion():
    from linkedin import orchestrator as orchestrator_module
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock(
            side_effect=RuntimeError("report failed")
        )
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()
        real_log_event = orchestrator_module.log_event

        async def process_with_save(search_string, _progress):
            _seed_runtime_judge_decisions(p, ["SAVE"])
            search_string.saves.append("candidate-save")

        def fail_late_diagnostic_events(path, event, **payload):
            if event in {"run_report_error", "pipeline_end"}:
                raise OSError(f"{event} write failed")
            return real_log_event(path, event, **payload)

        p._process_string = process_with_save
        with patch(
            "linkedin.orchestrator.log_event",
            side_effect=fail_late_diagnostic_events,
        ):
            asyncio.run(p.run_full(resume=True))

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "completed"
        assert run["stop_reason"] == RunStopReason.NORMAL
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_called_once_with(frozen_run_dir)
        p._generate_run_report.assert_called_once_with(p._progress)


def test_run_full_preserves_skipped_status_from_process_string():
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()

        async def skip_mid_string(search_string, _progress):
            search_string.status = "skipped"
            search_string.notes = "Skipped: no results on entry."

        p._process_string = skip_mid_string

        asyncio.run(p.run_full(resume=True))

        assert p._progress.strings[0].status == "skipped"
        saved = json.loads(Path(td, "progress.json").read_text())
        assert saved["strings"][0]["status"] == "skipped"
        p._enrich_run_snapshot.assert_called_once_with(frozen_run_dir)
        p._generate_run_report.assert_called_once_with(p._progress)


def test_run_full_drained_green_but_useless_must_not_finalize_completed():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="done")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._process_string = AsyncMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()

        with pytest.raises(RuntimeError, match="LinkedIn completion rejected"):
            asyncio.run(p.run_full(resume=True))

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] == "fatal_runtime_error: green_but_useless"
        assert run["ended_at"]
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()


def test_run_full_honesty_gate_failure_is_abnormal():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="done")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._process_string = AsyncMock()
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        frozen_run_dir = Path(td, "frozen-run")
        p._finalize_run_snapshot = MagicMock(return_value=frozen_run_dir)
        p._enrich_run_snapshot = MagicMock()
        p._run_health_summary = MagicMock(side_effect=RuntimeError("health exploded"))

        with pytest.raises(RuntimeError, match="health exploded"):
            asyncio.run(p.run_full(resume=True))

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "error"
        assert run["stop_reason"] == RunStopReason.FATAL_RUNTIME_ERROR
        assert run["stop_reason_detail"] is None
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()


def test_run_full_session_expired_status_is_not_rewritten_by_honesty_gate():
    from shared.governor import SessionExpired
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._finalize_run_snapshot = MagicMock(
            return_value=Path(td, "frozen-run")
        )
        p._enrich_run_snapshot = MagicMock()
        p._run_health_summary = MagicMock(side_effect=AssertionError("must not run"))

        async def expire_mid_string(_search_string, _progress):
            raise SessionExpired("session_duration_cap")

        p._process_string = expire_mid_string

        with pytest.raises(SessionExpired):
            asyncio.run(p.run_full(resume=True))

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "interrupted"
        assert run["stop_reason"] == RunStopReason.SESSION_EXPIRED
        assert run["stop_reason_detail"] is None
        p._run_health_summary.assert_not_called()
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()


def test_run_full_cancelled_error_finalizes_interrupted_operator_stop():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._finalize_run_snapshot = MagicMock(
            return_value=Path(td, "frozen-run")
        )
        p._enrich_run_snapshot = MagicMock()

        async def cancel_mid_string(_search_string, _progress):
            raise asyncio.CancelledError()

        p._process_string = cancel_mid_string

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(p.run_full(resume=True))

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "interrupted"
        assert run["stop_reason"] == RunStopReason.OPERATOR_STOP
        assert run["stop_reason_detail"] is None
        assert run["ended_at"]
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()


def test_run_full_keyboard_interrupt_finalizes_interrupted_operator_stop():
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [SearchString(id=1, name="one", boolean="a", status="queued")],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._finalize_run_snapshot = MagicMock(
            return_value=Path(td, "frozen-run")
        )
        p._enrich_run_snapshot = MagicMock()

        async def interrupt_mid_string(_search_string, _progress):
            raise KeyboardInterrupt()

        p._process_string = interrupt_mid_string

        with pytest.raises(KeyboardInterrupt):
            asyncio.run(p.run_full(resume=True))

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert run["status"] == "interrupted"
        assert run["stop_reason"] == RunStopReason.OPERATOR_STOP
        assert run["stop_reason_detail"] is None
        assert run["ended_at"]
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()


def test_run_full_operator_stop_event_before_second_string_interrupts_run():
    from shared.governor import OperatorStopRequested
    from shared.safety import RunStopReason

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        _seed_run_full_resume_progress(
            p,
            [
                SearchString(id=1, name="one", boolean="a", status="queued"),
                SearchString(id=2, name="two", boolean="b", status="queued"),
            ],
        )
        p.browser.connect = AsyncMock()
        p.browser.disconnect = AsyncMock()
        p.browser.page = MagicMock(
            url="https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
        )
        p._print_session_summary = MagicMock()
        p._print_summary = MagicMock()
        p._generate_run_report = MagicMock()
        p._finalize_run_snapshot = MagicMock(
            return_value=Path(td, "frozen-run")
        )
        p._enrich_run_snapshot = MagicMock()
        p._run_health_summary = MagicMock(side_effect=AssertionError("must not run"))
        processed: list[int] = []

        async def process_then_stop(search_string, _progress):
            processed.append(search_string.id)
            if search_string.id == 1:
                p._operator_stop_event.set()

        async def drive():
            p._operator_stop_event = asyncio.Event()
            p._process_string = process_then_stop
            await p.run_full(resume=True)

        with pytest.raises(OperatorStopRequested):
            asyncio.run(drive())

        run = p._runtime_state.get_run(p._runtime_run_id)
        assert processed == [1]
        assert run["status"] == "interrupted"
        assert run["stop_reason"] == RunStopReason.OPERATOR_STOP
        assert run["stop_reason_detail"] is None
        assert run["ended_at"]
        p._run_health_summary.assert_not_called()
        p._finalize_run_snapshot.assert_called_once_with()
        p._enrich_run_snapshot.assert_not_called()
        p._generate_run_report.assert_not_called()
