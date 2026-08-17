"""Tests for the LinkedIn lane compiler adapter — P9/C2."""

from __future__ import annotations

from linkedin.lane_compiler import LinkedInLaneCompiler
from shared.lane_compilers import LaneCompiler
from shared.sourcing_lanes import (
    LaneExecution,
    LaneVariant,
    SearchConstraint,
    SearchHypothesis,
    SearchSlice,
    SourcingLane,
)


def _make_lane(
    *,
    lane_id: str = "li-lane-1",
    acquisition_mode: str = "linkedin_boolean",
    constraints: list[SearchConstraint] | None = None,
    boolean_strategy: dict | None = None,
    structured_filters: dict | None = None,
) -> SourcingLane:
    return SourcingLane(
        lane_id=lane_id,
        lane_name="LinkedIn Test Lane",
        hypothesis=SearchHypothesis(
            hypothesis_id="h1",
            label="Test",
            target_archetype="engineer",
            why_this_pool_may_exist="testing",
        ),
        slice=SearchSlice(
            slice_id="s1",
            hypothesis_id="h1",
            label="Test Slice",
            objective="test",
            constraints=constraints or [],
        ),
        execution=LaneExecution(
            lane_id=lane_id,
            source="linkedin",
            acquisition_mode=acquisition_mode,
            boolean_strategy=boolean_strategy or {},
            structured_filters=structured_filters or {},
        ),
    )


def _make_variant(
    *,
    variant_id: str = "v1",
    lane_id: str = "li-lane-1",
    boolean_intent: str = '"ML" AND "engineer"',
    structured_controls: dict | None = None,
) -> LaneVariant:
    return LaneVariant(
        variant_id=variant_id,
        lane_id=lane_id,
        boolean_intent=boolean_intent,
        structured_controls=structured_controls or {},
    )


def test_compiler_satisfies_protocol():
    compiler = LinkedInLaneCompiler()
    assert isinstance(compiler, LaneCompiler)


def test_hybrid_lane_compiles_to_boolean_plus_structured():
    compiler = LinkedInLaneCompiler()
    lane = _make_lane(acquisition_mode="linkedin_hybrid")
    variant = _make_variant(
        boolean_intent='"ML" AND "engineer"',
        structured_controls={"titles": ["ML Engineer"], "companies": ["Google"]},
    )
    exe = compiler.compile(lane, variant)
    assert exe.source == "linkedin"
    assert exe.acquisition_mode == "linkedin_hybrid"
    payload = exe.query_payload
    assert payload["boolean"] == '"ML" AND "engineer"'
    assert payload["structured_filters"]["titles"] == ["ML Engineer"]
    assert payload["structured_filters"]["companies"] == ["Google"]
    assert "advanced_search_plan" in payload


def test_boolean_only_lane_compiles_without_structured():
    compiler = LinkedInLaneCompiler()
    lane = _make_lane(
        acquisition_mode="linkedin_boolean",
        boolean_strategy={"root_boolean": '"Python" AND "backend"'},
    )
    exe = compiler.compile(lane)
    payload = exe.query_payload
    assert payload["boolean"] == '"Python" AND "backend"'
    sf = payload["structured_filters"]
    assert sf.get("titles", []) == []
    assert sf.get("companies", []) == []


def test_lane_compiler_builds_boolean_from_boolean_surface_constraints():
    compiler = LinkedInLaneCompiler()
    lane = _make_lane(
        acquisition_mode="linkedin_boolean",
        constraints=[
            SearchConstraint(
                dimension="entry_signal",
                values=["engineer", "engineering"],
                execution_surface="boolean_keyword",
                operator="prefer",
            ),
            SearchConstraint(
                dimension="capability",
                values=["LLM", "generative AI"],
                execution_surface="boolean_keyword",
                operator="prefer",
            ),
            SearchConstraint(
                dimension="context",
                values=["Palantir Technologies"],
                execution_surface="linkedin_company_filter",
                operator="prefer",
            ),
        ],
        structured_filters={
            "target_employers": ["Palantir Technologies", "Scale AI"],
        },
    )

    exe = compiler.compile(lane)

    payload = exe.query_payload
    assert payload["boolean"] == (
        '("engineer" OR "engineering") AND ("LLM" OR "generative AI")'
    )
    assert exe.acquisition_mode == "linkedin_hybrid"
    assert payload["structured_filters"]["companies"] == [
        "Palantir Technologies",
        "Scale AI",
    ]


def test_company_filter_seeded_from_target_employers_warns():
    """Fix 3: when the company filter is seeded from target_employers (no reasoned
    `companies` key), emit the narrowing warning. The structured filter can be narrower
    than — and diverge from — the companies named in the keyword Boolean, with no other
    signal to the recruiter.
    """
    compiler = LinkedInLaneCompiler()
    lane = _make_lane(
        acquisition_mode="linkedin_boolean",
        boolean_strategy={
            "root_boolean": '("Nubank" OR "Bancolombia" OR "Davivienda")'
        },
        structured_filters={"target_employers": ["Nubank", "Bancolombia"]},
    )
    findings = compiler.lint(lane)
    codes = [f.code for f in findings]
    assert "structured_filter_seeded_from_target_employers" in codes, codes
    # The seeded employers still force hybrid acquisition (filter is non-empty).
    assert compiler.compile(lane).acquisition_mode == "linkedin_hybrid"


def test_reasoned_companies_key_does_not_warn():
    """No narrowing warning when `companies` is reasoned explicitly — the filter is
    intentional, not a silent target_employers fallback.
    """
    compiler = LinkedInLaneCompiler()
    lane = _make_lane(
        acquisition_mode="linkedin_hybrid",
        structured_filters={
            "companies": ["Nubank"],
            "target_employers": ["Nubank", "Bancolombia"],
        },
    )
    codes = [f.code for f in compiler.lint(lane)]
    assert "structured_filter_seeded_from_target_employers" not in codes, codes


def test_no_target_employers_does_not_warn():
    """No narrowing warning for a plain boolean lane with neither companies nor
    target_employers (byte-identical default path).
    """
    compiler = LinkedInLaneCompiler()
    lane = _make_lane(
        acquisition_mode="linkedin_boolean",
        boolean_strategy={"root_boolean": '"Python"'},
    )
    codes = [f.code for f in compiler.lint(lane)]
    assert "structured_filter_seeded_from_target_employers" not in codes, codes


def test_unsupported_controls_produce_findings():
    compiler = LinkedInLaneCompiler()
    lane = _make_lane(acquisition_mode="linkedin_hybrid")
    # fields_of_study is the remaining mock_only control after slice H graduated
    # job_titles + companies (and locations is stable_now), so it is the dimension
    # that still produces an unsupported_control finding.
    variant = _make_variant(
        structured_controls={
            "advanced_filters": {"fields_of_study": ["Computer Science"]},
            "sidebar_filters": {"locations": ["New York"]},
        },
    )
    findings = compiler.lint(lane, variant)
    unsupported_codes = [f.code for f in findings if f.code == "unsupported_control"]
    unsupported_dims = [f.dimension for f in findings if f.code == "unsupported_control"]
    assert len(unsupported_codes) > 0
    assert "fields_of_study" in unsupported_dims


def test_duplicate_title_semantics_raise_warning():
    compiler = LinkedInLaneCompiler()
    lane = _make_lane(acquisition_mode="linkedin_hybrid")
    variant = _make_variant(
        boolean_intent='"Engineer" AND "Python"',
        structured_controls={"titles": ["Engineer"]},
    )
    findings = compiler.lint(lane, variant)
    conflict_findings = [f for f in findings if f.code == "boolean_filter_conflict"]
    assert len(conflict_findings) >= 1
    assert any("Engineer" in f.message for f in conflict_findings)


def test_outputs_carry_lane_and_variant_id():
    compiler = LinkedInLaneCompiler()
    lane = _make_lane(lane_id="lane-42")
    variant = _make_variant(variant_id="var-7", lane_id="lane-42")
    exe = compiler.compile(lane, variant)
    assert exe.lane_id == "lane-42"
    assert exe.variant_id == "var-7"


def test_boolean_fallback_preserves_identity():
    compiler = LinkedInLaneCompiler()
    lane = _make_lane(
        lane_id="fallback-lane",
        acquisition_mode="linkedin_boolean",
        boolean_strategy={"root_boolean": '"test"'},
    )
    exe = compiler.compile(lane)
    assert exe.lane_id == "fallback-lane"
    assert exe.variant_id == ""
    assert exe.source == "linkedin"
