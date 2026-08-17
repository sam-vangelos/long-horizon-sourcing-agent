"""Tests for the GitHub lane compiler adapter — P9/C3."""

from __future__ import annotations

from pathlib import Path

from github.lane_compiler import GitHubLaneCompiler
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
    lane_id: str = "gh-lane-1",
    constraints: list[SearchConstraint] | None = None,
) -> SourcingLane:
    return SourcingLane(
        lane_id=lane_id,
        lane_name="GitHub Test Lane",
        hypothesis=SearchHypothesis(
            hypothesis_id="h1",
            label="OSS Maintainers",
            target_archetype="oss_maintainer",
            why_this_pool_may_exist="active contributors to key repos",
        ),
        slice=SearchSlice(
            slice_id="s1",
            hypothesis_id="h1",
            label="OSS Slice",
            objective="find maintainers",
            constraints=constraints or [],
        ),
        execution=LaneExecution(
            lane_id=lane_id,
            source="github",
            acquisition_mode="github",
        ),
    )


def test_github_compiler_satisfies_protocol():
    compiler = GitHubLaneCompiler()
    assert isinstance(compiler, LaneCompiler)


def test_github_accepts_oss_maintainer_lane():
    compiler = GitHubLaneCompiler()
    lane = _make_lane(
        constraints=[
            SearchConstraint(dimension="language", values=["python"], execution_surface="source_native"),
            SearchConstraint(dimension="topic", values=["reinforcement-learning"], execution_surface="source_native"),
            SearchConstraint(dimension="organization", values=["OpenAI"], execution_surface="source_native"),
        ],
    )
    exe = compiler.compile(lane)
    channels = exe.query_payload["channels"]
    assert "user_search" in channels
    assert "topic_search" in channels
    assert "org_exploration" in channels
    assert any("language:python" in frag for frag in channels["user_search"])
    assert any("topic:reinforcement-learning" in frag for frag in channels["topic_search"])
    assert any("org:OpenAI" in frag for frag in channels["org_exploration"])


def test_github_rejects_linkedin_only_dimensions():
    compiler = GitHubLaneCompiler()
    lane = _make_lane(
        constraints=[
            SearchConstraint(dimension="title", values=["Engineer"], execution_surface="source_native"),
            SearchConstraint(dimension="current_company", values=["Google"], execution_surface="source_native"),
            SearchConstraint(dimension="language", values=["python"], execution_surface="source_native"),
        ],
    )
    exe = compiler.compile(lane)
    assert "title" in exe.unsupported_dimensions
    assert "current_company" in exe.unsupported_dimensions
    assert exe.query_payload["channels"].get("user_search")


def test_github_lint_warns_on_linkedin_dims():
    compiler = GitHubLaneCompiler()
    lane = _make_lane(
        constraints=[
            SearchConstraint(dimension="seniority", values=["Senior"], execution_surface="source_native"),
        ],
    )
    findings = compiler.lint(lane)
    assert any(f.code == "unsupported_dimension" and f.dimension == "seniority" for f in findings)


def test_github_lint_warns_on_linkedin_surfaces():
    compiler = GitHubLaneCompiler()
    lane = _make_lane(
        constraints=[
            SearchConstraint(dimension="language", values=["python"], execution_surface="linkedin_title_filter"),
        ],
    )
    findings = compiler.lint(lane)
    assert any(f.code == "unsupported_surface" for f in findings)


def test_no_linkedin_types_imported():
    source = Path("github/lane_compiler.py").read_text()
    for keyword in ("LinkedInSearchVariant", "LinkedInStructuredFilters", "recruiter"):
        assert keyword not in source, (
            f"github/lane_compiler.py must not reference LinkedIn type '{keyword}'"
        )


def test_no_recruiter_save_semantics():
    compiler = GitHubLaneCompiler()
    lane = _make_lane(
        constraints=[
            SearchConstraint(dimension="language", values=["python"], execution_surface="source_native"),
        ],
    )
    exe = compiler.compile(lane)
    payload = exe.query_payload
    assert "save_eligible" not in payload
    assert "recruiter_profile_id" not in payload


def test_outputs_carry_lane_and_variant_id():
    compiler = GitHubLaneCompiler()
    lane = _make_lane(lane_id="gh-42")
    variant = LaneVariant(variant_id="v-3", lane_id="gh-42", boolean_intent="test")
    exe = compiler.compile(lane, variant)
    assert exe.lane_id == "gh-42"
    assert exe.variant_id == "v-3"
