"""Tests for the shared lane compiler contract — P9."""

from __future__ import annotations

from pathlib import Path

from shared.lane_compilers import (
    ExecutableSearch,
    LaneCompiler,
    LaneCompilerFinding,
)
from shared.sourcing_lanes import (
    EXECUTION_SURFACES,
    LaneExecution,
    LaneVariant,
    SearchConstraint,
    SearchHypothesis,
    SearchSlice,
    SourcingLane,
)


def _make_lane(
    *,
    lane_id: str = "test-lane",
    source: str = "linkedin",
    acquisition_mode: str = "linkedin_boolean",
    constraints: list[SearchConstraint] | None = None,
) -> SourcingLane:
    return SourcingLane(
        lane_id=lane_id,
        lane_name="Test Lane",
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
            source=source,
            acquisition_mode=acquisition_mode,
        ),
    )


class _StubCompiler:
    source: str = "stub"

    def compile(
        self,
        lane: SourcingLane,
        variant: LaneVariant | None = None,
    ) -> ExecutableSearch:
        return ExecutableSearch(
            source=self.source,
            acquisition_mode="stub_mode",
            display_name=lane.lane_name,
            query_payload={"stub_key": "stub_value"},
            lane_id=lane.lane_id,
            variant_id=variant.variant_id if variant else "",
        )

    def lint(
        self,
        lane: SourcingLane,
        variant: LaneVariant | None = None,
    ) -> list[LaneCompilerFinding]:
        return []


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


def test_protocol_structural_subtyping():
    compiler = _StubCompiler()
    assert isinstance(compiler, LaneCompiler)


def test_executable_search_carries_source_native_payload():
    payload = {"nested": {"deep": [1, 2, 3]}, "opaque": True}
    exe = ExecutableSearch(
        source="test",
        acquisition_mode="test_mode",
        display_name="Test",
        query_payload=payload,
    )
    d = exe.to_dict()
    assert d["query_payload"]["nested"]["deep"] == [1, 2, 3]
    assert d["query_payload"]["opaque"] is True


def test_compiler_finding_severity_levels():
    for severity in ("info", "warning", "error"):
        finding = LaneCompilerFinding(
            code=f"test_{severity}",
            severity=severity,  # type: ignore[arg-type]
            message=f"A {severity} finding",
            dimension="title",
            source="test",
        )
        d = finding.to_dict()
        assert d["code"] == f"test_{severity}"
        assert d["severity"] == severity
        assert d["dimension"] == "title"
        assert d["source"] == "test"


def test_unsupported_dimensions_are_structured():
    exe = ExecutableSearch(
        source="test",
        acquisition_mode="test_mode",
        display_name="Test",
        unsupported_dimensions=("seniority", "years_of_experience"),
        warnings=(
            LaneCompilerFinding(
                code="unsupported_dimension",
                severity="warning",
                message="seniority not supported",
                dimension="seniority",
                source="test",
            ),
        ),
    )
    assert len(exe.unsupported_dimensions) == 2
    assert exe.warnings[0].code == "unsupported_dimension"
    assert exe.warnings[0].dimension == "seniority"


def test_no_source_imports_in_shared_module():
    source = Path("shared/lane_compilers.py").read_text()
    for keyword in ("linkedin", "github", "researcher", "designer", "exec_search"):
        assert keyword not in source.lower(), (
            f"shared/lane_compilers.py must not reference source-specific "
            f"keyword '{keyword}'"
        )


def test_stub_compiler_compiles_lane():
    compiler = _StubCompiler()
    lane = _make_lane()
    exe = compiler.compile(lane)
    assert exe.source == "stub"
    assert exe.lane_id == "test-lane"
    assert exe.query_payload["stub_key"] == "stub_value"


def test_source_native_in_execution_surfaces():
    assert "source_native" in EXECUTION_SURFACES


# ---------------------------------------------------------------------------
# Cross-source parametrized tests (require C2 + C3)
# ---------------------------------------------------------------------------

import pytest
from github.lane_compiler import GitHubLaneCompiler
from linkedin.lane_compiler import LinkedInLaneCompiler


_COMPILERS = [LinkedInLaneCompiler(), GitHubLaneCompiler()]


@pytest.mark.parametrize("compiler", _COMPILERS, ids=lambda c: c.source)
def test_shared_protocol_parametrized(compiler):
    assert isinstance(compiler, LaneCompiler)
    lane = _make_lane()
    exe = compiler.compile(lane)
    assert exe.source == compiler.source
    findings = compiler.lint(lane)
    assert isinstance(findings, list)


def test_same_lane_compiles_for_both_sources():
    lane = _make_lane(
        constraints=[
            SearchConstraint(dimension="language", values=["python"], execution_surface="source_native"),
            SearchConstraint(dimension="title", values=["Engineer"], execution_surface="linkedin_title_filter"),
            SearchConstraint(dimension="topic", values=["rlhf"], execution_surface="source_native"),
        ],
    )
    li = LinkedInLaneCompiler().compile(lane)
    gh = GitHubLaneCompiler().compile(lane)

    # LinkedIn can handle title; GitHub cannot
    assert "title" not in li.unsupported_dimensions
    assert "title" in gh.unsupported_dimensions

    # GitHub can handle topic natively; LinkedIn treats it as soft_hint or unsupported
    assert gh.query_payload["channels"].get("topic_search")

    # Both carry lane identity
    assert li.lane_id == lane.lane_id
    assert gh.lane_id == lane.lane_id
