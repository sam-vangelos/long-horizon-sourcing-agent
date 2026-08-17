"""GitHub lane compiler adapter — P9/C3.

Proves the LaneCompiler protocol generalizes to a non-LinkedIn source.
Maps lane constraints into GitHub-native search channels without importing
any LinkedIn or Recruiter types.
"""

from __future__ import annotations

from typing import Any

from shared.lane_compilers import ExecutableSearch, LaneCompilerFinding
from shared.sourcing_lanes import LaneVariant, SourcingLane

GITHUB_SUPPORTED_DIMENSIONS = frozenset({
    "language",
    "ecosystem",
    "dependency",
    "topic",
    "organization",
    "contribution_recency",
    "repository_criticality",
    "capability",
    "location",
})

LINKEDIN_ONLY_DIMENSIONS = frozenset({
    "title",
    "current_company",
    "seniority",
    "years_of_experience",
    "fields_of_study",
})

_DIMENSION_TO_CHANNEL: dict[str, str] = {
    "language": "user_search",
    "ecosystem": "code_search",
    "dependency": "code_search",
    "topic": "topic_search",
    "organization": "org_exploration",
    "contribution_recency": "user_search",
    "repository_criticality": "stargazer_mining",
    "capability": "user_search",
    "location": "user_search",
}


def _constraint_to_query_fragment(dimension: str, values: list[str]) -> str:
    if dimension == "language" and values:
        return f"language:{values[0]}"
    if dimension == "topic" and values:
        return f"topic:{values[0]}"
    if dimension == "organization" and values:
        return f"org:{values[0]}"
    if dimension == "location" and values:
        return f"location:{values[0]}"
    if dimension == "contribution_recency" and values:
        return f"pushed:>{values[0]}"
    if dimension in ("ecosystem", "dependency") and values:
        return " ".join(f'"{v}"' for v in values)
    if dimension == "capability" and values:
        return " ".join(f'"{v}"' for v in values)
    if dimension == "repository_criticality" and values:
        return values[0]
    return " ".join(values) if values else ""


class GitHubLaneCompiler:
    """Compile SourcingLane / LaneVariant into GitHub-native search intent."""

    source: str = "github"

    def compile(
        self,
        lane: SourcingLane,
        variant: LaneVariant | None = None,
    ) -> ExecutableSearch:
        channels: dict[str, list[str]] = {}
        applied: list[str] = []
        unsupported: list[str] = []
        warnings: list[LaneCompilerFinding] = []

        constraints = lane.slice.constraints
        if variant and variant.structured_controls:
            for dim, vals in variant.structured_controls.items():
                if isinstance(vals, list) and vals:
                    from shared.sourcing_lanes import SearchConstraint
                    constraints = list(constraints) + [
                        SearchConstraint(
                            dimension=dim,
                            values=vals,
                            execution_surface="source_native",
                        )
                    ]

        for constraint in constraints:
            dim = constraint.dimension
            if dim in LINKEDIN_ONLY_DIMENSIONS:
                unsupported.append(dim)
                warnings.append(
                    LaneCompilerFinding(
                        code="unsupported_dimension",
                        severity="warning",
                        message=f"'{dim}' is a LinkedIn-only dimension; GitHub cannot execute it",
                        dimension=dim,
                        source="github",
                    )
                )
                continue

            surface = constraint.execution_surface
            if surface.startswith("linkedin_"):
                unsupported.append(dim)
                warnings.append(
                    LaneCompilerFinding(
                        code="unsupported_surface",
                        severity="warning",
                        message=f"'{surface}' is a LinkedIn execution surface; GitHub ignores it",
                        dimension=dim,
                        source="github",
                    )
                )
                continue

            if dim in GITHUB_SUPPORTED_DIMENSIONS:
                channel = _DIMENSION_TO_CHANNEL.get(dim, "user_search")
                fragment = _constraint_to_query_fragment(dim, constraint.values)
                if fragment:
                    channels.setdefault(channel, []).append(fragment)
                    applied.append(dim)
            else:
                warnings.append(
                    LaneCompilerFinding(
                        code="unknown_dimension",
                        severity="info",
                        message=f"'{dim}' is not a known GitHub dimension; treated as soft hint",
                        dimension=dim,
                        source="github",
                    )
                )

        return ExecutableSearch(
            source="github",
            acquisition_mode="github",
            display_name=lane.lane_name,
            query_payload={
                "channels": channels,
                "constraints_applied": applied,
                "constraints_unsupported": unsupported,
            },
            lane_id=lane.lane_id,
            variant_id=variant.variant_id if variant else "",
            unsupported_dimensions=tuple(unsupported),
            warnings=tuple(warnings),
        )

    def lint(
        self,
        lane: SourcingLane,
        variant: LaneVariant | None = None,
    ) -> list[LaneCompilerFinding]:
        exe = self.compile(lane, variant)
        return list(exe.warnings)
