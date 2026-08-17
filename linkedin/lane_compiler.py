"""LinkedIn lane compiler adapter — P9/C2.

Wraps existing LinkedIn compilation functions behind the shared LaneCompiler
protocol.  Delegates to compile_lane_variant_to_linkedin(),
compile_structured_filters_to_plan(), and lint_boolean_filter_conflicts()
without reimplementing their logic.
"""

from __future__ import annotations

from typing import Any

from linkedin.advanced_search import (
    classify_control,
    compile_structured_filters_to_plan,
    lint_boolean_filter_conflicts,
    snapshot_controls_from_plan,
)
from linkedin.search_intelligence import (
    LinkedInStructuredFilters,
    compile_lane_variant_to_linkedin,
)
from shared.lane_compilers import ExecutableSearch, LaneCompilerFinding
from shared.sourcing_lanes import LaneVariant, SourcingLane

# Slice constraints whose execution_surface maps to a structured LinkedIn control
# (via compile_constraint) are folded into the compiled filters. Only positive
# operators become inclusion filters; an exclude on a structured surface is a
# later concern (negative structured filtering is not wired).
_POSITIVE_OPERATORS = {"require", "prefer"}
_STRUCTURED_DIMENSION_FIELD = {"title": "titles", "company": "companies"}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _merge_slice_constraints_into_filters(
    structured: LinkedInStructuredFilters,
    lane: SourcingLane,
) -> LinkedInStructuredFilters:
    """Fold a lane's slice constraints into the compiled structured filters.

    ``compile_constraint`` already classifies a constraint's ``execution_surface``
    into a ``structured_control`` (dimension + values) for the ``linkedin_*_filter``
    surfaces. Before Phase 1 that control reached only the linter
    (``attach_constraint_lint_to_plan``); here we consume it so a ``require`` /
    ``prefer`` title or company constraint reaches ``query_payload['structured_filters']``
    instead of dead-ending. Returns a copy; the input is not mutated.
    """
    from linkedin.boolean_compiler import compile_constraint  # lazy: avoid import cycle

    slice_obj = getattr(lane, "slice", None)
    constraints = list(getattr(slice_obj, "constraints", []) or []) if slice_obj else []
    if not constraints:
        return structured

    merged = LinkedInStructuredFilters.from_dict(structured.to_dict())
    for constraint in constraints:
        if constraint.operator not in _POSITIVE_OPERATORS:
            continue
        control = compile_constraint(constraint, source="linkedin").structured_control
        values = [str(v).strip() for v in (control.get("values") or []) if str(v).strip()]
        if not values:
            continue
        dimension = control.get("dimension", "")
        if dimension == "location":
            # R3: route a slice location to sidebar_filters['locations'] — the ONLY key
            # compile_structured_filters_to_plan reads to emit a 'locations' control
            # (advanced_search.py:268-271). Parking it under advanced_filters (the generic
            # fallback below) dead-ends: compile reads advanced_filters only for
            # 'fields_of_study', so the location was silently dropped and the lane searched
            # geographically unbounded. The session-location path (Build-1 /
            # _apply_session_location_filter) is a SEPARATE direct-apply route — it reads
            # permanent_filters['Location'] and calls browser.apply_location_filter
            # directly, bypassing compile, so it never enters this bucket; slice and session
            # locations converge only at browser.apply_location_filter, not here.
            bucket = dict(merged.sidebar_filters)
            bucket["locations"] = _dedupe(list(bucket.get("locations", [])) + values)
            merged.sidebar_filters = bucket
            continue
        field_name = _STRUCTURED_DIMENSION_FIELD.get(dimension)
        if field_name is None:
            # No first-class field and no sidebar route (any other structured dimension):
            # park under advanced_filters keyed by '{dimension}s' (e.g. fields_of_study).
            bucket = dict(merged.advanced_filters)
            key = f"{dimension}s" if dimension else "other"
            bucket[key] = _dedupe(list(bucket.get(key, [])) + values)
            merged.advanced_filters = bucket
            continue
        setattr(merged, field_name, _dedupe(list(getattr(merged, field_name)) + values))
    return merged


def _compile_boolean_from_slice_constraints(lane: SourcingLane) -> str:
    """Compile Boolean-surface lane constraints into the lane keyword fallback."""
    from linkedin.boolean_compiler import compile_constraint  # lazy: avoid import cycle

    slice_obj = getattr(lane, "slice", None)
    constraints = list(getattr(slice_obj, "constraints", []) or []) if slice_obj else []
    positives: list[str] = []
    exclusions: list[str] = []
    for constraint in constraints:
        compiled = compile_constraint(constraint, source="linkedin")
        fragment = str(compiled.boolean_fragment or "").strip()
        if not fragment:
            continue
        if fragment.startswith("NOT "):
            exclusions.append(fragment)
        else:
            positives.append(fragment)
    if not positives:
        return ""
    return " AND ".join(positives + exclusions)


class LinkedInLaneCompiler:
    """Compile SourcingLane / LaneVariant into LinkedIn-native execution plans."""

    source: str = "linkedin"

    def compile(
        self,
        lane: SourcingLane,
        variant: LaneVariant | None = None,
    ) -> ExecutableSearch:
        acquisition_mode = lane.execution.acquisition_mode or "linkedin_boolean"
        warnings: list[LaneCompilerFinding] = []

        if variant is not None:
            li_variant = compile_lane_variant_to_linkedin(variant)
            boolean = li_variant.boolean
            structured = li_variant.structured_filters
        else:
            boolean = lane.execution.boolean_strategy.get("root_boolean", "")
            if not boolean:
                boolean = _compile_boolean_from_slice_constraints(lane)
            raw_filters = lane.execution.structured_filters
            structured = LinkedInStructuredFilters(
                titles=list(raw_filters.get("titles", [])),
                companies=list(
                    raw_filters.get("companies", [])
                    or raw_filters.get("target_employers", [])
                ),
                sidebar_filters=dict(raw_filters.get("sidebar_filters", {})),
                advanced_filters=dict(raw_filters.get("advanced_filters", {})),
            )
            # Surface the silent narrowing risk: when no reasoned `companies` key
            # exists, the company filter is seeded from `target_employers` (the
            # family's stated employer set), which can be narrower than — and
            # diverge from — the companies named in the keyword Boolean. The
            # recruiter then gets a tighter company bound than the Boolean implies,
            # with no signal. Warning-only; never gates execution.
            if not raw_filters.get("companies") and raw_filters.get("target_employers"):
                warnings.append(
                    LaneCompilerFinding(
                        code="structured_filter_seeded_from_target_employers",
                        severity="warning",
                        message=(
                            f"Company filter seeded from target_employers "
                            f"({len(structured.companies)} companies), not from a reasoned "
                            "companies constraint; verify it matches the intended employer "
                            "set and is not narrower than the keyword Boolean."
                        ),
                        dimension="companies",
                        source="linkedin",
                    )
                )

        # Phase 1 hop 1: fold slice-level structured constraints (title/company/
        # location) into the compiled filters so they reach query_payload, not just
        # the linter (attach_constraint_lint_to_plan).
        structured = _merge_slice_constraints_into_filters(structured, lane)

        # Phase 2 hop 3 trigger: any structured filter forces hybrid acquisition.
        # apply_variant rejects a boolean variant that carries structured filters
        # (search_mutation.py:60-73), so structured_filters and hybrid mode are
        # inseparable. Empty filters keep the lane on linkedin_boolean — the live
        # keyword path is untouched (and ~all lanes are filter-less today).
        if not structured.is_empty():
            acquisition_mode = "linkedin_hybrid"

        plan = compile_structured_filters_to_plan(
            structured,
            keyword_boolean=boolean,
            acquisition_mode=acquisition_mode,
        )

        for ctrl in plan.controls:
            tier = classify_control(ctrl.dimension)
            if tier in ("mock_only", "defer"):
                warnings.append(
                    LaneCompilerFinding(
                        code="unsupported_control",
                        severity="info",
                        message=f"{ctrl.dimension} is {tier}; not automated live",
                        dimension=ctrl.dimension,
                        source="linkedin",
                    )
                )

        unsupported = tuple(
            w.dimension for w in warnings
            if w.dimension and w.code == "unsupported_control"
        )

        return ExecutableSearch(
            source="linkedin",
            acquisition_mode=acquisition_mode,
            display_name=lane.lane_name,
            query_payload={
                "boolean": boolean,
                "structured_filters": structured.to_dict(),
                "advanced_search_plan": snapshot_controls_from_plan(plan),
            },
            lane_id=lane.lane_id,
            variant_id=variant.variant_id if variant else "",
            unsupported_dimensions=unsupported,
            warnings=tuple(warnings),
        )

    def lint(
        self,
        lane: SourcingLane,
        variant: LaneVariant | None = None,
    ) -> list[LaneCompilerFinding]:
        exe = self.compile(lane, variant)
        findings: list[LaneCompilerFinding] = list(exe.warnings)

        plan_dict = exe.query_payload.get("advanced_search_plan", {})
        from linkedin.advanced_search import AdvancedSearchControl, AdvancedSearchPlan

        controls = [
            AdvancedSearchControl(
                dimension=c.get("dimension", ""),
                values=c.get("values", []),
            )
            for c in plan_dict.get("controls", [])
        ]
        plan = AdvancedSearchPlan(
            controls=controls,
            keyword_boolean=exe.query_payload.get("boolean", ""),
            acquisition_mode=exe.acquisition_mode,
        )
        for conflict in lint_boolean_filter_conflicts(plan):
            findings.append(
                LaneCompilerFinding(
                    code="boolean_filter_conflict",
                    severity="warning",
                    message=conflict,
                    source="linkedin",
                )
            )

        return findings
