"""Staged advanced search controller — P3.

Orchestrates section-scoped control application on the LinkedIn Recruiter
sidebar, driven by the P3a DOM map in docs/linkedin-recruiter-dom-map.md.
Only ``stable_now`` controls are eligible for live automation; ``mock_only``
and ``defer`` controls are classified but never applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from linkedin.browser import LinkedInBrowser

STABLE_NOW_CONTROLS = frozenset({"keywords", "locations", "job_titles", "companies"})
MOCK_ONLY_CONTROLS = frozenset({"fields_of_study"})
DEFER_CONTROLS = frozenset({
    "seniority", "years_of_experience", "industries", "job_functions",
    "skills", "schools", "degrees", "projects", "notes", "ats_status",
})

ALL_KNOWN_CONTROLS = STABLE_NOW_CONTROLS | MOCK_ONLY_CONTROLS | DEFER_CONTROLS

_FACET_RECEIPT_LABELS = {
    "locations": "locations",
    "companies": "companies",
    "job_titles": "job_titles",
}
_FACET_ALREADY_COUNT_ATTRS = {
    "locations": "last_location_already_applied_count",
    "companies": "last_company_already_applied_count",
    "job_titles": "last_title_already_applied_count",
}


@dataclass
class AdvancedSearchControl:
    """A single structured filter intent for a search dimension."""

    dimension: str
    values: list[str] = field(default_factory=list)
    operator: str = "prefer"
    execution_surface: str = "soft_hint"
    expansion: str = "literal"
    temporal_scope: str = "any"
    confidence: float = 0.5


@dataclass
class AdvancedSearchPlan:
    """A set of controls to apply as a batch, with a keyword Boolean fallback."""

    controls: list[AdvancedSearchControl] = field(default_factory=list)
    keyword_boolean: str = ""
    preserve_existing: bool = False
    acquisition_mode: str = "boolean_only"


@dataclass
class ControlApplicationResult:
    """Structured outcome of applying an advanced search plan.

    ``success`` means "we landed a usable search" (at least the keyword Boolean,
    plus any structured control that applied). ``plan_fully_applied`` is the
    honesty axis on top of that: it is ``True`` only when *every* requested
    control actually applied — no ``failed`` and no ``unsupported`` controls. A
    hybrid plan where the keyword landed but a requested structured dimension
    fell to ``unsupported`` is ``plan_fully_applied=False``; if the keyword is
    the *only* thing that applied (every structured control dropped), it is also
    ``success=False`` so a keyword-only fallback is never reported as a fully
    applied structured plan.
    """

    success: bool
    applied_controls: list[str] = field(default_factory=list)
    failed_controls: list[str] = field(default_factory=list)
    unsupported_controls: list[str] = field(default_factory=list)
    fallback_to_boolean: bool = False
    reason: str = ""
    plan_fully_applied: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "applied_controls": self.applied_controls,
            "failed_controls": self.failed_controls,
            "unsupported_controls": self.unsupported_controls,
            "fallback_to_boolean": self.fallback_to_boolean,
            "reason": self.reason,
            "plan_fully_applied": self.plan_fully_applied,
        }


def classify_control(dimension: str) -> str:
    """Return the DOM-map classification for a search dimension.

    Returns one of ``stable_now``, ``mock_only``, ``defer``, or ``unknown``.
    """
    if dimension in STABLE_NOW_CONTROLS:
        return "stable_now"
    if dimension in MOCK_ONLY_CONTROLS:
        return "mock_only"
    if dimension in DEFER_CONTROLS:
        return "defer"
    return "unknown"


def _format_facet_receipt_line(
    browser: "LinkedInBrowser",
    ctrl: AdvancedSearchControl,
    *,
    applied: bool,
) -> str | None:
    label = _FACET_RECEIPT_LABELS.get(ctrl.dimension)
    if not label:
        return None
    if not applied:
        return f"  [facet] {label}: FAILED — falling back to keyword boolean"
    requested = len([v for v in ctrl.values if str(v).strip()])
    attr = _FACET_ALREADY_COUNT_ATTRS[ctrl.dimension]
    try:
        already = int(getattr(browser, attr, 0) or 0)
    except Exception:
        already = 0
    return f"  [facet] {label}: applied {requested}/{requested} ({already} already on sidebar)"


def lint_boolean_filter_conflicts(plan: AdvancedSearchPlan) -> list[str]:
    """Detect when Boolean terms duplicate or fight structured filters.

    Returns a list of human-readable conflict descriptions. An empty list
    means no conflicts detected.
    """
    if not plan.keyword_boolean or not plan.controls:
        return []

    conflicts = []
    boolean_lower = plan.keyword_boolean.lower()

    for ctrl in plan.controls:
        if ctrl.dimension == "keywords":
            continue
        for value in ctrl.values:
            if value.lower() in boolean_lower:
                conflicts.append(
                    f"Boolean already contains '{value}' which is also set as "
                    f"a structured {ctrl.dimension} filter"
                )

    return conflicts


async def apply_advanced_search_plan(
    browser: "LinkedInBrowser",
    plan: AdvancedSearchPlan,
) -> ControlApplicationResult:
    """Apply a plan's controls to the browser sidebar.

    For ``stable_now`` controls, delegates to browser methods. For
    ``mock_only`` and ``defer`` controls, returns them in
    ``unsupported_controls``. If any ``stable_now`` control fails
    verification, falls back to ``boolean_only``.

    Never clicks the global "Clear search" button.
    """
    applied: list[str] = []
    failed: list[str] = []
    unsupported: list[str] = []

    for ctrl in plan.controls:
        tier = classify_control(ctrl.dimension)
        if tier in ("mock_only", "defer", "unknown"):
            unsupported.append(ctrl.dimension)
            continue

        try:
            ok = await _apply_stable_control(browser, ctrl, plan)
            if ok:
                applied.append(ctrl.dimension)
            else:
                failed.append(ctrl.dimension)
            receipt_line = _format_facet_receipt_line(browser, ctrl, applied=ok)
            if receipt_line:
                print(receipt_line)
        except Exception:
            failed.append(ctrl.dimension)
            receipt_line = _format_facet_receipt_line(browser, ctrl, applied=False)
            if receipt_line:
                print(receipt_line)

    if plan.keyword_boolean and "keywords" not in applied:
        try:
            await browser.enter_search_string(plan.keyword_boolean)
            applied.append("keywords")
        except Exception:
            failed.append("keywords")

    fully_applied = not failed and not unsupported

    if failed:
        return ControlApplicationResult(
            success=False,
            applied_controls=applied,
            failed_controls=failed,
            unsupported_controls=unsupported,
            fallback_to_boolean=True,
            reason="stable_now_control_failed_fallback_to_boolean",
            plan_fully_applied=False,
        )

    if plan.controls and unsupported and not applied:
        return ControlApplicationResult(
            success=False,
            applied_controls=applied,
            failed_controls=failed,
            unsupported_controls=unsupported,
            fallback_to_boolean=False,
            reason="no_supported_controls_applied",
            plan_fully_applied=False,
        )

    # Honesty axis (R6): a requested structured control fell to ``unsupported``
    # and the keyword Boolean is the ONLY thing that applied. The keyword landed,
    # so the search runs — but it is a keyword-only fallback, NOT the structured
    # plan the caller asked for. Report success=False so a dropped structured
    # dimension is never masqueraded as a fully applied plan. (When a structured
    # control DID apply alongside the dropped ones, success stays True — a usable
    # hybrid — but plan_fully_applied is False below.)
    if unsupported and applied == ["keywords"]:
        return ControlApplicationResult(
            success=False,
            applied_controls=applied,
            failed_controls=failed,
            unsupported_controls=unsupported,
            fallback_to_boolean=True,
            reason="structured_controls_dropped_keyword_only",
            plan_fully_applied=False,
        )

    # Empty-plan phantom-success (slice E, slice-D carryover): a plan that landed
    # NOTHING — nothing applied, nothing failed, nothing unsupported, and no keyword
    # Boolean to fall back on — has not produced a usable search. Reporting it
    # success=True / all_stable_controls_applied would contradict the R6 honesty axis
    # this file enforces above (a dropped structured dim is never masqueraded as a
    # fully applied plan). There is no search here at all, so report success=False.
    if not applied and not failed and not unsupported and not plan.keyword_boolean:
        return ControlApplicationResult(
            success=False,
            applied_controls=applied,
            failed_controls=failed,
            unsupported_controls=unsupported,
            fallback_to_boolean=False,
            reason="empty_plan_nothing_applied",
            plan_fully_applied=False,
        )

    return ControlApplicationResult(
        success=True,
        applied_controls=applied,
        failed_controls=failed,
        unsupported_controls=unsupported,
        fallback_to_boolean=False,
        reason="all_stable_controls_applied" if fully_applied else "stable_controls_applied_some_unsupported",
        plan_fully_applied=fully_applied,
    )


async def _apply_stable_control(
    browser: "LinkedInBrowser",
    ctrl: AdvancedSearchControl,
    plan: AdvancedSearchPlan,
) -> bool:
    """Apply a single stable_now control. Returns True on success."""
    if ctrl.dimension == "keywords":
        value = ctrl.values[0] if ctrl.values else plan.keyword_boolean
        if value:
            await browser.enter_search_string(value)
            return True
        return False
    if ctrl.dimension == "locations":
        # Hop-4 canary: route a graduated Location control to the live sidebar apply
        # method. Reached only once 'locations' is in STABLE_NOW_CONTROLS (H4-S4,
        # gated on the Pass-5 DOM capture); until then classify_control shunts it to
        # unsupported_controls before this function is called.
        if ctrl.values:
            return await browser.apply_location_filter(
                ctrl.values, temporal_scope=ctrl.temporal_scope
            )
        return False
    if ctrl.dimension == "companies":
        # Pre-staged like the locations branch: reached only once 'companies' is in
        # STABLE_NOW_CONTROLS (gated on tools/hop4_company_smoke.py PASSING). Until then
        # classify_control shunts it to unsupported_controls before this is called.
        if ctrl.values:
            return await browser.apply_company_filter(
                ctrl.values, temporal_scope=ctrl.temporal_scope
            )
        return False
    if ctrl.dimension == "job_titles":
        if ctrl.values:
            return await browser.apply_title_filter(
                ctrl.values, temporal_scope=ctrl.temporal_scope
            )
        return False
    return False


def compile_structured_filters_to_plan(
    structured_filters: Any,
    *,
    keyword_boolean: str,
    acquisition_mode: str = "linkedin_hybrid",
    include_keyword: bool = True,
) -> AdvancedSearchPlan:
    """Compile LinkedIn structured filters into an AdvancedSearchPlan.

    ``include_keyword`` is the structured_only lever (Phase 2 hop 4, slice D).
    When ``False``, the keyword control is NOT appended AND the returned plan's
    ``keyword_boolean`` is forced to the empty string. The empty string is what
    trips the downstream ``if plan.keyword_boolean`` guards at
    ``apply_advanced_search_plan`` (~:156) and ``compile_recovery_plan_from_snapshot``
    (~:295), so this single lever suppresses the keyword end to end — the
    compile-time control, the apply-time re-add, and the recovery-time replay.
    """
    controls: list[AdvancedSearchControl] = []
    if keyword_boolean and include_keyword:
        controls.append(
            AdvancedSearchControl(dimension="keywords", values=[keyword_boolean])
        )
    for title in getattr(structured_filters, "titles", []) or []:
        if title:
            controls.append(AdvancedSearchControl(dimension="job_titles", values=[title]))
    for company in getattr(structured_filters, "companies", []) or []:
        if company:
            controls.append(AdvancedSearchControl(dimension="companies", values=[company]))
    sidebar = dict(getattr(structured_filters, "sidebar_filters", {}) or {})
    for location in sidebar.get("locations", []) or []:
        if location:
            controls.append(AdvancedSearchControl(dimension="locations", values=[location]))
    advanced = dict(getattr(structured_filters, "advanced_filters", {}) or {})
    for field in advanced.get("fields_of_study", []) or []:
        if field:
            controls.append(AdvancedSearchControl(dimension="fields_of_study", values=[field]))
    return AdvancedSearchPlan(
        controls=controls,
        keyword_boolean=keyword_boolean if include_keyword else "",
        acquisition_mode=acquisition_mode,
    )


def compile_recovery_plan_from_snapshot(snapshot: Any) -> AdvancedSearchPlan:
    """Build an AdvancedSearchPlan from a recovery snapshot's stored controls."""
    controls: list[AdvancedSearchControl] = []
    raw_controls = dict(getattr(snapshot, "advanced_search_controls", {}) or {}).get("controls", [])
    for item in raw_controls:
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension", "")).strip()
        values = [str(v) for v in item.get("values", []) if str(v).strip()]
        if dimension and values:
            controls.append(AdvancedSearchControl(dimension=dimension, values=values))
    keyword_boolean = str(getattr(snapshot, "keyword_boolean", "") or "")
    if keyword_boolean and not any(c.dimension == "keywords" for c in controls):
        controls.insert(
            0,
            AdvancedSearchControl(dimension="keywords", values=[keyword_boolean]),
        )
    return AdvancedSearchPlan(
        controls=controls,
        keyword_boolean=keyword_boolean,
        acquisition_mode=str(
            dict(getattr(snapshot, "advanced_search_controls", {}) or {}).get(
                "acquisition_mode",
                "boolean_only",
            )
        ),
    )


def snapshot_controls_from_plan(
    plan: AdvancedSearchPlan,
    *,
    applied_dimensions: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Build a dict snapshot from a plan for diagnostics/persistence.

    ``applied_dimensions`` is the R5 honesty filter for the *recovery* snapshot:
    when provided, only controls whose dimension actually applied are recorded,
    so a partially-applied (or fully-dropped) control is never persisted — and
    therefore never *replayed* — as if it had landed. The recruiter would
    otherwise believe an over-broad filter was in place and get unbounded
    results. When ``None`` (compile-time planning artifacts and the empty
    snapshot), the full plan is recorded unchanged.
    """
    if applied_dimensions is None:
        controls = list(plan.controls)
    else:
        allow = set(applied_dimensions)
        controls = [c for c in plan.controls if c.dimension in allow]
    return {
        "controls": [
            {
                "dimension": c.dimension,
                "values": c.values,
                "operator": c.operator,
                "tier": classify_control(c.dimension),
            }
            for c in controls
        ],
        "keyword_boolean": plan.keyword_boolean,
        "acquisition_mode": plan.acquisition_mode,
    }
