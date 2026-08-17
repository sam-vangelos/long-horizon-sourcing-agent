"""Surface receipt — per-run observability for structured-filter materialization.

This is the production-truth instrument for the structured-filter unlock. Strategy
output is no longer eyeballed across dozens of Boolean strings: the receipt answers,
deterministically and per executed string, two questions —

  1. INTENDED: what structured surface did the plan intend to apply (keyword-only vs
     a title / company / location filter)? Computed from the queued ``SearchString``s.
  2. APPLIED: at opening time, what actually landed on the live Recruiter sidebar
     (chip-confirmed) vs fell back to keyword? Computed from the controller's
     ``ControlApplicationResult``.

The gap between INTENDED and APPLIED is exactly the failure this whole effort exists
to close, and the receipt makes it visible every run — including catching a silent
DOM-drift the day a filter stops applying, rather than months later.

Pure functions only; the orchestrator wires them to ``print`` and ``log_event`` so the
module stays trivially testable and side-effect free.
"""

from __future__ import annotations

from typing import Any

from shared.receipts import ReceiptStatus, build_receipt

# First-class structured list fields on LinkedInStructuredFilters that represent a
# live, applied facet. ``locations`` lives under ``sidebar_filters`` and is surfaced
# separately below. ``skills``/``assessments`` are not yet stable_now controls but are
# reported so the receipt never silently hides a requested dimension.
_STRUCTURED_LIST_FIELDS = ("titles", "companies", "skills", "assessments")
_NORMALIZATION_GUARD_CODES = (
    "ubiquitous_and_gate",
    "token_subset_superstring_pruned",
)


def _boolean_normalization_summary(search_string: Any) -> dict[str, Any]:
    raw = getattr(search_string, "boolean_normalization", None)
    if not isinstance(raw, dict):
        return {}

    finding_counts: dict[str, int] = {}
    finding_terms: dict[str, list[str]] = {}
    for finding in raw.get("findings") or []:
        code = (
            str(finding.get("code") or "").strip()
            if isinstance(finding, dict)
            else str(getattr(finding, "code", "") or "").strip()
        )
        if not code:
            continue
        finding_counts[code] = finding_counts.get(code, 0) + 1
        terms = finding.get("terms") if isinstance(finding, dict) else getattr(finding, "terms", ())
        for term in terms or ():
            finding_terms.setdefault(code, []).append(str(term))

    return {
        "changed": bool(raw.get("changed")),
        "finding_counts": finding_counts,
        "finding_terms": finding_terms,
        "guard_counts": {
            code: finding_counts.get(code, 0)
            for code in _NORMALIZATION_GUARD_CODES
        },
        "original_boolean": str(raw.get("original_boolean") or ""),
        "normalized_boolean": str(raw.get("normalized_boolean") or ""),
    }


def boolean_filter_overlap(boolean: str, dimensions: dict[str, list[str]]) -> list[str]:
    """Values that sit on BOTH a structured filter and the keyword Boolean.

    The kernel's harmony rule is that a value belongs on exactly one surface: a company
    on a company filter must not also appear in a keyword OR-clause, and likewise for a
    title. Returns the offending values so the receipt can flag the duplication
    (advisory — it never gates execution).
    """
    text = (boolean or "").lower()
    if not text:
        return []
    overlaps: list[str] = []
    for dim, values in dimensions.items():
        if dim == "locations":
            continue  # geo rides the session facet; keyword geo overlap is handled in Slice 3
        for value in values:
            v = str(value).strip().lower()
            if v and v in text and value not in overlaps:
                overlaps.append(value)
    return overlaps


def intended_surface_for_string(search_string: Any) -> dict[str, Any]:
    """Describe the structured surface a single queued SearchString intends to apply.

    Duck-typed on purpose: accepts a real ``SearchString`` or any object exposing
    ``id``/``name``/``acquisition_mode``/``surface``/``structured_filters``.
    """
    raw = getattr(search_string, "structured_filters", None)
    filters = raw if isinstance(raw, dict) else {}

    dimensions: dict[str, list[str]] = {}
    for field_name in _STRUCTURED_LIST_FIELDS:
        values = filters.get(field_name)
        if values:
            dimensions[field_name] = [str(v) for v in values]

    sidebar = filters.get("sidebar_filters")
    if isinstance(sidebar, dict):
        locations = sidebar.get("locations")
        if locations:
            dimensions["locations"] = [str(v) for v in locations]

    acquisition_mode = str(getattr(search_string, "acquisition_mode", "") or "")
    overlap = boolean_filter_overlap(
        str(getattr(search_string, "boolean", "") or ""), dimensions
    )
    return {
        "id": getattr(search_string, "id", None),
        "name": str(getattr(search_string, "name", "") or ""),
        "acquisition_mode": acquisition_mode,
        "surface": str(getattr(search_string, "surface", "") or ""),
        "structured_dimensions": dimensions,
        "boolean_normalization": _boolean_normalization_summary(search_string),
        # Kernel harmony violation: a value duplicated across a filter and the keyword
        # Boolean. Advisory signal, surfaced for the planner pass (Slice 2).
        "boolean_filter_overlap": overlap,
        # A string is a genuine hybrid only when it is BOTH marked hybrid AND carries
        # at least one structured dimension. Mode alone can lie (see the Fix 2 honesty
        # work); the dimensions are the ground truth.
        "is_hybrid": acquisition_mode == "linkedin_hybrid" and bool(dimensions),
    }


def summarize_intended_surfaces(strings: list[Any]) -> dict[str, Any]:
    """Aggregate the intended-surface map across the whole execution queue."""
    rows = [intended_surface_for_string(s) for s in strings]
    hybrid_rows = [r for r in rows if r["is_hybrid"]]
    dimension_counts: dict[str, int] = {}
    normalization_finding_counts: dict[str, int] = {}
    normalization_strings_with_findings = 0
    for row in hybrid_rows:
        for dim in row["structured_dimensions"]:
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
    for row in rows:
        normalization = row.get("boolean_normalization") or {}
        finding_counts = normalization.get("finding_counts") or {}
        if finding_counts:
            normalization_strings_with_findings += 1
        for code, count in finding_counts.items():
            normalization_finding_counts[code] = (
                normalization_finding_counts.get(code, 0) + int(count or 0)
            )
    overlap_strings = sum(1 for r in rows if r.get("boolean_filter_overlap"))
    return {
        "total_strings": len(rows),
        "hybrid_strings": len(hybrid_rows),
        "dimension_counts": dimension_counts,
        "overlap_strings": overlap_strings,
        "normalization_strings_with_findings": normalization_strings_with_findings,
        "normalization_finding_counts": normalization_finding_counts,
        "normalization_guard_counts": {
            code: normalization_finding_counts.get(code, 0)
            for code in _NORMALIZATION_GUARD_CODES
        },
        "rows": rows,
    }


def intended_surface_receipt(summary: dict[str, Any]) -> dict[str, Any]:
    """Typed receipt for the queued structured-filter surface summary."""

    receipt = build_receipt(
        receipt_type="pipeline_stage",
        stage="surface_intended",
        input_payload=summary,
        actual_status=ReceiptStatus.OK,
        intended_postcondition=(
            "queued search strings expose their intended structured-filter surface"
        ),
        actual_detail=summary,
        producer="linkedin.surface_receipt",
        version_pins={"surface_receipt": "intended-v1"},
    )
    return receipt.to_dict()


def format_intended_summary(summary: dict[str, Any]) -> str:
    """Render the intended-surface summary for the console.

    Leads with the headline (how many of N strings will actually apply a structured
    filter, and on which dimensions), then lists only the hybrid strings — a run with
    zero structured filters prints one honest line instead of dozens of noise rows.
    """
    total = summary.get("total_strings", 0)
    hybrid = summary.get("hybrid_strings", 0)
    dim_counts = summary.get("dimension_counts", {})
    dim_str = (
        ", ".join(f"{k}={v}" for k, v in dim_counts.items()) if dim_counts else "none"
    )
    lines = [
        f"  [surface] {hybrid}/{total} queued strings carry a structured filter "
        f"(dimensions: {dim_str})"
    ]
    guard_counts = summary.get("normalization_guard_counts") or {}
    guard_str = ", ".join(
        f"{code}={int(guard_counts.get(code, 0) or 0)}"
        for code in _NORMALIZATION_GUARD_CODES
    )
    lines.append(f"  [surface] normalization guards: {guard_str}")
    overlap_strings = summary.get("overlap_strings", 0)
    if overlap_strings:
        lines.append(
            f"  [surface] WARNING: {overlap_strings} string(s) duplicate a filter value "
            "in their keyword Boolean (kernel harmony violation)"
        )
    for row in summary.get("rows", []):
        if not row["is_hybrid"]:
            continue
        dims = "; ".join(
            f"{dim}={values}" for dim, values in row["structured_dimensions"].items()
        )
        suffix = ""
        if row.get("boolean_filter_overlap"):
            suffix = f"  [DUP in boolean: {row['boolean_filter_overlap']}]"
        lines.append(f"    #{row['id']} {row['name'][:48]} -> {dims}{suffix}")
    return "\n".join(lines)


def apply_receipt_fields(search_string: Any, result: Any) -> dict[str, Any]:
    """Structured fields for a ``surface_applied`` log event from a controller result.

    ``result`` is a ``ControlApplicationResult`` (advanced_search.py). The honesty axis
    that matters: a structured dimension that fell to ``unsupported``/``failed`` while
    only the keyword landed is NOT a successful structured apply, and ``fell_back``
    captures that so a keyword-only fallback is never read as a filter that applied.
    """
    applied = list(getattr(result, "applied_controls", []) or [])
    failed = list(getattr(result, "failed_controls", []) or [])
    unsupported = list(getattr(result, "unsupported_controls", []) or [])
    structured_applied = [c for c in applied if c != "keywords"]
    fell_back = bool(getattr(result, "fallback_to_boolean", False)) or (
        not structured_applied and (failed or unsupported)
    )
    # P2.2: per-dimension value counts. A control applies its whole value list
    # atomically at this layer (dimension grain), so a dimension that applied
    # counts all its requested values as applied; failed/unsupported count 0.
    requested_dimensions = intended_surface_for_string(search_string)[
        "structured_dimensions"
    ]
    requested_value_counts = {
        dim: len(values) for dim, values in requested_dimensions.items()
    }
    applied_value_counts = {
        dim: (count if dim in structured_applied else 0)
        for dim, count in requested_value_counts.items()
    }
    return {
        "string_id": getattr(search_string, "id", None),
        "string_name": str(getattr(search_string, "name", "") or "")[:64],
        "acquisition_mode": str(getattr(search_string, "acquisition_mode", "") or ""),
        "applied_controls": applied,
        "failed_controls": failed,
        "unsupported_controls": unsupported,
        "structured_applied": structured_applied,
        "requested_value_counts": requested_value_counts,
        "applied_value_counts": applied_value_counts,
        "boolean_normalization": _boolean_normalization_summary(search_string),
        "plan_fully_applied": bool(getattr(result, "plan_fully_applied", False)),
        "fell_back_to_keyword": fell_back,
        "reason": str(getattr(result, "reason", "") or ""),
    }


def applied_surface_receipt(fields: dict[str, Any]) -> dict[str, Any]:
    """Typed receipt for what landed on the live Recruiter sidebar."""

    fell_back = bool(fields.get("fell_back_to_keyword"))
    status = ReceiptStatus.POSTCONDITION_FAIL if fell_back else ReceiptStatus.OK
    receipt = build_receipt(
        receipt_type="pipeline_stage",
        stage="surface_applied",
        input_payload=fields,
        actual_status=status,
        intended_postcondition=(
            "live Recruiter controls match the planned structured search surface"
        ),
        actual_detail=fields,
        producer="linkedin.surface_receipt",
        version_pins={"surface_receipt": "applied-v1"},
    )
    return receipt.to_dict()


def format_apply_receipt(fields: dict[str, Any]) -> str:
    """One-line console receipt for what actually applied on the live sidebar."""
    structured = fields.get("structured_applied") or []
    if structured and fields.get("plan_fully_applied"):
        verdict = f"APPLIED {structured}"
    elif structured:
        verdict = f"PARTIAL applied={structured} dropped={fields.get('unsupported_controls')}"
    elif fields.get("fell_back_to_keyword"):
        verdict = "FELL BACK to keyword (no structured filter landed)"
    else:
        verdict = "keyword-only"
    return f"  [surface] #{fields.get('string_id')} {verdict}"
