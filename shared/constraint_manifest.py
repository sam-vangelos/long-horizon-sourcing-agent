"""Constraint manifest — P3b (plans/sourcing-rigor-hardening.md, Wave 2).

Rule: for each constraint class the brief can carry (geography, seniority,
employer blacklist, compensation), the system must be able to answer —
mechanically, per run — *who owns it, did it actually apply, and how do we
know*. A stated constraint with ZERO owners aborts intake with a named error
instead of evaporating (audit R5: a recruiter-stated comp band drops at intake
with no record of the drop).

The manifest is built at run start from the final brief (post-preflight),
persisted as ``constraint_manifest.json`` in the run's state dir, and rendered
in the run report — where the report-time aggregation also folds in the
defer-dimension counter (structured controls requested by lanes that no
LinkedIn actuator supports, from the per-string surface receipts).
"""

from __future__ import annotations

import re
from typing import Any


class ConstraintManifestError(RuntimeError):
    """Raised at intake when a stated constraint has zero owners.

    Same doctrine as PreflightRegimeError / GeographyRegimeError: a run that
    silently drops a recruiter constraint is worse than no run. The operator's
    one action is to strip the constraint from intake or give it an owner.
    """


MANIFEST_FILENAME = "constraint_manifest.json"

# Compensation mentions are scanned ONLY in recruiter-authored command
# surfaces (intake notes, instructions) — never in the JD-derived role
# description, where "competitive salary" boilerplate is routine and is not a
# search constraint.
_COMPENSATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcompensation\b",
        r"\bsalary\b",
        r"\bcomp\s+band\b",
        r"\bcomp\s+range\b",
        r"\bpay\s+range\b",
        r"\bbase\s+pay\b",
        r"\bOTE\b",
        r"\btotal\s+comp\b",
        r"\$\s?\d{2,3}\s?k\b",
        r"\$\d{3},\d{3}",
    )
)


def _compensation_mentions(brief: Any) -> list[str]:
    """Recruiter-authored comp mentions, one evidence snippet per hit source."""
    sources: list[tuple[str, str]] = []
    intake_notes = getattr(brief, "intake_notes", "") or ""
    if isinstance(intake_notes, str) and intake_notes.strip():
        sources.append(("intake_notes", intake_notes))
    instructions = getattr(brief, "instructions", None) or []
    if isinstance(instructions, (list, tuple)):
        for index, instruction in enumerate(instructions):
            text = str(instruction or "")
            if text.strip():
                sources.append((f"instructions[{index}]", text))

    mentions: list[str] = []
    for label, text in sources:
        for pattern in _COMPENSATION_PATTERNS:
            match = pattern.search(text)
            if match:
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                snippet = " ".join(text[start:end].split())
                mentions.append(f"{label}: …{snippet}…")
                break  # one evidence snippet per source is enough
    return mentions


def _geography_value(brief: Any) -> str:
    permanent_filters = getattr(brief, "permanent_filters", None)
    if not isinstance(permanent_filters, dict):
        return ""
    return str(permanent_filters.get("Location", "") or "").strip()


def _seniority_stated(brief: Any) -> tuple[bool, str]:
    from shared.strict_seniority import minimum_years_experience

    years = minimum_years_experience(brief)
    if years > 0:
        return True, f"minimum_years_experience={years}"
    experience_floor = getattr(brief, "experience_floor", None)
    if isinstance(experience_floor, dict) and experience_floor:
        return True, f"experience_floor={experience_floor}"
    return False, ""


def build_constraint_manifest(brief: Any) -> dict:
    """Build the per-run constraint-ownership manifest from the FINAL brief.

    Statuses: ``owned`` (stated, has owner/actuator/verifier), ``unstated``
    (the brief does not carry it), ``zero_owner`` (stated but nothing in the
    system can act on it — aborts intake via
    ``assert_constraint_manifest_runnable``).
    """
    classes: dict[str, dict] = {}

    geography = _geography_value(brief)
    classes["geography"] = {
        "stated_in_brief": bool(geography),
        "stated_value": geography,
        "owner_layer": "session filter (orchestrator._apply_session_location_filter)" if geography else "",
        "actuator": "browser.apply_location_filter (fail-closed, exact-facet)" if geography else "",
        "verify_method": "applied-chip confirmation + pre-string chip invariant" if geography else "",
        "status": "owned" if geography else "unstated",
    }

    seniority_stated, seniority_value = _seniority_stated(brief)
    classes["seniority"] = {
        "stated_in_brief": seniority_stated,
        "stated_value": seniority_value,
        "owner_layer": "judge prompts (role_level interpolation + decision matrix + minimum bar)" if seniority_stated else "",
        "actuator": "prompt layer only — the structured seniority facet is DEFER (advanced_search.DEFER_CONTROLS)" if seniority_stated else "",
        "verify_method": "none mechanical (prompt-enforced); defer-dimension counter tracks structured requests" if seniority_stated else "",
        "status": "owned" if seniority_stated else "unstated",
    }

    blacklist = [
        str(entry).strip()
        for entry in (getattr(brief, "employer_blacklist", None) or [])
        if str(entry).strip()
    ]
    classes["employer_blacklist"] = {
        "stated_in_brief": bool(blacklist),
        "stated_value": "; ".join(blacklist),
        "owner_layer": "snippet-stage current_company gate + full-eval employer block (P3c)" if blacklist else "",
        "actuator": "deterministic substring gate (facial stage); judge visibility (full eval)" if blacklist else "",
        "verify_method": "facial_shadow path stamp employer_blacklist on gated decisions" if blacklist else "",
        "status": "owned" if blacklist else "unstated",
    }

    compensation_mentions = _compensation_mentions(brief)
    classes["compensation"] = {
        "stated_in_brief": bool(compensation_mentions),
        "stated_value": "; ".join(compensation_mentions),
        "owner_layer": "",
        "actuator": "",
        "verify_method": "",
        # ZERO owners repo-wide (audit R5 census): LinkedIn exposes no comp
        # facet, no judge template weighs comp, and no boolean encodes it.
        "status": "zero_owner" if compensation_mentions else "unstated",
    }

    return {
        "schema_version": 1,
        "classes": classes,
        # Filled at report time from the per-string surface receipts
        # (requested-but-unsupported structured dimensions, audit R5-F2).
        "requested_but_unsupported": {},
    }


def assert_constraint_manifest_runnable(manifest: dict) -> None:
    """Abort intake when any stated constraint has zero owners.

    The manifest makes the drop a decision instead of an accident: the
    operator either strips the constraint from intake or the system grows an
    owner for it.
    """
    zero_owner = [
        name
        for name, entry in (manifest.get("classes") or {}).items()
        if entry.get("status") == "zero_owner"
    ]
    if zero_owner:
        evidence = "; ".join(
            f"{name}: {(manifest['classes'][name].get('stated_value') or '')[:160]}"
            for name in zero_owner
        )
        raise ConstraintManifestError(
            f"Stated constraint(s) with zero owners: {', '.join(zero_owner)}. "
            f"Nothing in the system can apply or verify them, so the run would "
            f"silently drop them. Strip them from intake or add an owner. "
            f"Evidence — {evidence}"
        )


def aggregate_unsupported_dimensions(strings: list[Any]) -> dict[str, int]:
    """Count requested-but-unsupported structured dimensions across a run.

    Reads each string's surface receipt (``unsupported_controls``). Closes the
    audit R5-F2 gap: per-event honesty existed (plan_fully_applied=False), but
    nothing aggregated "this brief asked for a dead facet N times".
    """
    counts: dict[str, int] = {}
    for search_string in strings:
        receipt = getattr(search_string, "surface_receipt", None) or {}
        if not isinstance(receipt, dict):
            continue
        for dimension in receipt.get("unsupported_controls") or []:
            key = str(dimension).strip()
            if key:
                counts[key] = counts.get(key, 0) + 1
    return counts
