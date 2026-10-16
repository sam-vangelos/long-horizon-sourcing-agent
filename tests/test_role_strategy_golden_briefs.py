"""Golden classification suite for shared/role_strategy.py's routing logic.

A forensic audit found `infer_role_strategy_profile_id` misrouting on four
fronts: (a) any Senior/Staff/Principal role_level fell into
`ic_frontier_engineer` even for non-engineering roles, because `_is_ic_level`
fires on role_level alone; (b) pattern matching was bare substring over a
concatenated text blob that included `instructions`, so JD boilerplate
("collaborate with product designers", "PhD preferred", "reports to the
CTO") could flip the whole profile; (c) `maintainership_level not in ("",
"contributor")` routed junk values like "none" to `oss_maintainer`, and a
bare `target_projects` flipped the profile on its own; (d) an unused
employer-pattern helper was dead code.

These cases pin the corrected behavior: word-boundary matching, title-scoped
matching for the TITLE-CLASS pattern sets, an engineering-signal gate on the
IC-frontier route, a stricter oss_maintainer membership gate, and
`instructions` excluded from the matched text entirely.

Stub briefs are lightweight `SimpleNamespace` objects mirroring exactly how
`shared/role_strategy.py` reads a real `Brief`: `role_title` and
`role_description` are top-level attributes; `role_level`, `role_summary`,
`intake_notes`, and `capability_areas` live under `.raw` (as they do on a
real `Brief.raw` dict); `instructions`, `target_modules`, `target_projects`,
`maintainership_level`, and `minimum_years_experience` are top-level
attributes (as they are on the real `Brief` dataclass). None of these stubs
set `_new_brief`, so `getattr(brief, "_new_brief", None)` resolves to `None`
exactly as it does for a legacy (non-V2) brief.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from shared.role_strategy import infer_role_strategy_profile_id


def _stub_brief(
    *,
    role_title: str = "",
    role_description: str = "",
    role_level: str = "",
    role_summary: str = "",
    intake_notes: str = "",
    capability_areas: list[dict[str, str]] | None = None,
    instructions: list[str] | None = None,
    target_modules: list[str] | None = None,
    target_projects: list[str] | None = None,
    maintainership_level: str = "",
    minimum_years_experience: int = 0,
    role_strategy_profile: str = "",
) -> SimpleNamespace:
    """Build a stub brief mirroring role_strategy.py's getattr/raw reads."""
    raw: dict[str, Any] = {
        "role_level": role_level,
        "role_summary": role_summary,
        "intake_notes": intake_notes,
        "capability_areas": capability_areas or [],
    }
    if role_strategy_profile:
        raw["role_strategy_profile"] = role_strategy_profile
    return SimpleNamespace(
        role_title=role_title,
        role_description=role_description,
        raw=raw,
        instructions=instructions or [],
        target_modules=target_modules or [],
        target_projects=target_projects or [],
        maintainership_level=maintainership_level,
        minimum_years_experience=minimum_years_experience,
    )


# ---------------------------------------------------------------------------
# Case 1 — the audit's headline case: Senior role_level alone must not force
# ic_frontier_engineer for a non-engineering role.
# ---------------------------------------------------------------------------


def test_senior_non_engineering_role_falls_to_generic() -> None:
    brief = _stub_brief(
        role_title="Senior Operations Manager",
        role_level="Senior",
        capability_areas=[{"name": "Vendor management"}, {"name": "Process optimization"}],
    )
    profile_id, _metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "generic"


# ---------------------------------------------------------------------------
# Case 2 — a single academic-pattern hit in the JD body must not preempt an
# otherwise-clear engineering title.
# ---------------------------------------------------------------------------


def test_single_academic_hit_does_not_preempt_ic_frontier() -> None:
    brief = _stub_brief(
        role_title="Staff Software Engineer",
        role_level="Staff",
        role_description="Build training infrastructure. PhD preferred but not required.",
    )
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "ic_frontier_engineer"
    assert "academic_title" not in metadata["matched_signals"]
    assert "academic_patterns_x2" not in metadata["matched_signals"]


# ---------------------------------------------------------------------------
# Case 3 — "designer" language in the JD body (not the title) must not fire
# the designer route.
# ---------------------------------------------------------------------------


def test_designer_language_in_description_does_not_fire_designer_route() -> None:
    brief = _stub_brief(
        role_title="Senior Backend Engineer",
        role_level="Senior",
        role_description="You will collaborate closely with product designers and researchers across the org.",
    )
    profile_id, _metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "ic_frontier_engineer"


# ---------------------------------------------------------------------------
# Case 4 — exec language in the JD body (not the title) must not fire the
# executive_search route; Director alone is not IC-level either.
# ---------------------------------------------------------------------------


def test_exec_language_in_description_does_not_fire_from_full_text() -> None:
    brief = _stub_brief(
        role_title="Director of Data Platform",
        role_level="Director",
        role_description="This role reports to the CTO and partners with product and platform leadership.",
    )
    profile_id, _metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "generic"


# ---------------------------------------------------------------------------
# Case 5 — documents the decided routing: "president" word-boundary hit in
# the title, combined with a senior level, does route executive_search.
# ---------------------------------------------------------------------------


def test_vp_title_with_president_word_boundary_routes_executive_search() -> None:
    brief = _stub_brief(
        role_title="Vice President of Engineering",
        role_level="VP",
    )
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "executive_search"
    assert "executive_title_patterns" in metadata["matched_signals"]


# ---------------------------------------------------------------------------
# Case 6 — one "tenure" hit is not the two distinct academic-pattern hits
# the academic route requires, and there is no engineering signal either.
# ---------------------------------------------------------------------------


def test_single_tenure_hit_does_not_route_academic() -> None:
    brief = _stub_brief(
        role_title="Senior Account Executive",
        role_level="Senior",
        role_description="Looking for long tenure with strategic accounts and a consistent track record.",
    )
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "generic"
    assert "academic_patterns_x2" not in metadata["matched_signals"]


# ---------------------------------------------------------------------------
# Case 7 — a strict-seniority BFS brief (years + trigger phrase + bank/
# financial vocabulary) routes senior_bfs_ai_leader via strict_seniority
# alone.
# ---------------------------------------------------------------------------


def test_strict_seniority_bfs_brief_routes_senior_bfs_ai_leader() -> None:
    brief = _stub_brief(
        role_title="Executive Director, Applied AI Lab",
        role_level="Executive Director",
        role_description=(
            "Executive director scope leading the applied ai lab for a global "
            "financial services bank. 14+ years building production agent systems."
        ),
        minimum_years_experience=14,
    )
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "senior_bfs_ai_leader"
    assert "strict_seniority" in metadata["matched_signals"]


# ---------------------------------------------------------------------------
# Case 8 — the same strict-seniority trigger phrase ("lab-head scope") and
# years bar, but NO bank/financial/bfsi vocabulary anywhere: is_strict_
# seniority_brief's third gate fails, and bfs_domain is also false, so this
# must NOT route senior_bfs_ai_leader.
# ---------------------------------------------------------------------------


def test_healthcare_lab_head_scope_without_bfs_vocabulary_is_not_senior_bfs() -> None:
    brief = _stub_brief(
        role_title="Director of Clinical AI",
        role_level="Director",
        role_description=(
            "Executive lab-head scope leading applied AI for oncology and clinical "
            "trial diagnostics across our hospital network. 14+ years experience."
        ),
        minimum_years_experience=14,
    )
    profile_id, _metadata = infer_role_strategy_profile_id(brief)
    assert profile_id != "senior_bfs_ai_leader"
    assert profile_id == "generic"


# ---------------------------------------------------------------------------
# Case 9 — a clean designer title routes designer.
# ---------------------------------------------------------------------------


def test_senior_product_designer_routes_designer() -> None:
    brief = _stub_brief(role_title="Senior Product Designer", role_level="Senior")
    profile_id, _metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "designer"


# ---------------------------------------------------------------------------
# Case 10 — "Research Scientist" is itself an _ACADEMIC_PATTERNS title hit.
# ---------------------------------------------------------------------------


def test_research_scientist_title_routes_academic_researcher() -> None:
    brief = _stub_brief(
        role_title="Research Scientist",
        role_description="Strong candidates will have postdoc and faculty collaborations.",
    )
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "academic_researcher"
    assert "academic_title" in metadata["matched_signals"]


# ---------------------------------------------------------------------------
# Case 11 — a recognized elevated maintainership_level routes oss_maintainer
# on its own.
# ---------------------------------------------------------------------------


def test_maintainer_level_routes_oss_maintainer() -> None:
    brief = _stub_brief(maintainership_level="maintainer")
    profile_id, _metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "oss_maintainer"


# ---------------------------------------------------------------------------
# Case 12 — a junk maintainership_level value ("none") must not route to
# oss_maintainer; falls through to the clean engineering title instead.
# ---------------------------------------------------------------------------


def test_junk_maintainership_value_does_not_route_oss() -> None:
    brief = _stub_brief(
        maintainership_level="none",
        role_title="Senior Software Engineer",
        role_level="Senior",
    )
    profile_id, _metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "ic_frontier_engineer"


# ---------------------------------------------------------------------------
# Case 13 — a bare target_projects list, with no github module and no
# recognized maintainership_level, no longer flips the profile by itself.
# ---------------------------------------------------------------------------


def test_bare_target_projects_does_not_flip_to_oss() -> None:
    brief = _stub_brief(
        target_projects=["vllm"],
        maintainership_level="",
        role_title="Senior ML Engineer",
        role_level="Senior",
    )
    profile_id, _metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "ic_frontier_engineer"


# ---------------------------------------------------------------------------
# Case 14 — an explicit role_strategy_profile override always wins, with
# profile_source "explicit".
# ---------------------------------------------------------------------------


def test_explicit_override_routes_designer_with_explicit_source() -> None:
    brief = _stub_brief(role_strategy_profile="designer")
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "designer"
    assert metadata["profile_source"] == "explicit"


# ---------------------------------------------------------------------------
# Case 15 — a clean FDE title routes fde_enterprise_genai.
# ---------------------------------------------------------------------------


def test_forward_deployed_engineer_title_routes_fde() -> None:
    brief = _stub_brief(role_title="Forward Deployed Engineer")
    profile_id, _metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "fde_enterprise_genai"


# ---------------------------------------------------------------------------
# Case 16 — an empty/sparse brief falls back to generic with
# profile_source "generic_fallback".
# ---------------------------------------------------------------------------


def test_empty_brief_falls_back_to_generic() -> None:
    brief = _stub_brief()
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "generic"
    assert metadata["profile_source"] == "generic_fallback"


# ---------------------------------------------------------------------------
# Instructions-pollution case — academic terms living only in `instructions`
# (a command about the search, not a description of the role) must not
# reach the matched text at all, let alone route academic_researcher.
# ---------------------------------------------------------------------------


def test_academic_terms_in_instructions_do_not_route_academic() -> None:
    brief = _stub_brief(
        role_title="Senior Machine Learning Engineer",
        role_level="Senior",
        instructions=["exclude professors and postdocs"],
    )
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "ic_frontier_engineer"
    assert "academic_title" not in metadata["matched_signals"]
    assert "academic_patterns_x2" not in metadata["matched_signals"]


# Case 18 — "project_lead" (the schema's top maintainership tier per
# RECOGNIZED_MAINTAINERSHIP_LEVELS in shared/brief_v2_schema.py) routes
# oss_maintainer. Locks the allowlist to the codebase's real enum.
def test_project_lead_maintainership_routes_oss_maintainer():
    brief = _stub_brief(maintainership_level="project_lead")
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "oss_maintainer"
    assert metadata["profile_source"] == "inferred"


# Case 19 — an explicit engineering-IC title is not overridden by academic
# body vocabulary ("PhD" + "research scientist" are among the most common
# phrases in ML-engineer JDs). Correctness lens, Wave 1.
def test_frontier_ic_title_beats_academic_body_vocabulary():
    brief = _stub_brief(
        role_title="Staff ML Engineer",
        role_level="Staff",
        role_description=(
            "Work alongside PhD researchers; collaboration with our "
            "research scientist team on training pipelines."
        ),
    )
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    assert profile_id == "ic_frontier_engineer"
    assert "academic_patterns_x2" not in metadata["matched_signals"]
