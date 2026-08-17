"""Role-class strategy profile inference — MECHANISM ONLY (P8 / Wave 2 slice 9).

Role-class profiles seed slice/lane templates and ambiguity posture as hints.
They are not a hard taxonomy — when no profile fits, inference falls back to
brief-field signals safely.

Wave 2 (plans/sourcing-rigor-hardening.md, addendum item 6): this module
carries ZERO vertical vocabulary. The eight built-in profiles, every routing
pattern list, and the ordered trigger-rule registry live in
shared/role_strategy_profiles.py (profile DATA — vertical templates by
definition). This module defines the trigger-rule SHAPES and the generic
iteration that evaluates them, so the vertical-vocab ratchet holds an EMPTY
allowlist here — no carve-out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from shared.sourcing_lanes import SourcingLane, normalize_lane_id
from shared.strict_seniority import is_strict_seniority_brief

PROFILE_IDS: frozenset[str] = frozenset(
    {
        "senior_bfs_ai_leader",
        "fde_enterprise_genai",
        "ic_frontier_engineer",
        "oss_maintainer",
        "academic_researcher",
        "designer",
        "executive_search",
        "generic",
    }
)


@dataclass
class RoleStrategyProfile:
    profile_id: str
    label: str
    source_defaults: dict[str, Any] = field(default_factory=dict)
    lane_templates: list[SourcingLane] = field(default_factory=list)
    ambiguity_defaults: dict[str, Any] = field(default_factory=dict)
    boolean_defaults: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "label": self.label,
            "source_defaults": dict(self.source_defaults),
            "lane_templates": [lane.to_dict() for lane in self.lane_templates],
            "ambiguity_defaults": dict(self.ambiguity_defaults),
            "boolean_defaults": dict(self.boolean_defaults),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> RoleStrategyProfile:
        payload = payload or {}
        lane_payloads = payload.get("lane_templates", [])
        lane_templates = [
            SourcingLane.from_dict(item)
            for item in lane_payloads
            if isinstance(item, dict)
        ]
        return cls(
            profile_id=str(payload.get("profile_id", "")),
            label=str(payload.get("label", "")),
            source_defaults=dict(payload.get("source_defaults", {})),
            lane_templates=lane_templates,
            ambiguity_defaults=dict(payload.get("ambiguity_defaults", {})),
            boolean_defaults=dict(payload.get("boolean_defaults", {})),
        )


# ---------------------------------------------------------------------------
# Brief readers (mechanism — no vocabulary)
# ---------------------------------------------------------------------------


def _brief_raw(brief: Any) -> dict[str, Any]:
    raw = getattr(brief, "raw", None)
    return dict(raw) if isinstance(raw, dict) else {}


def _brief_text(brief: Any) -> str:
    # NOTE (rule 1): `instructions` are deliberately excluded — they are
    # commands about the search (e.g. "exclude professors and postdocs"),
    # not descriptions of the role, and including them let JD/instruction
    # boilerplate flip the whole profile (the audit's finding (b)/(c)).
    parts = [
        str(getattr(brief, "role_title", "") or ""),
        str(getattr(brief, "role_description", "") or ""),
        str(_brief_raw(brief).get("role_summary", "") or ""),
        str(_brief_raw(brief).get("intake_notes", "") or ""),
    ]
    new_brief = getattr(brief, "_new_brief", None)
    if new_brief is not None:
        parts.append(str(getattr(new_brief, "role_summary", "") or ""))
        for area in getattr(new_brief, "capability_areas", []) or []:
            parts.append(str(getattr(area, "name", "") or ""))
            parts.append(str(getattr(area, "description", "") or ""))
    return " ".join(part for part in parts if part).lower()


def _brief_role_level(brief: Any) -> str:
    raw = _brief_raw(brief)
    level = raw.get("role_level", "")
    if level:
        return str(level)
    new_brief = getattr(brief, "_new_brief", None)
    if new_brief is not None:
        return str(getattr(new_brief, "role_level", "") or "")
    return ""


def _title_text(brief: Any) -> str:
    """Title-only text for the TITLE-CLASS pattern sets (rule 3).

    Deliberately narrower than :func:`_brief_text` — role_description /
    role_summary / intake_notes / capability-area copy must never feed the
    title-scoped matchers, or JD boilerplate ("collaborate with product
    designers", "reports to the CTO") flips the profile the way the audit
    found.
    """
    role_title = str(getattr(brief, "role_title", "") or "")
    role_level = _brief_role_level(brief)
    return f"{role_title} {role_level}".lower()


def _brief_capability_names(brief: Any) -> list[str]:
    new_brief = getattr(brief, "_new_brief", None)
    if new_brief is not None:
        return [
            str(getattr(area, "name", "") or "").strip()
            for area in getattr(new_brief, "capability_areas", []) or []
            if str(getattr(area, "name", "") or "").strip()
        ]
    raw = _brief_raw(brief)
    names: list[str] = []
    for area in raw.get("capability_areas", []) or []:
        if isinstance(area, dict):
            name = str(area.get("name", "") or "").strip()
            if name:
                names.append(name)
    return names


def _brief_target_modules(brief: Any) -> list[str]:
    modules = getattr(brief, "target_modules", None) or _brief_raw(brief).get("target_modules", [])
    return [str(item).strip().lower() for item in modules or [] if str(item).strip()]


def _explicit_profile_id(brief: Any) -> str:
    raw = _brief_raw(brief)
    explicit = str(raw.get("role_strategy_profile", "") or "").strip()
    if explicit:
        return explicit
    new_brief = getattr(brief, "_new_brief", None)
    if new_brief is not None:
        return str(getattr(new_brief, "role_strategy_profile", "") or "").strip()
    return ""


def _matches_any(text: str, patterns: tuple[str, ...]) -> list[str]:
    """Word-boundary pattern matches; returns the distinct patterns that hit."""
    lowered = text.lower()
    hits: list[str] = []
    for pattern in patterns:
        stripped = pattern.strip()
        if not stripped:
            continue
        if re.search(rf"\b{re.escape(stripped)}\b", lowered):
            hits.append(pattern)
    return hits


def _is_senior_level(role_level: str) -> bool:
    level = role_level.strip().upper()
    if not level:
        return False
    return any(token in level for token in ("VP", "SVP", "DIRECTOR", "HEAD", "EXEC", "FELLOW", "DISTINGUISHED"))


def _is_ic_level(role_level: str) -> bool:
    level = role_level.strip().upper()
    if not level:
        return False
    return level.startswith("IC") or any(token in level for token in ("STAFF", "PRINCIPAL", "SENIOR"))


# ---------------------------------------------------------------------------
# Trigger-rule SHAPES (mechanism). Vocabulary arrives as constructor data
# from shared/role_strategy_profiles.py's PROFILE_TRIGGER_RULES registry.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerContext:
    text: str
    title_text: str
    role_level: str
    target_modules: tuple[str, ...]
    maintainership_level: str
    has_target_projects: bool
    strict_seniority: bool
    engineering_text: str


@dataclass(frozen=True)
class TriggerOutcome:
    matched: bool
    signals: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModuleTrigger:
    module: str
    signal: str
    profile_id: str
    rank: int

    def evaluate(self, ctx: TriggerContext) -> TriggerOutcome:
        if self.module in ctx.target_modules:
            return TriggerOutcome(True, (self.signal,))
        return TriggerOutcome(False)


@dataclass(frozen=True)
class ModuleWithDomainTrigger:
    module: str
    module_signal: str
    domain_patterns: tuple[str, ...]
    domain_signal: str
    profile_id: str
    rank: int

    def evaluate(self, ctx: TriggerContext) -> TriggerOutcome:
        if self.module in ctx.target_modules and _matches_any(ctx.text, self.domain_patterns):
            return TriggerOutcome(True, (self.module_signal, self.domain_signal))
        return TriggerOutcome(False)


@dataclass(frozen=True)
class MaintainerTrigger:
    module: str
    levels: frozenset[str]
    signal: str
    projects_signal: str
    profile_id: str
    rank: int

    def evaluate(self, ctx: TriggerContext) -> TriggerOutcome:
        if self.module in ctx.target_modules or ctx.maintainership_level in self.levels:
            signals: tuple[str, ...] = (self.signal,)
            if ctx.has_target_projects:
                signals = signals + (self.projects_signal,)
            return TriggerOutcome(True, signals)
        return TriggerOutcome(False)


@dataclass(frozen=True)
class TitleTrigger:
    title_patterns: tuple[str, ...]
    signal: str
    profile_id: str
    rank: int

    def evaluate(self, ctx: TriggerContext) -> TriggerOutcome:
        if _matches_any(ctx.title_text, self.title_patterns):
            return TriggerOutcome(True, (self.signal,))
        return TriggerOutcome(False)


@dataclass(frozen=True)
class SeniorDomainCompositeTrigger:
    domain_patterns: tuple[str, ...]
    domain_signal: str
    title_patterns: tuple[str, ...]
    title_signal: str
    strict_signal: str
    profile_id: str
    rank: int

    def evaluate(self, ctx: TriggerContext) -> TriggerOutcome:
        domain_hit = bool(_matches_any(ctx.text, self.domain_patterns))
        title_hit = bool(_matches_any(ctx.title_text, self.title_patterns))
        matched = ctx.strict_seniority or (
            domain_hit and title_hit and _is_senior_level(ctx.role_level)
        )
        if not matched:
            return TriggerOutcome(False)
        signals: list[str] = []
        if ctx.strict_seniority:
            signals.append(self.strict_signal)
        if domain_hit:
            signals.append(self.domain_signal)
        if title_hit:
            signals.append(self.title_signal)
        return TriggerOutcome(
            True, tuple(signals), {"strict_seniority": ctx.strict_seniority}
        )


@dataclass(frozen=True)
class TitleWithLevelTrigger:
    title_patterns: tuple[str, ...]
    signal: str
    profile_id: str
    rank: int

    def evaluate(self, ctx: TriggerContext) -> TriggerOutcome:
        if _matches_any(ctx.title_text, self.title_patterns) and _is_senior_level(ctx.role_level):
            return TriggerOutcome(True, (self.signal,))
        return TriggerOutcome(False)


@dataclass(frozen=True)
class TitleOrBodyDomainTrigger:
    title_patterns: tuple[str, ...]
    title_signal: str
    body_patterns: tuple[str, ...]
    body_min_hits: int
    body_signal: str
    yield_to_title_patterns: tuple[str, ...]
    profile_id: str
    rank: int

    def evaluate(self, ctx: TriggerContext) -> TriggerOutcome:
        title_hits = _matches_any(ctx.title_text, self.title_patterns)
        body_hits = _matches_any(ctx.text, self.body_patterns)
        yield_hit = bool(_matches_any(ctx.title_text, self.yield_to_title_patterns))
        # Body-vocabulary routing yields to an explicit yield-class title
        # (correctness lens, Wave 1): a body-only pattern shower must not
        # out-route an unambiguous title. A TITLE hit routes regardless.
        body_route = len(body_hits) >= self.body_min_hits and not yield_hit
        if not (title_hits or body_route):
            return TriggerOutcome(False)
        signals: list[str] = []
        if title_hits:
            signals.append(self.title_signal)
        if body_route:
            signals.append(self.body_signal)
        return TriggerOutcome(True, tuple(signals))


@dataclass(frozen=True)
class TitleOrLevelSignalTrigger:
    title_patterns: tuple[str, ...]
    title_signal: str
    engineering_terms: tuple[str, ...]
    arm_signal: str
    profile_id: str
    rank: int

    def evaluate(self, ctx: TriggerContext) -> TriggerOutcome:
        title_hit = bool(_matches_any(ctx.title_text, self.title_patterns))
        level_arm = _is_ic_level(ctx.role_level) and bool(
            _matches_any(ctx.engineering_text, self.engineering_terms)
        )
        if not (title_hit or level_arm):
            return TriggerOutcome(False)
        signals: list[str] = []
        if title_hit:
            signals.append(self.title_signal)
        if level_arm:
            signals.append(self.arm_signal)
        return TriggerOutcome(True, tuple(signals))


def _build_trigger_context(brief: Any) -> TriggerContext:
    role_title = str(getattr(brief, "role_title", "") or "")
    capability_names = _brief_capability_names(brief)
    target_projects = getattr(brief, "target_projects", None) or _brief_raw(brief).get(
        "target_projects", []
    )
    maintainership_level = str(
        getattr(brief, "maintainership_level", "")
        or _brief_raw(brief).get("maintainership_level", "")
        or ""
    ).lower()
    return TriggerContext(
        text=_brief_text(brief),
        title_text=_title_text(brief),
        role_level=_brief_role_level(brief),
        target_modules=tuple(_brief_target_modules(brief)),
        maintainership_level=maintainership_level,
        has_target_projects=bool(target_projects),
        strict_seniority=is_strict_seniority_brief(brief),
        engineering_text=f"{role_title} {' '.join(capability_names)}",
    )


# ---------------------------------------------------------------------------
# Generic fallback (brief-derived — mechanism, no vocabulary)
# ---------------------------------------------------------------------------


def _generic_profile_from_brief(brief: Any, *, matched_signals: list[str]) -> RoleStrategyProfile:
    from shared.role_strategy_profiles import _lane

    capability_names = _brief_capability_names(brief)
    lane_templates: list[SourcingLane] = []
    for idx, name in enumerate(capability_names[:4]):
        lane_id = normalize_lane_id(name) or f"generic_capability_{idx + 1}"
        lane_templates.append(
            _lane(
                lane_id=lane_id,
                lane_name=name,
                target_archetype=f"Builders with evidence in {name}",
                objective=f"Seed a capability-led lane from brief field {name!r}.",
                why="No role-class profile matched; infer safely from explicit brief capability areas.",
                capability_signals=[name],
                priority=10 + idx * 10,
            )
        )
    if not lane_templates:
        lane_templates.append(
            _lane(
                lane_id="generic_role_scope",
                lane_name="Role-scope discovery",
                target_archetype=str(getattr(brief, "role_title", "") or "target role"),
                objective="Use role title and summary as a soft discovery hint.",
                why="Legacy or sparse briefs fall back to role-scope hints without forcing a taxonomy.",
                capability_signals=[str(getattr(brief, "role_title", "") or "").strip()],
                priority=50,
            )
        )
    return RoleStrategyProfile(
        profile_id="generic",
        label="Generic brief-derived hints",
        source_defaults={"primary_source": "linkedin"},
        ambiguity_defaults={"mode": "resolve"},
        boolean_defaults={"title_anchor_strength": "medium", "inferred_from_brief_fields": True},
        lane_templates=lane_templates,
    )


def get_builtin_profile(profile_id: str) -> RoleStrategyProfile | None:
    from shared.role_strategy_profiles import _BUILTIN_PROFILES

    return _BUILTIN_PROFILES.get(profile_id)


def infer_role_strategy_profile_id(brief: Any) -> tuple[str, dict[str, Any]]:
    """Infer a role-class profile id and structured inference metadata.

    Generic iteration over the profile-declared trigger registry
    (shared/role_strategy_profiles.PROFILE_TRIGGER_RULES) in rank order —
    the ladder's ordering and signal labels are profile DATA, not code.
    """
    from shared.role_strategy_profiles import _BUILTIN_PROFILES, PROFILE_TRIGGER_RULES

    explicit = _explicit_profile_id(brief)
    if explicit:
        if explicit in PROFILE_IDS or explicit in _BUILTIN_PROFILES:
            return explicit, {
                "profile_source": "explicit",
                "matched_signals": ["role_strategy_profile"],
            }
        return explicit, {
            "profile_source": "explicit_unknown",
            "matched_signals": ["role_strategy_profile"],
            "warning": f"Unknown explicit profile {explicit!r}; treating as explicit override",
        }

    ctx = _build_trigger_context(brief)
    for rule in sorted(PROFILE_TRIGGER_RULES, key=lambda r: r.rank):
        outcome = rule.evaluate(ctx)
        if outcome.matched:
            metadata: dict[str, Any] = {
                "profile_source": "inferred",
                "matched_signals": list(outcome.signals),
            }
            if outcome.metadata:
                metadata.update(outcome.metadata)
            return rule.profile_id, metadata

    matched_signals: list[str] = []
    generic = _generic_profile_from_brief(brief, matched_signals=matched_signals)
    return "generic", {
        "profile_source": "generic_fallback",
        "matched_signals": matched_signals,
        "fallback_lane_count": len(generic.lane_templates),
    }


def resolve_role_strategy_profile(brief: Any) -> tuple[RoleStrategyProfile, dict[str, Any]]:
    profile_id, metadata = infer_role_strategy_profile_id(brief)
    profile = get_builtin_profile(profile_id)
    if profile is None:
        if profile_id == "generic":
            profile = _generic_profile_from_brief(brief, matched_signals=metadata.get("matched_signals", []))
        else:
            profile = RoleStrategyProfile(
                profile_id=profile_id,
                label=profile_id.replace("_", " ").title(),
                source_defaults={"primary_source": "linkedin"},
                ambiguity_defaults={"mode": "resolve"},
                boolean_defaults={"title_anchor_strength": "medium"},
            )
    metadata = {
        **metadata,
        "profile_id": profile.profile_id,
        "profile_label": profile.label,
    }
    return profile, metadata


def _merge_hint_lanes(plan: Any, profile: RoleStrategyProfile) -> None:
    existing_lane_ids = {
        normalize_lane_id(str(item.get("lane_id", "")))
        for item in getattr(plan, "sourcing_lanes", []) or []
        if isinstance(item, dict)
    }
    hint_lanes: list[dict[str, Any]] = []
    hint_hypotheses: list[dict[str, Any]] = []
    hint_slices: list[dict[str, Any]] = []
    for lane in profile.lane_templates:
        lane_id = normalize_lane_id(lane.lane_id)
        if lane_id and lane_id in existing_lane_ids:
            continue
        payload = lane.to_dict()
        hint_lanes.append(payload)
        hint_hypotheses.append(payload["hypothesis"])
        hint_slices.append(payload["slice"])
        if lane_id:
            existing_lane_ids.add(lane_id)
    if not hint_lanes:
        return
    plan.sourcing_lanes = list(getattr(plan, "sourcing_lanes", []) or []) + hint_lanes
    plan.search_hypotheses = list(getattr(plan, "search_hypotheses", []) or []) + hint_hypotheses
    plan.search_slices = list(getattr(plan, "search_slices", []) or []) + hint_slices


def apply_role_strategy_to_plan(
    brief: Any,
    plan: Any,
    *,
    merge_lane_templates: bool = True,
) -> dict[str, Any]:
    """Attach role-class metadata and optionally seed hint lane templates on ``plan``.

    Does not mutate ``generated_strings`` — those remain the execution contract.
    LinkedIn passes ``merge_lane_templates=True``; other modules should pass
    ``False`` until source-native lane templates are compiled.
    """
    profile, metadata = resolve_role_strategy_profile(brief)
    plan.role_strategy_profile = profile.profile_id
    plan.role_strategy_metadata = {
        **metadata,
        "source_defaults": dict(profile.source_defaults),
        "ambiguity_defaults": dict(profile.ambiguity_defaults),
        "boolean_defaults": dict(profile.boolean_defaults),
    }
    if merge_lane_templates:
        _merge_hint_lanes(plan, profile)
    return metadata
