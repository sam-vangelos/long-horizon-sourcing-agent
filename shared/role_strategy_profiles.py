"""Role-class profile DATA for shared/role_strategy.py's routing mechanism.

This file is profile data — vertical vocabulary is expected and exempt from
the vertical-vocab ratchet by file-level exclusion. Every profile-specific
pattern list (title/domain vocabulary for BFS, FDE, IC-frontier, academic,
design, executive, OSS-maintainer routing), the eight built-in profile
factories, the `_lane` construction helper they share, and the ordered
`PROFILE_TRIGGER_RULES` registry live here by design. `shared/role_strategy.py`
is the mechanism — it defines the trigger-rule SHAPES (imported below) and
contains none of this vocabulary itself.
"""

from __future__ import annotations

from shared.role_strategy import (
    MaintainerTrigger,
    ModuleTrigger,
    ModuleWithDomainTrigger,
    RoleStrategyProfile,
    SeniorDomainCompositeTrigger,
    TitleOrBodyDomainTrigger,
    TitleOrLevelSignalTrigger,
    TitleTrigger,
    TitleWithLevelTrigger,
)
from shared.sourcing_lanes import (
    DEFAULT_ACQUISITION_MODE,
    DEFAULT_SEARCH_POSTURE,
    LaneExecution,
    SearchConstraint,
    SearchHypothesis,
    SearchSlice,
    SourcingLane,
)

_FDE_TITLE_PATTERNS = (
    "forward deployed",
    "forward-deployed",
    "fde",
    "fdse",
    "customer engineer",
    "solutions engineer",
    "field engineer",
    "implementation engineer",
    "delivery engineer",
)

_BFS_DOMAIN_PATTERNS = (
    "bfs",
    "bfsi",
    "bank",
    "banking",
    "financial services",
    "capital markets",
    "investment bank",
)

_SENIOR_AI_LEADER_PATTERNS = (
    "head of applied ai",
    "head of ai",
    "head of ai lab",
    "applied ai lab",
    "ai lab",
    "executive director ai",
    "vp ai",
    "svp ai",
)

_FRONTIER_IC_PATTERNS = (
    "staff engineer",
    "principal engineer",
    "senior engineer",
    "research engineer",
    "ml engineer",
    "machine learning engineer",
    "systems engineer",
    "software engineer",
)

_ACADEMIC_PATTERNS = (
    "professor",
    "postdoc",
    "phd",
    "research scientist",
    "assistant professor",
    "associate professor",
    "faculty",
    "tenure",
)

_DESIGN_PATTERNS = (
    "product designer",
    "ux designer",
    "ui designer",
    "design lead",
    "design director",
    "visual designer",
    "interaction designer",
)

_EXEC_PATTERNS = (
    "chief",
    "cto",
    "cpo",
    "cfo",
    "coo",
    "ceo",
    "president",
    "executive vice president",
    "evp",
    "svp",
    "managing director",
)

# Membership set for the OSS Maintainers routing gate. Mirrors the elevated
# tiers of `RECOGNIZED_MAINTAINERSHIP_LEVELS` in shared/brief_v2_schema.py
# (`{"contributor", "maintainer", "project_lead"}` minus "contributor") and
# `_ELEVATED_MAINTAINERSHIP_LEVELS` in github/health.py — the two existing
# authorities for what counts as maintainer-tier. Junk values ("none",
# "n/a") and "contributor" must NOT route a brief to the oss_maintainer
# profile.
_MAINTAINER_LEVELS = frozenset({"maintainer", "project_lead"})

# Word-boundary engineering-signal vocabulary for the ic_frontier_engineer
# gate's second arm — TitleOrLevelSignalTrigger's `_is_ic_level(role_level)
# and a word-boundary hit in ctx.engineering_text` check (rank 100 in
# PROFILE_TRIGGER_RULES below).
_ENGINEERING_SIGNAL_TERMS = (
    "engineer",
    "engineering",
    "developer",
    "swe",
    "software",
    "machine learning",
    "ml",
    "infrastructure",
    "platform",
    "backend",
    "frontend",
    "full stack",
    "full-stack",
    "systems",
)


def _lane(
    *,
    lane_id: str,
    lane_name: str,
    target_archetype: str,
    objective: str,
    why: str,
    capability_signals: list[str],
    constraints: list[SearchConstraint] | None = None,
    acquisition_mode: str = DEFAULT_ACQUISITION_MODE,
    search_posture: str = DEFAULT_SEARCH_POSTURE,
    ambiguity_mode: str = "preserve",
    source: str = "linkedin",
    priority: int = 50,
) -> SourcingLane:
    hypothesis = SearchHypothesis(
        hypothesis_id=lane_id,
        label=lane_name,
        target_archetype=target_archetype,
        why_this_pool_may_exist=why,
        capability_signals=capability_signals,
        source="role_strategy_profile",
    )
    search_slice = SearchSlice(
        slice_id=f"{lane_id}_slice",
        hypothesis_id=lane_id,
        label=lane_name,
        objective=objective,
        constraints=constraints or [],
        priority=priority,
    )
    execution = LaneExecution(
        lane_id=lane_id,
        source=source,
        acquisition_mode=acquisition_mode,
        search_posture=search_posture,
    )
    return SourcingLane(
        lane_id=lane_id,
        lane_name=lane_name,
        hypothesis=hypothesis,
        slice=search_slice,
        execution=execution,
        ambiguity_policy={"mode": ambiguity_mode},
    )


def _senior_bfs_ai_leader_profile() -> RoleStrategyProfile:
    return RoleStrategyProfile(
        profile_id="senior_bfs_ai_leader",
        label="Senior BFS Applied AI Leader",
        source_defaults={"primary_source": "linkedin", "secondary_sources": ["researcher"]},
        ambiguity_defaults={
            "mode": "preserve",
            "allow_non_save_review": True,
            "sparse_senior_inference": True,
        },
        boolean_defaults={
            "title_anchor_strength": "medium",
            "prefer_narrow_titles": True,
            "domain_anchor_strength": "high",
        },
        lane_templates=[
            _lane(
                lane_id="bfs_senior_obvious_pool",
                lane_name="BFS senior obvious pool",
                target_archetype="Senior bank AI leaders with explicit GenAI scope",
                objective="Open with bank-native senior titles plus applied AI proof.",
                why="Large obvious pool of bank technologists already labeled as AI leaders.",
                capability_signals=["applied ai", "genai", "agentic systems"],
                constraints=[
                    SearchConstraint(
                        dimension="domain",
                        values=["banking", "financial services", "capital markets"],
                        operator="prefer",
                        execution_surface="boolean_keyword",
                    ),
                    SearchConstraint(
                        dimension="seniority",
                        values=["executive director", "head of applied ai", "head of ai"],
                        operator="prefer",
                        # Slice A part 2: these are literal titles LinkedIn indexes as a
                        # facet that bounds the pool -> structured title filter, not a
                        # Boolean keyword. The domain constraint above stays boolean_keyword
                        # (industry semantics, not a company facet). Company-name constraints
                        # are deliberately NOT hardcoded into this generic template — they
                        # belong on the dynamic prompt path where a specific brief names real
                        # employers (over-narrowing every BFS search otherwise).
                        execution_surface="linkedin_title_filter",
                    ),
                ],
                priority=10,
            ),
            _lane(
                lane_id="bfs_ed_analogs",
                lane_name="BFS ED analogs",
                target_archetype="Executive Director analog leaders one layer below broad enterprise executives",
                objective="Probe ED-analog titles with AI platform or lab scope.",
                why="Strict-seniority briefs often map to ED analogs rather than broad MD buckets.",
                capability_signals=["ai platform", "lab scope", "executive director"],
                priority=20,
            ),
            _lane(
                lane_id="bfs_hidden_technical_leader",
                lane_name="Hidden technical leader",
                target_archetype="Senior technologists with AI-adjacent org scope but sparse GenAI copy",
                objective="Preserve sparse senior profiles with structural bank + technical education signals.",
                why="High-upside leaders may not advertise GenAI vocabulary explicitly.",
                capability_signals=["distinguished engineer", "principal architect", "cs education"],
                ambiguity_mode="preserve",
                priority=30,
            ),
            _lane(
                lane_id="bfs_semantic_discovery",
                lane_name="Creative semantic discovery",
                target_archetype="Agentic workflow and orchestration leaders in BFS contexts",
                objective="Use semantic capability discovery beyond literal title buckets.",
                why="Creative semantic lanes surface non-obvious agentic systems builders.",
                capability_signals=["agentic workflows", "orchestration", "evaluation harness"],
                search_posture="boolean_led",
                priority=40,
            ),
        ],
    )


def _fde_enterprise_genai_profile() -> RoleStrategyProfile:
    return RoleStrategyProfile(
        profile_id="fde_enterprise_genai",
        label="FDE / Enterprise GenAI",
        source_defaults={"primary_source": "linkedin", "secondary_sources": ["github"]},
        ambiguity_defaults={"mode": "resolve", "require_builder_proof": True},
        boolean_defaults={
            "title_anchor_strength": "low",
            "capability_led": True,
            "require_deployment_proof": True,
        },
        lane_templates=[
            _lane(
                lane_id="fde_capability_led",
                lane_name="Capability-led FDE pool",
                target_archetype="Forward-deployed builders with workflow and orchestration proof",
                objective="Lead with capability terms and lighter title anchors.",
                why="FDE pools are often hidden behind delivery and customer engineering titles.",
                capability_signals=["workflow orchestration", "tool calling", "agent platform"],
                priority=10,
            ),
            _lane(
                lane_id="fde_customer_deployment_proof",
                lane_name="Customer deployment proof",
                target_archetype="Builders with production customer deployment evidence",
                objective="Require production, deployment, or customer-facing delivery proof.",
                why="Generic seniority without builder proof is noisy for FDE searches.",
                capability_signals=["production deployment", "customer onboarding", "enterprise rollout"],
                constraints=[
                    SearchConstraint(
                        dimension="evidence",
                        values=["production", "deployment", "customer", "enterprise"],
                        operator="prefer",
                        execution_surface="boolean_keyword",
                    ),
                ],
                priority=20,
            ),
            _lane(
                lane_id="fde_enterprise_platform",
                lane_name="Enterprise GenAI platform integrators",
                target_archetype="Enterprise GenAI platform engineers integrating models into customer workflows",
                objective="Target platform, guardrails, and evaluation infrastructure builders.",
                why="Enterprise GenAI FDE work often sits on shared platform teams.",
                capability_signals=["llm platform", "guardrails", "evaluation", "governance"],
                priority=30,
            ),
        ],
    )


def _ic_frontier_engineer_profile() -> RoleStrategyProfile:
    return RoleStrategyProfile(
        profile_id="ic_frontier_engineer",
        label="IC Frontier Engineer",
        source_defaults={"primary_source": "linkedin", "secondary_sources": ["github"]},
        ambiguity_defaults={"mode": "resolve", "require_builder_proof": True},
        boolean_defaults={"title_anchor_strength": "medium", "capability_led": True},
        lane_templates=[
            _lane(
                lane_id="ic_frontier_systems",
                lane_name="Frontier systems builders",
                target_archetype="IC engineers building frontier systems and infrastructure",
                objective="Open with systems, infra, and training-adjacent capability signals.",
                why="Frontier IC pools are capability-led rather than title-led.",
                capability_signals=["distributed systems", "training infrastructure", "inference"],
                priority=10,
            ),
            _lane(
                lane_id="ic_research_engineering",
                lane_name="Research engineering hybrids",
                target_archetype="Engineers spanning research prototypes and production systems",
                objective="Blend research and production builder vocabulary.",
                why="Frontier IC roles often sit between research and product engineering.",
                capability_signals=["research engineering", "prototype to production", "eval harness"],
                priority=20,
            ),
        ],
    )


def _oss_maintainer_profile() -> RoleStrategyProfile:
    return RoleStrategyProfile(
        profile_id="oss_maintainer",
        label="OSS Maintainer",
        source_defaults={"primary_source": "github", "secondary_sources": ["linkedin"]},
        ambiguity_defaults={"mode": "resolve"},
        boolean_defaults={"title_anchor_strength": "low"},
        lane_templates=[
            _lane(
                lane_id="oss_core_maintainers",
                lane_name="Core maintainers",
                target_archetype="Maintainers with sustained contribution history on target projects",
                objective="Prioritize repo/org mining over title search.",
                why="Maintainer signal lives in contribution graphs, not LinkedIn titles.",
                capability_signals=["maintainer", "core contributor", "commit history"],
                acquisition_mode="github",
                source="github",
                priority=10,
            ),
            _lane(
                lane_id="oss_ecosystem_adjacent",
                lane_name="Ecosystem-adjacent builders",
                target_archetype="Adjacent contributors in the same dependency or topic neighborhood",
                objective="Expand to dependency and topic adjacency when core maintainer pool is sparse.",
                why="High-upside maintainers may appear on sibling projects first.",
                capability_signals=["dependency graph", "topic cluster", "downstream adopter"],
                acquisition_mode="github",
                source="github",
                priority=20,
            ),
        ],
    )


def _academic_researcher_profile() -> RoleStrategyProfile:
    return RoleStrategyProfile(
        profile_id="academic_researcher",
        label="Academic Researcher",
        source_defaults={"primary_source": "researcher", "secondary_sources": ["linkedin"]},
        ambiguity_defaults={"mode": "preserve"},
        boolean_defaults={"title_anchor_strength": "low"},
        lane_templates=[
            _lane(
                lane_id="academic_topic_core",
                lane_name="Core topic researchers",
                target_archetype="Researchers publishing on the brief's core topics",
                objective="Lead with topic, venue, and citation signals.",
                why="Academic pools are topic-native rather than title-native.",
                capability_signals=["topic cluster", "venue prestige", "citation velocity"],
                acquisition_mode="researcher",
                source="researcher",
                priority=10,
            ),
            _lane(
                lane_id="academic_method_adjacent",
                lane_name="Method-adjacent researchers",
                target_archetype="Researchers using adjacent methods or cross-disciplinary venues",
                objective="Probe method neighborhoods when the core topic pool is narrow.",
                why="Important researchers may publish under adjacent method labels.",
                capability_signals=["method transfer", "cross venue", "lab lineage"],
                acquisition_mode="researcher",
                source="researcher",
                priority=20,
            ),
        ],
    )


def _designer_profile() -> RoleStrategyProfile:
    return RoleStrategyProfile(
        profile_id="designer",
        label="Designer",
        source_defaults={"primary_source": "designer", "secondary_sources": ["linkedin"]},
        ambiguity_defaults={"mode": "resolve"},
        boolean_defaults={"title_anchor_strength": "medium"},
        lane_templates=[
            _lane(
                lane_id="designer_portfolio_core",
                lane_name="Portfolio-core designers",
                target_archetype="Designers with portfolio evidence in the target product category",
                objective="Prioritize portfolio medium, product category, and system complexity.",
                why="Designer signal is portfolio-native rather than keyword-native.",
                capability_signals=["portfolio medium", "product category", "design system"],
                acquisition_mode="designer",
                source="designer",
                priority=10,
            ),
            _lane(
                lane_id="designer_toolchain_adjacent",
                lane_name="Toolchain-adjacent designers",
                target_archetype="Designers with toolchain and workflow complexity proof",
                objective="Expand to toolchain and workflow complexity when the core pool is sparse.",
                why="Strong designers may appear under adjacent product or toolchain labels.",
                capability_signals=["figma", "prototyping", "design ops"],
                acquisition_mode="designer",
                source="designer",
                priority=20,
            ),
        ],
    )


def _executive_search_profile() -> RoleStrategyProfile:
    return RoleStrategyProfile(
        profile_id="executive_search",
        label="Executive Search",
        source_defaults={"primary_source": "exec_search", "secondary_sources": ["linkedin"]},
        ambiguity_defaults={"mode": "preserve", "sparse_executive_inference": True},
        boolean_defaults={"title_anchor_strength": "high", "prefer_org_scope": True},
        lane_templates=[
            _lane(
                lane_id="exec_title_path",
                lane_name="Title-path executives",
                target_archetype="Executives on the expected title path for the role",
                objective="Open with title path, org scope, and company stage signals.",
                why="Executive searches are title- and scope-native.",
                capability_signals=["org scope", "company stage", "board signals"],
                acquisition_mode="exec_search",
                source="exec_search",
                priority=10,
            ),
            _lane(
                lane_id="exec_market_segment",
                lane_name="Market-segment executives",
                target_archetype="Executives with domain-specific market segment leadership",
                objective="Probe market segment and adjacent company archetypes.",
                why="Executive pools fragment by market segment and company stage.",
                capability_signals=["market segment", "growth stage", "transformation"],
                acquisition_mode="exec_search",
                source="exec_search",
                priority=20,
            ),
        ],
    )


_BUILTIN_PROFILES: dict[str, RoleStrategyProfile] = {
    "senior_bfs_ai_leader": _senior_bfs_ai_leader_profile(),
    "fde_enterprise_genai": _fde_enterprise_genai_profile(),
    "ic_frontier_engineer": _ic_frontier_engineer_profile(),
    "oss_maintainer": _oss_maintainer_profile(),
    "academic_researcher": _academic_researcher_profile(),
    "designer": _designer_profile(),
    "executive_search": _executive_search_profile(),
}


# Ordered trigger-rule registry for `infer_role_strategy_profile_id` in
# shared/role_strategy.py. Ranks mirror the exact ladder order of the
# original if/elif chain (10 == first-checked ... 100 == last-checked);
# the mechanism iterates this tuple in rank order and returns on the first
# matched rule, else falls through to the generic brief-derived profile.
PROFILE_TRIGGER_RULES: tuple = (
    ModuleTrigger(
        module="exec_search",
        signal="target_modules:exec_search",
        profile_id="executive_search",
        rank=10,
    ),
    ModuleTrigger(
        module="designer",
        signal="target_modules:designer",
        profile_id="designer",
        rank=20,
    ),
    ModuleWithDomainTrigger(
        module="researcher",
        module_signal="target_modules:researcher",
        domain_patterns=_ACADEMIC_PATTERNS,
        domain_signal="academic_patterns",
        profile_id="academic_researcher",
        rank=30,
    ),
    MaintainerTrigger(
        module="github",
        levels=_MAINTAINER_LEVELS,
        signal="github_or_maintainer_signals",
        projects_signal="target_projects_present",
        profile_id="oss_maintainer",
        rank=40,
    ),
    TitleTrigger(
        title_patterns=_FDE_TITLE_PATTERNS,
        signal="fde_title_patterns",
        profile_id="fde_enterprise_genai",
        rank=50,
    ),
    SeniorDomainCompositeTrigger(
        domain_patterns=_BFS_DOMAIN_PATTERNS,
        domain_signal="bfs_domain",
        title_patterns=_SENIOR_AI_LEADER_PATTERNS,
        title_signal="senior_ai_leader",
        strict_signal="strict_seniority",
        profile_id="senior_bfs_ai_leader",
        rank=60,
    ),
    TitleWithLevelTrigger(
        title_patterns=_EXEC_PATTERNS,
        signal="executive_title_patterns",
        profile_id="executive_search",
        rank=70,
    ),
    TitleTrigger(
        title_patterns=_DESIGN_PATTERNS,
        signal="design_patterns",
        profile_id="designer",
        rank=80,
    ),
    TitleOrBodyDomainTrigger(
        title_patterns=_ACADEMIC_PATTERNS,
        title_signal="academic_title",
        body_patterns=_ACADEMIC_PATTERNS,
        body_min_hits=2,
        body_signal="academic_patterns_x2",
        yield_to_title_patterns=_FRONTIER_IC_PATTERNS,
        profile_id="academic_researcher",
        rank=90,
    ),
    TitleOrLevelSignalTrigger(
        title_patterns=_FRONTIER_IC_PATTERNS,
        title_signal="ic_frontier_patterns",
        engineering_terms=_ENGINEERING_SIGNAL_TERMS,
        arm_signal="ic_level_plus_engineering_signal",
        profile_id="ic_frontier_engineer",
        rank=100,
    ),
)
