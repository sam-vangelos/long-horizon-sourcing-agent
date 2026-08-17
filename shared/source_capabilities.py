"""Canonical source-capability manifest for Cloris.

This is product truth, not documentation copy. Prompt builders consume this
module so source strategy stays consistent across intake and chief-of-staff
dispatch without hardcoding a second matrix inside prompt strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SourceStrategyRole = Literal[
    "primary",
    "secondary",
    "corroborating",
    "investigation_first",
]


@dataclass(frozen=True)
class SourceCapability:
    key: str
    display_name: str
    recruiter_name: str
    capability: str
    evidence_boundaries: tuple[str, ...]
    corroboration_rules: tuple[str, ...]
    deployment_bias: str


SOURCE_CAPABILITIES: tuple[SourceCapability, ...] = (
    SourceCapability(
        key="linkedin",
        display_name="LinkedIn",
        recruiter_name="LinkedIn",
        capability=(
            "Maps role history, titles, employer patterns, geography, seniority, "
            "and reachable talent pools."
        ),
        evidence_boundaries=(
            "Cannot prove artifact quality, code ownership, publication depth, "
            "portfolio taste, or confidential executive movement on its own.",
        ),
        corroboration_rules=(
            "Pair with GitHub when technical artifacts matter.",
            "Pair with Researcher when publication or domain research proof matters.",
            "Pair with Designer when visual craft is the hiring bar.",
            "Pair with Exec Search when market mapping or confidential movement matters.",
        ),
        deployment_bias=(
            "Best default primary source for most commercial hiring briefs because "
            "it gives broad people coverage and recruiter actionability."
        ),
    ),
    SourceCapability(
        key="github",
        display_name="GitHub / open source",
        recruiter_name="GitHub and open source",
        capability=(
            "Reads public code, maintainership, project quality, repository context, "
            "and open-source collaboration patterns."
        ),
        evidence_boundaries=(
            "Cannot prove current employment fit, compensation posture, private-work "
            "performance, or candidates who do strong work outside public repositories.",
        ),
        corroboration_rules=(
            "Use as corroborating evidence for engineering roles where public work "
            "can confirm depth.",
            "Pair with LinkedIn to connect artifact strength to role trajectory and "
            "current actionability.",
        ),
        deployment_bias=(
            "Use when the role rewards builders with inspectable technical artifacts, "
            "maintainership, or ecosystem reputation."
        ),
    ),
    SourceCapability(
        key="researcher",
        display_name="Researcher",
        recruiter_name="Researcher",
        capability=(
            "Finds academic and industrial researchers through publications, topics, "
            "conference venues, and research impact signals."
        ),
        evidence_boundaries=(
            "Cannot prove commercial operating scope, people-management range, or "
            "recruiter contactability on its own.",
        ),
        corroboration_rules=(
            "Pair with LinkedIn when the person must also be currently reachable or "
            "commercially legible.",
            "Pair with GitHub when the research must translate into working systems.",
        ),
        deployment_bias=(
            "Use when the role depends on research depth, publication track record, "
            "scientific novelty, or technical fields where papers are meaningful proof."
        ),
    ),
    SourceCapability(
        key="designer",
        display_name="Designer",
        recruiter_name="Designer",
        capability=(
            "Evaluates portfolios, visual judgment, taste calibration, medium fit, "
            "and evidence of shipped creative work."
        ),
        evidence_boundaries=(
            "Cannot prove organizational scope, current availability, or non-portfolio "
            "operating performance on its own.",
        ),
        corroboration_rules=(
            "Pair with LinkedIn when seniority, employer context, or reachability "
            "matters.",
            "Pair with Exec Search when the brief is a confidential design leadership "
            "market map.",
        ),
        deployment_bias=(
            "Use when portfolio quality or visual judgment is a gating part of the hire."
        ),
    ),
    SourceCapability(
        key="exec_search",
        display_name="Exec Search",
        recruiter_name="Exec Search",
        capability=(
            "Builds market maps and dossiers for senior, confidential, or low-volume "
            "leadership searches."
        ),
        evidence_boundaries=(
            "Cannot produce high-volume sourcing by design, and dossiers need companion "
            "evidence before outreach decisions feel complete.",
        ),
        corroboration_rules=(
            "Pair with LinkedIn for profile coverage and recruiter workflow.",
            "Pair with Researcher, GitHub, or Designer when the executive read depends "
            "on domain-specific proof.",
        ),
        deployment_bias=(
            "Use as investigation-first when the recruiter needs market shape, target "
            "company mapping, confidentiality, or senior leadership dossiers."
        ),
    ),
)


def source_capability_manifest() -> tuple[SourceCapability, ...]:
    """Return the immutable capability manifest."""

    return SOURCE_CAPABILITIES


def source_capability_keys() -> tuple[str, ...]:
    """Return source keys in manifest order."""

    return tuple(record.key for record in SOURCE_CAPABILITIES)


def source_capability_prompt_block() -> str:
    """Render the manifest as a compact prompt block."""

    sections: list[str] = []
    for source in SOURCE_CAPABILITIES:
        boundaries = "; ".join(source.evidence_boundaries)
        corroboration = "; ".join(source.corroboration_rules)
        sections.append(
            "\n".join(
                [
                    f"- {source.display_name} (`{source.key}`)",
                    f"  Capability: {source.capability}",
                    f"  Evidence boundaries: {boundaries}",
                    f"  Corroboration rules: {corroboration}",
                    f"  Deployment bias: {source.deployment_bias}",
                ]
            )
        )
    return "\n".join(sections)


def display_name_for_source(source_key: str) -> str:
    """Humanize a source key for recruiter-facing strings."""

    for source in SOURCE_CAPABILITIES:
        if source.key == source_key:
            return source.display_name
    return " ".join(part.capitalize() for part in source_key.split("_") if part)


def recommend_source_strategy_from_text(text: str) -> list[dict[str, str]]:
    """Return a conservative source strategy from role/source text.

    This is a deterministic fallback for prompts and CTA-time recovery.
    LLM paths may produce richer recommendations, but the fallback keeps
    Cloris from collapsing back to a LinkedIn-only mental model when no
    provider is available.
    """

    lower = (text or "").lower()
    strategy: list[dict[str, str]] = [
        {
            "source": "linkedin",
            "role": "primary",
            "rationale": (
                "Broad people coverage, current roles, geography, and recruiter "
                "actionability make it the safest starting point."
            ),
        }
    ]

    def add(source: str, role: SourceStrategyRole, rationale: str) -> None:
        if any(item["source"] == source for item in strategy):
            return
        strategy.append({"source": source, "role": role, "rationale": rationale})

    technical_markers = (
        "engineer",
        "engineering",
        "developer",
        "platform",
        "infrastructure",
        "open source",
        "github",
        "code",
        "repository",
        "maintainer",
        "ml",
        "ai",
        "llm",
    )
    research_markers = (
        "research",
        "scientist",
        "publication",
        "paper",
        "conference",
        "phd",
        "applied ai",
        "ai lab",
        "machine learning",  # VERTICAL-VOCAB(source-routing-markers)
    )
    designer_markers = (
        "designer",
        "design",
        "portfolio",
        "visual",
        "brand",
        "product design",
        "motion",
        "ux",
    )
    executive_markers = (
        "executive",
        "vp ",
        "vice president",
        "chief",
        "cxo",
        "confidential",
        "market map",
        "dossier",
        "head of",
        "gm ",
    )

    if any(marker in lower for marker in technical_markers):
        add(
            "github",
            "corroborating",
            "Public technical work can confirm whether the builder signal is real.",
        )
    if any(marker in lower for marker in research_markers):
        add(
            "researcher",
            "corroborating",
            "Publication and topic evidence can confirm research depth.",
        )
    if any(marker in lower for marker in designer_markers):
        # For design-heavy briefs, Designer should sit ahead of LinkedIn
        # because portfolio judgment is the gating proof.
        strategy = [
            {
                "source": "designer",
                "role": "primary",
                "rationale": (
                    "Portfolio quality and visual judgment need direct review."
                ),
            },
            {
                "source": "linkedin",
                "role": "secondary",
                "rationale": (
                    "Role history and reachability complete the portfolio read."
                ),
            },
        ] + [item for item in strategy if item["source"] not in {"designer", "linkedin"}]
    if any(marker in lower for marker in executive_markers):
        add(
            "exec_search",
            "investigation_first",
            "Market mapping and dossiers help when the brief is senior, confidential, or sparse.",
        )

    return strategy


def target_modules_from_strategy(strategy: list[dict[str, str]]) -> list[str]:
    """Return stable source keys from a strategy list."""

    known = set(source_capability_keys())
    out: list[str] = []
    for item in strategy:
        source = item.get("source") if isinstance(item, dict) else None
        if isinstance(source, str) and source in known and source not in out:
            out.append(source)
    return out or ["linkedin"]


__all__ = [
    "SOURCE_CAPABILITIES",
    "SourceCapability",
    "SourceStrategyRole",
    "display_name_for_source",
    "recommend_source_strategy_from_text",
    "source_capability_keys",
    "source_capability_manifest",
    "source_capability_prompt_block",
    "target_modules_from_strategy",
]
