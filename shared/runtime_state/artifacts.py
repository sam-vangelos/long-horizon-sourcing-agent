"""Artifact ownership registry for runtime-state-backed runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path


class ArtifactOwnership(str, Enum):
    PROJECTION_OWNED = "projection_owned"
    DIRECT_SIDE_EFFECT = "direct_side_effect"
    ANALYTICAL_DEBUG = "analytical_debug"


@dataclass(frozen=True)
class ArtifactContract:
    pattern: str
    ownership: ArtifactOwnership
    description: str


ARTIFACT_CONTRACTS: tuple[ArtifactContract, ...] = (
    ArtifactContract("progress.json", ArtifactOwnership.PROJECTION_OWNED, "Projected runtime progress"),
    ArtifactContract("snippets.jsonl", ArtifactOwnership.PROJECTION_OWNED, "Projected snippet records"),
    ArtifactContract("facial_judgments.jsonl", ArtifactOwnership.PROJECTION_OWNED, "Projected facial decisions"),
    ArtifactContract("profile_summaries.jsonl", ArtifactOwnership.PROJECTION_OWNED, "Projected profile summaries"),
    ArtifactContract("final_judgments.jsonl", ArtifactOwnership.PROJECTION_OWNED, "Projected full decisions"),
    ArtifactContract("candidate_history-*.jsonl", ArtifactOwnership.PROJECTION_OWNED, "Projected LinkedIn candidate history"),
    ArtifactContract("search_memory-*.json", ArtifactOwnership.PROJECTION_OWNED, "Projected LinkedIn search memory"),
    ArtifactContract("run_log.jsonl", ArtifactOwnership.DIRECT_SIDE_EFFECT, "Direct operational event log"),
    ArtifactContract("saves.jsonl", ArtifactOwnership.DIRECT_SIDE_EFFECT, "Direct save artifact"),
    ArtifactContract("outreach.jsonl", ArtifactOwnership.DIRECT_SIDE_EFFECT, "Direct outreach artifact"),
    ArtifactContract("*.csv", ArtifactOwnership.DIRECT_SIDE_EFFECT, "Direct export artifact"),
    ArtifactContract("linkedin_reconciliation.jsonl", ArtifactOwnership.DIRECT_SIDE_EFFECT, "Direct GitHub-to-LinkedIn reconciliation artifact"),
    ArtifactContract("linkedin_reconciliation.csv", ArtifactOwnership.DIRECT_SIDE_EFFECT, "Direct GitHub-to-LinkedIn reconciliation export"),
    ArtifactContract("run-report.json", ArtifactOwnership.DIRECT_SIDE_EFFECT, "Direct structured run report"),
    ArtifactContract("run-report.md", ArtifactOwnership.DIRECT_SIDE_EFFECT, "Direct markdown run report"),
    ArtifactContract("bias_monitor*.json", ArtifactOwnership.DIRECT_SIDE_EFFECT, "Direct bias checkpoint artifact"),
    ArtifactContract("execution_plan.json", ArtifactOwnership.ANALYTICAL_DEBUG, "Strategy/debug artifact"),
    ArtifactContract("kit_strings.json", ArtifactOwnership.ANALYTICAL_DEBUG, "Kit extraction/debug artifact"),
    ArtifactContract("noise_discoveries-*.jsonl", ArtifactOwnership.ANALYTICAL_DEBUG, "Auxiliary noise/debug artifact"),
    ArtifactContract("shadow_final_judgments.jsonl", ArtifactOwnership.ANALYTICAL_DEBUG, "Shadow-eval/debug artifact"),
)


def classify_artifact(path: str | Path) -> ArtifactOwnership | None:
    """Return the ownership class for a known output artifact."""

    name = Path(path).name
    for contract in ARTIFACT_CONTRACTS:
        if fnmatch(name, contract.pattern):
            return contract.ownership
    return None
