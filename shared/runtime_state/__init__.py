"""Canonical runtime state store and compatibility projections."""

from .admin import (
    clear_candidate_terminal_state,
    inspect_candidate_side_effects,
    inspect_orphaned_attempts,
    replay_candidate_side_effect,
    rebuild_compat_projections,
    requeue_work_unit,
)
from .artifacts import ARTIFACT_CONTRACTS, ArtifactContract, ArtifactOwnership, classify_artifact
from .github import GitHubRuntimeStateBridge
from .interfaces import RuntimeStateBridge
from .lock import RuntimeStateLock
from .projections import (
    project_github_facial_judgments,
    project_github_final_judgments,
    project_github_profile_summaries,
    project_github_progress,
    project_github_snippets,
    project_linkedin_candidate_history,
    project_linkedin_facial_judgments,
    project_linkedin_final_judgments,
    project_linkedin_profile_summaries,
    project_linkedin_progress,
    project_linkedin_search_memory,
    project_linkedin_snippets,
)
from .linkedin import LinkedInRuntimeStateBridge
from .store import (
    CURRENT_SCHEMA_VERSION,
    GITHUB_GRAPH_SEED_KIND,
    GITHUB_QUERY_KIND,
    LINKEDIN_STRING_KIND,
    RuntimeStateStore,
)
from .event_log import install_runtime_event_log

install_runtime_event_log(
    RuntimeStateStore,
    runtime_state_schema_version=CURRENT_SCHEMA_VERSION,
)

__all__ = [
    "GITHUB_GRAPH_SEED_KIND",
    "GITHUB_QUERY_KIND",
    "LINKEDIN_STRING_KIND",
    "RuntimeStateLock",
    "RuntimeStateStore",
    "GitHubRuntimeStateBridge",
    "LinkedInRuntimeStateBridge",
    "RuntimeStateBridge",
    "ArtifactContract",
    "ArtifactOwnership",
    "ARTIFACT_CONTRACTS",
    "classify_artifact",
    "project_github_snippets",
    "project_github_facial_judgments",
    "project_github_profile_summaries",
    "project_github_final_judgments",
    "clear_candidate_terminal_state",
    "inspect_candidate_side_effects",
    "inspect_orphaned_attempts",
    "project_github_progress",
    "project_linkedin_candidate_history",
    "project_linkedin_facial_judgments",
    "project_linkedin_final_judgments",
    "project_linkedin_profile_summaries",
    "project_linkedin_progress",
    "project_linkedin_search_memory",
    "project_linkedin_snippets",
    "replay_candidate_side_effect",
    "rebuild_compat_projections",
    "requeue_work_unit",
]
