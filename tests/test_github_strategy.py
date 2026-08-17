"""Tests for OSS Maintainers Slice 7 — target-project strategy seeding.

Pins the contract for ``_append_target_project_queries``:

- Empty ``brief.target_projects`` ⇒ no queries appended (byte-
  identical strategy for classic github briefs).
- Each entry seeds one ``repo_mining`` query with the project as
  ``target_repo``. Stargazer mining is disabled fail-closed.
- Dedups against existing ``target_repo`` entries (so LLM-emitted +
  default repo seeding can't collide with target-project seeds).
- Returns the post-append ``next_id`` so the caller can continue
  numbering.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from shared.brief_loader import load_brief
from shared.schemas import ExecutionPlan

from github.schemas import GitHubSearchQuery
from github.strategy import (
    _append_target_project_queries,
    form_github_strategy,
    form_strategy_for_registry,
)


GITHUB_BRIEF_PATH = str(
    Path(__file__).parent.parent
    / "config"
    / "Forward-Deployed-Engineer-NYC"
    / "brief-forward-deployed-engineer-us-github-v1.json"
)

if not Path(GITHUB_BRIEF_PATH).is_file():
    pytest.skip(
        "Optional GitHub FDE brief JSON not found under config/",
        allow_module_level=True,
    )

_REGISTRY_ADAPTER_MOCK_PLAN = {
    "strategy_rationale": "mock github rationale",
    "user_search_queries": [
        {
            "name": "agentic builders",
            "query": 'language:python "langgraph" followers:>50',
        }
    ],
    "code_search_queries": [],
    "topic_search_queries": [],
    "stargazer_repos": [],
    "seed_experts": [],
}


class _FakeBrief:
    """Minimal Brief-shaped object for strategy helpers."""

    def __init__(
        self,
        *,
        target_projects: list[str] | None = None,
        target_stacks: list[str] | None = None,
        capability_areas: list | None = None,
        _new_brief: object | None = None,
        permanent_filters: dict | None = None,
        role_title: str = "",
        id: str | None = None,
    ) -> None:
        self.target_projects = list(target_projects or [])
        self.target_stacks = list(target_stacks or [])
        self.capability_areas = list(capability_areas or [])
        self._new_brief = _new_brief
        self.permanent_filters = dict(permanent_filters or {})
        self.role_title = role_title
        self.id = id


def test_no_target_projects_appends_nothing() -> None:
    """Behavior-preserving: classic github briefs are byte-identical."""

    queries: list[GitHubSearchQuery] = []
    new_id = _append_target_project_queries(_FakeBrief(), queries, next_id=10)
    assert queries == []
    assert new_id == 10


def test_seeds_one_repo_mining_query_per_target_project() -> None:
    queries: list[GitHubSearchQuery] = []
    new_id = _append_target_project_queries(
        _FakeBrief(target_projects=["kubernetes/kubernetes"]),
        queries,
        next_id=1,
    )

    assert len(queries) == 1
    assert queries[0].channel == "repo_mining"
    assert queries[0].target_repo == "kubernetes/kubernetes"
    assert {q.id for q in queries} == {1}
    assert new_id == 2


def test_seeds_per_target_in_order() -> None:
    queries: list[GitHubSearchQuery] = []
    new_id = _append_target_project_queries(
        _FakeBrief(
            target_projects=["kubernetes/kubernetes", "etcd-io/etcd"]
        ),
        queries,
        next_id=1,
    )

    assert len(queries) == 2
    assert queries[0].target_repo == "kubernetes/kubernetes"
    assert queries[0].channel == "repo_mining"
    assert queries[1].target_repo == "etcd-io/etcd"
    assert queries[1].channel == "repo_mining"
    assert new_id == 3


def test_stargazer_drop_logs_loudly(capsys) -> None:
    queries: list[GitHubSearchQuery] = []
    _append_target_project_queries(
        _FakeBrief(target_projects=["kubernetes/kubernetes"]),
        queries,
        next_id=1,
    )

    err = capsys.readouterr().err
    assert "stargazer lane unavailable" in err
    assert "kubernetes/kubernetes" in err


def test_dedups_against_existing_target_repos() -> None:
    """If LLM strategy or default seeding already named the repo, skip it."""

    existing = [
        GitHubSearchQuery(
            id=99,
            name="LLM-emitted: Mine contributors of kubernetes/kubernetes",
            query="",
            channel="repo_mining",
            target_repo="kubernetes/kubernetes",
        )
    ]
    new_id = _append_target_project_queries(
        _FakeBrief(target_projects=["kubernetes/kubernetes"]),
        existing,
        next_id=100,
    )

    # No new queries appended (target already in the set).
    assert len(existing) == 1
    assert new_id == 100


def test_dedups_case_insensitively() -> None:
    """GitHub treats owner/repo case-insensitively; so does our dedup."""

    existing = [
        GitHubSearchQuery(
            id=1,
            name="Existing",
            query="",
            channel="repo_mining",
            target_repo="Kubernetes/Kubernetes",
        )
    ]
    _append_target_project_queries(
        _FakeBrief(target_projects=["kubernetes/kubernetes"]),
        existing,
        next_id=2,
    )

    # Still just the one existing entry.
    assert len(existing) == 1


def test_skips_blank_target_entries() -> None:
    queries: list[GitHubSearchQuery] = []
    _append_target_project_queries(
        _FakeBrief(target_projects=["", "  ", "kubernetes/kubernetes"]),
        queries,
        next_id=1,
    )
    assert len(queries) == 1
    assert queries[0].target_repo == "kubernetes/kubernetes"


def test_skips_non_string_entries() -> None:
    queries: list[GitHubSearchQuery] = []
    _append_target_project_queries(
        _FakeBrief(target_projects=["kubernetes/kubernetes", 42, None]),  # type: ignore[list-item]
        queries,
        next_id=1,
    )
    assert len(queries) == 1
    assert queries[0].target_repo == "kubernetes/kubernetes"


# ---------------------------------------------------------------------------
# Multi-Agent Execution Plan Slice 1.6 — registry adapter
# ---------------------------------------------------------------------------


def test_form_strategy_for_registry_returns_execution_plan() -> None:
    """The registry adapter normalizes GitHub's native
    ``(list[GitHubSearchQuery], str)`` tuple shape into the
    :class:`shared.schemas.ExecutionPlan` shape consumed by
    ``cloris.launchers.LauncherEntry.form_strategy_fn``."""

    brief = load_brief(GITHUB_BRIEF_PATH)

    with patch(
        "github.strategy.opus_llm_cached",
        return_value=_REGISTRY_ADAPTER_MOCK_PLAN,
    ):
        plan = form_strategy_for_registry(brief)

    assert isinstance(plan, ExecutionPlan)
    assert plan.strategy_rationale == "mock github rationale"
    # Without vertical default injections, the mocked LLM response is the
    # sole query source unless the brief names target projects.
    assert plan.generated_strings, "registry adapter must populate generated_strings"
    channels = {q.get("channel") for q in plan.generated_strings}
    assert channels == {"user_search"}


def test_form_strategy_for_registry_matches_native_call_shape() -> None:
    """Registry adapter output mirrors a native
    :func:`form_github_strategy` call: each :class:`GitHubSearchQuery`
    serialized via :meth:`to_dict` lands in ``generated_strings``;
    the rationale lands in ``strategy_rationale``. No information is
    dropped or fabricated by the wrapping."""

    brief = load_brief(GITHUB_BRIEF_PATH)

    with patch(
        "github.strategy.opus_llm_cached",
        return_value=_REGISTRY_ADAPTER_MOCK_PLAN,
    ):
        native_queries, native_rationale = form_github_strategy(brief)
        adapter_plan = form_strategy_for_registry(brief)

    assert adapter_plan.strategy_rationale == native_rationale
    assert adapter_plan.generated_strings == [
        q.to_dict() for q in native_queries
    ]


def test_form_strategy_for_registry_does_not_merge_linkedin_lane_templates() -> None:
    brief = load_brief(GITHUB_BRIEF_PATH)

    with patch(
        "github.strategy.opus_llm_cached",
        return_value=_REGISTRY_ADAPTER_MOCK_PLAN,
    ):
        plan = form_strategy_for_registry(brief)

    assert plan.role_strategy_profile
    assert plan.sourcing_lanes == []
    assert plan.search_hypotheses == []
    assert plan.search_slices == []
