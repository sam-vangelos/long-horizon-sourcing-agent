"""Registry maintainer discovery strategy — config-independent executable tests.

Pins ``_append_registry_queries`` and :func:`form_github_strategy` registry
ordering without the optional GitHub FDE brief fixture. Probes are patched at
the ``github.health`` seam (late-bound at call time in strategy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from github.schemas import GitHubBatchReport, GitHubSearchQuery
from github.strategy import (
    _REGISTRY_PACKAGE_CAP,
    _append_registry_queries,
    _append_roster_queries,
    _append_target_project_queries,
    adapt_after_batch,
    form_github_strategy,
)


@dataclass
class _CapabilityArea:
    name: str
    github_code_signals: list[str] = field(default_factory=list)


@dataclass
class _NewBrief:
    capability_areas: list[_CapabilityArea] = field(default_factory=list)


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
        role_description: str = "",
        minimum_bar: str = "",
        archetypes: list | None = None,
        noise_archetypes: list | None = None,
        jd_text: str = "",
        intake_notes: str = "",
        search_priorities: list[str] | None = None,
        additional_search_terms: list[str] | None = None,
        instructions: list[str] | None = None,
        id: str | None = None,
    ) -> None:
        self.target_projects = list(target_projects or [])
        self.target_stacks = list(target_stacks or [])
        self.capability_areas = list(capability_areas or [])
        self._new_brief = _new_brief
        self.permanent_filters = dict(permanent_filters or {})
        self.role_title = role_title
        self.role_description = role_description
        self.minimum_bar = minimum_bar
        self.archetypes = list(archetypes or [])
        self.noise_archetypes = list(noise_archetypes or [])
        self.jd_text = jd_text
        self.intake_notes = intake_notes
        self.search_priorities = list(search_priorities or [])
        self.additional_search_terms = list(additional_search_terms or [])
        self.instructions = list(instructions or [])
        self.id = id


_STRATEGY_MOCK_PLAN = {
    "strategy_rationale": "mock github rationale",
    "user_search_queries": [
        {
            "name": "rust practitioners",
            "query": "language:rust followers:>50",
        }
    ],
    "code_search_queries": [],
    "topic_search_queries": [],
    "stargazer_repos": [],
    "seed_experts": [],
}


@pytest.fixture(autouse=True)
def _block_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed if any test reaches the network without an explicit patch."""

    def _fail_unpatched_urlopen(*_args, **_kwargs) -> None:
        pytest.fail("urllib.request.urlopen reached without patch")

    monkeypatch.setattr("urllib.request.urlopen", _fail_unpatched_urlopen)


def _crates_brief() -> _FakeBrief:
    return _FakeBrief(
        target_stacks=["rust"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Serialization",
                    github_code_signals=[
                        "use serde::Deserialize",
                        "cargo add tokio",
                    ],
                )
            ]
        ),
    )


def test_brief_with_no_relevant_stack_emits_no_registry_queries() -> None:
    queries: list[GitHubSearchQuery] = []
    brief = _FakeBrief(
        target_stacks=["go", "kubernetes"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Infra",
                    github_code_signals=["verl", "trl", "OpenRLHF"],
                )
            ]
        ),
    )

    with patch("github.health.probe_npm_registry", return_value=True), patch(
        "github.health.probe_crates_registry",
        return_value=True,
    ):
        _append_registry_queries(brief, queries, next_id=1)

    assert queries == []


def test_registry_query_carries_ecosystem_and_seeds() -> None:
    queries: list[GitHubSearchQuery] = []

    with patch("github.health.probe_crates_registry", return_value=True):
        _append_registry_queries(_crates_brief(), queries, next_id=1)

    assert len(queries) == 1
    query = queries[0]
    assert query.channel == "registry_maintainer_discovery"
    assert query.query == ""
    assert query.target_ecosystem == "crates.io"
    assert query.target_packages == ["serde", "tokio"]
    assert "serde" in query.name
    assert "tokio" in query.name


def test_probe_failure_drops_hub_queries_fail_closed(capsys) -> None:
    brief = _FakeBrief(
        target_stacks=["npm", "rust"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Mixed",
                    github_code_signals=[
                        "require('lodash')",
                        "use tokio::spawn",
                    ],
                )
            ]
        ),
    )
    queries: list[GitHubSearchQuery] = []

    with patch("github.health.probe_npm_registry", return_value=False), patch(
        "github.health.probe_crates_registry",
        return_value=True,
    ):
        _append_registry_queries(brief, queries, next_id=1)

    assert len(queries) == 1
    assert queries[0].target_ecosystem == "crates.io"
    err = capsys.readouterr().err
    assert "registry hub unreachable" in err
    assert "npmjs.org" in err


def test_registry_empty_seed_stack_only_logs_and_skips(capsys) -> None:
    queries: list[GitHubSearchQuery] = []
    brief = _FakeBrief(target_stacks=["rust"], _new_brief=_NewBrief())

    with patch("github.health.probe_crates_registry", return_value=True):
        _append_registry_queries(brief, queries, next_id=1)

    assert queries == []
    err = capsys.readouterr().err
    assert "no seed packages derivable from github_code_signals" in err


def test_registry_seed_packages_capped(capsys) -> None:
    signals = [f"require('pkg{i}')" for i in range(_REGISTRY_PACKAGE_CAP + 5)]
    brief = _FakeBrief(
        target_stacks=["npm"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(name="NPM", github_code_signals=signals)
            ]
        ),
    )
    queries: list[GitHubSearchQuery] = []

    with patch("github.health.probe_npm_registry", return_value=True):
        _append_registry_queries(brief, queries, next_id=1)

    assert len(queries) == 1
    assert len(queries[0].target_packages) == _REGISTRY_PACKAGE_CAP
    err = capsys.readouterr().err
    assert f"capped npmjs.org seed packages at {_REGISTRY_PACKAGE_CAP}" in err


def test_adaptation_does_not_emit_registry_queries() -> None:
    brief = _FakeBrief(role_title="Platform Engineer")
    batch_report = GitHubBatchReport(batch_name="batch-1")
    remaining = [
        GitHubSearchQuery(
            id=1,
            name="queued registry",
            query="",
            channel="registry_maintainer_discovery",
            target_ecosystem="crates.io",
            target_packages=["tokio"],
        )
    ]
    adaptation_payload = {
        "new_user_queries": [{"query": "language:rust", "name": "rust users"}],
        "new_code_queries": [],
        "new_topic_queries": [],
        "new_stargazer_repos": [],
        "new_repos_to_mine": [],
        "skip_query_ids": [],
        "rationale": "broaden rust pool",
    }

    with patch("github.strategy.opus_llm_cached", return_value=adaptation_payload):
        new_queries, _rationale, _skipped = adapt_after_batch(
            brief,
            batch_report,
            remaining,
        )

    assert all(q.channel != "registry_maintainer_discovery" for q in new_queries)


def test_target_projects_no_longer_queue_stargazer_queries() -> None:
    queries: list[GitHubSearchQuery] = []
    _append_target_project_queries(
        _FakeBrief(target_projects=["kubernetes/kubernetes", "etcd-io/etcd"]),
        queries,
        next_id=1,
    )

    channels = {q.channel for q in queries}
    assert channels == {"repo_mining"}
    assert "stargazer_mining" not in channels
    assert len(queries) == 2


def test_registry_queries_run_first() -> None:
    brief = _FakeBrief(
        target_stacks=["rust"],
        target_projects=["kubernetes/kubernetes"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Serialization",
                    github_code_signals=["use serde::Deserialize"],
                )
            ]
        ),
    )

    with patch("github.strategy.opus_llm_cached", return_value=_STRATEGY_MOCK_PLAN), patch(
        "github.health.probe_crates_registry",
        return_value=True,
    ):
        queries, _rationale = form_github_strategy(brief)

    assert queries, "expected strategy queries"
    assert queries[0].channel == "registry_maintainer_discovery"
    assert queries[0].target_ecosystem == "crates.io"
    channels = [q.channel for q in queries]
    registry_idx = channels.index("registry_maintainer_discovery")
    user_idx = channels.index("user_search")
    repo_idx = channels.index("repo_mining")
    assert registry_idx < user_idx < repo_idx


def test_target_projects_produce_roster_queries() -> None:
    queries: list[GitHubSearchQuery] = []
    _append_roster_queries(
        _FakeBrief(target_projects=["kubernetes/kubernetes", "etcd-io/etcd"]),
        queries,
        next_id=1,
    )

    assert len(queries) == 1
    query = queries[0]
    assert query.channel == "roster_ingest"
    assert query.query == ""
    assert query.target_packages == ["kubernetes/kubernetes", "etcd-io/etcd"]


def test_roster_repos_batch_into_single_query() -> None:
    queries: list[GitHubSearchQuery] = []
    _append_roster_queries(
        _FakeBrief(
            target_projects=[
                "kubernetes/kubernetes",
                "etcd-io/etcd",
                "prometheus/prometheus",
            ]
        ),
        queries,
        next_id=1,
    )

    assert len(queries) == 1
    query = queries[0]
    assert query.channel == "roster_ingest"
    assert query.target_packages == [
        "kubernetes/kubernetes",
        "etcd-io/etcd",
        "prometheus/prometheus",
    ]


def test_feedstock_repos_produce_roster_queries() -> None:
    brief = _FakeBrief(
        target_stacks=["python"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Scientific Python",
                    github_code_signals=["import numpy as np"],
                )
            ]
        ),
    )
    queries: list[GitHubSearchQuery] = []

    _append_roster_queries(brief, queries, next_id=1)

    assert len(queries) == 1
    query = queries[0]
    assert query.channel == "roster_ingest"
    assert query.query == ""
    assert query.target_packages == ["conda-forge/numpy-feedstock"]


def test_feedstocks_gated_on_python_stack() -> None:
    signals = _NewBrief(
        capability_areas=[
            _CapabilityArea(
                name="Scientific Python",
                github_code_signals=["import numpy as np"],
            )
        ]
    )

    python_queries: list[GitHubSearchQuery] = []
    _append_roster_queries(
        _FakeBrief(target_stacks=["python"], _new_brief=signals),
        python_queries,
        next_id=1,
    )
    assert len(python_queries) == 1
    assert python_queries[0].target_packages == ["conda-forge/numpy-feedstock"]

    rust_queries: list[GitHubSearchQuery] = []
    _append_roster_queries(
        _FakeBrief(target_stacks=["rust"], _new_brief=signals),
        rust_queries,
        next_id=1,
    )
    assert rust_queries == []


def test_no_roster_queries_without_sources() -> None:
    queries: list[GitHubSearchQuery] = []
    brief = _FakeBrief(
        target_stacks=["npm"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Frontend",
                    github_code_signals=["require('react')"],
                )
            ]
        ),
    )

    _append_roster_queries(brief, queries, next_id=1)

    assert queries == []


def test_declared_channels_lead_in_order() -> None:
    brief = _FakeBrief(
        target_stacks=["rust"],
        target_projects=["kubernetes/kubernetes"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Scientific Python",
                    github_code_signals=[
                        "use serde::Deserialize",
                        "import numpy as np",
                    ],
                )
            ]
        ),
    )

    with patch("github.strategy.opus_llm_cached", return_value=_STRATEGY_MOCK_PLAN), patch(
        "github.health.probe_crates_registry",
        return_value=True,
    ):
        queries, _rationale = form_github_strategy(brief)

    assert queries, "expected strategy queries"
    channels = [q.channel for q in queries]
    registry_idx = channels.index("registry_maintainer_discovery")
    roster_idx = channels.index("roster_ingest")
    user_idx = channels.index("user_search")
    repo_idx = channels.index("repo_mining")
    assert registry_idx < roster_idx < user_idx < repo_idx

    ids = [q.id for q in queries]
    assert ids == list(range(1, len(queries) + 1))
    assert len(ids) == len(set(ids))


def test_adaptation_creates_no_roster_queries() -> None:
    brief = _FakeBrief(role_title="Platform Engineer")
    batch_report = GitHubBatchReport(batch_name="batch-1")
    remaining = [
        GitHubSearchQuery(
            id=1,
            name="queued roster",
            query="",
            channel="roster_ingest",
            target_repo="kubernetes/kubernetes",
        )
    ]
    adaptation_payload = {
        "new_user_queries": [{"query": "language:rust", "name": "rust users"}],
        "new_code_queries": [],
        "new_topic_queries": [],
        "new_stargazer_repos": [],
        "new_repos_to_mine": ["owner/repo"],
        "skip_query_ids": [],
        "rationale": "broaden rust pool",
    }

    with patch("github.strategy.opus_llm_cached", return_value=adaptation_payload):
        new_queries, _rationale, _skipped = adapt_after_batch(
            brief,
            batch_report,
            remaining,
        )

    assert all(q.channel != "roster_ingest" for q in new_queries)
