"""Tests for the roster_ingest channel (W3-B1)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from github.maintainership import MaintainershipClassification
from github.query_validator import ExhaustionState
from github.rosters import RosterEntry, RosterResult
from github.schemas import GitHubProgress, GitHubSearchQuery
import github.maintainer_signal_cache as mcache

from tests.test_github_pipeline import (
    _make_candidate,
    _make_pipeline,
    _make_query,
    github_orchestrator,
)


def _roster_query(**overrides) -> GitHubSearchQuery:
    defaults = dict(
        id=1,
        name="roster seed",
        query="",
        channel="roster_ingest",
        target_repo="kubernetes/kubernetes",
    )
    defaults.update(overrides)
    return GitHubSearchQuery(**defaults)


def _codeowners_roster_result() -> RosterResult:
    return RosterResult(
        repo="kubernetes/kubernetes",
        entries=[
            RosterEntry(
                handle="k8s-owner",
                role="code_owner",
                source_file=".github/CODEOWNERS",
                repo="kubernetes/kubernetes",
            ),
        ],
        team_entries=["@kubernetes/sig-api-machinery"],
        files_found=[".github/CODEOWNERS"],
    )


class TestRosterChannelDiscoversDeclaredMaintainers:
    def test_roster_channel_discovers_declared_maintainers(self) -> None:
        pipeline = _make_pipeline()
        pipeline._client = MagicMock()
        pipeline._ensure_services = MagicMock()
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        roster_result = _codeowners_roster_result()
        query = _roster_query()

        with patch.object(
            github_orchestrator,
            "fetch_repo_roster",
            AsyncMock(return_value=roster_result),
        ):
            pre_dedup, usernames = asyncio.run(pipeline._ingest_rosters(query))

        assert pre_dedup == 1
        assert usernames == ["k8s-owner"]
        evidence = pipeline._registry_evidence_by_username["k8s-owner"]
        assert evidence["declared_roles"] == [
            {
                "hub": "governance",
                "handle": "k8s-owner",
                "package": "kubernetes/kubernetes",
                "repo": "kubernetes/kubernetes",
                "role": "code_owner",
                "corroborated_github_login": "k8s-owner",
                "source_file": ".github/CODEOWNERS",
            }
        ]
        assert evidence["packages"][0]["hub"] == "governance"
        assert evidence["packages"][0]["name"] == "kubernetes/kubernetes"
        assert pipeline.stats["roster_handles_discovered"] == 1
        assert pipeline.stats["roster_team_entries_skipped"] == 1

    def test_old_checkpoint_single_target_repo_still_ingests(self) -> None:
        pipeline = _make_pipeline()
        pipeline._client = MagicMock()
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        roster_result = _codeowners_roster_result()
        query = _roster_query(target_repo="kubernetes/kubernetes", target_packages=[])

        with patch.object(
            github_orchestrator,
            "fetch_repo_roster",
            AsyncMock(return_value=roster_result),
        ):
            pre_dedup, usernames = asyncio.run(pipeline._ingest_rosters(query))

        assert pre_dedup == 1
        assert usernames == ["k8s-owner"]


class TestRosterFailureModes:
    def test_roster_fetch_failure_fails_query_no_exhaustion(self) -> None:
        pipeline = _make_pipeline()
        pipeline._exhaustion = ExhaustionState()
        pipeline._ensure_services = MagicMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._governor.check_limits_or_raise = MagicMock()
        pipeline._governor.should_enter_enrichment_only = MagicMock(return_value=False)
        pipeline._dedup_usernames = lambda usernames: list(usernames)
        pipeline._client = MagicMock()

        query = _roster_query()
        progress = GitHubProgress(brief_name="test", queries=[query])

        client = MagicMock()
        enricher = MagicMock()
        with patch.object(
            github_orchestrator,
            "fetch_repo_roster",
            AsyncMock(side_effect=RuntimeError("contents API down")),
        ):
            asyncio.run(pipeline._execute_queries(client, enricher, progress))

        assert query.status == "failed"
        assert "repo fetch(es) failed" in query.notes
        assert "roster_ingest" not in pipeline._exhaustion.channels

    def test_roster_empty_files_record_zero_normally(self) -> None:
        pipeline = _make_pipeline()
        pipeline._exhaustion = ExhaustionState()
        pipeline._ensure_services = MagicMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._governor.check_limits_or_raise = MagicMock()
        pipeline._governor.should_enter_enrichment_only = MagicMock(return_value=False)
        pipeline._dedup_usernames = lambda usernames: list(usernames)
        pipeline._client = MagicMock()

        empty_result = RosterResult(
            repo="acme/empty",
            entries=[],
            team_entries=[],
            files_found=[],
        )
        query = _roster_query(target_repo="acme/empty")
        progress = GitHubProgress(brief_name="test", queries=[query])

        client = MagicMock()
        enricher = MagicMock()
        with patch.object(
            github_orchestrator,
            "fetch_repo_roster",
            AsyncMock(return_value=empty_result),
        ):
            asyncio.run(pipeline._execute_queries(client, enricher, progress))

        assert query.status == "done"
        channel_stats = pipeline._exhaustion.channels["roster_ingest"]
        assert channel_stats.queries_run == 1
        assert channel_stats.zero_result_streak == 1

    def test_three_rosterless_repos_do_not_exhaust_the_channel(self) -> None:
        pipeline = _make_pipeline()
        pipeline._exhaustion = ExhaustionState()
        pipeline._ensure_services = MagicMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._governor.check_limits_or_raise = MagicMock()
        pipeline._governor.should_enter_enrichment_only = MagicMock(return_value=False)
        pipeline._dedup_usernames = lambda usernames: list(usernames)
        pipeline._client = MagicMock()
        pipeline._search_users = AsyncMock(return_value=(0, []))

        empty_result = RosterResult(
            repo="acme/empty",
            entries=[],
            team_entries=[],
            files_found=[],
        )
        roster_query = _roster_query(
            target_repo="",
            target_packages=[
                "acme/empty-one",
                "acme/empty-two",
                "acme/empty-three",
            ],
        )
        user_query = _make_query(channel="user_search", query="language:rust")
        progress = GitHubProgress(brief_name="test", queries=[roster_query, user_query])

        client = MagicMock()
        enricher = MagicMock()
        with patch.object(
            github_orchestrator,
            "fetch_repo_roster",
            AsyncMock(return_value=empty_result),
        ):
            asyncio.run(pipeline._execute_queries(client, enricher, progress))

        roster_stats = pipeline._exhaustion.channels["roster_ingest"]
        assert roster_stats.queries_run == 1
        assert roster_stats.zero_result_streak == 1
        assert roster_stats.status == "active"
        assert user_query.status == "done"


class TestDeclaredRoleLeadsClassifierAtSeam:
    def test_declared_role_leads_classifier_at_seam(self) -> None:
        pipeline = _make_pipeline()
        pipeline.brief_obj.target_projects = ["kubernetes/kubernetes"]
        pipeline.brief_obj.has_v2_schema = True
        pipeline._client = MagicMock()
        pipeline._ensure_services = MagicMock()
        pipeline._work_unit_service = MagicMock()
        pipeline._execution_engine = MagicMock()
        pipeline._candidate_record = MagicMock(return_value={})
        pipeline._execution_envelope = MagicMock(return_value=MagicMock(source_cursor={}))
        pipeline._start_stage_attempt = MagicMock(return_value=1)
        pipeline._finish_failure_decision_attempt = MagicMock()
        pipeline._mark_terminal = MagicMock()
        pipeline._side_effects_service = MagicMock()
        pipeline._side_effects_service.handle_full_decision = AsyncMock(return_value=None)

        evidence = {
            "declared_roles": [
                {
                    "hub": "governance",
                    "handle": "k8s-owner",
                    "package": "kubernetes/kubernetes",
                    "repo": "kubernetes/kubernetes",
                    "role": "code_owner",
                    "corroborated_github_login": "k8s-owner",
                    "source_file": ".github/CODEOWNERS",
                }
            ],
            "packages": [
                {
                    "hub": "governance",
                    "name": "kubernetes/kubernetes",
                    "downloads_last_month": None,
                    "reverse_dependencies": None,
                    "latest_release": "",
                    "release_cadence": "",
                    "deprecated": False,
                }
            ],
        }

        candidate = _make_candidate("k8s-owner", "K8s Owner")
        candidate.registry_evidence = evidence
        query = _make_query(channel="roster_ingest", target_repo="kubernetes/kubernetes")
        facial_decision = SimpleNamespace(
            decision="FACIAL_YES",
            confidence=1.0,
            rationale="",
            candidate_name="K8s Owner",
            profile_url="https://github.com/k8s-owner",
            to_dict=lambda: {},
        )

        inferred = MaintainershipClassification(
            level="contributor",
            confidence=0.22,
            evidence_sources=["merge_authority:kubernetes/kubernetes:2PRs"],
            signals={"merge_authority": 0.5},
        )

        with patch.object(
            github_orchestrator,
            "classify_maintainership",
            AsyncMock(return_value=inferred),
        ), patch.object(
            github_orchestrator,
            "github_facial_judge_batch",
            return_value=[facial_decision],
        ), patch.object(
            github_orchestrator,
            "github_full_judge",
            return_value=SimpleNamespace(
                decision="REJECT",
                confidence=0.5,
                path="direct",
                rationale="no",
                to_dict=lambda: {},
            ),
        ):
            asyncio.run(
                pipeline._process_v2_candidates_batch(
                    [("k8s-owner", candidate)],
                    query,
                    GitHubProgress(brief_name="test"),
                )
            )

        assert candidate.maintainership is not None
        assert candidate.maintainership["role_certainty"] == "declared"
        assert candidate.maintainership["level"] == "maintainer"
        assert candidate.maintainership["corroboration"] == {
            "level": "contributor",
            "confidence": 0.22,
        }
        assert candidate.maintainership["confidence"] == 0.22
        assert candidate.maintainership["evidence_sources"][0] == (
            "declared:kubernetes/kubernetes:.github/CODEOWNERS"
        )


class TestRegistryDeclaredRoleScopedToTargetProjects:
    def test_registry_declared_role_does_not_touch_maintainership_without_target_projects(
        self,
    ) -> None:
        pipeline = _make_pipeline()
        pipeline.brief_obj.target_projects = []
        pipeline.brief_obj.has_v2_schema = True
        pipeline._client = MagicMock()
        pipeline._ensure_services = MagicMock()
        pipeline._work_unit_service = MagicMock()
        pipeline._execution_engine = MagicMock()
        pipeline._candidate_record = MagicMock(return_value={})
        pipeline._execution_envelope = MagicMock(return_value=MagicMock(source_cursor={}))
        pipeline._start_stage_attempt = MagicMock(return_value=1)
        pipeline._finish_failure_decision_attempt = MagicMock()
        pipeline._mark_terminal = MagicMock()
        pipeline._side_effects_service = MagicMock()
        pipeline._side_effects_service.handle_full_decision = AsyncMock(return_value=None)

        registry_evidence = {
            "declared_roles": [
                {
                    "hub": "crates",
                    "handle": "dtolnay",
                    "package": "serde",
                    "role": "owner",
                    "corroborated_github_login": "dtolnay",
                }
            ],
            "packages": [
                {
                    "hub": "crates",
                    "name": "serde",
                    "downloads_last_month": 500000000,
                    "reverse_dependencies": None,
                    "latest_release": "",
                    "release_cadence": "",
                    "deprecated": False,
                }
            ],
        }

        candidate = _make_candidate("dtolnay", "David Tolnay")
        candidate.registry_evidence = registry_evidence
        candidate.maintainership = None
        baseline_registry_section = candidate.to_evidence_text()

        query = _make_query(channel="registry_maintainer_discovery")
        facial_decision = SimpleNamespace(
            decision="FACIAL_YES",
            confidence=1.0,
            rationale="",
            candidate_name="David Tolnay",
            profile_url="https://github.com/dtolnay",
            to_dict=lambda: {},
        )

        captured: dict = {}

        def _fake_full_judge(evidence_text, brief=None):
            captured["text"] = evidence_text
            return SimpleNamespace(
                decision="REJECT",
                confidence=0.5,
                path="direct",
                rationale="no",
                to_dict=lambda: {},
            )

        with patch.object(
            github_orchestrator,
            "classify_maintainership",
            AsyncMock(),
        ) as classify_mock, patch.object(
            github_orchestrator,
            "github_facial_judge_batch",
            return_value=[facial_decision],
        ), patch.object(
            github_orchestrator,
            "github_full_judge",
            side_effect=_fake_full_judge,
        ):
            asyncio.run(
                pipeline._process_v2_candidates_batch(
                    [("dtolnay", candidate)],
                    query,
                    GitHubProgress(brief_name="test"),
                )
            )

        classify_mock.assert_not_awaited()
        assert candidate.maintainership is None
        assert "MAINTAINERSHIP EVIDENCE" not in captured["text"]
        assert candidate.to_evidence_text() == baseline_registry_section

    def test_declared_role_on_target_project_merges(self) -> None:
        pipeline = _make_pipeline()
        pipeline.brief_obj.target_projects = ["kubernetes/kubernetes"]
        pipeline.brief_obj.has_v2_schema = True
        pipeline._client = MagicMock()
        pipeline._ensure_services = MagicMock()
        pipeline._work_unit_service = MagicMock()
        pipeline._execution_engine = MagicMock()
        pipeline._candidate_record = MagicMock(return_value={})
        pipeline._execution_envelope = MagicMock(return_value=MagicMock(source_cursor={}))
        pipeline._start_stage_attempt = MagicMock(return_value=1)
        pipeline._finish_failure_decision_attempt = MagicMock()
        pipeline._mark_terminal = MagicMock()
        pipeline._side_effects_service = MagicMock()
        pipeline._side_effects_service.handle_full_decision = AsyncMock(return_value=None)

        registry_evidence = {
            "declared_roles": [
                {
                    "hub": "governance",
                    "handle": "k8s-owner",
                    "package": "kubernetes/kubernetes",
                    "repo": "kubernetes/kubernetes",
                    "role": "code_owner",
                    "corroborated_github_login": "k8s-owner",
                    "source_file": ".github/CODEOWNERS",
                }
            ],
            "packages": [],
        }

        candidate = _make_candidate("k8s-owner", "K8s Owner")
        candidate.registry_evidence = registry_evidence
        query = _make_query(channel="roster_ingest", target_repo="kubernetes/kubernetes")
        facial_decision = SimpleNamespace(
            decision="FACIAL_YES",
            confidence=1.0,
            rationale="",
            candidate_name="K8s Owner",
            profile_url="https://github.com/k8s-owner",
            to_dict=lambda: {},
        )

        with patch.object(
            github_orchestrator,
            "classify_maintainership",
            AsyncMock(return_value=None),
        ), patch.object(
            github_orchestrator,
            "github_facial_judge_batch",
            return_value=[facial_decision],
        ), patch.object(
            github_orchestrator,
            "github_full_judge",
            return_value=SimpleNamespace(
                decision="REJECT",
                confidence=0.5,
                path="direct",
                rationale="no",
                to_dict=lambda: {},
            ),
        ):
            asyncio.run(
                pipeline._process_v2_candidates_batch(
                    [("k8s-owner", candidate)],
                    query,
                    GitHubProgress(brief_name="test"),
                )
            )

        assert candidate.maintainership is not None
        assert candidate.maintainership["role_certainty"] == "declared"
        assert candidate.maintainership["level"] == "maintainer"
        assert candidate.maintainership.get("confidence") is None
        assert "corroboration" not in candidate.maintainership
        text = candidate.to_evidence_text()
        assert "Declared level: maintainer" in text
        assert "Classifier corroborates" not in text

        inferred = MaintainershipClassification(
            level="contributor",
            confidence=0.22,
            evidence_sources=["merge_authority:kubernetes/kubernetes:2PRs"],
            signals={"merge_authority": 0.5},
        )
        candidate2 = _make_candidate("k8s-owner-2", "K8s Owner Two")
        candidate2.registry_evidence = registry_evidence

        with patch.object(
            github_orchestrator,
            "classify_maintainership",
            AsyncMock(return_value=inferred),
        ), patch.object(
            github_orchestrator,
            "github_facial_judge_batch",
            return_value=[facial_decision],
        ), patch.object(
            github_orchestrator,
            "github_full_judge",
            return_value=SimpleNamespace(
                decision="REJECT",
                confidence=0.5,
                path="direct",
                rationale="no",
                to_dict=lambda: {},
            ),
        ):
            asyncio.run(
                pipeline._process_v2_candidates_batch(
                    [("k8s-owner-2", candidate2)],
                    query,
                    GitHubProgress(brief_name="test"),
                )
            )

        assert candidate2.maintainership["corroboration"] == {
            "level": "contributor",
            "confidence": 0.22,
        }
        assert "Classifier corroborates at contributor (0.22)" in candidate2.to_evidence_text()


class TestResidualBatchFlushesAfterFinalUsernameException:
    def test_residual_batch_flushes_after_final_username_exception(self) -> None:
        pipeline = _make_pipeline()
        pipeline.brief_obj.has_v2_schema = True
        pipeline._client = MagicMock()
        pipeline._ensure_services = MagicMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._governor.check_limits_or_raise = MagicMock()
        pipeline._governor.should_enter_enrichment_only = MagicMock(return_value=False)
        pipeline._dedup_usernames = lambda usernames: list(usernames)
        pipeline._observer.on_error = MagicMock()
        pipeline._search_users = AsyncMock(return_value=(2, ["alice", "broken"]))

        query = _make_query(channel="user_search")
        progress = GitHubProgress(brief_name="test", queries=[query])
        client = MagicMock()
        enricher = MagicMock()

        good_candidate = _make_candidate("alice", "Alice")
        batch_mock = AsyncMock()

        async def _prepare(
            _enricher,
            username,
            _query,
            _progress,
            result_rank=0,
        ):
            if username == "broken":
                raise RuntimeError("enrich failed")
            return good_candidate

        pipeline._prepare_candidate_for_evaluation = _prepare
        pipeline._process_v2_candidates_batch = batch_mock

        asyncio.run(
            pipeline._execute_single_query(
                client,
                enricher,
                query,
                progress,
            )
        )

        batch_mock.assert_awaited_once()
        flushed_batch = batch_mock.await_args.args[0]
        assert flushed_batch == [("alice", good_candidate)]
        pipeline._observer.on_error.assert_called_once()
    def test_roster_cache_kinds_registered(self) -> None:
        expected = {
            "roster_codeowners",
            "roster_maintainers",
            "roster_governance",
            "roster_recipe",
        }
        assert expected.issubset(mcache.SIGNAL_KINDS)
        for kind in expected:
            assert kind in mcache.TTL_BY_KIND
            assert mcache.TTL_BY_KIND[kind].days == 7


def test_ingest_rosters_accumulates_across_batched_repos():
    """Moved from the strategy tests: the handler half of repo batching.

    Lives here because this module imports github_orchestrator through
    tests.test_github_pipeline — importing github.orchestrator directly from
    another test module reloads github.strategy and breaks the pipeline cost
    tests (W3-B1 report; confirmed again in the PX2 gauntlet).
    """
    pipeline = _make_pipeline()
    pipeline._client = MagicMock()
    pipeline._dedup_usernames = lambda usernames: list(usernames)

    query = _roster_query(
        target_repo="",
        target_packages=[
            "kubernetes/kubernetes",
            "etcd-io/etcd",
            "prometheus/prometheus",
        ],
    )

    results_by_repo = {
        "kubernetes/kubernetes": RosterResult(
            repo="kubernetes/kubernetes",
            entries=[
                RosterEntry(
                    handle="k8s-owner",
                    role="code_owner",
                    source_file=".github/CODEOWNERS",
                    repo="kubernetes/kubernetes",
                )
            ],
            team_entries=[],
            files_found=[".github/CODEOWNERS"],
        ),
        "etcd-io/etcd": RosterResult(
            repo="etcd-io/etcd",
            entries=[
                RosterEntry(
                    handle="etcd-owner",
                    role="code_owner",
                    source_file=".github/CODEOWNERS",
                    repo="etcd-io/etcd",
                )
            ],
            team_entries=[],
            files_found=[".github/CODEOWNERS"],
        ),
        "prometheus/prometheus": RosterResult(
            repo="prometheus/prometheus",
            entries=[],
            team_entries=[],
            files_found=[],
        ),
    }

    async def _fetch(_client, owner_repo):
        return results_by_repo[owner_repo]

    with patch.object(github_orchestrator, "fetch_repo_roster", _fetch):
        pre_dedup, usernames = asyncio.run(pipeline._ingest_rosters(query))

    assert pre_dedup == 2
    assert usernames == ["k8s-owner", "etcd-owner"]
    assert "k8s-owner" in pipeline._registry_evidence_by_username
    assert "etcd-owner" in pipeline._registry_evidence_by_username
