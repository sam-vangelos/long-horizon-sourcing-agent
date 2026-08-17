"""Tests for the registry_maintainer_discovery channel (W2-B1)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from github.query_validator import ExhaustionState
from github.schemas import GitHubProgress, GitHubRepo, GitHubSearchQuery
from shared.runtime_state.github import PersonKeySet
from shared.storage import append_jsonl

from tests.test_github_pipeline import (
    _make_candidate,
    _make_pipeline,
    _make_query,
    github_orchestrator,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text())


def _registry_query(**overrides) -> GitHubSearchQuery:
    defaults = dict(
        id=1,
        name="registry seed",
        query="",
        channel="registry_maintainer_discovery",
        target_ecosystem="crates.io",
        target_packages=["serde"],
    )
    defaults.update(overrides)
    return GitHubSearchQuery(**defaults)


class TestUnregisteredChannelFailsLoudly:
    def test_unregistered_channel_fails_loudly(self, capsys) -> None:
        pipeline = _make_pipeline()
        pipeline._exhaustion = ExhaustionState()
        pipeline._ensure_services = MagicMock()

        query = GitHubSearchQuery(
            id=99,
            name="bad channel",
            query="",
            channel="nonsense",
        )
        progress = GitHubProgress(brief_name="test")
        client = MagicMock()
        enricher = MagicMock()

        asyncio.run(
            pipeline._execute_single_query(client, enricher, query, progress)
        )

        assert query.status == "failed"
        assert query.notes == "unknown channel: nonsense"
        assert "nonsense" in capsys.readouterr().err

    def test_unknown_channel_does_not_record_exhaustion_via_execute_queries(self) -> None:
        """Production _execute_queries must not teach exhaustion for bogus channels.

        Stub _execute_single_query to simulate a misconfigured dispatch arm that
        completes successfully without setting status=failed — the KNOWN_CHANNELS
        guard is what prevents a ChannelStats entry. Verified by temporarily
        commenting out the guard in orchestrator.py: this assertion fails.
        """
        pipeline = _make_pipeline()
        pipeline._exhaustion = ExhaustionState()
        pipeline._ensure_services = MagicMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._governor.check_limits_or_raise = MagicMock()
        pipeline._governor.should_enter_enrichment_only = MagicMock(return_value=False)

        query = GitHubSearchQuery(
            id=99,
            name="bad channel",
            query="",
            channel="nonsense",
        )
        progress = GitHubProgress(brief_name="test", queries=[query])

        async def _fake_execute(_client, _enricher, active_query, _progress):
            active_query.result_count = 0
            active_query.candidates_discovered = 0
            active_query._pre_dedup_count = 0

        pipeline._execute_single_query = _fake_execute

        client = MagicMock()
        enricher = MagicMock()
        asyncio.run(pipeline._execute_queries(client, enricher, progress))

        assert query.status == "done"
        assert "nonsense" not in pipeline._exhaustion.channels


class TestCratesOwnerDiscovery:
    def test_crates_owner_discovers_declared_maintainer(self) -> None:
        pipeline = _make_pipeline()
        pipeline._ensure_services = MagicMock()
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        owners = [
            {
                "login": "dtolnay",
                "kind": "user",
                "github_login": "dtolnay",
            }
        ]
        crate_info = {
            "name": "serde",
            "downloads": 500000000,
            "recent_versions": [{"version": "1.0.210", "created_at": "2026-05-01T12:00:00Z"}],
            "repository_url": "https://github.com/serde-rs/serde",
        }

        hub = AsyncMock()
        hub.get_owner_users = AsyncMock(return_value=owners)
        hub.get_crate = AsyncMock(return_value=crate_info)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(target_ecosystem="crates.io", target_packages=["serde"])

        with patch.object(github_orchestrator, "CratesHubClient", return_value=hub):
            pre_dedup, usernames = asyncio.run(
                pipeline._discover_registry_maintainers(query)
            )

        assert pre_dedup == 1
        assert usernames == ["dtolnay"]
        evidence = pipeline._registry_evidence_by_username["dtolnay"]
        assert evidence["declared_roles"] == [
            {
                "hub": "crates",
                "handle": "dtolnay",
                "package": "serde",
                "role": "owner",
                "corroborated_github_login": "dtolnay",
            }
        ]
        assert evidence["packages"][0]["hub"] == "crates"
        assert evidence["packages"][0]["name"] == "serde"
        assert evidence["packages"][0]["downloads_last_month"] == 500000000


class TestNpmUncorroboratedHandle:
    def test_npm_uncorroborated_handle_is_not_emitted(self) -> None:
        pipeline = _make_pipeline()
        pipeline._ensure_services = MagicMock()
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        packument = {
            "name": "chalk",
            "maintainer_handles": ["sindresorhus", "wronghandle"],
            "repository_url": "https://github.com/sindresorhus/chalk",
            "latest_version": "5.0.0",
            "deprecated": False,
        }

        hub = AsyncMock()
        hub.get_packument = AsyncMock(return_value=packument)
        hub.get_downloads_last_month = AsyncMock(return_value=658979149)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="npmjs.org",
            target_packages=["chalk"],
        )

        with patch.object(github_orchestrator, "NpmHubClient", return_value=hub):
            pre_dedup, usernames = asyncio.run(
                pipeline._discover_registry_maintainers(query)
            )

        assert pre_dedup == 1
        assert usernames == ["sindresorhus"]
        assert pipeline.stats["registry_unresolved_maintainers"] == 1
        assert "wronghandle" not in pipeline._registry_evidence_by_username


class TestNpmOrgOwnedRepoCorroboration:
    def test_npm_org_owned_repo_corroborates_via_contributors(self) -> None:
        pipeline = _make_pipeline()
        pipeline._ensure_services = MagicMock()
        pipeline._dedup_usernames = lambda usernames: list(usernames)
        pipeline._client = MagicMock()
        pipeline._client.get_repo_contributors = AsyncMock(
            return_value=[{"login": "jdalton"}],
        )

        packument = {
            "name": "lodash",
            "maintainer_handles": ["jdalton"],
            "repository_url": "https://github.com/lodash/lodash",
            "latest_version": "4.17.21",
            "deprecated": False,
        }

        hub = AsyncMock()
        hub.get_packument = AsyncMock(return_value=packument)
        hub.get_downloads_last_month = AsyncMock(return_value=50000000)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="npmjs.org",
            target_packages=["lodash"],
        )

        with patch.object(github_orchestrator, "NpmHubClient", return_value=hub):
            pre_dedup, usernames = asyncio.run(
                pipeline._discover_registry_maintainers(query)
            )

        assert pre_dedup == 1
        assert usernames == ["jdalton"]
        pipeline._client.get_repo_contributors.assert_awaited_once_with("lodash/lodash")
        evidence = pipeline._registry_evidence_by_username["jdalton"]
        assert evidence["declared_roles"][0]["corroborated_github_login"] == "jdalton"


class TestRegistryEvidenceSidecar:
    def test_already_terminal_maintainer_evidence_lands_in_sidecar(self, tmp_path) -> None:
        pipeline = _make_pipeline()
        pipeline.output_dir = tmp_path
        pipeline._runtime_run_id = 42
        pipeline._ensure_services = MagicMock()
        pipeline._seen_usernames = PersonKeySet(["dtolnay"])
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        owners = [
            {
                "login": "dtolnay",
                "kind": "user",
                "github_login": "dtolnay",
            }
        ]
        crate_info = {
            "name": "serde",
            "downloads": 500000000,
            "recent_versions": [{"version": "1.0.210", "created_at": "2026-05-01T12:00:00Z"}],
            "repository_url": "https://github.com/serde-rs/serde",
        }

        hub = AsyncMock()
        hub.get_owner_users = AsyncMock(return_value=owners)
        hub.get_crate = AsyncMock(return_value=crate_info)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(target_ecosystem="crates.io", target_packages=["serde"])

        with patch.object(github_orchestrator, "CratesHubClient", return_value=hub):
            asyncio.run(pipeline._discover_registry_maintainers(query))

        sidecar_path = tmp_path / "registry_evidence.jsonl"
        assert sidecar_path.exists()
        rows = [json.loads(line) for line in sidecar_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["username"] == "dtolnay"
        assert rows[0]["run_id"] == 42
        assert rows[0]["query_id"] == query.id
        assert rows[0]["evidence"]["declared_roles"][0]["package"] == "serde"
        assert pipeline.stats["registry_evidence_sidecar_rows"] == 1


class TestRegistryEvidenceAttach:
    def test_registry_evidence_attached_to_candidate(self) -> None:
        pipeline = _make_pipeline()
        pipeline.brief_obj.has_v2_schema = True
        pipeline._ensure_services = MagicMock()

        evidence = {
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
                    "latest_release": "1.0.210",
                    "release_cadence": "",
                    "deprecated": False,
                }
            ],
        }
        pipeline._registry_evidence_by_username = {"dtolnay": evidence}

        candidate = _make_candidate("dtolnay", "David Tolnay")

        async def _fake_discover(_query: GitHubSearchQuery) -> tuple[int, list[str]]:
            pipeline._registry_evidence_by_username = {"dtolnay": evidence}
            return 1, ["dtolnay"]

        pipeline._discover_registry_maintainers = AsyncMock(side_effect=_fake_discover)
        pipeline._prepare_candidate_for_evaluation = AsyncMock(return_value=candidate)

        client = MagicMock()
        enricher = MagicMock()
        query = _registry_query()
        progress = GitHubProgress(brief_name="test")

        with patch.object(
            github_orchestrator,
            "github_facial_judge_batch",
            return_value=[
                SimpleNamespace(
                    decision="FACIAL_NO",
                    confidence=1.0,
                    rationale="out of scope",
                )
            ],
        ):
            asyncio.run(
                pipeline._execute_single_query(client, enricher, query, progress)
            )

        assert candidate.registry_evidence == evidence

    def test_candidates_jsonl_carries_registry_evidence_after_attach(self, tmp_path) -> None:
        pipeline = _make_pipeline()
        pipeline.brief_obj.has_v2_schema = True
        pipeline._ensure_services = MagicMock()
        pipeline.candidates_path = tmp_path / "candidates.jsonl"

        evidence = {
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
                    "latest_release": "1.0.210",
                    "release_cadence": "",
                    "deprecated": False,
                }
            ],
        }
        pipeline._registry_evidence_by_username = {"dtolnay": evidence}

        candidate = _make_candidate("dtolnay", "David Tolnay")
        append_jsonl(pipeline.candidates_path, pipeline._candidate_record(candidate))

        async def _fake_discover(_query: GitHubSearchQuery) -> tuple[int, list[str]]:
            pipeline._registry_evidence_by_username = {"dtolnay": evidence}
            return 1, ["dtolnay"]

        pipeline._discover_registry_maintainers = AsyncMock(side_effect=_fake_discover)
        pipeline._prepare_candidate_for_evaluation = AsyncMock(return_value=candidate)

        client = MagicMock()
        enricher = MagicMock()
        query = _registry_query()
        progress = GitHubProgress(brief_name="test")

        with patch.object(
            github_orchestrator,
            "github_facial_judge_batch",
            return_value=[
                SimpleNamespace(
                    decision="FACIAL_NO",
                    confidence=1.0,
                    rationale="out of scope",
                )
            ],
        ):
            asyncio.run(
                pipeline._execute_single_query(client, enricher, query, progress)
            )

        rows = [json.loads(line) for line in pipeline.candidates_path.read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0].get("registry_evidence") is None
        assert rows[-1]["registry_evidence"] == evidence


class TestRegistryHubFailureModes:
    def test_unreachable_hub_fails_query_no_exhaustion(self) -> None:
        pipeline = _make_pipeline()
        pipeline._exhaustion = ExhaustionState()
        pipeline._ensure_services = MagicMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._governor.check_limits_or_raise = MagicMock()
        pipeline._governor.should_enter_enrichment_only = MagicMock(return_value=False)
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        hub = AsyncMock()
        hub.get_packument = AsyncMock(return_value=None)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="npmjs.org",
            target_packages=["lodash"],
        )
        progress = GitHubProgress(brief_name="test", queries=[query])

        client = MagicMock()
        enricher = MagicMock()
        with patch.object(github_orchestrator, "NpmHubClient", return_value=hub):
            asyncio.run(pipeline._execute_queries(client, enricher, progress))

        assert query.status == "failed"
        assert "unreachable" in query.notes
        assert "registry_maintainer_discovery" not in pipeline._exhaustion.channels

    def test_empty_roster_records_zero_normally(self) -> None:
        pipeline = _make_pipeline()
        pipeline._exhaustion = ExhaustionState()
        pipeline._ensure_services = MagicMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._governor.check_limits_or_raise = MagicMock()
        pipeline._governor.should_enter_enrichment_only = MagicMock(return_value=False)
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        hub = AsyncMock()
        hub.get_owner_users = AsyncMock(return_value=[])
        hub.get_crate = AsyncMock(return_value={"name": "empty-crate", "downloads": 0})
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="crates.io",
            target_packages=["empty-crate"],
        )
        progress = GitHubProgress(brief_name="test", queries=[query])

        client = MagicMock()
        enricher = MagicMock()
        with patch.object(github_orchestrator, "CratesHubClient", return_value=hub):
            asyncio.run(pipeline._execute_queries(client, enricher, progress))

        assert query.status == "done"
        channel_stats = pipeline._exhaustion.channels["registry_maintainer_discovery"]
        assert channel_stats.queries_run == 1
        assert channel_stats.zero_result_streak == 1


class TestEcosystemsResolverLifecycle:
    def test_resolver_session_opened_in_production_getter(self) -> None:
        pipeline = _make_pipeline()
        session = MagicMock()
        session.closed = False

        with patch("shared.resolvers.ecosystems.aiohttp.ClientSession", return_value=session) as session_ctor:
            resolver = asyncio.run(pipeline._get_ecosystems_resolver())

        session_ctor.assert_called_once()
        assert resolver._session is session
        assert pipeline._ecosystems_resolver is resolver


class TestProjectQualityUnnamedPath:
    def test_project_quality_lands_on_unnamed_path(self) -> None:
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

        candidate = _make_candidate("alice", "Alice")
        candidate.top_repos = [
            GitHubRepo(
                name="serde",
                full_name="serde-rs/serde",
                owner_login="serde-rs",
            )
        ]

        quality = SimpleNamespace(score=0.72, criticality_band="established")
        query = _make_query(channel="user_search")
        facial_decision = SimpleNamespace(
            decision="FACIAL_YES",
            confidence=1.0,
            rationale="",
            candidate_name="Alice",
            profile_url="https://github.com/alice",
            to_dict=lambda: {},
        )

        with patch.object(
            github_orchestrator, "score_project", AsyncMock(return_value=quality)
        ) as score_mock, patch.object(
            pipeline,
            "_get_ecosystems_resolver",
            AsyncMock(return_value=MagicMock()),
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
                    [("alice", candidate)],
                    query,
                    GitHubProgress(brief_name="test"),
                )
            )

        score_mock.assert_awaited_once()
        assert candidate.portfolio_summary["project_quality"] == {
            "score": 0.72,
            "band": "established",
            "repo": "serde-rs/serde",
        }

    def test_project_quality_unaffected_by_declared_evidence(self) -> None:
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

        candidate = _make_candidate("dtolnay", "David Tolnay")
        candidate.registry_evidence = {
            "declared_roles": [
                {
                    "hub": "crates",
                    "handle": "dtolnay",
                    "package": "serde",
                    "role": "owner",
                    "corroborated_github_login": "dtolnay",
                }
            ],
            "packages": [],
        }
        candidate.top_repos = [
            GitHubRepo(
                name="serde",
                full_name="serde-rs/serde",
                owner_login="serde-rs",
            )
        ]

        quality = SimpleNamespace(score=0.72, criticality_band="established")
        query = _make_query(channel="registry_maintainer_discovery")
        facial_decision = SimpleNamespace(
            decision="FACIAL_YES",
            confidence=1.0,
            rationale="",
            candidate_name="David Tolnay",
            profile_url="https://github.com/dtolnay",
            to_dict=lambda: {},
        )

        with patch.object(
            github_orchestrator, "score_project", AsyncMock(return_value=quality)
        ) as score_mock, patch.object(
            pipeline,
            "_get_ecosystems_resolver",
            AsyncMock(return_value=MagicMock()),
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
                    [("dtolnay", candidate)],
                    query,
                    GitHubProgress(brief_name="test"),
                )
            )

        score_mock.assert_awaited_once()
        assert candidate.maintainership is None
        assert candidate.portfolio_summary["project_quality"] == {
            "score": 0.72,
            "band": "established",
            "repo": "serde-rs/serde",
        }

    def test_project_quality_scorer_exception_is_fail_soft(self) -> None:
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

        candidate = _make_candidate("alice", "Alice")
        candidate.top_repos = [
            GitHubRepo(
                name="serde",
                full_name="serde-rs/serde",
                owner_login="serde-rs",
            )
        ]
        query = _make_query(channel="user_search")
        facial_decision = SimpleNamespace(
            decision="FACIAL_YES",
            confidence=1.0,
            rationale="",
            candidate_name="Alice",
            profile_url="https://github.com/alice",
            to_dict=lambda: {},
        )

        with patch.object(
            github_orchestrator,
            "score_project",
            AsyncMock(side_effect=RuntimeError("scorer down")),
        ), patch.object(
            pipeline,
            "_get_ecosystems_resolver",
            AsyncMock(return_value=MagicMock()),
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
                    [("alice", candidate)],
                    query,
                    GitHubProgress(brief_name="test"),
                )
            )

        assert "project_quality" not in candidate.portfolio_summary
        error_stages = [call.args[0] for call in pipeline._observer.on_error.call_args_list]
        assert "project_quality" in error_stages
