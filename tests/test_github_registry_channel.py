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

from github.rosters import RosterEntry, RosterResult
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


def _resolver_lookup_side_effect(mapping: dict[str, dict | None]):
    async def _lookup(_registry: str, name: str) -> dict | None:
        return mapping.get(name)

    return _lookup


def _mock_ecosystems_resolver(pipeline, lookup_map: dict[str, dict | None]) -> MagicMock:
    resolver = MagicMock()
    resolver.lookup_package = AsyncMock(side_effect=_resolver_lookup_side_effect(lookup_map))
    pipeline._get_ecosystems_resolver = AsyncMock(return_value=resolver)
    return resolver


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
            "recent_downloads": 500000000,
            "recent_versions": [{"version": "1.0.210", "created_at": "2026-05-01T12:00:00Z"}],
            "repository_url": "https://github.com/serde-rs/serde",
        }

        hub = AsyncMock()
        hub.get_owner_users = AsyncMock(return_value=owners)
        hub.get_crate = AsyncMock(return_value=crate_info)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(target_ecosystem="crates.io", target_packages=["serde"])

        _mock_ecosystems_resolver(
            pipeline,
            {"serde": {"name": "serde", "dependent_packages_count": 42}},
        )

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

        _mock_ecosystems_resolver(
            pipeline,
            {"chalk": {"name": "chalk", "dependent_packages_count": 100}},
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

        _mock_ecosystems_resolver(
            pipeline,
            {"lodash": {"name": "lodash", "dependent_packages_count": 200}},
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
            "recent_downloads": 500000000,
            "recent_versions": [{"version": "1.0.210", "created_at": "2026-05-01T12:00:00Z"}],
            "repository_url": "https://github.com/serde-rs/serde",
        }

        hub = AsyncMock()
        hub.get_owner_users = AsyncMock(return_value=owners)
        hub.get_crate = AsyncMock(return_value=crate_info)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(target_ecosystem="crates.io", target_packages=["serde"])

        _mock_ecosystems_resolver(
            pipeline,
            {"serde": {"name": "serde", "dependent_packages_count": 42}},
        )

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

        _mock_ecosystems_resolver(
            pipeline,
            {"lodash": {"name": "lodash", "dependent_packages_count": 200}},
        )

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
        hub.get_crate = AsyncMock(return_value={"name": "empty-crate", "recent_downloads": 0})
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="crates.io",
            target_packages=["empty-crate"],
        )
        progress = GitHubProgress(brief_name="test", queries=[query])

        _mock_ecosystems_resolver(
            pipeline,
            {"empty-crate": {"name": "empty-crate", "dependent_packages_count": 0}},
        )

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


class TestBareTokenResolution:
    def test_bare_tokens_resolve_before_hub_fetch(self) -> None:
        pipeline = _make_pipeline()
        pipeline._ensure_services = MagicMock()
        pipeline._dedup_usernames = lambda usernames: list(usernames)
        pipeline._client = MagicMock()
        pipeline._client.get_repo = AsyncMock(
            return_value={
                "owner": {"login": "user-owner", "type": "User"},
            }
        )

        roster_result = RosterResult(
            repo="huggingface/trl",
            entries=[
                RosterEntry(
                    handle="trl-maint",
                    role="maintainer",
                    source_file="CODEOWNERS",
                    repo="huggingface/trl",
                ),
            ],
            team_entries=[],
            files_found=["CODEOWNERS"],
        )

        hub = AsyncMock()
        hub.get_project = AsyncMock(
            return_value={
                "name": "trl",
                "latest_version": "0.9.0",
                "repository_url": "https://github.com/huggingface/trl",
                "yanked": False,
            }
        )
        hub.get_downloads_last_month = AsyncMock(return_value=50000)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="pypi.org",
            target_packages=["trl", "grpo"],
        )

        _mock_ecosystems_resolver(
            pipeline,
            {
                "trl": {
                    "name": "trl",
                    "dependent_packages_count": 120,
                    "repository_url": "https://github.com/huggingface/trl",
                },
                "grpo": {"missing": True},
            },
        )

        with patch("github.hubs.pypi.PyPiHubClient", return_value=hub), patch.object(
            github_orchestrator,
            "fetch_repo_roster",
            AsyncMock(return_value=roster_result),
        ):
            pre_dedup, usernames = asyncio.run(
                pipeline._discover_registry_maintainers(query)
            )

        hub.get_project.assert_awaited_once_with("trl")
        hub.get_downloads_last_month.assert_awaited_once_with("trl")
        assert pipeline.stats["registry_tokens_unresolved"] == 1
        assert pre_dedup == 2
        assert set(usernames) == {"trl-maint", "user-owner"}


class TestPyPiRosterComposition:
    def test_pypi_discovers_via_roster_composition(self) -> None:
        pipeline = _make_pipeline()
        pipeline._ensure_services = MagicMock()
        pipeline._dedup_usernames = lambda usernames: list(usernames)
        pipeline._client = MagicMock()
        pipeline._client.get_repo = AsyncMock(
            return_value={
                "owner": {"login": "user-owner", "type": "User"},
            }
        )

        roster_result = RosterResult(
            repo="huggingface/trl",
            entries=[
                RosterEntry(
                    handle="roster-maint",
                    role="maintainer",
                    source_file="CODEOWNERS",
                    repo="huggingface/trl",
                ),
            ],
            team_entries=[],
            files_found=["CODEOWNERS"],
        )

        hub = AsyncMock()
        hub.get_project = AsyncMock(
            return_value={
                "name": "trl",
                "latest_version": "0.9.0",
                "repository_url": "https://github.com/huggingface/trl",
                "yanked": False,
            }
        )
        hub.get_downloads_last_month = AsyncMock(return_value=50000)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="pypi.org",
            target_packages=["trl"],
        )

        _mock_ecosystems_resolver(
            pipeline,
            {
                "trl": {
                    "name": "trl",
                    "dependent_packages_count": 88,
                    "repository_url": "https://github.com/huggingface/trl",
                },
            },
        )

        with patch("github.hubs.pypi.PyPiHubClient", return_value=hub), patch.object(
            github_orchestrator,
            "fetch_repo_roster",
            AsyncMock(return_value=roster_result),
        ):
            pre_dedup, usernames = asyncio.run(
                pipeline._discover_registry_maintainers(query)
            )

        assert pre_dedup == 2
        assert set(usernames) == {"roster-maint", "user-owner"}

        roster_evidence = pipeline._registry_evidence_by_username["roster-maint"]
        assert roster_evidence["declared_roles"][0] == {
            "hub": "pypi",
            "handle": "roster-maint",
            "package": "trl",
            "repo": "huggingface/trl",
            "role": "maintainer",
            "corroborated_github_login": "roster-maint",
            "source_file": "CODEOWNERS",
        }
        assert roster_evidence["packages"][0]["hub"] == "pypi"
        assert roster_evidence["packages"][0]["reverse_dependencies"] == 88

        owner_evidence = pipeline._registry_evidence_by_username["user-owner"]
        assert owner_evidence["declared_roles"][0]["role"] == "repository_owner"

    def test_pypi_org_owner_not_emitted_bare(self) -> None:
        pipeline = _make_pipeline()
        pipeline._ensure_services = MagicMock()
        pipeline._dedup_usernames = lambda usernames: list(usernames)
        pipeline._client = MagicMock()
        pipeline._client.get_repo = AsyncMock(
            return_value={
                "owner": {"login": "huggingface", "type": "Organization"},
            }
        )

        roster_result = RosterResult(
            repo="huggingface/trl",
            entries=[],
            team_entries=[],
            files_found=[],
        )

        hub = AsyncMock()
        hub.get_project = AsyncMock(
            return_value={
                "name": "trl",
                "latest_version": "0.9.0",
                "repository_url": "https://github.com/huggingface/trl",
                "yanked": False,
            }
        )
        hub.get_downloads_last_month = AsyncMock(return_value=50000)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="pypi.org",
            target_packages=["trl"],
        )

        _mock_ecosystems_resolver(
            pipeline,
            {
                "trl": {
                    "name": "trl",
                    "dependent_packages_count": 88,
                    "repository_url": "https://github.com/huggingface/trl",
                },
            },
        )

        with patch("github.hubs.pypi.PyPiHubClient", return_value=hub), patch.object(
            github_orchestrator,
            "fetch_repo_roster",
            AsyncMock(return_value=roster_result),
        ):
            pre_dedup, usernames = asyncio.run(
                pipeline._discover_registry_maintainers(query)
            )

        assert pre_dedup == 0
        assert usernames == []
        evidence_by_username = getattr(pipeline, "_registry_evidence_by_username", {})
        assert "huggingface" not in evidence_by_username


class TestHuggingFaceEvidenceGate:
    def test_hf_gate_closed_yields_legitimate_zero(self, capsys) -> None:
        pipeline = _make_pipeline()
        pipeline._exhaustion = ExhaustionState()
        pipeline._ensure_services = MagicMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._governor.check_limits_or_raise = MagicMock()
        pipeline._governor.should_enter_enrichment_only = MagicMock(return_value=False)
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        hub = AsyncMock()
        hub.find_exact_artifacts = AsyncMock(return_value=[])
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="huggingface.co",
            target_packages=["missing-token"],
        )
        progress = GitHubProgress(brief_name="test", queries=[query])

        client = MagicMock()
        enricher = MagicMock()
        with patch("github.hubs.hf.HuggingFaceHubClient", return_value=hub):
            asyncio.run(pipeline._execute_queries(client, enricher, progress))

        assert query.status == "done"
        err = capsys.readouterr().err
        assert "no exact huggingface artifacts" in err
        assert "registry_maintainer_discovery" in pipeline._exhaustion.channels

    def test_hf_author_requires_card_link_corroboration(self) -> None:
        pipeline = _make_pipeline()
        pipeline._ensure_services = MagicMock()
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        hub = AsyncMock()
        hub.find_exact_artifacts = AsyncMock(
            return_value=[
                {
                    "artifact_id": "alice/trl-model",
                    "kind": "model",
                    "author": "alice",
                    "downloads": 1000,
                    "likes": 50,
                },
                {
                    "artifact_id": "bob/trl-model",
                    "kind": "model",
                    "author": "bob",
                    "downloads": 900,
                    "likes": 40,
                },
            ]
        )
        hub.get_model_card_links = AsyncMock(
            side_effect=lambda artifact_id: (
                ["https://github.com/alice/trl"]
                if artifact_id == "alice/trl-model"
                else ["https://github.com/other-org/trl"]
            )
        )
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="huggingface.co",
            target_packages=["trl-model"],
        )

        with patch("github.hubs.hf.HuggingFaceHubClient", return_value=hub):
            pre_dedup, usernames = asyncio.run(
                pipeline._discover_registry_maintainers(query)
            )

        assert pre_dedup == 1
        assert usernames == ["alice"]
        assert pipeline.stats["hf_authors_unresolved"] == 1
        assert "bob" not in pipeline._registry_evidence_by_username
        evidence = pipeline._registry_evidence_by_username["alice"]
        assert evidence["packages"][0]["kind"] == "model"
        assert evidence["packages"][0]["downloads_window"] == "last-month"
        assert evidence["declared_roles"][0]["role"] == "author"


class TestPyPiHubFailureModes:
    def test_unreachable_pypi_hub_fails_query_no_exhaustion(self) -> None:
        pipeline = _make_pipeline()
        pipeline._exhaustion = ExhaustionState()
        pipeline._ensure_services = MagicMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._governor.check_limits_or_raise = MagicMock()
        pipeline._governor.should_enter_enrichment_only = MagicMock(return_value=False)
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        hub = AsyncMock()
        hub.get_project = AsyncMock(return_value=None)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="pypi.org",
            target_packages=["trl"],
        )
        progress = GitHubProgress(brief_name="test", queries=[query])

        _mock_ecosystems_resolver(
            pipeline,
            {
                "trl": {
                    "name": "trl",
                    "dependent_packages_count": 88,
                    "repository_url": "https://github.com/huggingface/trl",
                },
            },
        )

        client = MagicMock()
        pipeline._client = client
        enricher = MagicMock()
        with patch("github.hubs.pypi.PyPiHubClient", return_value=hub):
            asyncio.run(pipeline._execute_queries(client, enricher, progress))

        assert query.status == "failed"
        assert "unreachable" in query.notes
        assert "registry_maintainer_discovery" not in pipeline._exhaustion.channels


class TestResolverOutageFailClosed:
    def test_resolver_outage_fails_query_no_exhaustion(self) -> None:
        pipeline = _make_pipeline()
        pipeline._exhaustion = ExhaustionState()
        pipeline._ensure_services = MagicMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._governor.check_limits_or_raise = MagicMock()
        pipeline._governor.should_enter_enrichment_only = MagicMock(return_value=False)
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        query = _registry_query(
            target_ecosystem="pypi.org",
            target_packages=["trl", "grpo"],
        )
        progress = GitHubProgress(brief_name="test", queries=[query])

        _mock_ecosystems_resolver(
            pipeline,
            {
                "trl": None,
                "grpo": None,
            },
        )

        client = MagicMock()
        enricher = MagicMock()
        asyncio.run(pipeline._execute_queries(client, enricher, progress))

        assert query.status == "failed"
        assert "ecosyste.ms resolver unavailable" in query.notes
        assert pipeline.stats["resolver_errors"] == 2
        assert "registry_maintainer_discovery" not in pipeline._exhaustion.channels

    def test_all_missing_tokens_is_legitimate_zero(self) -> None:
        pipeline = _make_pipeline()
        pipeline._exhaustion = ExhaustionState()
        pipeline._ensure_services = MagicMock()
        pipeline._save_progress = MagicMock()
        pipeline._get_api_status = MagicMock(return_value={})
        pipeline._governor.check_limits_or_raise = MagicMock()
        pipeline._governor.should_enter_enrichment_only = MagicMock(return_value=False)
        pipeline._dedup_usernames = lambda usernames: list(usernames)

        query = _registry_query(
            target_ecosystem="pypi.org",
            target_packages=["trl", "grpo"],
        )
        progress = GitHubProgress(brief_name="test", queries=[query])

        _mock_ecosystems_resolver(
            pipeline,
            {
                "trl": {"missing": True},
                "grpo": {"missing": True},
            },
        )

        client = MagicMock()
        enricher = MagicMock()
        asyncio.run(pipeline._execute_queries(client, enricher, progress))

        assert query.status == "done"
        assert pipeline.stats["registry_tokens_unresolved"] == 2
        assert pipeline.stats.get("resolver_errors", 0) == 0
        channel_stats = pipeline._exhaustion.channels["registry_maintainer_discovery"]
        assert channel_stats.queries_run == 1
        assert channel_stats.zero_result_streak == 1


class TestHfOrgAuthorGuard:
    def test_hf_org_author_not_emitted(self) -> None:
        pipeline = _make_pipeline()
        pipeline._ensure_services = MagicMock()
        pipeline._dedup_usernames = lambda usernames: list(usernames)
        pipeline._client = MagicMock()
        pipeline._client.get_repo = AsyncMock(
            side_effect=lambda owner_repo: {
                "princeton-nlp/SWE-bench": {
                    "owner": {"login": "princeton-nlp", "type": "Organization"},
                },
                "alice/trl": {
                    "owner": {"login": "alice", "type": "User"},
                },
            }.get(owner_repo)
        )

        hub = AsyncMock()
        hub.find_exact_artifacts = AsyncMock(
            side_effect=lambda token: (
                [
                    {
                        "artifact_id": "princeton-nlp/SWE-bench",
                        "kind": "dataset",
                        "author": "princeton-nlp",
                        "downloads": 5000,
                        "likes": 10,
                    }
                ]
                if token == "swe-bench"
                else [
                    {
                        "artifact_id": "alice/trl-model",
                        "kind": "model",
                        "author": "alice",
                        "downloads": 1000,
                        "likes": 50,
                    }
                ]
            )
        )
        hub.get_model_card_links = AsyncMock(
            side_effect=lambda artifact_id: (
                ["https://github.com/princeton-nlp/SWE-bench"]
                if artifact_id == "princeton-nlp/SWE-bench"
                else ["https://github.com/alice/trl"]
            )
        )
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="huggingface.co",
            target_packages=["swe-bench", "trl-model"],
        )

        with patch("github.hubs.hf.HuggingFaceHubClient", return_value=hub):
            pre_dedup, usernames = asyncio.run(
                pipeline._discover_registry_maintainers(query)
            )

        assert pre_dedup == 1
        assert usernames == ["alice"]
        assert pipeline.stats["hf_org_authors_skipped"] == 1
        assert "princeton-nlp" not in pipeline._registry_evidence_by_username


class TestBatchReportRegistryDiagnostics:
    def test_batch_report_carries_registry_diagnostics(self) -> None:
        pipeline = _make_pipeline()
        pipeline.stats.update(
            {
                "registry_tokens_unresolved": 3,
                "resolver_errors": 2,
                "hf_authors_unresolved": 4,
                "hf_org_authors_skipped": 1,
                "pypi_packages_without_repo": 5,
            }
        )

        report = pipeline._build_batch_report([])

        assert report.registry_diagnostics == {
            "registry_tokens_unresolved": 3,
            "resolver_errors": 2,
            "hf_authors_unresolved": 4,
            "hf_org_authors_skipped": 1,
            "pypi_packages_without_repo": 5,
        }


class TestPyPiDeclaredRoleRepo:
    def test_pypi_declared_role_carries_repo(self) -> None:
        pipeline = _make_pipeline()
        pipeline._ensure_services = MagicMock()
        pipeline._dedup_usernames = lambda usernames: list(usernames)
        pipeline._client = MagicMock()
        pipeline._client.get_repo = AsyncMock(
            return_value={
                "owner": {"login": "huggingface", "type": "Organization"},
            }
        )

        roster_result = RosterResult(
            repo="huggingface/trl",
            entries=[
                RosterEntry(
                    handle="roster-maint",
                    role="maintainer",
                    source_file="CODEOWNERS",
                    repo="huggingface/trl",
                ),
            ],
            team_entries=[],
            files_found=["CODEOWNERS"],
        )

        hub = AsyncMock()
        hub.get_project = AsyncMock(
            return_value={
                "name": "trl",
                "summary": "Transformer Reinforcement Learning",
                "latest_version": "0.9.0",
                "repository_url": "https://github.com/huggingface/trl",
                "yanked": False,
            }
        )
        hub.get_downloads_last_month = AsyncMock(return_value=50000)
        hub.__aenter__ = AsyncMock(return_value=hub)
        hub.__aexit__ = AsyncMock(return_value=False)

        query = _registry_query(
            target_ecosystem="pypi.org",
            target_packages=["trl"],
        )

        _mock_ecosystems_resolver(
            pipeline,
            {
                "trl": {
                    "name": "trl",
                    "dependent_packages_count": 88,
                    "repository_url": "https://github.com/huggingface/trl",
                },
            },
        )

        with patch("github.hubs.pypi.PyPiHubClient", return_value=hub), patch.object(
            github_orchestrator,
            "fetch_repo_roster",
            AsyncMock(return_value=roster_result),
        ):
            asyncio.run(pipeline._discover_registry_maintainers(query))

        evidence = pipeline._registry_evidence_by_username["roster-maint"]
        assert evidence["declared_roles"][0]["repo"] == "huggingface/trl"
        assert evidence["packages"][0]["summary"] == "Transformer Reinforcement Learning"
