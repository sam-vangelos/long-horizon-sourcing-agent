"""Tests for :mod:`github.rosters` (Wave 3 — governance roster reader)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from github import maintainer_signal_cache as mcache
from github.client import GitHubClient
from github.rosters import (
    RosterEntry,
    _ROSTER_PATH_SEP,
    _pack_cached_roster_payload,
    _unpack_cached_roster_payload,
    derive_feedstock_repos,
    fetch_repo_roster,
    parse_codeowners,
    parse_maintainers_text,
    parse_recipe_maintainers,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

ROSTER_CACHE_KINDS = frozenset(
    {
        "roster_codeowners",
        "roster_recipe",
    }
)


@pytest.fixture(autouse=True)
def _isolated_roster_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcache, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(
        mcache,
        "SIGNAL_KINDS",
        mcache.SIGNAL_KINDS | ROSTER_CACHE_KINDS,
    )
    for kind in ROSTER_CACHE_KINDS:
        monkeypatch.setitem(mcache.TTL_BY_KIND, kind, mcache.TTL_BY_KIND["governance"])


def _fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def _all_output_strings(result: Any) -> list[str]:
    strings: list[str] = []
    if hasattr(result, "entries"):
        for entry in result.entries:
            strings.extend([entry.handle, entry.role, entry.source_file, entry.repo])
    if hasattr(result, "team_entries"):
        strings.extend(result.team_entries)
    if hasattr(result, "files_found"):
        strings.extend(result.files_found)
    if isinstance(result, list):
        for entry in result:
            if isinstance(entry, RosterEntry):
                strings.extend([entry.handle, entry.role, entry.source_file, entry.repo])
            elif isinstance(entry, str):
                strings.append(entry)
    return strings


def test_parse_codeowners_extracts_handles_teams_and_skips_email() -> None:
    text = _fixture_text("roster_codeowners_sample")
    entries, teams = parse_codeowners(
        text,
        source_file=".github/CODEOWNERS",
        repo="acme/widget",
    )

    handles = {entry.handle for entry in entries}
    assert handles == {"alice", "bob", "carol"}
    assert all(entry.role == "code_owner" for entry in entries)
    assert teams == ["acme/platform-team", "acme/docs-team"]

    alice_entries = [entry for entry in entries if entry.handle == "alice"]
    assert len(alice_entries) == 1
    assert alice_entries[0].source_file == ".github/CODEOWNERS"

    output = _all_output_strings((entries, teams))
    assert not any("@" in value and "." in value.split("@", 1)[-1] for value in output)
    assert "owner-one@example.com" not in json.dumps(output)


def test_bare_names_are_never_extracted() -> None:
    text = _fixture_text("roster_maintainers_sample.md")
    entries = parse_maintainers_text(
        text,
        source_file="MAINTAINERS.md",
        repo="acme/widget",
        role="maintainer",
    )

    handles = {entry.handle for entry in entries}
    assert handles == {"dana", "erin-dev"}
    for forbidden in ("jane", "smith", "bob", "johnson", "frank", "miller"):
        assert forbidden not in handles


def test_parse_recipe_maintainers_with_comments_and_quoting() -> None:
    text = _fixture_text("roster_recipe_meta.yaml")
    entries = parse_recipe_maintainers(text, repo="conda-forge/numpy-feedstock")

    handles = [entry.handle for entry in entries]
    assert handles == ["numpy-owner", "quoted-maintainer", "double-quoted"]
    assert all(entry.role == "recipe_maintainer" for entry in entries)


def test_parse_recipe_maintainers_malformed_fails_soft() -> None:
    text = _fixture_text("roster_recipe_meta_malformed.yaml")
    assert parse_recipe_maintainers(text, repo="conda-forge/broken-feedstock") == []


def test_fetch_repo_roster_caches_second_call_makes_zero_client_calls() -> None:
    contents = {
        ("acme/widget", ".github/CODEOWNERS"): _fixture_text("roster_codeowners_sample"),
        ("acme/widget", "MAINTAINERS.md"): _fixture_text("roster_maintainers_sample.md"),
        ("acme/widget", "GOVERNANCE.md"): "@gov-lead",
    }
    client = GitHubClient(token="dummy")
    mock_contents = AsyncMock(side_effect=lambda owner_repo, path: contents.get((owner_repo, path)))
    client.get_repo_contents = mock_contents  # type: ignore[assignment]

    first = asyncio.run(fetch_repo_roster(client, "acme/widget"))
    second = asyncio.run(fetch_repo_roster(client, "acme/widget"))

    assert mock_contents.await_count > 0
    call_count_after_first = mock_contents.await_count
    assert first.entries
    assert second.entries == first.entries
    assert mock_contents.await_count == call_count_after_first


def test_fetch_repo_roster_fail_soft_on_client_errors() -> None:
    client = GitHubClient(token="dummy")

    async def _raising_contents(owner_repo: str, path: str) -> str | None:
        if path == ".github/CODEOWNERS":
            raise RuntimeError("network down")
        if path == "CODEOWNERS":
            return "* @solo-owner"
        return None

    client.get_repo_contents = _raising_contents  # type: ignore[assignment]

    result = asyncio.run(fetch_repo_roster(client, "acme/resilient"))

    assert any(entry.handle == "solo-owner" for entry in result.entries)
    assert ".github/CODEOWNERS" not in result.files_found


def test_fetch_repo_roster_reads_recipe_for_feedstock_repos() -> None:
    contents = {
        ("conda-forge/numpy-feedstock", "recipe/meta.yaml"): _fixture_text(
            "roster_recipe_meta.yaml"
        ),
    }
    client = GitHubClient(token="dummy")
    client.get_repo_contents = AsyncMock(  # type: ignore[assignment]
        side_effect=lambda owner_repo, path: contents.get((owner_repo, path))
    )

    result = asyncio.run(fetch_repo_roster(client, "conda-forge/numpy-feedstock"))

    assert "recipe/meta.yaml" in result.files_found
    assert any(entry.role == "recipe_maintainer" for entry in result.entries)


@dataclass
class _CapabilityArea:
    name: str
    github_code_signals: list[str] = field(default_factory=list)


@dataclass
class _NewBrief:
    capability_areas: list[_CapabilityArea] = field(default_factory=list)


@dataclass
class _Brief:
    target_stacks: list[str] = field(default_factory=list)
    capability_areas: list[_CapabilityArea] = field(default_factory=list)
    _new_brief: _NewBrief | None = None


def test_derives_feedstock_repos_from_python_import_signals() -> None:
    brief = _Brief(
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Scientific Python",
                    github_code_signals=[
                        "import numpy as np",
                        "from pandas import DataFrame",
                        "import numpy",  # duplicate package token
                    ],
                )
            ]
        )
    )

    assert derive_feedstock_repos(brief) == [
        "conda-forge/numpy-feedstock",
        "conda-forge/pandas-feedstock",
    ]


def test_no_feedstocks_for_briefs_without_python_signals() -> None:
    brief = _Brief(
        target_stacks=["npm"],
        _new_brief=_NewBrief(
            capability_areas=[
                _CapabilityArea(
                    name="Frontend",
                    github_code_signals=[
                        "require('react')",
                        "use serde::Deserialize",
                    ],
                )
            ]
        )
    )

    assert derive_feedstock_repos(brief) == []


def test_fetch_failure_does_not_negative_cache() -> None:
    client = GitHubClient(token="dummy")
    mock_contents = AsyncMock(side_effect=RuntimeError("api outage"))
    client.get_repo_contents = mock_contents  # type: ignore[assignment]

    result = asyncio.run(fetch_repo_roster(client, "acme/outage"))

    assert result.entries == []
    assert mcache.get("acme", "outage", "roster_codeowners") is None
    assert mcache.get("acme", "outage", "maintainers_file") is None

    mock_contents.reset_mock()
    mock_contents.side_effect = lambda owner_repo, path: None

    asyncio.run(fetch_repo_roster(client, "acme/missing"))

    assert mcache.get("acme", "missing", "roster_codeowners") is not None
    assert mcache.get("acme", "missing", "roster_codeowners").data == ""


def test_recipe_maintainers_survive_sibling_keys_and_comments() -> None:
    text = """\
package:
  name: example
recipe-maintainers:
  - alice-smithy
  - bob-smithy  # primary maintainer
  # rotation note
  - carol-smithy
feedstock-name: my-package
extra:
  recipe-maintainers:
    - ignored-nested
"""
    entries = parse_recipe_maintainers(text, repo="conda-forge/my-package-feedstock")

    assert [entry.handle for entry in entries] == [
        "alice-smithy",
        "bob-smithy",
        "carol-smithy",
    ]


def test_team_refs_not_extracted_as_orgs_in_free_text() -> None:
    text = (
        "Governance contact: @kubernetes/steering-committee\n"
        "Also see @solo-maintainer for day-to-day work.\n"
    )
    entries = parse_maintainers_text(
        text,
        source_file="GOVERNANCE.md",
        repo="kubernetes/kubernetes",
        role="governance_listed",
    )

    assert [entry.handle for entry in entries] == ["solo-maintainer"]


def test_rosters_share_classifier_cache_kinds() -> None:
    maintainers_text = _fixture_text("roster_maintainers_sample.md")
    mcache.put("acme", "widget", "maintainers_file", maintainers_text)

    contents = {
        ("acme/widget", ".github/CODEOWNERS"): _fixture_text("roster_codeowners_sample"),
        ("acme/widget", "GOVERNANCE.md"): "@gov-lead",
    }
    client = GitHubClient(token="dummy")
    mock_contents = AsyncMock(side_effect=lambda owner_repo, path: contents.get((owner_repo, path)))
    client.get_repo_contents = mock_contents  # type: ignore[assignment]

    result = asyncio.run(fetch_repo_roster(client, "acme/widget"))

    assert any(entry.handle == "dana" for entry in result.entries)
    assert not any(
        call.args == ("acme/widget", "MAINTAINERS.md") for call in mock_contents.await_args_list
    )
    assert not any(
        call.args == ("acme/widget", "MAINTAINERS") for call in mock_contents.await_args_list
    )

    mock_contents.reset_mock()
    mock_contents.side_effect = lambda owner_repo, path: contents.get((owner_repo, path))

    asyncio.run(fetch_repo_roster(client, "acme/widget"))

    assert not any(
        call.args[1] in {"MAINTAINERS.md", "MAINTAINERS", "docs/MAINTAINERS.md"}
        for call in mock_contents.await_args_list
    )


def test_roster_cache_pack_unpack_round_trip() -> None:
    assert _ROSTER_PATH_SEP == "\n"
    path = ".github/CODEOWNERS"
    text = "* @alice\n/docs/ @bob\n"
    packed = _pack_cached_roster_payload(path, text, "roster_codeowners")
    unpacked_text, unpacked_path = _unpack_cached_roster_payload(
        packed,
        (".github/CODEOWNERS", "CODEOWNERS"),
        "roster_codeowners",
    )
    assert unpacked_path == path
    assert unpacked_text == text
