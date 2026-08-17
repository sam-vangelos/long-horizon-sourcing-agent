"""Tests for OSS Maintainers Slice 3 GitHubClient endpoint additions.

Smoke + behavior tests for the five new methods on
:class:`github.client.GitHubClient`:

- :meth:`get_user_orgs` — ``/users/{login}/orgs``
- :meth:`list_repo_pulls` — ``/repos/{o}/{r}/pulls``
- :meth:`get_pull_reviews` — ``/repos/{o}/{r}/pulls/{n}/reviews``
- :meth:`list_repo_releases` — ``/repos/{o}/{r}/releases``
- :meth:`get_repo_contents` — ``/repos/{o}/{r}/contents/{path}``

The client itself uses ``_paginate`` (which calls ``_get``) for
list-shaped endpoints and ``_get`` directly for the single-file
``get_repo_contents``. Tests mock at the appropriate boundary so the
endpoint-shape contract is the unit under test.
"""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock

import pytest

from github.client import GitHubClient


# ---------------------------------------------------------------------------
# get_user_orgs
# ---------------------------------------------------------------------------


def test_get_user_orgs_returns_list_from_paginate() -> None:
    client = GitHubClient(token="dummy")
    client._paginate = AsyncMock(
        return_value=[{"login": "kubernetes"}, {"login": "etcd-io"}]
    )

    result = asyncio.run(client.get_user_orgs("torvalds"))

    assert result == [{"login": "kubernetes"}, {"login": "etcd-io"}]
    args, kwargs = client._paginate.call_args
    assert args[0] == "/users/torvalds/orgs"


def test_get_user_orgs_returns_empty_when_paginate_empty() -> None:
    client = GitHubClient(token="dummy")
    client._paginate = AsyncMock(return_value=[])

    result = asyncio.run(client.get_user_orgs("nobody"))

    assert result == []


# ---------------------------------------------------------------------------
# list_repo_pulls
# ---------------------------------------------------------------------------


def test_list_repo_pulls_uses_closed_state_by_default() -> None:
    client = GitHubClient(token="dummy")
    client._paginate = AsyncMock(
        return_value=[{"number": 1, "merged_by": {"login": "alice"}}]
    )

    result = asyncio.run(client.list_repo_pulls("kubernetes/kubernetes"))

    assert result == [{"number": 1, "merged_by": {"login": "alice"}}]
    args, kwargs = client._paginate.call_args
    assert args[0] == "/repos/kubernetes/kubernetes/pulls"
    params = kwargs["params"]
    assert params["state"] == "closed"
    assert params["sort"] == "updated"
    assert params["direction"] == "desc"


def test_list_repo_pulls_propagates_state_override() -> None:
    client = GitHubClient(token="dummy")
    client._paginate = AsyncMock(return_value=[])

    asyncio.run(client.list_repo_pulls("etcd-io/etcd", state="open"))

    _args, kwargs = client._paginate.call_args
    assert kwargs["params"]["state"] == "open"


# ---------------------------------------------------------------------------
# get_pull_reviews
# ---------------------------------------------------------------------------


def test_get_pull_reviews_returns_paginate_result() -> None:
    client = GitHubClient(token="dummy")
    client._paginate = AsyncMock(
        return_value=[
            {"user": {"login": "alice"}, "state": "APPROVED"},
            {"user": {"login": "bob"}, "state": "COMMENTED"},
        ]
    )

    result = asyncio.run(client.get_pull_reviews("kubernetes/kubernetes", 12345))

    assert len(result) == 2
    args, _kwargs = client._paginate.call_args
    assert args[0] == "/repos/kubernetes/kubernetes/pulls/12345/reviews"


# ---------------------------------------------------------------------------
# list_repo_releases
# ---------------------------------------------------------------------------


def test_list_repo_releases_returns_releases_with_author() -> None:
    client = GitHubClient(token="dummy")
    fake_releases = [
        {"tag_name": "v1.30.0", "author": {"login": "alice"}},
        {"tag_name": "v1.29.5", "author": {"login": "alice"}},
        {"tag_name": "v1.29.4", "author": {"login": "bob"}},
    ]
    client._paginate = AsyncMock(return_value=fake_releases)

    result = asyncio.run(client.list_repo_releases("kubernetes/kubernetes"))

    assert result == fake_releases
    args, _kwargs = client._paginate.call_args
    assert args[0] == "/repos/kubernetes/kubernetes/releases"


# ---------------------------------------------------------------------------
# get_repo_contents
# ---------------------------------------------------------------------------


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def test_get_repo_contents_decodes_plaintext() -> None:
    client = GitHubClient(token="dummy")
    contents = "alice <alice@example.com>\nbob <bob@example.com>\n"
    client._get = AsyncMock(
        return_value=(
            200,
            {"content": _b64(contents), "size": len(contents)},
            {},
        )
    )

    result = asyncio.run(
        client.get_repo_contents("kubernetes/kubernetes", "MAINTAINERS")
    )

    assert result == contents
    args, kwargs = client._get.call_args
    assert args[0] == "/repos/kubernetes/kubernetes/contents/MAINTAINERS"
    assert kwargs["use_etag"] is True


def test_get_repo_contents_returns_none_on_404() -> None:
    client = GitHubClient(token="dummy")
    client._get = AsyncMock(return_value=(404, None, {}))

    result = asyncio.run(
        client.get_repo_contents("kubernetes/kubernetes", "GOVERNANCE.md")
    )

    assert result is None


def test_get_repo_contents_returns_none_for_oversized_files() -> None:
    """Files >1MB are skipped — the contents API uses a different shape."""

    client = GitHubClient(token="dummy")
    client._get = AsyncMock(
        return_value=(
            200,
            {"content": _b64("x" * 200), "size": 2_000_000},
            {},
        )
    )

    result = asyncio.run(
        client.get_repo_contents("some-org/huge-repo", "GIANT_FILE.md")
    )

    assert result is None


def test_get_repo_contents_returns_none_on_decode_failure() -> None:
    client = GitHubClient(token="dummy")
    client._get = AsyncMock(
        return_value=(
            200,
            {"content": "!!!not-valid-base64!!!", "size": 100},
            {},
        )
    )

    result = asyncio.run(
        client.get_repo_contents("kubernetes/kubernetes", "CONTRIBUTORS.md")
    )

    # decode("utf-8", errors="replace") will produce *some* string from
    # malformed base64 once `b64decode` raises — the wrapper catches and
    # returns None. So either None or an empty/garbage string is the
    # contract here. The strict contract is "no crash"; we assert None
    # because b64decode validates its input.
    assert result is None


def test_get_repo_contents_returns_none_when_content_key_missing() -> None:
    client = GitHubClient(token="dummy")
    client._get = AsyncMock(return_value=(200, {"size": 100}, {}))

    result = asyncio.run(
        client.get_repo_contents("kubernetes/kubernetes", "CONTRIBUTORS.md")
    )

    assert result is None
