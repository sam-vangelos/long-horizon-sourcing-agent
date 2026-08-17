import asyncio
from unittest.mock import AsyncMock

import pytest

from github.client import GitHubClient


def test_validate_credentials_checks_rate_limit_endpoint():
    client = GitHubClient(token="dummy")
    client._get = AsyncMock(return_value=(200, {"rate": {"limit": 5000}}, {}))

    asyncio.run(client.validate_credentials())

    client._get.assert_awaited_once_with("/rate_limit", "rest")


def test_validate_credentials_rejects_non_success_status():
    client = GitHubClient(token="dummy")
    client._get = AsyncMock(return_value=(500, None, {}))

    with pytest.raises(RuntimeError, match="GitHub credential preflight failed"):
        asyncio.run(client.validate_credentials())
