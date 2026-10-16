"""GitHub session orchestrator wiring smoke tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from github.session_orchestrator import _run_session


@pytest.mark.asyncio
async def test_run_session_delegates_to_github_pipeline() -> None:
    with patch("github.session_orchestrator.GitHubPipeline") as mock_cls:
        instance = mock_cls.return_value
        instance.run = AsyncMock(return_value={"ok": True})

        out = await _run_session("/tmp/brief.json", None, False)

        mock_cls.assert_called_once_with(brief_path="/tmp/brief.json", output_dir=None)
        instance.run.assert_awaited_once_with(resume=False)
        assert out == {"ok": True}
