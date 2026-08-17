"""Direct tests for Cloris's dedicated Chrome launcher."""

from __future__ import annotations

import json
from typing import Any

import pytest

from cloris import chrome_launcher
from cloris.chrome_launcher import ChromeStatus


def test_ensure_running_closes_restored_local_scaffold_tabs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy CDP profile can still be visually stale if Chrome restored
    raw source scaffold tabs. The no-op healthy path should scrub only those
    local scaffold tabs before returning.
    """

    closed: dict[str, int] = {}

    monkeypatch.setattr(chrome_launcher.sys, "platform", "darwin")
    monkeypatch.setattr(chrome_launcher, "_chrome_installed", lambda: True)
    monkeypatch.setattr(chrome_launcher, "is_healthy", lambda: True)
    monkeypatch.setattr(
        chrome_launcher,
        "_close_local_scaffold_tabs",
        lambda: closed.setdefault("count", 1),
    )

    result = chrome_launcher.ensure_running(force=False)

    assert result == ChromeStatus(
        state="healthy",
        cdp_url="http://127.0.0.1:9222",
        profile_dir=str(chrome_launcher.chrome_profile_dir()),
        message="Chrome is open and ready.",
    )
    assert closed == {"count": 1}


def test_close_local_scaffold_tabs_only_closes_repo_scaffold_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stale-tab scrubber is deliberately narrow: close local scaffold
    file tabs, leave LinkedIn and ordinary web tabs alone.
    """

    opened_urls: list[str] = []
    targets = [
        {
            "id": "scaffold-1",
            "url": (
                "file:///Users/operator/Personal/cloris/"
                "cloris/frontend/scaffolds/09-workspace.html"
            ),
        },
        {"id": "linkedin", "url": "https://www.linkedin.com/talent/search"},
        {"id": "app", "url": "http://127.0.0.1:49449/"},
        {"id": "other-file", "url": "file:///Users/operator/Desktop/note.html"},
    ]

    class FakeResponse:
        status = 200

        def __init__(self, body: bytes = b"{}") -> None:
            self.body = body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return self.body

    def fake_urlopen(url: str, *, timeout: float) -> FakeResponse:
        opened_urls.append(url)
        if url.endswith("/json/list"):
            return FakeResponse(json.dumps(targets).encode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(chrome_launcher.urllib.request, "urlopen", fake_urlopen)

    assert chrome_launcher._close_local_scaffold_tabs() == 1
    assert opened_urls == [
        "http://127.0.0.1:9222/json/list",
        "http://127.0.0.1:9222/json/close/scaffold-1",
    ]
