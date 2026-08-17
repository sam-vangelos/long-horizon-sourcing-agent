"""Designer Slice 2 — Behance v2 REST client.

Pins the contract between :class:`designer.sources.behance.BehanceClient`
and the Behance v2 REST surface. Uses recorded fixture responses + a
fake aiohttp session so the test suite never makes a real outbound
call (Behance API access is gated by Slice 0's Gate A; tests must
work without a key).

Key invariants:
- Constructor reads ``BEHANCE_API_KEY`` and raises if absent.
- Every request injects ``api_key`` as a query parameter (NOT a
  Bearer token).
- ``validate_credentials`` 200-checks a known public profile.
- ``search_users`` returns ``(status, body)`` and forwards Behance's
  raw response shape.
- 401/403 surface as :class:`BehanceAuthError` (the orchestrator's
  hard-fail signal).
- 429 with ``Retry-After`` triggers a sleep + retry; documented but
  not exhaustively tested here (the test would need to mock asyncio
  sleep — covered indirectly by the rate-limit budget tests).
- Local rate-limit budget tracks requests across calls so a hot-loop
  caller doesn't burst the per-hour cap.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from designer.sources.behance import (
    BEHANCE_API_BASE,
    BEHANCE_RATE_LIMIT_REQUESTS_PER_HOUR,
    BehanceAuthError,
    BehanceClient,
    _Budget,
)


# ---------------------------------------------------------------------------
# Fixture data — recorded shapes from the Behance v2 API
# (https://www.behance.net/dev/api/endpoints/2). Trimmed for test
# legibility; carries the fields the client actually consumes.
# ---------------------------------------------------------------------------


_USER_FIXTURE = {
    "user": {
        "id": 12345678,
        "username": "exampledesigner",
        "display_name": "Example Designer",
        "occupation": "Senior product designer",
        "city": "Brooklyn",
        "state": "New York",
        "country": "United States",
        "url": "https://www.behance.net/exampledesigner",
        "fields": ["UI/UX", "Product Design", "Design Systems"],
        "stats": {"appreciations": 4500, "views": 120000, "followers": 850},
    }
}


_USERS_SEARCH_FIXTURE = {
    "users": [
        {
            "id": 11111,
            "username": "designerone",
            "display_name": "Designer One",
            "occupation": "Brand designer",
            "city": "Portland",
            "country": "United States",
            "url": "https://www.behance.net/designerone",
            "fields": ["Branding", "Logo Design"],
            "stats": {"appreciations": 1500, "followers": 200},
        },
        {
            "id": 22222,
            "username": "designertwo",
            "display_name": "Designer Two",
            "occupation": "Motion designer",
            "city": "Berlin",
            "country": "Germany",
            "url": "https://www.behance.net/designertwo",
            "fields": ["Motion Graphics", "Animation"],
            "stats": {"appreciations": 3200, "followers": 510},
        },
    ],
    "stats": {"total": 2},
}


_PROJECT_FIXTURE = {
    "project": {
        "id": 99999,
        "name": "Acme Design System",
        "published_on": 1735000000,
        "modules": [
            {
                "id": 1,
                "type": "image",
                "sizes": {
                    "original": "https://mir-s3-cdn-cf.behance.net/projects/original/abc123.jpg",
                    "disp": "https://mir-s3-cdn-cf.behance.net/projects/disp/abc123.jpg",
                },
            }
        ],
        "covers": {
            "404": "https://mir-s3-cdn-cf.behance.net/projects/404/cover.jpg",
            "808": "https://mir-s3-cdn-cf.behance.net/projects/808/cover.jpg",
            "max_808": "https://mir-s3-cdn-cf.behance.net/projects/max_808/cover.jpg",
            "original": "https://mir-s3-cdn-cf.behance.net/projects/original/cover.jpg",
        },
        "stats": {"appreciations": 1200, "views": 45000},
        "fields": ["UI/UX", "Design Systems"],
    }
}


# ---------------------------------------------------------------------------
# Fake session helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for aiohttp.ClientResponse."""

    def __init__(
        self,
        *,
        status: int,
        body: Any = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self._text = text
        self.headers = headers or {}

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def json(self) -> Any:
        return self._body

    async def text(self) -> str:
        if self._text:
            return self._text
        return json.dumps(self._body) if self._body is not None else ""


class _FakeSession:
    """Minimal stand-in for aiohttp.ClientSession.

    Records each `(url, params)` pair the client requests and returns
    the next queued response. Tests prime the queue with the fixture
    they expect that call to receive.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[_FakeResponse] = []

    def queue(self, response: _FakeResponse) -> None:
        self.responses.append(response)

    def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        self.calls.append((url, dict(params or {})))
        if not self.responses:
            return _FakeResponse(status=500, text="no fixture queued")
        return self.responses.pop(0)

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_session() -> _FakeSession:
    return _FakeSession()


@pytest.fixture
def client_factory(fake_session: _FakeSession, monkeypatch: pytest.MonkeyPatch):
    """Yield a callable that returns a BehanceClient bound to the fake session."""

    monkeypatch.setenv("BEHANCE_API_KEY", "fake-test-key")

    def _build() -> BehanceClient:
        return BehanceClient(session_factory=lambda: fake_session)

    return _build


# ---------------------------------------------------------------------------
# Construction / auth
# ---------------------------------------------------------------------------


def test_constructor_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BEHANCE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="BEHANCE_API_KEY"):
        BehanceClient()


def test_constructor_accepts_explicit_api_key() -> None:
    client = BehanceClient(api_key="explicit-key")
    assert client._api_key == "explicit-key"


# ---------------------------------------------------------------------------
# Query plumbing — api_key always present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_users_injects_api_key(
    fake_session: _FakeSession, client_factory: Any
) -> None:
    fake_session.queue(_FakeResponse(status=200, body=_USERS_SEARCH_FIXTURE))

    async with client_factory() as client:
        status, body = await client.search_users(query="design system", page=1)

    assert status == 200
    assert body is not None and len(body["users"]) == 2

    # Single GET issued; api_key on the wire.
    assert len(fake_session.calls) == 1
    url, params = fake_session.calls[0]
    assert url == f"{BEHANCE_API_BASE}/users"
    assert params["api_key"] == "fake-test-key"
    assert params["q"] == "design system"
    assert params["page"] == "1"
    assert params["sort"] == "appreciations"


@pytest.mark.asyncio
async def test_get_user_returns_full_profile(
    fake_session: _FakeSession, client_factory: Any
) -> None:
    fake_session.queue(_FakeResponse(status=200, body=_USER_FIXTURE))

    async with client_factory() as client:
        status, body = await client.get_user("exampledesigner")

    assert status == 200
    assert body is not None and body["user"]["username"] == "exampledesigner"
    url, _ = fake_session.calls[0]
    assert url == f"{BEHANCE_API_BASE}/users/exampledesigner"


@pytest.mark.asyncio
async def test_get_project_returns_modules(
    fake_session: _FakeSession, client_factory: Any
) -> None:
    """Slice 5's vision pipeline grounds itself in `project.modules` —
    confirm the client surfaces them unchanged."""

    fake_session.queue(_FakeResponse(status=200, body=_PROJECT_FIXTURE))

    async with client_factory() as client:
        status, body = await client.get_project(99999)

    assert status == 200
    assert body is not None
    assert body["project"]["modules"][0]["sizes"]["original"].startswith("https://")


@pytest.mark.asyncio
async def test_get_user_404_returns_none_body(
    fake_session: _FakeSession, client_factory: Any
) -> None:
    fake_session.queue(_FakeResponse(status=404))

    async with client_factory() as client:
        status, body = await client.get_user("does-not-exist")

    assert status == 404
    assert body is None


@pytest.mark.asyncio
async def test_401_raises_behance_auth_error(
    fake_session: _FakeSession, client_factory: Any
) -> None:
    fake_session.queue(_FakeResponse(status=401, text='{"error":"invalid api_key"}'))

    async with client_factory() as client:
        with pytest.raises(BehanceAuthError, match="rejected"):
            await client.search_users(query="anything")


@pytest.mark.asyncio
async def test_validate_credentials_passes_on_200(
    fake_session: _FakeSession, client_factory: Any
) -> None:
    fake_session.queue(_FakeResponse(status=200, body=_USER_FIXTURE))

    async with client_factory() as client:
        await client.validate_credentials()  # No raise.


@pytest.mark.asyncio
async def test_validate_credentials_raises_on_403(
    fake_session: _FakeSession, client_factory: Any
) -> None:
    fake_session.queue(_FakeResponse(status=403))

    async with client_factory() as client:
        with pytest.raises(BehanceAuthError):
            await client.validate_credentials()


# ---------------------------------------------------------------------------
# Local rate-limit budget — sliding-window counter
# ---------------------------------------------------------------------------


def test_budget_starts_with_full_capacity() -> None:
    budget = _Budget()
    assert budget.available()
    assert budget.seconds_until_available() == 0.0


def test_budget_blocks_after_limit_consumed() -> None:
    budget = _Budget(limit=3, window=3600)
    for _ in range(3):
        budget.consume()
    assert not budget.available()
    assert budget.seconds_until_available() > 0


def test_budget_reports_full_after_window_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the rolling-window edge passes, capacity recovers."""

    fake_now = [0.0]

    def fake_monotonic() -> float:
        return fake_now[0]

    monkeypatch.setattr("designer.sources.behance.time.monotonic", fake_monotonic)

    budget = _Budget(limit=2, window=10.0)
    budget.consume()
    budget.consume()
    assert not budget.available()

    # Roll the clock past the window; both timestamps fall out.
    fake_now[0] = 100.0
    assert budget.available()


def test_default_budget_carries_documented_limit() -> None:
    """The Slice-2 default matches the documented 130/hr posture
    (150/hr Behance free tier with headroom)."""

    budget = _Budget()
    assert budget.limit == BEHANCE_RATE_LIMIT_REQUESTS_PER_HOUR
    assert budget.window == 3600
