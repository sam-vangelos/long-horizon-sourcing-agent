"""GitHubClient._request honors Retry-After (CLO-437).

GitHub sends Retry-After as delay-seconds or as an HTTP-date. The client must
wait what the server asked for on a 403 rate limit and on a 429, survive the
HTTP-date form, and never wait longer than the ceiling. No network: a fake
session replays canned responses and asyncio.sleep records the waits.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import github.client as gh_client
from github.client import GitHubClient


class _FakeResponse:
    def __init__(self, status: int, headers: dict | None = None, text: str = "", body=None):
        self.status = status
        self.headers = dict(headers or {})
        self._text = text
        self._body = body

    async def json(self):
        return self._body

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls = 0

    def request(self, method, url, params=None, headers=None):
        self.calls += 1
        return self._responses.pop(0)


class _FakeLimiter:
    async def acquire(self, endpoint_class="rest"):
        return 0.0

    def update_from_headers(self, endpoint_class, headers):
        return None


def _run(responses: list[_FakeResponse], monkeypatch) -> tuple[tuple, list[float], _FakeSession]:
    waits: list[float] = []

    async def _fake_sleep(seconds, *args, **kwargs):
        waits.append(float(seconds))

    monkeypatch.setattr(gh_client.asyncio, "sleep", _fake_sleep)
    client = GitHubClient(token="dummy")
    session = _FakeSession(responses)
    client._session = session
    client._limiter = _FakeLimiter()
    result = asyncio.run(client._request("GET", "/users/octocat"))
    return result, waits, session


def _ok() -> _FakeResponse:
    return _FakeResponse(200, body={"login": "octocat"})


def _rate_limited_403(retry_after: str | None) -> _FakeResponse:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return _FakeResponse(403, headers=headers, text="API rate limit exceeded for user")


def test_403_rate_limit_http_date_retry_after_waits_until_that_date(monkeypatch):
    target = datetime.now(timezone.utc) + timedelta(seconds=30)
    http_date = format_datetime(target, usegmt=True)

    (status, body, _headers), waits, session = _run(
        [_rate_limited_403(http_date), _ok()], monkeypatch
    )

    assert status == 200
    assert body == {"login": "octocat"}
    assert session.calls == 2
    assert len(waits) == 1
    # HTTP-date has one-second resolution; allow for it and for test latency.
    assert 27.0 <= waits[0] <= 31.0


def test_403_rate_limit_delay_seconds_retry_after_waits_that_long(monkeypatch):
    (status, _body, _headers), waits, _session = _run(
        [_rate_limited_403("7"), _ok()], monkeypatch
    )

    assert status == 200
    assert waits == [7.0]


def test_429_retry_after_seconds_is_honored_not_exponential_backoff(monkeypatch):
    (status, _body, _headers), waits, _session = _run(
        [_FakeResponse(429, headers={"Retry-After": "12"}), _ok()], monkeypatch
    )

    assert status == 200
    assert waits == [12.0]


def test_retry_after_is_clamped_to_the_ceiling(monkeypatch):
    (status, _body, _headers), waits, _session = _run(
        [_rate_limited_403("100000"), _ok()], monkeypatch
    )

    assert status == 200
    assert waits == [300.0]


def test_429_retry_after_is_clamped_to_the_ceiling(monkeypatch):
    (status, _body, _headers), waits, _session = _run(
        [_FakeResponse(429, headers={"Retry-After": "100000"}), _ok()], monkeypatch
    )

    assert status == 200
    assert waits == [300.0]


def test_403_rate_limit_without_retry_after_waits_sixty_seconds(monkeypatch):
    (status, _body, _headers), waits, _session = _run(
        [_rate_limited_403(None), _ok()], monkeypatch
    )

    assert status == 200
    assert waits == [60.0]


def test_403_rate_limit_unparseable_retry_after_falls_back_to_sixty(monkeypatch):
    (status, _body, _headers), waits, _session = _run(
        [_rate_limited_403("soon-ish"), _ok()], monkeypatch
    )

    assert status == 200
    assert waits == [60.0]


def test_429_without_retry_after_keeps_exponential_backoff(monkeypatch):
    monkeypatch.setattr(gh_client, "human_delay", lambda lo, hi: 0.0)

    (status, _body, _headers), waits, _session = _run(
        [_FakeResponse(429), _FakeResponse(429), _ok()], monkeypatch
    )

    assert status == 200
    assert waits == [1.0, 2.0]

