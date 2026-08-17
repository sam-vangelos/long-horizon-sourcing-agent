"""Tests for :mod:`github.hubs.npm` and :mod:`github.hubs.crates`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from github.hubs import crates, npm
from github.hubs.base import _cache_path
import github.hubs.base as hub_base

FIXTURES = Path(__file__).parent / "fixtures"


class _MockResponse:
    def __init__(self, status: int, payload: object) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> _MockResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _MockSession:
    def __init__(self, responses: list[_MockResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: object) -> _MockResponse:
        self.calls.append(url)
        if not self._responses:
            raise AssertionError(f"unexpected request: {url}")
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _isolated_npm_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_root = tmp_path / "npm_cache"
    monkeypatch.setattr(npm.NpmHubClient, "cache_root", property(lambda self: cache_root))
    return cache_root


@pytest.fixture
def isolated_crates_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_root = tmp_path / "crates_cache"
    monkeypatch.setattr(
        crates.CratesHubClient, "cache_root", property(lambda self: cache_root)
    )
    return cache_root


def _load_fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.asyncio
async def test_npm_get_packument_from_fixture() -> None:
    payload = _load_fixture("hub_npm_packument_with_emails.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with npm.NpmHubClient(session=session) as client:  # type: ignore[arg-type]
        packument = await client.get_packument("lodash")

    assert packument == {
        "name": "lodash",
        "maintainer_handles": ["jdalton", "mathias"],
        "repository_url": "https://github.com/lodash/lodash",
        "latest_version": "4.17.21",
        "deprecated": False,
        "license": "MIT",
    }
    assert "/lodash" in session.calls[0]


@pytest.mark.asyncio
async def test_npm_get_downloads_last_month_from_fixture() -> None:
    payload = _load_fixture("hub_npm_downloads.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with npm.NpmHubClient(session=session) as client:  # type: ignore[arg-type]
        count = await client.get_downloads_last_month("lodash")

    assert count == 658979149
    assert "/downloads/point/last-month/lodash" in session.calls[0]


@pytest.mark.asyncio
async def test_npm_malformed_payload_fails_soft() -> None:
    payload = _load_fixture("hub_npm_malformed.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with npm.NpmHubClient(session=session) as client:  # type: ignore[arg-type]
        packument = await client.get_packument("lodash")

    assert packument is None


@pytest.mark.asyncio
async def test_npm_non_200_fails_soft() -> None:
    session = _MockSession([_MockResponse(404, {})])

    async with npm.NpmHubClient(session=session) as client:  # type: ignore[arg-type]
        packument = await client.get_packument("lodash")

    assert packument is None


@pytest.mark.asyncio
async def test_npm_rate_policy_enforces_spacing() -> None:
    packument_payload = _load_fixture("hub_npm_packument_with_emails.json")
    downloads_payload = _load_fixture("hub_npm_downloads.json")
    session = _MockSession(
        [
            _MockResponse(200, packument_payload),
            _MockResponse(200, downloads_payload),
        ]
    )
    monotonic_values = iter([0.0, 0.1, 0.2, 1.0])
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    with patch.object(hub_base.time, "monotonic", side_effect=monotonic_values):
        with patch.object(hub_base.asyncio, "sleep", new=fake_sleep):
            async with npm.NpmHubClient(session=session) as client:  # type: ignore[arg-type]
                await client.get_packument("lodash")
                await client.get_downloads_last_month("lodash")

    assert len(sleep_calls) == 2
    assert sleep_calls[0] == pytest.approx(0.8)
    assert sleep_calls[1] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_npm_probe_returns_true_on_success() -> None:
    payload = _load_fixture("hub_npm_probe_ok.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with npm.NpmHubClient(session=session) as client:  # type: ignore[arg-type]
        assert await client.probe() is True


@pytest.mark.asyncio
async def test_npm_probe_returns_false_on_failure() -> None:
    session = _MockSession([_MockResponse(503, None)])

    async with npm.NpmHubClient(session=session) as client:  # type: ignore[arg-type]
        assert await client.probe() is False


@pytest.mark.asyncio
async def test_npm_payloads_carry_no_emails(
    _isolated_npm_cache: Path,
) -> None:
    payload = _load_fixture("hub_npm_packument_with_emails.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with npm.NpmHubClient(session=session) as client:  # type: ignore[arg-type]
        packument = await client.get_packument("lodash")

    assert packument is not None
    serialized = json.dumps(packument)
    assert "@" not in serialized

    cache_files = list(_isolated_npm_cache.rglob("*.json"))
    assert cache_files
    cache_text = "".join(path.read_text() for path in cache_files)
    assert "@" not in cache_text


@pytest.mark.asyncio
async def test_crates_get_crate_from_fixture(isolated_crates_cache: Path) -> None:
    payload = _load_fixture("hub_crates_crate.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with crates.CratesHubClient(session=session) as client:  # type: ignore[arg-type]
        crate = await client.get_crate("serde")

    assert crate == {
        "name": "serde",
        "total_downloads": 500000000,
        "recent_downloads": None,
        "recent_versions": [
            {"version": "1.0.210", "created_at": "2026-05-01T12:00:00Z"},
            {"version": "1.0.209", "created_at": "2026-04-01T12:00:00Z"},
        ],
        "repository_url": "https://github.com/serde-rs/serde",
        "description": "Serde serialization framework",
    }
    assert "/crates/serde" in session.calls[0]


@pytest.mark.asyncio
async def test_crates_client_returns_recent_downloads(isolated_crates_cache: Path) -> None:
    payload = {
        "crate": {
            "name": "tokio",
            "downloads": 900000000,
            "recent_downloads": 45000000,
            "description": "Async runtime",
            "repository": "https://github.com/tokio-rs/tokio",
        },
        "versions": [],
    }
    session = _MockSession([_MockResponse(200, payload)])

    async with crates.CratesHubClient(session=session) as client:  # type: ignore[arg-type]
        crate = await client.get_crate("tokio")

    assert crate is not None
    assert crate["total_downloads"] == 900000000
    assert crate["recent_downloads"] == 45000000
    assert crate["total_downloads"] != crate["recent_downloads"]


@pytest.mark.asyncio
async def test_crates_get_owner_users_from_fixture(isolated_crates_cache: Path) -> None:
    payload = _load_fixture("hub_crates_owners.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with crates.CratesHubClient(session=session) as client:  # type: ignore[arg-type]
        owners = await client.get_owner_users("serde")

    assert owners == [
        {"login": "dtolnay", "kind": "user", "github_login": "dtolnay"},
        {"login": "serde-team", "kind": "team"},
    ]


@pytest.mark.asyncio
async def test_crates_malformed_payload_fails_soft(isolated_crates_cache: Path) -> None:
    payload = _load_fixture("hub_crates_malformed.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with crates.CratesHubClient(session=session) as client:  # type: ignore[arg-type]
        crate = await client.get_crate("serde")

    assert crate is None


@pytest.mark.asyncio
async def test_crates_non_200_fails_soft(isolated_crates_cache: Path) -> None:
    session = _MockSession([_MockResponse(429, None)])

    async with crates.CratesHubClient(session=session) as client:  # type: ignore[arg-type]
        owners = await client.get_owner_users("serde")

    assert owners is None


@pytest.mark.asyncio
async def test_crates_rate_policy_enforces_spacing(isolated_crates_cache: Path) -> None:
    crate_payload = _load_fixture("hub_crates_crate.json")
    owners_payload = _load_fixture("hub_crates_owners.json")
    session = _MockSession(
        [
            _MockResponse(200, crate_payload),
            _MockResponse(200, owners_payload),
        ]
    )
    monotonic_values = iter([0.0, 0.1, 0.2, 1.2])
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    with patch.object(hub_base.time, "monotonic", side_effect=monotonic_values):
        with patch.object(hub_base.asyncio, "sleep", new=fake_sleep):
            async with crates.CratesHubClient(session=session) as client:  # type: ignore[arg-type]
                await client.get_crate("serde")
                await client.get_owner_users("serde")

    assert len(sleep_calls) == 2
    assert sleep_calls[0] == pytest.approx(1.0)
    assert sleep_calls[1] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_crates_probe_returns_true_on_success(isolated_crates_cache: Path) -> None:
    payload = _load_fixture("hub_crates_probe_ok.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with crates.CratesHubClient(session=session) as client:  # type: ignore[arg-type]
        assert await client.probe() is True


@pytest.mark.asyncio
async def test_crates_probe_returns_false_on_failure(isolated_crates_cache: Path) -> None:
    session = _MockSession([_MockResponse(503, None)])

    async with crates.CratesHubClient(session=session) as client:  # type: ignore[arg-type]
        assert await client.probe() is False


def test_cache_keys_do_not_collide(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    path_a = _cache_path(cache_root, "packument", "lodash")
    path_b = _cache_path(cache_root, "packument", "lodash-extra")
    assert path_a != path_b
