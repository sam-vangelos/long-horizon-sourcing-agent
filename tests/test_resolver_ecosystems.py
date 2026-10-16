"""Tests for :mod:`shared.resolvers.ecosystems`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import shared.resolvers.ecosystems as ecosystems


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
def _isolated_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(ecosystems, "CACHE_ROOT", cache_root)
    return cache_root


def _load_fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.asyncio
async def test_resolve_repo_packages_round_trip_from_fixture() -> None:
    payload = _load_fixture("ecosystems_lodash_packages_lookup.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        packages = await resolver.resolve_repo_packages("https://github.com/lodash/lodash")

    assert len(packages) == 2
    assert packages[0] == {
        "registry": "npm",
        "name": "lodash",
        "latest_release": "4.18.1",
        "downloads": 658979149,
    }
    assert packages[1]["name"] == "lodash.partialright"
    assert "packages/lookup?repository_url=https%3A%2F%2Fgithub.com%2Flodash%2Flodash" in session.calls[0]


@pytest.mark.asyncio
async def test_reverse_dependency_count_from_fixture() -> None:
    payload = _load_fixture("ecosystems_lodash_package_detail.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        count = await resolver.reverse_dependency_count("npm", "lodash")

    assert count == 159122
    assert "/registries/npmjs.org/packages/lodash" in session.calls[0]


@pytest.mark.asyncio
async def test_malformed_payload_fails_soft() -> None:
    payload = _load_fixture("ecosystems_malformed_packages_lookup.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        packages = await resolver.resolve_repo_packages("https://github.com/lodash/lodash")

    assert packages == []


@pytest.mark.asyncio
async def test_non_200_fails_soft() -> None:
    payload = _load_fixture("ecosystems_non_200.json")
    session = _MockSession([_MockResponse(404, payload)])

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        packages = await resolver.resolve_repo_packages("https://github.com/lodash/lodash")

    assert packages == []


@pytest.mark.asyncio
async def test_rate_policy_enforces_spacing() -> None:
    payload = _load_fixture("ecosystems_lodash_packages_lookup.json")
    session = _MockSession(
        [
            _MockResponse(200, payload),
            _MockResponse(200, _load_fixture("ecosystems_lodash_package_detail.json")),
        ]
    )
    monotonic_values = iter([0.0, 0.1, 0.2, 1.0])
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    with patch.object(ecosystems.time, "monotonic", side_effect=monotonic_values):
        with patch.object(ecosystems.asyncio, "sleep", new=fake_sleep):
            async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
                await resolver.resolve_repo_packages("https://github.com/lodash/lodash")
                await resolver.reverse_dependency_count("npm", "lodash")

    assert len(sleep_calls) == 2
    assert sleep_calls[0] == pytest.approx(0.8)
    assert sleep_calls[1] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_repo_packages_uses_disk_cache(tmp_path: Path) -> None:
    payload = _load_fixture("ecosystems_lodash_packages_lookup.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        first = await resolver.resolve_repo_packages("https://github.com/lodash/lodash")
        second = await resolver.resolve_repo_packages("https://github.com/lodash/lodash")

    assert first == second
    assert len(session.calls) == 1


def _cache_files(cache_root: Path) -> list[Path]:
    return list(cache_root.rglob("*.json"))


@pytest.mark.asyncio
async def test_failed_fetch_is_not_cached(_isolated_cache_root: Path) -> None:
    success_payload = _load_fixture("ecosystems_lodash_packages_lookup.json")
    session = _MockSession(
        [
            _MockResponse(429, None),
            _MockResponse(200, success_payload),
        ]
    )

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        first = await resolver.resolve_repo_packages("https://github.com/lodash/lodash")

    assert first == []
    assert _cache_files(_isolated_cache_root) == []

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        second = await resolver.resolve_repo_packages("https://github.com/lodash/lodash")

    assert len(second) == 2
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_partial_pagination_is_not_cached(_isolated_cache_root: Path) -> None:
    page1 = [{"registry": "npm", "name": f"pkg-{index}"} for index in range(100)]
    session = _MockSession(
        [
            _MockResponse(200, page1),
            _MockResponse(429, None),
        ]
    )

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        packages = await resolver.resolve_repo_packages("https://github.com/lodash/lodash")

    assert len(packages) == 100
    assert _cache_files(_isolated_cache_root) == []


@pytest.mark.asyncio
async def test_clean_empty_result_is_cached(_isolated_cache_root: Path) -> None:
    session = _MockSession(
        [
            _MockResponse(200, []),
            _MockResponse(200, [{"registry": "npm", "name": "should-not-fetch"}]),
        ]
    )

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        first = await resolver.resolve_repo_packages("https://github.com/empty/repo")
        second = await resolver.resolve_repo_packages("https://github.com/empty/repo")

    assert first == []
    assert second == []
    assert len(_cache_files(_isolated_cache_root)) == 1
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_cache_keys_do_not_collide(_isolated_cache_root: Path) -> None:
    payload_a = [{"registry": "npm", "name": "from-a_b"}]
    payload_b = [{"registry": "npm", "name": "from-b_c"}]
    session = _MockSession(
        [
            _MockResponse(200, payload_a),
            _MockResponse(200, payload_b),
        ]
    )

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        packages_a = await resolver.resolve_repo_packages("https://github.com/a_b/c")
        packages_b = await resolver.resolve_repo_packages("https://github.com/a/b_c")

    assert packages_a == [{"registry": "npm", "name": "from-a_b"}]
    assert packages_b == [{"registry": "npm", "name": "from-b_c"}]
    cache_files = _cache_files(_isolated_cache_root)
    assert len(cache_files) == 2
    assert cache_files[0] != cache_files[1]
    path_a = ecosystems._cache_path(
        "repo_packages",
        "https://github.com/a_b/c",
    )
    path_b = ecosystems._cache_path(
        "repo_packages",
        "https://github.com/a/b_c",
    )
    assert path_a != path_b


@pytest.mark.asyncio
async def test_lookup_package_round_trip_from_fixture() -> None:
    payload = _load_fixture("ecosystems_lookup_lodash_package.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        package = await resolver.lookup_package("npm", "lodash")

    assert package == {
        "registry": "npm",
        "name": "lodash",
        "repository_url": "https://github.com/lodash/lodash",
        "latest_release": "4.18.1",
        "downloads": 658979149,
        "downloads_period": "last month",
        "dependent_packages_count": 159122,
    }
    assert "/registries/npmjs.org/packages/lodash" in session.calls[0]


@pytest.mark.asyncio
async def test_lookup_package_404_returns_missing_sentinel_and_caches(
    _isolated_cache_root: Path,
) -> None:
    session = _MockSession(
        [
            _MockResponse(404, None),
            _MockResponse(200, {"name": "should-not-fetch"}),
        ]
    )

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        first = await resolver.lookup_package("npm", "no-such-package")
        second = await resolver.lookup_package("npm", "no-such-package")

    assert first == {"missing": True}
    assert second == {"missing": True}
    assert len(session.calls) == 1
    cache_files = _cache_files(_isolated_cache_root)
    assert len(cache_files) == 1
    cached_payload = json.loads(cache_files[0].read_text())
    assert cached_payload["data"] == {"missing": True}


@pytest.mark.asyncio
async def test_lookup_package_network_error_not_cached(_isolated_cache_root: Path) -> None:
    import aiohttp

    payload = _load_fixture("ecosystems_lookup_lodash_package.json")
    inner = _MockSession([_MockResponse(200, payload)])

    class _FlakySession:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self._attempt = 0

        def get(self, url: str, **kwargs: object) -> _MockResponse:
            self.calls.append(url)
            self._attempt += 1
            if self._attempt == 1:
                raise aiohttp.ClientError("connection reset")
            return inner.get(url, **kwargs)

    session = _FlakySession()

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        first = await resolver.lookup_package("npm", "lodash")

    assert first is None
    assert _cache_files(_isolated_cache_root) == []

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        second = await resolver.lookup_package("npm", "lodash")

    assert second is not None
    assert second["name"] == "lodash"
    assert len(_cache_files(_isolated_cache_root)) == 1
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_lookup_no_fuzzy_endpoints_used() -> None:
    payload = _load_fixture("ecosystems_lookup_lodash_package.json")
    session = _MockSession([_MockResponse(200, payload)])

    async with ecosystems.EcosystemsResolver(session=session) as resolver:  # type: ignore[arg-type]
        await resolver.lookup_package("npm", "lodash")

    assert len(session.calls) == 1
    url = session.calls[0]
    assert url.endswith("/registries/npmjs.org/packages/lodash")
    assert "/search" not in url
    assert "/packages/lookup" not in url
