"""Tests for :mod:`github.network_dependents` (Slice 3 — OSS Maintainers).

The HTTP fetch is impractical to unit-test without recording the page
HTML; we instead test the regex parser directly with realistic
fixtures lifted from the dependents page markup pattern. The parser
is the brittle surface (per spec §12) so this is where defense lives.
"""

from __future__ import annotations

from github import network_dependents as nd


def test_parse_extracts_repository_count() -> None:
    html = """
    <a class="btn-link selected" href="/owner/repo/network/dependents">
      <svg></svg>
      12,345
      <span>Repositories</span>
    </a>
    <a class="btn-link" href="/owner/repo/network/dependents?dependent_type=PACKAGE">
      <svg></svg>
      678
      <span>Packages</span>
    </a>
    """
    assert nd._parse_count(html) == 12345


def test_parse_falls_back_to_packages_when_repositories_absent() -> None:
    html = """
    <a class="btn-link selected">
      <svg></svg>
      890
      <span>Packages</span>
    </a>
    """
    assert nd._parse_count(html) == 890


def test_parse_handles_lowercase_label() -> None:
    """Resilience against minor markup changes — case-insensitive label match."""

    html = "<span>123,456 repositories</span>"
    assert nd._parse_count(html) == 123456


def test_parse_returns_none_for_empty_or_unrecognized() -> None:
    assert nd._parse_count("") is None
    assert nd._parse_count("<html><body>Nothing here</body></html>") is None


def test_parse_returns_none_for_non_numeric_segment() -> None:
    """Defense against DOM mutation that puts non-digits where the count was."""

    html = "<span>--- Repositories</span>"
    assert nd._parse_count(html) is None


def test_parse_extracts_large_count() -> None:
    """Heavily-depended-upon projects (millions of dependents) parse cleanly."""

    html = "<span>5,432,109 Repositories</span>"
    assert nd._parse_count(html) == 5432109


# ---------------------------------------------------------------------------
# Audit Move #22 — PyPI registry signals
# ---------------------------------------------------------------------------


import asyncio
import json

import pytest


def test_pypi_recent_parses_last_month() -> None:
    body = json.dumps(
        {
            "data": {"last_day": 100, "last_week": 700, "last_month": 12345},
            "package": "requests",
            "type": "recent_downloads",
        }
    )
    assert nd._parse_pypi_recent_downloads(body) == 12345


def test_pypi_recent_returns_none_for_malformed_json() -> None:
    assert nd._parse_pypi_recent_downloads("not-json") is None


def test_pypi_recent_returns_none_when_data_missing() -> None:
    body = json.dumps({"package": "requests", "type": "recent_downloads"})
    assert nd._parse_pypi_recent_downloads(body) is None


def test_pypi_recent_returns_none_when_last_month_not_int() -> None:
    body = json.dumps({"data": {"last_month": "12345"}})
    assert nd._parse_pypi_recent_downloads(body) is None


def test_fetch_pypi_recent_downloads_round_trips_with_stub(tmp_path: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: stub fetch_text + monkeypatch the cache root so the
    function round-trips a fixture response without touching network or
    disk in a shared location."""

    from github import maintainer_signal_cache as mcache

    monkeypatch.setattr(mcache, "CACHE_ROOT", tmp_path)

    captured_urls: list[str] = []

    async def _stub_fetch(url: str, *, headers: dict | None = None) -> str | None:
        captured_urls.append(url)
        return json.dumps({"data": {"last_month": 9876}})

    result = asyncio.run(
        nd.fetch_pypi_recent_downloads(
            "Requests",
            throttle_seconds=0.0,
            fetch_text=_stub_fetch,
        )
    )
    assert result == 9876
    assert len(captured_urls) == 1
    assert "/api/packages/requests/recent" in captured_urls[0]
    # Second call hits the cache (no second URL fetch).
    result2 = asyncio.run(
        nd.fetch_pypi_recent_downloads(
            "requests",
            throttle_seconds=0.0,
            fetch_text=_stub_fetch,
        )
    )
    assert result2 == 9876
    assert len(captured_urls) == 1


def test_fetch_pypi_recent_downloads_returns_none_on_fetch_failure(
    tmp_path: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    from github import maintainer_signal_cache as mcache

    monkeypatch.setattr(mcache, "CACHE_ROOT", tmp_path)

    async def _stub_fetch(url: str, *, headers: dict | None = None) -> str | None:
        return None  # simulating 404 / network error

    result = asyncio.run(
        nd.fetch_pypi_recent_downloads(
            "nonexistent-pkg-xyz",
            throttle_seconds=0.0,
            use_cache=False,
            fetch_text=_stub_fetch,
        )
    )
    assert result is None


def test_fetch_pypi_recent_downloads_returns_none_for_empty_package() -> None:
    result = asyncio.run(
        nd.fetch_pypi_recent_downloads(
            "",
            throttle_seconds=0.0,
            use_cache=False,
            fetch_text=lambda url, headers=None: _aiohttp_unused(),
        )
    )
    assert result is None


# ---------------------------------------------------------------------------
# Audit Move #22 — npm registry signals
# ---------------------------------------------------------------------------


def test_npm_downloads_parses_count() -> None:
    body = json.dumps(
        {
            "downloads": 4567890,
            "start": "2026-04-01",
            "end": "2026-04-30",
            "package": "react",
        }
    )
    assert nd._parse_npm_downloads(body) == 4567890


def test_npm_downloads_returns_none_on_error_payload() -> None:
    """The npm registry returns ``{"error": "package not found"}`` for
    unknown packages — must not be parsed as a successful 0 count."""

    body = json.dumps({"error": "package nonexistent not found"})
    assert nd._parse_npm_downloads(body) is None


def test_npm_downloads_returns_none_for_malformed_json() -> None:
    assert nd._parse_npm_downloads("not-json") is None


def test_fetch_npm_recent_downloads_url_encodes_scoped_package(
    tmp_path: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scoped packages (``@scope/name``) must url-encode the slash for
    the npm registry's downloads endpoint to resolve."""

    from github import maintainer_signal_cache as mcache

    monkeypatch.setattr(mcache, "CACHE_ROOT", tmp_path)

    captured_urls: list[str] = []

    async def _stub_fetch(url: str, *, headers: dict | None = None) -> str | None:
        captured_urls.append(url)
        return json.dumps({"downloads": 5432})

    result = asyncio.run(
        nd.fetch_npm_recent_downloads(
            "@scope/widget",
            throttle_seconds=0.0,
            fetch_text=_stub_fetch,
        )
    )
    assert result == 5432
    assert "%2F" in captured_urls[0]
    assert "@scope%2Fwidget" in captured_urls[0]


def test_fetch_npm_recent_downloads_caches_via_signal_cache(
    tmp_path: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    from github import maintainer_signal_cache as mcache

    monkeypatch.setattr(mcache, "CACHE_ROOT", tmp_path)

    fetch_calls = {"n": 0}

    async def _stub_fetch(url: str, *, headers: dict | None = None) -> str | None:
        fetch_calls["n"] += 1
        return json.dumps({"downloads": 100})

    asyncio.run(
        nd.fetch_npm_recent_downloads(
            "react",
            throttle_seconds=0.0,
            fetch_text=_stub_fetch,
        )
    )
    asyncio.run(
        nd.fetch_npm_recent_downloads(
            "react",
            throttle_seconds=0.0,
            fetch_text=_stub_fetch,
        )
    )
    assert fetch_calls["n"] == 1


# Helper used by the empty-package-name guard test above.
async def _aiohttp_unused() -> str | None:
    return None
