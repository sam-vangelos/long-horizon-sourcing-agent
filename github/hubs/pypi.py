"""PyPI registry hub client.

Project metadata carries author and maintainer email addresses. This client
strips them — only the normalized shape in return values, cache files, and logs.
PyPI declares no maintainer handles; discovery composes elsewhere.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from urllib.parse import quote

from github.hubs.base import BaseHubClient, normalize_repo_url

API_BASE = "https://pypi.org/pypi"
STATS_BASE = "https://pypistats.org/api"
USER_AGENT = "cloris-sourcing-agent/1.0 (pypi hub client)"
MIN_REQUEST_INTERVAL_SECONDS = 1.0

CACHE_KINDS: frozenset[str] = frozenset(
    {
        "project",
        "downloads_last_month",
    }
)

TTL_BY_KIND: dict[str, timedelta] = {
    "project": timedelta(days=7),
    "downloads_last_month": timedelta(days=1),
}


def _github_repo_candidates(info: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    project_urls = info.get("project_urls")
    if isinstance(project_urls, dict):
        for value in project_urls.values():
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    home_page = info.get("home_page")
    if isinstance(home_page, str) and home_page.strip():
        candidates.append(home_page.strip())
    return candidates


def _extract_repo_url(info: dict[str, Any]) -> str | None:
    for candidate in _github_repo_candidates(info):
        normalized = normalize_repo_url(candidate)
        if normalized is not None:
            return normalized
    return None


def _is_yanked(raw: dict[str, Any], latest: str | None) -> bool:
    info = raw.get("info")
    if isinstance(info, dict):
        yanked = info.get("yanked")
        if isinstance(yanked, bool):
            return yanked
    if latest is None:
        return False
    releases = raw.get("releases")
    if not isinstance(releases, dict):
        return False
    version_files = releases.get(latest)
    if not isinstance(version_files, list):
        return False
    for entry in version_files:
        if isinstance(entry, dict) and entry.get("yanked") is True:
            return True
    return False


def normalize_project(raw: dict[str, Any]) -> dict[str, Any] | None:
    info = raw.get("info")
    if not isinstance(info, dict):
        return None
    name = str(info.get("name") or "").strip()
    if not name:
        return None
    version = info.get("version")
    latest = version.strip() if isinstance(version, str) and version.strip() else None
    summary = info.get("summary")
    license_value = info.get("license")
    return {
        "name": name,
        "latest_version": latest,
        "summary": str(summary).strip() if isinstance(summary, str) else None,
        "license": (
            str(license_value).strip() if isinstance(license_value, str) else None
        ),
        "repository_url": _extract_repo_url(info),
        "yanked": _is_yanked(raw, latest),
    }


class PyPiHubClient(BaseHubClient):
    """Async client for the PyPI JSON API and pypistats downloads API."""

    hub = "pypi"
    user_agent = USER_AGENT
    min_request_interval_seconds = MIN_REQUEST_INTERVAL_SECONDS
    cache_kinds = CACHE_KINDS
    ttl_by_kind = TTL_BY_KIND

    def _probe_url(self) -> str:
        # ``six`` is a tiny, long-stable project (~few KB JSON) — cheap liveness
        # check against the discovery substrate (pypi.org), not pypistats.
        return f"{API_BASE}/six/json"

    async def get_project(self, package: str) -> dict[str, Any] | None:
        package_name = str(package or "").strip()
        if not package_name:
            return None

        cache_key = package_name
        cached = self._cache_get("project", cache_key)
        if isinstance(cached, dict):
            return cached

        encoded = quote(package_name, safe="")
        url = f"{API_BASE}/{encoded}/json"
        _status, body = await self._get_json(url)
        if not isinstance(body, dict):
            return None

        normalized = normalize_project(body)
        if normalized is None or not normalized.get("name"):
            return None

        self._cache_put("project", cache_key, normalized)
        return normalized

    async def get_downloads_last_month(self, package: str) -> int | None:
        package_name = str(package or "").strip()
        if not package_name:
            return None

        cache_key = package_name
        cached = self._cache_get("downloads_last_month", cache_key)
        if isinstance(cached, int):
            return cached

        encoded = quote(package_name, safe="")
        url = f"{STATS_BASE}/packages/{encoded}/recent"
        _status, body = await self._get_json(url)
        if not isinstance(body, dict):
            return None

        data = body.get("data")
        if not isinstance(data, dict):
            return None
        last_month = data.get("last_month")
        if not isinstance(last_month, (int, float)) or isinstance(last_month, bool):
            return None

        count = int(last_month)
        self._cache_put("downloads_last_month", cache_key, count)
        return count
