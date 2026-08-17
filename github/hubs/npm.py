"""npm registry hub client.

Packuments carry maintainer email addresses. This client strips them — handles
only in return values, cache files, and logs.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from urllib.parse import quote

from github.hubs.base import BaseHubClient, normalize_repo_url

REGISTRY_BASE = "https://registry.npmjs.org"
DOWNLOADS_BASE = "https://api.npmjs.org"
USER_AGENT = "cloris-sourcing-agent/1.0 (npm hub client)"
MIN_REQUEST_INTERVAL_SECONDS = 0.8

CACHE_KINDS: frozenset[str] = frozenset(
    {
        "packument",
        "downloads_last_month",
    }
)

TTL_BY_KIND: dict[str, timedelta] = {
    "packument": timedelta(days=7),
    "downloads_last_month": timedelta(days=1),
}


def _extract_repo_url(raw: dict[str, Any]) -> str | None:
    repository = raw.get("repository")
    if isinstance(repository, str):
        return normalize_repo_url(repository)
    if isinstance(repository, dict):
        url = repository.get("url")
        if isinstance(url, str):
            text = url
            if text.startswith("git+"):
                text = text[4:]
            return normalize_repo_url(text)
    return None


def _extract_maintainer_handles(raw: dict[str, Any]) -> list[str]:
    handles: list[str] = []
    maintainers = raw.get("maintainers")
    if not isinstance(maintainers, list):
        return handles
    for entry in maintainers:
        if not isinstance(entry, dict):
            continue
        handle = entry.get("name")
        if isinstance(handle, str) and handle.strip() and "@" not in handle:
            handles.append(handle.strip())
    return handles


def _latest_version(raw: dict[str, Any]) -> str | None:
    dist_tags = raw.get("dist-tags")
    if isinstance(dist_tags, dict):
        latest = dist_tags.get("latest")
        if isinstance(latest, str) and latest.strip():
            return latest.strip()
    return None


def _is_deprecated(raw: dict[str, Any], latest: str | None) -> bool:
    if latest is None:
        return False
    versions = raw.get("versions")
    if not isinstance(versions, dict):
        return False
    version_payload = versions.get(latest)
    if not isinstance(version_payload, dict):
        return False
    deprecated = version_payload.get("deprecated")
    if deprecated is None:
        return False
    if isinstance(deprecated, bool):
        return deprecated
    if isinstance(deprecated, str):
        return bool(deprecated.strip())
    return False


def _extract_license(raw: dict[str, Any], latest: str | None) -> str | None:
    license_value = raw.get("license")
    if isinstance(license_value, str) and license_value.strip():
        return license_value.strip()
    if latest is not None:
        versions = raw.get("versions")
        if isinstance(versions, dict):
            version_payload = versions.get(latest)
            if isinstance(version_payload, dict):
                nested = version_payload.get("license")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return None


def normalize_packument(raw: dict[str, Any]) -> dict[str, Any]:
    name = str(raw.get("name") or "").strip()
    latest = _latest_version(raw)
    return {
        "name": name,
        "maintainer_handles": _extract_maintainer_handles(raw),
        "repository_url": _extract_repo_url(raw),
        "latest_version": latest,
        "deprecated": _is_deprecated(raw, latest),
        "license": _extract_license(raw, latest),
    }


class NpmHubClient(BaseHubClient):
    """Async client for the npm registry and downloads API."""

    hub = "npm"
    user_agent = USER_AGENT
    min_request_interval_seconds = MIN_REQUEST_INTERVAL_SECONDS
    cache_kinds = CACHE_KINDS
    ttl_by_kind = TTL_BY_KIND

    def _probe_url(self) -> str:
        return f"{REGISTRY_BASE}/-/ping"

    async def get_packument(self, package: str) -> dict[str, Any] | None:
        package_name = str(package or "").strip()
        if not package_name:
            return None

        cache_key = package_name
        cached = self._cache_get("packument", cache_key)
        if isinstance(cached, dict):
            return cached

        encoded = quote(package_name, safe="@/")
        url = f"{REGISTRY_BASE}/{encoded}"
        _status, body = await self._get_json(url)
        if not isinstance(body, dict):
            return None

        normalized = normalize_packument(body)
        if not normalized.get("name"):
            return None

        self._cache_put("packument", cache_key, normalized)
        return normalized

    async def get_downloads_last_month(self, package: str) -> int | None:
        package_name = str(package or "").strip()
        if not package_name:
            return None

        cache_key = package_name
        cached = self._cache_get("downloads_last_month", cache_key)
        if isinstance(cached, int):
            return cached

        encoded = quote(package_name, safe="@/")
        url = f"{DOWNLOADS_BASE}/downloads/point/last-month/{encoded}"
        _status, body = await self._get_json(url)
        if not isinstance(body, dict):
            return None

        downloads = body.get("downloads")
        if not isinstance(downloads, (int, float)) or isinstance(downloads, bool):
            return None

        count = int(downloads)
        self._cache_put("downloads_last_month", cache_key, count)
        return count
