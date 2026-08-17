"""crates.io registry hub client.

Crawler policy (https://crates.io/data-access): at most one request per second
and a named User-Agent identifying the client.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from urllib.parse import quote

from github.hubs.base import BaseHubClient, normalize_repo_url

API_BASE = "https://crates.io/api/v1"
USER_AGENT = "cloris-sourcing-agent/1.0 (crates hub client)"
MIN_REQUEST_INTERVAL_SECONDS = 1.0

CACHE_KINDS: frozenset[str] = frozenset(
    {
        "crate",
        "owner_users",
    }
)

TTL_BY_KIND: dict[str, timedelta] = {
    "crate": timedelta(days=7),
    "owner_users": timedelta(days=7),
}


def _recent_versions(raw: dict[str, Any], *, limit: int = 5) -> list[dict[str, str]]:
    versions = raw.get("versions")
    if not isinstance(versions, list):
        return []
    recent: list[dict[str, str]] = []
    for entry in versions[:limit]:
        if not isinstance(entry, dict):
            continue
        num = entry.get("num")
        created_at = entry.get("created_at")
        if isinstance(num, str) and num.strip() and isinstance(created_at, str):
            recent.append({"version": num.strip(), "created_at": created_at.strip()})
    return recent


def normalize_crate(raw: dict[str, Any]) -> dict[str, Any] | None:
    crate = raw.get("crate")
    if not isinstance(crate, dict):
        return None
    name = str(crate.get("name") or "").strip()
    if not name:
        return None
    total_downloads_raw = crate.get("downloads")
    total_download_count: int | None = None
    if isinstance(total_downloads_raw, (int, float)) and not isinstance(
        total_downloads_raw, bool
    ):
        total_download_count = int(total_downloads_raw)
    recent_downloads_raw = crate.get("recent_downloads")
    recent_download_count: int | None = None
    if isinstance(recent_downloads_raw, (int, float)) and not isinstance(
        recent_downloads_raw, bool
    ):
        recent_download_count = int(recent_downloads_raw)
    repository = crate.get("repository")
    repository_url = (
        normalize_repo_url(repository) if isinstance(repository, str) else None
    )
    description = crate.get("description")
    return {
        "name": name,
        "total_downloads": total_download_count,
        "recent_downloads": recent_download_count,
        "recent_versions": _recent_versions(raw),
        "repository_url": repository_url,
        "description": str(description).strip() if isinstance(description, str) else None,
    }


def normalize_owner_users(raw: dict[str, Any]) -> list[dict[str, str]] | None:
    users = raw.get("users")
    if not isinstance(users, list):
        return None
    owners: list[dict[str, str]] = []
    for entry in users:
        if not isinstance(entry, dict):
            continue
        login = entry.get("login")
        kind = entry.get("kind")
        if not isinstance(login, str) or not login.strip():
            continue
        if not isinstance(kind, str) or not kind.strip():
            continue
        login = login.strip()
        if "@" in login:
            continue
        owner: dict[str, str] = {
            "login": login,
            "kind": kind.strip(),
        }
        # crates.io authenticates users through GitHub OAuth; for user owners
        # the crates.io login is the GitHub login.
        if kind.strip() == "user":
            owner["github_login"] = login
        owners.append(owner)
    return owners


class CratesHubClient(BaseHubClient):
    """Async client for the crates.io registry API."""

    hub = "crates"
    user_agent = USER_AGENT
    min_request_interval_seconds = MIN_REQUEST_INTERVAL_SECONDS
    cache_kinds = CACHE_KINDS
    ttl_by_kind = TTL_BY_KIND

    def _probe_url(self) -> str:
        return f"{API_BASE}/summary"

    async def get_crate(self, crate: str) -> dict[str, Any] | None:
        crate_name = str(crate or "").strip()
        if not crate_name:
            return None

        cache_key = crate_name
        cached = self._cache_get("crate", cache_key)
        if isinstance(cached, dict):
            return cached

        encoded = quote(crate_name, safe="")
        url = f"{API_BASE}/crates/{encoded}"
        _status, body = await self._get_json(url)
        if not isinstance(body, dict):
            return None

        normalized = normalize_crate(body)
        if normalized is None:
            return None

        self._cache_put("crate", cache_key, normalized)
        return normalized

    async def get_owner_users(self, crate: str) -> list[dict[str, str]] | None:
        crate_name = str(crate or "").strip()
        if not crate_name:
            return None

        cache_key = crate_name
        cached = self._cache_get("owner_users", cache_key)
        if isinstance(cached, list):
            return cached

        encoded = quote(crate_name, safe="")
        url = f"{API_BASE}/crates/{encoded}/owners"
        _status, body = await self._get_json(url)
        if not isinstance(body, dict):
            return None

        normalized = normalize_owner_users(body)
        if normalized is None:
            return None

        self._cache_put("owner_users", cache_key, normalized)
        return normalized
