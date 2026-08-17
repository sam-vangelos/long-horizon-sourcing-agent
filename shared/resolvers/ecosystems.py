"""ecosyste.ms resolver client for repo→package mapping and reverse-dependency counts.

Fail-soft on every path: network errors, non-200 responses, malformed JSON, and
missing fields return ``[]`` / ``None`` with a single log line — never an exception
to callers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import aiohttp
import certifi

from shared.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

API_BASE = "https://packages.ecosyste.ms/api/v1"
USER_AGENT = "cloris-sourcing-agent/1.0 (ecosyste.ms resolver)"
MIN_REQUEST_INTERVAL_SECONDS = 0.8

CACHE_KINDS: frozenset[str] = frozenset(
    {
        "repo_packages",
        "reverse_dependency_count",
    }
)

TTL_BY_KIND: dict[str, timedelta] = {
    "repo_packages": timedelta(days=7),
    "reverse_dependency_count": timedelta(days=7),
}

CACHE_ROOT: Path = OUTPUT_DIR / "cache" / "shared" / "ecosystems_resolver"

REGISTRY_ALIASES: dict[str, str] = {
    "npm": "npmjs.org",
    "npmjs": "npmjs.org",
    "npmjs.org": "npmjs.org",
    "pypi": "pypi.org",
    "pypi.org": "pypi.org",
    "crates.io": "crates.io",
    "cargo": "crates.io",
    "rust": "crates.io",
    "nuget": "nuget.org",
    "nuget.org": "nuget.org",
    "rubygems": "rubygems.org",
    "rubygems.org": "rubygems.org",
    "go": "proxy.golang.org",
    "proxy.golang.org": "proxy.golang.org",
}


@dataclass(frozen=True)
class CacheEntry:
    fetched_at: datetime
    cache_kind: str
    cache_key: str
    data: Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_segment(value: str) -> str:
    return value.strip().lower().replace("/", "_").replace(":", "_") or "_unknown"


def _cache_path(cache_kind: str, cache_key: str) -> Path:
    safe_kind = cache_kind.strip()
    prefix = _safe_segment(cache_key)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    return CACHE_ROOT / safe_kind / f"{prefix}_{digest}.json"


def _cache_get(cache_kind: str, cache_key: str) -> Any | None:
    if cache_kind not in CACHE_KINDS:
        logger.warning("ecosystems_resolver cache get: unknown cache_kind=%r", cache_kind)
        return None
    path = _cache_path(cache_kind, cache_key)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "ecosystems_resolver cache get: corrupt cache at %s (%s); treating as miss",
            path,
            exc,
        )
        return None
    fetched_at_str = raw.get("fetched_at")
    if not isinstance(fetched_at_str, str):
        return None
    try:
        fetched_at = datetime.fromisoformat(fetched_at_str)
    except ValueError:
        return None
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    ttl = TTL_BY_KIND.get(cache_kind)
    if ttl is None or _now() - fetched_at > ttl:
        return None
    stored_key = raw.get("cache_key")
    if stored_key != cache_key:
        return None
    return raw.get("data")


def _cache_put(cache_kind: str, cache_key: str, data: Any) -> None:
    if cache_kind not in CACHE_KINDS:
        logger.warning("ecosystems_resolver cache put: unknown cache_kind=%r", cache_kind)
        return
    path = _cache_path(cache_kind, cache_key)
    payload = {
        "fetched_at": _now().isoformat(),
        "cache_kind": cache_kind,
        "cache_key": cache_key,
        "data": data,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError as exc:
        logger.warning(
            "ecosystems_resolver cache put: write failed at %s (%s); proceeding without cache",
            path,
            exc,
        )


def _normalize_repo_url(repo_url: str) -> str | None:
    text = str(repo_url or "").strip()
    if not text:
        return None
    parsed = urlparse(text)
    if not parsed.scheme:
        parsed = urlparse(f"https://{text}")
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return None
    parts = [segment for segment in parsed.path.strip("/").split("/") if segment]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return f"https://github.com/{owner}/{repo}"


def _normalize_registry(registry: str) -> str | None:
    key = str(registry or "").strip().lower()
    if not key:
        return None
    return REGISTRY_ALIASES.get(key, key)


def _normalize_package_payload(raw: dict[str, Any]) -> dict[str, Any]:
    registry = str(raw.get("registry") or raw.get("ecosystem") or "").strip()
    normalized: dict[str, Any] = {
        "registry": registry,
        "name": str(raw.get("name") or "").strip(),
    }
    latest_release = raw.get("latest_release_number")
    if latest_release is not None:
        normalized["latest_release"] = str(latest_release)
    downloads = raw.get("downloads")
    if isinstance(downloads, (int, float)) and not isinstance(downloads, bool):
        normalized["downloads"] = int(downloads)
    return normalized


class EcosystemsResolver:
    """Async client for ecosyste.ms package resolution and dependency counts."""

    def __init__(self, *, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owns_session = session is None
        self._rate_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def __aenter__(self) -> EcosystemsResolver:
        if self._session is None:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(
                connector=connector,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def _throttle(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            wait_for = MIN_REQUEST_INTERVAL_SECONDS - (now - self._last_request_at)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_request_at = time.monotonic()

    async def _get_json(self, url: str) -> tuple[int, Any | None]:
        if self._session is None:
            logger.warning("ecosystems_resolver request skipped: session not open (%s)", url)
            return 0, None
        await self._throttle()
        try:
            async with self._session.get(url) as resp:
                status = resp.status
                if status != 200:
                    logger.warning(
                        "ecosystems_resolver non-200 response status=%s url=%s",
                        status,
                        url,
                    )
                    return status, None
                try:
                    return status, await resp.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "ecosystems_resolver malformed JSON url=%s (%s)",
                        url,
                        exc,
                    )
                    return status, None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("ecosystems_resolver network error url=%s (%s)", url, exc)
            return 0, None

    async def resolve_repo_packages(self, repo_url: str) -> list[dict]:
        normalized_repo = _normalize_repo_url(repo_url)
        if not normalized_repo:
            logger.warning("ecosystems_resolver invalid repo_url=%r", repo_url)
            return []

        cache_key = normalized_repo
        cached = _cache_get("repo_packages", cache_key)
        if isinstance(cached, list):
            return cached

        packages: list[dict] = []
        page = 1
        fetch_ok = True
        while True:
            lookup_url = (
                f"{API_BASE}/packages/lookup"
                f"?repository_url={quote(normalized_repo, safe='')}"
                f"&per_page=100&page={page}"
            )
            status, body = await self._get_json(lookup_url)
            if status != 200 or not isinstance(body, list):
                fetch_ok = False
                break
            if not body:
                break
            for item in body:
                if isinstance(item, dict):
                    packages.append(_normalize_package_payload(item))
            if len(body) < 100:
                break
            page += 1

        if fetch_ok:
            _cache_put("repo_packages", cache_key, packages)
        return packages

    async def reverse_dependency_count(self, registry: str, package: str) -> int | None:
        normalized_registry = _normalize_registry(registry)
        package_name = str(package or "").strip()
        if not normalized_registry or not package_name:
            logger.warning(
                "ecosystems_resolver invalid reverse-dependency lookup registry=%r package=%r",
                registry,
                package,
            )
            return None

        cache_key = f"{normalized_registry}/{package_name}"
        cached = _cache_get("reverse_dependency_count", cache_key)
        if isinstance(cached, int):
            return cached

        encoded_package = quote(package_name, safe="")
        package_url = (
            f"{API_BASE}/registries/{quote(normalized_registry, safe='')}"
            f"/packages/{encoded_package}"
        )
        _status, body = await self._get_json(package_url)
        if not isinstance(body, dict):
            return None

        count = body.get("dependent_packages_count")
        if not isinstance(count, (int, float)) or isinstance(count, bool):
            logger.warning(
                "ecosystems_resolver missing dependent_packages_count registry=%r package=%r",
                registry,
                package,
            )
            return None

        normalized_count = int(count)
        _cache_put("reverse_dependency_count", cache_key, normalized_count)
        return normalized_count
