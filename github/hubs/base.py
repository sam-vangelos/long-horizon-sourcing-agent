"""Shared async base for package-registry hub clients.

Fail-soft on every path: network errors, non-200 responses, malformed JSON, and
missing fields return ``None`` with a single log line — never an exception to
callers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import ssl
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import certifi

from shared.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_segment(value: str) -> str:
    return value.strip().lower().replace("/", "_").replace(":", "_") or "_unknown"


def normalize_repo_url(repo_url: str) -> str | None:
    """Normalize a repository URL to ``https://github.com/{owner}/{repo}``."""
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


def _cache_path(cache_root: Path, cache_kind: str, cache_key: str) -> Path:
    safe_kind = cache_kind.strip()
    prefix = _safe_segment(cache_key)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    return cache_root / safe_kind / f"{prefix}_{digest}.json"


def _cache_get(
    cache_root: Path,
    cache_kinds: frozenset[str],
    ttl_by_kind: dict[str, timedelta],
    cache_kind: str,
    cache_key: str,
    *,
    hub: str,
) -> Any | None:
    if cache_kind not in cache_kinds:
        logger.warning("%s hub cache get: unknown cache_kind=%r", hub, cache_kind)
        return None
    path = _cache_path(cache_root, cache_kind, cache_key)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "%s hub cache get: corrupt cache at %s (%s); treating as miss",
            hub,
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
    ttl = ttl_by_kind.get(cache_kind)
    if ttl is None or _now() - fetched_at > ttl:
        return None
    stored_key = raw.get("cache_key")
    if stored_key != cache_key:
        return None
    return raw.get("data")


def _cache_put(
    cache_root: Path,
    cache_kinds: frozenset[str],
    cache_kind: str,
    cache_key: str,
    data: Any,
    *,
    hub: str,
) -> None:
    if cache_kind not in cache_kinds:
        logger.warning("%s hub cache put: unknown cache_kind=%r", hub, cache_kind)
        return
    path = _cache_path(cache_root, cache_kind, cache_key)
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
            "%s hub cache put: write failed at %s (%s); proceeding without cache",
            hub,
            path,
            exc,
        )


class BaseHubClient:
    """Async base client for package-registry hub APIs."""

    hub: str
    user_agent: str
    min_request_interval_seconds: float
    cache_kinds: frozenset[str]
    ttl_by_kind: dict[str, timedelta]

    def __init__(self, *, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owns_session = session is None
        self._rate_lock = asyncio.Lock()
        self._last_request_at = 0.0

    @property
    def cache_root(self) -> Path:
        return OUTPUT_DIR / "cache" / "github_hubs" / self.hub

    async def __aenter__(self) -> BaseHubClient:
        if self._session is None:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(
                connector=connector,
                headers={
                    "User-Agent": self.user_agent,
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
            wait_for = self.min_request_interval_seconds - (now - self._last_request_at)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_request_at = time.monotonic()

    async def _get_json(self, url: str) -> tuple[int, Any | None]:
        if self._session is None:
            logger.warning("%s hub request skipped: session not open (%s)", self.hub, url)
            return 0, None
        await self._throttle()
        try:
            async with self._session.get(url) as resp:
                status = resp.status
                if status != 200:
                    logger.warning(
                        "%s hub non-200 response status=%s url=%s",
                        self.hub,
                        status,
                        url,
                    )
                    return status, None
                try:
                    return status, await resp.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "%s hub malformed JSON url=%s (%s)",
                        self.hub,
                        url,
                        exc,
                    )
                    return status, None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("%s hub network error url=%s (%s)", self.hub, url, exc)
            return 0, None

    def _cache_get(self, cache_kind: str, cache_key: str) -> Any | None:
        return _cache_get(
            self.cache_root,
            self.cache_kinds,
            self.ttl_by_kind,
            cache_kind,
            cache_key,
            hub=self.hub,
        )

    def _cache_put(self, cache_kind: str, cache_key: str, data: Any) -> None:
        _cache_put(
            self.cache_root,
            self.cache_kinds,
            cache_kind,
            cache_key,
            data,
            hub=self.hub,
        )

    async def probe(self) -> bool:
        """Hit one cheap endpoint; return False on any failure."""
        url = self._probe_url()
        status, body = await self._get_json(url)
        return status == 200 and body is not None

    def _probe_url(self) -> str:
        raise NotImplementedError
