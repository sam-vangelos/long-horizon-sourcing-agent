"""Google Custom Search Engine (CSE) v1 REST API client.

Designer module Slice 3. Secondary discovery anchor for personal
portfolio sites (Behance is the structured-taxonomy primary anchor;
CSE covers personal sites Behance doesn't). Filtered to portfolio-host
domains (cargo.site, squarespace.com, format.com, semplice.com,
awwwards.com, siteinspire.com) per spec §3.1.

The CSE v1 API:

- Base URL: ``https://www.googleapis.com/customsearch/v1``.
- Authentication: ``key`` (Google API key) + ``cx`` (CSE ID) on every
  request. The ``cx`` is a programmable search engine ID configured
  in the Google Cloud console; Cloris's CSE is restricted at config
  time to the portfolio-host domains.
- Rate limit: 100 queries/day free; paid expansion to 10K/day at
  $5/1K. Headers don't expose remaining-quota; this client tracks
  the daily window locally with the same sliding-window pattern as
  :class:`designer.sources.behance._Budget`.
- Endpoint used: ``GET /customsearch/v1`` with ``q`` (search text)
  and pagination via ``start`` (1, 11, 21 — CSE returns 10 per page).
- Result shape: ``{"items": [{"title", "link", "displayLink",
  "snippet", "pagemap": {"cse_thumbnail": [{"src", ...}]}}, ...]}``.

Asset-rights posture: per ``designer/SOURCE_RIGHTS.md``, Cloris uses
``pagemap.cse_thumbnail.src`` URLs as evaluation inputs (Google's CSE
ToS permits display in custom search results). Direct fetch of the
linked portfolio page is OFF in v1; v1.5 enables per-host with ToS
sign-off.
"""

from __future__ import annotations

import asyncio
import os
import ssl
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import aiohttp
import certifi


GOOGLE_CSE_API_BASE = "https://www.googleapis.com/customsearch/v1"

# Google CSE free tier — 100 queries / 24h. Leave headroom (90).
GOOGLE_CSE_RATE_LIMIT_REQUESTS_PER_DAY = 90
GOOGLE_CSE_RATE_WINDOW_SECONDS = 86400

# Polite spacing between consecutive CSE calls.
GOOGLE_CSE_MIN_SPACING_SECONDS = 0.3

GOOGLE_CSE_REQUEST_TIMEOUT_SECONDS = 30

_MAX_RETRIES = 4
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# Portfolio-host domains the Designer module restricts CSE discovery
# to. Slice 3 ships with this baseline; expansion (or per-customer
# additions) is a follow-up.
PORTFOLIO_HOST_DOMAINS: tuple[str, ...] = (
    "cargo.site",
    "squarespace.com",
    "format.com",
    "semplice.com",
    "awwwards.com",
    "siteinspire.com",
)


class GoogleCSEAuthError(RuntimeError):
    """Raised when Google rejects the API key or CSE ID."""

    status_code = 401


class GoogleCSEQuotaExhausted(RuntimeError):
    """Raised when CSE returns 429 even after the documented Retry-After.

    Distinct from :class:`GoogleCSEAuthError` because the orchestrator
    surfaces these as different stop-reasons (auth = recruiter must
    fix config; quota = wait + retry tomorrow).
    """


@dataclass
class _Budget:
    """Sliding-window request counter for CSE's per-day limit."""

    limit: int = GOOGLE_CSE_RATE_LIMIT_REQUESTS_PER_DAY
    window: float = GOOGLE_CSE_RATE_WINDOW_SECONDS
    timestamps: deque[float] = field(default_factory=deque)

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.window
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def available(self) -> bool:
        self._prune()
        return len(self.timestamps) < self.limit

    def seconds_until_available(self) -> float:
        self._prune()
        if len(self.timestamps) < self.limit:
            return 0.0
        return max(0.0, (self.timestamps[0] + self.window) - time.monotonic())

    def consume(self) -> None:
        self.timestamps.append(time.monotonic())


class GoogleCSEClient:
    """Async Google CSE v1 client with local quota tracking.

    Usage::

        async with GoogleCSEClient() as client:
            await client.validate_credentials()
            results = await client.search(query="design system Cargo", start=1)

    Reads ``GOOGLE_CSE_API_KEY`` and ``GOOGLE_CSE_ID`` from environment.
    Tests pass them explicitly + a ``session_factory`` for the fake
    session.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cse_id: str | None = None,
        base_url: str = GOOGLE_CSE_API_BASE,
        budget: _Budget | None = None,
        session_factory: Any = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get(
            "GOOGLE_CSE_API_KEY", ""
        )
        self._cse_id = cse_id if cse_id is not None else os.environ.get(
            "GOOGLE_CSE_ID", ""
        )
        if not self._api_key:
            raise RuntimeError(
                "GOOGLE_CSE_API_KEY not set. Add to .env or pass to "
                "GoogleCSEClient(api_key=...)."
            )
        if not self._cse_id:
            raise RuntimeError(
                "GOOGLE_CSE_ID not set. Add to .env or pass to "
                "GoogleCSEClient(cse_id=...)."
            )
        self._base = base_url.rstrip("/")
        self._budget = budget or _Budget()
        self._session: aiohttp.ClientSession | None = None
        self._session_factory = session_factory
        self._last_request_at: float = 0.0
        self._total_calls: int = 0

    async def __aenter__(self) -> GoogleCSEClient:
        if self._session_factory is not None:
            self._session = self._session_factory()
        else:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(
                connector=connector,
                headers={
                    "User-Agent": "sourcing-agent-designer/1.0",
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=GOOGLE_CSE_REQUEST_TIMEOUT_SECONDS),
            )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session is not None:
            await self._session.close()

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def budget_remaining(self) -> int:
        self._budget._prune()
        return max(0, self._budget.limit - len(self._budget.timestamps))

    async def validate_credentials(self) -> None:
        """Smoke-probe the API key + CSE ID with a trivial query."""

        status, body = await self.search(query="behance", start=1)
        if status != 200 or body is None:
            raise GoogleCSEAuthError(
                f"Google CSE credential preflight failed with status {status}. "
                "Check GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID."
            )

    async def search(
        self,
        *,
        query: str,
        start: int = 1,
        site_filter: str | None = None,
    ) -> tuple[int, dict | None]:
        """Issue a CSE search.

        ``start`` is 1-indexed and CSE returns 10 results per page;
        valid start values are 1, 11, 21, ..., 91. Higher values
        (>91) are rejected by the API.

        ``site_filter``, when set, prepends ``site:<domain>`` to the
        query so CSE returns results only from that host. Used by the
        orchestrator to fan a single capability-area query out across
        the portfolio-host set.
        """

        params: dict[str, Any] = {
            "key": self._api_key,
            "cx": self._cse_id,
            "q": (f"site:{site_filter} {query}" if site_filter else query).strip(),
            "start": str(start),
            "num": "10",
        }
        return await self._get(params=params)

    async def _get(
        self, *, params: dict[str, Any]
    ) -> tuple[int, dict | None]:
        if self._session is None:
            raise RuntimeError(
                "GoogleCSEClient session not initialized — use `async with GoogleCSEClient()`."
            )

        url = self._base
        for attempt in range(_MAX_RETRIES):
            await self._wait_for_budget()
            await self._enforce_min_spacing()

            try:
                async with self._session.get(url, params=params) as resp:
                    self._budget.consume()
                    self._last_request_at = time.monotonic()
                    self._total_calls += 1

                    if resp.status == 200:
                        body = await resp.json()
                        return resp.status, body

                    if resp.status in (401, 403):
                        text = (await resp.text())[:300]
                        raise GoogleCSEAuthError(
                            f"Google CSE rejected request: HTTP {resp.status}. "
                            f"Body excerpt: {text!r}. Check GOOGLE_CSE_API_KEY/ID."
                        )

                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", "60"))
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status in _RETRYABLE_STATUS_CODES:
                        wait = (2 ** attempt) + 0.5
                        await asyncio.sleep(wait)
                        continue

                    text = (await resp.text())[:500]
                    raise RuntimeError(
                        f"Google CSE error {resp.status}: {text!r}"
                    )

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < _MAX_RETRIES - 1:
                    wait = (2 ** attempt) + 0.5
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Google CSE connection failed after {_MAX_RETRIES} retries: {exc}"
                    ) from exc

        raise GoogleCSEQuotaExhausted(
            f"Google CSE request failed after {_MAX_RETRIES} retries"
        )

    async def _wait_for_budget(self) -> None:
        while not self._budget.available():
            wait = self._budget.seconds_until_available()
            if wait <= 0:
                return
            # Cap individual sleeps at 60s so the orchestrator can
            # surface progress even on long quota waits.
            await asyncio.sleep(min(wait + 0.1, 60.0))

    async def _enforce_min_spacing(self) -> None:
        if self._last_request_at == 0.0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < GOOGLE_CSE_MIN_SPACING_SECONDS:
            await asyncio.sleep(GOOGLE_CSE_MIN_SPACING_SECONDS - elapsed)


# ---------------------------------------------------------------------------
# CSE result → DesignerSnippet mapping
# ---------------------------------------------------------------------------


def cse_result_to_identity_key(link: str) -> str:
    """Canonical identity-key for a CSE-discovered designer.

    Uses ``cse:<host>/<first-path-segment>`` so two results from the
    same designer's site (e.g., ``cargo.site/joe/work`` and
    ``cargo.site/joe/about``) collapse to ``cse:cargo.site/joe``.
    For root-domain pages (no path), the host alone is the key
    (``cse:exampledesigner.com``).
    """

    parsed = urlparse(link if "://" in link else f"https://{link}")
    host = parsed.netloc.lower().lstrip("www.") or link.lower()
    # Drop trailing slash, take first non-empty path segment.
    path_segments = [seg for seg in parsed.path.split("/") if seg]
    if path_segments:
        return f"cse:{host}/{path_segments[0]}"
    return f"cse:{host}"


def cse_result_to_display_name(item: dict[str, Any]) -> str:
    """Best-effort display name from a CSE result item.

    Falls through:
    - `pagemap.metatags[0]['og:title']` (most precise)
    - `title` minus host suffix (e.g., "Joe Designer — Cargo" → "Joe Designer")
    - host name as last resort
    """

    pagemap = item.get("pagemap")
    if isinstance(pagemap, dict):
        metatags = pagemap.get("metatags")
        if isinstance(metatags, list) and metatags:
            first = metatags[0]
            if isinstance(first, dict):
                og_title = first.get("og:title")
                if isinstance(og_title, str) and og_title.strip():
                    return og_title.strip()

    title = item.get("title")
    if isinstance(title, str) and title.strip():
        # CSE titles often include " — <Host>" or " | <Host>" suffix.
        for separator in (" — ", " | ", " - "):
            if separator in title:
                left = title.split(separator)[0].strip()
                if left:
                    return left
        return title.strip()

    display_link = item.get("displayLink")
    if isinstance(display_link, str) and display_link.strip():
        return display_link.strip()

    return ""


def cse_result_thumbnail_url(item: dict[str, Any]) -> str:
    """Extract the cse_thumbnail src if present, else empty string."""

    pagemap = item.get("pagemap")
    if not isinstance(pagemap, dict):
        return ""
    thumbnails = pagemap.get("cse_thumbnail")
    if not isinstance(thumbnails, list) or not thumbnails:
        return ""
    first = thumbnails[0]
    if not isinstance(first, dict):
        return ""
    src = first.get("src")
    return src if isinstance(src, str) else ""
