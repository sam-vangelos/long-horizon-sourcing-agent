"""Behance v2 REST API client.

Designer module Slice 2. The anchor source for designer discovery
(per the spec's source-by-source posture in §3.1: Behance carries the
structured taxonomy; Google CSE in Slice 3 covers personal portfolio
sites; Dribbble in Slice 10 enriches when brief specifies design-system
signals).

The Behance v2 API:

- Base URL: ``https://api.behance.net/v2/``.
- Authentication: ``api_key`` query parameter on every request (NOT a
  Bearer token; NOT an OAuth flow). Adobe stopped accepting new
  developer client registrations in 2020 — Slice 0's Gate A confirms
  whether Cloris's existing key works before this client is exercised
  in anger.
- Rate limit: documented as 150 req/hr free tier. Behance does NOT
  expose ``X-RateLimit-*`` response headers, so this client tracks the
  budget locally with a simple sliding-window counter and respects
  ``Retry-After`` on 429s.
- Endpoints used (Slice 2):
    * ``GET /v2/users`` — search users by free-text + filters.
    * ``GET /v2/users/{username}`` — single user profile.
    * ``GET /v2/users/{username}/projects`` — paginated list of a
      user's projects.
    * ``GET /v2/projects/{project_id}`` — single project (the
      ``modules`` array carries the image URLs the vision-evaluation
      pipeline grounds itself in; Slice 5 wires that consumption).

Design choice: this client is independent of :mod:`shared.rate_limiter`
(which is github-specific — its ``RateLimiter.__init__`` reaches into
``github.config`` for the per-endpoint budgets). A shared abstraction
across sources is a Phase-3 cleanup; today, Behance's much simpler
single-budget shape doesn't justify cross-module coupling.
"""

from __future__ import annotations

import asyncio
import os
import ssl
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import certifi


BEHANCE_API_BASE = "https://api.behance.net/v2"

# Behance free tier — empirically observed; not header-reported.
# Documented at the Adobe Developer console as 150 req/hr; we leave
# generous headroom (130) so a transient burst won't trip the cap.
BEHANCE_RATE_LIMIT_REQUESTS_PER_HOUR = 130
BEHANCE_RATE_WINDOW_SECONDS = 3600

# Default polite spacing between consecutive requests so even a
# slow-burst of 130 calls in an hour stays well-paced.
BEHANCE_MIN_SPACING_SECONDS = 0.5

# Total request timeout per call. Behance's project endpoint can be
# slow when the response carries a long `modules` array (one project
# can have 50+ image modules); 30s is generous but not infinite.
BEHANCE_REQUEST_TIMEOUT_SECONDS = 30

# Retry config — only retry on transient failures. Auth failures
# (401, 403 with "API key" body) are not retried.
_MAX_RETRIES = 4
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class BehanceAuthError(RuntimeError):
    """Raised when Behance rejects the configured API key.

    The blocker fires at :func:`BehanceClient.validate_credentials`
    so strategy formation never burns prompt tokens forming queries
    Cloris cannot execute. Mirrors :class:`github.client.GitHubAuthError`.
    """

    status_code = 401


class BehanceRateLimitExhausted(RuntimeError):
    """Raised when Behance returns 429 even after the documented
    Retry-After window. The client retries up to ``_MAX_RETRIES`` then
    surfaces this so the orchestrator can surface a stop-reason to the
    workspace rather than silently degrading."""


@dataclass
class _Budget:
    """Sliding-window request counter for Behance's per-hour limit.

    Behance does not expose ``X-RateLimit-*`` headers so we count
    locally. The deque holds the unix timestamp of each request in
    the current rolling hour; ``available()`` is "did we make fewer
    than `limit` requests in the last `window` seconds?"
    """

    limit: int = BEHANCE_RATE_LIMIT_REQUESTS_PER_HOUR
    window: float = BEHANCE_RATE_WINDOW_SECONDS
    timestamps: deque[float] = field(default_factory=deque)

    def _prune(self) -> None:
        """Drop timestamps outside the rolling window."""

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
        # The oldest timestamp leaves the window when window seconds
        # have passed since it was recorded.
        return max(0.0, (self.timestamps[0] + self.window) - time.monotonic())

    def consume(self) -> None:
        self.timestamps.append(time.monotonic())


class BehanceClient:
    """Async Behance v2 REST API client with local rate limiting.

    Usage::

        async with BehanceClient() as client:
            await client.validate_credentials()
            results = await client.search_users(query="design system", page=1)
            user = await client.get_user("exampledesigner")
            projects = await client.get_user_projects("exampledesigner")
            project = await client.get_project(project_id=12345)

    The constructor reads ``BEHANCE_API_KEY`` from the environment.
    Tests can pass ``api_key=`` and ``session_factory=`` to inject a
    fake session (see ``tests/test_designer_behance_client.py``).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = BEHANCE_API_BASE,
        budget: _Budget | None = None,
        session_factory: Any = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("BEHANCE_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "BEHANCE_API_KEY not set. Add it to .env or pass to BehanceClient(api_key=...)."
            )
        self._base = base_url.rstrip("/")
        self._budget = budget or _Budget()
        self._session: aiohttp.ClientSession | None = None
        self._session_factory = session_factory  # for tests
        self._last_request_at: float = 0.0
        self._total_calls: int = 0

    async def __aenter__(self) -> BehanceClient:
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
                timeout=aiohttp.ClientTimeout(total=BEHANCE_REQUEST_TIMEOUT_SECONDS),
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
        """Approximate remaining requests in the current hour window."""

        self._budget._prune()
        return max(0, self._budget.limit - len(self._budget.timestamps))

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    async def validate_credentials(self) -> None:
        """Smoke-probe the API key by fetching a known public profile.

        Behance's developer console doesn't expose a `/rate_limit`-style
        endpoint, so we use the lightest-weight authenticated request
        (a known long-active public profile) as the probe.
        """

        status, body = await self._get(
            "/users/jurgenmaier",
            params={},
            allow_404=False,
        )
        if status != 200 or body is None:
            raise BehanceAuthError(
                f"Behance credential preflight failed with status {status}. "
                "Check BEHANCE_API_KEY in .env (Adobe stopped accepting new "
                "keys in 2020; see plans/designer-readiness-gate.md Gate A)."
            )

    async def search_users(
        self,
        *,
        query: str = "",
        country: str | None = None,
        state: str | None = None,
        sort: str = "appreciations",
        page: int = 1,
    ) -> tuple[int, dict | None]:
        """Search Behance users.

        Returns ``(status_code, response_body)``. The body is the raw
        Behance response shape (``{"users": [...], "stats": {...}}``);
        :mod:`designer.acquisition` is the layer that maps it to
        :class:`DesignerSnippet` instances.
        """

        params: dict[str, Any] = {"sort": sort, "page": str(page)}
        if query:
            params["q"] = query
        if country:
            params["country"] = country
        if state:
            params["state"] = state
        return await self._get("/users", params=params)

    async def get_user(self, username: str) -> tuple[int, dict | None]:
        """Fetch a single user profile."""

        return await self._get(f"/users/{username}", params={})

    async def get_user_projects(
        self,
        username: str,
        *,
        sort: str = "published_date",
        page: int = 1,
        per_page: int = 12,
    ) -> tuple[int, dict | None]:
        """List a user's projects (paginated).

        The list endpoint returns lightweight project records (cover
        image, title, appreciation count) — call :func:`get_project`
        to get the full ``modules`` array with all asset URLs.
        """

        params = {"sort": sort, "page": str(page), "per_page": str(per_page)}
        return await self._get(f"/users/{username}/projects", params=params)

    async def get_project(self, project_id: int | str) -> tuple[int, dict | None]:
        """Fetch a single project including its full ``modules`` array.

        The image URLs the vision-evaluation pipeline grounds itself in
        live at ``response["project"]["modules"][i]["sizes"]["original"]``
        (or ``["sizes"]["disp"]`` for ~1024px). Slice 5 wires that
        consumption.
        """

        return await self._get(f"/projects/{project_id}", params={})

    # ------------------------------------------------------------------
    # Internal request plumbing
    # ------------------------------------------------------------------

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, Any],
        allow_404: bool = True,
    ) -> tuple[int, dict | None]:
        """Issue a rate-limited GET. Returns ``(status, body or None)``.

        Returns ``(404, None)`` for a missing resource when
        ``allow_404=True`` (the default — most public lookups can
        legitimately 404 on a deleted user or project; the caller
        decides whether that's a soft or hard error).
        """

        if self._session is None:
            raise RuntimeError(
                "BehanceClient session not initialized — use `async with BehanceClient()`."
            )

        url = f"{self._base}{path}"
        merged_params = {**params, "api_key": self._api_key}

        for attempt in range(_MAX_RETRIES):
            await self._wait_for_budget()
            await self._enforce_min_spacing()

            try:
                async with self._session.get(url, params=merged_params) as resp:
                    self._budget.consume()
                    self._last_request_at = time.monotonic()
                    self._total_calls += 1

                    if resp.status == 200:
                        body = await resp.json()
                        return resp.status, body

                    if resp.status == 404 and allow_404:
                        return 404, None

                    if resp.status in (401, 403):
                        text = (await resp.text())[:300]
                        raise BehanceAuthError(
                            f"Behance API rejected request: HTTP {resp.status}. "
                            f"Body excerpt: {text!r}. Check BEHANCE_API_KEY."
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
                        f"Behance API error {resp.status} on {path}: {text!r}"
                    )

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < _MAX_RETRIES - 1:
                    wait = (2 ** attempt) + 0.5
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Behance API connection failed after {_MAX_RETRIES} retries: {exc}"
                    ) from exc

        raise BehanceRateLimitExhausted(
            f"Behance API request failed after {_MAX_RETRIES} retries: {path}"
        )

    async def _wait_for_budget(self) -> None:
        while not self._budget.available():
            wait = self._budget.seconds_until_available()
            if wait <= 0:
                return
            # Cap individual sleep at 60s so the orchestrator can
            # surface progress updates if the rate-limit wait is long.
            await asyncio.sleep(min(wait + 0.1, 60.0))

    async def _enforce_min_spacing(self) -> None:
        if self._last_request_at == 0.0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < BEHANCE_MIN_SPACING_SECONDS:
            await asyncio.sleep(BEHANCE_MIN_SPACING_SECONDS - elapsed)
