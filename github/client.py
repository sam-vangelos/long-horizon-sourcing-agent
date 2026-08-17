"""Async GitHub API client with rate limiting and pagination.

Wraps GitHub REST API v3 with:
- Per-endpoint rate limiting via RateLimiter
- Automatic pagination following Link headers
- Error handling with retries for transient failures
- ETag caching for conditional requests

Usage:
    async with GitHubClient() as client:
        users = await client.search_users("language:python location:Brazil")
        user = await client.get_user("torvalds")
        repos = await client.get_user_repos("torvalds")
        contributors = await client.get_repo_contributors("owner/repo")
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional, AsyncIterator

import ssl

import aiohttp
import certifi

import github.config as gc
from shared.rate_limiter import RateLimiter
from shared.human_timing import human_delay


# ---------------------------------------------------------------------------
# Retry config
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


class GitHubAuthError(RuntimeError):
    """Raised when GitHub rejects the configured token."""

    status_code = 401


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GitHubClient:
    """Async GitHub REST API client with rate limiting."""

    def __init__(self, token: Optional[str] = None):
        self._token = token or gc.GITHUB_TOKEN
        if not self._token:
            raise RuntimeError(
                "GITHUB_TOKEN not set. Add it to .env or pass to GitHubClient()."
            )
        self._base = gc.GITHUB_API_BASE
        self._session: Optional[aiohttp.ClientSession] = None
        self._limiter = RateLimiter()
        self._etag_cache: dict[str, tuple[str, any]] = {}  # url -> (etag, cached_data)

    async def __aenter__(self) -> GitHubClient:
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        self._session = aiohttp.ClientSession(
            connector=connector,
            headers={
                "Authorization": f"token {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "sourcing-agent/1.0",
            },
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    @property
    def limiter(self) -> RateLimiter:
        return self._limiter

    async def validate_credentials(self) -> None:
        """Fail fast before strategy formation if the configured token is invalid."""
        status, body, _headers = await self._get("/rate_limit", "rest")
        if status != 200 or body is None:
            raise RuntimeError(
                f"GitHub credential preflight failed with status {status}. "
                "Check GITHUB_TOKEN in .env."
            )

    # ── Core request method ──────────────────────────────────────────

    async def _request(
        self,
        method: str,
        url: str,
        endpoint_class: str = "rest",
        params: Optional[dict] = None,
        use_etag: bool = False,
    ) -> tuple[int, dict | list | None, dict]:
        """Make a rate-limited request. Returns (status_code, json_body, headers)."""
        if not url.startswith("http"):
            url = f"{self._base}{url}"

        for attempt in range(_MAX_RETRIES):
            await self._limiter.acquire(endpoint_class)

            headers = {}
            if use_etag and url in self._etag_cache:
                etag, _ = self._etag_cache[url]
                headers["If-None-Match"] = etag

            try:
                async with self._session.request(method, url, params=params, headers=headers) as resp:
                    # Update rate limiter from response headers
                    self._limiter.update_from_headers(endpoint_class, dict(resp.headers))

                    # ETag: 304 Not Modified — return cached data (free, doesn't count)
                    if resp.status == 304 and url in self._etag_cache:
                        _, cached = self._etag_cache[url]
                        return 304, cached, dict(resp.headers)

                    # Success
                    if resp.status in (200, 201):
                        body = await resp.json()
                        # Cache ETag if present
                        etag = resp.headers.get("ETag")
                        if etag and use_etag:
                            self._etag_cache[url] = (etag, body)
                        return resp.status, body, dict(resp.headers)

                    # Rate limited — wait and retry
                    if resp.status == 403 and "rate limit" in (await resp.text()).lower():
                        retry_after = resp.headers.get("Retry-After")
                        wait = float(retry_after) if retry_after else 60.0
                        print(f"    [github] Rate limited. Waiting {wait:.0f}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait)
                        continue

                    # Retryable server errors
                    if resp.status in _RETRYABLE_STATUS_CODES:
                        wait = (2 ** attempt) + human_delay(0.5, 2.0)
                        print(f"    [github] HTTP {resp.status}, retrying in {wait:.1f}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait)
                        continue

                    # 404 — not found (not an error for optional endpoints)
                    if resp.status == 404:
                        return 404, None, dict(resp.headers)

                    if resp.status == 401:
                        text = await resp.text()
                        raise GitHubAuthError(
                            "GitHub API authentication failed: 401 Bad credentials. "
                            f"Check GITHUB_TOKEN in .env. Response: {text[:300]}"
                        )

                    # Other errors
                    text = await resp.text()
                    raise RuntimeError(f"GitHub API error {resp.status}: {text[:500]}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < _MAX_RETRIES - 1:
                    wait = (2 ** attempt) + human_delay(0.5, 2.0)
                    print(f"    [github] Connection error ({e}), retrying in {wait:.1f}s")
                    await asyncio.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"GitHub API request failed after {_MAX_RETRIES} retries: {url}")

    async def _get(self, url: str, endpoint_class: str = "rest", params: Optional[dict] = None, use_etag: bool = False) -> tuple[int, dict | list | None, dict]:
        return await self._request("GET", url, endpoint_class, params, use_etag)

    # ── Pagination helper ────────────────────────────────────────────

    async def _paginate(
        self,
        url: str,
        endpoint_class: str = "rest",
        params: Optional[dict] = None,
        max_pages: int = 10,
        max_items: int = 0,
    ) -> list[dict]:
        """Fetch all pages of a paginated endpoint. Returns accumulated items."""
        params = dict(params or {})
        params.setdefault("per_page", gc.RESULTS_PER_PAGE)

        all_items = []
        current_url = url if url.startswith("http") else f"{self._base}{url}"

        for page_num in range(1, max_pages + 1):
            if page_num > 1:
                params["page"] = page_num

            status, body, headers = await self._get(current_url, endpoint_class, params)
            if status != 200 or body is None:
                break

            # Search endpoints wrap results in {"items": [...]}
            items = body.get("items", body) if isinstance(body, dict) else body
            if not isinstance(items, list):
                break

            all_items.extend(items)

            if max_items and len(all_items) >= max_items:
                all_items = all_items[:max_items]
                break

            # Check if there are more pages via Link header
            link = headers.get("Link", "")
            if 'rel="next"' not in link:
                break

        return all_items

    # ── Search endpoints ─────────────────────────────────────────────

    async def search_users(
        self,
        query: str,
        max_results: int = 0,
        sort: str = "",
        order: str = "desc",
    ) -> tuple[int, list[dict]]:
        """Search users. Returns (total_count, list of user dicts).

        query: GitHub search syntax, e.g. "language:python location:Brazil followers:>50"
        """
        params = {"q": query, "per_page": gc.RESULTS_PER_PAGE}
        if sort:
            params["sort"] = sort
            params["order"] = order

        max_pages = (max_results // gc.RESULTS_PER_PAGE + 1) if max_results else 10
        max_pages = min(max_pages, gc.MAX_RESULTS_PER_QUERY // gc.RESULTS_PER_PAGE)

        # First page to get total_count
        status, body, headers = await self._get("/search/users", "search", params)
        if status != 200 or body is None:
            return 0, []

        total_count = body.get("total_count", 0)
        items = body.get("items", [])

        # Fetch remaining pages
        if max_pages > 1 and len(items) >= gc.RESULTS_PER_PAGE:
            for page in range(2, max_pages + 1):
                params["page"] = page
                status, body, headers = await self._get("/search/users", "search", params)
                if status != 200 or body is None:
                    break
                page_items = body.get("items", [])
                if not page_items:
                    break
                items.extend(page_items)
                if max_results and len(items) >= max_results:
                    items = items[:max_results]
                    break

        return total_count, items

    async def search_code(
        self,
        query: str,
        max_results: int = 100,
    ) -> tuple[int, list[dict]]:
        """Search code. Returns (total_count, list of code result dicts).

        Very rate-limited: 10 req/min. Use sparingly.
        query: e.g. '"from trl import" language:python'
        """
        params = {"q": query, "per_page": gc.RESULTS_PER_PAGE}

        status, body, headers = await self._get("/search/code", "code_search", params)
        if status != 200 or body is None:
            return 0, []

        total_count = body.get("total_count", 0)
        items = body.get("items", [])

        # Code search is so rate-limited we usually only fetch page 1
        if max_results > gc.RESULTS_PER_PAGE and len(items) >= gc.RESULTS_PER_PAGE:
            max_pages = min(max_results // gc.RESULTS_PER_PAGE + 1, 3)  # cap at 3 pages
            for page in range(2, max_pages + 1):
                params["page"] = page
                status, body, headers = await self._get("/search/code", "code_search", params)
                if status != 200 or body is None:
                    break
                page_items = body.get("items", [])
                if not page_items:
                    break
                items.extend(page_items)

        return total_count, items[:max_results]

    async def search_repos(
        self,
        query: str,
        max_results: int = 100,
        sort: str = "stars",
        order: str = "desc",
    ) -> tuple[int, list[dict]]:
        """Search repositories. Returns (total_count, list of repo dicts)."""
        params = {"q": query, "per_page": gc.RESULTS_PER_PAGE, "sort": sort, "order": order}

        status, body, headers = await self._get("/search/repositories", "search", params)
        if status != 200 or body is None:
            return 0, []

        total_count = body.get("total_count", 0)
        items = body.get("items", [])
        return total_count, items[:max_results]

    # ── User endpoints ───────────────────────────────────────────────

    async def get_user(self, username: str) -> Optional[dict]:
        """Fetch full user profile."""
        status, body, headers = await self._get(f"/users/{username}", use_etag=True)
        if status in (200, 304):
            return body
        return None

    async def get_user_repos(
        self,
        username: str,
        sort: str = "pushed",
        max_repos: int = 0,
    ) -> list[dict]:
        """Fetch user's repositories sorted by most recently pushed."""
        max_repos = max_repos or gc.MAX_REPOS_PER_USER
        params = {"sort": sort, "direction": "desc", "per_page": min(max_repos, gc.RESULTS_PER_PAGE)}
        items = await self._paginate(
            f"/users/{username}/repos",
            params=params,
            max_pages=(max_repos // gc.RESULTS_PER_PAGE + 1),
            max_items=max_repos,
        )
        return items

    async def get_followers(
        self,
        username: str,
        max_results: int = 200,
    ) -> list[dict]:
        """Fetch followers of a user.

        Returns list of user dicts (login, id, avatar_url, etc.).
        """
        params = {"per_page": gc.RESULTS_PER_PAGE}
        items = await self._paginate(
            f"/users/{username}/followers",
            params=params,
            max_pages=(max_results // gc.RESULTS_PER_PAGE + 1),
            max_items=max_results,
        )
        return items

    async def get_following(
        self,
        username: str,
        max_results: int = 200,
    ) -> list[dict]:
        """Fetch users that a user follows.

        Returns list of user dicts (login, id, avatar_url, etc.).
        """
        params = {"per_page": gc.RESULTS_PER_PAGE}
        items = await self._paginate(
            f"/users/{username}/following",
            params=params,
            max_pages=(max_results // gc.RESULTS_PER_PAGE + 1),
            max_items=max_results,
        )
        return items

    # ── Repo endpoints ───────────────────────────────────────────────

    async def get_repo_contributors(
        self,
        owner_repo: str,
        max_contributors: int = 500,
    ) -> list[dict]:
        """Fetch repository contributors sorted by commit count.

        owner_repo: "owner/repo" format
        """
        params = {"per_page": gc.RESULTS_PER_PAGE, "anon": "false"}
        items = await self._paginate(
            f"/repos/{owner_repo}/contributors",
            params=params,
            max_pages=5,
            max_items=max_contributors,
        )
        return items

    async def get_repo(self, owner_repo: str) -> Optional[dict]:
        """Fetch repository metadata."""
        status, body, headers = await self._get(f"/repos/{owner_repo}", use_etag=True)
        if status in (200, 304):
            return body
        return None

    async def get_repo_languages(self, owner_repo: str) -> dict[str, int]:
        """Fetch language breakdown for a repo. Returns {language: bytes}."""
        status, body, headers = await self._get(f"/repos/{owner_repo}/languages", use_etag=True)
        if status in (200, 304) and isinstance(body, dict):
            return body
        return {}

    async def get_repo_readme(self, owner_repo: str) -> Optional[str]:
        """Fetch a repo's README content.

        owner_repo: "owner/repo" format
        Returns decoded text content capped at 2KB, or None if not found.
        """
        status, body, headers = await self._get(
            f"/repos/{owner_repo}/readme",
            use_etag=True,
        )
        if status in (200, 304) and body and "content" in body:
            import base64
            try:
                content = base64.b64decode(body["content"]).decode("utf-8", errors="replace")
                return content[:2048]  # Cap at 2KB
            except Exception:
                return None
        return None

    async def get_stargazers(
        self,
        owner_repo: str,
        max_results: int = 500,
    ) -> list[dict]:
        """Fetch stargazers of a repository.

        owner_repo: "owner/repo" format
        Returns list of user dicts (login, id, avatar_url, etc.).
        """
        params = {"per_page": gc.RESULTS_PER_PAGE}
        items = await self._paginate(
            f"/repos/{owner_repo}/stargazers",
            params=params,
            max_pages=(max_results // gc.RESULTS_PER_PAGE + 1),
            max_items=max_results,
        )
        return items

    # ── Commit endpoints ─────────────────────────────────────────────

    async def get_user_commits(
        self,
        owner_repo: str,
        author: str,
        max_commits: int = 0,
    ) -> list[dict]:
        """Fetch commits by a specific author in a repo.

        Used for email discovery and contribution quality assessment.
        """
        max_commits = max_commits or gc.MAX_COMMITS_FOR_EMAIL
        params = {"author": author, "per_page": min(max_commits, gc.RESULTS_PER_PAGE)}
        items = await self._paginate(
            f"/repos/{owner_repo}/commits",
            params=params,
            max_pages=1,
            max_items=max_commits,
        )
        return items

    # ── Org endpoints ────────────────────────────────────────────────

    async def get_org_members(
        self,
        org: str,
        max_members: int = 500,
    ) -> list[dict]:
        """Fetch public members of an organization."""
        params = {"per_page": gc.RESULTS_PER_PAGE}
        items = await self._paginate(
            f"/orgs/{org}/members",
            params=params,
            max_pages=5,
            max_items=max_members,
        )
        return items

    # ── User-organization membership (Slice 3 — OSS Maintainers) ────

    async def get_user_orgs(self, username: str) -> list[dict]:
        """Fetch the user's public org memberships.

        OSS Maintainers Slice 3. Returns ``[]`` on 404 or non-200
        (private org membership is not visible to the API; the spec
        notes this as expected). Used by the maintainership classifier
        to corroborate "maintainer at this project's owning org"
        signals when present.
        """

        params = {"per_page": gc.RESULTS_PER_PAGE}
        items = await self._paginate(
            f"/users/{username}/orgs",
            params=params,
            max_pages=2,
            max_items=200,
        )
        return items

    # ── Pull request endpoints (Slice 3 — OSS Maintainers) ──────────

    async def list_repo_pulls(
        self,
        owner_repo: str,
        state: str = "closed",
        sort: str = "updated",
        max_results: int = 100,
    ) -> list[dict]:
        """Fetch pull requests for a repo with `merged_by` populated.

        OSS Maintainers Slice 3. ``state`` defaults to ``closed`` and
        ``sort`` to ``updated`` because the maintainership classifier
        is interested in recently-merged PRs (used to score merge
        authority via ``merged_by.login``). Returned dicts carry the
        full PR shape, including ``merged_by``, ``user``, and
        ``merged_at``.

        Note: GitHub's ``/pulls`` does NOT filter by author server-
        side. Callers filter the returned list by ``merged_by.login``
        to find merge-authority signals; the per-target-project
        budget caps the number of items pulled.
        """

        params: dict = {
            "state": state,
            "sort": sort,
            "direction": "desc",
            "per_page": gc.RESULTS_PER_PAGE,
        }
        items = await self._paginate(
            f"/repos/{owner_repo}/pulls",
            params=params,
            max_pages=(max_results // gc.RESULTS_PER_PAGE + 1),
            max_items=max_results,
        )
        return items

    async def get_pull_reviews(
        self,
        owner_repo: str,
        pull_number: int,
    ) -> list[dict]:
        """Fetch reviews for a single pull request.

        OSS Maintainers Slice 3. Used by the classifier's "reviewer
        activity" signal: sample a handful of recent merged PRs from
        :meth:`list_repo_pulls`, then call this per-PR to count
        reviews authored by the candidate username. Returns ``[]`` on
        404 or non-200; the classifier treats absence as zero
        reviews, not as an error.
        """

        params = {"per_page": gc.RESULTS_PER_PAGE}
        items = await self._paginate(
            f"/repos/{owner_repo}/pulls/{pull_number}/reviews",
            params=params,
            max_pages=2,
            max_items=100,
        )
        return items

    # ── Release endpoints (Slice 3 — OSS Maintainers) ───────────────

    async def list_repo_releases(
        self,
        owner_repo: str,
        max_results: int = 30,
    ) -> list[dict]:
        """Fetch a repo's releases, newest first, with ``author`` populated.

        OSS Maintainers Slice 3. Used by the classifier's "release tag
        authorship" signal (count releases authored by the candidate)
        AND by the project-quality sub-index (Slice 5) for release
        cadence. ETag-cached at the underlying ``_get`` layer when
        possible; the maintainer-signal cache layer above adds a 7d
        TTL.
        """

        params = {"per_page": gc.RESULTS_PER_PAGE}
        items = await self._paginate(
            f"/repos/{owner_repo}/releases",
            params=params,
            max_pages=(max_results // gc.RESULTS_PER_PAGE + 1),
            max_items=max_results,
        )
        return items

    # ── Repo contents (Slice 3 — OSS Maintainers) ───────────────────

    async def get_repo_contents(
        self,
        owner_repo: str,
        path: str,
    ) -> Optional[str]:
        """Fetch a single file's plaintext content via ``/contents/``.

        OSS Maintainers Slice 3. Used to read CONTRIBUTORS / MAINTAINERS
        / GOVERNANCE.md files for the maintainership classifier's
        text-mine signals. Returns the decoded plaintext (capped at
        ~64KB to bound memory; governance docs are typically < 10KB)
        or ``None`` on 404 or any decode failure.

        Files larger than 1MB return None (GitHub's contents API
        returns a different shape for large files; the maintainer
        classifier's signals all live in small governance docs so
        the cap is intentional).
        """

        status, body, _headers = await self._get(
            f"/repos/{owner_repo}/contents/{path}",
            use_etag=True,
        )
        if status not in (200, 304) or not isinstance(body, dict):
            return None
        if "content" not in body:
            return None
        size = body.get("size")
        if isinstance(size, int) and size > 1_000_000:
            return None
        import base64

        try:
            content = base64.b64decode(body["content"]).decode(
                "utf-8", errors="replace"
            )
            return content[:65536]
        except Exception:
            return None

    # ── Profile README ───────────────────────────────────────────────

    async def get_profile_readme(self, username: str) -> Optional[str]:
        """Fetch user's profile README content (the {username}/{username} repo).

        Returns decoded text content, or None if no profile README exists.
        """
        status, body, headers = await self._get(
            f"/repos/{username}/{username}/readme",
            use_etag=True,
        )
        if status in (200, 304) and body and "content" in body:
            import base64
            try:
                content = base64.b64decode(body["content"]).decode("utf-8", errors="replace")
                return content[:5000]  # Cap at 5KB
            except Exception:
                return None
        return None

    # ── Status ───────────────────────────────────────────────────────

    def status_line(self) -> str:
        """One-line status for console output."""
        return self._limiter.status_line()
