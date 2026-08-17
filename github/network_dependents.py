"""Throttled HTML scrape for ``/network/dependents`` + registry signal
acquisition for PyPI / npm (audit Move #22).

Per OSS Maintainers Module Spec §9, GitHub does not expose downstream-
dependents count via the REST API. The number is publicly rendered on
the project's ``/network/dependents`` HTML page, which we scrape with
a defensive regex (no DOM path) and a conservative throttle. Per spec
§12, the parse is fail-soft: any parse failure returns ``None``, the
caller logs a warning, and the project-quality sub-index drops the
signal for that project.

Audit Move #22 extends this module with PyPI + npm registry integrations
so the project-quality sub-index can fold downstream-package signal
into projects that ship to either ecosystem:

- ``fetch_pypi_recent_downloads`` — `pypistats.org` last-month count
- ``fetch_npm_recent_downloads`` — `api.npmjs.org` last-month count

Both registries skip a "dependents count" surface (neither exposes it
canonically); the GitHub ``/network/dependents`` page above remains
the authoritative dependents signal. The registry signals layer in as
adjacency proxies: heavy package downloads imply broader user base /
ecosystem reach than dependents-graph alone.

The cache layer at :mod:`github.maintainer_signal_cache` serves as the
front-line throttle (30-day TTL per spec §9): once we know the count
for ``kubernetes/kubernetes``, we don't re-scrape for a month even if
multiple briefs target it. The ``throttle_seconds`` knob below adds
per-call delay when the cache misses, to avoid hammering the public
surface during a burst of cache rebuilds.

Behaviour posture (uniform across all three integrations):

- Network errors ⇒ warn + return None.
- Non-200 HTTP ⇒ warn + return None.
- 404 (package not found) ⇒ debug log, return None silently (a missing
  package is a data signal, not an error).
- Markup / JSON that doesn't match the expected shape ⇒ warn + return None.
- The integer is the parse result, no upper bound enforced.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import ssl
from typing import Optional

import aiohttp
import certifi

from github import maintainer_signal_cache as mcache

logger = logging.getLogger(__name__)


# The dependents page renders the count inline as e.g.::
#
#     <a class="btn-link selected" href="/{owner}/{repo}/network/dependents?dependent_type=REPOSITORY">
#       <svg ...></svg>
#       12,345,678
#       <span>Repositories</span>
#     </a>
#
# We don't bind to the DOM path (per spec §12 — defensive regex on
# the count label, not DOM path). The regex matches an integer with
# optional thousands separators preceded by ANY interleaving HTML
# whitespace + tags before the "Repositories" / "Packages" label.
# Permissive on the gap so minor markup tweaks don't break parsing;
# strict on the count + label so we don't match unrelated digits.
# "Repositories" is the canonical answer; if present we prefer that.
# "Packages" is a fallback (some repos only expose package
# dependents). Case-insensitive across the label.
_REPOSITORY_DEPENDENTS_RE = re.compile(
    r"([0-9][0-9,]{0,15})[\s\S]{0,200}?Repositories",
    re.IGNORECASE,
)
_PACKAGE_DEPENDENTS_RE = re.compile(
    r"([0-9][0-9,]{0,15})[\s\S]{0,200}?Packages",
    re.IGNORECASE,
)


_USER_AGENT = "sourcing-agent/1.0 (oss-maintainers-module)"
_TIMEOUT_SECONDS = 15


async def fetch_dependents_count(
    owner: str,
    repo: str,
    *,
    throttle_seconds: float = 2.0,
    use_cache: bool = True,
) -> Optional[int]:
    """Return the count of repository dependents for ``owner/repo``, or None.

    Cache-aware: hits the 30-day cache via
    :mod:`github.maintainer_signal_cache` first; only scrapes on miss.
    The ``throttle_seconds`` knob applies post-fetch (a defensive
    sleep so back-to-back cache rebuilds don't rate-limit GitHub's
    public HTML surface).

    Spec §12 contract: any failure mode (network, HTTP, parse) returns
    ``None`` and emits a single warning. Callers (project-quality sub-
    index in Slice 5) treat ``None`` as "signal absent for this
    project" and continue.
    """

    if use_cache:
        cached = mcache.get(owner, repo, "network_dependents")
        if cached is not None and isinstance(cached.data, int):
            return cached.data

    url = f"https://github.com/{owner}/{repo}/network/dependents"
    html = await _fetch_html(url)
    if html is None:
        return None

    count = _parse_count(html)
    if count is None:
        logger.warning(
            "network_dependents: parse failed for %s/%s — page markup may have changed",
            owner,
            repo,
        )
        return None

    if use_cache:
        mcache.put(owner, repo, "network_dependents", count)
    if throttle_seconds > 0:
        await asyncio.sleep(throttle_seconds)
    return count


async def _fetch_html(url: str) -> Optional[str]:
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)
    headers = {"User-Agent": _USER_AGENT, "Accept": "text/html"}
    try:
        async with aiohttp.ClientSession(
            connector=connector, headers=headers, timeout=timeout
        ) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(
                        "network_dependents: HTTP %d for %s", resp.status, url
                    )
                    return None
                return await resp.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning("network_dependents: fetch failed for %s (%s)", url, exc)
        return None


def _parse_count(html: str) -> Optional[int]:
    """Extract the integer count from the dependents page HTML.

    Prefers the "Repositories" label; falls back to "Packages" only
    when the repo doesn't expose repo-level dependents.
    """

    match = _REPOSITORY_DEPENDENTS_RE.search(html)
    if match is None:
        match = _PACKAGE_DEPENDENTS_RE.search(html)
    if match is None:
        return None
    raw = match.group(1).replace(",", "")
    try:
        return int(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# PyPI registry signals (audit Move #22)
# ---------------------------------------------------------------------------


_PYPI_USER_AGENT = "sourcing-agent/1.0 (oss-maintainers-module; pypi-signal-fetch)"
_PYPI_STATS_RECENT_URL = "https://pypistats.org/api/packages/{package}/recent"


async def fetch_pypi_recent_downloads(
    package: str,
    *,
    throttle_seconds: float = 1.0,
    use_cache: bool = True,
    fetch_text: Optional["AsyncTextFetcher"] = None,
) -> Optional[int]:
    """Return the last-month PyPI download count for ``package``, or None.

    Audit Move #22. Cache-aware via :mod:`github.maintainer_signal_cache`
    under the ``pypi_recent_downloads`` signal kind. Failure-mode posture
    matches :func:`fetch_dependents_count` — every error path returns
    ``None`` and emits a single warning so the project-quality sub-index
    can drop the signal cleanly.

    The ``fetch_text`` injection point is for tests — production callers
    let the default :func:`_fetch_text_default` (aiohttp) run.
    """

    package = package.strip().lower()
    if not package:
        return None

    if use_cache:
        cached = mcache.get(package, "_pypi", "pypi_recent_downloads")
        if cached is not None and isinstance(cached.data, int):
            return cached.data

    fetcher = fetch_text or _fetch_text_default
    url = _PYPI_STATS_RECENT_URL.format(package=package)
    body = await fetcher(url, headers={"User-Agent": _PYPI_USER_AGENT})
    if body is None:
        return None

    count = _parse_pypi_recent_downloads(body)
    if count is None:
        logger.warning(
            "network_dependents: pypi parse failed for %s — pypistats payload "
            "may have changed shape",
            package,
        )
        return None

    if use_cache:
        mcache.put(package, "_pypi", "pypi_recent_downloads", count)
    if throttle_seconds > 0:
        await asyncio.sleep(throttle_seconds)
    return count


def _parse_pypi_recent_downloads(body: str) -> Optional[int]:
    """Parse `pypistats.org/api/packages/<pkg>/recent` JSON.

    Expected shape::

        {"data": {"last_day": ..., "last_week": ..., "last_month": 12345},
         "package": "...", "type": "recent_downloads"}
    """

    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None
    last_month = data.get("last_month")
    if not isinstance(last_month, int):
        return None
    return last_month


# ---------------------------------------------------------------------------
# npm registry signals (audit Move #22)
# ---------------------------------------------------------------------------


_NPM_USER_AGENT = "sourcing-agent/1.0 (oss-maintainers-module; npm-signal-fetch)"
_NPM_DOWNLOADS_LAST_MONTH_URL = (
    "https://api.npmjs.org/downloads/point/last-month/{package}"
)


async def fetch_npm_recent_downloads(
    package: str,
    *,
    throttle_seconds: float = 1.0,
    use_cache: bool = True,
    fetch_text: Optional["AsyncTextFetcher"] = None,
) -> Optional[int]:
    """Return the last-month npm download count for ``package``, or None.

    Audit Move #22. Cache-aware via :mod:`github.maintainer_signal_cache`
    under the ``npm_recent_downloads`` signal kind. Scoped npm packages
    (``@scope/name``) are URL-encoded once per the npm registry's
    convention.
    """

    package = package.strip()
    if not package:
        return None

    if use_cache:
        cached = mcache.get(package, "_npm", "npm_recent_downloads")
        if cached is not None and isinstance(cached.data, int):
            return cached.data

    fetcher = fetch_text or _fetch_text_default
    url_package = package.replace("/", "%2F") if package.startswith("@") else package
    url = _NPM_DOWNLOADS_LAST_MONTH_URL.format(package=url_package)
    body = await fetcher(url, headers={"User-Agent": _NPM_USER_AGENT})
    if body is None:
        return None

    count = _parse_npm_downloads(body)
    if count is None:
        logger.warning(
            "network_dependents: npm parse failed for %s — registry payload "
            "may have changed shape",
            package,
        )
        return None

    if use_cache:
        mcache.put(package, "_npm", "npm_recent_downloads", count)
    if throttle_seconds > 0:
        await asyncio.sleep(throttle_seconds)
    return count


def _parse_npm_downloads(body: str) -> Optional[int]:
    """Parse `api.npmjs.org/downloads/point/last-month/<pkg>` JSON.

    Expected shape::

        {"downloads": 12345, "start": "...", "end": "...", "package": "..."}
    """

    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if "error" in payload:  # 404-ish response includes {"error": "..."}
        return None
    downloads = payload.get("downloads")
    if not isinstance(downloads, int):
        return None
    return downloads


# ---------------------------------------------------------------------------
# Shared HTTP helper (uniform fetch surface for tests to mock)
# ---------------------------------------------------------------------------


from typing import Awaitable, Callable

AsyncTextFetcher = Callable[..., Awaitable[Optional[str]]]


async def _fetch_text_default(url: str, *, headers: Optional[dict] = None) -> Optional[str]:
    """Default async fetch returning response body text or None on any failure.

    Mirrors :func:`_fetch_html`'s posture (TLS via certifi,
    aiohttp-based, fail-soft on every error path) but generic over
    response content type — both PyPI and npm return JSON, while the
    GitHub HTML scraper has its own helper.
    """

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    try:
        async with aiohttp.ClientSession(
            connector=connector, headers=request_headers, timeout=timeout
        ) as session:
            async with session.get(url) as resp:
                if resp.status == 404:
                    logger.debug("network_dependents: 404 for %s", url)
                    return None
                if resp.status != 200:
                    logger.warning(
                        "network_dependents: HTTP %d for %s", resp.status, url
                    )
                    return None
                return await resp.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning("network_dependents: fetch failed for %s (%s)", url, exc)
        return None
