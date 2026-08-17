"""News-API signal adapter for executive-search dossier evaluation.

Slice 4 of the executive-search module. Pulls executive-movement news
mentions for a candidate's company / role for a configurable window
(``brief.executive_movement_window_days``, default 180). Provider is
NewsAPI.org (chosen for v1 per spec — cheap, broad, decent recall;
trade-off: lower quality than Bing News). Adapter interface
deliberately abstracts the provider so a future swap is one-file.

Caching:

- Per-state-dir 7-day TTL JSONL at
  ``output/state/exec_search/<state_key>/news_cache.jsonl``.
- Cache key: ``(query, window_days)`` hash. Cache hits avoid API
  spend; misses populate the cache after a successful fetch.
- 7 days is "long enough that a per-search dossier eval doesn't
  re-fetch, short enough that exec movement is ~current."

Failure modes:

- No API key configured: :class:`SignalFailure(reason="disabled_no_api_key")`
- HTTP / quota / timeout: :class:`SignalFailure` with a typed reason
  matching the provider's error class. NEVER raises out to callers.
- Empty result set: :class:`SignalResult` with a "no news found"
  placeholder section. The dossier eval reads this and reasons
  honestly about the gap.

Slice 4 ships the adapter, cache, and tests against fixture HTTP.
Slices 5-10 may extend the per-source telemetry surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from shared.brief_schema import Brief
from shared.schemas import CandidateProfileSummary

from exec_search.signals import SignalFailure, SignalRequestContext, SignalResult


# --------------------------------------------------------------------------
# Provider abstraction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NewsArticle:
    """One news article from any provider, normalized."""

    title: str
    description: str
    url: str
    published_at: str
    source_name: str = ""


class NewsApiClient:
    """Thin NewsAPI.org client. Slice 4's only provider.

    Replaceable: any class with a ``fetch(query, from_iso, to_iso) ->
    list[NewsArticle]`` shape works. Swapping providers is a one-line
    change to the adapter constructor; the dossier evidence layer is
    provider-agnostic.
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str, *, http_session: Any | None = None) -> None:
        self.api_key = api_key
        self._session = http_session

    def fetch(
        self,
        *,
        query: str,
        from_iso: str,
        to_iso: str,
        page_size: int = 25,
    ) -> list[NewsArticle]:
        if self._session is None:
            import urllib.parse
            import urllib.request

            params = urllib.parse.urlencode(
                {
                    "q": query,
                    "from": from_iso,
                    "to": to_iso,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": str(page_size),
                    "apiKey": self.api_key,
                }
            )
            url = f"{self.BASE_URL}?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "Cloris/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status >= 400:
                    raise NewsApiError(
                        f"newsapi http {resp.status}",
                        status_code=resp.status,
                    )
                payload = json.loads(resp.read().decode("utf-8"))
        else:
            response = self._session.get(
                self.BASE_URL,
                params={
                    "q": query,
                    "from": from_iso,
                    "to": to_iso,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": page_size,
                    "apiKey": self.api_key,
                },
                timeout=30,
            )
            if response.status_code >= 400:
                raise NewsApiError(
                    f"newsapi http {response.status_code}",
                    status_code=response.status_code,
                )
            payload = response.json()
        return _parse_news_payload(payload)


class NewsApiError(RuntimeError):
    """NewsAPI.org returned a non-success response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _parse_news_payload(payload: Mapping[str, Any]) -> list[NewsArticle]:
    """Map NewsAPI.org's response shape into NewsArticle list."""

    articles_raw = payload.get("articles", []) if isinstance(payload, Mapping) else []
    out: list[NewsArticle] = []
    if not isinstance(articles_raw, list):
        return out
    for raw in articles_raw:
        if not isinstance(raw, Mapping):
            continue
        source_obj = raw.get("source") or {}
        source_name = ""
        if isinstance(source_obj, Mapping):
            source_name = str(source_obj.get("name") or "")
        out.append(
            NewsArticle(
                title=str(raw.get("title") or ""),
                description=str(raw.get("description") or ""),
                url=str(raw.get("url") or ""),
                published_at=str(raw.get("publishedAt") or ""),
                source_name=source_name,
            )
        )
    return out


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


# 7 days, in seconds. Long enough that a per-search dossier eval
# doesn't re-fetch the same candidate; short enough that "current"
# executive movement stays current.
NEWS_CACHE_TTL_SECONDS: int = 7 * 24 * 3600


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    fetched_at_epoch: float
    articles: list[NewsArticle]


def _cache_key(query: str, window_days: int) -> str:
    """Stable hash of the cache lookup tuple. SHA-256 short prefix.

    Hash is content-stable so a query rewrite (case, whitespace) maps
    to the same key — a query string normalization step happens before
    hashing.
    """

    normalized = " ".join((query or "").split()).lower()
    seed = f"{normalized}|{window_days}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:16]


def load_cached(path: Path, query: str, window_days: int) -> CacheEntry | None:
    """Look up a fresh cache entry, or ``None`` if missing / expired."""

    if not path.exists():
        return None
    target = _cache_key(query, window_days)
    now = time.time()
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("cache_key") != target:
                continue
            fetched_at = float(row.get("fetched_at_epoch") or 0.0)
            if now - fetched_at > NEWS_CACHE_TTL_SECONDS:
                continue
            articles = [
                NewsArticle(
                    title=a.get("title", ""),
                    description=a.get("description", ""),
                    url=a.get("url", ""),
                    published_at=a.get("published_at", ""),
                    source_name=a.get("source_name", ""),
                )
                for a in row.get("articles", [])
                if isinstance(a, Mapping)
            ]
            return CacheEntry(
                cache_key=target,
                fetched_at_epoch=fetched_at,
                articles=articles,
            )
    except OSError:
        return None
    return None


def save_cached(path: Path, entry: CacheEntry) -> None:
    """Append a cache entry to the JSONL file (best effort)."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "cache_key": entry.cache_key,
            "fetched_at_epoch": entry.fetched_at_epoch,
            "articles": [
                {
                    "title": a.title,
                    "description": a.description,
                    "url": a.url,
                    "published_at": a.published_at,
                    "source_name": a.source_name,
                }
                for a in entry.articles
            ],
        }
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------


@dataclass
class NewsSignalSource:
    """Off-LinkedIn news signal source backed by NewsAPI.org."""

    name: str = "news"
    client: NewsApiClient | None = None
    cache_path_resolver: Any = None  # callable: (state_dir: Path) -> Path

    def fetch(
        self,
        *,
        candidate: CandidateProfileSummary,
        brief: Brief,
        context: SignalRequestContext,
    ) -> SignalResult | SignalFailure:
        api_key = (os.environ.get("NEWSAPI_KEY") or "").strip()
        if self.client is None and not api_key:
            return SignalFailure(
                source=self.name,
                reason="disabled_no_api_key",
                detail="NEWSAPI_KEY not set",
            )
        client = self.client or NewsApiClient(api_key=api_key)

        window_days = int(getattr(brief, "executive_movement_window_days", 180) or 180)
        query = _build_query(candidate)
        if not query:
            return SignalFailure(
                source=self.name,
                reason="empty_query",
                detail="candidate has no name + company to search on",
            )

        cache_path = _resolve_cache_path(brief, context, self.cache_path_resolver)
        cached = load_cached(cache_path, query, window_days)
        articles: list[NewsArticle]
        cache_hit = False
        if cached is not None:
            articles = cached.articles
            cache_hit = True
        else:
            now = time.time()
            from_iso, to_iso = _window_to_iso(window_days, now)
            try:
                articles = client.fetch(query=query, from_iso=from_iso, to_iso=to_iso)
            except NewsApiError as exc:
                return SignalFailure(
                    source=self.name,
                    reason="upstream_error",
                    detail=str(exc),
                )
            except Exception as exc:
                return SignalFailure(
                    source=self.name,
                    reason="adapter_exception",
                    detail=f"{exc.__class__.__name__}: {exc}",
                )
            save_cached(
                cache_path,
                CacheEntry(
                    cache_key=_cache_key(query, window_days),
                    fetched_at_epoch=now,
                    articles=articles,
                ),
            )

        section_text = _format_news_section(
            query=query,
            window_days=window_days,
            articles=articles,
            cache_hit=cache_hit,
        )
        citations = tuple(a.url for a in articles if a.url)
        return SignalResult(
            source=self.name,
            section_text=section_text,
            citations=citations,
        )


def _build_query(candidate: CandidateProfileSummary) -> str:
    """Compose a recruiter-readable news query from candidate identity.

    Format: ``"<name>" AND ("<company1>" OR "<company2>" ...)``.

    Falls back to name-only if no company is detectable. Returns
    empty string if neither name nor company is present (caller
    treats this as ``empty_query``).
    """

    name = (candidate.name or "").strip()
    companies = []
    for exp in (candidate.experiences or [])[:3]:
        company = (getattr(exp, "company", "") or "").strip()
        if company and company not in companies:
            companies.append(company)
    if name and companies:
        company_clause = " OR ".join(f'"{c}"' for c in companies)
        return f'"{name}" AND ({company_clause})'
    if name:
        return f'"{name}"'
    if companies:
        return " OR ".join(f'"{c}"' for c in companies)
    return ""


def _window_to_iso(window_days: int, now_epoch: float) -> tuple[str, str]:
    """Return ``(from_iso, to_iso)`` for the news window.

    NewsAPI.org accepts ISO-8601 dates. We use date-only (UTC) so the
    cache key stays stable across same-day calls.
    """

    import datetime as _dt

    end = _dt.datetime.fromtimestamp(now_epoch, tz=_dt.timezone.utc).date()
    start = end - _dt.timedelta(days=int(window_days))
    return start.isoformat(), end.isoformat()


def _resolve_cache_path(
    brief: Brief,
    context: SignalRequestContext,
    override: Any | None,
) -> Path:
    """Pick the per-state-dir cache JSONL path.

    Default: ``output/state/exec_search/<brief_id>/news_cache.jsonl``.
    Tests pass a callable ``override`` that returns a path under
    ``tmp_path`` so the suite doesn't write to the canonical state
    dir.
    """

    if override is not None:
        return Path(override(brief, context))
    from shared.output_paths import resolve_exec_search_state_dir

    state_dir = resolve_exec_search_state_dir(
        brief_path=Path("brief.json"),  # tolerated when state_dir derives from brief
        brief=None,
    )
    return state_dir / "news_cache.jsonl"


def _format_news_section(
    *,
    query: str,
    window_days: int,
    articles: list[NewsArticle],
    cache_hit: bool,
) -> str:
    """Render news articles into a recruiter-readable section."""

    if not articles:
        return (
            f"News mentions ({window_days}-day window):\n"
            f"  [no news mentions found for query {query}]"
        )
    cache_marker = " (cached)" if cache_hit else ""
    lines = [
        f"News mentions ({window_days}-day window){cache_marker}:",
    ]
    for article in articles[:10]:
        title = article.title or "[untitled]"
        published = article.published_at[:10] if article.published_at else ""
        source = article.source_name or "[unknown]"
        lines.append(f"  - {published} {source}: {title}")
        if article.description:
            lines.append(f"      {article.description}")
        if article.url:
            lines.append(f"      {article.url}")
    return "\n".join(lines)


__all__ = (
    "CacheEntry",
    "NEWS_CACHE_TTL_SECONDS",
    "NewsApiClient",
    "NewsApiError",
    "NewsArticle",
    "NewsSignalSource",
    "load_cached",
    "save_cached",
)
