"""OpenAlex REST client — Slice 2.

OpenAlex is the spine of the Researcher module per Spec Opinion 1: CC0
license, 100K calls/day at 10 req/s polite pool, 250M+ works, 90M+
authors with ROR-IDed institutions and ORCID, h-index + citation count
+ topic concepts on the author object.

This module is a thin REST wrapper that returns parsed dicts. No
orchestration; the acquisition layer in Slice 4 composes calls.

Polite pool semantics:

- Set ``polite_pool_email`` (the recruiter's contact email) so OpenAlex
  routes us through the polite pool (10 req/s). Without it we land in
  the common pool (no SLA, throttled aggressively).

Rate limiting:

- Default minimum spacing 0.1s ⇒ ~10 req/s. The polite pool's hard
  limit is 10 req/s; we stay one request below the ceiling so a burst
  doesn't trigger 429.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests

from researcher.sources._rate_limit import MinSpacingLimiter


OPENALEX_BASE_URL = "https://api.openalex.org"

# Polite-pool ceiling is 10 req/s; we sit at 9 req/s to leave headroom.
_DEFAULT_MIN_SPACING_S = 0.11


@dataclass
class OpenAlexClient:
    """Thin OpenAlex REST client.

    ``polite_pool_email`` should be a real reachable address — OpenAlex
    documents this as the contract for the polite pool's elevated rate
    limit. ``http_get`` is injectable so tests can swap in a recorder.
    """

    polite_pool_email: str = ""
    user_agent: str = "Cloris-Researcher/1.0"
    min_spacing_seconds: float = _DEFAULT_MIN_SPACING_S
    base_url: str = OPENALEX_BASE_URL
    http_get: Any = field(default=requests.get)
    _limiter: MinSpacingLimiter = field(init=False)

    def __post_init__(self) -> None:
        self._limiter = MinSpacingLimiter(
            min_spacing_seconds=self.min_spacing_seconds
        )

    # -----------------------------------------------------------------
    # Low-level GET
    # -----------------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        """Rate-limited GET against the OpenAlex API.

        Always injects the polite-pool ``mailto`` parameter when
        configured; sets a User-Agent so OpenAlex's ops can identify
        Cloris in their logs.
        """

        merged_params = dict(params or {})
        if self.polite_pool_email:
            merged_params.setdefault("mailto", self.polite_pool_email)

        url = f"{self.base_url}{path}"
        if merged_params:
            url = f"{url}?{urlencode(merged_params, doseq=True)}"

        self._limiter.wait()
        response = self.http_get(
            url,
            headers={"User-Agent": self.user_agent},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    # -----------------------------------------------------------------
    # Authors
    # -----------------------------------------------------------------

    def search_authors(
        self,
        *,
        concept_ids: list[str] | None = None,
        country_codes: list[str] | None = None,
        min_works: int | None = None,
        min_citations: int | None = None,
        cursor: str = "*",
        per_page: int = 25,
    ) -> dict:
        """Search OpenAlex authors with the given filters.

        Returns the raw OpenAlex envelope ``{meta, results}`` where
        ``meta.next_cursor`` paginates and ``results`` is the list of
        author objects (each carrying ``id``, ``orcid``, ``display_name``,
        ``works_count``, ``cited_by_count``, ``summary_stats``,
        ``last_known_institutions``, ``x_concepts``).

        Filters compose via OpenAlex's ``filter=k:v,k2:v2`` syntax.
        """

        filters = _format_author_filters(
            concept_ids=concept_ids,
            country_codes=country_codes,
            min_works=min_works,
            min_citations=min_citations,
        )
        params: dict[str, Any] = {
            "per-page": per_page,
            "cursor": cursor,
        }
        if filters:
            params["filter"] = filters
        return self.get("/authors", params)

    def get_author(self, author_id: str) -> dict:
        """Fetch a single author by OpenAlex ID (e.g., ``A1234567890``).

        The id may be the bare OpenAlex id, the full URL, or an ORCID
        URL — OpenAlex resolves all three at the ``/authors/{id}``
        endpoint.
        """

        return self.get(f"/authors/{author_id}")

    # -----------------------------------------------------------------
    # Works (for venue-filtered discovery + h-index validation)
    # -----------------------------------------------------------------

    def search_works(
        self,
        *,
        concept_ids: list[str] | None = None,
        venue_ids: list[str] | None = None,
        author_id: str | None = None,
        since_year: int | None = None,
        min_citations: int | None = None,
        cursor: str = "*",
        per_page: int = 25,
    ) -> dict:
        """Search OpenAlex works with the given filters.

        Used by the acquisition layer (Slice 4) for venue-driven author
        discovery: query works at NeurIPS / ICML / ICLR; aggregate
        author IDs from the results.
        """

        filters = _format_work_filters(
            concept_ids=concept_ids,
            venue_ids=venue_ids,
            author_id=author_id,
            since_year=since_year,
            min_citations=min_citations,
        )
        params: dict[str, Any] = {
            "per-page": per_page,
            "cursor": cursor,
        }
        if filters:
            params["filter"] = filters
        return self.get("/works", params)

    def get_works_by_author(
        self,
        author_id: str,
        *,
        since_year: int | None = None,
        per_page: int = 25,
        cursor: str = "*",
    ) -> dict:
        """Convenience wrapper for ``search_works(author_id=...)``."""

        return self.search_works(
            author_id=author_id,
            since_year=since_year,
            per_page=per_page,
            cursor=cursor,
        )


def _format_author_filters(
    *,
    concept_ids: list[str] | None,
    country_codes: list[str] | None,
    min_works: int | None,
    min_citations: int | None,
) -> str:
    parts: list[str] = []
    if concept_ids:
        parts.append("concepts.id:" + "|".join(concept_ids))
    if country_codes:
        parts.append(
            "last_known_institutions.country_code:" + "|".join(country_codes)
        )
    if min_works is not None:
        parts.append(f"works_count:>{int(min_works) - 1}")
    if min_citations is not None:
        parts.append(f"cited_by_count:>{int(min_citations) - 1}")
    return ",".join(parts)


def _format_work_filters(
    *,
    concept_ids: list[str] | None,
    venue_ids: list[str] | None,
    author_id: str | None,
    since_year: int | None,
    min_citations: int | None,
) -> str:
    parts: list[str] = []
    if concept_ids:
        parts.append("concepts.id:" + "|".join(concept_ids))
    if venue_ids:
        parts.append("locations.source.id:" + "|".join(venue_ids))
    if author_id:
        parts.append(f"author.id:{author_id}")
    if since_year is not None:
        parts.append(f"from_publication_date:{int(since_year)}-01-01")
    if min_citations is not None:
        parts.append(f"cited_by_count:>{int(min_citations) - 1}")
    return ",".join(parts)
