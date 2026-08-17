"""Semantic Scholar REST client — Slice 2.

Semantic Scholar enriches the OpenAlex spine per Spec Opinion 1: paper
similarity (via the SPECTER2 embedding) for "find me authors of papers
like X" queries, and h-index cross-validation when OpenAlex's count
looks suspect.

Free tier: 1 request per second sustained. We sit at 1 req / 1.05 sec
to leave headroom. API key is optional (free tier works without one);
when present, it raises the rate ceiling.

This module is a thin REST wrapper that returns parsed dicts. No
orchestration; the acquisition/evaluation layers compose calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests

from researcher.sources._rate_limit import MinSpacingLimiter


SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Free-tier ceiling is 1 req/s; sit just under to avoid 429 bursts.
_DEFAULT_MIN_SPACING_S = 1.05


@dataclass
class SemanticScholarClient:
    """Thin Semantic Scholar REST client."""

    api_key: str = ""
    user_agent: str = "Cloris-Researcher/1.0"
    min_spacing_seconds: float = _DEFAULT_MIN_SPACING_S
    base_url: str = SEMANTIC_SCHOLAR_BASE_URL
    http_get: Any = field(default=requests.get)
    _limiter: MinSpacingLimiter = field(init=False)

    def __post_init__(self) -> None:
        self._limiter = MinSpacingLimiter(
            min_spacing_seconds=self.min_spacing_seconds
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        """Rate-limited GET."""

        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"

        headers = {"User-Agent": self.user_agent}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        self._limiter.wait()
        response = self.http_get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_author(self, author_id: str, *, fields: list[str] | None = None) -> dict:
        """Fetch a Semantic Scholar author by ID.

        ``fields`` selects which payload keys to return; default returns
        the keys we use for h-index cross-validation: ``hIndex``,
        ``citationCount``, ``paperCount``, ``name``, ``affiliations``.
        """

        params: dict[str, Any] = {}
        if fields is None:
            fields = [
                "hIndex",
                "citationCount",
                "paperCount",
                "name",
                "affiliations",
            ]
        if fields:
            params["fields"] = ",".join(fields)
        return self.get(f"/author/{author_id}", params)

    def search_authors(
        self,
        query: str,
        *,
        limit: int = 10,
        fields: list[str] | None = None,
    ) -> dict:
        """Fuzzy author search by name.

        Useful for cross-source resolution when ORCID is absent.
        """

        params: dict[str, Any] = {"query": query, "limit": limit}
        if fields:
            params["fields"] = ",".join(fields)
        return self.get("/author/search", params)

    def get_paper(self, paper_id: str, *, fields: list[str] | None = None) -> dict:
        """Fetch a paper by ID (DOI / arXiv ID / S2 paper ID)."""

        params: dict[str, Any] = {}
        if fields:
            params["fields"] = ",".join(fields)
        return self.get(f"/paper/{paper_id}", params)

    def get_paper_recommendations(
        self,
        paper_id: str,
        *,
        limit: int = 10,
        fields: list[str] | None = None,
    ) -> dict:
        """SPECTER2-driven paper-similarity recommendations.

        The recommendations endpoint returns papers semantically similar
        to the seed paper. The acquisition layer (Slice 4) uses this for
        "find authors of papers like X" expansion.
        """

        params: dict[str, Any] = {"limit": limit}
        if fields:
            params["fields"] = ",".join(fields)
        return self.get(
            f"/recommendations/v1/papers/forpaper/{paper_id}",
            params,
        )
