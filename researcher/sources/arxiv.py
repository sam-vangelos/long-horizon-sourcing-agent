"""arXiv REST client — Slice 2.

arXiv supplies preprint discovery to complement OpenAlex's published-work
focus. Atom XML responses (not JSON); rate limit is one request per
three seconds per the arXiv terms of service.

We use the ``query`` interface at ``http://export.arxiv.org/api/query``.
Filters supported in v1:

- ``cs.LG`` / ``cs.CL`` / ``cs.AI`` (and other) categories
- author last-name match
- date window

Returns parsed dicts (XML → dict via stdlib ElementTree). The acquisition
layer (Slice 4) uses this for preprint-driven discovery and recency
scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import requests

from researcher.sources._rate_limit import MinSpacingLimiter


ARXIV_BASE_URL = "http://export.arxiv.org/api/query"

# arXiv ToS: 1 request per 3 seconds. Sit at 3.1s to be safe.
_DEFAULT_MIN_SPACING_S = 3.1

# Atom XML namespaces.
_ATOM_NS = "http://www.w3.org/2005/Atom"
_OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


@dataclass
class ArxivClient:
    """Thin arXiv API client."""

    user_agent: str = "Cloris-Researcher/1.0 (mailto:hello@cloris.ai)"
    min_spacing_seconds: float = _DEFAULT_MIN_SPACING_S
    base_url: str = ARXIV_BASE_URL
    http_get: Any = field(default=requests.get)
    _limiter: MinSpacingLimiter = field(init=False)

    def __post_init__(self) -> None:
        self._limiter = MinSpacingLimiter(
            min_spacing_seconds=self.min_spacing_seconds
        )

    def search(
        self,
        *,
        categories: list[str] | None = None,
        author_lastname: str | None = None,
        full_text_query: str | None = None,
        max_results: int = 25,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> dict:
        """Run an arXiv API query; return ``{total_results, entries}``.

        ``entries`` is a list of dicts, each: ``{id, title, summary,
        published, updated, authors, primary_category, categories,
        arxiv_id}``.
        """

        search_query = _build_search_query(
            categories=categories,
            author_lastname=author_lastname,
            full_text_query=full_text_query,
        )
        params: dict[str, Any] = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        url = f"{self.base_url}?{urlencode(params, doseq=True)}"

        self._limiter.wait()
        response = self.http_get(
            url,
            headers={"User-Agent": self.user_agent},
            timeout=60,
        )
        response.raise_for_status()
        return _parse_atom_feed(response.text)


def _build_search_query(
    *,
    categories: list[str] | None,
    author_lastname: str | None,
    full_text_query: str | None,
) -> str:
    parts: list[str] = []
    if categories:
        cat_terms = " OR ".join(f"cat:{c}" for c in categories)
        parts.append(f"({cat_terms})")
    if author_lastname:
        parts.append(f'au:"{author_lastname}"')
    if full_text_query:
        parts.append(f"all:{full_text_query}")
    if not parts:
        # Empty query is invalid — default to a wide AI category sweep
        # that the caller can paginate. Slice 4 callers always supply
        # at least one filter; this is defensive.
        return "cat:cs.AI"
    return " AND ".join(parts)


def _parse_atom_feed(xml_text: str) -> dict:
    """Parse an arXiv Atom feed into ``{total_results, entries}``."""

    root = ET.fromstring(xml_text)
    total_results_node = root.find(f"{{{_OPENSEARCH_NS}}}totalResults")
    total_results = (
        int(total_results_node.text or "0") if total_results_node is not None else 0
    )

    entries: list[dict] = []
    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        entries.append(_parse_entry(entry))

    return {"total_results": total_results, "entries": entries}


def _parse_entry(entry: ET.Element) -> dict:
    def _text(tag: str) -> str:
        node = entry.find(f"{{{_ATOM_NS}}}{tag}")
        return (node.text or "").strip() if node is not None else ""

    full_id = _text("id")
    arxiv_id = full_id.rsplit("/", 1)[-1] if full_id else ""

    authors: list[str] = []
    for author in entry.findall(f"{{{_ATOM_NS}}}author"):
        name_node = author.find(f"{{{_ATOM_NS}}}name")
        if name_node is not None and name_node.text:
            authors.append(name_node.text.strip())

    primary_cat_node = entry.find(f"{{{_ARXIV_NS}}}primary_category")
    primary_category = (
        primary_cat_node.attrib.get("term", "") if primary_cat_node is not None else ""
    )

    categories: list[str] = []
    for cat in entry.findall(f"{{{_ATOM_NS}}}category"):
        term = cat.attrib.get("term")
        if term:
            categories.append(term)

    return {
        "id": full_id,
        "arxiv_id": arxiv_id,
        "title": _text("title"),
        "summary": _text("summary"),
        "published": _text("published"),
        "updated": _text("updated"),
        "authors": authors,
        "primary_category": primary_category,
        "categories": categories,
    }
