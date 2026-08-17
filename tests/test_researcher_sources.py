"""Researcher module Slice 2 — source-client coverage.

Three thin REST wrappers (OpenAlex, Semantic Scholar, arXiv); each test
uses a stub HTTP getter (no real network) and an injected fake clock
to assert per-source rate-limit spacing without sleeping.

Per Researcher Module Spec Slice 2: rate-limit honored at the source
boundary. The existing `shared.rate_limiter` is GitHub-specific; this
slice ships per-source minimum-spacing limiters inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from researcher.sources._rate_limit import MinSpacingLimiter
from researcher.sources.arxiv import ArxivClient
from researcher.sources.openalex import OpenAlexClient
from researcher.sources.semantic_scholar import SemanticScholarClient


# ---------------------------------------------------------------------------
# Fake HTTP recorder + fake clock
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self, *, json_payload: dict | None = None, text_payload: str = ""
    ) -> None:
        self._json = json_payload or {}
        self.text = text_payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._json


@dataclass
class _HttpRecorder:
    """Records every (url, headers) call; returns canned responses in order."""

    responses: list[Any] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    _idx: int = 0

    def __call__(self, url: str, *, headers: dict | None = None, timeout: int = 30):
        self.calls.append(
            {"url": url, "headers": dict(headers or {}), "timeout": timeout}
        )
        if self._idx >= len(self.responses):
            raise AssertionError(
                f"_HttpRecorder: no canned response for call #{self._idx + 1} "
                f"to {url!r}"
            )
        response = self.responses[self._idx]
        self._idx += 1
        return response


@dataclass
class _FakeClock:
    """Monotonic clock + sleep stub. Tracks total sleep time."""

    now: float = 0.0
    slept_total: float = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            return
        self.slept_total += seconds
        self.now += seconds


# ---------------------------------------------------------------------------
# MinSpacingLimiter
# ---------------------------------------------------------------------------


def test_min_spacing_limiter_first_call_does_not_sleep() -> None:
    clock = _FakeClock()
    limiter = MinSpacingLimiter(
        min_spacing_seconds=1.0,
        _now=clock.time,
        _sleep=clock.sleep,
    )
    waited = limiter.wait()
    assert waited == 0.0
    assert clock.slept_total == 0.0


def test_min_spacing_limiter_back_to_back_calls_sleep_for_remainder() -> None:
    clock = _FakeClock()
    limiter = MinSpacingLimiter(
        min_spacing_seconds=1.0,
        _now=clock.time,
        _sleep=clock.sleep,
    )
    limiter.wait()  # t=0; first call free
    clock.now = 0.3  # 300ms elapsed
    limiter.wait()  # second call: 700ms remaining
    assert clock.slept_total == 0.7


def test_min_spacing_limiter_long_gap_no_sleep() -> None:
    clock = _FakeClock()
    limiter = MinSpacingLimiter(
        min_spacing_seconds=1.0,
        _now=clock.time,
        _sleep=clock.sleep,
    )
    limiter.wait()
    clock.now = 5.0  # plenty of time elapsed
    limiter.wait()
    assert clock.slept_total == 0.0


# ---------------------------------------------------------------------------
# OpenAlexClient
# ---------------------------------------------------------------------------


def test_openalex_search_authors_builds_filter_string_and_includes_polite_pool() -> None:
    recorder = _HttpRecorder(
        responses=[
            _FakeResponse(
                json_payload={
                    "meta": {"next_cursor": "abc123"},
                    "results": [
                        {"id": "https://openalex.org/A1234", "display_name": "Jane R."}
                    ],
                }
            )
        ]
    )
    client = OpenAlexClient(
        polite_pool_email="hello@cloris.ai",
        http_get=recorder,
        min_spacing_seconds=0.0,
    )

    response = client.search_authors(
        concept_ids=["C2778407487", "C41008148"],
        country_codes=["US", "GB"],
        min_works=10,
        min_citations=100,
        per_page=25,
    )

    assert response["meta"]["next_cursor"] == "abc123"
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    url = call["url"]
    assert url.startswith("https://api.openalex.org/authors?")
    assert "mailto=hello%40cloris.ai" in url
    assert "per-page=25" in url
    assert "cursor=%2A" in url  # cursor=*
    # Filters use OR (|) for set membership, AND (,) between filters.
    assert "concepts.id%3AC2778407487%7CC41008148" in url
    assert "last_known_institutions.country_code%3AUS%7CGB" in url
    assert "works_count%3A%3E9" in url  # min_works 10 → >9
    assert "cited_by_count%3A%3E99" in url
    assert call["headers"]["User-Agent"].startswith("Cloris-Researcher")


def test_openalex_get_author_targets_id_path() -> None:
    recorder = _HttpRecorder(
        responses=[_FakeResponse(json_payload={"id": "A1", "display_name": "X"})]
    )
    client = OpenAlexClient(
        polite_pool_email="hello@cloris.ai",
        http_get=recorder,
        min_spacing_seconds=0.0,
    )
    response = client.get_author("A1")
    assert response["id"] == "A1"
    assert recorder.calls[0]["url"] == (
        "https://api.openalex.org/authors/A1?mailto=hello%40cloris.ai"
    )


def test_openalex_search_works_supports_venue_and_year_filters() -> None:
    recorder = _HttpRecorder(
        responses=[_FakeResponse(json_payload={"meta": {}, "results": []})]
    )
    client = OpenAlexClient(
        polite_pool_email="hello@cloris.ai",
        http_get=recorder,
        min_spacing_seconds=0.0,
    )
    client.search_works(
        venue_ids=["S1234"],
        since_year=2023,
        min_citations=20,
    )
    url = recorder.calls[0]["url"]
    assert "locations.source.id%3AS1234" in url
    assert "from_publication_date%3A2023-01-01" in url
    assert "cited_by_count%3A%3E19" in url


def test_openalex_rate_limiter_spaces_calls_at_polite_pool_ceiling() -> None:
    """Two back-to-back calls should sleep for the configured spacing."""

    clock = _FakeClock()
    recorder = _HttpRecorder(
        responses=[
            _FakeResponse(json_payload={"meta": {}, "results": []}),
            _FakeResponse(json_payload={"meta": {}, "results": []}),
        ]
    )
    client = OpenAlexClient(
        polite_pool_email="hello@cloris.ai",
        http_get=recorder,
        min_spacing_seconds=0.11,  # default polite-pool spacing
    )
    # Inject the fake clock into the post_init-built limiter.
    client._limiter = MinSpacingLimiter(
        min_spacing_seconds=0.11,
        _now=clock.time,
        _sleep=clock.sleep,
    )

    client.search_authors(concept_ids=["C1"])
    client.search_authors(concept_ids=["C1"])

    assert clock.slept_total == 0.11


# ---------------------------------------------------------------------------
# SemanticScholarClient
# ---------------------------------------------------------------------------


def test_semantic_scholar_get_author_includes_default_fields_and_no_api_key_header() -> None:
    recorder = _HttpRecorder(
        responses=[
            _FakeResponse(
                json_payload={"authorId": "12345", "name": "X", "hIndex": 14}
            )
        ]
    )
    client = SemanticScholarClient(
        http_get=recorder,
        min_spacing_seconds=0.0,
    )
    response = client.get_author("12345")
    assert response["hIndex"] == 14
    url = recorder.calls[0]["url"]
    assert url.startswith("https://api.semanticscholar.org/graph/v1/author/12345?")
    assert "fields=hIndex%2CcitationCount%2CpaperCount%2Cname%2Caffiliations" in url
    # Free tier: no x-api-key header.
    assert "x-api-key" not in recorder.calls[0]["headers"]


def test_semantic_scholar_with_api_key_sends_x_api_key_header() -> None:
    recorder = _HttpRecorder(
        responses=[_FakeResponse(json_payload={"authorId": "1"})]
    )
    client = SemanticScholarClient(
        api_key="secret",
        http_get=recorder,
        min_spacing_seconds=0.0,
    )
    client.get_author("1", fields=["name"])
    headers = recorder.calls[0]["headers"]
    assert headers.get("x-api-key") == "secret"


def test_semantic_scholar_rate_limiter_spaces_calls_at_one_per_second() -> None:
    clock = _FakeClock()
    recorder = _HttpRecorder(
        responses=[
            _FakeResponse(json_payload={"authorId": "1"}),
            _FakeResponse(json_payload={"authorId": "2"}),
        ]
    )
    client = SemanticScholarClient(
        http_get=recorder,
        min_spacing_seconds=1.05,
    )
    client._limiter = MinSpacingLimiter(
        min_spacing_seconds=1.05,
        _now=clock.time,
        _sleep=clock.sleep,
    )

    client.get_author("1", fields=["name"])
    client.get_author("2", fields=["name"])

    assert clock.slept_total == 1.05


# ---------------------------------------------------------------------------
# ArxivClient
# ---------------------------------------------------------------------------


_ARXIV_SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>arXiv Query: cat:cs.LG</title>
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <updated>2024-01-02T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>RLHF: A Survey</title>
    <summary>We survey RLHF...</summary>
    <author><name>Alice Researcher</name></author>
    <author><name>Bob Reviewer</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.LG"/>
    <category term="cs.LG"/>
    <category term="cs.CL"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <updated>2024-01-03T00:00:00Z</updated>
    <published>2024-01-02T00:00:00Z</published>
    <title>Inference Systems</title>
    <summary>Inference cost matters...</summary>
    <author><name>Carol Compiler</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.AI"/>
    <category term="cs.AI"/>
  </entry>
</feed>
"""


def test_arxiv_search_parses_atom_into_entries_and_total() -> None:
    recorder = _HttpRecorder(
        responses=[_FakeResponse(text_payload=_ARXIV_SAMPLE_FEED)]
    )
    client = ArxivClient(
        http_get=recorder,
        min_spacing_seconds=0.0,
    )

    response = client.search(categories=["cs.LG", "cs.CL"], max_results=2)

    assert response["total_results"] == 2
    assert len(response["entries"]) == 2

    first = response["entries"][0]
    assert first["arxiv_id"] == "2401.00001v1"
    assert first["title"] == "RLHF: A Survey"
    assert first["authors"] == ["Alice Researcher", "Bob Reviewer"]
    assert first["primary_category"] == "cs.LG"
    assert "cs.LG" in first["categories"]

    url = recorder.calls[0]["url"]
    assert url.startswith("http://export.arxiv.org/api/query?")
    # Categories OR-joined inside the search_query.
    assert "search_query=%28cat%3Acs.LG+OR+cat%3Acs.CL%29" in url


def test_arxiv_search_combines_categories_with_author_lastname() -> None:
    recorder = _HttpRecorder(
        responses=[_FakeResponse(text_payload=_ARXIV_SAMPLE_FEED)]
    )
    client = ArxivClient(
        http_get=recorder,
        min_spacing_seconds=0.0,
    )
    client.search(categories=["cs.AI"], author_lastname="Smith", max_results=10)
    url = recorder.calls[0]["url"]
    # Categories AND author filter.
    assert "%28cat%3Acs.AI%29+AND+au%3A%22Smith%22" in url


def test_arxiv_rate_limiter_spaces_calls_at_3_seconds() -> None:
    clock = _FakeClock()
    recorder = _HttpRecorder(
        responses=[
            _FakeResponse(text_payload=_ARXIV_SAMPLE_FEED),
            _FakeResponse(text_payload=_ARXIV_SAMPLE_FEED),
        ]
    )
    client = ArxivClient(
        http_get=recorder,
        min_spacing_seconds=3.1,
    )
    client._limiter = MinSpacingLimiter(
        min_spacing_seconds=3.1,
        _now=clock.time,
        _sleep=clock.sleep,
    )

    client.search(categories=["cs.LG"], max_results=2)
    client.search(categories=["cs.LG"], max_results=2)

    assert clock.slept_total == 3.1
