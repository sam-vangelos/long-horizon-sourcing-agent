"""Tests for Executive Search Slice 4 — News API integration.

Pins the contract:

- :class:`NewsSignalSource` returns
  :class:`SignalFailure(reason="disabled_no_api_key")` when neither
  the env var nor an injected client is present.
- A successful fetch caches articles per
  ``(query, window_days)`` to a JSONL at the per-state-dir path.
- A second call within :data:`NEWS_CACHE_TTL_SECONDS` reads from the
  cache (no second HTTP call).
- A stale cache (> TTL) re-fetches.
- Provider HTTP errors degrade to :class:`SignalFailure(reason="upstream_error")`.
- Empty result sets render a "no news found" placeholder rather
  than disappearing silently.
- :func:`_build_query` builds a quoted-name-AND-company query and
  falls back to name-only when no company is present.
- The registry exposes ``"news"`` after Slice 4 lands.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from exec_search.signals import (
    SIGNAL_REGISTRY,
    SignalFailure,
    SignalRequestContext,
    SignalResult,
    known_signal_sources,
)
from exec_search.signals.news import (
    NEWS_CACHE_TTL_SECONDS,
    NewsApiClient,
    NewsApiError,
    NewsArticle,
    NewsSignalSource,
    _build_query,
    _cache_key,
    _parse_news_payload,
    load_cached,
    save_cached,
)
from shared.brief_schema import (
    Brief,
    CapabilityArea,
    DepthDistinction,
    FacialCalibration,
    MarketDensity,
)
from shared.schemas import CandidateProfileSummary, Experience


def _exec_brief(window_days: int = 90) -> Brief:
    return Brief(
        role_title="VP Engineering",
        role_level="Executive",
        role_summary="Owns engineering leadership.",
        geography="United States",
        linkedin_project="exec",
        capability_areas=[
            CapabilityArea(
                name="x",
                description="y",
                builder_signals=["z"],
                user_signals=[],
            )
        ],
        depth_distinction=DepthDistinction(
            builder_definition="a", user_definition="b", edge_case_guidance="c",
        ),
        non_fit_patterns=[],
        employer_signal_rules=[],
        minimum_years_experience=12,
        minimum_bar_description="12+ years.",
        facial_calibration=FacialCalibration(
            expected_yes_rate_low=0.3,
            expected_yes_rate_high=0.5,
            fast_exit_patterns=[],
            trajectory_yes_patterns=[],
            trajectory_ambiguous_patterns=[],
            trajectory_no_patterns=[],
        ),
        market_density=MarketDensity.MODERATE,
        target_modules=["linkedin", "exec_search"],
        executive_movement_window_days=window_days,
    )


def _candidate(name: str = "Jane Doe", company: str = "AcmeCorp") -> CandidateProfileSummary:
    return CandidateProfileSummary(
        name=name,
        headline=f"VP at {company}",
        profile_url="https://linkedin.com/in/jane",
        experiences=[
            Experience(
                title="VP Engineering",
                company=company,
                start="2020",
                end="present",
                summary_bullets=[],
            ),
        ],
    )


def _stub_client(articles: list[NewsArticle]) -> NewsApiClient:
    """Return a NewsApiClient stub whose ``fetch`` returns ``articles``
    and counts invocations."""

    client = NewsApiClient(api_key="stub")
    client.fetch = MagicMock(return_value=articles)  # type: ignore[method-assign]
    return client


# ---------------------------------------------------------------------------
# Registry membership
# ---------------------------------------------------------------------------


def test_registry_includes_news_after_slice_4() -> None:
    assert "news" in SIGNAL_REGISTRY
    assert "news" in known_signal_sources()


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def test_build_query_combines_name_and_company() -> None:
    cand = _candidate(name="Jane Doe", company="AcmeCorp")
    q = _build_query(cand)
    assert q == '"Jane Doe" AND ("AcmeCorp")'


def test_build_query_falls_back_to_name_only() -> None:
    cand = CandidateProfileSummary(
        name="Solo Person",
        headline="self-employed",
        profile_url="https://linkedin.com/in/solo",
    )
    q = _build_query(cand)
    assert q == '"Solo Person"'


def test_build_query_falls_back_to_empty_when_no_identity() -> None:
    cand = CandidateProfileSummary(
        name="",
        headline="",
        profile_url="",
    )
    q = _build_query(cand)
    assert q == ""


# ---------------------------------------------------------------------------
# Adapter — config gate
# ---------------------------------------------------------------------------


def test_adapter_returns_disabled_no_api_key_when_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    source = NewsSignalSource(
        cache_path_resolver=lambda b, ctx: tmp_path / "news_cache.jsonl",
    )
    result = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(result, SignalFailure)
    assert result.reason == "disabled_no_api_key"


# ---------------------------------------------------------------------------
# Adapter — successful fetch + cache miss
# ---------------------------------------------------------------------------


def test_adapter_fetches_and_caches_on_first_call(tmp_path: Path) -> None:
    cache_path = tmp_path / "news_cache.jsonl"
    articles = [
        NewsArticle(
            title="Doe joins AcmeCorp as VP Engineering",
            description="The company announced...",
            url="https://news.example.com/1",
            published_at="2026-04-15T12:00:00Z",
            source_name="ExampleNews",
        ),
    ]
    client = _stub_client(articles)
    source = NewsSignalSource(
        client=client,
        cache_path_resolver=lambda b, ctx: cache_path,
    )

    result = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(window_days=90),
        context=SignalRequestContext(brief_id="b1"),
    )

    assert isinstance(result, SignalResult)
    assert result.source == "news"
    assert "Doe joins AcmeCorp" in result.section_text
    assert "https://news.example.com/1" in result.citations
    # Cache file written.
    assert cache_path.exists()
    # Provider was called once.
    assert client.fetch.call_count == 1


def test_adapter_renders_cache_hit_marker_on_second_call(tmp_path: Path) -> None:
    cache_path = tmp_path / "news_cache.jsonl"
    articles = [
        NewsArticle(
            title="Doe joins AcmeCorp as VP Engineering",
            description="",
            url="https://news.example.com/1",
            published_at="2026-04-15T12:00:00Z",
            source_name="ExampleNews",
        ),
    ]
    client = _stub_client(articles)
    source = NewsSignalSource(
        client=client,
        cache_path_resolver=lambda b, ctx: cache_path,
    )
    cand = _candidate()
    brief = _exec_brief(window_days=90)
    ctx = SignalRequestContext(brief_id="b1")

    first = source.fetch(candidate=cand, brief=brief, context=ctx)
    second = source.fetch(candidate=cand, brief=brief, context=ctx)

    assert isinstance(first, SignalResult)
    assert isinstance(second, SignalResult)
    # Cache hit marker only on second call.
    assert "(cached)" not in first.section_text
    assert "(cached)" in second.section_text
    # Provider called only once.
    assert client.fetch.call_count == 1


def test_stale_cache_entry_triggers_refetch(tmp_path: Path) -> None:
    cache_path = tmp_path / "news_cache.jsonl"
    # Pre-seed the cache with a stale row.
    stale_row = {
        "cache_key": _cache_key('"Jane Doe" AND ("AcmeCorp")', 90),
        "fetched_at_epoch": time.time() - NEWS_CACHE_TTL_SECONDS - 60,
        "articles": [
            {
                "title": "stale",
                "description": "",
                "url": "https://stale.example.com",
                "published_at": "2024-01-01T00:00:00Z",
                "source_name": "stale",
            }
        ],
    }
    cache_path.write_text(json.dumps(stale_row) + "\n")

    articles = [
        NewsArticle(
            title="fresh",
            description="",
            url="https://fresh.example.com",
            published_at="2026-04-15T12:00:00Z",
            source_name="fresh",
        ),
    ]
    client = _stub_client(articles)
    source = NewsSignalSource(
        client=client,
        cache_path_resolver=lambda b, ctx: cache_path,
    )

    result = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(window_days=90),
        context=SignalRequestContext(brief_id="b1"),
    )

    assert isinstance(result, SignalResult)
    assert "fresh" in result.section_text
    assert "stale" not in result.section_text
    assert client.fetch.call_count == 1


# ---------------------------------------------------------------------------
# Adapter — failure modes
# ---------------------------------------------------------------------------


def test_adapter_translates_news_api_error_into_upstream_error(
    tmp_path: Path,
) -> None:
    client = NewsApiClient(api_key="stub")
    def _explode(*, query: str, from_iso: str, to_iso: str, page_size: int = 25) -> list[NewsArticle]:
        raise NewsApiError("newsapi http 503", status_code=503)
    client.fetch = _explode  # type: ignore[method-assign]

    source = NewsSignalSource(
        client=client,
        cache_path_resolver=lambda b, ctx: tmp_path / "news_cache.jsonl",
    )
    result = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(result, SignalFailure)
    assert result.reason == "upstream_error"


def test_adapter_translates_unexpected_exception_into_adapter_exception(
    tmp_path: Path,
) -> None:
    client = NewsApiClient(api_key="stub")
    def _explode(*, query: str, from_iso: str, to_iso: str, page_size: int = 25) -> list[NewsArticle]:
        raise RuntimeError("boom")
    client.fetch = _explode  # type: ignore[method-assign]

    source = NewsSignalSource(
        client=client,
        cache_path_resolver=lambda b, ctx: tmp_path / "news_cache.jsonl",
    )
    result = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(result, SignalFailure)
    assert result.reason == "adapter_exception"


def test_adapter_returns_empty_query_when_candidate_lacks_identity(
    tmp_path: Path,
) -> None:
    client = _stub_client([])
    source = NewsSignalSource(
        client=client,
        cache_path_resolver=lambda b, ctx: tmp_path / "news_cache.jsonl",
    )
    cand = CandidateProfileSummary(
        name="",
        headline="",
        profile_url="",
    )
    result = source.fetch(
        candidate=cand,
        brief=_exec_brief(),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(result, SignalFailure)
    assert result.reason == "empty_query"
    # Provider was never called.
    assert client.fetch.call_count == 0


def test_adapter_renders_empty_articles_with_placeholder(tmp_path: Path) -> None:
    client = _stub_client([])
    source = NewsSignalSource(
        client=client,
        cache_path_resolver=lambda b, ctx: tmp_path / "news_cache.jsonl",
    )
    result = source.fetch(
        candidate=_candidate(),
        brief=_exec_brief(window_days=90),
        context=SignalRequestContext(brief_id="b1"),
    )
    assert isinstance(result, SignalResult)
    assert "[no news mentions found" in result.section_text


# ---------------------------------------------------------------------------
# Cache primitives
# ---------------------------------------------------------------------------


def test_cache_key_normalizes_whitespace_and_case() -> None:
    a = _cache_key("  VP Engineering  ", 90)
    b = _cache_key("vp engineering", 90)
    assert a == b


def test_cache_key_distinguishes_window_days() -> None:
    a = _cache_key("VP Engineering", 90)
    b = _cache_key("VP Engineering", 30)
    assert a != b


def test_load_cached_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert load_cached(tmp_path / "missing.jsonl", "q", 30) is None


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------


def test_parse_news_payload_handles_empty_articles() -> None:
    out = _parse_news_payload({"articles": []})
    assert out == []


def test_parse_news_payload_pulls_source_name() -> None:
    out = _parse_news_payload(
        {
            "articles": [
                {
                    "title": "T",
                    "description": "D",
                    "url": "U",
                    "publishedAt": "2026-04-15T12:00:00Z",
                    "source": {"name": "ExampleNews", "id": "ex"},
                },
            ]
        }
    )
    assert out[0].source_name == "ExampleNews"
    assert out[0].published_at == "2026-04-15T12:00:00Z"
