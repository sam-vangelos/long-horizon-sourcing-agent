"""Designer Slice 5 — image acquisition + SQLite asset cache.

Pins:

- :class:`AssetCache` schema is created on first connect; idempotent
  on re-open.
- ``write_asset`` inserts new rows; on (candidate, url) conflict it
  updates the existing row (refreshes ``retrieved_at``,
  ``content_hash``, etc.).
- ``content_hash`` is sha256 of the bytes; empty bytes → empty hash.
- Per-candidate cap honored.
- Failed fetches don't block subsequent fetches; ``failed_urls``
  carries the dropped URLs.
- Behance project image URL extraction prefers ``disp`` size,
  falls back through ``original`` and others.
- Google CSE acquisition pulls thumbnails via the Slice-3 helper.
- ToS provenance string lands per source (audit-readable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from designer.image_acquisition import (
    DEFAULT_MAX_IMAGES_PER_CANDIDATE,
    TOS_BEHANCE,
    TOS_GOOGLE_CSE,
    AssetCache,
    AssetAcquisitionResult,
    acquire_behance_images_for_candidate,
    acquire_cse_thumbnails_for_candidate,
    behance_project_id,
    behance_project_image_urls,
    behance_project_title,
)


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------


def _behance_project_response(
    *, project_id: int = 100, name: str = "Acme Design System"
) -> dict[str, Any]:
    return {
        "project": {
            "id": project_id,
            "name": name,
            "modules": [
                {
                    "id": 1,
                    "type": "image",
                    "sizes": {
                        "original": f"https://mir-s3-cdn-cf.behance.net/p/{project_id}/orig/1.jpg",
                        "disp": f"https://mir-s3-cdn-cf.behance.net/p/{project_id}/disp/1.jpg",
                    },
                },
                {
                    "id": 2,
                    "type": "text",  # not an image — should be skipped
                    "text": "Some prose.",
                },
                {
                    "id": 3,
                    "type": "image",
                    "sizes": {
                        "original": f"https://mir-s3-cdn-cf.behance.net/p/{project_id}/orig/2.jpg",
                        "disp": f"https://mir-s3-cdn-cf.behance.net/p/{project_id}/disp/2.jpg",
                    },
                },
            ],
            "covers": {
                "808": f"https://mir-s3-cdn-cf.behance.net/p/{project_id}/cover/808.jpg",
            },
            "stats": {"appreciations": 10, "views": 100},
            "fields": ["UI/UX"],
        }
    }


def _cse_result_item(*, link: str, thumbnail: str) -> dict[str, Any]:
    return {
        "link": link,
        "title": "Designer Portfolio",
        "displayLink": link.split("/")[2],
        "pagemap": {
            "cse_thumbnail": [{"src": thumbnail}],
        },
    }


def _make_fetcher(
    *, succeed: list[str], fail: list[str] | None = None
) -> Any:
    """Build a fetcher that returns known-bytes for success URLs and None for failures."""

    fail_set = set(fail or [])

    def _fetcher(url: str) -> bytes | None:
        if url in fail_set:
            return None
        if url in succeed:
            return f"bytes:{url}".encode()
        return None

    return _fetcher


# ---------------------------------------------------------------------------
# AssetCache schema + write/read
# ---------------------------------------------------------------------------


def test_cache_creates_schema_on_first_connect(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "assets.sqlite3")
    # Schema exists; cache is queryable.
    assert cache.count() == 0


def test_cache_write_round_trips(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "assets.sqlite3")
    asset = cache.write_asset(
        candidate_identity_key="behance:joe",
        asset_url="https://example.com/img1.jpg",
        source="behance",
        image_bytes=b"hello",
        tos_source=TOS_BEHANCE,
        project_id=42,
        project_title="Test",
    )
    assert asset.candidate_identity_key == "behance:joe"
    assert asset.image_bytes == b"hello"
    assert asset.content_hash == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert asset.tos_source == TOS_BEHANCE

    rows = cache.list_assets_for_candidate("behance:joe")
    assert len(rows) == 1
    assert rows[0].asset_url == "https://example.com/img1.jpg"


def test_cache_write_replaces_existing_row_on_conflict(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "assets.sqlite3")
    cache.write_asset(
        candidate_identity_key="behance:joe",
        asset_url="https://example.com/img1.jpg",
        source="behance",
        image_bytes=b"old",
        tos_source=TOS_BEHANCE,
    )
    cache.write_asset(
        candidate_identity_key="behance:joe",
        asset_url="https://example.com/img1.jpg",
        source="behance",
        image_bytes=b"new",
        tos_source=TOS_BEHANCE,
    )
    rows = cache.list_assets_for_candidate("behance:joe")
    # Still one row; bytes refreshed.
    assert len(rows) == 1
    assert rows[0].image_bytes == b"new"


def test_cache_persists_across_connect(tmp_path: Path) -> None:
    db_path = tmp_path / "assets.sqlite3"
    cache_a = AssetCache(db_path)
    cache_a.write_asset(
        candidate_identity_key="behance:joe",
        asset_url="x",
        source="behance",
        image_bytes=b"x",
        tos_source=TOS_BEHANCE,
    )
    # Re-open as a fresh instance — schema migration is idempotent.
    cache_b = AssetCache(db_path)
    assert cache_b.count() == 1


# ---------------------------------------------------------------------------
# Behance image URL extraction
# ---------------------------------------------------------------------------


def test_behance_project_image_urls_skips_non_image_modules() -> None:
    response = _behance_project_response()
    urls = behance_project_image_urls(response, prefer_size="disp")
    # Two image modules; the text module is skipped.
    assert len(urls) == 2
    assert all("disp" in url for url in urls)


def test_behance_project_image_urls_prefers_requested_size() -> None:
    response = _behance_project_response()
    urls = behance_project_image_urls(response, prefer_size="original")
    assert all("orig" in url for url in urls)


def test_behance_project_id_returns_int_or_none() -> None:
    assert behance_project_id(_behance_project_response(project_id=42)) == 42
    assert behance_project_id({"project": {}}) is None
    assert behance_project_id({}) is None


def test_behance_project_title_extracts_name() -> None:
    assert behance_project_title(_behance_project_response(name="X")) == "X"
    assert behance_project_title({}) == ""


# ---------------------------------------------------------------------------
# Behance acquisition orchestrator
# ---------------------------------------------------------------------------


def test_behance_acquisition_caches_each_image_with_provenance(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "assets.sqlite3")
    response = _behance_project_response(project_id=42)
    urls = behance_project_image_urls(response)
    fetcher = _make_fetcher(succeed=urls)

    result = acquire_behance_images_for_candidate(
        candidate_identity_key="behance:joe",
        project_responses=[response],
        cache=cache,
        fetcher=fetcher,
    )

    assert isinstance(result, AssetAcquisitionResult)
    assert len(result.cached_assets) == 2
    for asset in result.cached_assets:
        assert asset.source == "behance"
        assert asset.tos_source == TOS_BEHANCE
        assert asset.project_id == 42
        assert asset.project_title == "Acme Design System"


def test_behance_acquisition_dedups_repeated_urls_across_projects(
    tmp_path: Path,
) -> None:
    cache = AssetCache(tmp_path / "assets.sqlite3")
    # Two projects but same image URL repeated — dedup at acquisition
    # layer means only one cache row.
    response1 = _behance_project_response(project_id=1)
    response2 = _behance_project_response(project_id=1)  # identical
    urls = behance_project_image_urls(response1)
    fetcher = _make_fetcher(succeed=urls)

    result = acquire_behance_images_for_candidate(
        candidate_identity_key="behance:joe",
        project_responses=[response1, response2],
        cache=cache,
        fetcher=fetcher,
    )

    # Two unique URLs, regardless of duplicate projects.
    assert len(result.cached_assets) == 2


def test_behance_acquisition_caps_at_max_images_per_candidate(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "assets.sqlite3")
    # Synthesize many image modules across one project.
    response = {
        "project": {
            "id": 99,
            "name": "Big",
            "modules": [
                {
                    "id": idx,
                    "type": "image",
                    "sizes": {"disp": f"https://example.com/img{idx}.jpg"},
                }
                for idx in range(20)
            ],
        }
    }
    urls = behance_project_image_urls(response)
    fetcher = _make_fetcher(succeed=urls)

    result = acquire_behance_images_for_candidate(
        candidate_identity_key="behance:joe",
        project_responses=[response],
        cache=cache,
        fetcher=fetcher,
    )
    assert len(result.cached_assets) == DEFAULT_MAX_IMAGES_PER_CANDIDATE


def test_behance_acquisition_records_failed_fetches(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "assets.sqlite3")
    response = _behance_project_response(project_id=1)
    urls = behance_project_image_urls(response)
    # First URL fails; second succeeds.
    fetcher = _make_fetcher(succeed=[urls[1]], fail=[urls[0]])

    result = acquire_behance_images_for_candidate(
        candidate_identity_key="behance:joe",
        project_responses=[response],
        cache=cache,
        fetcher=fetcher,
    )
    assert len(result.cached_assets) == 1
    assert urls[0] in result.failed_urls
    assert urls[1] not in result.failed_urls


# ---------------------------------------------------------------------------
# CSE thumbnail acquisition
# ---------------------------------------------------------------------------


def test_cse_acquisition_caches_thumbnails_with_provenance(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "assets.sqlite3")
    items = [
        _cse_result_item(
            link="https://joe.cargo.site/work",
            thumbnail="https://thumb.example.com/joe.jpg",
        ),
        _cse_result_item(
            link="https://sara.squarespace.com/branding",
            thumbnail="https://thumb.example.com/sara.jpg",
        ),
    ]
    fetcher = _make_fetcher(
        succeed=["https://thumb.example.com/joe.jpg", "https://thumb.example.com/sara.jpg"]
    )

    result = acquire_cse_thumbnails_for_candidate(
        candidate_identity_key="cse:joe.cargo.site/work",
        cse_result_items=items,
        cache=cache,
        fetcher=fetcher,
    )

    assert len(result.cached_assets) == 2
    for asset in result.cached_assets:
        assert asset.source == "google_cse"
        assert asset.tos_source == TOS_GOOGLE_CSE


def test_cse_acquisition_skips_items_without_thumbnail(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "assets.sqlite3")
    items = [
        {"link": "https://joe.cargo.site/work", "title": "Joe"},  # no pagemap
        _cse_result_item(
            link="https://sara.squarespace.com",
            thumbnail="https://thumb.example.com/sara.jpg",
        ),
    ]
    fetcher = _make_fetcher(succeed=["https://thumb.example.com/sara.jpg"])

    result = acquire_cse_thumbnails_for_candidate(
        candidate_identity_key="cse:cross",
        cse_result_items=items,
        cache=cache,
        fetcher=fetcher,
    )
    assert len(result.cached_assets) == 1
