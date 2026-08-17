"""Tests for ``designer.prune_cache`` — audit Move #17.

Asserts the SOURCE_RIGHTS.md TTL contract:
- Image blobs older than 30 days get NULLed (provenance row survives).
- Provenance rows older than 90 days get deleted.
- Recent assets are untouched.
- ``--dry-run`` counts work without modifying the cache.
- ``--state-root`` mode discovers per-customer caches.

Uses :class:`AssetCache` for fixture writes so the schema stays in
lockstep with image_acquisition.py's writer.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from designer.image_acquisition import AssetCache
from designer.prune_cache import (
    IMAGE_BLOB_TTL_DAYS,
    PROVENANCE_RETENTION_DAYS,
    discover_caches,
    main,
    prune_cache,
)


def _seed_asset(
    cache: AssetCache,
    *,
    candidate_id: str,
    asset_url: str,
    retrieved_at: datetime,
    image_bytes: bytes | None = b"img",
) -> None:
    """Seed one row, then forcibly rewrite ``retrieved_at`` so we can
    backdate the row regardless of write_asset's clock."""

    cache.write_asset(
        candidate_identity_key=candidate_id,
        asset_url=asset_url,
        source="behance",
        image_bytes=image_bytes,
        tos_source="behance_v2_api_cache_for_evaluation",
        project_title="Project",
    )
    conn = sqlite3.connect(cache.db_path)
    conn.execute(
        "UPDATE assets SET retrieved_at = ? WHERE candidate_identity_key = ? AND asset_url = ?",
        (retrieved_at.isoformat(), candidate_id, asset_url),
    )
    conn.commit()
    conn.close()


def test_blob_ttl_nulls_old_image_bytes_keeps_row(tmp_path: Path) -> None:
    """A 45-day-old asset: image_bytes get NULLed; provenance survives."""

    cache = AssetCache(tmp_path / "assets.sqlite3")
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    _seed_asset(
        cache,
        candidate_id="c1",
        asset_url="https://example.com/old.jpg",
        retrieved_at=now - timedelta(days=45),
    )

    result = prune_cache(cache.db_path, now=now)
    assert result.blobs_dropped == 1
    assert result.rows_deleted == 0
    assert result.rows_kept == 1

    # Verify the blob is NULL but the row is intact.
    conn = sqlite3.connect(cache.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT image_bytes, retrieved_at FROM assets"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["image_bytes"] is None


def test_provenance_retention_deletes_rows_past_90_days(tmp_path: Path) -> None:
    """A 100-day-old asset: row gets deleted entirely."""

    cache = AssetCache(tmp_path / "assets.sqlite3")
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    _seed_asset(
        cache,
        candidate_id="c1",
        asset_url="https://example.com/older.jpg",
        retrieved_at=now - timedelta(days=100),
    )

    result = prune_cache(cache.db_path, now=now)
    assert result.rows_deleted == 1
    assert result.rows_kept == 0

    assert cache.count() == 0


def test_recent_assets_untouched(tmp_path: Path) -> None:
    """A 5-day-old asset: nothing prunes."""

    cache = AssetCache(tmp_path / "assets.sqlite3")
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    _seed_asset(
        cache,
        candidate_id="c1",
        asset_url="https://example.com/fresh.jpg",
        retrieved_at=now - timedelta(days=5),
    )

    result = prune_cache(cache.db_path, now=now)
    assert result.blobs_dropped == 0
    assert result.rows_deleted == 0
    assert result.rows_kept == 1

    conn = sqlite3.connect(cache.db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT image_bytes FROM assets").fetchone()
    conn.close()
    assert row["image_bytes"] is not None  # blob still present


def test_mixed_ages_prune_each_correctly(tmp_path: Path) -> None:
    """Three assets — fresh / 45 days / 100 days. Each handled per
    its bracket."""

    cache = AssetCache(tmp_path / "assets.sqlite3")
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    _seed_asset(cache, candidate_id="c1", asset_url="fresh", retrieved_at=now - timedelta(days=5))
    _seed_asset(cache, candidate_id="c1", asset_url="mid", retrieved_at=now - timedelta(days=45))
    _seed_asset(cache, candidate_id="c1", asset_url="old", retrieved_at=now - timedelta(days=100))

    result = prune_cache(cache.db_path, now=now)
    assert result.blobs_dropped == 1  # mid
    assert result.rows_deleted == 1  # old
    assert result.rows_kept == 2  # fresh + mid (mid's row survives, just blob gone)

    conn = sqlite3.connect(cache.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT asset_url, image_bytes FROM assets ORDER BY asset_url"
    ).fetchall()
    conn.close()
    by_url = {row["asset_url"]: row["image_bytes"] for row in rows}
    assert "fresh" in by_url
    assert "mid" in by_url
    assert "old" not in by_url
    assert by_url["fresh"] is not None
    assert by_url["mid"] is None


def test_re_running_prune_does_not_double_count_blobs(tmp_path: Path) -> None:
    """An already-NULLed blob row shouldn't show up in blobs_dropped on
    a second run."""

    cache = AssetCache(tmp_path / "assets.sqlite3")
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    _seed_asset(
        cache, candidate_id="c1", asset_url="mid",
        retrieved_at=now - timedelta(days=45),
    )

    first = prune_cache(cache.db_path, now=now)
    assert first.blobs_dropped == 1

    second = prune_cache(cache.db_path, now=now)
    assert second.blobs_dropped == 0
    assert second.rows_deleted == 0
    assert second.rows_kept == 1


def test_dry_run_counts_without_modifying(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "assets.sqlite3")
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    _seed_asset(cache, candidate_id="c1", asset_url="mid", retrieved_at=now - timedelta(days=45))
    _seed_asset(cache, candidate_id="c1", asset_url="old", retrieved_at=now - timedelta(days=100))

    result = prune_cache(cache.db_path, now=now, dry_run=True)
    assert result.blobs_dropped == 1
    assert result.rows_deleted == 1

    # Cache state unchanged: both rows still present, mid's blob intact.
    conn = sqlite3.connect(cache.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT asset_url, image_bytes FROM assets ORDER BY asset_url"
    ).fetchall()
    conn.close()
    by_url = {row["asset_url"]: row["image_bytes"] for row in rows}
    assert set(by_url.keys()) == {"mid", "old"}
    assert by_url["mid"] is not None
    assert by_url["old"] is not None


def test_discover_caches_finds_per_customer_caches(tmp_path: Path) -> None:
    """When state-root is the parent of multiple state-dirs, all child
    assets.sqlite3 files are discovered."""

    state_root = tmp_path / "designer"
    (state_root / "customer-a").mkdir(parents=True)
    (state_root / "customer-b").mkdir(parents=True)
    (state_root / "customer-empty").mkdir(parents=True)
    AssetCache(state_root / "customer-a" / "assets.sqlite3")
    AssetCache(state_root / "customer-b" / "assets.sqlite3")

    caches = discover_caches(state_root)
    assert len(caches) == 2
    assert {p.parent.name for p in caches} == {"customer-a", "customer-b"}


def test_discover_caches_handles_single_state_dir(tmp_path: Path) -> None:
    """state-root pointing directly at a state-dir (containing
    assets.sqlite3) returns just that one."""

    AssetCache(tmp_path / "assets.sqlite3")
    caches = discover_caches(tmp_path)
    assert len(caches) == 1
    assert caches[0].name == "assets.sqlite3"


def test_main_state_dir_dry_run_returns_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache = AssetCache(tmp_path / "assets.sqlite3")
    now = datetime.now(timezone.utc)
    _seed_asset(cache, candidate_id="c1", asset_url="old", retrieved_at=now - timedelta(days=100))

    rc = main(["--state-dir", str(tmp_path), "--dry-run"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "rows_deleted=1" in out

    # Cache unchanged (dry run).
    assert cache.count() == 1


def test_main_state_dir_missing_returns_2(tmp_path: Path) -> None:
    rc = main(["--state-dir", str(tmp_path / "nope")])
    assert rc == 2


def test_ttl_constants_match_source_rights() -> None:
    """SOURCE_RIGHTS.md says 30 days for blobs, 90 days for rows.
    Pin the constants so a drive-by change doesn't silently widen the
    bounded-cache claim."""

    assert IMAGE_BLOB_TTL_DAYS == 30
    assert PROVENANCE_RETENTION_DAYS == 90
