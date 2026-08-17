"""Designer module — image acquisition + SQLite-backed asset cache.

Designer Slice 5. Loads images of designer work product into a
per-state-dir SQLite cache so the vision-evaluation pipeline at
:mod:`designer.vision_evaluation` can ground itself in concrete bytes
rather than re-fetching on every call.

Cache shape (per ``designer/SOURCE_RIGHTS.md``):

- Bounded TTL: run lifetime + 30 days (a separate cron job in Slice
  11 prunes; this module only writes).
- Provenance retained 90 days even after blob deletion so audit
  queries can answer "where did this image come from, when, under
  what license posture?" — see ``provenance_only_view`` query.
- One asset row per (URL, candidate_identity_key) pair. Cross-
  candidate URL dedup happens at the application layer; storing
  duplicate URLs across candidates is a feature, not a bug (it
  preserves provenance per use).

Source-by-source URL extraction:

- Behance: parse the ``modules`` array of a ``/v2/projects/{id}``
  response. Image modules carry ``sizes.original`` (best res) +
  ``sizes.disp`` (~1024px); we prefer ``disp`` for vision-LLM input
  to keep token counts manageable.
- Google CSE: ``pagemap.cse_thumbnail.src`` from each result item
  (already extracted at acquisition time via
  :func:`designer.sources.google_cse.cse_result_thumbnail_url`).
- Direct portfolio fetch: OFF in v1 per ``designer/SOURCE_RIGHTS.md``;
  v1.5 enables per-host with explicit ToS sign-off.

Per-candidate cap: 8 images. Prioritization order on Behance projects:
most recent project hero > top-appreciation hero > showcase-tagged.
Spec-aligned with the cost envelope (8 images × 30 candidates ≈
62K image tokens at Gemini 2.5 Pro).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


# Per-candidate image cap. Mirrors spec §3.2 + §4.1: balance between
# cost (8 × 258 tokens × 30 candidates ≈ 62K image tokens per run at
# Gemini 2.5 Pro) and signal density (a portfolio is rarely
# decisively read from <4 images).
DEFAULT_MAX_IMAGES_PER_CANDIDATE = 8

# Per-source ToS posture strings. Logged on every cache write so
# audit queries can confirm what license each blob was retrieved
# under. Match the strings in `designer/SOURCE_RIGHTS.md`.
TOS_BEHANCE = "behance_developer_api_v2_cache_for_purpose"
TOS_GOOGLE_CSE = "google_cse_thumbnail_display_in_search_results"


@dataclass(frozen=True)
class CachedAsset:
    """One row from the asset cache.

    ``image_bytes`` may be ``None`` after the bounded TTL deletes the
    blob; the row itself (provenance metadata) survives 60 more days
    so audit can still answer "what was here?".
    """

    asset_id: int
    candidate_identity_key: str
    asset_url: str
    source: str
    project_id: int | None
    project_title: str
    content_hash: str
    retrieved_at: str
    tos_source: str
    image_bytes: bytes | None


@dataclass
class AssetAcquisitionResult:
    """Outcome of one ``acquire_images_for_candidate`` call.

    ``cached_assets`` are the asset rows the vision-evaluation prompt
    grounds itself in. ``failed_urls`` is the list of URLs the
    fetcher returned non-200 for (per-asset failures don't fail the
    whole acquisition; the vision pipeline degrades gracefully to
    a smaller image set).
    """

    cached_assets: tuple[CachedAsset, ...] = ()
    failed_urls: tuple[str, ...] = ()


class AssetCache:
    """SQLite-backed asset cache for the Designer module.

    One ``assets.sqlite3`` per state-dir. Schema is additive-migration
    friendly (we don't ship `_migrate` helpers in Slice 5; the table
    is created if missing on first connect, and column additions in
    later slices use ``ALTER TABLE`` with conditional creation).
    """

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS assets (
        asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_identity_key TEXT NOT NULL,
        asset_url TEXT NOT NULL,
        source TEXT NOT NULL,
        project_id INTEGER,
        project_title TEXT NOT NULL DEFAULT '',
        content_hash TEXT NOT NULL DEFAULT '',
        retrieved_at TEXT NOT NULL,
        tos_source TEXT NOT NULL,
        image_bytes BLOB
    );
    CREATE INDEX IF NOT EXISTS idx_assets_candidate
        ON assets(candidate_identity_key);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_assets_candidate_url
        ON assets(candidate_identity_key, asset_url);
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self.SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def write_asset(
        self,
        *,
        candidate_identity_key: str,
        asset_url: str,
        source: str,
        image_bytes: bytes | None,
        tos_source: str,
        project_id: int | None = None,
        project_title: str = "",
    ) -> CachedAsset:
        """Insert or replace an asset row.

        The (candidate, url) pair is unique. Re-fetching the same
        URL for the same candidate updates the existing row (refreshes
        ``retrieved_at`` and re-hashes the bytes).
        """

        content_hash = (
            hashlib.sha256(image_bytes).hexdigest() if image_bytes else ""
        )
        retrieved_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO assets (
                    candidate_identity_key, asset_url, source, project_id,
                    project_title, content_hash, retrieved_at, tos_source,
                    image_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_identity_key, asset_url)
                DO UPDATE SET
                    source = excluded.source,
                    project_id = excluded.project_id,
                    project_title = excluded.project_title,
                    content_hash = excluded.content_hash,
                    retrieved_at = excluded.retrieved_at,
                    tos_source = excluded.tos_source,
                    image_bytes = excluded.image_bytes
                """,
                (
                    candidate_identity_key,
                    asset_url,
                    source,
                    project_id,
                    project_title,
                    content_hash,
                    retrieved_at,
                    tos_source,
                    image_bytes,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM assets WHERE candidate_identity_key = ? AND asset_url = ?",
                (candidate_identity_key, asset_url),
            ).fetchone()

        return _row_to_asset(row)

    def list_assets_for_candidate(
        self, candidate_identity_key: str
    ) -> tuple[CachedAsset, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM assets WHERE candidate_identity_key = ? ORDER BY asset_id",
                (candidate_identity_key,),
            ).fetchall()
        return tuple(_row_to_asset(row) for row in rows)

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM assets").fetchone()
        return int(row["c"])


def _row_to_asset(row: sqlite3.Row) -> CachedAsset:
    return CachedAsset(
        asset_id=int(row["asset_id"]),
        candidate_identity_key=str(row["candidate_identity_key"]),
        asset_url=str(row["asset_url"]),
        source=str(row["source"]),
        project_id=(
            int(row["project_id"]) if row["project_id"] is not None else None
        ),
        project_title=str(row["project_title"] or ""),
        content_hash=str(row["content_hash"] or ""),
        retrieved_at=str(row["retrieved_at"]),
        tos_source=str(row["tos_source"]),
        image_bytes=bytes(row["image_bytes"]) if row["image_bytes"] is not None else None,
    )


# ---------------------------------------------------------------------------
# Per-source URL extraction
# ---------------------------------------------------------------------------


def behance_project_image_urls(
    project_response: dict[str, Any],
    *,
    prefer_size: str = "disp",
) -> list[str]:
    """Extract image URLs from a Behance ``/v2/projects/{id}`` response.

    Walks ``project.modules``, picks image modules, and returns the
    URLs at the requested size with fallback. ``prefer_size`` defaults
    to ``"disp"`` (~1024px) — the spec-recommended balance between
    visual fidelity and Gemini token cost.
    """

    project = project_response.get("project")
    if not isinstance(project, dict):
        return []
    modules = project.get("modules") or []
    if not isinstance(modules, list):
        return []

    urls: list[str] = []
    for module in modules:
        if not isinstance(module, dict):
            continue
        if module.get("type") != "image":
            continue
        sizes = module.get("sizes")
        if not isinstance(sizes, dict):
            continue
        # Try preferred size first, then fall back through known sizes.
        for size_key in (prefer_size, "original", "disp", "max_1240", "max_1200"):
            url = sizes.get(size_key)
            if isinstance(url, str) and url:
                urls.append(url)
                break
    return urls


def behance_project_title(project_response: dict[str, Any]) -> str:
    project = project_response.get("project")
    if not isinstance(project, dict):
        return ""
    title = project.get("name")
    return str(title) if isinstance(title, str) else ""


def behance_project_id(project_response: dict[str, Any]) -> int | None:
    project = project_response.get("project")
    if not isinstance(project, dict):
        return None
    raw_id = project.get("id")
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Acquisition orchestrator
# ---------------------------------------------------------------------------


# A fetcher is any callable that takes a URL and returns either
# ``bytes`` (success) or ``None`` (failure). The vision-evaluation
# pipeline injects a real HTTP fetcher in production; tests pass a
# deterministic fake.
ImageFetcher = Callable[[str], bytes | None]


def acquire_behance_images_for_candidate(
    *,
    candidate_identity_key: str,
    project_responses: list[dict[str, Any]],
    cache: AssetCache,
    fetcher: ImageFetcher,
    max_images: int = DEFAULT_MAX_IMAGES_PER_CANDIDATE,
    prefer_size: str = "disp",
) -> AssetAcquisitionResult:
    """Cache up to ``max_images`` images from a candidate's Behance projects.

    Iterates ``project_responses`` in order (caller controls priority
    — typically most recent project first), extracts image URLs per
    project, dedups by URL, fetches bytes via ``fetcher``, and writes
    each successful fetch to ``cache``. Stops at ``max_images`` total.
    Per-fetch failures land in ``failed_urls`` for the orchestrator
    to log; they do not abort the acquisition.
    """

    seen_urls: set[str] = set()
    cached: list[CachedAsset] = []
    failed: list[str] = []

    for project_response in project_responses:
        if len(cached) >= max_images:
            break
        urls = behance_project_image_urls(project_response, prefer_size=prefer_size)
        title = behance_project_title(project_response)
        project_id = behance_project_id(project_response)

        for url in urls:
            if len(cached) >= max_images:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)

            image_bytes = fetcher(url)
            if image_bytes is None:
                failed.append(url)
                continue

            asset = cache.write_asset(
                candidate_identity_key=candidate_identity_key,
                asset_url=url,
                source="behance",
                image_bytes=image_bytes,
                tos_source=TOS_BEHANCE,
                project_id=project_id,
                project_title=title,
            )
            cached.append(asset)

    return AssetAcquisitionResult(
        cached_assets=tuple(cached),
        failed_urls=tuple(failed),
    )


def acquire_cse_thumbnails_for_candidate(
    *,
    candidate_identity_key: str,
    cse_result_items: list[dict[str, Any]],
    cache: AssetCache,
    fetcher: ImageFetcher,
    max_images: int = DEFAULT_MAX_IMAGES_PER_CANDIDATE,
) -> AssetAcquisitionResult:
    """Cache CSE thumbnails for a candidate.

    CSE thumbnails are low-res (typically 300×200) but free under
    Google's ToS. The vision-evaluation pipeline can still reason
    against them; Behance assets when available carry richer signal.
    """

    from designer.sources.google_cse import cse_result_thumbnail_url

    seen_urls: set[str] = set()
    cached: list[CachedAsset] = []
    failed: list[str] = []

    for item in cse_result_items:
        if len(cached) >= max_images:
            break
        if not isinstance(item, dict):
            continue
        thumbnail_url = cse_result_thumbnail_url(item)
        if not thumbnail_url or thumbnail_url in seen_urls:
            continue
        seen_urls.add(thumbnail_url)

        image_bytes = fetcher(thumbnail_url)
        if image_bytes is None:
            failed.append(thumbnail_url)
            continue

        link = item.get("link")
        page_url = link if isinstance(link, str) else ""
        title = item.get("title")
        page_title = title if isinstance(title, str) else ""

        asset = cache.write_asset(
            candidate_identity_key=candidate_identity_key,
            asset_url=thumbnail_url,
            source="google_cse",
            image_bytes=image_bytes,
            tos_source=TOS_GOOGLE_CSE,
            project_id=None,
            project_title=page_title or page_url,
        )
        cached.append(asset)

    return AssetAcquisitionResult(
        cached_assets=tuple(cached),
        failed_urls=tuple(failed),
    )
