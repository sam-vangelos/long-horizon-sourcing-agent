"""Tests for :mod:`github.maintainer_signal_cache` (Slice 3 — OSS Maintainers).

Covers:

- Path discipline (owner/repo lowercased, signal_kind verbatim).
- Round-trip read after write returns the same payload.
- TTL expiry: a stale entry returns ``None`` even though the file
  exists on disk.
- Unknown signal kinds: ``put`` and ``get`` warn but don't raise.
- Corrupt JSON returns ``None`` (cache miss) without raising.
- Missing files return ``None`` (cache miss) without raising.

All tests redirect ``CACHE_ROOT`` to a tmp_path-derived directory via
monkeypatch so the host filesystem isn't touched.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import github.maintainer_signal_cache as mcache


@pytest.fixture(autouse=True)
def _isolated_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(mcache, "CACHE_ROOT", cache_root)
    return cache_root


def test_path_discipline_lowercases_and_segments(_isolated_cache_root: Path) -> None:
    p = mcache._path_for("Kubernetes", "Kubernetes", "releases")
    assert p == _isolated_cache_root / "kubernetes" / "kubernetes" / "releases.json"


def test_put_then_get_round_trip() -> None:
    payload = [{"tag_name": "v1.0.0"}, {"tag_name": "v0.9.0"}]
    mcache.put("kubernetes", "kubernetes", "releases", payload)

    entry = mcache.get("kubernetes", "kubernetes", "releases")

    assert entry is not None
    assert entry.signal_kind == "releases"
    assert entry.owner == "kubernetes"
    assert entry.repo == "kubernetes"
    assert entry.data == payload


def test_get_returns_none_for_missing_file() -> None:
    assert mcache.get("nonexistent", "repo", "releases") is None


def test_is_fresh_matches_get(_isolated_cache_root: Path) -> None:
    assert mcache.is_fresh("kubernetes", "kubernetes", "releases") is False
    mcache.put("kubernetes", "kubernetes", "releases", [{"tag_name": "v1"}])
    assert mcache.is_fresh("kubernetes", "kubernetes", "releases") is True


def test_get_returns_none_when_ttl_expired(_isolated_cache_root: Path) -> None:
    """Manually backdate the file so it's older than the TTL window."""

    mcache.put("kubernetes", "kubernetes", "pr_merges", [])
    path = mcache._path_for("kubernetes", "kubernetes", "pr_merges")
    raw = json.loads(path.read_text())
    # pr_merges TTL is 24h; backdate by 48h.
    stale = datetime.now(timezone.utc) - timedelta(hours=48)
    raw["fetched_at"] = stale.isoformat()
    path.write_text(json.dumps(raw))

    assert mcache.get("kubernetes", "kubernetes", "pr_merges") is None


def test_get_within_ttl_window(_isolated_cache_root: Path) -> None:
    """A fresh file just under the TTL window returns a hit."""

    mcache.put("kubernetes", "kubernetes", "releases", [{"tag_name": "v1"}])
    path = mcache._path_for("kubernetes", "kubernetes", "releases")
    raw = json.loads(path.read_text())
    # releases TTL is 7 days; backdate by 6 days (still fresh).
    fresh = datetime.now(timezone.utc) - timedelta(days=6)
    raw["fetched_at"] = fresh.isoformat()
    path.write_text(json.dumps(raw))

    entry = mcache.get("kubernetes", "kubernetes", "releases")
    assert entry is not None
    assert entry.data == [{"tag_name": "v1"}]


def test_unknown_signal_kind_get_returns_none() -> None:
    assert mcache.get("kubernetes", "kubernetes", "not_a_real_kind") is None


def test_unknown_signal_kind_put_does_not_raise() -> None:
    # Should warn, not raise; the file should not be created.
    mcache.put("kubernetes", "kubernetes", "not_a_real_kind", {"x": 1})
    p = mcache._path_for("kubernetes", "kubernetes", "not_a_real_kind")
    assert not p.exists()


def test_corrupt_json_returns_none(_isolated_cache_root: Path) -> None:
    path = mcache._path_for("kubernetes", "kubernetes", "releases")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")

    assert mcache.get("kubernetes", "kubernetes", "releases") is None


def test_signal_kinds_and_ttls_in_sync() -> None:
    """Every recognized signal kind has a TTL entry."""

    assert set(mcache.SIGNAL_KINDS) <= set(mcache.TTL_BY_KIND.keys())
