"""Disk-backed cache for OSS Maintainers signal acquisition (Slice 3).

Per OSS Maintainers Module Spec §9 caching strategy: the per-target-
project signals (PR merges, releases, governance files, downstream-
dependents page) are expensive to fetch repeatedly across briefs and
runs. The cache lives at ``output/cache/github/maintainer_signals/``
and is shared across briefs/runs — two briefs targeting
"kubernetes/kubernetes" only incur the API cost once per TTL window.

TTL by signal kind (spec §9):

- ``releases``, ``governance``, ``contributors_file`` — 7 days
- ``pr_merges``, ``pr_reviews`` — 24 hours
- ``network_dependents`` — 30 days

The file format is JSON with the following shape::

    {
        "fetched_at": "<ISO 8601 timestamp>",
        "signal_kind": "<kind>",
        "owner": "<owner>",
        "repo": "<repo>",
        "data": <kind-specific payload>
    }

This module owns the path discipline (signal kind ⇒ on-disk subpath)
and the TTL enforcement. Per-signal payload shape lives with the
classifier in :mod:`github.maintainership` (Slice 4) and the
project-quality sub-index in :mod:`github.project_quality` (Slice 5).

Failure mode posture: cache reads are best-effort. A corrupt JSON
file or unreadable path returns ``None`` (cache miss); the caller
re-fetches and re-writes. Cache writes that fail are logged but do
not raise (a cache write failure should not abort a slice that
already has the data in memory).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from shared.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


# Signal kinds the cache knows about. Adding a new kind = add an
# entry here AND a TTL below. The classifier will fail loud at
# import time if it tries to use an unknown kind, which is the
# intent — TTL discipline lives at the cache layer, not scattered
# across signals.
SIGNAL_KINDS: frozenset[str] = frozenset(
    {
        "releases",
        "governance",
        "contributors_file",
        "maintainers_file",
        "readme",
        "pr_merges",
        "pr_reviews",
        "network_dependents",
        "user_orgs",
        # Audit Move #22: PyPI / npm recent-download counts. Keyed
        # under sentinel "owner" values _pypi / _npm with the package
        # name in the "repo" position (the cache layer is owner/repo
        # agnostic at the file-naming level).
        "pypi_recent_downloads",
        "npm_recent_downloads",
        "roster_codeowners",
        "roster_maintainers",
        "roster_governance",
        "roster_recipe",
    }
)


# Per-kind TTL. Spec §9 values; tweakable here without touching call
# sites. A signal kind in :data:`SIGNAL_KINDS` MUST appear here, or
# :func:`get` returns ``None`` (treated as never-fresh) and emits a
# warning.
TTL_BY_KIND: dict[str, timedelta] = {
    "releases": timedelta(days=7),
    "governance": timedelta(days=7),
    "contributors_file": timedelta(days=7),
    "maintainers_file": timedelta(days=7),
    "readme": timedelta(days=7),
    "pr_merges": timedelta(hours=24),
    "pr_reviews": timedelta(hours=24),
    "network_dependents": timedelta(days=30),
    "user_orgs": timedelta(days=7),
    # Audit Move #22: registry download counts refresh slowly enough
    # that 7 days is plenty.
    "pypi_recent_downloads": timedelta(days=7),
    "npm_recent_downloads": timedelta(days=7),
    "roster_codeowners": timedelta(days=7),
    "roster_maintainers": timedelta(days=7),
    "roster_governance": timedelta(days=7),
    "roster_recipe": timedelta(days=7),
}


CACHE_ROOT: Path = OUTPUT_DIR / "cache" / "github" / "maintainer_signals"


@dataclass(frozen=True)
class CacheEntry:
    """In-memory shape of a cache hit.

    ``data`` is the kind-specific payload (e.g. a list of release
    dicts, a string of GOVERNANCE.md text, an int of dependent count).
    Callers are responsible for shape validation; the cache layer
    treats it as opaque JSON.
    """

    fetched_at: datetime
    signal_kind: str
    owner: str
    repo: str
    data: Any


def _path_for(owner: str, repo: str, signal_kind: str) -> Path:
    """Return the on-disk path for a (owner, repo, signal_kind) tuple.

    Owner/repo are lowercased to keep "Kubernetes/Kubernetes" and
    "kubernetes/kubernetes" sharing a cache slot. The signal kind is
    NOT lowercased (kinds are internal constants).
    """

    safe_owner = owner.strip().lower().replace("/", "_") or "_unknown"
    safe_repo = repo.strip().lower().replace("/", "_") or "_unknown"
    return CACHE_ROOT / safe_owner / safe_repo / f"{signal_kind}.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def is_fresh(owner: str, repo: str, signal_kind: str) -> bool:
    """Return True if a cache entry exists and is within its TTL window."""

    if signal_kind not in SIGNAL_KINDS:
        logger.warning(
            "maintainer_signal_cache.is_fresh: unknown signal_kind=%r", signal_kind
        )
        return False
    entry = get(owner, repo, signal_kind)
    return entry is not None


def get(owner: str, repo: str, signal_kind: str) -> Optional[CacheEntry]:
    """Read a cache entry. Returns ``None`` on miss, corruption, or TTL expiry.

    The miss case is silent (cache miss is the default, expected
    state). A corrupt file is logged but treated as a miss; the
    caller re-fetches and overwrites. A stale file (past TTL) is
    treated as a miss but the file is left in place — the next
    successful :func:`put` overwrites it.
    """

    if signal_kind not in SIGNAL_KINDS:
        logger.warning(
            "maintainer_signal_cache.get: unknown signal_kind=%r", signal_kind
        )
        return None
    path = _path_for(owner, repo, signal_kind)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "maintainer_signal_cache.get: corrupt cache at %s (%s); treating as miss",
            path,
            exc,
        )
        return None
    fetched_at_str = raw.get("fetched_at")
    if not isinstance(fetched_at_str, str):
        return None
    try:
        fetched_at = datetime.fromisoformat(fetched_at_str)
    except ValueError:
        return None
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    ttl = TTL_BY_KIND.get(signal_kind)
    if ttl is None:
        return None
    if _now() - fetched_at > ttl:
        return None
    return CacheEntry(
        fetched_at=fetched_at,
        signal_kind=signal_kind,
        owner=raw.get("owner", owner),
        repo=raw.get("repo", repo),
        data=raw.get("data"),
    )


def put(owner: str, repo: str, signal_kind: str, data: Any) -> None:
    """Write a cache entry. Failures are logged, never raised."""

    if signal_kind not in SIGNAL_KINDS:
        logger.warning(
            "maintainer_signal_cache.put: unknown signal_kind=%r", signal_kind
        )
        return
    path = _path_for(owner, repo, signal_kind)
    payload = {
        "fetched_at": _now().isoformat(),
        "signal_kind": signal_kind,
        "owner": owner,
        "repo": repo,
        "data": data,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError as exc:
        logger.warning(
            "maintainer_signal_cache.put: write failed at %s (%s); proceeding without cache",
            path,
            exc,
        )
