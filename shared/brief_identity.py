"""Phase 3: brief content identity helpers.

The runs table now carries three optional fields (Phase 3 schema bump):

- ``brief_path_at_launch``: the on-disk path the orchestrator loaded the
  brief from at run-start.
- ``brief_content_hash``: a stable hash of the brief's JSON content.
- ``brief_snapshot_json``: the canonical JSON of the brief at run-start
  (so Run Review and Next Run Learning can show "what the brief said
  when this run executed" even if the on-disk file has since changed).

This module is the single source of truth for how those values are
computed. Callers must NOT roll their own hashing — the chosen scheme
is canonical-JSON SHA-256 with a ``"sha256:"`` algorithm prefix
(per design-plan critique A4):

- We hash the post-``json.loads`` dict (canonical-JSON form), not the
  raw bytes on disk. This keeps the hash stable across BOM/CRLF/
  whitespace differences that don't affect what the brief loader sees.
- ``sort_keys=True``, ``separators=(",", ":")``, ``ensure_ascii=False``
  pin the canonical-JSON shape.
- The ``"sha256:"`` prefix means a future migration to a different
  algorithm can be detected by inspecting the hash string.

Brief drift detection (in cloris.control_plane.aggregate_status) reads
``runs.brief_path_at_launch`` and ``runs.brief_content_hash``, then
re-hashes whatever is on disk now and compares. If hashes differ, the
aggregator surfaces ``brief_drift_since_last_run=True`` so the UI can
warn that the brief was modified after the run started.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict


class BriefIdentity(TypedDict):
    """The three fields ``RuntimeStateStore.start_run`` accepts for pinning.

    All three are optional from the store's perspective; legacy callers
    that don't compute identity end up with NULL/empty columns and the
    aggregator handles those rows gracefully (no drift detection,
    state_key fallback for the UI heading).
    """

    brief_path_at_launch: str
    brief_content_hash: str
    brief_snapshot_json: str


def canonical_brief_hash(raw: dict) -> str:
    """Return the canonical-JSON SHA-256 of ``raw`` with a ``"sha256:"`` prefix.

    Stable across:

    - Key ordering differences (sort_keys=True at hash time).
    - Whitespace and indentation differences (separators are tight).
    - BOM / CRLF / encoding artifacts on disk (we hash the post-load dict,
      not the raw bytes).

    Not stable across:

    - Any actual content change in the brief (added/removed/modified key).

    The prefix is intentional: we expect to migrate to a stronger
    algorithm at some point, and prefixing the algorithm name lets
    consumers detect the version without an out-of-band schema bump.
    """

    canonical = json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def canonical_brief_snapshot(raw: dict) -> str:
    """Return the canonical-JSON serialization stored in ``brief_snapshot_json``.

    Same canonical form used for hashing — that way the snapshot's hash
    matches ``brief_content_hash`` and Run Review can re-verify the
    snapshot on read.
    """

    return json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def compute_brief_identity(brief_path: str | Path) -> BriefIdentity | None:
    """Build a :class:`BriefIdentity` for ``brief_path``, or ``None``.

    Returns ``None`` when:
    - ``brief_path`` is empty/None
    - the file doesn't exist
    - the file isn't valid JSON
    - the JSON top level isn't a dict (briefs are dicts)

    Best-effort: a malformed brief should not block run-start. The
    orchestrator will fail loudly later when ``brief_loader.load_brief``
    tries to parse the same file. Returning ``None`` here just means
    this run won't have pinning — the legacy NULL behavior the
    aggregator already handles.
    """

    if not brief_path:
        return None
    path = Path(brief_path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return BriefIdentity(
        brief_path_at_launch=str(path),
        brief_content_hash=canonical_brief_hash(raw),
        brief_snapshot_json=canonical_brief_snapshot(raw),
    )


def hash_current_brief_on_disk(brief_path: str | Path) -> str | None:
    """Return the canonical hash of whatever is at ``brief_path`` now.

    Used by the aggregator for drift detection: compare the result
    against ``runs.brief_content_hash``. ``None`` when the file is
    missing/unreadable; the aggregator surfaces that as
    ``brief_drift_since_last_run=None`` (unknown).
    """

    if not brief_path:
        return None
    path = Path(brief_path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return canonical_brief_hash(raw)
