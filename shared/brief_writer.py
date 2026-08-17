"""Atomic brief-write + version snapshot helper.

Phase D Slice D3 extracts the write contract D2's PUT path codified
inline so the intake-complete endpoint (``POST /api/intake/sessions/
{id}/complete``) and the brief-edit endpoint (``PUT /api/brief/
{brief_id}``) share one truth. The architectural-fit critique flagged
this as risk #3 ("D2's brief-write contract diverges from D3's brief-
write contract"); centralizing the write here closes it.

The contract:

1. Atomic write canonical FIRST via ``tempfile.NamedTemporaryFile`` +
   ``os.replace`` (cross-fs safety: the tempfile lives in the same
   directory as the canonical so the rename is atomic on every POSIX
   filesystem we care about).
2. fsync the tempfile before replace so a crash mid-write doesn't
   leave a half-written canonical visible.
3. After the canonical is in place, snapshot it to
   ``versions/<timestamp>.json`` via ``shutil.copy2`` so the audit
   trail can never run ahead of the canonical. (D2 architectural-fit
   critique catch — earlier draft wrote versions/ first, which left
   a window where ``versions/<latest>.json`` existed but
   ``brief.json`` was stale.)
4. Timestamp uses ISO-8601 with ``:`` → ``-`` for filesystem safety;
   includes microseconds so concurrent writes in the same second
   don't collide.

Errors raise :class:`OSError` (or one of its subclasses); callers map
to HTTP 500.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_brief_atomic(
    *,
    abs_path: Path,
    payload: dict[str, Any],
) -> Path:
    """Write ``payload`` to ``abs_path`` and snapshot to sibling versions/.

    ``abs_path`` is the canonical destination — typically
    ``config/<role-slug>/brief.json``. The parent directory must
    already exist; callers create it for new briefs (D3 intake
    complete) or rely on it existing for edits (D2 PUT).

    Returns the path of the version snapshot just written, so callers
    can log it or surface it in telemetry. The canonical's location is
    unchanged from the input.

    The function does NOT validate ``payload``; callers are expected
    to validate via :func:`shared.brief_v2_schema.validate_v2_brief`
    before calling here. This separation keeps the writer purely
    mechanical so it stays reusable across surfaces with different
    validation rules (e.g. a future "import legacy brief" surface
    that intentionally skips V2 validation).
    """

    parent = abs_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(parent),
        delete=False,
        suffix=".tmp",
    ) as tf:
        json.dump(payload, tf, indent=2, ensure_ascii=False)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_path = Path(tf.name)
    os.replace(tmp_path, abs_path)

    versions_dir = parent / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat().replace(":", "-")
    version_path = versions_dir / f"{stamp}.json"
    shutil.copy2(str(abs_path), str(version_path))
    return version_path
