"""Designer asset cache pruning — audit Move #17.

Two-stage pruning per :file:`designer/SOURCE_RIGHTS.md`:

1. **Image-blob TTL: 30 days.** Once an asset's ``retrieved_at`` is
   older than 30 days, the ``image_bytes`` BLOB column is NULLed out.
   The provenance metadata row survives so audit can still answer
   "where did Cloris get this image, when, under what license posture?"
2. **Provenance retention: 90 days.** Once ``retrieved_at`` is older
   than 90 days, the entire row is deleted.

Invocable from the operator's shell:

    python -m designer.prune_cache --state-dir output/state/designer/<key>
    python -m designer.prune_cache --state-root output/state/designer
    python -m designer.prune_cache --state-root output/state/designer --dry-run

Wiring this into a cron job is an operator concern — not packaged
here. The script is a one-shot CLI; the operator schedules it.

Posture per SOURCE_RIGHTS.md "Bounded cache" section: the pruning
discipline is the load-bearing artifact of Cloris's "we don't
warehouse portfolio assets" claim. Don't bypass.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator


IMAGE_BLOB_TTL_DAYS = 30
PROVENANCE_RETENTION_DAYS = 90

ASSETS_FILENAME = "assets.sqlite3"


@dataclass
class PruneResult:
    """Per-cache pruning outcome."""

    db_path: Path
    blobs_dropped: int
    rows_deleted: int
    rows_kept: int

    def as_log_line(self, *, dry_run: bool) -> str:
        prefix = "[dry-run] " if dry_run else ""
        return (
            f"{prefix}{self.db_path}: "
            f"blobs_dropped={self.blobs_dropped} "
            f"rows_deleted={self.rows_deleted} "
            f"rows_kept={self.rows_kept}"
        )


def prune_cache(
    db_path: str | Path,
    *,
    now: datetime | None = None,
    image_blob_ttl_days: int = IMAGE_BLOB_TTL_DAYS,
    provenance_retention_days: int = PROVENANCE_RETENTION_DAYS,
    dry_run: bool = False,
) -> PruneResult:
    """Prune one ``assets.sqlite3`` against the SOURCE_RIGHTS TTL contract.

    ``now`` defaults to UTC now; tests inject a fixed time. Returns a
    :class:`PruneResult` summarizing the operation. ``dry_run=True``
    counts the work without performing it.
    """

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"prune_cache: assets.sqlite3 not found at {db_path}"
        )

    now = now or datetime.now(timezone.utc)
    blob_cutoff = (now - timedelta(days=image_blob_ttl_days)).isoformat()
    row_cutoff = (now - timedelta(days=provenance_retention_days)).isoformat()

    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        # Count the rows that would be deleted entirely first (oldest
        # rows past the 90-day threshold). retrieved_at is stored as
        # ISO 8601 UTC strings; lexical comparison matches chronological
        # ordering.
        rows_to_delete = conn.execute(
            "SELECT COUNT(*) AS c FROM assets WHERE retrieved_at < ?",
            (row_cutoff,),
        ).fetchone()["c"]

        # Then count rows whose blob would get NULLed (older than 30
        # days but younger than 90 days, AND whose blob is currently
        # non-null — re-running the pruner shouldn't double-count).
        blobs_to_drop = conn.execute(
            """
            SELECT COUNT(*) AS c FROM assets
            WHERE retrieved_at < ?
              AND retrieved_at >= ?
              AND image_bytes IS NOT NULL
            """,
            (blob_cutoff, row_cutoff),
        ).fetchone()["c"]

        rows_total = conn.execute(
            "SELECT COUNT(*) AS c FROM assets"
        ).fetchone()["c"]

        if not dry_run:
            conn.execute(
                "DELETE FROM assets WHERE retrieved_at < ?",
                (row_cutoff,),
            )
            conn.execute(
                """
                UPDATE assets
                SET image_bytes = NULL
                WHERE retrieved_at < ?
                  AND image_bytes IS NOT NULL
                """,
                (blob_cutoff,),
            )
            conn.commit()

    rows_kept = max(rows_total - rows_to_delete, 0)
    return PruneResult(
        db_path=db_path,
        blobs_dropped=int(blobs_to_drop),
        rows_deleted=int(rows_to_delete),
        rows_kept=rows_kept,
    )


def discover_caches(state_root: str | Path) -> list[Path]:
    """Return every ``assets.sqlite3`` under ``state_root``.

    Looks one or two levels deep — ``state_root`` may be either a
    single state-dir (containing ``assets.sqlite3`` directly) or the
    parent ``output/state/designer/`` (each child a state-dir). Both
    layouts are supported so the operator can prune one customer's
    cache or the full set with the same command.
    """

    root = Path(state_root)
    if not root.exists():
        return []

    if (root / ASSETS_FILENAME).exists():
        return [root / ASSETS_FILENAME]

    found: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        candidate = child / ASSETS_FILENAME
        if candidate.exists():
            found.append(candidate)
    return found


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="designer.prune_cache",
        description=(
            "Prune Designer asset caches per SOURCE_RIGHTS.md TTL: "
            "30-day blobs, 90-day provenance."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--state-dir",
        help="Single state-dir containing assets.sqlite3 (e.g., output/state/designer/<key>).",
    )
    group.add_argument(
        "--state-root",
        help="Designer state-dir parent (e.g., output/state/designer); prunes every child cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count what would be pruned without modifying the cache.",
    )
    parser.add_argument(
        "--blob-ttl-days",
        type=int,
        default=IMAGE_BLOB_TTL_DAYS,
        help=f"Image-blob TTL in days. Default {IMAGE_BLOB_TTL_DAYS}.",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=PROVENANCE_RETENTION_DAYS,
        help=(
            "Provenance row retention in days. Default "
            f"{PROVENANCE_RETENTION_DAYS}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.state_dir:
        candidate = Path(args.state_dir) / ASSETS_FILENAME
        if not candidate.exists():
            sys.stderr.write(
                f"designer.prune_cache: no assets.sqlite3 at {candidate}\n"
            )
            return 2
        caches = [candidate]
    else:
        caches = discover_caches(args.state_root)
        if not caches:
            sys.stderr.write(
                f"designer.prune_cache: no assets.sqlite3 found under "
                f"{args.state_root}\n"
            )
            return 2

    total_blobs = 0
    total_rows = 0
    for db_path in caches:
        result = prune_cache(
            db_path,
            image_blob_ttl_days=args.blob_ttl_days,
            provenance_retention_days=args.retention_days,
            dry_run=args.dry_run,
        )
        total_blobs += result.blobs_dropped
        total_rows += result.rows_deleted
        sys.stdout.write(result.as_log_line(dry_run=args.dry_run) + "\n")

    summary_prefix = "[dry-run] " if args.dry_run else ""
    sys.stdout.write(
        f"{summary_prefix}designer.prune_cache: caches={len(caches)} "
        f"blobs_dropped_total={total_blobs} rows_deleted_total={total_rows}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
