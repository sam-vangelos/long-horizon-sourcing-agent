#!/usr/bin/env python3
"""Sync recruiter ``judgment_accuracy`` markers into Langfuse datasets.

Phase 2 of Langfuse adoption. Walks every per-source state-dir under
``output/state/`` and exports the ``judgment_accuracy`` markers
persisted in each ``runtime_state.sqlite3`` into a Langfuse dataset
named per the brief / module / discipline. Operators run this on
demand or via cron to backfill historical recruiter feedback into
Langfuse, where the prompt-regression runner
(``tools/run_prompt_regression.py``) consumes them.

## Idempotency

Each dataset row carries an ``identity_key`` in its ``metadata``.
The sync tool dedupes by ``(dataset_name, identity_key)`` against
the dataset's existing items so re-running the sync against the
same state-dir doesn't double-emit. Identity keys are stable across
brief revisions (``shared/runtime_state/store.py`` writes them at
candidate-creation time and never mutates them).

## Rate limiting

Free-tier Langfuse cloud carries ~50K observations/month. A first
sync against an established trial cohort can backfill thousands of
historical markers; without batching the per-call latency dominates
and the sync hits the daily rate ceiling. This tool:

- Batches dataset-item creates in chunks of ``BATCH_SIZE`` (default
  100) per dataset.
- Sleeps ``INTER_BATCH_SLEEP_S`` (default 0.2s) between batches.
- On a 429 response from the Langfuse API, parses the ``Retry-After``
  header (or falls back to exponential backoff capped at 60s) and
  resumes from the last batch boundary so partial syncs don't
  re-emit completed rows.

## Usage

    python -m tools.sync_judgment_datasets \\
        --output-root output/ \\
        --source linkedin \\
        --brief-id frontier-ai-fde \\
        [--brief-path config/brief-frontier-ai.json] \\
        [--dry-run]

Or sync every (source, brief_id) tuple in the output root:

    python -m tools.sync_judgment_datasets --output-root output/

The CLI reads Langfuse credentials from the ``LANGFUSE_PUBLIC_KEY``
/ ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST`` env vars. When
credentials are absent OR ``LANGFUSE_DISABLE=1``, the tool exits
with code 2 + a clear message — the dataset push is the whole
point of the tool, so no-op-on-degraded would be silently broken.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.runtime_state.calibration import (
    LangfuseDatasetRow,
    build_langfuse_dataset_rows,
)

logger = logging.getLogger(__name__)


# Pinned per the plan body. Free-tier Langfuse cloud caps at ~50K
# observations/month; chunks of 100 + 200ms inter-batch keep first-
# sync floods well under the per-second ceiling.
BATCH_SIZE: int = 100
INTER_BATCH_SLEEP_S: float = 0.2
RETRY_AFTER_FALLBACK_CAP_S: float = 60.0
MAX_RATE_LIMIT_RETRIES: int = 5


# Sentinel used by the per-source dataset-name resolver. Kept as a
# constant so the regression runner reads from the same naming.
JUDGMENT_DATASET_PREFIX: str = "judgment-accuracy"


@dataclass
class SyncResult:
    """Outcome of one (source, brief_id) sync.

    The CLI's exit code is determined by the aggregate result across
    all syncs: any ``failed_count > 0`` exits non-zero.
    """

    source: str
    brief_id: str
    dataset_name: str
    rows_built: int
    rows_pushed: int
    rows_skipped_idempotent: int
    failed_count: int
    dry_run: bool


def dataset_name_for(*, source: str, brief_id: str) -> str:
    """Per-(source, brief) dataset name with a stable shape.

    Operators can filter / group by the prefix in the Langfuse UI.
    """

    return f"{JUDGMENT_DATASET_PREFIX}-{source}-{brief_id}"


def _discover_state_dirs(
    *,
    output_root: Path,
    sources: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Walk ``output/state/<source>/<state_key>/`` and return state-dirs.

    Returns ``(source, state_dir_path)`` tuples. When ``sources`` is
    set, filter to those sources only. Otherwise enumerate every
    source dir present.
    """

    state_root = output_root / "state"
    if not state_root.exists():
        return []
    found: list[tuple[str, Path]] = []
    for source_dir in sorted(state_root.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        if sources is not None and source not in sources:
            continue
        for state_dir in sorted(source_dir.iterdir()):
            if not state_dir.is_dir():
                continue
            db_path = state_dir / "runtime_state.sqlite3"
            if db_path.exists():
                found.append((source, state_dir))
    return found


def _brief_dict_for_state_dir(state_dir: Path) -> dict | None:
    """Recover the brief's recruiter-authored content for the dataset
    row's ``capability_areas`` + ``depth_distinction`` fields.

    Reads the most-recent run's ``brief_snapshot_json`` from
    ``runtime_state.sqlite3`` so the export survives brief revisions.
    Returns ``None`` if no snapshot is available — the dataset row
    will be emitted with empty placeholders for those fields.
    """

    db_path = state_dir / "runtime_state.sqlite3"
    if not db_path.exists():
        return None
    try:
        import sqlite3

        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=10
        )
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT brief_snapshot_json FROM runs "
                "WHERE brief_snapshot_json IS NOT NULL "
                "ORDER BY started_at DESC, id DESC "
                "LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        snapshot = json.loads(row["brief_snapshot_json"] or "{}")
        return snapshot if isinstance(snapshot, dict) else None
    except (sqlite3.OperationalError, sqlite3.DatabaseError, ValueError, json.JSONDecodeError):
        return None


def _brief_ids_in_state_dir(state_dir: Path) -> list[str]:
    """Enumerate distinct ``brief_id`` values represented in this state-dir.

    A single state-dir typically corresponds to one brief, but
    historical re-keys or snapshot rebuilds can leave more than one
    brief_id in the candidates table. Sync each separately so the
    dataset name stays scoped per brief.
    """

    db_path = state_dir / "runtime_state.sqlite3"
    if not db_path.exists():
        return []
    try:
        import sqlite3

        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=10
        )
        try:
            rows = conn.execute(
                "SELECT DISTINCT brief_id FROM candidates "
                "WHERE brief_id IS NOT NULL AND judgment_accuracy IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
        return [row[0] for row in rows if row[0]]
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []


def _existing_identity_keys(client_inner: object, dataset_name: str) -> set[str]:
    """Best-effort fetch of identity_keys already present in the dataset.

    Used for idempotency: we skip a row whose identity_key already
    appears in the dataset. The Langfuse SDK surface for listing
    dataset items varies across major versions — try the v3 path
    first, fall back to v2. On any error, return an empty set
    (so the sync proceeds; the SDK's own create-with-duplicate-id
    behavior becomes the secondary safety net).
    """

    try:
        # v3 surface: dataset = client.api.datasets.get(name=dataset_name)
        if hasattr(client_inner, "api") and hasattr(
            getattr(client_inner, "api", None), "datasets"
        ):
            dataset = client_inner.api.datasets.get(name=dataset_name)  # type: ignore[attr-defined]
            items = getattr(dataset, "items", []) or []
            keys: set[str] = set()
            for item in items:
                meta = getattr(item, "metadata", None) or {}
                if isinstance(meta, dict):
                    key = meta.get("identity_key")
                    if isinstance(key, str) and key:
                        keys.add(key)
            return keys
        # v2 surface: client.get_dataset(name=dataset_name)
        if hasattr(client_inner, "get_dataset"):
            dataset = client_inner.get_dataset(name=dataset_name)  # type: ignore[attr-defined]
            items = getattr(dataset, "items", []) or []
            keys = set()
            for item in items:
                meta = getattr(item, "metadata", None) or {}
                if isinstance(meta, dict):
                    key = meta.get("identity_key")
                    if isinstance(key, str) and key:
                        keys.add(key)
            return keys
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "could not fetch existing dataset items for %s: %s",
            dataset_name,
            exc.__class__.__name__,
        )
    return set()


def _ensure_dataset(client_inner: object, dataset_name: str) -> None:
    """Create the dataset if it doesn't exist. Idempotent."""

    try:
        if hasattr(client_inner, "create_dataset"):
            client_inner.create_dataset(name=dataset_name)  # type: ignore[attr-defined]
            return
        if hasattr(client_inner, "api") and hasattr(
            getattr(client_inner, "api", None), "datasets"
        ):
            client_inner.api.datasets.create(name=dataset_name)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        # Most "already exists" errors raise here; that's fine.
        logger.debug(
            "create_dataset(%s) raised (likely already exists): %s",
            dataset_name,
            exc.__class__.__name__,
        )


def _create_dataset_item(
    client_inner: object,
    *,
    dataset_name: str,
    row: LangfuseDatasetRow,
) -> None:
    """Push one dataset row to Langfuse via the SDK."""

    item = row.to_dataset_item()
    if hasattr(client_inner, "create_dataset_item"):
        client_inner.create_dataset_item(  # type: ignore[attr-defined]
            dataset_name=dataset_name,
            input=item["input"],
            expected_output=item["expected_output"],
            metadata=item["metadata"],
        )
    elif hasattr(client_inner, "api") and hasattr(
        getattr(client_inner, "api", None), "dataset_items"
    ):
        client_inner.api.dataset_items.create(  # type: ignore[attr-defined]
            dataset_name=dataset_name,
            input=item["input"],
            expected_output=item["expected_output"],
            metadata=item["metadata"],
        )
    else:
        raise RuntimeError(
            "Langfuse SDK has no recognized dataset_items create surface; "
            "tested v2 (create_dataset_item) + v3 (api.dataset_items.create)."
        )


def _push_with_rate_limit_handling(
    client_inner: object,
    *,
    dataset_name: str,
    row: LangfuseDatasetRow,
) -> bool:
    """Push one row with bounded retry on rate-limit responses.

    Returns ``True`` on success; ``False`` after exhausting retries.
    Detection: a 429 status raises an SDK exception whose
    ``status_code`` attribute (Langfuse's convention) is 429. The
    catch parses ``Retry-After`` from the response when available.
    """

    import re

    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            _create_dataset_item(
                client_inner, dataset_name=dataset_name, row=row
            )
            return True
        except Exception as exc:  # noqa: BLE001
            status_code = getattr(exc, "status_code", None)
            if status_code != 429:
                logger.warning(
                    "create_dataset_item failed for identity_key=%s: %s",
                    row.identity_key,
                    exc,
                )
                return False
            # 429: parse Retry-After if available.
            retry_after_raw = ""
            response = getattr(exc, "response", None)
            if response is not None:
                headers = getattr(response, "headers", None) or {}
                if isinstance(headers, dict):
                    retry_after_raw = str(headers.get("Retry-After", ""))
            try:
                retry_after = float(retry_after_raw)
            except ValueError:
                retry_after_match = re.search(r"\d+\.?\d*", retry_after_raw)
                retry_after = (
                    float(retry_after_match.group(0))
                    if retry_after_match
                    else 0.0
                )
            if retry_after <= 0:
                # Exponential backoff fallback when Retry-After is
                # absent / unparseable. Capped at the documented limit
                # so a sustained outage doesn't pause the sync forever.
                retry_after = min(
                    (2 ** attempt) + 0.5, RETRY_AFTER_FALLBACK_CAP_S
                )
            logger.info(
                "Langfuse rate-limited; sleeping %.1fs before retry "
                "(attempt %d/%d)",
                retry_after,
                attempt + 1,
                MAX_RATE_LIMIT_RETRIES,
            )
            time.sleep(retry_after)
    return False


def sync_one(
    *,
    state_dir: Path,
    source: str,
    brief_id: str,
    dry_run: bool,
) -> SyncResult:
    """Sync one (source, brief_id) combination."""

    db_path = state_dir / "runtime_state.sqlite3"
    brief_dict = _brief_dict_for_state_dir(state_dir)
    rows = build_langfuse_dataset_rows(
        db_path,
        brief_id=brief_id,
        brief_dict=brief_dict,
        source=source,
    )

    dataset_name = dataset_name_for(source=source, brief_id=brief_id)

    if dry_run:
        logger.info(
            "DRY-RUN: would push %d row(s) to dataset %s",
            len(rows),
            dataset_name,
        )
        return SyncResult(
            source=source,
            brief_id=brief_id,
            dataset_name=dataset_name,
            rows_built=len(rows),
            rows_pushed=0,
            rows_skipped_idempotent=0,
            failed_count=0,
            dry_run=True,
        )

    from shared.observability import is_active
    from shared.observability.langfuse_client import flush, get_client

    if not is_active():
        # The whole point of the sync tool is the Langfuse push. No-op
        # on degraded would silently look like success.
        raise RuntimeError(
            "Langfuse client is null / disabled / network-degraded. "
            "Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (and clear "
            "LANGFUSE_DISABLE) before running the sync."
        )

    client = get_client()
    inner = getattr(client, "_inner", None)
    if inner is None:
        raise RuntimeError(
            "Langfuse client has no inner SDK reference — sync cannot "
            "proceed."
        )

    _ensure_dataset(inner, dataset_name)
    existing_keys = _existing_identity_keys(inner, dataset_name)

    pushed = 0
    skipped = 0
    failed = 0
    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        for row in batch:
            if row.identity_key in existing_keys:
                skipped += 1
                continue
            ok = _push_with_rate_limit_handling(
                inner, dataset_name=dataset_name, row=row
            )
            if ok:
                pushed += 1
                # Track the just-pushed key so within-batch duplicates
                # (rare but possible — multiple markers per candidate)
                # don't double-emit.
                existing_keys.add(row.identity_key)
            else:
                failed += 1
        if batch_start + BATCH_SIZE < len(rows):
            # Inter-batch sleep so the Langfuse free-tier rate ceiling
            # doesn't saturate on first-sync floods.
            time.sleep(INTER_BATCH_SLEEP_S)

    flush()

    return SyncResult(
        source=source,
        brief_id=brief_id,
        dataset_name=dataset_name,
        rows_built=len(rows),
        rows_pushed=pushed,
        rows_skipped_idempotent=skipped,
        failed_count=failed,
        dry_run=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sync_judgment_datasets",
        description=(
            "Sync recruiter judgment_accuracy markers into Langfuse "
            "datasets. Idempotent + rate-limit-aware."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Cloris output root containing state/<source>/<state_key>/",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Filter to one source (linkedin / github / researcher / designer / exec_search). "
        "Defaults to all sources.",
    )
    parser.add_argument(
        "--brief-id",
        default=None,
        help="Filter to one brief_id. Defaults to all briefs in the output root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build dataset rows but don't push to Langfuse.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="DEBUG-level logs.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_root = Path(args.output_root)
    if not output_root.exists():
        logger.error("output root does not exist: %s", output_root)
        return 1

    sources = [args.source] if args.source else None
    state_dirs = _discover_state_dirs(output_root=output_root, sources=sources)
    if not state_dirs:
        logger.warning(
            "no state-dirs found under %s/state (sources=%s)",
            output_root,
            sources,
        )
        return 0

    aggregate_results: list[SyncResult] = []
    for source, state_dir in state_dirs:
        brief_ids = _brief_ids_in_state_dir(state_dir)
        if args.brief_id:
            brief_ids = [b for b in brief_ids if b == args.brief_id]
        for brief_id in brief_ids:
            try:
                result = sync_one(
                    state_dir=state_dir,
                    source=source,
                    brief_id=brief_id,
                    dry_run=args.dry_run,
                )
                aggregate_results.append(result)
                logger.info(
                    "synced %s/%s → %s: built=%d pushed=%d skipped=%d failed=%d%s",
                    source,
                    brief_id,
                    result.dataset_name,
                    result.rows_built,
                    result.rows_pushed,
                    result.rows_skipped_idempotent,
                    result.failed_count,
                    " (dry-run)" if result.dry_run else "",
                )
            except RuntimeError as exc:
                logger.error("sync failed for %s/%s: %s", source, brief_id, exc)
                return 2

    total_failed = sum(r.failed_count for r in aggregate_results)
    if total_failed > 0:
        logger.warning(
            "sync completed with %d failed row push(es) across %d sync(s)",
            total_failed,
            len(aggregate_results),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
