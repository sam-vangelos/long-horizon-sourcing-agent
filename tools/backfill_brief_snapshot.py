"""Backfill `runs.brief_snapshot_json` for legacy runs (Ledger L8).

Phase F Slice F4. The plan called this "brief role title backfill,"
but `brief_role_title` is a DERIVED value — it's pulled at read time
from `runs.brief_snapshot_json` (see
`cloris.control_plane._extract_brief_role_title`). The actual
backfill target is the snapshot itself: legacy runs (pre-Phase-3
brief identity pinning) carry `brief_snapshot_json='{}'`, which
makes every derived field — `brief_role_title`,
`brief_linkedin_project`, drift-detection — return None.

What this script does
=====================

For every `(source, state_dir)` discovered under `output/state/`:

1. Open the per-state-dir `runtime_state.sqlite3` read/write.
2. SELECT every run with `brief_snapshot_json='{}'` OR empty.
3. For each such run:
   a. Find the brief in `config/` whose `derive_brief_id()` /
      `github_state_key()` matches `runs.brief_id`.
   b. If found: read the brief JSON, write it into
      `brief_snapshot_json` AND update `brief_path_at_launch` +
      `brief_content_hash` to match the canonical pinning shape.
   c. If not found (orphaned run): log a warning, skip.
4. Print a summary: rows scanned, rows backfilled, orphans skipped,
   per-state-dir error counts.

The script is **idempotent**: a row whose snapshot is already
populated is left alone. Running this twice is a no-op. Per-row
errors are caught and logged so one bad row doesn't fail the batch.

Usage
=====

    .venv/bin/python tools/backfill_brief_snapshot.py            # dry-run summary
    .venv/bin/python tools/backfill_brief_snapshot.py --apply    # actually update

Dry-run is the default — the script prints what it WOULD update
without touching disk. `--apply` is required to commit changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

# Make `shared.*` and `cloris.*` importable when this script is run
# directly (e.g. `.venv/bin/python tools/backfill_brief_snapshot.py`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _enumerate_state_dirs(state_root: Path) -> Iterator[tuple[str, Path]]:
    """Walk `output/state/<source>/<state_key>` directories.

    Mirrors `cloris.control_plane.enumerate_state_dirs()` but lives
    here standalone so the script doesn't need to import the API
    layer (which would pull FastAPI at runtime — irrelevant for a
    backfill tool).
    """

    if not state_root.exists() or not state_root.is_dir():
        return
    for source_dir in sorted(state_root.iterdir()):
        if not source_dir.is_dir():
            continue
        for state_dir in sorted(source_dir.iterdir()):
            if state_dir.is_dir() and (state_dir / "runtime_state.sqlite3").exists():
                yield source_dir.name, state_dir


def _scan_briefs(config_dir: Path) -> list[Path]:
    """Walk `config/` for every brief.json / brief-*.json. Mirrors
    `cloris.api._scan_authored_briefs` but path-only — we just need
    the disk locations so we can hash each brief's id.
    """

    if not config_dir.exists() or not config_dir.is_dir():
        return []
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in ("brief-*.json", "brief.json"):
        for path in config_dir.rglob(pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            name = path.name
            if name.endswith("-draft.json") or ".bak-" in name:
                continue
            out.append(path)
    return out


def _build_brief_index(config_dir: Path) -> dict[tuple[str, str], Path]:
    """Index `(source, brief_id) -> brief_path` so the backfill can
    look up a brief by its source-specific state-key.

    A single brief on disk can produce a LinkedIn brief_id (via
    `derive_brief_id`) AND a GitHub brief_id (via
    `github_state_key`); we register it under both.
    """

    from shared.output_paths import github_state_key, derive_brief_id

    index: dict[tuple[str, str], Path] = {}
    for brief_path in _scan_briefs(config_dir):
        try:
            li_key = derive_brief_id(brief_path=str(brief_path))
            index[("linkedin", li_key)] = brief_path
        except Exception:
            pass
        try:
            gh_key = github_state_key(brief_path=str(brief_path))
            index[("github", gh_key)] = brief_path
        except Exception:
            pass
    return index


def _canonical_brief_payload(brief_path: Path) -> tuple[str, str]:
    """Return `(canonical_json, sha256_hex)` for a brief file.

    The canonical form sorts keys + uses `(",", ":")` separators so
    the hash matches the Phase 3 pinning convention used at
    run-start. If we deviated here, drift-detection would falsely
    flag every backfilled run.
    """

    raw = json.loads(brief_path.read_text())
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return canonical, digest


def backfill_one_db(
    db_path: Path,
    brief_index: dict[tuple[str, str], Path],
    source: str,
    *,
    apply: bool,
    project_root: Path,
) -> dict[str, int]:
    """Backfill all empty-snapshot rows in a single state_dir's DB.

    Routes through :class:`shared.runtime_state.store.RuntimeStateStore`
    so the schema migration runs before the SELECT — old state_dirs
    that pre-date the `brief_snapshot_json` column get migrated up
    transparently. This is identical to what every other read/write
    in the API does, so we can't end up at a different schema version
    than production.

    Returns a dict of per-state-dir counts: scanned, backfilled,
    orphans, errors.
    """

    from shared.runtime_state.store import RuntimeStateStore

    counts = {"scanned": 0, "backfilled": 0, "orphans": 0, "errors": 0}

    store = RuntimeStateStore(db_path)
    with store.connect() as conn:
        cur = conn.execute(
            """
            SELECT id, brief_id, brief_snapshot_json
            FROM runs
            WHERE brief_snapshot_json IS NULL
               OR brief_snapshot_json = ''
               OR brief_snapshot_json = '{}'
            ORDER BY id
            """
        )
        rows = cur.fetchall()
        counts["scanned"] = len(rows)

        for row in rows:
            run_id = row["id"]
            brief_id = row["brief_id"]
            brief_path = brief_index.get((source, brief_id))
            if brief_path is None:
                counts["orphans"] += 1
                continue

            try:
                canonical, digest = _canonical_brief_payload(brief_path)
            except Exception as exc:
                counts["errors"] += 1
                sys.stderr.write(
                    f"[backfill] db={db_path} run_id={run_id}: "
                    f"failed to read/canonicalize {brief_path}: {exc}\n"
                )
                continue

            try:
                rel_path = str(brief_path.relative_to(project_root))
            except ValueError:
                rel_path = str(brief_path)

            if apply:
                try:
                    conn.execute(
                        """
                        UPDATE runs
                           SET brief_snapshot_json = ?,
                               brief_path_at_launch = COALESCE(brief_path_at_launch, ?),
                               brief_content_hash = COALESCE(brief_content_hash, ?)
                         WHERE id = ?
                        """,
                        (canonical, rel_path, digest, run_id),
                    )
                except sqlite3.Error as exc:
                    counts["errors"] += 1
                    sys.stderr.write(
                        f"[backfill] db={db_path} run_id={run_id}: "
                        f"UPDATE failed: {exc}\n"
                    )
                    continue
            counts["backfilled"] += 1

        # `connect()` context manager commits on exit — no explicit
        # commit needed.

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill runs.brief_snapshot_json for legacy runs whose "
            "snapshot was never pinned (Phase F Slice F4 / Ledger L8)."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Commit the backfill. Without this flag, the script prints "
            "what it would update without touching any DB."
        ),
    )
    parser.add_argument(
        "--state-root",
        default=None,
        help=(
            "Override `output/state/`. Defaults to the canonical "
            "`shared.output_paths.STATE_ROOT`."
        ),
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help=(
            "Override `config/` (where briefs live). Defaults to the "
            "project-root sibling of this script's parent."
        ),
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent

    if args.state_root is not None:
        state_root = Path(args.state_root)
    else:
        from shared.output_paths import STATE_ROOT

        state_root = STATE_ROOT

    if args.config_dir is not None:
        config_dir = Path(args.config_dir)
    else:
        config_dir = project_root / "config"

    print(
        f"[backfill] state_root={state_root}\n"
        f"[backfill] config_dir={config_dir}\n"
        f"[backfill] apply={'YES' if args.apply else 'no (dry-run)'}\n"
    )

    brief_index = _build_brief_index(config_dir)
    print(f"[backfill] indexed {len(brief_index)} (source, brief_id) keys\n")

    totals = {"scanned": 0, "backfilled": 0, "orphans": 0, "errors": 0}
    state_dir_count = 0
    for source, state_dir in _enumerate_state_dirs(state_root):
        state_dir_count += 1
        db_path = state_dir / "runtime_state.sqlite3"
        counts = backfill_one_db(
            db_path,
            brief_index,
            source,
            apply=args.apply,
            project_root=project_root,
        )
        if counts["scanned"] > 0:
            print(
                f"  {source}/{state_dir.name}: "
                f"scanned={counts['scanned']} "
                f"backfilled={counts['backfilled']} "
                f"orphans={counts['orphans']} "
                f"errors={counts['errors']}"
            )
        for k in totals:
            totals[k] += counts[k]

    print(
        f"\n[backfill] state_dirs={state_dir_count} "
        f"scanned={totals['scanned']} "
        f"backfilled={totals['backfilled']} "
        f"orphans={totals['orphans']} "
        f"errors={totals['errors']}"
    )
    if not args.apply and totals["backfilled"] > 0:
        print("[backfill] dry-run; re-run with --apply to commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
