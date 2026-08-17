#!/usr/bin/env python3
"""Refresh data/ossf_criticality_snapshot.csv from the OSSF upstream.

OSSF publishes the criticality_score dataset weekly — see
https://github.com/ossf/criticality_score and the published GCS bucket
at https://commondatastorage.googleapis.com/ossf-criticality-score/.

This script pulls a CSV snapshot from a configurable upstream URL,
validates the format, and writes ``data/ossf_criticality_snapshot.csv``
with an updated metadata header. It is invokable as a module:

    python -m scripts.refresh_ossf_criticality \\
        --source-url https://commondatastorage.googleapis.com/ossf-criticality-score/index.html \\
        --top-n 200

Audit Move #20: this is the script half. The library half is the
staleness warning emitted by :mod:`github.ossf_criticality` at load
time. Wiring the script into a cron job is an operator concern, not an
in-process concern.

Output schema (matches the existing snapshot format consumed by
:mod:`github.ossf_criticality`):

    # snapshot_source: <source_url>
    # snapshot_date: YYYY-MM-DD (UTC)
    # refresh_cadence_target: weekly (manual until tooling lands)
    # notes: <free-text>
    repo,score
    kubernetes/kubernetes,0.99432
    rust-lang/rust,0.97845
    ...

Exit codes:
    0 — snapshot refreshed successfully
    1 — validation failure (no upstream rows parsable)
    2 — I/O failure (network, file write, etc.)
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests


# Default upstream. OSSF moves the published location periodically; the
# operator can override via --source-url. Pinned to the criticality_score
# repo's "latest" CSV release per their release page convention.
DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/ossf/criticality_score/main/"
    "data/criticality_score.csv"
)

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "ossf_criticality_snapshot.csv"
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refresh_ossf_criticality",
        description=(
            "Refresh data/ossf_criticality_snapshot.csv from the OSSF "
            "upstream. Cron wiring is an operator concern."
        ),
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help=(
            "Upstream URL for the OSSF criticality CSV. Override when the "
            "upstream publishing location moves."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the snapshot. Defaults to data/ossf_criticality_snapshot.csv.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help=(
            "If > 0, keep only the top N rows by score. The shipped "
            "snapshot is small (calibration anchors); production refresh "
            "may want the full dataset (top-n=0)."
        ),
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Read the CSV from a local file instead of the network. "
            "Useful for one-off refreshes from a manually-downloaded "
            "snapshot."
        ),
    )
    return parser


def _fetch_csv(*, source_url: str, input_file: str | None) -> str:
    """Return the upstream CSV text. ``--input-file`` short-circuits
    the network and is the preferred path for offline/manual refreshes.
    """

    if input_file:
        return Path(input_file).read_text(encoding="utf-8")
    response = requests.get(source_url, timeout=60)
    response.raise_for_status()
    return response.text


def _parse_rows(csv_text: str) -> list[tuple[str, float]]:
    """Parse the upstream CSV into ``(repo, score)`` tuples.

    Tolerant of upstream column-naming churn: looks for the first column
    that contains a slash (assumed to be ``owner/repo``) and the column
    named ``criticality_score`` or ``score``.
    """

    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        return []
    fieldnames = [f.strip() for f in reader.fieldnames]

    score_col = None
    for candidate in ("criticality_score", "score", "default_score"):
        if candidate in fieldnames:
            score_col = candidate
            break
    if score_col is None:
        return []

    repo_col = None
    for candidate in ("repo", "url", "name"):
        if candidate in fieldnames:
            repo_col = candidate
            break
    if repo_col is None:
        return []

    rows: list[tuple[str, float]] = []
    for raw in reader:
        repo_value = (raw.get(repo_col) or "").strip()
        # Accept either bare "owner/repo" or full URL; reduce to the
        # final two path segments.
        repo_key = _normalize_repo_key(repo_value)
        if not repo_key:
            continue
        try:
            score = float(raw.get(score_col, "") or "")
        except ValueError:
            continue
        score = max(0.0, min(score, 1.0))
        rows.append((repo_key, score))
    return rows


def _normalize_repo_key(value: str) -> str:
    """Reduce ``https://github.com/foo/bar`` (or similar) to ``foo/bar``."""

    value = value.strip().lower()
    if not value:
        return ""
    # Strip protocol + host.
    for prefix in ("https://github.com/", "http://github.com/", "github.com/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    # Drop trailing slashes / query strings.
    value = value.rstrip("/").split("?", 1)[0]
    parts = [p for p in value.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return ""


def _write_snapshot(
    *,
    output_path: Path,
    rows: Iterable[tuple[str, float]],
    source_url: str,
) -> int:
    """Write the snapshot CSV with the metadata header. Returns the row
    count actually written."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    today_iso = datetime.now(timezone.utc).date().isoformat()

    written = 0
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# snapshot_source: {source_url}\n")
        fh.write(f"# snapshot_date: {today_iso}\n")
        fh.write("# refresh_cadence_target: weekly (manual until tooling lands)\n")
        fh.write(
            "# notes: Refreshed via scripts/refresh_ossf_criticality.py.\n"
        )
        writer = csv.writer(fh)
        writer.writerow(["repo", "score"])
        for repo, score in rows:
            writer.writerow([repo, f"{score:.5f}"])
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        csv_text = _fetch_csv(
            source_url=args.source_url,
            input_file=args.input_file,
        )
    except (requests.RequestException, OSError) as exc:
        sys.stderr.write(
            f"refresh_ossf_criticality: fetch failed ({exc!r}); "
            "the existing snapshot is unchanged.\n"
        )
        return 2

    rows = _parse_rows(csv_text)
    if not rows:
        sys.stderr.write(
            "refresh_ossf_criticality: parsed zero rows from upstream "
            f"(source={args.source_url}); the existing snapshot is unchanged. "
            "Upstream column names may have shifted — inspect the source "
            "and update _parse_rows() column candidates if needed.\n"
        )
        return 1

    rows.sort(key=lambda pair: pair[1], reverse=True)
    if args.top_n > 0:
        rows = rows[: args.top_n]

    output_path = Path(args.output_path)
    written = _write_snapshot(
        output_path=output_path,
        rows=rows,
        source_url=args.source_url,
    )
    sys.stdout.write(
        f"refresh_ossf_criticality: wrote {written} rows to {output_path}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
