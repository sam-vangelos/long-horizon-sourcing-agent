"""Pure-read projection drift sweep (engineering-quality remediation D2).

``runtime_state.sqlite3`` is canonical for sourcing runtime state;
``progress.json`` beside a run's artifacts is a compatibility PROJECTION.
Nothing today detects when the two disagree — a projection can go stale or be
hand-edited, and no tool reports it.

This sweep is the detector. For a state directory it rebuilds the progress
object from SQLite via the official projectors
(:func:`shared.runtime_state.projections.project_linkedin_progress` /
:func:`shared.runtime_state.projections.project_github_progress`) and diffs
that canonical view against the on-disk ``progress.json``. Drift is reported
field-by-field; absent inputs are recorded as ``missing`` (not drift).

PURE READ. Opens the runtime DB read-only via ``file:...?mode=ro`` (not
``immutable=1``) so the sweep sees WAL-resident commits from a concurrently
running or recently finished run; ``immutable=1`` would disable locking and
hide uncheckpointed WAL pages, producing false ``missing`` or drift. Never
calls :class:`~shared.runtime_state.store.RuntimeStateStore` ``initialize()``;
never writes to any DB or state-dir file.

Most-recent run resolution (when ``run_id`` is ``None``): the highest
``runs.id`` among rows whose ``source`` matches ``module`` (``linkedin`` or
``github``), ordered by ``started_at DESC, id DESC`` so lexically later
timestamps win and ``id`` breaks ties.

CLI exit codes:
  0 — no drift (``missing`` entries alone still exit 0)
  1 — at least one field drift
  2 — usage or I/O error
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from github.schemas import GitHubProgress
from shared.runtime_state.projections import (
    project_github_progress,
    project_linkedin_progress,
)
from shared.runtime_state.store import RuntimeStateStore
from shared.schemas import Progress

_PROGRESS_FILE = "progress.json"
_DB_FILE = "runtime_state.sqlite3"


@dataclass(frozen=True)
class FieldDrift:
    """One scalar or per-unit field that differs between disk and canonical."""

    field: str
    on_disk: Any
    canonical: Any


@dataclass(frozen=True)
class ProjectionDriftReport:
    """Outcome of comparing on-disk progress.json to a canonical projection."""

    state_dir: Path
    module: str
    run_id: int | None
    drift: tuple[FieldDrift, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def has_drift(self) -> bool:
        return bool(self.drift)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_dir": str(self.state_dir),
            "module": self.module,
            "run_id": self.run_id,
            "has_drift": self.has_drift,
            "missing": list(self.missing),
            "drift": [asdict(row) for row in self.drift],
        }


@contextmanager
def _readonly(db_path: Path) -> Iterator[sqlite3.Connection | None]:
    """Open ``db_path`` read-only; yield ``None`` if absent or unreadable."""

    if not db_path.exists():
        yield None
        return
    try:
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
        )
    except sqlite3.Error:
        yield None
        return
    conn.row_factory = sqlite3.Row
    try:
        try:
            conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error:
            conn.close()
            yield None
            return
        yield conn
    finally:
        conn.close()


class _ReadOnlyRuntimeStateStore(RuntimeStateStore):
    """RuntimeStateStore read surface without schema bootstrap or writes."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._state_dir = self.db_path.parent

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"runtime state DB not found: {self.db_path}")
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
            )
        except sqlite3.Error as exc:
            raise FileNotFoundError(f"runtime state DB not readable: {self.db_path}") from exc
        conn.row_factory = sqlite3.Row
        try:
            try:
                conn.execute("PRAGMA busy_timeout=5000")
            except sqlite3.Error as exc:
                conn.close()
                raise FileNotFoundError(
                    f"runtime state DB not readable: {self.db_path}"
                ) from exc
            yield conn
        finally:
            conn.close()


def _resolve_run_id(db_path: Path, *, module: str, run_id: int | None) -> int | None:
    if run_id is not None:
        with _readonly(db_path) as conn:
            if conn is None:
                return None
            try:
                row = conn.execute(
                    "SELECT id FROM runs WHERE id = ? AND source = ?",
                    (run_id, module),
                ).fetchone()
            except sqlite3.Error:
                return None
        return int(row["id"]) if row else None

    with _readonly(db_path) as conn:
        if conn is None:
            return None
        try:
            row = conn.execute(
                """
                SELECT id FROM runs
                WHERE source = ?
                ORDER BY started_at DESC, id DESC
                LIMIT 1
                """,
                (module,),
            ).fetchone()
        except sqlite3.Error:
            return None
    return int(row["id"]) if row else None


def _db_is_readable(db_path: Path) -> bool:
    """Return False when the DB file exists but cannot be queried."""

    with _readonly(db_path) as conn:
        if conn is None:
            return False
        try:
            conn.execute("SELECT 1").fetchone()
        except sqlite3.Error:
            return False
    return True


def _load_on_disk_progress(
    path: Path,
    *,
    module: str,
) -> tuple[Progress | GitHubProgress | None, str | None]:
    """Load on-disk progress; return ``(obj, missing_label)``.

  ``missing_label`` is ``"progress.json:unreadable"`` when the file exists
  but cannot be parsed into a progress object; ``None`` on success or when
  the file is simply absent.
    """

    if not path.exists():
        return None, None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, f"{_PROGRESS_FILE}:unreadable"
    if not isinstance(raw, dict):
        return None, f"{_PROGRESS_FILE}:unreadable"
    try:
        if module == "linkedin":
            return Progress.from_dict(raw), None
        return GitHubProgress.from_dict(raw), None
    except (TypeError, AttributeError, ValueError):
        return None, f"{_PROGRESS_FILE}:unreadable"


def _project_canonical(
    store: _ReadOnlyRuntimeStateStore, *, module: str, run_id: int
) -> Progress | GitHubProgress:
    if module == "linkedin":
        return project_linkedin_progress(store, run_id)
    return project_github_progress(store, run_id)


def _append_scalar_drift(
    drift: list[FieldDrift],
    *,
    field: str,
    on_disk: Any,
    canonical: Any,
) -> None:
    if on_disk != canonical:
        drift.append(FieldDrift(field=field, on_disk=on_disk, canonical=canonical))


def _compare_linkedin(on_disk: Progress, canonical: Progress) -> list[FieldDrift]:
    drift: list[FieldDrift] = []
    for field in (
        "brief_name",
        "candidates_saved",
        "candidates_rejected",
        "current_string_id",
        "current_page",
        "pivot_count",
    ):
        _append_scalar_drift(
            drift,
            field=field,
            on_disk=getattr(on_disk, field),
            canonical=getattr(canonical, field),
        )

    on_disk_by_id = {string.id: string for string in on_disk.strings}
    canonical_by_id = {string.id: string for string in canonical.strings}
    for string_id in sorted(set(on_disk_by_id) | set(canonical_by_id)):
        disk_string = on_disk_by_id.get(string_id)
        canon_string = canonical_by_id.get(string_id)
        _append_scalar_drift(
            drift,
            field=f"strings[{string_id}].status",
            on_disk=None if disk_string is None else disk_string.status,
            canonical=None if canon_string is None else canon_string.status,
        )
        disk_saves_len = 0 if disk_string is None else len(disk_string.saves)
        canon_saves_len = 0 if canon_string is None else len(canon_string.saves)
        _append_scalar_drift(
            drift,
            field=f"strings[{string_id}].saves_count",
            on_disk=disk_saves_len,
            canonical=canon_saves_len,
        )
    return drift


def _compare_github(on_disk: GitHubProgress, canonical: GitHubProgress) -> list[FieldDrift]:
    drift: list[FieldDrift] = []
    for field in (
        "brief_name",
        "candidates_discovered",
        "candidates_enriched",
        "candidates_saved",
        "candidates_rejected",
        "candidates_insufficient",
        "current_query_id",
        "api_calls_made",
    ):
        _append_scalar_drift(
            drift,
            field=field,
            on_disk=getattr(on_disk, field),
            canonical=getattr(canonical, field),
        )

    on_disk_by_id = {query.id: query for query in on_disk.queries}
    canonical_by_id = {query.id: query for query in canonical.queries}
    for query_id in sorted(set(on_disk_by_id) | set(canonical_by_id)):
        disk_query = on_disk_by_id.get(query_id)
        canon_query = canonical_by_id.get(query_id)
        _append_scalar_drift(
            drift,
            field=f"queries[{query_id}].status",
            on_disk=None if disk_query is None else disk_query.status,
            canonical=None if canon_query is None else canon_query.status,
        )
        disk_saves_len = 0 if disk_query is None else len(disk_query.saves)
        canon_saves_len = 0 if canon_query is None else len(canon_query.saves)
        _append_scalar_drift(
            drift,
            field=f"queries[{query_id}].saves_count",
            on_disk=disk_saves_len,
            canonical=canon_saves_len,
        )
    return drift


def sweep_projection_drift(
    state_dir: Path,
    *,
    run_id: int | None = None,
    module: str = "linkedin",
) -> ProjectionDriftReport:
    """Compare on-disk ``progress.json`` to the canonical SQLite projection.

    Absent DB, run row, or projection file are recorded in ``missing``; the
    sweep never raises for ordinary missing inputs.
    """

    state_dir = Path(state_dir)
    if module not in {"linkedin", "github"}:
        raise ValueError(f"unsupported module: {module}")

    db_path = state_dir / _DB_FILE
    progress_path = state_dir / _PROGRESS_FILE
    missing: list[str] = []

    if not db_path.exists():
        missing.append(_DB_FILE)
        return ProjectionDriftReport(
            state_dir=state_dir,
            module=module,
            run_id=run_id,
            missing=tuple(missing),
        )

    if not _db_is_readable(db_path):
        missing.append(f"{_DB_FILE}:unreadable")
        return ProjectionDriftReport(
            state_dir=state_dir,
            module=module,
            run_id=run_id,
            missing=tuple(missing),
        )

    resolved_run_id = _resolve_run_id(db_path, module=module, run_id=run_id)
    if resolved_run_id is None:
        missing.append(f"run:{module}" if run_id is None else f"run:{run_id}")
        return ProjectionDriftReport(
            state_dir=state_dir,
            module=module,
            run_id=run_id,
            missing=tuple(missing),
        )

    on_disk, progress_missing = _load_on_disk_progress(progress_path, module=module)
    if on_disk is None:
        missing.append(progress_missing or _PROGRESS_FILE)
        return ProjectionDriftReport(
            state_dir=state_dir,
            module=module,
            run_id=resolved_run_id,
            missing=tuple(missing),
        )

    store = _ReadOnlyRuntimeStateStore(db_path)
    try:
        canonical = _project_canonical(store, module=module, run_id=resolved_run_id)
    except ValueError:
        missing.append(f"run:{resolved_run_id}")
        return ProjectionDriftReport(
            state_dir=state_dir,
            module=module,
            run_id=resolved_run_id,
            missing=tuple(missing),
        )

    if module == "linkedin":
        drift = _compare_linkedin(on_disk, canonical)  # type: ignore[arg-type]
    else:
        drift = _compare_github(on_disk, canonical)  # type: ignore[arg-type]

    return ProjectionDriftReport(
        state_dir=state_dir,
        module=module,
        run_id=resolved_run_id,
        drift=tuple(drift),
        missing=tuple(missing),
    )


def _format_report(report: ProjectionDriftReport) -> str:
    lines = [
        f"Projection drift sweep: {report.state_dir}",
        f"Module: {report.module}, run_id: {report.run_id}",
    ]
    if report.missing:
        lines.append(f"Missing: {', '.join(report.missing)}")
    if not report.has_drift:
        lines.append("No drift detected.")
        return "\n".join(lines)
    lines.append(f"Drift ({len(report.drift)} field(s)):")
    for row in report.drift:
        lines.append(
            f"  {row.field}: on_disk={row.on_disk!r} canonical={row.canonical!r}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report drift between progress.json and runtime_state.sqlite3",
    )
    parser.add_argument("state_dir", type=Path, help="State directory to inspect")
    parser.add_argument("--run-id", type=int, default=None, help="Run id (default: latest)")
    parser.add_argument(
        "--module",
        choices=["linkedin", "github"],
        default="linkedin",
        help="Source module namespace (default: linkedin)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    try:
        report = sweep_projection_drift(
            args.state_dir,
            run_id=args.run_id,
            module=args.module,
        )
    except (ValueError, OSError, sqlite3.Error, TypeError, AttributeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(_format_report(report))

    if any(":unreadable" in entry for entry in report.missing):
        return 2
    return 1 if report.has_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
