"""Tests for the pure-read progress.json projection drift sweep (D2)."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from shared.runtime_state import RuntimeStateStore
from shared.runtime_state.projections import write_linkedin_progress_projection
from shared.runtime_state.store import LINKEDIN_STRING_KIND
from shared.schemas import SearchString

from tools.projection_drift_sweep import main as sweep_main, sweep_projection_drift


def _make_store(state_dir: Path) -> RuntimeStateStore:
    return RuntimeStateStore(state_dir / "runtime_state.sqlite3")


def _seed_linkedin_state(
    state_dir: Path,
    *,
    store: RuntimeStateStore | None = None,
) -> tuple[RuntimeStateStore, int]:
    store = store if store is not None else _make_store(state_dir)
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-1",
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": "brief-1"},
    )
    store.update_run_resume_state(
        run_id,
        {
            "brief_name": "brief-1",
            "current_string_id": 1,
            "current_page": 2,
            "candidates_saved": 1,
            "candidates_rejected": 0,
            "pivot_count": 0,
        },
    )
    done_string = SearchString(
        id=1,
        name="Payments edge case",
        boolean="payments AND fednow",
        status="done",
        saves=["https://linkedin.com/in/unique-candidate-xyzzy"],
        family_key="payments",
        novelty_bucket="edge_case",
        domain_lane="payments",
    )
    queued_string = SearchString(
        id=2,
        name="Capital markets canonical",
        boolean="capital markets AND workflow",
        status="queued",
        family_key="capital_markets",
        novelty_bucket="canonical",
        domain_lane="capital_markets",
    )
    store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        kind=LINKEDIN_STRING_KIND,
        source_unit_id=str(done_string.id),
        display_name=done_string.name,
        ordering_index=0,
        status="done",
        payload=done_string.to_dict(),
        family_key=done_string.family_key,
        novelty_bucket=done_string.novelty_bucket,
        domain_lane=done_string.domain_lane,
        counters={"saves_count": len(done_string.saves)},
    )
    store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        kind=LINKEDIN_STRING_KIND,
        source_unit_id=str(queued_string.id),
        display_name=queued_string.name,
        ordering_index=1,
        status="queued",
        payload=queued_string.to_dict(),
        family_key=queued_string.family_key,
        novelty_bucket=queued_string.novelty_bucket,
        domain_lane=queued_string.domain_lane,
    )
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-1",
        identity_key="https://linkedin.com/in/unique-candidate-xyzzy",
        display_name="UNIQUE_CANDIDATE_NAME_XYZZY",
        profile_url="https://linkedin.com/in/unique-candidate-xyzzy",
    )
    return store, run_id


def _snapshot_state_dir(state_dir: Path) -> tuple[list[str], tuple[int, int] | None]:
    files = sorted(str(path.relative_to(state_dir)) for path in state_dir.rglob("*") if path.is_file())
    db_path = state_dir / "runtime_state.sqlite3"
    db_stat = None
    if db_path.exists():
        stat = db_path.stat()
        db_stat = (stat.st_mtime_ns, stat.st_size)
    return files, db_stat


def test_sweep_reports_no_drift_when_projection_matches_canonical(tmp_path: Path) -> None:
    store, run_id = _seed_linkedin_state(tmp_path)
    write_linkedin_progress_projection(store, run_id, tmp_path / "progress.json")

    report = sweep_projection_drift(tmp_path, run_id=run_id, module="linkedin")

    assert report.has_drift is False
    assert report.drift == ()
    assert report.missing == ()


def test_sweep_detects_scalar_field_drift(tmp_path: Path) -> None:
    store, run_id = _seed_linkedin_state(tmp_path)
    write_linkedin_progress_projection(store, run_id, tmp_path / "progress.json")

    progress_path = tmp_path / "progress.json"
    on_disk = json.loads(progress_path.read_text())
    on_disk["candidates_saved"] = 99
    progress_path.write_text(json.dumps(on_disk))

    report = sweep_projection_drift(tmp_path, run_id=run_id, module="linkedin")

    assert report.has_drift is True
    drift_by_field = {row.field: row for row in report.drift}
    assert "candidates_saved" in drift_by_field
    assert drift_by_field["candidates_saved"].on_disk == 99
    assert drift_by_field["candidates_saved"].canonical == 1


def test_sweep_detects_per_string_status_drift(tmp_path: Path) -> None:
    store, run_id = _seed_linkedin_state(tmp_path)
    write_linkedin_progress_projection(store, run_id, tmp_path / "progress.json")

    progress_path = tmp_path / "progress.json"
    on_disk = json.loads(progress_path.read_text())
    on_disk["strings"][0]["status"] = "error"
    progress_path.write_text(json.dumps(on_disk))

    report = sweep_projection_drift(tmp_path, run_id=run_id, module="linkedin")

    assert report.has_drift is True
    assert any(row.field == "strings[1].status" for row in report.drift)


def test_sweep_reports_missing_rather_than_drift_for_absent_progress_json(tmp_path: Path) -> None:
    store, run_id = _seed_linkedin_state(tmp_path)

    report = sweep_projection_drift(tmp_path, run_id=run_id, module="linkedin")

    assert report.has_drift is False
    assert "progress.json" in report.missing


def test_sweep_writes_nothing(tmp_path: Path) -> None:
    store, run_id = _seed_linkedin_state(tmp_path)
    write_linkedin_progress_projection(store, run_id, tmp_path / "progress.json")

    files_before, db_stat_before = _snapshot_state_dir(tmp_path)

    report = sweep_projection_drift(tmp_path, run_id=run_id, module="linkedin")

    files_after, db_stat_after = _snapshot_state_dir(tmp_path)

    assert report.has_drift is False
    # mode=ro can touch WAL sidecars (-wal/-shm) during read; scope the
    # no-write guarantee to the main DB file and report artifacts only.
    db_path = tmp_path / "runtime_state.sqlite3"
    db_files_before = {path for path in files_before if path == db_path.name}
    db_files_after = {path for path in files_after if path == db_path.name}
    report_files_before = {path for path in files_before if path.endswith(".json")}
    report_files_after = {path for path in files_after if path.endswith(".json")}
    assert db_files_before == db_files_after
    assert report_files_before == report_files_after
    assert db_stat_before == db_stat_after


def test_sweep_sees_wal_resident_rows_without_checkpoint(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    store = _make_store(tmp_path)

    # Hold a write connection open so committed rows stay in the -wal sidecar
    # instead of being checkpointed into the main DB before the sweep runs.
    keeper = sqlite3.connect(str(db_path), timeout=5.0)
    keeper.execute("PRAGMA journal_mode=WAL")
    keeper.execute("PRAGMA busy_timeout=5000")
    try:
        store, run_id = _seed_linkedin_state(tmp_path, store=store)
        write_linkedin_progress_projection(store, run_id, tmp_path / "progress.json")

        wal_path = Path(f"{db_path}-wal")
        assert wal_path.is_file(), "precondition: WAL sidecar must exist"
        assert wal_path.stat().st_size > 0, "precondition: WAL sidecar must be non-empty"

        report = sweep_projection_drift(tmp_path, run_id=run_id, module="linkedin")

        assert "runtime_state.sqlite3" not in report.missing
        assert f"run:{run_id}" not in report.missing
        assert "run:linkedin" not in report.missing
        assert report.has_drift is False
    finally:
        keeper.close()


def test_report_contains_no_candidate_identifiers(tmp_path: Path) -> None:
    store, run_id = _seed_linkedin_state(tmp_path)
    write_linkedin_progress_projection(store, run_id, tmp_path / "progress.json")

    progress_path = tmp_path / "progress.json"
    on_disk = json.loads(progress_path.read_text())
    on_disk["candidates_saved"] = 99
    progress_path.write_text(json.dumps(on_disk))

    report = sweep_projection_drift(tmp_path, run_id=run_id, module="linkedin")
    payload = json.dumps(report.to_dict())

    assert "UNIQUE_CANDIDATE_NAME_XYZZY" not in payload
    assert "https://linkedin.com/in/unique-candidate-xyzzy" not in payload


@pytest.mark.parametrize(
    "malformed_payload",
    [
        [],
        {"strings": "not-a-list"},
    ],
)
def test_sweep_reports_missing_for_malformed_progress_json(
    tmp_path: Path,
    malformed_payload: object,
) -> None:
    store, run_id = _seed_linkedin_state(tmp_path)
    write_linkedin_progress_projection(store, run_id, tmp_path / "progress.json")

    progress_path = tmp_path / "progress.json"
    progress_path.write_text(json.dumps(malformed_payload))

    report = sweep_projection_drift(tmp_path, run_id=run_id, module="linkedin")

    assert report.has_drift is False
    assert "progress.json:unreadable" in report.missing


def test_sweep_reports_missing_for_corrupt_database(tmp_path: Path) -> None:
    store, run_id = _seed_linkedin_state(tmp_path)
    write_linkedin_progress_projection(store, run_id, tmp_path / "progress.json")

    db_path = tmp_path / "runtime_state.sqlite3"
    db_path.write_bytes(b"not-a-sqlite-database")

    report = sweep_projection_drift(tmp_path, run_id=run_id, module="linkedin")

    assert report.has_drift is False
    assert "runtime_state.sqlite3:unreadable" in report.missing


def test_cli_exits_2_on_malformed_input(tmp_path: Path) -> None:
    store, run_id = _seed_linkedin_state(tmp_path)
    write_linkedin_progress_projection(store, run_id, tmp_path / "progress.json")

    progress_path = tmp_path / "progress.json"
    progress_path.write_text("[]")

    rc = sweep_main([str(tmp_path), "--run-id", str(run_id), "--module", "linkedin"])
    assert rc == 2

    db_path = tmp_path / "runtime_state.sqlite3"
    write_linkedin_progress_projection(store, run_id, progress_path)
    db_path.write_bytes(b"corrupt-db-bytes")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.projection_drift_sweep",
            str(tmp_path),
            "--run-id",
            str(run_id),
            "--module",
            "linkedin",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
