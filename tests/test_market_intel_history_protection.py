"""MI slice 0: a fold never silently evicts run history (CLO-419).

plans/market-intel-diagnosis-2026-10-06.md section 2.1. Every fold re-opens the
runs listed in the previous artifact's ``evidence_index`` with ``trust=False``.
Before this slice such a run survived only if ``_evidence_matches_brief`` found
the brief's keys in ``run-report.json`` (scrubbed on most snapshots) or in the
snapshot's own ``runtime_state.sqlite3``. The 2026-09-18 disk cleanup deleted
every non-newest snapshot DB, so the next fold would silently drop 46 of the 74
indexed runs, although each one still exists in the live state DB and in the
newest snapshot DB.

Pinned here:

(a) a prior run whose snapshot DB is gone is re-resolved from the live
    ``<state_dir>/runtime_state.sqlite3``, or from the newest sibling snapshot
    DB that holds the same run, with its real runtime summary;
(b) a prior run with no recoverable source keeps its previous index entry and
    is logged and recorded in ``evidence_index.unloaded_runs``;
(c) ``run-manifest.json`` ``brief_id`` and ``market_key`` count as brief-match
    keys;
(d) snapshot and live DBs are opened read-only.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from pathlib import Path

import pytest

from market_intelligence import finalize_run_snapshot, resolve_market_intel_artifact_path
import market_intelligence.agent_backends as agent_backends_mod
import market_intelligence.engine as engine_mod
from shared.brief_loader import load_brief
from shared.output_paths import resolve_linkedin_state_dir
from shared.runtime_state import RuntimeStateStore
from shared.storage import read_json, write_json
from tests.brief_stand_ins import HEAD_AI_LAB_PROJECT_ID
from tests.test_market_intelligence import (
    SOURCE_BRIEF,
    _require_source_brief_path,
    _write_linkedin_inputs,
)


RUN_1_WINDOW = ("2026-04-08T10:00:00+00:00", "2026-04-08T13:00:00+00:00")
RUN_2_WINDOW = ("2026-04-09T10:00:00+00:00", "2026-04-09T13:00:00+00:00")
DB_FILES = ("runtime_state.sqlite3", "runtime_state.sqlite3-wal", "runtime_state.sqlite3-shm")


def _start_run(
    store: RuntimeStateStore,
    *,
    window: tuple[str, str],
    status: str,
    saves: int,
    unit_id: str,
) -> int:
    run_id = store.start_run(
        source="linkedin",
        brief_id=HEAD_AI_LAB_PROJECT_ID,
        output_dir=str(store.db_path.parent),
        mode="full_run",
        resume_state={"brief_name": "head-ai-lab"},
    )
    started_at, ended_at = window
    with store.connect() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, started_at = ?, ended_at = ? WHERE id = ?",
            (status, started_at, ended_at, run_id),
        )
        conn.execute(
            """
            INSERT INTO work_units(
                run_id, source, brief_id, kind, source_unit_id, display_name, ordering_index, status,
                payload_json, checkpoint_json, metrics_json, family_key, novelty_bucket, domain_lane,
                result_count, candidates_discovered, facial_yes_count, facial_no_count, saves_count,
                rejected_count, notes, started_at, ended_at
            )
            VALUES (?, 'linkedin', ?, 'linkedin_string', ?, 'Research copilot lane', 1, 'done',
                    ?, ?, ?, 'research_copilot_asset_mgmt', 'edge_case', 'asset_management',
                    526, 40, 10, 12, ?, 14, 'Strong lane', ?, ?)
            """,
            (
                run_id,
                HEAD_AI_LAB_PROJECT_ID,
                unit_id,
                json.dumps({"boolean": "research copilot lane"}),
                json.dumps({"pages_reviewed": 4, "duplicates_count": 1}),
                json.dumps({"pages_reviewed": 4, "duplicates_count": 1}),
                saves,
                started_at,
                ended_at,
            ),
        )
    return run_id


def _delete_db(directory: Path) -> None:
    for name in DB_FILES:
        (directory / name).unlink(missing_ok=True)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Two finished runs of one campaign, the first already folded into MI.

    Returns a namespace with the brief, the live state dir, both snapshot run
    dirs, and the artifact as written by the first fold. Run 1's snapshot DB is
    then deleted, the way the 2026-09-18 cleanup deleted every non-newest one.
    """

    # No network: every LLM stage takes its heuristic path.
    monkeypatch.setattr(agent_backends_mod, "_has_llm_access", lambda: False)

    brief_path = tmp_path / SOURCE_BRIEF.name
    write_json(brief_path, read_json(_require_source_brief_path()))
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path,
        state_dir=tmp_path / "output" / "state" / "linkedin" / HEAD_AI_LAB_PROJECT_ID,
    )
    _write_linkedin_inputs(
        state_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
        include_report_input=False,
        include_report=False,
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")

    run_1 = _start_run(
        store, window=RUN_1_WINDOW, status="governor_limit_reached", saves=7, unit_id="1"
    )
    run_dir_1 = finalize_run_snapshot(
        source="linkedin", brief_path=brief_path, state_dir=state_dir, run_id=run_1
    )
    engine_mod.update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir_1,
        mode="post_run",
        with_external_research=False,
    )
    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir_1)
    first_artifact = read_json(artifact_path)
    assert [record["run_id"] for record in first_artifact["evidence_index"]["runs"]] == [run_1]

    run_2 = _start_run(
        store, window=RUN_2_WINDOW, status="governor_limit_reached", saves=3, unit_id="2"
    )
    run_dir_2 = finalize_run_snapshot(
        source="linkedin", brief_path=brief_path, state_dir=state_dir, run_id=run_2
    )

    # The cleanup: run 1's snapshot keeps its manifest (which still lists the
    # DB in artifacts_present) but the DB itself is gone.
    _delete_db(run_dir_1)
    assert "runtime_state.sqlite3" in read_json(run_dir_1 / "run-manifest.json")[
        "artifacts_present"
    ]

    class _Campaign:
        pass

    c = _Campaign()
    c.brief_path = brief_path
    c.state_dir = state_dir
    c.store = store
    c.run_1 = run_1
    c.run_2 = run_2
    c.run_dir_1 = run_dir_1
    c.run_dir_2 = run_dir_2
    c.artifact_path = artifact_path
    c.prior_run_1_record = first_artifact["evidence_index"]["runs"][0]
    return c


def _collect(c) -> list:
    raw = read_json(c.brief_path)
    brief = load_brief(str(c.brief_path))
    previous = engine_mod._load_previous_artifact(c.artifact_path)
    return engine_mod._collect_evidence_batches(
        brief_path=c.brief_path,
        brief=brief,
        raw=raw,
        mode="post_run",
        run_dir=c.run_dir_2,
        report_path=None,
        previous_artifact=previous,
        reconstruct_report_analysis=False,
    )


def _batch_for(batches, run_dir: Path):
    matches = [b for b in batches if Path(b.run_dir).resolve() == run_dir.resolve()]
    return matches[0] if matches else None


def _fold(c):
    return engine_mod.update_market_intel(
        brief_path=c.brief_path,
        run_dir=c.run_dir_2,
        mode="post_run",
        with_external_research=False,
    )


def _strip_manifest_identity(run_dir: Path, *, state_dir: str | None = None) -> None:
    manifest = read_json(run_dir / "run-manifest.json")
    manifest.pop("brief_id", None)
    manifest.pop("market_key", None)
    if state_dir is not None:
        manifest["state_dir"] = state_dir
    write_json(run_dir / "run-manifest.json", manifest)


# -- (a) runtime summary recovered from the live DB or the newest snapshot ---


def test_missing_snapshot_db_resolves_runtime_summary_from_live_state_db(campaign):
    c = campaign
    batch = _batch_for(_collect(c), c.run_dir_1)

    assert batch is not None, "prior run with a deleted snapshot DB was evicted"
    summary = batch.runtime_summary
    assert summary["brief_ids"] == [HEAD_AI_LAB_PROJECT_ID]
    assert summary["latest_run_started_at"] == RUN_1_WINDOW[0]
    assert summary["latest_run_status"] == "governor_limit_reached"
    # Run-scoped: run 1's own work unit, not run 2's.
    assert [unit["saves_count"] for unit in summary["work_units"]] == [7]
    assert summary["saved"] == 7


def test_fold_keeps_prior_run_whose_snapshot_db_was_deleted(campaign):
    c = campaign
    artifact = _fold(c)

    run_ids = sorted(record.get("run_id") for record in artifact.evidence_index["runs"])
    assert run_ids == [c.run_1, c.run_2]
    assert artifact.aggregate_metrics["run_count"] == 2
    assert artifact.evidence_index["unloaded_runs"] == []


def test_missing_snapshot_and_live_db_resolve_from_newest_sibling_snapshot(campaign):
    c = campaign
    # The live DB is gone too; run 2's snapshot DB (a copy of the campaign DB
    # taken after run 1 ended) still holds run 1.
    _delete_db(c.state_dir)

    batch = _batch_for(_collect(c), c.run_dir_1)

    assert batch is not None
    assert batch.runtime_summary["brief_ids"] == [HEAD_AI_LAB_PROJECT_ID]
    assert batch.runtime_summary["latest_run_started_at"] == RUN_1_WINDOW[0]
    assert [unit["saves_count"] for unit in batch.runtime_summary["work_units"]] == [7]


def test_fallback_db_holding_a_different_run_under_the_same_id_is_rejected(campaign, caplog):
    """A reset state dir reuses run ids; started_at must agree before we trust it."""
    c = campaign
    _delete_db(c.state_dir)
    _delete_db(c.run_dir_2)
    reset_store = RuntimeStateStore(c.state_dir / "runtime_state.sqlite3")
    reused_id = _start_run(
        reset_store,
        window=("2026-05-01T10:00:00+00:00", "2026-05-01T11:00:00+00:00"),
        status="completed",
        saves=99,
        unit_id="9",
    )
    assert reused_id == c.run_1

    with caplog.at_level(logging.WARNING, logger=engine_mod.log.name):
        batches = _collect(c)

    batch = _batch_for(batches, c.run_dir_1)
    # Still kept (the manifest names this brief), but never with the other
    # run's numbers.
    assert batch is not None
    assert batch.runtime_summary.get("saved", 0) != 99
    assert not batch.runtime_summary.get("work_units")
    assert any("started_at" in record.getMessage() for record in caplog.records)


# -- (b) no recoverable source: keep the prior entry, log, record ------------


def test_unrecoverable_prior_run_keeps_its_prior_index_entry(campaign, caplog):
    c = campaign
    # Older manifest shape with no identity keys, pointing at a state dir that
    # no longer exists: nothing on disk can say which brief run 1 belonged to.
    _strip_manifest_identity(c.run_dir_1, state_dir=str(c.state_dir.parent / "gone"))

    with caplog.at_level(logging.WARNING, logger=engine_mod.log.name):
        artifact = _fold(c)

    runs = artifact.evidence_index["runs"]
    run_1_records = [r for r in runs if r["run_ref"] == c.prior_run_1_record["run_ref"]]
    assert run_1_records == [c.prior_run_1_record]

    unloaded = artifact.evidence_index["unloaded_runs"]
    assert len(unloaded) == 1
    entry = unloaded[0]
    assert entry["run_ref"] == c.prior_run_1_record["run_ref"]
    assert entry["outcome"] == "retained_prior_entry"
    assert entry["reason"]

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        c.prior_run_1_record["run_ref"] in message and entry["reason"] in message
        for message in messages
    ), messages


def test_prior_run_whose_run_dir_was_deleted_keeps_its_prior_index_entry(campaign, caplog):
    c = campaign
    shutil.rmtree(c.run_dir_1)

    with caplog.at_level(logging.WARNING, logger=engine_mod.log.name):
        artifact = _fold(c)

    runs = artifact.evidence_index["runs"]
    assert c.prior_run_1_record in runs
    unloaded = artifact.evidence_index["unloaded_runs"]
    assert [(u["run_ref"], u["outcome"]) for u in unloaded] == [
        (c.prior_run_1_record["run_ref"], "retained_prior_entry")
    ]
    assert unloaded[0]["reason"] == "run_dir_missing"
    assert any("run_dir_missing" in record.getMessage() for record in caplog.records)


def test_retained_entry_survives_a_second_fold(campaign):
    """Retention is durable: the entry is retried, not lost, on the next fold."""
    c = campaign
    shutil.rmtree(c.run_dir_1)
    _fold(c)
    artifact = _fold(c)

    assert c.prior_run_1_record in artifact.evidence_index["runs"]


# -- (c) manifest brief_id and market_key are brief-match keys ---------------


def _matches(c, manifest: dict) -> bool:
    return engine_mod._evidence_matches_brief(
        brief_path=c.brief_path,
        brief=load_brief(str(c.brief_path)),
        raw=read_json(c.brief_path),
        report=None,
        runtime_summary={},
        manifest=manifest,
    )


def test_manifest_brief_id_matches_brief(campaign):
    c = campaign
    assert _matches(c, {"brief_id": HEAD_AI_LAB_PROJECT_ID})
    assert not _matches(c, {"brief_id": "9999999999"})


def test_manifest_market_key_matches_brief(campaign):
    c = campaign
    market_key = read_json(c.run_dir_1 / "run-manifest.json")["market_key"]
    assert _matches(c, {"market_key": market_key})
    assert not _matches(c, {"market_key": "some_other_role__nowhere__unknown"})
    assert not _matches(c, {})


def test_prior_run_with_no_db_anywhere_loads_on_manifest_identity(campaign):
    c = campaign
    # No snapshot DB, no live DB, no sibling DB, no report: only the manifest
    # says this run belongs to the brief, and that is enough to load it.
    _delete_db(c.state_dir)
    _delete_db(c.run_dir_2)

    batches = _collect(c)
    batch = _batch_for(batches, c.run_dir_1)

    assert batch is not None
    assert batch.runtime_summary == {}
    assert batches.unloaded_runs == []


# -- (d) read-only opens ------------------------------------------------------


def test_runtime_summary_opens_sqlite_read_only(campaign, monkeypatch):
    c = campaign
    calls: list[tuple[tuple, dict]] = []
    real_connect = sqlite3.connect

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(engine_mod.sqlite3, "connect", _spy)

    summary = engine_mod._load_runtime_summary(
        c.run_dir_2 / "runtime_state.sqlite3", run_id=c.run_2
    )

    assert summary["brief_ids"] == [HEAD_AI_LAB_PROJECT_ID]
    assert calls
    for args, kwargs in calls:
        assert kwargs.get("uri") is True
        assert "mode=ro" in str(args[0])


def test_runtime_summary_of_empty_db_file_is_empty(tmp_path):
    db_path = tmp_path / "runtime_state.sqlite3"
    db_path.write_bytes(b"")

    assert engine_mod._load_runtime_summary(db_path, run_id=1) == {}
    assert db_path.read_bytes() == b""
