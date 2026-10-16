"""Safety and idempotency tests for the LinkedIn contamination reset tool."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from shared.runtime_state.lock import RuntimeStateLock
from shared.runtime_state.store import LINKEDIN_STRING_KIND, RuntimeStateStore
from shared.schemas import SearchString
from tools import reset_contaminated_linkedin_run as reset_tool


BRIEF_ID = "brief-reset-tool"


def _seed_state(tmp_path: Path) -> dict:
    output_root = tmp_path / "output"
    state_dir = output_root / "state" / "linkedin" / "reset-fixture"
    state_dir.mkdir(parents=True)
    (state_dir / "runtime_state.lock").write_bytes(b"")
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id=BRIEF_ID,
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": BRIEF_ID},
    )
    string = SearchString(
        id=1,
        name="reset string",
        boolean="python",
        family_key="builders",
        lane_id="lane-builders",
    )
    for status in ("in_progress", "done"):
        work_unit_id = store.upsert_work_unit(
            run_id=run_id,
            source="linkedin",
            brief_id=BRIEF_ID,
            kind=LINKEDIN_STRING_KIND,
            source_unit_id="1",
            display_name=string.name,
            ordering_index=1,
            status=status,
            payload=string.to_dict(),
            checkpoint={"pages_reviewed": 1},
            metrics={"pages_reviewed": 1},
            family_key=string.family_key,
            counters={"result_count": 15, "rejected_count": 1},
        )
    identity_key = "https://linkedin.com/in/reset-reject"
    candidate_id = store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        display_name="Reset Reject",
        profile_url=identity_key,
    )
    store.mark_candidate_terminal_runtime(
        run_id=run_id,
        source="linkedin",
        brief_id=BRIEF_ID,
        identity_key=identity_key,
        decision="REJECT",
        payload={"full_decision": {"decision": "REJECT"}},
        last_work_unit_id=work_unit_id,
    )
    store.finish_run(run_id, "interrupted")

    for filename in (
        "progress.json",
        f"candidate_history-{BRIEF_ID}.jsonl",
        f"search_memory-{BRIEF_ID}.json",
        "snippets.jsonl",
        "facial_judgments.jsonl",
        "profile_summaries.jsonl",
        "final_judgments.jsonl",
    ):
        (state_dir / filename).write_text(f"pre-reset:{filename}\n")

    immutable_dir = output_root / "runs" / "linkedin" / BRIEF_ID / "run-1"
    immutable_dir.mkdir(parents=True)
    (immutable_dir / "snapshot.json").write_text('{"immutable":true}\n')

    manifest_payload = {
        "brief_id": BRIEF_ID,
        "reviewed": True,
        "executions": [
            {
                "run_id": run_id,
                "source_unit_id": "1",
                "classification": "contaminated",
                "reason": "facet lifecycle leakage",
            }
        ],
    }
    manifest_path = tmp_path / "reviewed-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    )
    loaded_manifest = reset_tool.load_reviewed_manifest(manifest_path)
    confirmation = reset_tool.expected_confirmation(
        brief_id=BRIEF_ID,
        manifest_sha256=loaded_manifest.sha256,
        executed_count=1,
    )
    return {
        "state_dir": state_dir,
        "db_path": state_dir / "runtime_state.sqlite3",
        "manifest_path": manifest_path,
        "manifest_payload": manifest_payload,
        "manifest": loaded_manifest,
        "confirmation": confirmation,
        "run_id": run_id,
        "work_unit_id": work_unit_id,
        "candidate_id": candidate_id,
        "output_runs": output_root / "runs",
    }


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _main_args(fixture: dict, *extra: str) -> list[str]:
    return [
        "--state-dir",
        str(fixture["state_dir"]),
        "--manifest",
        str(fixture["manifest_path"]),
        *extra,
    ]


def test_dry_run_and_wrong_confirmation_mutate_nothing(
    tmp_path: Path,
    capsys,
) -> None:
    fixture = _seed_state(tmp_path)
    before = _tree_hashes(fixture["state_dir"])
    output_runs_before = _tree_hashes(fixture["output_runs"])

    with patch.object(
        reset_tool,
        "RuntimeStateStore",
        side_effect=AssertionError("dry-run constructed a writable store"),
    ):
        assert reset_tool.main(_main_args(fixture)) == 0
    dry_run_output = capsys.readouterr().out
    assert "[reset] DRY-RUN" in dry_run_output
    assert f"confirmation={fixture['confirmation']}" in dry_run_output
    assert _tree_hashes(fixture["state_dir"]) == before

    assert (
        reset_tool.main(
            _main_args(fixture, "--apply", "--confirm", "wrong-confirmation")
        )
        == 2
    )
    assert "confirmation mismatch" in capsys.readouterr().err
    assert _tree_hashes(fixture["state_dir"]) == before
    assert _tree_hashes(fixture["output_runs"]) == output_runs_before


def test_apply_requires_enabled_flag_and_free_nonrunning_state(
    tmp_path: Path,
    capsys,
) -> None:
    fixture = _seed_state(tmp_path)
    apply_args = _main_args(
        fixture,
        "--apply",
        "--confirm",
        fixture["confirmation"],
    )
    before = _tree_hashes(fixture["state_dir"])

    assert reset_tool.main(apply_args) == 2
    assert "must be true before apply" in capsys.readouterr().err
    assert _tree_hashes(fixture["state_dir"]) == before

    with (
        patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True),
        RuntimeStateLock(fixture["state_dir"]),
    ):
        assert reset_tool.main(apply_args) == 2
    assert "already locked" in capsys.readouterr().err
    assert _tree_hashes(fixture["state_dir"]) == before

    with closing(sqlite3.connect(fixture["db_path"])) as conn:
        conn.execute(
            "UPDATE runs SET status = 'running', ended_at = NULL WHERE id = ?",
            (fixture["run_id"],),
        )
        conn.commit()
    running_before = _tree_hashes(fixture["state_dir"])
    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        assert reset_tool.main(apply_args) == 2
    assert "is still running" in capsys.readouterr().err
    assert _tree_hashes(fixture["state_dir"]) == running_before


def test_apply_backs_up_rebuilds_receipts_and_is_idempotent(
    tmp_path: Path,
    capsys,
) -> None:
    fixture = _seed_state(tmp_path)
    output_runs_before = _tree_hashes(fixture["output_runs"])
    apply_args = _main_args(
        fixture,
        "--apply",
        "--confirm",
        fixture["confirmation"],
    )

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        assert reset_tool.main(apply_args) == 0
    assert "[reset] APPLIED" in capsys.readouterr().out

    reset_id = fixture["manifest"].sha256[:20]
    backup_path = (
        fixture["state_dir"]
        / "backups"
        / f"facet-contamination-reset-{reset_id}.sqlite3"
    )
    receipt_path = (
        fixture["state_dir"] / f"facet-contamination-reset-{reset_id}.json"
    )
    assert backup_path.is_file()
    assert receipt_path.is_file()
    with closing(sqlite3.connect(f"{backup_path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        backup_candidate = conn.execute(
            "SELECT terminal_decision FROM candidates WHERE id = ?",
            (fixture["candidate_id"],),
        ).fetchone()
        assert backup_candidate[0] == "REJECT"
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM events WHERE event_type = 'linkedin_execution_invalidated'"
            ).fetchone()[0]
            == 0
        )

    with closing(sqlite3.connect(fixture["db_path"])) as conn:
        candidate = conn.execute(
            """
            SELECT current_lifecycle_state, terminal_decision, terminal_payload_json
            FROM candidates
            WHERE id = ?
            """,
            (fixture["candidate_id"],),
        ).fetchone()
        assert candidate == ("failed_retryable", None, "{}")
        assert (
            conn.execute(
                """
                SELECT COUNT(*)
                FROM events
                WHERE event_type = 'linkedin_execution_invalidated'
                  AND work_unit_id = ?
                """,
                (fixture["work_unit_id"],),
            ).fetchone()[0]
            == 1
        )

    receipt = json.loads(receipt_path.read_text())
    assert receipt["version"] == reset_tool.RECEIPT_VERSION
    assert receipt["status"] == "completed"
    assert receipt["brief_id"] == BRIEF_ID
    assert receipt["manifest_sha256"] == fixture["manifest"].sha256
    assert receipt["manifest"] == fixture["manifest_payload"]
    assert receipt["executed_count"] == 1
    assert receipt["actions"]["counts"] == {
        "attempts_to_abandon": 0,
        "candidates_to_release": 1,
        "executions": 1,
        "executions_to_invalidate": 1,
    }
    assert receipt["before_hashes"]["sqlite_snapshot_sha256"] == receipt["backup"][
        "sha256"
    ]
    assert receipt["after_hashes"]["sqlite_snapshot_sha256"] != receipt[
        "before_hashes"
    ]["sqlite_snapshot_sha256"]
    assert json.loads(
        (fixture["state_dir"] / f"search_memory-{BRIEF_ID}.json").read_text()
    ) == {}
    assert all(
        value is not None
        for value in receipt["after_hashes"]["compatibility_projections"].values()
    )
    for filename, digest in receipt["after_hashes"][
        "compatibility_projections"
    ].items():
        assert hashlib.sha256(
            (fixture["state_dir"] / filename).read_bytes()
        ).hexdigest() == digest
    assert _tree_hashes(fixture["output_runs"]) == output_runs_before

    state_after_first_apply = _tree_hashes(fixture["state_dir"])
    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        assert reset_tool.main(apply_args) == 0
    assert "[reset] NO-OP" in capsys.readouterr().out
    assert _tree_hashes(fixture["state_dir"]) == state_after_first_apply
    assert _tree_hashes(fixture["output_runs"]) == output_runs_before


def test_manifest_with_unresolved_ambiguity_is_rejected(tmp_path: Path) -> None:
    fixture = _seed_state(tmp_path)
    payload = json.loads(fixture["manifest_path"].read_text())
    payload["executions"][0]["classification"] = "ambiguous"
    fixture["manifest_path"].write_text(json.dumps(payload))

    before = _tree_hashes(fixture["state_dir"])
    assert reset_tool.main(_main_args(fixture)) == 2
    assert _tree_hashes(fixture["state_dir"]) == before


def test_manifest_must_be_reviewed_and_explicitly_contaminated(
    tmp_path: Path,
    capsys,
) -> None:
    fixture = _seed_state(tmp_path)
    payload = json.loads(fixture["manifest_path"].read_text())
    payload["reviewed"] = False
    fixture["manifest_path"].write_text(json.dumps(payload))
    before = _tree_hashes(fixture["state_dir"])

    assert reset_tool.main(_main_args(fixture)) == 2
    assert "reviewed: true" in capsys.readouterr().err
    assert _tree_hashes(fixture["state_dir"]) == before

    payload["reviewed"] = True
    payload["executions"][0]["classification"] = "clean"
    fixture["manifest_path"].write_text(json.dumps(payload))
    assert reset_tool.main(_main_args(fixture)) == 2
    assert "classified contaminated" in capsys.readouterr().err
    assert _tree_hashes(fixture["state_dir"]) == before


def test_retry_after_projection_failure_receipts_original_actions(
    tmp_path: Path,
    capsys,
) -> None:
    fixture = _seed_state(tmp_path)
    progress_path = fixture["state_dir"] / "progress.json"
    original_progress_hash = hashlib.sha256(progress_path.read_bytes()).hexdigest()
    apply_args = _main_args(
        fixture,
        "--apply",
        "--confirm",
        fixture["confirmation"],
    )

    def partial_projection_rebuild(*args, **kwargs) -> None:
        progress_path.write_text("partial-post-reset-projection\n")
        raise RuntimeError("injected projection failure")

    with (
        patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True),
        patch.object(
            reset_tool,
            "rebuild_compat_projections",
            side_effect=partial_projection_rebuild,
        ),
    ):
        assert reset_tool.main(apply_args) == 2
    assert "injected projection failure" in capsys.readouterr().err
    assert progress_path.read_text() == "partial-post-reset-projection\n"

    reset_id = fixture["manifest"].sha256[:20]
    receipt_path = (
        fixture["state_dir"] / f"facet-contamination-reset-{reset_id}.json"
    )
    assert not receipt_path.exists()
    with closing(sqlite3.connect(fixture["db_path"])) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM events WHERE event_type = 'linkedin_execution_invalidated'"
            ).fetchone()[0]
            == 1
        )

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        assert reset_tool.main(apply_args) == 0
    receipt = json.loads(receipt_path.read_text())
    assert receipt["before_hashes"]["compatibility_projections"][
        "progress.json"
    ] == original_progress_hash
    assert receipt["actions"]["counts"] == {
        "attempts_to_abandon": 0,
        "candidates_to_release": 1,
        "executions": 1,
        "executions_to_invalidate": 1,
    }


def test_retry_refuses_canonical_drift_before_another_write(
    tmp_path: Path,
    capsys,
) -> None:
    fixture = _seed_state(tmp_path)
    apply_args = _main_args(
        fixture,
        "--apply",
        "--confirm",
        fixture["confirmation"],
    )
    with (
        patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True),
        patch.object(
            reset_tool,
            "apply_linkedin_contamination_reset",
            side_effect=RuntimeError("injected before reset transaction"),
        ),
    ):
        assert reset_tool.main(apply_args) == 2
    assert "injected before reset transaction" in capsys.readouterr().err

    with closing(sqlite3.connect(fixture["db_path"])) as conn:
        conn.execute(
            "UPDATE work_units SET notes = 'drifted' WHERE id = ?",
            (fixture["work_unit_id"],),
        )
        conn.commit()
    before_retry = _tree_hashes(fixture["state_dir"])

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        assert reset_tool.main(apply_args) == 2
    assert "changed after the pre-reset backup" in capsys.readouterr().err
    assert _tree_hashes(fixture["state_dir"]) == before_retry
    with closing(sqlite3.connect(fixture["db_path"])) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM events WHERE event_type = 'linkedin_execution_invalidated'"
            ).fetchone()[0]
            == 0
        )


def test_retry_refuses_drift_after_reset_transaction(
    tmp_path: Path,
    capsys,
) -> None:
    fixture = _seed_state(tmp_path)
    apply_args = _main_args(
        fixture,
        "--apply",
        "--confirm",
        fixture["confirmation"],
    )
    with (
        patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True),
        patch.object(
            reset_tool,
            "rebuild_compat_projections",
            side_effect=RuntimeError("injected after reset transaction"),
        ),
    ):
        assert reset_tool.main(apply_args) == 2
    assert "injected after reset transaction" in capsys.readouterr().err

    with closing(sqlite3.connect(fixture["db_path"])) as conn:
        conn.execute(
            "UPDATE work_units SET notes = 'post-reset drift' WHERE id = ?",
            (fixture["work_unit_id"],),
        )
        conn.commit()
    before_retry = _tree_hashes(fixture["state_dir"])

    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        assert reset_tool.main(apply_args) == 2
    assert "work_units changed outside the reset plan" in capsys.readouterr().err
    assert _tree_hashes(fixture["state_dir"]) == before_retry
    reset_id = fixture["manifest"].sha256[:20]
    assert not (
        fixture["state_dir"] / f"facet-contamination-reset-{reset_id}.json"
    ).exists()


def test_failed_integrity_check_refuses_without_creating_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    fixture = _seed_state(tmp_path)
    fixture["db_path"].write_bytes(b"not a sqlite database")
    before = _tree_hashes(fixture["state_dir"])

    assert reset_tool.main(_main_args(fixture)) == 2
    assert "database" in capsys.readouterr().err.lower()
    assert _tree_hashes(fixture["state_dir"]) == before
    assert not (fixture["state_dir"] / "backups").exists()


def test_apply_leaves_store_startable_and_blocks_dirty_resume(
    tmp_path: Path, capsys
) -> None:
    """F5 end-to-end through the tool: apply writes the root AND completion
    events (in that order, completion last), so a fresh flag-on run can start
    while any resume/clone from the invalidated run refuses."""
    import pytest

    from shared.runtime_state.execution_validity import (
        ContaminationRunStartRefused,
    )

    fixture = _seed_state(tmp_path)
    apply_args = _main_args(
        fixture, "--apply", "--confirm", fixture["confirmation"]
    )
    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        assert reset_tool.main(apply_args) == 0

        with closing(sqlite3.connect(fixture["db_path"])) as conn:
            event_ids = {
                row[1]: row[0]
                for row in conn.execute(
                    """
                    SELECT id, event_type
                    FROM events
                    WHERE event_type IN (
                        'linkedin_contamination_reset',
                        'linkedin_contamination_reset_completed'
                    )
                    """
                ).fetchall()
            }
        assert set(event_ids) == {
            "linkedin_contamination_reset",
            "linkedin_contamination_reset_completed",
        }
        # Completion is the LAST canonical write of the apply.
        assert (
            event_ids["linkedin_contamination_reset_completed"]
            > event_ids["linkedin_contamination_reset"]
        )

        store = RuntimeStateStore(fixture["db_path"])
        fresh_run_id = store.start_run(
            source="linkedin",
            brief_id=BRIEF_ID,
            output_dir=str(fixture["state_dir"]),
            mode="fresh",
        )
        assert fresh_run_id

        with pytest.raises(ContaminationRunStartRefused) as exc_info:
            store.start_run(
                source="linkedin",
                brief_id=BRIEF_ID,
                output_dir=str(fixture["state_dir"]),
                mode="resume",
                resumed_from_run_id=fixture["run_id"],
            )
        assert exc_info.value.reason == "invalidated_ancestry"

    # Flag off after a completed reset: the repaired campaign refuses to run.
    with pytest.raises(ContaminationRunStartRefused) as flag_info:
        store.start_run(
            source="linkedin",
            brief_id=BRIEF_ID,
            output_dir=str(fixture["state_dir"]),
            mode="fresh",
        )
    assert flag_info.value.reason == "reset_flag_required"


def test_apply_scrubs_false_lesson_sidecars_and_appends_run_log_boundary(
    tmp_path: Path, capsys
) -> None:
    """F6: bias, noise, live market-intel, and the execution plan carry false
    lessons with no provenance — apply must back them up under backups/ and
    reset them, then append ONE contamination_reset_boundary row to
    run_log.jsonl so boundary-aware readers ignore pre-reset events."""
    fixture = _seed_state(tmp_path)
    state_dir = fixture["state_dir"]
    bias_path = state_dir / f"bias_monitor-{BRIEF_ID}.json"
    bias_path.write_text('{"poisoned": true}')
    noise_path = state_dir / f"noise_discoveries-{BRIEF_ID}.jsonl"
    noise_path.write_text('{"term": "false-lesson"}\n')
    market_dir = state_dir / "market_intel"
    market_dir.mkdir()
    (market_dir / "block-1.json").write_text('{"contaminated": true}')
    plan_path = state_dir / "execution_plan.json"
    plan_path.write_text('{"queue": ["poisoned"]}')

    apply_args = _main_args(
        fixture, "--apply", "--confirm", fixture["confirmation"]
    )
    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        assert reset_tool.main(apply_args) == 0

    reset_id = fixture["manifest"].sha256[:20]
    sidecar_backup_dir = (
        state_dir / "backups" / f"facet-contamination-reset-{reset_id}-sidecars"
    )
    # Originals reset; byte-identical copies live under the reset's backup.
    assert not bias_path.exists()
    assert not noise_path.exists()
    assert not market_dir.exists()
    assert not plan_path.exists()
    assert (sidecar_backup_dir / bias_path.name).read_text() == '{"poisoned": true}'
    assert (
        sidecar_backup_dir / noise_path.name
    ).read_text() == '{"term": "false-lesson"}\n'
    assert (
        sidecar_backup_dir / "market_intel" / "block-1.json"
    ).read_text() == '{"contaminated": true}'
    assert (
        sidecar_backup_dir / "execution_plan.json"
    ).read_text() == '{"queue": ["poisoned"]}'

    run_log_path = state_dir / "run_log.jsonl"
    rows = [
        json.loads(line)
        for line in run_log_path.read_text().splitlines()
        if line.strip()
    ]
    boundary_rows = [
        row for row in rows if row.get("event") == "contamination_reset_boundary"
    ]
    assert len(boundary_rows) == 1
    assert boundary_rows[0]["reset_id"] == reset_id
    assert sorted(boundary_rows[0]["sidecars"]) == [
        f"bias_monitor-{BRIEF_ID}.json",
        "execution_plan.json",
        "market_intel",
        f"noise_discoveries-{BRIEF_ID}.jsonl",
    ]

    receipt = json.loads(
        (state_dir / f"facet-contamination-reset-{reset_id}.json").read_text()
    )
    assert receipt["sidecars"] == {
        f"bias_monitor-{BRIEF_ID}.json": "reset",
        f"noise_discoveries-{BRIEF_ID}.jsonl": "reset",
        "execution_plan.json": "reset",
        "market_intel": "reset",
    }

    # Idempotent rerun: no second boundary row, sidecars stay reset.
    with patch("shared.config.LINKEDIN_CONTAMINATION_RESET_ENABLED", True):
        assert reset_tool.main(apply_args) == 0
    rows_after = [
        json.loads(line)
        for line in run_log_path.read_text().splitlines()
        if line.strip()
    ]
    assert (
        len(
            [
                row
                for row in rows_after
                if row.get("event") == "contamination_reset_boundary"
            ]
        )
        == 1
    )
    assert not bias_path.exists()
