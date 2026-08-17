"""Tests for the Phase 1C brief-vs-state-dir taxonomy.

The aggregator now classifies every state directory it discovers into
exactly one of:
  - ``authored_brief`` — has a run and isn't archived
  - ``archived`` — ``runs.is_archived = 1``
  - ``orphaned_state_dir`` — no run row at all
  - ``intake_only`` — reserved for Phase 4 (intake-session FK lookup)

These tests pin the classifier semantics + the BriefCounts roll-up so
the masthead's "5 active briefs" sentence stays grounded as we add /
remove run statuses in future schema migrations.
"""

from __future__ import annotations

from pathlib import Path

import sqlite3

from cloris.control_plane import (
    _classify_entry,
    _compute_brief_counts,
    aggregate_status,
)
from cloris.models import BriefCounts, RunSummary, StateDirEntry
from shared.runtime_state.store import RuntimeStateStore


def _build_state_dir(tmp_path: Path, source: str, key: str) -> Path:
    state_dir = tmp_path / source / key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


# ---- _classify_entry pure function ---------------------------------------


class TestClassifyEntry:
    def test_no_run_classifies_orphaned(self) -> None:
        assert _classify_entry(latest_run=None, is_archived=False) == "orphaned_state_dir"

    def test_run_with_id_none_classifies_orphaned(self) -> None:
        # Defensive case: latest_run object exists but its id is None.
        empty_run = RunSummary()
        assert (
            _classify_entry(latest_run=empty_run, is_archived=False)
            == "orphaned_state_dir"
        )

    def test_archived_run_classifies_archived(self) -> None:
        run = RunSummary(id=1, status="completed")
        assert _classify_entry(latest_run=run, is_archived=True) == "archived"

    def test_archived_with_no_run_still_archived(self) -> None:
        # is_archived wins even without a run; this is a corner case
        # where the user archived a brief whose canonical run row was
        # later purged.
        assert _classify_entry(latest_run=None, is_archived=True) == "archived"

    def test_run_present_classifies_authored(self) -> None:
        run = RunSummary(id=42, status="completed")
        assert (
            _classify_entry(latest_run=run, is_archived=False) == "authored_brief"
        )

    def test_running_classifies_authored(self) -> None:
        run = RunSummary(id=1, status="running")
        assert (
            _classify_entry(latest_run=run, is_archived=False) == "authored_brief"
        )


# ---- _compute_brief_counts roll-up ----------------------------------------


def _entry(
    *,
    kind: str,
    status: str | None = None,
    source: str = "linkedin",
    resumable: bool | None = None,
) -> StateDirEntry:
    latest = RunSummary(id=1, status=status) if status else None
    return StateDirEntry(
        source=source,  # type: ignore[arg-type]
        state_key=f"{kind}-{status}",
        runtime_state_present=latest is not None,
        latest_run=latest,
        kind=kind,  # type: ignore[arg-type]
        resumable=resumable,
    )


class TestBriefCountsRollUp:
    def test_empty_returns_zeros(self) -> None:
        counts = _compute_brief_counts([])
        assert counts == BriefCounts()

    def test_orphans_counted_separately(self) -> None:
        counts = _compute_brief_counts([
            _entry(kind="orphaned_state_dir"),
            _entry(kind="orphaned_state_dir"),
        ])
        assert counts.orphaned == 2
        assert counts.active == 0

    def test_archived_counted_separately(self) -> None:
        counts = _compute_brief_counts([
            _entry(kind="archived", status="completed"),
        ])
        assert counts.archived == 1
        assert counts.finished == 0  # archived NOT double-counted as finished

    def test_running_counts_as_working_and_active(self) -> None:
        counts = _compute_brief_counts([
            _entry(kind="authored_brief", status="running"),
        ])
        assert counts.working == 1
        assert counts.active == 1
        assert counts.paused == 0

    def test_paused_status_counts_as_paused_and_active_when_resumable(self) -> None:
        # P7.4: paused now REQUIRES resumable is True — this is the
        # "still has queued work" case.
        counts = _compute_brief_counts([
            _entry(kind="authored_brief", status="interrupted", resumable=True),
            _entry(
                kind="authored_brief",
                status="governor_limit_reached",
                resumable=True,
            ),
        ])
        assert counts.paused == 2
        assert counts.active == 2
        assert counts.working == 0

    def test_interrupted_not_resumable_counts_as_finished_not_paused(self) -> None:
        # P7.4 RED-FIRST case: a fully-completed interrupted run (queue
        # drained, resumable is False) must NOT render as paused. This is
        # the standing "98% of paused runs are actually finished" bug.
        counts = _compute_brief_counts([
            _entry(kind="authored_brief", status="interrupted", resumable=False),
            _entry(
                kind="authored_brief",
                status="governor_limit_reached",
                resumable=False,
            ),
        ])
        assert counts.paused == 0
        assert counts.finished == 2
        assert counts.active == 0

    def test_interrupted_unknown_resumability_counts_as_lost_not_paused(
        self,
    ) -> None:
        # resumable is None (e.g. non-LinkedIn source, or missing/malformed
        # progress.json) — no positive evidence of completion, so this
        # fails closed into "lost" rather than the false-promise "paused"
        # bucket or an unproven "finished" claim.
        counts = _compute_brief_counts([
            _entry(
                kind="authored_brief",
                status="interrupted",
                source="github",
                resumable=None,
            ),
            _entry(
                kind="authored_brief",
                status="governor_limit_reached",
                source="github",
                resumable=None,
            ),
        ])
        assert counts.paused == 0
        assert counts.finished == 0
        assert counts.lost == 2
        assert counts.active == 0

    def test_completed_counts_as_finished_only(self) -> None:
        counts = _compute_brief_counts([
            _entry(kind="authored_brief", status="completed"),
            _entry(kind="authored_brief", status="succeeded"),
        ])
        assert counts.finished == 2
        assert counts.active == 0

    def test_abandoned_error_and_failed_count_as_lost(self) -> None:
        counts = _compute_brief_counts([
            _entry(kind="authored_brief", status="abandoned"),
            _entry(kind="authored_brief", status="error"),
            _entry(kind="authored_brief", status="failed"),
        ])
        assert counts.lost == 3
        assert counts.active == 0

    def test_phantom_paused_literal_has_no_producer(self) -> None:
        # The literal status "paused" was never emitted by any writer;
        # P7.4 deletes it from the bucketing tuple entirely. If a
        # "paused" status string ever showed up on an entry it must NOT
        # be treated as a magic passthrough into the paused bucket — it
        # falls into the resumable-gated unknown-status branch like any
        # other unrecognized status.
        counts = _compute_brief_counts([
            _entry(kind="authored_brief", status="paused", resumable=None),
        ])
        assert counts.paused == 0
        assert counts.lost == 1

    def test_intake_only_buckets_with_orphans(self) -> None:
        # v0: intake_only collapses to orphan bucket pending Phase 4 wiring.
        counts = _compute_brief_counts([_entry(kind="intake_only")])
        assert counts.orphaned == 1

    def test_mixed_population(self) -> None:
        counts = _compute_brief_counts([
            _entry(kind="authored_brief", status="running"),
            _entry(kind="authored_brief", status="interrupted", resumable=True),
            _entry(
                kind="authored_brief",
                status="governor_limit_reached",
                resumable=False,
            ),
            _entry(kind="authored_brief", status="completed"),
            _entry(kind="authored_brief", status="abandoned"),
            _entry(kind="authored_brief", status="failed"),
            _entry(kind="archived"),
            _entry(kind="orphaned_state_dir"),
            _entry(kind="orphaned_state_dir"),
        ])
        # 1 running + 1 paused (resumable=True) = 2 active, of which 1 is
        # working. The governor_limit_reached/resumable=False entry rolls
        # into finished instead of paused (P7.4).
        assert counts.active == 2
        assert counts.working == 1
        assert counts.paused == 1
        assert counts.finished == 2
        assert counts.lost == 2
        assert counts.archived == 1
        assert counts.orphaned == 2


# ---- aggregator end-to-end -----------------------------------------------


class TestAggregateStatusEndToEnd:
    def test_empty_state_root_has_zero_counts(self, tmp_path: Path) -> None:
        response = aggregate_status(tmp_path)
        assert response.entries == []
        assert response.counts == BriefCounts()

    def test_orphaned_state_dir_classified(self, tmp_path: Path) -> None:
        _build_state_dir(tmp_path, "linkedin", "no-run")
        response = aggregate_status(tmp_path)
        assert len(response.entries) == 1
        assert response.entries[0].kind == "orphaned_state_dir"
        assert response.counts.orphaned == 1

    def test_authored_brief_classified(self, tmp_path: Path) -> None:
        sd = _build_state_dir(tmp_path, "linkedin", "real")
        store = RuntimeStateStore(sd / "runtime_state.sqlite3")
        run_id = store.start_run(
            source="linkedin",
            brief_id="brief-1",
            output_dir=str(sd),
            mode="fresh",
        )
        store.finish_run(run_id, "completed")
        response = aggregate_status(tmp_path)
        entry = response.entries[0]
        assert entry.kind == "authored_brief"
        assert response.counts.finished == 1

    def test_fully_completed_interrupted_run_renders_as_finished_not_paused(
        self, tmp_path: Path
    ) -> None:
        # P7.4 RED-FIRST: end-to-end through aggregate_status (not just the
        # pure roll-up). progress.json with an empty `strings` list is
        # exactly what shared.runtime_state.read_models.has_pending_work
        # treats as "queue drained" -> resumable=False. Before the fix
        # this rendered as paused (bug); after, it renders as finished.
        sd = _build_state_dir(tmp_path, "linkedin", "drained")
        store = RuntimeStateStore(sd / "runtime_state.sqlite3")
        run_id = store.start_run(
            source="linkedin",
            brief_id="brief-drained",
            output_dir=str(sd),
            mode="fresh",
        )
        store.finish_run(run_id, "interrupted", stop_reason="user_stop")
        (sd / "progress.json").write_text('{"strings": []}')

        response = aggregate_status(tmp_path)
        entry = response.entries[0]
        assert entry.resumable is False
        assert response.counts.paused == 0
        assert response.counts.finished == 1
        assert response.counts.lost == 0

    def test_interrupted_run_with_pending_work_stays_paused(
        self, tmp_path: Path
    ) -> None:
        # Control case: a genuinely resumable interrupted run (queue still
        # has queued work) must still bucket to paused.
        sd = _build_state_dir(tmp_path, "linkedin", "resumable")
        store = RuntimeStateStore(sd / "runtime_state.sqlite3")
        run_id = store.start_run(
            source="linkedin",
            brief_id="brief-resumable",
            output_dir=str(sd),
            mode="fresh",
        )
        store.upsert_work_unit(
            run_id=run_id,
            source="linkedin",
            brief_id="brief-resumable",
            kind="linkedin_string",
            source_unit_id="1",
            display_name="queued",
            ordering_index=0,
            status="queued",
        )
        store.finish_run(run_id, "interrupted", stop_reason="user_stop")
        (sd / "progress.json").write_text(
            '{"strings": [{"status": "queued"}]}'
        )

        response = aggregate_status(tmp_path)
        entry = response.entries[0]
        assert entry.resumable is True
        assert response.counts.paused == 1
        assert response.counts.finished == 0

    def test_archived_run_classified(self, tmp_path: Path) -> None:
        sd = _build_state_dir(tmp_path, "linkedin", "filed-away")
        store = RuntimeStateStore(sd / "runtime_state.sqlite3")
        run_id = store.start_run(
            source="linkedin",
            brief_id="brief-2",
            output_dir=str(sd),
            mode="fresh",
        )
        store.finish_run(run_id, "completed")
        # Mark archived directly via SQL; Phase 4 will add an API endpoint.
        with sqlite3.connect(sd / "runtime_state.sqlite3") as conn:
            conn.execute("UPDATE runs SET is_archived = 1 WHERE id = ?", (run_id,))
            conn.commit()
        response = aggregate_status(tmp_path)
        entry = response.entries[0]
        assert entry.kind == "archived"
        assert response.counts.archived == 1
        assert response.counts.finished == 0

    def test_mixed_population_counts_correctly(self, tmp_path: Path) -> None:
        # One orphan + one authored running + one archived completed.
        _build_state_dir(tmp_path, "linkedin", "orphan")

        sd_running = _build_state_dir(tmp_path, "linkedin", "running")
        store_r = RuntimeStateStore(sd_running / "runtime_state.sqlite3")
        store_r.start_run(
            source="linkedin",
            brief_id="brief-r",
            output_dir=str(sd_running),
            mode="fresh",
        )

        sd_arch = _build_state_dir(tmp_path, "linkedin", "arch")
        store_a = RuntimeStateStore(sd_arch / "runtime_state.sqlite3")
        run_id_a = store_a.start_run(
            source="linkedin",
            brief_id="brief-a",
            output_dir=str(sd_arch),
            mode="fresh",
        )
        store_a.finish_run(run_id_a, "completed")
        with sqlite3.connect(sd_arch / "runtime_state.sqlite3") as conn:
            conn.execute("UPDATE runs SET is_archived = 1 WHERE id = ?", (run_id_a,))
            conn.commit()

        response = aggregate_status(tmp_path)
        assert len(response.entries) == 3
        assert response.counts.orphaned == 1
        assert response.counts.working == 1
        assert response.counts.active == 1
        assert response.counts.archived == 1
