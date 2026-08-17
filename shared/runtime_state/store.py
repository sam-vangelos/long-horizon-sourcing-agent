"""SQLite-backed canonical runtime state store."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from shared.contracts import TARGET_CANDIDATE_LIFECYCLE
from shared.runtime_state.heartbeat import bump_heartbeat


CURRENT_SCHEMA_VERSION = "12"
_UNSET = object()


def _stop_reason_helpers() -> tuple[set[str], "Callable[[Any], str]"]:
    """Lazy-import the stop-reasons module to break a circular dependency.

    ``shared.safety.__init__`` eagerly imports ``coordinator``, which
    imports ``RuntimeStateStore`` from this module. A module-level
    ``from shared.safety.stop_reasons import ...`` here would deadlock
    the import graph. The helpers we need (the canonical enum set and
    the normalizer) are imported on first call instead.
    """

    from shared.safety.stop_reasons import (  # noqa: PLC0415 - lazy by design
        _KNOWN_STOP_REASONS,
        normalize_stop_reason,
    )
    return _KNOWN_STOP_REASONS, normalize_stop_reason

GITHUB_QUERY_KIND = "github_query"
GITHUB_GRAPH_SEED_KIND = "github_graph_seed"
LINKEDIN_STRING_KIND = "linkedin_string"
RESEARCHER_AUTHOR_QUERY_KIND = "researcher_author_query"
DESIGNER_BEHANCE_QUERY_KIND = "designer_behance_query"
DESIGNER_CSE_QUERY_KIND = "designer_cse_query"
# A.8 — Executive Search work-unit kind. The exec_search pipeline
# (Phase D.1) extends LinkedIn's full-eval branch with a
# DOSSIER_RATIONALE block; per-candidate work units carry this kind
# so the control plane's progress aggregation can distinguish
# exec_search dossier work from LinkedIn's plain-eval work even
# though both run inside the LinkedIn orchestrator process.
EXEC_SEARCH_QUERY_KIND = "exec_search_query"
SIDE_EFFECT_TERMINAL_STATUSES = {"succeeded", "failed", "skipped", "invalidated"}
# A failed side effect is normally capped at three executions. LinkedIn saves
# are the exception: their stable idempotency row is always retryable because
# the browser rechecks whether the profile is already saved before any click.
# ``interrupted`` marks a crash-interrupted ``pending`` row and retries without
# consuming an attempt.
SIDE_EFFECT_MAX_ATTEMPTS = 3
SIDE_EFFECT_RETRYABLE_STATUSES = {"failed", "interrupted", "invalidated"}

TERMINAL_WORK_UNIT_STATUSES = {"done", "skipped", "error"}
VALID_RUN_STATUSES = frozenset(
    {
        "completed",
        "interrupted",
        "error",
        "governor_limit_reached",
        "abandoned",
        "succeeded",
        "failed",
        "running",
    }
)
KNOWN_CANDIDATE_SOURCES = frozenset(
    {"linkedin", "github", "exec_search", "designer", "researcher"}
)
VALID_WORK_UNIT_STATUSES = frozenset(
    {"queued", "in_progress", "done", "skipped", "error"}
)
DEDUP_BLOCKING_RUNTIME_DECISIONS = {
    "LEGACY_TERMINAL",
    "GEO_FILTERED",
    "PRESCREEN_SKIP",
    "INSUFFICIENT_DATA",
}
DEDUP_BLOCKING_LINKEDIN_DECISIONS = {
    "FACIAL_NO",
    "FACIAL_SKIP",
    "SAVE",
    "REJECT",
    "INFERENTIAL_SAVE",
    "TRANSFERABLE_SAVE",
    "SIGNAL_SAVE",
    # Added 2026-07-28. Review-parked candidates were the one terminal class
    # that never suppressed, so every string that re-surfaced one re-spent a
    # facial (and often a full eval) re-deciding the same person under the
    # same brief — Jingran Zhou was REVIEW_FLAGGED at 12:55 and dispatched
    # into another facial at 13:18, which also detonated the lifecycle guard.
    # The hash-aware clause still RELEASES these on a brief change, which is
    # the only time re-judging a parked person can produce a different answer.
    "REVIEW_FLAGGED",
    "REVIEW_INFERRED",
}
DEDUP_BLOCKING_DECISIONS = (
    DEDUP_BLOCKING_RUNTIME_DECISIONS | DEDUP_BLOCKING_LINKEDIN_DECISIONS
)
# P3.1: SAVE-family verdicts always suppress regardless of brief revision —
# never re-save someone already in the pipeline. (Recruiter-made decisions
# are handled via user_status IS NOT NULL in the suppression clause.)
SAVE_FAMILY_DECISIONS = {
    "SAVE",
    "INFERENTIAL_SAVE",
    "TRANSFERABLE_SAVE",
    "SIGNAL_SAVE",
}

ALLOWED_LIFECYCLE_TRANSITIONS: dict[str, set[str]] = {
    "discovered": {"snippet_extracted", "failed_retryable", "failed_terminal"},
    # snippet_extracted may go directly to full_started for sources whose
    # discovery seeds carry enough identity signal to bypass facial triage
    # (Exec Search ships per-company dossier candidates from a vetted list).
    "snippet_extracted": {"facial_started", "full_started", "failed_retryable", "failed_terminal"},
    "facial_started": {"facial_terminal", "failed_retryable", "failed_terminal"},
    # `snippet_extracted` here completes P3.1 (see the failed_terminal note
    # below), which established that a candidate re-eligible after a BRIEF
    # REVISION may re-enter the pipeline — but applied it to failed_terminal
    # only. The judgment terminals kept absorbing, which put them in direct
    # contradiction with `_HASH_AWARE_SUPPRESSION_CLAUSE`: that clause
    # deliberately RELEASES a non-SAVE verdict made under a different
    # brief_content_hash so the new criteria can re-judge the person, and then
    # the first re-encounter raised `invalid lifecycle transition` and took the
    # whole session down with it. Measured 2026-07-27: a live run died six
    # minutes in, on string 1, page 1, card 2.
    #
    # Safe because release is already narrow. A SAVE-family verdict suppresses
    # regardless of hash (first clause), as does any recruiter-touched row
    # (`user_status IS NOT NULL`). The only rows that reach a re-snippet are
    # non-SAVE, non-recruiter-touched verdicts made under a brief that no longer
    # governs — exactly the population re-judging is for. Verified against the
    # live store: of the 12 released rows, zero were SAVE-family and zero
    # carried a user_status.
    # Re-entry is stage-complete, mirroring failed_terminal below: a released
    # candidate re-enters wherever the pipeline actually meets them. The
    # 2026-07-27 fix opened only snippet_extracted, and the very next session
    # died on the edge it left closed — batch facial dispatched facial_started
    # against full_terminal (run 7, 2026-07-28 13:18, one profile open spent).
    # The forward discipline is untouched: no path may still SKIP a started
    # stage on the way to a terminal one.
    "facial_terminal": {"snippet_extracted", "facial_started", "full_started", "failed_retryable", "failed_terminal"},
    "full_started": {"full_terminal", "failed_retryable", "failed_terminal"},
    "full_terminal": {"snippet_extracted", "facial_started", "full_started", "failed_retryable", "failed_terminal"},
    "failed_retryable": {"snippet_extracted", "facial_started", "full_started", "failed_retryable", "failed_terminal"},
    # P3.1: failed_terminal is no longer a permanent verdict — a candidate
    # re-eligible after a brief revision (or an explicit retry via
    # clear_candidate_terminal_state) may re-enter the pipeline. Extraction
    # failures must not suppress a person forever.
    "failed_terminal": {"snippet_extracted", "facial_started", "full_started", "failed_retryable"},
}


class RuntimeStateStore:
    """Authoritative runtime state for candidate lifecycle and resume semantics."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Phase 1.6: every canonical write checkpoint also bumps the
        # worker.json heartbeat. The state_dir is db_path.parent (the
        # sidecar contract pins worker.json at <state_dir>/worker.json).
        # ``bump_heartbeat`` is a no-op when the sidecar doesn't exist —
        # which covers tests and orchestrators run standalone outside
        # of Cloris's launch path. No signature change needed.
        self._state_dir: Path = self.db_path.parent
        self.initialize()

    def _bump_heartbeat(self) -> None:
        """Best-effort heartbeat refresh after a canonical write.

        Must never raise: the orchestrator's primary write path has
        already succeeded by the time this fires. A heartbeat write
        failure should never roll back useful work or surface as a
        run-killing exception.
        """

        bump_heartbeat(self._state_dir)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            # Slice A.3 hardening: explicit ``busy_timeout`` so concurrent
            # writers (e.g., 4-thread store construction during the
            # ``test_concurrent_migration_does_not_crash`` race, OR the
            # production case of two recruiter UI processes both running
            # ``_migrate`` on first launch) wait briefly for the SQLite
            # write lock instead of immediately raising
            # ``OperationalError("database is locked")``. 5000ms is the
            # pattern used elsewhere in the repo for read-only handles
            # (see ``shared/runtime_state/read_models._open_readonly``)
            # and is comfortably larger than the migration's worst-case
            # wall-clock time even after A.3 added 3 new tables + indexes.
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            _execute_with_lock_retry(conn, "PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def _write_connection(
        self,
        conn: sqlite3.Connection | None,
    ) -> Iterator[sqlite3.Connection]:
        """Reuse an outer transaction or own one for a standalone write."""

        if conn is not None:
            yield conn
            return
        with self.connect() as owned_conn:
            yield owned_conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    brief_id TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stop_reason TEXT NOT NULL DEFAULT 'normal',
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    resumed_from_run_id INTEGER,
                    resume_state_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(resumed_from_run_id) REFERENCES runs(id)
                );

                CREATE TABLE IF NOT EXISTS work_units (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    brief_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source_unit_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    ordering_index INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    checkpoint_json TEXT NOT NULL DEFAULT '{}',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    family_key TEXT NOT NULL DEFAULT '',
                    novelty_bucket TEXT NOT NULL DEFAULT '',
                    domain_lane TEXT NOT NULL DEFAULT '',
                    result_count INTEGER NOT NULL DEFAULT 0,
                    candidates_discovered INTEGER NOT NULL DEFAULT 0,
                    candidates_enriched INTEGER NOT NULL DEFAULT 0,
                    candidates_insufficient INTEGER NOT NULL DEFAULT 0,
                    facial_yes_count INTEGER NOT NULL DEFAULT 0,
                    facial_no_count INTEGER NOT NULL DEFAULT 0,
                    facial_borderline_count INTEGER NOT NULL DEFAULT 0,
                    saves_count INTEGER NOT NULL DEFAULT 0,
                    rejected_count INTEGER NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT '',
                    started_at TEXT,
                    ended_at TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
                    UNIQUE(run_id, kind, source_unit_id)
                );

                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    brief_id TEXT NOT NULL,
                    identity_key TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    profile_url TEXT NOT NULL DEFAULT '',
                    current_lifecycle_state TEXT NOT NULL,
                    terminal_decision TEXT,
                    terminal_payload_json TEXT NOT NULL DEFAULT '{}',
                    last_work_unit_id INTEGER,
                    last_attempt_id INTEGER,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    UNIQUE(brief_id, source, identity_key),
                    FOREIGN KEY(last_work_unit_id) REFERENCES work_units(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS candidate_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    candidate_id INTEGER NOT NULL,
                    work_unit_id INTEGER,
                    stage TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    batch_key TEXT,
                    status TEXT NOT NULL,
                    failure_kind TEXT,
                    failure_reason TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    source_cursor_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id) ON DELETE CASCADE,
                    FOREIGN KEY(work_unit_id) REFERENCES work_units(id) ON DELETE SET NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_open_attempt
                ON candidate_attempts(candidate_id)
                WHERE status = 'started';

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    work_unit_id INTEGER,
                    candidate_id INTEGER,
                    attempt_id INTEGER,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
                    FOREIGN KEY(work_unit_id) REFERENCES work_units(id) ON DELETE CASCADE,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id) ON DELETE CASCADE,
                    FOREIGN KEY(attempt_id) REFERENCES candidate_attempts(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS side_effects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    candidate_id INTEGER NOT NULL,
                    attempt_id INTEGER,
                    effect_type TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    attempt_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    invalidated_at TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id) ON DELETE CASCADE,
                    FOREIGN KEY(attempt_id) REFERENCES candidate_attempts(id) ON DELETE SET NULL,
                    UNIQUE(candidate_id, effect_type, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS recruiter_write_intentions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_kind TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    source_brief_id TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    dedup_key TEXT NOT NULL,
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(dedup_key)
                );

                CREATE INDEX IF NOT EXISTS idx_recruiter_intentions_incomplete
                    ON recruiter_write_intentions(created_at)
                    WHERE completed_at IS NULL;
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("target_candidate_lifecycle", json.dumps(TARGET_CANDIDATE_LIFECYCLE)),
            )
            self._migrate(conn)
            self._install_external_schemas(conn)
            # Pin the schema version AFTER _migrate runs so the
            # version-gated normalization steps inside _migrate fire on
            # legacy rows. Pinning before _migrate would make the
            # version-gate observe a too-new version and skip the
            # migration on existing data (Phase 1.5 idempotency
            # requirement: the rewrite must run exactly once on legacy
            # data and never on already-normalized data).
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("schema_version", CURRENT_SCHEMA_VERSION),
            )

    def _migrate(self, conn: sqlite3.Connection) -> None:
        _ensure_schema_migrations_table(conn)
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(work_units)").fetchall()
        }
        if "metrics_json" not in columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE work_units ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}'",
            )
            _record_schema_migration(conn, "work_units.metrics_json")
        if "facial_borderline_count" not in columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE work_units ADD COLUMN facial_borderline_count INTEGER NOT NULL DEFAULT 0",
            )
            _record_schema_migration(conn, "work_units.facial_borderline_count")
        run_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        if "stop_reason" not in run_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE runs ADD COLUMN stop_reason TEXT NOT NULL DEFAULT 'normal'",
            )
            _record_schema_migration(conn, "runs.stop_reason")
        # Phase 1.5: stop_reason_detail captures the original freeform
        # error string when the orchestrator's catchall path stuffed
        # something like "error: KeyError" into stop_reason. The
        # canonical enum value goes to stop_reason; the original (if
        # non-canonical) goes to stop_reason_detail.
        if "stop_reason_detail" not in run_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE runs ADD COLUMN stop_reason_detail TEXT",
            )
            _record_schema_migration(conn, "runs.stop_reason_detail")

        # Phase 3: brief identity pinning. brief_path_at_launch records
        # where the orchestrator loaded the brief from; brief_content_hash
        # is the canonical-JSON SHA-256 of the brief at run-start; and
        # brief_snapshot_json carries the full canonical JSON so Run
        # Review can show "what the brief said when this run executed"
        # even if the on-disk file has since changed. All three are
        # nullable / empty-default so legacy rows survive untouched.
        if "brief_path_at_launch" not in run_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE runs ADD COLUMN brief_path_at_launch TEXT",
            )
            _record_schema_migration(conn, "runs.brief_path_at_launch")
        if "brief_content_hash" not in run_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE runs ADD COLUMN brief_content_hash TEXT",
            )
            _record_schema_migration(conn, "runs.brief_content_hash")
        if "brief_snapshot_json" not in run_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE runs ADD COLUMN brief_snapshot_json TEXT NOT NULL DEFAULT '{}'",
            )
            _record_schema_migration(conn, "runs.brief_snapshot_json")

        # reopen Stage 2 (R5a-1): soft recruiter ownership on runs. Records
        # WHICH recruiter launched this run so the read-only taste
        # aggregator (R5a-4) can attribute adaptation decisions to a
        # recruiter without a per-brief join. Nullable, no default, NO
        # foreign key on purpose: recruiters live in a SEPARATE database
        # (the global recruiter store), so this is a soft cross-DB
        # reference, not an enforced FK. recruiter_id IS NULL means
        # "unknown recruiter" — every legacy row and every pre-R5a-3
        # bridge caller leaves it NULL, and the aggregator skips those
        # rows rather than mis-attributing them. Existence-gated like
        # stop_reason_detail / brief_path_at_launch above, so it adds the
        # column WITHOUT bumping CURRENT_SCHEMA_VERSION (the version-gate
        # is reserved for normalization rewrites, not additive columns).
        if "recruiter_id" not in run_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE runs ADD COLUMN recruiter_id INTEGER",
            )
            _record_schema_migration(conn, "runs.recruiter_id")

        # Schema v6: is_archived flag on runs. Drives Phase 1C brief-vs-
        # state-dir taxonomy: `is_archived=1` means the user has filed
        # this brief away (or the reconciler has auto-archived an old
        # orphan). Default 0 preserves legacy semantics — every existing
        # row stays surfaced as "live" until explicitly archived.
        if "is_archived" not in run_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE runs ADD COLUMN is_archived INTEGER NOT NULL DEFAULT 0",
            )
            _record_schema_migration(conn, "runs.is_archived")

        # Schema v7: intake_session_id FK on runs. Links a launched run
        # back to the brief-authoring session that produced it, so the
        # aggregator can distinguish "authored brief that has run history"
        # from "filesystem state-dir artifact left behind by a one-off
        # CLI launch." Nullable — legacy runs (pre-v7) and CLI-launched
        # runs both leave this NULL, in which case the aggregator treats
        # the run as "authored" by default for backwards compatibility.
        if "intake_session_id" not in run_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE runs ADD COLUMN intake_session_id INTEGER "
                "REFERENCES intake_sessions(id)",
            )
            _record_schema_migration(conn, "runs.intake_session_id")

        # Schema v8: recruiter-authored note log + status override on
        # candidates. Phase C, slice C3: notes are an append-only JSON
        # array of `{body, created_at}` so the candidate-detail page can
        # surface every note in reverse-chrono. user_status is a recruiter
        # override that wins over Cloris's terminal_decision when set
        # (NULL = "use Cloris's judgment"). Both are additive — legacy
        # rows survive with empty notes and NULL user_status.
        candidate_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(candidates)"
            ).fetchall()
        }
        if "notes" not in candidate_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE candidates ADD COLUMN notes TEXT NOT NULL "
                "DEFAULT '[]'",
            )
            _record_schema_migration(conn, "candidates.notes")
        if "user_status" not in candidate_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE candidates ADD COLUMN user_status TEXT",
            )
            _record_schema_migration(conn, "candidates.user_status")

        # Schema v9: closed-loop feedback substrate. Phase C-bis Slice
        # 0.5. ``judgment_accuracy`` is the recruiter's calibration
        # signal on Cloris's terminal decision — explicitly distinct
        # from ``user_status`` (a pipeline action). Keeping them
        # schema-distinct from day one means the future Next Run
        # Learning surface can read calibration signal without
        # conflating it with recruiter pipeline action. NULL = no
        # signal. ``judgment_accuracy_at`` mirrors the pattern set by
        # other timestamp pairs (set together, cleared together).
        if "judgment_accuracy" not in candidate_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE candidates ADD COLUMN judgment_accuracy TEXT",
            )
            _record_schema_migration(conn, "candidates.judgment_accuracy")
        if "judgment_accuracy_at" not in candidate_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE candidates ADD COLUMN judgment_accuracy_at TEXT",
            )
            _record_schema_migration(conn, "candidates.judgment_accuracy_at")

        # Hardening P3.1: brief-version-aware suppression. Terminal decisions
        # are stamped with the brief_content_hash of the run that made them;
        # cross-run suppression then honors the verdict only while the brief
        # is unchanged (SAVE-family and recruiter-made decisions always
        # suppress). NULL = legacy row, backfilled with the next run's hash
        # at start_run (preserves today's behavior until the brief revises).
        if "brief_content_hash" not in candidate_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE candidates ADD COLUMN brief_content_hash TEXT",
            )
            _record_schema_migration(conn, "candidates.brief_content_hash")

        # W1-S2: cross-hub person key for dedup joins. GitHub rows backfill
        # from identity_key; other sources stay NULL until their adoption wave.
        if "person_key" not in candidate_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE candidates ADD COLUMN person_key TEXT",
            )
            conn.execute(
                """
                UPDATE candidates
                SET person_key = 'gh:' || lower(identity_key)
                WHERE source = 'github' AND person_key IS NULL
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candidates_brief_person_key
                ON candidates(brief_id, person_key)
                """
            )

        # Hardening P1.1: side-effect retry semantics. ``attempt_count``
        # tracks how many executions this ledger row has consumed so a
        # ``failed`` row can retry (up to SIDE_EFFECT_MAX_ATTEMPTS) instead
        # of skipping forever. DEFAULT 1 backfills existing rows as "one
        # attempt consumed" — exactly today's semantics for rows that
        # already executed once.
        side_effect_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(side_effects)"
            ).fetchall()
        }
        if side_effect_columns and "attempt_count" not in side_effect_columns:
            _add_column_if_missing(
                conn,
                "ALTER TABLE side_effects ADD COLUMN attempt_count "
                "INTEGER NOT NULL DEFAULT 1",
            )
            _record_schema_migration(conn, "side_effects.attempt_count")

        # Phase 1.5: one-shot normalization of legacy rows whose
        # stop_reason was written before the canonical enum was enforced
        # at the write boundary. Gated on meta.schema_version<4 so a
        # mixed-version process race doesn't redo the migration after a
        # newer writer normalized and an older writer re-wrote a freeform
        # value (the older writer would have re-set schema_version to
        # "3"; on next initialize, the gate fires again and re-normalizes
        # — exactly the idempotency property critique A1 asked for).
        prior_version = _read_schema_version(conn)
        if _version_lt(prior_version, "4"):
            _normalize_legacy_stop_reasons(conn)
            _record_schema_migration(conn, "normalize_legacy_stop_reasons_v4")

    def _install_external_schemas(self, conn: sqlite3.Connection) -> None:
        """Install non-runtime tables into the shared SQLite database."""

        from cloris.intake_sessions import install_schema as install_intake_schema
        from shared.brief_corpus import install_schema as install_brief_corpus_schema
        from shared.runtime_state.reflection import (
            install_schema as install_reflection_schema,
        )
        from shared.save_destination.candidate_workspace import (
            install_schema as install_workspace_schema,
        )

        install_intake_schema(conn)
        install_reflection_schema(conn)
        install_workspace_schema(conn)
        # Schema v11: accepted-brief corpus tables for source-packet
        # intake exemplar retrieval. Kept in the shared SQLite store so
        # the brief-authoring loop can learn from previously accepted
        # briefs without creating another persistence plane.
        install_brief_corpus_schema(conn)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def start_run(
        self,
        *,
        source: str,
        brief_id: str,
        output_dir: str,
        mode: str,
        resume_state: dict | None = None,
        resumed_from_run_id: int | None = None,
        clone_work_units_from_run_id: int | None = None,
        brief_path_at_launch: str | None = None,
        brief_content_hash: str | None = None,
        brief_snapshot_json: str | None = None,
        recruiter_id: int | None = None,
    ) -> int:
        now = _utc_now()
        # Phase 3: brief identity pinning. All three are optional with
        # safe defaults so legacy callers continue to work; computing
        # them is the responsibility of the bridge layer (which has
        # access to brief_path) — see shared.brief_identity.
        #
        # reopen Stage 2 (R5a-2): ``recruiter_id`` records which recruiter
        # launched this run (soft cross-DB ref to the global recruiter
        # store). Optional with a None default so every legacy caller and
        # every test continues to work unchanged — None lands as NULL,
        # which the R5a-4 taste aggregator reads as "unknown recruiter"
        # and skips. The run-start bridges resolve it via
        # ``shared.recruiter_context.get_current_recruiter_id`` (R5a-3).
        snapshot_json = brief_snapshot_json if brief_snapshot_json is not None else "{}"
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs(source, brief_id, output_dir, mode, status, started_at,
                                 resumed_from_run_id, resume_state_json,
                                 brief_path_at_launch, brief_content_hash,
                                 brief_snapshot_json, recruiter_id)
                VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    brief_id,
                    str(output_dir),
                    mode,
                    now,
                    resumed_from_run_id,
                    _json_dumps(resume_state or {}),
                    brief_path_at_launch,
                    brief_content_hash,
                    snapshot_json,
                    recruiter_id,
                ),
            )
            run_id = int(cursor.lastrowid)
            if brief_content_hash:
                # P3.1 backfill: legacy terminal rows (no hash yet) adopt THIS
                # run's brief hash — byte-preserves today's suppression until
                # the brief next revises, at which point they become
                # re-eligible like any hash-stamped row.
                conn.execute(
                    """
                    UPDATE candidates
                    SET brief_content_hash = ?
                    WHERE source = ? AND brief_id = ?
                      AND brief_content_hash IS NULL
                      AND (
                        current_lifecycle_state = 'failed_terminal'
                        OR terminal_decision IS NOT NULL
                      )
                    """,
                    (brief_content_hash, source, brief_id),
                )
                # Log the candidates this brief revision re-opened (hash
                # mismatch, not SAVE-family, not recruiter-decided) so the
                # re-eligibility is observable per run.
                save_placeholders = ",".join("?" for _ in SAVE_FAMILY_DECISIONS)
                reopened = conn.execute(
                    f"""
                    SELECT id, identity_key, terminal_decision, brief_content_hash
                    FROM candidates
                    WHERE source = ? AND brief_id = ?
                      AND brief_content_hash IS NOT NULL
                      AND brief_content_hash != ?
                      AND user_status IS NULL
                      AND (
                        current_lifecycle_state = 'failed_terminal'
                        OR terminal_decision IS NOT NULL
                      )
                      AND (
                        terminal_decision IS NULL
                        OR terminal_decision NOT IN ({save_placeholders})
                      )
                    """,
                    (
                        source,
                        brief_id,
                        brief_content_hash,
                        *sorted(SAVE_FAMILY_DECISIONS),
                    ),
                ).fetchall()
                for row in reopened:
                    self._insert_event(
                        conn,
                        run_id=run_id,
                        candidate_id=int(row["id"]),
                        event_type="candidate_re_eligible",
                        payload={
                            "identity_key": row["identity_key"],
                            "terminal_decision": row["terminal_decision"],
                            "old_brief_content_hash": row["brief_content_hash"],
                            "new_brief_content_hash": brief_content_hash,
                        },
                    )
            if clone_work_units_from_run_id:
                rows = conn.execute(
                    """
                    SELECT source, brief_id, kind, source_unit_id, display_name, ordering_index, status,
                           payload_json, checkpoint_json, metrics_json, family_key, novelty_bucket, domain_lane,
                           result_count, candidates_discovered, candidates_enriched, candidates_insufficient,
                           facial_yes_count, facial_no_count, facial_borderline_count, saves_count, rejected_count, notes
                    FROM work_units
                    WHERE run_id = ?
                    ORDER BY ordering_index ASC, id ASC
                    """,
                    (clone_work_units_from_run_id,),
                ).fetchall()
                for row in rows:
                    conn.execute(
                        """
                        INSERT INTO work_units(
                            run_id, source, brief_id, kind, source_unit_id, display_name, ordering_index, status,
                            payload_json, checkpoint_json, metrics_json, family_key, novelty_bucket, domain_lane,
                            result_count, candidates_discovered, candidates_enriched, candidates_insufficient,
                            facial_yes_count, facial_no_count, facial_borderline_count, saves_count, rejected_count, notes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            row["source"],
                            row["brief_id"],
                            row["kind"],
                            row["source_unit_id"],
                            row["display_name"],
                            row["ordering_index"],
                            row["status"],
                            row["payload_json"],
                            row["checkpoint_json"],
                            row["metrics_json"],
                            row["family_key"],
                            row["novelty_bucket"],
                            row["domain_lane"],
                            row["result_count"],
                            row["candidates_discovered"],
                            row["candidates_enriched"],
                            row["candidates_insufficient"],
                            row["facial_yes_count"],
                            row["facial_no_count"],
                            row["facial_borderline_count"],
                            row["saves_count"],
                            row["rejected_count"],
                            row["notes"],
                        ),
                    )
            self._insert_event(conn, run_id=run_id, event_type="run_started", payload={"mode": mode})
        self._bump_heartbeat()
        return run_id

    def append_candidate_note(
        self,
        candidate_id: int,
        body: str,
        *,
        created_at: str | None = None,
    ) -> int:
        """Append a recruiter note to a candidate's notes log.

        Phase C, slice C3: notes are stored as a JSON array of objects
        ``[{body: str, created_at: str}, ...]`` on ``candidates.notes``,
        append-only. Returns the count of notes after the append. Raises
        ``ValueError`` when the candidate id is not present (the API
        layer maps that to HTTP 404).
        """

        if not body.strip():
            raise ValueError("note body must not be empty or whitespace-only")
        stamp = created_at or _utc_now()
        new_note = {"body": body.strip(), "created_at": stamp}
        with self.connect() as conn:
            row = conn.execute(
                "SELECT notes FROM candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"candidate {candidate_id} not found")
            try:
                existing = json.loads(row["notes"] or "[]")
            except (json.JSONDecodeError, TypeError):
                existing = []
            if not isinstance(existing, list):
                existing = []
            existing.append(new_note)
            conn.execute(
                "UPDATE candidates SET notes = ?, last_seen_at = ? WHERE id = ?",
                (json.dumps(existing), stamp, candidate_id),
            )
        self._bump_heartbeat()
        return len(existing)

    def set_candidate_user_status(
        self,
        candidate_id: int,
        user_status: str | None,
    ) -> None:
        """Set or clear the recruiter-overridden status on a candidate.

        Phase C, slice C3: ``user_status`` lives alongside
        ``terminal_decision`` and wins when displayed (R24 — the
        recruiter override is the primary signal once set). Setting to
        ``None`` clears the override and falls back to Cloris's judgment.
        Raises ``ValueError`` if the candidate id is not present.
        """

        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"candidate {candidate_id} not found")
            conn.execute(
                "UPDATE candidates SET user_status = ?, last_seen_at = ? WHERE id = ?",
                (user_status, _utc_now(), candidate_id),
            )
        self._bump_heartbeat()

    def set_candidate_judgment_accuracy(
        self,
        candidate_id: int,
        judgment_accuracy: str | None,
    ) -> None:
        """Set or clear the recruiter's judgment-accuracy signal.

        Phase C-bis Slice 0.5: closed-loop feedback substrate. Distinct
        from ``user_status`` (pipeline action) — this column captures
        whether Cloris's *judgment* was useful, wrong, or off-rubric.
        NULL = no signal. The timestamp column moves in lockstep
        (set when value is set, cleared when value is cleared) so the
        Next Run Learning surface can sort/filter by recency without
        a separate index.

        Allowed values (None to clear):
          - ``"useful"``        — judgment was correct + useful
          - ``"wrong"``         — judgment was wrong
          - ``"off_rubric"``    — judged on the wrong axis
          - ``"overstated_depth"``  — Cloris overstated depth
          - ``"understated_depth"`` — Cloris understated depth

        Raises ``ValueError`` for unknown candidate ids or unknown
        accuracy values.
        """

        allowed = {
            "useful",
            "wrong",
            "off_rubric",
            "overstated_depth",
            "understated_depth",
        }
        if judgment_accuracy is not None and judgment_accuracy not in allowed:
            raise ValueError(
                f"invalid judgment_accuracy: {judgment_accuracy!r}; "
                f"expected one of {sorted(allowed)} or None"
            )

        now = _utc_now()
        stamp = now if judgment_accuracy is not None else None
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"candidate {candidate_id} not found")
            conn.execute(
                "UPDATE candidates SET judgment_accuracy = ?, "
                "judgment_accuracy_at = ?, last_seen_at = ? WHERE id = ?",
                (judgment_accuracy, stamp, now, candidate_id),
            )
        self._bump_heartbeat()

    def record_candidate_principle_marker(
        self,
        *,
        source: str,
        brief_id: str,
        identity_key: str,
        judgment_accuracy: str,
        principle_marker: dict[str, Any],
    ) -> None:
        """Record a per-principle calibration marker on a candidate.

        Slice 3.6 of ``plans/multi-agent-execution-plan.md``: unify
        per-principle Designer feedback with the candidate-level
        ``judgment_accuracy`` column. Same column, additional detail —
        the column gets the unified five-value enum value, the
        per-principle nuance lands in ``terminal_payload_json`` under
        ``principle_markers``.

        The full update — column write + JSON merge — runs in a single
        transaction so the calibration aggregator
        (``shared/runtime_state/calibration.py``) never sees a row where
        ``judgment_accuracy`` is set but the metadata trail is missing
        (or vice versa).

        ``judgment_accuracy`` is validated against the same set as
        :meth:`set_candidate_judgment_accuracy` (see store.py:660-666);
        the writer-side enum is the single source of truth so legacy or
        garbage values can't leak into the rollup. ``principle_marker``
        is appended verbatim to ``terminal_payload_json["principle_markers"]``
        — the caller (e.g., ``designer.recruiter_annotations``) owns
        the metadata shape because the meaning is module-specific.

        Raises ``ValueError`` for unknown candidates or invalid
        ``judgment_accuracy`` values. Does NOT auto-create candidates;
        the candidate must already exist (Designer's
        ``record_candidate_discovery`` puts it there).
        """

        allowed = {
            "useful",
            "wrong",
            "off_rubric",
            "overstated_depth",
            "understated_depth",
        }
        if judgment_accuracy not in allowed:
            raise ValueError(
                f"invalid judgment_accuracy: {judgment_accuracy!r}; "
                f"expected one of {sorted(allowed)}"
            )

        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, terminal_payload_json FROM candidates "
                "WHERE source = ? AND brief_id = ? AND identity_key = ?",
                (source, brief_id, identity_key),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"candidate not found: {source}:{brief_id}:{identity_key}"
                )

            try:
                payload = _json_loads(row["terminal_payload_json"])
            except (json.JSONDecodeError, TypeError):
                # Defensive: the column carries '{}' by default and writers
                # always serialize a dict, but a corrupt or hand-edited row
                # could break this invariant. The aggregator's read helper
                # at ``read_models.candidate_terminal_payload`` (read_models.py:821-841)
                # already collapses malformed payloads to ``None`` rather
                # than raising — mirror that here so the recruiter's
                # marker write doesn't surface a 500.
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            existing_markers = payload.get("principle_markers")
            if not isinstance(existing_markers, list):
                existing_markers = []
            existing_markers.append(dict(principle_marker))
            payload["principle_markers"] = existing_markers

            conn.execute(
                "UPDATE candidates SET "
                "judgment_accuracy = ?, judgment_accuracy_at = ?, "
                "terminal_payload_json = ?, last_seen_at = ? "
                "WHERE id = ?",
                (
                    judgment_accuracy,
                    now,
                    _json_dumps(payload),
                    now,
                    int(row["id"]),
                ),
            )
        self._bump_heartbeat()

    def finish_run(self, run_id: int, status: str, *, stop_reason: str = "normal") -> None:
        if status not in VALID_RUN_STATUSES:
            raise ValueError(
                f"invalid run status {status!r}; "
                f"valid values: {sorted(VALID_RUN_STATUSES)}"
            )
        normalized, detail = _split_stop_reason(stop_reason)
        with self.connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, stop_reason = ?, "
                "stop_reason_detail = ?, ended_at = ? WHERE id = ?",
                (status, normalized, detail, _utc_now(), run_id),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                event_type="run_finished",
                payload={
                    "status": status,
                    "stop_reason": normalized,
                    # Original (unnormalized) string preserved in the event
                    # payload so callers reading the event stream get the
                    # full forensic detail; the column carries the canonical
                    # enum value the UI consumes.
                    "stop_reason_detail": detail,
                },
            )
        self._bump_heartbeat()

    def set_run_stop_reason(self, run_id: int, stop_reason: str) -> None:
        normalized, detail = _split_stop_reason(stop_reason)
        with self.connect() as conn:
            conn.execute(
                "UPDATE runs SET stop_reason = ?, stop_reason_detail = ? "
                "WHERE id = ?",
                (normalized, detail, run_id),
            )

    def get_latest_run(self, *, source: str, brief_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM runs
                WHERE source = ? AND brief_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (source, brief_id),
            ).fetchone()
            return _row_to_dict(row) if row else None

    def list_runs(self, *, source: str, brief_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM runs
                WHERE source = ? AND brief_id = ?
                ORDER BY id DESC
                """,
                (source, brief_id),
            ).fetchall()
            return [_row_to_dict(row) for row in rows]

    def get_run(self, run_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return _row_to_dict(row) if row else None

    def update_run_resume_state(
        self,
        run_id: int,
        resume_state: dict,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        with self._write_connection(conn) as write_conn:
            write_conn.execute(
                "UPDATE runs SET resume_state_json = ? WHERE id = ?",
                (_json_dumps(resume_state), run_id),
            )

    def get_run_resume_state(self, run_id: int) -> dict:
        run = self.get_run(run_id)
        if not run:
            return {}
        return _json_loads(run.get("resume_state_json"))

    # ------------------------------------------------------------------
    # Work units
    # ------------------------------------------------------------------

    def has_work_units(self, run_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT 1 FROM work_units WHERE run_id = ? LIMIT 1", (run_id,)).fetchone()
            return row is not None

    def list_work_units(self, run_id: int, *, kind: str | None = None) -> list[dict]:
        sql = "SELECT * FROM work_units WHERE run_id = ?"
        params: list[Any] = [run_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY ordering_index ASC, id ASC"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_dict(row) for row in rows]

    def get_work_unit_by_source_id(self, run_id: int, *, kind: str, source_unit_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM work_units
                WHERE run_id = ? AND kind = ? AND source_unit_id = ?
                """,
                (run_id, kind, str(source_unit_id)),
            ).fetchone()
            return _row_to_dict(row) if row else None

    def get_work_unit_id(self, run_id: int, *, kind: str, source_unit_id: str) -> int | None:
        unit = self.get_work_unit_by_source_id(run_id, kind=kind, source_unit_id=source_unit_id)
        return int(unit["id"]) if unit else None

    def upsert_work_unit(
        self,
        *,
        run_id: int,
        source: str,
        brief_id: str,
        kind: str,
        source_unit_id: str,
        display_name: str,
        ordering_index: int,
        status: str,
        payload: dict | None = None,
        checkpoint: dict | None = None,
        metrics: dict | None = None,
        family_key: str = "",
        novelty_bucket: str = "",
        domain_lane: str = "",
        counters: dict | None = None,
        notes: str = "",
        conn: sqlite3.Connection | None = None,
        validate_status: bool = True,
    ) -> int:
        # Status validation is replay-tolerant: an unknown status passes only when
        # it is already this row's stored status (checked against the existing row
        # below, inside the write transaction). A forward-compat status persisted
        # by a newer version must survive an older version's checkpoint round-trip
        # — replayed, never rejected — while a status minted by the running
        # process fails closed. validate_status=False is the full opt-out for
        # true hydration writers (the legacy importer) that create rows from
        # persisted state this version cannot vouch for.
        status_needs_replay_check = (
            validate_status and status not in VALID_WORK_UNIT_STATUSES
        )
        if not isinstance(run_id, int) or run_id <= 0:
            raise ValueError(f"upsert_work_unit requires a positive int run_id, got {run_id!r}")
        if source not in KNOWN_CANDIDATE_SOURCES:
            raise ValueError(
                f"upsert_work_unit invalid source {source!r}; "
                f"valid values: {sorted(KNOWN_CANDIDATE_SOURCES)}"
            )
        try:
            payload_json = _json_dumps(payload or {})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"upsert_work_unit payload is not JSON-serializable: {exc}"
            ) from exc
        try:
            checkpoint_json = _json_dumps(checkpoint or {})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"upsert_work_unit checkpoint is not JSON-serializable: {exc}"
            ) from exc
        try:
            metrics_json = _json_dumps(metrics or {})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"upsert_work_unit metrics is not JSON-serializable: {exc}"
            ) from exc
        counters = counters or {}
        started_at = _utc_now() if status == "in_progress" else None
        ended_at = _utc_now() if status in TERMINAL_WORK_UNIT_STATUSES else None
        with self._write_connection(conn) as write_conn:
            existing = write_conn.execute(
                """
                SELECT id, started_at, status FROM work_units
                WHERE run_id = ? AND kind = ? AND source_unit_id = ?
                """,
                (run_id, kind, str(source_unit_id)),
            ).fetchone()
            if status_needs_replay_check and not (
                existing and existing["status"] == status
            ):
                raise ValueError(
                    f"invalid work unit status {status!r}; "
                    f"valid values: {sorted(VALID_WORK_UNIT_STATUSES)}"
                )
            if existing:
                started_at = existing["started_at"] or started_at
                if status not in TERMINAL_WORK_UNIT_STATUSES:
                    ended_at = None
                self._bump_heartbeat()
                write_conn.execute(
                    """
                    UPDATE work_units
                    SET display_name = ?, ordering_index = ?, status = ?, payload_json = ?, checkpoint_json = ?,
                        metrics_json = ?,
                        family_key = ?, novelty_bucket = ?, domain_lane = ?, result_count = ?, candidates_discovered = ?,
                        candidates_enriched = ?, candidates_insufficient = ?, facial_yes_count = ?, facial_no_count = ?,
                        facial_borderline_count = ?,
                        saves_count = ?, rejected_count = ?, notes = ?, started_at = ?, ended_at = ?
                    WHERE id = ?
                    """,
                    (
                        display_name,
                        ordering_index,
                        status,
                        payload_json,
                        checkpoint_json,
                        metrics_json,
                        family_key,
                        novelty_bucket,
                        domain_lane,
                        int(counters.get("result_count", 0)),
                        int(counters.get("candidates_discovered", 0)),
                        int(counters.get("candidates_enriched", 0)),
                        int(counters.get("candidates_insufficient", 0)),
                        int(counters.get("facial_yes_count", 0)),
                        int(counters.get("facial_no_count", 0)),
                        int(counters.get("facial_borderline_count", 0)),
                        int(counters.get("saves_count", 0)),
                        int(counters.get("rejected_count", 0)),
                        notes,
                        started_at,
                        ended_at,
                        int(existing["id"]),
                    ),
                )
                return int(existing["id"])

            cursor = write_conn.execute(
                """
                INSERT INTO work_units(
                    run_id, source, brief_id, kind, source_unit_id, display_name, ordering_index, status,
                    payload_json, checkpoint_json, metrics_json, family_key, novelty_bucket, domain_lane, result_count,
                    candidates_discovered, candidates_enriched, candidates_insufficient, facial_yes_count,
                    facial_no_count, facial_borderline_count, saves_count, rejected_count, notes, started_at, ended_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    source,
                    brief_id,
                    kind,
                    str(source_unit_id),
                    display_name,
                    ordering_index,
                    status,
                    payload_json,
                    checkpoint_json,
                    metrics_json,
                    family_key,
                    novelty_bucket,
                    domain_lane,
                    int(counters.get("result_count", 0)),
                    int(counters.get("candidates_discovered", 0)),
                    int(counters.get("candidates_enriched", 0)),
                    int(counters.get("candidates_insufficient", 0)),
                    int(counters.get("facial_yes_count", 0)),
                    int(counters.get("facial_no_count", 0)),
                    int(counters.get("facial_borderline_count", 0)),
                    int(counters.get("saves_count", 0)),
                    int(counters.get("rejected_count", 0)),
                    notes,
                    started_at,
                    ended_at,
                ),
            )
            return int(cursor.lastrowid)

    def delete_missing_work_units(
        self,
        run_id: int,
        *,
        kind: str,
        keep_source_unit_ids: set[str],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        with self._write_connection(conn) as write_conn:
            existing = write_conn.execute(
                "SELECT source_unit_id FROM work_units WHERE run_id = ? AND kind = ?",
                (run_id, kind),
            ).fetchall()
            for row in existing:
                if row["source_unit_id"] not in keep_source_unit_ids:
                    write_conn.execute(
                        "DELETE FROM work_units WHERE run_id = ? AND kind = ? AND source_unit_id = ?",
                        (run_id, kind, row["source_unit_id"]),
                    )

    def requeue_work_unit(self, run_id: int, *, kind: str, source_unit_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE work_units
                SET status = 'queued', checkpoint_json = '{}', started_at = NULL, ended_at = NULL
                WHERE run_id = ? AND kind = ? AND source_unit_id = ?
                """,
                (run_id, kind, str(source_unit_id)),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                work_unit_id=self.get_work_unit_id(run_id, kind=kind, source_unit_id=source_unit_id),
                event_type="work_unit_requeued",
                payload={"kind": kind, "source_unit_id": str(source_unit_id)},
            )

    # ------------------------------------------------------------------
    # Candidate lifecycle
    # ------------------------------------------------------------------

    def _current_brief_content_hash(
        self, conn: sqlite3.Connection, *, source: str, brief_id: str
    ) -> str | None:
        """The latest run's brief hash — the 'current brief' for suppression."""
        row = conn.execute(
            """
            SELECT brief_content_hash FROM runs
            WHERE source = ? AND brief_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (source, brief_id),
        ).fetchone()
        return row["brief_content_hash"] if row else None

    # P3.1: the hash-aware suppression clause. A terminal verdict suppresses
    # a candidate iff it was made under the CURRENT brief content — EXCEPT
    # SAVE-family decisions and recruiter-made decisions (user_status), which
    # always suppress: never re-save, never overrule a human. NULL hashes
    # (legacy rows not yet backfilled) keep today's behavior (suppressed).
    _HASH_AWARE_SUPPRESSION_CLAUSE = """
                  AND (
                    terminal_decision IN ({save_placeholders})
                    OR user_status IS NOT NULL
                    OR brief_content_hash IS NULL
                    OR brief_content_hash = ?
                  )
    """

    def list_terminal_identity_keys(self, *, source: str, brief_id: str) -> list[str]:
        placeholders = ",".join("?" for _ in DEDUP_BLOCKING_DECISIONS)
        with self.connect() as conn:
            current_hash = self._current_brief_content_hash(
                conn, source=source, brief_id=brief_id
            )
            hash_clause = ""
            hash_params: list[Any] = []
            if current_hash:
                save_placeholders = ",".join("?" for _ in SAVE_FAMILY_DECISIONS)
                hash_clause = self._HASH_AWARE_SUPPRESSION_CLAUSE.format(
                    save_placeholders=save_placeholders
                )
                hash_params = [*sorted(SAVE_FAMILY_DECISIONS), current_hash]
            rows = conn.execute(
                f"""
                SELECT identity_key
                FROM candidates
                WHERE source = ? AND brief_id = ?
                  AND (
                    current_lifecycle_state = 'failed_terminal'
                    OR terminal_decision IN ({placeholders})
                  )
                {hash_clause}
                ORDER BY identity_key ASC
                """,
                (source, brief_id, *sorted(DEDUP_BLOCKING_DECISIONS), *hash_params),
            ).fetchall()
            return [str(row["identity_key"]) for row in rows]

    def list_terminal_person_keys(self, *, source: str, brief_id: str) -> list[str]:
        placeholders = ",".join("?" for _ in DEDUP_BLOCKING_DECISIONS)
        with self.connect() as conn:
            current_hash = self._current_brief_content_hash(
                conn, source=source, brief_id=brief_id
            )
            hash_clause = ""
            hash_params: list[Any] = []
            if current_hash:
                save_placeholders = ",".join("?" for _ in SAVE_FAMILY_DECISIONS)
                hash_clause = self._HASH_AWARE_SUPPRESSION_CLAUSE.format(
                    save_placeholders=save_placeholders
                )
                hash_params = [*sorted(SAVE_FAMILY_DECISIONS), current_hash]
            rows = conn.execute(
                f"""
                SELECT person_key, identity_key
                FROM candidates
                WHERE source = ? AND brief_id = ?
                  AND (
                    current_lifecycle_state = 'failed_terminal'
                    OR terminal_decision IN ({placeholders})
                  )
                {hash_clause}
                ORDER BY COALESCE(person_key, identity_key) ASC
                """,
                (source, brief_id, *sorted(DEDUP_BLOCKING_DECISIONS), *hash_params),
            ).fetchall()
            keys: list[str] = []
            for row in rows:
                person_key = row["person_key"]
                if person_key:
                    keys.append(str(person_key))
                elif source == "github":
                    keys.append(f"gh:{str(row['identity_key']).lower()}")
            return keys

    def is_dedup_blocked(self, *, source: str, brief_id: str, identity_key: str) -> bool:
        placeholders = ",".join("?" for _ in DEDUP_BLOCKING_DECISIONS)
        with self.connect() as conn:
            current_hash = self._current_brief_content_hash(
                conn, source=source, brief_id=brief_id
            )
            hash_clause = ""
            hash_params: list[Any] = []
            if current_hash:
                save_placeholders = ",".join("?" for _ in SAVE_FAMILY_DECISIONS)
                hash_clause = self._HASH_AWARE_SUPPRESSION_CLAUSE.format(
                    save_placeholders=save_placeholders
                )
                hash_params = [*sorted(SAVE_FAMILY_DECISIONS), current_hash]
            row = conn.execute(
                f"""
                SELECT 1
                FROM candidates
                WHERE source = ? AND brief_id = ? AND identity_key = ?
                  AND (
                    current_lifecycle_state = 'failed_terminal'
                    OR terminal_decision IN ({placeholders})
                  )
                {hash_clause}
                LIMIT 1
                """,
                (
                    source,
                    brief_id,
                    identity_key,
                    *sorted(DEDUP_BLOCKING_DECISIONS),
                    *hash_params,
                ),
            ).fetchone()
            return row is not None

    def is_person_key_blocked(self, *, brief_id: str, person_key: str) -> bool:
        if not person_key:
            return False
        placeholders = ",".join("?" for _ in DEDUP_BLOCKING_DECISIONS)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT 1
                FROM candidates
                WHERE brief_id = ? AND person_key = ?
                  AND (
                    current_lifecycle_state = 'failed_terminal'
                    OR terminal_decision IN ({placeholders})
                  )
                LIMIT 1
                """,
                (brief_id, person_key, *sorted(DEDUP_BLOCKING_DECISIONS)),
            ).fetchone()
            return row is not None

    def get_blocked_person_keys(self, brief_id: str, person_keys: list[str]) -> set[str]:
        if not person_keys:
            return set()
        placeholders = ",".join("?" for _ in person_keys)
        decision_placeholders = ",".join("?" for _ in DEDUP_BLOCKING_DECISIONS)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT person_key
                FROM candidates
                WHERE brief_id = ?
                  AND person_key IN ({placeholders})
                  AND (
                    current_lifecycle_state = 'failed_terminal'
                    OR terminal_decision IN ({decision_placeholders})
                  )
                """,
                [brief_id, *person_keys, *sorted(DEDUP_BLOCKING_DECISIONS)],
            ).fetchall()
            return {str(row["person_key"]) for row in rows if row["person_key"]}

    def ensure_candidate(
        self,
        *,
        source: str,
        brief_id: str,
        identity_key: str,
        display_name: str = "",
        profile_url: str = "",
        initial_state: str = "discovered",
        person_key: str | None = None,
    ) -> int:
        now = _utc_now()
        if person_key is None and source == "github":
            person_key = f"gh:{identity_key.lower()}"
        # Phase C-bis 0.4: defense-in-depth URL normalization for LinkedIn.
        # The acquisition layer already cleans tracking parameters, but
        # any future code path that bypasses acquisition (e.g. a manual
        # backfill, a different module) gets cleaned here as a safety
        # net. The normalizer is idempotent and safe on empty strings.
        if source == "linkedin" and profile_url:
            from shared.identity_resolution import normalize_public_linkedin_url

            profile_url = normalize_public_linkedin_url(profile_url)
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, display_name, profile_url, current_lifecycle_state
                FROM candidates
                WHERE source = ? AND brief_id = ? AND identity_key = ?
                """,
                (source, brief_id, identity_key),
            ).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE candidates
                    SET display_name = ?, profile_url = ?, last_seen_at = ?,
                        person_key = COALESCE(person_key, ?)
                    WHERE id = ?
                    """,
                    (
                        display_name or row["display_name"],
                        profile_url or row["profile_url"],
                        now,
                        person_key,
                        int(row["id"]),
                    ),
                )
                return int(row["id"])

            cursor = conn.execute(
                """
                INSERT INTO candidates(
                    source, brief_id, identity_key, person_key, display_name,
                    profile_url, current_lifecycle_state, terminal_decision,
                    terminal_payload_json, first_seen_at, last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, '{}', ?, ?)
                """,
                (
                    source,
                    brief_id,
                    identity_key,
                    person_key,
                    display_name,
                    profile_url,
                    initial_state,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def record_candidate_discovery(
        self,
        *,
        run_id: int,
        work_unit_id: int | None,
        source: str,
        brief_id: str,
        identity_key: str,
        display_name: str = "",
        profile_url: str = "",
        payload: dict | None = None,
        person_key: str | None = None,
    ) -> int:
        if str(identity_key).strip() == "":
            raise ValueError("record_candidate_discovery requires a non-empty identity_key")
        if not isinstance(run_id, int) or run_id <= 0:
            raise ValueError(
                f"record_candidate_discovery requires a positive int run_id, got {run_id!r}"
            )
        if source not in KNOWN_CANDIDATE_SOURCES:
            raise ValueError(
                f"record_candidate_discovery invalid source {source!r}; "
                f"valid values: {sorted(KNOWN_CANDIDATE_SOURCES)}"
            )
        if payload is not None:
            try:
                _json_dumps(payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"record_candidate_discovery payload is not JSON-serializable: {exc}"
                ) from exc
        # Phase C-bis 0.4: defense-in-depth URL normalization for LinkedIn.
        # ensure_candidate normalizes on insert, but the UPDATE below would
        # otherwise overwrite the clean value with whatever the caller
        # passed in. Normalize once here so both branches converge on the
        # cleaned URL. Idempotent + scoped to LinkedIn.
        if source == "linkedin" and profile_url:
            from shared.identity_resolution import normalize_public_linkedin_url

            profile_url = normalize_public_linkedin_url(profile_url)
        candidate_id = self.ensure_candidate(
            source=source,
            brief_id=brief_id,
            identity_key=identity_key,
            display_name=display_name,
            profile_url=profile_url,
            initial_state="discovered",
            person_key=person_key,
        )
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE candidates
                SET display_name = ?, profile_url = ?,
                    current_lifecycle_state = CASE
                        WHEN current_lifecycle_state IN ('full_terminal', 'failed_terminal')
                        THEN current_lifecycle_state
                        ELSE 'discovered'
                    END,
                    last_work_unit_id = ?, last_seen_at = ?
                WHERE id = ?
                """,
                (
                    display_name,
                    profile_url,
                    work_unit_id,
                    _utc_now(),
                    candidate_id,
                ),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                work_unit_id=work_unit_id,
                candidate_id=candidate_id,
                event_type="candidate_discovered",
                payload=payload or {"identity_key": identity_key},
            )
        return candidate_id

    def get_candidate(self, *, source: str, brief_id: str, identity_key: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM candidates
                WHERE source = ? AND brief_id = ? AND identity_key = ?
                """,
                (source, brief_id, identity_key),
            ).fetchone()
            return _row_to_dict(row) if row else None

    def set_candidate_state(
        self,
        *,
        run_id: int | None,
        source: str,
        brief_id: str,
        identity_key: str,
        new_state: str,
        terminal_decision: str | None = None,
        terminal_payload: dict | None = None,
        last_work_unit_id: int | None = None,
    ) -> None:
        candidate = self.get_candidate(source=source, brief_id=brief_id, identity_key=identity_key)
        if not candidate:
            raise ValueError(f"candidate not found: {source}:{brief_id}:{identity_key}")
        retrying_failed_linkedin_save = False
        if (
            source == "linkedin"
            and candidate["current_lifecycle_state"] == "full_terminal"
            and new_state == "full_started"
            and candidate.get("terminal_decision") in SAVE_FAMILY_DECISIONS
        ):
            with self.connect() as conn:
                latest_save = conn.execute(
                    """
                    SELECT status
                    FROM side_effects
                    WHERE candidate_id = ?
                      AND effect_type = 'linkedin_save'
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (int(candidate["id"]),),
                ).fetchone()
                retrying_failed_linkedin_save = bool(
                    latest_save
                    and latest_save["status"]
                    in {
                        "failed",
                        "failed_permanent",
                        "interrupted",
                        "invalidated",
                    }
                )
        if not retrying_failed_linkedin_save:
            _guard_save_family_reentry(
                candidate["current_lifecycle_state"],
                candidate.get("terminal_decision"),
                new_state,
            )
            _guard_transition(candidate["current_lifecycle_state"], new_state)
        with self.connect() as conn:
            # P3.1: stamp the deciding run's brief hash onto terminal
            # decisions so suppression can be brief-version-aware. No caller
            # change needed — the run row already carries the hash.
            terminal_hash = None
            if (terminal_decision or new_state == "failed_terminal") and run_id:
                run_row = conn.execute(
                    "SELECT brief_content_hash FROM runs WHERE id = ?",
                    (run_id,),
                ).fetchone()
                if run_row:
                    terminal_hash = run_row["brief_content_hash"]
            conn.execute(
                """
                UPDATE candidates
                SET current_lifecycle_state = ?, terminal_decision = ?, terminal_payload_json = ?,
                    last_work_unit_id = COALESCE(?, last_work_unit_id), last_seen_at = ?,
                    brief_content_hash = COALESCE(?, brief_content_hash)
                WHERE id = ?
                """,
                (
                    new_state,
                    terminal_decision,
                    _json_dumps(terminal_payload or {}),
                    last_work_unit_id,
                    _utc_now(),
                    terminal_hash,
                    candidate["id"],
                ),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                work_unit_id=last_work_unit_id,
                candidate_id=int(candidate["id"]),
                event_type="candidate_state_transition",
                payload={
                    "from_state": candidate["current_lifecycle_state"],
                    "to_state": new_state,
                    "terminal_decision": terminal_decision,
                },
            )

    def mark_candidate_terminal_runtime(
        self,
        *,
        run_id: int | None,
        source: str,
        brief_id: str,
        identity_key: str,
        decision: str,
        payload: dict | None = None,
        last_work_unit_id: int | None = None,
    ) -> None:
        self.set_candidate_state(
            run_id=run_id,
            source=source,
            brief_id=brief_id,
            identity_key=identity_key,
            new_state="failed_terminal",
            terminal_decision=decision,
            terminal_payload=payload,
            last_work_unit_id=last_work_unit_id,
        )

    def clear_candidate_terminal_state(
        self,
        *,
        source: str,
        brief_id: str,
        identity_key: str,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        with self._write_connection(conn) as write_conn:
            candidate = write_conn.execute(
                """
                SELECT id FROM candidates
                WHERE source = ? AND brief_id = ? AND identity_key = ?
                """,
                (source, brief_id, identity_key),
            ).fetchone()
            if not candidate:
                raise ValueError(f"candidate not found: {source}:{brief_id}:{identity_key}")
            write_conn.execute(
                """
                UPDATE candidates
                SET terminal_decision = NULL,
                    terminal_payload_json = '{}',
                    current_lifecycle_state = CASE
                        WHEN current_lifecycle_state = 'failed_terminal' THEN 'failed_retryable'
                        ELSE current_lifecycle_state
                    END,
                    last_seen_at = ?
                WHERE id = ?
                """,
                (_utc_now(), int(candidate["id"])),
            )
            self._insert_event(
                write_conn,
                candidate_id=int(candidate["id"]),
                event_type="candidate_terminal_cleared",
                payload={"identity_key": identity_key},
            )

    # ------------------------------------------------------------------
    # Attempts
    # ------------------------------------------------------------------

    def start_attempt(
        self,
        *,
        run_id: int,
        source: str,
        brief_id: str,
        identity_key: str,
        stage: str,
        work_unit_id: int | None = None,
        batch_key: str | None = None,
        payload: dict | None = None,
        source_cursor: dict | None = None,
        display_name: str = "",
        profile_url: str = "",
    ) -> int:
        candidate_id = self.ensure_candidate(
            source=source,
            brief_id=brief_id,
            identity_key=identity_key,
            display_name=display_name,
            profile_url=profile_url,
        )
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(attempt_number), 0) AS max_attempt
                FROM candidate_attempts
                WHERE candidate_id = ? AND stage = ?
                """,
                (candidate_id, stage),
            ).fetchone()
            attempt_number = int(row["max_attempt"]) + 1
            cursor = conn.execute(
                """
                INSERT INTO candidate_attempts(
                    run_id, candidate_id, work_unit_id, stage, attempt_number, batch_key,
                    status, payload_json, source_cursor_json, started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'started', ?, ?, ?)
                """,
                (
                    run_id,
                    candidate_id,
                    work_unit_id,
                    stage,
                    attempt_number,
                    batch_key,
                    _json_dumps(payload or {}),
                    _json_dumps(source_cursor or {}),
                    _utc_now(),
                ),
            )
            attempt_id = int(cursor.lastrowid)
            conn.execute(
                """
                UPDATE candidates
                SET last_attempt_id = ?, last_work_unit_id = COALESCE(?, last_work_unit_id), last_seen_at = ?
                WHERE id = ?
                """,
                (attempt_id, work_unit_id, _utc_now(), candidate_id),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                work_unit_id=work_unit_id,
                candidate_id=candidate_id,
                attempt_id=attempt_id,
                event_type="attempt_started",
                payload={"stage": stage, "attempt_number": attempt_number},
            )
            return attempt_id

    def finish_attempt_success(
        self,
        *,
        attempt_id: int,
        new_state: str,
        terminal_decision: str | None = None,
        payload: dict | None = None,
        terminal_payload: dict | None | object = _UNSET,
        run_id: int | None = None,
    ) -> None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT ca.*, c.source, c.brief_id, c.identity_key,
                       c.current_lifecycle_state, c.terminal_decision
                FROM candidate_attempts ca
                JOIN candidates c ON c.id = ca.candidate_id
                WHERE ca.id = ?
                """,
                (attempt_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"attempt not found: {attempt_id}")
            _guard_save_family_reentry(
                row["current_lifecycle_state"], row["terminal_decision"], new_state
            )
            _guard_transition(row["current_lifecycle_state"], new_state)
            attempt_payload = payload or {}
            effective_terminal_payload = (
                attempt_payload
                if terminal_payload is _UNSET
                else (terminal_payload or {})
            )
            event_observability = _event_observability_fields(attempt_payload)
            conn.execute(
                """
                UPDATE candidate_attempts
                SET status = 'succeeded', payload_json = ?, ended_at = ?
                WHERE id = ?
                """,
                (_json_dumps(attempt_payload), _utc_now(), attempt_id),
            )
            conn.execute(
                """
                UPDATE candidates
                SET current_lifecycle_state = ?, terminal_decision = ?, terminal_payload_json = ?,
                    last_attempt_id = ?, last_work_unit_id = COALESCE(?, last_work_unit_id), last_seen_at = ?
                WHERE id = ?
                """,
                (
                    new_state,
                    terminal_decision,
                    _json_dumps(effective_terminal_payload),
                    attempt_id,
                    row["work_unit_id"],
                    _utc_now(),
                    row["candidate_id"],
                ),
            )
            self._insert_event(
                conn,
                run_id=run_id or row["run_id"],
                work_unit_id=row["work_unit_id"],
                candidate_id=row["candidate_id"],
                attempt_id=attempt_id,
                event_type="attempt_succeeded",
                payload={
                    "stage": row["stage"],
                    "new_state": new_state,
                    "terminal_decision": terminal_decision,
                    **event_observability,
                },
            )
        self._bump_heartbeat()

    def finish_attempt_failure(
        self,
        *,
        attempt_id: int,
        failure_kind: str,
        failure_reason: str,
        retryable: bool,
        payload: dict | None = None,
        run_id: int | None = None,
    ) -> None:
        new_state = "failed_retryable" if retryable else "failed_terminal"
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT ca.*, c.current_lifecycle_state
                FROM candidate_attempts ca
                JOIN candidates c ON c.id = ca.candidate_id
                WHERE ca.id = ?
                """,
                (attempt_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"attempt not found: {attempt_id}")
            _guard_transition(row["current_lifecycle_state"], new_state)
            attempt_payload = payload or {}
            event_observability = _event_observability_fields(attempt_payload)
            conn.execute(
                """
                UPDATE candidate_attempts
                SET status = 'failed', failure_kind = ?, failure_reason = ?, payload_json = ?, ended_at = ?
                WHERE id = ?
                """,
                (
                    failure_kind,
                    failure_reason,
                    _json_dumps(attempt_payload),
                    _utc_now(),
                    attempt_id,
                ),
            )
            conn.execute(
                """
                UPDATE candidates
                SET current_lifecycle_state = ?, last_attempt_id = ?, last_work_unit_id = COALESCE(?, last_work_unit_id), last_seen_at = ?
                WHERE id = ?
                """,
                (
                    new_state,
                    attempt_id,
                    row["work_unit_id"],
                    _utc_now(),
                    row["candidate_id"],
                ),
            )
            self._insert_event(
                conn,
                run_id=run_id or row["run_id"],
                work_unit_id=row["work_unit_id"],
                candidate_id=row["candidate_id"],
                attempt_id=attempt_id,
                event_type="attempt_failed",
                payload={
                    "stage": row["stage"],
                    "new_state": new_state,
                    "failure_kind": failure_kind,
                    "failure_reason": failure_reason,
                    **event_observability,
                },
            )
        self._bump_heartbeat()

    def list_orphaned_attempts(self, *, source: str, brief_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ca.*, c.identity_key, c.source, c.brief_id
                FROM candidate_attempts ca
                JOIN candidates c ON c.id = ca.candidate_id
                WHERE ca.status = 'started' AND c.source = ? AND c.brief_id = ?
                ORDER BY ca.id ASC
                """,
                (source, brief_id),
            ).fetchall()
            return [_row_to_dict(row) for row in rows]

    def reconcile_open_attempts(self, *, source: str, brief_id: str) -> int:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ca.id, ca.run_id, ca.work_unit_id, ca.stage, ca.candidate_id, c.current_lifecycle_state
                FROM candidate_attempts ca
                JOIN candidates c ON c.id = ca.candidate_id
                WHERE ca.status = 'started' AND c.source = ? AND c.brief_id = ?
                ORDER BY ca.id ASC
                """,
                (source, brief_id),
            ).fetchall()
            reconciled = 0
            for row in rows:
                current_state = row["current_lifecycle_state"]
                if current_state not in ("full_terminal", "failed_terminal"):
                    _guard_transition(current_state, "failed_retryable")
                    conn.execute(
                        """
                        UPDATE candidates
                        SET current_lifecycle_state = 'failed_retryable', last_attempt_id = ?, last_work_unit_id = COALESCE(?, last_work_unit_id), last_seen_at = ?
                        WHERE id = ?
                        """,
                        (
                            row["id"],
                            row["work_unit_id"],
                            _utc_now(),
                            row["candidate_id"],
                        ),
                    )
                conn.execute(
                    """
                    UPDATE candidate_attempts
                    SET status = 'reconciled',
                        failure_kind = COALESCE(failure_kind, 'orphaned_attempt'),
                        failure_reason = COALESCE(failure_reason, 'interrupted before attempt completion'),
                        ended_at = ?
                    WHERE id = ?
                    """,
                    (_utc_now(), row["id"]),
                )
                self._insert_event(
                    conn,
                    run_id=row["run_id"],
                    work_unit_id=row["work_unit_id"],
                    candidate_id=row["candidate_id"],
                    attempt_id=row["id"],
                    event_type="orphaned_attempt_reconciled",
                    payload={"stage": row["stage"]},
                )
                reconciled += 1
            return reconciled

    # ------------------------------------------------------------------
    # Candidate side effects
    # ------------------------------------------------------------------

    def begin_candidate_side_effect(
        self,
        *,
        run_id: int | None,
        source: str,
        brief_id: str,
        identity_key: str,
        attempt_id: int | None,
        effect_type: str,
        idempotency_key: str,
        payload: dict | None = None,
    ) -> dict:
        candidate = self.get_candidate(source=source, brief_id=brief_id, identity_key=identity_key)
        if not candidate:
            raise ValueError(f"candidate not found: {source}:{brief_id}:{identity_key}")
        now = _utc_now()
        with self.connect() as conn:
            existing = conn.execute(
                """
                SELECT *
                FROM side_effects
                WHERE candidate_id = ? AND effect_type = ? AND idempotency_key = ?
                """,
                (int(candidate["id"]), effect_type, idempotency_key),
            ).fetchone()
            if existing:
                existing_dict = _row_to_dict(existing)
                existing_status = str(existing["status"])
                attempt_count = int(existing_dict.get("attempt_count") or 1)

                if existing_status == "invalidated":
                    # Deliberate reset (rediscovery): fresh attempt cycle.
                    conn.execute(
                        """
                        UPDATE side_effects
                        SET run_id = ?, attempt_id = ?, status = 'pending', payload_json = ?, attempt_count = 1, updated_at = ?, invalidated_at = NULL
                        WHERE id = ?
                        """,
                        (
                            run_id,
                            attempt_id,
                            _json_dumps(payload or {}),
                            now,
                            int(existing["id"]),
                        ),
                    )
                    self._insert_event(
                        conn,
                        run_id=run_id,
                        candidate_id=int(candidate["id"]),
                        attempt_id=attempt_id,
                        event_type="side_effect_pending",
                        payload={
                            "effect_type": effect_type,
                            "idempotency_key": idempotency_key,
                            "replayed_from_invalidated": True,
                        },
                    )
                    existing_dict.update(
                        {
                            "run_id": run_id,
                            "attempt_id": attempt_id,
                            "status": "pending",
                            "payload_json": _json_dumps(payload or {}),
                            "attempt_count": 1,
                            "updated_at": now,
                            "invalidated_at": None,
                        }
                    )
                    return {"should_execute": True, "side_effect": existing_dict}

                if existing_status == "interrupted":
                    # Crash-interrupted execution: retry WITHOUT consuming an
                    # attempt (the interrupted execution may never have run).
                    conn.execute(
                        """
                        UPDATE side_effects
                        SET run_id = ?, attempt_id = ?, status = 'pending', payload_json = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            run_id,
                            attempt_id,
                            _json_dumps(payload or {}),
                            now,
                            int(existing["id"]),
                        ),
                    )
                    self._insert_event(
                        conn,
                        run_id=run_id,
                        candidate_id=int(candidate["id"]),
                        attempt_id=attempt_id,
                        event_type="side_effect_pending",
                        payload={
                            "effect_type": effect_type,
                            "idempotency_key": idempotency_key,
                            "replayed_from_interrupted": True,
                            "attempt_count": attempt_count,
                        },
                    )
                    existing_dict.update(
                        {
                            "run_id": run_id,
                            "attempt_id": attempt_id,
                            "status": "pending",
                            "payload_json": _json_dumps(payload or {}),
                            "updated_at": now,
                        }
                    )
                    return {"should_execute": True, "side_effect": existing_dict}

                linkedin_save = (
                    source == "linkedin" and effect_type == "linkedin_save"
                )
                if existing_status in {"failed", "failed_permanent"} and (
                    linkedin_save or attempt_count < SIDE_EFFECT_MAX_ATTEMPTS
                ):
                    # Retryable failure: consume an attempt. The execute path
                    # is double-save-proof (linkedin/side_effects checks
                    # is_already_saved()), so a retry can never double-save.
                    new_attempt_count = attempt_count + 1
                    conn.execute(
                        """
                        UPDATE side_effects
                        SET run_id = ?, attempt_id = ?, status = 'pending', payload_json = ?, attempt_count = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            run_id,
                            attempt_id,
                            _json_dumps(payload or {}),
                            new_attempt_count,
                            now,
                            int(existing["id"]),
                        ),
                    )
                    self._insert_event(
                        conn,
                        run_id=run_id,
                        candidate_id=int(candidate["id"]),
                        attempt_id=attempt_id,
                        event_type="side_effect_pending",
                        payload={
                            "effect_type": effect_type,
                            "idempotency_key": idempotency_key,
                            "replayed_from_failed": existing_status == "failed",
                            "replayed_from_failed_permanent": (
                                existing_status == "failed_permanent"
                            ),
                            "attempt_count": new_attempt_count,
                        },
                    )
                    existing_dict.update(
                        {
                            "run_id": run_id,
                            "attempt_id": attempt_id,
                            "status": "pending",
                            "payload_json": _json_dumps(payload or {}),
                            "attempt_count": new_attempt_count,
                            "updated_at": now,
                        }
                    )
                    return {"should_execute": True, "side_effect": existing_dict}

                if existing_status == "failed":
                    # Attempts exhausted: flip to failed_permanent so the row
                    # is skipped AND distinguishable from a retryable failure
                    # in save-health reporting (P1.3).
                    conn.execute(
                        """
                        UPDATE side_effects
                        SET status = 'failed_permanent', updated_at = ?
                        WHERE id = ?
                        """,
                        (now, int(existing["id"])),
                    )
                    self._insert_event(
                        conn,
                        run_id=run_id,
                        candidate_id=int(candidate["id"]),
                        attempt_id=attempt_id,
                        event_type="side_effect_result",
                        payload={
                            "effect_type": effect_type,
                            "status": "failed_permanent",
                            "idempotency_key": idempotency_key,
                            "skip_reason": "attempts_exhausted",
                            "attempt_count": attempt_count,
                        },
                    )
                    existing_dict.update(
                        {"status": "failed_permanent", "updated_at": now}
                    )
                    return {"should_execute": False, "side_effect": existing_dict}

                self._insert_event(
                    conn,
                    run_id=run_id,
                    candidate_id=int(candidate["id"]),
                    attempt_id=attempt_id,
                    event_type="side_effect_result",
                    payload={
                        "effect_type": effect_type,
                        "status": "skipped",
                        "idempotency_key": idempotency_key,
                        "skip_reason": f"existing_{existing['status']}",
                    },
                )
                return {"should_execute": False, "side_effect": _row_to_dict(existing)}

            cursor = conn.execute(
                """
                INSERT INTO side_effects(
                    run_id, candidate_id, attempt_id, effect_type, idempotency_key,
                    status, payload_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    run_id,
                    int(candidate["id"]),
                    attempt_id,
                    effect_type,
                    idempotency_key,
                    _json_dumps(payload or {}),
                    now,
                    now,
                ),
            )
            side_effect_id = int(cursor.lastrowid)
            self._insert_event(
                conn,
                run_id=run_id,
                candidate_id=int(candidate["id"]),
                attempt_id=attempt_id,
                event_type="side_effect_pending",
                payload={
                    "effect_type": effect_type,
                    "idempotency_key": idempotency_key,
                },
            )
            row = conn.execute("SELECT * FROM side_effects WHERE id = ?", (side_effect_id,)).fetchone()
            return {"should_execute": True, "side_effect": _row_to_dict(row)}

    def complete_candidate_side_effect(
        self,
        *,
        side_effect_id: int,
        status: str,
        payload: dict | None = None,
    ) -> None:
        if status not in SIDE_EFFECT_TERMINAL_STATUSES:
            raise ValueError(f"invalid side_effect status: {status}")
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM side_effects WHERE id = ?", (side_effect_id,)).fetchone()
            if not row:
                raise ValueError(f"side effect not found: {side_effect_id}")
            conn.execute(
                """
                UPDATE side_effects
                SET status = ?, payload_json = ?, updated_at = ?, invalidated_at = CASE
                    WHEN ? = 'invalidated' THEN ?
                    ELSE invalidated_at
                END
                WHERE id = ?
                """,
                (
                    status,
                    _json_dumps(payload or {}),
                    _utc_now(),
                    status,
                    _utc_now(),
                    side_effect_id,
                ),
            )
            self._insert_event(
                conn,
                run_id=row["run_id"],
                candidate_id=row["candidate_id"],
                attempt_id=row["attempt_id"],
                event_type="side_effect_result",
                payload={
                    "effect_type": row["effect_type"],
                    "status": status,
                    "idempotency_key": row["idempotency_key"],
                    **(payload or {}),
                },
            )

    def list_candidate_side_effects(
        self,
        *,
        source: str,
        brief_id: str,
        status: str | None = None,
        identity_key: str | None = None,
    ) -> list[dict]:
        sql = """
            SELECT se.*, c.identity_key, c.source, c.brief_id
            FROM side_effects se
            JOIN candidates c ON c.id = se.candidate_id
            WHERE c.source = ? AND c.brief_id = ?
        """
        params: list[Any] = [source, brief_id]
        if status is not None:
            sql += " AND se.status = ?"
            params.append(status)
        if identity_key is not None:
            sql += " AND c.identity_key = ?"
            params.append(identity_key)
        sql += " ORDER BY se.id ASC"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_dict(row) for row in rows]

    def reconcile_pending_side_effects(self, *, source: str, brief_id: str) -> int:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT se.*, c.identity_key
                FROM side_effects se
                JOIN candidates c ON c.id = se.candidate_id
                WHERE se.status = 'pending' AND c.source = ? AND c.brief_id = ?
                ORDER BY se.id ASC
                """,
                (source, brief_id),
            ).fetchall()
            reconciled = 0
            for row in rows:
                # P1.1: crash-interrupted pending rows become 'interrupted'
                # (retryable, no attempt consumed) — NOT 'failed'. The
                # original payload is preserved; the interruption is noted
                # under its own key instead of clobbering the payload.
                original_payload = {}
                try:
                    original_payload = json.loads(row["payload_json"] or "{}")
                except (TypeError, ValueError):
                    original_payload = {}
                if not isinstance(original_payload, dict):
                    original_payload = {"original_payload": original_payload}
                original_payload["interruption"] = {
                    "reason": "interrupted",
                    "reconciled_at": _utc_now(),
                }
                conn.execute(
                    """
                    UPDATE side_effects
                    SET status = 'interrupted',
                        payload_json = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        _json_dumps(original_payload),
                        _utc_now(),
                        int(row["id"]),
                    ),
                )
                self._insert_event(
                    conn,
                    run_id=row["run_id"],
                    candidate_id=row["candidate_id"],
                    attempt_id=row["attempt_id"],
                    event_type="side_effect_result",
                    payload={
                        "effect_type": row["effect_type"],
                        "status": "interrupted",
                        "idempotency_key": row["idempotency_key"],
                        "reason": "interrupted",
                    },
                )
                reconciled += 1
            return reconciled

    def invalidate_candidate_side_effects(
        self,
        *,
        source: str,
        brief_id: str,
        identity_key: str,
        effect_type: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        with self._write_connection(conn) as write_conn:
            candidate = write_conn.execute(
                """
                SELECT id FROM candidates
                WHERE source = ? AND brief_id = ? AND identity_key = ?
                """,
                (source, brief_id, identity_key),
            ).fetchone()
            if not candidate:
                return 0
            sql = """
                SELECT id, run_id, candidate_id, attempt_id, effect_type, idempotency_key
                FROM side_effects
                WHERE candidate_id = ?
                  AND status != 'invalidated'
            """
            params: list[Any] = [int(candidate["id"])]
            if effect_type is not None:
                sql += " AND effect_type = ?"
                params.append(effect_type)
            rows = write_conn.execute(sql, params).fetchall()
            for row in rows:
                write_conn.execute(
                    """
                    UPDATE side_effects
                    SET status = 'invalidated', updated_at = ?, invalidated_at = ?
                    WHERE id = ?
                    """,
                    (_utc_now(), _utc_now(), int(row["id"])),
                )
                self._insert_event(
                    write_conn,
                    run_id=row["run_id"],
                    candidate_id=row["candidate_id"],
                    attempt_id=row["attempt_id"],
                    event_type="side_effect_result",
                    payload={
                        "effect_type": row["effect_type"],
                        "status": "invalidated",
                        "idempotency_key": row["idempotency_key"],
                    },
                )
            return len(rows)

    # ------------------------------------------------------------------
    # GitHub-specific helpers
    # ------------------------------------------------------------------

    def sync_github_progress(self, run_id: int, progress: GitHubProgress) -> None:
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"run not found: {run_id}")

        query_ids: set[str] = set()
        for index, query in enumerate(progress.queries):
            query_ids.add(str(query.id))
            self.upsert_work_unit(
                run_id=run_id,
                source="github",
                brief_id=run["brief_id"],
                kind=GITHUB_QUERY_KIND,
                source_unit_id=str(query.id),
                display_name=query.name,
                ordering_index=index,
                status=query.status,
                payload=query.to_dict(),
                checkpoint={"hit_result_cap": query.hit_result_cap},
                counters={
                    "result_count": query.result_count,
                    "candidates_discovered": query.candidates_discovered,
                    "saves_count": len(query.saves),
                },
                notes=query.notes,
            )
        self.delete_missing_work_units(run_id, kind=GITHUB_QUERY_KIND, keep_source_unit_ids=query_ids)

        seed_ids: set[str] = set()
        queued_usernames = {entry.get("username", "") for entry in progress.graph_expansion_queue if entry.get("username")}
        processed_usernames = set(progress.graph_expansion_processed or [])
        for index, username in enumerate(sorted(queued_usernames | processed_usernames)):
            if not username:
                continue
            seed_ids.add(username)
            entry = next((item for item in progress.graph_expansion_queue if item.get("username") == username), None) or {
                "username": username,
            }
            status = "done" if username in processed_usernames and username not in queued_usernames else "queued"
            self.upsert_work_unit(
                run_id=run_id,
                source="github",
                brief_id=run["brief_id"],
                kind=GITHUB_GRAPH_SEED_KIND,
                source_unit_id=username,
                display_name=f"Graph expansion seed: {username}",
                ordering_index=len(progress.queries) + index,
                status=status,
                payload=entry,
            )
        self.delete_missing_work_units(run_id, kind=GITHUB_GRAPH_SEED_KIND, keep_source_unit_ids=seed_ids)

        for username in progress.discovered_usernames:
            person_key = f"gh:{username.lower()}"
            with self.connect() as conn:
                existing = conn.execute(
                    """
                    SELECT identity_key
                    FROM candidates
                    WHERE source = 'github' AND brief_id = ? AND person_key = ?
                    """,
                    (run["brief_id"], person_key),
                ).fetchone()
            resolved_identity = str(existing["identity_key"]) if existing else username
            candidate_id = self.ensure_candidate(
                source="github",
                brief_id=run["brief_id"],
                identity_key=resolved_identity,
                display_name=resolved_identity,
                profile_url=f"https://github.com/{resolved_identity}",
                initial_state="failed_terminal",
                person_key=person_key,
            )
            with self.connect() as conn:
                conn.execute(
                    """
                    UPDATE candidates
                    SET current_lifecycle_state = 'failed_terminal',
                        terminal_decision = COALESCE(terminal_decision, 'LEGACY_TERMINAL'),
                        terminal_payload_json = CASE
                            WHEN terminal_payload_json = '{}' THEN ?
                            ELSE terminal_payload_json
                        END,
                        last_seen_at = ?
                    WHERE id = ?
                    """,
                    (
                        _json_dumps({"username": username}),
                        _utc_now(),
                        candidate_id,
                    ),
                )

        self.update_run_resume_state(
            run_id,
            {
                "brief_name": progress.brief_name,
                "candidates_discovered": progress.candidates_discovered,
                "candidates_enriched": progress.candidates_enriched,
                "candidates_saved": progress.candidates_saved,
                "candidates_rejected": progress.candidates_rejected,
                "candidates_insufficient": progress.candidates_insufficient,
                "current_query_id": progress.current_query_id,
                "mined_repos": progress.mined_repos,
                "api_calls_made": progress.api_calls_made,
            },
        )

    def load_github_progress(self, run_id: int) -> GitHubProgress:
        # Lazy import keeps store.py free of a module-level dependency on the
        # GitHub source adapter; GitHubProgress/GitHubSearchQuery are only
        # constructed here at runtime (the method/param annotations are strings
        # under ``from __future__ import annotations``). Mirrors the lazy-import
        # discipline already used for stop-reasons and cloris schema installs.
        from github.schemas import (  # noqa: PLC0415 - lazy by design
            GitHubProgress,
            GitHubSearchQuery,
        )

        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"run not found: {run_id}")
        resume_state = _json_loads(run.get("resume_state_json"))
        query_rows = self.list_work_units(run_id, kind=GITHUB_QUERY_KIND)
        queries: list[GitHubSearchQuery] = []
        for row in query_rows:
            payload = _json_loads(row["payload_json"])
            query_id = _coerce_int(payload.get("id"), fallback=_coerce_int(row["source_unit_id"]))
            query_name = str(
                payload.get("name") or row["display_name"] or f"query-{query_id}"
            ).strip()
            query_text = str(
                payload.get("query")
                or payload.get("boolean")
                or payload.get("search_query")
                or query_name
            ).strip()
            channel = str(payload.get("channel") or "user_search").strip()
            payload.update(
                {
                    "id": query_id,
                    "name": query_name,
                    "query": query_text,
                    "channel": channel,
                    "status": row["status"],
                    "result_count": row["result_count"],
                    "candidates_discovered": row["candidates_discovered"],
                    "notes": row["notes"],
                    "hit_result_cap": _json_loads(row["checkpoint_json"]).get("hit_result_cap", payload.get("hit_result_cap", False)),
                }
            )
            queries.append(GitHubSearchQuery.from_dict(payload))

        graph_rows = self.list_work_units(run_id, kind=GITHUB_GRAPH_SEED_KIND)
        graph_queue = []
        graph_processed = []
        for row in graph_rows:
            payload = _json_loads(row["payload_json"])
            payload.setdefault("username", row["source_unit_id"])
            if row["status"] == "queued":
                graph_queue.append(payload)
            elif row["status"] == "done":
                graph_processed.append(row["source_unit_id"])

        return GitHubProgress(
            brief_name=resume_state.get("brief_name", run["brief_id"]),
            queries=queries,
            candidates_discovered=int(resume_state.get("candidates_discovered", 0)),
            candidates_enriched=int(resume_state.get("candidates_enriched", 0)),
            candidates_saved=int(resume_state.get("candidates_saved", 0)),
            candidates_rejected=int(resume_state.get("candidates_rejected", 0)),
            candidates_insufficient=int(resume_state.get("candidates_insufficient", 0)),
            current_query_id=resume_state.get("current_query_id"),
            discovered_usernames=self.list_terminal_identity_keys(source="github", brief_id=run["brief_id"]),
            mined_repos=list(resume_state.get("mined_repos", [])),
            api_calls_made=int(resume_state.get("api_calls_made", 0)),
            graph_expansion_queue=sorted(graph_queue, key=lambda item: item.get("confidence", 0), reverse=True),
            graph_expansion_processed=sorted(graph_processed),
        )

    def enqueue_graph_expansion_seed(
        self,
        *,
        run_id: int,
        username: str,
        reason: str,
        confidence: float,
        capability_area: str,
    ) -> None:
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"run not found: {run_id}")
        existing = self.get_work_unit_by_source_id(
            run_id,
            kind=GITHUB_GRAPH_SEED_KIND,
            source_unit_id=username,
        )
        payload = {
            "username": username,
            "reason": reason,
            "confidence": confidence,
            "capability_area": capability_area,
            "added_at": _utc_now(),
        }
        ordering_index = len(self.list_work_units(run_id, kind=GITHUB_QUERY_KIND)) + len(
            self.list_work_units(run_id, kind=GITHUB_GRAPH_SEED_KIND)
        )
        status = existing["status"] if existing and existing["status"] == "done" else "queued"
        self.upsert_work_unit(
            run_id=run_id,
            source="github",
            brief_id=run["brief_id"],
            kind=GITHUB_GRAPH_SEED_KIND,
            source_unit_id=username,
            display_name=f"Graph expansion seed: {username}",
            ordering_index=ordering_index,
            status=status,
            payload=payload,
        )

    def list_graph_expansion_seeds(self, run_id: int, *, status: str = "queued") -> list[dict]:
        seeds = self.list_work_units(run_id, kind=GITHUB_GRAPH_SEED_KIND)
        out = []
        for seed in seeds:
            if seed["status"] != status:
                continue
            payload = _json_loads(seed["payload_json"])
            payload.setdefault("username", seed["source_unit_id"])
            payload["_work_unit_id"] = seed["id"]
            out.append(payload)
        return out

    def mark_graph_expansion_seed_processed(self, run_id: int, username: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE work_units
                SET status = 'done', ended_at = ?
                WHERE run_id = ? AND kind = ? AND source_unit_id = ?
                """,
                (_utc_now(), run_id, GITHUB_GRAPH_SEED_KIND, username),
            )

    def get_github_blocked_usernames(self, brief_id: str, usernames: list[str]) -> set[str]:
        if not usernames:
            return set()
        from shared.runtime_state.github import github_person_key

        person_keys = [github_person_key(username) for username in usernames]
        blocked_keys = self.get_blocked_person_keys(brief_id, person_keys)
        return {
            username
            for username, person_key in zip(usernames, person_keys, strict=True)
            if person_key in blocked_keys
        }

    def has_candidates(self, *, source: str, brief_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM candidates
                WHERE source = ? AND brief_id = ?
                LIMIT 1
                """,
                (source, brief_id),
            ).fetchone()
            return row is not None

    def record_event(
        self,
        *,
        event_type: str,
        payload: dict | None = None,
        run_id: int | None = None,
        work_unit_id: int | None = None,
        candidate_id: int | None = None,
        attempt_id: int | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        with self._write_connection(conn) as write_conn:
            self._insert_event(
                write_conn,
                event_type=event_type,
                payload=payload,
                run_id=run_id,
                work_unit_id=work_unit_id,
                candidate_id=candidate_id,
                attempt_id=attempt_id,
            )

    # ------------------------------------------------------------------
    # Recruiter adaptation aggregation (reopen Stage 2 — R5a-4)
    # ------------------------------------------------------------------

    def aggregate_recruiter_adaptations(self) -> list[dict]:
        """Aggregate adaptation decisions per recruiter, PURE READ.

        reopen Stage 2 (R5a-4): the read-only roll-up that lets the
        recruiter-taste distiller (deferred R5a-5) see how a recruiter's
        sourcing adapted across briefs and sources. This method itself
        writes nothing — it only reads ``events`` joined to ``runs`` and
        returns grouped aggregates.

        Mechanics:

        - Join ``events`` to ``runs`` on ``events.run_id = runs.id`` so
          each adaptation event inherits the launching recruiter from
          ``runs.recruiter_id`` (R5a-1/2/3). The join is the ONLY way the
          recruiter reaches an event — events have no recruiter column of
          their own, by design (the run is the recruiter-owned entity).
        - Filter to ``event_type = ADAPTATION_EVENT_TYPE`` and decode each
          ``payload_json`` through
          :meth:`shared.adaptive.AdaptationDecision.from_dict` so the
          source-native fields (``source``, ``lane``, ``work_unit_family``,
          ``action``, markers) are read through the canonical model rather
          than poked at as raw JSON.
        - Group by ``(recruiter_id, source, lane, work_unit_family)``.
          ``work_unit_family`` is a free field on the decision
          (``adaptive.py`` AdaptationDecision.work_unit_family, populated
          from the payload by ``from_dict``); grouping on it costs nothing
          and gives the distiller a finer grain than lane alone.
        - Aggregate ``action_counts`` (a histogram over
          :class:`shared.adaptive.AdaptiveAction` values) plus rolled-up
          ``signal_markers`` / ``noise_markers`` counts by marker kind,
          and a ``decision_count`` total.
        - SKIP any run whose ``recruiter_id IS NULL`` (legacy rows,
          pre-R5a-3 launches): an unknown recruiter must never be folded
          into another recruiter's taste, and the SQL ``WHERE
          runs.recruiter_id IS NOT NULL`` does the skip at the source so a
          large legacy corpus is never even materialized.

        Returns one dict per group, ordered deterministically by
        ``(recruiter_id, source, lane, work_unit_family)`` so callers and
        tests get a stable sequence. The shape is plain ``dict`` to match
        this module's read methods (``list_runs``,
        ``list_incomplete_write_intentions``); the distiller maps these
        into recruiter-store rows in R5a-5.
        """

        # Lazy import to keep store.py free of a module-level dependency on
        # shared.adaptive (mirrors the _stop_reason_helpers lazy-import
        # pattern that breaks the safety<->store import cycle).
        from shared.adaptive import (  # noqa: PLC0415 - lazy by design
            ADAPTATION_EVENT_TYPE,
            AdaptationDecision,
        )

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT runs.recruiter_id AS recruiter_id,
                       events.payload_json AS payload_json
                FROM events
                JOIN runs ON events.run_id = runs.id
                WHERE events.event_type = ?
                  AND runs.recruiter_id IS NOT NULL
                ORDER BY events.id ASC
                """,
                (ADAPTATION_EVENT_TYPE,),
            ).fetchall()

        # grain -> accumulator. Keyed by the 4-tuple; insertion-order
        # preserved by dict, then sorted before return for determinism.
        groups: dict[tuple[int, str, str, str], dict] = {}
        for row in rows:
            recruiter_id = row["recruiter_id"]
            if recruiter_id is None:  # belt-and-suspenders; SQL already filters
                continue
            recruiter_id = int(recruiter_id)
            payload = _json_loads(row["payload_json"])
            if not isinstance(payload, dict):
                continue
            decision = AdaptationDecision.from_dict(payload)

            key = (
                recruiter_id,
                decision.source,
                decision.lane,
                decision.work_unit_family,
            )
            bucket = groups.get(key)
            if bucket is None:
                bucket = {
                    "recruiter_id": recruiter_id,
                    "source": decision.source,
                    "lane": decision.lane,
                    "work_unit_family": decision.work_unit_family,
                    "decision_count": 0,
                    "action_counts": {},
                    "signal_markers": {},
                    "noise_markers": {},
                }
                groups[key] = bucket

            bucket["decision_count"] += 1
            action_value = decision.action.value
            bucket["action_counts"][action_value] = (
                bucket["action_counts"].get(action_value, 0) + 1
            )
            for marker in decision.metrics.signal_markers:
                count = marker.count or 1
                bucket["signal_markers"][marker.kind] = (
                    bucket["signal_markers"].get(marker.kind, 0) + count
                )
            for marker in decision.metrics.noise_markers:
                count = marker.count or 1
                bucket["noise_markers"][marker.kind] = (
                    bucket["noise_markers"].get(marker.kind, 0) + count
                )

        return [groups[key] for key in sorted(groups.keys())]

    # ------------------------------------------------------------------
    # Recruiter write intentions (reopen Stage 2 — committed-intent ledger)
    # ------------------------------------------------------------------
    #
    # The double-write from a brief-scoped surface (designer principle
    # feedback, market-intel archetype preference) into the GLOBAL
    # recruiter store is fail-soft: the recruiter store can be momentarily
    # unavailable (locked, read-only FS) without failing the recruiter's
    # primary action. But a silently-dropped taste signal is
    # undetectable/unrecoverable — the global store has no per-brief view
    # to reconcile against (adversarial-ledger flaw "consistency", major).
    #
    # The fix: a committed write-intentions ledger on THIS per-state-dir
    # DB. The sequence at every double-write site is:
    #   1. record_write_intention(...)        — must succeed (committed here)
    #   2. fail-soft recruiter_store.record_taste_signal(...)
    #   3. on success, mark_write_intention_complete(intention_id)
    # An idempotent backfill replays incomplete intentions
    # (``completed_at IS NULL``) into the recruiter store later. The
    # ``dedup_key`` makes both the record and the replay idempotent.

    def record_write_intention(
        self,
        *,
        signal_kind: str,
        domain: str,
        dedup_key: str,
        payload: dict | None = None,
        source_brief_id: str | None = None,
        confidence: float = 0.5,
    ) -> int:
        """Commit the intent to write a recruiter taste signal; return its id.

        Idempotent on ``dedup_key``: re-recording the same logical
        intention (e.g. the same recruiter marking the same principle on
        the same candidate) returns the existing row's id and refreshes
        the payload rather than inserting a duplicate. This is what makes
        the backfill replay safe to run repeatedly.

        Raises on a genuine DB failure — this is the "must succeed" step;
        a caller that can't even commit the intention should surface that,
        not proceed to a fail-soft recruiter write whose loss would then
        be unrecoverable.
        """

        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO recruiter_write_intentions(
                    signal_kind, domain, source_brief_id, payload_json,
                    confidence, dedup_key, completed_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(dedup_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    confidence = excluded.confidence
                """,
                (
                    signal_kind,
                    domain,
                    source_brief_id,
                    _json_dumps(payload or {}),
                    confidence,
                    dedup_key,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT id FROM recruiter_write_intentions WHERE dedup_key = ?",
                (dedup_key,),
            ).fetchone()
        self._bump_heartbeat()
        return int(row["id"])

    def mark_write_intention_complete(self, intention_id: int) -> None:
        """Mark an intention as durably mirrored to the recruiter store.

        Idempotent: marking an already-complete intention is a no-op-ish
        timestamp refresh. Called only after the fail-soft recruiter
        write succeeded; an intention left with ``completed_at IS NULL``
        is exactly what the backfill replays.
        """

        with self.connect() as conn:
            conn.execute(
                "UPDATE recruiter_write_intentions SET completed_at = ? WHERE id = ?",
                (_utc_now(), intention_id),
            )

    def list_incomplete_write_intentions(self) -> list[dict]:
        """Return every intention not yet mirrored to the recruiter store.

        The backfill replay reads these, writes each to
        ``recruiter_taste_signals``, then calls
        :meth:`mark_write_intention_complete`. ``payload`` is decoded for
        the caller (mirrors :meth:`active_taste_signals` on the recruiter
        store).
        """

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM recruiter_write_intentions
                WHERE completed_at IS NULL
                ORDER BY id ASC
                """
            ).fetchall()
        out: list[dict] = []
        for row in rows:
            record = _row_to_dict(row) or {}
            record["payload"] = _json_loads(record.pop("payload_json", None))
            out.append(record)
        return out

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _insert_event(
        self,
        conn: sqlite3.Connection,
        *,
        event_type: str,
        payload: dict | None = None,
        run_id: int | None = None,
        work_unit_id: int | None = None,
        candidate_id: int | None = None,
        attempt_id: int | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO events(run_id, work_unit_id, candidate_id, attempt_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                work_unit_id,
                candidate_id,
                attempt_id,
                event_type,
                _json_dumps(payload or {}),
                _utc_now(),
            ),
        )


def _ensure_schema_migrations_table(conn: sqlite3.Connection) -> None:
    # Ledger records migration steps (ALTERs / normalizations) applied after the
    # initial CREATE on this DB — NOT a feature inventory; a fresh DB whose
    # CREATE already includes all columns records only the normalization step.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT NOT NULL PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )


def _record_schema_migration(conn: sqlite3.Connection, version: str) -> None:
    existing = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE version = ?",
        (version,),
    ).fetchone()
    if existing is not None:
        return
    conn.execute(
        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (version, _utc_now()),
    )


def _add_column_if_missing(conn: sqlite3.Connection, ddl: str) -> None:
    """Execute ``ddl`` (an ALTER TABLE ADD COLUMN), tolerating "duplicate
    column name" errors so concurrent migrations from two RuntimeStateStore
    constructions don't crash one another.

    Per critique A2 (preserve-and-migrate plan, Phase 1.5): SQLite's WAL
    serializes the writes but two concurrent ALTERs can produce
    ``OperationalError: duplicate column name``. Treating that as success
    is correct — the column is now there, which is what we wanted.
    """

    try:
        conn.execute(ddl)
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def _execute_with_lock_retry(
    conn: sqlite3.Connection,
    sql: str,
    *,
    timeout_seconds: float = 5.0,
) -> None:
    """Execute a setup statement, retrying transient SQLite lock races."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            conn.execute(sql)
            return
        except sqlite3.OperationalError as exc:
            if (
                "database is locked" not in str(exc).lower()
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(0.05)


def _read_schema_version(conn: sqlite3.Connection) -> str | None:
    """Return the meta.schema_version value, or ``None`` if absent."""

    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    return None if row is None else str(row["value"])


def _version_lt(prior: str | None, target: str) -> bool:
    """Return True iff ``prior`` represents a schema version earlier than
    ``target`` (lexicographic compare on simple integer-like version strings).

    A missing prior is treated as the earliest version: a fresh DB will
    take the same normalization path so on-disk shape is consistent
    regardless of how the DB was created (initialize from scratch vs
    migrated from a prior version).
    """

    if prior is None:
        return True
    try:
        return int(prior) < int(target)
    except ValueError:
        # Unknown version format — be conservative and treat as "needs
        # migration." The migration is idempotent so re-running is safe.
        return True


def _normalize_legacy_stop_reasons(conn: sqlite3.Connection) -> None:
    """Walk runs once, normalize any stop_reason value that isn't in the
    canonical enum, and stash the original in stop_reason_detail.

    Idempotent against already-normalized rows: a row whose stop_reason is
    already canonical and whose stop_reason_detail is NULL is left
    unchanged.
    """

    known, normalize = _stop_reason_helpers()
    rows = conn.execute("SELECT id, stop_reason FROM runs").fetchall()
    for row in rows:
        original = row["stop_reason"]
        if original in known:
            continue
        canonical = normalize(original)
        # Preserve the unrecognized original so future tooling can still
        # see what the orchestrator actually wrote.
        conn.execute(
            "UPDATE runs SET stop_reason = ?, "
            "stop_reason_detail = COALESCE(stop_reason_detail, ?) "
            "WHERE id = ?",
            (canonical, original, int(row["id"])),
        )


def _split_stop_reason(stop_reason: str | None) -> tuple[str, str | None]:
    """Return ``(canonical, detail)`` where canonical is the enum value
    and detail is the original string (only when it differs).

    The canonical value goes to ``runs.stop_reason``; the detail goes to
    ``runs.stop_reason_detail``. When the caller already passed a
    canonical value, detail is ``None`` so we don't pollute the column
    with redundant copies.
    """

    _, normalize = _stop_reason_helpers()
    canonical = normalize(stop_reason)
    if stop_reason is None or stop_reason == canonical:
        return canonical, None
    return canonical, stop_reason


_REENTRY_STAGE_STATES = frozenset({"snippet_extracted", "facial_started", "full_started"})


def _guard_save_family_reentry(
    current_state: str, terminal_decision: object, new_state: str
) -> None:
    """A save-family terminal is never re-entered — decision-aware belt.

    Stage-complete re-entry (2026-07-28) opened every started stage from the
    judgment terminals so RELEASED candidates re-enter wherever the pipeline
    meets them. Release only ever applies to non-SAVE verdicts (the hash-aware
    clause suppresses SAVE-family regardless of brief), so a saved candidate
    reaching a re-entry write means suppression failed upstream — refuse here
    rather than let a save be silently re-evaluated and its terminal_decision
    overwritten (the double-save door). The one sanctioned exception, retrying
    a FAILED save side-effect, bypasses this in set_candidate_state only.
    """
    if (
        current_state in ("facial_terminal", "full_terminal")
        and terminal_decision in SAVE_FAMILY_DECISIONS
        and new_state in _REENTRY_STAGE_STATES
    ):
        raise ValueError(
            f"invalid lifecycle transition: {current_state} -> {new_state} "
            f"(save-family terminal_decision {terminal_decision!r} is never "
            "re-entered; suppression should have blocked this candidate)"
        )


def _guard_transition(current_state: str, new_state: str) -> None:
    if current_state == new_state:
        return
    allowed = ALLOWED_LIFECYCLE_TRANSITIONS.get(current_state)
    if allowed is None:
        raise ValueError(f"unknown lifecycle state: {current_state}")
    if new_state not in allowed:
        raise ValueError(f"invalid lifecycle transition: {current_state} -> {new_state}")


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _json_dumps(data: Any) -> str:
    if is_dataclass(data):
        data = asdict(data)
    return json.dumps(data or {}, sort_keys=True)


def _json_loads(raw: str | bytes | None) -> Any:
    if not raw:
        return {}
    return json.loads(raw)


def _event_observability_fields(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}

    observability: dict[str, Any] = {}
    prompt_capture = payload.get("prompt_capture")
    if isinstance(prompt_capture, dict):
        observability.update(prompt_capture)
    compact_observability = payload.get("observability")
    if isinstance(compact_observability, dict):
        observability.update(compact_observability)

    out: dict[str, str] = {}
    for key in ("trace_id", "observation_id"):
        value = observability.get(key)
        if isinstance(value, str) and value:
            out[key] = value
    return out


def _coerce_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
