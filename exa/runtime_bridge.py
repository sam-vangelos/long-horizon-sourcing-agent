"""Exa runtime-state bridge.

Maps :class:`~exa.schemas.ExaProgress` to generic
:class:`shared.runtime_state.store.RuntimeStateStore` rows in both directions
and writes the compatibility ``progress.json`` projection. The store owns run,
work-unit, candidate-lifecycle, and dedupe state; this module owns Exa's
translation and plan reconciliation.

A resumed run rebuilds progress from canonical work-unit rows, never from the
projection file. SQLite is authoritative when the two planes disagree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from exa.schemas import ExaProgress, ExaSearchQuery
from shared.runtime_state.store import EXA_QUERY_KIND, RuntimeStateStore


SOURCE = "exa"

# Lifecycle states from which a person may still be judged. Other states are
# terminal or reconciled; ``full_started`` remains accepted if an interrupted
# row survives ``RuntimeStateStore.reconcile_open_attempts``.
PENDING_CANDIDATE_STATES = frozenset(
    {"discovered", "snippet_extracted", "failed_retryable", "full_started"}
)


@dataclass
class CampaignCandidate:
    """One person as the CAMPAIGN holds them, not as a session saw them."""

    candidate_id: int
    identity_key: str
    person_key: str = ""
    display_name: str = ""
    profile_url: str = ""
    state: str = ""
    terminal_decision: str = ""
    record: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    last_attempt_id: int | None = None
    last_work_unit_id: int | None = None
    # A recruiter's recorded decision, which no automated pass overrules.
    user_status: str = ""
    # The brief content hash of the run that made the terminal verdict, and
    # the judging-criteria fingerprint stamped on it; the re-judge pass
    # compares both with the current brief (``exa/rejudge.py``).
    brief_content_hash: str = ""
    judgment_fingerprint: str = ""
    # A re-judge whose two samples split (one save-type, one reject): both
    # samples, under ``exa.rejudge.REJUDGE_SPLIT_KEY``. The stored decision
    # is the save-type sample; this is what puts the person on worth-a-look.
    rejudge_split: dict = field(default_factory=dict)

    @property
    def is_pending(self) -> bool:
        return self.state in PENDING_CANDIDATE_STATES

    @property
    def is_terminal(self) -> bool:
        return bool(self.terminal_decision) and self.state == "full_terminal"

# Work-unit statuses a resumed run still has to execute. ``in_progress``
# is included deliberately: it means a previous session died mid-query,
# and the campaign-keeps-itself-alive posture re-runs it rather than
# silently dropping whatever that query would have found. Re-running is
# safe because every person it re-discovers is filtered by the seen-key
# and blocked-person-key checks before any model spend.
RUNNABLE_QUERY_STATUSES = frozenset({"queued", "in_progress"})

# The note a lane-dropped query carries, and the marker that lets a
# later brief take it back. Written into the work unit's ``notes``
# column, so it survives the clone into every subsequent session and a
# human reading the row sees WHY the query stopped rather than an
# unexplained ``skipped``.
LANE_DROPPED_NOTE = "lane no longer in brief"

# The note on a query the brief still has a lane for but no longer
# compiles — a firm deleted, an archetype retired.
PLAN_DROPPED_NOTE = "no longer in the brief's plan"

# The note on a query whose stored payload could not be read and which
# the current plan cannot rebuild. Distinct from the two above because
# it is a DEFECT, not an edit: the query is lost, not retired.
PAYLOAD_LOST_NOTE = "payload lost; not in current plan"

# The notes a requeue is allowed to lift. A query skipped for any other
# reason stays skipped — a marker written by a future rule must not be
# swept up by this one.
REQUEUABLE_NOTES = frozenset({LANE_DROPPED_NOTE, PLAN_DROPPED_NOTE})

# How many query names a plan-revision console line spells out before it
# summarises the rest. The console rule is one line per event; six
# archetype names fit, twenty firm names do not.
PLAN_LINE_NAME_LIMIT = 4


def _one_line(text: str) -> str:
    """Collapse console text to one line while preserving leading indentation.

    This local copy keeps the bridge independent of acquisition imports.
    """

    raw = str(text)
    indent = raw[: len(raw) - len(raw.lstrip(" "))]
    return indent + " ".join(raw.replace("\r", " ").replace("\n", " ").split())


def _queries_word(count: int) -> str:
    return "query" if count == 1 else "queries"


def _name_list(queries: list[ExaSearchQuery]) -> str:
    """The names, truncated to one console line's worth."""

    names = [q.name or f"query {q.id}" for q in queries]
    head = ", ".join(names[:PLAN_LINE_NAME_LIMIT])
    extra = len(names) - PLAN_LINE_NAME_LIMIT
    return f"{head} (+{extra} more)" if extra > 0 else head


@dataclass
class PlanMerge:
    """What reconciling a stored plan against a revised brief changed."""

    added: list[ExaSearchQuery] = field(default_factory=list)
    # Skipped because the LANE left the brief, and skipped because the
    # lane stayed but this query left it (a firm deleted, an archetype
    # retired). Two lists rather than one because they are two different
    # edits and the operator reads them as two different sentences.
    skipped: list[ExaSearchQuery] = field(default_factory=list)
    dropped: list[ExaSearchQuery] = field(default_factory=list)
    requeued: list[ExaSearchQuery] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.skipped or self.dropped or self.requeued)

    def console_lines(self) -> list[str]:
        lines: list[str] = []
        if self.added:
            lines.append(
                _one_line(
                    f"    [exa] plan revised — {len(self.added)} new "
                    f"{_queries_word(len(self.added))} added: "
                    f"{_name_list(self.added)}"
                )
            )
        if self.skipped:
            lines.append(
                _one_line(
                    f"    [exa] plan revised — {len(self.skipped)} "
                    f"{_queries_word(len(self.skipped))} skipped "
                    f"(lane dropped from the brief)"
                )
            )
        if self.dropped:
            lines.append(
                _one_line(
                    f"    [exa] plan revised — {len(self.dropped)} "
                    f"{_queries_word(len(self.dropped))} skipped "
                    f"(no longer in the brief's plan): {_name_list(self.dropped)}"
                )
            )
        if self.requeued:
            lines.append(
                _one_line(
                    f"    [exa] plan revised — {len(self.requeued)} "
                    f"{_queries_word(len(self.requeued))} requeued "
                    f"(lane back in the brief)"
                )
            )
        return lines


@dataclass
class PayloadRecovery:
    """What an unreadable work-unit payload cost, and what was recovered."""

    rebuilt: list[ExaSearchQuery] = field(default_factory=list)
    skipped: list[ExaSearchQuery] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.rebuilt or self.skipped)

    def console_lines(self) -> list[str]:
        lines: list[str] = []
        if self.rebuilt:
            lines.append(
                _one_line(
                    f"    [exa] {len(self.rebuilt)} stored "
                    f"{_queries_word(len(self.rebuilt))} lost their payload — "
                    f"rebuilt from the brief: {_name_list(self.rebuilt)}"
                )
            )
        if self.skipped:
            lines.append(
                _one_line(
                    f"    [exa] {len(self.skipped)} stored "
                    f"{_queries_word(len(self.skipped))} skipped — payload lost "
                    f"and not in the current plan: {_name_list(self.skipped)}"
                )
            )
        return lines


def _degraded_query(row: dict) -> ExaSearchQuery:
    """The identity a row still carries when its payload is unreadable.

    The family key is a firm on the firm lanes and ``archetype:<id>`` on
    the archetype lane. Reading it back as a firm on both would put
    "archetype:a01-…" into an archetype query's employer field — and,
    worse, would give the row a ``coverage_key`` the plan cannot match,
    so the rebuild would miss and a recoverable query would be skipped.
    """

    lane = str(row.get("domain_lane") or "")
    family_key = str(row.get("family_key") or "")
    archetype_id = ""
    if lane == "archetype" and family_key.startswith("archetype:"):
        archetype_id = family_key.split(":", 1)[1]
        family_key = ""
    return ExaSearchQuery(
        id=_int_or(row.get("source_unit_id"), 0),
        name=str(row.get("display_name") or ""),
        lane=lane,
        firm=family_key,
        archetype_id=archetype_id,
        query="",
    )


def _resolve_recruiter_id() -> int | None:
    """Resolve the acting recruiter for a run-start, fail-soft to None.

    Same contract as the github/designer bridges: the resolver is the
    single auth seam, and a launch must never die on recruiter
    resolution — a None recruiter_id is a clean "unknown".
    """

    try:
        from shared.recruiter_context import get_current_recruiter_id

        return get_current_recruiter_id()
    except Exception:  # noqa: BLE001 — resolution must never break a run launch
        return None


def _query_counters(query: ExaSearchQuery) -> dict[str, int]:
    """The three counters an Exa query row carries.

    ``upsert_work_unit`` zeroes every counter it is not given, so the
    dict is built in one place rather than at each call site — a caller
    that forgot ``saves_count`` would silently reset the saves column on
    the next sync.
    """

    return {
        "result_count": int(query.result_count or 0),
        "candidates_discovered": int(query.candidates_discovered or 0),
        "saves_count": len(query.saves or []),
    }


class ExaRuntimeStateBridge:
    """Keeps Exa progress/resume semantics DB-authoritative."""

    def __init__(
        self,
        *,
        store: RuntimeStateStore,
        output_dir: str | Path,
        brief_id: str,
        brief_name: str,
        brief_path: str | None = None,
    ) -> None:
        self.store = store
        self.output_dir = Path(output_dir)
        self.brief_id = brief_id
        self.brief_name = brief_name
        self.brief_path = brief_path
        self.progress_path = self.output_dir / "progress.json"

    # ── Runs ─────────────────────────────────────────────────────────

    def has_runtime_state(self) -> bool:
        latest_run = self.store.get_latest_run(source=SOURCE, brief_id=self.brief_id)
        return bool(
            latest_run
            or self.store.has_candidates(source=SOURCE, brief_id=self.brief_id)
        )

    def start_or_resume_run(
        self,
        *,
        resume: bool,
        initial_progress: ExaProgress | None = None,
        active_lanes: Iterable[str] | None = None,
    ) -> tuple[int, ExaProgress]:
        """Open a run row and return ``(run_id, progress)`` to execute.

        Resume owns a clone of the previous run's work-unit rows. The current
        brief remains the authority for the compiled plan; ``merge_plan``
        reconciles the clone against it so added or removed lanes take effect.
        ``active_lanes`` distinguishes a dropped lane from a lane with no
        compiled queries. Omitting ``active_lanes`` or passing
        ``initial_progress=None`` skips reconciliation for callers without a
        brief.
        """
        self.store.reconcile_open_attempts(source=SOURCE, brief_id=self.brief_id)
        self.store.reconcile_pending_side_effects(
            source=SOURCE, brief_id=self.brief_id
        )
        latest_run = self.store.get_latest_run(source=SOURCE, brief_id=self.brief_id)

        from shared.brief_identity import compute_brief_identity

        identity = (
            compute_brief_identity(self.brief_path) if self.brief_path else None
        )
        identity_kwargs: dict[str, Any] = {}
        if identity is not None:
            identity_kwargs = {
                "brief_path_at_launch": identity["brief_path_at_launch"],
                "brief_content_hash": identity["brief_content_hash"],
                "brief_snapshot_json": identity["brief_snapshot_json"],
            }

        recruiter_id = _resolve_recruiter_id()

        if resume and latest_run and self.store.has_work_units(int(latest_run["id"])):
            run_id = self.store.start_run(
                source=SOURCE,
                brief_id=self.brief_id,
                output_dir=str(self.output_dir),
                mode="resume",
                resume_state=self.store.get_run_resume_state(int(latest_run["id"])),
                resumed_from_run_id=int(latest_run["id"]),
                clone_work_units_from_run_id=int(latest_run["id"]),
                recruiter_id=recruiter_id,
                **identity_kwargs,
            )
            progress, recovery = self.load_progress_with_recovery(
                run_id, compiled=initial_progress
            )
            for line in recovery.console_lines():
                print(line, flush=True)
            merge = self.merge_plan(
                progress,
                compiled=initial_progress,
                active_lanes=active_lanes,
            )
            for line in merge.console_lines():
                print(line, flush=True)
            if merge.changed or recovery.changed:
                # Sync only on a real revision. An unchanged resume must
                # leave the rows exactly as the clone wrote them, and the
                # drops ride ``metrics_json``, which ``upsert_work_unit``
                # OVERWRITES — so the merge sync has to carry the cloned
                # metrics back in or a resumed campaign's coverage notes
                # reset to "no drops" the moment a brief is edited.
                self.sync_progress(
                    run_id, progress, metrics_by_query=self.load_query_drops(run_id)
                )
            else:
                self._write_progress_projection(progress)
            return run_id, progress

        run_id = self.store.start_run(
            source=SOURCE,
            brief_id=self.brief_id,
            output_dir=str(self.output_dir),
            mode="resume" if resume else "fresh",
            resume_state={"brief_name": self.brief_name},
            resumed_from_run_id=(
                int(latest_run["id"]) if resume and latest_run else None
            ),
            recruiter_id=recruiter_id,
            **identity_kwargs,
        )

        progress = initial_progress or ExaProgress(brief_name=self.brief_name)
        # A fresh run over a state dir that already holds terminal
        # verdicts still must not re-judge those people: the run row is
        # new, the candidate rows are not.
        progress.discovered_identity_keys = self._terminal_identity_keys()
        self.sync_progress(run_id, progress)
        return run_id, progress

    # ── Plan merge ───────────────────────────────────────────────────

    def merge_plan(
        self,
        stored: ExaProgress,
        *,
        compiled: ExaProgress | None,
        active_lanes: Iterable[str] | None = None,
    ) -> PlanMerge:
        """Reconcile a resumed plan against the brief, in place.

        The brief is the authority on WHAT runs; the store is the
        authority on what already ran. Three reconciliations, in that
        order of consequence:

        1. **A query the brief added** has no stored counterpart, so it
           is appended ``queued``, with an id continuing after the
           highest stored id. Continuing rather than reusing the
           compiled id is the point: compiled ids are POSITIONAL
           (``compile_exa_queries`` numbers from 1 down the lane list),
           so dropping a lane renumbers everything after it, and a
           merge that trusted those ids would hand an archetype query
           the id of a firm query that already ran.
        2. **A query the brief no longer compiles** is marked
           ``skipped``, never deleted. Deleting it would take its result
           counts and its drops out of the campaign's coverage — the
           session that ran it did real work, and the slate has to keep
           saying so. Only a RUNNABLE query is touched; a ``done`` one
           keeps its verdict. The note records WHICH edit retired it:
           :data:`LANE_DROPPED_NOTE` when the whole lane left the brief,
           :data:`PLAN_DROPPED_NOTE` when the lane stayed and this query
           did not (a firm deleted, an archetype retired). Both are
           edits an operator makes deliberately, and both must stop the
           spend they retired — a lane removed on Tuesday whose stored
           rows keep running is the bug this whole merge exists for.
        3. **A query the brief takes back** requeues, matched on those
           notes, so a query skipped for any future reason is not swept
           up by it.

        What it deliberately does NOT do: re-run a query whose text the
        brief edited. A firm renamed, a title term rewritten, a
        ``num_results`` raised — the stored query keeps running as
        stored, silently. Text edits arrive constantly while a campaign
        is tuned, and treating each as a new query would re-pay the Exa
        and Opus cost of everything already swept every time someone
        fixed a typo. A genuinely new search is expressed as a new firm,
        a new lane, or a new archetype, all three of which this merge
        does pick up.

        Matching is on ``(lane, coverage_key)``, never on id. Where that
        pair is not unique — the firm-less fallback compiles one
        people-search query per title term, all with an empty firm — the
        matches are consumed in order, which degrades to positional
        pairing for exactly the case where nothing better exists.
        """

        merge = PlanMerge()
        if compiled is None:
            return merge

        by_key: dict[tuple[str, str], list[ExaSearchQuery]] = {}
        for query in stored.queries:
            by_key.setdefault((query.lane, query.coverage_key), []).append(query)

        matched: set[int] = set()
        next_id = max((int(q.id) for q in stored.queries), default=0) + 1
        for query in compiled.queries:
            bucket = by_key.get((query.lane, query.coverage_key))
            if bucket:
                matched.add(id(bucket.pop(0)))
                continue
            query.id = next_id
            next_id += 1
            query.status = "queued"
            merge.added.append(query)

        lanes = None if active_lanes is None else set(active_lanes)
        for query in stored.queries:
            in_plan = id(query) in matched
            if in_plan:
                if query.status == "skipped" and query.notes in REQUEUABLE_NOTES:
                    query.status = "queued"
                    query.notes = ""
                    merge.requeued.append(query)
                continue
            if query.status not in RUNNABLE_QUERY_STATUSES:
                continue
            lane_gone = lanes is not None and query.lane not in lanes
            query.status = "skipped"
            query.notes = LANE_DROPPED_NOTE if lane_gone else PLAN_DROPPED_NOTE
            (merge.skipped if lane_gone else merge.dropped).append(query)

        stored.queries.extend(merge.added)
        return merge

    def load_query_drops(self, run_id: int) -> dict[str, dict]:
        """Per-query acquisition drop counts, as this run's rows hold them.

        The reader half of the ``metrics_json`` mapping ``sync_progress``
        writes. It lives here rather than in the orchestrator because
        this bridge owns that translation in both directions, and
        because any writer that syncs without it silently zeroes the
        column.
        """

        drops: dict[str, dict] = {}
        for row in self.store.list_work_units(run_id, kind=EXA_QUERY_KIND):
            recorded = _json_object(row.get("metrics_json")).get("drops")
            if isinstance(recorded, dict) and recorded:
                drops[str(row["source_unit_id"])] = {
                    str(key): int(value) for key, value in recorded.items()
                }
        return drops

    # ── Progress ─────────────────────────────────────────────────────

    def sync_progress(
        self,
        run_id: int,
        progress: ExaProgress,
        *,
        metrics_by_query: dict[str, dict] | None = None,
    ) -> None:
        """Write query rows and project the current progress file.

        ``metrics_by_query`` stores acquisition drop counts in ``metrics_json``;
        those metrics are cloned with work units so coverage notes survive a
        resumed run.
        """

        metrics_by_query = metrics_by_query or {}
        keep: set[str] = set()
        for index, query in enumerate(progress.queries):
            source_unit_id = str(query.id)
            keep.add(source_unit_id)
            drops = metrics_by_query.get(source_unit_id) or {}
            self.store.upsert_work_unit(
                run_id=run_id,
                source=SOURCE,
                brief_id=self.brief_id,
                kind=EXA_QUERY_KIND,
                source_unit_id=source_unit_id,
                display_name=query.name,
                ordering_index=index,
                status=query.status,
                # ``coverage_key`` is a derived property, so it is
                # written in EXPLICITLY: it is the query's stable
                # identity (the id is an ordinal), and a row that has to
                # be read back by a version that computes it differently
                # should be able to say what it was keyed on when it
                # was written.
                payload={**query.to_dict(), "coverage_key": query.coverage_key},
                checkpoint={
                    "lane": query.lane,
                    "firm": query.firm,
                    "coverage_key": query.coverage_key,
                },
                metrics={"drops": dict(drops)} if drops else {},
                # NOT `query.firm`: an archetype query has no firm, and
                # keying six of them on "" reported six hypotheses as
                # one anonymous coverage row.
                family_key=query.coverage_key,
                domain_lane=query.lane,
                counters=_query_counters(query),
                notes=query.notes,
            )
        self.store.delete_missing_work_units(
            run_id, kind=EXA_QUERY_KIND, keep_source_unit_ids=keep
        )
        self._write_progress_projection(progress)

    def load_progress(
        self, run_id: int, *, compiled: ExaProgress | None = None
    ) -> ExaProgress:
        """Rebuild an :class:`ExaProgress` from this run's canonical rows."""

        progress, _ = self.load_progress_with_recovery(run_id, compiled=compiled)
        return progress

    def load_progress_with_recovery(
        self, run_id: int, *, compiled: ExaProgress | None = None
    ) -> tuple[ExaProgress, PayloadRecovery]:
        """Rebuild progress with payload recovery and report what was lost.

        The payload carries the compiled query; the row carries status and
        counters, and the row wins on both. Repair unreadable payloads before
        applying the fail-closed fallback:

        * match the current plan on ``(lane, coverage_key)`` and rebuild from
          its deterministic compiled query;
        * otherwise mark a still-runnable row ``skipped`` with
          :data:`PAYLOAD_LOST_NOTE`, while preserving exact ``done`` rows and
          their verdicts/counters.
        """
        by_key: dict[tuple[str, str], list[ExaSearchQuery]] = {}
        for query in (compiled.queries if compiled else []):
            by_key.setdefault((query.lane, query.coverage_key), []).append(query)

        recovery = PayloadRecovery()
        queries: list[ExaSearchQuery] = []
        for row in self.store.list_work_units(run_id, kind=EXA_QUERY_KIND):
            payload = _row_payload(row)
            query = ExaSearchQuery.from_dict(payload) if payload else None
            lost = query is None
            if lost:
                query = _degraded_query(row)
                match = by_key.get((query.lane, query.coverage_key))
                if match:
                    rebuilt = ExaSearchQuery.from_dict(match[0].to_dict())
                    rebuilt.id = query.id
                    rebuilt.name = query.name or rebuilt.name
                    query = rebuilt
                    recovery.rebuilt.append(query)
            query.status = str(row.get("status") or "queued")
            query.result_count = int(row.get("result_count") or 0)
            query.candidates_discovered = int(row.get("candidates_discovered") or 0)
            query.notes = str(row.get("notes") or "")
            if lost and not query.query and query.status in RUNNABLE_QUERY_STATUSES:
                query.status = "skipped"
                query.notes = PAYLOAD_LOST_NOTE
                recovery.skipped.append(query)
            queries.append(query)

        progress = ExaProgress(brief_name=self.brief_name, queries=queries)
        progress.discovered_identity_keys = self._terminal_identity_keys()
        progress.candidates_discovered = sum(q.candidates_discovered for q in queries)
        progress.candidates_saved = sum(len(q.saves or []) for q in queries)
        return progress, recovery

    def _terminal_identity_keys(self) -> list[str]:
        return list(
            self.store.list_terminal_identity_keys(
                source=SOURCE, brief_id=self.brief_id
            )
        )

    def _write_progress_projection(self, progress: ExaProgress) -> None:
        """Write ``progress.json`` as a non-authoritative projection.

        Exa has no generic projection dispatch in
        ``shared/runtime_state/admin.py:rebuild_compat_projections``; projection
        failure is fail-soft after canonical state is written.
        """

        try:
            progress.save(str(self.progress_path))
        except OSError:
            pass

    # ── The campaign plane ───────────────────────────────────────────

    def list_campaign_candidates(self) -> list[CampaignCandidate]:
        """Every person this BRIEF has ever met, across every session.

        Read-only SELECTs on ``candidates`` and ``candidate_attempts``,
        the same posture ``exa/known.py`` takes toward a sibling
        campaign, and for the same reason: the store exposes no read that
        returns candidate rows for a (source, brief_id) pair —
        ``list_terminal_identity_keys`` returns bare keys and
        ``get_candidate`` needs a key you already have. Adding a store
        method for one module's report would put Exa's shape inside a
        high-risk shared file, so the query lives here. Two statements,
        no writes; the run holds the state dir's advisory lock.

        The Exa record itself is carried in the payload columns rather
        than re-derived: ``terminal_payload_json`` for a judged person
        (``finish_stage_success`` stores ``candidate_record`` there), and
        the newest attempt payload that has one for everybody else. That
        is what lets a resumed session re-judge a pending candidate with
        no Exa call — the evidence is already in the database.
        """

        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, identity_key, person_key, display_name, profile_url,
                       current_lifecycle_state, terminal_decision,
                       terminal_payload_json, last_attempt_id, last_work_unit_id,
                       user_status, brief_content_hash
                FROM candidates
                WHERE source = ? AND brief_id = ?
                ORDER BY id ASC
                """,
                (SOURCE, self.brief_id),
            ).fetchall()
            if not rows:
                return []
            attempts = conn.execute(
                """
                SELECT ca.candidate_id, ca.payload_json
                FROM candidate_attempts ca
                JOIN candidates c ON c.id = ca.candidate_id
                WHERE c.source = ? AND c.brief_id = ?
                ORDER BY ca.id ASC
                """,
                (SOURCE, self.brief_id),
            ).fetchall()

        record_by_candidate: dict[int, dict] = {}
        for row in attempts:
            payload = _json_object(row["payload_json"])
            record = payload.get("candidate_record")
            if isinstance(record, dict) and record:
                record_by_candidate[int(row["candidate_id"])] = record

        out: list[CampaignCandidate] = []
        for row in rows:
            terminal_payload = _json_object(row["terminal_payload_json"])
            record = terminal_payload.get("candidate_record")
            if not isinstance(record, dict) or not record:
                record = record_by_candidate.get(int(row["id"]), {})
            decision = terminal_payload.get("full_decision")
            out.append(
                CampaignCandidate(
                    candidate_id=int(row["id"]),
                    identity_key=str(row["identity_key"] or ""),
                    person_key=str(row["person_key"] or ""),
                    display_name=str(row["display_name"] or ""),
                    profile_url=str(row["profile_url"] or ""),
                    state=str(row["current_lifecycle_state"] or ""),
                    terminal_decision=str(row["terminal_decision"] or ""),
                    record=dict(record),
                    decision=dict(decision) if isinstance(decision, dict) else {},
                    last_attempt_id=(
                        int(row["last_attempt_id"]) if row["last_attempt_id"] else None
                    ),
                    last_work_unit_id=(
                        int(row["last_work_unit_id"])
                        if row["last_work_unit_id"]
                        else None
                    ),
                    user_status=str(row["user_status"] or ""),
                    brief_content_hash=str(row["brief_content_hash"] or ""),
                    # exa.judge.JUDGMENT_FINGERPRINT_KEY, spelled out so the
                    # bridge does not import the judge.
                    judgment_fingerprint=str(
                        terminal_payload.get("judgment_fingerprint") or ""
                    ),
                    # exa.rejudge.REJUDGE_SPLIT_KEY, spelled out for the same reason.
                    rejudge_split=_json_object(terminal_payload.get("rejudge_split")),
                )
            )
        return out

    def list_event_payloads(self, event_type: str) -> list[dict]:
        """Every payload of one event type this brief's runs recorded, oldest first.

        Read-only, like :meth:`list_campaign_candidates`. Scoped through the
        run row's ``(source, brief_id)``, since the event row carries only
        the run id.
        """

        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT e.payload_json
                FROM events e
                JOIN runs r ON r.id = e.run_id
                WHERE e.event_type = ? AND r.source = ? AND r.brief_id = ?
                ORDER BY e.id ASC
                """,
                (event_type, SOURCE, self.brief_id),
            ).fetchall()
        return [_json_object(row["payload_json"]) for row in rows]

    def succeeded_side_effect_keys(self, effect_type: str) -> set[str]:
        """Identity keys whose side effect of this type actually landed.

        ``list_candidate_side_effects`` is an existing store read; the
        status filter is the point — a ``pending`` row that a crash
        turned into ``interrupted`` is NOT a completed save, and the
        replay pass has to be able to tell.
        """

        return {
            str(row["identity_key"])
            for row in self.store.list_candidate_side_effects(
                source=SOURCE, brief_id=self.brief_id, status="succeeded"
            )
            if row.get("effect_type") == effect_type and row.get("identity_key")
        }

    # ── Dedupe + events ──────────────────────────────────────────────

    def load_blocked_person_keys(self, person_keys: Iterable[str]) -> set[str]:
        keys = [key for key in person_keys if key]
        if not keys:
            return set()
        return self.store.get_blocked_person_keys(self.brief_id, keys)

    def record_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any] | None = None,
        run_id: int | None = None,
        work_unit_id: int | None = None,
    ) -> None:
        self.store.record_event(
            run_id=run_id,
            work_unit_id=work_unit_id,
            event_type=event_type,
            payload=payload,
        )


def _row_payload(row: dict) -> dict:
    return _json_object(row.get("payload_json"))


def _json_object(raw: Any) -> dict:
    """A dict from a JSON column, or ``{}`` — never a raise.

    A payload this version cannot parse is an absence, not a crash: the
    campaign plane is read on every session start and a single malformed
    row must not stop the run that would repair it.
    """

    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "LANE_DROPPED_NOTE",
    "PAYLOAD_LOST_NOTE",
    "PLAN_DROPPED_NOTE",
    "REQUEUABLE_NOTES",
    "PENDING_CANDIDATE_STATES",
    "RUNNABLE_QUERY_STATUSES",
    "SOURCE",
    "CampaignCandidate",
    "ExaRuntimeStateBridge",
    "PayloadRecovery",
    "PlanMerge",
]
