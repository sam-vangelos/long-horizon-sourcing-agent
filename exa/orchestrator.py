"""Exa pipeline — brief in, runtime state and a slate out.

The pipeline compiles a brief into Exa queries, acquires and deduplicates candidates,
judges them, persists canonical state, and writes a campaign slate. Acquisition spends
money and can be interrupted, so ordering protects budget and resumability.

SQLite is the campaign plane; each run row is one session. Campaign-scoped "seen",
pending work, and verdicts come from ``runtime_state.sqlite3``, never from a JSONL
projection or a session-local counter. The slate is campaign-wide. Costs stay
session-scoped for the run directory's ``pipeline_end`` record, with campaign-to-date
totals computed from append-only usage receipts.

The ordering rules:

1. Lint before lock, state, or the first Exa call; a blocking finding cannot spend.
2. Hold ``RuntimeStateLock(output_dir)`` for the run so one writer owns the state
   directory.
3. Persist each query's full harvest before judging it, so resume rejudges without
   repeating Exa spend.
4. Enforce discovery → snippet → full before judgment; append the judgment before the
   workspace side effect so a crash leaves a replayable save.
5. Write artifacts before ``finish_run``; a report failure degrades close-out without
   changing the run status chosen by the loop.

The console emits one prebuilt line per event, with no JSON, multi-line blocks, or model
prose.

Run snapshots currently copy the state directory's JSONL projections because Exa has
no registered progress projection; the slate remains one campaign-wide file overwritten
per session.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from exa.acquisition import is_transient_query_failure, run_query
from exa.client import ExaAuthError, ExaClient, ExaCreditsError
from exa.cost_meter import ExaCostMeter, ledger_spend_usd
from exa.employer_gate import (
    ALUMNI_GATE_PATH,
    GATE_PATH,
    alumni_blacklist_match,
    current_employer_signals,
    current_title_signals,
    employer_blacklist_match,
    gate_decision,
    gate_rationale,
    is_gate_decision,
)
from exa.identity import DoNotSaveGate, dedupe_candidates, exa_person_key
from exa.judge import (
    BUCKET_FAILURE,
    BUCKET_REJECT,
    BUCKET_REVIEW,
    BUCKET_SAVE,
    JUDGMENT_FINGERPRINT_KEY,
    JudgmentOutcome,
    classify_verdict,
    judge_candidate,
    judgment_fingerprint,
)
from exa.known import KnownPeople, load_known_people
from exa.location_gate import (
    LOCATION_GATE_PATH,
    OUTSIDE as LOCATION_OUTSIDE,
    LocationScope,
    LocationVerdict,
    is_location_gate_decision,
    location_gate_decision,
    location_scope_errors,
)
from exa.lane_compiler import (
    active_lanes,
    blocking_findings,
    compile_exa_queries,
    lint_exa_brief,
)
from exa.rejudge import (
    REJUDGE_EVENT,
    REJUDGE_KEPT_EVENT,
    REJUDGE_SPLIT_KEY,
    STALE_REJECT,
    Resolution,
    rejudge_status,
    resolve_samples,
    samples_for,
    selected_statuses,
)
from exa.report import (
    ArchetypeCoverage,
    FirmCoverage,
    RunView,
    SlateEntry,
    render_slate,
    structured_report_payload,
)
from exa.runtime_bridge import (
    LANE_DROPPED_NOTE,
    PAYLOAD_LOST_NOTE,
    PLAN_DROPPED_NOTE,
    RUNNABLE_QUERY_STATUSES,
    CampaignCandidate,
    ExaRuntimeStateBridge,
)
from exa.schemas import ExaCandidate, ExaProgress, ExaSearchQuery
from exa.scope_gate import (
    OUT_OF_SCOPE,
    CurrentEmployerScope,
    ScopeVerdict,
    scope_errors,
)
from shared.attainability import AttainabilityRules, classify_signals
from shared.brief_loader import load_brief
from shared.cost_rollup import (
    _sum_token_cost_log_usd,
    aggregate_cost_for_run,
    write_cost_rollup_sidecar,
)
from shared.execution.engine import CandidateExecutionEngine
from shared.failures import ApiBudgetExhaustedError
from shared.judger import init_judger
from shared.llm_usage import llm_usage_session, resolve_cost_log_run_id
from shared.output_paths import derive_brief_id
from shared.run_report_schema import StructuredRunReport
from shared.runtime_state.lock import RuntimeStateLock
from shared.runtime_state.store import EXA_QUERY_KIND, RuntimeStateStore
from shared.safety.stop_reasons import RunStopReason
from shared.save_destination import SAVE_DECISION_FAMILY
from shared.save_destination.candidate_workspace import (
    CandidateWorkspaceSaveDestination,
)
from shared.schemas import OpusDecision
from shared.storage import append_jsonl, log_event, read_jsonl


logger = logging.getLogger(__name__)

SOURCE = "exa"
RUNTIME_DB_FILENAME = "runtime_state.sqlite3"
WORKSPACE_EFFECT_TYPE = "candidate_workspace"
SLATE_FILENAME = "exa-slate.md"
# The saves.jsonl status of a line withdrawing an earlier save: the person
# left the short list on re-judge. The file is append-only, so the last line
# per identity key is that person's state.
SAVE_WITHDRAWN_STATUS = "superseded"

# Half a session's judgments failing is the line between "some calls
# retried badly" and "the slate is an artifact of an outage". Set at the
# midpoint deliberately: below it the surviving verdicts still carry the
# search, above it the run's shape is the transport's, not the market's.
RUN_DEGRADED_FAILURE_RATIO = 0.5

# How many re-judge samples run at once (``EXA_REJUDGE_CONCURRENCY``). A
# DeepSeek full judgment takes about a minute (mean 68s, median 51s over 74
# replayed SPL judgments), so 16 at a time re-judges the SPL campaign's 463
# earlier rejects in about half an hour instead of nine hours serially.
DEFAULT_REJUDGE_CONCURRENCY = 16

# States the pending-judgment pre-pass can actually judge FROM. It is a
# subset of the bridge's PENDING set: ``full_started`` cannot legally
# transition to ``full_started`` again, and it cannot occur after
# ``reconcile_open_attempts`` converts crash-orphaned attempts to
# ``failed_retryable``. A survivor is recorded as an event rather than
# forced through a guard that exists for a reason.
PRE_PASS_STATES = ("discovered", "snippet_extracted", "failed_retryable")

# The console verb per bucket. Padded to a fixed width so a wall of
# verdicts scans as columns rather than ragged prose.
_VERDICT_VERB = {
    BUCKET_SAVE: "SAVE  ",
    BUCKET_REVIEW: "REVIEW",
    BUCKET_REJECT: "REJECT",
    BUCKET_FAILURE: "FAIL  ",
}


class ExaBriefBlockedError(RuntimeError):
    """The brief carries a blocking lint finding; nothing ran."""


class ExaStateDirBusyError(RuntimeError):
    """Another process holds this state dir's advisory lock."""


@dataclass
class ExaRunStats:
    """What one SESSION did. The campaign's numbers live in SQLite."""

    run_id: int = 0
    queries_run: int = 0
    candidates_discovered: int = 0
    judged: int = 0
    saved: int = 0
    replayed_saves: int = 0
    review: int = 0
    rejected: int = 0
    failed: int = 0
    # Rejected by the employer-blacklist gate with no judgment spent. Not
    # counted in ``judged`` or ``rejected``: no model saw these people.
    blacklist_gated: int = 0
    # Rejected by the alumni-blacklist gate (anyone who has EVER worked at
    # a listed employer), likewise with no judgment spent.
    alumni_gated: int = 0
    # Rejected by the location-scope gate (a card location positively
    # outside the brief's countries), likewise with no judgment spent.
    location_gated: int = 0
    # Saves the current-employer scope check moved to review: the judge
    # saved them, and the record names a current employer outside the
    # brief's scope. Counted in ``review`` too; zero on a brief without
    # the scope block.
    scope_moved: int = 0
    # Verdicts made under earlier judging criteria and judged again this
    # session (exa/rejudge.py). Counted in ``judged`` and its buckets too.
    rejudged: int = 0
    # Of those: short-list verdicts the judge (or the scope check) moved off
    # the short list. People a deterministic gate removed are NOT in it; they
    # are the two counts after it.
    rejudge_left_short_list: int = 0
    rejudge_location_excluded: int = 0
    rejudge_blacklist_excluded: int = 0
    # Slate verdicts whose two samples split, now worth a look; earlier
    # rejects re-judged (the opt-in), and how many of them a sample promoted.
    rejudge_split: int = 0
    rejudge_rejects: int = 0
    rejudge_promoted: int = 0
    # Re-judges that left the stored verdict as it was, because a sample
    # failed; the next resume tries them again.
    rejudge_kept: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    exa_cost_usd: float = 0.0
    llm_cost_usd: float = 0.0
    campaign_exa_cost_usd: float = 0.0
    campaign_llm_cost_usd: float = 0.0
    paused: bool = False

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "queries_run": self.queries_run,
            "candidates_discovered": self.candidates_discovered,
            "judged": self.judged,
            "saved": self.saved,
            "replayed_saves": self.replayed_saves,
            "review": self.review,
            "rejected": self.rejected,
            "failed": self.failed,
            "blacklist_gated": self.blacklist_gated,
            "alumni_gated": self.alumni_gated,
            "location_gated": self.location_gated,
            "scope_moved": self.scope_moved,
            "rejudged": self.rejudged,
            "rejudge_left_short_list": self.rejudge_left_short_list,
            "rejudge_location_excluded": self.rejudge_location_excluded,
            "rejudge_blacklist_excluded": self.rejudge_blacklist_excluded,
            "rejudge_split": self.rejudge_split,
            "rejudge_rejects": self.rejudge_rejects,
            "rejudge_promoted": self.rejudge_promoted,
            "rejudge_kept": self.rejudge_kept,
            "skipped": dict(self.skipped),
            "exa_cost_usd": round(self.exa_cost_usd, 6),
            "llm_cost_usd": round(self.llm_cost_usd, 6),
            "campaign_exa_cost_usd": round(self.campaign_exa_cost_usd, 6),
            "campaign_llm_cost_usd": round(self.campaign_llm_cost_usd, 6),
            "paused": self.paused,
        }


@dataclass
class _RejudgeJob:
    """One stored verdict the re-judge pass judges again."""

    entry: CampaignCandidate
    candidate: ExaCandidate
    old: OpusDecision | None
    old_bucket: str
    status: str
    # Where the old verdict sat on the slate the recruiter last saw: its
    # bucket after the scope check (a save the scope moved was worth a look).
    old_slate_bucket: str = ""
    # "location" or "blacklist" when a deterministic gate settles it with no
    # model call, else "".
    gate: str = ""
    query: ExaSearchQuery | None = None
    outcomes: list = field(default_factory=list)


@dataclass(frozen=True)
class _WorkspaceEnvelope:
    """Fields consumed by ``CandidateWorkspaceSaveDestination``.

    The shared execution envelope omits ``candidate_id`` and
    ``runtime_state_path``, while the workspace destination requires both plus
    ``brief_id``. Keep this adapter-local envelope so the shared type stays
    unchanged for other sources.
    """

    brief_id: str
    candidate_id: int
    runtime_state_path: Path


class ExaPipeline:
    """One brief, one state dir, one resumable Exa campaign."""

    def __init__(
        self,
        brief_path: str | Path,
        output_dir: str | Path,
        *,
        client_factory: Callable[[ExaCostMeter], Any] | None = None,
        judge: Callable[..., Any] = judge_candidate,
        extractor: Callable | None = None,
        known_from: Sequence[Path] = (),
        rejudge_stale: bool = False,
        rejudge_stale_rejects: bool = False,
        rejudge_concurrency: int | None = None,
    ):
        self.brief_path = Path(brief_path)
        # The re-judge pass's opt-ins (exa/rejudge.py): slate verdicts made
        # under earlier criteria, and earlier rejects as well. Both off by
        # default: a campaign asks for the pass (``exa.run --rejudge-stale``),
        # because a resume that re-judges every legacy verdict whose brief
        # file moved spends model money the campaign never chose to spend.
        # A row a dead re-judge left cleared is finished either way.
        self._rejudge_stale = bool(rejudge_stale)
        self._rejudge_stale_rejects = bool(rejudge_stale_rejects)
        self._rejudge_concurrency = max(
            1,
            int(
                rejudge_concurrency
                if rejudge_concurrency is not None
                else _env_int("EXA_REJUDGE_CONCURRENCY", DEFAULT_REJUDGE_CONCURRENCY)
            ),
        )
        self.output_dir = Path(output_dir)
        self._client_factory = client_factory
        self._judge = judge
        self._extractor = extractor
        self._known_from = [Path(p) for p in (known_from or ())]

        self.brief = load_brief(str(self.brief_path))
        # The UNIVERSAL derivation, not ``brief.id``. Two different
        # identities were being conflated here. The state DIRECTORY is
        # keyed by ``exa_state_key`` (brief.id → role_title → stem) and
        # stays that way — it is a per-source folder name. The brief
        # IDENTITY written into store rows, workspace idempotency keys,
        # and cross-source lookups has to be the one every other plane
        # uses, or Exa's rows for a brief are invisible to any query
        # that resolves that brief the normal way. For the CoS brief
        # that identity is the LinkedIn project id (9000000006), which
        # is exactly the join the "already known" bucket depends on;
        # ``brief.id`` would have made it the role title and quietly
        # orphaned every row.
        self.brief_id = str(
            derive_brief_id(brief_path=self.brief_path, brief=self.brief)
            or self.brief_path.stem
        )
        self.brief_name = str(self.brief.role_title or self.brief_id)
        # The current-employer scope check, or None when the brief does not
        # carry ``exa_current_employer_scope``. A malformed block is a
        # blocking lint finding, so a run never starts without the check the
        # brief asked for; here it simply stays off.
        self._scope: CurrentEmployerScope | None = (
            None if scope_errors(self.brief) else CurrentEmployerScope.from_brief(self.brief)
        )
        # The location-scope gate, or None when the brief does not carry
        # ``exa_location_scope``. Same contract as the scope check: a
        # malformed block is a blocking lint finding, so it never runs off.
        self.location_scope: LocationScope | None = (
            None
            if location_scope_errors(self.brief)
            else LocationScope.from_brief(self.brief)
        )

        self.store: RuntimeStateStore | None = None
        self.bridge: ExaRuntimeStateBridge | None = None
        self._known: KnownPeople = KnownPeople()
        self._query_drops: dict[str, dict] = {}
        # What the judge is told about the role, hashed; stamped on every
        # model verdict so a later session can tell which verdicts the
        # criteria have moved past. "" when it cannot be computed.
        self._judgment_fingerprint: str = judgment_fingerprint(self.brief)

    # ── Paths ────────────────────────────────────────────────────────

    @property
    def db_path(self) -> Path:
        return self.output_dir / RUNTIME_DB_FILENAME

    @property
    def log_path(self) -> Path:
        return self.output_dir / "run_log.jsonl"

    @property
    def usage_log_path(self) -> Path:
        return self.output_dir / "token-cost-log.jsonl"

    @property
    def exa_usage_path(self) -> Path:
        return self.output_dir / "exa-usage.jsonl"

    @property
    def judgments_path(self) -> Path:
        return self.output_dir / "judgments.jsonl"

    @property
    def candidates_path(self) -> Path:
        return self.output_dir / "candidates.jsonl"

    @property
    def saves_path(self) -> Path:
        return self.output_dir / "saves.jsonl"

    @property
    def skipped_path(self) -> Path:
        return self.output_dir / "skipped.jsonl"

    @property
    def slate_path(self) -> Path:
        return self.output_dir / SLATE_FILENAME

    # ── Entry point ──────────────────────────────────────────────────

    async def run(
        self,
        *,
        resume: bool = False,
        max_queries: int | None = None,
    ) -> ExaRunStats:
        self.assert_brief_launchable()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock = RuntimeStateLock(self.output_dir, resource_name="Exa state dir")
        try:
            lock.acquire()
        except RuntimeError as exc:
            print(
                _one_line(
                    "    [exa] NO-GO — another Exa process holds this state dir"
                ),
                flush=True,
            )
            raise ExaStateDirBusyError(str(exc)) from exc

        try:
            return await self._run_locked(resume=resume, max_queries=max_queries)
        finally:
            lock.release()

    async def _run_locked(
        self, *, resume: bool, max_queries: int | None
    ) -> ExaRunStats:
        store = RuntimeStateStore(self.db_path)
        self.store = store
        bridge = ExaRuntimeStateBridge(
            store=store,
            output_dir=self.output_dir,
            brief_id=self.brief_id,
            brief_name=self.brief_name,
            brief_path=str(self.brief_path),
        )
        self.bridge = bridge
        self._known = (
            load_known_people(self._known_from) if self._known_from else KnownPeople()
        )

        initial = ExaProgress(
            brief_name=self.brief_name, queries=compile_exa_queries(self.brief)
        )
        # The brief is the authority on WHAT runs. On a resume the
        # cloned work units carry what already ran, and the two are
        # reconciled inside the bridge — without the compiled plan and
        # the lane list going in, a lane added mid-campaign never runs
        # and a lane deleted mid-campaign runs anyway.
        run_id, progress = bridge.start_or_resume_run(
            resume=resume,
            initial_progress=initial,
            active_lanes=active_lanes(self.brief),
        )
        self._query_drops = self._load_query_drops(run_id)
        if resume:
            self._requeue_transient_errors(run_id, progress)
        init_judger(self.brief)

        stats = ExaRunStats(run_id=run_id)
        meter = ExaCostMeter(
            cap_usd=_run_budget_usd(), sink_path=self.exa_usage_path
        )
        engine = CandidateExecutionEngine(
            store=store,
            output_dir=str(self.output_dir),
            brief_id=self.brief_id,
            source=SOURCE,
        )
        run_status = "completed"
        stop_reason = RunStopReason.NORMAL
        stop_detail = ""
        primary_error = False

        try:
            with llm_usage_session(
                self.usage_log_path,
                source=SOURCE,
                brief_id=self.brief_id,
                run_id=run_id,
            ):
                # Campaign plane first: finish what the last session
                # started before spending a cent on new acquisition. The
                # re-judge runs before the replay so a stranded save the
                # current criteria no longer make never lands.
                self._rejudge_stale_verdicts(
                    run_id=run_id, progress=progress, engine=engine, stats=stats
                )
                self._replay_interrupted_saves(
                    run_id=run_id, progress=progress, engine=engine, stats=stats
                )
                self._judge_pending(
                    run_id=run_id, progress=progress, engine=engine, stats=stats
                )

                runnable = [
                    query
                    for query in progress.queries
                    if query.status in RUNNABLE_QUERY_STATUSES
                ]
                budget = len(runnable) if max_queries is None else max_queries
                if runnable and budget > 0:
                    async with contextlib.AsyncExitStack() as stack:
                        client = (
                            self._client_factory(meter)
                            if self._client_factory is not None
                            else ExaClient(meter=meter)
                        )
                        if hasattr(client, "__aenter__"):
                            client = await stack.enter_async_context(client)
                        await self._execute(
                            run_id=run_id,
                            progress=progress,
                            stats=stats,
                            engine=engine,
                            client=client,
                            max_queries=max_queries,
                        )

                remaining = sum(
                    1
                    for query in progress.queries
                    if query.status in RUNNABLE_QUERY_STATUSES
                )
                if remaining and max_queries is not None:
                    # An operator max-queries stop is interruption, not completion. The
                    # canonical stop-reason enum has no member for this detail, so the
                    # event payload carries it while the column records OPERATOR_STOP.
                    run_status = "interrupted"
                    stop_reason = RunStopReason.OPERATOR_STOP
                    stop_detail = "max_queries"
                    stats.paused = True
                    print(
                        _one_line(
                            f"    [exa] stopped after {stats.queries_run} queries "
                            f"— {remaining} remain; resume with the same command"
                        ),
                        flush=True,
                    )
        except ApiBudgetExhaustedError as exc:
            run_status = "interrupted"
            stop_reason = RunStopReason.API_BUDGET_EXHAUSTED
            stats.paused = True
            print(
                _one_line(
                    f"    [exa] paused — run budget reached "
                    f"(${meter.committed_usd:.2f} of ${meter.cap_usd:.2f}); "
                    f"resume with the same command"
                ),
                flush=True,
            )
            logger.debug("exa run paused on budget: %s", exc)
        except (ExaAuthError, ExaCreditsError) as exc:
            run_status = "error"
            stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
            primary_error = True
            kind = (
                "credits exhausted"
                if isinstance(exc, ExaCreditsError)
                else "auth failed"
            )
            print(
                _one_line(f"    [exa] stopped — Exa {kind}; run recorded as error"),
                flush=True,
            )
            raise
        except KeyboardInterrupt:
            run_status = "interrupted"
            stop_reason = RunStopReason.OPERATOR_STOP
            stop_detail = "keyboard_interrupt"
            print(_one_line("    [exa] interrupted — progress saved"), flush=True)
        except Exception:
            run_status = "error"
            stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
            primary_error = True
            raise
        finally:
            self._close_out(
                run_id=run_id,
                progress=progress,
                stats=stats,
                meter=meter,
                run_status=run_status,
                stop_reason=stop_reason,
                stop_detail=stop_detail,
                primary_error=primary_error,
            )

        return stats

    # ── Preflight ────────────────────────────────────────────────────

    def lint(self) -> list:
        return lint_exa_brief(self.brief)

    def assert_brief_launchable(self) -> None:
        """Print and raise on blocking findings — before lock, state, spend."""

        blocking = blocking_findings(self.lint())
        if not blocking:
            return
        for finding in blocking:
            print(
                _one_line(f"    [exa] BLOCKED {finding.code}: {finding.message}"),
                flush=True,
            )
        raise ExaBriefBlockedError(
            f"{len(blocking)} blocking lint finding(s); nothing ran"
        )

    # ── Campaign pre-pass ────────────────────────────────────────────

    def _replay_interrupted_saves(
        self,
        *,
        run_id: int,
        progress: ExaProgress,
        engine: CandidateExecutionEngine,
        stats: ExaRunStats,
    ) -> None:
        """Finish saves that a previous session decided but never landed.

        The window is real: a SAVE verdict is durable the instant
        ``finish_stage_success`` commits, and the workspace write happens
        after it. A crash in between leaves a person the campaign
        believes it saved and a workspace that has never heard of them —
        and nothing would ever revisit them, because a SAVE-family
        terminal suppresses re-judgement by design.

        The idempotency key is the same one the live path uses, so a
        replay of a save that DID land is a no-op the store refuses
        rather than a duplicate row.
        """

        assert self.bridge is not None
        already = self.bridge.succeeded_side_effect_keys(WORKSPACE_EFFECT_TYPE)
        replayed = 0
        for entry in self.bridge.list_campaign_candidates():
            if not entry.is_terminal:
                continue
            if entry.terminal_decision not in SAVE_DECISION_FAMILY:
                continue
            if entry.identity_key in already:
                continue
            candidate = _candidate_from_record(entry)
            decision = _decision_from_entry(entry)
            if candidate is None or decision is None:
                continue
            # Only a short-list verdict was ever written to the workspace. A
            # hedged save (TRANSFERABLE_SAVE, SIGNAL_SAVE) classifies to
            # review, so the live path never wrote it and its missing
            # side-effect row is not an interrupted save.
            if classify_verdict(decision) != BUCKET_SAVE:
                continue
            # A save decided before the brief excluded this person never
            # lands: the gates that would reject them today run first.
            gate = self._blacklisted_employer(candidate, during="replay check")
            if gate:
                gate_path, (blocked, _field, _value) = gate
                which = (
                    "alumni blacklist"
                    if gate_path == ALUMNI_GATE_PATH
                    else "employer blacklist"
                )
                print(
                    _one_line(
                        f"    [exa] {candidate.name or '(unnamed)'} — interrupted "
                        f"save not replayed: {which} ({blocked})"
                    ),
                    flush=True,
                )
                continue
            # Nor does a save whose current employer the brief's scope rules
            # out: the live path sent it to review, never to the workspace.
            scope = self._scope_verdict(candidate, during="replay check")
            if scope is not None and scope.status == OUT_OF_SCOPE:
                print(
                    _one_line(
                        f"    [exa] {candidate.name or '(unnamed)'} — interrupted "
                        f"save not replayed: {scope.note}"
                    ),
                    flush=True,
                )
                continue
            # Nor a save whose card location the brief's location scope rules out.
            located = self._location_excluded(candidate, during="replay check")
            if located is not None:
                print(
                    _one_line(
                        f"    [exa] {candidate.name or '(unnamed)'} — interrupted "
                        f"save not replayed: {located.note(self.location_scope)}"
                    ),
                    flush=True,
                )
                continue
            # Nor a re-judge whose samples split: it is worth a look, and the
            # live path never wrote it to the workspace.
            if entry.rejudge_split:
                continue
            envelope = self._replay_envelope(
                run_id=run_id, progress=progress, entry=entry, engine=engine
            )
            if self._save_to_workspace(
                run_id=run_id,
                envelope=envelope,
                engine=engine,
                candidate=candidate,
                decision=decision,
                attempt_id=entry.last_attempt_id,
            ):
                replayed += 1
        if replayed:
            stats.replayed_saves = replayed
            stats.saved += replayed
            print(
                _one_line(
                    f"    [exa] replayed {replayed} interrupted save(s) from the "
                    "last session"
                ),
                flush=True,
            )

    def _judge_pending(
        self,
        *,
        run_id: int,
        progress: ExaProgress,
        engine: CandidateExecutionEngine,
        stats: ExaRunStats,
    ) -> None:
        """Judge everyone a previous session acquired but never judged.

        No Exa call: the record was written to canonical state before the
        first Opus call of the session that found them, so re-judging
        them costs model spend only. This is what makes acquisition
        durable rather than merely logged — an interrupted campaign never
        re-pays Exa for people it already has.
        """

        assert self.bridge is not None
        pending = [
            entry
            for entry in self.bridge.list_campaign_candidates()
            if entry.state in PRE_PASS_STATES and entry.record
        ]
        stranded = [
            entry
            for entry in self.bridge.list_campaign_candidates()
            if entry.is_pending and entry.state not in PRE_PASS_STATES
        ]
        for entry in stranded:
            self.bridge.record_event(
                event_type="exa_pending_candidate_stranded",
                payload={"identity_key": entry.identity_key, "state": entry.state},
                run_id=run_id,
            )
        if not pending:
            return

        print(
            _one_line(
                f"    [exa] resuming {len(pending)} pending judgment(s) from the "
                "last session"
            ),
            flush=True,
        )
        for entry in pending:
            candidate = _candidate_from_record(entry)
            if candidate is None:
                continue
            query = self._query_for(candidate, progress)
            envelope = self._envelope(
                run_id=run_id, query=query, candidate=candidate, engine=engine
            )
            if entry.state == "discovered":
                # The lifecycle guard requires a snippet before full; record it so a
                # resumed discovered entry reaches judgment without bypassing that invariant.
                engine.runtime.record_snippet_extracted(
                    envelope,
                    payload={
                        "snippet": envelope.snippet.to_dict(),
                        "candidate_record": candidate.to_dict(),
                    },
                )
            self._judge_one(
                run_id=run_id,
                query=query,
                candidate=candidate,
                engine=engine,
                stats=stats,
            )

    def _rejudge_stale_verdicts(
        self,
        *,
        run_id: int,
        progress: ExaProgress,
        engine: CandidateExecutionEngine,
        stats: ExaRunStats,
    ) -> None:
        """Judge stored verdicts again when the criteria have moved past them.

        Which verdicts, how many samples each, and how the samples combine
        is ``exa/rejudge.py``'s to say; the campaign's opt-ins choose whether
        slate verdicts and earlier rejects are in at all. The samples run in
        parallel (``EXA_REJUDGE_CONCURRENCY`` at a time) and touch no state;
        each person is then settled on this thread through
        :meth:`_judge_one`, the live path, so the gates, the scope check, the
        workspace write and the console line are the ones a new person gets.
        No Exa call: the record is in the store. A person a deterministic
        gate now excludes is settled by the gate, with no model call.

        The store refuses to re-open a save-family verdict
        (``_guard_save_family_reentry``), so the verdict is cleared first,
        through the store's own ``clear_candidate_terminal_state``, and only
        once its samples are in. The old verdict and the samples go to a
        durable event before the clear, because the clear empties the
        terminal payload. A process that dies between the clear and the new
        verdict leaves the row ``CLEARED``, and the next session picks it up
        here, with the old verdict read back from that event.

        A sample that failed leaves the stored verdict untouched. A save that
        leaves the short list keeps its workspace row, which carries the
        recruiter's review and outreach fields; its ``saves.jsonl`` line is
        followed by one that withdraws it (``SAVE_WITHDRAWN_STATUS``), and the
        slate lists it under "Left the short list on re-judge".
        """

        assert self.bridge is not None and self.store is not None
        run_row = self.store.get_run(run_id) or {}
        current_hash = str(run_row.get("brief_content_hash") or "")
        selected = selected_statuses(
            slate=self._rejudge_stale, rejects=self._rejudge_stale_rejects
        )
        old_by_key = {
            str(payload.get("identity_key") or ""): payload
            for payload in self.bridge.list_event_payloads(REJUDGE_EVENT)
        }
        jobs: list[_RejudgeJob] = []
        for entry in self.bridge.list_campaign_candidates():
            if not entry.record:
                continue
            status = rejudge_status(
                state=entry.state,
                terminal_decision=entry.terminal_decision,
                decision=entry.decision,
                judgment_fingerprint=entry.judgment_fingerprint,
                user_status=entry.user_status,
                brief_content_hash=entry.brief_content_hash,
                current_fingerprint=self._judgment_fingerprint,
                current_brief_hash=current_hash,
            )
            if status not in selected:
                continue
            candidate = _candidate_from_record(entry)
            if candidate is None:
                continue
            old = (
                _decision_from_entry(entry)
                if entry.terminal_decision
                else _decision_from_dict(
                    (old_by_key.get(entry.identity_key) or {}).get("old_decision")
                )
            )
            gate = ""
            if self._blacklisted_employer(candidate, during="gate"):
                gate = "blacklist"
            elif self._location_excluded(candidate, during="gate") is not None:
                gate = "location"
            old_bucket = classify_verdict(old) if old is not None else ""
            old_slate_bucket = old_bucket
            if old_bucket == BUCKET_SAVE:
                scope = self._scope_for_verdict(
                    candidate, old, old_bucket, during="re-judge"
                )
                if scope is not None and scope.note:
                    old_slate_bucket = BUCKET_REVIEW
            jobs.append(
                _RejudgeJob(
                    entry=entry,
                    candidate=candidate,
                    old=old,
                    old_bucket=old_bucket,
                    status=status,
                    gate=gate,
                    query=self._query_for(candidate, progress),
                    old_slate_bucket=old_slate_bucket,
                )
            )
        if not jobs:
            return

        rejects = sum(1 for job in jobs if job.status == STALE_REJECT)
        slate = len(jobs) - rejects
        parts = []
        if slate:
            parts.append(
                f"{slate} slate {'verdict' if slate == 1 else 'verdicts'} "
                "(two samples each)"
            )
        if rejects:
            parts.append(
                f"{rejects} earlier {'reject' if rejects == 1 else 'rejects'} "
                "(one sample each)"
            )
        print(
            _one_line(
                f"    [exa] re-judging {' and '.join(parts)} made under earlier "
                f"judging criteria, {self._rejudge_concurrency} at a time "
                "(no Exa call)"
            ),
            flush=True,
        )

        landed = self.bridge.succeeded_side_effect_keys(WORKSPACE_EFFECT_TYPE)
        superseded = _superseded_save_keys(self.saves_path)

        def settle(job: _RejudgeJob, resolution: Resolution | None) -> None:
            self._settle_rejudge(
                job,
                resolution,
                run_id=run_id,
                engine=engine,
                stats=stats,
                current_hash=current_hash,
                landed=landed,
                superseded=superseded,
            )

        for job in jobs:
            if job.gate:
                settle(job, None)

        sampled = [job for job in jobs if not job.gate]
        if not sampled:
            return
        pool = ThreadPoolExecutor(
            max_workers=self._rejudge_concurrency,
            thread_name_prefix="exa-rejudge",
        )
        try:
            futures = {}
            for job in sampled:
                for _ in range(samples_for(job.old_bucket)):
                    # Each sample runs in a copy of this context, so its usage
                    # row lands in this session's token-cost log.
                    future = pool.submit(
                        contextvars.copy_context().run,
                        self._judge,
                        job.candidate,
                        self.brief,
                        query=job.query,
                    )
                    futures[future] = job
            for future in as_completed(futures):
                job = futures[future]
                job.outcomes.append(future.result())
                if len(job.outcomes) == samples_for(job.old_bucket):
                    settle(job, resolve_samples(job.old_bucket, job.outcomes))
        except BaseException:
            # A budget stop or an interrupt: queued samples are dropped, and
            # nobody whose samples were not all in has been touched.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

    def _settle_rejudge(
        self,
        job: _RejudgeJob,
        resolution: Resolution | None,
        *,
        run_id: int,
        engine: CandidateExecutionEngine,
        stats: ExaRunStats,
        current_hash: str,
        landed: set[str],
        superseded: set[str],
    ) -> None:
        """Settle one re-judged verdict; ``resolution`` is None for a gate's."""

        assert self.bridge is not None and self.store is not None
        entry, candidate = job.entry, job.candidate
        old_payload = job.old.to_dict() if job.old is not None else {}
        if resolution is not None and resolution.outcome is None:
            stats.rejudge_kept += 1
            self.bridge.record_event(
                event_type=REJUDGE_KEPT_EVENT,
                payload={
                    "identity_key": entry.identity_key,
                    "reason": resolution.reason,
                    "old_decision": old_payload,
                    "samples": list(resolution.samples),
                },
                run_id=run_id,
            )
            print(
                _one_line(
                    f"    [exa] KEEP   {_flatten(candidate.name) or '(unnamed)'} — a "
                    "re-judge sample failed; the stored verdict stands until the "
                    "next resume"
                ),
                flush=True,
            )
            return

        split = bool(resolution is not None and resolution.split)
        if entry.terminal_decision:
            self.bridge.record_event(
                event_type=REJUDGE_EVENT,
                payload={
                    "identity_key": entry.identity_key,
                    "old_decision": old_payload,
                    "old_bucket": job.old_bucket,
                    "old_slate_bucket": job.old_slate_bucket,
                    "old_judgment_fingerprint": entry.judgment_fingerprint,
                    "old_brief_content_hash": entry.brief_content_hash,
                    "judgment_fingerprint": self._judgment_fingerprint,
                    "brief_content_hash": current_hash,
                    "gate": job.gate,
                    "samples": list(resolution.samples) if resolution else [],
                    "split": split,
                },
                run_id=run_id,
            )
            self.store.clear_candidate_terminal_state(
                source=SOURCE,
                brief_id=self.brief_id,
                identity_key=entry.identity_key,
            )
        bucket = self._judge_one(
            run_id=run_id,
            query=job.query,
            candidate=candidate,
            engine=engine,
            stats=stats,
            outcome=resolution.outcome if resolution is not None else None,
            split_samples=resolution.samples if split else (),
            rejudged=True,
        )
        stats.rejudged += 1
        if job.status == STALE_REJECT:
            stats.rejudge_rejects += 1
            if bucket in (BUCKET_SAVE, BUCKET_REVIEW):
                stats.rejudge_promoted += 1
        if split:
            stats.rejudge_split += 1
        # Measured against the slate the recruiter last saw: a gate's
        # exclusion counts only for someone who was on it, and "left the
        # short list" only for someone who was on that.
        on_slate = job.old_slate_bucket in (BUCKET_SAVE, BUCKET_REVIEW)
        if job.gate == "location":
            stats.rejudge_location_excluded += int(on_slate)
        elif job.gate:
            stats.rejudge_blacklist_excluded += int(on_slate)
        elif job.old_slate_bucket == BUCKET_SAVE and bucket != BUCKET_SAVE:
            stats.rejudge_left_short_list += 1

        # saves.jsonl is the operator's deliverable alongside the slate. A
        # landed save that leaves the short list gets a line withdrawing it;
        # one a later re-judge puts back gets a line restoring it.
        key = entry.identity_key
        if key not in landed:
            return
        settled, scope_note = self._settled_decision(key)
        reason = _flatten(settled.rationale if settled is not None else "")
        if split:
            reason = _split_reason(resolution.samples if resolution else ())
        elif scope_note and bucket == BUCKET_REVIEW:
            reason = scope_note
        if bucket != BUCKET_SAVE and key not in superseded:
            self._append_saves_marker(
                run_id=run_id,
                candidate=candidate,
                status=SAVE_WITHDRAWN_STATUS,
                decision=settled,
                previous=str(old_payload.get("decision") or ""),
                now=_now_label(bucket, job.gate),
                reason=reason,
            )
            superseded.add(key)
        elif bucket == BUCKET_SAVE and key in superseded:
            self._append_saves_marker(
                run_id=run_id,
                candidate=candidate,
                status="succeeded",
                decision=settled,
                previous=str(old_payload.get("decision") or ""),
                now="short list",
                reason=reason,
            )
            superseded.discard(key)

    def _settled_decision(self, identity_key: str) -> tuple[OpusDecision | None, str]:
        """The verdict the store now holds, and the scope note it carries."""

        assert self.store is not None
        row = self.store.get_candidate(
            source=SOURCE, brief_id=self.brief_id, identity_key=identity_key
        )
        payload = _json_object((row or {}).get("terminal_payload_json"))
        decision = _decision_from_dict(payload.get("full_decision"))
        scope = _json_object(payload.get("employer_scope"))
        note = ""
        if scope.get("status") == OUT_OF_SCOPE and scope.get("employer"):
            note = f"scope: current employer {scope['employer']}"
        return decision, note

    def _append_saves_marker(
        self,
        *,
        run_id: int,
        candidate: ExaCandidate,
        status: str,
        decision: OpusDecision | None,
        previous: str,
        now: str,
        reason: str,
    ) -> None:
        """One ``saves.jsonl`` line recording that a landed save moved.

        Appended, never rewritten: the file is append-only like every other
        projection here, and its last line for an identity key is that
        person's current state. ``SAVE_WITHDRAWN_STATUS`` withdraws the save
        (the person left the short list); ``succeeded`` with ``reinstated``
        puts it back.
        """

        record = {
            "run_id": run_id,
            "identity_key": candidate.identity_key,
            "person_key": exa_person_key(candidate),
            "name": candidate.name,
            "profile_url": _profile_url(candidate),
            "decision": decision.decision if decision is not None else "",
            "destination": "candidate_workspace",
            "status": status,
            "evidence_urls": candidate.evidence_urls(),
            "verify_before_outreach": True,
            "rejudge": {
                "previous_decision": previous,
                "now": now,
                "reason": reason,
            },
            "timestamp": _now_utc_iso(),
        }
        if status != SAVE_WITHDRAWN_STATUS:
            record["reinstated"] = True
        append_jsonl(self.saves_path, record)

    def _query_for(
        self, candidate: ExaCandidate, progress: ExaProgress
    ) -> ExaSearchQuery:
        """The query that found this person, or a stand-in carrying its id.

        Matched on the STABLE identity — ``(source_lane,
        source_coverage_key)`` — before the ordinal id. Compiled ids
        move: dropping a lane or an archetype renumbers every query
        after it, so a person discovered by id 61 in one session is
        matched to whatever became id 61 in the next, and the pending
        re-judgment is then run against another query's firm and
        another query's text. The id is the fallback for records written
        before ``source_coverage_key`` existed, where it is the only
        identity there is.

        A brief edited between sessions can drop the query a pending
        person came from. The work unit is gone, so the envelope's
        work-unit lookup returns None and the attempt simply carries no
        work unit — the person is still judged, which is the outcome that
        matters.
        """

        if candidate.source_coverage_key:
            for query in progress.queries:
                if (
                    query.lane == candidate.source_lane
                    and query.coverage_key == candidate.source_coverage_key
                ):
                    return query
        else:
            for query in progress.queries:
                if query.id == candidate.source_query_id:
                    return query
        return ExaSearchQuery(
            id=candidate.source_query_id,
            name=f"{candidate.source_lane or 'exa'} · (retired query)",
            lane=candidate.source_lane,
            firm=candidate.current_firm,
            query="",
        )

    # ── The loop ─────────────────────────────────────────────────────

    async def _execute(
        self,
        *,
        run_id: int,
        progress: ExaProgress,
        stats: ExaRunStats,
        engine: CandidateExecutionEngine,
        client: Any,
        max_queries: int | None,
    ) -> None:
        assert self.bridge is not None and self.store is not None
        bridge = self.bridge
        gate = DoNotSaveGate.from_brief(self.brief)
        seen = {
            entry.identity_key
            for entry in bridge.list_campaign_candidates()
            if entry.identity_key
        }

        for query in progress.queries:
            if query.status not in RUNNABLE_QUERY_STATUSES:
                continue
            if max_queries is not None and stats.queries_run >= max_queries:
                break

            query.status = "in_progress"
            progress.current_query_id = query.id
            self._sync(run_id, progress)

            try:
                outcome = await run_query(
                    query,
                    client=client,
                    extractor=self._extractor or _default_extractor(),
                    usage_context={"brief_id": self.brief_id},
                )
            except ApiBudgetExhaustedError:
                # Budget exhaustion is raised before dispatch, so this query yielded no
                # candidates and must remain queued for a later session.
                self._requeue(run_id, progress, query)
                raise

            stats.queries_run += 1
            query.result_count = outcome.result_count
            progress.api_calls_made += outcome.exa_calls

            deduped = dedupe_candidates(
                outcome.candidates,
                seen_identity_keys=seen,
                blocked_person_keys=bridge.load_blocked_person_keys(
                    [exa_person_key(c) for c in outcome.candidates]
                ),
                gate=gate,
            )
            self._record_skips(query, deduped, stats, run_id)
            self._record_drops(query, outcome.dropped, deduped, stats)

            # PHASE 1 — the whole batch becomes canonical state before a
            # single Opus call. This is the durability guarantee: after
            # this loop the Exa spend for this query can never be lost,
            # whatever happens during judgment.
            persisted: list[ExaCandidate] = []
            for candidate in deduped.kept:
                self._persist_candidate(
                    run_id=run_id, query=query, candidate=candidate, engine=engine
                )
                seen.add(candidate.identity_key)
                stats.candidates_discovered += 1
                persisted.append(candidate)

            query.candidates_discovered += len(persisted)
            progress.candidates_discovered += len(persisted)
            query.status = "error" if outcome.notes else "done"
            query.notes = outcome.notes
            progress.current_query_id = None
            # Marked done BEFORE judgment, deliberately: the Exa half of
            # this query is finished and persisted. If judgment then hits
            # the budget cap, the next session must NOT re-run the search
            # — it picks the batch up from the store instead.
            self._sync(run_id, progress)

            # PHASE 2 — judgment, one person at a time.
            for candidate in persisted:
                self._judge_one(
                    run_id=run_id,
                    query=query,
                    candidate=candidate,
                    engine=engine,
                    stats=stats,
                )

            self._sync(run_id, progress)
            engine.runtime.flush_projections_if_needed(run_id=run_id)

    def _requeue_transient_errors(
        self, run_id: int, progress: ExaProgress
    ) -> None:
        """Put every query a network fault errored back in the queue.

        ``run_query`` absorbs a failed search into the query's note and the
        loop marks it ``error``, which no session ever runs again. When the
        fault was the network's (a DNS outage, a dropped connection, Exa's
        edge failing five times), the query itself is fine and the people it
        would have found are still unfound, so a resume runs it. A query that
        failed on its merits (a 4xx) stays ``error``. The original note rides
        a durable event, because the row's note is cleared with the requeue.
        """

        retried = [
            query
            for query in progress.queries
            if query.status == "error" and is_transient_query_failure(query.notes)
        ]
        if not retried:
            return
        assert self.bridge is not None
        self.bridge.record_event(
            event_type="exa_queries_requeued_after_transient_error",
            payload={
                "queries": [
                    {"query_id": query.id, "name": query.name, "note": query.notes}
                    for query in retried
                ]
            },
            run_id=run_id,
        )
        for query in retried:
            query.status = "queued"
            query.notes = ""
        self._sync(run_id, progress)
        names = [query.name or f"query {query.id}" for query in retried]
        listed = ", ".join(names[:4]) + (
            f" (+{len(names) - 4} more)" if len(names) > 4 else ""
        )
        print(
            _one_line(
                f"    [exa] {len(retried)} "
                f"{'query' if len(retried) == 1 else 'queries'} requeued after a "
                f"transient network error: {listed}"
            ),
            flush=True,
        )

    def _sync(self, run_id: int, progress: ExaProgress) -> None:
        assert self.bridge is not None
        self.bridge.sync_progress(
            run_id, progress, metrics_by_query=self._query_drops
        )

    def _requeue(
        self, run_id: int, progress: ExaProgress, query: ExaSearchQuery
    ) -> None:
        query.status = "queued"
        progress.current_query_id = None
        try:
            self._sync(run_id, progress)
        except Exception as exc:  # noqa: BLE001 — never mask the real failure
            logger.debug("requeue sync failed for q%s: %s", query.id, exc)

    # ── Per candidate ────────────────────────────────────────────────

    def _envelope(
        self,
        *,
        run_id: int,
        query: ExaSearchQuery,
        candidate: ExaCandidate,
        engine: CandidateExecutionEngine,
    ):
        return engine.envelope(
            source=SOURCE,
            brief_id=self.brief_id,
            run_id=run_id,
            work_unit_kind=EXA_QUERY_KIND,
            work_unit_source_id=str(query.id),
            identity_key=candidate.identity_key,
            display_name=candidate.name,
            profile_url=_profile_url(candidate),
            snippet=candidate.to_snippet(
                source_string_id=query.id, source_string_name=query.name
            ),
            source_cursor={
                "query_id": query.id,
                "lane": query.lane,
                "coverage_key": query.coverage_key,
            },
            metadata={
                "person_key": exa_person_key(candidate),
                "verify_before_outreach": True,
                "data_sufficiency": candidate.data_sufficiency,
                # The query's STABLE identity, alongside the ordinal id
                # in ``source_cursor``. Attribution is read back off
                # this: ids are positional and a brief edit renumbers
                # them, so a metadata row that carried only the id would
                # credit this person to a different query next session.
                "source_coverage_key": candidate.source_coverage_key,
                "source_lane": candidate.source_lane,
            },
        )

    def _replay_envelope(
        self,
        *,
        run_id: int,
        progress: ExaProgress,
        entry: CampaignCandidate,
        engine: CandidateExecutionEngine,
    ):
        """The envelope a replayed save lands under.

        The work unit is resolved through :meth:`_query_for`, on the
        query's stable identity, for the same reason the judgment path
        is: the ordinal id a person was discovered under may belong to a
        different query by the time this session compiles the brief, and
        a side-effect row filed against the wrong work unit is a save
        credited to a search that never found them.
        """

        candidate = _candidate_from_record(entry)
        query = self._query_for(candidate, progress) if candidate else None
        query_id = query.id if query else 0
        lane = query.lane if query else ""
        return engine.envelope(
            source=SOURCE,
            brief_id=self.brief_id,
            run_id=run_id,
            work_unit_kind=EXA_QUERY_KIND,
            work_unit_source_id=str(query_id),
            identity_key=entry.identity_key,
            display_name=entry.display_name,
            profile_url=entry.profile_url,
            source_cursor={
                "query_id": query_id,
                "lane": lane,
                "coverage_key": query.coverage_key if query else "",
                "replay": True,
            },
            metadata={"person_key": entry.person_key},
        )

    def _persist_candidate(
        self,
        *,
        run_id: int,
        query: ExaSearchQuery,
        candidate: ExaCandidate,
        engine: CandidateExecutionEngine,
    ) -> int:
        """Persist discovery and snippet rows, carrying the candidate record.

        ``ensure_candidate`` runs first because it accepts an explicit
        ``person_key``. The store mints that key only for GitHub; Exa supplies
        ``exa_person_key(candidate)`` on the ensure call and every envelope so
        cross-source identity lookups remain complete.

        The snippet payload carries the full candidate record, including evidence,
        so a later session can rejudge without another Exa call.
        """

        assert self.store is not None
        candidate_id = self.store.ensure_candidate(
            source=SOURCE,
            brief_id=self.brief_id,
            identity_key=candidate.identity_key,
            display_name=candidate.name,
            profile_url=_profile_url(candidate),
            person_key=exa_person_key(candidate),
        )
        envelope = self._envelope(
            run_id=run_id, query=query, candidate=candidate, engine=engine
        )
        engine.runtime.record_discovery(envelope, payload=candidate.to_dict())
        append_jsonl(
            self.candidates_path,
            {
                "run_id": run_id,
                "query_id": query.id,
                "lane": query.lane,
                "firm": query.firm,
                "person_key": exa_person_key(candidate),
                **candidate.to_dict(),
            },
        )
        engine.runtime.record_snippet_extracted(
            envelope,
            payload={
                "snippet": envelope.snippet.to_dict(),
                "candidate_record": candidate.to_dict(),
            },
        )
        return candidate_id

    def _judge_one(
        self,
        *,
        run_id: int,
        query: ExaSearchQuery,
        candidate: ExaCandidate,
        engine: CandidateExecutionEngine,
        stats: ExaRunStats,
        outcome: JudgmentOutcome | None = None,
        split_samples: Sequence[dict] = (),
        rejudged: bool = False,
    ) -> str:
        """Judge one person and settle the verdict; returns the slate bucket.

        A deterministic gate's rejection returns ``BUCKET_REJECT``; a failed
        judgment ``BUCKET_FAILURE``; a save the scope check moved returns
        ``BUCKET_REVIEW``.

        The re-judge pass hands in an ``outcome`` its samples already
        produced, so no model call happens here; ``split_samples`` marks a
        split re-judge, whose outcome ``exa.rejudge.resolve_samples`` already
        bucketed as worth-a-look, so it never reaches the workspace.
        """

        envelope = self._envelope(
            run_id=run_id, query=query, candidate=candidate, engine=engine
        )
        blacklist_hit = self._blacklisted_employer(candidate, during="gate")
        attempt_id = engine.runtime.start_stage(envelope, stage="full")

        if blacklist_hit:
            gate_path, hit = blacklist_hit
            self._settle_blacklisted(
                run_id=run_id,
                query=query,
                candidate=candidate,
                engine=engine,
                envelope=envelope,
                attempt_id=attempt_id,
                hit=hit,
                stats=stats,
                path=gate_path,
            )
            return BUCKET_REJECT

        located = self._location_excluded(candidate, during="gate")
        if located is not None:
            self._settle_location_gated(
                run_id=run_id,
                query=query,
                candidate=candidate,
                engine=engine,
                envelope=envelope,
                attempt_id=attempt_id,
                verdict=located,
                stats=stats,
            )
            return BUCKET_REJECT

        if outcome is None:
            try:
                outcome = self._judge(candidate, self.brief, query=query)
            except ApiBudgetExhaustedError as exc:
                # Budget exhaustion is terminal by default; force_retryable keeps
                # this candidate rejudgeable after a pause, and the pending
                # pre-pass resumes it in the next session.
                engine.runtime.finish_stage_failure(
                    attempt_id=attempt_id,
                    envelope=envelope,
                    stage="full",
                    error_or_failure_decision=exc,
                    extra_payload={
                        "force_retryable": True,
                        "paused": True,
                        "candidate_record": candidate.to_dict(),
                    },
                )
                raise

        decision = outcome.full
        bucket = outcome.bucket
        stats.judged += 1

        # The current-employer scope check, after the verdict and without
        # changing it: a save at an identified employer outside the brief's
        # scope goes to review rather than the workspace.
        scope = self._scope_for_verdict(candidate, decision, bucket, during="judgment")
        scope_note = scope.note if scope is not None else ""
        if scope_note and bucket == BUCKET_SAVE:
            bucket = BUCKET_REVIEW
            stats.scope_moved += 1
        scope_payload = (
            {
                "employer_scope": {
                    "status": scope.status,
                    "employer": scope.employer,
                    "parent": scope.parent,
                }
            }
            if scope is not None
            else {}
        )

        if bucket == BUCKET_FAILURE:
            # A failed judgment is not a verdict about the person; record a retryable
            # stage failure so a later session can try again.
            stats.failed += 1
            engine.runtime.finish_stage_failure(
                attempt_id=attempt_id,
                envelope=envelope,
                stage="full",
                error_or_failure_decision=decision,
                extra_payload={"candidate_record": candidate.to_dict()},
            )
        else:
            fingerprint_payload = (
                {JUDGMENT_FINGERPRINT_KEY: self._judgment_fingerprint}
                if self._judgment_fingerprint
                else {}
            )
            split_payload = (
                {REJUDGE_SPLIT_KEY: {"samples": list(split_samples)}}
                if split_samples
                else {}
            )
            engine.runtime.finish_stage_success(
                attempt_id=attempt_id,
                envelope=envelope,
                stage="full",
                decision=decision,
                profile_summary=candidate.to_profile_summary(),
                extra_payload={
                    "candidate_record": candidate.to_dict(),
                    **scope_payload,
                    **fingerprint_payload,
                    **split_payload,
                },
            )

        # The judgment row is appended BEFORE the side effect, so a crash
        # between them leaves a verdict whose save is replayable — not a
        # save with no record of why.
        known_label = self._known.match(candidate) or ""
        split_note = _split_note(split_samples)
        self._append_judgment(
            run_id=run_id,
            query=query,
            candidate=candidate,
            decision=decision,
            bucket=bucket,
            known_label=known_label,
            scope=scope_note,
            rejudge="split" if split_samples else ("rejudged" if rejudged else ""),
        )
        line = _verdict_line(bucket, candidate, decision)
        for note in (scope_note, split_note):
            if note:
                line = f"{line} · {_flatten(note)}"
        print(line, flush=True)

        if bucket == BUCKET_SAVE:
            if self._save_to_workspace(
                run_id=run_id,
                envelope=envelope,
                engine=engine,
                candidate=candidate,
                decision=decision,
                attempt_id=attempt_id,
            ):
                stats.saved += 1
                query.saves.append(candidate.identity_key)
        elif bucket == BUCKET_REVIEW:
            stats.review += 1
        elif bucket == BUCKET_REJECT:
            stats.rejected += 1
        return bucket

    def _blacklisted_employer(
        self, candidate: ExaCandidate, *, during: str
    ) -> tuple[str, tuple[str, str, str]] | None:
        """``(gate_path, (blocked_name, field_name, field_value))`` or None.

        The current-employer blacklist first, so a current employee keeps
        its ``employer_blacklist`` stamp; then the alumni blacklist over the
        whole work history.

        Fail-soft by contract. The judge still carries both rules in its
        prompt, so a fault here degrades to that path with one console
        line; it never raises into the run.
        """

        try:
            hit = employer_blacklist_match(
                candidate, getattr(self.brief, "employer_blacklist", None)
            )
            if hit is not None:
                return GATE_PATH, hit
            alumni = getattr(self.brief, "alumni_blacklist", None)
            hit = alumni_blacklist_match(
                candidate, alumni if isinstance(alumni, (list, tuple)) else None
            )
            if hit is not None:
                return ALUMNI_GATE_PATH, hit
        except Exception as exc:  # noqa: BLE001 — the gate must never raise
            logger.warning(
                "exa employer-blacklist %s failed for %s: %s",
                during,
                candidate.identity_key,
                exc,
            )
            what = {
                "gate": "judging normally",
                "replay check": "replaying the save",
            }.get(during, "left on the slate")
            print(
                _one_line(
                    f"    [exa] {candidate.name or '(unnamed)'} — employer "
                    f"blacklist check failed ({type(exc).__name__}); {what}"
                ),
                flush=True,
            )
            return None
        return None

    def _location_excluded(
        self, candidate: ExaCandidate, *, during: str
    ) -> LocationVerdict | None:
        """The location verdict when it puts the person OUTSIDE the scope, else None.

        None when the brief has no ``exa_location_scope``, when the card
        location is inside it, and when it resolves to no country: an empty
        or unreadable location is the judge's to weigh, never a reason to
        drop someone. Fail-soft like the blacklist check: a fault degrades to
        ordinary judgment with one console line.
        """

        if self.location_scope is None:
            return None
        try:
            verdict = self.location_scope.check(candidate)
        except Exception as exc:  # noqa: BLE001 — the gate must never raise
            logger.warning(
                "exa location %s failed for %s: %s", during, candidate.identity_key, exc
            )
            what = {
                "gate": "judging normally",
                "replay check": "replaying the save",
            }.get(during, "left on the slate")
            print(
                _one_line(
                    f"    [exa] {candidate.name or '(unnamed)'} — location check "
                    f"failed ({type(exc).__name__}); {what}"
                ),
                flush=True,
            )
            return None
        return verdict if verdict.status == LOCATION_OUTSIDE else None

    def _settle_location_gated(
        self,
        *,
        run_id: int,
        query: ExaSearchQuery,
        candidate: ExaCandidate,
        engine: CandidateExecutionEngine,
        envelope,
        attempt_id: int | None,
        verdict: LocationVerdict,
        stats: ExaRunStats,
    ) -> None:
        """Record the location gate's REJECT through the writes a verdict takes.

        The same order as :meth:`_settle_blacklisted`: store row, then
        ``judgments.jsonl``, then the console line.
        """

        assert self.location_scope is not None
        decision = location_gate_decision(candidate, verdict, self.location_scope)
        engine.runtime.finish_stage_success(
            attempt_id=attempt_id,
            envelope=envelope,
            stage="full",
            decision=decision,
            profile_summary=candidate.to_profile_summary(),
            extra_payload={
                "candidate_record": candidate.to_dict(),
                "gate": LOCATION_GATE_PATH,
                "gate_match": {
                    "location": verdict.location,
                    "country": verdict.country,
                },
            },
        )
        self._append_judgment(
            run_id=run_id,
            query=query,
            candidate=candidate,
            decision=decision,
            bucket=BUCKET_REJECT,
            known_label=self._known.match(candidate) or "",
            gate=LOCATION_GATE_PATH,
        )
        stats.location_gated += 1
        print(
            _location_gate_line(candidate, verdict.note(self.location_scope)),
            flush=True,
        )

    def _attainability(
        self, candidate: ExaCandidate, rules: AttainabilityRules
    ) -> tuple[str, str] | None:
        """``(tier, reason)`` for the slate, or None when the check faults.

        Fail-soft like the blacklist check: a fault leaves the person where
        the judge put them, unmarked, with one console line.
        """

        try:
            companies, headlines = current_employer_signals(candidate)
            return classify_signals(
                companies=companies,
                headlines=headlines,
                titles=current_title_signals(candidate),
                rules=rules,
            )
        except Exception as exc:  # noqa: BLE001 — a slate check must never raise
            logger.warning(
                "exa attainability check failed for %s: %s", candidate.identity_key, exc
            )
            print(
                _one_line(
                    f"    [exa] {candidate.name or '(unnamed)'} — attainability "
                    f"check failed ({type(exc).__name__}); left unmarked"
                ),
                flush=True,
            )
            return None

    def _scope_verdict(
        self, candidate: ExaCandidate, *, during: str
    ) -> ScopeVerdict | None:
        """The scope check's verdict, or None when the brief has no scope.

        Fail-soft like the blacklist check: a fault leaves the person where
        the judge put them, with one console line.
        """

        if self._scope is None:
            return None
        try:
            return self._scope.check(candidate)
        except Exception as exc:  # noqa: BLE001 — the scope check must never raise
            logger.warning(
                "exa scope %s failed for %s: %s", during, candidate.identity_key, exc
            )
            print(
                _one_line(
                    f"    [exa] {candidate.name or '(unnamed)'} — scope check "
                    f"failed ({type(exc).__name__}); verdict left as judged"
                ),
                flush=True,
            )
            return None

    def _scope_for_verdict(
        self,
        candidate: ExaCandidate | None,
        decision: Any,
        bucket: str,
        *,
        during: str,
    ) -> ScopeVerdict | None:
        """The scope verdict for a save-family verdict; None for anything else."""

        if candidate is None or bucket not in (BUCKET_SAVE, BUCKET_REVIEW):
            return None
        if getattr(decision, "decision", "") not in SAVE_DECISION_FAMILY:
            return None
        return self._scope_verdict(candidate, during=during)

    def _settle_blacklisted(
        self,
        *,
        run_id: int,
        query: ExaSearchQuery,
        candidate: ExaCandidate,
        engine: CandidateExecutionEngine,
        envelope,
        attempt_id: int | None,
        hit: tuple[str, str, str],
        stats: ExaRunStats,
        path: str = GATE_PATH,
    ) -> None:
        """Record the gate's REJECT through the same writes a verdict takes.

        Store row, then ``judgments.jsonl``, then the console line, in the
        order :meth:`_judge_one` uses, so resume, the slate, and the
        projection read a gated person exactly as they read a judged one.
        """

        blocked, matched_field, matched_value = hit
        decision = gate_decision(candidate, blocked, path)
        engine.runtime.finish_stage_success(
            attempt_id=attempt_id,
            envelope=envelope,
            stage="full",
            decision=decision,
            profile_summary=candidate.to_profile_summary(),
            extra_payload={
                "candidate_record": candidate.to_dict(),
                "gate": path,
                "gate_match": {
                    "blocked": blocked,
                    "field": matched_field,
                    "value": matched_value,
                },
            },
        )
        self._append_judgment(
            run_id=run_id,
            query=query,
            candidate=candidate,
            decision=decision,
            bucket=BUCKET_REJECT,
            known_label=self._known.match(candidate) or "",
            gate=path,
        )
        if path == ALUMNI_GATE_PATH:
            stats.alumni_gated += 1
        else:
            stats.blacklist_gated += 1
        print(_gate_line(candidate, blocked, path), flush=True)

    def _save_to_workspace(
        self,
        *,
        run_id: int,
        envelope,
        engine: CandidateExecutionEngine,
        candidate: ExaCandidate,
        decision,
        attempt_id: int | None,
    ) -> bool:
        """Write the workspace row and return whether it landed.

        Stats and ``saves.jsonl`` follow this result, not the verdict. A SAVE whose
        destination fails is not a save, and the report must preserve that distinction.
        """

        assert self.store is not None
        row = self.store.get_candidate(
            source=SOURCE, brief_id=self.brief_id, identity_key=candidate.identity_key
        )
        candidate_id = int(row["id"]) if row else 0
        idempotency_key = f"exa:{self.brief_id}:{candidate.identity_key}:workspace"
        begun = engine.runtime.begin_candidate_side_effect(
            envelope=envelope,
            attempt_id=attempt_id,
            effect_type=WORKSPACE_EFFECT_TYPE,
            idempotency_key=idempotency_key,
            payload={"identity_key": candidate.identity_key},
        )
        if not begun.get("should_execute"):
            return False
        side_effect_id = int(begun["side_effect"]["id"])

        result = CandidateWorkspaceSaveDestination().save(
            envelope=_WorkspaceEnvelope(
                brief_id=self.brief_id,
                candidate_id=candidate_id,
                runtime_state_path=self.db_path,
            ),
            decision=decision,
            evidence_payload={
                "surface_type": "exa_web_evidence",
                "evidence_urls": candidate.evidence_urls(),
                "verify_before_outreach": True,
                "data_sufficiency": candidate.data_sufficiency,
                "judgment": decision.to_dict(),
            },
            attempt_id=attempt_id,
        )
        succeeded = result.status == "succeeded"
        engine.runtime.complete_candidate_side_effect(
            side_effect_id=side_effect_id,
            status="succeeded" if succeeded else "failed",
            payload={**(result.payload or {}), "error": result.error},
        )
        if not succeeded:
            return False
        append_jsonl(
            self.saves_path,
            {
                "run_id": run_id,
                "identity_key": candidate.identity_key,
                "person_key": exa_person_key(candidate),
                "name": candidate.name,
                "profile_url": _profile_url(candidate),
                "decision": decision.decision,
                "destination": result.destination,
                "status": result.status,
                "evidence_urls": candidate.evidence_urls(),
                "verify_before_outreach": True,
            },
        )
        return True

    # ── Projections ──────────────────────────────────────────────────

    def _record_skips(
        self,
        query: ExaSearchQuery,
        deduped,
        stats: ExaRunStats,
        run_id: int,
    ) -> None:
        """Record every dropped and flagged person with its reason.

        Flagged people remain in ``kept``; refusing a weak-key fold makes the
        possible collision visible while keeping both records.
        """

        for candidate, reason in deduped.skipped:
            _bump(stats.skipped, _reason_bucket(reason))
            append_jsonl(
                self.skipped_path,
                {
                    "run_id": run_id,
                    "query_id": query.id,
                    "firm": query.firm,
                    "identity_key": candidate.identity_key,
                    "name": candidate.name,
                    "reason": reason,
                    "reason_class": _reason_bucket(reason),
                    "kept": False,
                },
            )
        for candidate, reason in deduped.flagged:
            append_jsonl(
                self.skipped_path,
                {
                    "run_id": run_id,
                    "query_id": query.id,
                    "firm": query.firm,
                    "identity_key": candidate.identity_key,
                    "name": candidate.name,
                    "reason": reason,
                    "reason_class": _reason_bucket(reason),
                    "kept": True,
                },
            )

    def _record_drops(
        self,
        query: ExaSearchQuery,
        dropped: dict[str, int],
        deduped,
        stats: ExaRunStats,
    ) -> None:
        """Acquisition drops onto the work unit's ``metrics`` column.

        Coverage notes have to survive a resume, and the drop counts are
        the only part of a query's story that is not already a counter
        column. Writing them to ``metrics_json`` puts them on the plane
        that gets cloned into the next session's run.
        """

        bucket = self._query_drops.setdefault(str(query.id), {})
        for reason, count in (dropped or {}).items():
            if count:
                bucket[reason] = bucket.get(reason, 0) + int(count)
        for _candidate, reason in deduped.skipped:
            key = _reason_bucket(reason)
            bucket[key] = bucket.get(key, 0) + 1

    def _load_query_drops(self, run_id: int) -> dict[str, dict]:
        assert self.store is not None
        drops: dict[str, dict] = {}
        for row in self.store.list_work_units(run_id, kind=EXA_QUERY_KIND):
            metrics = _json_object(row.get("metrics_json"))
            recorded = metrics.get("drops")
            if isinstance(recorded, dict) and recorded:
                drops[str(row["source_unit_id"])] = {
                    str(k): int(v) for k, v in recorded.items()
                }
        return drops

    def _append_judgment(
        self,
        *,
        run_id: int,
        query: ExaSearchQuery,
        candidate: ExaCandidate,
        decision,
        bucket: str,
        known_label: str,
        gate: str = "",
        scope: str = "",
        rejudge: str = "",
    ) -> None:
        semantic = getattr(decision, "semantic_evidence", None) or {}
        append_jsonl(
            self.judgments_path,
            {
                "run_id": run_id,
                "query_id": query.id,
                "lane": query.lane,
                "firm": query.firm,
                "identity_key": candidate.identity_key,
                "name": candidate.name,
                "title": candidate.current_title,
                "current_firm": candidate.current_firm,
                "location": candidate.location,
                "linkedin_url": candidate.linkedin_url,
                "evidence_urls": candidate.evidence_urls(),
                "bucket": bucket,
                "decision": decision.decision,
                "confidence": decision.confidence,
                "rationale": decision.rationale or "",
                "review_reason_code": getattr(decision, "review_reason_code", "") or "",
                "case_against": str(semantic.get("case_against") or ""),
                "known_label": known_label,
                # The deterministic gate that produced this row, or "" for
                # a model verdict. Machine-readable so a consumer never has
                # to parse the rationale to tell the two apart.
                "gate": gate,
                # The scope check's reason when it moved or flagged this
                # verdict ("scope: current employer X"), else "". The
                # decision field above is still the judge's.
                "scope": scope,
                # "rejudged" when this row re-judges a verdict made under
                # earlier criteria, "split" when its two samples disagreed
                # and it settled as worth-a-look, else "".
                "rejudge": rejudge,
                "verify_before_outreach": True,
                "timestamp": _now_utc_iso(),
            },
        )

    # ── Close-out ────────────────────────────────────────────────────

    def _close_out(
        self,
        *,
        run_id: int,
        progress: ExaProgress,
        stats: ExaRunStats,
        meter: ExaCostMeter,
        run_status: str,
        stop_reason: str,
        stop_detail: str,
        primary_error: bool,
    ) -> None:
        """Artifacts first, ``finish_run`` last, degradation reported.

        A failure here is DEGRADATION, not a verdict on the campaign: the
        loop already decided the run status, and a report that failed to
        render does not turn a completed run into a failed one. But it
        must not be silent either — the previous shape logged at WARNING,
        which nothing on the operator's surface reads. Each failure gets
        one console line and a durable event.
        """

        degraded: list[tuple[str, str]] = []
        for label, step in (
            ("progress", lambda: self._sync_final_progress(run_id, progress)),
            (
                "costs",
                lambda: self._write_costs(run_id, stats, meter, stop_detail),
            ),
            ("report", lambda: self._write_report(run_id, stats)),
            ("snapshot", lambda: self._finalize_snapshot(run_id)),
        ):
            try:
                step()
            except Exception as exc:  # noqa: BLE001 — never mask the real error
                degraded.append((label, _one_line(f"{type(exc).__name__}: {exc}")))
                logger.warning("exa close-out step %s failed: %s", label, exc)

        try:
            self._finish_run(run_id, run_status, stop_reason)
        except Exception as exc:  # noqa: BLE001
            degraded.append(("finish_run", _one_line(f"{type(exc).__name__}: {exc}")))
            logger.warning("exa close-out step finish_run failed: %s", exc)

        for label, reason in degraded:
            if not primary_error:
                print(
                    _one_line(f"    [exa] close-out degraded — {label}: {reason}"),
                    flush=True,
                )
            try:
                if self.bridge is not None:
                    self.bridge.record_event(
                        event_type="close_out_degraded",
                        payload={"step": label, "reason": reason},
                        run_id=run_id,
                    )
            except Exception:  # noqa: BLE001 — the event is the last thing tried
                pass

        self._report_run_degraded(run_id=run_id, stats=stats)

    def _report_run_degraded(self, *, run_id: int, stats: ExaRunStats) -> None:
        """Report when at least half of this session's judgments failed.

        A failed judgment remains retryable and is not a verdict. A completed-looking
        slate with mostly failed judgments is a transport outage, so record
        ``run_degraded`` after artifacts are written for operator visibility.
        """

        if stats.judged <= 0:
            return
        if stats.failed / stats.judged < RUN_DEGRADED_FAILURE_RATIO:
            return

        print(
            _one_line(
                f"    [exa] run degraded — {stats.failed} of {stats.judged} "
                "judgments failed; verdicts are not trustworthy"
            ),
            flush=True,
        )
        try:
            if self.bridge is not None:
                self.bridge.record_event(
                    event_type="run_degraded",
                    payload={
                        "judged": stats.judged,
                        "failed": stats.failed,
                        "failure_ratio": round(stats.failed / stats.judged, 4),
                        "threshold": RUN_DEGRADED_FAILURE_RATIO,
                    },
                    run_id=run_id,
                )
        except Exception:  # noqa: BLE001 — telemetry never breaks close-out
            pass

    def _sync_final_progress(self, run_id: int, progress: ExaProgress) -> None:
        if self.bridge is not None:
            self._sync(run_id, progress)

    def _finish_run(self, run_id: int, run_status: str, stop_reason: str) -> None:
        if self.store is not None:
            self.store.finish_run(run_id, run_status, stop_reason=stop_reason)

    def _write_costs(
        self,
        run_id: int,
        stats: ExaRunStats,
        meter: ExaCostMeter,
        stop_detail: str,
    ) -> None:
        """Session spend on the run-dir plane, campaign spend beside it.

        ``pipeline_end.cost_usd`` stays SESSION-scoped because that is
        what a per-run-dir rollup means — ``shared.cost_rollup`` reads
        the last ``pipeline_end`` in this state dir's run log and
        attributes it to this run's snapshot. The campaign figures come
        from the append-only receipts, which span every session, and ride
        alongside rather than replacing it.
        """

        session_exa = float(meter.committed_usd)
        session_llm = _sum_token_cost_log_usd(
            self.usage_log_path,
            run_id=resolve_cost_log_run_id(self.usage_log_path, run_id),
            exclude_rows_with=("shadow_stage",),
        )
        campaign_llm = _sum_token_cost_log_usd(
            self.usage_log_path, exclude_rows_with=("shadow_stage",)
        )
        campaign_exa = _sum_exa_usage_usd(self.exa_usage_path)

        stats.exa_cost_usd = session_exa
        stats.llm_cost_usd = float(session_llm or 0.0)
        stats.campaign_exa_cost_usd = float(
            campaign_exa if campaign_exa is not None else session_exa
        )
        stats.campaign_llm_cost_usd = float(campaign_llm or 0.0)

        payload = {
            "cost_usd": round(session_exa + float(session_llm or 0.0), 6),
            "exa_cost_usd": round(session_exa, 6),
            "llm_cost_usd": None if session_llm is None else round(session_llm, 6),
            "campaign_exa_cost_usd": round(stats.campaign_exa_cost_usd, 6),
            "campaign_llm_cost_usd": round(stats.campaign_llm_cost_usd, 6),
            "saves": stats.saved,
            "stats": stats.to_dict(),
        }
        if stop_detail:
            payload["stop_detail"] = stop_detail
        if self.bridge is not None:
            self.bridge.record_event(
                event_type="pipeline_end", payload=payload, run_id=run_id
            )
        # ``shared.cost_rollup._cost_from_pipeline_end`` reads the FILE,
        # not the events table: `{"event": "pipeline_end", "cost_usd": ...}`.
        log_event(self.log_path, "pipeline_end", **payload)
        rollup = aggregate_cost_for_run({SOURCE: self.output_dir})
        write_cost_rollup_sidecar(rollup, run_dir=self.output_dir)

    def _write_report(self, run_id: int, stats: ExaRunStats) -> None:
        view = self.build_run_view(run_id, stats)
        markdown = render_slate(view)
        # Keep one campaign-wide slate and overwrite it each session. Run snapshots
        # refer to this current slate, so a per-run copy would leave stale siblings.
        self.slate_path.write_text(markdown)
        (self.output_dir / "run-report.md").write_text(markdown)
        payload = structured_report_payload(
            view, role_title=str(self.brief.role_title or "")
        )
        report = StructuredRunReport.from_dict(payload)
        (self.output_dir / "run-report.json").write_text(
            json.dumps(report.to_dict(), indent=2)
        )

    def _finalize_snapshot(self, run_id: int) -> Path:
        from market_intelligence.run_snapshots import finalize_run_snapshot

        return finalize_run_snapshot(
            source=SOURCE,
            brief_path=self.brief_path,
            state_dir=self.output_dir,
            run_id=run_id,
        )

    # ── Report assembly ──────────────────────────────────────────────

    def _left_short_list(
        self, where: dict[str, tuple[SlateEntry, str]]
    ) -> list[SlateEntry]:
        """Everyone a re-judge took off the short list who is still off it.

        Campaign-wide, from the re-judge events: the latest event per person
        whose old verdict was a short-list one names the verdict they held,
        and ``where`` says where the current verdict put them. A person a
        later re-judge put back on the short list is not listed; a person
        the slate files as already known is in that section instead.
        """

        assert self.bridge is not None
        previous: dict[str, dict] = {}
        for payload in self.bridge.list_event_payloads(REJUDGE_EVENT):
            old_place = payload.get("old_slate_bucket", payload.get("old_bucket"))
            if old_place == BUCKET_SAVE:
                key = str(payload.get("identity_key") or "")
                previous.pop(key, None)
                previous[key] = _json_object(payload.get("old_decision"))
        out: list[SlateEntry] = []
        for key, old in previous.items():
            placed = where.get(key)
            if placed is None or placed[1] in ("short list", "already known"):
                continue
            entry, now = placed
            entry.previous_decision = str(old.get("decision") or "")
            confidence = old.get("confidence")
            entry.previous_confidence = (
                float(confidence)
                if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                else None
            )
            entry.now = now
            out.append(entry)
        return out

    def build_run_view(self, run_id: int, stats: ExaRunStats) -> RunView:
        """Assemble the slate from the canonical plane.

        Read candidates and their stored terminal decisions from SQLite-backed state.
        Do not use ``judgments.jsonl`` or session counters: the projection may lag and
        can retain earlier decisions after rejudgment.
        """

        view = RunView(
            brief_name=self.brief_name,
            run_id=run_id,
            session_judged=stats.judged,
            session_exa_cost_usd=stats.exa_cost_usd,
            session_llm_cost_usd=stats.llm_cost_usd,
            campaign_exa_cost_usd=stats.campaign_exa_cost_usd,
            campaign_llm_cost_usd=stats.campaign_llm_cost_usd,
            paused=stats.paused,
            location_scope_label=(
                self.location_scope.label if self.location_scope is not None else ""
            ),
        )

        # Which query placed each person, so an archetype row can say
        # what its sentence actually caught. Keyed by the query's STABLE
        # identity (lane + coverage key) off the candidate's own record
        # — not this session's counters, so a person judged three
        # sessions ago still counts toward the query that found them,
        # and not the ordinal id, which moves the moment the brief drops
        # a lane or an archetype. Records written before that key
        # existed fall back to the id (`_LEGACY_QUERY_KEY`).
        placed: dict[str, dict[str, int]] = {}
        # The loader validated the block; a malformed one fails the load.
        attainability_rules = AttainabilityRules.from_brief(self.brief)
        # Where each terminal person landed on this slate, so the re-judge
        # departures can be read after the buckets are built.
        where: dict[str, tuple[SlateEntry, str]] = {}

        if self.bridge is not None:
            entries = self.bridge.list_campaign_candidates()
            # Everybody this campaign found, so a known "Larissa M." is
            # matched only when one of them fits it.
            self._known.set_pool(entry.display_name for entry in entries)
            for entry in entries:
                if entry.state in ("failed_retryable", "failed_terminal"):
                    view.failed += 1
                    continue
                if not entry.is_terminal:
                    continue
                decision = _decision_from_entry(entry)
                if decision is None:
                    continue
                slate_entry = _entry_from_campaign(entry, decision)
                candidate = _candidate_from_record(entry)
                if candidate is not None:
                    slate_entry.known_label = self._known.match(candidate) or ""
                    tier = self._attainability(candidate, attainability_rules)
                    if tier is not None:
                        slate_entry.attainability, slate_entry.attainability_reason = tier
                bucket = classify_verdict(decision)
                if is_location_gate_decision(decision):
                    view.gated_location += 1
                    where[entry.identity_key] = (
                        slate_entry, _now_label(BUCKET_REJECT, "location")
                    )
                elif is_gate_decision(decision):
                    if decision.path == ALUMNI_GATE_PATH:
                        view.gated_alumni += 1
                    else:
                        view.gated_employer += 1
                    where[entry.identity_key] = (
                        slate_entry, _now_label(BUCKET_REJECT, "blacklist")
                    )
                elif bucket in (BUCKET_SAVE, BUCKET_REVIEW) and candidate is not None:
                    # A stored save the judge let through at a blacklisted
                    # current employer, or of anyone who has ever worked at
                    # an alumni-blacklisted one, comes off the slate here,
                    # with no re-judgment; the verdict in the store is
                    # unchanged.
                    gate = self._blacklisted_employer(candidate, during="slate check")
                    if gate:
                        gate_path, hit = gate
                        slate_entry.excluded_reason = _excluded_reason(hit, gate_path)
                        if gate_path == ALUMNI_GATE_PATH:
                            view.excluded_alumni.append(slate_entry)
                        else:
                            view.excluded_employer.append(slate_entry)
                        where[entry.identity_key] = (
                            slate_entry, _now_label(BUCKET_REJECT, "blacklist")
                        )
                        continue
                    # A stored save whose card location the brief's location
                    # scope rules out leaves the slate the same way; the
                    # verdict in the store is unchanged.
                    located = self._location_excluded(candidate, during="slate check")
                    if located is not None:
                        slate_entry.excluded_reason = located.note(self.location_scope)
                        view.excluded_location.append(slate_entry)
                        where[entry.identity_key] = (
                            slate_entry, _now_label(BUCKET_REJECT, "location")
                        )
                        continue
                if not is_gate_decision(decision):
                    # Same scope check the live path ran, applied to the
                    # stored verdict, so a brief that opts in mid-campaign
                    # moves earlier saves too. The store is unchanged.
                    scope = self._scope_for_verdict(
                        candidate, decision, bucket, during="slate check"
                    )
                    if scope is not None and scope.note:
                        slate_entry.scope_note = scope.note
                        if bucket == BUCKET_SAVE:
                            bucket = BUCKET_REVIEW
                            slate_entry.scope_moved = True
                            view.scope_moved += 1
                # A re-judge whose samples split is worth a look, with the
                # reject sample's reason; the stored decision is the save's.
                if entry.rejudge_split and bucket in (BUCKET_SAVE, BUCKET_REVIEW):
                    samples = entry.rejudge_split.get("samples") or ()
                    slate_entry.rejudge_note = _split_note(samples)
                    slate_entry.rejudge_reason = _split_reason(samples)
                    bucket = BUCKET_REVIEW
                if slate_entry.known_label:
                    view.already_known.append(slate_entry)
                    where.setdefault(entry.identity_key, (slate_entry, "already known"))
                elif bucket == BUCKET_SAVE:
                    view.short_list.append(slate_entry)
                    _bump_query_bucket(placed, candidate, "short_list")
                    where[entry.identity_key] = (slate_entry, "short list")
                elif bucket == BUCKET_REVIEW:
                    view.worth_a_look.append(slate_entry)
                    _bump_query_bucket(placed, candidate, "worth_a_look")
                    where[entry.identity_key] = (slate_entry, _now_label(BUCKET_REVIEW, ""))
                else:
                    view.rejected += 1
                    # A gate's reject already recorded which gate it was.
                    where.setdefault(entry.identity_key, (slate_entry, "rejected"))
            view.left_short_list = self._left_short_list(where)

        for row in _safe_jsonl(self.skipped_path):
            if row.get("kept"):
                continue
            _bump(view.skipped, str(row.get("reason_class") or "skipped"))

        # Two coverage planes, split on the work unit's family key. A
        # firm's three lanes are three angles on one firm and belong on
        # one row; an archetype query is its own hypothesis and gets
        # its own. Folding both into the firm map put six archetypes
        # under one "(no firm)" row, where a query that found nine
        # people and one that found none read identically.
        coverage: dict[str, FirmCoverage] = {}
        archetypes: dict[str, ArchetypeCoverage] = {}
        if self.store is not None:
            for row in self.store.list_work_units(run_id, kind=EXA_QUERY_KIND):
                family_key = str(row.get("family_key") or "")
                archetype_id = _archetype_id_from_family_key(family_key)
                if archetype_id:
                    entry: FirmCoverage | ArchetypeCoverage = archetypes.setdefault(
                        archetype_id, ArchetypeCoverage(archetype_id=archetype_id)
                    )
                else:
                    entry = coverage.setdefault(
                        family_key, FirmCoverage(firm=family_key)
                    )
                entry.queries += 1
                entry.results += int(row.get("result_count") or 0)
                entry.candidates += int(row.get("candidates_discovered") or 0)
                if row.get("status") == "skipped":
                    entry.skipped += 1
                    _bump(
                        view.skipped_queries,
                        _skipped_query_reason(str(row.get("notes") or "")),
                    )
                if isinstance(entry, ArchetypeCoverage):
                    buckets = _placed_for_row(placed, row)
                    entry.short_list += buckets.get("short_list", 0)
                    entry.worth_a_look += buckets.get("worth_a_look", 0)
                drops = _json_object(row.get("metrics_json")).get("drops")
                if isinstance(drops, dict):
                    for reason, count in drops.items():
                        entry.drops[str(reason)] = entry.drops.get(
                            str(reason), 0
                        ) + int(count or 0)
                        _bump_by(view.dropped, str(reason), int(count or 0))
                # A skipped query never ran. Counting it as run put the
                # campaign's "queries run" figure ahead of its actual
                # spend the moment a brief edit retired anything.
                if row.get("status") in ("done", "error"):
                    view.queries_run += 1
                view.results_total += int(row.get("result_count") or 0)
                view.candidates_total += int(row.get("candidates_discovered") or 0)
        view.coverage = list(coverage.values())
        view.archetype_coverage = list(archetypes.values())
        return view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_extractor():
    from exa.extraction import extract_people_from_page

    return extract_people_from_page


def _run_budget_usd() -> float:
    from shared.config import EXA_RUN_BUDGET_USD

    return float(EXA_RUN_BUDGET_USD)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _one_line(text: str) -> str:
    """Collapse a console line while preserving its four-space indent.

    Keep the console indent before flattening embedded newlines and repeated whitespace.
    """

    raw = str(text)
    indent = raw[: len(raw) - len(raw.lstrip(" "))]
    return indent + " ".join(raw.replace("\r", " ").replace("\n", " ").split())


def _profile_url(candidate: ExaCandidate) -> str:
    return candidate.linkedin_url or (candidate.evidence_urls() or [""])[0]


def _archetype_id_from_family_key(family_key: str) -> str:
    """The archetype slug behind an ``archetype:<id>`` family key, or "".

    The prefix is what keeps the two coverage namespaces apart, so a
    firm someone one day names "a01-mbb-to-vc" cannot land in the
    archetype block. A bare ``archetype:`` with nothing after it reads
    as no archetype rather than as one with an empty name.
    """

    prefix = "archetype:"
    if not family_key.startswith(prefix):
        return ""
    return family_key[len(prefix) :].strip()


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Separator inside a composite placement key. A unit separator cannot
# occur in a lane name or a firm name, so "people_search" + "Benchmark"
# can never collide with a firm someone named "people_search Benchmark".
_KEY_SEP = "\x1f"


def _coverage_placement_key(lane: str, coverage_key: str) -> str:
    return f"{lane}{_KEY_SEP}{coverage_key}"


def _legacy_placement_key(query_id: int) -> str:
    return f"#id{_KEY_SEP}{query_id}"


def _bump_query_bucket(
    placed: dict[str, dict[str, int]], candidate: Any, bucket: str
) -> None:
    """Credit a placement to the query that discovered the person.

    Keyed on ``(source_lane, source_coverage_key)`` — the identity that
    survives a brief edit. A record written before that field existed
    carries only the ordinal ``source_query_id``, and is keyed on that
    instead; the two namespaces are kept apart by the ``#id`` prefix so
    an old record can never be credited to a new one's query.

    Silent when the record is absent — a row whose payload never
    survived cannot say which query found it, and guessing would
    attribute someone's short-list place to whichever query happens to
    sort first.
    """

    if candidate is None:
        return
    coverage_key = getattr(candidate, "source_coverage_key", "") or ""
    lane = getattr(candidate, "source_lane", "") or ""
    if coverage_key:
        key = _coverage_placement_key(lane, coverage_key)
    else:
        query_id = getattr(candidate, "source_query_id", None)
        if not isinstance(query_id, int):
            return
        key = _legacy_placement_key(query_id)
    placed.setdefault(key, {})[bucket] = placed.setdefault(key, {}).get(bucket, 0) + 1


def _placed_for_row(
    placed: dict[str, dict[str, int]], row: dict
) -> dict[str, int]:
    """This work unit's placements, under both key generations.

    The row's ``family_key`` IS the query's coverage key (the bridge
    writes it there), so a row and a candidate agree without the two
    having to meet. Legacy id-keyed placements are added on top rather
    than instead: one campaign can hold records from both sides of this
    change, and a person is only ever counted under one of the two keys.
    """

    out: dict[str, int] = {}
    keys = []
    coverage_key = str(row.get("family_key") or "")
    if coverage_key:
        keys.append(
            _coverage_placement_key(str(row.get("domain_lane") or ""), coverage_key)
        )
    keys.append(_legacy_placement_key(_int_or(row.get("source_unit_id"), -1)))
    for key in keys:
        for bucket, count in (placed.get(key) or {}).items():
            out[bucket] = out.get(bucket, 0) + int(count)
    return out


# How a retired query's stored note reads on the slate. The note is
# machine state (it is also the marker a requeue matches on); the phrase
# is what a recruiter reads, and the two should not be forced to be the
# same string.
_SKIPPED_QUERY_REASONS = {
    LANE_DROPPED_NOTE: "lane dropped from the brief",
    PLAN_DROPPED_NOTE: "no longer in the brief's plan",
    PAYLOAD_LOST_NOTE: "payload lost, not in the current plan",
}


def _skipped_query_reason(note: str) -> str:
    return _SKIPPED_QUERY_REASONS.get(note, note or "retired from the brief")


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _bump_by(counter: dict[str, int], key: str, amount: int) -> None:
    if amount:
        counter[key] = counter.get(key, 0) + amount


def _reason_bucket(reason: str) -> str:
    """The reason CLASS — the part before the first colon.

    ``do_not_save:Avery Holt: on the brief's list`` is one reason class
    with a person-specific tail; counting the tails would produce a
    histogram with a bucket per person.
    """

    return str(reason or "").split(":", 1)[0] or "unknown"


def _json_object(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _sum_exa_usage_usd(path: Path) -> float | None:
    """Sum campaign-to-date Exa spend from usage receipts.

    Use ``actual_usd`` when present and ``estimate_usd`` otherwise; missing telemetry
    must never appear free.
    """

    total, _receipts = ledger_spend_usd(path)
    return total


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _decision_from_dict(raw: Any) -> OpusDecision | None:
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        return OpusDecision.from_dict(raw)
    except (TypeError, ValueError):
        return None


def _split_note(samples: Sequence[dict]) -> str:
    """What a split re-judge's reject sample said, or "" for no split."""

    if not samples:
        return ""
    reject = next((s for s in samples if s.get("bucket") == BUCKET_REJECT), {})
    kept = next((s for s in samples if s.get("bucket") != BUCKET_REJECT), {})
    reason = str(reject.get("reject_reason") or "").strip()
    return (
        f"re-judge split: one sample {kept.get('decision') or 'SAVE'}, one REJECT"
        + (f" ({reason})" if reason else "")
    )


def _split_reason(samples: Sequence[dict]) -> str:
    """The split note, then what the reject sample said."""

    note = _split_note(samples)
    reject = next((s for s in samples if s.get("bucket") == BUCKET_REJECT), {})
    rationale = _flatten(reject.get("rationale"))
    return f"{note}: {rationale}" if note and rationale else note


def _now_label(bucket: str, gate: str) -> str:
    if gate == "location":
        return "excluded (location scope)"
    if gate:
        return "excluded (blacklisted employer)"
    if bucket == BUCKET_REVIEW:
        return "worth a look"
    if bucket == BUCKET_SAVE:
        return "short list"
    return "rejected"


def _superseded_save_keys(path: Path) -> set[str]:
    """Identity keys whose latest ``saves.jsonl`` line withdraws the save."""

    latest: dict[str, str] = {}
    for row in _safe_jsonl(path):
        key = str(row.get("identity_key") or "")
        if key:
            latest[key] = str(row.get("status") or "")
    return {key for key, status in latest.items() if status == SAVE_WITHDRAWN_STATUS}


def _candidate_from_record(entry: CampaignCandidate) -> ExaCandidate | None:
    if not entry.record:
        return None
    try:
        return ExaCandidate.from_dict(entry.record)
    except (TypeError, ValueError, KeyError):
        return None


def _decision_from_entry(entry: CampaignCandidate) -> OpusDecision | None:
    """The stored decision, or one reconstructed from the terminal column.

    A candidate row always carries ``terminal_decision``; the full
    payload can be absent on a row written by an older path. Rebuilding
    the minimum from the column keeps such a person ON the slate under
    their real verdict, with the rationale stated as absent rather than
    invented.
    """

    if entry.decision:
        try:
            return OpusDecision.from_dict(entry.decision)
        except (TypeError, ValueError):
            pass
    if not entry.terminal_decision:
        return None
    return OpusDecision(
        stage="full",
        decision=entry.terminal_decision,
        path="",
        confidence=None,
        rationale="",
        candidate_name=entry.display_name,
        profile_url=entry.profile_url,
    )


def _verdict_line(bucket: str, candidate: ExaCandidate, decision) -> str:
    """The ONE console line a verdict is allowed.

    Each component is collapsed on its own and then assembled, rather
    than collapsing the finished line: the verb is padded so a wall of
    verdicts scans as a column, and a whole-line collapse would eat that
    padding along with the model's stray newlines.
    """

    verb = _VERDICT_VERB.get(bucket, "FAIL  ")
    name = _flatten(candidate.name) or "(unnamed)"
    where = ", ".join(
        _flatten(part)
        for part in (candidate.current_title, candidate.current_firm)
        if _flatten(part)
    )
    confidence = getattr(decision, "confidence", None)
    conf = (
        f" ({confidence:.2f})"
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        else ""
    )
    url = _flatten(_profile_url(candidate))
    line = f"    [exa] {verb} {name}"
    if where:
        line = f"{line} — {where}"
    return f"{line}{conf}{' · ' + url if url else ''}"


def _excluded_reason(hit: tuple[str, str, str], path: str = GATE_PATH) -> str:
    """The slate's reason for an exclusion, quoting the text that matched.

    The quote is omitted when it is just the employer's name. Anything
    longer ("Head of Data at Mercor", "The Charles Babbage Institute") is shown,
    so a reader can spot a match that names the wrong organisation.
    """

    blocked, _field, value = hit
    reason = gate_rationale(blocked, path)
    quoted = _flatten(value)
    if quoted and quoted.casefold() != _flatten(blocked).casefold():
        reason = f'{reason}, matched in "{quoted}"'
    return reason


def _gate_line(candidate: ExaCandidate, blocked: str, path: str = GATE_PATH) -> str:
    """The one console line for a candidate an employer gate rejected."""

    name = _flatten(candidate.name) or "(unnamed)"
    where = ", ".join(
        _flatten(part)
        for part in (candidate.current_title, candidate.current_firm)
        if _flatten(part)
    )
    line = f"    [exa] SKIP   {name}"
    if where:
        line = f"{line} — {where}"
    which = "alumni blacklist" if path == ALUMNI_GATE_PATH else "employer blacklist"
    return f"{line} · {which} ({_flatten(blocked)})"


def _location_gate_line(candidate: ExaCandidate, note: str) -> str:
    """The one console line for a candidate the location gate rejected."""

    name = _flatten(candidate.name) or "(unnamed)"
    where = ", ".join(
        _flatten(part)
        for part in (candidate.current_title, candidate.current_firm)
        if _flatten(part)
    )
    line = f"    [exa] SKIP   {name}"
    if where:
        line = f"{line} — {where}"
    return f"{line} · {_flatten(note)}"


def _flatten(text: Any) -> str:
    return " ".join(str(text or "").split())


def _entry_from_campaign(
    entry: CampaignCandidate, decision: OpusDecision
) -> SlateEntry:
    record = entry.record or {}
    semantic = getattr(decision, "semantic_evidence", None) or {}
    evidence_urls = [
        str(ev.get("url"))
        for ev in (record.get("evidence") or [])
        if isinstance(ev, dict) and ev.get("url")
    ]
    return SlateEntry(
        identity_key=entry.identity_key,
        name=str(record.get("name") or entry.display_name),
        title=str(record.get("current_title") or ""),
        firm=str(record.get("current_firm") or ""),
        location=str(record.get("location") or ""),
        linkedin_url=str(record.get("linkedin_url") or ""),
        evidence_urls=evidence_urls,
        rationale=str(decision.rationale or ""),
        decision=str(decision.decision or ""),
        confidence=decision.confidence,
        review_reason_code=str(getattr(decision, "review_reason_code", "") or ""),
        case_against=str(semantic.get("case_against") or ""),
        verify_before_outreach=bool(record.get("verify_before_outreach", True)),
    )


def _safe_jsonl(path: Path) -> list[dict]:
    try:
        return [row for row in read_jsonl(path) if isinstance(row, dict)]
    except (OSError, ValueError):
        return []


__all__ = [
    "PRE_PASS_STATES",
    "RUN_DEGRADED_FAILURE_RATIO",
    "SLATE_FILENAME",
    "SOURCE",
    "WORKSPACE_EFFECT_TYPE",
    "ExaBriefBlockedError",
    "ExaPipeline",
    "ExaRunStats",
    "ExaStateDirBusyError",
]
