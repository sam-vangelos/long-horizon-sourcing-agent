"""GitHub sourcing pipeline orchestrator.

Connects: strategy → multi-channel search → enrichment → evaluation → save.
No browser needed — pure API-based. Mirrors orchestrator.py's Pipeline pattern.

Usage:
    pipeline = GitHubPipeline(brief_path="config/brief-fdl-brazil-v3.json")
    await pipeline.run()
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
import signal
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from github.client import GitHubAuthError, GitHubClient
from shared.cost_rollup import (
    _cost_per_save_usd,
    _sum_token_cost_log_usd,
    aggregate_cost_for_run,
    write_cost_rollup_sidecar,
)
from github.hubs.crates import CratesHubClient
from github.hubs.npm import NpmHubClient
from github.project_quality import score_project
from github.acquisition import GitHubAcquisitionService
from github.enricher import GitHubEnricher
from github.maintainership import classify as classify_maintainership
from github.maintainership import (
    declared_entries_for_target_projects,
    merge_declared_maintainership,
)
from github.rosters import fetch_repo_roster
from github.schemas import (
    GitHubCandidate,
    GitHubSearchQuery,
    GitHubProgress,
    GitHubBatchReport,
)
from github.side_effects import GitHubSideEffectsService
from github.strategy import (
    adapt_after_batch,
    build_github_adaptation_decision,
    form_github_strategy,
)
from github.work_units import GitHubWorkUnitService
from github.governor import (
    GitHubGovernor,
    GitHubGovernorLimitReached,
    GitHubSessionExpired,
)
from github.query_validator import ExhaustionState
from github.observability import SessionObserver
from shared.contact_discovery import merge_profile_contact
from shared.adaptive import record_adaptation_decision
from shared.observability import observe
from shared.output_paths import resolve_github_state_dir

from shared.failures import ApiBudgetExhaustedError, judgment_failure_decision
from shared.execution import CandidateExecutionEngine
from shared.execution.types import CandidateExecutionEnvelope
from shared.runtime_state.github import PersonKeySet
from shared.runtime_state import GitHubRuntimeStateBridge, RuntimeStateLock, RuntimeStateStore
from shared.runtime_state.store import GITHUB_QUERY_KIND
from shared.safety import RunSafetyCoordinator, RunStopReason
from shared.schemas import CandidateSnippet, CandidateProfileSummary, OpusDecision
from shared.judger import facial_judge, full_judge, init_judger, github_facial_judge, github_facial_judge_batch, github_full_judge, extract_priority_rank, is_failure_decision
from github.outreach import generate_outreach
from shared.storage import append_jsonl, read_json, read_jsonl_set, log_event
from shared.brief_loader import load_brief, Brief
from shared.bias_controls import AlertType, BiasMonitor, DecisionRecord, is_save_decision
from shared.resolvers.ecosystems import EcosystemsResolver
import github.config as gc


# Adaptation batch size — run adaptation after this many queries
_ADAPTATION_BATCH_SIZE = 10

# Checkpoint frequency — save progress after this many enrichments
_CHECKPOINT_EVERY = 10

# GitHub V2 facial batching — keep batches modest to avoid oversized prompts.
_GITHUB_FACIAL_BATCH_SIZE = 10

# Single source for dispatch + exhaustion eligibility.
KNOWN_CHANNELS: frozenset[str] = frozenset({
    "user_search",
    "code_search",
    "repo_mining",
    "org_exploration",
    "topic_search",
    "stargazer_mining",
    "graph_expansion",
    "registry_maintainer_discovery",
    "roster_ingest",
})

_logger = logging.getLogger(__name__)

# Derive/resolver vocabulary → registry-evidence hub labels (schemas comment).
_REGISTRY_ECOSYSTEM_TO_HUB: dict[str, str] = {
    "npmjs.org": "npm",
    "crates.io": "crates",
}


def _github_owner_from_repo_url(repo_url: str | None) -> str | None:
    """Extract the GitHub owner login from a normalized github.com URL."""
    if not repo_url:
        return None
    parsed = urlparse(repo_url)
    parts = [segment for segment in parsed.path.strip("/").split("/") if segment]
    if not parts:
        return None
    return parts[0]


def _github_owner_repo_from_repo_url(repo_url: str | None) -> tuple[str, str] | None:
    """Extract owner/repo from a normalized github.com URL."""
    if not repo_url:
        return None
    parsed = urlparse(repo_url)
    parts = [segment for segment in parsed.path.strip("/").split("/") if segment]
    if len(parts) < 2:
        return None
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return parts[0], repo


class JudgeContractError(RuntimeError):
    """A judge call violated its calling contract (e.g. TypeError) — fatal, never maskable as a judgment failure."""


class GitHubPipeline:
    """Orchestrates the GitHub multi-model sourcing pipeline."""

    def __init__(
        self,
        brief_path: str,
        output_dir: Optional[str] = None,
    ):
        self.brief_path = str(brief_path)
        self.brief_obj = load_brief(brief_path)
        self.state_dir = resolve_github_state_dir(
            brief_path=self.brief_path,
            brief=self.brief_obj,
            state_dir=output_dir,
        )
        self.output_dir = self.state_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize judger with the Brief
        init_judger(self.brief_obj)

        # Output file paths
        self.candidates_path = self.output_dir / "candidates.jsonl"
        self.snippets_path = self.output_dir / "snippets.jsonl"
        self.facial_path = self.output_dir / "facial_judgments.jsonl"
        self.profiles_path = self.output_dir / "profile_summaries.jsonl"
        self.final_path = self.output_dir / "final_judgments.jsonl"
        self.progress_path = self.output_dir / "progress.json"
        self.log_path = self.output_dir / "run_log.jsonl"
        self.bias_path = self.output_dir / "bias_monitor.json"
        self.outreach_path = self.output_dir / "outreach.jsonl"
        self.saves_path = self.output_dir / "saves.jsonl"
        self.runtime_db_path = self.output_dir / "runtime_state.sqlite3"

        # Dedup
        self._seen_usernames: PersonKeySet = PersonKeySet()

        # Stats
        self.stats = {
            "candidates_discovered": 0,
            "candidates_enriched": 0,
            "facial_yes": 0,
            "facial_no": 0,
            "saved": 0,
            "rejected": 0,
            "insufficient": 0,
        }

        # Progress
        self._progress: Optional[GitHubProgress] = None

        # Governor
        self._governor = GitHubGovernor()

        # Bias monitor. P6.4: brief-scoped checkpoint (mirrors linkedin's
        # lifecycle) — load a prior session's decisions/alerts if a
        # checkpoint exists at this output_dir, so alert dedup and
        # consecutive-run streaks survive a resume.
        self._bias_monitor: Optional[BiasMonitor] = None
        if self.brief_obj.has_v2_schema:
            self._bias_monitor = BiasMonitor.from_brief(self.brief_obj._new_brief)
            if self.bias_path.exists():
                self._bias_monitor.load_checkpoint(str(self.bias_path))
        # P6.4: pause severity stops the current query (not print-and-
        # continue theater). Reset per query in _execute_single_query.
        self._bias_pause_active: bool = False

        # Exhaustion state
        self._exhaustion = ExhaustionState()

        # P6.3: batch-scoped accumulators for _build_batch_report honesty.
        # Reset alongside `batch_stats`/`executed_since_batch` in
        # _execute_queries (initial + post-adaptation). `_batch_baseline_stats`
        # snapshots self.stats at batch start so total_rejects/
        # total_insufficient can be read as deltas without github/
        # acquisition.py or github/side_effects.py (which own the actual
        # increments) needing to know about batch boundaries.
        self._batch_baseline_stats: dict = dict(self.stats)
        self._batch_save_candidates: list = []

        # Client reference (set during run)
        self._client: Optional[GitHubClient] = None

        # OSS multi-hub: lazy resolver for project-quality dependents signal.
        self._ecosystems_resolver: Optional[EcosystemsResolver] = None

        # Registry discovery: username → evidence dict for the current query.
        self._registry_evidence_by_username: dict[str, dict] = {}

        # Project-quality score cache keyed by owner/repo for the run.
        self._project_quality_memo: dict[str, dict] = {}

        # Observer
        self._observer: Optional[SessionObserver] = None

        # Shutdown flag
        self._shutdown_requested = False
        self._runtime_state = RuntimeStateStore(self.runtime_db_path)
        self._runtime_lock = RuntimeStateLock(self.output_dir)
        self._runtime_run_id: Optional[int] = None
        self._runtime_bridge = GitHubRuntimeStateBridge(
            store=self._runtime_state,
            output_dir=self.output_dir,
            brief_id=self.brief_obj.id,
            brief_name=self.brief_obj.id,
            brief_path=self.brief_path,
        )
        self._execution_engine = CandidateExecutionEngine(
            store=self._runtime_state,
            output_dir=str(self.output_dir),
            brief_id=self.brief_obj.id,
            source="github",
        )
        self._safety = RunSafetyCoordinator(
            store=self._runtime_state,
            output_dir=self.output_dir,
            source="github",
            brief_id=self.brief_obj.id,
        )
        self._work_unit_service = GitHubWorkUnitService(self)
        self._acquisition_service = GitHubAcquisitionService(self)
        self._side_effects_service = GitHubSideEffectsService(self)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @observe(name="github.run")
    async def run(self, resume: bool = False) -> dict:
        """Run the full autonomous GitHub sourcing pipeline.

        1. Strategy formation (Opus generates GitHub queries from brief)
        2. Multi-channel search execution
        3. Enrichment → facial judgment → full evaluation → save
        4. Adaptation after batches
        """
        self._ensure_runtime_state()
        self._ensure_services()

        # Create session observer
        session_ts = time.strftime("%Y%m%d_%H%M%S")
        session_id = f"{self.brief_obj.id}_{session_ts}"
        self._observer = SessionObserver(session_id, self.output_dir, self.brief_obj)

        # Dedup sets — _seen_usernames holds terminal person keys only (loaded from
        # progress checkpoint on resume).  _in_flight_usernames tracks candidates
        # currently being processed and is deliberately NOT persisted.
        self._seen_usernames = PersonKeySet()
        self._in_flight_usernames: set[str] = set()

        progress: Optional[GitHubProgress] = None
        run_status = "completed"
        stop_reason = RunStopReason.NORMAL
        lock_acquired = False

        try:
            try:
                self._runtime_lock.acquire()
                lock_acquired = True
            except RuntimeError as exc:
                stop_reason = RunStopReason.LOCK_CONFLICT
                self._runtime_state.record_event(
                    event_type="runtime_lock_conflict",
                    payload={"error": str(exc), "traceback": traceback.format_exc(), "output_dir": str(self.output_dir)},
                )
                raise

            # P6.3: capture prior_run_data BEFORE _load_or_create_progress()
            # runs. That call's runtime-state sync (GitHubRuntimeStateBridge.
            # start_or_resume_run -> sync_progress -> rebuild_artifacts)
            # rewrites this same progress.json path as a compat projection
            # of the run being started — reading it AFTER that call would
            # only ever see the current (possibly brand-new, empty) run's
            # own just-synced projection, never genuine prior-run history.
            # Mirrors linkedin/orchestrator.py's actual sequencing: it reads
            # prior_data from progress_path before its own
            # start_or_resume_run call (linkedin/orchestrator.py:1535).
            prior_run_data = None
            if self.progress_path.exists():
                prior_run_data = read_json(str(self.progress_path))

            # Load or create progress
            progress = self._load_or_create_progress(resume)
            self._progress = progress

            # Install Ctrl+C handler
            self._install_signal_handler()

            async with GitHubClient() as client:
                self._client = client
                await client.validate_credentials()

                log_event(self.log_path, "pipeline_start", mode="autonomous")

                from shared.llm_usage import llm_usage_session, resolve_cost_log_run_id

                self._llm_usage_cm = llm_usage_session(
                    self.output_dir / "token-cost-log.jsonl",
                    module="github",
                    brief_id=self.brief_obj.id,
                    run_id=self._runtime_run_id,
                )
                self._llm_usage_cm.__enter__()

                self._governor.start_session()

                enricher = GitHubEnricher(
                    client,
                    brief=self.brief_obj,
                    safety_event_recorder=self._record_safety_event,
                )

                # Step 1: Strategy (if not resuming with existing queries)
                if not progress.queries or not resume:
                    # P6.3: prior_run_data captured above, before
                    # _load_or_create_progress() overwrote progress.json.
                    # Prior this fix, the native orchestrator passed no
                    # prior_run_data at all, so form_github_strategy's
                    # ``## Prior Run Data`` prompt section never rendered.
                    queries, rationale = form_github_strategy(self.brief_obj, prior_run_data)
                    progress.queries = queries
                    self._save_progress()
                    self._observer.on_strategy_formed(queries, rationale)
                else:
                    self._observer.console.emit_info(
                        f"Resuming with {len(progress.queries)} queries from checkpoint"
                    )

                self._observer.on_session_start(self.brief_obj, len(progress.queries))

                # Step 2: Execute queries
                try:
                    await self._execute_queries(client, enricher, progress)
                    if self._shutdown_requested:
                        run_status = "interrupted"
                        stop_reason = RunStopReason.OPERATOR_STOP
                except GitHubGovernorLimitReached as e:
                    run_status = "governor_limit_reached"
                    stop_reason = RunStopReason.GOVERNOR_LIMIT
                    if getattr(self, "_runtime_run_id", None):
                        self._safety.record_governor_limit(
                            run_id=self._runtime_run_id,
                            reason=e.reason,
                        )
                    self._observer.console.emit_info(f"Governor limit reached: {e.reason}")
                except KeyboardInterrupt:
                    run_status = "interrupted"
                    stop_reason = RunStopReason.OPERATOR_STOP
                    self._observer.console.emit_info("Graceful shutdown — saving progress...")
                except ApiBudgetExhaustedError as e:
                    run_status = "error"
                    stop_reason = RunStopReason.API_BUDGET_EXHAUSTED
                    log_event(
                        self.log_path,
                        "budget_exhausted",
                        error=str(e),
                        candidates_processed=self.stats.get("candidates_discovered", 0),
                    )
                    self._observer.on_error("budget", e)
                    raise
                except Exception as e:
                    run_status = "error"
                    stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
                    self._observer.on_error("pipeline", e)
                    traceback.print_exc()
                finally:
                    await self._close_ecosystems_resolver()
                    if hasattr(self, "_llm_usage_cm"):
                        self._llm_usage_cm.__exit__(None, None, None)
                    self._governor.end_session()
                    self._save_progress()
                    # P6.4: save the bias monitor checkpoint on exit
                    # (normal completion, governor limit, interrupt, or
                    # error) — mirrors linkedin's lifecycle.
                    if self._bias_monitor:
                        self._bias_monitor.save_checkpoint(str(self.bias_path))
                    # Move #10 / P4.2: cost_usd alongside the existing stats so
                    # shared.cost_rollup can sum across modules. The prior
                    # hardcoded 0.0 here claimed GitHub spend was "not yet
                    # metered in dollar terms" — false: the enricher/outreach/
                    # strategy LLM calls already record real estimated_cost_usd
                    # rows into this session's token-cost-log.jsonl (opened
                    # above as self._llm_usage_cm). Sum it instead. Never an
                    # affirmative zero: cost_usd/cost_per_save_usd are omitted
                    # entirely when the log has no usable cost signal, rather
                    # than claiming $0 spend.
                    cost_usd = _sum_token_cost_log_usd(
                        self.output_dir / "token-cost-log.jsonl",
                        run_id=resolve_cost_log_run_id(
                            self.output_dir / "token-cost-log.jsonl",
                            self._runtime_run_id,
                        ),
                        exclude_rows_with=("shadow_stage",),
                    )
                    pipeline_end_kwargs: dict = {"stats": self.stats}
                    if cost_usd is not None:
                        self.stats["cost_usd"] = cost_usd
                        pipeline_end_kwargs["cost_usd"] = cost_usd
                        cost_per_save = _cost_per_save_usd(
                            cost_usd, self.stats.get("saved", 0)
                        )
                        if cost_per_save is not None:
                            self.stats["cost_per_save_usd"] = cost_per_save
                            pipeline_end_kwargs["cost_per_save_usd"] = cost_per_save
                    # P4.3.1: compute run health at finalize time and log a
                    # run_degraded event when the monitors trip. Lands in
                    # self.stats["run_health"] so it reaches final_stats in
                    # session_*_metrics.jsonl the same way facial/save
                    # health data does for LinkedIn.
                    run_health = self._run_health_summary()
                    self.stats["run_health"] = run_health
                    if run_health.get("degraded"):
                        log_event(
                            self.log_path,
                            "run_degraded",
                            run_id=self._runtime_run_id,
                            reasons=run_health.get("degraded_reasons", []),
                            green_but_useless=run_health.get("green_but_useless"),
                            judge_parse_failure_rate=run_health.get(
                                "judge_parse_failure_rate"
                            ),
                            baseline_judge_parse_failure_rate=run_health.get(
                                "baseline_judge_parse_failure_rate"
                            ),
                        )
                    log_event(
                        self.log_path,
                        "pipeline_end",
                        **pipeline_end_kwargs,
                    )
                    try:
                        rollup = aggregate_cost_for_run(
                            {"github": self.output_dir}
                        )
                        write_cost_rollup_sidecar(
                            rollup, run_dir=self.output_dir
                        )
                    except Exception as exc:
                        _logger.debug(
                            "cost rollup sidecar write failed: %s", exc
                        )
        except GitHubAuthError as e:
            run_status = "error"
            stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
            self._observer.on_error("github_auth", e)
            log_event(self.log_path, "github_auth_failed", error=str(e), traceback=traceback.format_exc())
            raise
        except ApiBudgetExhaustedError:
            raise
        except Exception as e:
            if stop_reason == RunStopReason.NORMAL:
                run_status = "error"
                stop_reason = RunStopReason.FATAL_RUNTIME_ERROR
            self._observer.on_error("pipeline", e)
            log_event(
                self.log_path,
                "pipeline_error",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            self._client = None
            if getattr(self, "_runtime_run_id", None):
                self._safety.finish_run(
                    run_id=self._runtime_run_id,
                    status=run_status,
                    stop_reason=stop_reason,
                )
            if lock_acquired:
                self._runtime_lock.release()

        # Session end — writes all layer files + report
        # P6.4: pass the real bias summary instead of a hardcoded None.
        bias_summary = self._bias_monitor.session_summary() if self._bias_monitor else None
        self.stats["api_status"] = self._get_api_status()
        self._observer.on_session_end(self.stats, progress or GitHubProgress(brief_name=self.brief_obj.id), bias_summary)
        self._finalize_run_snapshot()

        # Export CSV for Gem/Greenhouse import
        if self.stats["saved"] > 0:
            self._side_effects_service.export_saved_candidates_csv()

        return self.stats

    # ------------------------------------------------------------------
    # Query execution loop
    # ------------------------------------------------------------------

    async def _execute_queries(
        self,
        client: GitHubClient,
        enricher: GitHubEnricher,
        progress: GitHubProgress,
    ):
        """Execute all queries in the queue with adaptation."""
        queries = progress.queries
        batch_stats: list[dict] = []
        executed_since_batch = 0
        # P6.3: batch-scoped baseline for total_rejects/total_insufficient
        # deltas + the saved-candidate list for common_languages_in_saves/
        # common_repos_in_saves. Reset here and after every adaptation.
        self._batch_baseline_stats = dict(self.stats)
        self._batch_save_candidates = []

        for i, query in enumerate(queries):
            if self._shutdown_requested:
                break

            if query.status in ("done", "skipped"):
                continue

            # Skip exhausted channel queries
            channel_stats = self._exhaustion.channels.get(query.channel)
            if channel_stats and channel_stats.status == "exhausted":
                query.status = "skipped"
                query.notes = f"Channel {query.channel} exhausted"
                self._observer.console.emit_info(
                    f"Skipping query #{query.id} — channel {query.channel} exhausted"
                )
                self._save_progress()
                continue

            # Check governor limits
            self._governor.check_limits_or_raise()

            # Check if we should enter enrichment-only mode
            if self._governor.should_enter_enrichment_only(client.limiter.remaining("rest")):
                self._observer.on_enrichment_only_mode()
                break

            # Query start
            session_stats = {
                **self.stats,
                "total_queries": len(queries),
            }
            api_status = self._get_api_status()
            self._observer.on_query_start(query, session_stats, api_status)

            query.status = "in_progress"
            progress.current_query_id = query.id
            self._save_progress()

            query_started_at = time.monotonic()
            try:
                await self._execute_single_query(client, enricher, query, progress)
                if query.status != "failed":
                    query.status = "done"
                    batch_stats.append({
                        "query_id": query.id,
                        "name": query.name,
                        "query_string": query.query,
                        "channel": query.channel,
                        "saves": len(query.saves),
                        "candidates": query.candidates_discovered,
                        # P6.3: read by _build_batch_report to populate
                        # queries_hitting_result_cap. Set per-query at
                        # `_search_users` — was already tracked, just never
                        # threaded into the batch snapshot.
                        "hit_result_cap": query.hit_result_cap,
                    })

                    # Record result in exhaustion state — skip unknown channels
                    # so a bad dispatch arm cannot teach a false exhaustion streak.
                    if query.channel in KNOWN_CHANNELS:
                        pre_dedup = getattr(query, '_pre_dedup_count', query.result_count)
                        self._exhaustion.record_query_result(
                            channel=query.channel,
                            saves=len(query.saves),
                            candidates=query.candidates_discovered,
                            pre_dedup=pre_dedup,
                            post_dedup=query.result_count,
                        )
                    # Move #6: per-query telemetry mirroring linkedin's
                    # string_complete shape. Payload identifies the query
                    # and includes per-query counts.
                    log_event(
                        self.log_path,
                        "string_complete",
                        string_id=query.id,
                        channel=query.channel,
                        saves=len(query.saves),
                        candidates=query.candidates_discovered,
                        elapsed_ms=int((time.monotonic() - query_started_at) * 1000),
                    )
                # P6.4: checkpoint the bias monitor at per-query cadence too
                # (mirrors linkedin's save at string_complete), not just on
                # final exit — so a crash mid-run doesn't lose alert dedup
                # state from completed queries.
                if self._bias_monitor:
                    self._bias_monitor.save_checkpoint(str(self.bias_path))
            except GitHubGovernorLimitReached:
                raise
            except GitHubAuthError:
                raise
            except JudgeContractError:
                raise
            except ApiBudgetExhaustedError:
                raise
            except Exception as e:
                self._observer.on_error("query", e, query)
                query.notes = f"Error: {e}"
                query.status = "error"
                # Move #6: per-query error telemetry mirroring linkedin's
                # string_error shape. Captures exception class + elapsed
                # for triage without surfacing the full traceback.
                log_event(
                    self.log_path,
                    "string_error",
                    string_id=query.id,
                    channel=query.channel,
                    error=str(e),
                    traceback=traceback.format_exc(),
                    error_class=type(e).__name__,
                    elapsed_ms=int((time.monotonic() - query_started_at) * 1000),
                )

            self._save_progress()
            executed_since_batch += 1

            # Adaptation after batch
            if executed_since_batch >= _ADAPTATION_BATCH_SIZE:
                remaining = [q for q in queries if q.status == "queued"]
                if remaining:
                    batch_report = self._build_batch_report(batch_stats)
                    new_queries, rationale, skipped_ids = adapt_after_batch(
                        self.brief_obj, batch_report, remaining,
                        executed_queries=self._get_executed_query_strings(),
                        exhaustion_context=self._exhaustion.to_adaptation_context(),
                    )
                    if new_queries:
                        self._insert_queries_by_priority(queries, new_queries, i)
                        progress.queries = queries
                    decision = build_github_adaptation_decision(
                        batch_report=batch_report,
                        new_queries=new_queries,
                        rationale=rationale,
                        skipped_ids=skipped_ids,
                    )
                    if getattr(self, "_runtime_run_id", None):
                        record_adaptation_decision(
                            self._runtime_state,
                            run_id=self._runtime_run_id,
                            decision=decision,
                        )
                    log_event(
                        self.log_path,
                        "adaptation_decision",
                        **decision.to_dict(),
                    )

                    self._observer.on_adaptation(
                        batch_report, new_queries, skipped_ids, rationale, self.stats
                    )
                    self._save_progress()

                    # Metrics checkpoint at adaptation time — act on stop signal
                    api_status = self._get_api_status()
                    stop_rec = self._observer.write_metrics_checkpoint(api_status, self.stats)
                    if stop_rec and stop_rec.startswith("STOP"):
                        self._observer.console.emit_info(f"Session stop: {stop_rec}")
                        return

                executed_since_batch = 0
                batch_stats = []
                # P6.3: reset the batch-scoped accumulators alongside
                # batch_stats so the next _build_batch_report call reports
                # only this new batch's rejects/insufficient/save-signals.
                self._batch_baseline_stats = dict(self.stats)
                self._batch_save_candidates = []

                # Process graph expansion queue between batches
                if progress.graph_expansion_queue:
                    await self._process_graph_expansion_queue(progress, queries)

    async def _execute_single_query(
        self,
        client: GitHubClient,
        enricher: GitHubEnricher,
        query: GitHubSearchQuery,
        progress: GitHubProgress,
    ):
        """Execute a single search query and process results."""
        # P6.4: reset the bias-pause flag per query — a pause stops the
        # query it fired in, not every subsequent query in the session.
        self._bias_pause_active = False
        self._registry_evidence_by_username = {}
        usernames: list[str] = []
        pre_dedup_count = 0

        if query.channel == "user_search":
            pre_dedup_count, usernames = await self._search_users(client, query)
        elif query.channel == "code_search":
            pre_dedup_count, usernames = await self._search_code(client, query)
        elif query.channel == "repo_mining":
            pre_dedup_count, usernames = await self._mine_repo(client, query, progress)
        elif query.channel == "org_exploration":
            pre_dedup_count, usernames = await self._explore_org(client, query)
        elif query.channel == "topic_search":
            pre_dedup_count, usernames = await self._search_topics(client, query)
        elif query.channel == "stargazer_mining":
            pre_dedup_count, usernames = await self._mine_stargazers(client, query)
        elif query.channel == "graph_expansion":
            pre_dedup_count, usernames = await self._expand_graph(client, query, progress)
        elif query.channel == "registry_maintainer_discovery":
            pre_dedup_count, usernames = await self._discover_registry_maintainers(query)
            if query.status == "failed":
                return
        elif query.channel == "roster_ingest":
            pre_dedup_count, usernames = await self._ingest_rosters(query)
            if query.status == "failed":
                return
        else:
            print(
                f"github orchestrator: unknown channel {query.channel!r}",
                file=sys.stderr,
            )
            query.status = "failed"
            query.notes = f"unknown channel: {query.channel}"
            return

        query.result_count = len(usernames)
        query.candidates_discovered = len(usernames)
        # Store pre-dedup count for exhaustion tracking
        query._pre_dedup_count = pre_dedup_count
        self._observer.on_query_results(query, usernames, pre_dedup_count)

        # Process each candidate with mid-query adaptation
        stats_before = {k: v for k, v in self.stats.items()}
        if self.brief_obj.has_v2_schema:
            batch_candidates: list[tuple[str, GitHubCandidate]] = []

            for j, username in enumerate(usernames):
                if self._shutdown_requested:
                    break
                self._governor.check_limits_or_raise()

                self._observer.on_candidate_discovered(username, query)

                try:
                    candidate = await self._prepare_candidate_for_evaluation(
                        enricher, username, query, progress, result_rank=j + 1,
                    )
                except GitHubGovernorLimitReached:
                    raise
                except JudgeContractError:
                    raise
                except ApiBudgetExhaustedError:
                    raise
                except Exception as e:
                    self._observer.on_error("candidate", e, query)
                    continue

                if candidate:
                    registry_evidence = self._registry_evidence_by_username.get(username)
                    if registry_evidence is not None:
                        candidate.registry_evidence = registry_evidence
                        # candidates.jsonl is what github/export.py joins
                        # against by username to build the CSV. The record
                        # acquisition.py appended during light/full enrich
                        # predates registry evidence attach and has no
                        # registry_evidence field. Append-only log: this
                        # second record for the same username wins the
                        # username-keyed lookup in export.py (last one read
                        # wins). Also refresh the on-disk copy so downstream
                        # joins see declared registry roles and packages.
                        candidate_record = self._candidate_record(candidate)
                        append_jsonl(self.candidates_path, candidate_record)
                    batch_candidates.append((username, candidate))

                processed = j + 1
                should_flush = len(batch_candidates) >= _GITHUB_FACIAL_BATCH_SIZE
                if processed % 25 == 0 or processed == len(usernames):
                    should_flush = should_flush or bool(batch_candidates)

                if should_flush and batch_candidates:
                    await self._process_v2_candidates_batch(batch_candidates, query, progress)
                    batch_candidates = []

                # P6.4: a bias pause fired inside the batch just processed —
                # stop feeding this query more candidates.
                if self._bias_pause_active:
                    self._observer.on_query_stopped_early(
                        query, "bias monitor pause", processed, len(usernames)
                    )
                    break

                # Mid-query adaptation checkpoint every 25 discovered candidates
                if processed % 25 == 0 and processed < len(usernames):
                    query_stats = {
                        k: self.stats[k] - stats_before.get(k, 0)
                        for k in ("saved", "rejected", "facial_yes", "facial_no")
                    }
                    query_stats["processed"] = processed
                    query_stats["geo_filtered"] = self.stats.get("geo_filtered", 0) - stats_before.get("geo_filtered", 0)
                    if self._should_stop_query(query_stats):
                        evaluated = processed - query_stats["geo_filtered"]
                        reason = f"{evaluated} evaluated, {query_stats['saved']} saves, {query_stats['facial_yes']} facial_yes"
                        self._observer.on_query_stopped_early(query, reason, processed, len(usernames))
                        break

            if batch_candidates:
                await self._process_v2_candidates_batch(batch_candidates, query, progress)
                batch_candidates = []
        else:
            for j, username in enumerate(usernames):
                if self._shutdown_requested:
                    break
                self._governor.check_limits_or_raise()

                self._observer.on_candidate_discovered(username, query)

                try:
                    await self._process_candidate(
                        enricher, username, query, progress,
                        result_rank=j + 1,
                    )
                except GitHubGovernorLimitReached:
                    raise
                except JudgeContractError:
                    raise
                except ApiBudgetExhaustedError:
                    raise
                except Exception as e:
                    self._observer.on_error("candidate", e, query)
                    continue

                # Mid-query adaptation checkpoint every 25 candidates
                processed = j + 1
                if processed % 25 == 0 and processed < len(usernames):
                    query_stats = {
                        k: self.stats[k] - stats_before.get(k, 0)
                        for k in ("saved", "rejected", "facial_yes", "facial_no")
                    }
                    query_stats["processed"] = processed
                    query_stats["geo_filtered"] = self.stats.get("geo_filtered", 0) - stats_before.get("geo_filtered", 0)
                    if self._should_stop_query(query_stats):
                        evaluated = processed - query_stats["geo_filtered"]
                        reason = f"{evaluated} evaluated, {query_stats['saved']} saves, {query_stats['facial_yes']} facial_yes"
                        self._observer.on_query_stopped_early(query, reason, processed, len(usernames))
                        break

        # Per-query summary
        if usernames:
            qd = {k: self.stats.get(k, 0) - stats_before.get(k, 0)
                  for k in ("candidates_discovered", "geo_filtered", "insufficient", "facial_no", "facial_yes", "saved", "rejected")}
            qd["found"] = len(usernames)
            self._observer.on_query_end(query, qd)

    # ------------------------------------------------------------------
    # Search channel implementations
    # ------------------------------------------------------------------

    async def _search_users(self, client: GitHubClient, query: GitHubSearchQuery) -> tuple[int, list[str]]:
        """Execute a user search query. Returns (pre_dedup_count, deduplicated usernames)."""
        total, items = await client.search_users(query.query)
        query.hit_result_cap = total > gc.MAX_RESULTS_PER_QUERY
        if query.hit_result_cap:
            self._observer.on_result_cap(query, total)

        all_logins = [item.get("login", "") for item in items]
        deduped = self._dedup_usernames(all_logins)
        return len(all_logins), deduped

    async def _search_code(self, client: GitHubClient, query: GitHubSearchQuery) -> tuple[int, list[str]]:
        """Execute a code search query. Extract repo owners/contributors."""
        total, items = await client.search_code(query.query)

        # Extract unique repo owners from code results
        usernames = set()
        repos_seen = set()
        for item in items:
            repo = item.get("repository", {})
            full_name = repo.get("full_name", "")
            owner = repo.get("owner", {}).get("login", "")
            if owner:
                usernames.add(owner)
            # Also get top contributors for highly-starred repos
            if full_name and full_name not in repos_seen:
                repos_seen.add(full_name)
                stars = repo.get("stargazers_count", 0)
                if stars > 10 and len(repos_seen) <= 5:  # Limit API calls
                    try:
                        contributors = await client.get_repo_contributors(full_name, max_contributors=20)
                        for c in contributors:
                            login = c.get("login", "")
                            if login:
                                usernames.add(login)
                    except Exception:
                        pass

        all_usernames = list(usernames)
        deduped = self._dedup_usernames(all_usernames)
        return len(all_usernames), deduped

    async def _mine_repo(self, client: GitHubClient, query: GitHubSearchQuery, progress: GitHubProgress) -> tuple[int, list[str]]:
        """Mine contributors from a specific repository."""
        repo = query.target_repo
        if not repo:
            return 0, []
        if repo in progress.mined_repos:
            return 0, []

        contributors = await client.get_repo_contributors(repo)
        progress.mined_repos.append(repo)

        all_logins = [c.get("login", "") for c in contributors]
        deduped = self._dedup_usernames(all_logins)
        return len(all_logins), deduped

    async def _explore_org(self, client: GitHubClient, query: GitHubSearchQuery) -> tuple[int, list[str]]:
        """Explore public members of an organization."""
        org = query.target_org
        if not org:
            return 0, []

        members = await client.get_org_members(org)
        all_logins = [m.get("login", "") for m in members]
        deduped = self._dedup_usernames(all_logins)
        return len(all_logins), deduped

    async def _search_topics(self, client: GitHubClient, query: GitHubSearchQuery) -> tuple[int, list[str]]:
        """Search repos by topic, extract owner usernames."""
        total, items = await client.search_repos(query.query)
        usernames = set()
        for item in items:
            owner = item.get("owner", {}).get("login", "")
            if owner:
                usernames.add(owner)
        all_usernames = list(usernames)
        deduped = self._dedup_usernames(all_usernames)
        return len(all_usernames), deduped

    async def _mine_stargazers(self, client: GitHubClient, query: GitHubSearchQuery) -> tuple[int, list[str]]:
        """Mine stargazers from a specific repository."""
        repo = query.target_repo
        if not repo:
            return 0, []
        stargazers = await client.get_stargazers(repo, max_results=gc.MAX_STARGAZERS_PER_REPO)
        all_logins = [s.get("login", "") for s in stargazers]
        deduped = self._dedup_usernames(all_logins)
        return len(all_logins), deduped

    async def _expand_graph(self, client: GitHubClient, query: GitHubSearchQuery, progress: GitHubProgress) -> tuple[int, list[str]]:
        """Expand social graph from a seed username."""
        seed = query.query  # username stored in query field
        if not seed or seed in progress.graph_expansion_processed:
            return 0, []

        followers = await client.get_followers(seed, max_results=gc.MAX_FOLLOWERS_PER_SEED)
        following = await client.get_following(seed, max_results=gc.MAX_FOLLOWERS_PER_SEED)

        usernames = set()
        for user in followers + following:
            login = user.get("login", "")
            if login:
                usernames.add(login)

        # P6.2: mark-processed lives HERE — after the fetch actually ran —
        # not at enqueue time. github/work_units.py's
        # process_graph_expansion_queue used to mark the seed processed the
        # moment it created the graph_expansion query; by the time this
        # method executed that same query, the guard above
        # (`seed in progress.graph_expansion_processed`) had already
        # self-cancelled it, so every expansion query returned (0, [])
        # without ever calling get_followers/get_following, and those fake
        # zero-results fed ExhaustionState's zero-result streak.
        progress.graph_expansion_processed.append(seed)
        if getattr(self, "_runtime_run_id", None):
            self._runtime_state.mark_graph_expansion_seed_processed(
                self._runtime_run_id, seed
            )
        all_usernames = list(usernames)
        deduped = self._dedup_usernames(all_usernames)
        return len(all_usernames), deduped

    def _stash_registry_evidence(
        self,
        username: str,
        *,
        declared_role: dict,
        package_entry: dict,
    ) -> None:
        evidence_by_username = getattr(self, "_registry_evidence_by_username", None)
        if evidence_by_username is None:
            evidence_by_username = {}
            self._registry_evidence_by_username = evidence_by_username
        bucket = evidence_by_username.setdefault(
            username,
            {"declared_roles": [], "packages": []},
        )
        bucket["declared_roles"].append(declared_role)
        pkg_key = (package_entry.get("hub"), package_entry.get("name"))
        existing = {
            (pkg.get("hub"), pkg.get("name"))
            for pkg in bucket["packages"]
        }
        if pkg_key not in existing:
            bucket["packages"].append(package_entry)

    async def _discover_registry_maintainers(
        self,
        query: GitHubSearchQuery,
    ) -> tuple[int, list[str]]:
        """Discover maintainers from npm/crates registry rosters."""
        packages = list(query.target_packages or [])
        if not packages:
            print(
                "registry_maintainer_discovery: empty target_packages — skipping",
                file=sys.stderr,
            )
            return 0, []

        hub_label = _REGISTRY_ECOSYSTEM_TO_HUB.get(query.target_ecosystem, "")
        if not hub_label:
            print(
                "registry_maintainer_discovery: unsupported target_ecosystem "
                f"{query.target_ecosystem!r} — skipping",
                file=sys.stderr,
            )
            return 0, []

        all_usernames: list[str] = []
        npm_unresolved = 0
        fetch_attempts = 0
        fetch_failures = 0
        contributors_cache: dict[str, list[dict]] = {}

        if hub_label == "crates":
            async with CratesHubClient() as hub:
                for seed in packages:
                    if seed.startswith((".", "/")):
                        continue
                    fetch_attempts += 1
                    owners = await hub.get_owner_users(seed)
                    if owners is None:
                        fetch_failures += 1
                        print(
                            f"registry_maintainer_discovery: crates owners unavailable "
                            f"for {seed!r} — skipping seed",
                            file=sys.stderr,
                        )
                        continue
                    crate_info = await hub.get_crate(seed) or {}
                    downloads = crate_info.get("downloads")
                    recent_downloads = crate_info.get("recent_downloads")
                    recent_versions = crate_info.get("recent_versions") or []
                    latest_release = ""
                    if recent_versions:
                        latest_release = recent_versions[0].get("version", "") or ""
                    package_entry = {
                        "hub": "crates",
                        "name": seed,
                        "downloads_window": "last-90-days",
                        "downloads_last_month": (
                            recent_downloads if recent_downloads is not None else downloads
                        ),
                        "reverse_dependencies": None,
                        "latest_release": latest_release,
                        "release_cadence": "",
                        "deprecated": False,
                    }
                    for owner in owners:
                        github_login = owner.get("github_login")
                        if not github_login:
                            continue
                        all_usernames.append(github_login)
                        self._stash_registry_evidence(
                            github_login,
                            declared_role={
                                "hub": "crates",
                                "handle": owner.get("login", github_login),
                                "package": seed,
                                "role": "owner",
                                "corroborated_github_login": github_login,
                            },
                            package_entry=package_entry,
                        )
        elif hub_label == "npm":
            async with NpmHubClient() as hub:
                for seed in packages:
                    if seed.startswith((".", "/")):
                        continue
                    fetch_attempts += 1
                    packument = await hub.get_packument(seed)
                    if packument is None:
                        fetch_failures += 1
                        print(
                            f"registry_maintainer_discovery: npm packument unavailable "
                            f"for {seed!r} — skipping seed",
                            file=sys.stderr,
                        )
                        continue
                    downloads = await hub.get_downloads_last_month(seed)
                    repo_url = packument.get("repository_url")
                    package_entry = {
                        "hub": "npm",
                        "name": seed,
                        "downloads_window": "last-month",
                        "downloads_last_month": downloads,
                        "reverse_dependencies": None,
                        "latest_release": packument.get("latest_version") or "",
                        "release_cadence": "",
                        "deprecated": bool(packument.get("deprecated")),
                    }
                    for handle in packument.get("maintainer_handles") or []:
                        corroborated_login = await self._corroborate_npm_maintainer_github_login(
                            handle,
                            repo_url,
                            contributors_cache=contributors_cache,
                        )
                        if corroborated_login:
                            all_usernames.append(corroborated_login)
                            self._stash_registry_evidence(
                                corroborated_login,
                                declared_role={
                                    "hub": "npm",
                                    "handle": handle,
                                    "package": seed,
                                    "role": "maintainer",
                                    "corroborated_github_login": corroborated_login,
                                },
                                package_entry=package_entry,
                            )
                        else:
                            npm_unresolved += 1

        if fetch_attempts > 0 and fetch_failures == fetch_attempts:
            query.status = "failed"
            query.notes = (
                f"registry hub unreachable for {query.target_ecosystem}: "
                f"all {fetch_attempts} seed fetch(es) failed"
            )
            return 0, []

        if npm_unresolved:
            self.stats.setdefault("registry_unresolved_maintainers", 0)
            self.stats["registry_unresolved_maintainers"] += npm_unresolved
            print(
                "registry_maintainer_discovery: "
                f"{npm_unresolved} npm maintainer handle(s) lacked GitHub "
                "repository corroboration",
                file=sys.stderr,
            )

        self._flush_terminal_registry_evidence_sidecar(query)

        deduped = self._dedup_usernames(all_usernames)
        return len(all_usernames), deduped

    async def _ingest_rosters(
        self,
        query: GitHubSearchQuery,
    ) -> tuple[int, list[str]]:
        """Discover maintainers from governance roster files for target repos."""
        repos = [repo.strip() for repo in (query.target_packages or []) if repo.strip()]
        if not repos:
            fallback = (query.target_repo or "").strip()
            if fallback:
                repos = [fallback]

        if not repos:
            print(
                "roster_ingest: empty target_repo — skipping",
                file=sys.stderr,
            )
            return 0, []

        client = self._client
        if client is None:
            query.status = "failed"
            query.notes = f"roster ingest: no GitHub client for {repos[0]}"
            return 0, []

        all_usernames: list[str] = []
        team_skipped = 0
        fetch_attempts = 0
        fetch_failures = 0

        for owner_repo in repos:
            fetch_attempts += 1
            try:
                result = await fetch_repo_roster(client, owner_repo)
            except Exception as exc:
                fetch_failures += 1
                print(
                    f"roster contents API failed for {owner_repo}: {exc}",
                    file=sys.stderr,
                )
                continue

            package_entry = {
                "hub": "governance",
                "name": owner_repo,
                "downloads_window": "",
                "downloads_last_month": None,
                "reverse_dependencies": None,
                "latest_release": "",
                "release_cadence": "",
                "deprecated": False,
            }

            team_skipped += len(result.team_entries)
            for entry in result.entries:
                handle = entry.handle
                if not handle:
                    continue
                all_usernames.append(handle)
                self._stash_registry_evidence(
                    handle,
                    declared_role={
                        "hub": "governance",
                        "handle": handle,
                        "package": owner_repo,
                        "repo": entry.repo,
                        "role": entry.role,
                        "corroborated_github_login": handle,
                        "source_file": entry.source_file,
                    },
                    package_entry=package_entry,
                )

        if fetch_attempts > 0 and fetch_failures == fetch_attempts:
            query.status = "failed"
            query.notes = (
                f"roster ingest: all {fetch_attempts} repo fetch(es) failed"
            )
            return 0, []

        if team_skipped:
            self.stats.setdefault("roster_team_entries_skipped", 0)
            self.stats["roster_team_entries_skipped"] += team_skipped

        handles_discovered = len(all_usernames)
        if handles_discovered:
            self.stats.setdefault("roster_handles_discovered", 0)
            self.stats["roster_handles_discovered"] += handles_discovered

        self._flush_terminal_registry_evidence_sidecar(query)

        deduped = self._dedup_usernames(all_usernames)
        return len(all_usernames), deduped

    async def _corroborate_npm_maintainer_github_login(
        self,
        handle: str,
        repo_url: str | None,
        *,
        contributors_cache: dict[str, list[dict]],
    ) -> str | None:
        """Map a declared npm maintainer handle to a corroborated GitHub login.

        Architecture doc §5: declared registry record + repository linkage is
        strong evidence; bare name similarity is never enough. Accept when the
        handle matches the packument repository owner OR (for org-owned repos)
        matches a contributor login on the canonical repository.
        """
        repo_owner = _github_owner_from_repo_url(repo_url)
        if repo_owner and handle.lower() == repo_owner.lower():
            return handle

        repo_parts = _github_owner_repo_from_repo_url(repo_url)
        client = self._client
        if not repo_parts or client is None:
            return None

        owner, repo = repo_parts
        cache_key = f"{owner}/{repo}"
        if cache_key not in contributors_cache:
            contributors_cache[cache_key] = await client.get_repo_contributors(cache_key)
        for contributor in contributors_cache[cache_key]:
            login = contributor.get("login")
            if login and login.lower() == handle.lower():
                return login
        return None

    def _flush_terminal_registry_evidence_sidecar(self, query: GitHubSearchQuery) -> None:
        """Persist registry evidence for usernames already terminal this run."""
        from shared.runtime_state.github import github_person_key

        evidence_by_username = getattr(self, "_registry_evidence_by_username", None)
        if not evidence_by_username:
            return

        sidecar_path = Path(self.output_dir) / "registry_evidence.jsonl"
        run_id = getattr(self, "_runtime_run_id", None)
        for username, evidence in evidence_by_username.items():
            if username not in self._seen_usernames:
                continue
            append_jsonl(
                sidecar_path,
                {
                    "username": username,
                    "person_key": github_person_key(username),
                    "evidence": evidence,
                    "run_id": run_id,
                    "query_id": query.id,
                },
            )
            self.stats.setdefault("registry_evidence_sidecar_rows", 0)
            self.stats["registry_evidence_sidecar_rows"] += 1

    async def _get_ecosystems_resolver(self) -> EcosystemsResolver:
        resolver = getattr(self, "_ecosystems_resolver", None)
        if resolver is None:
            resolver = EcosystemsResolver()
            self._ecosystems_resolver = resolver
        if resolver._session is None:
            await resolver.__aenter__()
        return resolver

    async def _close_ecosystems_resolver(self) -> None:
        resolver = getattr(self, "_ecosystems_resolver", None)
        if resolver is None:
            return
        try:
            await resolver.__aexit__(None, None, None)
        except Exception:
            pass
        self._ecosystems_resolver = None

    # ------------------------------------------------------------------
    # Candidate processing
    # ------------------------------------------------------------------

    async def _prepare_candidate_for_evaluation(
        self,
        enricher: GitHubEnricher,
        username: str,
        query: GitHubSearchQuery,
        progress: GitHubProgress,
        result_rank: int = 0,
    ) -> GitHubCandidate | None:
        self._ensure_services()
        result = await self._acquisition_service.prepare_candidate_for_evaluation(
            enricher,
            username,
            query,
            progress,
            result_rank=result_rank,
        )
        if result.terminal_decision:
            return None
        return result.candidate

    async def _process_v2_candidates_batch(
        self,
        batch_candidates: list[tuple[str, GitHubCandidate]],
        query: GitHubSearchQuery,
        progress: GitHubProgress,
    ) -> None:
        """Batch the GitHub V2 facial stage, then run full eval sequentially."""
        portfolio_texts = [
            (candidate.user.name or username, candidate.user.profile_url, candidate.to_portfolio_text())
            for username, candidate in batch_candidates
        ]
        facial_decisions = github_facial_judge_batch(portfolio_texts, brief=self.brief_obj)

        for (username, candidate), facial_decision in zip(batch_candidates, facial_decisions):
            # P6.4: pause severity stops the current query — checked at the
            # top of each candidate iteration so a pause fired mid-batch
            # (below) skips the rest of this batch's candidates without
            # aborting the candidate already in flight.
            if self._bias_pause_active:
                break
            try:
                candidate_record = self._candidate_record(candidate)
                envelope = self._execution_envelope(
                    username=username,
                    query=query,
                    result_rank=0,
                    candidate=candidate,
                    metadata={"candidate_record": candidate_record},
                )
                facial_decision.candidate_name = candidate.user.name or username
                facial_decision.profile_url = candidate.user.profile_url
                facial_attempt_id = self._start_stage_attempt(
                    username=username,
                    stage="facial",
                    query=query,
                    candidate=candidate,
                    result_rank=0,
                    payload={
                        "cursor": envelope.source_cursor,
                        "candidate_record": candidate_record,
                    },
                )

                if self._bias_monitor:
                    self._bias_monitor.record_decision(DecisionRecord(
                        candidate_id=username,
                        stage="facial",
                        decision=facial_decision.decision,
                        confidence=facial_decision.confidence,
                        capability_area=None,
                        string_id=str(query.id),
                    ))

                if is_failure_decision(facial_decision.decision):
                    self.stats.setdefault("parse_failures", 0)
                    self.stats["parse_failures"] += 1
                    self._finish_failure_decision_attempt(
                        attempt_id=facial_attempt_id,
                        username=username,
                        stage="facial",
                        decision=facial_decision,
                        query=query,
                        result_rank=0,
                        candidate=candidate,
                        candidate_record=candidate_record,
                    )
                    self._observer.on_facial_decision(username, facial_decision.decision, facial_decision.rationale, query)
                    continue

                if facial_decision.decision == "FACIAL_NO":
                    self.stats["facial_no"] += 1
                    self._execution_engine.runtime.finish_stage_success(
                        attempt_id=facial_attempt_id,
                        envelope=envelope,
                        stage="facial",
                        decision=facial_decision,
                        extra_payload={"candidate_record": candidate_record},
                    )
                    self._observer.on_facial_decision(username, "FACIAL_NO", facial_decision.rationale, query)
                    self._mark_terminal(username)
                    continue

                self.stats["facial_yes"] += 1
                self._execution_engine.runtime.finish_stage_success(
                    attempt_id=facial_attempt_id,
                    envelope=envelope,
                    stage="facial",
                    decision=facial_decision,
                    extra_payload={"candidate_record": candidate_record},
                )
                self._observer.on_facial_decision(username, "FACIAL_YES", "", query)

                # P6.1: THE integration seam. github/maintainership.py:classify
                # had zero production callers before this — every caller was
                # under tests/. Wire it here: after facial-YES (only
                # facial-passers get classified, capping spend), before the
                # full judge (so candidate.maintainership is populated when
                # to_evidence_text() renders the MAINTAINERSHIP EVIDENCE
                # section the judge prompt already expects). Behavior-
                # preserving for classic github briefs: empty
                # brief.target_projects means classify() is never called and
                # candidate.maintainership stays None, so to_evidence_text()
                # renders byte-identically to today.
                classification = None
                if self.brief_obj.target_projects and self._client is not None:
                    try:
                        classification = await classify_maintainership(
                            username,
                            self.brief_obj.target_projects,
                            self._client,
                        )
                    except Exception as e:
                        classification = None
                        self._observer.on_error("maintainership_classify", e, query)

                declared_entries: list[dict] = []
                registry_evidence = getattr(candidate, "registry_evidence", None)
                if isinstance(registry_evidence, dict):
                    declared_entries = list(
                        registry_evidence.get("declared_roles") or []
                    )

                scoped_declared = declared_entries_for_target_projects(
                    declared_entries,
                    self.brief_obj.target_projects,
                )

                if scoped_declared or classification is not None:
                    candidate.maintainership = merge_declared_maintainership(
                        scoped_declared,
                        classification,
                    )
                    # candidates.jsonl is what github/export.py joins
                    # against by username to build the CSV (P0.3's
                    # Maintainership Level/Confidence/Evidence columns).
                    # The record acquisition.py appended during
                    # light/full enrich predates this classify() call and
                    # has no maintainership field. Append-only log: this
                    # second record for the same username wins the
                    # username-keyed lookup in export.py (last one read
                    # wins). Also refresh the local candidate_record so
                    # the runtime-state payloads recorded below for this
                    # candidate (finish_stage_success / failure payloads)
                    # carry maintainership too, not just the exported CSV.
                    candidate_record = self._candidate_record(candidate)
                    append_jsonl(self.candidates_path, candidate_record)
                elif self._client is not None and candidate.top_repos:
                    top_repo = candidate.top_repos[0]
                    owner = top_repo.owner_login or ""
                    repo_name = top_repo.name
                    if owner and repo_name:
                        repo_key = f"{owner}/{repo_name}"
                        memo = getattr(self, "_project_quality_memo", None)
                        if memo is None:
                            memo = {}
                            self._project_quality_memo = memo
                        cached_quality = memo.get(repo_key)
                        if cached_quality is not None:
                            candidate.portfolio_summary["project_quality"] = cached_quality
                        else:
                            try:
                                quality = await score_project(
                                    owner,
                                    repo_name,
                                    self._client,
                                    ecosystems_resolver=await self._get_ecosystems_resolver(),
                                )
                                quality_payload = {
                                    "score": quality.score,
                                    "band": quality.criticality_band,
                                    "repo": repo_key,
                                }
                                memo[repo_key] = quality_payload
                                candidate.portfolio_summary["project_quality"] = quality_payload
                            except Exception as e:
                                self._observer.on_error("project_quality", e, query)

                evidence_text = candidate.to_evidence_text()
                full_attempt_id = self._start_stage_attempt(
                    username=username,
                    stage="full",
                    query=query,
                    candidate=candidate,
                    result_rank=0,
                    payload={
                        "cursor": envelope.source_cursor,
                        "candidate_record": candidate_record,
                        "facial_decision": facial_decision.to_dict(),
                    },
                )
                try:
                    full_decision = github_full_judge(evidence_text, brief=self.brief_obj)
                except TypeError as e:
                    full_decision = judgment_failure_decision(
                        stage="full",
                        candidate_name=candidate.user.name or username,
                        profile_url=candidate.user.profile_url,
                        error=e,
                        source="judgment",
                    )
                    full_decision.candidate_name = candidate.user.name or username
                    full_decision.profile_url = candidate.user.profile_url
                    self.stats.setdefault("parse_failures", 0)
                    self.stats["parse_failures"] += 1
                    self._finish_failure_decision_attempt(
                        attempt_id=full_attempt_id,
                        username=username,
                        stage="full",
                        decision=full_decision,
                        query=query,
                        result_rank=0,
                        candidate=candidate,
                        candidate_record=candidate_record,
                    )
                    raise JudgeContractError(str(e)) from e
                except ApiBudgetExhaustedError:
                    raise
                except Exception as e:
                    full_decision = judgment_failure_decision(
                        stage="full",
                        candidate_name=candidate.user.name or username,
                        profile_url=candidate.user.profile_url,
                        error=e,
                        source="judgment",
                    )
                full_decision.candidate_name = candidate.user.name or username
                full_decision.profile_url = candidate.user.profile_url

                if self._bias_monitor:
                    self._bias_monitor.record_decision(DecisionRecord(
                        candidate_id=username,
                        stage="full",
                        decision=full_decision.decision,
                        confidence=full_decision.confidence,
                        capability_area=getattr(full_decision, "path", None),
                        string_id=str(query.id),
                    ))
                    # P6.4: check_alerts at this batch checkpoint. GitHub
                    # KEEPS its stop-the-query behavior: LinkedIn demoted
                    # the count-based checks to telemetry (2026-07-04)
                    # because its adaptation seam carries the per-string
                    # bias context and owns the decision — GitHub's
                    # adaptation report has no such channel yet (OSS plan:
                    # "bias monitor write-only"), so the stop trigger moves
                    # from the now-retired "pause" severity to the alert
                    # TYPES, preserving behavior byte-identical until the
                    # OSS workstream decides its own posture.
                    alerts = self._bias_monitor.check_alerts(str(query.id))
                    for alert in alerts:
                        if alert.alert_type in (
                            AlertType.CONSECUTIVE_SAVES,
                            AlertType.SAVE_RATE_SPIKE,
                        ):
                            self._bias_pause_active = True
                            self._observer.console.emit_warn(f"BIAS PAUSE: {alert.message}")
                            log_event(self.log_path, "bias_alert", severity=alert.severity,
                                      alert_type=alert.alert_type, message=alert.message,
                                      string_id=alert.string_id, action="query_stopped")
                        elif alert.severity == "flag":
                            self._observer.console.emit_warn(f"BIAS FLAG: {alert.message}")
                            log_event(self.log_path, "bias_alert", severity="flag",
                                      alert_type=alert.alert_type, message=alert.message,
                                      string_id=alert.string_id)
                        elif alert.severity == "info":
                            self._observer.console.emit_info(f"BIAS INFO: {alert.message}")

                if is_failure_decision(full_decision.decision):
                    self.stats.setdefault("parse_failures", 0)
                    self.stats["parse_failures"] += 1
                    self._finish_failure_decision_attempt(
                        attempt_id=full_attempt_id,
                        username=username,
                        stage="full",
                        decision=full_decision,
                        query=query,
                        result_rank=0,
                        candidate=candidate,
                        candidate_record=candidate_record,
                    )
                    continue

                await self._side_effects_service.handle_full_decision(
                    username=username,
                    candidate=candidate,
                    query=query,
                    progress=progress,
                    full_decision=full_decision,
                    envelope=envelope,
                    full_attempt_id=full_attempt_id,
                )
                # P6.3: batch-scoped save signal for common_languages_in_saves
                # / common_repos_in_saves — read by _build_batch_report.
                if is_save_decision(full_decision.decision):
                    self._batch_save_candidates.append(candidate)

                candidate_record["outreach_copy"] = dict(candidate.outreach_copy or {})
                self._execution_engine.runtime.finish_stage_success(
                    attempt_id=full_attempt_id,
                    envelope=envelope,
                    stage="full",
                    decision=full_decision,
                    extra_payload={"candidate_record": candidate_record},
                )
                self._mark_terminal(username)

                if (progress.candidates_enriched % _CHECKPOINT_EVERY) == 0:
                    self._save_progress()
            except GitHubGovernorLimitReached:
                raise
            except JudgeContractError:
                raise
            except ApiBudgetExhaustedError:
                raise
            except TypeError as e:
                raise JudgeContractError(str(e)) from e
            except Exception as e:
                self._observer.on_error("candidate", e, query)

    async def _process_candidate(
        self,
        enricher: GitHubEnricher,
        username: str,
        query: GitHubSearchQuery,
        progress: GitHubProgress,
        result_rank: int = 0,
    ):
        """Enrich and evaluate a single candidate.

        For non-user_search channels (code_search, repo_mining, stargazer_mining,
        etc.), uses a light enrichment path: fetch only the user profile first,
        check geography, and only do full enrichment if the candidate passes.
        This avoids wasting ~10 API calls per geo-filtered candidate.
        """
        candidate = await self._prepare_candidate_for_evaluation(
            enricher, username, query, progress, result_rank=result_rank,
        )
        if not candidate:
            return

        # --- GitHub-native evaluation pipeline ---
        if self.brief_obj.has_v2_schema:
            # GitHub facial triage using portfolio text
            portfolio_text = candidate.to_portfolio_text()
            candidate_record = self._candidate_record(candidate)
            envelope = self._execution_envelope(
                username=username,
                query=query,
                result_rank=result_rank,
                candidate=candidate,
                metadata={"candidate_record": candidate_record},
            )
            facial_attempt_id = self._start_stage_attempt(
                username=username,
                stage="facial",
                query=query,
                candidate=candidate,
                result_rank=result_rank,
                payload={
                    "cursor": envelope.source_cursor,
                    "candidate_record": candidate_record,
                },
            )
            try:
                facial_decision = github_facial_judge(portfolio_text, brief=self.brief_obj)
            except TypeError as e:
                facial_decision = judgment_failure_decision(
                    stage="facial",
                    candidate_name=candidate.user.name or username,
                    profile_url=candidate.user.profile_url,
                    error=e,
                    source="judgment",
                )
                facial_decision.candidate_name = candidate.user.name or username
                facial_decision.profile_url = candidate.user.profile_url
                self.stats.setdefault("parse_failures", 0)
                self.stats["parse_failures"] += 1
                self._finish_failure_decision_attempt(
                    attempt_id=facial_attempt_id,
                    username=username,
                    stage="facial",
                    decision=facial_decision,
                    query=query,
                    result_rank=result_rank,
                    candidate=candidate,
                    candidate_record=candidate_record,
                )
                raise JudgeContractError(str(e)) from e
            except ApiBudgetExhaustedError:
                raise
            except Exception as e:
                facial_decision = judgment_failure_decision(
                    stage="facial",
                    candidate_name=candidate.user.name or username,
                    profile_url=candidate.user.profile_url,
                    error=e,
                    source="judgment",
                )
            facial_decision.candidate_name = candidate.user.name or username
            facial_decision.profile_url = candidate.user.profile_url
        else:
            # Fallback to LinkedIn-style facial (old briefs)
            snippet = candidate.to_snippet(
                source_string_id=query.id,
                source_string_name=query.name,
                result_rank=result_rank,
            )
            candidate_record = self._candidate_record(candidate)
            envelope = self._execution_envelope(
                username=username,
                query=query,
                result_rank=result_rank,
                candidate=candidate,
                snippet=snippet,
                metadata={"candidate_record": candidate_record},
            )
            facial_attempt_id = self._start_stage_attempt(
                username=username,
                stage="facial",
                query=query,
                candidate=candidate,
                result_rank=result_rank,
                snippet=snippet,
                payload={
                    "cursor": envelope.source_cursor,
                    "candidate_record": candidate_record,
                    "snippet": snippet.to_dict(),
                },
            )
            try:
                facial_decision = facial_judge(snippet, brief=self.brief_obj)
            except ApiBudgetExhaustedError:
                raise
            except Exception as e:
                facial_decision = judgment_failure_decision(
                    stage="facial",
                    candidate_name=snippet.name,
                    profile_url=snippet.profile_url,
                    error=e,
                    source="judgment",
                )

        if self._bias_monitor:
            self._bias_monitor.record_decision(DecisionRecord(
                candidate_id=username,
                stage="facial",
                decision=facial_decision.decision,
                confidence=1.0,
                capability_area=None,
                string_id=str(query.id),
            ))

        if is_failure_decision(facial_decision.decision):
            self.stats.setdefault("parse_failures", 0)
            self.stats["parse_failures"] += 1
            self._finish_failure_decision_attempt(
                attempt_id=facial_attempt_id,
                username=username,
                stage="facial",
                decision=facial_decision,
                query=query,
                result_rank=result_rank,
                candidate=candidate,
                candidate_record=candidate_record,
                snippet=snippet if not self.brief_obj.has_v2_schema else None,
                extra_payload={"snippet": snippet.to_dict()} if not self.brief_obj.has_v2_schema else None,
            )
            self._observer.on_facial_decision(username, facial_decision.decision, facial_decision.rationale, query)
            return

        if facial_decision.decision == "FACIAL_NO":
            self.stats["facial_no"] += 1
            self._execution_engine.runtime.finish_stage_success(
                attempt_id=facial_attempt_id,
                envelope=envelope,
                stage="facial",
                decision=facial_decision,
                extra_payload={
                    "candidate_record": candidate_record,
                    **({"snippet": snippet.to_dict()} if not self.brief_obj.has_v2_schema else {}),
                },
            )
            self._observer.on_facial_decision(username, "FACIAL_NO", facial_decision.rationale, query)
            self._mark_terminal(username)
            return

        self.stats["facial_yes"] += 1
        self._execution_engine.runtime.finish_stage_success(
            attempt_id=facial_attempt_id,
            envelope=envelope,
            stage="facial",
            decision=facial_decision,
            extra_payload={
                "candidate_record": candidate_record,
                **({"snippet": snippet.to_dict()} if not self.brief_obj.has_v2_schema else {}),
            },
        )
        self._observer.on_facial_decision(username, "FACIAL_YES", "", query)

        # Full evaluation
        full_attempt_id = self._start_stage_attempt(
            username=username,
            stage="full",
            query=query,
            candidate=candidate,
            result_rank=result_rank,
            payload={
                "cursor": envelope.source_cursor,
                "candidate_record": candidate_record,
                "facial_decision": facial_decision.to_dict(),
            },
        )
        if self.brief_obj.has_v2_schema:
            evidence_text = candidate.to_evidence_text()
            try:
                full_decision = github_full_judge(evidence_text, brief=self.brief_obj)
            except TypeError as e:
                full_decision = judgment_failure_decision(
                    stage="full",
                    candidate_name=candidate.user.name or username,
                    profile_url=candidate.user.profile_url,
                    error=e,
                    source="judgment",
                )
                full_decision.candidate_name = candidate.user.name or username
                full_decision.profile_url = candidate.user.profile_url
                self.stats.setdefault("parse_failures", 0)
                self.stats["parse_failures"] += 1
                self._finish_failure_decision_attempt(
                    attempt_id=full_attempt_id,
                    username=username,
                    stage="full",
                    decision=full_decision,
                    query=query,
                    result_rank=result_rank,
                    candidate=candidate,
                    candidate_record=candidate_record,
                )
                raise JudgeContractError(str(e)) from e
            except ApiBudgetExhaustedError:
                raise
            except Exception as e:
                full_decision = judgment_failure_decision(
                    stage="full",
                    candidate_name=candidate.user.name or username,
                    profile_url=candidate.user.profile_url,
                    error=e,
                    source="judgment",
                )
            full_decision.candidate_name = candidate.user.name or username
            full_decision.profile_url = candidate.user.profile_url
        else:
            profile_summary = candidate.to_profile_summary()
            try:
                full_decision = full_judge(profile_summary, brief=self.brief_obj)
            except TypeError as e:
                full_decision = judgment_failure_decision(
                    stage="full",
                    candidate_name=profile_summary.name,
                    profile_url=profile_summary.profile_url,
                    error=e,
                    source="judgment",
                )
                self.stats.setdefault("parse_failures", 0)
                self.stats["parse_failures"] += 1
                self._finish_failure_decision_attempt(
                    attempt_id=full_attempt_id,
                    username=username,
                    stage="full",
                    decision=full_decision,
                    query=query,
                    result_rank=result_rank,
                    candidate=candidate,
                    candidate_record=candidate_record,
                    snippet=snippet,
                    extra_payload={"profile_summary": profile_summary.to_dict()},
                )
                raise JudgeContractError(str(e)) from e
            except ApiBudgetExhaustedError:
                raise
            except Exception as e:
                full_decision = judgment_failure_decision(
                    stage="full",
                    candidate_name=profile_summary.name,
                    profile_url=profile_summary.profile_url,
                    error=e,
                    source="judgment",
                )

        if self._bias_monitor:
            self._bias_monitor.record_decision(DecisionRecord(
                candidate_id=username,
                stage="full",
                decision=full_decision.decision,
                confidence=full_decision.confidence,
                capability_area=getattr(full_decision, 'path', None),
                string_id=str(query.id),
            ))

        if is_failure_decision(full_decision.decision):
            self.stats.setdefault("parse_failures", 0)
            self.stats["parse_failures"] += 1
            self._finish_failure_decision_attempt(
                attempt_id=full_attempt_id,
                username=username,
                stage="full",
                decision=full_decision,
                query=query,
                result_rank=result_rank,
                candidate=candidate,
                candidate_record=candidate_record,
                snippet=snippet if not self.brief_obj.has_v2_schema else None,
                extra_payload={"profile_summary": profile_summary.to_dict()} if not self.brief_obj.has_v2_schema else None,
            )
            return

        await self._side_effects_service.handle_full_decision(
            username=username,
            candidate=candidate,
            query=query,
            progress=progress,
            full_decision=full_decision,
            envelope=envelope,
            full_attempt_id=full_attempt_id,
        )
        candidate_record["outreach_copy"] = dict(candidate.outreach_copy or {})
        self._execution_engine.runtime.finish_stage_success(
            attempt_id=full_attempt_id,
            envelope=envelope,
            stage="full",
            decision=full_decision,
            extra_payload={"candidate_record": candidate_record},
            profile_summary=profile_summary if not self.brief_obj.has_v2_schema else None,
        )
        self._mark_terminal(username)

        # Checkpoint periodically
        if (progress.candidates_enriched % _CHECKPOINT_EVERY) == 0:
            self._save_progress()

    # ------------------------------------------------------------------
    # Pre-screen (rule-based, light data only)
    # ------------------------------------------------------------------

    def _prescreen_bio_terms(self) -> list[str]:
        """Brief-derived bio terms for prescreen: capability names, key terms, code signals."""
        terms: list[str] = []
        seen: set[str] = set()
        if self.brief_obj.has_v2_schema and self.brief_obj._new_brief:
            for capability_area in self.brief_obj._new_brief.capability_areas:
                name = getattr(capability_area, "name", None)
                if name and str(name).strip():
                    token = str(name).strip().lower()
                    if token not in seen:
                        terms.append(token)
                        seen.add(token)
                for key_term in getattr(capability_area, "key_terms", None) or []:
                    if key_term and str(key_term).strip():
                        token = str(key_term).strip().lower()
                        if token not in seen:
                            terms.append(token)
                            seen.add(token)
                for code_signal in getattr(capability_area, "github_code_signals", None) or []:
                    if code_signal and str(code_signal).strip():
                        token = str(code_signal).strip().lower()
                        if token not in seen:
                            terms.append(token)
                            seen.add(token)
        return terms

    def _prescreen_light(self, candidate: GitHubCandidate) -> str:
        """Rule-based pre-screen using light_enrich data only.

        Returns: "pass" | "hard_skip"
        """
        user = candidate.user

        # Organization account — never a person
        if user.account_type == "Organization":
            return "hard_skip"

        # Zero repos AND no bio — nothing to evaluate
        if user.public_repos == 0 and not user.bio:
            return "hard_skip"

        # Bio keyword scan: if bio exists but has zero overlap with
        # brief-derived terms, and <3 repos
        if user.bio and user.public_repos < 3:
            bio_terms = self._prescreen_bio_terms()
            if bio_terms:
                bio_lower = user.bio.lower()
                has_signal = any(
                    re.search(rf"\b{re.escape(term)}\b", bio_lower)
                    for term in bio_terms
                )
                if not has_signal:
                    return "hard_skip"

        return "pass"

    # ------------------------------------------------------------------
    # Mid-query adaptation
    # ------------------------------------------------------------------

    @staticmethod
    def _should_stop_query(query_stats: dict) -> bool:
        """Heuristic: abandon a query mid-processing if signal is too low.

        Uses evaluated count (processed minus geo-filtered) so that queries
        with many geo-filtered candidates aren't prematurely abandoned.
        """
        processed = query_stats.get("processed", 0)
        geo_filtered = query_stats.get("geo_filtered", 0)
        evaluated = processed - geo_filtered
        saves = query_stats.get("saved", 0)
        facial_yes = query_stats.get("facial_yes", 0)

        # Zero saves after 50 evaluated candidates — abandon
        if evaluated >= 50 and saves == 0:
            return True
        # <2% save rate after 100 evaluated candidates — abandon
        if evaluated >= 100 and (saves / evaluated) < 0.02:
            return True
        # Zero facial_yes after 50 evaluated — extremely noisy query, abandon
        if evaluated >= 50 and facial_yes == 0:
            return True
        return False

    # ------------------------------------------------------------------
    # Geography filter
    # ------------------------------------------------------------------

    _GEO_KEYWORDS: dict[str, list[str]] = {
        "Brazil": [
            "brazil", "brasil", "são paulo", "sao paulo", "rio de janeiro",
            "belo horizonte", "curitiba", "porto alegre", "recife", "salvador",
            "brasília", "brasilia", "fortaleza", "campinas", "florianópolis",
            "florianopolis", "manaus", "belém", "belem", "goiânia", "goiania",
        ],
        "Colombia": [
            "colombia", "bogotá", "bogota", "medellín", "medellin", "cali",
            "barranquilla", "cartagena", "bucaramanga",
        ],
        "San Francisco Bay Area": gc.BAY_AREA_GEO_KEYWORDS,
    }

    # Top-level domains that signal geography
    # P6.5: bare ".co" removed from Colombia — ".co" is a generic
    # startup/short-link gTLD-adjacent ccTLD (t.co, bit.co, and countless
    # non-Colombian domains); it was a false-positive machine. Require a
    # Colombia-specific signal (".com.co") instead.
    _GEO_TLDS: dict[str, list[str]] = {
        "Brazil": [".br", ".com.br"],
        "Colombia": [".com.co"],
    }

    # Well-known companies headquartered in each geography
    _GEO_COMPANIES: dict[str, list[str]] = {
        "Brazil": [
            "nubank", "itaú", "itau", "bradesco", "stone", "pagseguro",
            "totvs", "vtex", "ifood", "mercado livre", "mercadolivre",
            "globo", "magazineluiza", "magalu", "b3", "xp inc", "creditas",
            "loft", "quintoandar", "quinto andar", "picpay", "inter",
            "embraer", "petrobras", "vale", "ambev",
        ],
        "Colombia": [
            "rappi", "bancolombia", "ecopetrol", "grupo nutresa",
            "mercado libre colombia", "platzi",
        ],
    }

    # High-frequency Portuguese words distinct from Spanish/English.
    # P6.5: dropped como/para/sobre/sistema/ambiente/utilizar — all six are
    # Spanish cognates too (identical or near-identical spelling), so their
    # presence proved nothing about Portuguese vs. Spanish and inflated the
    # false-positive rate for Spanish-speaking (non-Brazil) candidates.
    _PORTUGUESE_MARKERS: list[str] = [
        "não", "você", "também", "projeto", "desenvolvimento",
        "trabalho", "aplicação", "dados", "utilizado",
        "implementação", "funcionalidades", "configuração", "usuário",
        "através", "ainda", "pode",
        "repositório", "construído", "ferramenta", "objetivo",
        "executar", "necessário",
    ]

    def _passes_geography_check(
        self, candidate: GitHubCandidate, query: GitHubSearchQuery, stage: str = "light"
    ) -> bool:
        """Check if a candidate's location matches the brief's target geography.

        Uses a multi-signal approach:
        1. User search channel → always pass (API already filters)
        2. Location field matches geo keywords → pass
        3. Location field explicitly non-matching → reject
        4. Location blank → check secondary proxies (bio, blog TLD, email
           domain, company, Portuguese text in repos/README) → pass if ANY
           matches, reject if none
        """
        # User search already filters by location at the API level
        if query.channel == "user_search":
            return True

        # No geography requirement in brief → pass everyone
        geo = ""
        if self.brief_obj.has_v2_schema and self.brief_obj._new_brief:
            geo = getattr(self.brief_obj._new_brief, "geography", "")
        if not geo:
            geo = self.brief_obj.permanent_filters.get("Location", "")
        if not geo:
            return True

        # P6.5: fail OPEN for any geography without an authored dictionary.
        # Authored keyword/TLD/company dictionaries exist only for
        # Brazil/Colombia (below). Every other geography used to fall
        # through to `[geo.lower()]` as its only keyword — a fallback so
        # weak that any blank-location candidate (the majority of non-
        # user_search channels) had virtually no way to pass, since the
        # TLD/company secondary proxies were also empty for that geo. That
        # mass-rejected blank-location candidates for any geography besides
        # the two configured ones. Let the model see location evidence
        # itself for unconfigured geographies instead of vetoing blind.
        if geo not in self._GEO_KEYWORDS:
            # P6.5 follow-up: this branch fails OPEN (returns True) rather
            # than terminating the candidate, so for non-user_search
            # channels acquisition.py calls this check twice per candidate
            # — once at light-enrich, once at full-enrich (mirrors
            # geo_filtered's light/full call sites in acquisition.py).
            # Unlike geo_filtered, a fail-open never short-circuits the
            # second call, so incrementing on every call double-counted the
            # same candidate. `stage` mirrors on_geo_filtered's stage
            # argument: the light-stage call is always the first touchpoint
            # for a candidate that reaches this branch (user_search skips
            # it entirely via the early return above, at every stage), so
            # only the light call bumps the honest per-candidate aggregate;
            # the full-stage call only bumps its own stage-scoped counter.
            self.stats.setdefault(f"geo_unconfigured_{stage}", 0)
            self.stats[f"geo_unconfigured_{stage}"] += 1
            if stage == "light":
                self.stats["geo_unconfigured"] = self.stats.get("geo_unconfigured", 0) + 1
            return True

        keywords = self._GEO_KEYWORDS[geo]

        location = (candidate.user.location or "").lower().strip()
        if location:
            if geo == "San Francisco Bay Area":
                if any(marker in location for marker in gc.NON_BAY_US_MARKERS):
                    self.stats.setdefault("geo_rejected_stated", 0)
                    self.stats["geo_rejected_stated"] += 1
                    return False
                if any(kw in location for kw in keywords):
                    return True
                self.stats.setdefault("geo_rejected_stated", 0)
                self.stats["geo_rejected_stated"] += 1
                return False

            # Location is set — check if it matches
            if any(kw in location for kw in keywords):
                return True
            # Explicitly non-matching location → reject
            return False

        if geo == "San Francisco Bay Area":
            candidate.portfolio_summary["_geo_status"] = "unverified"
            return True

        # --- Location is blank — check secondary proxies ---

        # Bio mentions country/city
        bio = (candidate.user.bio or "").lower()
        if bio and any(kw in bio for kw in keywords):
            return True

        # Blog has country TLD
        blog = (candidate.user.blog or "").lower()
        if blog:
            tlds = self._GEO_TLDS.get(geo, [])
            if any(blog.rstrip("/").endswith(tld) or f"{tld}/" in blog for tld in tlds):
                return True

        # Email has country TLD
        if candidate.contact and candidate.contact.emails:
            tlds = self._GEO_TLDS.get(geo, [])
            for email in candidate.contact.emails:
                if any(email.lower().endswith(tld) for tld in tlds):
                    return True

        # Company matches known geo companies
        company = (candidate.user.company or "").lower().strip().lstrip("@")
        if company:
            geo_companies = self._GEO_COMPANIES.get(geo, [])
            if any(co_name in company for co_name in geo_companies):
                return True

        # Portuguese text detection (for Brazil)
        if geo == "Brazil" and self._has_portuguese_text(candidate):
            return True

        # No secondary signal found → reject
        return False

    def _has_portuguese_text(self, candidate: GitHubCandidate) -> bool:
        """Check if candidate's repos, README, or bio contain Portuguese text.

        Scans repo names/descriptions, profile README, and repo READMEs for
        high-frequency Portuguese words that are distinct from Spanish/English.
        Requires at least 2 marker hits to reduce false positives.
        """
        text_parts: list[str] = []

        # Repo names and descriptions
        for repo in candidate.top_repos:
            if repo.description:
                text_parts.append(repo.description.lower())
            text_parts.append(repo.name.lower().replace("-", " ").replace("_", " "))

        # Profile README
        if candidate.readme_text:
            text_parts.append(candidate.readme_text[:3000].lower())

        # Repo READMEs
        for readme in candidate.repo_readmes.values():
            if readme:
                text_parts.append(readme[:2000].lower())

        combined = " ".join(text_parts)
        if not combined:
            return False

        hits = sum(1 for marker in self._PORTUGUESE_MARKERS if marker in combined)
        return hits >= 2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _process_graph_expansion_queue(
        self,
        progress: GitHubProgress,
        queries: list[GitHubSearchQuery],
    ):
        self._ensure_services()
        await self._work_unit_service.process_graph_expansion_queue(progress, queries)

    @staticmethod
    def _insert_queries_by_priority(
        queries: list[GitHubSearchQuery],
        new_queries: list[GitHubSearchQuery],
        current_index: int,
    ):
        """Insert new queries with three priority tiers.

        Tier 1 (head): user_search — highest efficiency, geo-targeted.
        Tier 2 (after user_search): graph_expansion — known-good seeds.
        Tier 3 (tail): everything else (code_search, repo_mining, etc.).
        """
        geo_queries = [q for q in new_queries if q.channel == "user_search"]
        graph_queries = [q for q in new_queries if q.channel == "graph_expansion"]
        global_queries = [q for q in new_queries
                          if q.channel not in ("user_search", "graph_expansion")]

        # Find insertion point: first queued query after current_index
        insert_at = current_index + 1
        while insert_at < len(queries) and queries[insert_at].status != "queued":
            insert_at += 1

        # Insert geo queries at head
        for j, q in enumerate(geo_queries):
            queries.insert(insert_at + j, q)

        # Insert graph queries immediately after geo queries
        graph_insert = insert_at + len(geo_queries)
        for j, q in enumerate(graph_queries):
            queries.insert(graph_insert + j, q)

        # Append global queries at the end
        queries.extend(global_queries)

    def _mark_terminal(self, username: str):
        """Promote username from in-flight to permanent dedup."""
        self._in_flight_usernames.discard(username)
        self._seen_usernames.add(username)

    def _dedup_usernames(self, usernames: list[str]) -> list[str]:
        self._ensure_services()
        return self._work_unit_service.dedup_usernames(usernames)

    def _load_or_create_progress(self, resume: bool = False) -> GitHubProgress:
        self._ensure_services()
        return self._work_unit_service.load_or_create_progress(resume=resume)

    def _save_progress(self):
        self._ensure_services()
        self._work_unit_service.save_progress()

    def _finalize_run_snapshot(self) -> None:
        """Freeze the current state_dir into an immutable run_dir snapshot."""
        if not getattr(self, "_runtime_run_id", None):
            return
        try:
            from market_intelligence.run_snapshots import finalize_run_snapshot

            run_dir = finalize_run_snapshot(
                source="github",
                brief_path=self.brief_path,
                state_dir=self.state_dir,
                run_id=int(self._runtime_run_id),
            )
            log_event(
                self.log_path,
                "run_snapshot_finalized",
                run_id=int(self._runtime_run_id),
                run_dir=str(run_dir),
            )
            if self._observer:
                self._observer.console.emit_info(f"Run snapshot: {run_dir}")
        except Exception as exc:
            if self._observer:
                self._observer.console.emit_warn(f"Run snapshot finalization failed: {exc}")
            else:
                print(f"[warn] Run snapshot finalization failed: {exc}")

    def _build_batch_report(self, batch_stats: list[dict]) -> GitHubBatchReport:
        channel_rollup: dict[str, dict] = {}
        for stat in batch_stats:
            channel = stat.get("channel") or "unknown"
            bucket = channel_rollup.setdefault(
                channel,
                {"channel": channel, "queries": 0, "candidates": 0, "saves": 0},
            )
            bucket["queries"] += 1
            bucket["candidates"] += int(stat.get("candidates") or 0)
            bucket["saves"] += int(stat.get("saves") or 0)
        signal_markers = [
            {
                "kind": "productive_query",
                "label": "queries with saves",
                "count": sum(1 for s in batch_stats if s["saves"] > 0),
                "examples": [
                    str(s.get("name") or s.get("query_string") or s.get("query_id"))
                    for s in batch_stats
                    if s["saves"] > 0
                ][:5],
            }
        ]
        noise_markers = [
            {
                "kind": "zero_save_query",
                "label": "queries with zero saves",
                "count": sum(1 for s in batch_stats if s["saves"] == 0),
                "examples": [
                    str(s.get("name") or s.get("query_string") or s.get("query_id"))
                    for s in batch_stats
                    if s["saves"] == 0
                ][:5],
            }
        ]
        exhaustion_markers = [
            {
                "channel": channel,
                "reason": stats.status,
            }
            for channel, stats in sorted(self._exhaustion.channels.items())
            if stats.status in {"degraded", "exhausted"}
        ]

        # P6.3: adaptation inputs made honest. These five fields used to be
        # left at their dataclass defaults (0 / []) every batch, so
        # form_github_strategy/adapt_after_batch's prompt always rendered
        # "0 rejected, 0 insufficient" and never mentioned saved-candidate
        # signal or result-cap hits — Opus adapted blind to them.
        total_rejects = self.stats.get("rejected", 0) - self._batch_baseline_stats.get("rejected", 0)
        total_insufficient = self.stats.get("insufficient", 0) - self._batch_baseline_stats.get("insufficient", 0)

        language_counts: Counter = Counter()
        repo_counts: Counter = Counter()
        for saved_candidate in self._batch_save_candidates:
            if saved_candidate.languages:
                top_language = max(saved_candidate.languages, key=saved_candidate.languages.get)
                if top_language:
                    language_counts[top_language] += 1
            for repo in saved_candidate.top_repos[:3]:
                if repo.name:
                    repo_counts[repo.name] += 1
        common_languages_in_saves = [lang for lang, _ in language_counts.most_common(5)]
        common_repos_in_saves = [repo for repo, _ in repo_counts.most_common(5)]

        queries_hitting_result_cap = [
            s["query_id"] for s in batch_stats if s.get("hit_result_cap")
        ]

        report = GitHubBatchReport(
            batch_name=f"Batch ({len(batch_stats)} queries)",
            queries_run=len(batch_stats),
            queries_with_saves=sum(1 for s in batch_stats if s["saves"] > 0),
            total_candidates_discovered=sum(s["candidates"] for s in batch_stats),
            total_saves=sum(s["saves"] for s in batch_stats),
            total_rejects=max(total_rejects, 0),
            total_insufficient=max(total_insufficient, 0),
            top_performing_queries=[s for s in batch_stats if s["saves"] > 0],
            zero_save_query_ids=[s["query_id"] for s in batch_stats if s["saves"] == 0],
            common_languages_in_saves=common_languages_in_saves,
            common_repos_in_saves=common_repos_in_saves,
            queries_hitting_result_cap=queries_hitting_result_cap,
            query_details=[
                {
                    "query_id": s["query_id"],
                    "name": s["name"],
                    "query_string": s.get("query_string", ""),
                    "channel": s.get("channel", ""),
                    "saves": s["saves"],
                    "candidates": s["candidates"],
                }
                for s in batch_stats
            ],
            channel_metrics=list(channel_rollup.values()),
            signal_markers=signal_markers,
            noise_markers=noise_markers,
            exhaustion_markers=exhaustion_markers,
        )
        return report

    def _get_executed_query_strings(self) -> set[str]:
        """Collect all query strings that have been executed."""
        if not self._progress:
            return set()
        return {
            q.query.lower().strip()
            for q in self._progress.queries
            if q.status in ("done", "in_progress") and q.query
        }

    def _get_api_status(self) -> dict:
        """Get current API budget status from client."""
        if self._client:
            return {
                "rest": self._client.limiter.remaining("rest"),
                "search": self._client.limiter.remaining("search"),
                "code_search": self._client.limiter.remaining("code_search"),
            }
        return {"rest": 0, "search": 0, "code_search": 0}

    def _run_health_summary(self) -> dict:
        """P4.3.1: wire shared.observability_monitors (green_but_useless,
        judge parse-failure baseline) into run finalization. Fail-soft —
        this is observability, its failure must not affect the run."""
        if not getattr(self, "_runtime_run_id", None) or not Path(self.runtime_db_path).exists():
            return {"status": "no_runtime_state"}
        try:
            from shared.observability_monitors import compute_run_health

            health = compute_run_health(self.runtime_db_path, self._runtime_run_id)
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        if health is None:
            return {"status": "run_not_found"}
        return {"status": "ok", **health.to_dict()}

    def _install_signal_handler(self):
        def _handler(sig, frame):
            if self._shutdown_requested:
                print("\n[shutdown] Force exit!")
                sys.exit(1)
            self._shutdown_requested = True
            if self._observer:
                self._observer.console.emit_info("Graceful shutdown requested — finishing current candidate...")
            else:
                print("\n[shutdown] Graceful shutdown requested — finishing current candidate...")
        signal.signal(signal.SIGINT, _handler)

    def _ensure_runtime_state(self) -> None:
        output_dir = Path(self.output_dir)
        if not hasattr(self, "runtime_db_path") or self.runtime_db_path is None:
            self.runtime_db_path = output_dir / "runtime_state.sqlite3"
        if not hasattr(self, "_runtime_state") or self._runtime_state is None:
            self._runtime_state = RuntimeStateStore(self.runtime_db_path)
        if not hasattr(self, "_runtime_lock") or self._runtime_lock is None:
            self._runtime_lock = RuntimeStateLock(output_dir)
        if not hasattr(self, "_runtime_bridge") or self._runtime_bridge is None:
            self._runtime_bridge = GitHubRuntimeStateBridge(
                store=self._runtime_state,
                output_dir=output_dir,
                brief_id=self.brief_obj.id,
                brief_name=self.brief_obj.id,
                brief_path=self.brief_path,
            )
        if not hasattr(self, "_execution_engine") or self._execution_engine is None:
            self._execution_engine = CandidateExecutionEngine(
                store=self._runtime_state,
                output_dir=str(output_dir),
                brief_id=self.brief_obj.id,
                source="github",
            )
        if not hasattr(self, "_runtime_run_id"):
            self._runtime_run_id = None
        if not hasattr(self, "_safety") or self._safety is None:
            self._safety = RunSafetyCoordinator(
                store=self._runtime_state,
                output_dir=output_dir,
                source="github",
                brief_id=self.brief_obj.id,
            )
        self._ensure_services()

    def _ensure_services(self) -> None:
        if not hasattr(self, "_work_unit_service") or self._work_unit_service is None:
            self._work_unit_service = GitHubWorkUnitService(self)
        if not hasattr(self, "_acquisition_service") or self._acquisition_service is None:
            self._acquisition_service = GitHubAcquisitionService(self)
        if not hasattr(self, "_side_effects_service") or self._side_effects_service is None:
            self._side_effects_service = GitHubSideEffectsService(self)

    def _get_query_work_unit_id(self, query: GitHubSearchQuery) -> int | None:
        if not getattr(self, "_runtime_run_id", None):
            return None
        return self._runtime_state.get_work_unit_id(
            self._runtime_run_id,
            kind=GITHUB_QUERY_KIND,
            source_unit_id=str(query.id),
        )

    @staticmethod
    def _candidate_record(candidate: GitHubCandidate) -> dict:
        return {"username": candidate.user.username, **candidate.to_dict()}

    @staticmethod
    def _build_runtime_cursor(query: GitHubSearchQuery, result_rank: int) -> dict:
        return {
            "query_id": query.id,
            "query_name": query.name,
            "query_string": query.query,
            "channel": query.channel,
            "result_rank": result_rank,
        }

    def _execution_envelope(
        self,
        *,
        username: str,
        query: GitHubSearchQuery,
        result_rank: int,
        candidate: GitHubCandidate | None = None,
        snippet: CandidateSnippet | None = None,
        metadata: dict | None = None,
    ):
        display_name = username
        profile_url = f"https://github.com/{username}"
        if candidate is not None:
            display_name = candidate.user.name or username
            profile_url = candidate.user.profile_url
        elif snippet is not None:
            display_name = snippet.name
            profile_url = snippet.profile_url
        from shared.runtime_state.github import github_person_key

        return CandidateExecutionEnvelope(
            source="github",
            brief_id=self.brief_obj.id,
            run_id=getattr(self, "_runtime_run_id", 0) or 0,
            work_unit_kind=GITHUB_QUERY_KIND,
            work_unit_source_id=str(query.id),
            identity_key=username,
            display_name=display_name,
            profile_url=profile_url,
            snippet=snippet,
            source_cursor=self._build_runtime_cursor(query, result_rank),
            metadata={**(metadata or {}), "person_key": github_person_key(username)},
        )

    def _record_safety_event(self, event_type: str, payload: dict) -> None:
        if getattr(self, "_runtime_run_id", None):
            self._runtime_state.record_event(
                run_id=self._runtime_run_id,
                event_type=event_type,
                payload=payload,
            )
        if event_type == "enrichment_failure":
            obs = getattr(self, "_observer", None)
            if obs is not None:
                obs.on_enrichment_failure(str(payload.get("kind", "")))

    def _start_stage_attempt(
        self,
        *,
        username: str,
        stage: str,
        query: GitHubSearchQuery,
        candidate: GitHubCandidate,
        result_rank: int = 0,
        snippet: CandidateSnippet | None = None,
        payload: dict | None = None,
    ) -> int | None:
        if not getattr(self, "_runtime_run_id", None):
            return None
        envelope = self._execution_envelope(
            username=username,
            query=query,
            result_rank=result_rank,
            candidate=candidate,
            snippet=snippet,
        )
        return self._execution_engine.runtime.start_stage(
            envelope,
            stage=stage,
            payload=payload or {},
        )

    def _finish_failure_decision_attempt(
        self,
        *,
        attempt_id: int | None,
        username: str,
        stage: str,
        decision: OpusDecision,
        query: GitHubSearchQuery,
        result_rank: int,
        candidate: GitHubCandidate,
        candidate_record: dict,
        snippet: CandidateSnippet | None = None,
        extra_payload: dict | None = None,
    ) -> None:
        if not attempt_id:
            return
        envelope = self._execution_envelope(
            username=username,
            query=query,
            result_rank=result_rank,
            candidate=candidate,
            snippet=snippet,
            metadata={"candidate_record": candidate_record},
        )
        payload = {"candidate_record": candidate_record}
        if extra_payload:
            payload.update(extra_payload)
        self._execution_engine.runtime.finish_stage_failure(
            attempt_id=attempt_id,
            envelope=envelope,
            stage=stage,
            error_or_failure_decision=decision,
            extra_payload=payload,
        )
        self._in_flight_usernames.discard(candidate_record.get("username", ""))

    def _finish_runtime_failure(
        self,
        *,
        attempt_id: int | None,
        username: str,
        query: GitHubSearchQuery,
        result_rank: int,
        candidate: GitHubCandidate | None = None,
        snippet: CandidateSnippet | None = None,
        error: Exception,
        payload: dict | None = None,
    ) -> None:
        if not attempt_id:
            return
        envelope = self._execution_envelope(
            username=username,
            query=query,
            result_rank=result_rank,
            candidate=candidate,
            snippet=snippet,
        )
        self._execution_engine.runtime.finish_stage_failure(
            attempt_id=attempt_id,
            envelope=envelope,
            stage="preparation",
            error_or_failure_decision=error,
            extra_payload=payload or {},
        )

    def _finish_preparation_terminal(
        self,
        *,
        attempt_id: int | None,
        username: str,
        decision: str,
        query: GitHubSearchQuery,
        candidate: GitHubCandidate,
        result_rank: int,
        candidate_record: dict | None = None,
    ) -> None:
        if not attempt_id:
            return
        envelope = self._execution_envelope(
            username=username,
            query=query,
            result_rank=result_rank,
            candidate=candidate,
            metadata={"candidate_record": candidate_record or self._candidate_record(candidate)},
        )
        payload = {
            "cursor": self._build_runtime_cursor(query, result_rank),
            "candidate_record": candidate_record or self._candidate_record(candidate),
            "terminal_reason": decision,
        }
        self._execution_engine.runtime.record_terminal_runtime_decision(
            attempt_id=attempt_id,
            envelope=envelope,
            decision=decision,
            payload=payload,
        )
