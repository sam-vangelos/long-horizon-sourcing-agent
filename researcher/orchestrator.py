"""Researcher pipeline orchestrator — Slice 6.

Composes Slices 2–5 into one runnable pipeline:

  brief → form_strategy → for each query:
      acquire → disambiguate → for each kept candidate:
          build snippet → facial judge
          if FACIAL_YES / FACIAL_BORDERLINE: full judge → record terminal
          else: record FACIAL_NO terminal

Per Researcher Module Spec Slice 6:

- Mirrors :class:`linkedin.orchestrator.LinkedInPipeline._process_string`
  shape: the outer loop iterates queries (the substrate's work_units),
  the inner loop is per-candidate.
- Resume semantics: re-read ``work_units WHERE status IN ('queued',
  'in_progress')``; pagination cursor lives in ``checkpoint_json``.
- Saves stay in the ``candidates`` table with SAVE-class
  ``terminal_decision`` per Spec Opinion 4 (workspace-only saves).

The orchestrator is dependency-injectable so tests don't require real
OpenAlex / Opus calls — pass stub `OpenAlexClient` and `llm_caller`s.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from researcher.acquisition import execute_query
from researcher.discipline_defaults import resolve_floors
from researcher.identity import disambiguate
from researcher.schemas import (
    ResearcherCandidate,
    ResearcherSnippet,
    identity_key_for_candidate,
)
from researcher.sources.arxiv import ArxivClient
from researcher.sources.openalex import OpenAlexClient
from researcher.sources.semantic_scholar import SemanticScholarClient
from researcher.strategy import (
    ResearcherAdaptiveReport,
    ResearcherQueryReport,
    adapt_after_research_batch,
    form_strategy,
)
from shared.brief_loader import Brief, load_brief
from shared.brief_v2_schema import source_config_for
from shared.adaptive import record_adaptation_decision
from shared.observability import observe
from shared.execution import CandidateExecutionEngine  # noqa: F401  (used via bridge)
from shared.judger import (
    researcher_facial_judge_batch,
    researcher_full_judge,
)
from shared.external_evidence import (
    should_request_external_evidence_for_researcher,
)
from shared.runtime_state.researcher import ResearcherRuntimeStateBridge
from shared.runtime_state.store import RuntimeStateStore
from shared.schemas import OpusDecision
from shared.contracts import FAILURE_DECISIONS
from shared.failures import ApiBudgetExhaustedError
from shared.storage import log_event


logger = logging.getLogger(__name__)


class _FrontierStop(BaseException):
    def __init__(self, reason):
        self.reason = reason


@dataclass
class PipelineRunStats:
    """Per-run aggregate counters."""

    queries_total: int = 0
    queries_completed: int = 0
    candidates_discovered: int = 0
    facial_yes: int = 0
    facial_no: int = 0
    facial_borderline: int = 0
    saves: int = 0
    rejects: int = 0
    reviews: int = 0
    failures: int = 0
    pending: int = 0
    stop_reason: str = ""
    artifact_error: bool = False
    per_query: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "queries_total": self.queries_total,
            "queries_completed": self.queries_completed,
            "candidates_discovered": self.candidates_discovered,
            "facial_yes": self.facial_yes,
            "facial_no": self.facial_no,
            "facial_borderline": self.facial_borderline,
            "saves": self.saves,
            "rejects": self.rejects,
            "reviews": self.reviews,
            "failures": self.failures,
            "pending": self.pending,
            "stop_reason": self.stop_reason,
            "artifact_error": self.artifact_error,
            "per_query": self.per_query,
        }


@dataclass
class ResearcherPipeline:
    """Synchronous pipeline that drives a researcher run end-to-end.

    Construction is dependency-injectable so tests can swap in stubs:
    - ``openalex_client``: any object with ``search_authors(**kwargs)``
    - ``facial_llm_caller`` / ``full_llm_caller``: callables passed
      through to the judges
    """

    brief: Brief
    bridge: ResearcherRuntimeStateBridge
    openalex_client: OpenAlexClient
    facial_llm_caller: Callable[[str, str], Any] | None = None
    full_llm_caller: Callable[[str, str], Any] | None = None
    strategy_llm_caller: Callable[[str, str], dict] | None = None
    max_authors_per_query: int = 200
    log_path: Path | None = None
    # Audit Move #4 R1: optional arXiv + Semantic Scholar clients.
    # When set, the acquisition pipeline enriches each candidate with
    # arxiv_categories (preprint signal OpenAlex hasn't indexed) and
    # reconciles h_index against Semantic Scholar (S2 wins on
    # materially-higher values). Pre-R1 callers (OpenAlex-only)
    # leave both None and behavior is byte-identical.
    arxiv_client: ArxivClient | None = None
    semantic_scholar_client: SemanticScholarClient | None = None
    adaptation_batch_size: int = 3
    page_fetcher: Callable | None = None
    evidence_fetcher: Callable | None = None
    stop_requested: Callable[[], bool] = lambda: False
    after_page_candidates: Callable | None = None
    exa_client_factory: Callable | None = None
    public_extract_caller: Callable | None = None
    public_fetcher: Callable | None = None
    cost_estimates: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Default the run-log to <bridge.output_dir>/run_log.jsonl so it
        # lands alongside the per-source SQLite + projections, matching
        # the LinkedIn / GitHub convention.
        if self.log_path is None:
            self.log_path = Path(self.bridge.output_dir) / "run_log.jsonl"

    @observe(name="researcher.run")
    def run(self, *, run_id: int, prior_data: dict | None = None) -> PipelineRunStats:
        """Execute the full pipeline; return aggregate stats.

        ``run_id`` is the run row created by
        :meth:`ResearcherRuntimeStateBridge.start_or_resume_run`.
        """

        # Move #6: telemetry envelope. Emit pipeline_start before any
        # work; pipeline_error if the run aborts; pipeline_end on the
        # way out (whether success or failure). Mirrors
        # linkedin/orchestrator.py:717-787 namespace + payload shape.
        if source_config_for(self._brief_raw(), "researcher").get("role_mode"):
            return self._run_frontier(run_id)
        log_event(self.log_path, "pipeline_start", mode="full")

        source_config = source_config_for(self._brief_raw(), "researcher")
        floors = resolve_floors(source_config)

        plan = form_strategy(
            self.brief,
            prior_data=prior_data,
            llm_caller=self.strategy_llm_caller,
        )

        queries = list(plan.generated_strings)
        stats = PipelineRunStats(queries_total=len(queries))
        batch_reports: list[ResearcherQueryReport] = []

        # Track adaptive skips in a local set rather than mutating each
        # query dict — the underscore-prefixed key it replaced rode into
        # ``payload_json`` and worked-by-accident on resume.  The canonical
        # ``work_units.status`` column is the truth; this set is just the
        # in-memory mirror so the main loop can short-circuit cheaply.
        skipped_query_ids: set[int] = set()

        # P7.5(a) — mirrors designer/orchestrator.py and
        # exec_search/orchestrator.py's finish_run idiom. Without this,
        # a successful researcher run never calls
        # ``RuntimeStateStore.finish_run``, so the ``runs`` row stays
        # ``status='running'`` forever and the reconciler
        # (cloris/reconciler.py) finalizes the finished run as
        # ``abandoned`` — a false negative on every successful run.
        finish_status = "completed"
        finish_stop_reason = "normal"
        end_status = "ok"
        try:
            query_index = 0
            while query_index < len(queries):
                query = queries[query_index]
                ordering_index = query_index + 1
                query_id = int(query.get("id") or 0)
                if query_id and query_id in skipped_query_ids:
                    # Idempotent re-affirmation of the canonical skip; the
                    # adapt step already wrote status="skipped" through
                    # the bridge, but mirroring keeps the ordering_index
                    # column truthful as adapted queries shift positions.
                    self.bridge.upsert_query_work_unit(
                        run_id=run_id,
                        query=query,
                        status="skipped",
                        ordering_index=ordering_index,
                    )
                    query_index += 1
                    continue
                self.bridge.upsert_query_work_unit(
                    run_id=run_id,
                    query=query,
                    status="in_progress",
                    ordering_index=ordering_index,
                )
                string_id = int(query.get("id") or ordering_index)
                started_at = time.monotonic()
                try:
                    per_query_stats = self._run_one_query(
                        run_id=run_id,
                        query=query,
                        source_config=source_config,
                        floors=floors,
                    )
                except Exception as exc:  # noqa: BLE001 — telemetry must capture all
                    elapsed_ms = int((time.monotonic() - started_at) * 1000)
                    log_event(
                        self.log_path,
                        "string_error",
                        string_id=string_id,
                        error=str(exc),
                        error_class=type(exc).__name__,
                        elapsed_ms=elapsed_ms,
                    )
                    raise
                stats.candidates_discovered += per_query_stats["candidates_discovered"]
                stats.facial_yes += per_query_stats["facial_yes_count"]
                stats.facial_no += per_query_stats["facial_no_count"]
                stats.facial_borderline += per_query_stats[
                    "facial_borderline_count"
                ]
                stats.saves += per_query_stats["saves_count"]
                stats.rejects += per_query_stats["rejected_count"]
                stats.per_query.append(per_query_stats)
                batch_reports.append(
                    ResearcherQueryReport.from_query_stats(query, per_query_stats)
                )
                self.bridge.upsert_query_work_unit(
                    run_id=run_id,
                    query=query,
                    status="done",
                    ordering_index=ordering_index,
                    cursor="exhausted",
                    **per_query_stats,
                )
                stats.queries_completed += 1
                stats.queries_total = len(queries)
                log_event(
                    self.log_path,
                    "string_complete",
                    string_id=string_id,
                    **per_query_stats,
                )
                if (
                    self.adaptation_batch_size > 0
                    and len(batch_reports) >= self.adaptation_batch_size
                ):
                    self._adapt_after_batch(
                        run_id=run_id,
                        queries=queries,
                        current_index=query_index,
                        batch_reports=batch_reports,
                        skipped_query_ids=skipped_query_ids,
                    )
                    stats.queries_total = len(queries)
                    batch_reports = []
                query_index += 1
        except Exception as exc:  # noqa: BLE001 — telemetry must capture all
            finish_status = "error"
            finish_stop_reason = f"error: {type(exc).__name__}"
            end_status = "error"
            log_event(
                self.log_path,
                "pipeline_error",
                error=str(exc),
                error_class=type(exc).__name__,
            )
            raise
        finally:
            # Emit aggregate-only fields on pipeline_end (drop per_query
            # to match LinkedIn's flat-stats shape; per-query detail is
            # already captured in the string_complete events above).
            # Move #10: cost_usd lands here so shared.cost_rollup can
            # sum across modules without re-invoking APIs. Researcher
            # publishes 0.0 today — Anthropic full-judge spend isn't
            # yet metered at the orchestrator boundary; landing the
            # field is the per-Move-6 contract bump so the
            # cross-module aggregator has a stable shape to read.
            agg = {k: v for k, v in stats.as_dict().items() if k != "per_query"}
            agg["cost_usd"] = 0.0
            log_event(self.log_path, "pipeline_end", status=end_status, **agg)
            # P7.5(a) — close the canonical run row. Mirrors
            # designer/orchestrator.py:246 and
            # exec_search/orchestrator.py:238-240.
            self.bridge.store.finish_run(
                run_id, finish_status, stop_reason=finish_stop_reason
            )

        return stats


    def _run_frontier(self, run_id: int) -> PipelineRunStats:
        import json
        import math
        from researcher.strategy import frontier_plan, query_signature
        from shared.safety.stop_reasons import RunStopReason

        campaign = self.bridge.campaign(run_id)
        limits = campaign["limits"]
        started = time.monotonic()
        stats = PipelineRunStats()
        status, stop_reason = "completed", RunStopReason.NORMAL
        self._frontier_started = started
        self._frontier_run_id = run_id
        try:
            self._check_stop()
            from researcher.budget import CampaignBudget, CampaignExaMeter
            self._active_query_id = None
            def on_dispatch(stage, state):
                if stage in {"openalex", "exa", "acquisition"} and self._active_query_id is not None:
                    ids = state.setdefault("dispatched_queries", [])
                    if self._active_query_id not in ids:
                        if len(ids) >= state["limits"]["max_queries"]:
                            raise _FrontierStop(RunStopReason.GOVERNOR_LIMIT)
                        ids.append(self._active_query_id)
            self._budget = CampaignBudget(self.bridge, run_id, self._check_stop, on_dispatch)
            self._facial_caller = self._budget.model_caller("facial", self.facial_llm_caller, estimate=self.cost_estimates.get("facial"))
            self._full_caller = self._budget.model_caller("full", self.full_llm_caller, estimate=self.cost_estimates.get("full"))
            self._extract_caller = self._budget.model_caller("extraction", self.public_extract_caller, estimate=self.cost_estimates.get("extraction"))
            if isinstance(self.openalex_client, OpenAlexClient):
                self.openalex_client.dispatch = lambda stage, call: self._budget.dispatch(stage, call, retries=2)
            def exa_factory():
                from exa.client import ExaClient
                return (self.exa_client_factory or ExaClient)(meter=CampaignExaMeter(self._budget))
            if not campaign.get("plan") or campaign.get("replan_required"):
                plan = frontier_plan(
                    self.brief, reference_date=campaign["reference_date"],
                    revision=campaign["plan_revision"],
                    brief_hash=self.bridge.store.get_run(run_id)["brief_content_hash"],
                )
                old_plan = campaign.get("plan", [])
                campaign["plan"] = plan.generated_strings
                campaign.pop("replan_required", None)
                self.bridge.save_campaign(run_id, campaign)
                live_signatures = {query_signature(q) for q in campaign["plan"]}
                for query in old_plan:
                    if query_signature(query) not in live_signatures:
                        self.bridge.upsert_query_work_unit(run_id=run_id, query=query, status="skipped",
                                                            metrics={"retired_reason": "brief revision"})
            for order, query in enumerate(campaign["plan"], 1):
                existing = self.bridge.query_row(run_id, query["id"])
                if not existing:
                    self.bridge.upsert_query_work_unit(run_id=run_id, query=query, ordering_index=order)
                elif json.loads(existing["payload_json"]).get("brief_hash") != query.get("brief_hash"):
                    old = json.loads(existing["payload_json"])
                    query.update(compiled_params=old.get("compiled_params", {}), resolved_entities=old.get("resolved_entities", {}))
                    self.bridge.upsert_query_work_unit(run_id=run_id, query=query, status=existing["status"], ordering_index=order)
            # Pending stage obligations precede new acquisition on every session.
            self._drain_frontier_candidates(run_id)
            query_index = 0
            while query_index < len(self.bridge.campaign(run_id)["plan"]):
                query = self.bridge.campaign(run_id)["plan"][query_index]
                query_index += 1
                self._check_stop()
                row = self.bridge.query_row(run_id, query["id"])
                if row["status"] in {"done", "skipped"}:
                    continue
                query = json.loads(row["payload_json"])
                while True:
                    self._check_stop()
                    campaign = self.bridge.campaign(run_id)
                    admitted = campaign.get("admitted_keys", [])
                    if len(admitted) >= limits["max_candidates"]:
                        raise _FrontierStop(RunStopReason.GOVERNOR_LIMIT)
                    dispatched = campaign.setdefault("dispatched_queries", [])
                    if query["id"] not in dispatched:
                        if len(dispatched) >= limits["max_queries"]:
                            raise _FrontierStop(RunStopReason.GOVERNOR_LIMIT)
                    self._active_query_id = query["id"]
                    self._budget.query_id = query["id"]
                    checkpoint = json.loads(self.bridge.query_row(run_id, query["id"])["checkpoint_json"])
                    page_limit = max(query["bounds"]["max_pages"], math.ceil(limits["max_candidates"] / query["bounds"]["per_page"]))
                    if checkpoint.get("pages_fetched", 0) >= page_limit:
                        self.bridge.upsert_query_work_unit(run_id=run_id, query=query, status="in_progress",
                                                            checkpoint={"cap_reason": "page_cap"})
                        raise _FrontierStop(RunStopReason.GOVERNOR_LIMIT)
                    self.bridge.upsert_query_work_unit(run_id=run_id, query=query, status="in_progress",
                                                        checkpoint={"request_started": True, "cap_reason": None})
                    try:
                        if checkpoint.get("received_page"):
                            page = dict(checkpoint["received_page"])
                            page["candidates"] = [ResearcherCandidate.from_dict(c) for c in page["candidates"]]
                        elif self.page_fetcher:
                            page = self._budget.dispatch("acquisition", lambda: self.page_fetcher(query, checkpoint.get("cursor", "*")), estimate=self.cost_estimates.get("acquisition", 0.001))
                        elif query["provider"] == "exa":
                            from researcher.sources.public_web import fetch_public_page
                            from researcher.strategy import compile_query
                            query.update(compile_query(query, None))
                            self.bridge.upsert_query_work_unit(run_id=run_id, query=query, status="in_progress")
                            page = fetch_public_page(query=query, client_factory=exa_factory,
                                                     llm_caller=self._extract_caller, cached_response=checkpoint.get("raw_response"),
                                                     persist_response=lambda data: self.bridge.upsert_query_work_unit(run_id=run_id, query=query, status="in_progress", checkpoint={"raw_response": data}))
                        else:
                            from researcher.acquisition import fetch_query_page
                            def persist_compiled(compiled):
                                query.update(compiled)
                                self.bridge.upsert_query_work_unit(run_id=run_id, query=query, status="in_progress")
                            page = fetch_query_page(query=query, client=self.openalex_client,
                                                    cursor=checkpoint.get("cursor", "*"), persist_compiled=persist_compiled)
                        self.bridge.persist_page(run_id=run_id, query=query, page=page,
                                                  after_candidates=self.after_page_candidates)
                    except ApiBudgetExhaustedError:
                        raise
                    except Exception as exc:
                        self.bridge.upsert_query_work_unit(run_id=run_id, query=query, status="error",
                                                            metrics={"acquisition_failure": type(exc).__name__})
                        raise
                    self._drain_frontier_candidates(run_id)
                    if not page.get("next_cursor"):
                        break
                from researcher.strategy import adapt_frontier_order
                adapt_frontier_order(self.bridge, run_id, query["id"])
            self._check_stop()
            failed = [r for r in self.bridge.pending_candidates() if r["current_lifecycle_state"].startswith("failed")]
            if failed:
                status, stop_reason = "error", RunStopReason.FATAL_RUNTIME_ERROR
        except _FrontierStop as exc:
            status, stop_reason = "interrupted", exc.reason
            if exc.reason == RunStopReason.API_BUDGET_EXHAUSTED:
                self.bridge.defer_open_attempts(run_id)
        except ApiBudgetExhaustedError:
            status, stop_reason = "interrupted", RunStopReason.API_BUDGET_EXHAUSTED
            self.bridge.defer_open_attempts(run_id)
        except KeyboardInterrupt:
            status, stop_reason = "interrupted", RunStopReason.OPERATOR_STOP
        except Exception:
            status, stop_reason = "error", RunStopReason.FATAL_RUNTIME_ERROR
            raise
        finally:
            self.bridge.store.finish_run(run_id, status, stop_reason=stop_reason)
            from researcher.report import build_report, publish_report
            report = build_report(self.bridge, run_id)
            stats.candidates_discovered = report["counts"]["discovered"]
            stats.saves = report["counts"]["accepted"]
            stats.rejects = report["counts"]["rejected"]
            stats.reviews = report["counts"]["review"]
            stats.failures = report["counts"]["failed"]
            stats.pending = report["counts"]["pending"]
            stats.stop_reason = stop_reason
            units = self.bridge.store.list_work_units(run_id)
            stats.queries_total = len(units)
            stats.queries_completed = sum(u["status"] == "done" for u in units)
            try:
                publish_report(self.bridge, run_id)
            except (Exception, KeyboardInterrupt) as exc:
                stats.artifact_error = True
                campaign = self.bridge.campaign(run_id)
                campaign["artifact_degradation"] = type(exc).__name__
                self.bridge.save_campaign(run_id, campaign)
                if isinstance(exc, KeyboardInterrupt):
                    self.bridge.store.finish_run(run_id, "interrupted", stop_reason=RunStopReason.OPERATOR_STOP)
                    stats.stop_reason = RunStopReason.OPERATOR_STOP
            try:
                log_event(self.log_path, "pipeline_end", status=status, **stats.as_dict(), cost_usd=report["cost"]["charged_usd"],
                          cost_certainty="estimated" if report["cost"]["unavailable_calls"] else "usage_priced")
            except OSError:
                pass
        return stats

    def _check_stop(self):
        from shared.safety.stop_reasons import RunStopReason
        limits = self.bridge.campaign(self._frontier_run_id)["limits"]
        if self.stop_requested():
            raise _FrontierStop(RunStopReason.OPERATOR_STOP)
        if time.monotonic() - self._frontier_started >= limits["max_minutes"] * 60:
            raise _FrontierStop(RunStopReason.SESSION_EXPIRED)
        from researcher.budget import cost_summary
        if cost_summary(self.bridge.campaign(self._frontier_run_id).get("cost_receipts", []))["charged_usd"] >= limits["max_cost_usd"]:
            raise _FrontierStop(RunStopReason.API_BUDGET_EXHAUSTED)

    def _drain_frontier_candidates(self, run_id):
        import json
        from shared.safety.stop_reasons import RunStopReason
        config = source_config_for(self._brief_raw(), "researcher")
        def fetch_public(url):
            self._check_stop()
            from researcher.sources.public_web import fetch_public_text
            return (self.public_fetcher or fetch_public_text)(url)
        for row in self.bridge.pending_candidates():
            self._check_stop()
            candidate, query_id = self.bridge.candidate_record(row["identity_key"])
            self._active_query_id = query_id
            self._budget.query_id = query_id
            campaign = self.bridge.campaign(run_id)
            admitted = campaign.setdefault("admitted_keys", [])
            if row["identity_key"] not in admitted:
                if len(admitted) >= campaign["limits"]["max_candidates"]:
                    raise _FrontierStop(RunStopReason.GOVERNOR_LIMIT)
                admitted.append(row["identity_key"])
                self.bridge.save_campaign(run_id, campaign)
            snippet = _snippet_from_candidate(candidate, {"id": query_id})
            terminal = json.loads(row["terminal_payload_json"] or "{}")
            facial_done = terminal.get("facial_decision", {}).get("decision") in {"FACIAL_YES", "FACIAL_BORDERLINE"}
            if not facial_done:
                with self.bridge.store.connect() as conn:
                    prior_facial = conn.execute(
                        "SELECT ca.payload_json, r.brief_content_hash FROM candidate_attempts ca JOIN runs r ON r.id=ca.run_id WHERE ca.candidate_id=? AND ca.stage='facial' AND ca.status='succeeded' ORDER BY ca.id DESC LIMIT 1",
                        (row["id"],),
                    ).fetchone()
                if prior_facial and prior_facial[1] == self.bridge.store.get_run(run_id)["brief_content_hash"]:
                    facial_done = json.loads(prior_facial[0]).get("facial_decision", {}).get("decision") in {"FACIAL_YES", "FACIAL_BORDERLINE"}
            if row.get("brief_content_hash") and row["brief_content_hash"] != self.bridge.store.get_run(run_id)["brief_content_hash"]:
                facial_done = False
            for stage in (["full"] if facial_done else ["facial", "full"]):
                while True:
                    self._check_stop()
                    with self.bridge.store.connect() as conn:
                        attempts = conn.execute("SELECT COUNT(*) FROM candidate_attempts ca JOIN runs r ON r.id=ca.run_id WHERE ca.candidate_id=? AND ca.stage=? AND ca.status IN ('failed','reconciled') AND COALESCE(ca.failure_kind,'') != 'budget_deferred' AND r.brief_content_hash IS ?", (row["id"], stage, self.bridge.store.get_run(run_id)["brief_content_hash"])).fetchone()[0]
                    if attempts >= 2:
                        break
                    if stage == "facial":
                        attempt_id = self.bridge.start_facial(run_id=run_id, snippet=snippet)
                        decision = researcher_facial_judge_batch([snippet], brief=self.brief, source_config=config,
                                                                  llm_caller=self._facial_caller)[0]
                        self.bridge.record_facial_decision(run_id=run_id, snippet=snippet, decision=decision, attempt_id=attempt_id)
                    else:
                        attempt_id = self.bridge.start_full(run_id=run_id, candidate=candidate)
                        try:
                            if self.evidence_fetcher:
                                self._budget.dispatch("enrichment", lambda: self.evidence_fetcher(candidate), estimate=self.cost_estimates.get("enrichment", 0.001))
                            elif not self.page_fetcher:
                                from researcher.acquisition import enrich_research_candidate
                                enrich_research_candidate(candidate, client=self.openalex_client, source_config=config,
                                                          public_fetcher=fetch_public,
                                                          persist=lambda c: self.bridge.record_candidate_discovery(run_id=run_id, query_id=query_id, candidate=c))
                            self.bridge.record_candidate_discovery(run_id=run_id, query_id=query_id, candidate=candidate)
                            if identity_key_for_candidate(candidate) != row["identity_key"]:
                                original, _ = self.bridge.candidate_record(row["identity_key"])
                                self.bridge.runtime.finish_attempt_success(attempt_id=attempt_id,
                                    envelope=self.bridge.envelope(run_id, original, query_id), new_state="full_terminal",
                                    terminal_decision="PRESCREEN_SKIP", payload={"identity_alias_of": identity_key_for_candidate(candidate), "candidate_record": candidate.to_dict()})
                                break
                            self._check_stop()
                            decision = researcher_full_judge(candidate, brief=self.brief, llm_caller=self._full_caller)
                        except (_FrontierStop, KeyboardInterrupt, ApiBudgetExhaustedError):
                            raise
                        except Exception as exc:
                            from shared.failures import judgment_failure_decision
                            decision = judgment_failure_decision(stage="full", candidate_name=candidate.name,
                                                                 profile_url=candidate.profile_url, error=RuntimeError(type(exc).__name__), source="judgment")
                        self.bridge.record_full_decision(run_id=run_id, candidate=candidate, decision=decision, attempt_id=attempt_id)
                    if self._budget.fatal_error is not None:
                        raise self._budget.fatal_error
                    if decision.decision not in FAILURE_DECISIONS:
                        break
                current = self.bridge.candidate_row(row["identity_key"])
                if current["terminal_decision"] or current["current_lifecycle_state"].startswith("failed"):
                    break
    def _adapt_after_batch(
        self,
        *,
        run_id: int,
        queries: list[dict],
        current_index: int,
        batch_reports: list[ResearcherQueryReport],
        skipped_query_ids: set[int],
    ) -> None:
        remaining_queries = [
            query
            for query in queries[current_index + 1 :]
            if int(query.get("id") or 0) not in skipped_query_ids
        ]
        if not remaining_queries:
            return
        report = ResearcherAdaptiveReport(
            batch_name=f"Batch ({len(batch_reports)} researcher queries)",
            query_reports=list(batch_reports),
            source_mix=self._source_mix_for_queries(batch_reports),
        )
        plan = adapt_after_research_batch(
            self.brief,
            report,
            remaining_queries,
            llm_caller=self.strategy_llm_caller,
        )
        if plan.decision is not None:
            record_adaptation_decision(
                self.bridge.store,
                run_id=run_id,
                decision=plan.decision,
            )
            log_event(
                self.log_path,
                "adaptation_decision",
                **plan.decision.to_dict(),
            )

        skip_ids = set(plan.skipped_query_ids)
        for idx, query in enumerate(queries, start=1):
            query_id = int(query.get("id") or 0)
            if query_id in skip_ids and query_id not in skipped_query_ids:
                skipped_query_ids.add(query_id)
                self.bridge.upsert_query_work_unit(
                    run_id=run_id,
                    query=query,
                    status="skipped",
                    ordering_index=idx,
                )

        if plan.reordered_query_ids:
            reorder_rank = {
                query_id: rank
                for rank, query_id in enumerate(plan.reordered_query_ids)
            }
            prefix = queries[: current_index + 1]
            suffix = queries[current_index + 1 :]
            suffix.sort(
                key=lambda query: (
                    reorder_rank.get(int(query.get("id") or 0), len(reorder_rank)),
                    int(query.get("id") or 0),
                )
            )
            queries[:] = prefix + suffix

        if plan.new_queries:
            # Eagerly upsert adapted queries as "queued" before the next
            # outer-loop iteration so a crash in this window leaves the
            # canonical work_units table truthful — the previous
            # late-upsert dropped adapted queries on resume.
            insert_at = current_index + 1
            for offset, query in enumerate(plan.new_queries):
                queries.insert(insert_at + offset, query)
                self.bridge.upsert_query_work_unit(
                    run_id=run_id,
                    query=query,
                    status="queued",
                    ordering_index=insert_at + offset + 1,
                )

    @staticmethod
    def _source_mix_for_queries(
        reports: list[ResearcherQueryReport],
    ) -> dict[str, int]:
        mix = {"openalex": len(reports)}
        return {key: value for key, value in mix.items() if value}

    # -----------------------------------------------------------------
    # Per-query execution
    # -----------------------------------------------------------------

    def _run_one_query(
        self,
        *,
        run_id: int,
        query: dict,
        source_config: dict,
        floors: dict[str, int],
    ) -> dict:
        """Execute one researcher query end-to-end; return per-query stats."""

        result = execute_query(
            query=query,
            client=self.openalex_client,
            papers_in_window_months=floors["papers_in_window_months"],
            max_authors=self.max_authors_per_query,
            now=datetime.now(timezone.utc),
            arxiv_client=self.arxiv_client,
            semantic_scholar_client=self.semantic_scholar_client,
        )

        # Disambiguate against the brief's hard constraints.
        disambiguation = disambiguate(
            result.candidates,
            allowed_country_codes=list(query.get("ror_country_filter") or []),
            required_concept_ids=list(query.get("topic_concepts") or []),
            papers_in_window_floor=floors["papers_in_window_floor"],
        )

        kept_results = [r for r in disambiguation.results if r.kept]

        processable_results = []
        # Discover all kept candidates (so the workspace surfaces them
        # even if facial cuts them — useful for triage telemetry), but do
        # not reopen terminal identities on resume. The shared store refuses
        # to reset terminal lifecycle rows; this caller gate avoids re-paying
        # facial/full evaluation for identities that are already terminal.
        for kept in kept_results:
            if self.bridge.store.is_dedup_blocked(
                source="researcher",
                brief_id=self.bridge.brief_id,
                identity_key=kept.identity_key,
            ):
                log_event(
                    self.log_path,
                    "candidate_dedup_blocked",
                    identity_key=kept.identity_key,
                )
                continue
            self.bridge.record_candidate_discovery(
                run_id=run_id,
                query_id=int(query.get("id") or 0),
                candidate=kept.candidate,
            )
            processable_results.append(kept)

        # Facial triage in batch.
        snippets = [
            _snippet_from_candidate(r.candidate, query)
            for r in processable_results
        ]
        facial_attempts = [self.bridge.start_facial(run_id=run_id, snippet=snippet) for snippet in snippets]
        facial_decisions = researcher_facial_judge_batch(
            snippets,
            brief=self.brief,
            source_config=source_config,
            llm_caller=self.facial_llm_caller,
        )

        facial_yes = facial_no = facial_borderline = 0
        saves = rejects = 0

        for kept_result, snippet, decision, facial_attempt in zip(
            processable_results, snippets, facial_decisions, facial_attempts
        ):
            self.bridge.record_facial_decision(
                run_id=run_id,
                snippet=snippet,
                decision=decision,
                attempt_id=facial_attempt,
            )
            if decision.decision in FAILURE_DECISIONS:
                continue
            if decision.decision == "FACIAL_YES":
                facial_yes += 1
            elif decision.decision == "FACIAL_BORDERLINE":
                facial_borderline += 1
            else:
                facial_no += 1
                continue

            # Audit Move #23: external-evidence cross-check before
            # the full SAVE/NO call. Mirrors LinkedIn's
            # should_request_external_evidence shape; researcher-side
            # heuristics fire on missing ORCID, thin publication
            # record, or recent burst. We always invoke the gate
            # (cheap, pure function) so telemetry can report whether
            # the cross-check WOULD have fired even when the provider
            # isn't yet wired — that landing slice (Perplexity-backed
            # arXiv + news provider) is a follow-up for Researcher.
            #
            # P7.5(c): the "fired" branch emits ``external_evidence_
            # unavailable``, NOT ``external_evidence_fetched`` — no
            # provider is wired yet (see comment above), so nothing was
            # actually fetched. ``external_evidence_fetched`` is
            # LinkedIn's real-fetch event (linkedin/orchestrator.py);
            # reusing it here was a false claim of work that never
            # happened.
            ext_evidence_decision = (
                should_request_external_evidence_for_researcher(
                    candidate=kept_result.candidate,
                    brief=None,
                )
            )
            log_event(
                self.log_path,
                "external_evidence_skipped"
                if not ext_evidence_decision.should_run
                else "external_evidence_unavailable",
                identity_key=kept_result.identity_key,
                reason=ext_evidence_decision.reason,
                skip_reason=ext_evidence_decision.skip_reason,
                **ext_evidence_decision.signals,
            )

            # Escalate to full eval.
            full_attempt = self.bridge.start_full(run_id=run_id, candidate=kept_result.candidate)
            full_decision = researcher_full_judge(
                kept_result.candidate,
                brief=self.brief,
                llm_caller=self.full_llm_caller,
            )
            # Move #26: pipe the disambiguator's common-name collision
            # flag through to the terminal payload so the workspace card
            # (audit Move #4) can surface a manual-review affordance for
            # ORCID-less name collisions per Spec Opinion 3.
            self.bridge.record_full_decision(
                run_id=run_id,
                candidate=kept_result.candidate,
                decision=full_decision,
                needs_identity_confirmation=kept_result.needs_manual_review,
                identity_review_note=kept_result.review_note,
                attempt_id=full_attempt,
            )
            if full_decision.decision in {
                "SAVE",
                "INFERENTIAL_SAVE",
                "TRANSFERABLE_SAVE",
                "SIGNAL_SAVE",
            }:
                saves += 1
            elif full_decision.decision == "REJECT":
                rejects += 1

        return {
            "candidates_discovered": len(processable_results),
            "facial_yes_count": facial_yes,
            "facial_no_count": facial_no,
            "facial_borderline_count": facial_borderline,
            "saves_count": saves,
            "rejected_count": rejects,
        }

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _brief_raw(self) -> dict:
        from researcher.strategy import _brief_raw
        return _brief_raw(self.brief)


def _snippet_from_candidate(
    candidate: ResearcherCandidate,
    query: dict,
) -> ResearcherSnippet:
    """Build the facial-input view from a hydrated candidate."""

    return ResearcherSnippet(
        name=candidate.name,
        current_affiliation=candidate.affiliations[0] if candidate.affiliations else "",
        h_index=candidate.h_index,
        citation_count=candidate.citation_count,
        papers_in_window=candidate.papers_in_window,
        top_paper_titles=[p.title for p in candidate.top_papers[:5]],
        # Audit Move #4 R1: surface the arxiv_categories the acquisition
        # pipeline populated when an ArxivClient was wired. Empty list
        # when the OpenAlex-only path ran.
        arxiv_categories=list(candidate.arxiv_categories or []),
        profile_url=candidate.profile_url,
        source_query_id=int(query.get("id") or 0),
        source_query_name=str(query.get("name") or ""),
        identity_key=identity_key_for_candidate(candidate),
        evidence_summary=list(candidate.evidence),
    )


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def build_pipeline(
    *,
    brief_path: str | Path,
    state_dir: str | Path,
    openalex_polite_pool_email: str = "",
    facial_llm_caller: Callable[[str, str], Any] | None = None,
    full_llm_caller: Callable[[str, str], Any] | None = None,
    strategy_llm_caller: Callable[[str, str], dict] | None = None,
    enable_arxiv: bool = False,
    enable_semantic_scholar: bool = False,
    semantic_scholar_api_key: str = "",
    resume: bool = False,
    replan: bool = False,
    revisit: str | None = None,
    limits: dict | None = None,
    openalex_client=None,
    exa_client_factory=None,
    public_extract_caller=None,
    public_fetcher=None,
    page_fetcher=None,
    evidence_fetcher=None,
    stop_requested=None,
    cost_estimates=None,
    campaign_brief_id=None,
) -> tuple[ResearcherPipeline, int]:
    """Convenience constructor used by the session orchestrator.

    Loads the brief, opens the runtime state store, builds the bridge,
    and returns ``(pipeline, run_id)`` so the caller just calls
    :meth:`ResearcherPipeline.run`.

    OpenAlex and Exa serve frontier campaigns. Legacy arXiv/S2 enrichers
    remain opt-in; this constructor does not authorize live provider use.
    """

    brief = load_brief(str(brief_path))
    if campaign_brief_id:
        brief.id = campaign_brief_id
    state_dir_path = Path(state_dir)
    state_dir_path.mkdir(parents=True, exist_ok=True)
    db_path = state_dir_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    bridge = ResearcherRuntimeStateBridge(
        store=store,
        output_dir=state_dir_path,
        brief_id=brief.id or brief.role_title or Path(brief_path).stem,
        brief_name=brief.role_title or brief.id,
        brief_path=str(brief_path),
    )
    import os
    client = openalex_client or OpenAlexClient(polite_pool_email=openalex_polite_pool_email, api_key=os.environ.get("OPENALEX_API_KEY", ""))
    arxiv_client = ArxivClient() if enable_arxiv else None
    semantic_scholar_client = (
        SemanticScholarClient(api_key=semantic_scholar_api_key)
        if enable_semantic_scholar
        else None
    )
    pipeline = ResearcherPipeline(
        brief=brief,
        bridge=bridge,
        openalex_client=client,
        facial_llm_caller=facial_llm_caller,
        full_llm_caller=full_llm_caller,
        strategy_llm_caller=strategy_llm_caller,
        arxiv_client=arxiv_client,
        semantic_scholar_client=semantic_scholar_client,
        exa_client_factory=exa_client_factory,
        public_extract_caller=public_extract_caller,
        public_fetcher=public_fetcher,
        page_fetcher=page_fetcher,
        evidence_fetcher=evidence_fetcher,
        stop_requested=stop_requested or (lambda: False),
        cost_estimates=cost_estimates or {},
    )
    run_id = bridge.start_or_resume_run(resume=resume, replan=replan, limits=limits)
    if revisit:
        try:
            bridge.revisit(run_id=run_id, key=revisit)
        except BaseException:
            store.finish_run(run_id, "error", stop_reason="fatal_runtime_error")
            raise
    return pipeline, run_id
