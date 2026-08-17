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
from shared.storage import log_event


logger = logging.getLogger(__name__)


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
        facial_decisions = researcher_facial_judge_batch(
            snippets,
            brief=self.brief,
            source_config=source_config,
            llm_caller=self.facial_llm_caller,
        )

        facial_yes = facial_no = facial_borderline = 0
        saves = rejects = 0

        for kept_result, snippet, decision in zip(
            processable_results, snippets, facial_decisions
        ):
            self.bridge.record_facial_decision(
                run_id=run_id,
                snippet=snippet,
                decision=decision,
            )
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
        raw = getattr(self.brief, "_new_brief", None)
        if isinstance(raw, dict):
            return raw
        if hasattr(raw, "raw_dict"):
            try:
                candidate = raw.raw_dict()
                if isinstance(candidate, dict):
                    return candidate
            except Exception:  # noqa: BLE001 - defensive
                pass
        return {}


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
    enable_arxiv: bool = True,
    enable_semantic_scholar: bool = True,
    semantic_scholar_api_key: str = "",
) -> tuple[ResearcherPipeline, int]:
    """Convenience constructor used by the session orchestrator.

    Loads the brief, opens the runtime state store, builds the bridge,
    and returns ``(pipeline, run_id)`` so the caller just calls
    :meth:`ResearcherPipeline.run`.

    Audit Move #4 R1: arXiv and Semantic Scholar are wired by default
    so a customer-launched run gets the full multi-source spine. The
    enable flags exist so tests / one-off runs can opt out without
    swapping client objects through the call site.
    """

    brief = load_brief(str(brief_path))
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
    run_id = bridge.start_or_resume_run(resume=False)

    client = OpenAlexClient(polite_pool_email=openalex_polite_pool_email)
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
    )
    return pipeline, run_id
