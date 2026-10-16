"""SessionObserver — event router that holds all layer instances.

Single observer object held by GitHubPipeline. All orchestrator print statements
are replaced with observer method calls. The observer routes each event to the
appropriate layer(s) and console.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from shared.brief_loader import Brief
from github.observability.strategy_layer import StrategyLayer
from github.observability.graph_layer import GraphLayer
from github.observability.candidate_layer import CandidateLayer
from github.observability.metrics_layer import MetricsLayer
from github.observability.console import ConsoleOutput, mask_email
from github.observability import report as report_mod


class SessionObserver:
    def __init__(self, session_id: str, output_dir: Path, brief: Brief):
        self.session_id = session_id
        self.output_dir = output_dir
        self._brief = brief
        self._start_time = time.time()

        # File paths
        prefix = f"session_{session_id}"
        self._strategy_path = output_dir / f"{prefix}_strategy.jsonl"
        self._graph_path = output_dir / f"{prefix}_graph.json"
        self._candidates_path = output_dir / f"{prefix}_candidates.json"
        self._metrics_path = output_dir / f"{prefix}_metrics.jsonl"
        self._report_path = output_dir / f"{prefix}_report.md"

        # Layer instances
        self.strategy = StrategyLayer(self._strategy_path)
        self.graph = GraphLayer(self._graph_path)
        self.candidates = CandidateLayer(self._candidates_path)
        self.metrics = MetricsLayer(self._metrics_path)
        self.console = ConsoleOutput()

        # Track current query for result_rank
        self._current_query_rank: dict[int, int] = {}  # query_id -> next rank

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_session_start(self, brief: Brief, query_count: int):
        self.console.emit_session_start(brief.id, query_count)

    def on_strategy_formed(self, queries: list, rationale: str):
        self.strategy.record_strategy_formed(queries, rationale)
        # Register all queries in graph
        for q in queries:
            self.graph.register_query(q)

    def on_session_end(self, stats: dict, progress, bias_summary: Optional[dict] = None):
        duration = time.time() - self._start_time
        queries = progress.queries if progress else []

        # Strategy narrative
        self.strategy.record_session_narrative(stats, queries)

        # Final metrics
        api_status = self._get_api_status_from_stats(stats)
        self.metrics.write_final(api_status, stats, bias_summary=bias_summary)

        # Write graph and candidates
        self.graph.write()
        self.candidates.write()

        # Console summary
        self.console.emit_session_end(stats, duration)

        # Generate report
        try:
            report_mod.generate_report(
                strategy_path=self._strategy_path,
                graph_path=self._graph_path,
                candidates_path=self._candidates_path,
                metrics_path=self._metrics_path,
                report_path=self._report_path,
                session_id=self.session_id,
                duration_seconds=duration,
            )
            self.console.emit_info(f"Report: {self._report_path}")
        except Exception as e:
            self.console.emit_error("report", str(e))

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def on_query_start(self, query, session_stats: dict, api_status: dict):
        total_queries = session_stats.get("total_queries", 0)
        saves = session_stats.get("saved", 0)
        fy = session_stats.get("facial_yes", 0)
        save_rate = f"{saves / fy * 100:.1f}%" if fy > 0 else "n/a"
        api_rest = api_status.get("rest", 0)

        self.console.emit_query_start(query, query.id, total_queries, saves, save_rate, api_rest)

        # Register in graph if not already
        self.graph.register_query(query)
        self._current_query_rank[query.id] = 1

    def on_query_results(self, query, usernames: list[str], pre_dedup_count: int):
        self.metrics.record_query_results(pre_dedup_count, len(usernames))

    def on_query_end(self, query, query_stats: dict):
        self.console.emit_query_end(query, query_stats)
        self.metrics.record_query_end(query, query_stats)

    def on_query_stopped_early(self, query, reason: str, processed: int, total: int):
        self.console.emit_query_stopped(query, reason, processed, total)
        self.strategy.record_pivot(query, reason, processed, total)

    # ------------------------------------------------------------------
    # Candidate processing
    # ------------------------------------------------------------------

    def on_candidate_discovered(self, username: str, query):
        rank = self._current_query_rank.get(query.id, 1)
        self.graph.record_discovery(username, query, rank)
        self._current_query_rank[query.id] = rank + 1

    def on_prescreen_filtered(self, username: str, candidate, query):
        # No console output — appears in query end summary via stats
        pass

    def on_geo_filtered(self, username: str, location: str, query, stage: str):
        # No console output — appears in query end summary
        pass

    def on_insufficient_data(self, username: str, query):
        # No console output — appears in query end summary
        pass

    def on_facial_decision(self, username: str, decision: str, rationale: str, query):
        if decision == "FACIAL_NO":
            self.graph.record_facial_no(username)
            self.candidates.record_facial_no(username, rationale, query)
        # No console output for individual decisions

    def on_full_decision(self, username: str, candidate, decision, query):
        # Routed through on_save or on_reject
        pass

    def on_save(self, username: str, candidate, decision, query):
        contact_str = (
            ", ".join(mask_email(email) for email in candidate.contact.emails[:2])
            if candidate.contact.emails
            else "no email"
        )

        # Console
        self.console.emit_save(
            username,
            candidate.user.name,
            decision.confidence,
            decision.decision,
            decision.path,
            contact_str,
        )

        # Graph
        self.graph.record_save(username, query, decision.confidence, decision.path)

        # Candidates layer
        rank = self._current_query_rank.get(query.id, 0)
        self.candidates.record_save(username, candidate, decision, query, result_rank=rank)

        # Metrics
        self.metrics.record_candidate_evaluated()

    def on_reject(self, username: str, candidate, decision, query):
        # No console output
        self.graph.record_reject(username)
        self.metrics.record_candidate_evaluated()

        # Track close rejects (passed facial but rejected on full)
        self.candidates.record_close_reject(username, candidate, decision, query)

    # ------------------------------------------------------------------
    # Adaptation & graph
    # ------------------------------------------------------------------

    def on_adaptation(self, batch_report, new_queries: list, skipped_ids: list, rationale: str, cumulative_stats: dict):
        self.console.emit_adaptation(len(new_queries), len(skipped_ids), rationale)
        self.strategy.record_adaptation(batch_report, new_queries, skipped_ids, rationale, cumulative_stats)

        # Register new queries in graph
        for q in new_queries:
            self.graph.register_query(q)

    def on_graph_expansion_queued(self, username: str, confidence: float, capability_area: str):
        self.graph.record_expansion_queued(username, confidence, capability_area)

    def on_graph_expansion_processed(self, seeds: list[dict], new_query_count: int):
        self.console.emit_graph_expansion(len(seeds), new_query_count)
        self.graph.record_expansion_processed(seeds, new_query_count)

    # ------------------------------------------------------------------
    # Operational
    # ------------------------------------------------------------------

    def on_enrichment_only_mode(self):
        self.console.emit_enrichment_only()

    def on_enrichment(self):
        self.metrics.record_enrichment()

    def on_enrichment_failure(self, kind: str = "") -> None:
        self.metrics.record_enrichment_failure(kind)

    def on_outreach_failure(self, username: str, query):
        self.console.emit_warn(f"Outreach generation failed for {username}")

    def on_error(self, context: str, error, query=None):
        self.console.emit_error(context, str(error))

    def on_result_cap(self, query, total: int):
        self.console.emit_warn(f"Q {query.id} hit 1,000 result cap ({total} total). Consider segmenting.")

    # ------------------------------------------------------------------
    # Metrics checkpoint (called from orchestrator at adaptation time)
    # ------------------------------------------------------------------

    def write_metrics_checkpoint(self, api_status: dict, stats: dict):
        stop_rec = self.metrics.write_checkpoint(api_status, stats)
        self.console.emit_stop_recommendation(stop_rec)
        return stop_rec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_api_status_from_stats(self, stats: dict) -> dict:
        return stats.get("api_status", {"rest": 0, "search": 0, "code_search": 0})
