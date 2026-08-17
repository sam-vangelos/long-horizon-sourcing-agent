"""Layer 4: Operational metrics — API budget, rolling save rate, cost per save, stop signal.

Appends to session_*_metrics.jsonl at each adaptation checkpoint + session end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


class MetricsLayer:
    def __init__(self, output_path: Path):
        self._path = output_path
        self._checkpoint_num = 0
        self._queries_completed = 0
        self._cumulative_saves = 0
        self._yield_curve: list[dict] = []
        self._query_save_counts: list[int] = []  # saves per query in order
        self._total_candidates_evaluated = 0
        self._enrichment_failures = 0
        self._enrichment_failures_by_kind: dict[str, int] = {}
        self._total_enrichments = 0
        self._dedup_filtered = 0
        self._total_pre_dedup = 0

        # For rolling save rate
        self._recent_query_saves: list[int] = []  # last N query save counts

        # Stop recommendation history
        self._consecutive_low_rate = 0

    def _write(self, event: dict):
        event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self._path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def record_query_end(self, query, query_stats: dict):
        self._queries_completed += 1
        saves = len(query.saves)
        self._query_save_counts.append(saves)
        self._recent_query_saves.append(saves)

        # Track yield curve
        self._cumulative_saves += saves
        total_processed = query_stats.get("candidates_discovered", 0)
        marginal = saves / total_processed if total_processed > 0 else 0
        self._yield_curve.append({
            "query_id": query.id,
            "cumulative_saves": self._cumulative_saves,
            "marginal_rate": round(marginal, 4),
        })

    def record_query_results(self, pre_dedup: int, post_dedup: int):
        self._total_pre_dedup += pre_dedup
        self._dedup_filtered += (pre_dedup - post_dedup)

    def record_enrichment(self):
        self._total_enrichments += 1

    def record_enrichment_failure(self, kind: str = "") -> None:
        self._enrichment_failures += 1
        self._enrichment_failures_by_kind[kind] = self._enrichment_failures_by_kind.get(kind, 0) + 1

    def record_candidate_evaluated(self):
        self._total_candidates_evaluated += 1

    def write_checkpoint(self, api_status: dict, stats: dict):
        self._checkpoint_num += 1

        dedup_rate = self._dedup_filtered / self._total_pre_dedup if self._total_pre_dedup > 0 else 0

        # Rolling save rate (last 10 queries)
        last_10 = self._recent_query_saves[-10:]
        rolling_total = sum(last_10)
        rolling_rate = rolling_total / (len(last_10) * 10) if last_10 else 0  # approx per-candidate

        # Cost per save
        cost_api = self._total_enrichments / self._cumulative_saves if self._cumulative_saves > 0 else 0
        cost_evals = self._total_candidates_evaluated / self._cumulative_saves if self._cumulative_saves > 0 else 0

        # Enrichment failure rate
        efr = self._enrichment_failures / self._total_enrichments if self._total_enrichments > 0 else 0

        # Stop recommendation
        stop_rec = self._evaluate_stop_signal(api_status, dedup_rate)

        checkpoint = {
            "event": "checkpoint",
            "checkpoint": self._checkpoint_num,
            "queries_completed": self._queries_completed,
            "api_budget": api_status,
            "dedup_rate": round(dedup_rate, 3),
            "cumulative_saves": self._cumulative_saves,
            "rolling_save_rate_last_10q": round(rolling_rate, 4),
            "cost_per_save": {
                "api_calls": round(cost_api, 1),
                "candidates_evaluated": round(cost_evals, 1),
            },
            "yield_curve": self._yield_curve[-10:],  # last 10 entries
            "enrichment_failure_rate": round(efr, 3),
            "enrichment_failures_by_kind": dict(self._enrichment_failures_by_kind),
            "stop_recommendation": stop_rec,
        }
        self._write(checkpoint)
        return stop_rec

    def write_final(self, api_status: dict, stats: dict, bias_summary: Optional[dict] = None):
        """Write final metrics at session end."""
        dedup_rate = self._dedup_filtered / self._total_pre_dedup if self._total_pre_dedup > 0 else 0
        efr = self._enrichment_failures / self._total_enrichments if self._total_enrichments > 0 else 0

        event = {
            "event": "session_final",
            "queries_completed": self._queries_completed,
            "api_budget": api_status,
            "dedup_rate": round(dedup_rate, 3),
            "cumulative_saves": self._cumulative_saves,
            "total_enrichments": self._total_enrichments,
            "enrichment_failure_rate": round(efr, 3),
            "enrichment_failures_by_kind": dict(self._enrichment_failures_by_kind),
            "yield_curve": self._yield_curve,
            "final_stats": stats,
        }
        # P6.4 follow-up: only-when-present discipline — bias_summary is
        # None when no monitor was active this run (v1 brief, or a
        # construction failure upstream). Omit the key entirely rather than
        # writing a null placeholder that downstream renderers would have
        # to distinguish from "monitor ran with nothing to report."
        if bias_summary is not None:
            event["bias_summary"] = bias_summary
        self._write(event)

    def _evaluate_stop_signal(self, api_status: dict, dedup_rate: float) -> str:
        # Check rolling save rate
        last_10 = self._recent_query_saves[-10:]
        if len(last_10) >= 10 and sum(last_10) == 0:
            self._consecutive_low_rate += 1
        else:
            self._consecutive_low_rate = 0

        reasons = []
        if self._consecutive_low_rate >= 3:
            reasons.append("zero saves in last 30 queries")
        if dedup_rate > 0.50:
            reasons.append(f"dedup rate {dedup_rate:.0%}")
        rest_remaining = api_status.get("rest", 5000)
        if rest_remaining < 500:
            reasons.append(f"API budget low ({rest_remaining} rest calls)")

        if reasons:
            return f"STOP — {'; '.join(reasons)}"

        remaining_estimate = rest_remaining // 45 if rest_remaining > 0 else 0
        return f"CONTINUE — API budget sufficient for ~{remaining_estimate} more queries"
