"""Layer 1: Strategy trace — thesis formation, adaptations, exhaustion signals.

Writes session_*_strategy.jsonl with event types:
  strategy_formed, adaptation_checkpoint, pivot_decision, session_narrative
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


class StrategyLayer:
    def __init__(self, output_path: Path):
        self._path = output_path
        self._lessons: list[str] = []
        self._adaptations: list[dict] = []
        self._strategy_rationale: str = ""
        self._query_count: int = 0
        self._channel_distribution: dict[str, int] = {}

    def _write(self, event: dict):
        event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self._path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def record_strategy_formed(self, queries: list, rationale: str):
        self._strategy_rationale = rationale
        self._query_count = len(queries)
        channels: dict[str, int] = {}
        for q in queries:
            ch = getattr(q, "channel", "unknown")
            channels[ch] = channels.get(ch, 0) + 1
        self._channel_distribution = channels

        self._write({
            "event": "strategy_formed",
            "query_count": len(queries),
            "channel_distribution": channels,
            "rationale": rationale,
        })

    def record_adaptation(
        self,
        batch_report,
        new_queries: list,
        skipped_ids: list,
        rationale: str,
        cumulative_stats: dict,
    ):
        # Extract lesson from rationale
        if rationale:
            self._lessons.append(rationale)

        checkpoint = {
            "event": "adaptation_checkpoint",
            "checkpoint": len(self._adaptations) + 1,
            "batch_summary": batch_report.to_summary_text() if batch_report else "",
            "new_queries_added": len(new_queries),
            "skipped_ids": skipped_ids,
            "rationale": rationale,
            "cumulative_saves": cumulative_stats.get("saved", 0),
            "cumulative_discovered": cumulative_stats.get("candidates_discovered", 0),
            "lessons_learned": list(self._lessons),
        }
        self._adaptations.append(checkpoint)
        self._write(checkpoint)

    def record_pivot(self, query, reason: str, processed: int, total: int):
        self._write({
            "event": "pivot_decision",
            "query_id": query.id,
            "query_name": query.name,
            "reason": reason,
            "processed": processed,
            "total": total,
        })

    def record_session_narrative(self, stats: dict, queries: list):
        # Channel performance
        channel_saves: dict[str, int] = {}
        for q in queries:
            ch = getattr(q, "channel", "unknown")
            channel_saves[ch] = channel_saves.get(ch, 0) + len(q.saves)

        self._write({
            "event": "session_narrative",
            "total_queries": len(queries),
            "queries_completed": sum(1 for q in queries if q.status == "done"),
            "queries_skipped": sum(1 for q in queries if q.status == "skipped"),
            "initial_rationale": self._strategy_rationale,
            "channel_distribution": self._channel_distribution,
            "channel_saves": channel_saves,
            "adaptations_count": len(self._adaptations),
            "lessons_learned": list(self._lessons),
            "final_stats": stats,
        })
