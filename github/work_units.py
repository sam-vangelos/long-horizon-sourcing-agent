"""GitHub work-unit coordination service."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from github.schemas import GitHubSearchQuery
from shared.execution import WorkUnitCheckpoint
from shared.runtime_state.github import (
    PersonKeySet,
    github_identity_keys_from_seen,
    github_person_key,
    person_key_seen,
)

if TYPE_CHECKING:
    from github.orchestrator import GitHubPipeline
    from github.schemas import GitHubProgress, GitHubSearchQuery


class GitHubWorkUnitService:
    """Owns GitHub progress, dedup, and graph-expansion work-unit semantics."""

    def __init__(self, pipeline: "GitHubPipeline"):
        self.pipeline = pipeline

    def load_or_create_progress(self, *, resume: bool = False):
        pipeline = self.pipeline
        pipeline._ensure_runtime_state()
        pipeline._runtime_run_id, progress = pipeline._runtime_bridge.start_or_resume_run(resume=resume)
        pipeline._seen_usernames = PersonKeySet(progress.discovered_usernames)
        pipeline._execution_engine.runtime.mark_progress_dirty()
        pipeline._execution_engine.runtime.flush_projections_if_needed(
            run_id=pipeline._runtime_run_id,
            force_artifacts=True,
        )
        return progress

    def save_progress(self) -> WorkUnitCheckpoint | None:
        pipeline = self.pipeline
        if not pipeline._progress:
            return None
        pipeline._progress.discovered_usernames = github_identity_keys_from_seen(
            pipeline._seen_usernames
        )
        if pipeline._client:
            pipeline._progress.api_calls_made = pipeline._client.limiter.total_calls
        if getattr(pipeline, "_runtime_run_id", None):
            pipeline._runtime_bridge.sync_progress(pipeline._runtime_run_id, pipeline._progress)
            pipeline._execution_engine.runtime.mark_progress_dirty()
            pipeline._execution_engine.runtime.flush_projections_if_needed(run_id=pipeline._runtime_run_id)
            pipeline._seen_usernames = PersonKeySet(
                pipeline._runtime_state.list_terminal_person_keys(
                    source="github",
                    brief_id=pipeline.brief_obj.id,
                )
            )
            pipeline._progress.discovered_usernames = list(
                pipeline._runtime_state.list_terminal_identity_keys(
                    source="github",
                    brief_id=pipeline.brief_obj.id,
                )
            )
            return WorkUnitCheckpoint(
                status="synced",
                cursor={"run_id": pipeline._runtime_run_id},
                metrics={"discovered_usernames": len(pipeline._seen_usernames)},
            )

        pipeline._progress.save(str(pipeline.progress_path))
        return WorkUnitCheckpoint(status="saved", cursor={}, metrics={})

    def dedup_usernames(self, usernames: list[str]) -> list[str]:
        pipeline = self.pipeline
        blocked = set()
        if getattr(pipeline, "_runtime_bridge", None):
            blocked = pipeline._runtime_bridge.load_blocked_usernames([u for u in usernames if u])
        new = []
        for username in usernames:
            if (
                username
                and username not in blocked
                and not person_key_seen(username, pipeline._seen_usernames)
                and username not in pipeline._in_flight_usernames
            ):
                pipeline._in_flight_usernames.add(username)
                new.append(username)
        return new

    async def process_graph_expansion_queue(
        self,
        progress: "GitHubProgress",
        queries: list["GitHubSearchQuery"],
    ) -> WorkUnitCheckpoint:
        pipeline = self.pipeline
        unprocessed = [
            entry for entry in progress.graph_expansion_queue
            if entry["username"] not in progress.graph_expansion_processed
        ]
        if not unprocessed:
            return WorkUnitCheckpoint(status="noop", payload={"seeds_processed": 0})

        unprocessed.sort(key=lambda item: item.get("confidence", 0), reverse=True)
        seeds = unprocessed[:5]

        next_id = max((query.id for query in queries), default=0) + 1
        new_queries = []
        for seed in seeds:
            username = seed["username"]
            new_queries.append(
                GitHubSearchQuery(
                    id=next_id,
                    name=f"Graph expansion: followers/following of {username}",
                    query=username,
                    channel="graph_expansion",
                )
            )
            next_id += 1

        if new_queries:
            current_idx = max(
                (i for i, query in enumerate(queries) if query.status in ("done", "in_progress")),
                default=0,
            )
            pipeline._insert_queries_by_priority(queries, new_queries, current_idx)
            pipeline._observer.on_graph_expansion_processed(seeds, len(new_queries))
            # P6.2: do NOT mark seeds processed here. This used to mark
            # `graph_expansion_processed` (in-memory + runtime_state) the
            # moment a graph_expansion query was CREATED for a seed, before
            # that query ever executed. GitHubPipeline._expand_graph
            # (orchestrator.py) refuses to re-process a seed already in
            # `graph_expansion_processed`, so every expansion query
            # self-cancelled on its first (and only) execution attempt,
            # returning (0, []) without ever calling get_followers/
            # get_following. The mark-processed step now lives in
            # `_expand_graph`, after the fetch actually runs.
        progress.queries = queries
        return WorkUnitCheckpoint(
            status="updated",
            metrics={"new_queries": len(new_queries)},
            payload={"processed_usernames": [seed["username"] for seed in seeds]},
        )

    def enqueue_graph_expansion_seed(
        self,
        *,
        progress: "GitHubProgress",
        username: str,
        reason: str,
        confidence: float,
        capability_area: str,
    ) -> WorkUnitCheckpoint:
        pipeline = self.pipeline
        progress.graph_expansion_queue.append(
            {
                "username": username,
                "reason": reason,
                "confidence": confidence,
                "capability_area": capability_area,
                "added_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )
        if pipeline._runtime_run_id:
            pipeline._runtime_state.enqueue_graph_expansion_seed(
                run_id=pipeline._runtime_run_id,
                username=username,
                reason=reason,
                confidence=confidence,
                capability_area=capability_area,
            )
        pipeline._observer.on_graph_expansion_queued(username, confidence, capability_area)
        return WorkUnitCheckpoint(
            status="queued",
            payload={"username": username, "reason": reason},
            metrics={"confidence": confidence},
        )
