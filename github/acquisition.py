"""GitHub acquisition service.

Keeps enrichment and pre-evaluation gating logic out of the orchestrator while
preserving the current query loop and shared execution runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.contact_discovery import merge_profile_contact
from shared.execution import AcquisitionResult
from shared.failures import ApiBudgetExhaustedError
from shared.storage import append_jsonl, log_event

if TYPE_CHECKING:
    from github.enricher import GitHubEnricher
    from github.orchestrator import GitHubPipeline
    from github.schemas import GitHubProgress, GitHubSearchQuery


class GitHubAcquisitionService:
    """Owns GitHub enrichment and candidate-preparation behavior."""

    def __init__(self, pipeline: "GitHubPipeline"):
        self.pipeline = pipeline

    async def prepare_candidate_for_evaluation(
        self,
        enricher: "GitHubEnricher",
        username: str,
        query: "GitHubSearchQuery",
        progress: "GitHubProgress",
        *,
        result_rank: int = 0,
    ) -> AcquisitionResult:
        pipeline = self.pipeline

        progress.candidates_discovered += 1
        pipeline.stats["candidates_discovered"] += 1
        source_query = query.query or query.target_repo or query.target_org
        runtime_cursor = pipeline._build_runtime_cursor(query, result_rank)
        envelope = pipeline._execution_envelope(
            username=username,
            query=query,
            result_rank=result_rank,
        )
        if getattr(pipeline, "_runtime_run_id", None):
            pipeline._execution_engine.runtime.record_discovery(
                envelope,
                payload=runtime_cursor,
            )
            prep_attempt_id = pipeline._execution_engine.runtime.start_stage(
                envelope,
                stage="preparation",
                payload={"cursor": runtime_cursor},
            )
        else:
            prep_attempt_id = None

        try:
            candidate = await enricher.light_enrich(
                username,
                source_strategy=query.channel,
                source_query=source_query,
            )
        except ApiBudgetExhaustedError:
            raise
        except Exception as exc:
            pipeline._finish_runtime_failure(
                attempt_id=prep_attempt_id,
                username=username,
                query=query,
                result_rank=result_rank,
                error=exc,
                payload={"cursor": runtime_cursor},
            )
            pipeline._in_flight_usernames.discard(username)
            raise

        if not candidate:
            if prep_attempt_id:
                pipeline._execution_engine.runtime.finish_stage_failure(
                    attempt_id=prep_attempt_id,
                    envelope=envelope,
                    stage="preparation",
                    error_or_failure_decision=RuntimeError("light_enrich returned no candidate"),
                    extra_payload={
                        "cursor": runtime_cursor,
                        "failure_kind_override": "empty_enrichment",
                    },
                )
            pipeline._in_flight_usernames.discard(username)
            return AcquisitionResult(
                terminal_decision="EMPTY_ENRICHMENT",
                skip_reason="light_enrich returned no candidate",
                metadata={"prep_attempt_id": prep_attempt_id},
            )

        if query.channel != "user_search" and not pipeline._passes_geography_check(candidate, query, stage="light"):
            pipeline.stats.setdefault("geo_filtered", 0)
            pipeline.stats["geo_filtered"] += 1
            pipeline.stats.setdefault("geo_filtered_light", 0)
            pipeline.stats["geo_filtered_light"] += 1
            pipeline._observer.on_geo_filtered(username, candidate.user.location or "no location", query, "light")
            pipeline._finish_preparation_terminal(
                attempt_id=prep_attempt_id,
                username=username,
                decision="GEO_FILTERED",
                query=query,
                candidate=candidate,
                result_rank=result_rank,
            )
            pipeline._mark_terminal(username)
            return AcquisitionResult(
                candidate=candidate,
                terminal_decision="GEO_FILTERED",
                skip_reason="light geography filter",
                metadata={"prep_attempt_id": prep_attempt_id},
            )

        prescreen = pipeline._prescreen_light(candidate)
        if prescreen == "hard_skip":
            pipeline.stats.setdefault("prescreen_filtered", 0)
            pipeline.stats["prescreen_filtered"] += 1
            pipeline._observer.on_prescreen_filtered(username, candidate, query)
            pipeline._finish_preparation_terminal(
                attempt_id=prep_attempt_id,
                username=username,
                decision="PRESCREEN_SKIP",
                query=query,
                candidate=candidate,
                result_rank=result_rank,
            )
            pipeline._mark_terminal(username)
            return AcquisitionResult(
                candidate=candidate,
                terminal_decision="PRESCREEN_SKIP",
                skip_reason="prescreen hard skip",
                metadata={"prep_attempt_id": prep_attempt_id},
            )

        try:
            candidate = await enricher.full_enrich(candidate)
        except ApiBudgetExhaustedError:
            raise
        except Exception as exc:
            pipeline._finish_runtime_failure(
                attempt_id=prep_attempt_id,
                username=username,
                query=query,
                result_rank=result_rank,
                candidate=candidate,
                error=exc,
                payload={
                    "cursor": runtime_cursor,
                    "candidate_record": pipeline._candidate_record(candidate),
                },
            )
            pipeline._in_flight_usernames.discard(username)
            raise

        # OSS Maintainers Slice 8: pass bio + profile README so the
        # cross-source resolver picks up LinkedIn URLs the recruiter
        # included in prose, not just the blog field. Provenance
        # ("blog" / "bio" / "readme") lands on
        # ``contact.linkedin_url_source`` for downstream banding.
        candidate.contact = merge_profile_contact(
            candidate.contact,
            candidate.user.email,
            candidate.user.twitter_username,
            candidate.user.blog,
            bio=candidate.user.bio,
            readme_text=candidate.readme_text,
        )

        pipeline._governor.record_enrichment()
        pipeline._observer.on_enrichment()
        progress.candidates_enriched += 1
        pipeline.stats["candidates_enriched"] += 1

        candidate_record = pipeline._candidate_record(candidate)
        append_jsonl(pipeline.candidates_path, candidate_record)

        if not pipeline._passes_geography_check(candidate, query, stage="full"):
            pipeline.stats.setdefault("geo_filtered", 0)
            pipeline.stats["geo_filtered"] += 1
            pipeline._observer.on_geo_filtered(username, candidate.user.location or "no location", query, "full")
            pipeline._finish_preparation_terminal(
                attempt_id=prep_attempt_id,
                username=username,
                decision="GEO_FILTERED",
                query=query,
                candidate=candidate,
                result_rank=result_rank,
                candidate_record=candidate_record,
            )
            pipeline._mark_terminal(username)
            return AcquisitionResult(
                candidate=candidate,
                candidate_record=candidate_record,
                terminal_decision="GEO_FILTERED",
                skip_reason="full geography filter",
                metadata={"prep_attempt_id": prep_attempt_id},
            )

        if candidate.data_sufficiency == "insufficient":
            pipeline.stats["insufficient"] += 1
            progress.candidates_insufficient += 1
            log_event(pipeline.log_path, "insufficient_data", username=username)
            pipeline._observer.on_insufficient_data(username, query)
            pipeline._finish_preparation_terminal(
                attempt_id=prep_attempt_id,
                username=username,
                decision="INSUFFICIENT_DATA",
                query=query,
                candidate=candidate,
                result_rank=result_rank,
                candidate_record=candidate_record,
            )
            pipeline._mark_terminal(username)
            return AcquisitionResult(
                candidate=candidate,
                candidate_record=candidate_record,
                terminal_decision="INSUFFICIENT_DATA",
                skip_reason="insufficient data",
                metadata={"prep_attempt_id": prep_attempt_id},
            )

        if prep_attempt_id:
            pipeline._execution_engine.runtime.finish_attempt_success(
                attempt_id=prep_attempt_id,
                envelope=pipeline._execution_envelope(
                    username=username,
                    query=query,
                    result_rank=result_rank,
                    candidate=candidate,
                    metadata={"candidate_record": candidate_record},
                ),
                new_state="snippet_extracted",
                payload={
                    "cursor": runtime_cursor,
                    "candidate_record": candidate_record,
                },
            )

        return AcquisitionResult(
            candidate=candidate,
            candidate_record=candidate_record,
            metadata={"prep_attempt_id": prep_attempt_id},
        )
