"""Runnable Designer sourcing pipeline."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

from designer.acquisition import (
    acquire_behance_candidates,
    acquire_google_cse_candidates,
)
from designer.image_acquisition import (
    AssetAcquisitionResult,
    AssetCache,
    CachedAsset,
    acquire_behance_images_for_candidate,
    acquire_cse_thumbnails_for_candidate,
)
from designer.judging import designer_facial_judge, designer_full_judge
from designer.schemas import (
    DesignerCandidate,
    DesignerProjectSummary,
    DesignerSearchQuery,
    DesignerSnippet,
    behance_project_to_summary,
    cse_item_to_snippet,
)
from designer.sources.behance import BehanceClient
from designer.sources.google_cse import GoogleCSEClient
from designer.strategy import form_designer_strategy, select_designer_sources
from designer.recommendation_pitch import assemble_recommendation_pitch
from designer.vision_evaluation import (
    VisionEvaluationResult,
    VisionLLMCall,
    evaluate_designer_visually,
    gemini_vision_llm_call,
    resolve_vision_fallback,
)
from shared.adaptive import (
    AdaptiveAction,
    AdaptationDecision,
    NoiseMarker,
    ScoutMetrics,
    SignalMarker,
    record_adaptation_decision,
)
from shared.runtime_state.designer import DesignerRuntimeStateBridge
from shared.runtime_state.store import RuntimeStateStore
from shared.schemas import OpusDecision
from shared.storage import log_event


DesignerCandidateAcquirer = Callable[
    [DesignerSearchQuery],
    list[DesignerCandidate] | Awaitable[list[DesignerCandidate]],
]
DesignerAssetAcquirer = Callable[
    [DesignerCandidate],
    AssetAcquisitionResult | Awaitable[AssetAcquisitionResult],
]


@dataclass
class DesignerRunStats:
    queries_total: int = 0
    queries_completed: int = 0
    candidates_discovered: int = 0
    facial_yes: int = 0
    facial_no: int = 0
    facial_borderline: int = 0
    saves: int = 0
    rejects: int = 0
    insufficient_visual_evidence: int = 0
    vision_fallbacks: int = 0
    cost_usd: float = 0.0
    per_query: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DesignerPipeline:
    brief: dict[str, Any]
    bridge: DesignerRuntimeStateBridge
    queries: list[DesignerSearchQuery] | None = None
    behance_client: BehanceClient | None = None
    google_cse_client: GoogleCSEClient | None = None
    facial_llm_caller: Callable[[str, str], str | dict[str, Any]] | None = None
    full_llm_caller: Callable[[str, str], str | dict[str, Any]] | None = None
    vision_llm_call: VisionLLMCall = gemini_vision_llm_call
    vision_model: str = "gemini-3.1-pro-preview"
    candidate_acquirer: DesignerCandidateAcquirer | None = None
    asset_acquirer: DesignerAssetAcquirer | None = None
    log_path: Path | None = None
    max_candidates_per_query: int = 12
    vision_disabled: bool = False

    def __post_init__(self) -> None:
        if self.log_path is None:
            self.log_path = Path(self.bridge.output_dir) / "run_log.jsonl"

    def _probe_vision_provider(self) -> None:
        """Check whether vision can run; emit one ``provider_unavailable``
        event per run and flip ``vision_disabled=True`` when it can't.

        Without this probe, every candidate hits the live vision call,
        the call raises inside an ``except Exception``, and each one
        lands at ``INFERENTIAL_SAVE`` with the diagnostic buried in
        ``terminal_payload.visual_judgment.fallback_reason``. One event
        per run is the legible failure mode.
        """

        from shared import config  # noqa: PLC0415 - keep config import lazy

        primary_key = bool(getattr(config, "GOOGLE_API_KEY", ""))
        fallback = resolve_vision_fallback()
        # If a caller injected a stub for testing, never probe — tests
        # supply their own callable.
        if self.vision_llm_call is not gemini_vision_llm_call:
            return
        if primary_key or fallback is not None:
            return
        log_event(
            self.log_path,
            "provider_unavailable",
            provider="vision_primary",
            fallback_available=False,
            detail="GOOGLE_API_KEY unset and no DESIGNER_VISION_FALLBACK_MODEL_NAME configured",
        )
        self.vision_disabled = True

    async def run(self, *, run_id: int) -> DesignerRunStats:
        log_event(self.log_path, "pipeline_start", mode="full")
        self._probe_vision_provider()
        sources = self._configured_sources()
        queries = list(self.queries or form_designer_strategy(self.brief, sources=sources))
        stats = DesignerRunStats(queries_total=len(queries))

        finish_status = "completed"
        finish_stop_reason = "normal"
        end_status = "ok"
        try:
            async with AsyncExitStack() as stack:
                if self.behance_client is None and "behance" in sources:
                    try:
                        self.behance_client = await stack.enter_async_context(BehanceClient())
                    except Exception as exc:  # noqa: BLE001 - config-gated provider
                        log_event(
                            self.log_path,
                            "provider_unavailable",
                            provider="behance",
                            error=str(exc),
                            error_class=type(exc).__name__,
                        )
                if self.google_cse_client is None and "google_cse" in sources:
                    try:
                        self.google_cse_client = await stack.enter_async_context(GoogleCSEClient())
                    except Exception as exc:  # noqa: BLE001 - config-gated provider
                        log_event(
                            self.log_path,
                            "provider_unavailable",
                            provider="google_cse",
                            error=str(exc),
                            error_class=type(exc).__name__,
                        )

                index = 0
                while index < len(queries):
                    query = queries[index]
                    ordering_index = index + 1
                    work_unit_id = self.bridge.upsert_query_work_unit(
                        run_id=run_id,
                        query=query,
                        ordering_index=ordering_index,
                        status="in_progress",
                    )
                    per_query = await self._run_one_query(
                        run_id=run_id,
                        query=query,
                        work_unit_id=work_unit_id,
                    )
                    stats.queries_completed += 1
                    stats.candidates_discovered += per_query["candidates_discovered"]
                    stats.facial_yes += per_query["facial_yes_count"]
                    stats.facial_no += per_query["facial_no_count"]
                    stats.facial_borderline += per_query["facial_borderline_count"]
                    stats.saves += per_query["saves_count"]
                    stats.rejects += per_query["rejected_count"]
                    stats.insufficient_visual_evidence += per_query["insufficient_visual_evidence"]
                    stats.vision_fallbacks += per_query["vision_fallbacks"]
                    stats.cost_usd += float(per_query.get("cost_usd") or 0.0)
                    stats.per_query.append(per_query)
                    self.bridge.upsert_query_work_unit(
                        run_id=run_id,
                        query=query,
                        ordering_index=ordering_index,
                        status="done",
                        counters=per_query,
                    )
                    log_event(
                        self.log_path,
                        "string_complete",
                        string_id=ordering_index,
                        source=query.source,
                        query=query.query_text,
                        **per_query,
                    )
                    adapted = self._adapt_after_query(
                        run_id=run_id,
                        query=query,
                        current_ordering_index=ordering_index,
                        per_query=per_query,
                        remaining=queries[index + 1 :],
                    )
                    if adapted:
                        queries[index + 1 : index + 1] = adapted
                        stats.queries_total = len(queries)
                    index += 1
        except Exception as exc:  # noqa: BLE001 - telemetry must capture all failure paths
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
            # stats.cost_usd is already in as_dict(); avoid passing the
            # legacy zero override that masked Designer's true spend.
            log_event(
                self.log_path,
                "pipeline_end",
                status=end_status,
                **{k: v for k, v in stats.as_dict().items() if k != "per_query"},
            )
            self.bridge.store.finish_run(run_id, finish_status, stop_reason=finish_stop_reason)
        return stats

    async def _run_one_query(
        self,
        *,
        run_id: int,
        query: DesignerSearchQuery,
        work_unit_id: int,
    ) -> dict[str, int]:
        candidates = await self._acquire_candidates(query)
        facial_yes = facial_no = facial_borderline = saves = rejects = 0
        insufficient_visual = vision_fallbacks = 0
        vision_cost = 0.0

        for candidate in candidates[: self.max_candidates_per_query]:
            snippet = candidate.snippet
            self.bridge.record_discovery(
                run_id=run_id,
                work_unit_id=work_unit_id,
                snippet=snippet,
            )
            facial = designer_facial_judge(
                snippet,
                brief=self.brief,
                llm_caller=self.facial_llm_caller,
            )
            self.bridge.record_facial_decision(
                run_id=run_id,
                work_unit_id=work_unit_id,
                snippet=snippet,
                decision=facial,
            )
            if facial.decision == "FACIAL_NO":
                facial_no += 1
                continue
            if facial.decision == "FACIAL_BORDERLINE":
                facial_borderline += 1
            else:
                facial_yes += 1

            full = designer_full_judge(
                candidate,
                brief=self.brief,
                llm_caller=self.full_llm_caller,
            )
            terminal_decision, terminal_payload = await self._terminal_from_full_and_vision(
                candidate,
                full,
            )
            if terminal_payload.get("visual_judgment", {}).get("fallback_reason"):
                vision_fallbacks += 1
            if terminal_payload.get("visual_judgment", {}).get("fallback_reason") == "no_images_acquired":
                insufficient_visual += 1
            vision_cost += float(
                terminal_payload.get("visual_judgment", {}).get("cost_estimate_usd")
                or 0.0
            )
            full.decision = terminal_decision
            terminal_payload["full_decision"] = full.to_dict()
            self.bridge.record_full_decision(
                run_id=run_id,
                work_unit_id=work_unit_id,
                candidate=candidate,
                decision=full,
                terminal_payload=terminal_payload,
            )
            if terminal_decision in {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}:
                saves += 1
            else:
                rejects += 1

        return {
            "candidates_discovered": len(candidates),
            "facial_yes_count": facial_yes,
            "facial_no_count": facial_no,
            "facial_borderline_count": facial_borderline,
            "saves_count": saves,
            "rejected_count": rejects,
            "insufficient_visual_evidence": insufficient_visual,
            "vision_fallbacks": vision_fallbacks,
            "cost_usd": vision_cost,
        }

    async def _acquire_candidates(self, query: DesignerSearchQuery) -> list[DesignerCandidate]:
        if self.candidate_acquirer is not None:
            result = self.candidate_acquirer(query)
            if asyncio.iscoroutine(result):
                result = await result
            return list(result)

        candidates: list[DesignerCandidate] = []
        if query.source == "behance" and self.behance_client is not None:
            async for snippet in acquire_behance_candidates([query], client=self.behance_client):
                candidates.append(await self._candidate_from_behance_snippet(snippet))
        elif query.source == "google_cse" and self.google_cse_client is not None:
            async for snippet, raw_items in self._acquire_cse_with_raw_items(query):
                candidates.append(DesignerCandidate(
                    snippet=snippet,
                    raw_payload={"cse_result_items": raw_items},
                ))
        return candidates

    async def _candidate_from_behance_snippet(
        self,
        snippet: DesignerSnippet,
    ) -> DesignerCandidate:
        if self.behance_client is None:
            return DesignerCandidate(snippet=snippet)
        username = snippet.identity_key.split(":", 1)[-1]
        project_summaries: list[DesignerProjectSummary] = []
        project_responses: list[dict[str, Any]] = []
        try:
            status, body = await self.behance_client.get_user_projects(username)
        except Exception:
            status, body = 0, None
        if status == 200 and isinstance(body, dict):
            projects = body.get("projects") or []
            if isinstance(projects, list):
                projects = sorted(
                    [p for p in projects if isinstance(p, dict)],
                    key=lambda p: (
                        -(int((p.get("stats") or {}).get("appreciations", 0) or 0)),
                        -(int(p.get("published_on") or 0)),
                    ),
                )
                for project in projects[:4]:
                    if isinstance(project, dict):
                        project_summaries.append(behance_project_to_summary(project))
                        project_id = project.get("id")
                        if project_id:
                            try:
                                project_status, project_body = await self.behance_client.get_project(project_id)
                            except Exception:
                                project_status, project_body = 0, None
                            if project_status == 200 and isinstance(project_body, dict):
                                project_responses.append(project_body)
        return DesignerCandidate(
            snippet=snippet,
            project_summaries=tuple(project_summaries),
            raw_payload={"project_responses": project_responses},
        )

    async def _acquire_cse_with_raw_items(
        self,
        query: DesignerSearchQuery,
    ) -> AsyncIterator[tuple[DesignerSnippet, list[dict[str, Any]]]]:
        """Yield ``(snippet, raw_items)`` pairs for CSE candidates.

        Mirrors the host-fanout logic of
        :func:`designer.acquisition.acquire_google_cse_candidates` but
        preserves the raw CSE result items so
        :func:`designer.image_acquisition.acquire_cse_thumbnails_for_candidate`
        can extract ``pagemap.cse_thumbnail`` metadata downstream.
        """

        from designer.sources.google_cse import PORTFOLIO_HOST_DOMAINS

        assert self.google_cse_client is not None
        seen: set[str] = set()

        for host in PORTFOLIO_HOST_DOMAINS:
            try:
                status, body = await self.google_cse_client.search(
                    query=query.query_text,
                    site_filter=host,
                    start=1,
                )
            except Exception:
                continue
            if status != 200 or not isinstance(body, dict):
                continue
            items = body.get("items") or []
            if not isinstance(items, list):
                continue
            for item in items[:10]:
                if not isinstance(item, dict):
                    continue
                snippet = cse_item_to_snippet(item)
                if snippet is None:
                    continue
                if snippet.identity_key in seen:
                    continue
                seen.add(snippet.identity_key)
                yield snippet, [item]

    async def _terminal_from_full_and_vision(
        self,
        candidate: DesignerCandidate,
        full_decision: OpusDecision,
    ) -> tuple[str, dict[str, Any]]:
        terminal_payload: dict[str, Any] = {
            "full_decision": full_decision.to_dict(),
            "candidate_record": {
                "snippet": {
                    "identity_key": candidate.snippet.identity_key,
                    "display_name": candidate.snippet.display_name,
                    "profile_url": candidate.snippet.profile_url,
                }
            },
        }
        if full_decision.decision == "REJECT":
            return "REJECT", terminal_payload

        terminal_payload["surface_type"] = "hitl_visual_review"

        if self.vision_disabled:
            terminal_payload["visual_judgment"] = {
                "model": self.vision_model,
                "principles": [],
                "overall_verdict": "borderline",
                "overall_confidence": 0.0,
                "fallback_reason": "provider_unavailable_pre_check",
                "skipped": True,
            }
            terminal_decision = "INFERENTIAL_SAVE"
        else:
            assets = await self._acquire_assets(candidate)
            image_assets = [asset for asset in assets.cached_assets if asset.image_bytes]
            if not image_assets:
                terminal_payload["visual_judgment"] = {
                    "model": self.vision_model,
                    "principles": [],
                    "overall_verdict": "borderline",
                    "overall_confidence": 0.0,
                    "fallback_reason": "no_images_acquired",
                }
                terminal_decision = "INFERENTIAL_SAVE"
            else:
                fallback = resolve_vision_fallback()
                fallback_call = fallback[0] if fallback else None
                fallback_model = fallback[1] if fallback else ""
                result = evaluate_designer_visually(
                    brief=self.brief,
                    candidate_display_name=candidate.snippet.display_name,
                    candidate_headline=candidate.snippet.headline,
                    image_bytes_list=[asset.image_bytes or b"" for asset in image_assets],
                    asset_metadata=[
                        (asset.asset_url, asset.source, asset.project_title)
                        for asset in image_assets
                    ],
                    vision_llm_call=self.vision_llm_call,
                    model=self.vision_model,
                    vision_fallback_llm_call=fallback_call,
                    fallback_model=fallback_model,
                )
                terminal_payload["visual_judgment"] = _visual_result_payload(result)
                verdict = result.judgment.overall_verdict
                if verdict == "no":
                    return "REJECT", terminal_payload
                if verdict == "yes" and not result.judgment.fallback_reason:
                    terminal_decision = "SAVE"
                else:
                    terminal_decision = "INFERENTIAL_SAVE"

        # D5b: stamp the recommendation pitch onto every non-REJECT payload.
        role_title = self.brief.get("role_title", "")
        terminal_payload["full_decision"] = full_decision.to_dict()
        pitch = assemble_recommendation_pitch(terminal_payload, role_title=role_title)
        if pitch is not None:
            terminal_payload["recommendation_pitch"] = pitch

        return terminal_decision, terminal_payload

    async def _acquire_assets(self, candidate: DesignerCandidate) -> AssetAcquisitionResult:
        if self.asset_acquirer is not None:
            result = self.asset_acquirer(candidate)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        cache = AssetCache(Path(self.bridge.output_dir) / "assets.sqlite3")
        fetcher = _default_image_fetcher
        if candidate.snippet.source == "behance":
            return acquire_behance_images_for_candidate(
                candidate_identity_key=candidate.snippet.identity_key,
                project_responses=list(candidate.raw_payload.get("project_responses") or []),
                cache=cache,
                fetcher=fetcher,
            )
        if candidate.snippet.source == "google_cse":
            return acquire_cse_thumbnails_for_candidate(
                candidate_identity_key=candidate.snippet.identity_key,
                cse_result_items=list(candidate.raw_payload.get("cse_result_items") or []),
                cache=cache,
                fetcher=fetcher,
            )
        return AssetAcquisitionResult()

    def _adapt_after_query(
        self,
        *,
        run_id: int,
        query: DesignerSearchQuery,
        current_ordering_index: int,
        per_query: dict[str, int],
        remaining: list[DesignerSearchQuery],
    ) -> list[DesignerSearchQuery]:
        new_queries: list[DesignerSearchQuery] = []
        action = AdaptiveAction.CONTINUE
        rationale = "Designer query produced enough signal to continue."
        noise_markers: list[NoiseMarker] = []
        signal_markers: list[SignalMarker] = []

        if per_query["saves_count"] > 0:
            action = AdaptiveAction.COMMIT
            signal_markers.append(
                SignalMarker(
                    kind="portfolio_save",
                    label="designer saves",
                    count=per_query["saves_count"],
                )
            )
        elif per_query["candidates_discovered"] == 0 and not query.extra_filters.get("adapted"):
            action = AdaptiveAction.BROADEN
            rationale = "Designer query returned no candidates; added a broader portfolio lane."
            noise_markers.append(
                NoiseMarker(kind="sparse_query", label="zero candidates", count=1)
            )
            if not any(q.query_text == f"{query.query_text} portfolio case study" for q in remaining):
                new_queries.append(
                    DesignerSearchQuery(
                        source=query.source,
                        query_text=f"{query.query_text} portfolio case study",
                        sort=query.sort,
                        capability_area_name=query.capability_area_name,
                        discipline=query.discipline,
                        extra_filters={**query.extra_filters, "adapted": True},
                    )
                )
        elif per_query["vision_fallbacks"] or per_query["insufficient_visual_evidence"]:
            action = AdaptiveAction.EXPERIMENT
            rationale = "Designer text signal lacked grounded images; experiment with image-richer discovery."
            noise_markers.append(
                NoiseMarker(
                    kind="image_acquisition_failure",
                    label="ungrounded visual evidence",
                    count=per_query["vision_fallbacks"] + per_query["insufficient_visual_evidence"],
                )
            )

        # Pre-upsert each adapted query as a "queued" work unit so a crash
        # between adaptation and the next outer-loop iteration leaves the
        # canonical work_units table truthful on resume. The captured
        # work_unit_id is the stable integer ID we expose in
        # ``inserted_work_units`` for cross-source telemetry joins.
        inserted_ids: list[str] = []
        for offset, new_query in enumerate(new_queries):
            new_ordering_index = current_ordering_index + 1 + offset
            work_unit_id = self.bridge.upsert_query_work_unit(
                run_id=run_id,
                query=new_query,
                ordering_index=new_ordering_index,
                status="queued",
            )
            inserted_ids.append(str(work_unit_id))

        decision = AdaptationDecision(
            source="designer",
            action=action,
            lane=query.source,
            rationale=rationale,
            metrics=ScoutMetrics(
                work_units_run=1,
                candidates_discovered=per_query["candidates_discovered"],
                facial_yes=per_query["facial_yes_count"],
                facial_no=per_query["facial_no_count"],
                facial_borderline=per_query["facial_borderline_count"],
                saves=per_query["saves_count"],
                rejects=per_query["rejected_count"],
                signal_markers=signal_markers,
                noise_markers=noise_markers,
            ),
            work_unit_kind=self.bridge.query_kind(query),
            work_unit_family=query.capability_area_name or query.query_text,
            inserted_work_units=inserted_ids,
            source_payload={
                "query": asdict(query),
                "new_queries": [asdict(q) for q in new_queries],
            },
        )
        record_adaptation_decision(self.bridge.store, run_id=run_id, decision=decision)
        log_event(self.log_path, "adaptation_decision", **decision.to_dict())
        return new_queries

    def _configured_sources(self) -> tuple[str, ...]:
        if self.queries:
            return tuple(dict.fromkeys(query.source for query in self.queries))
        if self.behance_client is not None or self.google_cse_client is not None:
            sources: list[str] = []
            if self.google_cse_client is not None:
                sources.append("google_cse")
            if self.behance_client is not None:
                sources.append("behance")
            return tuple(sources)
        return select_designer_sources()


def _visual_result_payload(result: VisionEvaluationResult) -> dict[str, Any]:
    return {
        "model": result.judgment.model,
        "principles": [
            {
                "name": principle.name,
                "score": principle.score,
                "anchor": principle.anchor,
                "reasoning": principle.reasoning,
                "image_ids": list(principle.image_ids),
                "anchor_consistency_pass": principle.anchor_consistency_pass,
            }
            for principle in result.judgment.principles
        ],
        "overall_verdict": result.judgment.overall_verdict,
        "overall_confidence": result.judgment.overall_confidence,
        "fallback_reason": result.judgment.fallback_reason,
        "cost_estimate_usd": result.judgment.cost_estimate_usd,
        "assets": [
            {
                "id": ref.image_id,
                "url": ref.asset_url,
                "source": ref.source,
                "project_title": ref.project_title,
            }
            for ref in result.asset_references
        ],
    }


def _default_image_fetcher(url: str) -> bytes | None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ClorisDesigner/1.0"})
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - source URLs come from configured portfolio APIs
            return response.read(5_000_000)
    except Exception:
        return None


def load_brief_dict(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


_DEFAULT_RUBRIC_PATH = Path(__file__).resolve().parent.parent / "config" / "design-rubrics" / "default.json"


def _hydrate_default_rubric(brief: dict[str, Any]) -> None:
    """Inject the default design rubric into briefs that lack one.

    Only fires for briefs targeting the Designer module. If the brief
    already carries a ``design_rubric`` dict with a non-empty
    ``principles`` list, the recruiter-authored rubric is preserved
    byte-for-byte.
    """
    target_modules = brief.get("target_modules") or []
    if "designer" not in target_modules:
        return
    existing = brief.get("design_rubric")
    if isinstance(existing, dict) and existing.get("principles"):
        return
    with _DEFAULT_RUBRIC_PATH.open("r", encoding="utf-8") as fh:
        brief["design_rubric"] = json.load(fh)


def build_pipeline(
    *,
    brief_path: str | Path,
    state_dir: str | Path,
    resume: bool = False,
) -> tuple[DesignerPipeline, int]:
    from shared import config

    brief = load_brief_dict(brief_path)
    _hydrate_default_rubric(brief)
    state_dir_path = Path(state_dir)
    state_dir_path.mkdir(parents=True, exist_ok=True)
    brief_id = str(brief.get("id") or brief.get("role_title") or Path(brief_path).stem)
    store = RuntimeStateStore(state_dir_path / "runtime_state.sqlite3")
    bridge = DesignerRuntimeStateBridge(
        store=store,
        output_dir=state_dir_path,
        brief_id=brief_id,
        brief_name=str(brief.get("role_title") or brief_id),
        brief_path=str(brief_path),
    )
    run_id = bridge.start_or_resume_run(resume=resume)
    pipeline = DesignerPipeline(
        brief=brief,
        bridge=bridge,
        vision_model=getattr(config, "DESIGNER_VISION_MODEL_NAME", "gemini-3.1-pro-preview"),
    )
    return pipeline, run_id
