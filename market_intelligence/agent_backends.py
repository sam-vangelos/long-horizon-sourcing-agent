"""Agentic planner, synthesis, and critic backends for market intelligence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from typing import Any

import shared.config as shared_config
# A.2 cache-gap remediation: planner / synthesis / critic each fire
# once per reflection cycle but their system prompts (~5-8KB each)
# overlap a lot with the brief polish + chief-of-staff calls in the
# same llm_usage_session window — caching catches the cross-stage
# overlap. Same signature as opus_llm.
from shared.llm_clients import opus_llm_cached as opus_llm
from shared.observability import observe

from market_intelligence.research_context import build_research_context_bundle
from market_intelligence.research_prompts import (
    build_critic_system_prompt,
    build_critic_user_prompt,
    build_internal_synthesis_system_prompt,
    build_internal_synthesis_user_prompt,
    build_planner_system_prompt,
    build_planner_user_prompt,
)
from market_intelligence.schema import (
    MarketEvidenceBatch,
    MarketIdentity,
    MarketIntelAgentState,
    MarketIntelArtifact,
    ResearchOpportunity,
    SectionGenerationMetadata,
    SourceRegistryEntry,
    market_thesis_summary_looks_like_review,
    sanitize_narrative_items,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _has_llm_access() -> bool:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return bool(getattr(shared_config, "ANTHROPIC_API_KEY", "").strip())


def _supporting_refs_from_batches(evidence_batches: list[MarketEvidenceBatch]) -> list[str]:
    return [
        batch.run_ref
        for batch in evidence_batches
        if _normalize_text(batch.run_ref)
    ]


@dataclass
class PlannerResult:
    planner_summary: str = ""
    active_hypotheses: list[dict] = field(default_factory=list)
    resolved_hypotheses: list[dict] = field(default_factory=list)
    open_unknowns: list[dict] = field(default_factory=list)
    research_backlog: list[dict] = field(default_factory=list)
    update_sections: list[str] = field(default_factory=list)
    confidence_ceiling_by_section: dict[str, float] = field(default_factory=dict)
    should_collect_external_research: bool = False
    external_research_focus: list[dict] = field(default_factory=list)
    should_collect_edge_case_research: bool = False
    edge_case_research_reasoning: str = ""
    edge_case_research_focus: list[dict] = field(default_factory=list)
    edge_case_confidence_ceiling: float | None = None

    def to_dict(self) -> dict:
        return {
            "planner_summary": self.planner_summary,
            "active_hypotheses": self.active_hypotheses,
            "resolved_hypotheses": self.resolved_hypotheses,
            "open_unknowns": self.open_unknowns,
            "research_backlog": self.research_backlog,
            "update_sections": self.update_sections,
            "confidence_ceiling_by_section": self.confidence_ceiling_by_section,
            "should_collect_external_research": self.should_collect_external_research,
            "external_research_focus": self.external_research_focus,
            "should_collect_edge_case_research": self.should_collect_edge_case_research,
            "edge_case_research_reasoning": self.edge_case_research_reasoning,
            "edge_case_research_focus": self.edge_case_research_focus,
            "edge_case_confidence_ceiling": self.edge_case_confidence_ceiling,
        }


@dataclass
class CriticResult:
    keep_sections: dict[str, Any] = field(default_factory=dict)
    section_generation_metadata: dict[str, dict] = field(default_factory=dict)
    delta_since_last_run: dict[str, list[str]] = field(default_factory=dict)
    confidence_by_claim_area: dict[str, float] = field(default_factory=dict)
    critique_summary: str = ""
    claim_adjudications: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "keep_sections": self.keep_sections,
            "section_generation_metadata": self.section_generation_metadata,
            "delta_since_last_run": self.delta_since_last_run,
            "confidence_by_claim_area": self.confidence_by_claim_area,
            "critique_summary": self.critique_summary,
            "claim_adjudications": self.claim_adjudications,
        }


class HeuristicPlannerBackend:
    """Local, evidence-led planning pass when an LLM is unavailable."""

    def plan(
        self,
        *,
        market_identity: MarketIdentity,
        deterministic_summary: dict,
        evidence_batches: list[MarketEvidenceBatch],
        previous_artifact: MarketIntelArtifact | None,
        previous_agent_state: MarketIntelAgentState | None,
    ) -> PlannerResult:
        bundle = build_research_context_bundle(market_identity, evidence_batches)
        supporting_refs = _supporting_refs_from_batches(evidence_batches)
        lane_summary = bundle.get("cross_run_aggregate", {}).get("lane_trend_summary", [])
        top_lane = lane_summary[0] if lane_summary else {}
        hypotheses: list[dict] = []
        if top_lane:
            statement = (
                f"The strongest recurring sourcing signal is currently in "
                f"{_normalize_text(top_lane.get('lane')) or 'the top lane'}."
            )
            hypotheses.append(
                {
                    "hypothesis_id": f"hyp-{_normalize_text(top_lane.get('lane')).lower().replace(' ', '-') or 'top-lane'}",
                    "statement": statement,
                    "status": "active",
                    "confidence": round(
                        min(
                            0.9,
                            0.45 + 0.1 * _safe_int(top_lane.get("winning_count"), 0),
                        ),
                        2,
                    ),
                    "rationale": "Repeated winning-lane evidence across completed runs.",
                    "section_targets": ["lane_intelligence", "brief_recommendations"],
                    "first_seen_at": evidence_batches[0].generated_at if evidence_batches else _utc_now(),
                    "last_seen_at": evidence_batches[-1].generated_at if evidence_batches else _utc_now(),
                    "supporting_run_refs": _supporting_refs_from_batches(
                        [
                            batch
                            for batch in evidence_batches
                            if batch.run_ref in set(top_lane.get("run_refs", []))
                        ]
                    )
                    or supporting_refs,
                }
            )

        unknowns: list[dict] = []
        for item in bundle.get("historical_rollup", {}).get("coverage_gap_themes", [])[:3]:
            label = _normalize_text(item.get("label"))
            if not label:
                continue
            unknowns.append(
                {
                    "question": f"How should we cover {label} more directly?",
                    "priority": "medium",
                    "next_step": "Run one targeted lane in the next sourcing cycle.",
                    "supporting_run_refs": supporting_refs,
                }
            )

        backlog: list[dict] = []
        employer_rollup = bundle.get("cross_run_aggregate", {}).get("employer_rollup", [])
        if not employer_rollup:
            backlog.append(
                {
                    "opportunity_id": "opp-employer-clusters",
                    "question": "Which employer clusters are actively hiring for this role family?",
                    "priority": "medium",
                    "status": "queued",
                    "reason": "Internal employer evidence is still thin.",
                    "supporting_run_refs": supporting_refs,
                }
            )
        if deterministic_summary.get("aggregate_metrics", {}).get("run_count", 0) >= 2:
            backlog.append(
                {
                    "opportunity_id": "opp-market-thesis",
                    "question": "Does external hiring activity confirm the current internal market thesis?",
                    "priority": "high",
                    "status": "queued",
                    "reason": "Cross-run signal is now strong enough to justify external validation.",
                    "supporting_run_refs": supporting_refs,
                }
            )

        update_sections = [
            "lane_intelligence",
            "candidate_signal_summary",
            "brief_recommendations",
            "open_questions",
        ]
        if employer_rollup:
            update_sections.append("employer_signal_intelligence")
        if deterministic_summary.get("aggregate_metrics", {}).get("saved_count", 0) >= 3:
            update_sections.append("talent_pool_intelligence")

        should_collect_external = bool(backlog) and deterministic_summary.get(
            "aggregate_metrics",
            {},
        ).get("saved_count", 0) > 0
        edge_case_context = bundle.get("edge_case_context", {})
        hidden_pool_signals = len(edge_case_context.get("hidden_pool_risk_signals", []))
        edge_case_lanes = len(edge_case_context.get("edge_case_lane_signals", []))
        title_fragmentation = len(
            edge_case_context.get("title_fragmentation_indicators", [])
        )
        false_negative_hypotheses = len(
            edge_case_context.get("false_negative_hypotheses_from_internal_evidence", [])
        )
        employer_gaps = len(bundle.get("employer_signal_gaps", []))
        candidate_blind_spots = len(
            edge_case_context.get("candidate_evidence_blind_spots", [])
        )
        trigger_score = 0
        trigger_notes: list[str] = []
        if hidden_pool_signals:
            trigger_score += 2
            trigger_notes.append("coverage gaps and hidden-pool risk signals are present")
        if edge_case_lanes:
            trigger_score += 2
            trigger_notes.append("novelty-heavy or edge-case lanes produced meaningful signal")
        if title_fragmentation:
            trigger_score += 1
            trigger_notes.append("observed titles suggest fragmented self-labeling")
        if false_negative_hypotheses:
            trigger_score += 2
            trigger_notes.append("internal evidence already points to plausible false negatives")
        if employer_gaps:
            trigger_score += 1
            trigger_notes.append("employer clustering is still thin")
        if candidate_blind_spots:
            trigger_score += 1
            trigger_notes.append("profile-level evidence is sparse")
        should_collect_edge_case = trigger_score >= 3 and bool(
            deterministic_summary.get("aggregate_metrics", {}).get("saved_count", 0)
        )
        edge_case_focus: list[dict] = []
        for section_name, default_reason in (
            ("hidden_pool_risk_signals", "Test whether a hidden submarket is being missed."),
            ("false_negative_hypotheses_from_internal_evidence", "Validate likely false-negative mechanisms."),
            ("title_fragmentation_indicators", "Check whether title drift is hiding relevant candidates."),
            ("self_labeling_risk_indicators", "Understand how relevant candidates may self-present differently."),
        ):
            for item in edge_case_context.get(section_name, [])[:2]:
                if not isinstance(item, dict):
                    continue
                label = _normalize_text(item.get("label"))
                summary = _normalize_text(item.get("summary"))
                if not (label or summary):
                    continue
                edge_case_focus.append(
                    {
                        "focus": label or summary,
                        "priority": "high" if section_name != "self_labeling_risk_indicators" else "medium",
                        "reason": summary or default_reason,
                        "supporting_run_refs": item.get("supporting_run_refs", supporting_refs),
                    }
                )
        edge_case_focus = edge_case_focus[:4]
        focus = [
            {
                "focus": item["question"],
                "priority": item["priority"],
                "reason": item["reason"],
                "supporting_run_refs": item.get("supporting_run_refs", []),
            }
            for item in backlog[:3]
        ]
        summary = (
            f"Tracking {len(hypotheses)} active hypotheses across "
            f"{deterministic_summary.get('aggregate_metrics', {}).get('run_count', 0)} run(s)."
        )
        return PlannerResult(
            planner_summary=summary,
            active_hypotheses=hypotheses,
            resolved_hypotheses=[
                item.to_dict() for item in (previous_agent_state.resolved_hypotheses if previous_agent_state else [])
            ],
            open_unknowns=unknowns,
            research_backlog=backlog,
            update_sections=sorted(set(update_sections)),
            confidence_ceiling_by_section={
                "market_thesis": 0.55
                if deterministic_summary.get("aggregate_metrics", {}).get("run_count", 0) <= 1
                else 0.72,
                "talent_pool_intelligence": 0.7,
                "employer_signal_intelligence": 0.68,
            },
            should_collect_external_research=should_collect_external,
            external_research_focus=focus,
            should_collect_edge_case_research=should_collect_edge_case,
            edge_case_research_reasoning=(
                "Planner-gated edge-case research triggered because "
                + "; ".join(trigger_notes[:3])
                + "."
                if should_collect_edge_case and trigger_notes
                else "Edge-case research not warranted from the current evidence."
            ),
            edge_case_research_focus=edge_case_focus,
            edge_case_confidence_ceiling=0.58 if should_collect_edge_case else 0.0,
        )


class LLMPlannerBackend:
    def __init__(self, fallback: HeuristicPlannerBackend | None = None) -> None:
        self.fallback = fallback or HeuristicPlannerBackend()

    @observe(name="market_intel.planner")
    def plan(
        self,
        *,
        market_identity: MarketIdentity,
        deterministic_summary: dict,
        evidence_batches: list[MarketEvidenceBatch],
        previous_artifact: MarketIntelArtifact | None,
        previous_agent_state: MarketIntelAgentState | None,
    ) -> PlannerResult:
        if not _has_llm_access():
            return self.fallback.plan(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                evidence_batches=evidence_batches,
                previous_artifact=previous_artifact,
                previous_agent_state=previous_agent_state,
            )
        try:
            bundle = build_research_context_bundle(market_identity, evidence_batches)
            raw = opus_llm(
                build_planner_system_prompt(),
                build_planner_user_prompt(
                    market_identity,
                    bundle,
                    previous_artifact.to_dict() if previous_artifact else None,
                    previous_agent_state.to_dict() if previous_agent_state else None,
                ),
                expect_json=True,
                max_tokens=12000,
                usage_context={
                    "stage": "market_intel_planner",
                    "market_key": market_identity.market_key,
                },
            )
            return _normalize_planner_result(raw, evidence_batches)
        except Exception:
            return self.fallback.plan(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                evidence_batches=evidence_batches,
                previous_artifact=previous_artifact,
                previous_agent_state=previous_agent_state,
            )


class LLMInternalSynthesisBackend:
    def __init__(self, fallback_backend: Any) -> None:
        self.fallback_backend = fallback_backend

    def synthesize(
        self,
        *,
        market_identity: MarketIdentity,
        deterministic_summary: dict,
        evidence_batches: list[MarketEvidenceBatch],
        previous_artifact: MarketIntelArtifact | None,
        planner_result: PlannerResult,
        external_research: Any | None,
    ) -> dict:
        if not _has_llm_access():
            return self.fallback_backend.synthesize(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                evidence_batches=evidence_batches,
                previous_artifact=previous_artifact,
                external_research=external_research,
            )
        try:
            bundle = build_research_context_bundle(market_identity, evidence_batches)
            raw = opus_llm(
                build_internal_synthesis_system_prompt(),
                build_internal_synthesis_user_prompt(
                    market_identity,
                    bundle,
                    planner_result.to_dict(),
                    previous_artifact.to_dict() if previous_artifact else None,
                ),
                expect_json=True,
                max_tokens=14000,
                usage_context={
                    "stage": "market_intel_synthesis",
                    "market_key": market_identity.market_key,
                },
            )
            return _normalize_draft_sections(raw)
        except Exception:
            return self.fallback_backend.synthesize(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                evidence_batches=evidence_batches,
                previous_artifact=previous_artifact,
                external_research=external_research,
            )


class HeuristicCriticBackend:
    def critique(
        self,
        *,
        market_identity: MarketIdentity,
        deterministic_summary: dict,
        evidence_batches: list[MarketEvidenceBatch],
        previous_artifact: MarketIntelArtifact | None,
        planner_result: PlannerResult,
        draft_sections: dict,
        external_research: Any | None,
    ) -> CriticResult:
        kept = _normalize_draft_sections(draft_sections)
        previous = previous_artifact.to_dict() if previous_artifact else {}
        for section in (
            "talent_pool_intelligence",
            "noise_patterns",
            "employer_signal_intelligence",
            "brief_recommendations",
            "open_questions",
        ):
            if not kept.get(section) and previous.get(section):
                kept[section] = previous.get(section, [])
        if (not kept.get("market_thesis") or not _normalize_text(
            kept.get("market_thesis", {}).get("summary")
        )) and previous.get("market_thesis"):
            kept["market_thesis"] = previous.get("market_thesis", {})
        metadata: dict[str, dict] = {}
        now = _utc_now()
        for section in (
            "lane_intelligence",
            "talent_pool_intelligence",
            "noise_patterns",
            "employer_signal_intelligence",
            "candidate_signal_summary",
            "market_thesis",
            "brief_recommendations",
            "open_questions",
        ):
            generation_mode = "llm_internal"
            if (
                external_research
                and section != "candidate_signal_summary"
                and _evidence_refs_for_section(section, kept)
            ):
                generation_mode = "llm_external"
            if not kept.get(section):
                generation_mode = "deterministic" if section == "candidate_signal_summary" else "heuristic"
            quality = _quality_for_section(
                section=section,
                evidence_batches=evidence_batches,
                kept=kept,
            )
            metadata[section] = SectionGenerationMetadata(
                generation_mode=generation_mode,
                quality_level=quality,
                updated_at=now,
                notes=_metadata_notes(
                    section=section,
                    evidence_batches=evidence_batches,
                    kept=kept,
                ),
                supporting_run_refs=_supporting_refs_for_section(section, kept, evidence_batches),
                evidence_refs=_evidence_refs_for_section(section, kept),
            ).to_dict()

        return CriticResult(
            keep_sections=kept,
            section_generation_metadata=metadata,
            delta_since_last_run=_build_delta(previous, kept, planner_result),
            confidence_by_claim_area=_confidence_by_claim_area(metadata),
            critique_summary=planner_result.planner_summary or "Heuristic critic completed.",
        )


class LLMCriticBackend:
    def __init__(self, fallback: HeuristicCriticBackend | None = None) -> None:
        self.fallback = fallback or HeuristicCriticBackend()

    def critique(
        self,
        *,
        market_identity: MarketIdentity,
        deterministic_summary: dict,
        evidence_batches: list[MarketEvidenceBatch],
        previous_artifact: MarketIntelArtifact | None,
        planner_result: PlannerResult,
        draft_sections: dict,
        external_research: Any | None,
    ) -> CriticResult:
        if not _has_llm_access():
            return self.fallback.critique(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                evidence_batches=evidence_batches,
                previous_artifact=previous_artifact,
                planner_result=planner_result,
                draft_sections=draft_sections,
                external_research=external_research,
            )
        try:
            bundle = build_research_context_bundle(market_identity, evidence_batches)
            raw = opus_llm(
                build_critic_system_prompt(),
                build_critic_user_prompt(
                    market_identity,
                    bundle,
                    planner_result.to_dict(),
                    draft_sections,
                    previous_artifact.to_dict() if previous_artifact else None,
                    _external_result_to_dict(external_research),
                ),
                expect_json=True,
                max_tokens=12000,
                usage_context={
                    "stage": "market_intel_critic",
                    "market_key": market_identity.market_key,
                },
            )
            return _normalize_critic_result(
                raw,
                evidence_batches,
                previous_artifact,
                planner_result,
                draft_sections,
                external_research,
            )
        except Exception:
            return self.fallback.critique(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                evidence_batches=evidence_batches,
                previous_artifact=previous_artifact,
                planner_result=planner_result,
                draft_sections=draft_sections,
                external_research=external_research,
            )


def _normalize_external_research_focus(raw_focus: Any, supporting_refs: list[str]) -> list[dict]:
    if not isinstance(raw_focus, list):
        return []
    focus_items: list[dict] = []
    for item in raw_focus:
        if not isinstance(item, dict):
            continue
        focus = _normalize_text(item.get("focus") or item.get("question"))
        reason = _normalize_text(item.get("reason") or item.get("next_step"))
        if not focus:
            continue
        focus_items.append(
            {
                "focus": focus,
                "priority": _normalize_text(item.get("priority")).lower() or "medium",
                "reason": reason or "Research this theme if external research runs.",
                "supporting_run_refs": [
                    ref
                    for ref in item.get("supporting_run_refs", []) or supporting_refs
                    if _normalize_text(ref)
                ],
                "evidence_refs": [
                    ref for ref in item.get("evidence_refs", []) if _normalize_text(ref)
                ],
            }
        )
    return focus_items


def _normalize_planner_result(raw: dict, evidence_batches: list[MarketEvidenceBatch]) -> PlannerResult:
    supporting_refs = _supporting_refs_from_batches(evidence_batches)
    active = sanitize_narrative_items("open_questions", raw.get("open_unknowns", []))
    backlog: list[dict] = []
    for item in raw.get("research_backlog", []):
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        payload.setdefault("supporting_run_refs", supporting_refs)
        try:
            backlog.append(ResearchOpportunity.from_dict(payload).to_dict())
        except Exception:
            continue
    hypotheses: list[dict] = []
    resolved: list[dict] = []
    for collection, target in (
        (raw.get("active_hypotheses", []), hypotheses),
        (raw.get("resolved_hypotheses", []), resolved),
    ):
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            payload.setdefault("supporting_run_refs", supporting_refs)
            payload.setdefault("first_seen_at", evidence_batches[0].generated_at if evidence_batches else _utc_now())
            payload.setdefault("last_seen_at", evidence_batches[-1].generated_at if evidence_batches else _utc_now())
            try:
                from market_intelligence.schema import MarketHypothesis

                target.append(MarketHypothesis.from_dict(payload).to_dict())
            except Exception:
                continue
    focus_items = _normalize_external_research_focus(
        raw.get("external_research_focus", raw.get("external_research_questions", [])),
        supporting_refs,
    )
    edge_case_focus_items = _normalize_external_research_focus(
        raw.get("edge_case_research_focus", []),
        supporting_refs,
    )
    return PlannerResult(
        planner_summary=_normalize_text(raw.get("planner_summary")),
        active_hypotheses=hypotheses,
        resolved_hypotheses=resolved,
        open_unknowns=active,
        research_backlog=backlog,
        update_sections=sorted(
            {
                _normalize_text(item)
                for item in raw.get("update_sections", [])
                if _normalize_text(item)
            }
        ),
        confidence_ceiling_by_section={
            str(key): min(1.0, max(0.0, _safe_float(value, 0.0)))
            for key, value in (raw.get("confidence_ceiling_by_section") or {}).items()
        },
        should_collect_external_research=bool(raw.get("should_collect_external_research")),
        external_research_focus=focus_items,
        should_collect_edge_case_research=bool(raw.get("should_collect_edge_case_research")),
        edge_case_research_reasoning=_normalize_text(raw.get("edge_case_research_reasoning")),
        edge_case_research_focus=edge_case_focus_items,
        edge_case_confidence_ceiling=(
            min(1.0, max(0.0, _safe_float(raw.get("edge_case_confidence_ceiling"), 0.0)))
            if raw.get("edge_case_confidence_ceiling") is not None
            else None
        ),
    )


def _normalize_draft_sections(raw: dict | None) -> dict:
    payload = dict(raw or {})
    for section in (
        "lane_intelligence",
        "talent_pool_intelligence",
        "noise_patterns",
        "employer_signal_intelligence",
        "brief_recommendations",
        "open_questions",
    ):
        payload[section] = sanitize_narrative_items(section, payload.get(section, []))
    market_thesis = payload.get("market_thesis")
    payload["market_thesis"] = _normalize_market_thesis_payload(market_thesis)
    return payload


def _normalize_market_thesis_payload(market_thesis: Any) -> dict:
    if not isinstance(market_thesis, dict):
        return {
            "summary": "",
            "supply_assessment": "unknown",
            "competition_assessment": "unknown",
            "external_context": [],
        }
    summary = _normalize_text(market_thesis.get("summary"))
    if market_thesis_summary_looks_like_review(summary):
        summary = ""
    return {
        "summary": summary,
        "supply_assessment": _normalize_text(market_thesis.get("supply_assessment"))
        or "unknown",
        "competition_assessment": _normalize_text(
            market_thesis.get("competition_assessment")
        )
        or "unknown",
        "external_context": sanitize_narrative_items(
            "market_thesis.external_context",
            market_thesis.get("external_context", []),
        ),
    }


def _market_thesis_has_substance(market_thesis: dict | None) -> bool:
    if not isinstance(market_thesis, dict):
        return False
    summary = _normalize_text(market_thesis.get("summary"))
    if summary and not market_thesis_summary_looks_like_review(summary):
        return True
    if sanitize_narrative_items(
        "market_thesis.external_context",
        market_thesis.get("external_context", []),
    ):
        return True
    if _normalize_text(market_thesis.get("supply_assessment")) not in {"", "unknown"}:
        return True
    if _normalize_text(market_thesis.get("competition_assessment")) not in {
        "",
        "unknown",
    }:
        return True
    return False


def _first_market_thesis_summary(*candidates: dict | None) -> str:
    for candidate in candidates:
        summary = _normalize_text((candidate or {}).get("summary"))
        if summary and not market_thesis_summary_looks_like_review(summary):
            return summary
    return ""


def _first_market_thesis_assessment(field: str, *candidates: dict | None) -> str:
    for candidate in candidates:
        value = _normalize_text((candidate or {}).get(field))
        if value and value != "unknown":
            return value
    return "unknown"


def _first_market_thesis_context(*candidates: dict | None) -> list[dict]:
    for candidate in candidates:
        items = sanitize_narrative_items(
            "market_thesis.external_context",
            (candidate or {}).get("external_context", []),
        )
        if items:
            return items
    return []


def _repair_market_thesis(
    primary: dict | None,
    *,
    draft: dict | None,
    fallback: dict | None,
    previous: dict | None,
) -> tuple[dict, bool]:
    repaired = _normalize_market_thesis_payload(primary)
    original = dict(repaired)
    if not _normalize_text(repaired.get("summary")):
        repaired["summary"] = _first_market_thesis_summary(draft, fallback, previous)
    if _normalize_text(repaired.get("supply_assessment")) in {"", "unknown"}:
        repaired["supply_assessment"] = _first_market_thesis_assessment(
            "supply_assessment",
            draft,
            fallback,
            previous,
        )
    if _normalize_text(repaired.get("competition_assessment")) in {"", "unknown"}:
        repaired["competition_assessment"] = _first_market_thesis_assessment(
            "competition_assessment",
            draft,
            fallback,
            previous,
        )
    if not sanitize_narrative_items(
        "market_thesis.external_context",
        repaired.get("external_context", []),
    ):
        repaired["external_context"] = _first_market_thesis_context(
            draft,
            fallback,
            previous,
        )
    return repaired, repaired != original


_ADJUDICATION_HOLDS = {"yes", "weaken", "drop"}


def _normalize_claim_adjudications(raw: Any) -> list[dict]:
    """Persist the critic's per-claim audit trail (the reason-first adjudication the
    critic prompt now requires). Keep well-formed entries — a non-empty ``claim`` and
    a ``holds`` verdict in {yes, weaken, drop}; drop malformed ones. These are
    free-form audit dicts, never routed through the narrative-item validators."""
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = _normalize_text(item.get("claim"))
        holds = str(item.get("holds") or "").strip().lower()
        if not claim or holds not in _ADJUDICATION_HOLDS:
            continue
        out.append({
            "claim": claim,
            "section": _normalize_text(item.get("section")),
            "evidence": _normalize_text(item.get("evidence")),
            "holds": holds,
            "why": _normalize_text(item.get("why")),
        })
    return out


def _normalize_critic_result(
    raw: dict,
    evidence_batches: list[MarketEvidenceBatch],
    previous_artifact: MarketIntelArtifact | None,
    planner_result: PlannerResult,
    draft_sections: dict[str, Any],
    external_research: Any | None,
) -> CriticResult:
    raw_keep_sections = raw.get("keep_sections", {})
    sanitized_keep_sections = _normalize_draft_sections(raw_keep_sections)
    sanitized_draft_sections = _normalize_draft_sections(draft_sections)
    fallback = HeuristicCriticBackend().critique(
        market_identity=previous_artifact.market_identity if previous_artifact else MarketIdentity(
            market_key="unknown",
            role_title="Unknown",
            role_level="",
            geography="",
            channels_seen=[],
            brief_ids_seen=[],
            brief_versions_seen=[],
        ),
        deterministic_summary={},
        evidence_batches=evidence_batches,
        previous_artifact=previous_artifact,
        planner_result=planner_result,
        draft_sections=sanitized_draft_sections,
        external_research=external_research,
    )
    repaired_sections: set[str] = set()
    # P3.7: critic drop decisions are honored. In an otherwise-valid critic
    # response (keep_sections is a non-empty object), an omitted or malformed
    # section means DROP — resurrecting the fallback/previous claims made
    # weak market-intel claims structurally immortal. Only a structurally
    # invalid response (missing/non-object keep_sections) falls back wholesale.
    structurally_invalid = (
        not isinstance(raw_keep_sections, dict) or not raw_keep_sections
    )
    if structurally_invalid:
        for section in (
            "lane_intelligence",
            "talent_pool_intelligence",
            "noise_patterns",
            "employer_signal_intelligence",
            "brief_recommendations",
            "open_questions",
        ):
            fallback_section = fallback.keep_sections.get(section, [])
            if fallback_section:
                sanitized_keep_sections[section] = fallback_section
                repaired_sections.add(section)

    fallback_market_thesis = fallback.keep_sections.get("market_thesis", {})
    draft_market_thesis = sanitized_draft_sections.get("market_thesis", {})
    previous_market_thesis = (
        previous_artifact.market_thesis if previous_artifact else {}
    )
    raw_market_thesis = raw_keep_sections.get("market_thesis")
    if "market_thesis" not in raw_keep_sections:
        repaired_market_thesis, _ = _repair_market_thesis(
            {},
            draft=draft_market_thesis,
            fallback=fallback_market_thesis,
            previous=previous_market_thesis,
        )
        if _market_thesis_has_substance(repaired_market_thesis):
            sanitized_keep_sections["market_thesis"] = repaired_market_thesis
            repaired_sections.add("market_thesis")
    elif not isinstance(raw_market_thesis, dict):
        repaired_market_thesis, _ = _repair_market_thesis(
            {},
            draft=draft_market_thesis,
            fallback=fallback_market_thesis,
            previous=previous_market_thesis,
        )
        if _market_thesis_has_substance(repaired_market_thesis):
            sanitized_keep_sections["market_thesis"] = repaired_market_thesis
            repaired_sections.add("market_thesis")
    elif raw_market_thesis:
        repaired_market_thesis, was_repaired = _repair_market_thesis(
            sanitized_keep_sections.get("market_thesis", {}),
            draft=draft_market_thesis,
            fallback=fallback_market_thesis,
            previous=previous_market_thesis,
        )
        if _market_thesis_has_substance(repaired_market_thesis):
            sanitized_keep_sections["market_thesis"] = repaired_market_thesis
        if was_repaired:
            repaired_sections.add("market_thesis")

    metadata = {}
    for section, value in (raw.get("section_generation_metadata") or {}).items():
        if not isinstance(value, dict):
            continue
        try:
            metadata[section] = SectionGenerationMetadata.from_dict(value).to_dict()
        except Exception:
            continue
    for section in repaired_sections:
        fallback_metadata = fallback.section_generation_metadata.get(section)
        if not fallback_metadata:
            continue
        repaired_metadata = dict(fallback_metadata)
        notes = list(repaired_metadata.get("notes", []))
        notes.append(
            "Recovered from invalid critic keep_sections output; fallback preservation applied."
        )
        repaired_metadata["notes"] = notes
        metadata[section] = repaired_metadata
    critique_summary = _normalize_text(raw.get("planner_summary")) or fallback.critique_summary
    if repaired_sections:
        critique_summary = (
            f"{critique_summary} Repaired invalid critic output for sections: "
            f"{', '.join(sorted(repaired_sections))}."
        ).strip()
    return CriticResult(
        keep_sections=sanitized_keep_sections,
        section_generation_metadata=metadata or fallback.section_generation_metadata,
        delta_since_last_run=dict(raw.get("delta_since_last_run") or fallback.delta_since_last_run),
        confidence_by_claim_area={
            str(key): min(1.0, max(0.0, _safe_float(value, 0.0)))
            for key, value in (raw.get("confidence_by_claim_area") or fallback.confidence_by_claim_area).items()
        },
        critique_summary=critique_summary,
        claim_adjudications=_normalize_claim_adjudications(raw.get("claim_adjudications")),
    )


def _external_result_to_dict(external_research: Any | None) -> dict:
    if external_research is None:
        return {}
    return {
        "sources": getattr(external_research, "sources", []),
        "inferred_research_questions": getattr(
            external_research,
            "inferred_research_questions",
            [],
        ),
        "market_findings": getattr(external_research, "market_findings", []),
        "sourcing_implications": getattr(
            external_research,
            "sourcing_implications",
            [],
        ),
        "market_thesis_context": getattr(external_research, "market_thesis_context", []),
        "open_questions": getattr(external_research, "open_questions", []),
        "edge_case_triggered": getattr(external_research, "edge_case_triggered", False),
        "edge_case_reasoning": getattr(external_research, "edge_case_reasoning", ""),
        "edge_case_focus": getattr(external_research, "edge_case_focus", []),
        "edge_case_inferred_research_questions": getattr(
            external_research,
            "edge_case_inferred_research_questions",
            [],
        ),
        "edge_case_submarkets": getattr(
            external_research,
            "edge_case_submarkets",
            [],
        ),
        "title_to_archetype_mapping": getattr(
            external_research,
            "title_to_archetype_mapping",
            [],
        ),
        "self_presentation_patterns": getattr(
            external_research,
            "self_presentation_patterns",
            [],
        ),
        "false_negative_hypotheses": getattr(
            external_research,
            "false_negative_hypotheses",
            [],
        ),
        "edge_case_sourcing_implications": getattr(
            external_research,
            "edge_case_sourcing_implications",
            [],
        ),
        "edge_case_open_questions": getattr(
            external_research,
            "edge_case_open_questions",
            [],
        ),
    }


def _quality_for_section(
    *,
    section: str,
    evidence_batches: list[MarketEvidenceBatch],
    kept: dict,
) -> str:
    run_count = len(evidence_batches)
    reconstructed = any(
        batch.context_quality in {"reconstructed_report", "raw_only"}
        for batch in evidence_batches
    )
    if section == "candidate_signal_summary":
        return "high" if run_count >= 1 else "low"
    if reconstructed and run_count <= 1:
        return "low"
    if run_count >= 2 and kept.get(section):
        return "high"
    if kept.get(section):
        return "medium"
    return "low"


def _metadata_notes(
    *,
    section: str,
    evidence_batches: list[MarketEvidenceBatch],
    kept: dict,
) -> list[str]:
    notes: list[str] = []
    if not kept.get(section):
        notes.append("No strong update for this section in the current pass.")
    if any(batch.context_quality == "reconstructed_report" for batch in evidence_batches):
        notes.append("Some evidence was reconstructed from raw artifacts.")
    if any(batch.context_quality == "raw_only" for batch in evidence_batches):
        notes.append("Some evidence arrived without an original report debrief.")
    return notes[:3]


def _supporting_refs_for_section(
    section: str,
    kept: dict,
    evidence_batches: list[MarketEvidenceBatch],
) -> list[str]:
    current = kept.get(section)
    refs: list[str] = []
    if isinstance(current, list):
        for item in current:
            if not isinstance(item, dict):
                continue
            refs.extend(item.get("supporting_run_refs", []))
    elif isinstance(current, dict):
        for item in current.get("external_context", []):
            if not isinstance(item, dict):
                continue
            refs.extend(item.get("supporting_run_refs", []))
    normalized = [ref for ref in refs if _normalize_text(ref)]
    return sorted(set(normalized or _supporting_refs_from_batches(evidence_batches)))


def _evidence_refs_for_section(section: str, kept: dict) -> list[str]:
    current = kept.get(section)
    refs: list[str] = []
    if isinstance(current, list):
        for item in current:
            if not isinstance(item, dict):
                continue
            refs.extend(item.get("evidence_refs", []))
    elif isinstance(current, dict):
        for item in current.get("external_context", []):
            if not isinstance(item, dict):
                continue
            refs.extend(item.get("evidence_refs", []))
    return sorted({ref for ref in refs if _normalize_text(ref)})


def _build_delta(previous: dict, kept: dict, planner_result: PlannerResult) -> dict[str, list[str]]:
    previous_lane_keys = {
        item.get("lane_key")
        for item in previous.get("lane_intelligence", [])
        if isinstance(item, dict)
    }
    current_lane_keys = {
        item.get("lane_key")
        for item in kept.get("lane_intelligence", [])
        if isinstance(item, dict)
    }
    became_more_true = sorted(
        {
            f"Lane strengthened: {key}"
            for key in current_lane_keys - previous_lane_keys
            if key
        }
    )[:5]
    became_less_true = sorted(
        {
            f"Lane cooled: {key}"
            for key in previous_lane_keys - current_lane_keys
            if key
        }
    )[:5]
    still_uncertain = [
        item.get("question")
        for item in planner_result.open_unknowns[:5]
        if _normalize_text(item.get("question"))
    ]
    next_run_changes = [
        item.get("proposal")
        for item in kept.get("brief_recommendations", [])[:5]
        if _normalize_text(item.get("proposal"))
    ]
    return {
        "became_more_true": became_more_true,
        "became_less_true": became_less_true,
        "still_uncertain": still_uncertain,
        "next_run_changes": next_run_changes,
    }


def _confidence_by_claim_area(metadata: dict[str, dict]) -> dict[str, float]:
    return {
        section: {
            "high": 0.82,
            "medium": 0.68,
            "low": 0.5,
        }.get(_normalize_text(value.get("quality_level")).lower(), 0.5)
        for section, value in metadata.items()
    }
