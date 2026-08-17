"""Integration-style tests for market intelligence artifacts and updates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from market_intelligence import (
    ExternalResearchResult,
    HeuristicPlannerBackend,
    MarketIntelArtifact,
    MarketIntelAgentState,
    MarketEvidenceBatch,
    MarketIdentity,
    build_research_context_bundle,
    finalize_run_snapshot,
    import_legacy_run_snapshot,
    live_market_intel_dir,
    record_block_checkpoint_and_get_context,
    record_page_checkpoint_and_get_context,
    render_market_intel_markdown,
    resolve_market_intel_agent_state_path,
    resolve_market_intel_artifact_path,
    resolve_market_intel_research_log_path,
    update_market_intel,
)
import market_intelligence.agent_backends as agent_backends_mod
import market_intelligence.engine as engine_mod
import market_intelligence.research_agent as research_agent_mod
from market_intelligence.research_prompts import build_research_user_prompt
from market_intelligence.research_prompts import (
    build_perplexity_edge_case_research_instructions,
    build_perplexity_edge_case_research_user_prompt,
    build_perplexity_research_instructions,
    build_perplexity_research_user_prompt,
)
from linkedin.orchestrator import Pipeline
from linkedin.strategy import adapt_after_block
from shared.brief_loader import load_brief
from shared.llm_usage import record_llm_usage
from shared.output_paths import classify_output_location, resolve_linkedin_state_dir
from shared.runtime_state import RuntimeStateStore
from shared.schemas import BlockReport, SearchString
from shared.storage import read_json, read_jsonl, write_json


ROOT = Path(__file__).parent.parent
SOURCE_BRIEF = ROOT / "config" / "brief-head-ai-lab-nyc-v2.json"
FDE_BRIEF = ROOT / "config" / "Forward-Deployed-Engineer-NYC" / "brief-forward-deployed-engineer-us-v1.3.json"


def _require_source_brief_path() -> Path:
    if not SOURCE_BRIEF.is_file():
        pytest.skip(f"Missing optional brief fixture: {SOURCE_BRIEF}")
    return SOURCE_BRIEF


def _require_fde_brief_path_v13() -> Path:
    if not FDE_BRIEF.is_file():
        pytest.skip(f"Missing optional brief fixture: {FDE_BRIEF}")
    return FDE_BRIEF


def _report_dict(*, version: str, generated_at: str, saved: int = 12) -> dict:
    return {
        "schema_version": 1,
        "run_metadata": {
            "role_title": "Head of Applied AI Lab",
            "brief_name": "head-ai-lab",
            "brief_version": version,
            "linkedin_project": "Head of Applied AI Lab",
            "linkedin_project_id": "3000000006",
            "generated_at": generated_at,
            "overall_summary": "Structured debrief input for market intelligence.",
        },
        "metrics_summary": {
            "strings_executed": 8,
            "strings_skipped": 2,
            "total_results": 2200,
            "total_pages_reviewed": 19,
            "candidates_evaluated": 180,
            "facial_yes": 40,
            "facial_no": 140,
            "saved": saved,
            "rejected": 28,
            "overall_save_rate": round(saved / 180, 4),
            "facial_yes_rate": round(40 / 180, 4),
        },
        "string_performance": [
            {
                "string_id": 2,
                "name": "Research copilot lane",
                "status": "done",
                "result_count": 526,
                "pages_reviewed": 4,
                "saves": 8,
                "save_rate": 0.08,
                "saved_candidates": ["Mithun Azhagappan"],
                "notes": "Strong lane",
                "facial_yes_count": 10,
                "facial_no_count": 12,
                "candidates_count": 22,
                "duplicates_count": 1,
                "family_key": "research_copilot_asset_mgmt",
                "novelty_bucket": "edge_case",
                "domain_lane": "asset_management",
            }
        ],
        "winning_lanes": [
            {
                "lane": "Research copilot / asset management",
                "string_ids": [2],
                "candidate_examples": ["Mithun Azhagappan"],
                "evidence": "Highest absolute save count in the run.",
                "why_it_worked": "Specific workflow language filtered for real builders.",
                "recommended_action": "Promote this lane earlier.",
            }
        ],
        "underperforming_lanes": [
            {
                "lane": "Surveillance",
                "string_ids": [8],
                "issue": "Mostly traditional compliance-tech noise.",
                "evidence": "Zero saves after two pages.",
                "recommended_action": "Only retry with an explicit GenAI AND-gate.",
            }
        ],
        "coverage_gaps": [
            {
                "gap": "Payments",
                "why_it_matters": "The run never explicitly targeted payments.",
                "suggested_search_strategy": "Add payment-orchestration strings.",
            }
        ],
        "noise_patterns": [
            {
                "pattern": "Product-heavy AI leadership",
                "evidence": "Several product/strategy AI officers failed the builder bar.",
                "mitigation": "Strengthen builder-authorship and systems language.",
            }
        ],
        "saved_candidate_patterns": {
            "standout_candidates": [
                {"name": "Mithun Azhagappan", "why": "Goldman AI platform architect."}
            ],
            "common_employers": [
                {
                    "employer": "JPMorgan",
                    "count": 3,
                    "note": "Strong bank GenAI-convert population.",
                }
            ],
            "common_titles": [
                {"title_family": "Executive Director", "count": 2, "note": "Right scope band."}
            ],
            "archetype_distribution": [
                {
                    "archetype": "BFSI-native GenAI converts",
                    "count": 6,
                    "note": "Dominant save archetype.",
                }
            ],
            "seniority_notes": [
                "VP bank builders were often technically strong but below the final scope bar."
            ],
        },
        "adaptation_assessment": {
            "summary": "Tight workflow strings outperformed broad archetype-first strings.",
            "effective_refinements": [
                "Research-copilot phrasing materially improved signal."
            ],
            "questionable_or_skipped": ["Payments remained under-covered."],
            "operational_notes": ["Keep early strings narrow and workflow-specific."],
        },
        "recommendations": {
            "try_next": ["Payments and transaction-banking builders"],
            "avoid_next": ["Ungated surveillance strings"],
            "prioritize_pipeline": ["Mithun Azhagappan"],
        },
        "brief_iteration_hints": {
            "instructions": ["Cover payments in the first block."],
            "search_priorities": ["Payments / transaction-banking builders"],
            "additional_search_terms": [
                "payment orchestration",
                "transaction banking",
            ],
            "intake_notes": "The latest run exposed a payments gap.",
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_linkedin_inputs(
    output_dir: Path,
    brief_path: Path,
    *,
    version: str,
    generated_at: str,
    saved: int = 12,
    include_report_input: bool = True,
    include_raw_artifacts: bool = True,
    include_report: bool = True,
) -> None:
    brief = load_brief(str(brief_path))
    report = _report_dict(version=version, generated_at=generated_at, saved=saved)
    if include_report:
        write_json(output_dir / "run-report.json", report)
    if include_report_input:
        write_json(
            output_dir / "run-report-input.json",
            {
                "schema_version": 1,
                "run_metadata": report["run_metadata"],
                "metrics_summary": report["metrics_summary"],
                "string_performance": report["string_performance"],
                "saved_candidate_summaries": [
                    {
                        "candidate_name": "Mithun Azhagappan",
                        "decision": "SAVE",
                        "path": "DIRECT:Research Copilot",
                        "confidence": 0.82,
                        "rationale": "Goldman AI platform architect with strong workflow evidence.",
                    }
                ],
                "rejected_candidate_summaries": [
                    {
                        "candidate_name": "Deepinder Gulati",
                        "decision": "REJECT",
                        "path": "NONE",
                        "confidence": 0.41,
                        "rationale": "Product-heavy AI leadership without hands-on builder evidence.",
                    }
                ],
                "bias_monitor_summary": "No critical bias issues. Review save clustering by large-bank brands.",
                "search_memory_summary": {
                    "project_id": brief.linkedin_project_id or brief_path.stem,
                    "overall": {
                        "families_tracked": 1,
                        "strings_seen": 1,
                        "save_rate": round(saved / 180, 4),
                        "duplicate_rate": 0.04,
                        "novelty_mix": {
                            "edge_case_saves": saved,
                            "canonical_saves": 0,
                        },
                    },
                    "families": [
                        {
                            "family_key": "research_copilot_asset_mgmt",
                            "novelty_bucket": "edge_case",
                            "domain_lane": "asset_management",
                            "status": "active",
                            "status_reason": "",
                            "save_rate": round(saved / 180, 4),
                            "duplicate_rate": 0.04,
                            "saves": saved,
                            "strings_seen": 1,
                            "dominant_anchors": [
                                "research copilot",
                                "asset management",
                                "workflow",
                            ],
                        }
                    ],
                },
            },
        )
    _write_jsonl(
        output_dir / "final_judgments.jsonl",
        [
            {
                "stage": "full",
                "decision": "SAVE",
                "path": "DIRECT:Research Copilot",
                "profile_url": "https://example.com/mithun",
                "confidence": 0.82,
                "rationale": "Built research copilot workflows and evaluation frameworks in production.",
                "candidate_name": "Mithun Azhagappan",
            },
            {
                "stage": "full",
                "decision": "REJECT",
                "path": "NONE",
                "profile_url": "https://example.com/deepinder",
                "confidence": 0.41,
                "rationale": "Product-heavy AI leadership without hands-on builder evidence.",
                "candidate_name": "Deepinder Gulati",
            },
        ],
    )
    if include_raw_artifacts:
        _write_jsonl(
            output_dir / "profile_summaries.jsonl",
            [
                {
                    "name": "Mithun Azhagappan",
                    "headline": "Executive Director, AI Platform",
                    "profile_url": "https://example.com/mithun",
                    "skills_snippet": "GenAI, platform, evaluation",
                    "education": [],
                    "experiences": [
                        {
                            "company": "JPMorgan",
                            "title": "Executive Director, AI Platform",
                            "location": "New York, NY",
                            "summary_bullets": [
                                "Built research copilots and evaluation systems for banking workflows.",
                                "Led production AI platform delivery with workflow-specific retrieval.",
                            ],
                        }
                    ],
                },
                {
                    "name": "Deepinder Gulati",
                    "headline": "Chief Product Officer, AI",
                    "profile_url": "https://example.com/deepinder",
                    "skills_snippet": "product, AI strategy",
                    "education": [],
                    "experiences": [
                        {
                            "company": "Strategy Bank",
                            "title": "Chief Product Officer, AI",
                            "location": "New York, NY",
                            "summary_bullets": [
                                "Led AI strategy and product planning without direct hands-on systems ownership."
                            ],
                        }
                    ],
                },
            ],
        )
        _write_jsonl(
            output_dir / "snippets.jsonl",
            [
                {
                    "name": "Mithun Azhagappan",
                    "headline": "Executive Director, AI Platform",
                    "current_company": "JPMorgan",
                    "current_title": "Executive Director, AI Platform",
                    "location": "New York, NY",
                    "profile_url": "https://example.com/mithun",
                    "source_string_id": 2,
                    "source_string_name": "Research copilot lane",
                    "page": 1,
                    "result_rank": 1,
                    "experience_entries": [
                        "Executive Director, AI Platform at JPMorgan",
                    ],
                },
                {
                    "name": "Deepinder Gulati",
                    "headline": "Chief Product Officer, AI",
                    "current_company": "Strategy Bank",
                    "current_title": "Chief Product Officer, AI",
                    "location": "New York, NY",
                    "profile_url": "https://example.com/deepinder",
                    "source_string_id": 8,
                    "source_string_name": "Surveillance",
                    "page": 1,
                    "result_rank": 2,
                    "experience_entries": [
                        "Chief Product Officer at Strategy Bank",
                    ],
                },
            ],
        )
        _write_jsonl(
            output_dir / "run_log.jsonl",
            [
                {
                    "timestamp": generated_at,
                    "event": "block_adaptation",
                    "string_id": 2,
                    "page": 1,
                    "phase": "paginate",
                    "action": "continue",
                    "rationale": "Research copilot workflow phrasing is surfacing strong builders.",
                    "refinement_depth": 0,
                },
                {
                    "timestamp": generated_at,
                    "event": "forced_narrow",
                    "string_id": 8,
                    "page": 2,
                    "phase": "paginate",
                    "action": "narrow",
                    "rationale": "Surveillance lane pulled compliance-tech noise without builder signal.",
                    "refinement_depth": 1,
                },
                {
                    "timestamp": generated_at,
                    "event": "bias_alert",
                    "severity": "flag",
                    "alert_type": "save_cluster",
                    "message": "Check large-bank clustering among saves.",
                    "string_id": "2",
                },
            ],
        )
    write_json(
        output_dir
        / f"search_memory-{brief.linkedin_project_id or brief_path.stem}.json",
        {
            "version": 1,
            "project_id": brief.linkedin_project_id or brief_path.stem,
            "updated_at": generated_at,
            "overall": {
                "strings_seen": 1,
                "pages_reviewed": 4,
                "candidates_seen": 22,
                "duplicates": 1,
                "saves": saved,
                "edge_case_saves": saved,
                "canonical_saves": 0,
            },
            "families": {
                "research_copilot_asset_mgmt": {
                    "family_key": "research_copilot_asset_mgmt",
                    "novelty_bucket": "edge_case",
                    "domain_lane": "asset_management",
                    "strings_seen": 1,
                    "pages_reviewed": 4,
                    "candidates_seen": 22,
                    "duplicates": 1,
                    "saves": saved,
                    "facial_yes": 10,
                    "facial_no": 12,
                    "dominant_anchors": [
                        "research copilot",
                        "asset management",
                        "workflow",
                    ],
                }
            },
        },
    )


def _write_github_inputs(output_dir: Path, brief_path: Path, *, generated_at: str) -> None:
    store = RuntimeStateStore(output_dir / "runtime_state.sqlite3")
    brief = load_brief(str(brief_path))
    with store.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO runs(source, brief_id, output_dir, mode, status, started_at, ended_at, resume_state_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "github",
                brief.id,
                str(output_dir),
                "autonomous",
                "completed",
                generated_at,
                generated_at,
                "{}",
            ),
        )
        run_id = int(cursor.lastrowid)
        conn.execute(
            """
            INSERT INTO work_units(
                run_id, source, brief_id, kind, source_unit_id, display_name, status,
                payload_json, checkpoint_json, metrics_json, family_key, novelty_bucket,
                domain_lane, candidates_discovered, facial_yes_count, facial_no_count,
                saves_count, rejected_count, started_at, ended_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                "github",
                brief.id,
                "github_query",
                "trl_contributors",
                "TRL contributors",
                "done",
                json.dumps({"boolean": "trl contributors"}),
                json.dumps({"pages_reviewed": 1, "duplicates_count": 0}),
                json.dumps({"candidates_count": 7}),
                "trl_contributors",
                "edge_case",
                "general",
                7,
                3,
                4,
                1,
                6,
                generated_at,
                generated_at,
            ),
        )


def _import_run_dir(
    *,
    brief_path: Path,
    legacy_output_dir: Path,
    source: str = "linkedin",
    run_id: int | None = None,
    reconstruct_report_analysis: bool = False,
) -> Path:
    return import_legacy_run_snapshot(
        brief_path=brief_path,
        legacy_output_dir=legacy_output_dir,
        source=source,
        run_id=run_id,
        reconstruct_report_analysis=reconstruct_report_analysis,
    )


def _insert_linkedin_attempt_payloads(
    *,
    store: RuntimeStateStore,
    run_id: int,
    brief_id: str,
    identity_key: str,
    candidate_name: str,
    decision: str,
    started_at: str,
    ended_at: str,
) -> None:
    with store.connect() as conn:
        candidate_id = int(
            conn.execute(
                """
                INSERT INTO candidates(
                    source, brief_id, identity_key, display_name, profile_url,
                    current_lifecycle_state, terminal_decision, terminal_payload_json,
                    first_seen_at, last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "linkedin",
                    brief_id,
                    identity_key,
                    candidate_name,
                    f"/talent/profile/{identity_key}",
                    "full_terminal",
                    decision,
                    json.dumps({"confidence": 0.81, "source_string_id": 2, "timestamp": ended_at}),
                    started_at,
                    ended_at,
                ),
            ).lastrowid
        )
        conn.execute(
            """
            INSERT INTO candidate_attempts(
                run_id, candidate_id, work_unit_id, stage, attempt_number,
                status, payload_json, source_cursor_json, started_at, ended_at
            )
            VALUES (?, ?, NULL, 'snippet', 1, 'completed', ?, '{}', ?, ?)
            """,
            (
                run_id,
                candidate_id,
                json.dumps(
                    {
                        "snippet": {
                            "candidate_name": candidate_name,
                            "profile_url": f"/talent/profile/{identity_key}",
                            "source_string_id": 2,
                            "source_string_name": "Run lane",
                            "page": 1,
                        }
                    }
                ),
                started_at,
                ended_at,
            ),
        )
        conn.execute(
            """
            INSERT INTO candidate_attempts(
                run_id, candidate_id, work_unit_id, stage, attempt_number,
                status, payload_json, source_cursor_json, started_at, ended_at
            )
            VALUES (?, ?, NULL, 'full', 1, 'completed', ?, '{}', ?, ?)
            """,
            (
                run_id,
                candidate_id,
                json.dumps(
                    {
                        "profile_summary": {
                            "candidate_name": candidate_name,
                            "skills_snippet": "agents, deployment",
                        },
                        "full_decision": {
                            "candidate_name": candidate_name,
                            "decision": decision,
                            "confidence": 0.81,
                            "path": "DIRECT:Forward Deployed Engineering",
                            "rationale": f"{candidate_name} matches the run-specific payload.",
                            "stage": "full",
                        },
                    }
                ),
                started_at,
                ended_at,
            ),
        )


class _StubResearchBackend:
    def collect(self, **_: object) -> ExternalResearchResult:
        return ExternalResearchResult(
            sources=[
                {
                    "source_id": "web:https://example.com/jobs",
                    "kind": "web_search",
                    "title": "Example Job Posting",
                    "url": "https://example.com/jobs",
                    "retrieved_at": "2026-04-08T14:00:00+00:00",
                    "used_for": ["market_thesis"],
                }
            ],
            inferred_research_questions=[
                {
                    "question": "Are employer title variants hiding relevant candidates?",
                    "priority": "high",
                    "why_it_matters": "The run may be missing a hidden pool.",
                    "sourcing_trigger": "Coverage gaps suggested title blind spots.",
                    "status": "answered",
                    "supporting_run_refs": ["linkedin:output"],
                    "evidence_refs": ["web:https://example.com/jobs"],
                }
            ],
            market_findings=[
                {
                    "finding_key": "title-variant-solutions-engineer",
                    "kind": "title_variant",
                    "label": "Solutions Engineer",
                    "summary": "Solutions Engineer appears to be a common title variant for this role family.",
                    "why_it_matters": "The sourcing run may be too narrow on title coverage.",
                    "confidence": 0.8,
                    "supporting_run_refs": ["linkedin:output"],
                    "evidence_refs": ["web:https://example.com/jobs"],
                },
                {
                    "finding_key": "employer-cluster-example",
                    "kind": "employer_cluster",
                    "label": "Example Employer Cluster",
                    "summary": "Example employer is hiring.",
                    "why_it_matters": "There is credible employer demand outside the initial observed pool.",
                    "confidence": 0.8,
                    "supporting_run_refs": ["linkedin:output"],
                    "evidence_refs": ["web:https://example.com/jobs"],
                },
            ],
            sourcing_implications=[
                {
                    "implication_id": "add-title-variants",
                    "category": "add_title_family",
                    "priority": "high",
                    "recommendation": "Add Solutions Engineer and adjacent deployment titles to the next run.",
                    "rationale": "External research suggests the first run missed meaningful title variants.",
                    "brief_target_field": "additional_search_terms",
                    "suggested_values": ["Solutions Engineer", "Deployment Engineer"],
                    "expected_effect": "Broader title coverage should reduce false negatives.",
                    "supporting_run_refs": ["linkedin:output"],
                    "evidence_refs": ["web:https://example.com/jobs"],
                }
            ],
            market_thesis_context=[
                {
                    "claim": "Example employer is hiring.",
                    "evidence_refs": ["web:https://example.com/jobs"],
                    "confidence": 0.8,
                }
            ],
            open_questions=[
                {
                    "question": "Which adjacent title family should be prioritized first?",
                    "priority": "medium",
                    "next_step": "Test at least one adjacent title-family string next cycle.",
                    "supporting_run_refs": ["linkedin:output"],
                    "evidence_refs": ["web:https://example.com/jobs"],
                }
            ],
        )


class _CountingResearchBackend:
    def __init__(self) -> None:
        self.calls = 0

    def collect(self, **_: object) -> ExternalResearchResult:
        self.calls += 1
        return ExternalResearchResult()


class _CustomQuestionSynthesisBackend:
    def synthesize(self, **_: object) -> dict:
        return {
            "lane_intelligence": [],
            "talent_pool_intelligence": [],
            "noise_patterns": [],
            "employer_signal_intelligence": [],
            "market_thesis": {
                "summary": "Custom synthesis ran.",
                "supply_assessment": "moderate",
                "competition_assessment": "high",
                "external_context": [],
            },
            "brief_recommendations": [],
            "open_questions": [
                {
                    "question": "Should we target adjacent fintech teams?",
                    "priority": "medium",
                    "next_step": "Run one fintech-focused lane.",
                    "supporting_run_refs": ["linkedin:output"],
                }
            ],
        }


class _PlannerCapturingResearchBackend:
    def __init__(self) -> None:
        self.focus: list[dict] = []
        self.calls = 0

    def collect(self, **kwargs: object) -> ExternalResearchResult:
        self.calls += 1
        self.focus = list(kwargs.get("research_focus") or [])
        return ExternalResearchResult(
            sources=[],
            market_thesis_context=[],
            open_questions=[],
        )


class _AlwaysResearchPlannerBackend:
    def plan(self, **_: object):
        from market_intelligence.agent_backends import PlannerResult

        return PlannerResult(
            planner_summary="Test planner",
            active_hypotheses=[
                {
                    "hypothesis_id": "hyp-1",
                    "statement": "Repeated lane signal exists",
                    "status": "active",
                    "confidence": 0.7,
                    "rationale": "Repeated signal",
                    "section_targets": ["lane_intelligence"],
                    "first_seen_at": "2026-04-08T12:00:00+00:00",
                    "last_seen_at": "2026-04-08T12:00:00+00:00",
                    "supporting_run_refs": ["linkedin:output"],
                }
            ],
            open_unknowns=[
                {
                    "question": "What employers are actively hiring?",
                    "priority": "high",
                    "next_step": "Run external research",
                    "supporting_run_refs": ["linkedin:output"],
                }
            ],
            research_backlog=[
                {
                    "opportunity_id": "opp-1",
                    "question": "Which employers are actively hiring?",
                    "priority": "high",
                    "status": "queued",
                    "reason": "Important",
                    "supporting_run_refs": ["linkedin:output"],
                }
            ],
            update_sections=["lane_intelligence", "market_thesis"],
            confidence_ceiling_by_section={"market_thesis": 0.6},
            should_collect_external_research=True,
            external_research_focus=[
                {
                    "focus": "Employer demand and title-variant blind spots",
                    "priority": "high",
                    "reason": "Validate the strongest external uncertainty.",
                    "supporting_run_refs": ["linkedin:output"],
                }
            ],
        )


class _AlwaysEdgeCasePlannerBackend(_AlwaysResearchPlannerBackend):
    def plan(self, **_: object):
        result = super().plan()
        result.should_collect_edge_case_research = True
        result.edge_case_research_reasoning = (
            "Coverage gaps, edge-case novelty, and sparse profile evidence suggest hidden-pool risk."
        )
        result.edge_case_research_focus = [
            {
                "focus": "Which hidden title families may self-label differently on LinkedIn?",
                "priority": "high",
                "reason": "Recover likely false negatives caused by title fragmentation.",
                "supporting_run_refs": ["linkedin:output"],
            }
        ]
        result.edge_case_confidence_ceiling = 0.58
        return result


class _StubEdgeCaseResearchBackend:
    def collect(self, **kwargs: object) -> ExternalResearchResult:
        research_mode = kwargs.get("research_mode", "general")
        if research_mode == "edge_case":
            return ExternalResearchResult(
                sources=[
                    {
                        "source_id": "web:https://example.com/edge-case",
                        "kind": "web_search",
                        "title": "Edge Case Role Map",
                        "url": "https://example.com/edge-case",
                        "retrieved_at": "2026-04-10T15:00:00+00:00",
                        "used_for": ["edge_case_research"],
                    }
                ],
                edge_case_inferred_research_questions=[
                    {
                        "question": "Are relevant FDE-like candidates self-labeling under solutions-oriented titles?",
                        "priority": "high",
                        "why_it_matters": "The first-pass search may be too title-narrow.",
                        "sourcing_trigger": "Coverage gaps plus edge-case novelty signal suggest title fragmentation.",
                        "status": "answered",
                        "supporting_run_refs": ["linkedin:output"],
                        "evidence_refs": ["web:https://example.com/edge-case"],
                    }
                ],
                edge_case_submarkets=[
                    {
                        "submarket_key": "solutions_delivery_builders",
                        "label": "Solutions / deployment builders",
                        "summary": "A hidden submarket of customer-embedded builders appears relevant to the role family.",
                        "why_it_is_easy_to_miss": "These candidates often avoid explicit Forward Deployed Engineer titles.",
                        "confidence": 0.67,
                        "supporting_run_refs": ["linkedin:output"],
                        "evidence_refs": ["web:https://example.com/edge-case"],
                    }
                ],
                title_to_archetype_mapping=[
                    {
                        "mapping_key": "solutions_engineer-builder",
                        "title_family": "Solutions Engineer",
                        "likely_archetype": "Customer-embedded platform builder",
                        "caveats": "Some profiles skew pre-sales; validate implementation ownership.",
                        "supporting_run_refs": ["linkedin:output"],
                        "evidence_refs": ["web:https://example.com/edge-case"],
                    }
                ],
                self_presentation_patterns=[
                    {
                        "pattern_key": "customer-facing-builder-language",
                        "label": "Customer-facing builder language",
                        "pattern": "Candidates describe themselves as deployment, solutions, or customer engineers rather than FDEs.",
                        "why_it_causes_false_negatives": "Boolean strings anchored on FDE titles may miss them.",
                        "supporting_run_refs": ["linkedin:output"],
                        "evidence_refs": ["web:https://example.com/edge-case"],
                    }
                ],
                false_negative_hypotheses=[
                    {
                        "hypothesis_key": "missed-solutions-builders",
                        "statement": "The sourcing run likely missed a hidden pool of solutions/deployment builders.",
                        "why_it_matters": "This could materially understate addressable candidate supply.",
                        "validation_task": "Add one title-family lane covering solutions and deployment builders.",
                        "confidence": 0.62,
                        "supporting_run_refs": ["linkedin:output"],
                        "evidence_refs": ["web:https://example.com/edge-case"],
                    }
                ],
                edge_case_sourcing_implications=[
                    {
                        "implication_id": "probe-hidden-title-family",
                        "category": "add_title_family",
                        "priority": "high",
                        "recommendation": "Add Solutions Engineer and Deployment Engineer title families.",
                        "rationale": "Edge-case research suggests relevant builders self-label under customer-facing implementation titles.",
                        "brief_target_field": "additional_search_terms",
                        "suggested_values": ["Solutions Engineer", "Deployment Engineer"],
                        "expected_effect": "Should recover hidden pools missed by title-narrow sourcing.",
                        "supporting_run_refs": ["linkedin:output"],
                        "evidence_refs": ["web:https://example.com/edge-case"],
                    }
                ],
                edge_case_open_questions=[
                    {
                        "question": "How often do these hidden pools meet the IC5/IC6 bar?",
                        "priority": "high",
                        "next_step": "Validate seniority and ownership depth in the next run.",
                        "supporting_run_refs": ["linkedin:output"],
                        "evidence_refs": ["web:https://example.com/edge-case"],
                    }
                ],
            )
        return ExternalResearchResult()


def _seed_legacy_invalid_open_question(artifact_path: Path, *, keep_valid: bool = True) -> None:
    artifact = read_json(artifact_path)
    open_questions = []
    if keep_valid:
        open_questions.extend(copy.deepcopy(artifact.get("open_questions", [])))
    open_questions.append(
        {
            "question": "Unsourced legacy question",
            "priority": "low",
            "next_step": "Leave it hanging.",
        }
    )
    artifact["open_questions"] = open_questions
    write_json(artifact_path, artifact)


def _make_research_batch(
    *,
    run_ref: str,
    generated_at: str,
    saved: int,
    candidates: int,
) -> MarketEvidenceBatch:
    packet = {
        "context_metadata": {
            "run_ref": run_ref,
            "source": "linkedin",
            "output_dir": f"/tmp/{run_ref}",
            "generated_at": generated_at,
            "brief_version": "2.1",
            "context_quality": "original_report",
            "analysis_provenance": "original_report",
            "artifact_paths_used": {},
            "research_input_path": f"/tmp/{run_ref}/market-intel-research-input.json",
        },
        "deterministic_snapshot": {
            "run_metadata": {"role_title": "Head of Applied AI Lab"},
            "metrics_summary": {
                "saved": saved,
                "candidates_evaluated": candidates,
            },
            "string_performance": [],
            "search_memory_summary": {"overall": {}, "families": []},
            "bias_monitor_summary": "",
        },
        "report_analysis": {
            "winning_lanes": [
                {
                    "lane": f"lane-{run_ref}",
                    "string_ids": [1],
                    "candidate_examples": ["Example"],
                    "evidence": f"{saved} saves",
                    "why_it_worked": "Good lane",
                    "recommended_action": "Keep testing",
                }
            ],
            "underperforming_lanes": [],
            "coverage_gaps": [],
            "noise_patterns": [],
            "saved_candidate_patterns": {
                "common_employers": [{"employer": "JPMorgan", "count": saved}],
                "archetype_distribution": [{"archetype": "Bank builders", "count": saved}],
            },
            "adaptation_assessment": {},
            "recommendations": {},
            "brief_iteration_hints": {},
        },
        "candidate_evidence": {"saved_examples": [], "rejected_examples": []},
        "adaptation_timeline": [],
    }
    return MarketEvidenceBatch(
        run_ref=run_ref,
        source="linkedin",
        output_dir=f"/tmp/{run_ref}",
        brief_version="2.1",
        generated_at=generated_at,
        metrics_summary={
            "run_count": 1,
            "saved": saved,
            "rejected": max(0, candidates - saved),
            "candidate_volume": candidates,
            "save_rate": round(saved / max(candidates, 1), 4),
        },
        research_context=packet,
        research_input_path=packet["context_metadata"]["research_input_path"],
        context_quality="original_report",
        analysis_provenance="original_report",
    )


def test_market_intel_creates_valid_artifact_from_first_run(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)

    artifact = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
    )
    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir)
    loaded = MarketIntelArtifact.from_dict(read_json(artifact_path))

    assert artifact.aggregate_metrics["saved_count"] == 12
    assert artifact.aggregate_metrics["run_count"] == 1
    assert artifact.channel_summaries["linkedin"]["top_lane_keys"]
    assert all(
        question.get("supporting_run_refs") or question.get("evidence_refs")
        for question in loaded.open_questions
    )
    assert (artifact_path.parent / "market-intel.md").exists()
    history_entries = list((artifact_path.parent / "history").iterdir())
    assert history_entries
    history_names = {path.name for path in history_entries}
    assert any(name.endswith(".json") and "__" not in name for name in history_names)
    assert any(name.endswith("__agent-state.json") for name in history_names)
    assert any(name.endswith("__market-intel.md") for name in history_names)
    assert any(name.endswith("__market-intel-technical.md") for name in history_names)
    research_input = read_json(run_dir / "market-intel-research-input.json")
    assert research_input["context_metadata"]["context_quality"] == "original_report"
    assert len(research_input["report_analysis"]["winning_lanes"]) <= 5
    assert len(research_input["candidate_evidence"]["saved_examples"]) <= 12
    assert len(research_input["adaptation_timeline"]) <= 25


def test_summarize_adaptation_event_handles_new_payload_shape() -> None:
    # The new AdaptationDecision payload puts rationale at the top level,
    # not inside report.summary or event.message. Without a fallback to
    # event.rationale, every non-LinkedIn adaptation event renders with
    # an empty summary line in market-intel reports.
    from market_intelligence.research_context import _summarize_adaptation_event

    event = {
        "timestamp": "2026-05-12T01:00:00+00:00",
        "event": "adaptation_decision",
        "source": "researcher",
        "action": "broaden",
        "lane": "academic_search",
        "rationale": "Sparse venue scout; broadened topic concepts.",
        "metrics": {"saves": 0, "candidates_discovered": 0},
        "source_payload": {"batch_report": {}},
    }
    summary = _summarize_adaptation_event(event)
    assert summary["source"] == "researcher"
    assert summary["action"] == "broaden"
    assert summary["lane"] == "academic_search"
    assert summary["rationale"] == "Sparse venue scout; broadened topic concepts."
    assert summary["summary"] == "Sparse venue scout; broadened topic concepts."


def test_market_intel_merges_linkedin_and_github_channels(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    linkedin_output = tmp_path / "output"
    github_output = tmp_path / "output" / "github"
    write_json(brief_path, read_json(_require_source_brief_path()))

    _write_linkedin_inputs(
        linkedin_output,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    linkedin_run_dir = _import_run_dir(
        brief_path=brief_path,
        legacy_output_dir=linkedin_output,
        source="linkedin",
    )
    update_market_intel(
        brief_path=brief_path,
        run_dir=linkedin_run_dir,
        mode="post_run",
    )

    _write_github_inputs(
        github_output,
        brief_path,
        generated_at="2026-04-08T14:00:00+00:00",
    )
    github_run_dir = _import_run_dir(
        brief_path=brief_path,
        legacy_output_dir=github_output,
        source="github",
    )
    merged = update_market_intel(
        brief_path=brief_path,
        run_dir=github_run_dir,
        mode="post_run",
    )

    assert sorted(merged.market_identity.channels_seen) == ["github", "linkedin"]
    assert set(merged.channel_summaries) == {"github", "linkedin"}
    assert merged.aggregate_metrics["run_count"] >= 2
    assert any(lane["lane_key"] == "trl_contributors" for lane in merged.lane_intelligence)


def test_two_lane_report_produces_two_lane_rows_in_artifact(tmp_path):
    """P7 Stage B (market-loop ingestion) regression guard.

    A single LinkedIn run report whose ``string_performance`` carries two
    strings with distinct ``family_key``/``domain_lane`` values must produce
    two distinct rows in the market-intelligence artifact's
    ``lane_intelligence`` — not collapse into one bucket. Lane collapse
    happens silently whenever every string gets labeled with the same
    (or a defaulted-to-"general") lane, so this locks the end-to-end path
    from a two-lane report through ``_aggregate_lane_intelligence`` to the
    persisted artifact.
    """
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    # Overwrite the run report with a two-lane string_performance, built off
    # the shared fixture's shape rather than editing _report_dict itself.
    report = _report_dict(version="2.1", generated_at="2026-04-08T12:00:00+00:00")
    report["string_performance"] = [
        {
            "string_id": 2,
            "name": "Payments orchestration lane",
            "status": "done",
            "result_count": 400,
            "pages_reviewed": 3,
            "saves": 6,
            "save_rate": 0.06,
            "saved_candidates": ["Payments Builder"],
            "notes": "Payments orchestration builders",
            "facial_yes_count": 8,
            "facial_no_count": 10,
            "candidates_count": 18,
            "duplicates_count": 1,
            "family_key": "payments_orchestration",
            "novelty_bucket": "edge_case",
            "domain_lane": "payments",
        },
        {
            "string_id": 3,
            "name": "Capital markets trading lane",
            "status": "done",
            "result_count": 300,
            "pages_reviewed": 2,
            "saves": 4,
            "save_rate": 0.05,
            "saved_candidates": ["Capital Markets Builder"],
            "notes": "Capital markets trading systems builders",
            "facial_yes_count": 6,
            "facial_no_count": 9,
            "candidates_count": 15,
            "duplicates_count": 0,
            "family_key": "capital_markets_trading",
            "novelty_bucket": "edge_case",
            "domain_lane": "capital_markets",
        },
    ]
    write_json(output_dir / "run-report.json", report)

    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)

    artifact = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
    )
    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir)
    loaded = MarketIntelArtifact.from_dict(read_json(artifact_path))

    lane_rows = artifact.lane_intelligence
    assert len(lane_rows) >= 2
    lane_keys = {row["lane_key"] for row in lane_rows}
    domain_lanes = {row["domain_lane"] for row in lane_rows}
    assert {"payments_orchestration", "capital_markets_trading"} <= lane_keys
    assert {"payments", "capital_markets"} <= domain_lanes

    loaded_lane_rows = loaded.lane_intelligence
    assert len(loaded_lane_rows) >= 2
    loaded_lane_keys = {row["lane_key"] for row in loaded_lane_rows}
    loaded_domain_lanes = {row["domain_lane"] for row in loaded_lane_rows}
    assert {"payments_orchestration", "capital_markets_trading"} <= loaded_lane_keys
    assert {"payments", "capital_markets"} <= loaded_domain_lanes


def test_legacy_canonical_artifact_is_repaired_on_load(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)
    first = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
    )
    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir)
    _seed_legacy_invalid_open_question(artifact_path, keep_valid=True)

    repaired = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="scheduled",
    )
    reloaded = MarketIntelArtifact.from_dict(read_json(artifact_path))

    assert repaired.aggregate_metrics["saved_count"] == first.aggregate_metrics["saved_count"]
    assert all(
        question.get("supporting_run_refs") or question.get("evidence_refs")
        for question in reloaded.open_questions
    )
    assert all(
        question["question"] != "Unsourced legacy question"
        for question in reloaded.open_questions
    )


def test_merge_path_keeps_valid_prior_questions_and_new_valid_questions_only(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)
    update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
    )
    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir)
    _seed_legacy_invalid_open_question(artifact_path, keep_valid=True)

    merged = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="scheduled",
        synthesis_backend=_CustomQuestionSynthesisBackend(),
    )

    questions = {question["question"] for question in merged.open_questions}
    assert "Unsourced legacy question" not in questions
    assert "How should we cover Payments more directly?" in questions
    assert "Should we target adjacent fintech teams?" in questions


def test_scheduled_refresh_with_legacy_invalid_open_questions_succeeds_without_new_run(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)
    update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
        external_research_backend=_StubResearchBackend(),
        with_external_research=True,
    )
    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir)
    _seed_legacy_invalid_open_question(artifact_path, keep_valid=False)

    refreshed = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="scheduled",
    )

    assert refreshed.artifact_version >= 2
    assert refreshed.open_questions
    assert all(
        question.get("supporting_run_refs") or question.get("evidence_refs")
        for question in refreshed.open_questions
    )


def test_markdown_renderer_handles_empty_open_questions_after_filtering(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)
    update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
    )
    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir)
    artifact = read_json(artifact_path)
    artifact["open_questions"] = [
        {
            "question": "Unsourced legacy question",
            "priority": "low",
            "next_step": "Leave it hanging.",
        }
    ]
    write_json(artifact_path, artifact)

    refreshed = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="scheduled",
        synthesis_backend=_CustomQuestionSynthesisBackend(),
    )
    refreshed.open_questions = []
    markdown = render_market_intel_markdown(refreshed)

    assert "## Risks and Unknowns" in markdown
    assert "- None" in markdown


def test_research_context_bundle_selects_bounded_full_packets():
    identity = MarketIntelArtifact.from_dict(
        {
            "schema_version": 1,
            "artifact_version": 1,
            "market_identity": {
                "market_key": "head_applied_ai_lab__new_york__director",
                "role_title": "Head of Applied AI Lab",
                "role_level": "Director",
                "geography": "New York",
                "channels_seen": ["linkedin"],
                "brief_ids_seen": ["head-ai-lab"],
                "brief_versions_seen": ["2.1"],
            },
            "freshness": {
                "artifact_updated_at": "2026-04-08T13:00:00+00:00",
                "internal_data_through": "2026-04-08T13:00:00+00:00",
                "external_research_through": "",
                "staleness_days": 0,
            },
            "evidence_index": {"runs": [], "external_sources": []},
            "aggregate_metrics": {
                "run_count": 0,
                "saved_count": 0,
                "rejected_count": 0,
                "facial_yes_rate": 0.0,
                "save_rate": 0.0,
                "candidate_volume_by_channel": {},
            },
            "channel_summaries": {},
            "lane_intelligence": [],
            "talent_pool_intelligence": [],
            "noise_patterns": [],
            "employer_signal_intelligence": [],
            "candidate_signal_summary": {
                "standout_signals": [],
                "borderline_signals": [],
                "disqualifying_signals": [],
            },
            "market_thesis": {
                "summary": "",
                "supply_assessment": "moderate",
                "competition_assessment": "high",
                "external_context": [],
            },
            "brief_recommendations": [],
            "open_questions": [],
        }
    ).market_identity
    bundle = build_research_context_bundle(
        identity,
        [
            _make_research_batch(
                run_ref="linkedin:output/run-a",
                generated_at="2026-04-08T10:00:00+00:00",
                saved=5,
                candidates=100,
            ),
            _make_research_batch(
                run_ref="linkedin:output/run-b",
                generated_at="2026-04-08T11:00:00+00:00",
                saved=16,
                candidates=120,
            ),
            _make_research_batch(
                run_ref="linkedin:output/run-c",
                generated_at="2026-04-08T12:00:00+00:00",
                saved=7,
                candidates=250,
            ),
            _make_research_batch(
                run_ref="linkedin:output/run-d",
                generated_at="2026-04-08T13:00:00+00:00",
                saved=6,
                candidates=140,
            ),
        ],
    )

    assert len(bundle["run_packets"]) <= 3
    run_refs = {
        packet["context_metadata"]["run_ref"] for packet in bundle["run_packets"]
    }
    assert "linkedin:output/run-d" in run_refs
    assert "linkedin:output/run-b" in run_refs
    assert "linkedin:output/run-c" in run_refs
    assert bundle["historical_rollup"]["additional_run_count"] == 1
    assert "observed_success_patterns" in bundle
    assert "observed_failures_and_false_negatives" in bundle
    assert "title_and_archetype_blind_spots" in bundle
    assert "adaptation_lessons" in bundle


def test_research_prompt_embeds_structured_bundle():
    identity = MarketEvidenceBatch(
        run_ref="linkedin:output/run-a",
        source="linkedin",
        output_dir="/tmp/run-a",
        brief_version="2.1",
        generated_at="2026-04-08T10:00:00+00:00",
    )
    bundle = build_research_context_bundle(
        MarketIntelArtifact.from_dict(
            {
                "schema_version": 1,
                "artifact_version": 1,
                "market_identity": {
                    "market_key": "head_applied_ai_lab__new_york__director",
                    "role_title": "Head of Applied AI Lab",
                    "role_level": "Director",
                    "geography": "New York",
                    "channels_seen": ["linkedin"],
                    "brief_ids_seen": ["head-ai-lab"],
                    "brief_versions_seen": ["2.1"],
                },
                "freshness": {
                    "artifact_updated_at": "2026-04-08T13:00:00+00:00",
                    "internal_data_through": "2026-04-08T13:00:00+00:00",
                    "external_research_through": "",
                    "staleness_days": 0,
                },
                "evidence_index": {"runs": [], "external_sources": []},
                "aggregate_metrics": {
                    "run_count": 0,
                    "saved_count": 0,
                    "rejected_count": 0,
                    "facial_yes_rate": 0.0,
                    "save_rate": 0.0,
                    "candidate_volume_by_channel": {},
                },
                "channel_summaries": {},
                "lane_intelligence": [],
                "talent_pool_intelligence": [],
                "noise_patterns": [],
                "employer_signal_intelligence": [],
                "candidate_signal_summary": {
                    "standout_signals": [],
                    "borderline_signals": [],
                    "disqualifying_signals": [],
                },
                "market_thesis": {
                    "summary": "",
                    "supply_assessment": "moderate",
                    "competition_assessment": "high",
                    "external_context": [],
                },
                "brief_recommendations": [],
                "open_questions": [],
            }
        ).market_identity,
        [identity],
    )
    prompt = build_research_user_prompt(
        MarketIntelArtifact.from_dict(
            {
                "schema_version": 1,
                "artifact_version": 1,
                "market_identity": {
                    "market_key": "head_applied_ai_lab__new_york__director",
                    "role_title": "Head of Applied AI Lab",
                    "role_level": "Director",
                    "geography": "New York",
                    "channels_seen": ["linkedin"],
                    "brief_ids_seen": ["head-ai-lab"],
                    "brief_versions_seen": ["2.1"],
                },
                "freshness": {
                    "artifact_updated_at": "2026-04-08T13:00:00+00:00",
                    "internal_data_through": "2026-04-08T13:00:00+00:00",
                    "external_research_through": "",
                    "staleness_days": 0,
                },
                "evidence_index": {"runs": [], "external_sources": []},
                "aggregate_metrics": {
                    "run_count": 0,
                    "saved_count": 0,
                    "rejected_count": 0,
                    "facial_yes_rate": 0.0,
                    "save_rate": 0.0,
                    "candidate_volume_by_channel": {},
                },
                "channel_summaries": {},
                "lane_intelligence": [],
                "talent_pool_intelligence": [],
                "noise_patterns": [],
                "employer_signal_intelligence": [],
                "candidate_signal_summary": {
                    "standout_signals": [],
                    "borderline_signals": [],
                    "disqualifying_signals": [],
                },
                "market_thesis": {
                    "summary": "",
                    "supply_assessment": "moderate",
                    "competition_assessment": "high",
                    "external_context": [],
                },
                "brief_recommendations": [],
                "open_questions": [],
            }
        ).market_identity,
        bundle,
    )
    assert "\"run_packets\"" in prompt
    assert "\"cross_run_aggregate\"" in prompt
    assert "\"observed_success_patterns\"" in prompt
    assert "\"title_and_archetype_blind_spots\"" in prompt


def test_normalize_draft_sections_backfills_missing_market_thesis_keys():
    normalized = agent_backends_mod._normalize_draft_sections(
        {
            "market_thesis": {
                "external_context": [],
            }
        }
    )
    assert normalized["market_thesis"]["summary"] == ""
    assert normalized["market_thesis"]["supply_assessment"] == "unknown"
    assert normalized["market_thesis"]["competition_assessment"] == "unknown"


def test_backfill_reconstructs_packet_from_raw_artifacts(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
        include_report_input=False,
        include_report=False,
    )
    store = RuntimeStateStore(output_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="3000000006",
        output_dir=str(output_dir),
        mode="full_run",
        resume_state={"brief_name": "head-ai-lab"},
    )
    with store.connect() as conn:
        conn.execute(
            """
            UPDATE runs
            SET status = ?, ended_at = ?
            WHERE id = ?
            """,
            (
                "governor_limit_reached",
                "2026-04-08T12:00:00+00:00",
                run_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO work_units(
                run_id, source, brief_id, kind, source_unit_id, display_name, ordering_index, status,
                payload_json, checkpoint_json, metrics_json, family_key, novelty_bucket, domain_lane,
                result_count, candidates_discovered, facial_yes_count, facial_no_count, saves_count,
                rejected_count, notes, started_at, ended_at
            )
            VALUES (?, 'linkedin', ?, 'linkedin_string', '2', 'Research copilot lane', 1, 'done',
                    ?, ?, ?, 'research_copilot_asset_mgmt', 'edge_case', 'asset_management',
                    526, 22, 10, 12, 8, 14, 'Strong lane', ?, ?)
            """,
            (
                run_id,
                "3000000006",
                json.dumps({"boolean": "research copilot lane"}),
                json.dumps({"pages_reviewed": 4, "duplicates_count": 1}),
                json.dumps({"pages_reviewed": 4, "duplicates_count": 1}),
                "2026-04-08T10:00:00+00:00",
                "2026-04-08T12:00:00+00:00",
            ),
        )

    artifact = update_market_intel(
        brief_path=brief_path,
        output_dir=output_dir,
        mode="backfill",
        reconstruct_report_analysis=True,
    )

    run_record = artifact.evidence_index["runs"][0]
    run_dir = Path(run_record.get("run_dir") or run_record["output_dir"])
    research_input = read_json(run_dir / "market-intel-research-input.json")
    assert research_input["context_metadata"]["context_quality"] == "reconstructed_report"
    assert research_input["report_analysis"]["winning_lanes"]
    assert artifact.evidence_index["runs"][0]["context_quality"] == "reconstructed_report"


def test_cli_post_run_smoke_with_run_dir(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "update_market_intel.py"),
            "--brief",
            str(brief_path),
            "--run-dir",
            str(run_dir),
            "--mode",
            "post_run",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Research input:" in result.stdout
    assert "Run dir:" in result.stdout


def test_market_intel_rejects_bare_output_root_for_post_run(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )

    try:
        update_market_intel(
            brief_path=brief_path,
            output_dir=output_dir,
            mode="post_run",
        )
    except ValueError as exc:
        assert "finalized run_dir" in str(exc)
    else:
        raise AssertionError("expected post_run bare output root to be rejected")


def test_external_research_skips_incomplete_explicit_linkedin_run(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    write_json(brief_path, read_json(_require_source_brief_path()))
    output_dir = resolve_linkedin_state_dir(brief_path=brief_path, state_dir=tmp_path / "output" / "state" / "linkedin" / "3000000006")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
        include_report_input=False,
        include_report=False,
    )
    store = RuntimeStateStore(output_dir / "runtime_state.sqlite3")
    store.start_run(
        source="linkedin",
        brief_id="3000000006",
        output_dir=str(output_dir),
        mode="full_run",
        resume_state={"brief_name": "head-ai-lab"},
    )
    backend = _CountingResearchBackend()

    update_market_intel(
        brief_path=brief_path,
        output_dir=output_dir,
        mode="post_run",
        external_research_backend=backend,
        with_external_research=True,
        allow_live_state_dir=True,
    )

    assert backend.calls == 0


def test_finalize_run_snapshot_creates_run_manifest_and_research_input(tmp_path):
    """Healthy run keeps its OWN report in the manifest + research-input.

    Reconciled for the run-scoped report-ownership scrub: the report's
    ``run_metadata.generated_at`` (2026-04-08T12:00:00Z) is now bracketed by
    the run's ``started_at``/``ended_at`` via an explicit UPDATE, so the
    report is genuinely owned by this run. As originally written the run
    window came from wall-clock ``finish_run``, leaving the 2026-04-08 report
    outside its own run window — i.e. the success-masquerade shape the scrub
    now rejects. This stamps a legitimately-healthy in-window run instead.
    """

    brief_path = tmp_path / SOURCE_BRIEF.name
    write_json(brief_path, read_json(_require_source_brief_path()))
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path,
        state_dir=tmp_path / "output" / "state" / "linkedin" / "3000000006",
    )
    _write_linkedin_inputs(
        state_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="3000000006",
        output_dir=str(state_dir),
        mode="full_run",
        resume_state={"brief_name": "head-ai-lab"},
    )
    store.finish_run(run_id, status="completed")
    # The report this run produced is stamped 2026-04-08T12:00:00Z; bracket the
    # run window around it so it reads as owned by this run (in-window).
    with store.connect() as conn:
        conn.execute(
            "UPDATE runs SET started_at = ?, ended_at = ? WHERE id = ?",
            (
                "2026-04-08T11:00:00+00:00",
                "2026-04-08T13:00:00+00:00",
                run_id,
            ),
        )

    run_dir = finalize_run_snapshot(
        source="linkedin",
        brief_path=brief_path,
        state_dir=state_dir,
        run_id=run_id,
    )

    manifest = read_json(run_dir / "run-manifest.json")
    assert classify_output_location(run_dir) == "run_dir"
    assert manifest["source"] == "linkedin"
    assert manifest["state_dir"] == str(state_dir.resolve())
    assert manifest["research_input_path"] == str(run_dir / "market-intel-research-input.json")
    assert "run-report.json" in manifest["artifacts_present"]
    assert (run_dir / "market-intel-research-input.json").exists()
    assert (run_dir / "runtime_state.sqlite3").exists()


# --- Run-scoped report ownership: the success-masquerade fix ----------------
#
# ``state_dir`` is the STABLE per-brief working dir, so a prior run's
# ``run-report.json`` physically persists there until the next run overwrites
# it. Brief-identity matching (role_title / brief_name / linkedin_project_id)
# cannot tell a prior run of the SAME brief from the current one. Before the
# fix, an interrupted / zero-save run that produced no fresh report inherited
# the prior run's healthy report (saved=12) and was indistinguishable from a
# healthy run at the snapshot surface the recruiter + market-intel ingest read.
#
# These tests use the committed ``head-of-applied-ai`` brief fixture (the
# stale optional ``config/brief-head-ai-lab-nyc-v2.json`` that the older
# fixtures above depend on is absent in CI), and write the report artifacts
# directly so ``run_metadata`` identity + ``generated_at`` are under test
# control.

HOA_BRIEF_FIXTURE = ROOT / "config" / "head-of-applied-ai-fixture" / "brief.json"


def _write_owned_report_artifacts(
    output_dir: Path,
    *,
    generated_at: str,
    saved: int = 12,
    run_id: int | None = None,
    include_report_input: bool = True,
) -> None:
    """Write run-report(.json/-input.json) + raw artifacts for the HOA fixture.

    ``run_metadata.role_title`` matches the committed fixture so the
    brief-identity gate keeps the report; ``generated_at`` (and optionally
    ``run_id``) decide run ownership.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata: dict = {
        "role_title": "Head of Applied AI",
        "brief_name": "brief",
        "generated_at": generated_at,
        "overall_summary": "Structured debrief input for market intelligence.",
    }
    if run_id is not None:
        run_metadata["run_id"] = run_id
    metrics_summary = {
        "candidates_evaluated": 180,
        "facial_yes": 40,
        "facial_no": 140,
        "saved": saved,
        "rejected": 28,
    }
    write_json(
        output_dir / "run-report.json",
        {
            "schema_version": 1,
            "run_metadata": run_metadata,
            "metrics_summary": metrics_summary,
        },
    )
    if include_report_input:
        write_json(
            output_dir / "run-report-input.json",
            {
                "schema_version": 1,
                "run_metadata": run_metadata,
                "metrics_summary": metrics_summary,
            },
        )
    _write_jsonl(
        output_dir / "final_judgments.jsonl",
        [
            {
                "stage": "full",
                "decision": "SAVE",
                "path": "DIRECT:Applied AI",
                "profile_url": "https://example.com/owned",
                "confidence": 0.82,
                "rationale": "Built applied AI platforms in production.",
                "candidate_name": "Owned Candidate",
            }
        ],
    )


def test_finalize_run_snapshot_does_not_inherit_prior_run_report_on_interrupted_run(
    tmp_path,
):
    """Interrupted / zero-save run must NOT inherit a prior run's report.

    Arrange a PRIOR run's healthy ``run-report.json`` (saved=12, old
    ``generated_at`` of 2026-01-01) lingering in the stable per-brief
    ``state_dir``. The CURRENT run is started and finished as ``interrupted``
    having saved nothing and produced no fresh report. The prior report
    matches the brief identity, so the legacy brief-identity scrub alone
    would keep it — the success-masquerade. The run-scoped scrub must reject
    it: its ``generated_at`` falls outside the current run's window.
    """

    brief_path = tmp_path / "brief.json"
    write_json(brief_path, read_json(HOA_BRIEF_FIXTURE))
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path,
        state_dir=tmp_path / "output" / "state" / "linkedin" / "haai",
    )
    # Prior run's healthy report + its raw artifacts, with an OLD generated_at.
    _write_owned_report_artifacts(
        state_dir, generated_at="2026-01-01T00:00:00+00:00", saved=12
    )
    write_json(state_dir / "market-intel-research-input.json", {"prior": "research"})

    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="haai",
        output_dir=str(state_dir),
        mode="full_run",
        resume_state={},
    )
    store.finish_run(run_id, status="interrupted")

    run_dir = finalize_run_snapshot(
        source="linkedin",
        brief_path=brief_path,
        state_dir=state_dir,
        run_id=run_id,
    )

    manifest = read_json(run_dir / "run-manifest.json")
    # The unowned prior report is scrubbed from BOTH the manifest and disk.
    assert "run-report.json" not in manifest["artifacts_present"]
    assert "run-report-input.json" not in manifest["artifacts_present"]
    assert not (run_dir / "run-report.json").exists()
    assert not (run_dir / "run-report-input.json").exists()
    # Only the unowned report is scrubbed — the run's other artifacts survive.
    assert (run_dir / "runtime_state.sqlite3").exists()
    assert (run_dir / "final_judgments.jsonl").exists()
    assert (run_dir / "market-intel-research-input.json").exists()


def test_finalize_run_snapshot_keeps_own_report_generated_within_run_window(tmp_path):
    """Non-regression: a HEALTHY run keeps its OWN in-window report.

    The fix must not over-scrub. A report whose ``generated_at`` lands inside
    the current run's ``[started_at, ended_at]`` window is owned by the run
    and must remain in both ``artifacts_present`` and on disk.
    """

    brief_path = tmp_path / "brief.json"
    write_json(brief_path, read_json(HOA_BRIEF_FIXTURE))
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path,
        state_dir=tmp_path / "output" / "state" / "linkedin" / "haai",
    )

    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="haai",
        output_dir=str(state_dir),
        mode="full_run",
        resume_state={},
    )
    # The run generates its own report DURING the window (after started_at).
    run_record = store.get_run(run_id)
    _write_owned_report_artifacts(
        state_dir, generated_at=run_record["started_at"], saved=12
    )
    store.finish_run(run_id, status="completed")

    run_dir = finalize_run_snapshot(
        source="linkedin",
        brief_path=brief_path,
        state_dir=state_dir,
        run_id=run_id,
    )

    manifest = read_json(run_dir / "run-manifest.json")
    assert "run-report.json" in manifest["artifacts_present"]
    assert (run_dir / "run-report.json").exists()


def test_import_legacy_run_snapshot_does_not_inherit_prior_run_report(tmp_path):
    """import_legacy_run_snapshot must apply the same run-scoped scrub.

    Legacy imports run the identical copy+scrub sequence; without the shared
    fix a legacy import would re-introduce the masquerade. Mirror the
    interrupted-run arrange against the import entrypoint.
    """

    brief_path = tmp_path / "brief.json"
    legacy_output_dir = tmp_path / "output"
    write_json(brief_path, read_json(HOA_BRIEF_FIXTURE))
    # Prior run's healthy report lingering in the legacy output dir, OLD stamp.
    _write_owned_report_artifacts(
        legacy_output_dir, generated_at="2026-01-01T00:00:00+00:00", saved=12
    )

    store = RuntimeStateStore(legacy_output_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="haai",
        output_dir=str(legacy_output_dir),
        mode="full_run",
        resume_state={},
    )
    # Interrupted current run with a known LATER window, well after 2026-01-01.
    with store.connect() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, started_at = ?, ended_at = ? WHERE id = ?",
            (
                "interrupted",
                "2026-04-08T00:00:00+00:00",
                "2026-04-08T04:43:28.957197+00:00",
                run_id,
            ),
        )

    run_dir = _import_run_dir(
        brief_path=brief_path,
        legacy_output_dir=legacy_output_dir,
        run_id=run_id,
    )

    manifest = read_json(run_dir / "run-manifest.json")
    assert "run-report.json" not in manifest["artifacts_present"]
    assert "run-report-input.json" not in manifest["artifacts_present"]
    assert not (run_dir / "run-report.json").exists()
    assert not (run_dir / "run-report-input.json").exists()
    assert (run_dir / "final_judgments.jsonl").exists()


def test_import_legacy_run_snapshot_honors_requested_run_id(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    legacy_output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        legacy_output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
        include_report_input=False,
        include_report=False,
    )
    store = RuntimeStateStore(legacy_output_dir / "runtime_state.sqlite3")
    first_run_id = store.start_run(
        source="linkedin",
        brief_id="3000000006",
        output_dir=str(legacy_output_dir),
        mode="full_run",
        resume_state={"brief_name": "head-ai-lab"},
    )
    second_run_id = store.start_run(
        source="linkedin",
        brief_id="3000000006",
        output_dir=str(legacy_output_dir),
        mode="resume",
        resume_state={"brief_name": "head-ai-lab"},
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
            ("completed", "2026-04-08T04:43:28.957197+00:00", first_run_id),
        )
        conn.execute(
            "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
            ("completed", "2026-04-08T22:44:23.171843+00:00", second_run_id),
        )

    run_dir = _import_run_dir(
        brief_path=brief_path,
        legacy_output_dir=legacy_output_dir,
        run_id=first_run_id,
        reconstruct_report_analysis=True,
    )

    manifest = read_json(run_dir / "run-manifest.json")
    assert manifest["run_id"] == first_run_id
    assert manifest["ended_at"] == "2026-04-08T04:43:28.957197+00:00"
    assert "2026-04-08T04-43-28" in run_dir.name


def test_import_legacy_run_snapshot_filters_unrelated_project_artifacts(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    legacy_output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        legacy_output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
        include_report_input=False,
        include_report=False,
    )

    write_json(legacy_output_dir / "bias_monitor-3000000006.json", {"decisions": 42})
    write_json(legacy_output_dir / "search_memory-999999.json", {"project_id": "999999"})
    write_json(
        legacy_output_dir / "bias_monitor-999999.backup-before-import.json",
        {"decisions": 7},
    )
    (legacy_output_dir / "candidate_history-3000000006.jsonl").write_text("{}\n")
    (legacy_output_dir / "candidate_history-999999.jsonl").write_text("{}\n")
    (legacy_output_dir / "noise_discoveries-3000000006.jsonl").write_text("{}\n")
    (legacy_output_dir / "noise_discoveries-999999.jsonl").write_text("{}\n")

    store = RuntimeStateStore(legacy_output_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="3000000006",
        output_dir=str(legacy_output_dir),
        mode="resume",
        resume_state={"brief_name": "head-ai-lab"},
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
            ("completed", "2026-04-08T22:44:23.171843+00:00", run_id),
        )

    run_dir = _import_run_dir(
        brief_path=brief_path,
        legacy_output_dir=legacy_output_dir,
        run_id=run_id,
        reconstruct_report_analysis=True,
    )

    assert (run_dir / "search_memory-3000000006.json").exists()
    assert (run_dir / "bias_monitor-3000000006.json").exists()
    assert (run_dir / "candidate_history-3000000006.jsonl").exists()
    assert (run_dir / "noise_discoveries-3000000006.jsonl").exists()
    assert not (run_dir / "search_memory-999999.json").exists()
    assert not (run_dir / "bias_monitor-999999.backup-before-import.json").exists()
    assert not (run_dir / "candidate_history-999999.jsonl").exists()
    assert not (run_dir / "noise_discoveries-999999.jsonl").exists()


def test_import_legacy_run_snapshot_filters_run_log_to_requested_window(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    legacy_output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        legacy_output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
        include_report_input=False,
        include_report=False,
    )

    _write_jsonl(
        legacy_output_dir / "run_log.jsonl",
        [
            {
                "timestamp": "2026-04-08T08:00:00+00:00",
                "event": "early_run",
                "message": "Should be excluded",
            },
            {
                "timestamp": "2026-04-08T22:00:00+00:00",
                "event": "target_run",
                "message": "Should be kept",
            },
            {
                "timestamp": "2026-04-09T01:00:00+00:00",
                "event": "later_run",
                "message": "Should be excluded",
            },
        ],
    )

    store = RuntimeStateStore(legacy_output_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="3000000006",
        output_dir=str(legacy_output_dir),
        mode="resume",
        resume_state={"brief_name": "head-ai-lab"},
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE runs SET started_at = ?, status = ?, ended_at = ? WHERE id = ?",
            (
                "2026-04-08T21:30:00+00:00",
                "completed",
                "2026-04-08T22:30:00+00:00",
                run_id,
            ),
        )

    run_dir = _import_run_dir(
        brief_path=brief_path,
        legacy_output_dir=legacy_output_dir,
        run_id=run_id,
        reconstruct_report_analysis=True,
    )

    run_log = read_jsonl(run_dir / "run_log.jsonl")
    assert [item["event"] for item in run_log] == ["target_run"]


def test_backfill_from_shared_legacy_dir_uses_run_scoped_runtime_payloads(tmp_path):
    brief_path = tmp_path / FDE_BRIEF.name
    legacy_output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_fde_brief_path_v13()))

    # Seed a mismatched LinkedIn report to simulate a mixed legacy workspace.
    _write_linkedin_inputs(
        legacy_output_dir,
        brief_path,
        version="1.3",
        generated_at="2026-04-09T18:26:34+00:00",
        include_report_input=True,
        include_report=True,
    )

    store = RuntimeStateStore(legacy_output_dir / "runtime_state.sqlite3")
    run_two_id = store.start_run(
        source="linkedin",
        brief_id="3000000007",
        output_dir=str(legacy_output_dir),
        mode="fresh",
        resume_state={"brief_name": "Forward Deployed Engineer"},
    )
    run_five_id = store.start_run(
        source="linkedin",
        brief_id="3000000007",
        output_dir=str(legacy_output_dir),
        mode="resume",
        resume_state={"brief_name": "Forward Deployed Engineer"},
    )
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO work_units(
                run_id, source, brief_id, kind, source_unit_id, display_name, ordering_index, status,
                payload_json, checkpoint_json, metrics_json, family_key, novelty_bucket, domain_lane,
                result_count, candidates_discovered, facial_yes_count, facial_no_count, saves_count,
                rejected_count, notes, started_at, ended_at
            )
            VALUES (?, 'linkedin', '3000000007', 'linkedin_string', '2', 'Older lane', 1, 'done',
                    ?, ?, ?, 'older_lane', 'edge_case', 'general', 120, 10, 4, 6, 1, 9, 'older',
                    ?, ?)
            """,
            (
                run_two_id,
                json.dumps({"id": 2, "name": "Older lane", "boolean": "older lane"}),
                json.dumps({"pages_reviewed": 2, "duplicates_count": 0}),
                json.dumps({"pages_reviewed": 2, "duplicates_count": 0}),
                "2026-04-08T00:21:03.014067+00:00",
                "2026-04-08T04:43:28.957197+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO work_units(
                run_id, source, brief_id, kind, source_unit_id, display_name, ordering_index, status,
                payload_json, checkpoint_json, metrics_json, family_key, novelty_bucket, domain_lane,
                result_count, candidates_discovered, facial_yes_count, facial_no_count, saves_count,
                rejected_count, notes, started_at, ended_at
            )
            VALUES (?, 'linkedin', '3000000007', 'linkedin_string', '5', 'Newer lane', 1, 'done',
                    ?, ?, ?, 'newer_lane', 'canonical', 'general', 300, 20, 10, 10, 7, 13, 'newer',
                    ?, ?)
            """,
            (
                run_five_id,
                json.dumps({"id": 5, "name": "Newer lane", "boolean": "newer lane"}),
                json.dumps({"pages_reviewed": 5, "duplicates_count": 2}),
                json.dumps({"pages_reviewed": 5, "duplicates_count": 2}),
                "2026-04-08T18:50:11.108763+00:00",
                "2026-04-08T22:44:23.171843+00:00",
            ),
        )
        conn.execute(
            "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
            ("completed", "2026-04-08T04:43:28.957197+00:00", run_two_id),
        )
        conn.execute(
            "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
            ("completed", "2026-04-08T22:44:23.171843+00:00", run_five_id),
        )

    _insert_linkedin_attempt_payloads(
        store=store,
        run_id=run_two_id,
        brief_id="3000000007",
        identity_key="older-candidate",
        candidate_name="Older Candidate",
        decision="SAVE",
        started_at="2026-04-08T00:21:03.014067+00:00",
        ended_at="2026-04-08T04:43:28.957197+00:00",
    )
    _insert_linkedin_attempt_payloads(
        store=store,
        run_id=run_five_id,
        brief_id="3000000007",
        identity_key="newer-candidate",
        candidate_name="Newer Candidate",
        decision="REJECT",
        started_at="2026-04-08T18:50:11.108763+00:00",
        ended_at="2026-04-08T22:44:23.171843+00:00",
    )

    artifact = update_market_intel(
        brief_path=brief_path,
        legacy_output_dir=legacy_output_dir,
        run_id=run_two_id,
        mode="backfill",
        reconstruct_report_analysis=True,
    )

    run_record = artifact.evidence_index["runs"][0]
    run_dir = Path(run_record["run_dir"])
    final_records = [json.loads(line) for line in (run_dir / "final_judgments.jsonl").read_text().splitlines()]
    research_input = read_json(run_dir / "market-intel-research-input.json")

    assert run_record["run_id"] == run_two_id
    assert run_record["generated_at"] == "2026-04-08T04:43:28.957197+00:00"
    assert [record["candidate_name"] for record in final_records] == ["Older Candidate"]
    assert not (run_dir / "run-report.json").exists()
    assert research_input["context_metadata"]["analysis_provenance"] == "reconstructed_from_raw"
    assert artifact.aggregate_metrics["saved_count"] == 1


def test_cli_backfill_with_reconstruction_smoke(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
        include_report_input=False,
        include_report=False,
    )
    store = RuntimeStateStore(output_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="3000000006",
        output_dir=str(output_dir),
        mode="full_run",
        resume_state={"brief_name": "head-ai-lab"},
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
            ("completed", "2026-04-08T12:00:00+00:00", run_id),
        )
        conn.execute(
            """
            INSERT INTO work_units(
                run_id, source, brief_id, kind, source_unit_id, display_name, ordering_index, status,
                payload_json, checkpoint_json, metrics_json, family_key, novelty_bucket, domain_lane,
                result_count, candidates_discovered, facial_yes_count, facial_no_count, saves_count,
                rejected_count, notes, started_at, ended_at
            )
            VALUES (?, 'linkedin', ?, 'linkedin_string', '2', 'Research copilot lane', 1, 'done',
                    ?, ?, ?, 'research_copilot_asset_mgmt', 'edge_case', 'asset_management',
                    526, 22, 10, 12, 8, 14, 'Strong lane', ?, ?)
            """,
            (
                run_id,
                "3000000006",
                json.dumps({"boolean": "research copilot lane"}),
                json.dumps({"pages_reviewed": 4, "duplicates_count": 1}),
                json.dumps({"pages_reviewed": 4, "duplicates_count": 1}),
                "2026-04-08T10:00:00+00:00",
                "2026-04-08T12:00:00+00:00",
            ),
        )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "update_market_intel.py"),
            "--brief",
            str(brief_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "backfill",
            "--reconstruct-report-analysis",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Run dir:" in result.stdout
    assert "Research input:" in result.stdout


def test_update_market_intel_persists_agent_state_and_generation_metadata(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)

    artifact = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
    )

    agent_state_path = resolve_market_intel_agent_state_path(brief_path, output_dir=run_dir)
    research_log_path = resolve_market_intel_research_log_path(brief_path, output_dir=run_dir)
    assert agent_state_path.exists()
    assert research_log_path.exists()

    agent_state = MarketIntelAgentState.from_dict(read_json(agent_state_path))
    assert agent_state.market_key == artifact.market_identity.market_key
    assert artifact.section_generation_metadata["aggregate_metrics"]["generation_mode"] == "deterministic"
    assert "lane_intelligence" in artifact.section_generation_metadata
    assert artifact.delta_since_last_run["still_uncertain"] is not None
    assert any(
        hypothesis.supporting_run_refs
        for hypothesis in agent_state.active_hypotheses
    )
    assert read_jsonl(research_log_path)


def test_planner_gates_external_research_and_passes_focus_areas(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)
    backend = _PlannerCapturingResearchBackend()

    update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
        planner_backend=_AlwaysResearchPlannerBackend(),
        external_research_backend=backend,
        with_external_research=True,
    )

    assert backend.calls == 1
    assert backend.focus == [
        {
            "focus": "Employer demand and title-variant blind spots",
            "priority": "high",
            "reason": "Validate the strongest external uncertainty.",
            "supporting_run_refs": ["linkedin:output"],
        }
    ]


def test_build_external_research_backend_prefers_perplexity_when_available(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        research_agent_mod.config,
        "MARKET_INTEL_EXTERNAL_RESEARCH_PROVIDER",
        "auto",
        raising=False,
    )
    monkeypatch.setattr(
        research_agent_mod.config,
        "PERPLEXITY_API_KEY",
        "pplx-test",
        raising=False,
    )
    monkeypatch.setattr(
        research_agent_mod.config,
        "ANTHROPIC_API_KEY",
        "anth-test",
        raising=False,
    )

    backend = research_agent_mod.build_external_research_backend()

    assert isinstance(backend, research_agent_mod.PerplexityResearchBackend)


def test_perplexity_research_prompt_is_optimized_for_primary_recent_geo_specific_sources():
    identity = MarketIdentity.from_dict(
        {
            "market_key": "forward_deployed_engineer__new_york__ic5_ic6",
            "role_title": "Forward Deployed Engineer",
            "role_level": "IC5-IC6",
            "geography": "New York, New York, United States",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["3000000007"],
            "brief_versions_seen": ["1.3"],
        }
    )
    bundle = {
        "market_identity": identity.to_dict(),
        "cross_run_aggregate": {"run_count": 1},
        "run_packets": [],
        "historical_rollup": {},
    }
    instructions = build_perplexity_research_instructions()
    prompt = build_perplexity_research_user_prompt(
        identity,
        bundle,
        selected_questions=[
            {
                "question": "Which NYC employers are actively hiring FDE-like profiles?",
                "priority": "high",
                "next_step": "Use external research",
            }
        ],
        planner_summary="Need employer demand and supply-side context.",
    )

    assert "infer the most decision-relevant market questions" in instructions.lower()
    assert "primary or near-primary sources" in instructions.lower()
    assert "recent sources" in instructions.lower()
    assert "role- and geography-specific" in instructions.lower()
    assert "official company job pages" in prompt.lower()
    assert "new york, new york, united states" in prompt.lower()
    assert "improving sourcing for this role" in prompt.lower()


def test_edge_case_research_prompt_is_anchored_on_hidden_pools_and_false_negatives():
    identity = MarketIdentity.from_dict(
        {
            "market_key": "forward_deployed_engineer__new_york__ic5_ic6",
            "role_title": "Forward Deployed Engineer",
            "role_level": "IC5-IC6",
            "geography": "New York, New York, United States",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["3000000007"],
            "brief_versions_seen": ["1.3"],
        }
    )
    bundle = {
        "market_identity": identity.to_dict(),
        "edge_case_context": {
            "hidden_pool_risk_signals": [
                {
                    "label": "Coverage gaps suggest hidden supply",
                    "summary": "The run likely missed adjacent title families.",
                    "supporting_run_refs": ["linkedin:output"],
                }
            ]
        },
        "run_packets": [],
        "historical_rollup": {},
    }

    instructions = build_perplexity_edge_case_research_instructions()
    prompt = build_perplexity_edge_case_research_user_prompt(
        identity,
        bundle,
        edge_case_focus=[
            {
                "focus": "Which title families hide relevant candidates?",
                "priority": "high",
                "reason": "Recover false negatives caused by title fragmentation.",
                "supporting_run_refs": ["linkedin:output"],
            }
        ],
        planner_summary="General external research found title-variant ambiguity.",
        edge_case_reasoning="Coverage gaps and novelty-heavy saves suggest hidden-pool risk.",
    )

    assert "hidden pools" in instructions.lower()
    assert "false-negative risk" in instructions.lower()
    assert "self-label differently" in instructions.lower()
    assert "edge-case context to prioritize" in prompt.lower()
    assert "which title families hide relevant candidates?" in prompt.lower()
    assert "coverage gaps and novelty-heavy saves suggest hidden-pool risk" in prompt.lower()


def test_heuristic_planner_gates_edge_case_research_from_multiple_signals():
    planner = HeuristicPlannerBackend()
    identity = MarketIdentity.from_dict(
        {
            "market_key": "forward_deployed_engineer__new_york__ic5_ic6",
            "role_title": "Forward Deployed Engineer",
            "role_level": "IC5-IC6",
            "geography": "New York, New York, United States",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["3000000007"],
            "brief_versions_seen": ["1.3"],
        }
    )
    batch = MarketEvidenceBatch(
        run_ref="linkedin:output",
        source="linkedin",
        output_dir="/tmp/run",
        brief_version="1.3",
        generated_at="2026-04-08T04:43:28.957197+00:00",
        metrics_summary={"saved": 3, "candidates_evaluated": 22},
        research_context={
            "search_memory_summary": {
                "overall": {},
                "families": [
                    {
                        "family_key": "fde_hidden_pool",
                        "novelty_bucket": "edge_case",
                        "saves": 2,
                        "strings_seen": 1,
                        "save_rate": 0.2,
                        "dominant_anchors": ["solutions", "deployment", "customer"],
                    }
                ],
            },
            "deterministic_snapshot": {
                "string_performance": [
                    {
                        "name": "Hidden pool lane",
                        "family_key": "fde_hidden_pool",
                        "novelty_bucket": "edge_case",
                        "saves": 2,
                        "candidates_count": 10,
                    }
                ]
            },
            "candidate_evidence": {"saved_examples": [], "rejected_examples": []},
            "report_analysis": {
                "coverage_gaps": [
                    {
                        "gap": "Solutions-oriented titles",
                        "suggested_search_strategy": "Test deployment and solutions title families.",
                    }
                ],
                "underperforming_lanes": [
                    {
                        "lane": "Narrow FDE string",
                        "issue": "Good infrastructure profiles were absent despite expected market depth.",
                    }
                ],
                "saved_candidate_patterns": {
                    "common_titles": [
                        {"title_family": "Forward Deployed Engineer", "note": "Observed save title."},
                        {"title_family": "Solutions Architect", "note": "Suggests title fragmentation."},
                    ]
                },
            },
        },
    )

    result = planner.plan(
        market_identity=identity,
        deterministic_summary={"aggregate_metrics": {"saved_count": 3, "run_count": 1}},
        evidence_batches=[batch],
        previous_artifact=None,
        previous_agent_state=None,
    )

    assert result.should_collect_edge_case_research is True
    assert result.edge_case_research_focus
    assert "hidden-pool" in result.edge_case_research_reasoning.lower()


def test_perplexity_backend_packages_deep_research_into_external_research_result(
    monkeypatch: pytest.MonkeyPatch,
):
    identity = MarketIdentity.from_dict(
        {
            "market_key": "forward_deployed_engineer__new_york__ic5_ic6",
            "role_title": "Forward Deployed Engineer",
            "role_level": "IC5-IC6",
            "geography": "New York, New York, United States",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["3000000007"],
            "brief_versions_seen": ["1.3"],
        }
    )
    batch = MarketEvidenceBatch(
        run_ref="linkedin:output/runs/linkedin/3000000007/run-2",
        source="linkedin",
        output_dir="/tmp/run",
        brief_version="1.3",
        generated_at="2026-04-08T04:43:28.957197+00:00",
        research_context={
            "context_metadata": {"context_quality": "raw_only"},
            "deterministic_snapshot": {"metrics_summary": {"saved": 34}},
            "report_analysis": {},
        },
    )
    calls: dict[str, object] = {}

    class _FakePerplexityResponse:
        output_text = json.dumps(
            {
                "inferred_research_questions": [
                    {
                        "question": "Are senior customer-facing AI platform engineers hiding under adjacent titles?",
                        "priority": "high",
                        "why_it_matters": "The run may be missing hidden title variants.",
                        "sourcing_trigger": "Coverage gaps suggested title-family blind spots.",
                        "status": "answered",
                        "supporting_run_refs": [batch.run_ref],
                        "evidence_refs": ["https://example.com/jobs/fde"],
                    }
                ],
                "market_findings": [
                    {
                        "kind": "title_variant",
                        "label": "Customer Engineer",
                        "summary": "Several NYC companies continue hiring senior customer-facing AI platform engineers.",
                        "why_it_matters": "This title family may hide relevant FDE-like candidates.",
                        "confidence": 0.74,
                        "supporting_run_refs": [batch.run_ref],
                        "evidence_refs": ["https://example.com/jobs/fde"],
                    }
                ],
                "sourcing_implications": [
                    {
                        "category": "add_title_family",
                        "priority": "high",
                        "recommendation": "Add Customer Engineer as a title family in the next run.",
                        "rationale": "External research indicates this title family overlaps materially with the target market.",
                        "brief_target_field": "additional_search_terms",
                        "suggested_values": ["Customer Engineer"],
                        "expected_effect": "Should improve title coverage in NYC.",
                        "supporting_run_refs": [batch.run_ref],
                        "evidence_refs": ["https://example.com/jobs/fde"],
                    }
                ],
                "open_questions": [
                    {
                        "question": "Are enterprise AI platform vendors hiring faster than consultancies?",
                        "priority": "high",
                        "next_step": "Track hiring mix over the next run.",
                        "supporting_run_refs": [batch.run_ref],
                        "evidence_refs": ["https://example.com/report/fde-market"],
                    }
                ],
            }
        )

        def model_dump(self) -> dict:
            return {
                "output": [
                    {
                        "type": "search_results",
                        "results": [
                            {
                                "title": "Forward Deployed Engineer Jobs",
                                "url": "https://example.com/jobs/fde",
                            }
                        ],
                    },
                    {
                        "type": "fetch_url_results",
                        "contents": [
                            {
                                "title": "FDE Market Report",
                                "url": "https://example.com/report/fde-market",
                                "snippet": "Employers continue hiring.",
                            }
                        ],
                    },
                ]
            }

    class _FakeResponses:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return _FakePerplexityResponse()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.responses = _FakeResponses()

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(
        research_agent_mod.config,
        "PERPLEXITY_API_KEY",
        "pplx-test",
        raising=False,
    )
    monkeypatch.setattr(
        research_agent_mod.config,
        "MARKET_INTEL_PERPLEXITY_PRESET",
        "deep-research",
        raising=False,
    )

    backend = research_agent_mod.PerplexityResearchBackend()
    result = backend.collect(
        market_identity=identity,
        previous_artifact=None,
        previous_agent_state=None,
        evidence_batches=[batch],
        planner_result=SimpleNamespace(planner_summary="Need employer demand context."),
        research_focus=[
            {
                "focus": "Employer demand and title variants",
                "priority": "high",
                "reason": "The sourcing run may be missing hidden pools.",
                "supporting_run_refs": [batch.run_ref],
            }
        ],
    )

    assert calls["init"]["base_url"] == "https://api.perplexity.ai/v1"
    assert "response_format" not in calls["kwargs"]
    assert calls["kwargs"]["extra_body"]["preset"] == "deep-research"
    assert calls["kwargs"]["extra_body"]["response_format"]["type"] == "json_schema"
    assert calls["kwargs"]["tools"][0]["filters"]["search_recency_filter"] == "year"
    assert calls["kwargs"]["tools"][0]["user_location"]["country"] == "US"
    assert calls["kwargs"]["tools"][1]["type"] == "fetch_url"
    assert result.market_findings
    assert result.inferred_research_questions[0]["status"] == "answered"
    assert result.sourcing_implications[0]["brief_target_field"] == "additional_search_terms"
    assert result.market_thesis_context[0]["evidence_refs"] == ["web:https://example.com/jobs/fde"]
    assert result.open_questions[0]["evidence_refs"] == ["web:https://example.com/report/fde-market"]
    assert result.sources[0]["url"] == "https://example.com/jobs/fde"
    assert result.sources[1]["used_for"] == ["fetch_url"]


def test_perplexity_backend_records_error_receipt_when_provider_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    identity = MarketIdentity.from_dict(
        {
            "market_key": "forward_deployed_engineer__new_york__ic5_ic6",
            "role_title": "Forward Deployed Engineer",
            "role_level": "IC5-IC6",
            "geography": "New York, New York, United States",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["3000000007"],
            "brief_versions_seen": ["1.3"],
        }
    )
    batch = _make_research_batch(
        run_ref="linkedin:output/runs/linkedin/3000000007/run-2",
        generated_at="2026-04-08T04:43:28.957197+00:00",
        saved=3,
        candidates=20,
    )

    attempts = 0

    class _FailingResponses:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("pplx_503")

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = _FailingResponses()

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(research_agent_mod.config, "PERPLEXITY_API_KEY", "pplx-test", raising=False)
    monkeypatch.setattr(research_agent_mod.config, "MARKET_INTEL_PERPLEXITY_PRESET", "deep-research", raising=False)
    monkeypatch.setattr("shared.llm_clients.time.sleep", lambda _seconds: None)
    usage_calls: list[dict] = []
    monkeypatch.setattr(
        research_agent_mod,
        "record_llm_usage",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    backend = research_agent_mod.PerplexityResearchBackend(model_name="sonar-test")
    with pytest.raises(RuntimeError, match="pplx_503"):
        backend.collect(
            market_identity=identity,
            previous_artifact=None,
            previous_agent_state=None,
            evidence_batches=[batch],
            planner_result=SimpleNamespace(planner_summary="Need demand context."),
            research_focus=[{"focus": "Employer demand", "priority": "high"}],
        )

    assert attempts == 3
    assert len(usage_calls) == 1
    call = usage_calls[0]
    assert call["provider"] == "perplexity"
    assert call["model"] == "sonar-test"
    assert call["actual_status"] == "error"
    assert call["usage"]["input_tokens"] == 0
    assert call["request"]["error_type"] == "RuntimeError"
    assert call["request"]["error_message"] == "pplx_503"
    assert call["usage_context"]["stage"] == "market_intel_external_research"
    assert call["usage_context"]["attempt"] == "initial"


def test_anthropic_research_backend_records_error_receipt_when_provider_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    identity = MarketIdentity.from_dict(
        {
            "market_key": "forward_deployed_engineer__new_york__ic5_ic6",
            "role_title": "Forward Deployed Engineer",
            "role_level": "IC5-IC6",
            "geography": "New York, New York, United States",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["3000000007"],
            "brief_versions_seen": ["1.3"],
        }
    )
    batch = _make_research_batch(
        run_ref="linkedin:output/runs/linkedin/3000000007/run-2",
        generated_at="2026-04-08T04:43:28.957197+00:00",
        saved=3,
        candidates=20,
    )

    attempts = 0

    class _FailingMessages:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("anthropic_503")

    class _FailingAnthropicClient:
        def __init__(self, **kwargs):
            self.messages = _FailingMessages()

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=_FailingAnthropicClient),
    )
    monkeypatch.setattr(research_agent_mod.config, "ANTHROPIC_API_KEY", "sk-ant-test", raising=False)
    monkeypatch.setattr("shared.llm_clients.time.sleep", lambda _seconds: None)
    usage_calls: list[dict] = []
    monkeypatch.setattr(
        research_agent_mod,
        "record_llm_usage",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    backend = research_agent_mod.AnthropicResearchBackend(
        model_name="claude-research-test",
        max_searches=2,
    )
    with pytest.raises(RuntimeError, match="anthropic_503"):
        backend.collect(
            market_identity=identity,
            previous_artifact=None,
            previous_agent_state=None,
            evidence_batches=[batch],
            planner_result=SimpleNamespace(planner_summary="Need demand context."),
            research_focus=[{"focus": "Employer demand", "priority": "high"}],
        )

    assert attempts == 3
    assert len(usage_calls) == 1
    call = usage_calls[0]
    assert call["provider"] == "anthropic"
    assert call["model"] == "claude-research-test"
    assert call["actual_status"] == "error"
    assert call["usage"]["input_tokens"] == 0
    assert call["request"]["error_type"] == "RuntimeError"
    assert call["request"]["error_message"] == "anthropic_503"
    assert call["usage_context"]["stage"] == "anthropic_research"
    assert call["usage_context"]["turn_index"] == 0


def test_perplexity_backend_packages_edge_case_research_into_external_research_result(
    monkeypatch: pytest.MonkeyPatch,
):
    identity = MarketIdentity.from_dict(
        {
            "market_key": "forward_deployed_engineer__new_york__ic5_ic6",
            "role_title": "Forward Deployed Engineer",
            "role_level": "IC5-IC6",
            "geography": "New York, New York, United States",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["3000000007"],
            "brief_versions_seen": ["1.3"],
        }
    )
    batch = MarketEvidenceBatch(
        run_ref="linkedin:output/runs/linkedin/3000000007/run-2",
        source="linkedin",
        output_dir="/tmp/run",
        brief_version="1.3",
        generated_at="2026-04-08T04:43:28.957197+00:00",
        research_context={
            "context_metadata": {"context_quality": "raw_only"},
            "deterministic_snapshot": {"metrics_summary": {"saved": 12}},
            "edge_case_context": {
                "hidden_pool_risk_signals": [
                    {
                        "label": "Coverage gaps suggest hidden supply",
                        "summary": "The run may be too title-narrow.",
                        "supporting_run_refs": ["linkedin:output/runs/linkedin/3000000007/run-2"],
                    }
                ]
            },
            "report_analysis": {},
        },
    )
    calls: dict[str, object] = {}

    class _FakePerplexityResponse:
        output_text = json.dumps(
            {
                "inferred_research_questions": [
                    {
                        "question": "Are relevant candidates self-labeling under solutions-oriented titles?",
                        "priority": "high",
                        "why_it_matters": "The sourcing run may be missing a hidden submarket.",
                        "sourcing_trigger": "Coverage gaps plus edge-case novelty suggested title fragmentation.",
                        "status": "answered",
                        "supporting_run_refs": [batch.run_ref],
                        "evidence_refs": ["https://example.com/edge-case"],
                    }
                ],
                "edge_case_submarkets": [
                    {
                        "label": "Solutions / deployment builders",
                        "summary": "A hidden pool of customer-embedded builders appears relevant.",
                        "why_it_is_easy_to_miss": "These profiles rarely use explicit FDE titles.",
                        "confidence": 0.68,
                        "supporting_run_refs": [batch.run_ref],
                        "evidence_refs": ["https://example.com/edge-case"],
                    }
                ],
                "title_to_archetype_mapping": [
                    {
                        "title_family": "Solutions Engineer",
                        "likely_archetype": "Customer-embedded platform builder",
                        "caveats": "Some titles skew pre-sales; validate implementation depth.",
                        "supporting_run_refs": [batch.run_ref],
                        "evidence_refs": ["https://example.com/edge-case"],
                    }
                ],
                "self_presentation_patterns": [
                    {
                        "label": "Customer-facing implementation language",
                        "pattern": "Candidates emphasize deployment and implementation rather than FDE language.",
                        "why_it_causes_false_negatives": "Title-narrow strings may miss these profiles.",
                        "supporting_run_refs": [batch.run_ref],
                        "evidence_refs": ["https://example.com/edge-case"],
                    }
                ],
                "false_negative_hypotheses": [
                    {
                        "statement": "The run likely missed a hidden pool of solutions/deployment builders.",
                        "why_it_matters": "This could materially understate addressable supply.",
                        "validation_task": "Add one solutions/deployment title-family lane next run.",
                        "confidence": 0.61,
                        "supporting_run_refs": [batch.run_ref],
                        "evidence_refs": ["https://example.com/edge-case"],
                    }
                ],
                "edge_case_sourcing_implications": [
                    {
                        "category": "add_title_family",
                        "priority": "high",
                        "recommendation": "Add Solutions Engineer and Deployment Engineer title families.",
                        "rationale": "These titles appear to hide relevant builders.",
                        "brief_target_field": "additional_search_terms",
                        "suggested_values": ["Solutions Engineer", "Deployment Engineer"],
                        "expected_effect": "Should reduce false negatives caused by title fragmentation.",
                        "supporting_run_refs": [batch.run_ref],
                        "evidence_refs": ["https://example.com/edge-case"],
                    }
                ],
                "open_questions": [
                    {
                        "question": "How often do these hidden pools meet the IC5/IC6 bar?",
                        "priority": "high",
                        "next_step": "Validate seniority and ownership depth in the next run.",
                        "supporting_run_refs": [batch.run_ref],
                        "evidence_refs": ["https://example.com/edge-case"],
                    }
                ],
            }
        )

        def model_dump(self) -> dict:
            return {
                "output": [
                    {
                        "type": "search_results",
                        "results": [
                            {
                                "title": "Edge Case Research",
                                "url": "https://example.com/edge-case",
                            }
                        ],
                    }
                ]
            }

    class _FakeResponses:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return _FakePerplexityResponse()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.responses = _FakeResponses()

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(research_agent_mod.config, "PERPLEXITY_API_KEY", "pplx-test", raising=False)
    monkeypatch.setattr(
        research_agent_mod.config,
        "MARKET_INTEL_PERPLEXITY_PRESET",
        "deep-research",
        raising=False,
    )

    backend = research_agent_mod.PerplexityResearchBackend()
    result = backend.collect(
        market_identity=identity,
        previous_artifact=None,
        previous_agent_state=None,
        evidence_batches=[batch],
        planner_result=SimpleNamespace(planner_summary="Need hidden-pool characterization."),
        research_focus=[
            {
                "focus": "Which title families hide relevant candidates?",
                "priority": "high",
                "reason": "Recover false negatives caused by title fragmentation.",
                "supporting_run_refs": [batch.run_ref],
            }
        ],
        research_mode="edge_case",
        edge_case_reasoning="Coverage gaps and novelty-heavy signal suggest hidden-pool risk.",
    )

    assert calls["kwargs"]["extra_body"]["response_format"]["json_schema"]["name"] == "market_intel_edge_case_research"
    assert result.edge_case_submarkets
    assert result.title_to_archetype_mapping[0]["title_family"] == "Solutions Engineer"
    assert result.edge_case_sourcing_implications[0]["brief_target_field"] == "additional_search_terms"
    assert result.edge_case_open_questions[0]["evidence_refs"] == ["web:https://example.com/edge-case"]


def test_parse_research_json_repairs_truncated_provider_output():
    full = json.dumps(
        {
            "inferred_research_questions": [
                {
                    "question": "Which adjacent titles hide FDE-like talent?",
                    "priority": "high",
                    "why_it_matters": "This may be the main false-negative source.",
                    "sourcing_trigger": "Sparse title-family evidence in the run.",
                    "status": "answered",
                    "supporting_run_refs": ["linkedin:run-1"],
                    "evidence_refs": ["https://example.com/title-variants"],
                }
            ],
            "market_findings": [
                {
                    "kind": "title-variants",
                    "label": "Customer Engineer",
                    "summary": "Several top employers use adjacent customer-engineering titles.",
                    "why_it_matters": "The current title set may be too narrow.",
                    "confidence": 0.72,
                    "supporting_run_refs": ["linkedin:run-1"],
                    "evidence_refs": ["https://example.com/title-variants"],
                }
            ],
            "sourcing_implications": [
                {
                    "category": "add_title_family",
                    "priority": "high",
                    "recommendation": "Add Customer Engineer title variants.",
                    "rationale": "This title family overlaps with modern FDE motions.",
                    "brief_target_field": "title_family",
                    "suggested_values": ["Customer Engineer"],
                    "expected_effect": "Broader coverage of adjacent pools.",
                    "supporting_run_refs": ["linkedin:run-1"],
                    "evidence_refs": ["https://example.com/title-variants"],
                },
                {
                    "category": "relax_boolean",
                    "priority": "medium",
                    "recommendation": "Loosen overly specific infra keywords.",
                    "rationale": "The keyword stack may be hiding good deployment engineers.",
                    "brief_target_field": "keyword_anchors",
                    "suggested_values": ["production deployment", "customer embed"],
                    "expected_effect": "Better recall on real FDE profiles.",
                    "supporting_run_refs": ["linkedin:run-1"],
                    "evidence_refs": ["https://example.com/title-variants"],
                },
            ],
            "open_questions": [
                {
                    "question": "Which employer cluster is most under-covered?",
                    "priority": "high",
                    "next_step": "Target top FDE employers next run.",
                    "supporting_run_refs": ["linkedin:run-1"],
                    "evidence_refs": ["https://example.com/title-variants"],
                }
            ],
        }
    )
    truncated = full[:-180]
    repaired = research_agent_mod._parse_research_json(truncated)

    assert repaired["inferred_research_questions"]
    assert repaired["market_findings"]
    assert repaired["sourcing_implications"]


def test_edge_case_external_research_enriches_artifact_and_technical_appendix(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)

    artifact = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
        planner_backend=_AlwaysEdgeCasePlannerBackend(),
        external_research_backend=_StubEdgeCaseResearchBackend(),
        with_external_research=True,
    )

    assert artifact.talent_pool_intelligence
    assert any(
        "hidden pool" in str(item.get("claim", "")).lower()
        for item in artifact.market_thesis.get("external_context", [])
    )
    assert any(
        "Solutions Engineer" in str(item.get("proposal", ""))
        for item in artifact.brief_recommendations
    )
    assert any(
        "IC5/IC6 bar" in str(item.get("question", ""))
        for item in artifact.open_questions
    )

    technical_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir).with_name(
        "market-intel-technical.md"
    )
    technical = technical_path.read_text()
    assert "Edge-Case Submarkets" in technical
    assert "Why edge-case research triggered".lower() in technical.lower()
    assert "False-Negative Hypotheses" in technical

    research_log = read_jsonl(resolve_market_intel_research_log_path(brief_path, output_dir=run_dir))
    latest = research_log[-1]
    assert latest["edge_case_triggered"] is True
    assert latest["edge_case_submarket_count"] >= 1


def test_extract_perplexity_sources_supports_dict_payload():
    payload = {
        "output": [
            {
                "type": "search_results",
                "results": [
                    {
                        "title": "OpenAI FDE",
                        "url": "https://example.com/openai-fde",
                        "snippet": "OpenAI is hiring FDE profiles.",
                    },
                ],
            },
            {
                "type": "fetch_url_results",
                "contents": [
                    {
                        "title": "FDE market report",
                        "url": "https://example.com/fde-report",
                        "snippet": "Customer-facing AI engineers are active.",
                    }
                ],
            },
        ]
    }

    sources = research_agent_mod._extract_perplexity_sources(payload)

    assert len(sources) == 2
    assert sources[0]["source_id"] == "web:https://example.com/openai-fde"
    assert sources[0]["snippet"] == "OpenAI is hiring FDE profiles."
    assert sources[1]["used_for"] == ["fetch_url"]
    assert sources[1]["snippet"] == "Customer-facing AI engineers are active."


def test_build_artifact_attaches_groundedness_without_optional_brief_fixture():
    identity = MarketIdentity.from_dict(
        {
            "market_key": "forward_deployed_engineer__new_york__ic5_ic6",
            "role_title": "Forward Deployed Engineer",
            "role_level": "IC5-IC6",
            "geography": "New York, New York, United States",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["3000000007"],
            "brief_versions_seen": ["1.3"],
        }
    )
    batch = MarketEvidenceBatch(
        run_ref="linkedin:output/runs/linkedin/3000000007/run-2",
        source="linkedin",
        output_dir="/tmp/run",
        brief_version="1.3",
        generated_at="2026-04-08T04:43:28.957197+00:00",
    )
    deterministic_summary = {
        "freshness": {"generated_at": "2026-04-08T04:43:28.957197+00:00"},
        "evidence_index": {
            "runs": [
                {
                    "run_ref": batch.run_ref,
                    "source": batch.source,
                    "output_dir": batch.output_dir,
                    "brief_version": batch.brief_version,
                    "generated_at": batch.generated_at,
                }
            ]
        },
        "aggregate_metrics": {"saved_count": 3},
        "channel_summaries": {},
        "lane_intelligence": [
            {
                "lane_key": "fde-lane",
                "domain_lane": "operator",
                "novelty_bucket": "core",
                "status": "winning",
                "first_seen_at": batch.generated_at,
                "last_seen_at": batch.generated_at,
                "supporting_run_refs": [batch.run_ref],
                "metrics": {
                    "strings_seen": 1,
                    "candidates_seen": 3,
                    "saves": 2,
                    "facial_yes": 2,
                    "facial_no": 1,
                    "duplicates": 0,
                    "save_rate": 0.6667,
                    "duplicate_rate": 0.0,
                },
                "dominant_anchors": ["FDE profiles"],
            }
        ],
        "candidate_signal_summary": {},
    }
    generated_sections = {
        "market_thesis": {
            "summary": "External research found employer demand.",
            "supply_assessment": "moderate",
            "competition_assessment": "high",
            "external_context": [
                {
                    "claim": "Example employer is hiring relevant FDE profiles",
                    "evidence_refs": ["web:https://example.com/jobs"],
                    "confidence": 0.8,
                }
            ],
        },
        "lane_intelligence": [
            {
                "lane_key": "fde-lane",
                "label": "FDE profiles",
                "thesis": "Example employer is hiring relevant FDE profiles.",
                "why_it_worked": "Example employer is hiring relevant FDE profiles.",
                "recommended_action": "Keep searching for relevant FDE profiles.",
                "supporting_run_refs": [batch.run_ref],
                "evidence_refs": ["web:https://example.com/jobs"],
            }
        ],
        "talent_pool_intelligence": [
            {
                "pool_key": "fde-profiles",
                "label": "FDE profiles",
                "status": "core_pool",
                "signal_strength": 0.8,
                "evidence_summary": "Example employer is hiring relevant FDE profiles",
                "evidence_refs": ["web:https://example.com/jobs"],
            }
        ],
        "noise_patterns": [
            {
                "pattern_key": "irrelevant-title-noise",
                "label": "Irrelevant title noise",
                "severity": "medium",
                "mitigations": ["Use evidence-backed FDE profile language."],
                "supporting_run_refs": [batch.run_ref],
                "evidence_refs": ["web:https://example.com/jobs"],
            }
        ],
        "employer_signal_intelligence": [
            {
                "cluster_key": "example-employer",
                "label": "Example employer",
                "status": "active",
                "supporting_employers": ["Example employer"],
                "signal_strength": 0.8,
                "evidence_summary": "Example employer is hiring relevant FDE profiles.",
                "supporting_run_refs": [batch.run_ref],
                "evidence_refs": ["web:https://example.com/jobs"],
            }
        ],
        "brief_recommendations": [
            {
                "recommendation_id": "add-fde-profile",
                "target_field": "additional_search_terms",
                "proposal": "Add FDE profile language.",
                "reason": "Example employer is hiring relevant FDE profiles.",
                "confidence": 0.8,
                "evidence_refs": ["web:https://example.com/jobs"],
            }
        ],
        "open_questions": [
            {
                "question": "Should the search prioritize Mars volcano sourcing?",
                "priority": "medium",
                "next_step": "Validate whether Mars volcano profiles are representative.",
                "evidence_refs": ["web:https://example.com/jobs"],
            }
        ],
    }
    artifact = engine_mod._build_artifact(
        brief=SimpleNamespace(
            role_title="Forward Deployed Engineer",
            retrieval_design={},
            raw={},
            search_priorities=[],
            additional_search_terms=[],
        ),
        market_identity=identity,
        deterministic_summary=deterministic_summary,
        evidence_batches=[batch],
        previous_artifact=None,
        generated_sections=generated_sections,
        preserve_previous_narrative=False,
        external_result=ExternalResearchResult(
            sources=[
                {
                    "source_id": "web:https://example.com/jobs",
                    "kind": "web_search",
                    "title": "Example employer jobs",
                    "url": "https://example.com/jobs",
                    "snippet": "Example employer is hiring relevant FDE profiles.",
                    "retrieved_at": "2026-04-08T14:00:00+00:00",
                    "used_for": ["market_thesis"],
                }
            ]
        ),
        section_generation_metadata={},
        delta_since_last_run={},
    )

    context = artifact.market_thesis["external_context"][0]
    assert context["typed_evidence_refs"][0]["source_type"] == "web_search"
    assert context["groundedness"]["status"] == "grounded"
    recommendation = artifact.brief_recommendations[0]
    assert recommendation["groundedness"]["status"] in {"grounded", "partial"}
    report = artifact.evidence_index["groundedness"]
    assert report["status"] in {"ok", "quarantine"}
    assert any(
        claim["claim_id"].startswith("market_thesis.external_context:")
        for claim in report["claims"]
    )
    for section_name, items in {
        "lane_intelligence": artifact.lane_intelligence,
        "talent_pool_intelligence": artifact.talent_pool_intelligence,
        "noise_patterns": artifact.noise_patterns,
        "employer_signal_intelligence": artifact.employer_signal_intelligence,
        "brief_recommendations": artifact.brief_recommendations,
        "open_questions": artifact.open_questions,
    }.items():
        assert items, section_name
        assert "typed_evidence_refs" in items[0], section_name
        assert "groundedness" in items[0], section_name
        assert any(
            claim["claim_id"].startswith(f"{section_name}:")
            for claim in report["claims"]
        ), section_name
    assert any(
        claim["claim_id"].startswith("open_questions:")
        for claim in report["quarantined_claims"]
    )


def test_external_result_normalization_maps_provider_aliases():
    parsed = {
        "inferred_research_questions": [
            {
                "question": "Which titles are adjacent to FDE?",
                "priority": "high",
                "why_it_matters": "Title mismatch may be the main blind spot.",
                "sourcing_trigger": "Sparse title-family evidence in the run.",
                "status": "answered",
                "supporting_run_refs": ["linkedin:run-1"],
                "evidence_refs": ["https://example.com/title-variants"],
            }
        ],
        "market_findings": [
            {
                "kind": "title-variants",
                "label": "Customer Engineer",
                "summary": "Adjacent title family used by relevant employers.",
                "why_it_matters": "Should expand recall.",
                "confidence": 0.7,
                "supporting_run_refs": ["linkedin:run-1"],
                "evidence_refs": ["https://example.com/title-variants"],
            },
            {
                "kind": "employer-clusters",
                "label": "OpenAI / Scale AI / Ramp",
                "summary": "NYC FDE employers cluster around frontier-deployment orgs.",
                "why_it_matters": "Employer-targeted strings should improve yield.",
                "confidence": 0.78,
                "supporting_run_refs": ["linkedin:run-1"],
                "evidence_refs": ["https://example.com/employers"],
            },
        ],
        "sourcing_implications": [
            {
                "category": "add_title_family",
                "priority": "high",
                "recommendation": "Add Customer Engineer variants.",
                "rationale": "This title family overlaps with FDE work.",
                "brief_target_field": "title_family",
                "suggested_values": ["Customer Engineer"],
                "expected_effect": "Improves title coverage.",
                "supporting_run_refs": ["linkedin:run-1"],
                "evidence_refs": ["https://example.com/title-variants"],
            },
            {
                "category": "add_employer_targets",
                "priority": "high",
                "recommendation": "Target frontier deployment employers directly.",
                "rationale": "Public hiring demand is concentrated in a few employer clusters.",
                "brief_target_field": "employer_targets",
                "suggested_values": ["OpenAI", "Scale AI", "Ramp"],
                "expected_effect": "Improves employer clustering and precision.",
                "supporting_run_refs": ["linkedin:run-1"],
                "evidence_refs": ["https://example.com/employers"],
            },
            {
                "category": "relax_boolean",
                "priority": "medium",
                "recommendation": "Relax the keyword stack.",
                "rationale": "Current boolean may be too narrow.",
                "brief_target_field": "keyword_anchors",
                "suggested_values": ["customer embed", "production deployment"],
                "expected_effect": "Improves recall.",
                "supporting_run_refs": ["linkedin:run-1"],
                "evidence_refs": ["https://example.com/title-variants"],
            },
        ],
        "open_questions": [],
    }
    sources = [
        {
            "source_id": "web:https://example.com/title-variants",
            "url": "https://example.com/title-variants",
            "title": "Title variants",
            "kind": "web_search",
            "retrieved_at": "2026-04-10T00:00:00+00:00",
            "used_for": ["web_search"],
        },
        {
            "source_id": "web:https://example.com/employers",
            "url": "https://example.com/employers",
            "title": "Employers",
            "kind": "web_search",
            "retrieved_at": "2026-04-10T00:00:00+00:00",
            "used_for": ["web_search"],
        },
    ]

    result = research_agent_mod._build_external_research_result(
        parsed=parsed,
        sources=sources,
        default_supporting_run_refs=["linkedin:run-1"],
    )

    assert result.market_findings[0]["kind"] == "title_variant"
    assert result.market_findings[1]["kind"] == "employer_cluster"
    assert result.sourcing_implications[0]["brief_target_field"] == "additional_search_terms"
    assert result.sourcing_implications[1]["category"] == "add_employer_target"
    assert result.sourcing_implications[1]["brief_target_field"] == "employer_signal_rules"
    assert result.sourcing_implications[2]["brief_target_field"] == "additional_search_terms"


def test_perplexity_backend_retries_after_truncated_initial_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    identity = MarketIdentity.from_dict(
        {
            "market_key": "forward_deployed_engineer__new_york__ic5_ic6",
            "role_title": "Forward Deployed Engineer",
            "role_level": "IC5-IC6",
            "geography": "New York, New York, United States",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["3000000007"],
            "brief_versions_seen": ["1.3"],
        }
    )
    batch = MarketEvidenceBatch(
        run_ref="linkedin:output/runs/linkedin/3000000007/run-2",
        source="linkedin",
        output_dir="/tmp/run",
        brief_version="1.3",
        generated_at="2026-04-08T04:43:28.957197+00:00",
        research_context={
            "context_metadata": {"context_quality": "raw_only"},
            "deterministic_snapshot": {"metrics_summary": {"saved": 34}},
            "report_analysis": {},
        },
    )

    valid_payload = json.dumps(
        {
            "inferred_research_questions": [
                {
                    "question": "Which title variants are most important?",
                    "priority": "high",
                    "why_it_matters": "Title mismatch may be the main false-negative source.",
                    "sourcing_trigger": "Sparse title-family evidence in the run.",
                    "status": "answered",
                    "supporting_run_refs": [batch.run_ref],
                    "evidence_refs": ["https://example.com/title-variants"],
                }
            ],
            "market_findings": [
                {
                    "kind": "title-variants",
                    "label": "Customer Engineer",
                    "summary": "Relevant title family used by adjacent employers.",
                    "why_it_matters": "Broadens title coverage for FDE-like talent.",
                    "confidence": 0.76,
                    "supporting_run_refs": [batch.run_ref],
                    "evidence_refs": ["https://example.com/title-variants"],
                }
            ],
            "sourcing_implications": [
                {
                    "category": "add_title_family",
                    "priority": "high",
                    "recommendation": "Add Customer Engineer variants.",
                    "rationale": "This title family overlaps with FDE work.",
                    "brief_target_field": "title_family",
                    "suggested_values": ["Customer Engineer"],
                    "expected_effect": "Improves title coverage.",
                    "supporting_run_refs": [batch.run_ref],
                    "evidence_refs": ["https://example.com/title-variants"],
                }
            ],
            "open_questions": [],
        }
    )

    class _FakePerplexityResponse:
        def __init__(self, output_text: str):
            self.output_text = output_text

        def model_dump(self) -> dict:
            return {
                "output": [
                    {
                        "type": "search_results",
                        "results": [
                            {
                                "title": "Title variants",
                                "url": "https://example.com/title-variants",
                            }
                        ],
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": self.output_text,
                            }
                        ],
                    },
                ]
            }

    calls = {"count": 0}

    class _FakeResponses:
        def create(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return _FakePerplexityResponse(valid_payload[:-90])
            return _FakePerplexityResponse(valid_payload)

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = _FakeResponses()

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(research_agent_mod.config, "PERPLEXITY_API_KEY", "pplx-test", raising=False)
    monkeypatch.setattr(research_agent_mod.config, "MARKET_INTEL_PERPLEXITY_PRESET", "deep-research", raising=False)
    monkeypatch.setattr(research_agent_mod.config, "PROJECT_ROOT", tmp_path, raising=False)

    backend = research_agent_mod.PerplexityResearchBackend()
    result = backend.collect(
        market_identity=identity,
        previous_artifact=None,
        previous_agent_state=None,
        evidence_batches=[batch],
        planner_result=SimpleNamespace(planner_summary="Need title-variant context."),
        research_focus=[
            {
                "focus": "Title variants",
                "priority": "high",
                "reason": "Likely false-negative source.",
                "supporting_run_refs": [batch.run_ref],
            }
        ],
    )

    assert calls["count"] == 2
    assert result.market_findings
    assert result.sourcing_implications[0]["brief_target_field"] == "additional_search_terms"
    debug_dir = tmp_path / "output" / "debug" / "perplexity_failures"
    assert debug_dir.exists()
    assert list(debug_dir.glob("*.json"))


def test_external_research_result_is_packaged_then_applied_to_artifact(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)

    artifact = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
        planner_backend=_AlwaysResearchPlannerBackend(),
        external_research_backend=_StubResearchBackend(),
        with_external_research=True,
    )

    assert artifact.evidence_index["external_sources"]
    assert artifact.market_thesis["external_context"]
    assert artifact.market_thesis["external_context"][0]["evidence_refs"] == [
        "web:https://example.com/jobs"
    ]
    assert artifact.market_thesis["external_context"][0]["typed_evidence_refs"][0][
        "source_id"
    ] == "web:https://example.com/jobs"
    assert artifact.market_thesis["external_context"][0]["groundedness"]["status"] in {
        "grounded",
        "partial",
        "ungrounded",
    }
    groundedness = artifact.evidence_index["groundedness"]
    assert groundedness["claims"]
    assert "quarantined_claims" in groundedness
    assert artifact.brief_recommendations
    assert artifact.brief_recommendations[0]["groundedness"]["claim_id"].startswith(
        "brief_recommendations:"
    )
    assert artifact.talent_pool_intelligence
    assert artifact.talent_pool_intelligence[0]["typed_evidence_refs"]
    assert artifact.employer_signal_intelligence
    assert (resolve_market_intel_artifact_path(brief_path, output_dir=run_dir).parent / "market-intel-technical.md").exists()


def test_update_market_intel_force_external_research_overrides_planner_skip(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)

    class _NeverResearchPlannerBackend:
        def plan(self, **_: object):
            return agent_backends_mod.PlannerResult(
                planner_summary="Planner chose not to research externally."
            )

    artifact = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
        planner_backend=_NeverResearchPlannerBackend(),
        external_research_backend=_StubResearchBackend(),
        force_external_research=True,
    )

    assert artifact.evidence_index["external_sources"]
    assert artifact.market_thesis["external_context"]


def test_update_market_intel_auto_builds_external_backend_when_available(tmp_path, monkeypatch):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)

    monkeypatch.setattr(
        "market_intelligence.engine._maybe_build_external_research_backend",
        lambda: _StubResearchBackend(),
    )

    artifact = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
        planner_backend=_AlwaysResearchPlannerBackend(),
        with_external_research=True,
    )

    assert artifact.evidence_index["external_sources"]


def test_update_market_intel_writes_stage_token_cost_log(tmp_path, monkeypatch):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)

    monkeypatch.setattr("market_intelligence.agent_backends._has_llm_access", lambda: True)

    def _fake_opus(
        system_prompt: str,
        user_prompt: str,
        expect_json: bool = True,
        max_tokens: int = 8192,
        usage_context: dict | None = None,
    ):
        record_llm_usage(
            provider="anthropic",
            model="claude-opus-4-6",
            usage={"input_tokens": 1000, "output_tokens": 200},
            request={"system_prompt_chars": len(system_prompt), "user_prompt_chars": len(user_prompt)},
            usage_context=usage_context,
        )
        stage = (usage_context or {}).get("stage")
        if stage == "market_intel_planner":
            return {
                "planner_summary": "Planner completed.",
                "active_hypotheses": [],
                "resolved_hypotheses": [],
                "open_unknowns": [],
                "research_backlog": [],
                "update_sections": ["market_thesis"],
                "confidence_ceiling_by_section": {"market_thesis": 0.7},
                "should_collect_external_research": False,
                "external_research_focus": [],
                "should_collect_edge_case_research": False,
                "edge_case_research_reasoning": "",
                "edge_case_research_focus": [],
            }
        if stage == "market_intel_synthesis":
            return {
                "lane_intelligence": [],
                "talent_pool_intelligence": [],
                "noise_patterns": [],
                "employer_signal_intelligence": [],
                "market_thesis": {
                    "summary": "Internal synthesis identified a concentrated winning lane.",
                    "supply_assessment": "moderate",
                    "competition_assessment": "medium",
                    "external_context": [],
                },
                "brief_recommendations": [],
                "open_questions": [],
            }
        if stage == "market_intel_critic":
            return {
                "planner_summary": "Critic completed.",
                "keep_sections": {
                    "market_thesis": {
                        "summary": "Internal synthesis identified a concentrated winning lane.",
                        "supply_assessment": "moderate",
                        "competition_assessment": "medium",
                        "external_context": [],
                    },
                    "lane_intelligence": [],
                    "talent_pool_intelligence": [],
                    "noise_patterns": [],
                    "employer_signal_intelligence": [],
                    "brief_recommendations": [],
                    "open_questions": [],
                },
                "section_generation_metadata": {},
                "delta_since_last_run": {},
                "confidence_by_claim_area": {"market_thesis": 0.7},
            }
        raise AssertionError(f"Unexpected stage: {stage}")

    monkeypatch.setattr("market_intelligence.agent_backends.opus_llm", _fake_opus)

    artifact = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
        with_external_research=False,
    )

    usage_log = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir).parent / "token-cost-log.jsonl"
    records = read_jsonl(usage_log)
    assert artifact.market_thesis["summary"].startswith("Internal synthesis identified")
    assert len(records) == 3
    assert {record["stage"] for record in records} == {
        "market_intel_planner",
        "market_intel_synthesis",
        "market_intel_critic",
    }


def test_internal_only_refresh_preserves_prior_external_provenance(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    output_dir = tmp_path / "output"
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_linkedin_inputs(
        output_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    run_dir = _import_run_dir(brief_path=brief_path, legacy_output_dir=output_dir)

    update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="post_run",
        planner_backend=_AlwaysResearchPlannerBackend(),
        external_research_backend=_StubResearchBackend(),
        with_external_research=True,
    )
    refreshed = update_market_intel(
        brief_path=brief_path,
        run_dir=run_dir,
        mode="scheduled",
        with_external_research=False,
    )

    assert refreshed.evidence_index["external_sources"]
    assert refreshed.market_thesis["external_context"]


def test_normalize_critic_result_recovers_invalid_keep_sections_from_fallback():
    evidence_batches = [
        MarketEvidenceBatch(
            run_ref="linkedin:output/runs/example",
            source="linkedin",
            output_dir="output/runs/example",
            brief_version="1.0",
            generated_at="2026-04-08T12:00:00+00:00",
            metrics_summary={"run_count": 1, "saved": 4, "candidate_volume": 20},
        )
    ]
    draft_sections = {
        "lane_intelligence": [],
        "talent_pool_intelligence": [
            {
                "pool_key": "deployment_engineer_pool",
                "label": "Deployment Engineer",
                "status": "adjacent_pool",
                "signal_strength": 0.62,
                "supporting_run_refs": ["linkedin:output/runs/example"],
                "evidence_refs": ["web:https://example.com/deployment"],
                "evidence_summary": "Relevant adjacent title family surfaced in external research.",
                "recommended_search_terms": ["Deployment Engineer"],
            }
        ],
        "noise_patterns": [],
        "employer_signal_intelligence": [
            {
                "cluster_key": "openai",
                "label": "OpenAI",
                "status": "positive",
                "supporting_employers": ["OpenAI"],
                "supporting_run_refs": ["linkedin:output/runs/example"],
                "evidence_refs": ["web:https://example.com/openai-role"],
                "evidence_summary": "OpenAI is hiring this role family.",
                "confidence": 0.68,
            }
        ],
        "market_thesis": {
            "summary": "External research confirms title fragmentation and supports targeted employer expansion.",
            "supply_assessment": "unknown_leaning_moderate",
            "competition_assessment": "medium",
            "external_context": [
                {
                    "claim": "OpenAI and Scale both hire adjacent deployment-oriented titles.",
                    "evidence_refs": ["web:https://example.com/openai-role"],
                    "confidence": 0.63,
                }
            ],
        },
        "brief_recommendations": [
            {
                "recommendation_id": "rec-title-variants",
                "target_field": "additional_search_terms",
                "proposal": "Technical Deployment Lead, Deployment Engineer",
                "reason": "External research surfaced adjacent title families.",
                "supporting_run_refs": ["linkedin:output/runs/example"],
                "evidence_refs": ["web:https://example.com/openai-role"],
                "confidence": 0.7,
            }
        ],
        "open_questions": [
            {
                "question": "Do these adjacent titles convert into true IC5-IC6 builder profiles?",
                "priority": "high",
                "next_step": "Validate against extracted saved profiles in the next run.",
                "supporting_run_refs": ["linkedin:output/runs/example"],
                "evidence_refs": ["web:https://example.com/openai-role"],
            }
        ],
    }
    raw = {
        "planner_summary": "Critic attempted a refinement pass.",
        "keep_sections": {
            "market_thesis": {
                "action": "keep_summary_with_edits",
                "summary_assessment": "Looks directionally correct.",
                "confidence_override": 0.4,
                "note": "Needs cleanup.",
                "summary": "",
                "supply_assessment": "unknown",
                "competition_assessment": "unknown",
                "external_context": [],
            },
            "talent_pool_intelligence": [
                {
                    "label": "Deployment Engineer",
                    "evidence_summary": "Missing canonical pool key so this record sanitizes away.",
                    "supporting_run_refs": ["linkedin:output/runs/example"],
                    "evidence_refs": ["web:https://example.com/deployment"],
                }
            ],
            "brief_recommendations": [
                {
                    "proposal": "Technical Deployment Lead",
                    "reason": "Missing canonical recommendation fields so this sanitizes away.",
                    "supporting_run_refs": ["linkedin:output/runs/example"],
                    "evidence_refs": ["web:https://example.com/openai-role"],
                }
            ],
            "open_questions": [],
        },
        "section_generation_metadata": {
            "market_thesis": {
                "generation_mode": "llm_external",
                "quality_level": "medium",
                "updated_at": "2026-04-08T12:00:00+00:00",
                "notes": ["Malformed critic payload."],
                "supporting_run_refs": ["linkedin:output/runs/example"],
                "evidence_refs": [],
            }
        },
    }

    result = agent_backends_mod._normalize_critic_result(
        raw,
        evidence_batches,
        previous_artifact=None,
        planner_result=agent_backends_mod.PlannerResult(),
        draft_sections=draft_sections,
        external_research=None,
    )

    assert result.keep_sections["market_thesis"]["summary"].startswith(
        "External research confirms title fragmentation"
    )
    assert "action" not in result.keep_sections["market_thesis"]
    # P3.7: the critic response IS structurally valid (keep_sections is a
    # non-empty object), so sections whose entries sanitize away are DROPPED,
    # not resurrected from the fallback — the critic's drop decision holds.
    assert result.keep_sections["talent_pool_intelligence"] == []
    assert result.keep_sections["brief_recommendations"] == []


def test_normalize_critic_result_structurally_invalid_restores_fallback():
    """P3.7: only a structurally invalid critic response (keep_sections
    missing/non-object) falls back wholesale."""

    evidence_batches = [
        MarketEvidenceBatch(
            run_ref="linkedin:output/runs/example",
            source="linkedin",
            output_dir="output/runs/example",
            brief_version="1.0",
            generated_at="2026-04-08T12:00:00+00:00",
            metrics_summary={"run_count": 1, "saved": 4, "candidate_volume": 20},
        )
    ]
    draft_sections = {
        "lane_intelligence": [],
        "talent_pool_intelligence": [
            {
                "pool_key": "deployment_engineer_pool",
                "label": "Deployment Engineer",
                "status": "adjacent_pool",
                "signal_strength": 0.62,
                "supporting_run_refs": ["linkedin:output/runs/example"],
                "evidence_refs": ["web:https://example.com/deployment"],
                "evidence_summary": "Relevant adjacent title family.",
                "recommended_search_terms": ["Deployment Engineer"],
            }
        ],
        "noise_patterns": [],
        "employer_signal_intelligence": [],
        "market_thesis": {},
        "brief_recommendations": [],
        "open_questions": [],
    }
    result = agent_backends_mod._normalize_critic_result(
        {"planner_summary": "no keep_sections at all"},
        evidence_batches,
        previous_artifact=None,
        planner_result=agent_backends_mod.PlannerResult(),
        draft_sections=draft_sections,
        external_research=None,
    )
    # Fallback restored the draft's talent pool section.
    assert result.keep_sections["talent_pool_intelligence"]


def test_normalize_critic_result_persists_claim_adjudications():
    """The critic's per-claim audit trail is parsed onto CriticResult: well-formed
    entries (non-empty claim + holds in {yes,weaken,drop}) are kept with holds
    lowercased; malformed entries (empty claim, bad verdict, non-dict) are dropped.
    FAILS if the parser/persistence is reverted."""
    evidence_batches = [
        MarketEvidenceBatch(
            run_ref="linkedin:output/runs/example",
            source="linkedin",
            output_dir="output/runs/example",
            brief_version="1.0",
            generated_at="2026-04-08T12:00:00+00:00",
            metrics_summary={"run_count": 1, "saved": 4, "candidate_volume": 20},
        )
    ]
    raw = {
        "planner_summary": "Critic pass.",
        "claim_adjudications": [
            {"claim": "supply is dense", "section": "market_thesis",
             "evidence": "linkedin:output/runs/example", "holds": "YES",
             "why": "save rate high across two runs"},
            {"claim": "OpenAI cluster is hot", "section": "employer_signal_intelligence",
             "evidence": "web:https://example.com", "holds": "drop",
             "why": "single weak anecdote"},
            {"claim": "", "holds": "yes"},                 # empty claim -> dropped
            {"claim": "bogus verdict", "holds": "maybe"},   # holds not in set -> dropped
            "not a dict",                                   # non-dict -> dropped
        ],
        "keep_sections": {},
    }

    result = agent_backends_mod._normalize_critic_result(
        raw,
        evidence_batches,
        previous_artifact=None,
        planner_result=agent_backends_mod.PlannerResult(),
        draft_sections={},
        external_research=None,
    )

    adj = result.claim_adjudications
    assert [a["claim"] for a in adj] == ["supply is dense", "OpenAI cluster is hot"]
    assert adj[0]["holds"] == "yes"  # normalized from "YES"
    assert adj[1]["holds"] == "drop"
    assert all({"claim", "section", "evidence", "holds", "why"} <= set(a) for a in adj)
    assert result.to_dict()["claim_adjudications"] == adj


def test_normalize_critic_result_preserves_draft_external_context_when_critic_drops_it():
    evidence_batches = [
        MarketEvidenceBatch(
            run_ref="linkedin:output/runs/example",
            source="linkedin",
            output_dir="output/runs/example",
            brief_version="1.0",
            generated_at="2026-04-08T12:00:00+00:00",
            metrics_summary={"run_count": 1, "saved": 4, "candidate_volume": 20},
        )
    ]
    draft_sections = {
        "market_thesis": {
            "summary": "Broad builder-entry lanes are surfacing overlooked deployment-oriented candidates.",
            "supply_assessment": "moderate",
            "competition_assessment": "medium",
            "external_context": [
                {
                    "claim": "OpenAI and Scale both hire adjacent deployment-oriented titles.",
                    "supporting_run_refs": ["linkedin:output/runs/example"],
                    "evidence_refs": ["web:https://example.com/openai-role"],
                    "confidence": 0.63,
                }
            ],
        },
        "brief_recommendations": [],
        "open_questions": [],
    }
    raw = {
        "keep_sections": {
            "market_thesis": {
                "summary": "Broad builder-entry lanes are surfacing overlooked deployment-oriented candidates.",
                "supply_assessment": "unknown",
                "competition_assessment": "unknown",
                "external_context": [],
            }
        }
    }

    result = agent_backends_mod._normalize_critic_result(
        raw,
        evidence_batches,
        previous_artifact=None,
        planner_result=agent_backends_mod.PlannerResult(),
        draft_sections=draft_sections,
        external_research=None,
    )

    assert result.keep_sections["market_thesis"]["summary"].startswith(
        "Broad builder-entry lanes are surfacing"
    )
    assert result.keep_sections["market_thesis"]["supply_assessment"] == "moderate"
    assert result.keep_sections["market_thesis"]["competition_assessment"] == "medium"
    assert result.keep_sections["market_thesis"]["external_context"]
    assert result.keep_sections["market_thesis"]["external_context"][0]["evidence_refs"] == [
        "web:https://example.com/openai-role"
    ]


def test_merge_market_thesis_rejects_review_meta_summary_and_keeps_previous_context():
    merged = engine_mod._merge_market_thesis(
        current={
            "summary": (
                "The market thesis summary is well-written and directionally correct. "
                "However, it needs the following adjustments: KEEP the first claim."
            ),
            "supply_assessment": "unknown",
            "competition_assessment": "unknown",
            "external_context": [],
        },
        previous={
            "summary": "The market is accessible but structurally under-explored by title-first search.",
            "supply_assessment": "moderate",
            "competition_assessment": "medium",
            "external_context": [
                {
                    "claim": "Adjacent deployment titles cluster around AI platform and customer-delivery teams.",
                    "supporting_run_refs": ["linkedin:output/runs/example"],
                    "evidence_refs": ["web:https://example.com/deployment"],
                    "confidence": 0.61,
                }
            ],
        },
        preserve_previous=False,
    )

    assert merged["summary"].startswith("The market is accessible")
    assert merged["supply_assessment"] == "moderate"
    assert merged["competition_assessment"] == "medium"
    assert merged["external_context"]


def test_live_advisory_sidecar_writes_checkpoint_files_and_context(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path,
        state_dir=tmp_path / "output" / "state" / "linkedin" / "3000000006",
    )
    write_json(brief_path, read_json(_require_source_brief_path()))
    _write_jsonl(
        state_dir / "final_judgments.jsonl",
        [
            {
                "decision": "SAVE",
                "candidate_name": "Saved Candidate",
                "path": "DIRECT",
                "rationale": "Strong workflow builder evidence.",
            }
        ],
    )
    _write_jsonl(
        state_dir / "run_log.jsonl",
        [
            {
                "timestamp": "2026-04-08T12:00:00+00:00",
                "event": "block_adaptation",
                "message": "Continue pagination",
            }
        ],
    )
    agent_state_path = resolve_market_intel_agent_state_path(brief_path, output_dir=state_dir)
    write_json(
        agent_state_path,
        {
            "schema_version": 1,
            "market_key": "head_of_applied_ai_lab__new_york_new_york__l8_l9",
            "updated_at": "2026-04-08T12:00:00+00:00",
            "active_hypotheses": [
                {
                    "hypothesis_id": "hyp-research-copilot",
                    "statement": "research_copilot_asset_mgmt is a productive lane",
                    "status": "active",
                    "confidence": 0.8,
                    "rationale": "Prior runs support it",
                    "section_targets": ["lane_intelligence"],
                    "first_seen_at": "2026-04-08T12:00:00+00:00",
                    "last_seen_at": "2026-04-08T12:00:00+00:00",
                    "supporting_run_refs": ["linkedin:prior-run"],
                }
            ],
            "resolved_hypotheses": [],
            "open_unknowns": [],
            "research_backlog": [],
            "source_registry": [],
            "confidence_by_claim_area": {},
            "prior_advisories": [],
            "section_generation_metadata": {},
        },
    )

    context = record_page_checkpoint_and_get_context(
        brief_path=brief_path,
        state_dir=state_dir,
        brief_id="3000000006",
        search_string_family_key="research_copilot_asset_mgmt",
        string_stats={"pages": 1, "saves": 1, "facial_yes": 1, "facial_no": 2},
        recent_candidates=[{"rationale": "Research copilot workflow builder"}],
        glance_summary=None,
        architecture="sniper",
    )
    block_context = record_block_checkpoint_and_get_context(
        brief_path=brief_path,
        state_dir=state_dir,
        brief_id="3000000006",
        block_name="Opening",
        block_report=SimpleNamespace(
            strings_run=2,
            strings_with_saves=1,
            total_saves=1,
            top_performers=[{"family_key": "research_copilot_asset_mgmt"}],
            zero_save_string_ids=[3],
        ),
        search_memory_summary={"families": []},
    )

    live_dir = live_market_intel_dir(state_dir)
    assert context
    assert block_context
    assert (live_dir / "live-observations.jsonl").exists()
    assert (live_dir / "live-advisories.jsonl").exists()
    assert (live_dir / "live-summary.json").exists()
    assert read_jsonl(live_dir / "live-advisories.jsonl")


def test_finalize_run_snapshot_copies_live_market_intel_sidecar(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    write_json(brief_path, read_json(_require_source_brief_path()))
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path,
        state_dir=tmp_path / "output" / "state" / "linkedin" / "3000000006",
    )
    _write_linkedin_inputs(
        state_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
    )
    live_dir = live_market_intel_dir(state_dir)
    write_json(live_dir / "live-summary.json", {"checkpoint_index": 2, "last_context": "ctx"})
    _write_jsonl(
        live_dir / "live-advisories.jsonl",
        [
            {
                "advisory_id": "adv-1",
                "scope": "block",
                "kind": "exploit",
                "rationale": "Lean into top lane.",
                "confidence": 0.7,
                "created_at": "2026-04-08T12:00:00+00:00",
                "expires_at_checkpoint": 3,
                "checkpoint_key": "block:opening:2",
                "supporting_run_refs": ["linkedin:prior-run"],
            }
        ],
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="3000000006",
        output_dir=str(state_dir),
        mode="full_run",
        resume_state={"brief_name": "head-ai-lab"},
    )
    store.finish_run(run_id, status="completed")

    run_dir = finalize_run_snapshot(
        source="linkedin",
        brief_path=brief_path,
        state_dir=state_dir,
        run_id=run_id,
    )

    assert (run_dir / "market_intel" / "live-summary.json").exists()
    assert (run_dir / "market_intel" / "live-advisories.jsonl").exists()


def test_heuristic_planner_promotes_repeated_signal_and_gates_external_research():
    planner = HeuristicPlannerBackend()
    identity = MarketIntelArtifact.from_dict(
        {
            "schema_version": 1,
            "artifact_version": 1,
            "market_identity": {
                "market_key": "head_applied_ai_lab__new_york__director",
                "role_title": "Head of Applied AI Lab",
                "role_level": "Director",
                "geography": "New York",
                "channels_seen": ["linkedin"],
                "brief_ids_seen": ["head-ai-lab"],
                "brief_versions_seen": ["2.1"],
            },
            "freshness": {
                "artifact_updated_at": "2026-04-08T13:00:00+00:00",
                "internal_data_through": "2026-04-08T13:00:00+00:00",
                "external_research_through": "",
                "staleness_days": 0,
            },
            "evidence_index": {"runs": [], "external_sources": []},
            "aggregate_metrics": {
                "run_count": 0,
                "saved_count": 0,
                "rejected_count": 0,
                "facial_yes_rate": 0.0,
                "save_rate": 0.0,
                "candidate_volume_by_channel": {},
            },
            "channel_summaries": {},
            "lane_intelligence": [],
            "talent_pool_intelligence": [],
            "noise_patterns": [],
            "employer_signal_intelligence": [],
            "candidate_signal_summary": {
                "standout_signals": [],
                "borderline_signals": [],
                "disqualifying_signals": [],
            },
            "market_thesis": {
                "summary": "",
                "supply_assessment": "moderate",
                "competition_assessment": "high",
                "external_context": [],
            },
            "brief_recommendations": [],
            "open_questions": [],
        }
    ).market_identity
    batches = [
        _make_research_batch(
            run_ref="linkedin:output/run-a",
            generated_at="2026-04-08T10:00:00+00:00",
            saved=5,
            candidates=100,
        ),
        _make_research_batch(
            run_ref="linkedin:output/run-b",
            generated_at="2026-04-08T11:00:00+00:00",
            saved=9,
            candidates=120,
        ),
    ]
    deterministic_summary = {
        "aggregate_metrics": {
            "run_count": 2,
            "saved_count": 14,
            "candidate_volume_by_channel": {"linkedin": 220},
        }
    }

    result = planner.plan(
        market_identity=identity,
        deterministic_summary=deterministic_summary,
        evidence_batches=batches,
        previous_artifact=None,
        previous_agent_state=None,
    )

    assert result.active_hypotheses
    assert result.should_collect_external_research is True
    assert result.external_research_focus


def test_adapt_after_block_renders_market_intel_as_typed_prior(monkeypatch):
    captured: dict[str, str] = {}

    def _fake_opus(
        system_prompt: str,
        user_prompt: str,
        expect_json: bool = True,
        max_tokens: int = 8192,
        usage_context: dict | None = None,
    ):
        captured["system"] = system_prompt
        return {
            "new_strings": [],
            "skip_remaining": [],
            "reorder": [],
            "noise_updates": [],
            "pivot_to_architecture": "",
            "pivot_rationale": "",
        }

    monkeypatch.setattr("linkedin.strategy.opus_llm", _fake_opus)
    brief = load_brief(_require_source_brief_path())
    block_report = BlockReport(
        block_name="Opening",
        strings_run=2,
        strings_with_saves=1,
        total_results=100,
        total_saves=1,
        top_performers=[{"string_id": 2, "name": "Research copilot", "saves": 1}],
        zero_save_string_ids=[3],
        string_details=[],
    )
    adapt_after_block(
        brief,
        block_report,
        remaining_strings=[SearchString(id=10, name="Queued", boolean="queued")],
        checkpoint_mode="normal_block_checkpoint",
        market_intel_advisory_context="## Market Intel Advisory Context\n- [exploit] Lean into this lane.",
    )

    assert "## Typed MarketSignalPrior" in captured["system"]
    assert "## Market Intel Advisory Context" not in captured["system"]
    assert '"signal_type": "exploit"' in captured["system"]
    assert "Lean into this lane." in captured["system"]


# ---------------------------------------------------------------------------
# Behavior-first / anti-employer-proxy guardrails on engine recommendations.
#
# These tests pin that lane and noise evidence drive brief recommendations
# while employer-inventory text is demoted: the engine still surfaces the
# information for narrative context, but it loses opening-retrieval primacy
# and is reframed as late-stage classifier signal targeting
# employer_signal_rules instead of search_priorities/additional_search_terms.
# Doctrine: docs/brief-authoring-guide.md ("Titles, employers, and keywords
# can support the pattern, but they should not be the pattern.")
# ---------------------------------------------------------------------------


def _make_synthesis_inputs(*, brief_iteration_hints: dict) -> dict:
    """Build the minimal pieces HeuristicMarketIntelSynthesisBackend needs.

    Returns a dict keyed for direct use by ``synthesize`` and the small set
    of inputs the recommendation-emission path reads. Lane intelligence is
    populated so the lane-narrative path stays the primary surface even
    while we exercise employer-inventory hints.
    """
    deterministic_summary = {
        "lane_intelligence": [
            {
                "lane_key": "research_copilot",
                "lane_label": "Research copilot",
                "supporting_run_refs": ["run-1"],
                "dominant_anchors": ["research copilot", "knowledge worker copilots"],
                "evidence_summary": "Highest absolute save count.",
                "metrics": {"saves": 8, "candidates": 40},
                "status": "live",
            }
        ],
        "aggregate_metrics": {"save_rate": 0.12, "saved_count": 14},
        "channel_summaries": {},
    }
    report = {
        "winning_lanes": [
            {
                "lane": "Research copilot",
                "string_ids": [2],
                "candidate_examples": ["Mithun Azhagappan"],
                "evidence": "Highest absolute save count.",
                "why_it_worked": "Specific workflow language.",
                "recommended_action": "Promote this lane earlier.",
            }
        ],
        "underperforming_lanes": [],
        "coverage_gaps": [],
        "noise_patterns": [
            {
                "pattern": "Product-heavy AI leadership",
                "evidence": "Several product leaders failed builder bar.",
                "mitigation": "Strengthen builder-authorship language.",
            }
        ],
        "saved_candidate_patterns": {
            "archetype_distribution": [
                {"archetype": "BFSI-native GenAI converts", "count": 6}
            ],
            "common_employers": [
                {"employer": "JPMorgan", "count": 3, "note": "Bank GenAI converts."}
            ],
        },
        "brief_iteration_hints": brief_iteration_hints,
        "string_performance": [
            {"string_id": 2, "name": "Research copilot lane"}
        ],
    }
    market_identity = MarketIdentity(
        market_key="head_of_applied_ai_lab__new_york_new_york_united_states__ic6",
        role_title="Head of Applied AI Lab",
        role_level="ic6",
        geography="New York, New York, United States",
        channels_seen=["linkedin"],
        brief_ids_seen=["head-ai-lab"],
        brief_versions_seen=["1.0"],
    )
    batch = MarketEvidenceBatch(
        run_ref="run-1",
        source="linkedin",
        output_dir="/tmp/snapshot",
        brief_version="1.0",
        generated_at="2026-04-25T00:00:00+00:00",
        report=report,
        research_context={"deterministic_snapshot": deterministic_summary},
        runtime_summary={"saves": 14, "candidates": 120},
    )
    return {
        "market_identity": market_identity,
        "deterministic_summary": deterministic_summary,
        "evidence_batches": [batch],
    }


def _synthesize(inputs: dict) -> dict:
    backend = engine_mod.HeuristicMarketIntelSynthesisBackend()
    return backend.synthesize(
        market_identity=inputs["market_identity"],
        deterministic_summary=inputs["deterministic_summary"],
        evidence_batches=inputs["evidence_batches"],
        previous_artifact=None,
        planner_result=None,
        external_research=None,
    )


def test_engine_keeps_lane_intelligence_primary_when_hints_are_lane_shaped():
    """Lane-shaped brief_iteration_hints stay first-class opening retrieval.

    This is the control case: when search_priorities and
    additional_search_terms are behavioral / lane-anchored, no demotion
    should fire. confidence stays high, target_field stays as the search
    fields, proposal_kind reads ``opening_retrieval``.
    """
    inputs = _make_synthesis_inputs(
        brief_iteration_hints={
            "search_priorities": [
                "Payments / transaction-banking builders",
            ],
            "additional_search_terms": [
                "payment orchestration",
                "transaction banking",
            ],
        }
    )
    result = _synthesize(inputs)
    recs_by_id = {rec["recommendation_id"]: rec for rec in result["brief_recommendations"]}
    priorities_rec = recs_by_id["rec-search-priorities"]
    terms_rec = recs_by_id["rec-additional-search-terms"]

    assert priorities_rec["target_field"] == "search_priorities"
    assert priorities_rec["proposal_kind"] == "opening_retrieval"
    assert priorities_rec["confidence"] >= 0.7
    assert "retrieval_update" in priorities_rec

    assert terms_rec["target_field"] == "additional_search_terms"
    assert terms_rec["proposal_kind"] == "opening_retrieval"
    assert terms_rec["confidence"] >= 0.7
    assert "retrieval_update" in terms_rec

    # Noise patterns remain a primary synthesis surface alongside lanes.
    assert result["noise_patterns"], "Noise patterns must remain primary surface"
    assert any(
        "Product-heavy" in pattern["label"]
        for pattern in result["noise_patterns"]
    )


def test_engine_demotes_employer_cluster_priorities_to_late_stage_gap_probe():
    """Employer-inventory search_priorities are demoted, not opening retrieval.

    When a run hint is dominated by employer-cluster prose, the engine
    must reframe the recommendation as a late-stage gap probe targeting
    employer_signal_rules with reduced confidence, not as opening
    retrieval. The retrieval_update layer is dropped on demoted items so
    employer findings cannot drive entry signals.
    """
    inputs = _make_synthesis_inputs(
        brief_iteration_hints={
            "search_priorities": [
                "Buy-side AI lab heads at BlackRock, Bridgewater, Citadel, Two Sigma, Point72, AQR",
            ],
            "additional_search_terms": [
                "behavioral build evidence: production agent platforms",
            ],
        }
    )
    result = _synthesize(inputs)
    recs_by_id = {rec["recommendation_id"]: rec for rec in result["brief_recommendations"]}
    priorities_rec = recs_by_id["rec-search-priorities"]

    assert priorities_rec["target_field"] == "employer_signal_rules"
    assert priorities_rec["proposal_kind"] == "late_stage_gap_probe"
    assert priorities_rec["confidence"] < 0.6
    assert "retrieval_update" not in priorities_rec
    assert "anti-employer-proxy rule" in priorities_rec["reason"]


def test_engine_demotes_employer_inventory_terms_to_classifier_signal():
    """Employer-inventory additional_search_terms get the same demotion.

    Even when search_priorities are clean, employer-inventory terms must
    be redirected from opening retrieval to employer_signal_rules.
    """
    inputs = _make_synthesis_inputs(
        brief_iteration_hints={
            "search_priorities": [
                "Capital markets AI builders with executive-builder scope",
            ],
            "additional_search_terms": [
                "company-first anchors: BlackRock, Bridgewater, Citadel, Two Sigma, Point72, AQR",
            ],
        }
    )
    result = _synthesize(inputs)
    recs_by_id = {rec["recommendation_id"]: rec for rec in result["brief_recommendations"]}
    priorities_rec = recs_by_id["rec-search-priorities"]
    terms_rec = recs_by_id["rec-additional-search-terms"]

    # Priorities are still behavioral, so they keep opening retrieval.
    assert priorities_rec["target_field"] == "search_priorities"
    assert priorities_rec["proposal_kind"] == "opening_retrieval"

    # Terms are employer-inventory, so they are demoted.
    assert terms_rec["target_field"] == "employer_signal_rules"
    assert terms_rec["proposal_kind"] == "late_stage_gap_probe"
    assert terms_rec["confidence"] < 0.6
    assert "retrieval_update" not in terms_rec


def test_engine_employer_findings_remain_secondary_classifier():
    """common_employers data still flows to employer_signal_intelligence.

    Even after the demotion of employer-inventory hints, the engine must
    continue to surface common_employers as employer_signal_intelligence
    so operators retain visibility for late-stage classification. The
    point is to keep them out of opening retrieval, not to delete them.
    """
    inputs = _make_synthesis_inputs(
        brief_iteration_hints={
            "search_priorities": [
                "JPMorgan, Goldman, Morgan Stanley, Citi, Barclays, HSBC, UBS, Deutsche Bank engineers",
            ],
            "additional_search_terms": [],
        }
    )
    result = _synthesize(inputs)

    # The literal cluster recommendation is demoted...
    recs_by_id = {rec["recommendation_id"]: rec for rec in result["brief_recommendations"]}
    priorities_rec = recs_by_id["rec-search-priorities"]
    assert priorities_rec["proposal_kind"] == "late_stage_gap_probe"

    # ...but employer findings still appear in employer_signal_intelligence.
    assert any(
        signal["label"].lower() == "jpmorgan"
        for signal in result["employer_signal_intelligence"]
    )


def test_lane_metrics_aggregator_is_importable_in_market_intel_context():
    """P5: ``shared.runtime_state.lane_metrics`` lives alongside (not
    inside) the market-intelligence package. P11 will fold the
    aggregator into market-intel input; this slice only proves the
    read primitive is reachable from the MI test surface without
    forcing an engine integration today.
    """

    from shared.runtime_state.lane_metrics import (
        LEGACY_LANE_ID,
        LaneMetricsRow,
        lane_metrics_for_run,
    )

    assert callable(lane_metrics_for_run)
    assert LEGACY_LANE_ID == "legacy"
    # Smoke a zero-state row construction so the dataclass keeps a
    # backward-compatible default shape (P11 will read these fields).
    row = LaneMetricsRow(lane_id=LEGACY_LANE_ID)
    assert row.legacy is False  # default; aggregator sets True when materialized
    assert row.review_by_reason == {}
    assert row.work_unit_source_ids == ()


# --- ADVERSARIAL VERIFIER: run-scoped report-ownership edge cases ------------
#
# Added by the adversarial verifier for the B8 success-masquerade scrub.
# These attack the time-window boundaries, the dormant run_id branch, the
# different-brief gate, clock-skew, and the unknown-window degenerate case
# that the implementer's three tests do not cover.


def _adv_state_dir(tmp_path):
    brief_path = tmp_path / "brief.json"
    write_json(brief_path, read_json(HOA_BRIEF_FIXTURE))
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path,
        state_dir=tmp_path / "output" / "state" / "linkedin" / "haai",
    )
    return brief_path, state_dir


def _adv_set_window(store, run_id, *, started_at, ended_at, status="completed"):
    with store.connect() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, started_at = ?, ended_at = ? WHERE id = ?",
            (status, started_at, ended_at, run_id),
        )


def test_adv_report_generated_exactly_at_started_at_is_kept(tmp_path):
    """Boundary: generated_at == started_at is INCLUSIVE (kept)."""
    brief_path, state_dir = _adv_state_dir(tmp_path)
    _write_owned_report_artifacts(
        state_dir, generated_at="2026-04-08T11:00:00+00:00", saved=12
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin", brief_id="haai",
        output_dir=str(state_dir), mode="full_run", resume_state={},
    )
    _adv_set_window(
        store, run_id,
        started_at="2026-04-08T11:00:00+00:00",
        ended_at="2026-04-08T13:00:00+00:00",
    )
    run_dir = finalize_run_snapshot(
        source="linkedin", brief_path=brief_path, state_dir=state_dir, run_id=run_id,
    )
    manifest = read_json(run_dir / "run-manifest.json")
    assert "run-report.json" in manifest["artifacts_present"]
    assert (run_dir / "run-report.json").exists()


def test_adv_report_generated_exactly_at_ended_at_is_kept(tmp_path):
    """Boundary: generated_at == ended_at is INCLUSIVE (kept)."""
    brief_path, state_dir = _adv_state_dir(tmp_path)
    _write_owned_report_artifacts(
        state_dir, generated_at="2026-04-08T13:00:00+00:00", saved=12
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin", brief_id="haai",
        output_dir=str(state_dir), mode="full_run", resume_state={},
    )
    _adv_set_window(
        store, run_id,
        started_at="2026-04-08T11:00:00+00:00",
        ended_at="2026-04-08T13:00:00+00:00",
    )
    run_dir = finalize_run_snapshot(
        source="linkedin", brief_path=brief_path, state_dir=state_dir, run_id=run_id,
    )
    manifest = read_json(run_dir / "run-manifest.json")
    assert "run-report.json" in manifest["artifacts_present"]
    assert (run_dir / "run-report.json").exists()


def test_adv_report_one_second_after_ended_at_is_scrubbed(tmp_path):
    """Just-outside boundary: generated_at one second past ended_at is scrubbed."""
    brief_path, state_dir = _adv_state_dir(tmp_path)
    _write_owned_report_artifacts(
        state_dir, generated_at="2026-04-08T13:00:01+00:00", saved=12
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin", brief_id="haai",
        output_dir=str(state_dir), mode="full_run", resume_state={},
    )
    _adv_set_window(
        store, run_id,
        started_at="2026-04-08T11:00:00+00:00",
        ended_at="2026-04-08T13:00:00+00:00",
    )
    run_dir = finalize_run_snapshot(
        source="linkedin", brief_path=brief_path, state_dir=state_dir, run_id=run_id,
    )
    manifest = read_json(run_dir / "run-manifest.json")
    assert "run-report.json" not in manifest["artifacts_present"]
    assert not (run_dir / "run-report.json").exists()


def test_adv_report_run_id_matches_current_run_is_kept_even_when_out_of_window(tmp_path):
    """The dormant Phase-1 branch: matching run_id wins over the time window.

    A report stamped with run_metadata.run_id == the current run id must be
    kept even if its generated_at lies OUTSIDE the run window — run_id is the
    authoritative ownership signal and short-circuits the time fallback.
    """
    brief_path, state_dir = _adv_state_dir(tmp_path)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin", brief_id="haai",
        output_dir=str(state_dir), mode="full_run", resume_state={},
    )
    # generated_at deliberately OUTSIDE the window, but run_id matches.
    _write_owned_report_artifacts(
        state_dir,
        generated_at="2020-01-01T00:00:00+00:00",
        saved=12,
        run_id=run_id,
    )
    _adv_set_window(
        store, run_id,
        started_at="2026-04-08T11:00:00+00:00",
        ended_at="2026-04-08T13:00:00+00:00",
    )
    run_dir = finalize_run_snapshot(
        source="linkedin", brief_path=brief_path, state_dir=state_dir, run_id=run_id,
    )
    manifest = read_json(run_dir / "run-manifest.json")
    assert "run-report.json" in manifest["artifacts_present"]
    assert (run_dir / "run-report.json").exists()


def test_adv_report_run_id_mismatch_is_scrubbed_even_when_in_window(tmp_path):
    """Inverse of the dormant branch: a DIFFERENT run_id is scrubbed.

    A report carrying run_metadata.run_id of a *prior* run must be scrubbed
    even when its generated_at happens to fall inside the current window —
    the run_id mismatch is decisive and the time fallback never runs.
    """
    brief_path, state_dir = _adv_state_dir(tmp_path)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin", brief_id="haai",
        output_dir=str(state_dir), mode="full_run", resume_state={},
    )
    # In-window generated_at, but stamped with a prior run's id (run_id - 1).
    _write_owned_report_artifacts(
        state_dir,
        generated_at="2026-04-08T12:00:00+00:00",
        saved=12,
        run_id=run_id - 1,
    )
    _adv_set_window(
        store, run_id,
        started_at="2026-04-08T11:00:00+00:00",
        ended_at="2026-04-08T13:00:00+00:00",
    )
    run_dir = finalize_run_snapshot(
        source="linkedin", brief_path=brief_path, state_dir=state_dir, run_id=run_id,
    )
    manifest = read_json(run_dir / "run-manifest.json")
    assert "run-report.json" not in manifest["artifacts_present"]
    assert not (run_dir / "run-report.json").exists()


def test_adv_different_brief_report_still_scrubbed_by_brief_identity(tmp_path):
    """Brief-identity gate still works: an in-window report for a DIFFERENT
    brief is scrubbed regardless of the time window."""
    brief_path, state_dir = _adv_state_dir(tmp_path)
    state_dir.mkdir(parents=True, exist_ok=True)
    # Report whose run_metadata identity does NOT match the HOA fixture.
    foreign_meta = {
        "role_title": "Staff Frontend Engineer",
        "brief_name": "some-other-brief",
        "linkedin_project_id": "999999",
        "generated_at": "2026-04-08T12:00:00+00:00",
        "overall_summary": "Different brief entirely.",
    }
    write_json(
        state_dir / "run-report.json",
        {"schema_version": 1, "run_metadata": foreign_meta,
         "metrics_summary": {"saved": 12}},
    )
    _write_jsonl(
        state_dir / "final_judgments.jsonl",
        [{"stage": "full", "decision": "SAVE", "path": "DIRECT:x",
          "profile_url": "https://example.com/x", "confidence": 0.8,
          "rationale": "r", "candidate_name": "X"}],
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin", brief_id="haai",
        output_dir=str(state_dir), mode="full_run", resume_state={},
    )
    _adv_set_window(
        store, run_id,
        started_at="2026-04-08T11:00:00+00:00",
        ended_at="2026-04-08T13:00:00+00:00",
    )
    run_dir = finalize_run_snapshot(
        source="linkedin", brief_path=brief_path, state_dir=state_dir, run_id=run_id,
    )
    manifest = read_json(run_dir / "run-manifest.json")
    assert "run-report.json" not in manifest["artifacts_present"]
    assert not (run_dir / "run-report.json").exists()


def test_adv_unknown_window_keeps_brief_matched_report(tmp_path):
    """Degenerate case: when the run window is unparseable, the time fallback
    cannot make a negative judgment, so a brief-matched report is KEPT
    (pre-fix behavior preserved for the no-window case)."""
    brief_path, state_dir = _adv_state_dir(tmp_path)
    _write_owned_report_artifacts(
        state_dir, generated_at="2026-01-01T00:00:00+00:00", saved=12
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin", brief_id="haai",
        output_dir=str(state_dir), mode="full_run", resume_state={},
    )
    # Blank out BOTH window bounds: no parseable started_at/ended_at.
    _adv_set_window(store, run_id, started_at="", ended_at="")
    run_dir = finalize_run_snapshot(
        source="linkedin", brief_path=brief_path, state_dir=state_dir, run_id=run_id,
    )
    manifest = read_json(run_dir / "run-manifest.json")
    assert "run-report.json" in manifest["artifacts_present"]
    assert (run_dir / "run-report.json").exists()


def test_adv_clock_skew_report_far_future_within_no_end_window_is_kept(tmp_path):
    """Clock-skew sanity: a half-open window (started_at set, ended_at blank)
    keeps a report generated AFTER started_at even far in the future — the
    end bound is absent so only the lower bound constrains ownership."""
    brief_path, state_dir = _adv_state_dir(tmp_path)
    _write_owned_report_artifacts(
        state_dir, generated_at="2030-12-31T23:59:59+00:00", saved=12
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin", brief_id="haai",
        output_dir=str(state_dir), mode="full_run", resume_state={},
    )
    _adv_set_window(
        store, run_id,
        started_at="2026-04-08T11:00:00+00:00",
        ended_at="",
    )
    run_dir = finalize_run_snapshot(
        source="linkedin", brief_path=brief_path, state_dir=state_dir, run_id=run_id,
    )
    manifest = read_json(run_dir / "run-manifest.json")
    assert "run-report.json" in manifest["artifacts_present"]
    assert (run_dir / "run-report.json").exists()


def test_adv_clock_skew_report_before_start_with_no_end_window_is_scrubbed(tmp_path):
    """Clock-skew sanity (negative): half-open window, report BEFORE started_at
    is scrubbed even though ended_at is blank — the lower bound alone rejects
    a report timestamped before the run began."""
    brief_path, state_dir = _adv_state_dir(tmp_path)
    _write_owned_report_artifacts(
        state_dir, generated_at="2026-01-01T00:00:00+00:00", saved=12
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin", brief_id="haai",
        output_dir=str(state_dir), mode="full_run", resume_state={},
    )
    _adv_set_window(
        store, run_id,
        started_at="2026-04-08T11:00:00+00:00",
        ended_at="",
    )
    run_dir = finalize_run_snapshot(
        source="linkedin", brief_path=brief_path, state_dir=state_dir, run_id=run_id,
    )
    manifest = read_json(run_dir / "run-manifest.json")
    assert "run-report.json" not in manifest["artifacts_present"]
    assert not (run_dir / "run-report.json").exists()


# ---------------------------------------------------------------------------
# P4.3.2 — stages_degraded lands on the artifact
# ---------------------------------------------------------------------------


def _minimal_build_artifact_fixture_kwargs():
    identity = MarketIdentity.from_dict(
        {
            "market_key": "test_market__remote__ic5",
            "role_title": "Test Role",
            "role_level": "IC5",
            "geography": "Remote",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["brief-1"],
            "brief_versions_seen": ["1.0"],
        }
    )
    batch = MarketEvidenceBatch(
        run_ref="linkedin:output/runs/linkedin/brief-1/run-1",
        source="linkedin",
        output_dir="/tmp/run",
        brief_version="1.0",
        generated_at="2026-07-02T00:00:00+00:00",
    )
    deterministic_summary = {
        "freshness": {"generated_at": batch.generated_at},
        "evidence_index": {
            "runs": [
                {
                    "run_ref": batch.run_ref,
                    "source": batch.source,
                    "output_dir": batch.output_dir,
                    "brief_version": batch.brief_version,
                    "generated_at": batch.generated_at,
                }
            ]
        },
        "aggregate_metrics": {},
        "channel_summaries": {},
        "lane_intelligence": [],
        "candidate_signal_summary": {},
    }
    return dict(
        brief=SimpleNamespace(
            role_title="Test Role",
            retrieval_design={},
            raw={},
            search_priorities=[],
            additional_search_terms=[],
        ),
        market_identity=identity,
        deterministic_summary=deterministic_summary,
        evidence_batches=[batch],
        previous_artifact=None,
        generated_sections={},
        preserve_previous_narrative=False,
        external_result=None,
        section_generation_metadata={},
        delta_since_last_run={},
    )


def test_build_artifact_populates_stages_degraded_from_stage_errors():
    artifact = engine_mod._build_artifact(
        **_minimal_build_artifact_fixture_kwargs(),
        stage_errors=["planner:boom", "critic:boom too", "planner:another one"],
    )

    assert artifact.stages_degraded == ["critic", "planner"]

    # Round-trips through to_dict/from_dict — the persisted artifact.json shape.
    reloaded = MarketIntelArtifact.from_dict(artifact.to_dict())
    assert reloaded.stages_degraded == ["critic", "planner"]


def test_build_artifact_stages_degraded_empty_when_no_stage_errors():
    artifact = engine_mod._build_artifact(**_minimal_build_artifact_fixture_kwargs())

    assert artifact.stages_degraded == []


# ---------------------------------------------------------------------------
# P4.5 — adaptation ROI passthrough into the market-intel packet
# ---------------------------------------------------------------------------


def test_extract_metrics_summary_passes_through_adaptation_roi():
    report = {
        "metrics_summary": {
            "candidates_evaluated": 10,
            "facial_yes": 4,
            "facial_no": 6,
            "saved": 2,
            "rejected": 1,
            "adaptation_roi": {"status": "ok", "net_saves_gained": 3},
        }
    }

    summary = engine_mod._extract_metrics_summary(report, {}, [])

    assert summary["adaptation_roi"] == {"status": "ok", "net_saves_gained": 3}


def test_extract_metrics_summary_adaptation_roi_defaults_to_empty_dict_when_absent():
    report = {
        "metrics_summary": {
            "candidates_evaluated": 1,
            "facial_yes": 1,
            "facial_no": 0,
            "saved": 1,
            "rejected": 0,
        }
    }

    summary = engine_mod._extract_metrics_summary(report, {}, [])

    assert summary["adaptation_roi"] == {}
