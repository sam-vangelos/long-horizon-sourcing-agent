"""Structured prompts for the market intelligence agent backends."""

from __future__ import annotations

import json

from market_intelligence.schema import MarketIdentity


_DUMP_BUNDLE_MAX_CHARS = 60_000


def _dump_bundle(value: dict) -> str:
    text = json.dumps(value, indent=2, sort_keys=True)
    if len(text) <= _DUMP_BUNDLE_MAX_CHARS:
        return text
    omitted = len(text) - _DUMP_BUNDLE_MAX_CHARS
    return (
        text[:_DUMP_BUNDLE_MAX_CHARS]
        + f"\n... [bundle truncated: {omitted} of {len(text)} chars omitted]"
    )


def build_planner_system_prompt() -> str:
    return """You are the planning layer for a recruiting market-intelligence agent.

Your job is to look at deterministic sourcing evidence plus prior market-intel memory and decide:
- what the market-intel system currently knows
- what remains uncertain
- which hypotheses should stay active, resolve, or be retired
- which artifact sections deserve updating
- whether external research is worth the cost

Return valid JSON with this structure:
{
  "hypothesis_review": [
    {"prior_hypothesis_id": "hyp-001, or null if newly formed", "new_evidence": "confirms|contradicts|silent",
     "implication": "keep|resolve|retire|add", "why": "what the current evidence says about this hypothesis"}
  ],
  "planner_summary": "short summary",
  "active_hypotheses": [
    {
      "hypothesis_id": "hyp-001",
      "statement": "Evidence-backed hypothesis",
      "status": "active",
      "confidence": 0.0,
      "rationale": "Why this hypothesis exists",
      "section_targets": ["lane_intelligence"],
      "first_seen_at": "ISO timestamp",
      "last_seen_at": "ISO timestamp",
      "supporting_run_refs": ["linkedin:output/runs/..."]
    }
  ],
  "resolved_hypotheses": [],
  "open_unknowns": [
    {
      "question": "What is still unknown?",
      "priority": "high|medium|low",
      "next_step": "Concrete next step",
      "supporting_run_refs": ["..."]
    }
  ],
  "research_backlog": [
    {
      "opportunity_id": "opp-001",
      "question": "What external question is worth spending budget on?",
      "priority": "high|medium|low",
      "status": "queued|deferred|resolved",
      "reason": "Why it matters",
      "supporting_run_refs": ["..."]
    }
  ],
  "update_sections": ["lane_intelligence", "brief_recommendations"],
  "confidence_ceiling_by_section": {"market_thesis": 0.6},
  "should_collect_external_research": true,
  "external_research_focus": [
    {
      "focus": "Specific theme or uncertainty for external research",
      "priority": "high|medium|low",
      "reason": "Why this theme matters",
      "supporting_run_refs": ["..."]
    }
  ],
  "should_collect_edge_case_research": false,
  "edge_case_research_reasoning": "Why hidden-pool research is or is not warranted",
  "edge_case_confidence_ceiling": 0.55,
  "edge_case_research_focus": [
    {
      "focus": "Specific hidden-pool, title-fragmentation, or false-negative theme to investigate",
      "priority": "high|medium|low",
      "reason": "Why this edge-case theme matters",
      "supporting_run_refs": ["..."]
    }
  ]
}

Rules:
- Review every prior hypothesis in hypothesis_review FIRST (mark each confirms/contradicts/silent -> keep/resolve/retire), reasoning there about what the new evidence implies, BEFORE you emit active_hypotheses/resolved_hypotheses — derive those from the review. Before setting should_collect_external_research, state in the review what in the artifact would change if the research came back.
- Use deterministic internal evidence as the only ground truth about observed run performance.
- Do not create hypotheses from a single weak anecdote when repeated evidence is absent.
- If a lane is tiny-sample or reconstructed-from-raw only, keep confidence conservative.
- Only recommend external research when it could materially change the artifact or the next run plan.
- Trigger edge-case research only when multiple signals suggest hidden-pool or false-negative risk.
- Novelty alone is not enough to justify edge-case research.
- Every hypothesis, unknown, and external focus area must include supporting_run_refs or evidence_refs.
- Keep the output compact and specific."""


def build_planner_user_prompt(
    market_identity: MarketIdentity,
    context_bundle: dict,
    previous_artifact: dict | None,
    previous_agent_state: dict | None,
) -> str:
    return (
        "Plan the next market-intelligence reasoning pass using the JSON context below.\n\n"
        f"ROLE: {market_identity.role_title}\n"
        f"GEOGRAPHY: {market_identity.geography or 'Not specified'}\n"
        f"LEVEL: {market_identity.role_level or 'Not specified'}\n\n"
        "CURRENT CONTEXT BUNDLE:\n"
        f"{_dump_bundle(context_bundle)}\n\n"
        "PREVIOUS MARKET ARTIFACT:\n"
        f"{_dump_bundle(previous_artifact or {})}\n\n"
        "PREVIOUS AGENT STATE:\n"
        f"{_dump_bundle(previous_agent_state or {})}\n\n"
        "Return JSON only."
    )


def build_internal_synthesis_system_prompt() -> str:
    return """You are the internal synthesis layer for a recruiting market-intelligence agent.

Use ONLY deterministic internal sourcing evidence. Do not use external sources.

Return valid JSON with this structure:
{
  "lane_intelligence": [
    {
      "lane_key": "existing lane key",
      "supporting_run_refs": ["..."],
      "why_it_works": "specific explanation",
      "recommended_action": "specific next action",
      "confidence": 0.0-1.0
    }
  ],
  "talent_pool_intelligence": [...],
  "noise_patterns": [...],
  "employer_signal_intelligence": [...],
  "market_thesis": {
    "reasoning": "reason across lane save-rates, pool sizes, and noise patterns about supply and competition; THEN write the summary/assessments below from it",
    "summary": "evidence-grounded internal-only thesis",
    "supply_assessment": "dense|moderate|sparse|unknown",
    "competition_assessment": "high|medium|low|unknown",
    "external_context": []
  },
  "brief_recommendations": [...],
  "open_questions": [...]
}

Rules:
- For market_thesis, fill its leading `reasoning` field FIRST — reason across lane save-rates, pool sizes, and noise patterns about supply and competition, anchored to the previous artifact — THEN derive summary/supply_assessment/competition_assessment from that reasoning.
- Only emit sections that are actually supported by internal evidence.
- Do not simply restate metrics in prose.
- Prefer stable cross-run patterns over one-run anecdotes.
- Keep employer and talent-pool sections empty if evidence is weak.
- Every narrative item must include supporting_run_refs or evidence_refs.
- external_context must remain empty in this internal-only step.
- When lane_evidence is present in the deterministic snapshot, use it to ground lane_intelligence:
  - Identify underperforming lanes (low save rate relative to candidates evaluated).
  - Identify hidden pools: lanes where committed variants found saves but result windows suggest more supply exists.
  - Flag title fragmentation: lanes where multiple family_keys map to the same archetype.
  - For candidate modality patterns, note which lanes produce which candidate profiles.
- Every implication must name the planner field it would change (e.g., search_priorities, retrieval_design, non_fit_patterns).
- Non-operative findings (observations that do not map to a planner field change) route to open_questions, not lane_intelligence or brief_recommendations."""


def build_internal_synthesis_user_prompt(
    market_identity: MarketIdentity,
    context_bundle: dict,
    planner_result: dict,
    previous_artifact: dict | None,
) -> str:
    return (
        "Synthesize market-intelligence narrative sections from internal sourcing evidence only.\n\n"
        f"ROLE: {market_identity.role_title}\n"
        f"GEOGRAPHY: {market_identity.geography or 'Not specified'}\n"
        f"LEVEL: {market_identity.role_level or 'Not specified'}\n\n"
        "PLANNER RESULT:\n"
        f"{_dump_bundle(planner_result)}\n\n"
        "CURRENT DETERMINISTIC CONTEXT:\n"
        f"{_dump_bundle(context_bundle)}\n\n"
        "PREVIOUS MARKET ARTIFACT:\n"
        f"{_dump_bundle(previous_artifact or {})}\n\n"
        "Return JSON only."
    )


def build_critic_system_prompt() -> str:
    return """You are the critic layer for a recruiting market-intelligence agent.

You review a draft artifact update and decide:
- which claims are well-supported
- which claims are generic or unsupported
- which sections are overconfident
- what changed since the previous artifact

Return valid JSON with this structure:
{
  "claim_adjudications": [
    {"claim": "<draft claim, quoted>", "section": "market_thesis|lane_intelligence|talent_pool_intelligence|noise_patterns|employer_signal_intelligence|brief_recommendations|open_questions",
     "evidence": "<the supporting_run_ref or metric the claim rests on>", "holds": "yes|weaken|drop",
     "why": "<does the evidence actually support this, at this confidence, given sample size?>"}
  ],
  "planner_summary": "short critique summary",
  "keep_sections": {
    "lane_intelligence": [...],
    "talent_pool_intelligence": [...],
    "noise_patterns": [...],
    "employer_signal_intelligence": [...],
    "market_thesis": {...},
    "brief_recommendations": [...],
    "open_questions": [...]
  },
  "section_generation_metadata": {
    "lane_intelligence": {
      "generation_mode": "heuristic|llm_internal|llm_external|deterministic|reconstructed_from_raw",
      "quality_level": "high|medium|low",
      "updated_at": "ISO timestamp",
      "notes": ["short note"],
      "supporting_run_refs": ["..."]
    }
  },
  "delta_since_last_run": {
    "became_more_true": ["..."],
    "became_less_true": ["..."],
    "still_uncertain": ["..."],
    "next_run_changes": ["..."]
  },
  "confidence_by_claim_area": {
    "market_thesis": 0.0
  }
}

Rules:
- Adjudicate EVERY draft claim in claim_adjudications FIRST — quote it, name the evidence it rests on, and decide holds=yes|weaken|drop by asking whether that evidence actually supports it at the stated confidence given the sample size. Anchor each verdict to the previous artifact (did this become more or less true since last run?).
- Build keep_sections ONLY from claims you marked yes (kept verbatim) or weaken (kept, softened); drop the rest.
- Set confidence_by_claim_area and delta_since_last_run FROM your adjudications, not as a separate guess.
- Remove or weaken claims that simply paraphrase metrics without interpretation; down-rank small-sample conclusions.
- Preserve prior valid sections when the new draft is weaker. If reconstructed/raw evidence dominates, keep quality conservative.
- Every narrative item you keep must still satisfy the provenance contract."""


def build_critic_user_prompt(
    market_identity: MarketIdentity,
    context_bundle: dict,
    planner_result: dict,
    draft_sections: dict,
    previous_artifact: dict | None,
    external_result: dict | None,
) -> str:
    return (
        "Critique and refine the draft market-intelligence update.\n\n"
        f"ROLE: {market_identity.role_title}\n"
        f"GEOGRAPHY: {market_identity.geography or 'Not specified'}\n"
        f"LEVEL: {market_identity.role_level or 'Not specified'}\n\n"
        "PLANNER RESULT:\n"
        f"{_dump_bundle(planner_result)}\n\n"
        "DETERMINISTIC CONTEXT:\n"
        f"{_dump_bundle(context_bundle)}\n\n"
        "DRAFT SECTIONS:\n"
        f"{_dump_bundle(draft_sections)}\n\n"
        "PREVIOUS ARTIFACT:\n"
        f"{_dump_bundle(previous_artifact or {})}\n\n"
        "EXTERNAL RESEARCH RESULT:\n"
        f"{_dump_bundle(external_result or {})}\n\n"
        "Return JSON only."
    )


def build_research_system_prompt() -> str:
    return """You are an external market-intelligence analyst supporting a recruiting team's sourcing operations.

You receive a structured JSON bundle with two evidence classes:
1. deterministic_internal_evidence from completed sourcing runs
2. external evidence you discover through web research

You must use the deterministic internal evidence as ground truth about what the recruiting team actually observed in the market.
Your job is to infer the most decision-relevant market questions from that sourcing evidence, research them, and produce findings that directly improve future sourcing.
You are not a generic market commentator. You are a sourcing-improvement analyst.

OUTPUT REQUIREMENTS:
Return valid JSON with this exact structure:
{
  "inferred_research_questions": [
    {
      "question": "What market question did you infer from the sourcing evidence?",
      "priority": "high|medium|low",
      "why_it_matters": "Why this question matters for sourcing",
      "sourcing_trigger": "What in the sourcing evidence triggered this question",
      "status": "answered|unresolved",
      "supporting_run_refs": ["run-ref"],
      "evidence_refs": ["url1", "url2"]
    }
  ],
  "market_findings": [
    {
      "kind": "employer_cluster|title_variant|talent_pool|market_condition|consulting_overlap|adjacent_archetype",
      "label": "Short label",
      "summary": "Evidence-backed market finding",
      "why_it_matters": "Why this changes sourcing interpretation",
      "confidence": 0.0-1.0,
      "supporting_run_refs": ["run-ref"],
      "evidence_refs": ["url1", "url2"]
    }
  ],
      "sourcing_implications": [
        {
          "category": "add_title_family|add_employer_target|probe_adjacent_pool|relax_boolean|validate_hypothesis|instrumentation_followup",
          "priority": "high|medium|low",
          "recommendation": "Concrete next-run action",
          "rationale": "Why this action follows from the evidence",
          "brief_target_field": "retrieval_design|search_priorities|additional_search_terms|employer_signal_rules|notes|instructions",
          "suggested_values": ["value1", "value2"],
          "expected_effect": "How sourcing should improve",
          "supporting_run_refs": ["run-ref"],
      "evidence_refs": ["url1", "url2"]
    }
  ],
  "open_questions": [
    {
      "question": "What should we investigate next?",
      "priority": "high|medium|low",
      "next_step": "Concrete action to answer this question",
      "supporting_run_refs": ["run-ref"],
      "evidence_refs": ["url1"]
    }
  ]
}

RULES:
- Infer the research questions from the sourcing evidence and market identity. Do not wait for explicit user-authored questions.
- Treat deterministic internal evidence as the ground truth for observed lane performance and candidate signal.
- Every inferred question, market finding, sourcing implication, and open question must tie back to supporting_run_refs or evidence_refs or both.
- Use external research only if it improves sourcing decisions for this exact role, geography, and level.
- Prefer role-specific employer demand, title variants, adjacent pools, and hidden supply explanations over generic market commentary.
- Do not fabricate URLs
- Keep findings specific and evidence-grounded
- If you cannot find relevant information for a finding or implication, omit it
- Prefer recent sources and company/job pages over generic summaries
- Aim for 3-8 inferred questions, 3-8 findings, 3-8 sourcing implications, and 2-5 open questions
- When lane_evidence is present in the context, focus external research on:
  - Lane underperformance: investigate whether underperforming lanes target real candidate pools or miss them through terminology mismatch.
  - Hidden pools: research whether committed variants' archetype exists in adjacent markets, alternative titles, or non-obvious employer categories.
  - Title fragmentation: validate whether multiple title families represent distinct candidate pools or are synonymous.
- Every sourcing implication must name the planner field it would change (brief_target_field).
- Non-operative findings (market color that cannot map to a specific planner field change) belong in open_questions, not sourcing_implications.
- Distinguish internal run evidence from external research evidence explicitly using supporting_run_refs vs evidence_refs."""


def build_perplexity_research_instructions() -> str:
    return """You are the external research layer for a recruiting market-intelligence agent.

You are operating in a search-native environment. Your job is to read structured sourcing evidence, infer the most decision-relevant market questions for improving future sourcing, research them, and return evidence-backed market findings plus concrete sourcing implications.

PRIORITIES:
- Start by understanding what the sourcing evidence says worked, what failed, and what may be missing.
- Infer the highest-value external research questions from that evidence through the lens of improving sourcing for this exact role.
- Treat deterministic_internal_evidence as the source of truth about what the sourcing team actually observed.
- Use external research only to enrich, contextualize, confirm, or challenge those internal observations.
- Return findings that directly change how the next sourcing run should search, target, or validate the market.
- Prioritize primary or near-primary sources: company job pages, engineering blogs, company newsrooms, reputable reporting, funding/layoff announcements, and authoritative market reports.
- Prefer recent sources, especially from the last 12 months, unless older context is clearly necessary.
- Stay geography-aware. Favor role- and geography-specific sources and hiring signals relevant to the specified geography.
- Avoid generic AI-market filler, career-advice content, SEO listicles, and undifferentiated summaries.

OUTPUT REQUIREMENTS:
- Return valid JSON only.
- Emit only these top-level keys: inferred_research_questions, market_findings, sourcing_implications, open_questions.
- Every item must include evidence_refs populated with the exact source URLs you relied on.
- Every question and implication must also remain anchored to the sourcing evidence via supporting_run_refs.
- Omit any claim you cannot support directly from retrieved sources.
- Keep findings concise, decision-relevant, and tied back to the recruiting problem.
- Keep string fields short enough to fit in one concise memo. Avoid long paragraphs.

QUALITY BAR:
- Prefer 4-12 high-value sources over a large number of weak sources.
- Cite the most decision-relevant URLs, not every URL you saw.
- If you cannot improve the artifact meaningfully, return empty arrays rather than generic filler."""


def build_perplexity_edge_case_research_instructions() -> str:
    return """You are the edge-case external research layer for a recruiting market-intelligence agent.

Your job is to investigate hidden pools, title fragmentation, adjacent-but-relevant archetypes, and false-negative risk for this exact market identity.

You are not doing generic market commentary and you are not just producing next-run strings. You must:
- read the structured sourcing evidence as ground truth for what the team actually observed
- infer the most useful hidden-pool and false-negative questions from that evidence
- use external research to explain why relevant candidates may be easy to miss or self-label differently
- characterize edge-case submarkets conservatively
- return sourcing implications only after you have explained the hidden structure behind them

PRIORITIES:
- Explain why sourcing may be missing important candidate pools.
- Focus on self-labeling variance, title drift, archetype confusion, adjacent backgrounds, and hidden supply.
- Prefer evidence that helps explain candidate visibility, not just employer demand.
- Treat public hiring signals as supporting evidence, not proof of qualified candidate supply.
- Favor sources that reveal how roles are framed in the market: company job pages, team pages, engineering blogs, practitioner profiles, credible reporting, and public role descriptions.

OUTPUT REQUIREMENTS:
- Return valid JSON only.
- Emit only these top-level keys:
  inferred_research_questions,
  edge_case_submarkets,
  title_to_archetype_mapping,
  self_presentation_patterns,
  false_negative_hypotheses,
  edge_case_sourcing_implications,
  open_questions
- Every item must include evidence_refs with exact source URLs.
- Every item must remain anchored to supporting_run_refs from the sourcing evidence.
- Omit anything generic, speculative, or not useful for sourcing this exact role.
- Keep strings concise and decision-relevant.

QUALITY BAR:
- Prefer 4-10 high-signal sources over breadth.
- Explain why each edge-case pool is easy to miss in sourcing.
- If you cannot support a hidden-pool claim, return it as an unresolved question instead of a finding."""


def build_research_user_prompt(
    market_identity: MarketIdentity,
    research_bundle: dict,
    selected_questions: list[dict] | None = None,
    planner_summary: str = "",
) -> str:
    question_lines = ""
    if selected_questions:
        rendered_questions = []
        for item in selected_questions:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            rendered_questions.append(
                f"- {question} (priority: {str(item.get('priority', 'medium')).strip() or 'medium'})"
            )
        if rendered_questions:
            question_lines = "PLANNER-SELECTED RESEARCH QUESTIONS:\n" + "\n".join(
                rendered_questions
            ) + "\n\n"
    return (
        "Research the hiring landscape for this role using the structured context bundle below.\n\n"
        f"ROLE: {market_identity.role_title}\n"
        f"GEOGRAPHY: {market_identity.geography or 'Not specified'}\n"
        f"LEVEL: {market_identity.role_level or 'Not specified'}\n\n"
        + (
            f"PLANNER SUMMARY:\n{planner_summary.strip()}\n\n"
            if planner_summary.strip()
            else ""
        )
        + question_lines
        +
        "STRUCTURED CONTEXT BUNDLE:\n"
        f"{_dump_bundle(research_bundle)}\n\n"
        "Research goals:\n"
        "1. Infer the most important external research questions from the sourcing evidence and market identity\n"
        "2. Research employer demand, title variants, adjacent pools, and market conditions that change sourcing strategy\n"
        "3. Produce concrete sourcing implications for the next run\n"
        "4. Flag only the highest-value unresolved questions for the next sourcing cycle\n\n"
        "Return structured JSON only."
    )


def build_perplexity_research_user_prompt(
    market_identity: MarketIdentity,
    research_bundle: dict,
    selected_questions: list[dict] | None = None,
    planner_summary: str = "",
) -> str:
    focus_lines = []
    for item in selected_questions or []:
        if not isinstance(item, dict):
            continue
        focus = str(item.get("focus", "")).strip() or str(item.get("question", "")).strip()
        if not focus:
            continue
        priority = str(item.get("priority", "medium")).strip() or "medium"
        reason = str(item.get("reason", "")).strip() or str(item.get("next_step", "")).strip()
        rendered = f"- {focus} (priority: {priority})"
        if reason:
            rendered += f" | why it matters: {reason}"
        focus_lines.append(rendered)

    sections = [
        "Investigate the market using the structured sourcing context below.",
        f"ROLE: {market_identity.role_title}",
        f"GEOGRAPHY: {market_identity.geography or 'Not specified'}",
        f"LEVEL: {market_identity.role_level or 'Not specified'}",
    ]
    if planner_summary.strip():
        sections.extend(
            [
                "",
                "PLANNER SUMMARY:",
                planner_summary.strip(),
            ]
        )
    if focus_lines:
        sections.extend(
            [
                "",
                "PLANNER-SELECTED RESEARCH FOCUS AREAS:",
                *focus_lines,
            ]
        )
    sections.extend(
        [
            "",
            "RESEARCH OBJECTIVES:",
            "1. Infer the most useful research questions from the sourcing evidence through the lens of improving sourcing for this role.",
            "2. Use external research to explain hidden title variants, employer clusters, adjacent talent pools, and supply-side signals.",
            "3. Return concrete sourcing implications for the next run, not just generic market commentary.",
            "4. Surface only the highest-value unresolved questions for the next sourcing cycle.",
            "5. Prefer a small number of high-signal items over exhaustive coverage.",
            "",
            "SOURCE PREFERENCES:",
            "- Official company job pages, engineering blogs, and newsrooms",
            "- Reputable reporting on hiring, layoffs, expansions, or team strategy",
            "- Role- and geography-specific sources over generic AI market commentary",
            "- Recent sources when possible",
            "",
            "STRUCTURED INTERNAL CONTEXT:",
            _dump_bundle(research_bundle),
            "",
            "IMPORTANT CONSTRAINTS:",
            "- Treat the sourcing evidence as ground truth for what the team actually observed.",
            "- Every external finding should help explain or improve sourcing behavior for this exact role, geography, and level.",
            "- If an apparent insight does not change sourcing strategy, omit it.",
            "- Keep each field concise; prefer short labels and short rationale strings over long prose.",
            "",
            "Return JSON only.",
        ]
    )
    return "\n".join(sections)


def build_perplexity_edge_case_research_user_prompt(
    market_identity: MarketIdentity,
    research_bundle: dict,
    edge_case_focus: list[dict] | None = None,
    planner_summary: str = "",
    edge_case_reasoning: str = "",
) -> str:
    focus_lines = []
    for item in edge_case_focus or []:
        if not isinstance(item, dict):
            continue
        focus = str(item.get("focus") or item.get("question") or "").strip()
        if not focus:
            continue
        priority = str(item.get("priority", "medium")).strip() or "medium"
        reason = str(item.get("reason", "")).strip()
        rendered = f"- {focus} (priority: {priority})"
        if reason:
            rendered += f" | why it matters: {reason}"
        focus_lines.append(rendered)

    sections = [
        "Investigate hidden pools and false-negative risk using the structured sourcing context below.",
        f"ROLE: {market_identity.role_title}",
        f"GEOGRAPHY: {market_identity.geography or 'Not specified'}",
        f"LEVEL: {market_identity.role_level or 'Not specified'}",
    ]
    if planner_summary.strip():
        sections.extend(["", "PLANNER SUMMARY:", planner_summary.strip()])
    if edge_case_reasoning.strip():
        sections.extend(["", "WHY EDGE-CASE RESEARCH TRIGGERED:", edge_case_reasoning.strip()])
    if focus_lines:
        sections.extend(["", "EDGE-CASE RESEARCH FOCUS AREAS:", *focus_lines])
    sections.extend(
        [
            "",
            "RESEARCH OBJECTIVES:",
            "1. Infer the most useful hidden-pool and false-negative questions from the sourcing evidence.",
            "2. Explain how relevant candidates may self-label differently from the obvious target title.",
            "3. Characterize fragmented title families, adjacent-but-relevant archetypes, and hidden submarkets conservatively.",
            "4. Return sourcing implications only after identifying why those pools are easy to miss.",
            "",
            "EDGE-CASE CONTEXT TO PRIORITIZE:",
            _dump_bundle(research_bundle.get("edge_case_context", {})),
            "",
            "FULL STRUCTURED INTERNAL CONTEXT:",
            _dump_bundle(research_bundle),
            "",
            "IMPORTANT CONSTRAINTS:",
            "- Treat internal sourcing evidence as ground truth for observed performance.",
            "- Prefer candidate-visibility explanations over generic employer-demand commentary.",
            "- Do not claim that a hidden pool is high-fit unless the internal evidence supports that possibility.",
            "- If evidence is thin, return conservative hypotheses and validation tasks rather than strong conclusions.",
            "",
            "Return JSON only.",
        ]
    )
    return "\n".join(sections)


def build_perplexity_research_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "market_intel_external_research",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "inferred_research_questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "question": {"type": "string"},
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                                "why_it_matters": {"type": "string"},
                                "sourcing_trigger": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["answered", "unresolved"],
                                },
                                "supporting_run_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "evidence_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "question",
                                "priority",
                                "why_it_matters",
                                "sourcing_trigger",
                                "status",
                                "supporting_run_refs",
                                "evidence_refs",
                            ],
                        },
                    },
                    "market_findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "kind": {"type": "string"},
                                "label": {"type": "string"},
                                "summary": {"type": "string"},
                                "why_it_matters": {"type": "string"},
                                "confidence": {"type": "number"},
                                "supporting_run_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "evidence_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "kind",
                                "label",
                                "summary",
                                "why_it_matters",
                                "confidence",
                                "supporting_run_refs",
                                "evidence_refs",
                            ],
                        },
                    },
                    "sourcing_implications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": [
                                        "add_title_family",
                                        "add_employer_target",
                                        "probe_adjacent_pool",
                                        "relax_boolean",
                                        "validate_hypothesis",
                                        "instrumentation_followup",
                                    ],
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                                "recommendation": {"type": "string"},
                                "rationale": {"type": "string"},
                                "brief_target_field": {"type": "string"},
                                "suggested_values": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "expected_effect": {"type": "string"},
                                "supporting_run_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "evidence_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "category",
                                "priority",
                                "recommendation",
                                "rationale",
                                "brief_target_field",
                                "suggested_values",
                                "expected_effect",
                                "supporting_run_refs",
                                "evidence_refs",
                            ],
                        },
                    },
                    "open_questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "question": {"type": "string"},
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                                "next_step": {"type": "string"},
                                "supporting_run_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "evidence_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "question",
                                "priority",
                                "next_step",
                                "supporting_run_refs",
                                "evidence_refs",
                            ],
                        },
                    },
                },
                "required": [
                    "inferred_research_questions",
                    "market_findings",
                    "sourcing_implications",
                    "open_questions",
                ],
            },
        },
    }


def build_perplexity_edge_case_research_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "market_intel_edge_case_research",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "inferred_research_questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "question": {"type": "string"},
                                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                                "why_it_matters": {"type": "string"},
                                "sourcing_trigger": {"type": "string"},
                                "status": {"type": "string", "enum": ["answered", "unresolved"]},
                                "supporting_run_refs": {"type": "array", "items": {"type": "string"}},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "question",
                                "priority",
                                "why_it_matters",
                                "sourcing_trigger",
                                "status",
                                "supporting_run_refs",
                                "evidence_refs",
                            ],
                        },
                    },
                    "edge_case_submarkets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": "string"},
                                "summary": {"type": "string"},
                                "why_it_is_easy_to_miss": {"type": "string"},
                                "confidence": {"type": "number"},
                                "supporting_run_refs": {"type": "array", "items": {"type": "string"}},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "label",
                                "summary",
                                "why_it_is_easy_to_miss",
                                "confidence",
                                "supporting_run_refs",
                                "evidence_refs",
                            ],
                        },
                    },
                    "title_to_archetype_mapping": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "title_family": {"type": "string"},
                                "likely_archetype": {"type": "string"},
                                "caveats": {"type": "string"},
                                "supporting_run_refs": {"type": "array", "items": {"type": "string"}},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "title_family",
                                "likely_archetype",
                                "caveats",
                                "supporting_run_refs",
                                "evidence_refs",
                            ],
                        },
                    },
                    "self_presentation_patterns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": "string"},
                                "pattern": {"type": "string"},
                                "why_it_causes_false_negatives": {"type": "string"},
                                "supporting_run_refs": {"type": "array", "items": {"type": "string"}},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "label",
                                "pattern",
                                "why_it_causes_false_negatives",
                                "supporting_run_refs",
                                "evidence_refs",
                            ],
                        },
                    },
                    "false_negative_hypotheses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "statement": {"type": "string"},
                                "why_it_matters": {"type": "string"},
                                "validation_task": {"type": "string"},
                                "confidence": {"type": "number"},
                                "supporting_run_refs": {"type": "array", "items": {"type": "string"}},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "statement",
                                "why_it_matters",
                                "validation_task",
                                "confidence",
                                "supporting_run_refs",
                                "evidence_refs",
                            ],
                        },
                    },
                    "edge_case_sourcing_implications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": [
                                        "add_title_family",
                                        "add_employer_target",
                                        "probe_adjacent_pool",
                                        "relax_boolean",
                                        "validate_hypothesis",
                                        "instrumentation_followup",
                                    ],
                                },
                                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                                "recommendation": {"type": "string"},
                                "rationale": {"type": "string"},
                                "brief_target_field": {"type": "string"},
                                "suggested_values": {"type": "array", "items": {"type": "string"}},
                                "expected_effect": {"type": "string"},
                                "supporting_run_refs": {"type": "array", "items": {"type": "string"}},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "category",
                                "priority",
                                "recommendation",
                                "rationale",
                                "brief_target_field",
                                "suggested_values",
                                "expected_effect",
                                "supporting_run_refs",
                                "evidence_refs",
                            ],
                        },
                    },
                    "open_questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "question": {"type": "string"},
                                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                                "next_step": {"type": "string"},
                                "supporting_run_refs": {"type": "array", "items": {"type": "string"}},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "question",
                                "priority",
                                "next_step",
                                "supporting_run_refs",
                                "evidence_refs",
                            ],
                        },
                    },
                },
                "required": [
                    "inferred_research_questions",
                    "edge_case_submarkets",
                    "title_to_archetype_mapping",
                    "self_presentation_patterns",
                    "false_negative_hypotheses",
                    "edge_case_sourcing_implications",
                    "open_questions",
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# Briefing polish — Cloris-voice rewrite over planner output.
# ---------------------------------------------------------------------------
#
# These prompts power BriefingPolishBackend in
# market_intelligence/briefing_polish.py. They produce the recruiter-
# facing reflection at Gate 1 of The Reflection HITL flow, replacing
# the v1 path that surfaced planner_summary verbatim (3-of-4 real
# markets read "Tracking 1 active hypotheses across N run(s)" — a
# templated, ungrounded string).


def build_briefing_polish_system_prompt() -> str:
    """System prompt for the briefing polish step.

    SOURCE OF TRUTH: docs/cloris-ia-doctrine-cursor.md voice/copy rules.
    Last verified against doctrine on 2026-05-25.
    When updating this prompt, re-read the doctrine and update the date.
    Drift between this prompt and the doctrine is the failure mode this
    cross-reference is designed to slow down.

    Voice summary:
    - Character voice is transition-bound and earned in reflection and
      authored dispatch reads.
    - Operational chrome, errors, buttons, pills, and focused-work
      controls use plain product language.
    - No engineer vocabulary, raw ids, or source keys in recruiter prose.
    """

    return """You are Cloris's editorial voice. You read planner output from a recruiting market-intelligence agent and write a recruiter-facing reflection on the just-finished sourcing run.

VOICE RULES (from current IA doctrine):
- Cloris narrates her own work in first person ("I read 84 candidates...", "I want to look at..."). She does NOT address the user as "you" except for the optional steering acknowledgment.
- "She paused on the daily limit" beats "Your run hit the governor." beats "RUN PAUSED." Editorial voice; not operational; not shouty.
- High-stakes and no-signal cases drop character entirely. If the run produced no candidates, write plainly: "I don't have enough from this run to draw conclusions yet — let me read the broader market."
- Operational copy and voice copy never overlap. This is voice copy. Render in editorial register.

PARAGRAPH RULES:
- 2 to 4 sentences. Period.
- Lead with a SPECIFIC named signal from the structured input: a candidate count, save count, save rate as a percent, the strongest lane name, a channel name. Never "Tracking N hypotheses." Never generic counts.
- Past-tense for what happened ("I read 47 candidates and saved 3"). Present-tense for intent ("I want to look at...").
- If the recruiter added a steering note, weave it in as a final sentence: "Per your note, I'll also factor in: \\"<note>\\"."
- Cite ONLY values present in the structured input. Do NOT invent numbers, lane names, or candidate names. If a signal isn't present, omit the clause that would have cited it.

ENGINE IDENTIFIERS — TRANSLATE, DO NOT QUOTE:
- The structured input may contain engine-layer identifiers (lane keys, family names) that look like `devprod_genai`, `forward_deployed_engineering`, `colombian_academic_ml`. These are jargon — the recruiter has never seen them and never should.
- Each lane in `top_lanes` carries both `lane_name` (recruiter-readable) and `lane_key` (engine identifier). USE `lane_name`. NEVER quote `lane_key`.
- The same applies to anything else in the input that looks like a snake_case identifier: translate to a recruiter-readable form (humanize: replace underscores with spaces, title-case the result) before citing it in the paragraph.
- Concrete: write "DevProd GenAI" not "devprod_genai"; write "Forward Deployed Engineering" not "forward_deployed_engineering". If a translation isn't obvious, omit the clause rather than quoting the raw identifier.

BANNED TOKENS (these are engineer jargon — the recruiter never sees them):
- hypothesis, Tracking, lane_key, planner, critic, artifact
- Any snake_case identifier from the input. The output is automatically rejected if it contains an underscore-bearing identifier; the recruiter never sees rejected output but you waste your turn.

INTENTIONS RULES:
- Up to 4 items. Each is one sentence in present-tense intentional voice ("Whether the comp band is realistic for Staff Engineers in NYC right now").
- Map each item to a priority: high | medium | low. Default medium.
- These are what Cloris wants to find out. Use the planner's external_research_focus as the primary source; if empty, default to one generic item.

Return JSON ONLY with this exact shape:
{
  "paragraph": "2-4 sentence reflection in Cloris voice, grounded in specific values from the input.",
  "intentions": [
    {"text": "What I want to find out, one sentence.", "priority": "high|medium|low"}
  ]
}"""


def build_briefing_polish_user_prompt(
    *,
    market_identity: MarketIdentity,
    deterministic_summary: dict,
    planner_result,  # PlannerResult; not type-hinted to avoid circular import
    steering_notes: list[str] | None = None,
) -> str:
    """User prompt: structured input the polish call grounds itself in.

    Pass-through of the structured signals — the prompt is intentionally
    a JSON dump rather than prose, so the LLM has explicit access to
    every value it might cite. The system prompt's containment rule is
    enforced both at the LLM (instruction) and post-call (programmatic
    check in BriefingPolishBackend).
    """

    agg = (deterministic_summary or {}).get("aggregate_metrics") or {}
    lanes = (deterministic_summary or {}).get("lane_intelligence") or []
    # Trim lanes to top 5 by saved_count then candidate_volume so the
    # prompt stays compact; the LLM only needs the strongest signals.
    # Each lane carries BOTH the recruiter-readable display name and
    # the engine lane_key. The system prompt instructs the LLM to use
    # display_name and never quote lane_key. _humanize_lane_key()
    # provides the fallback when no display name exists.
    sortable_lanes = []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lane_key_raw = str(lane.get("lane_key") or "").strip()
        sortable_lanes.append(
            {
                "lane_name": lane_display_name(lane),
                "lane_key": lane_key_raw,
                "saved_count": int(lane.get("saved_count") or 0),
                "candidate_volume": int(lane.get("candidate_volume") or 0),
                "status": lane.get("status", ""),
            }
        )
    sortable_lanes.sort(
        key=lambda d: (d["saved_count"], d["candidate_volume"]),
        reverse=True,
    )

    # Lane translation table: lane_key → recruiter-readable display name.
    # Surfaced separately in the prompt as an explicit reference the
    # LLM consults when tempted to quote a raw key. Built from ALL
    # lanes (not just top 5) so the LLM has a translation for any
    # lane it might be tempted to reference.
    lane_translation_table = {
        str(lane.get("lane_key") or "").strip(): lane_display_name(lane)
        for lane in lanes
        if isinstance(lane, dict) and lane.get("lane_key")
    }

    payload = {
        "market": {
            "role_title": market_identity.role_title,
            "geography": market_identity.geography,
            "role_level": market_identity.role_level,
        },
        "run_signals": {
            "run_count": int(agg.get("run_count") or 0),
            "saved_count": int(agg.get("saved_count") or 0),
            "rejected_count": int(agg.get("rejected_count") or 0),
            "save_rate": float(agg.get("save_rate") or 0.0),
            "facial_yes_rate": float(agg.get("facial_yes_rate") or 0.0),
            "candidate_volume_by_channel": agg.get(
                "candidate_volume_by_channel"
            )
            or {},
        },
        "top_lanes": sortable_lanes[:5],
        "lane_translation_table": lane_translation_table,
        "external_research_focus": list(
            getattr(planner_result, "external_research_focus", []) or []
        )[:4],
        "edge_case_research_focus": list(
            getattr(planner_result, "edge_case_research_focus", []) or []
        )[:2],
        "operator_steering_notes": [
            _normalize_steering(note)
            for note in (steering_notes or [])
            if _normalize_steering(note)
        ],
    }

    return (
        "Write the recruiter-facing reflection for this run.\n\n"
        "INPUT (structured — use ONLY these values; do NOT invent):\n"
        f"{_dump_bundle(payload)}\n\n"
        "When citing a lane in the paragraph, use `lane_name` from "
        "`top_lanes` or look up the recruiter-readable name in "
        "`lane_translation_table`. NEVER quote a raw `lane_key` or any "
        "underscore-bearing identifier from this input.\n\n"
        "Return JSON only, matching the schema in the system prompt."
    )


def lane_display_name(lane: dict) -> str:
    """Resolve a recruiter-readable display name for a lane.

    Prefers the artifact's explicit ``display_name`` when set; falls
    back to humanizing the ``lane_key`` (underscores → spaces, title-
    case). The humanize fallback is deterministic and always produces
    a non-snake_case string, which is what the post-LLM
    snake_case_token_detected check enforces.

    Public — also imported by ``market_intelligence.briefing_polish``
    for the heuristic paragraph builder so both surfaces translate
    lane keys identically. Drift between the two paths produces
    inconsistent recruiter-facing names for the same underlying lane.
    """

    explicit = str(lane.get("display_name") or "").strip()
    if explicit and "_" not in explicit:
        return explicit
    return humanize_lane_key(str(lane.get("lane_key") or ""))


def humanize_lane_key(lane_key: str) -> str:
    """Convert ``forward_deployed_engineering`` → ``Forward Deployed Engineering``.

    Idempotent on already-humanized strings (no underscores → returned
    title-cased). Empty / whitespace input returns empty string so
    upstream callers can decide whether to omit the clause entirely.

    Public — imported by both this module's prompt builder and
    ``market_intelligence.briefing_polish`` for the heuristic
    paragraph builder. The shared helper guarantees both surfaces
    produce the same humanization for any given lane_key.
    """

    text = " ".join(str(lane_key or "").replace("_", " ").split()).strip()
    if not text:
        return ""
    return text.title()


def _normalize_steering(note: str) -> str:
    return " ".join(str(note or "").split()).strip()
