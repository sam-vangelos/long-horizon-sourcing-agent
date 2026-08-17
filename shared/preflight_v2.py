"""
Preflight v2 — Structured brief generation from a JD.

Instead of generating freeform archetypes (which produced the "3+ years minimum bar"
and generic archetypes that caused the permissiveness problem), this preflight asks
Opus to answer specific structured questions. The answers become the Brief's
parametric content.

Usage:
    1. Opus reads the JD and answers the structured questions.
    2. The operator reviews and edits the answers (highest-leverage QA point).
    3. The brief assembles from the reviewed answers.

The operator review step is critical. Preflight generates a DRAFT — the operator
confirms the depth distinction, non-fit patterns, and employer signal rules before
the pipeline runs. This is where you catch "3+ years minimum bar" before it becomes
398 annotation workers in your pipeline.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# ILLUSTRATIVE EXAMPLE COMPOUNDS
# ---------------------------------------------------------------------------
# Two GENERIC, vertical-agnostic worked compounds, used in two places: shown in
# the prompt as a SHAPE demo, and imported by shared.brief_lint's copy-check so
# a model that echoes them back is caught. The vocabulary is deliberately
# placeholder — this file's prompt is scanned by the vertical-vocab ratchet
# (tools/lint_vertical_vocab.py), and the model is FORBIDDEN from copying these,
# so a real vendor/tool/vertical term would be both a ratchet hit and a bad
# exemplar. The first is a precision proof-of-practice string (a distinctive
# anchor term, no generic-verb AND gate); the second a multi-angle recall net
# (each OR group is one concept's synonyms, AND-joined across concepts). Both
# are lint_boolean-clean so they double as valid craft exemplars.
# ---------------------------------------------------------------------------

ILLUSTRATIVE_EXAMPLE_COMPOUNDS = [
    {
        "boolean": '("distinctive-practice-term" OR "specific-named-artifact")',
        "purpose": (
            "Precision proof-of-practice — a distinctive term only real "
            "practitioners write; no generic-verb AND gate"
        ),
        "novelty_bucket": "canonical",
    },
    {
        "boolean": (
            '("primary capability" OR "primary capability synonym") '
            'AND ("supporting signal" OR "supporting signal variant")'
        ),
        "purpose": (
            "Multi-angle recall net — one concept's synonyms per OR group, "
            "AND-joined across concepts"
        ),
        "novelty_bucket": "edge_case",
    },
]

# Rendered once for injection into the prompt as a shape demo (a .format arg,
# so its braces are never re-parsed by str.format). ensure_ascii=False keeps the
# em-dashes readable rather than escaped to \uXXXX in the model-facing prompt.
_ILLUSTRATIVE_EXAMPLE_COMPOUNDS_JSON = json.dumps(
    ILLUSTRATIVE_EXAMPLE_COMPOUNDS, indent=2, ensure_ascii=False
)


class PreflightRegimeError(RuntimeError):
    """Raised when preflight v2 fails twice in a row and the run must abort.

    P9.2: a v2 preflight failure used to fall back to the legacy
    ``shared/preflight.py`` regime, silently reinstating the hardcoded ML
    archetype (the exact permissiveness failure v2 was built to fix). A run
    on the wrong evaluation regime is worse than no run, so this error
    aborts the run instead of swapping regimes underneath the operator.
    """


PREFLIGHT_STAGE = "linkedin_preflight_v2"
PREFLIGHT_MAX_TOKENS = 32_768
PREFLIGHT_SYSTEM_PROMPT = (
    "You are generating structured evaluation criteria for an autonomous sourcing agent. "
    "Respond with ONLY the JSON object requested. No preamble."
)


@dataclass(slots=True)
class PreflightV2Generation:
    """One provider generation after parse and blocking-lint validation."""

    system_prompt: str
    user_prompt: str
    raw_response: str
    data: dict[str, Any]
    findings: tuple[Any, ...]
    usage_context: dict[str, Any]
    source_label: str


@dataclass(slots=True)
class PreflightV2Execution:
    """A validated generation finalized through the typed V2 loader."""

    generation: PreflightV2Generation
    brief_json: dict[str, Any]
    brief: Any


# ---------------------------------------------------------------------------
# PREFLIGHT PROMPT
# ---------------------------------------------------------------------------
# This prompt asks Opus to answer the specific questions that map to Brief fields.
# The output is structured JSON that can be reviewed, edited, and loaded.
# ---------------------------------------------------------------------------

PREFLIGHT_PROMPT = """You are analyzing a job description to generate evaluation criteria for an autonomous sourcing agent. The agent will use these criteria to evaluate hundreds of LinkedIn profiles, so precision matters — small biases compound across many evaluations.

Answer each question below based ONLY on the job description provided. If the JD doesn't provide enough information for a confident answer, say so — do NOT fill gaps with generic criteria. Generic criteria cause either false positives (saving everyone vaguely adjacent) or false negatives (rejecting on keyword absence).

RULES (a deterministic lint aborts the run on violations):
- Author engagement_context for this search. Set selectivity_posture to "selective" for dense or moderate markets and "coverage" only for sparse markets. Copy the top-level hiring_company into engagement_context.hiring_company when it is known; engagement_description and talent_bar_statement should stay specific to this search rather than generic recruiting prose.
- The company hiring for this role goes in "hiring_company" AND "employer_blacklist". It must NEVER appear in any employer_signal_rules tier — it is the one employer this search never sources from.
- Backgrounds the JD enumerates in its candidate-profile section ("who we're looking for") are BOTH caliber proxies AND sourcing pools. The qualifying bar is the JD's stated capabilities and fundamentals, EVIDENCED on the profile — industry membership counts in neither direction (an unnamed-industry candidate with the evidence clears; a named-industry candidate without it does not). Never compile domain-keyword PRESENCE into a hard gate (in user_definition, minimum_bar_description, or non_fit_patterns) unless the JD or an operator instruction explicitly states domain experience as a requirement.
- minimum_bar_description lists HARD REQUIREMENTS ONLY, and the downstream evaluator enforces every clause of it conjunctively — each dimension you write becomes a reject reason on its own. Three compile errors are forbidden. (1) Weighting language is not a requirement: when the operator says a signal is "the strongest", to "weight it above" something, or to "prioritize" it, that signal goes into capability areas and caliber guidance as the thing to weight — it must NOT appear in the bar as a required dimension. Measured 2026-07-27: an operator instruction to weight measurable model improvement above title compiled into "history contains (a) measurable improvement evidenced by benchmarks", and 43 of 58 full-evaluation rejects then cited its absence. (2) Conversation-discoverable is not profile-visible: evidence the operator says a candidate "can point to" or "can speak to" is discoverable in outreach, and profiles vary enormously in written detail — never compile it into an evidenced-on-the-profile requirement. The save decision is an OUTREACH decision, not a hiring decision. (3) A requirement stated alongside a target profile ("X-level with Y experience") is a strong positive unless the JD or operator EXPLICITLY marks it disqualifying to lack; absent that marking, put it in capability areas / employer signals, not the bar.
- role_level preserves the operator's stated level BAND verbatim. "Principal/Staff" is a band — never collapse it to its top rung. The evaluator compares every candidate's demonstrated level against this string, so dropping the lower half of a stated band silently rejects the entire lower half of the operator's own target pool (measured 2026-07-27: "Principal/Staff-level" compiled to "Principal (…)" and 50 of 58 rejects carried level BELOW).
- depth_distinction carries OWNERSHIP depth only — the verbs and artifacts of owning vs consuming the work. Never encode domain presence or absence there: a candidate missing domain vocabulary is a transferability question (Step 3), not a USER-depth verdict.
- Whenever you emit maximum_years_experience you MUST also emit experience_measure: one sentence defining exactly which years the band counts for THIS role. A band without a measure is unenforceable — the evaluator will otherwise pick whichever tenure supports its gut call.
- maximum_years_experience_is_hard MUST be false unless explicit operator guidance says the upper experience bound is a hard rejection gate. A stated experience range alone does not make the ceiling hard. The JD may justify emitting an advisory ceiling, but only the operator can authorize automatic rejection above it.
- experience_measure may only RESTRICT which years count if the JD or an operator instruction states the restriction. NEVER invent qualifiers like post-graduation, full-time, consecutive, or same-industry. Absent a stated restriction, experience_measure counts total professional experience, including research and advanced-degree years.
- Trajectory pattern strings DESCRIBE what a snippet shows; they never carry a disposition. Never write instructions like "default to YES" or "when in doubt, treat as X" inside a pattern string — the evaluation template, not this document, decides how each category is treated.
- expected_yes_rate_low/high are decimals in (0, 1) derived from THIS role's market: combine market_density with role seniority (sparse senior markets often land near 0.05-0.15; dense IC markets near 0.30-0.60). Never reuse a band from an example, and always justify yours in "yes_rate_rationale".
- domain_lane_hints must name 3-6 lanes specific to this role's market — the employer clusters, segments, or domains a search could target. The search planner labels every search string with one of these lanes; "general" is reserved for strings that fit none.
- example_compounds must be EXACTLY 2 worked LinkedIn Boolean strings you author for THIS role from its own lanes, capability key_terms, and the words strong candidates actually write on their profiles. The two must approach the population from two genuinely DIFFERENT angles (e.g. proof-of-practice anchor, title-doorway, community/artifact marker, employer-bound pass, NOT-as-discovery probe), and each compound's SHAPE follows its angle — there is no required clause count or template pair. Craft constraints still bind: never put a generic verb ("managed"/"owned"/"led") behind an AND gate, and every OR group holds ONE concept's synonyms, never a grab-bag. Every quoted term must be plausibly self-written by a candidate on their profile, never a JD heading phrase. Do NOT copy the illustrative shape shown at the end — its vocabulary is placeholder, not content.
- sequencing_heuristics is one short string naming which angles to open with and why, specific to this brief.

JOB DESCRIPTION:
{jd_text}

{intake_context}

{geography_context}

{operator_guidance_context}

═══════════════════════════════════════════════════════
Answer each question as a JSON object. Respond with ONLY the JSON, no preamble.
═══════════════════════════════════════════════════════

{{
  "role_title": "exact title from JD",
  "role_level": "IC level or seniority (e.g., IC4, Senior, Staff, Lead, Director). When the JD or operator states a BAND (e.g. Principal/Staff), reproduce the band verbatim — see the role_level rule above",
  "role_summary": "2-3 sentence description of what this person actually does day-to-day. Not a rephrasing of the JD — a synthesized description of the work.",

  "hiring_company": "the company hiring for this role, exactly as the JD names it — \\"\\" only if the JD truly never reveals it",
  "employer_blacklist": ["the hiring company plus any parent/subsidiary names the JD reveals — candidates currently at these companies are never saved"],
  "engagement_context": {{
    "hiring_company": "the same hiring company as the top-level field; omit or use \\"\\" only when the JD does not reveal it",
    "engagement_description": "one sentence describing this specific search and why the hire matters; omit when the JD does not support one",
    "talent_bar_statement": "one sentence defining what makes a candidate outreach-worthy in this market; omit when the JD does not support one",
    "selectivity_posture": "selective | coverage — selective for dense/moderate markets; coverage only for sparse markets"
  }},

  "capability_areas": [
    {{
      "name": "short name for this capability area",
      "description": "1-2 sentences: what work in this area looks like at this level",
      "builder_signals": ["specific evidence that someone BUILDS in this area — project types, methodologies, outputs"],
      "user_signals": ["specific evidence that someone USES outputs from this area — deploying, fine-tuning for apps, consuming APIs"],
      "key_terms": ["terms that discriminate builders from users in this area — terms only builders would use"],
      "candidate_register_terms": ["3-8 terms per area a qualified candidate would plausibly WRITE on their own profile — self-description vocabulary, never JD headings or requisition phrases; distinct channel from key_terms, which stay evaluation-facing"]
    }}
  ],

  "depth_distinction": {{
    "builder_definition": "what OWNING this work means for THIS role — the verbs and artifacts of designing/running the loop, never domain-keyword presence",
    "user_definition": "what CONSUMING this work looks like — the application-layer version that looks similar on a resume but isn't the same job; describe depth of ownership only, never the absence of a domain",
    "edge_case_guidance": "how to handle profiles that are genuinely ambiguous on OWNERSHIP — what evidence establishes BUILDER or USER, and what remains UNKNOWN"
  }},

  "transferable_fundamentals_bar": "when the JD invites backgrounds beyond the direct domain (its candidate-profile section names industries or a catch-all): the specific fundamentals evidence — in this role's terms — that a no-direct-domain candidate must SHOW on their profile to earn TRANSFERABLE_SAVE; \\"\\" when the role has no transfer pool",

  "non_fit_patterns": [
    {{
      "label": "short name",
      "description": "what this person actually builds every day",
      "why_not": "why their work doesn't connect to this role despite surface similarity",
      "examples": ["concrete example: 'warehouse slotting analytics at a QSR chain'"]
    }}
  ],

  "employer_signal_rules": [
    {{
      "tier": "frontier_lab | strong_ai | general_tech | neutral",
      "employer_patterns": ["company names or patterns"],
      "evidence_required": "what additional evidence beyond employer is needed to save",
      "save_on_employer_alone": false
    }}
  ],

  "minimum_years_experience": 0,
  "maximum_years_experience": "int or null — an advisory leveling CEILING, emitted ONLY when the JD or operator calibration states an upper experience bound; null when the role has no ceiling. By default it informs judgment but is not an automatic reject",
  "maximum_years_experience_is_hard": false,
  "experience_measure": "REQUIRED when maximum_years_experience is set: one sentence defining exactly which years the band counts for this role (e.g. 'total professional experience, including research and advanced-degree years' by default, or 'years owning <the specific relevant work>, in any industry' only when the JD or an operator instruction states that restriction); \\"\\" when there is no ceiling",
  "minimum_bar_description": "what the minimum bar means in practice — not just years, but what those years should contain. HARD REQUIREMENTS ONLY, each clause is enforced conjunctively downstream; operator weighting language and conversation-discoverable evidence never compile into this field — see the minimum_bar_description rule above",

  "facial_calibration": {{
    "expected_yes_rate_low": null,
    "expected_yes_rate_high": null,
    "yes_rate_rationale": "one sentence: why this band, from market density and role seniority. Replace the nulls above with your derived decimals — null is not an acceptable final value.",
    "fast_exit_patterns": ["career trajectories where the ENTIRE history is obviously outside scope — every position points away from relevance"],
    "trajectory_yes_patterns": ["career trajectory patterns detectable from title+company+dates that FAVOR passing to full evaluation — e.g., strong-signal employers for this domain, relevant role transitions, specific keywords in titles"],
    "trajectory_ambiguous_patterns": ["career trajectory patterns that CANNOT be resolved from a snippet alone — e.g., a relevant title at a strong company where the specific domain is unknown. Describe the pattern only — no disposition language."],
    "trajectory_no_patterns": ["career trajectory patterns that favor rejection ONLY if the ENTIRE career history matches. Even one exception in the trajectory means the pattern does not apply."]
  }},

  "facial_ambiguity_posture": "\\"ternary\\" or \\"binary\\" — ternary routes trajectory_ambiguous_patterns to FACIAL_BORDERLINE (full evaluation resolves them); binary requires extra positive signal for a YES. Choose ternary when snippets frequently cannot resolve fit for this role (sparse markets, roles whose substance is invisible in titles); binary when snippets resolve cleanly.",

  "domain_lane_hints": [
    {{
      "lane": "short_snake_case_label",
      "patterns": ["lowercase substrings that identify this lane inside a search string or its rationale"]
    }}
  ],

  "canonical_title_patterns": ["the exact or near-exact titles the OBVIOUS pool for this role carries — what a standard sourcing pass would search first"],
  "canonical_company_patterns": ["named employers of that canonical pool — reuse the employer names from your domain_lane_hints employer clusters"],
  "canonical_framework_patterns": ["named tools, methods, or artifacts only the canonical pool's practitioners list on profiles; [] if the role has no such vocabulary"],
  "canonical_broad_patterns": ["broad cohort phrases that describe the canonical population generically"],
  "edge_case_patterns": ["brief-specific phrases describing the transfer/adjacent populations a standard pass would systematically miss, in the language those candidates' profiles actually use"],
  "edge_case_company_patterns": ["named employers where those adjacent populations concentrate; [] if none are known"],

  "example_compounds": [
    {{
      "boolean": "a LinkedIn Boolean string built from THIS brief's own lanes and capability key_terms",
      "purpose": "what this string retrieves and why it earns a slot",
      "novelty_bucket": "canonical | edge_case"
    }}
  ],
  "sequencing_heuristics": "one short string: which of your example_compounds / angles to open with and why, specific to this brief",

  "geography": {{
    "facet_candidates": ["LinkedIn Recruiter location facet names this search should bound to, extracted from the JD — e.g. \\"New York City Metropolitan Area\\"; use [] if the JD states no geography"],
    "rationale": "one line: where in the JD this came from"
  }},

  "market_density": "sparse | moderate | dense",

  "preflight_confidence_notes": "flag any areas where the JD didn't provide enough info for a confident answer — these are the fields the operator should review most carefully"
}}

═══════════════════════════════════════════════════════
EXAMPLE — a DIFFERENT role (Director of Supply Chain Operations), showing the expected shape and specificity. Do NOT copy its content or its numbers.
═══════════════════════════════════════════════════════

{{
  "hiring_company": "Acme Logistics",
  "employer_blacklist": ["Acme Logistics"],
  "engagement_context": {{
    "hiring_company": "Acme Logistics",
    "engagement_description": "A director search for the leader who will redesign and run a multi-node distribution network.",
    "talent_bar_statement": "Outreach-worthy candidates show end-to-end network-design ownership, not only site operations.",
    "selectivity_posture": "selective"
  }},
  "capability_areas": [
    {{
      "name": "Network optimization",
      "description": "Designs and re-plans multi-node distribution networks under cost and SLA constraints.",
      "builder_signals": ["led a DC-network redesign end to end", "built optimization models that set the network plan"],
      "user_signals": ["operated sites inside an existing network plan", "consumed planning-team outputs"],
      "key_terms": ["network design", "S&OP", "mixed-integer optimization"],
      "candidate_register_terms": ["distribution network design", "S&OP planning", "network optimization model"]
    }}
  ],
  "facial_calibration": {{
    "expected_yes_rate_low": 0.12,
    "expected_yes_rate_high": 0.28,
    "yes_rate_rationale": "Director-level supply-chain leaders are a moderate-density pool and senior titles filter hard at snippet stage.",
    "trajectory_ambiguous_patterns": ["Operations Director at a large retailer — the snippet cannot show whether they owned network design or only ran sites"]
  }},
  "domain_lane_hints": [
    {{"lane": "parcel_carriers", "patterns": ["fedex", "ups", "parcel network"]}},
    {{"lane": "retail_distribution", "patterns": ["omnichannel", "retail distribution"]}}
  ],
  "canonical_title_patterns": ["director of supply chain", "head of distribution"],
  "canonical_company_patterns": ["fedex", "ups"],
  "edge_case_patterns": ["network planning at a large grocery chain", "military logistics officer transitioning to industry"]
}}
(abridged — your answer must include EVERY field in the schema above)

═══════════════════════════════════════════════════════
ILLUSTRATIVE example_compounds SHAPE — placeholder vocabulary, NOT content. Author your own 2 from THIS brief; do NOT copy these.
═══════════════════════════════════════════════════════

{illustrative_example_compounds}"""


def generate_preflight_prompt(
    jd_text: str,
    geography: Optional[str] = None,
    operator_guidance: Optional[str] = None,
    intake_notes: Optional[str] = None,
) -> str:
    """
    Build the preflight prompt from a JD, optional recruiter intake notes,
    optional geography context, and optional operator calibration guidance.

    ``intake_notes`` is the hiring-manager intake-meeting record (seed-brief
    ``intake_notes``): named target employers/titles, screen-outs, and
    recruiter-curated vocabulary. It renders as its own labeled block — never
    concatenated into the JD, because the prompt's register rules treat JD
    phrasing as requisition-suspect and would discount exactly the curated
    vocabulary the intake exists to supply.

    ``operator_guidance`` is the recruiter's own calibration for this search
    (seed-brief ``instructions`` entries — e.g. a seniority band the JD
    leaves ambiguous). It is authoritative over both the JD and the intake
    where they conflict: before these channels existed, preflight had no
    input besides the JD text, so an operator answer to a
    ``preflight_confidence_notes`` question had nowhere to land (2026-07-04
    SPL run: preflight flagged the SPL/Senior-SPL band as ambiguous and asked
    the operator to confirm; the confirmation had no seam to flow back
    through).
    Returns the prompt string to send to Opus.
    """
    intake_context = ""
    if intake_notes and intake_notes.strip():
        intake_context = (
            "RECRUITER INTAKE NOTES (hiring-manager intake meeting, "
            "recruiter-authored):\n"
            f"{intake_notes.strip()}\n\n"
            "Treat these notes as source material alongside the JD — answer "
            "from both together. On candidate requirements, screen-outs, "
            "target employers, titles, and sourcing strategy the intake is "
            "more specific and more current than the JD; where the two "
            "conflict, the intake wins (operator calibration, when present, "
            "outranks both). Vocabulary the intake lists passes the same "
            "register rules as every other source: candidate_register_terms "
            "carry only what a qualified person would plausibly write on "
            "their own profile."
        )

    geo_context = ""
    if geography:
        geo_context = (
            f"GEOGRAPHY CONTEXT: This search targets candidates in {geography}. "
            f"Consider local employer landscape, relevant institutions, and typical "
            f"profile patterns for this market when generating non-fit patterns and "
            f"employer signal rules."
        )

    guidance_context = ""
    if operator_guidance and operator_guidance.strip():
        guidance_context = (
            "OPERATOR CALIBRATION (authoritative — written by the recruiter "
            "running this search; where it conflicts with the JD, this "
            "calibration wins):\n"
            f"{operator_guidance.strip()}\n"
            "Bake these directives into the generated criteria — "
            "minimum_years_experience, minimum_bar_description, the "
            "facial_calibration band and trajectory patterns, and "
            "non_fit_patterns. Express soft ranges structurally (bands, "
            "trajectory patterns); the lint rules above still apply — never "
            "encode them as disposition language inside pattern strings."
        )

    return PREFLIGHT_PROMPT.format(
        jd_text=jd_text,
        intake_context=intake_context,
        geography_context=geo_context,
        operator_guidance_context=guidance_context,
        illustrative_example_compounds=_ILLUSTRATIVE_EXAMPLE_COMPOUNDS_JSON,
    )


def parse_preflight_response(raw: str) -> dict:
    """
    Parse the preflight JSON response from Opus.
    Strips markdown fences if present. Returns the raw dict for operator review.
    """
    cleaned = raw.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        # Remove first line (```json or ```)
        lines = cleaned.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop closing fence
        cleaned = "\n".join(lines)

    return json.loads(cleaned)


def preflight_to_brief_json(preflight_data: dict, overrides: Optional[dict] = None) -> dict:
    """
    Convert preflight output to a Brief-compatible JSON structure.

    The `overrides` dict lets the operator patch specific fields after review.
    This is the QA step — the operator reviews the preflight output, edits
    what needs editing, and the final brief assembles from the merge.

    Example overrides:
        {"minimum_years_experience": 5, "market_density": "dense"}
    """
    result = {**preflight_data}
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = {**result[key], **value}
            else:
                result[key] = value
    engagement_context = result.get("engagement_context")
    if isinstance(engagement_context, dict):
        engagement_context = {**engagement_context}
        if isinstance(result.get("hiring_company"), str):
            engagement_context["hiring_company"] = result["hiring_company"]
        result["engagement_context"] = engagement_context
    return result


def _seed_preflight_inputs(brief: Any) -> tuple[str, str, list[Any], str, str]:
    """Return normalized source channels using the production trust rules."""

    jd_text = str(brief.jd_text)
    permanent_filters = getattr(brief, "permanent_filters", {})
    geography = (
        str(permanent_filters.get("Location") or "").strip()
        if isinstance(permanent_filters, dict)
        else ""
    )
    raw_instructions = getattr(brief, "instructions", None)
    if not isinstance(raw_instructions, (list, tuple)):
        raw_instructions = []
    instructions = list(raw_instructions)
    operator_guidance = "\n".join(
        f"- {str(item).strip()}" for item in instructions if str(item).strip()
    )
    raw_intake = getattr(brief, "intake_notes", "")
    intake_notes = raw_intake.strip() if isinstance(raw_intake, str) else ""
    return jd_text, geography, instructions, operator_guidance, intake_notes


def preflight_overrides_from_brief(
    brief: Any,
    preflight_data: dict[str, Any],
    *,
    geography: str,
) -> dict[str, Any]:
    """Preserve seed destinations and operator constraints in generated V2."""

    overrides: dict[str, Any] = {}
    linkedin_project = getattr(brief, "linkedin_project", "")
    if linkedin_project:
        overrides["linkedin_project"] = linkedin_project
    linkedin_project_id = str(
        getattr(brief, "linkedin_project_id", "") or ""
    ).strip()
    if linkedin_project_id:
        overrides["linkedin_project_id"] = linkedin_project_id
        source_config_raw = preflight_data.get("source_config")
        source_config = (
            dict(source_config_raw) if isinstance(source_config_raw, dict) else {}
        )
        linkedin_source_raw = source_config.get("linkedin")
        linkedin_source = (
            dict(linkedin_source_raw)
            if isinstance(linkedin_source_raw, dict)
            else {}
        )
        linkedin_source["project_id"] = linkedin_project_id
        source_config["linkedin"] = linkedin_source
        overrides["source_config"] = source_config
    if geography:
        overrides["geography"] = geography
    kit_url = getattr(brief, "kit_url", "")
    if kit_url:
        overrides["kit_url"] = kit_url
    employer_blacklist = getattr(brief, "employer_blacklist", None)
    if employer_blacklist:
        overrides["employer_blacklist"] = employer_blacklist
    return overrides


def build_preflight_v2_prompt(brief: Any) -> tuple[str, str, str]:
    """Build the exact system/user prompt pair used by production preflight."""

    jd_text, geography, _instructions, operator_guidance, intake_notes = (
        _seed_preflight_inputs(brief)
    )
    user_prompt = generate_preflight_prompt(
        jd_text,
        geography or None,
        operator_guidance=operator_guidance or None,
        intake_notes=intake_notes or None,
    )
    source_label = "JD + intake notes" if intake_notes else "JD"
    return PREFLIGHT_SYSTEM_PROMPT, user_prompt, source_label


def generate_preflight_v2_once(
    brief: Any,
    *,
    model_name: str,
    usage_context: Optional[dict[str, Any]] = None,
    on_raw_response: Optional[Callable[[str, str, str], None]] = None,
    on_findings: Optional[Callable[[str], None]] = None,
    llm_call: Optional[Callable[..., str | dict]] = None,
) -> PreflightV2Generation:
    """Run the exact production provider/parse/blocking-lint seam once.

    Retry and strategy-shadow ownership intentionally stay outside this helper:
    the LinkedIn orchestrator owns its historical whole-operation retry and
    shadow comparison, while the paid proof probe must make exactly one call.
    A fresh usage-context copy prevents an outer retry from reusing the first
    attempt's generated logical-call id; callers may provide one shared parent
    id for correlation.
    """

    from shared.brief_lint import (
        GeneratedBriefLintError,
        blocking_findings,
        format_findings,
        lint_generated_brief,
    )
    from shared.llm_clients import opus_llm

    jd_text, _geography, instructions, _operator_guidance, intake_notes = (
        _seed_preflight_inputs(brief)
    )
    system_prompt, user_prompt, source_label = build_preflight_v2_prompt(
        brief
    )
    call_context = dict(usage_context or {})
    call_context["stage"] = PREFLIGHT_STAGE
    call_context.pop("logical_call_id", None)
    invoke = llm_call or opus_llm
    raw = invoke(
        system_prompt,
        user_prompt,
        expect_json=False,
        max_tokens=PREFLIGHT_MAX_TOKENS,
        usage_context=call_context,
        model_name=model_name,
    )
    if not isinstance(raw, str):
        raise TypeError("preflight provider response must be raw text")
    if on_raw_response is not None:
        on_raw_response(raw, system_prompt, user_prompt)

    data = parse_preflight_response(raw)
    if not isinstance(data, dict):
        raise TypeError("preflight response must decode to a JSON object")
    lint_source_text = f"{jd_text}\n\n{intake_notes}" if intake_notes else jd_text
    findings = tuple(
        lint_generated_brief(
            data,
            jd_text=lint_source_text,
            operator_instructions=instructions,
            seed_blacklist=list(getattr(brief, "employer_blacklist", None) or []),
        )
    )
    if findings and on_findings is not None:
        on_findings(format_findings(findings))
    blockers = blocking_findings(findings)
    if blockers:
        raise GeneratedBriefLintError(
            "generated brief failed lint: "
            + "; ".join(finding.code for finding in blockers)
        )

    return PreflightV2Generation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=raw,
        data=data,
        findings=findings,
        usage_context=call_context,
        source_label=source_label,
    )


def finalize_preflight_v2(
    brief: Any,
    generation: PreflightV2Generation,
) -> PreflightV2Execution:
    """Apply seed overrides/provenance and prove typed V2 loading once."""

    from shared.brief_loader import _load_v2_brief

    _jd_text, geography, _instructions, _guidance, _intake = (
        _seed_preflight_inputs(brief)
    )

    overrides = preflight_overrides_from_brief(
        brief,
        generation.data,
        geography=geography,
    )
    brief_json = preflight_to_brief_json(generation.data, overrides)
    brief_json["provenance"] = {
        "generated_by": "preflight_v2",
        "reviewed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    loaded = _load_v2_brief(brief_json)
    return PreflightV2Execution(
        generation=generation,
        brief_json=brief_json,
        brief=loaded,
    )


def execute_preflight_v2_once(
    brief: Any,
    *,
    model_name: str,
    usage_context: Optional[dict[str, Any]] = None,
    on_raw_response: Optional[Callable[[str, str, str], None]] = None,
    on_findings: Optional[Callable[[str], None]] = None,
    llm_call: Optional[Callable[..., str | dict]] = None,
) -> PreflightV2Execution:
    """Run and finalize the shared one-call seam used by the paid proof."""

    generation = generate_preflight_v2_once(
        brief,
        model_name=model_name,
        usage_context=usage_context,
        on_raw_response=on_raw_response,
        on_findings=on_findings,
        llm_call=llm_call,
    )
    return finalize_preflight_v2(brief, generation)


# ---------------------------------------------------------------------------
# OPERATOR REVIEW FORMATTING
# ---------------------------------------------------------------------------
# Formats the preflight output for human review before the pipeline runs.
# ---------------------------------------------------------------------------

def format_confidence_notes(preflight_data: dict) -> str:
    """The operator-review banner for preflight's own open questions.

    One source of wording for the two places it prints: inside
    format_for_review's full rendering, and re-printed as the LAST line of
    preflight generation (the full rendering scrolls ~80 lines up before
    "Preflight V2 complete", which is how the SPL/Senior-SPL band question
    went unread on the 2026-07-04 run). Returns "" when the generated brief
    carries no notes.
    """
    notes = str(preflight_data.get("preflight_confidence_notes", "") or "").strip()
    if not notes:
        return ""
    return "\n".join([
        "─" * 40,
        "⚠  PREFLIGHT CONFIDENCE NOTES — REVIEW THESE CAREFULLY",
        "─" * 40,
        f"  {notes}",
    ])


def format_for_review(preflight_data: dict) -> str:
    """
    Format preflight output as a readable review document.
    The operator reads this, edits what's wrong, and confirms.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("PREFLIGHT REVIEW — EDIT BEFORE CONFIRMING")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"Role: {preflight_data.get('role_title', '???')} ({preflight_data.get('role_level', '???')})")
    lines.append(f"Summary: {preflight_data.get('role_summary', '???')}")
    hiring_company = str(preflight_data.get("hiring_company", "") or "").strip()
    blacklist = [str(b) for b in preflight_data.get("employer_blacklist", []) or [] if str(b).strip()]
    if hiring_company or blacklist:
        lines.append(f"Hiring company: {hiring_company or '???'}")
        lines.append(f"Employer blacklist: {', '.join(blacklist) or '(EMPTY — review!)'}")
    engagement_context = preflight_data.get("engagement_context") or {}
    if isinstance(engagement_context, dict) and engagement_context:
        engagement_description = engagement_context.get("engagement_description")
        talent_bar = engagement_context.get("talent_bar_statement")
        if isinstance(engagement_description, str) and engagement_description:
            lines.append(f"Engagement: {engagement_description}")
        if isinstance(talent_bar, str) and talent_bar:
            lines.append(f"Talent bar: {talent_bar}")
        lines.append(
            "Selectivity posture: "
            f"{engagement_context.get('selectivity_posture', '???')}"
        )
    lines.append("")

    # Capability areas
    lines.append("─" * 40)
    lines.append("CAPABILITY AREAS")
    lines.append("─" * 40)
    for i, ca in enumerate(preflight_data.get("capability_areas", []), 1):
        lines.append(f"\n  {i}. {ca['name']}")
        lines.append(f"     What it looks like: {ca['description']}")
        lines.append(f"     Builder signals: {', '.join(ca['builder_signals'])}")
        lines.append(f"     User signals: {', '.join(ca['user_signals'])}")
        if ca.get("key_terms"):
            lines.append(f"     Key terms: {', '.join(ca['key_terms'])}")
        if ca.get("candidate_register_terms"):
            lines.append(
                "     Candidate-register terms: "
                f"{', '.join(ca['candidate_register_terms'])}"
            )
    lines.append("")

    # Depth distinction
    dd = preflight_data.get("depth_distinction", {})
    lines.append("─" * 40)
    lines.append("DEPTH DISTINCTION (highest-leverage review point)")
    lines.append("─" * 40)
    lines.append(f"  BUILDER: {dd.get('builder_definition', '???')}")
    lines.append(f"  USER:    {dd.get('user_definition', '???')}")
    lines.append(
        "  UNKNOWN: Missing evidence or genuinely ambiguous ownership stays "
        "UNKNOWN, not USER; resolve it from the whole profile during full review."
    )
    lines.append(f"  Edge cases: {dd.get('edge_case_guidance', '???')}")
    lines.append("")

    # Non-fit patterns
    lines.append("─" * 40)
    lines.append("NON-FIT PATTERNS")
    lines.append("─" * 40)
    for nf in preflight_data.get("non_fit_patterns", []):
        examples = f" (e.g., {', '.join(nf['examples'])})" if nf.get("examples") else ""
        lines.append(f"  - {nf['label']}: {nf['description']}{examples}")
        lines.append(f"    Why not: {nf['why_not']}")
    lines.append("")

    # Employer signals
    lines.append("─" * 40)
    lines.append("EMPLOYER SIGNAL RULES")
    lines.append("─" * 40)
    for rule in preflight_data.get("employer_signal_rules", []):
        companies = ", ".join(rule["employer_patterns"])
        lines.append(f"  [{rule['tier']}] {companies}")
        lines.append(f"    Evidence required: {rule['evidence_required']}")
        lines.append(f"    Save on employer alone: {rule['save_on_employer_alone']}")
    lines.append("")

    # Minimum bar
    lines.append("─" * 40)
    lines.append("MINIMUM BAR")
    lines.append("─" * 40)
    lines.append(f"  Years: {preflight_data.get('minimum_years_experience', '???')}+")
    maximum = preflight_data.get("maximum_years_experience")
    if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0:
        if preflight_data.get("maximum_years_experience_is_hard") is True:
            lines.append(f"  Ceiling: {maximum} years — HARD reject gate")
        else:
            lines.append(
                f"  Ceiling: {maximum} years — advisory leveling context; "
                "not an automatic reject"
            )
        measure = str(preflight_data.get("experience_measure", "") or "").strip()
        if measure:
            lines.append(f"  Measure: {measure}")
    lines.append(f"  Meaning: {preflight_data.get('minimum_bar_description', '???')}")
    lines.append("")

    # Confidence notes — shared wording with the end-of-generation re-print.
    notes_block = format_confidence_notes(preflight_data)
    if notes_block:
        lines.append(notes_block)
        lines.append("")

    # Facial calibration — band + trajectory patterns
    fc = preflight_data.get("facial_calibration", {})
    band_low = fc.get("expected_yes_rate_low")
    band_high = fc.get("expected_yes_rate_high")
    band_rationale = str(fc.get("yes_rate_rationale", "") or "").strip()
    if band_low is not None or band_high is not None or band_rationale:
        lines.append("─" * 40)
        lines.append("FACIAL YES-RATE BAND")
        lines.append("─" * 40)
        lines.append(f"  Expected: {band_low} – {band_high}")
        lines.append(f"  Rationale: {band_rationale or '(MISSING — review!)'}")
        lines.append("")

    lane_hints = preflight_data.get("domain_lane_hints", []) or []
    if lane_hints:
        lines.append("─" * 40)
        lines.append("DOMAIN LANE HINTS (search-planner lane vocabulary)")
        lines.append("─" * 40)
        for hint in lane_hints:
            if isinstance(hint, dict):
                patterns = ", ".join(str(p) for p in hint.get("patterns", []) or [])
                lines.append(f"  - {hint.get('lane', '???')}: {patterns}")
        lines.append("")

    geography = preflight_data.get("geography", {}) or {}
    if isinstance(geography, dict) and geography.get("facet_candidates"):
        lines.append("─" * 40)
        lines.append("GEOGRAPHY (extracted from JD)")
        lines.append("─" * 40)
        for facet in geography.get("facet_candidates", []) or []:
            lines.append(f"  - {facet}")
        rationale = str(geography.get("rationale", "") or "").strip()
        if rationale:
            lines.append(f"  Why: {rationale}")
        lines.append("")

    # Example compounds + sequencing — the model-authored search levers. This
    # is the QA point where the operator confirms the worked Booleans before
    # the strategy model composes from them (they render verbatim into the
    # formation prompt). Renders only when present; omits cleanly otherwise.
    example_compounds = [
        ec
        for ec in (preflight_data.get("example_compounds", []) or [])
        if isinstance(ec, dict) and str(ec.get("boolean", "") or "").strip()
    ]
    sequencing = str(preflight_data.get("sequencing_heuristics", "") or "").strip()
    if example_compounds or sequencing:
        lines.append("─" * 40)
        lines.append("EXAMPLE COMPOUNDS (model-authored search levers)")
        lines.append("─" * 40)
        for ec in example_compounds:
            boolean = str(ec.get("boolean", "") or "").strip()
            purpose = str(ec.get("purpose", "") or "").strip()
            bucket = str(ec.get("novelty_bucket", "") or "").strip()
            bucket_suffix = f" [{bucket}]" if bucket else ""
            if purpose:
                lines.append(f"  - {purpose}{bucket_suffix}")
                lines.append(f"      {boolean}")
            else:
                lines.append(f"  - {boolean}{bucket_suffix}")
        if sequencing:
            lines.append(f"  Sequencing: {sequencing}")
        lines.append("")

    lines.append("─" * 40)
    lines.append("FACIAL TRIAGE — TRAJECTORY PATTERNS")
    lines.append("─" * 40)
    lines.append("\n  Fast exits (entire career clearly outside scope):")
    for p in fc.get("fast_exit_patterns", []):
        lines.append(f"    - {p}")
    lines.append("\n  YES patterns (trajectory signals favoring full review):")
    for p in fc.get("trajectory_yes_patterns", []):
        lines.append(f"    - {p}")
    lines.append("\n  AMBIGUOUS patterns (snippet cannot resolve; the evaluation template decides treatment):")
    for p in fc.get("trajectory_ambiguous_patterns", []):
        lines.append(f"    - {p}")
    lines.append("\n  NO patterns (only if ENTIRE history matches):")
    for p in fc.get("trajectory_no_patterns", []):
        lines.append(f"    - {p}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("Review complete. Edit the JSON and confirm to proceed.")
    lines.append("=" * 70)

    return "\n".join(lines)
