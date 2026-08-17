"""Prompts for the chief-of-staff cross-source synthesis agent.

Mirrors the layered structure used by
``market_intelligence.research_prompts.build_briefing_polish_*``:

- System prompt = voice rules + paragraph rules + engine-identifier
  translation rules + banned tokens + weight semantics + priority
  rules + the exact JSON output schema. The grounding contracts live
  in the system prompt; the cascade in
  ``cloris.chief_of_staff.agent`` enforces them post-call.
- User prompt = a labeled JSON dump of the structured per-source
  signals + the existing single editorial briefing for the same run,
  plus a "use ONLY these values" instruction and a pointer back to
  the schema.

SOURCE OF TRUTH for voice rules: ``docs/cloris-ia-doctrine-cursor.md``
and ``Cloris-Product-North-Star.md``. When updating this prompt,
re-read those references and update the date in
:func:`build_chief_of_staff_system_prompt`'s docstring.

The ``per_specialist_weight`` semantic is **independent 0.0..1.0
trust scores per contributing specialist**, NOT a normalized share
of total trust. The system prompt names this explicitly so the LLM
doesn't default to a zero-sum framing — two specialists can both
land at 0.9, or one can be 0.9 and the other 0.1; the weights do
not sum to any particular total.
"""

from __future__ import annotations

import json
from typing import Any

from market_intelligence.schema import MarketIdentity
from shared.source_capabilities import source_capability_prompt_block


def _dump_bundle(value: dict) -> str:
    """JSON-dump a payload for inclusion in a user prompt.

    Mirrors :func:`market_intelligence.research_prompts._dump_bundle`
    formatting (``indent=2, sort_keys=True``) so the structured-input
    posture is identical across the polish backend and this agent.
    """

    return json.dumps(value, indent=2, sort_keys=True)


def build_chief_of_staff_system_prompt() -> str:
    """System prompt for the chief-of-staff synthesis call.

    SOURCE OF TRUTH: docs/cloris-ia-doctrine-cursor.md voice/copy
    rules plus Cloris-Product-North-Star.md voice commitments. Last
    verified on 2026-05-25. When updating this prompt, re-read those
    references and update the date. Drift between this prompt and
    current doctrine is the failure
    mode this cross-reference is designed to slow down.

    Voice summary:
    - Character voice is transition-bound and earned in intake,
      reflection, and authored dispatch reads.
    - Operational chrome, errors, buttons, pills, and focused-work
      controls use plain product language.
    - Multi-agent facts name the specialist source when provenance is
      part of the evidence.

    The chief-of-staff section in the reflection Gate-1 surface is
    voice copy. Render in editorial register, first person, narrating
    Cloris's read of the team's work.
    """

    return """You are Cloris speaking as chief of staff to her team of recruiting specialists. After a multi-source sourcing run, you read the per-source signals each specialist produced this run plus the editorial briefing already written for the run, and you return a team-level read for the principal: a synthesis paragraph in Cloris voice, an independent trust weight per contributing specialist, and a one-sentence priority for what the principal should look at first.

VOICE RULES (from current IA doctrine):
- Cloris narrates her own work in first person ("Across LinkedIn and GitHub, I read..."). She does NOT address the user as "you" except in the priority sentence ("Start with the LinkedIn saves first.").
- "She paused on the daily limit" beats "Your run hit the governor." beats "RUN PAUSED." Editorial voice; not operational; not shouty.
- This is voice copy. Render in editorial register. The recruiter is the principal; Cloris is bringing them a coordinated read of the team's work, not raw output.

PARAGRAPH RULES:
- 2 to 4 sentences. Period.
- Lead with a SPECIFIC named signal from the structured input: a per-source candidate count, a per-source save count, a source name, or a per-source top lane name. Never generic ("the team did some work").
- Past-tense for what each specialist did this run ("LinkedIn surfaced 47 candidates and 3 saves"; "GitHub returned 22 maintainers"). Present-tense for intent ("I'd start with...").
- Cite ONLY values present in the structured input. Do NOT invent numbers, source names, lane names, or candidate names. If a signal isn't present, omit the clause that would have cited it.
- The framing is honest: the team's read is cross-source pattern, not per-specialist judgment-weighting on individual candidates. Don't oversell what synthesis is.

ENGINE IDENTIFIERS — TRANSLATE, DO NOT QUOTE:
- The structured input may contain engine-layer identifiers (lane keys, family names) that look like `forward_deployed_engineering`, `devprod_genai`, `colombian_academic_ml`. These are jargon — the recruiter has never seen them and never should.
- When a `top_lane` value contains underscores, humanize it before citing (replace underscores with spaces, title-case if natural). Concrete: write "Forward Deployed Engineering" not "forward_deployed_engineering".
- The same applies to the source identifiers themselves ("linkedin" -> "LinkedIn", "github" -> "GitHub"). Capitalize naturally.
- If a translation isn't obvious, omit the clause rather than quoting the raw identifier.

BANNED TOKENS (engineer jargon — the recruiter never sees them):
- hypothesis, Tracking, lane_key, planner, critic, artifact
- Any snake_case identifier from the input. The output is automatically rejected if it contains an underscore-bearing identifier; the recruiter never sees rejected output but you waste your turn.

PER-SPECIALIST WEIGHT RULES:
- Each `weight` is an INDEPENDENT 0.0-to-1.0 score expressing how much the principal should trust this specialist's read on this brief — NOT a share of total trust. Two specialists can both be 0.9; one can be 0.9 and the other 0.1; the weights do not sum to any particular total.
- The keys of `per_specialist_weight` MUST be source identifiers from the structured input's `per_source_signals` set. Never invent a specialist that didn't run this brief; that output is automatically rejected. Use the EXACT source key strings as they appear in the input (lowercase, no humanization).
- Reason FIRST, then weight. In each `rationale` (one short clause, ≤ 25 words, citing a specific input signal — candidate count, save count, top lane), weigh this specialist's INFORMATIVENESS on this brief: save density, whether a zero-save negative read is itself informative ("returned 22 maintainers, none cleared the bar — the negative read is informative"), and sample size. THEN assign the weight that reasoning implies. Avoid hedge phrases ("seems decent").
- Anchors, not lookups: candidates-but-zero-saves tends to land moderate (~0.4-0.6); candidates-and-saves tends higher, scaled with save density. Reason from the specific read rather than matching an anchor.

PRIORITY FOR PRINCIPAL RULES:
- One sentence. Names what the recruiter should look at first when they open the workspace. Concrete and actionable: a specific source's saves, a specific top lane, a specific candidate count to triage first.
- Past-tense for context, present-tense for the action. Avoid hedging ("you might want to consider").
- This is the only place in the synthesis where you address the recruiter as "you" — and only via an action verb ("Start with..."). Don't write a second-person paragraph.

Return JSON ONLY with this exact shape:
{
  "paragraph": "2-4 sentence team-level read in Cloris voice, grounded in specific values from per_source_signals and the briefing.",
  "per_specialist_weight": {
    "<source_key_from_input>": {
      "rationale": "Reason FIRST about this specialist's informativeness on this brief; the weight follows from it.",
      "weight": 0.0_to_1.0_independent_score
    }
  },
  "priority_for_principal": "One sentence — what the recruiter should look at first."
}"""


def build_chief_of_staff_user_prompt(
    *,
    market_identity: MarketIdentity,
    per_source_signals: dict[str, dict],
    briefing_paragraph: str,
    deterministic_summary: dict | None = None,
    prior_handoff_payloads: dict[str, dict] | None = None,
) -> str:
    """User prompt: structured input the synthesis call grounds itself in.

    Pass-through of the structured signals as a JSON dump rather than
    prose, so the LLM has explicit access to every value it might
    cite. The system prompt's containment rule is enforced both at
    the LLM (instruction) and post-call (programmatic check in
    :class:`cloris.chief_of_staff.agent.ChiefOfStaffAgent`).

    ``per_source_signals`` carries one entry per contributing source
    (a source that produced candidates this run) with ``candidate_count``,
    ``save_count``, and an optional humanized ``top_lane`` display
    name. ``briefing_paragraph`` is the existing single-briefing
    Cloris-voice paragraph already written for this run by
    :class:`market_intelligence.briefing_polish.BriefingPolishBackend`
    — surfaced for context so the synthesis can build on rather than
    duplicate it.

    ``prior_handoff_payloads`` (audit Move #1) carries the structured
    per-source handoff summaries persisted at
    ``chief_of_staff_runs.handoff_payloads_json``. Each entry maps a
    source key to ``{candidate_count, save_count, confidence,
    per_source_signal_summary, top_saves}``. The LLM uses these for
    cross-source narrative depth — naming the substantive saves a
    given specialist surfaced rather than reading as
    "${source} ran ${count} candidates" alone. ``None`` (or empty
    dict) skips the section so single-source / pre-Move-1 runs render
    byte-identically to today's prompt.
    """

    contributing_sources = sorted(per_source_signals.keys())

    payload: dict[str, Any] = {
        "market": {
            "role_title": market_identity.role_title,
            "geography": market_identity.geography,
            "role_level": market_identity.role_level,
        },
        "contributing_sources": contributing_sources,
        "per_source_signals": {
            source: {
                "candidate_count": int(
                    (per_source_signals.get(source) or {}).get("candidate_count", 0)
                    or 0
                ),
                "save_count": int(
                    (per_source_signals.get(source) or {}).get("save_count", 0)
                    or 0
                ),
                "top_lane": (per_source_signals.get(source) or {}).get(
                    "top_lane"
                ),
            }
            for source in contributing_sources
        },
        "single_briefing_paragraph": briefing_paragraph or "",
    }

    if prior_handoff_payloads:
        payload["prior_handoff_payloads"] = prior_handoff_payloads

    handoff_rule_addendum = (
        "\n- prior_handoff_payloads carries each specialist's structured "
        "self-report from this run — the saves they thought worth handing "
        "off + a one-paragraph signal summary. Cite the substantive "
        "narratives (top_saves[*].role_fit_narrative) when they sharpen "
        "the cross-source read; do not paste them verbatim."
        if prior_handoff_payloads
        else ""
    )

    return (
        "Write the chief-of-staff synthesis for this multi-source run.\n\n"
        "INPUT (structured — use ONLY these values; do NOT invent):\n"
        f"{_dump_bundle(payload)}\n\n"
        "RULES:\n"
        "- per_specialist_weight keys MUST be drawn from contributing_sources. "
        "Never invent a specialist that didn't run this brief.\n"
        "- Each weight is an INDEPENDENT 0.0-1.0 trust score, not a share. "
        "Two specialists can both be 0.9.\n"
        "- The synthesis paragraph builds on the single-briefing paragraph "
        "but adds the team-level read; it is not a verbatim restatement.\n"
        "- Humanize raw source keys and snake_case lane names in prose "
        "(\"linkedin\" -> \"LinkedIn\", \"forward_deployed_engineering\" -> "
        "\"Forward Deployed Engineering\"). Use the raw source keys ONLY "
        "in per_specialist_weight."
        f"{handoff_rule_addendum}\n\n"
        "Return JSON only, matching the schema in the system prompt."
    )


def build_dispatch_system_prompt() -> str:
    """System prompt for the chief-of-staff dispatch planner (LLM path).

    Layered like :func:`build_chief_of_staff_system_prompt`: Cloris voice
    contract, JSON-only output shape, banned-token and snake_case
    discipline (enforced post-call in
    :class:`cloris.chief_of_staff.agent.ChiefOfStaffAgent`). Last
    verified against ``docs/cloris-ia-doctrine-cursor.md`` on
    2026-05-25.

    Output is a machine-oriented dispatch plan (module keys + optional
    handoff hints), not recruiter-facing prose — voice rules apply to
    any English strings inside ``handoff_condition``.
    """

    return """You are Cloris planning execution order for a multi-module sourcing brief.

VOICE & REGISTER:
- Output is JSON only — no markdown fences, no preamble. Keys and string values use plain product language; recruiter-facing copy belongs in other surfaces, but ``handoff_condition`` (when present) should read as plain English, not engineer narration ("Tracking hypothesis…").

ENGINE IDENTIFIERS — DO NOT LEAK:
- Use only source keys from the provided ``known_sources`` list (exact lowercase strings such as ``linkedin``, ``github``). Never invent a module name.
- In ``handoff_condition``, humanize ideas — do not paste snake_case lane keys or internal identifiers. Underscore-bearing tokens in any string field cause automatic rejection.

BANNED TOKENS (reject if present in any string value you emit):
- hypothesis, Tracking, lane_key, planner, critic, artifact

PER-STEP RULES:
- ``module_name``: REQUIRED, non-empty, MUST be one of ``known_sources``.
- ``handoff_condition``: optional string or null. Free-text placeholder for when this step should yield to the next (e.g. a qualitative gate). If unsure, use null. Do not reference modules that are not in ``known_sources``.
- Evidence boundaries are not permission to skip a module. They describe what a module cannot prove alone and what companion evidence completes the read.

Return JSON ONLY with this exact shape:
{
  "steps": [
    {"module_name": "<source-key from known_sources>", "handoff_condition": "<short string or null>"}
  ]
}"""


def build_dispatch_user_prompt(
    *,
    brief: object,
    prior_runs: list[dict],
    known_sources: tuple[str, ...],
) -> str:
    """User prompt for dispatch: brief context, prior runs, allowed modules."""

    role_title = str(getattr(brief, "role_title", "") or "")
    target_modules = getattr(brief, "target_modules", None)
    if not isinstance(target_modules, list):
        raw = getattr(brief, "raw", None)
        if isinstance(raw, dict):
            tm = raw.get("target_modules")
            target_modules = tm if isinstance(tm, list) else []
        else:
            target_modules = []

    declared = [str(m) for m in target_modules if isinstance(m, str) and m]

    market_identity_block: dict[str, Any] = {}
    raw = getattr(brief, "raw", None)
    if isinstance(raw, dict):
        mi = raw.get("market_identity")
        if isinstance(mi, dict):
            market_identity_block = dict(mi)

    payload: dict[str, Any] = {
        "role_title": role_title,
        "target_modules_declared": declared,
        "market_identity": market_identity_block,
        "known_sources": list(known_sources),
        "source_capability_manifest": source_capability_prompt_block(),
        "prior_runs": list(prior_runs or []),
    }

    return (
        "Propose an ordered dispatch plan for this brief.\n\n"
        "CONSTRAINTS:\n"
        "- Every ``module_name`` must appear exactly as one of "
        "``known_sources`` (same spelling and casing).\n"
        "- Order steps in the sequence you recommend for execution; "
        "the brief's declared ``target_modules`` are guidance, not a "
        "hard constraint if rerordering improves flow — but you must "
        "not introduce unknown modules.\n"
        "- Use the source_capability_manifest as product truth. Evidence "
        "boundaries mean companion evidence is needed; they do not mean "
        "the module should be ignored.\n"
        "- Keep ``handoff_condition`` short or null.\n\n"
        f"INPUT:\n{_dump_bundle(payload)}\n\n"
        "Return JSON only, matching the schema in the system prompt."
    )
