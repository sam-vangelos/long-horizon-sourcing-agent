"""
GitHub-native evaluation templates for the autonomous sourcing agent.

These templates mirror the structure and output format of judgment_templates.py
but evaluate GitHub-specific evidence (repos, READMEs, toolchain usage, target-
project contributions, papers/website) instead of LinkedIn evidence (career trajectory,
summary bullets).

CRITICAL DESIGN CONSTRAINT: The output format is IDENTICAL to the LinkedIn
templates so that parse_facial_response and parse_full_evaluation_response can
be reused without modification.

Evidence sources (GitHub):
  - Repository code: imports, configs, training scripts, eval harness extensions
  - README content: architecture explanations, training procedures, results tables
  - Target project contributions: PRs to projects named in the brief and other
    recognized high-signal repositories in the candidate's domain
  - Website + papers: personal sites, arxiv links
  - Repo metadata: topics, descriptions, stars, forks, language distribution
  - Profile: bio, profile README, follower count
"""

from __future__ import annotations

import re

from shared.brief_schema import Brief

# Re-export parse functions so callers can import everything from one module.
from shared.judgment.templates import parse_facial_response, parse_full_evaluation_response


# ---------------------------------------------------------------------------
# GITHUB FACIAL TRIAGE TEMPLATE
# ---------------------------------------------------------------------------
# Purpose: Filter out GitHub profiles where no reasonable full evaluation
# could produce a save. This is a TRIAGE, not a judgment.
# Expected pass-through: 20-50% (GitHub profiles skew noisier than LinkedIn).
# Parse failure default: YES (cost of false positive = one cheap extraction;
# cost of false negative = a permanently missed candidate).
# ---------------------------------------------------------------------------

GITHUB_FACIAL_TRIAGE_TEMPLATE = """You are triaging candidate profiles from GitHub search results.

ROLE: {role_title} ({role_level}) — {role_summary}

YOUR TASK: Decide whether this candidate's GitHub portfolio warrants a full profile review. You are deciding whether to spend tokens on a full read, not whether to save.

WHAT YOU HAVE: A username, bio, profile README (if any), public repo list with names/descriptions/topics/stars/languages, declared registry maintainership evidence (npm / crates.io), when present, and a toolchain summary listing detected frameworks and libraries across their repos. You do NOT have full repo contents, commit history, or code quality analysis yet.

═══════════════════════════════════════════════════════
STEP 1 — FAST EXITS
═══════════════════════════════════════════════════════

Reject immediately ONLY if the profile clearly indicates work outside scope:
{fast_exit_block}

A fast exit requires that NO repo, topic, or bio element has a plausible connection to the role. One relevant-looking repo or toolchain signal anywhere means this is NOT a fast exit.

═══════════════════════════════════════════════════════
STEP 2 — PORTFOLIO READ
═══════════════════════════════════════════════════════

The portfolio summary is your highest-signal field. Read ALL of: toolchain_detected, repo_summaries, external project contributions, website_papers, profile_summary. What you're looking for:

PORTFOLIO PATTERNS THAT FAVOR YES:
{portfolio_yes_patterns}

PORTFOLIO PATTERNS THAT ARE AMBIGUOUS (default YES — let the full evaluation resolve):
{portfolio_ambiguous_patterns}

PORTFOLIO PATTERNS THAT FAVOR NO (only if consistent across the ENTIRE portfolio):
{portfolio_no_patterns}

CAPABILITY AREAS for this role:
{capability_area_names}

═══════════════════════════════════════════════════════
DECISION
═══════════════════════════════════════════════════════

GitHub profiles can be misleading — a user with mostly forks may have significant private work, and repo names alone do not reveal depth. Do NOT try to make depth calls at this stage.

- FACIAL_YES: Any repo, toolchain signal, declared registry maintainership evidence, contribution, or bio element COULD connect to a capability area. Ambiguity favors YES.
- FACIAL_NO: The ENTIRE portfolio clearly indicates work outside all capability areas. Every repo, topic, and toolchain signal points away from relevance. No single element creates doubt.

CANDIDATE PORTFOLIO:
{candidate_portfolio}

Respond with EXACTLY this format:
DECISION: FACIAL_YES or FACIAL_NO
REASON: One sentence — what portfolio signal you see (if YES) or why the full portfolio is clearly outside scope (if NO)."""


GITHUB_FACIAL_TRIAGE_TEMPLATE_BATCH = """You are triaging candidate profiles from GitHub search results.

ROLE: {role_title} ({role_level}) — {role_summary}

YOUR TASK: For each candidate, decide whether the GitHub portfolio warrants a full profile review. You are deciding whether to spend tokens on a full read, not whether to save.

WHAT YOU HAVE: For each candidate, a username, bio, profile README (if any), public repo list with names/descriptions/topics/stars/languages, declared registry maintainership evidence (npm / crates.io), when present, and a toolchain summary listing detected frameworks and libraries across their repos. You do NOT have full repo contents, commit history, or code quality analysis yet.

FAST EXITS — reject ONLY if the profile clearly indicates work outside scope:
{fast_exit_block}

PORTFOLIO READ — read ALL of: toolchain_detected, repo_summaries, external project contributions, website_papers, profile_summary.

YES patterns:
{portfolio_yes_patterns}

AMBIGUOUS patterns (default YES — let the full evaluation resolve):
{portfolio_ambiguous_patterns}

NO patterns (only if consistent across the ENTIRE portfolio):
{portfolio_no_patterns}

CAPABILITY AREAS for this role:
{capability_area_names}

GitHub profiles can be misleading — a user with mostly forks may have significant private work, and repo names alone do not reveal depth. Do NOT try to make depth calls at this stage.

- FACIAL_YES: Any repo, toolchain signal, declared registry maintainership evidence, contribution, or bio element COULD connect to a capability area. Ambiguity favors YES.
- FACIAL_NO: The ENTIRE portfolio clearly indicates work outside all capability areas. Every repo, topic, and toolchain signal points away from relevance. No single element creates doubt.

CANDIDATE PORTFOLIOS:
{candidate_portfolios_numbered}

Respond with EXACTLY this format for each candidate, one per line:
[candidate_number] FACIAL_YES or FACIAL_NO | one-sentence reason citing the portfolio signal"""


# ---------------------------------------------------------------------------
# GITHUB FULL EVALUATION TEMPLATE
# ---------------------------------------------------------------------------
# Purpose: Determine whether a candidate should be saved to the pipeline.
# This is where the bar lives. Three-step claim-and-evidence procedure
# producing output IDENTICAL to the LinkedIn full evaluation template.
# Parse failure default: REJECT with PARSE_FAILURE flag (auditable, not silent).
# ---------------------------------------------------------------------------

GITHUB_FULL_EVALUATION_TEMPLATE = """You are evaluating a candidate for a specific technical role based on their GitHub presence. Follow the procedure below EXACTLY. Do not skip steps.

ROLE: {role_title} ({role_level})
{role_summary}

MINIMUM BAR: {minimum_years_experience}+ years hands-on. {minimum_bar_description}
{seniority_standard_block}
WHAT YOU HAVE: An enriched GitHub profile — bio, profile README, public repos with full README content, detected toolchain/framework usage across repos, contributions to external repos, declared registry maintainership evidence (npm / crates.io), when present, website content, papers, stars/forks counts, language distribution, and an account-age and repo-activity-span summary.

EVIDENCE HIERARCHY (GitHub-specific — ranked by diagnostic value). The ranking principle:
CODE EVIDENCE outranks DECLARED ROLES, declared roles outrank DOCUMENTED CLAIMS, and
documented claims outrank SELF-DESCRIPTION. GitHub is a passive skill surface — most
candidates never describe themselves for a recruiter's benefit, and the evaluation must
never require that they do.

Tier 1 — HIGHEST (code):
1. DOMAIN TOOLCHAIN USAGE — Repos that import, configure, or extend frameworks named in the brief's capability areas or discriminating skills. The brief's capability areas define what counts. A repo with substantive framework configuration and custom training or evaluation setup is stronger signal than any number of stars.

Tier 1.5 — HIGHEST (declared role):
2. DECLARED REGISTRY MAINTAINERSHIP OR GOVERNANCE ROSTER — A package registry's owner/maintainer record naming this person is a declared fact, not an inference; a governance roster (CODEOWNERS, MAINTAINERS, conda-forge recipe-maintainers) naming this person is equally declared. Weigh either above stars/forks and above inferred contribution signals. Downloads and reverse dependencies measure real-world dependence — the registry equivalent of dependents, stronger than attention metrics.

Tier 2 — HIGH (documented work artifacts):
3. README CONTENT + REPO STRUCTURE — Architecture explanations, training procedure documentation, directory structures showing data pipelines, eval harness configs, and results tables. A well-documented training repo is equivalent to LinkedIn summary bullets describing the same work.
4. TARGET PROJECT CONTRIBUTIONS — Merged PRs or substantive issues on projects named in the brief (target_projects) and other recognized high-signal repositories in the candidate's domain. Drive-by typo fixes do not count.
5. WEBSITE + PAPERS — Personal site with project writeups, papers with hands-on implementation sections.
6. REPO ACTIVITY SPAN — How long this person has been creating and pushing to substantive repos. A span of years is direct tenure
   evidence; a record that begins recently bounds what seniority the profile can support,
   whatever the bio claims.

Tier 3 — CORROBORATOR ONLY (self-description):
7. BIO + PROFILE README SELF-DESCRIPTION — What the person says about their own work.
   Treat as a corroborator: it can raise confidence within the band that code or declared
   evidence already established, and it can direct attention to private or employer work
   that public repos cannot show — but it can never establish BUILDER depth or seniority
   on its own. Specific toolchain naming (production frameworks, named systems) is worth
   more than fluent role vocabulary; fluent self-description with no public corroboration
   at all is a weaker profile than silent repos with strong code.

Tier 4 — MODERATE (attention and metadata):
8. REPO TOPICS + DESCRIPTIONS — Topics matching the brief's capability areas and code signals indicate domain awareness. But topics are self-reported — verify against actual repo content.
9. STARS/FORKS — High stars on domain-relevant repos indicate community validation. But stars can reflect novelty, not depth. Forks of popular repos without modifications are noise.
10. LANGUAGE DISTRIBUTION — Python-heavy is expected but not diagnostic. Rust/C++ in ML contexts (kernels, inference engines) is a mild positive.

═══════════════════════════════════════════════════════
SPARSE PROFILE CHECK (run FIRST, before anything else)
═══════════════════════════════════════════════════════

A sparse GitHub profile is one with FEW OR NO substantive repos — mostly forks with no modifications, empty repos, or only a profile README. GitHub profiles can be sparse because significant work lives in private repos or employer orgs. If the profile is sparse, check:

{inferential_save_block}

REGISTRY SPARSE-PROFILE CARVE-OUT: A profile sparse on repos that carries a REGISTRY MAINTAINERSHIP section with non-trivial dependence signals (downloads and/or reverse dependencies), or declared governance-roster maintainership (CODEOWNERS / MAINTAINERS / recipe-maintainers naming the person), is NOT sparse — judge it on the declared registry or roster evidence, not on repo count alone.

ADDITIONAL SPARSE SIGNAL: If the profile is sparse BUT the detected toolchain includes highly specific practitioner frameworks ({discriminating_skills_examples}), treat this as supporting evidence. These frameworks are too specialized to appear without hands-on experience. A sparse profile with contributions to target projects plus detected domain-specific framework usage strengthens an inferential case built on target-project contributions.

If a brief-defined inferential-save condition is met, respond with DECISION: INFERENTIAL_SAVE, confidence 0.4-0.6. These go to the recruiter for manual review.

If no inferential save applies AND the profile is sparse, respond REJECT — not enough signal.

If the profile HAS meaningful detail, proceed to Step 1.

═══════════════════════════════════════════════════════
STEP 1 — CAPABILITY MAPPING (signal, NOT a gate)
═══════════════════════════════════════════════════════

Try to map the candidate's ACTUAL WORK (as evidenced by repo contents, toolchain usage, and contributions) to one of the following capability areas. Areas are stack-ranked. Within the same match level (DIRECT or ADJACENT), score toward the TOP of the confidence range for areas ranked 1-3 and toward the BOTTOM for areas ranked 4+. Example: ADJACENT to area #1 → 0.65-0.75; ADJACENT to area #6 → 0.60-0.65.

{capability_area_block}

EMPLOYER SIGNAL RULES:
{employer_signal_block}

Note: On GitHub, "employer" signal comes from the profile's company field, bio mentions, and contribution graphs to employer repos. Weight accordingly — a bio saying "engineer at a major lab" with no public ML repos is weaker than someone with 10 ML repos and no employer mentioned.

RESULT — classify the match as one of:
- DIRECT: Repo contents, toolchain usage, or contributions demonstrate work that falls squarely within a capability area. Cite the area and the evidence.
- ADJACENT: The work touches a capability area but isn't core to it (e.g., built ML evaluation tools but for a non-LLM domain). Note what's adjacent and why.
- NONE: No capability area maps. This is NOT an automatic reject — proceed to Step 2.

═══════════════════════════════════════════════════════
STEP 2 — DEPTH TEST (runs REGARDLESS of Step 1 result)
═══════════════════════════════════════════════════════

This step evaluates the candidate's hands-on ML depth INDEPENDENT of whether their domain matches. Read the repo evidence across ALL repositories. Do they demonstrate hands-on ML work where data quality, model training, or evaluation methodology was a primary focus?

{depth_block}

Key distinction — look at CODE EVIDENCE and REPO PATTERNS:
- BUILDER evidence: Original repos with custom training loops, eval harness extensions, data pipeline code, fine-tuning configs with non-default hyperparameters, iterative commit history on ML systems, published papers with code, custom CUDA kernels, novel evaluation methods, training infrastructure code
- USER evidence: Only forks with minimal or no changes, API wrapper repos, tutorial/course notebook repos, default configs copied from docs, "using X model" READMEs without implementation, Gradio/Streamlit demos calling hosted APIs, awesome-list curation without original work

"Fine-tuned" in a README is ambiguous — check for actual training code. A repo claiming "fine-tuned LLaMA" with actual LoRA configs, training scripts, and loss curves is BUILDER. A repo with the same claim but only inference code calling a hosted model is USER.

CRITICAL — GITHUB ≠ LINKEDIN, IN BOTH DIRECTIONS: Most professionals' employer work lives
in private repos, so public thinness is weak evidence AGAINST a candidate. But the
correction for that is not letting self-description substitute for code: a bio/README
describing professional domain work (specific frameworks, production systems, named
pipelines) makes the profile worth a recruiter cross-check, not a BUILDER verdict. Route
it through SIGNAL_SAVE:
- Bio/README names brief-relevant SPECIFIC toolchain AND public activity shows at least
  domain-level engagement (relevant repos, contributions, or contribution history in the
  domain, even if shallow) → SIGNAL_SAVE is available.
- Bio/README fluency with NO public corroboration of any kind → not a save of any kind.
  Depth must be evidenced, not narrated.
- Silent profile with strong code → judge the code; self-description is not required for
  any verdict, including SAVE at full confidence.

═══════════════════════════════════════════════════════
STEP 3 — TRANSFERABILITY (only if Step 1 was ADJACENT or NONE)
═══════════════════════════════════════════════════════

If Step 1 found no direct capability area match, ask: does this person's METHODOLOGY transfer to the role, even though their DOMAIN doesn't match?

The test: "If you took this person's skills and methodology and pointed them at LLM training data / RL environments / model evaluation instead of their current domain, would the skills apply?"

TRANSFERS (methodology is domain-portable):
- Evaluation framework design in one domain -> evaluation framework design for the brief's target domain. The person knows how to measure system quality. The specific stack changes; the methodology of rigorous evaluation is the same.
- Data quality systems for model training in any domain -> data quality for large-scale model training in the role's domain. Someone who built data curation pipelines and quality metrics for computational biology models knows what training data quality means.
- Custom model training (architectures, training loops, hyperparameter optimization) in any domain -> can learn LLM training. Deep hands-on model training experience is the hardest skill to develop.
- Systems-level ML infrastructure (custom kernels, distributed training, inference optimization) -> transfers directly regardless of model type.

DOES NOT TRANSFER (domain gap is too wide AND methodology doesn't port):
- Web frontend repos with no ML component -> strong coding but no ML depth to port.
- DevOps/infrastructure repos (Terraform, Kubernetes) without ML workload focus -> operational skills but no model training methodology.
- Mobile app development -> different engineering discipline entirely.
- Data visualization / dashboarding without ML model building -> data-adjacent but no model training methodology.

RESULT: TRANSFERABLE (cite what methodology transfers) or NOT_TRANSFERABLE (explain why the gap is too wide).

═══════════════════════════════════════════════════════
STEP 4 — DECISION
═══════════════════════════════════════════════════════

State the strongest CASE FOR this candidate's relevance:
- What evidence supports their fit? (capability area match, depth evidence, transferable methodology)

State the strongest CASE AGAINST:
- What's missing, misaligned, or uncertain?

NON-FIT PATTERNS — work that is valuable but outside scope:
{non_fit_block}

CRITICAL — NON-FIT OVERRIDE RULE:
{non_fit_override_rule}

DECISION MATRIX — weigh the evidence from Steps 1-3 together:

DIRECT match + BUILDER depth = SAVE (high confidence, 0.80-0.95)
ADJACENT match + BUILDER depth = SAVE (moderate confidence, 0.60-0.75)
NONE match + BUILDER depth + TRANSFERABLE methodology = TRANSFERABLE_SAVE (moderate confidence, 0.45-0.55, for recruiter awareness)
NONE match + BUILDER depth + NOT TRANSFERABLE = REJECT
Any match level + USER depth (with no professional bio/README signal) = REJECT (application-layer work regardless of domain)
Any match level + USER public repos BUT bio/README names brief-relevant SPECIFIC toolchain AND public activity shows at least domain-level engagement = SIGNAL_SAVE (0.40-0.55) — specific toolchain in bio/README with corroborating public domain activity, but public repos don't fully demonstrate depth. A LinkedIn cross-check is warranted. This is NOT the same as INFERENTIAL_SAVE (which is for sparse profiles with strong employer/credential priors). SIGNAL_SAVE is for profiles with genuine professional signals that can't be fully verified from GitHub alone.
Sparse profile meeting inferential conditions = INFERENTIAL_SAVE (0.35-0.50)

CONFIDENCE CALIBRATION (within each range):
- Top of range: Multiple independent evidence sources. Example: repos + activity record + target-project contributions all confirm depth.
- Middle of range: One strong source, one ambiguous. Example: strong bio but sparse repos.
- Bottom of range: Single weak source. Example: one thin code signal with nothing corroborating it.
- Capability area stack rank also differentiates: areas ranked 1-3 should score toward the top of the applicable range; areas ranked 4+ toward the bottom.

{decision_standard_block}

On GitHub, professional self-description (bio, README) directs the recruiter's LinkedIn cross-check via SIGNAL_SAVE when it is corroborated by public domain activity — it never substitutes for evidenced depth. The agent's job is to surface candidates worth that cross-check without inflating unverified claims into depth verdicts.

The guard against permissiveness is the DEPTH TEST, not the capability mapping. A person must demonstrate hands-on ML builder depth to be saved — no exceptions. What the capability mapping determines is confidence level, not the binary decision. Strong domain match + depth = high confidence save. No domain match + depth + transferable methodology = moderate confidence save. No depth = reject regardless of domain.
{maintainership_block}
CANDIDATE EVIDENCE:
{candidate_evidence}

═══════════════════════════════════════════════════════
RESPOND WITH EXACTLY THIS FORMAT:
═══════════════════════════════════════════════════════

STEP_1_MATCH: DIRECT or ADJACENT or NONE
STEP_1_AREA: [capability area name if DIRECT/ADJACENT, or "N/A"]
STEP_1_EVIDENCE: [cite specific repo contents, toolchain signals, or contributions, 2-3 sentences max]

STEP_2_DEPTH: BUILDER or USER
STEP_2_EVIDENCE: [what code evidence and repo patterns indicate, 1-2 sentences]

STEP_3_TRANSFERABILITY: TRANSFERABLE or NOT_TRANSFERABLE or N/A (if DIRECT match)
STEP_3_EVIDENCE: [what methodology transfers, or why the gap is too wide, 1-2 sentences. Write "N/A" if Step 1 was DIRECT]

CASE_FOR: [strongest argument for relevance, 1-2 sentences]
CASE_AGAINST: [strongest argument against, 1-2 sentences]

DECISION: SAVE or REJECT or INFERENTIAL_SAVE or TRANSFERABLE_SAVE or SIGNAL_SAVE
CONFIDENCE: [0.0 to 1.0 — use the decision matrix ranges above]
SUMMARY: [one-line evaluation a hiring manager could act on]"""


# ---------------------------------------------------------------------------
# DEFAULT GITHUB FACIAL PATTERNS
# ---------------------------------------------------------------------------
# Used when the brief does not supply GitHub-specific facial calibration.
# ---------------------------------------------------------------------------

_DEFAULT_GITHUB_FAST_EXITS = [
    "Profile is an organization account, not a person",
    "Profile has zero repos AND zero contributions AND no bio — completely empty",
    "ALL repos are forks of web frontend frameworks (React, Vue, Angular) with zero ML content anywhere",
    "Profile is clearly a bot or auto-generated account",
]

_DEFAULT_PORTFOLIO_YES_PATTERNS = [
    "Any repo importing or configuring role-relevant frameworks named in the brief's capability areas or discriminating skills",
    "Declared registry maintainership on npm or crates.io for a package relevant to the brief",
    "Contributions to the brief's target projects or comparably significant projects in its domain",
    "Repos with topics matching the brief's capability areas and code signals",
    "Papers linked in bio or repos (arxiv, conference proceedings)",
    "Personal website with project writeups or research descriptions",
    "Repos containing training scripts, eval configs, or data pipeline code",
    "Bio mentions brief-relevant research, model work, or domain expertise",
]

_DEFAULT_PORTFOLIO_AMBIGUOUS_PATTERNS = [
    "Mix of brief-relevant and unrelated repos — some relevant repos exist alongside unrelated work",
    "Sparse profile with brief-relevant bio but few public repos (private work is common)",
    "Repos with brief-aligned topics but unclear depth from descriptions alone",
    "Data-oriented repos that could indicate model work or could be analytics-only",
    "Research-oriented profile with topics adjacent to the brief but unclear hands-on coding",
]

_DEFAULT_PORTFOLIO_NO_PATTERNS = [
    "ALL repos are web frontend (React, Next.js, Vue) with zero brief-relevant content",
    "ALL repos are DevOps/infrastructure (Terraform, Kubernetes, CI/CD) with no brief-aligned workloads",
    "ALL repos are mobile development (iOS, Android, Flutter)",
    "ALL repos are tutorial completions or bootcamp projects with no original brief-relevant work",
    "Profile is entirely game development, embedded systems, or blockchain with no brief intersection",
]


# ---------------------------------------------------------------------------
# ASSEMBLY FUNCTIONS
# ---------------------------------------------------------------------------
# These inject Brief content into template slots at runtime.
# The GitHub judger calls these — never constructs prompts directly.
# ---------------------------------------------------------------------------

def assemble_github_facial_system(brief: Brief) -> str:
    """Return the cacheable system prompt for GitHub facial triage (no candidate data)."""
    # GitHub-specific fast exits
    github_fast_exits = getattr(brief, "github_fast_exit_patterns", None)
    if github_fast_exits:
        fast_exit_block = "\n".join(f"- {p}" for p in github_fast_exits)
    else:
        fast_exit_block = "\n".join(f"- {p}" for p in _DEFAULT_GITHUB_FAST_EXITS)

    portfolio_yes = getattr(brief, "github_portfolio_yes_patterns", None)
    if portfolio_yes:
        portfolio_yes_block = "\n".join(f"- {p}" for p in portfolio_yes)
    else:
        portfolio_yes_block = "\n".join(f"- {p}" for p in _DEFAULT_PORTFOLIO_YES_PATTERNS)

    portfolio_ambiguous = getattr(brief, "github_portfolio_ambiguous_patterns", None)
    if portfolio_ambiguous:
        portfolio_ambiguous_block = "\n".join(f"- {p}" for p in portfolio_ambiguous)
    else:
        portfolio_ambiguous_block = "\n".join(f"- {p}" for p in _DEFAULT_PORTFOLIO_AMBIGUOUS_PATTERNS)

    portfolio_no = getattr(brief, "github_portfolio_no_patterns", None)
    if portfolio_no:
        portfolio_no_block = "\n".join(f"- {p}" for p in portfolio_no)
    else:
        portfolio_no_block = "\n".join(f"- {p}" for p in _DEFAULT_PORTFOLIO_NO_PATTERNS)

    return GITHUB_FACIAL_TRIAGE_TEMPLATE.format(
        role_title=brief.role_title,
        role_level=brief.role_level,
        role_summary=brief.role_summary,
        fast_exit_block=fast_exit_block,
        portfolio_yes_patterns=portfolio_yes_block,
        portfolio_ambiguous_patterns=portfolio_ambiguous_block,
        portfolio_no_patterns=portfolio_no_block,
        capability_area_names="\n".join(f"  - {name}" for name in brief.capability_area_names()),
        candidate_portfolio="[provided in user message]",
    )


def assemble_github_facial_batch_system(brief: Brief) -> str:
    """Return the cacheable system prompt for GitHub batch facial triage."""
    github_fast_exits = getattr(brief, "github_fast_exit_patterns", None)
    if github_fast_exits:
        fast_exit_block = "\n".join(f"- {p}" for p in github_fast_exits)
    else:
        fast_exit_block = "\n".join(f"- {p}" for p in _DEFAULT_GITHUB_FAST_EXITS)

    portfolio_yes = getattr(brief, "github_portfolio_yes_patterns", None)
    if portfolio_yes:
        portfolio_yes_block = "\n".join(f"- {p}" for p in portfolio_yes)
    else:
        portfolio_yes_block = "\n".join(f"- {p}" for p in _DEFAULT_PORTFOLIO_YES_PATTERNS)

    portfolio_ambiguous = getattr(brief, "github_portfolio_ambiguous_patterns", None)
    if portfolio_ambiguous:
        portfolio_ambiguous_block = "\n".join(f"- {p}" for p in portfolio_ambiguous)
    else:
        portfolio_ambiguous_block = "\n".join(f"- {p}" for p in _DEFAULT_PORTFOLIO_AMBIGUOUS_PATTERNS)

    portfolio_no = getattr(brief, "github_portfolio_no_patterns", None)
    if portfolio_no:
        portfolio_no_block = "\n".join(f"- {p}" for p in portfolio_no)
    else:
        portfolio_no_block = "\n".join(f"- {p}" for p in _DEFAULT_PORTFOLIO_NO_PATTERNS)

    return GITHUB_FACIAL_TRIAGE_TEMPLATE_BATCH.format(
        role_title=brief.role_title,
        role_level=brief.role_level,
        role_summary=brief.role_summary,
        fast_exit_block=fast_exit_block,
        portfolio_yes_patterns=portfolio_yes_block,
        portfolio_ambiguous_patterns=portfolio_ambiguous_block,
        portfolio_no_patterns=portfolio_no_block,
        capability_area_names="\n".join(f"  - {name}" for name in brief.capability_area_names()),
        candidate_portfolios_numbered="[provided in user message]",
    )


def assemble_github_full_evaluation_system(brief: Brief) -> str:
    """Return the cacheable system prompt for GitHub full evaluation (no candidate data)."""
    return GITHUB_FULL_EVALUATION_TEMPLATE.format(
        role_title=brief.role_title,
        role_level=brief.role_level,
        role_summary=brief.role_summary,
        minimum_years_experience=brief.minimum_years_experience,
        minimum_bar_description=brief.minimum_bar_description,
        capability_area_block=brief.capability_area_block(),
        depth_block=brief.depth_block(),
        non_fit_block=brief.non_fit_block(),
        non_fit_override_rule=brief.non_fit_override_rule_block(),
        employer_signal_block=brief.employer_signal_block(),
        inferential_save_block=brief.inferential_save_block(),
        discriminating_skills_examples=brief.discriminating_skills_examples(),
        maintainership_block=_assemble_maintainership_block(brief),
        seniority_standard_block=_assemble_seniority_block(brief),
        decision_standard_block=_assemble_decision_standard_block(brief),
        candidate_evidence="[provided in user message]",
    )


def assemble_github_facial_prompt(brief: Brief, portfolio_text: str) -> str:
    """Assemble a GitHub facial triage prompt for a single candidate.

    Uses GitHub-specific facial patterns from the brief if available
    (via getattr on github_facial_calibration or similar fields),
    otherwise falls back to sensible defaults.
    """
    # GitHub-specific fast exits
    github_fast_exits = getattr(brief, "github_fast_exit_patterns", None)
    if github_fast_exits:
        fast_exit_block = "\n".join(f"- {p}" for p in github_fast_exits)
    else:
        fast_exit_block = "\n".join(f"- {p}" for p in _DEFAULT_GITHUB_FAST_EXITS)

    # Portfolio YES patterns
    portfolio_yes = getattr(brief, "github_portfolio_yes_patterns", None)
    if portfolio_yes:
        portfolio_yes_block = "\n".join(f"- {p}" for p in portfolio_yes)
    else:
        portfolio_yes_block = "\n".join(f"- {p}" for p in _DEFAULT_PORTFOLIO_YES_PATTERNS)

    # Portfolio AMBIGUOUS patterns
    portfolio_ambiguous = getattr(brief, "github_portfolio_ambiguous_patterns", None)
    if portfolio_ambiguous:
        portfolio_ambiguous_block = "\n".join(f"- {p}" for p in portfolio_ambiguous)
    else:
        portfolio_ambiguous_block = "\n".join(f"- {p}" for p in _DEFAULT_PORTFOLIO_AMBIGUOUS_PATTERNS)

    # Portfolio NO patterns
    portfolio_no = getattr(brief, "github_portfolio_no_patterns", None)
    if portfolio_no:
        portfolio_no_block = "\n".join(f"- {p}" for p in portfolio_no)
    else:
        portfolio_no_block = "\n".join(f"- {p}" for p in _DEFAULT_PORTFOLIO_NO_PATTERNS)

    return GITHUB_FACIAL_TRIAGE_TEMPLATE.format(
        role_title=brief.role_title,
        role_level=brief.role_level,
        role_summary=brief.role_summary,
        fast_exit_block=fast_exit_block,
        portfolio_yes_patterns=portfolio_yes_block,
        portfolio_ambiguous_patterns=portfolio_ambiguous_block,
        portfolio_no_patterns=portfolio_no_block,
        capability_area_names="\n".join(f"  - {name}" for name in brief.capability_area_names()),
        candidate_portfolio=portfolio_text,
    )


def assemble_github_full_evaluation_prompt(brief: Brief, evidence_text: str) -> str:
    """Assemble a GitHub full evaluation prompt for one candidate.

    Uses the same Brief formatting methods as the LinkedIn template
    (capability_area_block, depth_block, etc.) since the brief's
    capability areas and depth distinctions are role-level, not
    platform-level.
    """
    return GITHUB_FULL_EVALUATION_TEMPLATE.format(
        role_title=brief.role_title,
        role_level=brief.role_level,
        role_summary=brief.role_summary,
        minimum_years_experience=brief.minimum_years_experience,
        minimum_bar_description=brief.minimum_bar_description,
        capability_area_block=brief.capability_area_block(),
        depth_block=brief.depth_block(),
        non_fit_block=brief.non_fit_block(),
        non_fit_override_rule=brief.non_fit_override_rule_block(),
        employer_signal_block=brief.employer_signal_block(),
        inferential_save_block=brief.inferential_save_block(),
        discriminating_skills_examples=brief.discriminating_skills_examples(),
        maintainership_block=_assemble_maintainership_block(brief),
        seniority_standard_block=_assemble_seniority_block(brief),
        decision_standard_block=_assemble_decision_standard_block(brief),
        candidate_evidence=evidence_text,
    )


def _assemble_maintainership_block(brief: Brief) -> str:
    """Return the maintainership-evaluation guidance block, or empty string.

    OSS Maintainers Slice 6 — when the brief carries explicit
    ``target_projects``, append a block instructing the LLM how to
    weigh the MAINTAINERSHIP EVIDENCE section in candidate evidence.
    Behavior-preserving for classic github briefs: empty
    ``target_projects`` ⇒ empty string ⇒ template renders byte-
    identically to today.
    """

    target_projects = list(getattr(brief, "target_projects", []) or [])
    if not target_projects:
        return ""

    desired_level = (
        getattr(brief, "maintainership_level", "contributor") or "contributor"
    )
    project_list = ", ".join(target_projects)
    return f"""

═══════════════════════════════════════════════════════
MAINTAINERSHIP-LEVEL EVALUATION (named-project mode)
═══════════════════════════════════════════════════════

This brief names specific target projects: {project_list}. The recruiter wants candidates classified at maintainership level: "{desired_level}". A separate classifier produces a MAINTAINERSHIP EVIDENCE section in the candidate evidence below — when present, weigh it as authoritative for these named projects (not generic OSS prestige).

Maintainership level interpretation:
- contributor: meaningful merged work on the project; not a trusted reviewer / merger.
- maintainer: holds review authority; evidenced by merged-by signals, CONTRIBUTORS / MAINTAINERS file mentions, and sustained reviewer activity.
- project_lead: holds direction-setting authority; evidenced by GOVERNANCE.md mentions, README lead designation, or being the consistent release tag author.

When the candidate's classified or declared level meets or exceeds the brief's "{desired_level}" requirement on a named project, treat it as a STRONG positive in Step 1's capability mapping (DIRECT match) and Step 2's depth test (BUILDER). When it falls below, the rest of the evaluation proceeds normally — maintainership is a positive lift, not a hard gate, because the recruiter may still want adjacent contributors who could grow into maintainership.

A "budget exhausted" note in MAINTAINERSHIP EVIDENCE means the classifier hit its API cap before all signals scored — partial evidence; weigh it as conservative-floor rather than ceiling."""


# GitHub-local deliberately: Brief.is_senior_role() (shared/brief_schema.py) encodes
# LinkedIn's L7+/DIRECTOR semantics and rejects "principal"/"staff". The GitHub judge
# needs the IC-track senior bands too. Divergence is intentional — do not unify.
_SENIOR_ROLE_LEVEL_RE = re.compile(
    r"\b(principal|staff|distinguished|senior|sr\.?|lead|director|vp|head)\b",
    re.IGNORECASE,
)
_JUNIOR_MARKER_RE = re.compile(
    r"\b(junior|jr\.?|intern(?:ship)?|entry[- ]level|graduate|associate|mid(?:[- ]level|[- ]senior)?)\b",
    re.IGNORECASE,
)


def _github_senior_standard_applies(brief: Brief) -> bool:
    level = str(getattr(brief, "role_level", "") or "")
    if _JUNIOR_MARKER_RE.search(level):
        return False
    try:
        years = float(getattr(brief, "minimum_years_experience", 0) or 0)
    except (TypeError, ValueError):
        years = 0
    return years >= 7 or bool(_SENIOR_ROLE_LEVEL_RE.search(level))


def _assemble_seniority_block(brief: Brief) -> str:
    if not _github_senior_standard_applies(brief):
        return ""

    role_level = str(getattr(brief, "role_level", "") or "")
    try:
        minimum_years = int(getattr(brief, "minimum_years_experience", 0) or 0)
    except (TypeError, ValueError):
        minimum_years = 0

    return f"""
═══════════════════════════════════════════════════════
SENIORITY EVIDENCE (this role is {role_level}; the bar is {minimum_years}+ years)
═══════════════════════════════════════════════════════

GitHub rarely states years of experience. Read seniority from the passive record, in this
order:
- ACCOUNT AGE and REPO ACTIVITY SPAN — the "Account age" line and the REPO ACTIVITY SPAN
  section. When those artifacts are absent, read the Account created date and the repos'
  created/pushed dates directly — absence of the summary lines is a rendering gap, not
  evidence of absent tenure. Sustained multi-year activity (not mere account existence) is
  the primary tenure signal. An account created recently, or a commit record concentrated
  in the last year or two, bounds the seniority this profile can support.
- MAINTAINER TENURE — years maintaining substantive projects, release cadence over time,
  stewardship of a project across versions.
- ENGINEERING MATURITY — release discipline, CI configuration, documentation quality,
  review activity on others' work. These are how senior engineers leave passive traces.
- EXPLICIT MARKERS when present — graduation years, student status, internship language,
  first-job signals. Student or early-career markers against a
  {minimum_years}+-year bar are disqualifying regardless of talent.
The standard for this role: evidence consistent with having OPERATED at {role_level}
scope — sustained multi-year building and ownership of consequential systems. "Talented
and early-career" is a REJECT for this role: name it strong-junior in CASE_AGAINST so the
recruiter can route them to junior requisitions.
"""


def _assemble_decision_standard_block(brief: Brief) -> str:
    if not _github_senior_standard_applies(brief):
        return (
            'The decision standard: would the hiring manager agree this person has the '
            'hands-on depth and quality instincts to learn the role? Not "already doing '
            'it at an elite specialist lab" — that\'s too high. Not "vaguely adjacent" '
            '— that\'s too low. "Has done hands-on work with enough depth to grow into '
            'this role, even if their current domain is different."'
        )

    role_level = str(getattr(brief, "role_level", "") or "")
    return (
        f"The decision standard: would the hiring manager agree this person has ALREADY "
        f"OPERATED at {role_level} depth and scope — not \"could grow into it.\" "
        f"Hands-on depth with a short track record is strong-junior: REJECT, with the "
        f"strong-junior note in CASE_AGAINST. Depth without currency (a strong record "
        f"that stops years ago) warrants lowered confidence and the staleness named in "
        f"CASE_AGAINST."
    )
