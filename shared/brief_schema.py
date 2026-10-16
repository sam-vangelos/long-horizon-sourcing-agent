"""
Brief schema for the autonomous sourcing agent.

The Brief carries ALL role-specific parametric content. The evaluation templates,
adaptation heuristics, and reporting layer read from this schema — nothing role-specific
lives outside it. Swapping roles means swapping briefs, nothing else.

Integrates with existing brief_loader.py — the loader normalizes whatever JSON format
you hand it into this dataclass.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from shared.retrieval_design import RetrievalDesign


class MarketDensity(str, Enum):
    """How talent-dense the search geography/domain is. Controls pagination depth."""
    SPARSE = "sparse"        # Few plausible candidates per string. Paginate conservatively.
    MODERATE = "moderate"    # Normal distribution. Default pagination.
    DENSE = "dense"          # Many plausible candidates per string. Paginate deeper but watch for volume inversion.


@dataclass
class PostSaveModifier:
    """
    A confidence modifier that fires ONLY after a save decision has been made
    on the core capability areas. Cannot trigger a rejection — only adjusts
    confidence on already-saved candidates.
    """
    name: str                       # e.g. "Client-Facing / Forward-Deployed Delivery Experience"
    trigger: str                    # When this modifier fires
    if_present: str                 # What to do if present (boost)
    if_absent: str                  # What to do if absent (no adjustment)
    signals: list[str] = field(default_factory=list)  # Evidence signals


@dataclass
class CapabilityArea:
    """
    A single domain that defines part of the role's scope.
    The evaluation template maps every candidate against these.
    """
    name: str                       # e.g. "Post-Training Data & RLHF Pipelines"
    description: str                # 1-2 sentences: what work in this area looks like
    builder_signals: list[str]      # Specific evidence that someone BUILDS in this area
    user_signals: list[str]         # Specific evidence that someone USES outputs from this area
    key_terms: list[str] = field(default_factory=list)  # Terms that discriminate builders from users
    # Terms filtered through the Maximum-Inclusion question ("would a
    # qualified person plausibly WRITE this on their profile?") — the SEARCH
    # channel. key_terms stay the EVALUATION channel.
    candidate_register_terms: list[str] = field(default_factory=list)
    github_code_signals: list[str] = field(default_factory=list)


@dataclass
class NonFitPattern:
    """
    A common profile type that appears in search results, looks adjacent, but isn't a fit.
    Must describe WORK, not titles or keywords.
    """
    label: str                      # e.g. "Applied ML for business metrics"
    description: str                # What this person actually builds every day
    why_not: str                    # Why it doesn't connect to the role despite surface similarity
    examples: list[str] = field(default_factory=list)  # Concrete examples: "fraud detection at Nubank"


@dataclass
class EmployerSignalRule:
    """
    How much weight company name carries, and what additional evidence is required.
    """
    tier: str                       # "frontier_lab" | "strong_ai" | "general_tech" | "neutral"
    employer_patterns: list[str]    # Company names or patterns that fall in this tier
    evidence_required: str          # What ADDITIONAL evidence beyond employer is needed to save
    save_on_employer_alone: bool    # Whether employer + relevant title is sufficient (almost always False)


@dataclass
class DepthDistinction:
    """
    The single most important calibration point. Defines what "builder" vs "user"
    means for THIS specific role. Role-specific, not generic.
    """
    builder_definition: str         # What "building" means for this role
    user_definition: str            # What "using" means — the application layer
    edge_case_guidance: str         # How to handle borderline cases (e.g., MLOps that touches training)


@dataclass
class FacialCalibration:
    """
    Expected pass-through rates for the facial triage stage.
    Used for anomaly detection, not as thresholds.
    """
    expected_yes_rate_low: float    # Lower bound of healthy range (e.g., 0.25)
    expected_yes_rate_high: float   # Upper bound of healthy range (e.g., 0.55)
    fast_exit_patterns: list[str]   # Narrow, concrete list of obviously-out-of-scope work

    # Trajectory patterns — what career histories signal at the snippet level.
    # These are the highest-information field available at facial stage.
    trajectory_yes_patterns: list[str]       # Career patterns that favor YES
    # Patterns a snippet cannot resolve. How they are TREATED is the
    # template's call, driven by Brief.facial_ambiguity_posture: binary
    # templates require extra positive signal for YES; ternary templates
    # route them to FACIAL_BORDERLINE so full eval decides.
    trajectory_ambiguous_patterns: list[str]
    trajectory_no_patterns: list[str]         # Patterns that favor NO (only if entire history matches)

    github_fast_exit_patterns: list[str] = field(default_factory=list)
    github_portfolio_yes_patterns: list[str] = field(default_factory=list)
    github_portfolio_ambiguous_patterns: list[str] = field(default_factory=list)
    github_portfolio_no_patterns: list[str] = field(default_factory=list)

    # Provenance for the expected-yes-rate band. Values identify brief authoring
    # (``preflight``) or code-filled fallback (``loader_default`` / ``synthesis_default``).
    band_source: str = ""


@dataclass
class BiasControls:
    """
    Role-level tuning for the compounding bias controls.
    The controls themselves are in the orchestrator; these are the parameters.
    """
    max_consecutive_saves: int = 5          # Auto-pause string after N saves with no reject
    max_consecutive_rejects: int = 20       # Flag for review — may indicate prompt drift or bad string
    parse_failure_alarm_rate: float = 0.03  # Flag if parse failures exceed this % of evaluations


# ---------------------------------------------------------------------------
# Vertical-agnostic calibration vocabulary.
# These dataclasses are shape-only; domain-specific taxonomy belongs in brief data.
# ---------------------------------------------------------------------------


@dataclass
class TransferabilityExample:
    """A worked example showing whether one context's experience transfers to another."""
    result: str                  # "transfers" | "does_not_transfer"
    source_context: str          # The candidate's actual experience context
    target_context: str          # The role's target context
    rationale: str               # Why it does or does not transfer


@dataclass
class WorkedExample:
    """A fully worked candidate evaluation the model should imitate.

    The highest-leverage prompt lever: a concrete, brief-authored walk through a
    real decision so the model calibrates its own reasoning against it. Stays
    vertical-agnostic in the schema — the vertical specificity lives in the
    brief-authored content, never in code.
    """
    decision: str                # e.g. SAVE / REJECT / INFERENTIAL_SAVE / TRANSFERABLE_SAVE
    profile: str                 # The candidate sketch (the load-bearing signals)
    reasoning: str               # The step-by-step walk the model should imitate


@dataclass
class BlacklistCategory:
    """A category of terms that should be down-weighted or excluded in search planning."""
    label: str                              # Short human label for the category
    rationale: str                          # Why these terms cause noise
    terms: list[str] = field(default_factory=list)


@dataclass
class AbbreviationCollision:
    """An abbreviation that collides with unrelated meanings outside this role's domain."""
    abbreviation: str            # e.g. "P&A"
    expansion: str               # The intended expansion in this role's domain
    standalone_allowed: bool = False  # Whether the abbreviation alone is enough signal
    note: str = ""               # Optional handling note (geography, pairing rules, etc.)


@dataclass
class ExampleCompound:
    """A worked Boolean compound used as a planner exemplar."""
    boolean: str                 # The Boolean string itself
    purpose: str                 # What this string is meant to retrieve (broad recall, edge case, etc.)
    novelty_bucket: str = ""     # Optional explicit novelty bucket: "canonical" | "edge_case" | etc.


@dataclass
class DomainLaneHint:
    """An explicit lane label and the patterns that map onto that lane.

    Used for search-memory normalization, not strategy classification. Strategy
    already annotates strings with `family_key` / `novelty_bucket` / `domain_lane`;
    these hints exist so search memory can normalize explicit metadata rather
    than re-inferring lanes from hardcoded vocabulary.
    """
    lane: str                                # Canonical lane label (e.g. "distribution")
    patterns: list[str] = field(default_factory=list)  # Patterns that should map onto this lane



@dataclass
class ExecutiveCalibration:
    """Executive-register calibration extensions.

    Optional bag of fields tightly scoped to executive-search briefs.
    Populated by the V2 loader; later slices (2, 5, 8) consume specific
    fields. Fields default to empty so a brief without an executive
    calibration block hydrates to an effectively-empty instance.
    """
    sector: str = ""
    stage: str = ""
    pnl_scale_usd: str = ""
    register_notes: str = ""


@dataclass
class PriorSearchContext:
    """Prior-search exclusion context for executive searches.

    The recruiter can record candidates already approached or formally ruled out.
    ``CandidateLedger._merge_prior_search_exclusion`` folds ruled-out URLs into
    the acquisition exclusion set before evaluation.
    """
    ruled_out_urls: list[str] = field(default_factory=list)
    ruled_out_notes: str = ""
    earlier_run_ids: list[str] = field(default_factory=list)


@dataclass
class BoardSignalRules:
    """Recruiter-authored rules for board and executive-alumni adjacency.

    These values provide context for dossier evaluation and may guide
    off-LinkedIn signal acquisition.
    """
    relevant_board_companies: list[str] = field(default_factory=list)
    relevant_executive_alumni_companies: list[str] = field(default_factory=list)
    adjacency_rationale: str = ""


@dataclass
class Brief:
    """
    Complete brief for one sourcing search.
    Everything the evaluation templates, adaptation layer, and orchestrator need.
    """

    # --- Identity ---
    role_title: str                                 # e.g. "Junior Frontier Data Lead"
    role_level: str                                 # e.g. "IC4"
    role_summary: str                               # 2-3 sentence description of what the role does
    geography: str                                  # e.g. "Brazil"
    linkedin_project: str                           # LinkedIn Recruiter project name/ID for saves

    # --- Capability Areas (the anchors for Step 1 of evaluation) ---
    capability_areas: list[CapabilityArea]           # 3-7 domains that define scope

    # --- Depth Distinction (the anchor for Step 2 of evaluation) ---
    depth_distinction: DepthDistinction

    # --- Non-Fit Patterns (checked AFTER capability mapping, not before) ---
    non_fit_patterns: list[NonFitPattern]

    # --- Employer Signal Rules ---
    employer_signal_rules: list[EmployerSignalRule]

    # --- Minimum Bar ---
    minimum_years_experience: int                   # Hard floor for experience
    minimum_bar_description: str                    # What the minimum bar means in practice

    # --- Facial Triage Calibration ---
    facial_calibration: FacialCalibration

    # --- Search Configuration ---
    market_density: MarketDensity = MarketDensity.MODERATE
    # Search-specific context for full-profile judgment. Generated briefs require a
    # selectivity_posture; the empty default keeps briefs loadable while the renderer
    # applies the documented fallback.
    engagement_context: dict = field(default_factory=dict)
    # Optional experience ceiling. minimum_years_experience renders the lower bound;
    # this field makes an explicit upper bound enforceable. None keeps the band open.
    maximum_years_experience: Optional[int] = None
    # A ceiling is leveling context by default. It becomes a hard rejection
    # gate only when the operator explicitly declares that intent in the
    # authored brief/preflight guidance.
    maximum_years_experience_is_hard: bool = False
    # The band's measure identifies which years count, such as total career or
    # role-relevant experience. Brief lint requires it whenever a ceiling is set.
    experience_measure: str = ""
    # The judge's level envelope, rendered by
    # shared/judgment/templates.py:_level_envelope_or_default:
    # ``target_level_statement``, ``over_level_looks_like`` and
    # ``under_level_looks_like``. Empty = the default envelope derived from
    # ``role_level``.
    role_level_envelope: dict = field(default_factory=dict)
    # Path-B bar for caliber/transfer pools: the fundamentals evidence a
    # candidate with no direct capability-area match must show on the profile
    # to earn TRANSFERABLE_SAVE. Empty = the role keeps the generic
    # domain-connection test as its only transfer evidence.
    transferable_fundamentals_bar: str = ""
    # Facial ambiguity posture: "ternary" routes snippet-unresolvable
    # trajectories to FACIAL_BORDERLINE (full eval decides), "binary" keeps
    # ambiguity-favors-NO, "" defers to the config flag. Preflight sets this
    # from market density and how resolvable the role is at snippet level.
    facial_ambiguity_posture: str = ""
    employer_blacklist: list[str] = field(default_factory=list)
    # Employers whose alumni are never leads: a name here blocks anyone who
    # has EVER worked there, current or past. ``employer_blacklist`` keys on
    # the current employer only.
    alumni_blacklist: list[str] = field(default_factory=list)
    kit_url: Optional[str] = None
    jd_path: Optional[str] = None

    # --- Bias Controls ---
    bias_controls: BiasControls = field(default_factory=BiasControls)

    # --- V3 Extensions (optional — backwards compatible with V2 briefs) ---
    inferential_save_rules: Optional[dict] = None       # Conditions for saving sparse high-prior profiles
    non_fit_override_rule: str = ""                       # Rule for when evidence overrides non-fit patterns
    calibration_examples: Optional[dict] = None          # Strong/incorrect/borderline examples for evaluation
    instructions: list[str] = field(default_factory=list) # Role-specific instructions injected into prompt
    post_evaluation_overrides: str = ""                   # Brief-authored REJECT-override text (opt-in; see post_evaluation_safety_net)

    # --- V4 Extensions ---
    post_save_modifiers: list[PostSaveModifier] = field(default_factory=list)
    additional_search_terms: list[str] = field(default_factory=list)
    retrieval_design: RetrievalDesign = field(default_factory=RetrievalDesign)

    # --- Vertical-agnostic calibration vocabulary ---
    # Domain vocabulary lives in the brief; helpers below expose it to consumers.
    domain_verbs: list[str] = field(default_factory=list)
    domain_depth_objects: list[str] = field(default_factory=list)
    transferability_examples: list[TransferabilityExample] = field(default_factory=list)
    worked_examples: list[WorkedExample] = field(default_factory=list)

    canonical_framework_patterns: list[str] = field(default_factory=list)
    canonical_company_patterns: list[str] = field(default_factory=list)
    canonical_title_patterns: list[str] = field(default_factory=list)
    canonical_broad_patterns: list[str] = field(default_factory=list)
    edge_case_patterns: list[str] = field(default_factory=list)
    edge_case_company_patterns: list[str] = field(default_factory=list)

    sequencing_heuristics: str = ""
    term_blacklist_categories: list[BlacklistCategory] = field(default_factory=list)
    abbreviation_collisions: list[AbbreviationCollision] = field(default_factory=list)
    example_compounds: list[ExampleCompound] = field(default_factory=list)
    domain_lane_hints: list[DomainLaneHint] = field(default_factory=list)

    # --- Metadata ---
    version: str = "1.0"
    author: str = ""
    notes: str = ""

    # --- Executive Search module ---
    # Optional fields for confidentiality, prior-search exclusions, board signals,
    # movement windows, and executive-register calibration. ``confidentiality_class``
    # is enforced by ``shared/confidentiality.py``; the other fields feed their
    # respective exclusion and dossier-evaluation consumers.
    confidentiality_class: str = "open"
    prior_search: PriorSearchContext = field(default_factory=PriorSearchContext)
    board_signals: BoardSignalRules = field(default_factory=BoardSignalRules)
    executive_movement_window_days: int = 180
    executive_calibration: Optional[ExecutiveCalibration] = None

    # --- Executive Search spend and company-stage signals ---
    # The dossier cap bounds research spend; stage signals refine company queries.
    # An empty map leaves acquisition to candidate-derived company names.
    dossier_spend_cap_usd: float = 200.0
    company_stage_signals: dict = field(default_factory=dict)

    # --- OSS Maintainers module ---
    # Optional top-level inputs for the github source. A nonempty
    # ``target_projects`` enables maintainership evidence in ``to_evidence_text()``
    # and the full-evaluation prompt; ``target_projects`` and ``target_stacks``
    # also seed acquisition queries. ``maintainership_level`` carries the
    # recruiter's posture, and brief polish preserves ``target_projects``.
    #
    # These are evaluation inputs, while ``source_config.*`` is reserved for
    # save-destination semantics.
    target_projects: list[str] = field(default_factory=list)
    target_stacks: list[str] = field(default_factory=list)
    maintainership_level: str = "contributor"

    # --- Exa source module ---
    # Optional top-level acquisition inputs. The lane compiler combines target
    # firms, lanes, and title terms and lints them against ``employer_blacklist``;
    # archetype queries describe a firm-independent career shape. ``do_not_save``
    # is source-agnostic and every module must honor it. Empty inputs leave Exa
    # acquisition and its gate inactive; brief polish preserves these fields as a group.
    exa_target_firms: list[dict] = field(default_factory=list)
    exa_query_lanes: list[str] = field(default_factory=list)
    exa_title_terms: list[str] = field(default_factory=list)
    exa_archetype_queries: list[dict] = field(default_factory=list)
    do_not_save: list[dict] = field(default_factory=list)

    # --- Slate ordering ---
    # The brief's ``attainability`` block as authored; empty when absent.
    # ``shared.attainability.AttainabilityRules.from_brief`` applies defaults.
    attainability: dict = field(default_factory=dict)

    # --- Multi-module routing ---
    # ``target_modules`` declares the sources a brief may launch (for example,
    # ``["linkedin"]`` or ``["linkedin", "exec_search"]``). It is mirrored on
    # the V2 schema and ``BriefInfo``; this compatibility field keeps branching
    # on one value.
    target_modules: list[str] = field(default_factory=list)

    @property
    def dossier_mode(self) -> bool:
        """Whether this brief evaluates in dossier (two-paragraph) mode.

        True when ``"exec_search"`` is in ``target_modules``. Executive briefs
        use the same evaluation pipeline with a paragraph rationale instead of
        a one-line ``SUMMARY:``.
        """
        return "exec_search" in self.target_modules

    def capability_area_names(self) -> list[str]:
        """Convenience: list of just the capability area names for template injection."""
        return [ca.name for ca in self.capability_areas]

    def capability_area_block(self) -> str:
        """
        Formats capability areas for injection into evaluation prompts.
        Each area gets its name, description, and builder signals.
        """
        lines = []
        for i, ca in enumerate(self.capability_areas, 1):
            lines.append(f"{i}. {ca.name}")
            lines.append(f"   What it looks like: {ca.description}")
            lines.append(f"   Builder signals: {', '.join(ca.builder_signals)}")
            if ca.user_signals:
                lines.append(f"   NOT this (user signals): {', '.join(ca.user_signals)}")
            if ca.key_terms:
                lines.append(f"   Discriminating terms: {', '.join(ca.key_terms)}")
            lines.append("")
        return "\n".join(lines)

    def depth_block(self) -> str:
        """Formats the depth distinction for injection into evaluation prompts."""
        d = self.depth_distinction
        return (
            f"BUILDER: {d.builder_definition}\n"
            f"USER: {d.user_definition}\n"
            "UNKNOWN: The profile does not provide enough evidence to establish "
            "either builder ownership or affirmative application-layer use. "
            "Missing evidence must be UNKNOWN, not USER.\n"
            f"Edge cases: {d.edge_case_guidance}"
        )

    def non_fit_block(self) -> str:
        """Formats non-fit patterns for injection into evaluation prompts."""
        lines = []
        for nf in self.non_fit_patterns:
            examples_str = f" (e.g., {', '.join(nf.examples)})" if nf.examples else ""
            lines.append(f"- {nf.label}: {nf.description}{examples_str}")
            lines.append(f"  Why not: {nf.why_not}")
        return "\n".join(lines)

    def employer_signal_block(self) -> str:
        """Format employer signal rules, including the current-employer blacklist.

        The blacklist remains visible to full evaluation when the snippet has no
        current-company value. Scope is current-company; alumni remain eligible,
        except alumni of a company on the alumni blacklist, which is rendered
        after it. Deterministic gates enforce both lists before judgment; the
        prompt text is the judge's copy of the same rule.
        """
        lines = []
        blacklist = [str(entry).strip() for entry in self.employer_blacklist if str(entry).strip()]
        alumni = [str(entry).strip() for entry in self.alumni_blacklist if str(entry).strip()]
        if blacklist:
            lines.append(
                "EMPLOYER BLACKLIST — never save a candidate whose CURRENT "
                f"employer is one of: {', '.join(blacklist)}."
            )
            if alumni:
                lines.append(
                    "  Past employment at a blacklisted company is not "
                    "disqualifying by itself, unless the company is on the "
                    "alumni blacklist below; otherwise weigh that evidence "
                    "like any other."
                )
            else:
                lines.append(
                    "  Past employment at a blacklisted company is not "
                    "disqualifying by itself; weigh that evidence like any other."
                )
        if alumni:
            lines.append(
                "ALUMNI BLACKLIST — never save a candidate who has EVER worked at "
                f"one of: {', '.join(alumni)}, current or past, as an employee "
                "or a contractor."
            )
        for rule in self.employer_signal_rules:
            companies = ", ".join(rule.employer_patterns)
            lines.append(f"- [{rule.tier}] {companies}")
            lines.append(f"  Required evidence: {rule.evidence_required}")
            if rule.save_on_employer_alone:
                lines.append(f"  Note: Employer + relevant title is sufficient for this tier.")
        return "\n".join(lines)

    def fast_exit_block(self) -> str:
        """Formats fast exit patterns for the facial triage prompt."""
        return "\n".join(f"- {p}" for p in self.facial_calibration.fast_exit_patterns)

    def trajectory_yes_block(self) -> str:
        """Formats trajectory YES patterns for facial triage."""
        return "\n".join(f"- {p}" for p in self.facial_calibration.trajectory_yes_patterns)

    def trajectory_ambiguous_block(self) -> str:
        """Formats trajectory AMBIGUOUS patterns for facial triage."""
        return "\n".join(f"- {p}" for p in self.facial_calibration.trajectory_ambiguous_patterns)

    def trajectory_no_block(self) -> str:
        """Formats trajectory NO patterns for facial triage."""
        return "\n".join(f"- {p}" for p in self.facial_calibration.trajectory_no_patterns)

    def trajectory_yes_compact(self) -> str:
        """One-line version for batch facial prompt."""
        return "; ".join(self.facial_calibration.trajectory_yes_patterns)

    def trajectory_ambiguous_compact(self) -> str:
        """One-line version for batch facial prompt."""
        return "; ".join(self.facial_calibration.trajectory_ambiguous_patterns)

    def trajectory_no_compact(self) -> str:
        """One-line version for batch facial prompt."""
        return "; ".join(self.facial_calibration.trajectory_no_patterns)

    def non_fit_compact(self) -> str:
        """Compact version of non-fit patterns for batch facial prompt.
        Bullet format with why_not reasoning (the actionable part), drops description.
        """
        parts = []
        for nf in self.non_fit_patterns:
            parts.append(f"• {nf.label} — {nf.why_not}")
        return "\n".join(parts)

    def capability_area_names_inline(self) -> str:
        """Comma-separated capability area names for compact prompts."""
        return ", ".join(ca.name for ca in self.capability_areas)

    def inferential_save_block(self) -> str:
        """Formats inferential save conditions for the full evaluation prompt."""
        if not self.inferential_save_rules:
            return (
                "No role-specific inferential-save shortcut is defined. This is not an "
                "automatic rejection: evaluate the whole profile through Steps 1-4, use "
                "UNKNOWN when ownership or builder depth is not evidenced, and reject only "
                "when the available evidence is genuinely insufficient to justify outreach."
            )
        conditions = self.inferential_save_rules.get("conditions", [])
        lines = [self.inferential_save_rules.get("description", "")]
        lines.append("")
        lines.append("Conditions (if ANY of these are met, respond INFERENTIAL_SAVE):")
        for c in conditions:
            lines.append(f"  - {c}")
        return "\n".join(lines)

    def non_fit_override_rule_block(self) -> str:
        """Formats the non-fit override rule for the full evaluation prompt."""
        if not self.non_fit_override_rule:
            return "Non-fit patterns apply as stated. No override rule."
        return self.non_fit_override_rule

    def github_fast_exit_block(self) -> str:
        """Formats GitHub fast exit patterns for the facial triage prompt."""
        patterns = self.facial_calibration.github_fast_exit_patterns
        if not patterns:
            # Sensible defaults
            patterns = [
                "Profile has zero non-fork repositories AND no bio AND no profile README",
                "ALL repositories are unmodified forks of tutorials or course materials with no original commits",
                "Account is an organization, not an individual",
            ]
        return "\n".join(f"- {p}" for p in patterns)

    def github_portfolio_yes_block(self) -> str:
        """Formats GitHub portfolio YES patterns for facial triage."""
        patterns = self.facial_calibration.github_portfolio_yes_patterns
        if not patterns:
            patterns = [
                "Repos using role-relevant toolchain frameworks named in the brief's capability areas or discriminating skills",
                "Contributed to widely-depended-on open source projects in the role's domain",
                "Repos tagged with capability area topics from the brief",
                "Bio or profile README mentions domain-relevant research, production work, evaluation, or specific frameworks from the brief",
                "Published papers in ML (arxiv links in profile README or website)",
                "Personal website with ML project descriptions or research portfolio",
                "Repos with >50 stars in ML-relevant domains",
            ]
        return "\n".join(f"- {p}" for p in patterns)

    def github_portfolio_ambiguous_block(self) -> str:
        """Formats GitHub portfolio AMBIGUOUS patterns for facial triage."""
        patterns = self.facial_calibration.github_portfolio_ambiguous_patterns
        if not patterns:
            patterns = [
                "Python repos with generic topics but company field shows a known tech/AI company",
                "Sparse public profile but high follower count (>100) or starred high-signal repos in the role's domain",
                "Private-heavy account (high account age, few public repos, but what exists looks relevant)",
                "Website exists but content unclear from portfolio summary",
                "Mix of ML and non-ML repos — direction unclear without deeper analysis",
            ]
        return "\n".join(f"- {p}" for p in patterns)

    def github_portfolio_no_block(self) -> str:
        """Formats GitHub portfolio NO patterns for facial triage."""
        patterns = self.facial_calibration.github_portfolio_no_patterns
        if not patterns:
            patterns = [
                "ALL repos are web frontend only (React, Vue, Angular, HTML/CSS) with no ML signal",
                "ALL repos are DevOps/infrastructure only (Terraform, Ansible, Docker, Kubernetes) with no ML",
                "ALL repos are data analytics with no ML (SQL, Tableau, pandas for reporting)",
                "Profile is clearly a student with only coursework repos and no framework usage",
                "ALL repos are mobile development (iOS/Android) with no ML component",
            ]
        return "\n".join(f"- {p}" for p in patterns)

    def github_code_signals_block(self) -> str:
        """Formats GitHub code signals from capability areas for evaluation."""
        lines = []
        for ca in self.capability_areas:
            if ca.github_code_signals:
                lines.append(f"  {ca.name}: {', '.join(ca.github_code_signals)}")
        if not lines:
            return "(No GitHub-specific code signals defined in brief — use general capability area key_terms)"
        return "\n".join(lines)

    def discriminating_skills_examples(self) -> str:
        """
        Collects key_terms from all capability areas as examples of
        discriminating skills — terms too specific to list without hands-on experience.
        Used in the sparse profile check to upgrade inferential saves.
        """
        terms = []
        for ca in self.capability_areas:
            terms.extend(ca.key_terms[:3])  # Top 3 from each area
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)
        return ", ".join(unique[:15])  # Cap at 15 examples

    # --- Seniority-Aware Evaluation Methods ---

    def is_senior_role(self) -> bool:
        """Whether this role targets L7+ / executive-track candidates."""
        level = self.role_level.upper()
        for n in range(7, 15):
            if f"L{n}" in level:
                return True
        for kw in ("DIRECTOR", "VP", "HEAD", "EXECUTIVE"):
            if kw in level:
                return True
        return False

    def seniority_calibration_block(self) -> str:
        """Render trajectory-shape evidence according to role level.

        L7+ roles get the full three-tier executive calibration because Tier 2
        uses scope, budget, and team-size evidence. Other roles get
        ``_trajectory_shape_inference_block()``: a concise, role-agnostic
        supporting signal that cannot replace the depth test.
        """
        if not self.is_senior_role():
            return self._trajectory_shape_inference_block()
        return f"""
SENIORITY CALIBRATION ({self.role_level}):
At the executive level (L7+), the evidence hierarchy shifts. Senior leaders describe work at a higher abstraction level — "pioneering new paradigms in the field" and "translating R&D into products delivering $10M+ impact" rather than naming the specific tools and pipelines they built. This is expected, not a deficiency.

For L7+ candidates, apply THREE TIERS of evidence:

Tier 1 — Direct Evidence: Profile explicitly names tools, systems, production metrics, action verbs with specificity. Standard evaluation. Strongest signal when present.

Tier 2 — Contextual Inference: Title + company + scope + budget + team size make it overwhelmingly likely the person has the capability. "Head of the target function at a major enterprise, built global R&D teams, $12-15M budget, $10M+ monthly economic impact" → this person almost certainly drove the architecture decisions and built production systems in the brief's capability areas, even though their bullets describe organizational achievements. For L7+ candidates, Tier 2 evidence is CO-PRIMARY with Tier 1 — not a fallback.

Tier 3 — Trajectory Inference: No single position provides evidence, but the career arc does. a rigorous technical education → hands-on senior IC work → executive leadership of the target function → a top-tier organization in the brief's domain. The trajectory tells you the person has deep technical foundations and has led the capability at enterprise scale. For L8-L9 candidates, Tier 3 can independently support a SAVE when the trajectory is unambiguous.

You MUST state which tier supports each of your conclusions. Do not reject a candidate for lacking Tier 1 evidence when Tier 2 or Tier 3 evidence is strong."""

    def _trajectory_shape_inference_block(self) -> str:
        """Role-agnostic trajectory-shape inference (P5.2).

        The reasoning behind the L7+ Tier 3 framework above — a coherent
        career arc can itself be evidence, even when no single position
        is — does not depend on seniority. This is the same idea scoped
        to IC/L4-L6 roles: no executive vocabulary, no budget/scope/team-
        size language, and explicitly bounded to a supporting role so it
        cannot substitute for the depth test.
        """
        return """
TRAJECTORY-SHAPE INFERENCE:
Evidence for a save doesn't have to live inside a single position's bullets — the shape of the career arc itself can be evidence. If no individual role explicitly names a capability area, but the sequence of employers, titles, and transitions only makes sense as a coherent path through the brief's capability areas (each move a plausible next step building on the last, at companies or teams known for that work), treat that trajectory shape as a real, positive signal to weigh alongside Steps 1-2 — not a replacement for them. A trajectory shape with no position showing any hands-on construction work does not, by itself, clear the depth test in Step 2."""

    def executive_builder_block(self) -> str:
        """Executive builder calibration for L7+ roles. Empty string for IC roles."""
        if not self.is_senior_role():
            return ""
        return f"""
EXECUTIVE BUILDER CALIBRATION ({self.role_level}):
At the L7+ level, the builder/user distinction shifts. An executive who:
- Built and led the target function at an enterprise from scratch
- Hired and developed the technical team that builds the systems
- Set the technical direction and made architecture-level decisions
- Directed multi-million-dollar R&D budgets toward production systems
- Drove R&D-to-production translation with measurable business impact
- Holds a PhD or deep technical background that preceded their leadership career

...is a BUILDER at the organizational level. They build the MACHINE that builds the systems. The test is NOT "do they still write code" — it's "did they build and technically direct the capability this brief targets, and does their background demonstrate they have the depth to evaluate and critique technical work at the architecture level?"

Organizational builder verbs for L7+: built (teams/functions/organizations), established (governance, technical standards for the function), directed (R&D budgets, technical strategy), pioneered (new paradigms, architectural approaches), translated (R&D into production). These are BUILDER signals at this seniority, not USER signals.

Career trajectory as depth evidence: A career arc from deep technical education → hands-on research or engineering → engineering leadership → head of the target function demonstrates hands-on technical foundations that preceded the executive role. The person didn't start as a manager — they started as a builder and scaled. Weight this heavily."""

    def decision_matrix_block(self) -> str:
        """Returns the full decision matrix text, seniority-aware for L7+ roles."""
        # Path-B evidence definition: brief-authored fundamentals bar when the
        # role declares a transfer pool; the generic domain-connection test
        # otherwise. Mechanism stays role-agnostic — what counts as transfer
        # evidence is content, so it comes from the brief.
        transfer_evidence = (
            str(getattr(self, "transferable_fundamentals_bar", "") or "").strip()
            or "positive evidence connecting to at least one capability area's domain (its key terms, named environments, or adjacent artifacts appearing on the profile)"
        )
        base = f"""DECISION STANDARD:
Eligibility is not the decision. A candidate can map to every capability area and still be a REJECT because the case is not strong enough to spend outreach attention on. SAVE requires ALL of:
1. No hard gate failed (employer blacklist, operator-declared hard bounds, non-fit pattern without a qualifying override).
2. A capability case meeting the brief's stated evidence bar, built from specific, first-party, current-or-recent evidence (STALE-only evidence cannot carry a save; it argues transferability at most).
3. Depth BUILDER, or UNKNOWN with the rest of the case strong. USER is never saved — it is an affirmative finding of application-layer consumption, never inferred from silence.
4. Level ALIGNED (Step 4), or ABOVE/BELOW handled per Step 4's rules — an otherwise-exceptional case one rung under the role's register is INFERENTIAL_SAVE when current direct capability fit is present, and a REJECT that names UNDER_LEVEL when it is not. Never a silent REJECT.
5. Opportunity coherence COHERENT, or UNCLEAR with the rest of the case strong. INCOHERENT is not saved; reject it and name the coherence driver (reason INCOHERENT_MOVE), however exceptional the rest of the case reads.
6. Caliber SOLID or STRONG under the search's selectivity posture. Under a selective posture, a real-but-ordinary case is REJECT (reason BAR_ORDINARY): being a typical qualified member of a dense pool is the noise floor, not a save.
A transfer-path save (no direct capability match) additionally requires TRANSFERABLE methodology plus {transfer_evidence} — flag it TRANSFERABLE_SAVE.
Sparse profiles meeting the inferential conditions AND Steps 4-5 = INFERENTIAL_SAVE (confidence 0.35-0.50).
On REJECT, state REJECT_REASON as the single dominant code: HARD_GATE | NON_FIT | CAPABILITY_INSUFFICIENT | EVIDENCE_STALE | OVER_LEVEL | UNDER_LEVEL | INCOHERENT_MOVE | DEPTH_CONSUMER | BAR_ORDINARY | IDENTITY_UNRESOLVED. Use IDENTITY_UNRESOLVED when the profile cannot be read or the identity evidence contradicts itself — a record you cannot attribute to one person is a rejection, not a deferral.
Within saves, set OUTREACH_TIER PRIORITY when caliber is STRONG and the fit is direct and current — the profiles a recruiter should read first; otherwise STANDARD. Employer brand never gates the tier: a candidate at an unknown employer with STRONG caliber and current direct fit is exactly who PRIORITY exists to surface.

CONFIDENCE expresses the strength of the whole case, never a category label:
- 0.85-0.95: multiple independent, specific, current evidence sources and no unresolved case-against.
- 0.70-0.84: a solid case with one thinner dimension.
- 0.55-0.69: a real but mixed case — the floor for SAVE.
- INFERENTIAL_SAVE stays 0.35-0.50. If you cannot honestly place a save at 0.55 or above, it is not a save."""
        if self.is_senior_role():
            base += "\nFor this L7+ role, requirement 3 reads USER depth through the Executive Builder Calibration above before it disqualifies: an organizational builder with a technical foundation is BUILDER at this level."
        return base

    def post_evaluation_safety_net(self) -> str:
        """Render brief-authored post-evaluation overrides when the brief opts in.

        Save-favoring overrides belong in ``post_evaluation_overrides`` so the
        recruiter can see and control the policy; this method supplies no default.
        """
        text = str(getattr(self, "post_evaluation_overrides", "") or "").strip()
        if not text:
            return ""
        return f"\nPOST-EVALUATION OVERRIDES (from the brief):\n{text}"

    def calibration_block(self) -> str:
        """Formats calibration examples for injection into evaluation prompts."""
        if not self.calibration_examples:
            return ""
        lines = ["\nCALIBRATION EXAMPLES (use these to anchor your judgment):"]
        strong = self.calibration_examples.get("strong_saves", [])
        if strong:
            lines.append("\nStrong Saves (correct — these should be SAVED):")
            for ex in strong:
                lines.append(f"  - {ex['name']}: {ex['why']}")
        incorrect = self.calibration_examples.get("incorrect_saves", [])
        if incorrect:
            lines.append("\nIncorrect Saves (should have been REJECTED):")
            for ex in incorrect:
                lines.append(f"  - {ex['name']}: {ex['why']}")
        borderline = self.calibration_examples.get("borderline_verify", [])
        if borderline:
            lines.append("\nBorderline (verify carefully):")
            for ex in borderline:
                lines.append(f"  - {ex['name']}: {ex['why']}")
        return "\n".join(lines)

    def instructions_block(self) -> str:
        """Formats role-specific instructions for injection into evaluation prompts."""
        if not self.instructions:
            return ""
        lines = ["\nROLE-SPECIFIC INSTRUCTIONS:"]
        for i, inst in enumerate(self.instructions, 1):
            lines.append(f"{i}. {inst}")
        return "\n".join(lines)

    def post_save_modifiers_block(self) -> str:
        """Formats post-save modifiers for injection into evaluation prompts."""
        if not self.post_save_modifiers:
            return ""
        lines = ["\nPOST-SAVE MODIFIERS (apply ONLY after a SAVE/INFERENTIAL_SAVE/TRANSFERABLE_SAVE/SIGNAL_SAVE decision):"]
        lines.append("These modifiers CANNOT change a REJECT to a SAVE. They only adjust confidence on already-saved candidates.\n")
        for mod in self.post_save_modifiers:
            lines.append(f"MODIFIER: {mod.name}")
            lines.append(f"  Trigger: {mod.trigger}")
            lines.append(f"  If present: {mod.if_present}")
            lines.append(f"  If absent: {mod.if_absent}")
            if mod.signals:
                lines.append(f"  Signals to look for:")
                for sig in mod.signals:
                    lines.append(f"    - {sig}")
            lines.append("")
        lines.append("After evaluating all modifiers, report which (if any) fired in the POST_SAVE_MODIFIER response field.")
        return "\n".join(lines)

    def additional_search_terms_block(self) -> str:
        """Formats additional search terms for injection into strategy prompts."""
        if not self.additional_search_terms:
            return ""
        return ", ".join(self.additional_search_terms)

    def retrieval_design_block(self) -> str:
        """Compact retrieval-design summary for strategy/adaptation prompts."""
        if not self.retrieval_design or self.retrieval_design.is_empty():
            return ""
        lines = ["Layered retrieval design:"]
        for family in self.retrieval_design.families[:8]:
            lines.append(f"- {family.family_id}: {family.label}")
            if family.objective:
                lines.append(f"  Objective: {family.objective}")
            if family.entry_signals:
                lines.append(
                    "  Entry signals: "
                    + ", ".join(item.label for item in family.entry_signals[:4])
                )
            if family.capability_proxies:
                lines.append(
                    "  Capability proxies: "
                    + ", ".join(item.label for item in family.capability_proxies[:4])
                )
            if family.reality_filters:
                lines.append(
                    "  Reality filters: "
                    + ", ".join(item.label for item in family.reality_filters[:3])
                )
            if family.context_constraints:
                lines.append(
                    "  Context constraints: "
                    + ", ".join(item.label for item in family.context_constraints[:3])
                )
            if family.anti_noise:
                lines.append(
                    "  Anti-noise: "
                    + ", ".join(item.label for item in family.anti_noise[:3])
                )
            if family.hypothesis_ids:
                lines.append(
                    "  Edge-case overlays: "
                    + ", ".join(family.hypothesis_ids[:3])
                )
        if self.retrieval_design.edge_case_hypotheses:
            lines.append("Edge-case hypotheses:")
            for hypothesis in self.retrieval_design.edge_case_hypotheses[:5]:
                lines.append(
                    f"- {hypothesis.hypothesis_id}: {hypothesis.hidden_cohort} "
                    f"(why missed: {hypothesis.why_missed})"
                )
        return "\n".join(lines)

    def capability_area_stack_rank_guidance(self) -> str:
        """Dynamic stack-rank guidance based on actual number of capability areas."""
        n = len(self.capability_areas)
        if n <= 2:
            return (f"Areas are stack-ranked. Within the same match level (DIRECT or ADJACENT), "
                    f"score toward the TOP of the confidence range for area #1 and toward the BOTTOM for area #{n}. "
                    f"Example: ADJACENT to area #1 → 0.65-0.75; ADJACENT to area #{n} → 0.60-0.65. "
                    f"Never score below the range floor regardless of area rank.")
        top = n - 1
        return (f"Areas are stack-ranked. Within the same match level (DIRECT or ADJACENT), "
                f"score toward the TOP of the confidence range for areas ranked 1-{top} and toward the BOTTOM for area #{n}. "
                f"Example: ADJACENT to area #1 → 0.65-0.75; ADJACENT to area #{n} → 0.60-0.65. "
                f"Never score below the range floor regardless of area rank.")

    # ------------------------------------------------------------------
    # Vertical-agnostic calibration rendering helpers.
    # These helpers provide the stable schema surface used by judgment templates.
    # Templates own fallback phrasing when a field is empty; helpers return ""
    # for empty fields so interpolation does not create dangling headers.
    # ------------------------------------------------------------------

    def domain_verbs_block(self) -> str:
        """Comma-separated list of domain verbs; empty string when none."""
        if not self.domain_verbs:
            return ""
        return ", ".join(self.domain_verbs)

    def domain_depth_objects_block(self) -> str:
        """Bulleted list of objects that signal depth in this domain."""
        if not self.domain_depth_objects:
            return ""
        return "\n".join(f"- {obj}" for obj in self.domain_depth_objects)

    def transferability_examples_block(self, result: str | None = None) -> str:
        """Render transferability examples, optionally filtered by result.

        Args:
            result: When set to "transfers" or "does_not_transfer", only examples
                with that ``result`` value are rendered. When ``None``, every
                example is rendered with its result label.
        """
        if not self.transferability_examples:
            return ""
        examples = self.transferability_examples
        if result is not None:
            examples = [ex for ex in examples if ex.result == result]
        if not examples:
            return ""
        lines: list[str] = []
        for ex in examples:
            if result is None:
                header = f"- [{ex.result}] {ex.source_context} → {ex.target_context}"
            else:
                header = f"- {ex.source_context} → {ex.target_context}"
            lines.append(header)
            if ex.rationale:
                lines.append(f"  Rationale: {ex.rationale}")
        return "\n".join(lines)

    def worked_examples_block(self) -> str:
        """Render brief-authored worked examples the model should imitate.

        Returns "" when there are none, so the judgment-template helper
        (``_calibration_worked_examples_or_default``) falls through to its
        vertical-agnostic fallback and the assembled prompt stays byte-identical
        for briefs that do not populate this field. Never returns a header with
        no body.
        """
        if not self.worked_examples:
            return ""
        lines: list[str] = []
        for ex in self.worked_examples:
            lines.append(f"- [{ex.decision}] {ex.profile}")
            if ex.reasoning:
                lines.append(f"  Reasoning: {ex.reasoning}")
        return "\n".join(lines)

    def term_blacklist_block(self) -> str:
        """Render term-blacklist categories as a labeled bullet list."""
        if not self.term_blacklist_categories:
            return ""
        lines: list[str] = []
        for cat in self.term_blacklist_categories:
            terms = ", ".join(cat.terms) if cat.terms else ""
            header = f"- {cat.label}: {cat.rationale}" if cat.rationale else f"- {cat.label}"
            lines.append(header)
            if terms:
                lines.append(f"  Terms: {terms}")
        return "\n".join(lines)

    def abbreviation_collisions_block(self) -> str:
        """Render abbreviation collisions as a bullet list with handling guidance."""
        if not self.abbreviation_collisions:
            return ""
        lines: list[str] = []
        for ab in self.abbreviation_collisions:
            standalone = "standalone allowed" if ab.standalone_allowed else "pair with expansion"
            line = f"- {ab.abbreviation} → {ab.expansion} ({standalone})"
            lines.append(line)
            if ab.note:
                lines.append(f"  Note: {ab.note}")
        return "\n".join(lines)

    def example_compounds_block(self) -> str:
        """Render example Boolean compounds with their purpose / novelty bucket."""
        if not self.example_compounds:
            return ""
        lines: list[str] = []
        for ex in self.example_compounds:
            bucket = f" [{ex.novelty_bucket}]" if ex.novelty_bucket else ""
            lines.append(f"- {ex.purpose}{bucket}: {ex.boolean}")
        return "\n".join(lines)

    def domain_lane_hints_map(self) -> dict[str, list[str]]:
        """Lane → patterns mapping for explicit search-memory normalization.

        Returns an empty dict when no hints are configured. Used by search
        memory to normalize explicit lane metadata, not for strategy
        classification (strategy already owns novelty/lane assignment upstream).
        """
        if not self.domain_lane_hints:
            return {}
        return {hint.lane: list(hint.patterns) for hint in self.domain_lane_hints}

    def strategy_pattern_sets(self) -> dict[str, list[str]]:
        """Return canonical and edge-case pattern sets for strategy planning.

        Each key maps one configured list; unconfigured sets remain empty.
        """
        return {
            "canonical_framework": list(self.canonical_framework_patterns),
            "canonical_company": list(self.canonical_company_patterns),
            "canonical_title": list(self.canonical_title_patterns),
            "canonical_broad": list(self.canonical_broad_patterns),
            "edge_case": list(self.edge_case_patterns),
            "edge_case_company": list(self.edge_case_company_patterns),
        }
