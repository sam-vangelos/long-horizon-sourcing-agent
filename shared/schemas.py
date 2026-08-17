"""Data schemas for the sourcing pipeline. All pipeline objects as dataclasses with JSON serialization."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json
import logging

from shared.reconciliation_schemas import RecruiterActivitySnapshot


def _coerce_known_fields(cls, d: dict) -> dict:
    known = cls.__dataclass_fields__
    dropped = [k for k in d if k not in known]
    if dropped:
        logging.getLogger(__name__).debug(
            "%s.from_dict dropped unknown keys: %s", cls.__name__, sorted(dropped)
        )
    return {k: v for k, v in d.items() if k in known}


# ---------------------------------------------------------------------------
# Kit string (extracted from Search Kit Library)
# ---------------------------------------------------------------------------

@dataclass
class KitString:
    id: int
    block: str  # e.g., "Post-Training & RLHF"
    subblock: str  # "Concepts", "Methods", or "Tools"
    string_type: str  # "Recall" or "Precision"
    boolean: str  # The actual Boolean parenthetical

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> KitString:
        return cls(**_coerce_known_fields(cls, d))


# ---------------------------------------------------------------------------
# Execution plan (Opus strategy output)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionPlan:
    strategy_rationale: str
    noise_predictions: list[dict] = field(default_factory=list)
    generated_strings: list[dict] = field(default_factory=list)
    retrieval_families: list[dict] = field(default_factory=list)
    coverage_gaps: list[dict] = field(default_factory=list)
    # Sourcing-judgment kernel (P1): optional lane payloads; not required for execution
    sourcing_lanes: list[dict] = field(default_factory=list)
    search_hypotheses: list[dict] = field(default_factory=list)
    search_slices: list[dict] = field(default_factory=list)
    # Role-class strategy hints (P8): defaults only; generated_strings remain executable
    role_strategy_profile: str = ""
    role_strategy_metadata: dict = field(default_factory=dict)
    # Search architecture
    architecture: str = ""  # sniper|dragnet|titration|negative_space|company_first|title_first
    architecture_rationale: str = ""
    architecture_success_criteria: list[str] = field(default_factory=list)
    architecture_pivot_triggers: list[str] = field(default_factory=list)
    original_architecture: str = ""  # Set once at plan creation, never updated on pivot
    # D4: lane feedback consumption tracking
    consumed_feedback_ids: list[str] = field(default_factory=list)
    # P7 Stage B: deterministic plan-level warnings (e.g. lane collapse),
    # computed at strategy formation and carried to block reports so
    # adaptation sees them. Each entry: {"code", "message", ...detail}.
    plan_warnings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionPlan:
        return cls(
            strategy_rationale=d.get("strategy_rationale", ""),
            noise_predictions=d.get("noise_predictions", []),
            generated_strings=d.get("generated_strings", []),
            retrieval_families=d.get("retrieval_families", []),
            coverage_gaps=d.get("coverage_gaps", []),
            sourcing_lanes=d.get("sourcing_lanes", []),
            search_hypotheses=d.get("search_hypotheses", []),
            search_slices=d.get("search_slices", []),
            role_strategy_profile=d.get("role_strategy_profile", ""),
            role_strategy_metadata=d.get("role_strategy_metadata", {}),
            architecture=d.get("architecture", ""),
            architecture_rationale=d.get("architecture_rationale", ""),
            architecture_success_criteria=d.get("architecture_success_criteria", []),
            architecture_pivot_triggers=d.get("architecture_pivot_triggers", []),
            original_architecture=d.get("original_architecture", ""),
            consumed_feedback_ids=d.get("consumed_feedback_ids", []),
            plan_warnings=d.get("plan_warnings", []),
        )


# ---------------------------------------------------------------------------
# Block report (per-block summary sent to Opus for adaptation)
# ---------------------------------------------------------------------------

@dataclass
class BlockReport:
    block_name: str
    strings_run: int = 0
    strings_with_saves: int = 0
    total_results: int = 0
    total_saves: int = 0
    top_performers: list[dict] = field(default_factory=list)
    zero_save_string_ids: list[int] = field(default_factory=list)
    noise_patterns_observed: list[dict] = field(default_factory=list)
    new_signals: list[str] = field(default_factory=list)
    string_details: list[dict] = field(default_factory=list)
    search_intelligence_summary: dict = field(default_factory=dict)
    # P7 Stage B: plan-level deterministic warnings (lane collapse etc.),
    # threaded from ExecutionPlan.plan_warnings so adaptation sees them.
    plan_warnings: list[str] = field(default_factory=list)
    # Telemetry demotion (2026-07-04): the brief's expected
    # opens-for-full-eval band, rendered ONCE per block (it is session-level
    # calibration, not per-string state) so the per-string bias-context
    # rates below have their reference point.
    bias_expected_band: list[float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary_text(self) -> str:
        lines = [f'Batch "{self.block_name}" — {self.strings_run} strings complete.']
        for warning in self.plan_warnings:
            lines.append(f"- Plan warning: {warning}")
        if self.bias_expected_band and len(self.bias_expected_band) == 2:
            lines.append(
                "- Expected opens-for-full-eval band (brief calibration): "
                f"{self.bias_expected_band[0]:.0%}–{self.bias_expected_band[1]:.0%}"
            )
        lines.append(f"- {self.strings_run} strings run, {self.strings_with_saves} produced saves")
        lines.append(f"- {self.strings_run - self.strings_with_saves} strings produced zero results or all noise")
        if self.top_performers:
            top = ", ".join(
                f"String #{p['string_id']} \"{p.get('name', '')}\" ({p.get('saves', 0)} saves from {p.get('results', 0)} results)"
                for p in self.top_performers
            )
            lines.append(f"- Top performers: {top}")
        if self.zero_save_string_ids:
            lines.append(f"- Zero-save strings: {', '.join(f'#{sid}' for sid in self.zero_save_string_ids)}")
        if self.noise_patterns_observed:
            noise = ", ".join(
                f"{p['term']} → {p.get('collision', 'unknown')} ({p.get('count', '?')} occurrences)"
                for p in self.noise_patterns_observed
            )
            lines.append(f"- Noise patterns observed: {noise}")
        if self.new_signals:
            lines.append(f"- New signal observed: {', '.join(self.new_signals)}")
        if self.search_intelligence_summary:
            summary = self.search_intelligence_summary
            if summary.get("strings_with_precommit_experiments"):
                lines.append(
                    "- Pre-commit experiments: "
                    + ", ".join(f"#{sid}" for sid in summary["strings_with_precommit_experiments"])
                )
            if summary.get("strings_rescued_by_drift"):
                lines.append(
                    "- Drift rescues that recovered signal: "
                    + ", ".join(f"#{sid}" for sid in summary["strings_rescued_by_drift"])
                )
            if summary.get("proven_family_keys"):
                lines.append(
                    "- Proven families worth exploiting: "
                    + ", ".join(summary["proven_family_keys"])
                )
            if summary.get("proven_domain_lanes"):
                lines.append(
                    "- Proven lanes worth exploiting: "
                    + ", ".join(summary["proven_domain_lanes"])
                )
            if summary.get("dead_family_keys"):
                lines.append(
                    "- Dead families to demote: "
                    + ", ".join(summary["dead_family_keys"])
                )
            if summary.get("contaminated_family_keys"):
                lines.append(
                    "- Families with seniority contamination: "
                    + ", ".join(summary["contaminated_family_keys"])
                )
            if summary.get("contaminated_domain_lanes"):
                lines.append(
                    "- Lanes with seniority contamination: "
                    + ", ".join(summary["contaminated_domain_lanes"])
                )
        if self.string_details:
            lines.append("- Per-string breakdown:")
            for sd in self.string_details:
                status = f"{sd['saves']} saves" if sd['saves'] else "zero saves"
                # P7 Stage B markers render beside the lane (Codex review,
                # Wave 3): adaptation must see keep-but-flag, not a lane that
                # looks first-class.
                lane_note = ""
                if sd.get("undeclared_lane"):
                    lane_note = " (undeclared)"
                elif sd.get("domain_lane_raw"):
                    lane_note = f" (remapped from {sd.get('domain_lane_raw')})"
                metadata = (
                    f"family={sd.get('family_key', 'unknown')} "
                    f"novelty={sd.get('novelty_bucket', 'unknown')} "
                    f"lane={sd.get('domain_lane', 'general')}{lane_note}"
                )
                lines.append(
                    f"  #{sd['string_id']} [{status}, {sd['pages_reviewed']}p, {sd['result_count']} results, {metadata}]: "
                    f"{sd['boolean'][:150]}"
                )
                if "duplicates" in sd or "candidates" in sd:
                    lines.append(
                        f"    Seen: candidates={sd.get('candidates', 0)}, "
                        f"duplicates={sd.get('duplicates', 0)}, "
                        f"facial_yes={sd.get('facial_yes', 0)}, facial_no={sd.get('facial_no', 0)}"
                    )
                if sd.get("lint_warnings"):
                    codes = ", ".join(sd.get("lint_warning_codes") or [])
                    suffix = f" ({codes})" if codes else ""
                    lines.append(
                        f"    Lint warnings: {sd['lint_warnings']}{suffix}"
                    )
                if sd.get('notes'):
                    lines.append(f"    Notes: {sd['notes']}")
                bias = sd.get("bias")
                if bias:
                    signals = ", ".join(bias.get("fired_alert_types") or [])
                    bias_line = (
                        f"    Bias context: {bias.get('saves', 0)}/{bias.get('full_evals', 0)} "
                        f"full evals saved ({bias.get('save_rate', 0):.0%}); "
                        f"opens-for-full-eval {bias.get('opens_for_full_eval_rate', 0):.0%} "
                        f"over {bias.get('facial_n', 0)} triaged"
                    )
                    if signals:
                        bias_line += f"; signals: {signals}"
                    lines.append(bias_line)
                if sd.get('save_names'):
                    lines.append(f"    Saved: {', '.join(sd['save_names'])}")
                if sd.get('saved_profiles'):
                    saved_profiles = ", ".join(
                        f"{p.get('name', '?')} ({p.get('title', '')} @ {p.get('company', '')})"
                        for p in sd['saved_profiles'][:3]
                    )
                    lines.append(f"    Save profiles: {saved_profiles}")
                search_intelligence = sd.get("search_intelligence") or {}
                if search_intelligence:
                    clauses: list[str] = []
                    if search_intelligence.get("precommit_recovery_attempts_used"):
                        clauses.append(
                            f"precommit_recovery={search_intelligence['precommit_recovery_attempts_used']}"
                        )
                    if search_intelligence.get("drift_attempt_count"):
                        clauses.append(
                            f"drift_attempts={search_intelligence['drift_attempt_count']}"
                        )
                    if search_intelligence.get("family_signal_total") is not None:
                        clauses.append(
                            f"family_signal={search_intelligence.get('family_signal_total', 0)}"
                        )
                    if search_intelligence.get("family_saves_total") is not None:
                        clauses.append(
                            f"family_saves={search_intelligence.get('family_saves_total', 0)}"
                        )
                    best_variant = search_intelligence.get("best_variant") or {}
                    if best_variant.get("variant_id"):
                        clauses.append(
                            f"best_variant={best_variant.get('variant_kind', 'variant')}:{best_variant['variant_id']}"
                        )
                    drift_summary = search_intelligence.get("drift_rescue_summary") or {}
                    if drift_summary.get("outcome"):
                        clauses.append(f"drift_outcome={drift_summary['outcome']}")
                    if clauses:
                        lines.append(f"    Search intelligence: {', '.join(clauses)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adaptation response (Opus mid-run adjustments)
# ---------------------------------------------------------------------------

@dataclass
class AdaptationResponse:
    new_strings: list[dict] = field(default_factory=list)
    new_retrieval_families: list[dict] = field(default_factory=list)
    hypothesis_updates: list[dict] = field(default_factory=list)
    skip_remaining: list[dict] = field(default_factory=list)
    reorder: list[dict] = field(default_factory=list)
    noise_updates: list[dict] = field(default_factory=list)
    # Architecture pivot (optional — empty = no pivot)
    pivot_to_architecture: str = ""
    pivot_rationale: str = ""
    # P11.1: explicit "no adaptation needed this checkpoint" decision — a
    # valid outcome, distinct from an empty/malformed response (still a
    # parse failure, handled upstream in adapt_after_block's caller).
    no_change: bool = False
    # P11.2: model-requested adaptation cadence. Holds the RAW parsed value
    # until adapt_after_block clamps it to [2, 8] (or None if absent/invalid)
    # and stashes the pre-clamp value on next_checkpoint_after_requested.
    next_checkpoint_after: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AdaptationResponse:
        return cls(
            new_strings=d.get("new_strings", []),
            new_retrieval_families=d.get("new_retrieval_families", []),
            hypothesis_updates=d.get("hypothesis_updates", []),
            skip_remaining=d.get("skip_remaining", []),
            reorder=d.get("reorder", []),
            noise_updates=d.get("noise_updates", []),
            pivot_to_architecture=d.get("pivot_to_architecture", ""),
            pivot_rationale=d.get("pivot_rationale", ""),
            no_change=bool(d.get("no_change", False)),
            next_checkpoint_after=d.get("next_checkpoint_after"),
        )


# ---------------------------------------------------------------------------
# Stage 1 output: extracted from list view by cheap model
# ---------------------------------------------------------------------------

@dataclass
class CandidateSnippet:
    name: str
    headline: str
    current_title: str
    current_company: str
    location: str
    education_snippet: str
    profile_url: str
    source_string_id: int
    source_string_name: str
    page: int
    result_rank: int
    experience_entries: list[str] = field(default_factory=list)
    card_index: int = -1  # DOM position of <li> in ol.profile-list; -1 = unknown
    already_saved: bool = False  # True if card shows "Change stage" instead of "Save to pipeline"
    recruiter_activity: RecruiterActivitySnapshot | None = None
    novelty_pressure: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.recruiter_activity is None:
            payload["recruiter_activity"] = None
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> CandidateSnippet:
        payload = _coerce_known_fields(cls, d)
        payload["recruiter_activity"] = RecruiterActivitySnapshot.from_dict(
            payload.get("recruiter_activity")
        )
        return cls(**payload)


# ---------------------------------------------------------------------------
# Stage 3 output: extracted from full profile by cheap model
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    title: str
    company: str
    location: str = ""
    start: str = ""
    end: str = ""
    summary_bullets: list[str] = field(default_factory=list)


@dataclass
class Education:
    degree: str
    school: str
    field: str = ""
    start: str = ""
    end: str = ""


@dataclass
class CandidateProfileSummary:
    name: str
    profile_url: str
    headline: str
    experiences: list[Experience] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    skills_snippet: list[str] = field(default_factory=list)
    # Expanded LinkedIn About/Summary text. Trailing default preserves every
    # existing constructor and older runtime payload; extraction keeps the
    # section verbatim so sparse position bullets do not erase first-party
    # evidence the candidate chose to put at profile level.
    about: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> CandidateProfileSummary:
        exps = [Experience(**e) for e in d.get("experiences", [])]
        edus = [Education(**e) for e in d.get("education", [])]
        return cls(
            name=d["name"],
            profile_url=d["profile_url"],
            headline=d.get("headline", ""),
            experiences=exps,
            education=edus,
            skills_snippet=d.get("skills_snippet", []),
            about=d.get("about", ""),
        )


# ---------------------------------------------------------------------------
# Contact information (cross-source discovery shape)
# ---------------------------------------------------------------------------

@dataclass
class ContactInfo:
    """Discovered contact information for a candidate."""
    emails: list[str] = field(default_factory=list)  # deduplicated, noreply filtered
    linkedin_url: str = ""
    twitter_url: str = ""
    website: str = ""
    # OSS Maintainers Slice 8: provenance label for `linkedin_url`. The
    # cross-source identity resolver and the recruiter workspace use
    # this to band confidence in the cross-link without inventing a
    # numeric confidence channel. One of:
    # - "" (no LinkedIn URL discovered)
    # - "blog" — discovered via the GitHub profile's blog/website field
    #   (highest confidence; recruiter set the URL deliberately).
    # - "bio" — extracted from bio prose (medium; URL may be a passing
    #   reference like "ex-LinkedIn" rather than the candidate's own
    #   profile, so the bio-extractor requires a full URL match).
    # - "readme" — extracted from the profile README (medium; same
    #   reasoning as bio).
    # Spec §12: bio/readme-derived URLs carry lower confidence than
    # blog-field URLs.
    linkedin_url_source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Opus judgment output (used for both facial and full stages)
# ---------------------------------------------------------------------------

@dataclass
class OpusDecision:
    stage: str  # "facial" | "full"
    decision: str  # "FACIAL_YES" | "FACIAL_NO" | "SAVE" | "REJECT" | "REVIEW_INFERRED" | "REVIEW_FLAGGED"
    path: str  # "pedigree" | "direct_experience" | "none"
    # P5.4 + P6 (Wave 2): honest typing. V2 facial-stage decisions have no
    # underlying confidence signal at all (binary triage contract), and a
    # full-stage decision whose CONFIDENCE line failed to parse now carries
    # None + confidence_parse_failed on the FullEvaluationResult instead of
    # a fabricated 0.5. Every consumer of `.confidence` MUST tolerate None
    # (audited P5.4; the full-stage summary print grew its guard in P6).
    confidence: float | None
    rationale: str
    candidate_name: str
    profile_url: str
    post_save_modifier: str = "NONE"  # V4: which modifier fired, if any
    novelty_value: str = ""
    value_rationale: str = ""
    # P4: bounded non-save review evidence. Populated only when ``decision``
    # is in ``shared.contracts.NON_SAVE_REVIEW_DECISIONS``. Defaults are
    # empty so SAVE / REJECT decisions serialize byte-identically (see
    # ``to_dict`` below — empty review fields are filtered out).
    review_reason_code: str = ""
    review_structural_evidence: list[str] = field(default_factory=list)
    review_recommended_next_step: str = ""
    review_ambiguity_reason: str = ""
    # Structured full-judgment disposition currency. A save-family decision
    # carries exactly one outreach tier; a rejection carries exactly one
    # canonical reason; review/facial decisions carry neither.
    outreach_tier: str = ""
    reject_reason: str = ""
    # Structured, model-returned evidence used to reach a full-profile
    # judgment. This is semantic telemetry, not hidden chain-of-thought: it
    # contains the forced-tool/text-contract match, depth, transferability,
    # and case fields exactly as validated. Empty for facial/legacy decisions.
    semantic_evidence: dict[str, Any] = field(default_factory=dict)
    # P1.2 (save-pipeline integrity): actuator truth for SAVE-class
    # decisions. The orchestrator stamps {status, persisted,
    # already_present, failure_reason} from the SideEffectOutcome after
    # handle_save_decision, so page/string stats and reports can
    # distinguish "judge said SAVE" from "the save physically landed".
    # Empty for non-save decisions (filtered from to_dict below).
    save_outcome: dict[str, Any] = field(default_factory=dict)
    prompt_capture: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        out = {
            key: value
            for key, value in asdict(self).items()
            if key != "prompt_capture"
        }
        # P4: filter empty review_* fields so SAVE / REJECT serialization is
        # byte-identical to pre-P4 output (no new keys leak into the
        # terminal_payload_json of non-review decisions).
        for key in (
            "review_reason_code",
            "review_recommended_next_step",
            "review_ambiguity_reason",
            "outreach_tier",
            "reject_reason",
        ):
            if out.get(key) == "":
                out.pop(key, None)
        if out.get("review_structural_evidence") == []:
            out.pop("review_structural_evidence", None)
        if out.get("semantic_evidence") == {}:
            out.pop("semantic_evidence", None)
        if out.get("save_outcome") == {}:
            out.pop("save_outcome", None)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> OpusDecision:
        return cls(**_coerce_known_fields(cls, d))


# ---------------------------------------------------------------------------
# Glance assessment (page-level pre-filter)
# ---------------------------------------------------------------------------

@dataclass
class GlanceResult:
    action: str        # "proceed" | "reformulate"
    summary: str       # Human-readable page description
    confidence: float  # 0.0 to 1.0
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Search string tracking
# ---------------------------------------------------------------------------

@dataclass
class SearchString:
    id: int
    name: str
    boolean: str
    status: str = "queued"  # "queued" | "in_progress" | "done" | "skipped" | "error"
    result_count: int = 0
    pages_reviewed: int = 0
    saves: list[str] = field(default_factory=list)  # candidate names
    notes: str = ""
    block: str = ""  # Kit block name, e.g. "Post-Training & RLHF"
    subblock: str = ""  # "Concepts", "Methods", or "Tools"
    string_type: str = ""  # "Recall" or "Precision"
    # Facial triage stats (persisted for block-level aggregate computation)
    facial_yes_count: int = 0
    facial_no_count: int = 0
    # Own bucket distinct from YES. Ternary facial triage routes both classes
    # to full review, while preserving the ambiguity signal for calibration,
    # reporting, resume, and adaptation.
    facial_borderline_count: int = 0
    # Settled full-profile funnel. These are semantic judge outcomes, distinct
    # from facial opens and from physical LinkedIn save actuator truth.
    full_reviewed_count: int = 0
    full_outreach_count: int = 0
    full_review_count: int = 0
    full_reject_count: int = 0
    candidates_count: int = 0
    duplicates_count: int = 0
    # P3.2: the slice of duplicates_count that came from PRIOR-SESSION
    # suppression (candidate already terminal in runtime history) rather than
    # within-run overlap. Only same-epoch overlap feeds family exhaustion —
    # prior-session suppressions after a brief revision must not poison it.
    suppressed_prior_session_count: int = 0
    # Two-phase adaptation fields
    phase: str = "scout"  # "scout" | "paginate"
    original_boolean: str = ""  # The original Boolean before any refinements
    refinement_stack: list[str] = field(default_factory=list)  # Stack of applied Booleans (push=narrow, pop=broaden)
    # Strategy metadata for cross-run memory and novelty accounting
    family_key: str = ""
    novelty_bucket: str = ""
    domain_lane: str = ""
    # P7 Stage B: lane-validation markers. domain_lane_raw preserves the
    # model's original spelling when validation remapped domain_lane to a
    # declared lane (a remap is never silent); undeclared_lane marks a lane
    # outside the declared universe that was kept as a hypothesis.
    domain_lane_raw: str = ""
    undeclared_lane: bool = False
    # RC2 (2026-07-04 SPL RCA): who the saves actually were — title/company
    # exemplars captured at save time, so search memory records the
    # DISCOVERED pocket (refinement mutates boolean in place under the same
    # family key; without exemplars the discovery is erased at write time
    # and formation inherits "the generic family is covered").
    save_exemplars: list[dict] = field(default_factory=list)
    seniority_risk: str = ""
    title_bucket_risk: str = ""
    opening_eligible: Optional[bool] = None
    retrieval_recipe: dict = field(default_factory=dict)
    retrieval_hypothesis_ids: list[str] = field(default_factory=list)
    # Sourcing-judgment kernel (P1): lane metadata; empty lane_id preserves legacy behavior
    lane_id: str = ""
    lane_name: str = ""
    lane_intent: str = ""
    acquisition_mode: str = ""
    # Sourcing-judgment kernel: lane-level execution posture telemetry. This
    # records whether the lane was Boolean-led, structured-only, or hybrid; it is
    # separate from acquisition_mode, which is the concrete LinkedIn executor mode.
    search_posture: str = ""
    lane_snapshot: dict = field(default_factory=dict)
    # Sourcing-judgment kernel (P2 hop 2): structured filters lifted from the lane
    # compiler's query_payload onto the executable queue. A plain dict (not a
    # linkedin-specific type) keeps shared/ free of a linkedin import; the runtime
    # hydrates it into LinkedInStructuredFilters at the mutation boundary (hop 3).
    # Empty dict preserves legacy behavior.
    structured_filters: dict = field(default_factory=dict)
    # Phase 2 hop 4 (slice G): the active variant's execution surface — "boolean"
    # (keyword entry), "hybrid" (keyword + filters), or "structured_only" (filters
    # carry it, no keyword) — persisted onto the compat record by apply_shadow.
    # SearchString has no LinkedInSearchVariant, so a cross-process / worker-death
    # resume that reconstructs via bootstrap_experiment_state (NOT the in-memory
    # from_dict) would otherwise mint variants with surface="" and lose the slice-D
    # keyword suppression. Default "" preserves the keyword-led / legacy default — a
    # checkpoint written before slice G still loads (from_dict drops unknown keys and
    # the default fills the gap).
    surface: str = ""
    # Sourcing Quality Kernel (M1C): deterministic Boolean normalizer report attached
    # by producers/adaptation. Empty dict preserves the legacy all-keyword path while
    # allowing receipts to report normalization guard findings when present.
    boolean_normalization: dict = field(default_factory=dict)
    # P2.2 (actuator honesty): the last surface-apply receipt for this string —
    # applied/failed/unsupported dimensions, per-dimension value counts, and
    # fell_back_to_keyword — persisted so block reports, adaptation, and the run
    # report can condition verdicts on whether the structured filters physically
    # landed. Empty dict = never applied / keyword-only string.
    surface_receipt: dict = field(default_factory=dict)
    # P5 (Wave 2): deterministic Boolean lint report attached at queue build.
    # Error-severity findings never reach a SearchString (they block queueing
    # upstream); warning-severity findings ride along so block reports can
    # surface craft health to adaptation. Empty dict = linted clean, or the
    # string predates the wired lint (checkpoint compatibility).
    boolean_lint: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SearchString:
        return cls(**_coerce_known_fields(cls, d))


# ---------------------------------------------------------------------------
# Progress checkpoint
# ---------------------------------------------------------------------------

@dataclass
class Progress:
    brief_name: str
    strings: list[SearchString] = field(default_factory=list)
    candidates_saved: int = 0
    candidates_rejected: int = 0
    current_string_id: Optional[int] = None
    current_page: int = 0
    pending_block_name: str = ""
    pending_block_string_ids: list[int] = field(default_factory=list)
    pending_block_ready: bool = False
    pivot_count: int = 0  # Architecture pivots used this run

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> Progress:
        strings = [SearchString.from_dict(s) for s in d.get("strings", [])]
        return cls(
            brief_name=d["brief_name"],
            strings=strings,
            candidates_saved=d.get("candidates_saved", 0),
            candidates_rejected=d.get("candidates_rejected", 0),
            current_string_id=d.get("current_string_id"),
            current_page=d.get("current_page", 0),
            pending_block_name=d.get("pending_block_name", ""),
            pending_block_string_ids=d.get("pending_block_string_ids", []),
            pending_block_ready=d.get("pending_block_ready", False),
            pivot_count=d.get("pivot_count", 0),
        )

    @classmethod
    def from_file(cls, path: str) -> Progress:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())


# ---------------------------------------------------------------------------
# External candidate evidence (Perplexity-augmented context for full eval)
# ---------------------------------------------------------------------------
# Slice 1 of the perplexity-evidence-augmentation feature: types only, with no
# callers. Strict separation between sourced facts, model inferences, and
# unresolved ambiguities is the durable contract — it must survive normalization
# all the way to the final judge.

@dataclass
class EvidenceRef:
    """A single citation backing an external fact or inference."""

    url: str
    title: str = ""
    source_quality: str = "unknown"  # "high" | "medium" | "low" | "unknown"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EvidenceRef:
        return cls(**_coerce_known_fields(cls, d))


@dataclass
class ExternalFactBlock:
    """A topic-grouped block of sourced facts with citations."""

    topic: str
    facts: list[str] = field(default_factory=list)
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    source_quality: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExternalFactBlock:
        refs = [EvidenceRef.from_dict(r) for r in d.get("evidence_refs", [])]
        return cls(
            topic=d.get("topic", ""),
            facts=list(d.get("facts", [])),
            evidence_refs=refs,
            source_quality=d.get("source_quality", "unknown"),
        )


@dataclass
class ExternalInference:
    """Model-synthesized claim derived from sourced facts. Kept distinct from facts."""

    claim: str
    basis_refs: list[EvidenceRef] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExternalInference:
        refs = [EvidenceRef.from_dict(r) for r in d.get("basis_refs", [])]
        return cls(
            claim=d.get("claim", ""),
            basis_refs=refs,
            confidence=float(d.get("confidence", 0.0) or 0.0),
        )


@dataclass
class ExternalCandidateEvidence:
    """Normalized public-web evidence layer used to enrich first-party profile evidence."""

    trigger_reason: str
    identity_confidence: float
    profile_facts_used_for_matching: list[str] = field(default_factory=list)
    external_fact_blocks: list[ExternalFactBlock] = field(default_factory=list)
    external_inferences: list[ExternalInference] = field(default_factory=list)
    unresolved_ambiguities: list[str] = field(default_factory=list)
    do_not_use_for_judgment: list[str] = field(default_factory=list)
    raw_provider_model: str = ""
    normalizer_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExternalCandidateEvidence:
        return cls(
            trigger_reason=d.get("trigger_reason", ""),
            identity_confidence=float(d.get("identity_confidence", 0.0) or 0.0),
            profile_facts_used_for_matching=list(d.get("profile_facts_used_for_matching", [])),
            external_fact_blocks=[
                ExternalFactBlock.from_dict(b)
                for b in d.get("external_fact_blocks", [])
            ],
            external_inferences=[
                ExternalInference.from_dict(i)
                for i in d.get("external_inferences", [])
            ],
            unresolved_ambiguities=list(d.get("unresolved_ambiguities", [])),
            do_not_use_for_judgment=list(d.get("do_not_use_for_judgment", [])),
            raw_provider_model=d.get("raw_provider_model", ""),
            normalizer_model=d.get("normalizer_model", ""),
        )


@dataclass
class ExternalEvidenceFailure:
    """Typed failure result from the external evidence pipeline.

    This is *not* an exception. The provider and normalizer return it directly so
    that callers can fall back to the baseline path without unwinding the stack
    or coupling external-evidence quota errors to LinkedIn run-pause logic.
    """

    reason: str  # see allowed values in the slice 1 spec
    detail: str = ""
    provider: str = ""
    http_status: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExternalEvidenceFailure:
        status = d.get("http_status")
        return cls(
            reason=d.get("reason", "unknown"),
            detail=d.get("detail", ""),
            provider=d.get("provider", ""),
            http_status=int(status) if isinstance(status, int) else None,
        )


@dataclass
class TriggerDecision:
    """Output of the external-evidence trigger gate."""

    should_run: bool
    reason: str
    skip_reason: str = ""
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TriggerDecision:
        return cls(
            should_run=bool(d.get("should_run", False)),
            reason=d.get("reason", ""),
            skip_reason=d.get("skip_reason", ""),
            signals=dict(d.get("signals", {})),
        )
