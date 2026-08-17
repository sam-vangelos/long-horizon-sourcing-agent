"""Brief loader — normalizes any brief JSON into a standard Brief dataclass.

Handles three brief formats:
1. Brazil FDL (old) — archetypes as dicts with save_signals/skip_signals
2. Head of AI Lab (old) — sweet_spot.archetypes as strings
3. V2 schema (new) — capability_areas, depth_distinction, non_fit_patterns, employer_signal_rules

For V2 briefs, loads into both the old Brief (for strategy.py/adaptation compat)
and the new brief_schema.Brief (for judgment_templates.py). The new brief is
stored as Brief._new_brief.
"""

from __future__ import annotations
import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from shared.retrieval_design import (
    derive_legacy_search_views,
    retrieval_design_from_payload,
    validate_retrieval_design,
)

KIT_BASE_URL = "https://search-kit-library.vercel.app/kit"

logger = logging.getLogger(__name__)


def _detach(value: Any) -> Any:
    """Return a deep copy of a calibration mirror value.

    Used at the compat-Brief construction site so the compat mirror cannot
    share mutable list/dataclass state with the structured `_new_brief`.
    """
    return copy.deepcopy(value)


def _normalize_non_fit_patterns(value: Any) -> list[dict[str, Any]]:
    """Coerce a raw non_fit_patterns value into a list of dicts.

    Tolerates both wire shapes the system produces: the conversational
    extractor emits an array of short strings, while the composer emits dicts
    (see :func:`shared.intake_conversation.composer._normalize_patterns`, which
    this mirrors). A bare string ``x`` becomes
    ``{"label": x.strip()[:80], "why_not": x.strip()}``. A dict is normalized to
    guarantee both ``label`` and ``why_not`` keys: ``label`` is derived from
    ``label``/``name`` (truncated to 80 chars), ``why_not`` from
    ``why_not``/``description``; each falls back to the other so neither can be
    missing, and a dict with no usable text for either is dropped (mirroring the
    composer's ``label and why_not`` guard). The dict's other keys
    (``description``, ``examples``, …) are preserved untouched. Falsy /
    non-str / non-dict elements are dropped. Returning dicts that always carry
    both keys lets every downstream subscript site (the ``NonFitPattern`` build
    and the ``noise_archetypes`` mapping) read ``nf["label"]`` and
    ``nf["why_not"]`` without a ``KeyError`` or ``TypeError``.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append({"label": text[:80], "why_not": text})
        elif isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or "").strip()[:80]
            why_not = str(item.get("why_not") or item.get("description") or "").strip()
            if not label and not why_not:
                continue
            d = dict(item)
            d["label"] = label or why_not
            d["why_not"] = why_not or label
            out.append(d)
    return out

# Stage-0 calibration validation (P9.5): which V2 fields are checked for
# presence. A gap only warns when the brief already claims calibration
# provenance (at least one of these fields is populated); an intake-born
# brief with none of them populated gets a single info line, not a
# warning.
_V2_CALIBRATION_REQUIRED_LIST_FIELDS = (
    "domain_verbs",
    "domain_depth_objects",
    "transferability_examples",
    "canonical_framework_patterns",
    "canonical_company_patterns",
    "canonical_title_patterns",
    "canonical_broad_patterns",
    "edge_case_patterns",
    "edge_case_company_patterns",
    "term_blacklist_categories",
    "abbreviation_collisions",
    "example_compounds",
)
_V2_CALIBRATION_REQUIRED_STR_FIELDS = (
    "sequencing_heuristics",
)


def _ensure_valid_retrieval_design(
    *,
    explicit_retrieval_design: Any,
    retrieval_design: Any,
) -> None:
    if not (isinstance(explicit_retrieval_design, dict) and retrieval_design.is_explicit()):
        return
    issues = validate_retrieval_design(retrieval_design)
    if not issues:
        return
    raise ValueError(
        "Invalid explicit retrieval_design: " + " | ".join(issues)
    )


def _is_v2_brief(raw: dict) -> bool:
    """Detect if a brief JSON uses the V2 schema (has capability_areas or core_areas)."""
    has_capability = "capability_areas" in raw or "core_areas" in raw
    return has_capability and "depth_distinction" in raw


@dataclass
class Brief:
    id: str
    role_title: str
    role_description: str
    kit_url: str
    linkedin_project: str
    linkedin_project_id: str
    minimum_bar: str
    archetypes: list[dict]
    noise_archetypes: list[dict]
    hard_skips: list[str]
    clear_skips_from_review: list[str]
    known_noise_patterns: list[dict]
    permanent_filters: dict
    save_instructions: dict
    experience_floor: dict
    search_priorities: list[str] = field(default_factory=list)
    noise_predictions: list[dict] = field(default_factory=list)
    # V2 brief fields surfaced for strategy formation
    market_density: str = ""  # "sparse" | "moderate" | "dense" — from V2 brief
    engagement_context: dict = field(default_factory=dict)
    key_terms_by_area: dict = field(default_factory=dict)  # {area_name: [terms]}
    candidate_register_terms_by_area: dict = field(default_factory=dict)  # {area_name: [terms]}
    # Lightweight brief fields — JD-driven mode
    jd_text: str = ""
    intake_notes: str = ""
    instructions: list[str] = field(default_factory=list)
    # P3a Stage B: who authored permanent_filters["Location"] — "operator"
    # (exact facet names; a miss aborts, never auto-resolves) or "preflight"
    # (model-extracted candidates; a typeahead miss gets ONE model resolution
    # against the real options). Empty = no geography.
    geography_source: str = ""
    employer_blacklist: list[str] = field(default_factory=list)
    additional_search_terms: list[str] = field(default_factory=list)
    retrieval_design: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    # --- Vertical-agnostic calibration mirror (Slice 1) ---
    # These fields mirror strategy-relevant calibration vocabulary from the
    # structured V2 brief so consumers like linkedin/strategy.py can read them
    # without spelunking _new_brief. All defaults are vertical-agnostic empties.
    domain_verbs: list[str] = field(default_factory=list)
    domain_depth_objects: list[str] = field(default_factory=list)
    transferability_examples: list[Any] = field(default_factory=list)
    worked_examples: list[Any] = field(default_factory=list)
    canonical_framework_patterns: list[str] = field(default_factory=list)
    canonical_company_patterns: list[str] = field(default_factory=list)
    canonical_title_patterns: list[str] = field(default_factory=list)
    canonical_broad_patterns: list[str] = field(default_factory=list)
    edge_case_patterns: list[str] = field(default_factory=list)
    edge_case_company_patterns: list[str] = field(default_factory=list)
    sequencing_heuristics: str = ""
    term_blacklist_categories: list[Any] = field(default_factory=list)
    abbreviation_collisions: list[Any] = field(default_factory=list)
    example_compounds: list[Any] = field(default_factory=list)
    domain_lane_hints: list[Any] = field(default_factory=list)
    # --- Executive Search module (Slice 1) ---
    # Mirrors of the executive-search V2 fields hydrated onto _new_brief.
    # Inert until later slices consume them; mirrored here so consumers
    # like Slice 6's confidentiality helpers and Slice 10's prior-search
    # exclusion can read from the compat Brief without spelunking
    # _new_brief. Dataclass-shaped fields default to None (consumers
    # null-guard); string/int fields default to the brief's "open" /
    # 180-day defaults so absent fields render as no-op.
    confidentiality_class: str = "open"
    prior_search: Any = None
    board_signals: Any = None
    executive_movement_window_days: int = 180
    executive_calibration: Any = None
    # --- OSS Maintainers module (Slice 2) ---
    # Mirrors of the OSS Maintainers top-level evaluation inputs onto
    # the compat Brief so consumers in `github/strategy.py` (Slice 7)
    # and `github/judgment_templates.py` (Slice 6 — imports the V2
    # Brief, but adapt-after-batch reads via the legacy Brief) can
    # read without spelunking _new_brief. Defaults match the V2
    # dataclass defaults so a brief without these fields renders as
    # behavior-preserving for classic github briefs (spec §11).
    target_projects: list[str] = field(default_factory=list)
    target_stacks: list[str] = field(default_factory=list)
    maintainership_level: str = "contributor"
    # --- Multi-module routing (Slice 2 substrate; partial mfm Slice 2) ---
    # Mirror of `_new_brief.target_modules` for compat consumers like
    # `linkedin/strategy.py` that take the compat Brief and need to
    # branch on whether the brief targets executive search (Slice 2's
    # `title_first` architecture bias) or any other module.
    target_modules: list[str] = field(default_factory=list)
    # V2 brief schema object (set when loading a V2 brief)
    _new_brief: Any = field(default=None, repr=False)

    def needs_preflight(self) -> bool:
        """Check if this brief needs Sourcing Preflight to fill eval criteria."""
        # V2 briefs never need preflight — they carry their own eval criteria
        if self._new_brief is not None:
            return False
        has_jd = bool(self.jd_text)
        has_archetypes = bool(self.archetypes)
        has_minimum_bar = bool(self.minimum_bar)
        return has_jd and (not has_archetypes or not has_minimum_bar)

    @property
    def has_v2_schema(self) -> bool:
        """Whether this brief was loaded from a V2 schema with structured evaluation."""
        return self._new_brief is not None

    def sourcing_signals(self) -> dict[str, list[str]]:
        """Capability-area names and non-fit labels for lightweight sourcing probes.

        Pulls from the structured V2 brief (``_new_brief``) — the compat Brief
        itself carries NEITHER ``capability_areas`` nor ``non_fit_patterns`` as
        attributes, so a ``getattr``-based read silently yields empties (this is
        the bug this accessor exists to prevent). For legacy / pre-V2 briefs
        there is no structured schema, so this returns clearly-empty lists
        rather than masking the absence behind a successful-looking read.

        Returns:
            ``{"capability_areas": [<name>, ...], "non_fit_patterns": [<label>, ...]}``
        """
        new_brief = self._new_brief
        if new_brief is None:
            # Legacy brief: no V2 schema. Return explicitly-empty, not silently-empty.
            return {"capability_areas": [], "non_fit_patterns": []}
        return {
            "capability_areas": [
                str(name).strip()
                for name in new_brief.capability_area_names()
                if str(name).strip()
            ],
            "non_fit_patterns": [
                str(nf.label).strip()
                for nf in new_brief.non_fit_patterns
                if str(getattr(nf, "label", "") or "").strip()
            ],
        }


def load_brief(path: str | Path) -> Brief:
    """Load a brief JSON file and return a normalized Brief dataclass."""
    with open(path) as f:
        raw = json.load(f)
    if _is_v2_brief(raw):
        return _load_v2_brief(raw)
    return normalize_brief(raw)


def _load_v2_brief(raw: dict) -> Brief:
    """Load a V2 brief: create the new brief_schema.Brief AND map to old Brief for compat."""
    from shared.brief_schema import Brief as NewBrief, CapabilityArea, DepthDistinction, \
        NonFitPattern, EmployerSignalRule, FacialCalibration, BiasControls, MarketDensity, \
        PostSaveModifier, TransferabilityExample, WorkedExample, BlacklistCategory, AbbreviationCollision, \
        ExampleCompound, DomainLaneHint, \
        ExecutiveCalibration, PriorSearchContext, BoardSignalRules

    # --- Build the new brief_schema.Brief ---
    # Normalize v3.1 field names: merge core_areas + differentiator_areas → capability_areas
    if "core_areas" in raw and "capability_areas" not in raw:
        raw["capability_areas"] = raw.get("core_areas", []) + raw.get("differentiator_areas", [])

    capability_areas = [
        CapabilityArea(
            name=ca["name"], description=ca["description"],
            builder_signals=ca.get("builder_signals", ca.get("positive_signals", [])),
            user_signals=ca.get("user_signals", ca.get("false_positive_signals", [])),
            key_terms=ca.get("key_terms", []),
            candidate_register_terms=ca.get("candidate_register_terms", []),
            github_code_signals=ca.get("github_code_signals", []),
        ) for ca in raw["capability_areas"]
    ]
    depth = DepthDistinction(
        builder_definition=raw["depth_distinction"]["builder_definition"],
        user_definition=raw["depth_distinction"]["user_definition"],
        edge_case_guidance=raw["depth_distinction"]["edge_case_guidance"],
    )
    # The conversational extractor emits non_fit_patterns as an array of short
    # strings (shared/intake_conversation/extractor.py); the composer emits
    # dicts. validate_v2_brief gates neither shape, and merge_extracted does a
    # wholesale replace with no coercion, so a string-shaped list can reach the
    # canonical loader and crash both subscript sites below. Normalize ONCE,
    # mirroring composer._normalize_patterns, and reuse the result everywhere.
    non_fit_patterns_raw = _normalize_non_fit_patterns(raw.get("non_fit_patterns", []))
    non_fit_patterns = [
        NonFitPattern(
            label=nf["label"],
            description=nf.get("description") or nf.get("why_not") or nf["label"],
            why_not=nf["why_not"],
            examples=nf.get("examples", []),
        ) for nf in non_fit_patterns_raw
    ]
    employer_rules = [
        EmployerSignalRule(
            tier=er["tier"], employer_patterns=er["employer_patterns"],
            evidence_required=er["evidence_required"],
            save_on_employer_alone=er.get("save_on_employer_alone", False),
        ) for er in raw.get("employer_signal_rules", [])
    ]
    fc_data = raw.get("facial_calibration", {})
    # P6 (Wave 2): keep the 0.25/0.55 fallback but make the value
    # attributable — consumers (BiasMonitor alerts, run-report calibration)
    # can now tell an authored band from the loader default.
    band_low = fc_data.get("expected_yes_rate_low")
    band_high = fc_data.get("expected_yes_rate_high")
    band_authored = band_low is not None and band_high is not None
    facial = FacialCalibration(
        # An explicit JSON null must take the default too — .get()'s default
        # only covers a MISSING key, and the preflight template ships null
        # placeholders the model is instructed to replace.
        expected_yes_rate_low=0.25 if band_low is None else band_low,
        expected_yes_rate_high=0.55 if band_high is None else band_high,
        # An explicit stamp in the JSON wins (e.g. "synthesis_default" from
        # source_packet_synthesis) — a synthesis-default band must not
        # launder into "preflight" just because values are present.
        band_source=str(fc_data.get("band_source") or "")
        or ("preflight" if band_authored else "loader_default"),
        fast_exit_patterns=fc_data.get("fast_exit_patterns", []),
        trajectory_yes_patterns=fc_data.get("trajectory_yes_patterns", []),
        trajectory_ambiguous_patterns=fc_data.get("trajectory_ambiguous_patterns", []),
        trajectory_no_patterns=fc_data.get("trajectory_no_patterns", []),
        github_fast_exit_patterns=fc_data.get("github_fast_exit_patterns", []),
        github_portfolio_yes_patterns=fc_data.get("github_portfolio_yes_patterns", []),
        github_portfolio_ambiguous_patterns=fc_data.get("github_portfolio_ambiguous_patterns", []),
        github_portfolio_no_patterns=fc_data.get("github_portfolio_no_patterns", []),
    )
    bc_data = raw.get("bias_controls", {})
    bias = BiasControls(
        max_consecutive_saves=bc_data.get("max_consecutive_saves", 5),
        max_consecutive_rejects=bc_data.get("max_consecutive_rejects", 20),
        parse_failure_alarm_rate=bc_data.get("parse_failure_alarm_rate", 0.03),
    )
    post_save_modifiers = [
        PostSaveModifier(
            name=psm["name"],
            trigger=psm.get("trigger", ""),
            if_present=psm.get("if_present", ""),
            if_absent=psm.get("if_absent", ""),
            signals=psm.get("signals", []),
        ) for psm in raw.get("post_save_modifiers", [])
    ]

    # --- Vertical-agnostic calibration vocabulary (Slice 1) ---
    # Parse the new V2 calibration fields once so we can hydrate both the
    # structured _new_brief and the compat Brief without drift.
    domain_verbs = list(raw.get("domain_verbs", []))
    domain_depth_objects = list(raw.get("domain_depth_objects", []))
    transferability_examples = [
        TransferabilityExample(
            result=te.get("result", ""),
            source_context=te.get("source_context", ""),
            target_context=te.get("target_context", ""),
            rationale=te.get("rationale", ""),
        ) for te in raw.get("transferability_examples", [])
    ]
    worked_examples = [
        WorkedExample(
            decision=we.get("decision", ""),
            profile=we.get("profile", ""),
            reasoning=we.get("reasoning", ""),
        ) for we in raw.get("worked_examples", [])
    ]
    canonical_framework_patterns = list(raw.get("canonical_framework_patterns", []))
    canonical_company_patterns = list(raw.get("canonical_company_patterns", []))
    canonical_title_patterns = list(raw.get("canonical_title_patterns", []))
    canonical_broad_patterns = list(raw.get("canonical_broad_patterns", []))
    edge_case_patterns = list(raw.get("edge_case_patterns", []))
    edge_case_company_patterns = list(raw.get("edge_case_company_patterns", []))
    sequencing_heuristics = raw.get("sequencing_heuristics", "") or ""
    term_blacklist_categories = [
        BlacklistCategory(
            label=bc.get("label", ""),
            rationale=bc.get("rationale", ""),
            terms=list(bc.get("terms", [])),
        ) for bc in raw.get("term_blacklist_categories", [])
    ]
    abbreviation_collisions = [
        AbbreviationCollision(
            abbreviation=ac.get("abbreviation", ""),
            expansion=ac.get("expansion", ""),
            standalone_allowed=bool(ac.get("standalone_allowed", False)),
            note=ac.get("note", ""),
        ) for ac in raw.get("abbreviation_collisions", [])
    ]
    example_compounds = [
        ExampleCompound(
            boolean=ec.get("boolean", ""),
            purpose=ec.get("purpose", ""),
            novelty_bucket=ec.get("novelty_bucket", ""),
        ) for ec in raw.get("example_compounds", [])
    ]
    def _lane_hint_patterns(value) -> list[str]:
        # A bare string must become [string], never list(str) — list("stripe")
        # is ["s","t","r","i","p","e"], and one-character patterns turn
        # infer_domain_lane's substring matching into a lane that swallows
        # every string (Codex review, Wave 1).
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return []
        return [str(p).strip() for p in value if str(p or "").strip()]

    domain_lane_hints = [
        DomainLaneHint(
            lane=dl.get("lane", ""),
            patterns=_lane_hint_patterns(dl.get("patterns", [])),
        ) for dl in raw.get("domain_lane_hints", []) if isinstance(dl, dict)
    ]
    # --- Executive Search module (Slice 1) ---
    # Hydrate exec_search V2 fields once. Defaults match the dataclass
    # defaults so a brief without these fields produces effectively-
    # empty instances downstream. The validator (validate_v2_brief)
    # has already gated `confidentiality_class` to the recognized enum.
    confidentiality_class_raw = raw.get("confidentiality_class") or "open"
    if not isinstance(confidentiality_class_raw, str):
        confidentiality_class_raw = "open"
    prior_search_raw = raw.get("prior_search") or {}
    if not isinstance(prior_search_raw, dict):
        prior_search_raw = {}
    prior_search = PriorSearchContext(
        ruled_out_urls=list(prior_search_raw.get("ruled_out_urls", []) or []),
        ruled_out_notes=str(prior_search_raw.get("ruled_out_notes") or ""),
        earlier_run_ids=list(prior_search_raw.get("earlier_run_ids", []) or []),
    )
    board_signals_raw = raw.get("board_signals") or {}
    if not isinstance(board_signals_raw, dict):
        board_signals_raw = {}
    board_signals = BoardSignalRules(
        relevant_board_companies=list(
            board_signals_raw.get("relevant_board_companies", []) or []
        ),
        relevant_executive_alumni_companies=list(
            board_signals_raw.get("relevant_executive_alumni_companies", []) or []
        ),
        adjacency_rationale=str(
            board_signals_raw.get("adjacency_rationale") or ""
        ),
    )
    executive_movement_window_days = raw.get("executive_movement_window_days", 180)
    if not isinstance(executive_movement_window_days, int):
        try:
            executive_movement_window_days = int(executive_movement_window_days)
        except (TypeError, ValueError):
            executive_movement_window_days = 180
    executive_calibration_raw = raw.get("executive_calibration")
    if isinstance(executive_calibration_raw, dict):
        executive_calibration = ExecutiveCalibration(
            sector=str(executive_calibration_raw.get("sector") or ""),
            stage=str(executive_calibration_raw.get("stage") or ""),
            pnl_scale_usd=str(executive_calibration_raw.get("pnl_scale_usd") or ""),
            register_notes=str(
                executive_calibration_raw.get("register_notes") or ""
            ),
        )
    else:
        executive_calibration = None

    # Slice 5: dossier-spend cap (recruiter-overridable; default
    # $200) + company-stage signal hints. Defensive coercion so a
    # malformed brief degrades to the default cap rather than
    # raising mid-load.
    dossier_spend_cap_raw = raw.get("dossier_spend_cap_usd", 200.0)
    try:
        dossier_spend_cap_usd = float(dossier_spend_cap_raw or 200.0)
    except (TypeError, ValueError):
        dossier_spend_cap_usd = 200.0
    if dossier_spend_cap_usd < 0.0:
        dossier_spend_cap_usd = 200.0
    company_stage_signals_raw = raw.get("company_stage_signals") or {}
    if not isinstance(company_stage_signals_raw, dict):
        company_stage_signals_raw = {}

    # --- OSS Maintainers module (Slice 2) ---
    # Hydrate top-level evaluation inputs: target_projects /
    # target_stacks / maintainership_level. The validator
    # (validate_v2_brief) has already gated maintainership_level
    # to the recognized enum and asserted both lists are
    # list-of-string when present. Defensive list / str coercion
    # here mirrors the Exec Search pattern above so a partially-
    # malformed brief still loads (the worst case is an empty
    # list / default level, which downgrades to classic-github
    # behavior per spec §11).
    target_projects = list(raw.get("target_projects", []) or [])
    target_projects = [p for p in target_projects if isinstance(p, str) and p]
    target_stacks = list(raw.get("target_stacks", []) or [])
    target_stacks = [s for s in target_stacks if isinstance(s, str) and s]
    maintainership_level_raw = raw.get("maintainership_level") or "contributor"
    if not isinstance(maintainership_level_raw, str) or not maintainership_level_raw:
        maintainership_level_raw = "contributor"

    # Multi-module routing (Slice 2 substrate). Defensive list / str
    # coercion: a brief that omits `target_modules` defaults to
    # ``[]`` so downstream consumers see "no modules declared" rather
    # than crash. Slice 2's `dossier_mode` derives from this list.
    target_modules = list(raw.get("target_modules", []) or [])
    target_modules = [m for m in target_modules if isinstance(m, str) and m]

    explicit_retrieval_design = raw.get("retrieval_design")
    retrieval_design = retrieval_design_from_payload(
        explicit_retrieval_design,
        legacy_search_priorities=raw.get("search_priorities", []),
        legacy_additional_search_terms=raw.get("additional_search_terms", []),
        role_title=raw.get("role_title", ""),
    )
    use_explicit_retrieval_design = (
        isinstance(explicit_retrieval_design, dict)
        and retrieval_design.is_explicit()
    )
    if use_explicit_retrieval_design:
        derived_search_priorities, derived_additional_search_terms = derive_legacy_search_views(
            retrieval_design
        )
        additional_search_terms = derived_additional_search_terms or raw.get("additional_search_terms", [])
        search_priorities = derived_search_priorities or raw.get("search_priorities", [])
    else:
        additional_search_terms = raw.get("additional_search_terms", [])
        search_priorities = raw.get("search_priorities", [])
    _ensure_valid_retrieval_design(
        explicit_retrieval_design=explicit_retrieval_design,
        retrieval_design=retrieval_design,
    )

    engagement_context_raw = raw.get("engagement_context")
    engagement_context = (
        dict(engagement_context_raw)
        if isinstance(engagement_context_raw, dict)
        else {}
    )
    authored_posture = str(
        engagement_context.get("selectivity_posture") or ""
    ).strip().lower()
    if authored_posture not in {"selective", "coverage"}:
        logger.warning(
            "V2 brief %r is missing engagement_context selectivity posture; "
            "using the market-density compatibility posture.",
            raw.get("role_title", ""),
        )
    raw_market_density = raw.get("market_density", "moderate")
    try:
        market_density = MarketDensity(raw_market_density)
    except (TypeError, ValueError):
        # Historical/hand-authored values outside the bounded enum use the
        # documented unknown-density posture rather than failing to load.
        market_density = MarketDensity.MODERATE

    new_brief = NewBrief(
        role_title=raw["role_title"],
        role_level=raw.get("role_level", ""),
        role_summary=raw.get("role_summary", ""),
        geography=raw.get("geography", ""),
        linkedin_project=raw.get("linkedin_project", ""),
        capability_areas=capability_areas,
        depth_distinction=depth,
        non_fit_patterns=non_fit_patterns,
        employer_signal_rules=employer_rules,
        minimum_years_experience=raw.get("minimum_years_experience", 4),
        # RC4: band ceiling — None (no ceiling) unless the brief carries one.
        maximum_years_experience=(
            raw.get("maximum_years_experience")
            if isinstance(raw.get("maximum_years_experience"), int)
            and not isinstance(raw.get("maximum_years_experience"), bool)
            else None
        ),
        maximum_years_experience_is_hard=(
            raw.get("maximum_years_experience_is_hard")
            if isinstance(raw.get("maximum_years_experience_is_hard"), bool)
            else False
        ),
        experience_measure=str(raw.get("experience_measure", "") or ""),
        transferable_fundamentals_bar=str(
            raw.get("transferable_fundamentals_bar", "") or ""
        ),
        facial_ambiguity_posture=str(raw.get("facial_ambiguity_posture", "") or ""),
        minimum_bar_description=raw.get("minimum_bar_description", ""),
        facial_calibration=facial,
        market_density=market_density,
        engagement_context=engagement_context,
        employer_blacklist=raw.get("employer_blacklist", []),
        kit_url=raw.get("kit_url"),
        jd_path=raw.get("jd_path"),
        bias_controls=bias,
        inferential_save_rules=raw.get("inferential_save_rules"),
        non_fit_override_rule=raw.get("non_fit_override_rule", ""),
        calibration_examples=raw.get("calibration_examples"),
        instructions=raw.get("instructions", []),
        post_evaluation_overrides=str(raw.get("post_evaluation_overrides", "") or ""),
        post_save_modifiers=post_save_modifiers,
        additional_search_terms=additional_search_terms,
        retrieval_design=retrieval_design,
        domain_verbs=domain_verbs,
        domain_depth_objects=domain_depth_objects,
        transferability_examples=transferability_examples,
        worked_examples=worked_examples,
        canonical_framework_patterns=canonical_framework_patterns,
        canonical_company_patterns=canonical_company_patterns,
        canonical_title_patterns=canonical_title_patterns,
        canonical_broad_patterns=canonical_broad_patterns,
        edge_case_patterns=edge_case_patterns,
        edge_case_company_patterns=edge_case_company_patterns,
        sequencing_heuristics=sequencing_heuristics,
        term_blacklist_categories=term_blacklist_categories,
        abbreviation_collisions=abbreviation_collisions,
        example_compounds=example_compounds,
        domain_lane_hints=domain_lane_hints,
        version=raw.get("version", "2.0"),
        author=raw.get("author", ""),
        notes=raw.get("notes", ""),
        confidentiality_class=confidentiality_class_raw,
        prior_search=prior_search,
        board_signals=board_signals,
        executive_movement_window_days=executive_movement_window_days,
        executive_calibration=executive_calibration,
        dossier_spend_cap_usd=dossier_spend_cap_usd,
        company_stage_signals=dict(company_stage_signals_raw),
        target_projects=target_projects,
        target_stacks=target_stacks,
        maintainership_level=maintainership_level_raw,
        target_modules=target_modules,
    )

    # --- Map V2 fields to old Brief for strategy.py / adaptation compat ---
    # capability_areas → archetypes
    archetypes = []
    for ca in raw["capability_areas"]:
        archetypes.append({
            "name": ca["name"],
            "capability_area": ca["name"],
            "pattern": ca["description"],
            "save_signals": ca.get("builder_signals", ca.get("positive_signals", [])),
            "skip_signals": ca.get("user_signals", ca.get("false_positive_signals", [])),
        })

    # non_fit_patterns → noise_archetypes (reuse the normalized list built
    # above so this site cannot crash on a string-shaped element either).
    noise_archetypes = []
    for nf in non_fit_patterns_raw:
        noise_archetypes.append({
            "name": nf["label"],
            "description": nf.get("description") or nf.get("why_not") or nf["label"],
            "signals": nf.get("examples", []),
        })

    # hard_skips from fast_exit_patterns
    hard_skips = fc_data.get("fast_exit_patterns", [])

    # minimum_bar from minimum_bar_description
    min_bar = raw.get("minimum_bar_description", "")
    min_years = raw.get("minimum_years_experience", 4)
    if min_bar and min_years:
        min_bar = f"{min_years}+ years. {min_bar}"

    # permanent_filters from geography. Two shapes are legitimate (Codex
    # review, Wave 1): a plain string (operator-authored / preflight override)
    # and the preflight v2 structured object {"facet_candidates": [...],
    # "rationale": ...}. The structured shape must NOT be stringified into
    # the Location facet — the fail-closed geography gate would then try to
    # apply a facet literally named "{'facet_candidates': ...}" and abort a
    # run whose JD had no geography at all. Empty candidates → no Location.
    permanent_filters = {}
    geography_source = ""
    geography_raw = raw.get("geography")
    if isinstance(geography_raw, str) and geography_raw.strip():
        permanent_filters["Location"] = geography_raw.strip()
        # P3a Stage B: a string shape is an exact facet name — operator-
        # authored or the operator override merged over preflight (the pin).
        geography_source = "operator"
    elif isinstance(geography_raw, dict):
        facet_candidates = [
            str(v).strip()
            for v in (geography_raw.get("facet_candidates") or [])
            if str(v or "").strip()
        ]
        if facet_candidates:
            permanent_filters["Location"] = "; ".join(facet_candidates)
            # Model-extracted candidates — eligible for the one-shot
            # typeahead resolution loop at the apply seam.
            geography_source = "preflight"

    # experience_floor
    experience_floor = {
        "required": f"{min_years}+ years hands-on",
        "disqualifying": "",
    }

    # Load JD text from jd_path if provided
    jd_text = raw.get("jd", "") or raw.get("jd_text", "")
    if not jd_text and raw.get("jd_path"):
        jd_file = Path(raw["jd_path"])
        if jd_file.exists():
            jd_text = jd_file.read_text()

    old_brief = Brief(
        id=raw.get("role_title", "v2-brief"),
        role_title=raw["role_title"],
        role_description=raw.get("role_summary", ""),
        kit_url=raw.get("kit_url", ""),
        linkedin_project=raw.get("linkedin_project", ""),
        linkedin_project_id=raw.get("linkedin_project_id", ""),
        minimum_bar=min_bar,
        archetypes=archetypes,
        noise_archetypes=noise_archetypes,
        hard_skips=hard_skips,
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters=permanent_filters,
        save_instructions={"destination": raw.get("linkedin_project", "")},
        experience_floor=experience_floor,
        geography_source=geography_source,
        employer_blacklist=raw.get("employer_blacklist", []),
        additional_search_terms=additional_search_terms,
        retrieval_design=retrieval_design.to_dict(),
        jd_text=jd_text,
        intake_notes=raw.get("intake_notes", ""),
        instructions=raw.get("instructions", []),
        search_priorities=search_priorities,
        market_density=new_brief.market_density.value if new_brief.market_density else "",
        engagement_context=_detach(engagement_context),
        key_terms_by_area={
            ca.name: ca.key_terms
            for ca in new_brief.capability_areas
            if ca.key_terms
        },
        candidate_register_terms_by_area={
            ca.name: ca.candidate_register_terms
            for ca in new_brief.capability_areas
            if ca.candidate_register_terms
        },
        domain_verbs=_detach(domain_verbs),
        domain_depth_objects=_detach(domain_depth_objects),
        transferability_examples=_detach(transferability_examples),
        worked_examples=_detach(worked_examples),
        canonical_framework_patterns=_detach(canonical_framework_patterns),
        canonical_company_patterns=_detach(canonical_company_patterns),
        canonical_title_patterns=_detach(canonical_title_patterns),
        canonical_broad_patterns=_detach(canonical_broad_patterns),
        edge_case_patterns=_detach(edge_case_patterns),
        edge_case_company_patterns=_detach(edge_case_company_patterns),
        # sequencing_heuristics is an immutable str — no detachment needed.
        sequencing_heuristics=sequencing_heuristics,
        term_blacklist_categories=_detach(term_blacklist_categories),
        abbreviation_collisions=_detach(abbreviation_collisions),
        example_compounds=_detach(example_compounds),
        domain_lane_hints=_detach(domain_lane_hints),
        confidentiality_class=confidentiality_class_raw,
        prior_search=_detach(prior_search),
        board_signals=_detach(board_signals),
        executive_movement_window_days=executive_movement_window_days,
        executive_calibration=_detach(executive_calibration) if executive_calibration is not None else None,
        target_projects=_detach(target_projects),
        target_stacks=_detach(target_stacks),
        maintainership_level=maintainership_level_raw,
        target_modules=_detach(target_modules),
        raw=raw,
        _new_brief=new_brief,
    )
    _validate_v2_calibration(new_brief, old_brief.id)
    return old_brief


def _validate_v2_calibration(new_brief: Any, brief_id: str) -> None:
    """Stage-0 calibration validator (P9.5).

    Fires only on V2 briefs (this helper is only called from `_load_v2_brief`).
    Legacy/old-format briefs never reach this code path.

    No intake producer emits these 13 calibration fields today (P9.3
    provenance stamping — a later slice — will let this detect
    machine-authored briefs directly; until then this uses the other
    detector the spec names: absence of the fields en bloc). Two cases:

    - NONE of the calibration fields are present: this is the ordinary
      shape of an intake-born, conversationally-composed brief. Not a
      defect — a single INFO line names what's absent, nothing more.
    - AT LEAST ONE calibration field is present: the brief is claiming
      calibration provenance (someone hand-authored calibration data for
      it), so a gap in that set is a real omission — WARN on what's
      still missing.

    This validator never raises; it is warning/info-only.
    """
    all_fields = list(_V2_CALIBRATION_REQUIRED_LIST_FIELDS) + list(
        _V2_CALIBRATION_REQUIRED_STR_FIELDS
    )

    missing: list[str] = []
    present_any = False
    for field_name in all_fields:
        if getattr(new_brief, field_name, None):
            present_any = True
        else:
            missing.append(field_name)

    if not missing:
        return
    if not present_any:
        logger.info(
            "V2 brief %r has no Stage-0 calibration fields (expected for "
            "intake-born briefs — no intake producer emits them today): %s",
            brief_id,
            ", ".join(missing),
        )
        return
    logger.warning(
        "V2 brief %r claims calibration provenance but is missing calibration fields (Stage 0 warning): %s",
        brief_id,
        ", ".join(missing),
    )


def normalize_brief(raw: dict) -> Brief:
    """Normalize an old-format brief dict into a Brief dataclass."""

    # --- ID ---
    brief_id = raw.get("name") or raw.get("brief_id") or "unknown"

    # --- Role title ---
    role_title = raw.get("role_title") or raw.get("project_name") or raw.get("name") or ""

    # --- Role description ---
    role_description = raw.get("description") or raw.get("role_summary") or ""

    # --- Kit URL ---
    kit_url = raw.get("kit_url") or ""
    if not kit_url:
        kit_id = raw.get("search_kit_id") or ""
        if kit_id:
            kit_url = f"{KIT_BASE_URL}/{kit_id}"

    # --- LinkedIn project ---
    linkedin_project = raw.get("linkedin_project") or raw.get("project_name") or ""
    linkedin_project_id = raw.get("linkedin_project_id") or ""

    # --- Minimum bar ---
    minimum_bar = raw.get("minimum_bar", "")
    if isinstance(minimum_bar, dict):
        minimum_bar = _minimum_bar_to_text(minimum_bar)

    # --- Archetypes ---
    archetypes = _normalize_archetypes(raw)

    # --- Noise archetypes ---
    noise_archetypes = raw.get("noise_archetypes", [])

    # --- Hard skips ---
    hard_skips = [str(s) for s in raw.get("hard_skips", [])]

    # --- Clear skips from review ---
    clear_skips_from_review = _normalize_clear_skips(raw.get("clear_skips_from_review", []))

    # --- Known noise patterns ---
    known_noise_patterns = raw.get("known_noise_patterns", [])

    # --- Permanent filters ---
    permanent_filters = raw.get("permanent_filters", {})

    # --- Save instructions ---
    save_instructions = raw.get("save_instructions", {})
    if not save_instructions:
        si = {}
        if raw.get("linkedin_project"):
            si["destination"] = raw["linkedin_project"]
        if raw.get("linkedin_project_id"):
            si["project_id"] = raw["linkedin_project_id"]
        save_instructions = si

    # --- Experience floor ---
    experience_floor = raw.get("evaluation", {}).get("experience_floor", {})
    if not experience_floor and isinstance(raw.get("minimum_bar"), dict):
        experience_floor = {
            "required": f"{raw['minimum_bar'].get('years_experience', '')} years",
            "disqualifying": raw["minimum_bar"].get("experience_note", ""),
        }

    # --- Strategy hints ---
    explicit_retrieval_design = raw.get("retrieval_design")
    retrieval_design = retrieval_design_from_payload(
        explicit_retrieval_design,
        legacy_search_priorities=raw.get("search_priorities", []),
        legacy_additional_search_terms=raw.get("additional_search_terms", []),
        role_title=role_title,
    )
    use_explicit_retrieval_design = (
        isinstance(explicit_retrieval_design, dict)
        and retrieval_design.is_explicit()
    )
    if use_explicit_retrieval_design:
        derived_search_priorities, derived_additional_search_terms = derive_legacy_search_views(
            retrieval_design
        )
        search_priorities = derived_search_priorities or raw.get("search_priorities", [])
        additional_search_terms = derived_additional_search_terms or raw.get("additional_search_terms", [])
    else:
        search_priorities = raw.get("search_priorities", [])
        additional_search_terms = raw.get("additional_search_terms", [])
    _ensure_valid_retrieval_design(
        explicit_retrieval_design=explicit_retrieval_design,
        retrieval_design=retrieval_design,
    )
    noise_predictions = raw.get("noise_predictions", [])

    # --- Lightweight brief fields (JD-driven mode) ---
    jd_text = raw.get("jd", "") or raw.get("jd_text", "")
    if jd_text and not jd_text.strip().startswith(("#", "*", "T", "W", "A")):
        jd_path_candidate = Path(jd_text)
        if jd_path_candidate.exists() and jd_path_candidate.suffix in (".md", ".txt"):
            jd_text = jd_path_candidate.read_text()
    # If no inline JD text, try loading from jd_path
    if not jd_text and raw.get("jd_path"):
        jd_file = Path(raw["jd_path"])
        if jd_file.exists():
            jd_text = jd_file.read_text()

    intake_notes = raw.get("intake_notes", "")
    instructions = raw.get("instructions", [])
    employer_blacklist = raw.get("employer_blacklist", [])

    return Brief(
        id=brief_id,
        role_title=role_title,
        role_description=role_description,
        kit_url=kit_url,
        linkedin_project=linkedin_project,
        linkedin_project_id=linkedin_project_id,
        minimum_bar=minimum_bar,
        archetypes=archetypes,
        noise_archetypes=noise_archetypes,
        hard_skips=hard_skips,
        clear_skips_from_review=clear_skips_from_review,
        known_noise_patterns=known_noise_patterns,
        permanent_filters=permanent_filters,
        save_instructions=save_instructions,
        experience_floor=experience_floor,
        search_priorities=search_priorities,
        noise_predictions=noise_predictions,
        jd_text=jd_text,
        intake_notes=intake_notes,
        instructions=instructions,
        # Legacy configs are operator-authored by definition (P3a Stage B).
        geography_source=(
            "operator"
            if str((permanent_filters or {}).get("Location", "") or "").strip()
            else ""
        ),
        employer_blacklist=employer_blacklist,
        additional_search_terms=additional_search_terms,
        retrieval_design=retrieval_design.to_dict(),
        raw=raw,
    )


def _minimum_bar_to_text(mb: dict) -> str:
    """Convert a minimum_bar dict into a readable text summary for Opus."""
    parts = []
    if mb.get("years_experience"):
        parts.append(f"{mb['years_experience']}+ years experience.")
    if mb.get("experience_note"):
        parts.append(mb["experience_note"])
    if mb.get("title_floor"):
        parts.append(f"Title floor: {mb['title_floor']}.")
    if mb.get("title_ceiling_note"):
        parts.append(mb["title_ceiling_note"])
    if mb.get("technical_depth"):
        parts.append(mb["technical_depth"])
    if mb.get("bfsi_domain"):
        parts.append(mb["bfsi_domain"])
    if mb.get("genai_fluency"):
        parts.append(mb["genai_fluency"])
    if mb.get("location"):
        parts.append(mb["location"])
    # Catch any keys not explicitly handled
    handled = {"years_experience", "experience_note", "title_floor", "title_ceiling_note",
               "technical_depth", "bfsi_domain", "genai_fluency", "location"}
    for k, v in mb.items():
        if k not in handled and isinstance(v, str) and v.strip():
            parts.append(v)
    return " ".join(parts)


def _normalize_archetypes(raw: dict) -> list[dict]:
    """Normalize archetypes to [{name, pattern, save_signals, skip_signals}]."""
    archetypes = raw.get("archetypes", [])
    if archetypes and isinstance(archetypes[0], dict) and "name" in archetypes[0]:
        # Already in standard format (Brazil brief)
        return archetypes

    # Head of AI Lab format: sweet_spot.archetypes is a list of strings
    sweet_spot = raw.get("sweet_spot", {})
    if isinstance(sweet_spot, dict):
        ss_archetypes = sweet_spot.get("archetypes", [])
        if ss_archetypes:
            return [
                {
                    "name": f"Sweet Spot Archetype {i + 1}",
                    "pattern": desc,
                    "save_signals": [],
                    "skip_signals": [],
                }
                for i, desc in enumerate(ss_archetypes)
                if isinstance(desc, str)
            ]

    return archetypes


def _normalize_clear_skips(raw_skips: list) -> list[str]:
    """Flatten clear_skips_from_review to a list of strings."""
    result = []
    for entry in raw_skips:
        if isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, dict):
            pattern = entry.get("pattern", "")
            reason = entry.get("reason", "")
            if pattern and reason:
                result.append(f"{pattern}: {reason}")
            elif pattern:
                result.append(pattern)
            elif reason:
                result.append(reason)
    return result
