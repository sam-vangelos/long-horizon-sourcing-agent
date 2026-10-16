"""Public V2 brief schema validation + legacy-merge surface.

Phase D Slice D-prep-C. Two callables Phase D's brief detail/edit
endpoint (D2) wraps:

- :func:`validate_v2_brief` raises :class:`BriefSchemaError` on invalid
  V2 input. The underlying parser is :func:`shared.brief_loader._load_v2_brief`;
  this module is the public API surface so D2's PUT path doesn't reach
  into private helpers.
- :func:`merge_legacy_brief` implements Fork B (master plan §"Phase D
  Execution Plan"): edit preserves all 45+ legacy keys; deprecated set
  lives in :data:`DEPRECATED_KEYS_BY_VERSION`. Returns ``(v2_data,
  deprecated_keys)`` so the UI can surface "drop from brief" affordances
  per deprecated key.

The deprecation manifest is intentionally a versioned constant rather
than a per-key flag — pruning happens via session iteration, not
free-form per-brief drift. New deprecations land here, not inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from shared.attainability import attainability_errors


# Versioned deprecation manifest. Each entry is a SCHEMA VERSION and the
# keys deprecated AT or BEFORE that version. The current canonical
# version is the highest key present.
#
# When we add a new deprecation: bump the version, list the keys, write
# a one-line rationale comment. Never delete a row; older briefs still
# need the historical context to show "this was deprecated at v2.1, drop
# from brief?" affordances.
DEPRECATED_KEYS_BY_VERSION: dict[str, dict[str, str]] = {
    # v2.0 — initial deprecation set surfaced during Phase D-prep audit.
    # Briefs in the catalog (config/*.json) carry 45+ legacy keys richer
    # than the V2 schema. The keys below are deprecated but preserved;
    # recruiter UI surfaces a "drop from brief" affordance per key.
    "2.0": {
        # Legacy authoring scaffolding — replaced by capability_areas.
        "must_haves": "Use `capability_areas[*].builder_signals` instead.",
        "nice_to_haves": "Use `capability_areas[*].user_signals` instead.",
        # Legacy save scaffolding — replaced by depth_distinction.
        "save_instructions": "Use `depth_distinction.builder_definition` for the criteria text.",
        "inference_save_rules": "Use `depth_distinction.edge_case_guidance` for borderline rules.",
        # Legacy search scaffolding — replaced by domain_*/canonical_* fields
        # on the V2 brief or by retrieval_design.
        "search_priorities": "Use `retrieval_design` and `canonical_*_patterns` instead.",
        "calibration_examples": "Use `transferability_examples` instead.",
        # Single-source flat field — Fork C migrates to nested layout
        # under config/<brief>/brief.json.
        "linkedin_project": "Use `source_config.linkedin.project_id` once authoring UI surfaces it.",
        # Legacy archetype list — replaced by capability_areas + non_fit_patterns
        # split. Kept legible during transition period.
        "archetypes": "Captured by `capability_areas` (positive) + `non_fit_patterns` (negative).",
        "noise_archetypes": "Use `non_fit_patterns` instead.",
    },
}

# Canonical V2 schema version — anything stricter goes in a new row above.
CURRENT_V2_VERSION = "2.0"

# Required V2 fields. A brief missing any of these is not a valid V2 brief
# and :func:`validate_v2_brief` raises.
REQUIRED_V2_KEYS = frozenset(
    {
        "capability_areas",
        "depth_distinction",
    }
)

SELECTIVITY_POSTURES = frozenset({"selective", "coverage"})


def normalize_generated_engagement_context(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Give a newly authored V2 brief canonical, run-stable context."""

    raw_context = data.get("engagement_context")
    context = dict(raw_context) if isinstance(raw_context, dict) else {}
    posture = str(context.get("selectivity_posture") or "").strip().lower()
    if posture not in SELECTIVITY_POSTURES:
        density = str(data.get("market_density") or "").strip().lower()
        posture = "coverage" if density == "sparse" else "selective"
    context["selectivity_posture"] = posture
    hiring_company = data.get("hiring_company")
    if isinstance(hiring_company, str):
        context["hiring_company"] = hiring_company
    data["engagement_context"] = context
    return context

# Recognized V2 fields (required + optional). Anything OUTSIDE this set
# AND outside the deprecated manifest is "unknown" — preserved on edit
# but flagged for recruiter review.
RECOGNIZED_V2_KEYS = frozenset(
    {
        "role_title",
        "role_level",
        # The judge's level envelope (target / above / below), rendered by
        # shared/judgment/templates.py:_level_envelope_or_default.
        "role_level_envelope",
        "role_summary",
        "role_description",
        "geography",
        "id",
        "kit_url",
        # Core V2 substrate (required + frequently-used optional).
        "capability_areas",
        "depth_distinction",
        "non_fit_patterns",
        "employer_signal_rules",
        "facial_calibration",
        "bias_controls",
        "market_density",
        "minimum_years_experience",
        "maximum_years_experience",
        "maximum_years_experience_is_hard",
        "experience_measure",
        "minimum_bar_description",
        "minimum_bar",
        # Multi-module (Phase F).
        "target_modules",
        "source_strategy",
        "source_config",
        # Legacy compat fields the loader still reads.
        "core_areas",
        "differentiator_areas",
        "post_save_modifiers",
        # Vertical-agnostic calibration mirror.
        "domain_verbs",
        "domain_depth_objects",
        "transferability_examples",
        "worked_examples",
        "canonical_framework_patterns",
        "canonical_company_patterns",
        "canonical_title_patterns",
        "canonical_broad_patterns",
        "blacklist_categories",
        "abbreviation_collisions",
        "example_compounds",
        "domain_lane_hints",
        # Retrieval design substrate.
        "retrieval_design",
        # Lightweight / JD-driven mode.
        "jd_text",
        "intake_notes",
        "instructions",
        "employer_blacklist",
        "alumni_blacklist",
        "excluded_locations",
        "employer_allowlist",
        # Preflight v2 structured outputs (P4, plans/sourcing-rigor-hardening.md).
        "hiring_company",
        "engagement_context",
        # Brief-authored post-evaluation override block (P8.3 — replaces the
        # code-injected L7+ safety net).
        "post_evaluation_overrides",
        "additional_search_terms",
        "search_priorities",
        "noise_predictions",
        # Executive Search module (Slice 1). Top-level fields, not under
        # `source_config.exec_search`, because per-knob settings are
        # eval/acquisition inputs rather than save-destination semantics.
        "confidentiality_class",
        "prior_search",
        "board_signals",
        "executive_movement_window_days",
        # Executive Search module (Slice 5). Per-search dossier-spend
        # cap (recruiter-overridable; default $200) and an optional
        # bag of company-stage signal hints used by the Crunchbase /
        # PitchBook adapters as query refinements.
        "dossier_spend_cap_usd",
        "company_stage_signals",
        # Designer module (Slice 1). Top-level structured rubric for
        # vision-LLM evaluation against recruiter-encoded principles.
        # Top-level rather than nested under source_config.designer
        # because the rubric is an EVALUATION input (consumed by the
        # visual-judgment pipeline + reflection polish), not a
        # save-destination configuration. Hydrated into
        # :class:`BriefDesignRubric` by the brief loader. Preserved
        # byte-for-byte across brief polish via the
        # ``_design_rubric_drift`` cascade entry in
        # :mod:`market_intelligence.brief_polish`.
        "design_rubric",
        # OSS Maintainers module (Slice 2). Top-level evaluation
        # inputs for maintainership-level classification on the
        # github source. Top-level rather than nested under
        # `source_config.github` (which stays empty per OSS
        # Maintainers Module Spec §8) because these are EVALUATION
        # inputs (consumed by the maintainership classifier + full-
        # eval prompt + strategy seeding), not save-destination
        # config. Brief polish preserves `target_projects` via the
        # `_target_projects_drift` cascade entry in
        # :mod:`market_intelligence.brief_polish`. Recruiter-readable
        # values: `target_projects` is a list of "owner/repo" strings
        # (e.g. ["kubernetes/kubernetes", "rust-lang/rust"]);
        # `target_stacks` is a list of language/framework/domain
        # tags; `maintainership_level` is one of
        # `RECOGNIZED_MAINTAINERSHIP_LEVELS`.
        "target_projects",
        "target_stacks",
        "maintainership_level",
        # Exa source module (S2). Top-level acquisition + evaluation
        # inputs for the exa source. Top-level rather than nested
        # under `source_config.exa` for the same reason as the OSS
        # block above: these drive query compilation and the
        # do-not-save gate, not save-destination semantics (Exa saves
        # land in the run folder + workspace like every other
        # source). Recruiter-readable values: `exa_target_firms` is a
        # list of `{"name", "domain", "aliases", "exclude_terms"}`
        # dicts, where `exclude_terms` is the namesake guard the
        # 2026-08-28 CoS slate needed (Accel → ["Accel-KKR"]);
        # `exa_query_lanes` is a subset of `EXA_QUERY_LANES`;
        # `exa_title_terms` is the title vocabulary to search
        # (e.g. ["Senior Associate", "Principal"]); a firm entry's own
        # optional `title_terms` replaces it for that firm alone.
        #
        # `exa_archetype_queries` (S8b) is the other half of the
        # acquisition surface: a list of
        # `{"id", "text", "axis", "num_results", "include_domains"}`
        # dicts, where `text` is a rich natural-language description of
        # the PERSON rather than a firm × title string. The firm lanes
        # can only find people at the twenty firms someone listed;
        # Sam's 2026-09-03 ruling ("prioritize top-3 consulting who
        # then go into investing") describes a career shape, not a firm
        # list, and this is the field that can search for it. The prior
        # art is `tools/exa_probe/queries-ciso.json` on the
        # `cloris-multi-hub` worktree — 17 archetype queries at 25
        # results each, which found people 60 mechanical firm strings
        # at 10 did not. Brief polish preserves all five via the
        # `_exa_fields_drift` cascade entry in
        # :mod:`market_intelligence.brief_polish`.
        "exa_target_firms",
        "exa_query_lanes",
        "exa_title_terms",
        "exa_archetype_queries",
        # Source-agnostic do-not-save gate (CLO-244) — no `exa_`
        # prefix because every module honors it. A list of
        # `{"name", "linkedin_url", "reason"}` dicts; the Exa module
        # is the first consumer, dropping named people at source the
        # way the CoS slate did by hand.
        "do_not_save",
        # Slate ordering, not judgment: `{"long_shot_employers": [...],
        # "long_shot_title_patterns": [...]}` decides who sorts last as a
        # long shot on both slates (`shared/attainability.py`).
        "attainability",
    }
)


class BriefSchemaError(ValueError):
    """Raised when a brief payload fails V2 validation.

    Carries ``missing_keys`` and ``invalid_keys`` so the API layer can
    return a structured 422 (the route layer maps this to its own
    response shape).
    """

    def __init__(
        self,
        message: str,
        *,
        missing_keys: tuple[str, ...] = (),
        invalid_keys: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.missing_keys = missing_keys
        self.invalid_keys = invalid_keys


@dataclass(frozen=True)
class MergedBrief:
    """Outcome of :func:`merge_legacy_brief`.

    ``v2_data`` is the canonical V2 fields (suitable for re-load via
    :func:`validate_v2_brief`). ``deprecated_keys`` lists every legacy
    key the original payload carried that is in the deprecation manifest;
    ``unknown_keys`` lists keys that are neither V2-recognized nor in the
    manifest. ``preserved_legacy`` is the full original-side dict
    (everything outside V2) so D2's PUT path can write the file back
    with all the recruiter-authored history intact.
    """

    v2_data: dict[str, Any]
    deprecated_keys: tuple[str, ...]
    unknown_keys: tuple[str, ...]
    preserved_legacy: dict[str, Any]


# Phase F Slice F2: per-source save-destination sub-schema. The
# ``source_config`` key is recognized as a top-level V2 key (above);
# this constant captures what each source's sub-dict is allowed to
# carry. Sources NOT listed here pass through with no per-key
# validation (forward-compat for new sources arriving via
# cloris.launchers).
#
# LinkedIn carries the project_id (where saves land in LinkedIn
# Recruiter); GitHub carries no fields today (saves land in the run
# folder); Researcher carries no save-destination fields (workspace
# is always available — see Researcher Module Spec §"Defended opinions"
# Opinion 4) but DOES carry recruiter-authored evaluation inputs:
# research_topics, conference_allowlist, discipline (load-bearing for
# layered floor resolution per Opinion 7), plus three optional
# power-user floor overrides (h_index_floor, papers_in_window_floor,
# papers_in_window_months) that override the discipline default.
SOURCE_CONFIG_REQUIRED_KEYS_BY_SOURCE: dict[str, frozenset[str]] = {
    "linkedin": frozenset(),
    "github": frozenset(),
    "researcher": frozenset(),
    "exec_search": frozenset(),
    "designer": frozenset(),
}

SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE: dict[str, frozenset[str]] = {
    "linkedin": frozenset({"project_id", "project_name"}),
    "github": frozenset(),
    # Researcher module (Slice 1): recruiter-authored evaluation inputs.
    # `discipline` is the load-bearing field per Researcher Module Spec
    # Opinion 7 (layered floor resolution). The three floor fields are
    # optional power-user overrides; the wizard does not surface them.
    "researcher": frozenset(
        {
            "research_topics",
            "role_mode",
            "seed_urls",
            "conference_allowlist",
            "discipline",
            "h_index_floor",
            "papers_in_window_floor",
            "papers_in_window_months",
        }
    ),
    "exec_search": frozenset(),
    # Designer module (Slice 1): no per-source save-destination
    # configuration. The structured rubric lives at top-level
    # ``design_rubric`` (an evaluation input, not a save-destination
    # config); the workspace is the implicit save destination.
    "designer": frozenset(),
}


# Per-source value-type map for keys that have specific structural
# requirements. The validator looks up the expected Python type per
# (source, key) pair; keys NOT listed here pass any value (per-source
# code paths handle deeper semantic checks). LinkedIn historically
# enforced strings via the recognized set; this map preserves that
# contract while letting Researcher carry list / int values without
# special-casing the validator.
SOURCE_CONFIG_KEY_VALUE_TYPES: dict[str, dict[str, type | tuple[type, ...]]] = {
    "linkedin": {
        "project_id": str,
        "project_name": str,
    },
    "researcher": {
        "research_topics": list,
        "role_mode": str,
        "seed_urls": list,
        "conference_allowlist": list,
        "discipline": str,
        "h_index_floor": int,
        "papers_in_window_floor": int,
        "papers_in_window_months": int,
    },
}


# Executive Search module (Slice 1): the values the brief's
# top-level `confidentiality_class` is allowed to carry.
RECOGNIZED_CONFIDENTIALITY_CLASSES: frozenset[str] = frozenset(
    {"open", "referenceable", "blind"}
)


# Designer module (Slice 1): the four anchor levels every rubric
# principle must define. The vision-evaluation prompt cites these by
# name in the "score against this anchor" instruction; the
# hallucination guard at :mod:`market_intelligence.brief_polish`
# (the `_design_rubric_drift` cascade entry) preserves them
# byte-for-byte. Order matters editorially (low to high) but the
# dict shape is permutation-stable.
RECOGNIZED_RUBRIC_ANCHORS: tuple[str, ...] = ("bad", "okay", "good", "excellent")

# Designer module (Slice 1): the disciplines a recruiter can pick
# in the intake wizard's `design_rubric` chapter. Each maps to a
# default rubric file at `config/design-rubrics/<discipline>.json`
# (Slice 1 ships only the `default.json`; per-discipline variants
# are a follow-up). `other` falls through to the default rubric.
RECOGNIZED_DESIGN_DISCIPLINES: frozenset[str] = frozenset(
    {"product", "brand", "motion", "illustration", "ux", "other"}
)


# OSS Maintainers module (Slice 2): the values the brief's
# top-level `maintainership_level` is allowed to carry. Ordered
# (low to high) so callers can compare ordinals when a slice's
# logic needs "at least maintainer." Stored as a tuple-of-tuples
# at module scope; the `frozenset` mirror is what
# :func:`validate_v2_brief` checks against.
MAINTAINERSHIP_LEVEL_ORDER: tuple[str, ...] = (
    "contributor",
    "maintainer",
    "project_lead",
)
RECOGNIZED_MAINTAINERSHIP_LEVELS: frozenset[str] = frozenset(
    MAINTAINERSHIP_LEVEL_ORDER
)


# Exa source module (S2): the query lanes an Exa campaign can run.
# Ordered the way the 2026-08-28 CoS slate ran them — people-search
# over LinkedIn profiles first, then firm team-page harvests, then
# operating-signal queries — with the archetype lane (S8b) last,
# because it is the one lane that does not iterate the firm list and
# its queries are emitted after the firm lanes'. `exa/lane_compiler.py`
# compiles one query set per active lane; an empty / absent
# `exa_query_lanes` means all four. The `frozenset` mirror is what
# :func:`validate_v2_brief` checks against.
EXA_QUERY_LANES: tuple[str, ...] = (
    "people_search",
    "team_page",
    "operating_signal",
    "archetype",
)
RECOGNIZED_EXA_QUERY_LANES: frozenset[str] = frozenset(EXA_QUERY_LANES)

# An archetype query's id: a slug, because it names the query in the
# console line, the slate's coverage row, and the work-unit id. Mixed
# case and spaces would render three different ways across those three
# surfaces.
EXA_ARCHETYPE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")

# Result-count bounds for one archetype query. The floor is 1 (a query
# that asks for nothing is a query that should not have been written);
# the ceiling is Exa's own per-search maximum. The prior art ran 25.
EXA_ARCHETYPE_NUM_RESULTS_MIN = 1
EXA_ARCHETYPE_NUM_RESULTS_MAX = 100

# What an archetype query asks for when the author says nothing. 25 is
# the prior art's number (`tools/exa_probe/queries-ciso.json`), and it
# is deliberately more than the firm lanes' 10: an archetype query has
# no firm allowlist narrowing it, so its top 10 is a thinner slice of
# its own population than a firm query's is.
#
# The domain default is LinkedIn because the archetype lane takes the
# people_search path in `exa/acquisition.py`, which drops every result
# that is not a public profile URL. Widening the domains there buys
# pages that are then dropped as `not_profile` — paid for, not read.
# The two defaults live here rather than in the compiler so that
# `market_intelligence.brief_polish`'s drift canonicalizer and the
# compiler agree on what an omitted field means; if they disagreed,
# omitting `num_results` and writing 25 would read as drift.
EXA_ARCHETYPE_DEFAULT_NUM_RESULTS = 25
EXA_ARCHETYPE_DEFAULT_INCLUDE_DOMAINS: tuple[str, ...] = ("linkedin.com",)


@dataclass(frozen=True)
class RubricPrinciple:
    """One scoring dimension of a Designer brief rubric.

    Carries the recruiter-readable name + description, anchor-level
    definitions for the four scoring tiers (bad / okay / good /
    excellent), and an optional weight used by discipline overrides
    to emphasize or skip the principle for a given discipline.

    Hydrated from the validated V2 brief by the brief loader; the
    vision-evaluation prompt at :mod:`designer.judgment_templates`
    renders the anchor definitions verbatim into the LLM system prompt.
    """

    name: str
    description: str
    anchors: dict[str, str]
    weight: float = 1.0


@dataclass(frozen=True)
class CalibrationExemplar:
    """One concrete portfolio + verdict pair the recruiter pointed at.

    Used both at intake time (Cloris reads back "you said X is a YES
    because Y; should the rubric agree?") and at evaluation time (the
    vision prompt receives 3-5 exemplars as in-context calibration).
    """

    portfolio_url: str
    discipline: str
    verdict: str
    per_principle_reasoning: dict[str, str]
    overall_reasoning: str


@dataclass(frozen=True)
class BriefDesignRubric:
    """Structured rubric for vision-LLM evaluation of designer portfolios.

    The load-bearing artifact for the Designer module's evaluation
    pipeline. Per-recruiter taste lives here (not in code, not in a
    fine-tuned model). Brief polish preserves this byte-for-byte via
    the ``_design_rubric_drift`` cascade entry in
    :mod:`market_intelligence.brief_polish`; reflection polish
    proposes per-principle weight refinements via the ``RUBRIC_REFINE``
    hunk kind (Slice 9), but only with explicit recruiter approval.

    Design choice: structured (not prose) so byte-equality preservation,
    per-principle vision scoring, and discipline-weight typed access
    work cleanly. Spec rationale at
    `/home/operator/.cursor/plans/designer-module-spec_5f3d48c1.plan.md` §2.
    """

    principles: tuple[RubricPrinciple, ...] = ()
    discipline_weight_overrides: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    calibration_exemplars: tuple[CalibrationExemplar, ...] = ()
    hard_reject_patterns: tuple[str, ...] = ()


def validate_v2_brief(data: dict[str, Any]) -> None:
    """Validate ``data`` is a parseable V2 brief.

    Raises :class:`BriefSchemaError` on:
    - Missing required keys (``capability_areas``, ``depth_distinction``).
    - ``capability_areas`` not a non-empty list of dicts with ``name`` +
      ``description`` keys.
    - ``depth_distinction`` not a dict with the three required builder /
      user / edge_case fields.
    - ``source_config`` (if present) not a dict, or any per-source
      sub-dict not a dict, or any per-source key with a value whose
      Python type doesn't match :data:`SOURCE_CONFIG_KEY_VALUE_TYPES`
      (e.g., LinkedIn's ``project_id`` must be a string; Researcher's
      ``research_topics`` must be a list).

    Deeper per-source semantic checks (e.g., research_topics is a list
    OF STRINGS, discipline is one of 8 enum values) live with the source.

    Does NOT raise on legacy or unknown keys — those are the merge-helper's
    job. Validation is "is this structurally a V2 brief," not "does this
    have a clean key set."
    """

    if not isinstance(data, dict):
        raise BriefSchemaError(
            "Brief payload must be a JSON object.",
        )

    missing = tuple(k for k in REQUIRED_V2_KEYS if k not in data)
    if missing:
        raise BriefSchemaError(
            f"Missing required V2 keys: {sorted(missing)}",
            missing_keys=missing,
        )

    invalid: list[str] = []
    cap_areas = data.get("capability_areas")
    if not isinstance(cap_areas, list) or not cap_areas:
        invalid.append("capability_areas")
    else:
        for idx, ca in enumerate(cap_areas):
            if not isinstance(ca, dict):
                invalid.append(f"capability_areas[{idx}]")
                continue
            if "name" not in ca or "description" not in ca:
                invalid.append(f"capability_areas[{idx}].name|description")

    depth = data.get("depth_distinction")
    if not isinstance(depth, dict):
        invalid.append("depth_distinction")
    else:
        for k in ("builder_definition", "user_definition", "edge_case_guidance"):
            if k not in depth:
                invalid.append(f"depth_distinction.{k}")

    engagement_context = data.get("engagement_context")
    if "engagement_context" in data:
        if not isinstance(engagement_context, dict):
            invalid.append("engagement_context")
        else:
            if (
                engagement_context.get("selectivity_posture")
                not in SELECTIVITY_POSTURES
            ):
                invalid.append("engagement_context.selectivity_posture")
            engagement = engagement_context
            for key in (
                "hiring_company",
                "engagement_description",
                "talent_bar_statement",
            ):
                if key in engagement and not isinstance(
                    engagement[key], str
                ):
                    invalid.append(f"engagement_context.{key}")
            # Optional: requirements the judge may never defer to the
            # outreach conversation (shared/judgment/templates.py). A list of
            # non-empty strings; anything else would silently render nothing.
            if "non_deferrable_requirements" in engagement:
                requirements = engagement["non_deferrable_requirements"]
                if not isinstance(requirements, list) or not all(
                    isinstance(item, str) and item.strip() for item in requirements
                ):
                    invalid.append("engagement_context.non_deferrable_requirements")

    # Phase F Slice F2: source_config sub-schema. Optional at the top
    # level; if present, must be a dict whose values per recognized
    # source are dicts with stringly-typed values.
    sc = data.get("source_config")
    if sc is not None:
        if not isinstance(sc, dict):
            invalid.append("source_config")
        else:
            for source, sub in sc.items():
                if not isinstance(sub, dict):
                    invalid.append(f"source_config.{source}")
                    continue
                key_types = SOURCE_CONFIG_KEY_VALUE_TYPES.get(source, {})
                for sub_key, sub_val in sub.items():
                    expected = key_types.get(sub_key)
                    if expected is not None and not isinstance(sub_val, expected):
                        invalid.append(f"source_config.{source}.{sub_key}")

    # Executive Search Slice 1: confidentiality_class is optional but
    # constrained to a known enum when present. Unknown values are a
    # schema error (not silently coerced) so the recruiter sees the
    # mistake at brief-edit time, not when redaction would have leaked.
    cc = data.get("confidentiality_class")
    if cc is not None and cc != "":
        if not isinstance(cc, str) or cc not in RECOGNIZED_CONFIDENTIALITY_CLASSES:
            invalid.append("confidentiality_class")

    # Designer Slice 1: design_rubric is optional but, when present,
    # must be a dict with `principles` (list of dicts each carrying
    # name + description + anchors-with-all-four-levels), optional
    # weight in [0, 5], optional `discipline_weight_overrides`,
    # optional `calibration_exemplars`, optional
    # `hard_reject_patterns`. The vision-evaluation prompt grounds
    # itself in this shape; an invalid rubric would silently produce
    # malformed prompts at run time, so we fail at brief-edit time.
    invalid.extend(_validate_design_rubric(data.get("design_rubric")))

    # OSS Maintainers Slice 2: target_projects / target_stacks are
    # optional lists of strings; maintainership_level is optional
    # but, when present, must be one of
    # `RECOGNIZED_MAINTAINERSHIP_LEVELS`. Invalid shapes here would
    # silently produce empty maintainership classification at run
    # time (the source-spec §11 contract: behavior-preserving when
    # `target_projects` is empty), so we fail at brief-edit time
    # rather than degrading silently.
    tp = data.get("target_projects")
    if tp is not None:
        if not isinstance(tp, list) or not all(isinstance(p, str) for p in tp):
            invalid.append("target_projects")
    ts = data.get("target_stacks")
    if ts is not None:
        if not isinstance(ts, list) or not all(isinstance(p, str) for p in ts):
            invalid.append("target_stacks")
    ml = data.get("maintainership_level")
    if ml is not None and ml != "":
        if not isinstance(ml, str) or ml not in RECOGNIZED_MAINTAINERSHIP_LEVELS:
            invalid.append("maintainership_level")

    # Exa source module (S2): the four acquisition inputs are all
    # optional and all default to empty, so a brief that omits them
    # loads byte-identically to before. Malformed shapes fail here
    # rather than at run time, where a firm entry missing `name`
    # would compile a query against the empty string and a
    # malformed `do_not_save` row would silently stop excluding
    # someone the recruiter named.
    invalid.extend(validate_exa_fields(data))
    invalid.extend(validate_attainability_field(data))
    invalid.extend(validate_alumni_blacklist_field(data))
    invalid.extend(validate_role_level_envelope_field(data))

    if invalid:
        raise BriefSchemaError(
            f"Invalid V2 schema fields: {invalid}",
            invalid_keys=tuple(invalid),
        )


def _validate_design_rubric(rubric: Any) -> list[str]:
    """Return invalid_keys descriptors for a malformed `design_rubric`.

    Designer Slice 1. Empty list when ``rubric is None`` (optional
    field) or ``rubric == {}`` (recruiter cleared it intentionally).
    """

    invalid: list[str] = []
    if rubric is None or rubric == {}:
        return invalid
    if not isinstance(rubric, dict):
        invalid.append("design_rubric")
        return invalid

    principles = rubric.get("principles")
    if principles is not None:
        if not isinstance(principles, list):
            invalid.append("design_rubric.principles")
        else:
            for idx, principle in enumerate(principles):
                if not isinstance(principle, dict):
                    invalid.append(f"design_rubric.principles[{idx}]")
                    continue
                if not isinstance(principle.get("name"), str) or not principle["name"]:
                    invalid.append(f"design_rubric.principles[{idx}].name")
                if not isinstance(principle.get("description"), str):
                    invalid.append(f"design_rubric.principles[{idx}].description")
                anchors = principle.get("anchors")
                if not isinstance(anchors, dict):
                    invalid.append(f"design_rubric.principles[{idx}].anchors")
                else:
                    for level in RECOGNIZED_RUBRIC_ANCHORS:
                        if level not in anchors or not isinstance(anchors[level], str):
                            invalid.append(
                                f"design_rubric.principles[{idx}].anchors.{level}"
                            )
                weight = principle.get("weight", 1.0)
                if not isinstance(weight, (int, float)) or not (0.0 <= float(weight) <= 5.0):
                    invalid.append(f"design_rubric.principles[{idx}].weight")

    overrides = rubric.get("discipline_weight_overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            invalid.append("design_rubric.discipline_weight_overrides")
        else:
            for discipline, weights in overrides.items():
                if not isinstance(weights, dict):
                    invalid.append(
                        f"design_rubric.discipline_weight_overrides.{discipline}"
                    )
                    continue
                for principle_name, w in weights.items():
                    if not isinstance(w, (int, float)) or not (0.0 <= float(w) <= 5.0):
                        invalid.append(
                            f"design_rubric.discipline_weight_overrides.{discipline}.{principle_name}"
                        )

    exemplars = rubric.get("calibration_exemplars")
    if exemplars is not None:
        if not isinstance(exemplars, list):
            invalid.append("design_rubric.calibration_exemplars")
        else:
            for idx, ex in enumerate(exemplars):
                if not isinstance(ex, dict):
                    invalid.append(f"design_rubric.calibration_exemplars[{idx}]")
                    continue
                for key in ("portfolio_url", "discipline", "verdict"):
                    if not isinstance(ex.get(key), str) or not ex[key]:
                        invalid.append(
                            f"design_rubric.calibration_exemplars[{idx}].{key}"
                        )
                if (
                    isinstance(ex.get("verdict"), str)
                    and ex["verdict"] not in {"yes", "no", "borderline"}
                ):
                    invalid.append(
                        f"design_rubric.calibration_exemplars[{idx}].verdict"
                    )

    patterns = rubric.get("hard_reject_patterns")
    if patterns is not None:
        if not isinstance(patterns, list) or not all(
            isinstance(p, str) for p in patterns
        ):
            invalid.append("design_rubric.hard_reject_patterns")

    return invalid


def validate_exa_fields(data: dict[str, Any]) -> list[str]:
    """Return invalid_keys descriptors for the Exa module's brief fields.

    Exa source module S2, extended by S8b. All five fields are
    optional; each returns no descriptor when absent, ``None``, or
    empty (the recruiter cleared it). Descriptors are the bare
    top-level key rather than an indexed path — the recruiter edits
    the field as a whole, so "which firm entry" is diagnostic detail,
    not the affordance.

    Public (unlike its ``_validate_design_rubric`` sibling) because
    :func:`shared.brief_loader._load_v2_brief` calls it too. Launch
    does not run :func:`validate_v2_brief` — that is the authoring
    gate — and a malformed ``do_not_save`` row silently hydrating to
    ``[]`` would let a ruled-out person through. These five fields
    fail closed at load; every other field keeps its load-time
    tolerance.
    """

    invalid: list[str] = []

    firms = data.get("exa_target_firms")
    if firms is not None and firms != []:
        if not isinstance(firms, list):
            invalid.append("exa_target_firms")
        else:
            for firm in firms:
                if not isinstance(firm, dict):
                    invalid.append("exa_target_firms")
                    break
                name = firm.get("name")
                if not isinstance(name, str) or not name.strip():
                    invalid.append("exa_target_firms")
                    break
                if "domain" in firm and not isinstance(firm["domain"], str):
                    invalid.append("exa_target_firms")
                    break
                bad_list = False
                for key in ("aliases", "exclude_terms"):
                    if key in firm:
                        value = firm[key]
                        if not isinstance(value, list) or not all(
                            isinstance(v, str) for v in value
                        ):
                            bad_list = True
                # A firm's own title vocabulary (req 1238's clients are searched
                # only in their data seats): non-empty, every term a real string.
                if "title_terms" in firm:
                    terms = firm["title_terms"]
                    if not isinstance(terms, list) or not terms or not all(
                        isinstance(t, str) and t.strip() for t in terms
                    ):
                        bad_list = True
                if bad_list:
                    invalid.append("exa_target_firms")
                    break

    lanes = data.get("exa_query_lanes")
    if lanes is not None and lanes != []:
        if not isinstance(lanes, list) or not all(
            isinstance(lane, str) and lane in RECOGNIZED_EXA_QUERY_LANES
            for lane in lanes
        ):
            invalid.append("exa_query_lanes")

    title_terms = data.get("exa_title_terms")
    if title_terms is not None and title_terms != []:
        if not isinstance(title_terms, list) or not all(
            isinstance(t, str) and t.strip() for t in title_terms
        ):
            invalid.append("exa_title_terms")

    archetypes = data.get("exa_archetype_queries")
    if archetypes is not None and archetypes != []:
        if not isinstance(archetypes, list):
            invalid.append("exa_archetype_queries")
        else:
            seen_ids: set[str] = set()
            for entry in archetypes:
                bad = False
                if not isinstance(entry, dict):
                    bad = True
                else:
                    entry_id = entry.get("id")
                    if (
                        not isinstance(entry_id, str)
                        or not EXA_ARCHETYPE_ID_PATTERN.match(entry_id.strip())
                    ):
                        bad = True
                    elif entry_id.strip() in seen_ids:
                        # Two queries under one id render as one row on
                        # the slate and one work-unit id in the store —
                        # the second query's people would look like the
                        # first query's.
                        bad = True
                    else:
                        seen_ids.add(entry_id.strip())
                    text = entry.get("text")
                    if not isinstance(text, str) or not text.strip():
                        bad = True
                    if "axis" in entry and not isinstance(entry["axis"], str):
                        bad = True
                    if "num_results" in entry:
                        num = entry["num_results"]
                        if (
                            isinstance(num, bool)
                            or not isinstance(num, int)
                            or not (
                                EXA_ARCHETYPE_NUM_RESULTS_MIN
                                <= num
                                <= EXA_ARCHETYPE_NUM_RESULTS_MAX
                            )
                        ):
                            bad = True
                    if "include_domains" in entry:
                        domains = entry["include_domains"]
                        if not isinstance(domains, list) or not all(
                            isinstance(d, str) for d in domains
                        ):
                            bad = True
                if bad:
                    invalid.append("exa_archetype_queries")
                    break

    do_not_save = data.get("do_not_save")
    if do_not_save is not None and do_not_save != []:
        if not isinstance(do_not_save, list):
            invalid.append("do_not_save")
        else:
            for entry in do_not_save:
                if not isinstance(entry, dict):
                    invalid.append("do_not_save")
                    break
                name = entry.get("name")
                if not isinstance(name, str) or not name.strip():
                    invalid.append("do_not_save")
                    break
                if any(
                    key in entry and not isinstance(entry[key], str)
                    for key in ("linkedin_url", "reason")
                ):
                    invalid.append("do_not_save")
                    break

    return invalid


def validate_attainability_field(data: dict[str, Any]) -> list[str]:
    """``["attainability"]`` when the block is malformed, else ``[]``.

    Fails closed at load as well as here, like :func:`validate_exa_fields`:
    a malformed block would otherwise sort the slates by the defaults while
    the brief says something else.
    """

    return ["attainability"] if attainability_errors(data.get("attainability")) else []


def validate_alumni_blacklist_field(data: dict[str, Any]) -> list[str]:
    """``["alumni_blacklist"]`` unless the field is absent or a list of names.

    Fails closed at load as well as here, like :func:`validate_exa_fields`:
    a malformed list hydrating to ``[]`` would let every alumnus through
    while the brief says they are excluded. ``None`` and ``[]`` are absent.
    """

    value = data.get("alumni_blacklist") if isinstance(data, dict) else None
    if value is None or value == []:
        return []
    if not isinstance(value, list) or not all(
        isinstance(name, str) and name.strip() for name in value
    ):
        return ["alumni_blacklist"]
    return []


# The keys a ``role_level_envelope`` may carry. ``target_level_statement`` is
# required whenever the envelope is set: without it the template renders the
# default envelope and the other two lines never reach the judge.
ROLE_LEVEL_ENVELOPE_KEYS: frozenset[str] = frozenset(
    {"target_level_statement", "over_level_looks_like", "under_level_looks_like"}
)


def validate_role_level_envelope_field(data: dict[str, Any]) -> list[str]:
    """``["role_level_envelope"]`` unless the envelope is absent or well formed.

    Well formed: a dict whose keys are all in :data:`ROLE_LEVEL_ENVELOPE_KEYS`,
    whose values are strings, and whose ``target_level_statement`` is not
    blank. ``None`` and ``{}`` are absent. Fails closed at load as well as
    here, like :func:`validate_alumni_blacklist_field`: a malformed envelope
    would otherwise render the default level text while the brief says the
    level is defined, and with a hard years ceiling that default is a hard
    over-level rule.
    """

    value = data.get("role_level_envelope") if isinstance(data, dict) else None
    if value is None or value == {}:
        return []
    if not isinstance(value, dict):
        return ["role_level_envelope"]
    if set(value) - ROLE_LEVEL_ENVELOPE_KEYS:
        return ["role_level_envelope"]
    if not all(isinstance(text, str) for text in value.values()):
        return ["role_level_envelope"]
    if not str(value.get("target_level_statement") or "").strip():
        return ["role_level_envelope"]
    return []


def source_config_for(data: dict[str, Any], source: str) -> dict[str, Any]:
    """Return the per-source sub-dict from ``data["source_config"]``.

    Phase F Slice F2 helper. Missing or malformed `source_config` ⇒
    empty dict. Missing source key ⇒ empty dict. Used by the launcher
    registry's `save_destination_blocker_fn` to read per-brief
    destinations without scattering schema-shape knowledge.
    """

    sc = data.get("source_config") if isinstance(data, dict) else None
    if not isinstance(sc, dict):
        return {}
    sub = sc.get(source)
    if not isinstance(sub, dict):
        return {}
    return sub


def linkedin_project_id_from_brief(data: dict[str, Any]) -> str | None:
    """Resolve the LinkedIn project id with backward-compat fallback.

    Phase F Slice F2. Reads ``source_config.linkedin.project_id`` first;
    falls back to the flat ``linkedin_project_id`` field for briefs not
    yet migrated to ``source_config``. Returns ``None`` if neither is
    populated.

    The fallback is what keeps ``derive_brief_id()`` stable across
    the migration — without it, every existing state_dir would be
    orphaned the moment F2 introduced the new path.
    """

    sub = source_config_for(data, "linkedin")
    candidate = sub.get("project_id")
    if isinstance(candidate, str) and candidate:
        return candidate
    flat = data.get("linkedin_project_id") if isinstance(data, dict) else None
    if isinstance(flat, str) and flat:
        return flat
    if flat is not None:
        # Defensive coercion for legacy briefs that stored project_id
        # as a number; mirrors `_scan_authored_briefs`'s coercion at
        # `cloris/api.py`.
        return str(flat)
    return None


def merge_legacy_brief(data: dict[str, Any]) -> MergedBrief:
    """Split a brief payload into V2 vs legacy parts.

    Implements the Fork B "merge-with-deprecation" strategy:
    - Keys in :data:`RECOGNIZED_V2_KEYS` end up in ``v2_data``.
    - Keys in :data:`DEPRECATED_KEYS_BY_VERSION` (any version) get listed
      in ``deprecated_keys`` AND preserved in ``preserved_legacy``.
    - Keys in neither set get listed in ``unknown_keys`` AND preserved
      (the recruiter authored them; we don't drop silently).

    The recruiter UI uses ``deprecated_keys`` to render "drop from brief"
    affordances per key. ``unknown_keys`` get a milder "Cloris doesn't
    recognize this — keep or drop?" treatment.

    This function does NOT validate the V2 surface — call
    :func:`validate_v2_brief` separately if needed. Merge is purely
    structural; validation is semantic.
    """

    if not isinstance(data, dict):
        raise BriefSchemaError("Brief payload must be a JSON object.")

    # Flatten the deprecation manifest into a single set across versions.
    # We don't currently surface "deprecated AT version X" granularity to
    # the UI; it's enough to know the key is deprecated. The version
    # split exists so future deprecations land cleanly.
    deprecated_set: set[str] = set()
    for version_keys in DEPRECATED_KEYS_BY_VERSION.values():
        deprecated_set.update(version_keys.keys())

    v2_data: dict[str, Any] = {}
    preserved_legacy: dict[str, Any] = {}
    deprecated_present: list[str] = []
    unknown_present: list[str] = []

    for key, value in data.items():
        if key in RECOGNIZED_V2_KEYS:
            v2_data[key] = value
        elif key in deprecated_set:
            deprecated_present.append(key)
            preserved_legacy[key] = value
        else:
            unknown_present.append(key)
            preserved_legacy[key] = value

    return MergedBrief(
        v2_data=v2_data,
        deprecated_keys=tuple(sorted(deprecated_present)),
        unknown_keys=tuple(sorted(unknown_present)),
        preserved_legacy=preserved_legacy,
    )


def deprecation_message(key: str) -> str | None:
    """Return the rationale string for a deprecated key, or ``None``.

    Walks :data:`DEPRECATED_KEYS_BY_VERSION` newest-version-first. The
    UI uses this to render the per-key "drop from brief" affordance with
    Cloris-voice rationale.
    """

    for version in sorted(DEPRECATED_KEYS_BY_VERSION.keys(), reverse=True):
        if key in DEPRECATED_KEYS_BY_VERSION[version]:
            return DEPRECATED_KEYS_BY_VERSION[version][key]
    return None
