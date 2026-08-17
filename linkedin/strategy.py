"""Strategy formation and adaptation — Opus plans and adjusts search execution.

Kit strings are VOCABULARY — raw Boolean terms organized by competency domain
that Opus uses as building blocks. Kit strings NEVER appear in the execution queue.
Opus synthesizes its own compound search strings from this vocabulary.

Two main functions:
1. form_strategy() — ONE Opus call at run start to synthesize compound search strings
2. adapt_after_block() — ONE Opus call after each block to generate new strings from vocabulary
"""

from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path
from shared.schemas import KitString, ExecutionPlan, BlockReport, AdaptationResponse, SearchString
import shared.config as _config
from shared.llm_clients import opus_llm_cached as opus_llm
from shared.strategy_shadow import dispatch_strategy_shadow, plan_metrics
from shared.brief_loader import Brief
from shared.retrieval_design import (
    RetrievalDesign,
    render_retrieval_design,
    summarize_retrieval_design,
)
from shared.search_memory import (
    _canonicalize_lane_id,
    format_search_memory_summary,
    get_search_memory_families,
    infer_domain_lane,
    normalize_family_key,
    normalize_novelty_bucket,
)
from shared.role_strategy import apply_role_strategy_to_plan
from shared.sourcing_lanes import populate_execution_plan_lane_payloads
from shared.strict_seniority import (
    classify_search_string_seniority,
    is_strict_seniority_brief,
)
from linkedin.boolean_compiler import (
    attach_boolean_lint_to_plan,
    repair_constraint_surfaces,
    summarize_kit_lint,
    ubiquitous_terms_from_brief,
)
from linkedin.adaptation_signal_state import (
    MarketSignalPrior,
    SearchSignalState,
    apply_adapted_string_firewall,
    coerce_market_signal_prior,
    render_market_signal_prior_for_prompt,
    render_signal_state_for_prompt,
)
from linkedin.matching_contract import (
    render_adaptation_matching_guidance,
    render_strategy_matching_guidance,
)
from linkedin.strategy_lane_compiler import apply_linkedin_lane_compiler_to_plan


_MAX_PROMOTED_EDGE_CASE_GAPS = 3

# Role-agnostic craft exemplars, one per composition angle, in slot notation.
# Deliberately keyword-free: concrete same-brief exemplars anchor the model to
# their vocabulary and skeleton (2026-07-05 SPL plan: 17/17 strings shared one
# shape with the portfolio doctrine already live). The <slots> demonstrate the
# principle; the vocabulary must come from the brief and the market.
_ANGLE_SHAPE_EXEMPLARS = """### Worked angle shapes (slot notation — the vocabulary is yours to author)

Each shape illustrates ONE angle and the craft principle it rides on. The <slots> are placeholders: fill them from THIS brief's market and the words real practitioners write. Never copy a shape onto a string whose angle calls for a different one.

- Title-doorway (recall): ("<title variant>" OR "<morphological variant>" OR "<abbreviation>" OR "<adjacent-title analog>") AND ("<domain concept>" OR "<same concept, practitioner register>")
  Principle: each OR group is ONE concept's synonym set — expand morphology and self-description register; never mix a title with a domain concept in one group.
- Proof-of-practice anchor (precision): ("<distinctive artifact only real practitioners carry>" OR "<specific tool/framework/benchmark name>")
  Principle: a precise string may be a single specific term — distinctiveness IS the filter; no generic-verb AND gate.
- Community/artifact angle: ("<conference, community, certification, or publication marker>" OR "<competition or award phrasing>") AND ("<capability concept>")
  Principle: reach people by where their work leaves traces, not by what their employer titles them.
- NOT-as-discovery probe: ("<broad capability concept>") NOT ("<titles you already know>" OR "<known-noise archetype>")
  Principle: subtract the known to surface titles you would never enumerate; keep NOT surgical — every exclusion also removes qualified people.
- Facet-bounded canonical pass (cleanup): ("<capability proxy>") with structured_filters bounding titles/companies to the brief's exact canonical values
  Principle: a deliberate canonical-pool bound rides the facet, never a keyword OR-clause of the same dimension.
"""


def _render_example_compounds_block(brief: Brief) -> str:
    """Render brief.example_compounds as an "Example rendered compound" block.

    The compat brief mirrors structured ``ExampleCompound`` dataclass instances
    with attributes ``boolean``, ``purpose`` and ``novelty_bucket``. If the
    brief carries no example compounds (or they are all empty), return ``""``
    so the caller can omit the section cleanly.

    This is read-only on the brief — it never mutates the mirrored list.
    """
    examples = getattr(brief, "example_compounds", None) or ()
    lines: list[str] = []
    for ex in examples:
        boolean = str(getattr(ex, "boolean", "") or "").strip()
        if not boolean:
            continue
        purpose = str(getattr(ex, "purpose", "") or "").strip()
        bucket = str(getattr(ex, "novelty_bucket", "") or "").strip()
        bucket_suffix = f" [{bucket}]" if bucket else ""
        if purpose:
            lines.append(f"- {purpose}{bucket_suffix}: {boolean}")
        else:
            lines.append(f"- {boolean}{bucket_suffix}")
    return "\n".join(lines)


def _render_term_blacklist_block(brief: Brief) -> str:
    """Render brief.term_blacklist_categories as a labeled bullet list.

    Mirrors the shape of ``shared.brief_schema.Brief.term_blacklist_block`` but
    operates on the compat ``Brief`` mirror (a list of ``BlacklistCategory``
    dataclass instances). Returns ``""`` when the brief carries no
    blacklist categories so the caller can omit the surrounding section.
    """
    categories = getattr(brief, "term_blacklist_categories", None) or ()
    lines: list[str] = []
    for cat in categories:
        label = str(getattr(cat, "label", "") or "").strip()
        rationale = str(getattr(cat, "rationale", "") or "").strip()
        terms = [str(t).strip() for t in (getattr(cat, "terms", None) or []) if str(t).strip()]
        if not (label or rationale or terms):
            continue
        if label and rationale:
            lines.append(f"- **{label}** — {rationale}")
        elif label:
            lines.append(f"- **{label}**")
        elif rationale:
            lines.append(f"- {rationale}")
        if terms:
            lines.append(f"  Terms: {', '.join(terms)}")
    return "\n".join(lines)


def _render_abbreviation_collisions_block(brief: Brief) -> str:
    """Render brief.abbreviation_collisions as a guidance bullet list.

    Mirrors ``shared.brief_schema.Brief.abbreviation_collisions_block``. The
    block produced is the body of the "Abbreviation Collision Filter" section
    — the surrounding rule prose is kept static in the prompt. Returns ``""``
    when no collisions are configured so the caller can omit the section.
    """
    collisions = getattr(brief, "abbreviation_collisions", None) or ()
    lines: list[str] = []
    for ab in collisions:
        abbreviation = str(getattr(ab, "abbreviation", "") or "").strip()
        expansion = str(getattr(ab, "expansion", "") or "").strip()
        if not abbreviation:
            continue
        standalone = bool(getattr(ab, "standalone_allowed", False))
        note = str(getattr(ab, "note", "") or "").strip()
        if expansion:
            handling = "standalone allowed" if standalone else "pair with expansion"
            lines.append(f"- \"{abbreviation}\" → {expansion} ({handling})")
        else:
            handling = "standalone allowed" if standalone else "do not use standalone"
            lines.append(f"- \"{abbreviation}\" — {handling}")
        if note:
            lines.append(f"  Note: {note}")
    return "\n".join(lines)


def _iter_brief_patterns(brief: Brief, attr: str) -> tuple[str, ...]:
    """Return the lowercased non-empty patterns under ``attr`` on the compat brief."""
    values = getattr(brief, attr, None) or ()
    return tuple(
        str(value).strip().lower()
        for value in values
        if str(value or "").strip()
    )


def _design_from_brief(brief: Brief) -> RetrievalDesign:
    return RetrievalDesign.from_dict(getattr(brief, "retrieval_design", {}) or {})


def _explicit_design_from_brief(brief: Brief) -> RetrievalDesign:
    design = _design_from_brief(brief)
    if design.is_explicit():
        return design
    return RetrievalDesign()


def _brief_targets_edge_case_opening(brief: Brief) -> bool:
    """Whether the brief explicitly calls for a tapped-market edge-case opening."""
    haystack = " ".join(
        [brief.role_description, brief.intake_notes, *(brief.instructions or [])]
    ).lower()
    triggers = (
        "tapped",
        "exhausted",
        "heavily worked",
        "already been worked",
        "obvious pool",
        "edge-case",
        "edge case",
        "nooks and crannies",
    )
    return any(trigger in haystack for trigger in triggers)


def _opening_priority(brief: Brief, boolean: str, rationale: str = "") -> tuple[int, int]:
    """Classify how suitable a string is for an edge-case opening sequence.

    The classification draws its vocabulary entirely from the calibration
    mirror on the compat ``Brief`` (canonical_*_patterns, edge_case_patterns,
    edge_case_company_patterns). When those mirrors are empty the function
    degrades to ``(1, 0)`` for every input — neutral / mixed.

    Returns (bucket, score):
      bucket 0 = edge-case / adjacent opening string
      bucket 1 = neutral / mixed
      bucket 2 = canonical cleanup string
    """
    text = f"{boolean} {rationale}".lower()

    framework_patterns = _iter_brief_patterns(brief, "canonical_framework_patterns")
    company_patterns = _iter_brief_patterns(brief, "canonical_company_patterns")
    title_patterns = _iter_brief_patterns(brief, "canonical_title_patterns")
    broad_patterns = _iter_brief_patterns(brief, "canonical_broad_patterns")
    edge_patterns = _iter_brief_patterns(brief, "edge_case_patterns")
    edge_company_patterns = _iter_brief_patterns(brief, "edge_case_company_patterns")

    framework_hits = sum(1 for pattern in framework_patterns if pattern in text)
    company_hits = sum(1 for pattern in company_patterns if pattern in text)
    title_hits = sum(1 for pattern in title_patterns if pattern in text)
    broad_hits = sum(1 for pattern in broad_patterns if pattern in text)
    edge_hits = sum(1 for pattern in edge_patterns if pattern in text)
    edge_hits += sum(1 for pattern in edge_company_patterns if pattern in text)

    framework_first = framework_hits >= 2 and edge_hits == 0
    company_first = company_hits >= 2 and edge_hits == 0
    title_first = title_hits >= 1 and edge_hits == 0
    broad_core = broad_hits >= 2 and edge_hits == 0

    canonical = framework_first or company_first or title_first or broad_core
    edge_case = edge_hits >= 2 or (edge_hits >= 1 and not canonical)

    if edge_case and not canonical:
        bucket = 0
    elif canonical and edge_hits <= 1:
        bucket = 2
    else:
        bucket = 1

    score = (
        edge_hits * 5
        - framework_hits * 4
        - company_hits * 3
        - title_hits * 3
        - broad_hits * 2
    )
    return bucket, score


def _sort_strings_for_edge_case_opening(brief: Brief, strings: list[dict]) -> list[dict]:
    annotated: list[tuple[tuple[int, int, int], dict]] = []
    for idx, item in enumerate(strings):
        bucket, score = _opening_priority(
            brief,
            item.get("boolean", ""),
            item.get("rationale", "") or item.get("gap", ""),
        )
        annotated.append(((bucket, -score, idx), item))
    annotated.sort(key=lambda pair: pair[0])
    return [item for _, item in annotated]


def _annotate_string_metadata(
    brief: Brief,
    item: dict,
    *,
    boolean_key: str = "boolean",
) -> dict:
    """Ensure generated strings carry stable family/novelty/domain labels."""
    boolean = item.get(boolean_key, "") or ""
    rationale = item.get("rationale", "") or item.get("gap", "") or ""
    bucket, _score = _opening_priority(brief, boolean, rationale)
    retrieval_recipe = item.get("retrieval_recipe", {}) if isinstance(item.get("retrieval_recipe"), dict) else {}
    hypothesis_ids = [
        str(hypothesis_id).strip()
        for hypothesis_id in retrieval_recipe.get("applied_hypothesis_ids", [])
        if str(hypothesis_id).strip()
    ]

    item["family_key"] = normalize_family_key(
        item.get("family_key") or retrieval_recipe.get("family_id"),
        boolean,
        rationale,
    )
    item["novelty_bucket"] = normalize_novelty_bucket(
        item.get("novelty_bucket")
        or ("edge_case" if hypothesis_ids or bucket == 0 else "canonical"),
        boolean,
        rationale,
        brief=brief,
    )
    item["domain_lane"] = infer_domain_lane(
        item.get("domain_lane")
        or retrieval_recipe.get("target_markets", [None])[0],
        boolean,
        rationale,
        brief=brief,
    )
    if is_strict_seniority_brief(brief):
        # BFSI-era risk vocabulary stamps ONLY the briefs that opted into the
        # strict-seniority regime; on every other brief the stamps stay
        # neutral so downstream consumers (exploitation overlay demotions,
        # opening sorts) cannot fire exec-search heuristics on it.
        seniority = classify_search_string_seniority(
            boolean,
            rationale,
            domain_lane=item["domain_lane"],
        )
        item["seniority_risk"] = seniority["seniority_risk"]
        item["title_bucket_risk"] = seniority["title_bucket_risk"]
        item["opening_eligible"] = bool(seniority["opening_eligible"])
    else:
        item.setdefault("seniority_risk", "")
        item.setdefault("title_bucket_risk", "")
        item.setdefault("opening_eligible", None)
    if retrieval_recipe:
        item["retrieval_recipe"] = retrieval_recipe
    if hypothesis_ids:
        item["retrieval_hypothesis_ids"] = hypothesis_ids
    return item


def _materialize_retrieval_plan(
    plan: ExecutionPlan,
    *,
    base_design: RetrievalDesign | None = None,
    prefer_rendered_strings: bool = False,
) -> None:
    if not plan.retrieval_families:
        return
    design_payload = (base_design.to_dict() if base_design else {})
    design_payload["families"] = plan.retrieval_families
    rendered_design = RetrievalDesign.from_dict(design_payload)
    rendered_families, rendered_strings = render_retrieval_design(rendered_design)
    if rendered_families:
        plan.retrieval_families = rendered_families
    if not rendered_strings:
        return
    if not prefer_rendered_strings and plan.generated_strings:
        return

    # P2.5 (deliberate behavior change): when the model authored
    # generated_strings, THEY keep queue priority — rendered variants merge
    # AFTER them (dedup unchanged; first-seen wins, so a model-authored twin
    # suppresses its rendered duplicate). Previously rendered output was
    # prepended, pushing e.g. the model's filter-bounded strings to queue
    # positions 24-29 (observed 2026-06-18). When the model supplied only
    # families, rendered strings are the plan, exactly as before.
    merged: list[dict] = []
    seen_booleans: set[str] = set()
    for item in list(plan.generated_strings) + rendered_strings:
        boolean = str(item.get("boolean", "")).strip()
        if not boolean:
            continue
        key = boolean.lower()
        if key in seen_booleans:
            continue
        seen_booleans.add(key)
        merged.append(item)
    plan.generated_strings = merged


def _materialize_retrieval_adaptation(
    adaptation: AdaptationResponse,
    *,
    base_design: RetrievalDesign | None = None,
    prefer_rendered_strings: bool = False,
) -> None:
    if not adaptation.new_retrieval_families:
        return
    design_payload = (base_design.to_dict() if base_design else {})
    design_payload["families"] = adaptation.new_retrieval_families
    rendered_design = RetrievalDesign.from_dict(design_payload)
    _rendered_families, rendered_strings = render_retrieval_design(rendered_design)
    if rendered_strings and (prefer_rendered_strings or not adaptation.new_strings):
        # P2.5: same renderer-defers-to-model rule as _materialize_retrieval_plan —
        # the model's authored new_strings keep priority, rendered variants append.
        adaptation.new_strings = list(adaptation.new_strings) + rendered_strings


def _apply_role_strategy_hints(brief: Brief, plan: ExecutionPlan) -> None:
    """Attach role-class metadata and seed hint lane templates without touching strings."""
    if plan.retrieval_families:
        populate_execution_plan_lane_payloads(plan)
    apply_role_strategy_to_plan(brief, plan)


def _brief_hint_lanes(brief: Brief) -> set[str]:
    """The lane ids the BRIEF itself declared via ``domain_lane_hints``.

    This is the ACTIVATION signal for P7 Stage B lane validation and the
    collapse gate: auto-merged profile lane templates must not count here —
    the generic fallback profile merges at least one lane template for every
    brief (shared/role_strategy.py generic fallback), so gating on the full
    universe would make "hint-less" unreachable in the production call order
    (correctness lens, slice 12)."""
    lanes: set[str] = set()
    for hint in getattr(brief, "domain_lane_hints", None) or []:
        lane = getattr(hint, "lane", None)
        if lane is None and isinstance(hint, dict):
            lane = hint.get("lane")
        lane_id = _canonicalize_lane_id(str(lane or ""))
        if lane_id and lane_id != "general":
            lanes.add(lane_id)
    return lanes


def _declared_lane_universe(brief: Brief, plan: ExecutionPlan) -> set[str]:
    """The lane ids a string may carry without being flagged (P7 Stage B).

    hints ∪ declared sourcing-lane ids (which include the role-strategy
    profile's merged lane templates) ∪ ``general``. Everything is
    canonicalized through the same normalizer ``infer_domain_lane`` applies
    to the string values, so membership compares like with like. This is the
    ACCEPTANCE set only — activation is decided by ``_brief_hint_lanes``.
    """
    declared = {"general"} | _brief_hint_lanes(brief)
    for lane_dict in getattr(plan, "sourcing_lanes", []) or []:
        if isinstance(lane_dict, dict):
            lane_id = _canonicalize_lane_id(str(lane_dict.get("lane_id", "") or ""))
            if lane_id:
                declared.add(lane_id)
    return declared


def _nearest_declared_lane(lane: str, declared: set[str]) -> str | None:
    """Map an undeclared lane to a declared one iff EXACTLY one slug-family
    candidate exists (prefix at an underscore boundary, either direction).
    Two candidates means guessing — the caller keeps-and-flags instead."""
    candidates = [
        d
        for d in declared
        if d != "general" and (lane.startswith(f"{d}_") or d.startswith(f"{lane}_"))
    ]
    return candidates[0] if len(candidates) == 1 else None


def _validate_lane_items(items: list, declared: set[str]) -> None:
    """Remap-or-flag each item's ``domain_lane`` against the declared set."""
    for item in items:
        if not isinstance(item, dict):
            continue
        lane = str(item.get("domain_lane", "") or "")
        if not lane or lane in declared:
            continue
        nearest = _nearest_declared_lane(lane, declared)
        if nearest:
            # Remap is never silent (P6 posture): the raw value rides along.
            item["domain_lane_raw"] = lane
            item["domain_lane"] = nearest
            item.pop("undeclared_lane", None)
        else:
            item["undeclared_lane"] = True


def _validate_plan_lane_keys(brief: Brief, plan: ExecutionPlan) -> None:
    """P7 Stage B: model-emitted lanes are validated against the declared
    universe. ACTIVE only when the brief ITSELF declares lanes via
    domain_lane_hints — a hint-less brief instructs the model to DERIVE its
    own lanes, so specific-but-undeclared labels there are expected. (The
    activation gate deliberately ignores auto-merged profile lane templates;
    the acceptance set includes them.)"""
    if not _brief_hint_lanes(brief):
        return
    declared = _declared_lane_universe(brief, plan)
    _validate_lane_items(plan.generated_strings, declared)
    # Codex review (Wave 3): coverage gaps with a suggested_boolean are a
    # SECOND executable surface — they queue as SearchStrings — so they get
    # the same validation, or moving a string from generated_strings to
    # coverage_gaps bypasses the lane contract.
    executable_gaps = [
        gap
        for gap in (plan.coverage_gaps or [])
        if isinstance(gap, dict) and gap.get("suggested_boolean")
    ]
    _validate_lane_items(executable_gaps, declared)
    # Visibility (P5 posture — a flag nobody can see is a dead lever): carry
    # aggregate remap/undeclared counts on plan_warnings, computed from the
    # per-item markers across BOTH executable surfaces so re-annotation
    # passes stay idempotent.
    plan.plan_warnings = [
        w
        for w in (getattr(plan, "plan_warnings", []) or [])
        if w.get("code") not in {"lane_remapped", "undeclared_lane"}
    ]
    executable_items = list(plan.generated_strings) + executable_gaps
    remapped = [
        f"{i.get('domain_lane_raw')}→{i.get('domain_lane')}"
        for i in executable_items
        if isinstance(i, dict) and i.get("domain_lane_raw")
    ]
    undeclared = [
        str(i.get("domain_lane") or "")
        for i in executable_items
        if isinstance(i, dict) and i.get("undeclared_lane")
    ]
    if remapped:
        plan.plan_warnings.append(
            {
                "code": "lane_remapped",
                "count": len(remapped),
                "message": (
                    f"{len(remapped)} string lane(s) remapped to declared "
                    f"lanes: {', '.join(sorted(remapped))}"
                ),
            }
        )
    if undeclared:
        plan.plan_warnings.append(
            {
                "code": "undeclared_lane",
                "count": len(undeclared),
                "message": (
                    f"{len(undeclared)} string(s) carry lanes outside the "
                    f"declared universe: {', '.join(sorted(set(undeclared)))}"
                ),
            }
        )


def _lane_collapse_warning(brief: Brief, plan: ExecutionPlan) -> dict | None:
    """>60% of strings in one lane on a ≥2-declared-lane brief (P7 Stage B).

    "Declared" here means the BRIEF's own ``domain_lane_hints`` — the gate
    fires only when the brief itself declared a multi-lane universe the model
    then collapsed. Auto-merged profile lane templates deliberately do NOT
    count: the generic profile derives one lane per capability area, which
    would make every multi-area brief "≥2-declared" and warn spuriously on
    legitimately single-market briefs. (They DO count for per-string
    validation — see ``_declared_lane_universe``.) Plans with fewer than two
    lane-bearing strings are exempt."""
    declared = _brief_hint_lanes(brief)
    if len(declared) < 2:
        return None
    lanes = [
        str(item.get("domain_lane") or "")
        for item in plan.generated_strings
        if isinstance(item, dict) and item.get("domain_lane")
    ]
    if len(lanes) < 2:
        return None
    top_lane, top_count = Counter(lanes).most_common(1)[0]
    if top_count / len(lanes) <= 0.60:
        return None
    return {
        "code": "lane_collapse",
        "lane": top_lane,
        "count": top_count,
        "total": len(lanes),
        "message": (
            f"lane collapse: {top_count}/{len(lanes)} strings in '{top_lane}' "
            f"on a {len(declared)}-declared-lane brief — per-lane learning "
            "is degrading toward one bucket"
        ),
    }


_PLAN_SHAPE_WARNING_THRESHOLDS = {
    "max_skeleton_share": 0.5,
    "distinct_skeletons_min": 3,
    "not_usage_rate": 0.6,
    # Cry-wolf guard: tiny salvage plans are too small for the distinct-skeleton floor.
    "min_strings": 5,
}


def _attach_plan_shape_telemetry(plan: ExecutionPlan) -> None:
    """Warn when the primary plan's Boolean structure collapses."""
    plan.plan_warnings = [
        w
        for w in (getattr(plan, "plan_warnings", []) or [])
        if w.get("code") != "plan_shape_telemetry"
    ]

    metrics = plan_metrics({"generated_strings": plan.generated_strings})
    n_strings = metrics.get("n_strings")
    if (
        not isinstance(n_strings, int)
        or n_strings < _PLAN_SHAPE_WARNING_THRESHOLDS["min_strings"]
    ):
        return

    breaches: list[str] = []
    max_skeleton_share = metrics.get("max_skeleton_share")
    if (
        max_skeleton_share is not None
        and max_skeleton_share > _PLAN_SHAPE_WARNING_THRESHOLDS["max_skeleton_share"]
    ):
        breaches.append(
            "max_skeleton_share="
            f"{max_skeleton_share}>{_PLAN_SHAPE_WARNING_THRESHOLDS['max_skeleton_share']}"
        )
    distinct_skeletons = metrics.get("distinct_skeletons")
    if (
        distinct_skeletons is not None
        and distinct_skeletons < _PLAN_SHAPE_WARNING_THRESHOLDS["distinct_skeletons_min"]
    ):
        breaches.append(
            "distinct_skeletons="
            f"{distinct_skeletons}<{_PLAN_SHAPE_WARNING_THRESHOLDS['distinct_skeletons_min']}"
        )
    not_usage_rate = metrics.get("not_usage_rate")
    if (
        not_usage_rate is not None
        and not_usage_rate > _PLAN_SHAPE_WARNING_THRESHOLDS["not_usage_rate"]
    ):
        breaches.append(
            "not_usage_rate="
            f"{not_usage_rate}>{_PLAN_SHAPE_WARNING_THRESHOLDS['not_usage_rate']}"
        )
    if not breaches:
        return

    message = "plan shape telemetry: " + "; ".join(breaches)
    plan.plan_warnings.append(
        {
            "code": "plan_shape_telemetry",
            "message": message,
        }
    )
    print(f"  [plan-shape] {message}")


def _annotate_plan_metadata(brief: Brief, plan: ExecutionPlan) -> None:
    plan.generated_strings = [
        _annotate_string_metadata(brief, dict(item))
        for item in plan.generated_strings
    ]

    annotated_gaps = []
    for gap in plan.coverage_gaps:
        gap_item = dict(gap)
        if gap_item.get("suggested_boolean"):
            gap_item = _annotate_string_metadata(
                brief, gap_item, boolean_key="suggested_boolean"
            )
        annotated_gaps.append(gap_item)
    plan.coverage_gaps = annotated_gaps

    _validate_plan_lane_keys(brief, plan)


def _annotate_adaptation_metadata(
    brief: Brief,
    adaptation: AdaptationResponse,
    plan: ExecutionPlan | None = None,
) -> None:
    adaptation.new_strings = [
        _annotate_string_metadata(brief, dict(item))
        for item in adaptation.new_strings
    ]
    # P7 Stage B: adaptation is a second string-producing path (the Wave-2
    # lint gate learned this the hard way) — adapted strings get the same
    # lane validation as plan strings when the plan context is available.
    # Same activation gate as _validate_plan_lane_keys: brief hints only.
    if plan is not None and _brief_hint_lanes(brief):
        _validate_lane_items(
            adaptation.new_strings, _declared_lane_universe(brief, plan)
        )


# Fallback lane-rank map for strict-seniority briefs that declare no
# domain_lane_hints. Self-consistent, not universal: this sort only runs when
# is_strict_seniority_brief matched — which requires BFS text in the brief —
# so a BFS vocabulary fallback describes the brief's own market. Hinted briefs
# never touch it. VERTICAL-VOCAB(R2-F2): grandfathered under the ratchet.
_STRICT_SENIORITY_FALLBACK_LANE_RANKS: dict[str, float] = {
    "capital_markets": 0,
    "market_infra": 1,
    "market_data": 2,
    "risk_compliance": 3,
    "bfsi_vendors": 4,
    "general": 5,
    "asset_management": 6,
    "payments": 7,
    "insurance": 8,
}


def _strict_seniority_lane_ranks(brief: Brief) -> dict[str, float]:
    """Preferred-lane ranks for the strict-seniority opening sort.

    P1 item 4 (Wave 2): built from the brief's own domain_lane_hints order
    (first hint = rank 0) when hints exist; the BFS map is only the
    no-hints fallback. Either way, an UNKNOWN-but-specific lane ranks ABOVE
    general — the code must never punish exactly the specific labeling the
    lane-collapse fix asks the model for (audit R2-F2). The sentinel key
    ``__unknown_specific__`` carries that rank for lanes not in the map.
    """
    hint_lanes: list[str] = []
    for hint in getattr(brief, "domain_lane_hints", None) or []:
        lane = str(getattr(hint, "lane", "") or "").strip().lower()
        if lane and lane not in hint_lanes:
            hint_lanes.append(lane)
    if hint_lanes:
        ranks: dict[str, float] = {lane: float(i) for i, lane in enumerate(hint_lanes)}
        ranks.setdefault("general", float(len(hint_lanes)) + 1.0)
    else:
        ranks = {lane: float(rank) for lane, rank in _STRICT_SENIORITY_FALLBACK_LANE_RANKS.items()}
    ranks["__unknown_specific__"] = ranks["general"] - 0.5
    return ranks


def _strict_seniority_opening_sort_key(
    item: dict,
    idx: int,
    *,
    lane_ranks: dict[str, float],
) -> tuple[int, int, float, int, int, int]:
    lane = str(item.get("domain_lane", "") or "").strip().lower()
    preferred_lane_rank = lane_ranks.get(
        lane, lane_ranks["__unknown_specific__"] if lane else lane_ranks["general"]
    )
    top_tier_cutoff = lane_ranks["general"] - 1.0
    novelty_rank = 0 if item.get("novelty_bucket") == "edge_case" else 1
    title_risk_rank = {"low": 0, "medium": 1, "high": 2}.get(
        str(item.get("title_bucket_risk", "low")).lower(),
        0,
    )
    seniority_risk_rank = {"low": 0, "medium": 1, "high": 2}.get(
        str(item.get("seniority_risk", "low")).lower(),
        0,
    )
    return (
        0 if item.get("opening_eligible", True) else 1,
        0
        if (
            preferred_lane_rank <= min(3.0, top_tier_cutoff)
            or "executive director" in str(item.get("boolean", "")).lower()
        )
        else 1,
        preferred_lane_rank,
        title_risk_rank,
        seniority_risk_rank,
        idx,
    )


def _apply_strict_seniority_plan_guardrails(brief: Brief, plan: ExecutionPlan) -> str:
    if not is_strict_seniority_brief(brief):
        return ""

    kept: list[dict] = []
    suppressed: list[dict] = []
    for item in plan.generated_strings:
        if (
            item.get("title_bucket_risk") == "high"
            and not item.get("opening_eligible", True)
        ):
            suppressed.append(item)
            continue
        kept.append(item)

    if suppressed and (len(kept) >= 3 or len(kept) >= max(1, len(plan.generated_strings) // 2)):
        plan.generated_strings = kept
    else:
        suppressed = []

    lane_ranks = _strict_seniority_lane_ranks(brief)
    plan.generated_strings = [
        item
        for _, item in sorted(
            enumerate(plan.generated_strings),
            key=lambda pair: _strict_seniority_opening_sort_key(
                pair[1], pair[0], lane_ranks=lane_ranks
            ),
        )
    ]

    if plan.coverage_gaps:
        reordered_gaps = []
        for idx, gap in enumerate(plan.coverage_gaps):
            if not gap.get("suggested_boolean"):
                reordered_gaps.append((idx, gap))
                continue
            risk = classify_search_string_seniority(
                gap.get("suggested_boolean", ""),
                gap.get("rationale", "") or gap.get("gap", ""),
                domain_lane=gap.get("domain_lane", ""),
            )
            gap["seniority_risk"] = risk["seniority_risk"]
            gap["title_bucket_risk"] = risk["title_bucket_risk"]
            gap["opening_eligible"] = risk["opening_eligible"]
            reordered_gaps.append((idx, gap))
        plan.coverage_gaps = [
            gap
            for _, gap in sorted(
                reordered_gaps,
                key=lambda pair: _strict_seniority_opening_sort_key(
                    pair[1], pair[0], lane_ranks=lane_ranks
                ),
            )
        ]

    if not suppressed:
        return ""
    return f"strict-seniority lint suppressed {len(suppressed)} broad title-bucket strings"


def _apply_strict_seniority_adaptation_guardrails(
    brief: Brief,
    adaptation: AdaptationResponse,
    remaining_strings: list[SearchString],
) -> str:
    if not is_strict_seniority_brief(brief):
        return ""

    kept: list[dict] = []
    suppressed = 0
    for item in adaptation.new_strings:
        if (
            item.get("title_bucket_risk") == "high"
            and not item.get("opening_eligible", True)
        ):
            suppressed += 1
            continue
        kept.append(item)
    lane_ranks = _strict_seniority_lane_ranks(brief)
    adaptation.new_strings = [
        item
        for _, item in sorted(
            enumerate(kept),
            key=lambda pair: _strict_seniority_opening_sort_key(
                pair[1], pair[0], lane_ranks=lane_ranks
            ),
        )
    ]

    remaining_by_id = {ss.id: ss for ss in remaining_strings}
    for reorder in adaptation.reorder:
        if reorder.get("move_to") != "next":
            continue
        ss = remaining_by_id.get(reorder.get("string_id"))
        if not ss:
            continue
        if ss.opening_eligible is False or ss.title_bucket_risk == "high":
            reorder["move_to"] = "last"
            reason = reorder.get("reason", "").strip()
            suffix = "Demoted by strict-seniority guardrail because this string uses a broad title bucket."
            reorder["reason"] = f"{reason} {suffix}".strip()

    if suppressed == 0:
        return ""
    return f"strict-seniority lint suppressed {suppressed} adaptive broad title-bucket strings"


def _attach_boolean_lint_to_plan(brief: Brief, plan: ExecutionPlan) -> None:
    """Attach warning-only Boolean lint metadata without reordering strings."""
    attach_boolean_lint_to_plan(brief, plan)


def _apply_search_memory_to_plan(
    plan: ExecutionPlan,
    search_memory: dict | None,
) -> str:
    """Demote exhausted search families so prior overlap does not lead the run again."""
    family_records = get_search_memory_families(search_memory)
    if not family_records:
        return ""

    family_status = {
        family.get("family_key", ""): family
        for family in family_records
    }
    annotated: list[tuple[tuple[int, int], dict]] = []
    demoted = 0

    for idx, item in enumerate(plan.generated_strings):
        family = family_status.get(item.get("family_key", ""), {})
        exhausted = family.get("status") == "exhausted"
        if exhausted:
            demoted += 1
        annotated.append(
            (
                (
                    1 if exhausted else 0,
                    idx,
                ),
                item,
            )
        )

    annotated.sort(key=lambda pair: pair[0])
    plan.generated_strings = [item for _, item in annotated]
    if demoted == 0:
        return ""
    return f"demoted {demoted} strings from exhausted families"


def _apply_search_memory_to_adaptation(
    adaptation: AdaptationResponse,
    remaining_strings: list[SearchString],
    search_memory: dict | None,
) -> None:
    family_records = get_search_memory_families(search_memory)
    if not family_records:
        return

    family_status = {
        family.get("family_key", ""): family
        for family in family_records
    }

    adaptation.new_strings.sort(
        key=lambda item: (
            1
            if family_status.get(item.get("family_key", ""), {}).get("status") == "exhausted"
            else 0,
            0 if item.get("novelty_bucket") == "edge_case" else 1,
        )
    )

    remaining_by_id = {ss.id: ss for ss in remaining_strings}
    for reorder in adaptation.reorder:
        if reorder.get("move_to") != "next":
            continue
        ss = remaining_by_id.get(reorder.get("string_id"))
        if not ss:
            continue
        family = family_status.get(ss.family_key or "", {})
        if family.get("status") == "exhausted":
            reorder["move_to"] = "last"
            reason = reorder.get("reason", "").strip()
            suffix = "Demoted because this family is exhausted from prior runs."
            reorder["reason"] = f"{reason} {suffix}".strip()


def _first_brief_pattern(brief: Brief, field_name: str, fallback: str) -> str:
    for pattern in getattr(brief, field_name, None) or []:
        text = str(pattern or "").strip()
        if text:
            return text
    return fallback


def _augment_novelty_metrics(brief: Brief, plan: ExecutionPlan) -> None:
    """Append tapped-market novelty success criteria / pivot triggers.

    P1 item 3 (Wave 2): the canonical-pool vocabulary comes from the brief's
    canonical_* pattern mirrors — the same vocabulary _opening_priority
    already consumes — with vertical-agnostic fallbacks. The old FDE literals
    handed every tapped-market brief architecture metrics about a population
    it does not target (audit R2-F3).
    """
    title_example = _first_brief_pattern(
        brief, "canonical_title_patterns", "exact canonical titles"
    )
    framework_example = _first_brief_pattern(
        brief, "canonical_framework_patterns", "canonical tooling terms"
    )
    company_example = _first_brief_pattern(
        brief, "canonical_company_patterns", "canonical employers"
    )
    broad_example = _first_brief_pattern(
        brief, "canonical_broad_patterns", "broad canonical vocabulary"
    )

    success_metric = (
        "At least 50% of saves from the first 2 blocks come from adjacent or "
        f"edge-case pools rather than exact-title ({title_example}), "
        f"{framework_example}-first, or canonical-company ({company_example}) strings"
    )
    pivot_trigger = (
        f"Early saves cluster in exact-title ({title_example}), "
        f"{framework_example}-first, or canonical-company ({company_example}) "
        "pools without surfacing adjacent populations"
    )
    sequencing_trigger = (
        f"The opening block relies mainly on {broad_example} or direct "
        f"{framework_example} strings instead of edge-case transfer populations"
    )

    if success_metric not in plan.architecture_success_criteria:
        plan.architecture_success_criteria.append(success_metric)
    if pivot_trigger not in plan.architecture_pivot_triggers:
        plan.architecture_pivot_triggers.append(pivot_trigger)
    if sequencing_trigger not in plan.architecture_pivot_triggers:
        plan.architecture_pivot_triggers.append(sequencing_trigger)


def _rebalance_execution_plan_for_edge_case_opening(brief: Brief, plan: ExecutionPlan) -> str:
    """Promote edge-case coverage gaps and demote canonical opening strings."""
    promoted_gaps: list[dict] = []
    remaining_gaps: list[dict] = []

    for gap in plan.coverage_gaps:
        boolean = gap.get("suggested_boolean")
        if not boolean:
            remaining_gaps.append(gap)
            continue

        bucket, score = _opening_priority(
            brief, boolean, f"{gap.get('gap', '')} {gap.get('rationale', '')}"
        )
        if bucket != 2 and score >= 20 and len(promoted_gaps) < _MAX_PROMOTED_EDGE_CASE_GAPS:
            promoted_gaps.append(
                {
                    "boolean": boolean,
                    "rationale": f"Promoted coverage gap — {gap.get('gap', gap.get('rationale', 'edge-case population'))}",
                    "vocabulary_sources": "coverage_gap",
                }
            )
        else:
            remaining_gaps.append(gap)

    combined = promoted_gaps + list(plan.generated_strings)
    plan.generated_strings = _sort_strings_for_edge_case_opening(brief, combined)
    plan.coverage_gaps = remaining_gaps
    _augment_novelty_metrics(brief, plan)

    canonical_early = sum(
        1
        for item in plan.generated_strings[:8]
        if _opening_priority(brief, item.get("boolean", ""), item.get("rationale", ""))[0] == 2
    )
    return (
        f"promoted {len(promoted_gaps)} edge-case coverage gaps; "
        f"reordered opening to reduce canonical cleanup strings "
        f"(canonical in first 8: {canonical_early})"
    )


def _rebalance_adaptation_for_edge_case_opening(
    brief: Brief,
    adaptation: AdaptationResponse,
    remaining_strings: list[SearchString],
) -> AdaptationResponse:
    adaptation.new_strings = _sort_strings_for_edge_case_opening(brief, adaptation.new_strings)

    remaining_by_id = {ss.id: ss for ss in remaining_strings}
    for reorder in adaptation.reorder:
        if reorder.get("move_to") != "next":
            continue
        ss = remaining_by_id.get(reorder.get("string_id"))
        if not ss:
            continue
        bucket, _score = _opening_priority(brief, ss.boolean, ss.name)
        if bucket == 2:
            reorder["move_to"] = "last"
            reason = reorder.get("reason", "").strip()
            suffix = "Demoted because this is a canonical cleanup string in a tapped market."
            reorder["reason"] = f"{reason} {suffix}".strip()

    return adaptation


_BLACKLISTED_EMPLOYER_LAYER_KEYS = (
    "entry_signals",
    "capability_proxies",
    "reality_filters",
    "context_constraints",
)


def _normalized_employer_key(value: object) -> str:
    # Exact normalized equality is deliberate: a blacklist entry like
    # "Acme" must not scrub a legitimate employer like "Alan Turing Institute".
    return " ".join(str(value or "").split()).casefold()


def _blacklisted_employer_keys(brief: Brief) -> set[str]:
    return {
        key
        for key in (
            _normalized_employer_key(entry)
            for entry in (getattr(brief, "employer_blacklist", None) or [])
        )
        if key
    }


def _scrub_blacklisted_employer_payloads(
    *,
    blacklist_keys: set[str],
    families: list[dict],
    strings: list[dict],
    warnings_sink: list[dict],
    family_root: str,
    string_root: str,
) -> None:
    removals: list[tuple[str, str]] = []

    def scrub_values(values: object, location: str) -> None:
        if not isinstance(values, list):
            return
        kept = []
        for value in values:
            if _normalized_employer_key(value) in blacklist_keys:
                employer = " ".join(str(value).split())
                removals.append((employer, location))
                warnings_sink.append(
                    {
                        "code": "blacklisted_employer_scrubbed",
                        "message": (
                            f'Scrubbed blacklisted employer "{employer}" from {location}.'
                        ),
                    }
                )
            else:
                kept.append(value)
        values[:] = kept

    for family_index, family in enumerate(families or []):
        if not isinstance(family, dict):
            continue
        scrub_values(
            family.get("target_employers"),
            f"{family_root}[{family_index}].target_employers",
        )
        for layer_key in _BLACKLISTED_EMPLOYER_LAYER_KEYS:
            layer_items = family.get(layer_key)
            if not isinstance(layer_items, list):
                continue
            for item_index, item in enumerate(layer_items):
                if not isinstance(item, dict):
                    continue
                if item.get("structured_surface") != "linkedin_company_filter":
                    continue
                scrub_values(
                    item.get("terms"),
                    f"{family_root}[{family_index}].{layer_key}[{item_index}].terms",
                )

    for string_index, item in enumerate(strings or []):
        if not isinstance(item, dict):
            continue
        structured_filters = item.get("structured_filters")
        if not isinstance(structured_filters, dict):
            continue
        scrub_values(
            structured_filters.get("companies"),
            f"{string_root}[{string_index}].structured_filters.companies",
        )

    if removals:
        names = ", ".join(sorted({employer for employer, _location in removals}))
        location_count = len({location for _employer, location in removals})
        print(
            "  [blacklist-scrub] removed blacklisted employer(s) "
            f"{names} from {location_count} plan location(s)"
        )


def _scrub_blacklisted_employers(brief: Brief, plan: ExecutionPlan) -> None:
    blacklist_keys = _blacklisted_employer_keys(brief)
    if not blacklist_keys:
        return
    _scrub_blacklisted_employer_payloads(
        blacklist_keys=blacklist_keys,
        families=plan.retrieval_families,
        strings=plan.generated_strings,
        warnings_sink=plan.plan_warnings,
        family_root="retrieval_families",
        string_root="generated_strings",
    )


def _scrub_blacklisted_employer_adaptation(
    brief: Brief,
    adaptation: AdaptationResponse,
    execution_plan: ExecutionPlan | None,
) -> None:
    blacklist_keys = _blacklisted_employer_keys(brief)
    if not blacklist_keys:
        return
    warnings_sink = execution_plan.plan_warnings if execution_plan is not None else []
    _scrub_blacklisted_employer_payloads(
        blacklist_keys=blacklist_keys,
        families=adaptation.new_retrieval_families,
        strings=adaptation.new_strings,
        warnings_sink=warnings_sink,
        family_root="new_retrieval_families",
        string_root="new_strings",
    )


# ---------------------------------------------------------------------------
# Strategy formation (run start)
# ---------------------------------------------------------------------------

def _finalize_execution_plan(
    brief: Brief,
    plan: ExecutionPlan,
    prior_run_data: dict | None,
) -> ExecutionPlan:
    """Run the full post-parse pipeline on a freshly parsed strategy plan.

    P8.1: both the ``form_strategy`` success path and the salvage path call
    this — a truncated response can no longer skip pipeline steps (edge-case
    rebalance, search-memory demotion, ``original_architecture`` stamping).
    The plans the two callers produce differ only in their input; the callers'
    own prints differ (the salvage ``[warn]`` line vs the success path's
    architecture-summary and completion lines).
    """
    explicit_design = _explicit_design_from_brief(brief)
    use_layered_retrieval = explicit_design.is_explicit()
    _materialize_retrieval_plan(
        plan,
        base_design=explicit_design if use_layered_retrieval else None,
        prefer_rendered_strings=use_layered_retrieval,
    )
    # Render first: explicit retrieval_design shared layers and edge-case
    # overlays merge into families/strings only during materialization. Scrubbing
    # here catches both brief-carried layers and model-carried family layers
    # before role hints build executable lane projections.
    _scrub_blacklisted_employers(brief, plan)
    _apply_role_strategy_hints(brief, plan)
    _annotate_plan_metadata(brief, plan)
    strict_summary = _apply_strict_seniority_plan_guardrails(brief, plan)
    if strict_summary:
        print(f"  Strict-seniority guardrail: {strict_summary}")
    if _brief_targets_edge_case_opening(brief):
        summary = _rebalance_execution_plan_for_edge_case_opening(brief, plan)
        _annotate_plan_metadata(brief, plan)
        strict_summary = _apply_strict_seniority_plan_guardrails(brief, plan)
        if strict_summary:
            print(f"  Strict-seniority guardrail: {strict_summary}")
        print(f"  Edge-case rebalance: {summary}")
    memory_summary = _apply_search_memory_to_plan(
        plan,
        (prior_run_data or {}).get("search_memory_summary"),
    )
    if memory_summary:
        print(f"  Search-memory rebalance: {memory_summary}")
    # Slice A part 4: repair obvious keyword/structured mismatches on the lane dicts
    # BEFORE compiling, so a flipped title surface folds into structured_filters.
    repair_constraint_surfaces(plan)
    apply_linkedin_lane_compiler_to_plan(plan)
    _attach_boolean_lint_to_plan(brief, plan)
    _attach_plan_shape_telemetry(plan)
    # P7 Stage B: deterministic lane-collapse warning — printed here and
    # carried on the plan so every block report (and adaptation) sees it.
    plan.plan_warnings = [
        w
        for w in (getattr(plan, "plan_warnings", []) or [])
        if w.get("code") != "lane_collapse"
    ]
    collapse = _lane_collapse_warning(brief, plan)
    if collapse:
        plan.plan_warnings.append(collapse)
        print(f"  [lane-collapse] {collapse['message']}")
    plan.original_architecture = plan.architecture  # Set once, never updated on pivot
    return plan


def form_strategy(
    brief: Brief,
    kit_strings: list[KitString],
    prior_run_data: dict | None = None,
    lane_feedback: list[dict] | None = None,
    shadow_dir: Path | None = None,
) -> ExecutionPlan:
    """Ask Opus to synthesize compound search strings from kit vocabulary.

    Kit strings are vocabulary — building blocks organized by competency domain.
    Opus uses them to create targeted compound Boolean strings for execution.

    Args:
        brief: Normalized Brief dataclass.
        kit_strings: Boolean vocabulary extracted from the kit.
        prior_run_data: Optional — performance data from a previous run for resume.
        shadow_dir: Optional — destination for shadow-strategist artifacts
            (plans/sourcing-generality-hardening.md item 19). When set AND
            config.SHADOW_STRATEGY_ENABLED, the primary's exact prompts are
            replayed against the shadow model on a background worker; None
            (every non-orchestrator caller) means no shadow regardless of flag.

    Returns:
        ExecutionPlan with generated compound strings and coverage gaps.
    """
    explicit_design = _explicit_design_from_brief(brief)
    use_layered_retrieval = explicit_design.is_explicit()
    system = _build_strategy_system(
        brief,
        has_kit=bool(kit_strings),
        use_layered_retrieval=use_layered_retrieval,
    )
    user_prompt = _build_strategy_user(
        brief,
        kit_strings,
        prior_run_data,
        use_layered_retrieval=use_layered_retrieval,
        lane_feedback=lane_feedback,
    )

    print(f"  Strategizing... ({_config.STRATEGY_MODEL_NAME.rsplit('/', 1)[-1]} is synthesizing compound search strings)")
    try:
        usage_context = {
            "stage": "linkedin_strategy_form",
            "brief_id": brief.id,
        }
        result = opus_llm(
            system,
            user_prompt,
            expect_json=True,
            # max_tokens caps THINKING + text jointly on always-thinking
            # models (claude-fable-5): at 16384 a full formation plan
            # (~6K text tokens) plus Fable's reasoning truncated at
            # stop_reason=max_tokens on the 2026-07-05 live run (and on
            # one of three same-day replays). 32768 gives the strategy
            # tier headroom under the 300s client timeout; Opus-family
            # primaries (thinking off) are unaffected — it is a cap,
            # not a target.
            max_tokens=32768,
            usage_context=usage_context,
            model_name=_config.STRATEGY_MODEL_NAME,
        )
        # Shadow strategist (plans/sourcing-generality-hardening.md item 19):
        # fire-and-forget, with FRESH context (2026-07-05 experiment-design
        # fix). The primary's user prompt may carry `## Prior Run Data`,
        # search-family memory ("PROVEN VEINS"), and lane-feedback diffs —
        # past-run performance the experiment must exclude: on the
        # 2026-07-05 live run the shadow's thinking opened with "I'm
        # looking at the results from my previous runs". The shadow's
        # question is what it produces from the brief alone, so its user
        # prompt is rebuilt WITHOUT prior_run_data / lane_feedback (every
        # other section of _build_strategy_user is brief/kit-derived); the
        # system prompt is brief-derived and shared as-is. Placed before
        # ExecutionPlan.from_dict so a downstream shape failure still
        # yields the comparison artifact; never fires when the primary call
        # itself raised (nothing comparable exists), so the salvage /
        # fallback paths below stay untouched. dispatch never raises.
        if _config.SHADOW_STRATEGY_ENABLED and shadow_dir is not None:
            shadow_user_prompt = _build_strategy_user(
                brief,
                kit_strings,
                None,
                use_layered_retrieval=use_layered_retrieval,
                lane_feedback=None,
            )
            dispatch_strategy_shadow(
                stage="linkedin_strategy_form",
                system_prompt=system,
                user_prompt=shadow_user_prompt,
                max_tokens=32768,
                shadow_dir=shadow_dir,
                primary_meta={
                    "primary_model": _config.STRATEGY_MODEL_NAME,
                    "metrics": plan_metrics(
                        result,
                        reference_text=system + "\n" + user_prompt,
                        novelty_reference="system+user",
                    ),
                    # The primary's actual plan, so the artifact renders both
                    # sides of the comparison without the run log.
                    "raw_response": result,
                },
                shadow_prompt_context="fresh",
                primary_prompt_included_prior_run_data=bool(prior_run_data),
            )
        plan = ExecutionPlan.from_dict(result)
        plan = _finalize_execution_plan(brief, plan, prior_run_data)
        if plan.architecture:
            print(f"  Architecture: {plan.architecture} — {plan.architecture_rationale[:120]}")
        print("  Strategy complete.")
        return plan
    except Exception as e:
        # Try to salvage a partial JSON response
        plan = _try_salvage_strategy(e)
        if plan:
            plan = _finalize_execution_plan(brief, plan, prior_run_data)
            print(f"  [warn] Strategy JSON was truncated — salvaged partial plan", file=sys.stderr)
            return plan

        # Fall back — no kit strings to queue, just an empty plan
        print(f"  [warn] Strategy formation failed ({e}) — no strings to execute", file=sys.stderr)
        return _default_strategy(kit_strings)


# ---------------------------------------------------------------------------
# Registry adapter (Multi-Agent Execution Plan Slice 1.6)
# ---------------------------------------------------------------------------

def form_strategy_for_registry(
    brief: Brief,
    prior_run_data: dict | None = None,
) -> ExecutionPlan:
    """Uniform-signature adapter wrapping :func:`form_strategy`.

    The launcher registry's ``form_strategy_fn`` field
    (``cloris/launchers/__init__.py:196``) declares
    ``Callable[..., ExecutionPlan]``; native LinkedIn strategy formation
    requires three positional inputs (``brief``, ``kit_strings``,
    ``prior_run_data``). This adapter sources ``kit_strings`` exactly
    the way :class:`linkedin.orchestrator.LinkedInOrchestrator` does at
    Phase 2 (`linkedin/orchestrator.py:1019-1031`): pull from
    ``brief.kit_url`` if set, else fall back to the documented
    "JD context only" path with an empty kit.

    Native callers (``LinkedInOrchestrator.run_full``) continue to call
    :func:`form_strategy` directly with a kit they extracted earlier in
    the run; the adapter is additive surface for chief-of-staff
    dispatch (Phase 2.5) and does not displace them.
    """

    kit_strings: list[KitString] = []
    kit_url = getattr(brief, "kit_url", "") or ""
    if kit_url:
        from shared.kit_extractor import extract_kit_strings

        kit_strings = extract_kit_strings(kit_url)
        # P5 (Wave 2): advisory craft check — kit strings are vocabulary,
        # never queued, so a defect summary print is the right weight.
        kit_lint_summary = summarize_kit_lint(kit_strings)
        if kit_lint_summary:
            print(f"  [kit-lint] {kit_lint_summary}")
    return form_strategy(brief, kit_strings, prior_run_data)


def _render_declared_lanes_section(brief: Brief) -> str:
    """Render the brief's declared domain lanes as prompt vocabulary.

    P7 Stage A (plans/sourcing-rigor-hardening.md): the lane-collapse root cause
    was that the prompt's only example lanes were BFSI-specific, so every
    non-BFSI brief labeled all strings "general" and per-lane learning collapsed
    into one bucket. When the brief carries ``domain_lane_hints`` (preflight now
    emits them), they become the declared lane vocabulary; the schema text
    handles the no-hints case with a derive-your-own instruction.
    """
    lanes = [
        str(getattr(hint, "lane", "") or "").strip()
        for hint in (getattr(brief, "domain_lane_hints", None) or [])
    ]
    lanes = [lane for lane in lanes if lane]
    if not lanes:
        return ""
    return (
        "\n## Declared Domain Lanes\n"
        f"This brief declares the market lanes: {', '.join(lanes)}.\n"
        "Label every generated string's \"domain_lane\" with one of these lanes; "
        "use \"general\" only for a string that genuinely fits none of them. Lane "
        "labels feed per-lane learning across runs — a run labeled all-\"general\" "
        "cannot learn which market segments work.\n"
    )


def _build_strategy_system(
    brief: Brief,
    has_kit: bool = True,
    *,
    use_layered_retrieval: bool = False,
) -> str:
    """Dispatch to the de-prescribed prompt on the live no-kit, non-layered
    path (plans/formation-prompt-de-prescribed.md, approved 2026-07-05);
    kit and layered modes keep the legacy builder until their own slices."""
    if not has_kit and not use_layered_retrieval:
        return _build_strategy_system_deprescribed(brief)
    return _build_strategy_system_legacy(
        brief, has_kit, use_layered_retrieval=use_layered_retrieval
    )


def _build_strategy_system_legacy(
    brief: Brief,
    has_kit: bool = True,
    *,
    use_layered_retrieval: bool = False,
) -> str:
    total_count_guidance = "Generate 8-18 retrieval families total." if use_layered_retrieval else "Generate 15-30 search strings total."
    mix_label = "family types" if use_layered_retrieval else "string types"
    # Shape-freedom renders ONLY non-layered: there is no correct clause/term
    # count. Layered mode keeps its structural family contract
    # (retrieval_contract_guidance), so the sentence would contradict it there.
    shape_doctrine = (
        ""
        if use_layered_retrieval
        else "There is no correct number of parentheticals or terms — a precise string may be a single specific term, a broad net several synonym clusters. Let each string's angle dictate its shape.\n\n"
    )
    recall_label = "Recall families" if use_layered_retrieval else "Recall strings"
    precision_label = (
        "Precision \"sniper\" families"
        if use_layered_retrieval
        else "Precision \"sniper\" strings"
    )
    # Slot-notation craft exemplars render only for keyword-string composition;
    # layered mode keeps its structural family contract instead.
    angle_shape_exemplars = "" if use_layered_retrieval else f"\n{_ANGLE_SHAPE_EXEMPLARS}"


    noise_section = ""
    if brief.noise_archetypes:
        noise_section = f"\n## Noise Archetypes\n{json.dumps(brief.noise_archetypes, indent=2)}"

    known_noise_section = ""
    if brief.known_noise_patterns:
        known_noise_section = f"\n## Known Noise Patterns\n{json.dumps(brief.known_noise_patterns, indent=2)}"

    key_terms_section = ""
    candidate_terms_by_area = getattr(brief, "candidate_register_terms_by_area", {}) or {}
    if candidate_terms_by_area:
        kt_lines = ["\n## Discriminating Vocabulary by Capability Area"]
        for area, terms in candidate_terms_by_area.items():
            kt_lines.append(f"- {area}: {', '.join(terms)}")
        kt_lines.append("\nUse these terms as candidate self-description anchors for Type B precision strings. They are the profile vocabulary qualified candidates plausibly write for each area.")
        key_terms_section = "\n".join(kt_lines)
    elif brief.key_terms_by_area:
        kt_lines = ["\n## Discriminating Vocabulary by Capability Area"]
        for area, terms in brief.key_terms_by_area.items():
            kt_lines.append(f"- {area}: {', '.join(terms)}")
        kt_lines.append("\nUse these terms as anchors for Type B precision strings. They are the specific technical vocabulary that distinguishes qualified candidates in each area.")
        key_terms_section = "\n".join(kt_lines)

    market_hint = f" Brief specifies: **{brief.market_density}**." if brief.market_density else ""

    declared_lanes_section = _render_declared_lanes_section(brief)

    example_compounds_block = _render_example_compounds_block(brief)
    example_compounds_section = (
        "\n\nBrief-supplied compound hints (vocabulary evidence from preflight "
        "— two of many possible angles, NOT a shape template; author your own "
        f"shapes per angle):\n{example_compounds_block}"
        if example_compounds_block
        else ""
    )

    sequencing_text = str(getattr(brief, "sequencing_heuristics", "") or "").strip()
    sequencing_section = (
        f"\n\nSEQUENCING:\n{sequencing_text}" if sequencing_text else ""
    )

    # The tapped-market playbook binds ONLY when the brief itself declares a
    # worked market (_brief_targets_edge_case_opening — same gate as the
    # deterministic opening sort). Authored in the strict-seniority era
    # (a2da0d9) for a brief whose market genuinely was tapped; rendered
    # unconditionally it taught formation to flee the canonical pool on
    # every JD-only brief — 2026-07-04 SPL RCA, root cause 1.
    tapped_market_section = (
        """### Tapped-Market / Edge-Case Opening (this brief declares the obvious pool is exhausted)

Your OPENING SEQUENCE must prioritize non-obvious adjacent populations rather than the canonical role vocabulary.

This is a SEQUENCING and pool-selection rule about which populations to open with. It is NOT a rule about which surface a token sits on. Whenever a literal title or named employer DOES appear in a string — opening or later — the keyword-vs-filter guidance above still governs: a deliberate pool bound goes on `linkedin_title_filter` / `linkedin_company_filter`, regardless of which lane the string serves.

Use this mental loop:
1. First ask: what would a generally solid technical sourcer search if they were doing a competent but standard pass for this role?
2. Then ask: which same-caliber candidates would that standard pass systematically miss because they use different titles, different product language, or sit in adjacent org structures?
3. Generate your opening strings primarily for THOSE missed populations.

For the FIRST 8 strings:
- At least 5 must target edge-case or transfer populations.
- Prefer intersections of an adjacent-capability population with the brief's own edge-case vocabulary — adjacent org structures, internal-tooling builders, delivery/implementation roles, practitioners who describe the work in product or problem language rather than the canonical role vocabulary, consultancy ICs with build evidence, or vertical specialists whose day job implies the capability. Draw the concrete terms from the brief's edge-case patterns and capability areas, never from a stock list.
- Do NOT open with strings whose primary POOL is the canonical population — i.e. strings that target the exact target-title pool, the canonical-employer pool, or a fashionable framework-name pool as their main reason to exist. This is about the pool, not the surface.
- Do NOT front-load broad core strings (the canonical role-vocabulary AND production-proof shape) unless they are crossed with a non-obvious adjacent-population qualifier.
- An edge-case opening string targets a non-canonical pool BY DESIGN; its entry_signals are recall doorways. So it should NOT carry a `linkedin_title_filter` set to the literal target title or a `linkedin_company_filter` set to the canonical employer pool — that facet would bound the pool back to exactly the canonical population the opening is trying to skip. This is the one place the keyword-vs-filter default is overridden, and it is overridden because of the POOL the string targets, not the token class.

The canonical pool still matters; it belongs in later cleanup passes once the edge-case populations have been tested. When a later string DOES intentionally target the canonical pool, the keyword-vs-filter guidance applies normally — a deliberate title/employer bound goes on its filter.

NOVELTY ACCOUNTING (MANDATORY for tapped markets):
- In a tapped market, productivity alone is not enough. Saves from exact-title pools, canonical-employer pools, and framework-first strings are useful confirmation but LOW-NOVELTY signal.
- Do NOT interpret a high save rate from those canonical pools as proof the opening sequence is correct.
- When you set architecture success criteria and pivot triggers, include at least one metric about novelty or pool mix, not just save rate and result count.

"""
        if _brief_targets_edge_case_opening(brief)
        else ""
    )

    abbreviation_block = _render_abbreviation_collisions_block(brief)
    abbreviation_text = (
        f"""### Abbreviation Collision Filter

An abbreviation MUST NOT appear standalone if it has a more common non-domain meaning on LinkedIn.

{abbreviation_block}

Rule: an abbreviation that fails alone IS acceptable when paired with its full expansion in the same OR group."""
        if abbreviation_block
        else ""
    )

    blacklist_block = _render_term_blacklist_block(brief)
    blacklist_text = (
        f"""### Blacklist — NEVER Include

{blacklist_block}"""
        if blacklist_block
        else ""
    )

    abbreviation_blacklist_block = "\n\n".join(
        section for section in (abbreviation_text, blacklist_text) if section
    )
    abbreviation_blacklist_section = (
        f"\n{abbreviation_blacklist_block}\n" if abbreviation_blacklist_block else ""
    )
    structured_surface_guidance = (
        "\n\n### Keyword clause vs. structured filter — choose the surface for each token\n"
        "Boolean creativity is the product advantage and stays FIRST-CLASS. Structured "
        "LinkedIn filters (`linkedin_title_filter`, `linkedin_company_filter`) are TACTICAL "
        "LEVERS layered onto a Boolean-led search, not a replacement for it — every search "
        "still has a working Boolean and falls back to Boolean if a filter cannot apply. So "
        "the default surface for a token is the Boolean keyword. Reach for a structured "
        "filter only when a token is a literal value you are using to DELIBERATELY BOUND the "
        "pool — not merely because the token happens to be a title or a company name.\n\n"
        "Set `structured_surface` when ALL of these hold:\n"
        "- the value is a literal facet LinkedIn indexes — a real job title practitioners "
        "carry (the brief's exact target title is the canonical example) -> "
        "`linkedin_title_filter`, or a real employer NAME (one of the brief's named "
        "employer clusters) -> `linkedin_company_filter`; AND\n"
        "- you intend it as a hard pool bound for this lane (the lane's job is to search "
        "INSIDE that title/employer set), not as a recall doorway fishing for adjacent "
        "people.\n\n"
        "Keep the token a keyword (OMIT `structured_surface`) when:\n"
        "- it is fuzzy or semantic — capability signals, tool/framework names, workflow "
        "descriptions, domain/industry concepts;\n"
        "- it is a near-analog title used as a recall DOORWAY rather than the literal target "
        "role (searching an adjacent title to fish for target-adjacent builders). A "
        "filter is an exact-match bound and would defeat the doorway intent;\n"
        "- the lane's purpose is to reach a non-canonical adjacent population — a facet would "
        "bound it back to the canonical pool, the opposite of the intent.\n\n"
        "NEVER place the same dimension on BOTH surfaces. If an employer set is a "
        "`linkedin_company_filter`, do NOT also list those companies in a keyword OR-clause; "
        "if a title is a `linkedin_title_filter`, do NOT also OR the title variants as "
        "keywords. A dimension is carried by exactly one surface — the filter bounds the "
        "pool, the keyword does not. Collapsing named employers into an "
        "(\"Employer A\" OR \"Employer B\" OR …) keyword clause is the anti-pattern: it "
        "matches the strings as profile text and does not bound the candidate pool.\n\n"
        "A literal title/employer facet belongs on `entry_signals` (the pool the lane starts "
        "from) or `context_constraints` (and-they-work-at) — never on `capability_proxies` "
        "or `reality_filters`, which exist for skill and execution-proof keyword signals.\n\n"
        "Worked example — entry_signals item bounding the pool by a literal title:\n"
        "  {\"item_id\": \"entry_exact_title\", \"label\": \"Literal target-title pool\", "
        "\"terms\": [\"<the brief's exact target title>\"], "
        "\"structured_surface\": \"linkedin_title_filter\", "
        "\"rationale\": \"Lane searches inside the literal target title — a pool bound, not a doorway.\"}\n"
        "Worked example — context_constraints item bounding by named employers:\n"
        "  {\"item_id\": \"ctx_named_employers\", \"label\": \"Named employer cluster\", "
        "\"terms\": [\"<Employer A>\", \"<Employer B>\"], "
        "\"structured_surface\": \"linkedin_company_filter\", "
        "\"rationale\": \"Deliberate employer bound for this lane; not also OR'd as keywords.\"}\n"
        "When you set a `structured_surface`, record the one-line WHY in the item's "
        "`label`/`rationale` (why this is a deliberate pool bound, not a recall doorway)."
    )
    per_string_filter_guidance = (
        "\n\n### Per-string structured filters — the executable lever on generated_strings\n"
        "Each generated string runs as a Boolean keyword search by default. You MAY also "
        "attach a `structured_filters` object to a string to bound its pool with a LinkedIn "
        "sidebar facet that runs ALONGSIDE the keyword Boolean: the keyword still runs, the "
        "facet bounds the candidate pool, and if a facet cannot apply the search falls back "
        "to keyword-only. This is the strongest lever you have — a company facet matches "
        "actual employees via LinkedIn's company index, not the literal company string in "
        "profile text — so use it deliberately, per these rules:\n\n"
        "- companies: set `structured_filters.companies` to a named employer set when that "
        "set IS the pool this string targets (a regional consultancy cluster, one of the "
        "brief's named employer clusters, a specific employer). This REPLACES collapsing those employers into a "
        "keyword OR-clause — never do both. EXCEPTION: do NOT put the canonical "
        "employer pool on a company facet in an edge-case opener whose purpose is to reach a "
        "NON-canonical adjacent population — that bounds the pool right back to the canonical "
        "set the opener is trying to skip; keep those as keyword doorways or omit them.\n"
        "- titles: set `structured_filters.titles` ONLY on a canonical or cleanup pass that "
        "bounds to an EXACT target title the brief names. For every recall / edge-case / "
        "doorway string — where a title is a near-analog used to fish for adjacent people — "
        "keep titles as keyword OR-groups, NEVER a facet (a title facet is an exact-match "
        "bound and defeats the doorway).\n"
        "- locations: NEVER set a location facet here. Geography is applied once per run from "
        "the brief, outside the per-string filters. Do not emit a location facet.\n\n"
        "Harmony rule (MANDATORY): a value lives on exactly ONE surface. If a company or "
        "title is on `structured_filters`, it must NOT also appear in that string's keyword "
        "Boolean, and vice versa. The facet bounds the pool; the keyword matches profile "
        "text — duplicating wastes the lever and double-bounds the search.\n"
        "Shape: \"structured_filters\": {\"companies\": [\"<Employer A>\", \"<Employer B>\"], "
        "\"titles\": [\"<the brief's exact target title>\"]}. Omit the key entirely for a pure "
        "keyword string — the default and most common case."
    )
    retrieval_contract_guidance = (
        "Every search family should be expressed as:\n"
        "- entry_signals\n"
        "- capability_proxies\n"
        "- reality_filters\n"
        "- optional context_constraints\n"
        "- optional anti_noise\n"
        "- optional edge-case hypothesis overlays\n\n"
        "The deterministic renderer downstream will convert these into executable booleans. "
        "retrieval_families are the primary planning contract for this brief."
        if use_layered_retrieval
        else "You may optionally include retrieval_families as structured metadata, "
        "but generated_strings remain the primary planning contract for this brief."
    ) + structured_surface_guidance + per_string_filter_guidance
    matching_guidance = render_strategy_matching_guidance()
    if use_layered_retrieval:
        task_object = "layered retrieval families and rendered search strings"
    elif has_kit:
        task_object = "compound Boolean search strings"
    else:
        task_object = "Boolean search strings"

    if has_kit:
        task_source_clause = "by combining terms from multiple kit clusters with domain qualifiers from the brief"
    elif use_layered_retrieval:
        task_source_clause = "from the role requirements, using LinkedIn-compatible Boolean syntax"
    else:
        task_source_clause = "from the role requirements, using LinkedIn-compatible Boolean syntax (quoted phrases, AND, OR, NOT, parentheses)"

    task_intro = (
        "You are given a Search Kit — a library of Boolean search terms organized by competency domain. These kit strings are YOUR VOCABULARY — raw building blocks, NOT executable queries. Do NOT include kit strings directly in the execution queue."
        if has_kit
        else (
            "No pre-built search kit is available. You will generate compound Boolean strings directly from the role description, archetypes, JD context, and any sourcing instructions provided."
            if use_layered_retrieval
            else "No pre-built search kit is available. You will generate Boolean search strings directly from the role description, archetypes, JD context, and any sourcing instructions provided."
        )
    )

    layered_intersection_guidance = (
        f"Create retrieval families that AND-gate high-signal layers {'from different kit clusters with' if has_kit else 'with'} domain/seniority qualifiers from the brief.\n\n"
        if use_layered_retrieval
        else ""
    )
    syntax_guidance = (
        "- Use LinkedIn-compatible Boolean syntax: parenthetical groups joined by AND"
        if use_layered_retrieval
        else "- Use LinkedIn-compatible Boolean syntax: quoted phrases, AND, OR, NOT, and parentheses"
    )

    return f"""You are a senior sourcing strategist planning a Boolean search execution for LinkedIn Recruiter.

Role: {brief.role_title}
{brief.role_description}

## Minimum Bar
{brief.minimum_bar}

## Archetypes
{json.dumps(brief.archetypes, indent=2)}
{noise_section}
{known_noise_section}
{key_terms_section}
{declared_lanes_section}
## Permanent Filters
{json.dumps(brief.permanent_filters, indent=2)}

## Stage 0: Classify the Search Architecture (telemetry, not a shape mandate)

Before generating any strings, analyze the role and market and classify your approach as a search ARCHITECTURE. The label is recorded for pivot review and mid-run adaptation context — it does NOT prescribe string counts, type ratios, or shapes. Composition is governed by the portfolio principles below, for every architecture.

Consider:
1. **Title distinctiveness:** Is the role title specific and reliably used, or ambiguous/variable?
2. **Market vocabulary consistency:** Do practitioners describe themselves consistently, or in many different ways?
3. **Market density:** Large pool (thousands in this geo) or sparse (dozens)?{market_hint}
4. **Company concentration:** Talent concentrated in known companies, or widely distributed?
5. **Noise landscape:** Easier to define what you want, or what you don't want?
6. **Your familiarity:** Strong kit vocabulary for this role, or general JD terms?

Available architectures:

1. **sniper** — Distinctive titles, large market; precision-led. Best when: the role title is specific, the market is large enough for exact matches, and false positives are expensive.

2. **dragnet** — Ambiguous titles, inconsistent vocabulary; recall-led, noise expected. Best when: practitioners lack standard titles, vocabulary is fragmented, over-include rather than miss people.

3. **titration** — Unknown market. Open with a few broad recon strings; hold most of your budget for post-block-1 adaptation when real data comes back. First block is data collection, not candidate collection. Best when: vocabulary is uncertain, role is novel, or geography is unfamiliar.

4. **negative_space** — Broad pool, easier to define what you DON'T want; NOT operators derived from noise_archetypes. Best when: the base pool is huge but contains a large, predictable non-fit population excludable by title/keyword.

5. **company_first** — Talent concentrated in known companies; organized by company cluster AND skill. The named employer set goes on `linkedin_company_filter` — do NOT also collapse the same employers into a keyword OR-clause. Best when: the brief identifies specific employer targets and the employer signal is the key differentiator.

6. **title_first** — Distinctive job title that practitioners actually use, anchored via `linkedin_title_filter` (a keyword OR-group of title variants is NOT a substitute and must not duplicate it). Best when: there IS a standard title specific enough to be a strong filter.

Select ONE architecture and explain your reasoning. Include in your JSON output:
- "architecture": one of "sniper", "dragnet", "titration", "negative_space", "company_first", "title_first"
- "architecture_rationale": Why this architecture fits this role/market (2-3 sentences)
- "architecture_success_criteria": Array of 2-4 measurable criteria (e.g., "save rate > 5% across first block")
- "architecture_pivot_triggers": Array of 2-3 signals that would indicate this architecture is wrong (e.g., ">50% of strings return <20 results")

## Your Task
{task_intro}

Your job: design targeted {task_object} {task_source_clause}.

{retrieval_contract_guidance}

### 1. DESIGN {"layered retrieval families" if use_layered_retrieval else "the search portfolio"}
{"The kit provides terms organized by skill cluster. This" if has_kit else "This"} role requires an INTERSECTION of skills.
{layered_intersection_guidance}{shape_doctrine}You are composing a PORTFOLIO of searches that approach the target population from genuinely different angles — self-description or title, skill or tool, employer or pedigree, adjacent or transferable population, accomplishment or outcome, community or artifact. No single search finds all the best people; the portfolio's value is coverage across angles, not a uniform shape.{example_compounds_section}

Principles for every string{" and every layer term" if use_layered_retrieval else ""}:
- Maximum-Inclusion gate: before a term is REQUIRED (AND-joined), ask "would essentially all qualified people write this on their profile?" If not, OR-expand it with real variants, move it to a softer signal, or drop it. Never AND-gate a generic verb (managed, led, owned, responsible for) or a low-signal buzzword — a strong candidate who did the work may simply not have written the word, and requiring it filters them out.
- Each OR cluster is ONE concept's synonyms, never a grab-bag: do not mix a domain concept with a bespoke title in the same parenthetical — that manufactures false matches.
- Search how strong candidates describe THEMSELVES, not the requisition's language; mine the brief's own vocabulary and the words real practitioners use.
- Titles are one angle and a trap as the sole anchor: OR-expand a title's real variants, and prefer a skill, tool, or outcome as the primary anchor.
- Recover precision with proof-of-practice proxies — specific tools, frameworks, benchmarks, certifications, artifacts, or accomplishment language that only a genuine practitioner carries — not by stacking generic AND gates.
- Every string earns its place by a DISTINCT hypothesis: name the population or angle it uniquely reaches. If it duplicates another string's angle, cut it.
- NOT excludes qualified people too (NOT manager also drops someone who "managed a team of 5"): use it surgically, and you may use it deliberately to surface unanticipated titles.

{angle_shape_exemplars}
{total_count_guidance} {"Each family may emit one or more concrete booleans." if use_layered_retrieval else ""} The portfolio must span TWO {mix_label} of angle:

**Type A: {recall_label}** — recall nets that surface the general cohort by casting wide across a concept's synonyms. 500-5000 expected results.

**Type B: {precision_label}** — anchored on proof-of-practice terms (specific tool, framework, benchmark, or method names) that only genuine practitioners carry. 20-500 expected results; few results, but nearly every one is a real practitioner. {"Use the kit's Precision cluster terms" if has_kit else "Mine the JD and role description for the most specific, least ambiguous terms"} — the specific term is higher-signal than any umbrella word.

EMISSION ORDER (cheapest verdict first): order "generated_strings" by expected pages-to-verdict, not by importance. Open with PROBES — strings whose expected result pool is small enough (roughly under ~500) to confirm or kill their hypothesis in 1-2 pages: precision snipers, narrow intersections, edge-case angles. Emit the large recall nets AFTER the probes, so they run informed by what the probes teach. The obvious canonical pool runs as later cleanup — its yield is stable and it is not going anywhere. A page costs the executor minutes of wall-clock; the opening sequence should buy the most information per hour, not the most coverage per string.{sequencing_section}

{tapped_market_section}### Opening Anchors (every brief)

Every string — opener or later — must carry at least one anchor drawn from THIS brief's own vocabulary on at least one of three axes: a capability-area term (candidate-register when the brief carries it, else key term) (domain axis), an employer-cluster lane pattern (employer axis), or a caliber signal the brief itself names (pedigree vocabulary from its own text). An adjacent-population string with no anchor on any axis is a generic net — do not queue it. Adjacent pools are entered through their INTERSECTION with the brief's domain, employer, or caliber vocabulary, never through the industry analog alone.

When a lane's patterns are employer names, at least one queued string serving that lane must operationalize those employers directly — the names as keyword terms, or a deliberate `linkedin_company_filter` bound (the keyword-vs-filter guidance above governs which surface).

Guidelines for all strings:
- {"Pull skill terms from the kit's high-value clusters" if has_kit else "Pull skill terms from the JD's capability areas and technical requirements"}
- Pull domain terms from the brief's archetypes, minimum bar, and role description
{syntax_guidance}

{matching_guidance}
{abbreviation_blacklist_section}

### 2. IDENTIFY coverage gaps
What candidate populations {"does the kit vocabulary NOT reach" if has_kit else "might your generated strings miss"}? Examples:
- People who describe their work differently {"than the kit's terminology" if has_kit else "than the JD's terminology"}
- Adjacent skill sets {"not represented in any kit block" if has_kit else "not covered by your generated strings"}
- Domain-specific terms the kit misses
- Title patterns or employer patterns that could surface candidates

For each gap, provide a ready-to-execute Boolean string if possible.

### 3. PREDICT noise collisions
Based on the vocabulary and this geography/role, predict which terms will produce noise and what the collision patterns will be.

IMPORTANT: Each string you generate is a NET — it catches whoever matches the Boolean, regardless of archetype.
A string built from post-training vocabulary might surface a STEM reasoning engineer or an RL environment builder.
That's good. The evaluator downstream judges every candidate against ALL archetypes, not just the one the string was "designed for."
Your strings are search tools, not archetype filters. Label them descriptively but do NOT treat them as archetype-scoped.

Return JSON with this structure:
- "architecture": Your selected architecture (string — one of the six listed above)
- "architecture_rationale": Why this architecture fits (string)
- "architecture_success_criteria": Array of 2-4 measurable success criteria (array of strings)
- "architecture_pivot_triggers": Array of 2-3 pivot trigger signals (array of strings)
- "strategy_rationale": Overall strategy explanation (string)
- "retrieval_families": Array of structured family objects in priority order. Each object:
  - "family_id": Stable id for the family (string)
  - "label": Human label (string)
  - "objective": Why this family exists (string)
  - "priority": Priority score (int)
  - "enabled": Whether to execute this family (bool)
  - "variants_to_emit": How many rendered variants to emit (int)
  - "entry_signals": Array of objects with "item_id", "label", "terms", optional "priority", "structured_surface": optional, "linkedin_title_filter" or "linkedin_company_filter"
  - "capability_proxies": Array of objects with "item_id", "label", "terms", optional "priority" (no structured_surface — capability proxies are always keyword signals)
  - "reality_filters": Array of objects with "item_id", "label", "terms", optional "priority" (no structured_surface — reality filters are always keyword signals)
  - "context_constraints": Array of objects with "item_id", "label", "terms", optional "priority", "structured_surface": optional, "linkedin_title_filter" or "linkedin_company_filter"
  - "anti_noise": Array of objects with "item_id", "label", "terms", optional "priority"
  - "target_employers": Optional employer targets (array of strings)
  - "target_markets": Optional market/lane labels (array of strings)
  - "hypothesis_ids": Optional applied edge-case hypotheses (array of strings)
- "generated_strings": Array of search strings to execute, in priority order. Each object:
  - "boolean": The full Boolean string (string)
  - "rationale": Why this string is likely to surface strong candidates for the role (string)
  - "vocabulary_sources": {"Which kit blocks/clusters the terms come from" if has_kit else "Which JD sections or capability areas the terms derive from"} (string) — this is for traceability only, NOT for scoping evaluation
  - "family_key": Short stable label for this search family (string) — use the same label for close variants of the same idea
  - "novelty_bucket": "edge_case" or "canonical" (string)
  - "domain_lane": Primary lane this string targets (string). Use one of this brief's declared lanes when the brief lists them; otherwise DERIVE 3-6 lane labels specific to THIS brief's market — employer clusters, segments, or domains a search could target — and reuse them consistently across strings. "general" is reserved for a string that genuinely fits no derived lane; never default every string to "general" (an all-"general" run cannot learn which market segments work)
  - "retrieval_recipe": Optional structured recipe describing the layer ids and applied hypotheses used to render this string
  - "structured_filters": Optional object bounding this string's pool with LinkedIn sidebar facets that run ALONGSIDE the keyword Boolean. Keys: "companies" (list of exact employer names), "titles" (list of exact job titles). Omit for a pure keyword string. Follow the per-string structured filters rules above; never include locations here, and never duplicate a value across the facet and the keyword Boolean.
- "coverage_gaps": Array of gaps identified. Each object:
  - "gap": Description of the missing coverage (string)
  - "suggested_boolean": Optional Boolean string to fill the gap, or null (string|null)
  - "rationale": Why this population matters for this role (string)
  - "family_key": Optional stable label for the gap string family (string)
  - "novelty_bucket": Optional "edge_case" or "canonical" (string)
  - "domain_lane": Optional primary lane label (string)
- "noise_predictions": Array of objects with "term" (string), "expected_collision" (string), "mitigation" (string)

Return valid JSON only."""


_DEPRESCRIBED_HEAD = """You are a world-class, expert-level talent sourcer and talent researcher.
You have practitioner-level command of LinkedIn Recruiter: how quoted
phrases, AND / OR / NOT, and parentheses compose; how the Company and Title
facets interact with keyword clauses; and how differently strong candidates
describe the same work in their own profiles. You know that any real talent
pool can be entered from several distinct angles, and that a good sourcing
pass is a portfolio of searches trading precision and recall against each
other, not a single best string.

Your task at this stage is strategy formation. From the role brief below,
design a portfolio of 15-30 LinkedIn Recruiter Boolean search strings — each
optionally paired with structured Company/Title facet selections — returned
in the JSON contract at the end of this prompt. Strings must be legal
LinkedIn Boolean: quoted phrases for multi-word terms, uppercase AND / OR /
NOT, parentheses for grouping. No pre-built search kit is available; the
brief is your source material.

How this pipeline consumes your output. Your strings execute exactly as you
emit them, in the order you emit them, and the executor pays minutes of
wall-clock per page of results — so order buys information: open with probes
(expected pools roughly under ~500 results that confirm or kill a hypothesis
in a page or two), emit the large recall nets after, informed by what the
probes teach, and leave the stable canonical pool for later cleanup. Every
string is a net, not an archetype filter — the downstream evaluator judges
each returned candidate against ALL of the brief's archetypes, so label
strings descriptively but never scope them. Downstream code blocks malformed
syntax, scrubs blacklisted employers, and records telemetry warnings for
low-signal AND gates, requisition-register vocabulary, and facet/keyword
duplication — but warnings repair nothing; the craft below is still yours to
get right.

Four craft principles. They earn their place because they are the failure
modes this pipeline has actually produced, and nothing downstream fixes them:

1. Search the language candidates use about themselves, not the language
   requisitions use about roles. Requisition register is common on job
   descriptions and rare in first-person profile text; ANDing it into a
   search removes real candidates without adding signal. Prefer concrete
   nouns, tools, methods, credentials, employer names, and titles as
   candidates actually write them. The brief's discriminating vocabulary is
   the tested channel — draw on it and extend it with real practitioner
   language.

2. Every clause must add information beyond its siblings. Before adding a
   clause, ask what it excludes that the rest of the string does not already
   exclude. A title cluster plus a company facet on a named employer set
   already implies that set's domain vocabulary — ANDing that vocabulary on
   top is a null clause that only shrinks the pool. Generic seniority verbs
   (led, owned, managed, scaled) are implied by any senior title and add
   nothing behind an AND. The test for any required term: would essentially
   all qualified people write it on their profile? If not, OR-expand it with
   its real variants, move it to a softer signal, or drop it.

3. Enter the pool from different starting surfaces. Some strings anchor on
   title patterns, some on employer sets, some on domain vocabulary, tools,
   or credentials candidates actually list, some on adjacent-pool bridges
   for people whose title does not match but whose work does. An
   adjacent-pool string must intersect the bridge population with the
   brief's own domain, employer, or caliber vocabulary — the industry analog
   alone is a generic net; do not queue it. When a lane's patterns are
   employer names, at least one string serving that lane must operationalize
   those employers directly — as keywords or a deliberate company-facet
   bound. Each OR group holds one concept's real variants — singular/plural,
   hyphenation, abbreviation alongside its expansion, ampersand form with its and-form twin, common misspellings,
   non-English variants where the geography warrants — and treat matching as
   literal: no stemming assumed, never case-only variants, no fabricated
   variants for proper-noun tools. Every string earns its place with a
   distinct hypothesis about the population it uniquely reaches; if it
   duplicates another string's angle, cut it.

4. Precision comes from distinctive terms, not stacked gates. A bare
   one-word term whose dominant LinkedIn meaning is broader than your target
   matches everyone and costs pages — qualify it into a compound or cut it.
   NOT is surgical: every exclusion also removes qualified people (NOT
   "manager" drops someone who "managed a team of 5"); exclude only
   predictable noise populations — though you may use NOT deliberately as
   discovery, subtracting the titles you know to surface the ones you would
   never enumerate.

## Structured filters — the executable levers

Each generated string runs as a keyword Boolean by default. You may attach
"structured_filters": {"companies": [...], "titles": [...]} to a string to
bound its pool with a LinkedIn sidebar facet that runs alongside the keyword
Boolean; a facet that cannot apply falls back to keyword-only.

- A company facet matches actual employees via LinkedIn's employer index —
  the strongest lever you have. Use it when a named employer set IS the pool
  the string targets; it replaces listing those employers as keywords.
- A title facet is an exact-match bound. Use it only on a canonical or
  cleanup pass bounding to the brief's exact target title — never on a
  doorway string where a near-analog title is fishing for adjacent people.
- One surface per value, always: a company or title on the facet must not
  also appear in that string's keyword Boolean, and vice versa.
- An edge-case string targeting a non-canonical population keeps its
  doorways as keywords — a facet would bound it back to exactly the
  canonical pool it is trying to skip.
- Never emit a location facet; geography is applied once per run from the
  brief.
- The same one-surface rule governs the optional retrieval_families
  metadata: a literal title/employer facet belongs on entry_signals or
  context_constraints, never on capability_proxies or reality_filters.

"""

_DEPRESCRIBED_LEGEND = """## Architecture label (telemetry)

The output includes an architecture classification. It is recorded for pivot
review and mid-run adaptation, and it tunes run behavior — a titration run
holds a larger pivot budget — so choose the label that honestly names your
approach:

- sniper — distinctive title, large market; precision-led.
- dragnet — fragmented vocabulary, no standard titles; recall-led.
- titration — unknown market; open with broad recon, hold budget for
  post-block-1 adaptation.
- negative_space — huge base pool, predictable non-fit population; NOT
  operators derived from the noise archetypes.
- company_first — talent concentrated in named employers; the employer set
  rides linkedin_company_filter.
- title_first — a standard title strong enough to bound the pool via
  linkedin_title_filter.

"""

_DEPRESCRIBED_TASK_AND_CONTRACT = """## Your task

From the brief above: design the search portfolio under the craft principles;
classify your architecture with rationale, measurable success criteria, and
pivot triggers; identify coverage gaps — populations your strings might miss,
each with a ready-to-execute Boolean where possible; and predict noise
collisions with mitigations.

Return JSON with this structure:
- "architecture": Your selected architecture (string — one of the six listed above)
- "architecture_rationale": Why this architecture fits (string)
- "architecture_success_criteria": Array of 2-4 measurable success criteria (array of strings)
- "architecture_pivot_triggers": Array of 2-3 pivot trigger signals (array of strings)
- "strategy_rationale": Overall strategy explanation (string)
- "retrieval_families": Array of structured family objects in priority order. Each object:
  - "family_id": Stable id for the family (string)
  - "label": Human label (string)
  - "objective": Why this family exists (string)
  - "priority": Priority score (int)
  - "enabled": Whether to execute this family (bool)
  - "variants_to_emit": How many rendered variants to emit (int)
  - "entry_signals": Array of objects with "item_id", "label", "terms", optional "priority", "structured_surface": optional, "linkedin_title_filter" or "linkedin_company_filter"
  - "capability_proxies": Array of objects with "item_id", "label", "terms", optional "priority" (no structured_surface — capability proxies are always keyword signals)
  - "reality_filters": Array of objects with "item_id", "label", "terms", optional "priority" (no structured_surface — reality filters are always keyword signals)
  - "context_constraints": Array of objects with "item_id", "label", "terms", optional "priority", "structured_surface": optional, "linkedin_title_filter" or "linkedin_company_filter"
  - "anti_noise": Array of objects with "item_id", "label", "terms", optional "priority"
  - "target_employers": Optional employer targets (array of strings)
  - "target_markets": Optional market/lane labels (array of strings)
  - "hypothesis_ids": Optional applied edge-case hypotheses (array of strings)
- "generated_strings": Array of search strings to execute, in priority order. Each object:
  - "boolean": The full Boolean string (string)
  - "rationale": Why this string is likely to surface strong candidates for the role (string)
  - "vocabulary_sources": Which JD sections or capability areas the terms derive from (string) — this is for traceability only, NOT for scoping evaluation
  - "family_key": Short stable label for this search family (string) — use the same label for close variants of the same idea
  - "novelty_bucket": "edge_case" or "canonical" (string)
  - "domain_lane": Primary lane this string targets (string). Use one of this brief's declared lanes when the brief lists them; otherwise DERIVE 3-6 lane labels specific to THIS brief's market — employer clusters, segments, or domains a search could target — and reuse them consistently across strings. "general" is reserved for a string that genuinely fits no derived lane; never default every string to "general" (an all-"general" run cannot learn which market segments work)
  - "retrieval_recipe": Optional structured recipe describing the layer ids and applied hypotheses used to render this string
  - "structured_filters": Optional object bounding this string's pool with LinkedIn sidebar facets that run ALONGSIDE the keyword Boolean. Keys: "companies" (list of exact employer names), "titles" (list of exact job titles). Omit for a pure keyword string. Follow the per-string structured filters rules above; never include locations here, and never duplicate a value across the facet and the keyword Boolean.
- "coverage_gaps": Array of gaps identified. Each object:
  - "gap": Description of the missing coverage (string)
  - "suggested_boolean": Optional Boolean string to fill the gap, or null (string|null)
  - "rationale": Why this population matters for this role (string)
  - "family_key": Optional stable label for the gap string family (string)
  - "novelty_bucket": Optional "edge_case" or "canonical" (string)
  - "domain_lane": Optional primary lane label (string)
- "noise_predictions": Array of objects with "term" (string), "expected_collision" (string), "mitigation" (string)

Return valid JSON only."""


def _build_strategy_system_deprescribed(brief: Brief) -> str:
    """De-prescribed formation prompt for the no-kit, non-layered path.

    plans/formation-prompt-de-prescribed.md (approved 2026-07-05): expert
    role framing + pipeline facts + four craft principles + filter/telemetry
    semantics around the byte-identical brief-data sections and JSON output
    contract. Brief-conditional data (compounds, sequencing, abbreviation
    collisions, blacklist) renders inside the <brief> block; the brief-gated
    tapped-market playbook renders unchanged between the filter and legend
    blocks. The contract text is a verbatim copy of the legacy non-layered
    contract — locked byte-identical by test."""
    noise_section = ""
    if brief.noise_archetypes:
        noise_section = f"\n## Noise Archetypes\n{json.dumps(brief.noise_archetypes, indent=2)}"

    known_noise_section = ""
    if brief.known_noise_patterns:
        known_noise_section = f"\n## Known Noise Patterns\n{json.dumps(brief.known_noise_patterns, indent=2)}"

    key_terms_section = ""
    candidate_terms_by_area = getattr(brief, "candidate_register_terms_by_area", {}) or {}
    if candidate_terms_by_area:
        kt_lines = ["\n## Discriminating Vocabulary by Capability Area"]
        for area, terms in candidate_terms_by_area.items():
            kt_lines.append(f"- {area}: {', '.join(terms)}")
        kt_lines.append("\nUse these terms as candidate self-description anchors for Type B precision strings. They are the profile vocabulary qualified candidates plausibly write for each area.")
        key_terms_section = "\n".join(kt_lines)
    elif brief.key_terms_by_area:
        kt_lines = ["\n## Discriminating Vocabulary by Capability Area"]
        for area, terms in brief.key_terms_by_area.items():
            kt_lines.append(f"- {area}: {', '.join(terms)}")
        kt_lines.append("\nUse these terms as anchors for Type B precision strings. They are the specific technical vocabulary that distinguishes qualified candidates in each area.")
        key_terms_section = "\n".join(kt_lines)

    declared_lanes_section = _render_declared_lanes_section(brief)

    market_density_section = (
        f"\n## Market Density\n{brief.market_density}\n" if brief.market_density else ""
    )

    abbreviation_block = _render_abbreviation_collisions_block(brief)
    abbreviation_text = (
        f"""### Abbreviation Collision Filter

An abbreviation MUST NOT appear standalone if it has a more common non-domain meaning on LinkedIn.

{abbreviation_block}

Rule: an abbreviation that fails alone IS acceptable when paired with its full expansion in the same OR group."""
        if abbreviation_block
        else ""
    )

    blacklist_block = _render_term_blacklist_block(brief)
    blacklist_text = (
        f"""### Blacklist — NEVER Include

{blacklist_block}"""
        if blacklist_block
        else ""
    )

    abbreviation_blacklist_block = "\n\n".join(
        section for section in (abbreviation_text, blacklist_text) if section
    )
    abbreviation_blacklist_section = (
        f"\n{abbreviation_blacklist_block}\n" if abbreviation_blacklist_block else ""
    )

    example_compounds_block = _render_example_compounds_block(brief)
    example_compounds_section = (
        "\n\nBrief-supplied compound hints (vocabulary evidence from preflight "
        "— two of many possible angles, NOT a shape template; author your own "
        f"shapes per angle):\n{example_compounds_block}"
        if example_compounds_block
        else ""
    )

    sequencing_text = str(getattr(brief, "sequencing_heuristics", "") or "").strip()
    sequencing_section = (
        f"\n\nSEQUENCING:\n{sequencing_text}" if sequencing_text else ""
    )

    # Brief-gated tapped-market playbook: unchanged text, same gate as the
    # deterministic opening sort; its own de-prescription is a later slice.
    tapped_market_section = (
        """### Tapped-Market / Edge-Case Opening (this brief declares the obvious pool is exhausted)

Your OPENING SEQUENCE must prioritize non-obvious adjacent populations rather than the canonical role vocabulary.

This is a SEQUENCING and pool-selection rule about which populations to open with. It is NOT a rule about which surface a token sits on. Whenever a literal title or named employer DOES appear in a string — opening or later — the keyword-vs-filter guidance above still governs: a deliberate pool bound goes on `linkedin_title_filter` / `linkedin_company_filter`, regardless of which lane the string serves.

Use this mental loop:
1. First ask: what would a generally solid technical sourcer search if they were doing a competent but standard pass for this role?
2. Then ask: which same-caliber candidates would that standard pass systematically miss because they use different titles, different product language, or sit in adjacent org structures?
3. Generate your opening strings primarily for THOSE missed populations.

For the FIRST 8 strings:
- At least 5 must target edge-case or transfer populations.
- Prefer intersections of an adjacent-capability population with the brief's own edge-case vocabulary — adjacent org structures, internal-tooling builders, delivery/implementation roles, practitioners who describe the work in product or problem language rather than the canonical role vocabulary, consultancy ICs with build evidence, or vertical specialists whose day job implies the capability. Draw the concrete terms from the brief's edge-case patterns and capability areas, never from a stock list.
- Do NOT open with strings whose primary POOL is the canonical population — i.e. strings that target the exact target-title pool, the canonical-employer pool, or a fashionable framework-name pool as their main reason to exist. This is about the pool, not the surface.
- Do NOT front-load broad core strings (the canonical role-vocabulary AND production-proof shape) unless they are crossed with a non-obvious adjacent-population qualifier.
- An edge-case opening string targets a non-canonical pool BY DESIGN; its entry_signals are recall doorways. So it should NOT carry a `linkedin_title_filter` set to the literal target title or a `linkedin_company_filter` set to the canonical employer pool — that facet would bound the pool back to exactly the canonical population the opening is trying to skip. This is the one place the keyword-vs-filter default is overridden, and it is overridden because of the POOL the string targets, not the token class.

The canonical pool still matters; it belongs in later cleanup passes once the edge-case populations have been tested. When a later string DOES intentionally target the canonical pool, the keyword-vs-filter guidance applies normally — a deliberate title/employer bound goes on its filter.

NOVELTY ACCOUNTING (MANDATORY for tapped markets):
- In a tapped market, productivity alone is not enough. Saves from exact-title pools, canonical-employer pools, and framework-first strings are useful confirmation but LOW-NOVELTY signal.
- Do NOT interpret a high save rate from those canonical pools as proof the opening sequence is correct.
- When you set architecture success criteria and pivot triggers, include at least one metric about novelty or pool mix, not just save rate and result count.

"""
        if _brief_targets_edge_case_opening(brief)
        else ""
    )

    brief_block = f"""<brief>
Role: {brief.role_title}
{brief.role_description}

## Minimum Bar
{brief.minimum_bar}

## Archetypes
{json.dumps(brief.archetypes, indent=2)}
{noise_section}
{known_noise_section}
{key_terms_section}
{declared_lanes_section}
## Permanent Filters
{json.dumps(brief.permanent_filters, indent=2)}
{market_density_section}{abbreviation_blacklist_section}{example_compounds_section}{sequencing_section}
</brief>

"""

    return (
        _DEPRESCRIBED_HEAD
        + tapped_market_section
        + _DEPRESCRIBED_LEGEND
        + brief_block
        + _DEPRESCRIBED_TASK_AND_CONTRACT
    )


def _build_strategy_user(
    brief: Brief,
    kit_strings: list[KitString],
    prior_run_data: dict | None = None,
    *,
    use_layered_retrieval: bool = False,
    lane_feedback: list[dict] | None = None,
) -> str:
    retrieval_design = _explicit_design_from_brief(brief) if use_layered_retrieval else RetrievalDesign()
    semantic_hint_mode = use_layered_retrieval or is_strict_seniority_brief(brief)
    # Group kit strings by block for clearer vocabulary presentation
    blocks: dict[str, list[KitString]] = {}
    for ks in kit_strings:
        blocks.setdefault(ks.block, []).append(ks)

    vocab_text = ""
    for block_name, strings in blocks.items():
        vocab_text += f"\n### {block_name}\n"
        for ks in strings:
            vocab_text += f"  [{ks.subblock} / {ks.string_type}]: {ks.boolean}\n"

    if kit_strings:
        prompt = f"""## Boolean Search Vocabulary ({len(kit_strings)} terms across {len(blocks)} competency blocks)
These are your raw building blocks. Combine terms from multiple blocks to create targeted compound searches.
{vocab_text}
"""
    else:
        prompt = """## No Kit Vocabulary Available
No pre-built search kit was provided. Generate compound Boolean strings directly from
the role description, archetypes, and JD context below. Apply all LinkedIn Boolean rules.
"""

    # Include JD text if available (supplementary context for string generation)
    if brief.jd_text:
        prompt += f"\n## Job Description (source material for search vocabulary)\n{brief.jd_text}\n"

    if brief.intake_notes:
        prompt += f"\n## Intake Notes\n{brief.intake_notes}\n"

    if use_layered_retrieval and not retrieval_design.is_empty():
        prompt += (
            "\n## Layered Retrieval Design\n"
            f"{json.dumps(summarize_retrieval_design(retrieval_design), indent=2)}\n"
            "\nPrefer using this structured retrieval design as the primary planning interface. "
            "If you expand or revise it, keep the same layered model: entry_signals, "
            "capability_proxies, reality_filters, optional context_constraints, optional anti_noise, "
            "and edge-case hypothesis overlays.\n"
        )

    if prior_run_data:
        raw_prior = dict(prior_run_data)
        raw_prior.pop("search_memory_summary", None)
        prompt += f"\n## Prior Run Data\n{json.dumps(raw_prior, indent=2)}\n"

        if prior_run_data.get("noise_discoveries"):
            noise_text = "\n## Noise Patterns Discovered in Prior Sessions\n"
            for nd in prior_run_data["noise_discoveries"]:
                noise_text += f"- {nd['term']}: [{nd['status']}] {nd.get('note', '')}\n"
            noise_text += "\nAvoid generating strings that primarily target confirmed_noise patterns.\n"
            prompt += noise_text

        if prior_run_data.get("search_memory_summary"):
            # RC2 (2026-07-04 SPL RCA): the old unconditional "treat reused
            # families as later cleanup passes" line taught formation to
            # flee every pool that had ever produced — the mis-attributed
            # saves read as coverage. Guidance is now yield-aware.
            prompt += (
                "\n## Search Family Memory\n"
                f"{format_search_memory_summary(prior_run_data['search_memory_summary'])}\n"
                "\nFamilies marked exhausted are later cleanup passes, not opening bets. "
                "Families with saves are PROVEN VEINS: their exemplars and winning boolean "
                "show where the pool actually lives. A vein earns a CHEAP PROBE as its opener "
                "— a small, fast-to-exhaust variant that reconfirms conversion in a page or two "
                "— never the vein's biggest recall net; expand novelty from what the probe "
                "confirms. Prior saves are evidence of pool location, never evidence the pool "
                "is finished. (If this brief explicitly declares a tapped market, the "
                "tapped-market opening rules take precedence over vein-first opening.)\n"
            )

    if brief.search_priorities:
        prompt += f"\n## User Hints\nSearch priorities: {', '.join(brief.search_priorities)}\n"
        if semantic_hint_mode:
            prompt += (
                "Treat these priorities as semantic guidance, not as a checklist of phrases to restate. "
                "Infer the target populations and generate your own discriminative search vocabulary.\n"
            )
            if is_strict_seniority_brief(brief):
                prompt += (
                    "For this strict-seniority brief, prefer technical-authority concepts, ED-scope signals, "
                    "architecture-deep language, and builder proof over broad management-title coverage.\n"
                )

    if brief.additional_search_terms:
        prompt += f"\n## Additional Search Terms\nThese terms should be used for search string generation but are NOT evaluation criteria:\n{', '.join(brief.additional_search_terms)}\n"
        if semantic_hint_mode:
            prompt += (
                "These terms are anchors and hints, not a mandate to repeat them verbatim. "
                "Use them to infer adjacent practitioner language, hidden title variants, workflow language, "
                "and more discriminative phrasing.\n"
            )
            if is_strict_seniority_brief(brief):
                prompt += (
                    "Do not turn these hints into broad OR groups of generic titles, company inventories, or loose seniority ladders.\n"
                )

    if brief.instructions:
        prompt += f"\n## Sourcing Instructions\n" + "\n".join(f"- {i}" for i in brief.instructions) + "\n"

    if lane_feedback:
        prompt += "\n## Lane Feedback Diffs\n"
        prompt += "The following executable diffs come from market-intelligence analysis of prior runs.\n"
        prompt += "Apply these according to their action:\n"
        prompt += "- 'add' diffs: incorporate as new lanes, hypotheses, or constraints.\n"
        prompt += "- 'retire' diffs: omit or downrank the targeted lane/hypothesis.\n"
        prompt += "- 'update' diffs: modify the targeted element.\n"
        prompt += "- 'validation_question' types: address in your strategy rationale.\n"
        prompt += "Record which diff_ids you consumed in the 'consumed_feedback_ids' field of your output.\n"
        prompt += "If a diff is unsupported or conflicts with the brief, note it in strategy_rationale but do not silently drop it.\n\n"
        prompt += json.dumps(lane_feedback, indent=2) + "\n"

    prompt += (
        "\nSynthesize layered retrieval families and rendered search strings"
        if use_layered_retrieval
        else "\nSynthesize compound Boolean search strings"
    ) + (" from this vocabulary." if kit_strings else " from the JD and role context.")
    return prompt


def _try_salvage_strategy(original_error: Exception) -> ExecutionPlan | None:
    """Try to salvage a partial strategy from a truncated JSON response."""
    err_msg = str(original_error)
    # Look for partial JSON in the error message
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = err_msg.find(start_char)
        if start == -1:
            continue
        # Walk backwards from the end, trying progressively shorter substrings
        text = err_msg[start:]
        for i in range(len(text) - 1, 0, -1):
            if text[i] == end_char:
                try:
                    data = json.loads(text[: i + 1])
                    if isinstance(data, dict):
                        return ExecutionPlan.from_dict(data)
                except json.JSONDecodeError:
                    continue
    return None


def _default_strategy(kit_strings: list[KitString]) -> ExecutionPlan:
    """Return an empty plan when strategy formation fails.

    Kit strings are vocabulary only — they are never queued directly.
    Without Opus-generated compounds, there are no strings to execute.
    """
    return ExecutionPlan(
        strategy_rationale="Strategy formation failed — no compound strings generated. Re-run to retry.",
        noise_predictions=[],
        generated_strings=[],
        coverage_gaps=[],
    )


# ---------------------------------------------------------------------------
# Adaptation after block completion
# ---------------------------------------------------------------------------

_NEXT_CHECKPOINT_AFTER_MIN = 2
_NEXT_CHECKPOINT_AFTER_MAX = 8

# Module-level so the characterization tests can assert it stays
# vertical-agnostic (Codex review, Wave 1: this block carried "BFSI /
# market-institution" into every brief's opening checkpoint).
_OPENING_CHECKPOINT_GUIDANCE = """

This is the OPENING CHECKPOINT. Treat it differently from a normal later-stage block adaptation:
- prioritize exploitation of productive institution/lane patterns over novelty-chasing
- skip dead hidden-population hypotheses sooner
- use new_strings to pull the run's proven productive lanes forward (the brief's declared lanes when they exist)
- do NOT spend this checkpoint rediscovering adjacent edge-case populations that already failed
- only recommend a pivot if the opening block is uniformly dead and coherently wrong
"""


def _coerce_next_checkpoint_after(raw: object) -> int | None:
    """Coerce a model-supplied next_checkpoint_after value to a plain int.

    ``bool`` is an int subclass in Python and is explicitly rejected here —
    a stray ``true``/``false`` must not silently become 1/0. A non-integral
    float or any other type is treated as absent rather than guessed at.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    return None


def _clamp_next_checkpoint_after(adaptation: AdaptationResponse, requested: object) -> None:
    """P11.2: clamp the model's requested adaptation cadence to [2, 8].

    ``requested`` is the raw value parsed off the model's JSON response
    (may be ``None``, out of range, or the wrong type). This overwrites
    ``adaptation.next_checkpoint_after`` with the clamped, orchestrator-
    facing value (or ``None`` when absent/invalid) and preserves the raw
    request on ``next_checkpoint_after_requested`` for logging. The model
    REQUESTS a cadence; this is the deterministic code that clamps/applies
    it — the orchestrator never receives an unbounded value.
    """
    coerced = _coerce_next_checkpoint_after(requested)
    applied = None
    if coerced is not None:
        applied = max(_NEXT_CHECKPOINT_AFTER_MIN, min(_NEXT_CHECKPOINT_AFTER_MAX, coerced))
        print(
            f"    [adapt] next_checkpoint_after: requested={requested!r} applied={applied}"
        )
    adaptation.next_checkpoint_after = applied
    setattr(adaptation, "next_checkpoint_after_requested", requested)


def adapt_after_block(
    brief: Brief,
    block_report: BlockReport,
    remaining_strings: list[SearchString],
    kit_vocabulary: list[KitString] | None = None,
    execution_plan: ExecutionPlan | None = None,
    pivot_count: int = 0,
    search_memory_summary: dict | None = None,
    checkpoint_mode: str = "normal_block_checkpoint",
    market_intel_advisory_context: str = "",
    market_signal_prior: MarketSignalPrior | dict | None = None,
) -> AdaptationResponse:
    """Ask Opus to adapt after a block completes — generate new strings from vocabulary.

    Args:
        brief: Normalized Brief dataclass.
        block_report: Summary of the completed block's performance.
        remaining_strings: SearchStrings not yet executed (generated compounds still queued).
        kit_vocabulary: Full kit vocabulary available for synthesizing new strings.
        execution_plan: Current execution plan (for architecture context).
        pivot_count: Number of architecture pivots already used this run.

    Returns:
        AdaptationResponse with new strings, skips, reorders, noise updates, optional pivot.
    """
    signal_state = SearchSignalState.from_block_report(block_report)
    typed_signal_state = render_signal_state_for_prompt(signal_state)
    market_prior = coerce_market_signal_prior(
        market_signal_prior
        if market_signal_prior is not None
        else market_intel_advisory_context
    )
    typed_market_prior = render_market_signal_prior_for_prompt(market_prior)

    # Build vocabulary section if available
    vocab_section = ""
    if kit_vocabulary:
        blocks: dict[str, list[KitString]] = {}
        for ks in kit_vocabulary:
            blocks.setdefault(ks.block, []).append(ks)
        vocab_lines = []
        for block_name, strings in blocks.items():
            vocab_lines.append(f"### {block_name}")
            for ks in strings:
                vocab_lines.append(f"  [{ks.subblock} / {ks.string_type}]: {ks.boolean}")
        vocab_section = "\n".join(vocab_lines)

    # Build architecture review section
    arch_review = ""
    if execution_plan and execution_plan.architecture:
        max_pivots = 2 if execution_plan.original_architecture == "titration" else 1
        pivots_remaining = max(0, max_pivots - pivot_count)
        criteria_text = "\n".join(f"  - {c}" for c in execution_plan.architecture_success_criteria) or "  (none set)"
        triggers_text = "\n".join(f"  - {t}" for t in execution_plan.architecture_pivot_triggers) or "  (none set)"
        pivot_note = f"\nNOTE: No pivots remaining. You cannot recommend an architecture change." if pivots_remaining == 0 else ""
        arch_review = f"""

## Architecture Review

Current architecture: {execution_plan.architecture}
Rationale: {execution_plan.architecture_rationale}

Success criteria:
{criteria_text}

Pivot triggers:
{triggers_text}

Pivots remaining this run: {pivots_remaining}

Based on the block report, evaluate whether the current architecture is meeting its success criteria.
If any pivot triggers are firing, you MAY recommend switching architectures by including:
- "pivot_to_architecture": the new architecture name (one of: sniper, dragnet, titration, negative_space, company_first, title_first)
- "pivot_rationale": detailed explanation of why the current approach failed and why the new one will work

A pivot clears remaining queued strings and replaces them with your new_strings (generated under the new architecture). Only recommend when evidence is clear.{pivot_note}"""

    opening_checkpoint_guidance = ""
    if checkpoint_mode == "opening_checkpoint":
        opening_checkpoint_guidance = _OPENING_CHECKPOINT_GUIDANCE

    explicit_design = _explicit_design_from_brief(brief)
    use_layered_retrieval = explicit_design.is_explicit()
    adaptation_generation_guidance = (
        "When generating new retrieval families, use the layered retrieval model:\n"
        "- entry_signals\n"
        "- capability_proxies\n"
        "- reality_filters\n"
        "- optional context_constraints\n"
        "- optional anti_noise\n"
        "- optional edge-case hypothesis overlays\n\n"
        "Rendered booleans follow the family contract:\n"
        "(entry_signals) AND (capability_proxies) AND (reality_filters) "
        "[AND context_constraints] [NOT anti_noise]"
        if use_layered_retrieval
        else "Continue composing from different angles; let each string's shape follow its angle rather than a fixed template. "
        "You may optionally include new_retrieval_families as traceability metadata, "
        "but new_strings remain primary."
    )
    adaptation_matching_guidance = render_adaptation_matching_guidance()
    declared_lanes_section = _render_declared_lanes_section(brief)
    # Same gate as formation (2026-07-04 SPL RCA, root cause 1): the
    # tapped-market adaptation posture binds only when the brief declares a
    # worked market.
    adaptation_tapped_guidance = (
        """If the brief says the obvious pool is tapped, prefer generating new strings that expand productive edge-case populations before emitting direct framework-name cleanup strings or canonical-pool cleanup strings. This is a pool-selection preference, not a surface rule: when a new string DOES target the canonical pool, put a deliberate title bound on `linkedin_title_filter` and named employer bounds on `linkedin_company_filter` rather than collapsing them into keyword OR-clauses.

"""
        if _brief_targets_edge_case_opening(brief)
        else ""
    )
    adaptation_tapped_block_quality = (
        """In tapped markets, evaluate BLOCK QUALITY on two axes:
1. productivity: saves, facial pass rate, result quality
2. novelty: whether the saves came from adjacent populations versus exact-title pools, framework-first strings, or canonical-employer pools

If a string is productive but mostly confirms the obvious pool, treat it as cleanup signal, not as the template for what should come next. In that case:
- prefer reordering similar queued strings later
- prefer generating adjacent expansions instead of "more of the same"
- recommend a pivot if the opening sequence is succeeding only in low-novelty pools

"""
        if _brief_targets_edge_case_opening(brief)
        else ""
    )

    system = f"""You are a sourcing strategist adapting a search plan mid-run.

Role: {brief.role_title}
{brief.role_description}

You've just received a report on a batch of completed Boolean searches. Based on the results:
1. Generate NEW {"layered retrieval families and/or rendered Boolean strings" if use_layered_retrieval else "rendered Boolean strings"} that target signal patterns you observed — use the kit vocabulary below as building blocks
2. Identify remaining queued strings to skip (redundant, similar to zero-save strings)
3. Suggest reordering of remaining queued strings based on observed signal
4. Update noise pattern knowledge
5. Decide whether this block genuinely needs a change at all, and how soon you want to check back in

If the evidence doesn't support any change, say so explicitly with "no_change": true rather than forcing a marginal edit. If you'd rather see more (or fewer) strings run before your next checkpoint, request it with "next_checkpoint_after".

IMPORTANT: Each Boolean string is a NET that catches candidates for ANY archetype, not just one.
A string built from post-training terms might surface a STEM reasoning engineer. That's expected and good.
Evaluate string productivity by total saves across ALL archetypes, not just the archetype the string was "designed for."

IMPORTANT: a queued string annotated "POOL BOUND BY STRUCTURED FILTER" is a filter-led
lane. Its specificity lives in the title/company facet, and its keyword clause is broad
BY DESIGN — the facet bounds who is in the pool, the keywords select capability within
it. Judge such a string on the facet plus the keywords TOGETHER. Do not skip one as
unanchored or as a grab-bag because its Boolean alone looks broad; that is the shape
working as intended, and these lanes reach populations the keyword-only lanes miss.

{adaptation_generation_guidance}
{declared_lanes_section}
When generating new strings or families, combine terms from the kit vocabulary with domain qualifiers. The most valuable updates will target the specific intersection of skills and domain this role requires. Every new string must carry at least one anchor from the brief's own vocabulary — a capability-area key term, an employer-cluster lane pattern, or a caliber signal the brief names; an adjacent-population string with no anchor on any axis is a generic net — do not emit it.

Compose new strings as a portfolio of angles, not one shape: recall nets (a concept's synonyms cast wide) AND precision "sniper" strings anchored on proof-of-practice terms — specific tool, framework, or benchmark names only real practitioners carry. Apply the Maximum-Inclusion test to every required AND term (would essentially all qualified people write it?), and keep each OR cluster to one concept's synonyms, never a grab-bag.

{adaptation_tapped_guidance}When adapting, continue using the same loop: identify what a standard sourcer would search next, then push one layer outward toward adjacent but same-caliber populations that the standard next step would still miss.

{opening_checkpoint_guidance}

{adaptation_tapped_block_quality}Per-string "Bias context" lines report save density and the opens-for-full-eval rate against the brief's expected band. Read them against the string's history: a dense pocket after a deliberate refinement is a vein worth exploiting, while a uniformly high rate on a broad opening net can mean the bar has loosened — weigh the saved profiles' own evidence before either call.

{adaptation_matching_guidance}
{typed_signal_state}
{arch_review}
{typed_market_prior}

Return JSON with this structure:
- "new_strings": Array of objects with "boolean" (string), "rationale" (string), "family_key" (string), "novelty_bucket" ("edge_case"|"canonical"), "domain_lane" (string — one of the brief's declared lanes when listed above, else a lane label consistent with the plan's existing lanes; "general" only when none fits), and OPTIONAL "structured_filters" (object with "titles" and/or "companies" arrays of exact values). structured_filters is a deliberate per-string pool bound with the SAME rules as strategy formation: use it only when the string targets a title- or company-bounded lane, never as a default; NEVER locations (geography rides the session facet); harmony rule — a value belongs on exactly one surface, so a title or company placed on a filter must NOT also appear in the keyword boolean
- "new_retrieval_families": Optional array of structured family objects using the same schema as strategy formation
- "hypothesis_updates": Optional array of objects with "hypothesis_id", "status", "reason", and optional "promote_to_family_id"
- "skip_remaining": Array of objects with "string_id" (int), "reason" (string)
- "reorder": Array of objects with "string_id" (int), "move_to" ("next" | "last"), "reason" (string)
- "noise_updates": Array of objects with "term" (string), "status" ("confirmed_signal" | "confirmed_noise" | "mixed"), "note" (string)
- "pivot_to_architecture": (optional) New architecture name if recommending a pivot (string)
- "pivot_rationale": (optional) Why the current architecture failed and why the new one will work (string)
- "no_change": (optional) true if this block genuinely needs no adaptation — no new strings, skips, reorders, noise updates, or pivot. A deliberate decision, not a fallback for an empty response (bool)
- "next_checkpoint_after": (optional) request a different number of strings before the next adaptation checkpoint instead of the default. Clamped to 2-8 (int)

Return valid JSON only."""

    remaining_text = ""
    for ss in remaining_strings:
        metadata = (
            f"family={ss.family_key or 'unknown'} "
            f"novelty={ss.novelty_bucket or 'unknown'} "
            f"lane={ss.domain_lane or 'general'}"
        )
        remaining_text += f"  #{ss.id} [{metadata}]: {ss.boolean[:200]}\n"
        # Render the FILTER half of a filter-led lane (CLO-77). Until
        # 2026-08-05 this loop showed the Boolean only, so a lane that
        # deliberately puts its specificity in a title/company facet and leaves
        # the keyword half broad — the documented shape of a hybrid lane —
        # reached the judge stripped of the very thing that anchors it. The
        # cost was not theoretical: the PRRE campaign's #21 carried
        # titles=[Research Engineer, Research Scientist, Member of Technical
        # Staff, Principal Research Engineer] and was rendered as the bare
        # clause ("coding agents" OR "code generation" OR "SWE-bench" OR
        # "post-training" OR "RL environments" OR "evals"), then skipped as a
        # "grab-bag ... with no lane anchor" — a correct reading of what it was
        # shown and a wrong verdict on the string. All 7 filter-led lanes in
        # that campaign were culled the same way and none ever executed, which
        # is self-reinforcing: a lane that never runs never earns a yield stat.
        raw_filters = getattr(ss, "structured_filters", None) or {}
        if not isinstance(raw_filters, dict):
            to_dict = getattr(raw_filters, "to_dict", None)
            raw_filters = to_dict() if callable(to_dict) else {}
        bound_dims = []
        for dimension in ("titles", "companies"):
            values = [
                str(value).strip()
                for value in (raw_filters.get(dimension) or [])
                if str(value).strip()
            ]
            if values:
                bound_dims.append(f"{dimension}: {', '.join(values)}")
        if bound_dims:
            remaining_text += (
                "      POOL BOUND BY STRUCTURED FILTER (this is the lane's anchor, "
                "not the keywords) — " + " | ".join(bound_dims) + "\n"
            )

    def _sanitize_adaptation_structured_filters(adaptation: AdaptationResponse) -> None:
        """P2.3: enforce the deterministic slice of the filter rules.

        The adaptation slot admits titles/companies ONLY — locations ride the
        session facet, and any other dimension is not a mid-run lever. This is
        validation of model output, not decision-making: keys outside the
        allowlist are dropped (with a console note), never invented.
        """

        allowed = {"titles", "companies"}
        for item in adaptation.new_strings or []:
            raw = item.get("structured_filters")
            if not isinstance(raw, dict):
                if raw is not None:
                    item.pop("structured_filters", None)
                continue
            dropped = sorted(set(raw) - allowed)
            cleaned = {
                key: [str(v) for v in values]
                for key, values in raw.items()
                if key in allowed and isinstance(values, (list, tuple)) and values
            }
            if dropped:
                print(
                    "    [adapt] structured_filters: dropped disallowed "
                    f"dimension(s) {dropped} from a new string (titles/companies only)"
                )
            if cleaned:
                item["structured_filters"] = cleaned
            else:
                item.pop("structured_filters", None)

    user_prompt = f"""{block_report.to_summary_text()}

## Remaining Queued Strings ({len(remaining_strings)})
{remaining_text}
"""
    if search_memory_summary:
        user_prompt += (
            "\n## Search Family Memory\n"
            f"{format_search_memory_summary(search_memory_summary)}\n"
        )
    if vocab_section:
        user_prompt += f"""
## Kit Vocabulary (building blocks for new strings)
{vocab_section}
"""
    if use_layered_retrieval and not explicit_design.is_empty():
        user_prompt += (
            "\n## Current Layered Retrieval Design\n"
            f"{json.dumps(summarize_retrieval_design(explicit_design), indent=2)}\n"
        )

    user_prompt += "\nSuggest adaptations."

    usage_context = {
        "stage": "linkedin_adapt_after_block",
        "brief_id": brief.id,
        "block_name": block_report.block_name,
        "checkpoint_mode": checkpoint_mode,
    }
    result = opus_llm(
        system,
        user_prompt,
        expect_json=True,
        # Explicit headroom over the 8192 default: on always-thinking
        # models (claude-fable-5) max_tokens caps thinking + text jointly
        # (see the formation call's note), and an adaptation response
        # trails a large block report.
        max_tokens=16384,
        usage_context=usage_context,
        model_name=_config.STRATEGY_MODEL_NAME,
    )
    adaptation = AdaptationResponse.from_dict(result)
    requested_checkpoint = adaptation.next_checkpoint_after

    if adaptation.no_change:
        # P11.1: an explicit "no_change": true is a valid decision — distinct
        # from an empty/malformed response, which still fails above (the
        # opus_llm/json parse raises before AdaptationResponse.from_dict is
        # ever reached). Rebuild a clean response so nothing else the model
        # included (new_strings, skips, reorders, a pivot) can leak through:
        # the decline wins deterministically, not by convention.
        adaptation = AdaptationResponse(no_change=True)
    else:
        _sanitize_adaptation_structured_filters(adaptation)
        _materialize_retrieval_adaptation(
            adaptation,
            base_design=explicit_design if use_layered_retrieval else None,
            prefer_rendered_strings=use_layered_retrieval,
        )
        _scrub_blacklisted_employer_adaptation(brief, adaptation, execution_plan)
        _annotate_adaptation_metadata(brief, adaptation, plan=execution_plan)
        _apply_strict_seniority_adaptation_guardrails(brief, adaptation, remaining_strings)
        if checkpoint_mode != "opening_checkpoint" and _brief_targets_edge_case_opening(brief):
            adaptation = _rebalance_adaptation_for_edge_case_opening(
                brief, adaptation, remaining_strings
            )
            _annotate_adaptation_metadata(brief, adaptation, plan=execution_plan)
            _apply_strict_seniority_adaptation_guardrails(brief, adaptation, remaining_strings)
        _apply_search_memory_to_adaptation(adaptation, remaining_strings, search_memory_summary)
        firewall_trace = apply_adapted_string_firewall(
            adaptation.new_strings,
            # P5 (Wave 2): the ubiquity gate runs on a live feed — brief
            # blacklist terms plus the structural set — instead of a term
            # set no producer ever supplied (audit R2-F5 dead lever). Gate
            # hits drop PER STRING (recorded on the trace); the rest of the
            # adaptation decision survives.
            ubiquitous_terms=ubiquitous_terms_from_brief(brief),
            enable_token_subset_pruning=True,
        )
        for dropped in firewall_trace.dropped:
            print(
                "  [adapt] Ubiquity gate dropped adapted string "
                f"({dropped.get('rationale') or dropped.get('boolean', '')[:60]})"
            )
        setattr(adaptation, "adapted_string_firewall", firewall_trace.to_dict())

    # P11.2: model-requested adaptation cadence, clamped to a deterministic
    # bound. Absent/invalid requests leave next_checkpoint_after as None so
    # the orchestrator keeps its existing default cadence untouched.
    _clamp_next_checkpoint_after(adaptation, requested_checkpoint)
    setattr(adaptation, "search_signal_state", signal_state.to_dict())
    setattr(adaptation, "market_signal_prior", market_prior.to_dict())
    return adaptation
