"""Structured layered-retrieval design helpers.

This module defines the additive retrieval abstraction used to generate
search strings from layered intent rather than flat keyword lists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from typing import Any


LAYER_NAMES: tuple[str, ...] = (
    "entry_signals",
    "capability_proxies",
    "reality_filters",
    "context_constraints",
    "anti_noise",
)
DEFAULT_MAX_TERMS_PER_GROUP = 7
DEFAULT_VARIANTS_PER_FAMILY = 3

# build #2: a layer item may opt into a structured LinkedIn filter surface instead of a
# Boolean keyword. Allow-listed to title/company — location is the session-fact path
# (Pipeline._apply_session_location_filter), NOT a per-item layer surface, to avoid
# double-applying the brief geography. An unknown/typo value collapses to "" (boolean).
_STRUCTURED_LAYER_SURFACES = frozenset({"linkedin_title_filter", "linkedin_company_filter"})


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _slugify(value: Any) -> str:
    lowered = "".join(
        ch.lower() if str(ch).isalnum() else "_" for ch in str(value or "")
    )
    while "__" in lowered:
        lowered = lowered.replace("__", "_")
    return lowered.strip("_") or "unknown"


def _dedupe_strings(values: list[Any], limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = _normalize_text(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit and len(out) >= limit:
            break
    return out


@dataclass
class RetrievalLayerItem:
    item_id: str
    label: str
    terms: list[str]
    priority: int = 50
    enabled: bool = True
    rationale: str = ""
    # build #2: optional structured-filter surface ("" = Boolean keyword, unchanged
    # default). Allow-listed to title/company in from_value; anti_noise never honors it.
    structured_surface: str = ""

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        fallback_prefix: str,
        index: int,
    ) -> "RetrievalLayerItem | None":
        if isinstance(value, str):
            text = _normalize_text(value)
            if not text:
                return None
            return cls(
                item_id=f"{fallback_prefix}_{index}",
                label=text,
                terms=[text],
            )
        if not isinstance(value, dict):
            return None
        terms = _dedupe_strings(value.get("terms", []))
        if not terms:
            single = _normalize_text(value.get("term"))
            if single:
                terms = [single]
        label = _normalize_text(value.get("label")) or ", ".join(terms[:2])
        if not (label and terms):
            return None
        raw_surface = _normalize_text(value.get("structured_surface"))
        structured_surface = raw_surface if raw_surface in _STRUCTURED_LAYER_SURFACES else ""
        return cls(
            item_id=_normalize_text(value.get("item_id"))
            or _normalize_text(value.get("id"))
            or f"{fallback_prefix}_{index}",
            label=label,
            terms=terms,
            priority=int(value.get("priority", 50) or 50),
            enabled=bool(value.get("enabled", True)),
            rationale=_normalize_text(value.get("rationale")),
            structured_surface=structured_surface,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalEdgeCaseHypothesis:
    hypothesis_id: str
    label: str
    hidden_cohort: str
    why_missed: str
    entry_signal_variants: list[RetrievalLayerItem] = field(default_factory=list)
    capability_proxy_variants: list[RetrievalLayerItem] = field(default_factory=list)
    reality_filter_variants: list[RetrievalLayerItem] = field(default_factory=list)
    context_constraint_variants: list[RetrievalLayerItem] = field(default_factory=list)
    anti_noise_variants: list[RetrievalLayerItem] = field(default_factory=list)
    noise_risks: list[str] = field(default_factory=list)
    validation_rule: str = ""
    status: str = "hypothesis"
    confidence: float = 0.0
    source: str = "brief"
    supporting_run_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RetrievalEdgeCaseHypothesis | None":
        payload = payload or {}
        label = _normalize_text(payload.get("label"))
        hidden_cohort = _normalize_text(payload.get("hidden_cohort"))
        why_missed = _normalize_text(payload.get("why_missed"))
        if not (label and hidden_cohort and why_missed):
            return None
        return cls(
            hypothesis_id=_normalize_text(payload.get("hypothesis_id"))
            or _slugify(label),
            label=label,
            hidden_cohort=hidden_cohort,
            why_missed=why_missed,
            entry_signal_variants=_parse_layer_items(
                payload.get("entry_signal_variants", []),
                "edge_entry",
            ),
            capability_proxy_variants=_parse_layer_items(
                payload.get("capability_proxy_variants", []),
                "edge_capability",
            ),
            reality_filter_variants=_parse_layer_items(
                payload.get("reality_filter_variants", []),
                "edge_reality",
            ),
            context_constraint_variants=_parse_layer_items(
                payload.get("context_constraint_variants", []),
                "edge_context",
            ),
            anti_noise_variants=_parse_layer_items(
                payload.get("anti_noise_variants", []),
                "edge_anti_noise",
            ),
            noise_risks=_dedupe_strings(payload.get("noise_risks", [])),
            validation_rule=_normalize_text(payload.get("validation_rule")),
            status=_normalize_text(payload.get("status")) or "hypothesis",
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            source=_normalize_text(payload.get("source")) or "brief",
            supporting_run_refs=_dedupe_strings(payload.get("supporting_run_refs", [])),
            evidence_refs=_dedupe_strings(payload.get("evidence_refs", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "entry_signal_variants",
            "capability_proxy_variants",
            "reality_filter_variants",
            "context_constraint_variants",
            "anti_noise_variants",
        ):
            data[key] = [item.to_dict() for item in getattr(self, key)]
        return data


@dataclass
class RetrievalSharedLayers:
    reality_filters: list[RetrievalLayerItem] = field(default_factory=list)
    context_constraints: list[RetrievalLayerItem] = field(default_factory=list)
    anti_noise: list[RetrievalLayerItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RetrievalSharedLayers":
        payload = payload or {}
        return cls(
            reality_filters=_parse_layer_items(
                payload.get("reality_filters", []),
                "shared_reality",
            ),
            context_constraints=_parse_layer_items(
                payload.get("context_constraints", []),
                "shared_context",
            ),
            anti_noise=_parse_layer_items(
                payload.get("anti_noise", []),
                "shared_anti_noise",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reality_filters": [item.to_dict() for item in self.reality_filters],
            "context_constraints": [item.to_dict() for item in self.context_constraints],
            "anti_noise": [item.to_dict() for item in self.anti_noise],
        }


@dataclass
class RetrievalFamily:
    family_id: str
    label: str
    objective: str
    priority: int = 50
    enabled: bool = True
    entry_signals: list[RetrievalLayerItem] = field(default_factory=list)
    capability_proxies: list[RetrievalLayerItem] = field(default_factory=list)
    reality_filters: list[RetrievalLayerItem] = field(default_factory=list)
    context_constraints: list[RetrievalLayerItem] = field(default_factory=list)
    anti_noise: list[RetrievalLayerItem] = field(default_factory=list)
    target_employers: list[str] = field(default_factory=list)
    target_markets: list[str] = field(default_factory=list)
    variants_to_emit: int = 1
    hypothesis_ids: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RetrievalFamily | None":
        payload = payload or {}
        label = _normalize_text(payload.get("label"))
        objective = _normalize_text(payload.get("objective"))
        if not label:
            return None
        return cls(
            family_id=_normalize_text(payload.get("family_id"))
            or _normalize_text(payload.get("id"))
            or _slugify(label),
            label=label,
            objective=objective or label,
            priority=int(payload.get("priority", 50) or 50),
            enabled=bool(payload.get("enabled", True)),
            entry_signals=_parse_layer_items(
                payload.get("entry_signals", []),
                f"{_slugify(label)}_entry",
            ),
            capability_proxies=_parse_layer_items(
                payload.get("capability_proxies", []),
                f"{_slugify(label)}_capability",
            ),
            reality_filters=_parse_layer_items(
                payload.get("reality_filters", []),
                f"{_slugify(label)}_reality",
            ),
            context_constraints=_parse_layer_items(
                payload.get("context_constraints", []),
                f"{_slugify(label)}_context",
            ),
            anti_noise=_parse_layer_items(
                payload.get("anti_noise", []),
                f"{_slugify(label)}_anti_noise",
            ),
            target_employers=_dedupe_strings(payload.get("target_employers", [])),
            target_markets=_dedupe_strings(payload.get("target_markets", [])),
            variants_to_emit=max(1, int(payload.get("variants_to_emit", 1) or 1)),
            hypothesis_ids=_dedupe_strings(payload.get("hypothesis_ids", [])),
            notes=_normalize_text(payload.get("notes")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "label": self.label,
            "objective": self.objective,
            "priority": self.priority,
            "enabled": self.enabled,
            "entry_signals": [item.to_dict() for item in self.entry_signals],
            "capability_proxies": [item.to_dict() for item in self.capability_proxies],
            "reality_filters": [item.to_dict() for item in self.reality_filters],
            "context_constraints": [item.to_dict() for item in self.context_constraints],
            "anti_noise": [item.to_dict() for item in self.anti_noise],
            "target_employers": list(self.target_employers),
            "target_markets": list(self.target_markets),
            "variants_to_emit": self.variants_to_emit,
            "hypothesis_ids": list(self.hypothesis_ids),
            "notes": self.notes,
        }


@dataclass
class RetrievalDesign:
    families: list[RetrievalFamily] = field(default_factory=list)
    shared_layers: RetrievalSharedLayers = field(default_factory=RetrievalSharedLayers)
    edge_case_hypotheses: list[RetrievalEdgeCaseHypothesis] = field(default_factory=list)
    derived_from_legacy: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RetrievalDesign":
        payload = payload or {}
        families = [
            family
            for family in (
                RetrievalFamily.from_dict(item)
                for item in payload.get("families", [])
            )
            if family is not None
        ]
        families.sort(key=lambda item: (-item.priority, item.label.lower()))
        hypotheses = [
            hypothesis
            for hypothesis in (
                RetrievalEdgeCaseHypothesis.from_dict(item)
                for item in payload.get("edge_case_hypotheses", [])
            )
            if hypothesis is not None
        ]
        return cls(
            families=families,
            shared_layers=RetrievalSharedLayers.from_dict(payload.get("shared_layers")),
            edge_case_hypotheses=hypotheses,
            derived_from_legacy=bool(payload.get("derived_from_legacy", False)),
        )

    def is_empty(self) -> bool:
        return not (
            self.families
            or self.shared_layers.reality_filters
            or self.shared_layers.context_constraints
            or self.shared_layers.anti_noise
            or self.edge_case_hypotheses
        )

    def is_explicit(self) -> bool:
        return not self.is_empty() and not self.derived_from_legacy

    def to_dict(self) -> dict[str, Any]:
        return {
            "families": [family.to_dict() for family in self.families],
            "shared_layers": self.shared_layers.to_dict(),
            "edge_case_hypotheses": [
                hypothesis.to_dict() for hypothesis in self.edge_case_hypotheses
            ],
            "derived_from_legacy": self.derived_from_legacy,
        }


def _parse_layer_items(values: Any, prefix: str) -> list[RetrievalLayerItem]:
    if not isinstance(values, list):
        return []
    items: list[RetrievalLayerItem] = []
    for index, value in enumerate(values, start=1):
        item = RetrievalLayerItem.from_value(
            value,
            fallback_prefix=prefix,
            index=index,
        )
        if item is not None:
            items.append(item)
    items.sort(key=lambda item: (-item.priority, item.label.lower()))
    return items


def retrieval_design_from_payload(
    payload: Any,
    *,
    legacy_search_priorities: list[str] | None = None,
    legacy_additional_search_terms: list[str] | None = None,
    role_title: str = "",
) -> RetrievalDesign:
    if isinstance(payload, RetrievalDesign):
        return payload
    if isinstance(payload, dict):
        design = RetrievalDesign.from_dict(payload)
        if not design.is_empty():
            return design
    return infer_retrieval_design_from_legacy(
        search_priorities=legacy_search_priorities or [],
        additional_search_terms=legacy_additional_search_terms or [],
        role_title=role_title,
    )


def infer_retrieval_design_from_legacy(
    *,
    search_priorities: list[str],
    additional_search_terms: list[str],
    role_title: str = "",
) -> RetrievalDesign:
    entry_signals = _parse_layer_items(
        [{"label": text, "terms": [text], "priority": max(10, 100 - index * 5)}
         for index, text in enumerate(_dedupe_strings(search_priorities[:6]), start=1)],
        "legacy_entry",
    )
    capability_proxies = _parse_layer_items(
        [{"label": text, "terms": [text], "priority": max(10, 100 - index * 2)}
         for index, text in enumerate(_dedupe_strings(additional_search_terms[:24]), start=1)],
        "legacy_capability",
    )
    label = _normalize_text(role_title) or "Legacy derived retrieval family"
    family = RetrievalFamily(
        family_id=_slugify(f"{label}_legacy"),
        label=f"{label} / legacy derived",
        objective="Compatibility family derived from legacy search priorities and additional search terms.",
        priority=50,
        enabled=True,
        entry_signals=entry_signals[:6],
        capability_proxies=capability_proxies[:12],
        variants_to_emit=2 if entry_signals and capability_proxies else 1,
        notes="Auto-derived from legacy search hints.",
    )
    return RetrievalDesign(
        families=[family] if (family.entry_signals or family.capability_proxies) else [],
        derived_from_legacy=True,
    )


def derive_legacy_search_views(design: RetrievalDesign) -> tuple[list[str], list[str]]:
    priorities: list[str] = []
    terms: list[str] = []

    for family in design.families:
        if not family.enabled:
            continue
        summary_parts = [
            family.label,
            family.objective,
        ]
        if family.target_markets:
            summary_parts.append(", ".join(family.target_markets[:2]))
        if family.target_employers:
            summary_parts.append(", ".join(family.target_employers[:3]))
        priorities.append(" — ".join(part for part in summary_parts if part))

        for layer_name in ("entry_signals", "capability_proxies", "reality_filters"):
            for item in getattr(family, layer_name):
                terms.extend(item.terms)
        for item in family.context_constraints:
            terms.extend(item.terms[:2])

    for hypothesis in design.edge_case_hypotheses:
        if hypothesis.status in {"rejected", "inactive"}:
            continue
        priorities.append(f"Edge-case hypothesis — {hypothesis.label}: {hypothesis.hidden_cohort}")
        for items in (
            hypothesis.entry_signal_variants,
            hypothesis.capability_proxy_variants,
            hypothesis.reality_filter_variants,
        ):
            for item in items:
                terms.extend(item.terms)

    return (
        _dedupe_strings(priorities, limit=12),
        _dedupe_strings(terms, limit=120),
    )


def validate_retrieval_design(design: RetrievalDesign) -> list[str]:
    issues: list[str] = []
    for family in design.families:
        if not family.enabled:
            continue
        if not family.entry_signals and not family.capability_proxies:
            issues.append(f"Retrieval family '{family.family_id}' has no entry signals or capability proxies.")
        if family.entry_signals and not (family.capability_proxies or family.reality_filters):
            issues.append(
                f"Retrieval family '{family.family_id}' is too broad: entry signals need capability proxies or reality filters."
            )
        anti_noise_terms = _dedupe_strings(
            [term for item in family.anti_noise for term in item.terms]
        )
        positive_terms = _dedupe_strings(
            [term for item in family.entry_signals + family.capability_proxies for term in item.terms]
        )
        if anti_noise_terms and positive_terms:
            overlap = {term.lower() for term in anti_noise_terms} & {
                term.lower() for term in positive_terms
            }
            if overlap:
                issues.append(
                    f"Retrieval family '{family.family_id}' has anti-noise terms that collide with positive intent: {', '.join(sorted(overlap))}."
                )

    for hypothesis in design.edge_case_hypotheses:
        if not hypothesis.validation_rule:
            issues.append(
                f"Edge-case hypothesis '{hypothesis.hypothesis_id}' is missing a validation rule."
            )
        if not (
            hypothesis.entry_signal_variants
            or hypothesis.capability_proxy_variants
            or hypothesis.reality_filter_variants
        ):
            issues.append(
                f"Edge-case hypothesis '{hypothesis.hypothesis_id}' does not perturb any retrieval layer."
            )
    return issues


def _take_terms(items: list[RetrievalLayerItem], *, per_item: int = 3, cap: int = DEFAULT_MAX_TERMS_PER_GROUP) -> list[str]:
    terms: list[str] = []
    for item in items:
        if not item.enabled:
            continue
        terms.extend(item.terms[:per_item])
        if len(terms) >= cap:
            break
    return _dedupe_strings(terms, limit=cap)


def _render_group(terms: list[str]) -> str:
    cleaned = _dedupe_strings(terms, limit=DEFAULT_MAX_TERMS_PER_GROUP)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return f"(\"{cleaned[0]}\")"
    return "(" + " OR ".join(f"\"{term}\"" for term in cleaned) + ")"


def _apply_hypothesis_overlays(
    family: RetrievalFamily,
    design: RetrievalDesign,
) -> tuple[list[RetrievalLayerItem], list[RetrievalLayerItem], list[RetrievalLayerItem], list[RetrievalLayerItem], list[RetrievalLayerItem], list[str]]:
    entry = list(family.entry_signals)
    capability = list(family.capability_proxies)
    reality = list(family.reality_filters) + list(design.shared_layers.reality_filters)
    context = list(family.context_constraints) + list(design.shared_layers.context_constraints)
    anti_noise = list(family.anti_noise) + list(design.shared_layers.anti_noise)
    applied_hypothesis_ids: list[str] = []

    by_id = {item.hypothesis_id: item for item in design.edge_case_hypotheses}
    for hypothesis_id in family.hypothesis_ids:
        hypothesis = by_id.get(hypothesis_id)
        if not hypothesis or hypothesis.status in {"rejected", "inactive"}:
            continue
        applied_hypothesis_ids.append(hypothesis.hypothesis_id)
        entry.extend(hypothesis.entry_signal_variants)
        capability.extend(hypothesis.capability_proxy_variants)
        reality.extend(hypothesis.reality_filter_variants)
        context.extend(hypothesis.context_constraint_variants)
        anti_noise.extend(hypothesis.anti_noise_variants)

    for layer in (entry, capability, reality, context, anti_noise):
        layer.sort(key=lambda item: (-item.priority, item.label.lower()))

    return entry, capability, reality, context, anti_noise, applied_hypothesis_ids


def render_family_variants(
    family: RetrievalFamily,
    design: RetrievalDesign,
) -> list[dict[str, Any]]:
    if not family.enabled:
        return []

    entry, capability, reality, context, anti_noise, applied_hypothesis_ids = _apply_hypothesis_overlays(
        family,
        design,
    )
    referenced_hypothesis_ids = (
        list(applied_hypothesis_ids)
        if applied_hypothesis_ids
        else list(family.hypothesis_ids)
    )

    # P2.4 (one carrier for structured surfaces): an item carrying a
    # structured_surface is EXCLUDED from the keyword OR-groups and its FULL
    # term list (no truncation) is emitted as variant["structured_filters"]
    # instead. Previously these terms were folded into the keyword boolean
    # (truncated to 2-4) and the lane compiler re-derived the filters, with
    # the normalizer cleaning up the duplication. Locations never ride this
    # path (_STRUCTURED_LAYER_SURFACES admits title/company only); anti_noise
    # never honors structured_surface. Families with no structured surfaces
    # render byte-identically (the partition is a no-op and the key is
    # omitted when empty).
    def _split_structured(
        items: list[RetrievalLayerItem],
    ) -> tuple[list[RetrievalLayerItem], list[RetrievalLayerItem]]:
        keyword_items = [item for item in items if not item.structured_surface]
        structured_items = [item for item in items if item.structured_surface]
        return keyword_items, structured_items

    entry, entry_structured = _split_structured(entry)
    capability, capability_structured = _split_structured(capability)
    reality, reality_structured = _split_structured(reality)
    context, context_structured = _split_structured(context)
    _surface_to_dimension = {
        "linkedin_title_filter": "titles",
        "linkedin_company_filter": "companies",
    }
    structured_filters: dict[str, list[str]] = {}
    for item in (
        entry_structured + capability_structured + reality_structured + context_structured
    ):
        dimension = _surface_to_dimension.get(item.structured_surface)
        if not dimension:
            continue
        bucket = structured_filters.setdefault(dimension, [])
        for term in item.terms:
            text = _normalize_text(term)
            if text and text not in bucket:
                bucket.append(text)

    if not (entry or capability):
        return []

    entry_groups = entry[: max(1, family.variants_to_emit)]
    capability_groups = capability[: max(1, family.variants_to_emit)]
    variant_pairs = list(
        zip_longest(
            entry_groups or [None],
            capability_groups or [None],
            fillvalue=None,
        )
    )
    variants: list[dict[str, Any]] = []
    requested = max(1, min(DEFAULT_VARIANTS_PER_FAMILY, family.variants_to_emit))

    for index, (entry_item, capability_item) in enumerate(variant_pairs[:requested], start=1):
        entry_terms = _take_terms([entry_item] if entry_item else entry, per_item=4)
        capability_terms = _take_terms([capability_item] if capability_item else capability, per_item=4)
        reality_terms = _take_terms(reality, per_item=3, cap=6)
        context_terms = _take_terms(context, per_item=2, cap=4)
        anti_noise_terms = _take_terms(anti_noise, per_item=2, cap=5)

        groups = [
            _render_group(entry_terms),
            _render_group(capability_terms),
            _render_group(reality_terms),
        ]
        groups = [group for group in groups if group]
        if context_terms:
            groups.append(_render_group(context_terms))
        if not groups:
            continue
        boolean = " AND ".join(groups[:4])
        anti_noise_group = _render_group(anti_noise_terms)
        if anti_noise_group:
            boolean = f"{boolean} NOT {anti_noise_group}"

        used_layer_items = {
            "entry_signals": [item.item_id for item in ([entry_item] if entry_item else []) if item],
            "capability_proxies": [item.item_id for item in ([capability_item] if capability_item else []) if item],
            "reality_filters": [item.item_id for item in reality[:2]],
            "context_constraints": [item.item_id for item in context[:2]],
            "anti_noise": [item.item_id for item in anti_noise[:2]],
        }
        variant: dict[str, Any] = {
            "boolean": boolean,
            "rationale": (
                f"{family.label}: broad retrieval doorway paired with capability and execution filters."
            ),
            "family_key": family.family_id,
            "novelty_bucket": "edge_case" if referenced_hypothesis_ids else "canonical",
            "domain_lane": _slugify(family.target_markets[0]) if family.target_markets else "general",
            "retrieval_hypothesis_ids": list(referenced_hypothesis_ids),
            "retrieval_recipe": {
                "family_id": family.family_id,
                "family_label": family.label,
                "variant_index": index,
                "objective": family.objective,
                "used_layer_item_ids": used_layer_items,
                "applied_hypothesis_ids": applied_hypothesis_ids,
                "referenced_hypothesis_ids": list(referenced_hypothesis_ids),
                "target_employers": list(family.target_employers),
                "target_markets": list(family.target_markets),
            },
        }
        if structured_filters:
            variant["structured_filters"] = {
                dimension: list(values)
                for dimension, values in structured_filters.items()
            }
        variants.append(variant)
    return variants


def render_retrieval_design(design: RetrievalDesign) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families_payload: list[dict[str, Any]] = []
    generated_strings: list[dict[str, Any]] = []
    for family in sorted(
        design.families,
        key=lambda item: (-item.priority, item.label.lower()),
    ):
        rendered = render_family_variants(family, design)
        if not rendered:
            continue
        family_record = family.to_dict()
        family_record["rendered_variant_count"] = len(rendered)
        family_record["rendered_variants"] = [
            {
                "boolean": item["boolean"],
                "variant_index": item["retrieval_recipe"]["variant_index"],
                "applied_hypothesis_ids": item["retrieval_recipe"]["applied_hypothesis_ids"],
            }
            for item in rendered
        ]
        families_payload.append(family_record)
        generated_strings.extend(rendered)
    return families_payload, generated_strings


def summarize_retrieval_design(design: RetrievalDesign) -> dict[str, Any]:
    families_summary = []
    for family in design.families:
        families_summary.append(
            {
                "family_id": family.family_id,
                "label": family.label,
                "objective": family.objective,
                "priority": family.priority,
                "entry_signal_labels": [item.label for item in family.entry_signals[:4]],
                "capability_proxy_labels": [item.label for item in family.capability_proxies[:4]],
                "reality_filter_labels": [item.label for item in family.reality_filters[:3]],
                "context_constraint_labels": [item.label for item in family.context_constraints[:3]],
                "anti_noise_labels": [item.label for item in family.anti_noise[:3]],
                "hypothesis_ids": list(family.hypothesis_ids),
            }
        )
    hypotheses_summary = []
    for hypothesis in design.edge_case_hypotheses:
        hypotheses_summary.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "label": hypothesis.label,
                "hidden_cohort": hypothesis.hidden_cohort,
                "status": hypothesis.status,
                "confidence": hypothesis.confidence,
                "source": hypothesis.source,
                "validation_rule": hypothesis.validation_rule,
            }
        )
    priorities, terms = derive_legacy_search_views(design)
    return {
        "families": families_summary,
        "edge_case_hypotheses": hypotheses_summary,
        "derived_search_priorities": priorities,
        "derived_additional_search_terms": terms,
        "derived_from_legacy": design.derived_from_legacy,
    }
