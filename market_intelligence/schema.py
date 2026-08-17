"""Validated market-intelligence schema and markdown rendering helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any


def _require_dict(name: str, value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict")
    return value


def _require_list(name: str, value: Any) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _stringify_list(value: list[Any]) -> list[str]:
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _slugify(value: Any) -> str:
    lowered = "".join(
        ch.lower() if str(ch).isalnum() else "_" for ch in str(value or "")
    )
    while "__" in lowered:
        lowered = lowered.replace("__", "_")
    return lowered.strip("_") or "unknown"


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def market_thesis_summary_looks_like_review(summary: Any) -> bool:
    text = _normalize_text(summary).lower()
    if not text:
        return False
    if text.startswith(
        (
            "the market thesis summary",
            "market thesis summary",
            "the summary is",
            "this summary is",
            "the draft is",
            "this draft is",
        )
    ):
        return True
    direct_markers = (
        "well-written",
        "directionally correct",
        "needs the following adjustments",
        "summary_assessment",
        "confidence_override",
        "keep_summary_with_edits",
        "strengthen",
        "weaken",
    )
    if any(marker in text for marker in direct_markers):
        return True
    review_score = 0
    if "keep" in text:
        review_score += 1
    if "claim" in text:
        review_score += 1
    if "opening" in text:
        review_score += 1
    if "adjustment" in text or "adjustments" in text:
        review_score += 1
    if "however" in text:
        review_score += 1
    if "directionally" in text:
        review_score += 1
    if review_score >= 3:
        return True
    return False


def public_market_thesis_summary(summary: Any) -> str:
    text = _normalize_text(summary)
    if not text:
        return ""
    if market_thesis_summary_looks_like_review(text):
        return ""
    return text


def _record_has_refs(record: dict) -> bool:
    return bool(
        _stringify_list(record.get("supporting_run_refs", []))
        or _stringify_list(record.get("evidence_refs", []))
    )


@dataclass(frozen=True)
class NarrativeSectionPolicy:
    required_keys: tuple[str, ...]
    provenance_required: bool = True


NARRATIVE_SECTION_POLICIES: dict[str, NarrativeSectionPolicy] = {
    "lane_intelligence": NarrativeSectionPolicy(
        required_keys=(
            "lane_key",
            "domain_lane",
            "novelty_bucket",
            "status",
            "first_seen_at",
            "last_seen_at",
            "supporting_run_refs",
            "metrics",
            "dominant_anchors",
        )
    ),
    "talent_pool_intelligence": NarrativeSectionPolicy(
        required_keys=(
            "pool_key",
            "label",
            "status",
            "signal_strength",
            "evidence_summary",
        )
    ),
    "noise_patterns": NarrativeSectionPolicy(
        required_keys=("pattern_key", "label", "severity", "mitigations")
    ),
    "employer_signal_intelligence": NarrativeSectionPolicy(
        required_keys=("cluster_key", "label", "status", "supporting_employers")
    ),
    "brief_recommendations": NarrativeSectionPolicy(
        required_keys=(
            "recommendation_id",
            "target_field",
            "proposal",
            "reason",
            "confidence",
        )
    ),
    "open_questions": NarrativeSectionPolicy(
        required_keys=("question", "priority", "next_step")
    ),
    "market_thesis.external_context": NarrativeSectionPolicy(
        required_keys=("claim", "evidence_refs", "confidence")
    ),
}

PROVENANCE_BEARING_SECTIONS = tuple(sorted(NARRATIVE_SECTION_POLICIES))
NARRATIVE_SECTION_EXCEPTIONS: tuple[str, ...] = ()
SOURCING_IMPLICATION_CATEGORIES: tuple[str, ...] = (
    "add_title_family",
    "add_employer_target",
    "probe_adjacent_pool",
    "relax_boolean",
    "validate_hypothesis",
    "instrumentation_followup",
)
MARKET_FINDING_KIND_ALIASES: dict[str, str] = {
    "title_variant": "title_variant",
    "title_variants": "title_variant",
    "title_family": "title_variant",
    "title_families": "title_variant",
    "employer_cluster": "employer_cluster",
    "employer_clusters": "employer_cluster",
    "employer_group": "employer_cluster",
    "talent_pool": "talent_pool",
    "talent_pools": "talent_pool",
    "market_condition": "market_condition",
    "market_conditions": "market_condition",
    "consulting_overlap": "consulting_overlap",
    "consulting_overlaps": "consulting_overlap",
    "adjacent_archetype": "adjacent_archetype",
    "adjacent_archetypes": "adjacent_archetype",
}
SOURCING_IMPLICATION_CATEGORY_ALIASES: dict[str, str] = {
    "add_title_family": "add_title_family",
    "add_title_families": "add_title_family",
    "add_employer_target": "add_employer_target",
    "add_employer_targets": "add_employer_target",
    "probe_adjacent_pool": "probe_adjacent_pool",
    "probe_adjacent_pools": "probe_adjacent_pool",
    "relax_boolean": "relax_boolean",
    "relax_booleans": "relax_boolean",
    "validate_hypothesis": "validate_hypothesis",
    "validate_hypotheses": "validate_hypothesis",
    "instrumentation_followup": "instrumentation_followup",
    "instrumentation_followups": "instrumentation_followup",
}
BRIEF_TARGET_FIELD_ALIASES: dict[str, str] = {
    "search_priorities": "search_priorities",
    "search_priority": "search_priorities",
    "additional_search_terms": "additional_search_terms",
    "additional_search_term": "additional_search_terms",
    "search_terms": "additional_search_terms",
    "search_term": "additional_search_terms",
    "title_family": "additional_search_terms",
    "title_families": "additional_search_terms",
    "title_variant": "additional_search_terms",
    "title_variants": "additional_search_terms",
    "keyword_anchor": "additional_search_terms",
    "keyword_anchors": "additional_search_terms",
    "employer_signal_rules": "employer_signal_rules",
    "employer_signal_rule": "employer_signal_rules",
    "employer_target": "employer_signal_rules",
    "employer_targets": "employer_signal_rules",
    "target_employers": "employer_signal_rules",
    "notes": "notes",
    "note": "notes",
    "instructions": "instructions",
    "instruction": "instructions",
    "intake_notes": "notes",
    "retrieval_design": "retrieval_design",
    "retrieval_family": "retrieval_design",
    "retrieval_families": "retrieval_design",
    "entry_signals": "retrieval_design",
    "capability_proxies": "retrieval_design",
    "reality_filters": "retrieval_design",
    "context_constraints": "retrieval_design",
    "anti_noise": "retrieval_design",
    "edge_case_hypotheses": "retrieval_design",
}


def _normalize_priority(value: Any) -> str:
    normalized = str(value or "medium").strip().lower()
    return normalized if normalized in {"high", "medium", "low"} else "medium"


def _normalize_label_key(value: Any) -> str:
    lowered = "".join(
        ch.lower() if str(ch).isalnum() else "_" for ch in str(value or "")
    )
    while "__" in lowered:
        lowered = lowered.replace("__", "_")
    return lowered.strip("_")


def _normalize_market_finding_kind(value: Any) -> str:
    normalized = _normalize_label_key(value)
    return MARKET_FINDING_KIND_ALIASES.get(normalized, normalized)


def _normalize_sourcing_implication_category(value: Any) -> str:
    normalized = _normalize_label_key(value)
    return SOURCING_IMPLICATION_CATEGORY_ALIASES.get(normalized, normalized)


def _normalize_brief_target_field(value: Any) -> str:
    normalized = _normalize_label_key(value)
    return BRIEF_TARGET_FIELD_ALIASES.get(normalized, "")


def sanitize_inferred_research_questions(
    items: Any,
    *,
    default_supporting_run_refs: list[str] | None = None,
) -> list[dict]:
    if not isinstance(items, list):
        return []
    default_refs = _stringify_list(default_supporting_run_refs or [])
    normalized_items: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_refs(item)
        question = str(normalized.get("question", "")).strip()
        why_it_matters = str(normalized.get("why_it_matters", "")).strip()
        sourcing_trigger = str(normalized.get("sourcing_trigger", "")).strip()
        status = str(normalized.get("status", "unresolved")).strip().lower()
        if status not in {"answered", "unresolved"}:
            status = "unresolved"
        supporting_run_refs = _stringify_list(
            normalized.get("supporting_run_refs", []) or default_refs
        )
        evidence_refs = _stringify_list(normalized.get("evidence_refs", []))
        if not (question and why_it_matters and sourcing_trigger):
            continue
        record = {
            "question": question,
            "priority": _normalize_priority(normalized.get("priority")),
            "why_it_matters": why_it_matters,
            "sourcing_trigger": sourcing_trigger,
            "status": status,
            "supporting_run_refs": supporting_run_refs,
            "evidence_refs": evidence_refs,
        }
        if not _record_has_refs(record):
            continue
        normalized_items.append(record)
    return normalized_items


def sanitize_market_findings(
    items: Any,
    *,
    default_supporting_run_refs: list[str] | None = None,
) -> list[dict]:
    if not isinstance(items, list):
        return []
    default_refs = _stringify_list(default_supporting_run_refs or [])
    normalized_items: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_refs(item)
        kind = _normalize_market_finding_kind(normalized.get("kind", ""))
        label = str(normalized.get("label", "")).strip()
        summary = str(normalized.get("summary", "")).strip()
        why_it_matters = str(normalized.get("why_it_matters", "")).strip()
        supporting_run_refs = _stringify_list(
            normalized.get("supporting_run_refs", []) or default_refs
        )
        evidence_refs = _stringify_list(normalized.get("evidence_refs", []))
        if not (kind and label and summary and why_it_matters):
            continue
        record = {
            "finding_key": str(normalized.get("finding_key", "")).strip()
            or _slugify(f"{kind}-{label}"),
            "kind": kind,
            "label": label,
            "summary": summary,
            "why_it_matters": why_it_matters,
            "confidence": min(
                1.0,
                max(0.0, float(normalized.get("confidence", 0.5) or 0.5)),
            ),
            "supporting_run_refs": supporting_run_refs,
            "evidence_refs": evidence_refs,
        }
        if not _record_has_refs(record):
            continue
        normalized_items.append(record)
    return normalized_items


def sanitize_sourcing_implications(
    items: Any,
    *,
    default_supporting_run_refs: list[str] | None = None,
) -> list[dict]:
    if not isinstance(items, list):
        return []
    default_refs = _stringify_list(default_supporting_run_refs or [])
    normalized_items: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_refs(item)
        category = _normalize_sourcing_implication_category(
            normalized.get("category", "")
        )
        recommendation = str(normalized.get("recommendation", "")).strip()
        rationale = str(normalized.get("rationale", "")).strip()
        supporting_run_refs = _stringify_list(
            normalized.get("supporting_run_refs", []) or default_refs
        )
        evidence_refs = _stringify_list(normalized.get("evidence_refs", []))
        if (
            category not in SOURCING_IMPLICATION_CATEGORIES
            or not recommendation
            or not rationale
        ):
            continue
        record = {
            "implication_id": str(normalized.get("implication_id", "")).strip()
            or _slugify(f"{category}-{recommendation}"),
            "category": category,
            "priority": _normalize_priority(normalized.get("priority")),
            "recommendation": recommendation,
            "rationale": rationale,
            "brief_target_field": _normalize_brief_target_field(
                normalized.get("brief_target_field", "")
            ),
            "suggested_values": _stringify_list(
                normalized.get("suggested_values", [])
            ),
            "expected_effect": str(normalized.get("expected_effect", "")).strip(),
            "supporting_run_refs": supporting_run_refs,
            "evidence_refs": evidence_refs,
        }
        if not _record_has_refs(record):
            continue
        normalized_items.append(record)
    return normalized_items


def sanitize_edge_case_submarkets(
    items: Any,
    *,
    default_supporting_run_refs: list[str] | None = None,
) -> list[dict]:
    if not isinstance(items, list):
        return []
    default_refs = _stringify_list(default_supporting_run_refs or [])
    normalized_items: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_refs(item)
        label = str(normalized.get("label", "")).strip()
        summary = str(normalized.get("summary", "")).strip()
        why_easy_to_miss = str(normalized.get("why_it_is_easy_to_miss", "")).strip()
        supporting_run_refs = _stringify_list(
            normalized.get("supporting_run_refs", []) or default_refs
        )
        evidence_refs = _stringify_list(normalized.get("evidence_refs", []))
        if not (label and summary and why_easy_to_miss):
            continue
        record = {
            "submarket_key": str(normalized.get("submarket_key", "")).strip()
            or _slugify(label),
            "label": label,
            "summary": summary,
            "why_it_is_easy_to_miss": why_easy_to_miss,
            "confidence": min(
                1.0,
                max(0.0, float(normalized.get("confidence", 0.5) or 0.5)),
            ),
            "supporting_run_refs": supporting_run_refs,
            "evidence_refs": evidence_refs,
        }
        if not _record_has_refs(record):
            continue
        normalized_items.append(record)
    return normalized_items


def sanitize_title_to_archetype_mapping(
    items: Any,
    *,
    default_supporting_run_refs: list[str] | None = None,
) -> list[dict]:
    if not isinstance(items, list):
        return []
    default_refs = _stringify_list(default_supporting_run_refs or [])
    normalized_items: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_refs(item)
        title_family = str(normalized.get("title_family", "")).strip()
        likely_archetype = str(normalized.get("likely_archetype", "")).strip()
        caveats = str(normalized.get("caveats", "")).strip()
        supporting_run_refs = _stringify_list(
            normalized.get("supporting_run_refs", []) or default_refs
        )
        evidence_refs = _stringify_list(normalized.get("evidence_refs", []))
        if not (title_family and likely_archetype and caveats):
            continue
        record = {
            "mapping_key": str(normalized.get("mapping_key", "")).strip()
            or _slugify(f"{title_family}-{likely_archetype}"),
            "title_family": title_family,
            "likely_archetype": likely_archetype,
            "caveats": caveats,
            "supporting_run_refs": supporting_run_refs,
            "evidence_refs": evidence_refs,
        }
        if not _record_has_refs(record):
            continue
        normalized_items.append(record)
    return normalized_items


def sanitize_self_presentation_patterns(
    items: Any,
    *,
    default_supporting_run_refs: list[str] | None = None,
) -> list[dict]:
    if not isinstance(items, list):
        return []
    default_refs = _stringify_list(default_supporting_run_refs or [])
    normalized_items: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_refs(item)
        label = str(normalized.get("label", "")).strip()
        pattern = str(normalized.get("pattern", "")).strip()
        why_false_negative = str(
            normalized.get("why_it_causes_false_negatives", "")
        ).strip()
        supporting_run_refs = _stringify_list(
            normalized.get("supporting_run_refs", []) or default_refs
        )
        evidence_refs = _stringify_list(normalized.get("evidence_refs", []))
        if not (label and pattern and why_false_negative):
            continue
        record = {
            "pattern_key": str(normalized.get("pattern_key", "")).strip()
            or _slugify(label),
            "label": label,
            "pattern": pattern,
            "why_it_causes_false_negatives": why_false_negative,
            "supporting_run_refs": supporting_run_refs,
            "evidence_refs": evidence_refs,
        }
        if not _record_has_refs(record):
            continue
        normalized_items.append(record)
    return normalized_items


def sanitize_false_negative_hypotheses(
    items: Any,
    *,
    default_supporting_run_refs: list[str] | None = None,
) -> list[dict]:
    if not isinstance(items, list):
        return []
    default_refs = _stringify_list(default_supporting_run_refs or [])
    normalized_items: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_refs(item)
        statement = str(normalized.get("statement", "")).strip()
        why_it_matters = str(normalized.get("why_it_matters", "")).strip()
        validation_task = str(normalized.get("validation_task", "")).strip()
        supporting_run_refs = _stringify_list(
            normalized.get("supporting_run_refs", []) or default_refs
        )
        evidence_refs = _stringify_list(normalized.get("evidence_refs", []))
        if not (statement and why_it_matters and validation_task):
            continue
        record = {
            "hypothesis_key": str(normalized.get("hypothesis_key", "")).strip()
            or _slugify(statement),
            "statement": statement,
            "why_it_matters": why_it_matters,
            "validation_task": validation_task,
            "confidence": min(
                1.0,
                max(0.0, float(normalized.get("confidence", 0.5) or 0.5)),
            ),
            "supporting_run_refs": supporting_run_refs,
            "evidence_refs": evidence_refs,
        }
        if not _record_has_refs(record):
            continue
        normalized_items.append(record)
    return normalized_items


def _normalize_refs(record: dict) -> dict:
    normalized = dict(record)
    if "supporting_run_refs" in normalized:
        normalized["supporting_run_refs"] = _stringify_list(
            _require_list(
                "supporting_run_refs",
                normalized.get("supporting_run_refs", []),
            )
        )
    if "evidence_refs" in normalized:
        normalized["evidence_refs"] = _stringify_list(
            _require_list("evidence_refs", normalized.get("evidence_refs", []))
        )
    return normalized


def _coerce_narrative_items(
    section_name: str,
    items: Any,
    *,
    strict: bool,
) -> list[dict]:
    policy = NARRATIVE_SECTION_POLICIES[section_name]
    values = _require_list(section_name, items)
    normalized_items: list[dict] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            if strict:
                raise ValueError(f"{section_name}[{index}] must be a dict")
            continue
        normalized = _normalize_refs(item)
        missing = [key for key in policy.required_keys if key not in normalized]
        if missing:
            if strict:
                raise ValueError(
                    f"{section_name}[{index}] missing keys: {', '.join(missing)}"
                )
            continue
        if policy.provenance_required and not _record_has_refs(normalized):
            if strict:
                raise ValueError(
                    f"{section_name}[{index}] must include supporting_run_refs or evidence_refs"
                )
            continue
        normalized_items.append(normalized)
    return normalized_items


def validate_narrative_items(section_name: str, items: Any) -> list[dict]:
    """Return normalized narrative records or raise on invalid entries."""
    return _coerce_narrative_items(section_name, items, strict=True)


def sanitize_narrative_items(section_name: str, items: Any) -> list[dict]:
    """Return only schema-valid narrative records for a section."""
    return _coerce_narrative_items(section_name, items, strict=False)


def sanitize_market_intel_payload(data: dict | None) -> dict:
    """Best-effort cleanup for legacy canonical artifacts before strict validation."""
    payload = deepcopy(data or {})
    if not isinstance(payload, dict):
        return {}
    for section_name in (
        "lane_intelligence",
        "talent_pool_intelligence",
        "noise_patterns",
        "employer_signal_intelligence",
        "brief_recommendations",
        "open_questions",
    ):
        if section_name in payload:
            payload[section_name] = sanitize_narrative_items(
                section_name,
                payload.get(section_name, []),
            )
    market_thesis = payload.get("market_thesis")
    if isinstance(market_thesis, dict):
        normalized_thesis = dict(market_thesis)
        normalized_thesis["summary"] = public_market_thesis_summary(
            market_thesis.get("summary")
        )
        normalized_thesis["external_context"] = sanitize_narrative_items(
            "market_thesis.external_context",
            market_thesis.get("external_context", []),
        )
        payload["market_thesis"] = normalized_thesis
    return payload


@dataclass
class MarketIdentity:
    market_key: str
    role_title: str
    role_level: str
    geography: str
    channels_seen: list[str]
    brief_ids_seen: list[str]
    brief_versions_seen: list[str]
    # Reopen R-SIM groundwork (S1): a normalized role-family for cross-brief
    # warm-start matching. Optional + defaulted so EVERY pre-R-SIM artifact
    # still loads (from_dict must NOT add this to its required tuple). Populated
    # at write only once a real family SOURCE is decided (Sam's a/b/c fork —
    # see plans/reopen-stage3-spine-inversion.md); empty today, so nothing
    # matches on it yet. The field exists now so the write/read sites have a
    # home to fill without a second schema migration later.
    role_family: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "MarketIdentity":
        required = (
            "market_key",
            "role_title",
            "role_level",
            "geography",
            "channels_seen",
            "brief_ids_seen",
            "brief_versions_seen",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"market_identity missing keys: {', '.join(missing)}")
        return cls(
            market_key=str(data["market_key"]).strip(),
            role_title=str(data["role_title"]).strip(),
            role_level=str(data["role_level"]).strip(),
            geography=str(data["geography"]).strip(),
            channels_seen=_stringify_list(
                _require_list("channels_seen", data["channels_seen"])
            ),
            brief_ids_seen=_stringify_list(
                _require_list("brief_ids_seen", data["brief_ids_seen"])
            ),
            brief_versions_seen=_stringify_list(
                _require_list("brief_versions_seen", data["brief_versions_seen"])
            ),
            # R-SIM S1: optional + defaulted — read via .get so OLD artifacts
            # (written before this field existed) load with role_family=''.
            # Deliberately NOT added to the `required` tuple above.
            role_family=str(data.get("role_family", "")).strip(),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketEvidenceBatch:
    run_ref: str
    source: str
    output_dir: str
    brief_version: str
    generated_at: str
    run_id: int | None = None
    run_dir: str = ""
    state_dir: str = ""
    report: dict | None = None
    report_input: dict | None = None
    search_memory: dict | None = None
    final_judgments: list[dict] = field(default_factory=list)
    runtime_summary: dict = field(default_factory=dict)
    external_sources: list[dict] = field(default_factory=list)
    metrics_summary: dict = field(default_factory=dict)
    research_context: dict | None = None
    research_input_path: str = ""
    context_quality: str = ""
    analysis_provenance: str = ""
    is_complete: bool = True

    def to_run_index_record(self) -> dict:
        record = {
            "run_ref": self.run_ref,
            "source": self.source,
            "output_dir": self.output_dir,
            "brief_version": self.brief_version,
            "generated_at": self.generated_at,
        }
        if self.run_id is not None:
            record["run_id"] = self.run_id
        if self.run_dir:
            record["run_dir"] = self.run_dir
        if self.state_dir:
            record["state_dir"] = self.state_dir
        if self.context_quality:
            record["context_quality"] = self.context_quality
        if self.analysis_provenance:
            record["analysis_provenance"] = self.analysis_provenance
        if self.research_input_path:
            record["research_input_path"] = self.research_input_path
        return record


@dataclass
class SectionGenerationMetadata:
    generation_mode: str
    quality_level: str
    updated_at: str
    notes: list[str] = field(default_factory=list)
    supporting_run_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SectionGenerationMetadata":
        required = ("generation_mode", "quality_level", "updated_at")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"section generation metadata missing keys: {', '.join(missing)}"
            )
        return cls(
            generation_mode=str(data["generation_mode"]).strip(),
            quality_level=str(data["quality_level"]).strip(),
            updated_at=str(data["updated_at"]).strip(),
            notes=_stringify_list(_require_list("notes", data.get("notes", []))),
            supporting_run_refs=_stringify_list(
                _require_list(
                    "supporting_run_refs",
                    data.get("supporting_run_refs", []),
                )
            ),
            evidence_refs=_stringify_list(
                _require_list("evidence_refs", data.get("evidence_refs", []))
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


MARKET_HYPOTHESIS_STATUSES = frozenset({"active", "validated", "rejected", "resolved"})


class MarketHypothesisStatusError(ValueError):
    """Raised on load when a hypothesis carries an untyped status string."""


@dataclass
class MarketHypothesis:
    hypothesis_id: str
    statement: str
    status: str
    confidence: float
    rationale: str
    section_targets: list[str]
    first_seen_at: str
    last_seen_at: str
    supporting_run_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    # P3.4: lifecycle counter — consecutive post-run updates whose planner
    # output did not mention this hypothesis. At 5, the engine retires it to
    # 'resolved' instead of letting the planner clobber the whole set.
    unrefreshed_runs: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "MarketHypothesis":
        required = (
            "hypothesis_id",
            "statement",
            "status",
            "confidence",
            "rationale",
            "section_targets",
            "first_seen_at",
            "last_seen_at",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"market hypothesis missing keys: {', '.join(missing)}"
            )
        normalized = _normalize_refs(data)
        if not _record_has_refs(normalized):
            raise ValueError(
                "market hypothesis must include supporting_run_refs or evidence_refs"
            )
        status = str(data["status"]).strip()
        # P3.4: typed status with validation on load — a free-string status
        # has no transitions and no lifecycle.
        if status not in MARKET_HYPOTHESIS_STATUSES:
            raise MarketHypothesisStatusError(
                f"market hypothesis status must be one of "
                f"{sorted(MARKET_HYPOTHESIS_STATUSES)}, got: {status!r}"
            )
        return cls(
            hypothesis_id=str(data["hypothesis_id"]).strip(),
            statement=str(data["statement"]).strip(),
            status=status,
            confidence=float(data["confidence"]),
            rationale=str(data["rationale"]).strip(),
            section_targets=_stringify_list(
                _require_list("section_targets", data["section_targets"])
            ),
            first_seen_at=str(data["first_seen_at"]).strip(),
            last_seen_at=str(data["last_seen_at"]).strip(),
            supporting_run_refs=_stringify_list(
                _require_list(
                    "supporting_run_refs",
                    normalized.get("supporting_run_refs", []),
                )
            ),
            evidence_refs=_stringify_list(
                _require_list("evidence_refs", normalized.get("evidence_refs", []))
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResearchOpportunity:
    opportunity_id: str
    question: str
    priority: str
    status: str
    reason: str
    supporting_run_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchOpportunity":
        required = ("opportunity_id", "question", "priority", "status", "reason")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"research opportunity missing keys: {', '.join(missing)}"
            )
        normalized = _normalize_refs(data)
        if not _record_has_refs(normalized):
            raise ValueError(
                "research opportunity must include supporting_run_refs or evidence_refs"
            )
        return cls(
            opportunity_id=str(data["opportunity_id"]).strip(),
            question=str(data["question"]).strip(),
            priority=str(data["priority"]).strip(),
            status=str(data["status"]).strip(),
            reason=str(data["reason"]).strip(),
            supporting_run_refs=_stringify_list(
                _require_list(
                    "supporting_run_refs",
                    normalized.get("supporting_run_refs", []),
                )
            ),
            evidence_refs=_stringify_list(
                _require_list("evidence_refs", normalized.get("evidence_refs", []))
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketIntelAdvisory:
    """A live checkpoint advisory: an actuated recommendation (kind + rationale).

    ``confidence`` is legacy-optional (P10): the only producer
    (``market_intelligence.live_advisory._heuristic_new_advisories``) used
    to author fabricated per-kind numbers (0.68-0.75) with no measurement
    behind them, and nothing downstream ever branched on the value — the
    sole reader was the rendered advisory line, which has since dropped it.
    The field stays, defaulted to 0.0, only so pre-P10 persisted
    ``live-advisories.jsonl`` records still round-trip through
    :meth:`from_dict`.
    """

    advisory_id: str
    scope: str
    kind: str
    rationale: str
    created_at: str
    expires_at_checkpoint: int
    checkpoint_key: str
    confidence: float = 0.0
    consumed_by: list[str] = field(default_factory=list)
    supporting_run_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "MarketIntelAdvisory":
        required = (
            "advisory_id",
            "scope",
            "kind",
            "rationale",
            "created_at",
            "expires_at_checkpoint",
            "checkpoint_key",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"market advisory missing keys: {', '.join(missing)}"
            )
        normalized = _normalize_refs(data)
        if not _record_has_refs(normalized):
            raise ValueError(
                "market advisory must include supporting_run_refs or evidence_refs"
            )
        return cls(
            advisory_id=str(data["advisory_id"]).strip(),
            scope=str(data["scope"]).strip(),
            kind=str(data["kind"]).strip(),
            rationale=str(data["rationale"]).strip(),
            created_at=str(data["created_at"]).strip(),
            expires_at_checkpoint=int(data["expires_at_checkpoint"]),
            checkpoint_key=str(data["checkpoint_key"]).strip(),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            consumed_by=_stringify_list(
                _require_list("consumed_by", data.get("consumed_by", []))
            ),
            supporting_run_refs=_stringify_list(
                _require_list(
                    "supporting_run_refs",
                    normalized.get("supporting_run_refs", []),
                )
            ),
            evidence_refs=_stringify_list(
                _require_list("evidence_refs", normalized.get("evidence_refs", []))
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceRegistryEntry:
    source_id: str
    kind: str
    title: str
    url: str
    retrieved_at: str
    used_for: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "SourceRegistryEntry":
        required = ("source_id", "kind", "title", "url", "retrieved_at", "used_for")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"source registry entry missing keys: {', '.join(missing)}"
            )
        return cls(
            source_id=str(data["source_id"]).strip(),
            kind=str(data["kind"]).strip(),
            title=str(data["title"]).strip(),
            url=str(data["url"]).strip(),
            retrieved_at=str(data["retrieved_at"]).strip(),
            used_for=_stringify_list(
                _require_list("used_for", data["used_for"])
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketIntelAgentState:
    schema_version: int
    market_key: str
    updated_at: str
    active_hypotheses: list[MarketHypothesis] = field(default_factory=list)
    resolved_hypotheses: list[MarketHypothesis] = field(default_factory=list)
    open_unknowns: list[dict] = field(default_factory=list)
    research_backlog: list[ResearchOpportunity] = field(default_factory=list)
    source_registry: list[SourceRegistryEntry] = field(default_factory=list)
    confidence_by_claim_area: dict[str, float] = field(default_factory=dict)
    prior_advisories: list[MarketIntelAdvisory] = field(default_factory=list)
    section_generation_metadata: dict[str, SectionGenerationMetadata] = field(
        default_factory=dict
    )

    @classmethod
    def from_dict(cls, data: dict) -> "MarketIntelAgentState":
        if not isinstance(data, dict):
            raise ValueError("agent state must be a dict")
        required = ("schema_version", "market_key", "updated_at")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"agent state missing keys: {', '.join(missing)}")
        open_unknowns = sanitize_narrative_items(
            "open_questions",
            data.get("open_unknowns", []),
        )
        return cls(
            schema_version=int(data["schema_version"]),
            market_key=str(data["market_key"]).strip(),
            updated_at=str(data["updated_at"]).strip(),
            active_hypotheses=[
                MarketHypothesis.from_dict(item)
                for item in _require_list(
                    "active_hypotheses",
                    data.get("active_hypotheses", []),
                )
            ],
            resolved_hypotheses=[
                MarketHypothesis.from_dict(item)
                for item in _require_list(
                    "resolved_hypotheses",
                    data.get("resolved_hypotheses", []),
                )
            ],
            open_unknowns=open_unknowns,
            research_backlog=[
                ResearchOpportunity.from_dict(item)
                for item in _require_list(
                    "research_backlog",
                    data.get("research_backlog", []),
                )
            ],
            source_registry=[
                SourceRegistryEntry.from_dict(item)
                for item in _require_list(
                    "source_registry",
                    data.get("source_registry", []),
                )
            ],
            confidence_by_claim_area={
                str(key): float(value)
                for key, value in _require_dict(
                    "confidence_by_claim_area",
                    data.get("confidence_by_claim_area", {}),
                ).items()
            },
            prior_advisories=[
                MarketIntelAdvisory.from_dict(item)
                for item in _require_list(
                    "prior_advisories",
                    data.get("prior_advisories", []),
                )
            ],
            section_generation_metadata={
                str(section): SectionGenerationMetadata.from_dict(value)
                for section, value in _require_dict(
                    "section_generation_metadata",
                    data.get("section_generation_metadata", {}),
                ).items()
                if isinstance(value, dict)
            },
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "market_key": self.market_key,
            "updated_at": self.updated_at,
            "active_hypotheses": [
                item.to_dict() for item in self.active_hypotheses
            ],
            "resolved_hypotheses": [
                item.to_dict() for item in self.resolved_hypotheses
            ],
            "open_unknowns": self.open_unknowns,
            "research_backlog": [item.to_dict() for item in self.research_backlog],
            "source_registry": [item.to_dict() for item in self.source_registry],
            "confidence_by_claim_area": self.confidence_by_claim_area,
            "prior_advisories": [item.to_dict() for item in self.prior_advisories],
            "section_generation_metadata": {
                key: value.to_dict()
                for key, value in self.section_generation_metadata.items()
            },
        }


@dataclass
class MarketIntelArtifact:
    schema_version: int
    artifact_version: int
    market_identity: MarketIdentity
    freshness: dict
    evidence_index: dict
    aggregate_metrics: dict
    channel_summaries: dict
    lane_intelligence: list[dict]
    talent_pool_intelligence: list[dict]
    noise_patterns: list[dict]
    employer_signal_intelligence: list[dict]
    candidate_signal_summary: dict
    market_thesis: dict
    brief_recommendations: list[dict]
    open_questions: list[dict]
    retrieval_design_summary: dict = field(default_factory=dict)
    section_generation_metadata: dict[str, dict] = field(default_factory=dict)
    delta_since_last_run: dict = field(default_factory=dict)
    planner_diffs: list[dict] = field(default_factory=list)
    # P3.3: expired/consumed planner diffs move here instead of accreting in
    # planner_diffs forever. Kept for audit; never served to strategy.
    archived_planner_diffs: list[dict] = field(default_factory=list)
    # P3.5: decayed narrative entries per section (unsupported for
    # NARRATIVE_DECAY_RUNS updates). Excluded from prompt/context rendering.
    archived_narratives: dict = field(default_factory=dict)
    # Persisted audit trail of the critic's per-claim adjudications (the council-style
    # evidence-led reasoning the critic prompt now requires). Optional + backward-
    # compatible: absent in pre-existing artifacts -> []. Free-form audit dicts, NOT
    # validated as narrative items.
    claim_adjudications: list[dict] = field(default_factory=list)
    # P3.6: the observed facial-calibration comparison (authored
    # expected_yes_rate_low/high vs. the actual per-run yes-rate) plus the
    # cross-run consecutive-out-of-band counter that gates the drift-warning
    # and the calibration-drift Gate-2 hunk. Optional + backward-compatible:
    # absent in pre-existing artifacts -> {}.
    facial_calibration_observed: dict = field(default_factory=dict)
    # P4.3.2: stage names (e.g. "planner", "synthesis") that fail-soft'd
    # during this update — the same failures already narrated in
    # ``stage_errors`` on the technical markdown, now also persisted on the
    # artifact itself so a consumer doesn't have to re-derive degradation
    # from prose. Optional + backward-compatible: absent in pre-existing
    # artifacts -> [].
    stages_degraded: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "MarketIntelArtifact":
        if not isinstance(data, dict):
            raise ValueError("market intel artifact must be a dict")
        required = (
            "schema_version",
            "artifact_version",
            "market_identity",
            "freshness",
            "evidence_index",
            "aggregate_metrics",
            "channel_summaries",
            "lane_intelligence",
            "talent_pool_intelligence",
            "noise_patterns",
            "employer_signal_intelligence",
            "candidate_signal_summary",
            "market_thesis",
            "brief_recommendations",
            "open_questions",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"market intel artifact missing keys: {', '.join(missing)}"
            )

        artifact = cls(
            schema_version=int(data["schema_version"]),
            artifact_version=int(data["artifact_version"]),
            market_identity=MarketIdentity.from_dict(
                _require_dict("market_identity", data["market_identity"])
            ),
            freshness=_require_dict("freshness", data["freshness"]),
            evidence_index=_require_dict("evidence_index", data["evidence_index"]),
            aggregate_metrics=_require_dict(
                "aggregate_metrics", data["aggregate_metrics"]
            ),
            channel_summaries=_require_dict(
                "channel_summaries", data["channel_summaries"]
            ),
            lane_intelligence=validate_narrative_items(
                "lane_intelligence",
                data["lane_intelligence"],
            ),
            talent_pool_intelligence=validate_narrative_items(
                "talent_pool_intelligence",
                data["talent_pool_intelligence"],
            ),
            noise_patterns=validate_narrative_items(
                "noise_patterns",
                data["noise_patterns"],
            ),
            employer_signal_intelligence=validate_narrative_items(
                "employer_signal_intelligence",
                data["employer_signal_intelligence"],
            ),
            candidate_signal_summary=_require_dict(
                "candidate_signal_summary",
                data["candidate_signal_summary"],
            ),
            market_thesis=_require_dict("market_thesis", data["market_thesis"]),
            brief_recommendations=validate_narrative_items(
                "brief_recommendations",
                data["brief_recommendations"],
            ),
            open_questions=validate_narrative_items(
                "open_questions",
                data["open_questions"],
            ),
            retrieval_design_summary=_require_dict(
                "retrieval_design_summary",
                data.get("retrieval_design_summary", {}),
            ),
            section_generation_metadata={
                str(section): SectionGenerationMetadata.from_dict(value).to_dict()
                for section, value in _require_dict(
                    "section_generation_metadata",
                    data.get("section_generation_metadata", {}),
                ).items()
                if isinstance(value, dict)
            },
            delta_since_last_run=_require_dict(
                "delta_since_last_run",
                data.get("delta_since_last_run", {}),
            ),
            planner_diffs=[
                item
                for item in data.get("planner_diffs", [])
                if isinstance(item, dict)
            ],
            archived_planner_diffs=[
                item
                for item in data.get("archived_planner_diffs", [])
                if isinstance(item, dict)
            ],
            archived_narratives=_require_dict(
                "archived_narratives",
                data.get("archived_narratives", {}),
            ),
            claim_adjudications=[
                item
                for item in data.get("claim_adjudications", [])
                if isinstance(item, dict)
            ],
            facial_calibration_observed=_require_dict(
                "facial_calibration_observed",
                data.get("facial_calibration_observed", {}),
            ),
            stages_degraded=[
                str(item)
                for item in data.get("stages_degraded", [])
                if isinstance(item, str) and item.strip()
            ],
        )
        artifact._validate()
        return artifact

    def _validate(self) -> None:
        evidence_runs = _require_list(
            "evidence_index.runs",
            self.evidence_index.get("runs", []),
        )
        external_sources = _require_list(
            "evidence_index.external_sources",
            self.evidence_index.get("external_sources", []),
        )
        for record in evidence_runs:
            if not isinstance(record, dict):
                raise ValueError("evidence_index.runs entries must be dicts")
            for key in (
                "run_ref",
                "source",
                "output_dir",
                "brief_version",
                "generated_at",
            ):
                if key not in record:
                    raise ValueError(f"evidence_index.runs entry missing key: {key}")
            for key in (
                "context_quality",
                "analysis_provenance",
                "research_input_path",
                "run_id",
                "run_dir",
                "state_dir",
            ):
                if key not in record:
                    continue
                if key == "run_id":
                    if not isinstance(record[key], int):
                        raise ValueError(
                            "evidence_index.runs entry key run_id must be an integer when present"
                        )
                elif not isinstance(record[key], str):
                    raise ValueError(
                        f"evidence_index.runs entry key {key} must be a string when present"
                    )
        for record in external_sources:
            if not isinstance(record, dict):
                raise ValueError(
                    "evidence_index.external_sources entries must be dicts"
                )
            for key in ("source_id", "kind", "title", "url", "retrieved_at", "used_for"):
                if key not in record:
                    raise ValueError(f"external source entry missing key: {key}")

        for section_name in (
            "lane_intelligence",
            "talent_pool_intelligence",
            "noise_patterns",
            "employer_signal_intelligence",
            "brief_recommendations",
            "open_questions",
        ):
            validate_narrative_items(section_name, getattr(self, section_name))

        market_thesis = _require_dict("market_thesis", self.market_thesis)
        for key in ("summary", "supply_assessment", "competition_assessment", "external_context"):
            if key not in market_thesis:
                raise ValueError(f"market_thesis missing key: {key}")
        validate_narrative_items(
            "market_thesis.external_context",
            market_thesis.get("external_context", []),
        )
        if self.section_generation_metadata:
            for section, value in self.section_generation_metadata.items():
                SectionGenerationMetadata.from_dict(
                    _require_dict(
                        f"section_generation_metadata.{section}",
                        value,
                    )
                )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "artifact_version": self.artifact_version,
            "market_identity": self.market_identity.to_dict(),
            "freshness": self.freshness,
            "evidence_index": self.evidence_index,
            "aggregate_metrics": self.aggregate_metrics,
            "channel_summaries": self.channel_summaries,
            "lane_intelligence": self.lane_intelligence,
            "talent_pool_intelligence": self.talent_pool_intelligence,
            "noise_patterns": self.noise_patterns,
            "employer_signal_intelligence": self.employer_signal_intelligence,
            "candidate_signal_summary": self.candidate_signal_summary,
            "market_thesis": self.market_thesis,
            "brief_recommendations": self.brief_recommendations,
            "open_questions": self.open_questions,
            "retrieval_design_summary": self.retrieval_design_summary,
            "section_generation_metadata": self.section_generation_metadata,
            "delta_since_last_run": self.delta_since_last_run,
            "planner_diffs": self.planner_diffs,
            "archived_planner_diffs": self.archived_planner_diffs,
            "archived_narratives": self.archived_narratives,
            "claim_adjudications": self.claim_adjudications,
            "facial_calibration_observed": self.facial_calibration_observed,
            "stages_degraded": self.stages_degraded,
        }


def _render_named_section(
    title: str,
    items: list[dict],
    heading_key: str,
    detail_keys: list[str],
    metadata: dict | None = None,
) -> list[str]:
    lines = [f"## {title}"]
    if metadata:
        mode = str(metadata.get("generation_mode", "")).strip() or "unknown"
        quality = str(metadata.get("quality_level", "")).strip() or "unknown"
        lines.append(f"- Generation mode: {mode}")
        lines.append(f"- Quality level: {quality}")
        notes = ", ".join(_stringify_list(metadata.get("notes", [])))
        if notes:
            lines.append(f"- Notes: {notes}")
    if not items:
        lines.append("- None")
        lines.append("")
        return lines
    for item in items:
        heading = str(item.get(heading_key, "Unnamed")).strip() or "Unnamed"
        lines.append(f"- **{heading}**")
        for key in detail_keys:
            value = item.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                rendered = ", ".join(_stringify_list(value))
            elif isinstance(value, dict):
                rendered = ", ".join(
                    f"{subkey}={subvalue}"
                    for subkey, subvalue in value.items()
                    if subvalue not in (None, "", [], {})
                )
            else:
                rendered = str(value).strip()
            if rendered:
                lines.append(f"  {key.replace('_', ' ')}: {rendered}")
    lines.append("")
    return lines


def _bullet_lines(values: list[str], *, empty: str = "None") -> list[str]:
    normalized = _stringify_list(values)
    if not normalized:
        return [f"- {empty}"]
    return [f"- {value}" for value in normalized]


def _top_lane_summary(artifact: MarketIntelArtifact, limit: int = 3) -> list[str]:
    items: list[str] = []
    for lane in artifact.lane_intelligence[:limit]:
        metrics = lane.get("metrics", {}) if isinstance(lane, dict) else {}
        lane_name = str(lane.get("lane_key", "")).replace("_", " ").strip()
        if not lane_name:
            continue
        saves = metrics.get("saves")
        candidate_volume = metrics.get("candidates_seen")
        detail = lane_name
        if saves not in (None, "") and candidate_volume not in (None, ""):
            detail = f"{detail} ({saves} saves from {candidate_volume} candidates)"
        items.append(detail)
    return items


def _executive_summary_points(artifact: MarketIntelArtifact) -> list[str]:
    points: list[str] = []
    thesis_summary = public_market_thesis_summary(artifact.market_thesis.get("summary"))
    if thesis_summary:
        points.append(thesis_summary)
    if artifact.delta_since_last_run:
        points.extend(
            _stringify_list(artifact.delta_since_last_run.get("became_more_true", []))[:2]
        )
    if artifact.brief_recommendations:
        points.append(
            f"Next-run priority: {str(artifact.brief_recommendations[0].get('proposal', '')).strip()}"
        )
    if artifact.open_questions:
        points.append(
            f"Key uncertainty: {str(artifact.open_questions[0].get('question', '')).strip()}"
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for point in points:
        text = str(point or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
        if len(deduped) >= 5:
            break
    return deduped


def _market_learning_points(artifact: MarketIntelArtifact) -> list[str]:
    points: list[str] = []
    for item in artifact.talent_pool_intelligence[:3]:
        label = str(item.get("label", "")).strip()
        summary = str(item.get("evidence_summary", "")).strip()
        if label and summary:
            points.append(f"{label}: {summary}")
    for item in artifact.employer_signal_intelligence[:3]:
        label = str(item.get("label", "")).strip()
        summary = str(item.get("evidence_summary", "")).strip()
        if label and summary:
            points.append(f"{label}: {summary}")
    for item in artifact.market_thesis.get("external_context", [])[:4]:
        claim = str(item.get("claim", "")).strip()
        if claim:
            points.append(claim)
    deduped: list[str] = []
    seen: set[str] = set()
    for point in points:
        key = point.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped[:8]


def _next_run_changes(artifact: MarketIntelArtifact) -> list[str]:
    points = [
        str(item.get("proposal", "")).strip()
        for item in artifact.brief_recommendations
        if str(item.get("proposal", "")).strip()
    ]
    points.extend(_stringify_list(artifact.delta_since_last_run.get("next_run_changes", [])))
    deduped: list[str] = []
    seen: set[str] = set()
    for point in points:
        key = point.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped[:8]


def _risk_and_unknowns(artifact: MarketIntelArtifact) -> list[str]:
    points = [
        str(item.get("question", "")).strip()
        for item in artifact.open_questions
        if str(item.get("question", "")).strip()
    ]
    points.extend(_stringify_list(artifact.delta_since_last_run.get("still_uncertain", [])))
    deduped: list[str] = []
    seen: set[str] = set()
    for point in points:
        key = point.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped[:8]


def _brief_evidence_notes(artifact: MarketIntelArtifact) -> list[str]:
    metrics = artifact.aggregate_metrics
    notes = [
        f"Observed across {metrics.get('run_count', 0)} run(s) with {metrics.get('saved_count', 0)} total saves and a {metrics.get('save_rate', 0):.1%} aggregate save rate.",
    ]
    top_lanes = _top_lane_summary(artifact)
    if top_lanes:
        notes.append(f"Top observed lanes: {'; '.join(top_lanes)}.")
    freshness = artifact.freshness
    if str(freshness.get("external_research_through", "")).strip():
        notes.append(
            f"External research is current through {freshness.get('external_research_through')}."
        )
    else:
        notes.append(
            f"This version relies only on internal sourcing evidence through {freshness.get('internal_data_through', 'n/a')}."
        )
    return notes


def render_market_intel_markdown(artifact: MarketIntelArtifact) -> str:
    identity = artifact.market_identity
    lines = [f"# Market Intelligence Memo: {identity.role_title}", ""]
    lines.append(
        f"Scope: {identity.geography or 'Unspecified geography'} | {identity.role_level or 'Unspecified level'}"
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.extend(_bullet_lines(_executive_summary_points(artifact)))
    lines.append("")

    lines.append("## What We Learned About the Market")
    lines.extend(_bullet_lines(_market_learning_points(artifact)))
    lines.append("")

    lines.append("## What Changes for the Next Sourcing Run")
    lines.extend(_bullet_lines(_next_run_changes(artifact)))
    lines.append("")

    lines.append("## Risks and Unknowns")
    lines.extend(_bullet_lines(_risk_and_unknowns(artifact)))
    lines.append("")

    lines.append("## Brief Evidence Notes")
    lines.extend(_bullet_lines(_brief_evidence_notes(artifact)))
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_market_intel_technical_markdown(
    artifact: MarketIntelArtifact,
    *,
    planner_summary: str = "",
    critique_summary: str = "",
    edge_case_research_reasoning: str = "",
    edge_case_focus: list[dict] | None = None,
    inferred_research_questions: list[dict] | None = None,
    market_findings: list[dict] | None = None,
    sourcing_implications: list[dict] | None = None,
    edge_case_inferred_research_questions: list[dict] | None = None,
    edge_case_submarkets: list[dict] | None = None,
    title_to_archetype_mapping: list[dict] | None = None,
    self_presentation_patterns: list[dict] | None = None,
    false_negative_hypotheses: list[dict] | None = None,
    edge_case_sourcing_implications: list[dict] | None = None,
    edge_case_open_questions: list[dict] | None = None,
    stage_errors: list[str] | None = None,
) -> str:
    identity = artifact.market_identity
    freshness = artifact.freshness
    metrics = artifact.aggregate_metrics
    section_metadata = artifact.section_generation_metadata or {}

    lines = [f"# Market Intelligence Technical Appendix: {identity.role_title}", ""]
    lines.append("## Identity")
    lines.append(f"- Market key: `{identity.market_key}`")
    lines.append(f"- Geography: {identity.geography or 'Unspecified'}")
    lines.append(f"- Role level: {identity.role_level or 'Unspecified'}")
    lines.append(f"- Channels seen: {', '.join(identity.channels_seen) or 'None'}")
    lines.append("")

    lines.append("## Pass Diagnostics")
    lines.append(f"- Planner summary: {planner_summary or 'None'}")
    lines.append(f"- Critique summary: {critique_summary or 'None'}")
    lines.append(
        f"- Edge-case reasoning: {edge_case_research_reasoning or 'Not triggered / no additional reasoning.'}"
    )
    if edge_case_focus:
        lines.append(
            f"- Edge-case focus areas: {', '.join(_stringify_list([item.get('focus') for item in edge_case_focus if isinstance(item, dict)]))}"
        )
    if stage_errors:
        lines.append(f"- Stage errors: {', '.join(_stringify_list(stage_errors))}")
    else:
        lines.append("- Stage errors: None")
    lines.append("")

    lines.append("## Why Edge-Case Research Triggered")
    lines.extend(
        _bullet_lines(
            [edge_case_research_reasoning]
            + _stringify_list(
                [
                    item.get("focus")
                    for item in (edge_case_focus or [])
                    if isinstance(item, dict) and _stringify_list([item.get("focus")])
                ]
            ),
            empty="Edge-case research did not trigger in this pass.",
        )
    )
    lines.append("")

    lines.append("## Freshness")
    lines.append(f"- Artifact updated: {freshness.get('artifact_updated_at', 'n/a')}")
    lines.append(
        f"- Internal data through: {freshness.get('internal_data_through', 'n/a')}"
    )
    lines.append(
        f"- External research through: {freshness.get('external_research_through', 'n/a')}"
    )
    lines.append("")

    lines.extend(
        [
            "## Aggregate Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Run count | {metrics.get('run_count', 0)} |",
            f"| Saved count | {metrics.get('saved_count', 0)} |",
            f"| Rejected count | {metrics.get('rejected_count', 0)} |",
            f"| Facial YES rate | {metrics.get('facial_yes_rate', 0):.1%} |",
            f"| Save rate | {metrics.get('save_rate', 0):.1%} |",
            "",
        ]
    )

    lines.extend(
        _render_named_section(
            "Lane Inventory",
            artifact.lane_intelligence,
            "lane_key",
            [
                "status",
                "domain_lane",
                "novelty_bucket",
                "dominant_anchors",
                "why_it_works",
                "recommended_action",
                "supporting_run_refs",
            ],
            section_metadata.get("lane_intelligence"),
        )
    )
    lines.extend(
        _render_named_section(
            "Talent Pool Intelligence",
            artifact.talent_pool_intelligence,
            "label",
            [
                "status",
                "signal_strength",
                "evidence_summary",
                "recommended_search_terms",
                "supporting_run_refs",
                "evidence_refs",
            ],
            section_metadata.get("talent_pool_intelligence"),
        )
    )
    lines.extend(
        _render_named_section(
            "Employer Signal Intelligence",
            artifact.employer_signal_intelligence,
            "label",
            [
                "status",
                "supporting_employers",
                "evidence_summary",
                "supporting_run_refs",
                "evidence_refs",
            ],
            section_metadata.get("employer_signal_intelligence"),
        )
    )
    lines.extend(
        _render_named_section(
            "Brief Recommendations",
            artifact.brief_recommendations,
            "recommendation_id",
            [
                "target_field",
                "proposal",
                "reason",
                "supporting_run_refs",
                "evidence_refs",
            ],
            section_metadata.get("brief_recommendations"),
        )
    )
    lines.extend(
        _render_named_section(
            "Open Questions",
            artifact.open_questions,
            "question",
            ["priority", "next_step", "supporting_run_refs", "evidence_refs"],
            section_metadata.get("open_questions"),
        )
    )

    lines.extend(
        _render_named_section(
            "Inferred Research Questions",
            inferred_research_questions or [],
            "question",
            [
                "priority",
                "status",
                "why_it_matters",
                "sourcing_trigger",
                "supporting_run_refs",
                "evidence_refs",
            ],
        )
    )
    lines.extend(
        _render_named_section(
            "External Market Findings",
            market_findings or [],
            "label",
            [
                "kind",
                "summary",
                "why_it_matters",
                "supporting_run_refs",
                "evidence_refs",
            ],
        )
    )
    lines.extend(
        _render_named_section(
            "Raw Sourcing Implications",
            sourcing_implications or [],
            "implication_id",
            [
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
        )
    )
    lines.extend(
        _render_named_section(
            "Edge-Case Inferred Research Questions",
            edge_case_inferred_research_questions or [],
            "question",
            [
                "priority",
                "status",
                "why_it_matters",
                "sourcing_trigger",
                "supporting_run_refs",
                "evidence_refs",
            ],
        )
    )
    lines.extend(
        _render_named_section(
            "Edge-Case Submarkets",
            edge_case_submarkets or [],
            "label",
            [
                "summary",
                "why_it_is_easy_to_miss",
                "confidence",
                "supporting_run_refs",
                "evidence_refs",
            ],
        )
    )
    lines.extend(
        _render_named_section(
            "Title To Archetype Mapping",
            title_to_archetype_mapping or [],
            "title_family",
            [
                "likely_archetype",
                "caveats",
                "supporting_run_refs",
                "evidence_refs",
            ],
        )
    )
    lines.extend(
        _render_named_section(
            "Self-Presentation Patterns",
            self_presentation_patterns or [],
            "label",
            [
                "pattern",
                "why_it_causes_false_negatives",
                "supporting_run_refs",
                "evidence_refs",
            ],
        )
    )
    lines.extend(
        _render_named_section(
            "False-Negative Hypotheses",
            false_negative_hypotheses or [],
            "statement",
            [
                "why_it_matters",
                "validation_task",
                "confidence",
                "supporting_run_refs",
                "evidence_refs",
            ],
        )
    )
    lines.extend(
        _render_named_section(
            "Edge-Case Sourcing Implications",
            edge_case_sourcing_implications or [],
            "implication_id",
            [
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
        )
    )
    lines.extend(
        _render_named_section(
            "Edge-Case Open Questions",
            edge_case_open_questions or [],
            "question",
            ["priority", "next_step", "supporting_run_refs", "evidence_refs"],
        )
    )
    return "\n".join(lines).rstrip() + "\n"
